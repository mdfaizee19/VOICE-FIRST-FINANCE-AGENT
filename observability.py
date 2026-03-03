"""
Observability Layer — Financial Simulation System (Phase 7)

Production-grade structured logging, Prometheus-compatible metrics,
error rate monitoring, stability score tracking, and tool error
frequency analysis.

Guarantees:
  - Zero business logic modification
  - No sensitive data leakage (no raw financial state in logs)
  - Deterministic metric updates
  - Graceful degradation (observability failures never break simulation)
  - Async-safe implementation
  - Real-time stability score monitoring via rolling window

Integration points:
  - Controller layer (FastAPI middleware)
  - Service layer (simulation service wrapper)
  - Tool execution wrapper

No domain engine modifications.  No risk engine modifications.
No Pydantic model modifications.  No simulation logic access.
"""

from __future__ import annotations

import collections
import json
import logging
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, List, Optional
from uuid import UUID

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


# ═══════════════════════════════════════════════════════════════════════
#  Section 1 — Structured JSON Logging
# ═══════════════════════════════════════════════════════════════════════

class _JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Merge structured fields attached via extra={}
        for key in (
            "correlation_id", "user_id", "event_type", "tool_name",
            "status", "latency_ms", "error_code", "detail",
        ):
            val = getattr(record, key, None)
            if val is not None:
                entry[key] = str(val) if isinstance(val, UUID) else val
        return json.dumps(entry, default=str)


def _setup_logger(name: str = "financial_engine") -> logging.Logger:
    """Create (or retrieve) a structured JSON logger.

    All output goes to stderr so it doesn't interfere with
    application stdout (metrics, API responses, etc.).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
    return logger


_logger = _setup_logger()


def log_event(
    event_type: str,
    correlation_id: str,
    *,
    user_id: Optional[UUID] = None,
    tool_name: Optional[str] = None,
    status: str = "success",
    latency_ms: float = 0.0,
    error_code: Optional[str] = None,
    detail: Optional[str] = None,
    level: int = logging.INFO,
) -> None:
    """Emit a structured log event.

    Never raises — logs errors to stderr if logging itself fails.

    Parameters
    ----------
    event_type
        Category: "simulation_request", "tool_invocation", "error", etc.
    correlation_id
        Request-scoped trace ID.
    user_id
        Optional user identifier (UUID, not sensitive).
    tool_name
        Tool that was invoked (if applicable).
    status
        "success" or "error".
    latency_ms
        Request duration in milliseconds.
    error_code
        Structured error code (if status == "error").
    detail
        Optional non-sensitive detail string.
    level
        Python logging level (default INFO).
    """
    try:
        _logger.log(
            level,
            event_type,
            extra={
                "event_type": event_type,
                "correlation_id": correlation_id,
                "user_id": user_id,
                "tool_name": tool_name,
                "status": status,
                "latency_ms": round(latency_ms, 2),
                "error_code": error_code,
                "detail": detail,
            },
        )
    except Exception:
        # Observability must never break simulation
        pass


# ═══════════════════════════════════════════════════════════════════════
#  Section 2 — Prometheus Metrics
# ═══════════════════════════════════════════════════════════════════════

# Dedicated registry (avoids polluting the default global registry)
REGISTRY = CollectorRegistry()

# --- Counters ---

simulation_requests_total = Counter(
    "simulation_requests_total",
    "Total number of simulation requests received",
    registry=REGISTRY,
)

simulation_success_total = Counter(
    "simulation_success_total",
    "Total number of successful simulations",
    registry=REGISTRY,
)

simulation_failure_total = Counter(
    "simulation_failure_total",
    "Total number of failed simulations",
    registry=REGISTRY,
)

tool_invocations_total = Counter(
    "tool_invocations_total",
    "Total tool invocations by tool name",
    ["tool_name"],
    registry=REGISTRY,
)

tool_errors_total = Counter(
    "tool_errors_total",
    "Total tool errors by tool name",
    ["tool_name"],
    registry=REGISTRY,
)

# --- Histograms ---

simulation_latency_seconds = Histogram(
    "simulation_latency_seconds",
    "Simulation request latency in seconds",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=REGISTRY,
)

stability_score_distribution = Histogram(
    "stability_score_distribution",
    "Distribution of stability scores across simulations",
    buckets=(0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
    registry=REGISTRY,
)

# --- Gauges ---

stability_score_average = Gauge(
    "stability_score_average",
    "Rolling average stability score (last 500 simulations)",
    registry=REGISTRY,
)


def get_metrics_output() -> bytes:
    """Generate Prometheus-compatible metrics text."""
    return generate_latest(REGISTRY)


# ═══════════════════════════════════════════════════════════════════════
#  Section 3 — Error Rate Monitoring
# ═══════════════════════════════════════════════════════════════════════

def compute_error_rate() -> float:
    """Compute current error rate.

    Returns simulation_failure_total / simulation_requests_total,
    or 0.0 if no requests have been made.
    """
    total = simulation_requests_total._value.get()
    if total == 0:
        return 0.0
    failures = simulation_failure_total._value.get()
    return round(failures / total, 4)


def check_error_rate(correlation_id: str = "system") -> Dict[str, Any]:
    """Check error rate and emit warnings if thresholds exceeded.

    Thresholds:
      - > 5% → WARNING
      - > 10% → CRITICAL
    """
    rate = compute_error_rate()
    total = simulation_requests_total._value.get()
    failures = simulation_failure_total._value.get()

    result: Dict[str, Any] = {
        "error_rate": rate,
        "total_requests": total,
        "total_failures": failures,
        "alert_level": "normal",
    }

    try:
        if rate > 0.10:
            result["alert_level"] = "critical"
            log_event(
                "error_rate_critical",
                correlation_id,
                status="error",
                error_code="ERROR_RATE_CRITICAL",
                detail=f"Error rate {rate:.1%} exceeds 10% threshold "
                       f"({failures}/{total})",
                level=logging.CRITICAL,
            )
        elif rate > 0.05:
            result["alert_level"] = "warning"
            log_event(
                "error_rate_warning",
                correlation_id,
                status="error",
                error_code="ERROR_RATE_WARNING",
                detail=f"Error rate {rate:.1%} exceeds 5% threshold "
                       f"({failures}/{total})",
                level=logging.WARNING,
            )
    except Exception:
        pass

    return result


# ═══════════════════════════════════════════════════════════════════════
#  Section 4 — Stability Score Rolling Tracker
# ═══════════════════════════════════════════════════════════════════════

_ROLLING_WINDOW_SIZE = 500
_score_lock = threading.Lock()
_score_window: Deque[float] = collections.deque(maxlen=_ROLLING_WINDOW_SIZE)


def record_stability_score(score: float) -> None:
    """Record a stability score into the rolling window.

    Thread-safe.  Updates the Prometheus gauge and histogram.
    Never raises.
    """
    try:
        with _score_lock:
            _score_window.append(score)
            avg = sum(_score_window) / len(_score_window)

        stability_score_average.set(round(avg, 2))
        stability_score_distribution.observe(score)
    except Exception:
        pass


def get_stability_stats() -> Dict[str, Any]:
    """Return current rolling stability score statistics."""
    with _score_lock:
        if not _score_window:
            return {
                "count": 0,
                "average": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
        scores = list(_score_window)

    return {
        "count": len(scores),
        "average": round(sum(scores) / len(scores), 2),
        "min": min(scores),
        "max": max(scores),
    }


# ═══════════════════════════════════════════════════════════════════════
#  Section 5 — Tool Error Frequency Tracker
# ═══════════════════════════════════════════════════════════════════════

_tool_stats_lock = threading.Lock()
_tool_invocation_counts: Dict[str, int] = {}
_tool_failure_counts: Dict[str, int] = {}


def record_tool_invocation(tool_name: str, success: bool) -> None:
    """Record a tool invocation (success or failure).

    Thread-safe.  Updates Prometheus counters and internal trackers.
    Never raises.
    """
    try:
        tool_invocations_total.labels(tool_name=tool_name).inc()
        with _tool_stats_lock:
            _tool_invocation_counts[tool_name] = (
                _tool_invocation_counts.get(tool_name, 0) + 1
            )
            if not success:
                tool_errors_total.labels(tool_name=tool_name).inc()
                _tool_failure_counts[tool_name] = (
                    _tool_failure_counts.get(tool_name, 0) + 1
                )
    except Exception:
        pass


def get_tool_error_rates() -> Dict[str, Dict[str, Any]]:
    """Get per-tool invocation/failure counts and failure ratios."""
    with _tool_stats_lock:
        tools = dict(_tool_invocation_counts)
        failures = dict(_tool_failure_counts)

    result: Dict[str, Dict[str, Any]] = {}
    for name, total in tools.items():
        fails = failures.get(name, 0)
        ratio = round(fails / total, 4) if total > 0 else 0.0
        result[name] = {
            "invocations": total,
            "failures": fails,
            "failure_ratio": ratio,
        }
    return result


def check_tool_error_rates(correlation_id: str = "system") -> Dict[str, Any]:
    """Check per-tool failure ratios and emit warnings if > 10%.

    Returns a dict of tool stats with alert flags.
    """
    rates = get_tool_error_rates()
    alerts: List[str] = []

    for name, stats in rates.items():
        if stats["failure_ratio"] > 0.10:
            alerts.append(name)
            try:
                log_event(
                    "tool_error_rate_warning",
                    correlation_id,
                    tool_name=name,
                    status="error",
                    error_code="TOOL_ERROR_RATE_HIGH",
                    detail=f"Tool '{name}' failure ratio {stats['failure_ratio']:.1%} "
                           f"exceeds 10% ({stats['failures']}/{stats['invocations']})",
                    level=logging.WARNING,
                )
            except Exception:
                pass

    return {"tools": rates, "alerts": alerts}


# ═══════════════════════════════════════════════════════════════════════
#  Section 6 — Simulation Service Integration
# ═══════════════════════════════════════════════════════════════════════

def observe_simulation(
    func: Callable[..., Dict[str, Any]],
) -> Callable[..., Dict[str, Any]]:
    """Decorator that wraps a simulation service function with observability.

    Tracks:
      - Request count
      - Success/failure counts
      - Latency
      - Stability score (if present in result)
      - Tool invocation metrics

    Never modifies the underlying function's behavior or return value.
    """
    def wrapper(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        correlation_id = kwargs.get("correlation_id", "unknown")
        tool_name = kwargs.get("tool_name")
        user_id = kwargs.get("user_id")

        simulation_requests_total.inc()
        start = time.perf_counter()

        try:
            result = func(*args, **kwargs)
        except Exception as exc:
            # Record failure metrics
            elapsed_ms = (time.perf_counter() - start) * 1000
            simulation_failure_total.inc()
            simulation_latency_seconds.observe(elapsed_ms / 1000)

            if tool_name:
                record_tool_invocation(tool_name, success=False)

            log_event(
                "simulation_error",
                correlation_id,
                user_id=user_id,
                tool_name=tool_name,
                status="error",
                latency_ms=elapsed_ms,
                error_code=type(exc).__name__,
                detail=str(exc)[:200],
            )
            raise

        elapsed_ms = (time.perf_counter() - start) * 1000
        simulation_latency_seconds.observe(elapsed_ms / 1000)
        status = result.get("status", "unknown")

        if status == "success":
            simulation_success_total.inc()
            if tool_name:
                record_tool_invocation(tool_name, success=True)

            # Extract stability score from result data
            data = result.get("data", {})
            for r in data.get("results", []):
                if isinstance(r, dict) and "stability_score" in r:
                    record_stability_score(r["stability_score"])

            log_event(
                "simulation_success",
                correlation_id,
                user_id=user_id,
                tool_name=tool_name,
                status="success",
                latency_ms=elapsed_ms,
            )
        else:
            simulation_failure_total.inc()
            if tool_name:
                record_tool_invocation(tool_name, success=False)

            log_event(
                "simulation_failure",
                correlation_id,
                user_id=user_id,
                tool_name=tool_name,
                status="error",
                latency_ms=elapsed_ms,
                error_code=result.get("error_code", "UNKNOWN"),
                detail=result.get("message", "")[:200],
            )

        return result

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


# ═══════════════════════════════════════════════════════════════════════
#  Section 7 — FastAPI Integration
# ═══════════════════════════════════════════════════════════════════════

try:
    from fastapi import FastAPI, Request, Response

    def setup_observability(app: FastAPI) -> None:
        """Register observability middleware and /metrics endpoint.

        Usage::

            app = FastAPI()
            setup_observability(app)
        """
        @app.get("/metrics")
        async def metrics_endpoint() -> Response:
            """Prometheus-compatible metrics endpoint."""
            return Response(
                content=get_metrics_output(),
                media_type="text/plain; version=0.0.4; charset=utf-8",
            )

        @app.middleware("http")
        async def observability_middleware(
            request: Request, call_next: Callable,
        ) -> Response:
            """Track request-level metrics without touching domain logic."""
            if request.url.path == "/metrics":
                return await call_next(request)

            start = time.perf_counter()
            response = await call_next(request)
            elapsed_ms = (time.perf_counter() - start) * 1000

            # Log at request level (no sensitive data)
            correlation_id = request.headers.get(
                "x-correlation-id", "no-correlation-id"
            )
            try:
                log_event(
                    "http_request",
                    correlation_id,
                    status="success" if response.status_code < 400 else "error",
                    latency_ms=elapsed_ms,
                    detail=f"{request.method} {request.url.path} → {response.status_code}",
                )
            except Exception:
                pass

            return response

    _FASTAPI_AVAILABLE = True

except ImportError:
    _FASTAPI_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════
#  Example Usage
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uuid as _uuid

    def _show(label: str, obj: Any) -> None:
        print(f"\n{'═' * 64}")
        print(f"  {label}")
        print(f"{'═' * 64}")
        if isinstance(obj, (dict, list)):
            print(json.dumps(obj, indent=2, default=str))
        elif isinstance(obj, bytes):
            print(obj.decode("utf-8"))
        else:
            print(str(obj))

    # ══════════════════════════════════════════════════════════════════
    # 1. STRUCTURED LOG — SUCCESS
    # ══════════════════════════════════════════════════════════════════

    cid = str(_uuid.uuid4())
    print("\n── Example structured logs (to stderr) ──")
    log_event(
        "simulation_success",
        cid,
        user_id=_uuid.uuid4(),
        tool_name="subscription_simulation",
        status="success",
        latency_ms=42.5,
    )

    # ══════════════════════════════════════════════════════════════════
    # 2. STRUCTURED LOG — ERROR
    # ══════════════════════════════════════════════════════════════════

    log_event(
        "simulation_failure",
        str(_uuid.uuid4()),
        tool_name="credit_strategy_simulation",
        status="error",
        latency_ms=12.3,
        error_code="INVALID_TOOL_INPUT",
        detail="Missing required field: strategy_type",
    )

    # ══════════════════════════════════════════════════════════════════
    # 3. SIMULATE 20 REQUESTS (mixed success/failure)
    # ══════════════════════════════════════════════════════════════════

    # 16 successes with varying scores
    scores = [
        92.5, 85.3, 78.1, 65.4, 55.0, 42.3, 38.7, 25.1,
        88.0, 91.2, 74.5, 60.8, 48.2, 35.6, 82.9, 70.0,
    ]
    for score in scores:
        simulation_requests_total.inc()
        simulation_success_total.inc()
        simulation_latency_seconds.observe(0.045)
        record_stability_score(score)
        record_tool_invocation("subscription_simulation", success=True)

    # 4 failures
    for _ in range(4):
        simulation_requests_total.inc()
        simulation_failure_total.inc()
        simulation_latency_seconds.observe(0.008)
        record_tool_invocation("credit_strategy_simulation", success=False)

    # ══════════════════════════════════════════════════════════════════
    # 4. ERROR RATE CHECK
    # ══════════════════════════════════════════════════════════════════

    error_report = check_error_rate(correlation_id="health-check")
    _show("ERROR RATE REPORT", error_report)

    # ══════════════════════════════════════════════════════════════════
    # 5. STABILITY SCORE STATS
    # ══════════════════════════════════════════════════════════════════

    stats = get_stability_stats()
    _show("STABILITY SCORE STATS (rolling window)", stats)

    # ══════════════════════════════════════════════════════════════════
    # 6. TOOL ERROR RATES
    # ══════════════════════════════════════════════════════════════════

    tool_report = check_tool_error_rates(correlation_id="health-check")
    _show("TOOL ERROR RATES", tool_report)

    # ══════════════════════════════════════════════════════════════════
    # 7. PROMETHEUS METRICS OUTPUT
    # ══════════════════════════════════════════════════════════════════

    metrics_text = get_metrics_output()
    _show("PROMETHEUS METRICS (/metrics)", metrics_text)

    # ══════════════════════════════════════════════════════════════════
    # 8. OBSERVE_SIMULATION DECORATOR DEMO
    # ══════════════════════════════════════════════════════════════════

    @observe_simulation
    def mock_simulation(**kwargs: Any) -> Dict[str, Any]:
        """Simulated service function."""
        return {
            "status": "success",
            "data": {
                "results": [
                    {"stability_score": 77.5, "risk_flags": []},
                ],
            },
        }

    demo_result = mock_simulation(
        correlation_id=str(_uuid.uuid4()),
        tool_name="subscription_simulation",
    )
    _show("DECORATOR DEMO RESULT", demo_result)

    # Final stats after decorator demo
    final_stats = get_stability_stats()
    _show("FINAL STABILITY STATS", final_stats)

    print(f"\n{'═' * 64}")
    print(f"  FastAPI integration available: {_FASTAPI_AVAILABLE}")
    print(f"{'═' * 64}")
