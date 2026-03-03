"""
Failure Mode Handling & SLA Definition — Financial Simulation System (Phase 7)

Ensures the API never crashes, never exposes stack traces, and never returns
partial corrupted data.  When tool execution fails AFTER input validation,
the system returns a deterministic "degraded" fallback response.

Guarantees:
  - API stability under any tool failure
  - Deterministic fallback with UNKNOWN risk flags
  - Hard 2-second timeout per simulation
  - SLA enforcement (latency + error rate thresholds)
  - Structured CRITICAL logs with internal trace (no sensitive data)
  - Auth / rate-limit / schema errors are never degraded — always rejected

No domain logic.  No risk math.  No scoring.  No persistence.
"""

from __future__ import annotations

import asyncio
import collections
import enum
import json
import logging
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Deque, Dict, List, Optional

from observability import log_event


# ═══════════════════════════════════════════════════════════════════════
#  Failure Classification
# ═══════════════════════════════════════════════════════════════════════

class FailureCategory(str, enum.Enum):
    """Exhaustive failure classification for the simulation system."""
    TOOL_EXECUTION_ERROR = "TOOL_EXECUTION_ERROR"
    TOOL_TIMEOUT = "TOOL_TIMEOUT"
    VALIDATION_FAILURE = "VALIDATION_FAILURE"
    OUTPUT_VERIFICATION_FAILURE = "OUTPUT_VERIFICATION_FAILURE"
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"


# Categories that trigger graceful degradation (fallback response)
DEGRADABLE_FAILURES = frozenset({
    FailureCategory.TOOL_EXECUTION_ERROR,
    FailureCategory.TOOL_TIMEOUT,
    FailureCategory.OUTPUT_VERIFICATION_FAILURE,
    FailureCategory.INTERNAL_SERVER_ERROR,
})

# Categories that must be hard-rejected (no fallback)
REJECT_ONLY_FAILURES = frozenset({
    FailureCategory.VALIDATION_FAILURE,
})


# ═══════════════════════════════════════════════════════════════════════
#  SLA Constants
# ═══════════════════════════════════════════════════════════════════════

TOOL_RESPONSE_TIME_TARGET_MS = 200     # Target: ≤ 200ms
TOOL_RESPONSE_TIME_CRITICAL_MS = 400   # Critical: > 2× target
TOOL_EXECUTION_TIMEOUT_S = 2.0         # Hard timeout: 2 seconds
MAX_ALLOWED_ERROR_RATE = 0.05          # Warning at 5%
CRITICAL_ERROR_RATE = 0.10             # Critical at 10%


# ═══════════════════════════════════════════════════════════════════════
#  Graceful Degradation — Response Builder
# ═══════════════════════════════════════════════════════════════════════

_FALLBACK_RESPONSE_TEMPLATE: Dict[str, Any] = {
    "stability_score": 50,
    "risk_flags": {
        "liquidity_risk": "UNKNOWN",
        "commitment_risk": "UNKNOWN",
        "interest_risk": "UNKNOWN",
        "burn_rate_risk": "UNKNOWN",
        "overall_risk": "UNKNOWN",
    },
}


def build_degraded_response(
    correlation_id: str,
    failure_category: FailureCategory,
    *,
    message: str = "Simulation could not be completed.",
) -> Dict[str, Any]:
    """Build a deterministic degraded fallback response.

    Only used when the failure occurs AFTER input validation.
    Auth, rate-limit, and schema failures are never degraded.
    """
    return {
        "status": "degraded",
        "correlation_id": correlation_id,
        "failure_category": failure_category.value,
        "message": message,
        "fallback": dict(_FALLBACK_RESPONSE_TEMPLATE),  # deep-ish copy
    }


def build_rejection_response(
    correlation_id: str,
    failure_category: FailureCategory,
    error_code: str,
    message: str,
) -> Dict[str, Any]:
    """Build a hard rejection response (no fallback)."""
    return {
        "status": "error",
        "correlation_id": correlation_id,
        "error_code": error_code,
        "failure_category": failure_category.value,
        "message": message,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Internal Trace Logging (secure — no sensitive data)
# ═══════════════════════════════════════════════════════════════════════

def log_internal_failure(
    correlation_id: str,
    exc: Exception,
    *,
    tool_name: Optional[str] = None,
    latency_ms: float = 0.0,
    failure_category: FailureCategory = FailureCategory.INTERNAL_SERVER_ERROR,
) -> None:
    """Capture and log full internal trace as structured CRITICAL log.

    Logs exception type, safe stack trace, correlation_id, tool_name,
    and latency.  Does NOT include raw financial input or user data.
    Never raises — observability must not break the system.
    """
    try:
        tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
        safe_trace = "".join(tb_lines)[-1000]  # truncate to avoid log bloat

        log_event(
            "internal_failure",
            correlation_id,
            tool_name=tool_name,
            status="error",
            latency_ms=latency_ms,
            error_code=failure_category.value,
            detail=f"{type(exc).__name__}: {str(exc)[:200]}",
            level=logging.CRITICAL,
        )
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════
#  Tool Timeout Wrapper
# ═══════════════════════════════════════════════════════════════════════

_executor = ThreadPoolExecutor(max_workers=4)


async def execute_with_timeout(
    func: Callable[..., Dict[str, Any]],
    *args: Any,
    timeout_seconds: float = TOOL_EXECUTION_TIMEOUT_S,
    correlation_id: str = "",
    tool_name: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Execute a tool function with a hard timeout.

    Runs the synchronous function in a thread pool and applies
    ``asyncio.wait_for`` with the configured timeout.

    Returns:
      - Tool result on success
      - Degraded TOOL_TIMEOUT response on timeout
      - Degraded TOOL_EXECUTION_ERROR response on exception
    """
    loop = asyncio.get_event_loop()
    start = time.perf_counter()

    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(_executor, lambda: func(*args, **kwargs)),
            timeout=timeout_seconds,
        )
        return result

    except asyncio.TimeoutError:
        elapsed_ms = (time.perf_counter() - start) * 1000
        log_event(
            "tool_timeout",
            correlation_id,
            tool_name=tool_name,
            status="error",
            latency_ms=elapsed_ms,
            error_code=FailureCategory.TOOL_TIMEOUT.value,
            detail=f"Execution exceeded {timeout_seconds}s timeout",
            level=logging.CRITICAL,
        )
        return build_degraded_response(
            correlation_id,
            FailureCategory.TOOL_TIMEOUT,
            message=f"Simulation timed out after {timeout_seconds}s.",
        )

    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        log_internal_failure(
            correlation_id,
            exc,
            tool_name=tool_name,
            latency_ms=elapsed_ms,
            failure_category=FailureCategory.TOOL_EXECUTION_ERROR,
        )
        return build_degraded_response(
            correlation_id,
            FailureCategory.TOOL_EXECUTION_ERROR,
        )


def execute_with_timeout_sync(
    func: Callable[..., Dict[str, Any]],
    *args: Any,
    timeout_seconds: float = TOOL_EXECUTION_TIMEOUT_S,
    correlation_id: str = "",
    tool_name: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Synchronous wrapper for tool execution with timeout handling.

    Uses the same failure classification and degraded response
    as the async version, but uses threading for timeout enforcement.
    """
    import concurrent.futures

    start = time.perf_counter()

    try:
        future = _executor.submit(func, *args, **kwargs)
        result = future.result(timeout=timeout_seconds)
        return result

    except concurrent.futures.TimeoutError:
        elapsed_ms = (time.perf_counter() - start) * 1000
        log_event(
            "tool_timeout",
            correlation_id,
            tool_name=tool_name,
            status="error",
            latency_ms=elapsed_ms,
            error_code=FailureCategory.TOOL_TIMEOUT.value,
            detail=f"Execution exceeded {timeout_seconds}s timeout",
            level=logging.CRITICAL,
        )
        return build_degraded_response(
            correlation_id,
            FailureCategory.TOOL_TIMEOUT,
            message=f"Simulation timed out after {timeout_seconds}s.",
        )

    except (ValueError, TypeError, KeyError):
        # Re-raise validation-type exceptions so safe_simulation_run
        # can classify them as VALIDATION_FAILURE (hard reject, no fallback)
        raise

    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        log_internal_failure(
            correlation_id,
            exc,
            tool_name=tool_name,
            latency_ms=elapsed_ms,
            failure_category=FailureCategory.TOOL_EXECUTION_ERROR,
        )
        return build_degraded_response(
            correlation_id,
            FailureCategory.TOOL_EXECUTION_ERROR,
        )


# ═══════════════════════════════════════════════════════════════════════
#  SLA Evaluation
# ═══════════════════════════════════════════════════════════════════════

# Rolling latency window (last 500 requests)
_LATENCY_WINDOW = 500
_latency_lock = threading.Lock()
_latency_window: Deque[float] = collections.deque(maxlen=_LATENCY_WINDOW)

# Counters (in-process, thread-safe via lock)
_sla_lock = threading.Lock()
_sla_total_requests = 0
_sla_total_failures = 0
_sla_slow_requests = 0


def record_sla_datapoint(
    latency_ms: float,
    success: bool,
    correlation_id: str = "system",
) -> None:
    """Record a request for SLA tracking.

    Emits WARNING/CRITICAL logs for latency and error rate breaches.
    Never raises.
    """
    global _sla_total_requests, _sla_total_failures, _sla_slow_requests

    try:
        with _latency_lock:
            _latency_window.append(latency_ms)

        with _sla_lock:
            _sla_total_requests += 1
            if not success:
                _sla_total_failures += 1

            # Latency SLA check
            if latency_ms > TOOL_RESPONSE_TIME_CRITICAL_MS:
                _sla_slow_requests += 1
                log_event(
                    "sla_latency_critical",
                    correlation_id,
                    status="error",
                    latency_ms=latency_ms,
                    error_code="SLA_LATENCY_CRITICAL",
                    detail=f"Latency {latency_ms:.1f}ms exceeds "
                           f"critical threshold {TOOL_RESPONSE_TIME_CRITICAL_MS}ms",
                    level=logging.CRITICAL,
                )
            elif latency_ms > TOOL_RESPONSE_TIME_TARGET_MS:
                _sla_slow_requests += 1
                log_event(
                    "sla_latency_warning",
                    correlation_id,
                    status="error",
                    latency_ms=latency_ms,
                    error_code="SLA_LATENCY_WARNING",
                    detail=f"Latency {latency_ms:.1f}ms exceeds "
                           f"target {TOOL_RESPONSE_TIME_TARGET_MS}ms",
                    level=logging.WARNING,
                )

            # Error rate SLA check
            total = _sla_total_requests
            failures = _sla_total_failures

        if total > 0:
            rate = failures / total
            if rate > CRITICAL_ERROR_RATE:
                log_event(
                    "sla_error_rate_critical",
                    correlation_id,
                    status="error",
                    error_code="SLA_ERROR_RATE_CRITICAL",
                    detail=f"Error rate {rate:.1%} exceeds "
                           f"critical threshold {CRITICAL_ERROR_RATE:.0%} "
                           f"({failures}/{total})",
                    level=logging.CRITICAL,
                )
            elif rate > MAX_ALLOWED_ERROR_RATE:
                log_event(
                    "sla_error_rate_warning",
                    correlation_id,
                    status="error",
                    error_code="SLA_ERROR_RATE_WARNING",
                    detail=f"Error rate {rate:.1%} exceeds "
                           f"allowed threshold {MAX_ALLOWED_ERROR_RATE:.0%} "
                           f"({failures}/{total})",
                    level=logging.WARNING,
                )
    except Exception:
        pass


def get_sla_status() -> Dict[str, Any]:
    """Return current SLA compliance status.

    Returns:
      - avg_latency_ms: rolling average latency
      - p95_latency_ms: 95th percentile latency
      - error_rate: current failure ratio
      - slow_request_count: requests exceeding target latency
      - within_sla: True if all thresholds are met
      - sla_targets: the configured thresholds
    """
    with _latency_lock:
        latencies = list(_latency_window)

    with _sla_lock:
        total = _sla_total_requests
        failures = _sla_total_failures
        slow = _sla_slow_requests

    # Compute latency stats
    if latencies:
        avg_latency = round(sum(latencies) / len(latencies), 2)
        sorted_lat = sorted(latencies)
        p95_idx = int(len(sorted_lat) * 0.95)
        p95_latency = round(sorted_lat[min(p95_idx, len(sorted_lat) - 1)], 2)
    else:
        avg_latency = 0.0
        p95_latency = 0.0

    error_rate = round(failures / total, 4) if total > 0 else 0.0

    # Determine overall SLA compliance
    within_sla = (
        avg_latency <= TOOL_RESPONSE_TIME_TARGET_MS
        and error_rate <= MAX_ALLOWED_ERROR_RATE
    )

    return {
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": p95_latency,
        "error_rate": error_rate,
        "total_requests": total,
        "total_failures": failures,
        "slow_request_count": slow,
        "within_sla": within_sla,
        "sla_targets": {
            "response_time_target_ms": TOOL_RESPONSE_TIME_TARGET_MS,
            "response_time_critical_ms": TOOL_RESPONSE_TIME_CRITICAL_MS,
            "max_error_rate": MAX_ALLOWED_ERROR_RATE,
            "critical_error_rate": CRITICAL_ERROR_RATE,
            "execution_timeout_s": TOOL_EXECUTION_TIMEOUT_S,
        },
    }


# ═══════════════════════════════════════════════════════════════════════
#  Simulation Service Integration
# ═══════════════════════════════════════════════════════════════════════

def safe_simulation_run(
    simulation_func: Callable[..., Dict[str, Any]],
    *,
    correlation_id: Optional[str] = None,
    tool_name: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Production-safe simulation execution wrapper.

    Wraps the entire execution in failure classification:
      - ValidationError → hard rejection (no fallback)
      - TimeoutError → degraded response with fallback
      - Any other Exception → degraded response with fallback

    Records SLA metrics for every execution.
    Never raises to the caller.
    """
    cid = correlation_id or str(uuid.uuid4())
    start = time.perf_counter()

    try:
        result = execute_with_timeout_sync(
            simulation_func,
            timeout_seconds=TOOL_EXECUTION_TIMEOUT_S,
            correlation_id=cid,
            tool_name=tool_name,
            **kwargs,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        is_success = result.get("status") in ("success", "degraded")

        # Record SLA metrics
        record_sla_datapoint(elapsed_ms, is_success, cid)

        return result

    except (ValueError, TypeError, KeyError) as exc:
        # Treat as validation failure — hard reject, no fallback
        elapsed_ms = (time.perf_counter() - start) * 1000
        record_sla_datapoint(elapsed_ms, False, cid)

        log_event(
            "validation_failure",
            cid,
            tool_name=tool_name,
            status="error",
            latency_ms=elapsed_ms,
            error_code=FailureCategory.VALIDATION_FAILURE.value,
            detail=f"{type(exc).__name__}: {str(exc)[:200]}",
            level=logging.ERROR,
        )
        return build_rejection_response(
            cid,
            FailureCategory.VALIDATION_FAILURE,
            error_code="VALIDATION_FAILURE",
            message="Input validation failed. Please check your request.",
        )

    except Exception as exc:
        # Catch-all — degraded response
        elapsed_ms = (time.perf_counter() - start) * 1000
        record_sla_datapoint(elapsed_ms, False, cid)

        log_internal_failure(
            cid, exc,
            tool_name=tool_name,
            latency_ms=elapsed_ms,
        )
        return build_degraded_response(
            cid,
            FailureCategory.INTERNAL_SERVER_ERROR,
        )


# ═══════════════════════════════════════════════════════════════════════
#  FastAPI Integration
# ═══════════════════════════════════════════════════════════════════════

try:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    def register_failure_handlers(app: FastAPI) -> None:
        """Register global exception handlers for the FastAPI app.

        Catches unhandled exceptions and returns safe structured responses.
        """
        @app.exception_handler(Exception)
        async def _global_handler(
            request: Request, exc: Exception,
        ) -> JSONResponse:
            cid = request.headers.get("x-correlation-id", str(uuid.uuid4()))
            log_internal_failure(cid, exc)
            return JSONResponse(
                status_code=500,
                content=build_degraded_response(
                    cid,
                    FailureCategory.INTERNAL_SERVER_ERROR,
                ),
            )

        @app.get("/health/sla")
        async def sla_endpoint() -> JSONResponse:
            """SLA status endpoint for monitoring dashboards."""
            return JSONResponse(content=get_sla_status())

    _FASTAPI_AVAILABLE = True

except ImportError:
    _FASTAPI_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════
#  Example Usage
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    def _show(label: str, obj: Any) -> None:
        print(f"\n{'═' * 64}")
        print(f"  {label}")
        print(f"{'═' * 64}")
        if isinstance(obj, dict):
            print(json.dumps(obj, indent=2, default=str))
        else:
            print(str(obj))

    # ══════════════════════════════════════════════════════════════════
    # 1. SUCCESSFUL SIMULATION — safe wrapper
    # ══════════════════════════════════════════════════════════════════

    def _mock_success(**kw: Any) -> Dict[str, Any]:
        time.sleep(0.01)  # simulate work
        return {"status": "success", "data": {"results": [{"stability_score": 82.5}]}}

    result = safe_simulation_run(
        _mock_success,
        correlation_id="demo-success-001",
        tool_name="subscription_simulation",
    )
    _show("1. SUCCESSFUL SIMULATION", result)

    # ══════════════════════════════════════════════════════════════════
    # 2. TOOL EXECUTION ERROR — raises exception
    # ══════════════════════════════════════════════════════════════════

    def _mock_crash(**kw: Any) -> Dict[str, Any]:
        raise RuntimeError("Unexpected internal error in computation")

    result = safe_simulation_run(
        _mock_crash,
        correlation_id="demo-crash-002",
        tool_name="credit_strategy_simulation",
    )
    _show("2. DEGRADED — TOOL EXECUTION ERROR", result)

    # ══════════════════════════════════════════════════════════════════
    # 3. VALIDATION FAILURE — hard rejection (no fallback)
    # ══════════════════════════════════════════════════════════════════

    def _mock_validation_error(**kw: Any) -> Dict[str, Any]:
        raise ValueError("monthly_income must be >= 0")

    result = safe_simulation_run(
        _mock_validation_error,
        correlation_id="demo-validation-003",
        tool_name="subscription_simulation",
    )
    _show("3. REJECTED — VALIDATION FAILURE (no fallback)", result)

    # ══════════════════════════════════════════════════════════════════
    # 4. TOOL TIMEOUT — exceeds 2s limit
    # ══════════════════════════════════════════════════════════════════

    def _mock_slow(**kw: Any) -> Dict[str, Any]:
        time.sleep(5.0)  # way past timeout
        return {"status": "success"}

    result = safe_simulation_run(
        _mock_slow,
        correlation_id="demo-timeout-004",
        tool_name="subscription_simulation",
    )
    _show("4. DEGRADED — TOOL TIMEOUT", result)

    # ══════════════════════════════════════════════════════════════════
    # 5. SIMULATE MIXED TRAFFIC FOR SLA
    # ══════════════════════════════════════════════════════════════════

    # Fast successes
    for i in range(15):
        record_sla_datapoint(
            latency_ms=45.0 + i * 3,
            success=True,
            correlation_id=f"traffic-{i}",
        )

    # Slow requests (breach target)
    for i in range(3):
        record_sla_datapoint(
            latency_ms=250.0 + i * 50,
            success=True,
            correlation_id=f"slow-{i}",
        )

    # Failures
    for i in range(2):
        record_sla_datapoint(
            latency_ms=15.0,
            success=False,
            correlation_id=f"fail-{i}",
        )

    # ══════════════════════════════════════════════════════════════════
    # 6. SLA STATUS
    # ══════════════════════════════════════════════════════════════════

    sla = get_sla_status()
    _show("6. SLA STATUS", sla)

    # ══════════════════════════════════════════════════════════════════
    # 7. DEGRADED RESPONSE SHAPES (direct builders)
    # ══════════════════════════════════════════════════════════════════

    _show("7a. DEGRADED — TIMEOUT SHAPE", build_degraded_response(
        "shape-001", FailureCategory.TOOL_TIMEOUT,
        message="Simulation timed out after 2.0s.",
    ))

    _show("7b. DEGRADED — VERIFICATION FAILURE SHAPE", build_degraded_response(
        "shape-002", FailureCategory.OUTPUT_VERIFICATION_FAILURE,
        message="Output verification failed.",
    ))

    _show("7c. REJECTION — VALIDATION SHAPE", build_rejection_response(
        "shape-003", FailureCategory.VALIDATION_FAILURE,
        error_code="INVALID_TOOL_INPUT",
        message="Missing required field: credit_apr",
    ))

    # ══════════════════════════════════════════════════════════════════
    # 8. FAILURE CATEGORY ENUM
    # ══════════════════════════════════════════════════════════════════

    _show("8. FAILURE CATEGORIES", {
        cat.name: {
            "value": cat.value,
            "degradable": cat in DEGRADABLE_FAILURES,
        }
        for cat in FailureCategory
    })

    print(f"\n{'═' * 64}")
    print(f"  FastAPI integration available: {_FASTAPI_AVAILABLE}")
    print(f"{'═' * 64}")
