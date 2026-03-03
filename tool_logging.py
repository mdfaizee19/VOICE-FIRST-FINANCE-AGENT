"""
Tool Logging & Latency Optimization — Financial Simulation System

Production-grade module providing:
  1. Structured logging with field redaction and correlation IDs
  2. Deterministic latency benchmarking with sub-100ms enforcement
  3. Precomputation of derived financial values
  4. Thread-safe, stateless design (no global mutable state)

Uses structlog for JSON-structured logging.
No print statements.  No external observability services.  No LLM logic.
"""

from __future__ import annotations

import copy
import functools
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, TypeVar

import structlog

from financial_schemas import FinancialState


# ═══════════════════════════════════════════════════════════════════════
# Structlog Configuration (module-level, immutable after init)
# ═══════════════════════════════════════════════════════════════════════

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(0),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

_logger = structlog.get_logger("financial_tools")


# ═══════════════════════════════════════════════════════════════════════
# Section 1 — Redaction
# ═══════════════════════════════════════════════════════════════════════

_REDACTED = "***REDACTED***"

_SENSITIVE_KEYS = frozenset({
    "monthly_income",
    "emergency_fund",
    "credit_balance",
    "fixed_expenses",
    "discretionary_expenses",
})


def redact_payload(payload: Any) -> Any:
    """Deep-redact sensitive fields from a payload for logging.

    Rules:
    - Keys in ``_SENSITIVE_KEYS`` → replaced with ``***REDACTED***``
    - Any numeric value > 1000 in a dict → replaced with ``***REDACTED***``
    - Recursively processes nested dicts and lists
    - Does NOT mutate the original payload; returns a new structure

    Thread-safe: operates on a deep copy.
    """
    if isinstance(payload, dict):
        redacted: Dict[str, Any] = {}
        for key, value in payload.items():
            if key in _SENSITIVE_KEYS:
                redacted[key] = _REDACTED
            elif isinstance(value, (int, float)) and not isinstance(value, bool) and value > 1000:
                redacted[key] = _REDACTED
            elif isinstance(value, (dict, list)):
                redacted[key] = redact_payload(value)
            else:
                redacted[key] = value
        return redacted
    elif isinstance(payload, list):
        return [redact_payload(item) for item in payload]
    else:
        return payload


# ═══════════════════════════════════════════════════════════════════════
# Section 1 — Logging Wrapper Decorator
# ═══════════════════════════════════════════════════════════════════════

F = TypeVar("F", bound=Callable[..., Any])


def tool_logging(tool_name: str) -> Callable[[F], F]:
    """Decorator that wraps a tool function with structured logging.

    For every invocation:
    - Generates a request-local ``correlation_id`` (UUID4)
    - Logs redacted input/output payloads
    - Measures ``execution_time_ms`` via ``time.perf_counter()``
    - Returns the original result enriched with ``correlation_id``
    - Catches and logs any unhandled exceptions

    Thread-safe: correlation_id is local to each call frame.
    No global mutable state.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # ── Request-local state ───────────────────────────────
            correlation_id = str(uuid.uuid4())
            log = _logger.bind(correlation_id=correlation_id, tool=tool_name)

            # ── Build input snapshot for logging ──────────────────
            input_payload = copy.deepcopy(kwargs) if kwargs else {}
            redacted_input = redact_payload(input_payload)

            log.info(
                "tool_invocation_started",
                input=redacted_input,
            )

            # ── Execute with timing ───────────────────────────────
            t_start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                t_end = time.perf_counter()
                execution_ms = round((t_end - t_start) * 1000, 2)
                log.error(
                    "tool_invocation_failed",
                    status="error",
                    error_code="UNHANDLED_EXCEPTION",
                    error_message=str(exc),
                    execution_time_ms=execution_ms,
                    input=redacted_input,
                )
                raise
            t_end = time.perf_counter()
            execution_ms = round((t_end - t_start) * 1000, 2)

            # ── Determine status from result ──────────────────────
            status = "success"
            error_code = None
            if isinstance(result, dict):
                status = result.get("status", "success")
                error_code = result.get("error_code")

            # ── Redact output for logging ─────────────────────────
            redacted_output = redact_payload(result) if isinstance(result, dict) else result

            log.info(
                "tool_invocation_completed",
                status=status,
                error_code=error_code,
                execution_time_ms=execution_ms,
                output=redacted_output,
            )

            # ── Enrich result with correlation_id ─────────────────
            if isinstance(result, dict):
                result["correlation_id"] = correlation_id
                result["execution_time_ms"] = execution_ms

            return result

        return wrapper  # type: ignore[return-value]

    return decorator


# ═══════════════════════════════════════════════════════════════════════
# Section 2 — Latency Benchmarking
# ═══════════════════════════════════════════════════════════════════════

def benchmark_tool_execution(
    tool_function: Callable[..., Any],
    sample_input: Dict[str, Any],
    iterations: int = 100,
    max_avg_ms: float = 100.0,
) -> Dict[str, Any]:
    """Run a tool function ``iterations`` times and compute latency metrics.

    Parameters
    ----------
    tool_function : callable
        The tool wrapper function to benchmark (not the LangChain tool).
    sample_input : dict
        Keyword arguments passed to ``tool_function(**sample_input)``.
    iterations : int
        Number of repetitions (default 100).
    max_avg_ms : float
        Maximum acceptable average latency.  If exceeded, ``RuntimeError``
        is raised with a diagnostic breakdown.

    Returns
    -------
    dict
        ``{"iterations", "avg_ms", "p95_ms", "max_ms", "min_ms",
          "all_ms", "status"}``

    Raises
    ------
    RuntimeError
        If ``avg_ms >= max_avg_ms``.
    """
    latencies: List[float] = []

    for _ in range(iterations):
        t0 = time.perf_counter()
        tool_function(**sample_input)
        t1 = time.perf_counter()
        latencies.append(round((t1 - t0) * 1000, 2))

    latencies_sorted = sorted(latencies)
    avg_ms = round(sum(latencies) / len(latencies), 2)
    p95_idx = int(len(latencies_sorted) * 0.95) - 1
    p95_ms = latencies_sorted[max(p95_idx, 0)]
    max_ms = latencies_sorted[-1]
    min_ms = latencies_sorted[0]

    metrics = {
        "iterations": iterations,
        "avg_ms": avg_ms,
        "p95_ms": p95_ms,
        "max_ms": max_ms,
        "min_ms": min_ms,
        "status": "pass" if avg_ms < max_avg_ms else "fail",
    }

    if avg_ms >= max_avg_ms:
        raise RuntimeError(
            f"Performance target violated: avg latency {avg_ms}ms >= {max_avg_ms}ms. "
            f"Diagnostic: p95={p95_ms}ms, max={max_ms}ms, min={min_ms}ms, "
            f"iterations={iterations}"
        )

    return metrics


# ═══════════════════════════════════════════════════════════════════════
# Section 3 — Precomputation of Derived Financial Values
# ═══════════════════════════════════════════════════════════════════════

def compute_derived_values(financial_state: FinancialState) -> Dict[str, float]:
    """Precompute frequently-used derived values once per invocation.

    Avoids redundant arithmetic inside simulation loops.

    Returns
    -------
    dict
        - ``monthly_rate``: credit_apr / 12
        - ``total_expenses``: fixed_expenses + discretionary_expenses
        - ``expense_ratio``: total_expenses / monthly_income (0.0 if no income)
        - ``monthly_interest``: credit_balance × monthly_rate
        - ``disposable_income``: monthly_income − total_expenses
        - ``commitment_total``: sum of all commitment amounts
        - ``available_after_obligations``: disposable − monthly_interest − commitments
    """
    monthly_rate = round(financial_state.credit_apr / 12.0, 6)
    total_expenses = round(
        financial_state.fixed_expenses + financial_state.discretionary_expenses, 2
    )
    expense_ratio = (
        round(total_expenses / financial_state.monthly_income, 4)
        if financial_state.monthly_income > 0
        else 0.0
    )
    monthly_interest = round(financial_state.credit_balance * monthly_rate, 2)
    disposable_income = round(financial_state.monthly_income - total_expenses, 2)
    commitment_total = round(
        sum(c.amount for c in financial_state.commitments), 2
    )
    available = round(disposable_income - monthly_interest - commitment_total, 2)

    return {
        "monthly_rate": monthly_rate,
        "total_expenses": total_expenses,
        "expense_ratio": expense_ratio,
        "monthly_interest": monthly_interest,
        "disposable_income": disposable_income,
        "commitment_total": commitment_total,
        "available_after_obligations": available,
    }


# ═══════════════════════════════════════════════════════════════════════
# Wrapped Tool Functions (logging-enabled versions)
# ═══════════════════════════════════════════════════════════════════════

# Import the raw tool wrappers from langchain_tools
from langchain_tools import (
    _run_credit_simulation,
    _run_stress_test,
    _run_subscription_simulation,
)

logged_subscription_simulation = tool_logging("simulate_subscription")(
    _run_subscription_simulation
)

logged_credit_simulation = tool_logging("simulate_credit_strategy")(
    _run_credit_simulation
)

logged_stress_test = tool_logging("run_stress_test")(
    _run_stress_test
)


# ═══════════════════════════════════════════════════════════════════════
# Example Usage
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json
    import sys

    log = _logger.bind(section="example_usage")

    # ── 1. Logged tool invocation ─────────────────────────────────────

    log.info("running_logged_subscription_simulation")

    result = logged_subscription_simulation(
        monthly_income=5000.0,
        fixed_expenses=2000.0,
        discretionary_expenses=800.0,
        emergency_fund=10000.0,
        credit_balance=3000.0,
        credit_apr=0.24,
        commitments=[],
        subscriptions=[
            {"name": "streaming", "monthly_cost": 15.0},
            {"name": "cloud-storage", "monthly_cost": 12.0},
        ],
    )
    log.info("subscription_result",
             status=result.get("status"),
             correlation_id=result.get("correlation_id"),
             execution_time_ms=result.get("execution_time_ms"))

    # ── 2. Precomputed derived values ─────────────────────────────────

    from financial_schemas import Commitment

    state = FinancialState(
        monthly_income=5000.0,
        fixed_expenses=2000.0,
        discretionary_expenses=800.0,
        emergency_fund=10000.0,
        credit_balance=3000.0,
        credit_apr=0.24,
        commitments=[Commitment(id="ins", amount=300.0, due_month=3)],
    )

    derived = compute_derived_values(state)
    log.info("derived_values", **derived)

    # ── 3. Benchmark: subscription tool ───────────────────────────────

    log.info("starting_benchmark", tool="simulate_subscription")

    sample = {
        "monthly_income": 5000.0,
        "fixed_expenses": 2000.0,
        "discretionary_expenses": 800.0,
        "emergency_fund": 10000.0,
        "credit_balance": 3000.0,
        "credit_apr": 0.24,
        "commitments": [],
        "subscriptions": [{"name": "basic", "monthly_cost": 10.0}],
    }

    metrics = benchmark_tool_execution(
        _run_subscription_simulation, sample, iterations=100
    )
    log.info("benchmark_complete", **metrics)

    # ── 4. Benchmark: stress test tool ────────────────────────────────

    log.info("starting_benchmark", tool="run_stress_test")

    stress_sample = {
        "monthly_income": 5000.0,
        "fixed_expenses": 2000.0,
        "discretionary_expenses": 800.0,
        "emergency_fund": 10000.0,
        "credit_balance": 3000.0,
        "credit_apr": 0.24,
    }

    stress_metrics = benchmark_tool_execution(
        _run_stress_test, stress_sample, iterations=100
    )
    log.info("benchmark_complete", **stress_metrics)

    # ── 5. Redaction demo ─────────────────────────────────────────────

    log.info("redaction_demo",
             original={"monthly_income": 5000.0, "credit_balance": 3000.0,
                        "credit_apr": 0.24, "projected_balance": 1500.0},
             redacted=redact_payload(
                 {"monthly_income": 5000.0, "credit_balance": 3000.0,
                  "credit_apr": 0.24, "projected_balance": 1500.0}))

    # ── 6. Intentional latency violation (commented) ──────────────────
    #
    # import time as _time
    #
    # def slow_tool(**kwargs):
    #     _time.sleep(0.15)  # 150ms sleep
    #     return {"status": "success", "data": {}}
    #
    # try:
    #     benchmark_tool_execution(slow_tool, {}, iterations=10, max_avg_ms=100.0)
    # except RuntimeError as exc:
    #     log.error("latency_violation", error=str(exc))
    #     # Output:
    #     # RuntimeError: Performance target violated: avg latency 150.xx ms >= 100.0ms.
    #     # Diagnostic: p95=..., max=..., min=..., iterations=10
