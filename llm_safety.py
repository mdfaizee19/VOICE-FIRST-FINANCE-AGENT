"""
LLM Orchestration Safety Layer — Financial Simulation System

Production-grade pre-execution validation, post-execution verification,
and controlled auto-repair for tool calls originating from LLM agents.

Components
----------
1. ``validate_tool_call``   — reject hallucinated / malformed inputs
2. ``verify_tool_output``   — enforce output schema bounds + numeric sanity
3. ``safe_tool_execution``  — full orchestration wrapper (validate → execute → verify)

No LLM calls.  No simulation logic.  No database.  No logging layer.
"""

from __future__ import annotations

import math
import uuid
from typing import Any, Callable, Dict, List, Literal, Optional, Type

from pydantic import BaseModel, ConfigDict, ValidationError


# ═══════════════════════════════════════════════════════════════════════
# Structured Response Helpers
# ═══════════════════════════════════════════════════════════════════════

ErrorCode = Literal[
    "INVALID_TOOL_INPUT",
    "INVALID_TOOL_OUTPUT",
    "SIMULATION_FAILURE",
    "VALIDATION_ERROR",
]


def _error_response(error_code: ErrorCode, message: str) -> Dict[str, Any]:
    """Build a standardised error envelope."""
    return {
        "status": "error",
        "error_code": error_code,
        "message": message,
    }


def _success_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """Build a standardised success envelope."""
    return {
        "status": "success",
        "data": data,
    }


# ═══════════════════════════════════════════════════════════════════════
# Section 1 — Tool Call Validation Layer
# ═══════════════════════════════════════════════════════════════════════

_MAGNITUDE_LIMIT = 1e9  # reject values beyond ±1 billion


def _scan_for_suspicious_values(payload: Any, path: str = "") -> List[str]:
    """Recursively scan a payload for values exceeding the magnitude limit.

    Returns a list of human-readable violation descriptions.
    """
    violations: List[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            child_path = f"{path}.{key}" if path else key
            violations.extend(_scan_for_suspicious_values(value, child_path))
    elif isinstance(payload, list):
        for idx, item in enumerate(payload):
            violations.extend(_scan_for_suspicious_values(item, f"{path}[{idx}]"))
    elif isinstance(payload, (int, float)) and not isinstance(payload, bool):
        if abs(payload) > _MAGNITUDE_LIMIT:
            violations.append(
                f"Suspicious value at '{path}': {payload} exceeds magnitude limit "
                f"of {_MAGNITUDE_LIMIT}"
            )
    elif isinstance(payload, str):
        # Reject numeric strings that look like coercion attempts
        stripped = payload.strip()
        if stripped and stripped.replace(".", "", 1).replace("-", "", 1).isdigit():
            violations.append(
                f"Numeric string detected at '{path}': '{payload}' — "
                f"implicit coercion is not allowed; use the correct type"
            )
    return violations


def validate_tool_call(
    tool_input: Dict[str, Any],
    input_model: Type[BaseModel],
) -> Dict[str, Any]:
    """Validate a raw LLM tool-call payload against a strict Pydantic schema.

    Pre-execution gate — if this returns an error, the tool must NOT execute.

    Checks performed:
    1. Reject unknown fields (via ``extra="forbid"`` on the model)
    2. Reject missing required fields
    3. Reject wrong data types (no implicit coercion)
    4. Reject numeric strings (``"50000"`` for a float field)
    5. Reject out-of-range values (APR, negatives, etc.)
    6. Reject suspiciously large numbers (> 10⁹)

    Parameters
    ----------
    tool_input : dict
        Raw input payload from the LLM agent.
    input_model : Type[BaseModel]
        The Pydantic model class to validate against.

    Returns
    -------
    dict
        On success: ``{"status": "valid", "validated": <model instance>}``
        On failure: ``{"status": "error", "error_code": "INVALID_TOOL_INPUT", ...}``
    """
    # ── 1. Magnitude scan (before Pydantic, catches extreme values) ───
    suspicious = _scan_for_suspicious_values(tool_input)
    if suspicious:
        return _error_response(
            "INVALID_TOOL_INPUT",
            f"Hallucination defense: {'; '.join(suspicious)}",
        )

    # ── 2. Strict Pydantic validation ─────────────────────────────────
    try:
        validated = input_model.model_validate(tool_input, strict=True)
    except ValidationError as exc:
        return _error_response(
            "INVALID_TOOL_INPUT",
            f"Input schema validation failed: {exc}",
        )

    return {"status": "valid", "validated": validated}


# ═══════════════════════════════════════════════════════════════════════
# Section 2 — Output Verification Layer
# ═══════════════════════════════════════════════════════════════════════

# Auto-repair thresholds
_SCORE_UPPER_CLAMP = 100.5      # stability_score ≤ 100.5 → clamp to 100
_SCORE_LOWER_CLAMP = -0.5       # stability_score ≥ -0.5  → clamp to 0
_LIQ_NEGATIVE_CLAMP = -0.01    # liquidity_ratio > -0.01 → set to 0
_COVERAGE_TOLERANCE = 0.01      # commitment_coverage outside [0, 1] by ≤ 0.01

# Hard rejection thresholds
_SCORE_HARD_UPPER = 110.0
_COVERAGE_HARD_UPPER = 1.1
_LIQ_HARD_LOWER = -1.0


def _check_finite(value: Any, field_name: str) -> Optional[str]:
    """Return an error string if value is NaN, inf, or -inf."""
    if isinstance(value, float) and not math.isfinite(value):
        return f"Non-finite value in '{field_name}': {value}"
    return None


def _auto_repair_result(result: Dict[str, Any]) -> tuple[Dict[str, Any], List[str]]:
    """Attempt minor auto-repairs on a single SimulationResult dict.

    Returns the (possibly repaired) dict and a list of repair descriptions.
    Does NOT modify the original — works on a shallow copy.
    """
    r = dict(result)
    repairs: List[str] = []

    # ── stability_score clamping ──────────────────────────────────────
    score = r.get("stability_score")
    if isinstance(score, (int, float)):
        if score > 100 and score <= _SCORE_UPPER_CLAMP:
            repairs.append(f"Clamped stability_score {score} → 100.0")
            r["stability_score"] = 100.0
        elif score < 0 and score >= _SCORE_LOWER_CLAMP:
            repairs.append(f"Clamped stability_score {score} → 0.0")
            r["stability_score"] = 0.0

    # ── liquidity_ratio clamping ──────────────────────────────────────
    liq = r.get("liquidity_ratio")
    if isinstance(liq, (int, float)):
        if liq < 0 and liq > _LIQ_NEGATIVE_CLAMP:
            repairs.append(f"Clamped liquidity_ratio {liq} → 0.0")
            r["liquidity_ratio"] = 0.0

    # ── commitment_coverage clamping ──────────────────────────────────
    cov = r.get("commitment_coverage")
    if isinstance(cov, (int, float)):
        if cov < 0 and cov >= -_COVERAGE_TOLERANCE:
            repairs.append(f"Clamped commitment_coverage {cov} → 0.0")
            r["commitment_coverage"] = 0.0
        elif cov > 1 and cov <= 1 + _COVERAGE_TOLERANCE:
            repairs.append(f"Clamped commitment_coverage {cov} → 1.0")
            r["commitment_coverage"] = 1.0

    return r, repairs


def _hard_reject_result(result: Dict[str, Any]) -> Optional[str]:
    """Check if a result exceeds hard-rejection thresholds.

    Returns an error message or None if acceptable.
    """
    score = result.get("stability_score")
    if isinstance(score, (int, float)):
        if score > _SCORE_HARD_UPPER:
            return f"stability_score {score} exceeds hard limit {_SCORE_HARD_UPPER}"
        if score < _SCORE_LOWER_CLAMP:
            return f"stability_score {score} below hard limit {_SCORE_LOWER_CLAMP}"

    cov = result.get("commitment_coverage")
    if isinstance(cov, (int, float)):
        if cov > _COVERAGE_HARD_UPPER:
            return f"commitment_coverage {cov} exceeds hard limit {_COVERAGE_HARD_UPPER}"

    liq = result.get("liquidity_ratio")
    if isinstance(liq, (int, float)):
        if liq < _LIQ_HARD_LOWER:
            return f"liquidity_ratio {liq} below hard limit {_LIQ_HARD_LOWER}"

    return None


def verify_tool_output(
    output: Dict[str, Any],
    output_model: Optional[Type[BaseModel]] = None,
) -> Dict[str, Any]:
    """Verify and optionally auto-repair a tool's output.

    Post-execution gate — ensures the simulation produced valid results
    before returning to the LLM agent.

    Steps:
    1. Check for NaN / inf / -inf in all numeric fields
    2. Hard-reject outputs that are wildly out of bounds
    3. Auto-repair minor deviations (within thresholds)
    4. Optionally validate against an output Pydantic model

    Parameters
    ----------
    output : dict
        The raw output dict from the tool function.
    output_model : Type[BaseModel] | None
        Optional Pydantic model for schema validation.

    Returns
    -------
    dict
        The verified (and possibly repaired) output with an added
        ``"repairs"`` key listing any auto-corrections applied.
        On failure: ``{"status": "error", "error_code": "INVALID_TOOL_OUTPUT", ...}``
    """
    if not isinstance(output, dict):
        return _error_response(
            "INVALID_TOOL_OUTPUT",
            f"Expected dict output, got {type(output).__name__}",
        )

    # ── Check top-level status ────────────────────────────────────────
    if output.get("status") == "error":
        # Pass through upstream errors unchanged
        return output

    data = output.get("data", output)
    all_repairs: List[str] = []

    # ── Process results list (SimulationResultSet) ────────────────────
    results = data.get("results") if isinstance(data, dict) else None
    if isinstance(results, list):
        verified_results: List[Dict[str, Any]] = []
        for idx, result in enumerate(results):
            if not isinstance(result, dict):
                return _error_response(
                    "INVALID_TOOL_OUTPUT",
                    f"Result at index {idx} is not a dict",
                )

            # NaN / inf check
            for field_name, value in result.items():
                err = _check_finite(value, f"results[{idx}].{field_name}")
                if err:
                    return _error_response("INVALID_TOOL_OUTPUT", err)

            # Hard rejection
            rejection = _hard_reject_result(result)
            if rejection:
                return _error_response(
                    "INVALID_TOOL_OUTPUT",
                    f"Hard rejection at results[{idx}]: {rejection}",
                )

            # Auto-repair
            repaired, repairs = _auto_repair_result(result)
            all_repairs.extend(f"results[{idx}]: {r}" for r in repairs)
            verified_results.append(repaired)

        # Reassemble
        if isinstance(data, dict) and "results" in data:
            data = dict(data)
            data["results"] = verified_results

    # ── Process scalar scenario results (StressTestResult) ────────────
    scenarios = data.get("scenarios") if isinstance(data, dict) else None
    if isinstance(scenarios, list):
        verified_scenarios: List[Dict[str, Any]] = []
        for idx, scenario in enumerate(scenarios):
            if not isinstance(scenario, dict):
                return _error_response(
                    "INVALID_TOOL_OUTPUT",
                    f"Scenario at index {idx} is not a dict",
                )

            for field_name, value in scenario.items():
                err = _check_finite(value, f"scenarios[{idx}].{field_name}")
                if err:
                    return _error_response("INVALID_TOOL_OUTPUT", err)

            rejection = _hard_reject_result(scenario)
            if rejection:
                return _error_response(
                    "INVALID_TOOL_OUTPUT",
                    f"Hard rejection at scenarios[{idx}]: {rejection}",
                )

            repaired, repairs = _auto_repair_result(scenario)
            all_repairs.extend(f"scenarios[{idx}]: {r}" for r in repairs)
            verified_scenarios.append(repaired)

        if isinstance(data, dict) and "scenarios" in data:
            data = dict(data)
            data["scenarios"] = verified_scenarios

    # ── Check top-level scalar fields (e.g. baseline_score) ───────────
    if isinstance(data, dict):
        for field_name, value in data.items():
            if isinstance(value, float):
                err = _check_finite(value, field_name)
                if err:
                    return _error_response("INVALID_TOOL_OUTPUT", err)

    # ── Optional Pydantic schema validation ───────────────────────────
    if output_model is not None:
        try:
            output_model.model_validate(data, strict=True)
        except ValidationError as exc:
            return _error_response(
                "INVALID_TOOL_OUTPUT",
                f"Output schema validation failed: {exc}",
            )

    # ── Reassemble verified output ────────────────────────────────────
    verified_output = dict(output)
    if "data" in output:
        verified_output["data"] = data
    if all_repairs:
        verified_output["repairs"] = all_repairs

    return verified_output


# ═══════════════════════════════════════════════════════════════════════
# Section 3 — Orchestration Wrapper
# ═══════════════════════════════════════════════════════════════════════

def safe_tool_execution(
    tool_function: Callable[..., Dict[str, Any]],
    raw_input: Dict[str, Any],
    input_model: Type[BaseModel],
    output_model: Optional[Type[BaseModel]] = None,
) -> Dict[str, Any]:
    """Full safety-wrapped tool execution pipeline.

    Flow:
    1. Generate correlation_id
    2. Validate input against ``input_model``
    3. Execute ``tool_function`` (try/except)
    4. Verify output (NaN check, bounds, auto-repair)
    5. Return standardised JSON response

    No unhandled exceptions.  Every code path returns a dict.

    Parameters
    ----------
    tool_function : callable
        The tool wrapper function (e.g. ``_run_subscription_simulation``).
    raw_input : dict
        Raw LLM agent payload.
    input_model : Type[BaseModel]
        Pydantic model for input validation.
    output_model : Type[BaseModel] | None
        Optional Pydantic model for output validation.

    Returns
    -------
    dict
        Always includes ``correlation_id``.  On success also includes
        ``data`` and optional ``repairs``.
    """
    correlation_id = str(uuid.uuid4())

    # ── Step 1: Validate input ────────────────────────────────────────
    validation = validate_tool_call(raw_input, input_model)
    if validation["status"] != "valid":
        validation["correlation_id"] = correlation_id
        return validation

    # ── Step 2: Execute tool ──────────────────────────────────────────
    try:
        # Pass validated fields as kwargs
        validated_model = validation["validated"]
        kwargs = validated_model.model_dump()
        result = tool_function(**kwargs)
    except Exception as exc:
        return {
            "status": "error",
            "error_code": "SIMULATION_FAILURE",
            "message": f"Tool execution failed: {exc}",
            "correlation_id": correlation_id,
        }

    # ── Step 3: Verify output ─────────────────────────────────────────
    verified = verify_tool_output(result, output_model)
    verified["correlation_id"] = correlation_id

    return verified


# ═══════════════════════════════════════════════════════════════════════
# Example Usage
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json

    from langchain_tools import (
        StressTestToolInput,
        SubscriptionToolInput,
        _run_stress_test,
        _run_subscription_simulation,
    )

    def _print(label: str, obj: Any) -> None:
        """Pretty-print a result, handling non-serializable fields."""
        # Remove Pydantic model instances for JSON printing
        printable = {k: v for k, v in obj.items() if not isinstance(v, BaseModel)}
        print(f"\n{'─' * 60}")
        print(f"  {label}")
        print(f"{'─' * 60}")
        print(json.dumps(printable, indent=2))

    # ══════════════════════════════════════════════════════════════════
    # 1. VALID INVOCATION
    # ══════════════════════════════════════════════════════════════════

    valid_input = {
        "monthly_income": 5000.0,
        "fixed_expenses": 2000.0,
        "discretionary_expenses": 800.0,
        "emergency_fund": 10000.0,
        "credit_balance": 3000.0,
        "credit_apr": 0.24,
        "subscriptions": [{"name": "streaming", "monthly_cost": 15.0}],
    }

    result = safe_tool_execution(
        _run_subscription_simulation,
        valid_input,
        SubscriptionToolInput,
    )
    _print("VALID INVOCATION", result)

    # ══════════════════════════════════════════════════════════════════
    # 2. REJECTED HALLUCINATED INPUT (unknown field)
    # ══════════════════════════════════════════════════════════════════

    hallucinated_input = {
        "monthly_income": 5000.0,
        "fixed_expenses": 2000.0,
        "discretionary_expenses": 800.0,
        "emergency_fund": 10000.0,
        "credit_balance": 3000.0,
        "credit_apr": 0.24,
        "hallucinated_field": "the model made this up",  # ← unknown field
        "subscriptions": [{"name": "x", "monthly_cost": 10.0}],
    }

    result = safe_tool_execution(
        _run_subscription_simulation,
        hallucinated_input,
        SubscriptionToolInput,
    )
    _print("REJECTED HALLUCINATED INPUT", result)

    # ══════════════════════════════════════════════════════════════════
    # 3. REJECTED SUSPICIOUS MAGNITUDE
    # ══════════════════════════════════════════════════════════════════

    suspicious_input = {
        "monthly_income": 5000.0,
        "fixed_expenses": 2000.0,
        "discretionary_expenses": 800.0,
        "emergency_fund": 99999999999.0,  # ← > 10^9
        "credit_balance": 3000.0,
        "credit_apr": 0.24,
    }

    result = safe_tool_execution(
        _run_stress_test,
        suspicious_input,
        StressTestToolInput,
    )
    _print("REJECTED SUSPICIOUS MAGNITUDE", result)

    # ══════════════════════════════════════════════════════════════════
    # 4. AUTO-REPAIR CASE
    # ══════════════════════════════════════════════════════════════════

    # Simulate an output with slightly out-of-bound values
    slight_drift_output = {
        "status": "success",
        "data": {
            "results": [
                {
                    "option_id": "test",
                    "stability_score": 100.3,   # slightly above 100
                    "liquidity_ratio": -0.005,   # slightly negative
                    "commitment_coverage": 1.005, # slightly above 1
                    "interest_cost": 0.0,
                    "projected_balance": 1000.0,
                    "risk_flags": [],
                }
            ]
        },
    }

    verified = verify_tool_output(slight_drift_output)
    _print("AUTO-REPAIRED OUTPUT", verified)

    # ══════════════════════════════════════════════════════════════════
    # 5. HARD REJECTION CASE
    # ══════════════════════════════════════════════════════════════════

    wild_output = {
        "status": "success",
        "data": {
            "results": [
                {
                    "option_id": "broken",
                    "stability_score": 150.0,    # way above 110 limit
                    "liquidity_ratio": 1.0,
                    "commitment_coverage": 0.5,
                    "interest_cost": 0.0,
                    "projected_balance": 1000.0,
                    "risk_flags": [],
                }
            ]
        },
    }

    rejected = verify_tool_output(wild_output)
    _print("HARD REJECTION", rejected)

    # ══════════════════════════════════════════════════════════════════
    # 6. NaN REJECTION CASE
    # ══════════════════════════════════════════════════════════════════

    nan_output = {
        "status": "success",
        "data": {
            "results": [
                {
                    "option_id": "nan-test",
                    "stability_score": float("nan"),
                    "liquidity_ratio": 1.0,
                    "commitment_coverage": 0.5,
                    "interest_cost": 0.0,
                    "projected_balance": 1000.0,
                    "risk_flags": [],
                }
            ]
        },
    }

    rejected_nan = verify_tool_output(nan_output)
    _print("NaN REJECTION", rejected_nan)
