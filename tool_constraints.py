"""
Tool Invocation Constraint Layer — Financial Simulation System

Production-grade orchestration constraints that enforce:
  1. Max 1 simulation tool per request turn
  2. No recursive tool calls
  3. Deterministic intent → tool routing (no LLM-driven selection)
  4. Explicit rejection of invalid / hallucinated invocation attempts

Components
----------
- ``ALLOWED_TOOLS``         — strict tool registry
- ``INTENT_TOOL_MAP``       — deterministic intent → tool name mapping
- ``route_tool()``          — resolve intent to authorised tool name
- ``ToolInvocationGuard``   — per-request invocation state machine
- ``execute_with_constraints()`` — full constrained execution wrapper

No LLM calls.  No simulation math.  No database.  No logging layer.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Literal, Optional, Type

from pydantic import BaseModel

from langchain_tools import (
    CreditStrategyToolInput,
    StressTestToolInput,
    SubscriptionToolInput,
)
from llm_safety import validate_tool_call, verify_tool_output


# ═══════════════════════════════════════════════════════════════════════
# Tool Registry & Intent Map
# ═══════════════════════════════════════════════════════════════════════

ALLOWED_TOOLS: Dict[str, Type[BaseModel]] = {
    "subscription_simulation": SubscriptionToolInput,
    "credit_strategy_simulation": CreditStrategyToolInput,
    "stress_test": StressTestToolInput,
}
"""Whitelist of tool names → their strict Pydantic input schemas."""

INTENT_TOOL_MAP: Dict[str, str] = {
    "subscription_purchase": "subscription_simulation",
    "credit_payment": "credit_strategy_simulation",
    "financial_stress_test": "stress_test",
}
"""Deterministic mapping from business intent to authorised tool name."""

# Tool names must never contain these suspicious patterns
_FORBIDDEN_PATTERNS: re.Pattern[str] = re.compile(
    r"(__call__|__init__|eval|exec|import|lambda|compile|getattr|setattr)",
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════════════
# Structured Error Helper
# ═══════════════════════════════════════════════════════════════════════

ErrorCode = Literal[
    "INVALID_INTENT",
    "UNAUTHORIZED_TOOL",
    "MULTIPLE_SIMULATION_CALLS",
    "RECURSIVE_TOOL_CALL",
    "TOOL_ROUTING_VIOLATION",
    "SUSPICIOUS_TOOL_NAME",
    "INVALID_TOOL_INPUT",
    "SIMULATION_FAILURE",
    "INVALID_TOOL_OUTPUT",
    "MULTI_TOOL_ATTEMPT",
]


def _error(
    error_code: ErrorCode,
    message: str,
    correlation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a standardised error envelope."""
    resp: Dict[str, Any] = {
        "status": "error",
        "error_code": error_code,
        "message": message,
    }
    if correlation_id is not None:
        resp["correlation_id"] = correlation_id
    return resp


# ═══════════════════════════════════════════════════════════════════════
# Section 1 — Deterministic Tool Routing
# ═══════════════════════════════════════════════════════════════════════

def route_tool(intent_type: str) -> Dict[str, Any]:
    """Resolve a business intent to an authorised tool name.

    Returns
    -------
    dict
        On success: ``{"status": "ok", "tool_name": "<name>"}``
        On failure: structured error with ``INVALID_INTENT``
    """
    if not isinstance(intent_type, str) or not intent_type.strip():
        return _error(
            "INVALID_INTENT",
            "intent_type must be a non-empty string",
        )

    tool_name = INTENT_TOOL_MAP.get(intent_type)
    if tool_name is None:
        return _error(
            "INVALID_INTENT",
            f"No tool mapped for intent '{intent_type}'. "
            f"Allowed intents: {sorted(INTENT_TOOL_MAP.keys())}",
        )

    return {"status": "ok", "tool_name": tool_name}


# ═══════════════════════════════════════════════════════════════════════
# Section 2 — Tool Invocation Guard
# ═══════════════════════════════════════════════════════════════════════

class ToolInvocationGuard:
    """Per-request invocation state machine enforcing orchestration rules.

    Create a **new instance for each request** — no shared mutable state.

    Rules enforced:
    1. Max 1 simulation tool per request
    2. No recursive tool calls (stack depth > 0)
    3. Tool must exist in ``ALLOWED_TOOLS``
    4. Tool name must match the routed intent
    5. Tool name must not contain suspicious patterns
    """

    __slots__ = ("correlation_id", "_invocation_count", "_active_stack")

    def __init__(self, correlation_id: str) -> None:
        self.correlation_id = correlation_id
        self._invocation_count: int = 0
        self._active_stack: List[str] = []

    # ── Public API ────────────────────────────────────────────────────

    def check_invocation(
        self,
        tool_name: str,
        routed_tool: str,
    ) -> Optional[Dict[str, Any]]:
        """Validate that a tool invocation is permitted.

        Returns ``None`` if the invocation is allowed, or a structured
        error dict if a constraint is violated.
        """
        # Rule 5: suspicious tool name patterns
        if _FORBIDDEN_PATTERNS.search(tool_name):
            return _error(
                "SUSPICIOUS_TOOL_NAME",
                f"Tool name '{tool_name}' contains a forbidden pattern",
                self.correlation_id,
            )

        # Rule 3: tool must be in whitelist
        if tool_name not in ALLOWED_TOOLS:
            return _error(
                "UNAUTHORIZED_TOOL",
                f"Tool '{tool_name}' is not in the allowed tool registry. "
                f"Allowed: {sorted(ALLOWED_TOOLS.keys())}",
                self.correlation_id,
            )

        # Rule 4: routed tool must match requested tool
        if tool_name != routed_tool:
            return _error(
                "TOOL_ROUTING_VIOLATION",
                f"Requested tool '{tool_name}' does not match routed tool "
                f"'{routed_tool}' for the given intent",
                self.correlation_id,
            )

        # Rule 2: no recursive invocation
        if self._active_stack:
            return _error(
                "RECURSIVE_TOOL_CALL",
                f"Recursive tool invocation detected: "
                f"'{tool_name}' attempted while "
                f"'{self._active_stack[-1]}' is still executing",
                self.correlation_id,
            )

        # Rule 1: max 1 simulation per request
        if self._invocation_count >= 1:
            return _error(
                "MULTIPLE_SIMULATION_CALLS",
                "Only one simulation tool is allowed per request "
                f"(already invoked {self._invocation_count} tool(s))",
                self.correlation_id,
            )

        return None  # all checks passed

    def enter(self, tool_name: str) -> None:
        """Mark a tool as actively executing (push onto stack)."""
        self._active_stack.append(tool_name)
        self._invocation_count += 1

    def exit(self) -> None:
        """Mark tool execution as complete (pop from stack)."""
        if self._active_stack:
            self._active_stack.pop()

    @property
    def invocation_count(self) -> int:
        return self._invocation_count

    @property
    def is_executing(self) -> bool:
        return len(self._active_stack) > 0


# ═══════════════════════════════════════════════════════════════════════
# Hallucination Defense — Multi-Tool Attempt Detection
# ═══════════════════════════════════════════════════════════════════════

def _reject_multi_tool_payload(raw_input: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Detect and reject payloads that embed multiple tool calls.

    LLMs sometimes hallucinate array-of-tools structures or nested
    ``tool_calls`` / ``tools`` keys.  Reject them pre-execution.
    """
    suspicious_keys = {"tool_calls", "tools", "actions", "function_calls"}
    found = suspicious_keys & raw_input.keys()
    if found:
        return _error(
            "MULTI_TOOL_ATTEMPT",
            f"Payload contains suspicious multi-tool keys: {sorted(found)}. "
            "Only one tool per turn is allowed.",
        )
    return None


# ═══════════════════════════════════════════════════════════════════════
# Section 3 — Constrained Execution Wrapper
# ═══════════════════════════════════════════════════════════════════════

def execute_with_constraints(
    intent_type: str,
    tool_name: str,
    raw_input: Dict[str, Any],
    tool_function: Callable[..., Dict[str, Any]],
    input_model: Type[BaseModel],
    output_model: Optional[Type[BaseModel]] = None,
    correlation_id: str = "",
) -> Dict[str, Any]:
    """Execute a simulation tool under full orchestration constraints.

    Pipeline:
    1. Route intent → authorised tool name
    2. Reject multi-tool payloads
    3. Verify tool name matches routed tool
    4. Enforce invocation guard (single-call, no recursion, whitelist)
    5. Validate input via ``validate_tool_call``
    6. Execute tool (try / except)
    7. Verify output via ``verify_tool_output``
    8. Return standardised response

    Every code path returns a dict — no unhandled exceptions.

    Parameters
    ----------
    intent_type : str
        Business intent (key into ``INTENT_TOOL_MAP``).
    tool_name : str
        Tool the caller wishes to invoke.
    raw_input : dict
        Raw payload from the LLM agent.
    tool_function : callable
        The underlying tool wrapper function.
    input_model : Type[BaseModel]
        Pydantic model for strict input validation.
    output_model : Type[BaseModel] | None
        Optional Pydantic model for output validation.
    correlation_id : str
        Pre-existing correlation ID (generated if empty).

    Returns
    -------
    dict
        Standardised response with ``status``, ``correlation_id``,
        ``tool``, and ``data`` (on success) or ``error_code`` + ``message``
        (on failure).
    """
    cid = correlation_id or __import__("uuid").uuid4().hex

    # ── Step 1: Route intent ──────────────────────────────────────────
    routing = route_tool(intent_type)
    if routing["status"] != "ok":
        routing["correlation_id"] = cid
        return routing

    routed_tool = routing["tool_name"]

    # ── Step 2: Reject multi-tool payloads ────────────────────────────
    multi_err = _reject_multi_tool_payload(raw_input)
    if multi_err is not None:
        multi_err["correlation_id"] = cid
        return multi_err

    # ── Step 3: Instantiate guard and check constraints ───────────────
    guard = ToolInvocationGuard(cid)
    constraint_err = guard.check_invocation(tool_name, routed_tool)
    if constraint_err is not None:
        return constraint_err

    # ── Step 4: Validate input ────────────────────────────────────────
    validation = validate_tool_call(raw_input, input_model)
    if validation["status"] != "valid":
        validation["correlation_id"] = cid
        return validation

    # ── Step 5: Execute tool ──────────────────────────────────────────
    guard.enter(tool_name)
    try:
        validated_model = validation["validated"]
        kwargs = validated_model.model_dump()
        result = tool_function(**kwargs)
    except Exception as exc:
        guard.exit()
        return _error("SIMULATION_FAILURE", f"Tool execution failed: {exc}", cid)
    guard.exit()

    # ── Step 6: Verify output ─────────────────────────────────────────
    verified = verify_tool_output(result, output_model)
    if verified.get("status") == "error":
        verified["correlation_id"] = cid
        return verified

    # ── Step 7: Return success ────────────────────────────────────────
    verified["correlation_id"] = cid
    verified["tool"] = tool_name
    return verified


# ═══════════════════════════════════════════════════════════════════════
# Example Usage
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json

    from langchain_tools import _run_credit_simulation, _run_subscription_simulation

    def _show(label: str, obj: Dict[str, Any]) -> None:
        printable = {k: v for k, v in obj.items() if not isinstance(v, BaseModel)}
        print(f"\n{'─' * 60}")
        print(f"  {label}")
        print(f"{'─' * 60}")
        print(json.dumps(printable, indent=2))

    # ══════════════════════════════════════════════════════════════════
    # 1. VALID EXECUTION
    # ══════════════════════════════════════════════════════════════════

    result = execute_with_constraints(
        intent_type="subscription_purchase",
        tool_name="subscription_simulation",
        raw_input={
            "monthly_income": 5000.0,
            "fixed_expenses": 2000.0,
            "discretionary_expenses": 800.0,
            "emergency_fund": 10000.0,
            "credit_balance": 3000.0,
            "credit_apr": 0.24,
            "subscriptions": [{"name": "streaming", "monthly_cost": 15.0}],
        },
        tool_function=_run_subscription_simulation,
        input_model=SubscriptionToolInput,
        correlation_id="example-valid-001",
    )
    _show("VALID EXECUTION", result)

    # ══════════════════════════════════════════════════════════════════
    # 2. ROUTING VIOLATION — wrong tool for intent
    # ══════════════════════════════════════════════════════════════════

    result = execute_with_constraints(
        intent_type="subscription_purchase",
        tool_name="credit_strategy_simulation",  # ← wrong tool!
        raw_input={},
        tool_function=_run_credit_simulation,
        input_model=CreditStrategyToolInput,
        correlation_id="example-routing-002",
    )
    _show("ROUTING VIOLATION", result)

    # ══════════════════════════════════════════════════════════════════
    # 3. INVALID INTENT
    # ══════════════════════════════════════════════════════════════════

    result = execute_with_constraints(
        intent_type="buy_stocks",  # ← no such intent
        tool_name="subscription_simulation",
        raw_input={},
        tool_function=_run_subscription_simulation,
        input_model=SubscriptionToolInput,
        correlation_id="example-intent-003",
    )
    _show("INVALID INTENT", result)

    # ══════════════════════════════════════════════════════════════════
    # 4. MULTIPLE SIMULATION CALLS (guard reuse)
    # ══════════════════════════════════════════════════════════════════

    guard = ToolInvocationGuard("example-multi-004")
    # First call — should pass
    err = guard.check_invocation("subscription_simulation", "subscription_simulation")
    assert err is None, "First call should pass"
    guard.enter("subscription_simulation")
    guard.exit()
    # Second call — should fail
    err = guard.check_invocation("subscription_simulation", "subscription_simulation")
    assert err is not None
    _show("MULTIPLE SIMULATION CALLS", err)

    # ══════════════════════════════════════════════════════════════════
    # 5. RECURSIVE TOOL CALL
    # ══════════════════════════════════════════════════════════════════

    guard2 = ToolInvocationGuard("example-recursive-005")
    guard2.enter("subscription_simulation")  # simulate active execution
    err = guard2.check_invocation("subscription_simulation", "subscription_simulation")
    assert err is not None
    _show("RECURSIVE TOOL CALL", err)
    guard2.exit()

    # ══════════════════════════════════════════════════════════════════
    # 6. UNAUTHORIZED TOOL
    # ══════════════════════════════════════════════════════════════════

    guard3 = ToolInvocationGuard("example-unauth-006")
    err = guard3.check_invocation("hacked_tool_eval", "subscription_simulation")
    assert err is not None
    _show("UNAUTHORIZED TOOL", err)

    # ══════════════════════════════════════════════════════════════════
    # 7. SUSPICIOUS TOOL NAME
    # ══════════════════════════════════════════════════════════════════

    guard4 = ToolInvocationGuard("example-suspicious-007")
    err = guard4.check_invocation("tool__call__hack", "subscription_simulation")
    assert err is not None
    _show("SUSPICIOUS TOOL NAME", err)

    # ══════════════════════════════════════════════════════════════════
    # 8. MULTI-TOOL PAYLOAD
    # ══════════════════════════════════════════════════════════════════

    result = execute_with_constraints(
        intent_type="subscription_purchase",
        tool_name="subscription_simulation",
        raw_input={
            "tool_calls": [{"name": "tool1"}, {"name": "tool2"}],  # ← LLM hallucination
            "monthly_income": 5000.0,
        },
        tool_function=_run_subscription_simulation,
        input_model=SubscriptionToolInput,
        correlation_id="example-multi-payload-008",
    )
    _show("MULTI-TOOL PAYLOAD REJECTION", result)
