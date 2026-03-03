"""
Structured Tool Calling Orchestration — Financial Decision Engine

Production-grade orchestration layer that enforces:
  1. Structured-only tool calling (no freeform LLM responses before execution)
  2. Single tool call per request (no arrays, no nesting, no chaining)
  3. Deterministic intent→tool routing (LLM proposals verified against map)
  4. Full safety pipeline (validate → guard → execute → verify)
  5. Hard-fail on schema violations, routing mismatches, or recursive calls

Flow:
    User Input → LLM (structured mode) → Tool Call JSON
        → parse_llm_tool_call()
        → enforce_structured_routing()
        → execute_with_constraints()
        → format_final_response()

No domain logic.  No simulation math.  No risk computation.
No persistence writes.  No freeform response parsing.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Callable, Dict, List, Literal, Optional, Type

from pydantic import BaseModel, ConfigDict

from langchain_tools import (
    CreditStrategyToolInput,
    SubscriptionToolInput,
    _run_credit_simulation,
    _run_subscription_simulation,
)
from tool_constraints import (
    ALLOWED_TOOLS,
    INTENT_TOOL_MAP,
    ToolInvocationGuard,
    execute_with_constraints,
    route_tool,
)


# ═══════════════════════════════════════════════════════════════════════
# Section 1 — Explicit Tool JSON Schemas
# ═══════════════════════════════════════════════════════════════════════

TOOL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    name: model.model_json_schema()
    for name, model in ALLOWED_TOOLS.items()
}
"""Exported JSON schemas for each tool (used by LLM function-calling config).
   Every schema has ``additionalProperties: false`` via Pydantic ``extra="forbid"``."""


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Return LLM function-calling tool definitions.

    Format is compatible with OpenAI / LangChain function calling:
    ``[{"type": "function", "function": {"name", "description", "parameters"}}]``
    """
    descriptions = {
        "subscription_simulation": (
            "Evaluate one or more subscription options against the user's "
            "financial state. Returns ranked SimulationResults."
        ),
        "credit_strategy_simulation": (
            "Evaluate credit payment strategies (minimum/partial/full) "
            "against the user's financial state. Returns ranked SimulationResults."
        ),
        "stress_test": (
            "Run deterministic stress tests (income shock, expense shock) "
            "on the user's financial state. Returns fragility index and risk level."
        ),
    }
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": descriptions.get(name, ""),
                "parameters": schema,
                "strict": True,
            },
        }
        for name, schema in TOOL_SCHEMAS.items()
    ]


# ═══════════════════════════════════════════════════════════════════════
# Section 2 — LLM Response Parser (Structured Mode Enforcement)
# ═══════════════════════════════════════════════════════════════════════

class StructuredToolCall(BaseModel):
    """Expected shape of LLM structured output."""
    model_config = ConfigDict(extra="forbid", strict=True)

    tool_name: str
    arguments: Dict[str, Any]


ErrorCode = Literal[
    "INVALID_LLM_STRUCTURE",
    "MULTIPLE_TOOL_CALLS_NOT_ALLOWED",
    "TOOL_ROUTING_VIOLATION",
    "FREEFORM_RESPONSE_REJECTED",
]


def _error(code: ErrorCode, message: str, cid: str = "") -> Dict[str, Any]:
    resp: Dict[str, Any] = {
        "status": "error",
        "error_code": code,
        "message": message,
    }
    if cid:
        resp["correlation_id"] = cid
    return resp


def parse_llm_tool_call(
    llm_output: Any,
    *,
    _retry: bool = False,
) -> Dict[str, Any]:
    """Parse and validate the LLM's structured tool-call output.

    Accepts either:
    - A dict with ``tool_name`` + ``arguments``
    - A JSON string encoding the same

    Rejects:
    - Freeform text (no valid JSON)
    - Arrays of tool calls
    - Missing ``tool_name`` or ``arguments``
    - Unknown fields

    Parameters
    ----------
    llm_output
        Raw LLM response (dict, str, or list).
    _retry
        Internal flag — set on second parse attempt.

    Returns
    -------
    dict
        ``{"status": "ok", "tool_name": ..., "arguments": ...}``
        or structured error.
    """
    # ── Reject arrays (multi-tool attempts) ───────────────────────────
    if isinstance(llm_output, list):
        return _error(
            "MULTIPLE_TOOL_CALLS_NOT_ALLOWED",
            f"LLM returned {len(llm_output)} tool calls; only one is allowed",
        )

    # ── Parse JSON string ─────────────────────────────────────────────
    parsed: Any = llm_output
    if isinstance(llm_output, str):
        try:
            parsed = json.loads(llm_output)
        except json.JSONDecodeError:
            if _retry:
                return _error(
                    "INVALID_LLM_STRUCTURE",
                    "LLM returned non-JSON text after retry",
                )
            return _error(
                "FREEFORM_RESPONSE_REJECTED",
                "LLM returned freeform text instead of structured tool call; "
                "will retry once",
            )

    # ── Reject non-dict ───────────────────────────────────────────────
    if not isinstance(parsed, dict):
        return _error(
            "INVALID_LLM_STRUCTURE",
            f"Expected dict, got {type(parsed).__name__}",
        )

    # ── Reject embedded multi-tool structures ─────────────────────────
    for key in ("tool_calls", "tools", "actions", "function_calls"):
        if key in parsed:
            return _error(
                "MULTIPLE_TOOL_CALLS_NOT_ALLOWED",
                f"Payload contains multi-tool key '{key}'; "
                "only one tool per request is allowed",
            )

    # ── Validate against strict schema ────────────────────────────────
    try:
        call = StructuredToolCall.model_validate(parsed, strict=True)
    except Exception as exc:
        return _error(
            "INVALID_LLM_STRUCTURE",
            f"Tool call does not match expected structure: {exc}",
        )

    return {
        "status": "ok",
        "tool_name": call.tool_name,
        "arguments": call.arguments,
    }


# ═══════════════════════════════════════════════════════════════════════
# Section 4 — Deterministic Routing Enforcement
# ═══════════════════════════════════════════════════════════════════════

def enforce_structured_routing(
    intent_type: str,
    proposed_tool: str,
    correlation_id: str,
) -> Optional[Dict[str, Any]]:
    """Verify the LLM's proposed tool matches the deterministic route.

    Returns ``None`` if routing is valid, or a structured error dict.
    """
    routing = route_tool(intent_type)
    if routing["status"] != "ok":
        routing["correlation_id"] = correlation_id
        return routing

    expected = routing["tool_name"]
    if proposed_tool != expected:
        return _error(
            "TOOL_ROUTING_VIOLATION",
            f"LLM proposed tool '{proposed_tool}' but intent "
            f"'{intent_type}' maps to '{expected}'",
            correlation_id,
        )

    return None  # routing valid


# ═══════════════════════════════════════════════════════════════════════
# Section 5 — Tool Function Registry
# ═══════════════════════════════════════════════════════════════════════

TOOL_FUNCTIONS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "subscription_simulation": _run_subscription_simulation,
    "credit_strategy_simulation": _run_credit_simulation,
}
"""Maps tool names to their underlying (pre-safety-layer) functions."""


# ═══════════════════════════════════════════════════════════════════════
# Section 5+6 — Structured Execution Pipeline
# ═══════════════════════════════════════════════════════════════════════

def process_structured_request(
    intent_type: str,
    llm_output: Any,
) -> Dict[str, Any]:
    """Full structured tool calling pipeline.

    1. Parse LLM output (structured mode enforced)
    2. Enforce deterministic routing
    3. Execute via ``execute_with_constraints()``
    4. Format final structured response

    Every code path returns a dict — no unhandled exceptions.

    Parameters
    ----------
    intent_type : str
        Business intent (key into ``INTENT_TOOL_MAP``).
    llm_output
        Raw LLM response (expected: ``{"tool_name": ..., "arguments": ...}``).

    Returns
    -------
    dict
        Structured response with ``status``, ``tool``, ``data``,
        ``risk_flags``, and ``correlation_id``.
    """
    correlation_id = str(uuid.uuid4())

    # ── Step 1: Parse LLM output ──────────────────────────────────────
    parsed = parse_llm_tool_call(llm_output)

    # Retry once on freeform rejection
    if parsed.get("error_code") == "FREEFORM_RESPONSE_REJECTED":
        parsed = parse_llm_tool_call(llm_output, _retry=True)

    if parsed["status"] != "ok":
        parsed["correlation_id"] = correlation_id
        return parsed

    tool_name = parsed["tool_name"]
    arguments = parsed["arguments"]

    # ── Step 2: Enforce deterministic routing ─────────────────────────
    routing_err = enforce_structured_routing(intent_type, tool_name, correlation_id)
    if routing_err is not None:
        return routing_err

    # ── Step 3: Resolve tool function ─────────────────────────────────
    tool_fn = TOOL_FUNCTIONS.get(tool_name)
    if tool_fn is None:
        return _error(
            "TOOL_ROUTING_VIOLATION",
            f"No function registered for tool '{tool_name}'",
            correlation_id,
        )

    # ── Step 4: Execute with full constraint pipeline ─────────────────
    input_model = ALLOWED_TOOLS.get(tool_name)
    if input_model is None:
        return _error(
            "TOOL_ROUTING_VIOLATION",
            f"No input schema for tool '{tool_name}'",
            correlation_id,
        )

    result = execute_with_constraints(
        intent_type=intent_type,
        tool_name=tool_name,
        raw_input=arguments,
        tool_function=tool_fn,
        input_model=input_model,
        correlation_id=correlation_id,
    )

    # ── Step 5: Format final response ─────────────────────────────────
    if result.get("status") != "success":
        return result

    return _format_final_response(result, tool_name, correlation_id)


def _format_final_response(
    result: Dict[str, Any],
    tool_name: str,
    correlation_id: str,
) -> Dict[str, Any]:
    """Build the final structured response envelope.

    Extracts ``risk_flags`` from the first result (if present)
    so they are surfaced at the top level for downstream consumers.
    """
    data = result.get("data", {})
    risk_flags: List[str] = []

    # Extract risk flags from results list
    results_list = data.get("results", [])
    if results_list and isinstance(results_list, list):
        for r in results_list:
            if isinstance(r, dict):
                risk_flags.extend(r.get("risk_flags", []))

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_flags: List[str] = []
    for flag in risk_flags:
        if flag not in seen:
            seen.add(flag)
            unique_flags.append(flag)

    response: Dict[str, Any] = {
        "status": "success",
        "tool": tool_name,
        "data": data,
        "risk_flags": unique_flags,
        "correlation_id": correlation_id,
    }

    # Include auto-repair notices if any
    repairs = result.get("repairs")
    if repairs:
        response["repairs"] = repairs

    return response


# ═══════════════════════════════════════════════════════════════════════
# Example Usage
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    def _show(label: str, obj: Dict[str, Any]) -> None:
        print(f"\n{'─' * 60}")
        print(f"  {label}")
        print(f"{'─' * 60}")
        print(json.dumps(obj, indent=2, default=str))

    # ── Tool definitions (for LLM config) ─────────────────────────────

    _show("TOOL DEFINITIONS", {"tools": get_tool_definitions()})

    # ══════════════════════════════════════════════════════════════════
    # 1. VALID STRUCTURED CALL
    # ══════════════════════════════════════════════════════════════════

    llm_response = {
        "tool_name": "subscription_simulation",
        "arguments": {
            "monthly_income": 5000.0,
            "fixed_expenses": 2000.0,
            "discretionary_expenses": 800.0,
            "emergency_fund": 10000.0,
            "credit_balance": 3000.0,
            "credit_apr": 0.24,
            "subscriptions": [{"name": "streaming", "monthly_cost": 15.0}],
        },
    }

    result = process_structured_request("subscription_purchase", llm_response)
    _show("VALID STRUCTURED CALL", result)

    # ══════════════════════════════════════════════════════════════════
    # 2. VALID JSON STRING (from LLM text output)
    # ══════════════════════════════════════════════════════════════════

    llm_json_str = json.dumps({
        "tool_name": "credit_strategy_simulation",
        "arguments": {
            "monthly_income": 6000.0,
            "fixed_expenses": 2500.0,
            "discretionary_expenses": 1000.0,
            "emergency_fund": 8000.0,
            "credit_balance": 5000.0,
            "credit_apr": 0.22,
            "strategies": ["minimum", "full"],
        },
    })

    result = process_structured_request("credit_payment", llm_json_str)
    _show("VALID JSON STRING", result)

    # ══════════════════════════════════════════════════════════════════
    # 3. REJECTED: FREEFORM TEXT
    # ══════════════════════════════════════════════════════════════════

    result = process_structured_request(
        "subscription_purchase",
        "I think the user should subscribe to the premium plan",
    )
    _show("REJECTED FREEFORM TEXT", result)

    # ══════════════════════════════════════════════════════════════════
    # 4. REJECTED: MULTIPLE TOOL CALLS (array)
    # ══════════════════════════════════════════════════════════════════

    result = process_structured_request(
        "subscription_purchase",
        [
            {"tool_name": "subscription_simulation", "arguments": {}},
            {"tool_name": "credit_strategy_simulation", "arguments": {}},
        ],
    )
    _show("REJECTED MULTIPLE TOOLS", result)

    # ══════════════════════════════════════════════════════════════════
    # 5. REJECTED: ROUTING VIOLATION (wrong tool for intent)
    # ══════════════════════════════════════════════════════════════════

    result = process_structured_request(
        "subscription_purchase",
        {"tool_name": "credit_strategy_simulation", "arguments": {}},
    )
    _show("REJECTED ROUTING VIOLATION", result)

    # ══════════════════════════════════════════════════════════════════
    # 6. REJECTED: EMBEDDED MULTI-TOOL KEY
    # ══════════════════════════════════════════════════════════════════

    result = process_structured_request(
        "subscription_purchase",
        {
            "tool_name": "subscription_simulation",
            "arguments": {},
            "tool_calls": [{"name": "extra"}],
        },
    )
    _show("REJECTED EMBEDDED MULTI-TOOL", result)

    # ══════════════════════════════════════════════════════════════════
    # 7. REJECTED: INVALID INTENT
    # ══════════════════════════════════════════════════════════════════

    result = process_structured_request(
        "buy_crypto",
        {"tool_name": "subscription_simulation", "arguments": {}},
    )
    _show("REJECTED INVALID INTENT", result)

    # ══════════════════════════════════════════════════════════════════
    # Example Agent Initialization (read-only snippet)
    # ══════════════════════════════════════════════════════════════════
    #
    # from langchain_openai import ChatOpenAI
    #
    # llm = ChatOpenAI(model="gpt-4o", temperature=0)
    #
    # # Bind tools in structured mode (no freeform)
    # llm_with_tools = llm.bind_tools(
    #     get_tool_definitions(),
    #     tool_choice="required",       # force tool call, no text
    #     parallel_tool_calls=False,     # single call only
    # )
    #
    # # On each request:
    # response = llm_with_tools.invoke(user_message)
    # tool_call = response.tool_calls[0]  # single call guaranteed
    #
    # result = process_structured_request(
    #     intent_type=determine_intent(user_message),  # deterministic
    #     llm_output={
    #         "tool_name": tool_call["name"],
    #         "arguments": tool_call["args"],
    #     },
    # )
