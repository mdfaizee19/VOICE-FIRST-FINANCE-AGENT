"""
LangChain Integration Adapter — Financial Decision Engine

Exposes the deterministic financial simulation tools as LangChain ``@tool``
functions suitable for binding to any LangChain agent.

Architecture:
    Each ``@tool`` wrapper is a **thin adapter** that delegates entirely to
    ``execute_with_constraints()`` from the constraint layer.  No wrapper
    contains domain logic, simulation math, validation code, or risk
    computation — all of that lives in the existing layers.

Guarantees:
  - Strict Pydantic v2 schema enforcement (via existing validation layer)
  - Single tool call per request (via invocation guard)
  - Deterministic intent→tool routing (via INTENT_TOOL_MAP)
  - Output verification with auto-repair (via verification layer)
  - Structured error responses on any failure
  - No LLM math, no database logic, no logging

Usage:
  - Import ``tools`` list and bind to a LangChain agent
  - Each tool returns structured JSON: success envelope or error envelope
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List

from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from langchain_tools import (
    CreditStrategyToolInput,
    SubscriptionToolInput,
    _run_credit_simulation,
    _run_subscription_simulation,
)
from tool_constraints import (
    ALLOWED_TOOLS,
    INTENT_TOOL_MAP,
    execute_with_constraints,
)


# ═══════════════════════════════════════════════════════════════════════
# Helper — Final Response Formatting
# ═══════════════════════════════════════════════════════════════════════

def _format_response(result: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
    """Build the final structured response.

    Extracts and deduplicates ``risk_flags`` from nested results.
    """
    if result.get("status") != "success":
        return result

    data = result.get("data", {})
    risk_flags: List[str] = []

    for r in data.get("results", []):
        if isinstance(r, dict):
            risk_flags.extend(r.get("risk_flags", []))

    # Deduplicate preserving order
    seen: set[str] = set()
    unique: List[str] = []
    for f in risk_flags:
        if f not in seen:
            seen.add(f)
            unique.append(f)

    response: Dict[str, Any] = {
        "status": "success",
        "tool": tool_name,
        "data": data,
        "risk_flags": unique,
        "correlation_id": result.get("correlation_id", ""),
    }
    if result.get("repairs"):
        response["repairs"] = result["repairs"]
    return response


# ═══════════════════════════════════════════════════════════════════════
# Error Handler for LangChain StructuredTool
# ═══════════════════════════════════════════════════════════════════════

def _handle_error(error: Exception) -> str:
    """Convert exceptions to structured JSON error strings."""
    return json.dumps({
        "status": "error",
        "error_code": "TOOL_EXECUTION_ERROR",
        "message": str(error),
    })


# ═══════════════════════════════════════════════════════════════════════
# Tool 1 — Subscription Simulation
# ═══════════════════════════════════════════════════════════════════════

def _subscription_adapter(**kwargs: Any) -> Dict[str, Any]:
    """Thin adapter: delegates to execute_with_constraints().

    No validation logic here — the constraint layer handles everything.
    """
    result = execute_with_constraints(
        intent_type="subscription_purchase",
        tool_name="subscription_simulation",
        raw_input=kwargs,
        tool_function=_run_subscription_simulation,
        input_model=SubscriptionToolInput,
        correlation_id=str(uuid.uuid4()),
    )
    return _format_response(result, "subscription_simulation")


subscription_simulation_tool = StructuredTool.from_function(
    func=_subscription_adapter,
    name="subscription_simulation",
    description=(
        "Evaluate one or more subscription options against the user's "
        "financial state. Returns a ranked list of SimulationResults "
        "sorted by stability_score (descending), including a "
        "'no-subscription' baseline. All monetary inputs must be monthly "
        "floats. APR is a decimal (e.g. 0.24 for 24%)."
    ),
    args_schema=SubscriptionToolInput,
    return_direct=False,
    handle_tool_error=_handle_error,
    handle_validation_error=_handle_error,
)


# ═══════════════════════════════════════════════════════════════════════
# Tool 2 — Credit Strategy Simulation
# ═══════════════════════════════════════════════════════════════════════

def _credit_strategy_adapter(**kwargs: Any) -> Dict[str, Any]:
    """Thin adapter: delegates to execute_with_constraints().

    No validation logic here — the constraint layer handles everything.
    """
    result = execute_with_constraints(
        intent_type="credit_payment",
        tool_name="credit_strategy_simulation",
        raw_input=kwargs,
        tool_function=_run_credit_simulation,
        input_model=CreditStrategyToolInput,
        correlation_id=str(uuid.uuid4()),
    )
    return _format_response(result, "credit_strategy_simulation")


credit_strategy_simulation_tool = StructuredTool.from_function(
    func=_credit_strategy_adapter,
    name="credit_strategy_simulation",
    description=(
        "Evaluate credit payment strategies (minimum, partial, full) "
        "against the user's financial state. Returns a ranked list of "
        "SimulationResults sorted by stability_score (descending). "
        "For 'partial' strategy, partial_percentage (0–1) is required. "
        "All monetary inputs must be monthly floats. APR as decimal."
    ),
    args_schema=CreditStrategyToolInput,
    return_direct=False,
    handle_tool_error=_handle_error,
    handle_validation_error=_handle_error,
)


# ═══════════════════════════════════════════════════════════════════════
# Tool Registry — LangChain tools list
# ═══════════════════════════════════════════════════════════════════════

tools: List[StructuredTool] = [
    subscription_simulation_tool,
    credit_strategy_simulation_tool,
]
"""LangChain-compatible tools list for agent binding.

Usage::

    from langchain_adapter import tools
    agent = create_react_agent(llm, tools)
"""


# ═══════════════════════════════════════════════════════════════════════
# Example Usage
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    def _show(label: str, obj: Any) -> None:
        print(f"\n{'─' * 60}")
        print(f"  {label}")
        print(f"{'─' * 60}")
        if isinstance(obj, str):
            # handle_validation_error returns a string
            try:
                print(json.dumps(json.loads(obj), indent=2))
            except Exception:
                print(obj)
        else:
            print(json.dumps(obj, indent=2, default=str))

    # ══════════════════════════════════════════════════════════════════
    # 1. VALID SUBSCRIPTION SIMULATION
    # ══════════════════════════════════════════════════════════════════

    result = subscription_simulation_tool.invoke({
        "monthly_income": 5000.0,
        "fixed_expenses": 2000.0,
        "discretionary_expenses": 800.0,
        "emergency_fund": 10000.0,
        "credit_balance": 3000.0,
        "credit_apr": 0.24,
        "subscriptions": [
            {"name": "streaming", "monthly_cost": 15.0},
            {"name": "cloud-storage", "monthly_cost": 12.0},
        ],
    })
    _show("VALID SUBSCRIPTION SIMULATION", result)

    # ══════════════════════════════════════════════════════════════════
    # 2. VALID CREDIT STRATEGY SIMULATION
    # ══════════════════════════════════════════════════════════════════

    result = credit_strategy_simulation_tool.invoke({
        "monthly_income": 6000.0,
        "fixed_expenses": 2500.0,
        "discretionary_expenses": 1000.0,
        "emergency_fund": 8000.0,
        "credit_balance": 5000.0,
        "credit_apr": 0.22,
        "strategy_type": "minimum",
    })
    _show("VALID CREDIT STRATEGY SIMULATION", result)

    # ══════════════════════════════════════════════════════════════════
    # 3. ERROR: UNKNOWN FIELD (schema rejects hallucinated key)
    # ══════════════════════════════════════════════════════════════════

    result = subscription_simulation_tool.invoke({
        "monthly_income": 5000.0,
        "fixed_expenses": 2000.0,
        "discretionary_expenses": 800.0,
        "emergency_fund": 10000.0,
        "credit_balance": 3000.0,
        "credit_apr": 0.24,
        "hallucinated_field": "LLM made this up",
        "subscriptions": [{"name": "x", "monthly_cost": 10.0}],
    })
    _show("ERROR: UNKNOWN FIELD", result)

    # ══════════════════════════════════════════════════════════════════
    # 4. ERROR: NEGATIVE INCOME (schema-level validation)
    # ══════════════════════════════════════════════════════════════════

    result = subscription_simulation_tool.invoke({
        "monthly_income": -1000.0,
        "fixed_expenses": 500.0,
        "discretionary_expenses": 200.0,
        "emergency_fund": 0.0,
        "credit_balance": 0.0,
        "credit_apr": 0.0,
        "subscriptions": [],
    })
    _show("ERROR: NEGATIVE INCOME", result)

    # ══════════════════════════════════════════════════════════════════
    # 5. ERROR: MISSING REQUIRED FIELD
    # ══════════════════════════════════════════════════════════════════

    result = credit_strategy_simulation_tool.invoke({
        "monthly_income": 5000.0,
        # missing: fixed_expenses, discretionary_expenses, etc.
        "strategy_type": "minimum",
    })
    _show("ERROR: MISSING REQUIRED FIELD", result)

    # ══════════════════════════════════════════════════════════════════
    # 6. TOOL SCHEMAS (for agent inspection)
    # ══════════════════════════════════════════════════════════════════

    for t in tools:
        _show(f"SCHEMA: {t.name}", {
            "name": t.name,
            "description": t.description[:80] + "...",
            "input_schema": t.args_schema.model_json_schema() if t.args_schema else {},
        })

    # ══════════════════════════════════════════════════════════════════
    # Example Agent Initialization (LangChain + LangGraph)
    # ══════════════════════════════════════════════════════════════════
    #
    # from langchain_openai import ChatOpenAI
    # from langgraph.prebuilt import create_react_agent
    #
    # llm = ChatOpenAI(model="gpt-4o", temperature=0)
    #
    # # Bind tools with structured mode enforcement
    # agent = create_react_agent(
    #     llm,
    #     tools,
    #     # The agent will use tool_choice="required" to force
    #     # structured output and disable freeform text.
    # )
    #
    # # Invoke with a user message
    # result = agent.invoke({
    #     "messages": [("user", "What happens if I add a $15/mo streaming subscription?")]
    # })
    #
    # # The agent automatically:
    # # 1. Selects subscription_simulation tool
    # # 2. LangChain validates input against SubscriptionToolInput schema
    # # 3. Tool adapter calls execute_with_constraints()
    # # 4. Constraint layer validates, guards, executes, verifies
    # # 5. Structured response returned to agent
    # # 6. Agent generates summary from structured data
