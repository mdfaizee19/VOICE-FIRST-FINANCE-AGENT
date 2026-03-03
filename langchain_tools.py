"""
LangChain Structured Tools — Financial Simulation System

Production-grade, type-safe tool wrappers for:
  1. simulate_subscription_tool
  2. simulate_credit_strategy_tool
  3. run_stress_test_tool

All tools use StructuredTool with explicit Pydantic v2 input schemas,
return JSON-serializable dicts only, and wrap every execution path
in guardrailed error handling.

No natural-language parsing.  No LLM calls.  No implicit coercion.
Stateless, thread-safe, deterministic.
"""

from __future__ import annotations

import copy
import traceback
from typing import Any, Callable, Dict, List, Literal, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from financial_schemas import (
    Commitment,
    CreditStrategyRequest,
    FinancialState,
    SubscriptionOption,
)
from simulation_engine import (
    _compute_result,
    _monthly_interest,
    simulate_credit_strategies,
    simulate_subscription_options,
)
from simulation_schemas import SimulationResult
from stress_testing import run_stress_test


# ═══════════════════════════════════════════════════════════════════════
# Shared Response Models
# ═══════════════════════════════════════════════════════════════════════

class ToolError(BaseModel):
    """Structured error payload returned by any tool on failure."""

    model_config = ConfigDict(extra="forbid", strict=True)

    error_code: Literal["INVALID_INPUT", "SIMULATION_FAILURE", "VALIDATION_ERROR"]
    message: str


def _success(data: dict) -> Dict[str, Any]:
    """Build a success envelope."""
    return {"status": "success", "data": data}


def _failure(error_code: Literal["INVALID_INPUT", "SIMULATION_FAILURE", "VALIDATION_ERROR"],
             message: str) -> Dict[str, Any]:
    """Build a failure envelope."""
    return ToolError(error_code=error_code, message=message).model_dump() | {"status": "error"}


def _handle_tool_error(error: Exception) -> str:
    """LangChain error handler — returns structured JSON string for schema-level failures."""
    import json
    return json.dumps(_failure("INVALID_INPUT", str(error)))


def _handle_validation_error(error: Exception) -> str:
    """LangChain validation error handler — returns structured JSON string."""
    import json
    return json.dumps(_failure("INVALID_INPUT", str(error)))


# ═══════════════════════════════════════════════════════════════════════
# 1) simulate_subscription_tool
# ═══════════════════════════════════════════════════════════════════════

class CommitmentInput(BaseModel):
    """A single financial commitment for tool input."""
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str
    amount: float
    due_month: int

    @field_validator("id")
    @classmethod
    def id_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Commitment id must be non-empty")
        return v

    @field_validator("amount")
    @classmethod
    def amount_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Commitment amount must be > 0 (got {v})")
        return v

    @field_validator("due_month")
    @classmethod
    def due_month_range(cls, v: int) -> int:
        if v < 1 or v > 12:
            raise ValueError(f"due_month must be 1–12 (got {v})")
        return v


class SubscriptionToolInput(BaseModel):
    """Input schema for simulate_subscription_tool.

    Accepts the user's financial state and a list of subscription options
    to evaluate.  All monetary values must be monthly floats.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    monthly_income: float
    fixed_expenses: float
    discretionary_expenses: float
    emergency_fund: float
    credit_balance: float
    credit_apr: float
    commitments: List[CommitmentInput] = []
    subscriptions: List[Dict[str, Any]]

    @field_validator("monthly_income", "fixed_expenses", "discretionary_expenses",
                     "emergency_fund", "credit_balance")
    @classmethod
    def non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"Value must be >= 0 (got {v})")
        return v

    @field_validator("credit_apr")
    @classmethod
    def apr_range(cls, v: float) -> float:
        if v < 0 or v > 1:
            raise ValueError(f"credit_apr must be 0–1 (got {v})")
        return v

    @field_validator("subscriptions")
    @classmethod
    def subscriptions_not_empty(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not v:
            raise ValueError("At least one subscription option is required")
        return v


def _run_subscription_simulation(
    monthly_income: float,
    fixed_expenses: float,
    discretionary_expenses: float,
    emergency_fund: float,
    credit_balance: float,
    credit_apr: float,
    commitments: List[CommitmentInput],
    subscriptions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Execute subscription simulation with full guardrails."""
    try:
        # ── Build domain objects ──────────────────────────────────
        commitment_models = [
            Commitment(id=c.id, amount=c.amount, due_month=c.due_month)
            for c in commitments
        ]
        state = FinancialState(
            monthly_income=monthly_income,
            fixed_expenses=fixed_expenses,
            discretionary_expenses=discretionary_expenses,
            emergency_fund=emergency_fund,
            credit_balance=credit_balance,
            credit_apr=credit_apr,
            commitments=commitment_models,
        )
    except (ValidationError, ValueError) as exc:
        return _failure("INVALID_INPUT", f"Financial state validation failed: {exc}")

    try:
        sub_options = [
            SubscriptionOption(name=s["name"], monthly_cost=s["monthly_cost"])
            for s in subscriptions
        ]
    except (ValidationError, ValueError, KeyError, TypeError) as exc:
        return _failure("INVALID_INPUT", f"Subscription option validation failed: {exc}")

    try:
        result_set = simulate_subscription_options(state, sub_options)
    except Exception as exc:
        return _failure("SIMULATION_FAILURE", f"Simulation engine error: {exc}")

    try:
        output = result_set.model_dump()
    except Exception as exc:
        return _failure("VALIDATION_ERROR", f"Output serialization failed: {exc}")

    return _success(output)


simulate_subscription_tool = StructuredTool.from_function(
    func=_run_subscription_simulation,
    name="simulate_subscription_options",
    description=(
        "Evaluate one or more subscription options against the user's "
        "financial state.  Returns a ranked list of SimulationResults "
        "sorted by stability_score (descending).  Includes a 'no-subscription' "
        "baseline for comparison.  All inputs must be monthly floats.  "
        "APR is expressed as a decimal (e.g., 0.24 for 24%)."
    ),
    args_schema=SubscriptionToolInput,
    return_direct=False,
    handle_tool_error=_handle_tool_error,
    handle_validation_error=_handle_validation_error,
)


# ═══════════════════════════════════════════════════════════════════════
# 2) simulate_credit_strategy_tool
# ═══════════════════════════════════════════════════════════════════════

class CreditStrategyToolInput(BaseModel):
    """Input schema for simulate_credit_strategy_tool.

    Accepts the user's financial state and a credit payment strategy
    to evaluate (minimum, partial, or full repayment).
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    monthly_income: float
    fixed_expenses: float
    discretionary_expenses: float
    emergency_fund: float
    credit_balance: float
    credit_apr: float
    commitments: List[CommitmentInput] = []
    strategy_type: Literal["minimum", "partial", "full"]
    partial_percentage: Optional[float] = None

    @field_validator("monthly_income", "fixed_expenses", "discretionary_expenses",
                     "emergency_fund", "credit_balance")
    @classmethod
    def non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"Value must be >= 0 (got {v})")
        return v

    @field_validator("credit_apr")
    @classmethod
    def apr_range(cls, v: float) -> float:
        if v < 0 or v > 1:
            raise ValueError(f"credit_apr must be 0–1 (got {v})")
        return v

    @field_validator("partial_percentage")
    @classmethod
    def partial_pct_range(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and (v < 0 or v > 1):
            raise ValueError(f"partial_percentage must be 0–1 (got {v})")
        return v

    @model_validator(mode="after")
    def strategy_consistency(self) -> CreditStrategyToolInput:
        if self.strategy_type == "partial" and self.partial_percentage is None:
            raise ValueError(
                "partial_percentage is required when strategy_type is 'partial'"
            )
        if self.strategy_type != "partial" and self.partial_percentage is not None:
            raise ValueError(
                f"partial_percentage must be None when strategy_type is '{self.strategy_type}'"
            )
        return self


def _run_credit_simulation(
    monthly_income: float,
    fixed_expenses: float,
    discretionary_expenses: float,
    emergency_fund: float,
    credit_balance: float,
    credit_apr: float,
    strategy_type: str,
    partial_percentage: Optional[float] = None,
    commitments: Optional[List[CommitmentInput]] = None,
) -> Dict[str, Any]:
    """Execute credit strategy simulation with full guardrails."""
    if commitments is None:
        commitments = []

    try:
        commitment_models = [
            Commitment(id=c.id, amount=c.amount, due_month=c.due_month)
            for c in commitments
        ]
        state = FinancialState(
            monthly_income=monthly_income,
            fixed_expenses=fixed_expenses,
            discretionary_expenses=discretionary_expenses,
            emergency_fund=emergency_fund,
            credit_balance=credit_balance,
            credit_apr=credit_apr,
            commitments=commitment_models,
        )
    except (ValidationError, ValueError) as exc:
        return _failure("INVALID_INPUT", f"Financial state validation failed: {exc}")

    try:
        strategy = CreditStrategyRequest(
            strategy_type=strategy_type,
            partial_percentage=partial_percentage,
        )
    except (ValidationError, ValueError) as exc:
        return _failure("INVALID_INPUT", f"Credit strategy validation failed: {exc}")

    try:
        result_set = simulate_credit_strategies(state, strategy)
    except Exception as exc:
        return _failure("SIMULATION_FAILURE", f"Simulation engine error: {exc}")

    try:
        output = result_set.model_dump()
    except Exception as exc:
        return _failure("VALIDATION_ERROR", f"Output serialization failed: {exc}")

    return _success(output)


simulate_credit_strategy_tool = StructuredTool.from_function(
    func=_run_credit_simulation,
    name="simulate_credit_strategies",
    description=(
        "Evaluate credit payment strategies (minimum, partial, full) against "
        "the user's financial state.  Returns a ranked list of SimulationResults "
        "sorted by stability_score (descending).  For 'partial' strategy, "
        "partial_percentage (0–1) is required.  All monetary inputs must be "
        "monthly floats.  APR as decimal."
    ),
    args_schema=CreditStrategyToolInput,
    return_direct=False,
    handle_tool_error=_handle_tool_error,
    handle_validation_error=_handle_validation_error,
)


# ═══════════════════════════════════════════════════════════════════════
# 3) run_stress_test_tool
# ═══════════════════════════════════════════════════════════════════════

class StressTestToolInput(BaseModel):
    """Input schema for run_stress_test_tool.

    Accepts the user's financial state and runs baseline, income-shock
    (-20%), and expense-shock (+30%) scenarios.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    monthly_income: float
    fixed_expenses: float
    discretionary_expenses: float
    emergency_fund: float
    credit_balance: float
    credit_apr: float
    commitments: List[CommitmentInput] = []

    @field_validator("monthly_income", "fixed_expenses", "discretionary_expenses",
                     "emergency_fund", "credit_balance")
    @classmethod
    def non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"Value must be >= 0 (got {v})")
        return v

    @field_validator("credit_apr")
    @classmethod
    def apr_range(cls, v: float) -> float:
        if v < 0 or v > 1:
            raise ValueError(f"credit_apr must be 0–1 (got {v})")
        return v


def _baseline_simulation(state: FinancialState) -> SimulationResult:
    """Default simulation function: base interest, no extra cost."""
    interest = _monthly_interest(state.credit_balance, state.credit_apr)
    return _compute_result("stress-test", state, 0.0, interest)


def _run_stress_test(
    monthly_income: float,
    fixed_expenses: float,
    discretionary_expenses: float,
    emergency_fund: float,
    credit_balance: float,
    credit_apr: float,
    commitments: Optional[List[CommitmentInput]] = None,
) -> Dict[str, Any]:
    """Execute stress test with full guardrails."""
    if commitments is None:
        commitments = []

    try:
        commitment_models = [
            Commitment(id=c.id, amount=c.amount, due_month=c.due_month)
            for c in commitments
        ]
        state = FinancialState(
            monthly_income=monthly_income,
            fixed_expenses=fixed_expenses,
            discretionary_expenses=discretionary_expenses,
            emergency_fund=emergency_fund,
            credit_balance=credit_balance,
            credit_apr=credit_apr,
            commitments=commitment_models,
        )
    except (ValidationError, ValueError) as exc:
        return _failure("INVALID_INPUT", f"Financial state validation failed: {exc}")

    try:
        result = run_stress_test(state, _baseline_simulation)
    except Exception as exc:
        return _failure("SIMULATION_FAILURE", f"Stress test engine error: {exc}")

    try:
        output = result.model_dump()
    except Exception as exc:
        return _failure("VALIDATION_ERROR", f"Output serialization failed: {exc}")

    return _success(output)


run_stress_test_tool = StructuredTool.from_function(
    func=_run_stress_test,
    name="run_stress_test",
    description=(
        "Run deterministic stress tests on the user's financial state.  "
        "Evaluates three scenarios: baseline (unchanged), income shock "
        "(monthly income reduced by 20%), and expense shock (fixed and "
        "discretionary expenses increased by 30%).  Returns fragility "
        "index and risk level (low/moderate/high).  All monetary inputs "
        "must be monthly floats.  APR as decimal."
    ),
    args_schema=StressTestToolInput,
    return_direct=False,
    handle_tool_error=_handle_tool_error,
    handle_validation_error=_handle_validation_error,
)


# ═══════════════════════════════════════════════════════════════════════
# Tool Registry
# ═══════════════════════════════════════════════════════════════════════

ALL_TOOLS = [
    simulate_subscription_tool,
    simulate_credit_strategy_tool,
    run_stress_test_tool,
]
"""List of all registered LangChain tools for agent binding."""


# ═══════════════════════════════════════════════════════════════════════
# Example Usage
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json

    # ── 1. Valid subscription simulation ──────────────────────────────

    print("=" * 60)
    print("TOOL 1: simulate_subscription_options")
    print("=" * 60)

    sub_result = simulate_subscription_tool.invoke({
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
    print(json.dumps(sub_result, indent=2))

    # ── 2. Valid credit strategy simulation ───────────────────────────

    print("\n" + "=" * 60)
    print("TOOL 2: simulate_credit_strategies")
    print("=" * 60)

    credit_result = simulate_credit_strategy_tool.invoke({
        "monthly_income": 5000.0,
        "fixed_expenses": 2000.0,
        "discretionary_expenses": 800.0,
        "emergency_fund": 10000.0,
        "credit_balance": 3000.0,
        "credit_apr": 0.24,
        "strategy_type": "partial",
        "partial_percentage": 0.25,
    })
    print(json.dumps(credit_result, indent=2))

    # ── 3. Valid stress test ──────────────────────────────────────────

    print("\n" + "=" * 60)
    print("TOOL 3: run_stress_test")
    print("=" * 60)

    stress_result = run_stress_test_tool.invoke({
        "monthly_income": 5000.0,
        "fixed_expenses": 2000.0,
        "discretionary_expenses": 800.0,
        "emergency_fund": 10000.0,
        "credit_balance": 3000.0,
        "credit_apr": 0.24,
    })
    print(json.dumps(stress_result, indent=2))

    # ── 4. Invalid input (negative income) → structured error ────────
    #    LangChain schema validation catches this BEFORE reaching the
    #    function body.  handle_validation_error returns structured JSON.

    print("\n" + "=" * 60)
    print("INVALID INPUT: Negative income (schema-level catch)")
    print("=" * 60)

    bad_result = run_stress_test_tool.invoke({
        "monthly_income": -1000.0,
        "fixed_expenses": 500.0,
        "discretionary_expenses": 200.0,
        "emergency_fund": 0.0,
        "credit_balance": 0.0,
        "credit_apr": 0.0,
    })
    print(bad_result)  # JSON string from handle_validation_error

    # ── 5. Invalid input (function-level catch) ──────────────────────
    #    Calling the wrapper directly bypasses LangChain schema validation,
    #    so the guardrail inside the function body catches it.

    print("\n" + "=" * 60)
    print("INVALID INPUT: Function-level guardrail")
    print("=" * 60)

    direct_result = _run_stress_test(
        monthly_income=-500.0,
        fixed_expenses=100.0,
        discretionary_expenses=50.0,
        emergency_fund=0.0,
        credit_balance=0.0,
        credit_apr=0.0,
    )
    print(json.dumps(direct_result, indent=2))

    # ── 6. Simulated runtime failure (commented) ─────────────────────
    #
    # If the simulation engine raised an unexpected RuntimeError:
    #
    # def broken_simulation(state):
    #     raise RuntimeError("Engine crashed unexpectedly")
    #
    # _run_stress_test would catch it and return:
    # {
    #   "status": "error",
    #   "error_code": "SIMULATION_FAILURE",
    #   "message": "Stress test engine error: Engine crashed unexpectedly"
    # }
