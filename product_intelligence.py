"""
Product Intelligence — Financial Simulation System (Phase 10)

Production-grade analytics and extensibility layer:
  1. Sensitivity analysis — how stability score responds to input shifts
  2. Extended financial state — variable income, multi-credit, investment hooks
  3. Future expansion scaffolding — backward-compatible schema evolution

Guarantees:
  - Deterministic sensitivity analysis (no randomness, no LLM)
  - Zero mutation of baseline state (deep copy per scenario)
  - Backward-compatible schema extension
  - No scoring formula changes
  - Clean architectural separation

No simulation logic modification.  No risk formula changes.
"""

from __future__ import annotations

import copy
import json
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from financial_schemas import Commitment, FinancialState
from simulation_engine import _compute_result, _monthly_interest


# ═══════════════════════════════════════════════════════════════════════
#  Section 1 — Sensitivity Analysis
# ═══════════════════════════════════════════════════════════════════════

# ── Scenario Definitions ─────────────────────────────────────────────

SENSITIVITY_SCENARIOS = [
    {
        "id": "income_plus_10",
        "label": "Income +10%",
        "factor": "income",
        "apply": lambda s: _mutate(s, monthly_income=round(s.monthly_income * 1.10, 2)),
    },
    {
        "id": "income_minus_10",
        "label": "Income -10%",
        "factor": "income",
        "apply": lambda s: _mutate(s, monthly_income=round(s.monthly_income * 0.90, 2)),
    },
    {
        "id": "expenses_plus_10",
        "label": "Expenses +10%",
        "factor": "expenses",
        "apply": lambda s: _mutate(
            s,
            fixed_expenses=round(s.fixed_expenses * 1.10, 2),
            discretionary_expenses=round(s.discretionary_expenses * 1.10, 2),
        ),
    },
    {
        "id": "expenses_minus_10",
        "label": "Expenses -10%",
        "factor": "expenses",
        "apply": lambda s: _mutate(
            s,
            fixed_expenses=round(s.fixed_expenses * 0.90, 2),
            discretionary_expenses=round(s.discretionary_expenses * 0.90, 2),
        ),
    },
    {
        "id": "apr_plus_5pp",
        "label": "APR +5 percentage points",
        "factor": "apr",
        "apply": lambda s: _mutate(
            s,
            credit_apr=min(round(s.credit_apr + 0.05, 4), 1.0),
        ),
    },
]


def _mutate(state: FinancialState, **overrides: Any) -> FinancialState:
    """Create a deep-copy of a FinancialState with specified field overrides.

    Never mutates the original state.
    """
    data = state.model_dump()
    data.update(overrides)
    return FinancialState(**data)


def _baseline_score(state: FinancialState) -> Dict[str, Any]:
    """Compute baseline simulation result for a financial state."""
    interest = _monthly_interest(state.credit_balance, state.credit_apr)
    result = _compute_result("baseline", state, 0.0, interest)
    return {
        "stability_score": result.stability_score,
        "risk_flags": result.risk_flags,
        "liquidity_ratio": result.liquidity_ratio,
        "commitment_coverage": result.commitment_coverage,
        "interest_cost": result.interest_cost,
        "projected_balance": result.projected_balance,
    }


def run_sensitivity_analysis(
    state: FinancialState,
    correlation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute deterministic sensitivity analysis across all scenarios.

    For a given validated FinancialState:
      1. Compute baseline stability score
      2. For each scenario, deep-copy state, apply mutation, recompute
      3. Determine most sensitive factor (highest |delta|)

    Returns structured response with baseline, scenarios, and most
    sensitive factor.

    Never mutates the original state.  Fully deterministic.
    """
    cid = correlation_id or str(uuid.uuid4())

    # Compute baseline
    baseline = _baseline_score(state)
    baseline_score = baseline["stability_score"]

    # Run each scenario
    scenarios: List[Dict[str, Any]] = []
    max_delta = 0.0
    most_sensitive = "income"

    for scenario_def in SENSITIVITY_SCENARIOS:
        # Deep copy + apply mutation
        mutated_state = scenario_def["apply"](state)

        # Recompute via deterministic simulation
        interest = _monthly_interest(
            mutated_state.credit_balance, mutated_state.credit_apr,
        )
        result = _compute_result(
            scenario_def["id"], mutated_state, 0.0, interest,
        )

        delta = round(result.stability_score - baseline_score, 2)

        scenarios.append({
            "scenario": scenario_def["id"],
            "label": scenario_def["label"],
            "factor": scenario_def["factor"],
            "new_score": result.stability_score,
            "delta": delta,
            "risk_flags": result.risk_flags,
        })

        # Track most sensitive factor
        if abs(delta) > max_delta:
            max_delta = abs(delta)
            most_sensitive = scenario_def["factor"]

    return {
        "status": "success",
        "correlation_id": cid,
        "baseline": baseline,
        "scenarios": scenarios,
        "most_sensitive_factor": most_sensitive,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Section 2A — Variable Income Support (Expansion Hook)
# ═══════════════════════════════════════════════════════════════════════

class IncomeStream(BaseModel):
    """A single income source with configurable frequency."""

    model_config = ConfigDict(extra="forbid", strict=True)

    name: str
    amount: float
    frequency: str  # "monthly" | "quarterly" | "yearly"

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Income stream name must be non-empty")
        return v

    @field_validator("amount")
    @classmethod
    def amount_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Income stream amount must be > 0 (got {v})")
        return v

    @field_validator("frequency")
    @classmethod
    def valid_frequency(cls, v: str) -> str:
        allowed = {"monthly", "quarterly", "yearly"}
        if v not in allowed:
            raise ValueError(
                f"frequency must be one of {allowed} (got '{v}')"
            )
        return v


def normalize_income_to_monthly(streams: List[IncomeStream]) -> float:
    """Normalize multiple income streams to a single monthly total.

    Conversion:
      - monthly: amount × 1
      - quarterly: amount / 3
      - yearly: amount / 12
    """
    multipliers = {"monthly": 1.0, "quarterly": 1.0 / 3.0, "yearly": 1.0 / 12.0}
    total = sum(
        round(stream.amount * multipliers[stream.frequency], 2)
        for stream in streams
    )
    return round(total, 2)


# ═══════════════════════════════════════════════════════════════════════
#  Section 2B — Multiple Credit Cards (Expansion Hook)
# ═══════════════════════════════════════════════════════════════════════

class CreditAccount(BaseModel):
    """A single credit card / revolving credit account."""

    model_config = ConfigDict(extra="forbid", strict=True)

    balance: float
    apr: float
    minimum_payment: float

    @field_validator("balance")
    @classmethod
    def balance_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"balance must be >= 0 (got {v})")
        return v

    @field_validator("apr")
    @classmethod
    def apr_valid(cls, v: float) -> float:
        if v < 0 or v > 1:
            raise ValueError(f"apr must be 0–1 (got {v})")
        return v

    @field_validator("minimum_payment")
    @classmethod
    def min_payment_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"minimum_payment must be >= 0 (got {v})")
        return v


def aggregate_credit_accounts(accounts: List[CreditAccount]) -> Dict[str, float]:
    """Aggregate multiple credit accounts into a single view.

    Returns:
      - total_balance: sum of all balances
      - weighted_apr: balance-weighted average APR
      - total_minimum_payment: sum of all minimum payments

    Maintains deterministic calculation.
    """
    total_balance = sum(a.balance for a in accounts)
    total_minimum = sum(a.minimum_payment for a in accounts)

    # Weighted APR: Σ(balance_i × apr_i) / Σ(balance_i)
    if total_balance > 0:
        weighted_apr = sum(a.balance * a.apr for a in accounts) / total_balance
    else:
        weighted_apr = 0.0

    return {
        "total_balance": round(total_balance, 2),
        "weighted_apr": round(weighted_apr, 4),
        "total_minimum_payment": round(total_minimum, 2),
    }


# ═══════════════════════════════════════════════════════════════════════
#  Section 2C — Investment Modeling Hook (Placeholder)
# ═══════════════════════════════════════════════════════════════════════

class InvestmentAccount(BaseModel):
    """Placeholder for future investment account modeling."""

    model_config = ConfigDict(extra="forbid", strict=True)

    current_value: float
    expected_return_rate: float
    volatility: float

    @field_validator("current_value")
    @classmethod
    def value_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"current_value must be >= 0 (got {v})")
        return v

    @field_validator("expected_return_rate")
    @classmethod
    def return_rate_valid(cls, v: float) -> float:
        if v < -1 or v > 1:
            raise ValueError(f"expected_return_rate must be -1 to 1 (got {v})")
        return v

    @field_validator("volatility")
    @classmethod
    def volatility_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"volatility must be >= 0 (got {v})")
        return v


def compute_investment_projection(
    accounts: List[InvestmentAccount],
    months: int = 12,
) -> Dict[str, Any]:
    """Placeholder for future investment projection logic.

    NOT integrated into stability score.
    Raises NotImplementedError until Phase N implements the projection model.
    """
    raise NotImplementedError(
        "Investment projection is a future expansion. "
        "Accounts are stored and passed through, but projection "
        "logic has not been implemented yet."
    )


# ═══════════════════════════════════════════════════════════════════════
#  Section 3 — Extended Financial State
# ═══════════════════════════════════════════════════════════════════════

CURRENT_SCHEMA_VERSION = "v3.0.0"
PREVIOUS_SCHEMA_VERSION = "v2.0.0"


class ExtendedFinancialState(BaseModel):
    """Backward-compatible extension of FinancialState.

    New optional fields (v3.0.0):
      - income_streams: variable income sources → normalized to monthly_income
      - credit_accounts: multiple credit cards → aggregated view
      - investment_accounts: placeholder for future investment modeling

    Backward compatibility:
      - All new fields are Optional with default None
      - If income_streams provided → overrides monthly_income
      - If credit_accounts provided → overrides credit_balance / credit_apr
      - If not provided → legacy fields used directly
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    # ── Legacy fields (always required) ───────────────────────────────
    monthly_income: float
    fixed_expenses: float
    discretionary_expenses: float
    emergency_fund: float
    credit_balance: float
    credit_apr: float
    commitments: List[Commitment] = []

    # ── v3.0.0 expansion fields (optional) ────────────────────────────
    income_streams: Optional[List[IncomeStream]] = None
    credit_accounts: Optional[List[CreditAccount]] = None
    investment_accounts: Optional[List[InvestmentAccount]] = None

    # Field validators — reuse from FinancialState
    @field_validator("monthly_income")
    @classmethod
    def income_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"monthly_income must be >= 0 (got {v})")
        return v

    @field_validator("fixed_expenses")
    @classmethod
    def fixed_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"fixed_expenses must be >= 0 (got {v})")
        return v

    @field_validator("discretionary_expenses")
    @classmethod
    def disc_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"discretionary_expenses must be >= 0 (got {v})")
        return v

    @field_validator("emergency_fund")
    @classmethod
    def fund_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"emergency_fund must be >= 0 (got {v})")
        return v

    @field_validator("credit_balance")
    @classmethod
    def balance_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"credit_balance must be >= 0 (got {v})")
        return v

    @field_validator("credit_apr")
    @classmethod
    def apr_valid(cls, v: float) -> float:
        if v < 0 or v > 1:
            raise ValueError(f"credit_apr must be 0–1 (got {v})")
        return v

    def to_core_state(self) -> FinancialState:
        """Convert to core FinancialState, applying expansion logic.

        If expansion fields are present:
          - income_streams → normalized monthly income overrides monthly_income
          - credit_accounts → aggregated balance/APR overrides legacy fields

        Returns a standard FinancialState compatible with all existing
        simulation and scoring logic.
        """
        income = self.monthly_income
        balance = self.credit_balance
        apr = self.credit_apr

        # Apply income stream override
        if self.income_streams:
            income = normalize_income_to_monthly(self.income_streams)

        # Apply credit account aggregation
        if self.credit_accounts:
            agg = aggregate_credit_accounts(self.credit_accounts)
            balance = agg["total_balance"]
            apr = agg["weighted_apr"]

        return FinancialState(
            monthly_income=income,
            fixed_expenses=self.fixed_expenses,
            discretionary_expenses=self.discretionary_expenses,
            emergency_fund=self.emergency_fund,
            credit_balance=balance,
            credit_apr=apr,
            commitments=self.commitments,
        )

    @property
    def schema_version(self) -> str:
        """Return the schema version based on which fields are used."""
        if (
            self.income_streams is not None
            or self.credit_accounts is not None
            or self.investment_accounts is not None
        ):
            return CURRENT_SCHEMA_VERSION
        return PREVIOUS_SCHEMA_VERSION


# ═══════════════════════════════════════════════════════════════════════
#  Section 4 — FastAPI Integration
# ═══════════════════════════════════════════════════════════════════════

try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    def register_intelligence_endpoints(app: FastAPI) -> None:
        """Register /sensitivity-analysis endpoint."""

        @app.post("/sensitivity-analysis")
        async def sensitivity_analysis_endpoint(
            request: dict,
        ) -> JSONResponse:
            """Compute sensitivity analysis for a financial state.

            Accepts FinancialState or ExtendedFinancialState input.
            """
            try:
                # Parse input
                input_data = request.get("input", {})
                cid = request.get("correlation_id", str(uuid.uuid4()))

                # Try extended state first, fall back to core
                try:
                    extended = ExtendedFinancialState(**input_data)
                    state = extended.to_core_state()
                except Exception:
                    state = FinancialState(**input_data)

                result = run_sensitivity_analysis(state, correlation_id=cid)
                return JSONResponse(content=result)

            except Exception as exc:
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "error",
                        "error_code": "SENSITIVITY_ANALYSIS_FAILED",
                        "message": str(exc)[:200],
                    },
                )

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
        if isinstance(obj, (dict, list)):
            print(json.dumps(obj, indent=2, default=str))
        else:
            print(str(obj))

    # ══════════════════════════════════════════════════════════════════
    # 1. SENSITIVITY ANALYSIS — standard state
    # ══════════════════════════════════════════════════════════════════

    state = FinancialState(
        monthly_income=5000.0,
        fixed_expenses=2000.0,
        discretionary_expenses=800.0,
        emergency_fund=10000.0,
        credit_balance=3000.0,
        credit_apr=0.24,
        commitments=[
            Commitment(id="insurance", amount=300.0, due_month=3),
        ],
    )

    result = run_sensitivity_analysis(state, correlation_id="sens-001")
    _show("1. SENSITIVITY ANALYSIS", result)

    # ══════════════════════════════════════════════════════════════════
    # 2. IMMUTABILITY — original state unchanged
    # ══════════════════════════════════════════════════════════════════

    assert state.monthly_income == 5000.0
    assert state.fixed_expenses == 2000.0
    assert state.credit_apr == 0.24
    _show("2. ORIGINAL STATE IMMUTABILITY", {
        "monthly_income": state.monthly_income,
        "fixed_expenses": state.fixed_expenses,
        "credit_apr": state.credit_apr,
        "result": "✅ unchanged after sensitivity analysis",
    })

    # ══════════════════════════════════════════════════════════════════
    # 3. VARIABLE INCOME — multiple streams
    # ══════════════════════════════════════════════════════════════════

    streams = [
        IncomeStream(name="Salary", amount=4000.0, frequency="monthly"),
        IncomeStream(name="Freelance", amount=3000.0, frequency="quarterly"),
        IncomeStream(name="Dividends", amount=6000.0, frequency="yearly"),
    ]

    monthly = normalize_income_to_monthly(streams)
    _show("3. VARIABLE INCOME → monthly", {
        "streams": [s.model_dump() for s in streams],
        "normalized_monthly": monthly,
        "calculation": "4000 + (3000/3) + (6000/12) = 4000 + 1000 + 500 = 5500",
    })

    # ══════════════════════════════════════════════════════════════════
    # 4. MULTIPLE CREDIT CARDS — aggregation
    # ══════════════════════════════════════════════════════════════════

    accounts = [
        CreditAccount(balance=2000.0, apr=0.22, minimum_payment=40.0),
        CreditAccount(balance=1500.0, apr=0.18, minimum_payment=30.0),
        CreditAccount(balance=500.0, apr=0.30, minimum_payment=15.0),
    ]

    agg = aggregate_credit_accounts(accounts)
    _show("4. MULTI-CREDIT AGGREGATION", {
        "accounts": [a.model_dump() for a in accounts],
        "aggregated": agg,
        "weighted_apr_calc": "(2000×0.22 + 1500×0.18 + 500×0.30) / 4000",
    })

    # ══════════════════════════════════════════════════════════════════
    # 5. EXTENDED STATE → CORE STATE
    # ══════════════════════════════════════════════════════════════════

    extended = ExtendedFinancialState(
        monthly_income=0.0,  # overridden by income_streams
        fixed_expenses=2000.0,
        discretionary_expenses=800.0,
        emergency_fund=10000.0,
        credit_balance=0.0,  # overridden by credit_accounts
        credit_apr=0.0,      # overridden by credit_accounts
        commitments=[Commitment(id="rent", amount=500.0, due_month=1)],
        income_streams=streams,
        credit_accounts=accounts,
    )

    core = extended.to_core_state()
    _show("5. EXTENDED → CORE STATE", {
        "schema_version": extended.schema_version,
        "monthly_income": core.monthly_income,
        "credit_balance": core.credit_balance,
        "credit_apr": core.credit_apr,
        "note": "Income from streams, credit from aggregated accounts",
    })

    # ══════════════════════════════════════════════════════════════════
    # 6. SENSITIVITY ON EXTENDED STATE
    # ══════════════════════════════════════════════════════════════════

    sens = run_sensitivity_analysis(core, correlation_id="ext-sens-001")
    _show("6. SENSITIVITY ON EXTENDED STATE", {
        "baseline_score": sens["baseline"]["stability_score"],
        "most_sensitive": sens["most_sensitive_factor"],
        "scenarios": [
            {"id": s["scenario"], "delta": s["delta"]}
            for s in sens["scenarios"]
        ],
    })

    # ══════════════════════════════════════════════════════════════════
    # 7. BACKWARD COMPATIBILITY — legacy state
    # ══════════════════════════════════════════════════════════════════

    legacy = ExtendedFinancialState(
        monthly_income=5000.0,
        fixed_expenses=2000.0,
        discretionary_expenses=800.0,
        emergency_fund=10000.0,
        credit_balance=3000.0,
        credit_apr=0.24,
    )
    legacy_core = legacy.to_core_state()
    _show("7. BACKWARD COMPATIBILITY", {
        "schema_version": legacy.schema_version,
        "monthly_income": legacy_core.monthly_income,
        "credit_balance": legacy_core.credit_balance,
        "credit_apr": legacy_core.credit_apr,
        "note": "No expansion fields → legacy values used directly",
    })

    # ══════════════════════════════════════════════════════════════════
    # 8. INVESTMENT HOOK — placeholder
    # ══════════════════════════════════════════════════════════════════

    inv = InvestmentAccount(
        current_value=50000.0,
        expected_return_rate=0.08,
        volatility=0.15,
    )
    try:
        compute_investment_projection([inv])
    except NotImplementedError as exc:
        _show("8. INVESTMENT HOOK (NotImplementedError)", {
            "account": inv.model_dump(),
            "error": str(exc),
        })

    # ══════════════════════════════════════════════════════════════════
    # 9. SCHEMA VERSIONING
    # ══════════════════════════════════════════════════════════════════

    _show("9. SCHEMA VERSIONING", {
        "legacy_state_version": legacy.schema_version,
        "extended_state_version": extended.schema_version,
        "current_version": CURRENT_SCHEMA_VERSION,
        "previous_version": PREVIOUS_SCHEMA_VERSION,
    })

    print(f"\n{'═' * 64}")
    print(f"  FastAPI available: {_FASTAPI_AVAILABLE}")
    print(f"{'═' * 64}")
