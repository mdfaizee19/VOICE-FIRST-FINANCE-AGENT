"""
Financial Domain Schemas — Pydantic v2 Strict Input Models

Production-grade validated schemas for a financial simulation system.
All monetary values are floats expressed in MONTHLY units.
APR is expressed as a decimal (e.g., 0.36 = 36%).

Models: Commitment, SubscriptionOption, CreditStrategyRequest, FinancialState
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


# ---------------------------------------------------------------------------
# Commitment
# ---------------------------------------------------------------------------

class Commitment(BaseModel):
    """A known future financial obligation due in a specific month.

    Represents a one-off or recurring payment the user is committed to,
    such as an insurance premium, tuition instalment, or annual subscription
    renewal.  The *due_month* indicates the calendar month (1–12) in which
    the payment falls.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    id: str
    amount: float
    due_month: int

    @field_validator("id")
    @classmethod
    def id_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError(
                "Commitment id must be a non-empty string; "
                "provide a unique identifier for this obligation"
            )
        return v

    @field_validator("amount")
    @classmethod
    def amount_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(
                f"Commitment amount must be > 0 (got {v}); "
                "a financial commitment cannot be zero or negative"
            )
        return v

    @field_validator("due_month")
    @classmethod
    def due_month_must_be_valid(cls, v: int) -> int:
        if v < 1 or v > 12:
            raise ValueError(
                f"due_month must be between 1 and 12 inclusive (got {v}); "
                "use calendar month numbers"
            )
        return v


# ---------------------------------------------------------------------------
# SubscriptionOption
# ---------------------------------------------------------------------------

class SubscriptionOption(BaseModel):
    """A recurring subscription the user may adopt or evaluate.

    Examples include streaming services, SaaS tools, gym memberships, etc.
    The *monthly_cost* must reflect the per-month charge (convert annual
    prices before input).
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    name: str
    monthly_cost: float

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError(
                "Subscription name must be a non-empty string; "
                "provide a descriptive label for the subscription"
            )
        return v

    @field_validator("monthly_cost")
    @classmethod
    def monthly_cost_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(
                f"monthly_cost must be > 0 (got {v}); "
                "a subscription with zero or negative cost is invalid"
            )
        return v


# ---------------------------------------------------------------------------
# CreditStrategyRequest
# ---------------------------------------------------------------------------

class CreditStrategyRequest(BaseModel):
    """Describes the user's chosen credit-payment strategy.

    - **minimum**: pay only the minimum amount due each month.
    - **partial**: pay a fixed percentage of the outstanding balance
      (requires *partial_percentage* between 0 and 1).
    - **full**: pay the entire outstanding balance each month.

    The model enforces logical consistency: *partial_percentage* must be
    provided **only** when *strategy_type* is ``"partial"``, and must be
    ``None`` otherwise.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    strategy_type: Literal["minimum", "partial", "full"]
    partial_percentage: Optional[float] = None

    @field_validator("partial_percentage")
    @classmethod
    def partial_percentage_range(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and (v < 0 or v > 1):
            raise ValueError(
                f"partial_percentage must be between 0 and 1 (got {v}); "
                "express the percentage as a decimal fraction"
            )
        return v

    @model_validator(mode="after")
    def check_strategy_consistency(self) -> CreditStrategyRequest:
        if self.strategy_type == "partial":
            if self.partial_percentage is None:
                raise ValueError(
                    "partial_percentage is required when strategy_type is 'partial'; "
                    "specify the fraction of the balance to pay each month"
                )
        else:
            if self.partial_percentage is not None:
                raise ValueError(
                    f"partial_percentage must be None when strategy_type is "
                    f"'{self.strategy_type}'; it is only valid for the 'partial' strategy"
                )
        return self


# ---------------------------------------------------------------------------
# FinancialState
# ---------------------------------------------------------------------------

class FinancialState(BaseModel):
    """Complete snapshot of a user's monthly financial position.

    All monetary fields are expressed in **monthly** units.
    *credit_apr* is the annualised percentage rate as a decimal
    (e.g., 0.36 for 36% APR).

    A sanity check ensures total expenses (fixed + discretionary) do not
    exceed 10× monthly income, guarding against data-entry errors.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    monthly_income: float
    fixed_expenses: float
    discretionary_expenses: float
    emergency_fund: float
    credit_balance: float
    credit_apr: float
    commitments: List[Commitment] = []

    @field_validator("monthly_income")
    @classmethod
    def income_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(
                f"monthly_income must be >= 0 (got {v}); "
                "income cannot be negative"
            )
        return v

    @field_validator("fixed_expenses")
    @classmethod
    def fixed_expenses_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(
                f"fixed_expenses must be >= 0 (got {v}); "
                "expenses cannot be negative"
            )
        return v

    @field_validator("discretionary_expenses")
    @classmethod
    def discretionary_expenses_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(
                f"discretionary_expenses must be >= 0 (got {v}); "
                "expenses cannot be negative"
            )
        return v

    @field_validator("emergency_fund")
    @classmethod
    def emergency_fund_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(
                f"emergency_fund must be >= 0 (got {v}); "
                "an emergency fund balance cannot be negative"
            )
        return v

    @field_validator("credit_balance")
    @classmethod
    def credit_balance_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(
                f"credit_balance must be >= 0 (got {v}); "
                "outstanding credit balance cannot be negative"
            )
        return v

    @field_validator("credit_apr")
    @classmethod
    def credit_apr_valid_range(cls, v: float) -> float:
        if v < 0 or v > 1:
            raise ValueError(
                f"credit_apr must be between 0 and 1 inclusive (got {v}); "
                "express APR as a decimal (e.g., 0.36 for 36%)"
            )
        return v

    @model_validator(mode="after")
    def total_expenses_sanity_check(self) -> FinancialState:
        total_expenses = self.fixed_expenses + self.discretionary_expenses
        if self.monthly_income > 0 and total_expenses > 10 * self.monthly_income:
            raise ValueError(
                f"Total expenses ({total_expenses}) exceed 10× monthly income "
                f"({self.monthly_income}); this likely indicates a data-entry error"
            )
        return self


# ---------------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pydantic import ValidationError

    # ── Valid Instantiations ──────────────────────────────────────────────

    commitment_a = Commitment(id="insurance-q2", amount=450.0, due_month=6)
    commitment_b = Commitment(id="tuition-fall", amount=1200.0, due_month=9)
    print("✓ Valid commitments:", commitment_a, commitment_b)

    sub = SubscriptionOption(name="Cloud Storage Pro", monthly_cost=12.99)
    print("✓ Valid subscription:", sub)

    strategy_full = CreditStrategyRequest(strategy_type="full")
    strategy_partial = CreditStrategyRequest(
        strategy_type="partial", partial_percentage=0.5
    )
    print("✓ Valid strategies:", strategy_full, strategy_partial)

    state = FinancialState(
        monthly_income=5000.0,
        fixed_expenses=2000.0,
        discretionary_expenses=800.0,
        emergency_fund=10000.0,
        credit_balance=3500.0,
        credit_apr=0.22,
        commitments=[commitment_a, commitment_b],
    )
    print("✓ Valid FinancialState:", state.model_dump_json(indent=2))

    # ── Failing Validations ──────────────────────────────────────────────

    # FAIL 1: Commitment with empty id
    try:
        Commitment(id="  ", amount=100.0, due_month=3)
    except ValidationError as e:
        print("\n✗ FAIL 1 — Empty commitment id:")
        print(e)

    # FAIL 2: CreditStrategyRequest "partial" without partial_percentage
    try:
        CreditStrategyRequest(strategy_type="partial")
    except ValidationError as e:
        print("\n✗ FAIL 2 — Partial strategy missing percentage:")
        print(e)

    # FAIL 3: FinancialState with negative income
    try:
        FinancialState(
            monthly_income=-1000.0,
            fixed_expenses=500.0,
            discretionary_expenses=200.0,
            emergency_fund=0.0,
            credit_balance=0.0,
            credit_apr=0.15,
        )
    except ValidationError as e:
        print("\n✗ FAIL 3 — Negative monthly income:")
        print(e)

    # FAIL 4: Extra field rejected (strict mode)
    # try:
    #     SubscriptionOption(name="Gym", monthly_cost=30.0, tier="premium")
    # except ValidationError as e:
    #     print("\n✗ FAIL 4 — Extra field rejected:", e)

    # FAIL 5: APR out of range
    # try:
    #     FinancialState(
    #         monthly_income=4000.0, fixed_expenses=1000.0,
    #         discretionary_expenses=500.0, emergency_fund=2000.0,
    #         credit_balance=1000.0, credit_apr=1.5,
    #     )
    # except ValidationError as e:
    #     print("\n✗ FAIL 5 — APR > 1:", e)

    # FAIL 6: Expenses exceed 10x income (sanity check)
    # try:
    #     FinancialState(
    #         monthly_income=1000.0, fixed_expenses=8000.0,
    #         discretionary_expenses=5000.0, emergency_fund=0.0,
    #         credit_balance=0.0, credit_apr=0.0,
    #     )
    # except ValidationError as e:
    #     print("\n✗ FAIL 6 — Expenses exceed 10x income:", e)
