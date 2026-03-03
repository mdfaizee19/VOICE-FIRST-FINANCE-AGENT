"""
Stress Testing Module — Deterministic Financial Stress Scenarios

Implements additive stress testing on top of the existing simulation engine.
Does NOT modify existing schemas or simulation logic.

Stress Scenarios
----------------
- **Income Shock** (``income_drop``): Reduce monthly_income by 20%.
- **Expense Shock** (``expense_spike``): Increase both fixed_expenses and
  discretionary_expenses by 30%.

Output
------
- ``StressScenarioResult``: Per-scenario metrics snapshot.
- ``StressTestResult``: Aggregated result with fragility index and risk level.
- ``run_stress_test(financial_state, simulation_function)``: Entry point.

All math is pure and deterministic.  No randomness, no time dependency,
no external services, no mutation of inputs.
"""

from __future__ import annotations

import copy
from typing import Callable, List, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from financial_schemas import FinancialState
from simulation_schemas import SimulationResult


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class StressScenarioResult(BaseModel):
    """Metrics snapshot for a single stress scenario.

    Captures the key financial indicators produced by running the
    simulation engine against a (possibly mutated) FinancialState.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    scenario_name: Literal["baseline", "income_drop", "expense_spike"]
    stability_score: float
    projected_balance: float
    liquidity_ratio: float
    commitment_coverage: float
    interest_cost: float

    @field_validator("stability_score")
    @classmethod
    def stability_score_range(cls, v: float) -> float:
        if v < 0 or v > 100:
            raise ValueError(
                f"stability_score must be between 0 and 100 inclusive (got {v})"
            )
        return v

    @field_validator("liquidity_ratio")
    @classmethod
    def liquidity_ratio_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(
                f"liquidity_ratio must be >= 0 (got {v})"
            )
        return v

    @field_validator("commitment_coverage")
    @classmethod
    def commitment_coverage_range(cls, v: float) -> float:
        if v < 0 or v > 1:
            raise ValueError(
                f"commitment_coverage must be between 0 and 1 inclusive (got {v})"
            )
        return v

    @field_validator("interest_cost")
    @classmethod
    def interest_cost_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(
                f"interest_cost must be >= 0 (got {v})"
            )
        return v


class StressTestResult(BaseModel):
    """Aggregated stress-test output with fragility index and risk level.

    The *fragility_index* measures how much worse the stability score
    becomes under stress.  Higher values indicate greater vulnerability.

    Risk levels:
    - ``low``:      fragility_index < 5
    - ``moderate``: 5 ≤ fragility_index < 15
    - ``high``:     fragility_index ≥ 15, **or** baseline_score < 20
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    baseline_score: float
    income_drop_score: float
    expense_spike_score: float
    fragility_index: float
    risk_level: Literal["low", "moderate", "high"]
    scenarios: List[StressScenarioResult]

    @field_validator("fragility_index")
    @classmethod
    def fragility_index_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(
                f"fragility_index must be >= 0 (got {v}); "
                "stress scenarios cannot improve the score"
            )
        return v

    @model_validator(mode="after")
    def validate_stress_scores(self) -> StressTestResult:
        """Stress scores must not exceed baseline (stress only worsens outcomes)."""
        if self.income_drop_score > self.baseline_score:
            raise ValueError(
                f"income_drop_score ({self.income_drop_score}) exceeds "
                f"baseline_score ({self.baseline_score}); "
                "stress scenario cannot improve financial stability"
            )
        if self.expense_spike_score > self.baseline_score:
            raise ValueError(
                f"expense_spike_score ({self.expense_spike_score}) exceeds "
                f"baseline_score ({self.baseline_score}); "
                "stress scenario cannot improve financial stability"
            )
        return self


# ---------------------------------------------------------------------------
# Type alias for the simulation callable
# ---------------------------------------------------------------------------

SimulationFunction = Callable[[FinancialState], SimulationResult]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_income_shock(state: FinancialState) -> FinancialState:
    """Create a deep copy with monthly_income reduced by 20%."""
    data = copy.deepcopy(state.model_dump())
    data["monthly_income"] = round(data["monthly_income"] * 0.80, 2)
    return FinancialState(**data)


def _apply_expense_shock(state: FinancialState) -> FinancialState:
    """Create a deep copy with fixed and discretionary expenses increased by 30%."""
    data = copy.deepcopy(state.model_dump())
    data["fixed_expenses"] = round(data["fixed_expenses"] * 1.30, 2)
    data["discretionary_expenses"] = round(data["discretionary_expenses"] * 1.30, 2)
    return FinancialState(**data)


def _extract_scenario(
    scenario_name: Literal["baseline", "income_drop", "expense_spike"],
    result: SimulationResult,
) -> StressScenarioResult:
    """Extract key metrics from a SimulationResult into a StressScenarioResult."""
    return StressScenarioResult(
        scenario_name=scenario_name,
        stability_score=round(min(max(result.stability_score, 0), 100), 2),
        projected_balance=round(result.projected_balance, 2),
        liquidity_ratio=round(max(result.liquidity_ratio, 0), 2),
        commitment_coverage=round(min(max(result.commitment_coverage, 0), 1.0), 2),
        interest_cost=round(result.interest_cost, 2),
    )


def _compute_fragility(baseline: float, income_drop: float, expense_spike: float) -> float:
    """Compute the fragility index (average score drop across stress scenarios)."""
    return round(
        ((baseline - income_drop) + (baseline - expense_spike)) / 2.0,
        2,
    )


def _determine_risk_level(
    fragility_index: float,
    baseline_score: float,
) -> Literal["low", "moderate", "high"]:
    """Map fragility index to a risk level, with baseline override."""
    if baseline_score < 20:
        return "high"
    if fragility_index >= 15:
        return "high"
    if fragility_index >= 5:
        return "moderate"
    return "low"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_stress_test(
    financial_state: FinancialState,
    simulation_function: SimulationFunction,
) -> StressTestResult:
    """Run deterministic stress tests against a financial state.

    Executes the provided *simulation_function* under three conditions:

    1. **Baseline** — original financial state (unchanged).
    2. **Income Shock** — monthly_income reduced by 20%.
    3. **Expense Shock** — fixed and discretionary expenses increased by 30%.

    The original *financial_state* is **never mutated**.

    Parameters
    ----------
    financial_state : FinancialState
        The user's current financial snapshot.
    simulation_function : Callable[[FinancialState], SimulationResult]
        A pure, deterministic function that produces a SimulationResult
        from a FinancialState.

    Returns
    -------
    StressTestResult
        Aggregated stress-test results including fragility index and
        risk level classification.

    Raises
    ------
    ValueError
        If a stress scenario produces a higher stability score than the
        baseline (indicates invalid model logic).
    """
    # ── Run simulations (original state is never modified) ────────────
    baseline_result = simulation_function(financial_state)
    income_drop_result = simulation_function(_apply_income_shock(financial_state))
    expense_spike_result = simulation_function(_apply_expense_shock(financial_state))

    # ── Extract scenario snapshots ───────────────────────────────────
    baseline_scenario = _extract_scenario("baseline", baseline_result)
    income_scenario = _extract_scenario("income_drop", income_drop_result)
    expense_scenario = _extract_scenario("expense_spike", expense_spike_result)

    b_score = baseline_scenario.stability_score
    i_score = income_scenario.stability_score
    e_score = expense_scenario.stability_score

    # ── Compute fragility and risk ───────────────────────────────────
    fragility = _compute_fragility(b_score, i_score, e_score)
    risk = _determine_risk_level(fragility, b_score)

    return StressTestResult(
        baseline_score=b_score,
        income_drop_score=i_score,
        expense_spike_score=e_score,
        fragility_index=fragility,
        risk_level=risk,
        scenarios=[baseline_scenario, income_scenario, expense_scenario],
    )


# ---------------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from financial_schemas import Commitment
    from simulation_engine import _compute_result, _monthly_interest

    # Wrap the engine's internal function into the expected signature
    def simple_simulation(state: FinancialState) -> SimulationResult:
        """Simulate with no extra cost, just base interest."""
        interest = _monthly_interest(state.credit_balance, state.credit_apr)
        return _compute_result("stress-baseline", state, 0.0, interest)

    # ── Valid stress test ─────────────────────────────────────────────

    state = FinancialState(
        monthly_income=5000.0,
        fixed_expenses=2000.0,
        discretionary_expenses=800.0,
        emergency_fund=10000.0,
        credit_balance=3000.0,
        credit_apr=0.24,
        commitments=[Commitment(id="ins", amount=300.0, due_month=3)],
    )

    result = run_stress_test(state, simple_simulation)
    print("✓ Stress Test Result:")
    print(f"  Baseline score:     {result.baseline_score}")
    print(f"  Income drop score:  {result.income_drop_score}")
    print(f"  Expense spike score:{result.expense_spike_score}")
    print(f"  Fragility index:    {result.fragility_index}")
    print(f"  Risk level:         {result.risk_level}")
    print()
    for s in result.scenarios:
        print(f"  [{s.scenario_name}] score={s.stability_score} "
              f"bal={s.projected_balance} liq={s.liquidity_ratio} "
              f"cov={s.commitment_coverage} int={s.interest_cost}")

    # ── Verify original state was not mutated ─────────────────────────
    assert state.monthly_income == 5000.0, "Original state was mutated!"
    assert state.fixed_expenses == 2000.0, "Original state was mutated!"
    print("\n✓ Original FinancialState was not mutated")

    # ── FAIL: Stress score exceeding baseline (commented) ─────────────
    # This would raise ValueError because stress cannot improve scores:
    #
    # from pydantic import ValidationError
    # try:
    #     StressTestResult(
    #         baseline_score=50.0,
    #         income_drop_score=60.0,   # ← higher than baseline!
    #         expense_spike_score=45.0,
    #         fragility_index=-5.0,     # ← negative = invalid
    #         risk_level="low",
    #         scenarios=[],
    #     )
    # except ValidationError as e:
    #     print("\n✗ FAIL — Stress score exceeds baseline:", e)
