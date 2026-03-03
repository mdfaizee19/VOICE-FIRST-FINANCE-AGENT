"""
Deterministic Test Suite — Stress Testing Module

Verifies mathematical correctness of:
  - run_stress_test()
  - StressScenarioResult / StressTestResult models
  - Fragility index computation
  - Risk level classification
  - Input immutability (no mutation of original state)

All tests use fixed numeric inputs, deterministic rounding to 2 decimals,
and pytest.approx(abs=1e-2) for floating-point comparisons.
"""

from __future__ import annotations

import copy
import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from financial_schemas import Commitment, FinancialState
from simulation_engine import _compute_result, _monthly_interest
from simulation_schemas import SimulationResult
from stress_testing import (
    StressScenarioResult,
    StressTestResult,
    run_stress_test,
)


# ---------------------------------------------------------------------------
# Shared simulation wrapper (no extra cost, just base interest)
# ---------------------------------------------------------------------------

def baseline_simulation(state: FinancialState) -> SimulationResult:
    """Simple simulation: base interest only, no extra cost."""
    interest = _monthly_interest(state.credit_balance, state.credit_apr)
    return _compute_result("stress-test", state, 0.0, interest)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def standard_state() -> FinancialState:
    """Healthy financial state for stress testing.

    Income=5000, Fixed=2000, Disc=800, Credit=3000 @ 24% APR.
    One commitment of 300 due month 3.
    Baseline score ≈ 52.97 (moderate fragility expected).
    """
    return FinancialState(
        monthly_income=5000.0,
        fixed_expenses=2000.0,
        discretionary_expenses=800.0,
        emergency_fund=10000.0,
        credit_balance=3000.0,
        credit_apr=0.24,
        commitments=[Commitment(id="ins", amount=300.0, due_month=3)],
    )


@pytest.fixture
def comfortable_state() -> FinancialState:
    """Very comfortable state — high income, low expenses, tiny credit.

    Expected fragility < 5 → risk_level = "low".
    """
    return FinancialState(
        monthly_income=10000.0,
        fixed_expenses=1000.0,
        discretionary_expenses=500.0,
        emergency_fund=50000.0,
        credit_balance=500.0,
        credit_apr=0.12,
        commitments=[],
    )


@pytest.fixture
def fragile_state() -> FinancialState:
    """Tight finances — high credit, high APR, low headroom.

    Expected fragility ≥ 15 → risk_level = "high".
    """
    return FinancialState(
        monthly_income=3000.0,
        fixed_expenses=1500.0,
        discretionary_expenses=800.0,
        emergency_fund=500.0,
        credit_balance=4000.0,
        credit_apr=0.36,
        commitments=[Commitment(id="rent", amount=500.0, due_month=1)],
    )


# ===================================================================
# SECTION 1: CORE STRESS TEST LOGIC
# ===================================================================


class TestStressTestCore:
    """Validate the run_stress_test function produces correct results."""

    def test_standard_state_moderate_risk(self, standard_state: FinancialState) -> None:
        """Standard state should produce moderate fragility (5 ≤ frag < 15).

        Baseline ≈ 52.97, income_drop ≈ 41.70, expense_spike ≈ 41.15
        fragility = ((52.97 - 41.70) + (52.97 - 41.15)) / 2 = 11.54
        """
        result = run_stress_test(standard_state, baseline_simulation)

        assert result.baseline_score == pytest.approx(52.97, abs=1e-2)
        assert result.income_drop_score == pytest.approx(41.70, abs=1e-2)
        assert result.expense_spike_score == pytest.approx(41.15, abs=1e-2)
        assert result.fragility_index == pytest.approx(11.54, abs=1e-2)
        assert result.risk_level == "moderate"

    def test_comfortable_state_low_risk(self, comfortable_state: FinancialState) -> None:
        """High income, low expenses → fragility < 5 → risk_level = "low".

        Baseline ≈ 93.96, fragility ≈ 1.66.
        """
        result = run_stress_test(comfortable_state, baseline_simulation)

        assert result.baseline_score == pytest.approx(93.96, abs=1e-2)
        assert result.fragility_index == pytest.approx(1.66, abs=1e-2)
        assert result.risk_level == "low"

    def test_fragile_state_high_risk(self, fragile_state: FinancialState) -> None:
        """Tight finances → fragility ≥ 15 → risk_level = "high".

        Baseline ≈ 29.52, income_drop ≈ 7.5, expense_spike ≈ 8.0
        fragility = ((29.52 - 7.5) + (29.52 - 8.0)) / 2 = 21.77
        """
        result = run_stress_test(fragile_state, baseline_simulation)

        assert result.baseline_score == pytest.approx(29.52, abs=1e-2)
        assert result.income_drop_score == pytest.approx(7.5, abs=1e-2)
        assert result.expense_spike_score == pytest.approx(8.0, abs=1e-2)
        assert result.fragility_index == pytest.approx(21.77, abs=1e-2)
        assert result.risk_level == "high"


# ===================================================================
# SECTION 2: SCENARIO DETAILS
# ===================================================================


class TestScenarioDetails:
    """Verify per-scenario metrics are computed correctly."""

    def test_income_drop_reduces_projected_balance(self, standard_state: FinancialState) -> None:
        """20% income drop: 5000 → 4000 income.

        Baseline balance = 5000 - 2000 - 800 - 60 - 300 = 1840
        Income drop balance = 4000 - 2000 - 800 - 60 - 300 = 840
        Difference = exactly 1000 (the 20% of 5000 lost).
        """
        result = run_stress_test(standard_state, baseline_simulation)

        baseline = next(s for s in result.scenarios if s.scenario_name == "baseline")
        income = next(s for s in result.scenarios if s.scenario_name == "income_drop")

        assert baseline.projected_balance == pytest.approx(1840.0, abs=1e-2)
        assert income.projected_balance == pytest.approx(840.0, abs=1e-2)
        assert baseline.projected_balance - income.projected_balance == pytest.approx(1000.0, abs=1e-2)

    def test_expense_spike_increases_outflows(self, standard_state: FinancialState) -> None:
        """30% expense increase: fixed 2000→2600, disc 800→1040.

        Baseline balance = 1840
        Expense spike balance = 5000 - 2600 - 1040 - 60 - 300 = 1000
        Extra outflow = 600 + 240 = 840.
        """
        result = run_stress_test(standard_state, baseline_simulation)

        baseline = next(s for s in result.scenarios if s.scenario_name == "baseline")
        expense = next(s for s in result.scenarios if s.scenario_name == "expense_spike")

        assert expense.projected_balance == pytest.approx(1000.0, abs=1e-2)
        expected_extra = round(2000.0 * 0.30 + 800.0 * 0.30, 2)  # 840.0
        assert baseline.projected_balance - expense.projected_balance == pytest.approx(expected_extra, abs=1e-2)

    def test_interest_cost_unchanged_across_scenarios(self, standard_state: FinancialState) -> None:
        """Interest depends only on credit_balance and APR, which are unchanged.

        All three scenarios should have identical interest_cost = 60.0.
        """
        result = run_stress_test(standard_state, baseline_simulation)

        for s in result.scenarios:
            assert s.interest_cost == pytest.approx(60.0, abs=1e-2)

    def test_three_scenarios_always_present(self, standard_state: FinancialState) -> None:
        """Result must contain exactly 3 scenarios: baseline, income_drop, expense_spike."""
        result = run_stress_test(standard_state, baseline_simulation)

        assert len(result.scenarios) == 3
        names = {s.scenario_name for s in result.scenarios}
        assert names == {"baseline", "income_drop", "expense_spike"}


# ===================================================================
# SECTION 3: FRAGILITY INDEX & RISK LEVEL
# ===================================================================


class TestFragilityAndRisk:
    """Verify fragility index computation and risk level classification."""

    def test_fragility_formula_is_correct(self, standard_state: FinancialState) -> None:
        """fragility = ((baseline - income_drop) + (baseline - expense_spike)) / 2

        52.97 - 41.70 = 11.27
        52.97 - 41.15 = 11.82
        (11.27 + 11.82) / 2 = 11.545 → rounded to 11.54
        """
        result = run_stress_test(standard_state, baseline_simulation)

        manual_fragility = round(
            ((result.baseline_score - result.income_drop_score)
             + (result.baseline_score - result.expense_spike_score)) / 2.0,
            2,
        )
        assert result.fragility_index == pytest.approx(manual_fragility, abs=1e-2)

    def test_fragility_index_always_non_negative(
        self, standard_state: FinancialState, comfortable_state: FinancialState, fragile_state: FinancialState
    ) -> None:
        """Fragility index must be >= 0 for any valid financial state."""
        for state in [standard_state, comfortable_state, fragile_state]:
            result = run_stress_test(state, baseline_simulation)
            assert result.fragility_index >= 0

    def test_stress_scores_never_exceed_baseline(
        self, standard_state: FinancialState, comfortable_state: FinancialState, fragile_state: FinancialState
    ) -> None:
        """Stress scenarios can only worsen (or equal) the baseline — never improve it."""
        for state in [standard_state, comfortable_state, fragile_state]:
            result = run_stress_test(state, baseline_simulation)
            assert result.income_drop_score <= result.baseline_score
            assert result.expense_spike_score <= result.baseline_score


# ===================================================================
# SECTION 4: IMMUTABILITY & DETERMINISM
# ===================================================================


class TestImmutabilityAndDeterminism:
    """Ensure original state is never mutated and results are deterministic."""

    def test_original_state_not_mutated(self, standard_state: FinancialState) -> None:
        """Deep equality check: original state must be identical before and after."""
        snapshot = standard_state.model_dump()

        _ = run_stress_test(standard_state, baseline_simulation)

        assert standard_state.model_dump() == snapshot

    def test_repeated_runs_produce_identical_results(self, standard_state: FinancialState) -> None:
        """10 repeated runs must produce bit-identical JSON output."""
        reference = run_stress_test(standard_state, baseline_simulation)
        ref_json = reference.model_dump_json()

        for _ in range(10):
            repeat = run_stress_test(standard_state, baseline_simulation)
            assert repeat.model_dump_json() == ref_json

    def test_rounding_consistency(self, fragile_state: FinancialState) -> None:
        """All monetary outputs must be rounded to exactly 2 decimal places."""
        result = run_stress_test(fragile_state, baseline_simulation)

        for s in result.scenarios:
            assert s.stability_score == round(s.stability_score, 2)
            assert s.projected_balance == round(s.projected_balance, 2)
            assert s.liquidity_ratio == round(s.liquidity_ratio, 2)
            assert s.commitment_coverage == round(s.commitment_coverage, 2)
            assert s.interest_cost == round(s.interest_cost, 2)

        assert result.fragility_index == round(result.fragility_index, 2)


# ===================================================================
# SECTION 5: OUTPUT INVARIANTS
# ===================================================================


class TestOutputInvariants:
    """Schema bounds must hold for every scenario under every state."""

    def test_all_scenarios_satisfy_bounds(
        self, standard_state: FinancialState, comfortable_state: FinancialState, fragile_state: FinancialState
    ) -> None:
        """For every result:
        - 0 <= stability_score <= 100
        - liquidity_ratio >= 0
        - 0 <= commitment_coverage <= 1
        - interest_cost >= 0
        """
        for state in [standard_state, comfortable_state, fragile_state]:
            result = run_stress_test(state, baseline_simulation)
            for s in result.scenarios:
                assert 0 <= s.stability_score <= 100, (
                    f"{s.scenario_name}: score {s.stability_score} out of [0, 100]"
                )
                assert s.liquidity_ratio >= 0, (
                    f"{s.scenario_name}: liquidity_ratio {s.liquidity_ratio} < 0"
                )
                assert 0 <= s.commitment_coverage <= 1, (
                    f"{s.scenario_name}: commitment_coverage {s.commitment_coverage} out of [0, 1]"
                )
                assert s.interest_cost >= 0, (
                    f"{s.scenario_name}: interest_cost {s.interest_cost} < 0"
                )


# ===================================================================
# SECTION 6: MODEL VALIDATION ERRORS
# ===================================================================


class TestModelValidation:
    """Verify that invalid StressTestResult constructions are rejected."""

    def test_stress_score_exceeding_baseline_raises_error(self) -> None:
        """income_drop_score > baseline_score must raise ValueError."""
        with pytest.raises(Exception, match="exceeds"):
            StressTestResult(
                baseline_score=50.0,
                income_drop_score=60.0,  # higher than baseline!
                expense_spike_score=45.0,
                fragility_index=0.0,
                risk_level="low",
                scenarios=[],
            )

    def test_negative_fragility_index_raises_error(self) -> None:
        """fragility_index < 0 must raise ValueError."""
        with pytest.raises(Exception, match="fragility_index must be >= 0"):
            StressTestResult(
                baseline_score=50.0,
                income_drop_score=45.0,
                expense_spike_score=45.0,
                fragility_index=-5.0,
                risk_level="low",
                scenarios=[],
            )
