"""
Deterministic Test Suite — Financial Simulation Engine

Verifies mathematical correctness of:
  - simulate_subscription_options()
  - simulate_credit_strategies()

All tests use fixed numeric inputs, round to 2 decimal places, and
compare with pytest.approx(tolerance=1e-2).  No randomness, no time
dependency, no external services.
"""

from __future__ import annotations

import sys
import os

import pytest

# Ensure the project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from financial_schemas import (
    Commitment,
    CreditStrategyRequest,
    FinancialState,
    SubscriptionOption,
)
from simulation_engine import simulate_credit_strategies, simulate_subscription_options


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_state() -> FinancialState:
    """Standard financial state used across subscription tests.

    Income: 5000, Fixed: 2000, Disc: 800, Emergency: 10000
    Credit: 3000 @ 24% APR, one commitment of 300 due month 3.
    Monthly interest = 3000 * 0.24 / 12 = 60.0
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


# ===================================================================
# SECTION 1: SUBSCRIPTION SCENARIOS (5 tests)
# ===================================================================


class TestSubscriptionScenarios:
    """Evaluate how adding subscription costs affects financial stability."""

    def test_low_cost_subscription_remains_stable(self, base_state: FinancialState) -> None:
        """A $10/mo subscription should barely dent a $5000 income.

        Baseline projected_balance = 5000 - 2000 - 800 - 60 - 300 = 1840
        With $10 sub:             = 5000 - 2000 - 800 - 10 - 60 - 300 = 1830
        Score should be very close to baseline (~52.97 vs ~52.89).
        No risk flags expected.
        """
        subs = [SubscriptionOption(name="basic-streaming", monthly_cost=10.0)]
        result_set = simulate_subscription_options(base_state, subs)

        # Results must be sorted descending by stability_score
        scores = [r.stability_score for r in result_set.results]
        assert scores == sorted(scores, reverse=True)

        baseline = next(r for r in result_set.results if r.option_id == "no-subscription")
        sub_result = next(r for r in result_set.results if r.option_id == "basic-streaming")

        # Baseline checks
        assert baseline.projected_balance == pytest.approx(1840.0, abs=1e-2)
        assert baseline.interest_cost == pytest.approx(60.0, abs=1e-2)
        assert baseline.risk_flags == []

        # Low-cost sub should be very close to baseline
        assert sub_result.projected_balance == pytest.approx(1830.0, abs=1e-2)
        assert sub_result.stability_score == pytest.approx(52.89, abs=1e-2)
        assert sub_result.risk_flags == []

        # Score drop should be minimal (< 1 point)
        assert baseline.stability_score - sub_result.stability_score < 1.0

    def test_moderate_subscription_slight_score_drop(self, base_state: FinancialState) -> None:
        """A $100/mo premium SaaS should cause a noticeable but non-critical drop.

        With $100 sub: balance = 5000 - 2000 - 800 - 100 - 60 - 300 = 1740
        Score should drop by ~1.5 points.  No risk flags.
        """
        subs = [SubscriptionOption(name="premium-saas", monthly_cost=100.0)]
        result_set = simulate_subscription_options(base_state, subs)

        baseline = next(r for r in result_set.results if r.option_id == "no-subscription")
        sub_result = next(r for r in result_set.results if r.option_id == "premium-saas")

        assert sub_result.projected_balance == pytest.approx(1740.0, abs=1e-2)
        assert sub_result.stability_score == pytest.approx(51.42, abs=1e-2)
        assert sub_result.liquidity_ratio == pytest.approx(0.54, abs=1e-2)
        assert sub_result.commitment_coverage == pytest.approx(1.0, abs=1e-2)
        assert sub_result.risk_flags == []

        # Baseline should outscore the subscription
        assert baseline.stability_score > sub_result.stability_score

    def test_high_cost_subscription_triggers_liquidity_risk(self, base_state: FinancialState) -> None:
        """A $1500/mo enterprise suite should trigger low_liquidity flag.

        With $1500 sub: balance = 5000 - 2000 - 800 - 1500 - 60 - 300 = 340
        Liquidity ratio drops to ~0.07 (< 0.5 threshold).
        """
        subs = [SubscriptionOption(name="enterprise-suite", monthly_cost=1500.0)]
        result_set = simulate_subscription_options(base_state, subs)

        sub_result = next(r for r in result_set.results if r.option_id == "enterprise-suite")

        assert sub_result.projected_balance == pytest.approx(340.0, abs=1e-2)
        assert sub_result.liquidity_ratio == pytest.approx(0.07, abs=1e-2)
        assert sub_result.stability_score == pytest.approx(33.17, abs=1e-2)
        assert "low_liquidity" in sub_result.risk_flags

    def test_multiple_subscriptions_combined_impact(self, base_state: FinancialState) -> None:
        """Three small subscriptions ($15 + $12 + $8 = $35 combined).

        Each is evaluated independently; the cheapest one ($8) should
        score highest among the three subs.
        """
        subs = [
            SubscriptionOption(name="streaming", monthly_cost=15.0),
            SubscriptionOption(name="cloud-storage", monthly_cost=12.0),
            SubscriptionOption(name="vpn-service", monthly_cost=8.0),
        ]
        result_set = simulate_subscription_options(base_state, subs)

        # Should have 4 results (3 subs + baseline)
        assert len(result_set.results) == 4

        # Sorted descending
        scores = [r.stability_score for r in result_set.results]
        assert scores == sorted(scores, reverse=True)

        # Baseline (no sub) should be ranked first
        assert result_set.results[0].option_id == "no-subscription"

        # The cheapest sub should rank highest among the three subs
        sub_scores = {r.option_id: r.stability_score for r in result_set.results if r.option_id != "no-subscription"}
        best_sub = max(sub_scores, key=sub_scores.get)
        assert best_sub == "vpn-service"

        # All balances should be positive, no risk flags
        for r in result_set.results:
            assert r.projected_balance > 0
            assert r.risk_flags == []

    def test_subscription_causing_commitment_failure(self) -> None:
        """When income is tight, a $200/mo gym membership can break commitment coverage.

        State: income=3000, fixed=1500, disc=600, credit=2000@24% → interest=40
        Commitment: 800 due month 6.
        Without sub: available = 3000 - 1500 - 600 - 40 = 860 → coverage = min(860/800, 1) = 1.0
        With $200 sub: available = 3000 - 1500 - 600 - 200 - 40 = 660 → coverage = 660/800 = 0.825
        Should trigger commitment_breach and negative_balance flags.
        """
        state = FinancialState(
            monthly_income=3000.0,
            fixed_expenses=1500.0,
            discretionary_expenses=600.0,
            emergency_fund=2000.0,
            credit_balance=2000.0,
            credit_apr=0.24,
            commitments=[Commitment(id="tuition", amount=800.0, due_month=6)],
        )
        subs = [SubscriptionOption(name="expensive-gym", monthly_cost=200.0)]
        result_set = simulate_subscription_options(state, subs)

        gym = next(r for r in result_set.results if r.option_id == "expensive-gym")

        assert gym.projected_balance == pytest.approx(-140.0, abs=1e-2)
        assert gym.commitment_coverage == pytest.approx(0.82, abs=1e-2)
        assert "commitment_breach" in gym.risk_flags
        assert "negative_balance" in gym.risk_flags
        assert "low_liquidity" in gym.risk_flags


# ===================================================================
# SECTION 2: CREDIT STRATEGY SCENARIOS (5 tests)
# ===================================================================


class TestCreditStrategyScenarios:
    """Evaluate credit payment strategies and their impact on financial health."""

    def test_minimum_payment_highest_interest(self, base_state: FinancialState) -> None:
        """Minimum payment (2% of balance = 60) → residual interest accrues.

        Interest on remaining balance (3000 - 60) * 0.24/12 = 58.80
        But minimum strategy still carries the full month interest = 60.
        The minimum strategy should score highest because it preserves cash.
        """
        result_set = simulate_credit_strategies(
            base_state, CreditStrategyRequest(strategy_type="minimum")
        )

        minimum = next(r for r in result_set.results if r.option_id == "minimum")

        assert minimum.interest_cost == pytest.approx(60.0, abs=1e-2)
        assert minimum.stability_score == pytest.approx(51.12, abs=1e-2)
        assert minimum.projected_balance == pytest.approx(1721.2, abs=1e-2)

    def test_partial_25_percent_moderate_interest(self, base_state: FinancialState) -> None:
        """25% partial payment (750) → reduced interest on remaining 2250.

        Interest = 2250 * 0.24/12 = 45.0
        More cash outflow than minimum but lower interest.
        """
        result_set = simulate_credit_strategies(
            base_state,
            CreditStrategyRequest(strategy_type="partial", partial_percentage=0.25),
        )

        partial = next(r for r in result_set.results if r.option_id == "partial")

        assert partial.interest_cost == pytest.approx(45.0, abs=1e-2)
        assert partial.projected_balance == pytest.approx(1060.0, abs=1e-2)
        assert partial.stability_score == pytest.approx(42.08, abs=1e-2)
        assert "low_liquidity" in partial.risk_flags

    def test_full_repayment_zero_interest(self, base_state: FinancialState) -> None:
        """Full repayment of 3000 → zero interest but large cash drain.

        Projected balance = 5000 - 2000 - 800 - 3000 - 0 - 300 = -1100
        Score should be lowest due to negative balance.
        """
        result_set = simulate_credit_strategies(
            base_state, CreditStrategyRequest(strategy_type="full")
        )

        full = next(r for r in result_set.results if r.option_id == "full")

        assert full.interest_cost == pytest.approx(0.0, abs=1e-2)
        assert full.projected_balance == pytest.approx(-1100.0, abs=1e-2)
        assert "negative_balance" in full.risk_flags
        assert "commitment_breach" in full.risk_flags

        # Full should score lower than minimum (cash drain)
        minimum = next(r for r in result_set.results if r.option_id == "minimum")
        assert minimum.stability_score > full.stability_score

    def test_high_apr_large_interest_cost(self) -> None:
        """90% APR on $8000 balance → monthly interest = 8000 * 0.9/12 = 600.

        Interest > 10% of income (600 > 500) → high_interest flag.
        """
        state = FinancialState(
            monthly_income=5000.0,
            fixed_expenses=2000.0,
            discretionary_expenses=800.0,
            emergency_fund=5000.0,
            credit_balance=8000.0,
            credit_apr=0.90,
            commitments=[],
        )
        result_set = simulate_credit_strategies(
            state, CreditStrategyRequest(strategy_type="minimum")
        )

        minimum = next(r for r in result_set.results if r.option_id == "minimum")

        assert minimum.interest_cost == pytest.approx(600.0, abs=1e-2)
        assert "high_interest" in minimum.risk_flags
        assert minimum.stability_score == pytest.approx(34.42, abs=1e-2)

    def test_credit_strategy_causing_negative_balance(self) -> None:
        """Full repayment of $5000 on $3000 income → deep negative balance.

        balance = 3000 - 1500 - 600 - 5000 - 0 = -4100
        Must not crash; should flag negative_balance.
        """
        state = FinancialState(
            monthly_income=3000.0,
            fixed_expenses=1500.0,
            discretionary_expenses=600.0,
            emergency_fund=1000.0,
            credit_balance=5000.0,
            credit_apr=0.36,
            commitments=[],
        )
        result_set = simulate_credit_strategies(
            state, CreditStrategyRequest(strategy_type="full")
        )

        full = next(r for r in result_set.results if r.option_id == "full")

        assert full.projected_balance == pytest.approx(-4100.0, abs=1e-2)
        assert full.interest_cost == pytest.approx(0.0, abs=1e-2)
        assert "negative_balance" in full.risk_flags
        assert "low_liquidity" in full.risk_flags
        assert full.liquidity_ratio >= 0  # must never be negative


# ===================================================================
# SECTION 3: EDGE CASE TESTS
# ===================================================================


class TestEdgeCases:
    """Boundary conditions that must not crash or produce invalid outputs."""

    def test_zero_income_no_division_by_zero(self) -> None:
        """Zero income should collapse stability without causing ZeroDivisionError.

        All strategies should produce valid results with low/zero scores.
        """
        state = FinancialState(
            monthly_income=0.0,
            fixed_expenses=0.0,
            discretionary_expenses=0.0,
            emergency_fund=500.0,
            credit_balance=1000.0,
            credit_apr=0.24,
            commitments=[],
        )
        # This must not raise ZeroDivisionError
        result_set = simulate_credit_strategies(
            state, CreditStrategyRequest(strategy_type="minimum")
        )

        for r in result_set.results:
            # All scores should be within valid bounds
            assert 0 <= r.stability_score <= 100
            assert r.liquidity_ratio >= 0
            # Stability should be low across the board
            assert r.stability_score <= 50

    def test_high_apr_correct_interest_calculation(self) -> None:
        """APR = 0.9 (90%) on $8000 → monthly interest = 600.

        Interest = 8000 * 0.90 / 12 = 600.0
        600 > 10% × 5000 (500) → high_interest flag must fire.
        """
        state = FinancialState(
            monthly_income=5000.0,
            fixed_expenses=2000.0,
            discretionary_expenses=800.0,
            emergency_fund=5000.0,
            credit_balance=8000.0,
            credit_apr=0.90,
            commitments=[],
        )
        result_set = simulate_credit_strategies(
            state, CreditStrategyRequest(strategy_type="minimum")
        )

        minimum = next(r for r in result_set.results if r.option_id == "minimum")
        assert minimum.interest_cost == pytest.approx(600.0, abs=1e-2)
        assert "high_interest" in minimum.risk_flags

    def test_negative_projected_balance_valid_output(self) -> None:
        """Full repayment exceeding income → negative balance.

        Must not crash.  liquidity_ratio clamps at 0.  stability is low.
        """
        state = FinancialState(
            monthly_income=3000.0,
            fixed_expenses=1500.0,
            discretionary_expenses=600.0,
            emergency_fund=1000.0,
            credit_balance=5000.0,
            credit_apr=0.36,
            commitments=[],
        )
        result_set = simulate_credit_strategies(
            state, CreditStrategyRequest(strategy_type="full")
        )

        full = next(r for r in result_set.results if r.option_id == "full")

        assert full.projected_balance < 0
        assert full.liquidity_ratio >= 0  # must clamp, never negative
        assert full.stability_score <= 50  # should be low
        assert full.stability_score >= 0   # must never go below 0


# ===================================================================
# SECTION 4: FLOATING-POINT CONSISTENCY
# ===================================================================


class TestFloatingPointConsistency:
    """Verify rounding consistency across repeated deterministic runs."""

    def test_repeated_runs_produce_identical_results(self, base_state: FinancialState) -> None:
        """Running the same simulation 10 times must yield bit-identical outputs.

        This guards against hidden non-determinism (e.g. dict ordering,
        floating-point drift from reordered operations).
        """
        subs = [
            SubscriptionOption(name="streaming", monthly_cost=15.0),
            SubscriptionOption(name="cloud-storage", monthly_cost=12.0),
        ]

        reference = simulate_subscription_options(base_state, subs)
        ref_json = reference.model_dump_json()

        for _ in range(10):
            repeat = simulate_subscription_options(base_state, subs)
            assert repeat.model_dump_json() == ref_json, (
                "Determinism violation: repeated run produced different output"
            )

        # Also verify monetary rounding to 2 decimal places
        for r in reference.results:
            assert r.projected_balance == round(r.projected_balance, 2)
            assert r.interest_cost == round(r.interest_cost, 2)
            assert r.liquidity_ratio == round(r.liquidity_ratio, 2)


# ===================================================================
# SECTION 5: OUTPUT INVARIANT GUARDS
# ===================================================================


class TestOutputInvariants:
    """Every SimulationResult must satisfy schema invariants regardless of input."""

    def test_all_outputs_satisfy_schema_bounds(self, base_state: FinancialState) -> None:
        """Across a variety of simulations, every result must satisfy:

        - 0 <= stability_score <= 100
        - liquidity_ratio >= 0
        - 0 <= commitment_coverage <= 1
        - interest_cost >= 0
        """
        # Run several diverse simulations
        subs = [
            SubscriptionOption(name="cheap", monthly_cost=5.0),
            SubscriptionOption(name="expensive", monthly_cost=2000.0),
        ]
        sub_results = simulate_subscription_options(base_state, subs)

        strategies = [
            CreditStrategyRequest(strategy_type="minimum"),
            CreditStrategyRequest(strategy_type="partial", partial_percentage=0.5),
            CreditStrategyRequest(strategy_type="full"),
        ]
        credit_results = [
            simulate_credit_strategies(base_state, s) for s in strategies
        ]

        # Collect every result
        all_results = list(sub_results.results)
        for cr in credit_results:
            all_results.extend(cr.results)

        for r in all_results:
            assert 0 <= r.stability_score <= 100, (
                f"{r.option_id}: score {r.stability_score} out of [0, 100]"
            )
            assert r.liquidity_ratio >= 0, (
                f"{r.option_id}: liquidity_ratio {r.liquidity_ratio} < 0"
            )
            assert 0 <= r.commitment_coverage <= 1, (
                f"{r.option_id}: commitment_coverage {r.commitment_coverage} out of [0, 1]"
            )
            assert r.interest_cost >= 0, (
                f"{r.option_id}: interest_cost {r.interest_cost} < 0"
            )
