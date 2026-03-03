"""
Stability Score Calibration & Backtesting — Financial Decision Engine

Production-grade calibration module that:
  1. Generates 20 deterministic synthetic financial profiles
  2. Backtests the current stability-score formula
  3. Analyzes score distribution quality
  4. Verifies monotonic behavior along each axis
  5. Calibrates weights via deterministic grid search
  6. Produces versioned calibration output

No LLM usage.  No probabilistic optimisation.  No neural fitting.
No modification of simulation logic, risk flags, or persistence.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from financial_schemas import Commitment, FinancialState
from simulation_schemas import SimulationResult


# ═══════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════

CURRENT_FORMULA_VERSION = "v1.0.0"


# ═══════════════════════════════════════════════════════════════════════
#  Section 1 — Synthetic Profile Generator
# ═══════════════════════════════════════════════════════════════════════

def generate_synthetic_profiles() -> List[Dict[str, Any]]:
    """Generate 20 deterministic synthetic financial profiles.

    Profiles cover a wide range of financial situations including
    edge cases, stress conditions, and comfortable scenarios.
    Each profile is fully reproducible (no random components).
    """
    profiles: List[Dict[str, Any]] = [
        # 1. Extremely safe — high income, low expenses, large buffer
        {
            "id": "P01_extremely_safe",
            "monthly_income": 12000.0,
            "fixed_expenses": 1500.0,
            "discretionary_expenses": 500.0,
            "emergency_fund": 50000.0,
            "credit_balance": 0.0,
            "credit_apr": 0.0,
            "commitments": [],
        },
        # 2. Comfortable professional
        {
            "id": "P02_comfortable_pro",
            "monthly_income": 8000.0,
            "fixed_expenses": 2500.0,
            "discretionary_expenses": 1000.0,
            "emergency_fund": 20000.0,
            "credit_balance": 1000.0,
            "credit_apr": 0.15,
            "commitments": [{"id": "car-ins", "amount": 200.0, "due_month": 3}],
        },
        # 3. Moderate stable profile
        {
            "id": "P03_moderate_stable",
            "monthly_income": 5000.0,
            "fixed_expenses": 2000.0,
            "discretionary_expenses": 800.0,
            "emergency_fund": 10000.0,
            "credit_balance": 3000.0,
            "credit_apr": 0.24,
            "commitments": [{"id": "ins", "amount": 300.0, "due_month": 3}],
        },
        # 4. High liquidity buffer
        {
            "id": "P04_high_liquidity",
            "monthly_income": 6000.0,
            "fixed_expenses": 2000.0,
            "discretionary_expenses": 600.0,
            "emergency_fund": 40000.0,
            "credit_balance": 500.0,
            "credit_apr": 0.18,
            "commitments": [],
        },
        # 5. Young professional — moderate debt
        {
            "id": "P05_young_pro_debt",
            "monthly_income": 4500.0,
            "fixed_expenses": 1800.0,
            "discretionary_expenses": 700.0,
            "emergency_fund": 5000.0,
            "credit_balance": 8000.0,
            "credit_apr": 0.22,
            "commitments": [{"id": "loan", "amount": 400.0, "due_month": 6}],
        },
        # 6. Near-break-even — tight margin
        {
            "id": "P06_near_breakeven",
            "monthly_income": 3500.0,
            "fixed_expenses": 2000.0,
            "discretionary_expenses": 800.0,
            "emergency_fund": 3000.0,
            "credit_balance": 2000.0,
            "credit_apr": 0.20,
            "commitments": [{"id": "rent-top", "amount": 500.0, "due_month": 1}],
        },
        # 7. High commitment load
        {
            "id": "P07_high_commitments",
            "monthly_income": 5500.0,
            "fixed_expenses": 2200.0,
            "discretionary_expenses": 600.0,
            "emergency_fund": 7000.0,
            "credit_balance": 4000.0,
            "credit_apr": 0.20,
            "commitments": [
                {"id": "ins-a", "amount": 500.0, "due_month": 2},
                {"id": "ins-b", "amount": 400.0, "due_month": 4},
                {"id": "ins-c", "amount": 300.0, "due_month": 6},
            ],
        },
        # 8. High APR debt stress
        {
            "id": "P08_high_apr_stress",
            "monthly_income": 4000.0,
            "fixed_expenses": 1800.0,
            "discretionary_expenses": 500.0,
            "emergency_fund": 2000.0,
            "credit_balance": 15000.0,
            "credit_apr": 0.30,
            "commitments": [{"id": "loan", "amount": 600.0, "due_month": 3}],
        },
        # 9. Low income, high expenses
        {
            "id": "P09_low_income_high_exp",
            "monthly_income": 2500.0,
            "fixed_expenses": 1800.0,
            "discretionary_expenses": 500.0,
            "emergency_fund": 1000.0,
            "credit_balance": 5000.0,
            "credit_apr": 0.25,
            "commitments": [{"id": "medical", "amount": 300.0, "due_month": 2}],
        },
        # 10. Negative projected balance
        {
            "id": "P10_negative_balance",
            "monthly_income": 3000.0,
            "fixed_expenses": 2500.0,
            "discretionary_expenses": 800.0,
            "emergency_fund": 500.0,
            "credit_balance": 10000.0,
            "credit_apr": 0.28,
            "commitments": [
                {"id": "rent", "amount": 800.0, "due_month": 1},
                {"id": "loan", "amount": 500.0, "due_month": 3},
            ],
        },
        # 11. Zero income edge case
        {
            "id": "P11_zero_income",
            "monthly_income": 0.0,
            "fixed_expenses": 500.0,
            "discretionary_expenses": 200.0,
            "emergency_fund": 3000.0,
            "credit_balance": 1000.0,
            "credit_apr": 0.20,
            "commitments": [],
        },
        # 12. Minimal everything
        {
            "id": "P12_minimal",
            "monthly_income": 2000.0,
            "fixed_expenses": 800.0,
            "discretionary_expenses": 300.0,
            "emergency_fund": 500.0,
            "credit_balance": 200.0,
            "credit_apr": 0.15,
            "commitments": [],
        },
        # 13. High income, high debt
        {
            "id": "P13_high_income_debt",
            "monthly_income": 10000.0,
            "fixed_expenses": 3000.0,
            "discretionary_expenses": 1500.0,
            "emergency_fund": 15000.0,
            "credit_balance": 25000.0,
            "credit_apr": 0.22,
            "commitments": [
                {"id": "mortgage", "amount": 2000.0, "due_month": 1},
            ],
        },
        # 14. Retiree — low income, large fund
        {
            "id": "P14_retiree",
            "monthly_income": 2000.0,
            "fixed_expenses": 1200.0,
            "discretionary_expenses": 300.0,
            "emergency_fund": 80000.0,
            "credit_balance": 0.0,
            "credit_apr": 0.0,
            "commitments": [{"id": "insurance", "amount": 200.0, "due_month": 6}],
        },
        # 15. Student — very low income, some debt
        {
            "id": "P15_student",
            "monthly_income": 1200.0,
            "fixed_expenses": 700.0,
            "discretionary_expenses": 300.0,
            "emergency_fund": 500.0,
            "credit_balance": 3000.0,
            "credit_apr": 0.20,
            "commitments": [],
        },
        # 16. Freelancer — variable-like moderate
        {
            "id": "P16_freelancer",
            "monthly_income": 4000.0,
            "fixed_expenses": 1500.0,
            "discretionary_expenses": 600.0,
            "emergency_fund": 8000.0,
            "credit_balance": 2000.0,
            "credit_apr": 0.18,
            "commitments": [{"id": "tools", "amount": 150.0, "due_month": 5}],
        },
        # 17. Over-leveraged — drowning in debt
        {
            "id": "P17_overleveraged",
            "monthly_income": 3500.0,
            "fixed_expenses": 2000.0,
            "discretionary_expenses": 600.0,
            "emergency_fund": 200.0,
            "credit_balance": 20000.0,
            "credit_apr": 0.28,
            "commitments": [
                {"id": "loan-a", "amount": 700.0, "due_month": 2},
                {"id": "loan-b", "amount": 500.0, "due_month": 4},
            ],
        },
        # 18. Dual income household simulation
        {
            "id": "P18_dual_income",
            "monthly_income": 9000.0,
            "fixed_expenses": 3000.0,
            "discretionary_expenses": 1200.0,
            "emergency_fund": 25000.0,
            "credit_balance": 2000.0,
            "credit_apr": 0.16,
            "commitments": [
                {"id": "car", "amount": 350.0, "due_month": 3},
                {"id": "insurance", "amount": 250.0, "due_month": 6},
            ],
        },
        # 19. New graduate — first job
        {
            "id": "P19_new_grad",
            "monthly_income": 3200.0,
            "fixed_expenses": 1400.0,
            "discretionary_expenses": 500.0,
            "emergency_fund": 2000.0,
            "credit_balance": 1500.0,
            "credit_apr": 0.19,
            "commitments": [{"id": "student-loan", "amount": 250.0, "due_month": 4}],
        },
        # 20. Financial crisis — extreme stress
        {
            "id": "P20_crisis",
            "monthly_income": 1500.0,
            "fixed_expenses": 1400.0,
            "discretionary_expenses": 400.0,
            "emergency_fund": 100.0,
            "credit_balance": 12000.0,
            "credit_apr": 0.30,
            "commitments": [
                {"id": "medical-a", "amount": 500.0, "due_month": 1},
                {"id": "medical-b", "amount": 400.0, "due_month": 2},
            ],
        },
    ]
    return profiles


def _to_financial_state(p: Dict[str, Any]) -> FinancialState:
    """Convert a profile dict to a FinancialState."""
    commitments = [
        Commitment(id=c["id"], amount=c["amount"], due_month=c["due_month"])
        for c in p.get("commitments", [])
    ]
    return FinancialState(
        monthly_income=p["monthly_income"],
        fixed_expenses=p["fixed_expenses"],
        discretionary_expenses=p["discretionary_expenses"],
        emergency_fund=p["emergency_fund"],
        credit_balance=p["credit_balance"],
        credit_apr=p["credit_apr"],
        commitments=commitments,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Core Scoring Function (parameterised weights)
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ScoringWeights:
    """Stability score weight configuration."""
    balance_weight: float = 0.40
    liquidity_weight: float = 0.30
    coverage_weight: float = 0.20
    interest_weight: float = 0.10
    interest_penalty_scale: float = 500.0  # multiplier in interest_health formula

    def total(self) -> float:
        return round(
            self.balance_weight
            + self.liquidity_weight
            + self.coverage_weight
            + self.interest_weight,
            6,
        )


DEFAULT_WEIGHTS = ScoringWeights()


def compute_score_components(
    state: FinancialState,
    extra_cost: float = 0.0,
    weights: ScoringWeights = DEFAULT_WEIGHTS,
) -> Dict[str, Any]:
    """Compute all intermediate scoring components and the final score.

    This mirrors the production formula but with configurable weights,
    allowing calibration without modifying the engine itself.
    """
    # --- Derived values ---
    income = state.monthly_income
    commit_total = sum(c.amount for c in state.commitments)
    interest = round(state.credit_balance * (state.credit_apr / 12), 2)
    total_outflows = (
        state.fixed_expenses
        + state.discretionary_expenses
        + extra_cost
        + interest
        + commit_total
    )
    projected_balance = round(income - total_outflows, 2)
    liquidity_ratio = round(
        state.emergency_fund / total_outflows if total_outflows > 0 else 0.0, 2
    )
    commitment_coverage = round(
        1.0
        if commit_total == 0
        else min((income - state.fixed_expenses - state.discretionary_expenses) / commit_total, 1.0)
        if commit_total > 0
        else 1.0,
        2,
    )
    interest_ratio = round(interest / income if income > 0 else 0.0, 4)
    burn_rate = round(total_outflows / income if income > 0 else 0.0, 4)

    # --- Component health scores (each 0–100) ---
    if income > 0:
        balance_health = min(max((projected_balance / income) * 100, 0), 100)
    else:
        balance_health = 0.0 if projected_balance <= 0 else 100.0

    liquidity_health = min(max(liquidity_ratio / 2.0, 0), 1.0) * 100
    coverage_health = min(max(commitment_coverage, 0), 1.0) * 100

    if income > 0:
        interest_health = max(100 - (interest / income) * weights.interest_penalty_scale, 0)
    else:
        interest_health = 0.0 if interest > 0 else 100.0

    # --- Weighted sum ---
    raw = (
        weights.balance_weight * balance_health
        + weights.liquidity_weight * liquidity_health
        + weights.coverage_weight * coverage_health
        + weights.interest_weight * interest_health
    )
    stability_score = round(min(max(raw, 0), 100), 2)

    # --- Risk flags (mirrors production) ---
    risk_flags: List[str] = []
    if income > 0 and interest > 0.10 * income:
        risk_flags.append("high_interest")
    if liquidity_ratio < 0.5:
        risk_flags.append("low_liquidity")
    if commitment_coverage < 1.0:
        risk_flags.append("commitment_breach")
    if projected_balance < 0:
        risk_flags.append("negative_balance")

    return {
        "projected_balance": projected_balance,
        "liquidity_ratio": liquidity_ratio,
        "commitment_coverage": commitment_coverage,
        "interest_cost": interest,
        "interest_ratio": interest_ratio,
        "burn_rate": burn_rate,
        "balance_health": round(balance_health, 2),
        "liquidity_health": round(liquidity_health, 2),
        "coverage_health": round(coverage_health, 2),
        "interest_health": round(interest_health, 2),
        "stability_score": stability_score,
        "risk_flags": risk_flags,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Section 2 — Backtest Engine
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BacktestResult:
    profile_id: str
    liquidity_ratio: float
    commitment_coverage: float
    interest_ratio: float
    burn_rate_ratio: float
    stability_score: float
    risk_flags: List[str]


def run_score_backtest(
    profiles: List[Dict[str, Any]],
    weights: ScoringWeights = DEFAULT_WEIGHTS,
) -> List[BacktestResult]:
    """Run the stability score formula against every profile.

    Returns a BacktestResult per profile with all intermediate metrics.
    """
    results: List[BacktestResult] = []
    for p in profiles:
        state = _to_financial_state(p)
        components = compute_score_components(state, extra_cost=0.0, weights=weights)
        results.append(BacktestResult(
            profile_id=p["id"],
            liquidity_ratio=components["liquidity_ratio"],
            commitment_coverage=components["commitment_coverage"],
            interest_ratio=components["interest_ratio"],
            burn_rate_ratio=components["burn_rate"],
            stability_score=components["stability_score"],
            risk_flags=components["risk_flags"],
        ))
    return results


# ═══════════════════════════════════════════════════════════════════════
#  Section 3 — Distribution Analysis
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DistributionMetrics:
    mean: float
    median: float
    min_score: float
    max_score: float
    std_dev: float
    band_0_40_pct: float
    band_40_60_pct: float
    band_60_80_pct: float
    band_80_100_pct: float
    spread: float  # max - min


def analyze_distribution(results: List[BacktestResult]) -> DistributionMetrics:
    """Compute distribution metrics for a set of backtest results."""
    scores = [r.stability_score for r in results]
    n = len(scores)

    count_0_40 = sum(1 for s in scores if s < 40)
    count_40_60 = sum(1 for s in scores if 40 <= s < 60)
    count_60_80 = sum(1 for s in scores if 60 <= s < 80)
    count_80_100 = sum(1 for s in scores if s >= 80)

    return DistributionMetrics(
        mean=round(statistics.mean(scores), 2),
        median=round(statistics.median(scores), 2),
        min_score=min(scores),
        max_score=max(scores),
        std_dev=round(statistics.stdev(scores), 2) if n > 1 else 0.0,
        band_0_40_pct=round(count_0_40 / n * 100, 1),
        band_40_60_pct=round(count_40_60 / n * 100, 1),
        band_60_80_pct=round(count_60_80 / n * 100, 1),
        band_80_100_pct=round(count_80_100 / n * 100, 1),
        spread=round(max(scores) - min(scores), 2),
    )


def check_distribution_criteria(metrics: DistributionMetrics) -> Dict[str, Any]:
    """Check whether the distribution meets acceptability criteria.

    Criteria:
      - >= 10% below 40
      - >= 10% above 80
      - <= 40% in the 60–80 band
      - Spread > 30 (no compression into narrow band)
    """
    checks = {
        "at_least_10pct_below_40": metrics.band_0_40_pct >= 10.0,
        "at_least_10pct_above_80": metrics.band_80_100_pct >= 10.0,
        "no_more_40pct_in_60_80": metrics.band_60_80_pct <= 40.0,
        "spread_gt_30": metrics.spread > 30.0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Section 4 — Monotonicity Tests
# ═══════════════════════════════════════════════════════════════════════

def test_monotonicity(weights: ScoringWeights = DEFAULT_WEIGHTS) -> Dict[str, Any]:
    """Verify monotonic behavior along each financial axis.

    Tests:
      1. Increasing income → score must not decrease
      2. Increasing expenses → score must not increase
      3. Increasing interest (via APR) → score must not increase
      4. Increasing commitment load → score must not increase
    """
    base = {
        "monthly_income": 5000.0,
        "fixed_expenses": 2000.0,
        "discretionary_expenses": 800.0,
        "emergency_fund": 10000.0,
        "credit_balance": 3000.0,
        "credit_apr": 0.20,
        "commitments": [{"id": "ins", "amount": 300.0, "due_month": 3}],
    }

    results: Dict[str, Any] = {}
    all_pass = True

    # Test 1: Increasing income
    income_steps = [2000, 3000, 4000, 5000, 6000, 8000, 10000, 12000]
    prev_score = -1.0
    income_scores = []
    for inc in income_steps:
        p = {**base, "monthly_income": float(inc)}
        s = compute_score_components(_to_financial_state(p), weights=weights)
        income_scores.append({"income": inc, "score": s["stability_score"]})
        if s["stability_score"] < prev_score - 0.01:  # tolerance for rounding
            results["income_monotonicity"] = {
                "passed": False,
                "violation": f"Score decreased from {prev_score} to {s['stability_score']} "
                             f"when income went from {income_steps[income_steps.index(inc)-1]} to {inc}",
                "scores": income_scores,
            }
            all_pass = False
            break
        prev_score = s["stability_score"]
    else:
        results["income_monotonicity"] = {"passed": True, "scores": income_scores}

    # Test 2: Increasing fixed expenses
    expense_steps = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
    prev_score = 101.0
    expense_scores = []
    for exp in expense_steps:
        p = {**base, "fixed_expenses": float(exp)}
        s = compute_score_components(_to_financial_state(p), weights=weights)
        expense_scores.append({"expenses": exp, "score": s["stability_score"]})
        if s["stability_score"] > prev_score + 0.01:
            results["expense_monotonicity"] = {
                "passed": False,
                "violation": f"Score increased from {prev_score} to {s['stability_score']} "
                             f"when expenses went from {expense_steps[expense_steps.index(exp)-1]} to {exp}",
                "scores": expense_scores,
            }
            all_pass = False
            break
        prev_score = s["stability_score"]
    else:
        results["expense_monotonicity"] = {"passed": True, "scores": expense_scores}

    # Test 3: Increasing interest cost (via APR)
    apr_steps = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    prev_score = 101.0
    interest_scores = []
    for apr in apr_steps:
        p = {**base, "credit_apr": apr}
        s = compute_score_components(_to_financial_state(p), weights=weights)
        interest_scores.append({"apr": apr, "score": s["stability_score"]})
        if s["stability_score"] > prev_score + 0.01:
            results["interest_monotonicity"] = {
                "passed": False,
                "violation": f"Score increased from {prev_score} to {s['stability_score']} "
                             f"when APR went from {apr_steps[apr_steps.index(apr)-1]} to {apr}",
                "scores": interest_scores,
            }
            all_pass = False
            break
        prev_score = s["stability_score"]
    else:
        results["interest_monotonicity"] = {"passed": True, "scores": interest_scores}

    # Test 4: Increasing commitment load
    # Single commitment with increasing amounts
    commit_steps = [0, 100, 300, 500, 800, 1200, 2000, 3000]
    prev_score = 101.0
    commit_scores = []
    for amt in commit_steps:
        if amt == 0:
            p = {**base, "commitments": []}
        else:
            p = {**base, "commitments": [{"id": "test", "amount": float(amt), "due_month": 3}]}
        s = compute_score_components(_to_financial_state(p), weights=weights)
        commit_scores.append({"commitment_amount": amt, "score": s["stability_score"]})
        if s["stability_score"] > prev_score + 0.01:
            results["commitment_monotonicity"] = {
                "passed": False,
                "violation": f"Score increased from {prev_score} to {s['stability_score']} "
                             f"when commitment_amount went from "
                             f"{commit_steps[commit_steps.index(amt)-1]} to {amt}",
                "scores": commit_scores,
            }
            all_pass = False
            break
        prev_score = s["stability_score"]
    else:
        results["commitment_monotonicity"] = {"passed": True, "scores": commit_scores}

    return {"all_passed": all_pass, "tests": results}


# ═══════════════════════════════════════════════════════════════════════
#  Section 5 — Weight Calibration (Deterministic Grid Search)
# ═══════════════════════════════════════════════════════════════════════

def calibrate_weights(
    initial_weights: ScoringWeights,
    profiles: List[Dict[str, Any]],
    step: float = 0.05,
    min_weight: float = 0.05,
    max_weight: float = 0.60,
) -> Dict[str, Any]:
    """Calibrate scoring weights via deterministic grid search.

    Iterates over candidate weight combinations that sum to 1.0.
    For each candidate, runs backtest, checks distribution criteria,
    and verifies monotonicity.  Returns the best passing configuration,
    or the initial weights if no better option is found.

    The search is fully deterministic — no randomness, no ML.
    """
    best_weights = initial_weights
    best_spread = 0.0
    best_metrics: Optional[DistributionMetrics] = None
    candidates_tested = 0

    # Generate weight grid (all 4 weights summing to 1.0)
    w_range = _frange(min_weight, max_weight, step)

    for b_w in w_range:
        for l_w in w_range:
            for c_w in w_range:
                i_w = round(1.0 - b_w - l_w - c_w, 6)
                if i_w < min_weight or i_w > max_weight:
                    continue

                candidates_tested += 1
                candidate = ScoringWeights(
                    balance_weight=b_w,
                    liquidity_weight=l_w,
                    coverage_weight=c_w,
                    interest_weight=i_w,
                    interest_penalty_scale=initial_weights.interest_penalty_scale,
                )

                # Quick backtest
                results = run_score_backtest(profiles, weights=candidate)
                metrics = analyze_distribution(results)
                criteria = check_distribution_criteria(metrics)

                if not criteria["passed"]:
                    continue

                # Monotonicity check (expensive, only for passing candidates)
                mono = test_monotonicity(weights=candidate)
                if not mono["all_passed"]:
                    continue

                # Prefer wider spread
                if metrics.spread > best_spread:
                    best_spread = metrics.spread
                    best_weights = candidate
                    best_metrics = metrics

    # Fallback: run distribution on initial weights if no better found
    if best_metrics is None:
        results = run_score_backtest(profiles, weights=initial_weights)
        best_metrics = analyze_distribution(results)

    return {
        "candidates_tested": candidates_tested,
        "calibrated_weights": {
            "balance_weight": best_weights.balance_weight,
            "liquidity_weight": best_weights.liquidity_weight,
            "coverage_weight": best_weights.coverage_weight,
            "interest_weight": best_weights.interest_weight,
            "interest_penalty_scale": best_weights.interest_penalty_scale,
        },
        "distribution_metrics": {
            "mean": best_metrics.mean,
            "median": best_metrics.median,
            "min_score": best_metrics.min_score,
            "max_score": best_metrics.max_score,
            "std_dev": best_metrics.std_dev,
            "spread": best_metrics.spread,
            "band_0_40_pct": best_metrics.band_0_40_pct,
            "band_40_60_pct": best_metrics.band_40_60_pct,
            "band_60_80_pct": best_metrics.band_60_80_pct,
            "band_80_100_pct": best_metrics.band_80_100_pct,
        },
    }


def _frange(start: float, stop: float, step: float) -> List[float]:
    """Generate a deterministic float range."""
    vals: List[float] = []
    v = start
    while v <= stop + 1e-9:
        vals.append(round(v, 6))
        v += step
    return vals


# ═══════════════════════════════════════════════════════════════════════
#  Section 6 — Versioning
# ═══════════════════════════════════════════════════════════════════════

def generate_calibration_output(
    calibration_result: Dict[str, Any],
    previous_version: str = CURRENT_FORMULA_VERSION,
) -> Dict[str, Any]:
    """Produce a versioned calibration report.

    Increments the minor version if weights changed from defaults,
    otherwise preserves the current version.
    """
    w = calibration_result["calibrated_weights"]
    changed = (
        w["balance_weight"] != DEFAULT_WEIGHTS.balance_weight
        or w["liquidity_weight"] != DEFAULT_WEIGHTS.liquidity_weight
        or w["coverage_weight"] != DEFAULT_WEIGHTS.coverage_weight
        or w["interest_weight"] != DEFAULT_WEIGHTS.interest_weight
    )

    # Parse version
    parts = previous_version.lstrip("v").split(".")
    major, minor = int(parts[0]), int(parts[1])
    if changed:
        new_version = f"v{major}.{minor + 1}.0"
    else:
        new_version = previous_version

    return {
        "new_version": new_version,
        "weights": w,
        "distribution_metrics": calibration_result["distribution_metrics"],
        "candidates_tested": calibration_result["candidates_tested"],
        "weights_changed": changed,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Example Usage
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json

    def _show(label: str, obj: Any) -> None:
        print(f"\n{'═' * 64}")
        print(f"  {label}")
        print(f"{'═' * 64}")
        if isinstance(obj, dict):
            print(json.dumps(obj, indent=2, default=str))
        else:
            print(obj)

    profiles = generate_synthetic_profiles()

    # ══════════════════════════════════════════════════════════════════
    # 1. BACKTEST WITH CURRENT WEIGHTS
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "─" * 64)
    print("  BACKTEST — CURRENT WEIGHTS (v1.0.0)")
    print("─" * 64)
    before_results = run_score_backtest(profiles, weights=DEFAULT_WEIGHTS)
    for r in before_results:
        flags_str = ", ".join(r.risk_flags) if r.risk_flags else "none"
        print(
            f"  {r.profile_id:<28s}  score={r.stability_score:6.2f}  "
            f"liq={r.liquidity_ratio:.2f}  cov={r.commitment_coverage:.2f}  "
            f"int_r={r.interest_ratio:.4f}  flags=[{flags_str}]"
        )

    before_metrics = analyze_distribution(before_results)
    _show("DISTRIBUTION — BEFORE CALIBRATION", {
        "mean": before_metrics.mean,
        "median": before_metrics.median,
        "min": before_metrics.min_score,
        "max": before_metrics.max_score,
        "std_dev": before_metrics.std_dev,
        "spread": before_metrics.spread,
        "band_0_40": f"{before_metrics.band_0_40_pct}%",
        "band_40_60": f"{before_metrics.band_40_60_pct}%",
        "band_60_80": f"{before_metrics.band_60_80_pct}%",
        "band_80_100": f"{before_metrics.band_80_100_pct}%",
    })

    criteria_before = check_distribution_criteria(before_metrics)
    _show("DISTRIBUTION CRITERIA — BEFORE", criteria_before)

    # ══════════════════════════════════════════════════════════════════
    # 2. MONOTONICITY TESTS — CURRENT WEIGHTS
    # ══════════════════════════════════════════════════════════════════

    mono_before = test_monotonicity(weights=DEFAULT_WEIGHTS)
    _show("MONOTONICITY — CURRENT WEIGHTS", {
        "all_passed": mono_before["all_passed"],
        "income": mono_before["tests"]["income_monotonicity"]["passed"],
        "expense": mono_before["tests"]["expense_monotonicity"]["passed"],
        "interest": mono_before["tests"]["interest_monotonicity"]["passed"],
        "commitment": mono_before["tests"]["commitment_monotonicity"]["passed"],
    })

    # ══════════════════════════════════════════════════════════════════
    # 3. WEIGHT CALIBRATION
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "═" * 64)
    print("  CALIBRATING (deterministic grid search)...")
    print("═" * 64)

    cal_result = calibrate_weights(DEFAULT_WEIGHTS, profiles)
    _show("CALIBRATION RESULT", cal_result)

    # ══════════════════════════════════════════════════════════════════
    # 4. BACKTEST WITH CALIBRATED WEIGHTS
    # ══════════════════════════════════════════════════════════════════

    new_w = ScoringWeights(**cal_result["calibrated_weights"])

    print("\n" + "─" * 64)
    print("  BACKTEST — CALIBRATED WEIGHTS")
    print("─" * 64)
    after_results = run_score_backtest(profiles, weights=new_w)
    for r in after_results:
        flags_str = ", ".join(r.risk_flags) if r.risk_flags else "none"
        print(
            f"  {r.profile_id:<28s}  score={r.stability_score:6.2f}  "
            f"liq={r.liquidity_ratio:.2f}  cov={r.commitment_coverage:.2f}  "
            f"int_r={r.interest_ratio:.4f}  flags=[{flags_str}]"
        )

    # ══════════════════════════════════════════════════════════════════
    # 5. BEFORE vs AFTER COMPARISON
    # ══════════════════════════════════════════════════════════════════

    print("\n" + "─" * 64)
    print("  BEFORE vs AFTER SCORE COMPARISON")
    print("─" * 64)
    print(f"  {'Profile':<28s}  {'Before':>8s}  {'After':>8s}  {'Delta':>8s}")
    print(f"  {'─'*28}  {'─'*8}  {'─'*8}  {'─'*8}")
    for b, a in zip(before_results, after_results):
        delta = a.stability_score - b.stability_score
        sign = "+" if delta >= 0 else ""
        print(
            f"  {b.profile_id:<28s}  {b.stability_score:8.2f}  "
            f"{a.stability_score:8.2f}  {sign}{delta:7.2f}"
        )

    # ══════════════════════════════════════════════════════════════════
    # 6. VERSIONED OUTPUT
    # ══════════════════════════════════════════════════════════════════

    versioned = generate_calibration_output(cal_result)
    _show("VERSIONED CALIBRATION OUTPUT", versioned)

    # ══════════════════════════════════════════════════════════════════
    # 7. MONOTONICITY — CALIBRATED WEIGHTS
    # ══════════════════════════════════════════════════════════════════

    mono_after = test_monotonicity(weights=new_w)
    _show("MONOTONICITY — CALIBRATED WEIGHTS", {
        "all_passed": mono_after["all_passed"],
        "income": mono_after["tests"]["income_monotonicity"]["passed"],
        "expense": mono_after["tests"]["expense_monotonicity"]["passed"],
        "interest": mono_after["tests"]["interest_monotonicity"]["passed"],
        "commitment": mono_after["tests"]["commitment_monotonicity"]["passed"],
    })
