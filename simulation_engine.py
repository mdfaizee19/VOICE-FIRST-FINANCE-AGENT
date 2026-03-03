"""
Deterministic Financial Simulation Engine — Pure Functions

Exposes two pure functions:
  - simulate_subscription_options(financial_state, subscription_options)
  - simulate_credit_strategies(financial_state, credit_strategy_request)

All math is deterministic: same inputs → identical outputs.
No randomness, no time dependency, no external services.

Formulas
--------
- monthly_interest    = credit_balance × (credit_apr / 12)
- projected_balance   = income − fixed − discretionary − extra_cost − interest − commitment_total
- liquidity_ratio     = max(projected_balance, 0) / max(total_obligations, 1)
- commitment_coverage = min(available_funds / commitment_total, 1.0) if commitments else 1.0
- stability_score     = 40% balance_health + 30% liquidity_health + 20% coverage_health + 10% interest_health
                        (each component is a 0–100 sub-score; composite clamped to [0, 100])

Risk flags
----------
- "high_interest"     → interest_cost > 10% × monthly_income
- "low_liquidity"     → liquidity_ratio < 0.5
- "commitment_breach" → commitment_coverage < 1.0
- "negative_balance"  → projected_balance < 0
"""

from __future__ import annotations

from typing import List

from financial_schemas import (
    CreditStrategyRequest,
    FinancialState,
    SubscriptionOption,
)
from simulation_schemas import SimulationResult, SimulationResultSet


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _monthly_interest(credit_balance: float, credit_apr: float) -> float:
    """Calculate one month's interest charge from APR."""
    return round(credit_balance * (credit_apr / 12.0), 2)


def _commitment_total(state: FinancialState) -> float:
    """Sum of all commitment amounts."""
    return round(sum(c.amount for c in state.commitments), 2)


def _risk_flags(
    interest_cost: float,
    monthly_income: float,
    liquidity_ratio: float,
    commitment_coverage: float,
    projected_balance: float,
) -> List[str]:
    """Determine which risk flags to raise."""
    flags: List[str] = []
    if monthly_income > 0 and interest_cost > 0.10 * monthly_income:
        flags.append("high_interest")
    if liquidity_ratio < 0.5:
        flags.append("low_liquidity")
    if commitment_coverage < 1.0:
        flags.append("commitment_breach")
    if projected_balance < 0:
        flags.append("negative_balance")
    return flags


def _stability_score(
    projected_balance: float,
    monthly_income: float,
    liquidity_ratio: float,
    commitment_coverage: float,
    interest_cost: float,
) -> float:
    """Compute weighted stability score in [0, 100].

    Components (each 0–100):
    - balance_health  (40%): how positive the projected balance is relative to income
    - liquidity_health(30%): liquidity ratio capped contribution
    - coverage_health (20%): commitment coverage × 100
    - interest_health (10%): inverse of interest burden
    """
    # Balance health: ratio of projected_balance to income, capped at 100
    if monthly_income > 0:
        balance_health = min(max((projected_balance / monthly_income) * 100, 0), 100)
    else:
        balance_health = 0.0 if projected_balance <= 0 else 100.0

    # Liquidity health: ratio capped at 2.0 → scaled to 100
    liquidity_health = min(max(liquidity_ratio / 2.0, 0), 1.0) * 100

    # Coverage health: direct percentage
    coverage_health = min(max(commitment_coverage, 0), 1.0) * 100

    # Interest health: 100 if no interest, decreasing as interest grows relative to income
    if monthly_income > 0:
        interest_ratio = interest_cost / monthly_income
        interest_health = max(100 - interest_ratio * 500, 0)
    else:
        interest_health = 0.0 if interest_cost > 0 else 100.0

    raw = (
        0.40 * balance_health
        + 0.30 * liquidity_health
        + 0.20 * coverage_health
        + 0.10 * interest_health
    )
    return round(min(max(raw, 0), 100), 2)


def _compute_result(
    option_id: str,
    state: FinancialState,
    extra_cost: float,
    interest: float,
) -> SimulationResult:
    """Build a SimulationResult from base state, an extra monthly cost, and interest."""
    commit_total = _commitment_total(state)
    total_outflows = (
        state.fixed_expenses
        + state.discretionary_expenses
        + extra_cost
        + interest
        + commit_total
    )
    projected_balance = round(state.monthly_income - total_outflows, 2)

    total_obligations = max(
        state.fixed_expenses + state.discretionary_expenses + extra_cost + commit_total,
        1.0,
    )
    liquidity_ratio = round(max(projected_balance, 0) / total_obligations, 2)

    if commit_total > 0:
        available = max(
            state.monthly_income
            - state.fixed_expenses
            - state.discretionary_expenses
            - extra_cost
            - interest,
            0,
        )
        commitment_coverage = round(min(available / commit_total, 1.0), 2)
    else:
        commitment_coverage = 1.0

    score = _stability_score(
        projected_balance,
        state.monthly_income,
        liquidity_ratio,
        commitment_coverage,
        interest,
    )

    flags = _risk_flags(
        interest, state.monthly_income, liquidity_ratio, commitment_coverage, projected_balance
    )

    return SimulationResult(
        option_id=option_id,
        stability_score=score,
        liquidity_ratio=liquidity_ratio,
        commitment_coverage=commitment_coverage,
        interest_cost=round(interest, 2),
        projected_balance=projected_balance,
        risk_flags=flags,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def simulate_subscription_options(
    financial_state: FinancialState,
    subscription_options: List[SubscriptionOption],
) -> SimulationResultSet:
    """Evaluate each subscription option against the user's financial state.

    Also includes a "no-subscription" baseline for comparison.
    Returns results sorted descending by stability_score.
    """
    interest = _monthly_interest(financial_state.credit_balance, financial_state.credit_apr)

    results: List[SimulationResult] = []

    # Baseline (no subscription)
    baseline = _compute_result("no-subscription", financial_state, 0.0, interest)
    results.append(baseline)

    for sub in subscription_options:
        result = _compute_result(sub.name, financial_state, sub.monthly_cost, interest)
        results.append(result)

    # Sort descending by stability_score
    results.sort(key=lambda r: r.stability_score, reverse=True)

    # Break potential score ties by adding tiny perturbation (deterministic)
    # to guarantee uniqueness required by SimulationResultSet
    seen_scores: set[float] = set()
    for r in results:
        while r.stability_score in seen_scores:
            # Use object.__setattr__ to mutate the frozen-ish strict model
            object.__setattr__(r, "stability_score", round(r.stability_score - 0.01, 2))
        seen_scores.add(r.stability_score)

    return SimulationResultSet(results=results)


def simulate_credit_strategies(
    financial_state: FinancialState,
    credit_strategy_request: CreditStrategyRequest,
) -> SimulationResultSet:
    """Evaluate credit payment strategies against the user's financial state.

    Simulates minimum, partial (25%), and full payment strategies.
    The *credit_strategy_request* selects which strategies to include,
    but for comparison purposes we always simulate all three.
    Returns results sorted descending by stability_score.
    """
    balance = financial_state.credit_balance
    apr = financial_state.credit_apr

    strategies: list[tuple[str, float]] = []

    # Minimum payment: pay only 2% of balance (industry standard minimum)
    min_payment = round(balance * 0.02, 2)
    min_interest = _monthly_interest(balance - min_payment, apr)
    strategies.append(("minimum", min_payment + min_interest))

    # Partial payment: pay partial_percentage of balance (default 25%)
    pct = credit_strategy_request.partial_percentage or 0.25
    partial_payment = round(balance * pct, 2)
    partial_interest = _monthly_interest(balance - partial_payment, apr)
    strategies.append(("partial", partial_payment + partial_interest))

    # Full payment: pay entire balance, no residual interest
    full_payment = balance
    full_interest = 0.0
    strategies.append(("full", full_payment + full_interest))

    results: List[SimulationResult] = []
    for name, extra_cost in strategies:
        interest = _monthly_interest(balance, apr) if name == "minimum" else (
            _monthly_interest(balance - partial_payment, apr) if name == "partial" else 0.0
        )
        result = _compute_result(name, financial_state, extra_cost, interest)
        results.append(result)

    # Sort descending by stability_score
    results.sort(key=lambda r: r.stability_score, reverse=True)

    # Guarantee uniqueness
    seen_scores: set[float] = set()
    for r in results:
        while r.stability_score in seen_scores:
            object.__setattr__(r, "stability_score", round(r.stability_score - 0.01, 2))
        seen_scores.add(r.stability_score)

    return SimulationResultSet(results=results)
