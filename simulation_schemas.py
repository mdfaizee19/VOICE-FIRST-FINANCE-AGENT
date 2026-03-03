"""
Simulation Output Schemas — Pydantic v2 Strict Output Contracts

Production-grade validated output models for financial simulation results.
All monetary values are floats.  All ratios are floats.

Models: SimulationResult → SimulationResultSet
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


# ---------------------------------------------------------------------------
# SimulationResult
# ---------------------------------------------------------------------------

class SimulationResult(BaseModel):
    """Output contract for a single simulation scenario.

    Represents the computed financial outcome of one option evaluated by
    the simulation engine.

    - *stability_score*: composite score (0–100) reflecting overall
      financial resilience under this option.
    - *liquidity_ratio*: ratio of liquid assets to near-term obligations;
      must be ≥ 0.
    - *commitment_coverage*: fraction (0–1) of known commitments that can
      be met without exceeding available funds.
    - *interest_cost*: projected monthly interest expense (≥ 0).
    - *projected_balance*: end-of-period balance; may be negative if the
      option leads to a deficit.
    - *risk_flags*: list of human-readable risk labels surfaced by the
      simulation (e.g. ``"high_utilization"``, ``"buffer_breach"``).
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    option_id: str
    stability_score: float
    liquidity_ratio: float
    commitment_coverage: float
    interest_cost: float
    projected_balance: float
    risk_flags: List[str] = []

    @field_validator("option_id")
    @classmethod
    def option_id_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError(
                "option_id must be a non-empty string; "
                "provide a unique identifier for this simulation option"
            )
        return v

    @field_validator("stability_score")
    @classmethod
    def stability_score_range(cls, v: float) -> float:
        if v < 0 or v > 100:
            raise ValueError(
                f"stability_score must be between 0 and 100 inclusive (got {v}); "
                "this is a normalised composite score"
            )
        return v

    @field_validator("liquidity_ratio")
    @classmethod
    def liquidity_ratio_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(
                f"liquidity_ratio must be >= 0 (got {v}); "
                "a negative liquidity ratio is not meaningful"
            )
        return v

    @field_validator("commitment_coverage")
    @classmethod
    def commitment_coverage_range(cls, v: float) -> float:
        if v < 0 or v > 1:
            raise ValueError(
                f"commitment_coverage must be between 0 and 1 inclusive (got {v}); "
                "express coverage as a decimal fraction"
            )
        return v

    @field_validator("interest_cost")
    @classmethod
    def interest_cost_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(
                f"interest_cost must be >= 0 (got {v}); "
                "interest expense cannot be negative"
            )
        return v


# ---------------------------------------------------------------------------
# SimulationResultSet
# ---------------------------------------------------------------------------

class SimulationResultSet(BaseModel):
    """Ordered collection of simulation results.

    Guarantees that results are:
    1. Non-empty.
    2. Sorted in **descending** order by *stability_score*.
    3. Free of duplicate *stability_score* values (ties are disallowed
       for deterministic ranking in the MVP).
    4. Capped at a maximum of 10 entries (guardrail against unbounded
       output).

    The validator **does not auto-sort**; if the caller provides results
    in the wrong order, a ``ValueError`` is raised.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    results: List[SimulationResult]

    @model_validator(mode="after")
    def validate_result_set(self) -> SimulationResultSet:
        results = self.results

        # ── Non-empty ─────────────────────────────────────────────────
        if len(results) == 0:
            raise ValueError(
                "results must not be empty; "
                "at least one SimulationResult is required"
            )

        # ── Max length guardrail ──────────────────────────────────────
        if len(results) > 10:
            raise ValueError(
                f"results must contain at most 10 entries (got {len(results)}); "
                "reduce the number of simulation options"
            )

        scores = [r.stability_score for r in results]

        # ── Uniqueness ────────────────────────────────────────────────
        if len(scores) != len(set(scores)):
            seen: set[float] = set()
            duplicates: list[float] = []
            for s in scores:
                if s in seen:
                    duplicates.append(s)
                seen.add(s)
            raise ValueError(
                f"stability_score values must be unique within the result set; "
                f"duplicate score(s) found: {duplicates}"
            )

        # ── Descending order ──────────────────────────────────────────
        for i in range(len(scores) - 1):
            if scores[i] <= scores[i + 1]:
                raise ValueError(
                    f"results must be sorted in descending order by stability_score; "
                    f"score at index {i} ({scores[i]}) is not greater than score "
                    f"at index {i + 1} ({scores[i + 1]}). "
                    f"Do NOT rely on auto-sorting — provide results pre-sorted"
                )

        return self


# ---------------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pydantic import ValidationError

    # ── Valid Result Set ──────────────────────────────────────────────

    result_a = SimulationResult(
        option_id="full-payoff",
        stability_score=87.5,
        liquidity_ratio=2.1,
        commitment_coverage=1.0,
        interest_cost=0.0,
        projected_balance=4200.0,
    )

    result_b = SimulationResult(
        option_id="partial-50",
        stability_score=72.3,
        liquidity_ratio=1.4,
        commitment_coverage=0.85,
        interest_cost=45.0,
        projected_balance=2800.0,
        risk_flags=["moderate_utilization"],
    )

    result_c = SimulationResult(
        option_id="minimum-only",
        stability_score=41.0,
        liquidity_ratio=0.6,
        commitment_coverage=0.55,
        interest_cost=120.0,
        projected_balance=-350.0,
        risk_flags=["high_utilization", "buffer_breach"],
    )

    result_set = SimulationResultSet(results=[result_a, result_b, result_c])
    print("✓ Valid SimulationResultSet:")
    print(result_set.model_dump_json(indent=2))

    # ── FAIL 1: Invalid stability_score (out of range) ───────────────

    try:
        SimulationResult(
            option_id="bad-score",
            stability_score=120.0,
            liquidity_ratio=1.0,
            commitment_coverage=0.5,
            interest_cost=10.0,
            projected_balance=1000.0,
        )
    except ValidationError as e:
        print("\n✗ FAIL 1 — stability_score out of range:")
        print(e)

    # ── FAIL 2: Invalid ordering (ascending instead of descending) ───

    try:
        SimulationResultSet(
            results=[
                SimulationResult(
                    option_id="low",
                    stability_score=30.0,
                    liquidity_ratio=1.0,
                    commitment_coverage=0.5,
                    interest_cost=10.0,
                    projected_balance=500.0,
                ),
                SimulationResult(
                    option_id="high",
                    stability_score=90.0,
                    liquidity_ratio=2.0,
                    commitment_coverage=1.0,
                    interest_cost=0.0,
                    projected_balance=5000.0,
                ),
            ]
        )
    except ValidationError as e:
        print("\n✗ FAIL 2 — Results not in descending order:")
        print(e)

    # ── FAIL 3: Invalid commitment_coverage (> 1) ────────────────────

    try:
        SimulationResult(
            option_id="bad-coverage",
            stability_score=50.0,
            liquidity_ratio=1.0,
            commitment_coverage=1.5,
            interest_cost=10.0,
            projected_balance=1000.0,
        )
    except ValidationError as e:
        print("\n✗ FAIL 3 — commitment_coverage out of range:")
        print(e)

    # ── FAIL 4 (commented): Duplicate stability scores ───────────────
    # try:
    #     SimulationResultSet(results=[
    #         SimulationResult(option_id="a", stability_score=80.0,
    #                          liquidity_ratio=1.0, commitment_coverage=0.9,
    #                          interest_cost=5.0, projected_balance=3000.0),
    #         SimulationResult(option_id="b", stability_score=80.0,
    #                          liquidity_ratio=1.2, commitment_coverage=0.8,
    #                          interest_cost=15.0, projected_balance=2500.0),
    #     ])
    # except ValidationError as e:
    #     print("\n✗ FAIL 4 — Duplicate stability scores:", e)

    # ── FAIL 5 (commented): Empty result set ─────────────────────────
    # try:
    #     SimulationResultSet(results=[])
    # except ValidationError as e:
    #     print("\n✗ FAIL 5 — Empty result set:", e)
