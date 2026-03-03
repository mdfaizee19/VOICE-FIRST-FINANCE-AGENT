"""
Simulation History Table — Financial Decision Engine (Phase 4)

Production-grade persistence layer designed for:
  - Behavioral modeling
  - Future ML training
  - Audit traceability
  - Longitudinal financial pattern analysis

Components
----------
- ``SimulationHistoryORM``          — SQLAlchemy model with CHECK constraints
- ``persist_simulation_history()``  — Validated, transactional insert
- ``get_user_simulation_history()`` — Paginated history query
- ``get_stability_trend()``         — Time-windowed stability scores
- ``get_high_risk_simulations()``   — Threshold-filtered risk query
- ``soft_delete_simulation_history()`` — Soft-delete only (no physical removal)

No simulation logic.  No orchestration.  No logging.  No API layer.
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, ConfigDict, ValidationError
from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    JSON,
    String,
    and_,
)
from sqlalchemy.orm import Session

from persistence import Base, _uuid_column


# ═══════════════════════════════════════════════════════════════════════
# Section 1 — SQLAlchemy ORM Model
# ═══════════════════════════════════════════════════════════════════════

class SimulationHistoryORM(Base):
    """Immutable record of a single simulation execution.

    Designed for longitudinal analysis and ML training.  After insert,
    only ``deleted_at`` may be mutated (soft delete).
    """

    __tablename__ = "simulation_history"

    # ── Identity ──────────────────────────────────────────────────────
    id = _uuid_column(primary_key=True, default=uuid.uuid4)
    user_id = _uuid_column(nullable=False, index=True)
    snapshot_id = _uuid_column(
        nullable=False,
        # ForeignKey omitted at Column level because _uuid_column
        # already creates the Column; we add FK via __table_args__.
    )
    correlation_id = Column(String(128), nullable=False, index=True)

    # ── Intent & Payloads ─────────────────────────────────────────────
    intent_type = Column(String(64), nullable=False, index=True)
    input_payload = Column(JSON, nullable=False)
    output_payload = Column(JSON, nullable=False)

    # ── Extracted Metrics (indexed / queryable) ───────────────────────
    stability_score = Column(Float, nullable=False, index=True)
    liquidity_ratio = Column(Float, nullable=False)
    commitment_coverage = Column(Float, nullable=False)
    interest_cost = Column(Float, nullable=False)
    risk_flags = Column(JSON, nullable=False, default=list)

    # ── Stress Testing (optional) ─────────────────────────────────────
    fragility_indicator = Column(Float, nullable=True)
    stress_test_result = Column(JSON, nullable=True)

    # ── Version Tracking ──────────────────────────────────────────────
    scoring_formula_version = Column(String(32), nullable=False)
    risk_model_version = Column(String(32), nullable=True)
    schema_version = Column(String(32), nullable=False)

    # ── Timestamps ────────────────────────────────────────────────────
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    # ── Constraints & Indexes ─────────────────────────────────────────
    __table_args__ = (
        # CHECK constraints (PostgreSQL-enforced; advisory on SQLite)
        CheckConstraint(
            "stability_score >= 0 AND stability_score <= 100",
            name="ck_stability_score_range",
        ),
        CheckConstraint(
            "liquidity_ratio >= 0",
            name="ck_liquidity_ratio_non_negative",
        ),
        CheckConstraint(
            "commitment_coverage >= 0 AND commitment_coverage <= 1",
            name="ck_commitment_coverage_range",
        ),
        CheckConstraint(
            "interest_cost >= 0",
            name="ck_interest_cost_non_negative",
        ),
        # Composite index for efficient user history queries
        Index("ix_history_user_created", "user_id", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<SimulationHistory id={self.id} user={self.user_id} "
            f"score={self.stability_score} intent={self.intent_type}>"
        )


# ═══════════════════════════════════════════════════════════════════════
# Structured Error Helper
# ═══════════════════════════════════════════════════════════════════════

def _error(message: str) -> Dict[str, Any]:
    return {
        "status": "error",
        "error_code": "INVALID_SIMULATION_HISTORY",
        "message": message,
    }


# ═══════════════════════════════════════════════════════════════════════
# Section 2 — Write Function
# ═══════════════════════════════════════════════════════════════════════

def persist_simulation_history(
    session: Session,
    user_id: uuid.UUID,
    snapshot_id: uuid.UUID,
    correlation_id: str,
    intent_type: str,
    input_model: BaseModel,
    output_model: BaseModel,
    scoring_formula_version: str,
    risk_model_version: str,
    schema_version: str,
    fragility_indicator: Optional[float] = None,
    stress_test_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Persist a validated simulation execution record.

    All numeric fields are checked for ``math.isfinite()`` before insert.
    The record is immutable after creation (only soft delete allowed).

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session (caller manages engine).
    input_model, output_model : BaseModel
        Validated Pydantic models — serialised to JSONB.
    fragility_indicator : float | None
        Optional stress-test fragility index.
    stress_test_result : dict | None
        Optional full stress-test payload.

    Returns
    -------
    dict
        ``{"status": "success", "record": {...}}`` or
        ``{"status": "error", ...}``
    """
    # ── Validate version strings ──────────────────────────────────────
    if not schema_version or not schema_version.strip():
        return _error("schema_version must be a non-empty string")
    if not scoring_formula_version or not scoring_formula_version.strip():
        return _error("scoring_formula_version must be a non-empty string")

    # ── Serialise models ──────────────────────────────────────────────
    try:
        input_json = input_model.model_dump()
    except Exception as exc:
        return _error(f"Failed to serialise input_model: {exc}")

    try:
        output_json = output_model.model_dump()
    except Exception as exc:
        return _error(f"Failed to serialise output_model: {exc}")

    # ── Extract metrics from output ───────────────────────────────────
    stability_score = output_json.get("stability_score")
    liquidity_ratio = output_json.get("liquidity_ratio")
    commitment_coverage = output_json.get("commitment_coverage")
    interest_cost = output_json.get("interest_cost")
    risk_flags = output_json.get("risk_flags", [])

    # ── Numeric validation ────────────────────────────────────────────
    checks = {
        "stability_score": stability_score,
        "liquidity_ratio": liquidity_ratio,
        "commitment_coverage": commitment_coverage,
        "interest_cost": interest_cost,
    }
    for field_name, value in checks.items():
        if value is None:
            return _error(f"Missing required field: {field_name}")
        if not isinstance(value, (int, float)):
            return _error(f"{field_name} must be numeric (got {type(value).__name__})")
        if not math.isfinite(value):
            return _error(f"{field_name} is not finite: {value}")

    # ── Range validation ──────────────────────────────────────────────
    if stability_score < 0 or stability_score > 100:
        return _error(
            f"stability_score must be 0–100 (got {stability_score})"
        )
    if liquidity_ratio < 0:
        return _error(
            f"liquidity_ratio must be >= 0 (got {liquidity_ratio})"
        )
    if commitment_coverage < 0 or commitment_coverage > 1:
        return _error(
            f"commitment_coverage must be 0–1 (got {commitment_coverage})"
        )
    if interest_cost < 0:
        return _error(
            f"interest_cost must be >= 0 (got {interest_cost})"
        )

    # ── Fragility validation (if provided) ────────────────────────────
    if fragility_indicator is not None:
        if not math.isfinite(fragility_indicator):
            return _error(
                f"fragility_indicator is not finite: {fragility_indicator}"
            )

    # ── Persist ───────────────────────────────────────────────────────
    try:
        now = datetime.now(timezone.utc)
        record = SimulationHistoryORM(
            id=uuid.uuid4(),
            user_id=user_id,
            snapshot_id=snapshot_id,
            correlation_id=correlation_id,
            intent_type=intent_type,
            input_payload=input_json,
            output_payload=output_json,
            stability_score=stability_score,
            liquidity_ratio=liquidity_ratio,
            commitment_coverage=commitment_coverage,
            interest_cost=interest_cost,
            risk_flags=risk_flags,
            fragility_indicator=fragility_indicator,
            stress_test_result=stress_test_result,
            scoring_formula_version=scoring_formula_version,
            risk_model_version=risk_model_version,
            schema_version=schema_version,
            created_at=now,
        )
        session.add(record)
        session.flush()

        result = {
            "status": "success",
            "record": _serialise_record(record),
        }
        session.commit()
        return result

    except Exception as exc:
        session.rollback()
        return _error(f"Persistence failed: {exc}")


# ═══════════════════════════════════════════════════════════════════════
# Section 3 — Query Helpers (Behavioral Modeling)
# ═══════════════════════════════════════════════════════════════════════

def get_user_simulation_history(
    session: Session,
    user_id: uuid.UUID,
    limit: int = 50,
) -> Dict[str, Any]:
    """Retrieve a user's simulation history (most recent first).

    Excludes soft-deleted records.  Uses ``(user_id, created_at)`` index.
    """
    try:
        records = (
            session.query(SimulationHistoryORM)
            .filter(
                SimulationHistoryORM.user_id == user_id,
                SimulationHistoryORM.deleted_at.is_(None),
            )
            .order_by(SimulationHistoryORM.created_at.desc())
            .limit(limit)
            .all()
        )
        return {
            "status": "success",
            "count": len(records),
            "records": [_serialise_record(r) for r in records],
        }
    except Exception as exc:
        return _error(f"Query failed: {exc}")


def get_stability_trend(
    session: Session,
    user_id: uuid.UUID,
    days: int = 30,
) -> Dict[str, Any]:
    """Retrieve stability scores within a time window for trend analysis.

    Returns ``(created_at, stability_score, intent_type)`` tuples
    ordered chronologically (ascending) for charting / ML features.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    try:
        records = (
            session.query(SimulationHistoryORM)
            .filter(
                SimulationHistoryORM.user_id == user_id,
                SimulationHistoryORM.deleted_at.is_(None),
                SimulationHistoryORM.created_at >= cutoff,
            )
            .order_by(SimulationHistoryORM.created_at.asc())
            .all()
        )
        trend = [
            {
                "timestamp": r.created_at.isoformat() if r.created_at else None,
                "stability_score": r.stability_score,
                "intent_type": r.intent_type,
            }
            for r in records
        ]
        return {
            "status": "success",
            "days": days,
            "data_points": len(trend),
            "trend": trend,
        }
    except Exception as exc:
        return _error(f"Trend query failed: {exc}")


def get_high_risk_simulations(
    session: Session,
    user_id: uuid.UUID,
    threshold: float = 50.0,
) -> Dict[str, Any]:
    """Retrieve simulations where stability_score is below a threshold.

    Useful for identifying high-risk patterns across a user's history.
    """
    try:
        records = (
            session.query(SimulationHistoryORM)
            .filter(
                SimulationHistoryORM.user_id == user_id,
                SimulationHistoryORM.deleted_at.is_(None),
                SimulationHistoryORM.stability_score < threshold,
            )
            .order_by(SimulationHistoryORM.created_at.desc())
            .all()
        )
        return {
            "status": "success",
            "threshold": threshold,
            "count": len(records),
            "records": [_serialise_record(r) for r in records],
        }
    except Exception as exc:
        return _error(f"High-risk query failed: {exc}")


# ═══════════════════════════════════════════════════════════════════════
# Section 4 — Soft Delete
# ═══════════════════════════════════════════════════════════════════════

def soft_delete_simulation_history(
    session: Session,
    record_id: uuid.UUID,
    user_id: uuid.UUID,
) -> Dict[str, Any]:
    """Soft-delete a simulation history record.

    Sets ``deleted_at`` — never physically removes the row.
    Rejects if already deleted.
    """
    try:
        record = (
            session.query(SimulationHistoryORM)
            .filter(
                SimulationHistoryORM.id == record_id,
                SimulationHistoryORM.user_id == user_id,
            )
            .first()
        )

        if record is None:
            return {
                "status": "error",
                "error_code": "NOT_FOUND",
                "message": f"Record {record_id} not found for user {user_id}",
            }

        if record.deleted_at is not None:
            return {
                "status": "error",
                "error_code": "ALREADY_DELETED",
                "message": f"Record {record_id} is already soft-deleted",
            }

        record.deleted_at = datetime.now(timezone.utc)
        session.flush()
        result = {
            "status": "success",
            "record": _serialise_record(record),
        }
        session.commit()
        return result

    except Exception as exc:
        session.rollback()
        return _error(f"Soft-delete failed: {exc}")


# ═══════════════════════════════════════════════════════════════════════
# Serialisation Helper
# ═══════════════════════════════════════════════════════════════════════

def _serialise_record(r: SimulationHistoryORM) -> Dict[str, Any]:
    """Convert an ORM record to a JSON-serialisable dict."""
    return {
        "id": str(r.id),
        "user_id": str(r.user_id),
        "snapshot_id": str(r.snapshot_id),
        "correlation_id": r.correlation_id,
        "intent_type": r.intent_type,
        "stability_score": r.stability_score,
        "liquidity_ratio": r.liquidity_ratio,
        "commitment_coverage": r.commitment_coverage,
        "interest_cost": r.interest_cost,
        "risk_flags": r.risk_flags,
        "fragility_indicator": r.fragility_indicator,
        "scoring_formula_version": r.scoring_formula_version,
        "risk_model_version": r.risk_model_version,
        "schema_version": r.schema_version,
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "deleted_at": r.deleted_at.isoformat() if r.deleted_at else None,
    }


# ═══════════════════════════════════════════════════════════════════════
# Example Usage (SQLite in-memory)
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from financial_schemas import Commitment, FinancialState
    from simulation_schemas import SimulationResult

    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)

    def _show(label: str, obj: Dict[str, Any]) -> None:
        printable = {k: v for k, v in obj.items()
                     if not isinstance(v, BaseModel)}
        print(f"\n{'─' * 60}")
        print(f"  {label}")
        print(f"{'─' * 60}")
        print(json.dumps(printable, indent=2, default=str))

    user = uuid.uuid4()
    snap_id = uuid.uuid4()

    state = FinancialState(
        monthly_income=5000.0,
        fixed_expenses=2000.0,
        discretionary_expenses=800.0,
        emergency_fund=10000.0,
        credit_balance=3000.0,
        credit_apr=0.24,
        commitments=[Commitment(id="ins", amount=300.0, due_month=3)],
    )

    result = SimulationResult(
        option_id="baseline",
        stability_score=57.92,
        liquidity_ratio=0.76,
        commitment_coverage=1.0,
        interest_cost=60.0,
        projected_balance=2140.0,
        risk_flags=[],
    )

    # ══════════════════════════════════════════════════════════════════
    # 1. VALID INSERT
    # ══════════════════════════════════════════════════════════════════

    with SessionLocal() as sess:
        res = persist_simulation_history(
            session=sess,
            user_id=user,
            snapshot_id=snap_id,
            correlation_id="hist-001",
            intent_type="subscription_purchase",
            input_model=state,
            output_model=result,
            scoring_formula_version="v1.0.0",
            risk_model_version="v1.0.0",
            schema_version="v2.0.0",
            fragility_indicator=11.54,
            stress_test_result={"baseline_score": 57.92, "risk_level": "moderate"},
        )
        _show("VALID INSERT", res)

    # Insert a second record (low score) for querying
    low_result = SimulationResult(
        option_id="high-cost",
        stability_score=25.0,
        liquidity_ratio=0.1,
        commitment_coverage=0.6,
        interest_cost=200.0,
        projected_balance=-500.0,
        risk_flags=["low_liquidity", "negative_balance"],
    )

    with SessionLocal() as sess:
        persist_simulation_history(
            session=sess,
            user_id=user,
            snapshot_id=snap_id,
            correlation_id="hist-002",
            intent_type="credit_payment",
            input_model=state,
            output_model=low_result,
            scoring_formula_version="v1.0.0",
            risk_model_version="v1.0.0",
            schema_version="v2.0.0",
        )

    # ══════════════════════════════════════════════════════════════════
    # 2. QUERY: USER HISTORY
    # ══════════════════════════════════════════════════════════════════

    with SessionLocal() as sess:
        history = get_user_simulation_history(sess, user, limit=10)
        _show("USER HISTORY", history)

    # ══════════════════════════════════════════════════════════════════
    # 3. QUERY: HIGH-RISK SIMULATIONS (score < 50)
    # ══════════════════════════════════════════════════════════════════

    with SessionLocal() as sess:
        high_risk = get_high_risk_simulations(sess, user, threshold=50.0)
        _show("HIGH-RISK SIMULATIONS", high_risk)

    # ══════════════════════════════════════════════════════════════════
    # 4. QUERY: STABILITY TREND (last 30 days)
    # ══════════════════════════════════════════════════════════════════

    with SessionLocal() as sess:
        trend = get_stability_trend(sess, user, days=30)
        _show("STABILITY TREND", trend)

    # ══════════════════════════════════════════════════════════════════
    # 5. REJECTION: INVALID SCORE
    # ══════════════════════════════════════════════════════════════════

    bad_result = SimulationResult(
        option_id="broken",
        stability_score=100.0,  # will be manually overridden
        liquidity_ratio=0.5,
        commitment_coverage=0.5,
        interest_cost=10.0,
        projected_balance=500.0,
        risk_flags=[],
    )
    # Override to create an invalid output dict
    bad_output = bad_result.model_dump()
    bad_output["stability_score"] = 150.0  # out of range

    class _FakeOutput(BaseModel):
        model_config = ConfigDict(extra="allow")
        stability_score: float = 150.0
        liquidity_ratio: float = 0.5
        commitment_coverage: float = 0.5
        interest_cost: float = 10.0
        risk_flags: list = []
        option_id: str = "broken"
        projected_balance: float = 500.0

    with SessionLocal() as sess:
        rejected = persist_simulation_history(
            session=sess,
            user_id=user,
            snapshot_id=snap_id,
            correlation_id="hist-bad",
            intent_type="test",
            input_model=state,
            output_model=_FakeOutput(),
            scoring_formula_version="v1.0.0",
            risk_model_version="v1.0.0",
            schema_version="v2.0.0",
        )
        _show("REJECTION: INVALID SCORE", rejected)

    # ══════════════════════════════════════════════════════════════════
    # 6. SOFT DELETE
    # ══════════════════════════════════════════════════════════════════

    with SessionLocal() as sess:
        recs = get_user_simulation_history(sess, user, limit=1)
        if recs["count"] > 0:
            rec_id = uuid.UUID(recs["records"][0]["id"])
            deleted = soft_delete_simulation_history(sess, rec_id, user)
            _show("SOFT DELETE", deleted)

            # Verify it no longer appears in active history
            after = get_user_simulation_history(sess, user, limit=10)
            _show("HISTORY AFTER SOFT DELETE", after)
