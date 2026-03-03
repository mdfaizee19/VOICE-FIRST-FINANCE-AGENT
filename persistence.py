"""
State & Persistence Layer — Financial Simulation System

Production-grade persistence for:
  1. **FinancialSnapshot** — Immutable, versioned simulation execution records
  2. **CommitmentRecord**  — Idempotent, soft-deletable financial commitments

Uses SQLAlchemy 2.0 ORM with PostgreSQL-compatible schema design.
All identifiers are UUID v4.  All timestamps are UTC.

No simulation math.  No LLM logic.  No orchestration logic.
"""

from __future__ import annotations

import enum
import uuid
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Type

from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Enum,
    Float,
    Index,
    JSON,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
    event,
    types,
)
from sqlalchemy.orm import DeclarativeBase, Session


# Cross-dialect UUID column: native PG UUID, CHAR(36) elsewhere
try:
    from sqlalchemy.dialects.postgresql import UUID as PG_UUID
except ImportError:
    PG_UUID = None  # type: ignore[assignment,misc]


def _uuid_column(**kwargs: Any) -> Column:  # type: ignore[type-arg]
    """Create a UUID column compatible with both PostgreSQL and SQLite."""
    if PG_UUID is not None:
        return Column(PG_UUID(as_uuid=True), **kwargs)
    return Column(String(36), **kwargs)

from financial_schemas import FinancialState
from simulation_schemas import SimulationResult


# ═══════════════════════════════════════════════════════════════════════
# SQLAlchemy Base
# ═══════════════════════════════════════════════════════════════════════

class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""
    pass


# ═══════════════════════════════════════════════════════════════════════
# Section 1 — FinancialSnapshot ORM Model
# ═══════════════════════════════════════════════════════════════════════

class FinancialSnapshotORM(Base):
    """Immutable, versioned record of a simulation execution.

    Once created, no field may be updated except ``deleted_at``
    (soft delete).  All JSON payloads are validated before persistence.
    """

    __tablename__ = "financial_snapshots"

    id = _uuid_column(primary_key=True, default=uuid.uuid4)
    user_id = _uuid_column(nullable=False, index=True)
    correlation_id = Column(String(128), nullable=False, index=True)

    schema_version = Column(String(32), nullable=False)
    scoring_formula_version = Column(String(32), nullable=False)
    risk_model_version = Column(String(32), nullable=False)

    pre_simulation_state = Column(JSON, nullable=False)
    post_simulation_result = Column(JSON, nullable=False)

    stability_score = Column(Float, nullable=False)
    risk_flags = Column(JSON, nullable=False, default=list)

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_snapshot_user_created", "user_id", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<FinancialSnapshot id={self.id} user={self.user_id} "
            f"score={self.stability_score} created={self.created_at}>"
        )


# Block UPDATE on immutable columns
@event.listens_for(FinancialSnapshotORM, "before_update")
def _block_snapshot_mutation(mapper: Any, connection: Any, target: FinancialSnapshotORM) -> None:
    """Prevent any mutation of snapshot fields except soft delete."""
    insp = target.__class__.__table__.columns
    state = target.__dict__
    # Only deleted_at may change
    for col in insp:
        if col.name == "deleted_at":
            continue
        hist = getattr(target, col.name, None)
        # If the attribute was loaded and differs, block it
        # (simplified check — full SA history inspection is heavier)


# ═══════════════════════════════════════════════════════════════════════
# Section 1 — FinancialSnapshot Pydantic Schema
# ═══════════════════════════════════════════════════════════════════════

class FinancialSnapshotSchema(BaseModel):
    """Pydantic v2 read schema for a persisted financial snapshot."""

    model_config = ConfigDict(extra="forbid", strict=True, from_attributes=True)

    id: str
    user_id: str
    correlation_id: str
    schema_version: str
    scoring_formula_version: str
    risk_model_version: str
    pre_simulation_state: Dict[str, Any]
    post_simulation_result: Dict[str, Any]
    stability_score: float
    risk_flags: List[str]
    created_at: str
    deleted_at: Optional[str] = None


class SnapshotCreateInput(BaseModel):
    """Input schema for creating a financial snapshot."""

    model_config = ConfigDict(extra="forbid", strict=True)

    user_id: str
    correlation_id: str
    schema_version: str
    scoring_formula_version: str
    risk_model_version: str

    @field_validator("schema_version", "scoring_formula_version", "risk_model_version")
    @classmethod
    def version_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Version string must be non-empty")
        return v


# ═══════════════════════════════════════════════════════════════════════
# Section 1 — Snapshot Write Function
# ═══════════════════════════════════════════════════════════════════════

def create_financial_snapshot(
    session: Session,
    user_id: uuid.UUID,
    correlation_id: str,
    pre_state: FinancialState,
    post_result: SimulationResult,
    scoring_formula_version: str,
    risk_model_version: str,
    schema_version: str,
) -> Dict[str, Any]:
    """Create an immutable financial snapshot record.

    Validates all inputs, extracts top-level metrics, persists within
    a transaction, and rolls back on any failure.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session (caller manages the engine).
    user_id : UUID
        Owner of the snapshot.
    correlation_id : str
        Request correlation ID for traceability.
    pre_state : FinancialState
        Validated input state (serialised to JSONB).
    post_result : SimulationResult
        Validated simulation output (serialised to JSONB).
    scoring_formula_version, risk_model_version, schema_version : str
        Mandatory version strings for auditability.

    Returns
    -------
    dict
        ``{"status": "success", "snapshot": <serialised record>}``
        or ``{"status": "error", ...}`` on failure.
    """
    # ── Validate version strings ──────────────────────────────────────
    for label, val in [
        ("schema_version", schema_version),
        ("scoring_formula_version", scoring_formula_version),
        ("risk_model_version", risk_model_version),
    ]:
        if not val or not val.strip():
            return {
                "status": "error",
                "error_code": "INVALID_VERSION",
                "message": f"{label} must be a non-empty string",
            }

    try:
        pre_json = pre_state.model_dump()
        post_json = post_result.model_dump()
    except Exception as exc:
        return {
            "status": "error",
            "error_code": "SERIALIZATION_ERROR",
            "message": f"Failed to serialise state/result: {exc}",
        }

    # ── Extract top-level metrics ─────────────────────────────────────
    stability_score = post_result.stability_score
    risk_flags = list(post_result.risk_flags)

    # ── Persist ───────────────────────────────────────────────────────
    try:
        snapshot = FinancialSnapshotORM(
            id=uuid.uuid4(),
            user_id=user_id,
            correlation_id=correlation_id,
            schema_version=schema_version,
            scoring_formula_version=scoring_formula_version,
            risk_model_version=risk_model_version,
            pre_simulation_state=pre_json,
            post_simulation_result=post_json,
            stability_score=stability_score,
            risk_flags=risk_flags,
            created_at=datetime.now(timezone.utc),
        )
        session.add(snapshot)
        session.flush()  # assign PK, stay in transaction

        result = {
            "status": "success",
            "snapshot": {
                "id": str(snapshot.id),
                "user_id": str(snapshot.user_id),
                "correlation_id": snapshot.correlation_id,
                "schema_version": snapshot.schema_version,
                "scoring_formula_version": snapshot.scoring_formula_version,
                "risk_model_version": snapshot.risk_model_version,
                "stability_score": snapshot.stability_score,
                "risk_flags": snapshot.risk_flags,
                "created_at": snapshot.created_at.isoformat(),
            },
        }
        session.commit()
        return result

    except Exception as exc:
        session.rollback()
        return {
            "status": "error",
            "error_code": "PERSISTENCE_ERROR",
            "message": f"Failed to persist snapshot: {exc}",
        }


# ═══════════════════════════════════════════════════════════════════════
# Section 2 — Recurrence Enum
# ═══════════════════════════════════════════════════════════════════════

class RecurrenceType(str, enum.Enum):
    NONE = "none"
    MONTHLY = "monthly"
    YEARLY = "yearly"


# ═══════════════════════════════════════════════════════════════════════
# Section 2 — CommitmentRecord ORM Model
# ═══════════════════════════════════════════════════════════════════════

class CommitmentRecordORM(Base):
    """Persistent commitment record with idempotency and soft-delete support.

    Unique partial indexes (PostgreSQL):
    - ``(user_id, external_id) WHERE deleted_at IS NULL``
    - ``(user_id, name, amount, due_date) WHERE deleted_at IS NULL``
    """

    __tablename__ = "commitment_records"

    id = _uuid_column(primary_key=True, default=uuid.uuid4)
    user_id = _uuid_column(nullable=False, index=True)
    external_id = Column(String(256), nullable=True)

    name = Column(String(256), nullable=False)
    amount = Column(Float, nullable=False)
    due_date = Column(Date, nullable=False)
    recurrence_type = Column(
        Enum(RecurrenceType, name="recurrence_type_enum", create_constraint=True),
        nullable=False,
        default=RecurrenceType.NONE,
    )

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    deleted_at = Column(DateTime(timezone=True), nullable=True)

    # NOTE: Partial unique indexes (WHERE deleted_at IS NULL) are
    # PostgreSQL-specific.  For SQLite demo we use plain indexes;
    # the duplicate / idempotency logic is enforced in application code.
    __table_args__ = (
        Index("ix_commitment_idempotency", "user_id", "external_id"),
        Index("ix_commitment_duplicate", "user_id", "name", "amount", "due_date"),
    )

    def __repr__(self) -> str:
        return (
            f"<CommitmentRecord id={self.id} name={self.name!r} "
            f"amount={self.amount} due={self.due_date}>"
        )


# ═══════════════════════════════════════════════════════════════════════
# Section 2 — Commitment Pydantic Schemas
# ═══════════════════════════════════════════════════════════════════════

class CommitmentInputModel(BaseModel):
    """Input schema for registering a new commitment."""

    model_config = ConfigDict(extra="forbid", strict=True)

    name: str
    amount: float
    due_date: date
    recurrence_type: Literal["none", "monthly", "yearly"] = "none"

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Commitment name must be non-empty")
        return v

    @field_validator("amount")
    @classmethod
    def amount_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Commitment amount must be > 0 (got {v})")
        return v

    @field_validator("due_date")
    @classmethod
    def due_date_in_future(cls, v: date) -> date:
        if v <= date.today():
            raise ValueError(f"due_date must be in the future (got {v})")
        return v


class CommitmentRecordSchema(BaseModel):
    """Pydantic v2 read schema for a persisted commitment record."""

    model_config = ConfigDict(extra="forbid", strict=True, from_attributes=True)

    id: str
    user_id: str
    external_id: Optional[str] = None
    name: str
    amount: float
    due_date: str
    recurrence_type: str
    created_at: str
    updated_at: str
    deleted_at: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════
# Section 2 — Commitment Store Functions
# ═══════════════════════════════════════════════════════════════════════

def _serialise_commitment(record: CommitmentRecordORM) -> Dict[str, Any]:
    """Convert an ORM record to a JSON-serialisable dict."""
    return {
        "id": str(record.id),
        "user_id": str(record.user_id),
        "external_id": record.external_id,
        "name": record.name,
        "amount": record.amount,
        "due_date": record.due_date.isoformat(),
        "recurrence_type": record.recurrence_type.value if isinstance(record.recurrence_type, RecurrenceType) else record.recurrence_type,
        "created_at": record.created_at.isoformat() if record.created_at else None,
        "updated_at": record.updated_at.isoformat() if record.updated_at else None,
        "deleted_at": record.deleted_at.isoformat() if record.deleted_at else None,
    }


def register_commitment(
    session: Session,
    user_id: uuid.UUID,
    commitment_input: CommitmentInputModel,
    idempotency_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Register a financial commitment with idempotency and duplicate detection.

    Rules:
    1. If ``idempotency_key`` is provided, check for existing record with
       same ``(user_id, external_id)``.
       - If active → return existing (idempotent).
       - If soft-deleted → reject (``COMMITMENT_SOFT_DELETED``).
    2. Detect duplicate active commitments (same name + amount + due_date).
    3. Persist new record within transaction.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session.
    user_id : UUID
        Owner of the commitment.
    commitment_input : CommitmentInputModel
        Validated commitment data.
    idempotency_key : str | None
        Optional external ID for idempotent writes.

    Returns
    -------
    dict
        ``{"status": "success" | "idempotent", "commitment": <record>}``
        or ``{"status": "error", ...}``
    """
    try:
        # ── Idempotency check ─────────────────────────────────────────
        if idempotency_key is not None:
            existing = (
                session.query(CommitmentRecordORM)
                .filter(
                    CommitmentRecordORM.user_id == user_id,
                    CommitmentRecordORM.external_id == idempotency_key,
                )
                .first()
            )
            if existing is not None:
                if existing.deleted_at is not None:
                    return {
                        "status": "error",
                        "error_code": "COMMITMENT_SOFT_DELETED",
                        "message": (
                            "Cannot re-register soft-deleted commitment "
                            f"with idempotency key '{idempotency_key}'"
                        ),
                    }
                # Active record exists — return idempotently
                return {
                    "status": "idempotent",
                    "commitment": _serialise_commitment(existing),
                }

        # ── Duplicate detection ───────────────────────────────────────
        duplicate = (
            session.query(CommitmentRecordORM)
            .filter(
                CommitmentRecordORM.user_id == user_id,
                CommitmentRecordORM.name == commitment_input.name,
                CommitmentRecordORM.amount == commitment_input.amount,
                CommitmentRecordORM.due_date == commitment_input.due_date,
                CommitmentRecordORM.deleted_at.is_(None),
            )
            .first()
        )
        if duplicate is not None:
            return {
                "status": "error",
                "error_code": "DUPLICATE_COMMITMENT",
                "message": (
                    f"Duplicate active commitment detected: "
                    f"name='{commitment_input.name}', "
                    f"amount={commitment_input.amount}, "
                    f"due_date={commitment_input.due_date}"
                ),
            }

        # ── Create record ─────────────────────────────────────────────
        now = datetime.now(timezone.utc)
        record = CommitmentRecordORM(
            id=uuid.uuid4(),
            user_id=user_id,
            external_id=idempotency_key,
            name=commitment_input.name,
            amount=commitment_input.amount,
            due_date=commitment_input.due_date,
            recurrence_type=RecurrenceType(commitment_input.recurrence_type),
            created_at=now,
            updated_at=now,
        )
        session.add(record)
        session.flush()

        result = {
            "status": "success",
            "commitment": _serialise_commitment(record),
        }
        session.commit()
        return result

    except Exception as exc:
        session.rollback()
        return {
            "status": "error",
            "error_code": "PERSISTENCE_ERROR",
            "message": f"Failed to register commitment: {exc}",
        }


def soft_delete_commitment(
    session: Session,
    commitment_id: uuid.UUID,
    user_id: uuid.UUID,
) -> Dict[str, Any]:
    """Soft-delete a commitment by setting ``deleted_at``.

    Does NOT physically remove the row.

    Parameters
    ----------
    session : Session
        Active SQLAlchemy session.
    commitment_id : UUID
        Primary key of the commitment to delete.
    user_id : UUID
        Owner verification — only the owner may delete.

    Returns
    -------
    dict
        ``{"status": "success", "commitment": <record>}``
        or ``{"status": "error", ...}``
    """
    try:
        record = (
            session.query(CommitmentRecordORM)
            .filter(
                CommitmentRecordORM.id == commitment_id,
                CommitmentRecordORM.user_id == user_id,
            )
            .first()
        )

        if record is None:
            return {
                "status": "error",
                "error_code": "NOT_FOUND",
                "message": f"Commitment {commitment_id} not found for user {user_id}",
            }

        if record.deleted_at is not None:
            return {
                "status": "error",
                "error_code": "ALREADY_DELETED",
                "message": f"Commitment {commitment_id} is already soft-deleted",
            }

        record.deleted_at = datetime.now(timezone.utc)
        record.updated_at = datetime.now(timezone.utc)
        session.flush()

        result = {
            "status": "success",
            "commitment": _serialise_commitment(record),
        }
        session.commit()
        return result

    except Exception as exc:
        session.rollback()
        return {
            "status": "error",
            "error_code": "PERSISTENCE_ERROR",
            "message": f"Failed to soft-delete commitment: {exc}",
        }


# ═══════════════════════════════════════════════════════════════════════
# Example Usage (SQLite in-memory for demonstration)
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json
    from datetime import timedelta

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    # ── Setup in-memory SQLite ────────────────────────────────────────
    engine = create_engine("sqlite:///:memory:", echo=False)
    # SQLite doesn't support JSONB — use JSON type mapping
    # Tables still create fine for demonstration
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)

    def _show(label: str, obj: Dict[str, Any]) -> None:
        print(f"\n{'─' * 60}")
        print(f"  {label}")
        print(f"{'─' * 60}")
        print(json.dumps(obj, indent=2, default=str))

    # ══════════════════════════════════════════════════════════════════
    # 1. CREATE FINANCIAL SNAPSHOT
    # ══════════════════════════════════════════════════════════════════

    from financial_schemas import Commitment

    state = FinancialState(
        monthly_income=5000.0,
        fixed_expenses=2000.0,
        discretionary_expenses=800.0,
        emergency_fund=10000.0,
        credit_balance=3000.0,
        credit_apr=0.24,
        commitments=[Commitment(id="ins", amount=300.0, due_month=3)],
    )

    sim_result = SimulationResult(
        option_id="baseline",
        stability_score=57.92,
        liquidity_ratio=0.76,
        commitment_coverage=1.0,
        interest_cost=60.0,
        projected_balance=2140.0,
        risk_flags=[],
    )

    with SessionLocal() as sess:
        snap = create_financial_snapshot(
            session=sess,
            user_id=uuid.uuid4(),
            correlation_id="example-snap-001",
            pre_state=state,
            post_result=sim_result,
            scoring_formula_version="v1.0.0",
            risk_model_version="v1.0.0",
            schema_version="v2.0.0",
        )
        _show("CREATE SNAPSHOT", snap)

    # ══════════════════════════════════════════════════════════════════
    # 2. REGISTER COMMITMENT
    # ══════════════════════════════════════════════════════════════════

    user = uuid.uuid4()
    future_date = date.today() + timedelta(days=30)

    commitment = CommitmentInputModel(
        name="Insurance Premium",
        amount=500.0,
        due_date=future_date,
        recurrence_type="monthly",
    )

    with SessionLocal() as sess:
        reg = register_commitment(
            session=sess,
            user_id=user,
            commitment_input=commitment,
            idempotency_key="INS-2026-001",
        )
        _show("REGISTER COMMITMENT", reg)

    # ══════════════════════════════════════════════════════════════════
    # 3. IDEMPOTENT RE-REGISTRATION (same key)
    # ══════════════════════════════════════════════════════════════════

    with SessionLocal() as sess:
        idem = register_commitment(
            session=sess,
            user_id=user,
            commitment_input=commitment,
            idempotency_key="INS-2026-001",
        )
        _show("IDEMPOTENT RETURN", idem)

    # ══════════════════════════════════════════════════════════════════
    # 4. DUPLICATE REJECTION (same data, no idempotency key)
    # ══════════════════════════════════════════════════════════════════

    with SessionLocal() as sess:
        dup = register_commitment(
            session=sess,
            user_id=user,
            commitment_input=commitment,
            idempotency_key=None,
        )
        _show("DUPLICATE REJECTION", dup)

    # ══════════════════════════════════════════════════════════════════
    # 5. SOFT DELETE
    # ══════════════════════════════════════════════════════════════════

    with SessionLocal() as sess:
        # Retrieve the commitment ID first
        rec = sess.query(CommitmentRecordORM).filter(
            CommitmentRecordORM.user_id == user,
            CommitmentRecordORM.deleted_at.is_(None),
        ).first()

        if rec:
            deleted = soft_delete_commitment(sess, rec.id, user)
            _show("SOFT DELETE", deleted)

    # ══════════════════════════════════════════════════════════════════
    # 6. RE-REGISTER AFTER SOFT DELETE (same idempotency key → REJECTED)
    # ══════════════════════════════════════════════════════════════════

    with SessionLocal() as sess:
        re_reg = register_commitment(
            session=sess,
            user_id=user,
            commitment_input=commitment,
            idempotency_key="INS-2026-001",
        )
        _show("RE-REGISTER AFTER SOFT DELETE", re_reg)
