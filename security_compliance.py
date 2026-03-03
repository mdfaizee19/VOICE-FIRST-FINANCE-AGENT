"""
Security & Compliance Hardening — Financial Simulation System (Phase 9)

Production-grade security layer implementing:
  1. AES-256-GCM encryption for financial fields at rest
  2. Sensitive log masking to prevent data leakage
  3. Append-only, immutable audit trail

Guarantees:
  - No financial data stored in plaintext
  - No sensitive values in logs (income, expenses, debt, etc.)
  - Tamper-resistant audit records (no UPDATE, no DELETE)
  - Encryption key from environment variable only
  - Structured errors on encryption failure
  - Audit failures never block simulation execution

No simulation logic.  No risk math.  No scoring.  No tool validation.
"""

from __future__ import annotations

import json
import os
import uuid as _uuid_mod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Union

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from sqlalchemy import Column, DateTime, String, LargeBinary, JSON, event
from sqlalchemy.orm import Session

from persistence import Base, _uuid_column


# ═══════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════

ENV_KEY_NAME = "FINANCIAL_ENCRYPTION_KEY"

# Fields that must NEVER appear in plaintext logs
SENSITIVE_FIELD_NAMES: Set[str] = frozenset({
    # Income / balance
    "monthly_income", "income", "projected_balance", "balance",
    "emergency_fund", "fund_balance",
    # Expenses
    "fixed_expenses", "discretionary_expenses", "expenses",
    "total_outflows", "monthly_cost",
    # Debt / credit
    "credit_balance", "credit_apr", "interest_cost", "interest",
    "interest_ratio", "interest_health",
    # Commitments
    "amount", "commitment_coverage", "commit_total",
    # Scores & health (numeric but non-sensitive — kept for audit)
    # These are explicitly NOT masked:
    #   stability_score, liquidity_ratio, risk_flags, burn_rate
    # Encrypted blobs
    "pre_simulation_state", "post_simulation_result",
    "input_payload", "output_payload",
})

# Fields that are safe to keep in logs / audit metadata
SAFE_FIELD_NAMES: Set[str] = frozenset({
    "correlation_id", "error_code", "status", "tool_name",
    "event_type", "user_id", "entity_id", "action_type",
    "entity_type", "stability_score", "risk_level",
    "intent_type", "option_id", "id", "name",
    "scoring_formula_version", "risk_model_version",
    "schema_version", "idempotency_key", "cached",
})


# ═══════════════════════════════════════════════════════════════════════
#  Section 1 — AES-256-GCM Field Encryption
# ═══════════════════════════════════════════════════════════════════════

class EncryptionError(Exception):
    """Raised when encryption or decryption fails."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)

    def to_dict(self) -> Dict[str, str]:
        return {
            "status": "error",
            "error_code": "ENCRYPTION_FAILURE",
            "message": self.message,
        }


@dataclass(frozen=True)
class EncryptedPayload:
    """Container for AES-256-GCM encrypted data.

    Attributes:
        ciphertext: The encrypted data bytes.
        nonce: 12-byte random nonce used for encryption.
        tag: Authentication tag (appended to ciphertext by AESGCM).
    """
    ciphertext: bytes
    nonce: bytes

    def to_storage(self) -> bytes:
        """Serialize for database storage: nonce || ciphertext+tag."""
        return self.nonce + self.ciphertext

    @classmethod
    def from_storage(cls, blob: bytes) -> "EncryptedPayload":
        """Deserialize from database storage."""
        if len(blob) < 12:
            raise EncryptionError("Encrypted blob too short to contain nonce")
        return cls(
            nonce=blob[:12],
            ciphertext=blob[12:],
        )


class FieldEncryptor:
    """AES-256-GCM field-level encryptor for financial data.

    The encryption key is loaded from the environment variable
    specified by ``ENV_KEY_NAME``.  It must be exactly 32 bytes
    (hex-encoded = 64 characters).

    Usage::

        encryptor = FieldEncryptor()
        encrypted = encryptor.encrypt({"monthly_income": 5000.0})
        decrypted = encryptor.decrypt(encrypted)
    """

    def __init__(self, key: Optional[bytes] = None) -> None:
        if key is not None:
            self._key = key
        else:
            self._key = self._load_key_from_env()
        self._aesgcm = AESGCM(self._key)

    @staticmethod
    def _load_key_from_env() -> bytes:
        """Load encryption key from environment variable.

        Expects a 64-character hex string (32 bytes).
        """
        raw = os.environ.get(ENV_KEY_NAME)
        if not raw:
            raise EncryptionError(
                f"Encryption key not found. Set the {ENV_KEY_NAME} "
                f"environment variable (64 hex chars = 32 bytes)."
            )
        try:
            key_bytes = bytes.fromhex(raw.strip())
        except ValueError:
            raise EncryptionError(
                f"{ENV_KEY_NAME} must be a valid hex string"
            )
        if len(key_bytes) != 32:
            raise EncryptionError(
                f"{ENV_KEY_NAME} must be exactly 32 bytes "
                f"(64 hex chars), got {len(key_bytes)} bytes"
            )
        return key_bytes

    @staticmethod
    def generate_key() -> str:
        """Generate a random 256-bit key as hex string (for setup)."""
        return os.urandom(32).hex()

    def encrypt(self, data: Union[dict, str]) -> EncryptedPayload:
        """Encrypt a dict or string using AES-256-GCM.

        Returns an EncryptedPayload containing ciphertext and nonce.
        The authentication tag is appended to the ciphertext by AESGCM.
        """
        try:
            if isinstance(data, dict):
                plaintext = json.dumps(data, sort_keys=True).encode("utf-8")
            elif isinstance(data, str):
                plaintext = data.encode("utf-8")
            else:
                raise EncryptionError(
                    f"encrypt() requires dict or str, got {type(data).__name__}"
                )

            nonce = os.urandom(12)  # 96-bit nonce for GCM
            ciphertext = self._aesgcm.encrypt(nonce, plaintext, None)

            return EncryptedPayload(ciphertext=ciphertext, nonce=nonce)

        except EncryptionError:
            raise
        except Exception as exc:
            raise EncryptionError(f"Encryption failed: {exc}") from exc

    def decrypt(self, payload: EncryptedPayload) -> dict:
        """Decrypt an EncryptedPayload back to a dict.

        Raises EncryptionError if decryption or authentication fails.
        """
        try:
            plaintext = self._aesgcm.decrypt(
                payload.nonce, payload.ciphertext, None,
            )
            return json.loads(plaintext.decode("utf-8"))

        except EncryptionError:
            raise
        except Exception as exc:
            raise EncryptionError(f"Decryption failed: {exc}") from exc

    def encrypt_to_blob(self, data: Union[dict, str]) -> bytes:
        """Encrypt and serialize to a single bytes blob for DB storage."""
        return self.encrypt(data).to_storage()

    def decrypt_from_blob(self, blob: bytes) -> dict:
        """Deserialize a storage blob and decrypt to dict."""
        payload = EncryptedPayload.from_storage(blob)
        return self.decrypt(payload)


# ═══════════════════════════════════════════════════════════════════════
#  Section 2 — Sensitive Log Masking
# ═══════════════════════════════════════════════════════════════════════

MASK_VALUE = "***"


def mask_sensitive_data(
    payload: Any,
    *,
    _depth: int = 0,
    _max_depth: int = 10,
) -> Any:
    """Recursively mask sensitive financial values in a payload.

    Rules:
      - Numeric values in sensitive fields → "***"
      - String values in sensitive fields → "***"
      - Dict values in sensitive fields → {"masked": True}
      - List values in sensitive fields → ["***"]
      - Preserves structure and keys
      - Does not mask safe fields (correlation_id, error_code, etc.)
      - Does not remove any keys

    Returns a new dict (original is never mutated).
    """
    if _depth > _max_depth:
        return MASK_VALUE

    if isinstance(payload, dict):
        masked: Dict[str, Any] = {}
        for key, value in payload.items():
            if key in SENSITIVE_FIELD_NAMES:
                # Mask the value based on type
                if isinstance(value, (int, float)):
                    masked[key] = MASK_VALUE
                elif isinstance(value, str):
                    masked[key] = MASK_VALUE
                elif isinstance(value, dict):
                    masked[key] = {"masked": True}
                elif isinstance(value, list):
                    masked[key] = [MASK_VALUE]
                elif isinstance(value, bytes):
                    masked[key] = MASK_VALUE
                else:
                    masked[key] = MASK_VALUE
            else:
                # Recurse into non-sensitive fields
                masked[key] = mask_sensitive_data(
                    value, _depth=_depth + 1, _max_depth=_max_depth,
                )
        return masked

    elif isinstance(payload, list):
        return [
            mask_sensitive_data(item, _depth=_depth + 1, _max_depth=_max_depth)
            for item in payload
        ]

    else:
        return payload


def mask_for_audit(
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract only audit-safe fields from a payload.

    Returns a dict containing only fields from SAFE_FIELD_NAMES,
    plus stability_score and risk_flags summary.
    Used for AuditLog metadata — no financial values allowed.
    """
    safe: Dict[str, Any] = {}
    for key, value in payload.items():
        if key in SAFE_FIELD_NAMES:
            safe[key] = value
        elif key == "risk_flags" and isinstance(value, list):
            safe["risk_flag_count"] = len(value)
    return safe


# ═══════════════════════════════════════════════════════════════════════
#  Section 3 — Audit Trail (Append-Only)
# ═══════════════════════════════════════════════════════════════════════

class AuditLogORM(Base):
    """Immutable, append-only audit record.

    No UPDATE or DELETE operations are permitted.
    Stores only non-sensitive metadata — no financial values.
    """

    __tablename__ = "audit_logs"

    id = _uuid_column(primary_key=True, default=_uuid_mod.uuid4)
    user_id = _uuid_column(nullable=False, index=True)
    action_type = Column(String(64), nullable=False, index=True)
    entity_type = Column(String(64), nullable=False, index=True)
    entity_id = _uuid_column(nullable=False, index=True)
    correlation_id = Column(String(128), nullable=False, index=True)
    metadata_json = Column(JSON, nullable=False, default=dict)
    ip_address = Column(String(45), nullable=True)  # IPv4 or IPv6
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def __repr__(self) -> str:
        return (
            f"<AuditLog id={self.id} action={self.action_type} "
            f"entity={self.entity_type}/{self.entity_id}>"
        )


# ── Block UPDATE and DELETE on audit records ──────────────────────────

@event.listens_for(AuditLogORM, "before_update")
def _block_audit_update(mapper: Any, connection: Any, target: AuditLogORM) -> None:
    raise RuntimeError(
        "Audit logs are immutable. UPDATE operations are forbidden."
    )


@event.listens_for(AuditLogORM, "before_delete")
def _block_audit_delete(mapper: Any, connection: Any, target: AuditLogORM) -> None:
    raise RuntimeError(
        "Audit logs are immutable. DELETE operations are forbidden."
    )


# ── Audit Action Types ───────────────────────────────────────────────

class AuditAction:
    SIMULATION_EXECUTED = "SIMULATION_EXECUTED"
    COMMITMENT_CREATED = "COMMITMENT_CREATED"
    COMMITMENT_UPDATED = "COMMITMENT_UPDATED"
    COMMITMENT_DELETED = "COMMITMENT_DELETED"


class AuditEntity:
    SIMULATION = "SIMULATION"
    COMMITMENT = "COMMITMENT"


# ── Audit Write Function ─────────────────────────────────────────────

def create_audit_log(
    session: Session,
    user_id: _uuid_mod.UUID,
    action_type: str,
    entity_type: str,
    entity_id: _uuid_mod.UUID,
    correlation_id: str,
    metadata: Optional[Dict[str, Any]] = None,
    ip_address: Optional[str] = None,
) -> Dict[str, Any]:
    """Create an immutable audit log entry.

    The metadata dict is sanitized via ``mask_for_audit()`` to ensure
    no financial values are stored.  Only safe fields are persisted.

    Returns structured result dict.
    Never raises — audit failures are caught and returned as errors
    (but should NOT block the parent transaction).
    """
    try:
        # Sanitize metadata — strip all financial values
        safe_metadata = mask_for_audit(metadata or {})

        record = AuditLogORM(
            id=_uuid_mod.uuid4(),
            user_id=user_id,
            action_type=action_type,
            entity_type=entity_type,
            entity_id=entity_id,
            correlation_id=correlation_id,
            metadata_json=safe_metadata,
            ip_address=ip_address,
            created_at=datetime.now(timezone.utc),
        )
        session.add(record)
        session.flush()

        return {
            "status": "success",
            "audit_id": str(record.id),
            "action_type": record.action_type,
            "entity_type": record.entity_type,
            "entity_id": str(record.entity_id),
        }

    except Exception as exc:
        # Audit failure must never block simulation
        try:
            session.rollback()
        except Exception:
            pass
        return {
            "status": "error",
            "error_code": "AUDIT_LOG_FAILURE",
            "message": f"Audit logging failed: {str(exc)[:200]}",
        }


# ── Audit Query Helper ───────────────────────────────────────────────

def get_audit_trail(
    session: Session,
    user_id: _uuid_mod.UUID,
    entity_type: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    """Retrieve audit trail for a user, optionally filtered by entity type."""
    try:
        query = session.query(AuditLogORM).filter(
            AuditLogORM.user_id == user_id,
        )
        if entity_type:
            query = query.filter(AuditLogORM.entity_type == entity_type)

        records = (
            query.order_by(AuditLogORM.created_at.desc())
            .limit(limit)
            .all()
        )

        return {
            "status": "success",
            "count": len(records),
            "records": [
                {
                    "id": str(r.id),
                    "action_type": r.action_type,
                    "entity_type": r.entity_type,
                    "entity_id": str(r.entity_id),
                    "correlation_id": r.correlation_id,
                    "metadata": r.metadata_json,
                    "ip_address": r.ip_address,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in records
            ],
        }

    except Exception as exc:
        return {
            "status": "error",
            "error_code": "AUDIT_QUERY_FAILURE",
            "message": f"Audit query failed: {str(exc)[:200]}",
        }


# ═══════════════════════════════════════════════════════════════════════
#  Section 4 — Encrypted Persistence Helpers
# ═══════════════════════════════════════════════════════════════════════

def encrypt_for_storage(
    encryptor: FieldEncryptor,
    data: Union[dict, str],
) -> bytes:
    """Encrypt data and return a single blob for DB column storage.

    Raises EncryptionError on failure — caller must abort transaction.
    """
    return encryptor.encrypt_to_blob(data)


def decrypt_from_storage(
    encryptor: FieldEncryptor,
    blob: bytes,
) -> dict:
    """Decrypt a storage blob back to a dict.

    Only called in the service layer — never in the controller.
    """
    return encryptor.decrypt_from_blob(blob)


# ═══════════════════════════════════════════════════════════════════════
#  Example Usage
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    def _show(label: str, obj: Any) -> None:
        print(f"\n{'═' * 64}")
        print(f"  {label}")
        print(f"{'═' * 64}")
        if isinstance(obj, (dict, list)):
            print(json.dumps(obj, indent=2, default=str))
        elif isinstance(obj, bytes):
            print(f"  <{len(obj)} bytes>")
        else:
            print(str(obj))

    # ── Setup: generate a test key & encryptor ────────────────────────
    test_key_hex = FieldEncryptor.generate_key()
    os.environ[ENV_KEY_NAME] = test_key_hex
    encryptor = FieldEncryptor()

    # ── Setup: in-memory DB ───────────────────────────────────────────
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)

    # ══════════════════════════════════════════════════════════════════
    # 1. ENCRYPTION — Encrypt financial state
    # ══════════════════════════════════════════════════════════════════

    financial_state = {
        "monthly_income": 5000.0,
        "fixed_expenses": 2000.0,
        "discretionary_expenses": 800.0,
        "emergency_fund": 10000.0,
        "credit_balance": 3000.0,
        "credit_apr": 0.24,
        "commitments": [
            {"id": "ins", "amount": 300.0, "due_month": 3},
        ],
    }

    encrypted = encryptor.encrypt(financial_state)
    _show("1a. ENCRYPTED PAYLOAD", {
        "nonce_length": len(encrypted.nonce),
        "ciphertext_length": len(encrypted.ciphertext),
        "storage_blob_length": len(encrypted.to_storage()),
    })

    # Decrypt back
    decrypted = encryptor.decrypt(encrypted)
    _show("1b. DECRYPTED (matches original)", decrypted)

    # Verify round-trip
    assert decrypted == financial_state, "Round-trip encryption failed!"
    print("  ✅ Round-trip verification passed")

    # ══════════════════════════════════════════════════════════════════
    # 2. BLOB STORAGE — encrypt_to_blob / decrypt_from_blob
    # ══════════════════════════════════════════════════════════════════

    blob = encryptor.encrypt_to_blob(financial_state)
    restored = encryptor.decrypt_from_blob(blob)
    assert restored == financial_state
    _show("2. BLOB STORAGE (round-trip)", {
        "blob_size_bytes": len(blob),
        "round_trip": "✅ passed",
    })

    # ══════════════════════════════════════════════════════════════════
    # 3. LOG MASKING — mask_sensitive_data
    # ══════════════════════════════════════════════════════════════════

    raw_log = {
        "correlation_id": "req-001",
        "status": "success",
        "tool_name": "subscription_simulation",
        "monthly_income": 5000.0,
        "fixed_expenses": 2000.0,
        "emergency_fund": 10000.0,
        "credit_balance": 3000.0,
        "interest_cost": 60.0,
        "stability_score": 78.5,
        "risk_flags": ["low_liquidity"],
        "pre_simulation_state": {"monthly_income": 5000.0},
        "commitments": [{"id": "ins", "amount": 300.0}],
    }

    masked = mask_sensitive_data(raw_log)
    _show("3. MASKED LOG OUTPUT", masked)

    # ══════════════════════════════════════════════════════════════════
    # 4. AUDIT MASKING — mask_for_audit
    # ══════════════════════════════════════════════════════════════════

    audit_meta = mask_for_audit({
        "correlation_id": "req-001",
        "stability_score": 78.5,
        "monthly_income": 5000.0,
        "credit_balance": 3000.0,
        "risk_flags": ["low_liquidity"],
        "intent_type": "subscription_purchase",
    })
    _show("4. AUDIT-SAFE METADATA (financial values stripped)", audit_meta)

    # ══════════════════════════════════════════════════════════════════
    # 5. AUDIT LOG — create entries
    # ══════════════════════════════════════════════════════════════════

    user_id = _uuid_mod.uuid4()
    snapshot_id = _uuid_mod.uuid4()
    commit_id = _uuid_mod.uuid4()

    with SessionLocal() as sess:
        # Simulation executed
        r1 = create_audit_log(
            session=sess,
            user_id=user_id,
            action_type=AuditAction.SIMULATION_EXECUTED,
            entity_type=AuditEntity.SIMULATION,
            entity_id=snapshot_id,
            correlation_id="sim-001",
            metadata={
                "stability_score": 78.5,
                "monthly_income": 5000.0,  # will be stripped
                "intent_type": "subscription_purchase",
                "risk_flags": ["low_liquidity"],
            },
            ip_address="192.168.1.100",
        )
        sess.commit()
        _show("5a. AUDIT — SIMULATION_EXECUTED", r1)

        # Commitment created
        r2 = create_audit_log(
            session=sess,
            user_id=user_id,
            action_type=AuditAction.COMMITMENT_CREATED,
            entity_type=AuditEntity.COMMITMENT,
            entity_id=commit_id,
            correlation_id="sim-001",
            metadata={
                "name": "Car Insurance",
                "amount": 300.0,  # will be stripped
            },
            ip_address="192.168.1.100",
        )
        sess.commit()
        _show("5b. AUDIT — COMMITMENT_CREATED", r2)

        # Commitment updated
        r3 = create_audit_log(
            session=sess,
            user_id=user_id,
            action_type=AuditAction.COMMITMENT_UPDATED,
            entity_type=AuditEntity.COMMITMENT,
            entity_id=commit_id,
            correlation_id="sim-002",
            metadata={"name": "Car Insurance"},
        )
        sess.commit()
        _show("5c. AUDIT — COMMITMENT_UPDATED", r3)

        # Commitment soft deleted
        r4 = create_audit_log(
            session=sess,
            user_id=user_id,
            action_type=AuditAction.COMMITMENT_DELETED,
            entity_type=AuditEntity.COMMITMENT,
            entity_id=commit_id,
            correlation_id="sim-003",
            metadata={"name": "Car Insurance"},
        )
        sess.commit()
        _show("5d. AUDIT — COMMITMENT_DELETED", r4)

    # ══════════════════════════════════════════════════════════════════
    # 6. AUDIT TRAIL — query
    # ══════════════════════════════════════════════════════════════════

    with SessionLocal() as sess:
        trail = get_audit_trail(sess, user_id)
        _show("6. FULL AUDIT TRAIL", trail)

    # ══════════════════════════════════════════════════════════════════
    # 7. IMMUTABILITY — attempt UPDATE (must fail)
    # ══════════════════════════════════════════════════════════════════

    with SessionLocal() as sess:
        record = sess.query(AuditLogORM).first()
        if record:
            try:
                record.action_type = "TAMPERED"
                sess.flush()
                print("\n  ❌ ERROR: UPDATE should have been blocked!")
            except RuntimeError as exc:
                _show("7. IMMUTABILITY ENFORCED", {
                    "attempted": "UPDATE audit record",
                    "result": "BLOCKED",
                    "error": str(exc),
                })
                sess.rollback()

    # ══════════════════════════════════════════════════════════════════
    # 8. IMMUTABILITY — attempt DELETE (must fail)
    # ══════════════════════════════════════════════════════════════════

    with SessionLocal() as sess:
        record = sess.query(AuditLogORM).first()
        if record:
            try:
                sess.delete(record)
                sess.flush()
                print("\n  ❌ ERROR: DELETE should have been blocked!")
            except RuntimeError as exc:
                _show("8. DELETE BLOCKED", {
                    "attempted": "DELETE audit record",
                    "result": "BLOCKED",
                    "error": str(exc),
                })
                sess.rollback()

    # ══════════════════════════════════════════════════════════════════
    # 9. ENCRYPTION ERROR HANDLING
    # ══════════════════════════════════════════════════════════════════

    try:
        bad_encryptor = FieldEncryptor(key=b"short_key")
    except Exception as exc:
        _show("9. ENCRYPTION ERROR (bad key)", {
            "error": str(exc),
        })

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════

    _show("SUMMARY", {
        "encryption": "AES-256-GCM with random 96-bit nonce",
        "key_source": f"Environment variable: {ENV_KEY_NAME}",
        "masking": f"{len(SENSITIVE_FIELD_NAMES)} sensitive field names masked",
        "audit_actions": [
            AuditAction.SIMULATION_EXECUTED,
            AuditAction.COMMITMENT_CREATED,
            AuditAction.COMMITMENT_UPDATED,
            AuditAction.COMMITMENT_DELETED,
        ],
        "audit_immutability": "UPDATE and DELETE blocked via SQLAlchemy events",
    })
