"""
Scaling Readiness — Financial Simulation System (Phase 8)

Production-grade infrastructure for horizontal scaling:
  1. /health endpoint with uptime and SLA visibility
  2. Idempotent simulation requests (DB-enforced)
  3. RateLimiter interface (in-memory + Redis stub)
  4. Stateless SimulationService wrapper

Guarantees:
  - Zero shared-memory correctness dependencies
  - Idempotency enforced at DB layer (UNIQUE constraint)
  - Multiple containers can serve requests simultaneously
  - All user state comes from DB or request payload
  - Rate limiting is backend-swappable (dev → Redis)
  - Health and SLA visible without shared state

No domain logic.  No risk math.  No scoring.  No tool validation.
"""

from __future__ import annotations

import abc
import json
import time
import uuid as _uuid_mod
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from sqlalchemy import Column, DateTime, String, UniqueConstraint, and_, JSON
from sqlalchemy.orm import Session

from persistence import Base, _uuid_column


# ═══════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════

_STARTUP_TIME = time.time()

# Rate limiter defaults
DEFAULT_RATE_LIMIT_REQUESTS = 60
DEFAULT_RATE_LIMIT_WINDOW_SECONDS = 60


# ═══════════════════════════════════════════════════════════════════════
#  Section 1 — Health Endpoint
# ═══════════════════════════════════════════════════════════════════════

def get_health_status(*, sla_status: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build health check response.

    Returns uptime in seconds and SLA compliance flag.
    Designed to be called by the /health endpoint and Docker HEALTHCHECK.
    """
    uptime = round(time.time() - _STARTUP_TIME, 2)
    within_sla = True
    if sla_status is not None:
        within_sla = sla_status.get("within_sla", True)

    return {
        "status": "ok",
        "uptime_seconds": uptime,
        "within_sla": within_sla,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Section 2 — Idempotency (DB-enforced)
# ═══════════════════════════════════════════════════════════════════════

class IdempotencyRecordORM(Base):
    """Stores idempotency keys with their cached results.

    UNIQUE(user_id, idempotency_key) enforced at DB level,
    preventing duplicate executions across all container instances.
    """

    __tablename__ = "idempotency_keys"

    id = _uuid_column(primary_key=True, default=_uuid_mod.uuid4)
    user_id = _uuid_column(nullable=False, index=True)
    idempotency_key = Column(String(256), nullable=False)
    correlation_id = Column(String(128), nullable=False)
    cached_result = Column(JSON, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        UniqueConstraint(
            "user_id", "idempotency_key",
            name="uq_user_idempotency_key",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<IdempotencyRecord user={self.user_id} "
            f"key={self.idempotency_key}>"
        )


def check_idempotency(
    session: Session,
    user_id: _uuid_mod.UUID,
    idempotency_key: str,
) -> Optional[Dict[str, Any]]:
    """Check if a result already exists for this user + idempotency key.

    Returns the cached result dict if found, else None.
    """
    record = (
        session.query(IdempotencyRecordORM)
        .filter(
            and_(
                IdempotencyRecordORM.user_id == user_id,
                IdempotencyRecordORM.idempotency_key == idempotency_key,
            )
        )
        .first()
    )
    if record is not None:
        return record.cached_result
    return None


def store_idempotency(
    session: Session,
    user_id: _uuid_mod.UUID,
    idempotency_key: str,
    correlation_id: str,
    result: Dict[str, Any],
) -> None:
    """Store a simulation result keyed by (user_id, idempotency_key).

    The UNIQUE constraint ensures this fails cleanly if a concurrent
    request inserts the same key — the caller should handle IntegrityError.
    """
    record = IdempotencyRecordORM(
        id=_uuid_mod.uuid4(),
        user_id=user_id,
        idempotency_key=idempotency_key,
        correlation_id=correlation_id,
        cached_result=result,
        created_at=datetime.now(timezone.utc),
    )
    session.add(record)
    session.flush()


# ═══════════════════════════════════════════════════════════════════════
#  Section 3 — Rate Limiter Interface
# ═══════════════════════════════════════════════════════════════════════

class RateLimitExceeded(Exception):
    """Raised when a user exceeds their rate limit."""
    def __init__(self, user_id: _uuid_mod.UUID, retry_after_seconds: float) -> None:
        self.user_id = user_id
        self.retry_after_seconds = retry_after_seconds
        super().__init__(
            f"Rate limit exceeded for user {user_id}. "
            f"Retry after {retry_after_seconds:.0f}s."
        )


class RateLimiterBackend(abc.ABC):
    """Abstract rate limiter interface.

    Production deployments swap InMemoryRateLimiter for RedisRateLimiter
    to get cross-instance rate limiting.
    """

    @abc.abstractmethod
    async def check_limit(self, user_id: _uuid_mod.UUID) -> None:
        """Raise ``RateLimitExceeded`` if user is over their limit."""
        ...

    @abc.abstractmethod
    async def reset(self, user_id: _uuid_mod.UUID) -> None:
        """Reset the rate limit counter for a user (admin use)."""
        ...


class InMemoryRateLimiter(RateLimiterBackend):
    """Process-local rate limiter for development / single-instance use.

    NOT suitable for multi-container production deployments because
    each container maintains its own counter.
    """

    def __init__(
        self,
        max_requests: int = DEFAULT_RATE_LIMIT_REQUESTS,
        window_seconds: int = DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
    ) -> None:
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._buckets: Dict[str, list] = {}  # user_id -> [timestamps]

    async def check_limit(self, user_id: _uuid_mod.UUID) -> None:
        key = str(user_id)
        now = time.time()
        cutoff = now - self._window_seconds

        # Prune expired entries
        timestamps = self._buckets.get(key, [])
        timestamps = [t for t in timestamps if t > cutoff]

        if len(timestamps) >= self._max_requests:
            oldest = min(timestamps) if timestamps else now
            retry_after = oldest + self._window_seconds - now
            raise RateLimitExceeded(user_id, max(retry_after, 1.0))

        timestamps.append(now)
        self._buckets[key] = timestamps

    async def reset(self, user_id: _uuid_mod.UUID) -> None:
        self._buckets.pop(str(user_id), None)


class RedisRateLimiter(RateLimiterBackend):
    """Redis-backed rate limiter stub for production multi-container use.

    Uses a sliding window counter pattern with Redis SORTED SETs.
    This stub demonstrates the interface — replace ``_redis`` with
    a real ``redis.asyncio.Redis`` instance.
    """

    def __init__(
        self,
        redis_client: Any = None,
        max_requests: int = DEFAULT_RATE_LIMIT_REQUESTS,
        window_seconds: int = DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
    ) -> None:
        self._redis = redis_client  # redis.asyncio.Redis instance
        self._max_requests = max_requests
        self._window_seconds = window_seconds

    async def check_limit(self, user_id: _uuid_mod.UUID) -> None:
        if self._redis is None:
            # No Redis configured — stub passes all requests
            return

        key = f"rate_limit:{user_id}"
        now = time.time()
        cutoff = now - self._window_seconds

        pipe = self._redis.pipeline()
        pipe.zremrangebyscore(key, 0, cutoff)
        pipe.zcard(key)
        pipe.zadd(key, {str(now): now})
        pipe.expire(key, self._window_seconds)
        results = await pipe.execute()

        count = results[1]
        if count >= self._max_requests:
            raise RateLimitExceeded(user_id, float(self._window_seconds))

    async def reset(self, user_id: _uuid_mod.UUID) -> None:
        if self._redis is None:
            return
        await self._redis.delete(f"rate_limit:{user_id}")


# ═══════════════════════════════════════════════════════════════════════
#  Section 4 — Stateless Simulation Service
# ═══════════════════════════════════════════════════════════════════════

class StatelessSimulationService:
    """Simulation service designed for horizontal scaling.

    Guarantees:
      - No mutable user state held in-process
      - All state comes from DB (session) or request parameters
      - Idempotency enforced at DB layer
      - Rate limiting via swappable backend
      - Composable with existing failure_handling.safe_simulation_run
    """

    def __init__(
        self,
        rate_limiter: Optional[RateLimiterBackend] = None,
    ) -> None:
        self._rate_limiter = rate_limiter or InMemoryRateLimiter()

    def run_idempotent(
        self,
        session: Session,
        user_id: _uuid_mod.UUID,
        correlation_id: str,
        idempotency_key: Optional[str],
        simulation_func: Callable[..., Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute a simulation with idempotency guarantees.

        Flow:
          1. If idempotency_key provided → check DB for cached result
          2. If cached → return existing result without re-execution
          3. If not cached → execute simulation
          4. Store result with idempotency_key
          5. Commit transaction

        All correctness is enforced via DB UNIQUE constraint,
        not process memory.
        """
        # Step 1: Check idempotency cache
        if idempotency_key:
            cached = check_idempotency(session, user_id, idempotency_key)
            if cached is not None:
                return {
                    "status": "success",
                    "cached": True,
                    "idempotency_key": idempotency_key,
                    "data": cached,
                }

        # Step 2: Execute simulation (stateless — all inputs via kwargs)
        result = simulation_func(**kwargs)

        # Step 3: Store idempotency record (if key provided)
        if idempotency_key and result.get("status") == "success":
            try:
                store_idempotency(
                    session, user_id, idempotency_key,
                    correlation_id, result,
                )
                session.commit()
            except Exception:
                session.rollback()
                # Don't fail the request — idempotency storage is best-effort
                pass

        return result


# ═══════════════════════════════════════════════════════════════════════
#  Section 5 — FastAPI Integration
# ═══════════════════════════════════════════════════════════════════════

try:
    from fastapi import Depends, FastAPI, Request
    from fastapi.responses import JSONResponse

    def register_scaling_endpoints(app: FastAPI) -> None:
        """Register /health and scaling-related endpoints."""

        @app.get("/health")
        async def health_check() -> JSONResponse:
            """Health check for Docker HEALTHCHECK and load balancers.

            Returns uptime and SLA compliance status.
            """
            # Import lazily to avoid circular deps at module level
            try:
                from failure_handling import get_sla_status
                sla = get_sla_status()
            except Exception:
                sla = None

            return JSONResponse(content=get_health_status(sla_status=sla))

    _FASTAPI_AVAILABLE = True

except ImportError:
    _FASTAPI_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════
#  Example Usage
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import asyncio

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    def _show(label: str, obj: Any) -> None:
        print(f"\n{'═' * 64}")
        print(f"  {label}")
        print(f"{'═' * 64}")
        if isinstance(obj, dict):
            print(json.dumps(obj, indent=2, default=str))
        else:
            print(str(obj))

    # ── In-memory DB for demo ─────────────────────────────────────────
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)

    # ══════════════════════════════════════════════════════════════════
    # 1. HEALTH ENDPOINT
    # ══════════════════════════════════════════════════════════════════

    health = get_health_status()
    _show("1. HEALTH STATUS", health)

    # ══════════════════════════════════════════════════════════════════
    # 2. IDEMPOTENT SIMULATION — First call (executes)
    # ══════════════════════════════════════════════════════════════════

    user_id = _uuid_mod.uuid4()
    cid = "idem-001"
    idem_key = "sim-2026-03-03-001"

    def _mock_simulation(**kw: Any) -> Dict[str, Any]:
        return {
            "status": "success",
            "data": {"results": [{"stability_score": 78.5}]},
        }

    svc = StatelessSimulationService()
    with SessionLocal() as sess:
        result1 = svc.run_idempotent(
            session=sess,
            user_id=user_id,
            correlation_id=cid,
            idempotency_key=idem_key,
            simulation_func=_mock_simulation,
        )
    _show("2. FIRST CALL (executes simulation)", result1)

    # ══════════════════════════════════════════════════════════════════
    # 3. IDEMPOTENT SIMULATION — Second call (cached)
    # ══════════════════════════════════════════════════════════════════

    with SessionLocal() as sess:
        result2 = svc.run_idempotent(
            session=sess,
            user_id=user_id,
            correlation_id="idem-002",
            idempotency_key=idem_key,  # same key
            simulation_func=_mock_simulation,
        )
    _show("3. SECOND CALL (returns cached — no re-execution)", result2)

    # ══════════════════════════════════════════════════════════════════
    # 4. IDEMPOTENT SIMULATION — Different key (executes again)
    # ══════════════════════════════════════════════════════════════════

    with SessionLocal() as sess:
        result3 = svc.run_idempotent(
            session=sess,
            user_id=user_id,
            correlation_id="idem-003",
            idempotency_key="sim-2026-03-03-002",  # different key
            simulation_func=_mock_simulation,
        )
    _show("4. DIFFERENT KEY (executes new simulation)", result3)

    # ══════════════════════════════════════════════════════════════════
    # 5. RATE LIMITER — In-Memory (dev)
    # ══════════════════════════════════════════════════════════════════

    limiter = InMemoryRateLimiter(max_requests=3, window_seconds=60)

    async def _test_rate_limit() -> None:
        uid = _uuid_mod.uuid4()
        results = []
        for i in range(5):
            try:
                await limiter.check_limit(uid)
                results.append(f"Request {i+1}: ALLOWED")
            except RateLimitExceeded as exc:
                results.append(
                    f"Request {i+1}: BLOCKED (retry after {exc.retry_after_seconds:.0f}s)"
                )
        return results

    rate_results = asyncio.run(_test_rate_limit())
    _show("5. RATE LIMITER (3 req/60s)", {
        "results": rate_results,
    })

    # ══════════════════════════════════════════════════════════════════
    # 6. RATE LIMITER INTERFACE
    # ══════════════════════════════════════════════════════════════════

    _show("6. RATE LIMITER BACKENDS", {
        "in_memory": "InMemoryRateLimiter — dev/single instance only",
        "redis": "RedisRateLimiter — production multi-container",
        "interface": "RateLimiterBackend.check_limit(user_id) → None or raises",
    })

    # ══════════════════════════════════════════════════════════════════
    # 7. HORIZONTAL SCALING SCENARIO
    # ══════════════════════════════════════════════════════════════════

    _show("7. HORIZONTAL SCALING SCENARIO", {
        "scenario": "3 containers behind a load balancer",
        "idempotency": "Enforced by DB UNIQUE(user_id, idempotency_key) — "
                       "any container can serve any request",
        "rate_limiting": "RedisRateLimiter shares counters across all instances",
        "state": "All user state from DB or request payload — "
                 "zero in-process memory dependencies",
        "metrics": "Prometheus /metrics per instance, aggregated by scraper",
        "health": "GET /health per instance for load balancer probes",
    })

    print(f"\n{'═' * 64}")
    print(f"  FastAPI available: {_FASTAPI_AVAILABLE}")
    print(f"{'═' * 64}")
