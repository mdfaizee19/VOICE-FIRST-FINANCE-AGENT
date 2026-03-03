# SNEMOSYNE — Financial Simulation Engine

> Production-grade, deterministic financial simulation system with structured tool calling, risk intelligence, and LLM-agent orchestration support.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Module Reference](#module-reference)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Tool Calling Flow](#tool-calling-flow)
- [Schemas](#schemas)
- [Simulation Engine](#simulation-engine)
- [Risk & Scoring](#risk--scoring)
- [Stress Testing](#stress-testing)
- [Sensitivity Analysis](#sensitivity-analysis)
- [Security & Compliance](#security--compliance)
- [Observability](#observability)
- [Failure Handling & SLA](#failure-handling--sla)
- [Scaling & Deployment](#scaling--deployment)
- [Testing](#testing)
- [Configuration](#configuration)
- [Contributing](#contributing)

---

## Overview

This system is a **deterministic financial simulation engine** designed to evaluate the impact of financial decisions (subscriptions, credit strategies, commitments) on a user's overall financial health. It computes a **Stability Score (0–100)**, identifies **risk flags**, and runs **stress tests** and **sensitivity analyses** — all through structured, validated tool calls.

### Key Design Principles

| Principle | Description |
|---|---|
| **Deterministic** | Same inputs → identical outputs. No randomness, no time dependency. |
| **Defense in Depth** | Input sanitization → schema validation → tool constraints → output verification → risk flagging |
| **Fail Safe** | Graceful degradation on tool failure — never crashes, never leaks data |
| **Stateless** | No in-memory user state. All state from DB or request payload. |
| **LLM-Agent Ready** | Structured tool calling with LangChain integration and safety guardrails |

### Tech Stack

- **Language**: Python 3.11+
- **Framework**: FastAPI (ASGI)
- **Validation**: Pydantic v2 (strict mode)
- **Database**: SQLAlchemy 2.0 + PostgreSQL (SQLite for dev)
- **Encryption**: AES-256-GCM (cryptography library)
- **Metrics**: Prometheus client
- **Server**: Gunicorn + Uvicorn workers
- **Container**: Docker (python:3.11-slim)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         API Layer (FastAPI)                         │
│  /simulate   /sensitivity-analysis   /health   /health/sla /metrics│
├─────────────────────────────────────────────────────────────────────┤
│                    Security & Input Pipeline                        │
│  input_sanitization → llm_safety → tool_constraints → auth/rate    │
├─────────────────────────────────────────────────────────────────────┤
│                   Orchestration & Tool Calling                      │
│  structured_orchestration → langchain_tools → langchain_adapter     │
├─────────────────────────────────────────────────────────────────────┤
│                      Simulation Service                             │
│  failure_handling.safe_simulation_run() → timeout → SLA tracking    │
├───────────────────────────┬─────────────────────────────────────────┤
│    Simulation Engine      │        Risk & Analysis                  │
│  simulate_subscription    │   stress_testing                        │
│  simulate_credit_strat    │   sensitivity_analysis                  │
│  _stability_score()       │   score_calibration                     │
├───────────────────────────┴─────────────────────────────────────────┤
│                    Persistence & Security                           │
│  persistence (snapshots) → simulation_history → encryption → audit  │
├─────────────────────────────────────────────────────────────────────┤
│                     Observability Layer                             │
│  structured logs → prometheus metrics → error rate → SLA reporting  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Module Reference

### Core Domain (DO NOT MODIFY)

| Module | Lines | Purpose |
|---|---|---|
| `financial_schemas.py` | 339 | Pydantic v2 input models: `FinancialState`, `Commitment`, `SubscriptionOption`, `CreditStrategyRequest` |
| `simulation_schemas.py` | 287 | Output contracts: `SimulationResult`, `SimulationResultSet` |
| `simulation_engine.py` | 267 | Pure-function simulation: `simulate_subscription_options()`, `simulate_credit_strategies()`, `_stability_score()` |

### Risk & Analysis

| Module | Lines | Purpose |
|---|---|---|
| `stress_testing.py` | ~300 | Income drop / expense spike / combined stress scenarios with fragility index |
| `score_calibration.py` | 448 | Stability Score weight calibration via grid search, monotonicity tests, distribution analysis |
| `product_intelligence.py` | ~500 | Sensitivity analysis (5 scenarios), extended schemas (variable income, multi-credit, investment hooks) |

### Tool Calling & Orchestration

| Module | Lines | Purpose |
|---|---|---|
| `structured_orchestration.py` | ~430 | Intent→tool routing, LLM output parsing, structured tool call enforcement |
| `langchain_tools.py` | ~430 | LangChain `StructuredTool` definitions with Pydantic schemas |
| `langchain_adapter.py` | ~300 | Thin adapter wrapping simulation functions for LangChain compatibility |
| `tool_constraints.py` | ~430 | Pre-execution constraint validation (field ranges, consistency checks) |
| `tool_logging.py` | ~350 | Tool invocation logging with latency tracking |

### Security

| Module | Lines | Purpose |
|---|---|---|
| `input_sanitization.py` | 430 | Payload size limits, type enforcement, SQL/script injection blocking, JSON hardening |
| `llm_safety.py` | ~500 | Prompt injection detection, output safety validation, jailbreak prevention |
| `security_compliance.py` | ~500 | AES-256-GCM field encryption, log masking, immutable audit trail |

### Infrastructure

| Module | Lines | Purpose |
|---|---|---|
| `observability.py` | ~460 | Structured JSON logging, 8 Prometheus metrics, error rate monitoring, stability score tracking |
| `failure_handling.py` | ~700 | Failure classification, graceful degradation, tool timeout (2s), SLA enforcement |
| `scaling_readiness.py` | ~500 | Health endpoint, DB-enforced idempotency, rate limiter interface (in-memory + Redis), stateless service |
| `persistence.py` | 740 | SQLAlchemy ORM: `FinancialSnapshotORM`, `CommitmentRecordORM`, CRUD with immutability |
| `simulation_history.py` | 637 | `SimulationHistoryORM` — longitudinal records, trend queries, high-risk filters |

---

## Quick Start

### Local Development

```bash
# 1. Clone
git clone https://github.com/mdfaizee19/HARMONY.git
cd HARMONY

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set encryption key (required for security_compliance)
export FINANCIAL_ENCRYPTION_KEY=$(python3 -c "import os; print(os.urandom(32).hex())")

# 5. Run tests
pytest tests/ -v

# 6. Run any module standalone (each has example usage)
python3 simulation_engine.py
python3 stress_testing.py
python3 product_intelligence.py
python3 observability.py
python3 failure_handling.py
python3 scaling_readiness.py
python3 security_compliance.py
```

### Docker

```bash
# Build
docker build -t financial-engine .

# Run (single instance)
docker run -p 8000:8000 \
  -e FINANCIAL_ENCRYPTION_KEY=$(python3 -c "import os; print(os.urandom(32).hex())") \
  financial-engine

# Horizontal scaling (3 instances)
docker compose up --build --scale app=3
```

---

## API Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Health check (uptime + SLA status) — used by Docker HEALTHCHECK |
| `GET` | `/health/sla` | Full SLA compliance report (avg latency, p95, error rate) |
| `GET` | `/metrics` | Prometheus-compatible metrics endpoint |
| `POST` | `/sensitivity-analysis` | Sensitivity analysis across 5 parameter shift scenarios |

### Health Check Response

```json
{
  "status": "ok",
  "uptime_seconds": 3621.45,
  "within_sla": true
}
```

### SLA Status Response

```json
{
  "avg_latency_ms": 45.2,
  "p95_latency_ms": 120.0,
  "error_rate": 0.02,
  "total_requests": 1500,
  "total_failures": 30,
  "within_sla": true,
  "sla_targets": {
    "response_time_target_ms": 200,
    "response_time_critical_ms": 400,
    "max_error_rate": 0.05,
    "critical_error_rate": 0.10,
    "execution_timeout_s": 2.0
  }
}
```

---

## Tool Calling Flow

When an LLM-based agent receives a user query, the execution flow is:

```
User: "Can I afford a $30/month gym membership?"
  │
  ▼
┌─────────────────────────────────────────────┐
│  1. Input Sanitization                       │
│     • Payload size ≤ 50KB                   │
│     • No SQL/script injection               │
│     • No prototype pollution                │
│     • JSON structure validation             │
└──────────────┬──────────────────────────────┘
               ▼
┌─────────────────────────────────────────────┐
│  2. LLM Safety Layer                         │
│     • Prompt injection detection            │
│     • Jailbreak pattern blocking            │
│     • Output safety validation              │
└──────────────┬──────────────────────────────┘
               ▼
┌─────────────────────────────────────────────┐
│  3. Structured Orchestration                 │
│     • Intent → tool routing                 │
│     • parse_llm_tool_call()                 │
│     • enforce_structured_routing()          │
└──────────────┬──────────────────────────────┘
               ▼
┌─────────────────────────────────────────────┐
│  4. Tool Constraint Validation               │
│     • Field range checks                    │
│     • Cross-field consistency               │
│     • Schema adherence                      │
└──────────────┬──────────────────────────────┘
               ▼
┌─────────────────────────────────────────────┐
│  5. Simulation Execution (with timeout)      │
│     • 2-second hard timeout                 │
│     • Graceful degradation on failure       │
│     • SLA tracking                          │
└──────────────┬──────────────────────────────┘
               ▼
┌─────────────────────────────────────────────┐
│  6. Output Verification                      │
│     • Schema validation (Pydantic v2)       │
│     • Score bounds [0, 100]                 │
│     • Risk flag consistency                 │
└──────────────┬──────────────────────────────┘
               ▼
┌─────────────────────────────────────────────┐
│  7. Persistence & Audit                      │
│     • Encrypted storage (AES-256-GCM)       │
│     • Idempotency check                     │
│     • Audit log (append-only)               │
└──────────────┬──────────────────────────────┘
               ▼
         Structured JSON Response
```

### Available Tools

| Tool Name | Intent | Input Schema |
|---|---|---|
| `subscription_simulation` | Evaluate subscription affordability | `FinancialState` + `List[SubscriptionOption]` |
| `credit_strategy_simulation` | Compare min/partial/full payment | `FinancialState` + `CreditStrategyRequest` |

---

## Schemas

### Input: `FinancialState`

```json
{
  "monthly_income": 5000.0,
  "fixed_expenses": 2000.0,
  "discretionary_expenses": 800.0,
  "emergency_fund": 10000.0,
  "credit_balance": 3000.0,
  "credit_apr": 0.24,
  "commitments": [
    { "id": "insurance", "amount": 300.0, "due_month": 3 }
  ]
}
```

### Output: `SimulationResult`

```json
{
  "option_id": "no-subscription",
  "stability_score": 52.97,
  "liquidity_ratio": 0.59,
  "commitment_coverage": 1.0,
  "interest_cost": 60.0,
  "projected_balance": 1840.0,
  "risk_flags": []
}
```

### Extended: `ExtendedFinancialState` (v3.0.0)

```json
{
  "monthly_income": 0,
  "fixed_expenses": 2000.0,
  "discretionary_expenses": 800.0,
  "emergency_fund": 10000.0,
  "credit_balance": 0,
  "credit_apr": 0,
  "income_streams": [
    { "name": "Salary", "amount": 4000.0, "frequency": "monthly" },
    { "name": "Freelance", "amount": 3000.0, "frequency": "quarterly" }
  ],
  "credit_accounts": [
    { "balance": 2000.0, "apr": 0.22, "minimum_payment": 40.0 },
    { "balance": 1500.0, "apr": 0.18, "minimum_payment": 30.0 }
  ],
  "investment_accounts": [
    { "current_value": 50000.0, "expected_return_rate": 0.08, "volatility": 0.15 }
  ]
}
```

> **Backward compatible**: if expansion fields are omitted, behaves identically to v2.0.0.

---

## Simulation Engine

### Stability Score Formula

```
stability_score = 40% × balance_health
                + 30% × liquidity_health
                + 20% × coverage_health
                + 10% × interest_health
```

| Component | Formula | Range |
|---|---|---|
| `balance_health` | `(projected_balance / income) × 100` | 0–100 |
| `liquidity_health` | `(liquidity_ratio / 2.0) × 100` | 0–100 |
| `coverage_health` | `commitment_coverage × 100` | 0–100 |
| `interest_health` | `100 - (interest_cost / income) × 500` | 0–100 |

### Risk Flags

| Flag | Trigger |
|---|---|
| `high_interest` | `interest_cost > 10% × monthly_income` |
| `low_liquidity` | `liquidity_ratio < 0.5` |
| `commitment_breach` | `commitment_coverage < 1.0` |
| `negative_balance` | `projected_balance < 0` |

---

## Risk & Scoring

### Score Calibration (`score_calibration.py`)

Run calibration to optimize scoring weights:

```bash
python3 score_calibration.py
```

Outputs:
- 20 synthetic financial profiles
- Backtest results with distribution analysis
- Monotonicity verification across 4 axes (income, expenses, APR, commitments)
- Grid-search weight optimization
- Versioned calibration report

---

## Stress Testing

The stress testing module (`stress_testing.py`) evaluates financial resilience under adverse conditions:

| Scenario | Parameters |
|---|---|
| Income Drop | Income reduced by 30% |
| Expense Spike | Expenses increased by 25% |
| Combined Stress | Income -30% AND expenses +25% |

**Output**: fragility index, per-scenario stability scores, risk escalation flags.

```bash
python3 stress_testing.py
```

---

## Sensitivity Analysis

The sensitivity analysis endpoint (`product_intelligence.py`) measures how the Stability Score responds to parameter shifts:

| Scenario | Shift |
|---|---|
| `income_plus_10` | Monthly income +10% |
| `income_minus_10` | Monthly income -10% |
| `expenses_plus_10` | All expenses +10% |
| `expenses_minus_10` | All expenses -10% |
| `apr_plus_5pp` | APR +5 percentage points (capped at 100%) |

Returns the **most sensitive factor** (highest |delta|).

```bash
python3 product_intelligence.py
```

---

## Security & Compliance

### Encryption at Rest

- **Algorithm**: AES-256-GCM
- **Key source**: `FINANCIAL_ENCRYPTION_KEY` environment variable (64 hex chars)
- **Encrypted fields**: `pre_simulation_state`, `post_simulation_result`, `input_payload`, `output_payload`, commitment amounts
- **Storage format**: `nonce (12 bytes) || ciphertext + tag`

### Log Masking

Sensitive fields are automatically masked in all logs:

```json
{
  "correlation_id": "req-001",
  "monthly_income": "***",
  "emergency_fund": "***",
  "stability_score": 78.5,
  "risk_flags": ["low_liquidity"]
}
```

Masked fields include: `monthly_income`, `fixed_expenses`, `discretionary_expenses`, `emergency_fund`, `credit_balance`, `credit_apr`, `interest_cost`, `amount`, and all encrypted payloads.

### Audit Trail

- **Append-only**: UPDATE and DELETE operations are blocked via SQLAlchemy event listeners
- **Actions tracked**: `SIMULATION_EXECUTED`, `COMMITMENT_CREATED`, `COMMITMENT_UPDATED`, `COMMITMENT_DELETED`
- **No financial values in metadata**: only `stability_score`, `intent_type`, `risk_flag_count`

---

## Observability

### Prometheus Metrics

| Metric | Type | Description |
|---|---|---|
| `simulation_requests_total` | Counter | Total simulation requests |
| `simulation_success_total` | Counter | Successful simulations |
| `simulation_failure_total` | Counter | Failed simulations |
| `tool_invocations_total` | Counter | Per-tool invocation count |
| `tool_errors_total` | Counter | Per-tool error count |
| `simulation_latency_seconds` | Histogram | Request latency distribution |
| `stability_score_distribution` | Histogram | Score distribution (0–100 buckets) |
| `stability_score_average` | Gauge | Rolling average (last 500) |

### Structured Logging

All logs are JSON-formatted with:
- `timestamp` (UTC ISO 8601)
- `correlation_id`
- `event_type`
- `tool_name`
- `status`
- `latency_ms`
- `error_code` (if applicable)

---

## Failure Handling & SLA

### Failure Categories

| Category | Degradable? | Response |
|---|---|---|
| `TOOL_EXECUTION_ERROR` | ✅ | Fallback: score=50, all risks UNKNOWN |
| `TOOL_TIMEOUT` | ✅ | Fallback after 2s hard timeout |
| `OUTPUT_VERIFICATION_FAILURE` | ✅ | Fallback response |
| `INTERNAL_SERVER_ERROR` | ✅ | Catch-all fallback |
| `VALIDATION_FAILURE` | ❌ | Hard rejection, no fallback |

### Degraded Response Shape

```json
{
  "status": "degraded",
  "correlation_id": "...",
  "failure_category": "TOOL_TIMEOUT",
  "message": "Simulation timed out after 2.0s.",
  "fallback": {
    "stability_score": 50,
    "risk_flags": {
      "liquidity_risk": "UNKNOWN",
      "commitment_risk": "UNKNOWN",
      "interest_risk": "UNKNOWN",
      "burn_rate_risk": "UNKNOWN",
      "overall_risk": "UNKNOWN"
    }
  }
}
```

### SLA Thresholds

| Metric | Target | Critical |
|---|---|---|
| Response latency | ≤ 200ms | > 400ms |
| Error rate | ≤ 5% | > 10% |
| Execution timeout | 2s hard limit | — |

---

## Scaling & Deployment

### Container Architecture

- **Base image**: `python:3.11-slim`
- **Non-root user**: `appuser` (UID 1000)
- **Server**: Gunicorn + UvicornWorker (2 workers)
- **Health check**: `GET /health` every 30s

### Horizontal Scaling

```bash
docker compose up --build --scale app=3
```

Scaling guarantees:
- **Idempotency**: DB `UNIQUE(user_id, idempotency_key)` — any container can serve any request
- **Rate limiting**: Swappable backend (`InMemoryRateLimiter` → `RedisRateLimiter`)
- **Stateless**: Zero in-process memory dependencies for correctness
- **Metrics**: Per-instance `/metrics`, aggregated by Prometheus scraper

### Rate Limiter Backends

| Backend | Use Case |
|---|---|
| `InMemoryRateLimiter` | Development / single-instance |
| `RedisRateLimiter` | Production multi-container (sorted set sliding window) |

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Current test suites
# tests/test_simulation_engine.py — 15 tests
#   • Subscription scenarios (5)
#   • Credit strategy scenarios (5)
#   • Edge cases (3)
#   • Floating-point consistency (1)
#   • Output invariants (1)
#
# tests/test_stress_testing.py — 16 tests
#   • Stress test core (3)
#   • Scenario details (4)
#   • Fragility & risk (3)
#   • Immutability & determinism (3)
#   • Output invariants (1)
#   • Model validation (2)
```

Every module also has a standalone `if __name__ == "__main__"` section with verification examples:

```bash
python3 simulation_engine.py      # Simulation examples
python3 stress_testing.py         # Stress test examples
python3 observability.py          # Metrics + logging demo
python3 failure_handling.py       # Failure modes + SLA demo
python3 scaling_readiness.py      # Health + idempotency demo
python3 security_compliance.py    # Encryption + audit demo
python3 product_intelligence.py   # Sensitivity + expansion demo
python3 input_sanitization.py     # Sanitization examples
python3 score_calibration.py      # Calibration report
```

---

## Configuration

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `FINANCIAL_ENCRYPTION_KEY` | Yes (production) | 64-character hex string (32 bytes) for AES-256-GCM |
| `DATABASE_URL` | No | SQLAlchemy connection string (default: SQLite in-memory) |
| `PYTHONUNBUFFERED` | Recommended | Set to `1` for container logging |

### Generate Encryption Key

```bash
python3 -c "import os; print(os.urandom(32).hex())"
```

---

## Project Structure

```
TOOLS/
├── financial_schemas.py          # Input models (FinancialState, Commitment, etc.)
├── simulation_schemas.py         # Output contracts (SimulationResult, SimulationResultSet)
├── simulation_engine.py          # Deterministic simulation engine
├── stress_testing.py             # Stress testing scenarios
├── score_calibration.py          # Stability Score calibration & validation
├── product_intelligence.py       # Sensitivity analysis & expansion hooks
├── structured_orchestration.py   # Intent→tool routing & LLM output parsing
├── langchain_tools.py            # LangChain StructuredTool definitions
├── langchain_adapter.py          # LangChain adapter layer
├── tool_constraints.py           # Pre-execution constraint validation
├── tool_logging.py               # Tool invocation logging
├── input_sanitization.py         # Input sanitization (size, injection, structure)
├── llm_safety.py                 # LLM prompt injection & jailbreak prevention
├── security_compliance.py        # AES-256-GCM encryption, masking, audit trail
├── observability.py              # Structured logging & Prometheus metrics
├── failure_handling.py           # Failure classification, degradation, SLA
├── scaling_readiness.py          # Health endpoint, idempotency, rate limiting
├── persistence.py                # SQLAlchemy ORM models & CRUD
├── simulation_history.py         # Simulation history persistence & queries
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Production container
├── docker-compose.yml            # Multi-container orchestration
├── .dockerignore                 # Docker build exclusions
├── .gitignore                    # Git exclusions
└── tests/
    ├── test_simulation_engine.py # 15 simulation tests
    └── test_stress_testing.py    # 16 stress testing tests
```

---

## Contributing

1. **Do not modify** core domain files: `simulation_engine.py`, `financial_schemas.py`, `simulation_schemas.py`
2. **Do not modify** scoring formula, risk logic, or tool math
3. All new features must be backward compatible
4. Every module must have a standalone `__main__` verification section
5. All financial values must be encrypted before persistence
6. No sensitive data in logs — use `mask_sensitive_data()` from `security_compliance.py`
7. Run `pytest tests/ -v` and confirm 31 tests pass before pushing

---

## License

Internal use only. Contact the repository owner for licensing information.
