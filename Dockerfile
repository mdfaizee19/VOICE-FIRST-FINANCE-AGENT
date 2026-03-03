# ─────────────────────────────────────────────────────────────
#  Financial Simulation Engine — Dockerfile
#
#  Production-grade container:
#    - Python 3.11 slim base (minimal attack surface)
#    - Non-root user (appuser)
#    - Gunicorn + UvicornWorker (multi-process ASGI)
#    - Docker HEALTHCHECK
#    - No dev dependencies, no caches
# ─────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# ── Env ───────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ── System deps ───────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# ── Non-root user ─────────────────────────────────────────────
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --create-home appuser

# ── Workdir ───────────────────────────────────────────────────
WORKDIR /app

# ── Dependencies (cached layer) ──────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ─────────────────────────────────────────
COPY *.py .

# ── Ownership ─────────────────────────────────────────────────
RUN chown -R appuser:appuser /app

# ── Switch to non-root ────────────────────────────────────────
USER appuser

# ── Expose ────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ──────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────
CMD ["gunicorn", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "-w", "2", \
     "-b", "0.0.0.0:8000", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--timeout", "30", \
     "server:app"]
