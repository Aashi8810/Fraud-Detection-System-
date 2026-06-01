# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — Fraud Detection API
#
# WHY python:3.11-slim (not python:3.11):
#   The full Python image is ~900MB. The slim variant strips docs,
#   test suites, and unused locales — bringing it to ~130MB.
#   For a production API container that might be pulled millions of times,
#   image size directly affects deployment speed and storage cost.
#
# WHY a multi-stage build pattern would be even better:
#   For this project we use a single stage for simplicity.
#   In production you'd have a 'builder' stage (installs build tools + deps)
#   and a 'runtime' stage (copies only the compiled artifacts) to get
#   an even smaller final image. This is documented in the README.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Metadata labels — good practice for Docker Hub and internal registries
LABEL maintainer="fraud-detection-project"
LABEL description="Fraud Detection API with SHAP Explainability"
LABEL version="1.0.0"

# ── System dependencies ───────────────────────────────────────────────────────
# WHY these packages:
#   libgomp1 → required by LightGBM and XGBoost for OpenMP parallelisation
#   curl     → used in the Docker HEALTHCHECK command below
# WHY --no-install-recommends + clean in same RUN:
#   Docker layers are immutable. If we clean in a separate RUN layer,
#   the apt cache still exists in the previous layer and inflates image size.
#   Combining in one RUN means the layer only contains what we want.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
# WHY /app (not root /):
#   Running as root in the filesystem root is a security anti-pattern.
#   /app is the conventional directory for application code in containers.
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# WHY COPY requirements.txt BEFORE copying source code:
#   Docker caches each layer. If we copy source code first, ANY code change
#   invalidates the layer and re-installs all dependencies from scratch.
#   Copying requirements.txt first means dependency installation is cached
#   as long as requirements.txt hasn't changed — much faster rebuilds.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy application code ─────────────────────────────────────────────────────
COPY src/     ./src/
COPY api/     ./api/
COPY models/  ./models/

# ── Non-root user ─────────────────────────────────────────────────────────────
# WHY run as non-root:
#   If a vulnerability in the API or its dependencies allows code execution,
#   an attacker running as root has full container control. A non-root user
#   limits the blast radius significantly.
RUN useradd --no-create-home --shell /bin/false appuser \
    && chown -R appuser:appuser /app
USER appuser

# ── Environment variables ─────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODELS_DIR=/app/models \
    PORT=8000

# ── Healthcheck ───────────────────────────────────────────────────────────────
# WHY HEALTHCHECK:
#   Docker and orchestrators (Kubernetes, ECS) use this to know if the
#   container is actually ready to serve traffic, not just running.
#   Without it, a container where the model failed to load would still
#   receive traffic.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

EXPOSE ${PORT}

# ── Start command ─────────────────────────────────────────────────────────────
# WHY uvicorn with --host 0.0.0.0:
#   By default uvicorn binds to 127.0.0.1 (localhost only).
#   Inside a container, we need 0.0.0.0 to accept connections from outside.
# WHY --workers 1:
#   Multiple workers would each load the model into memory separately.
#   For a model this size, 1 worker is fine. For production scale, use
#   gunicorn as the process manager with multiple uvicorn workers.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
