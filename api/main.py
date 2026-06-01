"""
api/main.py
───────────
FastAPI application entry point — defines all HTTP routes.

WHY FastAPI (not Flask):
    1. Automatic OpenAPI docs at /docs (Swagger UI) — zero extra work
    2. Async-native: handles concurrent requests without blocking
    3. Pydantic integration: request/response validation is automatic
    4. Type hints drive everything: the schema IS the documentation
    5. Production adoption: used by Uber, Microsoft, Netflix — not a toy

Running locally:
    uvicorn api.main:app --reload --port 8000

Running in Docker:
    docker-compose up
    Then open: http://localhost:8000/docs
"""

import sys
import os
# Ensure project root is on path so src/ imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import TransactionRequest, PredictionResponse, HealthResponse, ErrorResponse
from api.predictor import FraudPredictor

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan: load the model ONCE at startup, release at shutdown
#
# WHY lifespan (not @app.on_event which is deprecated):
#   The lifespan context manager is FastAPI's modern pattern for startup/
#   shutdown logic. Code before 'yield' runs at startup; code after runs
#   at shutdown. The predictor is stored in app.state so all route handlers
#   can access it without globals.
#
# WHY store in app.state (not a global variable):
#   app.state is the FastAPI-endorsed place for shared application state.
#   It survives across requests, is accessible in routes via request.app.state,
#   and is properly scoped to the application lifecycle. Globals work but
#   are harder to test and can cause issues with multiple workers.
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts on startup, clean up on shutdown."""
    # ── Startup ───────────────────────────────────────────────────────────────
    logger.info("Starting up — loading model artifacts...")
    models_dir = os.environ.get("MODELS_DIR", "models")
    app.state.predictor = FraudPredictor(models_dir=models_dir)
    logger.info("Startup complete — API ready to serve predictions")

    yield  # Application runs here

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("Shutting down — releasing resources")
    del app.state.predictor


# ─────────────────────────────────────────────────────────────────────────────
# App instantiation
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Fraud Detection API",
    description = (
        "Real-time transaction fraud scoring with SHAP-based explainability.\n\n"
        "**How it works:**\n"
        "1. POST a raw transaction JSON to `/predict`\n"
        "2. The API preprocesses it through the trained feature pipeline\n"
        "3. XGBoost scores it for fraud probability\n"
        "4. SHAP values explain which features drove the decision\n"
        "5. A structured response is returned with decision + explanation\n\n"
        "**Decision threshold:** 0.30 (tuned in Phase 4 for optimal F1)"
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
)

# ── CORS middleware ───────────────────────────────────────────────────────────
# WHY CORS: if a browser-based dashboard calls this API, CORS headers
# must be present or the browser will block the request.
# In production, replace "*" with your actual frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model = HealthResponse,
    summary        = "Health check",
    description    = "Returns API status and loaded model metadata. Use for liveness probes in Docker/Kubernetes.",
    tags           = ["Operations"]
)
async def health(request: Request):
    """
    Liveness check endpoint.

    WHY a /health endpoint:
        Load balancers, Docker healthchecks, and Kubernetes probes all
        need a lightweight endpoint to verify the service is alive and
        the model is loaded. This is standard practice in any deployed API.
    """
    predictor: FraudPredictor = request.app.state.predictor
    return predictor.health_info


@app.post(
    "/predict",
    response_model   = PredictionResponse,
    summary          = "Predict fraud risk",
    description      = (
        "Score a single transaction for fraud risk.\n\n"
        "Returns the fraud probability, a FRAUD/LEGITIMATE decision, "
        "and the top SHAP-based risk factors explaining the decision."
    ),
    responses        = {422: {"model": ErrorResponse, "description": "Validation error — invalid input"}},
    tags             = ["Prediction"]
)
async def predict(transaction: TransactionRequest, request: Request):
    """
    Main prediction endpoint.

    Accepts a raw transaction JSON, runs it through the full ML pipeline,
    and returns a structured fraud assessment with explanations.

    **Example flow:**
    - Input: raw transaction with amount, type, device, user history
    - Processing: imputation → feature engineering → XGBoost scoring → SHAP
    - Output: { decision, fraud_probability, top_risk_factors, summary }
    """
    predictor: FraudPredictor = request.app.state.predictor

    # Convert Pydantic model → dict
    # WHY .model_dump(): this serialises the Pydantic model to a plain Python dict,
    # including None for optional fields not provided by the caller.
    # The predictor expects a dict; the pipeline handles None values via imputation.
    transaction_dict = transaction.model_dump()

    result = predictor.predict(transaction_dict)
    return result


@app.post(
    "/predict/batch",
    summary     = "Batch fraud prediction",
    description = "Score multiple transactions in one request. Returns a list of predictions.",
    tags        = ["Prediction"]
)
async def predict_batch(transactions: list[TransactionRequest], request: Request):
    """
    Batch prediction endpoint — useful for testing or bulk scoring.

    WHY a batch endpoint:
        Some use cases (overnight batch scoring, testing 100 transactions
        at once) don't need real-time responses. A batch endpoint reduces
        HTTP overhead. In production this would use async processing;
        here it's synchronous for simplicity.

    Limit: 100 transactions per request (to avoid memory issues).
    """
    if len(transactions) > 100:
        raise HTTPException(
            status_code = 400,
            detail      = "Batch size exceeds limit of 100 transactions per request"
        )

    predictor: FraudPredictor = request.app.state.predictor
    results = [predictor.predict(t.model_dump()) for t in transactions]
    return {
        "count"       : len(results),
        "predictions" : results
    }


@app.get(
    "/",
    summary     = "API root",
    description = "Welcome message with links to documentation.",
    tags        = ["Operations"]
)
async def root():
    return {
        "message"  : "Fraud Detection API v1.0.0",
        "docs"     : "/docs",
        "health"   : "/health",
        "predict"  : "POST /predict",
    }
