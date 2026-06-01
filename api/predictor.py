"""
api/predictor.py
────────────────
Model loading and inference logic — the core engine of the API.

WHY a separate predictor module (not inside main.py):
    main.py should only handle HTTP routing — receive request, call predictor,
    return response. The prediction logic lives here so it can be:
      - Unit tested independently of the HTTP layer
      - Swapped out (new model version) without touching routing code
      - Imported by test scripts or batch scoring scripts directly

This is the "Service Layer" pattern in API design:
    HTTP Layer (main.py) → Service Layer (predictor.py) → ML Layer (src/)
"""

import numpy as np
import pandas as pd
import joblib
import logging
from pathlib import Path
from typing import Dict, Any

from src.explain import build_explainer, build_explanation_narrative

# Configure structured logging
# WHY logging (not print): in production, logs are collected by services
# like CloudWatch, Datadog, or ELK. print() statements don't get picked up.
logging.basicConfig(
    level  = logging.INFO,
    format = '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)


class FraudPredictor:
    """
    Loads model artifacts once at startup and exposes a predict() method
    that handles the full inference pipeline:
        raw dict → DataFrame → pipeline.transform() → model.predict_proba()
        → shap_values → build_explanation_narrative() → structured response

    WHY a class (not module-level functions):
        State management. The model, pipeline, explainer, and config must all
        be loaded once at startup and reused across requests. A class holds
        this state cleanly. Module-level globals work too, but classes are
        easier to test (you can instantiate with a mock model) and easier
        to extend (e.g., add model versioning or A/B testing later).

    Lifecycle:
        app startup → FraudPredictor() → loads artifacts → ready
        POST /predict → predictor.predict(data) → returns response dict
    """

    def __init__(self, models_dir: str = "models"):
        """
        Load all artifacts from disk on instantiation.

        Args:
            models_dir: Path to the directory containing saved .joblib files.

        WHY load everything eagerly (at startup, not per-request):
            Loading a 10MB XGBoost model + SHAP explainer takes ~500ms.
            If we loaded per-request, every prediction would add 500ms latency.
            Load once at startup: first request is instant.
        """
        self.models_dir = Path(models_dir)
        self._load_artifacts()

    def _load_artifacts(self):
        """Load all model artifacts. Called once at startup."""
        logger.info(f"Loading model artifacts from {self.models_dir}/")

        self.model         = joblib.load(self.models_dir / "xgboost_best.joblib")
        self.pipeline      = joblib.load(self.models_dir / "preprocessing_pipeline.joblib")
        self.feature_names = joblib.load(self.models_dir / "feature_names.joblib")
        self.config        = joblib.load(self.models_dir / "model_config.joblib")

        # Build SHAP explainer from loaded model
        # WHY rebuild instead of loading saved explainer:
        #   The explainer wraps the model object. If we save and reload it,
        #   we need the model to be loaded first anyway. Rebuilding is cleaner
        #   and avoids potential joblib version mismatch issues.
        self.explainer = build_explainer(self.model)
        self.threshold = self.config["best_threshold"]

        logger.info(
            f"Loaded: {self.config['model_name']} | "
            f"PR-AUC={self.config['pr_auc']} | "
            f"threshold={self.threshold}"
        )

    def predict(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the full prediction pipeline for one transaction.

        Args:
            transaction: Dict matching the TransactionRequest schema fields.
                         Comes from Pydantic after validation — types are guaranteed.

        Returns:
            Dict matching the PredictionResponse schema fields.

        Steps:
            1. Convert dict to single-row DataFrame (pipeline expects DataFrame)
            2. Transform via fitted preprocessing pipeline
            3. Score with XGBoost → fraud probability
            4. Compute SHAP values for this transaction
            5. Build explanation narrative
            6. Return structured response
        """
        transaction_id = transaction.get("Transaction_ID")
        logger.info(f"Scoring transaction: {transaction_id}")

        # ── Step 1: dict → DataFrame ──────────────────────────────────────────
        # WHY a DataFrame (not a numpy array):
        #   Our pipeline's FraudFeatureEngineer uses column names internally
        #   (e.g., X['Transaction_Amount']). A numpy array has no column names.
        #   The pipeline was fitted on DataFrames and must be called with them.
        input_df = pd.DataFrame([transaction])

        # ── Step 2: Feature engineering + preprocessing ───────────────────────
        # This applies ALL transformations learned during training:
        # imputation, log transform, binning, one-hot encoding, scaling.
        X_proc = self.pipeline.transform(input_df)  # shape: (1, 32)

        # ── Step 3: Fraud probability ─────────────────────────────────────────
        fraud_probability = float(self.model.predict_proba(X_proc)[0, 1])

        # ── Step 4: SHAP explanation ──────────────────────────────────────────
        # WHY compute SHAP for every prediction (not just flagged ones):
        #   Regulators and analysts want to understand EVERY decision,
        #   not just fraud alerts. A declined transaction also needs justification.
        shap_vals = self.explainer.shap_values(X_proc)[0]  # shape: (32,)

        # ── Step 5: Build human-readable explanation ──────────────────────────
        explanation = build_explanation_narrative(
            fraud_probability = fraud_probability,
            shap_vals         = shap_vals,
            feature_names     = self.feature_names,
            threshold         = self.threshold,
            top_n             = 5
        )

        logger.info(
            f"TxID={transaction_id} | "
            f"P(fraud)={fraud_probability:.4f} | "
            f"decision={explanation['decision']}"
        )

        # ── Step 6: Assemble response ─────────────────────────────────────────
        return {
            "transaction_id"    : transaction_id,
            "decision"          : explanation["decision"],
            "is_fraud"          : explanation["is_fraud"],
            "fraud_probability" : round(fraud_probability, 4),
            "threshold"         : self.threshold,
            "top_risk_factors"  : explanation["top_risk_factors"],
            "summary"           : explanation["summary"],
            "model_version"     : f"{self.config['model_name']}-{self.config['imbalance_strategy']}",
        }

    @property
    def is_loaded(self) -> bool:
        """True if model artifacts are loaded and ready."""
        return hasattr(self, 'model') and self.model is not None

    @property
    def health_info(self) -> Dict[str, Any]:
        """Returns metadata for the /health endpoint."""
        return {
            "status"       : "ok",
            "model_loaded" : self.is_loaded,
            "model_name"   : self.config.get("model_name", "unknown"),
            "pr_auc"       : self.config.get("pr_auc", 0.0),
            "threshold"    : self.threshold,
        }
