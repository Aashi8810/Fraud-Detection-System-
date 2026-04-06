import numpy as np
import shap
from typing import List, Dict, Any


# ─────────────────────────────────────────────────────────────────────────────
# SHAP Explainer factory
#
# WHY TreeExplainer (not KernelExplainer or LinearExplainer):
#   - TreeExplainer is exact — it computes true Shapley values, not
#     approximations. For tree-based models (XGBoost, LightGBM, RF),
#     it uses a specialised algorithm that is O(TLD²) where T=trees,
#     L=leaves, D=depth. For our model, this runs in milliseconds.
#   - KernelExplainer is model-agnostic but 100-1000x slower — not viable
#     for real-time API inference.
#   - LinearExplainer works only for linear models.
#   Rule: always use the most specific explainer for your model type.
# ─────────────────────────────────────────────────────────────────────────────

def build_explainer(model) -> shap.TreeExplainer:
    """
    Build a SHAP TreeExplainer for a fitted tree-based model.

    Args:
        model: A fitted XGBoost, LightGBM, or RandomForest model.

    Returns:
        shap.TreeExplainer fitted to the model.
    """
    return shap.TreeExplainer(model)


def compute_shap_values(explainer: shap.TreeExplainer,
                         X: np.ndarray) -> np.ndarray:
    """
    Compute SHAP values for a batch of processed feature vectors.

    Args:
        explainer   : Fitted TreeExplainer.
        X           : Processed feature matrix (numpy array), shape (n, features).

    Returns:
        SHAP values array, shape (n, features).
        Each value represents how much that feature pushed the prediction
        up (positive) or down (negative) from the base rate.
    """
    return explainer.shap_values(X)


# ─────────────────────────────────────────────────────────────────────────────
# Local explanation → human-readable narrative
#
# WHY a narrative function in addition to raw SHAP values:
#   Raw SHAP values ("+1.70 for log_amount") mean nothing to a fraud analyst
#   or compliance officer. They need "unusually high transaction amount" in
#   plain language. This function converts SHAP numbers into a structured
#   explanation that can go directly into:
#     - The API response JSON
#     - A fraud alert dashboard
#     - A compliance audit trail
# ─────────────────────────────────────────────────────────────────────────────

# Human-readable labels for feature names
# WHY not just use the raw column names:
#   'log_amount' means nothing to a non-technical reviewer.
#   'Previous_Fraudulent_Transactions' is too long for a UI card.
FEATURE_LABELS: Dict[str, str] = {
    'log_amount'                             : 'Transaction amount',
    'Time_of_Transaction'                    : 'Time of transaction',
    'Previous_Fraudulent_Transactions'       : 'Prior fraud history',
    'Account_Age'                            : 'Account age',
    'Number_of_Transactions_Last_24H'        : 'Transaction velocity (24h)',
    'is_unknown_device'                      : 'Unknown device',
    'is_invalid_payment'                     : 'Invalid payment method',
    'high_amount_new_account'                : 'High amount on new account',
    'velocity_risk_very_high'                : 'Very high transaction velocity',
    'Transaction_Type_Online Purchase'       : 'Online purchase type',
}


def get_risk_factors(shap_vals: np.ndarray,
                     feature_names: List[str],
                     top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Convert SHAP values for one transaction into a structured risk factor list.

    Args:
        shap_vals     : 1D array of SHAP values for a single transaction.
        feature_names : Ordered list of feature names (same order as shap_vals).
        top_n         : Number of top contributing features to return.

    Returns:
        List of dicts, each with:
          'feature'    : raw feature name
          'label'      : human-readable label
          'shap_value' : raw SHAP value (positive = increases fraud risk)
          'direction'  : 'increases_risk' | 'decreases_risk'
          'impact'     : 'high' | 'medium' | 'low'

    WHY return both raw SHAP and label:
        The API client might want to display the label in a UI card and
        log the raw feature name for debugging — both are included.
    """
    # Sort by absolute SHAP magnitude — largest impact first
    sorted_indices = np.argsort(np.abs(shap_vals))[::-1]
    top_indices    = sorted_indices[:top_n]

    factors = []
    for idx in top_indices:
        val  = float(shap_vals[idx])
        name = feature_names[idx]
        abs_val = abs(val)

        # Classify impact magnitude
        if abs_val >= 0.5:
            impact = 'high'
        elif abs_val >= 0.1:
            impact = 'medium'
        else:
            impact = 'low'

        factors.append({
            'feature'    : name,
            'label'      : FEATURE_LABELS.get(name, name.replace('_', ' ').title()),
            'shap_value' : round(val, 4),
            'direction'  : 'increases_risk' if val > 0 else 'decreases_risk',
            'impact'     : impact,
        })

    return factors


def build_explanation_narrative(
        fraud_probability : float,
        shap_vals         : np.ndarray,
        feature_names     : List[str],
        threshold         : float = 0.30,
        top_n             : int   = 3
) -> Dict[str, Any]:
    """
    Build a complete, structured explanation for a single transaction.

    This is the function the FastAPI calls to generate the explanation
    payload attached to each prediction response.

    Args:
        fraud_probability : Model output probability of fraud (0–1).
        shap_vals         : 1D SHAP value array for this transaction.
        feature_names     : Feature names in same order as shap_vals.
        threshold         : Decision threshold (default 0.30 from Phase 4).
        top_n             : Top risk factors to include in the narrative.

    Returns:
        Dict with full explanation, ready to serialize as JSON.

    Example output:
        {
          "decision": "FRAUD",
          "fraud_probability": 0.9581,
          "threshold": 0.30,
          "top_risk_factors": [
            {"feature": "log_amount", "label": "Transaction amount",
             "shap_value": 1.7008, "direction": "increases_risk", "impact": "high"},
            ...
          ],
          "summary": "Flagged due to: Transaction amount (+1.70), ..."
        }
    """
    is_fraud    = fraud_probability >= threshold
    risk_factors = get_risk_factors(shap_vals, feature_names, top_n=top_n)

    # Build a one-sentence summary for quick reading
    top_positive = [f for f in risk_factors if f['direction'] == 'increases_risk']
    if top_positive:
        parts   = [f"{f['label']} ({f['shap_value']:+.2f})" for f in top_positive[:3]]
        summary = f"{'Flagged' if is_fraud else 'Reviewed'} due to: {', '.join(parts)}"
    else:
        summary = "No strong risk factors detected — transaction appears legitimate"

    return {
        'decision'          : 'FRAUD' if is_fraud else 'LEGITIMATE',
        'fraud_probability' : round(fraud_probability, 4),
        'threshold'         : threshold,
        'is_fraud'          : bool(is_fraud),
        'top_risk_factors'  : risk_factors,
        'summary'           : summary,
    }
