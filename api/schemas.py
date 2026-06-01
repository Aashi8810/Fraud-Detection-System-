"""
api/schemas.py
──────────────
Pydantic models that define the shape of every request and response
flowing through the API.

WHY Pydantic for validation:
    FastAPI uses Pydantic under the hood. When a request hits POST /predict,
    FastAPI automatically:
      1. Parses the JSON body
      2. Validates every field against the schema defined here
      3. Returns a 422 Unprocessable Entity with a detailed error message
         if validation fails — BEFORE the request ever reaches our model code.

    This means we never call model.predict() on garbage input. The schema
    is the contract between the API consumer and our model.

WHY Optional fields with None defaults:
    The dataset has ~5% missing values across Transaction_Amount,
    Time_of_Transaction, Device_Used, Location, and Payment_Method.
    Making these Optional in the schema mirrors reality: a real payment
    system might not always capture every field. The preprocessing pipeline
    handles imputation — the API schema just needs to accept None gracefully.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from enum import Enum


# ─────────────────────────────────────────────────────────────────────────────
# Enums — constrain categorical inputs to valid values
#
# WHY Enum instead of just str:
#   If the API accepts any string, a typo ("UPi" instead of "UPI") silently
#   falls through to the pipeline's mode imputation — you get a wrong answer
#   with no error. Enum validation catches this at the schema layer,
#   returning a clear error to the caller.
# ─────────────────────────────────────────────────────────────────────────────

class TransactionType(str, Enum):
    atm_withdrawal  = "ATM Withdrawal"
    bill_payment    = "Bill Payment"
    bank_transfer   = "Bank Transfer"
    pos_payment     = "POS Payment"
    online_purchase = "Online Purchase"

class DeviceUsed(str, Enum):
    desktop        = "Desktop"
    mobile         = "Mobile"
    tablet         = "Tablet"
    unknown_device = "Unknown Device"

class CityLocation(str, Enum):
    boston        = "Boston"
    new_york      = "New York"
    seattle       = "Seattle"
    chicago       = "Chicago"
    houston       = "Houston"
    los_angeles   = "Los Angeles"
    miami         = "Miami"
    san_francisco = "San Francisco"

class PaymentMethod(str, Enum):
    upi            = "UPI"
    debit_card     = "Debit Card"
    net_banking    = "Net Banking"
    credit_card    = "Credit Card"
    invalid_method = "Invalid Method"

class ImpactLevel(str, Enum):
    high   = "high"
    medium = "medium"
    low    = "low"

class RiskDirection(str, Enum):
    increases_risk = "increases_risk"
    decreases_risk = "decreases_risk"


# ─────────────────────────────────────────────────────────────────────────────
# Request schema — what the caller sends to POST /predict
# ─────────────────────────────────────────────────────────────────────────────

class TransactionRequest(BaseModel):
    """
    A single transaction to be scored for fraud risk.

    All fields mirror the original dataset columns, with appropriate
    types and validation constraints. Optional fields reflect real-world
    data quality limitations — the pipeline handles imputation.
    """

    # Identifiers — not used by the model but useful for logging / audit
    Transaction_ID : Optional[str]   = Field(None,  description="Unique transaction identifier")
    User_ID        : Optional[int]   = Field(None,  description="User identifier", ge=1000, le=9999)

    # Core transaction features
    Transaction_Amount : Optional[float] = Field(
        None, description="Transaction amount in USD", ge=0,
        json_schema_extra={"example": 2896.04})

    Transaction_Type   : TransactionType = Field(
        ..., description="Type of transaction",
        json_schema_extra={"example": "POS Payment"})

    Time_of_Transaction : Optional[float] = Field(
        None, description="Hour of day (0–23)", ge=0, le=23,
        json_schema_extra={"example": 13.0})

    Device_Used        : Optional[DeviceUsed] = Field(
        None, description="Device used for transaction",
        json_schema_extra={"example": "Tablet"})

    Location           : Optional[CityLocation] = Field(
        None, description="Transaction location (city)",
        json_schema_extra={"example": "Seattle"})

    # Behavioural features
    Previous_Fraudulent_Transactions : int = Field(
        ..., description="Number of prior fraudulent transactions by this user",
        ge=0, le=10, json_schema_extra={"example": 1})

    Account_Age        : int = Field(
        ..., description="Account age in months",
        ge=0, le=200, json_schema_extra={"example": 100})

    Number_of_Transactions_Last_24H : int = Field(
        ..., description="Number of transactions in the last 24 hours",
        ge=0, le=100, json_schema_extra={"example": 13})

    Payment_Method     : Optional[PaymentMethod] = Field(
        None, description="Payment method used",
        json_schema_extra={"example": "Net Banking"})

    @field_validator('Transaction_Amount')
    @classmethod
    def amount_reasonable(cls, v):
        # WHY this validator: amounts above $100k are almost certainly
        # data errors, not real transactions. Reject early rather than
        # letting a $9,999,999 value pass through the pipeline.
        if v is not None and v > 100_000:
            raise ValueError('Transaction_Amount exceeds $100,000 — likely a data error')
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "Transaction_ID": "T42053",
                "User_ID": 3580,
                "Transaction_Amount": 2896.04,
                "Transaction_Type": "POS Payment",
                "Time_of_Transaction": 13.0,
                "Device_Used": "Tablet",
                "Location": "Seattle",
                "Previous_Fraudulent_Transactions": 1,
                "Account_Age": 100,
                "Number_of_Transactions_Last_24H": 13,
                "Payment_Method": "Net Banking"
            }
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# Response schemas — what the API returns
# ─────────────────────────────────────────────────────────────────────────────

class RiskFactor(BaseModel):
    """One feature's contribution to the fraud prediction."""
    feature    : str          = Field(..., description="Raw feature name")
    label      : str          = Field(..., description="Human-readable label")
    shap_value : float        = Field(..., description="SHAP contribution (positive=increases risk)")
    direction  : RiskDirection= Field(..., description="Whether feature increases or decreases fraud risk")
    impact     : ImpactLevel  = Field(..., description="Magnitude of impact: high/medium/low")


class PredictionResponse(BaseModel):
    """
    Full prediction response returned by POST /predict.

    Designed to be self-explanatory to a fraud analyst:
      - decision:           human-readable verdict
      - fraud_probability:  exact model score
      - threshold:          the cutoff used for this decision
      - top_risk_factors:   which features drove the prediction
      - summary:            one-sentence explanation
      - transaction_id:     echo back for correlation
      - model_version:      for audit trail
    """
    transaction_id    : Optional[str] = Field(None,  description="Echoed from request")
    decision          : str           = Field(...,   description="FRAUD or LEGITIMATE")
    is_fraud          : bool          = Field(...,   description="True if fraud probability >= threshold")
    fraud_probability : float         = Field(...,   description="Model fraud probability score (0–1)")
    threshold         : float         = Field(...,   description="Decision threshold applied")
    top_risk_factors  : List[RiskFactor] = Field(..., description="Top SHAP contributors")
    summary           : str           = Field(...,   description="One-sentence human-readable explanation")
    model_version     : str           = Field(...,   description="Model identifier for audit trail")


class HealthResponse(BaseModel):
    """Response from GET /health — liveness check."""
    status        : str  = Field(..., description="'ok' if healthy")
    model_loaded  : bool = Field(..., description="True if model is in memory")
    model_name    : str  = Field(..., description="Name of loaded model")
    pr_auc        : float= Field(..., description="PR-AUC from evaluation phase")
    threshold     : float= Field(..., description="Current decision threshold")


class ErrorResponse(BaseModel):
    """Standard error response shape."""
    error   : str = Field(..., description="Error type")
    message : str = Field(..., description="Human-readable error description")
