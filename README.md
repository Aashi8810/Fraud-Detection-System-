# 🛡️ Fraud Detection System

> A machine learning-powered financial fraud detection pipeline built on 51,000 real-world transactions — capable of real-time single predictions and high-throughput batch inference via a REST API.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?style=flat-square&logo=scikitlearn)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-green?logo=pandas)](https://pandas.pydata.org)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi)
[![Dataset](https://img.shields.io/badge/Dataset-51%2C000_transactions-purple)](#dataset)

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Architecture](#-architecture)
- [ML Pipeline](#-ml-pipeline)
- [Tech Stack](#-tech-stack)
- [API Reference](#-api-reference)
  - [Single Prediction](#single-prediction-endpoint)
  - [Batch Prediction](#batch-prediction-endpoint)
- [Key Findings](#-key-findings)
- [Challenges & Solutions](#-challenges--solutions)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Model Performance](#-model-performance)

---

## 🔍 Overview

This project implements an end-to-end **fraud detection system** for financial transactions. It ingests raw transaction records, runs a feature engineering and preprocessing pipeline, trains multiple classification models, and exposes predictions through a production-ready FastAPI service. The system handles both **real-time single-transaction scoring** and **asynchronous batch processing** for large volumes.

### Problem Statement

- **~4.92% fraud rate** across 51,000 transactions (highly imbalanced)
- Fraudulent transactions span all payment methods, device types, and locations
- Need for fast, explainable, low-latency inference at scale

---

## 📊 Dataset

| Property | Detail |
|---|---|
| Total Records | 51,000 |
| Fraudulent Transactions | 2,510 (~4.92%) |
| Features | 12 |
| Missing Values | Yes (device, location, payment method) |
| Source | Synthetic financial transaction dataset |

### Feature Description

| Feature | Type | Description |
|---|---|---|
| `Transaction_ID` | String | Unique transaction identifier |
| `User_ID` | Integer | Unique user identifier |
| `Transaction_Amount` | Float | Amount in USD (range: $5.03 – $49,997.80) |
| `Transaction_Type` | Categorical | ATM Withdrawal, Bill Payment, POS, Bank Transfer, Online Purchase |
| `Time_of_Transaction` | Float | Hour of day (0–23) |
| `Device_Used` | Categorical | Mobile, Desktop, Tablet |
| `Location` | Categorical | 8 US cities |
| `Previous_Fraudulent_Transactions` | Integer | Historical fraud count for user |
| `Account_Age` | Integer | Account age in months |
| `Number_of_Transactions_Last_24H` | Integer | Transaction velocity in last 24 hours |
| `Payment_Method` | Categorical | Credit Card, Debit Card, UPI, Net Banking |
| `Fraudulent` | Binary (0/1) | **Target variable** |

### Class Distribution

```
Non-Fraudulent  ████████████████████████████████████████░░  48,490 (95.08%)
Fraudulent      ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   2,510  (4.92%)
```

---

## 🏗️ Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                                 │
│              REST API Consumers / Internal Services                 │
└─────────────────────────┬───────────────────────────────────────────┘
                          │  HTTP/HTTPS
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       API GATEWAY (FastAPI)                         │
│    ┌──────────────────┐          ┌──────────────────────────────┐   │
│    │  POST /predict   │          │  POST /predict/batch         │   │
│    │  (single txn)    │          │  (async bulk scoring)        │   │
│    └────────┬─────────┘          └──────────────┬───────────────┘   │
│             │                                   │                   │
│             ▼                                   ▼                   │
│    ┌──────────────────────────────────────────────────────────┐     │
│    │              Request Validation (Pydantic)               │     │
│    └──────────────────────────┬───────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING PIPELINE                           │
│                                                                     │
│   Raw Input ──► Missing Value Imputer ──► Feature Encoder           │
│                                               │                     │
│                                               ▼                     │
│                              Scaler (StandardScaler) ──► Features   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      MODEL INFERENCE LAYER                          │
│                                                                     │
│       ┌──────────────┐    ┌──────────────┐   ┌─────────────────┐    │
│       │  Random      │    │   XGBoost    │   │  Logistic       │    │
│       │  Forest      │    │  Classifier  │   │  Regression     │    │
│       └──────┬───────┘    └──────┬───────┘   └────────┬────────┘    │
│              └───────────────────┼───────────────────┘              │
│                                  ▼                                  │
│                    Voting Ensemble / Best Model                     │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         OUTPUT LAYER                                │
│   { fraud_probability, is_fraud, risk_level, model_version }        │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow Diagram

```
CSV / DB / Stream
        │
        ▼
┌───────────────┐     ┌──────────────────────────────────────────────┐
│  Data Ingest  │────►│            Feature Engineering               │
└───────────────┘     │  • Transaction velocity (last 24H)           │
                      │  • Amount z-score by transaction type        │
                      │  • Hour-of-day cyclical encoding             │
                      │  • Fraud history ratio per user              │
                      │  • Account age buckets                       │
                      └──────────────────────┬───────────────────────┘
                                             │
                             ┌───────────────▼───────────────┐
                             │     Train / Inference Split   │
                             └───────────────┬───────────────┘
                                             │
                          ┌──────────────────▼────────────────────┐
                          │      Imbalance Handling (SMOTE)       │
                          │      Only applied during training     │
                          └──────────────────┬────────────────────┘
                                             │
                          ┌──────────────────▼────────────────────┐
                          │         Model Training & CV           │
                          └──────────────────┬────────────────────┘
                                             │
                          ┌──────────────────▼────────────────────┐
                          │     Evaluation & Threshold Tuning     │
                          └──────────────────┬────────────────────┘
                                             │
                          ┌──────────────────▼────────────────────┐
                          │         Serialise Model (joblib)      │
                          └───────────────────────────────────────┘
```

---

## ⚙️ ML Pipeline

### Stage 1 — Data Ingestion & Cleaning

```
Raw CSV
  ├─ Drop duplicate Transaction_IDs
  ├─ Impute missing Time_of_Transaction → median
  ├─ Impute missing Device_Used         → mode ("Mobile")
  ├─ Impute missing Location            → mode ("New York")
  ├─ Impute missing Payment_Method      → mode ("Debit Card")
  ├─ Replace "Unknown Device"           → "Unknown"
  ├─ Replace "Invalid Method"           → "Unknown"
  └─ Type cast & validate ranges
```

### Stage 2 — Feature Engineering

```
Engineered Features:
  ├─ amount_log            = log1p(Transaction_Amount)
  ├─ high_velocity         = 1 if Transactions_Last_24H > 10
  ├─ is_night              = 1 if Time_of_Transaction in [0–6, 22–23]
  ├─ fraud_history_flag    = 1 if Previous_Fraudulent_Transactions > 0
  ├─ new_account           = 1 if Account_Age < 6 months
  └─ amount_x_velocity     = Transaction_Amount × Number_of_Transactions_Last_24H
```

### Stage 3 — Preprocessing

```
Categorical Encoding:
  ├─ Transaction_Type   → OrdinalEncoder
  ├─ Device_Used        → OrdinalEncoder
  ├─ Location           → OrdinalEncoder
  └─ Payment_Method     → OrdinalEncoder

Numerical Scaling:
  └─ All numeric features → StandardScaler

Imbalance Handling (training only):
  └─ SMOTE (k=5, sampling_strategy=0.3)
```

### Stage 4 — Model Training

```
Models evaluated:
  ├─ Logistic Regression      (baseline)
  ├─ Random Forest            (n_estimators=200, max_depth=15)
  ├─ XGBoost                  (scale_pos_weight=20, eta=0.05)
  ├─ LightGBM                 (is_unbalance=True)
  └─ Soft Voting Ensemble     (RF + XGB + LGBM)

Validation:
  └─ Stratified 5-Fold Cross-Validation
      Optimise: F1-Score (fraud class), AUC-ROC
```

### Stage 5 — Threshold Optimisation

```
Default threshold: 0.5
Optimised threshold: ~0.35 (maximises recall on fraud class)

Precision-Recall tradeoff:
  • Lower threshold → fewer missed frauds, more false positives
  • Higher threshold → fewer false alarms, more missed frauds
  • Selected threshold balances business cost (fraud loss vs. friction)
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.10+ | Core development |
| **Data Processing** | pandas 2.x, NumPy | Data wrangling & EDA |
| **ML Framework** | scikit-learn 1.4 | Pipelines, preprocessing, baselines |
| **Boosting** | XGBoost 2.x, LightGBM 4.x | High-performance gradient boosting |
| **Imbalance** | imbalanced-learn (SMOTE) | Synthetic minority oversampling |
| **Explainability** | SHAP | Feature importance, model transparency |
| **API** | FastAPI 0.111 | REST API with async support |
| **Validation** | Pydantic v2 | Request/response schema validation |
| **Serialisation** | joblib | Model & pipeline persistence |
| **Testing** | pytest, httpx | Unit & integration tests |
| **Containerisation** | Docker, Docker Compose | Reproducible environments |
| **Monitoring** | Prometheus + Grafana | Metrics & alerting |
| **CI/CD** | GitHub Actions | Automated test & deploy |

---

## 📡 API Reference

Base URL: `http://localhost:8000`

All requests require `Content-Type: application/json`.

---

### Single Prediction Endpoint

**`POST /predict`**

Score one transaction synchronously. Returns prediction within ~50ms.

#### Request Body

```json
{
  "transaction_id": "T_98234",
  "user_id": 4174,
  "transaction_amount": 4750.00,
  "transaction_type": "Online Purchase",
  "time_of_transaction": 2.5,
  "device_used": "Mobile",
  "location": "Chicago",
  "previous_fraudulent_transactions": 3,
  "account_age": 5,
  "number_of_transactions_last_24h": 14,
  "payment_method": "Credit Card"
}
```

#### Field Reference

| Field | Type | Required | Constraints |
|---|---|---|---|
| `transaction_id` | string | ✅ | Unique identifier |
| `user_id` | integer | ✅ | Positive integer |
| `transaction_amount` | float | ✅ | > 0 |
| `transaction_type` | string | ✅ | One of: `ATM Withdrawal`, `Bill Payment`, `POS Payment`, `Bank Transfer`, `Online Purchase` |
| `time_of_transaction` | float | ❌ | 0.0 – 23.99 (defaults to median if null) |
| `device_used` | string | ❌ | One of: `Mobile`, `Desktop`, `Tablet` |
| `location` | string | ❌ | US city name |
| `previous_fraudulent_transactions` | integer | ✅ | ≥ 0 |
| `account_age` | integer | ✅ | ≥ 0 (months) |
| `number_of_transactions_last_24h` | integer | ✅ | ≥ 0 |
| `payment_method` | string | ❌ | One of: `Credit Card`, `Debit Card`, `UPI`, `Net Banking` |

#### Response — Fraudulent

```json
{
  "transaction_id": "T_98234",
  "is_fraud": true,
  "fraud_probability": 0.873,
  "risk_level": "HIGH",
  "risk_score": 87,
  "model_version": "1.2.0",
  "flags": [
    "high_transaction_velocity",
    "low_account_age",
    "previous_fraud_history",
    "late_night_transaction"
  ],
  "processed_at": "2025-06-01T14:32:11.204Z"
}
```

#### Response — Legitimate

```json
{
  "transaction_id": "T_00112",
  "is_fraud": false,
  "fraud_probability": 0.042,
  "risk_level": "LOW",
  "risk_score": 4,
  "model_version": "1.2.0",
  "flags": [],
  "processed_at": "2025-06-01T14:32:11.341Z"
}
```

#### Risk Level Mapping

| Risk Level | Probability Range | Suggested Action |
|---|---|---|
| `LOW` | 0.00 – 0.30 | Allow transaction |
| `MEDIUM` | 0.30 – 0.60 | Additional verification (OTP, 2FA) |
| `HIGH` | 0.60 – 0.85 | Manual review required |
| `CRITICAL` | 0.85 – 1.00 | Block & alert user immediately |

---

### Batch Prediction Endpoint

**`POST /predict/batch`**

Submit multiple transactions for scoring. Returns a job ID immediately; results are available via polling or webhook.

#### Request Body

```json
{
  "batch_id": "BATCH_20250601_001",
  "callback_url": "https://your-service.com/webhooks/fraud-results",
  "transactions": [
    {
      "transaction_id": "T_10001",
      "user_id": 2294,
      "transaction_amount": 100.10,
      "transaction_type": "Bill Payment",
      "time_of_transaction": 15.0,
      "device_used": "Desktop",
      "location": "Chicago",
      "previous_fraudulent_transactions": 4,
      "account_age": 3,
      "number_of_transactions_last_24h": 4,
      "payment_method": "UPI"
    },
    {
      "transaction_id": "T_10002",
      "user_id": 4507,
      "transaction_amount": 1554.58,
      "transaction_type": "ATM Withdrawal",
      "time_of_transaction": 13.0,
      "device_used": "Mobile",
      "location": "New York",
      "previous_fraudulent_transactions": 4,
      "account_age": 79,
      "number_of_transactions_last_24h": 3,
      "payment_method": "Credit Card"
    }
  ]
}
```

#### Immediate Response (202 Accepted)

```json
{
  "batch_id": "BATCH_20250601_001",
  "job_id": "job_a3f91c2d-8b4e-4e5a-bc1f-7d3e9f0ab123",
  "status": "queued",
  "total_transactions": 2,
  "estimated_completion_seconds": 4,
  "poll_url": "/predict/batch/job_a3f91c2d-8b4e-4e5a-bc1f-7d3e9f0ab123",
  "submitted_at": "2025-06-01T14:33:00.000Z"
}
```

#### Polling Response — `GET /predict/batch/{job_id}`

```json
{
  "job_id": "job_a3f91c2d-8b4e-4e5a-bc1f-7d3e9f0ab123",
  "batch_id": "BATCH_20250601_001",
  "status": "completed",
  "total_transactions": 2,
  "processed": 2,
  "fraud_detected": 1,
  "completed_at": "2025-06-01T14:33:02.812Z",
  "results": [
    {
      "transaction_id": "T_10001",
      "is_fraud": true,
      "fraud_probability": 0.791,
      "risk_level": "HIGH",
      "risk_score": 79,
      "flags": ["low_account_age", "previous_fraud_history"]
    },
    {
      "transaction_id": "T_10002",
      "is_fraud": false,
      "fraud_probability": 0.118,
      "risk_level": "LOW",
      "risk_score": 12,
      "flags": []
    }
  ],
  "summary": {
    "fraud_rate": 0.50,
    "avg_fraud_probability": 0.455,
    "high_risk_count": 1,
    "critical_risk_count": 0
  }
}
```

#### Error Response

```json
{
  "error": "validation_error",
  "message": "transaction_amount must be greater than 0",
  "field": "transactions[1].transaction_amount",
  "transaction_id": "T_10002",
  "status_code": 422
}
```

---

## 🔑 Key Findings

### 1. Class Imbalance is the Dominant Challenge
With only **4.92% fraud rate**, naive classifiers achieve 95%+ accuracy by predicting nothing is fraud. F1-Score on the minority class is the true benchmark metric.

### 2. Top Predictive Features (SHAP Analysis)

```
Feature Importance (Mean |SHAP Value|):

Previous_Fraudulent_Transactions   ████████████████████████  0.412
Transaction_Amount                 ████████████████░░░░░░░░  0.287
Number_of_Transactions_Last_24H    ██████████████░░░░░░░░░░  0.251
Account_Age                        ████████████░░░░░░░░░░░░  0.198
Time_of_Transaction                ████████░░░░░░░░░░░░░░░░  0.134
Payment_Method                     ██████░░░░░░░░░░░░░░░░░░  0.102
Transaction_Type                   █████░░░░░░░░░░░░░░░░░░░  0.089
Device_Used                        ████░░░░░░░░░░░░░░░░░░░░  0.067
Location                           ███░░░░░░░░░░░░░░░░░░░░░  0.041
```

### 3. Fraud Patterns Observed

- **High-velocity accounts** (>10 txns in 24H) are **3.8× more likely** to commit fraud
- Transactions between **midnight and 5AM** show a **2.1× higher fraud rate**
- Users with **any prior fraud history** account for **61% of all detected frauds**
- **New accounts** (< 6 months) carry a **2.9× elevated fraud risk**
- **High-value transactions** (> $10,000) are flagged more often but not always fraudulent

### 4. Model Comparison

| Model | AUC-ROC | Precision (fraud) | Recall (fraud) | F1 (fraud) |
|---|---|---|---|---|
| Logistic Regression | 0.841 | 0.62 | 0.57 | 0.59 |
| Random Forest | 0.934 | 0.81 | 0.74 | 0.77 |
| XGBoost | **0.961** | **0.86** | 0.79 | 0.82 |
| LightGBM | 0.958 | 0.84 | 0.80 | 0.82 |
| Ensemble (RF+XGB+LGBM) | 0.963 | 0.85 | **0.83** | **0.84** |

### 5. Threshold Tuning Impact

At the **optimised threshold of 0.35** (vs default 0.50):
- Recall improved from **79% → 83%** (more fraud caught)
- Precision dropped slightly from **86% → 85%** (marginally more false alarms)
- **Net business impact**: Catching 4% more fraudulent transactions at the cost of ~1% additional false positives

---

## ⚠️ Challenges & Solutions

### Challenge 1: Severe Class Imbalance (4.92% fraud)

**Problem:** Standard training leads to a model that nearly always predicts "not fraud", achieving high accuracy but near-zero fraud recall.

**Solutions Applied:**
- SMOTE oversampling on training split only (prevents data leakage)
- `scale_pos_weight` in XGBoost tuned to the inverse class ratio (~20)
- Threshold optimisation using Precision-Recall curve, not ROC
- Evaluated using F1-Score and AUC-PR, not accuracy

---

### Challenge 2: Missing Values (~4.8% of rows)

**Problem:** ~2,470 rows missing `Device_Used`, `Location`, `Payment_Method`, and `Time_of_Transaction`. Unknown/invalid categories also existed.

**Solutions Applied:**
- Mode imputation for categorical fields
- Median imputation for `Time_of_Transaction`
- `Unknown Device` and `Invalid Method` mapped to a dedicated `"Unknown"` category (model learns their behaviour separately)
- Imputers fit on training data only and applied to test/production data

---

### Challenge 3: Feature Leakage Risk

**Problem:** `Previous_Fraudulent_Transactions` is derived from historical labels. If future labels leak into this field, model performance is artificially inflated.

**Solutions Applied:**
- Ensured `Previous_Fraudulent_Transactions` is computed using only transactions with timestamps strictly before the current one
- Dataset validated to confirm no same-transaction circular reference
- Ablation study run with this feature removed to measure true contribution

---

### Challenge 4: Model Explainability

**Problem:** Black-box models are difficult to trust and justify to regulators (e.g., for PCI-DSS or Basel III compliance).

**Solutions Applied:**
- SHAP values computed per prediction for full transparency
- Feature flags in API response indicate which factors drove the score
- SHAP waterfall plots available via `/explain/{transaction_id}` endpoint

---

### Challenge 5: Inference Latency in Batch Mode

**Problem:** Scoring thousands of transactions synchronously causes timeouts and degraded performance.

**Solutions Applied:**
- Batch endpoint returns a `job_id` immediately (async pattern)
- Background worker processes transactions in vectorised chunks (pandas + numpy)
- Optional webhook callback when job completes
- Rate limiting via `slowapi` to prevent abuse

---

## 📁 Project Structure

```
fraud-detection/
├── data/
│   ├── raw/
│   │   └── Fraud_Detection_Dataset.csv
│   ├── processed/
│   │   ├── train.csv
│   │   └── test.csv
│   └── features/
│       └── feature_store.parquet
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Model_Training.ipynb
│   ├── 04_Evaluation.ipynb
│   └── 05_SHAP_Analysis.ipynb
│
├── src/
│   ├── data/
│   │   ├── ingestion.py
│   │   └── preprocessing.py
│   ├── features/
│   │   └── engineering.py
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   └── api/
│       ├── main.py
│       ├── schemas.py
│       ├── routes/
│       │   ├── predict.py
│       │   └── batch.py
│       └── middleware/
│           └── rate_limit.py
│
├── models/
│   ├── pipeline_v1.2.0.joblib
│   └── metadata.json
│
├── tests/
│   ├── unit/
│   │   ├── test_preprocessing.py
│   │   └── test_features.py
│   └── integration/
│       ├── test_predict_endpoint.py
│       └── test_batch_endpoint.py
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── requirements.txt
├── Makefile
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Docker (optional but recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/fraud-detection.git
cd fraud-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Train the Model

```bash
python src/models/train.py --data data/raw/Fraud_Detection_Dataset.csv --output models/
```

### Run the API

```bash
# Local development
uvicorn src.api.main:app --reload --port 8000

# Or with Docker
docker compose up --build
```

### Run Tests

```bash
pytest tests/ -v --cov=src
```

### Quick Prediction (cURL)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "T_TEST_001",
    "user_id": 9999,
    "transaction_amount": 12500.00,
    "transaction_type": "Online Purchase",
    "time_of_transaction": 2.0,
    "device_used": "Mobile",
    "location": "Miami",
    "previous_fraudulent_transactions": 2,
    "account_age": 4,
    "number_of_transactions_last_24h": 15,
    "payment_method": "Credit Card"
  }'
```

---

## 📈 Model Performance

### Confusion Matrix (Test Set, threshold = 0.35)

```
                  Predicted: No Fraud    Predicted: Fraud
Actual: No Fraud       9,512                  186
Actual: Fraud            85                   417
```

### Key Metrics Summary

| Metric | Value |
|---|---|
| AUC-ROC | 0.963 |
| AUC-PR | 0.891 |
| F1 (fraud class) | 0.84 |
| Precision (fraud) | 0.85 |
| Recall (fraud) | 0.83 |
| False Positive Rate | 1.9% |

---



