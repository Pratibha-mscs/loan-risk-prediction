import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import MODELS_DIR, LGD_DEFAULT
from src.risk_scoring import generate_risk_profile
from src.utils import get_logger

logger = get_logger("api")

app = FastAPI(
    title="Credit Risk Prediction API",
    description="Production-grade credit risk modeling for consumer lending",
    version="2.0.0",
)


class LoanApplication(BaseModel):
    annual_income: float = Field(..., gt=0, description="Annual income in USD")
    loan_amount: float = Field(..., gt=0, description="Requested loan amount")
    dti: float = Field(..., ge=0, le=100, description="Debt-to-income ratio (e.g. 18.0 for 18%)")
    int_rate: float = Field(..., ge=0, le=40, description="Interest rate (e.g. 12.0 for 12%)")
    installment: float = Field(..., ge=0, description="Monthly installment in USD")
    emp_length_years: float = Field(0, ge=0, le=15, description="Years of employment")
    revol_util: float = Field(0, ge=0, le=150, description="Revolving utilization (e.g. 45.0 for 45%)")
    revol_bal: float = Field(0, ge=0, description="Revolving balance in USD")
    open_acc: float = Field(5, ge=0, description="Open credit accounts")
    total_acc: float = Field(10, ge=0, description="Total credit accounts")
    delinq_2yrs: float = Field(0, ge=0, description="Delinquencies in past 2 years")
    inq_last_6mths: float = Field(0, ge=0, description="Credit inquiries in last 6 months")
    pub_rec: float = Field(0, ge=0, description="Public records (bankruptcies, liens)")
    grade_numeric: float = Field(2, ge=0, le=6, description="Loan grade (0=A, 1=B, ..., 6=G)")
    sub_grade_numeric: float = Field(10, ge=0, le=34, description="Sub-grade numeric (0-34)")


class PredictionResponse(BaseModel):
    probability_of_default: float
    credit_score: int
    risk_tier: str
    expected_loss: float
    expected_loss_formula: str
    decision: str
    top_risk_factors: list[str]
    protective_factors: list[str]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class ModelInfoResponse(BaseModel):
    model_type: str
    n_features: int
    feature_names: list[str]


_cached_model = None


def load_model():
    global _cached_model
    if _cached_model is not None:
        return _cached_model
    model_path = MODELS_DIR / "best_model.joblib"
    if not model_path.exists():
        return None
    _cached_model = joblib.load(model_path)
    return _cached_model


def get_expected_features(model) -> list | None:
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "feature_names_"):
        return list(model.feature_names_)
    # CalibratedClassifierCV wraps the estimator
    if hasattr(model, "estimator"):
        return get_expected_features(model.estimator)
    if hasattr(model, "calibrated_classifiers_"):
        inner = model.calibrated_classifiers_[0].estimator
        return get_expected_features(inner)
    return None


def build_feature_vector(app_data: LoanApplication) -> pd.DataFrame:
    from src.feature_engineering import run_feature_engineering

    grades = "ABCDEFG"
    grade_idx = int(min(max(app_data.grade_numeric, 0), 6))
    grade_letter = grades[grade_idx]
    sub_idx = int(app_data.sub_grade_numeric % 5) + 1
    sub_grade_letter = f"{grade_letter}{sub_idx}"

    raw_row = {
        "annual_inc": app_data.annual_income,
        "loan_amnt": app_data.loan_amount,
        "funded_amnt": app_data.loan_amount,
        "dti": app_data.dti,
        "int_rate": app_data.int_rate,
        "installment": app_data.installment,
        "emp_length_years": app_data.emp_length_years,
        "revol_util": app_data.revol_util,
        "revol_bal": app_data.revol_bal,
        "open_acc": app_data.open_acc,
        "total_acc": app_data.total_acc,
        "delinq_2yrs": app_data.delinq_2yrs,
        "inq_last_6mths": app_data.inq_last_6mths,
        "pub_rec": app_data.pub_rec,
        "term_months": 36,
        "grade": grade_letter,
        "sub_grade": sub_grade_letter,
    }

    df = pd.DataFrame([raw_row])
    df = run_feature_engineering(df)
    return df


def generate_risk_factors(app_data: LoanApplication, pd_value: float) -> tuple[list[str], list[str]]:
    """Rule-based explanation engine. Produces human-readable factors that
    drove risk up or down, grounded in the input values themselves."""
    risk_factors = []
    protective = []

    # Grade
    grade_labels = {0: "A (prime)", 1: "B (near-prime)", 2: "C (mid-tier)",
                    3: "D (subprime)", 4: "E (deep subprime)", 5: "F (high risk)", 6: "G (highest risk)"}
    g = int(app_data.grade_numeric)
    if g >= 4:
        risk_factors.append(f"Loan grade {grade_labels.get(g, g)} — significantly increases risk")
    elif g >= 2:
        risk_factors.append(f"Loan grade {grade_labels.get(g, g)} — moderate risk tier")
    else:
        protective.append(f"Loan grade {grade_labels.get(g, g)} — lower risk tier")

    # Income to loan ratio
    lti = app_data.loan_amount / max(app_data.annual_income, 1)
    if lti < 0.15:
        protective.append(f"Strong income-to-loan ratio ({lti:.0%}) — loan is small relative to income")
    elif lti > 0.5:
        risk_factors.append(f"High loan-to-income ratio ({lti:.0%}) — loan is large relative to income")

    # DTI
    if app_data.dti <= 12:
        protective.append(f"Low DTI ({app_data.dti:.0f}%) — healthy debt load")
    elif app_data.dti >= 28:
        risk_factors.append(f"High DTI ({app_data.dti:.0f}%) — heavy existing debt burden")
    elif app_data.dti >= 20:
        risk_factors.append(f"Moderate DTI ({app_data.dti:.0f}%) — manageable but elevated")

    # Interest rate
    if app_data.int_rate <= 8:
        protective.append(f"Low interest rate ({app_data.int_rate:.1f}%) — favorable pricing")
    elif app_data.int_rate >= 18:
        risk_factors.append(f"High interest rate ({app_data.int_rate:.1f}%) — priced for risk")
    elif app_data.int_rate >= 13:
        risk_factors.append(f"Above-average interest rate ({app_data.int_rate:.1f}%)")

    # Revolving utilization
    if app_data.revol_util <= 25:
        protective.append(f"Low revolving utilization ({app_data.revol_util:.0f}%) — healthy credit usage")
    elif app_data.revol_util >= 70:
        risk_factors.append(f"High revolving utilization ({app_data.revol_util:.0f}%) — near credit limits")
    elif app_data.revol_util >= 45:
        risk_factors.append(f"Moderate revolving utilization ({app_data.revol_util:.0f}%)")

    # Delinquencies
    if app_data.delinq_2yrs == 0:
        protective.append("No delinquencies in past 2 years — clean payment history")
    elif app_data.delinq_2yrs >= 2:
        risk_factors.append(f"{int(app_data.delinq_2yrs)} delinquencies in past 2 years — payment issues")
    else:
        risk_factors.append(f"{int(app_data.delinq_2yrs)} delinquency in past 2 years")

    # Employment
    if app_data.emp_length_years >= 7:
        protective.append(f"Stable employment ({int(app_data.emp_length_years)} years)")
    elif app_data.emp_length_years <= 1:
        risk_factors.append(f"Short employment history ({int(app_data.emp_length_years)} year)")

    # Public records
    if app_data.pub_rec > 0:
        risk_factors.append(f"{int(app_data.pub_rec)} public record(s) — bankruptcies or liens")

    # Inquiries
    if app_data.inq_last_6mths >= 4:
        risk_factors.append(f"{int(app_data.inq_last_6mths)} credit inquiries in 6 months — active credit seeking")

    return risk_factors[:5], protective[:5]


@app.get("/health", response_model=HealthResponse)
def health_check():
    model = load_model()
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        version="2.0.0",
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication):
    model = load_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    features_df = build_feature_vector(application)

    expected_features = get_expected_features(model)
    if expected_features is not None:
        for col in expected_features:
            if col not in features_df.columns:
                features_df[col] = 0
        features_df = features_df[expected_features]

    pd_value = float(model.predict_proba(features_df)[:, 1][0])

    risk_profile = generate_risk_profile(
        pd_value, application.loan_amount,
        application.annual_income, application.dti, application.int_rate,
    )

    risk_factors, protective = generate_risk_factors(application, pd_value)

    tier = risk_profile["risk_tier"]
    if tier == "High Risk":
        decision = "HIGH RISK (Subprime) — Recommend Denial or High-Rate Pricing"
    elif tier == "Medium Risk":
        decision = "MEDIUM RISK (Near-Prime) — Standard Approval with Review"
    else:
        decision = "LOW RISK (Prime) — Recommend Approval at Best Rates"

    el = risk_profile["expected_loss"]
    formula = f"PD({pd_value:.2%}) x EAD(${application.loan_amount:,.0f}) x LGD({LGD_DEFAULT:.0%}) = ${el:,.0f}"

    return PredictionResponse(
        probability_of_default=risk_profile["probability_of_default"],
        credit_score=risk_profile["credit_score"],
        risk_tier=risk_profile["risk_tier"],
        expected_loss=el,
        expected_loss_formula=formula,
        decision=decision,
        top_risk_factors=risk_factors,
        protective_factors=protective,
        model_version="2.0.0-calibrated",
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    model = load_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    feature_names = get_expected_features(model) or []
    model_type = type(model).__name__
    if hasattr(model, "calibrated_classifiers_"):
        inner = model.calibrated_classifiers_[0].estimator
        model_type = f"Calibrated({type(inner).__name__})"
    return ModelInfoResponse(
        model_type=model_type,
        n_features=len(feature_names),
        feature_names=feature_names,
    )


@app.get("/metrics")
def get_metrics():
    model = load_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_type": type(model).__name__,
        "status": "Model loaded and ready for predictions",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
