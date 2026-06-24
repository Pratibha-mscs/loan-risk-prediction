"""Sanity-check tests: verify model predictions are business-sensible.

These are NOT unit tests for code correctness — they validate that the
trained model + calibration + FICO-aligned scoring produces defensible
outputs for known borrower archetypes.

FICO Tier Alignment:
  Low Risk  (Prime):      740–850, PD < ~16%
  Medium Risk (Near-Prime): 670–739, PD ~16–26%
  High Risk (Subprime):   300–669, PD > ~26%
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from fastapi.testclient import TestClient
from app.api import app
from src.config import MODELS_DIR

client = TestClient(app)

# These tests require a trained model — skip in CI where no model exists
MODEL_EXISTS = (MODELS_DIR / "best_model.joblib").exists()
requires_model = pytest.mark.skipif(not MODEL_EXISTS, reason="No trained model available (run pipeline first)")


def predict(payload: dict) -> dict:
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200, f"API returned {resp.status_code}: {resp.text}"
    return resp.json()


CASE_A_STRONG = {
    "annual_income": 120000, "loan_amount": 10000, "dti": 10.0,
    "int_rate": 7.0, "installment": 310.0, "emp_length_years": 10,
    "revol_util": 20.0, "revol_bal": 4000, "open_acc": 12,
    "total_acc": 30, "delinq_2yrs": 0, "inq_last_6mths": 0,
    "pub_rec": 0, "grade_numeric": 0, "sub_grade_numeric": 2,
}

CASE_B_MODERATE = {
    "annual_income": 95000, "loan_amount": 15000, "dti": 18.0,
    "int_rate": 12.0, "installment": 450.0, "emp_length_years": 7,
    "revol_util": 45.0, "revol_bal": 12000, "open_acc": 8,
    "total_acc": 20, "delinq_2yrs": 0, "inq_last_6mths": 1,
    "pub_rec": 0, "grade_numeric": 2, "sub_grade_numeric": 12,
}

CASE_C_RISKY = {
    "annual_income": 42000, "loan_amount": 30000, "dti": 35.0,
    "int_rate": 24.0, "installment": 900.0, "emp_length_years": 1,
    "revol_util": 88.0, "revol_bal": 25000, "open_acc": 4,
    "total_acc": 8, "delinq_2yrs": 3, "inq_last_6mths": 5,
    "pub_rec": 1, "grade_numeric": 5, "sub_grade_numeric": 27,
}



@requires_model
def test_strong_borrower_is_low_risk():
    result = predict(CASE_A_STRONG)
    assert result["risk_tier"] == "Low Risk", \
        f"Strong borrower got {result['risk_tier']} (score={result['credit_score']})"
    assert result["credit_score"] >= 740



@requires_model
def test_moderate_borrower_is_medium_or_low():
    result = predict(CASE_B_MODERATE)
    assert result["risk_tier"] in ("Low Risk", "Medium Risk"), \
        f"Moderate borrower got {result['risk_tier']} (score={result['credit_score']})"
    assert result["credit_score"] >= 670



@requires_model
def test_risky_borrower_is_high_risk():
    result = predict(CASE_C_RISKY)
    assert result["risk_tier"] == "High Risk", \
        f"Risky borrower got {result['risk_tier']} (score={result['credit_score']})"
    assert result["credit_score"] < 670



@requires_model
def test_credit_score_inversely_correlated_with_pd():
    strong = predict(CASE_A_STRONG)
    risky = predict(CASE_C_RISKY)
    assert strong["credit_score"] > risky["credit_score"], \
        f"Strong ({strong['credit_score']}) should exceed risky ({risky['credit_score']})"



@requires_model
def test_expected_loss_formula_present():
    result = predict(CASE_B_MODERATE)
    assert "expected_loss_formula" in result
    assert "PD" in result["expected_loss_formula"]
    assert "EAD" in result["expected_loss_formula"]
    assert "LGD" in result["expected_loss_formula"]



@requires_model
def test_risk_factors_present():
    result = predict(CASE_B_MODERATE)
    assert len(result["top_risk_factors"]) > 0
    assert len(result["protective_factors"]) > 0



@requires_model
def test_model_version_present():
    result = predict(CASE_B_MODERATE)
    assert "model_version" in result
    assert result["model_version"] != ""


if __name__ == "__main__":
    for name, payload in [("STRONG", CASE_A_STRONG), ("MODERATE", CASE_B_MODERATE), ("RISKY", CASE_C_RISKY)]:
        r = predict(payload)
        print(f"\n=== {name} BORROWER ===")
        print(f"  PD:            {r['probability_of_default']}%")
        print(f"  Credit Score:  {r['credit_score']}")
        print(f"  Risk Tier:     {r['risk_tier']}")
        print(f"  Expected Loss: ${r['expected_loss']:,.0f}")
        print(f"  Decision:      {r['decision']}")
        print(f"  Formula:       {r['expected_loss_formula']}")
        print(f"  Risk Factors:  {r['top_risk_factors']}")
        print(f"  Protective:    {r['protective_factors']}")
