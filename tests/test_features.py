import pandas as pd
import pytest

from src.feature_engineering import (
    add_financial_ratios, add_credit_features, add_risk_features,
    run_feature_engineering,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "loan_amnt": [10000, 25000, 5000],
        "funded_amnt": [10000, 25000, 5000],
        "annual_inc": [60000, 100000, 35000],
        "dti": [15.0, 22.0, 30.0],
        "installment": [300, 800, 150],
        "int_rate": [10.0, 15.0, 8.0],
        "revol_util": [45.0, 80.0, 20.0],
        "revol_bal": [5000, 20000, 2000],
        "open_acc": [8, 12, 4],
        "total_acc": [20, 30, 10],
        "delinq_2yrs": [0, 1, 2],
        "pub_rec": [0, 0, 1],
        "tot_cur_bal": [50000, 100000, 15000],
        "total_rev_hi_lim": [15000, 40000, 8000],
        "emp_length_years": [5, 10, 1],
        "term_months": [36, 60, 36],
        "total_pymnt": [11000, 20000, 4000],
        "total_rec_prncp": [9000, 15000, 3500],
        "last_pymnt_amnt": [300, 800, 150],
        "is_default": [0, 1, 0],
    })


def test_financial_ratios(sample_df):
    result = add_financial_ratios(sample_df)
    assert "loan_income_ratio" in result.columns
    assert "debt_income_ratio" in result.columns
    assert "installment_income_ratio" in result.columns
    assert result["loan_income_ratio"].iloc[0] > 0


def test_credit_features(sample_df):
    result = add_credit_features(sample_df)
    assert "total_credit_lines_ratio" in result.columns
    assert "delinquency_rate" in result.columns
    assert all(result["total_credit_lines_ratio"] >= 0)


def test_risk_features(sample_df):
    sample_df = add_financial_ratios(sample_df)
    sample_df = add_credit_features(sample_df)
    result = add_risk_features(sample_df)
    assert "financial_health_score" in result.columns
    assert "credit_strength_score" in result.columns
    assert all(result["financial_health_score"].between(0, 1))
    assert all(result["credit_strength_score"].between(0, 1))


def test_full_feature_engineering(sample_df):
    result = run_feature_engineering(sample_df)
    assert "loan_income_ratio" in result.columns
    assert "financial_health_score" in result.columns
    assert "income_x_credit_strength" in result.columns
    assert "income_bracket" in result.columns
