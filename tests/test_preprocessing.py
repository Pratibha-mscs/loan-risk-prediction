import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    drop_high_null_columns, impute_numerical, impute_categorical,
    encode_categoricals, detect_outliers_iqr,
)


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "loan_amnt": np.random.uniform(1000, 40000, n),
        "annual_inc": np.random.uniform(20000, 200000, n),
        "dti": np.random.uniform(0, 40, n),
        "grade": np.random.choice(["A", "B", "C", "D"], n),
        "purpose": np.random.choice(["credit_card", "debt_consolidation", "home"], n),
        "is_default": np.random.choice([0, 1], n, p=[0.8, 0.2]),
    })
    df.loc[:9, "annual_inc"] = np.nan
    df.loc[:4, "grade"] = np.nan
    return df


def test_drop_high_null_columns(sample_df):
    sample_df["mostly_null"] = np.nan
    result = drop_high_null_columns(sample_df, threshold=0.50)
    assert "mostly_null" not in result.columns
    assert "loan_amnt" in result.columns


def test_impute_numerical(sample_df):
    assert sample_df["annual_inc"].isnull().sum() > 0
    result = impute_numerical(sample_df.copy(), strategy="median")
    assert result["annual_inc"].isnull().sum() == 0


def test_impute_categorical(sample_df):
    assert sample_df["grade"].isnull().sum() > 0
    result = impute_categorical(sample_df.copy())
    assert result["grade"].isnull().sum() == 0


def test_encode_categoricals(sample_df):
    sample_df = impute_categorical(sample_df)
    result, encoders = encode_categoricals(sample_df.copy(), method="label")
    assert result["grade"].dtype in [np.int64, np.int32, int]
    assert "grade" in encoders


def test_outlier_detection(sample_df):
    mask = detect_outliers_iqr(sample_df, columns=["loan_amnt", "annual_inc"])
    assert isinstance(mask, pd.Series)
    assert mask.dtype == bool
