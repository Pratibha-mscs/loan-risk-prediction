import pandas as pd

from src.config import TARGET_BINARY, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from src.utils import get_logger

logger = get_logger(__name__)


def validate_schema(df: pd.DataFrame) -> dict:
    results = {"passed": True, "checks": []}

    if TARGET_BINARY not in df.columns:
        results["passed"] = False
        results["checks"].append(f"FAIL: Missing target column '{TARGET_BINARY}'")
    else:
        results["checks"].append(f"PASS: Target column '{TARGET_BINARY}' present")

    present_numerical = [c for c in NUMERICAL_FEATURES if c in df.columns]
    missing_numerical = [c for c in NUMERICAL_FEATURES if c not in df.columns]
    results["checks"].append(
        f"INFO: {len(present_numerical)}/{len(NUMERICAL_FEATURES)} numerical features present"
    )
    if missing_numerical:
        results["checks"].append(f"WARN: Missing numerical: {missing_numerical}")

    present_categorical = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    results["checks"].append(
        f"INFO: {len(present_categorical)}/{len(CATEGORICAL_FEATURES)} categorical features present"
    )

    return results


def validate_data_quality(df: pd.DataFrame) -> dict:
    results = {"passed": True, "checks": []}

    if len(df) == 0:
        results["passed"] = False
        results["checks"].append("FAIL: DataFrame is empty")
        return results

    results["checks"].append(f"PASS: {len(df):,} rows loaded")

    dup_count = df.duplicated().sum()
    if dup_count > 0:
        results["checks"].append(f"WARN: {dup_count:,} duplicate rows")

    null_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    high_null = null_pct[null_pct > 50]
    if len(high_null) > 0:
        results["checks"].append(
            f"WARN: {len(high_null)} columns with >50% missing values"
        )

    if TARGET_BINARY in df.columns:
        class_dist = df[TARGET_BINARY].value_counts(normalize=True)
        minority_pct = class_dist.min()
        results["checks"].append(f"INFO: Minority class = {minority_pct:.2%}")
        if minority_pct < 0.05:
            results["checks"].append("WARN: Severe class imbalance (<5% minority)")

    return results


def run_validation(df: pd.DataFrame) -> dict:
    logger.info("Running data validation...")
    schema_results = validate_schema(df)
    quality_results = validate_data_quality(df)

    all_checks = schema_results["checks"] + quality_results["checks"]
    passed = schema_results["passed"] and quality_results["passed"]

    for check in all_checks:
        logger.info(check)

    logger.info(f"Validation {'PASSED' if passed else 'FAILED'}")
    return {"passed": passed, "checks": all_checks}
