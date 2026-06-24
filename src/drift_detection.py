import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.config import REPORTS_DIR, NUMERICAL_FEATURES
from src.utils import get_logger

logger = get_logger(__name__)

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_data_drift_report(reference: pd.DataFrame, current: pd.DataFrame,
                                output_name: str = "data_drift_report") -> str:
    num_cols = [c for c in NUMERICAL_FEATURES if c in reference.columns and c in current.columns]
    ref_subset = reference[num_cols].copy()
    cur_subset = current[num_cols].copy()

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(current_data=cur_subset, reference_data=ref_subset)

    output_path = REPORTS_DIR / f"{output_name}.html"
    snapshot.save_html(str(output_path))
    logger.info(f"Data drift report saved to {output_path}")
    return str(output_path)


def run_drift_monitoring(train_df: pd.DataFrame, test_df: pd.DataFrame):
    logger.info("Running drift monitoring...")
    generate_data_drift_report(train_df, test_df)
    logger.info("Drift monitoring complete")
