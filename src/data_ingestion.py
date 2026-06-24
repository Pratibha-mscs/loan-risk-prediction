import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import (
    RAW_DIR, PROCESSED_DIR, LOAN_STATUSES_KEEP,
    TARGET_COL, TARGET_BINARY, FEATURES_TO_DROP,
)
from src.utils import get_logger

logger = get_logger(__name__)

RAW_CSV = RAW_DIR / "accepted_2007_to_2018q4.csv" / "accepted_2007_to_2018Q4.csv"


def load_raw_data(nrows: int | None = None) -> pd.DataFrame:
    logger.info(f"Loading raw CSV (nrows={nrows})...")
    df = pd.read_csv(RAW_CSV, low_memory=False, nrows=nrows)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def filter_loan_status(df: pd.DataFrame) -> pd.DataFrame:
    initial = len(df)
    df = df[df[TARGET_COL].isin(LOAN_STATUSES_KEEP)].copy()
    logger.info(f"Filtered loan_status: {initial:,} → {len(df):,} rows")
    return df


def create_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    df[TARGET_BINARY] = (df[TARGET_COL] == "Charged Off").astype(int)
    default_rate = df[TARGET_BINARY].mean()
    logger.info(f"Default rate: {default_rate:.2%}")
    return df


def drop_leaky_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in FEATURES_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    logger.info(f"Dropped {len(cols_to_drop)} columns")
    return df


def clean_percentage_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["int_rate", "revol_util"]:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].str.rstrip("%").astype(float)
    return df


def clean_employment_length(df: pd.DataFrame) -> pd.DataFrame:
    if "emp_length" not in df.columns:
        return df
    mapping = {"< 1 year": 0, "1 year": 1, "10+ years": 10}
    for i in range(2, 10):
        mapping[f"{i} years"] = i
    df["emp_length_years"] = df["emp_length"].map(mapping)
    return df


def clean_term(df: pd.DataFrame) -> pd.DataFrame:
    if "term" in df.columns and df["term"].dtype == object:
        df["term_months"] = df["term"].str.strip().str.split().str[0].astype(float)
    return df


def save_to_parquet(df: pd.DataFrame, name: str = "loans_cleaned") -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / f"{name}.parquet"
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path, compression="snappy")
    logger.info(f"Saved {len(df):,} rows to {path}")


def load_from_parquet(name: str = "loans_cleaned") -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}.parquet"
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df):,} rows from {path}")
    return df


def run_ingestion(sample_size: int | None = None) -> pd.DataFrame:
    df = load_raw_data(nrows=sample_size)
    df = filter_loan_status(df)
    df = create_binary_target(df)
    df = drop_leaky_columns(df)
    df = clean_percentage_columns(df)
    df = clean_employment_length(df)
    df = clean_term(df)
    save_to_parquet(df)
    return df


if __name__ == "__main__":
    run_ingestion()
