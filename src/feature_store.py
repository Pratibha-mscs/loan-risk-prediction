import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import FEATURE_STORE_DIR
from src.utils import get_logger

logger = get_logger(__name__)

FEATURE_STORE_DIR.mkdir(parents=True, exist_ok=True)


def save_features(df: pd.DataFrame, name: str = "engineered_features") -> None:
    path = FEATURE_STORE_DIR / f"{name}.parquet"
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path, compression="snappy")
    logger.info(f"Saved {len(df):,} rows, {len(df.columns)} cols to feature store: {path.name}")


def load_features(name: str = "engineered_features") -> pd.DataFrame:
    path = FEATURE_STORE_DIR / f"{name}.parquet"
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} cols from feature store: {path.name}")
    return df


def list_features() -> list[str]:
    return [f.stem for f in FEATURE_STORE_DIR.glob("*.parquet")]


def get_feature_metadata(name: str = "engineered_features") -> dict:
    path = FEATURE_STORE_DIR / f"{name}.parquet"
    pf = pq.read_metadata(path)
    return {
        "num_rows": pf.num_rows,
        "num_columns": pf.num_columns,
        "file_size_mb": path.stat().st_size / (1024 * 1024),
        "schema": [field.name for field in pq.read_schema(path)],
    }
