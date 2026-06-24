import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import (
    LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler,
)
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

from src.config import (
    TARGET_BINARY, RANDOM_STATE, TEST_SIZE,
)
from src.utils import get_logger

logger = get_logger(__name__)


def drop_high_null_columns(df: pd.DataFrame, threshold: float = 0.50) -> pd.DataFrame:
    null_pct = df.isnull().mean()
    cols_to_drop = null_pct[null_pct > threshold].index.tolist()
    cols_to_drop = [c for c in cols_to_drop if c != TARGET_BINARY]
    df = df.drop(columns=cols_to_drop)
    logger.info(f"Dropped {len(cols_to_drop)} columns with >{threshold:.0%} nulls")
    return df


def impute_numerical(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != TARGET_BINARY and df[c].isnull().any()]

    if not num_cols:
        return df

    if strategy == "knn":
        imputer = KNNImputer(n_neighbors=5)
        df[num_cols] = imputer.fit_transform(df[num_cols])
    else:
        imputer = SimpleImputer(strategy=strategy)
        df[num_cols] = imputer.fit_transform(df[num_cols])

    logger.info(f"Imputed {len(num_cols)} numerical columns using {strategy}")
    return df


def impute_categorical(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols = [c for c in cat_cols if df[c].isnull().any()]
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "Unknown")
    logger.info(f"Imputed {len(cat_cols)} categorical columns with mode")
    return df


def encode_categoricals(df: pd.DataFrame, method: str = "label") -> tuple[pd.DataFrame, dict]:
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in [TARGET_BINARY, "loan_status"]]
    encoders = {}

    if method == "label":
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
    elif method == "target":
        for col in cat_cols:
            means = df.groupby(col)[TARGET_BINARY].mean()
            df[col] = df[col].map(means)
            encoders[col] = means.to_dict()
    elif method == "onehot":
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    logger.info(f"Encoded {len(cat_cols)} categorical columns using {method}")
    return df, encoders


def scale_features(df: pd.DataFrame, method: str = "robust",
                   exclude_cols: list[str] | None = None) -> tuple[pd.DataFrame, object]:
    exclude = exclude_cols or [TARGET_BINARY]
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in exclude]

    scalers = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}
    scaler = scalers[method]()

    df[num_cols] = scaler.fit_transform(df[num_cols])
    logger.info(f"Scaled {len(num_cols)} features using {method}")
    return df, scaler


def detect_outliers_iqr(df: pd.DataFrame, columns: list[str] | None = None,
                        factor: float = 1.5) -> pd.Series:
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    cols = [c for c in cols if c != TARGET_BINARY]
    outlier_mask = pd.Series(False, index=df.index)

    for col in cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        outlier_mask |= (df[col] < lower) | (df[col] > upper)

    logger.info(f"IQR outliers: {outlier_mask.sum():,} rows ({outlier_mask.mean():.1%})")
    return outlier_mask


def detect_outliers_isolation_forest(df: pd.DataFrame,
                                     columns: list[str] | None = None,
                                     contamination: float = 0.05) -> pd.Series:
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    cols = [c for c in cols if c != TARGET_BINARY]
    subset = df[cols].fillna(0)

    iso = IsolationForest(contamination=contamination, random_state=RANDOM_STATE, n_jobs=-1)
    preds = iso.fit_predict(subset)
    outlier_mask = pd.Series(preds == -1, index=df.index)

    logger.info(f"Isolation Forest outliers: {outlier_mask.sum():,} rows ({outlier_mask.mean():.1%})")
    return outlier_mask


def split_data(df: pd.DataFrame, target_col: str = TARGET_BINARY,
               use_model_features: bool = True):
    from src.config import MODEL_FEATURES

    y = df[target_col]
    if use_model_features:
        available = [c for c in MODEL_FEATURES if c in df.columns]
        missing = [c for c in MODEL_FEATURES if c not in df.columns]
        if missing:
            logger.warning(f"MODEL_FEATURES missing from data: {missing}")
        X = df[available].copy()
        logger.info(f"Using {len(available)} canonical model features")
    else:
        X = df.drop(columns=[target_col, "loan_status"] if "loan_status" in df.columns else [target_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Split: train={len(X_train):,}, test={len(X_test):,}, features={X.shape[1]}")
    return X_train, X_test, y_train, y_test


def run_preprocessing(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    logger.info(f"Starting preprocessing on {len(df):,} rows, {len(df.columns)} cols...")

    df = drop_high_null_columns(df, threshold=0.50)
    df = impute_numerical(df, strategy="median")
    df = impute_categorical(df)

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in [TARGET_BINARY, "loan_status"]]
    for col in cat_cols:
        df[col] = df[col].astype(str)

    df, encoders = encode_categoricals(df, method="target")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = impute_numerical(df, strategy="median")

    logger.info(f"Preprocessing complete: {len(df):,} rows, {len(df.columns)} cols")
    return df, encoders
