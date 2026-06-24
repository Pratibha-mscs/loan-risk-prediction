import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier

from src.config import RANDOM_STATE
from src.utils import get_logger

logger = get_logger(__name__)


def mutual_information_scores(X: pd.DataFrame, y: pd.Series,
                               top_n: int = 30) -> pd.Series:
    X_filled = X.fillna(0)
    mi_scores = mutual_info_classif(X_filled, y, random_state=RANDOM_STATE, n_neighbors=5)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    logger.info(f"Top {top_n} features by mutual information:")
    for feat, score in mi_series.head(top_n).items():
        logger.info(f"  {feat}: {score:.4f}")
    return mi_series


def permutation_importance_scores(X: pd.DataFrame, y: pd.Series,
                                   top_n: int = 30) -> pd.Series:
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    X_filled = X.fillna(0)
    rf.fit(X_filled, y)

    result = permutation_importance(rf, X_filled, y, n_repeats=5,
                                     random_state=RANDOM_STATE, n_jobs=-1)
    pi_series = pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)
    logger.info(f"Top {top_n} features by permutation importance:")
    for feat, score in pi_series.head(top_n).items():
        logger.info(f"  {feat}: {score:.4f}")
    return pi_series


def select_features(X: pd.DataFrame, y: pd.Series,
                    method: str = "mutual_info", top_n: int = 30) -> list[str]:
    if method == "mutual_info":
        scores = mutual_information_scores(X, y, top_n)
    elif method == "permutation":
        scores = permutation_importance_scores(X, y, top_n)
    else:
        raise ValueError(f"Unknown method: {method}")

    selected = scores.head(top_n).index.tolist()
    logger.info(f"Selected {len(selected)} features using {method}")
    return selected
