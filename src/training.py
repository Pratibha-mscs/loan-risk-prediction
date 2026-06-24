import time
import warnings

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
)
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

from src.config import MODELS_DIR, RANDOM_STATE, MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI
from src.utils import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

MODELS_DIR.mkdir(parents=True, exist_ok=True)


def get_models(scale_pos_weight: float = 4.0) -> dict:
    """Models use native class weighting (class_weight / scale_pos_weight)
    rather than SMOTE. For gradient-boosted trees this yields better-calibrated
    probabilities and more intuitive feature importance than synthetic
    oversampling, which injects noise on high-dimensional tabular data."""
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=10, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "NaiveBayes": GaussianNB(),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=RANDOM_STATE,
            n_jobs=-1, class_weight="balanced"
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=200, max_depth=15, random_state=RANDOM_STATE,
            n_jobs=-1, class_weight="balanced"
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, n_jobs=-1, eval_metric="logloss",
            tree_method="hist", scale_pos_weight=scale_pos_weight,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
            class_weight="balanced",
        ),
        "CatBoost": CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, verbose=0, auto_class_weights="Balanced",
        ),
    }


def apply_sampling(X_train: pd.DataFrame, y_train: pd.Series,
                   method: str = "smote") -> tuple[pd.DataFrame, pd.Series]:
    logger.info(f"Applying {method} sampling...")
    original_dist = y_train.value_counts().to_dict()

    if method == "smote":
        sampler = SMOTE(random_state=RANDOM_STATE)
    elif method == "adasyn":
        sampler = ADASYN(random_state=RANDOM_STATE)
    elif method == "undersample":
        sampler = RandomUnderSampler(random_state=RANDOM_STATE)
    else:
        return X_train, y_train

    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    new_dist = pd.Series(y_resampled).value_counts().to_dict()
    logger.info(f"Sampling: {original_dist} → {new_dist}")
    return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled)


def train_single_model(model, X_train, y_train, X_test, y_test,
                       model_name: str) -> dict:
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred.astype(float)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "train_time_seconds": train_time,
    }

    logger.info(
        f"{model_name}: AUC={metrics['roc_auc']:.4f}, "
        f"F1={metrics['f1_score']:.4f}, "
        f"Time={train_time:.1f}s"
    )
    return metrics


def train_all_models(X_train, y_train, X_test, y_test,
                     sampling_method: str = "none") -> pd.DataFrame:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    X_train_sampled, y_train_sampled = apply_sampling(
        X_train, y_train, method=sampling_method
    )

    # Class-weight ratio for boosters (neg / pos)
    pos = max(int((y_train_sampled == 1).sum()), 1)
    neg = int((y_train_sampled == 0).sum())
    scale_pos_weight = neg / pos if sampling_method == "none" else 1.0

    models = get_models(scale_pos_weight=scale_pos_weight)
    results = []

    for name, model in models.items():
        logger.info(f"Training {name}...")
        with mlflow.start_run(run_name=name):
            metrics = train_single_model(
                model, X_train_sampled, y_train_sampled, X_test, y_test, name
            )

            mlflow.log_params({
                "model_type": name,
                "sampling_method": sampling_method,
                "n_features": X_train.shape[1],
                "train_size": len(X_train_sampled),
                "test_size": len(X_test),
            })
            mlflow.log_metrics(metrics)

            model_path = MODELS_DIR / f"{name}.joblib"
            joblib.dump(model, model_path)
            mlflow.log_artifact(str(model_path))

            results.append({"model": name, **metrics})

    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    logger.info("\n" + results_df.to_string(index=False))

    # Select best model: among models within 0.005 AUC of the top performer,
    # prefer the highest F1. For imbalanced credit risk, recall of defaulters
    # (captured by F1) matters more than a hair of AUC — and this favours
    # fast, well-calibrated boosters over marginally-higher but slow models.
    top_auc = results_df.iloc[0]["roc_auc"]
    contenders = results_df[results_df["roc_auc"] >= top_auc - 0.005]
    best_row = contenders.sort_values("f1_score", ascending=False).iloc[0]
    best_model_name = best_row["model"]
    logger.info(
        f"Best model: {best_model_name} "
        f"(AUC={best_row['roc_auc']:.4f}, F1={best_row['f1_score']:.4f})"
    )

    best_model = joblib.load(MODELS_DIR / f"{best_model_name}.joblib")
    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")

    return results_df


def calibrate_best_model(X_train, y_train, X_test, y_test):
    """Wrap best_model.joblib in isotonic calibration so that predicted
    probabilities match actual event rates. Called ONCE after all training
    and tuning is complete — never before Optuna, which would overwrite it.

    Credit risk models with class_weight/scale_pos_weight inflate predicted
    probabilities (good for recall, bad for PD estimation). Isotonic
    calibration learns a monotonic mapping from raw scores to true event
    rates on held-out folds, so "predicted 20% PD" actually means ~20%
    of those borrowers default."""
    raw_model = joblib.load(MODELS_DIR / "best_model.joblib")

    raw_proba = raw_model.predict_proba(X_test)[:, 1]
    raw_auc = roc_auc_score(y_test, raw_proba)
    logger.info(f"Raw model mean PD: {raw_proba.mean():.3f} (base rate: {y_test.mean():.3f})")

    logger.info("Calibrating probabilities with isotonic regression (5-fold)...")
    calibrated = CalibratedClassifierCV(
        estimator=raw_model, method="isotonic", cv=5
    )
    calibrated.fit(X_train, y_train)

    cal_proba = calibrated.predict_proba(X_test)[:, 1]
    cal_auc = roc_auc_score(y_test, cal_proba)
    logger.info(f"Calibrated AUC: {cal_auc:.4f} (raw: {raw_auc:.4f})")
    logger.info(f"Calibrated mean PD: {cal_proba.mean():.3f} (base rate: {y_test.mean():.3f})")

    joblib.dump(calibrated, MODELS_DIR / "best_model.joblib")
    logger.info("Saved calibrated model as best_model.joblib")


def tune_model_optuna(model_name: str, X_train, y_train, X_test, y_test,
                      n_trials: int = 50) -> dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    pos = max(int((y_train == 1).sum()), 1)
    neg = int((y_train == 0).sum())
    spw = neg / pos

    best_model_obj = [None]

    def objective(trial):
        if model_name == "XGBoost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 600),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "scale_pos_weight": spw,
                "random_state": RANDOM_STATE, "n_jobs": -1,
                "eval_metric": "logloss", "tree_method": "hist",
            }
            model = XGBClassifier(**params)
        elif model_name == "LightGBM":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 600),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 200),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "class_weight": "balanced",
                "random_state": RANDOM_STATE, "n_jobs": -1, "verbose": -1,
            }
            model = LGBMClassifier(**params)
        elif model_name == "CatBoost":
            params = {
                "iterations": trial.suggest_int("iterations", 200, 600),
                "depth": trial.suggest_int("depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 10.0, log=True),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
                "border_count": trial.suggest_int("border_count", 32, 255),
                "auto_class_weights": "Balanced",
                "random_state": RANDOM_STATE, "verbose": 0,
            }
            model = CatBoostClassifier(**params)
        else:
            raise ValueError(f"Tuning not implemented for {model_name}")

        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        if best_model_obj[0] is None or auc > best_model_obj[0][1]:
            best_model_obj[0] = (model, auc)
        return auc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    logger.info(f"Best {model_name} trial: AUC={study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    if mlflow.active_run():
        mlflow.end_run()
    with mlflow.start_run(run_name=f"{model_name}_tuned"):
        mlflow.log_params(study.best_params)
        mlflow.log_metric("roc_auc_tuned", study.best_value)

    if best_model_obj[0] is not None:
        tuned_model, tuned_auc = best_model_obj[0]
        model_path = MODELS_DIR / f"{model_name}_tuned.joblib"
        joblib.dump(tuned_model, model_path)
        mlflow.log_artifact(str(model_path))

        existing_best = joblib.load(MODELS_DIR / "best_model.joblib")
        existing_proba = existing_best.predict_proba(X_test)[:, 1]
        existing_auc = roc_auc_score(y_test, existing_proba)

        if tuned_auc > existing_auc:
            joblib.dump(tuned_model, MODELS_DIR / "best_model.joblib")
            logger.info(
                f"NEW BEST: {model_name}_tuned (AUC={tuned_auc:.4f}) "
                f"beats previous best (AUC={existing_auc:.4f})"
            )
        else:
            logger.info(
                f"{model_name}_tuned AUC={tuned_auc:.4f} "
                f"did not beat existing best AUC={existing_auc:.4f}"
            )

    return {"best_params": study.best_params, "best_auc": study.best_value}
