"""
Production-Grade Credit Risk Modeling & Loan Default Prediction Platform
========================================================================
End-to-end pipeline: Ingest → Validate → Feature Engineering → Train → Evaluate → Deploy

Usage:
    python run_pipeline.py                  # Full pipeline (50K dev sample)
    python run_pipeline.py --sample 250000  # Tuning sample
    python run_pipeline.py --full           # Full dataset
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import DEV_SAMPLE_SIZE, TUNE_SAMPLE_SIZE
from src.utils import get_logger

logger = get_logger("pipeline")


def run(sample_size: int | None = None):
    start = time.time()

    # Step 1: Data Ingestion
    logger.info("=" * 60)
    logger.info("STEP 1: DATA INGESTION")
    logger.info("=" * 60)
    from src.data_ingestion import run_ingestion, load_from_parquet
    from src.data_validation import run_validation

    df = run_ingestion(sample_size=sample_size)
    validation = run_validation(df)
    if not validation["passed"]:
        logger.error("Data validation failed. Aborting.")
        sys.exit(1)

    # Step 2: Database Setup
    logger.info("=" * 60)
    logger.info("STEP 2: DATABASE SETUP")
    logger.info("=" * 60)
    from src.database import setup_database
    setup_database(df)

    # Step 3: Feature Engineering
    logger.info("=" * 60)
    logger.info("STEP 3: FEATURE ENGINEERING")
    logger.info("=" * 60)
    from src.feature_engineering import run_feature_engineering
    from src.feature_store import save_features
    df = run_feature_engineering(df)
    save_features(df)

    # Step 4: Preprocessing
    logger.info("=" * 60)
    logger.info("STEP 4: PREPROCESSING")
    logger.info("=" * 60)
    from src.preprocessing import run_preprocessing, split_data
    df, encoders = run_preprocessing(df)
    X_train, X_test, y_train, y_test = split_data(df)

    # Step 5: Model Training
    logger.info("=" * 60)
    logger.info("STEP 5: MODEL TRAINING")
    logger.info("=" * 60)
    from src.training import train_all_models, tune_model_optuna, calibrate_best_model
    results_df = train_all_models(X_train, y_train, X_test, y_test, sampling_method="none")

    # Step 5b: Optuna hyperparameter tuning on top 3 boosters (if sample >= 100K)
    if len(X_train) >= 80_000:
        logger.info("=" * 60)
        logger.info("STEP 5b: HYPERPARAMETER TUNING (Optuna)")
        logger.info("=" * 60)
        n_trials = 40 if len(X_train) < 200_000 else 30
        for booster in ["CatBoost", "LightGBM", "XGBoost"]:
            logger.info(f"Tuning {booster} ({n_trials} trials)...")
            tuned = tune_model_optuna(
                booster, X_train, y_train, X_test, y_test, n_trials=n_trials
            )
            logger.info(f"{booster} tuned AUC: {tuned['best_auc']:.4f}")

    # Step 5c: Probability calibration — MUST run last, after all tuning
    logger.info("=" * 60)
    logger.info("STEP 5c: PROBABILITY CALIBRATION")
    logger.info("=" * 60)
    calibrate_best_model(X_train, y_train, X_test, y_test)

    # Step 6: Evaluation Plots
    logger.info("=" * 60)
    logger.info("STEP 6: MODEL EVALUATION")
    logger.info("=" * 60)
    import joblib
    from src.config import MODELS_DIR
    from src.evaluation import (
        plot_confusion_matrix, plot_roc_curves, plot_precision_recall_curves,
        plot_lift_curve, plot_model_comparison,
    )

    # Use the model actually selected/saved by train_all_models (best_model.joblib),
    # which may differ from the top-AUC row due to F1 tie-breaking. Map the class
    # name back to its registry name (e.g. "CatBoostClassifier" -> "CatBoost").
    best_model = joblib.load(MODELS_DIR / "best_model.joblib")
    class_to_name = {
        "LogisticRegression": "LogisticRegression", "DecisionTreeClassifier": "DecisionTree",
        "GaussianNB": "NaiveBayes", "RandomForestClassifier": "RandomForest",
        "ExtraTreesClassifier": "ExtraTrees", "GradientBoostingClassifier": "GradientBoosting",
        "XGBClassifier": "XGBoost", "LGBMClassifier": "LightGBM",
        "CatBoostClassifier": "CatBoost",
    }
    best_model_name = class_to_name.get(type(best_model).__name__, type(best_model).__name__)

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    plot_confusion_matrix(y_test, y_pred, best_model_name)
    plot_lift_curve(y_test, y_proba, best_model_name)
    plot_model_comparison(results_df)

    models_dict = {}
    for name in results_df["model"].values[:5]:
        model_path = MODELS_DIR / f"{name}.joblib"
        if model_path.exists():
            models_dict[name] = joblib.load(model_path)
    plot_roc_curves(models_dict, X_test, y_test)
    plot_precision_recall_curves(models_dict, X_test, y_test)

    # Step 7: Explainability
    logger.info("=" * 60)
    logger.info("STEP 7: MODEL EXPLAINABILITY")
    logger.info("=" * 60)
    from src.explainability import compute_shap_values, plot_shap_summary, plot_shap_bar

    try:
        shap_values, X_sample = compute_shap_values(best_model, X_test)
        plot_shap_summary(shap_values, X_sample)
        plot_shap_bar(shap_values, X_sample)
    except Exception as e:
        logger.warning(f"SHAP failed: {e}")

    # Step 8: Risk Scoring
    logger.info("=" * 60)
    logger.info("STEP 8: RISK SCORING")
    logger.info("=" * 60)
    from src.risk_scoring import score_portfolio
    import pandas as pd

    test_df = X_test.copy()
    test_df["predicted_pd"] = y_proba
    test_df["loan_amnt"] = X_test["loan_amnt"] if "loan_amnt" in X_test.columns else 15000
    test_df = score_portfolio(test_df, "predicted_pd")

    # Step 9: Drift Detection
    logger.info("=" * 60)
    logger.info("STEP 9: DRIFT MONITORING")
    logger.info("=" * 60)
    from src.drift_detection import run_drift_monitoring

    train_sample = X_train.sample(min(5000, len(X_train)), random_state=42)
    test_sample = X_test.sample(min(5000, len(X_test)), random_state=42)
    train_sample["is_default"] = y_train.loc[train_sample.index].values
    test_sample["is_default"] = y_test.loc[test_sample.index].values

    try:
        run_drift_monitoring(train_sample, test_sample)
    except Exception as e:
        logger.warning(f"Drift monitoring failed: {e}")

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info(f"PIPELINE COMPLETE in {elapsed:.0f}s")
    deployed_auc = results_df.loc[results_df["model"] == best_model_name, "roc_auc"]
    auc_str = f"{deployed_auc.iloc[0]:.4f}" if len(deployed_auc) else "n/a"
    logger.info(f"Deployed Model: {best_model_name} (AUC={auc_str})")
    logger.info(f"Results saved to reports/figures/")
    logger.info(f"Models saved to models/")
    logger.info("=" * 60)

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Risk Pipeline")
    parser.add_argument("--sample", type=int, default=DEV_SAMPLE_SIZE)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    sample = None if args.full else args.sample
    run(sample_size=sample)
