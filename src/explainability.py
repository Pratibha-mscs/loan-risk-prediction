import pandas as pd
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer

from src.utils import get_logger, save_figure, set_plot_style

logger = get_logger(__name__)
set_plot_style()

FEATURE_DESCRIPTIONS = {
    "int_rate": "Interest Rate",
    "dti": "Debt-to-Income Ratio",
    "annual_inc": "Annual Income",
    "loan_amnt": "Loan Amount",
    "revol_util": "Revolving Credit Utilization",
    "revol_bal": "Revolving Balance",
    "installment": "Monthly Installment",
    "total_acc": "Total Credit Accounts",
    "open_acc": "Open Credit Accounts",
    "delinq_2yrs": "Delinquencies (2 years)",
    "inq_last_6mths": "Recent Credit Inquiries",
    "pub_rec": "Public Records",
    "grade_numeric": "Loan Grade",
    "sub_grade_numeric": "Loan Sub-Grade",
    "loan_income_ratio": "Loan-to-Income Ratio",
    "debt_income_ratio": "Debt-to-Income Ratio",
    "financial_health_score": "Financial Health Score",
    "credit_strength_score": "Credit Strength Score",
    "repayment_capacity_score": "Repayment Capacity",
    "employment_stability": "Employment Stability",
}


def compute_shap_values(model, X: pd.DataFrame, max_samples: int = 1000):
    logger.info(f"Computing SHAP values on {min(len(X), max_samples)} samples...")
    X_sample = X.sample(n=min(len(X), max_samples), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values_raw = explainer.shap_values(X_sample)
    # For binary classifiers, shap_values is a list [class_0, class_1]
    if isinstance(shap_values_raw, list):
        shap_vals = shap_values_raw[1]
    else:
        shap_vals = shap_values_raw
    return shap_vals, X_sample


def plot_shap_summary(shap_values, X_sample: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 10))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
    save_figure(plt.gcf(), "shap_summary")
    return fig


def plot_shap_bar(shap_values, X_sample: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=20)
    save_figure(plt.gcf(), "shap_bar")
    return fig


def plot_shap_dependence(shap_values, X_sample: pd.DataFrame,
                         feature: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.dependence_plot(feature, shap_values.values, X_sample, show=False, ax=ax)
    save_figure(fig, f"shap_dependence_{feature}")
    return fig


def explain_single_prediction_lime(model, X_train: pd.DataFrame,
                                    instance: pd.Series) -> dict:
    explainer = LimeTabularExplainer(
        X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=["Fully Paid", "Default"],
        mode="classification",
    )
    exp = explainer.explain_instance(
        instance.values, model.predict_proba, num_features=10
    )
    return {
        "factors": exp.as_list(),
        "prediction": model.predict_proba(instance.values.reshape(1, -1))[0],
    }


def generate_business_explanation(model, instance: pd.Series,
                                   feature_names: list[str]) -> dict:
    proba = model.predict_proba(instance.values.reshape(1, -1))[0]
    pd_score = proba[1]

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(instance.values.reshape(1, -1))
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]
        else:
            shap_vals = shap_values[0]

        feature_impacts = sorted(
            zip(feature_names, shap_vals, instance.values),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
    except Exception:
        feature_impacts = [(f, 0, v) for f, v in zip(feature_names, instance.values)]

    top_risk_factors = []
    top_positive_factors = []
    for feat, impact, value in feature_impacts[:10]:
        desc = FEATURE_DESCRIPTIONS.get(feat, feat.replace("_", " ").title())
        if impact > 0:
            top_risk_factors.append(f"{desc} (value: {value:.2f})")
        else:
            top_positive_factors.append(f"{desc} (value: {value:.2f})")

    if pd_score >= 0.5:
        decision = "HIGH RISK — Recommend Denial"
    elif pd_score >= 0.25:
        decision = "ELEVATED RISK — Recommend Review"
    elif pd_score >= 0.10:
        decision = "MODERATE RISK — Conditional Approval"
    else:
        decision = "LOW RISK — Recommend Approval"

    return {
        "probability_of_default": round(pd_score * 100, 2),
        "decision": decision,
        "top_risk_factors": top_risk_factors[:5],
        "top_positive_factors": top_positive_factors[:5],
    }
