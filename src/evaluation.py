import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve,
)

from src.utils import get_logger, save_figure, set_plot_style

logger = get_logger(__name__)
set_plot_style()


def compute_all_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }


def plot_confusion_matrix(y_true, y_pred, model_name: str = "Model") -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Fully Paid", "Default"],
                yticklabels=["Fully Paid", "Default"])
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    save_figure(fig, f"confusion_matrix_{model_name.lower().replace(' ', '_')}")
    return fig


def plot_roc_curves(models_dict: dict, X_test, y_test) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 8))
    for name, model in models_dict.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend(loc="lower right")
    save_figure(fig, "roc_curves_comparison")
    return fig


def plot_precision_recall_curves(models_dict: dict, X_test, y_test) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 8))
    for name, model in models_dict.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        ax.plot(recall, precision, label=f"{name} (AP={ap:.4f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — All Models")
    ax.legend(loc="upper right")
    save_figure(fig, "pr_curves_comparison")
    return fig


def plot_lift_curve(y_true, y_proba, model_name: str = "Model") -> plt.Figure:
    sorted_idx = np.argsort(y_proba)[::-1]
    y_sorted = np.array(y_true)[sorted_idx]
    cum_defaults = np.cumsum(y_sorted)
    total_defaults = y_sorted.sum()
    pct_population = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
    pct_defaults = cum_defaults / total_defaults
    lift = pct_defaults / pct_population

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(pct_population * 100, pct_defaults * 100)
    axes[0].plot([0, 100], [0, 100], "k--")
    axes[0].set_xlabel("% of Population")
    axes[0].set_ylabel("% of Defaults Captured")
    axes[0].set_title(f"Cumulative Gain Curve — {model_name}")

    axes[1].plot(pct_population * 100, lift)
    axes[1].axhline(y=1, color="k", linestyle="--")
    axes[1].set_xlabel("% of Population")
    axes[1].set_ylabel("Lift")
    axes[1].set_title(f"Lift Curve — {model_name}")

    fig.tight_layout()
    save_figure(fig, f"lift_gain_{model_name.lower().replace(' ', '_')}")
    return fig


def create_comparison_table(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    df = df.sort_values("roc_auc", ascending=False).reset_index(drop=True)
    logger.info("\nModel Comparison:\n" + df.to_string(index=False))
    return df


def plot_model_comparison(results_df: pd.DataFrame) -> plt.Figure:
    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    present = [m for m in metrics if m in results_df.columns]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(results_df))
    width = 0.15

    for i, metric in enumerate(present):
        ax.bar(x + i * width, results_df[metric], width, label=metric)

    ax.set_xticks(x + width * len(present) / 2)
    ax.set_xticklabels(results_df["model"], rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — All Metrics")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    save_figure(fig, "model_comparison")
    return fig
