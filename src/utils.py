import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from src.config import FIGURES_DIR


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def save_figure(fig: plt.Figure, name: str, dpi: int = 150) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def set_plot_style():
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update({
        "figure.figsize": (12, 6),
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })
