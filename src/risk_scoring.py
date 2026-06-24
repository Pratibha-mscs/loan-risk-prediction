import numpy as np
import pandas as pd

from src.config import CREDIT_SCORE_TIERS, RISK_SCORE_RANGE, LGD_DEFAULT, PD_SCORE_MULTIPLIER
from src.utils import get_logger

logger = get_logger(__name__)


def pd_to_credit_score(pd_value: float) -> int:
    min_score, max_score = RISK_SCORE_RANGE
    score = max_score - (pd_value * PD_SCORE_MULTIPLIER)
    return int(np.clip(score, min_score, max_score))


def assign_risk_tier(credit_score: int) -> str:
    for tier_name, (low, high) in CREDIT_SCORE_TIERS.items():
        if low <= credit_score <= high:
            return tier_name
    return "High Risk"


def calculate_expected_loss(pd_value: float, exposure: float,
                            lgd: float = LGD_DEFAULT) -> float:
    return pd_value * exposure * lgd


def generate_risk_profile(pd_value: float, loan_amount: float,
                          annual_income: float, dti: float,
                          int_rate: float) -> dict:
    credit_score = pd_to_credit_score(pd_value)
    risk_tier = assign_risk_tier(credit_score)
    expected_loss = calculate_expected_loss(pd_value, loan_amount)

    return {
        "probability_of_default": round(pd_value * 100, 2),
        "credit_score": credit_score,
        "risk_tier": risk_tier,
        "expected_loss": round(expected_loss, 2),
        "loan_amount": loan_amount,
        "annual_income": annual_income,
        "dti": dti,
        "interest_rate": int_rate,
        "loan_to_income": round(loan_amount / max(annual_income, 1) * 100, 2),
    }


def score_portfolio(df: pd.DataFrame, pd_col: str = "predicted_pd") -> pd.DataFrame:
    df["credit_score"] = df[pd_col].apply(pd_to_credit_score)
    df["risk_tier"] = df["credit_score"].apply(assign_risk_tier)
    df["expected_loss"] = df.apply(
        lambda row: calculate_expected_loss(row[pd_col], row["loan_amnt"]),
        axis=1,
    )

    logger.info("Portfolio Risk Summary:")
    tier_summary = df.groupby("risk_tier").agg(
        count=("credit_score", "size"),
        avg_credit_score=("credit_score", "mean"),
        avg_pd=(pd_col, "mean"),
        total_exposure=("loan_amnt", "sum"),
        total_expected_loss=("expected_loss", "sum"),
    ).round(2)
    logger.info("\n" + tier_summary.to_string())

    return df
