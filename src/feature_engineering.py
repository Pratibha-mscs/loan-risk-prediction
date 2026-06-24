import pandas as pd

from src.config import LGD_DEFAULT, TARGET_RISK_TIER, TARGET_EXPECTED_LOSS
from src.utils import get_logger

logger = get_logger(__name__)


def add_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df["loan_income_ratio"] = df["loan_amnt"] / (df["annual_inc"] + 1)
    df["debt_income_ratio"] = df["dti"] / 100
    df["installment_income_ratio"] = df["installment"] / (df["annual_inc"] / 12 + 1)
    df["revolving_utilization_ratio"] = df["revol_util"].clip(0, 150) / 100
    df["monthly_payment_ratio"] = df["installment"] / (df["loan_amnt"] / df.get("term_months", pd.Series([36] * len(df))) + 1)
    df["funded_ratio"] = df["funded_amnt"] / (df["loan_amnt"] + 1)
    df["interest_payment_ratio"] = (df["int_rate"] / 100 * df["loan_amnt"]) / (df["annual_inc"] + 1)
    logger.info("Added 7 financial ratio features")
    return df


def add_credit_features(df: pd.DataFrame) -> pd.DataFrame:
    df["total_credit_lines_ratio"] = df["open_acc"] / (df["total_acc"] + 1)
    df["delinquency_rate"] = df["delinq_2yrs"] / (df["total_acc"] + 1)
    if "inq_last_6mths" in df.columns:
        df["inquiry_frequency"] = df["inq_last_6mths"] / 6
    else:
        df["inquiry_frequency"] = 0.0
    df["public_record_ratio"] = df["pub_rec"] / (df["total_acc"] + 1)
    df["revolving_balance_per_account"] = df["revol_bal"] / (df["open_acc"] + 1)
    if "tot_cur_bal" in df.columns and "total_rev_hi_lim" in df.columns:
        df["credit_utilization_overall"] = df["tot_cur_bal"] / (df["total_rev_hi_lim"] + 1)
    logger.info("Added 6 credit features")
    return df


def add_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    df["financial_health_score"] = (
        0.3 * (1 - df["debt_income_ratio"].clip(0, 1))
        + 0.25 * (1 - df["revolving_utilization_ratio"].clip(0, 1))
        + 0.2 * (1 - df["delinquency_rate"].clip(0, 1))
        + 0.15 * (1 - df["loan_income_ratio"].clip(0, 1))
        + 0.1 * df["total_credit_lines_ratio"].clip(0, 1)
    )

    df["credit_strength_score"] = (
        0.35 * (1 - df.get("revolving_utilization_ratio", pd.Series([0.5] * len(df))).clip(0, 1))
        + 0.25 * df["total_credit_lines_ratio"].clip(0, 1)
        + 0.2 * (1 - df["delinquency_rate"].clip(0, 1))
        + 0.2 * (1 - df["public_record_ratio"].clip(0, 1))
    )

    df["repayment_capacity_score"] = (
        0.4 * (1 - df["installment_income_ratio"].clip(0, 1))
        + 0.3 * (1 - df["debt_income_ratio"].clip(0, 1))
        + 0.3 * (1 - df["loan_income_ratio"].clip(0, 1))
    )
    logger.info("Added 3 risk score features")
    return df


def add_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    if "emp_length_years" in df.columns:
        df["employment_stability"] = df["emp_length_years"].fillna(0) / 10
    else:
        df["employment_stability"] = 0.5

    if "total_pymnt" in df.columns and "funded_amnt" in df.columns:
        df["payment_completion_ratio"] = df["total_pymnt"] / (df["funded_amnt"] + 1)

    if "total_rec_prncp" in df.columns and "total_pymnt" in df.columns:
        df["principal_payment_ratio"] = df["total_rec_prncp"] / (df["total_pymnt"] + 1)

    if "last_pymnt_amnt" in df.columns and "installment" in df.columns:
        df["last_payment_ratio"] = df["last_pymnt_amnt"] / (df["installment"] + 1)

    logger.info("Added 4 behavioral features")
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df["income_x_credit_strength"] = df["annual_inc"] * df["credit_strength_score"]
    df["dti_x_loan_amount"] = df["dti"] * df["loan_amnt"]
    df["revol_bal_x_income"] = df["revol_bal"] / (df["annual_inc"] + 1)
    df["int_rate_x_loan"] = df["int_rate"] * df["loan_amnt"]
    df["installment_x_dti"] = df["installment"] * df["dti"]
    df["income_x_employment"] = df["annual_inc"] * df["employment_stability"]
    logger.info("Added 6 interaction features")
    return df


def add_binned_features(df: pd.DataFrame) -> pd.DataFrame:
    df["income_bracket"] = pd.cut(
        df["annual_inc"],
        bins=[0, 30000, 50000, 75000, 100000, 150000, float("inf")],
        labels=["<30K", "30-50K", "50-75K", "75-100K", "100-150K", "150K+"],
    )
    df["loan_amount_bracket"] = pd.cut(
        df["loan_amnt"],
        bins=[0, 5000, 10000, 15000, 25000, float("inf")],
        labels=["<5K", "5-10K", "10-15K", "15-25K", "25K+"],
    )
    df["int_rate_bracket"] = pd.cut(
        df["int_rate"],
        bins=[0, 7, 10, 13, 17, float("inf")],
        labels=["Low", "Medium-Low", "Medium", "Medium-High", "High"],
    )
    logger.info("Added 3 binned features")
    return df


def add_grade_numeric(df: pd.DataFrame) -> pd.DataFrame:
    if "grade" in df.columns:
        grade_map = {g: i for i, g in enumerate(["A", "B", "C", "D", "E", "F", "G"])}
        df["grade_numeric"] = df["grade"].map(grade_map)
    if "sub_grade" in df.columns:
        sub_grades = [f"{g}{n}" for g in "ABCDEFG" for n in range(1, 6)]
        sg_map = {sg: i for i, sg in enumerate(sub_grades)}
        df["sub_grade_numeric"] = df["sub_grade"].map(sg_map)
    logger.info("Added grade numeric features")
    return df


def create_risk_tiers(df: pd.DataFrame, pd_col: str = "predicted_pd") -> pd.DataFrame:
    if pd_col not in df.columns:
        logger.info("No PD column found; skipping risk tier creation")
        return df
    from src.risk_scoring import pd_to_credit_score, assign_risk_tier
    df["credit_score"] = df[pd_col].apply(pd_to_credit_score)
    df[TARGET_RISK_TIER] = df["credit_score"].apply(assign_risk_tier)
    logger.info("Created risk tier labels")
    return df


def create_expected_loss(df: pd.DataFrame, pd_col: str = "predicted_pd") -> pd.DataFrame:
    if pd_col not in df.columns:
        logger.info("No PD column found; skipping expected loss")
        return df
    df[TARGET_EXPECTED_LOSS] = df[pd_col] * df["loan_amnt"] * LGD_DEFAULT
    logger.info("Created expected loss column")
    return df


def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    logger.info(f"Starting feature engineering on {len(df):,} rows...")
    df = add_financial_ratios(df)
    df = add_credit_features(df)
    df = add_risk_features(df)
    df = add_behavioral_features(df)
    df = add_interaction_features(df)
    df = add_binned_features(df)
    df = add_grade_numeric(df)

    new_features = [
        "loan_income_ratio", "debt_income_ratio", "installment_income_ratio",
        "revolving_utilization_ratio", "monthly_payment_ratio", "funded_ratio",
        "interest_payment_ratio", "total_credit_lines_ratio", "delinquency_rate",
        "inquiry_frequency", "public_record_ratio", "revolving_balance_per_account",
        "credit_utilization_overall", "financial_health_score", "credit_strength_score",
        "repayment_capacity_score", "employment_stability", "payment_completion_ratio",
        "principal_payment_ratio", "last_payment_ratio", "income_x_credit_strength",
        "dti_x_loan_amount", "revol_bal_x_income", "int_rate_x_loan",
        "installment_x_dti", "income_x_employment", "grade_numeric", "sub_grade_numeric",
    ]
    present = [f for f in new_features if f in df.columns]
    logger.info(f"Feature engineering complete: {len(present)} new features added")
    return df
