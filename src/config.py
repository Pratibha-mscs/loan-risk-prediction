from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURE_STORE_DIR = DATA_DIR / "feature_store"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "loan_risk_db",
    "user": "pratibhagiri",
    "password": "",
}

DB_URL = (
    f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
)

TARGET_COL = "loan_status"
TARGET_BINARY = "is_default"
TARGET_RISK_TIER = "risk_tier"
TARGET_EXPECTED_LOSS = "expected_loss"

DEFAULT_LABEL = 1  # Charged Off
PAID_LABEL = 0     # Fully Paid

# FICO-aligned credit score tiers (score-based, not PD-based)
# Low Risk (Prime/Super-Prime): 740–850, real-world default < 5%
# Medium Risk (Near-Prime):     670–739, real-world default 5–15%
# High Risk (Subprime):         300–669, real-world default 30–50%+
CREDIT_SCORE_TIERS = {
    "Low Risk": (740, 850),
    "Medium Risk": (670, 739),
    "High Risk": (300, 669),
}

RISK_SCORE_RANGE = (300, 850)

# PD-to-score multiplier. Steeper than 550 so that FICO tier boundaries
# (740, 670) align with meaningful PD thresholds:
#   PD ~16% → score 740 (Low/Medium boundary)
#   PD ~26% → score 670 (Medium/High boundary)
PD_SCORE_MULTIPLIER = 700

# Industry standard LGD for unsecured consumer lending (Moody's, Basel III guidance)
LGD_DEFAULT = 0.45

LOAN_STATUSES_KEEP = ["Fully Paid", "Charged Off"]

DEV_SAMPLE_SIZE = 50_000
TUNE_SAMPLE_SIZE = 250_000

RANDOM_STATE = 42
TEST_SIZE = 0.2

MLFLOW_EXPERIMENT = "credit_risk_models"
MLFLOW_TRACKING_URI = "sqlite:///mlruns/mlflow.db"

NUMERICAL_FEATURES = [
    "loan_amnt", "funded_amnt", "int_rate", "installment", "annual_inc",
    "dti", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "tot_cur_bal", "total_rev_hi_lim",
]

CATEGORICAL_FEATURES = [
    "term", "grade", "sub_grade", "emp_length", "home_ownership",
    "verification_status", "purpose", "addr_state", "application_type",
]

# Canonical feature set used for BOTH training and serving.
# Keeping these identical ensures the API/dashboard feed the model exactly
# what it was trained on (no zero-padding, no leakage, coherent SHAP).
MODEL_FEATURES = [
    # Raw applicant inputs
    "annual_inc", "loan_amnt", "funded_amnt", "dti", "int_rate", "installment",
    "emp_length_years", "revol_util", "revol_bal", "open_acc", "total_acc",
    "delinq_2yrs", "inq_last_6mths", "pub_rec", "grade_numeric", "sub_grade_numeric",
    # Engineered financial ratios
    # NOTE: 'monthly_payment_ratio' (near-constant noise) and 'funded_ratio'
    # (always 1.0 at serving → train/serve skew) deliberately excluded.
    "loan_income_ratio", "debt_income_ratio", "installment_income_ratio",
    "revolving_utilization_ratio", "interest_payment_ratio",
    # Credit features
    # NOTE: 'credit_utilization_overall' excluded — requires tot_cur_bal and
    # total_rev_hi_lim which are not available at application time in our form.
    "total_credit_lines_ratio", "delinquency_rate", "inquiry_frequency",
    "public_record_ratio", "revolving_balance_per_account",
    # Risk & behavioral scores
    "employment_stability", "financial_health_score", "credit_strength_score",
    "repayment_capacity_score",
    # Interaction features
    "income_x_credit_strength", "dti_x_loan_amount", "revol_bal_x_income",
    "int_rate_x_loan", "installment_x_dti", "income_x_employment",
]

FEATURES_TO_DROP = [
    "id", "member_id", "url", "desc", "title", "zip_code",
    "emp_title", "issue_d", "earliest_cr_line", "last_pymnt_d",
    "next_pymnt_d", "last_credit_pull_d", "policy_code",
    # Post-outcome leakage: only known after loan repayment
    "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "last_pymnt_amnt", "collection_recovery_fee",
    "recoveries", "out_prncp", "out_prncp_inv",
]
