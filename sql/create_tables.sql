CREATE TABLE IF NOT EXISTS raw_loans (
    id SERIAL PRIMARY KEY,
    loan_amnt NUMERIC,
    funded_amnt NUMERIC,
    term VARCHAR(20),
    int_rate NUMERIC,
    installment NUMERIC,
    grade VARCHAR(5),
    sub_grade VARCHAR(5),
    emp_length VARCHAR(20),
    home_ownership VARCHAR(20),
    annual_inc NUMERIC,
    verification_status VARCHAR(30),
    loan_status VARCHAR(30),
    purpose VARCHAR(30),
    addr_state VARCHAR(5),
    dti NUMERIC,
    delinq_2yrs NUMERIC,
    inq_last_6mths NUMERIC,
    open_acc NUMERIC,
    pub_rec NUMERIC,
    revol_bal NUMERIC,
    revol_util NUMERIC,
    total_acc NUMERIC,
    application_type VARCHAR(20),
    tot_cur_bal NUMERIC,
    total_rev_hi_lim NUMERIC,
    is_default INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS feature_store (
    id SERIAL PRIMARY KEY,
    loan_id INTEGER REFERENCES raw_loans(id),
    debt_income_ratio NUMERIC,
    loan_income_ratio NUMERIC,
    revolving_utilization_ratio NUMERIC,
    monthly_payment_ratio NUMERIC,
    credit_history_length NUMERIC,
    delinquency_rate NUMERIC,
    inquiry_frequency NUMERIC,
    financial_health_score NUMERIC,
    credit_strength_score NUMERIC,
    repayment_capacity_score NUMERIC,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    loan_id INTEGER,
    model_version VARCHAR(50),
    probability_of_default NUMERIC,
    risk_tier VARCHAR(20),
    expected_loss NUMERIC,
    risk_score INTEGER,
    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_raw_loans_status ON raw_loans(loan_status);
CREATE INDEX IF NOT EXISTS idx_raw_loans_grade ON raw_loans(grade);
CREATE INDEX IF NOT EXISTS idx_predictions_tier ON predictions(risk_tier);
