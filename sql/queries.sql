-- Default rate by loan grade
SELECT
    grade,
    COUNT(*) AS total_loans,
    SUM(is_default) AS defaults,
    ROUND(AVG(is_default) * 100, 2) AS default_rate_pct,
    ROUND(AVG(loan_amnt), 2) AS avg_loan_amount,
    ROUND(AVG(int_rate), 2) AS avg_interest_rate
FROM raw_loans
GROUP BY grade
ORDER BY grade;

-- Default rate by home ownership
SELECT
    home_ownership,
    COUNT(*) AS total_loans,
    ROUND(AVG(is_default) * 100, 2) AS default_rate_pct,
    ROUND(AVG(annual_inc), 2) AS avg_income
FROM raw_loans
GROUP BY home_ownership
ORDER BY default_rate_pct DESC;

-- Risk distribution by state
SELECT
    addr_state,
    COUNT(*) AS total_loans,
    ROUND(AVG(is_default) * 100, 2) AS default_rate_pct,
    ROUND(AVG(loan_amnt), 2) AS avg_loan_amount
FROM raw_loans
GROUP BY addr_state
ORDER BY default_rate_pct DESC
LIMIT 20;

-- Loan purpose analysis
SELECT
    purpose,
    COUNT(*) AS total_loans,
    ROUND(AVG(is_default) * 100, 2) AS default_rate_pct,
    ROUND(AVG(loan_amnt), 2) AS avg_loan_amount,
    ROUND(AVG(int_rate), 2) AS avg_interest_rate
FROM raw_loans
GROUP BY purpose
ORDER BY default_rate_pct DESC;

-- Income bracket analysis
SELECT
    CASE
        WHEN annual_inc < 30000 THEN '<30K'
        WHEN annual_inc < 50000 THEN '30-50K'
        WHEN annual_inc < 75000 THEN '50-75K'
        WHEN annual_inc < 100000 THEN '75-100K'
        WHEN annual_inc < 150000 THEN '100-150K'
        ELSE '150K+'
    END AS income_bracket,
    COUNT(*) AS total_loans,
    ROUND(AVG(is_default) * 100, 2) AS default_rate_pct,
    ROUND(AVG(dti), 2) AS avg_dti
FROM raw_loans
GROUP BY income_bracket
ORDER BY MIN(annual_inc);

-- Portfolio summary
SELECT
    COUNT(*) AS total_loans,
    SUM(loan_amnt) AS total_portfolio_value,
    ROUND(AVG(is_default) * 100, 2) AS overall_default_rate,
    ROUND(AVG(int_rate), 2) AS avg_interest_rate,
    ROUND(AVG(annual_inc), 2) AS avg_borrower_income,
    ROUND(AVG(dti), 2) AS avg_dti
FROM raw_loans;

-- Expected loss estimation
SELECT
    grade,
    COUNT(*) AS total_loans,
    ROUND(SUM(loan_amnt * is_default * 0.60), 2) AS total_expected_loss,
    ROUND(AVG(loan_amnt * is_default * 0.60), 2) AS avg_expected_loss_per_loan
FROM raw_loans
GROUP BY grade
ORDER BY grade;
