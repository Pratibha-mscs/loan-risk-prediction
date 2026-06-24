import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import MODELS_DIR, PROCESSED_DIR, LGD_DEFAULT
from src.risk_scoring import (
    pd_to_credit_score, assign_risk_tier, calculate_expected_loss, score_portfolio,
)

st.set_page_config(
    page_title="Credit Risk Analytics Platform",
    page_icon="🏦",
    layout="wide",
)


@st.cache_resource
def load_model():
    model_path = MODELS_DIR / "best_model.joblib"
    if model_path.exists():
        return joblib.load(model_path)
    return None


@st.cache_data
def load_portfolio_data():
    path = PROCESSED_DIR / "loans_cleaned.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


def get_expected_features(model) -> list | None:
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if hasattr(model, "feature_names_"):
        return list(model.feature_names_)
    if hasattr(model, "estimator"):
        return get_expected_features(model.estimator)
    if hasattr(model, "calibrated_classifiers_"):
        inner = model.calibrated_classifiers_[0].estimator
        return get_expected_features(inner)
    return None


def build_features_from_inputs(inputs: dict) -> pd.DataFrame:
    """Build features using the SAME pipeline as training — no duplication."""
    from src.feature_engineering import run_feature_engineering

    grades = "ABCDEFG"
    g = int(min(max(inputs["grade_numeric"], 0), 6))
    sg_idx = int(inputs["sub_grade_numeric"] % 5) + 1

    raw = {
        "annual_inc": inputs["annual_inc"],
        "loan_amnt": inputs["loan_amnt"],
        "funded_amnt": inputs["loan_amnt"],
        "dti": inputs["dti"],
        "int_rate": inputs["int_rate"],
        "installment": inputs["installment"],
        "emp_length_years": inputs["emp_length_years"],
        "revol_util": inputs["revol_util"],
        "revol_bal": inputs["revol_bal"],
        "open_acc": inputs["open_acc"],
        "total_acc": inputs["total_acc"],
        "delinq_2yrs": inputs["delinq_2yrs"],
        "inq_last_6mths": inputs["inq_last_6mths"],
        "pub_rec": inputs["pub_rec"],
        "term_months": 36,
        "grade": grades[g],
        "sub_grade": f"{grades[g]}{sg_idx}",
    }

    df = pd.DataFrame([raw])
    df = run_feature_engineering(df)
    return df


def render_prediction_page():
    st.header("Individual Loan Risk Assessment")

    col1, col2, col3 = st.columns(3)
    with col1:
        annual_income = st.number_input("Annual Income ($)", 10000, 500000, 65000, 5000)
        loan_amount = st.number_input("Loan Amount ($)", 1000, 40000, 15000, 1000)
        dti = st.number_input("Debt-to-Income Ratio (%)", 0.0, 60.0, 18.0, 1.0)
        int_rate = st.number_input("Interest Rate (%)", 5.0, 30.0, 12.0, 0.5)
        installment = st.number_input("Monthly Installment ($)", 50.0, 1500.0, 450.0, 50.0)

    with col2:
        emp_length = st.slider("Employment Length (years)", 0, 15, 5)
        revol_util = st.number_input("Revolving Utilization (%)", 0.0, 150.0, 45.0, 5.0)
        revol_bal = st.number_input("Revolving Balance ($)", 0, 200000, 12000, 1000)
        open_acc = st.number_input("Open Credit Accounts", 0, 50, 8)
        total_acc = st.number_input("Total Credit Accounts", 1, 100, 20)

    with col3:
        delinq_2yrs = st.number_input("Delinquencies (2 yrs)", 0, 20, 0)
        inq_last_6mths = st.number_input("Inquiries (6 months)", 0, 10, 1)
        pub_rec = st.number_input("Public Records", 0, 10, 0)
        grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"], index=2)
        grade_map = {g: i for i, g in enumerate("ABCDEFG")}
        grade_numeric = grade_map[grade]
        sub_grade_numeric = grade_numeric * 5 + 2

    if st.button("Assess Risk", type="primary", use_container_width=True):
        model = load_model()
        if model is None:
            st.error("Model not loaded. Please run the training pipeline first.")
            return

        inputs = {
            "annual_inc": annual_income, "loan_amnt": loan_amount, "dti": dti,
            "int_rate": int_rate, "installment": installment,
            "emp_length_years": emp_length, "revol_util": revol_util,
            "revol_bal": revol_bal, "open_acc": open_acc, "total_acc": total_acc,
            "delinq_2yrs": delinq_2yrs, "inq_last_6mths": inq_last_6mths,
            "pub_rec": pub_rec, "grade_numeric": grade_numeric,
            "sub_grade_numeric": sub_grade_numeric,
        }

        features_df = build_features_from_inputs(inputs)

        expected_features = get_expected_features(model)
        if expected_features is not None:
            for col in expected_features:
                if col not in features_df.columns:
                    features_df[col] = 0
            features_df = features_df[expected_features]

        pd_value = float(model.predict_proba(features_df)[:, 1][0])
        credit_score = pd_to_credit_score(pd_value)
        risk_tier = assign_risk_tier(credit_score)
        expected_loss = calculate_expected_loss(pd_value, loan_amount)

        st.divider()

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Default Probability", f"{pd_value * 100:.1f}%")
        r2.metric("Credit Score", f"{credit_score}")
        r3.metric("Risk Tier", risk_tier)
        r4.metric("Expected Loss", f"${expected_loss:,.0f}")

        tier_colors = {
            "Low Risk": "green", "Medium Risk": "orange", "High Risk": "red",
        }
        color = tier_colors.get(risk_tier, "gray")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=credit_score,
            title={"text": "Credit Score"},
            gauge={
                "axis": {"range": [300, 850]},
                "bar": {"color": color},
                "steps": [
                    {"range": [300, 670], "color": "#ffcccc"},
                    {"range": [670, 740], "color": "#ffffcc"},
                    {"range": [740, 850], "color": "#99ff99"},
                ],
            },
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        if risk_tier == "High Risk":
            st.error("**HIGH RISK (Subprime)** — Recommend Denial or High-Rate Pricing")
        elif risk_tier == "Medium Risk":
            st.warning("**MEDIUM RISK (Near-Prime)** — Standard Approval with Review")
        else:
            st.success("**LOW RISK (Prime)** — Recommend Approval at Best Rates")

        # Explanation section
        st.subheader("Risk Assessment Details")

        formula = f"PD ({pd_value:.2%}) x EAD (${loan_amount:,.0f}) x LGD ({LGD_DEFAULT:.0%}) = **${expected_loss:,.0f}**"
        st.markdown(f"**Expected Loss Formula:** {formula}")

        # Rule-based explanations
        risk_factors = []
        protective = []

        lti = loan_amount / max(annual_income, 1)
        grade_labels = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G"}
        gl = grade_labels.get(grade_numeric, "?")

        if grade_numeric >= 4:
            risk_factors.append(f"Loan grade {gl} — high risk tier")
        elif grade_numeric >= 2:
            risk_factors.append(f"Loan grade {gl} — moderate risk tier")
        else:
            protective.append(f"Loan grade {gl} — lower risk tier")

        if lti < 0.15:
            protective.append(f"Strong income-to-loan ratio ({lti:.0%})")
        elif lti > 0.5:
            risk_factors.append(f"High loan-to-income ratio ({lti:.0%})")

        if dti <= 12:
            protective.append(f"Low DTI ({dti:.0f}%)")
        elif dti >= 25:
            risk_factors.append(f"High DTI ({dti:.0f}%)")

        if int_rate <= 8:
            protective.append(f"Low interest rate ({int_rate:.1f}%)")
        elif int_rate >= 16:
            risk_factors.append(f"High interest rate ({int_rate:.1f}%)")

        if revol_util <= 30:
            protective.append(f"Low revolving utilization ({revol_util:.0f}%)")
        elif revol_util >= 65:
            risk_factors.append(f"High revolving utilization ({revol_util:.0f}%)")

        if delinq_2yrs == 0:
            protective.append("No delinquencies — clean payment history")
        else:
            risk_factors.append(f"{delinq_2yrs} delinquencies in past 2 years")

        if emp_length >= 7:
            protective.append(f"Stable employment ({emp_length} years)")
        elif emp_length <= 1:
            risk_factors.append(f"Short employment history ({emp_length} year)")

        if pub_rec > 0:
            risk_factors.append(f"{pub_rec} public record(s)")

        ex1, ex2 = st.columns(2)
        with ex1:
            st.markdown("**Risk Factors:**")
            if risk_factors:
                for f in risk_factors[:5]:
                    st.markdown(f"- {f}")
            else:
                st.markdown("- None identified")
        with ex2:
            st.markdown("**Protective Factors:**")
            if protective:
                for f in protective[:5]:
                    st.markdown(f"- {f}")
            else:
                st.markdown("- None identified")


def render_portfolio_page():
    st.header("Portfolio Analytics Dashboard")

    df = load_portfolio_data()
    if df is None:
        st.warning("No portfolio data available. Run the data pipeline first.")
        return

    model = load_model()
    if model is not None and "predicted_pd" not in df.columns:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        target_cols = ["is_default", "loan_status", "predicted_pd", "risk_score",
                       "risk_tier", "expected_loss"]
        feature_cols = [c for c in num_cols if c not in target_cols]

        expected = get_expected_features(model)
        if expected is None:
            expected = feature_cols
        X = df.reindex(columns=expected, fill_value=0)
        X = X.fillna(0)

        try:
            df["predicted_pd"] = model.predict_proba(X)[:, 1]
            df = score_portfolio(df, "predicted_pd")
        except Exception:
            pass

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Loans", f"{len(df):,}")
    if "is_default" in df.columns:
        k2.metric("Default Rate", f"{df['is_default'].mean() * 100:.1f}%")
    k3.metric("Avg Loan Amount", f"${df['loan_amnt'].mean():,.0f}")
    if "expected_loss" in df.columns:
        k4.metric("Total Expected Loss", f"${df['expected_loss'].sum():,.0f}")
    elif "is_default" in df.columns:
        k4.metric("Total Defaults", f"{df['is_default'].sum():,}")

    col1, col2 = st.columns(2)

    with col1:
        if "risk_tier" in df.columns:
            tier_counts = df["risk_tier"].value_counts()
            fig = px.pie(values=tier_counts.values, names=tier_counts.index,
                         title="Risk Tier Distribution",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)

        if "grade" in df.columns and "is_default" in df.columns:
            grade_default = df.groupby("grade")["is_default"].mean().reset_index()
            grade_default.columns = ["Grade", "Default Rate"]
            fig = px.bar(grade_default, x="Grade", y="Default Rate",
                         title="Default Rate by Loan Grade",
                         color="Default Rate", color_continuous_scale="RdYlGn_r")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "predicted_pd" in df.columns:
            fig = px.histogram(df, x="predicted_pd", nbins=50,
                               title="Default Probability Distribution",
                               labels={"predicted_pd": "Probability of Default"})
            st.plotly_chart(fig, use_container_width=True)

        if "expected_loss" in df.columns and "grade" in df.columns:
            el_by_grade = df.groupby("grade")["expected_loss"].sum().reset_index()
            el_by_grade.columns = ["Grade", "Total Expected Loss"]
            fig = px.bar(el_by_grade, x="Grade", y="Total Expected Loss",
                         title="Expected Loss by Grade",
                         color="Total Expected Loss", color_continuous_scale="Reds")
            st.plotly_chart(fig, use_container_width=True)

    if "purpose" in df.columns and "is_default" in df.columns:
        st.subheader("Default Rate by Loan Purpose")
        purpose_df = df.groupby("purpose").agg(
            count=("is_default", "size"),
            default_rate=("is_default", "mean"),
            avg_amount=("loan_amnt", "mean"),
        ).reset_index().sort_values("default_rate", ascending=False)
        purpose_df["default_rate"] = (purpose_df["default_rate"] * 100).round(2)
        purpose_df["avg_amount"] = purpose_df["avg_amount"].round(0)
        purpose_df.columns = ["Purpose", "Count", "Default Rate (%)", "Avg Amount ($)"]
        st.dataframe(purpose_df, use_container_width=True)


def main():
    st.sidebar.title("Credit Risk Platform")
    page = st.sidebar.radio("Navigation", ["Risk Assessment", "Portfolio Analytics"])

    if page == "Risk Assessment":
        render_prediction_page()
    else:
        render_portfolio_page()

    st.sidebar.divider()
    st.sidebar.caption("Credit Risk Analytics Platform v2.0")
    st.sidebar.caption("Calibrated CatBoost + SHAP + FastAPI")


if __name__ == "__main__":
    main()
