"""Streamlit dashboard for credit scoring predictions."""

import streamlit as st
from pathlib import Path


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "baseline_model.joblib"


def load_model(path: Path):
    """Load a persisted model artifact."""
    import joblib

    return joblib.load(path)


def main():
    st.set_page_config(page_title="Credit Scoring Model", layout="wide")
    st.title("Credit Scoring Creditworthiness Prediction")

    st.sidebar.header("Applicant Profile")
    income = st.sidebar.number_input("Annual Income", min_value=0.0, value=50000.0, step=1000.0)
    debt = st.sidebar.number_input("Total Debt", min_value=0.0, value=10000.0, step=500.0)
    loan_amount = st.sidebar.number_input("Loan Amount", min_value=0.0, value=15000.0, step=500.0)
    employment_length = st.sidebar.number_input("Employment Length (years)", min_value=0, max_value=40, value=5)
    credit_limit = st.sidebar.number_input("Credit Limit", min_value=0.0, value=20000.0, step=500.0)
    credit_balance = st.sidebar.number_input("Credit Balance", min_value=0.0, value=5000.0, step=100.0)

    if st.sidebar.button("Predict Creditworthiness"):
        model = load_model(MODEL_PATH)
        features = [income, debt, loan_amount, employment_length, credit_limit, credit_balance]
        prediction = model.predict([features])[0]
        score = model.predict_proba([features])[0][1]

        st.metric("Creditworthy", "Yes" if prediction == 1 else "No", delta=f"Confidence: {score:.1%}")

        st.write("### Model output")
        st.write({"prediction": int(prediction), "confidence": float(score)})


if __name__ == "__main__":
    main()
