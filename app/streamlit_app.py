"""Streamlit dashboard for credit scoring predictions."""

from pathlib import Path

import joblib
import streamlit as st

from src.config.config import ProjectConfig
from src.feature_engineering.feature_engineering import build_feature_set


def load_model(path: Path):
    """Load a persisted model artifact."""
    return joblib.load(path)


def format_input_row(values: dict) -> list:
    """Create a model-ready feature row from user inputs."""
    return [
        values["income"],
        values["debt"],
        values["loan_amount"],
        values["employment_length"],
        values["credit_limit"],
        values["credit_balance"],
    ]


def main() -> None:
    config = ProjectConfig()
    model_path = config.model_artifact_path

    st.set_page_config(page_title="Credit Scoring Model", layout="wide")
    st.title("Credit Scoring Model")
    st.markdown("Use this dashboard to evaluate creditworthiness using financial ratios and model confidence.")

    st.sidebar.header("Applicant Profile")
    user_data = {
        "income": st.sidebar.number_input("Annual Income", min_value=0.0, value=50000.0, step=1000.0),
        "debt": st.sidebar.number_input("Total Debt", min_value=0.0, value=10000.0, step=500.0),
        "loan_amount": st.sidebar.number_input("Loan Amount", min_value=0.0, value=15000.0, step=500.0),
        "employment_length": st.sidebar.number_input("Employment Length (years)", min_value=0, max_value=40, value=5),
        "credit_limit": st.sidebar.number_input("Credit Limit", min_value=0.0, value=20000.0, step=500.0),
        "credit_balance": st.sidebar.number_input("Credit Balance", min_value=0.0, value=5000.0, step=100.0),
    }

    if not model_path.exists():
        st.error(f"Model artifact not found at {model_path}. Train a model first.")
        return

    if st.sidebar.button("Predict Creditworthiness"):
        model = load_model(model_path)
        feature_row = format_input_row(user_data)
        prediction = model.predict([feature_row])[0]
        score = float(model.predict_proba([feature_row])[0][1])

        st.metric("Creditworthy", "Yes" if prediction == 1 else "No", delta=f"Confidence: {score:.1%}")
        st.write("### Engineered Features")
        display_data = build_feature_set(
            st.session_state.get("raw", None) or {
                "income": [user_data["income"]],
                "debt": [user_data["debt"]],
                "loan_amount": [user_data["loan_amount"]],
                "employment_length": [user_data["employment_length"]],
                "credit_limit": [user_data["credit_limit"]],
                "credit_balance": [user_data["credit_balance"]],
            }
        )
        st.dataframe(display_data)

        st.write("### Model output")
        st.json({"prediction": int(prediction), "confidence": score})


if __name__ == "__main__":
    main()
