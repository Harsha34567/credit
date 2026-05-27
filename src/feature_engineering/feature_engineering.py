"""Feature engineering utilities for credit scoring."""

import pandas as pd


def add_financial_ratios(data: pd.DataFrame) -> pd.DataFrame:
    """Create industry-relevant financial ratio features."""
    data = data.copy()

    if "income" in data.columns and "debt" in data.columns:
        data["debt_to_income_ratio"] = data["debt"] / data["income"].replace(0, 1)

    if "credit_limit" in data.columns and "credit_balance" in data.columns:
        data["credit_utilization_ratio"] = (
            data["credit_balance"] / data["credit_limit"].replace(0, 1)
        )

    if "loan_amount" in data.columns and "income" in data.columns:
        data["loan_income_ratio"] = data["loan_amount"] / data["income"].replace(0, 1)

    return data
