"""Feature engineering utilities for credit scoring."""

from __future__ import annotations

from typing import Optional

import pandas as pd


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.replace(0, 1)).fillna(0)


def _map_known_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Map raw dataset columns to standard feature names when available."""
    column_mapping = {
        "AMT_INCOME_TOTAL": "income",
        "AMT_CREDIT": "debt",
        "AMT_ANNUITY": "loan_amount",
        "AMT_GOODS_PRICE": "goods_price",
        "AMT_CREDIT_SUM": "credit_balance",
        "AMT_CREDIT_SUM_LIMIT": "credit_limit",
    }

    data = data.copy()
    for original_name, standardized_name in column_mapping.items():
        if original_name in data.columns and standardized_name not in data.columns:
            data[standardized_name] = data[original_name]

    return data


def add_financial_ratios(data: pd.DataFrame) -> pd.DataFrame:
    """Create industry-relevant financial ratio features."""
    data = data.copy()

    if "income" in data.columns and "debt" in data.columns:
        data["debt_to_income_ratio"] = _safe_divide(data["debt"], data["income"])

    if "credit_limit" in data.columns and "credit_balance" in data.columns:
        data["credit_utilization_ratio"] = _safe_divide(
            data["credit_balance"], data["credit_limit"]
        )

    if "loan_amount" in data.columns and "income" in data.columns:
        data["loan_income_ratio"] = _safe_divide(data["loan_amount"], data["income"])

    return data


def add_payment_behavior_score(data: pd.DataFrame) -> pd.DataFrame:
    """Add behavior score based on repayment and delinquency features."""
    data = data.copy()

    score = pd.Series(0.0, index=data.index)

    if "payment_history_12m" in data.columns:
        score += data["payment_history_12m"].fillna(0) / 100

    if "missed_payments" in data.columns:
        score -= _safe_divide(data["missed_payments"], data.get("loan_amount", pd.Series(1, index=data.index)))

    if "days_since_last_payment" in data.columns:
        score -= _safe_divide(data["days_since_last_payment"], pd.Series(365.0, index=data.index))

    data["payment_behavior_score"] = score.clip(-1, 1)
    return data


def add_financial_risk_score(data: pd.DataFrame) -> pd.DataFrame:
    """Combine financial ratios into a single risk score feature."""
    data = data.copy()

    if "debt_to_income_ratio" not in data.columns:
        data = add_financial_ratios(data)

    score = pd.Series(0.0, index=data.index)
    if "debt_to_income_ratio" in data.columns:
        score += _safe_divide(data["debt_to_income_ratio"], pd.Series(10.0, index=data.index))
    if "credit_utilization_ratio" in data.columns:
        score += _safe_divide(data["credit_utilization_ratio"], pd.Series(2.0, index=data.index))
    if "loan_income_ratio" in data.columns:
        score += _safe_divide(data["loan_income_ratio"], pd.Series(5.0, index=data.index))
    if "payment_behavior_score" in data.columns:
        score -= data["payment_behavior_score"]

    data["financial_risk_score"] = score.clip(0, 1)
    return data


def build_feature_set(data: pd.DataFrame | dict) -> pd.DataFrame:
    """Build all engineered features for model training and inference."""
    if isinstance(data, dict):
        data = pd.DataFrame(data)

    data = _map_known_columns(data)
    data = add_financial_ratios(data)
    data = add_payment_behavior_score(data)
    data = add_financial_risk_score(data)
    return data
