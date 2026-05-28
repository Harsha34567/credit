
"""Create a simple model that works with the 6 features from Streamlit app."""
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_synthetic_data(n_samples: int = 5000, random_state: int = 42):
    """Generate realistic synthetic applicant data for training."""
    rng = np.random.default_rng(random_state)
    income = rng.normal(70000, 25000, n_samples).clip(5000, 250000)
    debt = rng.normal(15000, 15000, n_samples).clip(0, 150000)
    loan_amount = rng.normal(12000, 10000, n_samples).clip(0, 150000)
    employment_length = rng.integers(0, 41, n_samples)
    credit_limit = rng.normal(25000, 18000, n_samples).clip(1000, 250000)
    credit_balance = rng.normal(6000, 8000, n_samples).clip(0, 150000)

    X = np.column_stack([
        income,
        debt,
        loan_amount,
        employment_length,
        credit_limit,
        credit_balance,
    ])

    debt_to_income = debt / np.maximum(income, 1)
    utilization = credit_balance / np.maximum(credit_limit, 1)
    loan_to_income = loan_amount / np.maximum(income, 1)

    y = (
        (debt_to_income < 0.4)
        & (utilization < 0.35)
        & (loan_to_income < 0.3)
        & (employment_length >= 2)
    ).astype(int)

    # Introduce a small amount of label noise for realism.
    flip_mask = rng.random(n_samples) < 0.05
    y = np.where(flip_mask, 1 - y, y)
    return X, y


simple_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
])

X_train, y_train = make_synthetic_data()
simple_pipeline.fit(X_train, y_train)

joblib.dump(simple_pipeline, "models/baseline_model.joblib")
print("Simple model pipeline saved to models/baseline_model.joblib!")
