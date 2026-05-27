
"""Create a simple model that works with the 6 features from Streamlit app."""
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Create a simple model pipeline (matches what train_baseline_model does)
simple_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(random_state=42, max_iter=1000))
])

# Train with dummy data (just to initialize the model properly)
# We'll use 6 features matching the Streamlit app's inputs:
# [income, debt, loan_amount, employment_length, credit_limit, credit_balance]
np.random.seed(42)
X_dummy = np.random.rand(100, 6)  # 100 samples, 6 features
y_dummy = np.random.randint(0, 2, size=100)  # binary labels

simple_pipeline.fit(X_dummy, y_dummy)

# Save the pipeline
joblib.dump(simple_pipeline, "models/baseline_model.joblib")
print("Simple model pipeline saved to models/baseline_model.joblib!")
