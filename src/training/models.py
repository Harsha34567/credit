"""Model training and comparison utilities for credit scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None

from src.evaluation.metrics import compute_classification_metrics
from src.utils.seed import set_random_seed


def get_model_candidates(random_seed: int = 42) -> Dict[str, object]:
    """Return a dictionary of candidate model estimators."""
    models: Dict[str, object] = {
        "logistic_regression": LogisticRegression(random_state=random_seed, max_iter=1000),
        "decision_tree": DecisionTreeClassifier(random_state=random_seed, max_depth=6),
        "random_forest": RandomForestClassifier(random_state=random_seed, n_estimators=200),
    }

    if xgb is not None:
        models["xgboost"] = xgb.XGBClassifier(
            random_state=random_seed,
            use_label_encoder=False,
            eval_metric="logloss",
            n_jobs=-1,
        )

    if lgb is not None:
        models["lightgbm"] = lgb.LGBMClassifier(random_state=random_seed, n_jobs=-1)

    return models


def train_model(
    name: str,
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> Tuple[Pipeline, dict]:
    """Train a classification model, save it, and return metrics."""
    set_random_seed(random_seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )

    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    y_prob = estimator.predict_proba(X_test)[:, 1]

    metrics = compute_classification_metrics(y_test, y_pred, y_prob)

    model_path = output_dir / f"{name}.joblib"
    joblib.dump(estimator, model_path)

    return estimator, metrics


def train_all_models(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
    random_seed: int = 42,
) -> Dict[str, dict]:
    """Train all candidate models and return evaluation metrics."""
    models = get_model_candidates(random_seed=random_seed)
    results: Dict[str, dict] = {}

    for name, estimator in models.items():
        trained, metrics = train_model(name, estimator, X, y, output_dir, random_seed=random_seed)
        results[name] = metrics

    return results
