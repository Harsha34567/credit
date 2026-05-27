"""Scalable preprocessing pipeline for credit scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class CreditScoringPreprocessor:
    numeric_features: List[str]
    categorical_features: List[str]
    target_column: str
    pipeline: Optional[ColumnTransformer] = field(init=False, default=None)

    def build_pipeline(self) -> ColumnTransformer:
        """Build a reusable sklearn preprocessing pipeline."""
        numeric_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )

        return ColumnTransformer(
            [
                ("numeric", numeric_pipeline, self.numeric_features),
                ("categorical", categorical_pipeline, self.categorical_features),
            ],
            remainder="drop",
        )

    def fit(self, X: pd.DataFrame) -> np.ndarray:
        """Fit the preprocessing pipeline to the training data."""
        self.pipeline = self.build_pipeline()
        return self.pipeline.fit_transform(X)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features using the fitted preprocessing pipeline."""
        if self.pipeline is None:
            raise RuntimeError("The preprocessing pipeline must be fitted before transform.")
        return self.pipeline.transform(X)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit and transform the training data."""
        return self.fit(X)
