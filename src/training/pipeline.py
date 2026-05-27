"""Training pipeline orchestration for the credit scoring project."""

import json
from pathlib import Path
from shutil import copy2
from typing import Tuple

import pandas as pd

from src.config.config import ProjectConfig
from src.feature_engineering.feature_engineering import build_feature_set
from src.preprocessing.pipeline import CreditScoringPreprocessor
from src.preprocessing.preprocess import load_raw_data, save_processed_data
from src.training.models import train_all_models
from src.utils.kaggle_utils import download_kaggle_dataset
from src.utils.logger import initialize_logger


def _normalize_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """Map raw dataset columns to standard feature names when possible."""
    mapping = {
        "AMT_INCOME_TOTAL": "income",
        "AMT_CREDIT": "debt",
        "AMT_ANNUITY": "loan_amount",
        "AMT_GOODS_PRICE": "goods_price",
    }

    data = data.copy()
    for source, target in mapping.items():
        if source in data.columns and target not in data.columns:
            data[target] = data[source]

    return data


def _infer_feature_lists(data: pd.DataFrame, target_column: str) -> Tuple[list[str], list[str]]:
    """Infer numeric and categorical feature columns from a processed DataFrame."""
    feature_data = data.drop(columns=[target_column], errors="ignore")
    numeric_features = feature_data.select_dtypes(include="number").columns.tolist()
    categorical_features = feature_data.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return numeric_features, categorical_features


def ensure_raw_data(config: ProjectConfig, force_download: bool = False) -> None:
    """Ensure the raw dataset exists locally, downloading from Kaggle when needed."""
    if config.raw_dataset_path.exists() and not force_download:
        return

    download_kaggle_dataset(config.kaggle_dataset_name, config.raw_data_dir, force=force_download)


def prepare_training_dataframe(raw_df: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    """Create a processed training DataFrame from raw input data."""
    raw_df = _normalize_column_names(raw_df)
    processed_df = build_feature_set(raw_df)

    if config.target_column not in processed_df.columns:
        raise ValueError(
            f"Target column '{config.target_column}' was not found in the processed dataset."
        )

    return processed_df


def run_training_pipeline(
    config: ProjectConfig,
    force_download: bool = False,
    save_processed: bool = True,
) -> dict:
    """Run the full data preparation and training pipeline."""
    logger = initialize_logger(config.log_file)
    logger.info("Starting training pipeline.")

    ensure_raw_data(config, force_download=force_download)
    raw_df = load_raw_data(config.raw_dataset_path)
    processed_df = prepare_training_dataframe(raw_df, config)

    if save_processed:
        processed_path = config.processed_data_dir / "processed_training.csv"
        save_processed_data(processed_df, processed_path)
        logger.info(f"Saved processed training data to {processed_path}.")

    numeric_features, categorical_features = _infer_feature_lists(processed_df, config.target_column)
    logger.info("Inferred %d numeric and %d categorical features.", len(numeric_features), len(categorical_features))

    preprocessor = CreditScoringPreprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        target_column=config.target_column,
    )

    X = preprocessor.fit_transform(processed_df.drop(columns=[config.target_column]))
    y = processed_df[config.target_column]

    metrics = train_all_models(pd.DataFrame(X), y, config.models_dir)
    logger.info("Completed model training for %d models.", len(metrics))

    metrics_path = config.reports_dir / "model_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info(f"Saved model metrics to {metrics_path}.")

    best_model_name = max(metrics, key=lambda key: metrics[key].get("roc_auc", 0.0))
    best_model_path = config.models_dir / f"{best_model_name}.joblib"
    final_model_path = config.model_artifact_path
    copy2(best_model_path, final_model_path)
    logger.info(f"Best model '{best_model_name}' saved as {final_model_path}.")

    return {
        "best_model": best_model_name,
        "metrics_path": str(metrics_path),
        "saved_model": str(final_model_path),
    }
