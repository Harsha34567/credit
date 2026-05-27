"""Kaggle dataset download utilities for secure data access."""

from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import os


load_dotenv()

KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")
KAGGLE_API_TOKEN = os.getenv("KAGGLE_API_TOKEN")


def _ensure_token_file(token: str) -> None:
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    token_path = kaggle_dir / "access_token"
    token_path.write_text(token)
    try:
        token_path.chmod(0o600)
    except OSError:
        pass
    os.environ["KAGGLE_CONFIG_DIR"] = str(kaggle_dir)


def download_kaggle_dataset(dataset: str, output_dir: Path, force: bool = False) -> None:
    """Download a Kaggle dataset to the project data folder."""
    if KAGGLE_API_TOKEN:
        _ensure_token_file(KAGGLE_API_TOKEN)
    elif KAGGLE_USERNAME and KAGGLE_KEY:
        os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
        os.environ["KAGGLE_KEY"] = KAGGLE_KEY
    else:
        raise EnvironmentError(
            "KAGGLE_API_TOKEN or KAGGLE_USERNAME/KAGGLE_KEY must be defined in the environment or .env file."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    if force or not any(output_dir.iterdir()):
        api.dataset_download_files(dataset, path=str(output_dir), unzip=True, quiet=False)
    else:
        print(f"Dataset already exists at {output_dir}. Use force=True to re-download.")
