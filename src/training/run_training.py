"""Command-line entrypoint for training the credit scoring pipeline."""

import argparse

from src.config.config import ProjectConfig
from src.training.pipeline import run_training_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the credit scoring training pipeline.")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force redownload of the Kaggle dataset even if raw files already exist.",
    )
    parser.add_argument(
        "--skip-save-processed",
        action="store_true",
        help="Do not save the processed training dataset to disk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ProjectConfig()
    training_summary = run_training_pipeline(
        config,
        force_download=args.force_download,
        save_processed=not args.skip_save_processed,
    )

    print("Training completed.")
    print(training_summary)


if __name__ == "__main__":
    main()
