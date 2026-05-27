"""Entry point for the credit scoring model project."""

from src.config.config import ProjectConfig
from src.utils.logger import initialize_logger


def main() -> None:
    """Run a starter validation flow for the project setup."""
    config = ProjectConfig()
    logger = initialize_logger(config.log_file)

    logger.info("Credit scoring model project initialized successfully.")
    print("Project scaffold is ready. Activate the virtual environment and install dependencies.")


if __name__ == "__main__":
    main()
