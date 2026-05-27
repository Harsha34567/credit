"""Project configuration and global settings."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    """Immutable configuration values for the credit scoring model."""

    project_root: Path = Path(__file__).resolve().parents[2]
    data_root: Path = project_root / "data"
    raw_data_dir: Path = data_root / "raw"
    processed_data_dir: Path = data_root / "processed"
    external_data_dir: Path = data_root / "external"
    models_dir: Path = project_root / "models"
    logs_dir: Path = project_root / "logs"
    reports_dir: Path = project_root / "reports"
    app_dir: Path = project_root / "app"
    assets_dir: Path = project_root / "assets"
    notebook_dir: Path = project_root / "notebooks"
    random_seed: int = 42
    log_file: Path = logs_dir / "project.log"
