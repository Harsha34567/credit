"""Basic tests to validate project scaffolding."""

from src.config.config import ProjectConfig


def test_project_config_paths() -> None:
    config = ProjectConfig()
    assert config.project_root.exists()
    assert config.data_root.name == "data"
