from pathlib import Path

from cnn_feature_map_lab.paths import find_project_root, project_path


def test_find_project_root_from_tests_dir() -> None:
    root = find_project_root(Path(__file__).parent)
    assert (root / "pyproject.toml").exists()


def test_project_path_builds_relative_path() -> None:
    path = project_path("notebooks", start=Path(__file__).parent)
    assert path.name == "notebooks"
