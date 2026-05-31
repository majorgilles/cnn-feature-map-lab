"""Path helpers for notebooks and scripts."""

from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Find the repository root by walking upward until `pyproject.toml` is found."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not find project root containing pyproject.toml")


def project_path(*parts: str, start: Path | None = None) -> Path:
    """Build a path relative to the project root."""
    return find_project_root(start=start).joinpath(*parts)
