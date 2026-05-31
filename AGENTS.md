# AGENTS.md

Guidance for agents and humans working in this repository.

## Intent

This is a learning-first CNN and feature-map lab. Favor clarity, official PyTorch tutorial fidelity, visual inspection, and short reflection notes over clever abstractions or benchmark chasing.

## Rules

- Use `uv` for all Python commands.
- Target Windows native PowerShell commands in documentation and issues.
- Keep scripts small and readable.
- Add Python type hints to new or changed code.
- Prefer official sources: PyTorch, torchvision, and uv documentation.
- Do not turn this into a classifier-focused curriculum; classification is only a tiny vehicle for learning filters and feature maps.
- Save generated artifacts under `outputs/` and checkpoints under `models/`.
- Every learning issue should remain human-in-the-loop: inspect outputs, tweak one parameter, and write a short note.

## Quality checks

Run before committing code changes:

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest
```
