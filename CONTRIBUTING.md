# Contributing to spotoptim

Thank you for contributing! This guide covers the development setup, documentation workflow, and CI pipeline.

---

## Development Setup

```sh
# Clone and set up the project venv (Python 3.14, managed by uv)
git clone https://github.com/sequential-parameter-optimization/spotoptim.git
cd spotoptim
uv sync --all-extras --group dev
```

---

## Documentation Setup

The API docs use [Quarto](https://quarto.org) + [quartodoc](https://machow.github.io/quartodoc/).
A **separate Python 3.13 venv** is required because `quartodoc 0.11.x` is incompatible with Python 3.14.

```sh
# Create the docs venv (one-time setup — run once from the project root)
uv venv --python 3.13 .venv-docs
uv pip install --python .venv-docs/bin/python -r requirements-docs.txt
uv pip install --python .venv-docs/bin/python -e .   # make spotoptim importable during doc builds
```

---

## Documentation Workflow

After editing docstrings in `src/spotoptim/**`:

```sh
# 1. Regenerate API reference stubs from docstrings
.venv-docs/bin/quartodoc build --config _quarto.yml

# 2. (Optional) Preview rendered HTML locally
quarto render docs/reference/
# Output goes to _site/docs/reference/
```

> **No need to commit the generated `.qmd` files** — they are gitignored and rebuilt
> fresh by CI on every push. Only commit changes to `_quarto.yml` or docstrings.

---

## Pre-push Hook (Recommended)

Install a fast pre-push hook that validates docstrings parse correctly before every push:

```sh
cat > .git/hooks/pre-push << 'EOF'
#!/usr/bin/env bash
set -euo pipefail
VENV=".venv-docs"
if [[ ! -f "$VENV/bin/quartodoc" ]]; then
  echo "⚠️  pre-push: $VENV not found, skipping doc check."
  exit 0
fi
echo "🔍 pre-push: checking quartodoc can parse all docstrings..."
if "$VENV/bin/quartodoc" build --config _quarto.yml; then
  echo "✅ pre-push: quartodoc build OK"
else
  echo "❌ pre-push: quartodoc build FAILED — fix docstring errors before pushing"
  exit 1
fi
EOF
chmod +x .git/hooks/pre-push
```

To bypass the hook for a WIP commit: `git push --no-verify`

---

## CI Pipeline

| Workflow | Trigger | What it does |
|----------|---------|--------------|
| `ci.yml` | Push/PR to `main` or `develop` | Runs tests, linting, security scan |
| `docs.yml` | Push to `main` touching `src/**`, `docs/**`, or `_quarto.yml` | Runs quartodoc build → quarto render → deploys to GitHub Pages |

**Key point**: docs are only deployed on pushes to `main`. Working on `develop` does **not** trigger a deployment — preview locally with the workflow above.

---

## Docstring Style

Use **Google-style docstrings** (configured via `parser: google` in `_quarto.yml`):

```python
def my_function(x: np.ndarray, n: int = 10) -> float:
    """One-line summary.

    Args:
        x (np.ndarray): Input array, shape (n_samples, n_features).
        n (int, optional): Number of evaluations. Defaults to 10.

    Returns:
        float: The result.

    Examples:
        >>> import numpy as np
        >>> my_function(np.array([[0.5, 0.5]]))
        0.0
    """
```

---

## Branch Strategy

- **`develop`** — active development; PRs target this branch
- **`main`** — stable releases; merging here triggers docs deployment
