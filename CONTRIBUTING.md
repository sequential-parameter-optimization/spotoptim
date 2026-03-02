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

> **Optional**: activate for shorter commands in interactive shell sessions:
> ```sh
> source .venv/bin/activate   # then use python, pytest, ruff, … directly
> ```
> Not required when using `uv run <cmd>` or full paths like `.venv/bin/pytest`.

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

> **Optional**: activate for shorter commands in interactive shell sessions:
> ```sh
> source .venv-docs/bin/activate   # then use quartodoc, python, … directly
> ```
> Not required — calling `.venv-docs/bin/quartodoc` directly uses the correct
> interpreter and packages without any shell state changes.

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

## Keeping `_quarto.yml` in Sync

`_quarto.yml` is **hand-maintained** — it is not auto-generated from `src/`.
It has two distinct parts:

### 1. The `quartodoc:` sections block (manual)

This declares *which modules and symbols* appear in the API reference.
When you **add or rename a submodule** in `src/spotoptim/`, you must add it to the
relevant `sections: contents:` list in `_quarto.yml`. For example:

```yaml
# _quarto.yml (excerpt)
quartodoc:
  sections:
    - title: "My New Area"
      contents:
        - mymodule          # top-level module page
        - mymodule.submod   # submodule page
```

`generated_sections.yml` is a **manually-regenerated snapshot** — it is not read by
the build and is not auto-updated. To refresh it, run:
```sh
python check_links.py   # regenerates generated_sections.yml from hardcoded lists
```
Note that `check_links.py` has its own hardcoded copy of the sections list, so if you
add a module you must update **both** `_quarto.yml` **and** `check_links.py` if you
want `generated_sections.yml` to stay in sync. In practice you can safely ignore
`generated_sections.yml` — **`_quarto.yml` is the only file that matters for the build**.

### 2. The rendered `.qmd` stubs (automatic)

Once the sections config is correct, `quartodoc build` reads the docstrings from
`src/` and writes stub files like `docs/reference/mymodule.qmd`. These are
**gitignored** and rebuilt on every CI run — you never commit them manually.

### Update workflow

```
New/renamed module in src/spotoptim/
        │
        ▼
1. Edit _quarto.yml            ← MANUAL
   Add module to sections: contents:

        │
        ▼
2. .venv-docs/bin/quartodoc build --config _quarto.yml
   → writes/updates docs/reference/*.qmd stubs   ← AUTO

        │
        ▼
3. quarto render               ← AUTO
   → builds _site/ HTML from the .qmd stubs
```

### Summary

| File | Updated by | Used by build? |
|------|-----------|----------------|
| `_quarto.yml` | You, manually | ✅ Yes — the only file that matters |
| `generated_sections.yml` | Running `python check_links.py` manually | ❌ No — just a snapshot |
| `check_links.py` | You, manually | Only if you want to refresh the snapshot |
| `docs/reference/*.qmd` | `quartodoc build` automatically | ✅ Yes — but gitignored, never commit |

> **Only commit changes to `_quarto.yml`** (and `generated_sections.yml` if used).
> Never commit the generated `.qmd` stubs — they are gitignored and rebuilt by CI.

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
