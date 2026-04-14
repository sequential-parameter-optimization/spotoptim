# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resume a spotoptim experiment from a pickle file and save the result.

Usage:
    python run_spotoptim.py <prefix>_exp.pkl

Loads the experiment with :py:meth:`SpotOptim.load_experiment`, runs
:py:meth:`SpotOptim.optimize` (parallel if ``n_jobs > 1`` was set when the
experiment was built), then writes ``<prefix>_res.pkl`` next to the input.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from spotoptim import SpotOptim


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("exp_pkl", type=Path, help="Path to <prefix>_exp.pkl")
    args = p.parse_args()

    exp_path: Path = args.exp_pkl.resolve()
    if not exp_path.name.endswith("_exp.pkl"):
        raise SystemExit(f"Expected <prefix>_exp.pkl, got {exp_path.name}")
    prefix = exp_path.name.removesuffix("_exp.pkl")

    opt = SpotOptim.load_experiment(str(exp_path))
    result = opt.optimize()
    opt.save_result(prefix=prefix, path=str(exp_path.parent))

    print(f"nfev={result.nfev}  fun={result.fun:.6g}  x={result.x}")


if __name__ == "__main__":
    main()
