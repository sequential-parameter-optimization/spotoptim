# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression gate: verify that importing spotoptim and running a Kriging optimization
does NOT eagerly load torch, matplotlib, requests, statsmodels, or seaborn.

Runs the actual assertion in a subprocess so that the test environment's
pre-imported modules do not mask eager-import regressions (even though all
extras ARE installed in the dev environment).
"""

import subprocess
import sys

_SUBPROCESS_CODE = """
import sys

import spotoptim
from spotoptim import SpotOptim, Kriging
from spotoptim.function import sphere

# Run a tiny Kriging optimization using the real public API
result = SpotOptim(
    fun=sphere,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=4,
    n_initial=3,
    seed=42,
).optimize()

# Assert no heavy optional dependencies were pulled in
FORBIDDEN = ("torch", "matplotlib", "requests", "statsmodels", "seaborn")
leaked = [name for name in FORBIDDEN if name in sys.modules]
if leaked:
    loaded = sorted(k for k in sys.modules if any(k == f or k.startswith(f + ".") for f in FORBIDDEN))
    print("LEAKED MODULES:", loaded, file=sys.stderr)
    sys.exit(1)

print("lean import OK")
print(f"result.fun = {result.fun}")
"""


def test_core_import_is_lean():
    """Import spotoptim and run a kriging optimization in a clean subprocess.

    Asserts that none of torch, matplotlib, requests, statsmodels, or seaborn
    appear in sys.modules after the import and optimization run.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_CODE],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"Lean-import smoke test FAILED.\n"
        f"stdout: {proc.stdout}\n"
        f"stderr: {proc.stderr}"
    )
    assert (
        "lean import OK" in proc.stdout
    ), f"Expected 'lean import OK' in subprocess output, got:\n{proc.stdout}"
