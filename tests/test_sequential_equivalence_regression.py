# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression test: the sequential engine is deterministic and converges.

Guards the single sequential optimization engine (after the parallel subsystem
was removed) against behavioural regressions, using **platform-portable**
invariants:

1. Same-seed determinism — two runs with the same seed are bit-for-bit identical
   *within a platform* (this is the property the parallelism removal had to
   preserve, and it is checked exactly with ``assert_array_equal``).
2. Evaluation budget — ``nfev`` / ``nit`` / ``success`` are budget-controlled and
   therefore platform-independent; checked exactly against the captured golden.
3. Convergence quality — the best value stays in the same ballpark as the golden.

Note on byte-identity across platforms: seeded SpotOptim results are bit-identical
only on the *same* platform/BLAS. Across macOS<->Linux the iterative surrogate
trajectory amplifies floating-point rounding into different (but equally valid)
optima, so the per-coordinate ``best_x`` / history are intentionally **not**
asserted bit-exact against a fixture captured on one machine. The golden fixture
(``fixtures/sequential_golden.json``) supplies the case definitions and the
budget/quality references.
"""

import json
import pathlib

import numpy as np
import pytest

from spotoptim import SpotOptim
from spotoptim.function.so import sphere, rosenbrock

_FIXTURE_PATH = pathlib.Path(__file__).parent / "fixtures" / "sequential_golden.json"

# Mapping from the string stored in the fixture to the actual function object.
_FUN_MAP = {
    "sphere": sphere,
    "rosenbrock": rosenbrock,
}


def _load_cases():
    """Return list of (case_id, kwargs, expected_result) tuples."""
    with _FIXTURE_PATH.open() as fh:
        data = json.load(fh)
    cases = []
    for case_id, entry in data.items():
        raw_kwargs = entry["kwargs"]
        kwargs = dict(raw_kwargs)
        kwargs["fun"] = _FUN_MAP[raw_kwargs["fun"]]
        # bounds stored as list-of-lists → list of tuples
        kwargs["bounds"] = [tuple(b) for b in raw_kwargs["bounds"]]
        cases.append((case_id, kwargs, entry["result"]))
    return cases


_CASES = _load_cases()


@pytest.mark.parametrize("case_id,kwargs,expected", _CASES, ids=[c[0] for c in _CASES])
def test_sequential_equivalence(case_id, kwargs, expected):
    """Sequential engine is deterministic, budget-correct, and converges."""
    r1 = SpotOptim(**kwargs).optimize()
    r2 = SpotOptim(**kwargs).optimize()

    # 1. Same-seed determinism — bit-identical within a platform.
    np.testing.assert_array_equal(
        np.asarray(r1.X), np.asarray(r2.X), err_msg=f"[{case_id}] non-deterministic X"
    )
    np.testing.assert_array_equal(
        np.asarray(r1.y), np.asarray(r2.y), err_msg=f"[{case_id}] non-deterministic y"
    )
    np.testing.assert_array_equal(
        np.asarray(r1.x), np.asarray(r2.x), err_msg=f"[{case_id}] non-deterministic x"
    )
    assert r1.fun == r2.fun, f"[{case_id}] non-deterministic fun"

    # 2. Evaluation budget is exact and platform-independent.
    assert (
        r1.nfev == expected["nfev"]
    ), f"[{case_id}] nfev {r1.nfev} != {expected['nfev']}"
    assert r1.nit == expected["nit"], f"[{case_id}] nit {r1.nit} != {expected['nit']}"
    assert (
        r1.success == expected["success"]
    ), f"[{case_id}] success {r1.success} != {expected['success']}"

    # 3. Convergence quality stays in the golden ballpark (generous tolerance
    #    absorbs cross-platform floating-point trajectory divergence; a real
    #    regression that fails to converge would be orders of magnitude worse).
    assert (
        r1.fun <= expected["fun"] * 10.0 + 1e-3
    ), f"[{case_id}] fun {r1.fun} regressed vs golden {expected['fun']}"
