# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression test: sequential engine byte-identity after parallelism removal.

Loads the pre-change golden fixture captured on the unchanged codebase and
verifies that every stored case reproduces bit-for-bit identical results after
the parallel subsystem has been removed.  Uses ``assert_array_equal`` (not
``allclose``) to enforce byte-identity.  If this test fails, STOP — do not
weaken the assertion.
"""

import json
import pathlib

import numpy as np
import pytest

from spotoptim import SpotOptim
from spotoptim.function.so import sphere, rosenbrock  # noqa: F401

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
    """Sequential engine reproduces the pre-change golden results exactly."""
    opt = SpotOptim(**kwargs)
    result = opt.optimize()

    np.testing.assert_array_equal(
        result.x,
        expected["x"],
        err_msg=f"[{case_id}] result.x mismatch",
    )
    np.testing.assert_array_equal(
        np.array(result.X),
        np.array(expected["X"]),
        err_msg=f"[{case_id}] result.X mismatch",
    )
    np.testing.assert_array_equal(
        result.y,
        expected["y"],
        err_msg=f"[{case_id}] result.y mismatch",
    )
    assert (
        result.fun == expected["fun"]
    ), f"[{case_id}] result.fun mismatch: {result.fun} != {expected['fun']}"
    assert (
        result.nfev == expected["nfev"]
    ), f"[{case_id}] result.nfev mismatch: {result.nfev} != {expected['nfev']}"
    assert (
        result.nit == expected["nit"]
    ), f"[{case_id}] result.nit mismatch: {result.nit} != {expected['nit']}"
    assert (
        result.success == expected["success"]
    ), f"[{case_id}] result.success mismatch: {result.success} != {expected['success']}"
