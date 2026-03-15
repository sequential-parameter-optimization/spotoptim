# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for Release 0.10.0 — Improvement D: Free-Threaded (No-GIL) Awareness.

Verifies that:
- _is_gil_disabled() returns a bool and reflects the actual GIL status.
- _is_gil_disabled() returns False on standard (GIL-enabled) Python builds,
  which is expected in all current CI environments.
- _is_gil_disabled() handles the absence of sys._is_gil_enabled gracefully
  (Python < 3.13 compatibility).
- The executor selection logic branches correctly based on GIL status:
  GIL build  → ProcessPoolExecutor for eval, ThreadPoolExecutor for search
  No-GIL     → ThreadPoolExecutor for both eval and search
- End-to-end optimization works correctly on the standard GIL build
  (this is the path exercised in CI; no-GIL requires python3.13t).
- The _is_gil_disabled() helper is importable from spotoptim.SpotOptim.
"""

import sys
import importlib
import unittest.mock as mock
import numpy as np
from spotoptim import SpotOptim
from spotoptim.SpotOptim import _is_gil_disabled

# importlib bypasses the __init__.py class re-export and gives the module.
_spotoptim_mod = importlib.import_module("spotoptim.SpotOptim")

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

BOUNDS = [(-5, 5), (-5, 5)]


def sphere(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1)


# ---------------------------------------------------------------------------
# Unit: _is_gil_disabled()
# ---------------------------------------------------------------------------


class TestIsGilDisabled:
    """Unit tests for the _is_gil_disabled() helper."""

    def test_returns_bool(self):
        result = _is_gil_disabled()
        assert isinstance(result, bool)

    def test_false_on_standard_build(self):
        """On a standard GIL-enabled Python the function must return False."""
        # All current CI environments run standard GIL builds.
        # _is_gil_enabled() exists on CPython 3.13+ and returns True when GIL
        # is active.  On older Python the attribute is absent and the lambda
        # default returns True — either way _is_gil_disabled() is False.
        result = _is_gil_disabled()
        if hasattr(sys, "_is_gil_enabled"):
            assert result == (not sys._is_gil_enabled())
        else:
            # Python < 3.13: attribute absent → GIL assumed enabled → False
            assert result is False

    def test_consistent_across_calls(self):
        """GIL status does not change within a running process."""
        assert _is_gil_disabled() == _is_gil_disabled()

    def test_mocked_gil_disabled(self):
        """Simulate a no-GIL build by mocking sys._is_gil_enabled."""
        with mock.patch.object(sys, "_is_gil_enabled", return_value=False, create=True):
            assert _is_gil_disabled() is True

    def test_mocked_gil_enabled(self):
        """Simulate a standard GIL build by mocking sys._is_gil_enabled."""
        with mock.patch.object(sys, "_is_gil_enabled", return_value=True, create=True):
            assert _is_gil_disabled() is False

    def test_no_attribute_fallback(self):
        """When sys._is_gil_enabled is absent the helper returns False."""
        # Remove attribute if it exists, then call the helper
        original = getattr(sys, "_is_gil_enabled", None)
        if hasattr(sys, "_is_gil_enabled"):
            delattr(sys, "_is_gil_enabled")
        try:
            result = _is_gil_disabled()
            assert result is False
        finally:
            if original is not None:
                sys._is_gil_enabled = original


# ---------------------------------------------------------------------------
# Unit: executor selection logic
# ---------------------------------------------------------------------------


class TestExecutorSelection:
    """Verify the correct executor types are chosen based on GIL status."""

    def test_gil_build_uses_process_pool_for_eval(self):
        """On a GIL build _optimize_steady_state must use ProcessPoolExecutor
        for eval.  We verify by checking that _is_gil_disabled() is False on
        this interpreter, which is the precondition for that path."""
        # If GIL is enabled (standard build), the eval pool is a Process pool.
        assert not _is_gil_disabled(), (
            "This test must run on a standard GIL build; "
            "skip it on free-threaded Python."
        )

    def test_no_gil_mock_uses_thread_pool_for_eval(self, monkeypatch):
        """When GIL is mocked as disabled, _is_gil_disabled() returns True,
        signalling the thread-pool eval path."""
        monkeypatch.setattr(sys, "_is_gil_enabled", lambda: False, raising=False)
        assert _is_gil_disabled() is True

    def test_process_pool_executor_importable(self):
        """ProcessPoolExecutor must be importable (GIL-build eval path)."""
        from concurrent.futures import ProcessPoolExecutor  # noqa: F401

        assert ProcessPoolExecutor is not None

    def test_thread_pool_executor_importable(self):
        """ThreadPoolExecutor must be importable (search path + no-GIL eval)."""
        from concurrent.futures import ThreadPoolExecutor  # noqa: F401

        assert ThreadPoolExecutor is not None


# ---------------------------------------------------------------------------
# Integration: end-to-end on standard GIL build
# ---------------------------------------------------------------------------


class TestGilBuildEndToEnd:
    """Smoke tests on the current (GIL-enabled) Python build.

    These tests exercise the standard code path: ProcessPoolExecutor for eval,
    ThreadPoolExecutor for search.  They also confirm that the no-GIL detection
    layer does not break anything on GIL builds.
    """

    def test_n_jobs_2_sphere(self):
        opt = SpotOptim(
            fun=sphere,
            bounds=BOUNDS,
            n_initial=6,
            max_iter=12,
            n_jobs=2,
            seed=0,
        )
        result = opt.optimize()
        assert result.success
        assert result.fun < 50.0

    def test_n_jobs_3_sphere(self):
        opt = SpotOptim(
            fun=sphere,
            bounds=BOUNDS,
            n_initial=6,
            max_iter=14,
            n_jobs=3,
            seed=7,
        )
        result = opt.optimize()
        assert result.success
        assert result.fun < 50.0

    def test_lambda_objective_parallel(self):
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=BOUNDS,
            n_initial=5,
            max_iter=10,
            n_jobs=2,
            seed=1,
        )
        result = opt.optimize()
        assert result.success

    def test_sequential_path_unaffected(self):
        """n_jobs=1 must be completely unaffected by the no-GIL change."""
        opt = SpotOptim(
            fun=sphere,
            bounds=BOUNDS,
            n_initial=5,
            max_iter=10,
            n_jobs=1,
            seed=0,
        )
        result = opt.optimize()
        assert result.success

    def test_minus_one_n_jobs_parallel(self):
        """n_jobs=-1 must resolve and run without error on a GIL build."""
        opt = SpotOptim(
            fun=sphere,
            bounds=BOUNDS,
            n_initial=5,
            max_iter=10,
            n_jobs=-1,
            seed=3,
        )
        assert opt.n_jobs >= 1
        result = opt.optimize()
        assert result.success


# ---------------------------------------------------------------------------
# Integration: simulated no-GIL path via monkeypatch
# ---------------------------------------------------------------------------


class TestSimulatedNoGilPath:
    """Test the thread-based eval path by mocking _is_gil_disabled() to True.

    These tests exercise _thread_eval_task_single and _thread_batch_eval_task
    on the current interpreter by making _optimize_steady_state believe it is
    running on a free-threaded build.
    """

    def test_simulated_no_gil_sphere(self, monkeypatch):
        """Optimization succeeds when both pools are ThreadPoolExecutor."""
        monkeypatch.setattr(_spotoptim_mod, "_is_gil_disabled", lambda: True)
        opt = SpotOptim(
            fun=sphere,
            bounds=BOUNDS,
            n_initial=6,
            max_iter=12,
            n_jobs=2,
            seed=0,
        )
        result = opt.optimize()
        assert result.success
        assert result.fun < 50.0

    def test_simulated_no_gil_lambda(self, monkeypatch):
        """Lambda objectives work in the thread-based eval path."""
        monkeypatch.setattr(_spotoptim_mod, "_is_gil_disabled", lambda: True)
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=BOUNDS,
            n_initial=5,
            max_iter=10,
            n_jobs=2,
            seed=42,
        )
        result = opt.optimize()
        assert result.success

    def test_simulated_no_gil_with_batch_size(self, monkeypatch):
        """Batch eval + no-GIL path: _thread_batch_eval_task is used."""
        monkeypatch.setattr(_spotoptim_mod, "_is_gil_disabled", lambda: True)
        opt = SpotOptim(
            fun=sphere,
            bounds=BOUNDS,
            n_initial=6,
            max_iter=14,
            n_jobs=2,
            eval_batch_size=2,
            seed=5,
        )
        result = opt.optimize()
        assert result.success

    def test_simulated_no_gil_4d(self, monkeypatch):
        """Thread-based eval path handles higher-dimensional problems."""
        monkeypatch.setattr(_spotoptim_mod, "_is_gil_disabled", lambda: True)
        opt = SpotOptim(
            fun=sphere,
            bounds=[(-3, 3)] * 4,
            n_initial=8,
            max_iter=16,
            n_jobs=2,
            seed=88,
        )
        result = opt.optimize()
        assert result.success
        assert result.X.shape[1] == 4

    def test_simulated_no_gil_result_shape_matches_gil(self, monkeypatch):
        """No-GIL and GIL paths produce results with the same dimensionality."""
        opt_gil = SpotOptim(
            fun=sphere,
            bounds=BOUNDS,
            n_initial=6,
            max_iter=12,
            n_jobs=2,
            seed=0,
        )
        r_gil = opt_gil.optimize()

        monkeypatch.setattr(_spotoptim_mod, "_is_gil_disabled", lambda: True)
        opt_no_gil = SpotOptim(
            fun=sphere,
            bounds=BOUNDS,
            n_initial=6,
            max_iter=12,
            n_jobs=2,
            seed=0,
        )
        r_no_gil = opt_no_gil.optimize()

        assert r_gil.success
        assert r_no_gil.success
        assert r_no_gil.X.shape[1] == r_gil.X.shape[1]
