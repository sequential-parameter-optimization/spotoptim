# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the n_jobs=-1 convention (release 0.7.0, Improvement E)."""

import os
import pytest
import numpy as np
from spotoptim import SpotOptim


def sphere(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1)


BOUNDS = [(-5, 5), (-5, 5)]


class TestNJobsResolution:
    """Unit tests for n_jobs validation and -1 resolution."""

    def test_default_is_sequential(self):
        opt = SpotOptim(fun=sphere, bounds=BOUNDS, n_initial=3, max_iter=5)
        assert opt.n_jobs == 1

    def test_positive_integer_unchanged(self):
        opt = SpotOptim(fun=sphere, bounds=BOUNDS, n_initial=3, max_iter=5, n_jobs=2)
        assert opt.n_jobs == 2

    def test_minus_one_resolves_to_cpu_count(self):
        opt = SpotOptim(fun=sphere, bounds=BOUNDS, n_initial=3, max_iter=5, n_jobs=-1)
        expected = os.cpu_count() or 1
        assert opt.n_jobs == expected

    def test_minus_one_resolves_to_positive(self):
        opt = SpotOptim(fun=sphere, bounds=BOUNDS, n_initial=3, max_iter=5, n_jobs=-1)
        assert opt.n_jobs >= 1

    # --- invalid values raise ValueError ---

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="n_jobs"):
            SpotOptim(fun=sphere, bounds=BOUNDS, n_initial=3, max_iter=5, n_jobs=0)

    def test_minus_two_raises(self):
        with pytest.raises(ValueError, match="n_jobs"):
            SpotOptim(fun=sphere, bounds=BOUNDS, n_initial=3, max_iter=5, n_jobs=-2)

    def test_minus_ten_raises(self):
        with pytest.raises(ValueError, match="n_jobs"):
            SpotOptim(fun=sphere, bounds=BOUNDS, n_initial=3, max_iter=5, n_jobs=-10)

    def test_error_message_contains_value(self):
        with pytest.raises(ValueError, match="-5"):
            SpotOptim(fun=sphere, bounds=BOUNDS, n_initial=3, max_iter=5, n_jobs=-5)


class TestNJobsMinusOneOptimizes:
    """Smoke tests: n_jobs=-1 actually runs optimization end-to-end."""

    def test_sequential_fallback_single_core(self):
        """When os.cpu_count()==1 (or resolved to 1), n_jobs=-1 runs sequentially."""
        opt = SpotOptim(
            fun=sphere,
            bounds=BOUNDS,
            n_initial=5,
            max_iter=8,
            n_jobs=-1,
            seed=42,
        )
        # n_jobs must be a valid positive integer after resolution
        assert opt.n_jobs >= 1
        result = opt.optimize()
        assert result.success
        assert result.nfev >= 5

    def test_n_jobs_minus_one_same_result_structure_as_positive(self):
        """n_jobs=-1 produces the same result structure as n_jobs=1."""
        opt1 = SpotOptim(
            fun=sphere, bounds=BOUNDS, n_initial=5, max_iter=8, n_jobs=1, seed=0
        )
        opt_auto = SpotOptim(
            fun=sphere, bounds=BOUNDS, n_initial=5, max_iter=8, n_jobs=-1, seed=0
        )
        r1 = opt1.optimize()
        r_auto = opt_auto.optimize()

        assert r1.success
        assert r_auto.success
        assert r_auto.X.shape[1] == r1.X.shape[1]
        assert r_auto.fun < 100.0  # sanity: sphere minimum is 0


class TestNJobsStoredAfterResolution:
    """Verify the resolved value is what the optimizer actually uses."""

    def test_config_reflects_resolved_value(self):
        opt = SpotOptim(fun=sphere, bounds=BOUNDS, n_initial=3, max_iter=5, n_jobs=-1)
        # The config attribute must hold the resolved (positive) value
        assert opt.config.n_jobs == opt.n_jobs
        assert opt.config.n_jobs >= 1

    def test_large_positive_n_jobs_unchanged(self):
        opt = SpotOptim(fun=sphere, bounds=BOUNDS, n_initial=3, max_iter=5, n_jobs=4)
        assert opt.n_jobs == 4
        assert opt.config.n_jobs == 4
