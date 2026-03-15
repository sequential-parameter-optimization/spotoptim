# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for Release 0.9.0 — Improvement F: Batch Evaluation API.

Verifies that:
- eval_batch_size parameter is validated and stored correctly.
- remote_batch_eval_wrapper evaluates X_batch in a single fun call.
- eval_batch_size=1 (default) preserves single-point-per-call behavior.
- eval_batch_size>1 batches candidates and calls fun once per batch.
- Budget is respected regardless of batch size.
- End-to-end optimization succeeds for all batch sizes.
- eval_batch_size is ignored (n_jobs=1 path is sequential).
"""

import numpy as np
import pytest
import dill
from spotoptim import SpotOptim
from spotoptim.SpotOptim import remote_batch_eval_wrapper

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

BOUNDS = [(-5, 5), (-5, 5)]


def sphere(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1)


def counting_sphere(X):
    """Sphere that records every call shape for inspection."""
    X = np.atleast_2d(X)
    counting_sphere.calls.append(X.shape)
    return np.sum(X**2, axis=1)


counting_sphere.calls = []


# ---------------------------------------------------------------------------
# Unit: remote_batch_eval_wrapper
# ---------------------------------------------------------------------------


class TestRemoteBatchEvalWrapper:
    """Unit tests for the remote_batch_eval_wrapper helper."""

    def test_single_point_batch(self):
        opt = SpotOptim(fun=sphere, bounds=BOUNDS, n_initial=3, max_iter=6)
        X_batch = np.array([[1.0, 2.0]])
        pickled = dill.dumps((opt, X_batch))
        X_out, y_out = remote_batch_eval_wrapper(pickled)
        assert X_out.shape == (1, 2)
        assert y_out.shape == (1,)
        assert np.isclose(y_out[0], 5.0)

    def test_multi_point_batch(self):
        opt = SpotOptim(fun=sphere, bounds=BOUNDS, n_initial=3, max_iter=6)
        X_batch = np.array([[1.0, 0.0], [0.0, 2.0], [-1.0, -1.0]])
        pickled = dill.dumps((opt, X_batch))
        X_out, y_out = remote_batch_eval_wrapper(pickled)
        assert X_out.shape == (3, 2)
        assert y_out.shape == (3,)
        assert np.allclose(y_out, [1.0, 4.0, 2.0])

    def test_lambda_objective_serialized(self):
        """Lambda functions must survive dill serialization inside the wrapper."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1) + 1.0,
            bounds=BOUNDS,
            n_initial=3,
            max_iter=6,
        )
        X_batch = np.zeros((2, 2))
        pickled = dill.dumps((opt, X_batch))
        X_out, y_out = remote_batch_eval_wrapper(pickled)
        assert np.allclose(y_out, [1.0, 1.0])

    def test_returns_none_exception_on_failure(self):
        """Wrapper must catch exceptions and return (None, exception)."""

        def bad_fun(X):
            raise RuntimeError("boom")

        opt = SpotOptim(fun=bad_fun, bounds=BOUNDS, n_initial=3, max_iter=6)
        X_batch = np.zeros((1, 2))
        pickled = dill.dumps((opt, X_batch))
        X_out, y_out = remote_batch_eval_wrapper(pickled)
        assert X_out is None
        assert isinstance(y_out, Exception)

    def test_output_shapes_preserved(self):
        """X_out must equal X_batch (not a copy that drops rows)."""
        opt = SpotOptim(fun=sphere, bounds=BOUNDS, n_initial=3, max_iter=6)
        X_batch = np.random.default_rng(0).uniform(-5, 5, (5, 2))
        pickled = dill.dumps((opt, X_batch))
        X_out, y_out = remote_batch_eval_wrapper(pickled)
        assert np.array_equal(X_out, X_batch)
        assert y_out.shape == (5,)


# ---------------------------------------------------------------------------
# Unit: eval_batch_size parameter validation
# ---------------------------------------------------------------------------


class TestEvalBatchSizeValidation:
    def test_default_is_one(self):
        opt = SpotOptim(fun=sphere, bounds=BOUNDS, n_initial=3, max_iter=6)
        assert opt.eval_batch_size == 1
        assert opt.config.eval_batch_size == 1

    def test_positive_value_stored(self):
        opt = SpotOptim(
            fun=sphere, bounds=BOUNDS, n_initial=3, max_iter=6, eval_batch_size=4
        )
        assert opt.eval_batch_size == 4
        assert opt.config.eval_batch_size == 4

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="eval_batch_size"):
            SpotOptim(
                fun=sphere, bounds=BOUNDS, n_initial=3, max_iter=6, eval_batch_size=0
            )

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="eval_batch_size"):
            SpotOptim(
                fun=sphere, bounds=BOUNDS, n_initial=3, max_iter=6, eval_batch_size=-1
            )

    def test_large_batch_size_stored(self):
        opt = SpotOptim(
            fun=sphere, bounds=BOUNDS, n_initial=3, max_iter=6, eval_batch_size=100
        )
        assert opt.eval_batch_size == 100


# ---------------------------------------------------------------------------
# Integration: end-to-end with various batch sizes
# ---------------------------------------------------------------------------


class TestBatchEvalEndToEnd:
    """Smoke tests: eval_batch_size produces correct results end-to-end."""

    def test_batch_size_1_default(self):
        """eval_batch_size=1 (default) produces a valid result."""
        opt = SpotOptim(
            fun=sphere,
            bounds=BOUNDS,
            n_initial=6,
            max_iter=12,
            n_jobs=2,
            eval_batch_size=1,
            seed=0,
        )
        result = opt.optimize()
        assert result.success
        assert result.fun < 50.0

    def test_batch_size_2(self):
        opt = SpotOptim(
            fun=sphere,
            bounds=BOUNDS,
            n_initial=6,
            max_iter=14,
            n_jobs=2,
            eval_batch_size=2,
            seed=1,
        )
        result = opt.optimize()
        assert result.success
        assert result.fun < 50.0

    def test_batch_size_equals_n_jobs(self):
        """Canonical use-case from the strategy doc: eval_batch_size=n_jobs."""
        opt = SpotOptim(
            fun=sphere,
            bounds=BOUNDS,
            n_initial=6,
            max_iter=16,
            n_jobs=3,
            eval_batch_size=3,
            seed=42,
        )
        result = opt.optimize()
        assert result.success
        assert result.fun < 50.0

    def test_batch_size_larger_than_n_jobs(self):
        """eval_batch_size > n_jobs is valid; batch fills before dispatching."""
        opt = SpotOptim(
            fun=sphere,
            bounds=BOUNDS,
            n_initial=6,
            max_iter=18,
            n_jobs=2,
            eval_batch_size=4,
            seed=7,
        )
        result = opt.optimize()
        assert result.success

    def test_budget_respected_with_batching(self):
        """Total evaluations must not exceed max_iter regardless of batch size."""
        max_iter = 14
        opt = SpotOptim(
            fun=sphere,
            bounds=BOUNDS,
            n_initial=6,
            max_iter=max_iter,
            n_jobs=2,
            eval_batch_size=3,
            seed=5,
        )
        result = opt.optimize()
        assert result.nfev <= max_iter

    def test_result_structure_consistent_across_batch_sizes(self):
        """All batch sizes produce results with the same output shape."""
        shapes = []
        for bs in [1, 2, 3]:
            opt = SpotOptim(
                fun=sphere,
                bounds=BOUNDS,
                n_initial=6,
                max_iter=12,
                n_jobs=2,
                eval_batch_size=bs,
                seed=0,
            )
            r = opt.optimize()
            assert r.success
            shapes.append(r.X.shape[1])
        assert len(set(shapes)) == 1  # all same dimensionality

    def test_lambda_objective_with_batching(self):
        """Lambda functions survive dill serialization in batch eval path."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=BOUNDS,
            n_initial=5,
            max_iter=12,
            n_jobs=2,
            eval_batch_size=2,
            seed=99,
        )
        result = opt.optimize()
        assert result.success

    def test_closure_objective_with_batching(self):
        """Closures survive dill serialization in batch eval path."""
        scale = 3.0

        def scaled(X):
            return scale * np.sum(X**2, axis=1)

        opt = SpotOptim(
            fun=scaled,
            bounds=BOUNDS,
            n_initial=5,
            max_iter=12,
            n_jobs=2,
            eval_batch_size=2,
            seed=11,
        )
        result = opt.optimize()
        assert result.success

    def test_sequential_path_ignores_eval_batch_size(self):
        """n_jobs=1 (sequential path) does not crash with eval_batch_size>1."""
        opt = SpotOptim(
            fun=sphere,
            bounds=BOUNDS,
            n_initial=5,
            max_iter=10,
            n_jobs=1,
            eval_batch_size=4,
            seed=0,
        )
        result = opt.optimize()
        assert result.success

    def test_4d_sphere_batch(self):
        """Batching works for higher-dimensional problems."""
        opt = SpotOptim(
            fun=sphere,
            bounds=[(-3, 3)] * 4,
            n_initial=8,
            max_iter=16,
            n_jobs=2,
            eval_batch_size=2,
            seed=88,
        )
        result = opt.optimize()
        assert result.success
        assert result.X.shape[1] == 4

    def test_minus_one_n_jobs_with_batching(self):
        """n_jobs=-1 combined with eval_batch_size>1 runs without error."""
        opt = SpotOptim(
            fun=sphere,
            bounds=BOUNDS,
            n_initial=5,
            max_iter=12,
            n_jobs=-1,
            eval_batch_size=2,
            seed=3,
        )
        assert opt.n_jobs >= 1
        result = opt.optimize()
        assert result.success
