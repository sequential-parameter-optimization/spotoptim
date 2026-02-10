# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
from spotoptim import SpotOptim


def test_n_infill_default():
    """Test default n_infill_points=1 behavior."""
    opt = SpotOptim(fun=lambda x: x.sum(), bounds=[(0, 1)], n_infill_points=1)
    assert opt.n_infill_points == 1

    # Mock optimizer returning single point
    opt.optimize_acquisition_func = lambda: np.array([0.5])
    opt.X_ = np.array([[0.1]])  # Existing
    opt.y_ = np.array([0.1])
    opt.var_type = ["float"]  # Needed for repair

    next_point = opt.suggest_next_infill_point()
    assert next_point.shape == (1, 1)  # (N, D)


def test_n_infill_multiple_from_optimizer():
    """Test obtaining multiple unique points from optimizer."""
    opt = SpotOptim(
        fun=lambda x: x.sum(),
        bounds=[(0, 1)],
        n_infill_points=3,
        acquisition_fun_return_size=10,  # Optimizer returns enough candidates
    )

    # Mock optimizer returning multiple distinct points
    unique_points = np.linspace(0.1, 0.9, 5).reshape(-1, 1)  # 5 points
    opt.optimize_acquisition_func = lambda: unique_points

    # Existing points different from new ones
    opt.X_ = np.array([[0.0]])
    opt.y_ = np.array([0.0])
    opt.var_type = ["float"]

    next_points = opt.suggest_next_infill_point()

    assert next_points.shape == (3, 1)
    # Check uniqueness
    assert len(np.unique(next_points, axis=0)) == 3
    # Check they are from our pool
    for p in next_points:
        assert p in unique_points


def test_n_infill_with_fallback():
    """Test filling remaining processing with fallback when optimizer has insufficient unique points."""
    opt = SpotOptim(
        fun=lambda x: x.sum(),
        bounds=[(0, 1)],
        n_infill_points=3,
        acquisition_fun_return_size=1,  # Optimizer only returns 1 point
    )

    # Optimizer returns 1 point
    opt.optimize_acquisition_func = lambda: np.array([0.5])

    opt.X_ = np.array([[0.0]])
    opt.y_ = np.array([0.0])
    opt.var_type = ["float"]

    # Mock fallback to return distant points
    # Fallback uses _handle_acquisition_failure -> random
    # We can rely on default random behavior if bounds are wide enough

    next_points = opt.suggest_next_infill_point()

    assert next_points.shape == (3, 1)
    # First point should be 0.5 (from optimizer)
    assert np.allclose(next_points[0], 0.5)

    # Others should be different (from fallback)
    assert len(np.unique(next_points, axis=0)) == 3


def test_repeats_with_batch_infill():
    """Test _update_repeats_infill_points with batch input."""
    opt = SpotOptim(
        fun=lambda x: x.sum(), bounds=[(0, 1)], n_infill_points=2, repeats_surrogate=3
    )

    # Batch input (2 points, 1 dim)
    batch = np.array([[0.1], [0.9]])

    repeated = opt._update_repeats_infill_points(batch)

    # Expect 2 * 3 = 6 rows
    assert repeated.shape == (6, 1)

    # Check repetition pattern: [A, A, A, B, B, B]
    assert np.allclose(repeated[0:3], 0.1)
    assert np.allclose(repeated[3:6], 0.9)


def test_duplicates_avoidance():
    """Test that it actively avoids duplicates already in X_."""
    opt = SpotOptim(fun=lambda x: x.sum(), bounds=[(0, 1)], n_infill_points=2)

    opt.X_ = np.array([[0.5]])  # 0.5 exists
    opt.y_ = np.array([0.5])
    opt.var_type = ["float"]

    # Optimizer suggests 0.5 (duplicate) and 0.8 (new)
    candidates = np.array([0.5, 0.8]).reshape(-1, 1)
    opt.optimize_acquisition_func = lambda: candidates
    opt.acquisition_fun_return_size = 2

    # Should skip 0.5, take 0.8, and fill 2nd spot with fallback
    next_points = opt.suggest_next_infill_point()

    assert next_points.shape == (2, 1)
    assert np.allclose(next_points[0], 0.8)  # 0.8 should be first
    assert not np.any(np.isclose(next_points, 0.5))  # 0.5 should not be there
