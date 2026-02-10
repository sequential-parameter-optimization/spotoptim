# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
from spotoptim import SpotOptim


def dummy_obj(X):
    return np.sum(X**2, axis=1)


def test_multiple_x0_support():
    print("Testing multiple x0 support...")

    # Define 3 initial points
    X_start = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]])

    # Initialize SpotOptim with these points
    # n_initial=5, so it should add 2 more random points
    optimizer = SpotOptim(
        fun=dummy_obj,
        bounds=[(0, 1), (0, 1)],
        n_initial=5,
        max_iter=5,  # only initial design
        x0=X_start,
        seed=42,
    )

    # Optimization runs (only initial design evaluation)
    optimizer.optimize()

    # Check X_
    # The first 3 points should match X_start
    # Note: internal X is in transformed scale. Here bounds match internal scale (0-1 usually if no transform).
    # Bounds are (0,1), no transform, so scale matches.

    X_evaluated = optimizer.X_
    print(f"Evaluated X shape: {X_evaluated.shape}")
    print("First 3 points:")
    print(X_evaluated[:3])

    assert X_evaluated.shape[0] == 5
    np.testing.assert_allclose(X_evaluated[:3], X_start, atol=1e-6)

    print("PASSED: test_multiple_x0_support")


def test_x0_exceeds_n_initial():
    print("Testing x0 > n_initial...")

    X_start = np.random.rand(10, 2)

    # n_initial=5, provided 10. Should use all 10.
    optimizer = SpotOptim(
        fun=dummy_obj,
        bounds=[(0, 1), (0, 1)],
        n_initial=5,
        max_iter=10,
        x0=X_start,
        seed=42,
    )

    optimizer.optimize()

    X_evaluated = optimizer.X_
    print(f"Evaluated X shape: {X_evaluated.shape}")

    assert X_evaluated.shape[0] == 10
    np.testing.assert_allclose(X_evaluated, X_start, atol=1e-6)

    print("PASSED: test_x0_exceeds_n_initial")


if __name__ == "__main__":
    test_multiple_x0_support()
    test_x0_exceeds_n_initial()
