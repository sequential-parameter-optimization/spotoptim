# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
from spotoptim import SpotOptim
from spotoptim.surrogate import MLPSurrogate


def sphere(X):
    """Simple sphere function for testing."""
    return np.sum(X**2, axis=1)


def rosenbrock(X):
    """Rosenbrock function."""
    X = np.atleast_2d(X)
    return np.sum(100 * (X[:, 1:] - X[:, :-1] ** 2) ** 2 + (1 - X[:, :-1]) ** 2, axis=1)


def test_spotoptim_initialization_with_mlp_surrogate():
    """Test that SpotOptim accepts MLPSurrogate."""
    bounds = [(-5, 5), (-5, 5)]
    mlp = MLPSurrogate(epochs=10, seed=42)

    opt = SpotOptim(
        fun=sphere,
        bounds=bounds,
        surrogate=mlp,
        max_iter=10,
        n_initial=5,
        seed=42,
        verbose=False,
    )

    assert opt.surrogate == mlp
    assert opt.surrogate.optimizer_name == "AdamWScheduleFree"


def test_spotoptim_optimize_loop_mlp():
    """Test that the optimization loop runs with MLP surrogate."""
    bounds = [(-5, 5), (-5, 5)]
    # Minimal training for speed test
    mlp = MLPSurrogate(epochs=5, verbose=False, seed=42)

    opt = SpotOptim(
        fun=sphere,
        bounds=bounds,
        surrogate=mlp,
        max_iter=5,  # Short loop
        n_initial=5,
        acquisition="ei",  # Use uncertainty-based acquisition
        seed=42,
        verbose=False,
    )

    result = opt.optimize()

    assert result.success
    assert result.nfev == 5
    # Just check it ran
    assert result.fun < 100.0


def test_mlp_surrogate_uncertainty_in_loop():
    """Test that MLPSurrogate provides uncertainty estimates used by SpotOptim."""
    bounds = [(-2, 2), (-2, 2)]
    # Use high dropout to ensure uncertainty
    mlp = MLPSurrogate(epochs=5, dropout=0.5, mc_dropout_passes=5, seed=42)

    opt = SpotOptim(
        fun=sphere,
        bounds=bounds,
        surrogate=mlp,
        max_iter=5,
        n_initial=3,
        acquisition="ei",
        seed=42,
        verbose=False,
    )

    opt.optimize()

    # Check predictions directly
    X_test = np.array([[0.0, 0.0], [1.0, 1.0], [1.5, 1.5]])
    mean, std = opt._predict_with_uncertainty(X_test)

    assert mean.shape == (3,)
    assert std.shape == (3,)
    # Should have non-zero uncertainty due to MC dropout
    assert np.any(std > 0), "Expected non-zero uncertainty from MLP with dropout"


def test_optimization_convergence_rosenbrock():
    """Test that it runs on Rosenbrock (sanity check)."""
    bounds = [(-2, 2), (-2, 2)]
    mlp = MLPSurrogate(epochs=10, seed=42)

    opt = SpotOptim(
        fun=rosenbrock,
        bounds=bounds,
        surrogate=mlp,
        max_iter=5,
        n_initial=5,
        acquisition="lcb",
        seed=42,
        verbose=False,
    )

    result = opt.optimize()

    # Just check successful execution
    assert result.success
