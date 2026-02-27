# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
from spotoptim import SpotOptim, Kriging


def rosenbrock(X):
    X = np.atleast_2d(X)
    x, y = X[:, 0], X[:, 1]
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


def test_quick_start():
    # Set up optimization
    bounds = [(-2, 2), (-2, 2)]

    optimizer = SpotOptim(
        fun=rosenbrock,
        bounds=bounds,
        max_iter=5,  # Reduced for testing
        n_initial=5,  # Reduced for testing
        seed=42,
    )

    # Run optimization
    result = optimizer.optimize()

    assert hasattr(result, "x")
    assert hasattr(result, "fun")
    assert len(result.x) == 2


def test_kriging_surrogate():
    bounds = [(-2, 2), (-2, 2)]

    # Create Kriging surrogate
    kriging = Kriging(noise=1e-6, min_theta=-3.0, max_theta=2.0, seed=42)

    # Use with SpotOptim
    optimizer = SpotOptim(
        fun=rosenbrock,
        bounds=bounds,
        surrogate=kriging,  # Use Kriging instead of default GP
        max_iter=5,
        n_initial=5,
        seed=42,
    )

    result = optimizer.optimize()
    assert hasattr(result, "x")


def test_visualizing_results(monkeypatch):
    # Mock plt.show to avoid blocking
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)

    bounds = [(-2, 2), (-2, 2)]
    optimizer = SpotOptim(
        fun=rosenbrock, bounds=bounds, max_iter=5, n_initial=5, seed=42
    )
    optimizer.optimize()

    # After running optimization
    optimizer.plot_surrogate(
        i=0,
        j=1,  # Dimensions to plot
        var_name=["x1", "x2"],  # Variable names
        add_points=True,  # Show evaluated points
        cmap="viridis",  # Colormap
        show=True,
    )


def test_point_selection():
    bounds = [(-2, 2), (-2, 2)]

    def expensive_function(X):
        return rosenbrock(X)

    optimizer = SpotOptim(
        fun=expensive_function,
        bounds=bounds,
        max_iter=10,
        n_initial=10,
        max_surrogate_points=5,  # Use only 5 points for surrogate training
        selection_method="distant",  # or 'best'
        verbose=True,
    )
    result = optimizer.optimize()
    assert hasattr(result, "x")
