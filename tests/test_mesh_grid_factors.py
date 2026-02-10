# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for _generate_mesh_grid_with_factors method."""

import pytest
import numpy as np
from spotoptim import SpotOptim
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for testing
import matplotlib.pyplot as plt


def objective_with_factors(X):
    """Objective function for testing with factor variables."""
    results = []
    for params in X:
        l1 = int(params[0])
        activation = params[1]  # Factor variable
        lr = params[2]

        # Simple function based on parameters
        activation_score = {"Tanh": 1.0, "ReLU": 0.8, "Sigmoid": 1.2}
        score = l1 * 0.01 + activation_score.get(activation, 1.0) + lr * 10
        results.append(score)
    return np.array(results)


def test_generate_mesh_grid_with_factors_both_numeric():
    """Test mesh grid generation with two numeric parameters."""
    bounds = [
        (16, 128),  # l1: int
        ("Tanh", "ReLU", "Sigmoid"),  # activation: factor
        (0.001, 0.1),  # lr: float
    ]
    var_type = ["int", "factor", "float"]
    var_name = ["l1", "activation", "lr"]

    opt = SpotOptim(
        fun=objective_with_factors,
        bounds=bounds,
        var_type=var_type,
        var_name=var_name,
        max_iter=5,
        n_initial=5,
        seed=42,
    )
    result = opt.optimize()

    # Test with two numeric dimensions (0=l1, 2=lr)
    X_i, X_j, grid_points, labels_i, labels_j = opt._generate_mesh_grid_with_factors(
        i=0, j=2, num=50, is_factor_i=False, is_factor_j=False
    )

    # Check shapes
    assert X_i.shape == (50, 50), "X_i should be 50x50"
    assert X_j.shape == (50, 50), "X_j should be 50x50"
    assert grid_points.shape == (2500, 3), "grid_points should be (2500, 3)"

    # Check that factor labels are None for numeric dimensions
    assert labels_i is None, "labels_i should be None for numeric dimension"
    assert labels_j is None, "labels_j should be None for numeric dimension"

    # Check that grid_points are finite
    assert np.all(np.isfinite(grid_points)), "All grid points should be finite"

    # Check ranges
    assert np.min(X_i) >= 16 and np.max(X_i) <= 128, "X_i should be in [16, 128]"
    assert np.min(X_j) >= 0.001 and np.max(X_j) <= 0.1, "X_j should be in [0.001, 0.1]"


def test_generate_mesh_grid_with_factors_one_factor():
    """Test mesh grid generation with one factor and one numeric parameter."""
    bounds = [
        (16, 128),  # l1: int
        ("Tanh", "ReLU", "Sigmoid"),  # activation: factor
        (0.001, 0.1),  # lr: num
    ]
    var_type = ["int", "factor", "float"]
    var_name = ["l1", "activation", "lr"]

    opt = SpotOptim(
        fun=objective_with_factors,
        bounds=bounds,
        var_type=var_type,
        var_name=var_name,
        max_iter=5,
        n_initial=5,
        seed=42,
    )
    result = opt.optimize()

    # Test with one factor (1=activation) and one numeric (2=lr)
    X_i, X_j, grid_points, labels_i, labels_j = opt._generate_mesh_grid_with_factors(
        i=1, j=2, num=50, is_factor_i=True, is_factor_j=False
    )

    # Check shapes - meshgrid with 3 factor levels × 50 numeric points
    assert X_i.shape == (
        50,
        3,
    ), "X_i should be 50x3 (50 lr points × 3 activation levels)"
    assert X_j.shape == X_i.shape, "X_j should have same shape as X_i"
    assert grid_points.shape == (150, 3), "grid_points should be (150, 3)"

    # Check factor labels
    assert labels_i == [
        "Tanh",
        "ReLU",
        "Sigmoid",
    ], "labels_i should contain factor levels"
    assert labels_j is None, "labels_j should be None for numeric dimension"

    # Check that grid_points are finite
    assert np.all(np.isfinite(grid_points)), "All grid points should be finite"

    # Check that factor dimension uses integer indices
    unique_i = np.unique(X_i.ravel())
    assert len(unique_i) == 3, "Factor dimension should have 3 unique values"
    assert np.all(unique_i == [0, 1, 2]), "Factor indices should be [0, 1, 2]"


def test_generate_mesh_grid_with_factors_both_factors():
    """Test mesh grid generation with two factor parameters."""
    bounds = [
        (16, 128),  # l1: int
        ("Tanh", "ReLU", "Sigmoid"),  # activation: factor
        ("Adam", "SGD"),  # optimizer: factor
    ]
    var_type = ["int", "factor", "factor"]
    var_name = ["l1", "activation", "optimizer"]

    def objective_two_factors(X):
        results = []
        for params in X:
            l1 = int(params[0])
            activation = params[1]
            optimizer = params[2]

            activation_score = {"Tanh": 1.0, "ReLU": 0.8, "Sigmoid": 1.2}
            optimizer_score = {"Adam": 0.5, "SGD": 0.7}
            score = (
                l1 * 0.01
                + activation_score.get(activation, 1.0)
                + optimizer_score.get(optimizer, 0.5)
            )
            results.append(score)
        return np.array(results)

    opt = SpotOptim(
        fun=objective_two_factors,
        bounds=bounds,
        var_type=var_type,
        var_name=var_name,
        max_iter=5,
        n_initial=5,
        seed=42,
    )
    result = opt.optimize()

    # Test with two factors (1=activation, 2=optimizer)
    X_i, X_j, grid_points, labels_i, labels_j = opt._generate_mesh_grid_with_factors(
        i=1, j=2, num=50, is_factor_i=True, is_factor_j=True
    )

    # Check shapes - meshgrid with 2 optimizer levels × 3 activation levels
    assert X_i.shape == (
        2,
        3,
    ), "X_i should be 2x3 (2 optimizer levels × 3 activation levels)"
    assert X_j.shape == X_i.shape, "X_j should have same shape as X_i"
    assert grid_points.shape == (6, 3), "grid_points should be (6, 3)"

    # Check factor labels
    assert labels_i == [
        "Tanh",
        "ReLU",
        "Sigmoid",
    ], "labels_i should contain activation levels"
    assert labels_j == ["Adam", "SGD"], "labels_j should contain optimizer levels"

    # Check that grid_points are finite
    assert np.all(np.isfinite(grid_points)), "All grid points should be finite"

    # Check that both dimensions use integer indices
    unique_i = np.unique(X_i.ravel())
    unique_j = np.unique(X_j.ravel())
    assert np.all(unique_i == [0, 1, 2]), "Activation indices should be [0, 1, 2]"
    assert np.all(unique_j == [0, 1]), "Optimizer indices should be [0, 1]"


def test_generate_mesh_grid_with_factors_custom_num():
    """Test mesh grid generation with custom num parameter."""
    bounds = [
        (16, 128),  # l1: int
        ("Tanh", "ReLU", "Sigmoid"),  # activation: factor
        (0.001, 0.1),  # lr: num
    ]
    var_type = ["int", "factor", "float"]
    var_name = ["l1", "activation", "lr"]

    opt = SpotOptim(
        fun=objective_with_factors,
        bounds=bounds,
        var_type=var_type,
        var_name=var_name,
        max_iter=5,
        n_initial=5,
        seed=42,
    )
    result = opt.optimize()

    # Test with custom num=20 for numeric dimensions
    X_i, X_j, grid_points, labels_i, labels_j = opt._generate_mesh_grid_with_factors(
        i=0, j=2, num=20, is_factor_i=False, is_factor_j=False
    )

    # Check that custom num is respected
    assert X_i.shape == (20, 20), "X_i should be 20x20"
    assert X_j.shape == (20, 20), "X_j should be 20x20"
    assert grid_points.shape == (400, 3), "grid_points should be (400, 3)"


def test_generate_mesh_grid_with_factors_mean_computation():
    """Test that mean values are computed correctly for non-plotted dimensions."""
    bounds = [
        (16, 128),  # l1: int
        ("Tanh", "ReLU", "Sigmoid"),  # activation: factor
        (0.001, 0.1),  # lr: num
    ]
    var_type = ["int", "factor", "float"]
    var_name = ["l1", "activation", "lr"]

    opt = SpotOptim(
        fun=objective_with_factors,
        bounds=bounds,
        var_type=var_type,
        var_name=var_name,
        max_iter=5,
        n_initial=5,
        seed=42,
    )
    result = opt.optimize()

    # Generate mesh for dimensions 0 and 2 (l1 and lr)
    # Dimension 1 (activation) should be held at a constant value
    X_i, X_j, grid_points, labels_i, labels_j = opt._generate_mesh_grid_with_factors(
        i=0, j=2, num=10, is_factor_i=False, is_factor_j=False
    )

    # The middle dimension (activation) should be constant across all grid points
    # Check that all values in dimension 1 are the same
    dim1_values = grid_points[:, 1]
    assert (
        len(np.unique(dim1_values)) == 1
    ), "Non-plotted dimension should have constant value"

    # Check that the constant value is valid (should be 0, 1, or 2 after transformation)
    unique_val = dim1_values[0]
    assert np.isfinite(unique_val), "Constant value should be finite"


def test_plot_important_hyperparameter_contour_with_factors():
    """Integration test for plot_important_hyperparameter_contour with factors."""
    bounds = [
        (16, 128),  # l1: int
        ("Tanh", "ReLU", "Sigmoid"),  # activation: factor
        (0.001, 0.1),  # lr: num
        (0.01, 1.0),  # alpha: num
    ]
    var_type = ["int", "factor", "float", "float"]
    var_name = ["l1", "activation", "lr", "alpha"]

    def objective_4d(X):
        results = []
        for params in X:
            l1 = int(params[0])
            activation = params[1]
            lr = params[2]
            alpha = params[3]

            activation_score = {"Tanh": 1.0, "ReLU": 0.8, "Sigmoid": 1.2}
            score = (
                l1 * 0.01 + activation_score.get(activation, 1.0) + lr * 10 + alpha * 5
            )
            results.append(score)
        return np.array(results)

    opt = SpotOptim(
        fun=objective_4d,
        bounds=bounds,
        var_type=var_type,
        var_name=var_name,
        max_iter=10,
        n_initial=8,
        seed=42,
    )
    result = opt.optimize()

    # Test plotting - should handle mixed types
    opt.plot_important_hyperparameter_contour(max_imp=3, show=False)
    plt.close("all")


def test_generate_mesh_grid_with_factors_no_factor_maps():
    """Test handling when factor maps are not initialized."""
    bounds = [(16, 128), (0.001, 0.1)]  # l1: int  # lr: num
    var_type = ["int", "float"]
    var_name = ["l1", "lr"]

    def simple_objective(X):
        return np.sum(X, axis=1)

    opt = SpotOptim(
        fun=simple_objective,
        bounds=bounds,
        var_type=var_type,
        var_name=var_name,
        max_iter=5,
        n_initial=5,
        seed=42,
    )
    result = opt.optimize()

    # Should work fine without factor variables
    X_i, X_j, grid_points, labels_i, labels_j = opt._generate_mesh_grid_with_factors(
        i=0, j=1, num=30, is_factor_i=False, is_factor_j=False
    )

    assert X_i.shape == (30, 30), "Should work with no factors"
    assert labels_i is None and labels_j is None, "No factor labels expected"
    assert np.all(np.isfinite(grid_points)), "All grid points should be finite"


def test_plot_surrogate_with_factors_integration():
    """Test _plot_surrogate_with_factors method directly."""
    bounds = [
        (16, 128),  # l1: int
        ("Tanh", "ReLU", "Sigmoid"),  # activation: factor
        (0.001, 0.1),  # lr: num
    ]
    var_type = ["int", "factor", "float"]
    var_name = ["l1", "activation", "lr"]

    opt = SpotOptim(
        fun=objective_with_factors,
        bounds=bounds,
        var_type=var_type,
        var_name=var_name,
        max_iter=5,
        n_initial=5,
        seed=42,
    )
    result = opt.optimize()

    # Test plotting with factor on i dimension
    opt._plot_surrogate_with_factors(i=1, j=2, show=False, num=20)
    plt.close("all")

    # Test plotting with factor on j dimension
    opt._plot_surrogate_with_factors(i=0, j=1, show=False, num=20)
    plt.close("all")

    # Test plotting with two numeric dimensions
    opt._plot_surrogate_with_factors(i=0, j=2, show=False, num=20)
    plt.close("all")


def test_generate_mesh_grid_with_factors_in_other_dims():
    """Test _generate_mesh_grid when plotting non-factors but dataset has factors."""
    np.random.seed(42)

    # Custom objective function for this test
    def objective_factor_in_last_dim(X):
        """Objective with factor in last dimension."""
        results = []
        for params in X:
            num_layers = int(params[0])
            alpha = float(params[1])
            activation = params[2]  # Factor in last position

            activation_score = {"Tanh": 1.0, "ReLU": 0.8, "Sigmoid": 1.2}
            score = num_layers * 0.01 + alpha + activation_score.get(activation, 1.0)
            results.append(score)
        return np.array(results)

    # Create optimizer with factors in OTHER dimensions (not the ones being plotted)
    bounds = [
        (1, 3),  # num_layers: int
        (0.0, 1.0),  # alpha: num
        ("Tanh", "ReLU", "Sigmoid"),  # activation: factor
    ]
    var_type = ["int", "float", "factor"]
    var_name = ["num_layers", "alpha", "activation"]

    opt = SpotOptim(
        fun=objective_factor_in_last_dim,
        bounds=bounds,
        var_type=var_type,
        var_name=var_name,
        max_iter=5,
        n_initial=5,
        seed=42,
    )
    result = opt.optimize()

    # Test plotting dimensions 0 and 1 (both non-factors)
    # This should use _generate_mesh_grid, not _generate_mesh_grid_with_factors
    # The key test: can it handle factor variables in OTHER dimensions?
    try:
        X_i, X_j, grid = opt._generate_mesh_grid(0, 1, 10)

        # Check shapes
        assert X_i.shape == (10, 10)
        assert X_j.shape == (10, 10)
        assert grid.shape == (100, 3)

        # Check that grid is finite (no inf/nan from factor handling)
        assert np.all(np.isfinite(grid)), "Grid should not contain inf/nan values"

    except Exception as e:
        pytest.fail(f"Test failed with error: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
