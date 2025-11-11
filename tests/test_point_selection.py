"""
Tests for point selection mechanisms in SpotOptim.
"""

import numpy as np
import pytest
from spotoptim import SpotOptim


def sphere(X):
    """Simple sphere function for testing."""
    return np.sum(X**2, axis=1)


def test_select_distant_points_basic():
    """Test basic functionality of _select_distant_points method."""
    optimizer = SpotOptim(
        fun=sphere,
        bounds=[(-5, 5), (-5, 5)],
        max_surrogate_points=5,
        selection_method="distant",
        seed=42,
    )

    # Create test data
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([1, 2, 3, 4, 5])
    k = 3

    selected_X, selected_y = optimizer._select_distant_points(X, y, k)

    # Check shapes
    assert selected_X.shape == (k, 2), f"Expected shape ({k}, 2), got {selected_X.shape}"
    assert selected_y.shape == (k,), f"Expected shape ({k},), got {selected_y.shape}"

    # Check that selected points are from original set
    for point in selected_X:
        assert any(np.allclose(point, x) for x in X), "Selected point not in original set"


def test_select_best_cluster_basic():
    """Test basic functionality of _select_best_cluster method."""
    optimizer = SpotOptim(
        fun=sphere,
        bounds=[(-5, 5), (-5, 5)],
        max_surrogate_points=5,
        selection_method="best",
        seed=42,
    )

    # Create test data with clear clusters
    X = np.array([[1, 1], [1.1, 1.1], [1.2, 1.2], [10, 10], [10.1, 10.1]])
    y = np.array([1, 1.1, 1.2, 10, 10.1])
    k = 2

    selected_X, selected_y = optimizer._select_best_cluster(X, y, k)

    # Check that we got points back
    assert selected_X.shape[0] > 0, "No points selected"
    assert selected_y.shape[0] > 0, "No y values selected"

    # Check that selected points are from original set
    for point in selected_X:
        assert any(np.allclose(point, x) for x in X), "Selected point not in original set"

    # Check that the best cluster (with smaller y values) was selected
    assert np.mean(selected_y) < 5, "Expected best cluster with smaller y values"


def test_selection_dispatcher_distant():
    """Test selection dispatcher with 'distant' method."""
    optimizer = SpotOptim(
        fun=sphere,
        bounds=[(-5, 5), (-5, 5)],
        max_surrogate_points=3,
        selection_method="distant",
        seed=42,
    )

    X = np.random.rand(20, 2)
    y = np.random.rand(20)

    selected_X, selected_y = optimizer._selection_dispatcher(X, y)

    assert selected_X.shape == (3, 2), "Expected 3 points selected"
    assert selected_y.shape == (3,), "Expected 3 y values selected"


def test_selection_dispatcher_best():
    """Test selection dispatcher with 'best' method."""
    optimizer = SpotOptim(
        fun=sphere,
        bounds=[(-5, 5), (-5, 5)],
        max_surrogate_points=3,
        selection_method="best",
        seed=42,
    )

    X = np.random.rand(20, 2)
    y = np.random.rand(20)

    selected_X, selected_y = optimizer._selection_dispatcher(X, y)

    # Should return points from best cluster (not necessarily exactly 3)
    assert selected_X.shape[0] > 0, "Expected some points selected"
    assert selected_y.shape[0] > 0, "Expected some y values selected"


def test_selection_dispatcher_no_limit():
    """Test selection dispatcher when max_surrogate_points is None."""
    optimizer = SpotOptim(
        fun=sphere,
        bounds=[(-5, 5), (-5, 5)],
        max_surrogate_points=None,
        selection_method="distant",
        seed=42,
    )

    X = np.random.rand(20, 2)
    y = np.random.rand(20)

    selected_X, selected_y = optimizer._selection_dispatcher(X, y)

    # Should return all points
    assert selected_X.shape == X.shape, "Expected all points returned"
    assert selected_y.shape == y.shape, "Expected all y values returned"
    np.testing.assert_array_equal(selected_X, X)
    np.testing.assert_array_equal(selected_y, y)


def test_fit_surrogate_with_selection():
    """Test that surrogate fitting uses point selection when needed."""
    optimizer = SpotOptim(
        fun=sphere,
        bounds=[(-5, 5), (-5, 5)],
        max_iter=10, n_initial=10,
        max_surrogate_points=5,
        selection_method="distant",
        verbose=False,
        seed=42,
    )

    # Generate initial design
    X = optimizer._generate_initial_design()
    y = optimizer._evaluate_function(X)

    # Add more points to exceed max_surrogate_points
    X_extra = np.random.rand(10, 2) * 10 - 5
    y_extra = optimizer._evaluate_function(X_extra)

    X_all = np.vstack([X, X_extra])
    y_all = np.concatenate([y, y_extra])

    # Fit surrogate - should trigger selection
    optimizer._fit_surrogate(X_all, y_all)

    # Verify surrogate is fitted
    X_test = np.array([[0, 0]])
    y_pred = optimizer.surrogate.predict(X_test)
    assert y_pred is not None, "Surrogate prediction failed"


def test_optimize_with_max_surrogate_points():
    """Test full optimization with max_surrogate_points limit."""
    optimizer = SpotOptim(
        fun=sphere,
        bounds=[(-5, 5), (-5, 5)],
        max_iter=10,
        n_initial=5,
        max_surrogate_points=8,
        selection_method="distant",
        verbose=False,
        seed=42,
    )

    result = optimizer.optimize()

    # Check that optimization completed
    assert result.success, "Optimization did not succeed"
    assert result.nfev > 0, "No function evaluations"
    assert result.fun < 25, "Did not find a good solution"  # Starting bound is (-5, 5)


def test_optimize_with_best_selection():
    """Test full optimization with 'best' selection method."""
    optimizer = SpotOptim(
        fun=sphere,
        bounds=[(-5, 5), (-5, 5)],
        max_iter=10,
        n_initial=5,
        max_surrogate_points=8,
        selection_method="best",
        verbose=False,
        seed=42,
    )

    result = optimizer.optimize()

    # Check that optimization completed
    assert result.success, "Optimization did not succeed"
    assert result.nfev > 0, "No function evaluations"
    assert result.fun < 25, "Did not find a good solution"


def test_too_few_points_for_clustering():
    """Test behavior when k is greater than number of points."""
    optimizer = SpotOptim(
        fun=sphere,
        bounds=[(-5, 5), (-5, 5)],
        max_surrogate_points=10,
        selection_method="distant",
        seed=42,
    )

    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 2])
    k = 5

    # Should raise an error from KMeans
    with pytest.raises(ValueError):
        optimizer._select_distant_points(X, y, k)


def test_identical_points_handling():
    """Test that selection handles duplicate points correctly."""
    optimizer = SpotOptim(
        fun=sphere,
        bounds=[(-5, 5), (-5, 5)],
        max_surrogate_points=3,
        selection_method="distant",
        seed=42,
    )

    # Create data with some duplicate points
    X = np.array([[1, 1], [1, 1], [2, 2], [3, 3], [4, 4]])
    y = np.array([1, 1, 2, 3, 4])

    selected_X, selected_y = optimizer._select_distant_points(X, y, 3)

    assert selected_X.shape == (3, 2), "Expected 3 points selected"
    assert selected_y.shape == (3,), "Expected 3 y values selected"


def test_verbose_output(capsys):
    """Test that verbose output is produced when selecting points."""
    optimizer = SpotOptim(
        fun=sphere,
        bounds=[(-5, 5), (-5, 5)],
        max_iter=5, n_initial=5,
        max_surrogate_points=8,
        selection_method="distant",
        verbose=True,
        seed=42,
    )

    # Create enough points to trigger selection
    X = np.random.rand(15, 2) * 10 - 5
    y = optimizer._evaluate_function(X)

    optimizer._fit_surrogate(X, y)

    captured = capsys.readouterr()
    assert "Selecting subset" in captured.out, "Expected verbose output about point selection"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
