"""Tests for x0 (starting point) parameter in SpotOptim."""

import numpy as np
import pytest
from spotoptim import SpotOptim


def test_x0_basic_usage():
    """Test basic usage of x0 parameter."""

    def objective(X):
        return np.sum(X**2, axis=1)

    x0 = np.array([1.0, 2.0])

    opt = SpotOptim(
        fun=objective,
        bounds=[(-5, 5), (-5, 5)],
        x0=x0,
        max_iter=20,
        n_initial=10,
        seed=42,
        verbose=False,
    )

    result = opt.optimize()

    # Check that optimization completed
    assert result.success is True

    # Check that x0 was evaluated (should be first point in history)
    # Transform x0 to compare with stored values
    x0_transformed = opt._transform_X(x0.reshape(1, -1)).ravel()
    assert np.allclose(opt.X_[0], x0_transformed, atol=1e-6)


def test_x0_within_bounds():
    """Test that x0 must be within bounds."""

    def objective(X):
        return np.sum(X**2, axis=1)

    # x0 outside bounds
    x0 = np.array([10.0, 2.0])  # First value exceeds upper bound of 5

    with pytest.raises(ValueError, match="outside bounds"):
        SpotOptim(
            fun=objective, bounds=[(-5, 5), (-5, 5)], x0=x0, max_iter=10, n_initial=5
        )


def test_x0_correct_dimensions():
    """Test that x0 must have correct number of dimensions."""

    def objective(X):
        return np.sum(X**2, axis=1)

    # Wrong number of dimensions
    x0 = np.array([1.0, 2.0, 3.0])  # 3 dims, but bounds specify 2

    with pytest.raises(ValueError, match="expected 2 dimensions"):
        SpotOptim(
            fun=objective, bounds=[(-5, 5), (-5, 5)], x0=x0, max_iter=10, n_initial=5
        )


def test_x0_scalar_value_error():
    """Test that x0 cannot be a scalar."""

    def objective(X):
        return np.sum(X**2, axis=1)

    x0 = 1.0  # Scalar, not array

    with pytest.raises(ValueError, match="must be a 1D array-like"):
        SpotOptim(
            fun=objective, bounds=[(-5, 5), (-5, 5)], x0=x0, max_iter=10, n_initial=5
        )


def test_x0_2d_array_single_point():
    """Test that x0 can be a 2D array with single point."""

    def objective(X):
        return np.sum(X**2, axis=1)

    x0 = np.array([[1.0, 2.0]])  # 2D array, shape (1, 2)

    opt = SpotOptim(
        fun=objective,
        bounds=[(-5, 5), (-5, 5)],
        x0=x0,
        max_iter=10,
        n_initial=5,
        seed=42,
        verbose=False,
    )

    result = opt.optimize()
    assert result.success is True


def test_x0_2d_array_multiple_points_error():
    """Test that x0 cannot have multiple points."""

    def objective(X):
        return np.sum(X**2, axis=1)

    x0 = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2D array, shape (2, 2)

    with pytest.raises(ValueError, match="must be a single point"):
        SpotOptim(
            fun=objective, bounds=[(-5, 5), (-5, 5)], x0=x0, max_iter=10, n_initial=5
        )


def test_x0_improves_optimization():
    """Test that good x0 can improve optimization performance."""

    def objective(X):
        # Function with minimum near [1, 1]
        return np.sum((X - 1) ** 2, axis=1)

    # Good starting point near optimum
    x0_good = np.array([0.8, 0.9])

    opt_with_x0 = SpotOptim(
        fun=objective,
        bounds=[(0, 5), (0, 5)],
        x0=x0_good,
        max_iter=20,
        n_initial=10,
        seed=42,
        verbose=False,
    )

    result_with_x0 = opt_with_x0.optimize()

    # The good starting point should help find a better solution faster
    assert result_with_x0.success is True
    assert result_with_x0.fun < 0.5  # Should get close to minimum


def test_x0_with_transformations():
    """Test x0 with variable transformations (log scale)."""

    def objective(X):
        return np.sum(np.log(X + 1) ** 2, axis=1)

    # x0 in original scale
    x0 = np.array([2.0, 3.0])

    opt = SpotOptim(
        fun=objective,
        bounds=[(0.1, 10), (0.1, 10)],
        x0=x0,
        var_trans=["log10", "log10"],
        max_iter=15,
        n_initial=8,
        seed=42,
        verbose=False,
    )

    result = opt.optimize()
    assert result.success is True


def test_x0_with_integer_variables():
    """Test x0 with integer variable types."""

    def objective(X):
        return np.sum(X**2, axis=1)

    x0 = np.array([2.0, 3.0])  # Will be rounded to integers

    opt = SpotOptim(
        fun=objective,
        bounds=[(0, 5), (0, 5)],
        x0=x0,
        var_type=["int", "int"],
        max_iter=15,
        n_initial=8,
        seed=42,
        verbose=False,
    )

    result = opt.optimize()
    assert result.success is True


def test_x0_with_mixed_variable_types():
    """Test x0 with mixed variable types."""

    def objective(X):
        return np.sum(X**2, axis=1)

    x0 = np.array([1.5, 3.0, 2.7])

    opt = SpotOptim(
        fun=objective,
        bounds=[(-5, 5), (0, 5), (-3, 3)],
        x0=x0,
        var_type=["float", "int", "float"],
        max_iter=15,
        n_initial=8,
        seed=42,
        verbose=False,
    )

    result = opt.optimize()
    assert result.success is True


def test_x0_with_fixed_dimension():
    """Test x0 with dimension reduction (fixed dimensions)."""

    def objective(X):
        return np.sum(X**2, axis=1)

    # Second dimension is fixed at 2
    x0 = np.array([1.0, 2.0, 3.0])

    opt = SpotOptim(
        fun=objective,
        bounds=[(-5, 5), (2, 2), (-5, 5)],  # Middle dim fixed
        x0=x0,
        max_iter=15,
        n_initial=8,
        seed=42,
        verbose=False,
    )

    result = opt.optimize()
    assert result.success is True


def test_x0_fixed_dimension_wrong_value():
    """Test that x0 must match fixed dimension value."""

    def objective(X):
        return np.sum(X**2, axis=1)

    # Second dimension should be 2, but x0 has 3
    x0 = np.array([1.0, 3.0, 3.0])

    with pytest.raises(ValueError, match="fixed dimension"):
        SpotOptim(
            fun=objective,
            bounds=[(-5, 5), (2, 2), (-5, 5)],
            x0=x0,
            max_iter=10,
            n_initial=5,
        )


def test_x0_verbose_output(capsys):
    """Test verbose output when x0 is provided."""

    def objective(X):
        return np.sum(X**2, axis=1)

    x0 = np.array([1.0, 2.0])

    opt = SpotOptim(
        fun=objective,
        bounds=[(-5, 5), (-5, 5)],
        x0=x0,
        max_iter=10,
        n_initial=5,
        seed=42,
        verbose=True,
    )

    captured = capsys.readouterr()
    assert "Starting point x0 validated" in captured.out

    # Run optimization
    opt.optimize()

    captured = capsys.readouterr()
    assert "Including starting point x0" in captured.out


def test_x0_none_default():
    """Test default behavior when x0 is None."""

    def objective(X):
        return np.sum(X**2, axis=1)

    opt = SpotOptim(
        fun=objective,
        bounds=[(-5, 5), (-5, 5)],
        x0=None,  # Explicitly None
        max_iter=10,
        n_initial=5,
        seed=42,
        verbose=False,
    )

    result = opt.optimize()
    assert result.success is True


def test_x0_with_noisy_function():
    """Test x0 with noisy objective function and repeated evaluations."""

    def noisy_objective(X):
        base = np.sum(X**2, axis=1)
        noise = np.random.normal(0, 0.1, size=base.shape)
        return base + noise

    x0 = np.array([1.0, 1.0])

    opt = SpotOptim(
        fun=noisy_objective,
        bounds=[(-5, 5), (-5, 5)],
        x0=x0,
        max_iter=25,
        n_initial=10,
        repeats_initial=2,
        repeats_surrogate=2,
        seed=42,
        verbose=False,
    )

    result = opt.optimize()
    assert result.success is True


def test_x0_included_in_initial_design():
    """Test that x0 is actually included in the initial design."""

    def objective(X):
        return np.sum(X**2, axis=1)

    x0 = np.array([2.5, -3.5])

    opt = SpotOptim(
        fun=objective,
        bounds=[(-5, 5), (-5, 5)],
        x0=x0,
        max_iter=20,
        n_initial=10,
        seed=42,
        verbose=False,
    )

    result = opt.optimize()

    # Check that we still have n_initial + sequential iterations
    assert len(opt.y_) == 20  # max_iter evaluations total

    # First point should be x0 (in internal scale)
    x0_internal = opt._transform_X(x0.reshape(1, -1)).ravel()
    assert np.allclose(opt.X_[0], x0_internal, atol=1e-6)


def test_x0_list_input():
    """Test that x0 can be provided as a list."""

    def objective(X):
        return np.sum(X**2, axis=1)

    x0 = [1.0, 2.0]  # List instead of numpy array

    opt = SpotOptim(
        fun=objective,
        bounds=[(-5, 5), (-5, 5)],
        x0=x0,
        max_iter=10,
        n_initial=5,
        seed=42,
        verbose=False,
    )

    result = opt.optimize()
    assert result.success is True


def test_x0_high_dimensional():
    """Test x0 with higher-dimensional problem."""

    def objective(X):
        return np.sum(X**2, axis=1)

    n_dim = 5
    x0 = np.ones(n_dim) * 0.5
    bounds = [(-5, 5)] * n_dim

    opt = SpotOptim(
        fun=objective,
        bounds=bounds,
        x0=x0,
        max_iter=30,
        n_initial=15,
        seed=42,
        verbose=False,
    )

    result = opt.optimize()
    assert result.success is True


def test_x0_with_custom_variable_names():
    """Test x0 with custom variable names."""

    def objective(X):
        return np.sum(X**2, axis=1)

    x0 = np.array([1.0, 2.0, 3.0])

    opt = SpotOptim(
        fun=objective,
        bounds=[(-5, 5), (-5, 5), (-5, 5)],
        x0=x0,
        var_name=["alpha", "beta", "gamma"],
        max_iter=15,
        n_initial=8,
        seed=42,
        verbose=False,
    )

    result = opt.optimize()
    assert result.success is True


def test_x0_boundary_values():
    """Test x0 at boundary values."""

    def objective(X):
        return np.sum(X**2, axis=1)

    # x0 at lower bounds
    x0_lower = np.array([-5.0, -5.0])

    opt1 = SpotOptim(
        fun=objective,
        bounds=[(-5, 5), (-5, 5)],
        x0=x0_lower,
        max_iter=10,
        n_initial=5,
        seed=42,
        verbose=False,
    )

    result1 = opt1.optimize()
    assert result1.success is True

    # x0 at upper bounds
    x0_upper = np.array([5.0, 5.0])

    opt2 = SpotOptim(
        fun=objective,
        bounds=[(-5, 5), (-5, 5)],
        x0=x0_upper,
        max_iter=10,
        n_initial=5,
        seed=42,
        verbose=False,
    )

    result2 = opt2.optimize()
    assert result2.success is True


def test_x0_validation_error_messages():
    """Test that validation error messages are helpful."""

    def objective(X):
        return np.sum(X**2, axis=1)

    # Test wrong dimensions error message
    x0_wrong_dim = np.array([1.0])

    with pytest.raises(ValueError) as exc_info:
        SpotOptim(
            fun=objective,
            bounds=[(-5, 5), (-5, 5)],
            x0=x0_wrong_dim,
            max_iter=10,
            n_initial=5,
        )

    error_msg = str(exc_info.value)
    assert "dimensions" in error_msg.lower()
    assert "expected 2" in error_msg

    # Test out of bounds error message
    x0_out_of_bounds = np.array([10.0, 2.0])

    with pytest.raises(ValueError) as exc_info:
        SpotOptim(
            fun=objective,
            bounds=[(-5, 5), (-5, 5)],
            x0=x0_out_of_bounds,
            max_iter=10,
            n_initial=5,
        )

    error_msg = str(exc_info.value)
    assert "outside bounds" in error_msg.lower()
    assert "10.0" in error_msg or "10" in error_msg
