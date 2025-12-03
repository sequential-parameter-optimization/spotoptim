"""Tests for the normalize_X function."""

import numpy as np
import pytest
from spotoptim.utils.stats import normalize_X


class TestNormalizeXBasic:
    """Test basic normalize_X functionality."""

    def test_normalize_x_normal_case(self):
        """Test normalization with varying values."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = normalize_X(X)
        
        # Check shape
        assert result.shape == X.shape
        
        # Check values
        np.testing.assert_array_almost_equal(result[0], [0.0, 0.0])
        np.testing.assert_array_almost_equal(result[1], [0.5, 0.5])
        np.testing.assert_array_almost_equal(result[2], [1.0, 1.0])

    def test_normalize_x_constant_dimensions(self):
        """Test normalization when all values in a dimension are constant."""
        X = np.array([[1.0, 5.0], [1.0, 5.0], [1.0, 5.0]])
        result = normalize_X(X)
        
        # Constant dimensions should be 0.5
        expected = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_normalize_x_mixed_dimensions(self):
        """Test normalization with one constant and one varying dimension."""
        X = np.array([[1.0, 2.0], [1.0, 4.0], [1.0, 6.0]])
        result = normalize_X(X)
        
        # First dimension constant (0.5), second dimension varies [0, 0.5, 1]
        expected = np.array([[0.5, 0.0], [0.5, 0.5], [0.5, 1.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_normalize_x_single_point(self):
        """Test normalization with a single point (all dimensions constant)."""
        X = np.array([[1.0, 2.0, 3.0]])
        result = normalize_X(X)
        
        # All dimensions should be 0.5
        expected = np.array([[0.5, 0.5, 0.5]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_normalize_x_two_points_identical(self):
        """Test normalization with two identical points."""
        X = np.array([[1.0, 2.0], [1.0, 2.0]])
        result = normalize_X(X)
        
        # All dimensions constant
        expected = np.array([[0.5, 0.5], [0.5, 0.5]])
        np.testing.assert_array_almost_equal(result, expected)


class TestNormalizeXEdgeCases:
    """Test edge cases for normalize_X."""

    def test_normalize_x_very_small_range(self):
        """Test normalization with very small range (near machine epsilon)."""
        X = np.array([[1.0, 2.0], [1.0 + 1e-15, 2.0 + 1e-15]])
        result = normalize_X(X)
        
        # Should be treated as constant
        expected = np.array([[0.5, 0.5], [0.5, 0.5]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_normalize_x_custom_eps(self):
        """Test normalization with custom epsilon value."""
        X = np.array([[1.0, 2.0], [1.001, 2.001]])
        
        # With large eps, should be treated as constant
        result = normalize_X(X, eps=0.01)
        expected = np.array([[0.5, 0.5], [0.5, 0.5]])
        np.testing.assert_array_almost_equal(result, expected)
        
        # With small eps, should normalize
        result = normalize_X(X, eps=1e-12)
        assert result[0, 0] == 0.0
        assert result[1, 0] == 1.0

    def test_normalize_x_negative_values(self):
        """Test normalization with negative values."""
        X = np.array([[-5.0, -2.0], [-1.0, 0.0], [3.0, 2.0]])
        result = normalize_X(X)
        
        # Check that min becomes 0 and max becomes 1
        assert result.min(axis=0)[0] == pytest.approx(0.0)
        assert result.max(axis=0)[0] == pytest.approx(1.0)
        assert result.min(axis=0)[1] == pytest.approx(0.0)
        assert result.max(axis=0)[1] == pytest.approx(1.0)

    def test_normalize_x_zero_values(self):
        """Test normalization with zero values."""
        X = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0]])
        result = normalize_X(X)
        
        np.testing.assert_array_almost_equal(result[0], [0.0, 0.0])
        np.testing.assert_array_almost_equal(result[1], [0.5, 0.5])
        np.testing.assert_array_almost_equal(result[2], [1.0, 1.0])


class TestNormalizeXDimensions:
    """Test normalize_X with different dimensions."""

    def test_normalize_x_1d(self):
        """Test normalization with 1D array (single dimension)."""
        X = np.array([[1.0], [3.0], [5.0]])
        result = normalize_X(X)
        
        expected = np.array([[0.0], [0.5], [1.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_normalize_x_high_dimensional(self):
        """Test normalization with high-dimensional data."""
        X = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 4.0, 6.0, 8.0, 10.0],
            [3.0, 6.0, 9.0, 12.0, 15.0]
        ])
        result = normalize_X(X)
        
        # Check shape
        assert result.shape == X.shape
        
        # Check that each dimension is normalized
        for dim in range(X.shape[1]):
            assert result[:, dim].min() == pytest.approx(0.0)
            assert result[:, dim].max() == pytest.approx(1.0)

    def test_normalize_x_many_points(self):
        """Test normalization with many points."""
        n_points = 1000
        X = np.random.randn(n_points, 3) * 10 + 5
        result = normalize_X(X)
        
        # Check shape
        assert result.shape == X.shape
        
        # Check that each dimension is normalized to [0, 1]
        for dim in range(X.shape[1]):
            assert result[:, dim].min() >= 0.0
            assert result[:, dim].max() <= 1.0
            assert result[:, dim].min() == pytest.approx(0.0)
            assert result[:, dim].max() == pytest.approx(1.0)


class TestNormalizeXReturnTypes:
    """Test return types of normalize_X."""

    def test_normalize_x_returns_ndarray(self):
        """Test that normalize_X returns numpy ndarray."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = normalize_X(X)
        assert isinstance(result, np.ndarray)

    def test_normalize_x_returns_float_dtype(self):
        """Test that normalize_X returns float dtype."""
        X = np.array([[1, 2], [3, 4]])
        result = normalize_X(X)
        assert np.issubdtype(result.dtype, np.floating)

    def test_normalize_x_preserves_shape(self):
        """Test that normalize_X preserves input shape."""
        shapes = [(5, 2), (10, 3), (3, 5), (100, 10)]
        for shape in shapes:
            X = np.random.randn(*shape)
            result = normalize_X(X)
            assert result.shape == X.shape


class TestNormalizeXNumericalProperties:
    """Test numerical properties of normalize_X."""

    def test_normalize_x_bounds(self):
        """Test that normalized values are in [0, 1] (or 0.5 for constant)."""
        X = np.random.randn(50, 5) * 100
        result = normalize_X(X)
        
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_normalize_x_monotonicity(self):
        """Test that normalization preserves order within dimensions."""
        X = np.array([[1.0, 5.0], [2.0, 3.0], [3.0, 1.0], [4.0, 2.0]])
        result = normalize_X(X)
        
        # First dimension: [1, 2, 3, 4] -> [0, 0.33, 0.67, 1]
        # Order should be preserved
        assert result[0, 0] < result[1, 0] < result[2, 0] < result[3, 0]
        
        # Second dimension: [5, 3, 1, 2] -> [1, 0.5, 0, 0.25]
        # Order should be preserved
        assert result[2, 1] < result[3, 1] < result[1, 1] < result[0, 1]

    def test_normalize_x_linearity(self):
        """Test that normalization is linear."""
        X = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
        result = normalize_X(X)
        
        # Differences should be equal (linearity)
        diff1 = result[1, 0] - result[0, 0]
        diff2 = result[2, 0] - result[1, 0]
        diff3 = result[3, 0] - result[2, 0]
        
        assert diff1 == pytest.approx(diff2)
        assert diff2 == pytest.approx(diff3)


class TestNormalizeXSpecialCases:
    """Test special cases for normalize_X."""

    def test_normalize_x_already_normalized(self):
        """Test normalization of already normalized data."""
        X = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        result = normalize_X(X)
        
        # Should return the same values
        np.testing.assert_array_almost_equal(result, X)

    def test_normalize_x_uniform_spacing(self):
        """Test normalization with uniformly spaced points."""
        X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
        result = normalize_X(X)
        
        expected = np.array([[0.0], [0.25], [0.5], [0.75], [1.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_normalize_x_outliers(self):
        """Test normalization with outliers."""
        X = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [100.0, 4.0]])
        result = normalize_X(X)
        
        # Check that outlier is mapped to 1.0
        assert result[3, 0] == pytest.approx(1.0)
        assert result[3, 1] == pytest.approx(1.0)
        
        # Check that other values are compressed
        assert result[0, 0] == pytest.approx(0.0)
        assert result[1, 0] < 0.1  # Close to 0 due to outlier


class TestNormalizeXInputValidation:
    """Test input validation for normalize_X."""

    def test_normalize_x_with_list_input(self):
        """Test that normalize_X works with list input."""
        X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        result = normalize_X(np.array(X))
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 2)

    def test_normalize_x_with_integer_input(self):
        """Test that normalize_X works with integer input."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        result = normalize_X(X)
        
        # Should convert to float and normalize
        assert np.issubdtype(result.dtype, np.floating)
        np.testing.assert_array_almost_equal(result[0], [0.0, 0.0])
        np.testing.assert_array_almost_equal(result[2], [1.0, 1.0])


class TestNormalizeXDocstring:
    """Test examples from the docstring."""

    def test_docstring_example_1(self):
        """Test the first example from docstring."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        result = normalize_X(X)
        
        expected = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_docstring_example_2(self):
        """Test the constant dimension example from docstring."""
        X_const = np.array([[1, 5], [1, 5], [1, 5]])
        result = normalize_X(X_const)
        
        expected = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_empty_array(self):
        """Test normalization with empty array."""
        X_empty = np.zeros((0, 2))
        result = normalize_X(X_empty)
        
        # Should return empty array with same shape
        assert result.shape == (0, 2)
        assert result.size == 0
