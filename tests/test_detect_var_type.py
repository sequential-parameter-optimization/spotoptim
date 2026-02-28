# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for detect_var_type() method in SpotOptim."""

import numpy as np
from spotoptim import SpotOptim


class TestDetectVarTypeBasic:
    """Test basic functionality of detect_var_type()."""

    def test_single_factor_variable(self):
        """Test detection for a single factor variable."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[("red", "green", "blue")],
            max_iter=5,
            n_initial=3,
        )

        # Should auto-detect as factor
        assert opt.var_type == ["factor"]

        # Calling detect_var_type() directly should return same result
        detected = opt.detect_var_type()
        assert detected == ["factor"]

    def test_single_numeric_variable(self):
        """Test detection for a single numeric variable."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0.0, 10.0)],
            max_iter=5,
            n_initial=3,
        )

        # Should auto-detect as float
        assert opt.var_type == ["float"]

        # Calling detect_var_type() directly should return same result
        detected = opt.detect_var_type()
        assert detected == ["float"]

    def test_mixed_factor_and_numeric(self):
        """Test detection for mixed factor and numeric variables."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[
                (-5.0, 5.0),  # numeric
                ("low", "medium", "high"),  # factor
                (0.0, 100.0),  # numeric
            ],
            max_iter=5,
            n_initial=3,
        )

        # Should auto-detect correctly
        assert opt.var_type == ["float", "factor", "float"]

        # Calling detect_var_type() directly should return same result
        detected = opt.detect_var_type()
        assert detected == ["float", "factor", "float"]

    def test_multiple_factor_variables(self):
        """Test detection for multiple factor variables."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[
                ("red", "green", "blue"),  # factor
                ("small", "medium", "large"),  # factor
                ("yes", "no"),  # factor
            ],
            max_iter=5,
            n_initial=3,
        )

        # Should auto-detect all as factor
        assert opt.var_type == ["factor", "factor", "factor"]

        # Calling detect_var_type() directly should return same result
        detected = opt.detect_var_type()
        assert detected == ["factor", "factor", "factor"]

    def test_multiple_numeric_variables(self):
        """Test detection for multiple numeric variables."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[
                (-10.0, 10.0),
                (0.0, 1.0),
                (-100.0, 100.0),
            ],
            max_iter=5,
            n_initial=3,
        )

        # Should auto-detect all as float
        assert opt.var_type == ["float", "float", "float"]

        # Calling detect_var_type() directly should return same result
        detected = opt.detect_var_type()
        assert detected == ["float", "float", "float"]


class TestDetectVarTypeWithExplicitTypes:
    """Test that detect_var_type() respects explicitly provided var_type."""

    def test_explicit_var_type_not_overridden(self):
        """Test that explicit var_type is not overridden by detection."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[
                (0.0, 10.0),  # numeric bound
                ("red", "green"),  # factor bound
            ],
            var_type=["int", "factor"],  # Explicit types
            max_iter=5,
            n_initial=3,
        )

        # Should use explicit var_type, not auto-detected
        assert opt.var_type == ["int", "factor"]

    def test_explicit_float_for_integer_bounds(self):
        """Test explicit float type for integer-valued bounds."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0, 10), (0, 100)],
            var_type=["float", "float"],  # Explicit float
            max_iter=5,
            n_initial=3,
        )

        # Should use explicit var_type
        assert opt.var_type == ["float", "float"]

    def test_explicit_int_for_numeric_bounds(self):
        """Test explicit int type for numeric bounds."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0.0, 10.0), (-5.0, 5.0)],
            var_type=["int", "int"],  # Explicit int
            max_iter=5,
            n_initial=3,
        )

        # Should use explicit var_type
        assert opt.var_type == ["int", "int"]


class TestDetectVarTypeEdgeCases:
    """Test edge cases for detect_var_type()."""

    def test_two_level_factor(self):
        """Test detection for binary (two-level) factor variable."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[("yes", "no")],
            max_iter=5,
            n_initial=3,
        )

        # Should auto-detect as factor
        assert opt.var_type == ["factor"]
        detected = opt.detect_var_type()
        assert detected == ["factor"]

    def test_many_level_factor(self):
        """Test detection for factor variable with many levels."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[("a", "b", "c", "d", "e", "f", "g", "h", "i", "j")],
            max_iter=5,
            n_initial=3,
        )

        # Should auto-detect as factor
        assert opt.var_type == ["factor"]
        detected = opt.detect_var_type()
        assert detected == ["factor"]

    def test_factor_with_special_characters(self):
        """Test detection for factor variables with special characters."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[("option-1", "option_2", "option.3")],
            max_iter=5,
            n_initial=3,
        )

        # Should auto-detect as factor
        assert opt.var_type == ["factor"]
        detected = opt.detect_var_type()
        assert detected == ["factor"]

    def test_integer_bounds(self):
        """Test detection for integer bounds (should be float by default)."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0, 10)],
            max_iter=5,
            n_initial=3,
        )

        # Should auto-detect as float (not int)
        assert opt.var_type == ["float"]
        detected = opt.detect_var_type()
        assert detected == ["float"]


class TestDetectVarTypeConsistency:
    """Test consistency of detect_var_type() across multiple calls."""

    def test_multiple_calls_same_result(self):
        """Test that multiple calls to detect_var_type() return consistent results."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[
                (0.0, 10.0),
                ("a", "b", "c"),
                (-5.0, 5.0),
            ],
            max_iter=5,
            n_initial=3,
        )

        # Call detect_var_type() multiple times
        result1 = opt.detect_var_type()
        result2 = opt.detect_var_type()
        result3 = opt.detect_var_type()

        # All results should be identical
        assert result1 == result2 == result3
        assert result1 == ["float", "factor", "float"]

    def test_result_matches_attribute(self):
        """Test that detect_var_type() result matches var_type attribute."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[
                ("red", "green", "blue"),
                (0.0, 100.0),
                ("low", "high"),
            ],
            max_iter=5,
            n_initial=3,
        )

        # detect_var_type() should match var_type attribute
        detected = opt.detect_var_type()
        assert detected == opt.var_type
        assert detected == ["factor", "float", "factor"]


class TestDetectVarTypeReturnType:
    """Test return type and structure of detect_var_type()."""

    def test_returns_list(self):
        """Test that detect_var_type() returns a list."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0.0, 10.0), ("a", "b")],
            max_iter=5,
            n_initial=3,
        )

        result = opt.detect_var_type()
        assert isinstance(result, list)

    def test_list_length_matches_dimensions(self):
        """Test that returned list length matches number of dimensions."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[
                (0.0, 10.0),
                ("a", "b", "c"),
                (-5.0, 5.0),
                ("x", "y"),
            ],
            max_iter=5,
            n_initial=3,
        )

        result = opt.detect_var_type()
        assert len(result) == 4
        assert len(result) == opt.n_dim

    def test_list_contains_valid_types(self):
        """Test that returned list contains only valid type strings."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[
                (0.0, 10.0),
                ("a", "b"),
                (-5.0, 5.0),
            ],
            max_iter=5,
            n_initial=3,
        )

        result = opt.detect_var_type()
        # With auto-detection, should only return 'float' or 'factor'
        valid_types = {"float", "factor"}
        assert all(vtype in valid_types for vtype in result)


class TestDetectVarTypeComplexScenarios:
    """Test detect_var_type() in complex scenarios."""

    def test_alternating_factor_numeric(self):
        """Test detection for alternating factor and numeric variables."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[
                ("a", "b"),  # factor
                (0.0, 10.0),  # numeric
                ("x", "y", "z"),  # factor
                (-5.0, 5.0),  # numeric
                ("low", "high"),  # factor
            ],
            max_iter=5,
            n_initial=3,
        )

        expected = ["factor", "float", "factor", "float", "factor"]
        assert opt.var_type == expected
        assert opt.detect_var_type() == expected

    def test_after_dimension_reduction(self):
        """Test detect_var_type() behavior with dimension reduction (fixed bounds)."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[
                (0.0, 10.0),
                (5.0, 5.0),  # Fixed dimension
                ("a", "b"),
            ],
            max_iter=5,
            n_initial=3,
        )

        # var_type should reflect all original dimensions (before reduction)
        # This is set during __init__ before dimension reduction
        assert len(opt.var_type) == len(opt.bounds)

    def test_verbose_output(self):
        """Test detect_var_type() with verbose mode (no direct effect expected)."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[
                (0.0, 10.0),
                ("a", "b", "c"),
            ],
            max_iter=5,
            n_initial=3,
            verbose=True,
        )

        # Should still work correctly with verbose=True
        result = opt.detect_var_type()
        assert result == ["float", "factor"]


class TestDetectVarTypeIntegration:
    """Integration tests for detect_var_type() with optimization workflow."""

    def test_used_in_initialization(self):
        """Test that detect_var_type() is properly used during initialization."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[
                (0.0, 10.0),
                ("red", "green", "blue"),
            ],
            max_iter=10,
            n_initial=5,
        )

        # var_type should be set from detect_var_type()
        assert opt.var_type == ["float", "factor"]

        # Verify factor mapping was created
        assert 1 in opt._factor_maps

    def test_affects_bounds_processing(self):
        """Test that detected var_type affects bounds processing."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[
                (0.0, 10.0),
                ("a", "b", "c"),
            ],
            max_iter=5,
            n_initial=3,
        )

        # Factor variable bounds should be converted to integer range
        assert opt.bounds[0] == (0.0, 10.0)  # Float unchanged
        assert opt.bounds[1] == (0, 2)  # Factor converted to (0, n_levels-1)

    def test_optimization_runs_successfully(self):
        """Test that optimization runs successfully with auto-detected types."""

        # Use a function that handles factor variables by taking only the first column
        def objective(X):
            # For mixed factor/numeric, only sum numeric columns
            return np.sum(X[:, 0:1].astype(float) ** 2, axis=1)

        opt = SpotOptim(
            fun=objective,
            bounds=[
                (-5.0, 5.0),
                ("option1", "option2", "option3"),
            ],
            max_iter=10,
            n_initial=5,
            seed=42,
        )

        # Should complete without errors
        result = opt.optimize()
        assert result is not None
        assert hasattr(result, "x")
        assert hasattr(result, "fun")


class TestDetectVarTypeDocstringExample:
    """Validate the living code example in the detect_var_type docstring."""

    def test_example_detect_var_type(self):
        """Test the example code provided in the docstring."""
        from spotoptim import SpotOptim

        # Define a simple objective mapping names to values for demonstration
        def objective(X):
            # X has shape (n_samples, n_dimensions)
            return X[:, 0] + X[:, 1]

        # The first dimension has factor levels ('red', 'green', 'blue')
        # The second dimension is continuous bounds (0, 10)
        spot = SpotOptim(fun=objective, bounds=[("red", "green", "blue"), (0, 10)])
        detected_types = spot.detect_var_type()

        # Verify the auto-detected types match the example output
        assert detected_types == ["factor", "float"]
        assert spot.var_type == ["factor", "float"]
