# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for modify_bounds_based_on_var_type() method in SpotOptim."""

import numpy as np
import pytest
from spotoptim import SpotOptim


class TestModifyBoundsBasedOnVarTypeBasic:
    """Test basic functionality of modify_bounds_based_on_var_type()."""

    def test_int_type_ceiling_floor(self):
        """Test that int type applies ceiling to lower and floor to upper bounds."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0.5, 10.5)],
            var_type=["int"],
            max_iter=5,
            n_initial=3,
        )

        # Should apply ceiling to lower (0.5 -> 1) and floor to upper (10.5 -> 10)
        assert opt.bounds[0] == (1, 10)
        assert isinstance(opt.bounds[0][0], (int, np.integer))
        assert isinstance(opt.bounds[0][1], (int, np.integer))

    def test_float_type_explicit_conversion(self):
        """Test that float type explicitly converts bounds to float."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0, 10)],
            var_type=["float"],
            max_iter=5,
            n_initial=3,
        )

        # Should convert to float
        assert opt.bounds[0] == (0.0, 10.0)
        assert isinstance(opt.bounds[0][0], float)
        assert isinstance(opt.bounds[0][1], float)

    def test_factor_type_unchanged(self):
        """Test that factor type leaves bounds unchanged (already processed)."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[("red", "green", "blue")],
            max_iter=5,
            n_initial=3,
        )

        # Factor bounds should already be converted to (0, n_levels-1) by process_factor_bounds
        assert opt.bounds[0] == (0, 2)
        assert isinstance(opt.bounds[0][0], (int, np.integer))
        assert isinstance(opt.bounds[0][1], (int, np.integer))

    def test_mixed_types(self):
        """Test modification with mixed variable types."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[
                (0.5, 10.5),  # int
                (0.0, 1.0),  # float
                ("low", "medium", "high"),  # factor
            ],
            var_type=["int", "float", "factor"],
            max_iter=5,
            n_initial=3,
        )

        # Check each bound type
        assert opt.bounds[0] == (1, 10)  # int: ceiling/floor
        assert opt.bounds[1] == (0.0, 1.0)  # float: explicit conversion
        assert opt.bounds[2] == (0, 2)  # factor: unchanged (already 0 to n_levels-1)


class TestModifyBoundsBasedOnVarTypeIntegerBounds:
    """Test integer bound modifications."""

    def test_int_type_with_exact_integers(self):
        """Test int type with bounds that are already exact integers."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(1.0, 10.0)],
            var_type=["int"],
            max_iter=5,
            n_initial=3,
        )

        # Should convert to int
        assert opt.bounds[0] == (1, 10)
        assert isinstance(opt.bounds[0][0], (int, np.integer))
        assert isinstance(opt.bounds[0][1], (int, np.integer))

    def test_int_type_with_small_fractional_parts(self):
        """Test int type with small fractional parts."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0.1, 9.9)],
            var_type=["int"],
            max_iter=5,
            n_initial=3,
        )

        # Should apply ceiling to 0.1 -> 1, floor to 9.9 -> 9
        assert opt.bounds[0] == (1, 9)

    def test_int_type_with_negative_bounds(self):
        """Test int type with negative bounds."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5.7, -0.3)],
            var_type=["int"],
            max_iter=5,
            n_initial=3,
        )

        # Should apply ceiling to -5.7 -> -5, floor to -0.3 -> -1
        assert opt.bounds[0] == (-5, -1)

    def test_int_type_narrow_range(self):
        """Test int type with very narrow range."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(3.2, 5.8)],
            var_type=["int"],
            max_iter=5,
            n_initial=3,
        )

        # Should apply ceiling to 3.2 -> 4, floor to 5.8 -> 5
        assert opt.bounds[0] == (4, 5)


class TestModifyBoundsBasedOnVarTypeFloatBounds:
    """Test float bound modifications."""

    def test_float_type_with_integer_input(self):
        """Test float type with integer input bounds."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0, 100)],
            var_type=["float"],
            max_iter=5,
            n_initial=3,
        )

        # Should convert to float
        assert opt.bounds[0] == (0.0, 100.0)
        assert isinstance(opt.bounds[0][0], float)
        assert isinstance(opt.bounds[0][1], float)

    def test_float_type_preserves_precision(self):
        """Test float type preserves precision."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0.123456789, 9.876543210)],
            var_type=["float"],
            max_iter=5,
            n_initial=3,
        )

        # Should preserve float precision
        assert opt.bounds[0] == (0.123456789, 9.876543210)

    def test_float_type_with_negative_bounds(self):
        """Test float type with negative bounds."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-100.5, -0.5)],
            var_type=["float"],
            max_iter=5,
            n_initial=3,
        )

        # Should preserve as float
        assert opt.bounds[0] == (-100.5, -0.5)
        assert isinstance(opt.bounds[0][0], float)
        assert isinstance(opt.bounds[0][1], float)


class TestModifyBoundsBasedOnVarTypeFactorBounds:
    """Test factor bound modifications."""

    def test_factor_type_binary(self):
        """Test factor type with binary (two-level) variable."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[("yes", "no")],
            max_iter=5,
            n_initial=3,
        )

        # Factor should be converted to (0, 1)
        assert opt.bounds[0] == (0, 1)
        assert opt.var_type[0] == "factor"

    def test_factor_type_multiple_levels(self):
        """Test factor type with multiple levels."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[("red", "green", "blue", "yellow", "orange")],
            max_iter=5,
            n_initial=3,
        )

        # Factor should be converted to (0, 4)
        assert opt.bounds[0] == (0, 4)
        assert opt.var_type[0] == "factor"

    def test_factor_type_maintains_integer_bounds(self):
        """Test that factor bounds remain as integers."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[("a", "b", "c")],
            max_iter=5,
            n_initial=3,
        )

        # Bounds should be integers (from process_factor_bounds)
        assert isinstance(opt.bounds[0][0], (int, np.integer))
        assert isinstance(opt.bounds[0][1], (int, np.integer))


class TestModifyBoundsBasedOnVarTypeMultipleDimensions:
    """Test modifications with multiple dimensions."""

    def test_all_int_type(self):
        """Test all dimensions with int type."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[
                (0.5, 10.5),
                (1.2, 5.8),
                (-3.7, 2.3),
            ],
            var_type=["int", "int", "int"],
            max_iter=5,
            n_initial=3,
        )

        assert opt.bounds[0] == (1, 10)
        assert opt.bounds[1] == (2, 5)
        assert opt.bounds[2] == (-3, 2)

    def test_all_float_type(self):
        """Test all dimensions with float type."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[
                (0, 10),
                (1, 5),
                (-3, 2),
            ],
            var_type=["float", "float", "float"],
            max_iter=5,
            n_initial=3,
        )

        assert opt.bounds[0] == (0.0, 10.0)
        assert opt.bounds[1] == (1.0, 5.0)
        assert opt.bounds[2] == (-3.0, 2.0)

    def test_alternating_types(self):
        """Test alternating variable types."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[
                (0.5, 10.5),  # int
                (0.0, 1.0),  # float
                ("a", "b"),  # factor
                (5, 15),  # int
                (-1.0, 1.0),  # float
            ],
            var_type=["int", "float", "factor", "int", "float"],
            max_iter=5,
            n_initial=3,
        )

        assert opt.bounds[0] == (1, 10)
        assert opt.bounds[1] == (0.0, 1.0)
        assert opt.bounds[2] == (0, 1)
        assert opt.bounds[3] == (5, 15)
        assert opt.bounds[4] == (-1.0, 1.0)


class TestModifyBoundsBasedOnVarTypeErrorHandling:
    """Test error handling in modify_bounds_based_on_var_type()."""

    def test_unsupported_var_type(self):
        """Test that unsupported var_type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported var_type 'string'"):
            SpotOptim(
                fun=lambda X: np.sum(X**2, axis=1),
                bounds=[(0.0, 10.0)],
                var_type=["string"],
                max_iter=5,
                n_initial=3,
            )

    def test_invalid_var_type(self):
        """Test that invalid var_type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported var_type 'invalid'"):
            SpotOptim(
                fun=lambda X: np.sum(X**2, axis=1),
                bounds=[(0.0, 10.0)],
                var_type=["invalid"],
                max_iter=5,
                n_initial=3,
            )

    def test_mixed_valid_invalid_types(self):
        """Test that one invalid type in a list raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported var_type"):
            SpotOptim(
                fun=lambda X: np.sum(X**2, axis=1),
                bounds=[
                    (0.0, 10.0),
                    (0.0, 1.0),
                    (-5.0, 5.0),
                ],
                var_type=["int", "unknown", "float"],
                max_iter=5,
                n_initial=3,
            )


class TestModifyBoundsBasedOnVarTypeAutoDetection:
    """Test interaction with auto-detection of variable types."""

    def test_auto_detected_factor_type(self):
        """Test that auto-detected factor types are properly modified."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[
                (0.0, 10.0),
                ("red", "green", "blue"),
            ],
            # No var_type specified, should auto-detect
            max_iter=5,
            n_initial=3,
        )

        # First should be float, second should be factor
        assert opt.var_type == ["float", "factor"]
        assert opt.bounds[0] == (0.0, 10.0)
        assert opt.bounds[1] == (0, 2)

    def test_auto_detected_numeric_types(self):
        """Test that auto-detected numeric types default to float."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[
                (0, 10),
                (-5, 5),
                (0.5, 9.5),
            ],
            # No var_type specified, should auto-detect as float
            max_iter=5,
            n_initial=3,
        )

        # All should be auto-detected as float
        assert opt.var_type == ["float", "float", "float"]
        assert opt.bounds[0] == (0.0, 10.0)
        assert opt.bounds[1] == (-5.0, 5.0)
        assert opt.bounds[2] == (0.5, 9.5)


class TestModifyBoundsBasedOnVarTypeIntegration:
    """Integration tests for modify_bounds_based_on_var_type()."""

    def test_bounds_affect_lower_upper_arrays(self):
        """Test that modified bounds correctly update lower and upper arrays."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[
                (0.5, 10.5),  # int -> (1, 10)
                (0.0, 1.0),  # float -> (0.0, 1.0)
            ],
            var_type=["int", "float"],
            max_iter=5,
            n_initial=3,
        )

        # Check lower and upper arrays
        np.testing.assert_array_equal(opt.lower, np.array([1, 0.0]))
        np.testing.assert_array_equal(opt.upper, np.array([10, 1.0]))

    def test_optimization_runs_with_modified_bounds(self):
        """Test that optimization runs successfully with modified bounds."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[
                (0.5, 10.5),  # int
                (0.0, 1.0),  # float
            ],
            var_type=["int", "float"],
            max_iter=10,
            n_initial=5,
            seed=42,
        )

        result = opt.optimize()
        assert result is not None
        assert hasattr(result, "x")
        assert hasattr(result, "fun")

    def test_bounds_modification_before_transformations(self):
        """Test that bounds are modified before transformations are applied."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(1, 100)],
            var_type=["float"],
            var_trans=["log10"],
            max_iter=5,
            n_initial=3,
        )

        # Bounds should first be converted to float (1.0, 100.0)
        # Then log10 transformation applied: (0.0, 2.0)
        assert opt.bounds[0] == (0.0, 2.0)


class TestModifyBoundsBasedOnVarTypeCallability:
    """Test that modify_bounds_based_on_var_type() can be called directly."""

    def test_direct_method_call(self):
        """Test calling modify_bounds_based_on_var_type() directly."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0.0, 10.0)],
            var_type=["float"],
            max_iter=5,
            n_initial=3,
        )

        # Method should have been called during __init__
        assert opt.bounds[0] == (0.0, 10.0)

        # Manually modify bounds and var_type
        opt.bounds = [(0.5, 10.5)]
        opt.var_type = ["int"]

        # Call method directly
        opt.modify_bounds_based_on_var_type()

        # Should now be integer bounds
        assert opt.bounds[0] == (1, 10)


class TestModifyBoundsBasedOnVarTypeEdgeCases:
    """Test edge cases for modify_bounds_based_on_var_type()."""

    def test_int_type_close_bounds(self):
        """Test int type with bounds that round to adjacent integers."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(4.2, 6.8)],
            var_type=["int"],
            max_iter=5,
            n_initial=3,
        )

        # ceiling(4.2) = 5, floor(6.8) = 6
        # Valid range with two integer values
        assert opt.bounds[0] == (5, 6)

    def test_float_type_with_small_range(self):
        """Test float type with small range."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0, 0.1)],
            var_type=["float"],
            max_iter=5,
            n_initial=3,
        )

        assert opt.bounds[0] == (0.0, 0.1)
        assert isinstance(opt.bounds[0][0], float)
        assert isinstance(opt.bounds[0][1], float)

    def test_int_type_with_large_numbers(self):
        """Test int type with large numbers."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(1e6 + 0.5, 1e9 + 0.5)],
            var_type=["int"],
            max_iter=5,
            n_initial=3,
        )

        assert opt.bounds[0] == (1000001, 1000000000)
        assert isinstance(opt.bounds[0][0], (int, np.integer))
        assert isinstance(opt.bounds[0][1], (int, np.integer))
