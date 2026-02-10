# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for process_factor_bounds() method in SpotOptim."""

import numpy as np
import pytest
from spotoptim import SpotOptim


class TestProcessFactorBoundsBasic:
    """Test basic functionality of process_factor_bounds()."""

    def test_single_factor_variable(self):
        """Test processing a single factor variable."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[("red", "green", "blue")],
            max_iter=5,
            n_initial=3,
        )

        # Check factor mapping was created
        assert 0 in opt._factor_maps
        assert opt._factor_maps[0] == {0: "red", 1: "green", 2: "blue"}

        # Check bounds were converted to integer range
        assert opt.bounds[0] == (0, 2)

    def test_single_numeric_variable(self):
        """Test processing a single numeric variable (no factor)."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0.0, 10.0)],
            max_iter=5,
            n_initial=3,
        )

        # No factor mappings should exist
        assert len(opt._factor_maps) == 0

        # Bounds should remain unchanged
        assert opt.bounds[0] == (0.0, 10.0)

    def test_mixed_factor_and_numeric(self):
        """Test processing mixed factor and numeric variables."""
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

        # Only dimension 1 should have factor mapping
        assert 0 not in opt._factor_maps
        assert 1 in opt._factor_maps
        assert 2 not in opt._factor_maps

        # Check mapping
        assert opt._factor_maps[1] == {0: "low", 1: "medium", 2: "high"}

        # Check bounds
        assert opt.bounds[0] == (-5.0, 5.0)
        assert opt.bounds[1] == (0, 2)
        assert opt.bounds[2] == (0.0, 100.0)

    def test_multiple_factor_variables(self):
        """Test processing multiple factor variables."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[
                ("a", "b"),  # factor with 2 levels
                ("x", "y", "z"),  # factor with 3 levels
                ("alpha", "beta", "gamma", "delta"),  # factor with 4 levels
            ],
            max_iter=5,
            n_initial=3,
        )

        # All dimensions should have factor mappings
        assert 0 in opt._factor_maps
        assert 1 in opt._factor_maps
        assert 2 in opt._factor_maps

        # Check mappings
        assert opt._factor_maps[0] == {0: "a", 1: "b"}
        assert opt._factor_maps[1] == {0: "x", 1: "y", 2: "z"}
        assert opt._factor_maps[2] == {0: "alpha", 1: "beta", 2: "gamma", 3: "delta"}

        # Check bounds
        assert opt.bounds[0] == (0, 1)
        assert opt.bounds[1] == (0, 2)
        assert opt.bounds[2] == (0, 3)


class TestProcessFactorBoundsEdgeCases:
    """Test edge cases for process_factor_bounds()."""

    def test_two_level_factor(self):
        """Test factor variable with exactly 2 levels (binary factor)."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[("yes", "no")],
            max_iter=5,
            n_initial=3,
        )

        assert opt._factor_maps[0] == {0: "yes", 1: "no"}
        assert opt.bounds[0] == (0, 1)

    def test_many_level_factor(self):
        """Test factor variable with many levels."""
        levels = tuple(f"level_{i}" for i in range(20))
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[levels],
            max_iter=5,
            n_initial=3,
        )

        assert len(opt._factor_maps[0]) == 20
        assert opt.bounds[0] == (0, 19)

        # Check first and last mappings
        assert opt._factor_maps[0][0] == "level_0"
        assert opt._factor_maps[0][19] == "level_19"

    def test_integer_bounds(self):
        """Test numeric bounds with integers."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0, 10), (5, 15)],
            max_iter=5,
            n_initial=3,
        )

        # No factor mappings
        assert len(opt._factor_maps) == 0

    def test_float_bounds(self):
        """Test numeric bounds with floats."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0.1, 10.5), (-3.14, 3.14)],
            max_iter=5,
            n_initial=3,
        )

        # No factor mappings
        assert len(opt._factor_maps) == 0

    def test_mixed_int_float_bounds(self):
        """Test numeric bounds with mixed int and float."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0, 10.5), (5.5, 15)],
            max_iter=5,
            n_initial=3,
        )

        # No factor mappings
        assert len(opt._factor_maps) == 0

    def test_list_bounds(self):
        """Test that list bounds are also accepted."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[
                [0.0, 10.0],  # list for numeric
                ["a", "b", "c"],  # list for factor
            ],
            max_iter=5,
            n_initial=3,
        )

        # Factor mapping should be created for dimension 1
        assert 1 in opt._factor_maps
        assert opt._factor_maps[1] == {0: "a", 1: "b", 2: "c"}

    def test_factor_preserves_order(self):
        """Test that factor level order is preserved."""
        levels = ("zebra", "apple", "mango", "banana")
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[levels],
            max_iter=5,
            n_initial=3,
        )

        # Order should be preserved (not alphabetical)
        assert opt._factor_maps[0][0] == "zebra"
        assert opt._factor_maps[0][1] == "apple"
        assert opt._factor_maps[0][2] == "mango"
        assert opt._factor_maps[0][3] == "banana"


class TestProcessFactorBoundsValidation:
    """Test validation and error handling in process_factor_bounds()."""

    def test_invalid_empty_bound(self):
        """Test that empty bounds raise an error."""
        with pytest.raises(ValueError, match="Invalid bound"):
            SpotOptim(
                fun=lambda X: np.sum(X**2, axis=1),
                bounds=[()],
                max_iter=5,
                n_initial=3,
            )

    def test_invalid_single_numeric(self):
        """Test that single numeric value raises an error (expected tuple)."""
        with pytest.raises((ValueError, TypeError)):
            SpotOptim(
                fun=lambda X: np.sum(X**2, axis=1),
                bounds=[5],  # Single value, not a tuple
                max_iter=5,
                n_initial=3,
            )

    def test_invalid_mixed_strings_numbers(self):
        """Test that mixed strings and numbers in a bound raise an error."""
        with pytest.raises(ValueError, match="Invalid bound"):
            SpotOptim(
                fun=lambda X: np.sum(X**2, axis=1),
                bounds=[("a", 1, "b")],  # Mixed types
                max_iter=5,
                n_initial=3,
            )

    def test_invalid_three_numeric_values(self):
        """Test that three numeric values raise an error."""
        with pytest.raises(ValueError, match="Invalid bound"):
            SpotOptim(
                fun=lambda X: np.sum(X**2, axis=1),
                bounds=[(0, 5, 10)],  # Three numeric values
                max_iter=5,
                n_initial=3,
            )

    def test_invalid_none_bound(self):
        """Test that None as bound raises an error."""
        with pytest.raises((ValueError, TypeError)):
            SpotOptim(
                fun=lambda X: np.sum(X**2, axis=1),
                bounds=[None],
                max_iter=5,
                n_initial=3,
            )


class TestProcessFactorBoundsVerbose:
    """Test verbose output of process_factor_bounds()."""

    def test_verbose_output(self, capsys):
        """Test that verbose mode prints factor information."""
        _ = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[("a", "b", "c")],
            max_iter=5,
            n_initial=3,
            verbose=True,
        )

        captured = capsys.readouterr()
        assert "Factor variable at dimension 0" in captured.out
        assert "Levels: ['a', 'b', 'c']" in captured.out
        assert "Mapped to integers: 0 to 2" in captured.out

    def test_no_verbose_output(self, capsys):
        """Test that non-verbose mode doesn't print factor information."""
        _ = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[("a", "b", "c")],
            max_iter=5,
            n_initial=3,
            verbose=False,
        )
        captured = capsys.readouterr()
        assert "Factor variable" not in captured.out


class TestProcessFactorBoundsIntegration:
    """Test integration of process_factor_bounds() with other components."""

    def test_var_type_auto_detection_with_factors(self):
        """Test that var_type is correctly auto-detected for factors."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[
                (0.0, 10.0),  # should be float
                ("low", "high"),  # should be factor
                (1, 100),  # should be float (default)
            ],
            max_iter=5,
            n_initial=3,
        )

        assert opt.var_type[0] == "float"
        assert opt.var_type[1] == "factor"
        assert opt.var_type[2] == "float"

    def test_n_dim_correct_with_factors(self):
        """Test that n_dim is correctly set with factor variables."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[
                ("a", "b"),
                (0.0, 10.0),
                ("x", "y", "z"),
            ],
            max_iter=5,
            n_initial=3,
        )

        assert opt.n_dim == 3

    def test_lower_upper_arrays_with_factors(self):
        """Test that lower and upper arrays are correctly set with factors."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[
                ("a", "b", "c"),  # factor: (0, 2)
                (5.0, 15.0),  # numeric
            ],
            max_iter=5,
            n_initial=3,
        )

        # Factor bounds should be 0 to n_levels-1
        assert opt.lower[0] == 0
        assert opt.upper[0] == 2

        # Numeric bounds should be as specified
        assert opt.lower[1] == 5.0
        assert opt.upper[1] == 15.0

    def test_original_bounds_preserved(self):
        """Test that original bounds are preserved in _original_bounds."""
        original = [("a", "b", "c"), (0.0, 10.0)]
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=original.copy(),
            max_iter=5,
            n_initial=3,
        )

        # _original_bounds should contain the original bounds
        assert opt._original_bounds == original

    def test_factor_with_single_level_optimization(self):
        """Test that optimization works with single-level factor (degenerate case)."""
        # This is an edge case - a factor with only one level
        # While unusual, it should work (essentially a fixed dimension)
        opt = SpotOptim(
            fun=lambda X: np.sum(X[:, 1:] ** 2, axis=1),  # Ignore factor dimension
            bounds=[
                ("only_option",),  # Single level factor
                (0.0, 10.0),
            ],
            max_iter=5,
            n_initial=3,
        )

        assert opt._factor_maps[0] == {0: "only_option"}
        # Single level factor has bounds (0, 0) before dimension reduction
        # After dimension reduction, it becomes a fixed dimension
        # Check the original bounds in all_lower/all_upper
        assert opt.all_lower[0] == 0
        assert opt.all_upper[0] == 0
        # The dimension is fixed, so red_dim should be True
        assert opt.red_dim
        assert opt.ident[0]  # First dimension is fixed


class TestProcessFactorBoundsSpecialCharacters:
    """Test factor variables with special characters and strings."""

    def test_factor_with_spaces(self):
        """Test factor levels with spaces."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[("option one", "option two", "option three")],
            max_iter=5,
            n_initial=3,
        )

        assert opt._factor_maps[0][0] == "option one"
        assert opt._factor_maps[0][1] == "option two"
        assert opt._factor_maps[0][2] == "option three"

    def test_factor_with_special_chars(self):
        """Test factor levels with special characters."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[("a-b", "c_d", "e.f", "g/h")],
            max_iter=5,
            n_initial=3,
        )

        assert opt._factor_maps[0][0] == "a-b"
        assert opt._factor_maps[0][1] == "c_d"
        assert opt._factor_maps[0][2] == "e.f"
        assert opt._factor_maps[0][3] == "g/h"

    def test_factor_with_unicode(self):
        """Test factor levels with unicode characters."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[("α", "β", "γ", "δ")],
            max_iter=5,
            n_initial=3,
        )

        assert opt._factor_maps[0][0] == "α"
        assert opt._factor_maps[0][1] == "β"
        assert opt._factor_maps[0][2] == "γ"
        assert opt._factor_maps[0][3] == "δ"

    def test_factor_with_numeric_strings(self):
        """Test factor levels that are numeric as strings."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[("1", "2", "3", "10", "100")],
            max_iter=5,
            n_initial=3,
        )

        # These should be treated as strings, not numbers
        assert opt._factor_maps[0][0] == "1"
        assert opt._factor_maps[0][1] == "2"
        assert opt._factor_maps[0][4] == "100"
        assert opt.bounds[0] == (0, 4)

    def test_factor_with_empty_string(self):
        """Test factor level with empty string."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[("", "non-empty", "another")],
            max_iter=5,
            n_initial=3,
        )

        assert opt._factor_maps[0][0] == ""
        assert opt._factor_maps[0][1] == "non-empty"
