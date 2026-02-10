# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the transform_bounds() method in SpotOptim."""

import numpy as np
import pytest
from spotoptim import SpotOptim


class TestTransformBoundsBasic:
    """Test basic transform_bounds() functionality."""

    def test_transform_bounds_no_transformation(self):
        """Test transform_bounds() with no transformations."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0, 10), (-5, 5)],
            var_trans=[None, None],
            max_iter=1,
            n_initial=1,
        )

        # Bounds should remain unchanged
        assert opt.bounds == [(0.0, 10.0), (-5.0, 5.0)]
        assert opt.lower[0] == 0.0
        assert opt.upper[0] == 10.0
        assert opt.lower[1] == -5.0
        assert opt.upper[1] == 5.0

    def test_transform_bounds_log10(self):
        """Test transform_bounds() with log10 transformation."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(1, 10), (0.1, 100)],
            var_trans=["log10", "log10"],
            max_iter=1,
            n_initial=1,
        )

        # Check bounds are transformed
        assert opt.lower[0] == pytest.approx(np.log10(1.0))
        assert opt.upper[0] == pytest.approx(np.log10(10.0))
        assert opt.lower[1] == pytest.approx(np.log10(0.1))
        assert opt.upper[1] == pytest.approx(np.log10(100.0))

        # Check bounds list is updated
        assert opt.bounds[0] == (pytest.approx(0.0), pytest.approx(1.0))
        assert opt.bounds[1] == (pytest.approx(-1.0), pytest.approx(2.0))

    def test_transform_bounds_sqrt(self):
        """Test transform_bounds() with sqrt transformation."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(1, 100), (4, 16)],
            var_trans=["sqrt", "sqrt"],
            max_iter=1,
            n_initial=1,
        )

        # Check bounds are transformed
        assert opt.lower[0] == pytest.approx(np.sqrt(1.0))
        assert opt.upper[0] == pytest.approx(np.sqrt(100.0))
        assert opt.lower[1] == pytest.approx(np.sqrt(4.0))
        assert opt.upper[1] == pytest.approx(np.sqrt(16.0))

        # Check bounds list
        assert opt.bounds[0] == (pytest.approx(1.0), pytest.approx(10.0))
        assert opt.bounds[1] == (pytest.approx(2.0), pytest.approx(4.0))

    def test_transform_bounds_log(self):
        """Test transform_bounds() with natural log transformation."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0.1, 10.0)],
            var_trans=["log"],
            max_iter=1,
            n_initial=1,
        )

        # Check bounds are transformed
        assert opt.lower[0] == pytest.approx(np.log(0.1))
        assert opt.upper[0] == pytest.approx(np.log(10.0))

    def test_transform_bounds_exp(self):
        """Test transform_bounds() with exponential transformation."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0.1, 2.0)],
            var_trans=["exp"],
            max_iter=1,
            n_initial=1,
        )

        # Check bounds are transformed
        assert opt.lower[0] == pytest.approx(np.exp(0.1))
        assert opt.upper[0] == pytest.approx(np.exp(2.0))

    def test_transform_bounds_square(self):
        """Test transform_bounds() with square transformation."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(1.0, 10.0)],
            var_trans=["square"],
            max_iter=1,
            n_initial=1,
        )

        # Check bounds are transformed
        assert opt.lower[0] == pytest.approx(1.0**2)
        assert opt.upper[0] == pytest.approx(10.0**2)

    def test_transform_bounds_cube(self):
        """Test transform_bounds() with cube transformation."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(1.0, 5.0)],
            var_trans=["cube"],
            max_iter=1,
            n_initial=1,
        )

        # Check bounds are transformed
        assert opt.lower[0] == pytest.approx(1.0**3)
        assert opt.upper[0] == pytest.approx(5.0**3)

    def test_transform_bounds_reciprocal(self):
        """Test transform_bounds() with reciprocal transformation."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0.1, 10.0)],
            var_trans=["inv"],
            max_iter=1,
            n_initial=1,
        )

        # Reciprocal reverses bounds: 1/0.1 = 10, 1/10 = 0.1
        # After swap: lower = 0.1, upper = 10
        assert opt.lower[0] == pytest.approx(1.0 / 10.0)
        assert opt.upper[0] == pytest.approx(1.0 / 0.1)


class TestTransformBoundsMixed:
    """Test transform_bounds() with mixed transformations."""

    def test_transform_bounds_mixed_transformations(self):
        """Test transform_bounds() with different transformations per dimension."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(1, 10), (0.1, 100), (0, 5), (1, 4)],
            var_trans=["log10", "sqrt", None, "log"],
            max_iter=1,
            n_initial=1,
        )

        # Check each dimension separately
        assert opt.lower[0] == pytest.approx(np.log10(1.0))
        assert opt.upper[0] == pytest.approx(np.log10(10.0))

        assert opt.lower[1] == pytest.approx(np.sqrt(0.1))
        assert opt.upper[1] == pytest.approx(np.sqrt(100.0))

        assert opt.lower[2] == 0.0
        assert opt.upper[2] == 5.0

        assert opt.lower[3] == pytest.approx(np.log(1.0))
        assert opt.upper[3] == pytest.approx(np.log(4.0))

    def test_transform_bounds_id_equivalent_to_none(self):
        """Test that 'id' transformation is equivalent to None."""
        opt1 = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0, 10), (-5, 5)],
            var_trans=[None, None],
            max_iter=1,
            n_initial=1,
        )

        opt2 = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0, 10), (-5, 5)],
            var_trans=["id", "id"],
            max_iter=1,
            n_initial=1,
        )

        # Both should have identical bounds
        assert opt1.lower[0] == opt2.lower[0]
        assert opt1.upper[0] == opt2.upper[0]
        assert opt1.lower[1] == opt2.lower[1]
        assert opt1.upper[1] == opt2.upper[1]


class TestTransformBoundsVarType:
    """Test transform_bounds() with different variable types."""

    def test_transform_bounds_with_int_var_type(self):
        """Test that int var_type produces integer bounds (no transformation)."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(1.0, 100.0)],
            var_trans=[None],  # No transformation
            var_type=["int"],
            max_iter=1,
            n_initial=1,
        )

        # Bounds should be converted to int
        assert opt.bounds[0] == (1, 100)
        assert isinstance(opt.bounds[0][0], int)
        assert isinstance(opt.bounds[0][1], int)

    def test_transform_bounds_with_float_var_type(self):
        """Test that float var_type produces float bounds."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(1, 100)],
            var_trans=["sqrt"],
            var_type=["float"],
            max_iter=1,
            n_initial=1,
        )

        # sqrt(1) = 1.0, sqrt(100) = 10.0
        assert opt.bounds[0] == (pytest.approx(1.0), pytest.approx(10.0))
        assert isinstance(opt.bounds[0][0], float)
        assert isinstance(opt.bounds[0][1], float)

    def test_transform_bounds_with_factor_var_type(self):
        """Test that factor var_type produces integer bounds."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[("red", "green", "blue"), (1, 100)],
            var_trans=[None, "sqrt"],
            max_iter=1,
            n_initial=1,
        )

        # Factor variable should have int bounds
        assert opt.bounds[0] == (0, 2)
        assert isinstance(opt.bounds[0][0], int)
        assert isinstance(opt.bounds[0][1], int)

        # Second dimension with sqrt transformation and auto-detected float type
        assert isinstance(opt.bounds[1][0], float)
        assert isinstance(opt.bounds[1][1], float)

    def test_transform_bounds_mixed_var_types(self):
        """Test transform_bounds() with mixed variable types."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(1, 100), (0.1, 10.0), (1, 16)],
            var_trans=["sqrt", "log10", "sqrt"],
            var_type=["int", "float", "int"],
            max_iter=1,
            n_initial=1,
        )

        # First dimension: int
        assert isinstance(opt.bounds[0][0], int)
        assert isinstance(opt.bounds[0][1], int)
        assert opt.bounds[0] == (1, 10)

        # Second dimension: float
        assert isinstance(opt.bounds[1][0], float)
        assert isinstance(opt.bounds[1][1], float)

        # Third dimension: int
        assert isinstance(opt.bounds[2][0], int)
        assert isinstance(opt.bounds[2][1], int)
        assert opt.bounds[2] == (1, 4)


class TestTransformBoundsBoundSwapping:
    """Test transform_bounds() handles bound swapping for reversed transformations."""

    def test_transform_bounds_reciprocal_swaps_bounds(self):
        """Test that reciprocal transformation swaps bounds correctly."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0.5, 10.0)],
            var_trans=["inv"],
            max_iter=1,
            n_initial=1,
        )

        # 1/0.5 = 2.0, 1/10.0 = 0.1
        # After swap: lower = 0.1, upper = 2.0
        assert opt.lower[0] < opt.upper[0]
        assert opt.lower[0] == pytest.approx(1.0 / 10.0)
        assert opt.upper[0] == pytest.approx(1.0 / 0.5)

    def test_transform_bounds_reciprocal_negative_bounds(self):
        """Test reciprocal with negative bounds."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-10.0, -1.0)],
            var_trans=["inv"],
            max_iter=1,
            n_initial=1,
        )

        # 1/-10 = -0.1, 1/-1 = -1.0
        # After swap: lower = -1.0, upper = -0.1
        assert opt.lower[0] < opt.upper[0]
        assert opt.lower[0] == pytest.approx(-1.0)
        assert opt.upper[0] == pytest.approx(-0.1)


class TestTransformBoundsOriginalBounds:
    """Test that transform_bounds() preserves original bounds."""

    def test_transform_bounds_preserves_original_lower_upper(self):
        """Test that _original_lower and _original_upper are preserved."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(1, 10), (0.1, 100)],
            var_trans=["log10", "sqrt"],
            max_iter=1,
            n_initial=1,
        )

        # Original bounds should be preserved
        assert opt._original_lower[0] == 1.0
        assert opt._original_upper[0] == 10.0
        assert opt._original_lower[1] == 0.1
        assert opt._original_upper[1] == 100.0

        # Internal bounds should be transformed
        assert opt.lower[0] != 1.0
        assert opt.upper[0] != 10.0
        assert opt.lower[1] != 0.1
        assert opt.upper[1] != 100.0

    def test_transform_bounds_all_dimensions(self):
        """Test that bounds are correctly updated for all dimensions."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(1, 10), (0.1, 1.0), (0, 5)],
            var_trans=["log10", "log10", None],
            max_iter=1,
            n_initial=1,
        )

        # Check that bounds list has same length
        assert len(opt.bounds) == 3

        # Check each bound is a tuple
        for bound in opt.bounds:
            assert isinstance(bound, tuple)
            assert len(bound) == 2
            assert bound[0] < bound[1]


class TestTransformBoundsReturnTypes:
    """Test that transform_bounds() returns correct Python types."""

    def test_transform_bounds_returns_python_float(self):
        """Test that float bounds are Python float, not numpy types."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(1, 10)],
            var_trans=["log10"],
            var_type=["float"],
            max_iter=1,
            n_initial=1,
        )

        assert isinstance(opt.bounds[0][0], float) and not isinstance(
            opt.bounds[0][0], np.floating
        )
        assert isinstance(opt.bounds[0][1], float) and not isinstance(
            opt.bounds[0][1], np.floating
        )
        assert not isinstance(opt.bounds[0][0], np.floating)
        assert not isinstance(opt.bounds[0][1], np.floating)

    def test_transform_bounds_returns_python_int(self):
        """Test that int bounds are Python int, not numpy types (no transformation)."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(1.0, 100.0)],
            var_trans=[None],  # No transformation
            var_type=["int"],
            max_iter=1,
            n_initial=1,
        )

        assert isinstance(opt.bounds[0][0], int) and not isinstance(
            opt.bounds[0][0], np.integer
        )
        assert isinstance(opt.bounds[0][1], int) and not isinstance(
            opt.bounds[0][1], np.integer
        )
        assert not isinstance(opt.bounds[0][0], np.integer)
        assert not isinstance(opt.bounds[0][1], np.integer)

    def test_transform_bounds_factor_returns_python_int(self):
        """Test that factor bounds are Python int, not numpy types."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[("a", "b", "c")],
            max_iter=1,
            n_initial=1,
        )

        assert isinstance(opt.bounds[0][0], int)
        assert isinstance(opt.bounds[0][1], int)
        assert opt.bounds[0] == (0, 2)


class TestTransformBoundsEdgeCases:
    """Test edge cases for transform_bounds()."""

    def test_transform_bounds_single_dimension(self):
        """Test transform_bounds() with single dimension."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(1, 100)],
            var_trans=["log10"],
            max_iter=1,
            n_initial=1,
        )

        assert len(opt.bounds) == 1
        assert opt.lower[0] == pytest.approx(np.log10(1.0))
        assert opt.upper[0] == pytest.approx(np.log10(100.0))

    def test_transform_bounds_many_dimensions(self):
        """Test transform_bounds() with many dimensions."""
        n_dims = 10
        bounds = [(1, 10) for _ in range(n_dims)]
        var_trans = ["log10" if i % 2 == 0 else "sqrt" for i in range(n_dims)]

        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=bounds,
            var_trans=var_trans,
            max_iter=1,
            n_initial=1,
        )

        assert len(opt.bounds) == n_dims

        for i in range(n_dims):
            if i % 2 == 0:
                # log10 transformation
                assert opt.lower[i] == pytest.approx(np.log10(1.0))
                assert opt.upper[i] == pytest.approx(np.log10(10.0))
            else:
                # sqrt transformation
                assert opt.lower[i] == pytest.approx(np.sqrt(1.0))
                assert opt.upper[i] == pytest.approx(np.sqrt(10.0))

    def test_transform_bounds_equal_bounds_with_transformation(self):
        """Test transform_bounds() with equal bounds (fixed dimension)."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(1, 10), (5, 5)],
            var_trans=["log10", "sqrt"],
            max_iter=1,
            n_initial=1,
        )

        # After dimension reduction, only the first dimension remains
        # (fixed dimensions are removed from lower/upper arrays)
        assert len(opt.lower) == 1
        assert len(opt.upper) == 1

        # First dimension should be transformed normally
        assert opt.lower[0] == pytest.approx(np.log10(1.0))
        assert opt.upper[0] == pytest.approx(np.log10(10.0))

        # Check that the original (pre-reduction) bounds include the fixed dimension
        assert opt.all_lower[0] == pytest.approx(np.log10(1.0))
        assert opt.all_upper[0] == pytest.approx(np.log10(10.0))
        assert opt.all_lower[1] == pytest.approx(np.sqrt(5.0))
        assert opt.all_upper[1] == pytest.approx(np.sqrt(5.0))


class TestTransformBoundsIntegration:
    """Test transform_bounds() integration with optimization."""

    def test_transform_bounds_used_in_optimization(self):
        """Test that transformed bounds are actually used during optimization."""
        call_log = []

        def objective(X):
            # Log calls in original scale
            call_log.extend(X.tolist())
            return np.sum(X**2, axis=1)

        opt = SpotOptim(
            fun=objective,
            bounds=[(1, 100)],
            var_trans=["log10"],
            max_iter=5,
            n_initial=3,
            seed=42,
        )

        result = opt.optimize()

        # Check that all calls were in original scale (1 to 100)
        for x_val in call_log:
            assert 1.0 <= x_val[0] <= 100.0

        # Check result is in original scale
        assert 1.0 <= result.x[0] <= 100.0

    def test_transform_bounds_with_inverse_transform(self):
        """Test that bounds transformation works with inverse transformation."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(1, 10)],
            var_trans=["log10"],
            max_iter=1,
            n_initial=1,
        )

        # Create a point in internal scale
        x_internal = np.array([[0.5]])  # log10(sqrt(10)) ≈ 0.5

        # Inverse transform should give original scale
        x_original = opt._inverse_transform_X(x_internal)

        # Should be approximately sqrt(10) ≈ 3.162
        assert x_original[0, 0] == pytest.approx(10**0.5, rel=1e-5)

    def test_transform_bounds_direct_call(self):
        """Test calling transform_bounds() directly after initialization."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(1, 100)],
            var_trans=[None],
            max_iter=1,
            n_initial=1,
        )

        # Store original bounds
        original_lower = opt.lower.copy()
        original_upper = opt.upper.copy()

        # Manually change transformation and call transform_bounds
        opt.var_trans = ["log10"]
        opt.transform_bounds()

        # Bounds should now be transformed
        assert opt.lower[0] != original_lower[0]
        assert opt.upper[0] != original_upper[0]
        assert opt.lower[0] == pytest.approx(np.log10(1.0))
        assert opt.upper[0] == pytest.approx(np.log10(100.0))


class TestTransformBoundsDocstring:
    """Test examples from the docstring."""

    def test_docstring_example(self):
        """Test the example from the transform_bounds() docstring."""
        spot = SpotOptim(fun=lambda x: x, bounds=[(1, 10), (0.1, 100)])
        spot.var_trans = ["log10", "sqrt"]
        spot.transform_bounds()

        # Expected: [(0.0, 1.0), (0.31622776601683794, 10.0)]
        assert spot.bounds[0] == (pytest.approx(0.0), pytest.approx(1.0))
        assert spot.bounds[1] == (
            pytest.approx(0.31622776601683794),
            pytest.approx(10.0),
        )
