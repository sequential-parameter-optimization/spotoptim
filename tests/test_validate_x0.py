# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Direct unit tests for SpotOptim.validate_x0().

These tests call validate_x0() directly (not via __init__) to cover all
internal branches:
  * scalar / 0-D input          → ValueError
  * ndim > 2 input              → ValueError
  * 1-D correct shape           → returns 1-D internal-scale array
  * 1-D wrong length            → ValueError
  * 2-D single row (1, n) shape → flattened and returned as 1-D
  * 2-D multi-row (m, n) shape  → returned as 2-D internal-scale array
  * 2-D wrong number of columns → ValueError
  * value outside bounds        → ValueError
  * value at exact boundary     → accepted
  * log10 transformation        → values in log10 space
  * dimension reduction active  → output has fewer columns
  * fixed dimension wrong value → ValueError
  * verbose output              → message printed
  * list input (array_like)     → converted and validated
"""

import numpy as np
import pytest

from spotoptim import SpotOptim
from spotoptim.function import sphere


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_opt(bounds, var_trans=None, var_type=None, seed=42):
    """Create a minimal SpotOptim instance without running optimize()."""
    return SpotOptim(
        fun=sphere,
        bounds=bounds,
        var_trans=var_trans,
        var_type=var_type,
        seed=seed,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# 1. Shape / dimensionality checks
# ---------------------------------------------------------------------------


class TestValidateX0Shape:
    """Tests for shape and dimensionality validation."""

    def test_scalar_raises(self):
        """A scalar (0-D array) must raise ValueError."""
        opt = make_opt([(-5, 5), (-5, 5)])
        with pytest.raises(ValueError, match="must be a 1D array-like"):
            opt.validate_x0(np.float64(1.0))

    def test_3d_array_raises(self):
        """A 3-D array must raise ValueError."""
        opt = make_opt([(-5, 5), (-5, 5)])
        x0 = np.ones((1, 1, 2))  # 3-D
        with pytest.raises(ValueError, match="got 3D array"):
            opt.validate_x0(x0)

    def test_1d_correct_shape_returns_1d(self):
        """1-D x0 with the right length is accepted and returns 1-D array."""
        opt = make_opt([(-5, 5), (-5, 5)])
        result = opt.validate_x0(np.array([1.0, 2.0]))
        assert result.ndim == 1
        assert result.shape == (2,)

    def test_1d_wrong_length_raises(self):
        """1-D x0 with wrong number of values raises ValueError."""
        opt = make_opt([(-5, 5), (-5, 5)])
        with pytest.raises(ValueError, match="expected 2 dimensions"):
            opt.validate_x0(np.array([1.0, 2.0, 3.0]))

    def test_2d_single_row_flattened_to_1d(self):
        """2-D array with shape (1, n) is flattened and returned as 1-D."""
        opt = make_opt([(-5, 5), (-5, 5)])
        x0 = np.array([[1.0, 2.0]])  # shape (1, 2)
        result = opt.validate_x0(x0)
        assert result.ndim == 1
        assert result.shape == (2,)

    def test_2d_multi_row_returns_2d(self):
        """2-D array with multiple rows is returned as 2-D."""
        opt = make_opt([(-5, 5), (-5, 5)])
        x0 = np.array([[1.0, 2.0], [3.0, 4.0]])  # shape (2, 2)
        result = opt.validate_x0(x0)
        assert result.ndim == 2
        assert result.shape == (2, 2)

    def test_2d_wrong_columns_raises(self):
        """2-D x0 with wrong number of columns raises ValueError."""
        opt = make_opt([(-5, 5), (-5, 5)])
        x0 = np.array([[1.0, 2.0, 3.0]])  # shape (1, 3) – too many columns
        with pytest.raises(ValueError, match="expected 2 dimensions"):
            opt.validate_x0(x0)

    def test_list_input_accepted(self):
        """A plain Python list is converted to an array and accepted."""
        opt = make_opt([(-5, 5), (-5, 5)])
        result = opt.validate_x0([1.0, -2.0])
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)


# ---------------------------------------------------------------------------
# 2. Bounds checks
# ---------------------------------------------------------------------------


class TestValidateX0Bounds:
    """Tests for value-in-bounds validation."""

    def test_value_outside_upper_bound_raises(self):
        """x0 value above the upper bound raises ValueError."""
        opt = make_opt([(-5, 5), (-5, 5)])
        with pytest.raises(ValueError, match="outside bounds"):
            opt.validate_x0(np.array([6.0, 1.0]))

    def test_value_outside_lower_bound_raises(self):
        """x0 value below the lower bound raises ValueError."""
        opt = make_opt([(-5, 5), (-5, 5)])
        with pytest.raises(ValueError, match="outside bounds"):
            opt.validate_x0(np.array([1.0, -6.0]))

    def test_value_at_lower_boundary_accepted(self):
        """x0 at exactly the lower boundary is accepted."""
        opt = make_opt([(-5, 5), (-5, 5)])
        result = opt.validate_x0(np.array([-5.0, 0.0]))
        assert result.ndim == 1

    def test_value_at_upper_boundary_accepted(self):
        """x0 at exactly the upper boundary is accepted."""
        opt = make_opt([(-5, 5), (-5, 5)])
        result = opt.validate_x0(np.array([0.0, 5.0]))
        assert result.ndim == 1

    def test_error_message_contains_var_name(self):
        """ValueError message includes the variable name and value."""
        opt = make_opt([(-5, 5), (-5, 5)])
        opt.var_name = ["alpha", "beta"]
        opt.all_var_name = ["alpha", "beta"]
        with pytest.raises(ValueError, match="alpha"):
            opt.validate_x0(np.array([10.0, 1.0]))

    def test_multi_row_out_of_bounds_raises(self):
        """validate_x0 raises if any row in a 2-D x0 is out of bounds."""
        opt = make_opt([(-5, 5), (-5, 5)])
        x0 = np.array([[1.0, 2.0], [10.0, 1.0]])  # second row is invalid
        with pytest.raises(ValueError, match="outside bounds"):
            opt.validate_x0(x0)


# ---------------------------------------------------------------------------
# 3. Transformation to internal scale
# ---------------------------------------------------------------------------


class TestValidateX0Transform:
    """Tests that the returned array is in internal (transformed) scale."""

    def test_no_transform_identity(self):
        """Without var_trans, result equals the input (no-op transform)."""
        opt = make_opt([(-5, 5), (-5, 5)])
        x0 = np.array([1.0, 2.0])
        result = opt.validate_x0(x0)
        np.testing.assert_allclose(result, x0)

    def test_log10_transform_applied(self):
        """With log10 var_trans, result is log10 of input values."""
        opt = make_opt([(1, 100), (1, 100)], var_trans=["log10", "log10"])
        x0 = np.array([10.0, 100.0])
        result = opt.validate_x0(x0)
        np.testing.assert_allclose(result, np.log10(x0), rtol=1e-6)

    def test_log10_transform_single_dim(self):
        """Mixed transforms: first dim log10, second identity."""
        opt = make_opt([(1, 100), (-5, 5)], var_trans=["log10", None])
        x0 = np.array([10.0, 2.0])
        result = opt.validate_x0(x0)
        assert np.isclose(result[0], np.log10(10.0))
        assert np.isclose(result[1], 2.0)

    def test_transform_preserves_2d_shape(self):
        """Transformation is applied correctly to 2-D multi-row x0."""
        opt = make_opt([(1, 100), (1, 100)], var_trans=["log10", "log10"])
        x0 = np.array([[10.0, 100.0], [1.0, 10.0]])
        result = opt.validate_x0(x0)
        expected = np.log10(x0)
        np.testing.assert_allclose(result, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# 4. Dimension reduction (fixed dimensions)
# ---------------------------------------------------------------------------


class TestValidateX0DimensionReduction:
    """Tests for dimension reduction when some bounds are fixed (lower == upper)."""

    def test_fixed_dim_removed_from_output(self):
        """With one fixed dimension, output has n_dim - 1 values."""
        # bounds: dim0 free (-5,5), dim1 fixed (2,2), dim2 free (-5,5)
        opt = make_opt([(-5, 5), (2, 2), (-5, 5)])
        x0 = np.array([1.0, 2.0, 3.0])
        result = opt.validate_x0(x0)
        # After reduction only the two free dims remain
        assert result.shape == (2,)

    def test_fixed_dim_wrong_value_raises(self):
        """Providing a value that does not match the fixed dim raises ValueError."""
        opt = make_opt([(-5, 5), (2, 2), (-5, 5)])
        x0 = np.array([1.0, 3.0, 3.0])  # dim1 should be 2, not 3
        with pytest.raises(ValueError, match="fixed dimension"):
            opt.validate_x0(x0)

    def test_all_free_dims_untouched(self):
        """When no dimension is fixed, output length equals input length."""
        opt = make_opt([(-5, 5), (-5, 5), (-5, 5)])
        x0 = np.array([1.0, 2.0, 3.0])
        result = opt.validate_x0(x0)
        assert result.shape == (3,)

    def test_multiple_fixed_dims(self):
        """Two fixed dims are both removed; output has n - 2 values."""
        opt = make_opt([(-5, 5), (0, 0), (-5, 5), (1, 1)])
        x0 = np.array([1.0, 0.0, 3.0, 1.0])
        result = opt.validate_x0(x0)
        assert result.shape == (2,)


# ---------------------------------------------------------------------------
# 5. Verbose output
# ---------------------------------------------------------------------------


class TestValidateX0Verbose:
    """Tests for the verbose output of validate_x0."""

    def test_verbose_prints_confirmation(self, capsys):
        """With verbose=True, a confirmation message is printed."""
        opt = SpotOptim(
            fun=sphere,
            bounds=[(-5, 5), (-5, 5)],
            verbose=True,
            seed=42,
        )
        opt.validate_x0(np.array([1.0, 2.0]))
        captured = capsys.readouterr()
        assert "validated" in captured.out.lower()

    def test_no_verbose_no_output(self, capsys):
        """With verbose=False (default), no output is printed."""
        opt = make_opt([(-5, 5), (-5, 5)])
        opt.validate_x0(np.array([1.0, 2.0]))
        captured = capsys.readouterr()
        assert captured.out == ""


# ---------------------------------------------------------------------------
# 6. Bug: var_trans misalignment with dimension reduction
# ---------------------------------------------------------------------------


class TestValidateX0TransformWithDimReduction:
    """Regression tests for var_trans applied via validate_x0 when a fixed
    dimension causes dimension reduction.

    Root cause:
        transform_X() iterates self.var_trans (the reduced list) using column
        index i.  When a fixed dimension sits between two free ones, the
        reduced var_trans loses the fixed-dim slot, so the index mapping is
        shifted:

            full dims  : [log10, id,   sqrt]   (indices 0, 1, 2)
            ident mask : [False, True, False]
            reduced    : [log10, sqrt]          (indices 0, 1 in reduced space)

        transform_X applied to full-dim x0 = [1.0, 5.0, 9.0]:
            i=0, trans="log10"  → column 0: log10(1.0)  = 0.0  ✓
            i=1, trans="sqrt"   → column 1: sqrt(5.0)   = 2.24 ✗ (wrong column!)
            column 2 (9.0) gets no transform → stays 9.0         ✗

        After removing the fixed dim (column 1), result = [0.0, 9.0]
        Expected result = [0.0, 3.0]  (sqrt(9.0) = 3.0 not applied)

    These tests document that the bug exists (marked xfail) and will be
    promoted to positive assertions once the fix is applied.
    """

    def test_sqrt_applied_to_correct_column_after_dim_reduction(self):
        """sqrt transform on dim 2 must survive fixed dim 1 being removed.

        Setup:
            bounds     = [(-5, 5), (5, 5), (-10, 10)]
            var_trans  = ["log10", "id",   "sqrt" ]
            x0         = [ 1.0,    5.0,     9.0  ]

        Expected x0 in internal (reduced) scale:
            dim 0: log10(1.0) = 0.0
            dim 2: sqrt(9.0)  = 3.0
            → [0.0, 3.0]

        Actual (buggy) result: [0.0, 9.0]
        """
        opt = SpotOptim(
            fun=sphere,
            bounds=[(-5, 5), (5, 5), (-10, 10)],
            x0=np.array([1.0, 5.0, 9.0]),
            var_trans=["log10", "id", "sqrt"],
        )
        np.testing.assert_allclose(opt.x0, [0.0, 3.0], rtol=1e-6)

    def test_log10_applied_to_correct_column_fixed_dim_first(self):
        """log10 on dim 1 must not be displaced by a fixed dim 0.

        Setup:
            bounds    = [(3, 3),  (1, 100)]
            var_trans = ["id",    "log10" ]
            x0        = [ 3.0,    10.0   ]

        Expected x0 in internal (reduced) scale:
            dim 1: log10(10.0) = 1.0
            → [1.0]

        Actual (buggy) result: [10.0]  (log10 applied to wrong slot)
        """
        opt = SpotOptim(
            fun=sphere,
            bounds=[(3, 3), (1, 100)],
            x0=np.array([3.0, 10.0]),
            var_trans=["id", "log10"],
        )
        np.testing.assert_allclose(opt.x0, [1.0], rtol=1e-6)

    def test_transforms_correct_with_multiple_fixed_dims(self):
        """Three free dims with two fixed dims interspersed must all transform correctly.

        Setup:
            bounds    = [(-5,5), (0,0), (1,100), (2,2), (-10,10)]
            var_trans = [None,   None,  "log10", None,  "sqrt"  ]
            x0        = [ 1.0,   0.0,   10.0,   2.0,    9.0   ]

        Expected x0 in internal (reduced) scale (dims 0, 2, 4 are free):
            dim 0: None  →  1.0
            dim 2: log10 → log10(10.0) = 1.0
            dim 4: sqrt  → sqrt(9.0)   = 3.0
            → [1.0, 1.0, 3.0]
        """
        opt = SpotOptim(
            fun=sphere,
            bounds=[(-5, 5), (0, 0), (1, 100), (2, 2), (-10, 10)],
            x0=np.array([1.0, 0.0, 10.0, 2.0, 9.0]),
            var_trans=[None, None, "log10", None, "sqrt"],
        )
        np.testing.assert_allclose(opt.x0, [1.0, 1.0, 3.0], rtol=1e-6)
