# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression tests for issue #87: int + log10 dimensions.

Before the fix, an integer dimension with a ``log10`` transform was int-cast
in *transformed* space — ``(10, 5000, "log10")`` became internal bounds
``(1, 3)`` — and internal rounding collapsed the dimension to the decade
exponents ``{10, 100, 1000, 10000}`` in natural scale, with ``10000``
silently exceeding the declared cap of ``5000`` (observed in a live
spotforecast2 tuning run on 2026-06-05).

After the fix the dimension is searched continuously in transformed space,
rounded to the nearest integer in natural space, and clipped to the declared
bounds.
"""

import numpy as np
import pytest

from spotoptim import SpotOptim
from spotoptim.utils.variables import internal_var_type, repair_natural_X

LOW, HIGH = 10, 5000
DECADES = {10, 100, 1000, 10000}


def _collecting_fun(seen):
    """Objective that records every natural-scale value of dimension 0."""

    def fun(X, **kwargs):
        arr = np.atleast_2d(X)
        seen.extend(arr[:, 0].tolist())
        rng = np.random.default_rng(0)
        return rng.random(arr.shape[0])

    return fun


def _make_optimizer(seen, **kwargs):
    defaults = dict(
        fun=_collecting_fun(seen),
        bounds=[(LOW, HIGH), (0.0, 1.0)],
        var_type=["int", "float"],
        var_trans=["log10", None],
        max_iter=40,
        n_initial=20,
        seed=42,
        verbose=False,
    )
    defaults.update(kwargs)
    return SpotOptim(**defaults)


class TestInternalBounds:
    """Transformed int dims keep continuous internal bounds."""

    def test_internal_bounds_stay_float(self):
        opt = _make_optimizer([])
        lo, hi = opt.bounds[0]
        assert isinstance(lo, float) and isinstance(hi, float)
        assert lo == pytest.approx(np.log10(LOW))
        assert hi == pytest.approx(np.log10(HIGH))

    def test_untransformed_int_bounds_still_int_cast(self):
        opt = SpotOptim(
            fun=lambda X: np.sum(np.atleast_2d(X) ** 2, axis=1),
            bounds=[(1.0, 100.0)],
            var_type=["int"],
            var_trans=[None],
            max_iter=1,
            n_initial=1,
        )
        assert opt.bounds[0] == (1, 100)
        assert isinstance(opt.bounds[0][0], int)
        assert isinstance(opt.bounds[0][1], int)

    def test_internal_var_type_masks_transformed_int(self):
        opt = _make_optimizer([])
        assert internal_var_type(opt) == ["float", "float"]
        assert opt.internal_var_type == ["float", "float"]

    def test_internal_var_type_keeps_plain_int_and_factor(self):
        opt = SpotOptim(
            fun=lambda X: np.sum(np.atleast_2d(X) ** 2, axis=1),
            bounds=[(1, 10), (0.0, 1.0)],
            var_type=["int", "float"],
            var_trans=[None, "sqrt"],
            max_iter=1,
            n_initial=1,
        )
        assert internal_var_type(opt) == ["int", "float"]


class TestNaturalValues:
    """The objective sees admissible integers, not decade exponents."""

    def test_values_are_integers_within_bounds(self):
        seen = []
        opt = _make_optimizer(seen)
        opt.optimize()
        values = np.asarray(seen)
        assert len(values) > 0
        assert np.all(values >= LOW), f"min {values.min()} < {LOW}"
        assert np.all(values <= HIGH), f"max {values.max()} > {HIGH} (issue #87)"
        np.testing.assert_array_equal(values, np.around(values))

    def test_dimension_no_longer_collapses_to_decades(self):
        seen = []
        opt = _make_optimizer(seen)
        opt.optimize()
        distinct = {int(v) for v in seen}
        non_decades = distinct - DECADES
        assert non_decades, (
            "all evaluated values are decade exponents — the dimension is "
            f"still collapsed (issue #87): {sorted(distinct)}"
        )

    def test_history_and_best_respect_bounds(self):
        seen = []
        opt = _make_optimizer(seen)
        opt.optimize()
        col = np.asarray(opt.X_)[:, 0]
        assert np.all(col >= LOW) and np.all(col <= HIGH)
        np.testing.assert_array_equal(col, np.around(col))
        assert LOW <= opt.best_x_[0] <= HIGH
        assert opt.best_x_[0] == round(opt.best_x_[0])

    def test_parallel_path_respects_bounds(self):
        seen = []
        opt = _make_optimizer(seen, n_jobs=2, max_iter=30, n_initial=15)
        opt.optimize()
        col = np.asarray(opt.X_)[:, 0]
        assert np.all(col >= LOW) and np.all(col <= HIGH)
        np.testing.assert_array_equal(col, np.around(col))


class TestRepairNaturalX:
    """Unit behaviour of the natural-scale repair."""

    def test_rounds_and_clips_transformed_int(self):
        opt = _make_optimizer([])
        X = np.array([[4999.99999, 0.5], [10.4, 0.5], [5000.2, 0.5], [9.6, 0.5]])
        out = opt.repair_natural_X(X)
        np.testing.assert_array_equal(out[:, 0], [5000.0, 10.0, 5000.0, 10.0])
        np.testing.assert_array_equal(out[:, 1], X[:, 1])  # float dim untouched

    def test_one_dim_input_keeps_shape(self):
        opt = _make_optimizer([])
        out = opt.repair_natural_X(np.array([123.6, 0.25]))
        assert out.shape == (2,)
        assert out[0] == 124.0
        assert out[1] == 0.25

    def test_unknown_width_returned_unchanged(self):
        opt = _make_optimizer([])
        X = np.array([[1.5, 2.5, 3.5]])  # three columns, optimizer has two dims
        out = repair_natural_X(opt, X)
        np.testing.assert_array_equal(out, X)

    def test_plain_int_dim_rounded_not_clipped_differently(self):
        opt = SpotOptim(
            fun=lambda X: np.sum(np.atleast_2d(X) ** 2, axis=1),
            bounds=[(1, 10)],
            var_type=["int"],
            var_trans=[None],
            max_iter=1,
            n_initial=1,
        )
        out = opt.repair_natural_X(np.array([[3.4]]))
        assert out[0, 0] == 3.0


class TestFloatLog10Unchanged:
    """Float + log10 dimensions keep their existing continuous behaviour."""

    def test_float_log10_values_in_bounds_and_continuous(self):
        seen = []

        def fun(X, **kwargs):
            arr = np.atleast_2d(X)
            seen.extend(arr[:, 0].tolist())
            rng = np.random.default_rng(1)
            return rng.random(arr.shape[0])

        opt = SpotOptim(
            fun=fun,
            bounds=[(0.0001, 0.3)],
            var_type=["float"],
            var_trans=["log10"],
            max_iter=25,
            n_initial=12,
            seed=7,
            verbose=False,
        )
        opt.optimize()
        values = np.asarray(seen)
        assert np.all(values >= 0.0001 - 1e-12)
        assert np.all(values <= 0.3 + 1e-12)
        # Continuous: values are not all integers
        assert not np.allclose(values, np.around(values))
