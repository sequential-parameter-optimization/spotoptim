# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression tests: warm-start ``x0`` for the differential-evolution acquisition.

Field failure (spotforecast2 team4 LightGBM tuning, 2026-06-27): a SpotOptim run
aborted mid-search with ``ValueError: Some entries in x0 lay outside the
specified bounds`` raised by scipy's ``differential_evolution``.

Root cause: scipy validates ``x0`` in a NORMALIZED metric
``scaled = (x0 - mid) * recip + 0.5`` (with ``mid = (lb + ub) / 2`` and
``recip = 1 / |ub - lb|``), not against the raw box. An incumbent that sits
exactly on a box bound — e.g. ``reg_alpha = reg_lambda = 0.01`` on the wide span
``(0.01, 100.0)`` — normalizes to ``-1.1e-16``, a hair below 0, purely from
floating-point rounding. The previous ``np.nextafter`` "nudge inward" was applied
in raw parameter space: one ULP at ``0.01`` is ``~1.7e-18``, orders of magnitude
below the rounding granularity of scipy's rescaling at the span's magnitude, so
the nudge was lost and scipy still rejected ``x0``.

Fix: :func:`spotoptim.optimizer.acquisition.sanitize_de_x0` clamps in scipy's own
normalized metric and maps back with scipy's exact inverse.
"""

import numpy as np
import pytest
from scipy.optimize import differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor

from spotoptim import SpotOptim
from spotoptim.optimizer.acquisition import sanitize_de_x0

# Exact internal-scale configuration captured from the failing field run.
# Dims 6 and 7 (reg_alpha, reg_lambda) sit on the lower bound 0.01 of (0.01, 100).
FIELD_BOUNDS = [
    (8.0, 256.0),
    (3.0, 16.0),
    (-4.0, -1.0),
    (1.0, 3.0),
    (0.5, 1.0),
    (0.5, 1.0),
    (0.01, 100.0),
    (0.01, 100.0),
    (0.0, 6.0),
]
FIELD_X0 = np.array([31.0, 3.0, -1.0, 2.0, 0.75, 0.75, 0.01, 0.01, 0.0])


def _scipy_accepts(x0, bounds):
    """Replicate scipy's exact x0-in-bounds test (DifferentialEvolutionSolver).

    Mirrors ``_differentialevolution.py`` byte-for-byte: ``scale_arg2`` is
    ``|lb - ub|``, ``recip`` is ``1/span`` with non-finite entries zeroed, and
    the rejection test is purely ``(scaled > 1) | (scaled < 0)`` — note scipy
    does NOT test finiteness of ``x0`` (NaN comparisons are False, so NaN is
    accepted; ``+/-inf`` normalizes outside [0, 1] and is rejected).
    """
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)
    mid = 0.5 * (lb + ub)
    span = np.abs(lb - ub)
    with np.errstate(divide="ignore", invalid="ignore"):
        recip = 1.0 / span
    recip[~np.isfinite(recip)] = 0.0
    scaled = (np.asarray(x0, dtype=float) - mid) * recip + 0.5
    return bool(not np.any((scaled > 1.0) | (scaled < 0.0)))


class TestSanitizeDeX0:
    """Unit tests for the normalized-metric x0 sanitiser."""

    def test_raw_field_incumbent_is_rejected_by_scipy_metric(self):
        # The incumbent on the wide-span lower bound normalizes outside [0, 1].
        assert not _scipy_accepts(FIELD_X0, FIELD_BOUNDS)

    def test_old_nextafter_nudge_did_not_help(self):
        # The previous raw-space nudge is too small to survive scipy's rescaling.
        lb = np.array([b[0] for b in FIELD_BOUNDS])
        ub = np.array([b[1] for b in FIELD_BOUNDS])
        nudged = np.clip(FIELD_X0, np.nextafter(lb, ub), np.nextafter(ub, lb))
        assert not _scipy_accepts(nudged, FIELD_BOUNDS)

    def test_sanitized_incumbent_is_accepted_and_faithful(self):
        fixed = sanitize_de_x0(FIELD_X0, FIELD_BOUNDS)
        assert fixed is not None
        assert _scipy_accepts(fixed, FIELD_BOUNDS)
        # The adjustment is negligible in parameter space.
        np.testing.assert_allclose(fixed, FIELD_X0, atol=1e-9)

    def test_scipy_de_constructor_rejects_raw_incumbent(self):
        # Pin the exact upstream behaviour we are guarding against.
        with pytest.raises(ValueError, match="lay outside"):
            differential_evolution(
                lambda x: float(np.sum(x)),
                FIELD_BOUNDS,
                x0=FIELD_X0,
                maxiter=1,
                popsize=4,
                seed=0,
            )

    def test_scipy_de_constructor_accepts_sanitized_incumbent(self):
        fixed = sanitize_de_x0(FIELD_X0, FIELD_BOUNDS)
        # Must not raise "Some entries in x0 lay outside the specified bounds".
        differential_evolution(
            lambda x: float(np.sum(x)),
            FIELD_BOUNDS,
            x0=fixed,
            maxiter=1,
            popsize=4,
            seed=0,
        )

    def test_interior_point_returned_bit_for_bit(self):
        x0 = np.array([50.0, 9.0, -2.0, 2.0, 0.7, 0.7, 50.0, 50.0, 3.0])
        out = sanitize_de_x0(x0, FIELD_BOUNDS)
        assert out is not None
        np.testing.assert_array_equal(out, x0)

    def test_upper_bound_incumbent_accepted(self):
        bounds = [(0.01, 100.0)]
        out = sanitize_de_x0(np.array([100.0]), bounds)
        assert out is not None and _scipy_accepts(out, bounds)

    def test_degenerate_span_accepted(self):
        # lb == ub: scipy maps everything to 0.5; never out of bounds.
        bounds = [(5.0, 5.0)]
        out = sanitize_de_x0(np.array([5.0]), bounds)
        assert out is not None and _scipy_accepts(out, bounds)

    def test_both_wide_span_bounds_of_a_dimension(self):
        # Lower and upper of the same wide span both survive sanitisation.
        bounds = [(0.01, 100.0)]
        for v in (0.01, 100.0):
            out = sanitize_de_x0(np.array([v]), bounds)
            assert out is not None and _scipy_accepts(out, bounds)

    def test_reversed_bounds_handled_like_scipy(self):
        # scipy normalizes with |ub - lb|, so reversed bounds are accepted; the
        # sanitiser must agree (the boundary value still maps inside [0, 1]).
        bounds = [(100.0, 0.01)]
        out = sanitize_de_x0(np.array([0.01]), bounds)
        assert out is not None and _scipy_accepts(out, bounds)

    def test_none_bounds_returns_none(self):
        assert sanitize_de_x0(np.array([0.5]), None) is None

    def test_nonfinite_x0_returns_none(self):
        # A non-finite incumbent is dropped (exercises the None path); the run
        # then proceeds without the optional warm-start instead of crashing.
        for bad in (np.nan, np.inf, -np.inf):
            x0 = FIELD_X0.copy()
            x0[3] = bad
            assert sanitize_de_x0(x0, FIELD_BOUNDS) is None


# Asymmetric wide spans seen in practice, raw and transform-induced: the
# sanitiser operates in INTERNAL scale, so a natural bound (0.01, 100) under
# log10/sqrt/ln reaches scipy as the transformed span below.
SPAN_CASES = [
    pytest.param((0.01, 100.0), id="raw-0.01-100"),
    pytest.param((0.001, 1000.0), id="raw-0.001-1000"),
    pytest.param((1e-06, 1.0), id="raw-1e-6-1"),
    pytest.param((-100.0, -0.01), id="raw-negative-asymmetric"),
    pytest.param((float(np.log10(1e-4)), float(np.log10(0.1))), id="log10-of-1e-4-0.1"),
    pytest.param((float(np.sqrt(0.01)), float(np.sqrt(100.0))), id="sqrt-of-0.01-100"),
    pytest.param((float(np.log(0.01)), float(np.log(100.0))), id="ln-of-0.01-100"),
]


class TestSanitizeDeX0SpanMatrix:
    """x0 exactly ON a bound survives scipy's REAL x0 validation for a matrix
    of asymmetric spans (raw and transform-induced), at both endpoints.

    Extends the FIELD_BOUNDS regression above: the existing scipy-constructor
    tests exercise only the captured field configuration.
    """

    @pytest.mark.parametrize("endpoint", ["lower", "upper"])
    @pytest.mark.parametrize("span", SPAN_CASES)
    def test_bound_incumbent_survives_real_scipy_validation(self, span, endpoint):
        bounds = [span]
        raw = span[0] if endpoint == "lower" else span[1]
        fixed = sanitize_de_x0(np.array([raw]), bounds)
        assert fixed is not None
        assert _scipy_accepts(fixed, bounds)
        # The real upstream constructor check must accept it (must not raise
        # "Some entries in x0 lay outside the specified bounds").
        differential_evolution(
            lambda x: float(np.sum(x**2)),
            bounds,
            x0=fixed,
            maxiter=1,
            popsize=4,
            seed=0,
        )
        # Faithful: the adjustment stays around the normalized margin x span,
        # negligible for a warm start.
        span_width = abs(span[1] - span[0])
        assert abs(fixed[0] - raw) <= 1e-9 * span_width

    @pytest.mark.parametrize("corner", ["lower", "upper"])
    def test_corner_incumbent_on_every_bound_survives(self, corner):
        # Mixed asymmetric spans; the incumbent sits on EVERY bound at once
        # (optimizers converge to corners).
        bounds = [(0.01, 100.0), (1e-05, 10.0), (-4.0, -1.0), (0.1, 10.0)]
        idx = 0 if corner == "lower" else 1
        x0 = np.array([b[idx] for b in bounds])
        fixed = sanitize_de_x0(x0, bounds)
        assert fixed is not None
        assert _scipy_accepts(fixed, bounds)
        differential_evolution(
            lambda x: float(np.sum(x**2)),
            bounds,
            x0=fixed,
            maxiter=1,
            popsize=4,
            seed=0,
        )
        spans = np.array([abs(b[1] - b[0]) for b in bounds])
        assert np.all(np.abs(fixed - x0) <= 1e-9 * spans)


class TestDeWarmStartIntegration:
    """End-to-end: a boundary incumbent must not abort the optimization."""

    def test_optimize_with_boundary_incumbent_does_not_crash(self):
        # Objective minimized exactly at the lower-bound corner, so best_x_ stays
        # on the wide-span bounds; de_x0_prob=1.0 forces the warm-start x0 every
        # infill -> reproduces the field crash on the pre-fix code.
        bounds = [(0.01, 100.0), (0.01, 100.0), (0.0001, 0.1)]
        lower = np.array([b[0] for b in bounds])

        def fun(X):
            X = np.atleast_2d(X)
            return np.sum((X - lower) ** 2, axis=1)

        opt = SpotOptim(
            fun=fun,
            bounds=bounds,
            var_type=["float", "float", "float"],
            var_trans=[None, None, "log10"],
            x0=lower.copy(),
            de_x0_prob=1.0,
            n_initial=5,
            max_iter=12,
            seed=0,
            surrogate=GaussianProcessRegressor(),
            verbose=False,
        )
        result = opt.optimize()
        assert result is not None
        # Best stays at the lower-bound corner and the run completed.
        assert np.allclose(opt.best_x_[:2], 0.01, atol=1e-6)
