# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression guard for the ``max_surrogate_points`` default.

History: every 0.12.x release and 1.0.0 used ``None`` (fit the GP on all
evaluated points). 1.0.1 silently capped it at ``30`` -- mislabeled as
"restoring the pre-1.0 default" -- which degrades runs exceeding 30 evaluations
(benchmarked: rosenbrock-5D ~2.9x worse median than all-points). A cap of 30 is
also far below the level at which a Kriging/GP fit is even statistically
sensible (geostatistics guidance: >=100-150 points) and has no runtime
justification (a 100-restart fit at n=30 is ~0.3s).

The default is now ``300``: past the quality knee (benchmarks show quality is
fully recovered by ~100 points), large enough that typical Bayesian-optimization
budgets ("a few hundred" evaluations, Frazier 2018) are fit on all points, while
bounding the O(n^3) GP-fit cost on very long runs (a single default fit is
~0.7s at n=100, ~3.8s at n=300, ~55s at n=1000).

These tests pin the default so the cap cannot silently drop back toward 30.
"""

from spotoptim import SpotOptim
from spotoptim.SpotOptim import SpotOptimConfig
from spotoptim.function import sphere

EXPECTED_DEFAULT = 300


def test_config_default_is_runtime_aware_cap():
    assert SpotOptimConfig().max_surrogate_points == EXPECTED_DEFAULT


def test_instance_default_matches_config():
    opt = SpotOptim(fun=sphere, bounds=[(-5, 5), (-5, 5)])
    assert opt.max_surrogate_points == EXPECTED_DEFAULT
    assert opt.config.max_surrogate_points == EXPECTED_DEFAULT


def test_default_exceeds_kriging_fit_minimum():
    """The default must stay well above the level where a GP/Kriging fit is
    statistically sensible -- guards against a re-introduction of the ~30 cap."""
    default = SpotOptimConfig().max_surrogate_points
    assert default is not None and default >= 150


def test_default_active_cap_is_set():
    opt = SpotOptim(fun=sphere, bounds=[(-5, 5), (-5, 5)])
    opt.init_surrogate()
    assert opt._active_max_surrogate_points == EXPECTED_DEFAULT
