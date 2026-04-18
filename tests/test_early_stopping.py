# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the patience-based early-stopping rule (``max_restarts``)."""

import numpy as np
from spotoptim import SpotOptim


def _constant_fun(X):
    """Returns zero for every row; no infill can beat the initial best."""
    X = np.atleast_2d(X)
    return np.zeros(len(X))


def test_max_restarts_triggers_early_stop():
    """After ``max_restarts`` consecutive restarts without improvement the run
    must terminate cleanly with the dedicated early-stop message."""
    opt = SpotOptim(
        fun=_constant_fun,
        bounds=[(-5, 5), (-5, 5)],
        max_iter=200,
        n_initial=5,
        restart_after_n=3,
        window_size=3,
        max_restarts=2,
        seed=0,
        verbose=False,
    )
    result = opt.optimize()

    assert opt._early_stopped is True
    assert result.success is True
    assert "early stopped" in result.message
    assert "2 consecutive restarts" in result.message
    # Budget should not be exhausted — early stop must fire first.
    assert result.nfev < opt.max_iter


def test_max_restarts_none_preserves_legacy_behavior():
    """``max_restarts=None`` (default) must not alter the optimizer output."""
    opt = SpotOptim(
        fun=_constant_fun,
        bounds=[(-5, 5), (-5, 5)],
        max_iter=20,
        n_initial=5,
        restart_after_n=3,
        window_size=3,
        max_restarts=None,
        seed=0,
        verbose=False,
    )
    result = opt.optimize()

    assert opt._early_stopped is False
    assert "early stopped" not in result.message


def test_max_restarts_does_not_trigger_with_improvement():
    """When the objective keeps improving, ``max_restarts`` must never fire."""

    opt = SpotOptim(
        fun=lambda X: np.sum(np.atleast_2d(X) ** 2, axis=1),
        bounds=[(-5, 5), (-5, 5)],
        max_iter=30,
        n_initial=10,
        max_restarts=1,
        seed=0,
        verbose=False,
    )
    result = opt.optimize()

    assert opt._early_stopped is False
    assert "early stopped" not in result.message


def test_max_restarts_zero_stops_at_first_stalled_restart():
    """``max_restarts=0`` means: the very first restart that fails to improve
    stops the run. Useful as a hard ceiling for total restarts."""
    opt = SpotOptim(
        fun=_constant_fun,
        bounds=[(-5, 5), (-5, 5)],
        max_iter=200,
        n_initial=5,
        restart_after_n=3,
        window_size=3,
        max_restarts=0,
        seed=0,
        verbose=False,
    )
    result = opt.optimize()

    assert opt._early_stopped is True
    assert "0 consecutive restarts" in result.message
    # Only one restart should have happened.
    assert len(opt.restarts_results_) == 1
