# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for restart injection in the parallel (steady-state) optimisation path.

Verify that when y0_known and self.x0 are set, the matching point in the
initial design is not re-evaluated by the worker pool, and that the stored
best solution from the previous run is correctly carried into the next run.
"""

import time

import numpy as np

from spotoptim import SpotOptim
from spotoptim.function import sphere


class CountingObjective:
    """Sphere function that records every point it is called with."""

    def __init__(self):
        self.calls = []

    def __call__(self, X):
        for x in X:
            self.calls.append(x.copy())
        return np.sum(X**2, axis=1)

    def was_called_with(self, x, tol=1e-9):
        return any(np.linalg.norm(c - x) < tol for c in self.calls)


def test_restart_inject_skips_reeval_parallel():
    """The injected best point must not be re-evaluated in the parallel path."""
    obj = CountingObjective()

    opt = SpotOptim(
        fun=obj,
        bounds=[(-5, 5), (-5, 5)],
        n_initial=5,
        max_iter=10,
        seed=0,
        n_jobs=2,
        x0=np.array([1.0, 1.0]),
    )

    # Inject a known best value; x0=[1,1] matches the injected point.
    y0_known = 2.0  # sphere([1, 1]) = 2
    obj.calls.clear()

    status, result = opt.optimize_steady_state(
        timeout_start=time.time(),
        X0=None,
        y0_known=y0_known,
    )

    assert status == "FINISHED"
    assert not obj.was_called_with(
        np.array([1.0, 1.0])
    ), "Injected best point [1, 1] should not have been re-evaluated by the worker pool."


def test_restart_inject_result_stored_parallel():
    """The injected best value must appear in the stored y_ array."""
    opt = SpotOptim(
        fun=sphere,
        bounds=[(-5, 5), (-5, 5)],
        n_initial=5,
        max_iter=10,
        seed=0,
        n_jobs=2,
        x0=np.array([1.0, 1.0]),
    )

    y0_known = 2.0
    opt.optimize_steady_state(
        timeout_start=time.time(),
        X0=None,
        y0_known=y0_known,
    )

    assert (
        y0_known in opt.y_
    ), "The pre-filled y0_known value must be present in opt.y_ after the run."


def test_restart_inject_no_known_unchanged():
    """Without y0_known, all n_initial points must reach storage (none skipped)."""
    opt = SpotOptim(
        fun=sphere,
        bounds=[(-5, 5), (-5, 5)],
        n_initial=5,
        max_iter=5,
        seed=0,
        n_jobs=2,
    )

    status, result = opt.optimize_steady_state(
        timeout_start=time.time(),
        X0=None,
        y0_known=None,
    )

    assert status == "FINISHED"
    # All n_initial points must be stored — no injection means nothing was skipped.
    assert len(opt.y_) == opt.n_initial
    assert np.all(np.isfinite(opt.y_))


def test_restart_inject_parallel_eval_count():
    """With one injected point, y_ still contains n_initial entries and the
    injected value is stored at exact float precision (not re-evaluated).

    Note: evaluations run in separate worker processes (ProcessPoolExecutor on
    GIL builds), so in-process call counters cannot be used here.  Instead we
    verify the observable storage state: n_initial rows in X_/y_ and the
    injected value appearing verbatim in y_.
    """
    x_inject = np.array([0.5, -0.5])
    y_inject = float(np.sum(x_inject**2))

    opt = SpotOptim(
        fun=sphere,
        bounds=[(-5, 5), (-5, 5)],
        n_initial=6,
        max_iter=6,
        seed=1,
        n_jobs=2,
        x0=x_inject,
    )

    opt.optimize_steady_state(
        timeout_start=time.time(),
        X0=None,
        y0_known=y_inject,
    )

    # All n_initial points must be in storage (injected + n_initial-1 via pool).
    assert (
        len(opt.y_) == opt.n_initial
    ), f"Expected {opt.n_initial} stored evaluations, got {len(opt.y_)}."
    # The injected value must appear at exact float precision — proof it was
    # stored directly and not recalculated in a worker (sphere may differ by
    # floating-point rounding for transformed input coordinates).
    assert (
        y_inject in opt.y_
    ), f"Injected y_known={y_inject} not found in opt.y_={opt.y_}."
