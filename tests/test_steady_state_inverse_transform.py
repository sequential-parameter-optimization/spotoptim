# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression tests for natural-scale storage in the steady-state path.

The steady-state (``n_jobs > 1``) loop must convert evaluated points to natural
scale via ``inverse_transform_X`` before storing them in ``X_`` -- mirroring the
sequential ``update_storage`` path. Before the fix, points were stored in
transformed scale, so a ``log10`` variable was re-transformed when the surrogate
was refit, yielding ``log10`` of a negative number => ``NaN`` => a crash in the
Gaussian-process fit (``ValueError: Input X contains NaN``).
"""

import numpy as np

from spotoptim import SpotOptim


def _obj(X):
    """Objective on the (log10) float column only; ignores the factor column."""
    import numpy as np

    X = np.atleast_2d(X)
    return np.array([float(np.asarray(row[0], dtype=float)) for row in X])


def test_steady_state_log10_does_not_crash_and_stores_natural_scale():
    bounds = [(0.001, 0.1), ["A", "B", "C"]]
    var_type = ["float", "factor"]
    var_trans = ["log10", None]

    opt = SpotOptim(
        fun=_obj,
        bounds=bounds,
        var_type=var_type,
        var_trans=var_trans,
        n_initial=5,
        max_iter=10,
        seed=7,
        n_jobs=2,
        verbose=False,
    )

    # Must not raise "ValueError: Input X contains NaN".
    opt.optimize()

    col0 = opt.X_[:, 0].astype(float)
    # Natural scale: every stored value lies within the original [0.001, 0.1]
    # bound, never in the log10-internal range [-3, -1].
    assert np.all(col0 >= 0.001 - 1e-9)
    assert np.all(col0 <= 0.1 + 1e-9)
    assert np.all(np.isfinite(col0))
    assert np.all(np.isfinite(opt.y_))


def test_steady_state_matches_sequential_initial_design_scale():
    """Sequential and steady-state must store the seeded initial design in the
    same (natural) scale -- the first ``n_initial`` rows coincide."""
    bounds = [(0.001, 0.1)]
    var_trans = ["log10"]

    def build(n_jobs):
        opt = SpotOptim(
            fun=_obj,
            bounds=bounds,
            var_trans=var_trans,
            n_initial=6,
            max_iter=8,
            seed=11,
            n_jobs=n_jobs,
            verbose=False,
        )
        opt.optimize()
        return np.asarray(opt.X_[:, 0], dtype=float)

    seq = build(1)
    par = build(2)
    # The seeded initial design (same seed) is identical across paths once both
    # store natural scale.
    n = min(6, len(seq), len(par))
    assert np.allclose(np.sort(seq[:n]), np.sort(par[:n]), rtol=1e-6, atol=1e-9)
    # And both lie in the natural bound.
    assert np.all(par >= 0.001 - 1e-9) and np.all(par <= 0.1 + 1e-9)
