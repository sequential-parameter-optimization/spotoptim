# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import time

import numpy as np

from spotoptim import SpotOptim


def test_optimize_steady_state_example():
    opt = SpotOptim(
        fun=lambda X: np.sum(X**2, axis=1),
        bounds=[(-5, 5), (-5, 5)],
        n_initial=5,
        max_iter=10,
        seed=0,
        n_jobs=2,
        verbose=True,
    )

    status, result = opt._optimize_steady_state(timeout_start=time.time(), X0=None)

    assert status == "FINISHED"
    assert result.message.splitlines()[0] == "Optimization finished (Steady State)"
    assert result.nfev == 10
