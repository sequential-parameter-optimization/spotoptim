# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import time

import numpy as np

from spotoptim import SpotOptim


def test_optimize_sequential_run_example():
    opt = SpotOptim(
        fun=lambda X: np.sum(X**2, axis=1),
        bounds=[(-5, 5), (-5, 5)],
        n_initial=5,
        max_iter=20,
        seed=0,
        n_jobs=1,
        verbose=False,
    )

    status, result = opt._optimize_sequential_run(timeout_start=time.time())

    assert status == "FINISHED"
    assert result.message.splitlines()[0] == (
        "Optimization terminated: maximum evaluations (20) reached"
    )
    assert result.nfev == len(result.y)
