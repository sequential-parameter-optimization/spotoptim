# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import time

import numpy as np

from spotoptim import SpotOptim


def test_run_sequential_loop_example():
    opt = SpotOptim(
        fun=lambda X: np.sum(X**2, axis=1),
        bounds=[(-5, 5), (-5, 5)],
        n_initial=5,
        max_iter=20,
        seed=0,
        n_jobs=1,
        verbose=True,
    )

    X0, y0 = opt._initialize_run(X0=None, y0_known=None)
    X0, y0, n_evaluated = opt._rm_NA_values(X0, y0)
    opt._check_size_initial_design(y0, n_evaluated)
    opt._init_storage(X0, y0)
    opt._zero_success_count = 0
    opt._success_history = []
    opt.update_stats()
    opt._get_best_xy_initial_design()

    status, result = opt._run_sequential_loop(
        timeout_start=time.time(), effective_max_iter=20
    )

    assert status == "FINISHED"
    assert (
        result.message.splitlines()[0]
        == "Optimization terminated: maximum evaluations (20) reached"
    )
    assert result.nfev == 20
