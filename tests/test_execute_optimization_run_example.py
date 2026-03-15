# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import time
import numpy as np
from spotoptim import SpotOptim


def testexecute_optimization_run_example():
    opt = SpotOptim(
        fun=lambda X: np.sum(X**2, axis=1),
        bounds=[(-5, 5), (-5, 5)],
        n_initial=5,
        max_iter=10,
        seed=0,
        n_jobs=1,
        verbose=False,
    )

    status, result = opt.execute_optimization_run(timeout_start=time.time())

    assert status == "FINISHED"
    assert result.message.splitlines()[0] == (
        "Optimization terminated: maximum evaluations (10) reached"
    )
    assert result.nfev == len(result.y)
