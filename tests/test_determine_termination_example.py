# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import time
import numpy as np
from spotoptim import SpotOptim


def test_determine_termination_example():
    opt = SpotOptim(
        fun=lambda X: np.sum(X**2, axis=1),
        bounds=[(-5, 5), (-5, 5)],
        max_iter=20,
        max_time=10.0,
    )

    # Case 1: Maximum evaluations reached
    opt.y_ = np.zeros(20)
    start_time = time.time()
    msg = opt._determine_termination(start_time)
    assert msg == "Optimization terminated: maximum evaluations (20) reached"

    # Case 2: Time limit exceeded
    opt.y_ = np.zeros(10)
    start_time = time.time() - 700
    msg = opt._determine_termination(start_time)
    assert msg == "Optimization terminated: time limit (10.00 min) reached"

    # Case 3: Successful completion
    opt.y_ = np.zeros(10)
    start_time = time.time()
    msg = opt._determine_termination(start_time)
    assert msg == "Optimization finished successfully"
