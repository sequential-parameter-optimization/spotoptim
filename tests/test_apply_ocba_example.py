# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
from spotoptim import SpotOptim


def test_apply_ocba_example():
    # Case 1: OCBA with sufficient points
    opt = SpotOptim(
        fun=lambda X: np.sum(X**2, axis=1) + np.random.normal(0, 0.1, X.shape[0]),
        bounds=[(-5, 5), (-5, 5)],
        n_initial=5,
        repeats_initial=2,
        ocba_delta=5,
        verbose=False,
    )

    # Simulate optimization state (normally done in optimize())
    opt.mean_X = np.array([[1, 2], [0, 0], [2, 1]])
    opt.mean_y = np.array([5.0, 0.1, 5.0])
    opt.var_y = np.array([0.1, 0.05, 0.15])

    X_ocba = opt._apply_ocba()

    assert X_ocba is not None
    assert X_ocba.shape[0] == 5

    # Case 2: OCBA skipped - insufficient points
    opt2 = SpotOptim(
        fun=lambda X: np.sum(X**2, axis=1),
        bounds=[(-5, 5), (-5, 5)],
        repeats_initial=2,
        ocba_delta=5,
        verbose=False,
    )

    opt2.mean_X = np.array([[1, 2], [0, 0]])
    opt2.mean_y = np.array([5.0, 0.1])
    opt2.var_y = np.array([0.1, 0.05])

    X_ocba = opt2._apply_ocba()

    assert X_ocba is None
