# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
from spotoptim import SpotOptim


def test_handle_na_new_points_example():
    opt = SpotOptim(
        fun=lambda X: np.sum(X**2, axis=1),
        bounds=[(-5, 5), (-5, 5)],
        n_initial=5,
        verbose=True,
        penalty=False,
    )

    # Simulate optimization state
    opt.y_ = np.array([1.0, 2.0, 3.0])
    opt.n_iter_ = 1

    # Case 1: Some valid values
    x_next = np.array([[1, 2], [3, 4], [5, 6]])
    y_next = np.array([5.0, np.nan, 10.0])
    x_clean, y_clean = opt._handle_NA_new_points(x_next, y_next)

    assert x_clean.shape == (2, 2)
    assert y_clean.shape == (2,)

    # Case 2: All NaN/inf - should skip iteration
    x_all_bad = np.array([[1, 2], [3, 4]])
    y_all_bad = np.array([np.nan, np.inf])
    x_clean, y_clean = opt._handle_NA_new_points(x_all_bad, y_all_bad)

    assert x_clean is None
    assert y_clean is None
