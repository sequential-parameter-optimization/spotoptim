# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np

from spotoptim import SpotOptim


def test_update_storage_steady_example():
    opt = SpotOptim(
        fun=lambda X: np.sum(X**2, axis=1),
        bounds=[(-5, 5), (-5, 5)],
        n_jobs=2,
    )

    opt._update_storage_steady(np.array([1.0, 2.0]), 5.0)

    assert opt.X_.shape == (1, 2)
    assert np.allclose(opt.X_, np.array([[1.0, 2.0]]))
    assert np.allclose(opt.y_, np.array([5.0]))
    assert np.allclose(opt.best_x_, np.array([1.0, 2.0]))
    assert opt.best_y_ == 5.0
