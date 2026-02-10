# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import numpy as np

from spotoptim import SpotOptim


def test_suggest_next_infill_point_example():
    opt = SpotOptim(
        fun=lambda X: np.sum(X**2, axis=1),
        bounds=[(-5, 5), (-5, 5)],
        n_initial=5,
        n_infill_points=2,
    )

    np.random.seed(0)
    opt.X_ = np.random.rand(10, 2)
    opt.y_ = np.random.rand(10)
    opt._fit_surrogate(opt.X_, opt.y_)

    x_next = opt.suggest_next_infill_point()

    assert x_next.shape == (2, 2)
