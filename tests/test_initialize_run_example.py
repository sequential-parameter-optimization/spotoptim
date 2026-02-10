# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
from spotoptim import SpotOptim


def test_initialize_run_example():
    opt = SpotOptim(
        fun=lambda X: np.sum(X**2, axis=1),
        bounds=[(-5, 5), (-5, 5)],
        n_initial=5,
        seed=0,
        x0=np.array([0.0, 0.0]),
        verbose=False,
    )

    X0, y0 = opt._initialize_run(X0=None, y0_known=None)

    assert X0.shape == (5, 2)
    assert np.allclose(y0, np.sum(X0**2, axis=1))
