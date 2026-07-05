# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the cold-start n_initial guidance warning (ADR 2026-07-05, D4)."""

import warnings

import numpy as np
import pytest
from spotoptim import SpotOptim


def _sphere(X):
    return np.sum(np.asarray(X, dtype=float) ** 2, axis=1)


class TestNInitialColdStartWarning:
    """n_initial < 2*n_dim warns; n_initial >= 2*n_dim stays silent."""

    def test_warns_below_two_times_n_dim(self):
        with pytest.warns(UserWarning, match=r"n_initial \(3\) is below 2 \* n_dim"):
            SpotOptim(
                fun=_sphere,
                bounds=[(0.0, 1.0)] * 4,
                n_initial=3,
                max_iter=10,
                seed=42,
            )

    def test_message_recommends_dim_aware_size(self):
        with pytest.warns(UserWarning, match=r"max\(10, 2 \* n_dim\) = 10"):
            SpotOptim(
                fun=_sphere,
                bounds=[(0.0, 1.0)] * 4,
                n_initial=3,
                max_iter=10,
                seed=42,
            )

    def test_silent_at_two_times_n_dim(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            SpotOptim(
                fun=_sphere,
                bounds=[(0.0, 1.0)] * 4,
                n_initial=8,
                max_iter=20,
                seed=42,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
