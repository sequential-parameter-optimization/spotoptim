# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for SpotOptim.init_surrogate()."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

from spotoptim import SpotOptim


def _sphere(X):
    return np.sum(X**2, axis=1)


# ---------------------------------------------------------------------------
# 1. Default surrogate (None → GaussianProcessRegressor)
# ---------------------------------------------------------------------------
class TestDefaultSurrogate:
    def test_creates_gpr(self):
        opt = SpotOptim(fun=_sphere, bounds=[(-5, 5), (-5, 5)], n_initial=5)
        assert isinstance(opt.surrogate, GaussianProcessRegressor)

    def test_gpr_kernel_is_matern(self):
        opt = SpotOptim(fun=_sphere, bounds=[(-5, 5), (-5, 5)], n_initial=5)
        kernel_str = str(opt.surrogate.kernel)
        assert "Matern" in kernel_str

    def test_gpr_normalize_y(self):
        opt = SpotOptim(fun=_sphere, bounds=[(-5, 5), (-5, 5)], n_initial=5)
        assert opt.surrogate.normalize_y is True

    def test_internal_attrs_none_for_default(self):
        opt = SpotOptim(fun=_sphere, bounds=[(-5, 5), (-5, 5)], n_initial=5)
        assert opt._surrogates_list is None
        assert opt._prob_surrogate is None


# ---------------------------------------------------------------------------
# 2. User-provided single surrogate
# ---------------------------------------------------------------------------
class TestUserProvidedSurrogate:
    def test_custom_surrogate_preserved(self):
        rf = RandomForestRegressor(n_estimators=10, random_state=0)
        opt = SpotOptim(
            fun=_sphere,
            bounds=[(-5, 5), (-5, 5)],
            n_initial=5,
            surrogate=rf,
        )
        assert opt.surrogate is rf

    def test_surrogates_list_is_none(self):
        rf = RandomForestRegressor(n_estimators=10, random_state=0)
        opt = SpotOptim(
            fun=_sphere,
            bounds=[(-5, 5), (-5, 5)],
            n_initial=5,
            surrogate=rf,
        )
        assert opt._surrogates_list is None
        assert opt._prob_surrogate is None


# ---------------------------------------------------------------------------
# 3. List of surrogates
# ---------------------------------------------------------------------------
class TestSurrogateList:
    def test_multi_surrogate_setup(self):
        surrogates = [GaussianProcessRegressor(), RandomForestRegressor()]
        opt = SpotOptim(
            fun=_sphere,
            bounds=[(-5, 5), (-5, 5)],
            n_initial=5,
            surrogate=surrogates,
        )
        # Active surrogate should be the first in the list
        assert isinstance(opt.surrogate, GaussianProcessRegressor)
        assert opt._surrogates_list is surrogates
        assert len(opt._prob_surrogate) == 2

    def test_uniform_probabilities_by_default(self):
        surrogates = [GaussianProcessRegressor(), RandomForestRegressor()]
        opt = SpotOptim(
            fun=_sphere,
            bounds=[(-5, 5), (-5, 5)],
            n_initial=5,
            surrogate=surrogates,
        )
        assert opt._prob_surrogate == [0.5, 0.5]

    def test_custom_probabilities(self):
        surrogates = [GaussianProcessRegressor(), RandomForestRegressor()]
        opt = SpotOptim(
            fun=_sphere,
            bounds=[(-5, 5), (-5, 5)],
            n_initial=5,
            surrogate=surrogates,
            prob_surrogate=[0.7, 0.3],
        )
        assert opt._prob_surrogate == [0.7, 0.3]

    def test_probabilities_normalized(self):
        surrogates = [GaussianProcessRegressor(), RandomForestRegressor()]
        opt = SpotOptim(
            fun=_sphere,
            bounds=[(-5, 5), (-5, 5)],
            n_initial=5,
            surrogate=surrogates,
            prob_surrogate=[2.0, 8.0],
        )
        assert np.isclose(opt._prob_surrogate[0], 0.2)
        assert np.isclose(opt._prob_surrogate[1], 0.8)

    def test_max_surrogate_points_broadcast(self):
        surrogates = [GaussianProcessRegressor(), RandomForestRegressor()]
        opt = SpotOptim(
            fun=_sphere,
            bounds=[(-5, 5), (-5, 5)],
            n_initial=5,
            surrogate=surrogates,
            max_surrogate_points=50,
        )
        assert opt._max_surrogate_points_list == [50, 50]

    def test_max_surrogate_points_per_surrogate(self):
        surrogates = [GaussianProcessRegressor(), RandomForestRegressor()]
        opt = SpotOptim(
            fun=_sphere,
            bounds=[(-5, 5), (-5, 5)],
            n_initial=5,
            surrogate=surrogates,
            max_surrogate_points=[30, 60],
        )
        assert opt._max_surrogate_points_list == [30, 60]
        assert opt._active_max_surrogate_points == 30


# ---------------------------------------------------------------------------
# 4. Error cases
# ---------------------------------------------------------------------------
class TestInitSurrogateErrors:
    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            SpotOptim(
                fun=_sphere,
                bounds=[(-5, 5), (-5, 5)],
                n_initial=5,
                surrogate=[],
            )

    def test_prob_surrogate_length_mismatch(self):
        surrogates = [GaussianProcessRegressor(), RandomForestRegressor()]
        with pytest.raises(ValueError, match="prob_surrogate"):
            SpotOptim(
                fun=_sphere,
                bounds=[(-5, 5), (-5, 5)],
                n_initial=5,
                surrogate=surrogates,
                prob_surrogate=[0.5],
            )

    def test_max_surrogate_points_list_mismatch(self):
        surrogates = [GaussianProcessRegressor(), RandomForestRegressor()]
        with pytest.raises(ValueError, match="max_surrogate_points"):
            SpotOptim(
                fun=_sphere,
                bounds=[(-5, 5), (-5, 5)],
                n_initial=5,
                surrogate=surrogates,
                max_surrogate_points=[10],
            )


# ---------------------------------------------------------------------------
# 5. Idempotency — calling init_surrogate() twice doesn't corrupt state
# ---------------------------------------------------------------------------
class TestIdempotency:
    def test_double_call_default(self):
        opt = SpotOptim(fun=_sphere, bounds=[(-5, 5), (-5, 5)], n_initial=5)
        assert isinstance(opt.surrogate, GaussianProcessRegressor)
        # Set surrogate to None and reinitialize
        opt.surrogate = None
        opt.init_surrogate()
        assert isinstance(opt.surrogate, GaussianProcessRegressor)

    def test_double_call_list(self):
        surrogates = [GaussianProcessRegressor(), RandomForestRegressor()]
        opt = SpotOptim(
            fun=_sphere,
            bounds=[(-5, 5), (-5, 5)],
            n_initial=5,
            surrogate=surrogates,
        )
        # Re-wrap in list and call again
        opt.surrogate = surrogates
        opt.init_surrogate()
        assert isinstance(opt.surrogate, GaussianProcessRegressor)
        assert opt._surrogates_list is surrogates
