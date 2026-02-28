# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import shutil
import os
from spotoptim import SpotOptim


def test_example_1_basic_usage():
    def objective(X):
        return np.sum(X**2, axis=1)

    bounds = [(-5, 5), (-5, 5)]
    optimizer = SpotOptim(
        fun=objective, bounds=bounds, max_iter=3, n_initial=2, verbose=False
    )
    result = optimizer.optimize()
    assert result.x is not None
    assert result.fun is not None


def test_example_2_custom_names():
    def objective(X):
        return np.sum(X**2, axis=1)

    optimizer = SpotOptim(
        fun=objective,
        bounds=[(-5, 5), (-5, 5)],
        var_name=["param1", "param2"],
        max_iter=3,
        n_initial=2,
    )
    _ = optimizer.optimize()
    optimizer.plot_surrogate(show=False)
    assert "param1" in optimizer.var_name
    assert "param2" in optimizer.var_name


def test_example_3_noisy_objective():
    def noisy_objective(X):
        base = np.sum(X**2, axis=1)
        noise = np.random.normal(0, 0.1, size=base.shape)
        return base + noise

    optimizer = SpotOptim(
        fun=noisy_objective,
        bounds=[(-5, 5), (-5, 5)],
        max_iter=6,
        n_initial=3,
        repeats_initial=2,
        repeats_surrogate=2,
        seed=42,
        verbose=False,
    )
    _ = optimizer.optimize()
    assert optimizer.mean_X is not None
    assert optimizer.min_mean_y is not None
    assert optimizer.min_var_y is not None


def test_example_4_ocba():
    def noisy_objective(X):
        base = np.sum(X**2, axis=1)
        noise = np.random.normal(0, 0.1, size=base.shape)
        return base + noise

    optimizer_ocba = SpotOptim(
        fun=noisy_objective,
        bounds=[(-5, 5), (-5, 5)],
        max_iter=10,
        n_initial=3,
        repeats_initial=2,
        repeats_surrogate=1,
        ocba_delta=2,
        seed=42,
        verbose=False,
    )
    result = optimizer_ocba.optimize()
    assert result.nfev > 0
    assert optimizer_ocba.mean_X is not None
    assert optimizer_ocba.min_mean_y is not None


def test_example_5_tensorboard():
    def objective(X):
        return np.sum(X**2, axis=1)

    tb_dir = "runs/test_example_5_tb"
    optimizer_tb = SpotOptim(
        fun=objective,
        bounds=[(-5, 5), (-5, 5)],
        max_iter=3,
        n_initial=2,
        tensorboard_log=True,
        tensorboard_path=tb_dir,
        verbose=False,
    )
    _ = optimizer_tb.optimize()

    assert os.path.exists(tb_dir)
    if os.path.exists(tb_dir):
        shutil.rmtree(tb_dir)


def test_example_6_kriging():
    from spotoptim.surrogate import Kriging

    def objective(X):
        return np.sum(X**2, axis=1)

    kriging_model = Kriging(
        noise=1e-10, kernel="gauss", min_theta=-3.0, max_theta=2.0, seed=42
    )
    optimizer_kriging = SpotOptim(
        fun=objective,
        bounds=[(-5, 5), (-5, 5)],
        surrogate=kriging_model,
        max_iter=3,
        n_initial=2,
        seed=42,
        verbose=False,
    )
    result = optimizer_kriging.optimize()
    assert result.x is not None


def test_example_7_custom_gp():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    def objective(X):
        return np.sum(X**2, axis=1)

    custom_kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(
        length_scale=1.0, length_scale_bounds=(1e-1, 10.0)
    ) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))

    gp_custom = GaussianProcessRegressor(
        kernel=custom_kernel, n_restarts_optimizer=2, normalize_y=True, random_state=42
    )

    optimizer_custom_gp = SpotOptim(
        fun=objective,
        bounds=[(-5, 5), (-5, 5)],
        surrogate=gp_custom,
        max_iter=3,
        n_initial=2,
        seed=42,
    )
    result = optimizer_custom_gp.optimize()
    assert result.fun is not None


def test_example_8_random_forest():
    from sklearn.ensemble import RandomForestRegressor

    def objective(X):
        return np.sum(X**2, axis=1)

    rf_model = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=42)

    optimizer_rf = SpotOptim(
        fun=objective,
        bounds=[(-5, 5), (-5, 5)],
        surrogate=rf_model,
        max_iter=3,
        n_initial=2,
        seed=42,
    )
    result = optimizer_rf.optimize()
    assert result.fun is not None


def test_example_9_different_kernels():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel

    def objective(X):
        return np.sum(X**2, axis=1)

    kernel_rbf = ConstantKernel(1.0) * RBF(length_scale=1.0)
    gp_rbf = GaussianProcessRegressor(kernel=kernel_rbf, normalize_y=True)

    optimizer_rbf = SpotOptim(
        fun=objective,
        bounds=[(-5, 5), (-5, 5)],
        surrogate=gp_rbf,
        max_iter=3,
        n_initial=2,
    )
    result = optimizer_rbf.optimize()
    assert result.fun is not None
