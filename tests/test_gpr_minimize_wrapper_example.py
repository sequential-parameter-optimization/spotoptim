# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
from spotoptim.SpotOptim import _gpr_minimize_wrapper


def test_gpr_minimize_wrapper_example():
    """Test _gpr_minimize_wrapper with L-BFGS-B method."""

    def obj_func(theta):
        value = np.sum(theta**2)
        grad = 2 * theta
        return value, grad

    initial_theta = np.array([1.0, 1.0])
    bounds = [(-5, 5), (-5, 5)]
    theta_opt, func_min = _gpr_minimize_wrapper(
        obj_func, initial_theta, bounds, method="L-BFGS-B", options={"maxiter": 100}
    )

    # Optimized theta should be close to [0, 0]
    assert np.allclose(theta_opt, [0.0, 0.0], atol=1e-6)

    # Minimum function value should be close to 0
    assert np.isclose(func_min, 0.0, atol=1e-10)


def test_gpr_minimize_wrapper_nelder_mead():
    """Test _gpr_minimize_wrapper with gradient-free method (Nelder-Mead)."""

    def obj_func(theta):
        """Objective function that returns value and gradient."""
        value = np.sum(theta**2)
        grad = 2 * theta
        return value, grad

    initial_theta = np.array([2.0, 3.0])
    bounds = [(-5, 5), (-5, 5)]

    # Nelder-Mead is gradient-free, should only use function values
    theta_opt, func_min = _gpr_minimize_wrapper(
        obj_func, initial_theta, bounds, method="Nelder-Mead", options={"maxiter": 200}
    )

    # Should find minimum near [0, 0]
    assert np.allclose(theta_opt, [0.0, 0.0], atol=1e-4)
    assert func_min < 1e-6


def test_gpr_minimize_wrapper_default_method():
    """Test _gpr_minimize_wrapper with default method (L-BFGS-B)."""

    def obj_func(theta):
        value = np.sum((theta - 1.0) ** 2)
        grad = 2 * (theta - 1.0)
        return value, grad

    initial_theta = np.array([0.0, 0.0])
    bounds = [(-5, 5), (-5, 5)]

    # Default method should be L-BFGS-B
    theta_opt, func_min = _gpr_minimize_wrapper(obj_func, initial_theta, bounds)

    # Should find minimum at [1, 1]
    assert np.allclose(theta_opt, [1.0, 1.0], atol=1e-6)
    assert np.isclose(func_min, 0.0, atol=1e-10)
