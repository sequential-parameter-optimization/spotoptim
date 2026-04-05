# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from scipy.optimize import minimize


def gpr_minimize_wrapper(obj_func, initial_theta, bounds, **kwargs):
    """
    Wrapper for scipy.optimize.minimize to be used with sklearn GaussianProcessRegressor.
    This function allows passing additional options (kwargs) to the optimizer.
    It automatically handles the `jac` parameter and objective function wrapping
    based on the specified optimization method.

    Args:
        obj_func (callable): The objective function to minimize. It should accept the
            parameter vector as its first argument and return the scalar function value
            and its gradient (tuple).
        initial_theta (np.ndarray): Initial guess for the hyperparameters (theta).
        bounds (list of tuple): Bounds for the hyperparameters.
        **kwargs: Arbitrary keyword arguments passed directly to `scipy.optimize.minimize`.
            Examples:
                method="Nelder-Mead"
                options={'maxiter': 100}

    Returns:
        tuple: A tuple containing:
            * theta_opt (np.ndarray): The optimized hyperparameters.
            * func_min (float): The value of the objective function at the minimum.

    Raises:
        ValueError: If an unsupported optimization method is specified.

    Examples:
        ```{python}
        import numpy as np
        from spotoptim.optimizer.wrapper import gpr_minimize_wrapper
        def obj_func(theta):
            value = np.sum(theta**2)
            grad = 2 * theta
            return value, grad
        initial_theta = np.array([1.0, 1.0])
        bounds = [(-5, 5), (-5, 5)]
        theta_opt, func_min = gpr_minimize_wrapper(
            obj_func,
            initial_theta,
            bounds,
            method="L-BFGS-B",
            options={'maxiter': 100}
        )
        print(np.allclose(theta_opt, [0.0, 0.0], atol=1e-6))
        print(np.isclose(func_min, 0.0, atol=1e-10))
        ```
    """
    # Default parameters if not specified
    if "method" not in kwargs:
        kwargs["method"] = "L-BFGS-B"

    # Clean kwargs for minimize
    # 'minimize' only accepts specific top-level arguments.
    # If users pass DE arguments (like maxiter, popsize) in acquisition_optimizer_kwargs,
    # they might end up here. We should move known option-like args to 'options' or ignore them
    # if they are specialized for another optimizer (like popsize for DE).
    # However, 'maxiter' is a valid option for almost all minimize methods, so we move it to options.

    valid_minimize_args = {
        "fun",
        "x0",
        "args",
        "method",
        "jac",
        "hess",
        "hessp",
        "bounds",
        "constraints",
        "tol",
        "callback",
        "options",
    }

    minimize_kwargs = {}
    options = kwargs.get("options", {}).copy()

    for k, v in kwargs.items():
        if k in valid_minimize_args:
            minimize_kwargs[k] = v
        else:
            # Assume unknown kwargs are options (e.g. maxiter, disp, ftol)
            # This allows sharing kwargs like {'maxiter': 100} between DE and L-BFGS-B
            options[k] = v

    minimize_kwargs["options"] = options
    method = minimize_kwargs["method"]

    # Methods that do NOT optimize based on gradients (gradient-free)
    # These methods cannot handle the (val, grad) tuple return from obj_func.
    gradient_free_methods = ["Nelder-Mead", "Powell", "COBYLA"]

    if method in gradient_free_methods:
        # Wrap obj_func to return only the scalar value
        def obj_func_wrapper(theta):
            return obj_func(theta)[0]

        # Call minimize without jac=True
        res = minimize(
            obj_func_wrapper, initial_theta, bounds=bounds, **minimize_kwargs
        )
    else:
        # Gradient-based methods (L-BFGS-B, BFGS, etc.)
        # Default behavior: use gradients provided by obj_func
        if "jac" not in minimize_kwargs:
            minimize_kwargs["jac"] = True

        res = minimize(obj_func, initial_theta, bounds=bounds, **minimize_kwargs)
    return res.x, res.fun
