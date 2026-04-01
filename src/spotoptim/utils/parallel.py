# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import sys

def is_gil_disabled() -> bool:
    """Return True when running on a free-threaded (no-GIL) Python build.
    Uses ``sys.is_gil_enabled()`` (public name) when available, falling back
    to ``sys._is_gil_enabled()`` (CPython 3.13 internal name).  On older
    interpreters the attribute is absent, so the lambda default returns
    ``True`` (GIL enabled), which is the safe fallback.

    Returns:
        bool: ``True`` if the GIL is disabled, ``False`` otherwise.

    Examples:
        ```{python}
        from spotoptim.utils.parallel import is_gil_disabled
        result = is_gil_disabled()
        print(isinstance(result, bool))  # True
        ```
    """
    checker = getattr(sys, "is_gil_enabled", None) or getattr(sys, "_is_gil_enabled", None)
    return not checker() if checker is not None else False


def remote_eval_wrapper(pickled_args):
    """
    Helper function for parallel evaluation with dill.
    Accepts a single argument (pickled tuple) to bypass standard pickling limitations.

    Args:
        pickled_args (bytes): A pickled tuple containing (optimizer, x) where:
            * optimizer: The SpotOptim instance (or surrogate model) to use for evaluation.
            * x: The point at which to evaluate the objective function.

    Returns:
        tuple: A tuple containing:
            * x (ndarray): The input point at which the function was evaluated.
            * y (ndarray or Exception): The function value(s) at x, or an Exception if evaluation failed.

    Raises:
        Exception: Any exception raised during the evaluation of the objective function is caught and returned as part of the output tuple.
        This allows the optimization process to continue even if some evaluations fail, and provides information about the failure for debugging.

    Examples:
        ```{python}
        import numpy as np
        from spotoptim.utils.parallel import remote_eval_wrapper
        class DummyOptimizer:
            def evaluate_function(self, X):
                return np.sum(X**2, axis=1)
        optimizer = DummyOptimizer()
        x = np.array([1.0, 2.0])
        import dill
        pickled_args = dill.dumps((optimizer, x))
        x_eval, y_eval = remote_eval_wrapper(pickled_args)
        print(np.allclose(x_eval, x))
        print(np.isclose(y_eval, 5.0))
        ```
    """
    try:
        # Import dill locally to ensure it's available in workers
        import dill

        optimizer, x = dill.loads(pickled_args)

        # Recast to 2D for evaluate_function
        x_2d = x.reshape(1, -1)
        y_arr = optimizer.evaluate_function(x_2d)
        return x, y_arr[0]
    except Exception as e:
        return None, e


def remote_batch_eval_wrapper(pickled_args):
    """Helper for parallel batch evaluation with dill.
    Evaluates a batch of candidate points in a single call to ``fun(X_batch)``,
    spreading process-spawn and IPC overhead across the whole batch.

    Args:
        pickled_args (bytes): A pickled tuple ``(optimizer, X_batch)`` where
            ``X_batch`` has shape ``(n, d)`` — ``n`` candidate points, ``d``
            dimensions each.

    Returns:
        tuple: ``(X_batch, y_batch)`` on success where ``y_batch`` has shape
            ``(n,)``, or ``(None, Exception)`` on failure.

    Examples:
        ```{python}
        import numpy as np
        from spotoptim.utils.parallel remote_batch_eval_wrapper
        from spotoptim import SpotOptim
        import dill

        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            n_initial=5,
            max_iter=10,
        )
        X_batch = np.array([[1.0, 2.0], [0.5, -0.5]])
        pickled_args = dill.dumps((opt, X_batch))
        X_out, y_out = remote_batch_eval_wrapper(pickled_args)
        print(X_out.shape)   # (2, 2)
        print(y_out.shape)   # (2,)
        print(np.allclose(y_out, [5.0, 0.5]))
        ```
    """
    try:
        import dill

        optimizer, X_batch = dill.loads(pickled_args)
        y_batch = optimizer.evaluate_function(X_batch)
        return X_batch, y_batch
    except Exception as e:
        return None, e


def remote_search_task(pickled_optimizer):
    """
    Helper function for parallel search with dill.

    Args:
        pickled_optimizer (bytes): A pickled SpotOptim instance that has been initialized with data
            and a fitted surrogate model (via X_, y_, and _fit_surrogate).

    Returns:
        ndarray or Exception: The suggested next infill point(s) as an array of shape (n_infill_points, n_features),
            or an Exception if the operation failed. When n_infill_points=1 (default), shape is (1, n_features).

    Raises:
        Exception: Any exception raised during the operation is caught and returned rather than raised,
            allowing parallel execution to continue. The calling code can check if the return value is an
            Exception instance and handle it appropriately.

    Examples:
        ```{python}
        import numpy as np
        from spotoptim.utils.parallel remote_search_task
        from spotoptim import SpotOptim
        import dill

        # Create and initialize an optimizer
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            n_initial=5,
            max_iter=10,
        )
        # Initialize with some data
        np.random.seed(0)
        opt.X_ = np.random.rand(10, 2) * 10 - 5
        opt.y_ = np.sum(opt.X_**2, axis=1)
        opt._fit_surrogate(opt.X_, opt.y_)

        # Use the function
        pickled_optimizer = dill.dumps(opt)
        x_next = remote_search_task(pickled_optimizer)
        isinstance(x_next, np.ndarray)
        x_next.shape
        # Verify the point is within bounds
        (-5 <= x_next[0, 0] <= 5) and (-5 <= x_next[0, 1] <= 5)
        ```
    """
    try:
        import dill

        optimizer = dill.loads(pickled_optimizer)
        x_new = optimizer.suggest_next_infill_point()
        return x_new
    except Exception as e:
        return e
