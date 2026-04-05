# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import random
import torch
from functools import partial
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, List, Any, Dict, Union, Literal
from scipy.optimize import OptimizeResult
from scipy.stats.qmc import LatinHypercube
from scipy.stats import norm
from sklearn.base import BaseEstimator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
import warnings
import matplotlib.pyplot as plt
from numpy import append
import time
import os
from sklearn.cluster import KMeans
from spotoptim.plot.visualization import (
    plot_surrogate,
    plot_progress,
    plot_important_hyperparameter_contour,
    _plot_surrogate_with_factors,
    _generate_mesh_grid,
    _generate_mesh_grid_with_factors,
)
from spotoptim.utils.convert import safe_float
from spotoptim.utils import tensorboard as _tb
from spotoptim.utils import ocba as _ocba
from spotoptim.utils import serialization as _serial
from spotoptim.reporting import results as _results
from spotoptim.reporting import analysis as _analysis
from spotoptim.utils import variables as _vars
from spotoptim.utils import transform as _trans
from spotoptim.utils import dimreduction as _dimred
from spotoptim.optimizer import acquisition as _acq
from spotoptim.core import storage as _storage
from spotoptim.optimizer import steady_state as _steady
from spotoptim.optimizer.wrapper import gpr_minimize_wrapper


@dataclass
class SpotOptimConfig:
    """Configuration parameters for SpotOptim.

    Attributes:
        bounds (Optional[list]): Bounds for the input variables.
        max_iter (int): Maximum number of iterations.
        n_initial (int): Number of initial points.
        surrogate (Optional[object]): Surrogate model.
        acquisition (str): Acquisition function.
        var_type (Optional[list]): Type of variables.
        var_name (Optional[list]): Name of variables.
        var_trans (Optional[list]): Transformation of variables.
        tolerance_x (Optional[float]): Tolerance for input variables.
        max_time (float): Maximum time.
        repeats_initial (int): Number of repeats for initial points.
        repeats_surrogate (int): Number of repeats for surrogate points.
        ocba_delta (int): Delta for OCBA.
        tensorboard_log (bool): Whether to log to TensorBoard.
        tensorboard_path (Optional[str]): Path to TensorBoard logs.
        tensorboard_clean (bool): Whether to clean TensorBoard logs.
        fun_mo2so (Optional[Callable]): Function to convert multi-objective to single-objective.
        seed (Optional[int]): Seed for random number generator.
        verbose (bool): Whether to print verbose output.
        warnings_filter (Literal["default", "error", "ignore"]): Filter for warnings.
        n_infill_points (int): Number of infill points.
        max_surrogate_points (Optional[Union[int, List[int]]]): Maximum number of surrogate points.
        selection_method (str): Method for selecting infill points.
        acquisition_failure_strategy (str): Strategy for handling acquisition function failures.
        penalty (bool): Whether to use penalty.
        penalty_val (Optional[float]): Penalty value.
        acquisition_fun_return_size (int): Size of the acquisition function return.
        acquisition_optimizer (Union[str, Callable]): Optimizer for the acquisition function.
        restart_after_n (int): Number of iterations after which to restart.
        restart_inject_best (bool): Whether to inject the best point after restart.
        x0 (Optional[np.ndarray]): Initial guess for the input variables.
        de_x0_prob (float): Probability of using differential evolution for initial guess.
        tricands_fringe (bool): Whether to use fringe for tricands.
        prob_de_tricands (float): Probability of using tricands for differential evolution.
        window_size (Optional[int]): Size of the window for tricands.
        min_tol_metric (str): Metric for minimum tolerance.
        prob_surrogate (Optional[List[float]]): Probability of using surrogate.
        n_jobs (int): Number of parallel workers. ``1`` runs sequentially.
            Values ``> 1`` activate steady-state parallel optimization.
            On standard GIL builds a hybrid executor is used:
            ``ProcessPoolExecutor`` for objective evaluations (process
            isolation; supports lambdas and closures via ``dill``) and
            ``ThreadPoolExecutor`` for surrogate search tasks (shared heap;
            zero serialization overhead).
            On free-threaded Python builds (``python3.13t``,
            ``--disable-gil``), both pools are ``ThreadPoolExecutor``
            instances, achieving true CPU-level parallelism without ``dill``
            for eval tasks.
            ``-1`` resolves to ``os.cpu_count()`` (all available CPU cores).
            ``0`` and values ``< -1`` raise ``ValueError``.
            Defaults to ``1``.
        eval_batch_size (int): Number of candidate points to accumulate before
            dispatching a single ``fun(X_batch)`` call to the process pool.
            ``1`` (default) dispatches each candidate immediately, preserving
            current behavior. Values ``> 1`` reduce process-spawn and IPC
            overhead when ``fun`` supports vectorized batch input.
            Must be ``>= 1``. Defaults to ``1``.
        acquisition_optimizer_kwargs (Optional[Dict[str, Any]]): Keyword arguments for the acquisition function optimizer.
        args (Tuple): Arguments for the objective function.
        kwargs (Optional[Dict[str, Any]]): Keyword arguments for the objective function.


    Examples:
        ```{python}
        import numpy as np
        from spotoptim.SpotOptim import SpotOptim
        from sklearn.gaussian_process import GaussianProcessRegressor
        from spotoptim.SpotOptim import SpotOptimConfig
        from sklearn.gaussian_process.kernels import Matern, ConstantKernel

        kernel = ConstantKernel(1.0, (1e-2, 1e12)) * Matern(length_scale=1.0, length_scale_bounds=(1e-4, 1e2), nu=2.5)
        surrogate = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100)

        config = SpotOptimConfig(
             bounds=[(0, 1), (0, 1)],
             max_iter=20,
             n_initial=10,
             surrogate=surrogate,
             acquisition="y",
             var_type=["continuous", "continuous"],
             var_name=["x1", "x2"],
             var_trans=["identity", "identity"],
             tolerance_x=1e-6,
             max_time=np.inf,
             repeats_initial=1,
             repeats_surrogate=1,
             ocba_delta=0,
             tensorboard_log=False,
             tensorboard_path=None,
             tensorboard_clean=False,
             fun_mo2so=None,
             seed=None,
             verbose=False,
             warnings_filter="ignore",
             n_infill_points=1,
             max_surrogate_points=None,
             selection_method="distant",
             acquisition_failure_strategy="random",
             penalty=False,
             penalty_val=None,
             acquisition_fun_return_size=3,
             acquisition_optimizer="differential_evolution",
             restart_after_n=100,
             restart_inject_best=True,
             x0=None,
             de_x0_prob=0.1,
             tricands_fringe=False,
             prob_de_tricands=0.8,
             window_size=None,
             min_tol_metric="chebyshev",
             prob_surrogate=None,
             n_jobs=1,
             acquisition_optimizer_kwargs=None,
             args=(),
             kwargs=None,
        )
        ```
    """

    bounds: Optional[list] = None
    max_iter: int = 20
    n_initial: int = 10
    surrogate: Optional[object] = None
    acquisition: str = "y"
    var_type: Optional[list] = None
    var_name: Optional[list] = None
    var_trans: Optional[list] = None
    tolerance_x: Optional[float] = None
    max_time: float = np.inf
    repeats_initial: int = 1
    repeats_surrogate: int = 1
    ocba_delta: int = 0
    tensorboard_log: bool = False
    tensorboard_path: Optional[str] = None
    tensorboard_clean: bool = False
    fun_mo2so: Optional[Callable] = None
    seed: Optional[int] = None
    verbose: bool = False
    warnings_filter: Literal["default", "error", "ignore"] = "ignore"
    n_infill_points: int = 1
    max_surrogate_points: Optional[Union[int, List[int]]] = None
    selection_method: str = "distant"
    acquisition_failure_strategy: str = "random"
    penalty: bool = False
    penalty_val: Optional[float] = None
    acquisition_fun_return_size: int = 3
    acquisition_optimizer: Union[str, Callable] = "differential_evolution"
    restart_after_n: int = 100
    restart_inject_best: bool = True
    x0: Optional[np.ndarray] = None
    de_x0_prob: float = 0.1
    tricands_fringe: bool = False
    prob_de_tricands: float = 0.8
    window_size: Optional[int] = None
    min_tol_metric: str = "chebyshev"
    prob_surrogate: Optional[List[float]] = None
    n_jobs: int = 1
    eval_batch_size: int = 1
    acquisition_optimizer_kwargs: Optional[Dict[str, Any]] = None
    args: Tuple = ()
    kwargs: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}


@dataclass
class SpotOptimState:
    """Mutable state of the optimization process.

    Attributes:
        X_ (np.ndarray): Input data.
        y_ (np.ndarray): Output data.
        y_mo (np.ndarray): Multi-objective output data.
        best_x_ (np.ndarray): Best input data.
        best_y_ (float): Best output data.
        n_iter_ (int): Number of iterations.
        counter (int): Counter.
        success_rate (float): Success rate.
        success_counter (int): Success counter.
        _success_history (List): History of success.
        _zero_success_count (int): Count of zero success.
        mean_X (np.ndarray): Mean of input data.
        mean_y (np.ndarray): Mean of output data.
        var_y (np.ndarray): Variance of output data.
        min_mean_X (np.ndarray): Minimum of mean input data.
        min_mean_y (float): Minimum of mean output data.
        min_var_y (float): Minimum of mean variance of output data.
        min_X (np.ndarray): Minimum of input data.
        min_y (float): Minimum of output data.
        restarts_results_ (List): History of restarts.
    """

    X_: Optional[np.ndarray] = None
    y_: Optional[np.ndarray] = None
    y_mo: Optional[np.ndarray] = None
    best_x_: Optional[np.ndarray] = None
    best_y_: Optional[float] = None
    n_iter_: int = 0
    counter: int = 0

    # Success tracking
    success_rate: float = 0.0
    success_counter: int = 0
    _success_history: List = field(default_factory=list)
    _zero_success_count: int = 0

    # Noise statistics
    mean_X: Optional[np.ndarray] = None
    mean_y: Optional[np.ndarray] = None
    var_y: Optional[np.ndarray] = None
    min_mean_X: Optional[np.ndarray] = None
    min_mean_y: Optional[float] = None
    min_var_y: Optional[float] = None

    # Best found
    min_X: Optional[np.ndarray] = None
    min_y: Optional[float] = None

    # Restart history
    restarts_results_: List = field(default_factory=list)


class SpotOptim(BaseEstimator):
    """SPOT optimizer compatible with scipy.optimize interface.

    Args:
        fun (callable):
            Objective function to minimize. Should accept array of shape (n_samples, n_features).
        bounds (list of tuple):
            Bounds for each dimension as [(low, high), ...].
        max_iter (int, optional):
            Maximum number of total function evaluations (including initial design).
            For example, max_iter=30 with n_initial=10 will perform 10 initial evaluations plus
            20 sequential optimization iterations. Defaults to 20.
        n_initial (int, optional):
            Number of initial design points. Defaults to 10.
        surrogate (object, optional):
            Surrogate model with scikit-learn interface (fit/predict methods).
            If None, uses a Gaussian Process Regressor with Matern kernel. Default configuration::
                * `from sklearn.gaussian_process import GaussianProcessRegressor`
                * `from sklearn.gaussian_process.kernels import Matern, ConstantKernel`
                * `kernel = ConstantKernel(1.0, (1e-2, 1e12)) * Matern(length_scale=1.0, length_scale_bounds=(1e-4, 1e2), nu=2.5)`
                * `surrogate = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100)`

            Alternative surrogates can be provided, including SpotOptim's Kriging model,
            Random Forests, or any scikit-learn compatible regressor. See Examples section.
            Defaults to None (uses default Gaussian Process configuration).
        acquisition (str, optional):
            Acquisition function ('ei', 'y', 'pi'). Defaults to 'y'.
        var_type (list of str, optional):
            Variable types for each dimension. Supported types:
                * 'float': Python floats, continuous optimization (no rounding)
                * 'int': Python int, float values will be rounded to integers
                * 'factor': Unordered categorical data, internally mapped to int values
                  (e.g., "red"->0, "green"->1, etc.)
            Defaults to None (which sets all dimensions to 'float').
        var_name (list of str, optional):
            Variable names for each dimension.
            If None, uses default names ['x0', 'x1', 'x2', ...]. Defaults to None.
        tolerance_x (float, optional):
            Minimum distance between points. Defaults to np.sqrt(np.spacing(1))
        var_trans (list of str, optional):
            Variable transformations for each dimension. Supported:
            It can be one of `id`, `log10`, `log`, `ln`, `sqrt`, `exp`, `square`, `cube`, `inv`, `reciprocal`, or `None`.
            Also supports dynamic strings like `log(x)`, `sqrt(x)`, `pow(x, p)`.
            Defaults to None (no transformations).
        max_time (float, optional):
            Maximum runtime in minutes. If np.inf (default), no time limit.
            The optimization terminates when either max_iter evaluations are reached OR max_time
            minutes have elapsed, whichever comes first. Defaults to np.inf.
        repeats_initial (int, optional):
            Number of times to evaluate each initial design point.
            Useful for noisy objective functions. If > 1, noise handling is activated and
            statistics (mean, variance) are tracked. Defaults to 1.
        repeats_surrogate (int, optional):
            Number of times to evaluate each surrogate-suggested point.
            Useful for noisy objective functions. If > 1, noise handling is activated and
            statistics (mean, variance) are tracked. Defaults to 1.
        ocba_delta (int, optional): Number of additional evaluations to allocate using Optimal Computing
            Budget Allocation (OCBA) when noise handling is active. OCBA determines which existing
            design points should be re-evaluated to best distinguish between alternatives. Only used
            when repeats_surrogate > 1 and ocba_delta > 0. Requires at least 3 design points with
            variance information. Defaults to 0 (no OCBA).
        tensorboard_log (bool, optional):
            Enable TensorBoard logging. If True, optimization metrics
            and hyperparameters are logged to TensorBoard. View logs by running:
            `tensorboard --logdir=<tensorboard_path>` in a separate terminal. Defaults to False.
        tensorboard_path (str, optional):
            Path for TensorBoard log files. If None and tensorboard_log
            is True, creates a default path: runs/spotoptim_YYYYMMDD_HHMMSS. Defaults to None.
        tensorboard_clean (bool, optional):
            If True, removes all old TensorBoard log directories from
            the 'runs' folder before starting optimization. Use with caution as this permanently
            deletes all subdirectories in 'runs'. Defaults to False.
        fun_mo2so (callable, optional):
            Function to convert multi-objective values to single-objective.
            Takes an array of shape (n_samples, n_objectives) and returns array of shape (n_samples,).
            If None and objective function returns multi-objective values, uses first objective.
            Defaults to None.
        seed (int, optional):
            Random seed for reproducibility. Defaults to None.
        verbose (bool, optional):
            Print progress information. Defaults to False.
        warnings_filter (Literal["default", "error", "ignore"], optional):
            Filter for warnings. One of "error", "ignore", "always", "all",
            "default", "module", or "once". Defaults to "ignore".
        n_infill_points (int, optional):
            Number of infill points to suggest at each iteration.
            Defaults to 1. If > 1, multiple distinct points are proposed using the optimizer
            and fallback strategies.
        max_surrogate_points (int, optional):
            Maximum number of points to use for surrogate model fitting.
            If None, all points are used. If the number of evaluated points exceeds this limit,
            a subset is selected using the selection method. Defaults to None.
        selection_method (str, optional):
            Method for selecting points when max_surrogate_points is exceeded.
            Options: 'distant' (Select points that are distant from each other via K-means clustering) or
            'best' (Select all points from the cluster with the best mean objective value).
            Defaults to 'distant'.
        acquisition_failure_strategy (str, optional):
            Strategy for handling acquisition function failures.
            Options: 'random' (space-filling design via Latin Hypercube Sampling)
            Defaults to 'random'.
        penalty (bool, optional):
            Whether to use penalty for handling NaN/inf values in objective function evaluations.
            Defaults to False.
        penalty_val (float, optional):
            Penalty value to replace NaN/inf values in objective function evaluations.
            When the objective function returns NaN or inf, these values are replaced with penalty plus
            a small random noise (sampled from N(0, 0.1)) to avoid identical penalty values.
            This allows optimization to continue despite occasional function evaluation failures.
            Defaults to None.
        acquisition_fun_return_size (int, optional):
            Number of top candidates to return from acquisition function optimization.
            Defaults to 3.
        acquisition_optimizer (str or callable, optional):
            Optimizer to use for maximizing acquisition function.
            Can be "differential_evolution" (default) or any method name supported by scipy.optimize.minimize
            (e.g., "Nelder-Mead", "L-BFGS-B"). Can also be a callable with signature compatible with
            scipy.optimize.minimize (fun, x0, bounds, ...). A specific version is "de_tricands", which combines DE with Tricands.
            It can be parameterized with "prob_de_tricands" (probability of using DE).
            Defaults to "differential_evolution".
        acquisition_optimizer_kwargs (dict, optional):
            Kwargs passed to the acquisition function optimizer
            and GPR surrogate optimizer. Defaults to {'maxiter': 10000, 'gtol': 1e-9}.
        restart_after_n (int, optional):
            Number of consecutive iterations with zero success rate
            before triggering a restart. Defaults to 100.
        restart_inject_best (bool, optional):
            Whether to inject the best solution found so far
            as a starting point for the next restart. Defaults to True.
        x0 (array-like, optional):
            Starting point for optimization, shape (n_features,).
            If provided, this point will be evaluated first and included in the initial design.
            The point should be within the bounds and will be validated before use.
            Defaults to None (no starting point, uses only LHS design).
        de_x0_prob (float, optional):
            Probability of using the best point as starting point for differential evolution.
            Defaults to 0.1.
        tricands_fringe (bool, optional):
            Whether to use the fringe of the design space for the initial design.
            Defaults to False.
        prob_de_tricands (float, optional):
            Probability of using differential evolution as an optimizer
            on the surrogate model. 1 - prob_de_tricands is the probability of using tricands. Defaults to 0.8.
        n_jobs (int, optional):
            Number of parallel workers. ``1`` (default) runs sequentially.
            Values ``> 1`` activate steady-state parallel optimization:
            objective evaluations and acquisition searches are dispatched
            across ``n_jobs`` processes. Pass ``-1`` to use all available
            CPU cores (``os.cpu_count()``). ``0`` and values ``< -1`` raise
            ``ValueError``. Defaults to ``1``.
        eval_batch_size (int, optional):
            Number of candidate points gathered from search tasks before a
            single ``fun(X_batch)`` call is dispatched to the process pool.
            ``1`` (default) preserves one-point-per-call behavior.
            Set to ``n_jobs`` or higher to exploit vectorized objective
            functions and reduce process-spawn overhead. Ignored when
            ``n_jobs == 1``. Must be ``>= 1``. Defaults to ``1``.
        window_size (int, optional):
            Window size for success rate calculation.
        min_tol_metric (str, optional):
            Distance metric used when checking `tolerance_x` for
            duplicate detection. Default is "chebyshev". Supports all metrics from
            scipy.spatial.distance.cdist, including:
                * "chebyshev": L-infinity distance (hypercube). Default. Matches previous behavior.
                * "euclidean": L2 distance (hypersphere).
                * "minkowski": Lp distance (default p=2).
                * "cityblock": Manhattan/L1 distance.
                * "cosine": Cosine distance.
                * "correlation": Correlation distance.
                * "canberra", "braycurtis", "sqeuclidean", etc.

    Attributes:
        X_ (ndarray): All evaluated points, shape (n_samples, n_features).
        y_ (ndarray): Function values at X_, shape (n_samples,). For multi-objective problems,
            these are the converted single-objective values.
        y_mo (ndarray or None): Multi-objective function values, shape (n_samples, n_objectives).
            None for single-objective problems.
        best_x_ (ndarray): Best point found, shape (n_features,).
        best_y_ (float): Best function value found.
        n_iter_ (int): Number of iterations performed. This is not the same as counter. Provided for compatibility with scipy.optimize routines.
        counter (int): Total number of function evaluations.
        success_rate (float): Rolling success rate over the last window_size evaluations.
            A success is counted when a new evaluation improves upon the best value found so far.
        warnings_filter (Literal["default", "error", "ignore"]): Filter for warnings during optimization.
        max_surrogate_points (int or None): Maximum number of points for surrogate fitting.
        selection_method (str): Point selection method.
        acquisition_failure_strategy (str): Strategy for handling acquisition failures ('random').
        mean_X (ndarray or None): Aggregated unique design points (if repeats_surrogate > 1).
        mean_y (ndarray or None): Mean y values per design point (if repeats_surrogate > 1).
        var_y (ndarray or None): Variance of y values per design point (if repeats_surrogate > 1).
        min_mean_X (ndarray or None): X value of best mean y (if repeats_surrogate > 1).
        min_mean_y (float or None): Best mean y value (if repeats_surrogate > 1).
        min_var_y (float or None): Variance of best mean y (if repeats_surrogate > 1).
        de_x0_prob (float): Probability of using the best point as starting point for differential evolution.
        tricands_fringe (bool): Whether to use the fringe of the design space for the initial design.
        prob_de_tricands (float): Probability of using differential evolution as an optimizer on the surrogate model.

    Examples:
        ```{python}
        import numpy as np
        from spotoptim import SpotOptim

        def objective(X):
            return np.sum(X**2, axis=1)

        # Example 1: Basic usage (deterministic function)
        bounds = [(-5, 5), (-5, 5)]
        optimizer = SpotOptim(fun=objective, bounds=bounds, max_iter=10, n_initial=5, verbose=True)
        result = optimizer.optimize()
        print("Best x:", result.x)
        print("Best f(x):", result.fun)
        ```

        ```{python}
        import numpy as np
        from spotoptim import SpotOptim

        def objective(X):
            return np.sum(X**2, axis=1)

        # Example 2: With custom variable names
        optimizer = SpotOptim(
            fun=objective,
            bounds=[(-5, 5), (-5, 5)],
            var_name=["param1", "param2"],
            max_iter=10,
            n_initial=5
        )
        result = optimizer.optimize()
        # Ensure we can use custom names in plots
        optimizer.plot_surrogate(show=False)
        ```

        ```{python}
        import numpy as np
        from spotoptim import SpotOptim

        # Example 3: Noisy function with repeated evaluations
        def noisy_objective(X):
            base = np.sum(X**2, axis=1)
            noise = np.random.normal(0, 0.1, size=base.shape)
            return base + noise

        optimizer = SpotOptim(
            fun=noisy_objective,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=10,
            n_initial=5,
            repeats_initial=1,      # Evaluate each initial point once
            repeats_surrogate=2,    # Evaluate each new point twice
            seed=42,                # For reproducibility
            verbose=True
        )
        result = optimizer.optimize()

        # Access noise statistics
        print("Unique design points:", optimizer.mean_X.shape[0])
        print("Best mean value:", optimizer.min_mean_y)
        print("Variance at best point:", optimizer.min_var_y)
        ```

        ```{python}
        import numpy as np
        from spotoptim import SpotOptim

        def noisy_objective(X):
            base = np.sum(X**2, axis=1)
            noise = np.random.normal(0, 0.1, size=base.shape)
            return base + noise

        # Example 4: Noisy function with OCBA (Optimal Computing Budget Allocation)
        optimizer_ocba = SpotOptim(
            fun=noisy_objective,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=20,
            n_initial=5,
            repeats_initial=2,      # Initial repeats
            repeats_surrogate=1,    # Surrogate repeats
            ocba_delta=3,           # Allocate 3 additional evaluations per iteration
            seed=42,
            verbose=True
        )
        result = optimizer_ocba.optimize()

        # OCBA intelligently re-evaluates promising points to reduce uncertainty
        print("Total evaluations:", result.nfev)
        print("Unique design points:", optimizer_ocba.mean_X.shape[0])
        print("Best mean value:", optimizer_ocba.min_mean_y)
        print("Variance at best point:", optimizer_ocba.min_var_y)
        ```

        ```{python}
        import numpy as np
        import shutil
        import os
        from spotoptim import SpotOptim

        def objective(X):
            return np.sum(X**2, axis=1)

        # Example 5: With TensorBoard logging
        tb_dir = "runs/my_optimization"
        optimizer_tb = SpotOptim(
            fun=objective,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=10,
            n_initial=5,
            tensorboard_log=True,   # Enable TensorBoard
            tensorboard_path=tb_dir,  # Optional custom path
            verbose=True
        )
        result = optimizer_tb.optimize()

        # View logs in browser: tensorboard --logdir=runs/my_optimization
        print("Logs saved to:", optimizer_tb.tensorboard_path)

        # Cleanup log dir
        if os.path.exists(tb_dir):
            shutil.rmtree(tb_dir)
        ```

        ```{python}
        import numpy as np
        from spotoptim import SpotOptim
        from spotoptim.surrogate import Kriging

        def objective(X):
            return np.sum(X**2, axis=1)

        # Example 6: Using SpotOptim's Kriging surrogate
        kriging_model = Kriging(
            noise=1e-10,           # Regularization parameter
            kernel='gauss',         # Gaussian/RBF kernel
            min_theta=-3.0,         # Min log10(theta) bound
            max_theta=2.0,          # Max log10(theta) bound
            seed=42
        )
        optimizer_kriging = SpotOptim(
            fun=objective,
            bounds=[(-5, 5), (-5, 5)],
            surrogate=kriging_model,
            max_iter=10,
            n_initial=5,
            seed=42,
            verbose=True
        )
        result = optimizer_kriging.optimize()
        print("Best solution found:", result.x)
        print("Best value:", result.fun)
        ```

        ```{python}
        import numpy as np
        from spotoptim import SpotOptim
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

        def objective(X):
            return np.sum(X**2, axis=1)

        # Example 7: Using sklearn Gaussian Process with custom kernel
        # Custom kernel: constant * RBF + white noise
        custom_kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(
            length_scale=1.0, length_scale_bounds=(1e-1, 10.0)
        ) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))

        gp_custom = GaussianProcessRegressor(
            kernel=custom_kernel,
            n_restarts_optimizer=15,
            normalize_y=True,
            random_state=42
        )

        optimizer_custom_gp = SpotOptim(
            fun=objective,
            bounds=[(-5, 5), (-5, 5)],
            surrogate=gp_custom,
            max_iter=10,
            n_initial=5,
            seed=42
        )
        result = optimizer_custom_gp.optimize()
        ```

        ```{python}
        import numpy as np
        from spotoptim import SpotOptim
        from sklearn.ensemble import RandomForestRegressor

        def objective(X):
            return np.sum(X**2, axis=1)

        # Example 8: Using Random Forest as surrogate
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )

        optimizer_rf = SpotOptim(
            fun=objective,
            bounds=[(-5, 5), (-5, 5)],
            surrogate=rf_model,
            max_iter=10,
            n_initial=5,
            seed=42
        )
        result = optimizer_rf.optimize()

        # Note: Random Forests don't provide uncertainty estimates,
        # so Expected Improvement (EI) may be less effective.
        # Consider using acquisition='y' for pure exploitation.
        ```

        ```{python}
        import numpy as np
        from spotoptim import SpotOptim
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, RationalQuadratic, ConstantKernel, RBF

        def objective(X):
            return np.sum(X**2, axis=1)

        # Example 9: Comparing different kernels for Gaussian Process
        # Matern kernel with nu=1.5 (once differentiable)
        kernel_matern15 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5)
        gp_matern15 = GaussianProcessRegressor(kernel=kernel_matern15, normalize_y=True)

        # Matern kernel with nu=2.5 (twice differentiable, DEFAULT)
        kernel_matern25 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        gp_matern25 = GaussianProcessRegressor(kernel=kernel_matern25, normalize_y=True)

        # RBF kernel (infinitely differentiable, smooth)
        kernel_rbf = ConstantKernel(1.0) * RBF(length_scale=1.0)
        gp_rbf = GaussianProcessRegressor(kernel=kernel_rbf, normalize_y=True)

        # Rational Quadratic kernel (mixture of RBF kernels)
        kernel_rq = ConstantKernel(1.0) * RationalQuadratic(length_scale=1.0, alpha=1.0)
        gp_rq = GaussianProcessRegressor(kernel=kernel_rq, normalize_y=True)

        # Use any of these as surrogate
        optimizer_rbf = SpotOptim(fun=objective, bounds=[(-5, 5), (-5, 5)],
                                  surrogate=gp_rbf, max_iter=10, n_initial=5)
        result = optimizer_rbf.optimize()
        ```
    """

    # ====================
    # Core
    # ====================

    def __init__(
        self,
        fun: Callable,
        bounds: Optional[list] = None,
        max_iter: int = 20,
        n_initial: int = 10,
        surrogate: Optional[object] = None,
        acquisition: str = "y",
        var_type: Optional[list] = None,
        var_name: Optional[list] = None,
        var_trans: Optional[list] = None,
        tolerance_x: Optional[float] = None,
        max_time: float = np.inf,
        repeats_initial: int = 1,
        repeats_surrogate: int = 1,
        ocba_delta: int = 0,
        tensorboard_log: bool = False,
        tensorboard_path: Optional[str] = None,
        tensorboard_clean: bool = False,
        fun_mo2so: Optional[Callable] = None,
        seed: Optional[int] = None,
        verbose: bool = False,
        warnings_filter: Literal["default", "error", "ignore"] = "ignore",
        n_infill_points: int = 1,
        max_surrogate_points: Optional[Union[int, List[int]]] = None,
        selection_method: str = "distant",
        acquisition_failure_strategy: str = "random",
        penalty: bool = False,
        penalty_val: Optional[float] = None,
        acquisition_fun_return_size: int = 3,
        acquisition_optimizer: Union[str, Callable] = "differential_evolution",
        restart_after_n: int = 100,
        restart_inject_best: bool = True,
        x0: Optional[np.ndarray] = None,
        de_x0_prob: float = 0.1,
        tricands_fringe: bool = False,
        prob_de_tricands: float = 0.8,
        window_size: Optional[int] = None,
        min_tol_metric: str = "chebyshev",
        prob_surrogate: Optional[List[float]] = None,
        n_jobs: int = 1,
        eval_batch_size: int = 1,
        acquisition_optimizer_kwargs: Optional[Dict[str, Any]] = None,
        args: Tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        warnings.filterwarnings(warnings_filter)

        self.eps = np.sqrt(np.spacing(1))

        if tolerance_x is None:
            tolerance_x = self.eps

        # Infer parameters from objective function if not provided
        if bounds is None:
            bounds = getattr(fun, "bounds", None)
            if bounds is None:
                raise ValueError(
                    "Bounds must be provided either as an argument or via the objective function."
                )

        if var_type is None:
            var_type = getattr(fun, "var_type", None)

        if var_name is None:
            var_name = getattr(fun, "var_name", None)

        if var_trans is None:
            var_trans = getattr(fun, "var_trans", None)

        # Validate parameters
        if max_iter < n_initial:
            raise ValueError(
                f"max_iter ({max_iter}) must be >= n_initial ({n_initial}). "
                f"max_iter represents the total function evaluation budget including initial design."
            )

        # Resolve n_jobs: -1 means "use all available CPU cores" (scikit-learn convention).
        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1
        elif n_jobs == 0 or n_jobs < -1:
            raise ValueError(
                f"n_jobs must be a positive integer or -1 (all CPU cores), got {n_jobs}."
            )

        # Validate eval_batch_size.
        if eval_batch_size < 1:
            raise ValueError(f"eval_batch_size must be >= 1, got {eval_batch_size}.")

        if acquisition_optimizer_kwargs is None:
            acquisition_optimizer_kwargs = {"maxiter": 10000, "gtol": 1e-9}

        # Initialize Configuration
        self.config = SpotOptimConfig(
            bounds=bounds,
            max_iter=max_iter,
            n_initial=n_initial,
            surrogate=surrogate,
            acquisition=acquisition.lower(),
            var_type=var_type,
            var_name=var_name,
            var_trans=var_trans,
            tolerance_x=tolerance_x,
            max_time=max_time,
            repeats_initial=repeats_initial,
            repeats_surrogate=repeats_surrogate,
            ocba_delta=ocba_delta,
            tensorboard_log=tensorboard_log,
            tensorboard_path=tensorboard_path,
            tensorboard_clean=tensorboard_clean,
            fun_mo2so=fun_mo2so,
            seed=seed,
            verbose=verbose,
            warnings_filter=warnings_filter,
            n_infill_points=n_infill_points,
            max_surrogate_points=max_surrogate_points,
            selection_method=selection_method,
            acquisition_failure_strategy=acquisition_failure_strategy,
            penalty=penalty,
            penalty_val=penalty_val,
            acquisition_fun_return_size=acquisition_fun_return_size,
            acquisition_optimizer=acquisition_optimizer,
            restart_after_n=restart_after_n,
            restart_inject_best=restart_inject_best,
            x0=x0,
            de_x0_prob=de_x0_prob,
            tricands_fringe=tricands_fringe,
            prob_de_tricands=prob_de_tricands,
            window_size=window_size,
            min_tol_metric=min_tol_metric,
            prob_surrogate=prob_surrogate,
            n_jobs=n_jobs,
            eval_batch_size=eval_batch_size,
            acquisition_optimizer_kwargs=acquisition_optimizer_kwargs,
            args=args,
            kwargs=kwargs,
        )

        # Initialize State
        self.state = SpotOptimState()

        # Other attributes
        self.fun = fun
        # The fun (objective function) object defines objective_names as a property (e.g., in torch_objective.py).
        # SpotOptim.__init__ copies it onto self so downstream code can access it via optimizer.objective_names.
        # The visualization module reads it from the optimizer to label plots.
        self.objective_names = getattr(
            fun, "objective_names", getattr(fun, "metrics", None)
        )

        # Initialize persistent RNG
        self.rng = np.random.RandomState(self.seed)
        self.set_seed()

        # Process bounds and factor variables
        # The _factor_maps structure is essentially a per-dimension integer-to-string lookup table —
        # the optimizer works entirely with integers internally, and _factor_maps is what
        # translates back to meaningful labels for output and visualization, e.g.:
        # self._factor_maps = {
        #     1: {0: "low",   1: "medium", 2: "high"},
        #     2: {0: "red",   1: "green",  2: "blue"},
        # }
        self._factor_maps = {}  # Maps dimension index to {int: str} mapping

        self._original_bounds = self.bounds.copy()  # Store original bounds
        self.process_factor_bounds()  # Maps factor bounds to integer indices (updates config.bounds)

        # Derived attribute dimension n_dim
        self.n_dim = len(self.bounds)

        # Default variable types
        if self.var_type is None:
            self.var_type = self.detect_var_type()

        # Modify bounds based on var_type
        self.modify_bounds_based_on_var_type()

        # Convert the bounds to numpy arrays, e.g.,
        # self.bounds = [(0, 1), (0, 1)] -> self.lower = [0, 0], self.upper = [1, 1]
        self.lower = np.array([b[0] for b in self.bounds])
        self.upper = np.array([b[1] for b in self.bounds])

        # Default variable names
        if self.var_name is None:
            self.var_name = [f"x{i}" for i in range(self.n_dim)]

        # Handle default variable transformations. No transformations are performed here,
        # only None and id transformations are set correctly.
        self.handle_default_var_trans()

        # Apply transformations to bounds (internal representation)
        self._original_lower = self.lower.copy()
        self._original_upper = self.upper.copy()
        self.transform_bounds()

        # Dimension reduction: backup original bounds and identify fixed dimensions
        self.setup_dimension_reduction()

        # Validate and process starting point if provided,
        # it must be in natural scale (full dimensions) and will be returned in internal scale (reduced dimensions)
        if self.x0 is not None:
            self.x0 = self.validate_x0(self.x0)

        # Initialize surrogate model(s)
        self.init_surrogate()

        # Design generator (from scipy.stats.qmc)
        self.lhs_sampler = LatinHypercube(d=self.n_dim, rng=self.seed)

        # Logic for window_size default based on restart_after_n
        if self.window_size is None:
            if self.restart_after_n is not None:
                self.window_size = self.restart_after_n
            else:
                self.window_size = 100

        # Clean old TensorBoard logs if requested
        self._clean_tensorboard_logs()

        # Initialize TensorBoard writer
        self._init_tensorboard_writer()

    def __getattr__(self, name):
        """Proxy attribute access to config and state."""
        try:
            config = super().__getattribute__("config")
            if hasattr(config, name):
                return getattr(config, name)
        except AttributeError:
            pass

        try:
            state = super().__getattribute__("state")
            if hasattr(state, name):
                return getattr(state, name)
        except AttributeError:
            pass

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __setattr__(self, name, value):
        """Proxy attribute assignment to config and state."""
        if name in ("config", "state"):
            super().__setattr__(name, value)
            return

        try:
            config = super().__getattribute__("config")
            if hasattr(config, name):
                setattr(config, name, value)
                return
        except AttributeError:
            pass

        try:
            state = super().__getattribute__("state")
            if hasattr(state, name):
                setattr(state, name, value)
                return
        except AttributeError:
            pass

        super().__setattr__(name, value)

    def __dir__(self):
        """Include config and state attributes in dir()."""
        d = set(super().__dir__())
        try:
            config = super().__getattribute__("config")
            d.update(dir(config))
        except AttributeError:
            pass

        try:
            state = super().__getattribute__("state")
            # Filter internal methods/fields from dir if desired, but good for now
            d.update(dir(state))
        except AttributeError:
            pass

        return list(d)

    def set_seed(self) -> None:
        """Set global random seeds for reproducibility.
        Sets seeds for:
            * random
            * numpy.random
            * torch (cpu and cuda)
        Only performs actions if self.seed is not None.

        Returns:
            None

        Examples:
           ```{python}
           from spotoptim import SpotOptim
           import numpy as np
           spot = SpotOptim(fun=lambda x: x, bounds=[(0, 1)], seed=42)
           spot.set_seed()
           np.random.rand()  # Should be deterministic
           ```
        """
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)

    # ====================
    # TASK_VARS:
    # * detect_var_type
    # * modify_bounds_based_on_var_type
    # * repair_non_numeric
    # * handle_default_var_trans
    # * process_factor_bounds
    # ====================

    def detect_var_type(self) -> list:
        """Auto-detect variable types based on factor mappings.

        Returns:
            list: List of variable types ('factor' or 'float') for each dimension.
                  Dimensions with factor mappings are assigned 'factor', others 'float'.

        Examples:
            ```{python}
            from spotoptim import SpotOptim

            # Define a simple objective mapping names to values for demonstration
            def objective(X):
                # X has shape (n_samples, n_dimensions)
                return X[:, 0] + X[:, 1]

            # The first dimension has factor levels ('red', 'green', 'blue')
            # The second dimension is continuous bounds (0, 10)
            spot = SpotOptim(fun=objective, bounds=[('red', 'green', 'blue'), (0, 10)])
            print(spot.detect_var_type())
            ```
        """
        return _vars.detect_var_type(self)

    def modify_bounds_based_on_var_type(self) -> None:
        """Modify bounds based on variable types.
        Adjusts bounds for each dimension according to its var_type:
            * 'int': Ensures bounds are integers (ceiling for lower, floor for upper)
            * 'factor': Bounds already set to (0, n_levels-1) by process_factor_bounds
            * 'float': Explicitly converts bounds to float

        Returns:
            None

        Raises:
            ValueError: If an unsupported var_type is encountered.

        Examples:
            ```{python}
            from spotoptim import SpotOptim
            spot = SpotOptim(fun=lambda x: x, bounds=[(0.5, 10.5)], var_type=['int'])
            print(spot.bounds)
            ```

            ```{python}
            from spotoptim import SpotOptim
            spot = SpotOptim(fun=lambda x: x, bounds=[(0, 10)], var_type=['float'])
            print(spot.bounds)
            ```
        """
        _vars.modify_bounds_based_on_var_type(self)

    def repair_non_numeric(self, X: np.ndarray, var_type: List[str]) -> np.ndarray:
        """Round non-numeric values to integers based on variable type.
        This method applies rounding to variables that are not continuous:
            * 'float': No rounding (continuous values)
            * 'int': Rounded to integers
            * 'factor': Rounded to integers (representing categorical values)

        Args:
            X (ndarray): X array with values to potentially round.
            var_type (list of str): List with type information for each dimension.

        Returns:
            ndarray: X array with non-continuous values rounded to integers.

        Examples:
            ```{python}
            import numpy as np
            from spotoptim import SpotOptim
            opt = SpotOptim(fun=lambda X: np.sum(X**2, axis=1),
                             bounds=[(-5, 5), (-5, 5)],
                             var_type=['int', 'float'])
            X = np.array([[1.2, 2.5], [3.7, 4.1], [5.9, 6.8]])
            X_repaired = opt.repair_non_numeric(X, opt.var_type)
            print(X_repaired)
            ```
        """
        return _vars.repair_non_numeric(X, var_type)

    def handle_default_var_trans(self) -> None:
        """Handle default variable transformations. Does not perform any transformations,
        only sets `var_trans` to a list of `None` values if not specified, or normalizes
        transformation names by converting `id`, `None`, or `None` to `None`.
        Also validates that `var_trans` length matches the number of dimensions.

        Returns:
            None

        Raises:
            ValueError: If var_trans length doesn't match n_dim.

        Examples:
            ```{python}
            from spotoptim import SpotOptim
            # Default behavior - all None
            spot = SpotOptim(fun=lambda x: x, bounds=[(0, 10), (0, 10)])
            print(f"spot.var_trans (should be [None, None]): {spot.var_trans}")
            ```
            ```{python}
            from spotoptim import SpotOptim
            # Normalize transformation names
            spot = SpotOptim(fun=lambda x: x, bounds=[(1, 10), (1, 100)],
                             var_trans=['log10', 'id'])
            print(f"spot.var_trans (should be ['log10', 'None']): {spot.var_trans}")
            ```
        """
        _vars.handle_default_var_trans(self)

    def process_factor_bounds(self) -> None:
        """Process `bounds` to handle factor variables.
        For dimensions with tuple bounds (factor variables), creates internal
        integer mappings and replaces bounds with (0, n_levels-1).
        Stores mappings in `self._factor_maps`: {dim_idx: {int_val: str_val}}

        Returns:
            None

        Raises:
            ValueError: If bounds are invalidly formatted.

        Examples:
            ```{python}
            from spotoptim import SpotOptim
            spot = SpotOptim(fun=lambda x: x, bounds=[('red', 'green', 'blue'), (0, 10)])
            spot.process_factor_bounds()
            print(f"spot.bounds (should be [(0, 2), (0, 10)]): {spot.bounds}")
            ```
        """
        _vars.process_factor_bounds(self)

    # ====================
    # TASK_SAVE_LOAD:
    # * get_pickle_safe_optimizer
    # * reinitialize_components
    # ====================

    def get_pickle_safe_optimizer(
        self, unpickleables: str = "file_io", verbosity: int = 0
    ) -> "SpotOptim":
        """Create a pickle-safe copy of the optimizer."""
        return _serial.get_pickle_safe_optimizer(self, unpickleables, verbosity)

    def reinitialize_components(self) -> None:
        """Reinitialize components that were excluded during pickling."""
        _serial.reinitialize_components(self)

    # ====================
    # TASK_DIM:
    # * setup_dimension_reduction()
    # * to_red_dim()
    # * to_all_dim()
    # ====================

    def setup_dimension_reduction(self) -> None:
        """Set up dimension reduction by identifying fixed dimensions.
        Identifies dimensions where lower and upper bounds are equal in Transformed Space.
        Reduces `self.bounds`, `self.lower`, `self.upper`, etc., to the Mapped Space
        (active variables only).
        The resulting `self.bounds` defines the Transformed and Mapped Space used
        for optimization.
        This method identifies variables that are fixed (constant) and excludes them
        from the optimization process. It stores:
            * Original bounds and metadata in `all_*` attributes
            * Boolean mask of fixed dimensions in `ident`
            * Reduced bounds, types, and names for optimization
            * `red_dim` flag indicating if reduction occurred

        Returns:
            None

        Examples:
            ```{python}
            from spotoptim import SpotOptim
            spot = SpotOptim(fun=lambda x: x, bounds=[(1, 10), (5, 5), (0, 1)])
            print("Original lower bounds:", spot.all_lower)
            print("Original upper bounds:", spot.all_upper)
            print("Fixed dimensions mask:", spot.ident)
            print("Reduced lower bounds:", spot.lower)
            print("Reduced upper bounds:", spot.upper)
            print("Reduced variable names:", spot.var_name)
            print("Is dimension reduction active?", spot.red_dim)
            ```
        """
        _dimred.setup_dimension_reduction(self)

    def to_red_dim(self, X_full: np.ndarray) -> np.ndarray:
        """Reduce full-dimensional points to optimization space.
        This method removes fixed dimensions from full-dimensional points,
        extracting only the varying dimensions used in optimization.

        Args:
            X_full (ndarray): Points in full space, shape (n_samples, n_original_dims).

        Returns:
            ndarray: Points in reduced space, shape (n_samples, n_reduced_dims).

        Examples:
            ```{python}
            import numpy as np
            from spotoptim import SpotOptim
            def sphere(X):
                X = np.atleast_2d(X)
                return np.sum(X**2, axis=1)
            # Create problem with one fixed dimension
            opt = SpotOptim(
                fun=sphere,
                bounds=[(-5, 5), (2, 2), (-5, 5)],  # x1 is fixed at 2
                max_iter=10,
                n_initial=3
            )
            X_full = np.array([[1.0, 2.0, 3.0], [4.0, 2.0, 5.0]])
            X_red = opt.to_red_dim(X_full)
            print(X_red.shape)
            print(np.array_equal(X_red, np.array([[1.0, 3.0], [4.0, 5.0]])))
            ```
        """
        return _dimred.to_red_dim(self, X_full)

    def to_all_dim(self, X_red: np.ndarray) -> np.ndarray:
        """Expand reduced-dimensional points to full-dimensional representation.
        This method restores points from the reduced optimization space to the
        full-dimensional space by inserting fixed values for constant dimensions.

        Args:
            X_red (ndarray): Points in reduced space, shape (n_samples, n_reduced_dims).

        Returns:
            ndarray: Points in full space, shape (n_samples, n_original_dims).

        Examples:
            ```{python}
            import numpy as np
            from spotoptim import SpotOptim
            def sphere(X):
                X = np.atleast_2d(X)
                return np.sum(X**2, axis=1)
            # Create problem with one fixed dimension
            opt = SpotOptim(
                fun=sphere,
                bounds=[(-5, 5), (2, 2), (-5, 5)],  # x1 is fixed at 2
                max_iter=10,
                n_initial=3
            )
            X_red = np.array([[1.0, 3.0], [2.0, 4.0]])  # Only x0 and x2
            X_full = opt.to_all_dim(X_red)
            print(X_full.shape)
            print(X_full[:, 1])
            ```
        """
        return _dimred.to_all_dim(self, X_red)

    # ====================
    # TASK_TRANSFORM:
    # * transform_value()
    # * inverse_transform_value()
    # * transform_X()
    # * inverse_transform_X()
    # * transform_bounds()
    # * map_to_factor_values()
    # ====================

    def transform_value(self, x: float, trans: Optional[str]) -> float:
        """Apply transformation to a single float value.

        Args:
            x: Value to transform
            trans: Transformation name. Can be one of `id`, `log10`, `log`, `ln`, `sqrt`,
                   `exp`, `square`, `cube`, `inv`, `reciprocal`, or `None`.
                   Also supports dynamic strings like `log(x)`, `sqrt(x)`, `pow(x, p)`.

        Returns:
            Transformed value

        Raises:
            TypeError: If x is not a float.
            ValueError: If an unknown transformation is specified.

        Notes:
            See also inverse_transform_value.

        Examples:
            ```{python}
            from spotoptim import SpotOptim
            from spotoptim.function import sphere
            spot = SpotOptim(fun=sphere, bounds=[(1, 10)])
            spot.transform_value(10, 'log10')
            spot.transform_value(100, 'log(x)')
            ```
        """
        return _trans.transform_value(x, trans)

    def inverse_transform_value(self, x: float, trans: Optional[str]) -> float:
        """Apply inverse transformation to a single float value.

        Args:
            x: Transformed value
            trans: Transformation name.

        Returns:
            Original value

        Notes:
            See also transform_value.

        Examples:
            ```{python}
            from spotoptim import SpotOptim
            from spotoptim.function import sphere
            spot = SpotOptim(fun=sphere, bounds=[(1, 10)])
            spot.inverse_transform_value(10, 'log10')
            spot.inverse_transform_value(100, 'log(x)')
            ```
        """
        return _trans.inverse_transform_value(x, trans)

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        """Transform parameter array from original (natural) to internal scale.
        Converts from natural space (Original) to transformed space (full dimension).
        Does NOT handle dimension reduction (mapping).

        Args:
            X (ndarray): Array in Natural Space, shape (n_samples, n_features)

        Returns:
            ndarray: Array in Transformed Space (Full Dimension)

        Examples:
            ```{python}
            from spotoptim import SpotOptim
            import numpy as np
            from spotoptim.function import sphere
            spot = SpotOptim(fun=sphere, bounds=[(1, 10)], var_trans=['log10'])
            X_orig = np.array([[1], [10], [100]])
            spot.transform_X(X_orig)
            ```
        """
        return _trans.transform_X(self, X)

    def inverse_transform_X(self, X: np.ndarray) -> np.ndarray:
        """Transform parameter array from internal to original scale.
        Converts from transformed space (full dimension) to natural space (original).
        Does NOT handle dimension expansion (un-mapping).

        Args:
            X (ndarray): Array in Transformed Space, shape (n_samples, n_features)

        Returns:
            ndarray: Array in Natural Space

        Examples:
            ```{python}
            from spotoptim import SpotOptim
            from spotoptim.function import sphere
            import numpy as np
            spot = SpotOptim(fun=sphere, bounds=[(1, 10)], var_trans=['log10'])
            X_trans = np.array([[0], [1], [2]])
            spot.inverse_transform_X(X_trans)
            ```
        """
        return _trans.inverse_transform_X(self, X)

    def transform_bounds(self) -> None:
        """Transform bounds from original to internal scale.
        Updates `self.bounds` (and `self.lower`, `self.upper`) from Natural Space
        to Transformed Space. Calls `transform_value` for each bound and converts
        numpy types to Python native types (`int` or `float` based on `var_type`).
        Handles also reversed bounds, e.g., as an effect of `reciprocal` transformation.

        Returns:
            None

        Notes:
            Uses settings in `self.var_trans`. It can be one of `id`, `log10`, `log`, `ln`, `sqrt`,
            `exp`, `square`, `cube`, `inv`, `reciprocal`, or `None`. Also supports dynamic
            strings like `log(x)`, `sqrt(x)`, `pow(x, p)`.

        Examples:
            ```{python}
            from spotoptim import SpotOptim
            from spotoptim.function import sphere
            import numpy as np
            spot = SpotOptim(fun=sphere, bounds=[(1, 10), (0.1, 100)])
            spot.var_trans = ['log10', 'sqrt']
            spot.transform_bounds()
            print(f"spot.bounds: {spot.bounds}")
            ```
        """
        _trans.transform_bounds(self)

    def map_to_factor_values(self, X: np.ndarray) -> np.ndarray:
        """Map internal integer factor values back to string labels.
        For factor variables, converts integer indices back to original string values.
        Other variable types remain unchanged.

        Args:
            X (ndarray): Design points with integer values for factors,
                shape (n_samples, n_features).

        Returns:
            ndarray: Design points with factor integers replaced by string labels.
                Dtype will be object or string if mixed types are present.

        Examples:
            ```{python}
            from spotoptim import SpotOptim
            from spotoptim.function import sphere
            import numpy as np
            spot = SpotOptim(
                fun=sphere,
                bounds=[('red', 'blue'), (0, 10)]
            )
            spot.process_factor_bounds()
            X_int = np.array([[0, 5.0], [1, 8.0]])
            X_str = spot.map_to_factor_values(X_int)
            print(X_str[0])
            ```
        """
        return _vars.map_to_factor_values(self, X)

    # ====================
    # TASK_INIT_DESIGN:
    # * get_initial_design()
    # * generate_initial_design()
    # * curate_initial_design()
    # * rm_initial_design_NA_values()
    # * validate_x0()
    # * check_size_initial_design()
    # * get_best_xy_initial_design()
    # * init_surrogate()
    # ====================

    def init_surrogate(self) -> None:
        """Initialize or configure the surrogate model for optimization. Handles three surrogate configurations:
            * List of surrogates: sets up multi-surrogate selection with probability weights and per-surrogate `max_surrogate_points`.
            * None (default): creates a `GaussianProcessRegressor` with a
              `ConstantKernel * Matern(nu=2.5)` kernel, 100 optimizer restarts,
              and `normalize_y=True`.
            * User-provided surrogate: accepted as-is; internal bookkeeping
              attributes (`_max_surrogate_points_list`,
              `_active_max_surrogate_points`) are still initialised.
        After this method returns the following attributes are set:
            * `self.surrogate` — the active surrogate model.
            * `self._surrogates_list` — `list | None`.
            * `self._prob_surrogate` — normalised selection probabilities or `None`.
            * `self._max_surrogate_points_list` — per-surrogate point caps or `None`.
            * `self._active_max_surrogate_points` — active cap.

        Raises:
            ValueError: If the surrogate list is empty.
            ValueError: If 'prob_surrogate' length does not match the surrogate list length.
            ValueError: If 'max_surrogate_points' list length does not match the surrogate list length.

        Returns:
            None

        Examples:
            ```{python}
            import numpy as np
            from spotoptim import SpotOptim
            # Default surrogate (GaussianProcessRegressor)
            opt = SpotOptim(
                fun=lambda X: np.sum(X**2, axis=1),
                bounds=[(-5, 5), (-5, 5)],
                n_initial=5,
            )
            print(type(opt.surrogate).__name__)
            ```

            ```{python}
            import numpy as np
            from spotoptim import SpotOptim
            from sklearn.ensemble import RandomForestRegressor
            # User-provided surrogate
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            opt = SpotOptim(
                fun=lambda X: np.sum(X**2, axis=1),
                bounds=[(-5, 5), (-5, 5)],
                n_initial=5,
                surrogate=rf,
            )
            print(type(opt.surrogate).__name__)
            ```

            ```{python}
            import numpy as np
            from spotoptim import SpotOptim
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.gaussian_process import GaussianProcessRegressor
            # List of surrogates with selection probabilities
            surrogates = [GaussianProcessRegressor(), RandomForestRegressor()]
            opt = SpotOptim(
                fun=lambda X: np.sum(X**2, axis=1),
                bounds=[(-5, 5), (-5, 5)],
                n_initial=5,
                surrogate=surrogates,
                prob_surrogate=[0.7, 0.3],
            )
            print(opt._prob_surrogate)
            print([type(s).__name__ for s in opt._surrogates_list])
            ```
        """
        self._surrogates_list = None
        self._prob_surrogate = None

        if isinstance(self.surrogate, list):
            self._surrogates_list = self.surrogate
            if not self._surrogates_list:
                raise ValueError("Surrogate list cannot be empty.")

            # Handle probabilities
            if self.config.prob_surrogate is None:
                # Uniform probability
                n = len(self._surrogates_list)
                self._prob_surrogate = [1.0 / n] * n
            else:
                probs = self.config.prob_surrogate
                if len(probs) != len(self._surrogates_list):
                    raise ValueError(
                        f"Length of prob_surrogate ({len(probs)}) must match "
                        f"number of surrogates ({len(self._surrogates_list)})."
                    )
                # Normalize probabilities
                total = sum(probs)
                if not np.isclose(total, 1.0) and total > 0:
                    self._prob_surrogate = [p / total for p in probs]
                else:
                    self._prob_surrogate = probs

            # Handle max_surrogate_points list
            self._max_surrogate_points_list = None
            if isinstance(self.config.max_surrogate_points, list):
                if len(self.config.max_surrogate_points) != len(self._surrogates_list):
                    raise ValueError(
                        f"Length of max_surrogate_points ({len(self.config.max_surrogate_points)}) "
                        f"must match number of surrogates ({len(self._surrogates_list)})."
                    )
                self._max_surrogate_points_list = self.config.max_surrogate_points
            else:
                # If int or None, broadcast to list for easier indexing
                self._max_surrogate_points_list = [
                    self.config.max_surrogate_points
                ] * len(self._surrogates_list)

            # Set initial surrogate and max points
            self.surrogate = self._surrogates_list[0]
            self._active_max_surrogate_points = self._max_surrogate_points_list[0]

        elif self.surrogate is None:
            # Default single surrogate case
            self._max_surrogate_points_list = None
            self._active_max_surrogate_points = self.config.max_surrogate_points

            kernel = ConstantKernel(1.0, (1e-2, 1e12)) * Matern(
                length_scale=1.0, length_scale_bounds=(1e-4, 1e2), nu=2.5
            )

            # Determine optimizer for GPR
            optimizer = "fmin_l_bfgs_b"  # Default used by sklearn
            if self.config.acquisition_optimizer_kwargs is not None:
                optimizer = partial(
                    gpr_minimize_wrapper, **self.config.acquisition_optimizer_kwargs
                )

            self.surrogate = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=100,
                normalize_y=True,
                random_state=self.seed,
                optimizer=optimizer,
            )

    def get_initial_design(self, X0: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate or process initial design points. Ensures that design points are in
        internal (transformed and reduced) scale.
        Calls `generate_initial_design()` if `X0` is None, otherwise processes user-provided `X0`.
        Handles three scenarios:
            * `X0` is None: Generate space-filling design using LHS
            * `X0` is None but starting point(s) `x0` is provided: Generate LHS and include `x0` as first point(s)
            * `X0` is provided: Transform and prepare user-provided initial design

        Args:
            X0 (ndarray, optional): User-provided initial design points in original scale,
                shape (n_initial, n_features). If None, generates space-filling design.
                Defaults to None.

        Returns:
            ndarray: Initial design points in internal (transformed and reduced) scale,
                shape (n_initial, n_features_reduced).

        Examples:
            ```{python}
            import numpy as np
            from spotoptim import SpotOptim
            from spotoptim.function import sphere
            from spotoptim.plot.visualization import plot_design_points
            opt = SpotOptim(
                fun=sphere,
                bounds=[(-5, 5), (-5, 5)],
                n_initial=10
            )
            # Generate default LHS design
            X0 = opt.get_initial_design()
            print(X0.shape)
            plot_design_points(X0)
            ```

            ```{python}
            import numpy as np
            from spotoptim import SpotOptim
            from spotoptim.function import sphere
            from spotoptim.plot.visualization import plot_design_points
            opt = SpotOptim(
                fun=sphere,
                bounds=[(-5, 5), (-5, 5)],
                n_initial=10,
                x0=np.array([0, 0])  # Starting point to include in initial design
            )
            X0 = opt.get_initial_design()
            print(X0.shape)
            plot_design_points(X0)
            ```
        """
        # Generate or use provided initial design
        if X0 is None:
            X0 = self.generate_initial_design()

            # If starting point(s) x0 was provided, include it/them in initial design
            if self.x0 is not None:
                # x0 is already validated and in internal scale
                # Check if x0 is 1D or 2D
                if self.x0.ndim == 1:
                    x0_points = self.x0.reshape(1, -1)
                else:
                    x0_points = self.x0

                n_x0 = x0_points.shape[0]

                # If we have more x0 points than n_initial, use all x0 points
                if n_x0 >= self.n_initial:
                    X0 = x0_points
                    if self.verbose:
                        print(f"Using provided x0 points ({n_x0}) as initial design.")
                else:
                    # Replace the first n_x0 points of LHS with x0 points
                    X0 = np.vstack([x0_points, X0[:-n_x0]])
                    if self.verbose:
                        print(
                            f"Including {n_x0} starting points from x0 in initial design."
                        )
        else:
            X0 = np.atleast_2d(X0)
            # If user provided X0, it's in original scale - transform it
            X0 = self.transform_X(X0)
            # If X0 is in full dimensions and we have dimension reduction, reduce it
            if self.red_dim and X0.shape[1] == len(self.ident):
                X0 = self.to_red_dim(X0)
            X0 = self.repair_non_numeric(X0, self.var_type)

        return X0

    def generate_initial_design(self) -> np.ndarray:
        """Generate initial space-filling design using Latin Hypercube Sampling.
        Used in the optimize() method to create the initial set of design points.

        Returns:
            ndarray: Initial design points, shape (n_initial, n_features). Points are in the intervals defined by `self.bounds`.

        Examples:
            ```{python}
            from spotoptim import SpotOptim
            from spotoptim.function import sphere
            opt = SpotOptim(fun=sphere,
                            bounds=[(-5, 5), (-5, 5)],
                            n_initial=3,
                            var_type=['float', 'int'],
                            var_trans=['log10', None])
            X0 = opt.generate_initial_design()
            print(X0.shape)
            ```
        """
        # Generate samples in [0, 1]^d
        X0_unit = self.lhs_sampler.random(n=self.n_initial)

        # Scale to [lower, upper]
        X0 = self.lower + X0_unit * (self.upper - self.lower)

        return self.repair_non_numeric(X0, self.var_type)

    def curate_initial_design(self, X0: np.ndarray) -> np.ndarray:
        """Remove duplicates and ensure sufficient unique points in initial design.

        This method handles deduplication that can occur after rounding integer/factor
        variables. If duplicates are found, it generates additional points to reach
        the target n_initial unique points. Also handles repeating points when
        repeats_initial > 1.

        Args:
            X0 (ndarray): Initial design points in internal scale,
                shape (n_samples, n_features).

        Returns:
            ndarray: Curated initial design with duplicates removed and repeated
                if necessary, shape (n_unique * repeats_initial, n_features).

        Examples:
            ```{python}
            import numpy as np
            from spotoptim import SpotOptim
            from spotoptim.function import sphere

            opt = SpotOptim(
                fun=sphere,
                bounds=[(-5, 5), (-5, 5)],
                n_initial=10,
                var_type=['int', 'int']  # Integer variables may cause duplicates
            )
            X0 = opt.get_initial_design()
            X0_curated = opt.curate_initial_design(X0)
            X0_curated.shape[0] == 10  # Should have n_initial unique points
            ```
            ```{python}
            import numpy as np
            from spotoptim import SpotOptim
            from spotoptim.function import sphere
            # With repeats
            opt_repeat = SpotOptim(
                fun=sphere,
                bounds=[(-5, 5), (-5, 5)],
                n_initial=5,
                repeats_initial=3
            )
            X0 = opt_repeat.get_initial_design()
            X0_curated = opt_repeat.curate_initial_design(X0)
            X0_curated.shape[0] == 15  # 5 unique points * 3 repeats
            ```
        """
        # Remove duplicates from initial design (can occur after rounding integers/factors)
        # Keep only unique rows based on rounded values
        X0_unique, unique_indices = np.unique(X0, axis=0, return_index=True)
        if len(X0_unique) < len(X0):
            n_duplicates = len(X0) - len(X0_unique)
            if self.verbose:
                print(
                    f"Removed {n_duplicates} duplicate(s) from initial design after rounding"
                )

            # Generate additional points to reach n_initial unique points
            if len(X0_unique) < self.n_initial:
                n_additional = self.n_initial - len(X0_unique)
                if self.verbose:
                    print(
                        f"Generating {n_additional} additional point(s) to reach n_initial={self.n_initial}"
                    )

                # Generate extra points and deduplicate again
                max_gen_attempts = 10
                for gen_attempt in range(max_gen_attempts):
                    X_extra_unit = self.lhs_sampler.random(
                        n=n_additional * 2
                    )  # Generate extras
                    X_extra = self.lower + X_extra_unit * (self.upper - self.lower)
                    X_extra = self.repair_non_numeric(X_extra, self.var_type)

                    # Combine and get unique
                    X_combined = np.vstack([X0_unique, X_extra])
                    X_combined_unique = np.unique(X_combined, axis=0)

                    if len(X_combined_unique) >= self.n_initial:
                        X0 = X_combined_unique[: self.n_initial]
                        break
                else:
                    # If still not enough unique points, just use what we have
                    X0 = X_combined_unique
                    if self.verbose:
                        print(
                            f"Warning: Could only generate {len(X0)} unique initial points (target was {self.n_initial})"
                        )
            else:
                X0 = X0_unique

        # Repeat initial design points if repeats_initial > 1
        if self.repeats_initial > 1:
            X0 = np.repeat(X0, self.repeats_initial, axis=0)

        return X0

    def validate_x0(self, x0: np.ndarray) -> np.ndarray:
        """Validate and process starting point x0. Called in `__init__` and `optimize`.
        This method checks that x0:
            * Is a numpy array
            * Has the correct number of dimensions
            * Has values within bounds (in original scale)
            * Is properly transformed to internal scale

        Args:
            x0 (array-like): Starting point in original scale

        Returns:
            ndarray: Validated and transformed x0 in internal scale, shape (n_features,)

        Raises:
            ValueError: If x0 is invalid

        Examples:
            ```{python}
            import numpy as np
            from spotoptim import SpotOptim
            from spotoptim.function import sphere
            opt = SpotOptim(
                fun=sphere,
                bounds=[(-5, 5), (5,5), (-10, 10)],
                x0=np.array([1.0, 5.0, 9.0]),
                var_trans=["log10", "id", "sqrt"]
            )
            # x0 is validated during initialization and transformed to internal scale
            print(f"x0 in internal scale: {opt.x0}")
            ```
        """
        # Convert to numpy array
        x0 = np.asarray(x0)

        # Check if x0 is 1D or can be flattened to 1D
        if x0.ndim == 0:
            raise ValueError(
                f"x0 must be a 1D array-like, got scalar value. "
                f"Expected shape: ({len(self.all_lower)},)"
            )
        elif x0.ndim > 2:
            raise ValueError(
                f"x0 must be a 1D array-like, got {x0.ndim}D array. "
                f"Expected shape: ({len(self.all_lower)},)"
            )
        elif x0.ndim == 2:
            pass  # 2D array is allowed now for x0
            if x0.shape[0] == 1:
                x0 = x0.ravel()

        # Check number of dimensions (compare with original full dimensions before reduction)
        # Check number of dimensions
        expected_dim = len(self.all_lower)
        if x0.ndim == 1:
            if len(x0) != expected_dim:
                raise ValueError(
                    f"x0 has {len(x0)} dimensions, but expected {expected_dim} dimensions. "
                    f"Bounds specify {expected_dim} parameters: {self.all_var_name}"
                )
        else:
            if x0.shape[1] != expected_dim:
                raise ValueError(
                    f"x0 has {x0.shape[1]} dimensions, but expected {expected_dim} dimensions. "
                    f"Bounds specify {expected_dim} parameters: {self.all_var_name}"
                )

        # Helper to validate a single point
        def check_point(pt):
            for i, (val, low, high, name) in enumerate(
                zip(pt, self._original_lower, self._original_upper, self.all_var_name)
            ):
                # Ensure val is scalar for comparison (zip iterates elements, but be safe)
                if self.red_dim and self.ident[i]:
                    if not np.isclose(val, low, atol=self.eps):
                        raise ValueError(
                            f"x0 ({name}) = {val} is a fixed dimension and must equal {low}. "
                        )
                else:
                    if not (low <= val <= high):
                        raise ValueError(
                            f"x0 ({name}) = {val} is outside bounds [{low}, {high}]. "
                        )

        if x0.ndim == 1:
            check_point(x0)
            # Apply transformations to x0 (from original to internal scale).
            # IMPORTANT: use all_var_trans (full-dimension list), not self.var_trans
            # (the reduced list).  transform_X iterates its list by index i and
            # applies trans to column i.  When a fixed dimension sits between two
            # free ones, the reduced var_trans list is shorter than the number of
            # columns in the full-dim x0, so transforms land on the wrong columns.
            # Using all_var_trans preserves the correct dim-to-transform mapping.
            x0_2d = x0.reshape(1, -1).astype(float).copy()
            for i, trans in enumerate(self.all_var_trans):
                if trans is not None:
                    x0_2d[0, i] = self.transform_value(x0[i], trans)
            x0_transformed = x0_2d.ravel()
        else:  # 2D case
            for idx, pt in enumerate(x0):
                check_point(pt)
            x0_transformed = x0.astype(float).copy()
            for i, trans in enumerate(self.all_var_trans):
                if trans is not None:
                    x0_transformed[:, i] = [
                        self.transform_value(v, trans) for v in x0[:, i]
                    ]

        # If dimension reduction is active, reduce x0 to non-fixed dimensions
        if self.red_dim:
            x0_transformed = x0_transformed[~self.ident]

        if self.verbose:
            print("Starting point x0 validated and processed successfully.")
            print(f"  Original scale: {x0}")
            print(f"  Internal scale: {x0_transformed}")

        return x0_transformed

    # ====================
    # TASK_FIT:
    # *fit_scheduler()
    # * fit_surrogate()
    # * fit_select_distant_points()
    # * fit_select_best_cluster()
    # * fit_selection_dispatcher()
    # ====================

    def fit_surrogate(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit surrogate model to data.
        Used by fit_scheduler() to fit the surrogate model.
        If the number of points exceeds `self.max_surrogate_points`,
        a subset of points is selected using the selection dispatcher.

        Args:
            X (ndarray): Design points, shape (n_samples, n_features).
            y (ndarray): Function values at X, shape (n_samples,).

        Returns:
            None

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> from sklearn.gaussian_process import GaussianProcessRegressor
            >>> def sphere(X):
            ...     X = np.atleast_2d(X)
            ...     return np.sum(X**2, axis=1)
            >>> opt = SpotOptim(fun=sphere,
            ...                 bounds=[(-5, 5), (-5, 5)],
            ...                 max_surrogate_points=10,
            ...                 surrogate=GaussianProcessRegressor())
            >>> X = np.random.rand(50, 2)
            >>> y = np.random.rand(50)
            >>> opt.fit_surrogate(X, y)
            >>> # Surrogate is now fitted
        """
        X_fit = X
        y_fit = y

        # Select subset if needed
        # Resolve active max points
        max_k = getattr(self, "_active_max_surrogate_points", self.max_surrogate_points)

        if max_k is not None and X.shape[0] > max_k:
            if self.verbose:
                print(
                    f"Selecting subset of {max_k} points "
                    f"from {X.shape[0]} total points for surrogate fitting."
                )
            X_fit, y_fit = self.fit_selection_dispatcher(X, y)

        self.surrogate.fit(X_fit, y_fit)

    def fit_scheduler(self) -> None:
        """Fit surrogate model using appropriate data based on noise handling.
        This method selects the appropriate training data for surrogate fitting:
            * For noisy functions (repeats_surrogate > 1): Uses mean_X and mean_y (aggregated values)
            * For deterministic functions: Uses X_ and y_ (all evaluated points)
        The data is transformed to internal scale before fitting the surrogate.

        Returns:
            None

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> from sklearn.gaussian_process import GaussianProcessRegressor
            >>> # Deterministic function
            >>> def sphere(X):
            ...     X = np.atleast_2d(X)
            ...     return np.sum(X**2, axis=1)
            >>> opt = SpotOptim(
            ...     fun=sphere,
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     surrogate=GaussianProcessRegressor(),
            ...     n_initial=5
            ... )
            >>> # Simulate optimization state
            >>> opt.X_ = np.array([[1, 2], [0, 0], [2, 1]])
            >>> opt.y_ = np.array([5.0, 0.0, 5.0])
            >>> opt.fit_scheduler()
            >>> # Surrogate fitted with X_ and y_
            >>>
            >>> # Noisy function
            >>> def sphere(X):
            ...     X = np.atleast_2d(X)
            ...     return np.sum(X**2, axis=1)
            >>> opt_noise = SpotOptim(
            ...     fun=sphere,
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     surrogate=GaussianProcessRegressor(),
            ...     n_initial=5,
            ...     repeats_initial=3,
            ... )
            >>> # Simulate noisy optimization state
            >>> opt_noise.mean_X = np.array([[1, 2], [0, 0]])
            >>> opt_noise.mean_y = np.array([5.0, 0.0])
            >>> opt_noise.fit_scheduler()
            >>> # Surrogate fitted with mean_X and mean_y
        """
        # Fit surrogate (use mean_y if noise, otherwise y_)
        # Transform X to internal scale for surrogate fitting

        # Handle multi-surrogate selection
        if getattr(self, "_surrogates_list", None) is not None:
            idx = self.rng.choice(len(self._surrogates_list), p=self._prob_surrogate)
            self.surrogate = self._surrogates_list[idx]
            # Update active max surrogate points
            self._active_max_surrogate_points = self._max_surrogate_points_list[idx]

        if (self.repeats_initial > 1) or (self.repeats_surrogate > 1):
            X_for_surrogate = self.transform_X(self.mean_X)
            self.fit_surrogate(X_for_surrogate, self.mean_y)
        else:
            X_for_surrogate = self.transform_X(self.X_)
            self.fit_surrogate(X_for_surrogate, self.y_)

    def fit_select_distant_points(
        self, X: np.ndarray, y: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Selects k points that are distant from each other using K-means clustering.
        This method performs K-means clustering to find k clusters, then selects
        the point closest to each cluster center. This ensures a space-filling
        subset of points for surrogate model training.

        Args:
            X (ndarray): Design points, shape (n_samples, n_features).
            y (ndarray): Function values at X, shape (n_samples,).
            k (int): Number of points to select.

        Returns:
            tuple: A tuple containing:
                * selected_X (ndarray): Selected design points, shape (k, n_features).
                * selected_y (ndarray): Function values at selected points, shape (k,).

        Examples:
            ```{python}
            import numpy as np
            from spotoptim import SpotOptim
            opt = SpotOptim(fun=lambda X: np.sum(X**2, axis=1),
                            bounds=[(-5, 5), (-5, 5)],
                            max_surrogate_points=5)
            X = np.random.rand(100, 2)
            y = np.random.rand(100)
            X_sel, y_sel = opt.fit_select_distant_points(X, y, 5)
            print(X_sel.shape)
            ```
        """
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(X)

        # Find the closest point to each cluster center
        selected_indices = []
        for center in kmeans.cluster_centers_:
            distances = np.linalg.norm(X - center, axis=1)
            closest_idx = np.argmin(distances)
            selected_indices.append(closest_idx)

        selected_indices = np.array(selected_indices)
        return X[selected_indices], y[selected_indices]

    def fit_select_best_cluster(
        self, X: np.ndarray, y: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Selects all points from the cluster with the smallest mean y value.
        This method performs K-means clustering and selects all points from the
        cluster whose center corresponds to the best (smallest) mean objective
        function value.

        Args:
            X (ndarray): Design points, shape (n_samples, n_features).
            y (ndarray): Function values at X, shape (n_samples,).
            k (int): Number of clusters.

        Returns:
            tuple: A tuple containing:
                * selected_X (ndarray): Selected design points from best cluster, shape (m, n_features).
                * selected_y (ndarray): Function values at selected points, shape (m,).

        Examples:
            ```{python}
            import numpy as np
            from spotoptim import SpotOptim
            opt = SpotOptim(fun=lambda X: np.sum(X**2, axis=1),
                            bounds=[(-5, 5), (-5, 5)],
                            max_surrogate_points=5,
                             selection_method='best')
            X = np.random.rand(100, 2)
            y = np.random.rand(100)
            X_sel, y_sel = opt.fit_select_best_cluster(X, y, 5)
            print(f"X_sel.shape: {X_sel.shape}")
            print(f"y_sel.shape: {y_sel.shape}")
            ```
        """
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(X)
        labels = kmeans.labels_

        # Compute mean y for each cluster
        cluster_means = []
        for cluster_idx in range(k):
            cluster_y = y[labels == cluster_idx]
            if len(cluster_y) == 0:
                cluster_means.append(np.inf)
            else:
                cluster_means.append(np.mean(cluster_y))

        # Find cluster with smallest mean y
        best_cluster = np.argmin(cluster_means)

        # Select all points from the best cluster
        mask = labels == best_cluster
        return X[mask], y[mask]

    def fit_selection_dispatcher(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Dispatcher for selection methods.
        Depending on the value of `self.selection_method`, this method calls
        the appropriate selection function to choose a subset of points for
        surrogate model training when the total number of points exceeds
        `self.max_surrogate_points`.

        Args:
            X (ndarray): Design points, shape (n_samples, n_features).
            y (ndarray): Function values at X, shape (n_samples,).

        Returns:
            tuple: A tuple containing:
                * selected_X (ndarray): Selected design points.
                * selected_y (ndarray): Function values at selected points.

        Examples:
            ```{python}
            import numpy as np
            from spotoptim import SpotOptim
            opt = SpotOptim(fun=lambda X: np.sum(X**2, axis=1),
                            bounds=[(-5, 5), (-5, 5)],
                            max_surrogate_points=5)
            X = np.random.rand(100, 2)
            y = np.random.rand(100)
            X_sel, y_sel = opt.fit_selection_dispatcher(X, y)
            print(X_sel.shape[0] <= 5)
            ```
        """
        # Resolve active max points
        max_k = getattr(self, "_active_max_surrogate_points", self.max_surrogate_points)

        if max_k is None:
            return X, y

        if self.selection_method == "distant":
            return self.fit_select_distant_points(X=X, y=y, k=max_k)
        elif self.selection_method == "best":
            return self.fit_select_best_cluster(X=X, y=y, k=max_k)
        else:
            # If no valid selection method, return all points
            return X, y

    # ====================
    # TASK_PREDICT:
    # * _predict_with_uncertainty()
    # * _acquisition_function()
    # ====================

    def _predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty estimates, handling surrogates without return_std.
        Used in the _acquisition_function() method and in the plot_surrogate() method.

        Args:
            X: Input points, shape (n_samples, n_features)

        Returns:
            Tuple of (predictions, std_deviations). If surrogate doesn't support
            return_std, returns predictions with zeros for std.

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> from sklearn.gaussian_process import GaussianProcessRegressor
            >>> def sphere(X):
            ...     X = np.atleast_2d(X)
            ...     return np.sum(X**2, axis=1)
            >>> opt = SpotOptim(
            ...     fun=sphere,
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     surrogate=GaussianProcessRegressor()
            ... )
            >>> X_train = np.array([[0, 0], [1, 1], [2, 2]])
            >>> y_train = np.array([0, 2, 8])
            >>> opt.fit_surrogate(X_train, y_train)
            >>> X_test = np.array([[1.5, 1.5], [3.0, 3.0]])
            >>> preds, stds = opt._predict_with_uncertainty(X_test)
            >>> print("Predictions:", preds)
            Predictions: [4.5 9. ]
            >>> print("Standard deviations:", stds)
            Standard deviations: [some values or zeros depending on surrogate]
        """
        try:
            # Try to get uncertainty estimates
            y_pred, y_std = self.surrogate.predict(X, return_std=True)
            return y_pred, y_std
        except (TypeError, AttributeError):
            # Surrogate doesn't support return_std (e.g., Random Forest, XGBoost)
            y_pred = self.surrogate.predict(X)
            y_std = np.zeros_like(y_pred)
            return y_pred, y_std

    def _acquisition_function(self, x: np.ndarray) -> np.ndarray:
        """Compute acquisition function value(s), supporting vectorised evaluation.
        Used in the suggest_next_infill_point() method.
        This implements "Infill Criteria" as described in Forrester et al. (2008),
        Section 3 "Exploring and Exploiting".
        Supports two calling conventions so it works both as a plain callable and
        as the ``func`` argument to ``differential_evolution(vectorized=True)``:
            * Single-point (x.ndim == 1, shape ``(n_dim,)``):
            returns a scalar float.
            * Vectorised batch (x.ndim == 2, shape ``(n_dim, n_population)``):
            scipy passes the whole DE population at once; each *column* is one
            candidate.  Returns an array of shape ``(n_population,)``.
            A single batched surrogate call replaces ``n_population`` serial calls,
            cutting per-generation overhead by ~``popsize``× (typically 30–150×).

        Args:
            x (ndarray): Single point ``(n_dim,)`` or population matrix
                ``(n_dim, n_population)``.

        Returns:
            float | ndarray: Scalar for single-point calls; array of shape
            ``(n_population,)`` for vectorised calls (values to be minimised).

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> from sklearn.gaussian_process import GaussianProcessRegressor
            >>> opt = SpotOptim(
            ...     fun=lambda X: np.sum(X**2, axis=1),
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     surrogate=GaussianProcessRegressor(),
            ...     acquisition='ei'
            ... )
            >>> X_train = np.array([[0, 0], [1, 1], [2, 2]])
            >>> y_train = np.array([0, 2, 8])
            >>> opt.fit_surrogate(X_train, y_train)
            >>> x_eval = np.array([1.5, 1.5])
            >>> acq_value = opt._acquisition_function(x_eval)
            >>> print("Acquisition function value:", acq_value)
            Acquisition function value: [some float value]
        """
        # Vectorised call from differential_evolution(vectorized=True):
        #   x is (n_dim, n_population) — columns are candidates.
        # Single-point call: x is (n_dim,).
        if x.ndim == 2:
            X = x.T  # (n_population, n_dim)
            batched = True
        else:
            X = x.reshape(1, -1)  # (1, n_dim)
            batched = False

        if self.acquisition == "y":
            vals = self.surrogate.predict(X)  # (n,)
            return vals if batched else float(vals[0])

        elif self.acquisition in ("ei", "pi"):
            mu, sigma = self._predict_with_uncertainty(X)  # (n,), (n,)
            y_best = np.min(self.y_)

            if self.acquisition == "ei":
                # Vectorised EI — guard against sigma ≈ 0 without branching
                safe_sigma = np.where(sigma < 1e-10, 1.0, sigma)
                improvement = y_best - mu
                Z = improvement / safe_sigma
                acq = np.where(
                    sigma < 1e-10,
                    0.0,
                    improvement * norm.cdf(Z) + sigma * norm.pdf(Z),
                )
            else:  # pi
                safe_sigma = np.where(sigma < 1e-10, 1.0, sigma)
                Z = (y_best - mu) / safe_sigma
                acq = np.where(sigma < 1e-10, 0.0, norm.cdf(Z))

            neg_acq = -acq
            return neg_acq if batched else float(neg_acq[0])

        raise ValueError(f"Unknown acquisition function: {self.acquisition}")

    # ====================
    # TASK_OPTIM:
    # * optimize()
    # * execute_optimization_run()
    # * evaluate_function()
    # * _optimize_acquisition_tricands()
    # * _prepare_de_kwargs()
    # * _optimize_acquisition_de()
    # * _optimize_acquisition_scipy()
    # * _try_optimizer_candidates()
    # * remove_nan()
    # * _handle_acquisition_failure()
    # * _try_fallback_strategy()
    # * get_shape()
    # * optimize_acquisition_func()
    # ====================

    def optimize(self, X0: Optional[np.ndarray] = None) -> OptimizeResult:
        """Run the optimization process. The optimization terminates when either the total function evaluations reach
            `max_iter` (including initial design), or the runtime exceeds max_time minutes. Input/Output spaces are
                * Input `X0`: Expected in Natural Space (original scale, physical units).
                * Output `result.x`: Returned in Natural Space.
                * Output `result.X`: Returned in Natural Space.
                * Internal Optimization: Performed in Transformed and Mapped Space.

        Args:
            X0 (ndarray, optional): Initial design points in Natural Space, shape (n_initial, n_features).
                If None, generates space-filling design. Defaults to None.

        Returns:
            OptimizeResult: Optimization result with fields:
                * x: best point found in Natural Space
                * fun: best function value
                * nfev: number of function evaluations (including initial design)
                * nit: number of sequential optimization iterations (after initial design)
                * success: whether optimization succeeded
                * message: termination message indicating reason for stopping, including statistics (function value, iterations, evaluations)
                * X: all evaluated points in Natural Space
                * y: all function values

        Examples:
            ```{python}
            import numpy as np
            from spotoptim import SpotOptim
            from spotoptim.function import sphere
            opt = SpotOptim(
                fun=sphere,
                bounds=[(-5, 5), (-5, 5)],
                n_initial=5,
                max_iter=10,
                seed=0,
                x0=np.array([0.1, -0.1]),
                verbose=True
            )
            result = opt.optimize()
            print(result.message.splitlines()[0])
            print("Best point:", result.x)
            print("Best value:", result.fun)
            ```
        """
        # Track results across restarts for final aggregation.
        self.restarts_results_ = []
        # Capture start time for timeout enforcement.
        timeout_start = time.time()

        # Initial run state.
        current_X0 = X0
        status = "START"

        while True:
            # Get best result so far if we have results
            best_res = (
                min(self.restarts_results_, key=lambda x: x.fun)
                if self.restarts_results_
                else None
            )

            # Compute injected best value for restarts, then run one optimization cycle.
            # y0_known_val carries the current global best objective so the next
            # run can skip re-evaluating that known point when restart injection is on.
            y0_known_val = (
                best_res.fun
                if (
                    status == "RESTART"
                    and self.restart_inject_best
                    and self.restarts_results_
                )
                else None
            )

            # Calculate remaining budget
            total_evals_so_far = sum(len(r.y) for r in self.restarts_results_)
            remaining_iter = self.max_iter - total_evals_so_far

            # If we don't have enough budget for at least initial design (or some minimal amount), stop
            if remaining_iter < self.n_initial:
                if self.verbose:
                    print("Global budget exhausted. Stopping restarts.")
                break

            # Execute one optimization run using the remaining budget; dispatcher
            # selects sequential vs parallel based on `n_jobs` and returns status/result.
            status, result = self.execute_optimization_run(
                timeout_start,
                current_X0,
                y0_known=y0_known_val,
                max_iter_override=remaining_iter,
            )
            self.restarts_results_.append(result)

            if status == "FINISHED":
                break
            elif status == "RESTART":
                # Prepare for a clean restart: let get_initial_design() regenerate the full design.
                current_X0 = None

                # Find the global best result across completed restarts.
                if self.restarts_results_:
                    best_res = min(self.restarts_results_, key=lambda r: r.fun)

                    if self.restart_inject_best:
                        # Inject the current global best into the next run's initial design.
                        # best_res.x is in natural scale; validate_x0 converts to internal scale
                        # so the injected point can be mixed with LHS samples.
                        self.x0 = self.validate_x0(best_res.x)
                        # Keep current_X0 unset so the initial design is rebuilt around the injected x0.
                        current_X0 = None

                        if self.verbose:
                            print(
                                f"Restart injection: Using best found point so far as starting point (f(x)={best_res.fun:.6f})."
                            )

                if self.seed is not None and self.n_jobs == 1:
                    # In sequential mode we advance the seed between restarts to diversify the LHS.
                    # Parallel mode increments seeds per worker during dispatch.
                    self.seed += 1
                # Continue loop
            else:
                # Should not happen
                break

        # Return best result
        if not self.restarts_results_:
            return result  # Fallback

        # Find best result based on 'fun'
        best_result = min(self.restarts_results_, key=lambda r: r.fun)

        # Merge results from all parallel runs (and sequential runs if any)
        X_all_list = [res.X for res in self.restarts_results_]
        y_all_list = [res.y for res in self.restarts_results_]

        # Concatenate all evaluations
        self.X_ = np.vstack(X_all_list)
        self.y_ = np.concatenate(y_all_list)
        self.counter = len(self.y_)

        # Aggregated iterations (sum of all runs)
        self.n_iter_ = sum(getattr(res, "nit", 0) for res in self.restarts_results_)

        # Update best solution found
        self.best_x_ = best_result.x
        self.best_y_ = best_result.fun

        return best_result

    def execute_optimization_run(
        self,
        timeout_start: float,
        X0: Optional[np.ndarray] = None,
        y0_known: Optional[float] = None,
        max_iter_override: Optional[int] = None,
        shared_best_y=None,  # New arg
        shared_lock=None,  # New arg
    ) -> Tuple[str, OptimizeResult]:
        """Dispatcher for optimization run (Sequential vs Steady-State Parallel).
        Depending on n_jobs, calls optimize_steady_state (n_jobs > 1) or optimize_sequential_run (n_jobs == 1).

        Args:
            timeout_start (float): Start time for timeout.
            X0 (Optional[np.ndarray]): Initial design points in Natural Space, shape (n_initial, n_features).
            y0_known (Optional[float]): Known best value for initial design.
            max_iter_override (Optional[int]): Override for maximum number of iterations.
            shared_best_y (Optional[float]): Shared best value for parallel runs.
            shared_lock (Optional[Lock]): Shared lock for parallel runs.

        Returns:
            Tuple[str, OptimizeResult]: Tuple containing status and optimization result.

        Examples:
            ```{python}
            import time
            import numpy as np
            from spotoptim import SpotOptim
            from spotoptim.function import sphere
            opt = SpotOptim(
                fun=sphere,
                bounds=[(-5, 5), (-5, 5)],
                n_initial=5,
                max_iter=10,
                seed=0,
                n_jobs=1,  # Use sequential optimization for deterministic output
                verbose=True
            )
            status, result = opt.execute_optimization_run(timeout_start=time.time())
            print(status)
            print(result.message.splitlines()[0])
            ```
        """

        # Dispatch to steady-state optimizer if proper parallelization is requested
        if self.n_jobs > 1:
            return self.optimize_steady_state(
                timeout_start,
                X0,
                y0_known=y0_known,
                max_iter_override=max_iter_override,
            )
        else:
            return self.optimize_sequential_run(
                timeout_start,
                X0,
                y0_known=y0_known,
                max_iter_override=max_iter_override,
                shared_best_y=shared_best_y,
                shared_lock=shared_lock,
            )

    def evaluate_function(self, X: np.ndarray) -> np.ndarray:
        """Evaluate objective function at points X.
        Used in the optimize() method to evaluate the objective function.

        Input Space: `X` is expected in Transformed and Mapped Space (Internal scale, Reduced dimensions).
        Process as follows:
            1. Expands `X` to Transformed Space (Full dimensions) if dimension reduction is active.
            2. Inverse transforms `X` to Natural Space (Original scale).
            3. Evaluates the user function with points in Natural Space.

        If dimension reduction is active, expands `X` to full dimensions before evaluation.
        Supports both single-objective and multi-objective functions. For multi-objective
        functions, converts to single-objective using `mo2so` method.

        Args:
            X (ndarray): Points to evaluate in Transformed and Mapped Space, shape (n_samples, n_reduced_features).

        Returns:
            ndarray: Function values, shape (n_samples,).

        Examples:
            ```{python}
            import numpy as np
            from spotoptim import SpotOptim
            # Single-objective function
            opt_so = SpotOptim(
                fun=lambda X: np.sum(X**2, axis=1),
                bounds=[(-5, 5), (-5, 5)],
                max_iter=10,
                n_initial=5
            )
            X = np.array([[1.0, 2.0], [3.0, 4.0]])
            y = opt_so.evaluate_function(X)
            print(f"Single-objective output: {y}")
            ```

            ```{python}
            import numpy as np
            from spotoptim import SpotOptim
            # Multi-objective function (default: use first objective)
            opt_mo = SpotOptim(
                fun=lambda X: np.column_stack([
                    np.sum(X**2, axis=1),
                    np.sum((X-1)**2, axis=1)
                ]),
                bounds=[(-5, 5), (-5, 5)],
                max_iter=10,
                n_initial=5
            )
            y_mo = opt_mo.evaluate_function(X)
            print(f"Multi-objective output (first obj): {y_mo}")
            ```
        """
        # Ensure X is 2D
        X = np.atleast_2d(X)

        # Expand to full dimensions if needed
        if self.red_dim:
            X = self.to_all_dim(X)

        # Apply inverse transformations to get original scale for function evaluation
        X_original = self.inverse_transform_X(X)

        # Map factor variables to original string values
        X_for_eval = self.map_to_factor_values(X_original)

        # Evaluate function
        y_raw = self.fun(X_for_eval, *self.args, **self.kwargs)

        # Convert to numpy array if needed
        if not isinstance(y_raw, np.ndarray):
            y_raw = np.array([y_raw])

        # Handle multi-objective case
        y = self.mo2so(y_raw)

        # Ensure y is 1D
        if y.ndim > 1:
            y = y.ravel()

        return y

    def get_best_xy_initial_design(self) -> None:
        """Determine and store the best point from initial design.
        Finds the best (minimum) function value in the initial design,
        stores the corresponding point and value in instance attributes,
        and optionally prints the results if verbose mode is enabled.
        For noisy functions, also reports the mean best value.

        Note:
            This method assumes self.X_ and self.y_ have been initialized
            with the initial design evaluations.

        Returns:
            None

        Examples:
            ```{python}
            import numpy as np
            from spotoptim import SpotOptim
            from spotoptim.function import sphere
            opt = SpotOptim(
                fun=sphere,
                bounds=[(-5, 5), (-5, 5)],
                n_initial=5,
                verbose=True
            )
            # Simulate initial design (normally done in optimize())
            opt.X_ = np.array([[1, 2], [0, 0], [2, 1]])
            opt.y_ = np.array([5.0, 0.0, 5.0])
            opt.get_best_xy_initial_design()
            print(f"Best x: {opt.best_x_}")
            print(f"Best y: {opt.best_y_}")
            ```

            ```{python}
            import numpy as np
            from spotoptim import SpotOptim
            from spotoptim.function import noisy_sphere
            # With noisy function
            opt_noise = SpotOptim(
                fun=noisy_sphere,
                bounds=[(-5, 5), (-5, 5)],
                n_initial=5,
                repeats_surrogate=2,
                verbose=True
            )
            opt_noise.X_ = np.array([[1, 2], [0, 0], [2, 1]])
            opt_noise.y_ = np.array([5.0, 0.0, 5.0])
            opt_noise.min_mean_y = 0.5  # Simulated mean best
            opt_noise.get_best_xy_initial_design()
            print(f"Best x: {opt_noise.best_x_}")
            print(f"Best y: {opt_noise.best_y_}")
            ```
        """
        # Initial best
        best_idx = np.argmin(self.y_)
        self.best_x_ = self.X_[best_idx].copy()
        self.best_y_ = self.y_[best_idx]

        if self.verbose:
            if (self.repeats_initial > 1) or (self.repeats_surrogate > 1):
                print(
                    f"Initial best: f(x) = {self.best_y_:.6f}, mean best: f(x) = {self.min_mean_y:.6f}"
                )
            else:
                print(f"Initial best: f(x) = {self.best_y_:.6f}")

    def _optimize_acquisition_tricands(self) -> np.ndarray:
        """Optimize using geometric infill strategy via triangulation candidates.

        Returns:
            ndarray: The optimized point(s).

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> opt = SpotOptim(
            ...     fun=lambda X: np.sum(X**2, axis=1),
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     n_initial=5
            ... )
            >>> # Requires points to work
            >>> opt.X_ = np.random.rand(10, 2)
            >>> x_next = opt._optimize_acquisition_tricands()
            >>> x_next.shape[1] == 2
            True
        """
        return _acq.optimize_acquisition_tricands(self)

    def _prepare_de_kwargs(self, x0=None):
        """Prepare kwargs for differential_evolution, extracting options if necessary."""
        return _acq.prepare_de_kwargs(self, x0)

    def _optimize_acquisition_de(self) -> np.ndarray:
        """Optimize using differential evolution.

        Returns:
            ndarray: The optimized point(s).

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> opt = SpotOptim(
            ...     fun=lambda X: np.sum(X**2, axis=1),
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     n_initial=5
            ... )
            >>> # Requires surrogate model and points to work effectively
            >>> # but can run without them (will return randomish or best_x if set)
            >>> x_next = opt._optimize_acquisition_de()
            >>> x_next.shape[0] >= 0
            True
        """
        return _acq.optimize_acquisition_de(self)

    def _optimize_acquisition_scipy(self) -> np.ndarray:
        """Optimize using scipy.optimize.minimize interface (default).


        Returns:
            np.ndarray: The optimized acquisition function values.

        Raises:
            ValueError: If acquisition optimizer is not a string or callable.

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>>
            >>> # Define objective function
            >>> def fun(x): return np.sum(x**2, axis=1)
            >>>
            >>> # Initialize optimizer with a scipy-compatible acquisition optimizer
            >>> # Note: default is 'differential_evolution' which uses a different method
            >>> optimizer = SpotOptim(
            ...     fun=fun,
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     acquisition_optimizer="L-BFGS-B"
            ... )
            >>>
            >>> # Create some dummy data to fit the surrogate model
            >>> X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
            >>> y = fun(X)
            >>>
            >>> # Fit the surrogate model manually
            >>> # Note: this is normally handled inside optimize()
            >>> optimizer.fit_surrogate(X, y)
            >>>
            >>> # Optimize the acquisition function using scipy's minimize
            >>> x_next = optimizer._optimize_acquisition_scipy()
            >>> x_next.shape
            (2,)

        """
        return _acq.optimize_acquisition_scipy(self)

    def _try_optimizer_candidates(
        self, n_needed: int = 1, current_batch: Optional[List[np.ndarray]] = None
    ) -> List[np.ndarray]:
        """Try candidates proposed by the acquisition result optimizer.

        Args:
            n_needed (int): Number of candidates needed.
            current_batch (list): Points already selected in current iteration (to check distance against).

        Returns:
            List[ndarray]: List of unique valid candidate points found.
        """
        return _acq.try_optimizer_candidates(
            self, n_needed=n_needed, current_batch=current_batch
        )

    def remove_nan(
        self, X: np.ndarray, y: np.ndarray, stop_on_zero_return: bool = True
    ) -> tuple:
        """Remove rows where y contains NaN or inf values.
        Used in the optimize() method after function evaluations.

        Args:
            X (ndarray): Design matrix, shape (n_samples, n_features).
            y (ndarray): Objective values, shape (n_samples,).
            stop_on_zero_return (bool): If True, raise error when all values are removed.

        Returns:
            tuple: (X_clean, y_clean) with NaN/inf rows removed.

        Raises:
            ValueError: If all values are NaN/inf and stop_on_zero_return is True.

        Examples:
            ```{python}
            import numpy as np
            from spotoptim import SpotOptim
            from spotoptim.function import sphere
            opt = SpotOptim(fun=sphere, bounds=[(-5, 5)])
            X = np.array([[1, 2], [3, 4], [5, 6]])
            y = np.array([1.0, np.nan, np.inf])
            X_clean, y_clean = opt.remove_nan(X, y, stop_on_zero_return=False)
            print("Clean X:", X_clean)
            print("Clean y:", y_clean)
            ```
        """
        return _acq.remove_nan(self, X, y, stop_on_zero_return=stop_on_zero_return)

    def _handle_acquisition_failure(self) -> np.ndarray:
        """Handle acquisition failure by proposing new design points.
        Used in the suggest_next_infill_point() method.
        This method is called when no new design points can be suggested
        by the surrogate model (e.g., when the proposed point is too close
        to existing points). It proposes a random space-filling design as a fallback.

        Returns:
            ndarray: New design point as a fallback, shape (n_features,).

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> opt = SpotOptim(
            ...     fun=lambda X: np.sum(X**2, axis=1),
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     acquisition_failure_strategy='random'
            ... )
            >>> opt.X_ = np.array([[0, 0], [1, 1]])
            >>> opt.y_ = np.array([0, 2])
            >>> x_fallback = opt._handle_acquisition_failure()
            >>> x_fallback.shape
            (2,)
            >>> print(x_fallback)
            [some new point within bounds]
        """
        return _acq.handle_acquisition_failure(self)

    def _try_fallback_strategy(
        self, max_attempts: int = 10, current_batch: Optional[List[np.ndarray]] = None
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Try fallback strategy (e.g. random search) to find a unique point.
        Calls _handle_acquisition_failure.

        Args:
            max_attempts (int): Maximum number of fallback attempts.
            current_batch (list): Points already selected in current iteration.

        Returns:
            Tuple[Optional[ndarray], ndarray]:
                - The first element is the unique valid candidate point if found, else None.
                - The second element is the last attempted point.
        """
        return _acq.try_fallback_strategy(
            self, max_attempts=max_attempts, current_batch=current_batch
        )

    def get_shape(self, y: np.ndarray) -> Tuple[int, Optional[int]]:
        """Get the shape of the objective function output.

        Args:
            y (ndarray): Objective function output, shape (n_samples,) or (n_samples, n_objectives).

        Returns:
            tuple: (n_samples, n_objectives) where n_objectives is None for single-objective.

        Examples:
            ```{python}
            import numpy as np
            from spotoptim import SpotOptim
            opt = SpotOptim(
                fun=lambda X: np.sum(X**2, axis=1),
                bounds=[(-5, 5), (-5, 5)],
                max_iter=10,
                n_initial=5
            )
            y_single = np.array([1.0, 2.0, 3.0])
            n, m = opt.get_shape(y_single)
            print(f"n={n}, m={m}")
            y_multi = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            n, m = opt.get_shape(y_multi)
            print(f"n={n}, m={m}")
            ```
        """
        if y.ndim == 1:
            return y.shape[0], None
        elif y.ndim == 2:
            return y.shape[0], y.shape[1]
        else:
            # For higher dimensions, flatten to 1D
            return y.size, None

    def optimize_acquisition_func(self) -> np.ndarray:
        """Optimize the acquisition function to find the next point to evaluate.

        Returns:
            ndarray: The optimized point(s).
                If acquisition_fun_return_size == 1, returns 1D array of shape (n_features,).
                If acquisition_fun_return_size > 1, returns 2D array of shape (N, n_features),
                where N is min(acquisition_fun_return_size, population_size).

        Examples:
            ```{python}
            import numpy as np
            from spotoptim import SpotOptim
            def sphere(X):
                X = np.atleast_2d(X)
                return np.sum(X**2, axis=1)
            opt = SpotOptim(
                fun=sphere,
                bounds=[(-5, 5), (-5, 5)],
                n_initial=5,
                max_iter=10,
                seed=0,
            )
            opt.optimize()
            x_next = opt.suggest_next_infill_point()
            print("Next point to evaluate:", x_next)
            ```
        """
        return _acq.optimize_acquisition_func(self)

    # ====================
    # TASK_OPTIM_SEQ:
    # * determine_termination()
    # * apply_penalty_NA()
    # * update_best_main_loop()
    # * handle_NA_new_points()
    # * optimize_sequential_run()
    # * _initialize_run()
    # * update_repeats_infill_points()
    # * _run_sequential_loop()
    # ====================

    def optimize_sequential_run(
        self,
        timeout_start: float,
        X0: Optional[np.ndarray] = None,
        y0_known: Optional[float] = None,
        max_iter_override: Optional[int] = None,
        shared_best_y=None,
        shared_lock=None,
    ) -> Tuple[str, OptimizeResult]:
        """Perform a single sequential optimization run.
        Calls _initialize_run, rm_initial_design_NA_values, check_size_initial_design, init_storage, get_best_xy_initial_design, and _run_sequential_loop.

        Args:
            timeout_start (float): Start time for timeout.
            X0 (Optional[np.ndarray]): Initial design points in Natural Space, shape (n_initial, n_features).
            y0_known (Optional[float]): Known best value for initial design.
            max_iter_override (Optional[int]): Override for maximum number of iterations.
            shared_best_y (Optional[float]): Shared best value for parallel runs.
            shared_lock (Optional[Lock]): Shared lock for parallel runs.

        Returns:
            Tuple[str, OptimizeResult]: Tuple containing status and optimization result.

        Raises:
            ValueError: If the initial design has no valid points after removing NaN/inf values, or if the initial design is too small to proceed.

        Examples:
            ```{python}
            import time
            import numpy as np
            from spotoptim import SpotOptim
            from spotoptim.function import sphere
            opt = SpotOptim(fun=sphere,
                            bounds=[(-5, 5), (-5, 5)],
                            n_initial=5,
                            max_iter=10,
                            seed=0,
                            n_jobs=1,  # Use sequential optimization for deterministic output
                            verbose=True
             )
            status, result = opt.optimize_sequential_run(timeout_start=time.time())
            print(status)
            print(result.message.splitlines()[0])
            ```
        """

        # Store shared variable if provided
        self.shared_best_y = shared_best_y
        self.shared_lock = shared_lock

        # Initialize: Set seed, Design, Evaluate Initial Design, Init Storage & TensorBoard
        X0, y0 = self._initialize_run(X0, y0_known)

        # Handle NaN/inf values in initial design (remove invalid points)
        X0, y0, n_evaluated = self.rm_initial_design_NA_values(X0, y0)

        # Check if we have enough valid points to continue
        self.check_size_initial_design(y0, n_evaluated)

        # Initialize storage and statistics
        self.init_storage(X0, y0)
        self._zero_success_count = 0
        self._success_history = []  # Clear success history for new run

        # Update stats after initial design
        self.update_stats()

        # Log initial design to TensorBoard
        self._init_tensorboard()

        # Determine and report initial best
        self.get_best_xy_initial_design()

        # Run the main sequential optimization loop
        effective_max_iter = (
            max_iter_override if max_iter_override is not None else self.max_iter
        )
        return self._run_sequential_loop(timeout_start, effective_max_iter)

    def _initialize_run(
        self, X0: Optional[np.ndarray], y0_known: Optional[float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize optimization run: seed, design generation, initial evaluation.
        Called from optimize_sequential_run (sequential path only).

        Args:
            X0 (Optional[np.ndarray]): Initial design points in Natural Space, shape (n_initial, n_features).
            y0_known (Optional[float]): Known best value for initial design.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing initial design and corresponding objective values.

        Raises:
            ValueError: If the initial design has no valid points after removing NaN/inf values, or if the initial design is too small to proceed.

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> opt = SpotOptim(
            ...     fun=lambda X: np.sum(X**2, axis=1),
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     n_initial=5,
            ...     seed=0,
            ...     x0=np.array([0.0, 0.0]),
            ...     verbose=True
            ... )
            >>> X0, y0 = opt._initialize_run(X0=None, y0_known=None)
            >>> X0.shape
            (5, 2)
            >>> np.allclose(y0, np.sum(X0**2, axis=1))
            True
        """
        # Set seed for reproducibility
        self.set_seed()

        # Set initial design (generate or process user-provided points)
        X0 = self.get_initial_design(X0)

        # Curate initial design (remove duplicates, generate additional points if needed, repeat if necessary)
        X0 = self.curate_initial_design(X0)

        # Evaluate initial design
        if y0_known is not None and self.x0 is not None:
            # Identify injected point to skip evaluation
            dists = np.linalg.norm(X0 - self.x0, axis=1)
            # Use a small tolerance for matching
            matches = dists < 1e-9

            if np.any(matches):
                if self.verbose:
                    print("Skipping re-evaluation of injected best point.")

                # Initialize y0
                y0 = np.empty(len(X0))
                y0[:] = np.nan

                # Set known values
                y0[matches] = y0_known

                # Evaluate others
                not_matches = ~matches
                if np.any(not_matches):
                    y0_others = self.evaluate_function(X0[not_matches])
                    y0[not_matches] = y0_others
            else:
                # Injected point lost during curation? Should not happen if it was unique
                y0 = self.evaluate_function(X0)
        else:
            y0 = self.evaluate_function(X0)

        return X0, y0

    def rm_initial_design_NA_values(
        self, X0: np.ndarray, y0: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Remove NaN/inf values from initial design evaluations.
        This method filters out design points that returned NaN or inf values
        during initial evaluation. Unlike the sequential optimization phase where
        penalties are applied, initial design points with invalid values are
        simply removed.

        Args:
            X0 (ndarray): Initial design points in internal scale,
                shape (n_samples, n_features).
            y0 (ndarray): Function values at X0, shape (n_samples,).

        Returns:
            Tuple[ndarray, ndarray, int]: Filtered (X0, y0) with only finite values
                and the original count before filtering. X0 has shape (n_valid, n_features),
                y0 has shape (n_valid,), and the int is the original size.

        Examples:
            ```{python}
            import numpy as np
            from spotoptim import SpotOptim
            from spotoptim.function import sphere
            opt = SpotOptim(
                fun=sphere,
                bounds=[(-5, 5), (-5, 5)],
                n_initial=10
            )
            X0 = np.array([[1, 2], [3, 4], [5, 6]])
            y0 = np.array([5.0, np.nan, np.inf])
            X0_clean, y0_clean, n_eval = opt.rm_initial_design_NA_values(X0, y0)
            print(X0_clean.shape) # (1, 2)
            print(y0_clean) # array([5.])
            print(n_eval) # 3
            # All valid values - no filtering
            X0 = np.array([[1, 2], [3, 4]])
            y0 = np.array([5.0, 25.0])
            X0_clean, y0_clean, n_eval = opt.rm_initial_design_NA_values(X0, y0)
            print(X0_clean.shape) # (2, 2)
            print(n_eval) # 2
            ```
        """
        # Handle NaN/inf values in initial design - REMOVE them instead of applying penalty

        # If y0 contains None (object dtype), convert None to NaN and ensure float dtype
        if y0.dtype == object:
            # Create a float array, replacing None with NaN
            # Use list comprehension for safe conversion of None
            y0 = np.array([np.nan if v is None else v for v in y0], dtype=float)

        finite_mask = np.isfinite(y0)
        n_non_finite = np.sum(~finite_mask)

        if n_non_finite > 0:
            if self.verbose:
                print(
                    f"Warning: {n_non_finite} initial design point(s) returned NaN/inf "
                    f"and will be ignored (reduced from {len(y0)} to {np.sum(finite_mask)} points)"
                )
            X0 = X0[finite_mask]
            y0 = y0[finite_mask]

            # Also filter y_mo if it exists (must match y0 size)
            if self.y_mo is not None:
                if len(self.y_mo) == len(finite_mask):  # Safety check
                    self.y_mo = self.y_mo[finite_mask]
                else:
                    # Fallback or warning if sizes already mismatched (shouldn't happen here normally)
                    if self.verbose:
                        print(
                            f"Warning: y_mo size ({len(self.y_mo)}) != mask size ({len(finite_mask)}) in initial design filtering"
                        )
                    # Try to filter only if sizes match, otherwise we might be in inconsistent state

        return X0, y0, len(finite_mask)

    def check_size_initial_design(self, y0: np.ndarray, n_evaluated: int) -> None:
        """Validate that initial design has sufficient points for surrogate fitting.

        Checks if the number of valid initial design points meets the minimum
        requirement for fitting a surrogate model. The minimum required is the
        smaller of:
            * (a) typical minimum for surrogate fitting (3 for multi-dimensional, 2 for 1D), or
            * (b) what the user requested (`n_initial`).

        Args:
            y0 (ndarray): Function values at initial design points (after filtering),
                shape (n_valid,).
            n_evaluated (int): Original number of points evaluated before filtering.

        Returns:
            None

        Raises:
            ValueError: If the number of valid points is less than the minimum required.

        Examples:
            ```{python}
            import numpy as np
            from spotoptim import SpotOptim
            from spotoptim.function import sphere

            opt = SpotOptim(
                fun=sphere,
                bounds=[(-5, 5), (-5, 5)],
                n_initial=10
            )
            # Sufficient points - no error
            y0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            opt.check_size_initial_design(y0, n_evaluated=10)

            # Insufficient points - raises ValueError
            y0_small = np.array([1.0])
            try:
                opt.check_size_initial_design(y0_small, n_evaluated=10)
            except ValueError as e:
                print(f"Error: {e}")

            # With verbose output
            opt_verbose = SpotOptim(
                fun=sphere,
                bounds=[(-5, 5), (-5, 5)],
                n_initial=10,
                verbose=True
            )
            y0_reduced = np.array([1.0, 2.0, 3.0])  # Less than n_initial but valid
            opt_verbose.check_size_initial_design(y0_reduced, n_evaluated=10)
            ```
        """
        # Check if we have enough points to continue
        # Use the smaller of: (a) typical minimum for surrogate fitting, or (b) what user requested
        min_points_typical = 3 if self.n_dim > 1 else 2
        min_points_required = min(min_points_typical, self.n_initial)

        if len(y0) < min_points_required:
            error_msg = (
                f"Insufficient valid initial design points: only {len(y0)} finite value(s) "
                f"out of {n_evaluated} evaluated. Need at least {min_points_required} "
                f"points to fit surrogate model. Please check your objective function or increase n_initial."
            )
            raise ValueError(error_msg)

        if len(y0) < self.n_initial and self.verbose:
            print(
                f"Note: Initial design size ({len(y0)}) is smaller than requested "
                f"({self.n_initial}) due to NaN/inf values"
            )

    def _run_sequential_loop(
        self, timeout_start: float, effective_max_iter: int
    ) -> Tuple[str, OptimizeResult]:
        """Execute the main sequential optimization loop.

        Args:
             timeout_start (float): Start time for timeout.
             effective_max_iter (int): Maximum number of iterations for this run (may be overridden for restarts).

         Returns:
             Tuple[str, OptimizeResult]: Tuple containing status and optimization result.

         Raises:
             ValueError:
                If excessive consecutive failures occur (e.g., due to NaN/inf values in evaluations), indicating a potential issue with the objective function.

         Examples:
             >>> import time
             >>> import numpy as np
             >>> from spotoptim import SpotOptim
             >>> opt = SpotOptim(
             ...     fun=lambda X: np.sum(X**2, axis=1),
             ...     bounds=[(-5, 5), (-5, 5)],
             ...     n_initial=5,
             ...     max_iter=10,
             ...     seed=0,
             ...     n_jobs=1,  # Use sequential optimization for deterministic output
             ...     verbose=True
             ... )
             >>> X0, y0 = opt._initialize_run(X0=None, y0_known=None)
             >>> X0, y0, n_evaluated = opt.rm_initial_design_NA_values(X0, y0)
             >>> opt.check_size_initial_design(y0, n_evaluated)
             >>> opt.init_storage(X0, y0)
             >>> opt._zero_success_count = 0
             >>> opt._success_history = []
             >>> opt.update_stats()
             >>> opt.get_best_xy_initial_design()
             >>> status, result = opt._run_sequential_loop(timeout_start=time.time(), effective_max_iter=10)
             >>> print(status)
             FINISHED
             >>> print(result.message.splitlines()[0])
             Optimization terminated: maximum evaluations (10) reached
        """
        consecutive_failures = 0

        while (len(self.y_) < effective_max_iter) and (
            time.time() < timeout_start + self.max_time * 60
        ):
            # Check for excessive consecutive failures (infinite loop prevention)
            if consecutive_failures > self.max_iter:
                msg = (
                    f"Optimization stopped due to {consecutive_failures} consecutive "
                    "invalid evaluations (NaN/inf). Check your objective function."
                )
                if self.verbose:
                    print(f"Warning: {msg}")
                return "FINISHED", OptimizeResult(
                    x=self.best_x_,
                    fun=self.best_y_,
                    nfev=len(self.y_),
                    nit=self.n_iter_,
                    success=False,
                    message=msg,
                    X=self.X_,
                    y=self.y_,
                )

            # Increment iteration counter. This is not the same as number of function evaluations.
            self.n_iter_ += 1

            # Fit surrogate (use mean_y if noise, otherwise y_)
            self.fit_scheduler()

            # Apply OCBA for noisy functions
            X_ocba = self.apply_ocba()

            # Suggest next point
            x_next = self.suggest_next_infill_point()

            # Repeat next point if repeats_surrogate > 1
            x_next_repeated = self.update_repeats_infill_points(x_next)

            # Append OCBA points to new design points (if applicable)
            if X_ocba is not None:
                x_next_repeated = append(X_ocba, x_next_repeated, axis=0)

            # Evaluate next point(s) including OCBA points
            y_next = self.evaluate_function(x_next_repeated)

            # Handle NaN/inf values in new evaluations
            x_next_repeated, y_next = self._handle_NA_new_points(
                x_next_repeated, y_next
            )
            if x_next_repeated is None:
                consecutive_failures += 1
                continue  # Skip iteration if all evaluations were invalid

            # Reset failure counter if we got valid points
            consecutive_failures = 0

            # Update success rate BEFORE updating storage (so it compares against previous best)
            self.update_success_rate(y_next)

            # Check for restart
            if self.success_rate == 0.0:
                self._zero_success_count += 1
            else:
                self._zero_success_count = 0

            if self._zero_success_count >= self.restart_after_n:
                if self.verbose:
                    print(
                        f"Restarting optimization: success_rate 0 for {self._zero_success_count} iterations."
                    )

                status_message = "Restart triggered due to lack of improvement."

                # Expand results to full dimensions if needed
                best_x_full = (
                    self.to_all_dim(self.best_x_.reshape(1, -1))[0]
                    if self.red_dim
                    else self.best_x_
                )
                X_full = self.to_all_dim(self.X_) if self.red_dim else self.X_

                # Map factor variables back to original strings
                best_x_result = self.map_to_factor_values(best_x_full.reshape(1, -1))[0]
                X_result = (
                    self.map_to_factor_values(X_full) if self._factor_maps else X_full
                )

                res = OptimizeResult(
                    x=best_x_result,
                    fun=self.best_y_,
                    nfev=len(self.y_),
                    nit=self.n_iter_,
                    success=False,
                    message=status_message,
                    X=X_result,
                    y=self.y_,
                )
                return "RESTART", res

            # Update storage
            self.update_storage(x_next_repeated, y_next)

            # Update stats
            self.update_stats()

            # Log to TensorBoard
            if self.tb_writer is not None:
                # Log each new evaluation
                for i in range(len(y_next)):
                    self._write_tensorboard_hparams(x_next_repeated[i], y_next[i])
                self._write_tensorboard_scalars()

            # Update best solution
            self._update_best_main_loop(
                x_next_repeated, y_next, start_time=timeout_start
            )

        # Expand results to full dimensions if needed
        # Note: best_x_ and X_ are already in original scale (stored that way)
        best_x_full = (
            self.to_all_dim(self.best_x_.reshape(1, -1))[0]
            if self.red_dim
            else self.best_x_
        )
        X_full = self.to_all_dim(self.X_) if self.red_dim else self.X_

        # Determine termination reason
        status_message = self.determine_termination(timeout_start)

        # Append statistics to match scipy.optimize.minimize format
        message = (
            f"{status_message}\n"
            f"         Current function value: {float(self.best_y_):.6f}\n"
            f"         Iterations: {self.n_iter_}\n"
            f"         Function evaluations: {len(self.y_)}"
        )

        # Close TensorBoard writer
        self._close_tensorboard_writer()

        # Map factor variables back to original strings for results
        best_x_result = self.map_to_factor_values(best_x_full.reshape(1, -1))[0]
        X_result = self.map_to_factor_values(X_full) if self._factor_maps else X_full

        # Return scipy-style result
        return "FINISHED", OptimizeResult(
            x=best_x_result,
            fun=self.best_y_,
            nfev=len(self.y_),
            nit=self.n_iter_,
            success=True,
            message=message,
            X=X_result,
            y=self.y_,
        )

    def determine_termination(self, timeout_start: float) -> str:
        """Determine termination reason for optimization.
        Checks the termination conditions and returns an appropriate message
        indicating why the optimization stopped. Three possible termination
        conditions are checked in order of priority:
            1. Maximum number of evaluations reached
            2. Maximum time limit exceeded
            3. Successful completion (neither limit reached)

        Args:
            timeout_start (float): Start time of optimization (from time.time()).

        Returns:
            str: Message describing the termination reason.

        Examples:
            ```{python}
            import numpy as np
            import time
            from spotoptim import SpotOptim
            opt = SpotOptim(
                fun=lambda X: np.sum(X**2, axis=1),
                bounds=[(-5, 5), (-5, 5)],
                max_iter=10,
                max_time=10.0
            )
            # Case 1: Maximum evaluations reached
            opt.y_ = np.zeros(20)  # Simulate 20 evaluations
            start_time = time.time()
            msg = opt.determine_termination(start_time)
            print(msg)
            ```
            ```{python}
            # Case 2: Time limit exceeded
            import numpy as np
            import time
            from spotoptim import SpotOptim
            opt.y_ = np.zeros(10)  # Only 10 evaluations
            start_time = time.time() - 700  # Simulate 11.67 minutes elapsed
            msg = opt.determine_termination(start_time)
            print(msg)
            ```
            ```{python}
            # Case 3: Successful completion
            import numpy as np
            import time
            from spotoptim import SpotOptim
            opt.y_ = np.zeros(10)  # Under max_iter
            start_time = time.time()  # Just started
            msg = opt.determine_termination(start_time)
            print(msg)
            ```
        """
        # Determine termination reason
        elapsed_time = time.time() - timeout_start
        if len(self.y_) >= self.max_iter:
            message = f"Optimization terminated: maximum evaluations ({self.max_iter}) reached"
        elif elapsed_time >= self.max_time * 60:
            message = (
                f"Optimization terminated: time limit ({self.max_time:.2f} min) reached"
            )
        else:
            message = "Optimization finished successfully"

        return message

    def apply_penalty_NA(
        self,
        y: np.ndarray,
        y_history: Optional[np.ndarray] = None,
        penalty_value: Optional[float] = None,
        sd: float = 0.1,
    ) -> np.ndarray:
        """Replace NaN and infinite values with penalty plus random noise.
        Used in the optimize() method after function evaluations.
        This method follows the approach from spotpython.utils.repair.apply_penalty_NA,
        replacing NaN/inf values with a penalty value plus random noise to avoid
        identical penalty values.

        Args:
            y (ndarray): Array of objective function values to be repaired.
            y_history (ndarray, optional): Historical objective function values used for
                computing penalty statistics. If None, uses y itself. Default is None.
            penalty_value (float, optional): Value to replace NaN/inf with.
                If None, computes penalty as: max(finite_y_history) + 3 * std(finite_y_history).
                If all values are NaN/inf or only one finite value exists, falls back
                to self.penalty_val. Default is None.
            sd (float): Standard deviation for normal distributed random noise added to penalty.
                Default is 0.1.

        Returns:
            ndarray:
                Array with NaN/inf replaced by penalty_value + random noise
                (normal distributed with mean 0 and standard deviation sd).

        Examples:
            ```{python}
            import numpy as np
            from spotoptim import SpotOptim
            opt = SpotOptim(fun=lambda X: np.sum(X**2, axis=1), bounds=[(-5, 5)])
            y_hist = np.array([1.0, 2.0, 3.0, 5.0])
            y_new = np.array([4.0, np.nan, np.inf])
            y_clean = opt.apply_penalty_NA(y_new, y_history=y_hist)
            print(f"np.all(np.isfinite(y_clean)): {np.all(np.isfinite(y_clean))}")
            print(f"y_clean: {y_clean}")
            # NaN/inf replaced with worst value from history + 3*std + noise
            print(f"y_clean[1] > 5.0: {y_clean[1] > 5.0}")  # Should be larger than max finite value in history
            ```
        """

        # Ensure y is a float array (maps non-convertible values like "error" or None to nan)
        y_flat = np.array(y).flatten()
        y = np.array([safe_float(v) for v in y_flat])
        # Identify NaN and inf values in y
        mask = ~np.isfinite(y)

        if np.any(mask):
            n_bad = np.sum(mask)

            # Compute penalty_value if not provided
            if penalty_value is None:
                # Get finite values from history for statistics
                # Use y_history if provided, otherwise fall back to y itself
                if y_history is not None:
                    finite_values = y_history[np.isfinite(y_history)]
                else:
                    # Use current y values
                    finite_values = y[~mask]

                # If we have at least 2 finite values, compute adaptive penalty
                if len(finite_values) >= 2:
                    max_y = np.max(finite_values)
                    std_y = np.std(finite_values, ddof=1)
                    penalty_value = max_y + 3.0 * std_y

                    if self.verbose:
                        print(
                            f"Warning: Found {n_bad} NaN/inf value(s), replacing with "
                            f"adaptive penalty (max + 3*std = {penalty_value:.4f})"
                        )
                else:
                    # Fallback to self.penalty if insufficient finite values
                    if self.penalty_val is not None:
                        penalty_value = self.penalty_val
                    elif len(finite_values) == 1:
                        # Use the single finite value + a large constant
                        penalty_value = finite_values[0] + 1000.0
                    else:
                        # All values are NaN/inf, use a large default
                        penalty_value = 1e10

                    if self.verbose:
                        print(
                            f"Warning: Found {n_bad} NaN/inf value(s), insufficient finite values "
                            f"for adaptive penalty. Using penalty_value = {penalty_value}"
                        )
            else:
                if self.verbose:
                    print(
                        f"Warning: Found {n_bad} NaN/inf value(s), replacing with {penalty_value} + noise"
                    )

            # Generate random noise and add to penalty
            random_noise = self.rng.normal(0, sd, y.shape)
            penalty_values = penalty_value + random_noise

            # Replace NaN/inf with penalty + noise
            y[mask] = penalty_values[mask]

        return y

    def _update_best_main_loop(
        self,
        x_next_repeated: np.ndarray,
        y_next: np.ndarray,
        start_time: Optional[float] = None,
    ) -> None:
        """Update best solution found during main optimization loop.
        Checks if any new evaluations improve upon the current best solution.
        If improvement is found, updates best_x_ and best_y_ attributes and
        prints progress if verbose mode is enabled.

        Args:
            x_next_repeated (ndarray): Design points that were evaluated in transformed space,
                shape (n_eval, n_features).
            y_next (ndarray): Function values at x_next_repeated, shape (n_eval,).

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> opt = SpotOptim(
            ...     fun=lambda X: np.sum(X**2, axis=1),
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     n_initial=5,
            ...     verbose=True
            ... )
            >>> # Simulate optimization state
            >>> opt.n_iter_ = 1
            >>> opt.best_x_ = np.array([1.0, 1.0])
            >>> opt.best_y_ = 2.0
            >>>
            >>> # Case 1: New best found
            >>> x_new = np.array([[0.1, 0.1], [0.5, 0.5]])
            >>> y_new = np.array([0.02, 0.5])
            >>> opt._update_best_main_loop(x_new, y_new)
            Iteration 1: New best f(x) = 0.020000
            >>> opt.best_y_
            0.02
            >>>
            >>> # Case 2: No improvement
            >>> opt.n_iter_ = 2
            >>> x_no_improve = np.array([[1.5, 1.5]])
            >>> y_no_improve = np.array([4.5])
            >>> opt._update_best_main_loop(x_no_improve, y_no_improve)
            Iteration 2: f(x) = 4.500000
            >>>
            >>> # Case 3: With noisy function
            >>> opt_noise = SpotOptim(
            ...     fun=lambda X: np.sum(X**2, axis=1),
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     repeats_surrogate=2,
            ...     verbose=True
            ... )
            >>> opt_noise.n_iter_ = 1
            >>> opt_noise.best_y_ = 2.0
            >>> opt_noise.min_mean_y = 1.5
            >>> y_noise = np.array([0.5])
            >>> x_noise = np.array([[0.5, 0.5]])
            >>> opt_noise._update_best_main_loop(x_noise, y_noise)
            Iteration 1: New best f(x) = 0.500000, mean best: f(x) = 1.500000
        """
        # Update best
        # Determine global best value for printing if shared variable exists
        global_best_val = None
        if hasattr(self, "shared_best_y") and self.shared_best_y is not None:
            # Sync with global shared value
            lock_obj = getattr(self, "shared_lock", None)
            if lock_obj is not None:
                with lock_obj:
                    if (
                        self.best_y_ is not None
                        and self.best_y_ < self.shared_best_y.value
                    ):
                        self.shared_best_y.value = self.best_y_

                    min_y_next = np.min(y_next)
                    if min_y_next < self.shared_best_y.value:
                        self.shared_best_y.value = min_y_next

                    global_best_val = self.shared_best_y.value

        current_best = np.min(y_next)
        if current_best < self.best_y_:
            best_idx_in_new = np.argmin(y_next)
            # x_next_repeated is in transformed space, convert to original for storage
            self.best_x_ = self.inverse_transform_X(
                x_next_repeated[best_idx_in_new].reshape(1, -1)
            )[0]
            self.best_y_ = current_best

            if self.verbose:
                # Calculate progress
                if self.max_time != np.inf and start_time is not None:
                    progress = (time.time() - start_time) / (self.max_time * 60) * 100
                    progress_str = f"Time: {progress:.1f}%"
                else:
                    prev_evals = sum(res.nfev for res in self.restarts_results_)
                    progress = (prev_evals + self.counter) / self.max_iter * 100
                    progress_str = f"Evals: {progress:.1f}%"

                msg = f"Iter {self.n_iter_}"
                if global_best_val is not None:
                    msg += f" | GlobalBest: {global_best_val:.6f}"
                msg += f" | Best: {self.best_y_:.6f} | Rate: {self.success_rate:.2f} | {progress_str}"

                if (self.repeats_initial > 1) or (self.repeats_surrogate > 1):
                    msg += f" | Mean Best: {self.min_mean_y:.6f}"

                print(msg)
        elif self.verbose:
            if self.max_time != np.inf and start_time is not None:
                progress = (time.time() - start_time) / (self.max_time * 60) * 100
                progress_str = f"Time: {progress:.1f}%"
            else:
                prev_evals = sum(res.nfev for res in self.restarts_results_)
                progress = (prev_evals + self.counter) / self.max_iter * 100
                progress_str = f"Evals: {progress:.1f}%"

            current_val = np.min(y_next)
            msg = f"Iter {self.n_iter_}"
            if global_best_val is not None:
                msg += f" | GlobalBest: {global_best_val:.6f}"
            msg += f" | Best: {self.best_y_:.6f} | Curr: {current_val:.6f} | Rate: {self.success_rate:.2f} | {progress_str}"

            if (self.repeats_initial > 1) or (self.repeats_surrogate > 1):
                mean_y_new = np.mean(y_next)
                msg += f" | Mean Curr: {mean_y_new:.6f}"
            print(msg)

    def _handle_NA_new_points(
        self, x_next: np.ndarray, y_next: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Handle NaN/inf values in new evaluation points.
        Applies penalties to NaN/inf values and removes any remaining invalid points.
        If all evaluations are invalid, returns None for both arrays to signal that
        the iteration should be skipped.

        Args:
            x_next (ndarray): Design points that were evaluated, shape (n_eval, n_features).
            y_next (ndarray): Function values at x_next, shape (n_eval,).

        Returns:
            Tuple[Optional[ndarray], Optional[ndarray]]: Tuple of (x_clean, y_clean).
                Both are None if all evaluations were NaN/inf (iteration should be skipped).
                Otherwise returns filtered arrays with only finite values.

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> opt = SpotOptim(
            ...     fun=lambda X: np.sum(X**2, axis=1),
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     n_initial=5,
            ...     verbose=True
            ... )
            >>> # Simulate optimization state
            >>> opt.y_ = np.array([1.0, 2.0, 3.0])  # Historical values
            >>> opt.n_iter_ = 1
            >>>
            >>> # Case 1: Some valid values
            >>> x_next = np.array([[1, 2], [3, 4], [5, 6]])
            >>> y_next = np.array([5.0, np.nan, 10.0])
            >>> x_clean, y_clean = opt._handle_NA_new_points(x_next, y_next)
            >>> x_clean.shape
            (2, 2)
            >>> y_clean.shape
            (2,)
            >>>
            >>> # Case 2: All NaN/inf - should skip iteration
            >>> x_all_bad = np.array([[1, 2], [3, 4]])
            >>> y_all_bad = np.array([np.nan, np.inf])
            >>> x_clean, y_clean = opt._handle_NA_new_points(x_all_bad, y_all_bad)
            Warning: All new evaluations were NaN/inf, skipping iteration 1
            >>> x_clean is None
            True
            >>> y_clean is None
            True
        """
        # Handle NaN/inf values in new evaluations
        # Use historical y values (self.y_) for computing penalty statistics
        if self.penalty:
            y_next = self.apply_penalty_NA(y_next, y_history=self.y_)

        # Identify which points are valid (finite) BEFORE removing them
        # Note: remove_nan filters based on y_next finite values

        # Ensure y_next is a float array (maps non-convertible values like "error" or None to nan)
        # This is critical if the objective function returns non-numeric values and penalty=False
        if y_next.dtype == object:
            y_flat = np.array(y_next).flatten()
            y_next = np.array([safe_float(v) for v in y_flat])

        finite_mask = np.isfinite(y_next)

        X_next_clean, y_next_clean = self.remove_nan(
            x_next, y_next, stop_on_zero_return=False
        )

        # If we have multi-objective values, we need to filter them too
        # The new MO values were appended to self.y_mo in evaluate_function -> mo2so -> store_mo
        # So self.y_mo currently contains the INVALID points at the end.
        if self.y_mo is not None:
            n_new = len(y_next)
            # Check if y_mo has the new points appended
            if len(self.y_mo) >= n_new:
                # The new points are at the end of y_mo
                y_mo_new = self.y_mo[-n_new:]
                y_mo_old = self.y_mo[:-n_new]

                # Filter the new MO points using the mask from y_next
                y_mo_new_clean = y_mo_new[finite_mask]

                # Reconstruct y_mo
                if len(y_mo_old) > 0:
                    self.y_mo = (
                        np.vstack([y_mo_old, y_mo_new_clean])
                        if len(y_mo_new_clean) > 0
                        else y_mo_old
                    )
                else:
                    self.y_mo = y_mo_new_clean
            else:
                if self.verbose:
                    print(
                        "Warning: y_mo size inconsistent with new points in _handle_NA_new_points"
                    )

        # Skip this iteration if all new points were NaN/inf
        if len(y_next_clean) == 0:
            if self.verbose:
                print(
                    f"Warning: All new evaluations were NaN/inf, skipping iteration {self.n_iter_}"
                )
            return None, None

        return X_next_clean, y_next_clean

    def update_repeats_infill_points(self, x_next: np.ndarray) -> np.ndarray:
        """Repeat infill point for noisy function evaluation. Used in the sequential_loop.
        For noisy objective functions (repeats_surrogate > 1), creates multiple
        copies of the suggested point for repeated evaluation. Otherwise, returns
        the point in 2D array format.

        Args:
            x_next (ndarray): Next point to evaluate, shape (n_features,).

        Returns:
            ndarray: Points to evaluate, shape (repeats_surrogate, n_features)
                or (1, n_features) if repeats_surrogate == 1.

        Examples:
            ```{python}
            import numpy as np
            from spotoptim import SpotOptim
            from spotoptim.function import sphere, noisy_sphere
            # Without repeats

            opt = SpotOptim(
                fun=sphere,
                bounds=[(-5, 5), (-5, 5)],
                repeats_surrogate=1
            )
            x_next = np.array([1.0, 2.0])
            x_repeated = opt.update_repeats_infill_points(x_next)
            print(x_repeated.shape)

            # With repeats for noisy function
            opt_noisy = SpotOptim(
                fun=noisy_sphere,
                bounds=[(-5, 5), (-5, 5)],
                repeats_surrogate=3
            )
            x_next = np.array([1.0, 2.0])
            x_repeated = opt_noisy.update_repeats_infill_points(x_next)
            print(x_repeated.shape)
            # All three copies should be identical
            np.all(x_repeated[0] == x_repeated[1])
            ```
        """
        if x_next.ndim == 1:
            x_next = x_next.reshape(1, -1)

        if self.repeats_surrogate > 1:
            # Repeat each row repeats_surrogate times
            # Note: np.repeat with axis=0 repeats rows [r1, r1, r2, r2...]
            x_next_repeated = np.repeat(x_next, self.repeats_surrogate, axis=0)
        else:
            x_next_repeated = x_next
        return x_next_repeated

    # ====================
    # TASK_OPTIM_PARALLEL:
    # * _update_storage_steady()
    # * optimize_steady_state()
    # ====================

    def _update_storage_steady(self, x, y):
        """Helper to safely append single point (for steady state).
            This method is designed for the steady-state parallel optimization scenario, where new points are evaluated and returned asynchronously.
            It safely appends new points to the existing storage of evaluated points and their function values,
            while also updating the current best solution if the new point is better.

        Args:
            x (ndarray):
                New point(s) in original scale, shape (n_features,) or (N, n_features).
            y (float or ndarray):
                Corresponding function value(s).

        Returns:
            None. This method updates the internal state of the optimizer.

        Note:
           - This method assumes that the caller handles any necessary synchronization if used in a parallel context
           (e.g., using locks when updating shared state).

        Raises:
            ValueError: If the input shapes are inconsistent or if y is not a scalar when x is a single point.

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> opt = SpotOptim(
            ...     fun=lambda X: np.sum(X**2, axis=1),
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     n_jobs=2
            ... )
            >>> opt._update_storage_steady(np.array([1.0, 2.0]), 5.0)
            >>> print(opt.X_)
            [[1. 2.]]
            >>> print(opt.y_)
            [5.]
            >>> print(opt.best_x_)
            [1. 2.]
            >>> print(opt.best_y_)
            5.0

        """
        _steady.update_storage_steady(self, x, y)

    def optimize_steady_state(
        self,
        timeout_start: float,
        X0: Optional[np.ndarray],
        y0_known: Optional[float] = None,
        max_iter_override: Optional[int] = None,
    ) -> Tuple[str, OptimizeResult]:
        """Perform steady-state asynchronous optimization (n_jobs > 1).
        This method implements a hybrid steady-state parallelization strategy.
        The executor types are selected at runtime based on GIL availability:
        Standard GIL build (Python ≤ 3.12 or GIL-enabled 3.13+):
            * ``ProcessPoolExecutor`` (``eval_pool``) — objective function evaluations.
            Process isolation ensures arbitrary callables (lambdas, closures)
            serialized with ``dill`` run safely without touching shared state.
            * ``ThreadPoolExecutor`` (``search_pool``) — surrogate search tasks.
            Threads share the main-process heap; zero ``dill`` overhead.
            A ``threading.Lock`` (``_surrogate_lock``) prevents a surrogate refit
            from racing with an in-flight search thread.
        Free-threaded build (``python3.13t`` / ``--disable-gil``):
            * Both ``eval_pool`` and ``search_pool`` are ``ThreadPoolExecutor``
                instances.  Threads achieve true CPU-level parallelism without the GIL.
                The ``dill`` serialization step for eval tasks is eliminated — ``fun``
                is called directly from the shared heap.  The ``_surrogate_lock`` is
                still used to serialize surrogate reads and refits.
        Pipeline:
            1.  Parallel Initial Design:
                ``n_initial`` points are dispatched to ``eval_pool``.  Results are
                collected via ``FIRST_COMPLETED`` until all initial evaluations finish.
            2.  First Surrogate Fit:
                Called on the main thread once all initial evaluations are in.
                No lock is needed here because no search threads are active yet.
            3.  Parallel Search (Thread Pool):
                Up to ``n_jobs`` search tasks are submitted to ``search_pool``.
                Each acquires ``_surrogate_lock`` before calling
                ``suggest_next_infill_point()``, serializing concurrent surrogate reads.
            4.  Steady-State Loop with Batch Dispatch:
                - Search completes → candidate appended to ``pending_cands``.
                - When ``len(pending_cands) >= eval_batch_size`` (or no search tasks
                remain), all pending candidates are stacked into ``X_batch`` and
                dispatched as a single eval call to ``eval_pool``.
                On GIL builds this calls ``remote_batch_eval_wrapper`` (dill);
                on free-threaded builds it calls ``fun`` directly in a thread.
                - Batch eval completes → storage updated for every point, surrogate
                refit once under ``_surrogate_lock``, new search slots filled.
                - ``eval_batch_size=1`` (default) dispatches immediately on each
                search completion, preserving the original one-point behavior.
                - This cycle continues until ``max_iter`` evaluations or ``max_time``
                minutes is reached.
        The optimization terminates when either:
        - Total function evaluations reach ``max_iter`` (including initial design), OR
        - Runtime exceeds ``max_time`` minutes

        Args:
            timeout_start (float): Start time for timeout.
            X0 (Optional[np.ndarray]): Initial design points in Natural Space, shape (n_initial, n_features).
            y0_known (Optional[float]): Known best objective value from a previous run.
                When provided together with ``self.x0``, the matching point in the initial
                design is pre-filled with this value and not re-submitted to the worker
                pool, saving one evaluation per restart (restart injection).
            max_iter_override (Optional[int]): Override for maximum number of iterations.

        Raises:
            RuntimeError: If all initial design evaluations fail, likely due to
                pickling issues or missing imports in the worker process.
                The error message provides guidance on how to address this issue.

        Returns:
            Tuple[str, OptimizeResult]: Tuple containing status and optimization result.

        Examples:
            ```{python}
            import time
            from spotoptim import SpotOptim
            from spotoptim.function import sphere
            opt = SpotOptim(
                 fun=sphere,
                 bounds=[(-5, 5), (-5, 5)],
                 n_initial=5,
                 max_iter=10,
                 seed=0,
                 n_jobs=2,
            )
            status, result = opt.optimize_steady_state(timeout_start=time.time(), X0=None)
            print(status)
            print(result.message.splitlines()[0])
            ```
        """
        return _steady.optimize_steady_state(
            self, timeout_start, X0, y0_known, max_iter_override
        )

    # ====================
    # TASK_MO:
    # * store_mo()
    # * mo2so()
    # ====================

    def store_mo(self, y_mo: np.ndarray) -> None:
        """Store multi-objective values in self.y_mo.
        If multi-objective values are present (ndim==2), they are stored in self.y_mo.
        New values are appended to existing ones. For single-objective problems,
        self.y_mo remains None.

        Args:
            y_mo (ndarray): If multi-objective, shape (n_samples, n_objectives).
                           If single-objective, shape (n_samples,).

        Examples:
            ```{python}
            import numpy as np
            from spotoptim import SpotOptim
            opt = SpotOptim(
                fun=lambda X: np.column_stack([
                    np.sum(X**2, axis=1),
                    np.sum((X-1)**2, axis=1)
                ]),
                bounds=[(-5, 5), (-5, 5)],
                max_iter=10,
                n_initial=5
            )
            y_mo_1 = np.array([[1.0, 2.0], [3.0, 4.0]])
            opt.store_mo(y_mo_1)
            print(f"y_mo after first call: {opt.y_mo}")
            y_mo_2 = np.array([[5.0, 6.0], [7.0, 8.0]])
            opt.store_mo(y_mo_2)
            print(f"y_mo after second call: {opt.y_mo}")
            ```
        """
        # Store y_mo in self.y_mo (append new values) if multi-objective
        if self.y_mo is None and y_mo.ndim == 2:
            self.y_mo = y_mo
        elif y_mo.ndim == 2:
            self.y_mo = np.vstack([self.y_mo, y_mo])

    def mo2so(self, y_mo: np.ndarray) -> np.ndarray:
        """Convert multi-objective values to single-objective.
        Converts multi-objective values to a single-objective value by applying a user-defined
        function from `fun_mo2so`. If no user-defined function is given, the
        values in the first objective column are used.

        This method is called after the objective function evaluation. It returns a 1D array
        with the single-objective values.

        Args:
            y_mo (ndarray): If multi-objective, shape (n_samples, n_objectives).
                           If single-objective, shape (n_samples,).

        Returns:
            ndarray: Single-objective values, shape (n_samples,).

        Examples:
            ```{python}
            import numpy as np
            from spotoptim import SpotOptim

            # Multi-objective function
            def mo_fun(X):
                return np.column_stack([
                    np.sum(X**2, axis=1),
                    np.sum((X-1)**2, axis=1)
                ])

            # Example 1: Default behavior (use first objective)
            opt1 = SpotOptim(
                fun=mo_fun,
                bounds=[(-5, 5), (-5, 5)],
                max_iter=10,
                n_initial=5
            )
            y_mo = np.array([[1.0, 2.0], [3.0, 4.0]])
            y_so = opt1.mo2so(y_mo)
            print(f"Single-objective (default): {y_so}")
            ```

            ```{python}
            import numpy as np
            from spotoptim import SpotOptim
            # Example 2: Custom conversion function (sum of objectives)
            def custom_mo2so(y_mo):
                return y_mo[:, 0] + y_mo[:, 1]

            opt2 = SpotOptim(
                fun=mo_fun,
                bounds=[(-5, 5), (-5, 5)],
                max_iter=10,
                n_initial=5,
                fun_mo2so=custom_mo2so
            )
            y_so_custom = opt2.mo2so(y_mo)
            print(f"Single-objective (custom): {y_so_custom}")
            ```
        """
        n, m = self.get_shape(y_mo)
        self.store_mo(y_mo)

        # Use ndim to check if multi-objective
        if y_mo.ndim == 2:
            if self.fun_mo2so is not None:
                # Apply user-defined conversion function
                y0 = self.fun_mo2so(y_mo)
            else:
                # Default: use first column
                if y_mo.size > 0:
                    y0 = y_mo[:, 0]
                else:
                    y0 = y_mo
        else:
            # Single-objective, return as-is
            y0 = y_mo

        return y0

    # ====================
    # TASK_OCBA:
    # * apply_ocba()
    # * get_ranks()
    # * get_ocba()
    # * get_ocba_X()
    # ====================

    def apply_ocba(self) -> Optional[np.ndarray]:
        """Apply Optimal Computing Budget Allocation for noisy functions."""
        return _ocba.apply_ocba(self)

    def get_ranks(self, x: np.ndarray) -> np.ndarray:
        """Returns ranks of numbers within input array x."""
        return _ocba.get_ranks(x)

    def get_ocba(
        self, means: np.ndarray, vars: np.ndarray, delta: int, verbose: bool = False
    ) -> np.ndarray:
        """Optimal Computing Budget Allocation (OCBA)."""
        return _ocba.get_ocba(means, vars, delta, verbose)

    def get_ocba_X(
        self,
        X: np.ndarray,
        means: np.ndarray,
        vars: np.ndarray,
        delta: int,
        verbose: bool = False,
    ) -> np.ndarray:
        """Calculate OCBA allocation and repeat input array X."""
        return _ocba.get_ocba_X(X, means, vars, delta, verbose)

    # ====================
    # TASK_SELECT:
    # * select_new()
    # * suggest_next_infill_point()
    # ====================

    def select_new(
        self, A: np.ndarray, X: np.ndarray, tolerance: float = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select rows from A that are not in X.
        Used in suggest_next_infill_point() to avoid duplicate evaluations.

        Args:
            A (ndarray): Array with new values.
            X (ndarray): Array with known values.
            tolerance (float, optional): Tolerance value for comparison. Defaults to 0.

        Returns:
            tuple: A tuple containing:
                * ndarray: Array with unknown (new) values.
                * ndarray: Array with True if value is new, otherwise False.

        Examples:
            ```{python}
            import numpy as np
            from spotoptim import SpotOptim
            def sphere(X):
                X = np.atleast_2d(X)
                return np.sum(X**2, axis=1)
            opt = SpotOptim(fun=sphere, bounds=[(-5, 5)])
            A = np.array([[1, 2], [3, 4], [5, 6]])
            X = np.array([[3, 4], [7, 8]])
            new_A, is_new = opt.select_new(A, X)
            print("New A:", new_A)
            print("Is new:", is_new)
            ```
        """
        return _acq.select_new(self, A, X, tolerance=tolerance)

    def suggest_next_infill_point(self) -> np.ndarray:
        """Suggest next point to evaluate (dispatcher).
        Used in both sequential and parallel optimization loops. This method orchestrates
        the process of generating candidate points from the acquisition function optimizer,
        handling any failures in the acquisition process with a fallback strategy, and
        ensuring that the returned point(s) are valid and ready for evaluation.
        The returned point is in the Transformed and Mapped Space (Internal Optimization Space).
        This means:
            1. Transformations (e.g., log, sqrt) have been applied.
            2. Dimension reduction has been applied (fixed variables removed).
        Process:
            1. Try candidates from acquisition function optimizer.
            2. Handle acquisition failure (fallback).
            3. Return last attempt if all fails.


        Returns:
            ndarray: Next point(s) to evaluate in Transformed and Mapped Space.
            Shape is (n_infill_points, n_features).

        Examples:
            ```{python}
            import numpy as np
            from spotoptim import SpotOptim
            def sphere(X):
                X = np.atleast_2d(X)
                return np.sum(X**2, axis=1)
            opt = SpotOptim(
                fun=sphere,
                bounds=[(-5, 5), (-5, 5)],
                n_initial=5,
                n_infill_points=2
            )
            # Need to initialize optimization state (X_, y_, surrogate)
            # Normally done inside optimize()
            np.random.seed(0)
            opt.X_ = np.random.rand(10, 2)
            opt.y_ = np.random.rand(10)
            opt.fit_surrogate(opt.X_, opt.y_)
            x_next = opt.suggest_next_infill_point()
            x_next.shape
            ```
        """
        return _acq.suggest_next_infill_point(self)

    # ====================
    # TASK_STATS:
    # * init_storage()
    # * update_storage()
    # * update_stats()
    # * update_success_rate()
    # * get_success_rate()
    # * aggregate_mean_var()
    # * get_best_hyperparameters
    # ====================

    def get_best_hyperparameters(
        self, as_dict: bool = True
    ) -> Union[Dict[str, Any], np.ndarray, None]:
        """
        Get the best hyperparameter configuration found during optimization.
        If noise handling is active (repeats_initial > 1 or OCBA), this returns the parameter
        configuration associated with the best *mean* objective value. Otherwise, it returns
        the configuration associated with the absolute best observed value.

        Args:
            as_dict (bool, optional): If True, returns a dictionary mapping parameter names
                to their values. If False, returns the raw numpy array. Defaults to True.

        Returns:
            Union[Dict[str, Any], np.ndarray, None]: The best hyperparameter configuration.
                Returns None if optimization hasn't started (no data).

        Examples:
            ```{python}
            from spotoptim import SpotOptim
            from spotoptim.function import sphere

            opt = SpotOptim(fun=sphere,
                            bounds=[(-5, 5), (0, 10)],
                            n_initial=5,
                            var_name=["x", "y"],
                            verbose=True)
            opt.optimize()
            best_params = opt.get_best_hyperparameters()
            print(best_params['x']) # Should be close to 0
            ```
        """
        if self.X_ is None or len(self.X_) == 0:
            return None

        # Determine which "best" to use
        if (self.repeats_initial > 1 or self.repeats_surrogate > 1) and hasattr(
            self, "min_mean_X"
        ):
            best_x = self.min_mean_X
        else:
            best_x = self.best_x_

        if not as_dict:
            return best_x

        # Map factors using existing method (handles 2D, returns 2D)
        # We pass best_x as (1, D) and get (1, D) back
        mapped_x = self.map_to_factor_values(best_x.reshape(1, -1))[0]

        # Convert to dictionary with types
        params = {}
        names = (
            self.var_name if self.var_name else [f"p{i}" for i in range(len(best_x))]
        )

        for i, name in enumerate(names):
            val = mapped_x[i]

            # Handle types if available (specifically int, as factors are already mapped)
            if self.var_type:
                v_type = self.var_type[i]
                if v_type == "int":
                    val = int(round(val))

            params[name] = val

        return params

    def init_storage(self, X0: np.ndarray, y0: np.ndarray) -> None:
        """Initialize storage for optimization.
        Sets up the initial data structures needed for optimization tracking:
            * X_: Evaluated design points (in original scale)
            * y_: Function values at evaluated points
            * n_iter_: Iteration counter
        Then updates statistics by calling `update_stats()`.

        Args:
            X0 (ndarray): Initial design points in internal scale, shape (n_samples, n_features).
            y0 (ndarray): Function values at X0, shape (n_samples,).

        Returns:
            None

        Examples:
            ```{python}
            import numpy as np
            from spotoptim import SpotOptim
            opt = SpotOptim(fun=lambda X: np.sum(X**2, axis=1),
                            bounds=[(-5, 5), (-5, 5)],
                            n_initial=5)
            X0 = np.array([[1, 2], [3, 4], [0, 1]])
            y0 = np.array([5.0, 25.0, 1.0])
            opt.init_storage(X0, y0)
            print(f"X_ = {opt.X_}")
            print(f"y_ = {opt.y_}")
            print(f"n_iter_ = {opt.n_iter_}")
            print(f"counter = {opt.counter}")
            ```
        """
        _storage.init_storage(self, X0, y0)

    def update_storage(self, X_new: np.ndarray, y_new: np.ndarray) -> None:
        """Update storage (`X_`, `y_`) with new evaluation points.
        Appends new design points and their function values to the storage arrays.
        Points are converted from internal scale to original scale before storage.

        Args:
            X_new (ndarray): New design points in internal scale, shape (n_new, n_features).
            y_new (ndarray): Function values at X_new, shape (n_new,).

        Returns:
            None

        Examples:
            ```{python}
            import numpy as np
            from spotoptim import SpotOptim
            opt = SpotOptim(fun=lambda X: np.sum(X**2, axis=1),
                            bounds=[(-5, 5), (-5, 5)],
                            n_initial=5)
            # Initialize with some data
            opt.X_ = np.array([[1, 2], [3, 4]])
            opt.y_ = np.array([5.0, 25.0])
            print("Initial storage:")
            print(opt.X_)
            print(opt.y_)
            # Add new points
            X_new = np.array([[0, 1], [2, 3]])
            y_new = np.array([1.0, 13.0])
            opt.update_storage(X_new, y_new)
            print("Updated storage:")
            print(opt.X_)
            print(opt.y_)
            ```
        """
        _storage.update_storage(self, X_new, y_new)

    def update_stats(self) -> None:
        """Update optimization statistics.
        Updates various statistics related to the optimization progress:
            * `min_y`: Minimum y value found so far
            * `min_X`: X value corresponding to minimum y
            * `counter`: Total number of function evaluations

        Notes:
            `success_rate` is updated separately via `update_success_rate()` method, which is called after each batch of function evaluations.

        If "noise" is True (`repeats_initial > 1` or `repeats_surrogate > 1`), additionally computes:
            * `mean_X`: Unique design points (aggregated from repeated evaluations)
            * `mean_y`: Mean y values per design point
            * `var_y`: Variance of y values per design point
            * `min_mean_X`: X value of the best mean y value
            * `min_mean_y`: Best mean y value
            * `min_var_y`: Variance of the best mean y value


        Returns:
            None

        Examples:
            ```{python}
            import numpy as np
            from spotoptim import SpotOptim
            from spotoptim.function import sphere
            # Without noise
            opt = SpotOptim(fun=sphere,
                            bounds=[(-5, 5), (-5, 5)],
                            max_iter=10, n_initial=5)
            opt.optimize()
            print("SpotOptim stats without noise:")
            print(f"opt.X_: {opt.X_}")
            print(f"opt.y_: {opt.y_}")
            print(f"opt.min_y: {opt.min_y}")
            print(f"opt.min_X: {opt.min_X}")
            print(f"opt.counter: {opt.counter}")
            ```

            ```{python}
            import numpy as np
            from spotoptim import SpotOptim
            from spotoptim.function import noisy_sphere
            # With noise
            opt_noise = SpotOptim(fun=noisy_sphere,
                                  bounds=[(-5, 5), (-5, 5)],
                                  n_initial=5,
                                  repeats_surrogate=2,
                                  repeats_initial=2)
            opt_noise.optimize()
            print("SpotOptim stats with noise:")
            print(f"opt_noise.X_: {opt_noise.X_}")
            print(f"opt_noise.y_: {opt_noise.y_}")
            print(f"opt_noise.min_y: {opt_noise.min_y}")
            print(f"opt_noise.min_X: {opt_noise.min_X}")
            print(f"opt_noise.counter: {opt_noise.counter}")
            print(f"opt_noise.mean_X: {opt_noise.mean_X}")
            print(f"opt_noise.mean_y: {opt_noise.mean_y}")
            print(f"opt_noise.var_y: {opt_noise.var_y}")
            print(f"opt_noise.min_mean_X: {opt_noise.min_mean_X}")
            print(f"opt_noise.min_mean_y: {opt_noise.min_mean_y}")
            print(f"opt_noise.min_var_y: {opt_noise.min_var_y}")
            ```
        """
        _storage.update_stats(self)

    def update_success_rate(self, y_new: np.ndarray) -> None:
        """Update the rolling success rate of the optimization process.
        A success is counted only if the new value is better (smaller) than the best
        found y value so far. The success rate is calculated based on the last
        `window_size` successes.
        Important: This method should be called BEFORE updating self.y_ to correctly
        track improvements against the previous best value.

        Args:
            y_new (ndarray): The new function values to consider for the success rate update.

        Examples:
            ```{python}
            import numpy as np
            from spotoptim import SpotOptim
            opt = SpotOptim(fun=lambda X: np.sum(X**2, axis=1),
                            bounds=[(-5, 5), (-5, 5)],
                            max_iter=10, n_initial=5)
            print(opt.success_rate)
            opt.X_ = np.array([[1, 2], [3, 4], [0, 1]])
            opt.y_ = np.array([5.0, 3.0, 2.0])
            opt.update_success_rate(np.array([1.5, 2.5]))
            print(opt.success_rate)
            ```
        """
        _storage.update_success_rate(self, y_new)

    def get_success_rate(self) -> float:
        """Get the current success rate of the optimization process.

        Returns:
            float: The current success rate.

        Examples:
            ```{python}
            from spotoptim import SpotOptim
            opt = SpotOptim(fun=lambda x: x,
                            bounds=[(-5, 5), (-5, 5)])
            print(opt.get_success_rate())
            ```
        """
        return _storage.get_success_rate(self)

    def aggregate_mean_var(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Aggregate X and y values to compute mean and variance per group.
        For repeated evaluations at the same design point, this method computes
        the mean function value and variance (using population variance, ddof=0).

        Args:
            X (ndarray): Design points, shape (n_samples, n_features).
            y (ndarray): Function values, shape (n_samples,).

        Returns:
            tuple: A tuple containing:
                * X_agg (ndarray): Unique design points, shape (n_groups, n_features)
                * y_mean (ndarray): Mean y values per group, shape (n_groups,)
                * y_var (ndarray): Variance of y values per group, shape (n_groups,)

        Examples:
            ```{python}
            import numpy as np
            from spotoptim import SpotOptim
            opt = SpotOptim(fun=lambda X: np.sum(X**2, axis=1),
                            bounds=[(-5, 5), (-5, 5)],
                            repeats_initial=2)
            X = np.array([[1, 2], [3, 4], [1, 2]])
            y = np.array([1, 2, 3])
            X_agg, y_mean, y_var = opt.aggregate_mean_var(X, y)
            print(X_agg.shape)
            print(y_mean)
            print(y_var)
            ```
        """
        return _storage.aggregate_mean_var(self, X, y)

    # ====================
    # TASK_RESULTS:
    # * save_result()
    # * load_result()
    # * save_experiment()
    # * load_experiment()
    # * get_result_filename()
    # * get_experiment_filename()
    # * print_results()
    # * print_best()
    # * get_results_table()
    # * get_design_table()
    # * gen_design_table()
    # * get_importance()
    # * sensitivity_spearman()
    # * get_stars()
    # ====================

    def save_result(
        self,
        filename: Optional[str] = None,
        prefix: str = "result",
        path: Optional[str] = None,
        overwrite: bool = True,
        verbosity: int = 0,
    ) -> None:
        """Save complete optimization results to a pickle file."""
        _serial.save_result(self, filename, prefix, path, overwrite, verbosity)

    @staticmethod
    def load_result(filename: str) -> "SpotOptim":
        """Load complete optimization results from a pickle file."""
        return _serial.load_result(filename)

    def save_experiment(
        self,
        filename: Optional[str] = None,
        prefix: str = "experiment",
        path: Optional[str] = None,
        overwrite: bool = True,
        unpickleables: str = "all",
        verbosity: int = 0,
    ) -> None:
        """Save experiment configuration to a pickle file."""
        _serial.save_experiment(
            self, filename, prefix, path, overwrite, unpickleables, verbosity
        )

    @staticmethod
    def load_experiment(filename: str) -> "SpotOptim":
        """Load experiment configuration from a pickle file."""
        return _serial.load_experiment(filename)

    def get_result_filename(self, prefix: str) -> str:
        """Generate result filename with '_res.pkl' suffix."""
        return _serial.get_result_filename(prefix)

    def get_experiment_filename(self, prefix: str) -> str:
        """Generate experiment filename with '_exp.pkl' suffix."""
        return _serial.get_experiment_filename(prefix)

    def print_results(self, *args: Any, **kwargs: Any) -> None:
        """Alias for print(get_results_table()) for compatibility.
        Prints the table.
        """
        print(self.get_results_table(*args, **kwargs))

    # --- Phase 4 delegations: Reporting & Analysis ---

    def print_best(
        self,
        result: Optional[OptimizeResult] = None,
        transformations: Optional[List[Optional[Callable]]] = None,
        show_name: bool = True,
        precision: int = 4,
    ) -> None:
        """Print the best solution found during optimization.
        This method displays the best hyperparameters and objective value in a
        formatted table. It supports custom transformations for parameters
        (e.g., converting log-scale values back to original scale).

        Args:
            result (OptimizeResult, optional): Optimization result object from optimize().
                If None, uses the stored best values from the optimizer. Defaults to None.
            transformations (list of callable, optional): List of transformation functions
                to apply to each parameter. Each function takes a single value and returns
                the transformed value. Use None for parameters that don't need transformation.
                Length must match number of dimensions. Example: [None, None, lambda x: 10**x]
                to convert the 3rd parameter from log10 scale. Defaults to None.
            show_name (bool, optional): Whether to display variable names. If False,
                uses generic names like 'x0', 'x1', etc. Defaults to True.
            precision (int, optional): Number of decimal places for floating point values.
                Defaults to 4.

        Examples:
            ```{python}
            import numpy as np
            from spotoptim import SpotOptim

            def sphere(X):
                X = np.atleast_2d(X)
                return np.sum(X**2, axis=1)
            opt = SpotOptim(
                fun=sphere,
                bounds=[(-5, 5), (-5, 5)],
                var_name=["x1", "x2"],
                max_iter=10,
                n_initial=5
            )
            result = opt.optimize()
            opt.print_best(result)
            ```
        """
        _results.print_best(
            self,
            result=result,
            transformations=transformations,
            show_name=show_name,
            precision=precision,
        )

    def get_results_table(
        self,
        tablefmt: str = "github",
        precision: int = 4,
        show_importance: bool = False,
    ) -> str:
        """Get a comprehensive table string of optimization results.
        This method generates a formatted table of the search space configuration,
        best values found, and optionally variable importance scores.

        Args:
            tablefmt (str, optional): Table format for tabulate library. Options include:
                'github', 'grid', 'simple', 'plain', 'html', 'latex', etc.
                Defaults to 'github'.
            precision (int, optional): Number of decimal places for float values.
                Defaults to 4.
            show_importance (bool, optional): Whether to include importance scores.
                Importance is calculated as the normalized standard deviation of each
                parameter's effect on the objective. Requires multiple evaluations.
                Defaults to False.

        Returns:
            str: Formatted table string that can be printed or saved.

        Examples:
            ```{python}
            import numpy as np
            from spotoptim import SpotOptim

            # Example 1: Basic usage after optimization
            def sphere(X):
                X = np.atleast_2d(X)
                return np.sum(X**2, axis=1)
            opt = SpotOptim(
                fun=sphere,
                bounds=[(-5, 5), (-5, 5), (-5, 5)],
                var_name=["x1", "x2", "x3"],
                var_type=["float", "float", "float"],
                max_iter=10,
                n_initial=5
            )
            result = opt.optimize()
            table = opt.get_results_table()
            print(table)
            table = opt.get_results_table(show_importance=True)
            print(table)
            ```
        """
        return _results.get_results_table(
            self,
            tablefmt=tablefmt,
            precision=precision,
            show_importance=show_importance,
        )

    def get_design_table(
        self,
        tablefmt: str = "github",
        precision: int = 4,
    ) -> str:
        """Get a table string showing the search space design before optimization.
        This method generates a table displaying the variable names, types, bounds,
        and defaults without requiring an optimization run. Useful for inspecting
        and documenting the search space configuration.

        Args:
            tablefmt (str, optional): Table format for tabulate library.
                Defaults to 'github'.
            precision (int, optional): Number of decimal places for float values.
                Defaults to 4.

        Returns:
            str: Formatted table string.

        Examples:
            ```{python}
            import numpy as np
            from spotoptim import SpotOptim

            def sphere(X):
                X = np.atleast_2d(X)
                return np.sum(X**2, axis=1)

            opt = SpotOptim(
                fun=sphere,
                bounds=[(-5, 5), (-10, 10), (0, 1)],
                var_name=["x1", "x2", "x3"],
                var_type=["float", "int", "float"],
                max_iter=10,
                n_initial=5
            )
            table = opt.get_design_table()
            print(table)
            ```
        """
        return _results.get_design_table(self, tablefmt=tablefmt, precision=precision)

    def gen_design_table(self, precision: int = 4, tablefmt: str = "github") -> str:
        """Generate a table of the design or results.
        If optimization has been run (results available), returns the results table.
        Otherwise, returns the design table (search space configuration).

        Args:
            tablefmt (str, optional): Table format. Defaults to 'github'.
            precision (int, optional): Number of decimal places for float values.
                Defaults to 4.

        Returns:
            str: Formatted table string.

        Examples:
            ```{python}
            import numpy as np
            from spotoptim import SpotOptim

            def sphere(X):
                X = np.atleast_2d(X)
                return np.sum(X**2, axis=1)

            opt = SpotOptim(
                fun=sphere,
                bounds=[(-5, 5), (-10, 10), (0, 1)],
                var_name=["x1", "x2", "x3"],
                var_type=["float", "int", "float"],
                max_iter=10,
                n_initial=5
            )
            table = opt.gen_design_table()
            print(table)
            ```
        """
        if self.best_x_ is not None:
            return self.get_results_table(precision=precision, tablefmt=tablefmt)
        else:
            return self.get_design_table(precision=precision, tablefmt=tablefmt)

    def get_importance(self) -> List[float]:
        """Calculate variable importance scores.
        Importance is computed as the normalized sensitivity of each parameter
        based on the variation in objective values across the evaluated points.
        Higher scores indicate parameters that have more influence on the objective.
        The importance is calculated as:
            1. For each dimension, compute the correlation between parameter values
               and objective values
            2. Normalize to percentage scale (0-100)
            3. Higher values indicate more important parameters

        Returns:
            List[float]: Importance scores for each dimension (0-100 scale).

        Examples:
            ```{python}
            import numpy as np
            from spotoptim import SpotOptim

            def test_func(X):
                # x0 has strong effect, x1 has weak effect
                return 10 * X[:, 0]**2 + 0.1 * X[:, 1]**2

            opt = SpotOptim(
                fun=test_func,
                bounds=[(-5, 5), (-5, 5)],
                var_name=["x0", "x1"],
                max_iter=10,
                n_initial=5,
                seed=42
            )
            result = opt.optimize()
            importance = opt.get_importance()
            print(f"x0 importance: {importance[0]:.2f}")
            print(f"x1 importance: {importance[1]:.2f}")

            # Use table to display importance
            table = opt.get_results_table(show_importance=True)
            print(table)
            ```
        """
        return _analysis.get_importance(self)

    def sensitivity_spearman(self) -> None:
        """Compute and print Spearman correlation between parameters and objective values.
        This method analyzes the sensitivity of the objective function to each
        hyperparameter by computing Spearman rank correlations. For categorical
        (factor) variables, correlation is not computed as they require visual
        inspection instead.
        The method automatically handles different parameter types:
            * Integer/float parameters: Direct correlation with objective values
            * Log-transformed parameters (log10, log, ln): Correlation in log-space
            * Factor (categorical) parameters: Skipped with informative message
        Significance levels:
            * ***: p < 0.001 (highly significant)
            * **: p < 0.01 (significant)
            * *: p < 0.05 (marginally significant)

        Examples:
            ```{python}
            from spotoptim import SpotOptim
            import numpy as np

            def test_func(X):
                # x0 has strong effect, x1 has weak effect
                X = np.atleast_2d(X)
                return 10 * X[:, 0]**2 + 0.1 * X[:, 1]**2

            opt = SpotOptim(
                fun=test_func,
                bounds=[(-5, 5), (-5, 5)],
                var_name=["x0", "x1"],
                max_iter=10,
                n_initial=5,
                seed=42
            )
            opt.optimize()
            opt.sensitivity_spearman()
            ```

        Note:
            Only meaningful after optimize() has been called with sufficient evaluations.
        """
        _analysis.sensitivity_spearman(self)

    def get_stars(self, input_list: list) -> list:
        """Converts a list of values to a list of stars.
        Used to visualize the importance of a variable.
        Thresholds: >99: ***, >75: **, >50: *, >10: .

        Args:
            input_list (list): A list of importance scores (0-100).

        Returns:
            list: A list of star strings.

        Examples:
            ```{python}
            from spotoptim import SpotOptim
            import numpy as np

            def test_func(X):
                return 10 * X[:, 0]**2 + 0.1 * X[:, 1]**2

            opt = SpotOptim(
                fun=test_func,
                bounds=[(-5, 5), (-5, 5)],
                var_name=["x0", "x1"],
                max_iter=10,
                n_initial=5,
                seed=42
            )
            opt.optimize()
            opt.get_stars([100, 75, 50, 10, 0])
            ```
        """
        return _analysis.get_stars(input_list)

    # ====================
    # TASK_TENSORBOARD:
    # * _clen_tensorboard_logs()
    # * _init_tensorboard_writer()
    # * _write_tensorboard_scalars()
    # * _write_tensorboard_hparams()
    # * _close_tensorboard_writer()
    # * init_tensorboard()
    # * _close_and_del_tensorboard_writer()
    # ====================

    def _clean_tensorboard_logs(self) -> None:
        """Clean old TensorBoard log directories from the runs folder."""
        _tb.clean_tensorboard_logs(self)

    def _init_tensorboard_writer(self) -> None:
        """Initialize TensorBoard SummaryWriter if logging is enabled."""
        _tb.init_tensorboard_writer(self)

    def _write_tensorboard_scalars(self) -> None:
        """Write scalar metrics to TensorBoard."""
        _tb.write_tensorboard_scalars(self)

    def _write_tensorboard_hparams(self, X: np.ndarray, y: float) -> None:
        """Write hyperparameters and metric to TensorBoard."""
        _tb.write_tensorboard_hparams(self, X, y)

    def _close_tensorboard_writer(self) -> None:
        """Close TensorBoard writer and cleanup."""
        _tb.close_tensorboard_writer(self)

    def _init_tensorboard(self) -> None:
        """Log initial design to TensorBoard."""
        _tb.init_tensorboard(self)

    def _close_and_del_tensorboard_writer(self) -> None:
        """Close and delete TensorBoard writer to prepare for pickling."""
        _tb.close_and_del_tensorboard_writer(self)

    # ====================
    # TASK_PLOT:
    # * plot_progress()
    # * plot_surrogate()
    # * plot_important_hyperparameter_contour()
    # * _plot_surrogate_with_factors()
    # * plot_importance()
    # * plot_parameter_scatter()
    # ====================

    def plot_progress(
        self,
        show: bool = True,
        log_y: bool = False,
        figsize: Tuple[int, int] = (10, 6),
        ylabel: str = "Objective Value",
        mo: bool = False,
    ) -> None:
        """Plot optimization progress using spotoptim.plot.visualization.plot_progress.

        Args:
            show (bool): Whether to show the plot.
            log_y (bool): Whether to use a logarithmic y-axis.
            figsize (tuple): The size of the plot.
            ylabel (str): The label for the y-axis.
            mo (bool): Whether the optimization is multi-objective.

        Returns:
            None

        Examples:
            ```{python}
            from spotoptim import SpotOptim
            import numpy as np

            def test_func(X):
                return 10 * X[:, 0]**2 + 0.1 * X[:, 1]**2

            opt = SpotOptim(
                fun=test_func,
                bounds=[(-5, 5), (-5, 5)],
                var_name=["x0", "x1"],
                max_iter=10,
                n_initial=5,
                seed=42
            )
            opt.optimize()
            opt.plot_progress()
            ```
        """
        plot_progress(
            self, show=show, log_y=log_y, figsize=figsize, ylabel=ylabel, mo=mo
        )

    def plot_surrogate(
        self,
        i: int = 0,
        j: int = 1,
        show: bool = True,
        alpha: float = 0.8,
        var_name: Optional[List[str]] = None,
        cmap: str = "jet",
        num: int = 100,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        add_points: bool = True,
        grid_visible: bool = True,
        contour_levels: int = 30,
        figsize: Tuple[int, int] = (12, 10),
    ) -> None:
        """Plot the surrogate model for two dimensions.
        Delegates to spotoptim.plot.visualization.plot_surrogate.

        Args:
            i (int): The index of the first dimension.
            j (int): The index of the second dimension.
            show (bool): Whether to show the plot.
            alpha (float): The alpha value for the plot.
            var_name (Optional[List[str]]): The names of the variables.
            cmap (str): The colormap to use.
            num (int): The number of points to use for the plot.
            vmin (Optional[float]): The minimum value for the plot.
            vmax (Optional[float]): The maximum value for the plot.
            add_points (bool): Whether to add points to the plot.
            grid_visible (bool): Whether to show the grid.
            contour_levels (int): The number of contour levels to use.
            figsize (tuple): The size of the plot.

        Returns:
            None

        Examples:
            ```{python}
            from spotoptim import SpotOptim
            import numpy as np

            def test_func(X):
                return 10 * X[:, 0]**2 + 0.1 * X[:, 1]**2

            opt = SpotOptim(
                fun=test_func,
                bounds=[(-5, 5), (-5, 5)],
                var_name=["x0", "x1"],
                max_iter=10,
                n_initial=5,
                seed=42
            )
            opt.optimize()
            opt.plot_surrogate()
            ```
        """
        plot_surrogate(
            self,
            i=i,
            j=j,
            show=show,
            alpha=alpha,
            var_name=var_name,
            cmap=cmap,
            num=num,
            vmin=vmin,
            vmax=vmax,
            add_points=add_points,
            grid_visible=grid_visible,
            contour_levels=contour_levels,
            figsize=figsize,
        )

    def plot_important_hyperparameter_contour(
        self,
        max_imp: int = 3,
        show: bool = True,
        alpha: float = 0.8,
        cmap: str = "jet",
        num: int = 100,
        add_points: bool = True,
        grid_visible: bool = True,
        contour_levels: int = 30,
        figsize: Tuple[int, int] = (12, 10),
    ) -> None:
        """Plot surrogate contours using spotoptim.plot.visualization.plot_important_hyperparameter_contour.

        Args:
            max_imp (int): The maximum number of important hyperparameters to plot.
            show (bool): Whether to show the plot.
            alpha (float): The alpha value for the plot.
            cmap (str): The colormap to use.
            num (int): The number of points to use for the plot.
            add_points (bool): Whether to add points to the plot.
            grid_visible (bool): Whether to show the grid.
            contour_levels (int): The number of contour levels to use.
            figsize (tuple): The size of the plot.

        Returns:
            None

        Examples:
            ```{python}
            from spotoptim import SpotOptim
            import numpy as np

            def test_func(X):
                return 10 * X[:, 0]**2 + 0.1 * X[:, 1]**2

            # 2-D problem: max_imp must not exceed n_dim (2)
            opt = SpotOptim(
                fun=test_func,
                bounds=[(-5, 5), (-5, 5)],
                var_name=["x0", "x1"],
                max_iter=10,
                n_initial=5,
                seed=42
            )
            opt.optimize()
            opt.plot_important_hyperparameter_contour(max_imp=2)
            ```
        """
        plot_important_hyperparameter_contour(
            self,
            max_imp=max_imp,
            show=show,
            alpha=alpha,
            cmap=cmap,
            num=num,
            add_points=add_points,
            grid_visible=grid_visible,
            contour_levels=contour_levels,
            figsize=figsize,
        )

    def _plot_surrogate_with_factors(
        self,
        i: int,
        j: int,
        show: bool = True,
        alpha: float = 0.8,
        cmap: str = "jet",
        num: int = 100,
        add_points: bool = True,
        grid_visible: bool = True,
        contour_levels: int = 30,
        figsize: Tuple[int, int] = (12, 10),
    ) -> None:
        """Delegates to spotoptim.plot.visualization._plot_surrogate_with_factors."""
        _plot_surrogate_with_factors(
            self,
            i=i,
            j=j,
            show=show,
            alpha=alpha,
            cmap=cmap,
            num=num,
            add_points=add_points,
            grid_visible=grid_visible,
            contour_levels=contour_levels,
            figsize=figsize,
        )

    def plot_importance(
        self, threshold: float = 0.0, figsize: Tuple[int, int] = (10, 6)
    ) -> None:
        """Plot variable importance.

        Args:
            threshold (float): Minimum importance percentage to include in plot.
            figsize (tuple): Figure size.

        Returns:
            None

        Examples:
            ```{python}
            from spotoptim import SpotOptim
            import numpy as np

            def test_func(X):
                return 10 * X[:, 0]**2 + 0.1 * X[:, 1]**2

            opt = SpotOptim(
                fun=test_func,
                bounds=[(-5, 5), (-5, 5)],
                var_name=["x0", "x1"],
                max_iter=10,
                n_initial=5,
                seed=42
            )
            opt.optimize()
            opt.plot_importance()
            ```
        """
        importance = self.get_importance()
        names = (
            self.all_var_name
            if self.all_var_name
            else [f"x{i}" for i in range(len(importance))]
        )

        # Filter by threshold
        filtered_data = [(n, i) for n, i in zip(names, importance) if i >= threshold]
        filtered_data.sort(key=lambda x: x[1], reverse=True)

        if not filtered_data:
            print("No variables met the importance threshold.")
            return

        names, values = zip(*filtered_data)

        plt.figure(figsize=figsize)
        y_pos = np.arange(len(names))
        plt.barh(y_pos, values, align="center")
        plt.yticks(y_pos, names)
        plt.xlabel("Importance (%)")
        plt.title("Variable Importance")
        plt.gca().invert_yaxis()  # Best on top
        plt.show()

    def plot_parameter_scatter(
        self,
        result: Optional[OptimizeResult] = None,
        show: bool = True,
        figsize: Tuple[int, int] = (12, 10),
        ylabel: str = "Objective Value",
        cmap: str = "viridis_r",
        show_correlation: bool = False,
        log_y: bool = False,
    ) -> None:
        """Plot parameter distributions showing relationship between each parameter and objective.
        Creates a grid of scatter plots, one for each parameter dimension, showing how
        the objective function value varies with each parameter. The best configuration
        is marked with a red star. Parameters with log-scale transformations (var_trans)
        are automatically displayed on a log x-axis.
        Optionally displays Spearman correlation coefficients in plot titles for
        sensitivity analysis. For factor (categorical) variables, correlation is not
        computed and they are displayed with discrete positions on the x-axis.

        Args:
            result (OptimizeResult, optional): Optimization result containing best parameters.
                If None, uses the best found values from self.best_x_ and self.best_y_.
            show (bool, optional): Whether to display the plot. Defaults to True.
            figsize (tuple, optional): Figure size as (width, height). Defaults to (12, 10).
            ylabel (str, optional): Label for y-axis. Defaults to "Objective Value".
            cmap (str, optional): Colormap for scatter plot. Defaults to "viridis_r".
            show_correlation (bool, optional): Whether to compute and display Spearman
                correlation coefficients in plot titles. Requires scipy. Defaults to False.
            log_y (bool, optional): Whether to use logarithmic scale for y-axis.
                Defaults to False.

        Raises:
            ValueError: If no optimization data is available.

        Examples:
            ```{python}
            import numpy as np
            from spotoptim import SpotOptim
            def objective(X):
                X = np.atleast_2d(X)
                return np.sum(X**2, axis=1)
            opt = SpotOptim(
                fun=objective,
                bounds=[(-5, 5), (-5, 5), (-5, 5), (-5, 5)],
                var_name=["x0", "x1", "x2", "x3"],
                max_iter=10,
                n_initial=5,
                seed=42
            )
            result = opt.optimize()
            # Plot parameter distributions
            opt.plot_parameter_scatter(result)
            # Plot with custom settings
            opt.plot_parameter_scatter(result, cmap="plasma", ylabel="Error")
            ```
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plot_parameter_scatter(). "
                "Install it with: pip install matplotlib"
            )

        # Import scipy if correlation is requested
        if show_correlation:
            try:
                from scipy.stats import spearmanr
            except ImportError:
                raise ImportError(
                    "scipy is required for show_correlation=True. "
                    "Install it with: pip install scipy"
                )

        if self.X_ is None or self.y_ is None or len(self.y_) == 0:
            raise ValueError("No optimization data available. Run optimize() first.")

        # Get best values
        if result is not None:
            best_x = result.x
            best_y = result.fun
        elif self.best_x_ is not None and self.best_y_ is not None:
            best_x = self.best_x_
            best_y = self.best_y_
        else:
            raise ValueError("No best solution available.")

        all_params = self.X_
        history = self.y_

        # Determine grid dimensions
        n_params = all_params.shape[1]
        n_cols = min(4, n_params)
        n_rows = int(np.ceil(n_params / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

        # Make axes always iterable
        if n_params == 1:
            axes = np.array([axes])
        axes_flat = axes.flatten() if n_params > 1 else axes

        for idx in range(n_params):
            ax = axes_flat[idx]
            param_values = all_params[:, idx]
            param_name = self.var_name[idx] if self.var_name else f"x{idx}"
            var_type = self.var_type[idx] if self.var_type else "float"
            var_trans = self.var_trans[idx] if self.var_trans else None

            # Check if this is a factor variable
            is_factor = var_type == "factor"

            # Compute correlation if requested
            corr, p_value = np.nan, np.nan
            if show_correlation and not is_factor:
                try:
                    if var_trans in ["log10", "log", "ln"]:
                        # For log-transformed parameters, correlate in log-space
                        param_values_numeric = param_values.astype(float)
                        valid_mask = (param_values_numeric > 0) & (history > 0)
                        if valid_mask.sum() >= 3:
                            corr, p_value = spearmanr(
                                np.log10(param_values_numeric[valid_mask]),
                                np.log10(history[valid_mask]),
                            )
                    else:
                        # Direct correlation for non-transformed parameters
                        param_values_numeric = param_values.astype(float)
                        corr, p_value = spearmanr(param_values_numeric, history)
                except (ValueError, TypeError):
                    pass  # Keep corr as nan

            # Handle factor variables differently
            if is_factor:
                # Map factor levels to integer positions
                unique_vals = np.unique(param_values)
                positions = {val: i for i, val in enumerate(unique_vals)}
                numeric_vals = np.array([positions[val] for val in param_values])

                # Scatter plot with discrete x positions
                ax.scatter(
                    numeric_vals,
                    history,
                    c=history,
                    cmap=cmap,
                    s=50,
                    alpha=0.7,
                    edgecolors="black",
                    linewidth=0.5,
                )

                # Mark best configuration
                best_val = best_x[idx]
                if best_val not in positions:
                    positions[best_val] = len(positions)
                    unique_vals = np.append(unique_vals, best_val)

                best_pos = positions[best_val]
                ax.scatter(
                    [best_pos],
                    [best_y],
                    color="red",
                    s=200,
                    marker="*",
                    edgecolors="black",
                    linewidth=1.5,
                    label="Best",
                    zorder=5,
                )

                # Set categorical x-axis labels
                ax.set_xticks(range(len(unique_vals)))
                ax.set_xticklabels(unique_vals, rotation=45, ha="right")
            else:
                # Standard scatter plot for numeric variables
                ax.scatter(
                    param_values,
                    history,
                    c=history,
                    cmap=cmap,
                    s=50,
                    alpha=0.7,
                    edgecolors="black",
                    linewidth=0.5,
                )

                # Mark best configuration
                ax.scatter(
                    [best_x[idx]],
                    [best_y],
                    color="red",
                    s=200,
                    marker="*",
                    edgecolors="black",
                    linewidth=1.5,
                    label="Best",
                    zorder=5,
                )

                # Use log scale for parameters with log transformations
                if var_trans in ["log10", "log", "ln"]:
                    ax.set_xscale("log")

            # Set labels
            ax.set_xlabel(param_name, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)

            # Set title with optional correlation
            if show_correlation and not np.isnan(corr):
                ax.set_title(
                    f"{param_name}\nCorr: {corr:.3f} (p={p_value:.3f})", fontsize=11
                )
            elif show_correlation and is_factor:
                ax.set_title(f"{param_name}\n(categorical)", fontsize=11)
            else:
                ax.set_title(f"{param_name} vs {ylabel}", fontsize=12)

            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

            # Use log scale for y-axis if requested
            if log_y:
                ax.set_yscale("log")

        # Hide unused subplots
        for idx in range(n_params, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        plt.tight_layout()

        if show:
            plt.show()

    def _generate_mesh_grid(self, i: int, j: int, num: int = 100):
        # Wrapper for _generate_mesh_grid from visualization module.
        return _generate_mesh_grid(self, i, j, num)

    def _generate_mesh_grid_with_factors(
        self, i: int, j: int, num: int, is_factor_i: bool, is_factor_j: bool
    ):
        # Wrapper for _generate_mesh_grid_with_factors from visualization module.
        return _generate_mesh_grid_with_factors(
            self, i, j, num, is_factor_i, is_factor_j
        )
