import numpy as np
import random
import dill
import re
import torch
from typing import Callable, Optional, Tuple, List, Any, Dict, Union
from scipy.optimize import OptimizeResult, differential_evolution, minimize
from scipy.stats.qmc import LatinHypercube
from scipy.stats import norm
from sklearn.base import BaseEstimator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
import warnings
import matplotlib.pyplot as plt
from numpy import linspace, meshgrid, append
import time
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import shutil
from tabulate import tabulate
from spotoptim.tricands import tricands
from spotoptim.sampling.design import generate_uniform_design


class SpotOptim(BaseEstimator):
    """SPOT optimizer compatible with scipy.optimize interface.

    Args:
        fun (callable): Objective function to minimize. Should accept array of shape (n_samples, n_features).
        bounds (list of tuple): Bounds for each dimension as [(low, high), ...].
        max_iter (int, optional): Maximum number of total function evaluations (including initial design).
            For example, max_iter=30 with n_initial=10 will perform 10 initial evaluations plus
            20 sequential optimization iterations. Defaults to 20.
        n_initial (int, optional): Number of initial design points. Defaults to 10.
        surrogate (object, optional): Surrogate model with scikit-learn interface (fit/predict methods).
            If None, uses a Gaussian Process Regressor with Matern kernel. Default configuration::

                from sklearn.gaussian_process import GaussianProcessRegressor
                from sklearn.gaussian_process.kernels import Matern, ConstantKernel

                kernel = ConstantKernel(1.0, (1e-2, 1e12)) * Matern(length_scale=1.0, length_scale_bounds=(1e-4, 1e2), nu=2.5)
                surrogate = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100)
                surrogate = GaussianProcessRegressor(
                    kernel=kernel,
                    n_restarts_optimizer=10,
                    normalize_y=True,
                    random_state=self.seed,
                )

            Alternative surrogates can be provided, including SpotOptim's Kriging model,
            Random Forests, or any scikit-learn compatible regressor. See Examples section.
            Defaults to None (uses default Gaussian Process configuration).
        acquisition (str, optional): Acquisition function ('ei', 'y', 'pi'). Defaults to 'y'.
        var_type (list of str, optional): Variable types for each dimension. Supported types:
            - 'float': Python floats, continuous optimization (no rounding)
            - 'int': Python int, float values will be rounded to integers
            - 'factor': Unordered categorical data, internally mapped to int values
              (e.g., "red"->0, "green"->1, etc.)
            Defaults to None (which sets all dimensions to 'float').
        var_name (list of str, optional): Variable names for each dimension.
            If None, uses default names ['x0', 'x1', 'x2', ...]. Defaults to None.
        tolerance_x (float, optional): Minimum distance between points. Defaults to np.sqrt(np.spacing(1))
        var_trans (list of str, optional): Variable transformations for each dimension. Supported:
            - 'log': Logarithmic transformation, e.g. "log10" for base-10 log
            - 'sqrt': Square root transformation, "sqrt" for square root
            - None or 'id' or 'None': No transformation
            Defaults to None (no transformations).
        max_time (float, optional): Maximum runtime in minutes. If np.inf (default), no time limit.
            The optimization terminates when either max_iter evaluations are reached OR max_time
            minutes have elapsed, whichever comes first. Defaults to np.inf.
        repeats_initial (int, optional): Number of times to evaluate each initial design point.
            Useful for noisy objective functions. If > 1, noise handling is activated and
            statistics (mean, variance) are tracked. Defaults to 1.
        repeats_surrogate (int, optional): Number of times to evaluate each surrogate-suggested point.
            Useful for noisy objective functions. If > 1, noise handling is activated and
            statistics (mean, variance) are tracked. Defaults to 1.
        ocba_delta (int, optional): Number of additional evaluations to allocate using Optimal Computing
            Budget Allocation (OCBA) when noise handling is active. OCBA determines which existing
            design points should be re-evaluated to best distinguish between alternatives. Only used
            when noise=True (repeats > 1) and ocba_delta > 0. Requires at least 3 design points with
            variance information. Defaults to 0 (no OCBA).
        tensorboard_log (bool, optional): Enable TensorBoard logging. If True, optimization metrics
            and hyperparameters are logged to TensorBoard. View logs by running:
            `tensorboard --logdir=<tensorboard_path>` in a separate terminal. Defaults to False.
        tensorboard_path (str, optional): Path for TensorBoard log files. If None and tensorboard_log
            is True, creates a default path: runs/spotoptim_YYYYMMDD_HHMMSS. Defaults to None.
        tensorboard_clean (bool, optional): If True, removes all old TensorBoard log directories from
            the 'runs' folder before starting optimization. Use with caution as this permanently
            deletes all subdirectories in 'runs'. Defaults to False.
        fun_mo2so (callable, optional): Function to convert multi-objective values to single-objective.
            Takes an array of shape (n_samples, n_objectives) and returns array of shape (n_samples,).
            If None and objective function returns multi-objective values, uses first objective.
            Defaults to None.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        verbose (bool, optional): Print progress information. Defaults to False.
        warnings_filter (str, optional): Filter for warnings. One of "error", "ignore", "always", "all",
            "default", "module", or "once". Defaults to "ignore".
        max_surrogate_points (int, optional): Maximum number of points to use for surrogate model fitting.
            If None, all points are used. If the number of evaluated points exceeds this limit,
            a subset is selected using the selection method. Defaults to None.
        selection_method (str, optional): Method for selecting points when max_surrogate_points is exceeded.
            Options: 'distant' (Select points that are distant from each other via K-means clustering) or
            'best' (Select all points from the cluster with the best mean objective value).
            Defaults to 'distant'.
        acquisition_failure_strategy (str, optional): Strategy for handling acquisition function failures.
            Options: 'random' (space-filling design via Latin Hypercube Sampling)
            Defaults to 'random'.
        penalty (bool, optional): Whether to use penalty for handling NaN/inf values in objective function evaluations.
            Defaults to False.
        penalty_val (float, optional): Penalty value to replace NaN/inf values in objective function evaluations.
            When the objective function returns NaN or inf, these values are replaced with penalty plus
            a small random noise (sampled from N(0, 0.1)) to avoid identical penalty values.
            This allows optimization to continue despite occasional function evaluation failures.
            Defaults to None.
        acquisition_fun_return_size (int, optional): Number of top candidates to return from acquisition function optimization.
            Defaults to 3.
        acquisition_optimizer (str or callable, optional): Optimizer to use for maximizing acquisition function.
            Can be "differential_evolution" (default) or any method name supported by scipy.optimize.minimize
            (e.g., "Nelder-Mead", "L-BFGS-B"). Can also be a callable with signature compatible with
            scipy.optimize.minimize (fun, x0, bounds, ...). Defaults to "differential_evolution".
        x0 (array-like, optional): Starting point for optimization, shape (n_features,).
            If provided, this point will be evaluated first and included in the initial design.
            The point should be within the bounds and will be validated before use.
            Defaults to None (no starting point, uses only LHS design).
        de_x0_prob (float, optional): Probability of using the best point as starting point for differential evolution.
            Defaults to 0.1.
        tricands_fringe (bool, optional): Whether to use the fringe of the design space for the initial design.
            Defaults to False.

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
        warnings_filter (str): Filter for warnings during optimization.
        max_surrogate_points (int or None): Maximum number of points for surrogate fitting.
        selection_method (str): Point selection method.
        acquisition_failure_strategy (str): Strategy for handling acquisition failures ('random').
        noise (bool): True if noise handling is active (repeats > 1).
        mean_X (ndarray or None): Aggregated unique design points (if noise=True).
        mean_y (ndarray or None): Mean y values per design point (if noise=True).
        var_y (ndarray or None): Variance of y values per design point (if noise=True).
        min_mean_X (ndarray or None): X value of best mean y (if noise=True).
        min_mean_y (float or None): Best mean y value (if noise=True).
        min_var_y (float or None): Variance of best mean y (if noise=True).
        de_x0_prob (float): Probability of using the best point as starting point for differential evolution.
        tricands_fringe (bool): Whether to use the fringe of the design space for the initial design.

    Examples:
        >>> import numpy as np
        >>> from spotoptim import SpotOptim
        >>> def objective(X):
        ...     return np.sum(X**2, axis=1)
        ...
        >>> # Example 1: Basic usage (deterministic function)
        >>> bounds = [(-5, 5), (-5, 5)]
        >>> optimizer = SpotOptim(fun=objective, bounds=bounds, max_iter=10, n_initial=5, verbose=True)
        >>> result = optimizer.optimize()
        >>> print("Best x:", result.x)
        >>> print("Best f(x):", result.fun)
        >>>
        >>> # Example 2: With custom variable names
        >>> optimizer = SpotOptim(
        ...     fun=objective,
        ...     bounds=[(-5, 5), (-5, 5)],
        ...     var_name=["param1", "param2"],
        ...     max_iter=10,
        ...     n_initial=5
        ... )
        >>> result = optimizer.optimize()
        >>> optimizer.plot_surrogate()  # Uses custom names in plot labels
        >>>
        >>> # Example 3: Noisy function with repeated evaluations
        >>> def noisy_objective(X):
        ...     import numpy as np
        ...     base = np.sum(X**2, axis=1)
        ...     noise = np.random.normal(0, 0.1, size=base.shape)
        ...     return base + noise
        ...
        >>> optimizer = SpotOptim(
        ...     fun=noisy_objective,
        ...     bounds=[(-5, 5), (-5, 5)],
        ...     max_iter=30,
        ...     n_initial=10,
        ...     repeats_initial=3,      # Evaluate each initial point 3 times
        ...     repeats_surrogate=2,    # Evaluate each new point 2 times
        ...     seed=42,                # For reproducibility
        ...     verbose=True
        ... )
        >>> result = optimizer.optimize()
        >>> # Access noise statistics
        >>> print("Unique design points:", optimizer.mean_X.shape[0])
        >>> print("Best mean value:", optimizer.min_mean_y)
        >>> print("Variance at best point:", optimizer.min_var_y)
        >>>
        >>> # Example 4: Noisy function with OCBA (Optimal Computing Budget Allocation)
        >>> optimizer_ocba = SpotOptim(
        ...     fun=noisy_objective,
        ...     bounds=[(-5, 5), (-5, 5)],
        ...     max_iter=50,
        ...     n_initial=10,
        ...     repeats_initial=2,      # Initial repeats
        ...     repeats_surrogate=1,    # Surrogate repeats
        ...     ocba_delta=3,           # Allocate 3 additional evaluations per iteration
        ...     seed=42,
        ...     verbose=True
        ... )
        >>> result = optimizer_ocba.optimize()
        >>> # OCBA intelligently re-evaluates promising points to reduce uncertainty
        >>> print("Total evaluations:", result.nfev)
        >>> print("Unique design points:", optimizer_ocba.mean_X.shape[0])
        >>> print("Best mean value:", optimizer.min_mean_y)
        >>> print("Variance at best point:", optimizer.min_var_y)
        >>>
        >>> # Example 5: With TensorBoard logging
        >>> optimizer_tb = SpotOptim(
        ...     fun=objective,
        ...     bounds=[(-5, 5), (-5, 5)],
        ...     max_iter=30,
        ...     n_initial=10,
        ...     tensorboard_log=True,   # Enable TensorBoard
        ...     tensorboard_path="runs/my_optimization",  # Optional custom path
        ...     verbose=True
        ... )
        >>> result = optimizer_tb.optimize()
        >>> # View logs in browser: tensorboard --logdir=runs/my_optimization
        >>> print("Logs saved to:", optimizer_tb.tensorboard_path)
        >>>
        >>> # Example 6: Using SpotOptim's Kriging surrogate
        >>> from spotoptim.surrogate import Kriging
        >>> kriging_model = Kriging(
        ...     noise=1e-10,           # Regularization parameter
        ...     kernel='gauss',         # Gaussian/RBF kernel
        ...     min_theta=-3.0,         # Min log10(theta) bound
        ...     max_theta=2.0,          # Max log10(theta) bound
        ...     seed=42
        ... )
        >>> optimizer_kriging = SpotOptim(
        ...     fun=objective,
        ...     bounds=[(-5, 5), (-5, 5)],
        ...     surrogate=kriging_model,
        ...     max_iter=30,
        ...     n_initial=10,
        ...     seed=42,
        ...     verbose=True
        ... )
        >>> result = optimizer_kriging.optimize()
        >>> print("Best solution found:", result.x)
        >>> print("Best value:", result.fun)
        >>>
        >>> # Example 7: Using sklearn Gaussian Process with custom kernel
        >>> from sklearn.gaussian_process import GaussianProcessRegressor
        >>> from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
        >>> # Custom kernel: constant * RBF + white noise
        >>> custom_kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(
        ...     length_scale=1.0, length_scale_bounds=(1e-1, 10.0)
        ... ) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
        >>> gp_custom = GaussianProcessRegressor(
        ...     kernel=custom_kernel,
        ...     n_restarts_optimizer=15,
        ...     normalize_y=True,
        ...     random_state=42
        ... )
        >>> optimizer_custom_gp = SpotOptim(
        ...     fun=objective,
        ...     bounds=[(-5, 5), (-5, 5)],
        ...     surrogate=gp_custom,
        ...     max_iter=30,
        ...     n_initial=10,
        ...     seed=42
        ... )
        >>> result = optimizer_custom_gp.optimize()
        >>>
        >>> # Example 8: Using Random Forest as surrogate
        >>> from sklearn.ensemble import RandomForestRegressor
        >>> rf_model = RandomForestRegressor(
        ...     n_estimators=100,
        ...     max_depth=10,
        ...     random_state=42
        ... )
        >>> optimizer_rf = SpotOptim(
        ...     fun=objective,
        ...     bounds=[(-5, 5), (-5, 5)],
        ...     surrogate=rf_model,
        ...     max_iter=30,
        ...     n_initial=10,
        ...     seed=42
        ... )
        >>> result = optimizer_rf.optimize()
        >>> # Note: Random Forests don't provide uncertainty estimates,
        >>> # so Expected Improvement (EI) may be less effective.
        >>> # Consider using acquisition='y' for pure exploitation.
        >>>
        >>> # Example 9: Comparing different kernels for Gaussian Process
        >>> from sklearn.gaussian_process.kernels import Matern, RationalQuadratic
        >>> # Matern kernel with nu=1.5 (once differentiable)
        >>> kernel_matern15 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5)
        >>> gp_matern15 = GaussianProcessRegressor(kernel=kernel_matern15, normalize_y=True)
        >>>
        >>> # Matern kernel with nu=2.5 (twice differentiable, DEFAULT)
        >>> kernel_matern25 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        >>> gp_matern25 = GaussianProcessRegressor(kernel=kernel_matern25, normalize_y=True)
        >>>
        >>> # RBF kernel (infinitely differentiable, smooth)
        >>> kernel_rbf = ConstantKernel(1.0) * RBF(length_scale=1.0)
        >>> gp_rbf = GaussianProcessRegressor(kernel=kernel_rbf, normalize_y=True)
        >>>
        >>> # Rational Quadratic kernel (mixture of RBF kernels)
        >>> kernel_rq = ConstantKernel(1.0) * RationalQuadratic(length_scale=1.0, alpha=1.0)
        >>> gp_rq = GaussianProcessRegressor(kernel=kernel_rq, normalize_y=True)
        >>>
        >>> # Use any of these as surrogate
        >>> optimizer_rbf = SpotOptim(fun=objective, bounds=[(-5, 5), (-5, 5)],
        ...                           surrogate=gp_rbf, max_iter=30, n_initial=10)
        >>> result = optimizer_rbf.optimize()
    """

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
        warnings_filter: str = "ignore",
        max_surrogate_points: Optional[int] = None,
        selection_method: str = "distant",
        acquisition_failure_strategy: str = "random",
        penalty: bool = False,
        penalty_val: Optional[float] = None,
        acquisition_fun_return_size: int = 3,
        acquisition_optimizer: Union[str, Callable] = "differential_evolution",
        x0: Optional[np.ndarray] = None,
        de_x0_prob: float = 0.1,
        tricands_fringe: bool = False,
        prob_de_tricands: float = 0.8,
        args: Tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
    ):

        warnings.filterwarnings(warnings_filter)

        # small value, converted to float
        self.eps = np.sqrt(np.spacing(1))

        if tolerance_x is None:
            self.tolerance_x = self.eps
        else:
            self.tolerance_x = tolerance_x

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

        self.fun = fun
        self.objective_names = getattr(
            fun, "objective_names", getattr(fun, "metrics", None)
        )
        self.bounds = bounds
        self.max_iter = max_iter
        self.n_initial = n_initial
        self.surrogate = surrogate
        self.acquisition = acquisition
        self.var_type = var_type
        self.var_name = var_name
        self.var_trans = var_trans
        self.max_time = max_time
        self.repeats_initial = repeats_initial
        self.repeats_surrogate = repeats_surrogate
        self.ocba_delta = ocba_delta
        self.tensorboard_log = tensorboard_log
        self.tensorboard_path = tensorboard_path
        self.tensorboard_clean = tensorboard_clean
        self.fun_mo2so = fun_mo2so
        self.seed = seed

        # Initialize persistent RNG
        self.rng = np.random.RandomState(self.seed)

        # Set global seeds if provided
        self._set_seed()

        self.verbose = verbose
        self.max_surrogate_points = max_surrogate_points
        self.selection_method = selection_method
        self.acquisition_failure_strategy = acquisition_failure_strategy
        self.acquisition_fun_return_size = acquisition_fun_return_size
        self.acquisition_optimizer = acquisition_optimizer
        self.penalty = penalty
        self.penalty_val = penalty_val
        self.x0 = x0
        self.de_x0_prob = de_x0_prob
        self.tricands_fringe = tricands_fringe
        self.prob_de_tricands = prob_de_tricands
        self.args = args
        self.kwargs = kwargs if kwargs is not None else {}

        # Determine if noise handling is active
        self.noise = (repeats_initial > 1) or (repeats_surrogate > 1)

        # Process bounds and factor variables
        self._factor_maps = {}  # Maps dimension index to {int: str} mapping
        self._original_bounds = bounds.copy()  # Store original bounds
        self.process_factor_bounds()  # Maps factor bounds to integer indices

        # Derived attribute dimension n_dim
        self.n_dim = len(bounds)

        # Default variable types
        if self.var_type is None:
            self.var_type = self.detect_var_type()

        # Modify bounds based on var_type
        self.modify_bounds_based_on_var_type()

        self.lower = np.array([b[0] for b in self.bounds])
        self.upper = np.array([b[1] for b in self.bounds])

        # Default variable names
        if self.var_name is None:
            self.var_name = [f"x{i}" for i in range(self.n_dim)]

        # Handle default variable transformations
        self.handle_default_var_trans()

        # Apply transformations to bounds (internal representation)
        self._original_lower = self.lower.copy()
        self._original_upper = self.upper.copy()
        self.transform_bounds()

        # Dimension reduction: backup original bounds and identify fixed dimensions
        self._setup_dimension_reduction()

        # Validate and process starting point if provided
        if self.x0 is not None:
            self.x0 = self._validate_x0(self.x0)

        # Initialize surrogate if not provided
        if self.surrogate is None:
            kernel = ConstantKernel(1.0, (1e-2, 1e12)) * Matern(
                length_scale=1.0, length_scale_bounds=(1e-4, 1e2), nu=2.5
            )
            self.surrogate = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=100,
                normalize_y=True,
                random_state=self.seed,
            )

        # Design generator
        self.lhs_sampler = LatinHypercube(d=self.n_dim, seed=self.seed)

        # Storage for results
        self.X_ = None
        self.y_ = None
        self.y_mo = None  # Multi-objective values (if applicable)
        self.best_x_ = None
        self.best_y_ = None
        self.n_iter_ = 0

        # Noise handling attributes (initialized in update_stats if noise=True)
        self.mean_X = None
        self.mean_y = None
        self.var_y = None
        self.min_mean_X = None
        self.min_mean_y = None
        self.min_var_y = None
        self.min_X = None
        self.min_y = None
        self.counter = 0

        # Success rate tracking (similar to Spot class)
        self.success_rate = 0.0
        self.success_counter = 0
        self.window_size = 100
        self._success_history = []

        # Clean old TensorBoard logs if requested
        self._clean_tensorboard_logs()

        # Initialize TensorBoard writer
        self._init_tensorboard_writer()

    def _set_seed(self) -> None:
        """Set global random seeds for reproducibility.

        Sets seeds for:
        - random
        - numpy.random
        - torch (cpu and cuda)

        Only performs actions if self.seed is not None.
        """
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

    def detect_var_type(self) -> list:
        """Auto-detect variable types based on factor mappings.

        Returns:
            list: List of variable types ('factor' or 'float') for each dimension.
                  Dimensions with factor mappings are assigned 'factor', others 'float'.

        Examples:
            >>> from spotoptim import SpotOptim
            >>> spot = SpotOptim(fun=lambda x: x, bounds=[('red', 'green', 'blue'), (0, 10)])
            >>> spot.detect_var_type()
            ['factor', 'float']
        """
        return [
            "factor" if i in self._factor_maps else "float" for i in range(self.n_dim)
        ]

    def modify_bounds_based_on_var_type(self) -> None:
        """Modify bounds based on variable types.

        Adjusts bounds for each dimension according to its var_type:
        - 'int': Ensures bounds are integers (ceiling for lower, floor for upper)
        - 'factor': Bounds already set to (0, n_levels-1) by process_factor_bounds
        - 'float': Explicitly converts bounds to float

        Raises:
            ValueError: If an unsupported var_type is encountered.

        Examples:
            >>> from spotoptim import SpotOptim
            >>> spot = SpotOptim(fun=lambda x: x, bounds=[(0.5, 10.5)], var_type=['int'])
            >>> spot.bounds
            [(1, 10)]
            >>> spot = SpotOptim(fun=lambda x: x, bounds=[(0, 10)], var_type=['float'])
            >>> spot.bounds
            [(0.0, 10.0)]
        """
        for i, vtype in enumerate(self.var_type):
            if vtype == "int":
                # For integer variables, ensure bounds are integers
                # Use Python's int() to convert numpy types to native Python int
                lower = int(np.ceil(self.bounds[i][0]))
                upper = int(np.floor(self.bounds[i][1]))
                self.bounds[i] = (lower, upper)
            elif vtype == "factor":
                # For factor variables, bounds are already set to (0, n_levels-1)
                # Ensure they are Python int, not numpy int64
                lower = int(self.bounds[i][0])
                upper = int(self.bounds[i][1])
                self.bounds[i] = (lower, upper)
            elif vtype == "float":
                # Continuous variable, convert explicitly to float bounds
                # Use Python's float() to convert numpy types to native Python float
                lower = float(self.bounds[i][0])
                upper = float(self.bounds[i][1])
                self.bounds[i] = (lower, upper)
            else:
                raise ValueError(
                    f"Unsupported var_type '{vtype}' at dimension {i}. "
                    f"Supported types are 'float', 'int', 'factor'."
                )

    def handle_default_var_trans(self) -> None:
        """Handle default variable transformations.

        Sets var_trans to a list of None values if not specified, or normalizes
        transformation names by converting 'id', 'None', or None to None.

        Also validates that var_trans length matches the number of dimensions.

        Raises:
            ValueError: If var_trans length doesn't match n_dim.

        Examples:
            >>> from spotoptim import SpotOptim
            >>> # Default behavior - all None
            >>> spot = SpotOptim(fun=lambda x: x, bounds=[(0, 10), (0, 10)])
            >>> spot.var_trans
            [None, None]
            >>>
            >>> # Normalize transformation names
            >>> spot = SpotOptim(fun=lambda x: x, bounds=[(1, 10), (1, 100)],
            ...                  var_trans=['log10', 'id', None, 'None'])
            >>> spot.var_trans
            ['log10', None, None, None]
        """
        # Default variable transformations (None means no transformation)
        if self.var_trans is None:
            self.var_trans = [None] * self.n_dim
        else:
            # Normalize transformation names
            self.var_trans = [
                None if (t is None or t == "id" or t == "None") else t
                for t in self.var_trans
            ]

        # Validate var_trans length
        if len(self.var_trans) != self.n_dim:
            raise ValueError(
                f"Length of var_trans ({len(self.var_trans)}) must match "
                f"number of dimensions ({self.n_dim})"
            )

    def process_factor_bounds(self) -> None:
        """Process bounds to handle factor variables.

        For dimensions with tuple bounds (factor variables), creates internal
        integer mappings and replaces bounds with (0, n_levels-1).

        Stores mappings in self._factor_maps: {dim_idx: {int_val: str_val}}

        Examples:
            >>> from spotoptim import SpotOptim
            >>> spot = SpotOptim(fun=lambda x: x, bounds=[('red', 'green', 'blue'), (0, 10)])
            >>> spot.process_factor_bounds()
            Factor variable at dimension 0:
              Levels: ['red', 'green', 'blue']
              Mapped to integers: 0 to 2
            >>> print(spot.bounds)
            [(0, 2), (0, 10)]
        """
        processed_bounds = []

        for dim_idx, bound in enumerate(self.bounds):
            if isinstance(bound, (tuple, list)) and len(bound) >= 1:
                # Check if this is a factor variable (contains strings)
                if all(isinstance(v, str) for v in bound) and len(bound) > 0:
                    # Factor variable: create integer mapping
                    factor_levels = list(bound)
                    n_levels = len(factor_levels)

                    # Create mapping: {0: "level1", 1: "level2", ...}
                    self._factor_maps[dim_idx] = {
                        i: level for i, level in enumerate(factor_levels)
                    }

                    # Replace with integer bounds (use Python int, not numpy types)
                    processed_bounds.append((int(0), int(n_levels - 1)))

                    if self.verbose:
                        print(f"Factor variable at dimension {dim_idx}:")
                        print(f"  Levels: {factor_levels}")
                        print(f"  Mapped to integers: 0 to {n_levels - 1}")
                elif len(bound) == 2 and all(
                    isinstance(v, (int, float, np.integer, np.floating)) for v in bound
                ):
                    # Numeric bound tuple (accepts Python and numpy numeric types)
                    # Always cast to Python float/int
                    low, high = float(bound[0]), float(bound[1])

                    # Convert to int if both are integer-valued
                    if low.is_integer() and high.is_integer():
                        low, high = int(low), int(high)

                    processed_bounds.append((low, high))
                else:
                    raise ValueError(
                        f"Invalid bound at dimension {dim_idx}: {bound}. "
                        f"Expected either (lower, upper) for numeric variables or "
                        f"tuple of strings for factor variables."
                    )
            else:
                raise ValueError(
                    f"Invalid bound at dimension {dim_idx}: {bound}. "
                    f"Expected a tuple/list with at least 1 element."
                )

        # Update bounds with processed values
        self.bounds = processed_bounds

    def transform_value(self, x: float, trans: Optional[str]) -> float:
        """Apply transformation to a single float value.

        Args:
            x: Value to transform
            trans: Transformation name. Can be one of 'id', 'log10', 'log', 'ln', 'sqrt',
                   'exp', 'square', 'cube', 'inv', 'reciprocal', or None.
                   Also supports dynamic strings like 'log(x)', 'sqrt(x)', 'pow(x, p)'.

        Returns:
            Transformed value

        Raises:
            TypeError: If x is not a float.
            ValueError: If an unknown transformation is specified.

        Notes:
            See also inverse_transform_value.

        Examples:
            >>> from spotoptim import SpotOptim
            >>> spot = SpotOptim(fun=lambda x: x, bounds=[(1, 10)])
            >>> spot.transform_value(10, 'log10')
            1.0
            >>> spot.transform_value(100, 'log(x)')
            4.605170185988092
        """
        # Ensure x is a float
        if not isinstance(x, float):
            try:
                x = float(x)
            except (ValueError, TypeError):
                raise TypeError(
                    f"transform_value expects a float, got {type(x).__name__} (value: {x})"
                )
        if trans is None or trans == "id":
            return x
        elif trans == "log10":
            return np.log10(x)
        elif trans == "log" or trans == "ln":
            return np.log(x)
        elif trans == "sqrt":
            return np.sqrt(x)
        elif trans == "exp":
            return np.exp(x)
        elif trans == "square":
            return x**2
        elif trans == "cube":
            return x**3
        elif trans == "inv" or trans == "reciprocal":
            return 1.0 / x

        # Dynamic Transformations
        import re

        if trans == "log(x)":
            return np.log(x)
        if trans == "sqrt(x)":
            return np.sqrt(x)

        m = re.match(r"pow\(x,\s*([0-9.]+)\)", trans)
        if m:
            p = float(m.group(1))
            return x**p

        m = re.match(r"pow\(([0-9.]+),\s*x\)", trans)
        if m:
            base = float(m.group(1))
            return base**x

        m = re.match(r"log\(x,\s*([0-9.]+)\)", trans)
        if m:
            base = float(m.group(1))
            return np.log(x) / np.log(base)

        raise ValueError(f"Unknown transformation: {trans}")

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
            >>> from spotoptim import SpotOptim
            >>> spot = SpotOptim(fun=lambda x: x, bounds=[(1, 10)])
            >>> spot.inverse_transform_value(10, 'log10')
            10.0
            >>> spot.inverse_transform_value(100, 'log(x)')
            10.0
        """
        # Ensure x is a float
        if not isinstance(x, float):
            try:
                x = float(x)
            except (ValueError, TypeError):
                raise TypeError(
                    f"transform_value expects a float, got {type(x).__name__} (value: {x})"
                )
        if trans is None or trans == "id":
            return x
        elif trans == "log10":
            return 10**x
        elif trans == "log" or trans == "ln":
            return np.exp(x)
        elif trans == "sqrt":
            return x**2
        elif trans == "exp":
            return np.log(x)
        elif trans == "square":
            return np.sqrt(x)
        elif trans == "cube":
            return np.power(x, 1.0 / 3.0)
        elif trans == "inv" or trans == "reciprocal":
            return 1.0 / x

        # Dynamic Transformations (Inverses)
        if trans == "log(x)":
            return np.exp(x)
        if trans == "sqrt(x)":
            return x**2

        m = re.match(r"pow\(x,\s*([0-9.]+)\)", trans)
        if m:
            p = float(m.group(1))
            return x ** (1.0 / p)

        m = re.match(r"pow\(([0-9.]+),\s*x\)", trans)
        if m:
            base = float(m.group(1))
            return np.log(x) / np.log(base)

        m = re.match(r"log\(x,\s*([0-9.]+)\)", trans)
        if m:
            base = float(m.group(1))
            return base**x

        raise ValueError(f"Unknown transformation: {trans}")

    def _transform_X(self, X: np.ndarray) -> np.ndarray:
        """Transform parameter array from original to internal scale.

        Converts from **Natural Space** (Original) to **Transformed Space** (Full Dimension).
        Does NOT handle dimension reduction (mapping).

        Args:
            X: Array in **Natural Space**, shape (n_samples, n_features)

        Returns:
            Array in **Transformed Space** (Full Dimension)

        Examples:
            >>> from spotoptim import SpotOptim
            >>> import numpy as np
            >>> spot = SpotOptim(fun=lambda x: x, bounds=[(1, 10)])
            >>> X_orig = np.array([[1], [10], [100]])
            >>> spot._transform_X(X_orig)
            array([[0.        ],
                   [1.        ],
                   [2.        ]])
        """
        X_transformed = X.copy()

        # Handle 1D array
        if X.ndim == 1:
            for i, trans in enumerate(self.var_trans):
                if trans is not None:
                    X_transformed[i] = self.transform_value(X[i], trans)
            return X_transformed

        # Handle 2D array
        for i, trans in enumerate(self.var_trans):
            if trans is not None:
                X_transformed[:, i] = np.array(
                    [self.transform_value(x, trans) for x in X[:, i]]
                )
        return X_transformed

    def _inverse_transform_X(self, X: np.ndarray) -> np.ndarray:
        """Transform parameter array from internal to original scale.

        Converts from **Transformed Space** (Full Dimension) to **Natural Space** (Original).
        Does NOT handle dimension expansion (un-mapping).

        Args:
            X: Array in **Transformed Space**, shape (n_samples, n_features)

        Returns:
            Array in **Natural Space**

        Examples:
            >>> from spotoptim import SpotOptim
            >>> import numpy as np
            >>> spot = SpotOptim(fun=lambda x: x, bounds=[(1, 10)])
            >>> X_trans = np.array([[0], [1], [2]])
            >>> spot._inverse_transform_X(X_trans)
            array([[  1.],
                   [ 10.],
                   [100.]])
        """
        X_original = X.copy()

        # Handle 1D array (single sample)
        if X.ndim == 1:
            for i, trans in enumerate(self.var_trans):
                if trans is not None:
                    # Element-wise transformation for 1D array
                    X_original[i] = self.inverse_transform_value(X[i], trans)
            return X_original

        # Handle 2D array (multiple samples)
        for i, trans in enumerate(self.var_trans):
            if trans is not None:
                X_original[:, i] = np.array(
                    [self.inverse_transform_value(x, trans) for x in X[:, i]]
                )
        return X_original

    def transform_bounds(self) -> None:
        """Transform bounds from original to internal scale.

        Updates `self.bounds` (and `self.lower`, `self.upper`) from **Natural Space**
        to **Transformed Space**.

        Examples:
            >>> from spotoptim import SpotOptim
            >>> spot = SpotOptim(fun=lambda x: x, bounds=[(1, 10), (0.1, 100)])
            >>> spot.var_trans = ['log10', 'sqrt']
            >>> spot.transform_bounds()
            >>> print(spot.bounds)
            [(0.0, 1.0), (0.31622776601683794, 10.0)]

        """
        for i, trans in enumerate(self.var_trans):
            if trans is not None:
                lower_t = self.transform_value(self.lower[i], trans)
                upper_t = self.transform_value(self.upper[i], trans)

                # Handle reversed bounds (e.g., reciprocal transformation)
                if lower_t > upper_t:
                    self.lower[i], self.upper[i] = upper_t, lower_t
                else:
                    self.lower[i], self.upper[i] = lower_t, upper_t

        # Update self.bounds to reflect transformed bounds
        # Convert numpy types to Python native types (int or float based on var_type)
        self.bounds = []
        for i in range(len(self.lower)):
            # Check if var_type has this index (handle mismatched lengths)
            if i < len(self.var_type) and (
                self.var_type[i] == "int" or self.var_type[i] == "factor"
            ):
                self.bounds.append((int(self.lower[i]), int(self.upper[i])))
            else:
                self.bounds.append((float(self.lower[i]), float(self.upper[i])))

    def _setup_dimension_reduction(self) -> None:
        """Set up dimension reduction by identifying fixed dimensions.

        identifies dimensions where lower and upper bounds are equal in **Transformed Space**.
        Reduces `self.bounds`, `self.lower`, `self.upper`, etc., to the **Mapped Space**
        (active variables only).

        The resulting `self.bounds` defines the **Transformed and Mapped Space** used
        for optimization.

        This method identifies variables that are fixed (constant) and excludes them
        from the optimization process. It stores:
        - Original bounds and metadata in `all_*` attributes
        - Boolean mask of fixed dimensions in `ident`
        - Reduced bounds, types, and names for optimization
        - `red_dim` flag indicating if reduction occurred

        Examples:
            >>> from spotoptim import SpotOptim
            >>> spot = SpotOptim(fun=lambda x: x, bounds=[(1, 10), (5, 5), (0, 1)])
            >>> spot._setup_dimension_reduction()
            >>> print("Original lower bounds:", spot.all_lower)
            Original lower bounds: [ 1  5  0]
            >>> print("Original upper bounds:", spot.all_upper)
            Original upper bounds: [10  5  1]
            >>> print("Fixed dimensions mask:", spot.ident)
            Fixed dimensions mask: [False  True False]
            >>> print("Reduced lower bounds:", spot.lower)
            Reduced lower bounds: [1 0]
            >>> print("Reduced upper bounds:", spot.upper)
            Reduced upper bounds: [10  1]
            >>> print("Reduced variable names:", spot.var_name)
            Reduced variable names: ['x0', 'x2']
            >>> print("Is dimension reduction active?", spot.red_dim)
            Is dimension reduction active? True
        """
        # Backup original values
        self.all_lower = self.lower.copy()
        self.all_upper = self.upper.copy()
        self.all_var_type = self.var_type.copy()
        self.all_var_name = self.var_name.copy()
        self.all_var_trans = self.var_trans.copy()

        # Identify fixed dimensions (lower == upper)
        self.ident = (self.upper - self.lower) == 0

        # Check if any dimension is fixed
        self.red_dim = self.ident.any()

        if self.red_dim:
            # Reduce bounds to only varying dimensions
            self.lower = self.lower[~self.ident]
            self.upper = self.upper[~self.ident]

            # Update dimension count
            self.n_dim = self.lower.size

            # Reduce variable types and names
            self.var_type = [
                vtype
                for vtype, fixed in zip(self.all_var_type, self.ident)
                if not fixed
            ]
            self.var_name = [
                vname
                for vname, fixed in zip(self.all_var_name, self.ident)
                if not fixed
            ]

            # Reduce transformations
            self.var_trans = [
                vtrans
                for vtrans, fixed in zip(self.all_var_trans, self.ident)
                if not fixed
            ]

            # Update bounds list for reduced dimensions
            # Convert numpy types to Python native types (int or float based on var_type)
            self.bounds = []
            for i in range(self.n_dim):
                # Check if var_type has this index (handle mismatched lengths)
                if i < len(self.var_type) and (
                    self.var_type[i] == "int" or self.var_type[i] == "factor"
                ):
                    self.bounds.append((int(self.lower[i]), int(self.upper[i])))
                else:
                    self.bounds.append((float(self.lower[i]), float(self.upper[i])))

            # Recreate LHS sampler with reduced dimensions
            self.lhs_sampler = LatinHypercube(d=self.n_dim, seed=self.seed)

    def _validate_x0(self, x0: np.ndarray) -> np.ndarray:
        """Validate and process starting point x0.

        This method checks that x0:
        - Is a numpy array
        - Has the correct number of dimensions
        - Has values within bounds (in original scale)
        - Is properly transformed to internal scale

        Args:
            x0 (array-like): Starting point in original scale

        Returns:
            ndarray: Validated and transformed x0 in internal scale, shape (n_features,)

        Raises:
            ValueError: If x0 is invalid

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> opt = SpotOptim(
            ...     fun=lambda X: np.sum(X**2, axis=1),
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     x0=np.array([1.0, 2.0])
            ... )
            >>> # x0 is validated during initialization
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
            # Apply transformations to x0 (from original to internal scale)
            x0_transformed = self._transform_X(x0.reshape(1, -1)).ravel()
        else:  # 2D case
            for idx, pt in enumerate(x0):
                check_point(pt)
            x0_transformed = self._transform_X(x0)

        # If dimension reduction is active, reduce x0 to non-fixed dimensions
        if self.red_dim:
            x0_transformed = x0_transformed[~self.ident]

        if self.verbose:
            print("Starting point x0 validated and processed successfully.")
            print(f"  Original scale: {x0}")
            print(f"  Internal scale: {x0_transformed}")

        return x0_transformed

    def to_all_dim(self, X_red: np.ndarray) -> np.ndarray:
        """Expand reduced-dimensional points to full-dimensional representation.

        This method restores points from the reduced optimization space to the
        full-dimensional space by inserting fixed values for constant dimensions.

        Args:
            X_red (ndarray): Points in reduced space, shape (n_samples, n_reduced_dims).

        Returns:
            ndarray: Points in full space, shape (n_samples, n_original_dims).

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> # Create problem with one fixed dimension
            >>> opt = SpotOptim(
            ...     fun=lambda X: np.sum(X**2, axis=1),
            ...     bounds=[(-5, 5), (2, 2), (-5, 5)],  # x1 is fixed at 2
            ...     max_iter=1,
            ...     n_initial=3
            ... )
            >>> X_red = np.array([[1.0, 3.0], [2.0, 4.0]])  # Only x0 and x2
            >>> X_full = opt.to_all_dim(X_red)
            >>> X_full.shape
            (2, 3)
            >>> X_full[:, 1]  # Middle dimension should be 2.0
            array([2., 2.])
        """
        if not self.red_dim:
            # No reduction occurred, return as-is
            return X_red

        # Number of samples and full dimensions
        n_samples = X_red.shape[0]
        n_full_dims = len(self.ident)

        # Initialize full-dimensional array
        X_full = np.zeros((n_samples, n_full_dims))

        # Track index in reduced array
        red_idx = 0

        # Fill in values dimension by dimension
        for i in range(n_full_dims):
            if self.ident[i]:
                # Fixed dimension: use stored value
                X_full[:, i] = self.all_lower[i]
            else:
                # Varying dimension: use value from reduced array
                X_full[:, i] = X_red[:, red_idx]
                red_idx += 1

        return X_full

    def to_red_dim(self, X_full: np.ndarray) -> np.ndarray:
        """Reduce full-dimensional points to optimization space.

        This method removes fixed dimensions from full-dimensional points,
        extracting only the varying dimensions used in optimization.

        Args:
            X_full (ndarray): Points in full space, shape (n_samples, n_original_dims).

        Returns:
            ndarray: Points in reduced space, shape (n_samples, n_reduced_dims).

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> # Create problem with one fixed dimension
            >>> opt = SpotOptim(
            ...     fun=lambda X: np.sum(X**2, axis=1),
            ...     bounds=[(-5, 5), (2, 2), (-5, 5)],  # x1 is fixed at 2
            ...     max_iter=1,
            ...     n_initial=3
            ... )
            >>> X_full = np.array([[1.0, 2.0, 3.0], [4.0, 2.0, 5.0]])
            >>> X_red = opt.to_red_dim(X_full)
            >>> X_red.shape
            (2, 2)
            >>> np.array_equal(X_red, np.array([[1.0, 3.0], [4.0, 5.0]]))
            True
        """
        if not self.red_dim:
            # No reduction occurred, return as-is
            return X_full

        # Handle 1D array
        if X_full.ndim == 1:
            return X_full[~self.ident]

        # Select only non-fixed dimensions (2D)
        return X_full[:, ~self.ident]

    def _aggregate_mean_var(
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
                - X_agg (ndarray): Unique design points, shape (n_groups, n_features)
                - y_mean (ndarray): Mean y values per group, shape (n_groups,)
                - y_var (ndarray): Variance of y values per group, shape (n_groups,)

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> opt = SpotOptim(fun=lambda X: np.sum(X**2, axis=1),
            ...                 bounds=[(-5, 5), (-5, 5)],
            ...                 repeats_initial=2)
            >>> X = np.array([[1, 2], [3, 4], [1, 2]])
            >>> y = np.array([1, 2, 3])
            >>> X_agg, y_mean, y_var = opt._aggregate_mean_var(X, y)
            >>> X_agg.shape
            (2, 2)
            >>> y_mean
            array([2., 2.])
            >>> y_var
            array([1., 0.])
        """
        # Input validation
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
            raise ValueError("Invalid input shapes for _aggregate_mean_var")

        if X.shape[0] == 0:
            return np.empty((0, X.shape[1])), np.array([]), np.array([])

        # Find unique rows and group indices
        _, unique_idx, inverse_idx = np.unique(
            X, axis=0, return_index=True, return_inverse=True
        )

        X_agg = X[unique_idx]

        # Calculate mean and variance for each group
        n_groups = len(unique_idx)
        y_mean = np.zeros(n_groups)
        y_var = np.zeros(n_groups)

        for i in range(n_groups):
            group_mask = inverse_idx == i
            group_y = y[group_mask]
            y_mean[i] = np.mean(group_y)
            # Use population variance (ddof=0) for consistency with Spot
            y_var[i] = np.var(group_y, ddof=0)

        return X_agg, y_mean, y_var

    def _init_storage(self, X0: np.ndarray, y0: np.ndarray) -> None:
        """Initialize storage for optimization.

        Sets up the initial data structures needed for optimization tracking:
        - X_: Evaluated design points (in original scale)
        - y_: Function values at evaluated points
        - n_iter_: Iteration counter

        Then updates statistics by calling update_stats().

        Args:
            X0 (ndarray): Initial design points in internal scale, shape (n_samples, n_features).
            y0 (ndarray): Function values at X0, shape (n_samples,).

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> opt = SpotOptim(fun=lambda X: np.sum(X**2, axis=1),
            ...                 bounds=[(-5, 5), (-5, 5)],
            ...                 n_initial=5)
            >>> X0 = np.array([[1, 2], [3, 4], [0, 1]])
            >>> y0 = np.array([5.0, 25.0, 1.0])
            >>> opt._init_storage(X0, y0)
            >>> opt.X_.shape
            (3, 2)
            >>> opt.y_.shape
            (3,)
            >>> opt.n_iter_
            0
            >>> opt.counter
            3
        """
        # Initialize storage (convert to original scale for user-facing storage)
        self.X_ = self._inverse_transform_X(X0.copy())
        self.y_ = y0.copy()
        self.n_iter_ = 0

    def _update_storage(self, X_new: np.ndarray, y_new: np.ndarray) -> None:
        """Update storage with new evaluation points.

        Appends new design points and their function values to the storage arrays.
        Points are converted from internal scale to original scale before storage.

        Args:
            X_new (ndarray): New design points in internal scale, shape (n_new, n_features).
            y_new (ndarray): Function values at X_new, shape (n_new,).

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> opt = SpotOptim(fun=lambda X: np.sum(X**2, axis=1),
            ...                 bounds=[(-5, 5), (-5, 5)],
            ...                 n_initial=5)
            >>> # Initialize with some data
            >>> opt.X_ = np.array([[1, 2], [3, 4]])
            >>> opt.y_ = np.array([5.0, 25.0])
            >>> # Add new points
            >>> X_new = np.array([[0, 1], [2, 3]])
            >>> y_new = np.array([1.0, 13.0])
            >>> opt._update_storage(X_new, y_new)
            >>> opt.X_.shape
            (4, 2)
            >>> opt.y_.shape
            (4,)
        """
        # Update storage (convert to original scale for user-facing storage)
        self.X_ = np.vstack([self.X_, self._inverse_transform_X(X_new)])
        self.y_ = np.append(self.y_, y_new)

    def update_stats(self) -> None:
        """Update optimization statistics.

        Updates:
        1. `min_y`: Minimum y value found so far
        2. `min_X`: X value corresponding to minimum y
        3. `counter`: Total number of function evaluations

        Note: `success_rate` is updated separately via `_update_success_rate()` method,
        which is called after each batch of function evaluations.

        If `noise` is True (repeats > 1), additionally computes:
        1. `mean_X`: Unique design points (aggregated from repeated evaluations)
        2. `mean_y`: Mean y values per design point
        3. `var_y`: Variance of y values per design point
        4. `min_mean_X`: X value of the best mean y value
        5. `min_mean_y`: Best mean y value
        6. `min_var_y`: Variance of the best mean y value

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> # Without noise
            >>> opt = SpotOptim(fun=lambda X: np.sum(X**2, axis=1),
            ...                 bounds=[(-5, 5), (-5, 5)],
            ...                 max_iter=10, n_initial=5)
            >>> opt.X_ = np.array([[1, 2], [3, 4], [0, 1]])
            >>> opt.y_ = np.array([5.0, 25.0, 1.0])
            >>> opt.update_stats()
            >>> opt.min_y
            1.0
            >>> opt.min_X
            array([0, 1])
            >>> opt.counter
            3
            >>>
            >>> # With noise
            >>> opt_noise = SpotOptim(fun=lambda X: np.sum(X**2, axis=1),
            ...                       bounds=[(-5, 5), (-5, 5)],
            ...                       repeats_initial=2)
            >>> opt_noise.X_ = np.array([[1, 2], [1, 2], [3, 4]])
            >>> opt_noise.y_ = np.array([4.0, 6.0, 25.0])
            >>> opt_noise.update_stats()
            >>> opt_noise.min_y
            4.0
            >>> opt_noise.mean_y
            array([ 5., 25.])
            >>> opt_noise.var_y
            array([1., 0.])
        """
        if self.y_ is None or len(self.y_) == 0:
            return

        # Basic stats
        self.min_y = np.min(self.y_)
        self.min_X = self.X_[np.argmin(self.y_)]
        self.counter = len(self.y_)

        # Aggregated stats for noisy functions
        if self.noise:
            self.mean_X, self.mean_y, self.var_y = self._aggregate_mean_var(
                self.X_, self.y_
            )
            # X value of the best mean y value so far
            best_mean_idx = np.argmin(self.mean_y)
            self.min_mean_X = self.mean_X[best_mean_idx]
            # Best mean y value so far
            self.min_mean_y = self.mean_y[best_mean_idx]
            # Variance of the best mean y value so far
            self.min_var_y = self.var_y[best_mean_idx]

    def _update_success_rate(self, y_new: np.ndarray) -> None:
        """Update the rolling success rate of the optimization process.

        A success is counted only if the new value is better (smaller) than the best
        found y value so far. The success rate is calculated based on the last
        `window_size` successes.

        Important: This method should be called BEFORE updating self.y_ to correctly
        track improvements against the previous best value.

        Args:
            y_new (ndarray): The new function values to consider for the success rate update.

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> opt = SpotOptim(fun=lambda X: np.sum(X**2, axis=1),
            ...                 bounds=[(-5, 5), (-5, 5)],
            ...                 max_iter=10, n_initial=5)
            >>> opt.X_ = np.array([[1, 2], [3, 4], [0, 1]])
            >>> opt.y_ = np.array([5.0, 3.0, 2.0])
            >>> opt._update_success_rate(np.array([1.5, 2.5]))
            >>> opt.success_rate > 0
            True
        """
        # Initialize or update the rolling history of successes (1 for success, 0 for failure)
        if not hasattr(self, "_success_history") or self._success_history is None:
            self._success_history = []

        # Get the best y value so far (before adding new evaluations)
        # Since this is called BEFORE updating self.y_, we can safely use min(self.y_)
        if self.y_ is not None and len(self.y_) > 0:
            best_y_before = min(self.y_)
        else:
            # This is the initial design, no previous best
            best_y_before = float("inf")

        successes = []
        current_best = best_y_before

        for val in y_new:
            if val < current_best:
                successes.append(1)
                current_best = val  # Update for next comparison within this batch
            else:
                successes.append(0)

        # Add new successes to the history
        self._success_history.extend(successes)
        # Keep only the last window_size successes
        self._success_history = self._success_history[-self.window_size :]

        # Calculate the rolling success rate
        window_size = len(self._success_history)
        num_successes = sum(self._success_history)
        self.success_rate = num_successes / window_size if window_size > 0 else 0.0

    def _get_success_rate(self) -> float:
        """Get the current success rate of the optimization process.

        Returns:
            float: The current success rate.

        Examples:
            >>> from spotoptim import SpotOptim
            >>> opt = SpotOptim(fun=lambda x: x,
            ...                 bounds=[(-5, 5), (-5, 5)])
            >>> print(opt._get_success_rate())
            0.0
        """
        return float(getattr(self, "success_rate", 0.0) or 0.0)

    def _clean_tensorboard_logs(self) -> None:
        """Clean old TensorBoard log directories from the runs folder.

        Removes all subdirectories in the 'runs' directory if tensorboard_clean is True.
        This is useful for removing old logs before starting a new optimization run.

        Warning:
            This will permanently delete all subdirectories in the 'runs' folder.
            Use with caution.

        Examples:
            >>> from spotoptim import SpotOptim
            >>> opt = SpotOptim(
            ...     fun=lambda X: np.sum(X**2, axis=1),
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     tensorboard_log=True,
            ...     tensorboard_clean=True
            ... )
            >>> # Old logs in 'runs' will be removed before optimization starts
        """
        if self.tensorboard_clean:
            runs_dir = "runs"
            if os.path.exists(runs_dir) and os.path.isdir(runs_dir):
                # Get all subdirectories in runs
                subdirs = [
                    os.path.join(runs_dir, d)
                    for d in os.listdir(runs_dir)
                    if os.path.isdir(os.path.join(runs_dir, d))
                ]

                if subdirs:
                    removed_count = 0
                    for subdir in subdirs:
                        try:
                            shutil.rmtree(subdir)
                            removed_count += 1
                            if self.verbose:
                                print(f"Removed old TensorBoard logs: {subdir}")
                        except Exception as e:
                            if self.verbose:
                                print(f"Warning: Could not remove {subdir}: {e}")

                    if self.verbose and removed_count > 0:
                        print(
                            f"Cleaned {removed_count} old TensorBoard log director{'y' if removed_count == 1 else 'ies'}"
                        )
                elif self.verbose:
                    print("No old TensorBoard logs to clean in 'runs' directory")
            elif self.verbose:
                print("'runs' directory does not exist, nothing to clean")

    def _init_tensorboard_writer(self) -> None:
        """Initialize TensorBoard SummaryWriter if logging is enabled.

        Creates a unique log directory based on timestamp if tensorboard_log is True.
        The log directory will be in the format: runs/spotoptim_YYYYMMDD_HHMMSS

        Examples:
            >>> from spotoptim import SpotOptim
            >>> opt = SpotOptim(
            ...     fun=lambda X: np.sum(X**2, axis=1),
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     tensorboard_log=True
            ... )
            >>> hasattr(opt, 'tb_writer')
            True
        """
        if self.tensorboard_log:
            if self.tensorboard_path is None:
                # Create default path with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.tensorboard_path = f"runs/spotoptim_{timestamp}"

            # Create directory if it doesn't exist
            os.makedirs(self.tensorboard_path, exist_ok=True)

            self.tb_writer = SummaryWriter(log_dir=self.tensorboard_path)
            if self.verbose:
                print(f"TensorBoard logging enabled: {self.tensorboard_path}")
        else:
            self.tb_writer = None
            if self.verbose:
                print("TensorBoard logging disabled")

    def _write_tensorboard_scalars(self) -> None:
        """Write scalar metrics to TensorBoard.

        Logs the following metrics:
        - Best y value found so far (min_y)
        - Last y value evaluated
        - Best X coordinates (for each dimension)
        - If noise=True: also logs mean values and variance
        """
        if self.tb_writer is None or self.y_ is None or len(self.y_) == 0:
            return

        step = self.counter
        y_last = self.y_[-1]

        if not self.noise:
            # Non-noisy optimization
            self.tb_writer.add_scalars(
                "y_values", {"min": self.min_y, "last": y_last}, step
            )
            # Log success rate
            self.tb_writer.add_scalar("success_rate", self.success_rate, step)
            # Log best X coordinates using var_name if available
            for i in range(self.n_dim):
                param_name = self.var_name[i] if self.var_name else f"x{i}"
                self.tb_writer.add_scalar(f"X_best/{param_name}", self.min_X[i], step)
        else:
            # Noisy optimization
            self.tb_writer.add_scalars(
                "y_values",
                {"min": self.min_y, "mean_best": self.min_mean_y, "last": y_last},
                step,
            )
            # Log variance of best mean
            self.tb_writer.add_scalar("y_variance_at_best", self.min_var_y, step)
            # Log success rate
            self.tb_writer.add_scalar("success_rate", self.success_rate, step)

            # Log best X coordinates (by mean) using var_name if available
            for i in range(self.n_dim):
                param_name = self.var_name[i] if self.var_name else f"x{i}"
                self.tb_writer.add_scalar(
                    f"X_mean_best/{param_name}", self.min_mean_X[i], step
                )

        self.tb_writer.flush()

    def _write_tensorboard_hparams(self, X: np.ndarray, y: float) -> None:
        """Write hyperparameters and metric to TensorBoard.

        Args:
            X (ndarray): Design point coordinates, shape (n_features,)
            y (float): Function value at X
        """
        if self.tb_writer is None:
            return

        # Create hyperparameter dict with variable names
        hparam_dict = {self.var_name[i]: float(X[i]) for i in range(self.n_dim)}
        metric_dict = {"hp_metric": float(y)}

        self.tb_writer.add_hparams(hparam_dict, metric_dict)
        self.tb_writer.flush()

    def _close_tensorboard_writer(self) -> None:
        """Close TensorBoard writer and cleanup."""
        if hasattr(self, "tb_writer") and self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()
            if self.verbose:
                print(
                    f"TensorBoard writer closed. View logs with: tensorboard --logdir={self.tensorboard_path}"
                )
            del self.tb_writer

    def _init_tensorboard(self) -> None:
        """Log initial design to TensorBoard.

        Logs all initial design points (hyperparameters and function values)
        and scalar metrics to TensorBoard. Only executes if TensorBoard logging
        is enabled (tb_writer is not None).

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> opt = SpotOptim(
            ...     fun=lambda X: np.sum(X**2, axis=1),
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     n_initial=5,
            ...     tensorboard_log=True,
            ...     verbose=False
            ... )
            >>> # Simulate initial design (normally done in optimize())
            >>> opt.X_ = np.array([[1, 2], [0, 0], [2, 1]])
            >>> opt.y_ = np.array([5.0, 0.0, 5.0])
            >>> opt._init_tensorboard()
            >>> # TensorBoard logs created for all initial points
        """
        if self.tb_writer is not None:
            for i in range(len(self.y_)):
                self._write_tensorboard_hparams(self.X_[i], self.y_[i])
            self._write_tensorboard_scalars()

    def _get_shape(self, y: np.ndarray) -> Tuple[int, Optional[int]]:
        """Get the shape of the objective function output.

        Args:
            y (ndarray): Objective function output, shape (n_samples,) or (n_samples, n_objectives).

        Returns:
            tuple: (n_samples, n_objectives) where n_objectives is None for single-objective.

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> opt = SpotOptim(
            ...     fun=lambda X: np.sum(X**2, axis=1),
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     max_iter=10,
            ...     n_initial=5
            ... )
            >>> y_single = np.array([1.0, 2.0, 3.0])
            >>> n, m = opt._get_shape(y_single)
            >>> print(f"n={n}, m={m}")
            n=3, m=None
            >>> y_multi = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            >>> n, m = opt._get_shape(y_multi)
            >>> print(f"n={n}, m={m}")
            n=3, m=2
        """
        if y.ndim == 1:
            return y.shape[0], None
        elif y.ndim == 2:
            return y.shape[0], y.shape[1]
        else:
            # For higher dimensions, flatten to 1D
            return y.size, None

    def _store_mo(self, y_mo: np.ndarray) -> None:
        """Store multi-objective values in self.y_mo.

        If multi-objective values are present (ndim==2), they are stored in self.y_mo.
        New values are appended to existing ones. For single-objective problems,
        self.y_mo remains None.

        Args:
            y_mo (ndarray): If multi-objective, shape (n_samples, n_objectives).
                           If single-objective, shape (n_samples,).

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> opt = SpotOptim(
            ...     fun=lambda X: np.column_stack([
            ...         np.sum(X**2, axis=1),
            ...         np.sum((X-1)**2, axis=1)
            ...     ]),
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     max_iter=10,
            ...     n_initial=5
            ... )
            >>> y_mo_1 = np.array([[1.0, 2.0], [3.0, 4.0]])
            >>> opt._store_mo(y_mo_1)
            >>> print(f"y_mo after first call: {opt.y_mo}")
            y_mo after first call: [[1. 2.]
             [3. 4.]]
            >>> y_mo_2 = np.array([[5.0, 6.0], [7.0, 8.0]])
            >>> opt._store_mo(y_mo_2)
            >>> print(f"y_mo after second call: {opt.y_mo}")
            y_mo after second call: [[1. 2.]
             [3. 4.]
             [5. 6.]
             [7. 8.]]
        """
        # Store y_mo in self.y_mo (append new values) if multi-objective
        if self.y_mo is None and y_mo.ndim == 2:
            self.y_mo = y_mo
        elif y_mo.ndim == 2:
            self.y_mo = np.vstack([self.y_mo, y_mo])

    def _mo2so(self, y_mo: np.ndarray) -> np.ndarray:
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
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> # Multi-objective function
            >>> def mo_fun(X):
            ...     return np.column_stack([
            ...         np.sum(X**2, axis=1),
            ...         np.sum((X-1)**2, axis=1)
            ...     ])
            >>>
            >>> # Example 1: Default behavior (use first objective)
            >>> opt1 = SpotOptim(
            ...     fun=mo_fun,
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     max_iter=10,
            ...     n_initial=5
            ... )
            >>> y_mo = np.array([[1.0, 2.0], [3.0, 4.0]])
            >>> y_so = opt1._mo2so(y_mo)
            >>> print(f"Single-objective (default): {y_so}")
            Single-objective (default): [1. 3.]
            >>>
            >>> # Example 2: Custom conversion function (sum of objectives)
            >>> def custom_mo2so(y_mo):
            ...     return y_mo[:, 0] + y_mo[:, 1]
            >>>
            >>> opt2 = SpotOptim(
            ...     fun=mo_fun,
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     max_iter=10,
            ...     n_initial=5,
            ...     fun_mo2so=custom_mo2so
            ... )
            >>> y_so_custom = opt2._mo2so(y_mo)
            >>> print(f"Single-objective (custom): {y_so_custom}")
            Single-objective (custom): [3. 7.]
        """
        n, m = self._get_shape(y_mo)
        self._store_mo(y_mo)

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

    def _get_ranks(self, x: np.ndarray) -> np.ndarray:
        """Returns ranks of numbers within input array x.

        Args:
            x (ndarray): Input array.

        Returns:
            ndarray: Ranks array where ranks[i] is the rank of x[i].

        Examples:
            >>> opt = SpotOptim(fun=lambda X: np.sum(X**2, axis=1), bounds=[(-5, 5)])
            >>> opt._get_ranks(np.array([2, 1]))
            array([1, 0])
            >>> opt._get_ranks(np.array([20, 10, 100]))
            array([1, 0, 2])
        """
        ts = x.argsort()
        ranks = np.empty_like(ts)
        ranks[ts] = np.arange(len(x))
        return ranks

    def _get_ocba(
        self, means: np.ndarray, vars: np.ndarray, delta: int, verbose: bool = False
    ) -> np.ndarray:
        """Optimal Computing Budget Allocation (OCBA).

        Calculates budget recommendations for given means, variances, and incremental
        budget using the OCBA algorithm.

        References:
            [1] Chun-Hung Chen and Loo Hay Lee: Stochastic Simulation Optimization:
                An Optimal Computer Budget Allocation, pp. 49 and pp. 215

        Args:
            means (ndarray): Array of means.
            vars (ndarray): Array of variances.
            delta (int): Incremental budget.
            verbose (bool): If True, print debug information. Defaults to False.

        Returns:
            ndarray: Array of budget recommendations, or None if conditions not met.

        Examples:
            >>> opt = SpotOptim(fun=lambda X: np.sum(X**2, axis=1), bounds=[(-5, 5)])
            >>> means = np.array([1, 2, 3, 4, 5])
            >>> vars = np.array([1, 1, 9, 9, 4])
            >>> allocations = opt._get_ocba(means, vars, 50)
            >>> allocations
            array([11,  9, 19,  9,  2])
        """
        if np.all(vars > 0) and (means.shape[0] > 2):
            n_designs = means.shape[0]
            allocations = np.zeros(n_designs, np.int32)
            ratios = np.zeros(n_designs, np.float64)
            budget = delta
            ranks = self._get_ranks(means)
            best, second_best = np.argpartition(ranks, 2)[:2]
            ratios[second_best] = 1.0
            select = [i for i in range(n_designs) if i not in [best, second_best]]
            temp = (means[best] - means[second_best]) / (means[best] - means[select])
            ratios[select] = np.square(temp) * (vars[select] / vars[second_best])
            select = [i for i in range(n_designs) if i not in [best]]
            temp = (np.square(ratios[select]) / vars[select]).sum()
            ratios[best] = np.sqrt(vars[best] * temp)
            more_runs = np.full(n_designs, True, dtype=bool)
            add_budget = np.zeros(n_designs, dtype=float)
            more_alloc = True

            if verbose:
                print("\nIn _get_ocba():")
                print(f"means: {means}")
                print(f"vars: {vars}")
                print(f"delta: {delta}")
                print(f"n_designs: {n_designs}")
                print(f"Ratios: {ratios}")
                print(f"Best: {best}, Second best: {second_best}")

            while more_alloc:
                more_alloc = False
                ratio_s = (more_runs * ratios).sum()
                add_budget[more_runs] = (budget / ratio_s) * ratios[more_runs]
                add_budget = np.around(add_budget).astype(int)
                mask = add_budget < allocations
                add_budget[mask] = allocations[mask]
                more_runs[mask] = 0

                if mask.sum() > 0:
                    more_alloc = True
                if more_alloc:
                    budget = allocations.sum() + delta
                    budget -= (add_budget * ~more_runs).sum()

            t_budget = add_budget.sum()

            # Adjust the best design to match the exact delta
            # Ensure we don't go below current allocations
            adjustment = allocations.sum() + delta - t_budget
            add_budget[best] = max(allocations[best], add_budget[best] + adjustment)

            return add_budget - allocations
        else:
            return None

    def _get_ocba_X(
        self,
        X: np.ndarray,
        means: np.ndarray,
        vars: np.ndarray,
        delta: int,
        verbose: bool = False,
    ) -> np.ndarray:
        """Calculate OCBA allocation and repeat input array X.
        Used in the optimize() method to generate new design points based on OCBA.

        Args:
            X (ndarray): Input array to be repeated, shape (n_designs, n_features).
            means (ndarray): Array of means for each design.
            vars (ndarray): Array of variances for each design.
            delta (int): Incremental budget.
            verbose (bool): If True, print debug information. Defaults to False.

        Returns:
            ndarray: Repeated array of X based on OCBA allocation, or None if
                     conditions not met.

        Examples:
            >>> opt = SpotOptim(fun=lambda X: np.sum(X**2, axis=1), bounds=[(-5, 5)])
            >>> X = np.array([[1, 2], [4, 5], [7, 8]])
            >>> means = np.array([1.5, 35, 550])
            >>> vars = np.array([0.5, 50, 5000])
            >>> X_new = opt._get_ocba_X(X, means, vars, delta=5, verbose=False)
            >>> X_new.shape[0] == 5  # Should have 5 additional evaluations
            True
        """
        if np.all(vars > 0) and (means.shape[0] > 2):
            o = self._get_ocba(means=means, vars=vars, delta=delta, verbose=verbose)
            return np.repeat(X, o, axis=0)
        else:
            return None

    def _evaluate_function(self, X: np.ndarray) -> np.ndarray:
        """Evaluate objective function at points X.
        Used in the optimize() method to evaluate the objective function.

        **Input Space**: `X` is expected in **Transformed and Mapped Space** (Internal scale, Reduced dimensions).
        **Process**:
        1. Expands `X` to **Transformed Space** (Full dimensions) if dimension reduction is active.
        2. Inverse transforms `X` to **Natural Space** (Original scale).
        3. Evaluates the user function with points in **Natural Space**.

        If dimension reduction is active, expands X to full dimensions before evaluation.
        Supports both single-objective and multi-objective functions. For multi-objective
        functions, converts to single-objective using _mo2so method.

        Args:
            X (ndarray): Points to evaluate in **Transformed and Mapped Space**, shape (n_samples, n_reduced_features).

        Returns:
            ndarray: Function values, shape (n_samples,).

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> # Single-objective function
            >>> opt_so = SpotOptim(
            ...     fun=lambda X: np.sum(X**2, axis=1),
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     max_iter=10,
            ...     n_initial=5
            ... )
            >>> X = np.array([[1.0, 2.0], [3.0, 4.0]])
            >>> y = opt_so._evaluate_function(X)
            >>> print(f"Single-objective output: {y}")
            Single-objective output: [ 5. 25.]
            >>>
            >>> # Multi-objective function (default: use first objective)
            >>> opt_mo = SpotOptim(
            ...     fun=lambda X: np.column_stack([
            ...         np.sum(X**2, axis=1),
            ...         np.sum((X-1)**2, axis=1)
            ...     ]),
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     max_iter=10,
            ...     n_initial=5
            ... )
            >>> y_mo = opt_mo._evaluate_function(X)
            >>> print(f"Multi-objective output (first obj): {y_mo}")
            Multi-objective output (first obj): [ 5. 25.]
        """
        # Ensure X is 2D
        X = np.atleast_2d(X)

        # Expand to full dimensions if needed
        if self.red_dim:
            X = self.to_all_dim(X)

        # Apply inverse transformations to get original scale for function evaluation
        X_original = self._inverse_transform_X(X)

        # Map factor variables to original string values
        X_for_eval = self._map_to_factor_values(X_original)

        # Evaluate function
        y_raw = self.fun(X_for_eval, *self.args, **self.kwargs)

        # Convert to numpy array if needed
        if not isinstance(y_raw, np.ndarray):
            y_raw = np.array([y_raw])

        # Handle multi-objective case
        y = self._mo2so(y_raw)

        # Ensure y is 1D
        if y.ndim > 1:
            y = y.ravel()

        return y

    def _generate_initial_design(self) -> np.ndarray:
        """Generate initial space-filling design using Latin Hypercube Sampling.
        Used in the optimize() method to create the initial set of design points.

        Returns:
            ndarray: Initial design points, shape (n_initial, n_features).

        Examples:
            >>> from spotoptim import SpotOptim
            >>> opt = SpotOptim(fun=lambda X: np.sum(X**2, axis=1),
            ...                 bounds=[(-5, 5), (-5, 5)],
            ...                 n_initial=10)
            >>> X0 = opt._generate_initial_design()
            >>> X0.shape
            (10, 2)
        """
        # Generate samples in [0, 1]^d
        X0_unit = self.lhs_sampler.random(n=self.n_initial)

        # Scale to [lower, upper]
        X0 = self.lower + X0_unit * (self.upper - self.lower)

        return self._repair_non_numeric(X0, self.var_type)

    def _select_distant_points(
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
                - selected_X (ndarray): Selected design points, shape (k, n_features).
                - selected_y (ndarray): Function values at selected points, shape (k,).

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> opt = SpotOptim(fun=lambda X: np.sum(X**2, axis=1),
            ...                 bounds=[(-5, 5), (-5, 5)],
            ...                 max_surrogate_points=5)
            >>> X = np.random.rand(100, 2)
            >>> y = np.random.rand(100)
            >>> X_sel, y_sel = opt._select_distant_points(X, y, 5)
            >>> X_sel.shape
            (5, 2)
        """
        from sklearn.cluster import KMeans

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

    def _select_best_cluster(
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
                - selected_X (ndarray): Selected design points from best cluster, shape (m, n_features).
                - selected_y (ndarray): Function values at selected points, shape (m,).

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> opt = SpotOptim(fun=lambda X: np.sum(X**2, axis=1),
            ...                 bounds=[(-5, 5), (-5, 5)],
            ...                 max_surrogate_points=5,
            ...                 selection_method='best')
            >>> X = np.random.rand(100, 2)
            >>> y = np.random.rand(100)
            >>> X_sel, y_sel = opt._select_best_cluster(X, y, 5)
        """
        from sklearn.cluster import KMeans

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

    def _selection_dispatcher(
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
                - selected_X (ndarray): Selected design points.
                - selected_y (ndarray): Function values at selected points.

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> opt = SpotOptim(fun=lambda X: np.sum(X**2, axis=1),
            ...                 bounds=[(-5, 5), (-5, 5)],
            ...                 max_surrogate_points=5)
            >>> X = np.random.rand(100, 2)
            >>> y = np.random.rand(100)
            >>> X_sel, y_sel = opt._selection_dispatcher(X, y)
            >>> X_sel.shape[0] <= 5
            True
        """
        if self.max_surrogate_points is None:
            return X, y

        if self.selection_method == "distant":
            return self._select_distant_points(X=X, y=y, k=self.max_surrogate_points)
        elif self.selection_method == "best":
            return self._select_best_cluster(X=X, y=y, k=self.max_surrogate_points)
        else:
            # If no valid selection method, return all points
            return X, y

    def _fit_surrogate(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit surrogate model to data.
        Used in optimize() to fit the surrogate model.

        If the number of points exceeds `self.max_surrogate_points`,
        a subset of points is selected using the selection dispatcher.

        Args:
            X (ndarray): Design points, shape (n_samples, n_features).
            y (ndarray): Function values at X, shape (n_samples,).

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> opt = SpotOptim(fun=lambda X: np.sum(X**2, axis=1),
            ...                 bounds=[(-5, 5), (-5, 5)],
            ...                 max_surrogate_points=10)
            >>> X = np.random.rand(50, 2)
            >>> y = np.random.rand(50)
            >>> opt._fit_surrogate(X, y)
            # Show the fitted surrogate model
            >>> print(opt.surrogate)

        """
        X_fit = X
        y_fit = y

        # Select subset if needed
        if (
            self.max_surrogate_points is not None
            and X.shape[0] > self.max_surrogate_points
        ):
            if self.verbose:
                print(
                    f"Selecting subset of {self.max_surrogate_points} points "
                    f"from {X.shape[0]} total points for surrogate fitting."
                )
            X_fit, y_fit = self._selection_dispatcher(X, y)

        self.surrogate.fit(X_fit, y_fit)

    def _fit_scheduler(self) -> None:
        """Fit surrogate model using appropriate data based on noise handling.

        This method selects the appropriate training data for surrogate fitting:
        - For noisy functions (noise=True): Uses mean_X and mean_y (aggregated values)
        - For deterministic functions: Uses X_ and y_ (all evaluated points)

        The data is transformed to internal scale before fitting the surrogate.

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> from sklearn.gaussian_process import GaussianProcessRegressor
            >>> # Deterministic function
            >>> opt = SpotOptim(
            ...     fun=lambda X: np.sum(X**2, axis=1),
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     surrogate=GaussianProcessRegressor(),
            ...     n_initial=5
            ... )
            >>> # Simulate optimization state
            >>> opt.X_ = np.array([[1, 2], [0, 0], [2, 1]])
            >>> opt.y_ = np.array([5.0, 0.0, 5.0])
            >>> opt._fit_scheduler()
            >>> # Surrogate fitted with X_ and y_
            >>>
            >>> # Noisy function
            >>> opt_noise = SpotOptim(
            ...     fun=lambda X: np.sum(X**2, axis=1),
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     surrogate=GaussianProcessRegressor(),
            ...     n_initial=5,
            ...     repeats_initial=3  # Activates noise handling
            ... )
            >>> # Simulate noisy optimization state
            >>> opt_noise.noise = True
            >>> opt_noise.mean_X = np.array([[1, 2], [0, 0]])
            >>> opt_noise.mean_y = np.array([5.0, 0.0])
            >>> opt_noise._fit_scheduler()
            >>> # Surrogate fitted with mean_X and mean_y
        """
        # Fit surrogate (use mean_y if noise, otherwise y_)
        # Transform X to internal scale for surrogate fitting
        if self.noise:
            X_for_surrogate = self._transform_X(self.mean_X)
            self._fit_surrogate(X_for_surrogate, self.mean_y)
        else:
            X_for_surrogate = self._transform_X(self.X_)
            self._fit_surrogate(X_for_surrogate, self.y_)

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
                - ndarray: Array with unknown (new) values.
                - ndarray: Array with True if value is new, otherwise False.

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> opt = SpotOptim(fun=lambda X: np.sum(X**2, axis=1), bounds=[(-5, 5)])
            >>> A = np.array([[1, 2], [3, 4], [5, 6]])
            >>> X = np.array([[3, 4], [7, 8]])
            >>> new_A, is_new = opt.select_new(A, X)
            >>> print("New A:", new_A)
            New A: [[1 2]
             [5 6]]
            >>> print("Is new:", is_new)
            Is new: [ True False  True]
        """
        B = np.abs(A[:, None] - X)
        ind = np.any(np.all(B <= tolerance, axis=2), axis=1)
        return A[~ind], ~ind

    def _map_to_factor_values(self, X: np.ndarray) -> np.ndarray:
        """Map internal integer values to original factor strings.

        For factor variables, converts integer indices back to original string values.
        Other variable types remain unchanged.

        Args:
            X (ndarray): Array with internal numeric values, shape (n_samples, n_features).

        Returns:
            ndarray: Array with factor values as strings where applicable, shape (n_samples, n_features).

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> opt = SpotOptim(
            ...     fun=lambda X: np.sum(X**2, axis=1),
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     var_type=['float', 'factor'],
            ...     var_level=[None, ['red', 'green', 'blue']]
            ... )
            >>> X_internal = np.array([[1.0, 0], [2.0, 1], [3.0, 2]])
            >>> X_mapped = opt._map_to_factor_values(X_internal)
            >>> print(X_mapped)
            [[1.0 'red']
             [2.0 'green']
             [3.0 'blue']]
        """
        if not self._factor_maps:
            # No factor variables
            return X

        X = np.atleast_2d(X)
        # Create object array to hold mixed types (strings and numbers)
        X_mapped = np.empty(X.shape, dtype=object)
        X_mapped[:] = X  # Copy numeric values

        for dim_idx, mapping in self._factor_maps.items():
            # Check if already mapped (strings) or needs mapping (numeric)
            col_values = X[:, dim_idx]

            # If already strings, keep them
            if isinstance(col_values[0], str):
                continue

            # Round to nearest integer and map to string
            int_values = np.round(col_values).astype(int)
            # Clip to valid range
            int_values = np.clip(int_values, 0, len(mapping) - 1)
            # Map to strings
            for i, val in enumerate(int_values):
                X_mapped[i, dim_idx] = mapping[int(val)]

        return X_mapped

    def _repair_non_numeric(self, X: np.ndarray, var_type: List[str]) -> np.ndarray:
        """Round non-numeric values to integers based on variable type.

        This method applies rounding to variables that are not continuous:
        - 'float': No rounding (continuous values)
        - 'int': Rounded to integers
        - 'factor': Rounded to integers (representing categorical values)

        Args:
            X (ndarray): X array with values to potentially round.
            var_type (list of str): List with type information for each dimension.

        Returns:
            ndarray: X array with non-continuous values rounded to integers.

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> opt = SpotOptim(fun=lambda X: np.sum(X**2, axis=1),
            ...                 bounds=[(-5, 5), (-5, 5)],
            ...                 var_type=['int', 'float'])
            >>> X = np.array([[1.2, 2.5], [3.7, 4.1], [5.9, 6.8]])
            >>> X_repaired = opt._repair_non_numeric(X, opt.var_type)
            >>> print(X_repaired)
            [[1. 2.5]
             [4. 4.1]
             [6. 6.8]]
        """
        # Don't round float or num types (continuous values)
        mask = np.isin(var_type, ["float", "float"], invert=True)
        X[:, mask] = np.around(X[:, mask])
        return X

    def _apply_penalty_NA(
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
            sd (float): Standard deviation for random noise added to penalty.
                Default is 0.1.

        Returns:
            ndarray: Array with NaN/inf replaced by penalty_value + random noise.

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> opt = SpotOptim(fun=lambda X: np.sum(X**2, axis=1), bounds=[(-5, 5)])
            >>> y_hist = np.array([1.0, 2.0, 3.0, 5.0])
            >>> y_new = np.array([4.0, np.nan, np.inf])
            >>> y_clean = opt._apply_penalty_NA(y_new, y_history=y_hist)
            >>> np.all(np.isfinite(y_clean))
            True
            >>> # NaN/inf replaced with worst value from history + 3*std + noise
            >>> y_clean[1] > 5.0  # Should be larger than max finite value in history
            True
        """

        # Ensure y is a float array (maps non-convertible values like "error" or None to nan)
        def _safe_float(v):
            try:
                return float(v)
            except (ValueError, TypeError):
                return np.nan

        y_flat = np.array(y).flatten()
        y = np.array([_safe_float(v) for v in y_flat])
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

    def _remove_nan(
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
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> opt = SpotOptim(fun=lambda X: np.sum(X**2, axis=1), bounds=[(-5, 5)])
            >>> X = np.array([[1, 2], [3, 4], [5, 6]])
            >>> y = np.array([1.0, np.nan, np.inf])
            >>> X_clean, y_clean = opt._remove_nan(X, y, stop_on_zero_return=False)
            >>> print("Clean X:", X_clean)
            Clean X: [[1 2]]
            >>> print("Clean y:", y_clean)
            Clean y: [1.]
        """
        # Find finite values
        finite_mask = np.isfinite(y)

        if not np.any(finite_mask):
            msg = "All objective function values are NaN or inf."
            if stop_on_zero_return:
                raise ValueError(msg)
            else:
                if self.verbose:
                    print(f"Warning: {msg} Returning empty arrays.")
                return np.array([]).reshape(0, X.shape[1]), np.array([])

        # Filter out non-finite values
        n_removed = np.sum(~finite_mask)
        if n_removed > 0 and self.verbose:
            print(f"Warning: Removed {n_removed} sample(s) with NaN/inf values")

        return X[finite_mask], y[finite_mask]

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
        if self.acquisition_failure_strategy == "random":
            # Default: random space-filling design (Latin Hypercube Sampling)
            if self.verbose:
                print(
                    "Acquisition failure: Using random space-filling design as fallback."
                )
            x_new_unit = self.lhs_sampler.random(n=1)[0]
            x_new = self.lower + x_new_unit * (self.upper - self.lower)

        return self._repair_non_numeric(x_new.reshape(1, -1), self.var_type)[0]

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
            >>> opt = SpotOptim(
            ...     fun=lambda X: np.sum(X**2, axis=1),
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     surrogate=GaussianProcessRegressor()
            ... )
            >>> X_train = np.array([[0, 0], [1, 1], [2, 2]])
            >>> y_train = np.array([0, 2, 8])
            >>> opt._fit_surrogate(X_train, y_train)
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

    def _acquisition_function(self, x: np.ndarray) -> float:
        """Compute acquisition function value.
        Used in the suggest_next_infill_point() method.

        This implements "Infill Criteria" as described in Forrester et al. (2008),
        Section 3 "Exploring and Exploiting".

        Args:
            x (ndarray): Point to evaluate, shape (n_features,).

        Returns:
            float: Acquisition function value (to be minimized).

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
            >>> opt._fit_surrogate(X_train, y_train)
            >>> x_eval = np.array([1.5, 1.5])
            >>> acq_value = opt._acquisition_function(x_eval)
            >>> print("Acquisition function value:", acq_value)
            Acquisition function value: [some float value]
        """
        x = x.reshape(1, -1)

        if self.acquisition == "y":
            # Predicted mean
            return self.surrogate.predict(x)[0]

        elif self.acquisition == "ei":
            # Expected Improvement
            mu, sigma = self._predict_with_uncertainty(x)
            mu = mu[0]
            sigma = sigma[0]

            if sigma < 1e-10:
                return 0.0

            y_best = np.min(self.y_)
            improvement = y_best - mu
            Z = improvement / sigma
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            return -ei  # Minimize negative EI

        elif self.acquisition == "pi":
            # Probability of Improvement
            mu, sigma = self._predict_with_uncertainty(x)
            mu = mu[0]
            sigma = sigma[0]

            if sigma < 1e-10:
                return 0.0

            y_best = np.min(self.y_)
            Z = (y_best - mu) / sigma
            pi = norm.cdf(Z)
            return -pi  # Minimize negative PI

            raise ValueError(f"Unknown acquisition function: {self.acquisition}")

    def optimize_acquisition_func(self) -> np.ndarray:
        """Optimize the acquisition function to find the next point to evaluate.

        Returns:
            ndarray: The optimized point(s).
                If acquisition_fun_return_size == 1, returns 1D array of shape (n_features,).
                If acquisition_fun_return_size > 1, returns 2D array of shape (N, n_features),
                where N is min(acquisition_fun_return_size, population_size).

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> opt = SpotOptim(
            ...     fun=lambda X: np.sum(X**2, axis=1),
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     acquisition='ei'
            ... )
            >>> X_train = np.array([[0, 0], [1, 1], [2, 2]])
            >>> y_train = np.array([0, 2, 8])
            >>> opt._fit_surrogate(X_train, y_train)
            >>> x_next = opt.optimize_acquisition_func()
            >>> print("Next point to evaluate:", x_next)
            Next point to evaluate: [some float values]
        """
        if self.acquisition_optimizer == "tricands":
            return self._optimize_acquisition_tricands()
        elif self.acquisition_optimizer == "differential_evolution":
            return self._optimize_acquisition_de()
        elif self.acquisition_optimizer == "de_tricands":
            val = self.rng.rand()

            if val < self.prob_de_tricands:
                return self._optimize_acquisition_de()
            else:
                return self._optimize_acquisition_tricands()
        else:
            return self._optimize_acquisition_scipy()

    def _optimize_acquisition_tricands(self) -> np.ndarray:
        """Optimize using geometric infill strategy via triangulation candidates."""
        # Use X_ (all evaluated points) as basis for triangulation
        # If no points yet (e.g. before initial design), fallback to LHS or random
        if not hasattr(self, "X_") or self.X_ is None or len(self.X_) < self.n_dim + 1:
            # Not enough points for valid triangulation (need n >= m + 1)
            # Fallback to random search using existing logic logic in 'else' block or explicit call pass
            # Will fall through to 'else' block which handles generic minimize/random x0
            # BUT 'tricands' isn't a valid minimize method, so we should handle this fallback specifically.
            # Actually, let's just use random sampling here for fallback.

            # Fallback to random search using generate_uniform_design
            # Return size defaults to 1 unless specified
            n_design = max(1, self.acquisition_fun_return_size)
            x0 = generate_uniform_design(self.bounds, n_design, seed=self.rng)

            # If we requested ONLY 1 point, return expected shape (n_dim,) for flatten behavior
            if self.acquisition_fun_return_size <= 1:
                return x0.flatten()
            return x0

        # Generate candidates
        # Default nmax to a reasonable multiple of desired return size, or just large enough
        # tricands handles nmax internally (default 100*m).
        # We pass nmax as max(100*m, acquisition_fun_return_size * 10) to ensure we have enough.
        nmax = max(100 * self.n_dim, self.acquisition_fun_return_size * 50)

        # Wrapper for tricands: Normalize -> Gen Candidates -> Denormalize
        # This handles non-hypercube bounds correctly as tricands assumes [lower, upper]^m box.

        # Normalize X_ to [0, 1] relative to bounds
        X_norm = (self.X_ - self.lower) / (self.upper - self.lower)

        # Generate candidates in [0, 1] space
        X_cands_norm = tricands(
            X_norm, nmax=nmax, lower=0.0, upper=1.0, fringe=self.tricands_fringe
        )

        # Denormalize candidates back to original space
        X_cands = X_cands_norm * (self.upper - self.lower) + self.lower

        # Evaluate acquisition function on all candidates
        # _acquisition_function returns NEGATIVE acquisition values (minimization)
        # We iterate to ensure correct handling of 1D/2D shapes by _acquisition_function
        acq_values = np.array([self._acquisition_function(x) for x in X_cands])

        # Sort indices (smallest is best because of negation)
        sorted_indices = np.argsort(acq_values)

        # Select top n
        top_n = min(self.acquisition_fun_return_size, len(sorted_indices))
        best_indices = sorted_indices[:top_n]
        return X_cands[best_indices]

    def _optimize_acquisition_de(self) -> np.ndarray:
        """Optimize using differential evolution."""
        # Variables to capture population from callback
        population = None
        population_energies = None
        # with probability .5 select best_x_ as x0 or None
        # Determine which "best" to use
        if self.noise and hasattr(self, "min_mean_X"):
            best_x = self.min_mean_X
        else:
            best_x = self.best_x_

        if best_x is not None:
            best_x = self._transform_X(best_x)
            best_X = best_x if self.rng.rand() < self.de_x0_prob else None
        else:
            best_X = None

        def callback(intermediate_result: OptimizeResult):
            nonlocal population, population_energies
            # Capture population if available (requires scipy >= 1.10.0)
            if hasattr(intermediate_result, "population"):
                population = intermediate_result.population
                population_energies = intermediate_result.population_energies

        result = differential_evolution(
            func=self._acquisition_function,
            bounds=self.bounds,
            seed=self.rng,
            maxiter=1000,
            callback=callback,
            x0=best_X,
        )

        if self.acquisition_fun_return_size > 1:
            if population is not None and population_energies is not None:
                # Sort by energy (ascending, since DE minimizes)
                sorted_indices = np.argsort(population_energies)

                # Determine how many to take
                top_n = min(self.acquisition_fun_return_size, len(sorted_indices))

                # First candidate is always the polished result (best)
                candidates = [result.x]

                # Add remaining candidates from population (skipping the best unpolished one which corresponds to result.x)
                if top_n > 1:
                    # Take next (top_n - 1) indices
                    # Start from 1 because 0 is the best unpolished
                    next_indices = sorted_indices[1:top_n]
                    candidates.extend(population[next_indices])

                return np.array(candidates)
            else:
                # Fallback if population not available (e.g. very fast convergence or old scipy)
                # Just return the best point as 2D array
                return result.x.reshape(1, -1)

        return result.x

    def _optimize_acquisition_scipy(self) -> np.ndarray:
        """Optimize using scipy.optimize.minimize interface (default).

        Args:
            None

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
            >>> optimizer._fit_surrogate(X, y)
            >>>
            >>> # Optimize the acquisition function using scipy's minimize
            >>> x_next = optimizer._optimize_acquisition_scipy()
            >>> x_next.shape
            (2,)

        """
        # Use scipy.optimize.minimize interface
        # Generate random x0 within bounds
        low = np.array([b[0] for b in self.bounds])
        high = np.array([b[1] for b in self.bounds])
        # Use persistent RNG
        x0 = self.rng.uniform(low, high)

        if isinstance(self.acquisition_optimizer, str):

            # It's a method name for minimize (e.g. "Nelder-Mead")
            result = minimize(
                fun=self._acquisition_function,
                x0=x0,
                bounds=self.bounds,
                method=self.acquisition_optimizer,
            )
        elif callable(self.acquisition_optimizer):
            # It's a custom callable compatible with minimize
            result = self.acquisition_optimizer(
                fun=self._acquisition_function, x0=x0, bounds=self.bounds
            )
        else:
            raise ValueError(
                f"Unknown acquisition optimizer type: {type(self.acquisition_optimizer)}"
            )

        # Minimize-based optimizers typically return a single point
        # If acquisition_fun_return_size > 1, we map to 2D array but only 1 unique point
        if self.acquisition_fun_return_size > 1:
            return result.x.reshape(1, -1)

        return result.x

    def _try_optimizer_candidates(self) -> Optional[np.ndarray]:
        """Try candidates proposed by the acquisition result optimizer.

        Returns:
            Optional[ndarray]: A unique valid candidate point, or None if all candidates are duplicates.
        """
        # Phase 1: Try candidates from acquisition function optimizer
        # These can be multiple if acquisition_fun_return_size > 1
        x_next_candidates = self.optimize_acquisition_func()

        # Ensure iterable of 1D arrays
        if x_next_candidates.ndim == 1:
            obs_candidates = [x_next_candidates]
        else:
            obs_candidates = [
                x_next_candidates[i] for i in range(x_next_candidates.shape[0])
            ]

        for i, x_next in enumerate(obs_candidates):
            # Apply rounding BEFORE checking tolerance
            x_next_rounded = self._repair_non_numeric(
                x_next.reshape(1, -1), self.var_type
            )[0]

            # Ensure minimum distance to existing points
            x_next_2d = x_next_rounded.reshape(1, -1)
            X_transformed = self._transform_X(self.X_)
            x_new, _ = self.select_new(
                A=x_next_2d, X=X_transformed, tolerance=self.tolerance_x
            )

            if x_new.shape[0] > 0:
                # Found a unique point!
                return x_next_rounded
            elif self.verbose:
                print(
                    f"Optimizer candidate {i+1}/{len(obs_candidates)} was duplicate after rounding."
                )
        return None

    def _try_fallback_strategy(
        self, max_attempts: int = 10
    ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """Try fallback strategy (e.g. random search) to find a unique point.
        Calls _handle_acquisition_failure.

        Args:
            max_attempts (int): Maximum number of fallback attempts.

        Returns:
            Tuple[Optional[ndarray], ndarray]:
                - The first element is the unique valid candidate point if found, else None.
                - The second element is the last attempted point (even if duplicate), logic requires returning something.
        """
        x_last = None
        for attempt in range(max_attempts):
            if self.verbose:
                print(
                    f"Fallback attempt {attempt + 1}/{max_attempts}: Using fallback strategy"
                )
            x_next = self._handle_acquisition_failure()

            x_next_rounded = self._repair_non_numeric(
                x_next.reshape(1, -1), self.var_type
            )[0]
            x_last = x_next_rounded

            x_next_2d = x_next_rounded.reshape(1, -1)
            X_transformed = self._transform_X(self.X_)
            x_new, _ = self.select_new(
                A=x_next_2d, X=X_transformed, tolerance=self.tolerance_x
            )

            if x_new.shape[0] > 0:
                return x_next_rounded, x_last

        return None, x_last

    def suggest_next_infill_point(self) -> np.ndarray:
        """Suggest next point to evaluate (dispatcher).

        The returned point is in the **Transformed and Mapped Space** (Internal Optimization Space).
        This means:
        1. Transformations (e.g., log, sqrt) have been applied.
        2. Dimension reduction has been applied (fixed variables removed).

        1. Try candidates from acquisition function optimizer.
        2. Handle_acquisition_failure (fallback).
        3. Return last attempt if all fails.

        Returns:
            ndarray: Next point to evaluate in **Transformed and Mapped Space**.

        Examples:
            >>> from spotoptim import SpotOptim
            >>> spot = SpotOptim()
            >>> x_internal = spot.suggest_next_infill_point()
        """
        # 1. Try optimizer candidates
        x_candidate = self._try_optimizer_candidates()
        if x_candidate is not None:
            return x_candidate

        # 2. Try fallback strategy
        max_attempts = 10
        x_candidate, x_last = self._try_fallback_strategy(max_attempts=max_attempts)
        if x_candidate is not None:
            return x_candidate

        # 3. Return last attempt
        if self.verbose:
            print(
                f"Warning: Could not find unique point after optimization candidates and {max_attempts} fallback attempts. "
                "Returning last candidate (may be duplicate)."
            )
        # Verify x_last is not None (should be handled by _try_fallback_strategy logic unless max_attempts=0)
        if x_last is None:
            # Should practically not happen if max_attempts > 0, but safe fallback
            return self._handle_acquisition_failure()
        return x_last

    def _update_repeats_infill_points(self, x_next: np.ndarray) -> np.ndarray:
        """Repeat infill point for noisy function evaluation.

        For noisy objective functions (repeats_surrogate > 1), creates multiple
        copies of the suggested point for repeated evaluation. Otherwise, returns
        the point in 2D array format.

        Args:
            x_next (ndarray): Next point to evaluate, shape (n_features,).

        Returns:
            ndarray: Points to evaluate, shape (repeats_surrogate, n_features)
                or (1, n_features) if repeats_surrogate == 1.

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> # Without repeats
            >>> opt = SpotOptim(
            ...     fun=lambda X: np.sum(X**2, axis=1),
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     repeats_surrogate=1
            ... )
            >>> x_next = np.array([1.0, 2.0])
            >>> x_repeated = opt._update_repeats_infill_points(x_next)
            >>> x_repeated.shape
            (1, 2)
            >>>
            >>> # With repeats for noisy function
            >>> opt_noisy = SpotOptim(
            ...     fun=lambda X: np.sum(X**2, axis=1),
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     repeats_surrogate=3
            ... )
            >>> x_next = np.array([1.0, 2.0])
            >>> x_repeated = opt_noisy._update_repeats_infill_points(x_next)
            >>> x_repeated.shape
            (3, 2)
            >>> # All three copies should be identical
            >>> np.all(x_repeated[0] == x_repeated[1])
            True
        """
        if self.repeats_surrogate > 1:
            x_next_repeated = np.repeat(
                x_next.reshape(1, -1), self.repeats_surrogate, axis=0
            )
        else:
            x_next_repeated = x_next.reshape(1, -1)
        return x_next_repeated

    def get_initial_design(self, X0: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate or process initial design points.

        Handles three scenarios:
        1. X0 is None: Generate space-filling design using LHS
        2. X0 is None but x0 is provided: Generate LHS and include x0 as first point
        3. X0 is provided: Transform and prepare user-provided initial design

        Args:
            X0 (ndarray, optional): User-provided initial design points in original scale,
                shape (n_initial, n_features). If None, generates space-filling design.
                Defaults to None.

        Returns:
            ndarray: Initial design points in internal (transformed and reduced) scale,
                shape (n_initial, n_features_reduced).

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> opt = SpotOptim(
            ...     fun=lambda X: np.sum(X**2, axis=1),
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     n_initial=10
            ... )
            >>> # Generate default LHS design
            >>> X0 = opt.get_initial_design()
            >>> X0.shape
            (10, 2)
            >>>
            >>> # Provide custom initial design
            >>> X0_custom = np.array([[0, 0], [1, 1], [2, 2]])
            >>> X0_processed = opt.get_initial_design(X0_custom)
            >>> X0_processed.shape
            (3, 2)
        """
        # Generate or use provided initial design
        if X0 is None:
            X0 = self._generate_initial_design()

            # If starting point x0 was provided, include it in initial design
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
            X0 = self._transform_X(X0)
            # If X0 is in full dimensions and we have dimension reduction, reduce it
            if self.red_dim and X0.shape[1] == len(self.ident):
                X0 = self.to_red_dim(X0)
            X0 = self._repair_non_numeric(X0, self.var_type)

        return X0

    def _curate_initial_design(self, X0: np.ndarray) -> np.ndarray:
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
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> opt = SpotOptim(
            ...     fun=lambda X: np.sum(X**2, axis=1),
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     n_initial=10,
            ...     var_type=['int', 'int']  # Integer variables may cause duplicates
            ... )
            >>> X0 = opt.get_initial_design()
            >>> X0_curated = opt._curate_initial_design(X0)
            >>> X0_curated.shape[0] == 10  # Should have n_initial unique points
            True
            >>>
            >>> # With repeats
            >>> opt_repeat = SpotOptim(
            ...     fun=lambda X: np.sum(X**2, axis=1),
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     n_initial=5,
            ...     repeats_initial=3
            ... )
            >>> X0 = opt_repeat.get_initial_design()
            >>> X0_curated = opt_repeat._curate_initial_design(X0)
            >>> X0_curated.shape[0] == 15  # 5 unique points * 3 repeats
            True
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
                    X_extra = self._repair_non_numeric(X_extra, self.var_type)

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

    def _rm_NA_values(
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
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> opt = SpotOptim(
            ...     fun=lambda X: np.sum(X**2, axis=1),
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     n_initial=10
            ... )
            >>> X0 = np.array([[1, 2], [3, 4], [5, 6]])
            >>> y0 = np.array([5.0, np.nan, np.inf])
            >>> X0_clean, y0_clean, n_eval = opt._rm_NA_values(X0, y0)
            >>> X0_clean.shape
            (1, 2)
            >>> y0_clean
            array([5.])
            >>> n_eval
            3
            >>>
            >>> # All valid values - no filtering
            >>> X0 = np.array([[1, 2], [3, 4]])
            >>> y0 = np.array([5.0, 25.0])
            >>> X0_clean, y0_clean, n_eval = opt._rm_NA_values(X0, y0)
            >>> X0_clean.shape
            (2, 2)
            >>> n_eval
            2
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

    def _check_size_initial_design(self, y0: np.ndarray, n_evaluated: int) -> None:
        """Validate that initial design has sufficient points for surrogate fitting.

        Checks if the number of valid initial design points meets the minimum
        requirement for fitting a surrogate model. The minimum required is the
        smaller of: (a) typical minimum for surrogate fitting (3 for multi-dimensional,
        2 for 1D), or (b) what the user requested (n_initial).

        Args:
            y0 (ndarray): Function values at initial design points (after filtering),
                shape (n_valid,).
            n_evaluated (int): Original number of points evaluated before filtering.

        Raises:
            ValueError: If the number of valid points is less than the minimum required.

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> opt = SpotOptim(
            ...     fun=lambda X: np.sum(X**2, axis=1),
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     n_initial=10
            ... )
            >>> # Sufficient points - no error
            >>> y0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            >>> opt._check_size_initial_design(y0, n_evaluated=10)
            >>>
            >>> # Insufficient points - raises ValueError
            >>> y0_small = np.array([1.0])
            >>> try:
            ...     opt._check_size_initial_design(y0_small, n_evaluated=10)
            ... except ValueError as e:
            ...     print(f"Error: {e}")
            Error: Insufficient valid initial design points: only 1 finite value(s) out of 10 evaluated...
            >>>
            >>> # With verbose output
            >>> opt_verbose = SpotOptim(
            ...     fun=lambda X: np.sum(X**2, axis=1),
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     n_initial=10,
            ...     verbose=True
            ... )
            >>> y0_reduced = np.array([1.0, 2.0, 3.0])  # Less than n_initial but valid
            >>> opt_verbose._check_size_initial_design(y0_reduced, n_evaluated=10)
            Note: Initial design size (3) is smaller than requested (10) due to NaN/inf values
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

    def _get_best_xy_initial_design(self) -> None:
        """Determine and store the best point from initial design.

        Finds the best (minimum) function value in the initial design,
        stores the corresponding point and value in instance attributes,
        and optionally prints the results if verbose mode is enabled.

        For noisy functions, also reports the mean best value.

        Note:
            This method assumes self.X_ and self.y_ have been initialized
            with the initial design evaluations.

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> opt = SpotOptim(
            ...     fun=lambda X: np.sum(X**2, axis=1),
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     n_initial=5,
            ...     verbose=True
            ... )
            >>> # Simulate initial design (normally done in optimize())
            >>> opt.X_ = np.array([[1, 2], [0, 0], [2, 1]])
            >>> opt.y_ = np.array([5.0, 0.0, 5.0])
            >>> opt._get_best_xy_initial_design()
            Initial best: f(x) = 0.000000
            >>> print(f"Best x: {opt.best_x_}")
            Best x: [0 0]
            >>> print(f"Best y: {opt.best_y_}")
            Best y: 0.0
            >>>
            >>> # With noisy function
            >>> opt_noise = SpotOptim(
            ...     fun=lambda X: np.sum(X**2, axis=1),
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     n_initial=5,
            ...     noise=True,
            ...     verbose=True
            ... )
            >>> opt_noise.X_ = np.array([[1, 2], [0, 0], [2, 1]])
            >>> opt_noise.y_ = np.array([5.0, 0.0, 5.0])
            >>> opt_noise.min_mean_y = 0.5  # Simulated mean best
            >>> opt_noise._get_best_xy_initial_design()
            Initial best: f(x) = 0.000000, mean best: f(x) = 0.500000
        """
        # Initial best
        best_idx = np.argmin(self.y_)
        self.best_x_ = self.X_[best_idx].copy()
        self.best_y_ = self.y_[best_idx]

        if self.verbose:
            if self.noise:
                print(
                    f"Initial best: f(x) = {self.best_y_:.6f}, mean best: f(x) = {self.min_mean_y:.6f}"
                )
            else:
                print(f"Initial best: f(x) = {self.best_y_:.6f}")

    def _apply_ocba(self) -> Optional[np.ndarray]:
        """Apply Optimal Computing Budget Allocation for noisy functions.

        Determines which existing design points should be re-evaluated based on
        OCBA algorithm. This method computes optimal budget allocation to improve
        the quality of the estimated best design.

        Returns:
            Optional[ndarray]: Array of design points to re-evaluate, shape (n_re_eval, n_features).
                Returns None if OCBA conditions are not met or OCBA is disabled.

        Note:
            OCBA is only applied when:
            - self.noise is True
            - self.ocba_delta > 0
            - All variances are > 0
            - At least 3 design points exist

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> opt = SpotOptim(
            ...     fun=lambda X: np.sum(X**2, axis=1) + np.random.normal(0, 0.1, X.shape[0]),
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     n_initial=5,
            ...     noise=True,
            ...     ocba_delta=5,
            ...     verbose=True
            ... )
            >>> # Simulate optimization state (normally done in optimize())
            >>> opt.mean_X = np.array([[1, 2], [0, 0], [2, 1]])
            >>> opt.mean_y = np.array([5.0, 0.1, 5.0])
            >>> opt.var_y = np.array([0.1, 0.05, 0.15])
            >>> X_ocba = opt._apply_ocba()
              OCBA: Adding 5 re-evaluation(s)
            >>> X_ocba.shape[0] == 5
            True
            >>>
            >>> # OCBA skipped - insufficient points
            >>> opt2 = SpotOptim(
            ...     fun=lambda X: np.sum(X**2, axis=1),
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     noise=True,
            ...     ocba_delta=5,
            ...     verbose=True
            ... )
            >>> opt2.mean_X = np.array([[1, 2], [0, 0]])
            >>> opt2.mean_y = np.array([5.0, 0.1])
            >>> opt2.var_y = np.array([0.1, 0.05])
            >>> X_ocba = opt2._apply_ocba()
            Warning: OCBA skipped (need >2 points with variance > 0)
            >>> X_ocba is None
            True
        """
        # OCBA: Compute optimal budget allocation for noisy functions
        # This determines which existing design points should be re-evaluated
        X_ocba = None
        if self.noise and self.ocba_delta > 0:
            # Check conditions for OCBA (need variance > 0 and at least 3 points)
            if not np.all(self.var_y > 0) and (self.mean_X.shape[0] <= 2):
                if self.verbose:
                    print("Warning: OCBA skipped (need >2 points with variance > 0)")
            elif np.all(self.var_y > 0) and (self.mean_X.shape[0] > 2):
                # Get OCBA allocation
                X_ocba = self._get_ocba_X(
                    self.mean_X,
                    self.mean_y,
                    self.var_y,
                    self.ocba_delta,
                    verbose=self.verbose,
                )
                if self.verbose and X_ocba is not None:
                    print(f"  OCBA: Adding {X_ocba.shape[0]} re-evaluation(s)")

        return X_ocba

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
            y_next = self._apply_penalty_NA(y_next, y_history=self.y_)

        # Identify which points are valid (finite) BEFORE removing them
        # Note: _remove_nan filters based on y_next finite values

        # Ensure y_next is a float array (maps non-convertible values like "error" or None to nan)
        # This is critical if the objective function returns non-numeric values and penalty=False
        if y_next.dtype == object:
            # Use safe float conversion similar to _apply_penalty_NA
            def _safe_float(v):
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return np.nan

            # Reconstruct as float array
            y_flat = np.array(y_next).flatten()
            y_next = np.array([_safe_float(v) for v in y_flat])

        finite_mask = np.isfinite(y_next)

        X_next_clean, y_next_clean = self._remove_nan(
            x_next, y_next, stop_on_zero_return=False
        )

        # If we have multi-objective values, we need to filter them too
        # The new MO values were appended to self.y_mo in _evaluate_function -> _mo2so -> _store_mo
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

    def _update_best_main_loop(
        self, x_next_repeated: np.ndarray, y_next: np.ndarray
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
            ...     noise=True,
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
        current_best = np.min(y_next)
        if current_best < self.best_y_:
            best_idx_in_new = np.argmin(y_next)
            # x_next_repeated is in transformed space, convert to original for storage
            self.best_x_ = self._inverse_transform_X(
                x_next_repeated[best_idx_in_new].reshape(1, -1)
            )[0]
            self.best_y_ = current_best

            if self.verbose:
                if self.noise:
                    print(
                        f"Iteration {self.n_iter_}: New best f(x) = {self.best_y_:.6f}, mean best: f(x) = {self.min_mean_y:.6f}"
                    )
                else:
                    print(
                        f"Iteration {self.n_iter_}: New best f(x) = {self.best_y_:.6f}"
                    )
        elif self.verbose:
            if self.noise:
                mean_y_new = np.mean(y_next)
                print(f"Iteration {self.n_iter_}: mean f(x) = {mean_y_new:.6f}")
            else:
                print(f"Iteration {self.n_iter_}: f(x) = {y_next[0]:.6f}")

    def _determine_termination(self, timeout_start: float) -> str:
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
            >>> import numpy as np
            >>> import time
            >>> from spotoptim import SpotOptim
            >>> opt = SpotOptim(
            ...     fun=lambda X: np.sum(X**2, axis=1),
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     max_iter=20,
            ...     max_time=10.0
            ... )
            >>> # Case 1: Maximum evaluations reached
            >>> opt.y_ = np.zeros(20)  # Simulate 20 evaluations
            >>> start_time = time.time()
            >>> msg = opt._determine_termination(start_time)
            >>> print(msg)
            Optimization terminated: maximum evaluations (20) reached
            >>>
            >>> # Case 2: Time limit exceeded
            >>> opt.y_ = np.zeros(10)  # Only 10 evaluations
            >>> start_time = time.time() - 700  # Simulate 11.67 minutes elapsed
            >>> msg = opt._determine_termination(start_time)
            >>> print(msg)
            Optimization terminated: time limit (10.00 min) reached
            >>>
            >>> # Case 3: Successful completion
            >>> opt.y_ = np.zeros(10)  # Under max_iter
            >>> start_time = time.time()  # Just started
            >>> msg = opt._determine_termination(start_time)
            >>> print(msg)
            Optimization finished successfully
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
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> opt = SpotOptim(fun=lambda x: np.sum(x**2), bounds=[(-5, 5)], var_name=["x"])
            >>> opt.optimize()
            >>> best_params = opt.get_best_hyperparameters()
            >>> print(best_params['x']) # Should be close to 0
        """
        if self.X_ is None or len(self.X_) == 0:
            return None

        # Determine which "best" to use
        if self.noise and hasattr(self, "min_mean_X"):
            best_x = self.min_mean_X
        else:
            best_x = self.best_x_

        if not as_dict:
            return best_x

        # Map factors using existing method (handles 2D, returns 2D)
        # We pass best_x as (1, D) and get (1, D) back
        mapped_x = self._map_to_factor_values(best_x.reshape(1, -1))[0]

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

    def optimize(self, X0: Optional[np.ndarray] = None) -> OptimizeResult:
        """Run the optimization process.

        The optimization terminates when either:
        - Total function evaluations reach max_iter (including initial design), OR
        - Runtime exceeds max_time minutes

        **Input/Output Spaces:**
        - **Input X0**: Expected in **Natural Space** (original scale, physical units).
        - **Output result.x**: Returned in **Natural Space**.
        - **Output result.X**: Returned in **Natural Space**.
        - **Internal Optimization**: Performed in **Transformed and Mapped Space**.

        Args:
            X0 (ndarray, optional): Initial design points in **Natural Space**, shape (n_initial, n_features).
                If None, generates space-filling design. Defaults to None.

        Returns:
            OptimizeResult: Optimization result with fields:
                - x: best point found in **Natural Space**
                - fun: best function value
                - nfev: number of function evaluations (including initial design)
                - nit: number of sequential optimization iterations (after initial design)
                - success: whether optimization succeeded
                - message: termination message indicating reason for stopping, including
                  statistics (function value, iterations, evaluations)
                - X: all evaluated points in **Natural Space**
                - y: all function values

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> opt = SpotOptim(
            ...     fun=lambda X: np.sum(X**2, axis=1),
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     n_initial=5,
            ...     max_iter=20,
            ...     verbose=True
            ... )
            >>> result = opt.optimize()
            >>> print(result.message)
            Optimization finished successfully
                     Current function value: ...
                     Iterations: ...
                     Function evaluations: ...
            >>> print("Best point:", result.x)
            Best point: [some point close to [0, 0]]
            >>> print("Best value:", result.fun)
            Best value: [some value close to 0]
        """
        # Start timer for max_time check
        timeout_start = time.time()

        # Set seed for reproducibility (crucial for ensuring identical results across runs)
        self._set_seed()

        # Set initial design (generate or process user-provided points)
        X0 = self.get_initial_design(X0)

        # Curate initial design (remove duplicates, generate additional points if needed, repeat if necessary)
        X0 = self._curate_initial_design(X0)

        # Evaluate initial design
        y0 = self._evaluate_function(X0)

        # Handle NaN/inf values in initial design (remove invalid points)
        X0, y0, n_evaluated = self._rm_NA_values(X0, y0)

        # Check if we have enough valid points to continue
        self._check_size_initial_design(y0, n_evaluated)

        # Initialize storage and statistics
        self._init_storage(X0, y0)

        # Update stats after initial design
        self.update_stats()

        # Log initial design to TensorBoard
        self._init_tensorboard()

        # Determine and report initial best
        self._get_best_xy_initial_design()

        # Main optimization loop
        # Termination: continue while (total_evals < max_iter) AND (elapsed_time < max_time)
        consecutive_failures = 0
        while (len(self.y_) < self.max_iter) and (
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
                return OptimizeResult(
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
            self._fit_scheduler()

            # Apply OCBA for noisy functions
            X_ocba = self._apply_ocba()

            # Suggest next point
            x_next = self.suggest_next_infill_point()

            # Repeat next point if repeats_surrogate > 1
            x_next_repeated = self._update_repeats_infill_points(x_next)

            # Append OCBA points to new design points (if applicable)
            if X_ocba is not None:
                x_next_repeated = append(X_ocba, x_next_repeated, axis=0)

            # Evaluate next point(s) including OCBA points
            y_next = self._evaluate_function(x_next_repeated)

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
            self._update_success_rate(y_next)

            # Update storage
            self._update_storage(x_next_repeated, y_next)

            # Update stats
            self.update_stats()

            # Log to TensorBoard
            if self.tb_writer is not None:
                # Log each new evaluation
                for i in range(len(y_next)):
                    self._write_tensorboard_hparams(x_next_repeated[i], y_next[i])
                self._write_tensorboard_scalars()

            # Update best solution
            self._update_best_main_loop(x_next_repeated, y_next)

        # Expand results to full dimensions if needed
        # Note: best_x_ and X_ are already in original scale (stored that way)
        best_x_full = (
            self.to_all_dim(self.best_x_.reshape(1, -1))[0]
            if self.red_dim
            else self.best_x_
        )
        X_full = self.to_all_dim(self.X_) if self.red_dim else self.X_

        # Determine termination reason
        status_message = self._determine_termination(timeout_start)

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
        best_x_result = self._map_to_factor_values(best_x_full.reshape(1, -1))[0]
        X_result = self._map_to_factor_values(X_full) if self._factor_maps else X_full

        # Return scipy-style result
        return OptimizeResult(
            x=best_x_result,
            fun=self.best_y_,
            nfev=len(self.y_),
            nit=self.n_iter_,
            success=True,
            message=message,
            X=X_result,
            y=self.y_,
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

        Creates a 2x2 plot showing:
        - Top left: 3D surface of predictions
        - Top right: 3D surface of prediction uncertainty
        - Bottom left: Contour plot of predictions with evaluated points
        - Bottom right: Contour plot of prediction uncertainty

        Args:
            i (int, optional): Index of the first dimension to plot. Defaults to 0.
            j (int, optional): Index of the second dimension to plot. Defaults to 1.
            show (bool, optional): If True, displays the plot immediately. Defaults to True.
            alpha (float, optional): Transparency of the 3D surface plots (0=transparent, 1=opaque).
                Defaults to 0.8.
            var_name (list of str, optional): Names for each dimension. If None, uses instance var_name.
                Defaults to None.
            cmap (str, optional): Matplotlib colormap name. Defaults to 'jet'.
            num (int, optional): Number of grid points per dimension for mesh grid. Defaults to 100.
            vmin (float, optional): Minimum value for color scale. If None, determined from data.
                Defaults to None.
            vmax (float, optional): Maximum value for color scale. If None, determined from data.
                Defaults to None.
            add_points (bool, optional): If True, overlay evaluated points on contour plots.
                Defaults to True.
            grid_visible (bool, optional): If True, show grid lines on contour plots. Defaults to True.
            contour_levels (int, optional): Number of contour levels. Defaults to 30.
            figsize (tuple of int, optional): Figure size in inches (width, height). Defaults to (12, 10).

        Raises:
            ValueError: If optimization hasn't been run yet, or if i, j are invalid.

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> def sphere(X):
            ...     return np.sum(X**2, axis=1)
            >>> # Example 1: Using var_name in constructor
            >>> opt = SpotOptim(fun=sphere, bounds=[(-5, 5), (-5, 5)],
            ...                 max_iter=10, n_initial=5, var_name=['x1', 'x2'])
            >>> result = opt.optimize()
            >>> opt.plot_surrogate(i=0, j=1)  # Uses instance var_name
            >>> # Example 2: Override var_name for this plot
            >>> opt.plot_surrogate(i=0, j=1, var_name=['custom1', 'custom2'])
        """
        # Validation
        if self.X_ is None or self.y_ is None:
            raise ValueError("No optimization data available. Run optimize() first.")

        k = self.n_dim
        if i >= k or j >= k:
            raise ValueError(f"Dimensions i={i} and j={j} must be less than n_dim={k}.")
        if i == j:
            raise ValueError("Dimensions i and j must be different.")

        # Use instance var_name if not provided
        if var_name is None:
            var_name = self.var_name

        # Generate mesh grid
        X_i, X_j, grid_points = self._generate_mesh_grid(i, j, num)

        # Predict on grid
        y_pred, y_std = self._predict_with_uncertainty(grid_points)
        Z_pred = y_pred.reshape(X_i.shape)
        Z_std = y_std.reshape(X_i.shape)

        # Create figure
        fig = plt.figure(figsize=figsize)

        # Plot 1: 3D surface of predictions
        ax1 = fig.add_subplot(221, projection="3d")
        ax1.plot_surface(X_i, X_j, Z_pred, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
        ax1.set_title("Prediction Surface")
        ax1.set_xlabel(var_name[i] if var_name else f"x{i}")
        ax1.set_ylabel(var_name[j] if var_name else f"x{j}")
        ax1.set_zlabel("Prediction")

        # Plot 2: 3D surface of prediction uncertainty
        ax2 = fig.add_subplot(222, projection="3d")
        ax2.plot_surface(X_i, X_j, Z_std, cmap=cmap, alpha=alpha)
        ax2.set_title("Prediction Uncertainty Surface")
        ax2.set_xlabel(var_name[i] if var_name else f"x{i}")
        ax2.set_ylabel(var_name[j] if var_name else f"x{j}")
        ax2.set_zlabel("Std. Dev.")

        # Plot 3: Contour of predictions
        ax3 = fig.add_subplot(223)
        contour3 = ax3.contourf(
            X_i, X_j, Z_pred, levels=contour_levels, cmap=cmap, vmin=vmin, vmax=vmax
        )
        plt.colorbar(contour3, ax=ax3)
        if add_points:
            ax3.scatter(
                self.X_[:, i],
                self.X_[:, j],
                c="red",
                s=30,
                edgecolors="black",
                zorder=5,
                label="Evaluated points",
            )
            ax3.legend()
        ax3.set_title("Prediction Contour")
        ax3.set_xlabel(var_name[i] if var_name else f"x{i}")
        ax3.set_ylabel(var_name[j] if var_name else f"x{j}")
        ax3.grid(visible=grid_visible)

        # Plot 4: Contour of prediction uncertainty
        ax4 = fig.add_subplot(224)
        contour4 = ax4.contourf(X_i, X_j, Z_std, levels=contour_levels, cmap=cmap)
        plt.colorbar(contour4, ax=ax4)
        if add_points:
            ax4.scatter(
                self.X_[:, i],
                self.X_[:, j],
                c="red",
                s=30,
                edgecolors="black",
                zorder=5,
                label="Evaluated points",
            )
            ax4.legend()
        ax4.set_title("Uncertainty Contour")
        ax4.set_xlabel(var_name[i] if var_name else f"x{i}")
        ax4.set_ylabel(var_name[j] if var_name else f"x{j}")
        ax4.grid(visible=grid_visible)

        plt.tight_layout()

        if show:
            plt.show()

    def plot_progress(
        self,
        show: bool = True,
        log_y: bool = False,
        figsize: Tuple[int, int] = (10, 6),
        ylabel: str = "Objective Value",
        mo: bool = False,
    ) -> None:
        """Plot optimization progress showing all evaluations and best-so-far curve.

        This method visualizes the optimization history, displaying both individual
        function evaluations and the cumulative best value found. Initial design points
        are shown as individual scatter points with a light grey background region,
        while sequential optimization iterations are connected with lines.

        Args:
            show (bool, optional): Whether to display the plot. Defaults to True.
            log_y (bool, optional): Whether to use log scale for y-axis. Defaults to False.
            figsize (tuple, optional): Figure size as (width, height). Defaults to (10, 6).
            ylabel (str, optional): Label for y-axis. Defaults to "Objective Value".
            mo (bool, optional): Whether to plot individual objectives if available. Defaults to False.

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> def objective(X):
            ...     return np.sum(X**2, axis=1)
            >>> opt = SpotOptim(
            ...     fun=objective,
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     max_iter=30,
            ...     n_initial=10,
            ...     seed=42
            ... )
            >>> result = opt.optimize()
            >>> # Plot with linear y-axis
            >>> opt.plot_progress()
            >>> # Plot with log y-axis for better visibility of small improvements
            >>> opt.plot_progress(log_y=True)
            >>> # Plot with multi-objective values (if available)
            >>> opt.plot_progress(mo=True, log_y=True)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plot_progress(). "
                "Install it with: pip install matplotlib"
            )

        if self.y_ is None or len(self.y_) == 0:
            raise ValueError("No optimization data available. Run optimize() first.")

        history = self.y_

        plt.figure(figsize=figsize)

        # Separate initial design points from sequential evaluations
        n_initial = min(self.n_initial, len(history))
        initial_y = history[:n_initial]
        sequential_y = history[n_initial:]

        # Add light grey background for initial design region
        if n_initial > 0:
            plt.axvspan(0, n_initial, alpha=0.15, color="gray", zorder=0)

        # Plot multi-objective values if requested and available
        if mo and self.y_mo is not None:
            n_samples, n_obj = self.y_mo.shape
            x_all = np.arange(1, n_samples + 1)

            # Determine names
            names = self.objective_names
            if names is None or len(names) != n_obj:
                names = [f"Objective {i+1}" for i in range(n_obj)]

            # Basic colors (excluding gray/red used for main plot)
            # Use a colormap or a set list
            _ = plt.cm.viridis(np.linspace(0, 1, n_obj))

            for i in range(n_obj):
                plt.plot(
                    x_all,
                    self.y_mo[:, i],
                    linestyle="--",
                    marker="x",
                    alpha=0.7,
                    label=f"{names[i]}",
                    zorder=1,
                )

        # Plot initial design points as scatter (not connected)
        sequential_y = history[n_initial:]

        # Add light grey background for initial design region
        if n_initial > 0:
            plt.axvspan(0, n_initial, alpha=0.15, color="gray", zorder=0)

        # Plot initial design points as scatter (not connected)
        if n_initial > 0:
            x_initial = np.arange(1, n_initial + 1)
            plt.scatter(
                x_initial,
                initial_y,
                alpha=0.6,
                s=50,
                label=f"Initial design (n={n_initial})",
                color="gray",
                edgecolors="black",
                linewidth=0.5,
                zorder=2,
            )

        # Plot sequential evaluations (connected with line)
        if len(sequential_y) > 0:
            x_sequential = np.arange(n_initial + 1, len(history) + 1)
            plt.plot(
                x_sequential,
                sequential_y,
                "o-",
                alpha=0.6,
                label="Sequential evaluations",
                markersize=5,
                zorder=3,
            )

        # Plot best-so-far curve starting after initial design
        if len(history) > n_initial:
            # Best so far across all evaluations
            best_so_far = np.minimum.accumulate(history)
            # Start the red line after initial design
            x_best = np.arange(n_initial + 1, len(history) + 1)
            y_best = best_so_far[n_initial:]
            plt.plot(
                x_best,
                y_best,
                "r-",
                linewidth=2,
                label="Best so far",
                zorder=4,
            )

        plt.xlabel("Iteration", fontsize=11)
        plt.ylabel(ylabel, fontsize=11)

        title = "Optimization Progress"
        if log_y:
            title += " (Log Scale)"
        plt.title(title, fontsize=12)

        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        if log_y:
            plt.yscale("log")

        plt.tight_layout()

        if show:
            plt.show()

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
        """Plot surrogate contours for all combinations of the top max_imp important parameters.

        This method identifies the most important parameters using importance scores,
        then generates surrogate contour plots for all pairwise combinations of these
        parameters. Factor (categorical) variables are handled by creating discrete grids
        and displaying factor level names on the axes.

        Args:
            max_imp (int, optional): Number of most important parameters to visualize.
                Defaults to 3. For max_imp=3, creates 3 plots: (0,1), (0,2), (1,2).
            show (bool, optional): If True, displays plots immediately. Defaults to True.
            alpha (float, optional): Transparency of 3D surface plots (0=transparent, 1=opaque).
                Defaults to 0.8.
            cmap (str, optional): Matplotlib colormap name. Defaults to 'jet'.
            num (int, optional): Number of grid points per dimension. Defaults to 100.
                For factor variables, uses the number of unique levels instead.
            add_points (bool, optional): If True, overlay evaluated points on contour plots.
                Defaults to True.
            grid_visible (bool, optional): If True, show grid lines. Defaults to True.
            contour_levels (int, optional): Number of contour levels. Defaults to 30.
            figsize (tuple of int, optional): Figure size in inches (width, height).
                Defaults to (12, 10).

        Raises:
            ValueError: If optimization hasn't been run yet or max_imp is invalid.

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> def sphere(X):
            ...     return np.sum(X**2, axis=1)
            >>> opt = SpotOptim(
            ...     fun=sphere,
            ...     bounds=[(-5, 5), (-5, 5), (-5, 5), (-5, 5)],
            ...     max_iter=20,
            ...     n_initial=10,
            ...     var_name=['x1', 'x2', 'x3', 'x4']
            ... )
            >>> result = opt.optimize()
            >>> # Plot surrogate contours for top 3 most important parameters
            >>> opt.plot_important_hyperparameter_contour(max_imp=3)
            Plotting surrogate contours for top 3 most important parameters:
              x1: importance = 35.24%
              x2: importance = 28.17%
              x3: importance = 22.45%
            <BLANKLINE>
            Generating 3 surrogate plots...
              Plotting x1 vs x2
              Plotting x1 vs x3
              Plotting x2 vs x3
        """
        from itertools import combinations

        # Validation
        if self.X_ is None or self.y_ is None:
            raise ValueError("No optimization data available. Run optimize() first.")

        if max_imp < 2:
            raise ValueError("max_imp must be at least 2 to generate pairwise plots.")

        if max_imp > self.n_dim:
            raise ValueError(
                f"max_imp ({max_imp}) cannot exceed number of dimensions ({self.n_dim})."
            )

        # Get importance scores
        importance = self.get_importance()

        # Get indices of most important parameters (sorted by importance, descending)
        importance_array = np.array(importance)
        top_indices = np.argsort(importance_array)[::-1][:max_imp]

        # Get parameter names for informative output
        param_names = (
            self.var_name
            if self.var_name is not None
            else [f"x{i}" for i in range(len(importance))]
        )

        print(
            f"Plotting surrogate contours for top {max_imp} most important parameters:"
        )
        for idx in top_indices:
            param_type = self.var_type[idx] if self.var_type else "float"
            print(
                f"  {param_names[idx]}: importance = {importance[idx]:.2f}% (type: {param_type})"
            )

        # Generate all pairwise combinations
        pairs = list(combinations(top_indices, 2))

        print(f"\nGenerating {len(pairs)} surrogate plots...")

        # Plot each combination
        for i, j in pairs:
            print(f"  Plotting {param_names[i]} vs {param_names[j]}")
            self._plot_surrogate_with_factors(
                i=int(i),
                j=int(j),
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
        """Plot surrogate model handling factor variables by mapping to integers.

        For factor variables, creates discrete grids and displays factor level names.

        Args:
            i (int): Index of the first dimension to plot.
            j (int): Index of the second dimension to plot.
            show (bool, optional): If True, displays the plot immediately. Defaults to True.
            alpha (float, optional): Transparency of the 3D surface plots (0=transparent, 1=opaque).
                Defaults to 0.8.
            cmap (str, optional): Matplotlib colormap name. Defaults to 'jet'.
            num (int, optional): Number of grid points per dimension for mesh grid. Defaults to 100.
            add_points (bool, optional): If True, overlay evaluated points on contour plots.
                Defaults to True.
            grid_visible (bool, optional): If True, show grid lines on contour plots. Defaults to True.
            contour_levels (int, optional): Number of contour levels. Defaults to 30.
            figsize (tuple of int, optional): Figure size in inches (width, height). Defaults to (12, 10).

        Raises:
            ImportError: If matplotlib is not installed.

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> def objective(X):
            ...     return np.sum(X**2, axis=1)
            >>> opt = SpotOptim(
            ...     fun=objective,
            ...     bounds=[(-5, 5), (-5, 5), (0, 2)],
            ...     var_type=['float', 'float', 'factor'],
            ...     var_name=['x1', 'x2', 'category'],
            ...     max_iter=20,
            ...     n_initial=10,
            ...     seed=42
            ... )
            >>> result = opt.optimize()
            >>> # Plot surrogate with factor handling for dimensions 0 and 2
            >>> opt._plot_surrogate_with_factors(i=0, j=2)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required. Install with: pip install matplotlib"
            )

        # Check if either dimension is a factor
        is_factor_i = self.var_type and self.var_type[i] == "factor"
        is_factor_j = self.var_type and self.var_type[j] == "factor"

        # Get parameter names
        var_name = (
            self.var_name if self.var_name else [f"x{k}" for k in range(self.n_dim)]
        )

        # Generate mesh grid with factor handling
        if is_factor_i or is_factor_j:
            (
                X_i,
                X_j,
                grid_points,
                factor_labels_i,
                factor_labels_j,
            ) = self._generate_mesh_grid_with_factors(
                i, j, num, is_factor_i, is_factor_j
            )
        else:
            X_i, X_j, grid_points = self._generate_mesh_grid(i, j, num)
            factor_labels_i, factor_labels_j = None, None

        # Predict on grid
        y_pred, y_std = self._predict_with_uncertainty(grid_points)
        Z_pred = y_pred.reshape(X_i.shape)
        Z_std = y_std.reshape(X_i.shape)

        # Create figure
        fig = plt.figure(figsize=figsize)

        # Plot 1: 3D surface of predictions
        ax1 = fig.add_subplot(221, projection="3d")
        ax1.plot_surface(X_i, X_j, Z_pred, cmap=cmap, alpha=alpha)
        ax1.set_title("Prediction Surface")
        ax1.set_xlabel(var_name[i])
        ax1.set_ylabel(var_name[j])
        ax1.set_zlabel("Prediction")

        # Set tick labels for factors
        if is_factor_i and factor_labels_i:
            ax1.set_xticks(range(len(factor_labels_i)))
            ax1.set_xticklabels(factor_labels_i, rotation=45, ha="right")
        if is_factor_j and factor_labels_j:
            ax1.set_yticks(range(len(factor_labels_j)))
            ax1.set_yticklabels(factor_labels_j, rotation=45, ha="right")

        # Plot 2: 3D surface of prediction uncertainty
        ax2 = fig.add_subplot(222, projection="3d")
        ax2.plot_surface(X_i, X_j, Z_std, cmap=cmap, alpha=alpha)
        ax2.set_title("Prediction Uncertainty Surface")
        ax2.set_xlabel(var_name[i])
        ax2.set_ylabel(var_name[j])
        ax2.set_zlabel("Std. Dev.")

        if is_factor_i and factor_labels_i:
            ax2.set_xticks(range(len(factor_labels_i)))
            ax2.set_xticklabels(factor_labels_i, rotation=45, ha="right")
        if is_factor_j and factor_labels_j:
            ax2.set_yticks(range(len(factor_labels_j)))
            ax2.set_yticklabels(factor_labels_j, rotation=45, ha="right")

        # Plot 3: Contour of predictions
        ax3 = fig.add_subplot(223)
        contour3 = ax3.contourf(X_i, X_j, Z_pred, levels=contour_levels, cmap=cmap)
        plt.colorbar(contour3, ax=ax3)

        if add_points:
            # Map factor variables in evaluated points to integers for display
            X_i_points = self.X_[:, i].copy()
            X_j_points = self.X_[:, j].copy()

            if is_factor_i and self._factor_maps and i in self._factor_maps:
                # Map string values back to integer indices
                factor_map = self._factor_maps[i]
                reverse_map = {v: k for k, v in enumerate(factor_map)}
                X_i_points = np.array([reverse_map.get(val, 0) for val in X_i_points])

            if is_factor_j and self._factor_maps and j in self._factor_maps:
                factor_map = self._factor_maps[j]
                reverse_map = {v: k for k, v in enumerate(factor_map)}
                X_j_points = np.array([reverse_map.get(val, 0) for val in X_j_points])

            ax3.scatter(
                X_i_points,
                X_j_points,
                c="red",
                s=30,
                edgecolors="black",
                zorder=5,
                label="Evaluated points",
            )
            ax3.legend()

        ax3.set_title("Prediction Contour")
        ax3.set_xlabel(var_name[i])
        ax3.set_ylabel(var_name[j])
        ax3.grid(visible=grid_visible)

        if is_factor_i and factor_labels_i:
            ax3.set_xticks(range(len(factor_labels_i)))
            ax3.set_xticklabels(factor_labels_i, rotation=45, ha="right")
        if is_factor_j and factor_labels_j:
            ax3.set_yticks(range(len(factor_labels_j)))
            ax3.set_yticklabels(factor_labels_j, rotation=45, ha="right")

        # Plot 4: Contour of prediction uncertainty
        ax4 = fig.add_subplot(224)
        contour4 = ax4.contourf(X_i, X_j, Z_std, levels=contour_levels, cmap=cmap)
        plt.colorbar(contour4, ax=ax4)

        if add_points:
            ax4.scatter(
                X_i_points,
                X_j_points,
                c="red",
                s=30,
                edgecolors="black",
                zorder=5,
                label="Evaluated points",
            )
            ax4.legend()

        ax4.set_title("Uncertainty Contour")
        ax4.set_xlabel(var_name[i])
        ax4.set_ylabel(var_name[j])
        ax4.grid(visible=grid_visible)

        if is_factor_i and factor_labels_i:
            ax4.set_xticks(range(len(factor_labels_i)))
            ax4.set_xticklabels(factor_labels_i, rotation=45, ha="right")
        if is_factor_j and factor_labels_j:
            ax4.set_yticks(range(len(factor_labels_j)))
            ax4.set_yticklabels(factor_labels_j, rotation=45, ha="right")

        plt.tight_layout()

        if show:
            plt.show()

    def _generate_mesh_grid_with_factors(
        self, i: int, j: int, num: int, is_factor_i: bool, is_factor_j: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list, list]:
        """Generate mesh grid with special handling for factor variables.

        Returns:
            X_i, X_j: Meshgrids for plotting
            grid_points: Points for prediction (in transformed space)
            factor_labels_i: Factor level names for dimension i (None if numeric)
            factor_labels_j: Factor level names for dimension j (None if numeric)

        Raises:
            ValueError: If generated grid points contain non-finite values after transformation.

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> def objective(X):
            ...     return np.sum(X**2, axis=1)
            >>> opt = SpotOptim(
            ...     fun=objective,
            ...     bounds=[(-5, 5), (-5, 5), (0, 2)],
            ...     var_type=['float', 'float', 'factor'],
            ...     var_name=['x1', 'x2', 'category'],
            ...     max_iter=20,
            ...     n_initial=10,
            ...     seed=42
            ... )
            >>> result = opt.optimize()
            >>> # Generate mesh grid with factor handling for dimensions 0 and 2
            >>> X_i, X_j, grid_points, factor_labels_i, factor_labels_j = (
            ...     opt._generate_mesh_grid_with_factors(0, 2, num=10, is_factor_i=False, is_factor_j=True)
            ... )
        """
        k = self.n_dim

        # Compute mean values, handling factor variables carefully
        mean_values = np.empty(k, dtype=object)
        for dim_idx in range(k):
            if self.var_type and self.var_type[dim_idx] == "factor":
                # For factor variables, use the most common value's integer index
                col_values = self.X_[:, dim_idx]
                unique_vals, counts = np.unique(col_values, return_counts=True)
                most_common_str = unique_vals[np.argmax(counts)]

                # Map string back to integer index
                if dim_idx in self._factor_maps:
                    # Find the integer key for this string value
                    reverse_map = {v: k for k, v in self._factor_maps[dim_idx].items()}
                    mean_values[dim_idx] = reverse_map.get(most_common_str, 0)
                else:
                    mean_values[dim_idx] = 0  # Default to first level
            else:
                # For numeric variables, use mean
                mean_values[dim_idx] = np.mean(self.X_[:, dim_idx].astype(float))

        # Handle dimension i
        # Helper function to avoid problematic values with log transforms
        def safe_bound(value, trans, is_lower):
            """Add epsilon to avoid problematic values with log transforms."""
            if trans in ["log10", "log", "ln"]:
                eps = 1e-10
                if is_lower and value <= 0:
                    return eps
                elif value <= 0:
                    return eps
            return value

        if is_factor_i and self._factor_maps and i in self._factor_maps:
            factor_map_i = self._factor_maps[i]
            n_levels_i = len(factor_map_i)
            x_i = np.arange(n_levels_i)  # Integer indices
            factor_labels_i = list(factor_map_i.values())  # Get the string labels
        else:
            lower_i = safe_bound(self._original_lower[i], self.var_trans[i], True)
            upper_i = safe_bound(self._original_upper[i], self.var_trans[i], False)
            x_i = linspace(lower_i, upper_i, num=num)
            factor_labels_i = None

        # Handle dimension j
        if is_factor_j and self._factor_maps and j in self._factor_maps:
            factor_map_j = self._factor_maps[j]
            n_levels_j = len(factor_map_j)
            x_j = np.arange(n_levels_j)  # Integer indices
            factor_labels_j = list(factor_map_j.values())  # Get the string labels
        else:
            lower_j = safe_bound(self._original_lower[j], self.var_trans[j], True)
            upper_j = safe_bound(self._original_upper[j], self.var_trans[j], False)
            x_j = linspace(lower_j, upper_j, num=num)
            factor_labels_j = None

        X_i, X_j = meshgrid(x_i, x_j)

        # Initialize grid points with mean values
        grid_points_original = np.tile(mean_values, (X_i.size, 1))
        grid_points_original[:, i] = X_i.ravel()
        grid_points_original[:, j] = X_j.ravel()

        # Convert to float array to handle numeric operations properly
        # Object dtype with np.float64/float values causes issues with np.around
        grid_points_float = np.zeros((grid_points_original.shape[0], k), dtype=float)
        for dim_idx in range(k):
            grid_points_float[:, dim_idx] = grid_points_original[:, dim_idx].astype(
                float
            )

        # Apply type constraints (convert to proper numeric types)
        grid_points_float = self._repair_non_numeric(grid_points_float, self.var_type)

        # Transform grid points for surrogate prediction
        grid_points_transformed = self._transform_X(grid_points_float)

        # Validate that transformed grid points are finite
        if not np.all(np.isfinite(grid_points_transformed)):
            raise ValueError(
                "Generated grid points contain non-finite values after transformation. "
                "This may indicate an issue with variable transformations or bounds."
            )

        return X_i, X_j, grid_points_transformed, factor_labels_i, factor_labels_j

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
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> def objective(X):
            ...     return np.sum(X**2, axis=1)
            >>> opt = SpotOptim(
            ...     fun=objective,
            ...     bounds=[(-5, 5), (-5, 5), (-5, 5), (-5, 5)],
            ...     var_name=["x0", "x1", "x2", "x3"],
            ...     max_iter=30,
            ...     n_initial=10,
            ...     seed=42
            ... )
            >>> result = opt.optimize()
            >>> # Plot parameter distributions
            >>> opt.plot_parameter_scatter(result)
            >>> # Plot with custom settings
            >>> opt.plot_parameter_scatter(result, cmap="plasma", ylabel="Error")
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

    def _generate_mesh_grid(
        self, i: int, j: int, num: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a mesh grid for two dimensions, filling others with mean values.

        Args:
            i (int): Index of the first dimension to vary.
            j (int): Index of the second dimension to vary.
            num (int, optional): Number of grid points per dimension. Defaults to 100.

        Returns:
            tuple: A tuple containing:
                - X_i (ndarray): Meshgrid for dimension i (in original scale).
                - X_j (ndarray): Meshgrid for dimension j (in original scale).
                - grid_points (ndarray): Grid points for prediction (in transformed scale), shape (num*num, n_dim).

        Raises:
            ValueError: If generated grid points contain non-finite values after transformation.

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> def objective(X):
            ...     return np.sum(X**2, axis=1)
            >>> opt = SpotOptim(
            ...     fun=objective,
            ...     bounds=[(-5, 5), (-5, 5), (-5, 5)],
            ...     max_iter=20,
            ...     n_initial=10,
            ...     seed=42
            ... )
            >>> result = opt.optimize()
            >>> # Generate mesh grid for dimensions 0 and 1
            >>> X_i, X_j, grid_points = opt._generate_mesh_grid(i=0, j=1, num=50)
        """
        k = self.n_dim
        # Compute mean values with proper handling of factor variables
        mean_values = np.empty(k, dtype=object)
        for dim_idx in range(k):
            if self.var_type and self.var_type[dim_idx] == "factor":
                # For factor variables, use most common value mapped to integer
                col_values = self.X_[:, dim_idx]
                unique_vals, counts = np.unique(col_values, return_counts=True)
                most_common_str = unique_vals[np.argmax(counts)]
                # Map string back to integer index
                if dim_idx in self._factor_maps:
                    reverse_map = {v: k for k, v in self._factor_maps[dim_idx].items()}
                    mean_values[dim_idx] = reverse_map.get(most_common_str, 0)
                else:
                    mean_values[dim_idx] = 0
            else:
                # For numeric/int variables, compute mean
                mean_values[dim_idx] = np.mean(self.X_[:, dim_idx].astype(float))

        # Convert mean_values to float array for numeric operations
        mean_values_float = mean_values.astype(float)

        # Create grid for dimensions i and j using ORIGINAL bounds for plotting
        # Add small epsilon for log-transformed variables to avoid log(0) = -inf
        def safe_bound(value, trans, is_lower):
            """Add epsilon to avoid problematic values with log transforms."""
            if trans in ["log10", "log", "ln"]:
                eps = 1e-10
                if is_lower and value <= 0:
                    return eps
                elif value <= 0:
                    return eps
            return value

        lower_i = safe_bound(self._original_lower[i], self.var_trans[i], True)
        upper_i = safe_bound(self._original_upper[i], self.var_trans[i], False)
        lower_j = safe_bound(self._original_lower[j], self.var_trans[j], True)
        upper_j = safe_bound(self._original_upper[j], self.var_trans[j], False)

        x_i = linspace(lower_i, upper_i, num=num)
        x_j = linspace(lower_j, upper_j, num=num)
        X_i, X_j = meshgrid(x_i, x_j)

        # Initialize grid points with mean values (in original scale)
        grid_points_original = np.tile(mean_values_float, (X_i.size, 1))
        grid_points_original[:, i] = X_i.ravel()
        grid_points_original[:, j] = X_j.ravel()

        # Apply type constraints
        grid_points_original = self._repair_non_numeric(
            grid_points_original, self.var_type
        )

        # Transform to internal scale for surrogate prediction
        grid_points = self._transform_X(grid_points_original)

        # Validate that transformed grid points are finite
        if not np.all(np.isfinite(grid_points)):
            # Provide detailed error information
            non_finite_mask = ~np.isfinite(grid_points)
            problem_dims = np.where(non_finite_mask.any(axis=0))[0]
            error_msg = (
                "Generated grid points contain non-finite values after transformation.\n"
                f"Problematic dimensions: {problem_dims.tolist()}\n"
            )
            for dim in problem_dims:
                dim_name = self.var_name[dim] if self.var_name else f"x{dim}"
                trans = self.var_trans[dim] if self.var_trans else None
                orig_vals = grid_points_original[:, dim]
                trans_vals = grid_points[:, dim]
                error_msg += (
                    f"  Dimension {dim} ({dim_name}):\n"
                    f"    Transform: {trans}\n"
                    f"    Original range: [{orig_vals.min():.6f}, {orig_vals.max():.6f}]\n"
                    f"    Transformed range: [{trans_vals[np.isfinite(trans_vals)].min() if np.any(np.isfinite(trans_vals)) else 'N/A':.6f}, "
                    f"{trans_vals[np.isfinite(trans_vals)].max() if np.any(np.isfinite(trans_vals)) else 'N/A':.6f}]\n"
                    f"    Non-finite count: {(~np.isfinite(trans_vals)).sum()}\n"
                )
            raise ValueError(error_msg)

        return X_i, X_j, grid_points

    def _get_experiment_filename(self, prefix: str) -> str:
        """Generate experiment filename from prefix.

        Args:
            prefix (str): Prefix for the filename.

        Returns:
            str: Filename with '_exp.pkl' suffix.

        Examples:
            >>> from spotoptim import SpotOptim
            >>> opt = SpotOptim(fun=lambda x: x, bounds=[(0, 1)])
            >>> exp_filename = opt._get_experiment_filename(prefix="my_experiment")
            >>> print(exp_filename)
            my_experiment_exp.pkl
        """
        if prefix is None:
            return "experiment_exp.pkl"
        return f"{prefix}_exp.pkl"

    def _get_result_filename(self, prefix: str) -> str:
        """Generate result filename from prefix.

        Args:
            prefix (str): Prefix for the filename.

        Returns:
            str: Filename with '_res.pkl' suffix.

        Examples:
            >>> from spotoptim import SpotOptim
            >>> opt = SpotOptim(fun=lambda x: x, bounds=[(0, 1)])
            >>> res_filename = opt._get_result_filename(prefix="my_experiment")
            >>> print(res_filename)
            my_experiment_res.pkl
        """
        if prefix is None:
            return "result_res.pkl"
        return f"{prefix}_res.pkl"

    def _close_and_del_tensorboard_writer(self) -> None:
        """Close and delete TensorBoard writer to prepare for pickling.

        Examples:
            >>> from spotoptim import SpotOptim
            >>> opt = SpotOptim(fun=lambda x: x, bounds=[(0, 1)])
            >>> # Assume tb_writer is initialized
            >>> opt.tb_writer = SomeTensorBoardWriter()
            >>> # Close and delete tb_writer before pickling
            >>> opt._close_and_del_tensorboard_writer()
        """
        if hasattr(self, "tb_writer") and self.tb_writer is not None:
            try:
                self.tb_writer.flush()
                self.tb_writer.close()
            except Exception:
                pass
            self.tb_writer = None

    def _get_pickle_safe_optimizer(
        self, unpickleables: str = "file_io", verbosity: int = 0
    ) -> "SpotOptim":
        """Create a pickle-safe copy of the optimizer.

        This method creates a copy of the optimizer instance with unpickleable components removed
        or set to None to enable safe serialization.

        Args:
            unpickleables (str): Type of unpickleable components to exclude.
                - "file_io": Excludes only file I/O components (tb_writer) and fun
                - "all": Excludes file I/O, fun, surrogate, and lhs_sampler
                Defaults to "file_io".
            verbosity (int): Verbosity level (0=silent, 1=basic, 2=detailed). Defaults to 0.

        Returns:
            SpotOptim: A copy of the optimizer with unpickleable components removed.

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> # Define optimizer
            >>> opt = SpotOptim(
            ...     fun=lambda X: np.sum(X**2, axis=1),
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     max_iter=30,
            ...     n_initial=10,
            ...     seed=42
            ... )
            >>> # Create pickle-safe copy excluding all unpickleables
            >>> opt_safe = opt._get_pickle_safe_optimizer(unpickleables="all", verbosity=1)

        """
        # Always exclude tb_writer (can't reliably pickle file handles)
        # Determine which additional attributes to exclude
        if unpickleables == "file_io":
            unpickleable_attrs = ["tb_writer"]
        else:
            # "all" or specific exclusions
            unpickleable_attrs = ["tb_writer", "surrogate", "lhs_sampler"]

        # Prepare picklable state dictionary
        picklable_state = {}

        for key, value in self.__dict__.items():
            if key not in unpickleable_attrs:
                try:
                    # Test if attribute can be pickled
                    dill.dumps(value, protocol=dill.HIGHEST_PROTOCOL)
                    picklable_state[key] = value
                    if verbosity > 1:
                        print(f"Attribute '{key}' is picklable and will be included.")
                except Exception as e:
                    if verbosity > 0:
                        print(
                            f"Attribute '{key}' is not picklable and will be excluded: {e}"
                        )
                    continue
            else:
                if verbosity > 1:
                    print(f"Attribute '{key}' explicitly excluded from pickling.")

        # Create new instance with picklable state
        picklable_instance = self.__class__.__new__(self.__class__)
        picklable_instance.__dict__.update(picklable_state)

        # Set excluded attributes to None
        for attr in unpickleable_attrs:
            if not hasattr(picklable_instance, attr):
                setattr(picklable_instance, attr, None)

        return picklable_instance

    def save_experiment(
        self,
        filename: Optional[str] = None,
        prefix: str = "experiment",
        path: Optional[str] = None,
        overwrite: bool = True,
        unpickleables: str = "all",
        verbosity: int = 0,
    ) -> None:
        """Save the experiment configuration to a pickle file.

        An experiment contains the optimizer configuration needed to run optimization,
        but excludes the results. This is useful for defining experiments locally and
        executing them on remote machines.

        The experiment includes:
        - Bounds, variable types, variable names
        - Optimization parameters (max_iter, n_initial, etc.)
        - Surrogate and acquisition settings
        - Random seed

        The experiment excludes:
        - Function evaluations (X_, y_)
        - Optimization results


        Args:
            filename (str, optional): Filename for the experiment file. If None, generates
                from prefix. Defaults to None.
            prefix (str): Prefix for auto-generated filename. Defaults to "experiment".
            path (str, optional): Directory path to save the file. If None, saves in current
                directory. Creates directory if it doesn't exist. Defaults to None.
            overwrite (bool): If True, overwrites existing file. If False, raises error if
                file exists. Defaults to True.
            unpickleables (str): Components to exclude for pickling:
                - "all": Excludes surrogate, lhs_sampler, tb_writer (experiment only)
                - "file_io": Excludes only tb_writer (lighter exclusion)
                Defaults to "all".
            verbosity (int): Verbosity level (0=silent, 1=basic, 2=detailed). Defaults to 0.

        Returns:
            None

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>>
            >>> # Define experiment locally
            >>> opt = SpotOptim(
            ...     fun=lambda X: np.sum(X**2, axis=1),
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     max_iter=30,
            ...     n_initial=10,
            ...     seed=42
            ... )
            >>>
            >>> # Save experiment (without results)
            >>> opt.save_experiment(prefix="sphere_opt")
            Experiment saved to sphere_opt_exp.pkl
            >>>
            >>> # On remote machine: load and run
            >>> # opt_remote = SpotOptim.load_experiment("sphere_opt_exp.pkl")
            >>> # result = opt_remote.optimize()
            >>> # opt_remote.save_result(prefix="sphere_opt")  # Save results
        """
        # Close TensorBoard writer before pickling
        self._close_and_del_tensorboard_writer()

        # Create pickle-safe copy
        optimizer_copy = self._get_pickle_safe_optimizer(
            unpickleables=unpickleables, verbosity=verbosity
        )

        # Determine filename
        if filename is None:
            filename = self._get_experiment_filename(prefix)

        # Add path if provided
        if path is not None:
            if not os.path.exists(path):
                os.makedirs(path)
            filename = os.path.join(path, filename)

        # Check for existing file
        if os.path.exists(filename) and not overwrite:
            raise FileExistsError(
                f"File {filename} already exists. Use overwrite=True to overwrite."
            )

        # Save to pickle file
        try:
            with open(filename, "wb") as handle:
                dill.dump(optimizer_copy, handle, protocol=dill.HIGHEST_PROTOCOL)
            print(f"Experiment saved to {filename}")
        except Exception as e:
            print(f"Error during pickling: {e}")
            raise

    def save_result(
        self,
        filename: Optional[str] = None,
        prefix: str = "result",
        path: Optional[str] = None,
        overwrite: bool = True,
        verbosity: int = 0,
    ) -> None:
        """Save the complete optimization results to a pickle file.

        A result contains all information from a completed optimization run, including
        the experiment configuration and all evaluation results. This is useful for
        saving completed runs for later analysis.

        The result includes everything in an experiment plus:
        - All evaluated points (X_)
        - All function values (y_)
        - Best point and best value
        - Iteration count
        - Success rate statistics
        - Noise statistics (if applicable)

        Args:
            filename (str, optional): Filename for the result file. If None, generates
                from prefix. Defaults to None.
            prefix (str): Prefix for auto-generated filename. Defaults to "result".
            path (str, optional): Directory path to save the file. If None, saves in current
                directory. Creates directory if it doesn't exist. Defaults to None.
            overwrite (bool): If True, overwrites existing file. If False, raises error if
                file exists. Defaults to True.
            verbosity (int): Verbosity level (0=silent, 1=basic, 2=detailed). Defaults to 0.

        Returns:
            None

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>>
            >>> # Run optimization
            >>> opt = SpotOptim(
            ...     fun=lambda X: np.sum(X**2, axis=1),
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     max_iter=30,
            ...     n_initial=10,
            ...     seed=42
            ... )
            >>> result = opt.optimize()
            >>>
            >>> # Save complete results
            >>> opt.save_result(prefix="sphere_opt")
            Result saved to sphere_opt_res.pkl
            >>>
            >>> # Later: load and analyze
            >>> # opt_loaded = SpotOptim.load_result("sphere_opt_res.pkl")
            >>> # print("Best value:", opt_loaded.best_y_)
            >>> # opt_loaded.plot_surrogate()
        """
        # Use save_experiment with file_io unpickleables to preserve results
        if filename is None:
            filename = self._get_result_filename(prefix)

        self.save_experiment(
            filename=filename,
            path=path,
            overwrite=overwrite,
            unpickleables="file_io",
            verbosity=verbosity,
        )

        # Update message
        if path is not None:
            full_path = os.path.join(path, filename)
        else:
            full_path = filename
        print(f"Result saved to {full_path}")

    def _reinitialize_components(self) -> None:
        """Reinitialize components that were excluded during pickling.

        This method recreates the surrogate model and LHS sampler that were
        excluded when saving an experiment or result.

        Returns:
            None

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>> # Load experiment
            >>> opt = SpotOptim.load_experiment("sphere_opt_exp.pkl")
            >>> # Reinitialize components
            >>> opt._reinitialize_components()
        """
        # Reinitialize LHS sampler if needed
        if not hasattr(self, "lhs_sampler") or self.lhs_sampler is None:
            self.lhs_sampler = LatinHypercube(d=self.n_dim, seed=self.seed)

        # Reinitialize surrogate if needed
        if not hasattr(self, "surrogate") or self.surrogate is None:
            kernel = ConstantKernel(1.0, (1e-2, 1e12)) * Matern(
                length_scale=1.0, length_scale_bounds=(1e-4, 1e2), nu=2.5
            )
            self.surrogate = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=100,
                random_state=self.seed,
                normalize_y=True,
            )

    @staticmethod
    def load_experiment(filename: str) -> "SpotOptim":
        """Load an experiment configuration from a pickle file.

        Loads an experiment that was saved with save_experiment(). The loaded optimizer
        will have the configuration and the objective function (thanks to dill).


        Args:
            filename (str): Path to the experiment pickle file.

        Returns:
            SpotOptim: Loaded optimizer instance (without fun attached).

        Raises:
            FileNotFoundError: If the specified file doesn't exist.

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>>
            >>> # Load experiment
            >>> opt = SpotOptim.load_experiment("sphere_opt_exp.pkl")
            Loaded experiment from sphere_opt_exp.pkl
            >>>
            >>> # Re-attach objective function
            >>> opt.fun = lambda X: np.sum(X**2, axis=1)
            >>>
            >>> # Run optimization
            >>> result = opt.optimize()
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Experiment file not found: {filename}")

        try:
            with open(filename, "rb") as handle:
                optimizer = dill.load(handle)
            print(f"Loaded experiment from {filename}")

            # Reinitialize components that were excluded
            optimizer._reinitialize_components()

            return optimizer
        except Exception as e:
            print(f"Error loading experiment: {e}")
            raise

    @staticmethod
    def load_result(filename: str) -> "SpotOptim":
        """Load complete optimization results from a pickle file.

        Loads results that were saved with save_result(). The loaded optimizer
        will have both configuration and all optimization results.

        Args:
            filename (str): Path to the result pickle file.

        Returns:
            SpotOptim: Loaded optimizer instance with complete results.

        Raises:
            FileNotFoundError: If the specified file doesn't exist.

        Examples:
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>>
            >>> # Load results
            >>> opt = SpotOptim.load_result("sphere_opt_res.pkl")
            Loaded result from sphere_opt_res.pkl
            >>>
            >>> # Analyze results
            >>> print("Best point:", opt.best_x_)
            >>> print("Best value:", opt.best_y_)
            >>> print("Total evaluations:", opt.counter)
            >>> print("Success rate:", opt.success_rate)
            >>>
            >>> # Continue optimization if needed
            >>> # opt.fun = lambda X: np.sum(X**2, axis=1)  # Re-attach if continuing
            >>> # opt.max_iter = 50  # Increase budget
            >>> # result = opt.optimize()
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Result file not found: {filename}")

        try:
            with open(filename, "rb") as handle:
                optimizer = dill.load(handle)
            print(f"Loaded result from {filename}")

            # Reinitialize components that were excluded
            optimizer._reinitialize_components()

            return optimizer
        except Exception as e:
            print(f"Error loading result: {e}")
            raise

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
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>>
            >>> # Example 1: Basic usage
            >>> def sphere(X):
            ...     return np.sum(X**2, axis=1)
            >>> opt = SpotOptim(
            ...     fun=sphere,
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     var_name=["x1", "x2"],
            ...     max_iter=20,
            ...     n_initial=10
            ... )
            >>> result = opt.optimize()
            >>> opt.print_best(result)
            <BLANKLINE>
            Best Solution Found:
            --------------------------------------------------
              x1: 0.0123
              x2: -0.0045
              Objective Value: 0.000173
              Total Evaluations: 20
            >>>
            >>> # Example 2: With log-scale transformations (e.g., for learning rates)
            >>> def objective(X):
            ...     # X[:, 0]: neurons (int), X[:, 1]: layers (int),
            ...     # X[:, 2]: log10(lr), X[:, 3]: log10(alpha)
            ...     return np.sum(X**2, axis=1)  # Placeholder
            >>> opt = SpotOptim(
            ...     fun=objective,
            ...     bounds=[(16, 128), (1, 4), (-3, 0), (-2, 1)],
            ...     var_type=["int", "int", "float", "float"],
            ...     var_name=["neurons", "layers", "log10_lr", "log10_alpha"],
            ...     max_iter=30,
            ...     n_initial=10
            ... )
            >>> result = opt.optimize()
            >>> # Transform log-scale parameters back to original scale
            >>> transformations = [
            ...     int,              # neurons -> int
            ...     int,              # layers -> int
            ...     lambda x: 10**x,  # log10_lr -> lr
            ...     lambda x: 10**x   # log10_alpha -> alpha
            ... ]
            >>> opt.print_best(result, transformations=transformations)
            <BLANKLINE>
            Best Solution Found:
            --------------------------------------------------
              neurons: 64
              layers: 2
              log10_lr: 0.0012
              log10_alpha: 0.0345
              Objective Value: 1.2345
              Total Evaluations: 30
            >>>
            >>> # Example 3: Without result object (using stored values)
            >>> opt.print_best()  # Uses opt.best_x_ and opt.best_y_
            >>>
            >>> # Example 4: Hide variable names
            >>> opt.print_best(result, show_name=False)
            <BLANKLINE>
            Best Solution Found:
            --------------------------------------------------
              x0: 0.0123
              x1: -0.0045
              Objective Value: 0.000173
              Total Evaluations: 20
        """
        # Get values from result or stored attributes
        if result is not None:
            best_x = result.x
            best_y = result.fun
            n_evals = result.nfev
        else:
            if self.best_x_ is None or self.best_y_ is None:
                print("No optimization results available. Run optimize() first.")
                return
            best_x = self.best_x_
            best_y = self.best_y_
            n_evals = self.counter

        # Expand to full dimensions if dimension reduction was applied
        if self.red_dim:
            best_x_full = self.to_all_dim(best_x.reshape(1, -1))[0]
        else:
            best_x_full = best_x

        # Map factor variables back to original string values
        best_x_full = self._map_to_factor_values(best_x_full.reshape(1, -1))[0]

        # Determine variable names to use
        if show_name and self.all_var_name is not None:
            var_names = self.all_var_name
        else:
            var_names = [f"x{i}" for i in range(len(best_x_full))]

        # Validate transformations length
        if transformations is not None:
            if len(transformations) != len(best_x_full):
                raise ValueError(
                    f"Length of transformations ({len(transformations)}) must match "
                    f"number of dimensions ({len(best_x_full)})"
                )
        else:
            transformations = [None] * len(best_x_full)

        # Print header
        print("\nBest Solution Found:")
        print("-" * 50)

        # Print each parameter
        for i, (name, value, transform) in enumerate(
            zip(var_names, best_x_full, transformations)
        ):
            # Apply transformation if provided
            if transform is not None:
                try:
                    display_value = transform(value)
                except Exception as e:
                    print(f"Warning: Transformation failed for {name}: {e}")
                    display_value = value
            else:
                display_value = value

            # Format based on variable type
            var_type = self.all_var_type[i] if i < len(self.all_var_type) else "float"

            if var_type == "int" or isinstance(display_value, (int, np.integer)):
                print(f"  {name}: {int(display_value)}")
            elif var_type == "factor" or isinstance(display_value, str):
                print(f"  {name}: {display_value}")
            else:
                print(f"  {name}: {display_value:.{precision}f}")

        # Print objective value and evaluations
        print(f"  Objective Value: {best_y:.{precision}f}")
        print(f"  Total Evaluations: {n_evals}")

    def sensitivity_spearman(self) -> None:
        """Compute and print Spearman correlation between parameters and objective values.

        This method analyzes the sensitivity of the objective function to each
        hyperparameter by computing Spearman rank correlations. For categorical
        (factor) variables, correlation is not computed as they require visual
        inspection instead.

        The method automatically handles different parameter types:
        - Integer/float parameters: Direct correlation with objective values
        - Log-transformed parameters (log10, log, ln): Correlation in log-space
        - Factor (categorical) parameters: Skipped with informative message

        Significance levels:
        - ***: p < 0.001 (highly significant)
        - **: p < 0.01 (significant)
        - *: p < 0.05 (marginally significant)

        Examples:
            >>> from spotoptim import SpotOptim
            >>> import numpy as np
            >>>
            >>> # After running optimization
            >>> opt = SpotOptim(...)
            >>> result = opt.optimize()
            >>> opt.sensitivity_spearman()
            Sensitivity Analysis (Spearman Correlation):
            --------------------------------------------------
              l1 (neurons)        : +0.005 (p=0.959)
              num_layers          : -0.192 (p=0.056)
              activation          : (categorical variable, use visual inspection)
              lr_unified          : -0.040 (p=0.689)
              alpha               : -0.233 (p=0.020) *

        Note:
            Requires scipy to be installed. If not available, raises ImportError.
            Only meaningful after optimize() has been called with sufficient evaluations.
        """
        try:
            from scipy.stats import spearmanr
        except ImportError:
            raise ImportError(
                "scipy is required for sensitivity_spearman(). "
                "Install it with: pip install scipy"
            )

        if self.X_ is None or self.y_ is None:
            raise ValueError("No optimization data available. Run optimize() first.")

        # Get optimization history and parameters
        history = self.y_
        all_params = self.X_

        # Get parameter names
        param_names = (
            self.var_name if self.var_name else [f"x{i}" for i in range(self.n_dim)]
        )

        print("\nSensitivity Analysis (Spearman Correlation):")
        print("-" * 50)

        for param_idx in range(self.n_dim):
            name = param_names[param_idx]
            param_values = all_params[:, param_idx]

            # Check if it's a factor variable
            var_type = self.var_type[param_idx] if self.var_type else "float"

            if var_type == "factor":
                # For categorical variables, skip correlation
                print(f"  {name:20s}: (categorical variable, use visual inspection)")
                continue

            # Check if parameter has log transformation
            var_trans = self.var_trans[param_idx] if self.var_trans else None

            # Compute correlation based on transformation
            if var_trans in ["log10", "log", "ln"]:
                # For log-transformed parameters, use log-space correlation
                try:
                    param_values_numeric = param_values.astype(float)
                    # Filter out non-positive values
                    valid_mask = (param_values_numeric > 0) & (history > 0)
                    if valid_mask.sum() < 3:
                        print(
                            f"  {name:20s}: (insufficient valid data for log correlation)"
                        )
                        continue

                    corr, p_value = spearmanr(
                        np.log10(param_values_numeric[valid_mask]),
                        np.log10(history[valid_mask]),
                    )
                except (ValueError, TypeError):
                    print(f"  {name:20s}: (error computing log correlation)")
                    continue
            else:
                # For integer/float parameters, direct correlation
                try:
                    param_values_numeric = param_values.astype(float)
                    corr, p_value = spearmanr(param_values_numeric, history)
                except (ValueError, TypeError):
                    print(f"  {name:20s}: (error computing correlation)")
                    continue

            # Determine significance level
            if p_value < 0.001:
                significance = " ***"
            elif p_value < 0.01:
                significance = " **"
            elif p_value < 0.05:
                significance = " *"
            else:
                significance = ""

            print(f"  {name:20s}: {corr:+.3f} (p={p_value:.3f}){significance}")

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
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>>
            >>> # Example 1: Basic usage after optimization
            >>> def sphere(X):
            ...     return np.sum(X**2, axis=1)
            >>> opt = SpotOptim(
            ...     fun=sphere,
            ...     bounds=[(-5, 5), (-5, 5), (-5, 5)],
            ...     var_name=["x1", "x2", "x3"],
            ...     var_type=["float", "float", "float"],
            ...     max_iter=30,
            ...     n_initial=10
            ... )
            >>> result = opt.optimize()
            >>> table = opt.get_results_table()
            >>> print(table)
            | name   | type   |   lower |   upper |   tuned |
            |--------|--------|---------|---------|---------|
            | x1     | num    |    -5.0 |     5.0 |  0.0123 |
            | x2     | num    |    -5.0 |     5.0 | -0.0234 |
            | x3     | num    |    -5.0 |     5.0 |  0.0345 |
            >>>
            >>> # Example 2: With importance scores
            >>> table = opt.get_results_table(show_importance=True)
            >>> print(table)
            | name   | type   |   lower |   upper |   tuned |   importance | stars   |
            |--------|--------|---------|---------|---------|--------------|---------|
            | x1     | num    |    -5.0 |     5.0 |  0.0123 |        45.23 | **      |
            | x2     | num    |    -5.0 |     5.0 | -0.0234 |        32.17 | *       |
            | x3     | num    |    -5.0 |     5.0 |  0.0345 |        22.60 | *       |
            >>>
            >>> # Example 3: Different table format
            >>> table = opt.get_results_table(tablefmt="grid")
            >>> print(table)
            +--------+--------+---------+---------+---------+
            | name   | type   |   lower |   upper |   tuned |
            +========+========+=========+=========+=========+
            | x1     | num    |    -5.0 |     5.0 |  0.0123 |
            +--------+--------+---------+---------+---------+
            ...
            >>>
            >>> # Example 4: With factor variables
            >>> opt = SpotOptim(
            ...     fun=lambda X: np.sum(X**2, axis=1),
            ...     bounds=[(-5, 5), ("red", "green", "blue")],
            ...     var_name=["size", "color"],
            ...     var_type=["float", "factor"],
            ...     max_iter=20,
            ...     n_initial=10
            ... )
            >>> result = opt.optimize()
            >>> table = opt.get_results_table()
            >>> print(table)
            | name   | type   | lower   | upper   | tuned   |
            |--------|--------|---------|---------|---------|
            | size   | num    | -5.0    | 5.0     | 0.0123  |
            | color  | factor | red     | blue    | green   |
        """
        if self.best_x_ is None or self.best_y_ is None:
            return "No optimization results available. Run optimize() first."

        # Get best solution in full dimensions
        # Note: best_x_ is already in original scale
        if self.red_dim:
            best_x_full = self.to_all_dim(self.best_x_.reshape(1, -1))[0]
        else:
            best_x_full = self.best_x_

        # Map factor variables back to original string values
        best_x_display = self._map_to_factor_values(best_x_full.reshape(1, -1))[0]

        # Prepare all variable transformations (use all_var_trans if dimension reduction occurred)
        if self.red_dim and hasattr(self, "all_var_trans"):
            all_var_trans = self.all_var_trans
        else:
            all_var_trans = self.var_trans

        # Prepare table data
        table_data = {
            "name": (
                self.all_var_name
                if self.all_var_name
                else [f"x{i}" for i in range(len(best_x_display))]
            ),
            "type": (
                self.all_var_type
                if self.all_var_type
                else ["float"] * len(best_x_display)
            ),
            "default": [],
            "lower": [],
            "upper": [],
            "tuned": [],
            "transform": [t if t is not None else "-" for t in all_var_trans],
        }

        # Helper to format values
        def fmt_val(v):
            if isinstance(v, (float, np.floating)):
                return f"{v:.{precision}f}"
            return v

        # Process bounds, defaults, and tuned values
        for i in range(len(best_x_display)):
            var_type = table_data["type"][i]

            # Handle bounds and defaults based on variable type
            if var_type == "factor":
                # For factors, show original string values
                if i in self._factor_maps:
                    factor_map = self._factor_maps[i]
                    # Default is middle level logic (matching get_design_table)
                    mid_idx = len(factor_map) // 2
                    default_str = factor_map[mid_idx]

                    table_data["lower"].append("-")
                    table_data["upper"].append("-")
                    table_data["default"].append(default_str)
                else:
                    table_data["lower"].append("-")
                    table_data["upper"].append("-")
                    table_data["default"].append("N/A")
            else:
                table_data["lower"].append(fmt_val(self._original_lower[i]))
                table_data["upper"].append(fmt_val(self._original_upper[i]))
                # Default is midpoint logic
                default_val = (self._original_lower[i] + self._original_upper[i]) / 2
                if var_type == "int":
                    table_data["default"].append(int(default_val))
                else:
                    table_data["default"].append(fmt_val(default_val))

            # Format tuned value
            tuned_val = best_x_display[i]
            if var_type == "int":
                table_data["tuned"].append(int(tuned_val))
            elif var_type == "factor":
                table_data["tuned"].append(str(tuned_val))
            else:
                table_data["tuned"].append(fmt_val(tuned_val))

        # Add importance if requested
        if show_importance:
            importance = self.get_importance()
            table_data["importance"] = [f"{x:.2f}" for x in importance]
            table_data["stars"] = self.get_stars(importance)

        # Generate table
        table = tabulate(
            table_data,
            headers="keys",
            tablefmt=tablefmt,
            numalign="right",
            stralign="right",
        )

        # Add interpretation if importance is shown
        if show_importance:
            table += "\n\nInterpretation: ***: >99%, **: >75%, *: >50%, .: >10%"

        return table

    def print_results_table(
        self,
        tablefmt: str = "github",
        precision: int = 4,
        show_importance: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        """Print (and return) a comprehensive table of optimization results.

        This method calls `get_results_table` to generate the table string, prints it,
        and then returns it.

        Args:
            tablefmt (str, optional): Table format. Defaults to 'github'.
            precision (int, optional): Decimal precision. Defaults to 4.
            show_importance (bool, optional): Show importance column. Defaults to False.
            *args: Arguments passed to get_results_table.
            **kwargs: Keyword arguments passed to get_results_table.

        Returns:
            str: Formatted table string.
        """
        table = self.get_results_table(
            tablefmt=tablefmt,
            precision=precision,
            show_importance=show_importance,
            *args,
            **kwargs,
        )
        print(table)
        return table

    def print_results(self, *args: Any, **kwargs: Any) -> None:
        """Alias for print_results_table for compatibility.
        Prints the table.
        """
        self.print_results_table(*args, **kwargs)

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
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>>
            >>> # Example 1: Numeric parameters
            >>> opt = SpotOptim(
            ...     fun=lambda X: np.sum(X**2, axis=1),
            ...     bounds=[(-5, 5), (-10, 10), (0, 1)],
            ...     var_name=["x1", "x2", "x3"],
            ...     var_type=["float", "int", "float"],
            ...     max_iter=20,
            ...     n_initial=10
            ... )
            >>> table = opt.get_design_table()
            >>> print(table)
            | name   | type   |   lower |   upper |   default |
            |--------|--------|---------|---------|-----------|
            | x1     | num    |    -5.0 |     5.0 |       0.0 |
            | x2     | int    |   -10.0 |    10.0 |       0.0 |
            | x3     | num    |     0.0 |     1.0 |       0.5 |
            >>>
            >>> # Example 2: With factor variables
            >>> opt = SpotOptim(
            ...     fun=lambda X: np.sum(X**2, axis=1),
            ...     bounds=[(10, 100), ("SGD", "Adam", "RMSprop"), (0.001, 0.1)],
            ...     var_name=["neurons", "optimizer", "lr"],
            ...     var_type=["int", "factor", "float"],
            ...     max_iter=30,
            ...     n_initial=10
            ... )
            >>> table = opt.get_design_table()
            >>> print(table)
            | name      | type   | lower   | upper   | default   |
            |-----------|--------|---------|---------|-----------|
            | neurons   | int    | 10.0    | 100.0   | 55.0      |
            | optimizer | factor | SGD     | RMSprop | Adam      |
            | lr        | num    | 0.001   | 0.1     | 0.0505    |
            >>>
            >>> # Example 3: Before running optimization
            >>> def hyperparameter_objective(X):
            ...     # X[:, 0]: layers, X[:, 1]: neurons, X[:, 2]: dropout
            ...     return np.sum(X**2, axis=1)  # Placeholder
            >>> opt = SpotOptim(
            ...     fun=hyperparameter_objective,
            ...     bounds=[(1, 5), (16, 256), (0.0, 0.5)],
            ...     var_name=["layers", "neurons", "dropout"],
            ...     var_type=["int", "int", "float"],
            ...     max_iter=50,
            ...     n_initial=15
            ... )
            >>> # Get design table before optimization
            >>> print("Search Space Configuration:")
            >>> table = opt.get_design_table()
            >>> print(table)
            Search Space Configuration:
            | name    | type   |   lower |   upper |   default |
            |---------|--------|---------|---------|-----------|
            | layers  | int    |     1.0 |     5.0 |       3.0 |
            | neurons | int    |    16.0 |   256.0 |     136.0 |
            | dropout | num    |     0.0 |     0.5 |      0.25 |
        """
        # Prepare all variable transformations (use all_var_trans if dimension reduction occurred)
        if self.red_dim and hasattr(self, "all_var_trans"):
            all_var_trans = self.all_var_trans
        else:
            all_var_trans = self.var_trans

        # Prepare table data
        table_data = {
            "name": (
                self.all_var_name
                if self.all_var_name
                else [f"x{i}" for i in range(len(self.all_lower))]
            ),
            "type": (
                self.all_var_type
                if self.all_var_type
                else ["float"] * len(self.all_lower)
            ),
            "lower": [],
            "upper": [],
            "default": [],
            "transform": [t if t is not None else "-" for t in all_var_trans],
        }

        # Helper to format values
        def fmt_val(v):
            if isinstance(v, (float, np.floating)):
                return f"{v:.{precision}f}"
            return v

        # Process bounds and compute defaults (use original bounds for display)
        for i in range(len(self._original_lower)):
            var_type = table_data["type"][i]

            if var_type == "factor":
                # For factors, show original string values
                if i in self._factor_maps:
                    factor_map = self._factor_maps[i]
                    # Default is middle level
                    mid_idx = len(factor_map) // 2
                    default_str = factor_map[mid_idx]
                    table_data["lower"].append("-")
                    table_data["upper"].append("-")
                    table_data["default"].append(default_str)
                else:
                    table_data["lower"].append("-")
                    table_data["upper"].append("-")
                    table_data["default"].append("N/A")
            else:
                table_data["lower"].append(fmt_val(self._original_lower[i]))
                table_data["upper"].append(fmt_val(self._original_upper[i]))
                # Default is midpoint
                default_val = (self._original_lower[i] + self._original_upper[i]) / 2
                if var_type == "int":
                    table_data["default"].append(int(default_val))
                else:
                    table_data["default"].append(fmt_val(default_val))

        # Generate table
        table = tabulate(
            table_data,
            headers="keys",
            tablefmt=tablefmt,
            numalign="right",
            stralign="right",
        )

        return table

    def print_design_table(
        self,
        tablefmt: str = "github",
        precision: int = 4,
    ) -> str:
        """Print (and return) a table showing the search space design before optimization.

        This method calls `get_design_table` to generate the table string, prints it,
        and then returns it.

        Args:
            tablefmt (str, optional): Table format for tabulate library.
                Defaults to 'github'.
            precision (int, optional): Number of decimal places for float values.
                Defaults to 4.

        Returns:
            str: Formatted table string.
        """
        table = self.get_design_table(tablefmt=tablefmt, precision=precision)
        print(table)
        return table

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
        """
        if self.best_x_ is not None:
            return self.get_results_table(precision=precision, tablefmt=tablefmt)
        else:
            return self.get_design_table(precision=precision, tablefmt=tablefmt)

    def get_stars(self, input_list: list) -> list:
        """Converts a list of values to a list of stars.

        Used to visualize the importance of a variable.
        Thresholds: >99: ***, >75: **, >50: *, >10: .

        Args:
            input_list (list): A list of importance scores (0-100).

        Returns:
            list: A list of star strings.
        """
        output_list = []
        for value in input_list:
            if value > 99:
                output_list.append("***")
            elif value > 75:
                output_list.append("**")
            elif value > 50:
                output_list.append("*")
            elif value > 10:
                output_list.append(".")
            else:
                output_list.append("")
        return output_list

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
            >>> import numpy as np
            >>> from spotoptim import SpotOptim
            >>>
            >>> # Example 1: Identify important parameters
            >>> def test_func(X):
            ...     # x0 has strong effect, x1 has weak effect
            ...     return 10 * X[:, 0]**2 + 0.1 * X[:, 1]**2
            >>> opt = SpotOptim(
            ...     fun=test_func,
            ...     bounds=[(-5, 5), (-5, 5)],
            ...     var_name=["x0", "x1"],
            ...     max_iter=30,
            ...     n_initial=10,
            ...     seed=42
            ... )
            >>> result = opt.optimize()
            >>> importance = opt.get_importance()
            >>> print(f"x0 importance: {importance[0]:.2f}")
            >>> print(f"x1 importance: {importance[1]:.2f}")
            x0 importance: 89.23
            x1 importance: 10.77
            >>>
            >>> # Example 2: With more dimensions
            >>> def rosenbrock(X):
            ...     return np.sum(100*(X[:, 1:] - X[:, :-1]**2)**2 + (1 - X[:, :-1])**2, axis=1)
            >>> opt = SpotOptim(
            ...     fun=rosenbrock,
            ...     bounds=[(-2, 2)] * 4,
            ...     var_name=["x0", "x1", "x2", "x3"],
            ...     max_iter=50,
            ...     n_initial=20,
            ...     seed=42
            ... )
            >>> result = opt.optimize()
            >>> importance = opt.get_importance()
            >>> for i, imp in enumerate(importance):
            ...     print(f"x{i}: {imp:.2f}%")
            x0: 32.15%
            x1: 28.43%
            x2: 25.67%
            x3: 13.75%
            >>>
            >>> # Example 3: Use in results table
            >>> table = opt.print_results_table(show_importance=True)
            >>> print(table)
        """
        if self.X_ is None or self.y_ is None or len(self.y_) < 3:
            # Not enough data to compute importance
            return [0.0] * len(self.all_lower)

        # Use full-dimensional data
        X_full = self.X_
        if self.red_dim:
            X_full = np.array([self.to_all_dim(x.reshape(1, -1))[0] for x in self.X_])

        # Calculate sensitivity for each dimension
        sensitivities = []
        for i in range(X_full.shape[1]):
            x_i = X_full[:, i]

            # Skip if no variation in this dimension
            if np.std(x_i) < 1e-10:
                sensitivities.append(0.0)
                continue

            # Compute correlation with objective
            try:
                correlation = np.abs(np.corrcoef(x_i, self.y_)[0, 1])
                if np.isnan(correlation):
                    correlation = 0.0
            except Exception:
                correlation = 0.0

            sensitivities.append(correlation)

        # Normalize to percentage
        total = sum(sensitivities)
        if total > 0:
            importance = [(s / total) * 100 for s in sensitivities]
        else:
            importance = [0.0] * len(sensitivities)

        return importance

    def plot_importance(
        self, threshold: float = 0.0, figsize: Tuple[int, int] = (10, 6)
    ) -> None:
        """Plot variable importance.

        Args:
            threshold (float): Minimum importance percentage to include in plot.
            figsize (tuple): Figure size.
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
