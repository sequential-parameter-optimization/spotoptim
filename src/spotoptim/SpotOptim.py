import numpy as np
from typing import Callable, Optional, Tuple, List
from scipy.optimize import OptimizeResult, differential_evolution
from scipy.stats.qmc import LatinHypercube
from sklearn.base import BaseEstimator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
import warnings
import matplotlib.pyplot as plt
from numpy import linspace, meshgrid, append
import time
from spotpython.budget.ocba import get_ocba_X
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import shutil


class SpotOptim(BaseEstimator):
    """SPOT optimizer compatible with scipy.optimize interface.

    Args:
        fun (callable): Objective function to minimize. Should accept array of shape (n_samples, n_features).
        bounds (list of tuple): Bounds for each dimension as [(low, high), ...].
        max_iter (int, optional): Maximum number of total function evaluations (including initial design).
            For example, max_iter=30 with n_initial=10 will perform 10 initial evaluations plus
            20 sequential optimization iterations. Defaults to 20.
        n_initial (int, optional): Number of initial design points. Defaults to 10.
        surrogate (object, optional): Surrogate model. Defaults to Gaussian Process with Matern kernel.
        acquisition (str, optional): Acquisition function ('ei', 'y', 'pi'). Defaults to 'ei'.
        var_type (list of str, optional): Variable types for each dimension. Supported types:
            - 'float': Python floats, continuous optimization (no rounding)
            - 'int': Python int, float values will be rounded to integers
            - 'factor': Unordered categorical data, internally mapped to int values
              (e.g., "red"->0, "green"->1, etc.)
            Defaults to None (which sets all dimensions to 'float').
        var_name (list of str, optional): Variable names for each dimension.
            If None, uses default names ['x0', 'x1', 'x2', ...]. Defaults to None.
        tolerance_x (float, optional): Minimum distance between points. Defaults to np.sqrt(np.spacing(1))
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

    Attributes:
        X_ (ndarray): All evaluated points, shape (n_samples, n_features).
        y_ (ndarray): Function values at X_, shape (n_samples,). For multi-objective problems,
            these are the converted single-objective values.
        y_mo (ndarray or None): Multi-objective function values, shape (n_samples, n_objectives).
            None for single-objective problems.
        best_x_ (ndarray): Best point found, shape (n_features,).
        best_y_ (float): Best function value found.
        n_iter_ (int): Number of iterations performed.
        warnings_filter (str): Filter for warnings during optimization.
        max_surrogate_points (int or None): Maximum number of points for surrogate fitting.
        selection_method (str): Point selection method.
        noise (bool): True if noise handling is active (repeats > 1).
        mean_X (ndarray or None): Aggregated unique design points (if noise=True).
        mean_y (ndarray or None): Mean y values per design point (if noise=True).
        var_y (ndarray or None): Variance of y values per design point (if noise=True).
        min_mean_X (ndarray or None): X value of best mean y (if noise=True).
        min_mean_y (float or None): Best mean y value (if noise=True).
        min_var_y (float or None): Variance of best mean y (if noise=True).

    Examples:
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
    """

    def __init__(
        self,
        fun: Callable,
        bounds: list,
        max_iter: int = 20,
        n_initial: int = 10,
        surrogate: Optional[object] = None,
        acquisition: str = "ei",
        var_type: Optional[list] = None,
        var_name: Optional[list] = None,
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
    ):

        warnings.filterwarnings(warnings_filter)

        # small value, converted to float
        self.eps = np.sqrt(np.spacing(1))

        if tolerance_x is None:
            self.tolerance_x = self.eps
        else:
            self.tolerance_x = tolerance_x

        # Validate parameters
        if max_iter < n_initial:
            raise ValueError(
                f"max_iter ({max_iter}) must be >= n_initial ({n_initial}). "
                f"max_iter represents the total function evaluation budget including initial design."
            )
        
        self.fun = fun
        self.bounds = bounds
        self.max_iter = max_iter
        self.n_initial = n_initial
        self.surrogate = surrogate
        self.acquisition = acquisition
        self.var_type = var_type
        self.var_name = var_name
        self.max_time = max_time
        self.repeats_initial = repeats_initial
        self.repeats_surrogate = repeats_surrogate
        self.ocba_delta = ocba_delta
        self.tensorboard_log = tensorboard_log
        self.tensorboard_path = tensorboard_path
        self.tensorboard_clean = tensorboard_clean
        self.fun_mo2so = fun_mo2so
        self.seed = seed
        self.verbose = verbose
        self.max_surrogate_points = max_surrogate_points
        self.selection_method = selection_method
        
        # Determine if noise handling is active
        self.noise = (repeats_initial > 1) or (repeats_surrogate > 1)

        # Derived attributes
        self.n_dim = len(bounds)
        self.lower = np.array([b[0] for b in bounds])
        self.upper = np.array([b[1] for b in bounds])

        # Default variable types
        if self.var_type is None:
            self.var_type = ["float"] * self.n_dim

        # Default variable names
        if self.var_name is None:
            self.var_name = [f"x{i}" for i in range(self.n_dim)]

        # Dimension reduction: backup original bounds and identify fixed dimensions
        self._setup_dimension_reduction()

        # Initialize surrogate if not provided
        if self.surrogate is None:
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
                length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5
            )
            self.surrogate = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=10,
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
        
        # Clean old TensorBoard logs if requested
        self._clean_tensorboard_logs()
        
        # Initialize TensorBoard writer
        self._init_tensorboard_writer()

    def _setup_dimension_reduction(self) -> None:
        """Set up dimension reduction by identifying fixed dimensions.

        This method identifies dimensions where lower and upper bounds are equal,
        indicating fixed (constant) variables. It stores:
        - Original bounds and metadata in `all_*` attributes
        - Boolean mask of fixed dimensions in `ident`
        - Reduced bounds, types, and names for optimization
        - `red_dim` flag indicating if reduction occurred
        """
        # Backup original values
        self.all_lower = self.lower.copy()
        self.all_upper = self.upper.copy()
        self.all_var_type = self.var_type.copy()
        self.all_var_name = self.var_name.copy()

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
                vtype for vtype, fixed in zip(self.all_var_type, self.ident) if not fixed
            ]
            self.var_name = [
                vname for vname, fixed in zip(self.all_var_name, self.ident) if not fixed
            ]

            # Update bounds list for reduced dimensions
            self.bounds = [(self.lower[i], self.upper[i]) for i in range(self.n_dim)]

            # Recreate LHS sampler with reduced dimensions
            self.lhs_sampler = LatinHypercube(d=self.n_dim, seed=self.seed)

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

        # Select only non-fixed dimensions
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

    def update_stats(self) -> None:
        """Update optimization statistics.

        Updates:
        1. `min_y`: Minimum y value found so far
        2. `min_X`: X value corresponding to minimum y
        3. `counter`: Total number of function evaluations

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
                        print(f"Cleaned {removed_count} old TensorBoard log director{'y' if removed_count == 1 else 'ies'}")
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
                "y_values",
                {"min": self.min_y, "last": y_last},
                step
            )
            # Log best X coordinates
            for i in range(self.n_dim):
                self.tb_writer.add_scalar(
                    f"X_best/x{i}",
                    self.min_X[i],
                    step
                )
        else:
            # Noisy optimization
            self.tb_writer.add_scalars(
                "y_values",
                {"min": self.min_y, "mean_best": self.min_mean_y, "last": y_last},
                step
            )
            # Log variance of best mean
            self.tb_writer.add_scalar("y_variance_at_best", self.min_var_y, step)
            
            # Log best X coordinates (by mean)
            for i in range(self.n_dim):
                self.tb_writer.add_scalar(
                    f"X_mean_best/x{i}",
                    self.min_mean_X[i],
                    step
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
        if hasattr(self, 'tb_writer') and self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()
            if self.verbose:
                print(f"TensorBoard writer closed. View logs with: tensorboard --logdir={self.tensorboard_path}")
            del self.tb_writer

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

    def _evaluate_function(self, X: np.ndarray) -> np.ndarray:
        """Evaluate objective function at points X.

        If dimension reduction is active, expands X to full dimensions before evaluation.
        Supports both single-objective and multi-objective functions. For multi-objective
        functions, converts to single-objective using _mo2so method.

        Args:
            X (ndarray): Points to evaluate in reduced space, shape (n_samples, n_reduced_features).

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

        # Evaluate function
        y_raw = self.fun(X)

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

        If the number of points exceeds `self.max_surrogate_points`,
        a subset of points is selected using the selection dispatcher.

        Args:
            X (ndarray): Design points, shape (n_samples, n_features).
            y (ndarray): Function values at X, shape (n_samples,).
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

    def _select_new(
        self, A: np.ndarray, X: np.ndarray, tolerance: float = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select rows from A that are not in X.

        Args:
            A (ndarray): Array with new values.
            X (ndarray): Array with known values.
            tolerance (float, optional): Tolerance value for comparison. Defaults to 0.

        Returns:
            tuple: A tuple containing:
                - ndarray: Array with unknown (new) values.
                - ndarray: Array with True if value is new, otherwise False.
        """
        B = np.abs(A[:, None] - X)
        ind = np.any(np.all(B <= tolerance, axis=2), axis=1)
        return A[~ind], ~ind

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
        """
        mask = np.isin(var_type, ["float"], invert=True)
        X[:, mask] = np.around(X[:, mask])
        return X

    def _acquisition_function(self, x: np.ndarray) -> float:
        """Compute acquisition function value.

        Args:
            x (ndarray): Point to evaluate, shape (n_features,).

        Returns:
            float: Acquisition function value (to be minimized).
        """
        x = x.reshape(1, -1)

        if self.acquisition == "y":
            # Predicted mean
            return self.surrogate.predict(x)[0]

        elif self.acquisition == "ei":
            # Expected Improvement
            mu, sigma = self.surrogate.predict(x, return_std=True)
            mu = mu[0]
            sigma = sigma[0]

            if sigma < 1e-10:
                return 0.0

            y_best = np.min(self.y_)
            improvement = y_best - mu
            Z = improvement / sigma

            from scipy.stats import norm

            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            return -ei  # Minimize negative EI

        elif self.acquisition == "pi":
            # Probability of Improvement
            mu, sigma = self.surrogate.predict(x, return_std=True)
            mu = mu[0]
            sigma = sigma[0]

            if sigma < 1e-10:
                return 0.0

            y_best = np.min(self.y_)
            Z = (y_best - mu) / sigma

            from scipy.stats import norm

            pi = norm.cdf(Z)
            return -pi  # Minimize negative PI

        else:
            raise ValueError(f"Unknown acquisition function: {self.acquisition}")

    def _suggest_next_point(self) -> np.ndarray:
        """Suggest next point to evaluate using acquisition function optimization.

        Returns:
            ndarray: Next point to evaluate, shape (n_features,).
        """
        result = differential_evolution(
            func=self._acquisition_function,
            bounds=self.bounds,
            seed=self.seed,
            maxiter=1000,
        )

        x_next = result.x

        # Ensure minimum distance to existing points
        x_next_2d = x_next.reshape(1, -1)
        x_new, _ = self._select_new(A=x_next_2d, X=self.X_, tolerance=self.tolerance_x)

        if x_new.shape[0] == 0:
            # If too close, generate random point
            if self.verbose:
                print("Proposed point too close, generating random point")
            # Generate a random point using LHS
            x_next_unit = self.lhs_sampler.random(n=1)[0]
            x_next = self.lower + x_next_unit * (self.upper - self.lower)

        return self._repair_non_numeric(x_next.reshape(1, -1), self.var_type)[0]

    def optimize(self, X0: Optional[np.ndarray] = None) -> OptimizeResult:
        """Run the optimization process.

        The optimization terminates when either:
        - Total function evaluations reach max_iter (including initial design), OR
        - Runtime exceeds max_time minutes

        Args:
            X0 (ndarray, optional): Initial design points, shape (n_initial, n_features).
                If None, generates space-filling design. Defaults to None.

        Returns:
            OptimizeResult: Optimization result with fields:
                - x: best point found
                - fun: best function value
                - nfev: number of function evaluations (including initial design)
                - nit: number of sequential optimization iterations (after initial design)
                - success: whether optimization succeeded
                - message: termination message indicating reason for stopping
                - X: all evaluated points
                - y: all function values

        Examples:
            >>> # Example 1: Budget-based termination
            >>> opt = SpotOptim(fun=objective, bounds=bounds, max_iter=30, n_initial=10)
            >>> result = opt.optimize()
            >>> # Will perform 10 initial + 20 sequential iterations = 30 total evaluations
            >>>
            >>> # Example 2: Time-based termination
            >>> opt = SpotOptim(fun=expensive_objective, bounds=bounds, 
            ...                 max_iter=1000, max_time=5.0)  # 5 minutes max
            >>> result = opt.optimize()
            >>> # Will stop after 5 minutes OR 1000 evaluations, whichever comes first
        """
        # Generate or use provided initial design
        if X0 is None:
            X0 = self._generate_initial_design()
        else:
            X0 = np.atleast_2d(X0)
            # If X0 is in full dimensions and we have dimension reduction, reduce it
            if self.red_dim and X0.shape[1] == len(self.ident):
                X0 = self.to_red_dim(X0)
            X0 = self._repair_non_numeric(X0, self.var_type)

        # Repeat initial design points if repeats_initial > 1
        if self.repeats_initial > 1:
            X0 = np.repeat(X0, self.repeats_initial, axis=0)

        # Evaluate initial design
        y0 = self._evaluate_function(X0)

        # Initialize storage
        self.X_ = X0.copy()
        self.y_ = y0.copy()
        self.n_iter_ = 0

        # Update stats after initial design
        self.update_stats()
        
        # Log initial design to TensorBoard
        if self.tb_writer is not None:
            for i in range(len(self.y_)):
                self._write_tensorboard_hparams(self.X_[i], self.y_[i])
            self._write_tensorboard_scalars()

        # Initial best
        best_idx = np.argmin(self.y_)
        self.best_x_ = self.X_[best_idx].copy()
        self.best_y_ = self.y_[best_idx]

        if self.verbose:
            if self.noise:
                print(f"Initial best: f(x) = {self.best_y_:.6f}, mean best: f(x) = {self.min_mean_y:.6f}")
            else:
                print(f"Initial best: f(x) = {self.best_y_:.6f}")

        # Start timer for max_time check
        timeout_start = time.time()

        # Main optimization loop
        # Termination: continue while (total_evals < max_iter) AND (elapsed_time < max_time)
        while (len(self.y_) < self.max_iter) and (time.time() < timeout_start + self.max_time * 60):
            self.n_iter_ += 1

            # Fit surrogate (use mean_y if noise, otherwise y_)
            if self.noise:
                self._fit_surrogate(self.mean_X, self.mean_y)
            else:
                self._fit_surrogate(self.X_, self.y_)

            # OCBA: Compute optimal budget allocation for noisy functions
            # This determines which existing design points should be re-evaluated
            X_ocba = None
            if self.noise and self.ocba_delta > 0:
                # Check conditions for OCBA (need variance > 0 and at least 3 points)
                if not np.all(self.var_y > 0) and (self.mean_X.shape[0] <= 2):
                    if self.verbose:
                        print(f"Warning: OCBA skipped (need >2 points with variance > 0)")
                elif np.all(self.var_y > 0) and (self.mean_X.shape[0] > 2):
                    # Get OCBA allocation
                    X_ocba = get_ocba_X(self.mean_X, self.mean_y, self.var_y, self.ocba_delta)
                    if self.verbose and X_ocba is not None:
                        print(f"  OCBA: Adding {X_ocba.shape[0]} re-evaluation(s)")

            # Suggest next point
            x_next = self._suggest_next_point()

            # Repeat next point if repeats_surrogate > 1
            if self.repeats_surrogate > 1:
                x_next_repeated = np.repeat(x_next.reshape(1, -1), self.repeats_surrogate, axis=0)
            else:
                x_next_repeated = x_next.reshape(1, -1)

            # Append OCBA points to new design points (if applicable)
            if X_ocba is not None:
                x_next_repeated = append(X_ocba, x_next_repeated, axis=0)

            # Evaluate next point(s) including OCBA points
            y_next = self._evaluate_function(x_next_repeated)

            # Update storage
            self.X_ = np.vstack([self.X_, x_next_repeated])
            self.y_ = np.append(self.y_, y_next)

            # Update stats
            self.update_stats()
            
            # Log to TensorBoard
            if self.tb_writer is not None:
                # Log each new evaluation
                for i in range(len(y_next)):
                    self._write_tensorboard_hparams(x_next_repeated[i], y_next[i])
                self._write_tensorboard_scalars()

            # Update best
            current_best = np.min(y_next)
            if current_best < self.best_y_:
                best_idx_in_new = np.argmin(y_next)
                self.best_x_ = x_next_repeated[best_idx_in_new].copy()
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

        # Expand results to full dimensions if needed
        best_x_full = self.to_all_dim(self.best_x_.reshape(1, -1))[0] if self.red_dim else self.best_x_
        X_full = self.to_all_dim(self.X_) if self.red_dim else self.X_

        # Determine termination reason
        elapsed_time = time.time() - timeout_start
        if len(self.y_) >= self.max_iter:
            message = f"Optimization terminated: maximum evaluations ({self.max_iter}) reached"
        elif elapsed_time >= self.max_time * 60:
            message = f"Optimization terminated: time limit ({self.max_time:.2f} min) reached"
        else:
            message = "Optimization finished successfully"
        
        # Close TensorBoard writer
        self._close_tensorboard_writer()

        # Return scipy-style result
        return OptimizeResult(
            x=best_x_full,
            fun=self.best_y_,
            nfev=len(self.y_),
            nit=self.n_iter_,
            success=True,
            message=message,
            X=X_full,
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
        y_pred, y_std = self.surrogate.predict(grid_points, return_std=True)
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
                - X_i (ndarray): Meshgrid for dimension i.
                - X_j (ndarray): Meshgrid for dimension j.
                - grid_points (ndarray): Grid points for prediction, shape (num*num, n_dim).
        """
        k = self.n_dim
        mean_values = self.X_.mean(axis=0)

        # Create grid for dimensions i and j
        x_i = linspace(self.lower[i], self.upper[i], num=num)
        x_j = linspace(self.lower[j], self.upper[j], num=num)
        X_i, X_j = meshgrid(x_i, x_j)

        # Initialize grid points with mean values
        grid_points = np.tile(mean_values, (X_i.size, 1))
        grid_points[:, i] = X_i.ravel()
        grid_points[:, j] = X_j.ravel()

        # Apply type constraints
        grid_points = self._repair_non_numeric(grid_points, self.var_type)

        return X_i, X_j, grid_points
