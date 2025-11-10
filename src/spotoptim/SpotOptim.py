import numpy as np
from typing import Callable, Optional, Tuple, List
from scipy.optimize import OptimizeResult, differential_evolution
from scipy.stats.qmc import LatinHypercube
from sklearn.base import BaseEstimator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
import warnings
import matplotlib.pyplot as plt
from numpy import linspace, meshgrid


class SpotOptim(BaseEstimator):
    """
    SPOT optimizer compatible with scipy.optimize interface.

    Parameters
    ----------
    fun : callable
        Objective function to minimize. Should accept array of shape (n_samples, n_features).
    bounds : list of tuple
        Bounds for each dimension as [(low, high), ...].
    max_iter : int, default=20
        Maximum number of optimization iterations.
    n_initial : int, default=10
        Number of initial design points.
    surrogate : object, optional
        Surrogate model (default: Gaussian Process with Matern kernel).
    acquisition : str, default='ei'
        Acquisition function ('ei', 'y', 'pi').
    var_type : list of str, optional
        Variable types for each dimension ('num', 'int', 'float', 'factor').
    tolerance_x : float, default=1e-6
        Minimum distance between points.
    seed : int, optional
        Random seed for reproducibility.
    verbose : bool, default=False
        Print progress information.
    warnings_filter : str, default="ignore"
        Filter for warnings. One of "error", "ignore", "always", "all", "default", "module", or "once".
    max_surrogate_points : int, optional
        Maximum number of points to use for surrogate model fitting.
        If None, all points are used. If the number of evaluated points exceeds this limit,
        a subset is selected using the selection method.
    selection_method : str, default='distant'
        Method for selecting points when max_surrogate_points is exceeded.
        Options:
        - 'distant': Select points that are distant from each other (K-means clustering).
        - 'best': Select all points from the cluster with the best mean objective value.

    Attributes
    ----------
    X_ : ndarray of shape (n_samples, n_features)
        All evaluated points.
    y_ : ndarray of shape (n_samples,)
        Function values at X_.
    best_x_ : ndarray of shape (n_features,)
        Best point found.
    best_y_ : float
        Best function value found.
    n_iter_ : int
        Number of iterations performed.
    warnings_filter : str
        Filter for warnings during optimization.
    max_surrogate_points : int or None
        Maximum number of points for surrogate fitting.
    selection_method : str
        Point selection method.

    Examples
    --------
    >>> def objective(X):
    ...     return np.sum(X**2, axis=1)
    ...
    >>> bounds = [(-5, 5), (-5, 5)]
    >>> optimizer = SpotOptim(fun=objective, bounds=bounds, max_iter=10, n_initial=5, verbose=True)
    >>> result = optimizer.optimize()
    >>> print("Best x:", result.x)
    >>> print("Best f(x):", result.fun)
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
        tolerance_x: Optional[float] = None,
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

        self.fun = fun
        self.bounds = bounds
        self.max_iter = max_iter
        self.n_initial = n_initial
        self.surrogate = surrogate
        self.acquisition = acquisition
        self.var_type = var_type
        self.seed = seed
        self.verbose = verbose
        self.max_surrogate_points = max_surrogate_points
        self.selection_method = selection_method

        # Derived attributes
        self.n_dim = len(bounds)
        self.lower = np.array([b[0] for b in bounds])
        self.upper = np.array([b[1] for b in bounds])

        # Default variable types
        if self.var_type is None:
            self.var_type = ["num"] * self.n_dim

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
        self.best_x_ = None
        self.best_y_ = None
        self.n_iter_ = 0

    def _evaluate_function(self, X: np.ndarray) -> np.ndarray:
        """Evaluate objective function at points X."""
        # Ensure X is 2D
        X = np.atleast_2d(X)

        # Evaluate function
        y = self.fun(X)

        # Ensure y is 1D
        if isinstance(y, np.ndarray) and y.ndim > 1:
            y = y.ravel()
        elif not isinstance(y, np.ndarray):
            y = np.array([y])

        return y

    def _generate_initial_design(self) -> np.ndarray:
        """Generate initial space-filling design using Latin Hypercube Sampling."""
        # Generate samples in [0, 1]^d
        X0_unit = self.lhs_sampler.random(n=self.n_initial)

        # Scale to [lower, upper]
        X0 = self.lower + X0_unit * (self.upper - self.lower)

        return self._repair_non_numeric(X0, self.var_type)

    def _select_distant_points(
        self, X: np.ndarray, y: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Selects k points that are distant from each other using K-means clustering.

        This method performs K-means clustering to find k clusters, then selects
        the point closest to each cluster center. This ensures a space-filling
        subset of points for surrogate model training.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Design points.
        y : ndarray of shape (n_samples,)
            Function values at X.
        k : int
            Number of points to select.

        Returns
        -------
        selected_X : ndarray of shape (k, n_features)
            Selected design points.
        selected_y : ndarray of shape (k,)
            Function values at selected points.

        Examples
        --------
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
        """
        Selects all points from the cluster with the smallest mean y value.

        This method performs K-means clustering and selects all points from the
        cluster whose center corresponds to the best (smallest) mean objective
        function value.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Design points.
        y : ndarray of shape (n_samples,)
            Function values at X.
        k : int
            Number of clusters.

        Returns
        -------
        selected_X : ndarray of shape (m, n_features)
            Selected design points from best cluster.
        selected_y : ndarray of shape (m,)
            Function values at selected points.

        Examples
        --------
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
        """
        Dispatcher for selection methods.

        Depending on the value of `self.selection_method`, this method calls
        the appropriate selection function to choose a subset of points for
        surrogate model training when the total number of points exceeds
        `self.max_surrogate_points`.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Design points.
        y : ndarray of shape (n_samples,)
            Function values at X.

        Returns
        -------
        selected_X : ndarray
            Selected design points.
        selected_y : ndarray
            Function values at selected points.

        Examples
        --------
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
            return self._select_distant_points(
                X=X, y=y, k=self.max_surrogate_points
            )
        elif self.selection_method == "best":
            return self._select_best_cluster(
                X=X, y=y, k=self.max_surrogate_points
            )
        else:
            # If no valid selection method, return all points
            return X, y

    def _fit_surrogate(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit surrogate model to data.

        If the number of points exceeds `self.max_surrogate_points`,
        a subset of points is selected using the selection dispatcher.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Design points.
        y : ndarray of shape (n_samples,)
            Function values at X.
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
        """
        Select rows from A that are not in X.

        Parameters
        ----------
        A : ndarray
            Array with new values.
        X : ndarray
            Array with known values.
        tolerance : float, default=0
            Tolerance value for comparison.

        Returns
        -------
        ndarray
            Array with unknown (new) values.
        ndarray
            Array with True if value is new, otherwise False.
        """
        B = np.abs(A[:, None] - X)
        ind = np.any(np.all(B <= tolerance, axis=2), axis=1)
        return A[~ind], ~ind

    def _repair_non_numeric(self, X: np.ndarray, var_type: List[str]) -> np.ndarray:
        """
        Round non-numeric values to integers.
        This applies to all variables except for "num" and "float".

        Parameters
        ----------
        X : ndarray
            X array.
        var_type : list of str
            List with type information.

        Returns
        -------
        ndarray
            X array with non-numeric values rounded to integers.
        """
        mask = np.isin(var_type, ["num", "float"], invert=True)
        X[:, mask] = np.around(X[:, mask])
        return X

    def _acquisition_function(self, x: np.ndarray) -> float:
        """
        Compute acquisition function value.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            Point to evaluate.

        Returns
        -------
        float
            Acquisition function value (to be minimized).
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
        """
        Suggest next point to evaluate using acquisition function optimization.

        Returns
        -------
        ndarray of shape (n_features,)
            Next point to evaluate.
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
        """
        Run the optimization process.

        Parameters
        ----------
        X0 : ndarray of shape (n_initial, n_features), optional
            Initial design points. If None, generates space-filling design.

        Returns
        -------
        OptimizeResult
            Optimization result with fields:
            - x : best point found
            - fun : best function value
            - nfev : number of function evaluations
            - success : whether optimization succeeded
            - message : termination message
            - X : all evaluated points
            - y : all function values
        """
        # Generate or use provided initial design
        if X0 is None:
            X0 = self._generate_initial_design()
        else:
            X0 = np.atleast_2d(X0)
            X0 = self._repair_non_numeric(X0, self.var_type)

        # Evaluate initial design
        y0 = self._evaluate_function(X0)

        # Initialize storage
        self.X_ = X0.copy()
        self.y_ = y0.copy()
        self.n_iter_ = 0

        # Initial best
        best_idx = np.argmin(self.y_)
        self.best_x_ = self.X_[best_idx].copy()
        self.best_y_ = self.y_[best_idx]

        if self.verbose:
            print(f"Initial best: f(x) = {self.best_y_:.6f}")

        # Main optimization loop
        for iteration in range(self.max_iter):
            self.n_iter_ = iteration + 1

            # Fit surrogate
            self._fit_surrogate(self.X_, self.y_)

            # Suggest next point
            x_next = self._suggest_next_point()

            # Evaluate next point
            y_next = self._evaluate_function(x_next.reshape(1, -1))

            # Update storage
            self.X_ = np.vstack([self.X_, x_next])
            self.y_ = np.append(self.y_, y_next)

            # Update best
            if y_next[0] < self.best_y_:
                self.best_x_ = x_next.copy()
                self.best_y_ = y_next[0]

                if self.verbose:
                    print(
                        f"Iteration {iteration+1}: New best f(x) = {self.best_y_:.6f}"
                    )
            elif self.verbose:
                print(f"Iteration {iteration+1}: f(x) = {y_next[0]:.6f}")

        # Return scipy-style result
        return OptimizeResult(
            x=self.best_x_,
            fun=self.best_y_,
            nfev=len(self.y_),
            nit=self.n_iter_,
            success=True,
            message="Optimization finished successfully",
            X=self.X_,
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
        """
        Plot the surrogate model for two dimensions.

        Creates a 2x2 plot showing:
        - Top left: 3D surface of predictions
        - Top right: 3D surface of prediction uncertainty
        - Bottom left: Contour plot of predictions with evaluated points
        - Bottom right: Contour plot of prediction uncertainty

        Parameters
        ----------
        i : int, default=0
            Index of the first dimension to plot.
        j : int, default=1
            Index of the second dimension to plot.
        show : bool, default=True
            If True, displays the plot immediately.
        alpha : float, default=0.8
            Transparency of the 3D surface plots (0=transparent, 1=opaque).
        var_name : list of str, optional
            Names for each dimension. If None, uses generic labels.
        cmap : str, default='jet'
            Matplotlib colormap name.
        num : int, default=100
            Number of grid points per dimension for mesh grid.
        vmin : float, optional
            Minimum value for color scale. If None, determined from data.
        vmax : float, optional
            Maximum value for color scale. If None, determined from data.
        add_points : bool, default=True
            If True, overlay evaluated points on contour plots.
        grid_visible : bool, default=True
            If True, show grid lines on contour plots.
        contour_levels : int, default=30
            Number of contour levels.
        figsize : tuple of int, default=(12, 10)
            Figure size in inches (width, height).

        Raises
        ------
        ValueError
            If optimization hasn't been run yet, or if i, j are invalid.

        Examples
        --------
        >>> import numpy as np
        >>> from spotoptim import SpotOptim
        >>> def sphere(X):
        ...     return np.sum(X**2, axis=1)
        >>> opt = SpotOptim(fun=sphere, bounds=[(-5, 5), (-5, 5)], 
        ...                 max_iter=10, n_initial=5)
        >>> result = opt.optimize()
        >>> opt.plot_surrogate(i=0, j=1, var_name=['x1', 'x2'])
        """
        # Validation
        if self.X_ is None or self.y_ is None:
            raise ValueError(
                "No optimization data available. Run optimize() first."
            )

        k = self.n_dim
        if i >= k or j >= k:
            raise ValueError(
                f"Dimensions i={i} and j={j} must be less than n_dim={k}."
            )
        if i == j:
            raise ValueError("Dimensions i and j must be different.")

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
            ax3.scatter(self.X_[:, i], self.X_[:, j], c='red', s=30, 
                       edgecolors='black', zorder=5, label='Evaluated points')
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
            ax4.scatter(self.X_[:, i], self.X_[:, j], c='red', s=30,
                       edgecolors='black', zorder=5, label='Evaluated points')
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
        """
        Generate a mesh grid for two dimensions, filling others with mean values.

        Parameters
        ----------
        i : int
            Index of the first dimension to vary.
        j : int
            Index of the second dimension to vary.
        num : int, default=100
            Number of grid points per dimension.

        Returns
        -------
        X_i : ndarray
            Meshgrid for dimension i.
        X_j : ndarray
            Meshgrid for dimension j.
        grid_points : ndarray of shape (num*num, n_dim)
            Grid points for prediction.
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
