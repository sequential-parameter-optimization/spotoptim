import numpy as np
from typing import Callable, Optional, Tuple
from scipy.optimize import OptimizeResult, differential_evolution
from sklearn.base import BaseEstimator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from spotpython.design.spacefilling import SpaceFilling
from spotpython.utils.repair import repair_non_numeric
from spotpython.utils.compare import selectNew


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
        tolerance_x: float = 1e-6,
        seed: Optional[int] = None,
        verbose: bool = False,
    ):
        self.fun = fun
        self.bounds = bounds
        self.max_iter = max_iter
        self.n_initial = n_initial
        self.surrogate = surrogate
        self.acquisition = acquisition
        self.var_type = var_type
        self.tolerance_x = tolerance_x
        self.seed = seed
        self.verbose = verbose

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
        self.design = SpaceFilling(k=self.n_dim, seed=self.seed)

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
        """Generate initial space-filling design."""
        X0 = self.design.scipy_lhd(
            n=self.n_initial, repeats=1, lower=self.lower, upper=self.upper
        )
        return repair_non_numeric(X0, self.var_type)

    def _fit_surrogate(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit surrogate model to data."""
        self.surrogate.fit(X, y)

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
        x_new, _ = selectNew(A=x_next_2d, X=self.X_, tolerance=self.tolerance_x)

        if x_new.shape[0] == 0:
            # If too close, generate random point
            if self.verbose:
                print("Proposed point too close, generating random point")
            x_next = self.design.scipy_lhd(
                n=1, repeats=1, lower=self.lower, upper=self.upper
            )[0]

        return repair_non_numeric(x_next.reshape(1, -1), self.var_type)[0]

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
            X0 = repair_non_numeric(X0, self.var_type)

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
