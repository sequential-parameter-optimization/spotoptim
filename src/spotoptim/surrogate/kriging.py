"""
Kriging (Gaussian Process) surrogate model for SpotOptim.


Adapted from spotpython.surrogate.kriging_basic for compatibility with SpotOptim.
This implementation follows Forrester et al. (2008) "Engineering Design via Surrogate Modelling".

Specific references:
- Section 2.4 "Kriging": Core implementation of the Kriging predictor and likelihood.
- Section 6 "Surrogate Modeling of Noisy Data": Implementation of "regression" and "reinterpolation" methods.
- Validated against the book's Matlab implementation:
    - `likelihood.m`: Concentrated log-likelihood calculation.
    - `pred.m`: Prediction and error estimation.
"""

import numpy as np
from numpy.linalg import LinAlgError, cond
from typing import Dict, Tuple, List, Optional
import typing
from scipy.optimize import differential_evolution
from sklearn.base import BaseEstimator, RegressorMixin

from .kernels import SpotOptimKernel


class Kriging(BaseEstimator, RegressorMixin):
    """
    A scikit-learn compatible Kriging model class for regression tasks.

    This class provides Ordinary Kriging with support for:
    - Mixed variable types (continuous, integer, factor)
    - Gaussian/RBF correlation function
    - Three fitting methods (Forrester (2008), Section 6):
    - Isotropic or anisotropic length scales

    Compatible with SpotOptim's variable type conventions:
    - 'float': continuous numeric variables
    - 'int': integer variables
    - 'factor': categorical/unordered variables

    Args:
        noise (float, optional): Small regularization term for numerical stability (nugget effect).
            If None, defaults to sqrt(machine epsilon). Only used for "interpolation" method.
            For "regression" and "reinterpolation", this is replaced by the Lambda parameter.
            Defaults to None.
        penalty (float, optional): Large negative log-likelihood assigned if correlation matrix
            is not positive-definite. Defaults to 1e4.
        method (str, optional): Fitting method (Forrester (2008), Section 6). Options:
            - "interpolation": Pure Kriging interpolation (Eq 2.X). Fits exact data points.
              Uses a small `noise` (nugget) for numerical stability.
            - "regression": Regression Kriging (Section 6.2). Optimizes a regularization parameter
              Lambda (nugget) along with theta. Suitable for noisy data.
            - "reinterpolation": Re-interpolation (Section 6.3). Fits hyperparameters using
              regression (with Lambda), but predicts using the "noise-free" correlation matrix
              (removing Lambda). This creates a surrogate that glosses over noise but passes
              closer to the underlying trend (interpolating the regression model).
            Defaults to "regression".
        var_type (List[str], optional): Variable types for each dimension.
            SpotOptim uses: 'float' (continuous), 'int' (integer), 'factor' (categorical).
            Defaults to ["float"].
        name (str, optional): Name of the Kriging instance. Defaults to "Kriging".
        seed (int, optional): Random seed for reproducibility. Defaults to 124.
        model_fun_evals (int, optional): Maximum function evaluations for hyperparameter
            optimization. Defaults to 100.
        n_theta (int, optional): Number of theta parameters. If None, set during fit.
            Defaults to None.
        min_theta (float, optional): Minimum log10(theta) bound. Defaults to -3.0.
        max_theta (float, optional): Maximum log10(theta) bound. Defaults to 2.0.
        theta_init_zero (bool, optional): Initialize theta to zero. Defaults to False.
        p_val (float, optional): Power parameter for correlation (fixed at 2.0 for Gaussian).
            Defaults to 2.0.
        n_p (int, optional): Number of p parameters (currently not optimized). Defaults to 1.
        optim_p (bool, optional): Optimize p parameters (currently not supported). Defaults to False.
        min_Lambda (float, optional): Minimum log10(Lambda) bound. Defaults to -9.0.
        max_Lambda (float, optional): Maximum log10(Lambda) bound. Defaults to 0.0.
        metric_factorial (str, optional): Distance metric for factor variables.
            Defaults to "canberra".
        isotropic (bool, optional): Use single theta for all dimensions. Defaults to False.
        theta (np.ndarray, optional): Initial theta values (log10 scale). Defaults to None.

    Attributes:
        X_ (ndarray): Training data, shape (n_samples, n_features).
        y_ (ndarray): Training targets, shape (n_samples,).
        theta_ (ndarray): Optimized log10(theta) parameters.
        Lambda_ (float or None): Optimized log10(Lambda) for regression methods.
        mu_ (float): Mean of Kriging predictor.
        sigma2_ (float): Variance of Kriging predictor.
        U_ (ndarray): Cholesky factor of correlation matrix.
        Psi_ (ndarray): Correlation matrix.
        negLnLike (float): Negative log-likelihood value.

    Examples:
        Basic usage with SpotOptim:

        >>> import numpy as np
        >>> from spotoptim import SpotOptim
        >>> from spotoptim.surrogate import Kriging
        >>>
        >>> # Define objective
        >>> def objective(X):
        ...     return np.sum(X**2, axis=1)
        >>>
        >>> # Create Kriging surrogate
        >>> kriging = Kriging(
        ...     noise=1e-10,
        ...     method='regression',
        ...     min_theta=-3.0,
        ...     max_theta=2.0,
        ...     seed=42
        ... )
        >>>
        >>> # Use with SpotOptim
        >>> opt = SpotOptim(
        ...     fun=objective,
        ...     bounds=[(-5, 5), (-5, 5)],
        ...     surrogate=kriging,
        ...     max_iter=30,
        ...     n_initial=10,
        ...     seed=42
        ... )
        >>> result = opt.optimize()

        Direct usage (scikit-learn compatible):

        >>> from spotoptim.surrogate import Kriging
        >>> import numpy as np
        >>>
        >>> # Training data
        >>> X_train = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        >>> y_train = np.array([0.1, 0.2, 0.3])
        >>>
        >>> # Fit model
        >>> model = Kriging(method='regression', seed=42)
        >>> model.fit(X_train, y_train)
        >>>
        >>> # Predict
        >>> X_test = np.array([[0.25, 0.25], [0.75, 0.75]])
        >>> y_pred = model.predict(X_test)
        >>>
        >>> # Predict with uncertainty
        >>> y_pred, std = model.predict(X_test, return_std=True)
    """

    def __init__(
        self,
        noise: Optional[float] = None,
        penalty: float = 1e4,
        method: str = "regression",
        var_type: Optional[List[str]] = None,
        name: str = "Kriging",
        seed: int = 124,
        model_fun_evals: Optional[int] = None,
        n_theta: Optional[int] = None,
        min_theta: float = -3.0,
        max_theta: float = 2.0,
        theta_init_zero: bool = False,
        p_val: float = 2.0,
        n_p: int = 1,
        optim_p: bool = False,
        min_Lambda: float = -9.0,
        max_Lambda: float = 0.0,
        metric_factorial: str = "canberra",
        isotropic: bool = False,
        theta: Optional[np.ndarray] = None,
        **kwargs,
    ):
        if noise is None:
            self.noise = self._get_eps()
        else:
            if noise <= 0:
                raise ValueError("noise must be positive")
            self.noise = noise

        self.penalty = penalty
        self.var_type = var_type if var_type is not None else ["float"]
        self.name = name
        self.seed = seed
        self.metric_factorial = metric_factorial
        self.min_theta = min_theta
        self.max_theta = max_theta
        self.min_Lambda = min_Lambda
        self.max_Lambda = max_Lambda
        self.n_theta = n_theta
        self.isotropic = isotropic
        self.p_val = p_val
        self.n_p = n_p
        self.optim_p = optim_p
        self.theta_init_zero = theta_init_zero
        self.theta = theta
        self.model_fun_evals = model_fun_evals if model_fun_evals is not None else 100

        if method not in ["interpolation", "regression", "reinterpolation"]:
            raise ValueError(
                "method must be 'interpolation', 'regression', or 'reinterpolation'"
            )
        self.method = method

        # Fitted attributes
        self.X_ = None
        self.y_ = None
        self.n = None
        self.k = None
        self.theta_ = None
        self.Lambda_ = None
        self.mu_ = None
        self.sigma2_ = None
        self.U_ = None
        self.Psi_ = None
        self.negLnLike = None
        self.inf_Psi = False
        self.cnd_Psi = None

    def _get_eps(self) -> float:
        """Get square root of machine epsilon."""
        return np.sqrt(np.finfo(float).eps)

    def _set_variable_types(self) -> None:
        """Set variable type masks for different variable types.

        Creates boolean masks for:
        - num_mask: 'float' variables
        - int_mask: 'int' variables
        - factor_mask: 'factor' variables
        - ordered_mask: 'float', or 'int' variables (ordered/numeric)
        """
        # Ensure var_type has appropriate length
        if len(self.var_type) < self.k:
            self.var_type = ["float"] * self.k

        var_type_array = np.array(self.var_type)
        # SpotOptim uses 'float' for continuous variables
        self.num_mask = np.isin(var_type_array, ["float"])
        self.int_mask = var_type_array == "int"
        self.factor_mask = var_type_array == "factor"
        # Ordered variables: numeric (float/num) and integer
        self.ordered_mask = np.isin(var_type_array, ["int", "float"])

    def _set_theta(self) -> None:
        """Set number of theta parameters based on isotropic flag."""
        if self.isotropic:
            self.n_theta = 1
        else:
            self.n_theta = self.k

    def _get_theta10_from_logtheta(self) -> np.ndarray:
        """Convert log10(theta) to linear scale and expand if isotropic."""
        theta10 = np.power(10.0, self.theta_)
        if self.n_theta == 1:
            theta10 = theta10 * np.ones(self.k)
        return theta10

    def _reshape_X(self, X: np.ndarray) -> np.ndarray:
        """Ensure X has shape (n_samples, n_features)."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, self.k)
        else:
            if X.shape[1] != self.k:
                if X.shape[0] == self.k:
                    X = X.T
                elif self.k == 1:
                    X = X.reshape(-1, 1)
                else:
                    raise ValueError(f"X has shape {X.shape}, expected (*, {self.k})")
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Kriging":
        """Fit the Kriging model to training data.

        Optimizes hyperparameters (theta, Lambda) by maximizing the concentrated
        log-likelihood using differential evolution.

        Args:
            X (np.ndarray): Training inputs, shape (n_samples, n_features).
            y (np.ndarray): Training targets, shape (n_samples,).

        Returns:
            Kriging: Fitted model instance (self).
        """
        X = np.asarray(X)
        y = np.asarray(y).flatten()
        self.X_ = X
        self.y_ = y
        self.n, self.k = self.X_.shape

        self._set_variable_types()

        if self.n_theta is None:
            self._set_theta()

        # Store data bounds
        self.min_X = np.min(self.X_, axis=0)
        self.max_X = np.max(self.X_, axis=0)

        # Setup bounds for optimization
        bounds = [(self.min_theta, self.max_theta)] * self.n_theta

        if self.method in ["regression", "reinterpolation"]:
            bounds += [(self.min_Lambda, self.max_Lambda)]

        if self.optim_p:
            bounds += [(1.0, 2.0)] * self.n_p

        # Optimize hyperparameters
        result = differential_evolution(
            func=self.objective,
            bounds=bounds,
            seed=self.seed,
            maxiter=self.model_fun_evals,
        )

        params = result.x
        self.theta_ = params[: self.n_theta]

        if self.method in ["regression", "reinterpolation"]:
            self.Lambda_ = params[self.n_theta : self.n_theta + 1][0]
        else:
            self.Lambda_ = None

        # Store final likelihood and matrices
        self.negLnLike, self.Psi_, self.U_ = self.likelihood(params)

        return self

    def objective(self, params: np.ndarray) -> float:
        """Objective function for hyperparameter optimization.

        Args:
            params (np.ndarray): Hyperparameters to evaluate.

        Returns:
            float: Negative concentrated log-likelihood.

        Examples:
            >>> import numpy as np
            >>> from spotoptim.surrogate import Kriging
            >>> X = np.array([[0.], [1.]])
            >>> y = np.array([0., 1.])
            >>> k = Kriging(seed=42).fit(X, y)
            >>> # Evaluate objective at optimal parameters
            >>> val = k.objective(np.concatenate([k.theta_, [k.Lambda_]]))
        """
        negLnLike, _, _ = self.likelihood(params)
        return negLnLike

    def predict(self, X: np.ndarray, return_std: bool = False) -> np.ndarray:
        """Predict at new points.

        Args:
            X (np.ndarray): Test points, shape (n_samples, n_features).
            return_std (bool, optional): Return standard deviations. Defaults to False.

        Returns:
            np.ndarray: Predictions, shape (n_samples,).
            tuple: (predictions, std_devs) if return_std=True.
        """
        X = self._reshape_X(X)

        if return_std:
            predictions, stds = zip(*[self.predict_single(x) for x in X])
            return np.array(predictions), np.array(stds)
        else:
            predictions = [self.predict_single(x)[0] for x in X]
            return np.array(predictions)

    def build_correlation_matrix(self) -> np.ndarray:
        """Build correlation matrix from training data.

        Returns:
            np.ndarray: Upper triangle of correlation matrix.

        Examples:
            >>> import numpy as np
            >>> from spotoptim.surrogate import Kriging
            >>> X = np.array([[0.], [1.]])
            >>> y = np.array([0., 1.])
            >>> k = Kriging(seed=42).fit(X, y)
            >>> Psi_upper = k.build_correlation_matrix()
            >>> print(Psi_upper.shape)
            (2, 2)
        """
        try:
            theta10 = self._get_theta10_from_logtheta()

            # Use the new SpotOptimKernel to compute the correlation matrix
            # Note: We create it on the fly here to use current parameters
            kernel = SpotOptimKernel(
                theta=theta10,
                var_type=self.var_type,
                p_val=self.p_val,
                metric_factorial=self.metric_factorial,
            )

            # Compute full correlation matrix
            Psi = kernel(self.X_)

            self.inf_Psi = np.isinf(Psi).any()
            self.cnd_Psi = cond(Psi)

            return np.triu(Psi, k=1)

        except LinAlgError as err:
            print(f"Building Psi failed: {err}")
            raise

    def likelihood(self, params: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """Compute negative concentrated log-likelihood.

        Args:
            params (np.ndarray): Hyperparameters [log10(theta), log10(Lambda)].

        Returns:
            tuple: (negLnLike, Psi, U) where U is Cholesky factor.

        Reference:
            Forrester et al. (2008), Section 2.4.
            Matches implementation in `likelihood.m` from the book's code.
            Concentrated Log-Likelihood approx: ln(L) ~ -(n/2)ln(sigma^2) - (1/2)ln|R|

        Examples:
            >>> import numpy as np
            >>> from spotoptim.surrogate import Kriging
            >>> X = np.array([[0.], [1.]])
            >>> y = np.array([0., 1.])
            >>> k = Kriging(seed=42).fit(X, y)
            >>> params = np.concatenate([k.theta_, [k.Lambda_]])
            >>> nll, _, _ = k.likelihood(params)
        """
        # Extract parameters
        self.theta_ = params[: self.n_theta]

        if self.method in ["regression", "reinterpolation"]:
            lambda_log = params[self.n_theta : self.n_theta + 1][0]
            lambda_ = 10.0**lambda_log
        else:
            lambda_ = self.noise

        n = self.n
        one = np.ones(n)

        # Build correlation matrix
        Psi_upper = self.build_correlation_matrix()
        Psi = Psi_upper + Psi_upper.T + np.eye(n) * (1.0 + lambda_)

        # Cholesky factorization
        try:
            U = np.linalg.cholesky(Psi)
        except LinAlgError:
            return self.penalty, Psi, None

        # log|R|
        LnDetPsi = 2.0 * np.sum(np.log(np.abs(np.diag(U))))

        # Solve for mu and sigma^2
        y = self.y_
        temp_y = np.linalg.solve(U, y)
        temp_one = np.linalg.solve(U, one)
        vy = np.linalg.solve(U.T, temp_y)
        vone = np.linalg.solve(U.T, temp_one)

        mu = (one @ vy) / (one @ vone)
        resid = y - one * mu
        tresid = np.linalg.solve(U, resid)
        tresid = np.linalg.solve(U.T, tresid)
        SigmaSqr = (resid @ tresid) / n

        # Concentrated negative log-likelihood
        # Corresponds to `likelihood.m` line:
        # NegLnLike=-1*(-(n/2)*log(SigmaSqr) - 0.5*LnDetPsi);
        # We minimize positive NegLnLike, which is equivalent to maximizing likelihood.
        negLnLike = (n / 2.0) * np.log(SigmaSqr) + 0.5 * LnDetPsi

        return negLnLike, Psi, U

    def build_psi_vector(self, x: np.ndarray) -> np.ndarray:
        """Build correlation vector between x and training points.

        Args:
            x (np.ndarray): Single test point, shape (n_features,).

        Returns:
            np.ndarray: Correlation vector, shape (n_samples,).

        Reference:
            Calculates the vector small psi from Eq 2.15 in Forrester (2008).

        Examples:
            >>> import numpy as np
            >>> from spotoptim.surrogate import Kriging
            >>> X = np.array([[0.], [1.]])
            >>> y = np.array([0., 1.])
            >>> k = Kriging(seed=42).fit(X, y)
            >>> x_new = np.array([0.5])
            >>> psi = k.build_psi_vector(x_new)
            >>> print(psi.shape)
            (2,)
        """
        theta10 = self._get_theta10_from_logtheta()

        kernel = SpotOptimKernel(
            theta=theta10,
            var_type=self.var_type,
            p_val=self.p_val,
            metric_factorial=self.metric_factorial,
        )

        # Calculate correlation between X_ (training) and x (new point)
        # kernel(X, Y) returns matrix of shape (n_X, n_Y)
        # We need vector of shape (n,).
        # kernel(self.X_, x.reshape(1, -1)) returns (n, 1)
        # flatten to get (n,)
        psi = kernel(self.X_, x.reshape(1, -1)).flatten()

        return psi

    def predict_single(self, x: np.ndarray) -> Tuple[float, float]:
        """Predict at a single point.

        Args:
            x (np.ndarray): Test point, shape (n_features,).

        Returns:
            tuple: (prediction, std_dev).

        Reference:
            Forrester et al. (2008), Eq 2.15 (Predictor) and Eq 2.19 (Error).
            Matches implementation in `pred.m` from the book's code.

        Examples:
            >>> import numpy as np
            >>> from spotoptim.surrogate import Kriging
            >>> X = np.array([[0.], [1.]])
            >>> y = np.array([0., 1.])
            >>> k = Kriging(seed=42).fit(X, y)
            >>> x_new = np.array([0.5])
            >>> y_pred, y_std = k.predict_single(x_new)
        """
        if self.method in ["regression", "reinterpolation"]:
            lambda_ = 10.0**self.Lambda_
        else:
            lambda_ = self.noise

        U = self.U_
        n = self.n
        y = self.y_
        one = np.ones(n)

        # Compute mu
        y_tilde = np.linalg.solve(U, y)
        y_tilde = np.linalg.solve(U.T, y_tilde)
        one_tilde = np.linalg.solve(U, one)
        one_tilde = np.linalg.solve(U.T, one_tilde)
        mu = (one @ y_tilde) / (one @ one_tilde)

        # Residual
        resid = y - one * mu
        resid_tilde = np.linalg.solve(U, resid)
        resid_tilde = np.linalg.solve(U.T, resid_tilde)

        # Correlation vector
        psi = self.build_psi_vector(x)

        # Prediction
        # Eq 2.15: \hat{y} = \hat{\mu} + \psi^T \Psi^{-1} (y - \mathbf{1}\hat{\mu})
        # Implemented using pre-computed Cholesky factors as in `pred.m`
        f = mu + psi @ resid_tilde

        # Variance
        # Eq 2.19 (Mean Squared Error)
        if self.method in ["interpolation", "regression"]:
            SigmaSqr = (resid @ resid_tilde) / n
            psi_tilde = np.linalg.solve(U, psi)
            psi_tilde = np.linalg.solve(U.T, psi_tilde)
            SSqr = SigmaSqr * (1.0 + lambda_ - psi @ psi_tilde)
        else:  # reinterpolation
            Psi_adjusted = self.Psi_ - np.eye(n) * lambda_ + np.eye(n) * self.noise
            SigmaSqr = (
                resid
                @ np.linalg.solve(U.T, np.linalg.solve(U, Psi_adjusted @ resid_tilde))
            ) / n
            Uint = np.linalg.cholesky(Psi_adjusted)
            psi_tilde = np.linalg.solve(Uint, psi)
            psi_tilde = np.linalg.solve(Uint.T, psi_tilde)
            SSqr = SigmaSqr * (1.0 - psi @ psi_tilde)

        s = np.sqrt(np.abs(SSqr))

        return float(f), float(s)

    def get_params(self, deep: bool = True) -> Dict:
        """Get parameters for this estimator (scikit-learn compatibility).

        Args:
            deep (bool): Ignored, for compatibility.

        Returns:
            dict: Parameter names mapped to their values.
        """
        return {
            "noise": self.noise,
            "penalty": self.penalty,
            "method": self.method,
            "var_type": self.var_type,
            "name": self.name,
            "seed": self.seed,
            "model_fun_evals": self.model_fun_evals,
            "n_theta": self.n_theta,
            "min_theta": self.min_theta,
            "max_theta": self.max_theta,
            "theta_init_zero": self.theta_init_zero,
            "p_val": self.p_val,
            "n_p": self.n_p,
            "optim_p": self.optim_p,
            "min_Lambda": self.min_Lambda,
            "max_Lambda": self.max_Lambda,
            "metric_factorial": self.metric_factorial,
            "isotropic": self.isotropic,
            "theta": self.theta,
        }

    def set_params(self, **params: "typing.Any") -> "Kriging":
        """Set parameters (scikit-learn compatibility).

        Args:
            **params: Parameter names and values.

        Returns:
            Kriging: Self.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
