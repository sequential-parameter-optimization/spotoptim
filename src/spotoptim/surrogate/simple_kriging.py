"""
Simplified SimpleKriging surrogate model for SpotOptim.

This is a streamlined version adapted from spotpython.surrogate.kriging
for use with the SpotOptim optimizer.
"""

import numpy as np
from typing import Optional
from numpy.linalg import LinAlgError, cholesky, solve
from scipy.optimize import differential_evolution
from sklearn.base import BaseEstimator, RegressorMixin


class SimpleKriging(BaseEstimator, RegressorMixin):
    """A simplified Kriging (Gaussian Process) surrogate model for SpotOptim.

    This class provides a scikit-learn compatible interface with fit() and predict()
    methods, making it suitable for use as a surrogate in SpotOptim.

    Args:
        noise (float, optional): Regularization parameter (nugget effect). If None, uses sqrt(eps).
            Defaults to None.
        kernel (str, optional): Kernel type. Currently only 'gauss' (Gaussian/RBF) is supported.
            Defaults to 'gauss'.
        n_theta (int, optional): Number of theta parameters. If None, uses k (number of dimensions).
            Defaults to None.
        min_theta (float, optional): Minimum log10(theta) bound for optimization. Defaults to -3.0.
        max_theta (float, optional): Maximum log10(theta) bound for optimization. Defaults to 2.0.
        seed (int, optional): Random seed for reproducibility. Defaults to None.

    Attributes:
        X_ (ndarray): Training data, shape (n_samples, n_features).
        y_ (ndarray): Training targets, shape (n_samples,).
        theta_ (ndarray): Optimized theta parameters (log10 scale).
        mu_ (float): Mean of the SimpleKriging predictor.
        sigma2_ (float): Variance of the SimpleKriging predictor.

    Examples:
        >>> import numpy as np
        >>> from spotoptim.surrogate import SimpleKriging
        >>> X = np.array([[0.0], [0.5], [1.0]])
        >>> y = np.array([0.0, 0.25, 1.0])
        >>> model = SimpleKriging()
        >>> model.fit(X, y)
        >>> predictions = model.predict(np.array([[0.25], [0.75]]))
    """

    def __init__(
        self,
        noise: Optional[float] = None,
        kernel: str = "gauss",
        n_theta: Optional[int] = None,
        min_theta: float = -3.0,
        max_theta: float = 2.0,
        seed: Optional[int] = None,
    ):
        self.noise = noise
        self.kernel = kernel
        self.n_theta = n_theta
        self.min_theta = min_theta
        self.max_theta = max_theta
        self.seed = seed

        # Fitted attributes
        self.X_ = None
        self.y_ = None
        self.theta_ = None
        self.mu_ = None
        self.sigma2_ = None
        self.U_ = None  # Cholesky factor
        self.Rinv_one_ = None
        self.Rinv_r_ = None

    def _get_noise(self) -> float:
        """Get the noise/regularization parameter.

        Returns:
            float: Noise/regularization value.
        """
        if self.noise is None:
            return np.sqrt(np.finfo(float).eps)
        return self.noise

    def _correlation(self, D: np.ndarray) -> np.ndarray:
        """Compute correlation from distance matrix using Gaussian kernel.

        Args:
            D (ndarray): Squared distance matrix.

        Returns:
            ndarray: Correlation matrix.
        """
        if self.kernel == "gauss":
            return np.exp(-D)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")

    def _build_correlation_matrix(self, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Build correlation matrix R for training data.

        Args:
            X (ndarray): Input data, shape (n, k).
            theta (ndarray): Theta parameters (10^theta used as weights), shape (k,).

        Returns:
            ndarray: Correlation matrix with noise on diagonal, shape (n, n).
        """
        n = X.shape[0]
        theta10 = 10.0**theta

        # Compute weighted squared distances
        R = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                diff = X[i] - X[j]
                dist = np.sum(theta10 * diff**2)
                R[i, j] = dist
                R[j, i] = dist

        # Apply correlation function
        R = self._correlation(R)

        # Add noise to diagonal
        noise_val = self._get_noise()
        np.fill_diagonal(R, 1.0 + noise_val)

        return R

    def _build_correlation_vector(
        self, x: np.ndarray, X: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """Build correlation vector between new point x and training data X.

        Args:
            x (ndarray): New point, shape (k,).
            X (ndarray): Training data, shape (n, k).
            theta (ndarray): Theta parameters, shape (k,).

        Returns:
            ndarray: Correlation vector, shape (n,).
        """
        theta10 = 10.0**theta
        diff = X - x.reshape(1, -1)
        D = np.sum(theta10 * diff**2, axis=1)
        return self._correlation(D)

    def _neg_log_likelihood(self, log_theta: np.ndarray) -> float:
        """Compute negative concentrated log-likelihood.

        Args:
            log_theta (ndarray): Log10(theta) parameters.

        Returns:
            float: Negative log-likelihood (to be minimized).
        """
        try:
            n = self.X_.shape[0]
            y = self.y_.flatten()
            one = np.ones(n)

            # Build correlation matrix
            R = self._build_correlation_matrix(self.X_, log_theta)

            # Cholesky decomposition
            try:
                U = cholesky(R)
            except LinAlgError:
                return 1e10  # Penalty for ill-conditioned matrix

            # Solve for mean and variance
            Uy = solve(U, y)
            Uone = solve(U, one)

            Rinv_y = solve(U.T, Uy)
            Rinv_one = solve(U.T, Uone)

            mu = (one @ Rinv_y) / (one @ Rinv_one)
            r = y - one * mu

            Ur = solve(U, r)
            Rinv_r = solve(U.T, Ur)

            sigma2 = (r @ Rinv_r) / n

            if sigma2 <= 0:
                return 1e10

            # Concentrated log-likelihood
            log_det_R = 2.0 * np.sum(np.log(np.abs(np.diag(U))))
            neg_log_like = (n / 2.0) * np.log(sigma2) + 0.5 * log_det_R

            return neg_log_like

        except (LinAlgError, ValueError):
            return 1e10

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleKriging":
        """Fit the SimpleKriging model to training data.

        Args:
            X (ndarray): Training input data, shape (n_samples, n_features).
            y (ndarray): Training target values, shape (n_samples,).

        Returns:
            SimpleKriging: Fitted estimator (self).
        """
        X = np.atleast_2d(X)
        y = np.asarray(y).flatten()

        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1-dimensional, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")

        self.X_ = X
        self.y_ = y
        n, k = X.shape

        # Set number of theta parameters
        if self.n_theta is None:
            self.n_theta = k

        # Optimize theta via maximum likelihood
        bounds = [(self.min_theta, self.max_theta)] * self.n_theta

        result = differential_evolution(
            func=self._neg_log_likelihood,
            bounds=bounds,
            seed=self.seed,
            maxiter=100,
            atol=1e-6,
            tol=0.01,
        )

        self.theta_ = result.x

        # Compute final model parameters
        one = np.ones(n)
        R = self._build_correlation_matrix(X, self.theta_)

        try:
            self.U_ = cholesky(R)
        except LinAlgError:
            # Add more regularization if needed
            R = self._build_correlation_matrix(X, self.theta_)
            R += np.eye(n) * 1e-8
            self.U_ = cholesky(R)

        Uy = solve(self.U_, y)
        Uone = solve(self.U_, one)

        Rinv_y = solve(self.U_.T, Uy)
        Rinv_one = solve(self.U_.T, Uone)

        self.mu_ = float((one @ Rinv_y) / (one @ Rinv_one))

        r = y - one * self.mu_
        Ur = solve(self.U_, r)
        Rinv_r = solve(self.U_.T, Ur)

        self.sigma2_ = float((r @ Rinv_r) / n)

        # Store for prediction
        self.Rinv_one_ = Rinv_one
        self.Rinv_r_ = Rinv_r

        return self

    def predict(self, X: np.ndarray, return_std: bool = False) -> np.ndarray:
        """Predict using the SimpleKriging model.

        Args:
            X (ndarray): Points to predict at, shape (n_samples, n_features).
            return_std (bool, optional): If True, return standard deviations as well.
                Defaults to False.

        Returns:
            ndarray or tuple: If return_std is False, returns predicted values (n_samples,).
                If return_std is True, returns tuple of (predictions, std_devs) both shape (n_samples,).
        """
        X = np.atleast_2d(X)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] != self.X_.shape[1]:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.X_.shape[1]}"
            )

        n_pred = X.shape[0]
        predictions = np.zeros(n_pred)

        if return_std:
            std_devs = np.zeros(n_pred)

        for i, x in enumerate(X):
            # Build correlation vector
            psi = self._build_correlation_vector(x, self.X_, self.theta_)

            # Predict mean
            predictions[i] = self.mu_ + psi @ self.Rinv_r_

            if return_std:
                # Predict variance
                Upsi = solve(self.U_, psi)
                psi_Rinv_psi = psi @ solve(self.U_.T, Upsi)

                variance = self.sigma2_ * (1.0 + self._get_noise() - psi_Rinv_psi)
                std_devs[i] = np.sqrt(max(0.0, variance))

        if return_std:
            return predictions, std_devs
        return predictions
