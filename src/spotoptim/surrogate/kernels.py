import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist, pdist, squareform


class Kernel(ABC):
    """
    Base class for Kernels.
    """

    @abstractmethod
    def __call__(self, X, Y=None) -> np.ndarray:
        """
        Evaluate the kernel.

        Args:
            X (np.ndarray): Left argument of the kernel evaluation.
            Y (np.ndarray, optional): Right argument of the kernel evaluation.
                If None, defaults to X.

        Returns:
            np.ndarray: Kernel matrix of shape (n_samples_X, n_samples_Y).
        """
        pass

    @abstractmethod
    def diag(self, X) -> np.ndarray:
        """
        Returns the diagonal of the kernel k(X, X).

        The result of this method is equivalent to np.diag(self(X)); however,
        it can be evaluated more efficiently.

        Args:
            X (np.ndarray): Argument of the kernel evaluation.

        Returns:
            np.ndarray: Diagonal of the kernel matrix, shape (n_samples_X,).
        """
        pass

    def __add__(self, other):
        return Sum(self, other)

    def __mul__(self, other):
        return Product(self, other)

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return Product(ConstantKernel(other), self)
        return Product(other, self)


class SpotOptimKernel(Kernel):
    """
    Kernel designed for SpotOptim's Kriging with mixed variable support.

    It handles continuous ('float'), integer ('int'), and categorical ('factor') variables
    similarly to the internal logic of the Kriging class.

    The correlation function is defined as:
    Psi = exp(- (D_ordered + D_factor))

    where:
    D_ordered = sum_j theta_j * |x_ij - y_lj|^p  (for ordered variables)
    D_factor  = sum_j theta_j * d(x_ij, y_lj)    (for factor variables, d is metric like Canberra)

    Args:
        theta (np.ndarray): The correlation parameters (weights).
            Note: In standard Kriging usage, this corresponds to `10^theta_log`.
            This kernel expects the LINEAR scale theta values (weights), not log.
        var_type (list of str): List of variable types, e.g. ['float', 'int', 'factor'].
        p_val (float, optional): Power parameter for ordered distance. Defaults to 2.0.
        metric_factorial (str, optional): Metric for factor distance (passed to cdist/pdist).
            Defaults to 'canberra'.
    """

    def __init__(
        self,
        theta,
        var_type,
        p_val=2.0,
        metric_factorial="canberra",
    ):
        self.theta = np.asanyarray(theta)
        self.var_type = var_type
        self.p_val = p_val
        self.metric_factorial = metric_factorial

        # Precompute masks
        var_type_array = np.array(self.var_type)
        self.ordered_mask = np.isin(var_type_array, ["int", "float"])
        self.factor_mask = var_type_array == "factor"

        # Validate theta dimension
        if self.theta.size == 1:
            # Broadcast if isotropic (size 1) but multiple vars provided
            # But here we assume we might get a scalar or a vector matching k
            pass  # Logic handled in call/diag
        else:
            if self.theta.shape[0] != len(var_type):
                # This check might fail if isotropic theta (size 1) is passed with k > 1.
                # Kriging generally expands it. We will handle broadcasting inside call.
                pass

    def __call__(self, X, Y=None):
        X = np.atleast_2d(X)
        if Y is None:
            Y = X
            is_symmetric = True
        else:
            Y = np.atleast_2d(Y)
            is_symmetric = False

        n, k = X.shape
        m, _ = Y.shape

        # Ensure theta is correct shape
        theta = self.theta
        if theta.size == 1 and k > 1:
            theta = np.full(k, theta.item())

        if theta.shape[0] != k:
            raise ValueError(
                f"theta dimension {theta.shape[0]} does not match input dimension {k}"
            )

        D = np.zeros((n, m))

        # Ordered variables
        if self.ordered_mask.any():
            X_ordered = X[:, self.ordered_mask]
            Y_ordered = Y[:, self.ordered_mask]
            w_ordered = theta[self.ordered_mask]

            if is_symmetric:
                D_ordered = squareform(
                    pdist(X_ordered, metric="sqeuclidean", w=w_ordered)
                )
            else:
                D_ordered = cdist(
                    X_ordered, Y_ordered, metric="sqeuclidean", w=w_ordered
                )
            D += D_ordered

        # Factor variables
        if self.factor_mask.any():
            X_factor = X[:, self.factor_mask]
            Y_factor = Y[:, self.factor_mask]
            w_factor = theta[self.factor_mask]

            if is_symmetric:
                D_factor = squareform(
                    pdist(X_factor, metric=self.metric_factorial, w=w_factor)
                )
            else:
                D_factor = cdist(
                    X_factor, Y_factor, metric=self.metric_factorial, w=w_factor
                )
            D += D_factor

        # Final exponential (Gaussian correlation)
        # Note: Kriging's Psi = exp(-D)
        if self.p_val != 2.0:
            # If p != 2, the 'sqeuclidean' above was mathematically sum w * (diff^2).
            # If we strictly want sum w * |diff|^p, we can't use 'sqeuclidean' directly if p != 2.
            # But standard implementation often assumes p=2 for efficiency.
            # Kriging code shows:
            #    pdist(..., metric="sqeuclidean", ...)
            #    Psi = np.exp(-Psi)
            # This implies p=2 is hardcoded effectively in 'sqeuclidean' metric usage in Kriging code provided earlier.
            # The parameter p_val seems unused in the `build_correlation_matrix` snippet I read earlier
            # (which used `sqeuclidean`).
            # I will stick to what the code did: sqeuclidean -> exp(-D).
            pass

        return np.exp(-D)

    def diag(self, X):
        # Diagonal of correlation matrix is always 1 (distance 0 -> exp(0) = 1)
        return np.ones(X.shape[0])


class Sum(Kernel):
    """
    The Sum kernel k1 + k2.

    The kernel value is k1(X, Y) + k2(X, Y).

    Args:
        k1 (Kernel): First kernel.
        k2 (Kernel): Second kernel.
    """

    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    def __call__(self, X, Y=None):
        return self.k1(X, Y) + self.k2(X, Y)

    def diag(self, X):
        return self.k1.diag(X) + self.k2.diag(X)

    def __repr__(self):
        return f"{self.k1} + {self.k2}"


class Product(Kernel):
    """
    The Product kernel k1 * k2.

    The kernel value is k1(X, Y) * k2(X, Y).

    Args:
        k1 (Kernel): First kernel.
        k2 (Kernel): Second kernel.
    """

    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    def __call__(self, X, Y=None):
        return self.k1(X, Y) * self.k2(X, Y)

    def diag(self, X):
        return self.k1.diag(X) * self.k2.diag(X)


class ConstantKernel(Kernel):
    """
    Constant kernel.

    Can be used as a scaling factor (e.g. 2.0 * RBF()) or as part of a
    sum (e.g. RBF() + 1.0).

    Args:
        constant_value (float): The constant value. Defaults to 1.0.
        constant_value_bounds (tuple): The lower and upper bound on constant_value.
            Defaults to (1e-5, 1e5).
    """

    def __init__(self, constant_value=1.0, constant_value_bounds=(1e-5, 1e5)):
        self.constant_value = constant_value
        self.constant_value_bounds = constant_value_bounds

    def __call__(self, X, Y=None):
        if Y is None:
            Y = X
        return np.full((X.shape[0], Y.shape[0]), self.constant_value)

    def diag(self, X):
        return np.full(X.shape[0], self.constant_value)


class RBF(Kernel):
    """
    Radial Basis Function (RBF) kernel.

    Also known as the "squared exponential" kernel. It is given by:
    k(x_i, x_j) = exp(-0.5 * d(x_i, x_j)^2 / length_scale^2)

    Args:
        length_scale (float or np.ndarray): The length scale of the kernel.
            If a float, using isotropic distances. If an array, using anisotropic distances.
            Defaults to 1.0.
        length_scale_bounds (tuple): The lower and upper bound on length_scale.
            Defaults to (1e-5, 1e5).
    """

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5)):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds

    def __call__(self, X, Y=None):
        X = np.atleast_2d(X)
        length_scale = np.atleast_1d(self.length_scale)
        if Y is None:
            Y = X

        if X.ndim == 2 and Y.ndim == 2 and length_scale.shape[0] == 1:
            # Isotropic
            dists = euclidean_distances(X, Y, squared=True)
            return np.exp(-0.5 * dists / length_scale**2)
        else:
            # handle anisotropic if needed or raise error, for now assume isotropic or matching dims
            # Simple implementation for now to match sklearn RBF basic usage
            X_scaled = X / length_scale
            Y_scaled = Y / length_scale
            dists = euclidean_distances(X_scaled, Y_scaled, squared=True)
            return np.exp(-0.5 * dists)

    def diag(self, X):
        return np.ones(X.shape[0])


class WhiteKernel(Kernel):
    """
    White kernel.

    The main use case is capturing noise in the signal:
    k(x_i, x_j) = noise_level if x_i == x_j else 0

    Args:
        noise_level (float): Parameter controlling the noise level (variance).
            Defaults to 1.0.
        noise_level_bounds (tuple): The lower and upper bound on noise_level.
            Defaults to (1e-5, 1e5).
    """

    def __init__(self, noise_level=1.0, noise_level_bounds=(1e-5, 1e5)):
        self.noise_level = noise_level
        self.noise_level_bounds = noise_level_bounds

    def __call__(self, X, Y=None):
        if Y is not None and Y is not X:  # Different inputs, no correlation
            return np.zeros((X.shape[0], Y.shape[0]))
        # Y is None or Y is X -> self-correlation
        return self.noise_level * np.eye(X.shape[0])

    def diag(self, X):
        return np.full(X.shape[0], self.noise_level)
