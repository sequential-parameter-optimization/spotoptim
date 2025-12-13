import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics.pairwise import euclidean_distances


class Kernel(ABC):
    """
    Base class for Kernels.
    """

    @abstractmethod
    def __call__(self, X, Y=None):
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
    def diag(self, X):
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
