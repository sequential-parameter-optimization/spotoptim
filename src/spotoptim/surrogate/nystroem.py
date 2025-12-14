import numpy as np
from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import pairwise_kernels
from .kernels import Kernel


class Nystroem:
    r"""
    Approximate a feature map of a kernel using a subset of data.

    The Nystroem method approximates a kernel map using a subset of the training data.
    It constructs an approximate feature map `X \mapsto X'` such that
    `X'.dot(X'.T) \approx K(X, X)`.

    This is particularly useful when:
    *   **n (samples) is moderate/large**: The exact kernel method scales as O(n^3).
        Nystroem reduces complexity to O(n * n_components^2) for training.
    *   **k (features) is large**: By setting `n_components` such that `k < n_components << n`,
        it projects high-dimensional data into a manageable feature space where
        distance calculations are cheaper (if followed by a linear model).

    Args:
        kernel (str or callable or Kernel, optional): Kernel map to be approximated.
            Can be a string (e.g., 'rbf'), a callable, or a `spotoptim.surrogate.kernels.Kernel`
            instance. Defaults to 'rbf'.
        n_components (int, optional): Number of features to construct.
            This corresponds to the number of samples used to construct the basis.
            Determines the dimension of the transformed feature space.
            Defaults to 100.
        random_state (int, RandomState instance or None, optional):
            Pseudo-random number generator to control the uniform sampling without replacement
            of n_components of the training data. Defaults to None.
    """

    def __init__(self, kernel="rbf", n_components=100, random_state=None):
        self.kernel = kernel
        self.n_components = n_components
        self.random_state = random_state
        self.components_ = None
        self.component_indices_ = None
        self.normalization_ = None

    def fit(self, X, y=None) -> "Nystroem":
        """
        Fit estimator to data.

        Samples a subset of `n_components` training points to serve as the basis,
        computes the kernel matrix on these points, and computes the normalization matrix.

        Args:
            X (np.ndarray): Training data, shape (n_samples, n_features).
            y (np.ndarray, optional): Target values (ignored).

        Returns:
            Nystroem: Returns the instance itself.
        """
        X = np.atleast_2d(X)
        rnd = check_random_state(self.random_state)
        n_samples = X.shape[0]

        # get basis vectors
        if self.n_components > n_samples:
            # User requested more components than samples.
            # We just take all samples.
            n_components = n_samples
        else:
            n_components = self.n_components

        inds = rnd.permutation(n_samples)
        basis_inds = inds[:n_components]
        basis = X[basis_inds]

        self.component_indices_ = basis_inds
        self.components_ = basis

        # Compute basis kernel matrix
        # If self.kernel is a Kernel object, call it.
        # If it is a string/callable, use pairwise_kernels.
        if isinstance(self.kernel, Kernel):
            K_W = self.kernel(basis, basis)
        else:
            # Fallback to sklearn's utility if standard string
            K_W = pairwise_kernels(basis, metric=self.kernel)

        # Compute normalization (svd decomposition of K_W)
        # K_W = U S U^T
        # We need W^(-1/2) for the feature map
        # But for stability usually use SVD: K_W = U @ S @ V.T (symmetric so U=V)
        # normalization = U / sqrt(S)
        U, S, V = np.linalg.svd(K_W)
        S = np.maximum(S, 1e-12)
        self.normalization_ = np.dot(U / np.sqrt(S), V)

        return self

    def transform(self, X) -> np.ndarray:
        """
        Apply feature map to X.

        Computes the kernel between X and the basis vectors, multiplied by the normalization.
        No sample reduction happens here; the output has the same number of samples as X.

        Args:
            X (np.ndarray): Data to transform, shape (n_samples, n_features).

        Returns:
            np.ndarray: Transformed data, shape (n_samples, n_components).
        """
        X = np.atleast_2d(X)
        if isinstance(self.kernel, Kernel):
            K_sum = self.kernel(X, self.components_)
        else:
            K_sum = pairwise_kernels(X, self.components_, metric=self.kernel)

        return np.dot(K_sum, self.normalization_.T)

    def fit_transform(self, X, y=None) -> np.ndarray:
        """
        Fit to data, then transform it.

        Args:
            X (np.ndarray): Training data.
            y (np.ndarray, optional): Target values.

        Returns:
            np.ndarray: Transformed data.
        """
        return self.fit(X, y).transform(X)
