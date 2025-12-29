
import numpy as np
import pytest
from spotoptim.sampling.lhs import rlh

class TestRlh:
    """Test suite for rlh (Random Latin Hypercube) function."""

    def test_rlh_shape(self):
        """Test output shape."""
        n, k = 10, 3
        X = rlh(n, k)
        assert X.shape == (n, k)
        assert np.all(X >= 0)
        assert np.all(X <= 1)

    def test_rlh_reproducibility(self):
        """Test reproducibility with seed."""
        n, k = 5, 2
        seed = 42
        X1 = rlh(n, k, seed=seed)
        X2 = rlh(n, k, seed=seed)
        np.testing.assert_array_equal(X1, X2)

        X3 = rlh(n, k, seed=seed + 1)
        # Unlikely to be equal
        assert not np.array_equal(X1, X3)

    def test_rlh_edges(self):
        """Test edges parameter."""
        n, k = 5, 2
        # Edges=1 means 0 and 1 should be reachable potentially (or at least full range)
        # Specifically: (x)/(n-1). so if x=0 -> 0, x=n-1 -> 1.
        X = rlh(n, k, edges=1, seed=42)
        
        # Check that we can map back to integers
        # X * (n-1) should be close to integers
        X_int = X * (n - 1)
        np.testing.assert_allclose(X_int, np.round(X_int))
        
        # Check that in each column we have a permutation of 0..n-1
        for col in range(k):
            vals = np.round(X[:, col] * (n - 1)).astype(int)
            assert set(vals) == set(range(n))

    def test_rlh_midpoints(self):
        """Test default edges=0 (midpoints)."""
        n, k = 5, 2
        X = rlh(n, k, edges=0, seed=42)
        
        # (x + 0.5) / n
        # X * n - 0.5 should be integers
        X_int = X * n - 0.5
        np.testing.assert_allclose(X_int, np.round(X_int))
        
        for col in range(k):
            vals = np.round(X_int[:, col]).astype(int)
            assert set(vals) == set(range(n))

    def test_invalid_args(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            rlh(0, 2)
        with pytest.raises(ValueError, match="k must be >= 1"):
            rlh(5, 0)
        with pytest.raises(ValueError, match="edges must be 0 or 1"):
            rlh(5, 2, edges=2)
