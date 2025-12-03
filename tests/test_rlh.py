"""
Tests for spotoptim.sampling.lhs.rlh
"""
import numpy as np
import pytest

from spotoptim.sampling.lhs import rlh


class TestRLHBasic:
    def test_shape_and_dtype_edges0(self):
        np.random.seed(123)
        n, k = 10, 3
        X = rlh(n=n, k=k, edges=0)
        assert X.shape == (n, k)
        assert X.dtype == float

    def test_values_in_unit_interval_edges0(self):
        np.random.seed(1)
        n, k = 8, 4
        X = rlh(n, k, edges=0)
        assert np.all(X > 0.0)
        assert np.all(X < 1.0)
        # Check expected min/max midpoints
        expected_min = 0.5 / n
        expected_max = (n - 0.5) / n
        assert np.isclose(X.min(), expected_min)
        assert np.isclose(X.max(), expected_max)

    def test_columns_cover_all_midpoint_bins_edges0(self):
        np.random.seed(7)
        n, k = 12, 5
        X = rlh(n, k, edges=0)
        expected_bins = np.sort((np.arange(n) + 0.5) / n)
        for j in range(k):
            col = np.sort(X[:, j])
            assert np.allclose(col, expected_bins)


class TestRLHEdgesOne:
    def test_values_in_unit_interval_edges1(self):
        np.random.seed(2)
        n, k = 9, 3
        X = rlh(n, k, edges=1)
        assert np.all(X >= 0.0)
        assert np.all(X <= 1.0)
        # Check that 0 and 1 are attainable
        assert np.isclose(X.min(), 0.0)
        assert np.isclose(X.max(), 1.0)

    def test_columns_cover_all_edge_bins_edges1(self):
        np.random.seed(9)
        n, k = 6, 4
        X = rlh(n, k, edges=1)
        expected_bins = np.sort(np.arange(n) / (n - 1))
        for j in range(k):
            col = np.sort(X[:, j])
            assert np.allclose(col, expected_bins)


class TestRLHProperties:
    def test_each_column_is_permutation(self):
        np.random.seed(3)
        n, k = 7, 3
        X = rlh(n, k, edges=0)
        # Map back to bin indices to check permutation property
        bins = np.argsort(np.argsort(X[:, 0]))  # not reliable since values unique but shuffled
        # Instead, verify uniqueness per column
        for j in range(k):
            assert len(np.unique(X[:, j])) == n

    def test_reproducibility_with_seed(self):
        n, k = 10, 2
        np.random.seed(42)
        X1 = rlh(n, k, edges=0)
        np.random.seed(42)
        X2 = rlh(n, k, edges=0)
        assert np.array_equal(X1, X2)

    def test_large_dimensions(self):
        np.random.seed(5)
        n, k = 50, 20
        X = rlh(n, k, edges=0)
        assert X.shape == (50, 20)
        assert np.all((X > 0.0) & (X < 1.0))


class TestRLHInvalidInputs:
    def test_n_less_than_one_raises(self):
        with pytest.raises(ValueError):
            rlh(n=0, k=2)

    def test_k_less_than_one_raises(self):
        with pytest.raises(ValueError):
            rlh(n=3, k=0)

    def test_negative_n_raises(self):
        with pytest.raises(ValueError):
            rlh(n=-5, k=2)

    def test_negative_k_raises(self):
        with pytest.raises(ValueError):
            rlh(n=5, k=-2)

    def test_invalid_edges_value_raises(self):
        with pytest.raises(ValueError):
            rlh(n=5, k=2, edges=2)

    def test_single_point_edges_one_is_zero(self):
        np.random.seed(0)
        X = rlh(n=1, k=3, edges=1)
        assert X.shape == (1, 3)
        assert np.allclose(X, 0.0)
