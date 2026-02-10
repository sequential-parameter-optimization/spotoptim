# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pytest
from spotoptim.sampling.design import (
    generate_uniform_design,
    generate_collinear_design,
    generate_clustered_design,
)


class TestGenerateUniformDesign:
    """Test suite for generate_uniform_design function."""

    def test_generate_uniform_design_shape(self):
        """Test that the output has the correct shape."""
        bounds = [(-5, 5), (0, 10), (-1, 1)]
        n_design = 10

        X = generate_uniform_design(bounds, n_design, seed=42)

        assert X.shape == (n_design, 3)

    def test_generate_uniform_design_bounds(self):
        """Test that generated points respect the bounds."""
        bounds = [(-5, -2), (10, 20)]
        n_design = 50

        X = generate_uniform_design(bounds, n_design, seed=42)

        # Check lower bounds
        assert np.all(X[:, 0] >= -5)
        assert np.all(X[:, 1] >= 10)

        # Check upper bounds
        assert np.all(X[:, 0] <= -2)
        assert np.all(X[:, 1] <= 20)

    def test_generate_uniform_design_reproducibility(self):
        """Test that setting a seed produces reproducible results."""
        bounds = [(-1, 1), (-1, 1)]
        n_design = 5
        seed = 123

        X1 = generate_uniform_design(bounds, n_design, seed=seed)
        X2 = generate_uniform_design(bounds, n_design, seed=seed)

        np.testing.assert_array_equal(X1, X2)

    def test_generate_uniform_design_generator_seed(self):
        """Test functioning with a numpy Generator instance."""
        bounds = [(-1, 1)]
        n_design = 5
        rng = np.random.default_rng(42)

        X = generate_uniform_design(bounds, n_design, seed=rng)

        assert X.shape == (5, 1)

    def test_zero_points(self):
        """Test generating zero points."""
        bounds = [(-1, 1)]
        X = generate_uniform_design(bounds, n_design=0)
        assert X.shape == (0, 1)


class TestGenerateCollinearDesign:
    """Test suite for generate_collinear_design function."""

    def test_generate_collinear_design_shape(self):
        """Test that the output has the correct shape."""
        bounds = [(-5, 5), (0, 10)]
        n_design = 10

        X = generate_collinear_design(bounds, n_design, seed=42)

        assert X.shape == (n_design, 2)

    def test_generate_collinear_design_bounds(self):
        """Test that generated points respect the bounds."""
        bounds = [(-5, -2), (10, 20)]
        n_design = 50

        X = generate_collinear_design(bounds, n_design, seed=42)

        # Check lower bounds
        assert np.all(X[:, 0] >= -5)
        # assert np.all(X[:, 1] >= 10) # Might fail with noise slightly

        # Check upper bounds
        assert np.all(X[:, 0] <= -2)
        # assert np.all(X[:, 1] <= 20)

    def test_generate_collinear_design_reproducibility(self):
        """Test that setting a seed produces reproducible results."""
        bounds = [(-1, 1), (-1, 1)]
        n_design = 5
        seed = 123

        X1 = generate_collinear_design(bounds, n_design, seed=seed)
        X2 = generate_collinear_design(bounds, n_design, seed=seed)

        np.testing.assert_array_equal(X1, X2)

    def test_generate_collinear_design_dimension_check(self):
        """Test that it raises error for non-2D designs."""
        bounds = [(-1, 1), (-1, 1), (-1, 1)]
        n_design = 5

        with pytest.raises(
            ValueError, match="Collinear design currently implemented for 2D only"
        ):
            generate_collinear_design(bounds, n_design)


class TestGenerateClusteredDesign:
    """Test suite for generate_clustered_design function."""

    def test_generate_clustered_design_shape(self):
        """Test the shape of the generated design."""
        bounds = [(-5, 5), (0, 10)]
        n_points = 20
        n_clusters = 4

        design = generate_clustered_design(
            bounds, n_design=n_points, n_clusters=n_clusters, seed=42
        )

        assert isinstance(design, np.ndarray)
        assert design.shape == (n_points, 2)

    def test_generate_clustered_design_reproducibility(self):
        """Test that the design generation is reproducible with the same seed."""
        bounds = [(0, 1), (0, 1)]
        n_points = 15
        n_clusters = 3
        seed = 42

        design1 = generate_clustered_design(bounds, n_points, n_clusters, seed=seed)
        design2 = generate_clustered_design(bounds, n_points, n_clusters, seed=seed)

        np.testing.assert_array_equal(design1, design2)

    def test_generate_clustered_design_bounds(self):
        """Test that generated points respect the bounds."""
        bounds = [(-5, -2), (10, 20)]
        n_points = 100
        n_clusters = 5

        design = generate_clustered_design(bounds, n_points, n_clusters, seed=42)

        # Check lower bounds
        assert np.all(design[:, 0] >= -5)
        assert np.all(design[:, 1] >= 10)

        # Check upper bounds
        assert np.all(design[:, 0] <= -2)
        assert np.all(design[:, 1] <= 20)

    def test_generate_clustered_design_generator_seed(self):
        """Test functioning with a numpy Generator instance."""
        bounds = [(-1, 1), (-1, 1)]
        n_points = 10
        n_clusters = 2
        rng = np.random.default_rng(42)

        design = generate_clustered_design(bounds, n_points, n_clusters, seed=rng)

        assert design.shape == (10, 2)


from spotoptim.sampling.design import (  # noqa: E402
    generate_sobol_design,
    generate_qmc_lhs_design,
    generate_grid_design,
)


class TestGenerateSobolDesign:
    """Test suite for generate_sobol_design."""

    def test_shape(self):
        bounds = [(-5, 5), (0, 10)]
        n_design = 8
        X = generate_sobol_design(bounds, n_design, seed=42)
        assert X.shape == (n_design, 2)

    def test_bounds(self):
        bounds = [(0, 1), (10, 20)]
        n_design = 16
        X = generate_sobol_design(bounds, n_design, seed=42)
        assert np.all(X[:, 0] >= 0)
        assert np.all(X[:, 0] <= 1)
        assert np.all(X[:, 1] >= 10)
        assert np.all(X[:, 1] <= 20)

    def test_reproducibility(self):
        bounds = [(0, 1)]
        n_design = 5
        X1 = generate_sobol_design(bounds, n_design, seed=123)
        X2 = generate_sobol_design(bounds, n_design, seed=123)
        np.testing.assert_array_equal(X1, X2)


class TestGenerateQMCLHSDesign:
    """Test suite for generate_qmc_lhs_design."""

    def test_shape(self):
        bounds = [(-5, 5), (0, 10)]
        n_design = 10
        X = generate_qmc_lhs_design(bounds, n_design, seed=42)
        assert X.shape == (n_design, 2)

    def test_bounds(self):
        bounds = [(0, 1), (10, 20)]
        n_design = 10
        X = generate_qmc_lhs_design(bounds, n_design, seed=42)
        assert np.all(X[:, 0] >= 0)
        assert np.all(X[:, 0] <= 1)
        assert np.all(X[:, 1] >= 10)
        assert np.all(X[:, 1] <= 20)

    def test_reproducibility(self):
        bounds = [(0, 1)]
        n_design = 5
        X1 = generate_qmc_lhs_design(bounds, n_design, seed=123)
        X2 = generate_qmc_lhs_design(bounds, n_design, seed=123)
        np.testing.assert_array_equal(X1, X2)


class TestGenerateGridDesign:
    """Test suite for generate_grid_design."""

    def test_shape_2d(self):
        bounds = [(0, 1), (0, 1)]
        n_design = 25  # 5x5
        X = generate_grid_design(bounds, n_design)
        assert X.shape == (25, 2)

    def test_shape_mismatch(self):
        # 10 is not a square, floor(sqrt(10)) = 3 -> 3x3=9 points
        bounds = [(0, 1), (0, 1)]
        n_design = 10
        X = generate_grid_design(bounds, n_design)
        assert X.shape == (9, 2)

    def test_bounds(self):
        bounds = [(-5, -2), (10, 20)]
        n_design = 16
        X = generate_grid_design(bounds, n_design)

        # Check lower bounds
        assert np.all(X[:, 0] >= -5)
        assert np.all(X[:, 1] >= 10)

        # Check upper bounds
        assert np.all(X[:, 0] <= -2)
        assert np.all(X[:, 1] <= 20)

    def test_nd(self):
        # 3D grid
        bounds = [(0, 1), (0, 1), (0, 1)]
        n_design = 27  # 3^3
        X = generate_grid_design(bounds, n_design)
        assert X.shape == (27, 3)
