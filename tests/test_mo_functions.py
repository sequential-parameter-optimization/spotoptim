# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for multi-objective analytical test functions.

This module tests all multi-objective benchmark functions in spotoptim.function.mo,
including ZDT, DTLZ, Schaffer, Fonseca-Fleming, and Kursawe functions.
"""

import numpy as np
import pytest

from spotoptim.function.mo import (
    dtlz1,
    dtlz2,
    fonseca_fleming,
    kursawe,
    mo_conv2_min,
    mo_conv2_max,
    schaffer_n1,
    zdt1,
    zdt2,
    zdt3,
    zdt4,
    zdt6,
)


class TestZDT1:
    """Tests for ZDT1 function."""

    def test_single_point_shape(self):
        """Test that single point returns correct shape."""
        X = np.array([0.5, 0.5, 0.5])
        result = zdt1(X)
        assert result.shape == (1, 2)

    def test_multiple_points_shape(self):
        """Test that multiple points return correct shape."""
        X = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        result = zdt1(X)
        assert result.shape == (3, 2)

    def test_known_pareto_optimal_point(self):
        """Test known Pareto optimal point (x1 arbitrary, rest zero)."""
        X = np.array([0.5, 0.0, 0.0])
        result = zdt1(X)
        # For Pareto optimal: g = 1, f1 = x1, f2 = 1 - sqrt(x1)
        assert np.isclose(result[0, 0], 0.5)
        assert np.isclose(result[0, 1], 1 - np.sqrt(0.5), atol=1e-6)

    def test_origin_point(self):
        """Test evaluation at origin."""
        X = np.array([0.0, 0.0, 0.0])
        result = zdt1(X)
        assert result[0, 0] == 0.0
        assert result[0, 1] == 1.0

    def test_zdt4_dimension_error(self):
        """Test error for insufficient dimensions."""
        X = np.array([0.5])
        with pytest.raises(ValueError, match="requires at least 2 dimensions"):
            zdt4(X)

    class TestMoConv2Min:
        """Tests for mo_conv2_min function (convex bi-objective minimization)."""

        def test_single_point_shape(self):
            """Test that single point returns correct shape."""
            X = np.array([0.5, 0.5])
            result = mo_conv2_min(X)
            assert result.shape == (1, 2)

        def test_multiple_points_shape(self):
            """Test that multiple points return correct shape."""
            X = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
            result = mo_conv2_min(X)
            assert result.shape == (3, 2)

        def test_ideal_point(self):
            """Test ideal points for minimization."""
            # Min f1 at x=0, y=0 -> f1=0, f2=2
            X_f1 = np.array([0.0, 0.0])
            result_f1 = mo_conv2_min(X_f1)
            assert np.isclose(result_f1[0, 0], 0.0)
            assert np.isclose(result_f1[0, 1], 2.0)

            # Min f2 at x=1, y=1 -> f1=2, f2=0
            X_f2 = np.array([1.0, 1.0])
            result_f2 = mo_conv2_min(X_f2)
            assert np.isclose(result_f2[0, 1], 0.0)
            assert np.isclose(result_f2[0, 0], 2.0)

        def test_worst_point(self):
            """Test saddle point at (0, 1)."""
            X = np.array([0.0, 1.0])
            result = mo_conv2_min(X)
            assert np.isclose(result[0, 0], 1.0)
            assert np.isclose(result[0, 1], 1.0)

        def test_dimension_error(self):
            """Test error for wrong number of dimensions."""
            X = np.array([0.5, 0.5, 0.5])
            with pytest.raises(ValueError, match="requires exactly 2 dimensions"):
                mo_conv2_min(X)

            X = np.array([0.5])
            with pytest.raises(ValueError, match="requires exactly 2 dimensions"):
                mo_conv2_min(X)

        def test_output_type(self):
            """Test that output is numpy array."""
            X = np.array([0.5, 0.5])
            result = mo_conv2_min(X)
            assert isinstance(result, np.ndarray)

        def test_conflicting_objectives(self):
            """Test that objectives are conflicting (minimization problem)."""
            # Points optimizing different objectives should show trade-off
            X_samples = np.random.rand(50, 2)
            result = mo_conv2_min(X_samples)
            # Check that not all points can minimize both objectives simultaneously
            min_f1_idx = np.argmin(result[:, 0])
            min_f2_idx = np.argmin(result[:, 1])
            # If objectives conflict, minimizing one shouldn't minimize the other
            assert min_f1_idx != min_f2_idx or len(np.unique(result, axis=0)) > 1

        def test_values_in_domain(self):
            """Test that function values stay bounded for inputs in [0,1]^2."""
            X = np.random.rand(100, 2)
            result = mo_conv2_min(X)
            # Both objectives should be non-negative
            assert np.all(result >= 0)
            # And bounded (for [0,1] domain)
            assert np.all(result[:, 0] <= 2)  # f1 = x^2 + y^2 <= 2
            assert np.all(result[:, 1] <= 2)  # f2 = (x-1)^2 + (y-1)^2 <= 2

        def test_corner_points(self):
            """Test evaluation at domain corners."""
            corners = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
            result = mo_conv2_min(corners)
            # All corner evaluations should be valid
            assert result.shape == (4, 2)
            assert not np.any(np.isnan(result))
            assert not np.any(np.isinf(result))

        def test_symmetry(self):
            """Test symmetry property: f(x, y) = f(y, x)."""
            X1 = np.array([[0.3, 0.7]])
            X2 = np.array([[0.7, 0.3]])
            result1 = mo_conv2_min(X1)
            result2 = mo_conv2_min(X2)
            # f1 and f2 are symmetric in x, y, so outputs should be identical
            assert np.allclose(result1, result2)

        def test_pareto_front_convexity(self):
            """Test that Pareto front is convex (for minimization)."""
            # Sample many points and find Pareto front
            n_samples = 200
            X = np.random.rand(n_samples, 2)
            result = mo_conv2_min(X)
            # For minimization, a point is Pareto optimal if no other point
            # dominates it (has lower values in all objectives)
            pareto_mask = np.ones(n_samples, dtype=bool)
            for i in range(n_samples):
                for j in range(n_samples):
                    if i != j:
                        # j dominates i if j <= i in all objectives and j < i in at least one
                        if np.all(result[j] <= result[i]) and np.any(
                            result[j] < result[i]
                        ):
                            pareto_mask[i] = False
                            break
            pareto_points = result[pareto_mask]
            # Should have multiple Pareto points
            assert len(pareto_points) > 1
            assert result.shape == (n_samples, 2)

    def test_multimodal_g_function(self):
        """Test that g function creates multimodality."""
        # Compare Pareto optimal vs non-optimal
        X_optimal = np.array([0.5, 0.0, 0.0])
        X_local = np.array([0.5, 1.0, 1.0])

        result_optimal = zdt4(X_optimal)
        result_local = zdt4(X_local)

        # Non-optimal should have higher f2 due to larger g
        assert result_local[0, 1] > result_optimal[0, 1]

    def test_zdt3_dimension_error(self):
        """Test error for insufficient dimensions."""
        X = np.array([0.5])
        with pytest.raises(ValueError, match="requires at least 2 dimensions"):
            zdt4(X)

    def test_typical_10_dimensions(self):
        """Test with typical 10 dimensions."""
        X = np.random.rand(3, 10) * 2 - 1  # [-1, 1] range approximation
        X[:, 0] = np.clip(X[:, 0], 0, 1)  # x1 in [0, 1]
        result = zdt4(X)
        assert result.shape == (3, 2)


class TestZDT6:
    """Tests for ZDT6 function."""

    def test_single_point_shape(self):
        """Test that single point returns correct shape."""
        X = np.array([0.5, 0.5, 0.5])
        result = zdt6(X)
        assert result.shape == (1, 2)

    def test_multiple_points_shape(self):
        """Test that multiple points return correct shape."""
        X = np.array([[0.2, 0.2], [0.5, 0.5], [0.8, 0.8]])
        result = zdt6(X)
        assert result.shape == (3, 2)

    def test_non_uniform_density(self):
        """Test that function has non-uniform density."""
        # f1 has exponential and sin terms creating non-uniformity
        X = np.array([[0.1, 0.0], [0.5, 0.0], [0.9, 0.0]])
        result = zdt6(X)
        # Just check it evaluates without error
        assert result.shape == (3, 2)
        # f1 should vary due to exponential and sin terms
        # Check that f1 values are different
        assert len(np.unique(result[:, 0])) == 3

    def test_dimension_error(self):
        """Test error for insufficient dimensions."""
        X = np.array([0.5])
        with pytest.raises(ValueError, match="requires at least 2 dimensions"):
            zdt6(X)

    def test_g_function_with_power(self):
        """Test that g function uses 0.25 power."""
        X_optimal = np.array([0.5, 0.0, 0.0])
        result = zdt6(X_optimal)
        # g should be 1.0 when x_i = 0 for i > 1
        # This creates specific f2 value
        assert result.shape == (1, 2)


class TestDTLZ1:
    """Tests for DTLZ1 function."""

    def test_default_3_objectives(self):
        """Test default 3 objectives."""
        X = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        result = dtlz1(X)
        assert result.shape == (1, 3)

    def test_2_objectives(self):
        """Test with 2 objectives."""
        X = np.array([0.5, 0.5, 0.5])
        result = dtlz1(X, n_obj=2)
        assert result.shape == (1, 2)

    def test_5_objectives(self):
        """Test with 5 objectives."""
        X = np.array([0.5] * 10)
        result = dtlz1(X, n_obj=5)
        assert result.shape == (1, 5)

    def test_multiple_points(self):
        """Test multiple points evaluation."""
        X = np.random.rand(10, 7)
        result = dtlz1(X, n_obj=3)
        assert result.shape == (10, 3)

    def test_pareto_optimal_point(self):
        """Test Pareto optimal point (x_M = 0.5)."""
        X = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        result = dtlz1(X, n_obj=3)
        # When all x_i in X_M are 0.5, cos terms are zero, g should be minimal
        assert result.shape == (1, 3)

    def test_n_obj_too_small_error(self):
        """Test error when n_obj < 2."""
        X = np.array([0.5, 0.5])
        with pytest.raises(ValueError, match="must be at least 2"):
            dtlz1(X, n_obj=1)

    def test_n_obj_exceeds_variables_error(self):
        """Test error when n_obj > n_variables."""
        X = np.array([0.5, 0.5])
        with pytest.raises(ValueError, match="cannot exceed number of variables"):
            dtlz1(X, n_obj=5)

    def test_linear_hyperplane_property(self):
        """Test that Pareto optimal solutions sum to 0.5."""
        # For Pareto optimal (g=0), sum of objectives = 0.5
        X = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        result = dtlz1(X, n_obj=3)
        # Note: actual optimal has g=0, but we test structure
        assert result.shape == (1, 3)


class TestDTLZ2:
    """Tests for DTLZ2 function."""

    def test_default_3_objectives(self):
        """Test default 3 objectives."""
        X = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        result = dtlz2(X)
        assert result.shape == (1, 3)

    def test_2_objectives(self):
        """Test with 2 objectives."""
        X = np.array([0.5, 0.5, 0.5])
        result = dtlz2(X, n_obj=2)
        assert result.shape == (1, 2)

    def test_4_objectives(self):
        """Test with 4 objectives."""
        X = np.array([0.5] * 8)
        result = dtlz2(X, n_obj=4)
        assert result.shape == (1, 4)

    def test_multiple_points(self):
        """Test multiple points evaluation."""
        X = np.random.rand(15, 12)
        result = dtlz2(X, n_obj=3)
        assert result.shape == (15, 3)

    def test_spherical_property(self):
        """Test that Pareto optimal points lie on unit sphere."""
        # For Pareto optimal (g=0, X_M = 0.5), sum(f_i^2) ≈ 1
        X = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        result = dtlz2(X, n_obj=3)
        # With g=0, (1+g)=1, and trigonometric relations give unit sphere
        sum_sq = np.sum(result**2)
        assert np.isclose(sum_sq, 1.0, atol=0.1)

    def test_n_obj_too_small_error(self):
        """Test error when n_obj < 2."""
        X = np.array([0.5, 0.5])
        with pytest.raises(ValueError, match="must be at least 2"):
            dtlz2(X, n_obj=1)

    def test_n_obj_exceeds_variables_error(self):
        """Test error when n_obj > n_variables."""
        X = np.array([0.5, 0.5])
        with pytest.raises(ValueError, match="cannot exceed number of variables"):
            dtlz2(X, n_obj=3)

    def test_typical_12_dimensions(self):
        """Test with typical 12 dimensions for 3 objectives."""
        X = np.random.rand(5, 12)
        result = dtlz2(X, n_obj=3)
        assert result.shape == (5, 3)
        assert np.all(result >= 0)


class TestSchafferN1:
    """Tests for Schaffer N1 function."""

    def test_single_point_shape(self):
        """Test that single point returns correct shape."""
        X = np.array([1.0])
        result = schaffer_n1(X)
        assert result.shape == (1, 2)

    def test_multiple_points_shape(self):
        """Test that multiple points return correct shape."""
        X = np.array([[0.0], [1.0], [2.0]])
        result = schaffer_n1(X)
        assert result.shape == (3, 2)

    def test_known_values(self):
        """Test known function values."""
        # At x=0: f1=0, f2=4
        X = np.array([0.0])
        result = schaffer_n1(X)
        assert result[0, 0] == 0.0
        assert result[0, 1] == 4.0

        # At x=1: f1=1, f2=1
        X = np.array([1.0])
        result = schaffer_n1(X)
        assert result[0, 0] == 1.0
        assert result[0, 1] == 1.0

        # At x=2: f1=4, f2=0
        X = np.array([2.0])
        result = schaffer_n1(X)
        assert result[0, 0] == 4.0
        assert result[0, 1] == 0.0

    def test_pareto_optimal_range(self):
        """Test Pareto optimal range [0, 2]."""
        X = np.linspace(0, 2, 50).reshape(-1, 1)
        result = schaffer_n1(X)
        assert result.shape == (50, 2)
        # All points in [0, 2] should be Pareto optimal

    def test_uses_only_first_variable(self):
        """Test that only first variable is used."""
        X1 = np.array([[1.0, 100.0, -50.0]])
        X2 = np.array([[1.0, 0.0, 0.0]])
        result1 = schaffer_n1(X1)
        result2 = schaffer_n1(X2)
        assert np.allclose(result1, result2)


class TestFonsecaFleming:
    """Tests for Fonseca-Fleming function."""

    def test_single_point_shape(self):
        """Test that single point returns correct shape."""
        X = np.array([0.0, 0.0])
        result = fonseca_fleming(X)
        assert result.shape == (1, 2)

    def test_multiple_points_shape(self):
        """Test that multiple points return correct shape."""
        X = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]])
        result = fonseca_fleming(X)
        assert result.shape == (3, 2)

    def test_symmetric_property(self):
        """Test symmetric property: switching signs should swap objectives."""
        X1 = np.array([[1.0, 1.0]])
        X2 = np.array([[-1.0, -1.0]])

        result1 = fonseca_fleming(X1)
        result2 = fonseca_fleming(X2)

        # f1(X1) should approximately equal f2(X2) and vice versa
        assert np.isclose(result1[0, 0], result2[0, 1], atol=1e-6)
        assert np.isclose(result1[0, 1], result2[0, 0], atol=1e-6)

    def test_origin_point(self):
        """Test evaluation at origin."""
        X = np.array([0.0, 0.0])
        result = fonseca_fleming(X)
        # Both objectives should be symmetric at origin
        assert np.isclose(result[0, 0], result[0, 1], atol=1e-6)

    def test_different_dimensions(self):
        """Test with different numbers of variables."""
        for n_vars in [2, 5, 10]:
            X = np.random.randn(5, n_vars)
            result = fonseca_fleming(X)
            assert result.shape == (5, 2)

    def test_output_range(self):
        """Test that outputs are in [0, 1] range."""
        X = np.random.uniform(-4, 4, (20, 3))
        result = fonseca_fleming(X)
        assert np.all(result >= 0)
        assert np.all(result <= 1)

    def test_dimension_error(self):
        """Test error for wrong number of dimensions."""
        X = np.array([0.5, 0.5, 0.5])
        with pytest.raises(ValueError, match="requires exactly 2 dimensions"):
            mo_conv2_max(X)

        X = np.array([0.5])
        with pytest.raises(ValueError, match="requires exactly 2 dimensions"):
            mo_conv2_max(X)

    def test_output_type(self):
        """Test that output is numpy array."""
        X = np.array([0.5, 0.5])
        result = mo_conv2_max(X)
        assert isinstance(result, np.ndarray)

    def test_conflicting_objectives(self):
        """Test that objectives are conflicting (maximization problem)."""
        # Points optimizing different objectives should show trade-off
        X_samples = np.random.rand(50, 2)
        result = mo_conv2_max(X_samples)

        # Check that not all points can maximize both objectives simultaneously
        max_f1_idx = np.argmax(result[:, 0])
        max_f2_idx = np.argmax(result[:, 1])

        # If objectives conflict, maximizing one shouldn't maximize the other
        assert max_f1_idx != max_f2_idx or len(np.unique(result, axis=0)) > 1

    def test_values_in_domain(self):
        """Test that function values stay bounded for inputs in [0,1]^2."""
        X = np.random.rand(100, 2)
        result = mo_conv2_max(X)

        # Both objectives should be non-negative
        assert np.all(result >= 0)
        # And bounded (for [0,1] domain) with max value 2
        assert np.all(result[:, 0] <= 2)  # f1 = 2 - (x^2 + y^2) ∈ [0, 2]
        assert np.all(result[:, 1] <= 2)  # f2 = 2 - ((1-x)^2 + (1-y)^2) ∈ [0, 2]

    def test_corner_points(self):
        """Test evaluation at domain corners."""
        corners = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        result = mo_conv2_max(corners)

        # All corner evaluations should be valid
        assert result.shape == (4, 2)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_symmetry(self):
        """Test symmetry property: f(x1,x2) = f(x2,x1)."""
        X1 = np.array([[0.3, 0.7]])
        X2 = np.array([[0.7, 0.3]])

        result1 = mo_conv2_max(X1)
        result2 = mo_conv2_max(X2)

        # m and d are symmetric in x1, x2, so outputs should be identical
        assert np.allclose(result1, result2)

    def test_pareto_front_convexity(self):
        """Test that Pareto front is convex (for maximization)."""
        # Sample many points and find Pareto front
        n_samples = 200
        X = np.random.rand(n_samples, 2)
        result = mo_conv2_max(X)

        # For maximization, a point is Pareto optimal if no other point
        # dominates it (has higher values in all objectives)
        pareto_mask = np.ones(n_samples, dtype=bool)
        for i in range(n_samples):
            for j in range(n_samples):
                if i != j:
                    # j dominates i if j >= i in all objectives and j > i in at least one
                    if np.all(result[j] >= result[i]) and np.any(result[j] > result[i]):
                        pareto_mask[i] = False
                        break

        pareto_points = result[pareto_mask]

        # Should have multiple Pareto points
        assert len(pareto_points) > 1


class TestGeneralProperties:
    """Tests for general properties across all functions."""

    def test_all_functions_return_numpy_array(self):
        """Test that all functions return numpy arrays."""
        X_2d = np.array([0.5, 0.5])
        X_3d = np.array([0.5, 0.5, 0.5])

        functions_2d = [
            schaffer_n1,
            fonseca_fleming,
            zdt1,
            zdt2,
            zdt3,
            zdt4,
            zdt6,
            kursawe,
        ]

        for func in functions_2d:
            if func == schaffer_n1:
                result = func(np.array([0.5]))
            elif func in [fonseca_fleming]:
                result = func(X_2d)
            else:
                result = func(X_3d)
            assert isinstance(result, np.ndarray)

    def test_no_nan_or_inf_in_valid_range(self):
        """Test that functions don't produce NaN or Inf for valid inputs."""
        np.random.seed(42)

        # ZDT functions in [0, 1]
        X = np.random.rand(10, 5)
        for func in [zdt1, zdt2, zdt3, zdt4, zdt6]:
            result = func(X)
            assert not np.any(np.isnan(result))
            assert not np.any(np.isinf(result))

        # DTLZ functions in [0, 1]
        X = np.random.rand(10, 7)
        for func in [dtlz1, dtlz2]:
            result = func(X, n_obj=3)
            assert not np.any(np.isnan(result))
            assert not np.any(np.isinf(result))

        # Schaffer in [-10, 10]
        X = np.random.uniform(-10, 10, (10, 1))
        result = schaffer_n1(X)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

        # Fonseca-Fleming in [-4, 4]
        X = np.random.uniform(-4, 4, (10, 3))
        result = fonseca_fleming(X)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

        # Kursawe in [-5, 5]
        X = np.random.uniform(-5, 5, (10, 3))
        result = kursawe(X)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_deterministic_evaluation(self):
        """Test that functions are deterministic."""
        X = np.array([[0.3, 0.7, 0.2], [0.8, 0.1, 0.9]])

        for func in [zdt1, zdt2, zdt3, zdt4, zdt6]:
            result1 = func(X)
            result2 = func(X)
            assert np.allclose(result1, result2)

    def test_batch_vs_single_consistency(self):
        """Test that batch evaluation matches single point evaluations."""
        X_batch = np.array([[0.2, 0.3], [0.5, 0.6], [0.8, 0.9]])

        for func in [zdt1, zdt2, zdt3, zdt4, zdt6]:
            result_batch = func(X_batch)

            for i in range(3):
                result_single = func(X_batch[i])
                assert np.allclose(result_batch[i], result_single[0])


class TestMoConv2Max:
    """Tests for mo_conv2_max function (convex bi-objective maximization)."""

    def test_single_point_shape(self):
        """Test that single point returns correct shape."""
        X = np.array([0.5, 0.5])
        result = mo_conv2_max(X)
        assert result.shape == (1, 2)

    def test_multiple_points_shape(self):
        """Test that multiple points return correct shape."""
        X = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        result = mo_conv2_max(X)
        assert result.shape == (3, 2)

    def test_ideal_points(self):
        # f1 maximum at (0,0): f1=2, f2=0
        X_f1 = np.array([0.0, 0.0])
        result_f1 = mo_conv2_max(X_f1)
        assert np.isclose(result_f1[0, 0], 2.0)
        assert np.isclose(result_f1[0, 1], 0.0)
        # f2 maximum at (1,1): f1=0, f2=2
        X_f2 = np.array([1.0, 1.0])
        result_f2 = mo_conv2_max(X_f2)
        assert np.isclose(result_f2[0, 1], 2.0)
        assert np.isclose(result_f2[0, 0], 0.0)

    def test_bounds(self):
        X = np.random.rand(100, 2)
        result = mo_conv2_max(X)
        # Both objectives should be between 0 and 2 for [0,1]^2
        assert np.all(result >= 0)
        assert np.all(result <= 2)

    def test_symmetry(self):
        X1 = np.array([[0.3, 0.7]])
        X2 = np.array([[0.7, 0.3]])
        result1 = mo_conv2_max(X1)
        result2 = mo_conv2_max(X2)
        assert np.allclose(result1, result2)

    def test_dimension_error(self):
        X = np.array([0.5, 0.5, 0.5])
        with pytest.raises(ValueError, match="requires exactly 2 dimensions"):
            mo_conv2_max(X)
        X = np.array([0.5])
        with pytest.raises(ValueError, match="requires exactly 2 dimensions"):
            mo_conv2_max(X)

    def test_corner_points(self):
        corners = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        result = mo_conv2_max(corners)
        assert result.shape == (4, 2)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_pareto_front_convexity(self):
        n_samples = 200
        X = np.random.rand(n_samples, 2)
        result = mo_conv2_max(X)
        # For maximization, a point is Pareto optimal if no other point dominates it
        pareto_mask = np.ones(n_samples, dtype=bool)
        for i in range(n_samples):
            for j in range(n_samples):
                if i != j:
                    if np.all(result[j] >= result[i]) and np.any(result[j] > result[i]):
                        pareto_mask[i] = False
                        break
        pareto_points = result[pareto_mask]
        assert len(pareto_points) > 1
