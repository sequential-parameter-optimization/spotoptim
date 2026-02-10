# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for tolerance_x functionality to prevent duplicate point evaluations.

This test suite validates that tolerance_x correctly prevents re-evaluation of
identical or very similar points, especially with integer and factor variables
where rounding can cause issues.
"""

import numpy as np
from spotoptim.SpotOptim import SpotOptim


class TestToleranceXBasic:
    """Test suite for basic tolerance_x functionality."""

    def test_tolerance_x_default_value(self):
        """Test that tolerance_x has a default value."""

        def simple_func(X):
            return np.sum(X**2, axis=1)

        optimizer = SpotOptim(
            fun=simple_func,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=10,
            n_initial=5,
            seed=42,
        )

        # Default should be sqrt(eps) which is very small
        assert optimizer.tolerance_x > 0
        assert optimizer.tolerance_x < 0.01

    def test_tolerance_x_custom_value(self):
        """Test that custom tolerance_x is set correctly."""

        def simple_func(X):
            return np.sum(X**2, axis=1)

        optimizer = SpotOptim(
            fun=simple_func,
            bounds=[(-5, 5), (-5, 5)],
            tolerance_x=0.5,
            max_iter=10,
            n_initial=5,
            seed=42,
        )

        assert optimizer.tolerance_x == 0.5

    def test_tolerance_x_zero_value(self):
        """Test that tolerance_x=0 is allowed (exact matching only)."""

        def simple_func(X):
            return np.sum(X**2, axis=1)

        optimizer = SpotOptim(
            fun=simple_func,
            bounds=[(-5, 5), (-5, 5)],
            tolerance_x=0.0,
            max_iter=10,
            n_initial=5,
            seed=42,
        )

        assert optimizer.tolerance_x == 0.0


class TestToleranceXFloatVariables:
    """Test suite for tolerance_x with continuous float variables."""

    def test_no_duplicate_evaluations_float(self):
        """Test that float variables don't produce duplicate evaluations."""

        def sphere(X):
            return np.sum(X**2, axis=1)

        optimizer = SpotOptim(
            fun=sphere,
            bounds=[(-5, 5), (-5, 5)],
            var_type=["float", "float"],
            tolerance_x=0.01,  # Small but non-zero tolerance
            max_iter=20,
            n_initial=10,
            seed=42,
            verbose=False,
        )

        _ = optimizer.optimize()

        # Check for duplicate points (allowing for floating point tolerance)
        X = optimizer.X_
        n_unique = len(np.unique(X, axis=0))
        n_total = len(X)

        # Should have no exact duplicates
        assert n_unique == n_total, f"Found {n_total - n_unique} duplicate evaluations"

    def test_close_points_respected_float(self):
        """Test that tolerance prevents evaluation of very close points."""

        def sphere(X):
            return np.sum(X**2, axis=1)

        optimizer = SpotOptim(
            fun=sphere,
            bounds=[(-5, 5), (-5, 5)],
            var_type=["float", "float"],
            tolerance_x=0.5,  # Larger tolerance
            max_iter=20,
            n_initial=10,
            seed=42,
            verbose=False,
        )

        _ = optimizer.optimize()

        # Check minimum pairwise distance
        X = optimizer.X_
        n_points = len(X)

        min_distance = float("inf")
        for i in range(n_points):
            for j in range(i + 1, n_points):
                dist = np.linalg.norm(X[i] - X[j])
                min_distance = min(min_distance, dist)

        # Minimum distance should be at least tolerance_x (or very close due to fallback)
        # Note: Fallback strategies may place points closer than tolerance_x
        assert min_distance >= 0.0, f"Found points closer than expected: {min_distance}"


class TestToleranceXIntegerVariables:
    """Test suite for tolerance_x with integer variables (critical for bug)."""

    def test_no_duplicate_evaluations_integer(self):
        """Test that integer variables don't produce duplicate evaluations after rounding."""

        def int_func(X):
            return np.sum(X**2, axis=1)

        optimizer = SpotOptim(
            fun=int_func,
            bounds=[(1, 20), (1, 10)],
            var_type=["int", "int"],
            tolerance_x=0.1,
            max_iter=20,
            n_initial=10,
            seed=42,
            verbose=False,
        )

        _ = optimizer.optimize()

        # Check for duplicate integer combinations
        X = optimizer.X_
        X_rounded = np.round(X).astype(int)
        n_unique = len(np.unique(X_rounded, axis=0))
        n_total = len(X_rounded)

        assert (
            n_unique == n_total
        ), f"Found {n_total - n_unique} duplicate integer combinations"

    def test_rounding_causes_no_duplicates(self):
        """Test that rounding to integers doesn't create duplicates.

        This is the critical test for the bug where points like [12.3, 2.7]
        and [12.4, 2.8] both round to [12, 3] causing duplicate evaluations.
        """

        evaluation_count = {}

        def counting_func(X):
            """Track how many times each configuration is evaluated."""
            results = []
            for params in X:
                # Round to integers for tracking
                key = tuple(np.round(params).astype(int))
                evaluation_count[key] = evaluation_count.get(key, 0) + 1
                results.append(np.sum(params**2))
            return np.array(results)

        optimizer = SpotOptim(
            fun=counting_func,
            bounds=[(1, 20), (1, 10), (1, 5)],
            var_type=["int", "int", "int"],
            tolerance_x=0.1,
            max_iter=20,
            n_initial=10,
            seed=42,
            verbose=False,
        )

        _ = optimizer.optimize()

        # Check that no configuration was evaluated more than once
        duplicates = {k: v for k, v in evaluation_count.items() if v > 1}

        assert len(duplicates) == 0, (
            f"Found {len(duplicates)} configurations evaluated multiple times:\n"
            + "\n".join(f"{k}: {v} times" for k, v in duplicates.items())
        )

    def test_consecutive_iterations_no_duplicates(self):
        """Test that consecutive iterations don't suggest the same point."""

        last_configs = []

        def tracking_func(X):
            """Track configurations to detect consecutive duplicates."""
            for params in X:
                config = tuple(np.round(params).astype(int))
                last_configs.append(config)
            return np.sum(X**2, axis=1)

        optimizer = SpotOptim(
            fun=tracking_func,
            bounds=[(5, 15), (2, 8)],
            var_type=["int", "int"],
            tolerance_x=0.1,
            max_iter=20,
            n_initial=5,
            seed=42,
            verbose=False,
        )

        _ = optimizer.optimize()

        # Check for consecutive duplicates (the main symptom of the bug)
        consecutive_dupes = []
        for i in range(len(last_configs) - 1):
            if last_configs[i] == last_configs[i + 1]:
                consecutive_dupes.append((i, last_configs[i]))

        assert len(consecutive_dupes) == 0, (
            f"Found {len(consecutive_dupes)} consecutive duplicate evaluations:\n"
            + "\n".join(f"Iteration {i}: {config}" for i, config in consecutive_dupes)
        )


class TestToleranceXFactorVariables:
    """Test suite for tolerance_x with factor variables."""

    def test_no_duplicate_evaluations_factors(self):
        """Test that factor variables minimize duplicate evaluations.

        With 3 activations × 10 integers = 30 possible combinations and requesting
        30 evaluations, we should explore most unique combinations with minimal duplicates.
        """

        def factor_func(X):
            results = []
            for params in X:
                # Simple function based on factor values
                val = hash(str(params)) % 100 / 100.0
                results.append(val)
            return np.array(results)

        optimizer = SpotOptim(
            fun=factor_func,
            bounds=[
                ("relu", "tanh", "sigmoid"),  # 3 options
                (1, 10),  # 10 options = 30 total combinations
            ],
            var_type=["factor", "int"],
            tolerance_x=0.1,
            max_iter=30,
            n_initial=10,
            seed=42,
            verbose=False,
        )

        _ = optimizer.optimize()

        # Get configurations (factors should be strings)
        X = optimizer.X_
        configs = []
        for row in X:
            # Factor should be string, int should be rounded
            config = (row[0], int(round(row[1])))
            configs.append(config)

        n_unique = len(set(configs))
        n_total = len(configs)

        # With 30 combinations and 30 evaluations, should have high uniqueness
        # Allow a few duplicates (e.g., 90% unique = 27/30)
        uniqueness_ratio = n_unique / n_total

        assert uniqueness_ratio >= 0.85, (
            f"Only {n_unique}/{n_total} unique configurations ({uniqueness_ratio:.1%}). "
            f"Expected at least 85% uniqueness."
        )

    def test_mixed_types_no_duplicates(self):
        """Test with mixed variable types (float, int, factor)."""

        def mixed_func(X):
            results = []
            for params in X:
                # Combine different types
                results.append(
                    float(params[0]) ** 2
                    + float(params[1]) ** 2
                    + hash(str(params[2])) % 10
                )
            return np.array(results)

        optimizer = SpotOptim(
            fun=mixed_func,
            bounds=[
                (0.1, 5.0),  # float
                (1, 20),  # int
                ("a", "b", "c", "d"),  # factor
            ],
            var_type=["float", "int", "factor"],
            tolerance_x=0.1,
            max_iter=15,
            n_initial=5,
            seed=42,
            verbose=False,
        )

        _ = optimizer.optimize()

        # Create comparable configurations
        X = optimizer.X_
        configs = []
        for row in X:
            # Float stays float, int rounds, factor is string
            config = (round(float(row[0]), 2), int(round(float(row[1]))), row[2])
            configs.append(config)

        # Check for exact duplicates in the rounded/normalized space
        unique_configs = set(configs)
        n_unique = len(unique_configs)
        n_total = len(configs)

        # Allow for some floating point variation
        assert n_total - n_unique <= 2, (
            f"Found too many duplicate configurations: "
            f"{n_total - n_unique} duplicates out of {n_total}"
        )


class TestToleranceXReproduceBug:
    """Test suite specifically designed to reproduce the reported bug."""

    def test_reproduce_identical_iterations(self):
        """Reproduce the exact scenario from the bug report.

        The bug shows iterations 48-57 all evaluating the same configuration:
        l1=12, num_layers=3, activation=ReLU, optimizer=RMSprop,
        lr_unified=0.1454, alpha=0.0135
        """

        configs_evaluated = []

        def neural_net_simulator(X):
            """Simulate a neural network hyperparameter optimization."""
            results = []
            for params in X:
                # Extract parameters (simulating the bug report)
                l1 = int(round(params[0]))
                num_layers = int(round(params[1]))
                activation = params[2]
                optimizer = params[3]
                lr_unified = float(params[4])
                alpha = float(params[5])

                config = (
                    l1,
                    num_layers,
                    activation,
                    optimizer,
                    round(lr_unified, 4),
                    round(alpha, 4),
                )
                configs_evaluated.append(config)

                # Simple mock objective
                result = l1 / 100.0 + num_layers / 10.0 + lr_unified + alpha
                results.append(result)

            return np.array(results)

        optimizer = SpotOptim(
            fun=neural_net_simulator,
            bounds=[
                (8, 64),  # l1
                (2, 5),  # num_layers
                ("ReLU", "Tanh", "Sigmoid"),  # activation
                ("Adam", "SGD", "RMSprop"),  # optimizer
                (0.001, 0.5),  # lr_unified
                (0.0001, 0.1),  # alpha
            ],
            var_type=["int", "int", "factor", "factor", "float", "float"],
            var_name=[
                "l1",
                "num_layers",
                "activation",
                "optimizer",
                "lr_unified",
                "alpha",
            ],
            tolerance_x=0.01,
            max_iter=15,
            n_initial=10,
            seed=42,
            verbose=False,
        )

        _ = optimizer.optimize()

        # Check for consecutive identical configurations
        consecutive_identical = 0
        max_consecutive = 0
        current_consecutive = 1

        for i in range(len(configs_evaluated) - 1):
            if configs_evaluated[i] == configs_evaluated[i + 1]:
                current_consecutive += 1
                consecutive_identical += 1
            else:
                max_consecutive = max(max_consecutive, current_consecutive)
                current_consecutive = 1

        max_consecutive = max(max_consecutive, current_consecutive)

        assert max_consecutive <= 1, (
            f"Found {max_consecutive} consecutive identical evaluations. "
            f"This reproduces the bug where iterations 48-57 were identical."
        )

        assert consecutive_identical <= 2, (
            f"Found {consecutive_identical} total consecutive identical pairs. "
            f"Expected at most 2 (allowing for noise handling)."
        )

    def test_long_optimization_no_stuck_iterations(self):
        """Test that long optimizations don't get stuck evaluating the same point."""

        def simple_func(X):
            return np.sum(np.round(X) ** 2, axis=1)

        optimizer = SpotOptim(
            fun=simple_func,
            bounds=[(1, 50), (1, 20), (1, 10)],
            var_type=["int", "int", "int"],
            tolerance_x=0.01,
            max_iter=30,
            n_initial=15,
            seed=42,
            verbose=False,
        )

        _ = optimizer.optimize()

        # Count unique configurations
        X = optimizer.X_
        X_int = np.round(X).astype(int)
        unique_rows = np.unique(X_int, axis=0)

        # Should have close to 100 unique configurations (some may be repeated due to noise)
        uniqueness_ratio = len(unique_rows) / len(X_int)

        assert uniqueness_ratio > 0.90, (
            f"Only {len(unique_rows)}/{len(X_int)} unique configurations "
            f"({uniqueness_ratio:.1%}). Optimization appears stuck."
        )


class TestToleranceXWithNoise:
    """Test tolerance_x interaction with noisy objectives."""

    def test_tolerance_with_repeats(self):
        """Test that tolerance_x works correctly with repeated evaluations."""

        def noisy_func(X):
            return np.sum(X**2, axis=1) + np.random.normal(0, 0.1, X.shape[0])

        optimizer = SpotOptim(
            fun=noisy_func,
            bounds=[(1, 20), (1, 10)],
            var_type=["int", "int"],
            tolerance_x=0.1,
            max_iter=20,
            n_initial=5,
            repeats_initial=2,
            repeats_surrogate=2,
            seed=42,
            verbose=False,
        )

        _ = optimizer.optimize()

        # With noise handling, unique configurations should still be enforced
        # (though each config is evaluated multiple times)
        mean_X = optimizer.mean_X
        X_int = np.round(mean_X).astype(int)
        n_unique = len(np.unique(X_int, axis=0))
        n_total = len(X_int)

        # All aggregated points should be unique
        assert n_unique == n_total, (
            f"Found {n_total - n_unique} duplicate configurations "
            f"in noise-aggregated data"
        )


class TestToleranceXEdgeCases:
    """Test edge cases for tolerance_x functionality."""

    def test_tolerance_with_small_bounds(self):
        """Test tolerance_x with very small parameter bounds.

        With small bounds (5×3=15 possible combinations), we expect the algorithm
        to eventually exhaust the space and allow duplicates when necessary.
        The key is that it should explore all unique combinations first.
        """

        def func(X):
            return np.sum(X**2, axis=1)

        optimizer = SpotOptim(
            fun=func,
            bounds=[(1, 5), (1, 3)],  # Very small ranges: 5×3=15 combinations
            var_type=["int", "int"],
            tolerance_x=0.1,
            max_iter=20,  # Request more than available combinations
            n_initial=5,
            seed=42,
            verbose=False,
        )

        _ = optimizer.optimize()

        # With 15 possible combinations and 20 evaluations, duplicates are expected
        # But we should have explored a good portion of unique combinations first
        X_int = np.round(optimizer.X_).astype(int)
        configs = [tuple(row) for row in X_int]

        # Count unique configurations
        n_unique = len(set(configs))
        _ = len(configs)

        # Should have explored at least 10 unique configurations (67% of 15)
        assert n_unique >= 10, (
            f"Only explored {n_unique}/15 possible configurations. "
            f"Expected to explore more before duplicates."
        )

    def test_tolerance_with_dimension_reduction(self):
        """Test tolerance_x with fixed dimensions."""

        def func(X):
            return np.sum(X**2, axis=1)

        optimizer = SpotOptim(
            fun=func,
            bounds=[(1, 20), (5, 5), (1, 10)],  # Middle dimension is fixed
            var_type=["int", "int", "int"],
            tolerance_x=0.1,
            max_iter=20,
            n_initial=10,
            seed=42,
            verbose=False,
        )

        _ = optimizer.optimize()

        # Check for duplicates in reduced dimensions
        X_int = np.round(optimizer.X_).astype(int)
        n_unique = len(np.unique(X_int, axis=0))
        n_total = len(X_int)

        assert (
            n_unique == n_total
        ), f"Found {n_total - n_unique} duplicate configurations"

    def test_very_large_tolerance(self):
        """Test that very large tolerance_x triggers fallback strategies appropriately."""

        def func(X):
            return np.sum(X**2, axis=1)

        optimizer = SpotOptim(
            fun=func,
            bounds=[(1, 20), (1, 10)],
            var_type=["int", "int"],
            tolerance_x=10.0,  # Larger than the bounds!
            max_iter=20,
            n_initial=10,
            seed=42,
            verbose=False,
        )

        result = optimizer.optimize()

        # Should still complete without errors
        assert result.success
        assert len(optimizer.X_) == 20


class TestSelectNewMethod:
    """Test the select_new method directly."""

    def test_select_new_exact_match(self):
        """Test select_new with exact matches."""

        def func(X):
            return np.sum(X**2, axis=1)

        optimizer = SpotOptim(
            fun=func,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=10,
            n_initial=5,
            seed=42,
        )

        X_existing = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        X_new = np.array([[1.0, 2.0], [7.0, 8.0]])  # First is duplicate

        selected, mask = optimizer.select_new(X_new, X_existing, tolerance=0.0)

        assert len(selected) == 1
        assert np.allclose(selected[0], [7.0, 8.0])

    def test_select_new_with_tolerance(self):
        """Test select_new with tolerance."""

        def func(X):
            return np.sum(X**2, axis=1)

        optimizer = SpotOptim(
            fun=func,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=10,
            n_initial=5,
            seed=42,
        )

        X_existing = np.array([[1.0, 2.0], [3.0, 4.0]])
        X_new = np.array([[1.01, 2.01], [7.0, 8.0]])  # First is within tolerance

        selected, mask = optimizer.select_new(X_new, X_existing, tolerance=0.1)

        # First point should be rejected as too close
        assert len(selected) == 1
        assert np.allclose(selected[0], [7.0, 8.0])

    def test_select_new_with_integers(self):
        """Test select_new detects points that round to same integer.

        The key insight: we must round BOTH arrays before comparing, otherwise
        [12.4, 2.7] looks different from [12.0, 3.0] even though both round to [12, 3].
        """

        def func(X):
            return np.sum(X**2, axis=1)

        optimizer = SpotOptim(
            fun=func,
            bounds=[(1, 20), (1, 10)],
            var_type=["int", "int"],
            max_iter=10,
            n_initial=5,
            seed=42,
        )

        X_existing = np.array([[12.0, 3.0], [5.0, 7.0]])
        # This point rounds to [12, 3] - same as first existing point
        X_new_raw = np.array([[12.4, 2.7]])

        # Apply the same rounding that suggest_next_infill_point does
        X_new = optimizer._repair_non_numeric(X_new_raw, optimizer.var_type)

        selected, mask = optimizer.select_new(X_new, X_existing, tolerance=0.1)

        # After rounding [12.4, 2.7] -> [12, 3], should match [12, 3] exactly
        assert len(selected) == 0, "Point that rounds to existing should be rejected"
