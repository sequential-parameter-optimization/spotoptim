"""Tests for factor variable handling in SpotOptim."""

import numpy as np
import pytest
from spotoptim import SpotOptim


class TestFactorVariables:
    """Test factor variable support in SpotOptim."""

    def test_factor_bounds_processing(self):
        """Test that factor bounds are correctly processed."""
        # Create optimizer with factor variable
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[
                (-5, 5),  # numeric
                ("low", "medium", "high"),  # factor
                (0, 10),  # numeric
            ],
            max_iter=5,
            n_initial=3,
        )

        # Check that factor mapping was created
        assert 1 in opt._factor_maps
        assert opt._factor_maps[1] == {0: "low", 1: "medium", 2: "high"}

        # Check that bounds were converted to integers
        assert opt.bounds[1] == (0, 2)
        assert opt.bounds[0] == (-5, 5)
        assert opt.bounds[2] == (0, 10)

    def test_factor_var_type_auto_detection(self):
        """Test that var_type is automatically set for factor variables."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[
                (-5, 5),  # numeric -> float
                ("a", "b", "c"),  # factor
                (0, 10),  # numeric -> float
            ],
            max_iter=5,
            n_initial=3,
        )

        # var_type should be auto-detected
        assert opt.var_type[0] == "float"
        assert opt.var_type[1] == "factor"
        assert opt.var_type[2] == "float"

    def test_factor_mapping_to_strings(self):
        """Test that internal integers are correctly mapped to strings."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[("alpha", "beta", "gamma")],
            max_iter=5,
            n_initial=3,
        )

        # Test mapping
        X_internal = np.array([[0], [1], [2]])
        X_mapped = opt._map_to_factor_values(X_internal)

        assert X_mapped[0, 0] == "alpha"
        assert X_mapped[1, 0] == "beta"
        assert X_mapped[2, 0] == "gamma"

    def test_factor_rounding_and_clipping(self):
        """Test that fractional values are rounded and clipped."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[("a", "b", "c")],
            max_iter=5,
            n_initial=3,
        )

        # Test with fractional values
        X_internal = np.array([[0.4], [1.6], [2.9], [-0.5], [3.5]])
        X_mapped = opt._map_to_factor_values(X_internal)

        assert X_mapped[0, 0] == "a"  # 0.4 -> 0
        assert X_mapped[1, 0] == "c"  # 1.6 -> 2 -> "c" (index 2)
        assert X_mapped[2, 0] == "c"  # 2.9 -> 3 -> clips to 2 -> "c"
        assert X_mapped[3, 0] == "a"  # -0.5 -> clips to 0 -> "a"
        assert X_mapped[4, 0] == "c"  # 3.5 -> clips to 2 -> "c"

    def test_objective_function_receives_strings(self):
        """Test that objective function receives factor values as strings."""
        received_values = []

        def objective(X):
            # Store received values
            received_values.append(X.copy())
            # For factor variables, we expect strings
            # Simple scoring: "small" = 1, "medium" = 2, "large" = 3
            scores = []
            for row in X:
                if row[0] == "small":
                    scores.append(1.0)
                elif row[0] == "medium":
                    scores.append(2.0)
                elif row[0] == "large":
                    scores.append(3.0)
                else:
                    scores.append(999.0)  # Should not happen
            return np.array(scores)

        opt = SpotOptim(
            fun=objective,
            bounds=[("small", "medium", "large")],
            max_iter=5,
            n_initial=3,
            seed=42,
        )

        result = opt.optimize()

        # Check that objective function received strings
        for X_batch in received_values:
            for row in X_batch:
                assert isinstance(row[0], str)
                assert row[0] in ["small", "medium", "large"]

        # Best result should be "small" (value 1.0)
        assert result.x[0] == "small"
        assert result.fun == 1.0

    def test_mixed_factor_and_numeric(self):
        """Test optimization with mixed factor and numeric variables."""

        def objective(X):
            results = []
            for row in X:
                # row[0] is numeric, row[1] is factor
                numeric_val = row[0]
                factor_val = row[1]

                # Compute based on factor
                if factor_val == "ReLU":
                    base = numeric_val**2
                elif factor_val == "Sigmoid":
                    base = (numeric_val - 2) ** 2
                elif factor_val == "Tanh":
                    base = (numeric_val + 2) ** 2
                else:
                    base = 999.0

                results.append(base)
            return np.array(results)

        opt = SpotOptim(
            fun=objective,
            bounds=[
                (-5, 5),  # numeric
                ("ReLU", "Sigmoid", "Tanh"),  # factor
            ],
            max_iter=20,
            n_initial=10,
            seed=42,
        )

        result = opt.optimize()

        # Best should be near x[0]=0 with factor="ReLU" (or x[0]=2 with "Sigmoid", etc.)
        assert isinstance(result.x[0], (int, float, np.integer, np.floating))
        assert isinstance(result.x[1], str)
        assert result.x[1] in ["ReLU", "Sigmoid", "Tanh"]

    def test_multiple_factor_variables(self):
        """Test optimization with multiple factor variables."""

        def objective(X):
            results = []
            for row in X:
                # Two factor variables
                act = row[0]
                opt_name = row[1]

                score = 0.0
                # Prefer certain combinations
                if act == "ReLU" and opt_name == "Adam":
                    score = 1.0
                elif act == "Sigmoid" and opt_name == "SGD":
                    score = 2.0
                else:
                    score = 10.0

                results.append(score)
            return np.array(results)

        opt = SpotOptim(
            fun=objective,
            bounds=[
                ("ReLU", "Sigmoid", "Tanh"),
                ("Adam", "SGD", "RMSprop"),
            ],
            max_iter=15,
            n_initial=9,
            seed=42,
        )

        result = opt.optimize()

        # Should find optimal combination
        assert result.x[0] in ["ReLU", "Sigmoid", "Tanh"]
        assert result.x[1] in ["Adam", "SGD", "RMSprop"]
        assert result.fun <= 2.0  # Should find one of the good combinations

    def test_factor_with_explicit_var_type(self):
        """Test that explicit var_type for factors works."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[
                (-5, 5),
                ("a", "b", "c"),
                (0, 10),
            ],
            var_type=["float", "factor", "int"],
            max_iter=5,
            n_initial=3,
        )

        assert opt.var_type == ["float", "factor", "int"]
        assert 1 in opt._factor_maps

    def test_result_X_contains_factor_strings(self):
        """Test that result.X contains factor strings, not integers."""

        def objective(X):
            results = []
            for row in X:
                if row[0] == "low":
                    results.append(1.0)
                elif row[0] == "high":
                    results.append(3.0)
                else:
                    results.append(2.0)
            return np.array(results)

        opt = SpotOptim(
            fun=objective,
            bounds=[("low", "medium", "high")],
            max_iter=10,
            n_initial=5,
            seed=42,
        )

        result = opt.optimize()

        # Check that all X values are strings
        for x in result.X:
            assert isinstance(x[0], str)
            assert x[0] in ["low", "medium", "high"]

    def test_empty_factor_levels_error(self):
        """Test that empty factor levels raise an error."""
        with pytest.raises(ValueError):
            SpotOptim(
                fun=lambda X: np.sum(X**2, axis=1),
                bounds=[()],  # Empty tuple
                max_iter=5,
                n_initial=3,
            )

    def test_single_factor_level(self):
        """Test that single-level factor (fixed) is handled via dimension reduction."""
        # Single-level factor behaves like a fixed dimension (lower==upper)
        # This triggers dimension reduction, resulting in 0-dimensional optimization
        opt = SpotOptim(
            fun=lambda X: np.array([1.0] * len(X)),
            bounds=[("only_option",)],
            max_iter=5,
            n_initial=3,
        )

        # Check that dimension reduction kicked in
        # n_dim becomes 0 after reduction since single-level = fixed dimension
        assert opt.red_dim == True  # Using == for numpy bool
        assert opt.n_dim == 0  # Reduced to 0 dimensions (was 1 originally)
        assert len(opt.lower) == 0  # Reduced bounds

        # The only level should be mapped
        assert 0 in opt._factor_maps
        assert opt._factor_maps[0][0] == "only_option"

    def test_many_factor_levels(self):
        """Test with many factor levels."""
        levels = [f"level_{i}" for i in range(20)]

        def objective(X):
            results = []
            for row in X:
                # Extract level number
                level_num = int(row[0].split("_")[1])
                # Quadratic with minimum at level_10
                results.append((level_num - 10) ** 2)
            return np.array(results)

        opt = SpotOptim(
            fun=objective,
            bounds=[tuple(levels)],
            max_iter=30,
            n_initial=10,
            seed=42,
        )

        result = opt.optimize()

        # Should find level near 10
        assert result.x[0].startswith("level_")
        best_level = int(result.x[0].split("_")[1])
        assert abs(best_level - 10) <= 2  # Allow some tolerance

    def test_neural_network_hyperparameter_example(self):
        """Test realistic example: neural network hyperparameter optimization."""

        def train_model(X):
            """Simulate training a model with given hyperparameters."""
            results = []
            for row in X:
                lr = 10 ** row[0]  # log scale
                l1 = int(row[1])
                num_layers = int(row[2])
                activation = row[3]  # Factor variable

                # Simple scoring based on known good configurations
                score = 100.0

                # Prefer ReLU
                if activation == "ReLU":
                    score -= 20.0
                elif activation == "Tanh":
                    score -= 10.0

                # Prefer moderate learning rate
                score += abs(np.log10(lr) + 3) * 5  # Best around 0.001

                # Prefer moderate network size
                score += abs(l1 - 64) * 0.1
                score += abs(num_layers - 2) * 5

                results.append(score)
            return np.array(results)

        opt = SpotOptim(
            fun=train_model,
            bounds=[
                (-4, -2),  # log10(lr): 0.0001 to 0.01
                (16, 128),  # l1: number of neurons
                (0, 4),  # num_hidden_layers
                ("ReLU", "Sigmoid", "Tanh", "LeakyReLU"),  # activation
            ],
            var_type=["float", "int", "int", "factor"],
            max_iter=30,
            n_initial=15,
            seed=42,
            verbose=False,
        )

        result = opt.optimize()

        # Check result structure
        assert len(result.x) == 4
        assert isinstance(result.x[0], (float, np.floating, np.number))
        # Note: values come from object array, may be float
        assert isinstance(result.x[1], (int, float, np.integer, np.floating, np.number))
        assert isinstance(result.x[2], (int, float, np.integer, np.floating, np.number))
        assert isinstance(result.x[3], str)
        assert result.x[3] in ["ReLU", "Sigmoid", "Tanh", "LeakyReLU"]

        # Should find reasonable solution
        lr_optimal = 10 ** result.x[0]
        assert 0.0001 <= lr_optimal <= 0.01
        assert 16 <= result.x[1] <= 128
        assert 0 <= result.x[2] <= 4

        # Likely to find ReLU since it's best
        print(
            f"Best configuration: lr={lr_optimal:.6f}, l1={result.x[1]}, "
            f"layers={result.x[2]}, activation={result.x[3]}"
        )
        print(f"Best score: {result.fun:.4f}")
