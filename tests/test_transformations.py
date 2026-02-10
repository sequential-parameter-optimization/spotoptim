# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for variable transformations in SpotOptim."""

import numpy as np
import pytest
from spotoptim import SpotOptim


class TestTransformationBasics:
    """Test basic transformation functionality."""

    def test_no_transformation(self):
        """Test that None/id transformations work correctly."""

        def objective(X):
            return np.sum(X**2, axis=1)

        opt = SpotOptim(
            fun=objective,
            bounds=[(0, 10), (-5, 5)],
            var_trans=[None, "id"],
            max_iter=5,
            n_initial=3,
            seed=42,
        )

        result = opt.optimize()

        # Check bounds are preserved
        assert 0 <= result.x[0] <= 10
        assert -5 <= result.x[1] <= 5

    def test_log10_transformation(self):
        """Test log10 transformation."""

        def objective(X):
            # Should receive values in original scale
            assert np.all(X[:, 0] >= 0.001)
            assert np.all(X[:, 0] <= 1.0)
            return np.sum(X**2, axis=1)

        opt = SpotOptim(
            fun=objective,
            bounds=[(0.001, 1.0)],
            var_trans=["log10"],
            max_iter=5,
            n_initial=3,
            seed=42,
        )

        result = opt.optimize()

        # Check result is in original scale
        assert 0.001 <= result.x[0] <= 1.0

        # Check internal bounds are transformed
        assert opt.lower[0] == pytest.approx(np.log10(0.001))
        assert opt.upper[0] == pytest.approx(np.log10(1.0))

        # Check original bounds are preserved
        assert opt._original_lower[0] == 0.001
        assert opt._original_upper[0] == 1.0

    def test_log_transformation(self):
        """Test natural log transformation."""

        def objective(X):
            return np.sum(X**2, axis=1)

        opt = SpotOptim(
            fun=objective,
            bounds=[(0.1, 10.0)],
            var_trans=["log"],
            max_iter=5,
            n_initial=3,
            seed=42,
        )

        result = opt.optimize()

        # Check result is in original scale
        assert 0.1 <= result.x[0] <= 10.0

        # Check internal bounds are transformed
        assert opt.lower[0] == pytest.approx(np.log(0.1))
        assert opt.upper[0] == pytest.approx(np.log(10.0))

    def test_sqrt_transformation(self):
        """Test square root transformation."""

        def objective(X):
            return np.sum(X**2, axis=1)

        opt = SpotOptim(
            fun=objective,
            bounds=[(1.0, 100.0)],
            var_trans=["sqrt"],
            max_iter=5,
            n_initial=3,
            seed=42,
        )

        result = opt.optimize()

        # Check result is in original scale
        assert 1 <= result.x[0] <= 100

        # Check internal bounds are transformed
        assert opt.lower[0] == pytest.approx(np.sqrt(1))
        assert opt.upper[0] == pytest.approx(np.sqrt(100))

    def test_exp_transformation(self):
        """Test exponential transformation."""

        def objective(X):
            return np.sum(X**2, axis=1)

        opt = SpotOptim(
            fun=objective,
            bounds=[(0.1, 2.0)],
            var_trans=["exp"],
            max_iter=5,
            n_initial=3,
            seed=42,
        )

        result = opt.optimize()

        # Check result is in original scale
        assert 0.1 <= result.x[0] <= 2.0

        # Check internal bounds are transformed
        assert opt.lower[0] == pytest.approx(np.exp(0.1))
        assert opt.upper[0] == pytest.approx(np.exp(2.0))

    def test_square_transformation(self):
        """Test square transformation."""

        def objective(X):
            return np.sum(X**2, axis=1)

        opt = SpotOptim(
            fun=objective,
            bounds=[(1.0, 10.0)],
            var_trans=["square"],
            max_iter=5,
            n_initial=3,
            seed=42,
        )

        result = opt.optimize()

        # Check result is in original scale
        assert 1 <= result.x[0] <= 10

    def test_cube_transformation(self):
        """Test cube transformation."""

        def objective(X):
            return np.sum(X**2, axis=1)

        opt = SpotOptim(
            fun=objective,
            bounds=[(1.0, 8.0)],
            var_trans=["cube"],
            max_iter=5,
            n_initial=3,
            seed=42,
        )

        result = opt.optimize()

        # Check result is in original scale
        assert 1 <= result.x[0] <= 8

    def test_reciprocal_transformation(self):
        """Test reciprocal transformation."""

        def objective(X):
            return np.sum(X**2, axis=1)

        opt = SpotOptim(
            fun=objective,
            bounds=[(0.1, 10.0)],
            var_trans=["inv"],
            max_iter=10,
            n_initial=3,
            seed=42,
        )

        result = opt.optimize()

        # Check result is in original scale
        assert 0.1 <= result.x[0] <= 10.0

        # Check internal bounds are transformed correctly
        # For reciprocal, bounds swap: min(1/0.1, 1/10.0) to max(1/0.1, 1/10.0)
        assert opt.lower[0] == pytest.approx(min(1.0 / 10.0, 1.0 / 0.1))
        assert opt.upper[0] == pytest.approx(max(1.0 / 10.0, 1.0 / 0.1))


class TestMultipleTransformations:
    """Test multiple transformations in one optimization."""

    def test_mixed_transformations(self):
        """Test mixing different transformation types."""
        call_count = [0]

        def objective(X):
            call_count[0] += X.shape[0]
            # All should be in original scale
            for params in X:
                assert 0.001 <= params[0] <= 1.0, f"x0={params[0]} out of bounds"
                assert 10 <= params[1] <= 1000, f"x1={params[1]} out of bounds"
                assert -5 <= params[2] <= 5, f"x2={params[2]} out of bounds"
                assert 0.1 <= params[3] <= 10, f"x3={params[3]} out of bounds"
            return np.sum(X**2, axis=1)

        opt = SpotOptim(
            fun=objective,
            bounds=[
                (0.001, 1.0),  # log10
                (10, 1000),  # sqrt
                (-5, 5),  # none
                (0.1, 10),  # inv
            ],
            var_trans=["log10", "sqrt", None, "inv"],
            var_name=["lr", "neurons", "bias", "temp"],
            max_iter=10,
            n_initial=5,
            seed=42,
        )

        result = opt.optimize()

        # Check all results are in original scale
        assert 0.001 <= result.x[0] <= 1.0
        assert 10 <= result.x[1] <= 1000
        assert -5 <= result.x[2] <= 5
        assert 0.1 <= result.x[3] <= 10

        # Check that objective was called
        assert call_count[0] > 0

    def test_transformation_with_var_types(self):
        """Test transformations with different variable types."""

        def objective(X):
            results = []
            for params in X:
                # Check integer is rounded
                assert params[0] == int(params[0])
                # Check bounds (allow small tolerance for rounding)
                assert 9 <= params[0] <= 100, f"x0={params[0]} out of bounds"
                assert 0.0009 <= params[1] <= 1.001, f"x1={params[1]} out of bounds"
                results.append(np.sum(params**2))
            return np.array(results)

        opt = SpotOptim(
            fun=objective,
            bounds=[(10.0, 100.0), (0.001, 1.0)],
            var_trans=["sqrt", "log10"],
            var_type=["int", "float"],
            max_iter=10,
            n_initial=5,
            seed=42,
        )

        result = opt.optimize()

        # Check integer constraint
        assert result.x[0] == int(result.x[0])
        # Check bounds (allow small tolerance for sqrt+int combination rounding)
        assert 9 <= result.x[0] <= 100, f"x0={result.x[0]} out of bounds"
        assert 0.0009 <= result.x[1] <= 1.001, f"x1={result.x[1]} out of bounds"


class TestTransformationValidation:
    """Test transformation validation and error handling."""

    def test_invalid_transformation_name(self):
        """Test that invalid transformation names raise errors."""
        with pytest.raises(ValueError, match="Unknown transformation"):
            opt = SpotOptim(
                fun=lambda X: np.sum(X**2, axis=1),
                bounds=[(0.0, 10.0)],
                var_trans=["invalid_transform"],
                max_iter=10,
                n_initial=3,
            )
            # Try to transform a value
            opt._transform_value(5, "invalid_transform")

    def test_var_trans_length_mismatch(self):
        """Test that var_trans length must match bounds."""
        with pytest.raises(ValueError, match="Length of var_trans"):
            SpotOptim(
                fun=lambda X: np.sum(X**2, axis=1),
                bounds=[(0, 10), (0, 5)],
                var_trans=["log10"],  # Only 1, but 2 dimensions
                max_iter=10,
                n_initial=3,
            )

    def test_none_transformation_in_list(self):
        """Test that None in var_trans list is handled correctly."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0.0, 10.0), (0.01, 5.0)],  # Avoid log10(0)
            var_trans=[None, "log10"],
            max_iter=10,
            n_initial=3,
            seed=42,
        )

        # First dimension should not be transformed
        assert opt.lower[0] == 0
        assert opt.upper[0] == 10

        # Second dimension should be transformed
        assert opt.lower[1] == pytest.approx(np.log10(0.01))

    def test_default_var_trans_none(self):
        """Test that default var_trans is None for all dimensions."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0, 10), (0, 5)],
            max_iter=10,
            n_initial=3,
        )

        assert len(opt.var_trans) == 2
        assert all(t is None for t in opt.var_trans)


class TestTransformationTables:
    """Test that transformations appear correctly in tables."""

    def test_design_table_shows_transformations(self):
        """Test that print_design_table shows transformation column."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0.001, 1.0), (10.0, 100.0), (-5.0, 5.0)],
            var_trans=["log10", "sqrt", None],
            var_name=["lr", "neurons", "bias"],
            max_iter=10,
            n_initial=3,
        )

        table = opt.print_design_table()

        # Check that table contains trans column
        assert "trans" in table
        assert "log10" in table
        assert "sqrt" in table
        assert "-" in table  # None should show as "-"

    def test_results_table_shows_transformations(self):
        """Test that print_results_table shows transformation column."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0.001, 1.0), (10.0, 100.0)],
            var_trans=["log10", "sqrt"],
            var_name=["lr", "neurons"],
            max_iter=5,
            n_initial=3,
            seed=42,
        )

        result = opt.optimize()
        table = opt.print_results_table()

        # Check that table contains trans column
        assert "trans" in table
        assert "log10" in table
        assert "sqrt" in table


class TestTransformationOptimization:
    """Test that transformations improve optimization."""

    def test_log_scale_optimization(self):
        """Test optimization on log-scale parameter."""

        def objective(X):
            # Optimal at lr=0.01
            results = []
            for params in X:
                lr = params[0]
                error = (lr - 0.01) ** 2
                results.append(error)
            return np.array(results)

        # With transformation
        opt_trans = SpotOptim(
            fun=objective,
            bounds=[(0.001, 1.0)],
            var_trans=["log10"],
            max_iter=20,
            n_initial=5,
            seed=42,
        )
        result_trans = opt_trans.optimize()

        # Without transformation
        opt_no_trans = SpotOptim(
            fun=objective,
            bounds=[(0.001, 1.0)],
            var_trans=[None],
            max_iter=20,
            n_initial=5,
            seed=43,  # Different seed for fair comparison
        )
        result_no_trans = opt_no_trans.optimize()

        # Both should find reasonable solutions (just check they work)
        assert result_trans.fun < 0.01  # Should find something close to optimum
        assert result_no_trans.fun < 0.1  # Less strict for no transformation

    def test_transformation_preserves_optimum(self):
        """Test that transformation doesn't change the location of optimum."""

        def objective(X):
            # Optimal at x=1.0
            return np.array([(x[0] - 1.0) ** 2 for x in X])

        opt = SpotOptim(
            fun=objective,
            bounds=[(0.1, 10.0)],
            var_trans=["log10"],
            max_iter=30,
            n_initial=10,
            seed=42,
        )

        result = opt.optimize()

        # Should find optimum near 1.0
        assert result.x[0] == pytest.approx(1.0, abs=0.5)


class TestTransformationStorage:
    """Test that transformed values are stored correctly."""

    def test_X_stored_in_original_scale(self):
        """Test that X_ is stored in original scale."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0.001, 1.0)],
            var_trans=["log10"],
            max_iter=10,
            n_initial=5,
            seed=42,
        )

        result = opt.optimize()

        # Check that all X_ values are in original scale
        assert np.all(opt.X_ >= 0.001)
        assert np.all(opt.X_ <= 1.0)

    def test_best_x_in_original_scale(self):
        """Test that best_x_ is in original scale."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0.001, 1.0), (10.0, 1000.0)],
            var_trans=["log10", "sqrt"],
            max_iter=10,
            n_initial=5,
            seed=42,
        )

        result = opt.optimize()

        # Check best_x_ is in original scale
        assert 0.001 <= opt.best_x_[0] <= 1.0
        assert 10 <= opt.best_x_[1] <= 1000

        # Check result.x is also in original scale
        assert 0.001 <= result.x[0] <= 1.0
        assert 10 <= result.x[1] <= 1000


class TestTransformationEdgeCases:
    """Test edge cases and special scenarios."""

    def test_transformation_with_user_provided_X0(self):
        """Test that user-provided X0 is handled correctly."""
        X0 = np.array([[0.01, 50], [0.1, 100], [0.5, 500]])

        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0.001, 1.0), (10.0, 1000.0)],
            var_trans=["log10", "sqrt"],
            max_iter=10,
            seed=42,
        )

        result = opt.optimize(X0=X0)

        # Should work without errors
        assert result.x is not None

    def test_ln_alias_for_log(self):
        """Test that 'ln' is an alias for 'log'."""
        opt1 = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0.1, 10.0)],
            var_trans=["ln"],
            max_iter=10,
            n_initial=3,
            seed=42,
        )

        opt2 = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0.1, 10.0)],
            var_trans=["log"],
            max_iter=10,
            n_initial=3,
            seed=42,
        )

        # Both should produce same transformed bounds
        assert opt1.lower[0] == pytest.approx(opt2.lower[0])
        assert opt1.upper[0] == pytest.approx(opt2.upper[0])

    def test_reciprocal_alias(self):
        """Test that 'reciprocal' is an alias for 'inv'."""
        opt1 = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0.1, 10.0)],
            var_trans=["inv"],
            max_iter=10,
            n_initial=3,
            seed=42,
        )

        opt2 = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0.1, 10.0)],
            var_trans=["reciprocal"],
            max_iter=10,
            n_initial=3,
            seed=42,
        )

        # Both should produce same transformed bounds
        assert opt1.lower[0] == pytest.approx(opt2.lower[0])
        assert opt1.upper[0] == pytest.approx(opt2.upper[0])

    def test_id_alias_for_none(self):
        """Test that 'id' is an alias for None."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0, 10)],
            var_trans=["id"],
            max_iter=10,
            n_initial=3,
            seed=42,
        )

        # Should normalize to None
        assert opt.var_trans[0] is None


class TestTransformationIntegration:
    """Integration tests for transformations with other features."""

    def test_transformation_with_factor_variables(self):
        """Test transformations work with factor variables."""

        def objective(X):
            results = []
            for params in X:
                activation = params[0]  # Factor variable (string)
                lr = params[1]  # Numeric with transformation

                # Simple scoring
                score = 1.0 if activation == "ReLU" else 2.0
                score += (lr - 0.01) ** 2
                results.append(score)
            return np.array(results)

        opt = SpotOptim(
            fun=objective,
            bounds=[("ReLU", "Tanh", "Sigmoid"), (0.001, 0.1)],
            var_type=["factor", "float"],
            var_trans=[None, "log10"],
            max_iter=15,
            n_initial=5,
            seed=42,
        )

        result = opt.optimize()

        # Check result
        assert isinstance(result.x[0], str)
        assert 0.001 <= result.x[1] <= 0.1

    def test_transformation_with_dimension_reduction(self):
        """Test transformations work with dimension reduction."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[
                (0.001, 1.0),  # log10
                (5, 5),  # fixed (will be reduced)
                (10, 100),  # sqrt
            ],
            var_trans=["log10", None, "sqrt"],
            max_iter=10,
            n_initial=5,
            seed=42,
        )

        result = opt.optimize()

        # Should handle dimension reduction correctly
        assert len(result.x) == 3  # Full dimensions in result
        assert 0.001 <= result.x[0] <= 1.0
        assert result.x[1] == 5.0  # Fixed value
        assert 10 <= result.x[2] <= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
