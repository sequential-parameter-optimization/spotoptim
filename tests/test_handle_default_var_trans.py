"""Tests for handle_default_var_trans() method in SpotOptim."""

import numpy as np
import pytest
from spotoptim import SpotOptim


class TestHandleDefaultVarTransBasic:
    """Test basic functionality of handle_default_var_trans()."""

    def test_default_none_transformations(self):
        """Test that default var_trans is list of None values."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0, 10), (0, 10)],
            max_iter=5,
            n_initial=3,
        )

        # Should create list of None values matching n_dim
        assert opt.var_trans == [None, None]
        assert len(opt.var_trans) == opt.n_dim

    def test_single_dimension_default(self):
        """Test default var_trans for single dimension."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0, 10)],
            max_iter=5,
            n_initial=3,
        )

        assert opt.var_trans == [None]
        assert len(opt.var_trans) == 1

    def test_multiple_dimensions_default(self):
        """Test default var_trans for multiple dimensions."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0, 10)] * 5,
            max_iter=5,
            n_initial=3,
        )

        assert opt.var_trans == [None] * 5
        assert len(opt.var_trans) == 5


class TestHandleDefaultVarTransNormalization:
    """Test normalization of transformation names."""

    def test_normalize_id_to_none(self):
        """Test that 'id' is normalized to None."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0, 10), (0, 10)],
            var_trans=["id", "id"],
            max_iter=5,
            n_initial=3,
        )

        assert opt.var_trans == [None, None]

    def test_normalize_None_string_to_none(self):
        """Test that string 'None' is normalized to None."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0, 10), (0, 10)],
            var_trans=["None", "None"],
            max_iter=5,
            n_initial=3,
        )

        assert opt.var_trans == [None, None]

    def test_normalize_none_object_unchanged(self):
        """Test that None object remains None."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0, 10), (0, 10)],
            var_trans=[None, None],
            max_iter=5,
            n_initial=3,
        )

        assert opt.var_trans == [None, None]

    def test_normalize_mixed_null_values(self):
        """Test normalization of mixed null-like values."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0, 10)] * 4,
            var_trans=["id", None, "None", "id"],
            max_iter=5,
            n_initial=3,
        )

        assert opt.var_trans == [None, None, None, None]

    def test_preserve_valid_transformations(self):
        """Test that valid transformation names are preserved."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(1, 10), (1, 100), (0, 10)],
            var_trans=["log10", "sqrt", None],
            max_iter=5,
            n_initial=3,
        )

        assert opt.var_trans == ["log10", "sqrt", None]

    def test_normalize_mixed_valid_and_null(self):
        """Test normalization with mix of valid transformations and null values."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(1, 10)] * 5,
            var_trans=["log10", "id", "sqrt", None, "None"],
            max_iter=5,
            n_initial=3,
        )

        assert opt.var_trans == ["log10", None, "sqrt", None, None]


class TestHandleDefaultVarTransValidation:
    """Test validation logic in handle_default_var_trans()."""

    def test_error_on_length_mismatch_too_short(self):
        """Test that shorter var_trans raises ValueError."""
        with pytest.raises(
            ValueError, match="Length of var_trans .* must match.*number of dimensions"
        ):
            SpotOptim(
                fun=lambda X: np.sum(X**2, axis=1),
                bounds=[(0, 10), (0, 10), (0, 10)],
                var_trans=["log10", "sqrt"],  # Too short
                max_iter=5,
                n_initial=3,
            )

    def test_error_on_length_mismatch_too_long(self):
        """Test that longer var_trans raises ValueError."""
        with pytest.raises(
            ValueError, match="Length of var_trans .* must match.*number of dimensions"
        ):
            SpotOptim(
                fun=lambda X: np.sum(X**2, axis=1),
                bounds=[(0, 10), (0, 10)],
                var_trans=["log10", "sqrt", "exp"],  # Too long
                max_iter=5,
                n_initial=3,
            )

    def test_error_message_includes_lengths(self):
        """Test that error message includes actual lengths."""
        with pytest.raises(ValueError) as exc_info:
            SpotOptim(
                fun=lambda X: np.sum(X**2, axis=1),
                bounds=[(0, 10), (0, 10), (0, 10)],
                var_trans=["log10"],
                max_iter=5,
                n_initial=3,
            )

        error_msg = str(exc_info.value)
        assert "1" in error_msg  # var_trans length
        assert "3" in error_msg  # n_dim


class TestHandleDefaultVarTransEdgeCases:
    """Test edge cases for handle_default_var_trans()."""

    def test_empty_list_for_zero_dimensions(self):
        """Test that empty var_trans works for zero dimensions."""
        # This is a hypothetical edge case - SpotOptim likely requires at least 1 dimension
        # but we test the method's behavior
        pass  # Skip as SpotOptim requires at least one bound

    def test_all_transformations_specified(self):
        """Test when all transformations are explicitly specified."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(1, 10)] * 4,
            var_trans=["log10", "sqrt", "log", "exp"],
            max_iter=5,
            n_initial=3,
        )

        assert opt.var_trans == ["log10", "sqrt", "log", "exp"]

    def test_all_null_values_specified(self):
        """Test when all transformations are null values."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0, 10)] * 4,
            var_trans=[None, "id", "None", "id"],
            max_iter=5,
            n_initial=3,
        )

        assert opt.var_trans == [None, None, None, None]


class TestHandleDefaultVarTransIntegration:
    """Integration tests for handle_default_var_trans()."""

    def test_integration_with_bounds_transformation(self):
        """Test that var_trans works correctly with _transform_bounds()."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(1, 10), (0.1, 100)],
            var_trans=["log10", "sqrt"],
            max_iter=5,
            n_initial=3,
        )

        # After transformation, bounds should be modified
        # log10([1, 10]) = [0, 1]
        # sqrt([0.1, 100]) = [0.316..., 10]
        assert opt.var_trans == ["log10", "sqrt"]
        assert opt.bounds[0] == (0.0, 1.0)
        assert opt.bounds[1][0] == pytest.approx(0.31622776601683794)
        assert opt.bounds[1][1] == 10.0

    def test_integration_with_null_transformations(self):
        """Test that null transformations don't modify bounds."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(1, 10), (0.1, 100)],
            var_trans=[None, "id"],
            max_iter=5,
            n_initial=3,
        )

        # Bounds should remain unchanged with null transformations
        assert opt.var_trans == [None, None]
        assert opt.bounds[0] == (1.0, 10.0)
        assert opt.bounds[1] == (0.1, 100.0)

    def test_optimization_runs_with_default_var_trans(self):
        """Test that optimization runs successfully with default var_trans."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0, 10), (0, 10)],
            max_iter=10,
            n_initial=5,
            seed=42,
        )

        result = opt.optimize()
        assert result is not None
        assert hasattr(result, "x")
        assert hasattr(result, "fun")
        assert opt.var_trans == [None, None]

    def test_optimization_runs_with_custom_var_trans(self):
        """Test that optimization runs successfully with custom var_trans."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(1, 10), (1, 100)],
            var_trans=["log10", "sqrt"],
            max_iter=10,
            n_initial=5,
            seed=42,
        )

        result = opt.optimize()
        assert result is not None
        assert hasattr(result, "x")
        assert hasattr(result, "fun")
        assert opt.var_trans == ["log10", "sqrt"]


class TestHandleDefaultVarTransCallability:
    """Test that handle_default_var_trans() can be called directly."""

    def test_direct_method_call(self):
        """Test calling handle_default_var_trans() directly."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0, 10), (0, 10)],
            max_iter=5,
            n_initial=3,
        )

        # Verify initial state
        assert opt.var_trans == [None, None]

        # Manually modify var_trans
        opt.var_trans = ["log10", "id"]

        # Call method directly
        opt.handle_default_var_trans()

        # Should normalize 'id' to None
        assert opt.var_trans == ["log10", None]

    def test_direct_call_with_length_mismatch(self):
        """Test that direct call validates length."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0, 10), (0, 10)],
            max_iter=5,
            n_initial=3,
        )

        # Manually set invalid var_trans
        opt.var_trans = ["log10"]

        # Should raise ValueError when called
        with pytest.raises(
            ValueError, match="Length of var_trans .* must match.*number of dimensions"
        ):
            opt.handle_default_var_trans()


class TestHandleDefaultVarTransWithFactors:
    """Test handle_default_var_trans() with factor variables."""

    def test_default_var_trans_with_factors(self):
        """Test default var_trans with factor variables."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[("red", "green", "blue"), (0, 10)],
            max_iter=5,
            n_initial=3,
        )

        # Factor variables should still get None for var_trans
        assert opt.var_trans == [None, None]

    def test_custom_var_trans_with_factors(self):
        """Test custom var_trans with factor variables."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[("a", "b", "c"), (1, 10)],
            var_trans=[None, "log10"],
            max_iter=5,
            n_initial=3,
        )

        assert opt.var_trans == [None, "log10"]


class TestHandleDefaultVarTransWithVarType:
    """Test interaction between var_trans and var_type."""

    def test_var_trans_independent_of_var_type_int(self):
        """Test that var_trans is independent of var_type='int'."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0, 10), (0, 10)],
            var_type=["int", "int"],
            max_iter=5,
            n_initial=3,
        )

        # var_trans should still default to None
        assert opt.var_trans == [None, None]

    def test_var_trans_independent_of_var_type_float(self):
        """Test that var_trans is independent of var_type='float'."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0, 10), (0, 10)],
            var_type=["float", "float"],
            max_iter=5,
            n_initial=3,
        )

        assert opt.var_trans == [None, None]

    def test_var_trans_with_mixed_var_types(self):
        """Test var_trans with mixed var_types."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(1, 10), (0, 10), ("a", "b")],
            var_type=["float", "int", "factor"],
            var_trans=["log10", None, "id"],
            max_iter=5,
            n_initial=3,
        )

        # var_trans should be normalized
        assert opt.var_trans == ["log10", None, None]


class TestHandleDefaultVarTransReturnValue:
    """Test return value of handle_default_var_trans()."""

    def test_returns_none(self):
        """Test that handle_default_var_trans() returns None."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(0, 10), (0, 10)],
            max_iter=5,
            n_initial=3,
        )

        # Method should return None
        result = opt.handle_default_var_trans()
        assert result is None


class TestHandleDefaultVarTransDocstring:
    """Test examples from the docstring."""

    def test_docstring_example_default(self):
        """Test the default behavior example from docstring."""
        spot = SpotOptim(fun=lambda x: x, bounds=[(0, 10), (0, 10)])
        assert spot.var_trans == [None, None]

    def test_docstring_example_normalize(self):
        """Test the normalization example from docstring (adjusted for 4 dims)."""
        spot = SpotOptim(
            fun=lambda x: x,
            bounds=[(1, 10), (1, 100), (1, 10), (1, 10)],
            var_trans=["log10", "id", None, "None"],
        )
        assert spot.var_trans == ["log10", None, None, None]
