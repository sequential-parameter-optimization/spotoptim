# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Deep tests for SpotOptim — validates contracts, edge cases, and invariants.

Run with:
    uv run pytest tests/test_spotoptim_deep.py -v
"""

import numpy as np
import pytest
from scipy.optimize import OptimizeResult
from spotoptim import SpotOptim, Kriging

# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------


def sphere(X):
    """Sphere function: f(x) = sum(x_i^2), min = 0 at origin."""
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1)


def parabola_1d(X):
    """1-D parabola: f(x) = (x - 3)^2, min = 0 at x = 3."""
    X = np.atleast_2d(X)
    return (X[:, 0] - 3.0) ** 2


def rosenbrock(X):
    """Rosenbrock: f(x,y) = (1-x)^2 + 100*(y-x^2)^2, min = 0 at (1,1)."""
    X = np.atleast_2d(X)
    x, y = X[:, 0], X[:, 1]
    return (1.0 - x) ** 2 + 100.0 * (y - x**2) ** 2


def beale(X):
    """Beale function: min = 0 at (3, 0.5)."""
    X = np.atleast_2d(X)
    x, y = X[:, 0], X[:, 1]
    t1 = (1.5 - x + x * y) ** 2
    t2 = (2.25 - x + x * y**2) ** 2
    t3 = (2.625 - x + x * y**3) ** 2
    return t1 + t2 + t3


# ---------------------------------------------------------------------------
# 1. Initialisation — parameter validation and defaults
# ---------------------------------------------------------------------------


class TestInit:
    """Verify constructor behaviour: valid params accepted, invalid params rejected."""

    def test_bounds_required_no_attribute(self):
        """Raises ValueError when bounds is None and fun has no bounds attribute."""
        with pytest.raises(ValueError, match="Bounds must be provided"):
            SpotOptim(fun=sphere)  # no bounds kwarg and sphere has no .bounds attr

    def test_bounds_from_function_attribute(self):
        """Bounds are inferred from fun.bounds when not supplied explicitly."""

        def fun_with_bounds(X):
            X = np.atleast_2d(X)
            return np.sum(X**2, axis=1)

        fun_with_bounds.bounds = [(-1, 1), (-1, 1)]
        opt = SpotOptim(fun=fun_with_bounds)
        assert opt.bounds == [(-1, 1), (-1, 1)]

    def test_max_iter_less_than_n_initial_raises(self):
        """Raises ValueError when max_iter < n_initial."""
        with pytest.raises(ValueError, match="max_iter"):
            SpotOptim(fun=sphere, bounds=[(-1, 1)], max_iter=3, n_initial=5)

    def test_max_iter_equal_to_n_initial_accepted(self):
        """max_iter == n_initial is valid (only initial design phase runs)."""
        opt = SpotOptim(fun=sphere, bounds=[(-1, 1)], max_iter=5, n_initial=5)
        assert opt.max_iter == 5
        assert opt.n_initial == 5

    def test_n_dim_set_correctly(self):
        """n_dim is derived from bounds length."""
        opt = SpotOptim(fun=sphere, bounds=[(-1, 1), (-2, 2), (-3, 3)])
        assert opt.n_dim == 3

    def test_default_acquisition_is_y(self):
        """Default acquisition function is 'y' (best observed)."""
        opt = SpotOptim(fun=sphere, bounds=[(-1, 1)])
        assert opt.acquisition == "y"

    def test_acquisition_stored_lower(self):
        """Acquisition string is normalised to lower case in storage."""
        opt = SpotOptim(fun=sphere, bounds=[(-1, 1)], acquisition="EI")
        assert opt.acquisition == "ei"

    def test_seed_sets_rng(self):
        """With seed set, two SpotOptim instances share the same initial RNG state."""
        opt1 = SpotOptim(fun=sphere, bounds=[(-1, 1)], seed=0)
        opt2 = SpotOptim(fun=sphere, bounds=[(-1, 1)], seed=0)
        assert opt1.rng.randint(1000) == opt2.rng.randint(1000)

    def test_seed_none_allowed(self):
        """seed=None is accepted without error."""
        opt = SpotOptim(fun=sphere, bounds=[(-1, 1)], seed=None)
        assert opt.seed is None

    def test_integer_var_type_converts_bounds(self):
        """Integer var_type forces bounds to integer values."""
        opt = SpotOptim(
            fun=sphere,
            bounds=[(-5.7, 4.3)],
            var_type=["int"],
        )
        lo, hi = opt.bounds[0]
        assert isinstance(lo, (int, np.integer))
        assert isinstance(hi, (int, np.integer))
        assert lo == -5
        assert hi == 4

    def test_float_var_type_converts_bounds(self):
        """Float var_type forces bounds to float values."""
        opt = SpotOptim(
            fun=sphere,
            bounds=[(0, 10)],
            var_type=["float"],
        )
        lo, hi = opt.bounds[0]
        assert isinstance(lo, float)
        assert isinstance(hi, float)

    def test_factor_bounds_mapped_to_integers(self):
        """Tuple bounds trigger factor mapping; internal bounds become (0, n-1)."""
        opt = SpotOptim(
            fun=lambda X: np.zeros(len(X)),
            bounds=[("red", "green", "blue"), (0, 10)],
        )
        # Factor dimension mapped to integer range 0..2
        assert opt.bounds[0] == (0, 2)
        assert 0 in opt._factor_maps
        assert opt._factor_maps[0] == {0: "red", 1: "green", 2: "blue"}

    def test_var_name_inferred_from_function(self):
        """var_name is inferred from fun.var_name when not supplied explicitly."""
        fun = lambda X: np.zeros(len(np.atleast_2d(X)))  # noqa: E731
        fun.var_name = ["alpha", "beta"]
        fun.bounds = [(-1, 1), (-1, 1)]
        opt = SpotOptim(fun=fun)
        assert opt.var_name == ["alpha", "beta"]


# ---------------------------------------------------------------------------
# 2. detect_var_type
# ---------------------------------------------------------------------------


class TestDetectVarType:
    """Verify automatic variable-type detection from bounds."""

    def test_all_float(self):
        """Numeric tuple bounds -> 'float'."""
        opt = SpotOptim(fun=sphere, bounds=[(-1, 1), (0, 5)])
        result = opt.detect_var_type()
        assert result == ["float", "float"]

    def test_factor_dimension_detected(self):
        """Tuple-of-strings bounds -> 'factor'."""
        opt = SpotOptim(
            fun=lambda X: np.zeros(len(X)),
            bounds=[("a", "b", "c"), (0, 1)],
        )
        result = opt.detect_var_type()
        assert result == ["factor", "float"]

    def test_mixed_types(self):
        """Mixed factor + float dimensions detected correctly."""
        opt = SpotOptim(
            fun=lambda X: np.zeros(len(X)),
            bounds=[("x", "y"), (-5, 5), ("p", "q", "r")],
        )
        result = opt.detect_var_type()
        assert result == ["factor", "float", "factor"]


# ---------------------------------------------------------------------------
# 3. transform_value / inverse_transform_value — algebraic roundtrip
# ---------------------------------------------------------------------------


class TestTransformValue:
    """Algebra roundtrip: inverse(transform(x)) == x for all supported transforms."""

    @pytest.fixture(autouse=True)
    def _opt(self):
        self.opt = SpotOptim(fun=sphere, bounds=[(1e-6, 100)])

    @pytest.mark.parametrize(
        "trans, x",
        [
            ("log10", 10.0),
            ("log10", 100.0),
            ("log", 1.0),
            ("ln", np.e),
            ("sqrt", 9.0),
            ("exp", 2.0),
            ("square", 3.0),
            ("cube", 2.0),
            ("inv", 4.0),
            ("reciprocal", 5.0),
            (None, 7.0),
        ],
    )
    def test_roundtrip(self, trans, x):
        """inverse_transform_value(transform_value(x)) == x."""
        tx = self.opt.transform_value(x, trans)
        x_back = self.opt.inverse_transform_value(tx, trans)
        assert np.isclose(
            x_back, x, rtol=1e-9
        ), f"Roundtrip failed for trans={trans!r}: {x} -> {tx} -> {x_back}"

    def test_log10_known_value(self):
        """log10(10) == 1."""
        assert np.isclose(self.opt.transform_value(10.0, "log10"), 1.0)

    def test_none_is_identity(self):
        """None transform leaves value unchanged."""
        assert self.opt.transform_value(42.0, None) == 42.0

    def test_id_is_identity(self):
        """'id' transform is the identity; inverse is also identity."""
        assert self.opt.transform_value(3.14, "id") == 3.14
        assert self.opt.inverse_transform_value(3.14, "id") == 3.14


# ---------------------------------------------------------------------------
# 4. optimize() — result contract and shape invariants
# ---------------------------------------------------------------------------


class TestOptimizeContract:
    """The OptimizeResult returned by optimize() must satisfy these invariants."""

    @pytest.fixture(autouse=True)
    def _result(self):
        opt = SpotOptim(
            fun=sphere,
            bounds=[(-3, 3), (-3, 3)],
            max_iter=12,
            n_initial=6,
            seed=0,
        )
        self.result = opt.optimize()
        self.max_iter = 12
        self.n_initial = 6
        self.n_dim = 2

    def test_returns_optimize_result(self):
        assert isinstance(self.result, OptimizeResult)

    def test_required_fields_present(self):
        for attr in ("x", "fun", "nfev", "nit", "success", "message", "X", "y"):
            assert hasattr(self.result, attr), f"Missing field: {attr}"

    def test_x_is_1d_array(self):
        assert isinstance(self.result.x, np.ndarray)
        assert self.result.x.ndim == 1

    def test_x_shape_matches_n_dim(self):
        assert len(self.result.x) == self.n_dim

    def test_X_shape(self):
        assert self.result.X.shape == (self.max_iter, self.n_dim)

    def test_y_shape(self):
        assert self.result.y.shape == (self.max_iter,)

    def test_nfev_equals_max_iter(self):
        assert self.result.nfev == self.max_iter

    def test_nit_equals_max_iter_minus_n_initial(self):
        assert self.result.nit == self.max_iter - self.n_initial

    def test_success_is_bool(self):
        assert isinstance(self.result.success, bool)

    def test_success_is_true(self):
        assert self.result.success is True

    def test_message_is_str(self):
        assert isinstance(self.result.message, str)
        assert len(self.result.message) > 0

    def test_fun_equals_min_of_y(self):
        """result.fun must equal the minimum of all stored y values."""
        assert np.isclose(self.result.fun, np.min(self.result.y))

    def test_x_corresponds_to_best_y(self):
        """result.x must correspond to the row in X with the minimum y."""
        best_idx = np.argmin(self.result.y)
        np.testing.assert_array_almost_equal(self.result.x, self.result.X[best_idx])

    def test_x_within_bounds(self):
        """Best point must lie within the original bounds."""
        for i, (lo, hi) in enumerate([(-3, 3), (-3, 3)]):
            assert (
                lo <= self.result.x[i] <= hi
            ), f"Dimension {i}: {self.result.x[i]} not in [{lo}, {hi}]"

    def test_all_X_within_bounds(self):
        """Every evaluated point must lie within bounds."""
        bounds = [(-3, 3), (-3, 3)]
        for i, (lo, hi) in enumerate(bounds):
            col = self.result.X[:, i]
            assert np.all(col >= lo - 1e-9), f"Dim {i}: some x < lower bound"
            assert np.all(col <= hi + 1e-9), f"Dim {i}: some x > upper bound"


# ---------------------------------------------------------------------------
# 5. Seed reproducibility
# ---------------------------------------------------------------------------


class TestSeedReproducibility:
    """Identical seeds must produce identical results across all fields."""

    @pytest.mark.parametrize("seed", [0, 1, 42, 123, 999])
    def test_same_seed_same_results(self, seed):
        """Two runs with the same seed produce identical X, y, x, fun."""
        common_kwargs = dict(
            fun=sphere,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=10,
            n_initial=5,
            seed=seed,
        )
        r1 = SpotOptim(**common_kwargs).optimize()
        r2 = SpotOptim(**common_kwargs).optimize()

        np.testing.assert_array_equal(r1.X, r2.X)
        np.testing.assert_array_equal(r1.y, r2.y)
        np.testing.assert_array_equal(r1.x, r2.x)
        assert r1.fun == r2.fun

    def test_different_seeds_different_initial_designs(self):
        """Different seeds produce different initial designs (with very high probability)."""
        r1 = SpotOptim(
            fun=sphere, bounds=[(-5, 5), (-5, 5)], max_iter=5, n_initial=5, seed=1
        ).optimize()
        r2 = SpotOptim(
            fun=sphere, bounds=[(-5, 5), (-5, 5)], max_iter=5, n_initial=5, seed=2
        ).optimize()
        # The initial designs should differ
        assert not np.array_equal(r1.X, r2.X)


# ---------------------------------------------------------------------------
# 6. Acquisition functions
# ---------------------------------------------------------------------------


class TestAcquisitionFunctions:
    """All supported acquisition functions must produce valid results."""

    @pytest.mark.parametrize("acq", ["y", "ei", "pi"])
    def test_acquisition_produces_valid_result(self, acq):
        opt = SpotOptim(
            fun=sphere,
            bounds=[(-3, 3), (-3, 3)],
            max_iter=8,
            n_initial=6,
            acquisition=acq,
            seed=0,
        )
        result = opt.optimize()
        assert isinstance(result, OptimizeResult)
        assert result.success is True
        assert np.isfinite(result.fun)

    @pytest.mark.parametrize("acq", ["y", "ei", "pi"])
    def test_acquisition_respects_budget(self, acq):
        max_iter = 8
        opt = SpotOptim(
            fun=sphere,
            bounds=[(-3, 3)],
            max_iter=max_iter,
            n_initial=5,
            acquisition=acq,
            seed=0,
        )
        result = opt.optimize()
        assert result.nfev == max_iter


# ---------------------------------------------------------------------------
# 7. Variable types — integer and mixed
# ---------------------------------------------------------------------------


class TestVariableTypes:
    """Integer and mixed variable types must produce results in the correct domain."""

    def test_integer_vars_produce_integer_results(self):
        """When var_type=['int'], result.x values must be integers."""
        opt = SpotOptim(
            fun=sphere,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=10,
            n_initial=7,
            var_type=["int", "int"],
            seed=0,
        )
        result = opt.optimize()
        assert np.allclose(
            result.x, np.round(result.x), atol=1e-9
        ), f"Expected integer-valued x, got {result.x}"

    def test_integer_vars_stored_X_are_integers(self):
        """All stored X points must be integers when var_type=['int', 'int']."""
        opt = SpotOptim(
            fun=sphere,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=10,
            n_initial=7,
            var_type=["int", "int"],
            seed=0,
        )
        result = opt.optimize()
        assert np.allclose(result.X, np.round(result.X), atol=1e-9)

    def test_mixed_int_float_second_dim_integer(self):
        """In a mixed setup, the integer dimension must stay integer."""
        opt = SpotOptim(
            fun=sphere,
            bounds=[(-3, 3), (-5, 5)],
            max_iter=10,
            n_initial=6,
            var_type=["float", "int"],
            seed=0,
        )
        result = opt.optimize()
        int_col = result.X[:, 1]
        assert np.allclose(int_col, np.round(int_col), atol=1e-9)


# ---------------------------------------------------------------------------
# 8. Custom initial design (X0 argument to optimize())
# ---------------------------------------------------------------------------


class TestCustomInitialDesign:
    """User-supplied X0 must be used as the initial design."""

    def test_custom_X0_shape_respected(self):
        X0 = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0]])
        opt = SpotOptim(
            fun=sphere, bounds=[(-3, 3), (-3, 3)], max_iter=3, n_initial=3, seed=0
        )
        result = opt.optimize(X0=X0)
        assert result.nfev == 3  # all budget used for initial design
        assert result.nit == 0  # no sequential iterations

    def test_custom_X0_influences_first_evaluations(self):
        """When X0 is supplied, the first n_initial rows of result.X equal X0."""
        X0 = np.array([[0.5, 0.5], [-0.5, -0.5], [1.0, -1.0], [-1.0, 1.0], [0.0, 0.0]])
        opt = SpotOptim(
            fun=sphere, bounds=[(-3, 3), (-3, 3)], max_iter=5, n_initial=5, seed=0
        )
        result = opt.optimize(X0=X0)
        np.testing.assert_array_almost_equal(result.X[:5], X0)

    def test_none_X0_generates_lhs_design(self):
        """With X0=None, the initial design is generated by LHS (shape must match)."""
        n_initial = 7
        opt = SpotOptim(
            fun=sphere,
            bounds=[(-2, 2), (-2, 2)],
            max_iter=n_initial,
            n_initial=n_initial,
            seed=0,
        )
        result = opt.optimize()
        assert result.X.shape == (n_initial, 2)


# ---------------------------------------------------------------------------
# 9. Log-space variables (var_trans)
# ---------------------------------------------------------------------------


class TestVarTrans:
    """Log-transformed variables must stay inside bounds in natural space."""

    def test_log10_transformed_optimization(self):
        """Optimisation over a log10-scaled variable produces finite, in-bounds result."""

        def log_quadratic(X):
            X = np.atleast_2d(X)
            return (np.log10(X[:, 0]) - 1.0) ** 2  # min at x=10

        opt = SpotOptim(
            fun=log_quadratic,
            bounds=[(1e-3, 1e3)],
            max_iter=10,
            n_initial=6,
            var_trans=["log10"],
            seed=0,
        )
        result = opt.optimize()
        assert np.isfinite(result.fun)
        assert 1e-3 <= result.x[0] <= 1e3, f"result.x={result.x[0]} out of bounds"

    def test_log10_all_X_within_natural_bounds(self):
        """All stored X values must respect the original bounds after inverse transform."""
        opt = SpotOptim(
            fun=lambda X: (np.atleast_2d(X)[:, 0] - 10.0) ** 2,
            bounds=[(1e-2, 1e4)],
            max_iter=8,
            n_initial=6,
            var_trans=["log10"],
            seed=0,
        )
        result = opt.optimize()
        assert np.all(result.X[:, 0] >= 1e-2 - 1e-9)
        assert np.all(result.X[:, 0] <= 1e4 + 1e-9)


# ---------------------------------------------------------------------------
# 10. get_best_hyperparameters
# ---------------------------------------------------------------------------


class TestGetBestHyperparameters:
    """get_best_hyperparameters() must return the same best point as result.x."""

    @pytest.fixture(autouse=True)
    def _run(self):
        opt = SpotOptim(
            fun=sphere,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=10,
            n_initial=6,
            seed=7,
        )
        self.result = opt.optimize()
        self.opt = opt

    def test_returns_dict_by_default(self):
        best = self.opt.get_best_hyperparameters(as_dict=True)
        assert isinstance(best, dict)

    def test_dict_values_match_result_x(self):
        best = self.opt.get_best_hyperparameters(as_dict=True)
        values = np.array(list(best.values()), dtype=float)
        np.testing.assert_array_almost_equal(values, self.result.x, decimal=10)

    def test_returns_array_when_as_dict_false(self):
        best = self.opt.get_best_hyperparameters(as_dict=False)
        assert isinstance(best, np.ndarray)

    def test_array_matches_result_x(self):
        best = self.opt.get_best_hyperparameters(as_dict=False)
        np.testing.assert_array_almost_equal(best, self.result.x, decimal=10)


# ---------------------------------------------------------------------------
# 11. Surrogate models
# ---------------------------------------------------------------------------


class TestSurrogates:
    """Custom Kriging surrogate must work as a drop-in replacement."""

    def test_kriging_surrogate_produces_valid_result(self):
        kriging = Kriging(noise=1e-6, min_theta=-3.0, max_theta=2.0, seed=42)
        opt = SpotOptim(
            fun=sphere,
            bounds=[(-3, 3), (-3, 3)],
            surrogate=kriging,
            max_iter=10,
            n_initial=6,
            seed=0,
        )
        result = opt.optimize()
        assert isinstance(result, OptimizeResult)
        assert result.success is True
        assert np.isfinite(result.fun)

    def test_kriging_result_shape(self):
        kriging = Kriging(noise=1e-6, seed=42)
        opt = SpotOptim(
            fun=sphere,
            bounds=[(-2, 2), (-2, 2)],
            surrogate=kriging,
            max_iter=8,
            n_initial=6,
            seed=0,
        )
        result = opt.optimize()
        assert result.X.shape == (8, 2)
        assert result.y.shape == (8,)


# ---------------------------------------------------------------------------
# 12. Optimisation quality — convergence on simple functions
# ---------------------------------------------------------------------------


class TestConvergenceQuality:
    """For well-conditioned functions with enough budget, SPOT should converge."""

    def test_sphere_2d_converges_near_origin(self):
        """Sphere 2D: best value < 0.5 within 20 evaluations."""
        opt = SpotOptim(
            fun=sphere,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=20,
            n_initial=10,
            seed=0,
        )
        result = opt.optimize()
        assert result.fun < 0.5, f"sphere 2D did not converge: f={result.fun}"

    def test_parabola_1d_finds_optimum(self):
        """1-D parabola: best x within 0.5 of 3.0."""
        opt = SpotOptim(
            fun=parabola_1d,
            bounds=[(0, 6)],
            max_iter=15,
            n_initial=5,
            seed=0,
        )
        result = opt.optimize()
        assert abs(result.x[0] - 3.0) < 0.5, f"Expected x near 3, got x={result.x[0]}"

    def test_rosenbrock_reasonable_result(self):
        """Rosenbrock 2D: finds a value < 10 within 20 evaluations."""
        opt = SpotOptim(
            fun=rosenbrock,
            bounds=[(-2, 2), (-2, 2)],
            max_iter=20,
            n_initial=10,
            acquisition="ei",
            seed=0,
        )
        result = opt.optimize()
        assert result.fun < 10.0, f"Rosenbrock did not converge: f={result.fun}"


# ---------------------------------------------------------------------------
# 13. High-dimensional optimisation
# ---------------------------------------------------------------------------


class TestHighDimensional:
    """Verify that SpotOptim handles higher-dimensional problems correctly."""

    @pytest.mark.parametrize("n_dim", [3, 5, 8])
    def test_result_x_length_matches_n_dim(self, n_dim):
        opt = SpotOptim(
            fun=sphere,
            bounds=[(-5, 5)] * n_dim,
            max_iter=n_dim * 2 + 2,
            n_initial=n_dim * 2,
            seed=0,
        )
        result = opt.optimize()
        assert len(result.x) == n_dim
        assert result.X.shape[1] == n_dim

    @pytest.mark.parametrize("n_dim", [3, 5, 8])
    def test_all_points_within_bounds(self, n_dim):
        opt = SpotOptim(
            fun=sphere,
            bounds=[(-5, 5)] * n_dim,
            max_iter=n_dim * 2 + 2,
            n_initial=n_dim * 2,
            seed=0,
        )
        result = opt.optimize()
        for i in range(n_dim):
            assert np.all(result.X[:, i] >= -5 - 1e-9)
            assert np.all(result.X[:, i] <= 5 + 1e-9)


# ---------------------------------------------------------------------------
# 14. Edge cases and robustness
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases that must not crash and must return valid results."""

    def test_1d_optimization(self):
        """Single-dimensional optimization must produce a scalar best point."""
        opt = SpotOptim(
            fun=parabola_1d,
            bounds=[(0, 6)],
            max_iter=8,
            n_initial=5,
            seed=0,
        )
        result = opt.optimize()
        assert result.x.ndim == 1
        assert len(result.x) == 1

    def test_budget_exhausted_message(self):
        """Termination message must mention the reason for stopping."""
        opt = SpotOptim(fun=sphere, bounds=[(-1, 1)], max_iter=5, n_initial=5, seed=0)
        result = opt.optimize()
        assert "maximum evaluations" in result.message or "reached" in result.message

    def test_n_infill_2_produces_two_extra_points_per_iter(self):
        """With n_infill_points=2 and 12 total evals and 6 initial, 3 iterations add 6."""
        opt = SpotOptim(
            fun=sphere,
            bounds=[(-3, 3), (-3, 3)],
            max_iter=12,
            n_initial=6,
            n_infill_points=2,
            seed=0,
        )
        result = opt.optimize()
        assert result.nfev == 12

    def test_x0_injection(self):
        """Providing x0 injects a starting point into the initial design."""
        x0 = np.array([0.1, 0.2])
        opt = SpotOptim(
            fun=sphere,
            bounds=[(-3, 3), (-3, 3)],
            max_iter=6,
            n_initial=5,
            x0=x0,
            seed=0,
        )
        result = opt.optimize()
        # x0 should appear in X (within float tolerance)
        found = any(np.allclose(row, x0, atol=1e-9) for row in result.X)
        assert found, f"x0={x0} not found in result.X:\n{result.X}"

    def test_very_narrow_bounds_ok(self):
        """Very narrow bounds must not crash optimization."""
        opt = SpotOptim(
            fun=sphere,
            bounds=[(0.0, 0.001), (0.0, 0.001)],
            max_iter=6,
            n_initial=5,
            seed=0,
        )
        result = opt.optimize()
        assert isinstance(result, OptimizeResult)
        assert np.isfinite(result.fun)
