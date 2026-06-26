# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression test: nominal (Hamming) kernel beats ordinal treatment of a factor.

Demonstrates the benefit of a categorical-aware kernel on ``factor_quadratic`` — a
mixed continuous + nominal-factor function whose per-level offsets are an
index-scrambled permutation (adjacency in the integer level code carries no
information about the objective).  Two effects are checked:

1. **Surrogate accuracy (deterministic).**  Fit on a sparse design (≈2 samples per
   level) and predict on a dense grid.  A Hamming-kernel Kriging predicts with
   substantially lower RMSE than the same Kriging treating the factor as a
   continuous/ordinal integer (sqeuclidean).  This is deterministic and is the
   robust core of the demonstration.

2. **Optimization outcome.**  Across seeds, the factor-aware (Hamming) arm reaches
   a better mean best-found value than the ordinal arm.  Both arms use the *same*
   Kriging engine; only the factor metric / var_type differs, so the contrast
   isolates the categorical-kernel effect.

See ``scripts/factor_kernel_benchmark.py`` for the full multi-seed run, the
Mann-Whitney statistics, and the convergence plot.
"""

import warnings

import numpy as np
import pytest

from spotoptim import SpotOptim
from spotoptim.function import factor_quadratic, FACTOR_QUADRATIC_LEVELS
from spotoptim.surrogate import Kriging

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------
_LEVELS = FACTOR_QUADRATIC_LEVELS  # 16 nominal levels "c00".."c15"
_BOUNDS = [(-3.0, 3.0), _LEVELS]
_OPT_LEVEL = "c04"  # global optimum level (offset == 0)
_N0 = 12  # initial design size
_M = 40  # total evaluation budget (including initial design)
_SEEDS = list(range(8))


def _surrogate_rmse() -> tuple[float, float]:
    """Deterministic surrogate-accuracy contrast on a sparse design.

    Returns:
        (rmse_hamming, rmse_ordinal) test-set RMSE for the two Kriging variants.
    """
    n = len(_LEVELS)
    rng = np.random.default_rng(7)
    x_tr = rng.uniform(-3.0, 3.0, size=2 * n)
    c_tr = rng.integers(0, n, size=2 * n).astype(float)
    X_tr = np.column_stack([x_tr, c_tr])
    y_tr = factor_quadratic(X_tr)  # numeric-index input is supported

    xs = np.linspace(-3.0, 3.0, 11)
    X_te = np.array([[x, c] for c in range(n) for x in xs], dtype=float)
    y_te = factor_quadratic(X_te)

    k_ham = Kriging(
        var_type=["float", "factor"],
        metric_factorial="hamming",
        seed=42,
        model_fun_evals=60,
    )
    k_ord = Kriging(var_type=["float", "float"], seed=42, model_fun_evals=60)
    k_ham.fit(X_tr, y_tr)
    k_ord.fit(X_tr, y_tr)

    rmse_ham = float(np.sqrt(np.mean((k_ham.predict(X_te) - y_te) ** 2)))
    rmse_ord = float(np.sqrt(np.mean((k_ord.predict(X_te) - y_te) ** 2)))
    return rmse_ham, rmse_ord


def _run_arm(arm_label: str, seed: int) -> tuple[float, bool]:
    """Run one optimization replicate for the given arm and seed.

    Args:
        arm_label: "H" for Hamming/factor-aware, "O" for ordinal/continuous.
        seed: Random seed for both the surrogate and SpotOptim.

    Returns:
        (best_fun, success) where success is True when the returned factor level
        equals the global optimum level ``"c04"``.
    """
    if arm_label == "H":
        surrogate = Kriging(
            var_type=["float", "factor"],
            metric_factorial="hamming",
            seed=seed,
            model_fun_evals=40,
        )
    else:
        surrogate = Kriging(var_type=["float", "float"], seed=seed, model_fun_evals=40)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        opt = SpotOptim(
            fun=factor_quadratic,
            bounds=_BOUNDS,
            surrogate=surrogate,
            n_initial=_N0,
            max_iter=_M,
            seed=seed,
        )
        result = opt.optimize()

    best_fun = float(result.fun)
    success = (len(result.x) > 1) and (result.x[1] == _OPT_LEVEL)
    return best_fun, success


@pytest.mark.slow
def test_factor_kernel_benchmark():
    """The factor-aware (Hamming) kernel beats ordinal treatment of the factor.

    Hard checks (validated 2026-06-26):
      * Surrogate RMSE: hamming ≈ 1.56 vs ordinal ≈ 2.45 (ratio ≈ 0.64), deterministic.
      * Optimization: mean best-found, hamming ≈ 0.21 vs ordinal ≈ 0.58 over 8 seeds.
    Margins are generous so the test is not flaky.
    """
    # --- 1. Deterministic surrogate-accuracy contrast (the robust core) ---
    rmse_ham, rmse_ord = _surrogate_rmse()
    assert np.isfinite(rmse_ham) and np.isfinite(rmse_ord)
    assert rmse_ham < 0.85 * rmse_ord, (
        f"Hamming surrogate RMSE ({rmse_ham:.3f}) should be clearly below the ordinal "
        f"surrogate RMSE ({rmse_ord:.3f}); ratio={rmse_ham / rmse_ord:.3f}"
    )

    # --- 2. Optimization outcome across seeds ---
    best_h, best_o, succ_h, succ_o = [], [], [], []
    for seed in _SEEDS:
        fH, sH = _run_arm("H", seed)
        fO, sO = _run_arm("O", seed)
        best_h.append(fH)
        best_o.append(fO)
        succ_h.append(sH)
        succ_o.append(sO)

    arr_h, arr_o = np.array(best_h), np.array(best_o)
    mean_h, mean_o = float(arr_h.mean()), float(arr_o.mean())
    rate_h, rate_o = float(np.mean(succ_h)), float(np.mean(succ_o))

    assert np.all(np.isfinite(arr_h)) and np.all(np.isfinite(arr_o))
    # The seeds are fixed, so these means are deterministic (not sampled). The
    # factor-aware arm reaches a better mean best-found value (~2.7x separation
    # observed: H≈0.21 vs O≈0.58); assert the robust direction.
    assert mean_h < mean_o, (
        f"Factor-aware (Hamming) mean best ({mean_h:.3f}) should be below the ordinal "
        f"arm ({mean_o:.3f}). success rates H={rate_h:.2f} O={rate_o:.2f}"
    )
    # Paired robustness: Hamming should win or tie on a majority of the (fixed) seeds.
    paired_h_at_least_as_good = int(np.sum(arr_h <= arr_o))
    assert (
        paired_h_at_least_as_good >= len(_SEEDS) // 2
    ), f"Hamming won/tied on only {paired_h_at_least_as_good}/{len(_SEEDS)} seeds"
    # The factor-aware arm should not find the optimum less often than the ordinal arm
    # (allow one seed of noise out of eight).
    assert (
        rate_h >= rate_o - 0.13
    ), f"Hamming success rate ({rate_h:.2f}) fell well below ordinal ({rate_o:.2f})"
