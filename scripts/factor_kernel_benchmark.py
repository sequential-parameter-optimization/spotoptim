# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Empirical benchmark: nominal (Hamming) factor kernel vs ordinal treatment.

Demonstrates the benefit of a categorical-aware kernel on ``factor_quadratic`` — a
mixed continuous + nominal-factor function with 16 unordered levels whose per-level
offsets are an index-scrambled permutation (so adjacency in the integer level code
carries *no* information about the objective value).

Two arms share the *same* Kriging engine and differ only in how the factor column
is modelled:

  - **Arm H** (factor-aware): ``Kriging(var_type=["float", "factor"],
    metric_factorial="hamming")`` — treats the nominal factor order-agnostically
    (all distinct levels equidistant).
  - **Arm O** (ordinal/continuous): ``Kriging(var_type=["float", "float"])`` —
    forces the factor column into the ordered/sqeuclidean distance block, i.e.
    treats the level index as a continuous number.

Global optimum of ``factor_quadratic`` (default 16 levels): level "c04", x=0, f=0.

Two complementary results are reported:

1. **Surrogate accuracy (deterministic).**  Fit each variant on a sparse design
   (≈2 samples per level) and measure test-set RMSE.  The Hamming kernel yields a
   markedly lower RMSE — it does not impose a false ordering on the levels.

2. **Optimization (over seeds).**  The factor-aware arm reaches a better mean
   best-found value.  The benefit is largest when the number of levels is large
   relative to the budget — the regime typical of expensive black-box problems,
   where most levels are observed only a few times and the surrogate must
   generalize.  Treating the unordered factor as ordinal injects a systematic bias.

Background: Bartz-Beielstein & Zaefferer, on order-agnostic (Hamming/Gower)
distances for qualitative variables in Kriging-based optimization.

Usage::

    uv run python scripts/factor_kernel_benchmark.py
"""

import warnings
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

from spotoptim import SpotOptim
from spotoptim.function import factor_quadratic, FACTOR_QUADRATIC_LEVELS
from spotoptim.surrogate import Kriging

# Default plot output, resolved relative to the repository root (not the cwd).
_PLOT_PATH = (
    Path(__file__).resolve().parent.parent / "img" / "factor_kernel_benchmark.png"
)

# ---------------------------------------------------------------------------
# Experiment constants — tune here
# ---------------------------------------------------------------------------
N_SEEDS = 15  # number of independent replicates
N0 = 12  # initial design size (total budget includes this)
M = 40  # total evaluation budget (including initial design)
SEEDS = list(range(N_SEEDS))
LEVELS = FACTOR_QUADRATIC_LEVELS  # 16 nominal levels "c00".."c15"
BOUNDS = [(-3.0, 3.0), LEVELS]
OPT_LEVEL = "c04"  # factor level at global optimum


def surrogate_rmse():
    """Deterministic surrogate-accuracy contrast on a sparse design.

    Returns:
        tuple: (rmse_hamming, rmse_ordinal) test-set RMSE for the two Kriging
        variants fit on the same sparse training set.
    """
    n = len(LEVELS)
    rng = np.random.default_rng(7)
    X_tr = np.column_stack(
        [
            rng.uniform(-3.0, 3.0, size=2 * n),
            rng.integers(0, n, size=2 * n).astype(float),
        ]
    )
    y_tr = factor_quadratic(X_tr)
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


def run_arm(arm_label, seed):
    """Run one optimization replicate for the given arm and seed.

    Args:
        arm_label (str): "H" for factor-aware, "O" for ordinal.
        seed (int): Random seed for both the surrogate and SpotOptim.

    Returns:
        tuple: (best_fun, success, convergence_trace).
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
            bounds=BOUNDS,
            surrogate=surrogate,
            n_initial=N0,
            max_iter=M,
            seed=seed,
        )
        result = opt.optimize()

    best_fun = float(result.fun)
    best_level = result.x[1] if len(result.x) > 1 else None
    success = best_level == OPT_LEVEL
    convergence_trace = np.minimum.accumulate(result.y)
    return best_fun, success, convergence_trace


def run_experiment():
    """Run the full benchmark over all seeds for both arms.

    Returns:
        dict: Results keyed by arm label ("H", "O").
    """
    results = {
        "H": {"best": [], "success": [], "traces": []},
        "O": {"best": [], "success": [], "traces": []},
    }
    for seed in SEEDS:
        for arm in ("H", "O"):
            best_fun, success, trace = run_arm(arm, seed)
            results[arm]["best"].append(best_fun)
            results[arm]["success"].append(success)
            results[arm]["traces"].append(trace)
    return results


def print_summary(results, rmse_ham, rmse_ord):
    """Print a summary table comparing both arms.

    Args:
        results (dict): Output of :func:`run_experiment`.
        rmse_ham (float): Hamming-kernel surrogate RMSE.
        rmse_ord (float): Ordinal-kernel surrogate RMSE.
    """
    best_h = np.array(results["H"]["best"])
    best_o = np.array(results["O"]["best"])
    succ_h = np.mean(results["H"]["success"])
    succ_o = np.mean(results["O"]["success"])
    stat, p_value = mannwhitneyu(best_h, best_o, alternative="less")

    header = (
        f"{'Arm':<6} {'Mean':>10} {'Std':>10} {'Median':>10} "
        f"{'SuccRate':>10}  (N={N_SEEDS}, N0={N0}, M={M})"
    )
    sep = "-" * len(header)
    print()
    print("=" * len(header))
    print("Factor-kernel benchmark: Hamming (H) vs Ordinal/continuous (O)")
    print("=" * len(header))
    print("Surrogate RMSE on a sparse design (deterministic):")
    print(
        f"    Hamming = {rmse_ham:.3f}   Ordinal = {rmse_ord:.3f}   "
        f"ratio = {rmse_ham / rmse_ord:.3f}  ({'Hamming better' if rmse_ham < rmse_ord else 'Ordinal better'})"
    )
    print(sep)
    print(header)
    print(sep)
    print(
        f"{'H':<6} {np.mean(best_h):>10.4f} {np.std(best_h):>10.4f} "
        f"{np.median(best_h):>10.4f} {succ_h:>10.2%}"
    )
    print(
        f"{'O':<6} {np.mean(best_o):>10.4f} {np.std(best_o):>10.4f} "
        f"{np.median(best_o):>10.4f} {succ_o:>10.2%}"
    )
    print(sep)
    print(
        f"Mann-Whitney U test (H < O): U={stat:.0f}, p={p_value:.4f} "
        f"({'significant' if p_value < 0.05 else 'not significant'} at α=0.05)"
    )
    verdict = (
        "Hamming (factor-aware) wins"
        if np.mean(best_h) < np.mean(best_o)
        else "Ordinal wins"
    )
    print(f"Optimization verdict: {verdict} on mean best-found.")
    print("=" * len(header))
    print()


def save_plot(results, path=_PLOT_PATH):
    """Save mean best-so-far convergence plot for both arms.

    Args:
        results (dict): Output of :func:`run_experiment`.
        path: File path for the PNG output (defaults to ``<repo>/img/``).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "WARNING: matplotlib is not installed. "
            "Install the 'viz' extra to generate plots: uv sync --extra viz"
        )
        return

    min_len = min(
        min(len(t) for t in results["H"]["traces"]),
        min(len(t) for t in results["O"]["traces"]),
    )
    traces_h = np.array([t[:min_len] for t in results["H"]["traces"]])
    traces_o = np.array([t[:min_len] for t in results["O"]["traces"]])
    evals = np.arange(1, min_len + 1)
    mean_h, std_h = traces_h.mean(axis=0), traces_h.std(axis=0)
    mean_o, std_o = traces_o.mean(axis=0), traces_o.std(axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(evals, mean_h, color="tab:blue", label="Arm H (Hamming / factor-aware)")
    ax.fill_between(evals, mean_h - std_h, mean_h + std_h, alpha=0.25, color="tab:blue")
    ax.plot(evals, mean_o, color="tab:orange", label="Arm O (ordinal / sqeuclidean)")
    ax.fill_between(
        evals, mean_o - std_o, mean_o + std_o, alpha=0.25, color="tab:orange"
    )
    ax.set_xlabel("Function evaluations")
    ax.set_ylabel("Best objective value (lower is better)")
    ax.set_title(
        f"factor_quadratic ({len(LEVELS)} nominal levels): Hamming vs Ordinal kernel\n"
        f"N={N_SEEDS} seeds, N0={N0}, M={M}, global opt = 0.0"
    )
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Convergence plot saved to: {path}")


if __name__ == "__main__":
    print(f"Running benchmark: N_SEEDS={N_SEEDS}, N0={N0}, M={M} ...")
    rmse_ham, rmse_ord = surrogate_rmse()
    results = run_experiment()
    print_summary(results, rmse_ham, rmse_ord)
    save_plot(results)
