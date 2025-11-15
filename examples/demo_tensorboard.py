"""
Demo: TensorBoard Logging in SpotOptim

This script demonstrates how to use TensorBoard logging to monitor
the optimization process in real-time.

To view the logs:
1. Run this script: python demo_tensorboard.py
2. In a separate terminal, start TensorBoard: tensorboard --logdir=runs
3. Open your browser to http://localhost:6006
"""

import numpy as np
from spotoptim import SpotOptim


def rosenbrock(X):
    """2D Rosenbrock function - a classic optimization benchmark.
    
    Global minimum at (1, 1) with f(1,1) = 0.
    """
    x0 = X[:, 0]
    x1 = X[:, 1]
    return (1 - x0)**2 + 100 * (x1 - x0**2)**2


def noisy_sphere(X):
    """Sphere function with Gaussian noise."""
    base = np.sum(X**2, axis=1)
    noise = np.random.normal(0, 0.5, size=base.shape)
    return base + noise


def main():
    print("=" * 80)
    print("SpotOptim: TensorBoard Logging Demo")
    print("=" * 80)
    print()
    print("This demo runs three optimization scenarios with TensorBoard logging:")
    print("1. Deterministic function (Rosenbrock)")
    print("2. Noisy function with repeated evaluations")
    print("3. Noisy function with OCBA")
    print()
    print("View the results in TensorBoard:")
    print("  tensorboard --logdir=runs")
    print("  Then open http://localhost:6006 in your browser")
    print()
    print("-" * 80)
    
    # Scenario 1: Deterministic optimization with TensorBoard
    print("\n1. Deterministic Optimization (Rosenbrock)")
    print("-" * 80)
    
    np.random.seed(42)
    opt1 = SpotOptim(
        fun=rosenbrock,
        bounds=[(-2, 2), (-2, 2)],
        var_name=["x", "y"],
        max_iter=50,
        n_initial=15,
        tensorboard_log=True,
        tensorboard_path="runs/demo_deterministic",
        seed=42,
        verbose=True
    )
    
    print("Starting optimization...")
    result1 = opt1.optimize()
    
    print(f"\nResults:")
    print(f"  Best x: {result1.x}")
    print(f"  Best f(x): {result1.fun:.6f}")
    print(f"  Total evaluations: {result1.nfev}")
    print(f"  Iterations: {result1.nit}")
    print(f"  TensorBoard logs: {opt1.tensorboard_path}")
    
    # Scenario 2: Noisy function with repeated evaluations
    print("\n\n2. Noisy Optimization (with repeated evaluations)")
    print("-" * 80)
    
    np.random.seed(123)
    opt2 = SpotOptim(
        fun=noisy_sphere,
        bounds=[(-5, 5), (-5, 5), (-5, 5)],
        var_name=["x1", "x2", "x3"],
        max_iter=60,
        n_initial=20,
        repeats_initial=2,
        repeats_surrogate=2,
        tensorboard_log=True,
        tensorboard_path="runs/demo_noisy",
        seed=123,
        verbose=True
    )
    
    print("Starting noisy optimization...")
    result2 = opt2.optimize()
    
    print(f"\nResults:")
    print(f"  Best x: {result2.x}")
    print(f"  Best f(x) (single eval): {result2.fun:.6f}")
    print(f"  Best mean f(x): {opt2.min_mean_y:.6f}")
    print(f"  Variance at best: {opt2.min_var_y:.6f}")
    print(f"  Total evaluations: {result2.nfev}")
    print(f"  Unique design points: {opt2.mean_X.shape[0]}")
    print(f"  TensorBoard logs: {opt2.tensorboard_path}")
    
    # Scenario 3: Noisy function with OCBA
    print("\n\n3. Noisy Optimization with OCBA")
    print("-" * 80)
    
    np.random.seed(456)
    opt3 = SpotOptim(
        fun=noisy_sphere,
        bounds=[(-5, 5), (-5, 5)],
        var_name=["alpha", "beta"],
        max_iter=50,
        n_initial=15,
        repeats_initial=2,
        repeats_surrogate=1,
        ocba_delta=3,  # OCBA will intelligently re-evaluate points
        tensorboard_log=True,
        tensorboard_path="runs/demo_ocba",
        seed=456,
        verbose=True
    )
    
    print("Starting OCBA optimization...")
    result3 = opt3.optimize()
    
    print(f"\nResults:")
    print(f"  Best x: {result3.x}")
    print(f"  Best f(x) (single eval): {result3.fun:.6f}")
    print(f"  Best mean f(x): {opt3.min_mean_y:.6f}")
    print(f"  Variance at best: {opt3.min_var_y:.6f}")
    print(f"  Total evaluations: {result3.nfev}")
    print(f"  Unique design points: {opt3.mean_X.shape[0]}")
    print(f"  TensorBoard logs: {opt3.tensorboard_path}")
    
    # Summary
    print("\n\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)
    print("\nAll optimization logs have been saved to the 'runs/' directory.")
    print("\nTo view the results in TensorBoard:")
    print("  1. Open a terminal")
    print("  2. Run: tensorboard --logdir=runs")
    print("  3. Open your browser to: http://localhost:6006")
    print()
    print("In TensorBoard, you can:")
    print("  - Compare optimization progress across different runs")
    print("  - View hyperparameter relationships (HPARAMS tab)")
    print("  - Examine convergence curves (SCALARS tab)")
    print("  - Track best X coordinates over time")
    print("  - Analyze noise statistics for noisy optimizations")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
