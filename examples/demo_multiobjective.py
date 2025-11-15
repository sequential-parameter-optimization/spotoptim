"""
Demonstration of multi-objective optimization support in SpotOptim.

This script shows how to use SpotOptim with multi-objective functions,
including default behavior and custom conversion strategies.
"""

import numpy as np
from spotoptim import SpotOptim
import matplotlib.pyplot as plt


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def scenario_1_default_behavior():
    """Scenario 1: Multi-objective with default behavior (use first objective)."""
    print_section("Scenario 1: Multi-Objective with Default Behavior")
    
    def bi_objective_sphere(X):
        """Two sphere functions with different centers."""
        obj1 = np.sum(X**2, axis=1)  # Minimize distance from origin
        obj2 = np.sum((X - 2)**2, axis=1)  # Minimize distance from (2, 2)
        return np.column_stack([obj1, obj2])
    
    print("\nObjective function:")
    print("  obj1: sum(x^2)           - minimize distance from origin")
    print("  obj2: sum((x-2)^2)       - minimize distance from (2, 2)")
    print("\nDefault behavior: optimize first objective (obj1)")
    
    optimizer = SpotOptim(
        fun=bi_objective_sphere,
        bounds=[(-5, 5), (-5, 5)],
        max_iter=30,
        n_initial=15,
        seed=42,
        verbose=True
    )
    
    result = optimizer.optimize()
    
    print(f"\nOptimization Results:")
    print(f"  Best x: {result.x}")
    print(f"  Best f(x): {result.fun:.6f}")
    print(f"  Multi-objective values stored: {optimizer.y_mo.shape}")
    print(f"    - Objective 1 at best: {optimizer.y_mo[np.argmin(optimizer.y_), 0]:.6f}")
    print(f"    - Objective 2 at best: {optimizer.y_mo[np.argmin(optimizer.y_), 1]:.6f}")
    print(f"\nSince we optimized obj1 (default), solution is near origin: {np.allclose(result.x, 0, atol=0.5)}")


def scenario_2_weighted_sum():
    """Scenario 2: Multi-objective with weighted sum scalarization."""
    print_section("Scenario 2: Multi-Objective with Weighted Sum")
    
    def bi_objective_sphere(X):
        """Two sphere functions with different centers."""
        obj1 = np.sum(X**2, axis=1)
        obj2 = np.sum((X - 2)**2, axis=1)
        return np.column_stack([obj1, obj2])
    
    def weighted_sum(y_mo, w1=0.5, w2=0.5):
        """Weighted sum scalarization."""
        return w1 * y_mo[:, 0] + w2 * y_mo[:, 1]
    
    print("\nUsing weighted sum: 0.5 * obj1 + 0.5 * obj2")
    print("This finds a compromise solution between the two objectives")
    
    optimizer = SpotOptim(
        fun=bi_objective_sphere,
        bounds=[(-5, 5), (-5, 5)],
        max_iter=30,
        n_initial=15,
        fun_mo2so=weighted_sum,
        seed=42,
        verbose=True
    )
    
    result = optimizer.optimize()
    
    print(f"\nOptimization Results:")
    print(f"  Best x: {result.x}")
    print(f"  Best f(x): {result.fun:.6f}")
    print(f"    - Objective 1 at best: {optimizer.y_mo[np.argmin(optimizer.y_), 0]:.6f}")
    print(f"    - Objective 2 at best: {optimizer.y_mo[np.argmin(optimizer.y_), 1]:.6f}")
    print(f"\nCompromise solution found near x=(1, 1): {np.allclose(result.x, 1, atol=0.5)}")


def scenario_3_min_max():
    """Scenario 3: Multi-objective with min-max scalarization."""
    print_section("Scenario 3: Multi-Objective with Min-Max Strategy")
    
    def bi_objective_rosenbrock(X):
        """Two Rosenbrock functions with different optima."""
        # First Rosenbrock: optimum at (1, 1)
        obj1 = (1 - X[:, 0])**2 + 100 * (X[:, 1] - X[:, 0]**2)**2
        # Second Rosenbrock: optimum at (-1, 1)
        obj2 = (-1 - X[:, 0])**2 + 100 * (X[:, 1] - X[:, 0]**2)**2
        return np.column_stack([obj1, obj2])
    
    def min_max(y_mo):
        """Min-max scalarization: minimize the maximum objective."""
        return np.max(y_mo, axis=1)
    
    print("\nUsing min-max: minimize max(obj1, obj2)")
    print("This ensures balanced performance across objectives")
    
    optimizer = SpotOptim(
        fun=bi_objective_rosenbrock,
        bounds=[(-2, 2), (-2, 2)],
        max_iter=50,
        n_initial=25,
        fun_mo2so=min_max,
        seed=42,
        verbose=True
    )
    
    result = optimizer.optimize()
    
    print(f"\nOptimization Results:")
    print(f"  Best x: {result.x}")
    print(f"  Best f(x): {result.fun:.6f}")
    print(f"    - Objective 1 at best: {optimizer.y_mo[np.argmin(optimizer.y_), 0]:.6f}")
    print(f"    - Objective 2 at best: {optimizer.y_mo[np.argmin(optimizer.y_), 1]:.6f}")


def scenario_4_noisy_multiobjective():
    """Scenario 4: Noisy multi-objective optimization."""
    print_section("Scenario 4: Noisy Multi-Objective Optimization")
    
    def noisy_bi_objective(X):
        """Noisy bi-objective function."""
        noise1 = np.random.normal(0, 0.05, X.shape[0])
        noise2 = np.random.normal(0, 0.05, X.shape[0])
        
        obj1 = np.sum(X**2, axis=1) + noise1
        obj2 = np.sum((X - 1)**2, axis=1) + noise2
        return np.column_stack([obj1, obj2])
    
    print("\nNoisy bi-objective function with repeated evaluations")
    print("Using 3 initial repeats and 2 surrogate repeats")
    
    optimizer = SpotOptim(
        fun=noisy_bi_objective,
        bounds=[(-5, 5), (-5, 5)],
        max_iter=40,
        n_initial=20,
        repeats_initial=3,
        repeats_surrogate=2,
        seed=42,
        verbose=True
    )
    
    result = optimizer.optimize()
    
    print(f"\nOptimization Results:")
    print(f"  Best x: {result.x}")
    print(f"  Best mean f(x): {result.fun:.6f}")
    print(f"  Noise handling active: {optimizer.noise}")
    print(f"  Total evaluations: {len(optimizer.y_)}")
    print(f"  Unique design points: {optimizer.mean_X.shape[0]}")
    print(f"  Multi-objective values stored: {optimizer.y_mo.shape}")


def scenario_5_three_objectives():
    """Scenario 5: Three-objective optimization."""
    print_section("Scenario 5: Three-Objective Optimization")
    
    def three_objective_function(X):
        """Three objectives with different characteristics."""
        obj1 = np.sum(X**2, axis=1)  # Convex, minimum at origin
        obj2 = np.sum(np.abs(X), axis=1)  # L1 norm
        obj3 = np.max(np.abs(X), axis=1)  # L-infinity norm
        return np.column_stack([obj1, obj2, obj3])
    
    def weighted_three(y_mo):
        """Weighted combination of three objectives."""
        return 0.4 * y_mo[:, 0] + 0.3 * y_mo[:, 1] + 0.3 * y_mo[:, 2]
    
    print("\nThree objectives:")
    print("  obj1: L2 norm (sum of squares)")
    print("  obj2: L1 norm (sum of absolute values)")
    print("  obj3: L-infinity norm (maximum absolute value)")
    print("\nUsing weighted combination: 0.4*obj1 + 0.3*obj2 + 0.3*obj3")
    
    optimizer = SpotOptim(
        fun=three_objective_function,
        bounds=[(-5, 5), (-5, 5), (-5, 5)],
        max_iter=35,
        n_initial=20,
        fun_mo2so=weighted_three,
        seed=42,
        verbose=True
    )
    
    result = optimizer.optimize()
    
    print(f"\nOptimization Results:")
    print(f"  Best x: {result.x}")
    print(f"  Best f(x): {result.fun:.6f}")
    print(f"  Multi-objective values at best:")
    best_idx = np.argmin(optimizer.y_)
    print(f"    - obj1 (L2): {optimizer.y_mo[best_idx, 0]:.6f}")
    print(f"    - obj2 (L1): {optimizer.y_mo[best_idx, 1]:.6f}")
    print(f"    - obj3 (Linf): {optimizer.y_mo[best_idx, 2]:.6f}")


def main():
    """Run all demonstration scenarios."""
    print("=" * 70)
    print("Multi-Objective Optimization in SpotOptim")
    print("=" * 70)
    print("\nThis demo shows various multi-objective optimization strategies:")
    print("  1. Default behavior (optimize first objective)")
    print("  2. Weighted sum scalarization")
    print("  3. Min-max scalarization")
    print("  4. Noisy multi-objective optimization")
    print("  5. Three-objective optimization")
    
    # Run scenarios
    scenario_1_default_behavior()
    scenario_2_weighted_sum()
    scenario_3_min_max()
    scenario_4_noisy_multiobjective()
    scenario_5_three_objectives()
    
    # Summary
    print_section("Summary")
    print("\nSpotOptim Multi-Objective Support:")
    print("  ✓ Automatic detection of multi-objective functions")
    print("  ✓ Default: use first objective")
    print("  ✓ Custom conversion via fun_mo2so parameter")
    print("  ✓ All multi-objective values stored in y_mo attribute")
    print("  ✓ Compatible with noise handling, OCBA, TensorBoard, etc.")
    print("\nCommon Scalarization Strategies:")
    print("  • Weighted Sum: w1*obj1 + w2*obj2 + ...")
    print("  • Min-Max: minimize max(obj1, obj2, ...)")
    print("  • Target Achievement: minimize deviations from targets")
    print("  • Lexicographic: prioritize objectives in order")
    print("\nFor true multi-objective optimization (Pareto front),")
    print("consider specialized tools like pymoo or platypus.")
    print("=" * 70)


if __name__ == "__main__":
    main()
