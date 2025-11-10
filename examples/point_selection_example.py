"""
Example demonstrating point selection mechanism in SpotOptim.

This example shows how SpotOptim can automatically select a subset of points
for surrogate model fitting when the number of evaluated points becomes large.
This is useful for:
- Reducing computational cost of surrogate model training
- Handling large-scale optimization problems
- Maintaining surrogate model accuracy with carefully selected points
"""

import numpy as np
from spotoptim import SpotOptim
import matplotlib.pyplot as plt


def expensive_function(X):
    """
    A multi-modal test function (Rastrigin-like).
    This simulates an expensive objective function.
    """
    return np.sum(X**2 - 10 * np.cos(2 * np.pi * X) + 10, axis=1)


def main():
    print("=" * 70)
    print("SpotOptim Point Selection Mechanism Example")
    print("=" * 70)
    print()

    # Problem setup
    n_dim = 5
    bounds = [(-5.12, 5.12)] * n_dim

    print(f"Problem: {n_dim}-dimensional Rastrigin function")
    print(f"Bounds: {bounds[0]}")
    print()

    # Test 1: No point selection (use all points)
    print("-" * 70)
    print("Test 1: Optimization WITHOUT point selection")
    print("-" * 70)
    optimizer1 = SpotOptim(
        fun=expensive_function,
        bounds=bounds,
        max_iter=50,
        n_initial=10,
        max_surrogate_points=None,  # Use all points
        selection_method="distant",
        verbose=False,
        seed=42,
    )
    result1 = optimizer1.optimize()
    print(f"Function evaluations: {result1.nfev}")
    print(f"Best function value: {result1.fun:.6f}")
    print(f"Total points used for surrogate: {len(optimizer1.X_)}")
    print()

    # Test 2: With distant point selection
    print("-" * 70)
    print("Test 2: Optimization WITH point selection (distant method)")
    print("-" * 70)
    optimizer2 = SpotOptim(
        fun=expensive_function,
        bounds=bounds,
        max_iter=50,
        n_initial=10,
        max_surrogate_points=20,  # Use only 20 points for surrogate
        selection_method="distant",
        verbose=False,
        seed=42,
    )
    result2 = optimizer2.optimize()
    print(f"Function evaluations: {result2.nfev}")
    print(f"Best function value: {result2.fun:.6f}")
    print(f"Total points evaluated: {len(optimizer2.X_)}")
    print(f"Max points for surrogate: {optimizer2.max_surrogate_points}")
    print()

    # Test 3: With best cluster selection
    print("-" * 70)
    print("Test 3: Optimization WITH point selection (best cluster method)")
    print("-" * 70)
    optimizer3 = SpotOptim(
        fun=expensive_function,
        bounds=bounds,
        max_iter=50,
        n_initial=10,
        max_surrogate_points=20,  # Use 20 clusters
        selection_method="best",  # Select points from best cluster
        verbose=False,
        seed=42,
    )
    result3 = optimizer3.optimize()
    print(f"Function evaluations: {result3.nfev}")
    print(f"Best function value: {result3.fun:.6f}")
    print(f"Total points evaluated: {len(optimizer3.X_)}")
    print(f"Max clusters: {optimizer3.max_surrogate_points}")
    print()

    # Comparison
    print("=" * 70)
    print("Summary Comparison")
    print("=" * 70)
    print(f"{'Method':<30} {'Function Evals':<20} {'Best Value':<15}")
    print("-" * 70)
    print(f"{'No selection':<30} {result1.nfev:<20} {result1.fun:<15.6f}")
    print(f"{'Distant selection':<30} {result2.nfev:<20} {result2.fun:<15.6f}")
    print(f"{'Best cluster selection':<30} {result3.nfev:<20} {result3.fun:<15.6f}")
    print()

    # Visualization for 2D case
    if n_dim == 2:
        print("Generating visualizations (2D case)...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for idx, (opt, title) in enumerate([
            (optimizer1, "No Selection"),
            (optimizer2, "Distant Selection"),
            (optimizer3, "Best Cluster Selection"),
        ]):
            ax = axes[idx]
            ax.scatter(opt.X_[:, 0], opt.X_[:, 1], c=opt.y_, cmap="viridis", alpha=0.6)
            ax.scatter(
                opt.best_x_[0],
                opt.best_x_[1],
                c="red",
                marker="*",
                s=500,
                edgecolors="black",
                label="Best point",
            )
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("point_selection_comparison.png", dpi=150)
        print("Saved visualization to 'point_selection_comparison.png'")

    print()
    print("=" * 70)
    print("Key Insights:")
    print("=" * 70)
    print("1. Point selection reduces computational cost of surrogate training")
    print("2. 'Distant' method: Selects space-filling points via K-means clustering")
    print("3. 'Best' method: Focuses on region with best objective values")
    print("4. Both methods maintain optimization quality with fewer points")
    print()


if __name__ == "__main__":
    main()
