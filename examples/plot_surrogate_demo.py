"""
Example demonstrating the plot_surrogate() method of SpotOptim.

This script shows how to visualize the fitted surrogate model after optimization.
"""

import numpy as np
from spotoptim import SpotOptim


def sphere_function(X):
    """2D Sphere function: f(x) = x1^2 + x2^2"""
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1)


def main():
    # Set up the optimization problem
    bounds = [(-5, 5), (-5, 5)]
    
    print("="*60)
    print("SpotOptim - Surrogate Model Visualization Example")
    print("="*60)
    print(f"Objective: Sphere function (2D)")
    print(f"Bounds: {bounds}")
    print(f"True optimum: [0, 0] with f(x) = 0")
    print()

    # Create optimizer
    optimizer = SpotOptim(
        fun=sphere_function,
        bounds=bounds,
        max_iter=15,
        n_initial=8,
        acquisition='ei',
        seed=42,
        verbose=True
    )

    # Run optimization
    print("Running optimization...")
    result = optimizer.optimize()

    # Print results
    print("\n" + "="*60)
    print("Optimization Results")
    print("="*60)
    print(f"Best point found: {result.x}")
    print(f"Best function value: {result.fun:.6f}")
    print(f"Number of evaluations: {result.nfev}")
    print(f"Distance to optimum: {np.linalg.norm(result.x):.6f}")

    # Visualize the surrogate model
    print("\n" + "="*60)
    print("Surrogate Model Visualization")
    print("="*60)
    print("The plot shows:")
    print("  • Top left: 3D surface of predictions")
    print("  • Top right: 3D surface of prediction uncertainty")
    print("  • Bottom left: Contour plot with evaluated points (red)")
    print("  • Bottom right: Contour plot of uncertainty")
    print()
    print("Red dots indicate the points evaluated during optimization.")
    print("Notice how uncertainty decreases near evaluated points.")
    print()

    optimizer.plot_surrogate(
        i=0, j=1,
        var_name=['x1', 'x2'],
        add_points=True,
        cmap='viridis',
        contour_levels=25,
        show=True
    )


if __name__ == "__main__":
    main()
