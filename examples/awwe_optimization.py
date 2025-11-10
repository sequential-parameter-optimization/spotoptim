"""
Aircraft Wing Weight Optimization with SpotOptim
=================================================

This script demonstrates surrogate-based optimization for the 9-dimensional
Aircraft Wing Weight Example (AWWE).

Run this script to see SpotOptim in action!
"""

import numpy as np
import matplotlib.pyplot as plt
from spotoptim import SpotOptim, Kriging

# Set random seed for reproducibility
np.random.seed(42)


def wingwt(X):
    """
    Aircraft Wing Weight function.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, 9)
        Input parameters in their natural scales:
        [Sw, Wfw, A, Lambda, q, lambda, Rtc, Nz, Wdg]
    
    Returns
    -------
    W : ndarray of shape (n_samples,)
        Wing weight in pounds
    """
    X = np.atleast_2d(X)
    
    # Extract parameters (already on natural scale)
    Sw = X[:, 0]      # Wing area
    Wfw = X[:, 1]     # Fuel weight
    A = X[:, 2]       # Aspect ratio
    Lambda = X[:, 3]  # Quarter-chord sweep (degrees)
    q = X[:, 4]       # Dynamic pressure
    lam = X[:, 5]     # Taper ratio (lambda)
    Rtc = X[:, 6]     # Thickness to chord ratio
    Nz = X[:, 7]      # Load factor
    Wdg = X[:, 8]     # Design gross weight
    
    # Convert sweep angle to radians
    Lambda_rad = Lambda * np.pi / 180
    
    # Calculate wing weight using the formula
    W = 0.036 * Sw**0.758 * Wfw**0.0035
    W *= (A / np.cos(Lambda_rad)**2)**0.6
    W *= q**0.006 * lam**0.04
    W *= (100 * Rtc / np.cos(Lambda_rad))**(-0.3)
    W *= (Nz * Wdg)**0.49
    
    return W


def main():
    """Run the optimization and display results."""
    
    print("\n" + "="*70)
    print("Aircraft Wing Weight Optimization with SpotOptim")
    print("="*70 + "\n")
    
    # Define bounds for each of the 9 parameters
    bounds = [
        (150, 200),    # Sw: Wing area
        (220, 300),    # Wfw: Fuel weight
        (6, 10),       # A: Aspect ratio
        (-10, 10),     # Lambda: Quarter-chord sweep
        (16, 45),      # q: Dynamic pressure
        (0.5, 1.0),    # lambda: Taper ratio
        (0.08, 0.18),  # Rtc: Thickness to chord ratio
        (2.5, 6.0),    # Nz: Load factor
        (1700, 2500)   # Wdg: Design gross weight
    ]
    
    var_names = ['Sw', 'Wfw', 'A', 'Lambda', 'q', 'lambda', 'Rtc', 'Nz', 'Wdg']
    
    # Baseline configuration (Cessna C172)
    baseline = np.array([[174, 252, 7.52, 0, 34, 0.672, 0.12, 3.8, 2000]])
    baseline_weight = wingwt(baseline)[0]
    
    print(f"Baseline wing weight: {baseline_weight:.2f} lb")
    print(f"Search space dimensions: {len(bounds)}")
    print(f"Total evaluations budget: 70 (20 initial + 50 optimization)\n")
    
    # Create and run optimizer
    print("Creating SpotOptim optimizer...")
    optimizer = SpotOptim(
        fun=wingwt,
        bounds=bounds,
        n_initial=20,
        max_iter=50,
        acquisition='ei',
        seed=42,
        verbose=True
    )
    
    print("\nStarting optimization...\n")
    result = optimizer.optimize()
    
    # Display results
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)
    print("\nOptimal Configuration:")
    print("-" * 70)
    for i, (name, value) in enumerate(zip(var_names, result.x)):
        baseline_val = baseline[0, i]
        change = ((value - baseline_val) / baseline_val) * 100
        print(f"  {name:8s} = {value:8.3f}  "
              f"(baseline: {baseline_val:7.3f}, change: {change:+6.1f}%)")
    
    print("\n" + "="*70)
    print(f"Optimal wing weight:   {result.fun:.2f} lb")
    print(f"Baseline weight:       {baseline_weight:.2f} lb")
    print(f"Weight reduction:      {baseline_weight - result.fun:.2f} lb "
          f"({((baseline_weight - result.fun)/baseline_weight*100):.1f}%)")
    print(f"Function evaluations:  {result.nfev}")
    print("="*70 + "\n")
    
    # Plot convergence
    print("Generating convergence plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # All evaluations
    ax1.scatter(range(len(optimizer.y_)), optimizer.y_, 
               alpha=0.6, s=50, c=range(len(optimizer.y_)), 
               cmap='viridis', edgecolors='black', linewidth=0.5)
    ax1.axhline(y=result.fun, color='r', linestyle='--', 
               linewidth=2, label=f'Best: {result.fun:.2f} lb')
    ax1.axhline(y=baseline_weight, color='g', linestyle='--', 
               linewidth=2, label=f'Baseline: {baseline_weight:.2f} lb')
    ax1.set_xlabel('Evaluation Number', fontsize=12)
    ax1.set_ylabel('Wing Weight (lb)', fontsize=12)
    ax1.set_title('All Function Evaluations', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Best so far
    best_so_far = np.minimum.accumulate(optimizer.y_)
    ax2.plot(best_so_far, linewidth=2.5, color='darkblue')
    ax2.axhline(y=result.fun, color='r', linestyle='--', linewidth=2, alpha=0.7)
    ax2.axhline(y=baseline_weight, color='g', linestyle='--', linewidth=2, alpha=0.7)
    ax2.axvline(x=optimizer.n_initial, color='orange', linestyle=':', linewidth=2)
    ax2.fill_between(range(optimizer.n_initial), 
                     ax2.get_ylim()[0], ax2.get_ylim()[1], 
                     alpha=0.2, color='orange')
    ax2.set_xlabel('Evaluation Number', fontsize=12)
    ax2.set_ylabel('Best Wing Weight (lb)', fontsize=12)
    ax2.set_title('Optimization Convergence', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('awwe_convergence.png', dpi=150, bbox_inches='tight')
    print("Saved convergence plot to 'awwe_convergence.png'\n")
    
    # Visualize surrogate for top 2 parameters
    print("Generating surrogate visualization...")
    changes = np.abs(result.x - baseline[0])
    relative_changes = changes / np.array([b[1] - b[0] for b in bounds])
    sorted_idx = np.argsort(relative_changes)[::-1]
    
    i, j = sorted_idx[0], sorted_idx[1]
    print(f"Plotting dimensions: {var_names[i]} (dim {i}) vs {var_names[j]} (dim {j})\n")
    
    optimizer.plot_surrogate(
        i=i, 
        j=j,
        var_name=var_names,
        add_points=True,
        cmap='viridis',
        contour_levels=25,
        show=False
    )
    plt.savefig('awwe_surrogate.png', dpi=150, bbox_inches='tight')
    print("Saved surrogate plot to 'awwe_surrogate.png'\n")
    
    print("="*70)
    print("Optimization complete! Check the generated PNG files.")
    print("="*70)


if __name__ == "__main__":
    main()
