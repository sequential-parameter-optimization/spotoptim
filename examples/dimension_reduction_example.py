"""Example demonstrating dimension reduction in SpotOptim.

This example shows how SpotOptim automatically handles fixed dimensions
(where lower == upper bounds) during optimization.
"""

import numpy as np
from spotoptim import SpotOptim
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


def objective_function(X):
    """Multi-dimensional objective function.
    
    f(x0, x1, x2, x3) = (x0 - 2)^2 + (x1 - 3)^2 + (x2 - 1)^2 + (x3 - 4)^2
    
    Args:
        X: Array of shape (n_samples, 4)
    
    Returns:
        Array of objective values
    """
    return np.sum((X - np.array([2, 3, 1, 4]))**2, axis=1)


print("=" * 80)
print("Dimension Reduction Feature in SpotOptim")
print("=" * 80)
print()

# Example 1: Optimization with one fixed dimension
print("-" * 80)
print("Example 1: Optimization with one fixed dimension")
print("-" * 80)

# Set up problem: x1 is fixed at 3.0, others vary
bounds_example1 = [
    (0, 5),      # x0: varying
    (3, 3),      # x1: FIXED at 3.0
    (-2, 4),     # x2: varying
    (2, 6)       # x3: varying
]

var_names_example1 = ["x0", "x1_fixed", "x2", "x3"]

opt1 = SpotOptim(
    fun=objective_function,
    bounds=bounds_example1,
    var_name=var_names_example1,
    max_iter=15,
    n_initial=10,
    seed=42,
    verbose=False
)

print(f"Original dimensions: {len(opt1.all_lower)}")
print(f"Reduced dimensions: {opt1.n_dim}")
print(f"Dimension reduction active: {opt1.red_dim}")
print(f"Fixed dimensions: {np.where(opt1.ident)[0].tolist()}")
print(f"Varying dimensions: {np.where(~opt1.ident)[0].tolist()}")
print(f"Variable names (reduced): {opt1.var_name}")
print()

result1 = opt1.optimize()

print(f"Optimization complete!")
print(f"Best objective value: {result1.fun:.6f}")
print(f"Best solution (full dimensions):")
for i, (name, value) in enumerate(zip(var_names_example1, result1.x)):
    status = " (FIXED)" if opt1.ident[i] else ""
    print(f"  {name}: {value:.6f}{status}")
print(f"Expected optimum: [2.0, 3.0, 1.0, 4.0]")
print()

# Example 2: Optimization with multiple fixed dimensions
print("-" * 80)
print("Example 2: Optimization with multiple fixed dimensions")
print("-" * 80)

# Set up problem: x0 and x2 are fixed, x1 and x3 vary
bounds_example2 = [
    (2, 2),      # x0: FIXED at 2.0
    (0, 5),      # x1: varying
    (1, 1),      # x2: FIXED at 1.0
    (2, 6)       # x3: varying
]

var_names_example2 = ["x0_fixed", "x1", "x2_fixed", "x3"]

opt2 = SpotOptim(
    fun=objective_function,
    bounds=bounds_example2,
    var_name=var_names_example2,
    max_iter=15,
    n_initial=10,
    seed=42,
    verbose=False
)

print(f"Original dimensions: {len(opt2.all_lower)}")
print(f"Reduced dimensions: {opt2.n_dim}")
print(f"Fixed dimensions: {np.where(opt2.ident)[0].tolist()}")
print(f"Varying dimensions: {np.where(~opt2.ident)[0].tolist()}")
print()

result2 = opt2.optimize()

print(f"Optimization complete!")
print(f"Best objective value: {result2.fun:.6f}")
print(f"Best solution (full dimensions):")
for i, (name, value) in enumerate(zip(var_names_example2, result2.x)):
    status = " (FIXED)" if opt2.ident[i] else ""
    print(f"  {name}: {value:.6f}{status}")
print()

# Example 3: Practical application - parameter tuning with constraints
print("-" * 80)
print("Example 3: Practical application with physical constraints")
print("-" * 80)

def chemical_process(X):
    """Simulate a chemical process with some fixed parameters.
    
    Parameters:
        X[:, 0]: temperature (varying)
        X[:, 1]: pressure (fixed at safety limit)
        X[:, 2]: flow_rate (varying)
        X[:, 3]: catalyst_amount (fixed by regulation)
    """
    temp = X[:, 0]
    pressure = X[:, 1]
    flow_rate = X[:, 2]
    catalyst = X[:, 3]
    
    # Simulate yield as function of parameters
    # Optimal: temp=300, pressure=100, flow=5, catalyst=2
    yield_temp = -(temp - 300)**2 / 100
    yield_pressure = -(pressure - 100)**2 / 50
    yield_flow = -(flow_rate - 5)**2 / 10
    yield_catalyst = -(catalyst - 2)**2 / 5
    
    return -(yield_temp + yield_pressure + yield_flow + yield_catalyst)  # Minimize negative yield

# Physical constraints: pressure at safety limit, catalyst fixed by regulation
bounds_process = [
    (250, 350),    # temperature: varying
    (100, 100),    # pressure: FIXED at safety limit
    (1, 10),       # flow_rate: varying
    (2, 2)         # catalyst_amount: FIXED by regulation
]

var_names_process = ["temperature", "pressure_safety", "flow_rate", "catalyst_regulation"]

opt_process = SpotOptim(
    fun=chemical_process,
    bounds=bounds_process,
    var_name=var_names_process,
    max_iter=20,
    n_initial=15,
    seed=42,
    verbose=False
)

print(f"Process parameters: {len(bounds_process)}")
print(f"Optimizable parameters: {opt_process.n_dim}")
print(f"Fixed parameters (constraints):")
for i, name in enumerate(var_names_process):
    if opt_process.ident[i]:
        print(f"  {name}: {opt_process.all_lower[i]} (constraint)")
print()

result_process = opt_process.optimize()

print(f"Optimization complete!")
print(f"Best yield (negative cost): {-result_process.fun:.6f}")
print(f"Optimal parameters:")
for i, (name, value) in enumerate(zip(var_names_process, result_process.x)):
    if opt_process.ident[i]:
        print(f"  {name}: {value:.1f} (CONSTRAINED)")
    else:
        print(f"  {name}: {value:.2f}")
print()

# Example 4: Verify that results are in full dimensions
print("-" * 80)
print("Example 4: Verifying full-dimensional results")
print("-" * 80)

opt_verify = SpotOptim(
    fun=lambda X: np.sum(X**2, axis=1),
    bounds=[(1, 1), (-5, 5), (2, 2), (-3, 3)],
    max_iter=5,
    n_initial=5,
    seed=42,
    verbose=False
)

result_verify = opt_verify.optimize()

print(f"Problem specification: 4 dimensions (2 fixed, 2 varying)")
print(f"Internal optimization: {opt_verify.n_dim} dimensions")
print(f"Result dimensions: {result_verify.x.shape[0]} dimensions")
print(f"Result X array shape: {result_verify.X.shape}")
print()
print(f"Fixed values in results:")
print(f"  Dimension 0 (fixed at 1): {result_verify.X[:, 0].mean():.1f} (all equal)")
print(f"  Dimension 2 (fixed at 2): {result_verify.X[:, 2].mean():.1f} (all equal)")
print()
print(f"Varying dimensions:")
print(f"  Dimension 1 (varying): {result_verify.X[:, 1].min():.2f} to {result_verify.X[:, 1].max():.2f}")
print(f"  Dimension 3 (varying): {result_verify.X[:, 3].min():.2f} to {result_verify.X[:, 3].max():.2f}")
print()

print("=" * 80)
print("All examples completed successfully!")
print("=" * 80)
print()
print("Key takeaways:")
print("1. SpotOptim automatically detects fixed dimensions (lower == upper)")
print("2. Optimization runs in reduced space for efficiency")
print("3. Results are automatically expanded to full dimensions")
print("4. Variable names and types are correctly handled")
print("5. Users work with full-dimensional data transparently")
