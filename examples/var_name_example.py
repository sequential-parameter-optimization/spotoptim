"""Example demonstrating the var_name feature in SpotOptim."""

import numpy as np
from spotoptim import SpotOptim
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for this example


def objective_function(X):
    """A simple multi-dimensional objective function.
    
    Args:
        X: Array of shape (n_samples, 3) with columns [temperature, pressure, flow_rate]
    
    Returns:
        Array of objective values
    """
    temperature = X[:, 0]
    pressure = X[:, 1]
    flow_rate = X[:, 2]
    
    # Some artificial objective function
    return (temperature - 20)**2 + (pressure - 100)**2 + (flow_rate - 5)**2


# Example 1: Using default variable names (x0, x1, x2)
print("=" * 60)
print("Example 1: Default variable names")
print("=" * 60)

opt1 = SpotOptim(
    fun=objective_function,
    bounds=[(10, 30), (50, 150), (1, 10)],
    max_iter=5,
    n_initial=10,
    seed=42,
    verbose=True
)

print(f"Variable names: {opt1.var_name}")
result1 = opt1.optimize()
print(f"Best result: {result1.fun:.4f}")
print(f"Best point: {result1.x}")
print()

# Example 2: Using custom variable names
print("=" * 60)
print("Example 2: Custom variable names")
print("=" * 60)

custom_names = ["temperature", "pressure", "flow_rate"]
opt2 = SpotOptim(
    fun=objective_function,
    bounds=[(10, 30), (50, 150), (1, 10)],
    max_iter=5,
    n_initial=10,
    var_name=custom_names,
    seed=42,
    verbose=True
)

print(f"Variable names: {opt2.var_name}")
result2 = opt2.optimize()
print(f"Best result: {result2.fun:.4f}")
print(f"Best point: {result2.x}")
print(f"  {custom_names[0]}: {result2.x[0]:.4f}")
print(f"  {custom_names[1]}: {result2.x[1]:.4f}")
print(f"  {custom_names[2]}: {result2.x[2]:.4f}")
print()

# Example 3: Using var_name with mixed variable types
print("=" * 60)
print("Example 3: Custom names with mixed variable types")
print("=" * 60)

def mixed_objective(X):
    """Objective with mixed variable types."""
    continuous = X[:, 0]  # temperature (float)
    discrete = X[:, 1]    # num_stages (int)
    categorical = X[:, 2]  # algorithm (factor: 0, 1, 2)
    
    return continuous**2 + discrete**2 + categorical**2

opt3 = SpotOptim(
    fun=mixed_objective,
    bounds=[(-5, 5), (1, 10), (0, 2)],
    var_type=["float", "int", "factor"],
    var_name=["temperature", "num_stages", "algorithm"],
    max_iter=5,
    n_initial=10,
    seed=42,
    verbose=True
)

print(f"Variable names: {opt3.var_name}")
print(f"Variable types: {opt3.var_type}")
result3 = opt3.optimize()
print(f"Best result: {result3.fun:.4f}")
print(f"Best point: {result3.x}")
print(f"  {opt3.var_name[0]} ({opt3.var_type[0]}): {result3.x[0]:.4f}")
print(f"  {opt3.var_name[1]} ({opt3.var_type[1]}): {result3.x[1]:.0f}")
print(f"  {opt3.var_name[2]} ({opt3.var_type[2]}): {result3.x[2]:.0f}")
print()

# Example 4: Plotting with custom variable names
print("=" * 60)
print("Example 4: Plotting with custom variable names")
print("=" * 60)

# Create a simple 2D problem for plotting
def sphere_2d(X):
    return np.sum(X**2, axis=1)

opt4 = SpotOptim(
    fun=sphere_2d,
    bounds=[(-5, 5), (-5, 5)],
    var_name=["x_position", "y_position"],
    max_iter=10,
    n_initial=8,
    seed=42
)

result4 = opt4.optimize()
print(f"Best result: {result4.fun:.4f}")
print(f"Best point: {result4.x}")

# Plot using instance var_name (automatically uses x_position, y_position)
print("Creating plot with custom variable names...")
opt4.plot_surrogate(i=0, j=1, show=False, num=50)
print("Plot created successfully (not displayed in this example)")
print()

print("=" * 60)
print("All examples completed successfully!")
print("=" * 60)
