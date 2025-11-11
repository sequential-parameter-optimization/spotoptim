"""
Quick example showing tensorboard_clean in action.
"""

from spotoptim import SpotOptim
import numpy as np

def sphere(X):
    return np.sum(X**2, axis=1)

# Example: Start fresh with clean logs
print("Creating optimizer with tensorboard_clean=True...")
print("This will remove any old TensorBoard logs before starting.\n")

optimizer = SpotOptim(
    fun=sphere,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=20,
    n_initial=10,
    tensorboard_log=True,      # Enable TensorBoard logging
    tensorboard_clean=True,    # Remove old logs first
    verbose=True,
    seed=42
)

print("\nRunning optimization...")
result = optimizer.optimize()

print(f"\nOptimization complete!")
print(f"Best value found: {result.fun:.6f}")
print(f"Best point: {result.x}")
print(f"\nTensorBoard logs saved to: {optimizer.tensorboard_path}")
print("\nView results with: tensorboard --logdir=runs")
