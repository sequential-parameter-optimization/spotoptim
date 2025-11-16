"""
Demonstration of SpotOptim's table printing methods.

This script shows how to use:
- print_design_table(): Display search space before optimization
- print_results_table(): Display optimization results with importance
- print_best(): Display best solution found
"""

import numpy as np
from spotoptim import SpotOptim

# Set random seed for reproducibility
np.random.seed(42)


def hyperparameter_objective(X):
    """
    Simulated hyperparameter optimization objective.
    
    X[:, 0]: l1 (neurons) - has strong effect
    X[:, 1]: num_layers - has moderate effect  
    X[:, 2]: log10(lr) - has weak effect
    X[:, 3]: log10(alpha) - has weak effect
    """
    neurons = X[:, 0]
    layers = X[:, 1]
    lr = 10 ** X[:, 2]
    alpha = 10 ** X[:, 3]
    
    # Simulated validation error with noise
    # neurons and layers have stronger effect
    base_error = (neurons - 64)**2 / 1000 + (layers - 2)**2 * 0.1
    lr_effect = (lr - 1.0)**2 * 0.01
    alpha_effect = (alpha - 0.05)**2 * 0.01
    noise = np.random.normal(0, 0.05, size=len(X))
    
    return base_error + lr_effect + alpha_effect + noise


print("=" * 80)
print("SPOTOPTIM TABLE METHODS DEMONSTRATION")
print("=" * 80)

# Define search space
bounds = [
    (16, 128),      # l1: neurons per layer
    (1, 4),         # num_layers: hidden layers
    (-1, 1),        # log10(lr_unified): learning rate
    (-2, 0)         # log10(alpha): physics weight
]

var_type = ["int", "int", "num", "num"]
var_name = ["l1", "num_layers", "log10_lr", "log10_alpha"]

# Create optimizer
print("\n1. Creating SpotOptim instance...")
optimizer = SpotOptim(
    fun=hyperparameter_objective,
    bounds=bounds,
    var_type=var_type,
    var_name=var_name,
    max_iter=30,
    n_initial=10,
    seed=42,
    verbose=False
)
print("   ✓ Optimizer created")

# Display design table BEFORE optimization
print("\n" + "=" * 80)
print("2. SEARCH SPACE DESIGN TABLE (before optimization)")
print("=" * 80)
design_table = optimizer.print_design_table(tablefmt="github")
print(design_table)

# Run optimization
print("\n" + "=" * 80)
print("3. RUNNING OPTIMIZATION")
print("=" * 80)
print(f"   Initial points: 10")
print(f"   Total iterations: 30")
print(f"   Seed: 42")
print()

result = optimizer.optimize()

print(f"\n   ✓ Optimization complete")
print(f"   Total evaluations: {result.nfev}")

# Display results with print_best()
print("\n" + "=" * 80)
print("4. BEST SOLUTION (using print_best method)")
print("=" * 80)

# Define transformations to convert log-scale back
transformations = [
    int,              # l1 -> int
    int,              # num_layers -> int
    lambda x: 10**x,  # log10_lr -> lr_unified
    lambda x: 10**x   # log10_alpha -> alpha
]

optimizer.print_best(result, transformations=transformations)

# Display comprehensive results table
print("\n" + "=" * 80)
print("5. RESULTS TABLE (without importance)")
print("=" * 80)
table = optimizer.print_results_table(tablefmt="github", precision=4)
print(table)

# Display results with importance scores
print("\n" + "=" * 80)
print("6. RESULTS TABLE WITH IMPORTANCE SCORES")
print("=" * 80)
table_with_importance = optimizer.print_results_table(
    tablefmt="github",
    precision=4, 
    show_importance=True
)
print(table_with_importance)

print("\nInterpretation of importance stars:")
print("  ***: Importance > 95% (highly significant)")
print("  **:  Importance > 50% (significant)")
print("  *:   Importance > 1% (moderate)")
print("  .:   Importance > 0.1% (weak)")

# Get raw importance values
print("\n" + "=" * 80)
print("7. RAW IMPORTANCE VALUES")
print("=" * 80)
importance = optimizer.get_importance()
for i, (name, imp) in enumerate(zip(var_name, importance)):
    print(f"   {name:15s}: {imp:6.2f}%")

print("\n" + "=" * 80)
print("Expected: l1 and num_layers should show higher importance")
print("since they have stronger effects in the objective function")
print("=" * 80)

# Try different table formats
print("\n" + "=" * 80)
print("8. ALTERNATIVE TABLE FORMATS")
print("=" * 80)

print("\nGrid format:")
print("-" * 80)
table_grid = optimizer.print_results_table(
    tablefmt="grid", 
    show_importance=True,
    precision=3
)
print(table_grid)

print("\nSimple format:")
print("-" * 80)
table_simple = optimizer.print_results_table(
    tablefmt="simple",
    show_importance=False,
    precision=3
)
print(table_simple)

print("\n" + "=" * 80)
print("DEMONSTRATION COMPLETE")
print("=" * 80)
