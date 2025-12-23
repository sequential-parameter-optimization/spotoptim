from spotoptim import SpotOptim
import numpy as np

def mixed_obj(X):
    # X[:, 0] is float, X[:, 1] is factor
    return np.sum(X[:, :1]**2, axis=1)

print("=== TEST 1: Mixed Types (Float + Factor) with Precision 2 ===")
opt = SpotOptim(
    fun=mixed_obj,
    bounds=[(-1.234567, 1.234567), ("A", "B", "C")],
    var_name=["x1", "cat"],
    var_type=["float", "factor"],
    n_initial=5,
    max_iter=5,
    seed=42
)
opt.optimize()
opt.print_results(precision=2)

print("\n=== TEST 2: Importance with Precision 5 ===")
# Fake importance for testing
opt.get_importance = lambda: [10.12345, 89.98765] 
# Note: Importance is always 2 decimals in my fix logic, but let's see defaults
opt.print_results(precision=5, show_importance=True)

print("\n=== TEST 3: Design Table with Precision 3 ===")
opt.print_design_table(precision=3)

print("\n=== TEST 4: Pure Numeric with Precision 0 ===")
opt2 = SpotOptim(
    fun=lambda X: np.sum(X**2, axis=1),
    bounds=[(-1.55, 1.55)],
    var_name=["x"],
    n_initial=5,
    max_iter=5,
    seed=42
)
opt2.optimize()
opt2.print_results(precision=0)
