from spotoptim import SpotOptim
import numpy as np

def obj(X):
    return np.sum(X[:, :1]**2, axis=1)

opt = SpotOptim(
    fun=obj,
    bounds=[(-1.0, 1.0), ("A", "B", "C")],
    var_name=["x1", "cat"],
    var_type=["float", "factor"],
    n_initial=3,
    max_iter=3,
    seed=42
)
opt.optimize()

print("--- Current Results Table ---")
opt.print_results(precision=2)

print("\n--- Current Design Table ---")
opt.print_design_table(precision=2)
