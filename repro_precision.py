from spotoptim import SpotOptim
import numpy as np

def mixed_obj(X):
    # X[:, 0] is float, X[:, 1] is factor (0, 1, 2)
    return np.sum(X[:, :1]**2, axis=1)

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

print("\n--- Results: Precision 2 (Mixed) ---")
opt.print_results(precision=2)
