
import numpy as np
from spotoptim import SpotOptim

def bad_fun(X):
    # Returns object array mixed with strings
    return np.array([1.0, "error"], dtype=object)

try:
    opt = SpotOptim(fun=bad_fun, bounds=[(-5, 5)], n_initial=2, max_iter=3)
    # We need to manually set X and evaluate to bypass initial design check which IS safe
    # Actually, let's just run optimize(). 
    # The initial design check handles "error" string by converting to NaN in _rm_NA_values.
    # So initial design will be filtered.
    # But later iterations might fail if they return bad values?
    # No, optimize() calls _RM_NA_values for initial design.
    
    # Wait, the user error is in "run 0", which might mean the first run of some external loop?
    # OR it implies initial design failed? 
    # Ah, the user changed _rm_NA_values recently.
    
    # Let's try to simulate checking NA values in _handle_NA_new_points
    # We can call the internal method directly.
    
    y_bad = np.array([1.0, "error"], dtype=object)
    
    print("Calling _handle_NA_new_points with bad input...")
    opt._handle_NA_new_points(np.zeros((2,1)), y_bad)
    print("Success!")
    
except Exception as e:
    print(f"Caught expected error: {e}")
