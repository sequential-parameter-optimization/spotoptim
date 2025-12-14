import numpy as np
import requests
import warnings
from spotoptim.function.remote import objective_remote

def verify_remote_function():
    print("--- Verifying objective_remote ---")
    
    # Test Data
    test_X = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [0.1, 0.2, 0.3]
    ])
    
    print(f"Input X:\n{test_X}")
    
    # Expected Result (sum of squares)
    expected = np.sum(test_X ** 2, axis=1)
    
    try:
        # Call the new function
        print("\nCalling objective_remote...")
        result = objective_remote(test_X)
        
        print(f"Result: {result}")
        print(f"Expected: {expected}")
        
        if np.allclose(result, expected):
            print("\n✅ SUCCESS: Remote result matches expected values.")
        else:
            print("\n❌ FAILURE: Remote result does NOT match.")
            raise AssertionError("Remote result does not match expected values.")
            
    except requests.exceptions.RequestException as e:
        print(f"\n⚠️ WARNING: Could not connect to remote server ({e}). Verification SKIPPED.")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise e

if __name__ == "__main__":
    verify_remote_function()
