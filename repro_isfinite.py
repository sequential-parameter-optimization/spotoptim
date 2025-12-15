
import numpy as np

def test_isfinite_with_none():
    y_raw = [1.0, None, 3.0]
    y = np.array(y_raw) # inferred as object
    print(f"y dtype: {y.dtype}")
    
    try:
        mask = ~np.isfinite(y)
        print("isfinite worked")
    except TypeError as e:
        print(f"isfinite failed: {e}")

    # fix attempt
    try:
        y_float = np.array(y, dtype=float)
        print(f"y_float dtype: {y_float.dtype}")
        print(f"y_float: {y_float}")
        mask = ~np.isfinite(y_float)
        print("isfinite worked on float cast")
    except Exception as e:
        print(f"Conversion failed: {e}")

if __name__ == "__main__":
    test_isfinite_with_none()
