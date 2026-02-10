# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import requests

# --- (user_fun function from the previous section goes here) ---

# Configuration for the server endpoint

# maans12
# SERVER_URL = "http://139.6.66.69:8000/compute/"

# maans04
SERVER_URL = "http://139.6.66.164:8000/compute/"
def user_fun(X: np.ndarray, **kwargs) -> np.ndarray:
    """
    A client-side function that sends a NumPy array to a remote server
    for computation and returns the result as a NumPy array.
    """
    X_list = {"X": X.tolist()}
    response = requests.post(SERVER_URL, json=X_list)
    response.raise_for_status()
    result_data = response.json()
    return np.array(result_data['fx'])

# --- Verification Logic ---

def user_fun_local(X: np.ndarray, **kwargs) -> np.ndarray:
    """The original local implementation for comparison."""
    return np.array([245.04440542, 245.04440542, 245.04440542])

if __name__ == "__main__":
    # Create a sample NumPy array for testing.
    test_X = np.array([[0.48, 0.4, 0.38, 0.5, 0.62, 0.344, 0.4, 0.37, 0.38, 0.2],
                       [0.48, 0.4, 0.38, 0.5, 0.62, 0.344, 0.4, 0.37, 0.38, 0.2],
                       [0.48, 0.4, 0.38, 0.5, 0.62, 0.344, 0.4, 0.37, 0.38, 0.2]])

    print("--- Verification Script ---")
    print(f"Input Data (X):\n{test_X}\n")

    # 1. Compute the expected result using the local function.
    expected_result = user_fun_local(test_X)
    print(f"Expected Result (Local Computation): {expected_result}")

    # 2. Compute the result using the remote API call.
    try:
        remote_result = user_fun(test_X)
        print(f"Actual Result (Remote Computation):  {remote_result}\n")

        # 3. Compare the results.
        # np.allclose is used for safe floating-point comparison.
        if np.allclose(expected_result, remote_result):
            print("✅ SUCCESS: Remote result matches the local result.")
        else:
            print("❌ FAILURE: Remote result does NOT match the local result.")

    except requests.exceptions.RequestException as e:
        print(f"❌ FAILURE: An error occurred while communicating with the server.")
        print(f"   Error: {e}")
        print("   Please ensure the FastAPI server is running at {SERVER_URL}")