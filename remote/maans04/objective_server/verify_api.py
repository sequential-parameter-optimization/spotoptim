# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import requests
import json
import math

url = "http://localhost:8000/compute/"
headers = {"Content-Type": "application/json"}

# 10-dimensional input
X = [[1.0] * 10, [2.0] * 10]

def test_success():
    payload = {"X": X}
    response = requests.post(url, json=payload)
    print(f"Success Test Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Success Test Output: {data}")
        # Check if values are not None and not NaN (JSON doesn't have NaN for floats usually unless allowed, but python requests .json() handles standard JSON)
        # Standard JSON uses null for NaN sometimes, or just fails.
        # But here we expect valid numbers.
        for val in data['fx']:
            if val is None or (isinstance(val, float) and math.isnan(val)):
                print("FAILURE: Expected numbers, got NaN/None")
                return False
        return True
    else:
        print(f"FAILURE: Status {response.status_code}")
        print(response.text)
        return False

def test_error_prob():
    payload = {"X": X, "err_probability": 1.0}
    response = requests.post(url, json=payload)
    print(f"Error Prob Test Status: {response.status_code}")
    if response.status_code == 200:
        # FastAPI/Pydantic default JSON encoder might encode NaN as NaN (invalid JSON) or null
        # requests.json() might choke on NaN if strict.
        # Let's see raw text first
        print(f"Error Prob Raw Text: {response.text}")
        try:
            data = response.json()
        except Exception:
            # If it returns NaN string, requests might fail if not configured to allow nan
            # Actually standard json decoder allows NaN
            pass
            
        # We expect NaNs. In JSON, often represented as null or NaN.
        # If the output string is `NaN`, python json parser accepts it.
        data = response.json()
        print(f"Error Prob Output: {data}")
        
        for val in data['fx']:
            # We expect NaN (parsed as nan float) or None (parsed from null)
            # Since we used np.nan and standard json encoder, it usually produces NaN
            if not (val is None or (isinstance(val, float) and math.isnan(val))):
                 print(f"FAILURE: Expected NaN/None, got {val}")
                 return False
        return True
    else:
        print(f"FAILURE: Status {response.status_code}")
        print(response.text)
        return False

if __name__ == "__main__":
    s1 = test_success()
    s2 = test_error_prob()
    if s1 and s2:
        print("VERIFICATION SUCCESSFUL")
    else:
        print("VERIFICATION FAILED")
