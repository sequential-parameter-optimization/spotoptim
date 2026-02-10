# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from spotoptim.function.so import wingwt

# 1. Pydantic Model Definitions (The Data Contract)

class ObjectiveFunctionInput(BaseModel):
    """Defines the expected structure and data types for the request body.

    FastAPI will use this model to validate incoming requests.

    Attributes:
        X: A list of lists of floats representing the input vectors.

    Examples:
        >>> input_data = ObjectiveFunctionInput(X=[[1.0, 2.0], [3.0, 4.0]])
        >>> print(input_data.X)
        [[1.0, 2.0], [3.0, 4.0]]
    """
    X: List[List[float]]

class ObjectiveFunctionOutput(BaseModel):
    """Defines the structure and data types for the response body.

    FastAPI will use this model to validate and serialize the outgoing response.

    Attributes:
        fx: A list of floats representing the computed objective function values.

    Examples:
        >>> output_data = ObjectiveFunctionOutput(fx=[5.0, 25.0])
        >>> print(output_data.fx)
        [5.0, 25.0]
    """
    fx: List[float]

# 2. FastAPI Application Instance

app = FastAPI(
    title="Objective Function API",
    description="An API to compute f(x) for input vectors.",
    version="1.0.0",
)

# 3. The Objective Function Endpoint

@app.post("/compute/", response_model=ObjectiveFunctionOutput)
async def compute_objective(data: ObjectiveFunctionInput):
    """Computes the objective function for a batch of vectors.

    This endpoint receives a batch of vectors, computes the objective function
    for each, and returns the results.

    Args:
        data: The request body, automatically parsed and validated by FastAPI
              into an instance of the ObjectiveFunctionInput model.

    Returns:
        A dictionary matching the ObjectiveFunctionOutput model, containing
        the computed values for each input vector.

    Examples:
        To test using curl:
        $ curl -X POST "http://localhost:8000/compute/" \\
            -H "Content-Type: application/json" \\
            -d '{"X": [[1.0, 2.0], [3.0, 4.0]]}'
        {"fx": [5.0, 25.0]}
    """
    # Step A: Deserialize the input data from a Python list to a NumPy array.
    # The Pydantic model has already ensured `data.X` is a List[List[float]].
    try:
        X_np = np.array(data.X)
    except Exception as e:
        # This is a fallback, as Pydantic validation should prevent malformed arrays.
        # In a real application, one might return a 400 Bad Request here.
        # For now, we rely on Pydantic's 422 Unprocessable Entity response.
        # This illustrates where conversion happens.
        pass

    # Step B: Perform the core computation using NumPy.
    # This is the user's original function logic.
    # fx_np = np.sum(X_np ** 2, axis=1)
    fx_np = wingwt(X_np)

    # Step C: Serialize the output NumPy array back to a Python list
    # and wrap it in a dictionary that matches the ObjectiveFunctionOutput model.
    # FastAPI will then convert this dictionary to a JSON response.
    return {"fx": fx_np.tolist()}
