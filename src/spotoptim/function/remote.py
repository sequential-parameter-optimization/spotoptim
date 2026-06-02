# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import requests

# Default configuration for the server endpoint
DEFAULT_SERVER_URL = "http://139.6.66.164:8000/compute/"
# Default (connect, read) timeout in seconds. A bounded connect timeout keeps
# the call from hanging indefinitely when the server is unreachable — e.g. in
# CI, where egress to the server may be silently dropped rather than refused.
DEFAULT_TIMEOUT: tuple[float, float] = (10.0, 120.0)


def objective_remote(
    X: np.ndarray,
    url: str = DEFAULT_SERVER_URL,
    timeout: float | tuple[float, float] = DEFAULT_TIMEOUT,
    **kwargs,
) -> np.ndarray:
    """
    Evaluates an objective function remotely via an HTTP POST request.

    Args:
        X (np.ndarray): Input data of shape (n_samples, n_features).
        url (str, optional): The URL of the remote computation server.
            Defaults to "http://139.6.66.164:8000/compute/".
        timeout (float | tuple[float, float], optional): Request timeout in
            seconds as a single value or a ``(connect, read)`` tuple. Defaults
            to ``(10, 120)``; the bounded connect time prevents the call from
            hanging when the server is unreachable.
        **kwargs (Any): Additional arguments to include in the request payload (optional).

    Returns:
        np.ndarray: The computed objective values of shape (n_samples,).

    Raises:
        requests.exceptions.RequestException: If the remote request fails.

    Examples:
        >>> import numpy as np
        >>> from spotoptim.function.remote import objective_remote
        >>> X = np.array([[1, 2], [3, 4]])
        >>> y = objective_remote(X)
        >>> print(y)
    """
    # Prepare the payload
    # X needs to be converted to a list for JSON serialization
    payload = {"X": np.asarray(X).tolist()}

    # Merge any additional kwargs into the payload
    if kwargs:
        payload.update(kwargs)

    # Perform the request
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()

    # Parse the response
    result_data = response.json()

    # Assuming the server returns a dictionary with 'fx' key containing the results
    # Adjust this key if the server API differs, but based on client.py this is correct.
    if "fx" not in result_data:
        raise ValueError(f"Server response missing 'fx' key. Response: {result_data}")

    return np.array(result_data["fx"])
