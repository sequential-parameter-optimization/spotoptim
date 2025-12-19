import numpy as np
import requests


# Default configuration for the server endpoint
DEFAULT_SERVER_URL = "http://139.6.66.164:8000/compute/"


def objective_remote(
    X: np.ndarray, url: str = DEFAULT_SERVER_URL, **kwargs
) -> np.ndarray:
    """
    Evaluates an objective function remotely via an HTTP POST request.

    Args:
        X (np.ndarray): Input data of shape (n_samples, n_features).
        url (str, optional): The URL of the remote computation server.
            Defaults to "http://139.6.66.164:8000/compute/".
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
    response = requests.post(url, json=payload)
    response.raise_for_status()

    # Parse the response
    result_data = response.json()

    # Assuming the server returns a dictionary with 'fx' key containing the results
    # Adjust this key if the server API differs, but based on client.py this is correct.
    if "fx" not in result_data:
        raise ValueError(f"Server response missing 'fx' key. Response: {result_data}")

    return np.array(result_data["fx"])
