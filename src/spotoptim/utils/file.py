import os
import datetime


def get_experiment_filename(
    PREFIX: str = None,
    fun_name: str = None,
    dim: int = None,
    fun_evals: int = None,
    path: str = "experiments",
    extension: str = "pkl",
) -> str:
    """
    Generates a standardized filename for experiments.

    Args:
        PREFIX (str): Prefix/identifier for the experiment
        fun_name (str): Name of the objective function
        dim (int): Dimensionality of the problem
        fun_evals (int): Number of function evaluations
        path (str): Directory path to save the file
        extension (str): File extension (default: "pkl")

    Returns:
        str: Absolute path to the experiment file
    """
    if not os.path.exists(path):
        os.makedirs(path)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    parts = []
    if PREFIX:
        parts.append(str(PREFIX))
    if fun_name:
        parts.append(str(fun_name))
    if dim:
        parts.append(f"d_{dim}")
    if fun_evals:
        parts.append(f"n_{fun_evals}")

    parts.append(timestamp)

    filename = "_".join(parts) + f".{extension}"
    return os.path.join(path, filename)
