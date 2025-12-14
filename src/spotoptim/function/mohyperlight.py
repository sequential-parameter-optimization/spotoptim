import numpy as np
from typing import Optional
from spotpython.fun.mohyperlight import MoHyperLight as BaseMoHyperLight


class MoHyperLight(BaseMoHyperLight):
    """
    Wrapper for spotpython's MoHyperLight to be compatible with SpotOptim.
    Aggregates fun_control into the class instance so that fun(X) can be called without extra arguments.

    Args:
        fun_control (dict): dictionary containing control parameters for the hyperparameter tuning.
        seed (int): seed for the random number generator. See Numpy Random Sampling.
        log_level (int): log level for the logger.
    """

    def __init__(self, fun_control: dict, seed: int = 126, log_level: int = 50) -> None:
        super().__init__(seed=seed, log_level=log_level)
        self.fun_control = fun_control

    def fun(self, X: np.ndarray, fun_control: Optional[dict] = None) -> np.ndarray:
        """
        Evaluates the function.
        If fun_control is not provided, uses the instance's fun_control.

        Args:
            X (np.ndarray):
                input array of shape (n, k) where n is the number of configurations evaluated
                and k is the number of hyperparameters.
            fun_control (dict, optional):
                dictionary containing control parameters for the hyperparameter tuning.
                If None, uses self.fun_control.

        Returns:
            (np.ndarray):
                (2, n) array where the first row contains the evaluation results (z_res)
                and the second row contains the extracted "epochs" values (epochs_res).
        """
        if fun_control is None:
            fun_control = self.fun_control
        return super().fun(X, fun_control)
