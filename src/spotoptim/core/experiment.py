from dataclasses import dataclass, field
from typing import Any, List, Optional, Dict
import torch
from spotoptim.core.data import SpotDataSet


@dataclass
class ExperimentControl:
    """
    Controls the experiment configuration, replacing the legacy fun_control dictionary.

    This class serves as the central configuration object for optimization experiments,
    holding the dataset, model configuration, hyperparameters, and other settings.

    Attributes:
        dataset (SpotDataSet): The dataset object containing the data.
        model_class (Any): The class of the model to be instantiated.
        hyperparameters (Any): The hyperparameters for the model.
        seed (int): The random seed for reproducibility.
        device (str): The device to run the model on (e.g. "cpu" or "cuda").
        num_workers (int): The number of workers to use for data loading.
        epochs (Optional[int]): The number of epochs to train the model.
        batch_size (int): The batch size for training.
        optimizer_class (Optional[Any]): The optimizer class to use for training.
        loss_function (Optional[Any]): The loss function to use for training.
        metrics (List[str]): The metrics to track during training.
        n_initial (int): The number of initial design points.
        max_evals (int): The maximum number of evaluations.
        experiment_name (str): The name of the experiment.
        verbosity (int): The verbosity level.

    Methods:
        to_dict(): Convert the object to a dictionary.
        torch_device(): Return the torch device object.

    Examples:
        >>> import numpy as np
        >>> import torch
        >>> from spotoptim.core.data import SpotDataFromArray
        >>> from spotoptim.core.experiment import ExperimentControl
        >>> from spotoptim.nn.mlp import MLP
        >>>
        >>> # 1. Prepare Data
        >>> X = np.array([[0.1, 0.2], [0.3, 0.4]])
        >>> y = np.array([[1.0], [2.0]])
        >>> dataset = SpotDataFromArray(x_train=X, y_train=y)
        >>>
        >>> # 2. Define Hyperparameters
        >>> params = {"l1": 16, "num_hidden_layers": 1, "lr": 1e-3}
        >>>
        >>> # 3. Initialize Control with Real Model
        >>> exp = ExperimentControl(
        ...     dataset=dataset,
        ...     model_class=MLP,
        ...     hyperparameters=params,
        ...     experiment_name="real_model_run",
        ...     seed=42
        ... )
        >>> print(exp.experiment_name)
        real_model_run
    """

    # Core Components
    dataset: SpotDataSet
    model_class: Any  # The class of the model to be instantiated
    hyperparameters: Any  # Should be ParameterSet type, using Any to avoid circular import issues for now

    # Execution Settings
    seed: int = 123
    device: str = "cpu"
    num_workers: int = 0

    # Model Training Settings
    epochs: Optional[int] = None
    batch_size: int = 32
    optimizer_class: Optional[Any] = None  # Torch optimizer class
    loss_function: Optional[Any] = None  # Torch loss function

    # Metrics
    metrics: List[str] = field(default_factory=list)

    # Optimization Settings (for SpotOptim)
    n_initial: int = 10
    max_evals: int = 50

    # Legacy/Misc
    experiment_name: str = "default_experiment"
    verbosity: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for partial backward compatibility or logging.

        Returns:
            Dict[str, Any]: Dictionary representation of the object.

        Examples:
            >>> import numpy as np
            >>> from spotoptim.core.data import SpotDataFromArray
            >>> from spotoptim.core.experiment import ExperimentControl
            >>>
            >>> # Setup
            >>> dataset = SpotDataFromArray(np.zeros((5,2)), np.zeros((5,1)))
            >>> exp = ExperimentControl(dataset, model_class=None, hyperparameters={})
            >>>
            >>> # Convert to dict
            >>> config = exp.to_dict()
            >>> config['seed']
            123
        """
        return self.__dict__.copy()

    @property
    def torch_device(self) -> torch.device:
        """Returns the torch.device object.

        Returns:
            torch.device: The torch device object.

        Examples:
            >>> import torch
            >>> from spotoptim.core.data import SpotDataFromArray
            >>> from spotoptim.core.experiment import ExperimentControl
            >>>
            >>> dataset = SpotDataFromArray(np.zeros((5,2)), np.zeros((5,1)))
            >>> exp = ExperimentControl(
            ...     dataset,
            ...     model_class=None,
            ...     hyperparameters={},
            ...     device="cpu"
            ... )
            >>>
            >>> exp.torch_device
            device(type='cpu')
        """
        return torch.device(self.device)
