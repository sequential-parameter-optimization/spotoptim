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
        """Convert to dictionary for partial backward compatibility or logging."""
        return self.__dict__.copy()

    @property
    def torch_device(self) -> torch.device:
        """Returns the torch.device object."""
        return torch.device(self.device)
