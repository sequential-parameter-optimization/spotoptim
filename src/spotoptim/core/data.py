from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional, Union
import numpy as np
import torch
from torch.utils.data import Dataset


class SpotDataSet(ABC):
    """
    Abstract base class for data handling in SpotOptim.

    Attributes:
        input_dim (int): Number of input features.
        output_dim (int): Number of output features.
        target_column (str, optional): Name of the target column if applicable.
    """

    def __init__(
        self, input_dim: int, output_dim: int, target_column: Optional[str] = None
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.target_column = target_column

    @abstractmethod
    def get_train_data(self) -> Any:
        """Returns the training data."""
        pass

    @abstractmethod
    def get_test_data(self) -> Any:
        """Returns the test data."""
        pass

    @abstractmethod
    def get_validation_data(self) -> Any:
        """Returns the validation data."""
        pass


class SpotDataFromArray(SpotDataSet):
    """
    Data handler for numpy arrays or torch tensors.
    """

    def __init__(
        self,
        x_train: Union[np.ndarray, torch.Tensor],
        y_train: Union[np.ndarray, torch.Tensor],
        x_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
        y_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
        x_test: Optional[Union[np.ndarray, torch.Tensor]] = None,
        y_test: Optional[Union[np.ndarray, torch.Tensor]] = None,
        target_column: Optional[str] = None,
    ):
        # Determine dimensions
        input_dim = x_train.shape[1] if hasattr(x_train, "shape") else 0
        output_dim = (
            y_train.shape[1]
            if hasattr(y_train, "shape") and len(y_train.shape) > 1
            else 1
        )

        super().__init__(input_dim, output_dim, target_column)

        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test

    def get_train_data(
        self,
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        return self.x_train, self.y_train

    def get_validation_data(
        self,
    ) -> Optional[
        Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]
    ]:
        if self.x_val is not None:
            return self.x_val, self.y_val
        return None

    def get_test_data(
        self,
    ) -> Optional[
        Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]
    ]:
        if self.x_test is not None:
            return self.x_test, self.y_test
        return None


class SpotDataFromTorchDataset(SpotDataSet):
    """
    Data handler for PyTorch Datasets.
    """

    def __init__(
        self,
        train_dataset: Dataset,
        input_dim: int,
        output_dim: int,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        target_column: Optional[str] = None,
    ):
        super().__init__(input_dim, output_dim, target_column)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def get_train_data(self) -> Dataset:
        return self.train_dataset

    def get_validation_data(self) -> Optional[Dataset]:
        return self.val_dataset

    def get_test_data(self) -> Optional[Dataset]:
        return self.test_dataset
