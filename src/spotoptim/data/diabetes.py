import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Callable, Union


class DiabetesDataset(Dataset):
    """
    Diabetes dataset wrapping sklearn's diabetes dataset or custom data.
    """

    def __init__(
        self,
        X: Optional[Union[np.ndarray, torch.Tensor]] = None,
        y: Optional[Union[np.ndarray, torch.Tensor]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        """
        Args:
            X: Features. If None, loads sklearn diabetes dataset.
            y: Targets. If None, loads sklearn diabetes dataset.
            transform: Optional transform to be applied on a sample.
            target_transform: Optional transform to be applied on the target.
        """
        if X is None or y is None:
            diabetes = load_diabetes()
            X = diabetes.data
            y = diabetes.target

        # Convert to tensors if numpy arrays
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()

        # Ensure y is 2D (N, 1)
        if y.ndim == 1:
            y = y.unsqueeze(1)

        self.X = X
        self.y = y
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.X[idx]
        target = self.y[idx]

        if self.transform:
            features = self.transform(features)

        if self.target_transform:
            target = self.target_transform(target)

        return features, target

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def n_samples(self) -> int:
        return len(self.X)


def get_diabetes_dataloaders(
    test_size: float = 0.2,
    batch_size: int = 32,
    scale_features: bool = True,
    shuffle_train: bool = True,
    shuffle_test: bool = False,
    random_state: int = 42,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, Optional[StandardScaler]]:
    """
    Returns train and test dataloaders for the Diabetes dataset.

    Args:
        test_size (float): Fraction of data to use for testing.
        batch_size (int): Batch size.
        scale_features (bool): Whether to standardize features using StandardScaler.
        shuffle_train (bool): Whether to shuffle the training data.
        shuffle_test (bool): Whether to shuffle the test data.
        random_state (int): Random seed for splitting.
        num_workers (int): Number of subprocesses to use for data loading.
        pin_memory (bool): If True, the data loader will copy Tensors into CUDA pinned memory before returning them.

    Returns:
        tuple: (train_loader, test_loader, scaler)
            scaler is the StandardScaler implementation if scale_features=True, else None.
    """
    # Load data
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Create datasets
    train_dataset = DiabetesDataset(X_train, y_train)
    test_dataset = DiabetesDataset(X_test, y_test)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle_test,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader, scaler
