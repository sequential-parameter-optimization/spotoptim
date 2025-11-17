"""Diabetes dataset for regression tasks.

Provides PyTorch Dataset and DataLoader utilities for the sklearn diabetes dataset.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DiabetesDataset(Dataset):
    """PyTorch Dataset for the diabetes dataset from sklearn.

    The diabetes dataset contains 10 baseline variables (age, sex, body mass index,
    average blood pressure, and six blood serum measurements) for 442 diabetes
    patients. The target is a quantitative measure of disease progression one year
    after baseline.

    This dataset is useful for testing regression algorithms.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Target values of shape (n_samples,) or (n_samples, 1).
        transform (callable, optional): Optional transform to be applied to features.
        target_transform (callable, optional): Optional transform to be applied to targets.

    Attributes:
        X (torch.Tensor): Feature tensor of shape (n_samples, n_features).
        y (torch.Tensor): Target tensor of shape (n_samples, 1).
        n_features (int): Number of features (10 for diabetes dataset).
        n_samples (int): Number of samples in the dataset.

    Examples:
        Basic usage:

        >>> from spotoptim.data import DiabetesDataset
        >>> from sklearn.datasets import load_diabetes
        >>> import numpy as np
        >>>
        >>> # Load data
        >>> diabetes = load_diabetes()
        >>> X, y = diabetes.data, diabetes.target.reshape(-1, 1)
        >>>
        >>> # Create dataset
        >>> dataset = DiabetesDataset(X, y)
        >>> print(f"Dataset size: {len(dataset)}")
        >>> print(f"Features shape: {dataset.X.shape}")
        >>> print(f"Targets shape: {dataset.y.shape}")
        >>>
        >>> # Get a sample
        >>> features, target = dataset[0]
        >>> print(f"Sample features: {features.shape}")
        >>> print(f"Sample target: {target.shape}")

        With data splitting and scaling:

        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.preprocessing import StandardScaler
        >>>
        >>> # Load and split data
        >>> diabetes = load_diabetes()
        >>> X, y = diabetes.data, diabetes.target.reshape(-1, 1)
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X, y, test_size=0.2, random_state=42
        ... )
        >>>
        >>> # Scale features
        >>> scaler = StandardScaler()
        >>> X_train = scaler.fit_transform(X_train)
        >>> X_test = scaler.transform(X_test)
        >>>
        >>> # Create datasets
        >>> train_dataset = DiabetesDataset(X_train, y_train)
        >>> test_dataset = DiabetesDataset(X_test, y_test)
    """

    def __init__(self, X, y, transform=None, target_transform=None):
        """Initialize the DiabetesDataset.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target values.
            transform (callable, optional): Transform for features.
            target_transform (callable, optional): Transform for targets.
        """
        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.transform = transform
        self.target_transform = target_transform
        self.n_features = self.X.shape[1]
        self.n_samples = self.X.shape[0]

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple:
        """Get a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (features, target) where features is a tensor of shape (n_features,)
                   and target is a tensor of shape (1,).
        """
        features = self.X[idx]
        target = self.y[idx]

        if self.transform:
            features = self.transform(features)
        if self.target_transform:
            target = self.target_transform(target)

        return features, target


def get_diabetes_dataloaders(
    test_size: float = 0.2,
    batch_size: int = 32,
    shuffle_train: bool = True,
    shuffle_test: bool = False,
    random_state: int = 42,
    scale_features: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple:
    """Get train and test DataLoaders for the diabetes dataset.

    Convenience function that loads the diabetes dataset, splits it into train/test,
    optionally scales features, creates Dataset objects, and returns DataLoaders.

    Args:
        test_size (float, optional): Proportion of dataset to include in test split.
            Defaults to 0.2 (20%).
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        shuffle_train (bool, optional): Whether to shuffle training data. Defaults to True.
        shuffle_test (bool, optional): Whether to shuffle test data. Defaults to False.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.
        scale_features (bool, optional): Whether to standardize features using
            StandardScaler. Defaults to True.
        num_workers (int, optional): Number of worker processes for data loading.
            Defaults to 0 (load in main process).
        pin_memory (bool, optional): If True, DataLoader will copy tensors into
            CUDA pinned memory before returning them. Useful when using GPU.
            Defaults to False.

    Returns:
        tuple: (train_loader, test_loader, scaler) where:
            - train_loader (DataLoader): DataLoader for training data
            - test_loader (DataLoader): DataLoader for test data
            - scaler (StandardScaler or None): Fitted scaler if scale_features=True,
              otherwise None

    Examples:
        Basic usage:

        >>> from spotoptim.data import get_diabetes_dataloaders
        >>>
        >>> # Get dataloaders with default settings
        >>> train_loader, test_loader, scaler = get_diabetes_dataloaders()
        >>>
        >>> print(f"Training batches: {len(train_loader)}")
        >>> print(f"Test batches: {len(test_loader)}")
        >>>
        >>> # Iterate over batches
        >>> for batch_X, batch_y in train_loader:
        ...     print(f"Batch features shape: {batch_X.shape}")
        ...     print(f"Batch targets shape: {batch_y.shape}")
        ...     break

        Custom configuration:

        >>> # Larger batches, no scaling
        >>> train_loader, test_loader, scaler = get_diabetes_dataloaders(
        ...     batch_size=64,
        ...     scale_features=False,
        ...     random_state=123
        ... )

        Complete training example:

        >>> from spotoptim.data import get_diabetes_dataloaders
        >>> from spotoptim.nn.linear_regressor import LinearRegressor
        >>> import torch.nn as nn
        >>>
        >>> # Get data
        >>> train_loader, test_loader, scaler = get_diabetes_dataloaders(
        ...     batch_size=32,
        ...     random_state=42
        ... )
        >>>
        >>> # Create model
        >>> model = LinearRegressor(
        ...     input_dim=10,
        ...     output_dim=1,
        ...     l1=32,
        ...     num_hidden_layers=2,
        ...     activation="ReLU"
        ... )
        >>>
        >>> # Get optimizer and loss
        >>> optimizer = model.get_optimizer("Adam", lr=0.01)
        >>> criterion = nn.MSELoss()
        >>>
        >>> # Training loop
        >>> for epoch in range(100):
        ...     model.train()
        ...     for batch_X, batch_y in train_loader:
        ...         optimizer.zero_grad()
        ...         predictions = model(batch_X)
        ...         loss = criterion(predictions, batch_y)
        ...         loss.backward()
        ...         optimizer.step()
        ...
        ...     # Validation
        ...     if (epoch + 1) % 20 == 0:
        ...         model.eval()
        ...         val_loss = 0.0
        ...         with torch.no_grad():
        ...             for batch_X, batch_y in test_loader:
        ...                 predictions = model(batch_X)
        ...                 val_loss += criterion(predictions, batch_y).item()
        ...         val_loss /= len(test_loader)
        ...         print(f'Epoch [{epoch+1}/100], Val Loss: {val_loss:.4f}')

    Note:
        - Features are automatically converted to float32 tensors
        - Targets are reshaped to (n_samples, 1) for compatibility with PyTorch
        - The scaler is fitted only on training data to prevent data leakage
        - Set num_workers > 0 for parallel data loading (may speed up training)
    """
    # Load diabetes dataset
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Optionally scale features
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Ensure targets are 2D
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

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
