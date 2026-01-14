"""
MLP Surrogate model for SpotOptim.

This module implements a standard Multi-Layer Perceptron (MLP) surrogate
that enables uncertainty estimation via Monte Carlo Dropout (MC Dropout).
It is designed to be a drop-in alternative to the Kriging surrogate
within the SpotOptim framework.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler

from spotoptim.nn.mlp import MLP


class MLPSurrogate(BaseEstimator, RegressorMixin):
    """
    A scikit-learn compatible MLP surrogate model with uncertainty estimation.

    This class wraps a PyTorch MLP (from spotoptim.nn.mlp) and provides
    the standard fit/predict interface required by SpotOptim.
    It uses Monte Carlo (MC) Dropout during prediction to estimate
    uncertainty (standard deviation) which is crucial for acquisition functions.

    Compatible with SpotOptim's variable type conventions:
    - 'float': continuous numeric variables
    - 'int': integer variables
    - 'factor': categorical/unordered variables

    Note: All input variables are currently treated as numeric and standardized.
    For best performance with categorical variables, this simple treatment
    may be suboptimal compared to embedding processing, but ensures compatibility.

    Args:
        in_channels (int, optional): Input dimension. If None, inferred during fit.
        hidden_channels (List[int], optional): Explicit list of hidden layer sizes.
        l1 (int, optional): Neurons per hidden layer (used if hidden_channels is None).
            Defaults to 64.
        num_hidden_layers (int, optional): Number of hidden layers (used if hidden_channels is None).
            Defaults to 2.
        dropout (float, optional): Dropout probability. Crucial for uncertainty estimation.
            Defaults to 0.1.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        activation (str, optional): Activation function. Defaults to "relu".
        epochs (int, optional): Number of training epochs. Defaults to 200.
        batch_size (int, optional): Training batch size. Defaults to 32.
        optimizer_name (str, optional): Name of PyTorch optimizer ("Adam", "SGD", etc.) or
            "AdamWScheduleFree". Defaults to "AdamWScheduleFree".
        mc_dropout_passes (int, optional): Number of forward passes for MC Dropout
            uncertainty estimation. Defaults to 30.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        var_type (List[str], optional): Variable types for each dimension.
            Defaults to None.
        name (str, optional): Name of the surrogate. Defaults to "MLPSurrogate".
        verbose (bool, optional): Whether to print training progress. Defaults to False.
    """

    def __init__(
        self,
        in_channels: int = None,
        hidden_channels: List[int] = None,
        l1: int = 128,
        num_hidden_layers: int = 3,
        dropout: float = 0.0,
        activation: str = "relu",
        lr: float = 1e-3,
        epochs: int = 200,
        batch_size: int = 32,
        optimizer_name: str = "AdamWScheduleFree",
        mc_dropout_passes: int = 30,
        seed: int = 42,
        var_type: Optional[List[str]] = None,
        name: str = "MLPSurrogate",
        verbose: bool = False,
    ):
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.l1 = l1
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.activation = activation
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer_name = optimizer_name
        self.mc_dropout_passes = mc_dropout_passes
        self.seed = seed
        self.var_type = var_type
        self.name = name
        self.verbose = verbose

        # State attributes
        self.model_ = None
        self.scaler_x_ = None
        self.scaler_y_ = None
        self.X_ = None
        self.y_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPSurrogate":
        """
        Fit the MLP model to the training data.

        Args:
            X (np.ndarray): Training inputs, shape (n_samples, n_features).
            y (np.ndarray): Training targets, shape (n_samples,).

        Returns:
            MLPSurrogate: The fitted model.
        """
        # Set seeds for reproducibility
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.X_ = X
        self.y_ = y.flatten()

        # Handle var_type (store for compatibility, though currently we treat all as float)
        if self.var_type is None:
            self.var_type = ["float"] * X.shape[1]

        # Scaling
        self.scaler_x_ = StandardScaler().fit(X)
        self.scaler_y_ = StandardScaler().fit(y)

        X_scaled = self.scaler_x_.transform(X)
        y_scaled = self.scaler_y_.transform(y)

        # Dataset
        dataset = TensorDataset(torch.tensor(X_scaled), torch.tensor(y_scaled))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize MLP
        input_dim = X.shape[1]
        output_dim = y.shape[1]

        # Resolve activation function
        activation_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "leakyrelu": nn.LeakyReLU,
            "elu": nn.ELU,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
        }
        activation_key = self.activation.lower()
        if activation_key not in activation_map:
            raise ValueError(
                f"Unsupported activation '{self.activation}'. "
                f"Validation options: {list(activation_map.keys())}"
            )
        activation_layer = activation_map[activation_key]

        # If hidden_channels provided, use it, otherwise use l1/num_hidden_layers
        self._current_hidden_channels = self.hidden_channels

        self.model_ = MLP(
            in_channels=input_dim,
            hidden_channels=self._current_hidden_channels,
            l1=self.l1,
            num_hidden_layers=self.num_hidden_layers,
            output_dim=output_dim,
            activation_layer=activation_layer,
            dropout=self.dropout,
            lr=self.lr,
        )

        optimizer = self.model_.get_optimizer(self.optimizer_name)
        criterion = nn.MSELoss()

        # Training loop
        self.model_.train()

        # Schedule-Free Optimizer requires specific train/eval mode switching
        if hasattr(optimizer, "train"):
            optimizer.train()

        for epoch in range(self.epochs):
            total_loss = 0.0
            num_batches = 0
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                pred = self.model_(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1

            if self.verbose and (epoch + 1) % (self.epochs // 10 or 1) == 0:
                print(
                    f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/num_batches:.6f}"
                )

        # Schedule-Free Optimizer finalization
        if hasattr(optimizer, "eval"):
            optimizer.eval()

        return self

    def predict(
        self, X: np.ndarray, return_std: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict targets for X.

        Args:
            X (np.ndarray): Input data, shape (n_samples, n_features).
            return_std (bool, optional): Whether to return the standard deviation
                (uncertainty) along with the mean prediction.
                Defaults to False.

        Returns:
            np.ndarray: Predicted mean values, shape (n_samples,).
            tuple: (mean, std) if return_std is True.
        """
        if self.model_ is None:
            raise RuntimeError("Model is not fitted yet. Call 'fit' first.")

        X = np.asarray(X, dtype=np.float32)
        X_scaled = self.scaler_x_.transform(X)
        X_tensor = torch.tensor(X_scaled)

        if return_std:
            # MC Dropout Prediction
            self.model_.train()  # Enable dropout
            preds_scaled = []

            with torch.no_grad():
                for _ in range(self.mc_dropout_passes):
                    out = self.model_(X_tensor)
                    preds_scaled.append(out.numpy())

            # Stack to shape (passes, n_samples, out_dim)
            preds_scaled = np.stack(preds_scaled)

            # Compute mean and std in scaled space
            mean_scaled = np.mean(preds_scaled, axis=0)
            std_scaled = np.std(preds_scaled, axis=0)

            # Inverse transform mean
            mean = self.scaler_y_.inverse_transform(mean_scaled)

            # Transform std: std_real = std_scaled * scale_
            std = std_scaled * self.scaler_y_.scale_

            # Flatten to match Kriging interface (n_samples,)
            if mean.shape[1] == 1:
                mean = mean.flatten()
                std = std.flatten()

            return mean, std

        else:
            # Single Deterministic Prediction
            self.model_.eval()  # Disable dropout
            with torch.no_grad():
                pred_scaled = self.model_(X_tensor).numpy()

            pred = self.scaler_y_.inverse_transform(pred_scaled)

            if pred.shape[1] == 1:
                pred = pred.flatten()

            return pred
