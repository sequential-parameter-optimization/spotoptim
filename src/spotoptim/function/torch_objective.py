import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Optional, Dict, Any
from spotoptim.core.experiment import ExperimentControl
from spotoptim.core.data import SpotDataFromArray, SpotDataFromTorchDataset
import logging
import inspect

logger = logging.getLogger(__name__)


class TorchObjective:
    """
    A callable objective function for SpotOptim that trains and evaluates a PyTorch model.
    """

    def __init__(self, experiment: ExperimentControl):
        self.experiment = experiment
        self.device = experiment.torch_device

    def _get_hyperparameters(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Converts the input vector X into a dictionary of hyperparameters
        using the names defined in the experiment's parameter set.
        """
        names = self.experiment.hyperparameters.names()
        # Handle case where X is (1, n) or (n,)
        if X.ndim == 2:
            X = X[0]

        params = {}
        for i, name in enumerate(names):
            val = X[i]
            # Convert to int if type is int in ParameterSet, but we only have names here.
            # We assume the user setup SpotOptim with correct types, so X comes in as
            # floats/ints/indices. For categorical, SpotOptim usually handles mapping if configured,
            # but if we passed names/bounds to SpotOptim, it handles it.
            # However, SpotOptim passes values.
            # We might need to cast to int if the parameter set says so.
            # For now, let's rely on the model to cast or just pass as is.
            # Actually, let's look at ParameterSet types if possible.
            # self.experiment.hyperparameters is a ParameterSet object.

            # Helper to cast if needed
            p_type = self.experiment.hyperparameters._var_types[i]
            if p_type == "int":
                val = int(round(val))
            elif p_type == "factor":
                # If SpotOptim handles factors, it might pass an index or the value?
                # SpotOptim usually handles factors by mapping to integers (indices).
                # Implementation details of SpotOptim: "Mapped to integers: 0 to n_levels - 1"
                # So we get an index. define choices.
                choices = self.experiment.hyperparameters._parameters[i].get("choices")
                if choices:
                    if isinstance(val, str):
                        # SpotOptim passed the string value directly
                        pass
                    else:
                        # SpotOptim passed a numeric index
                        idx = int(round(val))
                        # Clamp index just in case
                        idx = max(0, min(idx, len(choices) - 1))
                        val = choices[idx]

            params[name] = val

        return params

    def _prepare_data(self) -> tuple[DataLoader, DataLoader]:
        """Prepares DataLoaders from the experiment dataset."""
        data = self.experiment.dataset
        batch_size = self.experiment.batch_size
        num_workers = self.experiment.num_workers

        train_loader = None
        val_loader = None

        if isinstance(data, SpotDataFromArray):
            x_train, y_train = data.get_train_data()
            x_train = (
                torch.tensor(x_train, dtype=torch.float32)
                if not torch.is_tensor(x_train)
                else x_train.float()
            )
            y_train = (
                torch.tensor(y_train, dtype=torch.float32)
                if not torch.is_tensor(y_train)
                else y_train.float()
            )
            train_dataset = TensorDataset(x_train, y_train)
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            )

            val_data = data.get_validation_data()
            if val_data:
                x_val, y_val = val_data
                x_val = (
                    torch.tensor(x_val, dtype=torch.float32)
                    if not torch.is_tensor(x_val)
                    else x_val.float()
                )
                y_val = (
                    torch.tensor(y_val, dtype=torch.float32)
                    if not torch.is_tensor(y_val)
                    else y_val.float()
                )
                val_dataset = TensorDataset(x_val, y_val)
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                )

        elif isinstance(data, SpotDataFromTorchDataset):
            train_dataset = data.get_train_data()
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            )

            val_dataset = data.get_validation_data()
            if val_dataset:
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                )

        return train_loader, val_loader

    def train_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        params: Dict[str, Any],
    ) -> float:
        """
        Trains the model and returns the validation metric (loss).
        """
        # Optimizer
        # Check if model has get_optimizer method (like LinearRegressor)
        lr = params.get("lr", 1e-3)  # Default LR if not in params
        optimizer_name = params.get("optimizer", "Adam")

        if hasattr(model, "get_optimizer"):
            optimizer = model.get_optimizer(optimizer_name, lr=lr)
        else:
            # Fallback
            opt_class = getattr(optim, optimizer_name, optim.Adam)
            optimizer = opt_class(model.parameters(), lr=lr)

        criterion = self.experiment.loss_function or nn.MSELoss()

        epochs = self.experiment.epochs
        # Allow params to override epochs?
        epochs = params.get("epochs", epochs)

        model.to(self.device)

        min_val_loss = float("inf")

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            steps = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                steps += 1

            # Validation
            if val_loader:
                model.eval()
                val_loss = 0.0
                val_steps = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch, y_batch = X_batch.to(self.device), y_batch.to(
                            self.device
                        )
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
                        val_loss += loss.item()
                        val_steps += 1

                if val_steps > 0:
                    val_loss /= val_steps
                    if val_loss < min_val_loss:
                        min_val_loss = val_loss
            else:
                # If no validation set, use training loss? Or return 0?
                # Usually we want to minimize something.
                pass

        return min_val_loss if val_loader else train_loss / (steps if steps > 0 else 1)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        The function called by SpotOptim.
        X is shape (n_samples, n_params) or (n_params,).
        Returns y shape (n_samples, 1).
        """
        X = np.atleast_2d(X)
        n_samples = X.shape[0]
        results = []

        train_loader, val_loader = self._prepare_data()

        for i in range(n_samples):
            # Decode hyperparameters
            params = self._get_hyperparameters(X[i])

            # Instantiate model
            # We assume model_class takes input_dim/output_dim + kwargs matching params
            dataset = self.experiment.dataset
            model_kwargs = {
                "input_dim": dataset.input_dim,
                "output_dim": dataset.output_dim,
            }
            # Add hyperparams to kwargs
            model_kwargs.update(params)

            # Filter kwargs based on model signature
            sig = inspect.signature(self.experiment.model_class)
            has_var_keyword = any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            )

            if has_var_keyword:
                filtered_kwargs = model_kwargs
            else:
                filtered_kwargs = {
                    k: v for k, v in model_kwargs.items() if k in sig.parameters
                }

            try:
                # Try passing filtered params.
                model = self.experiment.model_class(**filtered_kwargs)
            except TypeError as e:
                # If unexpected argument, we might want to filter or warn.
                # For now, let's assume valid setup.
                # Re-raise with context
                raise TypeError(
                    f"Failed to instantiate model {self.experiment.model_class.__name__}: {e}"
                )

            loss = self.train_model(model, train_loader, val_loader, params)
            results.append(loss)

        return np.array(results).reshape(-1, 1)
