import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
from typing import Optional, Dict, Any, List, Tuple
from spotoptim.core.experiment import ExperimentControl
from spotoptim.core.data import SpotDataFromArray, SpotDataFromTorchDataset
import logging
import inspect

logger = logging.getLogger(__name__)


class TorchObjective:
    """
    A callable objective function for SpotOptim that trains and evaluates a PyTorch model.
    """

    def __init__(self, experiment: ExperimentControl, seed: Optional[int] = None):
        """
        Initialize the TorchObjective.

        Args:
            experiment (ExperimentControl): The experiment control object containing configuration,
                dataset, and hyperparameters.
            seed (Optional[int]): Random seed for reproducibility. If None, attempst to use
                experiment.seed. Defaults to None.

        Examples:
            >>> import torch
            >>> import torch.nn as nn
            >>> import numpy as np
            >>> from spotoptim.core.experiment import ExperimentControl
            >>> from spotoptim.function.torch_objective import TorchObjective
            >>> from spotoptim.hyperparameters import ParameterSet
            >>> from spotoptim.core.data import SpotDataFromArray
            >>>
            >>> # 1. Define a simple model
            >>> class SimpleModel(nn.Module):
            ...     def __init__(self, input_dim, output_dim, **kwargs):
            ...         super().__init__()
            ...         self.fc = nn.Linear(input_dim, output_dim)
            ...     def forward(self, x):
            ...         return self.fc(x)
            >>>
            >>> # 2. Prepare data
            >>> X = np.random.rand(10, 2)
            >>> y = np.random.rand(10, 1)
            >>> dataset = SpotDataFromArray(X, y)
            >>>
            >>> # 3. Define hyperparameters
            >>> params = ParameterSet()
            >>> params.add_float("lr", 1e-4, 1e-2, default=1e-3)
            >>>
            >>> # 4. Setup Experiment
            >>> exp = ExperimentControl(
            ...     name="test_exp",
            ...     model_class=SimpleModel,
            ...     dataset=dataset,
            ...     hyperparameters=params,
            ...     metrics=["val_loss"],
            ...     epochs=2,
            ...     batch_size=2
            ... )
            >>>
            >>> # 5. Initialize/Instantiate Objective
            >>> objective = TorchObjective(exp)
            >>> print(isinstance(objective, TorchObjective))
            True
        """
        self.experiment = experiment
        self.experiment = experiment
        self.device = experiment.torch_device

        # Use provided seed, or fall back to experiment seed, or None
        if seed is not None:
            self.seed = seed
        else:
            exp_seed = getattr(experiment, "seed", None)
            # Ensure it's a valid seed type (int) to avoid issues with Mocks in testing
            if isinstance(exp_seed, int):
                self.seed = exp_seed
            else:
                self.seed = None

    @property
    def bounds(self) -> List[Tuple[float, float]]:
        """
        Returns the bounds of the hyperparameters.

        Returns:
            List[Tuple[float, float]]: A list of tuples defining the (min, max) bounds for each parameter.
        """
        return self.experiment.hyperparameters.bounds

    @property
    def var_type(self) -> List[str]:
        """
        Returns the types of the hyperparameters.

        Returns:
            List[str]: A list of strings indicating the type of each parameter (e.g., 'float', 'int', 'factor').
        """
        return self.experiment.hyperparameters.var_type

    @property
    def var_name(self) -> List[str]:
        """
        Returns the names of the hyperparameters.

        Returns:
            List[str]: A list of parameter names.
        """
        return self.experiment.hyperparameters.var_name

    @property
    def var_trans(self) -> List[str]:
        """
        Returns the transformations of the hyperparameters.

        Returns:
            List[str]: A list of transformation strings (e.g., 'log', 'linear').
        """
        return self.experiment.hyperparameters.var_trans

    @property
    def objective_names(self) -> List[str]:
        """
        Returns the names of the objectives.

        Returns:
            List[str]: A list of objective metric names.
        """
        return self.experiment.metrics

    def _get_hyperparameters(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Converts the input vector X into a dictionary of hyperparameters.

        This method handles mapping numeric values from the optimization process back to
        parameter names and types defined in the experiment, including integer and
        categorical (factor) handling.

        Args:
            X (np.ndarray): Input parameter vector from the optimizer. Can be 1D or 2D.

        Returns:
            Dict[str, Any]: A dictionary where keys are parameter names and values are
                the corresponding values to be passed to the model.

        Examples:
            >>> import numpy as np
            >>> from spotoptim.core.experiment import ExperimentControl
            >>> from spotoptim.function.torch_objective import TorchObjective
            >>> from spotoptim.hyperparameters import ParameterSet
            >>>
            >>> # Setup parameters
            >>> params = ParameterSet()
            >>> params.add_float("lr", 0.001, 0.1)
            >>> params.add_int("batch_size", 16, 128)
            >>>
            >>> # Mock experiment for context (minimal setup)
            >>> class MockExp:
            ...     hyperparameters = params
            ...     torch_device = "cpu"
            >>>
            >>> objective = TorchObjective(MockExp())
            >>>
            >>> # Input vector matching parameter order (lr, batch_size)
            >>> X = np.array([0.01, 64.2])
            >>>
            >>> # Convert to dict
            >>> hyp = objective._get_hyperparameters(X)
            >>> print(hyp)
            {'lr': 0.01, 'batch_size': 64}
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

    def _prepare_data(
        self, batch_size: Optional[int] = None
    ) -> tuple[DataLoader, DataLoader]:
        """
        Prepares DataLoaders from the experiment dataset.

        Based on the data type (Array or TorchDataset), creates appropriate DataLoaders
        using the provided batch size or the one specified in the experiment.

        Args:
            batch_size (Optional[int]): Batch size to use. If None, uses self.experiment.batch_size.

        Returns:
            tuple[DataLoader, DataLoader]: A tuple containing (train_loader, val_loader).
                val_loader may be None if no validation data is available.
        """
        data = self.experiment.dataset
        if batch_size is None:
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
    ) -> Dict[str, float]:
        """
        Trains the model and returns a dictionary of metrics.

        Executes the training loop for the specified number of epochs. Handles optimizer
        creation, loss calculation, backward pass, and validation evaluation.

        Args:
            model (nn.Module): The PyTorch model to train.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (Optional[DataLoader]): DataLoader for validation data (can be None).
            params (Dict[str, Any]): Hyperparameters dictionary containing 'epochs', 'lr',
                'optimizer' name, etc.

        Returns:
            Dict[str, float]: Dictionary containing computed metrics, e.g.,
                {'val_loss': ..., 'train_loss': ..., 'mse': ..., 'epochs': ...}.

        Examples:
            >>> import torch
            >>> import torch.nn as nn
            >>> from torch.utils.data import DataLoader, TensorDataset
            >>> from spotoptim.function.torch_objective import TorchObjective
            >>> from unittest.mock import MagicMock
            >>>
            >>> # 1. Create dataset and loader
            >>> X = torch.randn(10, 2)
            >>> y = torch.randn(10, 1)
            >>> loader = DataLoader(TensorDataset(X, y), batch_size=2)
            >>>
            >>> # 2. Create model
            >>> model = nn.Linear(2, 1)
            >>>
            >>> # 3. Mock Objective context
            >>> exp = MagicMock()
            >>> exp.loss_function = nn.MSELoss()
            >>> exp.epochs = 1
            >>> exp.torch_device = "cpu"
            >>> objective = TorchObjective(exp)
            >>>
            >>> # 4. Train
            >>> params = {'lr': 1e-2, 'optimizer': 'Adam'}
            >>> metrics = objective.train_model(model, loader, None, params)
            >>> print(f"Train Loss: {metrics['train_loss']:.4f}")
            >>> print(f"Epochs: {metrics['epochs']}")
        """
        # Optimizer
        lr = params.get("lr", 1e-3)
        optimizer_name = params.get("optimizer", "Adam")

        if hasattr(model, "get_optimizer"):
            optimizer = model.get_optimizer(optimizer_name, lr=lr)
        else:
            opt_class = getattr(optim, optimizer_name, optim.Adam)
            optimizer = opt_class(model.parameters(), lr=lr)

        criterion = self.experiment.loss_function or nn.MSELoss()

        base_epochs = self.experiment.epochs
        if "epochs" in params:
            epochs = int(params["epochs"])
        elif base_epochs is not None:
            epochs = int(base_epochs)
        else:
            epochs = 100

        model.to(self.device)

        min_val_loss = float("inf")
        final_train_loss = 0.0

        # Track actual epochs trained (if early stopping implementation added later)
        # For now, we train for exact 'epochs'
        trained_epochs = epochs

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

            final_train_loss = train_loss / (steps if steps > 0 else 1)

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
                # If no validation set, min_val_loss tracks train loss
                if final_train_loss < min_val_loss:
                    min_val_loss = final_train_loss

        # Collect metrics
        metrics_out = {}

        # Default MSE / val_loss
        metrics_out["val_loss"] = min_val_loss
        metrics_out["train_loss"] = final_train_loss
        metrics_out["mse"] = min_val_loss  # Alias
        metrics_out["epochs"] = float(trained_epochs)

        return metrics_out

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the objective function for a given set of parameters.

        This is the main entry point called by the optimizer. It iterates over the
        samples in X, instantiates and trains a model for each sample, and collects
        the requested metrics.

        Args:
            X (np.ndarray): Input array of shape (n_samples, n_params) or (n_params,).
                Contains the hyperparameter configurations to evaluate.

        Returns:
            np.ndarray: Array of shape (n_samples, n_metrics) containing the evaluation results.

        Raises:
            TypeError: If the model class cannot be instantiated with the provided parameters.

        Examples:
            >>> import torch
            >>> import torch.nn as nn
            >>> import numpy as np
            >>> from spotoptim.core.experiment import ExperimentControl
            >>> from spotoptim.function.torch_objective import TorchObjective
            >>> from spotoptim.hyperparameters import ParameterSet
            >>> from spotoptim.core.data import SpotDataFromArray
            >>>
            >>> # 1. Define Model
            >>> class SimpleModel(nn.Module):
            ...     def __init__(self, input_dim, output_dim, **kwargs):
            ...         super().__init__()
            ...         self.fc = nn.Linear(input_dim, output_dim)
            ...     def forward(self, x):
            ...         return self.fc(x)
            >>>
            >>> # 2. Setup Data & Experiment
            >>> X_data = np.random.rand(10, 2)
            >>> y_data = np.random.rand(10, 1)
            >>> params = ParameterSet().add_float("lr", 1e-4, 1e-2)
            >>>
            >>> exp = ExperimentControl(
            ...     name="test_call",
            ...     model_class=SimpleModel,
            ...     dataset=SpotDataFromArray(X_data, y_data),
            ...     hyperparameters=params,
            ...     metrics=["val_loss"],
            ...     epochs=1,
            ...     batch_size=5
            ... )
            >>>
            >>> # 3. Initialize Objective
            >>> objective = TorchObjective(exp)
            >>>
            >>> # 4. Define input parameters to evaluate (e.g., lr=0.005)
            >>> X_eval = np.array([[0.005]])
            >>>
            >>> # 5. Evaluate
            >>> results = objective(X_eval)
            >>> print(f"Results shape: {results.shape}")
            Results shape: (1, 1)
            >>> print(f"Val Loss: {results[0, 0]:.4f}")
        """
        X = np.atleast_2d(X)
        n_samples = X.shape[0]
        results = []

        train_loader, val_loader = self._prepare_data()

        # Check if batch_size is a tunable parameter
        batch_size_tunable = "batch_size" in self.experiment.hyperparameters.names()

        # If batch_size is NOT tunable, we can reuse the initial loaders for all samples
        # to avoid overhead. If it IS tunable, we must recreate loaders per sample.
        recreate_loaders = batch_size_tunable

        # metrics to return
        requested_metrics = (
            self.experiment.metrics if self.experiment.metrics else ["val_loss"]
        )

        for i in range(n_samples):
            # Set seed if available to ensure reproducibility for each evaluation
            if self.seed is not None:
                self._set_seed(self.seed)

            # Decode hyperparameters
            params = self._get_hyperparameters(X[i])

            # Handle batch_size if it's being tuned
            if recreate_loaders and "batch_size" in params:
                current_batch_size = int(params["batch_size"])
                train_loader, val_loader = self._prepare_data(
                    batch_size=current_batch_size
                )

            # Instantiate model
            dataset = self.experiment.dataset
            model_kwargs = {
                "input_dim": dataset.input_dim,
                "in_channels": dataset.input_dim,
                "output_dim": dataset.output_dim,
            }
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
                model = self.experiment.model_class(**filtered_kwargs)
            except TypeError as e:
                raise TypeError(
                    f"Failed to instantiate model {self.experiment.model_class.__name__}: {e}"
                )

            metrics_out = self.train_model(model, train_loader, val_loader, params)

            # Extract requested metrics
            row = []
            for m in requested_metrics:
                # Fuzzy match for common names
                if m.lower() in ["val_loss", "mse", "loss"]:
                    row.append(metrics_out.get("val_loss", float("inf")))
                elif m.lower() in ["epochs", "epoch"]:
                    row.append(metrics_out.get("epochs", 0.0))
                else:
                    # Generic lookup
                    row.append(metrics_out.get(m, float("nan")))

            results.append(row)

        return np.array(results)

    def _set_seed(self, seed: int):
        """
        Sets the seed for random number generators.

        Args:
            seed (int): The seed value.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Ensure deterministic behavior
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False
