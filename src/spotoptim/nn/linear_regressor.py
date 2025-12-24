# generate a pytorch linear regression model for supervised learning
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from spotoptim.hyperparameters.parameters import ParameterSet


class LinearRegressor(nn.Module):
    """PyTorch neural network for regression with configurable architecture.

    A flexible regression model that supports:
    - Pure linear regression (no hidden layers)
    - Deep neural networks with multiple hidden layers
    - Various activation functions (ReLU, Tanh, Sigmoid, etc.)
    - Easy optimizer selection (Adam, SGD, RMSprop, etc.)

    Args:
        input_dim (int): Number of input features.
        output_dim (int): Number of output features/targets.
        l1 (int, optional): Number of neurons in each hidden layer. Defaults to 64.
        num_hidden_layers (int, optional): Number of hidden layers. Set to 0 for pure
            linear regression. Defaults to 0.
        activation (str, optional): Name of activation function from torch.nn to use
            between layers. Common options: "ReLU", "Sigmoid", "Tanh", "LeakyReLU",
            "ELU", "SELU", "GELU", "Softplus", "Softsign", "Mish". Defaults to "ReLU".
        lr (float, optional): Unified learning rate multiplier. This value is automatically
            scaled to optimizer-specific learning rates using the map_lr() function.
            A value of 1.0 corresponds to the optimizer's default learning rate.
            For example, lr=1.0 gives 0.001 for Adam and 0.01 for SGD. Typical range:
            [0.001, 100.0]. Defaults to 1.0.


    Attributes:
        input_dim (int): Number of input features.
        output_dim (int): Number of output features.
        l1 (int): Number of neurons per hidden layer.
        num_hidden_layers (int): Number of hidden layers in the network.
        activation_name (str): Name of the activation function.
        activation (nn.Module): Instance of the activation function.
        lr (float): Unified learning rate multiplier.
        network (nn.Sequential): The complete neural network architecture.


    Raises:
        ValueError: If the specified activation function is not found in torch.nn.

    Examples:
        Basic usage with pure linear regression:

        >>> import torch
        >>> from spotoptim.nn.linear_regressor import LinearRegressor
        >>>
        >>> # Pure linear regression (no hidden layers)
        >>> model = LinearRegressor(input_dim=10, output_dim=1)
        >>> x = torch.randn(32, 10)  # Batch of 32 samples
        >>> y_pred = model(x)
        >>> print(y_pred.shape)
        torch.Size([32, 1])

        Single hidden layer with custom neurons:

        >>> # Single hidden layer with 64 neurons and ReLU activation
        >>> model = LinearRegressor(input_dim=10, output_dim=1, l1=64, num_hidden_layers=1)
        >>> optimizer = model.get_optimizer("Adam", lr=0.001)

        Deep network with custom activation:

        >>> # Three hidden layers with 128 neurons each and Tanh activation
        >>> model = LinearRegressor(input_dim=10, output_dim=1, l1=128,
        ...                         num_hidden_layers=3, activation="Tanh")

        Complete example using diabetes dataset:

        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.preprocessing import StandardScaler
        >>> import torch
        >>> import torch.nn as nn
        >>>
        >>> # Load and prepare data
        >>> diabetes = load_diabetes()
        >>> X, y = diabetes.data, diabetes.target.reshape(-1, 1)
        >>>
        >>> # Split and scale data
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X, y, test_size=0.2, random_state=42
        ... )
        >>> scaler = StandardScaler()
        >>> X_train = scaler.fit_transform(X_train)
        >>> X_test = scaler.transform(X_test)
        >>>
        >>> # Convert to PyTorch tensors
        >>> X_train = torch.FloatTensor(X_train)
        >>> y_train = torch.FloatTensor(y_train)
        >>> X_test = torch.FloatTensor(X_test)
        >>> y_test = torch.FloatTensor(y_test)
        >>>
        >>> # Create model with 2 hidden layers
        >>> model = LinearRegressor(
        ...     input_dim=10,  # diabetes dataset has 10 features
        ...     output_dim=1,
        ...     l1=32,
        ...     num_hidden_layers=2,
        ...     activation="ReLU"
        ... )
        >>>
        >>> # Get optimizer and loss function
        >>> optimizer = model.get_optimizer("Adam", lr=0.01)
        >>> criterion = nn.MSELoss()
        >>>
        >>> # Training loop
        >>> for epoch in range(100):
        ...     # Forward pass
        ...     y_pred = model(X_train)
        ...     loss = criterion(y_pred, y_train)
        ...
        ...     # Backward pass and optimization
        ...     optimizer.zero_grad()
        ...     loss.backward()
        ...     optimizer.step()
        ...
        ...     if (epoch + 1) % 20 == 0:
        ...         print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')
        >>>
        >>> # Evaluate on test set
        >>> model.eval()
        >>> with torch.no_grad():
        ...     y_pred = model(X_test)
        ...     test_loss = criterion(y_pred, y_test)
        ...     print(f'Test Loss: {test_loss.item():.4f}')

        Using PyTorch Dataset and DataLoader (recommended for larger datasets):

        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.preprocessing import StandardScaler
        >>> import torch
        >>> from torch.utils.data import Dataset, DataLoader
        >>> import torch.nn as nn
        >>>
        >>> # Custom Dataset class for diabetes data
        >>> class DiabetesDataset(Dataset):
        ...     def __init__(self, X, y):
        ...         self.X = torch.FloatTensor(X)
        ...         self.y = torch.FloatTensor(y)
        ...
        ...     def __len__(self):
        ...         return len(self.X)
        ...
        ...     def __getitem__(self, idx):
        ...         return self.X[idx], self.y[idx]
        >>>
        >>> # Load and prepare data
        >>> diabetes = load_diabetes()
        >>> X, y = diabetes.data, diabetes.target.reshape(-1, 1)
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X, y, test_size=0.2, random_state=42
        ... )
        >>>
        >>> # Scale data
        >>> scaler = StandardScaler()
        >>> X_train = scaler.fit_transform(X_train)
        >>> X_test = scaler.transform(X_test)
        >>>
        >>> # Create Dataset and DataLoader
        >>> train_dataset = DiabetesDataset(X_train, y_train)
        >>> test_dataset = DiabetesDataset(X_test, y_test)
        >>> train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        >>> test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        >>>
        >>> # Create model
        >>> model = LinearRegressor(input_dim=10, output_dim=1, l1=32,
        ...                         num_hidden_layers=2, activation="ReLU")
        >>> optimizer = model.get_optimizer("Adam", lr=0.01)
        >>> criterion = nn.MSELoss()
        >>>
        >>> # Training loop with DataLoader
        >>> for epoch in range(100):
        ...     model.train()
        ...     for batch_X, batch_y in train_loader:
        ...         # Forward pass
        ...         predictions = model(batch_X)
        ...         loss = criterion(predictions, batch_y)
        ...
        ...         # Backward pass
        ...         optimizer.zero_grad()
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
        - When num_hidden_layers=0, the model performs pure linear regression
        - Activation functions are only applied between hidden layers, not on output
        - Use get_optimizer() method for convenient optimizer instantiation
        - For large datasets, use PyTorch Dataset and DataLoader for efficient batch processing
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        l1=64,
        num_hidden_layers=0,
        activation="ReLU",
        lr=1.0,
    ):
        super(LinearRegressor, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.l1 = l1
        self.num_hidden_layers = num_hidden_layers
        self.activation_name = activation
        self.lr = lr

        # Get activation function class from string
        if hasattr(nn, activation):
            activation_class = getattr(nn, activation)
            self.activation = activation_class()
        else:
            raise ValueError(
                f"Activation function '{activation}' not found in torch.nn. "
                f"Please use a valid PyTorch activation function name like 'ReLU', 'Sigmoid', 'Tanh', etc."
            )

        # Build the network layers
        layers = []

        if num_hidden_layers == 0:
            # Pure linear regression (no hidden layers)
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            # Input layer to first hidden layer
            layers.append(nn.Linear(input_dim, l1))
            layers.append(self.activation)

            # Additional hidden layers
            for _ in range(num_hidden_layers - 1):
                layers.append(nn.Linear(l1, l1))
                # Create a new instance of the activation for each layer
                layers.append(getattr(nn, self.activation_name)())

            # Final hidden layer to output
            layers.append(nn.Linear(l1, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x) -> "torch.Tensor":
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output predictions of shape (batch_size, output_dim).

        Examples:
            >>> model = LinearRegressor(input_dim=5, output_dim=1)
            >>> x = torch.randn(10, 5)  # 10 samples, 5 features
            >>> output = model(x)
            >>> print(output.shape)
            torch.Size([10, 1])
        """
        return self.network(x)

    def get_optimizer(
        self, optimizer_name: str = "Adam", lr: float = None, **kwargs: Any
    ) -> "optim.Optimizer":
        """Get a PyTorch optimizer configured for this model.

        Convenience method to instantiate optimizers using string names instead of
        importing optimizer classes. Automatically configures the optimizer with the
        model's parameters and applies learning rate mapping for unified interface.

        If lr is not specified, uses the model's lr attribute (default 1.0) which
        is automatically mapped to optimizer-specific learning rates using map_lr().
        For example, lr=1.0 gives 0.001 for Adam, 0.01 for SGD, etc.

        Args:
            optimizer_name (str, optional): Name of the optimizer from torch.optim.
                Common options: "Adam", "AdamW", "Adamax", "SGD", "RMSprop", "Adagrad",
                "Adadelta", "NAdam", "RAdam", "ASGD", "LBFGS", "Rprop".
                Defaults to "Adam".
            lr (float, optional): Unified learning rate multiplier. If None, uses self.lr.
                This value is automatically scaled to optimizer-specific learning rates.
                A value of 1.0 corresponds to the optimizer's default learning rate.
                Typical range: [0.001, 100.0]. Defaults to None (uses self.lr).
            **kwargs: Additional optimizer-specific parameters (e.g., momentum for SGD,
                weight_decay for AdamW, alpha for RMSprop).

        Returns:
            optim.Optimizer: Configured optimizer instance ready for training.

        Raises:
            ValueError: If the specified optimizer name is not found in torch.optim or
                not supported by map_lr().

        Examples:
            Basic usage with model's default unified lr (1.0):

            >>> model = LinearRegressor(input_dim=10, output_dim=1, lr=1.0)
            >>> optimizer = model.get_optimizer("Adam")  # Uses 1.0 * 0.001 = 0.001
            >>> optimizer = model.get_optimizer("SGD")   # Uses 1.0 * 0.01 = 0.01

            Using custom unified learning rate in model:

            >>> model = LinearRegressor(input_dim=10, output_dim=1, lr=0.5)
            >>> optimizer = model.get_optimizer("Adam")  # Uses 0.5 * 0.001 = 0.0005
            >>> optimizer = model.get_optimizer("SGD")   # Uses 0.5 * 0.01 = 0.005

            Override model's lr with method parameter:

            >>> model = LinearRegressor(input_dim=10, output_dim=1, lr=1.0)
            >>> optimizer = model.get_optimizer("Adam", lr=2.0)  # Uses 2.0 * 0.001 = 0.002

            SGD with momentum and unified learning rate:

            >>> optimizer = model.get_optimizer("SGD", lr=0.5, momentum=0.9)

            AdamW with weight decay:

            >>> optimizer = model.get_optimizer("AdamW", lr=1.0, weight_decay=0.01)

            Complete training example with diabetes dataset:

            >>> from sklearn.datasets import load_diabetes
            >>> from sklearn.preprocessing import StandardScaler
            >>> import torch
            >>> import torch.nn as nn
            >>>
            >>> # Prepare data
            >>> diabetes = load_diabetes()
            >>> X = StandardScaler().fit_transform(diabetes.data)
            >>> y = diabetes.target.reshape(-1, 1)
            >>> X_tensor = torch.FloatTensor(X)
            >>> y_tensor = torch.FloatTensor(y)
            >>>
            >>> # Create model and optimizer with unified learning rate
            >>> model = LinearRegressor(input_dim=10, output_dim=1, l1=16,
            ...                         num_hidden_layers=1, lr=10.0)
            >>> optimizer = model.get_optimizer("Adam")  # Uses 10.0 * 0.001 = 0.01
            >>> criterion = nn.MSELoss()
            >>>
            >>> # Training
            >>> for epoch in range(50):
            ...     optimizer.zero_grad()
            ...     predictions = model(X_tensor)
            ...     loss = criterion(predictions, y_tensor)
            ...     loss.backward()
            ...     optimizer.step()

            Using with DataLoader for mini-batch training:

            >>> from sklearn.datasets import load_diabetes
            >>> from sklearn.preprocessing import StandardScaler
            >>> from torch.utils.data import Dataset, DataLoader
            >>> import torch
            >>> import torch.nn as nn
            >>>
            >>> # Custom Dataset
            >>> class DiabetesDataset(Dataset):
            ...     def __init__(self, X, y):
            ...         self.X = torch.FloatTensor(X)
            ...         self.y = torch.FloatTensor(y)
            ...
            ...     def __len__(self):
            ...         return len(self.X)
            ...
            ...     def __getitem__(self, idx):
            ...         return self.X[idx], self.y[idx]
            >>>
            >>> # Prepare data
            >>> diabetes = load_diabetes()
            >>> X = StandardScaler().fit_transform(diabetes.data)
            >>> y = diabetes.target.reshape(-1, 1)
            >>>
            >>> # Create DataLoader
            >>> dataset = DiabetesDataset(X, y)
            >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            >>>
            >>> # Create model and optimizer with unified learning rate
            >>> model = LinearRegressor(input_dim=10, output_dim=1, l1=16,
            ...                         num_hidden_layers=1, lr=1.0)
            >>> optimizer = model.get_optimizer("SGD", momentum=0.9)  # Uses 1.0 * 0.01 = 0.01
            >>> criterion = nn.MSELoss()
            >>>
            >>> # Training with mini-batches
            >>> for epoch in range(100):
            ...     for batch_X, batch_y in dataloader:
            ...         optimizer.zero_grad()
            ...         predictions = model(batch_X)
            ...         loss = criterion(predictions, batch_y)
            ...         loss.backward()
            ...         optimizer.step()

            Hyperparameter optimization across optimizers:

            >>> from spotoptim import SpotOptim
            >>> import numpy as np
            >>>
            >>> def optimize_model(X):
            ...     results = []
            ...     for params in X:
            ...         lr_unified = 10 ** params[0]  # Log scale: [-2, 2]
            ...         optimizer_name = params[1]     # Factor: "Adam", "SGD", "RMSprop"
            ...
            ...         # Create model with unified lr - automatically scaled per optimizer
            ...         model = LinearRegressor(input_dim=10, output_dim=1, lr=lr_unified)
            ...         optimizer = model.get_optimizer(optimizer_name)
            ...
            ...         # Train and evaluate
            ...         # ... training code ...
            ...         results.append(test_loss)
            ...     return np.array(results)
            >>>
            >>> spot = SpotOptim(
            ...     fun=optimize_model,
            ...     bounds=[(-2, 2), ("Adam", "SGD", "RMSprop")],
            ...     var_type=["num", "factor"]
            ... )

        Note:
            - The optimizer uses self.parameters() automatically
            - Learning rates are mapped using spotoptim.utils.mapping.map_lr()
            - Unified lr interface enables fair comparison across optimizers
            - A unified lr of 1.0 always corresponds to optimizer's PyTorch default
            - DataLoader enables efficient mini-batch training and data shuffling
        """
        from spotoptim.utils.mapping import map_lr

        # Use model's lr if not specified
        if lr is None:
            lr = self.lr

        # Map unified learning rate to optimizer-specific learning rate
        try:
            lr_actual = map_lr(lr, optimizer_name)
        except ValueError:
            # If optimizer not in map_lr, try to use it directly with torch.optim
            if not hasattr(optim, optimizer_name):
                raise ValueError(
                    f"Optimizer '{optimizer_name}' not found in torch.optim and not supported by map_lr(). "
                    f"Please use a valid PyTorch optimizer name like 'Adam', 'SGD', 'AdamW', etc."
                )
            # Use unified lr directly if optimizer not in mapping
            lr_actual = lr

        # Check if optimizer exists in torch.optim
        if hasattr(optim, optimizer_name):
            optimizer_class = getattr(optim, optimizer_name)
            # Create optimizer with model parameters, mapped learning rate, and additional kwargs
            return optimizer_class(self.parameters(), lr=lr_actual, **kwargs)
        else:
            raise ValueError(
                f"Optimizer '{optimizer_name}' not found in torch.optim. "
                f"Please use a valid PyTorch optimizer name like 'Adam', 'SGD', 'AdamW', etc."
            )

    @staticmethod
    def get_default_parameters() -> "ParameterSet":
        """
        Returns a ParameterSet populated with default hyperparameters for this model.
        Users can modify bounds and defaults as needed.

        Returns:
            ParameterSet: Default hyperparameters.

        Examples:
            >>> params = LinearRegressor.get_default_parameters()
            >>> print(params)
            ParameterSet(
                l1=Parameter(
                    name='l1',
                    var_name='l1',
                    bounds=Bounds(low=16, high=128),
                    default=64,
                    log=False,
                    type='int'
                ),
                num_hidden_layers=Parameter(
                    name='num_hidden_layers',
                    var_name='num_hidden_layers',
                    bounds=Bounds(low=0, high=3),
                    default=0,
                    log=False,
                    type='int'
                ),
                activation=Parameter(
                    name='activation',
                    var_name='activation',
                    bounds=Bounds(low='ReLU', high='Tanh'),
                    default='ReLU',
                    log=False,
                    type='str'
                ),
                lr=Parameter(
                    name='lr',
                    var_name='lr',
                    bounds=Bounds(low=1e-4, high=1e-1),
                    default=1e-3,
                    log=True,
                    type='float'
                ),
                optimizer=Parameter(
                    name='optimizer',
                    var_name='optimizer',
                    bounds=Bounds(low='Adam', high='SGD'),
                    default='Adam',
                    log=False,
                    type='str'
                )
            )
        """
        from spotoptim.hyperparameters.parameters import ParameterSet

        params = ParameterSet()

        # l1: neurons in hidden layer
        params.add_int(name="l1", low=16, high=128, default=64)

        # num_hidden_layers: depth
        params.add_int(name="num_hidden_layers", low=0, high=3, default=0)

        # activation: function name
        params.add_factor(
            name="activation",
            choices=["ReLU", "Tanh", "Sigmoid", "LeakyReLU", "ELU"],
            default="ReLU",
        )
        # lr: Unified learning rate
        params.add_float(name="lr", low=1e-4, high=10.0, default=1.0, transform="log")

        # optimizer
        params.add_factor(
            "optimizer", ["Adam", "SGD", "RMSprop", "AdamW"], default="Adam"
        )

        return params
