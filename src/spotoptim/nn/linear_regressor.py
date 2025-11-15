# generate a pytorch linear regression model for supervised learning
import torch
import torch.nn as nn
import torch.optim as optim


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
        optimizer (str, optional): Name of the optimizer to use. Common options: "Adam", "SGD", "RMSprop". Defaults to "Adam".


    Attributes:
        input_dim (int): Number of input features.
        output_dim (int): Number of output features.
        l1 (int): Number of neurons per hidden layer.
        num_hidden_layers (int): Number of hidden layers in the network.
        activation_name (str): Name of the activation function.
        activation (nn.Module): Instance of the activation function.
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
        self, input_dim, output_dim, l1=64, num_hidden_layers=0, activation="ReLU"
    ):
        super(LinearRegressor, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.l1 = l1
        self.num_hidden_layers = num_hidden_layers
        self.activation_name = activation

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

    def forward(self, x):
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

    def get_optimizer(self, optimizer_name="Adam", lr=0.001, **kwargs):
        """Get a PyTorch optimizer configured for this model.

        Convenience method to instantiate optimizers using string names instead of
        importing optimizer classes. Automatically configures the optimizer with the
        model's parameters.

        Args:
            optimizer_name (str, optional): Name of the optimizer from torch.optim.
                Common options: "Adam", "AdamW", "Adamax", "SGD", "RMSprop", "Adagrad",
                "Adadelta", "NAdam", "RAdam", "ASGD", "LBFGS", "Rprop".
                Defaults to "Adam".
            lr (float, optional): Learning rate for the optimizer. Defaults to 0.001.
            **kwargs: Additional optimizer-specific parameters (e.g., momentum for SGD,
                weight_decay for AdamW, alpha for RMSprop).

        Returns:
            torch.optim.Optimizer: Configured optimizer instance ready for training.

        Raises:
            ValueError: If the specified optimizer name is not found in torch.optim.

        Examples:
            Basic usage with default Adam:

            >>> model = LinearRegressor(input_dim=10, output_dim=1)
            >>> optimizer = model.get_optimizer()  # Uses Adam with lr=0.001

            SGD with momentum:

            >>> optimizer = model.get_optimizer("SGD", lr=0.01, momentum=0.9)

            AdamW with weight decay for regularization:

            >>> optimizer = model.get_optimizer("AdamW", lr=0.001, weight_decay=0.01)

            RMSprop with custom alpha:

            >>> optimizer = model.get_optimizer("RMSprop", lr=0.01, alpha=0.99)

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
            >>> # Create model and optimizer
            >>> model = LinearRegressor(input_dim=10, output_dim=1, l1=16, num_hidden_layers=1)
            >>> optimizer = model.get_optimizer("Adam", lr=0.01)
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
            >>> # Create model and optimizer
            >>> model = LinearRegressor(input_dim=10, output_dim=1, l1=16, num_hidden_layers=1)
            >>> optimizer = model.get_optimizer("SGD", lr=0.01, momentum=0.9)
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

        Note:
            The optimizer is automatically initialized with self.parameters(), so you
            don't need to manually pass the model parameters. Using DataLoader enables
            efficient mini-batch training, shuffling, and parallel data loading.
        """
        # Check if optimizer exists in torch.optim
        if hasattr(optim, optimizer_name):
            optimizer_class = getattr(optim, optimizer_name)
            # Create optimizer with model parameters, learning rate, and additional kwargs
            return optimizer_class(self.parameters(), lr=lr, **kwargs)
        else:
            raise ValueError(
                f"Optimizer '{optimizer_name}' not found in torch.optim. "
                f"Please use a valid PyTorch optimizer name like 'Adam', 'SGD', 'AdamW', etc."
            )
