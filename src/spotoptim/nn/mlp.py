from typing import List, Optional, Callable, Any, TYPE_CHECKING
import torch
import torch.optim as optim


if TYPE_CHECKING:
    from spotoptim.hyperparameters.parameters import ParameterSet


class MLP(torch.nn.Sequential):
    """This block implements the multi-layer perceptron (MLP) module.

    Args:
        in_channels (int):
            Number of channels of the input
        hidden_channels (List[int]):
            List of the hidden channel dimensions. Note that the last element of this list is the output dimension of the network.
        norm_layer (Callable[..., torch.nn.Module], optional):
            Norm layer that will be stacked on top of the linear layer. If ``None`` this layer won't be used. Default: ``None``
        activation_layer (Callable[..., torch.nn.Module], optional):
            Activation function which will be stacked on top of the normalization layer (if not None),
            otherwise on top of the linear layer. If ``None`` this layer won't be used. Default: ``torch.nn.ReLU``
        inplace (bool, optional):
            Parameter for the activation layer, which can optionally do the operation in-place.
            Default is ``None``, which uses the respective default values of the ``activation_layer`` and Dropout layer.
        bias (bool):
            Whether to use bias in the linear layer. Default ``True``
        dropout (float):
            The probability for the dropout layer. Default: 0.0
        lr (float, optional):
            Unified learning rate multiplier. This value is automatically scaled to optimizer-specific learning rates using the map_lr() function.
            A value of 1.0 corresponds to the optimizer's default learning rate. Default: 1.0.
        l1 (int, optional):
            Number of neurons in each hidden layer. Will only be used if hidden_channels is None. Default: 64
        num_hidden_layers (int, optional):
            Number of hidden layers. Will only be used if hidden_channels is None. Default: 2

    Note:
        **Parameter Definitions:**

        *   **hidden_channels**: This defines the explicit architecture of the MLP. It is a list where each element is the size of a layer.
            The last element is the output dimension.
            Example: ``[32, 32, 1]`` means two hidden layers of size 32 and an output layer of size 1.

        *   **l1** and **num_hidden_layers**: These are helper parameters often used in hyperparameter optimization (see ``get_default_parameters()``).
            They will only be used if hidden_channels is None.
            *   ``l1``: The number of neurons in each hidden layer.
            *   ``num_hidden_layers``: The number of hidden layers *before* the output layer.

            They describe the architecture in a more compact way but are less flexible than ``hidden_channels``.
            Relationship: To convert ``l1`` and ``num_hidden_layers`` to ``hidden_channels`` for a given ``output_dim``:
            ``hidden_channels = [l1] * num_hidden_layers + [output_dim]``

    Examples:
        Basic usage:

        >>> import torch
        >>> from spotoptim.nn.mlp import MLP
        >>> # Input: 10 features. Output (is considered a hidden layer): 30 features. Hidden layer: 20 neurons.
        >>> mlp = MLP(in_channels=10, hidden_channels=[20, 30])
        >>> x = torch.randn(5, 10)
        >>> output = mlp(x)
        >>> print(output.shape)
        torch.Size([5, 30])

        Using get_optimizer:

        >>> model = MLP(in_channels=10, hidden_channels=[32, 1], lr=0.5)
        >>> optimizer = model.get_optimizer("Adam")  # Uses 0.5 * 0.001
        >>> print(optimizer)
        Adam (
        Parameter Group 0
            amsgrad: False
            betas: (0.9, 0.999)
            capturable: False
            differentiable: False
            eps: 1e-08
            foreach: None
            fused: None
            lr: 0.0005
            maximize: False
            weight_decay: 0
        )

        **Using l1 and num_hidden_layers parameters:**
        This example shows how to use the hyperparameters suggested by ``get_default_parameters()``
        to construct the ``hidden_channels`` list.

        >>> input_dim = 10
        >>> output_dim = 1
        >>>
        >>> # Hyperparameters (e.g., from spotoptim tuning)
        >>> l1 = 64
        >>> num_hidden_layers = 2
        >>>
        >>> # Construct hidden_channels list
        >>> # [64, 64, 1] -> 2 hidden layers of 64, output layer of 1
        >>> # Relationship: To convert l1 and num_hidden_layers to hidden_channels for a given output_dim:
        >>> # hidden_channels = [l1] * num_hidden_layers + [output_dim]
        >>> # but we can pass l1 and num_hidden_layers directly to the constructor
        >>> model = MLP(in_channels=input_dim, l1=l1, num_hidden_layers=num_hidden_layers, output_dim=output_dim)
        >>> print(model)
        MLP(
          (0): Linear(in_features=10, out_features=64, bias=True)
          (1): ReLU()
          (2): Dropout(p=0.0, inplace=False)
          (3): Linear(in_features=64, out_features=64, bias=True)
          (4): ReLU()
          (5): Dropout(p=0.0, inplace=False)
          (6): Linear(in_features=64, out_features=1, bias=True)
          (7): Dropout(p=0.0, inplace=False)
        )

        Getting default parameters for tuning:

        >>> params = MLP.get_default_parameters()
        >>> print(params.names())
        ['l1', 'num_hidden_layers', 'activation', 'lr', 'optimizer']
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int] = None,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        inplace: Optional[bool] = None,
        bias: bool = True,
        dropout: float = 0.0,
        lr: float = 1.0,
        l1: int = 64,
        num_hidden_layers: int = 2,
        output_dim: int = 1,
    ):
        self.lr = lr
        params = {} if inplace is None else {"inplace": inplace}
        layers = []
        in_dim = in_channels

        if hidden_channels is None:
            hidden_channels = [l1] * num_hidden_layers + [output_dim]

        # Loop over all hidden layers except the last one
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            if activation_layer is not None:
                layers.append(activation_layer(**params))
            layers.append(torch.nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        # Last layer
        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout, **params))

        super().__init__(*layers)

    def get_optimizer(
        self, optimizer_name: str = "Adam", lr: float = None, **kwargs: Any
    ) -> "optim.Optimizer":
        """Get a PyTorch optimizer configured for this model.

        Args:
            optimizer_name (str, optional): Name of the optimizer from torch.optim. Defaults to "Adam".
            lr (float, optional): Unified learning rate multiplier. If None, uses self.lr.
                This value is automatically scaled to optimizer-specific learning rates.
                A value of 1.0 corresponds to the optimizer's default learning rate.
                Defaults to None (uses self.lr).
            **kwargs: Additional optimizer-specific parameters.

        Returns:
            optim.Optimizer: Configured optimizer instance ready for training.
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
        """Returns a ParameterSet populated with default hyperparameters for this model.

        Note:
            Since MLP structure is generic (list of hidden channels), the default parameters
            provided here are a starting point assuming a simple structure similar to LinearRegressor
            (l1 units per layer, num_hidden_layers). This might need adjustment for specific architectures.

        Returns:
            ParameterSet: Default hyperparameters.

        Examples:
            >>> params = MLP.get_default_parameters()
            >>> print(params.names())
            ['l1', 'num_hidden_layers', 'activation', 'lr', 'optimizer']
        """
        from spotoptim.hyperparameters.parameters import ParameterSet

        params = ParameterSet()

        # l1: neurons in hidden layer
        params.add_int(name="l1", low=16, high=128, default=64, transform="log(x, 2)")

        # num_hidden_layers: depth
        params.add_int(name="num_hidden_layers", low=1, high=5, default=3)

        # activation: function name
        params.add_factor(
            name="activation",
            choices=["ReLU", "Tanh", "Sigmoid", "LeakyReLU", "ELU"],
            default="ReLU",
        )

        # lr: Unified learning rate
        params.add_float(name="lr", low=1e-4, high=100.0, default=10.0, transform="log")

        # optimizer
        params.add_factor(
            "optimizer", ["Adam", "SGD", "RMSprop", "AdamW"], default="Adam"
        )

        return params
