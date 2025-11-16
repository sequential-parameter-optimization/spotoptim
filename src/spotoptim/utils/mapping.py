"""Learning rate mapping functions for unified optimizer interface.

This module provides utilities to map a unified learning rate scale to
optimizer-specific learning rates, accounting for the different default
and typical ranges used by different PyTorch optimizers.
"""

from typing import Dict


# Default learning rates for each optimizer as defined in PyTorch documentation
# Source: https://pytorch.org/docs/stable/optim.html
OPTIMIZER_DEFAULT_LR: Dict[str, float] = {
    "Adadelta": 1.0,
    "Adagrad": 0.01,
    "Adam": 0.001,
    "AdamW": 0.001,
    "SparseAdam": 0.001,
    "Adamax": 0.002,
    "ASGD": 0.01,
    "LBFGS": 1.0,
    "NAdam": 0.002,
    "RAdam": 0.001,
    "RMSprop": 0.01,
    "Rprop": 0.01,
    "SGD": 0.01,
}


def map_lr(
    lr_unified: float, optimizer_name: str, use_default_scale: bool = True
) -> float:
    """Map a unified learning rate to an optimizer-specific learning rate.

    This function provides a unified interface for learning rates across different
    PyTorch optimizers. Different optimizers operate on vastly different learning
    rate scales (e.g., SGD typically uses lr ~ 0.01-0.1, while Adam uses lr ~ 0.0001-0.001).

    The mapping uses the default learning rates from PyTorch as scaling factors,
    allowing users to work with a normalized learning rate scale where 1.0 represents
    the optimizer's default learning rate.

    Args:
        lr_unified (float): Unified learning rate multiplier. A value of 1.0 corresponds
            to the optimizer's default learning rate. Values < 1.0 reduce the learning
            rate, values > 1.0 increase it. Typical range: [0.001, 100.0].
        optimizer_name (str): Name of the PyTorch optimizer. Must be one of:
            "Adadelta", "Adagrad", "Adam", "AdamW", "SparseAdam", "Adamax",
            "ASGD", "LBFGS", "NAdam", "RAdam", "RMSprop", "Rprop", "SGD".
        use_default_scale (bool, optional): Whether to scale by the optimizer's default
            learning rate. If True (default), lr_unified is multiplied by the default
            lr. If False, returns lr_unified directly. Defaults to True.

    Returns:
        float: The optimizer-specific learning rate.

    Raises:
        ValueError: If optimizer_name is not supported.
        ValueError: If lr_unified is not positive.

    Examples:
        Using unified learning rate with default scaling:

        >>> # Get Adam's default learning rate (0.001)
        >>> lr = map_lr(1.0, "Adam")
        >>> print(lr)
        0.001

        >>> # Get half of SGD's default learning rate (0.01 / 2 = 0.005)
        >>> lr = map_lr(0.5, "SGD")
        >>> print(lr)
        0.005

        >>> # Get 10x RMSprop's default learning rate (0.01 * 10 = 0.1)
        >>> lr = map_lr(10.0, "RMSprop")
        >>> print(lr)
        0.1

        Using unified learning rate without scaling:

        >>> # Use lr_unified directly (0.01)
        >>> lr = map_lr(0.01, "Adam", use_default_scale=False)
        >>> print(lr)
        0.01

        Practical example with model training:

        >>> import torch
        >>> import torch.nn as nn
        >>> from spotoptim.nn.linear_regressor import LinearRegressor
        >>> from spotoptim.utils.mapping import map_lr
        >>>
        >>> model = LinearRegressor(input_dim=10, output_dim=1)
        >>>
        >>> # Use unified learning rate of 0.5 for Adam (gives 0.0005)
        >>> lr_adam = map_lr(0.5, "Adam")
        >>> optimizer_adam = model.get_optimizer("Adam", lr=lr_adam)
        >>>
        >>> # Use same unified learning rate for SGD (gives 0.005)
        >>> lr_sgd = map_lr(0.5, "SGD")
        >>> optimizer_sgd = model.get_optimizer("SGD", lr=lr_sgd)

        Hyperparameter optimization example:

        >>> from spotoptim import SpotOptim
        >>> import numpy as np
        >>>
        >>> def train_model(X):
        ...     results = []
        ...     for params in X:
        ...         lr_unified = 10 ** params[0]  # Log scale: [-4, 0]
        ...         optimizer_name = params[1]     # Factor variable
        ...
        ...         # Map to optimizer-specific learning rate
        ...         lr_actual = map_lr(lr_unified, optimizer_name)
        ...
        ...         # Train model with this configuration
        ...         # ... training code ...
        ...         results.append(test_loss)
        ...     return np.array(results)
        >>>
        >>> optimizer = SpotOptim(
        ...     fun=train_model,
        ...     bounds=[(-4, 0), ("Adam", "SGD", "RMSprop")],
        ...     var_type=["num", "factor"],
        ...     max_iter=30
        ... )

    Note:
        - The unified learning rate provides a normalized scale across optimizers
        - A value of 1.0 always corresponds to the optimizer's PyTorch default
        - This enables fair comparison when optimizing over different optimizers
        - For log-scale optimization, use lr_unified = 10^x where x ∈ [-4, 2]
        - Default scaling is recommended for most use cases

    See Also:
        - PyTorch Optimizer Documentation: https://pytorch.org/docs/stable/optim.html
        - LinearRegressor.get_optimizer(): Convenience method using this mapping
    """
    if lr_unified <= 0:
        raise ValueError(
            f"Learning rate must be positive, got {lr_unified}. "
            f"Typical range is [0.001, 100.0] for unified scale."
        )

    if optimizer_name not in OPTIMIZER_DEFAULT_LR:
        supported = ", ".join(sorted(OPTIMIZER_DEFAULT_LR.keys()))
        raise ValueError(
            f"Optimizer '{optimizer_name}' not supported. "
            f"Supported optimizers: {supported}"
        )

    if not use_default_scale:
        return lr_unified

    # Scale by optimizer's default learning rate
    default_lr = OPTIMIZER_DEFAULT_LR[optimizer_name]
    return lr_unified * default_lr
