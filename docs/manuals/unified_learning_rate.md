# Unified Learning Rate Interface

This module provides a sophisticated unified learning rate interface for PyTorch optimizers through the `map_lr()` function and integration with `LinearRegressor`.

## Overview

Different PyTorch optimizers operate on vastly different learning rate scales:

- **Adam** typically uses lr ~ 0.0001-0.001
- **SGD** typically uses lr ~ 0.01-0.1
- **RMSprop** typically uses lr ~ 0.001-0.01

This makes it difficult to:

1. Compare optimizer performance fairly
2. Optimize learning rate as a hyperparameter across different optimizers
3. Switch between optimizers without retuning learning rates

The `map_lr()` function solves this by providing a unified learning rate scale where **lr=1.0 corresponds to each optimizer's PyTorch default**.

## Key Features

- ✅ **Unified Interface**: Single learning rate parameter works across all optimizers
- ✅ **Fair Comparison**: Same unified lr gives optimizer-specific optimal ranges
- ✅ **Hyperparameter Optimization**: Optimize one learning rate for multiple optimizers
- ✅ **Backward Compatible**: Existing code continues to work
- ✅ **Well-tested**: 36 comprehensive tests covering all use cases
- ✅ **Documented**: Extensive docstrings and examples

## Usage

### Basic Usage with LinearRegressor

```python
from spotoptim.nn.linear_regressor import LinearRegressor

# Create model with unified lr=1.0 (gives each optimizer its default)
model = LinearRegressor(input_dim=10, output_dim=1, lr=1.0)

# Adam gets 0.001 (its default)
optimizer_adam = model.get_optimizer("Adam")

# SGD gets 0.01 (its default)
optimizer_sgd = model.get_optimizer("SGD")

# RMSprop gets 0.01 (its default)
optimizer_rmsprop = model.get_optimizer("RMSprop")
```

### Using Custom Unified Learning Rate

```python
# Using lr=0.5 scales all optimizers by 0.5
model = LinearRegressor(input_dim=10, output_dim=1, lr=0.5)

optimizer_adam = model.get_optimizer("Adam")     # Gets 0.5 * 0.001 = 0.0005
optimizer_sgd = model.get_optimizer("SGD")       # Gets 0.5 * 0.01 = 0.005
```

### Direct Use of map_lr()

```python
from spotoptim.utils.mapping import map_lr

# Map unified lr to optimizer-specific lr
lr_adam = map_lr(1.0, "Adam")      # Returns 0.001
lr_sgd = map_lr(1.0, "SGD")        # Returns 0.01
lr_rmsprop = map_lr(1.0, "RMSprop")  # Returns 0.01

# Scale by 2x
lr_adam = map_lr(2.0, "Adam")      # Returns 0.002
lr_sgd = map_lr(2.0, "SGD")        # Returns 0.02
```

### Hyperparameter Optimization

```python
from spotoptim import SpotOptim
import numpy as np

def train_model(X):
    results = []
    for params in X:
        lr_unified = 10 ** params[0]  # Log scale: [-4, 0]
        optimizer_name = params[1]     # Factor: "Adam", "SGD", "RMSprop"
        
        # Create model with unified lr - automatically scaled per optimizer
        model = LinearRegressor(input_dim=10, output_dim=1, lr=lr_unified)
        optimizer = model.get_optimizer(optimizer_name)
        
        # Train and evaluate
        # ... training code ...
        results.append(test_loss)
    return np.array(results)

# Optimize unified lr across different optimizers
optimizer = SpotOptim(
    fun=train_model,
    bounds=[(-4, 0), ("Adam", "SGD", "RMSprop")],
    var_type=["num", "factor"],
    max_iter=30
)
result = optimizer.optimize()
```

## Supported Optimizers

All major PyTorch optimizers are supported with their default learning rates:

| Optimizer | Default LR | Typical Range |
|-----------|------------|---------------|
| Adam      | 0.001      | 0.0001-0.01   |
| AdamW     | 0.001      | 0.0001-0.01   |
| Adamax    | 0.002      | 0.0001-0.01   |
| NAdam     | 0.002      | 0.0001-0.01   |
| RAdam     | 0.001      | 0.0001-0.01   |
| SGD       | 0.01       | 0.001-0.1     |
| RMSprop   | 0.01       | 0.001-0.1     |
| Adagrad   | 0.01       | 0.001-0.1     |
| Adadelta  | 1.0        | 0.1-10.0      |
| ASGD      | 0.01       | 0.001-0.1     |
| LBFGS     | 1.0        | 0.1-10.0      |
| Rprop     | 0.01       | 0.001-0.1     |

## API Reference

### `map_lr(lr_unified, optimizer_name, use_default_scale=True)`

Maps a unified learning rate to an optimizer-specific learning rate.

**Parameters:**

- `lr_unified` (float): Unified learning rate multiplier. Typical range: [0.001, 100.0]
- `optimizer_name` (str): Name of the PyTorch optimizer
- `use_default_scale` (bool): Whether to scale by optimizer's default (default: True)

**Returns:**

- `float`: The optimizer-specific learning rate

**Example:**
```python
lr = map_lr(1.0, "Adam")  # Returns 0.001 (Adam's default)
lr = map_lr(0.5, "SGD")   # Returns 0.005 (0.5 * SGD's default)
```

### `LinearRegressor(..., lr=1.0)`

**Parameter:**

- `lr` (float): Unified learning rate multiplier. Default: 1.0

**New Behavior in `get_optimizer()`:**

- If `lr` is not specified, uses `self.lr`
- Automatically maps unified lr to optimizer-specific lr
- Can override model's lr by passing `lr` parameter

## Design Rationale

### Why Unified Learning Rates?

The approach is based on spotPython's `optimizer_handler()` but improved:

1. **Separation of Concerns**: Mapping logic in separate, testable module
2. **Flexibility**: Can be used independently or integrated with models
3. **Transparency**: Clear mapping based on PyTorch defaults
4. **Extensibility**: Easy to add new optimizers
5. **Type Safety**: Comprehensive error handling and validation

### Comparison with spotPython

| Feature | spotPython | spotoptim |
|---------|-----------|-----------|
| Approach | `lr_mult * default_lr` | `map_lr(lr_unified, optimizer)` |
| Module | `optimizer_handler()` | `map_lr()` + integration |
| Testing | Minimal | 36 comprehensive tests |
| Documentation | Basic | Extensive with examples |
| Reusability | Coupled | Standalone function |
| Error Handling | Basic | Comprehensive validation |

### Log-scale Optimization

For hyperparameter optimization, use log-scale for unified lr:

```python
# Sample from log10 scale [-4, 0]
log_lr = -2.5  # Sampled value
lr_unified = 10 ** log_lr  # 0.00316

# Map to optimizer-specific
lr_adam = map_lr(lr_unified, "Adam")  # 0.00316 * 0.001 = 0.00000316
lr_sgd = map_lr(lr_unified, "SGD")    # 0.00316 * 0.01 = 0.0000316
```

This gives a reasonable search range across all optimizers.

## Examples

See `examples/unified_learning_rate_demo.py` for comprehensive examples including:
1. Basic unified interface usage
2. Custom unified learning rates
3. Training with different optimizers
4. Direct use of map_lr()
5. Log-scale hyperparameter optimization
6. Complete hyperparameter optimization scenario


## References

- [PyTorch Optimizer Documentation](https://pytorch.org/docs/stable/optim.html)
- spotPython's `optimizer_handler()` function (inspiration)
- [Hyperparameter Optimization Best Practices](https://arxiv.org/abs/2003.05689)

## Contributing

When adding new optimizers:

1. Add default lr to `OPTIMIZER_DEFAULT_LR` dict in `mapping.py`
2. Verify the default against PyTorch documentation
3. Add tests in `test_mapping.py`
4. Update this README

## License

Same as spotoptim package (see main LICENSE file).
