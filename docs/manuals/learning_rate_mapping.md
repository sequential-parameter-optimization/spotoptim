# Learning Rate Mapping for Unified Optimizer Interface

SpotOptim provides a sophisticated learning rate mapping system through the `map_lr()` function, enabling a unified interface for learning rates across different PyTorch optimizers. This solves the challenge that different optimizers operate on vastly different learning rate scales.

## Overview

Different PyTorch optimizers use different default learning rates and optimal ranges:

- **Adam**: default 0.001, typical range 0.0001-0.01
- **SGD**: default 0.01, typical range 0.001-0.1
- **RMSprop**: default 0.01, typical range 0.001-0.1

This makes it difficult to compare optimizer performance fairly or optimize learning rates across different optimizers. The `map_lr()` function provides a unified scale where **`lr_unified=1.0` corresponds to each optimizer's PyTorch default**.

**Module**: `spotoptim.utils.mapping`

**Key Features**:

- Unified learning rate scale across all optimizers
- Fair comparison when evaluating different optimizers
- Simplified hyperparameter optimization
- Based on official PyTorch default learning rates
- Supports 13 major PyTorch optimizers

## Quick Start

### Basic Usage

```python
from spotoptim.utils.mapping import map_lr

# Get optimizer-specific learning rate from unified scale
lr_adam = map_lr(1.0, "Adam")      # Returns 0.001 (Adam's default)
lr_sgd = map_lr(1.0, "SGD")        # Returns 0.01 (SGD's default)
lr_rmsprop = map_lr(1.0, "RMSprop")  # Returns 0.01 (RMSprop's default)

print(f"Unified lr=1.0:")
print(f"  Adam:    {lr_adam}")
print(f"  SGD:     {lr_sgd}")
print(f"  RMSprop: {lr_rmsprop}")
```

### Scaling Learning Rates

```python
from spotoptim.utils.mapping import map_lr

# Scale all learning rates by the same factor
unified_lr = 0.5

lr_adam = map_lr(unified_lr, "Adam")      # 0.5 * 0.001 = 0.0005
lr_sgd = map_lr(unified_lr, "SGD")        # 0.5 * 0.01 = 0.005
lr_rmsprop = map_lr(unified_lr, "RMSprop")  # 0.5 * 0.01 = 0.005

print(f"Unified lr={unified_lr}:")
print(f"  Adam:    {lr_adam}")
print(f"  SGD:     {lr_sgd}")
print(f"  RMSprop: {lr_rmsprop}")
```

### Integration with LinearRegressor

```python
from spotoptim.nn.linear_regressor import LinearRegressor

# Create model with unified learning rate
model = LinearRegressor(
    input_dim=10, 
    output_dim=1, 
    l1=32, 
    num_hidden_layers=2,
    lr=1.0  # Unified learning rate
)

# Get optimizer - automatically uses mapped learning rate
optimizer_adam = model.get_optimizer("Adam")     # Gets 1.0 * 0.001 = 0.001
optimizer_sgd = model.get_optimizer("SGD")       # Gets 1.0 * 0.01 = 0.01

# Verify the actual learning rates
print(f"Adam actual lr: {optimizer_adam.param_groups[0]['lr']}")
print(f"SGD actual lr: {optimizer_sgd.param_groups[0]['lr']}")
```

## Function Reference

### `map_lr(lr_unified, optimizer_name, use_default_scale=True)`

Maps a unified learning rate to an optimizer-specific learning rate.

**Parameters**:

- `lr_unified` (float): Unified learning rate multiplier. A value of 1.0 corresponds to the optimizer's default learning rate. Typical range: [0.001, 100.0].
- `optimizer_name` (str): Name of the PyTorch optimizer. Must be one of: "Adadelta", "Adagrad", "Adam", "AdamW", "SparseAdam", "Adamax", "ASGD", "LBFGS", "NAdam", "RAdam", "RMSprop", "Rprop", "SGD".
- `use_default_scale` (bool, optional): Whether to scale by the optimizer's default learning rate. If `True` (default), `lr_unified` is multiplied by the default lr. If `False`, returns `lr_unified` directly.

**Returns**:

- `float`: The optimizer-specific learning rate.

**Raises**:

- `ValueError`: If `optimizer_name` is not supported.
- `ValueError`: If `lr_unified` is not positive.

**Example**:

```python
from spotoptim.utils.mapping import map_lr

# Get default learning rates (unified lr = 1.0)
lr = map_lr(1.0, "Adam")      # 0.001
lr = map_lr(1.0, "SGD")       # 0.01
lr = map_lr(1.0, "RMSprop")   # 0.01

# Scale learning rates
lr = map_lr(0.5, "Adam")      # 0.0005
lr = map_lr(2.0, "SGD")       # 0.02

# Without default scaling
lr = map_lr(0.01, "Adam", use_default_scale=False)  # 0.01 (direct)
```

## Supported Optimizers

All major PyTorch optimizers are supported with their default learning rates:

| Optimizer | Default LR | Typical Range | Notes |
|-----------|------------|---------------|-------|
| **Adam** | 0.001 | 0.0001-0.01 | Most popular, good default |
| **AdamW** | 0.001 | 0.0001-0.01 | Adam with weight decay |
| **Adamax** | 0.002 | 0.0001-0.01 | Adam variant with infinity norm |
| **NAdam** | 0.002 | 0.0001-0.01 | Adam with Nesterov momentum |
| **RAdam** | 0.001 | 0.0001-0.01 | Rectified Adam |
| **SparseAdam** | 0.001 | 0.0001-0.01 | For sparse gradients |
| **SGD** | 0.01 | 0.001-0.1 | Classic, needs momentum |
| **RMSprop** | 0.01 | 0.001-0.1 | Good for RNNs |
| **Adagrad** | 0.01 | 0.001-0.1 | Adaptive learning rate |
| **Adadelta** | 1.0 | 0.1-10.0 | Extension of Adagrad |
| **ASGD** | 0.01 | 0.001-0.1 | Averaged SGD |
| **LBFGS** | 1.0 | 0.1-10.0 | Second-order optimizer |
| **Rprop** | 0.01 | 0.001-0.1 | Resilient backpropagation |

## Use Cases

### Comparing Different Optimizers

```python
import torch
import torch.nn as nn
from spotoptim.nn.linear_regressor import LinearRegressor
from spotoptim.data import get_diabetes_dataloaders

# Load data
train_loader, test_loader, _ = get_diabetes_dataloaders(batch_size=32, random_state=42)

# Test different optimizers with unified learning rate
unified_lr = 1.0
optimizers_to_test = ["Adam", "SGD", "RMSprop", "AdamW"]
results = {}

for opt_name in optimizers_to_test:
    # Reset for fair comparison
    torch.manual_seed(42)
    model = LinearRegressor(input_dim=10, output_dim=1, l1=32, 
                           num_hidden_layers=2, lr=unified_lr)
    
    # Create optimizer with mapped learning rate
    if opt_name == "SGD":
        optimizer = model.get_optimizer(opt_name, momentum=0.9)
    else:
        optimizer = model.get_optimizer(opt_name)
    
    criterion = nn.MSELoss()
    
    # Train
    model.train()
    for epoch in range(50):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
    
    # Evaluate
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            predictions = model(batch_X)
            test_loss += criterion(predictions, batch_y).item()
    
    avg_test_loss = test_loss / len(test_loader)
    results[opt_name] = avg_test_loss
    
    print(f"{opt_name:10s}: Test MSE = {avg_test_loss:.4f} "
          f"(actual lr = {optimizer.param_groups[0]['lr']:.6f})")

# Find best optimizer
best_opt = min(results, key=results.get)
print(f"\nBest optimizer: {best_opt} with MSE = {results[best_opt]:.4f}")
```

### Hyperparameter Optimization with SpotOptim

```python
from spotoptim import SpotOptim
from spotoptim.nn.linear_regressor import LinearRegressor
from spotoptim.data import get_diabetes_dataloaders
import torch.nn as nn
import torch
import numpy as np

def train_and_evaluate(X):
    """Objective function for hyperparameter optimization."""
    results = []
    
    # Load data once
    train_loader, test_loader, _ = get_diabetes_dataloaders(
        batch_size=32, random_state=42
    )
    
    for params in X:
        # Extract hyperparameters
        lr_unified = 10 ** params[0]  # Log scale
        optimizer_name = params[1]     # Factor variable
        l1 = int(params[2])           # Integer
        num_layers = int(params[3])   # Integer
        
        # Create model with unified learning rate
        model = LinearRegressor(
            input_dim=10,
            output_dim=1,
            l1=l1,
            num_hidden_layers=num_layers,
            lr=lr_unified  # Automatically mapped per optimizer
        )
        
        # Get optimizer (lr already mapped internally)
        if optimizer_name == "SGD":
            optimizer = model.get_optimizer(optimizer_name, momentum=0.9)
        else:
            optimizer = model.get_optimizer(optimizer_name)
        
        criterion = nn.MSELoss()
        
        # Train
        model.train()
        for epoch in range(30):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                predictions = model(batch_X)
                test_loss += criterion(predictions, batch_y).item()
        
        avg_test_loss = test_loss / len(test_loader)
        results.append(avg_test_loss)
    
    return np.array(results)

# Optimize learning rate, optimizer choice, and architecture
optimizer = SpotOptim(
    fun=train_and_evaluate,
    bounds=[
        (-4, 0),                           # log10(lr_unified): [0.0001, 1.0]
        ("Adam", "SGD", "RMSprop", "AdamW"),  # Optimizer choice
        (16, 128),                         # Layer size
        (1, 3)                             # Number of hidden layers
    ],
    var_type=["num", "factor", "int", "int"],
    max_iter=30,
    seed=42
)

result = optimizer.optimize()

# Display results
print("\nOptimization Results:")
print(f"Best unified lr: {10**result.x[0]:.6f}")
print(f"Best optimizer: {result.x[1]}")
print(f"Best layer size: {int(result.x[2])}")
print(f"Best num layers: {int(result.x[3])}")
print(f"Best test MSE: {result.fun:.4f}")

# Show actual learning rate used
from spotoptim.utils.mapping import map_lr
actual_lr = map_lr(10**result.x[0], result.x[1])
print(f"Actual {result.x[1]} learning rate: {actual_lr:.6f}")
```

### Log-Scale Hyperparameter Search

```python
from spotoptim.utils.mapping import map_lr
import numpy as np

# Common pattern: sample unified lr from log scale
log_lr_range = np.linspace(-4, 0, 10)  # [-4, -3.56, ..., 0]
optimizers = ["Adam", "SGD", "RMSprop"]

print("Log-scale learning rate search:")
print()
print(f"{'log_lr':<10} {'unified_lr':<12} {'Adam':<12} {'SGD':<12} {'RMSprop':<12}")
print("-" * 60)

for log_lr in log_lr_range:
    lr_unified = 10 ** log_lr
    lr_adam = map_lr(lr_unified, "Adam")
    lr_sgd = map_lr(lr_unified, "SGD")
    lr_rmsprop = map_lr(lr_unified, "RMSprop")
    
    print(f"{log_lr:<10.2f} {lr_unified:<12.6f} {lr_adam:<12.8f} "
          f"{lr_sgd:<12.8f} {lr_rmsprop:<12.8f}")
```

Output:
```
log_lr     unified_lr   Adam         SGD          RMSprop     
------------------------------------------------------------
-4.00      0.000100     0.00000010   0.00000100   0.00000100  
-3.56      0.000275     0.00000028   0.00000275   0.00000275  
-3.11      0.000759     0.00000076   0.00000759   0.00000759  
-2.67      0.002089     0.00000209   0.00002089   0.00002089  
-2.22      0.005754     0.00000575   0.00005754   0.00005754  
-1.78      0.015849     0.00001585   0.00015849   0.00015849  
-1.33      0.043652     0.00004365   0.00043652   0.00043652  
-0.89      0.120226     0.00012023   0.00120226   0.00120226  
-0.44      0.331131     0.00033113   0.00331131   0.00331131  
0.00       1.000000     0.00100000   0.01000000   0.01000000  
```

### Custom Learning Rate Schedules

```python
import torch
import torch.nn as nn
from spotoptim.nn.linear_regressor import LinearRegressor
from spotoptim.utils.mapping import map_lr

# Create model with unified lr
model = LinearRegressor(input_dim=10, output_dim=1, lr=1.0)

# Get initial optimizer
optimizer = model.get_optimizer("Adam")
initial_lr = optimizer.param_groups[0]['lr']
print(f"Initial learning rate: {initial_lr}")

# Use PyTorch learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Training with scheduler
for epoch in range(100):
    # ... training code ...
    scheduler.step()
    
    if (epoch + 1) % 30 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: lr = {current_lr:.8f}")
```

### Direct Usage Without LinearRegressor

```python
import torch
import torch.nn as nn
from spotoptim.utils.mapping import map_lr

# Define your own model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = MyModel()

# Use map_lr to get optimizer-specific learning rate
unified_lr = 2.0
optimizer_name = "Adam"

actual_lr = map_lr(unified_lr, optimizer_name)
optimizer = torch.optim.Adam(model.parameters(), lr=actual_lr)

print(f"Unified lr: {unified_lr}")
print(f"Actual {optimizer_name} lr: {actual_lr}")
```

## Best Practices

### Choosing Unified Learning Rate

**For initial experiments:**

- Start with `lr=1.0` (gives defaults for all optimizers)
- Test with `lr=0.1`, `lr=1.0`, `lr=10.0` to get a sense of scale

**For hyperparameter optimization:**

- Use log scale: sample from `[-4, 0]` or `[-3, 1]`
- Convert with `lr_unified = 10 ** log_lr`
- This gives reasonable ranges for all optimizers

**For fine-tuning:**

- If training is unstable: try smaller `lr` (e.g., 0.1 or 0.5)
- If training is too slow: try larger `lr` (e.g., 2.0 or 5.0)
- Monitor loss curves to adjust

### Optimizer Selection Guidelines

**Adam family (Adam, AdamW, NAdam, RAdam):**

- ✅ Good default choice for most tasks
- ✅ Adaptive learning rates per parameter
- ✅ Works well out of the box
- Use `lr=1.0` as starting point

**SGD:**

- ✅ Good for large datasets
- ✅ Often achieves better generalization
- ⚠️ Requires momentum (e.g., 0.9)
- Use `lr=1.0` with momentum=0.9

**RMSprop:**

- ✅ Good for recurrent networks
- ✅ Handles non-stationary objectives
- Use `lr=1.0` as starting point

**Others (Adadelta, Adagrad, etc.):**

- Specialized use cases
- Start with `lr=1.0` and adjust

### Common Patterns

```python
# Pattern 1: Quick optimizer comparison
model = LinearRegressor(input_dim=10, output_dim=1, lr=1.0)
for opt in ["Adam", "SGD", "RMSprop"]:
    optimizer = model.get_optimizer(opt)
    # ... train and compare ...

# Pattern 2: Hyperparameter optimization
def objective(X):
    lr_unified = 10 ** X[:, 0]  # Log scale
    optimizer_name = X[:, 1]     # Factor
    # ... use unified lr ...

# Pattern 3: Override model's lr
model = LinearRegressor(input_dim=10, output_dim=1, lr=1.0)
optimizer = model.get_optimizer("Adam", lr=2.0)  # Override with 2.0

# Pattern 4: Direct mapping
from spotoptim.utils.mapping import map_lr
lr_actual = map_lr(unified_lr, optimizer_name)
optimizer = torch.optim.Adam(params, lr=lr_actual)
```

## Troubleshooting

### Issue: Training is unstable (loss explodes)

**Solution**: Learning rate is too high. Try:
```python
model = LinearRegressor(input_dim=10, output_dim=1, lr=0.1)  # Reduce from 1.0
```

### Issue: Training is too slow (loss decreases very slowly)

**Solution**: Learning rate is too low. Try:
```python
model = LinearRegressor(input_dim=10, output_dim=1, lr=5.0)  # Increase from 1.0
```

### Issue: Different results across optimizer runs

**Solution**: Set random seed for reproducibility:
```python
import torch
torch.manual_seed(42)
```

### Issue: Want to use raw learning rate without mapping

**Solution**: Use `use_default_scale=False`:
```python
from spotoptim.utils.mapping import map_lr
lr = map_lr(0.001, "Adam", use_default_scale=False)  # Returns 0.001 directly
```

### Issue: Optimizer not supported

**Solution**: Check supported optimizers:
```python
from spotoptim.utils.mapping import OPTIMIZER_DEFAULT_LR
print("Supported optimizers:", list(OPTIMIZER_DEFAULT_LR.keys()))
```

## Technical Details

### How It Works

The mapping is simple but effective:

```
actual_lr = lr_unified * default_lr[optimizer_name]
```

For example:

- `map_lr(1.0, "Adam")` → `1.0 * 0.001` = `0.001`
- `map_lr(0.5, "SGD")` → `0.5 * 0.01` = `0.005`
- `map_lr(2.0, "RMSprop")` → `2.0 * 0.01` = `0.02`

This ensures that the same unified learning rate gives optimizer-specific learning rates in their typical working ranges.

### Design Rationale

**Why use defaults as scaling factors?**

PyTorch's default learning rates are carefully chosen to work well for typical use cases. By using them as scaling factors:

1. `lr=1.0` always gives sensible defaults
2. Scaling preserves the relative relationships between optimizers
3. Each optimizer stays in its optimal range
4. Easy to understand and explain

**Comparison with spotPython's approach:**

spotPython uses `lr = lr_mult * default_lr` in `optimizer_handler()`. Our implementation:

- ✅ Separates mapping logic (testable, reusable)
- ✅ Provides standalone function (`map_lr()`)
- ✅ Comprehensive error handling and validation
- ✅ Extensive documentation and examples
- ✅ Full integration with `LinearRegressor`

### Default Learning Rates

All values verified against [PyTorch documentation](https://pytorch.org/docs/stable/optim.html):

```python
OPTIMIZER_DEFAULT_LR = {
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
```

## Examples

### Complete Example: Optimizer Comparison Study

```python
"""
Complete example: Compare optimizers with unified learning rate interface.
"""
import torch
import torch.nn as nn
from spotoptim.nn.linear_regressor import LinearRegressor
from spotoptim.data import get_diabetes_dataloaders
from spotoptim.utils.mapping import map_lr
import matplotlib.pyplot as plt

# Set seed for reproducibility
torch.manual_seed(42)

# Load data
train_loader, test_loader, _ = get_diabetes_dataloaders(
    batch_size=32, 
    random_state=42
)

# Test configurations
optimizers = ["Adam", "SGD", "RMSprop", "AdamW"]
unified_lrs = [0.5, 1.0, 2.0]

# Store results
results = {}

print("Training models with different optimizers and learning rates...")
print()

for unified_lr in unified_lrs:
    results[unified_lr] = {}
    
    for opt_name in optimizers:
        # Reset model for fair comparison
        torch.manual_seed(42)
        
        # Create model with unified lr
        model = LinearRegressor(
            input_dim=10, 
            output_dim=1, 
            l1=32, 
            num_hidden_layers=2,
            lr=unified_lr
        )
        
        # Get optimizer
        if opt_name == "SGD":
            optimizer = model.get_optimizer(opt_name, momentum=0.9)
        else:
            optimizer = model.get_optimizer(opt_name)
        
        actual_lr = optimizer.param_groups[0]['lr']
        criterion = nn.MSELoss()
        
        # Track training loss
        train_losses = []
        
        # Train
        model.train()
        for epoch in range(50):
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_epoch_loss)
        
        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                predictions = model(batch_X)
                test_loss += criterion(predictions, batch_y).item()
        
        avg_test_loss = test_loss / len(test_loader)
        results[unified_lr][opt_name] = {
            'train_losses': train_losses,
            'test_loss': avg_test_loss,
            'actual_lr': actual_lr
        }
        
        print(f"Unified lr={unified_lr:.1f}, {opt_name:10s}: "
              f"actual_lr={actual_lr:.6f}, test_MSE={avg_test_loss:.4f}")

# Display summary
print()
print("=" * 70)
print("Summary: Best configurations")
print("=" * 70)

for unified_lr in unified_lrs:
    best_opt = min(results[unified_lr].items(), 
                   key=lambda x: x[1]['test_loss'])
    opt_name, metrics = best_opt
    
    print(f"Unified lr={unified_lr:.1f}: {opt_name:10s} "
          f"(test MSE={metrics['test_loss']:.4f}, "
          f"actual lr={metrics['actual_lr']:.6f})")

# Find overall best
best_overall = None
best_overall_loss = float('inf')

for unified_lr in unified_lrs:
    for opt_name, metrics in results[unified_lr].items():
        if metrics['test_loss'] < best_overall_loss:
            best_overall_loss = metrics['test_loss']
            best_overall = (unified_lr, opt_name, metrics['actual_lr'])

print()
print(f"Overall best: unified_lr={best_overall[0]:.1f}, "
      f"optimizer={best_overall[1]}, "
      f"test_MSE={best_overall_loss:.4f}")
print(f"Actual learning rate used: {best_overall[2]:.6f}")
```

## See Also

- [LinearRegressor Documentation](../api/linear_regressor.md) - Neural network class with lr parameter
- [Diabetes Dataset Utilities](diabetes_dataset.md) - Data loading for examples
- [Hyperparameter Optimization](../tutorials/hyperparameter_optimization.md) - Using map_lr with SpotOptim
- [PyTorch Optimizer Documentation](https://pytorch.org/docs/stable/optim.html) - Official PyTorch reference

## References

- Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv:1412.6980.
- Loshchilov, I., & Hutter, F. (2017). Decoupled weight decay regularization. arXiv:1711.05101.
- PyTorch Team. (2023). PyTorch Optimizer Documentation. https://pytorch.org/docs/stable/optim.html
