---
title: Variable Transformations for Search Space Scaling
sidebar_position: 6
eval: true
---

SpotOptim supports automatic variable transformations to improve optimization in scaled search spaces. Instead of manually handling transformations (e.g., log-scale for learning rates), you can specify transformations via the `var_trans` parameter, and SpotOptim handles everything internally.

## Overview

**What are Variable Transformations?**

Variable transformations allow you to specify how search space dimensions should be scaled during optimization:

1. **Original scale** (user interface): Input bounds, output results, plots
2. **Transformed scale** (internal): Surrogate modeling, acquisition optimization
3. **Automatic conversion**: SpotOptim handles all transformations transparently

**Module**: `spotoptim.SpotOptim`

**Key Features**:

- Define transformations via `var_trans` parameter: `["log10", "sqrt", None, ...]`
- Optimization occurs in transformed space (better for surrogate models)
- All external interfaces use original scale (intuitive for users)
- Supported transformations: log10, log/ln, sqrt, exp, square, cube, inv/reciprocal
- Mix transformed and non-transformed variables

## Why Use Transformations?

### Problem: Poorly Scaled Search Spaces

Some hyperparameters span multiple orders of magnitude:

- **Learning rates**: 0.0001 to 1.0 (4 orders of magnitude)
- **Regularization**: 0.001 to 100 (5 orders of magnitude)
- **Network sizes**: 10 to 1000 neurons

Direct optimization in these spaces is inefficient because:

1. Surrogate models struggle with extreme scales
2. Uniform sampling wastes evaluations in unimportant regions
3. Acquisition functions behave poorly with skewed distributions

### Solution: Logarithmic and Other Transformations

Transform the space for optimization while maintaining user-friendly interfaces:

```python
# Without transformations (manual approach)
bounds = [(-4, 0)]  # log10(lr): awkward for users
lr = 10 ** params[0]  # Manual transformation in objective

# With transformations (automatic)
bounds = [(0.0001, 1.0)]  # lr in natural scale
var_trans = ["log10"]  # SpotOptim handles transformation
lr = params[0]  # Already in original scale!
```

## Quick Start

### Basic Log-Scale Transformation

```{python}
from spotoptim import SpotOptim
import numpy as np

def objective_function(X):
    """Objective receives parameters in ORIGINAL scale."""
    results = []
    for params in X:
        lr = params[0]  # Already in [0.001, 0.1] - original scale!
        alpha = params[1]  # Already in [0.01, 1.0] - original scale!
        
        # Simulate model training
        score = (lr - 0.01)**2 + (alpha - 0.1)**2 + np.random.normal(0, 0.01)
        results.append(score)
    
    return np.array(results)

# Create optimizer with transformations
optimizer = SpotOptim(
    fun=objective_function,
    bounds=[
        (0.001, 0.1),    # learning rate (original scale)
        (0.01, 1.0)      # alpha (original scale)
    ],
    var_trans=["log10", "log10"],  # Both use log10 transformation
    var_name=["lr", "alpha"],
    max_iter=20,
    seed=42
)

# Run optimization
result = optimizer.optimize()

print(f"Best lr: {result.x[0]:.6f}")      # In original scale
print(f"Best alpha: {result.x[1]:.6f}")   # In original scale
print(f"Best score: {result.fun:.6f}")
```

## Supported Transformations

SpotOptim supports the following transformations:

| Transformation | Forward (x → t) | Inverse (t → x) | Use Case |
|----------------|-----------------|-----------------|----------|
| `"log10"` | t = log₁₀(x) | x = 10^t | Learning rates, regularization |
| `"log"` or `"ln"` | t = ln(x) | x = e^t | Natural exponential scales |
| `"sqrt"` | t = √x | x = t² | Moderate scaling |
| `"exp"` | t = e^x | x = ln(t) | Inverse of natural log |
| `"square"` | t = x² | x = √t | Inverse of sqrt |
| `"cube"` | t = x³ | x = ∛t | Strong scaling |
| `"inv"` or `"reciprocal"` | t = 1/x | x = 1/t | Reciprocal relationships |
| `None` or `"id"` | t = x | x = t | No transformation |

### Transformation Guidelines

**When to use `"log10"` or `"log"`:**

- Parameters spanning multiple orders of magnitude
- Learning rates: `(1e-5, 1e-1)` → uniform sampling in log space
- Regularization parameters: `(1e-6, 1e2)`
- Batch sizes, hidden units when range is large

**When to use `"sqrt"`:**

- Moderate scaling (1-2 orders of magnitude)
- Batch sizes: `(16, 512)`
- Number of neurons: `(32, 256)`

**When to use `"inv"` (reciprocal):**

- Inverse relationships (e.g., 1/temperature)
- When smaller values are more important

**When to use `None`:**

- Parameters with narrow ranges
- Already well-scaled parameters
- Categorical indices (use with `var_type=["factor"]`)

## Detailed Examples

### Example 1: Neural Network Hyperparameter Tuning

```{python}
import torch
import torch.nn as nn
from spotoptim import SpotOptim
from spotoptim.data import get_diabetes_dataloaders
from spotoptim.nn.linear_regressor import LinearRegressor
import numpy as np

def train_neural_network(X):
    """Train neural network with hyperparameters in original scale."""
    results = []
    
    for params in X:
        # All parameters in original scale
        hidden_size = int(params[0])  # [16, 256]
        num_layers = int(params[1])   # [1, 4]
        lr = params[2]                # [0.0001, 0.1]
        weight_decay = params[3]      # [1e-6, 0.01]
        
        print(f"Training: hidden={hidden_size}, layers={num_layers}, "
              f"lr={lr:.6f}, wd={weight_decay:.6f}")
        
        # Load data
        train_loader, test_loader, _ = get_diabetes_dataloaders(batch_size=32)
        
        # Create model
        model = LinearRegressor(
            input_dim=10,
            output_dim=1,
            l1=hidden_size,
            num_hidden_layers=num_layers,
            activation="ReLU",
            lr=lr
        )
        
        # Get optimizer with weight decay
        optimizer = torch.optim.Adam(model.parameters(), 
                                     lr=lr, 
                                     weight_decay=weight_decay)
        
        # Train
        model.train()
        for epoch in range(50):
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = nn.MSELoss()(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # Evaluate
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                loss = nn.MSELoss()(outputs, batch_y)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(test_loader)
        results.append(avg_loss)
    
    return np.array(results)

# Create optimizer with appropriate transformations
optimizer = SpotOptim(
    fun=train_neural_network,
    bounds=[
        (16, 256),           # hidden_size: moderate range
        (1, 4),              # num_layers: small range
        (0.0001, 0.1),       # lr: 3 orders of magnitude
        (1e-6, 0.01)         # weight_decay: 4 orders of magnitude
    ],
    var_trans=[
        "sqrt",              # sqrt for hidden_size
        None,                # no transformation for num_layers
        "log10",             # log10 for learning rate
        "log10"              # log10 for weight_decay
    ],
    var_type=["int", "int", "num", "num"],
    var_name=["hidden_size", "num_layers", "lr", "weight_decay"],
    max_iter=30,
    n_initial=10,
    seed=42
)

result = optimizer.optimize()

print("\nBest Configuration:")
print(f"  Hidden Size: {int(result.x[0])}")
print(f"  Num Layers: {int(result.x[1])}")
print(f"  Learning Rate: {result.x[2]:.6f}")
print(f"  Weight Decay: {result.x[3]:.8f}")
print(f"  Best Loss: {result.fun:.6f}")
```

### Example 2: Physics-Informed Neural Networks (PINNs)

```{python}
from spotoptim import SpotOptim
import numpy as np

def train_pinn(X):
    """Train PINN with hyperparameters in original scale."""
    results = []
    
    for params in X:
        neurons = int(params[0])      # [16, 128]
        layers = int(params[1])       # [1, 4]
        lr = params[2]                # [0.1, 10.0]
        alpha = params[3]             # [0.01, 1.0]
        
        # Simulate PINN training
        # In practice, this would train an actual PINN model
        val_error = 0.1 * (1/lr) + 0.05 * alpha + np.random.normal(0, 0.01)
        results.append(val_error)
    
    return np.array(results)

# Define optimization with transformations
optimizer = SpotOptim(
    fun=train_pinn,
    bounds=[
        (16, 128),           # neurons
        (1, 4),              # layers
        (0.1, 10.0),         # lr: covers 2 orders of magnitude
        (0.01, 1.0)          # alpha: covers 2 orders of magnitude
    ],
    var_trans=[None, None, "log10", "log10"],
    var_type=["int", "int", "num", "num"],
    var_name=["neurons", "layers", "lr", "alpha"],
    max_iter=20,
    seed=42
)

result = optimizer.optimize()
optimizer.print_best(result)
```

### Example 3: Mixing Transformations

```{python}
from spotoptim import SpotOptim
import numpy as np

def complex_objective(X):
    """Objective with multiple transformation types."""
    results = []
    
    for params in X:
        # All in original scale
        x1 = params[0]  # sqrt-transformed: [10, 1000]
        x2 = params[1]  # log10-transformed: [0.001, 1.0]
        x3 = params[2]  # no transformation: [-5, 5]
        x4 = params[3]  # reciprocal: [0.1, 10]
        
        # Complex function
        result = (
            (x1/100 - 5)**2 + 
            (np.log10(x2) + 1.5)**2 + 
            x3**2 + 
            (1/x4 - 0.2)**2
        )
        results.append(result)
    
    return np.array(results)

optimizer = SpotOptim(
    fun=complex_objective,
    bounds=[
        (10, 1000),      # x1: moderate range
        (0.001, 1.0),    # x2: log scale
        (-5, 5),         # x3: symmetric range
        (0.1, 10)        # x4: for reciprocal
    ],
    var_trans=["sqrt", "log10", None, "inv"],
    var_name=["x1_sqrt", "x2_log", "x3_linear", "x4_inv"],
    max_iter=30,
    seed=42
)

result = optimizer.optimize()
```

## Viewing Transformations in Tables

The transformation type is displayed in the "trans" column of both design and results tables:

### Design Table (Before Optimization)

```{python}
from spotoptim import SpotOptim
import numpy as np

optimizer = SpotOptim(
    fun=lambda X: np.sum(X**2, axis=1),
    bounds=[
        (0.001, 1.0),
        (0.01, 10.0),
        (10, 1000),
        (-5, 5)
    ],
    var_trans=["log10", "log", "sqrt", None],
    var_name=["lr", "alpha", "neurons", "bias"],
    max_iter=10
)

# Display design table
print(optimizer.print_design_table())
```

Output:
```
| name    | type   |    lower |    upper |   default | trans   |
|---------|--------|----------|----------|-----------|---------|
| lr      | num    |   0.0010 |   1.0000 |    0.5005 | log10   |
| alpha   | num    |   0.0100 |  10.0000 |    5.0050 | log     |
| neurons | num    |  10.0000 | 1000.0000 |  505.0000 | sqrt    |
| bias    | num    |  -5.0000 |   5.0000 |    0.0000 | -       |
```

### Results Table (After Optimization)

```{python}
result = optimizer.optimize()

# Display results with transformations
print(optimizer.print_results_table())
```

Output shows the "trans" column with transformation types, helping you understand which parameters were optimized in which scale.

## Internal Architecture

Understanding how transformations work internally can help debug issues and understand behavior:

### Flow Diagram

```
User Input (Original Scale)
    ↓
[Transform to Internal Scale]
    ↓
Optimization (Transformed Scale)
  • Initial design generation
  • Surrogate model fitting
  • Acquisition function optimization
    ↓
[Inverse Transform to Original Scale]
    ↓
Objective Function Evaluation (Original Scale)
    ↓
Storage & Results (Original Scale)
```

### Key Components

1. **Bounds Transformation** (`_transform_bounds()`):

   - Called during initialization
   - Transforms `_original_lower` and `_original_upper` → `lower` and `upper`
   - Updates `self.bounds` for internal use

2. **Forward Transformation** (`_transform_X()`):

   - Converts from original scale to transformed scale
   - Used before surrogate fitting
   - Used when comparing distances

3. **Inverse Transformation** (`_inverse_transform_X()`):

   - Converts from transformed scale to original scale
   - Used before function evaluation
   - Used when storing results

4. **Storage**:

   - `self.X_` stores in **original scale**
   - `self.best_x_` stores in **original scale**
   - All external-facing data in original scale

## Best Practices

### 1. Choose Appropriate Transformations

```python
# Good: Log scale for learning rate
bounds = [(1e-5, 1e-1)]
var_trans = ["log10"]

# Bad: No transformation for wide range
bounds = [(1e-5, 1e-1)]
var_trans = [None]  # Poor sampling distribution
```

### 2. Match Transformation to Range

```python
# Wide range (>3 orders of magnitude): use log
bounds = [(1e-6, 1e-2)]
var_trans = ["log10"]

# Moderate range (1-2 orders): use sqrt
bounds = [(10, 500)]
var_trans = ["sqrt"]

# Narrow range (<1 order): no transformation
bounds = [(-1, 1)]
var_trans = [None]
```

### 3. Validate Transformation Choice

```python
# Check if transformation makes sense
import numpy as np

# Original space
x_orig = np.linspace(0.001, 1.0, 10)
print("Original:", x_orig)

# Log10 transformed space
x_trans = np.log10(x_orig)
print("Transformed:", x_trans)
print("Range ratio:", np.ptp(x_trans) / np.ptp(x_orig))
# Should be much more uniform distribution
```

### 4. Combine with Variable Types

```python
# Mix transformations with variable types
optimizer = SpotOptim(
    fun=objective,
    bounds=[
        (10, 200),                          # int with sqrt
        ("ReLU", "Tanh", "Sigmoid"),        # factor (no transform)
        (0.0001, 0.1),                      # num with log10
        (0.01, 1.0)                         # num with log10
    ],
    var_type=["int", "factor", "num", "num"],
    var_trans=["sqrt", None, "log10", "log10"],
    var_name=["neurons", "activation", "lr", "dropout"]
)
```

## Troubleshooting

### Issue: Values Out of Bounds

**Problem**: Objective function receives values outside specified bounds.

**Solution**: This should not happen with transformations. If it does:

```python
# Check transformation is applied correctly
print(f"Original bounds: {optimizer._original_lower} to {optimizer._original_upper}")
print(f"Transformed bounds: {optimizer.lower} to {optimizer.upper}")
print(f"Transformations: {optimizer.var_trans}")
```

### Issue: Poor Optimization Performance

**Problem**: Optimization doesn't find good solutions.

**Possible causes**:

1. Wrong transformation type for the parameter scale
2. Transformation not needed (adding unnecessary complexity)
3. Bounds too wide or too narrow

**Solution**:
```python
# Try different transformations
for trans in [None, "log10", "sqrt"]:
    optimizer = SpotOptim(
        fun=objective,
        bounds=[(0.001, 1.0)],
        var_trans=[trans],
        max_iter=20,
        seed=42
    )
    result = optimizer.optimize()
    print(f"Transformation: {trans}, Best: {result.fun:.6f}")
```

### Issue: Transformation Not Applied

**Problem**: Transformation doesn't seem to affect optimization.

**Check**:
```python
# Verify var_trans length matches dimensions
print(f"Number of dimensions: {len(optimizer.bounds)}")
print(f"Number of transformations: {len(optimizer.var_trans)}")
# These must match!

# Check transformation is not None/"id"
print(f"Transformations: {optimizer.var_trans}")
```

## Comparison: Manual vs Automatic Transformations

### Manual Approach (Old Way)

```python
def objective_manual(X):
    """Manual transformation - error-prone!"""
    results = []
    for params in X:
        # Must remember to transform
        lr = 10 ** params[0]  # Was in log scale
        alpha = 10 ** params[1]  # Was in log scale
        
        # Use parameters
        score = compute_score(lr, alpha)
        results.append(score)
    return np.array(results)

# Bounds in log scale - confusing!
optimizer = SpotOptim(
    fun=objective_manual,
    bounds=[(-4, -1), (-2, 0)],  # log10 scale
    var_name=["log10_lr", "log10_alpha"]  # Confusing names
)

result = optimizer.optimize()
# Must transform back for interpretation
best_lr = 10 ** result.x[0]
best_alpha = 10 ** result.x[1]
```

### Automatic Approach (New Way)

```python
def objective_auto(X):
    """Automatic transformation - clean!"""
    results = []
    for params in X:
        # Already in original scale
        lr = params[0]
        alpha = params[1]
        
        # Use parameters directly
        score = compute_score(lr, alpha)
        results.append(score)
    return np.array(results)

# Bounds in natural scale - intuitive!
optimizer = SpotOptim(
    fun=objective_auto,
    bounds=[(0.0001, 0.1), (0.01, 1.0)],
    var_trans=["log10", "log10"],  # Specify transformation
    var_name=["lr", "alpha"]  # Natural names
)

result = optimizer.optimize()
# Results already in original scale
best_lr = result.x[0]
best_alpha = result.x[1]
```

## Summary

**Key Takeaways**:

1. ✅ Use `var_trans` to specify transformations for each dimension
2. ✅ Transformations improve optimization for poorly scaled spaces
3. ✅ All user interfaces (bounds, results, plots) use original scale
4. ✅ Optimization happens internally in transformed space
5. ✅ Common transformations: `"log10"` for learning rates, `"sqrt"` for moderate scaling
6. ✅ View transformations in tables with "trans" column

**When to Use**:

- Parameters spanning multiple orders of magnitude → `"log10"` or `"log"`
- Moderate scaling (1-2 orders) → `"sqrt"`
- Reciprocal relationships → `"inv"`
- Well-scaled parameters → `None`

**Benefits**:

- Better surrogate model performance
- More efficient sampling
- Improved optimization convergence
- User-friendly interface (no manual transformations in objective function)

---

**See Also**:

- [Variable Types Manual](var_type.md) - Integer, numeric, and factor types
- [Factor Variables Manual](factor_variables.md) - Categorical optimization
- [Reproducibility Manual](reproducibility.md) - Setting seeds for consistent results
