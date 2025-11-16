---
title: Factor Variables for Categorical Hyperparameters
sidebar_position: 5
eval: true
---

SpotOptim supports factor variables for optimizing categorical hyperparameters, such as activation functions, optimizers, or any discrete string-based choices. Factor variables are automatically converted between string values (external interface) and integers (internal optimization), making categorical optimization seamless.

## Overview

**What are Factor Variables?**

Factor variables allow you to specify categorical choices as tuples of strings in the bounds. SpotOptim handles the conversion:

1. **String tuples in bounds** → Internal integer mapping (0, 1, 2, ...)
2. **Optimization uses integers** internally for surrogate modeling
3. **Objective function receives strings** after automatic conversion
4. **Results return strings** (not integers)

**Module**: `spotoptim.SpotOptim`

**Key Features**:

- Define categorical choices as string tuples: `("ReLU", "Sigmoid", "Tanh")`
- Automatic integer↔string conversion
- Seamless integration with neural network hyperparameters
- Mix factor variables with numeric/integer variables

## Quick Start

### Basic Factor Variable Usage


```{python}
from spotoptim import SpotOptim
import numpy as np

def objective_function(X):
    """Objective function receives string values."""
    results = []
    for params in X:
        activation = params[0]  # This is a string!
        print(f"Testing activation: {activation}")
        
        # Simple scoring based on activation choice (for demonstration)
        # In real use, you would train a model and return actual performance
        scores = {
            "ReLU": 3500.0,
            "Sigmoid": 4200.0,
            "Tanh": 3800.0,
            "LeakyReLU": 3600.0
        }
        score = scores.get(activation, 5000.0) + np.random.normal(0, 100)
        results.append(score)
    return np.array(results)  # Return numpy array

# Define bounds with factor variable
optimizer = SpotOptim(
    fun=objective_function,
    bounds=[("ReLU", "Sigmoid", "Tanh", "LeakyReLU")],
    var_type=["factor"],
    max_iter=20,
    seed=42
)

result = optimizer.optimize()
print(f"\nBest activation: {result.x[0]}")  # Returns string, e.g., "ReLU"
print(f"Best score: {result.fun:.4f}")
```

### Neural Network Activation Function Optimization

```{python}
import torch
import torch.nn as nn
from spotoptim import SpotOptim
from spotoptim.data import get_diabetes_dataloaders
from spotoptim.nn.linear_regressor import LinearRegressor
import numpy as np

def train_and_evaluate(X):
    """Train models with different activation functions."""
    results = []
    
    for params in X:
        activation = params[0]  # String: "ReLU", "Sigmoid", etc.
        
        # Load data
        train_loader, test_loader, _ = get_diabetes_dataloaders()
        
        # Create model with the activation function
        model = LinearRegressor(
            input_dim=10,
            output_dim=1,
            l1=64,
            num_hidden_layers=2,
            activation=activation  # Pass string directly!
        )
        
        # Train model
        optimizer = model.get_optimizer("Adam", lr=0.01)
        criterion = nn.MSELoss()
        
        for epoch in range(50):
            model.train()
            for batch_X, batch_y in train_loader:
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Evaluate
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                predictions = model(batch_X)
                test_loss += criterion(predictions, batch_y).item()
        
        avg_loss = test_loss / len(test_loader)
        results.append(avg_loss)
    
    return np.array(results)  # Return numpy array

# Optimize activation function choice
optimizer = SpotOptim(
    fun=train_and_evaluate,
    bounds=[("ReLU", "Sigmoid", "Tanh", "LeakyReLU", "ELU")],
    var_type=["factor"],
    max_iter=30
)

result = optimizer.optimize()
print(f"Best activation function: {result.x[0]}")
print(f"Best test MSE: {result.fun:.4f}")
```

## Mixed Variable Types

### Combining Factor, Integer, and Continuous Variables

```{python}
import numpy as np
import torch
import torch.nn as nn
from spotoptim import SpotOptim
from spotoptim.data import get_diabetes_dataloaders
from spotoptim.nn.linear_regressor import LinearRegressor

def comprehensive_optimization(X):
    """Optimize learning rate, layer size, depth, and activation."""
    results = []
    
    for params in X:
        log_lr = params[0]      # Continuous (log scale)
        l1 = int(params[1])     # Integer
        n_layers = int(params[2])  # Integer
        activation = params[3]   # Factor (string)
        
        lr = 10 ** log_lr  # Convert from log scale
        
        print(f"lr={lr:.6f}, l1={l1}, layers={n_layers}, activation={activation}")
        
        # Load data
        train_loader, test_loader, _ = get_diabetes_dataloaders(
            batch_size=32,
            random_state=42
        )
        
        # Create model
        model = LinearRegressor(
            input_dim=10,
            output_dim=1,
            l1=l1,
            num_hidden_layers=n_layers,
            activation=activation
        )
        
        # Train
        optimizer = model.get_optimizer("Adam", lr=lr)
        criterion = nn.MSELoss()
        
        for epoch in range(30):
            model.train()
            for batch_X, batch_y in train_loader:
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Evaluate
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                predictions = model(batch_X)
                test_loss += criterion(predictions, batch_y).item()
        
        results.append(test_loss / len(test_loader))
    
    return np.array(results)

# Optimize all four hyperparameters simultaneously
optimizer = SpotOptim(
    fun=comprehensive_optimization,
    bounds=[
        (-4, -2),                                    # log10(learning_rate)
        (16, 128),                                   # l1 (neurons per layer)
        (0, 4),                                      # num_hidden_layers
        ("ReLU", "Sigmoid", "Tanh", "LeakyReLU")   # activation function
    ],
    var_type=["num", "int", "int", "factor"],
    max_iter=50
)

result = optimizer.optimize()

# Results contain original string values
print("\nOptimization Results:")
print(f"Best learning rate: {10**result.x[0]:.6f}")
print(f"Best layer size: {int(result.x[1])}")
print(f"Best num layers: {int(result.x[2])}")
print(f"Best activation: {result.x[3]}")  # String value!
print(f"Best test MSE: {result.fun:.4f}")
```

## Multiple Factor Variables

### Optimizing Both Activation and Optimizer

```{python}
from spotoptim import SpotOptim
from spotoptim.data import get_diabetes_dataloaders
from spotoptim.nn.linear_regressor import LinearRegressor
import torch.nn as nn
import numpy as np

def optimize_activation_and_optimizer(X):
    """Optimize both activation function and optimizer choice."""
    results = []
    
    for params in X:
        activation = params[0]      # Factor variable 1
        optimizer_name = params[1]  # Factor variable 2
        lr = 10 ** params[2]        # Continuous variable
        
        train_loader, test_loader, _ = get_diabetes_dataloaders()
        
        model = LinearRegressor(
            input_dim=10,
            output_dim=1,
            l1=64,
            num_hidden_layers=2,
            activation=activation
        )
        
        # Use the optimizer string
        optimizer = model.get_optimizer(optimizer_name, lr=lr)
        criterion = nn.MSELoss()
        
        # Train
        for epoch in range(30):
            model.train()
            for batch_X, batch_y in train_loader:
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Evaluate
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                predictions = model(batch_X)
                test_loss += criterion(predictions, batch_y).item()
        
        results.append(test_loss / len(test_loader))
    
    return np.array(results)  # Return numpy array

# Two factor variables + one continuous
opt = SpotOptim(
    fun=optimize_activation_and_optimizer,
    bounds=[
        ("ReLU", "Tanh", "Sigmoid", "LeakyReLU"),    # Activation
        ("Adam", "SGD", "RMSprop", "AdamW"),         # Optimizer
        (-4, -2)                                      # log10(lr)
    ],
    var_type=["factor", "factor", "num"],
    max_iter=40
)

result = opt.optimize()
print(f"Best activation: {result.x[0]}")
print(f"Best optimizer: {result.x[1]}")
print(f"Best learning rate: {10**result.x[2]:.6f}")
```

## Advanced Usage

### Custom Categorical Choices

Factor variables work with any string values, not just activation functions:

```{python}
from spotoptim import SpotOptim
import numpy as np

def train_model_with_config(dropout_policy, batch_norm, weight_init):
    """Simulate model training with different configurations."""
    # In real use, this would train an actual model
    # Here we return synthetic scores for demonstration
    base_score = 3000.0
    
    # Dropout impact
    dropout_scores = {"none": 200, "light": 0, "heavy": 100}
    # Batch norm impact
    bn_scores = {"before": -50, "after": 0, "none": 150}
    # Weight init impact
    init_scores = {"xavier": 0, "kaiming": -30, "normal": 100}
    
    score = (base_score + 
             dropout_scores.get(dropout_policy, 0) + 
             bn_scores.get(batch_norm, 0) + 
             init_scores.get(weight_init, 0) +
             np.random.normal(0, 50))
    
    return score

def train_with_config(X):
    """Objective function with various categorical choices."""
    results = []
    
    for params in X:
        dropout_policy = params[0]  # "none", "light", "heavy"
        batch_norm = params[1]       # "before", "after", "none"
        weight_init = params[2]      # "xavier", "kaiming", "normal"
        
        # Use these strings to configure your model
        score = train_model_with_config(
            dropout_policy=dropout_policy,
            batch_norm=batch_norm,
            weight_init=weight_init
        )
        results.append(score)
    
    return np.array(results)  # Return numpy array

optimizer = SpotOptim(
    fun=train_with_config,
    bounds=[
        ("none", "light", "heavy"),           # Dropout policy
        ("before", "after", "none"),          # Batch norm position
        ("xavier", "kaiming", "normal")       # Weight initialization
    ],
    var_type=["factor", "factor", "factor"],
    max_iter=25,
    seed=42
)

result = optimizer.optimize()
print("Best configuration:")
print(f"  Dropout: {result.x[0]}")
print(f"  Batch norm: {result.x[1]}")
print(f"  Weight init: {result.x[2]}")
print(f"  Score: {result.fun:.4f}")
```

### Viewing All Evaluated Configurations

```{python}
import torch
import torch.nn as nn
from spotoptim import SpotOptim
from spotoptim.data import get_diabetes_dataloaders
from spotoptim.nn.linear_regressor import LinearRegressor
import numpy as np

def train_and_evaluate(X):
    """Train models with different activation functions."""
    results = []
    
    for params in X:
        l1 = int(params[0])         # Integer: layer size
        activation = params[1]       # String: activation function
        
        # Load data
        train_loader, test_loader, _ = get_diabetes_dataloaders()
        
        # Create model with the activation function
        model = LinearRegressor(
            input_dim=10,
            output_dim=1,
            l1=l1,
            num_hidden_layers=2,
            activation=activation  # Pass string directly!
        )
        
        # Train model
        optimizer = model.get_optimizer("Adam", lr=0.01)
        criterion = nn.MSELoss()
        
        for epoch in range(50):
            model.train()
            for batch_X, batch_y in train_loader:
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Evaluate
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                predictions = model(batch_X)
                test_loss += criterion(predictions, batch_y).item()
        
        avg_loss = test_loss / len(test_loader)
        results.append(avg_loss)
    
    return np.array(results)

optimizer = SpotOptim(
    fun=train_and_evaluate,
    bounds=[
        (16, 128),                                   # Layer size
        ("ReLU", "Sigmoid", "Tanh", "LeakyReLU")   # Activation
    ],
    var_type=["int", "factor"],  # IMPORTANT: Specify variable types!
    max_iter=30,
    seed=42
)

result = optimizer.optimize()

# Access all evaluated configurations
print("\nAll evaluated configurations:")
print("Layer Size | Activation | Test MSE")
print("-" * 42)
for i in range(min(10, len(result.X))):  # Show first 10
    l1 = int(result.X[i, 0])
    activation = result.X[i, 1]  # String value!
    loss = result.y[i]
    print(f"{l1:10d} | {activation:10s} | {loss:.4f}")

# Find top 5 configurations
sorted_indices = result.y.argsort()[:5]
print("\nTop 5 configurations:")
for idx in sorted_indices:
    print(f"l1={int(result.X[idx, 0]):3d}, "
          f"activation={result.X[idx, 1]:10s}, "
          f"MSE={result.y[idx]:.4f}")
```

## How It Works

### Internal Mechanism

SpotOptim handles factor variables through automatic conversion:

1. **Initialization**: String tuples in bounds are detected
   ```{python}
   bounds = [("ReLU", "Sigmoid", "Tanh")]
   # Internally mapped to: {0: "ReLU", 1: "Sigmoid", 2: "Tanh"}
   # Bounds become: [(0, 2)]
   ```

2. **Sampling**: Initial design samples from `[0, n_levels-1]` and rounds to integers
   ```{python}
   # Samples might be: [0.3, 1.8, 2.1]
   # After rounding: [0, 2, 2]
   ```

3. **Evaluation**: Before calling objective function, integers → strings
   ```{python}
   # [0, 2, 2] → ["ReLU", "Tanh", "Tanh"]
   # Objective function receives strings
   ```

4. **Optimization**: Surrogate model works with integers `[0, n_levels-1]`

5. **Results**: Final results mapped back to strings
   ```{python}
   result.x[0]  # Returns "ReLU", not 0
   result.X     # All rows contain strings for factor variables
   ```

### Variable Type Auto-Detection

If you don't specify `var_type`, SpotOptim automatically detects factor variables:

```{python}
# Example 1: Explicit var_type (recommended)
# This shows the syntax - replace my_function with your actual function

# optimizer = SpotOptim(
#     fun=my_function,
#     bounds=[(-4, -2), ("ReLU", "Tanh")],
#     var_type=["num", "factor"]  # Explicit
# )

# Example 2: Auto-detection (works but less explicit)
# optimizer = SpotOptim(
#     fun=my_function,
#     bounds=[(-4, -2), ("ReLU", "Tanh")]
#     # var_type automatically set to ["float", "factor"]
# )

# Here's a working example:
from spotoptim import SpotOptim
import numpy as np

def demo_function(X):
    results = []
    for params in X:
        lr = 10 ** params[0]  # Continuous parameter
        activation = params[1]  # Factor parameter
        score = 3000 + lr * 100 + {"ReLU": 0, "Tanh": 50}.get(activation, 100)
        results.append(score + np.random.normal(0, 10))
    return np.array(results)

# With explicit var_type (recommended)
optimizer = SpotOptim(
    fun=demo_function,
    bounds=[(-4, -2), ("ReLU", "Tanh")],
    var_type=["num", "factor"],  # Explicit is clearer
    max_iter=10,
    seed=42
)

result = optimizer.optimize()
print(f"Best lr: {10**result.x[0]:.6f}, Best activation: {result.x[1]}")
```

## Complete Example: Full Workflow

```{python}
"""
Complete example: Neural network hyperparameter optimization with factor variables.
"""
import numpy as np
import torch
import torch.nn as nn
from spotoptim import SpotOptim
from spotoptim.data import get_diabetes_dataloaders
from spotoptim.nn.linear_regressor import LinearRegressor


def objective_function(X):
    """Train and evaluate models with given hyperparameters."""
    results = []
    
    for params in X:
        # Extract hyperparameters
        log_lr = params[0]
        l1 = int(params[1])
        num_layers = int(params[2])
        activation = params[3]  # String!
        
        lr = 10 ** log_lr
        
        print(f"Testing: lr={lr:.6f}, l1={l1}, layers={num_layers}, "
              f"activation={activation}")
        
        # Load data
        train_loader, test_loader, _ = get_diabetes_dataloaders(
            test_size=0.2,
            batch_size=32,
            random_state=42
        )
        
        # Create and train model
        model = LinearRegressor(
            input_dim=10,
            output_dim=1,
            l1=l1,
            num_hidden_layers=num_layers,
            activation=activation
        )
        
        optimizer = model.get_optimizer("Adam", lr=lr)
        criterion = nn.MSELoss()
        
        # Training loop
        num_epochs = 30
        for epoch in range(num_epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Evaluation
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                test_loss += loss.item()
        
        avg_test_loss = test_loss / len(test_loader)
        results.append(avg_test_loss)
        print(f"  → Test MSE: {avg_test_loss:.4f}")
    
    return np.array(results)


def main():
    print("=" * 80)
    print("Neural Network Hyperparameter Optimization with Factor Variables")
    print("=" * 80)
    
    # Define optimization problem
    optimizer = SpotOptim(
        fun=objective_function,
        bounds=[
            (-4, -2),                                    # log10(learning_rate)
            (16, 128),                                   # l1 (neurons)
            (0, 4),                                      # num_hidden_layers
            ("ReLU", "Sigmoid", "Tanh", "LeakyReLU")   # activation (factor!)
        ],
        var_type=["num", "int", "int", "factor"],
        max_iter=50,
        seed=42
    )
    
    # Run optimization
    print("\nStarting optimization...")
    result = optimizer.optimize()
    
    # Display results
    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)
    print(f"Best learning rate: {10**result.x[0]:.6f}")
    print(f"Best layer size (l1): {int(result.x[1])}")
    print(f"Best num hidden layers: {int(result.x[2])}")
    print(f"Best activation function: {result.x[3]}")  # String value!
    print(f"Best test MSE: {result.fun:.4f}")
    
    # Show top 5 configurations
    print("\n" + "=" * 80)
    print("TOP 5 CONFIGURATIONS")
    print("=" * 80)
    sorted_indices = result.y.argsort()[:5]
    print(f"{'Rank':<6} {'LR':<12} {'L1':<6} {'Layers':<8} "
          f"{'Activation':<12} {'MSE':<10}")
    print("-" * 80)
    for rank, idx in enumerate(sorted_indices, 1):
        lr = 10 ** result.X[idx, 0]
        l1 = int(result.X[idx, 1])
        layers = int(result.X[idx, 2])
        activation = result.X[idx, 3]
        mse = result.y[idx]
        print(f"{rank:<6} {lr:<12.6f} {l1:<6} {layers:<8} "
              f"{activation:<12} {mse:<10.4f}")
    
    # Train final model with best configuration
    print("\n" + "=" * 80)
    print("TRAINING FINAL MODEL")
    print("=" * 80)
    
    best_lr = 10 ** result.x[0]
    best_l1 = int(result.x[1])
    best_layers = int(result.x[2])
    best_activation = result.x[3]
    
    print(f"Configuration: lr={best_lr:.6f}, l1={best_l1}, "
          f"layers={best_layers}, activation={best_activation}")
    
    train_loader, test_loader, _ = get_diabetes_dataloaders(
        test_size=0.2,
        batch_size=32,
        random_state=42
    )
    
    final_model = LinearRegressor(
        input_dim=10,
        output_dim=1,
        l1=best_l1,
        num_hidden_layers=best_layers,
        activation=best_activation
    )
    
    optimizer_final = final_model.get_optimizer("Adam", lr=best_lr)
    criterion = nn.MSELoss()
    
    # Extended training
    num_epochs = 100
    print(f"\nTraining for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        final_model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            predictions = final_model(batch_X)
            loss = criterion(predictions, batch_y)
            optimizer_final.zero_grad()
            loss.backward()
            optimizer_final.step()
            train_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            avg_train_loss = train_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}: Train MSE = {avg_train_loss:.4f}")
    
    # Final evaluation
    final_model.eval()
    final_test_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            predictions = final_model(batch_X)
            final_test_loss += criterion(predictions, batch_y).item()
    
    final_avg_loss = final_test_loss / len(test_loader)
    print(f"\nFinal Test MSE: {final_avg_loss:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
```

## Best Practices

### Do's

✅ **Use descriptive string values**

```{python}
bounds=[("xavier_uniform", "kaiming_normal", "orthogonal")]
```

✅ **Explicitly specify var_type for clarity**

```{python}
var_type=["num", "int", "factor"]
```

✅ **Access results as strings**

```{python}
# Example: Accessing factor variable results as strings
# (This assumes you've run an optimization with activation as a factor variable)

# If you have a result from the previous examples:
# best_activation = result.x[3]  # For 4-parameter optimization
# Or for simpler cases:
# best_activation = result.x[0]  # For single-parameter optimization

# Example with inline optimization:
from spotoptim import SpotOptim
import numpy as np

def quick_test(X):
    results = []
    for params in X:
        activation = params[0]
        score = {"ReLU": 3500, "Tanh": 3600}.get(activation, 4000)
        results.append(score + np.random.normal(0, 50))
    return np.array(results)

opt = SpotOptim(
    fun=quick_test,
    bounds=[("ReLU", "Tanh")],
    var_type=["factor"],
    max_iter=10,
    seed=42
)
result = opt.optimize()

# Access as string - this is the correct way
best_activation = result.x[0]  # String value like "ReLU"
print(f"Best activation: {best_activation} (type: {type(best_activation).__name__})")

# You can use it directly in your model
# model = LinearRegressor(activation=best_activation)
```

✅ **Mix factor variables with numeric/integer variables**

```{python}
bounds=[(-4, -2), (16, 128), ("ReLU", "Tanh")]
var_type=["num", "int", "factor"]
```

### Don'ts

❌ **Don't use integers in factor bounds**

```{python}
# Wrong: Use strings, not integers
bounds=[(0, 1, 2)]  # Wrong!
bounds=[("ReLU", "Sigmoid", "Tanh")]  # Correct!
```

❌ **Don't expect integers in objective function**

```{python}
def objective(X):
    activation = X[0][2]
    # activation is a string, not an integer!
    # Don't do: if activation == 0:  # Wrong!
    # Do: if activation == "ReLU":   # Correct!
```

❌ **Don't manually convert factor variables**

```{python}
# SpotOptim handles conversion automatically
# Don't do manual mapping in your objective function
```

❌ **Don't use empty tuples**

```{python}
# Wrong: Empty tuple
bounds=[()]

# Correct: At least one string
bounds=[("ReLU",)]  # Single choice (will be treated as fixed)
```

## Troubleshooting

### Common Issues

**Issue**: Objective function receives integers instead of strings

**Solution**: Ensure you're using the latest version of SpotOptim with factor variable support. Factor variables are automatically converted before calling the objective function.

---

**Issue**: `ValueError: could not convert string to float`

**Solution**: This occurs if there's a version mismatch. Update SpotOptim to ensure the object array conversion is implemented correctly.

---

**Issue**: Results show integers instead of strings

**Solution**: Check that you're accessing `result.x` (mapped values) instead of internal arrays. The result object automatically maps factor variables to their original strings.

---

**Issue**: Single-level factor variables cause dimension reduction

**Behavior**: If a factor variable has only one choice, e.g., `("ReLU",)`, SpotOptim treats it as a fixed dimension and may reduce the dimensionality. This is expected behavior.

**Solution**: Use at least two choices for optimization, or remove single-choice dimensions from bounds.

## Summary

Factor variables in SpotOptim enable:

- ✅ **Categorical optimization**: Optimize over discrete string choices
- ✅ **Automatic conversion**: Seamless integer↔string mapping
- ✅ **Neural network hyperparameters**: Optimize activation functions, optimizers, etc.
- ✅ **Mixed variable types**: Combine with continuous and integer variables
- ✅ **Clean interface**: Objective functions work with strings directly
- ✅ **String results**: Final results contain original string values

Factor variables make categorical hyperparameter optimization as easy as continuous optimization!

## See Also

- [LinearRegressor Documentation](https://sequential-parameter-optimization.github.io/spotPython/reference/) - Neural network class supporting string-based activation functions
- [Diabetes Dataset Utilities](diabetes_dataset.md) - Data loading utilities used in examples
- [Variable Types](var_type.md) - Overview of all variable types in SpotOptim
- [Save and Load](save_load.md) - Saving and loading optimization results with factor variables

