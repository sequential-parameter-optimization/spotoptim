# SpotOptim Surrogate Models

This module provides surrogate models for use with the SpotOptim optimizer.

## Overview

SpotOptim now offers **two Kriging (Gaussian Process) implementations**:

1. **Kriging** (in `kriging.py`) - Full-featured implementation with mixed variable type support
2. **SimpleKriging** (in `simple_kriging.py`) - Lightweight implementation for quick prototyping

### Which One to Use?

- **Use `Kriging`** (recommended for most cases):
  - Mixed variable types (continuous, integer, categorical)
  - Regression, interpolation, or reinterpolation methods
  - Lambda (nugget) optimization
  - Production optimization tasks

- **Use `SimpleKriging`**:
  - Quick prototyping
  - Only continuous (float) variables
  - Simpler parameter tuning
  - Educational purposes

---

## Kriging (Full-Featured)

The `Kriging` class provides a comprehensive Gaussian Process implementation adapted from spotPython, with full support for mixed variable types.

### Features

- **Scikit-learn compatible interface**: Implements `fit()` and `predict()` methods
- **Automatic hyperparameter optimization**: Uses maximum likelihood estimation
- **Gaussian (RBF) kernel**: Exponential correlation function
- **Prediction uncertainty**: Supports `return_std=True` for standard deviations
- **Reproducible**: Supports random seed for consistent results

### Key Features

- **Multiple methods**: interpolation, regression, reinterpolation
- **Mixed variable types**: `float`/`num` (continuous), `int` (integer), `factor` (categorical)
- **Lambda optimization**: Automatic nugget effect tuning for regression methods
- **Isotropic option**: Single or multiple length scales
- **Flexible bounds**: Configurable theta and Lambda ranges

### Basic Usage

```python
import numpy as np
from spotoptim import SpotOptim
from spotoptim.surrogate import Kriging

# Define objective function
def sphere(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1)

# Create Kriging surrogate with regression method
kriging = Kriging(
    method='regression',      # 'interpolation', 'regression', or 'reinterpolation'
    min_theta=-3.0,
    max_theta=2.0,
    seed=42
)

# Use with SpotOptim
optimizer = SpotOptim(
    fun=sphere,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=20,
    n_initial=10,
    surrogate=kriging,
    seed=42
)

result = optimizer.optimize()
```

### Mixed Variable Types Example

```python
from spotoptim.surrogate import Kriging

# 3D problem: continuous, integer, categorical
kriging = Kriging(
    method='regression',
    var_type=['float', 'int', 'factor'],  # or ['num', 'int', 'factor']
    seed=42
)

X_train = np.array([
    [0.5, 2, 0],  # [continuous, integer, category_id]
    [1.2, 3, 1],
    [2.1, 1, 0],
])
y_train = np.array([1.2, 2.3, 1.8])

kriging.fit(X_train, y_train)
y_pred, std = kriging.predict(np.array([[1.0, 2, 1]]), return_std=True)
```

### Parameters

- **method** (str, default='regression'): Fitting method
  - `'interpolation'`: Pure interpolation with small nugget
  - `'regression'`: Regression with optimized Lambda
  - `'reinterpolation'`: Regression with Lambda removed in variance calculation
- **var_type** (List[str], optional): Variable types for each dimension
  - `'float'` or `'num'`: Continuous variables
  - `'int'`: Integer variables
  - `'factor'`: Categorical variables
- **noise** (float, optional): Small regularization term. If None, uses sqrt(machine epsilon).
- **isotropic** (bool, default=False): Use single theta for all dimensions
- **n_theta** (int, optional): Number of theta parameters. If None, set during fit.
- **min_theta** (float, default=-3.0): Minimum log10(theta) bound.
- **max_theta** (float, default=2.0): Maximum log10(theta) bound.
- **min_Lambda** (float, default=-9.0): Minimum log10(Lambda) bound (for regression methods).
- **max_Lambda** (float, default=0.0): Maximum log10(Lambda) bound.
- **metric_factorial** (str, default='canberra'): Distance metric for factor variables.
- **seed** (int, default=124): Random seed for reproducibility.
- **model_fun_evals** (int, default=100): Maximum function evaluations for hyperparameter optimization.

### Methods

#### fit(X, y)

Fit the Kriging model to training data.

**Parameters:**
- `X`: ndarray of shape (n_samples, n_features) - Training input data
- `y`: ndarray of shape (n_samples,) - Training target values

**Returns:**
- `self`: Fitted estimator

#### predict(X, return_std=False)

Predict using the Kriging model.

**Parameters:**
- `X`: ndarray of shape (n_samples, n_features) - Points to predict at
- `return_std`: bool, default=False - If True, return standard deviations as well

**Returns:**
- `y_pred`: ndarray of shape (n_samples,) - Predicted values
- `y_std`: ndarray of shape (n_samples,) - Standard deviations (only if return_std=True)

---

## SimpleKriging (Lightweight)

The `SimpleKriging` class provides a simplified Gaussian Process implementation for quick prototyping.

### Features

- **Lightweight**: Minimal implementation (~350 lines)
- **Gaussian kernel**: RBF correlation function only
- **Continuous variables**: Best for float-only problems
- **Scikit-learn compatible**: Standard fit/predict interface

### Basic Usage

```python
from spotoptim import SpotOptim
from spotoptim.surrogate import SimpleKriging

def sphere(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1)

# Create SimpleKriging surrogate
simple_kriging = SimpleKriging(
    noise=1e-10,
    min_theta=-3.0,
    max_theta=2.0,
    seed=42
)

# Use with SpotOptim
optimizer = SpotOptim(
    fun=sphere,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=20,
    n_initial=10,
    surrogate=simple_kriging,
    seed=42
)

result = optimizer.optimize()
```

### Parameters

- **noise** (float, optional): Regularization parameter (nugget effect). If None, uses sqrt(machine epsilon).
- **kernel** (str, default='gauss'): Kernel type. Currently only 'gauss' (Gaussian/RBF) is supported.
- **n_theta** (int, optional): Number of theta parameters. If None, uses k (number of dimensions).
- **min_theta** (float, default=-3.0): Minimum log10(theta) bound for optimization.
- **max_theta** (float, default=2.0): Maximum log10(theta) bound for optimization.
- **seed** (int, optional): Random seed for reproducibility.

---

## Comparison Table

| Feature | Kriging (Full) | SimpleKriging | GaussianProcessRegressor |
|---------|----------------|---------------|--------------------------|
| Variable Types | float, int, factor | float only | float only |
| Methods | 3 (interp/regr/reinterp) | 1 (interpolation) | 1 (regression) |
| Lambda Optimization | ✅ Yes | ❌ No | ✅ Yes (via kernel) |
| Isotropic Option | ✅ Yes | ❌ No | ✅ Yes |
| Distance Metrics | Multiple | Gaussian only | Multiple kernels |
| Code Complexity | ~650 lines | ~350 lines | Complex |
| Best For | Production, mixed vars | Prototyping | scikit-learn integration |

### When to Use Each

**Kriging (Full)**:
- Production optimization with mixed variable types
- Problems with categorical/integer variables
- Need for different fitting methods (regression vs interpolation)
- Explicit Lambda (nugget) control

**SimpleKriging**:
- Quick prototyping and testing
- Pure continuous optimization
- Educational purposes
- When simplicity is prioritized

**GaussianProcessRegressor** (scikit-learn default):
- Need for specific kernel types (Matern, RationalQuadratic)
- Advanced scikit-learn pipeline integration
- Gradient-based predictions required

## Common Methods

Both Kriging classes implement the scikit-learn interface:

### fit(X, y)

Fit the model to training data.

**Parameters:**
- `X`: ndarray of shape (n_samples, n_features) - Training input data
- `y`: ndarray of shape (n_samples,) - Training target values

**Returns:**
- `self`: Fitted estimator

### predict(X, return_std=False)

Predict using the fitted model.

**Parameters:**
- `X`: ndarray of shape (n_samples, n_features) - Points to predict at
- `return_std`: bool, default=False - If True, return standard deviations as well

**Returns:**
- `y_pred`: ndarray of shape (n_samples,) - Predicted values
- `y_std`: ndarray of shape (n_samples,) - Standard deviations (only if return_std=True)

### get_params(deep=True) / set_params(**params)

Get and set parameters (scikit-learn compatibility).

---

## Example: Comparison

```python
import numpy as np
from spotoptim import SpotOptim
from spotoptim.surrogate import Kriging, SimpleKriging

def rosenbrock(X):
    X = np.atleast_2d(X)
    x, y = X[:, 0], X[:, 1]
    return (1 - x)**2 + 100 * (y - x**2)**2

bounds = [(-2, 2), (-2, 2)]

# With Kriging (full)
optimizer_kriging = SpotOptim(
    fun=rosenbrock,
    bounds=bounds,
    surrogate=Kriging(method='regression', seed=42),
    seed=42
)
result_kriging = optimizer_kriging.optimize()

# With SimpleKriging
optimizer_simple = SpotOptim(
    fun=rosenbrock,
    bounds=bounds,
    surrogate=SimpleKriging(noise=1e-10, seed=42),
    seed=42
)
result_simple = optimizer_simple.optimize()

print(f"Kriging result: {result_kriging.fun:.6f}")
print(f"SimpleKriging result: {result_simple.fun:.6f}")
```

---

## References

- Forrester, A., Sobester, A., & Keane, A. (2008). *Engineering Design via Surrogate Modelling: A Practical Guide*. Wiley.
- Jones, D. R., Schonlau, M., & Welch, W. J. (1998). Efficient global optimization of expensive black-box functions. *Journal of Global Optimization*, 13(4), 455-492.
- Bartz-Beielstein, T. (2022). *spotPython: Sequential Parameter Optimization Toolbox in Python*. https://github.com/sequential-parameter-optimization/spotPython
