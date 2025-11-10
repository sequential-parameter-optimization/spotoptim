# SpotOptim Surrogate Models

This module provides surrogate models for use with the SpotOptim optimizer.

## Kriging Surrogate

The `Kriging` class provides a simplified Gaussian Process (Kriging) surrogate model that can be used as an alternative to scikit-learn's `GaussianProcessRegressor`.

### Features

- **Scikit-learn compatible interface**: Implements `fit()` and `predict()` methods
- **Automatic hyperparameter optimization**: Uses maximum likelihood estimation
- **Gaussian (RBF) kernel**: Exponential correlation function
- **Prediction uncertainty**: Supports `return_std=True` for standard deviations
- **Reproducible**: Supports random seed for consistent results

### Basic Usage

```python
import numpy as np
from spotoptim import SpotOptim, Kriging

# Define objective function
def sphere(X):
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1)

# Create Kriging surrogate
kriging = Kriging(
    noise=1e-6,
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
    surrogate=kriging,  # Use Kriging surrogate
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

### Comparison with scikit-learn's GaussianProcessRegressor

| Feature | Kriging | GaussianProcessRegressor |
|---------|---------|--------------------------|
| Interface | scikit-learn compatible | Native scikit-learn |
| Kernel | Gaussian (RBF) | Multiple kernel types |
| Hyperparameters | Theta (length scales) | Flexible kernel parameters |
| Dependencies | NumPy, SciPy | NumPy, SciPy, scikit-learn |
| Optimization | Differential evolution | L-BFGS-B with restarts |
| Code complexity | Simplified, ~350 lines | Full-featured, complex |

### When to Use Kriging

- You want a **self-contained** surrogate without heavy scikit-learn dependency
- You need a **simple, interpretable** Gaussian kernel
- You want **explicit control** over hyperparameter bounds
- You're working with **moderate-dimensional** problems (< 20 dimensions)

### When to Use Default GP

- You need **multiple kernel types** (Matern, RationalQuadratic, etc.)
- You want **advanced features** like gradient-based predictions
- You're working with **very high-dimensional** problems
- You need **production-tested** robustness

## Example: Comparison

```python
from spotoptim import SpotOptim, Kriging

def rosenbrock(X):
    X = np.atleast_2d(X)
    x, y = X[:, 0], X[:, 1]
    return (1 - x)**2 + 100 * (y - x**2)**2

bounds = [(-2, 2), (-2, 2)]

# With Kriging
optimizer_kriging = SpotOptim(
    fun=rosenbrock,
    bounds=bounds,
    surrogate=Kriging(seed=42),
    seed=42
)
result_kriging = optimizer_kriging.optimize()

# With default GP (no surrogate argument)
optimizer_gp = SpotOptim(
    fun=rosenbrock,
    bounds=bounds,
    seed=42
)
result_gp = optimizer_gp.optimize()

print(f"Kriging result: {result_kriging.fun:.6f}")
print(f"GP result: {result_gp.fun:.6f}")
```

## Future Extensions

Planned features for future releases:

- Additional kernel types (Matern, Exponential, etc.)
- Anisotropic hyperparameters
- Gradient information
- Batch predictions
- Parallel hyperparameter optimization
