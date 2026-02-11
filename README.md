<div align="left">
  <img src="https://raw.githubusercontent.com/sequential-parameter-optimization/spotoptim/main/img/spotLogo.png" alt="spotoptim Logo" width="300">
</div>


# spotoptim

Sequential Parameter Optimization Toolbox


[![Python Version](https://img.shields.io/badge/python-3.14%2B-blue)](https://www.python.org/downloads/)
[![PyPI Version](https://img.shields.io/pypi/v/spotoptim)](https://pypi.org/project/spotoptim/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/spotoptim)](https://pypi.org/project/spotoptim/)
[![Total Downloads](https://static.pepy.tech/badge/spotoptim)](https://pepy.tech/project/spotoptim)
[![License](https://img.shields.io/github/license/sequential-parameter-optimization/spotoptim)](LICENSE)

**Testing & Quality**

[![Build Status](https://img.shields.io/github/actions/workflow/status/sequential-parameter-optimization/spotoptim/ci.yml?branch=main&label=Tests)](https://github.com/sequential-parameter-optimization/spotoptim/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/sequential-parameter-optimization/spotoptim/branch/main/graph/badge.svg)](https://codecov.io/gh/sequential-parameter-optimization/spotoptim)
[![REUSE Compliance](https://api.reuse.software/badge/github.com/sequential-parameter-optimization/spotoptim)](https://api.reuse.software/project/github.com/sequential-parameter-optimization/spotoptim)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/sequential-parameter-optimization/spotoptim/badge)](https://scorecard.dev/viewer/?uri=github.com/sequential-parameter-optimization/spotoptim)
[![Documentation](https://img.shields.io/badge/docs-passing-brightgreen)](https://sequential-parameter-optimization.github.io/spotoptim/)
[![GitHub Release](https://img.shields.io/github/v/release/sequential-parameter-optimization/spotoptim)](https://github.com/sequential-parameter-optimization/spotoptim/releases)

**Status**

[![Maintenance](https://img.shields.io/badge/maintenance-active-green)](https://github.com/sequential-parameter-optimization/spotoptim)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## About spotoptim

`spotoptim` is a Python toolbox for Sequential Parameter Optimization (SPO), designed for robust and efficient optimization of expensive-to-evaluate functions.


## Documentation

 Documentation (API) is available at: [https://sequential-parameter-optimization.github.io/spotoptim/](https://sequential-parameter-optimization.github.io/spotoptim/)

## License

`spotoptim` software: [AGPL-3.0-or-later License](LICENSE)




## Features

- **Surrogate Model Based Optimization**: Uses surrogate models to efficiently optimize expensive black-box functions
- **Multiple Acquisition Functions**: Expected Improvement (EI), Predicted Mean (y), Probability of Improvement (PI)
- **Flexible Surrogates**: Default Gaussian Process or custom Kriging surrogate
- **Variable Types**: Support for continuous, integer, and mixed variable types
- **scipy-compatible**: Returns OptimizeResult objects compatible with scipy.optimize

## Installation

```bash
pip install spotoptim
```

## Quick Start

```python
import numpy as np
from spotoptim import SpotOptim

# Define objective function
def rosenbrock(X):
    X = np.atleast_2d(X)
    x, y = X[:, 0], X[:, 1]
    return (1 - x)**2 + 100 * (y - x**2)**2

# Set up optimization
bounds = [(-2, 2), (-2, 2)]

optimizer = SpotOptim(
    fun=rosenbrock,
    bounds=bounds,
    max_iter=50,
    n_initial=10,
    seed=42
)

# Run optimization
result = optimizer.optimize()

print(f"Best point: {result.x}")
print(f"Best value: {result.fun}")
```

## Using Kriging Surrogate

SpotOptim includes a simplified Kriging (Gaussian Process) surrogate as an alternative to scikit-learn's GaussianProcessRegressor:

```python
from spotoptim import SpotOptim, Kriging

# Create Kriging surrogate
kriging = Kriging(
    noise=1e-6,
    min_theta=-3.0,
    max_theta=2.0,
    seed=42
)

# Use with SpotOptim
optimizer = SpotOptim(
    fun=rosenbrock,
    bounds=bounds,
    surrogate=kriging,  # Use Kriging instead of default GP
    seed=42
)

result = optimizer.optimize()
```

## API Reference

### SpotOptim

**Parameters:**

- `fun` (callable): Objective function to minimize
- `bounds` (list of tuples): Bounds for each dimension as [(low, high), ...]
- `max_iter` (int, default=20): Maximum number of optimization iterations
- `n_initial` (int, default=10): Number of initial design points
- `surrogate` (object, optional): Surrogate model (default: GaussianProcessRegressor)
- `acquisition` (str, default='ei'): Acquisition function ('ei', 'y', 'pi')
- `var_type` (list of str, optional): Variable types for each dimension
- `tolerance_x` (float, optional): Minimum distance between points
- `seed` (int, optional): Random seed for reproducibility
- `verbose` (bool, default=False): Print progress information
- `max_surrogate_points` (int, optional): Maximum number of points for surrogate fitting (default: None, use all points)
- `selection_method` (str, default='distant'): Point selection method ('distant' or 'best')

**Methods:**

- `optimize(X0=None)`: Run optimization, optionally with initial design points
- `plot_surrogate(i=0, j=1, show=True, **kwargs)`: Visualize the fitted surrogate model

## Point Selection for Surrogate Training

When optimizing expensive functions with many iterations, the number of evaluated points can become large, making surrogate model training computationally expensive. SpotOptim implements an automatic point selection mechanism to address this:

### Usage

```python
optimizer = SpotOptim(
    fun=expensive_function,
    bounds=bounds,
    max_iter=100,
    n_initial=20,
    max_surrogate_points=50,  # Use only 50 points for surrogate training
    selection_method='distant',  # or 'best'
    verbose=True
)
```

### Selection Methods

1. **'distant' (default)**: Uses K-means clustering to select points that are maximally distant from each other, ensuring good space-filling properties.

2. **'best'**: Clusters points and selects all points from the cluster with the best (lowest) mean objective function value, focusing on promising regions.

### Benefits

- **Reduced computational cost**: Surrogate training scales with the number of points
- **Maintained accuracy**: Carefully selected points preserve model quality
- **Scalability**: Enables optimization with hundreds or thousands of function evaluations

See `examples/point_selection_example.py` for a complete demonstration.

### Kriging

**Parameters:**

- `noise` (float, optional): Regularization parameter
- `kernel` (str, default='gauss'): Kernel type
- `n_theta` (int, optional): Number of theta parameters
- `min_theta` (float, default=-3.0): Minimum log10(theta) bound
- `max_theta` (float, default=2.0): Maximum log10(theta) bound
- `seed` (int, optional): Random seed

**Methods:**

- `fit(X, y)`: Fit the model to training data
- `predict(X, return_std=False)`: Predict at new points

## Visualizing Results

SpotOptim includes a `plot_surrogate()` method to visualize the fitted surrogate model:

```python
# After running optimization
optimizer.plot_surrogate(
    i=0, j=1,                    # Dimensions to plot
    var_name=['x1', 'x2'],       # Variable names
    add_points=True,             # Show evaluated points
    cmap='viridis',              # Colormap
    show=True
)
```

The plot shows:

- **Top left**: 3D surface of predictions
- **Top right**: 3D surface of prediction uncertainty
- **Bottom left**: Contour plot of predictions with evaluated points
- **Bottom right**: Contour plot of prediction uncertainty

For higher-dimensional problems, the method visualizes a 2D slice by fixing other dimensions at their mean values.

## Examples

### Notebooks

See `notebooks/demos.ipynb` for interactive examples:

1. 2D Rosenbrock function optimization
2. 6D Rosenbrock with budget constraints
3. Using Kriging surrogate vs default GP
4. Visualizing surrogate models with `plot_surrogate()`

### Real-World Applications

The `examples/` directory contains detailed tutorials:

**Aircraft Wing Weight Optimization (AWWE)**

- `awwe.qmd` - Comprehensive Quarto tutorial teaching surrogate-based optimization
- `awwe_optimization.py` - Standalone Python script demonstrating complete workflow
- 9-dimensional optimization problem from engineering design
- Includes homework exercise for 10-dimensional extension

Run the example:
```bash
cd examples
python awwe_optimization.py
```

See `examples/README.md` for more details and additional examples.

## Development

```bash
# Clone repository
git clone https://github.com/sequential-parameter-optimization/spotoptim.git
cd spotoptim

# Install with uv
uv pip install -e .

# Run tests
uv run pytest tests/

# Build package
uv build
```

## License

See LICENSE file.

## References

Based on the SPOT (Sequential Parameter Optimization Toolbox) methodology.
