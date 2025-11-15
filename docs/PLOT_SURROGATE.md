# Surrogate Model Visualization

This document describes the `plot_surrogate()` method added to the `SpotOptim` class, which provides visualization capabilities similar to the `plotkd()` function in the spotpython package.

## Overview

The `plot_surrogate()` method creates a comprehensive 4-panel visualization of the fitted surrogate model, showing both predictions and uncertainty estimates across two selected dimensions.

## Features

- **3D Surface Plots**: Visualize the surrogate's predictions and uncertainty as 3D surfaces
- **Contour Plots**: View 2D contours with overlaid evaluation points
- **Multi-dimensional Support**: Visualize any two dimensions of higher-dimensional problems
- **Customizable Appearance**: Control colors, resolution, transparency, and more

## Usage

### Basic Usage

```python
import numpy as np
from spotoptim import SpotOptim

# Define objective function
def sphere(X):
    return np.sum(X**2, axis=1)

# Run optimization
optimizer = SpotOptim(fun=sphere, bounds=[(-5, 5), (-5, 5)], max_iter=20)
result = optimizer.optimize()

# Visualize the surrogate model
optimizer.plot_surrogate(i=0, j=1, show=True)
```

### With Custom Parameters

```python
optimizer.plot_surrogate(
    i=0,                          # First dimension to plot
    j=1,                          # Second dimension to plot
    var_name=['x1', 'x2'],        # Variable names for axes
    add_points=True,              # Show evaluated points
    cmap='viridis',               # Colormap
    alpha=0.7,                    # Surface transparency
    num=100,                      # Grid resolution
    contour_levels=25,            # Number of contour levels
    grid_visible=True,            # Show grid on contours
    figsize=(12, 10),             # Figure size
    show=True                     # Display immediately
)
```

### Higher-Dimensional Problems

For problems with more than 2 dimensions, `plot_surrogate()` creates a 2D slice by fixing all other dimensions at their mean values:

```python
# 4D optimization problem
def sphere_4d(X):
    return np.sum(X**2, axis=1)

bounds = [(-3, 3)] * 4
optimizer = SpotOptim(fun=sphere_4d, bounds=bounds, max_iter=20)
result = optimizer.optimize()

# Visualize dimensions 0 and 2 (dimensions 1 and 3 fixed at mean)
optimizer.plot_surrogate(
    i=0, j=2,
    var_name=['x0', 'x1', 'x2', 'x3']
)

# Visualize different dimension pair
optimizer.plot_surrogate(i=1, j=3, var_name=['x0', 'x1', 'x2', 'x3'])
```

## Plot Interpretation

The visualization consists of 4 panels:

### Top Left: Prediction Surface

- Shows the surrogate model's predicted function values as a 3D surface
- Helps understand the model's belief about the objective function landscape
- Lower values (blue in default colormap) indicate predicted minima

### Top Right: Prediction Uncertainty Surface

- Shows the standard deviation of predictions as a 3D surface
- Indicates where the model is uncertain and might benefit from more samples
- Lower values (blue) indicate high confidence, higher values (red) indicate uncertainty

### Bottom Left: Prediction Contour with Points

- 2D contour plot of predictions
- Red dots show the actual points evaluated during optimization
- Useful for understanding the exploration-exploitation trade-off

### Bottom Right: Uncertainty Contour with Points

- 2D contour plot of prediction uncertainty
- Shows how uncertainty decreases around evaluated points
- Helps identify unexplored regions

## Parameters

### Dimension Selection

- `i` (int, default=0): Index of first dimension to plot
- `j` (int, default=1): Index of second dimension to plot

### Appearance

- `var_name` (list of str, optional): Names for each dimension
- `cmap` (str, default='jet'): Matplotlib colormap name
- `alpha` (float, default=0.8): Surface transparency (0=transparent, 1=opaque)
- `figsize` (tuple, default=(12, 10)): Figure size in inches (width, height)

### Grid and Resolution

- `num` (int, default=100): Number of grid points per dimension
- `contour_levels` (int, default=30): Number of contour levels
- `grid_visible` (bool, default=True): Show grid lines on contour plots

### Color Scaling

- `vmin` (float, optional): Minimum value for color scale
- `vmax` (float, optional): Maximum value for color scale

### Display

- `show` (bool, default=True): Display plot immediately
- `add_points` (bool, default=True): Overlay evaluated points on contours

## Examples

### Example 1: 2D Rosenbrock Function

```python
import numpy as np
from spotoptim import SpotOptim

def rosenbrock(X):
    X = np.atleast_2d(X)
    x, y = X[:, 0], X[:, 1]
    return (1 - x)**2 + 100 * (y - x**2)**2

optimizer = SpotOptim(
    fun=rosenbrock,
    bounds=[(-2, 2), (-2, 2)],
    max_iter=30,
    seed=42
)
result = optimizer.optimize()

# Visualize with custom colormap
optimizer.plot_surrogate(
    var_name=['x', 'y'],
    cmap='coolwarm',
    add_points=True
)
```

### Example 2: Using Kriging Surrogate

```python
from spotoptim import SpotOptim, Kriging

def sphere(X):
    return np.sum(X**2, axis=1)

optimizer = SpotOptim(
    fun=sphere,
    bounds=[(-5, 5), (-5, 5)],
    surrogate=Kriging(seed=42),  # Use Kriging instead of GP
    max_iter=20
)
result = optimizer.optimize()

# The plotting works the same with any surrogate
optimizer.plot_surrogate(var_name=['x1', 'x2'])
```

### Example 3: Comparing Different Dimension Pairs

```python
# 3D problem - visualize all dimension pairs
def sphere_3d(X):
    return np.sum(X**2, axis=1)

optimizer = SpotOptim(
    fun=sphere_3d,
    bounds=[(-5, 5)] * 3,
    max_iter=25
)
result = optimizer.optimize()

# Dimensions 0 vs 1
optimizer.plot_surrogate(i=0, j=1, var_name=['x0', 'x1', 'x2'])

# Dimensions 0 vs 2
optimizer.plot_surrogate(i=0, j=2, var_name=['x0', 'x1', 'x2'])

# Dimensions 1 vs 2
optimizer.plot_surrogate(i=1, j=2, var_name=['x0', 'x1', 'x2'])
```

## Tips and Best Practices

1. **Run Optimization First**: Always call `optimize()` before `plot_surrogate()`

2. **Choose Dimensions Wisely**: For high-dimensional problems, plot dimensions that you suspect are most important or interactive

3. **Adjust Resolution**: Use lower `num` values (e.g., 50) for faster plotting, higher values (e.g., 200) for smoother surfaces

4. **Color Scales**: Set `vmin` and `vmax` explicitly when comparing multiple plots to ensure consistent color scales

5. **Uncertainty Analysis**: High uncertainty areas (bright colors in uncertainty plots) are good candidates for additional sampling

6. **Exploration vs Exploitation**: Red dots clustered in low-prediction areas show exploitation; spread-out dots show exploration

## Comparison with spotpython's plotkd()

The `plot_surrogate()` method is inspired by spotpython's `plotkd()` function but adapted for SpotOptim's simplified interface:

### Similarities

- Same 4-panel layout (2 surfaces + 2 contours)
- Visualizes predictions and uncertainty
- Supports dimension selection and customization

### Differences

- **Integration**: Method of SpotOptim class (no separate function needed)
- **Simpler**: Fewer parameters, more sensible defaults
- **Automatic**: Uses optimizer's bounds and data automatically
- **Type Handling**: Automatically applies variable type constraints (int/float/factor)

## Error Handling

The method validates inputs and provides clear error messages:

```python
# Before optimization runs
optimizer.plot_surrogate()  # ValueError: No optimization data available

# Invalid dimension indices
optimizer.plot_surrogate(i=5, j=1)  # ValueError: i must be less than n_dim

# Same dimension twice
optimizer.plot_surrogate(i=0, j=0)  # ValueError: i and j must be different
```

## See Also

- `notebooks/demos.ipynb`: Example 4 demonstrates `plot_surrogate()`
- `examples/plot_surrogate_demo.py`: Standalone example script
- `tests/test_plot_surrogate.py`: Comprehensive test suite
