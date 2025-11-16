---
title: Multi-Objective Optimization Support in SpotOptim
sidebar_position: 5
eval: true
---


## Overview

SpotOptim supports multi-objective optimization functions with automatic detection and flexible scalarization strategies. This implementation follows the same approach as the Spot class from spotPython.

## What Was Implemented

### 1. Core Functionality

**Parameter:**

- `fun_mo2so` (callable, optional): Function to convert multi-objective values to single-objective
  - Takes array of shape `(n_samples, n_objectives)`
  - Returns array of shape `(n_samples,)`
  - If `None`, uses first objective (default behavior)

**Attribute:**

- `y_mo` (ndarray or None): Stores all multi-objective function values
  - Shape: `(n_samples, n_objectives)` for multi-objective problems
  - `None` for single-objective problems

**Methods:**

- `_get_shape(y)`: Get shape of objective function output
- `_store_mo(y_mo)`: Store multi-objective values with automatic appending
- `_mo2so(y_mo)`: Convert multi-objective to single-objective values

The method `_evaluate_function(X)` automatically detects multi-objective functions. It 
calls `_mo2so()` to convert multi-objective to single-objective. It also stores the original multi-objective values in `y_mo`. And it returns single-objective values for optimization.

## Usage Examples

### Example 1: Default Behavior (Use First Objective)

```{python}
import numpy as np
from spotoptim import SpotOptim

def bi_objective(X):
    """Two conflicting objectives."""
    obj1 = np.sum(X**2, axis=1)          # Minimize at origin
    obj2 = np.sum((X - 2)**2, axis=1)    # Minimize at (2, 2)
    return np.column_stack([obj1, obj2])

optimizer = SpotOptim(
    fun=bi_objective,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=30,
    n_initial=15,
    seed=42
)

result = optimizer.optimize()

print(f"Best x: {result.x}")                    # Near [0, 0]
print(f"Best f(x): {result.fun}")               # Minimizes obj1
print(f"MO values stored: {optimizer.y_mo.shape}")  # (30, 2)
```

### Example 2: Weighted Sum Scalarization

```{python}
def weighted_sum(y_mo):
    """Equal weighting of objectives."""
    return 0.5 * y_mo[:, 0] + 0.5 * y_mo[:, 1]

optimizer = SpotOptim(
    fun=bi_objective,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=30,
    n_initial=15,
    fun_mo2so=weighted_sum,  # Custom conversion
    seed=42
)

result = optimizer.optimize()
print(f"Compromise solution: {result.x}")  # Near [1, 1]
```

### Example 3: Min-Max Scalarization

```{python}
def min_max(y_mo):
    """Minimize the maximum objective."""
    return np.max(y_mo, axis=1)

optimizer = SpotOptim(
    fun=bi_objective,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=30,
    n_initial=15,
    fun_mo2so=min_max,
    seed=42
)

result = optimizer.optimize()
# Finds solution with balanced objective values
```

### Example 4: Three or More Objectives

```{python}
def three_objectives(X):
    """Three different norms."""
    obj1 = np.sum(X**2, axis=1)           # L2 norm
    obj2 = np.sum(np.abs(X), axis=1)      # L1 norm
    obj3 = np.max(np.abs(X), axis=1)      # L-infinity norm
    return np.column_stack([obj1, obj2, obj3])

def custom_scalarization(y_mo):
    """Weighted combination."""
    return 0.4 * y_mo[:, 0] + 0.3 * y_mo[:, 1] + 0.3 * y_mo[:, 2]

optimizer = SpotOptim(
    fun=three_objectives,
    bounds=[(-5, 5), (-5, 5), (-5, 5)],
    max_iter=35,
    n_initial=20,
    fun_mo2so=custom_scalarization,
    seed=42
)

result = optimizer.optimize()
```

### Example 5: With Noise Handling

```{python}
def noisy_bi_objective(X):
    """Noisy multi-objective function."""
    noise1 = np.random.normal(0, 0.05, X.shape[0])
    noise2 = np.random.normal(0, 0.05, X.shape[0])
    
    obj1 = np.sum(X**2, axis=1) + noise1
    obj2 = np.sum((X - 1)**2, axis=1) + noise2
    return np.column_stack([obj1, obj2])

optimizer = SpotOptim(
    fun=noisy_bi_objective,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=40,
    n_initial=20,
    repeats_initial=3,      # Handle noise
    repeats_surrogate=2,
    seed=42
)

result = optimizer.optimize()
# Works seamlessly with noise handling
```

## Common Scalarization Strategies

### 1. Weighted Sum
```{python}
def weighted_sum(y_mo, weights=[0.5, 0.5]):
    return sum(w * y_mo[:, i] for i, w in enumerate(weights))
```
**Use when:** Objectives have similar scales and you want linear trade-offs

### 2. Weighted Sum with Normalization
```{python}
def normalized_weighted_sum(y_mo, weights=[0.5, 0.5]):
    # Normalize each objective to [0, 1]
    y_norm = (y_mo - y_mo.min(axis=0)) / (y_mo.max(axis=0) - y_mo.min(axis=0) + 1e-10)
    return sum(w * y_norm[:, i] for i, w in enumerate(weights))
```
**Use when:** Objectives have very different scales

### 3. Min-Max (Chebyshev)
```{python}
def min_max(y_mo):
    return np.max(y_mo, axis=1)
```
**Use when:** You want balanced performance across all objectives

### 4. Target Achievement
```{python}
def target_achievement(y_mo, targets=[0.0, 0.0]):
    # Minimize deviation from targets
    return np.sum((y_mo - targets)**2, axis=1)
```
**Use when:** You have specific target values for each objective

### 5. Product
```{python}
def product(y_mo):
    return np.prod(y_mo + 1e-10, axis=1)  # Add small value to avoid zero
```
**Use when:** All objectives should be minimized together

## Integration with Other Features

Multi-objective support works seamlessly with:

✅ **Noise Handling** - Use `repeats_initial` and `repeats_surrogate`  
✅ **OCBA** - Use `ocba_delta` for intelligent re-evaluation  
✅ **TensorBoard Logging** - Logs converted single-objective values  
✅ **Dimension Reduction** - Fixed dimensions work normally  
✅ **Custom Variable Names** - `var_name` parameter supported  

## Implementation Details

### Automatic Detection

SpotOptim automatically detects multi-objective functions:

- If function returns 2D array (n_samples, n_objectives), it's multi-objective
- If function returns 1D array (n_samples,), it's single-objective

### Data Flow

```
User Function → y_mo (raw) → _mo2so() → y_ (single-objective)
                    ↓
               y_mo (stored)
```

1. Function returns multi-objective values
2. `_store_mo()` saves them in `y_mo` attribute
3. `_mo2so()` converts to single-objective using `fun_mo2so` or default
4. Surrogate model optimizes the single-objective values
5. All original multi-objective values remain accessible in `y_mo`

### Backward Compatibility

✅ Fully backward compatible:

- Single-objective functions work unchanged
- `fun_mo2so` defaults to `None`
- `y_mo` is `None` for single-objective problems
- No breaking changes to existing code


## Limitations and Notes

### What This Is

- ✅ Scalarization approach to multi-objective optimization
- ✅ Single solution found per optimization run
- ✅ Different scalarizations → different Pareto solutions
- ✅ Suitable for preference-based multi-objective optimization

### What This Is Not

- ❌ Not a true multi-objective optimizer (doesn't find Pareto front)
- ❌ Doesn't generate multiple solutions in one run
- ❌ Not suitable for discovering entire Pareto front

### For True Multi-Objective Optimization

For finding the complete Pareto front, consider specialized tools:

- **pymoo**: Comprehensive multi-objective optimization framework
- **platypus**: Multi-objective optimization library
- **NSGA-II, MOEA/D**: Dedicated multi-objective algorithms



## Demo Script

Run the comprehensive demo (the demos files are located in the `examples` folder):
```bash
python demo_multiobjective.py
```

This demonstrates:

- Default behavior (first objective)
- Weighted sum scalarization
- Min-max scalarization
- Noisy multi-objective optimization
- Three-objective optimization

## Summary

SpotOptim provides flexible multi-objective optimization support through:

- Automatic detection of multi-objective functions
- Customizable scalarization strategies via `fun_mo2so`
- Complete storage of multi-objective values in `y_mo`
- Full integration with existing features (noise, OCBA, TensorBoard, etc.)
- 100% backward compatible with existing code

This implementation mirrors the approach used in spotPython's Spot class, providing consistency across the ecosystem.
