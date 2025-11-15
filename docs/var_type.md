# Variable Type (var_type) Implementation

## Overview

This document describes the `var_type` implementation in SpotOptim, which allows users to specify different data types for optimization variables.

## Supported Variable Types

SpotOptim supports three main data types:

### 1. **'float'**

- **Purpose**: Continuous optimization with Python floats
- **Behavior**: No rounding applied, values remain continuous
- **Use case**: Standard continuous optimization variables
- **Example**: Temperature (23.5°C), Distance (1.234m)

### 2. **'int'**

- **Purpose**: Discrete integer optimization
- **Behavior**: Float values are automatically rounded to integers
- **Use case**: Count variables, discrete parameters
- **Example**: Number of layers (5), Population size (100)

### 3. **'factor'**

- **Purpose**: Unordered categorical data
- **Behavior**: Internally mapped to integer values (0, 1, 2, ...)
- **Use case**: Categorical choices like colors, algorithms, modes
- **Example**: Color ("red"→0, "green"→1, "blue"→2)
- **Note**: The actual string-to-int mapping is external to SpotOptim; the optimizer works with the integer representation

## Implementation Details

### Where `var_type` is Used

The `var_type` parameter is properly propagated throughout the optimization process:

1. **Initialization** (`__init__`):

   - Stored as `self.var_type`
   - Default: `["float"] * n_dim` if not specified

2. **Initial Design Generation** (`_generate_initial_design`):

   - Applies type constraints via `_repair_non_numeric()`
   - Ensures initial points respect variable types

3. **New Point Suggestion** (`_suggest_next_point`):

   - Applies type constraints to acquisition function optimization results
   - Ensures suggested points respect variable types

4. **User-Provided Initial Design** (`optimize`):

   - Applies type constraints to X0 if provided
   - Ensures consistency regardless of input source

5. **Mesh Grid Generation** (`_generate_mesh_grid`):

   - Used for plotting, respects variable types
   - Ensures visualization shows correct discrete/continuous behavior

### Core Method: `_repair_non_numeric()`

This method enforces variable type constraints:

```python
def _repair_non_numeric(self, X: np.ndarray, var_type: List[str]) -> np.ndarray:
    """Round non-continuous values to integers."""
    mask = np.isin(var_type, ["float"], invert=True)
    X[:, mask] = np.around(X[:, mask])
    return X
```

**Logic:**

- Variables with type `'float'`: No change (continuous)
- Variables with type `'int'` or `'factor'`: Rounded to integers

## Usage Examples

## 5. Example Usage

```python
import numpy as np
from spotoptim import SpotOptim

# Example 1: All float variables (default)
opt1 = SpotOptim(
    fun=lambda x: np.sum(x**2),
    lower=np.array([0, 0, 0]),
    upper=np.array([10, 10, 10])
    # var_type defaults to ["float", "float", "float"]
)
```

### Example 2: Pure Integer Optimization
```python
def discrete_func(X):
    return np.sum(np.round(X)**2, axis=1)

bounds = [(-5, 5), (-5, 5)]
var_type = ["int", "int"]

opt = SpotOptim(
    fun=discrete_func,
    bounds=bounds,
    var_type=var_type,
    max_iter=20,
    seed=42
)

result = opt.optimize()
# result.x will have integer values like [1.0, -2.0]
```

### Example 3: Categorical (Factor) Variables
```python
def categorical_func(X):
    # Assume X[:, 0] represents 3 categories: 0, 1, 2
    # Category 0 is best
    return (X[:, 0]**2) + (X[:, 1]**2)

bounds = [(0, 2), (0, 3)]  # 3 and 4 categories respectively
var_type = ["factor", "factor"]

opt = SpotOptim(
    fun=categorical_func,
    bounds=bounds,
    var_type=var_type,
    max_iter=20,
    seed=42
)

result = opt.optimize()
# result.x will be integers like [0.0, 1.0] representing categories
```

### Example 4: Mixed Variable Types
```python
def mixed_func(X):
    # X[:, 0]: continuous temperature
    # X[:, 1]: discrete number of iterations
    # X[:, 2]: categorical algorithm choice (0, 1, 2)
    return X[:, 0]**2 + X[:, 1]**2 + X[:, 2]**2

bounds = [(-5, 5), (1, 100), (0, 2)]
var_type = ["float", "int", "factor"]

opt = SpotOptim(
    fun=mixed_func,
    bounds=bounds,
    var_type=var_type,
    max_iter=20,
    seed=42
)

result = opt.optimize()
# result.x[0]: continuous float like 0.123
# result.x[1]: integer like 5.0
# result.x[2]: integer category like 0.0
```

## Key Findings

1. **Type Persistence**: Variable types are correctly maintained throughout the entire optimization process, from initial design through all iterations.

2. **Automatic Enforcement**: The `_repair_non_numeric()` method is called at all critical points, ensuring type constraints are never violated.

3. **Three Explicit Types**: Only `'float'`, `'int'`, and `'factor'` are supported. The legacy `'num'` type has been removed for clarity.

4. **User-Provided Data**: Type constraints are applied even to user-provided initial designs, ensuring consistency.

5. **Plotting Compatibility**: The plotting functionality respects variable types, ensuring correct visualization of discrete vs. continuous variables.

## Recommendations

1. **Always specify var_type explicitly** for clarity, especially in mixed-type problems
2. **Use appropriate bounds** for factor variables (e.g., `(0, n_categories-1)`)
3. **External mapping** for string categories: Maintain your own mapping dictionary outside SpotOptim (e.g., `{"red": 0, "green": 1, "blue": 2}`)
4. **Validation**: The current implementation doesn't validate var_type length matches bounds length - users should ensure this manually

## Future Enhancements (Optional)

Potential improvements that could be added:

1. **Validation**: Add validation in `__init__` to check `len(var_type) == len(bounds)`
2. **String Categories**: Add built-in support for automatic string-to-int mapping
3. **Ordered Categories**: Support ordered categorical variables (ordinal data)
4. **Type Checking**: Validate that var_type values are one of the allowed strings
5. **Bounds Checking**: Warn if factor bounds are not integer ranges
