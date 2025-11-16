# SpotOptim Table Methods

This document describes the sophisticated table printing methods available in SpotOptim, inspired by spotPython's `print_res_table()` functionality.

## Overview

SpotOptim now provides three methods for displaying optimization results in formatted tables:

1. **`print_design_table()`** - Show search space configuration before optimization
2. **`print_results_table()`** - Show optimization results with optional importance scores
3. **`print_best()`** - Show best solution in a simple format

## Methods

### 1. print_design_table()

Displays the search space design **before** optimization runs. Useful for documenting and inspecting the configuration.

**Parameters:**
- `tablefmt`: Table format (`"github"`, `"grid"`, `"simple"`, etc.)
- `precision`: Decimal places for floats (default: 4)

**Example:**
```python
from spotoptim import SpotOptim
import numpy as np

opt = SpotOptim(
    fun=lambda X: np.sum(X**2, axis=1),
    bounds=[(16, 128), (1, 4), (-3, 0), (-2, 1)],
    var_type=["int", "int", "num", "num"],
    var_name=["neurons", "layers", "log10_lr", "log10_alpha"],
    max_iter=30
)

# Before running optimization
table = opt.print_design_table()
print(table)
```

**Output:**
```
| name         | type   |   lower |   upper |   default |
|--------------|--------|---------|---------|-----------|
| neurons      | int    | 16.0000 | 128.0000|   72.0000 |
| layers       | int    |  1.0000 |   4.0000|    2.0000 |
| log10_lr     | num    | -3.0000 |   0.0000|   -1.5000 |
| log10_alpha  | num    | -2.0000 |   1.0000|   -0.5000 |
```

### 2. print_results_table()

Displays comprehensive optimization results **after** optimization. Shows best values found and optionally includes importance scores.

**Parameters:**
- `tablefmt`: Table format (default: `"github"`)
- `precision`: Decimal places for floats (default: 4)
- `show_importance`: Include variable importance scores (default: `False`)
- `importance_threshold`: Minimum threshold for significance stars (default: 0.0)

**Importance Stars:**
- `***`: Importance > 95% (highly significant)
- `**`: Importance > 50% (significant)
- `*`: Importance > 1% (moderate)
- `.`: Importance > 0.1% (weak)

**Example:**
```python
result = opt.optimize()

# Without importance
table = opt.print_results_table()
print(table)

# With importance scores
table = opt.print_results_table(show_importance=True)
print(table)
```

**Output (without importance):**
```
| name         | type   |   lower |    upper |   tuned |
|--------------|--------|---------|----------|---------|
| neurons      | int    | 16.0000 | 128.0000 | 64.0000 |
| layers       | int    |  1.0000 |   4.0000 |  2.0000 |
| log10_lr     | num    | -3.0000 |   0.0000 | -1.5234 |
| log10_alpha  | num    | -2.0000 |   1.0000 | -0.3456 |
```

**Output (with importance):**
```
| name         | type   |   lower |    upper |   tuned |   importance | stars   |
|--------------|--------|---------|----------|---------|--------------|---------|
| neurons      | int    | 16.0000 | 128.0000 | 64.0000 |        42.15 | **      |
| layers       | int    |  1.0000 |   4.0000 |  2.0000 |        31.28 | *       |
| log10_lr     | num    | -3.0000 |   0.0000 | -1.5234 |        15.47 | *       |
| log10_alpha  | num    | -2.0000 |   1.0000 | -0.3456 |        11.10 | *       |
```

### 3. print_best()

Simple, human-readable display of the best solution found.

**Parameters:**
- `result`: OptimizeResult object (optional, uses stored values if None)
- `transformations`: List of functions to transform parameters (optional)
- `show_name`: Whether to show variable names (default: `True`)
- `precision`: Decimal places for floats (default: 4)

**Example:**
```python
# Simple usage
opt.print_best(result)

# With transformations (e.g., log-scale to original scale)
transformations = [
    int,              # neurons -> int
    int,              # layers -> int
    lambda x: 10**x,  # log10_lr -> lr
    lambda x: 10**x   # log10_alpha -> alpha
]
opt.print_best(result, transformations=transformations)
```

**Output:**
```
Best Solution Found:
--------------------------------------------------
  neurons: 64
  layers: 2
  log10_lr: 0.0300
  log10_alpha: 0.4518
  Objective Value: 0.0123
  Total Evaluations: 30
```

## Variable Importance

The `get_importance()` method calculates variable importance based on correlation between parameter values and objective values across all evaluations. Higher scores indicate parameters that have more influence on the objective.

**Example:**
```python
importance = opt.get_importance()
for i, (name, imp) in enumerate(zip(var_names, importance)):
    print(f"{name}: {imp:.2f}%")
```

**Output:**
```
neurons: 42.15%
layers: 31.28%
log10_lr: 15.47%
log10_alpha: 11.10%
```

## Table Formats

The `tablefmt` parameter supports various formats via the `tabulate` library:

- `"github"`: GitHub-flavored Markdown (default)
- `"grid"`: Grid format with borders
- `"simple"`: Simple format without borders
- `"plain"`: Plain text
- `"html"`: HTML table
- `"latex"`: LaTeX table
- Many more (see tabulate documentation)

## Factor Variables

The methods automatically handle factor (categorical) variables, displaying original string values:

```python
opt = SpotOptim(
    fun=lambda X: np.sum(X**2, axis=1),
    bounds=[(10, 100), ("SGD", "Adam", "RMSprop"), (0.001, 0.1)],
    var_name=["neurons", "optimizer", "lr"],
    var_type=["int", "factor", "num"],
    max_iter=30
)

result = opt.optimize()
table = opt.print_results_table()
print(table)
```

**Output:**
```
| name      | type   | lower   | upper   | tuned   |
|-----------|--------|---------|---------|---------|
| neurons   | int    | 10.0    | 100.0   | 55.0    |
| optimizer | factor | SGD     | RMSprop | Adam    |
| lr        | num    | 0.001   | 0.1     | 0.0234  |
```

## Dependencies

These methods require the `tabulate` package:
```bash
pip install tabulate
# or
uv pip install tabulate
```

## Integration with Quarto

The methods work seamlessly in Quarto documents:

````markdown
```{python}
# Show search space
design_table = optimizer.print_design_table()
print(design_table)

# Run optimization
result = optimizer.optimize()

# Show results with importance
results_table = optimizer.print_results_table(show_importance=True)
print(results_table)
```
````

## Comparison with spotPython

These methods provide similar functionality to spotPython's `print_res_table()` but are adapted to SpotOptim's architecture:

| Feature | spotPython | SpotOptim |
|---------|------------|-----------|
| Design table | `print_exp_table()` | `print_design_table()` |
| Results table | `print_res_table()` | `print_results_table()` |
| Importance scores | ✓ | ✓ |
| Factor variables | ✓ | ✓ |
| Multiple formats | ✓ | ✓ |
| Transformations | - | ✓ (in `print_best()`) |
| Dimension reduction | - | ✓ (automatic) |

## Complete Example

See `examples/table_methods_demo.py` for a complete working example demonstrating all three methods.
