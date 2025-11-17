---
title: Adaptive Penalty Handling for Invalid Function Values
sidebar_position: 16
eval: true
---

SpotOptim provides intelligent penalty handling for objective functions that occasionally return NaN or infinite values. The `_apply_penalty_NA()` method automatically detects and replaces invalid values with adaptive penalties, allowing optimization to continue smoothly.

## Overview

**What is Penalty Handling?**

During optimization, the objective function may sometimes fail or return invalid values (NaN, inf, -inf) due to:

- Numerical instability (division by zero, logarithm of negative numbers)
- Out-of-domain inputs (e.g., sqrt of negative numbers)
- Simulation failures or timeouts
- Model training errors (divergent gradients, NaN losses)

SpotOptim's penalty handling automatically replaces these invalid values with high penalty values, enabling optimization to continue and explore more promising regions.

## Key Features

- **Adaptive Penalty Calculation**: Automatically computes penalty as `max(y) + 3 × std(y)` based on valid evaluations
- **Intelligent Fallback**: Uses `self.penalty` when insufficient valid data is available
- **Random Noise Addition**: Adds small random noise to prevent identical penalty values
- **Preserves Valid Values**: Only replaces NaN/inf values; valid evaluations remain unchanged
- **Verbose Reporting**: Optional detailed logging of penalty replacements

## Quick Start

### Basic Usage with Unstable Function

```{python}
from spotoptim import SpotOptim
import numpy as np

def unstable_function(X):
    """Function that occasionally produces NaN values."""
    results = []
    for params in X:
        x = params[0]
        y = params[1]
        
        # This may produce NaN for some inputs
        value = np.log(x + y) + np.sqrt(x * y)
        
        # Intentionally create some failures
        if x < -2 and y < -2:
            value = np.nan
        
        results.append(value)
    
    return np.array(results)

# Create optimizer - penalty handling is automatic
optimizer = SpotOptim(
    fun=unstable_function,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=50,
    n_initial=10,
    verbose=True,  # See penalty replacements in output
    seed=42
)

result = optimizer.optimize()

print(f"Best x: {result.x}")
print(f"Best f(x): {result.fun:.6f}")
print(f"Total evaluations: {result.nfev}")
```

## How Adaptive Penalty Works

### Penalty Calculation Strategy

When `penalty_value=None` (default), SpotOptim uses an **adaptive penalty** based on the current optimization data:

```
penalty = max(valid_y_values) + 3 × std(valid_y_values)
```

**Why this formula?**

1. **Worst than worst**: The penalty is guaranteed to be worse than any valid evaluation
2. **Scaled to problem**: Automatically adapts to the objective function's scale
3. **Statistical safety**: 3 standard deviations provides robust separation
4. **Encourages exploration**: Forces optimizer away from problematic regions

**Fallback behavior**: If fewer than 2 valid values exist, falls back to `self.penalty` (default: `np.inf`)

### Random Noise Addition

To prevent identical penalty values (which can confuse surrogate models), small random noise is added:

```
final_penalty = penalty + N(0, sd)
```

where `sd` (default: 0.1) controls noise magnitude.

## Detailed Examples

### Example 1: Adaptive Penalty in Action

```{python}
import numpy as np
from spotoptim import SpotOptim

def tricky_function(X):
    """Function with numerical instabilities."""
    results = []
    for params in X:
        x = params[0]
        
        # Undefined for x <= 0
        if x <= 0:
            value = np.nan
        else:
            value = x * np.log(x) - x**2
        
        results.append(value)
    
    return np.array(results)

# Test with verbose output to see adaptive penalties
optimizer = SpotOptim(
    fun=tricky_function,
    bounds=[(-2, 5)],  # Includes invalid region x <= 0
    max_iter=30,
    n_initial=10,
    verbose=True,
    seed=42
)

result = optimizer.optimize()

print("\nOptimization completed successfully!")
print(f"Best x: {result.x[0]:.6f}")
print(f"Best f(x): {result.fun:.6f}")

# The optimizer should find the optimum in the valid region (x > 0)
assert result.x[0] > 0, "Solution should be in valid region"
```

### Example 2: Custom Penalty Value

You can override the adaptive penalty with a fixed value:

```{python}
import numpy as np
from spotoptim import SpotOptim

def function_with_failures(X):
    """Function that sometimes fails."""
    results = []
    for params in X:
        x, y = params[0], params[1]
        
        # Simulate random failures
        if np.random.rand() < 0.1:  # 10% failure rate
            value = np.inf
        else:
            value = (x - 1)**2 + (y + 2)**2
    
    return np.array(results)

# Create optimizer with custom penalty
optimizer = SpotOptim(
    fun=function_with_failures,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=40,
    n_initial=15,
    penalty=1000.0,  # Custom fixed penalty
    verbose=True,
    seed=42
)

result = optimizer.optimize()

print(f"Optimum found at: ({result.x[0]:.4f}, {result.x[1]:.4f})")
print(f"Objective value: {result.fun:.6f}")
```

### Example 3: Handling Division by Zero

```{python}
import numpy as np
from spotoptim import SpotOptim

def reciprocal_function(X):
    """Function with potential division by zero."""
    results = []
    for params in X:
        x = params[0]
        
        # Division by zero when x = 0
        with np.errstate(divide='ignore', invalid='ignore'):
            value = 1.0 / x + x**2
        
        results.append(value)
    
    return np.array(results)

optimizer = SpotOptim(
    fun=reciprocal_function,
    bounds=[(-5, 5)],
    max_iter=30,
    n_initial=10,
    verbose=True,
    seed=42
)

result = optimizer.optimize()

print(f"Best x: {result.x[0]:.6f}")
print(f"Best f(x): {result.fun:.6f}")
print(f"x should be non-zero: {abs(result.x[0]) > 0.01}")
```

### Example 4: Machine Learning with Training Failures

```{python}
import numpy as np
from spotoptim import SpotOptim
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score

# Generate dataset
X_data, y_data = make_regression(n_samples=200, n_features=10, noise=10, random_state=42)

def train_mlp(X):
    """Train MLP with hyperparameters that may cause failures."""
    results = []
    
    for params in X:
        hidden_size = int(params[0])
        learning_rate = params[1]
        alpha = params[2]
        
        try:
            # Some configurations may fail to converge
            model = MLPRegressor(
                hidden_layer_sizes=(hidden_size,),
                learning_rate_init=learning_rate,
                alpha=alpha,
                max_iter=100,
                random_state=42
            )
            
            # Cross-validation score (negative MSE)
            scores = cross_val_score(model, X_data, y_data, cv=3, 
                                    scoring='neg_mean_squared_error')
            mse = -np.mean(scores)
            
            # Return NaN if convergence failed or MSE is unreasonable
            if np.isnan(mse) or mse > 1e6:
                results.append(np.nan)
            else:
                results.append(mse)
                
        except Exception as e:
            # Catch any training failures
            results.append(np.nan)
    
    return np.array(results)

# Optimize with adaptive penalty handling
optimizer = SpotOptim(
    fun=train_mlp,
    bounds=[
        (10, 100),      # hidden_size
        (0.001, 0.1),   # learning_rate
        (0.0001, 0.1)   # alpha (L2 regularization)
    ],
    var_name=["hidden_size", "learning_rate", "alpha"],
    var_type=["int", "float", "float"],
    var_trans=[None, "log10", "log10"],
    max_iter=30,
    n_initial=10,
    verbose=True,
    seed=42
)

result = optimizer.optimize()

print("\nBest Hyperparameters:")
print(f"  Hidden size: {int(result.x[0])}")
print(f"  Learning rate: {result.x[1]:.6f}")
print(f"  Alpha: {result.x[2]:.6f}")
print(f"  Best MSE: {result.fun:.2f}")
```

## Advanced Configuration

### Controlling Noise Level

The `sd` parameter controls the standard deviation of random noise added to penalties:

```{python}
import numpy as np
from spotoptim import SpotOptim

def noisy_objective(X):
    return np.array([np.nan if x[0] < 0 else x[0]**2 for x in X])

# Larger noise for more variation in penalty values
optimizer = SpotOptim(
    fun=noisy_objective,
    bounds=[(-5, 5)],
    max_iter=20,
    verbose=True
)

# The sd parameter is used internally in _apply_penalty_NA
# Default: sd=0.1
```

### Understanding Verbose Output

With `verbose=True`, you'll see different messages depending on the penalty strategy:

```python
# Adaptive penalty (sufficient valid values):
# "Warning: Found 2 NaN/inf value(s), replacing with adaptive penalty (max + 3*std = 15.2341)"

# Fallback penalty (insufficient valid values):
# "Warning: Found 3 NaN/inf value(s), insufficient finite values for adaptive penalty. 
#  Using self.penalty = inf"

# Custom penalty:
# "Warning: Found 1 NaN/inf value(s), replacing with 1000.0 + noise"
```

## Best Practices

### 1. Use Adaptive Penalty for Most Cases

Let SpotOptim automatically compute appropriate penalties:

```python
# Good: Adaptive penalty
optimizer = SpotOptim(fun=func, bounds=bounds)
```

### 2. Set Custom Penalty for Known Scale

If you know your objective function's scale, set an appropriate penalty:

```python
# Good: Custom penalty for known scale
optimizer = SpotOptim(
    fun=func,
    bounds=bounds,
    penalty=1000.0  # Much worse than expected worst value
)
```

### 3. Enable Verbose for Debugging

When developing, monitor penalty behavior:

```python
# Good: Enable verbose during development
optimizer = SpotOptim(
    fun=func,
    bounds=bounds,
    verbose=True
)
```

### 4. Handle Root Causes

Penalty handling is a safety net, but fixing root causes is better:

```python
# Better: Handle issues in objective function
def robust_objective(X):
    results = []
    for params in X:
        x = params[0]
        
        # Clip to valid range instead of returning NaN
        x_safe = np.clip(x, 0.001, None)
        value = np.log(x_safe)
        
        results.append(value)
    return np.array(results)
```

## Technical Details

### Method Signature

```python
def _apply_penalty_NA(
    self, 
    y: np.ndarray, 
    penalty_value: Optional[float] = None, 
    sd: float = 0.1
) -> np.ndarray:
    """Replace NaN and infinite values with penalty plus random noise.
    
    Args:
        y (ndarray): Array of objective function values.
        penalty_value (float, optional): Value to replace NaN/inf with.
            If None, computes penalty as: max(finite_y) + 3 * std(finite_y).
            If all values are NaN/inf or only one finite value exists, 
            falls back to self.penalty. Default is None.
        sd (float): Standard deviation for random noise added to penalty.
            Default is 0.1.
    
    Returns:
        ndarray: Array with NaN/inf replaced by penalty_value + random noise.
    """
```

### Decision Tree

```
NaN/inf detected?
├─ No → Return array unchanged
└─ Yes → Determine penalty value
    ├─ penalty_value provided? → Use penalty_value
    └─ penalty_value = None?
        ├─ ≥2 finite values? → Adaptive: max(y) + 3*std(y)
        └─ <2 finite values? → Fallback: self.penalty
    
    → Add random noise: N(0, sd)
    → Replace NaN/inf with penalty + noise
    → Return cleaned array
```

### Integration with Optimization

The `_apply_penalty_NA()` method is called automatically during optimization:

1. **Initial design evaluation** (line ~1935 in SpotOptim.py)
2. **New point evaluation** (line ~2024 in SpotOptim.py)

You don't need to call it manually - it's integrated into the optimization loop.

## Comparison with Alternative Approaches

| Approach | Pros | Cons |
|----------|------|------|
| **Adaptive Penalty** (SpotOptim default) | Automatically scales to problem; robust; no configuration needed | May be conservative if problem has high variance |
| **Fixed Penalty** | Predictable; simple to understand | Requires domain knowledge; may not scale well |
| **Remove Invalid Points** | Clean dataset; no artificial values | Loses information; may fail with many invalid points |
| **Constraint Handling** | Prevents invalid evaluations | Requires analytical knowledge of valid domain |

## Frequently Asked Questions

### Q: When should I use a custom penalty value?

**A:** Use custom penalty when:
- You know the expected range of your objective function
- The adaptive penalty is too conservative or aggressive
- You want deterministic penalty behavior across runs

### Q: What if most evaluations are invalid?

**A:** SpotOptim handles this gracefully:
- Adaptive penalty uses available valid data
- Falls back to `self.penalty` if needed
- With `penalty=np.inf`, optimization still works by comparing finite values

### Q: Does penalty handling slow down optimization?

**A:** No - the overhead is negligible:
- Only processes arrays when NaN/inf detected
- Vectorized NumPy operations are very fast
- No impact on evaluations that return valid values

### Q: Can I disable penalty handling?

**A:** Not directly, but you can:
- Ensure your objective function never returns NaN/inf
- Use `penalty=0.0` (though this may cause issues)
- Better approach: fix the objective function to handle edge cases

### Q: How does this interact with noise handling?

**A:** They work together seamlessly:
- Penalty handling cleans invalid values
- Noise handling (repeats) averages valid evaluations
- Both are automatic and require no special configuration

## Example: Complete Workflow

Here's a comprehensive example showing penalty handling in a realistic scenario:

```{python}
import numpy as np
from spotoptim import SpotOptim

def complex_optimization_function(X):
    """
    Realistic function with multiple failure modes:
    - Domain restrictions
    - Numerical instabilities
    - Computational errors
    """
    results = []
    
    for params in X:
        x1, x2, x3 = params[0], params[1], params[2]
        
        try:
            # Check domain restrictions
            if x1 <= 0 or x2 <= 0:
                results.append(np.nan)
                continue
            
            # Complex calculation with potential instabilities
            with np.errstate(all='raise'):
                term1 = np.log(x1) * x2
                term2 = np.sqrt(x2 * x3)
                term3 = 1.0 / (x1 + x2 + x3)
                
                value = term1 + term2 + term3
                
                # Sanity check
                if not np.isfinite(value):
                    results.append(np.nan)
                else:
                    results.append(value)
                    
        except (FloatingPointError, ValueError, RuntimeWarning):
            results.append(np.nan)
    
    return np.array(results)

print("=" * 60)
print("Complex Optimization with Automatic Penalty Handling")
print("=" * 60)

optimizer = SpotOptim(
    fun=complex_optimization_function,
    bounds=[
        (-2, 5),    # x1: includes invalid region (x1 <= 0)
        (-2, 5),    # x2: includes invalid region (x2 <= 0)
        (-5, 5)     # x3: all valid
    ],
    var_name=["x1", "x2", "x3"],
    max_iter=50,
    n_initial=15,
    verbose=True,
    seed=42
)

result = optimizer.optimize()

print("\n" + "=" * 60)
print("Optimization Results")
print("=" * 60)
print(f"Best parameters: x1={result.x[0]:.4f}, x2={result.x[1]:.4f}, x3={result.x[2]:.4f}")
print(f"Best objective: {result.fun:.6f}")
print(f"Total evaluations: {result.nfev}")
print(f"Success: {result.success}")

# Verify solution is in valid region
assert result.x[0] > 0, "x1 should be positive"
assert result.x[1] > 0, "x2 should be positive"
print("\n✓ Solution found in valid region!")
```

## Summary

SpotOptim's adaptive penalty handling provides:

- **Automatic**: No manual configuration needed
- **Intelligent**: Adapts to your problem's scale
- **Robust**: Handles various failure modes gracefully
- **Transparent**: Verbose mode shows what's happening
- **Flexible**: Customize when needed

This feature makes SpotOptim particularly well-suited for:
- Hyperparameter optimization with unstable models
- Engineering simulations with convergence issues
- Black-box optimization with unknown failure modes
- Multi-modal problems with undefined regions

## See Also

- [Noisy Optimization](./noisy_optimization.md) - Handling stochastic objectives
- [Variable Transformations](./transformations.md) - Scale search spaces properly
- [Success Rate Tracking](./success_rate.md) - Monitor optimization progress
- [OCBA](./ocba.md) - Optimal budget allocation for noisy functions
