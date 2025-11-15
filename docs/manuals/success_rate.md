# Success Rate Tracking in SpotOptim

SpotOptim tracks the **success rate** of the optimization process, which measures how often the optimizer finds improvements over recent evaluations. This metric helps you understand whether the optimization is making progress or has stalled.

## What is Success Rate?

The success rate is a **rolling metric** that tracks the percentage of recent evaluations that improved upon the best value found so far. It's calculated over a sliding window of the last 100 evaluations.

**Key Points:**
- A "success" occurs when a new evaluation finds a value **better (smaller)** than the best found so far
- The rate is computed over the last **100 evaluations** (window size)
- Values range from **0.0** (no recent improvements) to **1.0** (all recent evaluations improved)
- Helps identify when optimization is stalling and may need adjustment


## First Example

* Start `TensorBoard` to visualize success rate in real-time:

```bash
tensorboard --logdir=runs
```

The execute the following code:

```{python}
import numpy as np
from spotoptim import SpotOptim

def rosenbrock(X):
    """Rosenbrock function - challenging optimization problem"""
    x = X[:, 0]
    y = X[:, 1]
    return (1 - x)**2 + 100 * (y - x**2)**2

# Run optimization with periodic success rate checks
optimizer = SpotOptim(
    fun=rosenbrock,
    bounds=[(-2, 2), (-2, 2)],
    max_iter=200,
    n_initial=20,
    tensorboard_log=True,
    tensorboard_clean=True,
    seed=42
)

result = optimizer.optimize()

# Analyze final success rate
print(f"\nOptimization Results:")
print(f"Best value: {result.fun:.6f}")
print(f"Total evaluations: {optimizer.counter}")
print(f"Final success rate: {optimizer.success_rate:.2%}")

# Interpret the result
if optimizer.success_rate > 0.5:
    print("→ High success rate: Optimization is still making good progress")
elif optimizer.success_rate > 0.2:
    print("→ Medium success rate: Approaching convergence")
else:
    print("→ Low success rate: Optimization has likely converged")
```



## Second Example

```python
from spotoptim import SpotOptim
import numpy as np

def sphere(X):
    """Simple sphere function: f(x) = sum(x^2)"""
    return np.sum(X**2, axis=1)

# Create optimizer
optimizer = SpotOptim(
    fun=sphere,
    bounds=[(-5, 5), (-5, 5), (-5, 5)],
    max_iter=100,
    n_initial=20,
    verbose=True
)

# Run optimization
result = optimizer.optimize()

# Check success rate
print(f"Final success rate: {optimizer.success_rate:.2%}")
print(f"Total evaluations: {optimizer.counter}")
```

## Accessing Success Rate

The success rate is stored in the `success_rate` attribute:

```python
optimizer = SpotOptim(fun=objective, bounds=bounds, max_iter=50)
result = optimizer.optimize()

# Access success rate
current_rate = optimizer.success_rate
print(f"Success rate: {current_rate:.2%}")

# Also available via getter method
rate = optimizer._get_success_rate()
```

## Interpreting Success Rate

### High Success Rate (> 0.5)
```
Success Rate: 75%
```

**Interpretation:** The optimizer is finding improvements frequently. This typically indicates:
- The optimization is in an exploratory phase
- The surrogate model is effectively guiding the search
- There's still room for improvement in the search space

**Action:** Continue optimization - progress is good!

### Medium Success Rate (0.2 - 0.5)
```
Success Rate: 35%
```

**Interpretation:** The optimizer occasionally finds improvements. This suggests:
- The search is becoming more refined
- The optimizer is balancing exploration and exploitation
- Approaching a local or global optimum

**Action:** Monitor progress and consider stopping criteria.

### Low Success Rate (< 0.2)
```
Success Rate: 8%
```

**Interpretation:** Few recent evaluations improve the best value. This may indicate:
- The optimization has converged to a (local) optimum
- The search is stuck in a plateau region
- The budget may be exhausted in terms of meaningful progress

**Action:** Consider stopping optimization or adjusting parameters.

## TensorBoard Visualization

When TensorBoard logging is enabled, success rate is automatically logged and can be visualized in real-time:

```python
optimizer = SpotOptim(
    fun=objective,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=100,
    n_initial=20,
    tensorboard_log=True,  # Enable logging
    verbose=True
)

result = optimizer.optimize()
```

**View in TensorBoard:**
```bash
tensorboard --logdir=runs
```

In the TensorBoard interface, look for:
- **SCALARS** tab → `success_rate`: Rolling success rate over iterations
- Compare multiple runs side-by-side
- Identify when optimization stalls

## Example: Monitoring Optimization Progress

```python
import numpy as np
from spotoptim import SpotOptim

def rosenbrock(X):
    """Rosenbrock function - challenging optimization problem"""
    x = X[:, 0]
    y = X[:, 1]
    return (1 - x)**2 + 100 * (y - x**2)**2

# Run optimization with periodic success rate checks
optimizer = SpotOptim(
    fun=rosenbrock,
    bounds=[(-2, 2), (-2, 2)],
    max_iter=100,
    n_initial=20,
    tensorboard_log=True,
    seed=42
)

result = optimizer.optimize()

# Analyze final success rate
print(f"\nOptimization Results:")
print(f"Best value: {result.fun:.6f}")
print(f"Total evaluations: {optimizer.counter}")
print(f"Final success rate: {optimizer.success_rate:.2%}")

# Interpret the result
if optimizer.success_rate > 0.5:
    print("→ High success rate: Optimization is still making good progress")
elif optimizer.success_rate > 0.2:
    print("→ Medium success rate: Approaching convergence")
else:
    print("→ Low success rate: Optimization has likely converged")
```

## Example: Comparing Multiple Runs

```python
import numpy as np
from spotoptim import SpotOptim

def ackley(X):
    """Ackley function - multimodal test function"""
    a = 20
    b = 0.2
    c = 2 * np.pi
    n = X.shape[1]
    
    sum_sq = np.sum(X**2, axis=1)
    sum_cos = np.sum(np.cos(c * X), axis=1)
    
    return -a * np.exp(-b * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + a + np.e

# Run with different configurations
configs = [
    {"n_initial": 10, "max_iter": 50, "name": "Small initial"},
    {"n_initial": 30, "max_iter": 50, "name": "Large initial"},
]

results = []
for config in configs:
    optimizer = SpotOptim(
        fun=ackley,
        bounds=[(-5, 5), (-5, 5)],
        n_initial=config["n_initial"],
        max_iter=config["max_iter"],
        seed=42,
        verbose=False
    )
    result = optimizer.optimize()
    
    results.append({
        "name": config["name"],
        "best_value": result.fun,
        "success_rate": optimizer.success_rate,
        "n_evals": optimizer.counter
    })
    
    print(f"\n{config['name']}:")
    print(f"  Best value: {result.fun:.6f}")
    print(f"  Success rate: {optimizer.success_rate:.2%}")
    print(f"  Evaluations: {optimizer.counter}")

# Find best configuration
best = min(results, key=lambda x: x["best_value"])
print(f"\nBest configuration: {best['name']}")
print(f"  Achieved: f(x) = {best['best_value']:.6f}")
print(f"  Final success rate: {best['success_rate']:.2%}")
```

## Success Rate with Noisy Functions

For noisy functions (when `repeats_initial > 1` or `repeats_surrogate > 1`), the success rate tracks improvements in the **raw** y values, not the aggregated means:

```python
import numpy as np
from spotoptim import SpotOptim

def noisy_sphere(X):
    """Sphere function with Gaussian noise"""
    base = np.sum(X**2, axis=1)
    noise = np.random.normal(0, 0.5, size=base.shape)
    return base + noise

optimizer = SpotOptim(
    fun=noisy_sphere,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=50,
    n_initial=15,
    repeats_initial=3,    # 3 evaluations per initial point
    repeats_surrogate=2,  # 2 evaluations per new point
    seed=42,
    verbose=True
)

result = optimizer.optimize()

print(f"\nNoisy Optimization Results:")
print(f"Best raw value: {optimizer.min_y:.6f}")
print(f"Best mean value: {optimizer.min_mean_y:.6f}")
print(f"Success rate: {optimizer.success_rate:.2%}")
print(f"Total evaluations: {optimizer.counter}")
print(f"Unique design points: {optimizer.mean_X.shape[0]}")
```

**Note:** With noisy functions, the success rate may be lower because:
- Noise can mask true improvements
- Multiple evaluations of the same point contribute to the window
- Focus on the mean values (`min_mean_y`) for better assessment

## Advanced: Custom Window Size

The success rate is calculated over a window of 100 evaluations by default. This is controlled by the `window_size` attribute:

```python
optimizer = SpotOptim(
    fun=objective,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=50,
    n_initial=10
)

# Check default window size
print(f"Window size: {optimizer.window_size}")  # 100

# The window size is set during initialization
# To use a different window, you would need to modify it
# before running optimization (not typically recommended)
```

## Best Practices

### 1. Monitor During Long Runs

For expensive optimization runs, periodically check success rate:

```python
# Could be implemented with callbacks in future versions
# For now, success rate is updated automatically and logged to TensorBoard
```

### 2. Combine with TensorBoard

Always enable TensorBoard logging for visual monitoring:

```python
optimizer = SpotOptim(
    fun=expensive_function,
    bounds=bounds,
    max_iter=1000,
    tensorboard_log=True,  # Track success_rate visually
    tensorboard_path="runs/long_optimization"
)
```

### 3. Use as Stopping Criterion

Consider stopping when success rate drops very low:

```python
# Manual stopping check (conceptual)
if optimizer.success_rate < 0.05 and optimizer.counter > 50:
    print("Success rate very low - optimization has likely converged")
```

### 4. Compare Different Strategies

Use success rate to compare optimization strategies:

```python
strategies = ["ei", "pi", "y"]  # Different acquisition functions
for acq in strategies:
    opt = SpotOptim(fun=obj, bounds=bnds, acquisition=acq, max_iter=50)
    result = opt.optimize()
    print(f"{acq}: success_rate={opt.success_rate:.2%}, best={result.fun:.6f}")
```

## Technical Details

### How Success is Counted

A new evaluation `y_new` is considered a success if:

```python
y_new < best_y_so_far
```

where `best_y_so_far` is the minimum value found in all previous evaluations.

### Rolling Window Calculation

The success rate is computed as:

```python
success_rate = (number of successes in last 100 evals) / (window size)
```

- Window size defaults to 100
- If fewer than 100 evaluations have been performed, the window size is the number of evaluations
- The window slides forward with each new evaluation

### Update Frequency

The success rate is updated after:
1. Initial design evaluation
2. Each iteration's new point evaluation(s)
3. OCBA re-evaluations (if applicable)

## Summary

- **Success rate** measures the percentage of recent evaluations that improve the best value
- Calculated over a rolling window of the last **100 evaluations**
- Values range from **0.0** to **1.0**
- High rates (>0.5) indicate active progress
- Low rates (<0.2) suggest convergence
- Automatically logged to **TensorBoard** when logging is enabled
- Available via `optimizer.success_rate` attribute after optimization

Use success rate to:
- ✓ Monitor optimization progress in real-time
- ✓ Identify when to stop optimization
- ✓ Compare different optimization strategies
- ✓ Assess optimization difficulty for different problems
