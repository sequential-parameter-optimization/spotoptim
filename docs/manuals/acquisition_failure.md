# Acquisition Failure Handling in SpotOptim

SpotOptim provides sophisticated fallback strategies for handling acquisition function failures during optimization. This ensures robust optimization even when the surrogate model struggles to suggest new points.

## What is Acquisition Failure?

During surrogate-based optimization, the acquisition function suggests new points to evaluate. However, sometimes the suggested point is **too close** to existing points (within `tolerance_x` distance), which would provide little new information. When this happens, SpotOptim uses a **fallback strategy** to propose an alternative point.

## Fallback Strategies

SpotOptim supports two fallback strategies, controlled by the `acquisition_failure_strategy` parameter:

### 1. Random Space-Filling Design (Default)

**Strategy name**: `"random"`

This strategy uses Latin Hypercube Sampling (LHS) to generate a new space-filling point. LHS ensures good coverage of the search space by dividing each dimension into equal-probability intervals.

**When to use**:

- General-purpose optimization
- When you want simplicity and good space-filling properties
- Default choice for most problems

**Example**:
```python
from spotoptim import SpotOptim
import numpy as np

def sphere(X):
    return np.sum(X**2, axis=1)

optimizer = SpotOptim(
    fun=sphere,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=50,
    n_initial=10,
    acquisition_failure_strategy="random",  # Default
    verbose=True
)

result = optimizer.optimize()
```

### 2. Morris-Mitchell Minimizing Point

**Strategy name**: `"mm"`

This strategy finds a point that **maximizes the minimum distance** to all existing points. It evaluates 100 candidate points and selects the one with the largest minimum distance to the already-evaluated points, providing excellent space-filling properties.

**When to use**:

- When you want to ensure maximum exploration
- For problems where avoiding clustering of points is critical
- When the search space has been heavily sampled in some regions

**Example**:
```python
from spotoptim import SpotOptim
import numpy as np

def rosenbrock(X):
    x = X[:, 0]
    y = X[:, 1]
    return (1 - x)**2 + 100 * (y - x**2)**2

optimizer = SpotOptim(
    fun=rosenbrock,
    bounds=[(-2, 2), (-2, 2)],
    max_iter=100,
    n_initial=20,
    acquisition_failure_strategy="mm",  # Morris-Mitchell
    verbose=True
)

result = optimizer.optimize()
```

## How It Works

The acquisition failure handling is integrated into the optimization process:

1. **Acquisition optimization**: SpotOptim uses differential evolution to optimize the acquisition function
2. **Distance check**: The proposed point is checked against existing points using `tolerance_x`
3. **Fallback activation**: If the point is too close, `_handle_acquisition_failure()` is called
4. **Strategy execution**: The configured fallback strategy generates a new point
5. **Evaluation**: The fallback point is evaluated and added to the dataset

## Comparison of Strategies

| Aspect | Random (LHS) | Morris-Mitchell |
|--------|--------------|-----------------|
| **Computation** | Very fast | Moderate (100 candidates) |
| **Space-filling** | Good | Excellent |
| **Exploration** | Balanced | Maximum distance |
| **Clustering avoidance** | Good | Best |
| **Recommended for** | General use | Heavily sampled spaces |

## Complete Example: Comparing Strategies

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

# Test with random strategy
print("=" * 60)
print("Testing with Random Space-Filling Strategy")
print("=" * 60)

opt_random = SpotOptim(
    fun=ackley,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=50,
    n_initial=15,
    acquisition_failure_strategy="random",
    tolerance_x=0.1,  # Relatively large tolerance to trigger failures
    seed=42,
    verbose=True
)

result_random = opt_random.optimize()

print(f"\nRandom Strategy Results:")
print(f"  Best value: {result_random.fun:.6f}")
print(f"  Best point: {result_random.x}")
print(f"  Total evaluations: {result_random.nfev}")

# Test with Morris-Mitchell strategy
print("\n" + "=" * 60)
print("Testing with Morris-Mitchell Strategy")
print("=" * 60)

opt_mm = SpotOptim(
    fun=ackley,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=50,
    n_initial=15,
    acquisition_failure_strategy="mm",
    tolerance_x=0.1,  # Same tolerance
    seed=42,
    verbose=True
)

result_mm = opt_mm.optimize()

print(f"\nMorris-Mitchell Strategy Results:")
print(f"  Best value: {result_mm.fun:.6f}")
print(f"  Best point: {result_mm.x}")
print(f"  Total evaluations: {result_mm.nfev}")

# Compare
print("\n" + "=" * 60)
print("Comparison")
print("=" * 60)
print(f"Random strategy:        {result_random.fun:.6f}")
print(f"Morris-Mitchell strategy: {result_mm.fun:.6f}")
if result_random.fun < result_mm.fun:
    print("→ Random strategy found better solution")
else:
    print("→ Morris-Mitchell strategy found better solution")
```

## Advanced Usage: Setting Tolerance

The `tolerance_x` parameter controls when the fallback strategy is triggered. A larger tolerance means points need to be farther apart, triggering the fallback more often:

```python
# Strict tolerance (smaller value) - fewer fallbacks
optimizer_strict = SpotOptim(
    fun=objective,
    bounds=bounds,
    tolerance_x=1e-6,  # Very small - almost never triggers fallback
    acquisition_failure_strategy="mm"
)

# Relaxed tolerance (larger value) - more fallbacks
optimizer_relaxed = SpotOptim(
    fun=objective,
    bounds=bounds,
    tolerance_x=0.5,  # Larger - triggers fallback more often
    acquisition_failure_strategy="mm"
)
```

## Best Practices

### 1. Use Random for Most Problems

The random strategy (default) is sufficient for most optimization problems:

```python
optimizer = SpotOptim(
    fun=objective,
    bounds=bounds,
    acquisition_failure_strategy="random"  # Good default choice
)
```

### 2. Use Morris-Mitchell for Intensive Sampling

When you have a large budget and want maximum exploration:

```python
optimizer = SpotOptim(
    fun=expensive_objective,
    bounds=bounds,
    max_iter=200,  # Large budget
    acquisition_failure_strategy="mm"  # Maximize space coverage
)
```

### 3. Monitor Fallback Activations

Enable verbose mode to see when fallbacks are triggered:

```python
optimizer = SpotOptim(
    fun=objective,
    bounds=bounds,
    acquisition_failure_strategy="mm",
    verbose=True  # Shows fallback messages
)
```

### 4. Adjust Tolerance Based on Problem Scale

For problems with small search spaces, use smaller tolerance:

```python
# Small search space
optimizer_small = SpotOptim(
    fun=objective,
    bounds=[(-1, 1), (-1, 1)],
    tolerance_x=0.01,  # Small tolerance for small space
    acquisition_failure_strategy="random"
)

# Large search space
optimizer_large = SpotOptim(
    fun=objective,
    bounds=[(-100, 100), (-100, 100)],
    tolerance_x=1.0,  # Larger tolerance for large space
    acquisition_failure_strategy="mm"
)
```

## Technical Details

### Morris-Mitchell Implementation

The Morris-Mitchell strategy:

1. Generates 100 candidate points using Latin Hypercube Sampling
2. For each candidate, calculates the minimum distance to all existing points
3. Selects the candidate with the maximum minimum distance

This ensures the new point is as far as possible from the densest region of evaluated points.

### Random Strategy Implementation

The random strategy:

1. Generates a single point using Latin Hypercube Sampling
2. Ensures the point is within bounds
3. Applies variable type repairs (rounding for int/factor variables)

This is computationally efficient while maintaining good space-filling properties.

## Summary

- **Default strategy** (`"random"`): Fast, good space-filling, suitable for most problems
- **Morris-Mitchell** (`"mm"`): Better space-filling, maximizes minimum distance, ideal for intensive sampling
- **Trigger**: Activated when acquisition-proposed point is too close to existing points (within `tolerance_x`)
- **Control**: Set via `acquisition_failure_strategy` parameter
- **Monitoring**: Enable `verbose=True` to see when fallbacks occur

Choose the strategy that best matches your optimization goals:
- Use `"random"` for general-purpose optimization
- Use `"mm"` when you want maximum exploration and have a generous function evaluation budget
