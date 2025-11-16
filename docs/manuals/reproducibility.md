---
title: Reproducibility in SpotOptim
sidebar_position: 5
eval: true
---

## Introduction

SpotOptim provides full support for reproducible optimization runs through the `seed` parameter. This is essential for:

- **Scientific research**: Ensuring experiments can be replicated
- **Debugging**: Reproducing specific optimization behaviors
- **Benchmarking**: Fair comparison between different configurations
- **Production**: Consistent results in deployed applications

When you specify a seed, SpotOptim guarantees that running the same optimization multiple times will produce identical results. Without a seed, each run explores the search space differently, which can be useful for robustness testing.

## Basic Usage

### Making Optimization Reproducible

To ensure reproducible results, simply specify the `seed` parameter when creating the optimizer:

```{python}
import numpy as np
from spotoptim import SpotOptim

def sphere(X):
    """Simple sphere function: f(x) = sum(x^2)"""
    return np.sum(X**2, axis=1)

# Reproducible optimization
optimizer = SpotOptim(
    fun=sphere,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=30,
    n_initial=15,
    seed=42,  # This ensures reproducibility
    verbose=True
)

result = optimizer.optimize()
print(f"Best solution: {result.x}")
print(f"Best value: {result.fun}")
```

**Key Point**: Running this code multiple times (even on different days or machines) will always produce the same result.

### Running Independent Experiments

If you don't specify a seed, each optimization run will explore the search space differently:

```{python}
# Non-reproducible: different results each time
optimizer = SpotOptim(
    fun=sphere,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=30,
    n_initial=15
    # No seed specified
)

result = optimizer.optimize()
# Results will vary between runs
```

This is useful when you want to:
- Explore different regions of the search space
- Test the robustness of your results
- Run multiple independent optimization attempts

## Practical Examples

### Example 1: Comparing Different Configurations

When comparing different optimizer settings, use the same seed for fair comparison:

```{python}
import numpy as np
from spotoptim import SpotOptim

def rosenbrock(X):
    """Rosenbrock function"""
    x = X[:, 0]
    y = X[:, 1]
    return (1 - x)**2 + 100 * (y - x**2)**2

# Configuration 1: More initial points
opt1 = SpotOptim(
    fun=rosenbrock,
    bounds=[(-2, 2), (-2, 2)],
    max_iter=50,
    n_initial=20,
    seed=42  # Same seed for fair comparison
)
result1 = opt1.optimize()

# Configuration 2: Fewer initial points, more iterations
opt2 = SpotOptim(
    fun=rosenbrock,
    bounds=[(-2, 2), (-2, 2)],
    max_iter=50,
    n_initial=10,
    seed=42  # Same seed
)
result2 = opt2.optimize()

print(f"Config 1 (more initial): {result1.fun:.6f}")
print(f"Config 2 (fewer initial): {result2.fun:.6f}")
```

### Example 2: Reproducible Research Experiment

For scientific papers or reports, always use a fixed seed and document it:

```{python}
import numpy as np
from spotoptim import SpotOptim

def rastrigin(X):
    """Rastrigin function (multimodal)"""
    A = 10
    n = X.shape[1]
    return A * n + np.sum(X**2 - A * np.cos(2 * np.pi * X), axis=1)

# Documented seed for reproducibility
RANDOM_SEED = 12345

optimizer = SpotOptim(
    fun=rastrigin,
    bounds=[(-5.12, 5.12), (-5.12, 5.12), (-5.12, 5.12)],
    max_iter=100,
    n_initial=30,
    seed=RANDOM_SEED,
    verbose=True
)

result = optimizer.optimize()

print(f"\nExperiment Results (seed={RANDOM_SEED}):")
print(f"Best solution: {result.x}")
print(f"Best value: {result.fun}")
print(f"Iterations: {result.nit}")
print(f"Function evaluations: {result.nfev}")

# These results can now be cited in a paper
```

### Example 3: Multiple Independent Runs

To test robustness, run the same optimization with different seeds:

```{python}
import numpy as np
from spotoptim import SpotOptim

def ackley(X):
    """Ackley function"""
    a = 20
    b = 0.2
    c = 2 * np.pi
    n = X.shape[1]
    
    sum_sq = np.sum(X**2, axis=1)
    sum_cos = np.sum(np.cos(c * X), axis=1)
    
    return -a * np.exp(-b * np.sqrt(sum_sq / n)) - np.exp(sum_cos / n) + a + np.e

# Run 5 independent optimizations
results = []
seeds = [42, 123, 456, 789, 1011]

for seed in seeds:
    optimizer = SpotOptim(
        fun=ackley,
        bounds=[(-5, 5), (-5, 5)],
        max_iter=40,
        n_initial=20,
        seed=seed,
        verbose=False
    )
    result = optimizer.optimize()
    results.append(result.fun)
    print(f"Run with seed {seed:4d}: f(x) = {result.fun:.6f}")

# Analyze robustness
print(f"\nBest result: {min(results):.6f}")
print(f"Worst result: {max(results):.6f}")
print(f"Mean: {np.mean(results):.6f}")
print(f"Std dev: {np.std(results):.6f}")
```

### Example 4: Reproducible Initial Design

The seed ensures that even the initial design points are reproducible:

```{python}
import numpy as np
from spotoptim import SpotOptim

def simple_quadratic(X):
    return np.sum((X - 1)**2, axis=1)

# Create two optimizers with same seed
opt1 = SpotOptim(
    fun=simple_quadratic,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=25,
    n_initial=10,
    seed=999
)

opt2 = SpotOptim(
    fun=simple_quadratic,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=25,
    n_initial=10,
    seed=999  # Same seed
)

# Run both optimizations
result1 = opt1.optimize()
result2 = opt2.optimize()

# Verify identical results
print("Initial design points are identical:", 
      np.allclose(opt1.X_[:10], opt2.X_[:10]))
print("All evaluated points are identical:", 
      np.allclose(opt1.X_, opt2.X_))
print("All function values are identical:", 
      np.allclose(opt1.y_, opt2.y_))
print("Best solutions are identical:", 
      np.allclose(result1.x, result2.x))
```

### Example 5: Custom Initial Design with Seed

Even when providing a custom initial design, the seed ensures reproducible subsequent iterations:

```{python}
import numpy as np
from spotoptim import SpotOptim

def beale(X):
    """Beale function"""
    x = X[:, 0]
    y = X[:, 1]
    term1 = (1.5 - x + x * y)**2
    term2 = (2.25 - x + x * y**2)**2
    term3 = (2.625 - x + x * y**3)**2
    return term1 + term2 + term3

# Custom initial design (e.g., from previous knowledge)
X_start = np.array([
    [0.0, 0.0],
    [1.0, 1.0],
    [2.0, 2.0],
    [-1.0, -1.0]
])

# Run twice with same seed and initial design
opt1 = SpotOptim(
    fun=beale,
    bounds=[(-4.5, 4.5), (-4.5, 4.5)],
    max_iter=30,
    n_initial=10,
    seed=777
)
result1 = opt1.optimize(X0=X_start)

opt2 = SpotOptim(
    fun=beale,
    bounds=[(-4.5, 4.5), (-4.5, 4.5)],
    max_iter=30,
    n_initial=10,
    seed=777  # Same seed
)
result2 = opt2.optimize(X0=X_start)

print("Results are identical:", np.allclose(result1.x, result2.x))
print(f"Best value: {result1.fun:.6f}")
```

## Advanced Topics

### Seed and Noisy Functions

When optimizing noisy functions with repeated evaluations, the seed ensures reproducible noise:

```{python}
import numpy as np
from spotoptim import SpotOptim

def noisy_sphere(X):
    """Sphere function with Gaussian noise"""
    base = np.sum(X**2, axis=1)
    noise = np.random.normal(0, 0.1, size=base.shape)
    return base + noise

optimizer = SpotOptim(
    fun=noisy_sphere,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=40,
    n_initial=20,
    repeats_initial=3,  # 3 evaluations per point
    repeats_surrogate=2,
    seed=42  # Ensures same noise pattern
)

result = optimizer.optimize()
print(f"Best mean value: {optimizer.min_mean_y:.6f}")
print(f"Variance at best: {optimizer.min_var_y:.6f}")
```

**Important**: With the same seed, even the noise will be identical across runs!

### Different Seeds for Different Exploration

Use different seeds to explore different regions systematically:

```{python}
import numpy as np
from spotoptim import SpotOptim

def griewank(X):
    """Griewank function"""
    sum_sq = np.sum(X**2 / 4000, axis=1)
    prod_cos = np.prod(np.cos(X / np.sqrt(np.arange(1, X.shape[1] + 1))), axis=1)
    return sum_sq - prod_cos + 1

# Systematic exploration with different seeds
best_overall = float('inf')
best_seed = None

for seed in range(10, 20):  # Seeds 10-19
    optimizer = SpotOptim(
        fun=griewank,
        bounds=[(-600, 600), (-600, 600)],
        max_iter=50,
        n_initial=25,
        seed=seed
    )
    result = optimizer.optimize()
    
    if result.fun < best_overall:
        best_overall = result.fun
        best_seed = seed
    
    print(f"Seed {seed}: f(x) = {result.fun:.6f}")

print(f"\nBest result with seed {best_seed}: {best_overall:.6f}")
```

## Best Practices

### 1. Always Use Seeds for Production Code

```{python}
#| eval: false
# Good: Reproducible
optimizer = SpotOptim(fun=objective, bounds=bounds, seed=42)

# Risky: Non-reproducible
optimizer = SpotOptim(fun=objective, bounds=bounds)
```

### 2. Document Your Seeds

```{python}
#| eval: false
# Configuration for experiment reported in Section 4.2
EXPERIMENT_SEED = 2024
MAX_ITERATIONS = 100

optimizer = SpotOptim(
    fun=my_objective,
    bounds=my_bounds,
    max_iter=MAX_ITERATIONS,
    seed=EXPERIMENT_SEED
)
```

### 3. Use Different Seeds for Different Experiments

```{python}
#| eval: false
# Different experiments should use different seeds
BASELINE_SEED = 100
EXPERIMENT_A_SEED = 200
EXPERIMENT_B_SEED = 300
```

### 4. Test Robustness Across Multiple Seeds

```{python}
#| eval: false
# Run same optimization with multiple seeds
for seed in [42, 123, 456, 789, 1011]:
    optimizer = SpotOptim(fun=objective, bounds=bounds, seed=seed)
    result = optimizer.optimize()
    # Analyze results
```

## What the Seed Controls

The `seed` parameter ensures reproducibility by controlling:

1. **Initial Design Generation**: Latin Hypercube Sampling produces the same initial points
2. **Surrogate Model**: Gaussian Process random initialization is identical
3. **Acquisition Optimization**: Differential evolution explores the same candidates
4. **Random Sampling**: Any random exploration uses the same random numbers

This guarantees that the entire optimization pipeline is deterministic and reproducible.

## Common Questions

**Q: Can I use seed=0?**  
A: Yes, any integer (including 0) is a valid seed.

**Q: Will different Python versions give the same results?**  
A: Generally yes, but minor numerical differences may occur due to underlying library changes. Use the same environment for exact reproducibility.

**Q: Does the seed affect the objective function?**  
A: No, the seed only affects SpotOptim's internal random processes. If your objective function has its own randomness, you'll need to control that separately.

**Q: How do I choose a good seed value?**  
A: Any integer works. Common choices are 42, 123, or dates (e.g., 20241112). What matters is consistency, not the specific value.

## Summary

- Use `seed` parameter for reproducible optimization
- Same seed → identical results (every time)
- No seed → different results (random exploration)  
- Essential for research, debugging, and production
- Document your seeds for transparency
- Test robustness with multiple different seeds
