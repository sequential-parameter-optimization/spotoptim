# Deterministic Behavior Tests

## Overview

Created comprehensive pytest tests to verify both deterministic and non-deterministic behavior of the `SpotOptim` class, particularly the `_generate_initial_design()` method.

## Test File

**Location**: `tests/test_deterministic.py`

## Tests Created (10 tests)

### 1. Non-Deterministic Behavior Tests

1. **`test_generate_initial_design_without_seed_is_non_deterministic`**
   - Verifies that without a seed, different optimizer instances generate different initial designs
   - Ensures randomness is working correctly

2. **`test_optimize_without_seed_is_non_deterministic`**
   - Verifies that complete optimization runs produce different results without seed
   - Tests full optimization pipeline randomness

3. **`test_multiple_calls_with_same_seed_are_not_deterministic`**
   - Verifies that calling `_generate_initial_design()` multiple times on the same optimizer produces different results
   - Confirms that the internal sampler state advances with each call

### 2. Deterministic Behavior Tests

4. **`test_generate_initial_design_with_seed_is_deterministic`**
   - Verifies that with the same seed, different optimizer instances generate identical initial designs
   - Core reproducibility test

5. **`test_optimize_with_seed_is_deterministic`**
   - Verifies that complete optimization runs are reproducible with seed
   - Tests that all evaluated points (X_, y_) and results are identical

6. **`test_deterministic_with_provided_initial_design`**
   - Tests reproducibility when providing a custom initial design
   - Ensures seed affects subsequent iterations consistently

### 3. Seed Configuration Tests

7. **`test_generate_initial_design_different_seeds_produce_different_results`**
   - Verifies that different seed values produce different (but reproducible) results
   - Confirms seed parameter controls randomness correctly

8. **`test_seed_affects_surrogate_model`**
   - Verifies that seed is properly passed to the Gaussian Process surrogate
   - Checks `random_state` parameter of surrogate model

9. **`test_seed_parameter_types`**
   - Tests various seed parameter types (int, None, 0)
   - Ensures seed=0 works correctly (edge case)

10. **`test_reproducibility_across_dimensions`**
    - Tests that seeded behavior works across different problem dimensions (2D, 3D)
    - Verifies independent reproducibility for different problem sizes

## Test Results

```
✅ All 10 tests PASSED
✅ Total test suite: 53 tests PASSED (43 original + 10 new)
✅ No code quality issues found
```

## Key Findings

1. **Seed Parameter Works Correctly**: The `seed` parameter (line 67 in `SpotOptim.py`) successfully controls randomness for:
   - Latin Hypercube Sampling (`LatinHypercube`)
   - Gaussian Process surrogate model (`GaussianProcessRegressor`)
   - Differential evolution optimizer

2. **Reproducibility Guaranteed**: When `seed` is specified, the entire optimization process is deterministic and reproducible

3. **Default Behavior**: Without seed (`seed=None`), the optimizer produces different results on each run (expected random behavior)

## Usage Examples

### For Reproducible Results
```python
# Always specify a seed for reproducibility
optimizer = SpotOptim(
    fun=objective,
    bounds=[(-5, 5), (-5, 5)],
    seed=42  # Ensures deterministic behavior
)
result = optimizer.optimize()
```

### For Random Exploration
```python
# Omit seed for non-deterministic behavior
optimizer = SpotOptim(
    fun=objective,
    bounds=[(-5, 5), (-5, 5)]
    # No seed - different results each run
)
result = optimizer.optimize()
```

## Implementation Details

The seed controls three main random components:

1. **LHS Sampler** (line 119): `LatinHypercube(d=self.n_dim, seed=self.seed)`
2. **GP Surrogate** (line 117): `GaussianProcessRegressor(..., random_state=self.seed)`
3. **DE Optimizer** (line 462): `differential_evolution(..., seed=self.seed)`

This ensures complete reproducibility of the optimization process when a seed is provided.
