# Size-Invariant Morris-Mitchell Criterion

The Morris-Mitchell criterion is a widely used metric for evaluating the space-filling properties of a sampling plan (also known as a design of experiments). It combines the concept of maximizing the minimum distance between points with minimizing the number of pairs of points separated by that distance.

## The Standard Morris-Mitchell Criterion

The standard Morris-Mitchell criterion, $\Phi_q$, is defined as:

$$
\Phi_q = \left( \sum_{i=1}^{m} J_i d_i^{-q} \right)^{1/q}
$$

where:

- $d_1 < d_2 < \dots < d_m$ are the distinct distances between all pairs of points in the sampling plan.
- $J_i$ is the number of pairs of points separated by the distance $d_i$.
- $q$ is a large positive integer (e.g., $q=2, 5, 10, \dots$).

As $q \to \infty$, minimizing $\Phi_q$ is equivalent to maximizing the minimum distance $d_1$, and then minimizing the number of pairs $J_1$ at that distance, and so on.

## The Size-Invariant (Intensive) Criterion

One limitation of the standard $\Phi_q$ is that its value depends on the number of points in the sampling plan. If you double the number of points, the number of pairs increases quadratically, which inflates the value of $\Phi_q$. This makes it difficult to compare the quality of sampling plans with different sample sizes.

To address this, we can use a **size-invariant** or **intensive** version of the criterion. This is achieved by normalizing the sum by the total number of pairs, $M = \binom{n}{2} = \frac{n(n-1)}{2}$.

The intensive criterion is defined as:

$$
\Phi_{q, \text{intensive}} = \left( \frac{1}{M} \sum_{i=1}^{m} J_i d_i^{-q} \right)^{1/q}
$$

This normalization allows for a fairer comparison between designs of different sizes. A lower value still indicates a better space-filling property.

## Using `mmphi_intensive` in `spotoptim`

The `spotoptim` library provides the `mmphi_intensive` function to calculate this metric.

### Example

```python
import numpy as np
from spotoptim.sampling.mm import mmphi_intensive

# Create a simple 3-point sampling plan in 2D
X = np.array([
    [0.0, 0.0],
    [0.5, 0.5],
    [1.0, 1.0]
])

# Calculate the intensive space-fillingness metric with q=2, using Euclidean distances (p=2)
quality, J, d = mmphi_intensive(X, q=2, p=2)

print(f"Quality (Phi_q_intensive): {quality}")
print(f"Multiplicities (J): {J}")
print(f"Distinct Distances (d): {d}")
```

### Interpretation

- **Quality**: The returned `quality` is the $\Phi_{q, \text{intensive}}$ value. Lower is better.
- **J**: The array of multiplicities for each distinct distance.
- **d**: The array of distinct distances found in the design.

This metric is particularly useful when you are optimizing the sample size itself, or when comparing designs from different literature sources that use different $n$.

## Efficient Updates with `mmphi_intensive_update`

When constructing a design sequentially (e.g., adding one point at a time), recalculating the full distance matrix and metric from scratch can be inefficient. The `mmphi_intensive_update` function allows you to update the metric efficiently by only computing distances between the new point and the existing points.

### Example

```python
from spotoptim.sampling.mm import mmphi_intensive_update

# Assume we have the state from the previous example: X, quality, J, d
# New point to add
new_point = np.array([0.1, 0.1])

# Update the intensive criterion
new_quality, new_J, new_d = mmphi_intensive_update(X, new_point, J, d, q=2, p=2)

print(f"Updated Quality: {new_quality}")
```

This function returns the updated metric, multiplicities, and distinct distances, which can be used for the next update step.
