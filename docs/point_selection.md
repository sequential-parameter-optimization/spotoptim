# Point Selection Implementation

## Overview

This feature automatically selects a subset of evaluated points for surrogate model training when the total number of points exceeds a specified threshold.

It is implemented as a point selection mechanism for SpotOptim that mirrors the functionality in spotpython's `Spot` class. 

## Implementation Details

### Parameters

Added to `SpotOptim.__init__`:

- `max_surrogate_points` (int, optional): Maximum number of points to use for surrogate fitting
- `selection_method` (str, default='distant'): Method for selecting points ('distant' or 'best')

### Methods

1. **`_select_distant_points(X, y, k)`**

   - Uses K-means clustering to find k clusters
   - Selects the point closest to each cluster center
   - Ensures space-filling properties for surrogate training
   - Mimics `spotpython.utils.aggregate.select_distant_points`

2. **`_select_best_cluster(X, y, k)`**

   - Uses K-means clustering to find k clusters
   - Computes mean objective value for each cluster
   - Selects all points from the cluster with the best (lowest) mean value
   - Mimics `spotpython.utils.aggregate.select_best_cluster`

3. **`_selection_dispatcher(X, y)`**

   - Dispatcher method that routes to the appropriate selection function
   - Returns all points if `max_surrogate_points` is None
   - Mimics `spotpython.spot.spot.Spot.selection_dispatcher`

The method `_fit_surrogate(X, y)` checks if `X.shape[0] > self.max_surrogate_points`. If true, it calls `_selection_dispatcher` to get a subset. Then, it fits the surrogate only on the selected points.  This implementation matches the logic in `spotpython.spot.spot.Spot.fit_surrogate`

## Key Differences from spotpython

While the implementation follows spotpython's design, there is a difference: `spotoptim` uses a simplified clustering, it uses sklearn's KMeans directly instead of a custom implementation.


## Example Usage

```python
from spotoptim import SpotOptim

# Without point selection (default behavior)
optimizer1 = SpotOptim(
    fun=expensive_function,
    bounds=bounds,
    max_iter=100,
    n_initial=20
)

# With point selection using distant method
optimizer2 = SpotOptim(
    fun=expensive_function,
    bounds=bounds,
    max_iter=100,
    n_initial=20,
    max_surrogate_points=50,
    selection_method='distant'
)

# With point selection using best cluster method
optimizer3 = SpotOptim(
    fun=expensive_function,
    bounds=bounds,
    max_iter=100,
    n_initial=20,
    max_surrogate_points=50,
    selection_method='best'
)
```

## Benefits

1. **Scalability**: Enables efficient optimization with many function evaluations
2. **Computational efficiency**: Reduces surrogate training time for large datasets
3. **Maintained accuracy**: Careful point selection preserves model quality
4. **Flexibility**: Two selection methods for different optimization scenarios


## Comparison with spotpython

| Feature | spotpython | SpotOptim |
|---------|-----------|-----------|
| Point selection via clustering | ✓ | ✓ |
| 'distant' method | ✓ | ✓ |
| 'best' method | ✓ | ✓ |
| Selection dispatcher | ✓ | ✓ |
| Nyström approximation | ✓ | ✗ |
| Modular design | ✓ (utils.aggregate) | ✓ (class methods) |

## References

- spotpython implementation: `src/spotpython/spot/spot.py` lines 1646-1778
- spotpython utilities: `src/spotpython/utils/aggregate.py` lines 262-336
