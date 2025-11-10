# Point Selection Implementation Summary

## Overview

Implemented a point selection mechanism for SpotOptim that mirrors the functionality in spotpython's `Spot` class. This feature automatically selects a subset of evaluated points for surrogate model training when the total number of points exceeds a specified threshold.

## Implementation Details

### New Parameters

Added to `SpotOptim.__init__`:
- `max_surrogate_points` (int, optional): Maximum number of points to use for surrogate fitting
- `selection_method` (str, default='distant'): Method for selecting points ('distant' or 'best')

### New Methods

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

### Modified Methods

**`_fit_surrogate(X, y)`**
- Now checks if `X.shape[0] > self.max_surrogate_points`
- If true, calls `_selection_dispatcher` to get a subset
- Fits the surrogate only on the selected points
- Matches the logic in `spotpython.spot.spot.Spot.fit_surrogate`

## Key Differences from spotpython

While the implementation follows spotpython's design, there are some differences:

1. **No Nyström approximation**: SpotOptim doesn't implement the Nyström approximation option that spotpython uses
2. **Simplified clustering**: Uses sklearn's KMeans directly instead of custom implementations
3. **All methods in the class**: All selection methods belong to the SpotOptim class (no separate utility module)
4. **Consistent naming**: Methods follow Python naming conventions with leading underscores for internal methods

## Testing

Created comprehensive test suite (`tests/test_point_selection.py`) with 11 tests:

1. Basic functionality of `_select_distant_points`
2. Basic functionality of `_select_best_cluster`
3. Selection dispatcher with 'distant' method
4. Selection dispatcher with 'best' method
5. Selection dispatcher when no limit is set
6. Surrogate fitting with point selection
7. Full optimization with max_surrogate_points
8. Full optimization with 'best' selection
9. Error handling for too few points
10. Handling of duplicate points
11. Verbose output verification

**All tests pass** (43/43 total tests in the suite).

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

## Files Modified

1. `src/spotoptim/SpotOptim.py`: Added new parameters and methods
2. `README.md`: Added documentation for the new feature
3. `tests/test_point_selection.py`: Comprehensive test suite
4. `examples/point_selection_example.py`: Demonstration example

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
