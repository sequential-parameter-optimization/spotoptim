# TensorBoard Log Cleaning Feature

## Summary

Added automatic cleaning of old TensorBoard log directories with the `tensorboard_clean` parameter.

## Changes Made

### 1. Code Changes (`src/spotoptim/SpotOptim.py`)

- **Added import**: `shutil` for directory removal
- **New parameter**: `tensorboard_clean: bool = False`
- **New method**: `_clean_tensorboard_logs()` - removes all subdirectories from 'runs' folder
- **Updated initialization**: Calls cleaning method before initializing TensorBoard writer
- **Updated docstring**: Added documentation for `tensorboard_clean` parameter

### 2. New Tests (`tests/test_tensorboard_clean.py`)

Created 12 comprehensive tests:
- Parameter validation (default False, can be enabled)
- Log removal functionality
- Integration with TensorBoard logging
- Verbose output verification
- Edge cases (missing directory, nested directories, custom paths, multiple calls)
- Files vs directories handling

### 3. Demo Script (`demo_tensorboard_clean.py`)

Demonstrates four scenarios:
1. Creating multiple old logs
2. Running without cleaning (preserves logs)
3. Running with cleaning (removes old logs)
4. Cleaning without logging (cleanup only)

### 4. Documentation (`TENSORBOARD.md`)

Updated with:
- Section on cleaning old logs
- Warning about permanent deletion
- Use cases for different combinations of parameters
- FAQ entry about managing disk space
- Reference to demo script

## Usage

### Basic Usage

```python
from spotoptim import SpotOptim

# Remove old logs and create new log directory
optimizer = SpotOptim(
    fun=objective,
    bounds=[(-5, 5), (-5, 5)],
    tensorboard_log=True,
    tensorboard_clean=True,  # Removes all subdirectories in 'runs'
    verbose=True
)

result = optimizer.optimize()
```

### Use Cases

| `tensorboard_log` | `tensorboard_clean` | Behavior |
|-------------------|---------------------|----------|
| `True` | `True` | Clean old logs, create new log directory |
| `True` | `False` | Preserve old logs, create new log directory |
| `False` | `True` | Clean old logs, no new logging |
| `False` | `False` | No logging, no cleaning (default) |

## Implementation Details

### Cleaning Method

```python
def _clean_tensorboard_logs(self) -> None:
    """Clean old TensorBoard log directories from the runs folder."""
    if self.tensorboard_clean:
        runs_dir = "runs"
        if os.path.exists(runs_dir) and os.path.isdir(runs_dir):
            # Get all subdirectories in runs
            subdirs = [
                os.path.join(runs_dir, d)
                for d in os.listdir(runs_dir)
                if os.path.isdir(os.path.join(runs_dir, d))
            ]
            
            # Remove each subdirectory
            for subdir in subdirs:
                try:
                    shutil.rmtree(subdir)
                    if self.verbose:
                        print(f"Removed old TensorBoard logs: {subdir}")
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Could not remove {subdir}: {e}")
```

### Execution Flow

1. User creates `SpotOptim` instance with `tensorboard_clean=True`
2. During initialization, `_clean_tensorboard_logs()` is called
3. Method checks if 'runs' directory exists
4. Removes all subdirectories (but preserves files)
5. If `tensorboard_log=True`, a new log directory is created
6. Optimization proceeds normally

## Safety Features

- Only removes **directories**, not files in 'runs' folder
- Handles missing 'runs' directory gracefully
- Error handling for permission issues
- Verbose output shows what's being removed
- Default is `False` to prevent accidental deletion

## Test Results

All 176 tests pass:
- 136 original SpotOptim tests
- 11 OCBA tests
- 17 TensorBoard tests
- **12 new TensorBoard cleaning tests** ✅

## Warning

⚠️ **IMPORTANT**: Setting `tensorboard_clean=True` permanently deletes all subdirectories in the 'runs' folder. Make sure to save important logs elsewhere before enabling this feature.

## Files Modified

1. `src/spotoptim/SpotOptim.py` - Main implementation
2. `tests/test_tensorboard_clean.py` - New test file
3. `demo_tensorboard_clean.py` - New demo script
4. `TENSORBOARD.md` - Updated documentation

## Backward Compatibility

✅ Fully backward compatible:
- `tensorboard_clean` defaults to `False`
- No changes to existing behavior
- All existing tests pass
- No breaking changes to API
