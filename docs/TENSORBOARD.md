# TensorBoard Logging in SpotOptim

SpotOptim supports TensorBoard logging for monitoring optimization progress in real-time.

## Quick Start

### 1. Enable TensorBoard Logging

```python
from spotoptim import SpotOptim
import numpy as np

def objective(X):
    return np.sum(X**2, axis=1)

optimizer = SpotOptim(
    fun=objective,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=50,
    n_initial=15,
    tensorboard_log=True,  # Enable logging
    verbose=True
)

result = optimizer.optimize()
```

### 2. View Logs in TensorBoard

In a separate terminal, run:

```bash
tensorboard --logdir=runs
```

Then open your browser to [http://localhost:6006](http://localhost:6006)

## Cleaning Old Logs

You can automatically remove old TensorBoard logs before starting a new optimization:

```python
optimizer = SpotOptim(
    fun=objective,
    bounds=[(-5, 5), (-5, 5)],
    tensorboard_log=True,
    tensorboard_clean=True,  # Remove old logs from 'runs' directory
    verbose=True
)
```

**Warning:** This permanently deletes all subdirectories in the `runs` folder. Make sure to save important logs elsewhere before enabling this feature.

### Use Cases

1. **Clean Start** - Remove old logs and create new one:
   ```python
   tensorboard_log=True, tensorboard_clean=True
   ```

2. **Preserve History** - Keep old logs and add new one (default):
   ```python
   tensorboard_log=True, tensorboard_clean=False
   ```

3. **Just Clean** - Remove old logs without new logging:
   ```python
   tensorboard_log=False, tensorboard_clean=True
   ```

## Custom Log Directory

Specify a custom path for TensorBoard logs:

```python
optimizer = SpotOptim(
    fun=objective,
    bounds=[(-5, 5), (-5, 5)],
    tensorboard_log=True,
    tensorboard_path="my_experiments/run_001",
    ...
)
```

## What Gets Logged

### Scalar Metrics

**For Deterministic Functions:**

- `y_values/min`: Best (minimum) y value found so far
- `y_values/last`: Most recently evaluated y value
- `X_best/x0, X_best/x1, ...`: Coordinates of the best point

**For Noisy Functions (repeats > 1):**

- `y_values/min`: Best single evaluation
- `y_values/mean_best`: Best mean y value
- `y_values/last`: Most recent evaluation
- `y_variance_at_best`: Variance at the best mean point
- `X_mean_best/x0, X_mean_best/x1, ...`: Coordinates of best mean point

### Hyperparameters

Each function evaluation is logged with:

- Input coordinates (x0, x1, x2, ...)
- Function value (hp_metric)

This allows you to explore the relationship between hyperparameters and objective values in the HPARAMS tab.

## Examples

### Basic Usage

```python
optimizer = SpotOptim(
    fun=lambda X: np.sum(X**2, axis=1),
    bounds=[(-5, 5), (-5, 5)],
    max_iter=30,
    tensorboard_log=True,
    verbose=True
)
result = optimizer.optimize()
```

### Noisy Optimization

```python
def noisy_objective(X):
    base = np.sum(X**2, axis=1)
    noise = np.random.normal(0, 0.1, size=base.shape)
    return base + noise

optimizer = SpotOptim(
    fun=noisy_objective,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=50,
    repeats_initial=3,
    repeats_surrogate=2,
    tensorboard_log=True,
    tensorboard_path="runs/noisy_exp",
    seed=42
)
result = optimizer.optimize()
```

### With OCBA

```python
optimizer = SpotOptim(
    fun=noisy_objective,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=50,
    repeats_initial=2,
    ocba_delta=3,  # Re-evaluate 3 promising points per iteration
    tensorboard_log=True,
    seed=42
)
result = optimizer.optimize()
```

## Comparing Multiple Runs

Run multiple optimizations with different settings:

```python
# Run 1: Standard
opt1 = SpotOptim(..., tensorboard_path="runs/standard")
opt1.optimize()

# Run 2: With OCBA
opt2 = SpotOptim(..., ocba_delta=3, tensorboard_path="runs/with_ocba")
opt2.optimize()

# Run 3: More initial points
opt3 = SpotOptim(..., n_initial=20, tensorboard_path="runs/more_initial")
opt3.optimize()
```

Then view all runs together:
```bash
tensorboard --logdir=runs
```

## TensorBoard Features

### SCALARS Tab

- View convergence curves
- Compare optimization progress across runs
- Track how metrics change over iterations

### HPARAMS Tab

- Explore hyperparameter space
- See which parameter combinations work best
- Identify patterns in successful configurations

### Text Tab

- View configuration details
- Check run metadata

## Tips

1. **Organize Experiments**: Use descriptive tensorboard_path names:
   ```python
   tensorboard_path=f"runs/exp_{date}_{config_name}"
   ```

2. **Compare Algorithms**: Run multiple optimization strategies and compare:
   ```python
   # Different acquisition functions
   for acq in ['ei', 'pi', 'y']:
       opt = SpotOptim(..., acquisition=acq, tensorboard_path=f"runs/acq_{acq}")
       opt.optimize()
   ```

3. **Clean Up Old Runs**: Use `tensorboard_clean=True` for automatic cleanup, or manually:
   ```bash
   rm -rf runs/old_experiment
   ```

4. **Port Conflicts**: If port 6006 is busy, use a different port:
   ```bash
   tensorboard --logdir=runs --port=6007
   ```

## Demo Scripts

Run the comprehensive TensorBoard demo:
```bash
python demo_tensorboard.py
```

This demonstrates:

- Deterministic optimization (Rosenbrock function)
- Noisy optimization with repeated evaluations
- OCBA for intelligent re-evaluation

Run the log cleaning demo:
```bash
python demo_tensorboard_clean.py
```

This demonstrates:

- Creating multiple log directories
- Preserving old logs (default behavior)
- Cleaning old logs automatically
- Cleaning without creating new logs

This demonstrates:

- Deterministic optimization (Rosenbrock function)
- Noisy optimization with repeated evaluations
- OCBA for intelligent re-evaluation

## Troubleshooting

**Q: TensorBoard shows "No dashboards are active"**
A: Make sure you've run an optimization with `tensorboard_log=True` first.

**Q: Can't see my latest run**
A: Refresh TensorBoard (click the reload button in the upper right).

**Q: How do I stop TensorBoard?**
A: Press Ctrl+C in the terminal where TensorBoard is running.

**Q: Logs taking up too much space?**
A: Use `tensorboard_clean=True` to automatically remove old logs, or manually delete old run directories.

**Q: How do I remove all old logs at once?**
A: Set `tensorboard_clean=True` when creating your optimizer. This will remove all subdirectories in the `runs` folder.

## Related Parameters

- `tensorboard_log` (bool): Enable/disable logging (default: False)
- `tensorboard_path` (str): Custom log directory (default: auto-generated with timestamp)
- `tensorboard_clean` (bool): Remove old logs from 'runs' directory before starting (default: False)
- `verbose` (bool): Print progress to console (default: False)
- `var_name` (list): Custom names for variables (used in TensorBoard labels)

## Performance Notes

TensorBoard logging has minimal overhead:

- < 1% slowdown for typical optimizations
- Event files are efficiently buffered and written
- Writer is properly closed after optimization completes
