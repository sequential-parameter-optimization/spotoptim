"""
Demonstration of TensorBoard log cleaning functionality.

This script shows how to use the tensorboard_clean parameter to automatically
remove old TensorBoard logs before starting a new optimization run.
"""

import numpy as np
from spotoptim import SpotOptim
import os


def sphere(X):
    """Simple sphere function for testing."""
    return np.sum(X**2, axis=1)


def rosenbrock(X):
    """Rosenbrock function - a classic optimization benchmark."""
    x = X[:, 0]
    y = X[:, 1]
    return (1 - x)**2 + 100 * (y - x**2)**2


def count_log_dirs():
    """Count subdirectories in runs directory."""
    if not os.path.exists("runs"):
        return 0
    return len([d for d in os.listdir("runs") if os.path.isdir(os.path.join("runs", d))])


def main():
    print("=" * 70)
    print("TensorBoard Log Cleaning Demo")
    print("=" * 70)
    
    # Scenario 1: Create some old logs
    print("\n--- Scenario 1: Creating Old Logs ---")
    print("Creating 3 optimization runs with TensorBoard logging...")
    
    for i in range(3):
        opt = SpotOptim(
            fun=sphere,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=12,
            n_initial=8,
            tensorboard_log=True,
            seed=i,
            verbose=False,
        )
        result = opt.optimize()
        print(f"Run {i+1} completed: {opt.tensorboard_path}")
    
    old_count = count_log_dirs()
    print(f"\nTotal log directories in 'runs': {old_count}")
    
    # Scenario 2: New run without cleaning (preserves old logs)
    print("\n--- Scenario 2: New Run WITHOUT Cleaning ---")
    print("Running optimization with tensorboard_clean=False (default)...")
    
    opt = SpotOptim(
        fun=rosenbrock,
        bounds=[(-2, 2), (-2, 2)],
        max_iter=15,
        n_initial=10,
        tensorboard_log=True,
        tensorboard_clean=False,  # Default
        seed=42,
        verbose=True,
    )
    result = opt.optimize()
    
    new_count = count_log_dirs()
    print(f"\nTotal log directories in 'runs': {new_count}")
    print(f"Old logs preserved: {new_count > old_count}")
    
    # Scenario 3: New run with cleaning (removes old logs)
    print("\n--- Scenario 3: New Run WITH Cleaning ---")
    print("Running optimization with tensorboard_clean=True...")
    print("This will remove all old log directories!\n")
    
    opt = SpotOptim(
        fun=rosenbrock,
        bounds=[(-2, 2), (-2, 2)],
        max_iter=20,
        n_initial=12,
        tensorboard_log=True,
        tensorboard_clean=True,  # Enable cleaning
        seed=123,
        verbose=True,
    )
    result = opt.optimize()
    
    final_count = count_log_dirs()
    print(f"\nTotal log directories in 'runs': {final_count}")
    print(f"Old logs cleaned: {final_count == 1}")
    
    # Scenario 4: Cleaning without logging
    print("\n--- Scenario 4: Cleaning WITHOUT Logging ---")
    print("You can also use tensorboard_clean=True without logging enabled")
    print("to clean up old logs before starting your optimization...\n")
    
    # First, create a dummy log
    opt = SpotOptim(
        fun=sphere,
        bounds=[(-5, 5)],
        max_iter=10,
        n_initial=5,
        tensorboard_log=True,
        verbose=False,
    )
    result = opt.optimize()
    
    before_clean = count_log_dirs()
    print(f"Log directories before clean: {before_clean}")
    
    # Now clean without new logging
    opt = SpotOptim(
        fun=sphere,
        bounds=[(-5, 5)],
        max_iter=10,
        n_initial=5,
        tensorboard_log=False,  # No new logs
        tensorboard_clean=True,  # But clean old ones
        verbose=True,
    )
    
    after_clean = count_log_dirs()
    print(f"Log directories after clean: {after_clean}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\nUse cases for tensorboard_clean:")
    print("  1. Clean start: tensorboard_log=True, tensorboard_clean=True")
    print("     → Removes old logs, creates new log directory")
    print()
    print("  2. Preserve history: tensorboard_log=True, tensorboard_clean=False")
    print("     → Keeps old logs, adds new log directory")
    print()
    print("  3. Just clean: tensorboard_log=False, tensorboard_clean=True")
    print("     → Removes old logs, no new logging")
    print()
    print("  4. Default behavior: tensorboard_log=False, tensorboard_clean=False")
    print("     → No logging, no cleaning")
    print()
    print("\nWarning:")
    print("  - tensorboard_clean=True permanently deletes all subdirectories in 'runs'")
    print("  - Make sure to save important logs elsewhere before enabling!")
    print()
    print("\nView TensorBoard logs:")
    print("  $ tensorboard --logdir=runs")
    print("  Then open: http://localhost:6006")
    print("=" * 70)


if __name__ == "__main__":
    main()
