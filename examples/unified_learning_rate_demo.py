"""
Example: Using unified learning rate interface with LinearRegressor

This example demonstrates the new `lr` parameter in LinearRegressor that provides
a unified interface for learning rates across different PyTorch optimizers.
"""

import torch
import torch.nn as nn
from spotoptim.nn.linear_regressor import LinearRegressor
from spotoptim.utils.mapping import map_lr

# Generate synthetic data
torch.manual_seed(42)
X_train = torch.randn(100, 10)
y_train = torch.randn(100, 1)

print("=" * 80)
print("Example 1: Unified Learning Rate Interface")
print("=" * 80)

# With unified lr=1.0, each optimizer gets its PyTorch default
model = LinearRegressor(input_dim=10, output_dim=1, l1=16, num_hidden_layers=1, lr=1.0)

optimizers_to_test = ["Adam", "SGD", "RMSprop", "AdamW"]
for opt_name in optimizers_to_test:
    optimizer = model.get_optimizer(opt_name)
    actual_lr = optimizer.param_groups[0]["lr"]
    print(f"{opt_name:10s}: unified lr=1.0 → actual lr={actual_lr}")

print("\n" + "=" * 80)
print("Example 2: Custom Unified Learning Rate")
print("=" * 80)

# Using lr=0.5 scales all optimizers by 0.5
model = LinearRegressor(input_dim=10, output_dim=1, l1=16, num_hidden_layers=1, lr=0.5)

for opt_name in optimizers_to_test:
    optimizer = model.get_optimizer(opt_name)
    actual_lr = optimizer.param_groups[0]["lr"]
    print(f"{opt_name:10s}: unified lr=0.5 → actual lr={actual_lr}")

print("\n" + "=" * 80)
print("Example 3: Training with Different Optimizers")
print("=" * 80)

criterion = nn.MSELoss()
results = {}

for opt_name in optimizers_to_test:
    # Reset model for fair comparison
    torch.manual_seed(42)
    model = LinearRegressor(input_dim=10, output_dim=1, l1=16, 
                           num_hidden_layers=1, lr=1.0)
    
    # Create optimizer with unified lr
    if opt_name == "SGD":
        optimizer = model.get_optimizer(opt_name, momentum=0.9)
    else:
        optimizer = model.get_optimizer(opt_name)
    
    # Train
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        final_loss = criterion(model(X_train), y_train).item()
    
    results[opt_name] = final_loss
    print(f"{opt_name:10s}: final loss = {final_loss:.4f}")

print("\n" + "=" * 80)
print("Example 4: Direct use of map_lr() function")
print("=" * 80)

# The map_lr function can be used independently
lr_unified = 2.0
print(f"Unified lr: {lr_unified}")
print()

for opt_name in optimizers_to_test:
    actual_lr = map_lr(lr_unified, opt_name)
    print(f"{opt_name:10s}: {lr_unified} * default → {actual_lr}")

print("\n" + "=" * 80)
print("Example 5: Log-scale Hyperparameter Optimization")
print("=" * 80)

# Common pattern: sample from log10 scale, then map
import numpy as np

print("Log-scale range: [-4, 0] → unified lr range: [0.0001, 1.0]")
print()

for log_lr in [-4, -3, -2, -1, 0]:
    lr_unified = 10 ** log_lr
    lr_adam = map_lr(lr_unified, "Adam")
    lr_sgd = map_lr(lr_unified, "SGD")
    print(f"log_lr={log_lr:2d} → unified={lr_unified:.4f} → "
          f"Adam={lr_adam:.6f}, SGD={lr_sgd:.6f}")

print("\n" + "=" * 80)
print("Example 6: Hyperparameter Optimization Scenario")
print("=" * 80)

# Simulate hyperparameter optimization
def evaluate_config(lr_unified, optimizer_name):
    """Train model with given config and return validation loss."""
    torch.manual_seed(42)
    model = LinearRegressor(input_dim=10, output_dim=1, l1=16, 
                           num_hidden_layers=1, lr=lr_unified)
    
    if optimizer_name == "SGD":
        optimizer = model.get_optimizer(optimizer_name, momentum=0.9)
    else:
        optimizer = model.get_optimizer(optimizer_name)
    
    criterion = nn.MSELoss()
    
    # Train
    model.train()
    for _ in range(50):
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_train), y_train).item()
    
    return val_loss

# Test different configurations
configs = [
    (0.5, "Adam"),
    (1.0, "Adam"),
    (2.0, "Adam"),
    (0.5, "SGD"),
    (1.0, "SGD"),
    (2.0, "SGD"),
]

print("Testing different (unified_lr, optimizer) configurations:")
print()
print(f"{'Unified LR':<12} {'Optimizer':<10} {'Actual LR':<12} {'Val Loss':<10}")
print("-" * 50)

best_config = None
best_loss = float("inf")

for lr_unified, opt_name in configs:
    actual_lr = map_lr(lr_unified, opt_name)
    val_loss = evaluate_config(lr_unified, opt_name)
    
    print(f"{lr_unified:<12.2f} {opt_name:<10} {actual_lr:<12.6f} {val_loss:<10.4f}")
    
    if val_loss < best_loss:
        best_loss = val_loss
        best_config = (lr_unified, opt_name)

print()
print(f"Best configuration: lr={best_config[0]}, optimizer={best_config[1]}, "
      f"loss={best_loss:.4f}")

print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print("""
The unified learning rate interface provides:

1. ✓ Fair comparison across different optimizers
2. ✓ Single learning rate parameter to optimize
3. ✓ Automatic scaling based on optimizer defaults
4. ✓ Simplified hyperparameter tuning
5. ✓ Compatible with existing code (backward compatible)

Key points:
- lr=1.0 always gives the optimizer's PyTorch default
- lr scales linearly with actual learning rate
- Works with log-scale hyperparameter optimization
- Can be overridden in get_optimizer() if needed
""")
