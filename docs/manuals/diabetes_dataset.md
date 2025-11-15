# Diabetes Dataset Utilities

SpotOptim provides convenient utilities for working with the sklearn diabetes dataset, including PyTorch `Dataset` and `DataLoader` implementations. These utilities simplify data loading, preprocessing, and model training for regression tasks.

## Overview

The diabetes dataset contains 10 baseline variables (age, sex, body mass index, average blood pressure, and six blood serum measurements) for 442 diabetes patients. The target is a quantitative measure of disease progression one year after baseline.

**Module**: `spotoptim.data.diabetes`

**Key Components**:

- `DiabetesDataset`: PyTorch Dataset class
- `get_diabetes_dataloaders()`: Convenience function for complete data pipeline

## Quick Start

### Basic Usage

```python
from spotoptim.data import get_diabetes_dataloaders

# Load data with default settings
train_loader, test_loader, scaler = get_diabetes_dataloaders()

# Iterate through batches
for batch_X, batch_y in train_loader:
    print(f"Batch features: {batch_X.shape}")  # (32, 10)
    print(f"Batch targets: {batch_y.shape}")   # (32, 1)
    break
```

### Training a Model

```python
import torch
import torch.nn as nn
from spotoptim.data import get_diabetes_dataloaders
from spotoptim.nn.linear_regressor import LinearRegressor

# Load data
train_loader, test_loader, scaler = get_diabetes_dataloaders(
    test_size=0.2,
    batch_size=32,
    scale_features=True,
    random_state=42
)

# Create model
model = LinearRegressor(
    input_dim=10,
    output_dim=1,
    l1=64,
    num_hidden_layers=2,
    activation="ReLU"
)

# Setup training
criterion = nn.MSELoss()
optimizer = model.get_optimizer("Adam", lr=0.01)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        # Forward pass
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_train_loss:.4f}")

# Evaluation
model.eval()
test_loss = 0.0

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        test_loss += loss.item()

avg_test_loss = test_loss / len(test_loader)
print(f"Test MSE: {avg_test_loss:.4f}")
```

## Function Reference

### get_diabetes_dataloaders()

Loads the sklearn diabetes dataset and returns configured PyTorch DataLoaders.

**Signature:**
```python
get_diabetes_dataloaders(
    test_size=0.2,
    batch_size=32,
    shuffle_train=True,
    shuffle_test=False,
    random_state=42,
    scale_features=True,
    num_workers=0,
    pin_memory=False
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `test_size` | float | 0.2 | Proportion of dataset for testing (0.0 to 1.0) |
| `batch_size` | int | 32 | Number of samples per batch |
| `shuffle_train` | bool | True | Whether to shuffle training data |
| `shuffle_test` | bool | False | Whether to shuffle test data |
| `random_state` | int | 42 | Random seed for train/test split |
| `scale_features` | bool | True | Whether to standardize features |
| `num_workers` | int | 0 | Number of subprocesses for data loading |
| `pin_memory` | bool | False | Whether to pin memory (useful for GPU) |

**Returns:**

- `train_loader` (DataLoader): Training data loader
- `test_loader` (DataLoader): Test data loader
- `scaler` (StandardScaler or None): Fitted scaler if `scale_features=True`, else None

**Example:**
```python
from spotoptim.data import get_diabetes_dataloaders

# Custom configuration
train_loader, test_loader, scaler = get_diabetes_dataloaders(
    test_size=0.3,
    batch_size=64,
    shuffle_train=True,
    scale_features=True,
    random_state=123
)

print(f"Training batches: {len(train_loader)}")
print(f"Test batches: {len(test_loader)}")
print(f"Scaler mean: {scaler.mean_[:3]}")  # First 3 features
```

## DiabetesDataset Class

PyTorch Dataset implementation for the diabetes dataset.

**Signature:**
```python
DiabetesDataset(X, y, transform=None, target_transform=None)
```

**Parameters:**

- `X` (np.ndarray): Feature matrix of shape (n_samples, n_features)
- `y` (np.ndarray): Target values of shape (n_samples,) or (n_samples, 1)
- `transform` (callable, optional): Transform to apply to features
- `target_transform` (callable, optional): Transform to apply to targets

**Attributes:**

- `X` (torch.Tensor): Feature tensor (n_samples, n_features)
- `y` (torch.Tensor): Target tensor (n_samples, 1)
- `n_features` (int): Number of features (10 for diabetes)
- `n_samples` (int): Number of samples

**Methods:**

- `__len__()`: Returns number of samples
- `__getitem__(idx)`: Returns tuple (features, target) for given index

### Manual Dataset Creation

```python
from spotoptim.data import DiabetesDataset
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

# Load raw data
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create datasets
train_dataset = DiabetesDataset(X_train, y_train)
test_dataset = DiabetesDataset(X_test, y_test)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Inspect dataset
print(f"Dataset size: {len(train_dataset)}")
print(f"Features shape: {train_dataset.X.shape}")
print(f"Targets shape: {train_dataset.y.shape}")

# Get a sample
features, target = train_dataset[0]
print(f"Sample features: {features.shape}")  # (10,)
print(f"Sample target: {target.shape}")      # (1,)
```

## Advanced Usage

### Custom Transforms

```python
from spotoptim.data import DiabetesDataset
from sklearn.datasets import load_diabetes
import torch

# Define custom transforms
def add_noise(x):
    """Add Gaussian noise to features."""
    return x + torch.randn_like(x) * 0.01

def log_transform(y):
    """Apply log transform to target."""
    return torch.log1p(y)

# Load data
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# Create dataset with transforms
dataset = DiabetesDataset(
    X, y,
    transform=add_noise,
    target_transform=log_transform
)

# Transforms are applied when accessing items
features, target = dataset[0]
```

### Different Train/Test Splits

```python
from spotoptim.data import get_diabetes_dataloaders

# 70/30 split
train_loader, test_loader, scaler = get_diabetes_dataloaders(
    test_size=0.3,
    random_state=42
)
print(f"Training samples: {len(train_loader.dataset)}")  # ~310
print(f"Test samples: {len(test_loader.dataset)}")       # ~132

# 90/10 split
train_loader, test_loader, scaler = get_diabetes_dataloaders(
    test_size=0.1,
    random_state=42
)
print(f"Training samples: {len(train_loader.dataset)}")  # ~398
print(f"Test samples: {len(test_loader.dataset)}")       # ~44
```

### Without Feature Scaling

```python
from spotoptim.data import get_diabetes_dataloaders

# Load without scaling (useful for tree-based models)
train_loader, test_loader, scaler = get_diabetes_dataloaders(
    scale_features=False
)

print(f"Scaler: {scaler}")  # None

# Data is in original scale
for batch_X, batch_y in train_loader:
    print(f"Mean: {batch_X.mean(dim=0)[:3]}")  # Non-zero values
    break
```

### Larger Batch Sizes

```python
from spotoptim.data import get_diabetes_dataloaders

# Larger batches for faster training (if memory allows)
train_loader, test_loader, scaler = get_diabetes_dataloaders(
    batch_size=128
)
print(f"Batches per epoch: {len(train_loader)}")  # Fewer batches

# Smaller batches for more gradient updates
train_loader, test_loader, scaler = get_diabetes_dataloaders(
    batch_size=8
)
print(f"Batches per epoch: {len(train_loader)}")  # More batches
```

### GPU Training with Pin Memory

```python
import torch
from spotoptim.data import get_diabetes_dataloaders

# Enable pin_memory for faster GPU transfer
train_loader, test_loader, scaler = get_diabetes_dataloaders(
    batch_size=32,
    pin_memory=True  # Set to True when using GPU
)

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training loop with GPU
for batch_X, batch_y in train_loader:
    # Data is already pinned, faster transfer to GPU
    batch_X = batch_X.to(device, non_blocking=True)
    batch_y = batch_y.to(device, non_blocking=True)
    
    # ... training code ...
```

## Complete Training Example

Here's a complete example showing data loading, model training, and evaluation:

```python
import torch
import torch.nn as nn
from spotoptim.data import get_diabetes_dataloaders
from spotoptim.nn.linear_regressor import LinearRegressor

def train_diabetes_model():
    """Train a neural network on the diabetes dataset."""
    
    # Load data
    train_loader, test_loader, scaler = get_diabetes_dataloaders(
        test_size=0.2,
        batch_size=32,
        scale_features=True,
        random_state=42
    )
    
    # Create model
    model = LinearRegressor(
        input_dim=10,
        output_dim=1,
        l1=128,
        num_hidden_layers=3,
        activation="ReLU"
    )
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = model.get_optimizer("Adam", lr=0.001, weight_decay=1e-5)
    
    # Training configuration
    num_epochs = 200
    best_test_loss = float('inf')
    
    print("Starting training...")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Batches per epoch: {len(train_loader)}")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            # Forward pass
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Evaluation phase
        model.eval()
        test_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                test_loss += loss.item()
        
        avg_test_loss = test_loss / len(test_loader)
        
        # Track best model
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            # Could save model here: torch.save(model.state_dict(), 'best_model.pt')
        
        # Print progress
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs}: "
                  f"Train Loss = {avg_train_loss:.4f}, "
                  f"Test Loss = {avg_test_loss:.4f}")
    
    print("-" * 60)
    print(f"Training complete!")
    print(f"Best test loss: {best_test_loss:.4f}")
    
    return model, best_test_loss

# Run training
if __name__ == "__main__":
    model, best_loss = train_diabetes_model()
```

## Integration with SpotOptim

Use the diabetes dataset for hyperparameter optimization with SpotOptim:

```python
import numpy as np
import torch
import torch.nn as nn
from spotoptim import SpotOptim
from spotoptim.data import get_diabetes_dataloaders
from spotoptim.nn.linear_regressor import LinearRegressor

def evaluate_model(X):
    """Objective function for SpotOptim.
    
    Args:
        X: Array of hyperparameters [lr, l1, num_hidden_layers]
        
    Returns:
        Array of validation losses
    """
    results = []
    
    for params in X:
        lr, l1, num_hidden_layers = params
        lr = 10 ** lr  # Log scale for learning rate
        l1 = int(l1)
        num_hidden_layers = int(num_hidden_layers)
        
        # Load data
        train_loader, test_loader, _ = get_diabetes_dataloaders(
            test_size=0.2,
            batch_size=32,
            random_state=42
        )
        
        # Create model
        model = LinearRegressor(
            input_dim=10,
            output_dim=1,
            l1=l1,
            num_hidden_layers=num_hidden_layers,
            activation="ReLU"
        )
        
        # Train briefly
        criterion = nn.MSELoss()
        optimizer = model.get_optimizer("Adam", lr=lr)
        
        num_epochs = 50
        for epoch in range(num_epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Evaluate
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                test_loss += loss.item()
        
        results.append(test_loss / len(test_loader))
    
    return np.array(results)

# Optimize hyperparameters
optimizer = SpotOptim(
    fun=evaluate_model,
    bounds=[
        (-4, -2),   # log10(lr): 0.0001 to 0.01
        (16, 128),  # l1: number of neurons
        (0, 4)      # num_hidden_layers
    ],
    var_type=["num", "int", "int"],
    max_iter=30,
    n_initial=10,
    seed=42,
    verbose=True
)

result = optimizer.optimize()
print(f"Best hyperparameters found:")
print(f"  Learning rate: {10**result.x[0]:.6f}")
print(f"  Hidden neurons (l1): {int(result.x[1])}")
print(f"  Hidden layers: {int(result.x[2])}")
print(f"  Best MSE: {result.fun:.4f}")
```

## Best Practices

### 1. Always Use Feature Scaling

```python
# Good: Features are standardized
train_loader, test_loader, scaler = get_diabetes_dataloaders(
    scale_features=True
)
```

Neural networks typically perform better with normalized inputs.

### 2. Set Random Seeds for Reproducibility

```python
# Reproducible train/test splits
train_loader, test_loader, scaler = get_diabetes_dataloaders(
    random_state=42
)

# Also set PyTorch seed
import torch
torch.manual_seed(42)
```

### 3. Don't Shuffle Test Data

```python
# Good: Test data in consistent order
train_loader, test_loader, scaler = get_diabetes_dataloaders(
    shuffle_train=True,   # Shuffle training data
    shuffle_test=False    # Don't shuffle test data
)
```

This ensures consistent evaluation metrics across runs.

### 4. Choose Appropriate Batch Size

```python
# Small dataset (442 samples) - moderate batch size works well
train_loader, test_loader, scaler = get_diabetes_dataloaders(
    batch_size=32  # Good balance for this dataset
)
```

Too large: Fewer gradient updates per epoch  
Too small: Noisy gradients, slower training

### 5. Save the Scaler for Production

```python
import pickle
from spotoptim.data import get_diabetes_dataloaders

# Train with scaling
train_loader, test_loader, scaler = get_diabetes_dataloaders(
    scale_features=True
)

# Save scaler for production use
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Later: Load and use on new data
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

new_data_scaled = scaler.transform(new_data)
```

## Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce batch size or disable pin_memory

```python
train_loader, test_loader, scaler = get_diabetes_dataloaders(
    batch_size=16,      # Smaller batches
    pin_memory=False    # Disable if not using GPU
)
```

### Issue: Different Data Ranges

**Symptom**: Model not converging, loss is NaN

**Solution**: Ensure feature scaling is enabled

```python
train_loader, test_loader, scaler = get_diabetes_dataloaders(
    scale_features=True  # Must be True for neural networks
)
```

### Issue: Non-Reproducible Results

**Solution**: Set all random seeds

```python
import torch
import numpy as np

# Set all seeds
torch.manual_seed(42)
np.random.seed(42)

train_loader, test_loader, scaler = get_diabetes_dataloaders(
    random_state=42,
    shuffle_train=False  # Disable shuffle for full reproducibility
)
```

### Issue: Slow Data Loading

**Solution**: Use multiple workers (if not on Windows)

```python
train_loader, test_loader, scaler = get_diabetes_dataloaders(
    num_workers=4,      # Use 4 subprocesses
    pin_memory=True     # Enable for GPU
)
```

Note: On Windows, set `num_workers=0` to avoid multiprocessing issues.

## Summary

The diabetes dataset utilities in SpotOptim provide:

- **Easy data loading**: One function call gets complete data pipeline
- **PyTorch integration**: Native Dataset and DataLoader support
- **Preprocessing included**: Automatic feature scaling and train/test splitting
- **Flexible configuration**: Control batch size, splitting, scaling, and more
- **Production ready**: Save scalers and ensure reproducibility

For more examples, see:
- `examples/diabetes_dataset_example.py`
- `notebooks/demos.ipynb`
- Test suite: `tests/test_diabetes_dataset.py`
