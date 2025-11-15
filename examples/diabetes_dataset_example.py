"""
Example: Using the Diabetes Dataset with SpotOptim
===================================================

This example shows how to use the built-in diabetes dataset
utilities for training PyTorch models.
"""

import torch
import torch.nn as nn
from spotoptim.data import get_diabetes_dataloaders
from spotoptim.nn.linear_regressor import LinearRegressor


def main():
    print("=" * 60)
    print("Diabetes Dataset Example")
    print("=" * 60)
    
    # Load diabetes dataset with DataLoaders
    print("\n1. Loading diabetes dataset...")
    train_loader, test_loader, scaler = get_diabetes_dataloaders(
        test_size=0.2,
        batch_size=32,
        scale_features=True,
        random_state=42
    )
    
    print(f"   ✓ Training batches: {len(train_loader)}")
    print(f"   ✓ Test batches: {len(test_loader)}")
    print(f"   ✓ Features scaled: {scaler is not None}")
    
    # Create model
    print("\n2. Creating LinearRegressor model...")
    model = LinearRegressor(
        input_dim=10,
        output_dim=1,
        l1=64,
        num_hidden_layers=2,
        activation="ReLU"
    )
    print(f"   ✓ Model architecture: 10 → 64 → 64 → 1")
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = model.get_optimizer("Adam", lr=0.01)
    
    # Training loop
    print("\n3. Training model...")
    num_epochs = 50
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
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1:3d}/{num_epochs}: Loss = {avg_train_loss:.4f}")
    
    # Evaluation
    print("\n4. Evaluating model...")
    model.eval()
    test_loss = 0.0
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            test_loss += loss.item()
    
    avg_test_loss = test_loss / len(test_loader)
    print(f"   ✓ Test MSE: {avg_test_loss:.4f}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
