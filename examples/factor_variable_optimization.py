"""
Example: Using Factor Variables for Neural Network Hyperparameter Optimization
===============================================================================

This example demonstrates how to use SpotOptim with factor variables to optimize
neural network hyperparameters including activation functions as categorical choices.
"""

import numpy as np
import torch
import torch.nn as nn
from spotoptim import SpotOptim
from spotoptim.data import get_diabetes_dataloaders
from spotoptim.nn.linear_regressor import LinearRegressor


def train_and_evaluate_model(X):
    """Train a LinearRegressor with given hyperparameters.
    
    Args:
        X: Array where each row contains:
           - log10(learning_rate): float in [-4, -2]
           - l1 (neurons per layer): int in [16, 128]  
           - num_hidden_layers: int in [0, 4]
           - activation: string from {"ReLU", "Sigmoid", "Tanh", "LeakyReLU"}
    
    Returns:
        Array of validation MSE losses for each configuration.
    """
    results = []
    
    for params in X:
        lr = 10 ** params[0]  # Convert from log scale
        l1 = int(params[1])
        num_layers = int(params[2])
        activation = params[3]  # This will be a string!
        
        print(f"  Testing: lr={lr:.6f}, l1={l1}, layers={num_layers}, activation={activation}")
        
        # Load data
        train_loader, test_loader, _ = get_diabetes_dataloaders(
            test_size=0.2,
            batch_size=32,
            random_state=42
        )
        
        # Create model with specified hyperparameters
        model = LinearRegressor(
            input_dim=10,
            output_dim=1,
            l1=l1,
            num_hidden_layers=num_layers,
            activation=activation  # Pass the string directly!
        )
        
        # Setup training
        optimizer = model.get_optimizer("Adam", lr=lr)
        criterion = nn.MSELoss()
        
        # Train for a few epochs
        num_epochs = 30
        for epoch in range(num_epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                test_loss += loss.item()
        
        avg_test_loss = test_loss / len(test_loader)
        results.append(avg_test_loss)
        print(f"    → Test MSE: {avg_test_loss:.4f}")
    
    return np.array(results)


def main():
    print("=" * 80)
    print("Neural Network Hyperparameter Optimization with Factor Variables")
    print("=" * 80)
    print()
    print("Optimizing 4 hyperparameters:")
    print("  1. Learning rate (continuous, log scale)")
    print("  2. Hidden layer size (integer)")
    print("  3. Number of hidden layers (integer)")
    print("  4. Activation function (categorical factor variable)")
    print()
    print("=" * 80)
    print()
    
    # Create optimizer with factor variable for activation function
    optimizer = SpotOptim(
        fun=train_and_evaluate_model,
        bounds=[
            (-4, -2),                                    # log10(lr): 0.0001 to 0.01
            (16, 128),                                   # l1: number of neurons
            (0, 4),                                      # num_hidden_layers: 0 to 4
            ("ReLU", "Sigmoid", "Tanh", "LeakyReLU"),  # activation (factor!)
        ],
        var_type=["num", "int", "int", "factor"],
        var_name=["log_lr", "l1", "num_layers", "activation"],
        max_iter=25,          # Total evaluations
        n_initial=12,         # Initial random evaluations
        seed=42,
        verbose=True,
    )
    
    print("Starting optimization...")
    print()
    result = optimizer.optimize()
    
    print()
    print("=" * 80)
    print("Optimization Complete!")
    print("=" * 80)
    print()
    print("Best hyperparameters found:")
    print(f"  Learning rate: {10**result.x[0]:.6f}")
    print(f"  Hidden layer size (l1): {result.x[1]}")
    print(f"  Number of hidden layers: {result.x[2]}")
    print(f"  Activation function: {result.x[3]}")  # String result!
    print()
    print(f"Best MSE: {result.fun:.4f}")
    print(f"Total evaluations: {result.nfev}")
    print()
    
    # Show a few best configurations
    print("Top 5 configurations:")
    print("-" * 80)
    sorted_indices = np.argsort(result.y)[:5]
    for i, idx in enumerate(sorted_indices, 1):
        config = result.X[idx]
        mse = result.y[idx]
        print(f"{i}. lr={10**config[0]:.6f}, l1={config[1]}, "
              f"layers={config[2]}, activation={config[3]} → MSE={mse:.4f}")
    print()
    
    # Create and save the best model
    print("Training final model with best hyperparameters...")
    train_loader, test_loader, _ = get_diabetes_dataloaders(
        test_size=0.2,
        batch_size=32,
        random_state=42
    )
    
    best_model = LinearRegressor(
        input_dim=10,
        output_dim=1,
        l1=int(result.x[1]),
        num_hidden_layers=int(result.x[2]),
        activation=result.x[3]
    )
    
    optimizer_model = best_model.get_optimizer("Adam", lr=10**result.x[0])
    criterion = nn.MSELoss()
    
    # Train for more epochs
    print("Training for 100 epochs...")
    for epoch in range(100):
        best_model.train()
        for batch_X, batch_y in train_loader:
            predictions = best_model(batch_X)
            loss = criterion(predictions, batch_y)
            optimizer_model.zero_grad()
            loss.backward()
            optimizer_model.step()
        
        if (epoch + 1) % 20 == 0:
            best_model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    predictions = best_model(batch_X)
                    loss = criterion(predictions, batch_y)
                    test_loss += loss.item()
            print(f"  Epoch {epoch+1}/100: Test MSE = {test_loss/len(test_loader):.4f}")
    
    print()
    print("Done! Model trained with optimized hyperparameters.")
    print("=" * 80)


if __name__ == "__main__":
    main()
