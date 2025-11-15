"""
Tests for diabetes dataset utilities.
"""

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader
from spotoptim.data.diabetes import DiabetesDataset, get_diabetes_dataloaders
from sklearn.datasets import load_diabetes


class TestDiabetesDataset:
    """Test suite for DiabetesDataset class."""

    def test_dataset_initialization(self):
        """Test creating a DiabetesDataset."""
        diabetes = load_diabetes()
        X, y = diabetes.data, diabetes.target
        
        dataset = DiabetesDataset(X, y)
        
        assert len(dataset) == len(X)
        assert dataset.n_features == 10
        assert dataset.n_samples == len(X)

    def test_dataset_getitem(self):
        """Test getting items from the dataset."""
        diabetes = load_diabetes()
        X, y = diabetes.data[:10], diabetes.target[:10]
        
        dataset = DiabetesDataset(X, y)
        
        features, target = dataset[0]
        
        assert isinstance(features, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert features.shape == (10,)
        assert target.shape == (1,)

    def test_dataset_with_1d_targets(self):
        """Test that 1D targets are automatically reshaped to 2D."""
        X = np.random.randn(100, 10)
        y = np.random.randn(100)  # 1D
        
        dataset = DiabetesDataset(X, y)
        
        assert dataset.y.shape == (100, 1)
        
        features, target = dataset[0]
        assert target.shape == (1,)

    def test_dataset_with_2d_targets(self):
        """Test that 2D targets are preserved."""
        X = np.random.randn(100, 10)
        y = np.random.randn(100, 1)  # Already 2D
        
        dataset = DiabetesDataset(X, y)
        
        assert dataset.y.shape == (100, 1)

    def test_dataset_tensor_conversion(self):
        """Test that numpy arrays are converted to tensors."""
        X = np.random.randn(50, 10).astype(np.float32)
        y = np.random.randn(50, 1).astype(np.float32)
        
        dataset = DiabetesDataset(X, y)
        
        assert isinstance(dataset.X, torch.Tensor)
        assert isinstance(dataset.y, torch.Tensor)
        assert dataset.X.dtype == torch.float32
        assert dataset.y.dtype == torch.float32

    def test_dataset_length(self):
        """Test __len__ method."""
        X = np.random.randn(75, 10)
        y = np.random.randn(75)
        
        dataset = DiabetesDataset(X, y)
        
        assert len(dataset) == 75

    def test_dataset_indexing(self):
        """Test that indexing returns correct samples."""
        X = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])
        y = np.array([100, 200])
        
        dataset = DiabetesDataset(X, y)
        
        features, target = dataset[1]
        
        assert torch.allclose(features, torch.FloatTensor([11, 12, 13, 14, 15, 16, 17, 18, 19, 20]))
        assert torch.allclose(target, torch.FloatTensor([200]))

    def test_dataset_with_transforms(self):
        """Test dataset with custom transforms."""
        X = np.random.randn(10, 10)
        y = np.random.randn(10)
        
        # Simple transform that multiplies by 2
        transform = lambda x: x * 2
        target_transform = lambda x: x * 3
        
        dataset = DiabetesDataset(X, y, transform=transform, target_transform=target_transform)
        
        features, target = dataset[0]
        
        # Check transforms were applied
        expected_features = torch.FloatTensor(X[0]) * 2
        expected_target = torch.FloatTensor(y[0:1]) * 3
        
        assert torch.allclose(features, expected_features)
        assert torch.allclose(target, expected_target)


class TestGetDiabetesDataloaders:
    """Test suite for get_diabetes_dataloaders function."""

    def test_default_dataloaders(self):
        """Test creating dataloaders with default parameters."""
        train_loader, test_loader, scaler = get_diabetes_dataloaders()
        
        assert isinstance(train_loader, DataLoader)
        assert isinstance(test_loader, DataLoader)
        assert scaler is not None

    def test_dataloader_sizes(self):
        """Test that train/test split is correct."""
        train_loader, test_loader, scaler = get_diabetes_dataloaders(
            test_size=0.2,
            batch_size=32
        )
        
        # Diabetes dataset has 442 samples
        # 80% train = 353, 20% test = 89
        train_samples = sum(len(batch[1]) for batch in train_loader)
        test_samples = sum(len(batch[1]) for batch in test_loader)
        
        assert train_samples == 353
        assert test_samples == 89

    def test_custom_test_size(self):
        """Test with custom test_size."""
        train_loader, test_loader, scaler = get_diabetes_dataloaders(
            test_size=0.3,
            batch_size=10
        )
        
        train_samples = sum(len(batch[1]) for batch in train_loader)
        test_samples = sum(len(batch[1]) for batch in test_loader)
        
        # 70% train, 30% test of 442 samples
        assert train_samples == 309
        assert test_samples == 133

    def test_custom_batch_size(self):
        """Test with custom batch size."""
        train_loader, test_loader, scaler = get_diabetes_dataloaders(
            batch_size=64
        )
        
        # Get first batch
        batch_X, batch_y = next(iter(train_loader))
        
        # Last batch might be smaller, but most should be 64
        assert batch_X.shape[0] <= 64
        assert batch_X.shape[1] == 10  # 10 features

    def test_batch_shapes(self):
        """Test that batches have correct shapes."""
        train_loader, test_loader, scaler = get_diabetes_dataloaders(
            batch_size=32
        )
        
        for batch_X, batch_y in train_loader:
            assert batch_X.ndim == 2
            assert batch_y.ndim == 2
            assert batch_X.shape[1] == 10
            assert batch_y.shape[1] == 1
            break

    def test_scaling_applied(self):
        """Test that features are scaled when scale_features=True."""
        train_loader, test_loader, scaler = get_diabetes_dataloaders(
            scale_features=True
        )
        
        # Get all training data
        all_features = []
        for batch_X, _ in train_loader:
            all_features.append(batch_X)
        all_features = torch.cat(all_features, dim=0)
        
        # Check that features are approximately standardized
        mean = all_features.mean(dim=0)
        std = all_features.std(dim=0)
        
        # Mean should be close to 0, std close to 1
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-6)
        assert torch.allclose(std, torch.ones_like(std), atol=0.1)

    def test_no_scaling(self):
        """Test that scaler is None when scale_features=False."""
        train_loader, test_loader, scaler = get_diabetes_dataloaders(
            scale_features=False
        )
        
        assert scaler is None

    def test_shuffle_train(self):
        """Test that shuffle_train parameter works."""
        # Get dataloaders with and without shuffle
        train_loader_shuffle, _, _ = get_diabetes_dataloaders(
            shuffle_train=True,
            random_state=42,
            batch_size=10
        )
        train_loader_no_shuffle, _, _ = get_diabetes_dataloaders(
            shuffle_train=False,
            random_state=42,
            batch_size=10
        )
        
        # Collect all batches from both loaders
        batches_shuffle = []
        batches_no_shuffle = []
        
        for batch_X, _ in train_loader_shuffle:
            batches_shuffle.append(batch_X)
        
        for batch_X, _ in train_loader_no_shuffle:
            batches_no_shuffle.append(batch_X)
        
        # The datasets should have the same samples (after concatenating all batches)
        all_shuffle = torch.cat(batches_shuffle)
        all_no_shuffle = torch.cat(batches_no_shuffle)
        
        # Sort both to compare regardless of order
        sorted_shuffle, _ = torch.sort(all_shuffle.view(-1))
        sorted_no_shuffle, _ = torch.sort(all_no_shuffle.view(-1))
        
        assert torch.allclose(sorted_shuffle, sorted_no_shuffle)

    def test_reproducibility(self):
        """Test that random_state ensures reproducibility."""
        train_loader1, test_loader1, _ = get_diabetes_dataloaders(
            random_state=42
        )
        train_loader2, test_loader2, _ = get_diabetes_dataloaders(
            random_state=42
        )
        
        # Get first batch from each train loader
        batch1 = next(iter(train_loader1))
        batch2 = next(iter(train_loader2))
        
        # Should be identical when shuffle is disabled
        # Note: DataLoader shuffle is not controlled by random_state,
        # so we need shuffle_train=False for deterministic batches
        train_loader1_no_shuffle, _, _ = get_diabetes_dataloaders(
            random_state=42,
            shuffle_train=False
        )
        train_loader2_no_shuffle, _, _ = get_diabetes_dataloaders(
            random_state=42,
            shuffle_train=False
        )
        
        batch1_no_shuffle = next(iter(train_loader1_no_shuffle))
        batch2_no_shuffle = next(iter(train_loader2_no_shuffle))
        
        assert torch.allclose(batch1_no_shuffle[0], batch2_no_shuffle[0])
        assert torch.allclose(batch1_no_shuffle[1], batch2_no_shuffle[1])

    def test_different_random_states(self):
        """Test that different random_state gives different splits."""
        train_loader1, _, _ = get_diabetes_dataloaders(random_state=42)
        train_loader2, _, _ = get_diabetes_dataloaders(random_state=123)
        
        batch1 = next(iter(train_loader1))
        batch2 = next(iter(train_loader2))
        
        # Should be different (with very high probability)
        assert not torch.allclose(batch1[0], batch2[0])

    def test_dataloader_iteration(self):
        """Test that we can iterate through dataloaders."""
        train_loader, test_loader, _ = get_diabetes_dataloaders(
            batch_size=16
        )
        
        batch_count = 0
        for batch_X, batch_y in train_loader:
            batch_count += 1
            assert isinstance(batch_X, torch.Tensor)
            assert isinstance(batch_y, torch.Tensor)
        
        assert batch_count > 0

    def test_tensor_types(self):
        """Test that tensors are float32."""
        train_loader, test_loader, _ = get_diabetes_dataloaders()
        
        batch_X, batch_y = next(iter(train_loader))
        
        assert batch_X.dtype == torch.float32
        assert batch_y.dtype == torch.float32

    def test_no_data_leakage(self):
        """Test that scaler is only fitted on training data."""
        from sklearn.datasets import load_diabetes
        from sklearn.model_selection import train_test_split
        
        # Load data and split manually to get unscaled training data
        diabetes = load_diabetes()
        X_train, X_test, y_train, y_test = train_test_split(
            diabetes.data, diabetes.target, test_size=0.2, random_state=42
        )
        
        # Get dataloaders with scaling
        train_loader, test_loader, scaler = get_diabetes_dataloaders(
            scale_features=True,
            random_state=42
        )
        
        # Get statistics from scaler
        scaler_mean = scaler.mean_
        
        # Scaler mean should match unscaled training data mean
        unscaled_train_mean = X_train.mean(axis=0)
        assert np.allclose(scaler_mean, unscaled_train_mean, atol=1e-6)
        
        # Get scaled training data from dataloader
        all_train_features = []
        for batch_X, _ in train_loader:
            all_train_features.append(batch_X)
        all_train_features = torch.cat(all_train_features, dim=0).numpy()
        
        # Scaled training data should have mean close to 0
        scaled_train_mean = all_train_features.mean(axis=0)
        assert np.allclose(scaled_train_mean, 0, atol=1e-6)


class TestIntegration:
    """Integration tests combining dataset with model training."""

    def test_train_model_with_dataloaders(self):
        """Test that dataloaders work with actual model training."""
        from spotoptim.nn.linear_regressor import LinearRegressor
        import torch.nn as nn
        
        # Get dataloaders
        train_loader, test_loader, _ = get_diabetes_dataloaders(
            batch_size=32,
            random_state=42
        )
        
        # Create model
        model = LinearRegressor(input_dim=10, output_dim=1, l1=16, num_hidden_layers=1)
        optimizer = model.get_optimizer("Adam", lr=0.01)
        criterion = nn.MSELoss()
        
        # Train for a few epochs
        initial_loss = None
        for epoch in range(5):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                
                if initial_loss is None:
                    initial_loss = loss.item()
                
                loss.backward()
                optimizer.step()
        
        # Final loss should be less than initial
        final_loss = loss.item()
        assert final_loss < initial_loss

    def test_evaluation_with_dataloader(self):
        """Test model evaluation using test dataloader."""
        from spotoptim.nn.linear_regressor import LinearRegressor
        import torch.nn as nn
        
        train_loader, test_loader, _ = get_diabetes_dataloaders(batch_size=32)
        
        model = LinearRegressor(input_dim=10, output_dim=1, num_hidden_layers=0)
        criterion = nn.MSELoss()
        
        # Evaluate on test set
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Should have computed some loss
        assert avg_loss > 0
        assert num_batches > 0
