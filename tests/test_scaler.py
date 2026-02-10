# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch
import pytest
from spotoptim.utils.scaler import TorchStandardScaler


def test_scaler_basic():
    """Test basic fit and transform."""
    scaler = TorchStandardScaler()
    data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    scaler.fit(data)

    assert scaler.mean is not None
    assert scaler.std is not None

    # Check mean calculation (col 0: 3, col 1: 4)
    assert torch.allclose(scaler.mean, torch.tensor([[3.0, 4.0]]))

    transformed = scaler.transform(data)

    # Check if transformed mean is close to 0 and std close to 1
    t_mean = transformed.mean(dim=0)
    t_std = transformed.std(dim=0, unbiased=False)

    assert torch.allclose(t_mean, torch.zeros(2), atol=1e-6)
    assert torch.allclose(t_std, torch.ones(2), atol=1e-6)


def test_scaler_fit_transform():
    """Test fit_transform method."""
    scaler = TorchStandardScaler()
    data = torch.rand(10, 5)

    transformed = scaler.fit_transform(data)

    t_mean = transformed.mean(dim=0)
    t_std = transformed.std(dim=0, unbiased=False)

    assert torch.allclose(t_mean, torch.zeros(5), atol=1e-6)
    assert torch.allclose(t_std, torch.ones(5), atol=1e-6)


def test_scaler_errors():
    """Test error handling."""
    scaler = TorchStandardScaler()

    # Non-tensor input
    with pytest.raises(TypeError):
        scaler.fit([1, 2, 3])

    with pytest.raises(TypeError):
        scaler.transform([1, 2, 3])

    # Transform before fit
    data = torch.rand(5, 2)
    with pytest.raises(RuntimeError):
        scaler.transform(data)
