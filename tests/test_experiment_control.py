import numpy as np
import pytest
from spotoptim.core.data import SpotDataFromArray
from spotoptim.core.experiment import ExperimentControl
from spotoptim.hyperparameters.parameters import ParameterSet

def test_parameter_set():
    p = ParameterSet()
    p.add_float("lr", 1e-4, 1e-1, log=True)
    p.add_int("layers", 1, 3)
    p.add_categorical("optimizer", ["Adam", "SGD"])
    
    assert len(p.bounds) == 3
    assert p.var_type == ["float", "int", "factor"]
    assert p.var_name == ["lr", "layers", "optimizer"]
    assert p.bounds[2] == ["Adam", "SGD"]

def test_spot_data_from_array():
    x = np.random.rand(10, 5)
    y = np.random.rand(10, 1)
    
    data = SpotDataFromArray(x, y)
    assert data.input_dim == 5
    assert data.output_dim == 1
    
    xt, yt = data.get_train_data()
    assert xt.shape == (10, 5)
    assert yt.shape == (10, 1)

def test_experiment_control():
    p = ParameterSet().add_float("x", 0, 1)
    x = np.random.rand(10, 2)
    y = np.random.rand(10, 1)
    data = SpotDataFromArray(x, y)
    
    # Mock model class
    class MockModel:
        pass
        
    exp = ExperimentControl(
        dataset=data,
        model_class=MockModel,
        hyperparameters=p,
        epochs=10
    )
    
    assert exp.epochs == 10
    assert exp.dataset.input_dim == 2
    assert exp.model_class == MockModel
