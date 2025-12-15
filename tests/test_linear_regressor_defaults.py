from spotoptim.nn.linear_regressor import LinearRegressor

def test_get_default_parameters():
    params = LinearRegressor.get_default_parameters()
    
    assert "l1" in params.var_name
    assert "num_hidden_layers" in params.var_name
    assert "activation" in params.var_name
    assert "lr" in params.var_name
    assert "optimizer" in params.var_name
    
    # Check bounds logic
    assert len(params.bounds) == 5
    
    # Check defaults
    defaults = params.sample_default()
    assert defaults["l1"] == 64
    assert defaults["optimizer"] == "Adam"
