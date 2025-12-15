from spotoptim.nn.linear_regressor import LinearRegressor

def test_repr():
    params = LinearRegressor.get_default_parameters()
    print(params)

if __name__ == "__main__":
    test_repr()
