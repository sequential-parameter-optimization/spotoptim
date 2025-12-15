import numpy as np
from spotoptim.core.data import SpotDataFromArray
from spotoptim.core.experiment import ExperimentControl
from spotoptim.hyperparameters.parameters import ParameterSet
from spotoptim.nn.linear_regressor import LinearRegressor
from spotoptim.function.torch_objective import TorchObjective
from spotoptim.SpotOptim import SpotOptim

def demo_optimization():
    print("Starting SpotOptim Refactor Demo...")
    
    # 1. Setup Data
    print("1. Setting up Data...")
    X = np.random.randn(100, 10).astype(np.float32)
    # Target: y = 3*x0 + 2*x1 + noise
    y = 3*X[:, 0:1] + 2*X[:, 1:2] + 0.1*np.random.randn(100, 1).astype(np.float32)
    data = SpotDataFromArray(X, y)
    print(f"Data shape: {X.shape} -> {y.shape}")

    # 2. Define Hyperparameters (User Friendly API)
    print("2. Defining Hyperparameters...")
    params = ParameterSet() \
        .add_float("lr", 1e-4, 1e-1, transform="log") \
        .add_int("l1", 16, 64) \
        .add_int("num_hidden_layers", 0, 2) \
        .add_categorical("activation", ["ReLU", "Tanh"]) \
        .add_categorical("optimizer", ["Adam", "SGD"])
    
    # 3. Setup Experiment Control
    print("3. Setting up ExperimentControl...")
    exp = ExperimentControl(
        dataset=data,
        model_class=LinearRegressor,
        hyperparameters=params,
        epochs=5, # Low for demo
        batch_size=16,
        metrics=["mse"]
    )
    
    # 4. Create Objective Function
    print("4. Creating Objective Function...")
    objective = TorchObjective(exp)
    
    # 5. Run SpotOptim
    print("5. Running SpotOptim...")
    # We pass bounds/types from ParameterSet to SpotOptim
    optimizer = SpotOptim(
        fun=objective,
        bounds=params.bounds,
        var_type=params.var_type,
        var_name=params.var_name,
        max_iter=5, # Short run
        n_initial=3
    )
    
    res = optimizer.optimize()
    
    print("\nOptimization Finished!")
    print(f"Best Result: {optimizer.best_y_}")
    print(f"Best Parameters: {optimizer.best_x_}")
    print(f"Best Config Dictionary: {objective._get_hyperparameters(optimizer.best_x_)}")

if __name__ == "__main__":
    demo_optimization()
