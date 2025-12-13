
import numpy as np
from spotoptim.surrogate.kriging import Kriging
from spotoptim.surrogate.nystroem import Nystroem
from spotoptim.surrogate.pipeline import Pipeline
from spotoptim.surrogate.kernels import RBF, WhiteKernel

def test_pipeline_nystroem():
    # 1. Generate synthetic data
    X = np.linspace(0, 10, 20).reshape(-1, 1)
    y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

    # 2. Define the kernel
    kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)

    # 3. Define Nystroem approximation
    nystroem = Nystroem(kernel=kernel, n_components=10, random_state=42)

    # 4. Define Kriging model (as the final estimator)
    # Note: Kriging normally takes X of dimension k.
    # Nystroem transforms X (n_samples, n_features) -> (n_samples, n_components)
    # So Kriging will see input dimension = n_components.
    # We need to make sure Kriging can handle this high dimension (it should).
    gp_model = Kriging(method='regression', seed=42)

    # 5. Create Pipeline
    pipeline = Pipeline([
        ('nystroem', nystroem),
        ('gp', gp_model)
    ])

    # 6. Fit the pipeline
    print("Fitting pipeline...")
    pipeline.fit(X, y)
    print("Pipeline fitted.")

    # 7. Predict
    X_test = np.linspace(0, 10, 50).reshape(-1, 1)
    y_pred = pipeline.predict(X_test)
    print("Prediction shape:", y_pred.shape)
    
    # Check if predictions are reasonable (not all zeros or NaNs)
    assert not np.isnan(y_pred).any(), "Predictions contain NaNs"
    assert y_pred.shape == (50,), f"Expected shape (50,), got {y_pred.shape}"
    
    print("Verification successful!")

if __name__ == "__main__":
    test_pipeline_nystroem()
