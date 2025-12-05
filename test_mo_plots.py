import numpy as np
import matplotlib.pyplot as plt
from spotoptim.mo.pareto import mo_xy_surface, mo_xy_contour

class MockModel:
    def predict(self, X):
        # f(x, y) = x^2 + y^2 + others
        return np.sum(X**2, axis=1)

if __name__ == "__main__":
    bounds = [(0, 1), (0, 1), (-1, 1)]
    models = [MockModel(), MockModel()]
    target_names = ["Objective 1", "Objective 2"]
    
    print("Testing mo_xy_surface...")
    mo_xy_surface(
        models=models, 
        bounds=bounds, 
        target_names=target_names, 
        resolution=10,
        feature_pairs=[(0, 1)]
    )
    print("mo_xy_surface finished (check if plot showed briefly or ran without error).")

    print("Testing mo_xy_contour...")
    mo_xy_contour(
        models=models, 
        bounds=bounds, 
        target_names=target_names, 
        resolution=10,
        feature_pairs=[(0, 1)]
    )
    print("mo_xy_contour finished.")
