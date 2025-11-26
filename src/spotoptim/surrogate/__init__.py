"""Surrogate models for SpotOptim.

This module provides two Kriging (Gaussian Process) implementations:

1. **Kriging**: Full-featured implementation with:
   - Multiple methods: interpolation, regression, reinterpolation
   - Mixed variable types: float/num, int, factor
   - Isotropic/anisotropic correlation
   - Lambda (nugget) optimization for regression
   - Compatible with SpotOptim's variable type conventions

2. **SimpleKriging**: Lightweight implementation with:
   - Gaussian kernel only
   - Basic hyperparameter optimization
   - Faster for simple problems
   - Limited to continuous variables

For most SpotOptim applications, use **Kriging** (the default).
Use **SimpleKriging** for quick prototyping or simple continuous problems.

Examples:
    >>> from spotoptim.surrogate import Kriging
    >>> model = Kriging(method='regression', seed=42)
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)

    >>> from spotoptim.surrogate import SimpleKriging
    >>> simple_model = SimpleKriging(noise=1e-10, seed=42)
    >>> simple_model.fit(X_train, y_train)
"""

from .kriging import Kriging
from .simple_kriging import SimpleKriging

__all__ = ["Kriging", "SimpleKriging"]
