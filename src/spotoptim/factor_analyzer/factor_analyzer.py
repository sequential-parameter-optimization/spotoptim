# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-FileCopyrightText: 2022 Jeremy Biggs <jeremy.m.biggs@gmail.com>
# SPDX-FileCopyrightText: 2022 Nitin Madnani <nmadnani@ets.org>
# SPDX-FileCopyrightText: 2022 Educational Testing Service
#
# SPDX-License-Identifier: AGPL-3.0-or-later AND GPL-2.0-or-later
#
# Confirmatory factor analysis using machine learning methods.
# Re-implementation of the factor-analyzer package.
#
# See https://factor-analyzer.readthedocs.io/en/latest/introduction.html for
# more details.
#
# Authors of the original implementation:
# * Jeremy Biggs (jeremy.m.biggs@gmail.com)
# * Nitin Madnani (nmadnani@ets.org)
# Organization: Educational Testing Service
# Date: 2022-09-05
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

"""
Factor analysis using MINRES or ML, with optional rotation using Varimax or Promax.
"""

import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import scipy as sp
from scipy.optimize import minimize
from scipy.stats import chi2, pearsonr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.extmath import randomized_svd
from sklearn.utils.validation import check_is_fitted

from .factor_analyzer_rotator import OBLIQUE_ROTATIONS, POSSIBLE_ROTATIONS, Rotator
from .factor_analyzer_utils import corr, impute_values, partial_correlations, smc

POSSIBLE_SVDS = ["randomized", "lapack"]

POSSIBLE_IMPUTATIONS = ["mean", "median", "drop"]

POSSIBLE_METHODS = ["ml", "mle", "uls", "minres", "principal"]


def calculate_kmo(x):
    """
    Calculate the Kaiser-Meyer-Olkin criterion for items and overall.

    This statistic represents the degree to which each observed variable is
    predicted, without error, by the other variables in the dataset.
    In general, a KMO < 0.6 is considered inadequate.

    Args:
        x (array-like): The array from which to calculate KMOs.

    Returns:
        kmo_per_variable (numpy.ndarray): The KMO score per item.
        kmo_total (float): The overall KMO score.
    """
    # calculate the partial correlations
    partial_corr = partial_correlations(x)

    # calcualte the pair-wise correlations
    x_corr = corr(x)

    # fill matrix diagonals with zeros
    # and square all elements
    np.fill_diagonal(x_corr, 0)
    np.fill_diagonal(partial_corr, 0)

    partial_corr = partial_corr**2
    x_corr = x_corr**2

    # calculate KMO per item
    partial_corr_sum = np.sum(partial_corr, axis=0)
    corr_sum = np.sum(x_corr, axis=0)
    kmo_per_item = corr_sum / (corr_sum + partial_corr_sum)

    # calculate KMO overall
    corr_sum_total = np.sum(x_corr)
    partial_corr_sum_total = np.sum(partial_corr)
    kmo_total = corr_sum_total / (corr_sum_total + partial_corr_sum_total)
    return kmo_per_item, kmo_total


def calculate_bartlett_sphericity(x):
    """
    Compute the Bartlett sphericity test.

    H0: The matrix of population correlations is equal to I.
    H1: The matrix of population correlations is not equal to I.

    The formula for Bartlett's Sphericity test is:

    .. math:: -1 * (n - 1 - ((2p + 5) / 6)) * ln(det(R))

    Where R det(R) is the determinant of the correlation matrix,
    and p is the number of variables.

    Args:
        x (array-like): The array for which to calculate sphericity.

    Returns:
        statistic (float): The chi-square value.
        p_value (float): The associated p-value for the test.
    """
    n, p = x.shape
    x_corr = corr(x)

    corr_det = np.linalg.det(x_corr)
    statistic = -np.log(corr_det) * (n - 1 - (2 * p + 5) / 6)
    degrees_of_freedom = p * (p - 1) / 2
    p_value = chi2.sf(statistic, degrees_of_freedom)
    return statistic, p_value


class FactorAnalyzer(BaseEstimator, TransformerMixin):
    """
    The main exploratory factor analysis class.

    This class:
        (1) Fits a factor analysis model using minres, maximum likelihood,
            or principal factor extraction and returns the loading matrix
        (2) Optionally performs a rotation, with method including:

            (a) varimax (orthogonal rotation)
            (b) promax (oblique rotation)
            (c) oblimin (oblique rotation)
            (d) oblimax (orthogonal rotation)
            (e) quartimin (oblique rotation)
            (f) quartimax (orthogonal rotation)
            (g) equamax (orthogonal rotation)

    Args:
        n_factors (int): The number of factors to select. Defaults to 3.
        rotation (str, optional): The type of rotation to perform after fitting the factor analysis model.
            If set to None, no rotation will be performed, nor will any associated Kaiser normalization.
            Possible values include:
                (a) varimax (orthogonal rotation)
                (b) promax (oblique rotation)
                (c) oblimin (oblique rotation)
                (d) oblimax (orthogonal rotation)
                (e) quartimin (oblique rotation)
                (f) quartimax (orthogonal rotation)
                (g) equamax (orthogonal rotation)
            Defaults to 'promax'.
        method (str, optional): The fitting method to use, either 'minres', 'ml', or 'principal'.
            Defaults to 'minres'.
        use_smc (bool, optional): Whether to use squared multiple correlation as starting guesses for factor analysis.
            Defaults to True.
        bounds (tuple, optional): The lower and upper bounds on the variables for "L-BFGS-B" optimization.
            Defaults to (0.005, 1).
        impute (str, optional): How to handle missing values, if any, in the data: (a) use list-wise
            deletion ('drop'), or (b) impute the column median ('median'), or impute the column mean ('mean').
            Defaults to 'median'.
        is_corr_matrix (bool, optional): Set to True if the data is the correlation matrix.
            Defaults to False.
        svd_method (str, optional): The SVD method to use when method is 'principal'.
            If 'lapack', use standard SVD from scipy.linalg.
            If 'randomized', use faster randomized_svd function from scikit-learn.
            Defaults to 'randomized'.
        rotation_kwargs (dict, optional): Dictionary containing keyword arguments for the rotation method.
            Defaults to None.

    Attributes:
        loadings_ (numpy.ndarray): The factor loadings matrix. None, if fit() has not been called.
        corr_ (numpy.ndarray): The original correlation matrix. None, if fit() has not been called.
        rotation_matrix_ (numpy.ndarray): The rotation matrix, if a rotation has been performed. None otherwise.
        structure_ (numpy.ndarray or None): The structure loading matrix. This only exists if rotation is 'promax'.
        phi_ (numpy.ndarray or None): The factor correlations matrix. This only exists if rotation is 'oblique'.

    Notes:
        This code was partly derived from the excellent R package `psych`.

    References:
        [1] https://github.com/cran/psych/blob/master/R/fa.R

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from spotoptim.factor_analyzer import FactorAnalyzer
        >>> df_features = pd.read_csv('src/spotoptim/datasets/test02.csv')
        >>> fa = FactorAnalyzer(rotation=None)
        >>> fa = fa.fit(df_features)
        >>> np.round(fa.loadings_, 2)
        array([[-0.13,  0.16,  0.74],
               [ 0.04,  0.05,  0.01],
               [ 0.35,  0.61, -0.07],
               [ 0.45,  0.72, -0.08],
               [ 0.37,  0.44, -0.02],
               [ 0.74, -0.15,  0.3 ],
               [ 0.74, -0.16, -0.21],
               [ 0.83, -0.21,  0.05],
               [ 0.76, -0.24, -0.12],
               [ 0.82, -0.12,  0.18]])
        >>> np.round(fa.get_communalities(), 2)
        array([0.59, 0.  , 0.5 , 0.73, 0.33, 0.66, 0.62, 0.73, 0.65, 0.71])
    """

    def __init__(
        self,
        n_factors=3,
        rotation="promax",
        method="minres",
        use_smc=True,
        is_corr_matrix=False,
        bounds=(0.005, 1),
        impute="median",
        svd_method="randomized",
        rotation_kwargs=None,
    ):
        """Initialize the factor analyzer."""
        self.n_factors = n_factors
        self.rotation = rotation
        self.method = method
        self.use_smc = use_smc
        self.bounds = bounds
        self.impute = impute
        self.is_corr_matrix = is_corr_matrix
        self.svd_method = svd_method
        self.rotation_kwargs = rotation_kwargs

        # default matrices to None
        self.mean_ = None
        self.std_ = None

        self.phi_ = None
        self.structure_ = None

        self.corr_ = None
        self.loadings_ = None
        self.rotation_matrix_ = None
        self.weights_ = None

    def _arg_checker(self):
        """
        Check the input parameters to make sure they're properly formattted.

        We need to do this to ensure that the FactorAnalyzer class can be
        properly cloned when used with grid search CV, for example.
        """
        self.rotation = (
            self.rotation.lower() if isinstance(self.rotation, str) else self.rotation
        )
        if self.rotation not in POSSIBLE_ROTATIONS + [None]:
            raise ValueError(
                f"The rotation must be one of the following: {POSSIBLE_ROTATIONS + [None]}"
            )

        self.method = (
            self.method.lower() if isinstance(self.method, str) else self.method
        )
        if self.method not in POSSIBLE_METHODS:
            raise ValueError(
                f"The method must be one of the following: {POSSIBLE_METHODS}"
            )

        self.impute = (
            self.impute.lower() if isinstance(self.impute, str) else self.impute
        )
        if self.impute not in POSSIBLE_IMPUTATIONS:
            raise ValueError(
                f"The imputation must be one of the following: {POSSIBLE_IMPUTATIONS}"
            )

        self.svd_method = (
            self.svd_method.lower()
            if isinstance(self.svd_method, str)
            else self.svd_method
        )
        if self.svd_method not in POSSIBLE_SVDS:
            raise ValueError(
                f"The SVD method must be one of the following: {POSSIBLE_SVDS}"
            )

        if self.method == "principal" and self.is_corr_matrix:
            raise ValueError(
                "The principal method is only implemented using "
                "the full data set, not the correlation matrix."
            )

        self.rotation_kwargs = (
            {} if self.rotation_kwargs is None else self.rotation_kwargs
        )

    @staticmethod
    def _fit_uls_objective(psi, corr_mtx, n_factors):  # noqa: D401
        """
        The objective function passed for unweighted least-squares (ULS).

        Args:
            psi (array-like): Value passed to minimize the objective function.
            corr_mtx (array-like): The correlation matrix.
            n_factors (int): The number of factors to select.

        Returns:
            error (float): The scalar error calculated from the residuals of the loading matrix.
        """
        np.fill_diagonal(corr_mtx, 1 - psi)

        # get the eigen values and vectors for n_factors
        values, vectors = sp.linalg.eigh(corr_mtx)
        values = values[::-1]

        # this is a bit of a hack, borrowed from R's `fac()` function;
        # if values are smaller than the smallest representable positive
        # number * 100, set them to that number instead.
        values = np.maximum(values, np.finfo(float).eps * 100)

        # sort the values and vectors in ascending order
        values = values[:n_factors]
        vectors = vectors[:, ::-1][:, :n_factors]

        # calculate the loadings
        if n_factors > 1:
            loadings = np.dot(vectors, np.diag(np.sqrt(values)))
        else:
            loadings = vectors * np.sqrt(values[0])

        # calculate the error from the loadings model
        model = np.dot(loadings, loadings.T)

        # note that in a more recent version of the `fa()` source
        # code on GitHub, the minres objective function only sums the
        # lower triangle of the residual matrix; this could be
        # implemented here using `np.tril()` when this change is
        # merged into the stable version of `psych`.
        residual = (corr_mtx - model) ** 2
        error = np.sum(residual)
        return error

    @staticmethod
    def _normalize_uls(solution, corr_mtx, n_factors):
        """
        Weighted least squares normalization for loadings using MINRES.

        Args:
            solution (array-like): The solution from the L-BFGS-B optimization.
            corr_mtx (array-like): The correlation matrix.
            n_factors (int): The number of factors to select.

        Returns:
            loadings (numpy.ndarray): The factor loading matrix.
        """
        np.fill_diagonal(corr_mtx, 1 - solution)

        # get the eigenvalues and vectors for n_factors
        values, vectors = np.linalg.eigh(corr_mtx)

        # sort the values and vectors in ascending order
        values = values[::-1][:n_factors]
        vectors = vectors[:, ::-1][:, :n_factors]

        # calculate loadings
        # if values are smaller than 0, set them to zero
        loadings = np.dot(vectors, np.diag(np.sqrt(np.maximum(values, 0))))
        return loadings

    @staticmethod
    def _fit_ml_objective(psi, corr_mtx, n_factors):  # noqa: D401
        """
        The objective function for maximum likelihood.

        Args:
            psi (array-like): Value passed to minimize the objective function.
            corr_mtx (array-like): The correlation matrix.
            n_factors (int): The number of factors to select.

        Returns:
            error (float): The scalar error calculated from the residuals of the loading matrix.

        Note:
            The ML objective is based on the `factanal()` function from ``stats``
            package in R. It may generate different results from the ``fa()``
            function in ``psych``.

        References:
            [1] https://github.com/SurajGupta/r-source/blob/master/src/library/stats/R/factanal.R
        """
        sc = np.diag(1 / np.sqrt(psi))
        sstar = np.dot(np.dot(sc, corr_mtx), sc)

        # get the eigenvalues and eigenvectors for n_factors
        values, _ = np.linalg.eigh(sstar)
        values = values[::-1][n_factors:]

        # calculate the error
        error = -(np.sum(np.log(values) - values) - n_factors + corr_mtx.shape[0])
        return error

    @staticmethod
    def _normalize_ml(solution, corr_mtx, n_factors):
        """
        Normalize loadings estimated using maximum likelihood.

        Args:
            solution (array-like): The solution from the L-BFGS-B optimization.
            corr_mtx (array-like): The correlation matrix.
            n_factors (int): The number of factors to select.

        Returns:
            loadings (numpy.ndarray): The factor loading matrix.
        """
        sc = np.diag(1 / np.sqrt(solution))
        sstar = np.dot(np.dot(sc, corr_mtx), sc)

        # get the eigenvalues for n_factors
        values, vectors = np.linalg.eigh(sstar)

        # sort the values and vectors in ascending order
        values = values[::-1][:n_factors]
        vectors = vectors[:, ::-1][:, :n_factors]

        values = np.maximum(values - 1, 0)

        # get the loadings
        loadings = np.dot(vectors, np.diag(np.sqrt(values)))

        return np.dot(np.diag(np.sqrt(solution)), loadings)

    def _fit_principal(self, X):
        """
        Fit factor analysis model using principal factor analysis.

        Args:
            X (array-like): The full data set.

        Returns:
            loadings (numpy.ndarray): The factor loadings matrix.
        """
        # standardize the data
        X = X.copy()
        X = (X - X.mean(0)) / X.std(0)

        # if the number of rows is less than the number of columns,
        # warn the user that the number of factors will be constrained
        nrows, ncols = X.shape
        if nrows < ncols and self.n_factors >= nrows:
            warnings.warn(
                "The number of factors will be "
                "constrained to min(n_samples, n_features)"
                "={}.".format(min(nrows, ncols))
            )

        # perform the randomized singular value decomposition
        if self.svd_method == "randomized":
            _, _, V = randomized_svd(X, self.n_factors, random_state=1234567890)
        # otherwise, perform the full SVD
        else:
            _, _, V = np.linalg.svd(X, full_matrices=False)

        corr_mtx = np.dot(X, V.T)
        loadings = np.array([[pearsonr(x, c)[0] for c in corr_mtx.T] for x in X.T])
        return loadings

    def _fit_factor_analysis(self, corr_mtx):
        """
        Fit factor analysis model using either MINRES or maximum likelihood.

        Args:
            corr_mtx (array-like): The correlation matrix.

        Returns:
            loadings (numpy.ndarray): The factor loading matrix.

        Raises:
            ValueError: If any of the correlations are null, most likely due to having
                zero standard deviation.
        """
        # if `use_smc` is True, get get squared multiple correlations
        # and use these as initial guesses for optimizer
        if self.use_smc:
            smc_mtx = smc(corr_mtx)
            start = (np.diag(corr_mtx) - smc_mtx.T).squeeze()
        # otherwise, just start with a guess of 0.5 for everything
        else:
            start = [0.5 for _ in range(corr_mtx.shape[0])]

        # if `bounds`, set initial boundaries for all variables;
        # this must be a list passed to `minimize()`
        if self.bounds is not None:
            bounds = [self.bounds for _ in range(corr_mtx.shape[0])]
        else:
            bounds = self.bounds

        # minimize the appropriate objective function
        # and the L-BFGS-B algorithm
        if self.method == "ml" or self.method == "mle":
            objective = self._fit_ml_objective
        else:
            objective = self._fit_uls_objective

        # use scipy to perform the actual minimization
        res = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000},
            args=(corr_mtx, self.n_factors),
        )

        if not res.success:
            warnings.warn(f"Failed to converge: {res.message}")

        # transform the final loading matrix (using wls for MINRES,
        # and ml normalization for ML), and convert to DataFrame
        if self.method == "ml" or self.method == "mle":
            loadings = self._normalize_ml(res.x, corr_mtx, self.n_factors)
        else:
            loadings = self._normalize_uls(res.x, corr_mtx, self.n_factors)
        return loadings

    def fit(self, X, y=None) -> "FactorAnalyzer":
        """
        Fit factor analysis model using either MINRES, ML, or principal factor analysis.

        By default, use SMC as starting guesses.

        Args:
            X (array-like): The data to analyze.
            y (ignored): Ignored.

        Returns:
            self: The fitted factor analyzer object.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from spotoptim.factor_analyzer import FactorAnalyzer
            >>> df_features = pd.read_csv('src/spotoptim/datasets/test02.csv')
            >>> fa = FactorAnalyzer(rotation=None)
            >>> _ = fa.fit(df_features)
            >>> np.round(fa.loadings_, 2)
            array([[-0.13,  0.16,  0.74],
                   [ 0.04,  0.05,  0.01],
                   [ 0.35,  0.61, -0.07],
                   [ 0.45,  0.72, -0.08],
                   [ 0.37,  0.44, -0.02],
                   [ 0.74, -0.15,  0.3 ],
                   [ 0.74, -0.16, -0.21],
                   [ 0.83, -0.21,  0.05],
                   [ 0.76, -0.24, -0.12],
                   [ 0.82, -0.12,  0.18]])
        """
        # check the input arguments
        self._arg_checker()

        # check if the data is a data frame,
        # so we can convert it to an array
        if isinstance(X, pd.DataFrame):
            X = X.copy().values
        else:
            X = X.copy()

        # now check the array, and make sure it
        # meets all of our expected criteria
        X = check_array(X, ensure_all_finite="allow-nan", estimator=self, copy=True)

        # check to see if there are any null values, and if
        # so impute using the desired imputation approach
        if np.isnan(X).any() and not self.is_corr_matrix:
            X = impute_values(X, how=self.impute)

        # get the correlation matrix
        if self.is_corr_matrix:
            corr_mtx = X
        else:
            corr_mtx = corr(X)
            self.std_ = np.std(X, axis=0)
            self.mean_ = np.mean(X, axis=0)

        # save the original correlation matrix
        self.corr_ = corr_mtx.copy()

        # fit factor analysis model
        if self.method == "principal":
            loadings = self._fit_principal(X)
        else:
            loadings = self._fit_factor_analysis(corr_mtx)

        # only used if we do an oblique rotations;
        # default rotation matrix to None
        phi = None
        structure = None
        rotation_mtx = None

        # whether to rotate the loadings matrix
        if self.rotation is not None:
            if loadings.shape[1] <= 1:
                warnings.warn(
                    "No rotation will be performed when "
                    "the number of factors equals 1."
                )
            else:
                if "method" in self.rotation_kwargs:
                    warnings.warn(
                        "You cannot pass a rotation method to "
                        "`rotation_kwargs`. This will be ignored."
                    )
                    self.rotation_kwargs.pop("method")
                rotator = Rotator(method=self.rotation, **self.rotation_kwargs)
                loadings = rotator.fit_transform(loadings)
                rotation_mtx = rotator.rotation_
                phi = rotator.phi_
                # update the rotation matrix for everything, except promax
                if self.rotation != "promax":
                    rotation_mtx = np.linalg.inv(rotation_mtx).T

        if self.n_factors > 1:
            # update loading signs to match column sums
            # this is to ensure that signs align with R
            signs = np.sign(loadings.sum(0))
            signs[(signs == 0)] = 1
            loadings = np.dot(loadings, np.diag(signs))

            if phi is not None:
                # update phi, if it exists -- that is, if the rotation is oblique
                # create the structure matrix for any oblique rotation
                phi = np.dot(np.dot(np.diag(signs), phi), np.diag(signs))
                structure = (
                    np.dot(loadings, phi)
                    if self.rotation in OBLIQUE_ROTATIONS
                    else None
                )

        # resort the factors according to their variance,
        # unless the method is principal
        if self.method != "principal":
            variance = self._get_factor_variance(loadings)[0]
            new_order = list(reversed(np.argsort(variance)))
            loadings = loadings[:, new_order].copy()

            # if the structure matrix exists, reorder
            if structure is not None:
                structure = structure[:, new_order].copy()

        self.phi_ = phi
        self.structure_ = structure

        self.loadings_ = loadings
        self.rotation_matrix_ = rotation_mtx
        return self

    def transform(self, X):
        """
        Get factor scores for a new data set.

        Args:
            X (array-like): The data to score using the fitted factor model.
                            Shape should be (n_samples, n_features).

        Returns:
            scores (numpy.ndarray): The latent variables of X.
                                   Shape is (n_samples, n_components).

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from spotoptim.factor_analyzer import FactorAnalyzer
            >>> df_features = pd.read_csv('src/spotoptim/datasets/test02.csv')
            >>> fa = FactorAnalyzer(rotation=None)
            >>> _ = fa.fit(df_features)
            >>> np.round(fa.transform(df_features), 2)
            array([[-1.05,  0.58,  0.17],
                   [-1.6 ,  0.9 ,  0.04],
                   [-1.22, -1.16,  0.57],
                   ...,
                   [ 0.14,  0.04,  0.29],
                   [ 1.87, -0.35, -0.68],
                   [ 0.86,  0.18, -0.79]], shape=(1678, 3))
        """
        # check if the data is a data frame,
        # so we can convert it to an array
        if isinstance(X, pd.DataFrame):
            X = X.copy().values
        else:
            X = X.copy()

        # now check the array, and make sure it
        # meets all of our expected criteria
        X = check_array(X, ensure_all_finite=True, estimator=self, copy=True)

        # meets all of our expected criteria
        check_is_fitted(self, "loadings_")

        # see if we saved the original mean and std
        if self.mean_ is None or self.std_ is None:
            warnings.warn(
                "Could not find original mean and standard deviation; using"
                "the mean and standard deviation from the current data set."
            )
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
        else:
            mean = self.mean_
            std = self.std_

        # get the scaled data
        X_scale = (X - mean) / std

        # use the structure matrix, if it exists;
        # otherwise, just use the loadings matrix
        if self.structure_ is not None:
            structure = self.structure_
        else:
            structure = self.loadings_

        try:
            self.weights_ = np.linalg.solve(self.corr_, structure)
        except Exception as error:
            warnings.warn(
                "Unable to calculate the factor score weights; "
                "factor loadings used instead: {}".format(error)
            )
            self.weights_ = self.loadings_

        scores = np.dot(X_scale, self.weights_)
        return scores

    def get_eigenvalues(self):
        """
        Calculate the eigenvalues, given the factor correlation matrix.

        Returns:
            original_eigen_values (numpy.ndarray): The original eigenvalues.
            common_factor_eigen_values (numpy.ndarray): The common factor eigenvalues.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from spotoptim.factor_analyzer import FactorAnalyzer
            >>> df_features = pd.read_csv('src/spotoptim/datasets/test02.csv')
            >>> fa = FactorAnalyzer(rotation=None)
            >>> _ = fa.fit(df_features)
            >>> ev, v = fa.get_eigenvalues()
            >>> np.round(ev, 1)
            array([...])
            >>> np.round(v, 1)
            array([...])
        """
        # meets all of our expected criteria
        check_is_fitted(self, ["loadings_", "corr_"])
        corr_mtx = self.corr_.copy()

        e_values, _ = np.linalg.eigh(corr_mtx)
        e_values = e_values[::-1]

        communalities = self.get_communalities()
        communalities = communalities.copy()
        np.fill_diagonal(corr_mtx, communalities)

        values, _ = np.linalg.eigh(corr_mtx)
        values = values[::-1]
        return e_values, values

    def get_communalities(self):
        """
        Calculate the communalities, given the factor loading matrix.

        Returns:
            communalities (numpy.ndarray): The communalities from the factor loading matrix.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from spotoptim.factor_analyzer import FactorAnalyzer
            >>> df_features = pd.read_csv('src/spotoptim/datasets/test02.csv')
            >>> fa = FactorAnalyzer(rotation=None)
            >>> _ = fa.fit(df_features)
            >>> np.round(fa.get_communalities(), 2)
            array([0.59, 0.  , 0.5 , 0.73, 0.33, 0.66, 0.62, 0.73, 0.65, 0.71])
        """
        # meets all of our expected criteria
        check_is_fitted(self, "loadings_")
        loadings = self.loadings_.copy()
        communalities = (loadings**2).sum(axis=1)
        return communalities

    def get_uniquenesses(self):
        """
        Calculate the uniquenesses, given the factor loading matrix.

        Returns:
            uniquenesses (numpy.ndarray): The uniquenesses from the factor loading matrix.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from spotoptim.factor_analyzer import FactorAnalyzer
            >>> df_features = pd.read_csv('src/spotoptim/datasets/test02.csv')
            >>> fa = FactorAnalyzer(rotation=None)
            >>> _ = fa.fit(df_features)
            >>> np.round(fa.get_uniquenesses(), 2)
            array([0.41, 1.  , 0.5 , 0.27, 0.67, 0.34, 0.38, 0.27, 0.35, 0.29])
        """
        # meets all of our expected criteria
        check_is_fitted(self, "loadings_")
        communalities = self.get_communalities()
        communalities = communalities.copy()
        uniqueness = 1 - communalities
        return uniqueness

    @staticmethod
    def _get_factor_variance(loadings):
        """
        Get the factor variances.

        This is a private helper method to get the factor variances,
        because sometimes we need them even before the model is fitted.

        Args:
            loadings (array-like): The factor loading matrix, in whatever state.

        Returns:
            variance (numpy.ndarray): The factor variances.
            proportional_variance (numpy.ndarray): The proportional factor variances.
            cumulative_variances (numpy.ndarray): The cumulative factor variances.
        """
        n_rows = loadings.shape[0]

        # calculate variance
        loadings = loadings**2
        variance = np.sum(loadings, axis=0)

        # calculate proportional variance
        proportional_variance = variance / n_rows

        # calculate cumulative variance
        cumulative_variance = np.cumsum(proportional_variance, axis=0)

        return (variance, proportional_variance, cumulative_variance)

    def get_factor_variance(self):
        """
        Calculate factor variance information.

        The factor variance information including the variance,
        proportional variance, and cumulative variance for each factor.

        Returns:
            variance (numpy.ndarray): The factor variances.
            proportional_variance (numpy.ndarray): The proportional factor variances.
            cumulative_variances (numpy.ndarray): The cumulative factor variances.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from spotoptim.factor_analyzer import FactorAnalyzer
            >>> df_features = pd.read_csv('src/spotoptim/datasets/test02.csv')
            >>> fa = FactorAnalyzer(rotation=None)
            >>> _ = fa.fit(df_features)
            >>> var, prop_var, cum_var = fa.get_factor_variance()
            >>> np.round(var, 2)
            array([3.51, 1.28, 0.74])
            >>> np.round(prop_var, 2)
            array([0.35, 0.13, 0.07])
            >>> np.round(cum_var, 2)
            array([0.35, 0.48, 0.55])
        """
        # meets all of our expected criteria
        check_is_fitted(self, "loadings_")
        loadings = self.loadings_.copy()
        return self._get_factor_variance(loadings)

    def sufficiency(self, num_observations: int) -> Tuple[float, int, float]:
        """
        Perform the sufficiency test.

        The test calculates statistics under the null hypothesis that
        the selected number of factors is sufficient.

        Args:
            num_observations (int): The number of observations in the input data that this factor
                analyzer was fit using.

        Returns:
            statistic (float): The test statistic.
            degrees (int): The degrees of freedom.
            pvalue (float): The p-value of the test.

        References:
            [1] Lawley, D. N. and Maxwell, A. E. (1971). Factor Analysis as a
                 Statistical Method. Second edition. Butterworths. P. 36.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from spotoptim.factor_analyzer import FactorAnalyzer
            >>> df_features = pd.read_csv('src/spotoptim/datasets/test01.csv')
            >>> fa = FactorAnalyzer(n_factors=3, rotation=None, method="ml")
            >>> _ = fa.fit(df_features)
            >>> stat, df, p = fa.sufficiency(df_features.shape[0])
            >>> float(np.round(stat, 2))
            1475.88
            >>> df
            663
            >>> bool(p < 0.05)
            True
        """
        nvar = self.corr_.shape[0]
        degrees = ((nvar - self.n_factors) ** 2 - nvar - self.n_factors) // 2
        obj = self._fit_ml_objective(
            self.get_uniquenesses(), self.corr_, self.n_factors
        )
        statistic = (
            num_observations - 1 - (2 * nvar + 5) / 6 - (2 * self.n_factors) / 3
        ) * obj
        pvalue = chi2.sf(statistic, df=degrees)
        return statistic, degrees, pvalue
