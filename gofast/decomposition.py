# -*- coding: utf-8 -*-
#   Licence: BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Offers advanced decomposition functionalities for tasks like
dimensionality reduction, variance analysis, and discriminant modeling.
It replaces the more deeply nested import paths under
`gofast.analysis.decomposition` with a simpler two-level import. The
functions here can handle data preprocessing steps such as missing
value imputation and numeric feature selection, making it flexible for
various real-world datasets.

Example Usage:
    >>> import numpy as np
    >>> from sklearn.impute import SimpleImputer
    >>> from sklearn.datasets import load_iris
    >>> from gofast.utils.base_utils import select_features
    >>> from gofast.datasets import fetch_data
    >>> from gofast.decomposition import get_eigen_components

    >>> # Generate random data
    >>> X = np.random.rand(100, 5)
    >>> eigen_vals, eigen_vecs, X_transformed = get_eigen_components(X)

    >>> # Fetch a specialized dataset
    >>> data = fetch_data("bagoue analyses")
    >>> y = data['flow']
    >>> X = data.drop(columns='flow')
    >>> X = select_features(X, include='number')
    >>> X = SimpleImputer().fit_transform(X)
    >>> eigval, eigvecs, _ = get_eigen_components(X)
    >>> print(eigval)
    [1.97788909 1.34186216 1.14311674 1.02424284 0.94346533 0.92781335
     0.75249407 0.68835847 0.22168818]

    >>> # Use a classic dataset
    >>> X = load_iris().data
    >>> eigen_vals, eigen_vecs, _ = get_eigen_components(
    ...     X, scale=True, method='covariance')
    >>> eigen_vals.shape, eigen_vecs.shape
    ((4,), (4, 4))

Available Functions:
  * get_eigen_components
  * get_total_variance_ratio
  * transform_to_principal_components
  * plot_decision_regions
  * linear_discriminant_analysis
  * get_transformation_matrix

Use:
  >>> from gofast.decomposition import <function_name>
to access each function directly.
"""

from gofast.analysis.decomposition import get_eigen_components
from gofast.analysis.decomposition import get_total_variance_ratio
from gofast.analysis.decomposition import transform_to_principal_components
from gofast.analysis.decomposition import plot_decision_regions
from gofast.analysis.decomposition import linear_discriminant_analysis
from gofast.analysis.decomposition import get_transformation_matrix

__all__ = [
    "get_eigen_components",
    "plot_decision_regions",
    "transform_to_principal_components",
    "get_total_variance_ratio",
    "linear_discriminant_analysis",
    "get_transformation_matrix",
]
