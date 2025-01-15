# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Provides advanced dimensionality techniques, including
principal component analysis (PCA) variants, kernel-based approaches,
and local linear embeddings. By consolidating these functions, it
reduces import complexities, enabling convenient, high-level usage in
machine learning workflows.

Examples
--------
>>> from sklearn.datasets import load_iris
>>> from gofast.analysis.dimensionality import nPCA
>>> from gofast.datasets import fetch_data
>>> X = fetch_data('Bagoue analysed dataset')
>>> pca = nPCA(X, 0.95, n_axes=3, return_X=False)
>>> pca.components_
>>> pca.feature_importances_

>>> X = load_iris().data
>>> pca_result = nPCA(X, n_components=0.95, view=True, return_X=False)
>>> print(pca_result.components_)
>>> print(pca_result.feature_importances_)

Use:
  >>> from gofast.dimensionality import get_feature_importances
to access each functionality directly.
"""

from gofast.analysis.dimensionality import get_feature_importances
from gofast.analysis.dimensionality import nPCA, kPCA, iPCA
from gofast.analysis.dimensionality import LLE, find_f_importances
from gofast.analysis.dimensionality import get_most_variance_component
from gofast.analysis.dimensionality import project_ndim_vs_explained_variance

__all__ = [
    "nPCA",
    "kPCA",
    "LLE",
    "iPCA",
    "get_most_variance_component",
    "project_ndim_vs_explained_variance",
    "get_feature_importances",
    "find_f_importances",
]



