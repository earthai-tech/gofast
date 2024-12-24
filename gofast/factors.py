# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Provides advanced factor analysis functionalities, including
spectral factor extraction, dimensionality reduction, and various
rotation methods. These routines enable users to uncover and interpret
latent structures in complex datasets, such as those found in finance,
biology, or the social sciences. By consolidating and simplifying
imports, this file reduces the usual three-level import paths to more
concise two-level imports.

Example Usage:
    >>> import numpy as np
    >>> from gofast.factors import spectral_factor
    >>> data = np.array([[1.2, 2.3, 3.1],
    ...                  [1.8, 3.5, 4.2],
    ...                  [2.6, 4.0, 5.3],
    ...                  [0.9, 1.7, 2.0]])
    >>> loadings, factors, eigenvalues = spectral_factor(data,
    ...                                                  num_factors=2)
    >>> print("Factor Loadings Matrix:")
    >>> print(loadings)
    >>> print("Common Factors Matrix:")
    >>> print(factors)
    >>> print("Eigenvalues:")
    >>> print(eigenvalues)

Available Functions:
  * spectral_factor
  * promax_rotation
  * ledoit_wolf_score
  * evaluate_noise_impact_on_reduction
  * make_scedastic_data
  * rotated_factor
  * principal_axis_factoring
  * varimax_rotation
  * oblimin_rotation
  * evaluate_dimension_reduction
  * samples_hotellings_t_square

Use `from gofast.factors import <function_name>` to access these
capabilities directly.
"""

from gofast.analysis.factors import spectral_factor, promax_rotation
from gofast.analysis.factors import ledoit_wolf_score
from gofast.analysis.factors import evaluate_noise_impact_on_reduction
from gofast.analysis.factors import make_scedastic_data
from gofast.analysis.factors import rotated_factor, principal_axis_factoring
from gofast.analysis.factors import varimax_rotation, oblimin_rotation
from gofast.analysis.factors import evaluate_dimension_reduction
from gofast.analysis.factors import samples_hotellings_t_square

__all__ = [
    "ledoit_wolf_score",
    "evaluate_noise_impact_on_reduction",
    "make_scedastic_data",
    "rotated_factor",
    "principal_axis_factoring",
    "varimax_rotation",
    "oblimin_rotation",
    "evaluate_dimension_reduction",
    "samples_hotellings_t_square",
    "promax_rotation",
    "spectral_factor",
]


