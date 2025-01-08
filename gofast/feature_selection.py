# -*- coding: utf-8 -*-
#   Licence: BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Provides a simplified interface for feature selection functionalities 
from the `gofast.utils.ml.feature_selection` submodule. It includes methods for 
feature importance selection, correlation-based feature selection, and contribution 
analysis.

Users can access common feature selection tools directly from `gofast.feature_selection`.

Available functions:
- `bi_selector`: Selects relevant features based on bi-dimensional analysis.
- `get_correlated_features`: Identifies and selects features with high correlation.
- `select_feature_importances`: Selects features based on their importance scores.
- `get_feature_contributions`: Extracts feature contributions for interpretability.
- `display_feature_contributions`: Displays feature contributions visually.

"""

from gofast.utils.ml.feature_selection import ( 
    bi_selector,
    get_correlated_features,
    select_feature_importances, 
    get_feature_contributions,
    display_feature_contributions, 
    select_relevant_features
    
    )
from gofast.utils.base_utils import select_features 

__all__= [ 
    'bi_selector','select_features', 
    'get_correlated_features',
    'select_feature_importances', 
    'get_feature_contributions',
    'display_feature_contributions', 
    'select_relevant_features', 
    
    ]