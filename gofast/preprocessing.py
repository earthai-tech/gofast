# -*- coding: utf-8 -*-
#   Licence: BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
This module offers a collection of data preprocessing and feature engineering 
functions to prepare datasets for machine learning models. It includes tools for 
data transformation, scaling, encoding, and handling imbalances.

Users can access preprocessing utilities directly from `gofast.preprocessing`.

Available functions:
- `bin_counting`: Performs binning and counting operations for categorical data.
- `build_data_preprocessor`: Creates a customizable data preprocessing pipeline.
- `discretize_categories`: Discretizes continuous features into categorical bins.
- `generate_dirichlet_features`: Generates Dirichlet-based features for model input.
- `generate_proxy_feature`: Generates proxy features for model improvement.
- `handle_imbalance`: Addresses class imbalance using various resampling techniques.
- `make_pipe`: Constructs preprocessing pipelines for structured data.
- `one_click_prep`: Prepares the data for modeling with minimal configuration.
- `process_data_types`: Processes and converts data into suitable formats for ML models.
- `resampling`: Resamples data for balancing classes or increasing dataset size.
- `soft_encoder`: Applies soft encoding to categorical variables for smooth model training.
- `soft_imputer`: Handles missing values with soft imputation methods.
- `soft_scaler`: Scales features with soft scaling methods (e.g., robust scaling).
"""

from gofast.utils.ml.preprocessing import (
    bin_counting,
    build_data_preprocessor,
    discretize_categories,
    generate_dirichlet_features, 
    generate_proxy_feature, 
    handle_imbalance,
    make_pipe,
    one_click_prep,
    process_data_types, 
    resampling,
    soft_encoder,
    soft_imputer,
    soft_scaler,  
    )

__all__ = [
    'bin_counting',
    'build_data_preprocessor',
    'discretize_categories',
    'generate_dirichlet_features', 
    'generate_proxy_feature', 
    'handle_imbalance',
    'make_pipe',
    'one_click_prep',
    'process_data_types', 
    'resampling',
    'soft_encoder',
    'soft_imputer',
    'soft_scaler',
]
