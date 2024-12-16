# -*- coding: utf-8 -*-
"""
Machine Learning Utilities (ML Utils).

This module provides an easy-to-use interface for accessing common machine learning
utilities, including data handling, feature selection, model evaluation, model serving,
data preprocessing, and more.

Users can import various utilities directly from `gofast.utils.ml` for seamless data
preparation, model evaluation, feature engineering, and other ML-related tasks.

Available submodules:
- `data_handling`: Functions for managing data, such as fetching, batching, and 
  computing batch sizes.
- `feature_selection`: Tools for selecting relevant features and analyzing 
  feature importance.
- `model_evaluation`: Functions for evaluating models, computing scores, and 
  interpreting predictions.
- `model_serving`: Utilities for loading and fetching models.
- `preprocessing`: Preprocessing functions for preparing and transforming 
  datasets for ML models.
- `utils`: General-purpose utility functions for handling data splits, 
  stratification, smoothing, and more.
"""

from .data_handling import ( 
    fetch_tgz,
    fetch_tgz_in,
    save_dataframes,
    dynamic_batch_size,
    get_batch_size,
    compute_batch_size,
    
    )
from .feature_selection import ( 
    bi_selector,
    get_correlated_features,
    select_feature_importances, 
    get_feature_contributions,
    display_feature_contributions
    )
from .model_evaluation import ( 
    evaluate_model,
    format_model_score,
    get_global_score,
    stats_from_prediction, 
    )
from .model_serving import ( 
    load_model, 
    fetch_model
    )
from .preprocessing import (
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
from .utils import ( 
    smart_split,
    soft_data_split, 
    smart_label_classifier, 
    stratify_categories, 
    laplace_smoothing, 
    laplace_smoothing_categorical,
    laplace_smoothing_word,
    )

__all__ = [
    'bi_selector',
    'bin_counting',
    'build_data_preprocessor',
    'compute_batch_size', 
    'discretize_categories',
    'display_feature_contributions',
    'dynamic_batch_size', 
    'evaluate_model',
    'fetch_model',
    'fetch_tgz',
    'fetch_tgz_in',
    'format_model_score',
    'generate_dirichlet_features', 
    'generate_proxy_feature', 
    'get_correlated_features',
    'get_feature_contributions',
    'get_global_score',
    'get_batch_size', 
    'handle_imbalance',
    'laplace_smoothing',
    'laplace_smoothing_categorical',
    'laplace_smoothing_word',
    'load_model',
    'make_pipe',
    'one_click_prep',
    'process_data_types', 
    'resampling',
    'save_dataframes',
    'select_feature_importances',
    'smart_label_classifier', 
    'smart_split',
    'soft_data_split',
    'soft_encoder',
    'soft_imputer',
    'soft_scaler',
    'stats_from_prediction',
    'stratify_categories', 
]