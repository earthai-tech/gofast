"""
The Tools sub-package offers a variety of utilities for data handling, 
parameter computation, model estimation, and evaluation. It extends
mathematical concepts through the module :mod:`~gofast.tools.mathex`. 
Additionally, machine learning utilities and supplementary functionalities 
are facilitated by :mod:`~gofast.tools.mlutils` and 
:mod:`~gofast.tools.coreutils`, respectively.
 
"""

import importlib

# Define a dictionary to map 
# module names to their respective sub-packages
MODULE_MAPPING = {
    'baseutils': [
        'categorize_target', 
        'select_features', 
        'extract_target', 
        'array2hdf5',
        'get_target', 
        'fancier_downloader', 
        'labels_validator', 
        'rename_labels_in', 
        'save_or_load',
        'speed_rowwise_process'
    ],
    'coreutils': [
        'cleaner', 
        'denormalize', 
        'extract_coordinates',
        'features_in', 
        'find_features_in',
        'interpolate_grid', 
        'normalizer', 
        'parallelize_jobs',
        'pair_data',
        'projection_validator',
        'random_sampling', 
        'random_selector',
        'remove_outliers',
        'replace_data', 
        'resample_data', 
        'save_job',
        'smart_label_classifier', 
        'split_train_test',
        'split_train_test_by_id',
        'store_or_write_hdf5',
    ],
    'dataops': [
        'analyze_data_corr',
        'apply_bow_vectorization',
        'apply_tfidf_vectorization', 
        'apply_word_embeddings',
        'assess_outlier_impact', 
        'augment_data', 
        'audit_data', 
        'base_transform',
        'boxcox_transformation',
        'check_missing_data',
        'convert_date_features', 
        'correlation_ops', 
        'data_assistant', 
        'drop_correlated_features', 
        'enrich_data_spectrum', 
        'fetch_remote_data', 
        'format_long_column_names',
        'handle_categorical_features',
        'handle_datasets_with_hdfstore',
        'handle_duplicates', 
        'handle_unique_identifiers', 
        'handle_missing_data', 
        'handle_outliers_in',
        'handle_skew', 
        'inspect_data',
        'prepare_data',
        'read_data', 
        'request_data', 
        'sanitize', 
        'scale_data',
        'simple_extractive_summary',
        'store_or_retrieve_data',
        'summarize_text_columns',
        'transform_dates',
        'verify_data_integrity'
    ],
    'mathex': [
        'adaptive_moving_average', 
        'adjust_for_control_vars',
        'binning_statistic', 
        'calculate_binary_iv', 
        'calculate_optimal_bins', 
        'calculate_residuals', 
        'category_count', 
        'compute_effort_yield', 
        'compute_sunburst_data', 
        'cubic_regression', 
        'exponential_regression', 
        'get_bearing', 
        'get_distance',
        'infer_sankey_columns', 
        'interpolate1d', 
        'interpolate2d', 
        'label_importance',
        'linear_regression',
        'linkage_matrix', 
        'logarithmic_regression',
        'make_mxs', 
        'minmax_scaler',
        'moving_average',
        'normalize',
        'optimized_spearmanr', 
        'quality_control',
        'quadratic_regression', 
        'rank_data', 
        'savgol_filter', 
        'scale_y', 
        'sinusoidal_regression',
        'smooth1d', 
        'smoothing', 
        'soft_bin_stat',
        'standard_scaler', 
        'step_regression',
        'weighted_spearman_rank'
    ],
    'mlutils': [
        'bi_selector', 
        'bin_counting', 
        'build_data_preprocessor',
        'codify_variables', 
        'deserialize_data',
        'display_feature_contributions', 
        'discretize_categories', 
        'evaluate_model',
        'fetch_model', 
        'fetch_tgz', 
        'get_correlated_features',
        'get_global_score',
        'handle_imbalance',
        'laplace_smoothing', 
        'laplace_smoothing_categorical',
        'laplace_smoothing_word', 
        'load_csv', 
        'load_model',
        'make_pipe', 
        'one_click_preprocess', 
        'resampling',
        'save_dataframes', 
        'select_feature_importances', 
        'serialize_data', 
        'smart_split',
        'soft_data_split', 
        'soft_imputer', 
        'soft_scaler', 
        'stats_from_prediction',
        'stratify_categories'
    ]
}

# Lazy loader function
def __getattr__(name):
    # Loop through the module mapping to find the function
    for submodule, functions in MODULE_MAPPING.items():
        if name in functions:
            # Import the submodule if the function is found
            module = importlib.import_module('.' + submodule, 'gofast.tools')
            func = getattr(module, name)
            # Cache the imported function back into globals to speed up future imports
            globals()[name] = func
            return func
    raise AttributeError(f"No function named {name} found in gofast.tools.")

# Dynamically populate __all__ to make imports 
# like 'from gofast.tools import *' work correctly
__all__ = sum(MODULE_MAPPING.values(), [])

# Adjusted __dir__ to assist with autocomplete and dir() calls
def __dir__():
    return __all__






