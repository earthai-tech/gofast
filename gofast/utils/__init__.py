"""
The Tools sub-package offers a variety of utilities for data handling, 
parameter computation, model estimation, and evaluation. It extends
mathematical concepts through the module :mod:`~gofast.utils.mathext`. 
Additionally, machine learning utilities and supplementary functionalities 
are facilitated by :mod:`~gofast.utils.mlutils` and 
:mod:`~gofast.utils.datautils`, respectively.
 
"""

import importlib


MODULE_MAPPING = {
    'base_utils': [
         'array2hdf5',
         'binning_statistic',
         'categorize_target',
         'category_count',
         'denormalizer',
         'detect_categorical_columns', 
         'extract_target',
         'fancier_downloader',
         'fill_NaN',
         'get_target',
         'interpolate_data',
         'interpolate_grid',
         'labels_validator',
         'normalizer',
         'remove_outliers',
         'remove_target_from_array',
         'rename_labels_in',
         'scale_y',
         'select_features',
         'select_features',
         'smooth1d',
         'smoothing',
         'soft_bin_stat',
         'speed_rowwise_process', 
         'map_values', 
    ],
    'data_utils': [ 
        'cleaner',
        'data_extractor', 
        'nan_to_na',
        'pair_data',
        'process_and_extract_data',
        'random_sampling',
        'random_selector',
        'read_excel_sheets',
        'read_worksheets',
        'resample_data', 
        'replace_data', 
        'long_to_wide', 
        'wide_to_long', 
        'repeat_feature_accross', 
        'merge_datasets', 
        'swap_ic', 
        'to_categories', 
        'pop_labels_in', 
        'truncate_data', 
        'filter_data', 
        'nan_ops', 
        'build_df', 
        
    ], 
    'io_utils':[ 
        'deserialize_data', 
        'extract_tar_with_progress', 
        'fetch_tgz_from_url', 
        'fetch_tgz_locally', 
        'fetch_json_data_from_url',
        'load_serialized_data',
        'load_csv', 
        'parse_csv',
        'parse_json',
        'parse_md',
        'parse_yaml',
        'save_job',
        'save_path',
        'serialize_data',
        'store_or_write_hdf5',
        'to_hdf5',
        'zip_extractor'
        
    ], 
    'mathext': [
        'adjust_for_control_vars',
        'calculate_adjusted_lr', 
        'calculate_binary_iv', 
        'calculate_optimal_bins', 
        'calculate_residuals', 
        'compute_balance_accuracy',
        'compute_effort_yield',
        'compute_errors', 
        'compute_p_values',
        'compute_sunburst_data',
        'compute_cost_based_threshold', 
        'compute_sensitivity_specificity', 
        'compute_youdens_index',
        'cubic_regression', 
        'exponential_regression', 
        'get_time_steps', 
        'infer_sankey_columns', 
        'label_importance',
        'linear_regression',
        'linkage_matrix', 
        'logarithmic_regression',
        'minmax_scaler',
        'normalize',
        'optimized_spearmanr', 
        'quadratic_regression', 
        'rank_data', 
        'rescale_data', 
        'sinusoidal_regression',
        'standard_scaler', 
        'step_regression',
        'weighted_spearman_rank', 
        'rescale_data', 
        'compute_coverage', 
        'compute_coverages',
        'compute_importances', 
        'get_preds', 
    ],
    'ml': [
        'bi_selector', 
        'bin_counting', 
        'build_data_preprocessor',
        'display_feature_contributions', 
        'discretize_categories', 
        'evaluate_model',
        'fetch_model', 
        'get_batch_size', 
        'generate_dirichlet_features', 
        'generate_proxy_feature', 
        'get_correlated_features',
        'get_feature_contributions',
        'get_global_score',
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
        'soft_encoder', 
        'soft_data_split', 
        'soft_imputer', 
        'soft_scaler', 
        'stats_from_prediction',
        'stratify_categories', 
        'encode_target',
    ], 
    'sys_utils': [
        'WorkflowOptimizer',
        'parallelize_jobs', 
        'safe_optimize', 
        ], 
    'contextual': [ 
        'WorkflowManager', 
        ], 
    'spatial_utils': [ 
        'spatial_sampling', 
        'extract_coordinates', 
        'batch_spatial_sampling', 
        'make_mxs_labels',
        'extract_coordinates', 
        'dual_merge', 
        'extract_zones_from', 
        ]
}

# Lazy loader function
def __getattr__(name):
    # Loop through the module mapping to find the function
    for submodule, functions in MODULE_MAPPING.items():
        if name in functions:
            # Import the submodule if the function is found
            module = importlib.import_module('.' + submodule, 'gofast.utils')
            func = getattr(module, name)
            # Cache the imported function back into globals to speed up future imports
            globals()[name] = func
            return func
    raise AttributeError(f"No function named {name} found in gofast.utils.")

# Dynamically populate __all__ to make imports 
# like 'from gofast.utils import *' work correctly
__all__ = sum(MODULE_MAPPING.values(), [])

# Adjusted __dir__ to assist with autocomplete and dir() calls
def __dir__():
    return __all__






