# -*- coding: utf-8 -*-


TASK_MAPPING = {
    'data_preprocessing': [
        'cleaner', 
        'denormalize', 
        'normalizer', 
        'binning_statistic', 
        'category_count', 
        'handle_missing_data',
        'handle_outliers_in', 
        'handle_skew',
        'handle_duplicates', 
        'scale_data',
        'remove_outliers', 
        'replace_data',
        'handle_categorical_features',
        'correlation_ops', 
        'drop_correlated_features',
        'interpolate_grid',
        'resample_data'
    ],
    'data_transformation': [
        'boxcox_transformation', 
        'apply_bow_vectorization', 
        'apply_tfidf_vectorization', 
        'apply_word_embeddings',
        'convert_date_features', 
        'transform_dates'
    ],
    'data_management': [
        'array2hdf5',
        'store_or_retrieve_data',
        'store_or_write_hdf5',
        'handle_datasets_with_hdfstore',
        'save_or_load', 
        'fetch_remote_data',
        'read_data'
    ],
    'feature_engineering': [
        'select_features',
        'extract_target', 
        'find_features_in', 
        'features_in', 
        'get_target', 
        'pair_data',
        'select_feature_importances', 
        'bi_selector',
        'bin_counting'
    ],
    'model_preparation': [
        'split_train_test', 
        'split_train_test_by_id',
        'build_data_preprocessor',
        'make_pipe', 
        'one_click_preprocess',
        'soft_data_split',
        'smart_split',
        'soft_scaler', 
        'soft_imputer', 
        'handle_imbalance',
        'resampling'
    ],
    'data_analysis': [
        'analyze_data_corr', 
        'check_missing_data',
        'audit_data', 
        'inspect_data', 
        'quality_control',
        'verify_data_integrity',
        'correlation_ops'
    ],
    'statistics_and_math': [
        'adaptive_moving_average',
        'calculate_adjusted_lr',
        'calculate_optimal_bins',
        'calculate_binary_iv',
        'compute_effort_yield', 
        'compute_sensitivity_specificity', 
        'compute_cost_based_threshold',
        'compute_youdens_index', 
        'compute_errors',
        'linear_regression',
        'quadratic_regression',
        'cubic_regression', 
        'logarithmic_regression',
        'exponential_regression',
        'interpolate1d', 
        'interpolate2d', 
        'rank_data',
        'optimized_spearmanr', 
    ],
    'model_evaluation_and_analysis': [
        'evaluate_model', 
        'stats_from_prediction', 
        'display_feature_contributions',
        'get_feature_contributions',
        'get_correlated_features', 
        'get_global_score'
    ]
}

import warnings
import gofast.tools
from gofast.api.summary import ReportFactory, assemble_reports
from gofast.api.util import remove_extra_spaces, get_table_size 
TW = get_table_size() 

def assist_me(*tasks: str, on_error='warn'):
    """
    Provides some tool recommendations for specified tasks using the 
    :mod:`gofast.tools` library.

    Function dynamically fetches some tools related to user-specified tasks and
    organizes them into categories. It returns detailed descriptions and 
    recommendations for each tool. If an invalid task is provided, the function
    can warn the user or silently ignore the error based on the 'on_error' 
    parameter.

    Parameters
    ----------
    *tasks : str
        One or more task descriptions for which the user seeks tool recommendations.
    on_error : str, optional
        Error handling strategy when an invalid task is provided. Options are:
        - 'warn': Warn the user about the invalid input (default).
        - 'ignore': Silently ignore any invalid tasks.

    Returns
    -------
    str or dict
        If valid tasks are provided, returns a dictionary with tasks as keys 
        and tool descriptions as values. If no valid tasks are provided, 
        returns an error message.

    Examples
    --------
    >>> from gofast.tools.assistutils import assist_me
    >>> assist_me('data_preprocessing', 'model_evaluation_and_analysis')
  
    >>> assist_me('invalid_task', on_error='warn')
    Warning: Invalid task(s) provided: invalid_task. Please select from:\
        data_preprocessing, model_evaluation_and_analysis.
    """
    # Validate the input tasks against the available tasks
    valid_tasks = list(TASK_MAPPING.keys())
    invalid_tasks = [task for task in tasks if task not in valid_tasks]
    if invalid_tasks:
        if on_error == 'warn':
            warnings.warn(f"Invalid task(s) provided: {', '.join(invalid_tasks)}."
                          f" Please select from: {', '.join(valid_tasks)}")
        # Remove invalid tasks if necessary
        tasks = [task for task in tasks if task not in invalid_tasks]

    if not tasks:
        error_message = (
            "No valid tasks provided. Unable to proceed. Please provide at"
            " least one valid task. Available tasks are: {', '.join(valid_tasks)}."
            )
        return error_message

    # Initialize a dictionary to store task-tool mappings
    task_tool_mapping = {}
    for task in tasks:
        tools = TASK_MAPPING.get(task, [])
        task_tool_mapping[task] = {
            f'gofast.tools.{tool}': remove_extra_spaces( 
                getattr(gofast.tools, tool).__doc__.split(".")[0].strip().replace ("\n", '')
                )
            for tool in tools if hasattr(gofast.tools, tool)
        }

    # Check if the task_tool_mapping dictionary is empty
    if not task_tool_mapping:
        return "Task not found. Please provide a more specific task description."

    # Create reports for each category and tool description
    tables = []
    for category, tools in task_tool_mapping.items():
        tool_table = ReportFactory(title=category.replace ('_', ' ').title() )
        tool_table.add_mixed_types(tools, table_width= TW)
        tables.append(tool_table)

    # Assemble and display all the reports
    assemble_reports(*tables, display=True)  # Displays each tool with its short description

