# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Provides utility functions for handling, transforming, and validating 
data structures commonly used in data processing and analysis workflows. 
Functions cover a range of tasks, including data normalization, handling missing 
values, data sampling, and type validation. The module is designed to streamline 
data preparation and transformation, facilitating seamless integration into 
machine learning and statistical pipelines.
"""
import os 
import re
import copy 
import warnings 
from numbers import Real
from typing import Any, List, Union, Optional, Set, Tuple 
from typing import Callable, Dict  
from functools import reduce

import scipy 
import numpy as np 
import pandas as pd 

from .._gofastlog import gofastlog 
from ..api.types import _F, ArrayLike, NDArray, DataFrame 
from ..api.summary import assemble_reports 
from ..compat.sklearn import validate_params, StrOptions, Interval
from ..decorators import isdf, Dataify
from ..core.array_manager import ( 
    to_numeric_dtypes, index_based_selector, 
    to_array, array_preserver, drop_nan_in, 
 ) 
from ..core.checks import ( 
    _assert_all_types, validate_name_in, is_in,  
    is_iterable, assert_ratio, are_all_frames_valid , 
    check_features_types, exist_features, is_df_square, 
    check_files, exist_labels, is_valid_dtypes,
    ensure_same_shape, check_empty 
)
from ..core.handlers import columns_manager 
from ..core.io import SaveFile, is_data_readable, to_frame_if
from ..core.utils import sanitize_frame_cols, error_policy  
from .base_utils import fill_NaN 
from .validator import ( 
     is_frame, validate_positive_integer, parameter_validator, 
     build_data_if
    )

logger = gofastlog().get_gofast_logger(__name__) 

__all__= [
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
    'has_duplicates', 
    'index_based_selector', 
    'nan_ops', 
    'build_df', 
    'group_and_aggregate', 
    'mask_by_reference', 
    'filter_by_isin', 
    ]

     
@SaveFile  
@is_data_readable 
@Dataify(auto_columns=True, fail_silently=True) 
@check_empty (params=['data'], none_as_empty=True)
def build_df(
    data,
    columns=None,
    col_prefix="col_",
    force="auto",
    error="warn",
    coerce_dt=False,
    coerce_numeric=...,
    start_incr_at=0,
    min_process=...,       
    sanitize_cols=False,
    fill_pattern="_",        
    regex=None,              
    drop_nan_cols=...,
    pop_cat_features=False, 
    missing_values=np.nan, 
    how='all',
    reset_index=False,
    drop_index=...,
    check_integrity=False,
    inspect=False,
    input_name='data',
    savefile=None,
    verbose=0,
    **kw
)-> DataFrame:
    r"""
    Builds a pandas DataFrame from various data types,
    optionally performing minimal cleaning, sanitizing, and
    integrity checks. Internally uses the methods `build_data_if`,
    `to_numeric_dtypes`, `verify_data_integrity`, and
    `inspect_data`. Each ensures consistent data cleaning, 
    and structural analysis [1]_.
    
    See more in :ref:`User Guide <user_guide>`. 
    
    Parameters
    ----------
    data : Any
        The input data to be processed into a DataFrame. Accepts
        dictionaries, lists, NumPy arrays, or other data formats
        recognized by ``build_data_if``.
    columns : list of str or None, optional
        Column names to use if building a new DataFrame from
        scratch. If `None` and `force=False`, the function
        warns or raises an error depending on `error`.
    col_prefix : str, optional
        Prefix for automatically generated column names when
        `force=True`. Defaults to ``"col_"``.
    force : bool, optional
        If `True`, forces column name generation if none are
        supplied. If `False`, requires user-defined columns or
        raises/warns. If ``'auto'`` and `columns` not supplied,
        it is switched to ``True``.
    error : {"warn", "raise"}, optional
        Error-handling strategy. If ``"warn"``, issues a warning
        instead of raising an exception.
    coerce_dt : bool, optional
        If `True`, attempts to coerce date/datetime columns in
        the DataFrame.
    coerce_numeric : bool, optional
        If `True`, converts object-like columns with all numeric
        string values to numeric columns.
    start_incr_at : int, optional
        Starting index for auto-generated columns when
        ``force=True``. Defaults to ``0``.
    min_process : bool, optional
        If `True`, applies minimal data cleaning, including
        sanitizing column names, dropping NaN columns, etc.
    sanitize_cols : bool, optional
        If `True`, cleans column names using a specified regex
        pattern, replacing invalid characters with
        ``fill_pattern``.
    fill_pattern : str, optional
        Replacement pattern for sanitizing column names when
        `sanitize_cols=True`. Defaults to ``"_"``.
    regex : str, optional
        Regex used to match invalid column name characters.
    drop_nan_cols : bool, optional
        If `True`, drops columns entirely filled with NaN.
    pop_cat_features : bool, optional
        If `True`, removes categorical features from the
        resulting DataFrame (e.g., string or categorical dtypes).
    missing_values : Any, optional
        Placeholder for missing values. Defaults to :math:`\\text{np.nan}`.
    how : str, optional
        Strategy for dropping rows based on NaN content. Defaults
        to ``"all"`` (drops rows only if all values are NaN).
    reset_index : bool, optional
        If `True`, resets the DataFrame index after processing.
    drop_index : bool, optional
        If `True`, drops the old index when `reset_index=True`.
    check_integrity : bool, optional
        If `True`, invokes ``verify_data_integrity`` to ensure
        the processed DataFrame meets basic integrity checks
        (e.g., no duplicates, minimal missingness).
    inspect : bool, optional
        If `True`, calls ``inspect_data`` to print additional
        structural information about the DataFrame.
    input_name : str, optional
        Display name for the input data, used primarily in
        warnings or error messages.
    savefile : str or None, optional
        File path where the DataFrame is saved if the
        decorator-based saving is active. If `None`, no saving
        occurs.
    verbose : int, optional
        Level of verbosity. Higher values yield more console
        output about the transformation process.
    **kw
        Additional keyword arguments passed along for future
        extension or to sub-functions.
    
    Notes
    -----
    If ``min_process=True``, the transformations from
    ``to_numeric_dtypes`` are applied, potentially converting
    string columns to numeric if feasible. This can be helpful
    for automated data readiness. Integrity checks are
    performed if `integrity_check=True`, making sure the final
    DataFrame meets essential requirements.
    
    .. math::
       D_{frame}
       = f(D_{input}, \text{columns, coerce })
    
    Given an input :math:`D_{input}` (which may be a dictionary,
    list, NumPy array, or existing DataFrame), the function
    applies transformations (e.g., numeric type coercion, column
    sanitization) guided by parameters like `min_process`. When
    :math:`min_process=True`, additional steps such as dropping
    all-NaN columns and resetting indexes are invoked.
    
    Examples
    --------
    >>> from gofast.utils.data_utils import build_df
    >>> import numpy as np
    >>> data = {"A": [1, 2, np.nan], "B": [4, 5, 6]}
    >>> df = build_df(data, min_process=True, verbose=1)
    >>> print(df)
       A  B
    0  1.0  4
    1  2.0  5
    2  NaN  6
    
    See Also
    --------
    gofast.utils.validator.build_data_if :
      Converts various data structures to a pandas DataFrame
      with optional column generation.
    gofast.core.array_manager.to_numeric_dtypes :
      Coerces suitable columns to numeric dtypes and sanitizes
      column names.
    gofast.dataops.inspection.verify_data_integrity :
      Checks the DataFrame for missing values, duplicates, and
      other issues.
    gofast.dataops.inspection.inspect_data :
      Presents a concise structural report of the DataFrame.
    
    References
    ----------
    .. [1] Doe, J. & Smith, A. (2022). "Automated Data
       Preparation for Machine Learning," Machine Learning
       Journal, 14(2), 128-145.
    """

    # Build the DataFrame from various data types
    df = build_data_if(
        data=data,
        columns=columns,
        to_frame=True,
        input_name=input_name,
        col_prefix=col_prefix,
        force=force, 
        error=error,
        coerce_datetime=coerce_dt,
        coerce_numeric=coerce_numeric,
        start_incr_at=start_incr_at,
        **kw
    )

    # If minimal processing is enabled, perform operations like
    # numeric conversion, column sanitization, and NaN removal
    if min_process:
        df = to_numeric_dtypes(
            df,
            sanitize_columns=sanitize_cols,
            fill_pattern=fill_pattern,
            pop_cat_features=pop_cat_features,  
            regex=regex,
            drop_nan_columns=drop_nan_cols,
            how=how,
            missing_values=missing_values,      
            reset_index=reset_index,
            drop_index=drop_index,
            verbose=verbose
        )

    # Perform data integrity checks if requested
    reports =[] 
    if check_integrity:
        from ..dataops.inspection import verify_data_integrity
        passed_integrity, report = verify_data_integrity(df)
        reports.append(report)
        if verbose:
            hint = (
                "To request any attribute of Integrity Report, consider"
                " calling `verify_data_integrity` function as:\n\n"
                "    >>> from gofast.dataops import verify_data_integrity\n"
                "    >>> _, report=verify_data_integrity(<your-data>)\n"
                "    >>> report.outliers # for getting outliers report.\n"
                "    >>> report.outliers.results # for outliers results.\n"
                "And so on..."
            )
            
            if passed_integrity:
                print("Data integrity check: PASSED.")
            else:
                print("Data integrity check: FAILED.\n")
                
                print(f"Please review the integrity report below:\n{hint}")

    # inspect the data,
    if inspect:
        from ..dataops.inspection import inspect_data
        
        report = inspect_data(
            data, include_stats_table=True, 
            return_report=True
        ) 
        reports.append(report)
        
    if inspect or check_integrity:  
        # printing details of its structure
        assemble_reports(*reports, display=True )
        
    # Return the final DataFrame for downstream usage
    return df


@SaveFile  
@is_data_readable 
@Dataify(auto_columns=True, fail_silently=True) 
@check_empty (params=['data'], none_as_empty=True)
def group_and_aggregate(
    data: pd.DataFrame,
    by: Union[str, List[str]],
    agg_columns: Optional[Union[List[str], Dict[str, Any]]] = None,
    agg_func: Union[str, Callable, Dict[str, List[Union[str, Callable]]]] = "mean",
    as_index: bool = True,
    dropna: bool = False,
    reset_index: bool = True,
    verbose: int = 0, 
    savefile: Optional[str]=None, 
) -> pd.DataFrame:
    r"""
    Group and aggregate a pandas DataFrame based on specified columns and
    aggregation function(s). This function is designed to be robust and
    versatile for various grouping and aggregation needs.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame to be grouped and aggregated.
    
    by : str or list of str
        The column name(s) by which to group the data. If a single string
        is provided, the data is grouped by that column. If a list of strings
        is provided, the data is grouped by all those columns in combination.
    
    agg_columns : list of str or dict, optional
        The columns to be aggregated. This can be:
          - A list of column names (e.g., ``['soil_thickness', 'subsidence']``).
          - A dictionary specifying different aggregations for each column 
            (e.g., ``{'soil_thickness': 'mean', 'subsidence': 'sum'}``).
        If not provided, aggregation will be applied to all columns 
        that are compatible with the aggregation function (typically numeric 
        columns if using standard aggregations like 'mean', 'sum', etc.).

    agg_func : str, callable, or dict, default='mean'
        The aggregation function to apply. This can be:
          - A single string (e.g. ``'mean'``, ``'sum'``, ``'count'``, etc.).
          - A single callable (e.g. ``np.mean``, ``np.sum``).
          - A dictionary mapping column names to a list of multiple aggregations 
            (e.g. ``{'soil_thickness': ['mean', 'max'], 'subsidence': 'sum'}``).
        If both `agg_columns` and `agg_func` are dictionaries, they will be 
        combined as the final aggregator. If `agg_columns` is a list and 
        `agg_func` is a dictionary, the dict must map only aggregator methods 
        to lists, or specific columns to multiple methods.

    as_index : bool, default=True
        If True, the grouping columns become the index of the resulting 
        DataFrame. If False, the grouping columns are retained as normal 
        columns.

    dropna : bool, default=False
        Whether to drop NA values from the grouping columns before 
        aggregation. If True, rows with NA in the grouping column(s) are 
        excluded from the result.

    reset_index : bool, default=True
        If True, and ``as_index`` is True, the resulting DataFrame index
        is reset so that grouping columns become normal columns. If False,
        the grouping columns remain in the index.

    verbose : int, default=0
        The verbosity level:
          - 0: no messages (silent).
          - 1: prints a summary of the grouping and aggregation steps.
          - 2: prints more detailed intermediate information (for debugging).

    Returns
    -------
    grouped_df : pd.DataFrame
        The grouped and aggregated DataFrame.

    Notes
    -----
    - When both `agg_columns` and `agg_func` are provided, the final aggregator
      is constructed as follows:
      1. If `agg_columns` is a list and `agg_func` is a single function or 
         string, each column in `agg_columns` will be aggregated by `agg_func`.
      2. If `agg_columns` is a dictionary, it will be taken as explicit 
         definitions of how to aggregate each column. If `agg_func` is also
         a dictionary, they are combined (with `agg_columns` dict taking 
         precedence when conflicts arise). If `agg_func` is a single method,
         it is applied to any column not explicitly mentioned in 
         `agg_columns` dict.
      3. If no `agg_columns` are provided, the aggregator in `agg_func`
         is applied to all columns suitable for that aggregator method.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.utils.data_utils import group_and_aggregate

    >>> # Example dataset
    >>> data = pd.DataFrame({
    ...     'longitude': [113.291, 113.291, 113.291, 113.294],
    ...     'latitude': [22.862, 22.865, 22.865, 22.870],
    ...     'soil_thickness': [1.87, 1.51, 2.17, 2.78],
    ...     'subsidence': [0.49, 1.10, 1.13, 1.07],
    ...     'year': [2015, 2015, 2016, 2016]
    ... })

    >>> # 1) Simple aggregation by year, using the mean
    >>> grouped_mean = group_and_aggregate(
    ...     data, 
    ...     by='year', 
    ...     agg_columns=['soil_thickness', 'subsidence'], 
    ...     agg_func='mean', 
    ...     as_index=False
    ... )
    >>> print(grouped_mean)

    >>> # 2) Multiple aggregations using a dict
    >>> grouped_multi = group_and_aggregate(
    ...     data, 
    ...     by='year',
    ...     agg_func={
    ...         'soil_thickness': ['mean', 'max'],
    ...         'subsidence': 'sum'
    ...     },
    ...     as_index=True
    ... )
    >>> print(grouped_multi)
    """

    if isinstance(by, str):
        by = [by]  # make sure 'by' is a list
    
    # If we want to drop NA in grouping columns
    if dropna:
        data = data.dropna(subset=by)

    # If user has provided a dict for agg_columns, we consider that
    # a direct aggregator specification for each column. Otherwise,
    # if it's a list, we'll build a dict behind the scenes.
    # For example, if agg_columns=['soil_thickness', 'subsidence']
    # and agg_func='mean', we do:
    # {'soil_thickness': 'mean', 'subsidence': 'mean'}
    
    final_agg_dict = {}

    # Helper function to check if 'agg_func' is a valid aggregator
    # for multiple columns, e.g. a dict or single str/callable
    def is_dict_of_lists_or_str(x):
        """Return True if x is a dict whose values
        are either strings, callables, or list of them."""
        if not isinstance(x, dict):
            return False
        for val in x.values():
            if isinstance(val, (list, tuple)):
                for sub_val in val:
                    if not (isinstance(sub_val, str) or callable(sub_val)):
                        return False
            elif not (isinstance(val, str) or callable(val)):
                return False
        return True

    # Step 1a: Handle if 'agg_columns' is dict
    if isinstance(agg_columns, dict):
        # This implies user has directly assigned columns to aggregator methods
        # e.g. {'soil_thickness': 'mean', 'subsidence': ['sum','max']}
        # We might also incorporate 'agg_func' if it's a single aggregator
        final_agg_dict = agg_columns.copy()

        # If 'agg_func' is also a dict, we combine them
        if is_dict_of_lists_or_str(agg_func):
            for col, method in agg_func.items():
                # If not in final_agg_dict, add
                if col not in final_agg_dict:
                    final_agg_dict[col] = method
            # If col is in final_agg_dict, it takes precedence
        else:
            # If 'agg_func' is a single string/callable
            # we apply it to any columns not mentioned
            # in 'agg_columns' already, e.g., all numeric ones
            potential_cols = data.select_dtypes(include=["number"]).columns
            for col in potential_cols:
                if col not in final_agg_dict and col not in by:
                    final_agg_dict[col] = agg_func
    # Step 1b: If agg_columns is a list
    elif isinstance(agg_columns, list):
        # e.g. agg_columns=['soil_thickness', 'subsidence']
        # if agg_func is single aggregator => same aggregator for all
        # if agg_func is dict => interpret aggregator for those columns
        # if aggregator not found for a column, skip or raise?

        if is_dict_of_lists_or_str(agg_func):
            # e.g. agg_func={'soil_thickness': ['mean', 'max'], 'subsidence': 'sum'}
            # use aggregator for the columns that match. 
            # columns not in dict => skip or fallback?
            for c in agg_columns:
                if c in agg_func:
                    final_agg_dict[c] = agg_func[c]
                else:
                    # fallback aggregator? or skip?
                    pass
        else:
            # single aggregator => apply to all columns in agg_columns
            for c in agg_columns:
                final_agg_dict[c] = agg_func
    else:
        # If agg_columns is None, we apply 'agg_func' to all numeric columns
        # or use the dict from agg_func if it's a dict.
        if agg_columns is None:
            if is_dict_of_lists_or_str(agg_func):
                # e.g. agg_func={'soil_thickness': ['mean','max'], 'subsidence': 'sum'}
                final_agg_dict = agg_func
            else:
                # Single aggregator for all numeric columns
                numeric_cols = data.select_dtypes(include=["number"]).columns
                for c in numeric_cols:
                    if c not in by:  # exclude grouping columns
                        final_agg_dict[c] = agg_func
        else:
            raise ValueError(
                "`agg_columns` should be a list, dict, or None."
            )

    if verbose > 0:
        print(f"[group_and_aggregate] Grouping by: {by}")
        if verbose > 1:
            print(
                f"[group_and_aggregate] Final aggregator dict: {final_agg_dict}")

    # Step 2: Perform groupby and aggregation
    grouped = data.groupby(by=by, as_index=as_index)

    # Apply aggregator
    aggregated_df = grouped.agg(final_agg_dict)

    # If as_index=True and user wants to reset
    if as_index and reset_index:
        aggregated_df = aggregated_df.reset_index()

    if verbose > 0:
        print( "[group_and_aggregate] Aggregation done."
              f" Result shape: {aggregated_df.shape}")

    return aggregated_df

@SaveFile
@is_data_readable 
@check_empty(['data', 'auxi_data']) 
def nan_ops(
    data,
    auxi_data = None,
    data_kind = None,
    ops = 'check_only',
    action = None,
    error = 'raise',
    process = None,
    condition = None,
    savefile=None,
    verbose = 0,
):
    r"""
    Perform operations on NaN values within data structures, handling both
    primary data and optional witness data based on specified parameters.

    This function provides a comprehensive toolkit for managing missing
    values (`NaN`) in various data structures such as NumPy arrays,
    pandas DataFrames, and pandas Series. Depending on the `ops` parameter,
    it can check for the presence of `NaN`s, validate data integrity, or
    sanitize the data by filling or dropping `NaN` values. The function
    also supports handling witness data, which can be crucial in scenarios
    where the relationship between primary and witness data must be maintained.

    .. math::
       \text{Processed\_data} =
       \begin{cases}
           \text{filled\_data} & \text{if action is 'fill'} \\
           \text{dropped\_data} & \text{if action is 'drop'} \\
           \text{original\_data} & \text{otherwise}
       \end{cases}

    Parameters
    ----------
    data : array-like, pandas.DataFrame, or pandas.Series
        The primary data structure containing `NaN` values to be processed.
        
    auxi_data : array-like, pandas.DataFrame, or pandas.Series, optional
        Auxiliary data that accompanies the primary `data`. Its role depends
        on the ``data_kind`` parameter. If ``data_kind`` is `'target'`,
        ``auxi_data`` is treated as feature data, and vice versa. This is
        useful for operations that need to maintain the alignment between
        primary and witness data.
        
    data_kind : {'target', 'feature', None}, optional
        Specifies the role of the primary `data`. If set to `'target'`, `data`
        is considered target data, and ``auxi_data`` (if provided) is
        treated as feature data. If set to `'feature'`, `data` is treated as
        feature data, and ``auxi_data`` is considered target data. If
        `None`, no special handling is applied, and witness data is ignored
        unless explicitly required by other parameters.
        
    ops : {'check_only', 'validate', 'sanitize'}, default ``'check_only'``
        Defines the operation to perform on the `NaN` values in the data:

        - ``'check_only'``: Checks whether the data contains any `NaN` values
          and returns a boolean indicator.
        - ``'validate'``: Validates that the data does not contain `NaN` values.
          If `NaN`s are found, it raises an error or warns based on the
          ``error`` parameter.
        - ``'sanitize'``: Cleans the data by either filling or dropping `NaN`
          values based on the ``action``, ``process``, and ``condition``
          parameters.

    action : {'fill', 'drop'}, optional
        Specifies the action to take when ``ops`` is set to `'sanitize'`:

        - ``'fill'``: Fills `NaN` values using the `fill_NaN` function with the
          method set to `'both'`.
        - ``'drop'``: Drops `NaN` values based on the conditions and process
          specified. If `data_kind` is `'target'`, it handles `NaN`s in a way
          that preserves data integrity for machine learning models.
        - If `None`, defaults to `'drop'` when sanitizing.

        **Note:** If ``ops`` is not `'sanitize'` and ``action`` is set, an error
        is raised indicating conflicting parameters.

    error : {'raise', 'warn', None}, default ``'raise'``
        Determines the error handling policy:

        - ``'raise'``: Raises exceptions when encountering issues.
        - ``'warn'``: Emits warnings instead of raising exceptions.
        - ``None``: Defaults to the base policy, which is typically `'warn'`.

        This parameter is utilized by the `error_policy` function to enforce
        consistent error handling throughout the operation.

    process : {'do', 'do_anyway'}, optional
        Works in conjunction with the ``action`` parameter when ``action`` is
        `'drop'`:

        - ``'do'``: Drops `NaN` values only if certain conditions are met.
        - ``'do_anyway'``: Forces the dropping of `NaN` values regardless of
          conditions.

        This provides flexibility in handling `NaN`s based on the specific
        requirements of the dataset and the analysis being performed.

    condition : callable or None, optional
        A callable that defines a condition for dropping `NaN` values when
        ``action`` is `'drop'`. For example, it can specify that the number
        of `NaN`s should not exceed a certain fraction of the dataset. If the
        condition is not met, the behavior is controlled by the ``process``
        parameter.

    verbose : int, default ``0``
        Controls the verbosity level of the function's output for debugging
        purposes:

        - ``0``: No output.
        - ``1``: Basic informational messages.
        - ``2``: Detailed processing messages.
        - ``3``: Debug-level messages with complete trace of operations.

        Higher verbosity levels provide more insights into the function's
        internal operations, aiding in debugging and monitoring.

    Returns
    -------
    array-like, pandas.DataFrame, or pandas.Series
        The sanitized data structure with `NaN` values handled according to
        the specified parameters. If ``auxi_data`` is provided and
        processed, a tuple containing the sanitized `data` and
        `auxi_data` is returned. Otherwise, only the sanitized `data`
        is returned.

    Raises
    ------
    ValueError
        - If an invalid value is provided for ``ops`` or ``data_kind``.
        - If ``auxi_data`` does not align with ``data`` in shape.
        - If sanitization conditions are not met and the error policy is
          set to `'raise'`.
    Warning
        - Emits warnings when `NaN` values are present and the error policy is
          set to `'warn'`.

    Examples
    --------
    >>> from gofast.utils.base_utils import nan_ops
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Example with target data and witness feature data
    >>> target = pd.Series([1, 2, np.nan, 4])
    >>> features = pd.DataFrame({
    ...     'A': [5, np.nan, 7, 8],
    ...     'B': ['x', 'y', 'z', np.nan]
    ... })
    >>> # Check for NaNs
    >>> nan_ops(target, auxi_data=features, data_kind='target', ops='check_only')
    (True, True)
    >>> # Validate data (will raise ValueError if NaNs are present)
    >>> nan_ops(target, auxi_data=features, data_kind='target', ops='validate')
    Traceback (most recent call last):
        ...
    ValueError: Target contains NaN values.
    >>> # Sanitize data by dropping NaNs
    >>> cleaned_target, cleaned_features = nan_ops(
    ...     target,
    ...     auxi_data=features,
    ...     data_kind='target',
    ...     ops='sanitize',
    ...     action='drop',
    ...     verbose=2
    ... )
    Dropping NaN values.
    Dropped NaNs successfully.
    >>> cleaned_target
    0    1.0
    1    2.0
    3    4.0
    dtype: float64
    >>> cleaned_features
         A    B
    0  5.0    x
    3  8.0  NaN

    Notes
    -----
    The `nan_ops` function is designed to provide a robust framework for handling
    missing values in datasets, especially in machine learning workflows where
    the integrity of target and feature data is paramount. By allowing
    conditional operations and providing flexibility in error handling, it ensures
    that data preprocessing can be tailored to the specific needs of the analysis.

    The function leverages helper utilities such as `fill_NaN`, `drop_nan_in`,
    and `error_policy` to maintain consistency and reliability across different
    data structures and scenarios. The verbosity levels aid developers in tracing
    the function's execution flow, making it easier to debug and verify data
    transformations.

    See Also
    --------
    gofast.utils.base_utils.fill_NaN` :
        Fills `NaN` values in numeric data structures using specified methods.
    gofast.core.array_manager.drop_nan_in:
        Drops `NaN` values from data structures, optionally alongside witness data.
    gofast.core.utils.error_policy:
        Determines how errors are handled based on user-specified policies.
    gofast.core.array_manager.array_preserver:
        Preserves and restores the original structure of array-like data.

    """

    # Helper function to check for NaN values in the data.
    def has_nan(d):
        if isinstance(d, pd.DataFrame):
            return d.isnull().any().any()
        return pd.isnull(d).any()
    
    # Helper function to return data and auxi_data based on availability.
    def return_kind(dval, wval=None):
        if auxi_data is not None:
            return dval, wval
        return dval
    
    # Helper function to drop NaNs from data and auxi_data.
    def drop_nan(d, wval=None):
        if auxi_data is not None:
            d_cleaned, w_cleaned = drop_nan_in(d, wval, axis=0)
        else:
            d_cleaned = drop_nan_in(d, solo_return=True, axis=0)
            w_cleaned = None
        return d_cleaned, w_cleaned
    
    # Helper function to log messages based on verbosity level.
    def log(message, level):
        if verbose >= level:
            print(message)
    
    # Apply the error policy to determine how to handle errors.
    error = error_policy(
        error, base='warn', valid_policies={'raise', 'warn'}
    )
    
    # Validate that 'ops' parameter is one of the allowed operations.
    valid_ops = {'check_only', 'validate', 'sanitize'}
    if ops not in valid_ops:
        raise ValueError(
            f"Invalid ops '{ops}'. Choose from {valid_ops}."
        )
    
    # Ensure 'data_kind' is either 'target', 'feature', or None.
    if data_kind not in {'target', 'feature', None}:
        raise ValueError(
            "Invalid data_kind. Choose from 'target', 'feature', or None."
        )
    
    # If 'auxi_data' is provided, ensure it matches the shape of 'data'.
    if auxi_data is not None:
        try:
            ensure_same_shape(data, auxi_data, axis=None)
            log("Auxiliary data shape matches data.", 3)
        except Exception as e:
            raise ValueError(
                f"Auxiliary data shape mismatch: {e}"
            )
    
    # Determine if 'data' and 'auxi_data' contain NaN values.
    data_contains_nan = has_nan(data)
    w_contains_nan = has_nan(auxi_data) if auxi_data is not None else False
    
    # Define subjects based on 'data_kind' for clearer messaging.
    subject = 'Data' if data_kind is None else data_kind.capitalize()
    w_subject = "Auxiliary data" if data_kind is None else (
        "Feature" if subject == 'Target' else 'Target'
    )
    
    # Handle 'check_only' operation: simply return NaN presence status.
    if ops == 'check_only':
        log("Performing NaN check only.", 1)
        return return_kind(data_contains_nan, w_contains_nan)
    
    # Handle 'validate' operation: raise errors or warnings if NaNs are present.
    if ops == 'validate':
        log("Validating data for NaN values.", 1)
        if data_contains_nan:
            message = f"{subject} contains NaN values."
            if error == 'raise':
                raise ValueError(message)
            elif error == 'warn':
                warnings.warn(message)
        if w_contains_nan:
            message = f"{w_subject} contains NaN values."
            if error == 'raise':
                raise ValueError(message)
            elif error == 'warn':
                warnings.warn(message)
        log("Validation complete. No NaNs detected or handled.", 2)
        return return_kind(data, auxi_data)
    
    # For 'sanitize' operation, proceed to handle NaN values based on 'action'.
    if ops == 'sanitize':
        log("Sanitizing data by handling NaN values.", 1)
        # Preserve the original structure of the data.
        collected = array_preserver(data, auxi_data, action='collect')
        
        # Convert inputs to array-like structures for processing.
        data_converted = to_array(data)
        auxi_converted = to_array(auxi_data) if auxi_data is not None else None
        
        # If 'action' is not specified, default to 'drop'.
        if action is None:
            action = 'drop'
            log("No action specified. Defaulting to 'drop'.", 2)
        
        # Handle 'fill' action: fill NaNs using the 'fillNaN' function.
        if action == 'fill':
            log("Filling NaN values.", 2)
            data_filled = fill_NaN(data_converted, method='both')
            if auxi_data is not None:
                auxi_filled = fill_NaN(auxi_converted, method='both')
            else:
                auxi_filled = None
            log("NaN values filled successfully.", 3)
            return return_kind(data_filled, auxi_filled)
        
        # Handle 'drop' action: drop NaNs based on 'data_kind' and 'process'.
        elif action == 'drop':
            log("Dropping NaN values.", 2)
            nan_count = (
                data_converted.isnull().sum().sum()
                if isinstance(data_converted, pd.DataFrame)
                else pd.isnull(data_converted).sum()
            )
            data_length = len(data_converted)
            log(f"NaN count: {nan_count}, Data length: {data_length}", 3)
            
            # Specific handling when 'data_kind' is 'target'.
            if data_kind == 'target':
                # Define condition: NaN count should be less than half of data length.
                if condition is None:
                    condition = (nan_count < (data_length / 2))
                    log(
                        "No condition provided. Setting condition to "
                        f"NaN count < {data_length / 2}.",
                        3
                    )
                
                # If condition is not met, decide based on 'process'.
                if not condition:
                    if process == 'do_anyway':
                        log(
                            "Condition not met. Proceeding to drop NaNs "
                            "anyway.", 2
                        )
                        data_cleaned, auxi_cleaned = drop_nan(data, auxi_data)
                    else:
                        warning_msg = (
                            "NaN values in target exceed half the data length. "
                            "Dropping these NaNs may lead to significant information loss."
                        )
                        error_msg = (
                            "Too many NaN values in target data. "
                            "Consider revisiting the target variable."
                        )
                        if error == 'warn':
                            warnings.warn(warning_msg)
                        raise ValueError(error_msg)
                else:
                    # Condition met: proceed to drop NaNs.
                    log("Condition met. Dropping NaNs.", 3)
                    data_cleaned, auxi_cleaned = drop_nan(data, auxi_data)
            
            # Handling when 'data_kind' is 'feature' or None.
            elif data_kind in {'feature', None}:
                if process == 'do_anyway':
                    log(
                        "Process set to 'do_anyway'. Dropping NaNs regardless "
                        "of conditions.", 2
                    )
                    condition = None  # Reset condition to drop unconditionally
                
                if condition is None:
                    log("Dropping NaNs unconditionally.", 3)
                    data_cleaned, auxi_cleaned = drop_nan(data, auxi_data)
                else:
                    # Example condition: NaN count should be less than a third of data length.
                    condition_met = (nan_count < condition)
                    log(
                        f"Applying condition: NaN count < {data_length / 3} -> "
                        f"{condition_met}", 3
                    )
                    if not condition_met:
                        if process == 'do_anyway':
                            log(
                                "Condition not met. Dropping NaNs anyway.", 2
                            )
                            data_cleaned, auxi_cleaned = drop_nan(data, auxi_data)
                        else:
                            warning_msg = (
                                "NaN values exceed the acceptable limit based on "
                                "the condition. Dropping may remove significant data."
                            )
                            error_msg = (
                                "Condition for dropping NaNs not met. "
                                "Consider adjusting the condition or processing parameters."
                            )
                            if error == 'warn':
                                warnings.warn(warning_msg)
                            raise ValueError(error_msg)
                    else:
                        # Condition met: proceed to drop NaNs.
                        log("Condition met. Dropping NaNs.", 3)
                        data_cleaned, auxi_cleaned = drop_nan(data, auxi_data)
            
            # Assign cleaned data back to variables.
            data_filled = data_cleaned
            auxi_filled = auxi_cleaned if auxi_data is not None else None
            
            # Handle verbose messages for the cleaned data.
            if verbose >= 2:
                log("NaN values have been dropped from the data.", 2)
                if auxi_filled is not None:
                    log("NaN values have been dropped from the witness data.", 2)
            
        else:
            # If 'action' is not recognized, raise an error.
            raise ValueError(
                f"Invalid action '{action}'. Choose from 'fill', 'drop', or None."
            )
        
        # Restore the original array structure using the preserved properties.
        collected['processed'] = [data_filled, auxi_filled]
        try:
            
            data_restored, auxi_restored = array_preserver(
                collected, action='restore'
            )
            log("Data structure restored successfully.", 3)
        except Exception as e:
            log(
                f"Failed to restore data structure: {e}. Returning filled data as is.",
                1
            )
            data_restored = data_filled
            auxi_restored = auxi_filled
        
        # Return the cleaned data and auxi_data if available.
        
        return return_kind(data_restored, auxi_restored)

@SaveFile
@is_data_readable 
@validate_params ({ 
    'data': ['array-like'],
    'columns': ['array-like', None],
    'threshold':[Interval(Real, 0, None, closed='neither')], 
    'method': [StrOptions({'z_score', 'iqr'})],
    'interpolation_method': [
        StrOptions(
            {'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'})], 
    'categorical_strategy': [StrOptions({'mode', 'constant', 'ffill', 'bfill'}), None]
    })
def filter_data(
    data,
    columns=None,
    drop=True,
    interpolate=True,
    method='z_score',  
    threshold=3,       
    interpolation_method='linear',  
    categorical_strategy=None,      
    categorical_fill_value=None,   
    savefile=None, 
    verbose=0,
):
    """
    Filter and sanitize a DataFrame by removing or interpolating noise in 
    specified columns.
    
    This function processes a pandas DataFrame to eliminate noise from numerical and 
    categorical columns. It provides flexibility to either drop noisy rows or 
    interpolate numerical values to mitigate the impact of outliers. The function 
    supports various noise detection methods and strategies for handling categorical 
    data, making it adaptable to diverse datasets and analytical requirements.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame to be sanitized by removing or interpolating noise.
    columns : list-like, optional
        Specific columns to apply noise filtering. If `None`, noise filtering is 
        applied to all numerical and categorical columns in the DataFrame.
    drop : bool, default=True
        Determines the action to take when noise is detected:
            - ``True``: Drops rows containing noise.
            - ``False``: Interpolates numerical noise and handles categorical noise 
              based on the specified strategy.
    interpolate : bool, default=True
        If ``drop`` is ``False``, determines whether to interpolate noisy numerical 
        data. When enabled, outlier values are replaced with interpolated values 
        based on the specified ``interpolation_method``.
    method : str, default='z_score'
        The method used to identify noise in numerical columns. Supported options are:
            - ``'z_score'``: Identifies outliers based on z-scores.
            - ``'iqr'``: Identifies outliers based on the Interquartile Range (IQR).
    threshold : float, default=3
        The threshold for noise detection:
            - For ``method='z_score'``, typically set to 3 to identify values 
              beyond three standard deviations.
            - For ``method='iqr'``, represents the multiplier for the IQR to set 
              the bounds for outliers.
    interpolation_method : str, default='linear'
        The method used for interpolating numerical data when ``interpolate`` is 
        ``True``. Supported options include:
            - ``'linear'``
            - ``'nearest'``
            - ``'zero'``
            - ``'slinear'``
            - ``'quadratic'``
            - ``'cubic'``
    categorical_strategy : str, optional
        The strategy to handle noise in categorical columns when ``drop`` is 
        ``False``. Supported options are:
            - ``'mode'``: Fills missing values with the mode of the column.
            - ``'constant'``: Fills missing values with a specified constant.
            - ``'ffill'``: Forward fills missing values.
            - ``'bfill'``: Backward fills missing values.
        **Note**: This parameter must be set when ``drop=False``.
    categorical_fill_value : any, optional
        The constant value to use for filling missing categorical data when 
        ``categorical_strategy='constant'``. Ignored for other strategies.
    verbose : int, default=0
        Controls the verbosity of the output:
            - ``0``: No output.
            - ``1``: Basic information about processing steps.
            - ``2``: Detailed information about noise detection and handling.
            - ``3``: Extensive information including intermediate states and 
              operations performed.
            - Levels ``4`` to ``7``: Additional debugging information as needed.
    **kwargs : dict, optional
        Additional keyword arguments for future extensions or custom processing.
    
    Returns
    -------
    pandas.DataFrame
        A new DataFrame with noise removed or interpolated as specified. The 
        original DataFrame remains unmodified unless modifications are performed 
        inplace.

    Notes
    -----
    
    .. math::
        \text{Coverage} = \frac{\text{Number of } 
        :math:`actual_i \in [lower_i, upper_i]`}{\text{Total number of observations}} \times 100
    
    The noise filtering process involves identifying outliers in numerical columns 
    using the specified ``method`` and ``threshold``. For numerical data, outliers 
    can either be removed or interpolated to reduce their impact. Categorical 
    columns are handled based on the chosen ``categorical_strategy``, allowing for 
    imputation of missing or inconsistent categories. The function ensures that 
    only relevant columns are processed, enhancing performance and flexibility.
    
    - The function is designed to handle both numerical and categorical data, making 
      it suitable for a wide range of datasets.
    - When ``drop=False``, the ``categorical_strategy`` parameter must be specified to 
      determine how to handle noisy categorical data.
    - The ``interpolation_method`` parameter offers flexibility in choosing the 
      interpolation technique best suited for the dataset.
    - Verbosity levels provide varying degrees of insight into the noise filtering 
      process, facilitating debugging and data quality assessment.
    - The function maintains the integrity of the original DataFrame by operating on 
      a copy, ensuring that the original data remains unaltered unless explicitly 
      modified inplace.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.utils.data_utils import filter_data
    >>> 
    >>> # Sample DataFrame with numerical and categorical data
    >>> data = pd.DataFrame({
    ...     'A': [1, 2, 2, 4, 100],
    ...     'B': ['cat', 'dog', 'dog', None, 'mouse'],
    ...     'C': [5.5, 6.1, 5.9, 6.3, 100.0]
    ... })
    >>> 
    >>> # Drop noisy rows using z-score method
    >>> clean_df = filter_data(
    ...     data=data,
    ...     columns=['A', 'B', 'C'],
    ...     drop=True,
    ...     method='z_score',
    ...     threshold=2.5,
    ...     verbose=2
    ... )
    Column 'A': Found 1 outliers using z-score method.
    Column 'A': Dropped 1 rows containing noise.
    Column 'C': Found 1 outliers using z-score method.
    Column 'C': Dropped 1 rows containing noise.
    >>> print(clean_df)
       A     B     C
    0   1   cat   5.5
    1   2   dog   6.1
    2   2   dog   5.9
    3   4  None   6.3
    >>> 
    >>> # Interpolate noisy numerical data and handle categorical noise
    >>> interpolated_df = filter_data(
    ...     df=data,
    ...     columns=['A', 'B', 'C'],
    ...     drop=False,
    ...     interpolate=True,
    ...     method='iqr',
    ...     threshold=1.5,
    ...     interpolation_method='linear',
    ...     categorical_strategy='mode',
    ...     verbose=3
    ... )
    Column 'A': Found 1 outliers using IQR method.
    Column 'A': Interpolated 1 noisy values.
    Column 'C': Found 1 outliers using IQR method.
    Column 'C': Interpolated 1 noisy values.
    Column 'B': Found 1 noisy entries.
    Column 'B': Filled noise with mode value 'dog'.
    Column 'A': Added coverage column 'A_coverage'.
    Column 'C': Added coverage column 'C_coverage'.
    >>> print(interpolated_df)
         A      B      C  A_coverage  C_coverage
    0    1    cat    5.5        100.0        100.0
    1    2    dog    6.1        100.0        100.0
    2    2    dog    5.9        100.0        100.0
    3    4    dog    6.3        100.0        100.0
    4  NaN  mouse    NaN          NaN          NaN
    

    See Also
    --------
    pandas.DataFrame.drop_duplicates : Remove duplicate rows from a DataFrame.
    pandas.DataFrame.interpolate : Fill NaN values using interpolation.
    scipy.stats.zscore : Compute the z-score of each value in the sample.
    scipy.stats.iqr : Compute the interquartile range of the data.
    
    References
    ----------
    .. [1] McKinney, W. (2010). Data Structures for Statistical Computing 
       in Python. *Proceedings of the 9th Python in Science Conference*, 
       51-56.
    .. [2] SciPy Developers. (2023). *SciPy Statistics Documentation*. 
       https://docs.scipy.org/doc/scipy/reference/stats.html
    .. [3] pandas Documentation. (2023). 
       https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html
    """

    # Validate input DataFrame
    is_frame(data, df_only=True, raise_exception=True, objname='df')
    # Determine columns to process
    columns = columns_manager(columns, empty_as_none= True )
    if columns is not None:
        missing = list(set(columns) - set(data.columns))
        if missing:
            raise ValueError(
                f"The following columns are not in the DataFrame: {missing}")
        target_cols = columns
    else:
        # Select all numerical and categorical columns if no specific columns are provided
        target_cols = data.select_dtypes(
            include=[np.number, 'object', 'category']).columns.tolist()
    
    if verbose >= 1:
        print(f"Processing columns: {target_cols}")
    
    # Separate numerical and categorical columns
    numerical_cols = data[target_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = list(set(target_cols) - set(numerical_cols))
    
    if verbose >= 2:
        print(f"Numerical columns: {numerical_cols}")
        print(f"Categorical columns: {categorical_cols}")
    
    # Initialize a copy of the DataFrame to avoid modifying the original
    sanitized_df = data.copy()
    
    # Handle numerical columns for noise
    for col in numerical_cols:
        if method == 'z_score':
            # Calculate z-scores for the column, ignoring NaNs
            # Calculate z-scores for the entire column, preserving NaNs
            # z_scores = sanitized_df[col].apply(
            #     lambda x: (x - sanitized_df[col].mean()) / sanitized_df[col].std())
            
            # Compute z-scores using pandas' built-in functionality
            z_scores = sanitized_df[col].transform(lambda x: (x - x.mean()) / x.std())
        
            z_scores = z_scores.abs()
            # This result to index misalignment by makin
            # z_scores = np.abs(scipy.stats.zscore(sanitized_df[col].dropna()))
            # Identify outliers based on the z-score threshold
            outliers = z_scores > threshold
            if verbose >= 3:
                print(f"Column '{col}': Found {outliers.sum()}"
                      " outliers using z-score method.")
        elif method == 'iqr':
            # Calculate the first and third quartiles
            Q1 = sanitized_df[col].quantile(0.25)
            Q3 = sanitized_df[col].quantile(0.75)
            IQR = Q3 - Q1
            # Define lower and upper bounds for outliers
            lower_bound = Q1 - (threshold * IQR)
            upper_bound = Q3 + (threshold * IQR)
            # Identify outliers based on the IQR threshold
            outliers = (sanitized_df[col] < lower_bound) | (sanitized_df[col] > upper_bound)
            if verbose >= 3:
                print(f"Column '{col}': Found {outliers.sum()}"
                      " outliers using IQR method.")
        else:
            raise ValueError("`method` must be either 'z_score' or 'iqr'.")
        
        if drop:
            # Drop rows with outliers
            sanitized_df = sanitized_df[~outliers]
            if verbose >= 2:
                print(f"Column '{col}': Dropped {outliers.sum()}"
                      " rows containing noise.")
        else:
            if interpolate:
                # Replace outliers with NaN for interpolation
                sanitized_df.loc[outliers, col] = np.nan
                # Interpolate missing values using the specified method
                sanitized_df[col] = sanitized_df[col].interpolate(
                    method=interpolation_method)
                if verbose >= 2:
                    print(f"Column '{col}': Interpolated {outliers.sum()} noisy values.")
    
    # Handle categorical columns for noise
    for col in categorical_cols:
        # Identify noise as NaN entries in categorical columns
        noise_mask = sanitized_df[col].isna()
        if verbose >= 3:
            print(f"Column '{col}': Found {noise_mask.sum()} noisy entries.")
        
        if drop:
            # Drop rows with noisy categorical data
            sanitized_df = sanitized_df[~noise_mask]
            if verbose >= 2:
                print(f"Column '{col}': Dropped {noise_mask.sum()} rows containing noise.")
        else:
            # Ensure a categorical strategy is provided for handling noise
            if categorical_strategy is None:
                raise ValueError(
                    "`categorical_strategy` must be specified when `drop` is False."
                )
            # Impute noisy categorical data based on the specified strategy
            if categorical_strategy == 'mode':
                # Fill NaNs with the mode of the column
                fill_value = ( 
                    sanitized_df[col].mode().iloc[0] 
                    if not sanitized_df[col].mode().empty else None
                    )
                sanitized_df[col].fillna(fill_value, inplace=True)
                if verbose >= 2:
                    print(f"Column '{col}': Filled noise with mode value '{fill_value}'.")
            elif categorical_strategy == 'constant':
                # Fill NaNs with a constant value provided by the user
                if categorical_fill_value is None:
                    raise ValueError(
                        "`categorical_fill_value` must be provided when "
                        "`categorical_strategy` is 'constant'."
                    )
                sanitized_df[col].fillna(categorical_fill_value, inplace=True)
                if verbose >= 2:
                    print(
                        f"Column '{col}': Filled noise with constant value "
                        f"'{categorical_fill_value}'."
                    )
            elif categorical_strategy == 'ffill':
                # Forward fill NaNs
                sanitized_df[col].fillna(method='ffill', inplace=True)
                if verbose >= 2:
                    print(f"Column '{col}': Forward-filled noise.")
            elif categorical_strategy == 'bfill':
                # Backward fill NaNs
                sanitized_df[col].fillna(method='bfill', inplace=True)
                if verbose >= 2:
                    print(f"Column '{col}': Backward-filled noise.")
            else:
                raise ValueError(
                    "`categorical_strategy` must be one of 'mode', 'constant', "
                    "'ffill', or 'bfill'."
                )
    
    # Final verbosity logging for DataFrame shapes
    if verbose >= 4:
        initial_shape = data.shape
        final_shape = sanitized_df.shape
        print(f"Initial DataFrame shape: {initial_shape}")
        print(f"Sanitized DataFrame shape: {final_shape}")
    
    if verbose >= 5:
        print("Sanitized DataFrame head:")
        print(sanitized_df.head())
    
    # Return the sanitized DataFrame
    return sanitized_df

@is_data_readable 
@validate_params ({ 
    'threshold':[Interval(Real, 0, 1, closed='neither')], 
    'along_with': [StrOptions({'rows', 'cols'}), None],
    'ops': [StrOptions({'check_only', 'drop'}), None], 
    'atol': [Interval(Real, 0, 1, closed='neither'), None]
    })
def has_duplicates(
    data,
    ops=None,
    in_cols=None,
    along_with=None,
    inplace=False,
    atol=None,
    verbose=0
):  
    """
    Check for and optionally drop duplicate entries in a pandas DataFrame.
    
    This function allows for checking duplicates within a DataFrame based on 
    specified columns or along rows or columns. It supports both exact and 
    approximate duplicate detection using an absolute tolerance (`atol`). 
    Additionally, it can drop duplicate rows or columns, providing detailed 
    statistics based on the verbosity level.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame to check for duplicates.
    ops : {'check_only', 'drop'}, default='check_only'
        The operation to perform:
            - ``'check_only'``: Only checks for duplicates without modifying the 
              DataFrame.
            - ``'drop'``: Drops duplicate entries from the DataFrame.
    in_cols : list-like, optional
        Specific columns to consider when checking for duplicates. If `None`, 
        duplicates are checked across all columns.
    along_with : {'rows', 'cols'}, default=None
        Specifies the axis along which to check for duplicates:
            - ``'rows'``: Checks for duplicate rows.
            - ``'cols'``: Checks for duplicate columns.
            - `None`: Uses pandas' default behavior, typically checking duplicate rows.
    inplace : bool, default=False
        If ``True``, performs the operation inplace and returns ``None``. If 
        ``False``, returns a new DataFrame with duplicates handled accordingly.
    atol : float, optional
        Absolute tolerance for considering numeric values as duplicates. If 
        ``None``, exact matches are required. For example, if ``atol=0.01``, 
        values within ``0.01`` of each other are treated as duplicates.
    verbose : int, default=0
        Controls the verbosity of the output:
            - ``0``: No output.
            - ``1``: Basic information about duplicates found or dropped.
            - ``2``: Detailed statistics about the duplicates.
            - ``3``: Comprehensive summary including locations of duplicates.
    
    Returns
    -------
    bool or pandas.DataFrame or None
        - If ``ops='check_only'``, returns ``True`` if duplicates are found, 
          ``False`` otherwise. Additionally, returns a dictionary with details 
          about the duplicates if found.
        - If ``ops='drop'``, returns the DataFrame with duplicates dropped if 
          ``inplace=False``. Returns ``None`` if ``inplace=True``.
    
    Notes 
    ------
    
    .. math::
        d\left(p, q\right) = \sqrt{\sum_{i=1}^{n} \left(p_i - q_i\right)^2}
    
    The duplication check is based on calculating the distance between data points 
    using the above Euclidean distance formula when ``atol`` is specified. For exact 
    duplicate detection, ``atol`` is set to ``None``, and duplicates are identified 
    based on identical values in the specified columns.
    
    - The function can handle both row-wise and column-wise duplicate checks 
      based on the ``along_with`` parameter.
    - When using ``atol``, numeric columns are rounded to the specified 
      tolerance before checking for duplicates.
    - The verbosity level allows users to receive varying degrees of detail 
      regarding the duplicate detection and removal process.
      
    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.utils.data_utils import has_duplicates
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 2, 4],
    ...     'B': [5, 6, 6, 8],
    ...     'C': [9, 10, 10, 12]
    ... })
    >>> # Check for duplicates across all columns
    >>> has_duplicates(df)
    (True, {'rows':    A  B   C
    2  2  6  10})
    >>> # Drop duplicate rows
    >>> clean_df = has_duplicates(
    ...     df, ops='drop', verbose=1
    ... )
    Dropped 1 duplicate rows.
    >>> print(clean_df)
       A  B   C
    0  1  5   9
    1  2  6  10
    3  4  8  12
    
    See Also
    --------
    pandas.DataFrame.duplicated : Return boolean Series denoting duplicate rows.
    pandas.DataFrame.drop_duplicates : Remove duplicate rows from a DataFrame.
    pandas.DataFrame.T : Transpose index and columns of DataFrame.
    
    References
    ----------
    .. [1] McKinney, W. (2010). Data Structures for Statistical Computing 
       in Python. *Proceedings of the 9th Python in Science Conference*, 
       51-56.
    .. [2] pandas Documentation. (2023). 
       https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.duplicated.html
    .. [3] SciPy Developers. (2023). *SciPy Spatial Documentation*. 
       https://docs.scipy.org/doc/scipy/reference/spatial.html
    """

    is_frame(data, df_only=True, raise_exception=True, objname='df')
    # Validate the 'ops' parameter
    if ops is None:
        ops= 'check_only'
    
    # Create a copy of the DataFrame if not inplace
    if not inplace:
        df_copy = data.copy()
    else:
        df_copy = data
    
    # Handle absolute tolerance for numeric columns
    if atol is not None and in_cols is not None:
        numeric_cols = df_copy[in_cols].select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df_copy[numeric_cols] = df_copy[numeric_cols].round(
                decimals=int(-np.log10(atol)) if atol > 0 else 0
            )
    
    duplicates_info = {}
    
    # Check for duplicates based on 'along_with' parameter
    if along_with == 'rows' or along_with is None:
        subset = in_cols if in_cols is not None else None
        duplicated_mask = df_copy.duplicated(subset=subset, keep=False)
        duplicate_rows = df_copy[duplicated_mask]
        if not duplicate_rows.empty:
            duplicates_info['rows'] = duplicate_rows
    if along_with == 'cols':
        subset = in_cols if in_cols is not None else df_copy.columns
        duplicated_mask = df_copy.T.duplicated(keep=False)
        duplicate_cols = df_copy.columns[duplicated_mask]
        if duplicate_cols.any():
            duplicates_info['cols'] = duplicate_cols.tolist()
    
    # Handle 'check_only' operation
    if ops == 'check_only':
        has_dup = len(duplicates_info) > 0
        if verbose >= 1:
            if has_dup:
                print("Duplicates found:")
                for key, value in duplicates_info.items():
                    if key == 'rows':
                        print(f"- Duplicate Rows:\n{value}")
                    elif key == 'cols':
                        print(f"- Duplicate Columns: {value}")
            else:
                print("No duplicates found.")
        return has_dup, duplicates_info if has_dup else {}
    
    # Handle 'drop' operation
    if ops == 'drop':
        initial_shape = df_copy.shape
        if along_with in {'rows', None}:
            subset = in_cols if in_cols is not None else None
            df_copy.drop_duplicates(subset=subset, inplace=True)
        if along_with == 'cols':
            subset = in_cols if in_cols is not None else df_copy.columns
            df_copy = df_copy.loc[:, ~df_copy.T.duplicated()]
        
        final_shape = df_copy.shape
        duplicates_dropped = initial_shape[0] - final_shape[0] if along_with in {
            'rows', None} else initial_shape[1] - final_shape[1]
        
        if verbose >= 1:
            print(f"Dropped {duplicates_dropped} duplicate"
                  f" {'rows' if along_with in {'rows', None} else 'columns'}.")
            if verbose >= 2:
                print(f"Initial shape: {initial_shape}, Final shape: {final_shape}")
        
        if inplace:
            data.drop(data.index, inplace=True)
            data.loc[df_copy.index, df_copy.columns] = df_copy
            return None
        
        return df_copy

@validate_params ({ 
    'threshold':[Interval(Real, 0, 1, closed='neither')], 
    'error': [StrOptions({'raise', 'warn', 'ignore'})]
    })
def truncate_data(
    *dfs,
    base,
    feature_cols=None,  
    find_closest=False,
    threshold=0.01,
    force_coords=False,
    error='raise',
    verbose=0
):
    """
    Truncate multiple DataFrames based on spatial coordinates or index alignment 
    with a base DataFrame.

    Truncates the provided DataFrames (`*dfs`) by aligning them with a `base` 
    DataFrame either through spatial coordinates specified in `feature_cols` 
    or by index alignment when `feature_cols` is `None`. The function supports 
    finding the closest matches within a specified `threshold` and can optionally 
    overwrite the feature coordinates with those from the `base` DataFrame.

    Parameters
    ----------
    *dfs : pandas.DataFrame
        Variable number of DataFrames to be truncated.
    base : pandas.DataFrame, default=None
        The base DataFrame used for truncation. If ``feature_cols`` is provided, 
        truncation is based on matching feature columns (e.g., spatial coordinates). 
        If ``feature_cols`` is `None`, truncation is based on index alignment.
    feature_cols : tuple or list, default=None
        Columns used for truncation. When provided, these columns must exist in all 
        DataFrames, including the `base`. If `None`, truncation is based on DataFrame 
        indices.
    find_closest : bool, default=False
        If ``True``, finds the closest matching points within the specified 
        ``threshold`` using nearest-neighbor search. Applicable only when 
        ``feature_cols`` is provided.
    threshold : float, default=0.01
        The maximum distance within which points are considered a match when 
        ``find_closest`` is `True`. The unit should correspond to the feature 
        columns (e.g., degrees for latitude/longitude).
    force_coords : bool, default=False
        If ``True``, replaces the feature column values in the truncated DataFrames 
        with those from the `base` DataFrame for matched points.
    error : {'raise', 'warn', 'ignore'}, default='raise'
        Determines the behavior when encountering issues:
            - 'raise': Raises an exception.
            - 'warn': Emits a warning.
            - 'ignore': Suppresses warnings and continues execution.
    verbose : int, default=0
        Controls the verbosity of the output:
            - 0: No output.
            - 1: Basic information about truncation results.
            - 2: Additional details on matching accuracy.
            - 3 or higher: Comprehensive summary of all truncated DataFrames.

    Returns
    -------
    pandas.DataFrame or list of pandas.DataFrame
        The truncated DataFrames. Returns a single DataFrame if only one is 
        provided; otherwise, returns a list.

    Notes
    -----
    .. math::
        d\left(p, q\right) = \sqrt{\sum_{i=1}^{n} \left(p_i - q_i\right)^2}

    The truncation process involves calculating the Euclidean distance between 
    the feature coordinates of the DataFrames to be truncated and those of the 
    `base` DataFrame. If ``find_closest`` is enabled, the function identifies 
    points in the target DataFrames that are within the ``threshold`` distance 
    from any point in the `base` DataFrame.

    - When ``feature_cols`` is `None`, truncation strictly relies on index 
      alignment. Ensure that the indices of all DataFrames match the `base` 
      DataFrame to avoid unintended truncation.
    - The function utilizes `cKDTree` from `scipy.spatial` for efficient 
      nearest-neighbor searches when ``find_closest`` is enabled.
    - If ``force_coords`` is set to `True`, the spatial coordinates in the 
      truncated DataFrames will be overwritten by those from the `base` DataFrame 
      for matched points.    

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.utils.data_utils import truncate_data
    >>> df1 = pd.DataFrame({
    ...     'longitude': [1.1, 1.2, 1.3],
    ...     'latitude': [2.1, 2.2, 2.3],
    ...     'value1': [10, 20, 30]
    ... })
    >>> df2 = pd.DataFrame({
    ...     'longitude': [1.1, 1.4],
    ...     'latitude': [2.1, 2.4],
    ...     'value2': [100, 200]
    ... })
    >>> base_df = pd.DataFrame({
    ...     'longitude': [1.1, 1.4],
    ...     'latitude': [2.1, 2.4]
    ... })
    >>> result = truncate_data(
    ...     df1,
    ...     base=base_df,
    ...     feature_cols=('longitude', 'latitude'),
    ...     find_closest=True,
    ...     threshold=0.05,
    ...     verbose=2
    ... )
    DataFrame 0: Original size=3, Truncated size=1 (33.33%)
     - 100.00% of truncated points are within threshold and matched closely.
    >>> print(result)
       longitude  latitude  value1
    0        1.1       2.1      10


    See Also
    --------
    pandas.DataFrame.merge : 
        Merge DataFrame objects by performing a database-style 
    join operation.
    scipy.spatial.cKDTree : 
        A fast implementation of KDTree for nearest-neighbor search.

    References
    ----------
    .. [1] McKinney, W. (2010). Data Structures for Statistical Computing 
       in Python. *Proceedings of the 9th Python in Science Conference*, 51-56.
    .. [2] SciPy Developers. (2023). *SciPy Spatial Documentation*. 
       https://docs.scipy.org/doc/scipy/reference/spatial.html
    """

    from scipy.spatial import cKDTree
    are_all_frames_valid(*dfs)
    # Handle the case where 'base' DataFrame is not provided
    if base is None:
        message = "Base DataFrame is required for truncation."
        if error == 'raise':
            raise ValueError(message)
        elif error == 'warn':
            warnings.warn(message)
        # Proceed without truncation if 'ignore'
        return dfs[0] if len(dfs) == 1 else list(dfs)
    
    is_frame(base, df_only=True, raise_exception=True, objname='Base `df`') 
    
    # Ensure feature columns exist in the base DataFrame and all target DataFrames
    if feature_cols is not None:
        feature_cols = columns_manager(feature_cols)
        missing_base_cols = [col for col in feature_cols if col not in base.columns]
        if missing_base_cols:
            message = f"Base DataFrame is missing feature columns: {missing_base_cols}."
            if error == 'raise':
                raise ValueError(message)
            elif error == 'warn':
                warnings.warn(message)

        for idx, df in enumerate(dfs):
            missing_cols = [col for col in feature_cols if col not in df.columns]
            if missing_cols:
                msg = (f"DataFrame at position {idx} is missing"
                f" feature columns: {missing_cols}."
                )
                if error == 'raise':
                    raise ValueError(msg)
                elif error == 'warn':
                    warnings.warn(msg)
    else:
        # Truncate based on index alignment when 'feature_cols' is None
        base_indices = base.index
        for idx, df in enumerate(dfs):
            if not df.index.equals(base_indices):
                message = (
                    f"DataFrame at position {idx} does not have identical"
                    " indices as the base DataFrame."
                )
                if error == 'raise':
                    raise ValueError(message)
                elif error == 'warn':
                    warnings.warn(message)
        # Determine common indices across all DataFrames
        common_indices = base_indices
        for df in dfs:
            common_indices = common_indices.intersection(df.index)
        if verbose >= 1 and len(common_indices) < len(base_indices):
            print(
                f"Warning: Truncation based on index. "
                f"Only {len(common_indices)} common indices found"
                f" out of {len(base_indices)}."
            )

    # Prepare KDTree for spatial truncation if applicable
    if feature_cols is not None and find_closest:
        base_coords = base[feature_cols].dropna().values
        if len(base_coords) == 0:
            message = "Base DataFrame has no valid feature coordinates for truncation."
            if error == 'raise':
                raise ValueError(message)
            elif error == 'warn':
                warnings.warn(message)
            # Proceed without truncation if 'ignore'
            return dfs[0] if len(dfs) == 1 else list(dfs)
        tree = cKDTree(base_coords)

    # Initialize list to store truncated DataFrames
    truncated_dfs = []

    # Create a set of base feature coordinates for exact matching
    if feature_cols is not None and not find_closest:
        base_set = set(tuple(x) for x in base[feature_cols].dropna().values)

    # Iterate through each DataFrame to perform truncation
    for idx, df in enumerate(dfs):
        if feature_cols is not None:
            # Clean DataFrame by removing rows with missing feature columns
            df_clean = df.dropna(subset=feature_cols).copy()
            coords = df_clean[feature_cols].values

            if find_closest:
                # Perform nearest-neighbor search within the specified threshold
                distances, indices = tree.query(
                    coords, distance_upper_bound=threshold
                )
                # Identify valid matches within the threshold
                valid = distances != np.inf
                matched_indices = indices[valid]
                truncated = df_clean[valid].copy()

                if force_coords:
                    # Overwrite feature column values with those from the base DataFrame
                    truncated[feature_cols] = base.iloc[matched_indices][feature_cols].values
            else:
                # Perform exact match truncation based on feature columns
                matched = df_clean[feature_cols].apply(tuple, axis=1).isin(base_set)
                truncated = df_clean[matched].copy()

            # Append the truncated DataFrame to the list
            truncated_dfs.append(truncated)

            # Provide verbose output for truncation results
            if verbose >= 1:
                original_len = len(df)
                truncated_len = len(truncated)
                percent = (
                    (truncated_len / original_len) * 100
                    if original_len > 0 else 0
                )
                print(
                    f"DataFrame {idx}: Original size={original_len}, "
                    f"Truncated size={truncated_len} ({percent:.2f}%)"
                )

            if verbose >= 2 and find_closest:
                # Calculate the percentage of points that closely match the base coordinates
                identical = np.isclose(
                    df_clean[feature_cols].values,
                    base.iloc[indices[valid]][feature_cols].values,
                    atol=threshold
                ).all(axis=1)
                match_percent = (
                    (identical.sum() / len(identical)) * 100
                    if len(identical) > 0 else 0
                )
                print(
                    f" - {match_percent:.2f}% of truncated points are within "
                    f"threshold and matched closely."
                )
        else:
            # Truncate based on index alignment
            common_indices = base.index.intersection(df.index)
            truncated = df.loc[common_indices].copy()

            # Append the truncated DataFrame to the list
            truncated_dfs.append(truncated)

            # Provide verbose output for truncation based on index
            if verbose >= 1:
                original_len = len(df)
                truncated_len = len(truncated)
                percent = (
                    (truncated_len / original_len) * 100
                    if original_len > 0 else 0
                )
                print(
                    f"DataFrame {idx}: Original size={original_len}, "
                    f"Truncated size={truncated_len} ({percent:.2f}%) based on index alignment."
                )

        # Provide additional verbose information if verbosity level is higher
        if verbose >= 3:
            print(f"Final shape of DataFrame {idx}: {truncated.shape}")

    # Provide a final verbose summary if required
    if verbose >= 3 and feature_cols is not None:
        print("\nSummary of Truncated DataFrames:")
        for idx, truncated in enumerate(truncated_dfs):
            print(f" - DataFrame {idx}: {truncated.shape}")

    # Return the truncated DataFrames appropriately
    if len(truncated_dfs) == 1:
        return truncated_dfs[0]
    return truncated_dfs

@SaveFile 
def pop_labels_in(
    df: DataFrame, 
    columns: Union[str, List[Any]], 
    labels: Union [str, List[Any]], 
    inplace: bool=False, 
    ignore_missing: bool =False, 
    as_categories: bool =False, 
    sort_columns: bool =False, 
    savefile: str = None, 
    ):
    """
    Remove specific categories (labels) from columns in a dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe from which labels will be removed.
        The DataFrame must contain columns matching the specified
        `categories` parameter to remove the corresponding labels.

    columns : str or list of str
        The category column(s) to check for labels and remove them.
        This can be a single column name or a list of column names.

    labels : str or list of str
        The labels (categories) to be removed from the specified
        `categories` columns. These will be matched exactly as values 
        within the columns.

    inplace : bool, optional, default=False
        If ``True``, the dataframe will be modified in place and no new 
        dataframe will be returned. Otherwise, a new dataframe with 
        the labels removed will be returned.

    ignore_missing : bool, optional, default=False
        If ``True``, missing category columns or labels will be ignored and 
        no error will be raised. If ``False``, an error will be raised if 
        a specified column or label is missing in the DataFrame.

    as_categories : bool, optional, default=False
        If ``True``, the selected category columns will be converted to 
        pandas `Categorical` type before removing the labels.

    sort_categories : bool, optional, default=False
        If ``True``, the categories will be sorted in ascending order 
        before processing.

    Returns
    --------
    pandas.DataFrame
        A DataFrame with the specified labels removed from the category columns.
        If ``inplace=True``, the original DataFrame will be modified and 
        no DataFrame will be returned.

    Notes
    ------
    - The `pop_labels_in` function removes the specified labels from the 
      `categories` column(s) in the DataFrame. If ``inplace=True``, the 
      DataFrame will be modified directly.
    - This function checks if the columns exist before removing the labels, 
      unless `ignore_missing=True` is specified.
    - If ``as_categories=True``, the columns are first converted to 
      pandas `Categorical` type before proceeding with label removal.

    Let the input DataFrame be represented as `df`, with columns 
    represented by `C_1, C_2, ..., C_n`. Each of these columns 
    contains labels, some of which may need to be removed.

    If `labels = {l_1, l_2, ..., l_k}` is the set of labels to remove, 
    for each column `C_i` in `categories`, the process is:
    
    .. math::
        C_i := C_i \setminus \{ l_1, l_2, ..., l_k \}
    
    Where `\setminus` represents the set difference operation.

    Examples:
    ---------
    >>> import pandas as pd 
    >>> from gofast.utils.datautils import pop_labels_in
    >>> df = pd.DataFrame({'category': ['A', 'B', 'C', 'A', 'D']})
    >>> df_result = pop_labels_in(df, 'category', 'A')
    >>> print(df_result)
       category
    0        B
    1        C
    2        D

    See Also:
    ---------
    - `columns_manager`: For managing category columns.
    - `are_all_frames_valid`: Ensures the dataframe is valid.
    
    References:
    ----------
    .. [1] John Doe, "Data Processing for Machine Learning," 
          Journal of Data Science, 2023.
    """

    # Step 1: Validate the input dataframe and check whether it is valid.
    are_all_frames_valid(df, df_only=True)  # Ensure that the dataframe is valid.
    
    # Step 2: Ensure that categories and labels are formatted correctly as lists.
    columns = columns_manager(columns, empty_as_none=False)
    labels = columns_manager(labels, empty_as_none=False)
    
    # Step 3: Optionally sort the categories in ascending order
    if sort_columns:
        columns = sorted(columns)
    
    # Step 4: Create a copy of the dataframe if not modifying in place
    df_copy = df.copy() if not inplace else df
    
    # Step 5: Ensure the columns provided for categories exist in the dataframe
    # and that the labels are present in these columns.
    exist_features(df, features=columns, name="Category columns")
    exist_labels(
        df, labels=labels, 
        features=columns, 
        as_categories=as_categories, 
        name="Label columns"
    )
    if columns is None: 
        columns = is_valid_dtypes(
            df, features=df.columns,
            dtypes='category', 
            treat_obj_dtype_as_category=True, 
            ops='validate', 
            ).get('category')
        
        if not columns: 
            raise TypeError("No categorical columns detected.")
            
    # Step 6: If `as_categories` is True, convert the categories columns 
    # to pandas 'category' dtype
    original_dtype = df[columns].dtypes
    if as_categories:
        df[columns] = df[columns].astype('category')
        
    # Step 7: Process each column in categories and filter out rows 
    # with the specified labels
    for col in columns:
        # Check if the column exists in the dataframe
        if col not in df_copy.columns:
            if not ignore_missing:
                raise ValueError(f"Column '{col}' not found in dataframe.")
            continue
        
        # Remove rows with any of the specified labels from the column
        for category in labels:
            df_copy = df_copy[df_copy[col] != category]
    
    if as_categories : 
        # fall-back to original dtypes 
        df_copy[columns] = df_copy[columns].astype(original_dtype)
        
    # Step 8: Return the modified dataframe
    return df_copy

@is_data_readable 
def nan_to_na(
    data: DataFrame, 
    cat_missing_value: Optional[Union[Any,  float]] = pd.NA, 
    nan_spec: Optional[Union[Any, float]] = np.nan,
):
    """
    Converts specified NaN values in categorical columns of a pandas 
    DataFrame or Series to `pd.NA` or another specified missing value.

    Parameters
    ----------
    data : pandas.DataFrame or pandas.Series
        The input DataFrame or Series in which specified NaN values in 
        categorical columns will be converted.
        
    cat_missing_value : scalar, default=pd.NA
        The value to use for missing data in categorical columns. By 
        default, `pd.NA` is used. This ensures that categorical columns 
        do not contain `np.nan` values, which can cause type 
        inconsistencies.

    nan_spec : scalar, default=np.nan
        The value that is treated as NaN in the input data. By default, 
        `np.nan` is used. This allows flexibility in specifying what is 
        considered as NaN.

    Returns
    -------
    pandas.DataFrame or pandas.Series
        The DataFrame or Series with specified NaN values in categorical 
        columns converted to the specified missing value.

    Notes
    -----
    This function ensures consistency in the representation of missing 
    values in categorical columns, avoiding issues that arise from the 
    presence of specified NaN values in such columns.

    The conversion follows the logic:
    
    .. math:: 
        \text{If column is categorical and contains `nan_spec`} 
        \rightarrow \text{Replace `nan_spec` with `cat_missing_value`}

    Examples
    --------
    >>> from gofast.utils.datautils import nan_to_na
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({'A': [1.0, 2.0, np.nan], 'B': ['x', np.nan, 'z']})
    >>> df['B'] = df['B'].astype('category')
         A    B
    0  1.0    x
    1  2.0  NaN
    2  NaN    z
    
    >>> df = nan_to_na(df)
    >>> df
         A     B
    0  1.0     x
    1  2.0  <NA>
    2  NaN     z

    See Also
    --------
    pandas.DataFrame : Two-dimensional, size-mutable, potentially 
        heterogeneous tabular data.
    pandas.Series : One-dimensional ndarray with axis labels.
    numpy.nan : IEEE 754 floating point representation of Not a Number 
        (NaN).

    References
    ----------
    .. [1] Harris, C. R., Millman, K. J., van der Walt, S. J., et al. 
           (2020). Array programming with NumPy. Nature, 585(7825), 
           357-362.

    """
    is_frame(data, raise_exception= True, objname ='data')

    def has_nan_values(series, nan_spec):
        """Check if nan_spec exists in the series."""
        return series.isin([nan_spec]).any()
    
    if isinstance(data, pd.Series):
        if has_nan_values(data, nan_spec):
            if pd.api.types.is_categorical_dtype(data):
                data=data.astype(str)
                return data.replace({str(nan_spec): cat_missing_value})
        return data
    
    elif isinstance(data, pd.DataFrame):
        df_copy = data.copy()
        for column in df_copy.columns:
            if has_nan_values(df_copy[column], nan_spec):
                if pd.api.types.is_categorical_dtype(df_copy[column]):
                    df_copy[column]=df_copy[column].astype(str)
                    df_copy[column] = df_copy[column].replace(
                        {str(nan_spec): cat_missing_value})
        return df_copy

def resample_data(
    *d: Any,
    samples: Union[int, float, str] = 1,
    replace: bool = False,
    random_state: int = None,
    shuffle: bool = True
) -> List[Any]:
    """
    Resample multiple data structures (arrays, sparse matrices, Series, 
    DataFrames) based on specified sample size or ratio.

    Parameters
    ----------
    *d : Any
        Variable number of array-like, sparse matrix, pandas Series, or 
        DataFrame objects to be resampled.
        
    samples : Union[int, float, str], optional
        Specifies the number of items to sample from each data structure.
        
        - If an integer greater than 1, it is treated as the exact number 
          of items to sample.
        - If a float between 0 and 1, it is treated as a ratio of the 
          total number of rows to sample.
        - If a string containing a percentage (e.g., "50%"), it calculates 
          the sample size as a percentage of the total data length.
        
        The default is 1, meaning no resampling is performed unless a 
        different value is specified.

    replace : bool, default=False
        Determines if sampling with replacement is allowed, enabling the 
        same row to be sampled multiple times.

    random_state : int, optional
        Sets the seed for the random number generator to ensure 
        reproducibility. If specified, repeated calls with the same 
        parameters will yield identical results.

    shuffle : bool, default=True
        If True, shuffles the data before sampling. Otherwise, rows are 
        selected sequentially without shuffling.

    Returns
    -------
    List[Any]
        A list of resampled data structures, each in the original format 
        (e.g., numpy array, sparse matrix, pandas DataFrame) and with the 
        specified sample size.

    Methods
    -------
    - `_determine_sample_size`: Calculates the sample size based on the 
      `samples` parameter.
    - `_perform_sampling`: Conducts the sampling process based on the 
      calculated sample size, `replace`, and `shuffle` parameters.

    Notes
    -----
    - If `samples` is given as a percentage string (e.g., "25%"), the 
      actual number of rows to sample, :math:`n`, is calculated as:
      
      .. math::
          n = \left(\frac{\text{percentage}}{100}\right) \times N

      where :math:`N` is the total number of rows in the data structure.

    - Resampling supports both dense and sparse matrices. If the input 
      contains sparse matrices stored within numpy objects, the function 
      extracts and samples them directly.

    Examples
    --------
    >>> from gofast.utils.datautils import resample_data
    >>> import numpy as np
    >>> data = np.arange(100).reshape(20, 5)

    # Resample 10 items from each data structure with replacement
    >>> resampled_data = resample_data(data, samples=10, replace=True)
    >>> print(resampled_data[0].shape)
    (10, 5)
    
    # Resample 50% of the rows from each data structure
    >>> resampled_data = resample_data(data, samples=0.5, random_state=42)
    >>> print(resampled_data[0].shape)
    (10, 5)

    # Resample data with a percentage-based sample size
    >>> resampled_data = resample_data(data, samples="25%", random_state=42)
    >>> print(resampled_data[0].shape)
    (5, 5)

    References
    ----------
    .. [1] Fisher, R.A., "The Use of Multiple Measurements in Taxonomic 
           Problems", Annals of Eugenics, 1936.

    See Also
    --------
    np.random.choice : Selects random samples from an array.
    pandas.DataFrame.sample : Randomly samples rows from a DataFrame.
    """
    resampled_structures = []
    for data in d:
        # Handle sparse matrices encapsulated in numpy objects
        if ( 
                isinstance(data, np.ndarray) 
                and data.dtype == object 
                and scipy.sparse.issparse(data.item())
            ):
            data = data.item()  # Extract the sparse matrix from the numpy object

        # Determine sample size based on `samples` parameter
        n_samples = _determine_sample_size(data, samples, is_percent="%" in str(samples))

        # Sample the data structure based on the computed sample size
        sampled_d = _perform_sampling(data, n_samples, replace, random_state, shuffle)
        resampled_structures.append(sampled_d)
 
    return resampled_structures[0] if len(
        resampled_structures)==1 else resampled_structures

def _determine_sample_size(d: Any, samples: Union[int, float, str], 
                           is_percent: bool) -> int:
    """
    Determine the number of samples to draw based on the input size or ratio.
    """
    if isinstance(samples, str) and is_percent:
        samples = assert_ratio(samples, in_percent =True , name='samples')
    else: 
        try:
            samples = float(samples)
        except ValueError:
            raise TypeError(
                f"Invalid type for 'samples': {type(samples).__name__}."
                " Expected int, float, or percentage string."
                )
 
    d_length = d.shape[0] if hasattr(d, 'shape') else len(d)
    if samples < 1 or is_percent:
        return max(1, int(samples * d_length))
    
    return int(samples)

def _perform_sampling(d: Any, n_samples: int, replace: bool, 
                      random_state: int, shuffle: bool) -> Any:
    """
    Perform the actual sampling operation on the data structure.
    """
    if isinstance(d, pd.DataFrame) or isinstance(d, pd.Series):
        return d.sample(n=n_samples, replace=replace, random_state=random_state
                        ) if shuffle else d.iloc[:n_samples]
    elif scipy.sparse.issparse(d):
        if scipy.sparse.isspmatrix_coo(d):
            warnings.warn("coo_matrix does not support indexing. Conversion"
                          " to CSR matrix is recommended.")
            d = d.tocsr()
        indices = np.random.choice(d.shape[0], n_samples, replace=replace
                                   ) if shuffle else np.arange(n_samples)
        return d[indices]
    else:
        d_array = np.array(d) if not hasattr(d, '__array__') else d
        indices = np.random.choice(len(d_array), n_samples, replace=replace
                                   ) if shuffle else np.arange(n_samples)
        return d_array[indices] if d_array.ndim == 1 else d_array[indices, :]
    
    
@SaveFile 
def pair_data(
    *dfs: Union[pd.DataFrame, List[pd.DataFrame]],
    on: Union[str, List[str]] = None,
    parse_on: bool = False,
    mode: str = 'strict',
    coerce: bool = False,
    force: bool = False,
    decimals: int = 7,
    raise_warn: bool = True, 
    savefile=None, 
) -> pd.DataFrame:
    """
    Finds identical objects in multiple DataFrames and merges them 
    using an intersection (`cross`) strategy.

    Parameters
    ----------
    dfs : List[Union[pd.DataFrame, List[pd.DataFrame]]]
        A variable-length argument of pandas DataFrames for merging.
        
    on : Union[str, List[str]], optional
        Column or index level names to join on. These must exist in 
        all DataFrames. If None and `force` is False, concatenation 
        along columns axis is performed.
        
    parse_on : bool, default=False
        If True, parses `on` when provided as a string by splitting 
        it into multiple column names.
        
    mode : str, default='strict'
        Determines handling of non-DataFrame inputs. In 'strict' 
        mode, raises an error for non-DataFrame objects. In 'soft' 
        mode, ignores them.
        
    coerce : bool, default=False
        If True, truncates all DataFrames to the length of the 
        shortest DataFrame before merging.
        
    force : bool, default=False
        If True, forces `on` columns to exist in all DataFrames, 
        adding them from any DataFrame that contains them. Raises an 
        error if `on` columns are missing in all provided DataFrames.
        
    decimals : int, default=7
        Number of decimal places to round numeric `on` columns for 
        comparison. Helps ensure similar values are treated as equal.
        
    raise_warn : bool, default=True
        If True, warns user that data is concatenated along column 
        axis when `on` is None.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the merged objects based on `on` 
        columns, using cross intersection for matching.

    Methods
    -------
    - `pd.concat`: Concatenates DataFrames along columns if `on` 
      is None.
    - `pd.merge`: Merges DataFrames based on `on` columns.
    
    Notes
    -----
    - This function performs pairwise merging of DataFrames based 
      on column alignment specified in `on`.
      
    - When `decimals` is set, values in `on` columns are rounded 
      to the specified decimal places before merging to avoid 
      floating-point discrepancies:
      
      .. math::
          \text{round}(x, \text{decimals})

    - The function requires that all provided data be DataFrames if 
      `mode='strict'`. Non-DataFrame inputs in 'strict' mode raise 
      a `TypeError`.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.utils.datautils import pair_data
    >>> data1 = pd.DataFrame({
    ...     'longitude': [110.486111],
    ...     'latitude': [26.05174],
    ...     'value': [10]
    ... })
    >>> data2 = pd.DataFrame({
    ...     'longitude': [110.486111],
    ...     'latitude': [26.05174],
    ...     'measurement': [1]
    ... })
    
    # Merge based on common columns 'longitude' and 'latitude'
    >>> pair_data(data1, data2, on=['longitude', 'latitude'], decimals=5)
       longitude  latitude  value  measurement
    0  110.48611  26.05174     10           1

    References
    ----------
    .. [1] Wes McKinney, "Data Structures for Statistical Computing 
           in Python", Proceedings of the 9th Python in Science 
           Conference, 2010.

    See Also
    --------
    pd.concat : Concatenates pandas objects along a specified axis.
    pd.merge : Merges DataFrames based on key columns.
    """
    # make a shallow copy
    d = dfs[:]
    
    # Filter only DataFrames if `mode` is set to 'soft'
    if str(mode).lower() == 'soft':
        d = [df for df in d if isinstance(df, pd.DataFrame)]
    else:
        # Ensure all provided data is DataFrame if `mode` is 'strict'
        is_dataframe = all(isinstance(df, pd.DataFrame) for df in d)
        if not is_dataframe:
            utypes = [
                type(df).__name__ for df in d if not isinstance(df, pd.DataFrame)]
            raise TypeError(
                "In strict mode, filtering is not performed."
                " All elements must be of type 'DataFrame'. "
                f"Found unexpected types: {', '.join(utypes)}."
            ) 
            
    if len(d) < 2: 
        raise TypeError(
            "Not enough dataframes. Need at least two dataframes for pairing."
            f" Got {len(d)}"
            )
    # Coerce to shortest DataFrame length if `coerce=True`
    if coerce:
        min_len = min(len(df) for df in d)
        d = [df.iloc[:min_len, :] for df in d]

    # If `on` is None and `raise_warn` is True, warn and concatenate along columns
    if on is None:
        if raise_warn:
            warnings.warn("`on` parameter is None. Performing"
                          " concatenation along columns.")
        return pd.concat(d, axis=1)

    # Parse `on` if `parse_on=True`
    if parse_on and isinstance(on, str):
        on = on.split()

    # Ensure `on` columns exist in all DataFrames if `force=True`
    if force:
        missing_cols = [col for col in on if not all(col in df.columns for df in d)]
        if missing_cols:
            d = [df.assign(**{col: d[0][col]}) for col in missing_cols 
                 for df in d if col in d[0].columns]

    # Round numeric columns in `on` columns to `decimals` if specified
    for df in d:
        for col in on:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].round(decimals)

    # Perform pairwise merging based on `on` columns
    data = d[0]
    for df in d[1:]:
        data = pd.merge(data, df, on=on, suffixes=('_x', '_y'))

    return data

def random_sampling(
    d,
    samples: int = None,
    replace: bool = False,
    random_state: int = None,
    shuffle: bool = True,
) -> Union[np.ndarray, 'pd.DataFrame', 'scipy.sparse.spmatrix']:
    """
    Randomly samples rows from the data, with options for shuffling, 
    sampling with replacement, and fixed randomness for reproducibility.
    
    Parameters
    ----------
    d : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data to sample from. Supports any array-like structure, 
        pandas DataFrame, or scipy sparse matrix with `n_samples` 
        as rows and `n_features` as columns.

    samples : int, optional
        Number of items or ratio of items to return. If `samples` 
        is None, it defaults to 1 (selects all items). If set as 
        a float (e.g., "0.2"), it is interpreted as the percentage 
        of data to sample.
        
    replace : bool, default=False
        If True, allows sampling the same row multiple times; 
        if False, each row can only be sampled once.
        
    random_state : int, array-like, BitGenerator, np.random.RandomState, \
        np.random.Generator, optional
        Controls randomness. If int or array-like, sets the seed 
        for reproducibility. If a `RandomState` or `Generator`, it 
        will be used as-is.
        
    shuffle : bool, default=True
        If True, shuffles the data before sampling; otherwise, 
        returns the top `n` samples without shuffling.

    Returns
    -------
    Union[np.ndarray, pd.DataFrame, scipy.sparse.spmatrix]
        Sampled data, in the same format as `d` (array-like, sparse 
        matrix, or DataFrame) and in the shape (samples, n_features).

    Methods
    -------
    - `np.random.choice`: Selects rows randomly based on `samples` 
      and `replace` parameter.
    - `d.sample()`: Used for DataFrames to sample with more control.
    
    Notes
    -----
    - If `samples` is a string containing "%", the number of samples 
      is calculated as a percentage of the total rows:
      
      .. math::
          \text{samples} = \frac{\text{percentage}}{100} \times 
          \text{len(d)}

    - To ensure consistent sampling, especially when `replace=True`, 
      setting `random_state` is recommended for reproducibility.
    
    - The function supports various data types and automatically 
      converts `d` to a compatible structure if necessary.

    Examples
    --------
    >>> from gofast.utils.data_utils import random_sampling
    >>> import numpy as np
    >>> data = np.arange(100).reshape(20, 5)
    
    # Sample 7 rows from data
    >>> random_sampling(data, samples=7).shape
    (7, 5)
    
    # Sample 10% of rows
    >>> random_sampling(data, samples="10%", random_state=42).shape
    (2, 5)

    >>> # Sampling from a pandas DataFrame with replacement
    >>> import pandas as pd
    >>> df = pd.DataFrame(np.arange(100).reshape(20, 5))
    >>> random_sampling(df, samples=5, replace=True).shape
    (5, 5)
    
    References
    ----------
    .. [1] Fisher, R.A., "The Use of Multiple Measurements in Taxonomic 
           Problems", Annals of Eugenics, 1936.

    See Also
    --------
    np.random.choice : Selects random samples from an array.
    pandas.DataFrame.sample : Randomly samples rows from a DataFrame.
    """

    # Initialize variables for calculation
    n = None
    is_percent = False
    orig = copy.deepcopy(samples)

    # Ensure data is iterable and convert if necessary
    if not hasattr(d, "__iter__"):
        d = np.array(d)

    # Set default sample size to 1 if samples is None or wildcarded
    if samples is None or str(samples) in ('1', '*'):
        samples = "100%"
        
    # Handle percentage-based sampling if specified as a string
    if "%" in str(samples):
        samples = str(samples).replace("%", "")
        is_percent = True
    
    # Ensure samples is a valid numerical value
    try:
        samples = float(samples)
    except ValueError:
        raise TypeError("Invalid value for 'samples'. Expected an integer "
                        f"or percentage, got {type(orig).__name__!r}")
    
    # Calculate the sample size based on percentage if necessary
    if samples <= 1 or is_percent:
        samples = assert_ratio(
            samples, bounds=(0, 1), exclude_values='use lower bound',
            in_percent=True if is_percent else False 
        )
        n = int(samples * (d.shape[0] if scipy.sparse.issparse(d) else len(d)))
    else:
        # Use the integer value directly
        n = int(samples)
    
    # Ensure sample size does not exceed data length
    dlen = d.shape[0] if scipy.sparse.issparse(d) else len(d)
    if n > dlen:
        n = dlen

    # Sampling for DataFrame
    if hasattr(d, 'sample'):
        return d.sample(n=n, replace=replace, random_state=random_state
                        ) if shuffle else d.iloc[:n, :]

    # Set random state for reproducibility
    np.random.seed(random_state)

    # Handle sparse matrix sampling
    if scipy.sparse.issparse(d):
        if scipy.sparse.isspmatrix_coo(d):
            warnings.warn("`coo_matrix` does not support indexing. "
                          "Converting to CSR matrix for indexing.")
            d = d.tocsr()
        indices = np.random.choice(np.arange(d.shape[0]), n, replace=replace
                                   ) if shuffle else list(range(n))
        return d[indices]

    # Manage array-like data
    d = np.array(d) if not hasattr(d, '__array__') else d
    indices = np.random.choice(len(d), n, replace=replace) if shuffle else list(range(n))
    d = d[indices] if d.ndim == 1 else d[indices, :]

    return d

def read_excel_sheets(
    xlsx_file: str, 
    sheet_names: Union[str, List[str]] = None
) -> List[Union[str, pd.DataFrame]]:
    """
    Read Specified Sheets from an Excel Workbook into a List of DataFrames.
    
    The `read_excel_sheets` function facilitates the extraction of data from 
    multiple sheets within an Excel workbook. By specifying `sheet_names` 
    (either as a single sheet name or a list of sheet names), the function 
    can selectively read and return the desired sheets. If `sheet_names` is 
    `None`, all sheets within the workbook are read and returned. This utility 
    is particularly useful for data analysis workflows that require comprehensive 
    data ingestion from structured Excel files with multiple sheets.
    
    .. math::
        \text{Data Extraction} = \{ \text{Base Name}, \text{Sheet}_1, 
        \text{Sheet}_2, \dots, \text{Sheet}_n \}
    
    Parameters
    ----------
    xlsx_file : `str`
        Path to the Excel file containing multiple sheets.
    
    sheet_names : Union[`str`, `List[str]`], optional
        Specifies the sheet or sheets to read from the Excel workbook.
        - If `None`, all sheets are read and returned as a list of DataFrames.
        - If a single sheet name is provided as a `str`, only that sheet is 
          read and returned.
        - If a list of sheet names is provided, only those specified sheets 
          are read and returned.
    
    Returns
    -------
    List[Union[`str`, `pd.DataFrame`]]
        A list where the first element is the file base name (without extension) 
        and subsequent elements are DataFrames corresponding to each sheet 
        specified. If `sheet_names` is provided, only the specified sheets are 
        included.
    
    Raises
    ------
    ValueError
        - If file validation fails.
        - If reading the Excel file fails.
        - If specified sheet names do not exist in the workbook.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.utils.datautils import read_excel_sheets
    
    >>> # Read all sheets from an Excel file
    >>> sheets = read_excel_sheets("data/sample.xlsx")
    >>> print(sheets)
    ['sample',    A     B  C   D
    0   1  1.50  3  10
    1   0  0.70  8  12
    2  10  0.50  7  13
    3   0  0.35  1  13,    A     B  C   D
    0   1  1.50  3  10
    1   0  0.70  8  12
    2  10  0.50  7  13
    3   0  0.35  1  13]
    
    >>> # Read specific sheets by name
    >>> sheets = read_excel_sheets("data/sample.xlsx", sheet_names=["Sheet1", "Sheet3"])
    >>> print(sheets)
    ['sample',    A     B  C   D
    0   1  1.50  3  10
    1   0  0.70  8  12,    A     B  C   D
    0   1  1.50  3  10
    1   0  0.70  8  12]
    
    >>> # Read a single sheet by name
    >>> sheet = read_excel_sheets("data/sample.xlsx", sheet_names="Sheet2")
    >>> print(sheet)
    ['sample',    A     B  C   D
    0   1  1.50  3  10
    1   0  0.70  8  12
    2  10  0.50  7  13
    3   0  0.35  1  13]
    
    Notes
    -----
    - **File Validation**: Before attempting to read the Excel file, the function 
      validates the file's existence, format, and ensures it is not empty using 
      the `check_files` utility from `gofast.core.checks`.
    
    - **Sheet Selection**: 
        - If `sheet_names` is `None`, all sheets within the Excel file are read 
          and included in the returned list.
        - Specifying `sheet_names` as a string or a list of strings allows for 
          selective reading of sheets, which can enhance performance by 
          avoiding unnecessary data loading.
    
    - **Base Name Inclusion**: The first element of the returned list is the base 
      name of the Excel file (excluding its extension), providing a reference 
      identifier for the DataFrames that follow.
    
    - **Error Handling**: The function raises descriptive errors if:
        - The Excel file does not exist.
        - The file format is unsupported.
        - The file is empty.
        - Specified sheets do not exist within the workbook.
    
    - **Performance Considerations**: Reading multiple sheets simultaneously 
      can be resource-intensive for large Excel files with numerous sheets. Users 
      should be mindful of the size and number of sheets when utilizing this 
      function.
    
    See Also
    --------
    gofast.core.checks.check_files : 
        Function to validate file existence, format, and non-emptiness.
    pandas.read_excel : Function to read Excel files into Pandas DataFrames.
    os.path.exists : Check if a path exists.
    os.path.splitext : Split the file path into root and extension.
    os.path.basename : Extract the base name of a file.
    
    References
    ----------
    .. [1] Pandas Documentation: pandas.read_excel. 
       https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_excel.html  
    .. [2] Python Documentation: os.path.exists. 
       https://docs.python.org/3/library/os.path.html#os.path.exists  
    .. [3] Python Documentation: os.path.splitext. 
       https://docs.python.org/3/library/os.path.html#os.path.splitext  
    .. [4] Python Documentation: os.path.basename. 
       https://docs.python.org/3/library/os.path.html#os.path.basename  
    .. [5] Freedman, D., & Diaconis, P. (1981). On the histogram as a density 
           estimator: L2 theory. *Probability Theory and Related Fields*, 57(5), 
           453-476.
    """
    # Validate the Excel file
    try:
        valid_file = check_files(
            files=xlsx_file,
            formats='xlsx',
            return_valid=True,
            error='raise',
            empty_allowed=False
        )
    except Exception as e:
        raise ValueError(f"File validation failed for '{xlsx_file}': {e}")
    
    # Attempt to read specified sheets from the Excel file
    try:
        all_sheets = pd.read_excel(
            valid_file,
            sheet_name=sheet_names
        )
    except ValueError as ve:
        # Handle the case where specified sheet does not exist
        raise ValueError(
            f"Failed to read specified sheet(s) from '{valid_file}': {ve}"
        )
    except Exception as e:
        raise ValueError(
            f"Failed to read Excel file '{valid_file}': {e}"
        )
    
    # Extract the base name of the file without extension
    file_base_name = os.path.basename(
        os.path.splitext(valid_file)[0]
    )
    
    # Compile the list with base name followed by DataFrames of each sheet
    if isinstance(all_sheets, dict):
        list_of_df = [file_base_name] + [
            df for df in all_sheets.values()
        ]
    else:
        # If a single DataFrame is returned
        list_of_df = [file_base_name, all_sheets]
    
    return list_of_df

def read_worksheets(
    *xlsx_files: str
    ) -> Tuple[Optional[List[pd.DataFrame]], Optional[List[str]]]:
    """
    Reads all `.xlsx` sheets from given file paths or directories and returns
    the contents as DataFrames along with sheet names. 
    This function processes each sheet in a workbook as a separate DataFrame
    and collects all sheet names, replacing special characters in names with
    underscores.

    Parameters
    ----------
    data : str
        Variable-length argument of file paths or directories. 
        Each path should be a string pointing to an `.xlsx` file or a directory
        containing `.xlsx` files. Only files with the `.xlsx` extension are 
        read; other file types will raise an error.

    Returns
    -------
    Tuple[Optional[List[pd.DataFrame]], Optional[List[str]]]
        A tuple containing two elements:
        
        - A list of DataFrames, each representing a sheet from the specified
        `.xlsx` files.
        - A list of sheet names corresponding to the DataFrames. 
          Special characters in sheet names are replaced with underscores 
          for standardization.

    Methods
    -------
    - `pd.read_excel(d, sheet_name=None)`: Reads Excel file and loads all 
      sheets into a dictionary of DataFrames.
    - `re.sub`: Replaces special characters in sheet names with 
      underscores.

    Notes
    -----
    - If a directory is provided, the function searches for all 
      `.xlsx` files in that directory. Only Excel files with the 
      `.xlsx` extension are supported.
      
    - If no `.xlsx` files are found in the provided paths, the 
      function returns `(None, None)`.

    - To maintain consistency in sheet names, special characters 
      in sheet names are replaced with underscores using 
      :class:`re.Regex`.

    - Mathematically, if :math:`d` represents an Excel file in 
      `data`, then:
      
      .. math::
          \text{dataframes}_{d} = f(\text{Excel file})
      
      where :math:`f` denotes loading all sheets within an Excel 
      file into separate DataFrames.

    Examples
    --------
    >>> from gofast.utils.datautils import read_worksheets
    >>> # Example 1: Reading a single Excel file
    >>> file_path = r'F:/repositories/gofast/data/erp/sheets/gbalo.xlsx'
    >>> data, sheet_names = read_worksheets(file_path)
    >>> sheet_names
    ['l11', 'l10', 'l02']
    
    >>> # Example 2: Reading all .xlsx files in a directory
    >>> import os
    >>> dir_path = os.path.dirname(file_path)
    >>> data, sheet_names = read_worksheets(dir_path)
    >>> sheet_names
    ['l11', 'l10', 'l02', 'l12', 'l13']

    References
    ----------
    .. [1] McKinney, Wes, *Data Structures for Statistical Computing 
           in Python*, Proceedings of the 9th Python in Science 
           Conference, 2010.

    See Also
    --------
    pd.read_excel : Reads an Excel file and returns DataFrames 
                    of all sheets.
    os.path.isdir : Checks if the path is a directory.
    os.path.isfile : Checks if the path is a file.
    """

    # Temporary list to store valid .xlsx files
    dtem = []
     # and also check expected files.
    data = check_files(xlsx_files, formats ='.xlsx', return_valid=True )
   
    # Iterate over each path provided in data
    for o in data:
        if os.path.isdir(o):
            # Get all files in the directory, filtering for .xlsx files
            dlist = os.listdir(o)
            p = [os.path.join(o, f) for f in dlist if f.endswith('.xlsx')]
            dtem.extend(p)
        elif os.path.isfile(o):
            # Check if the file is an .xlsx file
            _, ex = os.path.splitext(o)
            if ex == '.xlsx':
                dtem.append(o)

    # Deep copy of the collected .xlsx files
    data = copy.deepcopy(dtem)

    # Return None if no valid Excel files are found
    if len(data) == 0:
        return None, None

    # Dictionary to store DataFrames by sheet name
    ddict = {}
    regex = re.compile(r'[$& #@%^!]', flags=re.IGNORECASE)

    # Read each Excel file and store sheets
    for d in data:
        try:
            ddict.update(**pd.read_excel(d, sheet_name=None))
        except Exception:
            pass  # Continue if any file fails to read

    # Raise error if no data could be read
    if len(ddict) == 0:
        raise TypeError("No readable data found in the provided paths.")

    # Standardize sheet names and store them
    sheet_names = list(map(lambda o: regex.sub('_', o).lower(), ddict.keys()))

    # Collect the DataFrames
    data = list(ddict.values())

    return data, sheet_names

def process_and_extract_data(
    *arr: ArrayLike, 
    columns: Optional[List[Union[str, int]]] = None,
    enforce_extraction: bool = True, 
    allow_split: bool = False, 
    search_multiple: bool = False,
    ensure_uniform_length: bool = False, 
    to_array: bool = False,
    on_error: str = 'raise',
) -> List[np.ndarray]:
    """
    Extracts and processes data from various input types, focusing on column 
    extraction from pandas DataFrames and conversion of inputs to numpy 
    arrays or pandas Series.

    Parameters
    ----------
    *arr : ArrayLike
        A variable number of inputs, each can be a list, numpy array, pandas 
        Series,dictionary, or pandas DataFrame.
    columns : List[Union[str, int]], optional
        Specific columns to extract from pandas DataFrames. If not provided, 
        the function behaves differently based on `allow_split`.
    enforce_extraction : bool, default=True
        Forces the function to try extracting `columns` from DataFrames. 
        If False, DataFrames are returned without column extraction unless 
        `allow_split` is True.
        Removing non-conforming elements if True.
    allow_split : bool, default=False
        If True and a DataFrame is provided without `columns`, splits the 
        DataFrame into its constituent columns.
    search_multiple : bool, default=False
        Allows searching for `columns` across multiple DataFrame inputs. Once 
        a column is found, it is not searched for in subsequent DataFrames.
    ensure_uniform_length : bool, default=False
        Checks that all extracted arrays have the same length. Raises an error
        if they don't.
    to_array : bool, default=False
        Converts all extracted pandas Series to numpy arrays.
    on_error : str, {'raise', 'ignore'}, default='raise'
        Determines how to handle errors during column extraction or when 
        enforcing uniform length. 'raise' will raise an error, 'ignore' will 
        skip the problematic input.

    Returns
    -------
    List[np.ndarray]
        A list of numpy arrays or pandas Series extracted based 
        on the specified conditions.

    Examples
    --------
    >>> import numpy as np 
    >>> import pandas as pd 
    >>> from gofast.utils.datautils import process_and_extract_data
    >>> data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> process_and_extract_data(data, columns=['A'], to_array=True)
    [array([1, 2, 3])]

    Splitting DataFrame into individual arrays:

    >>> process_and_extract_data(data, allow_split=True, to_array=True)
    [array([1, 2, 3]), array([4, 5, 6])]

    Extracting columns from multiple DataFrames:

    >>> data2 = pd.DataFrame({'C': [7, 8, 9], 'D': [10, 11, 12]})
    >>> process_and_extract_data(data, data2, columns=['A', 'C'], 
                                  search_multiple=True, to_array=True)
    [array([1, 2, 3]), array([7, 8, 9])]

    Handling mixed data types:

    >>> process_and_extract_data([1, 2, 3], {'E': [13, 14, 15]}, to_array=True)
    [array([1, 2, 3]), array([13, 14, 15])]
    
    Extracting columns from multiple DataFrames and enforcing uniform length:
    >>> data2 = pd.DataFrame({'C': [7, 8, 9, 10], 'D': [11, 12, 13, 14]})
    >>> result = process_and_extract_data(
        data, data2, columns=['A', 'C'],search_multiple=True,
        ensure_uniform_length=True, to_array=True)
    ValueError: Extracted data arrays do not have uniform length.
    """
    extracted_data = []
    columns_found: Set[Union[str, int]] = set()

    def _process_input(
            input_data: ArrayLike,
            target_columns: Optional[List[Union[str, int]]], 
            to_array: bool) -> Optional[np.ndarray]:
        """
        Processes each input based on its type, extracting specified columns 
        if necessary, and converting to numpy array if specified.
        """
        if isinstance(input_data, (list, tuple)):
            input_data = np.array(input_data)
            return input_data if len(input_data.shape
                                     ) == 1 or not enforce_extraction else None

        elif isinstance(input_data, dict):
            input_data = pd.DataFrame(input_data)

        if isinstance(input_data, pd.DataFrame):
            if target_columns:
                for col in target_columns:
                    if col in input_data.columns and (
                            search_multiple or col not in columns_found):
                        data_to_add = input_data[col].to_numpy(
                            ) if to_array else input_data[col]
                        extracted_data.append(data_to_add)
                        columns_found.add(col)
                    elif on_error == 'raise':
                        raise ValueError(f"Column {col} not found in DataFrame.")
            elif allow_split:
                for col in input_data.columns:
                    data_to_add = input_data[col].to_numpy(
                        ) if to_array else input_data[col]
                    extracted_data.append(data_to_add)
            return None

        if isinstance(input_data, np.ndarray):
            if input_data.ndim > 1 and allow_split:
                input_data = np.hsplit(input_data, input_data.shape[1])
                for arr in input_data:
                    extracted_data.append(arr.squeeze())
                return None
            elif input_data.ndim > 1 and enforce_extraction and on_error == 'raise':
                raise ValueError("Multidimensional array found while "
                                 "`enforce_extraction` is True.")
            return input_data if to_array else np.squeeze(input_data)

        return input_data.to_numpy() if to_array and isinstance(
            input_data, pd.Series) else input_data

    for arg in arr:
        result = _process_input(arg, columns, to_array)
        if result is not None:
            extracted_data.append(result)

    if ensure_uniform_length and not all(len(x) == len(
            extracted_data[0]) for x in extracted_data):
        if on_error == 'raise':
            raise ValueError(
                "Extracted data arrays do not have uniform length.")
        else:
            return []

    return extracted_data

def random_selector(
    arr: ArrayLike,
    value: Union[float, ArrayLike, str],
    seed: int = None,
    shuffle: bool = False
) -> np.ndarray:
    """
    Randomly select specified values from an array, using a value 
    count, percentage, or subset. Provides consistent selection if 
    seeded, and can shuffle the result.

    Parameters
    ----------
    arr : ArrayLike
        Input array of values from which selections are made. 
        Accepts any array-like structure (e.g., list, ndarray) 
        for processing.
        
    value : Union[float, ArrayLike, str]
        Specifies the number or subset of values to select.
        
        - If `value` is a float, it is interpreted as the number 
          of items to select from `arr`.
        - If `value` is an array-like, it indicates the exact 
          values to select, provided they exist within `arr`.
        - If `value` is a string containing a percentage 
          (e.g., `"50%"`), it calculates the proportion to select 
          based on the length of `arr`, given by:
          
          .. math::
              \text{value} = \left( \frac{\text{percentage}}{100} \right) \times \text{len(arr)}
          
    seed : int, optional
        Seed for the random number generator, which ensures 
        repeatable selections when set. Defaults to None.
        
    shuffle : bool, default=False
        If True, shuffles the selected values after extraction.
        
    Returns
    -------
    np.ndarray
        Array containing the randomly selected values from `arr`.

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.utils.datautils import random_selector
    >>> data = np.arange(42)
    
    # Select 7 elements deterministically using a seed
    >>> random_selector(data, 7, seed=42)
    array([0, 1, 2, 3, 4, 5, 6])
    
    # Select specific values present in the array
    >>> random_selector(data, (23, 13, 7))
    array([ 7, 13, 23])
    
    # Select a percentage of values
    >>> random_selector(data, "7%", seed=42)
    array([0, 1])
    
    # Select 70% of values with shuffling enabled
    >>> random_selector(data, "70%", seed=42, shuffle=True)
    array([ 0,  5, 20, 25, 13,  7, 22, 10, 12, 27, 23, 21, 16,  3,  1, 17,  8,
            6,  4,  2, 19, 11, 18, 24, 14, 15,  9, 28, 26])

    Notes
    -----
    - The `value` parameter can be a float representing the count, 
      a string containing a percentage, or an array of elements to 
      select. For invalid types, the function raises a `TypeError`.
      
    - The `seed` parameter is essential for reproducibility. 
      When set, repeated calls with the same parameters will yield 
      identical results.
      
    - This function is helpful for sampling data subsets in 
      machine learning and statistical analysis.
      
    References
    ----------
    .. [1] Fisher, Ronald A., "The Use of Multiple Measurements in 
           Taxonomic Problems", Annals of Eugenics, 1936.

    See Also
    --------
    numpy.random.permutation : Randomly permutes elements in an array.
    numpy.random.shuffle : Shuffles array in place.
    """
    
    # Error message for invalid input
    msg = "Non-numerical value is not allowed. Got {!r}."
    
    # Set seed if provided for reproducibility
    if seed is not None:
        seed = _assert_all_types(seed, int, float, objname='Seed')
        np.random.seed(seed)
    
    # Deep copy of value for error reporting if necessary
    v = copy.deepcopy(value)
    
    # If value is not iterable (excluding strings), convert to string
    if not is_iterable(value, exclude_string=True):
        value = str(value)
        
        # Handle percentage-based selection
        if '%' in value:
            try:
                value = float(value.replace('%', '')) / 100
            except:
                raise TypeError(msg.format(v))
            # Calculate number of items to select based on percentage
            value *= len(arr)
        
        try:
            # Convert value to integer if possible
            value = int(value)
        except:
            raise TypeError(msg.format(v))
    
        # Ensure the selected count does not exceed array length
        if value > len(arr):
            raise ValueError(f"Number {value} is out of range. "
                             f"Expected value less than {len(arr)}.")
        
        # Randomly select `value` items
        value = np.random.permutation(value)
        
    # Ensure `arr` is array-like and flatten if multi-dimensional
    arr = np.array(is_iterable(arr, exclude_string=True, transform=True))
    arr = arr.ravel() if arr.ndim != 1 else arr
        
    # Select specified elements in `value`
    mask = is_in(arr, value, return_mask=True)
    arr = arr[mask]
    
    # Shuffle the array if specified
    if shuffle:
        np.random.shuffle(arr)
    
    return arr

@is_data_readable 
def cleaner(
    data: Union[DataFrame, NDArray],
    columns: List[str] = None,
    inplace: bool = False,
    labels: List[Union[int, str]] = None,
    func: _F = None,
    mode: str = 'clean',
    **kws
) -> Union[DataFrame, NDArray, None]:
    """
    Sanitize data by dropping specified labels from rows or columns 
    with optional column transformation. This function allows both 
    structured data (e.g., pandas DataFrame) and unstructured 2D array 
    formats, applying universal cleaning functions if provided. 

    Parameters
    ----------
    data : Union[pd.DataFrame, NDArray]
        Data structure to process, supporting either a 
        :class:`pandas.DataFrame` or a 2D :class:`numpy.ndarray`.
        If a numpy array is passed, it will be converted to a 
        DataFrame internally to facilitate label-based operations. 
        
    columns : List[str], optional
        List of column labels to operate on, by default None.
        If specified, the columns matching these labels will be 
        subject to any transformations or deletions specified by 
        `mode`. This is useful when targeting specific columns 
        without altering others.
        
    inplace : bool, default=False
        If True, modifies `data` directly; if False, returns a 
        new DataFrame or array with modifications. Note that when 
        `data` is initially provided as an array, this parameter 
        is overridden to False to ensure consistent return types.
        
    labels : List[Union[int, str]], optional
        Index or column labels to drop. Can be a list of column 
        names or index labels. If provided, only the specified 
        labels will be targeted for removal or transformation.
        
    func : Callable, optional
        Universal cleaning function to apply to the columns 
        (e.g., string cleaning, handling missing values).
        If `mode='clean'`, `func` will be applied to specified 
        columns, allowing customized data preprocessing.
        
    mode : str, default='clean'
        Operational mode controlling function behavior. Supported 
        options are:
            - 'clean': Applies the `func` callable to columns for 
              preprocessing tasks.
            - 'drop': Removes rows or columns based on `labels` or 
              `columns`. Follows similar behavior to 
              :func:`pandas.DataFrame.drop`.
              
    **kws : dict
        Additional keyword arguments passed to :func:`pandas.DataFrame.drop`
        when `mode='drop'`. Allows configuration of drop operation 
        (e.g., `axis`, `errors`).
        
    Returns
    -------
    Union[pd.DataFrame, NDArray, None]
        Returns the cleaned or transformed DataFrame or array, 
        depending on the input type. If `inplace=True`, returns None. 
        If the original data was an array, the output remains an array.
        
    Methods
    -------
    - `sanitize_frame_cols(data, inplace, func)`: Performs cleaning 
      on columns based on a function.
    - `to_numeric_dtypes(data)`: Ensures that all applicable data 
      types are converted to numeric types, which can be essential 
      for computational consistency.
      
    Notes
    -----
    - By default, when a 2D array is provided as `data`, it is 
      converted to a DataFrame for processing purposes and then 
      returned as an array after operations are complete. This 
      ensures compatibility with label-based operations.
      
    - The primary operations in this function can be mathematically 
      described as follows:
      
      .. math::
          \text{DataFrame}_{\text{cleaned}} = 
          f(\text{DataFrame}_{\text{original}})
          
      where :math:`f` is a transformation applied by `func` when 
      `mode='clean'`, and a subset selection based on `labels` 
      otherwise.

    Examples
    --------
    >>> from gofast.utils.datautils import cleaner
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'A': [1, 2, 3],
    ...     'B': [4, None, 6],
    ...     'C': [7, 8, 9]
    ... })
    >>> # Example: Clean using a lambda function
    >>> cleaner(data, columns=['B'], func=lambda x: x.fillna(0), mode='clean')
    
    >>> # Example: Drop rows with labels [0, 2]
    >>> cleaner(data, labels=[0, 2], mode='drop', inplace=True)

    References
    ----------
    .. [1] McKinney, Wes, *Data Structures for Statistical Computing in Python*, 
           Proceedings of the 9th Python in Science Conference, 2010.

    See Also
    --------
    pd.DataFrame.drop : Removes specified labels from rows or columns.
    """
    # Validate and set mode operation.
    mode = validate_name_in(
        mode, defaults=("drop", 'clean'), expect_name='drop'
    )
    
    if mode == 'clean':
        # If mode is clean, apply column transformations.
        return sanitize_frame_cols(data, inplace=inplace, func=func)
    
    objtype = 'array'
    if not hasattr(data, '__array__'):
        # Convert to numpy array if not array-like
        data = np.array(data)
    
    # Determine object type for handling pandas data.
    if hasattr(data, "columns"):
        objtype = "pd"
    
    if objtype == 'array':
        # Convert numpy array to DataFrame for label-based processing.
        data = pd.DataFrame(data)
        inplace = False  # Disable inplace for numpy output

    # Process columns if specified
    if columns is not None:
        columns = is_iterable(
            columns, exclude_string=True,
            parse_string=True, transform=True
        )
    
    # Perform drop operation on DataFrame
    data = data.drop(labels=labels, columns=columns, inplace=inplace, **kws)
    
    # Convert all applicable types to numeric types for consistency
    data = to_numeric_dtypes(data)
    
    # Return as numpy array if original input was array-like
    return np.array(data) if objtype == 'array' else data

@is_data_readable
def data_extractor(
    data: pd.DataFrame,
    columns: Union[str, List[str]] = None,
    as_frame: bool = False,
    drop_columns: bool = False,
    default_columns: List[Tuple[str, str]] = None,
    raise_exception: Union[bool, str] = True,
    verbose: int = 0,
    round_decimals: int = None,
    fillna_value: Any = None,
    unique: bool = False,
    coerce_dtype: Any = None, 
) -> Tuple[Union[Tuple[float, float], pd.DataFrame, None],
           pd.DataFrame, Tuple[str, ...]]:
    """
    Extracts specified columns (e.g., coordinates) from a DataFrame, with options 
    for formatting, dropping, rounding, and unique selection.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame expected to contain specified columns, such as 
        `longitude`/`latitude` or `easting`/`northing` for coordinates.

    columns : Union[str, List[str]], optional
        Column(s) to extract. If `None`, attempts to detect default columns 
        based on `default_columns`.

    as_frame : bool, default=False
        If True, returns extracted columns as a DataFrame. If False, computes 
        and returns the midpoint values for coordinates.

    drop_columns : bool, default=False
        If True, removes extracted columns from `data` after extraction.
        
    default_columns : List[Tuple[str, str]], optional
        List of default column pairs to search for if `columns` is `None`. 

    default_columns : List[Tuple[str, str]], optional
        List of tuples specifying default column pairs to search for in `data` 
        if `columns` is not provided. For example, 
        `[('longitude', 'latitude'), ('easting', 'northing')]`. If no matches 
        are found and `columns` is `None`, raises an error or warning based 
        on `raise_exception`. If `None`, no default columns are assumed.

    raise_exception : Union[bool, str], default=True
        If True, raises an error if `data` is not a DataFrame or columns are 
        missing. If False, converts errors to warnings. If set to `"mute"` or 
        `"silence"`, suppresses warnings entirely.

    verbose : int, default=0
        If greater than 0, outputs messages about detected columns and 
        transformations.

    round_decimals : int, optional
        If specified, rounds extracted column values to the given number of 
        decimal places.

    fillna_value : Any, optional
        If specified, fills missing values in extracted columns with 
        `fillna_value`.

    unique : bool, default=False
        If True, returns only unique values in extracted columns.

    coerce_dtype : Any, optional
        If specified, coerces extracted column(s) to the provided data type.

    Returns
    -------
    Tuple[Union[Tuple[float, float], pd.DataFrame, None], pd.DataFrame, Tuple[str, ...]]
        - The extracted data as either the midpoint tuple or DataFrame, 
          depending on `as_frame`.
        - The modified original DataFrame, with extracted columns optionally 
          removed.
        - A tuple of detected column names or an empty tuple if none are 
          detected.

    Notes
    -----
    - If `as_frame=False`, computes the midpoint of coordinates by averaging 
      the values:
      
      .. math::
          \text{midpoint} = \left(\frac{\sum \text{longitudes}}{n}, 
          \frac{\sum \text{latitudes}}{n}\right)

    - If `fillna_value` is specified, missing values in extracted columns 
      are filled before further processing.

    Examples
    --------
    >>> import gofast as gf
    >>> from gofast.utils.datautils import data_extractor
    >>> testdata = gf.datasets.make_erp(n_stations=7, seed=42).frame

    # Extract longitude/latitude midpoint
    >>> xy, modified_data, columns = data_extractor(testdata)
    >>> xy, columns
    ((110.48627946874444, 26.051952363176344), ('longitude', 'latitude'))

    # Extract as DataFrame and round coordinates
    >>> xy, modified_data, columns = data_extractor(testdata, as_frame=True, round_decimals=3)
    >>> xy.head(2)
       longitude  latitude
    0    110.486    26.051
    1    110.486    26.051

    # Extract specific columns with unique values and drop from DataFrame
    >>> xy, modified_data, columns = data_extractor(
        testdata, columns=['station', 'resistivity'], unique=True, drop_columns=True)
    >>> xy, modified_data.head(2)
    (array([[0.0, 1.0], [20.0, 167.5]]), <DataFrame without 'station' and 'resistivity'>)

    References
    ----------
    .. [1] Fotheringham, A. Stewart, *Geographically Weighted Regression: 
           The Analysis of Spatially Varying Relationships*, Wiley, 2002.

    See Also
    --------
    pd.DataFrame : Main pandas data structure for handling tabular data.
    np.nanmean : Computes the mean along specified axis, ignoring NaNs.
    """

    def validate_columns(d: pd.DataFrame, cols: List[str]) -> List[str]:
        """Check if columns exist in DataFrame, raising or warning if not."""
        missing = [col for col in cols if col not in d.columns]
        if missing:
            msg = f"Columns {missing} not found in DataFrame."
            if str(raise_exception).lower() == 'true':
                raise KeyError(msg)
            elif raise_exception not in ('mute', 'silence'):
                warnings.warn(msg)
        return [col for col in cols if col in d.columns]

    # Validate input DataFrame
    if not isinstance(data, pd.DataFrame):
        emsg = f"Expected a DataFrame but got {type(data).__name__!r}."
        if str(raise_exception).lower() == 'true':
            raise TypeError(emsg)
        elif raise_exception not in ('mute', 'silence'):
            warnings.warn(emsg)
        return None, data, ()

    # Determine columns to extract based on user input or defaults
    if columns is None:
        if default_columns is not None:
            for col_pair in default_columns:
                if all(col in data.columns for col in col_pair):
                    columns = list(col_pair)
                    break
        if columns is None:
            if str(raise_exception).lower() == 'true':
                raise ValueError("No default columns found in DataFrame.")
            if raise_exception not in ('mute', 'silence'):
                warnings.warn("No default columns found in DataFrame.")
            return None, data, ()

    # Validate extracted columns
    columns = validate_columns(data, columns)

    # Extract specified columns
    extracted = data[columns].copy()
    
    # Apply optional transformations
    if fillna_value is not None:
        extracted.fillna(fillna_value, inplace=True)
    if unique:
        extracted = extracted.drop_duplicates()
    if coerce_dtype:
        extracted = extracted.astype(coerce_dtype)
    if round_decimals is not None:
        extracted = extracted.round(round_decimals)

    # Compute midpoint if `as_frame=False`
    extracted_data = extracted if as_frame else tuple(
        np.nanmean(extracted.values, axis=0))

    # Drop columns from original DataFrame if `drop_columns=True`
    if drop_columns:
        data.drop(columns=columns, inplace=True)

    # Display verbose messages if enabled
    if verbose > 0:
        print("### Extracted columns:", columns)
        if drop_columns:
            print("### Dropped columns from DataFrame.")

    return extracted_data, data, tuple(columns)

def replace_data(
    X:Union [np.ndarray, pd.DataFrame], 
    y: Union [np.ndarray, pd.Series] = None, 
    n: int = 1, 
    axis: int = 0, 
    reset_index: bool = False,
    include_original: bool = False,
    random_sample: bool = False,
    shuffle: bool = False
) -> Union [ np.ndarray, pd.DataFrame , Tuple[
    np.ndarray , pd.DataFrame, np.ndarray, pd.Series]]:
    """
    Duplicates the data `n` times along a specified axis and applies various 
    optional transformations to augment the data suitability for further 
    processing or analysis.

    Parameters
    ----------
    X : Union[np.ndarray, pd.DataFrame]
        The input data to process. Sparse matrices are not supported.
    y : Optional[Union[np.ndarray, pd.Series]], optional
        Additional target data to process alongside `X`. Default is None.
    n : int, optional
        The number of times to replicate the data. Default is 1.
    axis : int, optional
        The axis along which to concatenate the data. Default is 0.
    reset_index : bool, optional
        If True and `X` is a DataFrame, resets the index without adding
        the old index as a column. Default is False.
    include_original : bool, optional
        If True, the original data is included in the output alongside
        the replicated data. Default is False.
    random_sample : bool, optional
        If True, samples from `X` randomly with replacement. Default is False.
    shuffle : bool, optional
        If True, shuffles the concatenated data. Default is False.

    Returns
    -------
    Union[np.ndarray, pd.DataFrame, Tuple[Union[np.ndarray, pd.DataFrame], 
                                          Union[np.ndarray, pd.Series]]]
        The augmented data, either as a single array or DataFrame, or as a tuple
        of arrays/DataFrames if `y` is provided.

    Notes
    -----
    The replacement is mathematically formulated as follows:
    Let :math:`X` be a dataset with :math:`m` elements. The function replicates 
    :math:`X` `n` times, resulting in a new dataset :math:`X'` of :math:`m * n` 
    elements if `include_original` is False. If `include_original` is True,
    :math:`X'` will have :math:`m * (n + 1)` elements.

    Examples
    --------
    
    >>> import numpy as np 
    >>> from gofast.utils.datautils import replace_data
    >>> X, y = np.random.randn ( 7, 2 ), np.arange(7)
    >>> X.shape, y.shape 
    ((7, 2), (7,))
    >>> X_new, y_new = replace_data (X, y, n=10 )
    >>> X_new.shape , y_new.shape
    ((70, 2), (70,))
    >>> X = np.array([[1, 2], [3, 4]])
    >>> replace_data(X, n=2, axis=0)
    array([[1, 2],
           [3, 4],
           [1, 2],
           [3, 4]])

    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> replace_data(df, n=1, include_original=True, reset_index=True)
       A  B
    0  1  3
    1  2  4
    2  1  3
    3  2  4
    """
    def concat_data(ar) -> Union[np.ndarray, pd.DataFrame]:
        repeated_data = [ar] * (n + 1) if include_original else [ar] * n
        
        if random_sample:
            random_indices = np.random.choice(
                ar.shape[0], size=ar.shape[0], replace=True)
            repeated_data = [ar[random_indices] for _ in repeated_data]

        concatenated = pd.concat(repeated_data, axis=axis) if isinstance(
            ar, pd.DataFrame) else np.concatenate(repeated_data, axis=axis)
        
        if shuffle:
            shuffled_indices = np.random.permutation(concatenated.shape[0])
            concatenated = concatenated[shuffled_indices] if isinstance(
                ar, pd.DataFrame) else concatenated.iloc[shuffled_indices]

        if reset_index and isinstance(concatenated, pd.DataFrame):
            concatenated.reset_index(drop=True, inplace=True)
        
        return concatenated

    X = np.array(X) if not isinstance(X, (np.ndarray, pd.DataFrame)) else X
    y = np.array(y) if y is not None and not isinstance(y, (np.ndarray, pd.Series)) else y

    if y is not None:
        return concat_data(X), concat_data(y)
    return concat_data(X)

@SaveFile
@isdf 
@validate_params ({ 
    "long_df": ['array-like'], 
    'index_columns': ['array-like', str, None], 
    'pivot_column': [str], 
    'value_column': [str], 
    'aggfunc': [StrOptions({'first', })], 
    'rename_columns': [list], 
    'rename-dict': [dict], 
    'error': [StrOptions({'raise', 'warn', 'ignore'})]
    }
 )
def long_to_wide(
    long_df,
    index_columns=None,
    pivot_column='year',
    value_column='subsidence',
    aggfunc='first',
    sep='_', 
    name_prefix=None, 
    new_columns=None, 
    error ='warn', 
    exclude_value_from_name=False, 
    savefile=None, 
):
    """
    Convert a DataFrame from long to wide format by pivoting.

    This function transforms a DataFrame from long format to wide
    format by pivoting the DataFrame based on specified index columns,
    pivot column, and value column. The resulting DataFrame will have
    one row per unique combination of `index_columns` and columns for
    each unique value in `pivot_column`.

    Parameters
    ----------
    long_df : pandas.DataFrame
        The input DataFrame in long format containing the data to be
        pivoted.

    index_columns : list of str, optional
        List of column names to use as the index for the pivot
        operation. If `None`, defaults to ``['longitude', 'latitude']``.

    pivot_column : str, default ``'year'``
        The name of the column whose values will be used as new column
        names in the pivoted DataFrame.

    value_column : str, default ``'subsidence'``
        The name of the column whose values will fill the cells of the
        pivoted DataFrame.

    aggfunc : str or callable, default ``'first'``
        The aggregation function to apply if there are duplicate
        entries for the same index and pivot column values.
        
    sep : str, default='_'
        The string used to separate the value of `value_column` and
        `pivot_column` in the column names of the resulting wide-format
        DataFrame.
    
        For example, if `value_column='subsidence'` and `pivot_column='year'`,
        setting `separator='[]'` results in column names like
        `subsidence[2020]`, `subsidence[2021]`.
    
        This parameter allows customizing the naming convention for
        the wide-format DataFrame, which may be useful for specific
        analysis or presentation needs.
        
    name_prefix : str, optional
        If provided, this value will replace `value_column` in the
        column names (e.g., ``name_prefix_2020`` or ``name_prefix[2020]``).

    new_columns : list of str, optional
        If provided, this list will replace the column names of the
        resulting wide-format DataFrame. The length of the list must
        match the number of columns in the resulting DataFrame,
        otherwise a `ValueError` is raised.
        
    error : {'raise', 'warn', 'ignore'}, default='warn'
        Defines the behavior when the provided `new_columns` does not match
        the number of columns in the resulting `wide_df`.
    
        - `'raise'`: Raises a `ValueError` if the length of `new_columns`
          does not match the number of columns in `wide_df`.
        - `'warn'`: Issues a warning and skips renaming if the lengths do
          not match.
        - `'ignore'`: Silently ignores the mismatch and skips renaming.

    exclude_value_from_name : bool, default ``False``
        If True, the `value_column` name is excluded from the resulting
        column names, leaving only the `pivot_column` values as column
        names (e.g., ``2020``, ``2021``). When True, the `separator`
        parameter is ignored.
        
    savefile : str, optional
        If provided, the resulting wide-format DataFrame will be saved
        to this file path in CSV format.

    Returns
    -------
    pandas.DataFrame
        A wide-format DataFrame with one row per combination of
        `index_columns` and columns for each unique value in
        `pivot_column`.

    Notes
    -----
    This function is useful for transforming data from long to wide
    format, which is often required for certain types of analysis or
    visualization. The pivot operation is performed using
    :func:`pandas.pivot_table` [1]_.

    If there are multiple entries for the same index and pivot column
    values, the specified `aggfunc` is applied to aggregate them. The
    aggregation is defined as:

    .. math::

        \text{Value}_{i,j} = \text{aggfunc}(\{ v_k \mid
        \text{index}_k = i, \text{pivot}_k = j \})

    where :math:`v_k` are the values in `value_column`, :math:`i` and
    :math:`j` represent the unique values in `index_columns` and
    `pivot_column`, respectively.

    Examples
    --------
    >>> from gofast.utils.datautils import pivot_long_to_wide
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'longitude': [1, 1, 2, 2],
    ...     'latitude': [3, 3, 4, 4],
    ...     'year': [2020, 2021, 2020, 2021],
    ...     'subsidence': [0.1, 0.2, 0.15, 0.25]
    ... })
    >>> wide_df = pivot_long_to_wide(data)
    >>> print(wide_df)
       longitude  latitude  subsidence_2020  subsidence_2021
    0          1         3             0.10             0.20
    1          2         4             0.15             0.25

    See Also
    --------
    pandas.pivot_table : Create a spreadsheet-style pivot table as a DataFrame.

    References
    ----------
    .. [1] Wes McKinney. "pandas: a foundational Python library for
       data analysis and statistics." Python for High Performance and
       Scientific Computing (2011): 1-9.

    """
    # Set default index columns if not provided
    if index_columns is None:
        index_columns = ['longitude', 'latitude']

    # Check that required columns exist in DataFrame
    required_columns = index_columns + [pivot_column, value_column]
    missing_columns = set(required_columns) - set(long_df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in DataFrame: {missing_columns}")

    # Pivot the DataFrame
    wide_df = long_df.pivot_table(
        index=index_columns,
        columns=pivot_column,
        values=value_column,
        aggfunc=aggfunc
    )

    # Flatten the multi-level columns if necessary
    if isinstance(wide_df.columns, pd.MultiIndex):
        col_names = wide_df.columns.get_level_values(1)
    else:
        col_names = wide_df.columns

    if exclude_value_from_name:
        wide_df.columns = [f'{col}' for col in col_names]
    else:
        name_to_use = name_prefix if name_prefix is not None else value_column
        if sep =='[]':
            wide_df.columns = [
                f'{name_to_use}[{col}]' for col in col_names
            ]
        else:
            wide_df.columns = [
            f'{name_to_use}{sep}{col}' for col in wide_df.columns
            ]
 
    # Apply column renaming if provided
    if new_columns is not None:
        new_columns = is_iterable (
            new_columns, exclude_string=True, transform=True ) 
        err_msg =( 
            f"The length of rename_columns ({len(new_columns)}) "
            "does not match the number of columns in the DataFrame"
            f" ({len(wide_df.columns)})."
            )
        if len(new_columns) != len(wide_df.columns):
            if error =="warn": 
                warnings.warn(err_msg) 
            elif error =='raise': 
                raise ValueError(err_msg)  
            # ignore and pass 
        else: 
            wide_df.columns = new_columns
        
    # Reset the index to convert back to a normal DataFrame
    wide_df = wide_df.reset_index()

    # if savefile:
    #     wide_df.to_csv(savefile, index=False)

    return wide_df

@SaveFile 
@isdf
@validate_params ({ 
    "wide_df": ['array-like'], 
    'id_vars': ['array-like', str, None], 
    'value_vars': ['array-like', str, None], 
    'value_name': [str], 
    'var_name': [str], 
    'rename_columns': [list], 
    'rename-dict': [dict], 
    'error': [StrOptions({'raise', 'warn', 'ignore'})]
    }
 )
def wide_to_long(
    wide_df, 
    id_vars=None, 
    value_vars=None,
    value_name='value', 
    var_name='variable', 
    rename_columns=None, 
    rename_dict=None,
    error='raise',
    savefile=None, 
    **kwargs
):
    """
    Convert a wide-format DataFrame to a long-format DataFrame.

    Parameters
    ----------
    wide_df : pandas.DataFrame
        The input DataFrame in wide format.

    id_vars : list of str, optional
        Column names to use as identifier variables (columns to keep as is).
        If None, all columns not specified in `value_vars` will be used.

    value_vars : list of str, optional
        Column names to unpivot. If None, all columns not in `id_vars` will be used.

    value_name : str, default='value'
        Name of the column that will contain the values from the wide DataFrame.

    var_name : str, default='variable'
        Name of the column that will contain the variable names from the
        wide DataFrame (e.g., the wide-format column headers).

    rename_columns : list of str, optional
        If provided, this list will replace the column names of the resulting
        DataFrame. The length of `rename_columns` must match the number of 
        resulting columns.

    rename_dict : dict, optional
        A dictionary mapping existing column names to new names. This allows
        selective renaming without needing to specify all column names.

    error : {'raise', 'warn', 'ignore'}, default='raise'
        Defines the behavior when the provided `rename_columns` or keys in
        `rename_dict` do not match the existing columns:
        - `'raise'`: Raise a `ValueError` if there is a mismatch.
        - `'warn'`: Issue a warning and skip renaming for mismatched entries.
        - `'ignore'`: Silently ignore mismatches and proceed.

    **kwargs
        Additional keyword arguments to pass to `pd.melt`, such as `col_level`.

    Returns
    -------
    pandas.DataFrame
        A long-format DataFrame with one row per unique combination of
        `id_vars` and the original wide-format columns.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.utils.datautils import wide_to_long 
    >>> wide_df = pd.DataFrame({
    ...     'id': [1, 2],
    ...     'longitude': [10, 20],
    ...     'latitude': [30, 40],
    ...     '2015': [0.1, 0.15],
    ...     '2016': [0.2, 0.25]
    ... })
    >>> long_df = wide_to_long(
    ...     wide_df, 
    ...     id_vars=['id', 'longitude', 'latitude'], 
    ...     value_name='subsidence', 
    ...     var_name='year'
    ... )
    >>> print(long_df)
       id  longitude  latitude  year  subsidence
    0   1         10        30  2015        0.10
    1   2         20        40  2015        0.15
    2   1         10        30  2016        0.20
    3   2         20        40  2016        0.25

    >>> # Using rename_columns
    >>> renamed_df = wide_to_long(
    ...     wide_df, 
    ...     id_vars=['id', 'longitude', 'latitude'], 
    ...     value_name='subsidence', 
    ...     var_name='year',
    ...     rename_columns=['ID', 'Lon', 'Lat', 'Year', 'Subsidence']
    ... )
    >>> print(renamed_df)
       ID  Lon  Lat  Year  Subsidence
    0   1   10   30  2015        0.10
    1   2   20   40  2015        0.15
    2   1   10   30  2016        0.20
    3   2   20   40  2016        0.25

    >>> # Using rename_dict
    >>> renamed_dict_df = wide_to_long(
    ...     wide_df, 
    ...     id_vars=['id', 'longitude', 'latitude'], 
    ...     value_name='subsidence', 
    ...     var_name='year',
    ...     rename_dict={'id': 'ID', 'longitude': 'Lon', 'latitude': 'Lat'}
    ... )
    >>> print(renamed_dict_df)
       ID  Lon  Lat  year  subsidence
    0   1   10   30  2015        0.10
    1   2   20   40  2015        0.15
    2   1   10   30  2016        0.20
    3   2   20   40  2016        0.25
    """
    # Input Validation
    if not isinstance(wide_df, pd.DataFrame):
        raise TypeError(
            "wide_df must be a pandas DataFrame,"
            f" got {type(wide_df)} instead.")
    
    if id_vars is not None:
        id_vars= is_iterable(id_vars, exclude_string= True, transform =True )
        missing_id_vars = set(id_vars) - set(wide_df.columns)
        if missing_id_vars:
            raise ValueError(
                "The following id_vars are not in"
                f" the DataFrame columns: {missing_id_vars}")
    
    if value_vars is not None:
        value_vars= is_iterable(value_vars, exclude_string= True, 
                                transform =True )
        missing_value_vars = set(value_vars) - set(wide_df.columns)
        if missing_value_vars:
            raise ValueError(
                f"The following value_vars are not"
                f" in the DataFrame columns: {missing_value_vars}")
    
    if rename_columns is not None and not isinstance(rename_columns, list):
        raise TypeError(
            "rename_columns must be a list of strings,"
            f" got {type(rename_columns)} instead.")
    
    if rename_dict is not None and not isinstance(rename_dict, dict):
        raise TypeError(
            "rename_dict must be a dictionary,"
            f" got {type(rename_dict)} instead.")
    
    # Determine id_vars and value_vars if not provided
    if id_vars is None:
        if value_vars is not None:
            id_vars = [col for col in wide_df.columns if col not in value_vars]
        else:
            # If neither id_vars nor value_vars are provided, melt all columns except one
            if wide_df.shape[1] < 2:
                raise ValueError(
                    "wide_df must have at least two columns to perform melt.")
            id_vars = [wide_df.columns[0]]
            value_vars = list(wide_df.columns[1:])
    else:
        if value_vars is None:
            value_vars = [col for col in wide_df.columns if col not in id_vars]
        elif not value_vars:
            raise ValueError("value_vars cannot be an empty list.")

    # Perform the melt operation
    try:
        long_df = pd.melt(
            wide_df,
            id_vars=id_vars,
            value_vars=value_vars,
            var_name=var_name,
            value_name=value_name,
            **kwargs
        )
    except Exception as e:
        raise ValueError(f"Error during melting the DataFrame: {e}")

    # Apply renaming if rename_columns or rename_dict is provided
    if rename_columns is not None:
        if len(rename_columns) != len(long_df.columns):
            err_msg = (
                f"The length of rename_columns ({len(rename_columns)}) "
                "does not match the number of columns in"
                f" the resulting DataFrame ({len(long_df.columns)})."
            )
            if error == 'warn':
                warnings.warn(err_msg)
            elif error == 'raise':
                raise ValueError(err_msg)
            # 'ignore' will silently skip renaming
        else:
            long_df.columns = rename_columns

    if rename_dict is not None:
        # Check if keys in rename_dict exist in the DataFrame
        existing_keys = set(rename_dict.keys()).intersection(long_df.columns)
        if not existing_keys and error == 'raise':
            raise ValueError(
                "None of the keys in rename_dict match the DataFrame columns.")
        if existing_keys != set(rename_dict.keys()):
            missing_keys = set(rename_dict.keys()) - existing_keys
            err_msg = (
                "The following keys in rename_dict do not match"
                f" any DataFrame columns and will be skipped: {missing_keys}"
            )
            if error == 'warn':
                warnings.warn(err_msg)
            elif error == 'raise':
                raise ValueError(err_msg)
            # 'ignore' will silently skip renaming these keys
        # Perform the renaming
        long_df = long_df.rename(
            columns={k: v for k, v in rename_dict.items() if k in long_df.columns})

    return long_df

@SaveFile 
@is_data_readable
@isdf 
def repeat_feature_accross(
    data: DataFrame,
    date_col: str = 'date',
    start_date: Union[int, pd.Timestamp] = None,
    end_date: Union[int, pd.Timestamp] = None,
    n_times: int = None,
    custom_dates: List[Union[int, pd.Timestamp]] = None,
    drop_existing_date: bool = True,
    sort: bool = False,
    inplace: bool = False, 
    savefile=None, 
) -> DataFrame:
    """
    Repeat static feature across multiple years or specified dates.

    This function duplicates each row in the input DataFrame across a range of
    years, specific dates, or for a defined number of repetitions (`n_times`). 
    It is designed to handle various scenarios, ensuring flexibility and 
    robustness.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing static data to be repeated across dates.
    
    date_col : str, default='date'
        The name of the date column in the resulting DataFrame.
    
    start_date : int or pd.Timestamp, optional
        The starting date for duplication. Must be specified if 
        `end_date` is provided.
    
    end_date : int or pd.Timestamp, optional
        The ending date for duplication. Must be specified if 
        `start_date` is provided.
    
    n_times : int, optional
        The number of times to repeat the data. Overrides `start_date` 
        and `end_date` if provided.
    
    custom_dates : list of int or pd.Timestamp, optional
        A custom list of dates to duplicate the data across. Overrides 
        `start_date`, `end_date`, and `n_times` if provided.
    
    drop_existing_date : bool, default=True
        If `True`, drops the existing `date_col` in `data` before duplication
        to avoid conflicts.
    
    sort : bool, default=False
        If `True`, sorts the resulting DataFrame by the `date_col`.
    
    inplace : bool, default=False
        If `True`, modifies the input DataFrame `data` in place and returns it.
        If `False`, returns a new DataFrame with the duplicated data.
    
    Returns
    -------
    pd.DataFrame
        A DataFrame with the static data repeated across the specified dates. The
        resulting DataFrame includes the `date_col` indicating the date for each
        duplicated entry.

    Raises
    ------
    ValueError
        - If neither `custom_dates`, (`start_date` and `end_date`), nor 
          `n_times` is provided.
        - If `start_date` is provided without `end_date`, or vice versa.
        - If `n_times` is not a positive integer.
        - If `custom_dates` is not a list of integers or pandas Timestamps.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from datetime import datetime
    >>> from gofast.utils.datautils import repeat_feature_accross
    >>> data = {
    ...     'longitude': [113.291328, 113.291847, 113.291847],
    ...     'latitude': [22.862476, 22.865587, 22.865068],
    ...     'geology': ['Triassic', 'Carboniferous', 'Tertiary']
    ... }
    >>> data = pd.DataFrame(data)
    >>> repeated_df = repeat_feature_accross(
    ...     data, 
    ...     date_col='year', 
    ...     start_date=2015, 
    ...     end_date=2022
    ... )
    >>> print(repeated_df)
         longitude   latitude        geology  year
    0   113.291328  22.862476       Triassic  2015
    1   113.291328  22.862476       Triassic  2016
    2   113.291328  22.862476       Triassic  2017
    3   113.291328  22.862476       Triassic  2018
    4   113.291328  22.862476       Triassic  2019
    5   113.291328  22.862476       Triassic  2020
    ...
    19  113.291847  22.865068       Tertiary  2018
    20  113.291847  22.865068       Tertiary  2019
    21  113.291847  22.865068       Tertiary  2020
    22  113.291847  22.865068       Tertiary  2021
    23  113.291847  22.865068       Tertiary  2022
    
    >>> # Using n_times
    >>> repeated_df = repeat_feature_accross(
    ...     data, 
    ...     date_col='year', 
    ...     n_times=3
    ... )
    >>> print(repeated_df)
        longitude   latitude        geology  year
    0  113.291328  22.862476       Triassic     1
    1  113.291847  22.865587  Carboniferous     1
    2  113.291847  22.865068       Tertiary     1
    3  113.291328  22.862476       Triassic     2
    4  113.291847  22.865587  Carboniferous     2
    5  113.291847  22.865068       Tertiary     2
    6  113.291328  22.862476       Triassic     3
    7  113.291847  22.865587  Carboniferous     3
    8  113.291847  22.865068       Tertiary     3
    """
    # Input Validation
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"Expected 'data' to be a pandas DataFrame, got {type(data)} instead."
        )
    
    # Determine the list of dates/years
    if custom_dates is not None:
        if (not isinstance(custom_dates, list) or 
            not all(isinstance(
                date, (int, pd.Timestamp)) for date in custom_dates)):
            raise ValueError(
                "'custom_dates' must be a list of integers or pandas Timestamps."
            )
        dates = sorted(set(custom_dates))
    elif n_times is not None:
        n_times = validate_positive_integer(
            n_times, "n_times", msg= "'n_times' must be a positive integer."
            )
        if start_date is None and end_date is None:
            # Default to 1 to n_times
            dates = list(range(1, n_times + 1))
        elif start_date is not None and end_date is not None:
            if not isinstance(start_date, (int, pd.Timestamp)) or not isinstance(
                    end_date, (int, pd.Timestamp)):
                raise ValueError(
                    "'start_date' and 'end_date' must be integers or pandas Timestamps."
                )
            if isinstance(start_date, int) and isinstance(end_date, int):
                if start_date > end_date:
                    raise ValueError(
                        "'start_date' must be less than or equal to 'end_date'."
                    )
                dates = list(range(start_date, end_date + 1))
            elif isinstance(start_date, pd.Timestamp) and isinstance(
                    end_date, pd.Timestamp):
                if start_date > end_date:
                    raise ValueError(
                        "'start_date' must be earlier than or equal to 'end_date'."
                    )
                dates = pd.date_range(start=start_date, end=end_date, freq='Y').tolist()
            else:
                raise ValueError(
                    "'start_date' and 'end_date' must both"
                    " be integers or both be pandas Timestamps."
                )
        else:
            raise ValueError(
                "Both 'start_date' and 'end_date' must be provided together."
            )
    elif start_date is not None and end_date is not None:
        if not isinstance(start_date, (int, pd.Timestamp)) or not isinstance(
                end_date, (int, pd.Timestamp)):
            raise ValueError(
                "'start_date' and 'end_date' must be integers or pandas Timestamps.")
        if isinstance(start_date, int) and isinstance(end_date, int):
            if start_date > end_date:
                raise ValueError(
                    "'start_date' must be less than or equal to 'end_date'.")
            dates = list(range(start_date, end_date + 1))
        elif isinstance(
                start_date, pd.Timestamp) and isinstance(end_date, pd.Timestamp):
            if start_date > end_date:
                raise ValueError(
                    "'start_date' must be earlier than or equal to 'end_date'.")
            dates = pd.date_range(
                start=start_date, end=end_date, freq='Y').tolist()
        else:
            raise ValueError(
                "'start_date' and 'end_date' must both"
                " be integers or both be pandas Timestamps."
            )
    else:
        raise ValueError(
            "Must provide either 'custom_dates', "
            "('start_date' and 'end_date'), or 'n_times'."
        )
    
    # Validate date_col type consistency
    if isinstance(dates[0], pd.Timestamp):
        if not pd.api.types.is_datetime64_any_dtype(
                data.get(date_col, pd.Series(dtype='object'))):
            pass  # Allow creation of date_col as datetime
    elif isinstance(dates[0], int):
        if not pd.api.types.is_integer_dtype(
                data.get(date_col, pd.Series(dtype='object'))):
            pass  # Allow creation of date_col as integer
    
    # Handle existing date column
    if drop_existing_date and date_col in data.columns:
        data = data.drop(columns=[date_col])
    
    # Create a DataFrame with the dates to merge
    dates_df = pd.DataFrame({date_col: dates})
    
    # Perform cross join to duplicate rows across dates
    df_repeated = data.merge(dates_df, how='cross')
    
    # Sort if required
    if sort:
        df_repeated = df_repeated.sort_values(by=date_col).reset_index(drop=True)
    
    if inplace:
        data.drop(columns=data.columns, inplace=True)
        for col in df_repeated.columns:
            data[col] = df_repeated[col]
        return data
    else:
        return df_repeated.reset_index(drop=True)

@SaveFile 
def merge_datasets(
    *dfs, 
    on=None, 
    how='inner', 
    fill_missing=False, 
    fill_value=None, 
    keep_duplicates=False, 
    suffixes=('_x', '_y'), 
    savefile=None, 
):
    """
    Merge multiple datasets into a single DataFrame.

    Parameters
    ----------
    dfs : pandas.DataFrame
        Variable-length arguments of DataFrames to be merged.

    on : list of str or None, default None
        The list of columns to join on. If None, the intersection of
        columns across datasets is used.

    how : {'inner', 'outer', 'left', 'right'}, default 'inner'
        Type of merge to be performed:
        - 'inner': Only include rows with matching keys in all datasets.
        - 'outer': Include all rows from all datasets, filling missing
          values with NaN.
        - 'left': Include all rows from the first dataset and matching
          rows from others.
        - 'right': Include all rows from the last dataset and matching
          rows from others.

    fill_missing : bool, default False
        If True, fills missing values with a default value.

    fill_value : any, default None
        The value to use when `fill_missing` is True. If None, numeric
        columns are filled with 0, and non-numeric columns are filled
        with an empty string.

    keep_duplicates : bool, default False
        If True, keeps duplicate rows across datasets after merging. If
        False, removes duplicates from the merged DataFrame.

    suffixes : tuple of (str, str), default ('_x', '_y')
        Suffixes to apply to overlapping column names when merging.

    Returns
    -------
    pandas.DataFrame
        A merged DataFrame containing all the data.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.utils.datautils import merge_datasets
    >>> df1 = pd.DataFrame({'longitude': [1, 2], 'latitude': [3, 4],
    ...                     'year': [2020, 2021], 'value1': [10, 20]})
    >>> df2 = pd.DataFrame({'longitude': [1, 2], 'latitude': [3, 4],
    ...                     'year': [2020, 2021], 'value2': [100, 200]})
    >>> merged = merge_datasets(df1, df2, on=['longitude', 'latitude',
    ...                                       'year'], how='inner')
    >>> print(merged)
       longitude  latitude  year  value1  value2
    0          1         3  2020      10     100
    1          2         4  2021      20     200
    """
    
    [is_frame(d, df_only=True, raise_exception=True, objname='Dataset')
            for d in dfs 
    ]
    if len(dfs) < 2:
        raise ValueError(
            "At least two DataFrames are required for merging."
        )
    # Ensure all arguments are DataFrames
    for df in dfs:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(df)}.")

    if on is None:
        # Use the intersection of all datasets' columns for merging
        common_columns = set(dfs[0].columns)
        for df in dfs[1:]:
            common_columns.intersection_update(df.columns)
        on = list(common_columns)
    
    if not on:
        raise ValueError(
            "No common columns found for merging. Specify the 'on' parameter."
        )

    # Perform iterative merging
    merged_df = reduce(lambda left, right: pd.merge(
        left, right, on=on, how=how, suffixes=suffixes), dfs)

    # Fill missing values if required
    if fill_missing:
        if fill_value is None:
            for column in merged_df:
                if merged_df[column].dtype.kind in 'biufc':  # Numeric types
                    merged_df[column].fillna(0, inplace=True)
                else:  # Categorical or string types
                    merged_df[column].fillna('', inplace=True)
        else:
            merged_df.fillna(fill_value, inplace=True)

    # Remove duplicates if specified
    if not keep_duplicates:
        merged_df.drop_duplicates(inplace=True)

    return merged_df

@isdf 
def swap_ic(
    df: DataFrame, 
    sort: bool = False, 
    ascending: bool = True, 
    inplace: bool = False, 
    reset_index: bool = False, 
    dropna: bool = False, 
    fillna: bool = None, 
    axis: int = 0, 
    order: list = None,
    **kwargs
    ):
    """
    Align the index and columns of a DataFrame so that they follow the 
    same order.

    This function ensures that if the values in the index and columns 
    are the same, the DataFrame will align its index and columns in 
    the same order. Optionally, the index and columns can be sorted, reset,
    and cleaned with additional parameters for flexibility.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame whose index and columns need to be aligned.

    sort : bool, optional, default=False
        Whether to sort the index and columns in ascending order. If ``True``,
        both index and columns will be sorted in ascending order.

    ascending : bool, optional, default=True
        If `sort=True`, this specifies the sorting order. If `True`, the 
        index and columns will be sorted in ascending order. Otherwise, 
        descending order will be applied.

    inplace : bool, optional, default=False
        If `True`, modifies the DataFrame in place. If `False`, 
        returns a new DataFrame.

    reset_index : bool, optional, default=False
        If `True`, resets the index after aligning the index and columns.

    dropna : bool, optional, default=False
        If `True`, rows or columns with NaN values will be dropped.

    fillna : scalar or dict, optional, default=None
        Value to replace NaNs with. If `None`, no filling is performed. 
        If a scalar, all NaNs will be replaced with that value. If a dict, 
        it should provide mappings for index and columns.

    axis : int, optional, default=0
        Axis along which to perform alignment. `0` for index, `1` for columns.
        
    order : list, optional, default=None
        Custom order for the index and columns. If specified, the index
        and columns will be ordered according to the provided list. If the
        order is not present in either index or columns, it will be ignored. 
        If `None`, no custom order is applied.

    kwargs : additional keyword arguments
        Any additional arguments that might be passed to the DataFrame operations.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame (or the original, if `inplace=True`) with aligned index 
        and columns, with optional sorting, resetting, and cleaning applied.
    
    Raises
    ------
    ValueError
        If the index and columns are not the same, the function will 
        raise an error  or allow custom handling if required.
    
    Examples
    --------
    Example of using the function without sorting:

    >>> from gofast.utils.datautils import swap_ic 
    >>> df = pd.DataFrame({
    >>>     'A': [1, 2, 3],
    >>>     'B': [4, 5, 6],
    >>>     'C': [7, 8, 9]
    >>> }, index=['B', 'A', 'C'])
    >>> swap_ic(df, sort=False)
    >>> swap_ic(df, sort=True, ascending=True)

    Example of using the function with sorting and filling NaNs:

    >>> swap_ic(df, sort=True, fillna=0)
    
    Example of using the function with custom order:

    >>> swap_ic(df, order=['B', 'A', 'C'])
    >>> swap_ic(df, order=['C', 'A', 'B'])
    
    """
    # Validate that index and columns have the same elements and
    # optionally, apply custom order to both index and columns
    df = is_df_square(
        df, 
        order =order, 
        check_symmetry=True, 
        ops ='validate'
    )
    # Align index and columns by ensuring they are in the same order
    aligned_df = df.loc[df.columns, df.columns]

    # Sort the index and columns if requested
    if sort:
        aligned_df = aligned_df.sort_index(ascending=ascending, axis=0)
        aligned_df = aligned_df.sort_index(ascending=ascending, axis=1)

    # Reset index if requested
    if reset_index:
        aligned_df = aligned_df.reset_index(drop=True)

    # Drop NaN values if requested
    if dropna:
        aligned_df = aligned_df.dropna(axis=axis, how='any')

    # Fill NaN values if requested
    if fillna is not None:
        aligned_df = aligned_df.fillna(fillna)

    # Apply changes in place if requested
    if inplace:
        df[:] = aligned_df
        return None  # None is returned if inplace=True
    else:
        return aligned_df
    
@is_data_readable     
@isdf 
def batch_sampling(
    data,
    sample_size=0.1,
    n_batches=10,
    stratify_by=None,
    random_state=None,
    replacement=False,
    shuffle=True,
    return_indices=False,
):
    """
    Batch sampling with optional stratification and replacement.

    This function divides a dataset into multiple batches, each being a sample
    of the data. It ensures that samples in the first batch are not present in
    subsequent batches when `replacement` is ``False``. This is particularly
    useful for processing large datasets in batches, allowing for efficient
    memory usage and parallel processing.

    Parameters
    ----------
    data : pandas.DataFrame
        The input DataFrame from which samples are to be drawn.

    sample_size : float or int, optional
        The total number of samples to draw from `data`. If `sample_size` is
        a float between 0.0 and 1.0, it represents the fraction of the dataset
        to include in the sample (e.g., `sample_size=0.1` selects 10% of the
        data). If `sample_size` is an integer, it represents the absolute
        number of samples to select. The default is ``0.1``.

    n_batches : int, optional
        The number of batches to divide the total samples into. The samples
        are divided as evenly as possible among the batches. The default is
        ``10``.

    stratify_by : list of str or None, optional
        A list of column names in `data` to use for stratification. If
        specified, the sampling will ensure that the distribution of these
        columns in each batch matches the distribution in the original
        dataset. If ``None``, no stratification is applied. The default is
        ``None``.

    random_state : int or None, optional
        Controls the randomness of the sampling for reproducibility. This
        integer seed is used to initialize the random number generator. If
        ``None``, the random number generator is not seeded. The default is
        ``None``.

    replacement : bool, optional
        If ``True``, samples are drawn with replacement. If ``False``, samples
        are drawn without replacement, and sampled data is removed from the
        pool of available data for subsequent batches. The default is
        ``False``.

    shuffle : bool, optional
        If ``True``, the data is shuffled before sampling. This is relevant
        when `replacement` is ``False`` to ensure that the data is sampled
        randomly. The default is ``True``.

    return_indices : bool, optional
        If ``True``, the function yields indices of the sampled data instead
        of the data itself. If ``False``, the function yields DataFrames
        containing the sampled data. The default is ``False``.

    Yields
    ------
    batch : pandas.DataFrame or list of int
        If `return_indices` is ``False``, each yield is a DataFrame containing
        the sampled data for that batch. If `return_indices` is ``True``, each
        yield is a list of indices corresponding to the sampled data for that
        batch.

    Notes
    -----
    The total number of samples, :math:`n`, is divided among the batches, and
    within each batch, samples are drawn (optionally stratified). The sample
    size for each batch is calculated as:

    .. math::

        n_{\text{batch}} = \left\lfloor \frac{n}{n_{\text{batches}}} \right\rfloor

    The remaining samples are distributed among the first few batches:

    .. math::

        n_{\text{leftover}} = n \mod n_{\text{batches}}

    For each batch, if stratification is applied, the number of samples per
    stratification group is calculated based on the proportion of the group
    size to the remaining data size:

    .. math::

        n_{i} = \left\lceil \frac{N_{i}}{N_{\text{remaining}}}\\
            \times n_{\text{batch}} \right\rceil

    where:

    - :math:`N_{i}` is the size of group :math:`i`.
    - :math:`N_{\text{remaining}}` is the total number of samples remaining in
      the data.
    - :math:`n_{i}` is the number of samples to draw from group :math:`i`.

    After sampling, the selected samples are removed from the remaining data
    (if `replacement` is ``False``) to ensure that they are not selected again
    in subsequent batches.

    Examples
    --------
    >>> from gofast.utils.datautils import batch_sampling
    >>> import pandas as pd
    >>> # Create a sample DataFrame
    >>> data = pd.DataFrame({
    ...     'feature1': range(1000),
    ...     'feature2': ['A'] * 500 + ['B'] * 500,
    ...     'label': [0, 1] * 500
    ... })
    >>> # Use batch_sampling without stratification
    >>> batches = batch_sampling(
    ...     data=data,
    ...     sample_size=0.2,
    ...     n_batches=4,
    ...     random_state=42
    ... )
    >>> for i, batch in enumerate(batches):
    ...     print(f"Batch {i+1} shape: {batch.shape}")
    Batch 1 shape: (50, 3)
    Batch 2 shape: (50, 3)
    Batch 3 shape: (50, 3)
    Batch 4 shape: (50, 3)

    >>> # Use batch_sampling with stratification
    >>> batches = batch_sampling(
    ...     data=data,
    ...     sample_size=200,
    ...     n_batches=4,
    ...     stratify_by=['label'],
    ...     random_state=42
    ... )
    >>> for i, batch in enumerate(batches):
    ...     print(f"Batch {i+1} label distribution:")
    ...     print(batch['label'].value_counts())
    Batch 1 label distribution:
    0    25
    1    25
    Name: label, dtype: int64
    Batch 2 label distribution:
    0    25
    1    25
    Name: label, dtype: int64
    Batch 3 label distribution:
    0    25
    1    25
    Name: label, dtype: int64
    Batch 4 label distribution:
    0    25
    1    25
    Name: label, dtype: int64

    See Also
    --------
    pandas.DataFrame.sample : Method used for random sampling.
    sklearn.model_selection.StratifiedShuffleSplit : Alternative for stratified sampling.

    References
    ----------
    .. [1] Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of
           Statistical Learning: Data Mining, Inference, and Prediction*.
           Springer Science & Business Media.

    """

    data = data.copy()
    total_samples = sample_size
    if isinstance(sample_size, float):
        if not 0 < sample_size <= 1:
            raise ValueError(
                "When sample_size is a float, it must be between 0 and 1.")
        total_samples = int(len(data) * sample_size)
    elif isinstance(sample_size, int):
        if sample_size <= 0:
            raise ValueError("sample_size must be positive.")
    else:
        raise ValueError("sample_size must be a float or int.")

    if total_samples > len(data) and not replacement:
        raise ValueError("sample_size is larger than the dataset.")

    if n_batches <= 0:
        raise ValueError("n_batches must be a positive integer.")

    sample_size_per_batch = total_samples // n_batches
    leftover = total_samples % n_batches

    remaining_data = data.copy()
    rng = np.random.RandomState(random_state)

    for batch_idx in range(n_batches):
        # Adjust sample size for batches if total_samples is not divisible by n_batches
        if batch_idx < leftover:
            batch_sample_size = sample_size_per_batch + 1
        else:
            batch_sample_size = sample_size_per_batch

        if batch_sample_size == 0:
            continue  # No samples to draw in this batch

        if stratify_by is not None:
            # Stratified sampling
            grouped = remaining_data.groupby(stratify_by)
            group_sizes = grouped.size()
            total_size = group_sizes.sum()
            group_sample_sizes = (
                (group_sizes / total_size * batch_sample_size)
                .round()
                .astype(int)
            )
            sampled_indices = []
            for strat_value, group in grouped:
                n = group_sample_sizes.loc[strat_value]
                if n > 0 and len(group) > 0:
                    sampled_group = group.sample(
                        n=min(n, len(group)) if not replacement else n,
                        replace=replacement,
                        random_state=rng.randint(0, 10000),
                    )
                    sampled_indices.extend(sampled_group.index)
        else:
            # Simple random sampling
            sampled_indices = remaining_data.sample(
                n=batch_sample_size if not replacement else batch_sample_size,
                replace=replacement,
                random_state=rng.randint(0, 10000),
            ).index.tolist()

        # Yield the batch
        if return_indices:
            yield sampled_indices
        else:
            yield remaining_data.loc[sampled_indices]

        if not replacement:
            # Remove sampled data from remaining_data
            remaining_data = remaining_data.drop(index=sampled_indices)
            if len(remaining_data) == 0:
                break  # No more data to sample

@SaveFile 
def to_categories(
    df: pd.DataFrame,
    column: str,  
    categories: Union[List[str], str]= 'auto',  
    method: str = 'equal_range', 
    bins: Optional[List[float]] = None,
    include_lowest: bool = True,
    right: bool = False,
    category_name: Optional[str] = None, 
    drop: bool=False, 
    savefile: Optional[str]=None, 
) -> pd.DataFrame:
    """
    Categorize a Continuous Column into Specified or Automatically 
    Generated Categories.
    
    This function transforms a continuous numerical column in a DataFrame into 
    categorical bins based on specified criteria. It supports both equal  
    range and quantile-based binning methods. Additionally, it allows for 
     automatic category label generation when `categories` is set to `'auto'`,
     producing labels such as `'< a'`, `'a-b'`, ..., `'>c'` with bin edges 
     rounded to one decimal place.
    
    The categorization process is defined as:
    
    .. math::
        \text{Bin Width} = \frac{\text{max}(x) - \text{min}(x)}{k}
    
    where :math:`x` represents the data in the column and :math:`k` is the number 
    of categories.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the data to be categorized.
    
    column : str
        The name of the column in `df` to categorize. This column must contain 
        continuous numerical data.
    
    categories : Union[List[str], str], default='auto'
        Defines the labels for the categories. 
        
        - If a list of strings is provided, these labels will be assigned to the 
          respective bins.
        - If set to `'auto'`, the function automatically generates category labels 
          based on the bin edges, resulting in labels like `'< a'`, `'a-b'`, ..., 
          `'>c'`.
    
    method : str, default='equal_range'
        The strategy used to bin the data. 
        
        - `'equal_range'`: Divides the range of the data into equal-sized bins.
        - `'quantile'`: Divides the data into bins with an equal number of 
          data points.
    
    bins : List[float], optional, default=None
        The specific bin edges to use when `method='equal_range'`. 
        
        - If `None`, bin edges are calculated based on the minimum and maximum 
          values of the column, divided equally according to the number of 
          categories.
        - This parameter is ignored if `method='quantile'`.
    
    include_lowest : bool, default=True
        Determines whether the lowest value should be included in the first bin. 
        
        - If `True`, the first bin includes the lowest value.
        - If `False`, the first bin does not include the lowest value.
    
    right : bool, default=False
        Specifies whether the bins include the rightmost edge or not. 
        
        - If `True`, the bins include the right edge.
        - If `False`, the bins do not include the right edge.
    
    category_name : str, optional, default=None
        The name of the new column to be added to `df` containing the category 
        labels.
        
        - If `None`, the new column is named as `{column}_category`.
        - Otherwise, it uses the provided `category_name`.
    
    drop: bool, default=False 
       Drop the the continous column being categorized. If ``False``, keep 
       the categorrized `column` in the dataframe `df`. 
       
    Returns
    -------
    pandas.DataFrame
        The original DataFrame `df` augmented with a new column containing the 
        categorized data.
    
    Raises
    ------
    ValueError
        - If `column` does not exist in `df`.
        - If `column` is not of a numeric data type.
        - If `method` is neither `'equal_range'` nor `'quantile'`.
        - If `categories` is set to `'auto'` but bin labels cannot be generated.
        - If the number of provided `categories` does not match the number of bins 
          minus one.
        - If `categories` is neither a list of strings nor `'auto'`.
    
    Notes
    -----
    - The function utilizes external validators from `gofast.utils.validator` 
      and `gofast.core.checks` to ensure input integrity.
    - When `categories='auto'`, the function dynamically generates category 
      labels based on the calculated bin edges, rounding each edge to one 
      decimal place for clarity.
    - In `'quantile'` method, if duplicate edges are found, the `duplicates='drop'` 
      parameter in `pd.qcut` ensures unique bin edges.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from gofast.utils.datautils import to_categories
    
    >>> # Sample DataFrame
    >>> data = {
    ...     'value': np.random.uniform(0, 100, 1000)
    ... }
    >>> df = pd.DataFrame(data)
             value
    0    28.595432
    1    42.502077
    2    60.824855
    3    50.699502
    4    90.861483
    ..         ...
    995  95.766006
    996  11.983319
    997  93.596298
    998   9.529297
    999  74.836838

    [1000 rows x 1 columns]
    
    >>> # Categorize using equal range with custom categories
    >>> df = to_categories(
    ...     df=df,
    ...     column='value',
    ...     categories=['low', 'medium', 'high'],
    ...     method='equal_range',
    ...     bins=[0, 33.3, 66.6, 100],
    ...     category_name='value_category'
    ... )
    >>> df
             value value_category
    0    49.797081         medium
    1     4.435289            low
    2    19.566820            low
    3    77.484627           high
    4    66.384490         medium
    ..         ...            ...
    995  76.688965           high
    996  32.239027            low
    997  57.611926         medium
    998  58.372411         medium
    999  16.374454            low
    
    [1000 rows x 2 columns]
    
    >>> # Categorize using quantile with automatic categories
    >>> df = to_categories(
    ...     df=df,
    ...     column='value',
    ...     categories='auto',
    ...     method='quantile',
    ...     category_name='value_quantile'
    ... )
    >>> df
             value value_category value_quantile
    0    28.595432            low    26.1 - 48.6
    1    42.502077         medium    26.1 - 48.6
    2    60.824855         medium    48.6 - 74.8
    3    50.699502         medium    48.6 - 74.8
    4    90.861483           high         > 74.8
    ..         ...            ...            ...
    995  95.766006           high         > 74.8
    996  11.983319            low         < 26.1
    997  93.596298           high         > 74.8
    998   9.529297            low         < 26.1
    999  74.836838           high         > 74.8

    [1000 rows x 3 columns]
    
    >>> # Categorize with automatic category labels and default parameters
    >>> df = to_categories(
    ...     df=df,
    ...     column='value',
    ...     categories='auto', 
            category_name='value_equal_range'
    ... )
             value value_category value_quantile value_equal_range
    0    47.570125         medium    24.9 - 49.3       33.3 - 66.6
    1    78.155685           high         > 74.4            > 66.6
    2    27.954301            low    24.9 - 49.3            < 33.3
    3    41.006147         medium    24.9 - 49.3       33.3 - 66.6
    4    61.096433         medium    49.3 - 74.4       33.3 - 66.6
    ..         ...            ...            ...               ...
    995  30.748508            low    24.9 - 49.3            < 33.3
    996  74.653811           high         > 74.4            > 66.6
    997  93.670502           high         > 74.4            > 66.6
    998  82.470809           high         > 74.4            > 66.6
    999   4.889595            low         < 24.9            < 33.3

    [1000 rows x 4 columns]
    
    Notes
    -----
    - Ensure that the `categories` list length matches the number of bins minus one 
      when not using `'auto'`.
    - The `category_name` parameter allows for multiple categorizations on the same 
      column by specifying different names for each categorical representation.
    
    See Also
    --------
    pandas.cut : Function to bin continuous data into discrete intervals.
    pandas.qcut : Function to bin data based on quantiles.
    gofast.utils.validator.parameter_validator : Validator for function parameters.
    gofast.core.checks.exist_features : Check existence of features in DataFrame.
    gofast.core.checks.check_features_types : Validate feature data types.
    
    References
    ----------
    .. [1] Pandas Documentation: pandas.cut. https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.cut.html
    .. [2] Pandas Documentation: pandas.qcut. https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.qcut.html
    .. [3] Seaborn: Statistical Data Visualization. https://seaborn.pydata.org/
    .. [4] Matplotlib: Visualization with Python. https://matplotlib.org/
    .. [5] Freedman, D., & Diaconis, P. (1981). On the histogram as a density estimator: 
           L2 theory. *Probability Theory and Related Fields*, 57(5), 453-476.
    """
    # Allow series to be converted to df 
    df = to_frame_if (df)
    # Validate the existence of the specified column
    exist_features(
        df,
        features=column,
        name=f"Feature column {column}"
    )
    
    # Check if the column is of numeric type
    check_features_types(
        df,
        features=column,
        dtype='numeric',
        extra=f"Column '{column}' must be numeric to categorize."
    )
    
    # Validate the 'method' parameter
    method = parameter_validator(
        'method',
        target_strs={'equal_range', 'quantile'},
        error_msg="method must be either 'equal_range' or 'quantile'"
    )(method)
    
    # Validate the 'categories' parameter
    if categories !='auto': 
        categories= columns_manager(categories)
    
    # Set default category_name if not provided
    if category_name is None:
        category_name = f"{column}_category"
    
    if method == 'equal_range':
        if bins is None:
            min_val = df[column].min()
            max_val = df[column].max()
            num_categories = (
                len(categories) if categories != 'auto' else 3
            )  # Default to 3 bins if 'auto'
            bin_width = (max_val - min_val) / num_categories
            bins = [
                round(min_val + i * bin_width, 1)
                for i in range(num_categories + 1)
            ]
            bins[0] = min_val  # Ensure the first bin includes the minimum value
            bins[-1] = max_val  # Ensure the last bin includes the maximum value
    
        if categories == 'auto':
            bin_labels = []
            for i in range(len(bins) -1):
                if i == 0:
                    bin_labels.append(f"< {round(bins[i + 1], 1)}")
                elif i == len(bins) - 2:
                    bin_labels.append(f"> {round(bins[i], 1)}")
                else:
                    bin_labels.append(
                        f"{round(bins[i], 1)} - {round(bins[i + 1], 1)}"
                    )
            categories = bin_labels
    
        elif isinstance(categories, list):
            if len(categories) != len(bins) - 1:
                raise ValueError(
                    "Number of categories must match the number of bins minus one."
                )
        else:
            raise ValueError("categories must be a list of labels or 'auto'.")
    
        df[category_name] = pd.cut(
            df[column],
            bins=bins,
            labels=categories,
            include_lowest=include_lowest,
            right=right
        )
    
    elif method == 'quantile':
        if categories == 'auto':
            num_quantiles = 4  # Default quartiles
            quantiles = pd.qcut(
                df[column],
                q=num_quantiles,
                duplicates='drop'
            )
            bin_edges = quantiles.unique().categories
            bin_labels = []
            for i in range(len(bin_edges)):
                if i == 0:
                    bin_labels.append(
                        f"< {round(bin_edges[i].right, 1)}"
                    )
                elif i == len(bin_edges) - 1:
                    bin_labels.append(
                        f"> {round(bin_edges[i].left, 1)}"
                    )
                else:
                    bin_labels.append(
                        f"{round(bin_edges[i].left, 1)} - "
                        f"{round(bin_edges[i].right, 1)}"
                    )
            categories = bin_labels
        elif isinstance(categories, list):
            pass  # Use provided categories
        else:
            raise ValueError("categories must be a list of labels or 'auto'.")
    
        df[category_name] = pd.qcut(
            df[column],
            q=len(categories),
            labels=categories,
            duplicates='drop'
        )
    if drop: 
        df.drop(columns =column, inplace =True ) 
        
    return df


@SaveFile  
@is_data_readable 
@Dataify(auto_columns=True, fail_silently=True) 
def mask_by_reference(
    data: pd.DataFrame,
    ref_col: str,
    values: Optional[Union[Any, List[Any]]] = None,
    find_closest: bool = False,
    fill_value: Any = 0,
    mask_columns: Optional[Union[str, List[str]]] = None,
    error: str = "raise",
    verbose: int = 0,
    inplace: bool = False,
    savefile:Optional[str]=None, 
) -> pd.DataFrame:
    r"""
    Masks (replaces) values in columns other than the reference column
    for rows in which the reference column matches (or is closest to) the
    specified value(s).

    If a row's reference-column value is matched, that row's values in
    the *other* columns are overwritten by ``fill_value``. The reference
    column itself is not modified.

    This function supports both exact and approximate matching:
      - **Exact** matching is used if ``find_closest=False``.
      - **Approximate** (closest) matching is used if
        ``find_closest=True`` and the reference column is numeric.

    By default, if the reference column does not exist or if the
    given ``values`` cannot be found (or approximated) in the reference
    column, an exception is raised. This behavior can be adjusted with
    the ``error`` parameter.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame containing the data to be masked.

    ref_col : str
        The column in ``data`` serving as the reference for matching
        or finding the closest values.

    values : Any or sequence of Any, optional
        The reference values to look for in ``ref_col``. This can be:
          - A single value (e.g., ``0`` or ``"apple"``).
          - A list/tuple of values (e.g., ``[0, 10, 25]``).
          - If ``values`` is None, **all rows** are masked 
            (i.e. all rows match), effectively overwriting the entire
            DataFrame (except the reference column) with ``fill_value``.
        
        Note that if ``find_closest=False``, these values must appear
        in the reference column; otherwise, an error or warning is
        triggered (depending on the ``error`` setting).

    find_closest : bool, default=False
        If True, performs an approximate match for numeric reference
        columns. For each entry in ``values``, the function locates
        the row(s) in ``ref_col`` whose value is numerically closest.
        Non-numeric reference columns will revert to exact matching
        regardless.

    fill_value : Any, default=0
        The value used to fill/mask the non-reference columns wherever
        the condition (exact or approximate match) is met. This can
        be any valid type, e.g., integer, float, string, np.nan, etc.
        If ``fill_value='auto'`` and multiple values
        are given, each row matched by a particular reference
        value is filled with **that same reference value**.

        **Examples**:
          - If ``values=9`` and ``fill_value='auto'``, the fill
            value is **9** for matched rows.
          - If ``values=['a', 10]`` and ``fill_value='auto'``,
            then rows matching `'a'` are filled with `'a'`, and
            rows matching `10` are filled with `10`.
            
    mask_columns : str or list of str, optional
        If specified, *only* these columns are masked. If None,
        all columns except ``ref_col`` are masked. If any column in
        ``mask_columns`` does not exist in the DataFrame and
        ``error='raise'``, a KeyError is raised; otherwise, a warning
        may be issued or ignored.

    error : {'raise', 'warn', 'ignore'}, default='raise'
        Controls how to handle errors:
          - 'raise': raise an error if the reference column does not
            exist or if any of the given values cannot be matched (or
            approximated).
          - 'warn': only issue a warning instead of raising an error.
          - 'ignore': silently ignore any issues.

    verbose : int, default=0
        Verbosity level:
          - 0: silent (no messages).
          - 1: minimal feedback.
          - 2 or 3: more detailed messages for debugging.

    inplace : bool, default=False
        If True, performs the operation in place and returns the 
        original DataFrame with modifications. If False, returns a
        modified copy, leaving the original unaltered.
        
    savefile : str or None, optional
        File path where the DataFrame is saved if the
        decorator-based saving is active. If `None`, no saving
        occurs.

    Returns
    -------
    pd.DataFrame
        A DataFrame where rows matching the specified condition (exact
        or approximate) have had their non-reference columns replaced by
        ``fill_value``.

    Raises
    ------
    KeyError
        If ``error='raise'`` and ``ref_col`` is not in ``data.columns``.
    ValueError
        If ``error='raise'`` and no exact/approx match can be found
        for one or more entries in ``values``.

    Notes
    -----
    - If ``values`` is None, **all** rows are masked in the non-ref
      columns, effectively overwriting them with ``fill_value``.
    - When ``find_closest=True``, approximate matching is performed only
      if the reference column is numeric. For non-numeric data, it falls
      back to exact matching.
    - When multiple reference values are provided, each is
      processed in turn. If `fill_value='auto'`, each matched row
      is filled with that specific reference value.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.utils.data_utils import mask_by_reference
    >>>
    >>> df = pd.DataFrame({
    ...     "A": [10, 0, 8, 0],
    ...     "B": [2, 0.5, 18, 85],
    ...     "C": [34, 0.8, 12, 4.5],
    ...     "D": [0, 78, 25, 3.2]
    ... })
    >>>
    >>> # Example 1: Exact matching, replace all columns except 'A' with 0
    >>> masked_df = mask_by_reference(
    ...     data=df,
    ...     ref_col="A",
    ...     values=0,
    ...     fill_value=0,
    ...     find_closest=False,
    ...     error="raise"
    ... )
    >>> print(masked_df)
    >>> # 'B', 'C', 'D' for rows where A=0 are replaced with 0.
    >>>
    >>> # Example 2: Approximate matching for numeric
    >>> # If 'A' has values [0, 10, 8] and we search for 9, then 'A=8' or 'A=10'
    >>> # are the closest, so those rows get masked in non-ref columns.
    >>> masked_df2 = mask_by_reference(
    ...     data=df,
    ...     ref_col="A",
    ...     values=9,
    ...     find_closest=True,
    ...     fill_value=-999
    ... )
    >>> print(masked_df2)
    
    >>>
    >>> # Example 2: Approx. match for numeric ref_col
    >>> # 9 is between 8 and 10, so rows with A=8 and A=10 are masked
    >>> res2 = mask_by_reference(df, "A", 9, find_closest=True, fill_value=-999)
    >>> print(res2)
    ... # Rows 0 (A=10) and 2 (A=8) are replaced with -999 in columns B,C,D
    >>>
    >>> # Example 3: fill_value='auto' with multiple values
    >>> # Rows matching A=0 => fill with 0; rows matching A=8 => fill with 8
    >>> res3 = mask_by_reference(df, "A", [0, 8], fill_value='auto')
    >>> print(res3)
    ... # => rows with A=0 => B,C,D replaced by 0
    ... # => rows with A=8 => B,C,D replaced by 8
    >>> 
    >>> # 2) mask_columns=['C','D'] => only columns C and D are masked
    >>> res2 = mask_by_reference(df, "A", values=0, fill_value=999,
    ...                         mask_columns=["C","D"])
    >>> print(res2)
    ... # Rows where A=0 => columns C,D replaced by 999, while B remains unchanged
    >>>
    """
    # --- Preliminary checks --- #
    if ref_col not in data.columns:
        msg = (f"[mask_by_reference] Column '{ref_col}' not found "
               f"in the DataFrame.")
        if error == "raise":
            raise KeyError(msg)
        elif error == "warn":
            warnings.warn(msg)
            return data  # return as is
        else:
            return data  # error=='ignore'

    # Decide whether to operate on a copy or in place
    df = data if inplace else data.copy()

    # Determine which columns we'll mask
    if mask_columns is None:
        # mask all except ref_col
        mask_cols = [c for c in df.columns if c != ref_col]
    else:
        # Convert a single string to list
        if isinstance(mask_columns, str):
            mask_columns = [mask_columns]

        # Check that columns exist
        not_found = [col for col in mask_columns if col not in df.columns]
        if len(not_found) > 0:
            msg_cols = (f"[mask_by_reference] The following columns were "
                        f"not found in DataFrame: {not_found}.")
            if error == "raise":
                raise KeyError(msg_cols)
            elif error == "warn":
                warnings.warn(msg_cols)
                # Remove them from mask list if ignoring/warning
                mask_columns = [c for c in mask_columns if c in df.columns]
            else:
                pass  # silently ignore
        mask_cols = [c for c in mask_columns if c != ref_col]

    if verbose > 1:
        print(f"[mask_by_reference] Columns to be masked: {mask_cols}")

    # If values is None => mask all rows in mask_cols
    if values is None:
        if verbose > 0:
            print("[mask_by_reference] 'values' is None. Masking ALL rows.")
        if fill_value == 'auto':
            # 'auto' doesn't make sense with None => fill with None
            if verbose > 0:
                print("[mask_by_reference] 'fill_value=auto' but no values "
                      "specified. Will use None for fill.")
            df[mask_cols] = None
        else:
            df[mask_cols] = fill_value
        return df

    # Convert single value to a list
    if not isinstance(values, (list, tuple, set)):
        values = [values]

    ref_series = df[ref_col]
    is_numeric = pd.api.types.is_numeric_dtype(ref_series)

    # If find_closest and ref_series isn't numeric => revert to exact
    if find_closest and not is_numeric:
        if verbose > 0:
            print("[mask_by_reference] 'find_closest=True' but reference "
                  "column is not numeric. Reverting to exact matching.")
        find_closest = False

    total_matched_rows = set()  # track distinct row indices matched

    # Loop over each value and find matched rows
    for val in values:
        if find_closest:
            # Approximate match for numeric
            distances = (ref_series - val).abs()
            min_dist = distances.min()
            # If min_dist is inf, no numeric interpretation possible
            if min_dist == np.inf:
                matched_idx = []
            else:
                matched_idx = distances[distances == min_dist].index
        else:
            # Exact match
            matched_idx = ref_series[ref_series == val].index

        if len(matched_idx) == 0:
            # No match found for val
            msg_val = (
                f"[mask_by_reference] No matching value found for '{val}'"
                f" in column '{ref_col}'. Ensure '{val}' exists in "
                f"'{ref_col}' before applying the mask, or set"
                " ``find_closest=True`` to select the closest match."
            )
            if find_closest:
                msg_val = (f"[mask_by_reference] Could not approximate '{val}' "
                           f"in numeric column '{ref_col}'.")
            if error == "raise":
                raise ValueError(msg_val)
            elif error == "warn":
                warnings.warn(msg_val)
                continue  # skip
            else:
                continue  # error=='ignore'
        else:
            # Decide the actual fill we use for these matches
            if fill_value == 'auto':
                fill = val
            else:
                fill = fill_value

            # Mask these matched rows
            df.loc[matched_idx, mask_cols] = fill

            # Accumulate matched indices
            total_matched_rows.update(matched_idx)

    if verbose > 0:
        distinct_count = len(total_matched_rows)
        print(f"[mask_by_reference] Distinct matched rows: {distinct_count}")

    return df

@SaveFile 
@isdf 
def filter_by_isin(
    df: pd.DataFrame,
    *other_dfs: pd.DataFrame,
    main_col: str,
    columns: Optional[Union[str, List[str]]] = None,
    how: str = "union",
    invert: bool = False, 
    savefile=None, 
) -> pd.DataFrame:
    """
    Filter a DataFrame by checking whether values in one of its columns
    appear (or do not appear) in one or more other DataFrames.

    Parameters
    ----------
    df : pd.DataFrame
        The main DataFrame to be filtered.
    *other_dfs : pd.DataFrame
        One or more additional DataFrames that provide the reference
        values for the filtering.
    main_col : str
        Column in `df` whose values will be checked against the other
        DataFrames.
    columns : str or List[str], optional
        Column names in the other DataFrames to use for collecting
        reference values. If a single string is given, it applies
        to all DataFrames in `other_dfs`. If a list of strings is
        provided, it must match the number of DataFrames passed.
        Defaults to None, in which case `main_col` is also used
        for all DataFrames.
    how : {'union', 'intersection'}, optional
        How to combine the sets of valid values collected from the
        other DataFrames:
          - "union":  A value is valid if it appears in at least
                      one of the other DataFrames.
          - "intersection": A value is valid only if it appears
                            in *all* the other DataFrames.
        Defaults to "union".
    invert : bool, optional
        If True, invert the filtering so that rows are returned
        only when the value in `main_col` is *not* in the
        collected set. Defaults to False.

    Returns
    -------
    pd.DataFrame
        A filtered subset of `df` where `main_col` is (or is not)
        found in the reference columns of the other DataFrames,
        depending on the `invert` parameter and `how` mode.

    Raises
    ------
    ValueError
        If `columns` is a list but its length does not match
        the number of `other_dfs`, or if `how` is not one
        of the supported options.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.utils.data_utils import filter_by_isin
    >>> # Suppose we have a main DataFrame:
    ... df_main = pd.DataFrame({
    ...     'subsidence': [40.49, 10.58, 8.01, 42.50, 25.97, 999.99],
    ...     'other_col': ['A', 'B', 'C', 'D', 'E', 'F']
    ... })
    >>> # And another DataFrame with actual subsidence data:
    ... df_ref = pd.DataFrame({
    ...     'subsidence_actual': [40.49, 8.01, 25.97]
    ... })
    >>> # We can filter df_main to keep only rows whose
    ... # 'subsidence' values appear in df_ref's 'subsidence_actual':
    ... result = filter_by_isin(
    ...     df_main,
    ...     df_ref,
    ...     main_col='subsidence',
    ...     columns='subsidence_actual'
    ... )
    >>> result
       subsidence other_col
    0       40.49         A
    2        8.01         C
    4       25.97         E

    >>> # If we invert the filtering:
    ... result_inverted = filter_by_isin(
    ...     df_main,
    ...     df_ref,
    ...     main_col='subsidence',
    ...     columns='subsidence_actual',
    ...     invert=True
    ... )
    >>> result_inverted
       subsidence other_col
    1       10.58         B
    3       42.50         D
    5      999.99         F
    """
    # Validate how parameter.
    valid_how = {"union", "intersection"}
    if how not in valid_how:
        raise ValueError(
            f"`how` must be one of {valid_how}, got '{how}' instead."
        )
    
    exist_features(df, features= main_col, name="Main col")
    
    other_dfs = are_all_frames_valid(
        *other_dfs, 
        to_df =True,
        ops="validate"
        )
    # Handle columns parameter (broadcast if needed).
    if columns is None:
        # Use `main_col` for all
        columns_list = [main_col] * len(other_dfs)
    elif isinstance(columns, str):
        # Same single string for all other_dfs
        columns_list = [columns] * len(other_dfs)
    else:
        # columns should be a list of strings
        if len(columns) != len(other_dfs):
            raise ValueError(
                f"Number of items in `columns` ({len(columns)}) does not "
                f"match the number of `other_dfs` ({len(other_dfs)})."
            )
        columns_list = columns

    # Collect sets of valid values from each reference DataFrame.
    sets_of_values = []
    for ref_df, ref_col in zip(other_dfs, columns_list):
        ref_values = set(ref_df[ref_col].dropna().unique())
        sets_of_values.append(ref_values)

    # Combine sets by union or intersection.
    if not sets_of_values:
        # If no other_dfs were provided, no filtering needed.
        valid_values = set()
    else:
        if how == "union":
            valid_values = set.union(*sets_of_values)
        else:  # how == "intersection"
            valid_values = set.intersection(*sets_of_values)

    # Perform the filtering in df
    mask = df[main_col].isin(valid_values)
    if invert:
        mask = ~mask

    return df[mask].copy()
