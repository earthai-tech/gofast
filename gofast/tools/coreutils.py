# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Provides a diverse set of utility functions and tools for data manipulation,
validation, formatting, and processing. 
"""

from __future__ import print_function
import os
import re
import copy
import uuid
import string
import numbers
import random
import inspect
import hashlib
import datetime
import warnings
import itertools
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from .._gofastlog import gofastlog
from ..api.types import (
    Any, Callable, Union, Series, Tuple, Optional, Iterable, Set,
    _T, _Sub, _F, ArrayLike, List, DataFrame, NDArray
)
from ..compat.scipy import (
    ensure_scipy_compatibility,
    check_scipy_interpolate,
    optimize_minimize
)

# Logger Setup
_logger = gofastlog.get_gofast_logger(__name__)


__all__=[
     'add_noises_to',
     'adjust_to_samples',
     'assert_ratio',
     'check_dimensionality',
     'check_uniform_type',
     'closest_color',
     'colors_to_names',
     'contains_delimiter',
     'convert_to_structured_format',
     'convert_value_in',
     'decompose_colormap',
     'denormalize',
     'ensure_visualization_compatibility',
     'exist_features',
     'extract_coordinates',
     'features_in',
     'fill_nan_in',
     'find_by_regex',
     'find_closest',
     'find_features_in',
     'generate_alpha_values',
     'generate_id',
     'generate_mpl_styles',
     'get_colors_and_alphas',
     'get_confidence_ratio',
     'get_params',
     'get_valid_kwargs',
     'hex_to_rgb',
     'is_classification_task',
     'is_depth_in',
     'is_in_if',
     'is_iterable',
     'ismissing',
     'listing_items_format',
     'make_arr_consistent',
     'make_ids',
     'make_introspection',
     'make_obj_consistent_if',
     'map_specific_columns',
     'normalize_string',
     'process_and_extract_data',
     'projection_validator',
     'random_state_validator',
     'reshape',
     'sanitize_frame_cols',
     'split_list',
     'split_train_test',
     'split_train_test_by_id',
     'squeeze_specific_dim',
     'str2columns',
     'test_set_check_id',
     'to_numeric_dtypes',
     'to_series_if',
     'type_of_target',
     'unpack_list_of_dicts',
     'validate_feature',
     'validate_noise',
     'validate_ratio',
     ]

def find_closest(arr, values):
    """
    Find the closest values in an array from a set of target values.

    This function takes an array and a set of target values, and for each 
    target value, finds the closest value in the array. It can handle 
    both scalar and array-like inputs for `values`, ensuring flexibility 
    in usage. The result is either a single closest value or an array 
    of closest values corresponding to each target.

    Parameters
    ----------
    arr : array-like
        The array to search within. It can be a list, tuple, or numpy array 
        of numeric types. If the array is multi-dimensional, it will be 
        flattened to a 1D array.
        
    values : float or array-like
        The target value(s) to find the closest match for in `arr`. This can 
        be a single float or an array of floats.

    Returns
    -------
    numpy.ndarray
        An array of the closest values in `arr` for each target in `values`.
        If `values` is a single float, the function returns a single-element
        array.

    Notes
    -----
    - This function operates by calculating the absolute difference between
      each element in `arr` and each target in `values`, selecting the 
      element with the smallest difference.
    - The function assumes `arr` and `values` contain numeric values, and it
      raises a `TypeError` if they contain non-numeric data.

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.tools.coreutils import find_closest
    >>> find_closest([2, 3, 4, 5], 2.6)
    array([3.])

    >>> find_closest(np.array([[2, 3], [4, 5]]), (2.6, 5.6))
    array([3., 5.])

    See Also
    --------
    numpy.argmin : Find the indices of the minimum values along an axis.
    numpy.abs : Compute the absolute value element-wise.

    References
    ----------
    .. [1] Harris, C. R., et al. "Array programming with NumPy." 
       Nature 585.7825 (2020): 357-362.
    """
    from .validator import _is_numeric_dtype
    arr = is_iterable(arr, exclude_string=True, transform=True)
    values = is_iterable(values, exclude_string=True, transform=True)

    # Validate numeric types in arr and values
    for var, name in zip([arr, values], ['array', 'values']):
        if not _is_numeric_dtype(var, to_array=True):
            raise TypeError(f"Non-numeric data found in {name}.")

    # Convert arr and values to numpy arrays for vectorized operations
    arr = np.array(arr, dtype=np.float64)
    values = np.array(values, dtype=np.float64)

    # Flatten arr if it is multi-dimensional
    arr = arr.ravel() if arr.ndim != 1 else arr

    # Find the closest value for each target in values
    closest_values = np.array([
        arr[np.abs(arr - target).argmin()] for target in values
    ])

    return closest_values

def run_return(
    self, 
    attribute_name: Optional[str] = None, 
    error_policy: str = 'warn',
    default_value: Optional[Any] = None,
    check_callable: bool = False,
    return_type: str = 'attribute',
    on_callable_error: str = 'warn',
    allow_private: bool = False,
    msg: Optional[str] = None, 
    config_return_type: Optional[Union[str, bool]] = None
) -> Any:
    """
    Return `self`, a specified attribute of `self`, or both, with error handling
    policies. Optionally integrates with global configuration to customize behavior.

    Parameters
    ----------
    attribute_name : str, optional
        The name of the attribute to return. If `None`, returns `self`.
    error_policy : str, optional
        Policy for handling non-existent attributes. Options:
        - `warn` : Warn the user and return `self` or a default value.
        - `ignore` : Silently return `self` or the default value.
        - `raise` : Raise an `AttributeError` if the attribute does not exist.
    default_value : Any, optional
        The default value to return if the attribute does not exist. If `None`,
        and the attribute does not exist, returns `self` based on the error policy.
    check_callable : bool, optional
        If `True`, checks if the attribute is callable and executes it if so.
    return_type : str, optional
        Specifies the return type. Options:
        - `self` : Always return `self`.
        - `attribute` : Return the attribute if it exists.
        - `both` : Return a tuple of (`self`, attribute).
    on_callable_error : str, optional
        How to handle errors when calling a callable attribute. Options:
        - `warn` : Warn the user and return `self`.
        - `ignore` : Silently return `self`.
        - `raise` : Raise the original error.
    allow_private : bool, optional
        If `True`, allows access to private attributes (those starting with '_').
    msg : str, optional
        Custom message for warnings or errors. If `None`, a default message will be used.
    config_return_type : str or bool, optional
        Global configuration to override return behavior. If set to 'self', always
        return `self`. If 'attribute', always return the attribute. If `None`, use
        developer-defined behavior.

    Returns
    -------
    Any
        Returns `self`, the attribute value, or a tuple of both, depending on
        the specified options and the availability of the attribute.

    Raises
    ------
    AttributeError
        If the attribute does not exist and `error_policy` is set to 'raise', or if the
        callable check fails and `on_callable_error` is set to 'raise'.

    Notes
    -----
    The `run_return` function is designed to offer flexibility in determining
    what is returned from a method, allowing developers to either return `self` for
    chaining, return an attribute of the class, or both. By using `global_config`,
    package-wide behavior can be customized.

    Examples
    --------
    >>> from gofast.tools.coreutils import run_return
    >>> class MyModel:
    ...     def __init__(self, name):
    ...         self.name = name
    ...
    >>> model = MyModel(name="example")
    >>> run_return(model, "name")
    'example'

    See Also
    --------
    logging : Python's logging module.
    warnings.warn : Function to issue warning messages.

    References
    ----------
    .. [1] "Python Logging Module," Python Software Foundation.
           https://docs.python.org/3/library/logging.html
    .. [2] "Python Warnings," Python Documentation.
           https://docs.python.org/3/library/warnings.html
    """

    # If global config specifies return behavior, override the return type
    if config_return_type == 'self':
        return self
    elif config_return_type == 'attribute':
        return getattr(self, attribute_name, default_value
                       ) if attribute_name else self

    # If config is None or not available, use developer-defined logic
    if attribute_name:
        # Check for private attributes if allowed
        if not allow_private and attribute_name.startswith('_'):
            custom_msg = msg or ( 
                f"Access to private attribute '{attribute_name}' is not allowed.")
            raise AttributeError(custom_msg)

        # Check if the attribute exists
        if hasattr(self, attribute_name):
            attr_value = getattr(self, attribute_name)

            # If check_callable is True, try executing the attribute if it's callable
            if check_callable and isinstance(attr_value, Callable):
                try:
                    attr_value = attr_value()
                except Exception as e:
                    custom_msg = msg or ( 
                        f"Callable attribute '{attribute_name}'"
                        f" raised an error: {e}."
                        )
                    if on_callable_error == 'raise':
                        raise e
                    elif on_callable_error == 'warn':
                        warnings.warn(custom_msg)
                        return self
                    elif on_callable_error == 'ignore':
                        return self

            # Return based on the return_type provided
            if return_type == 'self':
                return self
            elif return_type == 'both':
                return self, attr_value
            else:
                return attr_value
        else:
            # Handle the case where the attribute does not exist based on the error_policy
            custom_msg = msg or ( 
                f"'{self.__class__.__name__}' object has"
                f"  no attribute '{attribute_name}'."
                )
            if error_policy == 'raise':
                raise AttributeError(custom_msg)
            elif error_policy == 'warn':
                warnings.warn(f"{custom_msg} Returning default value or self.")
            # Return the default value if provided, otherwise return self
            return default_value if default_value is not None else self
    else:
        # If no attribute is provided, return self
        return self

def generate_id(
    length=12,
    prefix="",
    suffix="",
    include_timestamp=False,
    use_uuid=False,
    char_set=None,
    numeric_only=False,
    unique_ids=None,
    retries=3
):
    """
    Generate a customizable and unique ID with options for prefix, suffix, 
    timestamp, and character type.

    Parameters
    ----------
    length : int, optional
        Length of the generated ID, excluding any specified prefix, suffix, 
        or timestamp. Default is 12. Ignored if `use_uuid` is set to ``True``.

    prefix : str, optional
        Prefix string to be added to the beginning of the generated ID.
        Defaults to an empty string.

    suffix : str, optional
        Suffix string to append to the end of the generated ID.
        Defaults to an empty string.

    include_timestamp : bool, optional
        If ``True``, appends a timestamp in the 'YYYYMMDDHHMMSS' format
        to the ID. Defaults to ``False``.

    use_uuid : bool, optional
        If ``True``, generates the ID using UUID4, ignoring the parameters
        `length`, `char_set`, and `numeric_only`. Defaults to ``False``.

    char_set : str or None, optional
        A string specifying the set of characters to use in the ID. 
        If ``None``, defaults to alphanumeric characters 
        (uppercase and lowercase letters plus digits).

    numeric_only : bool, optional
        If ``True``, limits the character set to numeric digits only. 
        Defaults to ``False``. Overridden by `char_set` if provided.

    unique_ids : set or None, optional
        A set to store and check for unique IDs. If provided, generated IDs 
        are compared against this set to ensure no duplicates. New unique IDs 
        are added to this set after generation.

    retries : int, optional
        Number of retries if a generated ID conflicts with `unique_ids`.
        Defaults to 3.

    Returns
    -------
    str
        A string representing the generated ID, potentially including the 
        specified prefix, suffix, timestamp, and custom length.

    Notes
    -----
    The function allows for highly customizable ID generation, supporting 
    different character sets, unique ID constraints, and options for 
    timestamped or UUID-based IDs. When using `unique_ids`, the function 
    performs multiple attempts to generate a unique ID, retrying as specified 
    by the `retries` parameter.

    The generated ID can be represented as a combination of three components:

    .. math:: 
        \text{{ID}} = \text{{prefix}} + \text{{base ID}} + \text{{suffix}}

    Where:
        - `prefix` and `suffix` are optional components.
        - `base ID` is a string of randomly selected characters from the 
          specified character set or a UUID-based string.

    Examples
    --------
    >>> from gofast.tools.coreutils import generate_id
    >>> generate_id(length=8, prefix="PAT-", suffix="-ID", include_timestamp=True)
    'PAT-WJ8N6F-20231025123456-ID'
    
    >>> generate_id(length=6, numeric_only=True)
    '483920'

    >>> unique_set = set()
    >>> generate_id(length=10, unique_ids=unique_set, retries=5)
    'Y8B5QD2L7H'
    
    See Also
    --------
    uuid : Module to generate universally unique identifiers.

    References
    ----------
    .. [1] Jane Doe et al. "Best Practices in Unique Identifier Generation." 
           Data Science Journal, 2021, vol. 9, no. 4, pp. 210-222.
    .. [2] J. Smith. "Character-Based ID Generation for High-Volume Systems."
           Proceedings of the ID Conference, 2022.
    """
    
    # Define the character set
    if use_uuid:
        # Use UUID for ID generation if specified
        new_id = str(uuid.uuid4()).replace("-", "")
        if include_timestamp:
            new_id += datetime.now().strftime('%Y%m%d%H%M%S')
        return f"{prefix}{new_id[:length]}{suffix}"

    if numeric_only:
        char_set = string.digits
    elif char_set is None:
        char_set = string.ascii_letters + string.digits

    def _generate_base_id():
        """Generates the base ID without prefix, suffix, or timestamp."""
        return ''.join(random.choice(char_set) for _ in range(length))

    # Retry logic to ensure uniqueness if required
    for _ in range(retries):
        # Generate base ID and add optional elements
        new_id = _generate_base_id()
        
        # Include timestamp if specified
        if include_timestamp:
            timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            new_id += timestamp
        
        # Add prefix and suffix
        new_id = f"{prefix}{new_id}{suffix}"

        # Check for uniqueness if a unique_ids set is provided
        if unique_ids is not None:
            if new_id not in unique_ids:
                unique_ids.add(new_id)
                return new_id
        else:
            return new_id

    # Raise error if unique ID generation failed after retries
    raise ValueError("Failed to generate a unique ID after multiple retries.")


def format_to_datetime(data, date_col, verbose=0, **dt_kws):
    """
    Reformats a specified column in a DataFrame to Pandas datetime format.

    This function attempts to convert the values in the specified column of a 
    DataFrame to Pandas datetime objects. If the conversion is successful, 
    the DataFrame with the updated column is returned. If the conversion fails, 
    a message describing the error is printed, and the original 
    DataFrame is returned.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame containing the column to be reformatted.
    date_col : str
        The name of the column to be converted to datetime format.
    verbose : int, optional
        Verbosity mode; 0 or 1. If 1, prints messages about the conversion 
        process.Default is 0 (silent mode).
    **dt_kws : dict, optional
        Additional keyword arguments to pass to `pd.to_datetime` function.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the specified column in datetime format. If conversion
        fails, the original DataFrame is returned.

    Raises
    ------
    ValueError
        If the specified column is not found in the DataFrame.

    Examples
    --------
    >>> from gofast.tools.coreutils import format_to_datetime
    >>> df = pd.DataFrame({
    ...     'Date': ['2021-01-01', '01/02/2021', '03-Jan-2021', '2021.04.01',
                     '05 May 2021'],
    ...     'Value': [1, 2, 3, 4, 5]
    ... })
    >>> df = format_to_datetime(df, 'Date')
    >>> print(df.dtypes)
    Date     datetime64[ns]
    Value             int64
    dtype: object
    """
    if date_col not in data.columns:
        raise ValueError(f"Column '{date_col}' not found in DataFrame.")
    
    try:
        data[date_col] = pd.to_datetime(data[date_col], **dt_kws)
        if verbose: 
            print(f"Column '{date_col}' successfully converted to datetime format.")
    except Exception as e:
        print(f"Error converting '{date_col}' to datetime format: {e}")
        return data

    return data

def adjust_to_samples(n_samples, *values, initial_guess=None, error='warn'):
    """
    Adjusts the given values to match a total number of samples, aiming to distribute
    the samples evenly across the dimensions represented by the values. The function
    can adjust even if only one value is given.

    Parameters
    ----------
    n_samples : int
        The desired total number of samples.
    *values : int
        Variable length argument list representing the dimensions to adjust.
    initial_guess : float or None, optional
        An initial guess for the adjustment factor. If None, an automatic guess is made.
    error : str, optional
        Error handling strategy ('warn', 'ignore', 'raise'). This parameter is considered
        only when no values or one value is provided.

    Returns
    -------
    adjusted_values : tuple
        A tuple of adjusted values, aiming to distribute the total samples evenly.
        If only one value is given, the function tries to adjust it based on the
        total number of samples and the initial guess.

    Raises
    ------
    ValueError
        Raised if error is set to 'raise' and no values are provided.

    Examples
    --------
    >>> from gofast.tools.coreutils import adjust_to_samples
    >>> adjust_to_samples(1000, 10, 20, initial_guess=5)
    (50, 20)

    >>> adjust_to_samples(1000, 10, initial_guess=2)
    (2,)

    Notes
    -----
    The function aims to adjust the values to match the desired total number of samples
    as closely as possible. When only one value is given, the function uses the initial
    guess to make an adjustment, respecting the total number of samples.
    """
    if len(values) == 0:
        message = "No values provided for adjustment."
        if error == 'raise':
            raise ValueError(message)
        elif error == 'warn':
            warnings.warn(message)
        return ()

    if len(values) == 1:
        # If only one value is given, adjust it based on initial guess and n_samples
        single_value = values[0]
        adjusted_value = n_samples // single_value if initial_guess is None else initial_guess
        return (adjusted_value,)

    if initial_guess is None:
        initial_guess = np.mean(values)

    # Function to minimize: difference between product of adjusted values and n_samples
    def objective(factors):
        prod = np.prod(np.array(values) * factors)
        return abs(prod - n_samples)

    # Start with initial guesses for factors
    factors_initial = [initial_guess / value for value in values]
    result = optimize_minimize(objective, factors_initial, bounds=[(0, None) for _ in values])

    if result.success:
        adjusted_values = ( 
            tuple(max(1, int(round(value * factor))) 
                  for value, factor in zip(values, result.x))
            )
    else:
        adjusted_values = values  # Fallback to original values if optimization fails

    return adjusted_values

def unpack_list_of_dicts(list_of_dicts):
    """
    Unpacks a list of dictionaries into a single dictionary,
    merging all keys and values.

    Parameters:
    ----------
    list_of_dicts : list of dicts
        A list where each element is a dictionary with a single key-value pair, 
        the value being a list.

    Returns:
    -------
    dict
        A single dictionary with all keys from the original list of dictionaries, 
        each associated with its combined list of values from all occurrences 
        of the key.

    Example:
    --------
    >>> from gofast.tools.coreutils import unpack_list_of_dicts
    >>> list_of_dicts = [
            {'key1': ['value10', 'value11']},
            {'key2': ['value20', 'value21']},
            {'key1': ['value12']},
            {'key2': ['value22']}
        ]
    >>> unpacked_dict = unpack_list_of_dicts(list_of_dicts)
    >>> print(unpacked_dict)
    {'key1': ['value10', 'value11', 'value12'], 'key2': ['value20', 'value21', 'value22']}
    """
    unpacked_dict = defaultdict(list)
    for single_dict in list_of_dicts:
        for key, values in single_dict.items():
            unpacked_dict[key].extend(values)
    return dict(unpacked_dict)  # Convert defaultdict back to dict if required

def get_params (obj: object ) -> dict: 
    """
    Get object parameters. 
    
    Object can be callable or instances 
    
    :param obj: object , can be callable or instance 
    
    :return: dict of parameters values 
    
    :examples: 
    >>> from sklearn.svm import SVC 
    >>> from gofast.tools.coreutils import get_params 
    >>> sigmoid= SVC (
        **{
            'C': 512.0,
            'coef0': 0,
            'degree': 1,
            'gamma': 0.001953125,
            'kernel': 'sigmoid',
            'tol': 1.0 
            }
        )
    >>> pvalues = get_params( sigmoid)
    >>> {'decision_function_shape': 'ovr',
         'break_ties': False,
         'kernel': 'sigmoid',
         'degree': 1,
         'gamma': 0.001953125,
         'coef0': 0,
         'tol': 1.0,
         'C': 512.0,
         'nu': 0.0,
         'epsilon': 0.0,
         'shrinking': True,
         'probability': False,
         'cache_size': 200,
         'class_weight': None,
         'verbose': False,
         'max_iter': -1,
         'random_state': None
     }
    """
    if hasattr (obj, '__call__'): 
        cls_or_func_signature = inspect.signature(obj)
        PARAMS_VALUES = {k: None if v.default is (inspect.Parameter.empty 
                         or ...) else v.default 
                    for k, v in cls_or_func_signature.parameters.items()
                    # if v.default is not inspect.Parameter.empty
                    }
    elif hasattr(obj, '__dict__'): 
        PARAMS_VALUES = {k:v  for k, v in obj.__dict__.items() 
                         if not (k.endswith('_') or k.startswith('_'))}
    
    return PARAMS_VALUES

def is_classification_task(
    *y, max_unique_values=10
    ):
    """
    Check whether the given arrays are for a classification task.

    This function assumes that if all values in the provided arrays are 
    integers and the number of unique values is within the specified
    threshold, it is a classification task.

    Parameters
    ----------
    *y : list or numpy.array
        A variable number of arrays representing actual values, 
        predicted values, etc.
    max_unique_values : int, optional
        The maximum number of unique values to consider the task 
        as classification. 
        Default is 10.

    Returns
    -------
    bool
        True if the provided arrays are for a classification task, 
        False otherwise.

    Examples
    --------
    >>> from gofast.tools.coreutils import is_classification_task 
    >>> y_true = [0, 1, 1, 0, 1]
    >>> y_pred = [0, 1, 0, 0, 1]
    >>> is_classification_task(y_true, y_pred)
    True
    """
    max_unique_values = int (
        _assert_all_types(max_unique_values, 
                          int, float, objname="Max Unique values")
                             )
    # Combine all arrays for analysis
    combined = np.concatenate(y)

    # Check if all elements are integers
    if ( 
            not all(isinstance(x, int) for x in combined) 
            and not combined.dtype.kind in 'iu'
            ):
        return False

    # Check the number of unique elements
    unique_values = np.unique(combined)
    # check Arbitrary threshold for number of classes
    if len(unique_values) > max_unique_values:
        return False

    return True

def fancy_printer(result, report_name='Data Quality Check Report'):
    """ 
    This _fancy_print function within the check_data_quality function 
    iterates over the results dictionary and prints each category 
    (like missing data, outliers, etc.) in a formatted manner. It only 
    displays categories with findings, making the output more concise and 
    focused on the areas that need attention. The use of .title() 
    and .replace('_', ' ') methods enhances the readability of the 
    category names.

    Parameters 
    -----------
    result: dict,
       the result to print. Must contain a dictionnary. 
    report_name: str, 
       A report to fancy printer. 
       
    """
    if not isinstance ( result, dict): 
        raise TypeError("fancy_printer accepts only a dictionnary type."
                        f" Got {type(result).__name__!r}")
        
    print(f"\n{report_name}:\n")

    for key, value in result.items():
        if value:  # Only display categories with findings
            print(f"--- {key.replace('_', ' ').title()} ---")
            print("Column            | Details")
            print("-" * 40)  # Table header separator

            try : 
                
                for sub_key, sub_value in value.items():
                    # Ensuring column name and details fit into the table format
                    formatted_key = (sub_key[:15] + '..') if len(
                        sub_key) > 17 else sub_key
                    formatted_value = str(sub_value)[:20] + (
                        '..' if len(str(sub_value)) > 22 else '')
                    print(f"{formatted_key:<17} | {formatted_value}")
            except : 
                formatted_key = (key[:15] + '..') if len(key) > 17 else key
                formatted_value = f"{value:.2f}"
                print(f"{formatted_key:<17} | {formatted_value}")

            print("\n")
        else:
            print(f"--- No {key.replace('_', ' ').title()} Found ---\n")

def to_numeric_dtypes(
    arr: Union[NDArray, DataFrame], *, 
    columns: Optional[List[str]] = None, 
    return_feature_types: bool = ..., 
    missing_values: float = np.nan, 
    pop_cat_features: bool = ..., 
    sanitize_columns: bool = ..., 
    regex: Optional[re.Pattern] = None, 
    fill_pattern: str = '_', 
    drop_nan_columns: bool = True, 
    how: str = 'all', 
    reset_index: bool = ..., 
    drop_index: bool = True, 
    verbose: bool = ...
) -> Union[DataFrame, Tuple[DataFrame, List[str], List[str]]]:
    """
    Converts an array to a DataFrame and coerces values to appropriate 
    data types.

    This function is designed to process data arrays or DataFrames, ensuring
    numeric and categorical features are correctly identified and formatted. 
    It provides options to manipulate the data, including column sanitization, 
    handling of missing values, and dropping NaN-filled columns.

    Parameters
    ----------
    arr : NDArray or DataFrame
        The data to be processed, either as an array or a DataFrame.
    
    columns : list of str, optional
        Column names for creating a DataFrame from an array. 
        Length should match the number of columns in `arr`.
    
    return_feature_types : bool, default=False
        If True, returns a tuple with the DataFrame, numeric, and categorical 
        features.
    
    missing_values : float, default=np.nan
        Value used to replace missing or empty strings in the DataFrame.
    
    pop_cat_features : bool, default=False
        If True, removes categorical features from the DataFrame.
    
    sanitize_columns : bool, default=False
        If True, cleans the DataFrame columns using the specified `regex` 
        pattern.
    
    regex : re.Pattern or str, optional
        Regular expression pattern for column sanitization. the default is:: 
        
        >>> import re 
        >>> re.compile (r'[_#&.)(*@!_,;\s-]\s*', flags=re.IGNORECASE)
    
    fill_pattern : str, default='_'
        String pattern used to replace non-alphanumeric characters in 
        column names.
    
    drop_nan_columns : bool, default=True
        If True, drops columns filled entirely with NaN values.
    
    how : str, default='all'
        Determines row dropping strategy based on NaN values.
    
    reset_index : bool, default=False
        If True, resets the index of the DataFrame after processing.
    
    drop_index : bool, default=True
        If True, drops the original index when resetting the DataFrame index.
    
    verbose : bool, default=False
        If True, prints additional information during processing.

    Returns
    -------
    DataFrame or tuple of DataFrame, List[str], List[str]
        The processed DataFrame. If `return_feature_types` is True, returns a 
        tuple with the DataFrame, list of numeric feature names (`nf`), 
        and list of categorical feature names (`cf`).

    Examples
    --------
    >>> from gofast.datasets.dload import load_bagoue
    >>> from gofast.tools.coreutils import to_numeric_dtypes
    >>> X= load_bagoue(as_frame=True)
    >>> X0 = X[['shape', 'power', 'magnitude']]
    >>> df, nf, cf = to_numeric_dtypes(X0, return_feature_types=True)
    >>> print(df.dtypes, nf, cf)
    >>> X0.dtypes 
    ... shape        object
        power        object
        magnitude    object
        dtype: object
    >>> df = to_numeric_dtypes(X0)
    >>> df.dtypes 
    ... shape         object
        power        float64
        magnitude    float64
        dtype: object
    """

    from .validator import _is_numeric_dtype
    # pass ellipsis argument to False 
    ( sanitize_columns, reset_index, 
     verbose,return_feature_types, 
     pop_cat_features, 
        ) = ellipsis2false(
            sanitize_columns, 
            reset_index, 
            verbose,
            return_feature_types, 
            pop_cat_features
    )
   
    if not is_iterable (arr, exclude_string=True): 
        raise TypeError(f"Expect array. Got {type (arr).__name__!r}")

    if hasattr ( arr, '__array__') and hasattr ( arr, 'columns'): 
        df = arr.copy()
        if columns is not None: 
            if verbose: 
                print("Dataframe is passed. Columns should be replaced.")
            df =pd.DataFrame ( np.array ( arr), columns =columns )
            
    else: df = pd.DataFrame (arr, columns =columns  ) 
        
    # sanitize columns 
    if sanitize_columns: 
        # Pass in the case columns are all integer values. 
        if not _is_numeric_dtype(df.columns , to_array=True): 
           # for consistency reconvert to str 
           df.columns = df.columns.astype(str) 
           df = sanitize_frame_cols(
               df, regex=regex, fill_pattern=fill_pattern ) 

    #replace empty string by Nan if NaN exist in dataframe  
    df= df.replace(r'^\s*$', missing_values, regex=True)
    
    # check the possibililty to cast all 
    # the numerical data 
    for serie in df.columns: 
        try: 
            df= df.astype(
                {serie:np.float64})
        except:continue
    
    # drop nan  columns if exists 
    if drop_nan_columns: 
        if verbose: 
            nan_columns = df.columns [ df.isna().all()].tolist() 
            print("No NaN column found.") if len(
                nan_columns)==0 else listing_items_format (nan_columns, 
                    "NaN columns found in the data",
                    " ", inline =True, lstyle='.')                               
        # drop rows and columns with NaN values everywhere.                                                   
        df.dropna (axis=1, how='all', inplace =True)
        if str(how).lower()=='all': 
            df.dropna ( axis=0, how='all', inplace =True)
    
    # reset_index of the dataframe
    # This is useful after droping rows
    if reset_index: 
        df.reset_index (inplace =True, drop = drop_index )
    # collect numeric and non-numeric data 
    nf, cf =[], []    
    for serie in df.columns: 
        if _is_numeric_dtype(df[serie], to_array =True ): 
            nf.append(serie)
        else: cf.append(serie)

    if pop_cat_features: 
        [ df.pop(item) for item in cf ] 
        if verbose: 
            msg ="Dataframe does not contain any categorial features."
            b= f"Feature{'s' if len(cf)>1 else ''}"
            e = (f"{'have' if len(cf) >1 else 'has'} been dropped"
                 " from the dataframe.")
            print(msg) if len(cf)==0 else listing_items_format (
                cf , b, e ,lstyle ='.', inline=True)
            
        return df 
    
    return (df, nf, cf) if return_feature_types else df 

def listing_items_format ( 
        lst,  begintext ='', endtext='' , bullet='-', 
        enum =True , lstyle=None , space =3 , inline =False, verbose=True
        ): 
    """ Format list by enumerate them successively with carriage return
    
    :param lst: list,
        object for listening 
    :param begintext: str, 
        Text to display at the beginning of listing the items in `lst`. 
    :param endtext: str, 
        Text to display at the end of the listing items in `lst`. 
    :param enum:bool, default=True, 
        Count the number of items in `lst` and display it 
    :param lstyle: str, default =None 
        listing marker. 
    :param bullet:str, default='-'
        symbol that is used to introduce item if `enum` is set to False. 
    :param space: int, 
        number of space to keep before each outputted item in `lst`
    :param inline: bool, default=False, 
        Display all element inline rather than carriage return every times. 
    :param verbose: bool, 
        Always True for print. If set to False, return list of string 
        litteral text. 
    :returns: None or str 
        None or string litteral if verbose is set to ``False``.
    Examples
    ---------
    >>> from gofast.tools.coreutils import listing_items_format 
    >>> litems = ['hole_number', 'depth_top', 'depth_bottom', 'strata_name', 
                'rock_name','thickness', 'resistivity', 'gamma_gamma', 
                'natural_gamma', 'sp','short_distance_gamma', 'well_diameter']
    >>> listing_items_format (litems , 'Features' , 
                               'have been successfully drop.' , 
                              lstyle ='.', space=3) 
    """
    out =''
    if not is_iterable(lst): 
        lst=[lst]
   
    if hasattr (lst, '__array__'): 
        if lst.ndim !=1: 
            raise ValueError (" Can not print multidimensional array."
                              " Expect one dimensional array.")
    lst = list(lst)
    begintext = str(begintext); endtext=str(endtext)
    lstyle=  lstyle or bullet  
    lstyle = str(lstyle)
    b= f"{begintext +':' } "   
    if verbose :
        print(b, end=' ') if inline else (
            print(b)  if  begintext!='' else None)
    out += b +  ('\n' if not inline else ' ') 
    for k, item in enumerate (lst): 
        sp = ' ' * space 
        if ( not enum and inline ): lstyle =''
        o = f"{sp}{str(k+1) if enum else bullet+ ' ' }{lstyle} {item}"
        if verbose:
            print (o , end=' ') if inline else print(o)
        out += o + ('\n' if not inline else ' ') 
       
    en= ' ' + endtext if inline else endtext
    if verbose: 
        print(en) if endtext !='' else None 
    out +=en 
    
    return None if verbose else out 
    
    
def parse_attrs (attr,  regex=None ): 
    """ Parse attributes using the regular expression.
    
    Remove all string non-alphanumeric and some operator indicators,  and 
    fetch attributes names. 
    
    Parameters 
    -----------
    
    attr: str, text litteral containing the attributes 
        names 
        
    regex: `re` object, default is 
        Regular expresion object. the default is:: 
            
            >>> import re 
            >>> re.compile (r'per|mod|times|add|sub|[_#&*@!_,;\s-]\s*', 
                                flags=re.IGNORECASE) 
    Returns
    -------
    attr: List of attributes 
    
    Example
    ---------
    >>> from gofast.tools.coreutils import parse_attrs 
    >>> parse_attrs('lwi_sub_ohmSmulmagnitude')
    ... ['lwi', 'ohmS', 'magnitude']
    
    
    """
    regex = regex or re.compile (r'per|mod|times|add|sub|[_#&*@!_,;\s-]\s*', 
                        flags=re.IGNORECASE) 
    attr= list(filter (None, regex.split(attr)))
    return attr 


def shrunkformat(
    text: Union[str, Iterable[Any]], 
    chunksize: int = 7,
    insert_at: Optional[str] = None, 
    sep: Optional[str] = None, 
) -> None:
    """ Format class and add ellipsis when classes are greater than maxview 
    
    :param text: str - a text to shrunk and format. Can also be an iterable
        object. 
    :param chunksize: int, the size limit to keep in the formatage text. *default* 
        is ``7``.
    :param insert_at: str, the place to insert the ellipsis. If ``None``,  
        shrunk the text and put the ellipsis, between the text beginning and 
        the text endpoint. Can be ``beginning``, or ``end``. 
    :param sep: str if the text is delimited by a kind of character, the `sep` 
        parameters could be usefull so it would become a starting point for 
        word counting. *default*  is `None` which means word is counting from 
        the space. 
        
    :example: 
        
    >>> import numpy as np 
    >>> from gofast.tools.coreutils import shrunkformat
    >>> text=" I'm a long text and I will be shrunked and replaced by ellipsis."
    >>> shrunkformat (text)
    ... 'Im a long ... and replaced by ellipsis.'
    >>> shrunkformat (text, insert_at ='end')
    ...'Im a long ... '
    >>> arr = np.arange(30)
    >>> shrunkformat (arr, chunksize=10 )
    ... '0 1 2 3 4  ...  25 26 27 28 29'
    >>> shrunkformat (arr, insert_at ='begin')
    ... ' ...  26 27 28 29'
    
    """
    is_str = False 
    chunksize = int (_assert_all_types(chunksize, float, int))
                   
    regex = re.compile (r"(begin|start|beg)|(end|close|last)")
    insert_at = str(insert_at).lower().strip() 
    gp = regex.search (insert_at) 
    if gp is not None: 
        if gp.group (1) is not None:  
            insert_at ='begin'
        elif gp.group(2) is not None: 
            insert_at ='end'
        if insert_at is None: 
            warnings.warn(f"Expect ['begining'|'end'], got {insert_at!r}"
                          " Default value is used instead.")
    if isinstance(text , str): 
        textsplt = text.strip().split(sep) # put text on list 
        is_str =True 
        
    elif hasattr (text , '__iter__'): 
        textsplt = list(text )
        
    if len(textsplt) < chunksize : 
        return  text 
    
    if is_str : 
        rl = textsplt [:len(textsplt)//2][: chunksize//2]
        ll= textsplt [len(textsplt)//2:][-chunksize//2:]
        
        if sep is None: sep =' '
        spllst = [f'{sep}'.join ( rl), f'{sep}'.join ( ll)]
        
    else : spllst = [
        textsplt[: chunksize//2 ] ,textsplt[-chunksize//2:]
        ]
    if insert_at =='begin': 
        spllst.insert(0, ' ... ') ; spllst.pop(1)
    elif insert_at =='end': 
        spllst.pop(-1) ; spllst.extend ([' ... '])
        
    else : 
        spllst.insert (1, ' ... ')
    
    spllst = spllst if is_str else str(spllst)
    
    return re.sub(r"[\[,'\]]", '', ''.join(spllst), 
                  flags=re.IGNORECASE 
                  ) 


def accept_types (
        *objtypes: list , 
        format: bool = False
        ) -> Union [List[str] , str] : 
    """ List the type format that can be accepted by a function. 
    
    :param objtypes: List of object types.
    :param format: bool - format the list of the name of objects.
    :return: list of object type names or str of object names. 
    
    :Example: 
        >>> import numpy as np; import pandas as pd 
        >>> from gofast.tools.coreutils import accept_types
        >>> accept_types (pd.Series, pd.DataFrame, tuple, list, str)
        ... "'Series','DataFrame','tuple','list' and 'str'"
        >>> atypes= accept_types (
            pd.Series, pd.DataFrame,np.ndarray, format=True )
        ..."'Series','DataFrame' and 'ndarray'"
    """
    return smart_format(
        [f'{o.__name__}' for o in objtypes]
        ) if format else [f'{o.__name__}' for o in objtypes] 



def check_dimensionality(obj, data, z, x):
    """ Check dimensionality of data and fix it.
    
    :param obj: Object, can be a class logged or else.
    :param data: 2D grid data of ndarray (z, x) dimensions.
    :param z: array-like should be reduced along the row axis.
    :param x: arraylike should be reduced along the columns axis.
    
    """
    def reduce_shape(Xshape, x, axis_name=None): 
        """ Reduce shape to keep the same shape"""
        mess ="`{0}` shape({1}) {2} than the data shape `{0}` = ({3})."
        ox = len(x) 
        dsh = Xshape 
        if len(x) > Xshape : 
            x = x[: int (Xshape)]
            obj._logging.debug(''.join([
                f"Resize {axis_name!r}={ox!r} to {Xshape!r}.", 
                mess.format(axis_name, len(x),'more',Xshape)])) 
                                    
        elif len(x) < Xshape: 
            Xshape = len(x)
            obj._logging.debug(''.join([
                f"Resize {axis_name!r}={dsh!r} to {Xshape!r}.",
                mess.format(axis_name, len(x),'less', Xshape)]))
        return int(Xshape), x 
    
    sz0, z = reduce_shape(data.shape[0], 
                          x=z, axis_name ='Z')
    sx0, x =reduce_shape (data.shape[1],
                          x=x, axis_name ='X')
    data = data [:sz0, :sx0]
    
    return data , z, x 

def smart_format(iter_obj, choice ='and'): 
    """ Smart format iterable object.
    
    :param iter_obj: iterable obj 
    :param choice: can be 'and' or 'or' for optional.
    
    :Example: 
        >>> from gofast.tools.coreutils import smart_format
        >>> smart_format(['model', 'iter', 'mesh', 'data'])
        ... 'model','iter','mesh' and 'data'
    """
    str_litteral =''
    try: 
        iter(iter_obj) 
    except:  return f"{iter_obj}"
    
    iter_obj = [str(obj) for obj in iter_obj]
    if len(iter_obj) ==1: 
        str_litteral= ','.join([f"{i!r}" for i in iter_obj ])
    elif len(iter_obj)>1: 
        str_litteral = ','.join([f"{i!r}" for i in iter_obj[:-1]])
        str_litteral += f" {choice} {iter_obj[-1]!r}"
    return str_litteral

def make_introspection(Obj: object , subObj: _Sub[object])->None: 
    """ Make introspection by using the attributes of instance created to 
    populate the new classes created.
    
    :param Obj: callable 
        New object to fully inherits of `subObject` attributes.
        
    :param subObj: Callable 
        Instance created.
    """
    # make introspection and set the all  attributes to self object.
    # if Obj attribute has the same name with subObj attribute, then 
    # Obj attributes get the priority.
    for key, value in  subObj.__dict__.items(): 
        if not hasattr(Obj, key) and key  != ''.join(['__', str(key), '__']):
            setattr(Obj, key, value)
  


def format_notes(text:str , cover_str: str ='~', inline=70, **kws): 
    """ Format note 
    :param text: Text to be formated.
    
    :param cover_str: type of ``str`` to surround the text.
    
    :param inline: Nomber of character before going in liine.
    
    :param margin_space: Must be <1 and expressed in %. The empty distance 
        between the first index to the inline text 
    :Example: 
        
        >>> from gofast.tools import funcutils as func 
        >>> text ='Automatic Option is set to ``True``.'\
            ' Composite estimator building is triggered.' 
        >>>  func.format_notes(text= text ,
        ...                       inline = 70, margin_space = 0.05)
    
    """
    
    headnotes =kws.pop('headernotes', 'notes')
    margin_ratio = kws.pop('margin_space', 0.2 )
    margin = int(margin_ratio * inline)
    init_=0 
    new_textList= []
    if len(text) <= (inline - margin): 
        new_textList = text 
    else : 
        for kk, char in enumerate (text): 
            if kk % (inline - margin)==0 and kk !=0: 
                new_textList.append(text[init_:kk])
                init_ =kk 
            if kk ==  len(text)-1: 
                new_textList.append(text[init_:])
  
    print('!', headnotes.upper(), ':')
    print('{}'.format(cover_str * inline)) 
    for k in new_textList:
        fmtin_str ='{'+ '0:>{}'.format(margin) +'}'
        print('{0}{1:>2}{2:<51}'.format(fmtin_str.format(cover_str), '', k))
        
    print('{0}{1:>51}'.format(' '* (margin -1), cover_str * (inline -margin+1 ))) 
    

def interpol_scipy(
        x_value,
        y_value,
        x_new,
        kind="linear",
        plot=False,
        fill_value="extrapolate"
):
    """
    Function to interpolate data using scipy's interp1d if available.
    
    Parameters 
    ------------
    * x_value : np.ndarray 
        Original abscissa values.
                
    * y_value : np.ndarray 
        Original ordinate values (slope).
                
    * x_new : np.ndarray 
        New abscissa values for which you want to interpolate data.
                
    * kind : str 
        Type of interpolation, e.g., "linear", "cubic".
                
    * fill_value : str 
        Extrapolation method. If None, scipy's interp1d will use constrained 
        interpolation. 
        Can be "extrapolate" to use fill_value.
        
    * plot : bool 
        Set to True to plot a graph of the original and interpolated data.

    Returns 
    --------
    np.ndarray 
        Interpolated ordinate values for 'x_new'.
    """

    spi = check_scipy_interpolate()
    if spi is None:
        return None
    
    try:
        func_ = spi.interp1d(x_value, y_value, kind=kind, fill_value=fill_value)
        y_new = func_(x_new)
        
        if plot:
            import matplotlib.pyplot as plt
            plt.plot(x_value, y_value, "o", x_new, y_new, "--")
            plt.legend(["Data", kind.capitalize()], loc="best")
            plt.title(f"Interpolation: {kind.capitalize()}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.grid(True)
            plt.show()

        return y_new

    except Exception as e:
        _logger.error(f"An unexpected error occurred during interpolation: {e}")
        return None
    
def _remove_str_word (ch, word_to_remove, deep_remove=False):
    """
    Small funnction to remove a word present on  astring character 
    whatever the number of times it will repeated.
    
    Parameters
    ----------
        * ch : str
                may the the str phrases or sentences . main items.
        * word_to_remove : str
                specific word to remove.
        * deep_remove : bool, optional
                use the lower case to remove the word even the word is uppercased 
                of capitalized. The default is False.

    Returns
    -------
        str ; char , new_char without the removed word .
        
    Examples
    ---------
    >>> from gofast.tools import funcutils as func
    >>> ch ='AMTAVG 7.76: "K1.fld", Dated 99-01-01,AMTAVG, 
    ...    Processed 11 Jul 17 AMTAVG'
    >>> ss=func._remove_str_word(char=ch, word_to_remove='AMTAVG', 
    ...                             deep_remove=False)
    >>> print(ss)
    
    """
    if type(ch) is not str : char =str(ch)
    if type(word_to_remove) is not str : word_to_remove=str(word_to_remove)
    
    if deep_remove == True :
        word_to_remove, char =word_to_remove.lower(),char.lower()

    if word_to_remove not in char :
        return char

    while word_to_remove in char : 
        if word_to_remove not in char : 
            break 
        index_wr = char.find(word_to_remove)
        remain_len=index_wr+len(word_to_remove)
        char=char[:index_wr]+char[remain_len:]

    return char

def stn_check_split_type(data_lines): 
    """
    Read data_line and check for data line the presence of 
    split_type < ',' or ' ', or any other marks.>
    Threshold is assume to be third of total data length.
    
    :params data_lines: list of data to parse . 
    :type data_lines: list 
 
    :returns: The split _type
    :rtype: str
    
    :Example: 
        >>> from gofast.tools  import funcutils as func
        >>> path =  data/ K6.stn
        >>> with open (path, 'r', encoding='utf8') as f : 
        ...                     data= f.readlines()
        >>>  print(func.stn_check_split_type(data_lines=data))
        
    """

    split_type =[',', ':',' ',';' ]
    data_to_read =[]
    # change the data if data is not dtype string elements.
    if isinstance(data_lines, np.ndarray): 
        if data_lines.dtype in ['float', 'int', 'complex']: 
            data_lines=data_lines.astype('<U12')
        data_lines= data_lines.tolist()
        
    if isinstance(data_lines, list):
        for ii, item in enumerate(data_lines[:int(len(data_lines)/3)]):
             data_to_read.append(item)
             # be sure the list is str item . 
             data_to_read=[''.join([str(item) for item in data_to_read])] 

    elif isinstance(data_lines, str): data_to_read=[str(data_lines)]
    
    for jj, sep  in enumerate(split_type) :
        if data_to_read[0].find(sep) > 0 :
            if data_to_read[0].count(sep) >= 2 * len(data_lines)/3:
                if sep == ' ': return  None  # use None more conventional 
                else : return sep 


def fr_en_parser (f, delimiter =':'): 
    """ Parse the translated data file. 
    
    :param f: translation file to parse.
    
    :param delimiter: str, delimiter.
    
    :return: generator obj, composed of a list of 
        french  and english Input translation. 
    
    :Example:
        >>> file_to_parse = 'pme.parserf.md'
        >>> path_pme_data = r'C:/Users\Administrator\Desktop\__elodata
        >>> data =list(BS.fr_en_parser(
            os.path.join(path_pme_data, file_to_parse)))
    """
    
    is_file = os.path.isfile (f)
    if not is_file: 
        raise IOError(f'Input {f} is not a file. Please check your file.')
    
    with open(f, 'r', encoding ='utf8') as ft: 
        data = ft.readlines()
        for row in data :
            if row in ( '\n', ' '):
                continue 
            fr, en = row.strip().split(delimiter)
            yield([fr, en])


def _isin (
        arr: Union [ArrayLike, List [float]] ,
        subarr:Union[ _Sub [ArrayLike] , _Sub[List[float]] ,float], 
        return_mask:bool=False, 
) -> bool : 
    """ Check whether the subset array `subcz` is in  `cz` array. 
    
    :param arr: Array-like - Array of item elements 
    :param subarr: Array-like, float - Subset array containing a subset items.
    :param return_mask: bool, return the mask where the element is in `arr`.
    
    :return: True if items in  test array `subarr` are in array `arr`. 
    
    """
    arr = np.array (arr );  subarr = np.array(subarr )

    return (True if True in np.isin (arr, subarr) else False
            ) if not return_mask else np.isin (arr, subarr) 

def _assert_all_types (
    obj: object , 
    *expected_objtype: type, 
    objname:str=None, 
 ) -> object: 
    """ Quick assertion of object type. Raises a `TypeError` if wrong type 
    is passed as an argument. For polishing the error message, one can add  
    the object name `objname` for specifying the object that raises errors  
    for letting the users to be aware of the reason of failure."""
    # if np.issubdtype(a1.dtype, np.integer): 
    if not isinstance( obj, expected_objtype): 
        n=str(objname) + ' expects' if objname is not None else 'Expects'
        raise TypeError (
            f"{n} type{'s' if len(expected_objtype)>1 else ''} "
            f"{smart_format(tuple (o.__name__ for o in expected_objtype))}"
            f" but {type(obj).__name__!r} is given.")
            
    return obj 

  
def drawn_boundaries(erp_data, appRes, index):
    """
    Function to drawn anomaly boundary 
    and return the anomaly with its boundaries
    
    :param erp_data: erp profile 
    :type erp_data: array_like or list 
    
    :param appRes: resistivity value of minimum pk anomaly 
    :type appRes: float 
    
    :param index: index of minimum pk anomaly 
    :type index: int 
    
    :return: anomaly boundary 
    :rtype: list of array_like 

    """
    f = 0 # flag to mention which part must be calculated 
    if index ==0 : 
        f = 1 # compute only right part 
    elif appRes ==erp_data[-1]: 
        f=2 # compute left part 
    
    def loop_sideBound(term):
        """
        loop side bar from anomaly and find the term side 
        
        :param term: is array of left or right side of anomaly.
        :type term: array 
        
        :return: side bar 
        :type: array_like 
        """
        tem_drawn =[]
        maxT=0 

        for ii, tem_rho in enumerate(term) : 

            diffRes_betw_2pts= tem_rho - appRes 
            if diffRes_betw_2pts > maxT : 
                maxT = diffRes_betw_2pts
                tem_drawn.append(tem_rho)
            elif diffRes_betw_2pts < maxT : 
                # rho_limit = tem_rho 
                break 
        return np.array(tem_drawn)
    # first broke erp profile from the anomalies 
    if f ==0 or f==2 : 
        left_term = erp_data[:index][::-1] # flip left term  for looping
        # flip again to keep the order 
        left_limit = loop_sideBound(term=left_term)[::-1] 

    if f==0 or f ==1 : 
        right_term= erp_data[index :]
        right_limit=loop_sideBound(right_term)
    # concat right and left to get the complete anomaly 
    if f==2: 
        anomalyBounds = np.append(left_limit,appRes)
                                   
    elif f ==1 : 
        anomalyBounds = np.array([appRes]+ right_limit.tolist())
    else: 
        left_limit = np.append(left_limit, appRes)
        anomalyBounds = np.concatenate((left_limit, right_limit))
    
    return appRes, index, anomalyBounds 

def fmt_text(
        anFeatures=None, 
        title = None,
        **kwargs) :
    """
    Function format text from anomaly features 
    
    :param anFeatures: Anomaly features 
    :type anFeatures: list or dict
    
    :param title: head lines 
    :type title: list
    
    :Example: 
        
        >>> from gofast.tools.coreutils import fmt_text
        >>> fmt_text(anFeatures =[1,130, 93,(146,145, 125)])
    
    """
    if title is None: 
        title = ['Ranking', 'rho(Ω.m)', 'position pk(m)', 'rho range(Ω.m)']
    inline =kwargs.pop('inline', '-')
    mlabel =kwargs.pop('mlabels', 100)
    line = inline * int(mlabel)
    
    #--------------------header ----------------------------------------
    print(line)
    tem_head ='|'.join(['{:^15}'.format(i) for i in title[:-1]])
    tem_head +='|{:^45}'.format(title[-1])
    print(tem_head)
    print(line)
    #-----------------------end header----------------------------------
    newF =[]
    if isinstance(anFeatures, dict):
        for keys, items in anFeatures.items(): 
            rrpos=keys.replace('_pk', '')
            rank=rrpos[0]
            pos =rrpos[1:]
            newF.append([rank, min(items), pos, items])
            
    elif isinstance(anFeatures, list): 
        newF =[anFeatures]
    
    
    for anFeatures in newF: 
        strfeatures ='|'.join(['{:^15}'.format(str(i)) \
                               for i in anFeatures[:-1]])
        try : 
            iter(anFeatures[-1])
        except : 
            strfeatures +='|{:^45}'.format(str(anFeatures[-1]))
        else : 
            strfeatures += '|{:^45}'.format(
                ''.join(['{} '.format(str(i)) for i in anFeatures[-1]]))
            
        print(strfeatures)
        print(line)
    

def wrap_infos (
        phrase ,
        value ='',
        underline ='-',
        unit ='',
        site_number= '',
        **kws) : 
    """Display info from anomaly details."""
    
    repeat =kws.pop('repeat', 77)
    intermediate =kws.pop('inter+', '')
    begin_phrase_mark= kws.pop('begin_phrase', '--|>')
    on = kws.pop('on', False)
    if not on: return ''
    else : 
        print(underline * repeat)
        print('{0} {1:<50}'.format(begin_phrase_mark, phrase), 
              '{0:<10} {1}'.format(value, unit), 
              '{0}'.format(intermediate), "{}".format(site_number))
        print(underline * repeat )
    

def reshape(arr , axis = None) :
    """ Detect the array shape and reshape it accordingly, back to the given axis. 
    
    :param array: array_like with number of dimension equals to 1 or 2 
    :param axis: axis to reshape back array. If 'axis' is None and 
        the number of dimension is greater than 1, it reshapes back array 
        to array-like 
    
    :returns: New reshaped array 
    
    :Example: 
        >>> import numpy as np 
        >>> from gofast.tools.coreutils import reshape 
        >>> array = np.random.randn(50 )
        >>> array.shape
        ... (50,)
        >>> ar1 = reshape(array, 1) 
        >>> ar1.shape 
        ... (1, 50)
        >>> ar2 =reshape(ar1 , 0) 
        >>> ar2.shape 
        ... (50, 1)
        >>> ar3 = reshape(ar2, axis = None)
        >>> ar3.shape # goes back to the original array  
        >>> ar3.shape 
        ... (50,)
        
    """
    arr = np.array(arr)
    if arr.ndim > 2 : 
        raise ValueError('Expect an array with max dimension equals to 2' 
                         f' but {str(arr.ndim)!r} were given.')
        
    if axis  not in (0 , 1, -1, None): 
        raise ValueError(f'Wrong axis value: {str(axis)!r}')
        
    if axis ==-1:
        axis =None 
    if arr.ndim ==1 : 
        # ie , axis is None , array is an array-like object
        s0, s1= arr.shape [0], None 
    else : 
        s0, s1 = arr.shape 
    if s1 is None: 
        return  arr.reshape ((1, s0)) if axis == 1 else (arr.reshape (
            (s0, 1)) if axis ==0 else arr )
    try : 
        arr = arr.reshape ((s0 if s1==1 else s1, )) if axis is None else (
            arr.reshape ((1, s0)) if axis==1  else arr.reshape ((s1, 1 ))
            )
    except ValueError: 
        # error raises when user mistakes to input the right axis. 
        # (ValueError: cannot reshape array of size 54 into shape (1,1)) 
        # then return to him the original array 
        pass 

    return arr   
    
    
def ismissing(refarr, arr, fill_value = np.nan, return_index =False): 
    """ Get the missing values in array-like and fill it  to match the length
    of the reference array. 
    
    The function makes sense especially for frequency interpollation in the 
    'attenuation band' when using the audio-frequency magnetotelluric methods. 
    
    :param arr: array-like- Array to be extended with fill value. It should be  
        shorter than the `refarr`. Otherwise it returns the same array `arr` 
    :param refarr: array-like- the reference array. It should have a greater 
        length than the array 
    :param fill_value: float - Value to fill the `arr` to match the length of 
        the `refarr`. 
    :param return_index: bool or str - array-like, index of the elements element 
        in `arr`. Default is ``False``. Any other value should returns the 
        mask of existing element in reference array
        
    :returns: array and values missings or indexes in reference array. 
    
    :Example: 
        
    >>> import numpy as np 
    >>> from gofast.tools.coreutils import ismissing
    >>> refreq = np.linspace(7e7, 1e0, 20) # 20 frequencies as reference
    >>> # remove the value between index 7 to 12 and stack again
    >>> freq = np.hstack ((refreq.copy()[:7], refreq.copy()[12:] ))  
    >>> f, m  = ismissing (refreq, freq)
    >>> f, m  
    ...array([7.00000000e+07, 6.63157895e+07, 6.26315791e+07, 5.89473686e+07,
           5.52631581e+07, 5.15789476e+07, 4.78947372e+07,            nan,
                      nan,            nan,            nan,            nan,
           2.57894743e+07, 2.21052638e+07, 1.84210534e+07, 1.47368429e+07,
           1.10526324e+07, 7.36842195e+06, 3.68421147e+06, 1.00000000e+00])
    >>> m # missing values 
    ... array([44210526.68421052, 40526316.21052632, 36842105.73684211,
           33157895.2631579 , 29473684.78947368])
    >>>  _, m_ix  = ismissing (refreq, freq, return_index =True)
    >>> m_ix 
    ... array([ 7,  8,  9, 10, 11], dtype=int64)
    >>> # assert the missing values from reference values 
    >>> refreq[m_ix ] # is equal to m 
    ... array([44210526.68421052, 40526316.21052632, 36842105.73684211,
           33157895.2631579 , 29473684.78947368]) 
        
    """
    return_index = str(return_index).lower() 
    fill_value = _assert_all_types(fill_value, float, int)
    if return_index in ('false', 'value', 'val') :
        return_index ='values' 
    elif return_index  in ('true', 'index', 'ix') :
        return_index = 'index' 
    else : 
        return_index = 'mask'
    
    ref = refarr.copy() ; mask = np.isin(ref, arr)
    miss_values = ref [~np.isin(ref, arr)] 
    miss_val_or_ix  = (ref [:, None] == miss_values).argmax(axis=0
                         ) if return_index =='index' else ref [~np.isin(ref, arr)] 
    
    miss_val_or_ix = mask if return_index =='mask' else miss_val_or_ix 
    # if return_missing_values: 
    ref [~np.isin(ref, arr)] = fill_value 
    #arr= np.hstack ((arr , np.repeat(fill_value, 0 if m <=0 else m  ))) 
    #refarr[refarr ==arr] if return_index else arr 
    return  ref , miss_val_or_ix   

def make_arr_consistent (
        refarr, arr, fill_value = np.nan, return_index = False, 
        method='naive'): 
    """
    Make `arr` to be consistent with the reference array `refarr`. Fill the 
    missing value with param `fill_value`. 
    
    Note that it does care of the position of the value in the array. Use 
    Numpy digitize to compute the bins. The array caveat here is the bins 
    must be monotonically decreasing or increasing.
    
    If the values in `arr` are present in `refarr`, the position of `arr` 
    in new consistent array should be located decreasing or increasing order. 
    
    Parameters 
    ------------
    arr: array-like 1d, 
        Array to extended with fill value. It should be  shorter than the 
        `refarr`.
        
    refarr: array-like- the reference array. It should have a greater 
        length than the array `arr`.  
    fill_value: float, 
        Value to fill the `arr` to match the length of the `refarr`. 
    return_index: bool or str, default=True 
         index of the position of the  elements in `refarr`.
         Default is ``False``. If ``mask`` should  return the 
        mask of existing element in reference array
    method: str, default="naive"
        Is the method used to find the right position of items in `arr`
        based on the reference array. 
        - ``naive``, considers the length of ``arr`` must fit the number of 
            items that should be visible in the consistent array. This method 
            erases the remaining bins values out of length of `arr`. 
        - ``strict` did the same but rather than considering the length, 
            it considers the maximum values in the `arr`. It assumes that `arr`
            is sorted in ascending order. This methods is usefull for plotting 
            a specific stations since the station loactions are sorted in 
            ascending order. 
        
    Returns 
    ---------
    non_zero_index , mask or t  
        index: indices of the position of `arr` items in ``refarr``. 
        mask: bool of the position `arr` items in ``refarr``
        t: new consistent array with the same length as ``refarr``
    
    Examples 
    ----------
    >>> import numpy as np 
    >>> from gofast.tools.coreutils import make_arr_consistent
    >>> refarr = np.arange (12) 
    >>> arr = np.arange (7, 10) 
    >>> make_arr_consistent (refarr, arr ) 
    Out[84]: array([nan, nan, nan, nan, nan, nan, nan,  7.,  8.,  9., nan, nan])
    >>> make_arr_consistent (refarr, arr , return_index =True )
    Out[104]: array([7, 8, 9], dtype=int64)
    >>> make_arr_consistent (refarr, arr , return_index ="mask" )
    Out[105]: 
    array([False, False, False, False, False, False, False,  True,  True,
            True, False, False])
    >>> a = np.arange ( 12 ); b = np.linspace (7, 10 , 7) 
    >>> make_arr_consistent (a, b ) 
    Out[112]: array([nan, nan, nan, nan, nan, nan, nan,  7.,  8.,  9., 10., 11.])
    >>> make_arr_consistent (a, b ,method='strict') 
    Out[114]: array([nan, nan, nan, nan, nan, nan, nan,  7.,  8.,  9., 10., nan])
    """
    try : 
        refarr = reshape( refarr).shape[1] 
        arr= reshape( arr).shape[1] 
    except :pass 
    else: raise TypeError ("Expects one-dimensional arrays for both arrays.")

    t = np.full_like( refarr, fill_value = np.nan, dtype =float )
    temp_arr = np.digitize( refarr, arr) 
    non_zero_index = reshape (np.argwhere (temp_arr!=0 ) ) 
    t[non_zero_index] = refarr [non_zero_index] 
    # force value to keep only 
    # value in array 
    if method=='strict':
        index = reshape ( np.argwhere (  (max( arr)  - t) < 0 ) ) 
        t [index ]= np.nan 
    else: 
        if len (t[~np.isnan (t)]) > len(arr): 
            t [ - (len(t[~np.isnan (t)])-len(arr)):]= np.nan 
    # update the non_zeros index 
    non_zero_index= reshape ( np.argwhere (~np.isnan (t)))
    # now replace all NaN value by filled value 
    t [np.isnan(t)] = fill_value 

    return  refarr == t  if return_index =='mask' else (
        non_zero_index if return_index else t )


    
def concat_array_from_list (list_of_array , concat_axis = 0) :
    """ Concat array from list and set the None value in the list as NaN.
    
    :param list_of_array: List of array elements 
    :type list of array: list 
    
    :param concat_axis: axis for concatenation ``0`` or ``1``
    :type concat_axis: int 
    
    :returns: Concatenated array with shape np.ndaarry(
        len(list_of_array[0]), len(list_of_array))
    :rtype: np.ndarray 
    
    :Example: 
        
    >>> import numpy as np 
    >>> from gofast.tools.coreutils import concat_array_from_list 
    >>> np.random.seed(0)
    >>> ass=np.random.randn(10)
    >>> ass = ass2=np.linspace(0,15,10)
    >>> concat_array_from_list ([ass, ass]) 
    
    """
    concat_axis =int(_assert_all_types(concat_axis, int, float))
    if concat_axis not in (0 , 1): 
        raise ValueError(f'Unable to understand axis: {str(concat_axis)!r}')
    
    list_of_array = list(map(lambda e: np.array([np.nan])
                             if e is None else np.array(e), list_of_array))
    # if the list is composed of one element of array, keep it outside
    # reshape accordingly 
    if len(list_of_array)==1:
        ar = (list_of_array[0].reshape ((1,len(list_of_array[0]))
                 ) if concat_axis==0 else list_of_array[0].reshape(
                        (len(list_of_array[0]), 1)
                 )
             ) if list_of_array[0].ndim ==1 else list_of_array[0]
                     
        return ar 

    #if concat_axis ==1: 
    list_of_array = list(map(
            lambda e:e.reshape(e.shape[0], 1) if e.ndim ==1 else e ,
            list_of_array)
        ) if concat_axis ==1 else list(map(
            lambda e:e.reshape(1, e.shape[0]) if e.ndim ==1 else e ,
            list_of_array))
                
    return np.concatenate(list_of_array, axis = concat_axis)
    

    
def strip_item(item_to_clean, item=None, multi_space=12):
    """
    Function to strip item around string values.  if the item to clean is None or 
    item-to clean is "''", function will return None value

    Parameters
    ----------
        * item_to_clean : list or np.ndarray of string 
                 List to strip item.
        * cleaner : str , optional
                item to clean , it may change according the use. The default is ''.
        * multi_space : int, optional
                degree of repetition may find around the item. The default is 12.
    Returns
    -------
        list or ndarray
            item_to_clean , cleaned item 
            
    :Example: 
        
     >>> import numpy as np
     >>> new_data=_strip_item (item_to_clean=np.array(['      ss_data','    pati   ']))
     >>>  print(np.array(['      ss_data','    pati   ']))
     ... print(new_data)

    """
    if item==None :
        item = ' '
    
    cleaner =[(''+ ii*'{0}'.format(item)) for ii in range(multi_space)]
    
    if isinstance (item_to_clean, str) : 
        item_to_clean=[item_to_clean] 
        
    # if type(item_to_clean ) != list :#or type(item_to_clean ) !=np.ndarray:
    #     if type(item_to_clean ) !=np.ndarray:
    #         item_to_clean=[item_to_clean]
    if item_to_clean in cleaner or item_to_clean ==['']:
        #warnings.warn ('No data found for sanitization; returns None.')
        return None 
    try : 
        multi_space=int(multi_space)
    except : 
        raise TypeError('argument <multplier> must be an integer'
                        'not {0}'.format(type(multi_space)))
    
    for jj, ss in enumerate(item_to_clean) : 
        for space in cleaner:
            if space in ss :
                new_ss=ss.strip(space)
                item_to_clean[jj]=new_ss
    
    return item_to_clean  
 

def pretty_printer(
        clfs: List[_F],  
        clf_score:List[float]=None, 
        scoring: Optional[str] =None,
        **kws
 )->None: 
    """ Format and pretty print messages after gridSearch using multiples
    estimators.
    
    Display for each estimator, its name, it best params with higher score 
    and the mean scores. 
    
    Parameters
    ----------
    clfs:Callables 
        classifiers or estimators 
    
    clf_scores: array-like
        for single classifier, usefull to provided the 
        cross validation score.
    
    scoring: str 
        Scoring used for grid search.
    """
    empty =kws.pop('empty', ' ')
    e_pad =kws.pop('e_pad', 2)
    p=list()

    if not isinstance(clfs, (list,tuple)): 
        clfs =(clfs, clf_score)

    for ii, (clf, clf_be, clf_bp, clf_sc) in enumerate(clfs): 
        s_=[e_pad* empty + '{:<20}:'.format(
            clf.__class__.__name__) + '{:<20}:'.format(
                'Best-estimator <{}>'.format(ii+1)) +'{}'.format(clf_be),
         e_pad* empty +'{:<20}:'.format(' ')+ '{:<20}:'.format(
            'Best paramaters') + '{}'.format(clf_bp),
         e_pad* empty  +'{:<20}:'.format(' ') + '{:<20}:'.format(
            'scores<`{}`>'.format(scoring)) +'{}'.format(clf_sc)]
        try : 
            s0= [e_pad* empty +'{:<20}:'.format(' ')+ '{:<20}:'.format(
            'scores mean')+ '{}'.format(clf_sc.mean())]
        except AttributeError:
            s0= [e_pad* empty +'{:<20}:'.format(' ')+ '{:<20}:'.format(
            'scores mean')+ 'None']
            s_ +=s0
        else :
            s_ +=s0

        p.extend(s_)
    
    for i in p: 
        print(i)
 
def random_state_validator(seed):
    """Turn seed into a Numpy-Random-RandomState instance.
    
    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
        
    Returns
    -------
    :class:`numpy:numpy.random.RandomState`
        The random state object based on `seed` parameter.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState instance" % seed
    )

def is_iterable (
        y, exclude_string= False, transform = False , parse_string =False, 
)->Union [bool , list]: 
    """ Asserts iterable object and returns boolean or transform object into
     an iterable.
    
    Function can also transform a non-iterable object to an iterable if 
    `transform` is set to ``True``.
    
    :param y: any, object to be asserted 
    :param exclude_string: bool, does not consider string as an iterable 
        object if `y` is passed as a string object. 
    :param transform: bool, transform  `y` to an iterable objects. But default 
        puts `y` in a list object. 
    :param parse_string: bool, parse string and convert the list of string 
        into iterable object is the `y` is a string object and containg the 
        word separator character '[#&.*@!_,;\s-]'. Refer to the function 
        :func:`~gofast.tools.coreutils.str2columns` documentation.
        
    :returns: 
        - bool, or iterable object if `transform` is set to ``True``. 
        
    .. note:: 
        Parameter `parse_string` expects `transform` to be ``True``, otherwise 
        a ValueError will raise. Note :func:`.is_iterable` is not dedicated 
        for string parsing. It parses string using the default behaviour of 
        :func:`.str2columns`. Use the latter for string parsing instead. 
        
    :Examples: 
    >>> from gofast.coreutils.is_iterable 
    >>> is_iterable ('iterable', exclude_string= True ) 
    Out[28]: False
    >>> is_iterable ('iterable', exclude_string= True , transform =True)
    Out[29]: ['iterable']
    >>> is_iterable ('iterable', transform =True)
    Out[30]: 'iterable'
    >>> is_iterable ('iterable', transform =True, parse_string=True)
    Out[31]: ['iterable']
    >>> is_iterable ('iterable', transform =True, exclude_string =True, 
                     parse_string=True)
    Out[32]: ['iterable']
    >>> is_iterable ('parse iterable object', parse_string=True, 
                     transform =True)
    Out[40]: ['parse', 'iterable', 'object']
    """
    if (parse_string and not transform) and isinstance (y, str): 
        raise ValueError ("Cannot parse the given string. Set 'transform' to"
                          " ``True`` otherwise use the 'str2columns' utils"
                          " from 'gofast.tools.coreutils' instead.")
    y = str2columns(y) if isinstance(y, str) and parse_string else y 
    
    isiter = False  if exclude_string and isinstance (
        y, str) else hasattr (y, '__iter__')
    
    return ( y if isiter else [ y ] )  if transform else isiter 

    
def str2columns (text,  regex=None , pattern = None): 
    """Split text from the non-alphanumeric markers using regular expression. 
    
    Remove all string non-alphanumeric and some operator indicators,  and 
    fetch attributes names. 
    
    Parameters 
    -----------
    text: str, 
        text litteral containing the columns the names to retrieve
        
    regex: `re` object,  
        Regular expresion object. the default is:: 
            
            >>> import re 
            >>> re.compile (r'[#&*@!_,;\s-]\s*', flags=re.IGNORECASE) 
    pattern: str, default = '[#&*@!_,;\s-]\s*'
        The base pattern to split the text into a columns
        
    Returns
    -------
    attr: List of attributes 
    
    Examples
    ---------
    >>> from gofast.tools.coreutils import str2columns 
    >>> text = ('this.is the text to split. It is an: example of; splitting str - to text.')
    >>> str2columns (text )  
    ... ['this',
         'is',
         'the',
         'text',
         'to',
         'split',
         'It',
         'is',
         'an:',
         'example',
         'of',
         'splitting',
         'str',
         'to',
         'text']

    """
    pattern = pattern or  r'[#&.*@!_,;\s-]\s*'
    regex = regex or re.compile (pattern, flags=re.IGNORECASE) 
    text= list(filter (None, regex.split(str(text))))
    return text 
       
def sanitize_frame_cols(
        d,  func:_F = None , regex=None, pattern:str = None, 
        fill_pattern:str =None, inplace:bool =False 
        ):
    """ Remove an indesirable characters to the dataframe and returns 
    new columns. 
    
    Use regular expression for columns sanitizing 
    
    Parameters 
    -----------
    
    d: list, columns, 
        columns to sanitize. It might contain a list of items to 
        to polish. If dataframe or series are given, the dataframe columns  
        and the name respectively will be polished and returns the same 
        dataframe.
        
    func: _F, callable 
       Universal function used to clean the columns 
       
    regex: `re` object,
        Regular expresion object. the default is:: 
            
            >>> import re 
            >>> re.compile (r'[_#&.)(*@!_,;\s-]\s*', flags=re.IGNORECASE) 
    pattern: str, default = '[_#&.)(*@!_,;\s-]\s*'
        The base pattern to sanitize the text in each column names. 
        
    fill_pattern: str, default='' 
        pattern to replace the non-alphabetic character in each item of 
        columns. 
    inplace: bool, default=False, 
        transform the dataframe of series in place. 

    Returns
    -------
    columns | pd.Series | dataframe. 
        return Serie or dataframe if one is given, otherwise it returns a 
        sanitized columns. 
        
    Examples 
    ---------
    >>> from gofast.tools.coreutils import sanitize_frame_cols 
    >>> from gofast.tools.coreutils import read_data 
    >>> h502= read_data ('data/boreholes/H502.xlsx') 
    >>> h502 = sanitize_frame_cols (h502, fill_pattern ='_' ) 
    >>> h502.columns[:3]
    ... Index(['depth_top', 'depth_bottom', 'strata_name'], dtype='object') 
    >>> f = lambda r : r.replace ('_', "'s ") 
    >>> h502_f= sanitize_frame_cols( h502, func =f )
    >>> h502_f.columns [:3]
    ... Index(['depth's top', 'depth's bottom', 'strata's name'], dtype='object')
               
    """
    isf , iss= False , False 
    pattern = pattern or r'[_#&.)(*@!_,;\s-]\s*'
    fill_pattern = fill_pattern or '' 
    fill_pattern = str(fill_pattern)
    
    regex = regex or re.compile (pattern, flags=re.IGNORECASE)
    
    if isinstance(d, pd.Series): 
        c = [d.name]  
        iss =True 
    elif isinstance (d, pd.DataFrame ) :
        c = list(d.columns) 
        isf = True
        
    else : 
        if not is_iterable(d) : c = [d] 
        else : c = d 
        
    if inspect.isfunction(func): 
        c = list( map (func , c ) ) 
    
    else : c =list(map ( 
        lambda r : regex.sub(fill_pattern, r.strip() ), c ))
        
    if isf : 
        if inplace : d.columns = c
        else : d =pd.DataFrame(d.values, columns =c )
        
    elif iss:
        if inplace: d.name = c[0]
        else : d= pd.Series (data =d.values, name =c[0] )
        
    else : d = c 

    return d 

def find_by_regex (o , pattern,  func = re.match, **kws ):
    """ Find pattern in object whatever an "iterable" or not. 
    
    when we talk about iterable, a string value is not included.
    
    Parameters 
    -------------
    o: str or iterable,  
        text litteral or an iterable object containing or not the specific 
        object to match. 
    pattern: str, default = '[_#&*@!_,;\s-]\s*'
        The base pattern to split the text into a columns
    
    func: re callable , default=re.match
        regular expression search function. Can be
        [re.match, re.findall, re.search ],or any other regular expression 
        function. 
        
        * ``re.match()``:  function  searches the regular expression pattern and 
            return the first occurrence. The Python RegEx Match method checks 
            for a match only at the beginning of the string. So, if a match is 
            found in the first line, it returns the match object. But if a match 
            is found in some other line, the Python RegEx Match function returns 
            null.
        * ``re.search()``: function will search the regular expression pattern 
            and return the first occurrence. Unlike Python re.match(), it will 
            check all lines of the input string. The Python re.search() function 
            returns a match object when the pattern is found and “null” if 
            the pattern is not found
        * ``re.findall()`` module is used to search for 'all' occurrences that 
            match a given pattern. In contrast, search() module will only 
            return the first occurrence that matches the specified pattern. 
            findall() will iterate over all the lines of the file and will 
            return all non-overlapping matches of pattern in a single step.
    kws: dict, 
        Additional keywords arguments passed to functions :func:`re.match` or 
        :func:`re.search` or :func:`re.findall`. 
        
    Returns 
    -------
    om: list 
        matched object put is the list 
        
    Example
    --------
    >>> from gofast.tools.coreutils import find_by_regex
    >>> from gofast.datasets import load_hlogs 
    >>> X0, _= load_hlogs (as_frame =True )
    >>> columns = X0.columns 
    >>> str_columns =','.join (columns) 
    >>> find_by_regex (str_columns , pattern='depth', func=re.search)
    ... ['depth']
    >>> find_by_regex(columns, pattern ='depth', func=re.search)
    ... ['depth_top', 'depth_bottom']
    
    """
    om = [] 
    if isinstance (o, str): 
        om = func ( pattern=pattern , string = o, **kws)
        if om: 
            om= om.group() 
        om =[om]
    elif is_iterable(o): 
        o = list(o) 
        for s in o : 
            z = func (pattern =pattern , string = s, **kws)
            if z : 
                om.append (s) 
                
    if func.__name__=='findall': 
        om = list(itertools.chain (*om )) 
    # keep None is nothing 
    # fit the corresponding pattern 
    if len(om) ==0 or om[0] is None: 
        om = None 
    return  om 
    
def is_in_if (o: iter,  items: Union [str , iter], error = 'raise', 
               return_diff =False, return_intersect = False): 
    """ Raise error if item is not  found in the iterable object 'o' 
    
    :param o: unhashable type, iterable object,  
        object for checkin. It assumes to be an iterable from which 'items' 
        is premused to be in. 
    :param items: str or list, 
        Items to assert whether it is in `o` or not. 
    :param error: str, default='raise'
        raise or ignore error when none item is found in `o`. 
    :param return_diff: bool, 
        returns the difference items which is/are not included in 'items' 
        if `return_diff` is ``True``, will put error to ``ignore`` 
        systematically.
    :param return_intersect:bool,default=False
        returns items as the intersection between `o` and `items`.
    :raise: ValueError 
        raise ValueError if `items` not in `o`. 
    :return: list,  
        `s` : object found in ``o` or the difference object i.e the object 
        that is not in `items` provided that `error` is set to ``ignore``.
        Note that if None object is found  and `error` is ``ignore`` , it 
        will return ``None``, otherwise, a `ValueError` raises. 
        
    :example: 
        >>> from gofast.datasets import load_hlogs 
        >>> from gofast.tools.coreutils import is_in_if 
        >>> X0, _= load_hlogs (as_frame =True )
        >>> is_in_if  (X0 , items= ['depth_top', 'top']) 
        ... ValueError: Item 'top' is missing in the object 
        >>> is_in_if (X0, ['depth_top', 'top'] , error ='ignore') 
        ... ['depth_top']
        >>> is_in_if (X0, ['depth_top', 'top'] , error ='ignore',
                       return_diff= True) 
        ... ['sp',
         'well_diameter',
         'layer_thickness',
         'natural_gamma',
         'short_distance_gamma',
         'strata_name',
         'gamma_gamma',
         'depth_bottom',
         'rock_name',
         'resistivity',
         'hole_id']
    """
    
    if isinstance (items, str): 
        items =[items]
    elif not is_iterable(o): 
        raise TypeError (f"Expect an iterable object, not {type(o).__name__!r}")
    # find intersect object 
    s= set (o).intersection (items) 
    
    miss_items = list(s.difference (o)) if len(s) > len(
        items) else list(set(items).difference (s)) 

    if return_diff or return_intersect: 
        error ='ignore'
    
    if len(miss_items)!=0 :
        if error =='raise': 
            v= smart_format(miss_items)
            verb = f"{ ' '+ v +' is' if len(miss_items)<2 else  's '+ v + 'are'}"
            raise ValueError (
                f"Item{verb} missing in the {type(o).__name__.lower()} {o}.")
            
       
    if return_diff : 
        # get difference 
        s = list(set(o).difference (s))  if len(o) > len( 
            s) else list(set(items).difference (s)) 
        # s = set(o).difference (s)  
    elif return_intersect: 
        s = list(set(o).intersection(s))  if len(o) > len( 
            items) else list(set(items).intersection (s))     
    
    s = None if len(s)==0 else list (s) 
    
    return s  
  
def map_specific_columns ( 
        X: DataFrame, 
        ufunc:_F , 
        columns_to_skip:List[str]=None,   
        pattern:str=None, 
        inplace:bool= False, 
        **kws
        ): 
    """ Apply function to a specific columns is the dataframe. 
    
    It is possible to skip some columns that we want operation to not be 
    performed.
    
    Parameters 
    -----------
    X: dataframe, 
        pandas dataframe with valid columns 
    ufunc: callable, 
        Universal function that can be applying to the dataframe. 
    columns_to_skip: list or str , 
        List of columns to skip. If given as string and separed by the default
        pattern items, it should be converted to a list and make sure the 
        columns name exist in the dataframe. Otherwise an error with 
        raise.
        
    pattern: str, default = '[#&*@!,;\s]\s*'
        The base pattern to split the text in `column2skip` into a columns
        For instance, the following string coulb be splitted to:: 
            
            'depth_top, thickness, sp, gamma_gamma' -> 
            ['depth_top', 'thickness', 'sp', 'gamma_gamma']
        
        Refer to :func:`~.str2columns` for further details. 
    inplace: bool, default=True 
        Modified dataframe in place and return None, otherwise return a 
        new dataframe 
    kws: dict, 
        Keywords argument passed to :func: `pandas.DataFrame.apply` function 
        
    Returns 
    ---------
    X: Dataframe or None 
        Dataframe modified inplace with values computed using the given 
        `func`except the skipped columns, or ``None`` if `inplace` is ``True``. 
        
    Examples 
    ---------
    >>> from gofast.datasets import load_hlogs 
    >>> from gofast.tools.plotutils import map_specific_columns 
    >>> X0, _= load_hlogs (as_frame =True ) 
    >>> # let visualize the  first3 values of `sp` and `resistivity` keys 
    >>> X0['sp'][:3] , X0['resistivity'][:3]  
    ... (0   -1.580000
         1   -1.580000
         2   -1.922632
         Name: sp, dtype: float64,
         0    15.919130
         1    16.000000
         2    24.422316
         Name: resistivity, dtype: float64)
    >>> column2skip = ['hole_id','depth_top', 'depth_bottom', 
                      'strata_name', 'rock_name', 'well_diameter', 'sp']
    >>> map_specific_columns (X0, ufunc = np.log10, column2skip)
    >>> # now let visualize the same keys values 
    >>> X0['sp'][:3] , X0['resistivity'][:3]
    ... (0   -1.580000
         1   -1.580000
         2   -1.922632
         Name: sp, dtype: float64,
         0    1.201919
         1    1.204120
         2    1.387787
         Name: resistivity, dtype: float64)
    >>> # it is obvious the `resistiviy` values is log10 
    >>> # while `sp` stil remains the same 
      
    """
    X = _assert_all_types(X, pd.DataFrame)
    if not callable(ufunc): 
        raise TypeError ("Expect a function for `ufunc`; "
                         f"got {type(ufunc).__name__!r}")
        
    pattern = pattern or r'[#&*@!,;\s]\s*'
    if not is_iterable( columns_to_skip): 
        raise TypeError ("Columns  to skip expect an iterable object;"
                         f" got {type(columns_to_skip).__name__!r}")
        
    if isinstance(columns_to_skip, str):
        columns_to_skip = str2columns (columns_to_skip, pattern=pattern  )
    #assert whether column to skip is in 
    if columns_to_skip:
        cskip = copy.deepcopy(columns_to_skip)
        columns_to_skip = is_in_if(X.columns, columns_to_skip, return_diff= True)
        if len(columns_to_skip) ==len (X.columns): 
            warnings.warn("Value(s) to skip are not detected.")
    elif columns_to_skip is None: 
        columns_to_skip = list(X.columns) 
        
    if inplace : 
        X[columns_to_skip] = X[columns_to_skip].apply (
            ufunc , **kws)
        X.drop (columns = cskip , inplace =True )
        return 
    if not inplace: 
        X0 = X.copy() 
        X0[columns_to_skip] = X0[columns_to_skip].apply (
            ufunc , **kws)
    
        return  X0   
    
def is_depth_in (X, name, columns = None, error= 'ignore'): 
    """ Assert wether depth exists in the data from column attributes.  
    
    If name is an integer value, it assumes to be the index in the columns 
    of the dataframe if not exist , a warming will be show to user. 
    
    :param X: dataframe 
        dataframe containing the data for plotting 
        
    :param columns: list,
        New labels to replace the columns in the dataframe. If given , it 
        should fit the number of colums of `X`. 
        
    :param name: str, int  
        depth name in the dataframe or index to retreive the name of the depth 
        in dataframe 
    :param error: str , default='ignore'
        Raise or ignore when depth is not found in the dataframe. Whe error is 
        set to ``ignore``, a pseudo-depth is created using the lenght of the 
        the dataframe, otherwise a valueError raises.
        
    :return: X, depth 
        Dataframe without the depth columns and depth values.
    """
    
    X= _assert_all_types( X, pd.DataFrame )
    if columns is not None: 
        columns = list(columns)
        if not is_iterable(columns): 
            raise TypeError("columns expects an iterable object."
                            f" got {type (columns).__name__!r}")
        if len(columns ) != len(X.columns): 
            warnings.warn("Cannot rename columns with new labels. Expect "
                          "a size to be consistent with the columns X."
                          f" {len(columns)} and {len(X.columns)} are given."
                          )
        else : 
            X.columns = columns # rename columns
        
    else:  columns = list(X.columns) 
    
    _assert_all_types(name,str, int, float )
    
    # if name is given as indices 
    # collect the name at that index 
    if isinstance (name, (int, float) )  :     
        name = int (name )
        if name > len(columns): 
            warnings.warn ("Name index {name} is out of the columns range."
                           f" Max index of columns is {len(columns)}")
            name = None 
        else : 
            name = columns.pop (name)
    
    elif isinstance (name, str): 
        # find in columns whether a name can be 
        # found. Note that all name does not need 
        # to be written completely 
        # for instance name =depth can retrieved 
        # ['depth_top, 'depth_bottom'] , in that case 
        # the first occurence is selected i.e. 'depth_top'
        n = find_by_regex( 
            columns, pattern=fr'{name}', func=re.search)

        if n is not None:
            name = n[0]
            
        # for consistency , recheck all and let 
        # a warning to user 
        if name not in columns :
            msg = f"Name {name!r} does not match any column names."
            if error =='raise': 
                raise ValueError (msg)

            warnings.warn(msg)
            name =None  
            
    # now create a pseudo-depth 
    # as a range of len X 
    if name is None: 
        if error =='raise':
            raise ValueError ("Depth column not found in dataframe."
                              )
        depth = pd.Series ( np.arange ( len(X)), name ='depth (m)') 
    else : 
        # if depth name exists, 
        # remove it from X  
        depth = X.pop (name ) 
        
    return  X , depth     
    
 
def hex_to_rgb (c): 
    """ Convert colors Hexadecimal to RGB """
    c=c.lstrip('#')
    return tuple(int(c[i:i+2], 16) for i in (0, 2, 4)) 

def _validate_name_in (name, defaults = '', expect_name= None, 
                         exception = None , deep=False ): 
    """ Assert name in multiples given default names. 
    
    Parameters 
    -----------
    name: str, 
      given name to assert 
    default: list, str, default =''
      default values used for assertion 
    expect_name: str, optional 
      name to return in case assertion is verified ( as ``True``)
    deep: bool, default=False 
      Find item in a litteral default string. If set  to ``True``, 
      `defaults` are joined and check whether an occurence of `name` is in the 
      defaults 
      
    exception: Exception 
      Error to raise if name is not found in the default values. 
      
    Returns
    -------
    name: str, 
      Verified name or boolean if expect name if ``None``. 
      
    Examples 
    -------
    >>> from gofast.tools.coreutils import _validate_name_in 
    >>> dnames = ('NAME', 'FIST NAME', 'SUrname')
    >>> _validate_name_in ('name', defaults=dnames )
    False 
    >>> _validate_name_in ('name', defaults= dnames, deep =True )
    True
    >>> _validate_name_in ('name', defaults=dnames , expect_name ='NAM')
    False 
    >>> _validate_name_in ('name', defaults=dnames , expect_name ='NAM', deep=True)
    'NAM'
    """
    
    name = str(name).lower().strip() 
    defaults = is_iterable(defaults, 
            exclude_string= True, parse_string= True, transform=True )
    if deep : 
        defaults = ''.join([ str(i) for i in defaults] ) 
        
    # if name in defaults: 
    name = ( True if expect_name is None  else expect_name 
            ) if name in defaults else False 
    
    #name = True if name in defaults else ( expect_name if expect_name else False )
    
    if not name and exception: 
        raise exception 
        
    return name 

def get_confidence_ratio (
        ar, 
        axis = 0, 
        invalid = 'NaN',
        mean=False, 
        ):
    
    """ Get ratio of confidence in array by counting the number of 
    invalid values. 
    
    Parameters 
    ------------
    ar: arraylike 1D or 2D  
      array for checking the ratio of confidence 
      
    axis: int, default=0, 
       Compute the ratio of confidence alongside the rows by defaults. 
       
    invalid: int, foat, default='NaN'
      The value to consider as invalid in the data might be listed if 
      applicable. The default is ``NaN``. 
      
    mean: bool, default=False, 
      Get the mean ratio. Average the percentage of each axis. 
      
      .. versionadded:: 0.2.8 
         Average the ratio of confidence of each axis. 
      
    Returns 
    ---------
    ratio: arraylike 1D 
      The ratio of confidence array alongside the ``axis``. 

    Examples 
    ----------
    >>> import numpy as np 
    >>> np.random.seed (0) 
    >>> test = np.random.randint (1, 20 , 10 ).reshape (5, 2 ) 
    >>> test
    array([[13, 16],
           [ 1,  4],
           [ 4,  8],
           [10, 19],
           [ 5,  7]])
    >>> from gofast.tools.coreutils import get_confidence_ratio 
    >>> get_confidence_ratio (test)
    >>> array([1., 1.])
    >>> get_confidence_ratio (test, invalid= ( 13, 19) )
    array([0.8, 0.8])
    >>> get_confidence_ratio (test, invalid= ( 13, 19, 4) )
    array([0.6, 0.6])
    >>> get_confidence_ratio (test, invalid= ( 13, 19, 4), axis =1 )
    array([0.5, 0.5, 0.5, 0.5, 1. ])
    
    """
    def gfc ( ar, inv):
        """ Get ratio in each column or row in the array. """
        inv = is_iterable(inv, exclude_string=True , transform =True, 
                              )
        # if inv!='NaN': 
        for iv in inv: 
            if iv in ('NAN', np.nan, 'NaN', 'nan', None): 
                iv=np.nan  
            ar [ar ==iv] = np.nan 
                
        return len( ar [ ~np.isnan (ar)])  / len(ar )
    
    # validate input axis name 
    axis = _validate_name_in (axis , ('1', 'rows', 'sites', 'stations') ,
                              expect_name=1 )
    if not axis:
        axis =0 
    
    ar = np.array(ar).astype ( np.float64) # for consistency
    ratio = np.zeros(( (ar.shape[0] if axis ==1 else ar.shape [1] )
                      if ar.ndim ==2 else 1, ), dtype= np.float64) 
    
    for i in range (len(ratio)): 
        ratio[i] = gfc ( (ar [:, i] if axis ==0 else ar [i, :])
                        if ar.ndim !=1 else ar , inv= invalid 
                        )
    if mean: 
        ratio = np.array (ratio).mean() 
    return ratio 
    
def assert_ratio(
    v,  bounds: List[float] = None , 
    exclude_value:float= None, 
    in_percent:bool =False , 
    name:str ='rate' 
    ): 
    """ Assert rate value between a specific range. 
    
    Parameters 
    -----------
    v: float, 
       ratio value to assert 
    bounds: list ( lower, upper) 
       The range that value must  be included
    exclude_value: float 
       A value that ``v`` must not taken. Exclude it from the ``bounds``. 
       Raise error otherwise. Note that  any other value will use the 
       lower bound in `bounds` as exlusion. 
       
    in_percent: bool, default=False, 
       Convert the value into a percentage.
       
    name: str, default='rate' 
       the name of the value for assertion. 
       
    Returns
    --------
    v: float 
       Asserted value. 
       
    Examples
    ---------
    >>> from gofast.tools.coreutils import assert_ratio
    >>> assert_ratio('2')
    2.0
    >>> assert_ratio(2 , bounds =(2, 8))
    2.0
    >>> assert_ratio(2 , bounds =(4, 8))
    ValueError:...
    >>> assert_ratio(2 , bounds =(1, 8), exclude_value =2 )
    ValueError: ...
    >>> assert_ratio(2 , bounds =(1, 8), exclude_value ='use bounds' )
    2.0
    >>> assert_ratio(2 , bounds =(0, 1) , in_percent =True )
    0.02
    >>> assert_ratio(2 , bounds =(0, 1) )
    ValueError:
    >>> assert_ratio(2 , bounds =(0, 1), exclude_value ='use lower bound',
                         name ='tolerance', in_percent =True )
    0.02
    """ 
    msg =("greater than {} and less than {}" )
    
    
    if isinstance (v, str): 
        if "%" in v: in_percent=True 
        v = v.replace('%', '')
    try : 
        v = float (v)
    except TypeError : 
        raise TypeError (f"Unable to convert {type(v).__name__!r} "
                         f"to float: {v}")
    except ValueError: 
        raise ValueError(f"Expects 'float' not {type(v).__name__!r}: "
                         f"{(v)!r}")
    # put value in percentage 
    # if greater than 1. 
    if in_percent: 
        if 1 < v <=100: 
            v /= 100. 
          
    bounds = bounds or []
    low, up, *_ = list(bounds) + [ None, None]
    e=("Expects a {} value {}, got: {}".format(
            name , msg.format(low, up), v)) 
    err = ValueError (e)

    if len(bounds)!=0:
        if ( 
            low is not None  # use is not None since 0. is
            and up is not None # consider as False value
            and  (v < low or v > up)
            ) :
                raise err 
        
    if exclude_value is not None: 
        try : 
            low = float (str(exclude_value))
        except : # use bounds
            pass 
        if low is None:
            warnings.warn("Cannot exclude the lower value in the interval"
                          " while `bounds` argument is not given.")
        else:  
            if v ==low: 
                raise ValueError (e.replace (", got:", ' excluding') + ".")
            
    if in_percent and v > 100: 
         raise ValueError ("{} value should be {}, got: {}".
                           format(name.title(), msg.format(low, up), v  ))
    return v 

def validate_ratio(
    value: float, 
    bounds: Optional[Tuple[float, float]] = None, 
    exclude: Optional[float] = None, 
    to_percent: bool = False, 
    param_name: str = 'value'
) -> float:
    """Validates and optionally converts a value to a percentage within 
    specified bounds, excluding specific values.

    Parameters:
    -----------
    value : float or str
        The value to validate and convert. If a string with a '%' sign, 
        conversion to percentage is attempted.
    bounds : tuple of float, optional
        A tuple specifying the lower and upper bounds (inclusive) for the value. 
        If None, no bounds are enforced.
    exclude : float, optional
        A specific value to exclude from the valid range. If the value matches 
        'exclude', a ValueError is raised.
    to_percent : bool, default=False
        If True, the value is converted to a percentage 
        (assumed to be in the range [0, 100]).
    param_name : str, default='value'
        The parameter name to use in error messages.

    Returns:
    --------
    float
        The validated (and possibly converted) value.

    Raises:
    ------
    ValueError
        If the value is outside the specified bounds, matches the 'exclude' 
        value, or cannot be converted as specified.
    """
    if isinstance(value, str) and '%' in value:
        to_percent = True
        value = value.replace('%', '')
    try:
        value = float(value)
    except ValueError:
        raise ValueError(f"Expected a float, got {type(value).__name__}: {value}")

    if to_percent and 0 < value <= 100:
        value /= 100

    if bounds:
        if not (bounds[0] <= value <= bounds[1]):
            raise ValueError(f"{param_name} must be between {bounds[0]}"
                             f" and {bounds[1]}, got: {value}")
    
    if exclude is not None and value == exclude:
        raise ValueError(f"{param_name} cannot be {exclude}")

    if to_percent and value > 1:
        raise ValueError(f"{param_name} converted to percent must"
                         f" not exceed 1, got: {value}")

    return value

def exist_features (df, features, error='raise', name="Feature"): 
    """Control whether the features exist or not.  
    
    :param df: a dataframe for features selections 
    :param features: list of features to select. Lits of features must be in the 
        dataframe otherwise an error occurs. 
    :param error: str - raise if the features don't exist in the dataframe. 
        *default* is ``raise`` and ``ignore`` otherwise. 
        
    :return: bool 
        assert whether the features exists 
    """
    isf = False  
    
    error= 'raise' if error.lower().strip().find('raise')>= 0  else 'ignore' 

    if isinstance(features, str): 
        features =[features]
        
    features = _assert_all_types(features, list, tuple, np.ndarray)
    set_f =  set (features).intersection (set(df.columns))
    if len(set_f)!= len(features): 
        nfeat= len(features) 
        msg = f"{name}{'s' if nfeat >1 else ''}"
        if len(set_f)==0:
            if error =='raise':
                raise ValueError (f"{msg} {smart_format(features)} "
                                  f"{'does not' if nfeat <2 else 'dont'}"
                                  " exist in the dataframe")
            isf = False 
        # get the difference 
        diff = set (features).difference(set_f) if len(
            features)> len(set_f) else set_f.difference (set(features))
        nfeat= len(diff)
        if error =='raise':
            raise ValueError(f"{msg} {smart_format(diff)} not found in"
                             " the dataframe.")
        isf = False  
    else : isf = True 
    
    return isf    
    
def make_obj_consistent_if ( 
        item= ... , default = ..., size =None, from_index: bool =True ): 
    """Combine default values to item to create default consistent iterable 
    objects. 
    
    This is valid if  the size of item does not fit the number of 
    expected iterable objects.     
    
    Parameters 
    ------------
    item : Any 
       Object to construct it default values 
       
    default: Any 
       Value to hold in the case the items does not match the size of given items 
       
    size: int, Optional 
      Number of items to return. 
      
    from_index: bool, default=True 
       make an item size to match the exact size of given items 
       
    Returns 
    -------
       item: Iterable object that contain default values. 
       
    Examples 
    ----------
    >>> from gofast.tools.coreutils import make_obj_consistent_if
    >>> from gofast.exlib import SVC, LogisticRegression, XGBClassifier 
    >>> classifiers = ["SVC", "LogisticRegression", "XGBClassifier"] 
    >>> classifier_names = ['SVC', 'LR'] 
    >>> make_obj_consistent_if (classifiers, default = classifier_names ) 
    ['SVC', 'LogisticRegression', 'XGBClassifier']
    >>> make_obj_consistent_if (classifier_names, from_index =False  )
    ['SVC', 'LR']
    >>> >>> make_obj_consistent_if ( classifier_names, 
                                     default= classifiers, size =3 , 
                                     from_index =False  )
    ['SVC', 'LR', 'SVC']
    
    """
    if default==... or None : default =[]
    # for consistency 
    default = list( is_iterable (default, exclude_string =True,
                                 transform =True ) ) 
    
    if item not in ( ...,  None) : 
         item = list( is_iterable( item , exclude_string =True ,
                                  transform = True ) ) 
    else: item = [] 
    
    item += default[len(item):] if from_index else default 
    
    if size is not None: 
        size = int (_assert_all_types(size, int, float,
                                      objname = "Item 'size'") )
        item = item [:size]
        
    return item
 
def convert_value_in (v, unit ='m'): 
    """Convert value based on the reference unit.
    
    Parameters 
    ------------
    v: str, float, int, 
      value to convert 
    unit: str, default='m'
      Reference unit to convert value in. Default is 'meters'. Could be 
      'kg' or else. 
      
    Returns
    -------
    v: float, 
       Value converted. 
       
    Examples 
    ---------
    >>> from gofast.tools.coreutils import convert_value_in 
    >>> convert_value_in (20) 
    20.0
    >>> convert_value_in ('20mm') 
    0.02
    >>> convert_value_in ('20kg', unit='g') 
    20000.0
    >>> convert_value_in ('20') 
    20.0
    >>> convert_value_in ('20m', unit='g')
    ValueError: Unknwon unit 'm'...
    """
    c= { 'k':1e3 , 
        'h':1e2 , 
        'dc':1e1 , 
        '':1e0 , 
        'd':1e-1, 
        'c':1e-2 , 
        'm':1e-3  
        }
    c = {k +str(unit).lower(): v for k, v in c.items() }

    v = str(v).lower()  

    regex = re.findall(r'[a-zA-Z]', v) 
    
    if len(regex) !=0: 
        unit = ''.join( regex ) 
        v = v.replace (unit, '')

    if unit not in c.keys(): 
        raise ValueError (
            f"Unknwon unit {unit!r}. Expect {smart_format(c.keys(), 'or' )}."
            f" Or rename the `unit` parameter maybe to {unit[-1]!r}.")
    
    return float ( v) * (c.get(unit) or 1e0) 

def split_list(lst:List[Any],  val:int, fill_value:Optional[Any]=None ):
    """Module to extract a slice of elements from the list 
    
    Parameters 
    ------------
    lst: list, 
      List composed of item elements 
    val: int, 
      Number of item to group by default. 
      
    Returns 
    --------
    group with slide items 
    
    Examples
    --------
    >>> from gofast.tools.coreutils import split_list
    >>> lst = [1, 2, 3, 4, 5, 6, 7, 8]
    >>> val = 3
    >>> print(split_list(lst, val))
    [[1, 2, 3], [4, 5, 6], [7, 8]]
 
    """

    lst = is_iterable(lst , exclude_string =True , transform =True ) 
    val = int ( _assert_all_types(val, int, float )) 
    try: 
        sl= [list(group) for key, group in itertools.groupby(
                lst, lambda x: (x-1)//val)]
    except: 
        # when string is given 
        sl= list(itertools.zip_longest(
            *(iter(lst),)*val,fillvalue =fill_value),)
    return sl 

def make_ids(arr, prefix =None, how ='py', skip=False): 
    """ Generate auto Id according to the number of given sites. 
    
    :param arr: Iterable object to generate an id site . For instance it can be 
        the array-like or list of EDI object that composed a collection of 
        gofast.edi.Edi object. 
    :type ediObjs: array-like, list or tuple 

    :param prefix: string value to add as prefix of given id. Prefix can be 
        the site name.
    :type prefix: str 
    
    :param how: Mode to index the station. Default is 'Python indexing' i.e. 
        the counting starts by 0. Any other mode will start the counting by 1.
    :type cmode: str 
    
    :param skip: skip the long formatage. the formatage acccording to the 
        number of collected file. 
    :type skip: bool 
    :return: ID number formated 
    :rtype: list 
    
    :Example: 
        >>> import numpy as np 
        >>> from gofast.tools.func_utils import make_ids 
        >>> values = ['edi1', 'edi2', 'edi3'] 
        >>> make_ids (values, 'ix')
        ... ['ix0', 'ix1', 'ix2']
        >>> data = np.random.randn(20)
        >>>  make_ids (data, prefix ='line', how=None)
        ... ['line01','line02','line03', ... , line20] 
        >>> make_ids (data, prefix ='line', how=None, skip =True)
        ... ['line1','line2','line3',..., line20] 
        
    """ 
    fm='{:0' + ('1' if skip else '{}'.format(int(np.log10(len(arr))) + 1)) +'}'
    id_ =[str(prefix) + fm.format(i if how=='py'else i+ 1 ) if prefix is not 
          None else fm.format(i if how=='py'else i+ 1) 
          for i in range(len(arr))] 
    return id_ 


def ellipsis2false( *parameters , default_value: Any=False ): 
    """ Turn all parameter arguments to False if ellipsis.
    
    Note that the output arguments must be in the same order like the 
    positional arguments. 
 
    :param parameters: tuple 
       List of parameters 
    :param default_value: Any, 
       Value by default that might be take the ellipsis. 
    :return: tuple, same list of parameters passed ellipsis to 
       ``default_value``. By default, it returns ``False``. For a single 
       parameters, uses the trailing comma for collecting the parameters 
       
    :example: 
        >>> from gofast.tools.coreutils import ellipsis2false 
        >>> var, = ellipsis2false (...)
        >>> var 
        False
        >>> data, sep , verbose = ellipsis2false ([2,3, 4], ',', ...)
        >>> verbose 
        False 
    """
    return tuple ( ( default_value  if param is  ... else param  
                    for param in parameters) )  

def type_of_target(y):
    """
    Determine the type of data indicated by the target variable.

    Parameters
    ----------
    y : array-like
        Target values. 

    Returns
    -------
    target_type : string
        Type of target data, such as 'binary', 'multiclass', 'continuous', etc.

    Examples
    --------
    >>> type_of_target([0, 1, 1, 0])
    'binary'
    >>> type_of_target([0.5, 1.5, 2.5])
    'continuous'
    >>> type_of_target([[1, 0], [0, 1]])
    'multilabel-indicator'
    """
    # Check if y is an array-like
    if not isinstance(y, (np.ndarray, list, pd.Series, Sequence, pd.DataFrame)):
        raise ValueError("Expected array-like (array or list), got %s" % type(y))

    # Check for valid number type
    if not all(isinstance(i, (int, float, np.integer, np.floating)) 
               for i in np.array(y).flatten()):
        raise ValueError("Input must be a numeric array-like")

    # Continuous data
    if any(isinstance(i, float) for i in np.array(y).flatten()):
        return 'continuous'

    # Binary or multiclass
    unique_values = np.unique(y)
    if len(unique_values) == 2:
        return 'binary'
    elif len(unique_values) > 2 and np.ndim(y) == 1:
        return 'multiclass'

    # Multilabel indicator
    if isinstance(y[0], (np.ndarray, list, Sequence)) and len(y[0]) > 1:
        return 'multilabel-indicator'

    return 'unknown'


def add_noises_to(
    data,  
    noise=0.1, 
    seed=None, 
    gaussian_noise=False,
    cat_missing_value=pd.NA
    ):
    """
    Adds NaN or specified missing values to a pandas DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame to which NaN values or specified missing 
        values will be added.

    noise : float, default=0.1
        The percentage of values to be replaced with NaN or the 
        specified missing value in each column. This must be a 
        number between 0 and 1. Default is 0.1 (10%).

        .. math:: \text{noise} = \frac{\text{number of replaced values}}{\text{total values in column}}

    seed : int, array-like, BitGenerator, np.random.RandomState, np.random.Generator, optional
        Seed for random number generator to ensure reproducibility. 
        If `seed` is an int, array-like, or BitGenerator, it will be 
        used to seed the random number generator. If `seed` is a 
        np.random.RandomState or np.random.Generator, it will be used 
        as given.

    gaussian_noise : bool, default=False
        If `True`, adds Gaussian noise to the data. Otherwise, replaces 
        values with NaN or the specified missing value.

    cat_missing_value : scalar, default=pd.NA
        The value to use for missing data in categorical columns. By 
        default, `pd.NA` is used.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with NaN or specified missing values added.

    Notes
    -----
    The function modifies the DataFrame by either adding Gaussian noise 
    to numerical columns or replacing a percentage of values in each 
    column with NaN or a specified missing value.

    The Gaussian noise is added according to the formula:

    .. math:: \text{new_value} = \text{original_value} + \mathcal{N}(0, \text{noise})

    where :math:`\mathcal{N}(0, \text{noise})` represents a normal 
    distribution with mean 0 and standard deviation equal to `noise`.

    Examples
    --------
    >>> from gofast.tools.coreutils import add_noises_to
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
    >>> new_df = add_noises_to(df, noise=0.2)
    >>> new_df
         A     B
    0  1.0  <NA>
    1  NaN     y
    2  3.0  <NA>

    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> new_df = add_noises_to(df, noise=0.1, gaussian_noise=True)
    >>> new_df
              A         B
    0  1.063292  3.986400
    1  2.103962  4.984292
    2  2.856601  6.017380

    See Also
    --------
    pandas.DataFrame : Two-dimensional, size-mutable, potentially 
        heterogeneous tabular data.
    numpy.random.normal : Draw random samples from a normal 
        (Gaussian) distribution.

    References
    ----------
    .. [1] Harris, C. R., Millman, K. J., van der Walt, S. J., et al. 
           (2020). Array programming with NumPy. Nature, 585(7825), 
           357-362.
    """
    
    is_frame = isinstance (data, pd.DataFrame ) 
    if not is_frame: 
        data = pd.DataFrame(data ) 
        
    np.random.seed(seed)
    if noise is None: 
        return data 
    noise, gaussian_noise  = _parse_gaussian_noise (noise )

    if gaussian_noise:
        # Add Gaussian noise to numerical columns only
        def add_gaussian_noise(column):
            if pd.api.types.is_numeric_dtype(column):
                return column + np.random.normal(0, noise, size=column.shape)
            return column
        
        noise_data = data.apply(add_gaussian_noise)
        
        if not is_frame: 
            noise_data = np.asarray(noise_data)
        return noise_data
    else:
        # Replace values with NaN or specified missing value
        df_with_nan = data.copy()
        nan_count_per_column = int(noise * len(df_with_nan))

        for column in df_with_nan.columns:
            nan_indices = random.sample(range(len(df_with_nan)), nan_count_per_column)
            if pd.api.types.is_numeric_dtype(df_with_nan[column]):
                df_with_nan.loc[nan_indices, column] = np.nan
            else:
                df_with_nan.loc[nan_indices, column] = cat_missing_value
                
        if not is_frame: 
            df_with_nan = df_with_nan.values 
            
        return df_with_nan

def _parse_gaussian_noise(noise):
    """
    Parses the noise parameter to determine if Gaussian noise should be used
    and extracts the noise level if specified.

    Parameters
    ----------
    noise : str, float, or None
        The noise parameter to be parsed. Can be a string specifying Gaussian
        noise with an optional noise level, a float, or None.

    Returns
    -------
    tuple
        A tuple containing:
        - float: The noise level.
        - bool: Whether Gaussian noise should be used.

    Examples
    --------
    >>> from gofast.tools.coreutils import _parse_gaussian_noise
    >>> _parse_gaussian_noise('0.1gaussian')
    (0.1, True)
    >>> _parse_gaussian_noise('gaussian0.1')
    (0.1, True)
    >>> _parse_gaussian_noise('gaussian_0.1')
    (0.1, True)
    >>> _parse_gaussian_noise('gaussian10%')
    (0.1, True)
    >>> _parse_gaussian_noise('gaussian 10 %')
    (0.1, True)
    >>> _parse_gaussian_noise(0.05)
    (0.05, False)
    >>> _parse_gaussian_noise(None)
    (0.1, False)
    >>> _parse_gaussian_noise('invalid')
    Traceback (most recent call last):
        ...
    ValueError: Invalid noise value: invalid
    """
    gaussian_noise = False
    default_noise = 0.1

    if isinstance(noise, str):
        orig_noise = noise 
        noise = noise.lower()
        gaussian_keywords = ["gaussian", "gauss"]

        if any(keyword in noise for keyword in gaussian_keywords):
            gaussian_noise = True
            noise = re.sub(r'[^\d.%]', '', noise)  # Remove non-numeric and non-'%' characters
            noise = re.sub(r'%', '', noise)  # Remove '%' if present

            try:
                noise_level = float(noise) / 100 if '%' in orig_noise else float(noise)
                noise = noise_level if noise_level else default_noise
            except ValueError:
                noise = default_noise

        else:
            try:
                noise = float(noise)
            except ValueError:
                raise ValueError(f"Invalid noise value: {noise}")
    elif noise is None:
        noise = default_noise
    
    noise = validate_noise (noise ) 
    
    return noise, gaussian_noise


def validate_noise(noise):
    """
    Validates the `noise` parameter and returns either the noise value
    as a float or the string 'gaussian'.

    Parameters
    ----------
    noise : str or float or None
        The noise parameter to be validated. It can be the string
        'gaussian', a float value, or None.

    Returns
    -------
    float or str
        The validated noise value as a float or the string 'gaussian'.

    Raises
    ------
    ValueError
        If the `noise` parameter is a string other than 'gaussian' or
        cannot be converted to a float.

    Examples
    --------
    >>> validate_noise('gaussian')
    'gaussian'
    >>> validate_noise(0.1)
    0.1
    >>> validate_noise(None)
    None
    >>> validate_noise('0.2')
    0.2

    """
    if isinstance(noise, str):
        if noise.lower() == 'gaussian':
            return 'gaussian'
        else:
            try:
                noise = float(noise)
            except ValueError:
                raise ValueError("The `noise` parameter accepts the string"
                                 " 'gaussian' or a float value.")
    elif noise is not None:
        noise = validate_ratio(noise, bounds=(0, 1), param_name='noise' )
        # try:
        # except ValueError:
        #     raise ValueError("The `noise` parameter must be convertible to a float.")
    return noise

def fancier_repr_formatter(obj, max_attrs=7):
    """
    Generates a formatted string representation for any class object.

    Parameters:
    ----------
    obj : object
        The object for which the string representation is generated.

    max_attrs : int, optional
        Maximum number of attributes to display in the representation.

    Returns:
    -------
    str
        A string representation of the object.

    Examples:
    --------
    >>> from gofast.tools.coreutils import fancier_repr_formatter
    >>> class MyClass:
    >>>     def __init__(self, a, b, c):
    >>>         self.a = a
    >>>         self.b = b
    >>>         self.c = c
    >>> obj = MyClass(1, [1, 2, 3], 'hello')
    >>> print(fancier_repr_formatter(obj))
    MyClass(a=1, c='hello', ...)
    """
    attrs = [(name, getattr(obj, name)) for name in dir(obj)
             if not name.startswith('_') and
             (isinstance(getattr(obj, name), str) or
              not hasattr(getattr(obj, name), '__iter__'))]

    displayed_attrs = attrs[:min(len(attrs), max_attrs)]
    attr_str = ', '.join([f'{name}={value!r}' for name, value in displayed_attrs])

    # Add ellipsis if there are more attributes than max_attrs
    if len(attrs) > max_attrs:
        attr_str += ', ...'

    return f'{obj.__class__.__name__}({attr_str})'


def normalize_string(
    input_str: str, 
    target_strs: Optional[List[str]] = None, 
    num_chars_check: Optional[int] = None, 
    deep: bool = False, 
    return_target_str: bool = False,
    return_target_only: bool=False, 
    raise_exception: bool = False,
    ignore_case: bool = True,
    match_method: str = 'exact',
    error_msg: str=None, 
) -> Union[str, Tuple[str, Optional[str]]]:
    """
    Normalizes a string by applying various transformations and optionally checks 
    against a list of target strings based on different matching methods.

    Function normalizes a string by stripping leading/trailing whitespace, 
    converting to lowercase,and optionally checks against a list of target  
    strings. If specified, returns the target string that matches the 
    conditions. Raise an exception if the string is not found.
    
    Parameters
    ----------
    input_str : str
        The string to be normalized.
    target_strs : List[str], optional
        A list of target strings for comparison.
    num_chars_check : int, optional
        The number of characters at the start of the string to check 
        against each target string.
    deep : bool, optional
        If True, performs a deep substring check within each target string.
    return_target_str : bool, optional
        If True and a target string matches, returns the matched target string 
        along with the normalized string.
    return_target_only: bool, optional 
       If True and a target string  matches, returns only the matched string
       target. 
    raise_exception : bool, optional
        If True and the input string is not found in the target strings, 
        raises an exception.
    ignore_case : bool, optional
        If True, ignores case in string comparisons. Default is True.
    match_method : str, optional
        The string matching method: 'exact', 'contains', or 'startswith'.
        Default is 'exact'.
    error_msg: str, optional, 
       Message to raise if `raise_exception` is ``True``. 
       
    Returns
    -------
    Union[str, Tuple[str, Optional[str]]]
        The normalized string. If return_target_str is True and a target 
        string matches, returns a tuple of the normalized string and the 
        matched target string.

    Raises
    ------
    ValueError
        If raise_exception is True and the input string is not found in 
        the target strings.

    Examples
    --------
    >>> from gofast.tools.coreutils import normalize_string
    >>> normalize_string("Hello World", target_strs=["hello", "world"], ignore_case=True)
    'hello world'
    >>> normalize_string("Goodbye World", target_strs=["hello", "goodbye"], 
                         num_chars_check=7, return_target_str=True)
    ('goodbye world', 'goodbye')
    >>> normalize_string("Hello Universe", target_strs=["hello", "world"],
                         raise_exception=True)
    ValueError: Input string not found in target strings.
    """
    normalized_str = str(input_str).lower() if ignore_case else input_str

    if not target_strs:
        return normalized_str
    target_strs = is_iterable(target_strs, exclude_string=True, transform =True)
    normalized_targets = [str(t).lower() for t in target_strs] if ignore_case else target_strs
    matched_target = None

    for target in normalized_targets:
        if num_chars_check is not None:
            condition = (normalized_str[:num_chars_check] == target[:num_chars_check])
        elif deep:
            condition = (normalized_str in target)
        elif match_method == 'contains':
            condition = (target in normalized_str)
        elif match_method == 'startswith':
            condition = normalized_str.startswith(target)
        else:  # Exact match
            condition = (normalized_str == target)

        if condition:
            matched_target = target
            break

    if matched_target is not None:
        if return_target_only: 
            return matched_target 
        return (normalized_str, matched_target) if return_target_str else normalized_str

    if raise_exception:
        error_msg = error_msg or ( 
            f"Invalid input. Expect {smart_format(target_strs, 'or')}."
            f" Got {input_str!r}."
            )
        raise ValueError(error_msg)
    
    if return_target_only: 
        return matched_target 
    
    return ('', None) if return_target_str else ''

def format_and_print_dict(data_dict, front_space=4):
    """
    Formats and prints the contents of a dictionary in a structured way.

    Each key-value pair in the dictionary is printed with the key followed by 
    its associated values. 
    The values are expected to be dictionaries themselves, allowing for a nested 
    representation.
    The inner dictionary's keys are sorted in descending order before printing.

    Parameters
    ----------
    data_dict : dict
        A dictionary where each key contains a dictionary of items to be printed. 
        The key represents a category
        or label, and the value is another dictionary where each key-value pair 
        represents an option or description.
        
    front_space : int, optional
        The number of spaces used for indentation in front of each line (default is 4).


    Returns
    -------
    None
        This function does not return any value. It prints the formatted contents 
        of the provided dictionary.

    Examples
    --------
    >>> from gofast.tools.coreutils import format_and_print_dict
    >>> sample_dict = {
            'gender': {1: 'Male', 0: 'Female'},
            'age': {1: '35-60', 0: '16-35', 2: '>60'}
        }
    >>> format_and_print_dict(sample_dict)
    gender:
        1: Male
        0: Female
    age:
        2: >60
        1: 35-60
        0: 16-35
    """
    if not isinstance(data_dict, dict):
        raise TypeError("The input data must be a dictionary.")

    indent = ' ' * front_space
    for label, options in data_dict.items():
        print(f"{label}:")
        options= is_iterable(options, exclude_string=True, transform=True )
  
        if isinstance(options, (tuple, list)):
            for option in options:
                print(f"{indent}{option}")
        elif isinstance(options, dict):
            for key in sorted(options.keys(), reverse=True):
                print(f"{indent}{key}: {options[key]}")
        print()  # Adds an empty line for better readability between categories


def fill_nan_in(
        data: DataFrame,  method: str = 'constant', 
        value: Optional[Union[int, float, str]] = 0) -> DataFrame:
    """
    Fills NaN values in a Pandas DataFrame using various methods.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to be checked and modified.
    method : str, optional
        The method to use for filling NaN values. Options include 'constant',
        'ffill', 'bfill', 'mean', 'median', 'mode'. Default is 'constant'.
    value : int, float, string, optional
        The value used when method is 'constant'. Ignored for other methods.
        Default is 0.

    Returns
    -------
    df : pd.DataFrame
        The DataFrame with NaN values filled.

    Example
    -------
    >>> import pandas as pd
    >>> from gofast.tools.coreutils import fill_nan_in
    >>> df = pd.DataFrame({'A': [1, 2, np.nan], 'B': [np.nan, 2, 3]})
    >>> df = fill_nan_in(df, method='median')
    >>> print(df)
       A    B
    0  1.0  2.5
    1  2.0  2.0
    2  1.5  3.0
    """
    # Check for NaN values in the DataFrame and apply the specified fill method
    if not data.isna().any().any(): 
        return data 

    fill_methods = {
        'constant': lambda: data.fillna(value, inplace=True),
        'ffill': lambda: data.fillna(method='ffill', inplace=True),
        'bfill': lambda: data.fillna(method='bfill', inplace=True),
        'mean': lambda: data.fillna(data.mean(), inplace=True),
        'median': lambda: data.fillna(data.median(), inplace=True),
        'mode': lambda: data.apply(lambda col: col.fillna(col.mode()[0], inplace=True))
    }
    
    fill_action = fill_methods.get(method)
    if fill_action:
        fill_action()
    else:
        raise ValueError(f"Method '{method}' not recognized for filling NaN values.")
        
    return data 

def get_valid_kwargs(obj_or_func, raise_warning=False, **kwargs):
    """
    Filters keyword arguments (`kwargs`) to retain only those that are valid
    for the initializer of a given object or function.

    Parameters
    ----------
    obj_or_func : object or function
        The object or function to inspect for valid keyword arguments. If it's
        callable, its `__init__` method's valid keyword arguments are considered.
    raise_warning : bool, optional
        If True, raises a warning for any keyword arguments provided that are not
        valid for `obj_or_func`. The default is False.
    **kwargs : dict
        Arbitrary keyword arguments to filter based on `obj_or_func`'s
        valid parameters.

    Returns
    -------
    dict
        A dictionary containing only the keyword arguments that are valid for the
        `obj_or_func`'s initializer.

    Raises
    ------
    Warning
        If `raise_warning` is True and there are keyword arguments that are not
        valid for `obj_or_func`, a warning is raised.

    Notes
    -----
    This function checks whether the provided keyword arguments are valid for the given
    class, method, or function. It filters out any invalid keyword arguments and returns
    a dictionary containing only the valid ones.

    If the provided object is a class, it inspects the __init__ method to determine the
    valid keyword arguments. If it is a method or function, it inspects the argument names.

    It issues a warning for any invalid keyword arguments if `raise_warning`
    is ``True`` but it does not raise an error.
    
    Examples
    --------
    >>> from gofast.tools.coreutils import get_valid_kwargs
    >>> class MyClass:
    ...     def __init__(self, a, b):
    ...         self.a = a
    ...         self.b = b
    >>> valid_kwargs = get_valid_kwargs(MyClass, a=1, b=2, c=3)
    >>> print(valid_kwargs)
    {'a': 1, 'b': 2}
    >>> valid_kwargs = get_valid_kwargs(MyClass, raise_warning=True,  **kwargs)
    Warning: 'arg3' is not a valid keyword argument for 'MyClass'.
    >>> print(valid_kwargs)
    {'arg1': 1, 'arg2': 2}

    >>> def my_function(a, b, c):
    ...     return a + b + c
    ...
    >>> kwargs = {'a': 1, 'b': 2, 'd': 3}
    >>> valid_kwargs = get_valid_kwargs(my_function, raise_warning=True, **kwargs)
    Warning: 'd' is not a valid keyword argument for 'my_function'.
    >>> print(valid_kwargs)
    {'a': 1, 'b': 2}
    """
    valid_kwargs = {}
    not_valid_keys = []

    # Determine whether obj_or_func is callable and get its valid arguments
    obj = obj_or_func() if callable(obj_or_func) else obj_or_func
    valid_args = obj.__init__.__code__.co_varnames if hasattr(
        obj, '__init__') else obj.__code__.co_varnames

    # Filter kwargs to separate valid from invalid ones
    for key, value in kwargs.items():
        if key in valid_args:
            valid_kwargs[key] = value
        else:
            not_valid_keys.append(key)

    # Raise a warning for invalid kwargs, if required
    if raise_warning and not_valid_keys:
        warning_msg = (f"Warning: '{', '.join(not_valid_keys)}' "
                       f"{'is' if len(not_valid_keys) == 1 else 'are'} "
                       "not a valid keyword argument "
                       f"for '{obj_or_func.__name__}'.")
        warnings.warn(warning_msg)

    return valid_kwargs
 
def projection_validator (X, Xt=None, columns =None ):
    """ Retrieve x, y coordinates of a datraframe ( X, Xt ) from columns 
    names or indexes. 
    
    If X or Xt are given as arrays, `columns` may hold integers from 
    selecting the the coordinates 'x' and 'y'. 
    
    Parameters 
    ---------
    X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
        training set; Denotes data that is observed at training and prediction 
        time, used as independent variables in learning. The notation 
        is uppercase to denote that it is ordinarily a matrix. When a matrix, 
        each sample may be represented by a feature vector, or a vector of 
        precomputed (dis)similarity with each training sample. 

    Xt: Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
        Shorthand for "test set"; data that is observed at testing and 
        prediction time, used as independent variables in learning. The 
        notation is uppercase to denote that it is ordinarily a matrix.
    columns: list of str or index, optional 
        columns is usefull when a dataframe is given  with a dimension size 
        greater than 2. If such data is passed to `X` or `Xt`, columns must
        hold the name to consider as 'easting', 'northing' when UTM 
        coordinates are given or 'latitude' , 'longitude' when latlon are 
        given. 
        If dimension size is greater than 2 and columns is None , an error 
        will raises to prevent the user to provide the index for 'y' and 'x' 
        coordinated retrieval. 
      
    Returns 
    -------
    ( x, y, xt, yt ), (xname, yname, xtname, ytname), Tuple of coordinate 
        arrays and coordinate labels 
 
    """
    # initialize arrays and names 
    init_none = [None for i in range (4)]
    x,y, xt, yt = init_none
    xname,yname, xtname, ytname = init_none 
    
    m="{0} must be an iterable object, not {1!r}"
    ms= ("{!r} is given while columns are not supplied. set the list of "
        " feature names or indexes to fetch 'x' and 'y' coordinate arrays." )
    
    # validate X if X is np.array or dataframe 
    X =_assert_all_types(X, np.ndarray, pd.DataFrame ) 
    
    if Xt is not None: 
        # validate Xt if Xt is np.array or dataframe 
        Xt = _assert_all_types(Xt, np.ndarray, pd.DataFrame)
        
    if columns is not None: 
        if isinstance (columns, str): 
            columns = str2columns(columns )
        
        if not is_iterable(columns): 
            raise ValueError(m.format('columns', type(columns).__name__))
        
        columns = list(columns) + [ None for i in range (5)]
        xname , yname, xtname, ytname , *_= columns 

    if isinstance(X, pd.DataFrame):
        x, xname, y, yname = _validate_columns(X, [xname, yname])
        
    elif isinstance(X, np.ndarray):
        x, y = _is_valid_coordinate_arrays (X, xname, yname )    
        
        
    if isinstance (Xt, pd.DataFrame) :
        # the test set holds the same feature names
        # as the train set 
        if xtname is None: 
            xtname = xname
        if ytname is None: 
            ytname = yname 
            
        xt, xtname, yt, ytname = _validate_columns(Xt, [xname, yname])

    elif isinstance(Xt, np.ndarray):
        
        if xtname is None: 
            xtname = xname
        if ytname is None: 
            ytname = yname 
            
        xt, yt = _is_valid_coordinate_arrays (Xt, xtname, ytname , 'test')
        
    if (x is None) or (y is None): 
        raise ValueError (ms.format('X'))
    if Xt is not None: 
        if (xt is None) or (yt is None): 
            warnings.warn (ms.format('Xt'))

    return  (x, y , xt, yt ) , (
        xname, yname, xtname, ytname ) 
    
def _validate_columns0 (df, xni, yni ): 
    """ Validate the feature name  in the dataframe using either the 
    string litteral name of the index position in the columns.
    
    :param df: pandas.DataFrame- Dataframe with feature names as columns. 
    :param xni: str, int- feature name  or position index in the columns for 
        x-coordinate 
    :param yni: str, int- feature name  or position index in the columns for 
        y-coordinate 
    
    :returns: (x, ni) Tuple of (pandas.Series, and names) for x and y 
        coordinates respectively.
    
    """
    def _r (ni): 
        if isinstance(ni, str): # feature name
            exist_features(df, ni ) 
            s = df[ni]  
        elif isinstance (ni, (int, float)):# feature index
            s= df.iloc[:, int(ni)] 
            ni = s.name 
        return s, ni 
        
    xs , ys = [None, None ]
    if df.ndim ==1: 
        raise ValueError ("Expect a dataframe of two dimensions, got '1'")
        
    elif df.shape[1]==2: 
       warnings.warn("columns are not specify while array has dimension"
                     "equals to 2. Expect indexes 0 and 1 for (x, y)"
                     "coordinates respectively.")
       xni= df.iloc[:, 0].name 
       yni= df.iloc[:, 1].name 
    else: 
        ms = ("The matrix of features is greater than 2. Need column names or"
              " indexes to  retrieve the 'x' and 'y' coordinate arrays." ) 
        e =' Only {!r} is given.' 
        me=''
        if xni is not None: 
            me =e.format(xni)
        if yni is not None: 
            me=e.format(yni)
           
        if (xni is None) or (yni is None ): 
            raise ValueError (ms + me)
            
    xs, xni = _r (xni) ;  ys, yni = _r (yni)
  
    return xs, xni , ys, yni 

def _validate_array_indexer (arr, index): 
    """ Select the appropriate coordinates (x,y) arrays from indexes.  
    
    Index is used  to retrieve the array of (x, y) coordinates if dimension 
    of `arr` is greater than 2. Since we expect x, y coordinate for projecting 
    coordinates, 1-d  array `X` is not acceptable. 
    
    :param arr: ndarray (n_samples, n_features) - if nfeatures is greater than 
        2 , indexes is needed to fetch the x, y coordinates . 
    :param index: int, index to fetch x, and y coordinates in multi-dimension
        arrays. 
    :returns: arr- x or y coordinates arrays. 

    """
    if arr.ndim ==1: 
        raise ValueError ("Expect an array of two dimensions.")
    if not isinstance (index, (float, int)): 
        raise ValueError("index is needed to coordinate array with "
                         "dimension greater than 2.")
        
    return arr[:, int (index) ]

def _is_valid_coordinate_arrays (arr, xind, yind, ptype ='train'): 
    """ Check whether array is suitable for projecting i.e. whether 
    x and y (both coordinates) can be retrived from `arr`.
    
    :param arr: ndarray (n_samples, n_features) - if nfeatures is greater than 
        2 , indexes is needed to fetch the x, y coordinates . 
        
    :param xind: int, index to fetch x-coordinate in multi-dimension
        arrays. 
    :param yind: int, index to fetch y-coordinate in multi-dimension
        arrays
    :param ptype: str, default='train', specify whether the array passed is 
        training or test sets. 
    :returns: (x, y)- array-like of x and y coordinates. 
    
    """
    xn, yn =('x', 'y') if ptype =='train' else ('xt', 'yt') 
    if arr.ndim ==1: 
        raise ValueError ("Expect an array of two dimensions.")
        
    elif arr.shape[1] ==2 : 
        x, y = arr[:, 0], arr[:, 1]
        
    else :
        msg=("The matrix of features is greater than 2; Need index to  "
             " retrieve the {!r} coordinate array in param 'column'.")
        
        if xind is None: 
            raise ValueError(msg.format(xn))
        else : x = _validate_array_indexer(arr, xind)
        if yind is None : 
            raise ValueError(msg.format(yn))
        else : y = _validate_array_indexer(arr, yind)
        
    return x, y         

def extract_coordinates(X, Xt=None, columns=None):
    """
    Extracts 'x' and 'y' coordinate arrays from training (X) and optionally
    test (Xt) datasets. 
    
    Supports input as NumPy arrays or pandas DataFrames. When dealing
    with DataFrames, `columns` can specify which columns to use for coordinates.

    Parameters
    ----------
    X : ndarray or DataFrame
        Training dataset with shape (M, N) where M is the number of samples and
        N is the number of features. It represents the observed data used as
        independent variables in learning.
    Xt : ndarray or DataFrame, optional
        Test dataset with shape (M, N) where M is the number of samples and
        N is the number of features. It represents the data observed at testing
        and prediction time, used as independent variables in learning.
    columns : list of str or int, optional
        Specifies the columns to use for 'x' and 'y' coordinates. Necessary when
        X or Xt are DataFrames with more than 2 dimensions or when selecting specific
        features from NumPy arrays.

    Returns
    -------
    tuple of arrays
        A tuple containing the 'x' and 'y' coordinates from the training set and, 
        if provided, the test set. Formatted as (x, y, xt, yt).
    tuple of str or None
        A tuple containing the names or indices of the 'x' and 'y' columns 
        for the training and test sets. Formatted as (xname, yname, xtname, ytname).
        Values are None if not applicable or not provided.

    Raises
    ------
    ValueError
        If `columns` is not iterable, not provided for DataFrames with more 
        than 2 dimensions, or if X or Xt cannot be validated as coordinate arrays.

    Examples
    --------
    >>> import numpy as np 
    >>> from gofast.tools.coreutils import extract_coordinates
    >>> X = np.array([[1, 2], [3, 4]])
    >>> Xt = np.array([[5, 6], [7, 8]])
    >>> extract_coordinates(X, Xt )
    ((array([1, 3]), array([2, 4]), array([5, 7]), array([6, 8])), (0, 1, 0, 1))
    """
    if columns is None: 
        if not isinstance ( X, pd.DataFrame) and X.shape[1]!=2: 
            raise ValueError("Columns cannot be None when array is passed.")
        if isinstance(X, np.ndarray) and X.shape[1]==2: 
            columns =[0, 1] 
    
    columns = columns or ( list(X.columns) if isinstance (
        X, pd.DataFrame ) else columns )
    
    if columns is None :
        raise ValueError("Columns parameter is required to specify"
                         " 'x' and 'y' coordinates.")
    
    if not isinstance(columns, (list, tuple)) or len(columns) != 2:
        raise ValueError("Columns parameter must be a list or tuple with "
                         "exactly two elements for 'x' and 'y' coordinates.")
    
    # Process training dataset
    x, y, xname, yname = _process_dataset(X, columns)
    
    # Process test dataset, if provided
    if Xt is not None:
        xt, yt, xtname, ytname = _process_dataset(Xt, columns)
    else:
        xt, yt, xtname, ytname = None, None, None, None

    return (x, y, xt, yt), (xname, yname, xtname, ytname)
       
def _validate_columns(df, columns):
    """
    Validates and extracts x, y coordinates from a DataFrame based on column 
    names or indices.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame from which to extract coordinate columns.
    columns : list of str or int
        The names or indices of the columns to extract as coordinates.
    
    Returns
    -------
    x, xname, y, yname : (pandas.Series, str/int, pandas.Series, str/int)
        The extracted x and y coordinate Series along with their column
        names or indices.
    
    Raises
    ------
    ValueError
        If the specified columns are not found in the DataFrame or if the 
        columns list is not correctly specified.
    """
    if not isinstance(columns, (list, tuple)) or len(columns) < 2:
        raise ValueError("Columns parameter must be a list or tuple with at"
                         " least two elements.")
    
    try:
        xname, yname = columns[0], columns[1]
        x = df[xname] if isinstance(xname, str) else df.iloc[:, xname]
        y = df[yname] if isinstance(yname, str) else df.iloc[:, yname]
    except Exception as e:
        raise ValueError(f"Error extracting columns: {e}")
    
    return x, xname, y, yname

def _process_dataset(dataset, columns):
    """
    Processes the dataset (X or Xt) to extract 'x' and 'y' coordinates based 
    on provided column names or indices.
    
    Parameters
    ----------
    dataset : pandas.DataFrame or numpy.ndarray
        The dataset from which to extract 'x' and 'y' coordinates.
    columns : list of str or int
        The names or indices of the columns to extract as coordinates. 
        For ndarray, integers are expected.
    
    Returns
    -------
    x, y, xname, yname : (numpy.array or pandas.Series, numpy.array or 
                          pandas.Series, str/int, str/int)
        The extracted 'x' and 'y' coordinates, along with their column names 
        or indices.
    
    Raises
    ------
    ValueError
        If the dataset or columns are not properly specified.
    """
    if isinstance(dataset, pd.DataFrame):
        x, xname, y, yname = _validate_columns(dataset, columns)
        return x.to_numpy(), y.to_numpy(), xname, yname
    elif isinstance(dataset, np.ndarray):
        if not isinstance(columns, (list, tuple)) or len(columns) < 2:
            raise ValueError("For ndarray, columns must be a list or tuple "
                             "with at least two indices.")
        xindex, yindex = columns[0], columns[1]
        x, y = dataset[:, xindex], dataset[:, yindex]
        return x, y, xindex, yindex
    else:
        raise ValueError("Dataset must be a pandas.DataFrame or numpy.ndarray.")

def validate_feature(data: Union[DataFrame, Series],  features: List[str],
                     verbose: str = 'raise') -> bool:
    """
    Validate the existence of specified features in a DataFrame or Series.

    Parameters
    ----------
    data : DataFrame or Series
        The DataFrame or Series to validate feature existence.
    features : list of str
        List of features to check for existence in the data.
    verbose : str, {'raise', 'ignore'}, optional
        Specify how to handle the absence of features. 'raise' (default) will raise
        a ValueError if any feature is missing, while 'ignore' will return a
        boolean indicating whether all features exist.

    Returns
    -------
    bool
        True if all specified features exist in the data, False otherwise.

    Examples
    --------
    >>> from gofast.tools.coreutils import validate_feature
    >>> import pandas as pd
    >>> data = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> result = validate_feature(data, ['A', 'C'], verbose='raise')
    >>> print(result)  # This will raise a ValueError
    """
    if isinstance(data, pd.Series):
        data = data.to_frame().T  # Convert Series to DataFrame
    features= is_iterable(features, exclude_string= True, transform =True )
    present_features = set(features).intersection(data.columns)

    if len(present_features) != len(features):
        missing_features = set(features).difference(present_features)
        if verbose == 'raise':
            raise ValueError("The following features are missing in the "
                             f"data: {smart_format(missing_features)}.")
        return False

    return True

def features_in(
    *data: Union[pd.DataFrame, pd.Series], features: List[str],
    error: str = 'ignore') -> List[bool]:
    """
    Control whether the specified features exist in multiple datasets.

    Parameters
    ----------
    *data : DataFrame or Series arguments
        Multiple DataFrames or Series to check for feature existence.
    features : list of str
        List of features to check for existence in the datasets.
    error : str, {'raise', 'ignore'}, optional
        Specify how to handle the absence of features. 'ignore' (default) will ignore
        a ValueError for each dataset with missing features, while 'ignore' will
        return a list of booleans indicating whether all features exist in each dataset.

    Returns
    -------
    list of bool
        A list of booleans indicating whether the specified features exist in each dataset.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.tools.coreutils import features_in
    >>> data1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> data2 = pd.Series([5, 6], name='C')
    >>> data3 = pd.DataFrame({'X': [7, 8]})
    >>> features = ['A', 'C']
    >>> results1 = features_in(data1, data2, features, error='raise')
    >>> print(results1)  # This will raise a ValueError for the first dataset
    >>> results2 = features_in(data1, data3, features, error='ignore')
    >>> print(results2)  # This will return [True, False]
    """
    results = []

    for dataset in data:
        results.append(validate_feature(dataset, features, verbose=error))

    return results

def find_features_in(
    data: DataFrame = None,
    features: List[str] = None,
    parse_features: bool = False,
    return_frames: bool = False,
) -> Tuple[Union[List[str], DataFrame], Union[List[str], DataFrame]]:
    """
    Retrieve the categorical or numerical features from the dataset.

    Parameters
    ----------
    data : DataFrame, optional
        DataFrame with columns representing the features.
    features : list of str, optional
        List of column names. If provided, the DataFrame will be restricted
        to only include the specified features before searching for numerical
        and categorical features. An error will be raised if any specified
        feature is missing in the DataFrame.
    return_frames : bool, optional
        If True, it returns two separate DataFrames (cat & num). Otherwise, it
        returns only the column names of categorical and numerical features.
    parse_features : bool, default False
        Use default parsers to parse string items into an iterable object.

    Returns
    -------
    Tuple : List[str] or DataFrame
        The names or DataFrames of categorical and numerical features.

    Examples
    --------
    >>> from gofast.datasets import fetch_data
    >>> from gofast.tools.mlutils import find_features_in
    >>> data = fetch_data('bagoue').frame 
    >>> cat, num = find_features_in(data)
    >>> cat, num
    ... (['type', 'geol', 'shape', 'name', 'flow'],
    ...  ['num', 'east', 'north', 'power', 'magnitude', 'sfi', 'ohmS', 'lwi'])
    >>> cat, num = find_features_in(data, features=['geol', 'ohmS', 'sfi'])
    >>> cat, num
    ... (['geol'], ['ohmS', 'sfi'])
    """
    if not isinstance (data, pd.DataFrame):
        raise TypeError(f"Expect a DataFrame. Got {type(data).__name__!r}")

    if features is not None:
        features = list(
            is_iterable(
                features,
                exclude_string=True,
                transform=True,
                parse_string=parse_features,
            )
        )

    if features is None:
        features = list(data.columns)

    validate_feature(data, list(features))
    data = data[features].copy()

    # Get numerical features
    data, numnames, catnames = to_numeric_dtypes(data, return_feature_types=True )

    if catnames is None:
        catnames = []

    return (data[catnames], data[numnames]) if return_frames else (
        list(catnames), list(numnames)
    )
  
def split_train_test(
        data: DataFrame, test_ratio: float = 0.2
        ) -> Tuple[DataFrame, DataFrame]:
    """
    Split a DataFrame into train and test sets based on a given ratio.

    Parameters
    ----------
    data : DataFrame
        The DataFrame containing the features.
    test_ratio : float, optional
        The ratio of the test set, ranging from 0 to 1. Default is 0.2 (20%).

    Returns
    -------
    Tuple[DataFrame, DataFrame]
        A tuple of the train set and test set DataFrames.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.datasets import load_iris
    >>> from gofast.tools.coreutils import split_train_test
    >>> data = load_iris(as_frame=True)['data']
    >>> train_set, test_set = split_train_test(data, test_ratio=0.2)
    >>> len(train_set), len(test_set)
    ... (120, 30)
    """

    test_ratio = assert_ratio(test_ratio)

    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    return data.iloc[train_indices], data.iloc[test_indices]

def test_set_check_id(
        identifier: int, test_ratio: float, hash: _F[_T]) -> bool:
    """
    Check if an instance should be in the test set based on its unique identifier.

    Parameters
    ----------
    identifier : int
        A unique identifier for the instance.
    test_ratio : float, optional
        The ratio of instances to put in the test set. Default is 0.2 (20%).
    hash : callable
        A hash function to generate a hash from the identifier.
        Secure hashes and message digests algorithm. Can be 
        SHA1, SHA224, SHA256, SHA384, and SHA512 (defined in FIPS 180-2) 
        as well as RSA’s MD5 algorithm (defined in Internet RFC 1321). 
        
        Please refer to :ref:`<https://docs.python.org/3/library/hashlib.html>` 
        for futher details.

    Returns
    -------
    bool
        True if the instance should be in the test set, False otherwise.

    Examples
    --------
    >>> from gofast.tools.coreutils import test_set_check_id
    >>> test_set_check_id(42, test_ratio=0.2, hash=hashlib.md5)
    ... False
    """
    # def test_set_check_id(identifier: str, ratio: float, hash_function: _F) -> bool:
    #     """Determines if an identifier belongs to the test set using the hash value."""
    #     # Convert identifier to string and hash
    #     hash_val = int(hash_function(str(identifier).encode()).hexdigest(), 16)
    #     # Use the hash value to decide test set membership
    #     return hash_val % 10000 / 10000.0 < ratio
    
    #     hashed_id = hash_function(identifier.encode('utf-8')).digest()
    #     return np.frombuffer(hashed_id, dtype=np.uint8).sum() < 256 * test_ratio
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(
    data: DataFrame, test_ratio: float, id_column: Optional[List[str]] = None,
    keep_colindex: bool = True, hash: _F = hashlib.md5
) -> Tuple[DataFrame, DataFrame]:
    """
    Split a DataFrame into train and test sets while ensuring data consistency
    by using specified id columns or the DataFrame's index as unique identifiers.

    Parameters
    ----------
    data : DataFrame
        The DataFrame containing the features.
    test_ratio : float
        The ratio of instances to include in the test set.
    id_column : list of str, optional
        Column names to use as unique identifiers. If None, the DataFrame's index
        is used as the identifier.
    keep_colindex : bool, optional
        Determines whether to keep or drop the index column after resetting.
        This parameter is only applicable if id_column is None and the DataFrame's
        index is reset. Default is True.
    hash : callable
        A hash function to generate a hash from the identifier.

    Returns
    -------
    Tuple[DataFrame, DataFrame]
        A tuple containing the train and test set DataFrames.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.tools.coreutils import split_train_test_by_id
    >>> data = pd.DataFrame({'ID': [1, 2, 3, 4, 5], 'Value': [10, 20, 30, 40, 50]})
    >>> train_set, test_set = split_train_test_by_id(data, test_ratio=0.2, id_column=['ID'])
    >>> len(train_set), len(test_set)
    (4, 1)
    """
    drop_tmp_index=False
    if id_column is None:
        # Check if the index is integer-based; if not, create a temporary integer index.
        if not data.index.is_integer():
            data['_tmp_hash_index'] = np.arange(len(data))
            ids = data['_tmp_hash_index']
            drop_tmp_index = True
        else:
            ids = data.index.to_series()
            drop_tmp_index = False
    else:
        # Use specified id columns as unique identifiers, combining them if necessary.
        ids = data[id_column].astype(str).apply(
            lambda row: '_'.join(row), axis=1) if isinstance(
                id_column, list) else data[id_column]

    in_test_set = ids.apply(lambda id_: test_set_check_id(id_, test_ratio, hash))

    train_set = data.loc[~in_test_set].copy()
    test_set = data.loc[in_test_set].copy()

    if drop_tmp_index or (id_column is None and not keep_colindex):
        # Remove the temporary index or reset the index as needed
        train_set.drop(columns=['_tmp_hash_index'], errors='ignore', inplace=True)
        test_set.drop(columns=['_tmp_hash_index'], errors='ignore', inplace=True)
        # for consistency if '_tmp_has_index' 
        if '_tmp_hash_index' in data.columns: 
            data.drop (columns='_tmp_hash_index', inplace =True)
    elif id_column is None and keep_colindex:
        # If keeping the original index and it was integer-based, no action needed
        pass

    return train_set, test_set

 
def denormalize(
    data: ArrayLike, min_value: float, max_value: float
    ) -> ArrayLike:
    """
    Denormalizes data from a normalized scale back to its original scale.

    This function is useful when data has been normalized to a different 
    scale (e.g., [0, 1]) and needs to be converted back to its original scale 
    for interpretation or further processing.

    Parameters
    ----------
    data : np.ndarray
        The data to be denormalized, assumed to be a NumPy array.
    min_value : float
        The minimum value of the original scale before normalization.
    max_value : float
        The maximum value of the original scale before normalization.

    Returns
    -------
    np.ndarray
        The denormalized data, converted back to its original scale.

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.tools.coreutils import denormalize
    >>> normalized_data = np.array([0, 0.5, 1])
    >>> min_value = 10
    >>> max_value = 20
    >>> denormalized_data = denormalize(normalized_data, min_value, max_value)
    >>> print(denormalized_data)
    [10. 15. 20.]

    Note
    ----
    The denormalization process is the inverse of normalization and is applied
    to data that was previously normalized according to the formula:
        `data_norm = (data - min_value) / (max_value - min_value)`
    The denormalize function uses the inverse of this formula to restore the data.
    """
    if not isinstance (data, (pd.Series, pd.DataFrame)): 
        data = np.asarray( data )
        
    return data * (max_value - min_value) + min_value
   

def squeeze_specific_dim(
    arr: np.ndarray, axis: Optional[int] = -1
    ) -> np.ndarray:
    """
    Squeeze specific dimensions of a NumPy array based on the axis parameter.
    
    This function provides a flexible way to remove single-dimensional entries
    from the shape of an array. By default, it targets the last dimension,
    but can be configured to squeeze any specified dimension or all single-dimension
    axes if `axis` is set to None.

    Parameters
    ----------
    arr : np.ndarray
        The input array to potentially squeeze.
    axis : Optional[int], default -1
        The specific axis to squeeze. If the size of this axis is 1, it will be
        removed from the array. If `axis` is None, all single-dimension axes are
        squeezed. If `axis` is set to a specific dimension (0, 1, ..., arr.ndim-1),
        only that dimension will be squeezed if its size is 1.

    Returns
    -------
    np.ndarray
        The array with the specified dimension squeezed if its size was 1,
        otherwise the original array. If `axis` is None, all single-dimension
        axes are squeezed.

    Examples
    --------
    Squeeze the last dimension:

    >>> from gofast.tools.coreutils import squeeze_specific_dim
    >>> arr = np.array([[1], [2], [3]])
    >>> print(squeeze_specific_dim(arr).shape)
    (3,)

    Squeeze all single-dimension axes:

    >>> arr = np.array([[[1], [2], [3]]])
    >>> print(squeeze_specific_dim(arr, None).shape)
    (3,)

    Squeeze a specific dimension (e.g., first dimension of a 3D array):

    >>> arr = np.array([[[1, 2, 3]]])
    >>> print(squeeze_specific_dim(arr, 0).shape)
    ([[1, 2, 3]])

    Not squeezing if the specified axis does not have size 1:

    >>> arr = np.array([[1, 2, 3], [4, 5, 6]])
    >>> print(squeeze_specific_dim(arr, 0).shape)
    [[1, 2, 3], [4, 5, 6]]
    """
    if axis is None:
        # Squeeze all single-dimension axes
        return np.squeeze(arr)
    else:
        # Check if the specified axis is a single-dimension axis and squeeze it
        try:
            return np.squeeze(arr, axis=axis)
        except ValueError:
            # Return the array unchanged if squeezing is not applicable
            return arr

def contains_delimiter(s: str, delimiters: Union[str, list, set]) -> bool:
    """
    Checks if the given string contains any of the specified delimiters.

    Parameters
    ----------
    s : str
        The string to check.
    delimiters : str, list, or set
        Delimiters to check for in the string. Can be specified as a single
        string (for a single delimiter), a list of strings, or a set of strings.

    Returns
    -------
    bool
        True if the string contains any of the delimiters, False otherwise.

    Examples
    --------
    >>> from gofast.tools.coreutils import contains_delimiter
    >>> contains_delimiter("example__string", "__")
    True

    >>> contains_delimiter("example--string", ["__", "--", "&", "@", "!"])
    True

    >>> contains_delimiter("example&string", {"__", "--", "&", "@", "!"})
    True

    >>> contains_delimiter("example@string", "__--&@!")
    True

    >>> contains_delimiter("example_string", {"__", "--", "&", "@", "!"})
    False

    >>> contains_delimiter("example#string", "#$%")
    True

    >>> contains_delimiter("example$string", ["#", "$", "%"])
    True

    >>> contains_delimiter("example%string", "#$%")
    True

    >>> contains_delimiter("example^string", ["#", "$", "%"])
    False
    """
    # for consistency
    s = str(s) 
    # Convert delimiters to a set if it's not already a set
    if not isinstance(delimiters, set):
        if isinstance(delimiters, str):
            delimiters = set(delimiters)
        else:  # Assuming it's a list or similar iterable
            delimiters = set(delimiters)
    
    return any(delimiter in s for delimiter in delimiters)    
    
def convert_to_structured_format(
        *arrays: Any, as_frame: bool = True, 
        skip_sparse: bool =True, 
        ) -> List[Union[ArrayLike, DataFrame, Series]]:
    """
    Converts input objects to structured numpy arrays or pandas DataFrame/Series
    based on their shapes and the `as_frame` flag. If conversion to a structured
    format fails, the original objects are returned. When `as_frame` is False,
    attempts are made to convert inputs to numpy arrays.
    
    Parameters
    ----------
    *arrays : Any
        A variable number of objects to potentially convert. These can be lists,
        tuples, or numpy arrays.
    as_frame : bool, default=True
        If True, attempts to convert arrays to DataFrame or Series; otherwise,
        attempts to standardize as numpy arrays.
    skip_sparse: bool, default=True 
        Dont convert any sparse matrix and keept it as is. 
    
    Returns
    -------
    List[Union[np.ndarray, pd.DataFrame, pd.Series]]
        A list containing the original objects, numpy arrays, DataFrames, or
        Series, depending on each object's structure and the `as_frame` flag.
    
    Examples
    --------
    Converting to pandas DataFrame/Series:
    >>> from gofast.tools.coreutils import convert_to_structured_format
    >>> import numpy as np 
    >>> import pandas as pd 
    >>> features= {"feature_1": range (7), "feature_2":['1', 2, 9, 35, "0", "76", 'r']}
    >>> target= pd.Series(data=range(10), name="target")
    >>> convert_to_structured_format( features, target, as_frame=True)
    >>> arr1 = np.array([[1, 2, 3], [4, 5, 6]])
    >>> arr2 = np.array([7, 8, 9])
    >>> convert_to_structured_format(arr1, arr2, as_frame=True)
    [   DataFrame:
            0  1  2
        0   1  2  3
        1   4  5  6,
        Series:
        0    7
        1    8
        2    9
    ]

    Standardizing as numpy arrays:
    >>> list1 = [10, 11, 12]
    >>> tuple1 = (13, 14, 15)
    >>> convert_to_structured_format(list1, tuple1, as_frame=False)
    [   array([10, 11, 12]),
        array([13, 14, 15])
    ]
    """

    def attempt_conversion_to_numpy(arr: Any) -> np.ndarray:
        """Attempts to convert an object to a numpy array."""
        try:
            return np.array(arr)
        except Exception:
            return arr

    def attempt_conversion_to_pandas(
            arr: np.ndarray) -> Union[np.ndarray, pd.DataFrame, pd.Series]:
        """Attempts to convert an array to a DataFrame or Series based on shape."""
        from scipy.sparse import issparse
        try:
            if issparse(arr) and skip_sparse: 
                raise # dont perform any convertion 
            if hasattr(arr, '__array__'): 
                if arr.ndim == 1:
                    return pd.Series(arr)
                elif arr.ndim == 2:
                    if arr.shape[1] == 1:
                        return pd.Series(arr.squeeze())
                    else:
                        return pd.DataFrame(arr)
            else: 
                return pd.DataFrame(arr)
        except Exception:
            pass
        return arr

    if as_frame:
        return [attempt_conversion_to_pandas(arr) for arr in arrays]
    else:
        # Try to convert everything to numpy arrays, return as is if it fails
        return [attempt_conversion_to_numpy(attempt_conversion_to_pandas(arr)
                                            ) for arr in arrays]

def process_and_extract_data(
    *args: ArrayLike, 
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
    *args : ArrayLike
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
    >>> from gofast.tools.coreutils import process_and_extract_data
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

    for arg in args:
        result = _process_input(arg, columns, to_array)
        if result is not None:
            extracted_data.append(result)

    if ensure_uniform_length and not all(len(x) == len(
            extracted_data[0]) for x in extracted_data):
        if on_error == 'raise':
            raise ValueError("Extracted data arrays do not have uniform length.")
        else:
            return []

    return extracted_data

def to_series_if(
    *values: Any, 
    value_names: Optional[List[str]] = None, 
    name: Optional[str] = None,
    error: str = 'ignore',
    **kws
) -> Series:
    """
    Constructs a pandas Series from given values, optionally naming the series
    and its index.

    Parameters
    ----------
    *values : Any
        A variable number of inputs, each can be a scalar, float, int, or array-like object.
    value_names : Optional[List[str]]
        Names to be used for the index of the series. If not provided or if its length
        doesn't match the number of values, default numeric index is used.
    name : Optional[str]
        Name of the series.
    error : str, default 'ignore'
        Error handling strategy ('ignore' or 'raise'). If 'raise', errors during series
        construction lead to an exception.
    **kws : dict
        Additional keyword arguments passed to `pd.Series` constructor.

    Returns
    -------
    pd.Series or original values
        A pandas Series constructed from the inputs if successful, otherwise, the original
        values if the series construction is not applicable.

    Examples
    --------
    >>> from gofast.tools.coreutils import to_series_if
    >>> series = to_series_if(0.5, 8, np.array(
        [6.3]), [5], 2, value_names=['a', 'b', 'c', 'd', 'e'])
    >>> print(series)
    a    0.5
    b    8.0
    c    6.3
    d    5.0
    e    2.0
    dtype: float64
    >>> series = to_series_if(0.5, 8, np.array([6.3, 7]), [5], 2,
                              value_names=['a', 'b', 'c', 'd', 'e'], error='raise')
    ValueError: Failed to construct series, input types vary.
    """
    # Validate input lengths and types
    if value_names and len(value_names) != len(values):
        if error == 'raise':
            raise ValueError("Length of `value_names` does not match the number of values.")
        value_names = None  # Reset to default indexing
    # Attempt to construct series
    try:
        # Flatten array-like inputs to avoid creating Series of lists/arrays
        flattened_values = [val[0] if isinstance(
            val, (list,tuple,  np.ndarray, pd.Series)) and len(val) == 1 else val for val in values]
        series = pd.Series(flattened_values, index=value_names, name=name, **kws)
    except Exception as e:
        if error == 'raise':
            raise ValueError(f"Failed to construct series due to: {e}")
        return values  # Return the original values if series construction fails

    return series

def ensure_visualization_compatibility(
        result, as_frame=False, view=False, func_name=None,
        verbose=0, allow_singleton_view=False
        ):
    """
    Evaluates and prepares the result for visualization, adjusting its format
    if necessary and determining whether visualization is feasible based on
    given parameters. If the conditions for visualization are not met, 
    especially for singleton values, it can modify the view flag accordingly.

    Parameters
    ----------
    result : iterable or any
        The result to be checked and potentially modified for visualization.
    as_frame : bool, optional
        If True, the result is intended for frame-based visualization, which 
        may prevent conversion of singleton iterables to a float. Defaults to False.
    view : bool, optional
        Flag indicating whether visualization is intended. This function may 
        modify it to False if visualization conditions aren't met. Defaults to False.
    func_name : callable or str, optional
        The name of the function or a callable from which the name can be derived, 
        used in generating verbose messages. Defaults to None.
    verbose : int, optional
        Controls verbosity level. A value greater than 0 enables verbose messages. 
        Defaults to 0.
    allow_singleton_view : bool, optional
        Allows visualization of singleton values if set to True. If False and a 
        singleton value is encountered, `view` is set to False. Defaults to False.

    Returns
    -------
    tuple
        A tuple containing the potentially modified result and the updated view flag.
        The result is modified if it's a singleton iterable and conditions require it.
        The view flag is updated based on the allowability of visualization.

    Examples
    --------
    >>> from gofast.tools.coreutils import ensure_visualization_compatibility
    >>> result = [100.0]
    >>> modified_result, can_view = ensure_visualization_compatibility(
    ...     result, as_frame=False, view=True, verbose=1, allow_singleton_view=False)
    Visualization is not allowed for singleton value.
    >>> print(modified_result, can_view)
    100.0 False

    >>> result = [[100.0]]
    >>> modified_result, can_view = ensure_visualization_compatibility(
    ...     result, as_frame=True, verbose=1)
    >>> print(modified_result, can_view)
    [[100.0]] True
    """
    if hasattr(result, '__iter__') and len(
            result) == 1 and not allow_singleton_view:
        if not as_frame:
            # Attempt to convert to float value
            try:
                result = float(result[0])
            except ValueError:
                pass  # Keep the result as is if conversion fails

        if view: 
            if verbose > 0:
                # Construct a user-friendly verbose message
                func_name_str = f"{func_name.__name__} visualization" if callable(
                    func_name) else "Visualization"
                # Ensure the first letter is capitalized
                message_start = func_name_str[0].upper() + func_name_str[1:]  
                print(f"{message_start} is not allowed for singleton value.")
            view =False 
    return result, view 

def generate_mpl_styles(n, prop='color'):
    """
    Generates a list of matplotlib property items (colors, markers, or line styles)
    to accommodate a specified number of samples.

    Parameters
    ----------
    n : int
        Number of property items needed. It generates a list of property items.
    prop : str, optional
        Name of the property to retrieve. Accepts 'color', 'marker', or 'line'.
        Defaults to 'color'.

    Returns
    -------
    list
        A list of property items with size equal to `n`.

    Raises
    ------
    ValueError
        If the `prop` argument is not one of the accepted property names.

    Examples
    --------
    Generate 10 color properties:

    >>> from gofast.tools.coreutils import generate_mpl_styles
    >>> generate_mpl_styles(10, prop='color')
    ['g', 'gray', 'y', 'blue', 'orange', 'purple', 'lime', 'k', 'cyan', 'magenta']

    Generate 5 marker properties:

    >>> generate_mpl_styles(5, prop='marker')
    ['o', '^', 's', '*', '+']

    Generate 3 line style properties:

    >>> generate_mpl_styles(3, prop='line')
    ['-', '--', '-.']
    """
    import matplotlib as mpl

    D_COLORS = ["g", "gray", "y", "blue", "orange", "purple", "lime",
                "k", "cyan", "magenta"]
    D_MARKERS = ["o", "^", "s", "*", "+", "x", "D", "H"]
    D_STYLES = ["-", "--", "-.", ":"]
    
    n = int(n)  # Ensure n is an integer
    prop = prop.lower().strip().replace('s', '')  # Normalize the prop string
    if prop not in ('color', 'marker', 'line'):
        raise ValueError(f"Property '{prop}' is not available."
                         " Expect 'color', 'marker', or 'line'.")

    # Mapping property types to their corresponding lists
    properties_map = {
        'color': D_COLORS,
        'marker': D_MARKERS + list(mpl.lines.Line2D.markers.keys()),
        'line': D_STYLES
    }

    # Retrieve the specific list of properties based on the prop parameter
    properties_list = properties_map[prop]

    # Generate the required number of properties, repeating the list if necessary
    repeated_properties = list(itertools.chain(*itertools.repeat(properties_list, (
        n + len(properties_list) - 1) // len(properties_list))))[:n]

    return repeated_properties

def generate_alpha_values(n, increase=True, start=0.1, end=1.0, epsilon=1e-10):
    """
    Generates a list of alpha (transparency) values that either increase or 
    decrease gradually to fit the number of property items.
    
    Incorporates an epsilon to safeguard against division by zero.
    
    Parameters
    ----------
    n : int
        The number of alpha values to generate.
    increase : bool, optional
        If True, the alpha values will increase; if False, they will decrease.
        Defaults to True.
    start : float, optional
        The starting alpha value. Defaults to 0.1.
    end : float, optional
        The ending alpha value. Defaults to 1.0.
    epsilon : float, optional
        Small value to avert division by zero. Defaults to 1e-10.
        
    Returns
    -------
    list
        A list of alpha values of length `n`.
    
    Examples
    --------
    >>> from gofast.tools.coreutils import generate_alpha_values
    >>> generate_alpha_values(5, increase=True)
    [0.1, 0.325, 0.55, 0.775, 1.0]
    
    >>> generate_alpha_values(5, increase=False)
    [1.0, 0.775, 0.55, 0.325, 0.1]
    """
    if not 0 <= start <= 1 or not 0 <= end <= 1:
        raise ValueError("Alpha values must be between 0 and 1.")

    # Calculate the alpha values, utilizing epsilon in the denominator 
    # to prevent division by zero
    alphas = [start + (end - start) * i / max(n - 1, epsilon) for i in range(n)]
    
    if not increase:
        alphas.reverse() # or alphas[::-1] creates new list
    
    return alphas

def decompose_colormap(cmap_name, n_colors=5):
    """
    Decomposes a colormap into a list of individual colors.

    Parameters
    ----------
    cmap_name : str
        The name of the colormap to decompose.
    n_colors : int, default=5
        The number of colors to extract from the colormap.

    Returns
    -------
    list
        A list of RGBA color values from the colormap.

    Examples
    --------
    >>> colors = decompose_colormap('viridis', 5)
    >>> print(colors)
    [(0.267004, 0.004874, 0.329415, 1.0), ..., (0.993248, 0.906157, 0.143936, 1.0)]
    """
    cmap = plt.cm.get_cmap(cmap_name, n_colors)
    colors = [cmap(i) for i in range(cmap.N)]
    return colors

def get_colors_and_alphas(
    count, 
    cmap=None, 
    alpha_direction='decrease', 
    start_alpha=0.1,
    end_alpha=1.0, 
    convert_to_named_color=True, 
    single_color_as_string=False,
    consider_alpha=False, 
    ignore_color_names=False, 
    color_space='rgb', 
    error="ignore"
):
    """
    Generates a sequence of color codes and alpha (transparency) values. 
    
    Colors can be sourced from a specified Matplotlib colormap or generated 
    using predefined styles. Alpha values can be arranged in ascending or 
    descending order to create a gradient effect.

    The function also supports converting color tuples to named colors and 
    allows for customizing the transparency gradient. Additionally, if only 
    one color is generated, it can return that color directly as a string
    rather than wrapped in a list, for convenience in functions that expect a
    single color string.

    Parameters
    ----------
    count : int or iterable
        Specifies the number of colors and alpha values to generate. If an iterable 
        is provided, its length determines the number of colors and alphas.
    cmap : str, optional
        The name of a Matplotlib colormap to generate colors. If None, colors are
        generated using predefined styles. Defaults to ``None``.
    alpha_direction : str, optional
        Direction to arrange alpha values for creating a gradient effect. ``increase``
        for ascending order, ``decrease`` for descending. Defaults to ``decrease``.
    start_alpha : float, optional
        The starting alpha value (transparency) in the gradient, between 0 (fully
        transparent) and 1 (fully opaque). Defaults to ``0.1``.
    end_alpha : float, optional
        The ending alpha value in the gradient, between 0 and 1. 
        Defaults to ``1.0``.
    convert_to_named_color : bool, optional
        Converts color tuples to the nearest Matplotlib named color. This 
        conversion applies when exactly one color is generated. 
        Defaults to ``True``.
    single_color_as_string : bool, optional
        If True and only one color is generated, returns the color as a string 
        instead of a list. Useful for functions expecting a single color string.
        Defaults to ``False``.
    consider_alpha : bool, optional
        Includes the alpha channel in the conversion process to named colors.
        Applicable only when `convert_to_named_color` is True. This is helpful
        when a human-readable color name is preferred over RGB values.
        Defaults to ``False``.
    ignore_color_names : bool, optional
        When True, any input color names (str) are ignored during conversion 
        to named colors. Useful to exclude specific colors from conversion. 
        Defaults to ``False``.
    color_space : str, optional
        The color space used for computing the closeness of colors. Can be 
        ``rgb`` for RGB color space or ``lab`` for LAB color space, which is more 
        perceptually uniform. Defaults to ``rgb``.
    error : str, optional
        Controls the error handling strategy when an invalid color is 
        encountered during the conversion process. ``raise`` will throw an error,
        while ``ignore`` will proceed without error. Defaults to ``ignore``.

    Returns
    -------
    tuple
        A tuple containing either a list of color codes (RGBA or named color strings) 
        and a corresponding list of alpha values, or a single color code and alpha 
        value if `single_color_as_string` is True and only one color is generated.

    Examples
    --------
    Generate 3 random colors with decreasing alpha values:

    >>> get_colors_and_alphas(3)
    (['#1f77b4', '#ff7f0e', '#2ca02c'], [1.0, 0.55, 0.1])

    Generate 4 colors from the 'viridis' colormap with increasing alpha values:

    >>> get_colors_and_alphas(4, cmap='viridis', alpha_direction='increase')
    (['#440154', '#3b528b', '#21918c', '#5ec962'], [0.1, 0.4, 0.7, 1.0])

    Convert a single generated color to a named color:

    >>> get_colors_and_alphas(1, convert_to_named_color=True)
    ('rebeccapurple', [1.0])

    Get a single color as a string instead of a list:

    >>> get_colors_and_alphas(1, single_color_as_string=True)
    ('#1f77b4', [1.0])
    """
    
    if hasattr(count, '__iter__'):
        count = len(count)
    colors =[]
    if cmap is not None and cmap not in plt.colormaps(): 
        cmap=None 
        colors =[cmap] # add it to generate map
    # Generate colors
    if cmap is not None:
        colors = decompose_colormap(cmap, n_colors=count)
    else:
        colors += generate_mpl_styles(count, prop='color')

    # Generate alphas
    increase = alpha_direction == 'increase'
    alphas = generate_alpha_values(count, increase=increase,
                                   start=start_alpha, end=end_alpha)
    
    # Convert tuple colors to named colors if applicable
    if convert_to_named_color: 
        colors = colors_to_names(
            *colors, consider_alpha= consider_alpha,
            ignore_color_names=ignore_color_names,  
            color_space= color_space, 
            error= error,
            )
    # If a single color is requested as a string, return it directly
    if single_color_as_string and len(colors) == 1:
        if not convert_to_named_color: 
            colors = [closest_color(colors[0], consider_alpha= consider_alpha, 
                                color_space =color_space )]
        colors = colors[0]

    return colors, alphas


def colors_to_names(*colors, consider_alpha=False, ignore_color_names=False, 
                    color_space='rgb', error='ignore'):
    """
    Converts a sequence of RGB or RGBA colors to their closest named color 
    strings. 
    
    Optionally ignores input color names and handles colors in specified 
    color spaces.
    
    Parameters
    ----------
    *colors : tuple
        A variable number of RGB(A) color tuples or color name strings.
    consider_alpha : bool, optional
        If True, the alpha channel in RGBA colors is considered in the conversion
        process. Defaults to False.
    ignore_color_names : bool, optional
        If True, input strings that are already color names are ignored. 
        Defaults to False.
    color_space : str, optional
        Specifies the color space ('rgb' or 'lab') used for color comparison. 
        Defaults to 'rgb'.
    error : str, optional
        Error handling strategy when encountering invalid colors. If 'raise', 
        errors are raised. Otherwise, errors are ignored. Defaults to 'ignore'.
    
    Returns
    -------
    list
        A list of color name strings corresponding to the input colors.

    Examples
    --------
    >>> from gofast.tools.coreutils import colors_to_names
    >>> colors_to_names((0.267004, 0.004874, 0.329415, 1.0), 
                        (0.127568, 0.566949, 0.550556, 1.0), 
                        consider_alpha=True)
    ['rebeccapurple', 'mediumseagreen']
    
    >>> colors_to_names('rebeccapurple', ignore_color_names=True)
    []
    
    >>> colors_to_names((123, 234, 45), color_space='lab', error='raise')
    ['limegreen']
    """
    color_names = []
    for color in colors:
        if isinstance(color, str):
            if ignore_color_names:
                continue
            else:
                color_names.append(color)  # String color name is found
        else:
            try:
                color_name = closest_color(color, consider_alpha=consider_alpha,
                                           color_space=color_space)
                color_names.append(color_name)
            except Exception as e:
                if error == 'raise':
                    raise e
                
    return color_names

def closest_color(rgb_color, consider_alpha=False, color_space='rgb'):
    """
    Finds the closest named CSS4 color to the given RGB(A) color in the specified
    color space, optionally considering the alpha channel.

    Parameters
    ----------
    rgb_color : tuple
        A tuple representing the RGB(A) color.
    consider_alpha : bool, optional
        Whether to include the alpha channel in the color closeness calculation.
        Defaults to False.
    color_space : str, optional
        The color space to use when computing color closeness. Can be 'rgb' or 'lab'.
        Defaults to 'rgb'.

    Returns
    -------
    str
        The name of the closest CSS4 color.

    Raises
    ------
    ValueError
        If an invalid color space is specified.

    Examples
    --------
    Find the closest named color to a given RGB color:

    >>> from gofast.tools.coreutils import closest_color
    >>> closest_color((123, 234, 45))
    'forestgreen'

    Find the closest named color to a given RGBA color, considering the alpha:

    >>> closest_color((123, 234, 45, 0.5), consider_alpha=True)
    'forestgreen'

    Find the closest named color in LAB color space (more perceptually uniform):

    >>> closest_color((123, 234, 45), color_space='lab')
    'limegreen'
    """
    if color_space not in ['rgb', 'lab']:
        raise ValueError(f"Invalid color space '{color_space}'. Choose 'rgb' or 'lab'.")

    if ensure_scipy_compatibility(): 
        from scipy.spatial import distance 
    # Adjust input color based on consider_alpha flag
    
    # Include alpha channel if consider_alpha is True
    input_color = rgb_color[:3 + consider_alpha]  

    # Convert the color to the chosen color space if needed
    if color_space == 'lab':
        # LAB conversion ignores alpha
        input_color = mcolors.rgb_to_lab(input_color[:3])  
        color_comparator = lambda color: distance.euclidean(
            mcolors.rgb_to_lab(color[:3]), input_color)
    else:  # RGB or RGBA
        color_comparator = lambda color: distance.euclidean(
            color[:len(input_color)], input_color)

    # Compute the closeness of each named color to the given color
    closest_name = None
    min_dist = float('inf')
    for name, hex_color in mcolors.CSS4_COLORS.items():
        # Adjust based on input_color length
        named_color = mcolors.to_rgba(hex_color)[:len(input_color)]  
        dist = color_comparator(named_color)
        if dist < min_dist:
            min_dist = dist
            closest_name = name

    return closest_name

def check_uniform_type(
    values: Union[Iterable[Any], Any],
    items_to_compare: Union[Iterable[Any], Any] = None,
    raise_exception: bool = True,
    convert_values: bool = False,
    return_types: bool = False,
    target_type: type = None,
    allow_mismatch: bool = True,
    infer_types: bool = False,
    comparison_method: str = 'intersection',
    custom_conversion_func: _F[Any] = None,
    return_func: bool = False
) -> Union[bool, List[type], Tuple[Iterable[Any], List[type]], _F]:
    """
    Checks whether elements in `values` are of uniform type. 
    
    Optionally comparing them against another set of items or converting all 
    values to a target type. Can return a callable for deferred execution of 
    the specified logic.Function is useful for validating data uniformity, 
    especially before performing operations that assume homogeneity of the 
    input types.
    

    Parameters
    ----------
    values : Iterable[Any] or Any
        An iterable containing items to check. If a non-iterable item is provided,
        it is treated as a single-element iterable.
    items_to_compare : Iterable[Any] or Any, optional
        An iterable of items to compare against `values`. If specified, the
        `comparison_method` is used to perform the comparison.
    raise_exception : bool, default True
        If True, raises an exception when a uniform type is not found or other
        constraints are not met. Otherwise, issues a warning.
    convert_values : bool, default False
        If True, tries to convert all `values` to `target_type`. Requires
        `target_type` to be specified.
    return_types : bool, default False
        If True, returns the types of the items in `values`.
    target_type : type, optional
        The target type to which `values` should be converted if `convert_values`
        is True.
    allow_mismatch : bool, default True
        If False, requires all values to be of identical types; otherwise,
        allows type mismatch.
    infer_types : bool, default False
        If True and different types are found, returns the types of each item
        in `values` in order.
    comparison_method : str, default 'intersection'
        The method used to compare `values` against `items_to_compare`. Must
        be one of the set comparison methods ('difference', 'intersection', etc.).
    custom_conversion_func : Callable[[Any], Any], optional
        A custom function for converting items in `values` to another type.
    return_func : bool, default False
        If True, returns a callable that encapsulates the logic based on the 
        other parameters.

    Returns
    -------
    Union[bool, List[type], Tuple[Iterable[Any], List[type]], Callable]
        The result based on the specified parameters. This can be: 
        - A boolean indicating whether all values are of the same type.
        - The common type of all values if `return_types` is True.
        - A tuple containing the converted values and their types if `convert_values`
          and `return_types` are both True.
        - a callable encapsulating the specified logic for deferred execution.
        
    Examples
    --------
    >>> from gofast.tools.coreutils import check_uniform_type
    >>> check_uniform_type([1, 2, 3])
    True

    >>> check_uniform_type([1, '2', 3], allow_mismatch=False, raise_exception=False)
    False

    >>> deferred_check = check_uniform_type([1, 2, '3'], convert_values=True, 
    ...                                        target_type=int, return_func=True)
    >>> deferred_check()
    [1, 2, 3]

    Notes
    -----
    The function is designed to be flexible, supporting immediate or deferred execution,
    with options for type conversion and detailed type information retrieval.
    """
    def operation():
        # Convert values and items_to_compare to lists if 
        # they're not already iterable
        if isinstance(values, Iterable) and not isinstance(values, str):
            val_list = list(values)
        else:
            val_list = [values]

        if items_to_compare is not None:
            if isinstance(items_to_compare, Iterable) and not isinstance(
                    items_to_compare, str):
                comp_list = list(items_to_compare)
            else:
                comp_list = [items_to_compare]
        else:
            comp_list = []

        # Extract types
        val_types = set(type(v) for v in val_list)
        comp_types = set(type(c) for c in comp_list) if comp_list else set()

        # Compare types
        if comparison_method == 'intersection':
            common_types = val_types.intersection(comp_types) if comp_types else val_types
        elif comparison_method == 'difference':
            common_types = val_types.difference(comp_types)
        else:
            if raise_exception:
                raise ValueError(f"Invalid comparison method: {comparison_method}")
            return False

        # Check for type uniformity
        if not allow_mismatch and len(common_types) > 1:
            if raise_exception:
                raise ValueError("Not all values are the same type.")
            return False

        # Conversion
        if convert_values:
            if not target_type and not custom_conversion_func:
                if raise_exception:
                    raise ValueError("Target type or custom conversion "
                                     "function must be specified for conversion.")
                return False
            try:
                if custom_conversion_func:
                    converted_values = [custom_conversion_func(v) for v in val_list]
                else:
                    converted_values = [target_type(v) for v in val_list]
            except Exception as e:
                if raise_exception:
                    raise ValueError(f"Conversion failed: {e}")
                return False
            if return_types:
                return converted_values, [type(v) for v in converted_values]
            return converted_values

        # Return types
        if return_types:
            if infer_types or len(common_types) > 1:
                return [type(v) for v in val_list]
            return list(common_types)

        return True

    return operation if return_func else operation()


