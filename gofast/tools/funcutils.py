# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 21:21:15 2024

@author: Daniel
"""

from typing import Dict, Any, Callable, List,  Optional, Tuple , Union 
import time
import functools
import logging
import subprocess
import sys  # Import sys to use sys.executable
import pkg_resources
import numpy as np
import pandas as pd

def merge_dicts(*dicts: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Merges multiple dictionaries into a single dictionary, with later 
    dictionary values overriding those from earlier dictionaries.

    Parameters
    ----------
    *dicts : Dict[Any, Any]
        An arbitrary number of dictionary arguments to be merged.

    Returns
    -------
    Dict[Any, Any]
        A single dictionary resulting from the merging of input dictionaries.

    Examples
    --------
    >>> from gofast.tools.funcutils import merge_dicts
    >>> dict_a = {'a': 1, 'b': 2}
    >>> dict_b = {'b': 3, 'c': 4}
    >>> merged_dict = merge_dicts(dict_a, dict_b)
    >>> print(merged_dict)
    {'a': 1, 'b': 3, 'c': 4}
    """
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result

def retry_operation(func: Callable, retries: int = 3, delay: float = 1.0, 
                    catch_exceptions: tuple = (Exception,), 
                    backoff_factor: float = 1.0, 
                    on_retry: Optional[Callable[[int, Exception], None]] = None):
    """
    Retries a specified function upon failure, for a defined number of retries
    and with a delay between attempts. Allows specifying which exceptions to catch,
    a backoff strategy for delays, and an optional callback after each retry.

    Parameters
    ----------
    function : Callable
        The function to be executed and possibly retried upon failure.
    retries : int, optional
        The number of retries to attempt upon failure. Default is 3.
    delay : float, optional
        The initial delay between retries in seconds. Default is 1.0.
    catch_exceptions : tuple, optional
        A tuple of exception classes to catch and retry. Default is (Exception,).
    backoff_factor : float, optional
        The factor by which the delay increases after each retry. Default is 1.0.
    on_retry : Optional[Callable[[int, Exception], None]], optional
        An optional callback function that is called after a failed attempt.
        It receives the current attempt number and the exception raised.

    Returns
    -------
    Any
        The return value of the function if successful.

    Raises
    ------
    Exception
        The exception from the last attempt if all retries fail.

    Examples
    --------
    >>> from gofast.tools.funcutils import retry_operation
    >>> def test_func():
    ...     print("Trying...")
    ...     raise ValueError("Fail")
    >>> def on_retry_callback(attempt, exception):
    ...     print(f"Attempt {attempt}: {exception}")
    >>> try:
    ...     retry_operation(test_func, retries=2, delay=0.5, 
    ...                     on_retry=on_retry_callback)
    ... except ValueError as e:
    ...     print(e)
    """
    current_delay = delay
    for attempt in range(1, retries + 1):
        try:
            return func()
        except catch_exceptions as e:
            if attempt < retries:
                if on_retry:
                    on_retry(attempt, e)
                time.sleep(current_delay)
                current_delay *= backoff_factor
            else:
                raise e

def flatten_list(nested_list: List[Any], depth: int = -1) -> List[Any]:
    """
    Flattens a nested list into a single list of values, with an optional 
    depth parameter.

    Parameters
    ----------
    nested_list : List[Any]
        The list to flatten, which may contain nested lists of any depth.
    depth : int, optional
        The maximum depth of list nesting to flatten. A depth of -1 (default)
        means fully flatten, a depth of 0 means no flattening, a depth of 1
        means flatten one level of nesting, and so on.

    Returns
    -------
    List[Any]
        A single, flat list containing all values from the nested list to the
        specified depth.

    Examples
    --------
    >>> from gofast.tools.funcutils import flatten_list
    >>> nested = [1, [2, 3], [4, [5, 6]], 7]
    >>> flat = flatten_list(nested, depth=1)
    >>> print(flat)
    [1, 2, 3, 4, [5, 6], 7]
    """
    if depth == 0:
        return nested_list
    result = []
    for item in nested_list:
        if isinstance(item, list) and depth != 1:
            result.extend(flatten_list(item, depth-1))
        else:
            result.append(item)
    return result


def timeit_decorator(
        logger: Optional[logging.Logger] = None, level: int = logging.INFO):
    """
    A decorator that measures the execution time of a function and optionally logs it.

    Parameters
    ----------
    logger : Optional[logging.Logger], optional
        A logger object to log the execution time message. If None (default),
        the message is printed using the print function.
    level : int, optional
        The logging level at which to log the execution time message. Default
        is logging.INFO.

    Returns
    -------
    Callable
        A wrapped version of the input function that, when called, logs or prints
        the execution time.

    Examples
    --------
    >>> from gofast.tools.funcutils import timeit_decorator
    >>> import logging
    >>> logger = logging.getLogger('MyLogger')
    >>> @timeit_decorator(logger=logger)
    ... def example_function(delay: float):
    ...     '''Example function that sleeps for a given delay.'''
    ...     time.sleep(delay)
    >>> example_function(1)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            message = f"'{func.__name__}' executed in {end_time - start_time:.2f}s"
            if logger:
                logger.log(level, message)
            else:
                print(message)
            return result
        return wrapper
    return decorator

def conditional_decorator(
    predicate: Callable[[Callable], bool], 
    decorator: Callable[[Callable], Callable], 
    else_decorator: Optional[Callable[[Callable], Callable]] = None
    ) -> Callable:
    """
    Applies a decorator to a function conditionally based on a predicate. 
    
    Optionally, applies an alternative decorator if the condition is not met.

    Parameters
    ----------
    predicate : Callable[[Callable], bool]
        A function that takes a function as input and returns True if the primary
        decorator should be applied, False otherwise.
    decorator : Callable[[Callable], Callable]
        The primary decorator to apply if the predicate returns True.
    else_decorator : Optional[Callable[[Callable], Callable]], optional
        An alternative decorator to apply if the predicate returns False. If None,
        the function is returned unchanged in case of False. Default is None.

    Returns
    -------
    Callable
        A new decorator that conditionally applies the given decorators based on
        the predicate.

    Examples
    --------
    >>> from gofast.tools.funcutils import conditional_decorator
    >>> def my_decorator(func):
    ...     def wrapper(*args, **kwargs):
    ...         print("Decorated")
    ...         return func(*args, **kwargs)
    ...     return wrapper
    >>> def is_even(func):
    ...     return func(2) % 2 == 0
    >>> @conditional_decorator(predicate=is_even, decorator=my_decorator)
    ... def add_one(x):
    ...     return x + 1
    >>> add_one(2)
    'Decorated'
    3
    """
    def new_decorator(func: Callable) -> Callable:
        if predicate(func):
            return decorator(func)
        elif else_decorator:
            return else_decorator(func)
        return func
    return new_decorator

def batch_processor(
    func: Callable, 
    on_error: Optional[Callable[[Exception, Any], Any]] = None,
    on_success: Optional[Callable[[Any, Any], None]] = None) -> Callable:
    """
    Process a batch of inputs, with optional error handling and success 
    callbacks.

    Parameters
    ----------
    func : Callable
        The original function to be applied to each element of the input batch.
    on_error : Optional[Callable[[Exception, Any], Any]], optional
        A callback function to be called if an exception occurs while processing
        an element. It should take two parameters: the exception and the input
        that caused it, and return a value to be included in the results list.
        If None (default), exceptions are raised immediately, stopping the batch.
    on_success : Optional[Callable[[Any, Any], None]], optional
        A callback function to be called after successfully processing an element.
        It should take two parameters: the result of the function call and the
        input that led to it. This can be used for logging or further processing.

    Returns
    -------
    Callable
        A function that takes a list of inputs and returns a list of results,
        applying `func` to each input element, handling exceptions and successes
        as specified.

    Examples
    --------
    >>> from gofast.tools.funcutils import batch_processor
    >>> def safe_divide(x):
    ...     return 10 / x
    >>> def handle_error(e, x):
    ...     print(f"Error {e} occurred with input {x}")
    ...     return None  # Default value in case of error
    >>> batch_divide = batch_processor(safe_divide, on_error=handle_error)
    >>> print(batch_divide([2, 1, 0]))  # 0 will cause a division by zero error
    Error division by zero occurred with input 0
    [5.0, 10.0, None]
    """
    def process_batch(inputs: List[Any]) -> List[Any]:
        results = []
        for input in inputs:
            try:
                result = func(input)
                if on_success:
                    on_success(result, input)
                results.append(result)
            except Exception as e:
                if on_error:
                    error_result = on_error(e, input)
                    results.append(error_result)
                else:
                    raise e
        return results
    return process_batch

def is_valid_if(*expected_types: Tuple[type]):
    """
    A decorator to verify the datatype of positional parameters of a function.
    
    Raises a TypeError if the actual parameter types do not match the 
    expected types.

    Parameters
    ----------
    *expected_types : Tuple[type]
        The expected types of the positional parameters.

    Returns
    -------
    Callable
        The decorated function with type verification.

    Raises
    ------
    TypeError
        If the types of the actual parameters do not match the expected types.

    Examples
    --------
    >>> from gofast.tools.funcutils import is_valid_if
    >>> @is_valid_if(int, float)
    ... def add(a, b):
    ...     return a + b
    >>> add(1, 2.5)
    3.5
    >>> add(1, "2.5")  # This will raise a TypeError
    TypeError: Argument 2 of 'add' requires <class 'float'> but got <class 'str'>.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for i, (arg, expected_type) in enumerate(zip(args, expected_types), 1):
                if not isinstance(arg, expected_type):
                    raise TypeError(f"Argument {i} of '{func.__name__}' "
                                    f"requires {expected_type} but got {type(arg)}.")
            return func(*args, **kwargs)
        return wrapper
    return decorator

def ensure_pkg(package_name: str, auto_install: bool = False):
    """
    A decorator to ensure a specific package is installed before executing 
    the function.
    
    Optionally installs the package automatically if not found.

    Parameters
    ----------
    package_name : str
        The name of the package to check or install.
    auto_install : bool, optional
        If True, the package will be automatically installed if not found. 
        Default is False.

    Returns
    -------
    Callable
        The decorated function.

    Examples
    --------
    >>> from gofast.tools.funcutils import ensure_pkg
    >>> @ensure_pkg("numpy", auto_install=True)
    ... def use_numpy():
    ...     import numpy as np
    ...     return np.array([1, 2, 3])
    
    Note: Using auto_install=True might require administrative privileges.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Attempt to load the package to see if it's already installed
                pkg_resources.get_distribution(package_name)
            except pkg_resources.DistributionNotFound:
                if auto_install:
                    print(f"Package '{package_name}' not found. Installing...")
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", package_name,
                        '--progress-bar', 'ascii'])
                    print(f"Package '{package_name}' installed successfully.")
                else:
                    raise ModuleNotFoundError(
                        f"Package '{package_name}' is required but not installed."
                        " Set auto_install=True to install automatically.")
            return func(*args, **kwargs)
        return wrapper
    return decorator

def drop_nan_if(thresh: float, meth: str = 'drop_cols'):
    """
    A decorator that preprocesses the first positional argument 'data' of the
    decorated function to ensure it is a DataFrame and then drops rows or columns 
    based on a missing value threshold. If the data is an array or Series, it is
    converted to a DataFrame. After dropping, the data is cast back to its original
    data types.

    Parameters
    ----------
    thresh : float
        The threshold for missing values. Rows or columns with missing values
        exceeding this threshold will be dropped.
    meth : str, optional
        The method to apply: 'drop_rows' for dropping rows, 'drop_cols' for
        dropping columns. Default is 'drop_cols'.

    Returns
    -------
    Callable
        The decorated function with data preprocessing.

    Examples
    --------
    >>> @drop_nan_if(thresh=1, meth='drop_rows')
    ... def process_data(data):
    ...     print(data.dtypes)
    >>> data = pd.DataFrame({
    ...     'A': [1, np.nan, 3],
    ...     'B': ['x', 'y', np.nan],
    ...     'C': pd.to_datetime(['2021-01-01', np.nan, '2021-01-03'])
    ... })
    >>> process_data(data)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            data = args[0]
            original_dtypes = None
            if isinstance(data, np.ndarray):
                data = pd.DataFrame(data)
            elif isinstance(data, pd.Series):
                data = data.to_frame()
            elif not isinstance(data, pd.DataFrame):
                raise TypeError("The first positional argument must be a "
                                "pandas DataFrame, a numpy array, or a pandas"
                                " Series.")
            else:
                # Store original data types for later restoration
                original_dtypes = data.dtypes
            
            if meth == 'drop_rows':
                processed_data = data.dropna(axis=0, thresh=len(data.columns) * thresh)
            elif meth == 'drop_cols':
                processed_data = data.dropna(axis=1, thresh=len(data) * thresh)
            else:
                raise ValueError("Method argument 'meth' must be either 'drop_rows' or 'drop_cols'.")

            # Restore original data types
            if original_dtypes is not None:
                for col, dtype in original_dtypes.items():
                    if col in processed_data.columns:
                        processed_data[col] = processed_data[col].astype(dtype)

            new_args = (processed_data,) + args[1:]
            return func(*new_args, **kwargs)
        return wrapper
    return decorator

def conditional_apply(
        predicate: Callable[[Any], bool], 
        default_value: Any = None
        ) -> Callable:
    """
    Decorator that conditionally applies the decorated function based 
    on a predicate.
    
    If the predicate returns False, a default value is returned instead.

    Parameters
    ----------
    predicate : Callable[[Any], bool]
        A function that takes the same arguments as the decorated function and
        returns True if the function should be applied, False otherwise.
    default_value : Any, optional
        The value to return if the predicate evaluates to False. Default is None.

    Returns
    -------
    Callable
        The decorated function with conditional application logic.

    Examples
    --------
    >>> from gofast.tools.funcutils import conditional_apply
    >>> @conditional_apply(predicate=lambda x: x > 0, default_value=0)
    ... def reciprocal(x):
    ...     return 1 / x
    >>> print(reciprocal(2))
    0.5
    >>> print(reciprocal(-1))
    0
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if predicate(*args, **kwargs):
                return func(*args, **kwargs)
            else:
                return default_value
        return wrapper
    return decorator


def apply_transform(
        transformations: List[Callable]) -> Callable:
    """
    Creates a callable that applies a sequence of transformation functions 
    to its input.

    Parameters
    ----------
    transformations : List[Callable]
        A list of functions that will be applied sequentially to the input data.

    Returns
    -------
    Callable
        A new function that takes input data and applies each transformation 
        in sequence.

    Examples
    --------
    >>> from gofast.tools.funcutils import apply_transform
    >>> def square(x):
    ...     return x ** 2
    >>> def increment(x):
    ...     return x + 1
    >>> transform = apply_transform([square, increment])
    >>> print(transform(4))
    17
    """
    def transformed_callable(data):
        for transform in transformations:
            data = transform(data)
        return data
    return transformed_callable

# XXX TODOD 
def make_data_dynamic(
    expected_type: str = 'numeric', 
    capture_columns: bool = False, 
    drop_na: bool = False, 
    na_thresh: Optional[float] = None, 
    na_meth: str = 'drop_rows', 
    reset_index: bool = False
) -> Callable:
    """
    A decorator for preprocessing data before passing it to a function, 
    with options for data type filtering, column selection, missing value 
    handling, and index resetting.

    Parameters
    ----------
    expected_type : str, optional
        Specifies the type of data the function expects: 'numeric' for numeric 
        data, 'categorical' for categorical data, or 'both' for no filtering. 
        Defaults to 'numeric'.
    capture_columns : bool, optional
        If True, uses the 'columns' keyword argument from the decorated function to 
        filter the DataFrame columns. Defaults to False.
    drop_na : bool, optional
        If True, applies missing value handling according to `meth` and `thresh`. 
        Defaults to False.
    na_thresh : Optional[float], optional
        Threshold for dropping rows or columns based on the proportion of missing 
        values. Used only if `drop_na` is True. Defaults to None, meaning no threshold.
    na_meth : str, optional
        Method for dropping missing values: 'drop_rows' to drop rows with missing 
        values, 'drop_cols' to drop columns with missing values, or any other value 
        to drop all missing values without thresholding. Defaults to 'drop_rows'.
    reset_index : bool, optional
        If True, resets the DataFrame index (dropping the current index) before 
        passing it to the function. Defaults to False.

    Examples
    --------
    >>> @make_data_dynamic(expected_type='numeric', capture_columns=True, 
    ... drop_na=True, thresh=0.5, meth='drop_rows', reset_index=True)
    ... def calculate_mean(data: Union[pd.DataFrame, np.ndarray]):
    ...     return data.mean()
    
    >>> data = pd.DataFrame({"A": [1, 2, np.nan], "B": [np.nan, 5, 6]})
    >>> print(calculate_mean(data))
    # This will calculate the mean after preprocessing the input data 
    # according to the specified rules.

    The decorated function seamlessly preprocesses input data, ensuring 
    that it meets the specified criteria for data type, column selection, 
    missing value handling, and index state.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not args:
                raise ValueError("Function requires at least one argument.")
            
            data = _check_and_convert_input(args[0])
            data = _preprocess_data(
                data, capture_columns, expected_type, drop_na, na_thresh,
                na_meth, reset_index, **kwargs)
            
            new_args = (data,) + args[1:]
            return func(*new_args, **kwargs)
        
        return wrapper
    return decorator

def _check_and_convert_input(input_data):
    """
    Check the type of the input data and convert it to a suitable 
    format for processing.
    
    Parameters
    ----------
    input_data : any
        The first argument passed to the decorated function, expected 
        to be either  a pd.DataFrame, dict, np.ndarray, or an iterable that can be 
        converted to an np.ndarray.
    
    Returns
    -------
    pd.DataFrame
        The input data converted to a pandas DataFrame for further processing.
        
    Raises
    ------
    ValueError
        If the input data is not a pd.DataFrame, dict, np.ndarray, or 
        convertible iterable.
    """
    if isinstance(input_data, (pd.DataFrame, np.ndarray)):
        return pd.DataFrame(input_data)
    elif isinstance(input_data, dict):
        return pd.DataFrame(input_data)
    elif hasattr(input_data, '__iter__'):  # Check if it's iterable
        try:
            return pd.DataFrame(np.array(input_data))
        except Exception:
            raise ValueError("Expect the first argument to be an interable object"
                             " with minimum samples equal to two.")
    else:
        raise ValueError("First argument must be a pd.DataFrame, dict,"
                         " np.ndarray, or an iterable object.")

def _preprocess_data(
        data, capture_columns, expected_type, drop_na, na_thresh, 
        na_meth, reset_index, **kwargs):
    """
    Apply preprocessing steps to the input data based on the decorator's 
    parameters.
    
    Parameters
    ----------
    data : pd.DataFrame
        The input data to be preprocessed.
    capture_columns : bool
        Whether to filter the DataFrame columns using the 'columns' 
        keyword argument.
    expected_type : str
        Specifies the expected data type: 'numeric', 'categorical', or 'both'.
    drop_na : bool
        Whether to apply missing value handling.
    na_thresh : Optional[float]
        Threshold for dropping rows or columns based on the proportion of 
        missing values.
    na_meth : str
        Method for dropping missing values.
    reset_index : bool
        Whether to reset the DataFrame index before passing it to the function.
    kwargs : dict
        Additional keyword arguments passed to the decorated function.
    
    Returns
    -------
    pd.DataFrame
        The preprocessed data.
    """
    if capture_columns and 'columns' in kwargs:
        columns = kwargs.pop('columns')
        try:
            data = data[columns]
        except KeyError:
            print("Specified columns do not match, ignoring columns.")

    if expected_type == 'numeric':
        data = data.select_dtypes(include=[np.number])
    elif expected_type == 'categorical':
        data = data.select_dtypes(exclude=[np.number])

    if drop_na:
        if na_meth == 'drop_rows':
            data = data.dropna(axis=0, thresh=(na_thresh * len(
                data.columns) if na_thresh is not None else None))
        elif na_meth == 'drop_cols':
            data = data.dropna(axis=1, thresh=(na_thresh * len(
                data) if na_thresh is not None else None))
        else:
            data = data.dropna()

    if reset_index:
        data = data.reset_index(drop=True)

    return data

def make_data_dynamic0(
    expected_type: str = 'numeric', 
    capture_columns: bool = False, 
    drop_na: bool = False, 
    na_thresh: Optional[float] = None, 
    na_meth: str = 'drop_rows', 
    reset_index: bool = False
):
    """
    A decorator for preprocessing data before passing it to a function, 
    with options for data type filtering, column selection, missing value 
    handling, and index resetting.

    Parameters
    ----------
    expected_type : str, optional
        Specifies the type of data the function expects: 'numeric' for numeric 
        data, 'categorical' for categorical data, or 'both' for no filtering. 
        Defaults to 'numeric'.
    capture_columns : bool, optional
        If True, uses the 'columns' keyword argument from the decorated function to 
        filter the DataFrame columns. Defaults to False.
    drop_na : bool, optional
        If True, applies missing value handling according to `meth` and `thresh`. 
        Defaults to False.
    na_thresh : Optional[float], optional
        Threshold for dropping rows or columns based on the proportion of missing 
        values. Used only if `drop_na` is True. Defaults to None, meaning no threshold.
    na_meth : str, optional
        Method for dropping missing values: 'drop_rows' to drop rows with missing 
        values, 'drop_cols' to drop columns with missing values, or any other value 
        to drop all missing values without thresholding. Defaults to 'drop_rows'.
    reset_index : bool, optional
        If True, resets the DataFrame index (dropping the current index) before 
        passing it to the function. Defaults to False.

    Examples
    --------
    >>> @make_data_dynamic(expected_type='numeric', capture_columns=True, 
    ... drop_na=True, thresh=0.5, meth='drop_rows', reset_index=True)
    ... def calculate_mean(data: Union[pd.DataFrame, np.ndarray]):
    ...     return data.mean()
    
    >>> data = pd.DataFrame({"A": [1, 2, np.nan], "B": [np.nan, 5, 6]})
    >>> print(calculate_mean(data))
    # This will calculate the mean after preprocessing the input data 
    # according to the specified rules.

    The decorated function seamlessly preprocesses input data, ensuring 
    that it meets the specified criteria for data type, column selection, 
    missing value handling, and index state.
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if args and isinstance(args[0], (pd.DataFrame, dict, np.ndarray)):
                data = args[0]
            else:
                raise ValueError(
                    "First argument must be a pd.DataFrame, dict, or np.ndarray")
            if isinstance(data, np.ndarray):
                columns = kwargs.pop('columns', None)
                data = pd.DataFrame(data, columns=( 
                    columns if columns and len(columns) == data.shape[1] else None)
                    )
            
            if capture_columns and 'columns' in kwargs:
                columns = kwargs.pop('columns')
                if columns is not None:
                    try:
                        data = data[columns]
                    except KeyError:
                        print("Specified columns do not match, ignoring columns.")

            if expected_type == 'numeric':
                data = data.select_dtypes(include=[np.number])
            elif expected_type == 'categorical':
                data = data.select_dtypes(exclude=[np.number])

            if drop_na:
                if na_meth=='drop_rows': 
                    data = data.dropna( axis =0 , thresh=( 
                        na_thresh * len(data.columns) if na_thresh is not None else None )
                        ) 
                elif na_meth=='drop_cols': 
                    data = data.dropna( axis =1 , thresh=( 
                        na_thresh * len(data) if na_thresh is not None else None )
                        ) 
                else: data = data.dropna()

            if reset_index:
                data = data.reset_index(drop=True)

            new_args = (data,) + args[1:]
            
            return func(*new_args, **kwargs)
        
        if not hasattr(pd.DataFrame, func.__name__):
            setattr(pd.DataFrame, func.__name__, wrapper)
        
        return wrapper
    return decorator

def preserve_input_type(keep_columns_intact: bool = False):
    """
    Decorator to preserve the original data type of the first positional 
    argument.
    
    It handles special cases for pandas DataFrame to optionally keep the
    original columns intact.
    
    Parameters
    ----------
    keep_columns_intact : bool, optional
        If True and the original data type is a pandas DataFrame, attempts 
        to preserve the original columns in the returned DataFrame. 
        If the conversion with original columns is not feasible, falls back 
        to default DataFrame conversion. Default is False.

    Returns
    -------
    Callable
        The decorated function with preserved input data type functionality.

    Example
    -------
    >>> @preserve_input_type(keep_columns_intact=True)
    ... def to_list_then_back(data):
    ...     return list(data.values())
    ...
    >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> result = to_list_then_back(df)
    >>> print(result)
    >>> print(type(result))
    >>> print(result.columns)
    
    Parameters
    ----------
    keep_columns_intact : bool, optional
        Indicates whether to attempt to preserve the original DataFrame 
        columns in the output.
        Defaults to False.
    
    Returns
    -------
    Callable
        A wrapper function that ensures the result is returned in the 
        original input type.
    """
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if not args:
                return func(*args, **kwargs)
            
            original_type, original_columns = _get_original_type_columns(args[0])
            result = func(*args, **kwargs)
            return _convert_result(result, original_type, original_columns,
                                   keep_columns_intact)

        return wrapper
    
    return decorator

def _get_original_type_columns(data: Any) -> tuple:
    """
    Determines the original type and columns (if applicable) of the input data.
    
    Parameters
    ----------
    data : Any
        The input data to the decorated function.
    
    Returns
    -------
    tuple
        A tuple containing the original type of `data` and its columns 
        if it's a DataFrame.
    """
    original_type = type(data)
    original_columns = getattr(data, 'columns', None)
    return original_type, original_columns

def _convert_result(result: Any, original_type: type,
                    original_columns: pd.Index, keep_columns_intact: bool
                    ) -> Any:
    """
    Attempts to convert the function's result back to the original input type.
    
    Parameters
    ----------
    result : Any
        The result returned by the decorated function.
    original_type : type
        The type of the input data.
    original_columns : pd.Index
        The original columns of the input data if it was a DataFrame.
    keep_columns_intact : bool
        Indicates whether to try to preserve the original DataFrame columns.
    
    Returns
    -------
    Any
        The result possibly converted back to the original input type.
    """
    if original_type is pd.DataFrame:
        return _convert_to_dataframe(result, original_columns, keep_columns_intact)
    elif original_type is not type(result):
        return _attempt_type_conversion(result, original_type)
    return result

def _convert_to_dataframe(result: Any, original_columns: pd.Index, 
                          keep_columns_intact: bool) -> pd.DataFrame:
    """
    Converts the result to a DataFrame, attempting to preserve original 
    columns if specified.
    
    Parameters
    ----------
    result : Any
        The result to be converted.
    original_columns : pd.Index
        Original DataFrame columns to be preserved.
    keep_columns_intact : bool
        Flag indicating whether to preserve original columns.
    
    Returns
    -------
    pd.DataFrame
        The result converted to a DataFrame, with an attempt to preserve 
        original columns.
    """
    if not isinstance(result, pd.DataFrame):
        result = pd.DataFrame(result)
    if keep_columns_intact and original_columns is not None:
        try:
            result = pd.DataFrame(result, columns=original_columns)
        except Exception:
            pass  # If conversion fails, return the result as is
    return result

def _attempt_type_conversion(result: Any, original_type: type) -> Any:
    """
    Attempts to convert the result to the original type.
    
    Parameters
    ----------
    result : Any
        The result to be converted.
    original_type : type
        The original type to convert the result to.
    
    Returns
    -------
    Any
        The result converted to the original type, if possible.
    """
    try:
        if issubclass(original_type, (set, list, tuple)):
            return original_type(result)
    except Exception:
        pass  # If conversion fails, return the result as is
    return result

# Example usage
if __name__ == "__main__":
    @preserve_input_type(keep_columns_intact=True)
    def example_function(data):
        # Example function that converts input to list for demonstration
        return list(data.values())

    original_df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    modified_df = example_function(original_df)
    print(f"Modified type: {type(modified_df)}")
    print(modified_df)

    @preserve_input_type
    def example_function(data):
        # Example function that converts input to list for demonstration
        return list(data)

    original_set = {1, 2, 3}
    modified_set = example_function(original_set)
    print(f"Original type: {type(original_set)}, Modified type: {type(modified_set)}")
    # This should print: Original type: <class 'set'>, Modified type: <class 'set'>
















