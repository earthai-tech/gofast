# -*- coding: utf-8 -*-
"""
Funcutils
=========

`gofast.tools.funcutils` is a utilities package providing various 
functionalities for functional programming tasks.

Features:
    - Currying and Partial Application
    - Function Composition
    - Memoization
    - High-order Functions
    - Utility Functions
"""

import time
import functools
import logging
import subprocess
import sys  
import numpy as np
import pandas as pd
from .._typing import Dict, Any, Callable, List
from .._typing import Optional, Tuple , Union,_T 
from ._dependency import import_optional_dependency
from .coreutils import to_numeric_dtypes

def curry(func):
    """
    Decorator for currying a function.

    Parameters
    ----------
    func : callable
        The function to be curried.

    Returns
    -------
    callable
        The curried function.

    Examples
    --------
    @curry
    def add(x, y):
        return x + y

    add_five = add(5)
    print(add_five(3))  # Output: 8
    """
    @functools.wraps(func)
    def curried(*args, **kwargs):
        if len(args) + len(kwargs) >= func.__code__.co_argcount:
            return func(*args, **kwargs)
        else:
            return functools.partial(curried, *args, **kwargs)
    return curried

def compose(*functions):
    """
    Decorator for composing multiple functions.

    Parameters
    ----------
    *functions : callable
        Functions to be composed.

    Returns
    -------
    callable
        The composed function.

    Examples
    --------
    @compose
    def double(x):
        return x * 2

    increment_and_double = compose(lambda x: x + 1, double)
    print(increment_and_double(3))  # Output: 8
    """
    def composed(*args, **kwargs):
        result = args[0] if args else kwargs.get('result', None)
        for f in reversed(functions):
            result = f(result)
        return result
    return composed

def memoize(func):
    """
    Decorator for memoizing a function.

    Parameters
    ----------
    func : callable
        The function to be memoized.

    Returns
    -------
    callable
        The memoized function.

    Examples
    --------
    @memoize
    def fibonacci(n):
        if n < 2:
            return n
        return fibonacci(n - 1) + fibonacci(n - 2)

    print(fibonacci(10))  # Output: 55
    """
    memo = {}

    @functools.wraps(func)
    def memoized(*args, **kwargs):
        if args not in memo:
            memo[args] = func(*args, **kwargs)
        return memo[args]

    return memoized

def merge_dicts(
    *dicts: Dict[Any, Any], deep_merge: bool = False,
     list_merge: Union[bool, Callable] = False) -> Dict[Any, Any]:
    """
    Merges multiple dictionaries into a single dictionary. Allows for deep
    merging and custom handling of list concatenation.

    Parameters
    ----------
    *dicts : Dict[Any, Any]
        Variable number of dictionary objects to be merged.
    deep_merge : bool, optional
        Enables deep merging of nested dictionaries. Defaults to False.
    list_merge : bool or Callable, optional
        Determines how list values are handled during merge. If True, lists
        are concatenated. If a Callable is provided, it is used to merge lists.
        Defaults to False, which replaces lists from earlier dicts with those
        from later dicts.

    Returns
    -------
    Dict[Any, Any]
        The resulting dictionary from merging all input dictionaries.

    Examples
    --------
    >>> from gofast.tools.funcutils import merge_dicts
    >>> dict_a = {'a': 1, 'b': [2], 'c': {'d': 4}}
    >>> dict_b = {'b': [3], 'c': {'e': 5}}
    >>> print(merge_dicts(dict_a, dict_b))
    {'a': 1, 'b': [3], 'c': {'e': 5}}

    Deep merge with list concatenation:

    >>> print(merge_dicts(dict_a, dict_b, deep_merge=True, list_merge=True))
    {'a': 1, 'b': [2, 3], 'c': {'d': 4, 'e': 5}}

    Deep merge with custom list merge function (taking the max of each list):

    >>> print(merge_dicts(dict_a, dict_b, deep_merge=True,
    ... list_merge=lambda x, y: [max(x + y)]))
    {'a': 1, 'b': [3], 'c': {'d': 4, 'e': 5}}
    """
    def deep_merge_dicts(target, source):
        for key, value in source.items():
            if deep_merge and isinstance(value, dict) and key in target and isinstance(
                    target[key], dict):
                deep_merge_dicts(target[key], value)
            elif isinstance(value, list) and key in target and isinstance(
                    target[key], list):
                if list_merge is True:
                    target[key].extend(value)  # Changed from += to extend for clarity
                elif callable(list_merge):
                    target[key] = list_merge(target[key], value)
                else:
                    target[key] = value
            else:
                target[key] = value
    
    result = {}
    for dictionary in dicts:
        if deep_merge:
            deep_merge_dicts(result, dictionary)
        else:
            for key, value in dictionary.items():
                if key in result and isinstance(result[key], list) and isinstance(
                        value, list):
                    if list_merge is True:
                        result[key].extend(value)
                    elif callable(list_merge):
                        result[key] = list_merge(result[key], value)
                    else:
                        result[key] = value
                else:
                    result.update({key: value})
    return result

def retry_operation(
    func: Callable, 
    retries: int = 3, 
    delay: float = 1.0, 
    catch_exceptions: tuple = (Exception,), 
    backoff_factor: float = 1.0, 
    on_retry: Optional[Callable[[int, Exception], None]] = None,
    pre_process_args: Optional[Callable[[int, Tuple[Any, ...], dict],
                                        Tuple[Tuple[Any, ...], dict]]] = None
):
    """
    Retries a specified function upon failure, for a defined number of retries
    and with a delay between attempts. Enhancements include the ability to 
    pre-process function arguments before each retry attempt.

    Parameters
    ----------
    func : Callable
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
        A callback function that is called after a failed attempt.
    pre_process_args : Optional[Callable[
        [int, Tuple[Any, ...], dict], Tuple[Tuple[Any, ...], dict]]], optional
        A function to pre-process the arguments to `func` before each retry.
        It receives the current attempt number, the arguments, and keyword
        arguments to `func` and returns a tuple of processed arguments and
        keyword arguments.

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
    >>> def test_func(x):
    ...     print(f"Trying with x={x}...")
    ...     raise ValueError("Fail")
    >>> def pre_process(attempt, args, kwargs):
    ...     # Increase argument x by 1 on each retry
    ...     return ((args[0] + 1,), kwargs)
    >>> try:
    ...     retry_operation(test_func, retries=2, delay=0.5,
    ...                        pre_process_args=pre_process,args=(1,))
    ... except ValueError as e:
    ...     print(e)
    """
    args, kwargs = (), {}
    for attempt in range(1, retries + 1):
        try:
            if pre_process_args:
                args, kwargs = pre_process_args(attempt, args, kwargs)
            return func(*args, **kwargs)
        except catch_exceptions as e:
            if attempt < retries:
                if on_retry:
                    on_retry(attempt, e)
                time.sleep(delay)
                delay *= backoff_factor
            else:
                raise e

def flatten_list(
    nested_list: List[Any], 
    depth: int = -1, 
    process_item: Optional[Callable[[Any], Any]] = None
    ) -> List[Any]:
    """
    Flattens a nested list into a single list of values, with optional depth 
    and item processing parameters.

    Parameters
    ----------
    nested_list : List[Any]
        The list to flatten, which may contain nested lists of any depth.
    depth : int, optional
        The maximum depth of list nesting to flatten. A depth of -1 (default)
        means fully flatten, a depth of 0 means no flattening, a depth of 1
        means flatten one level of nesting, and so on.
    process_item : Optional[Callable[[Any], Any]], optional
        A callable that processes each item during flattening. It takes a single
        item and returns the processed item. Default is None, which means no processing.

    Returns
    -------
    List[Any]
        A single, flat list containing all values from the nested list to the
        specified depth, with each item processed if a callable is provided.

    Examples
    --------
    >>> from gofast.tools.funcutils import flatten_list
    >>> nested = [1, [2, 3], [4, [5, 6]], 7]
    >>> flat = flatten_list(nested, depth=1)
    >>> print(flat)
    [1, 2, 3, 4, [5, 6], 7]

    With item processing to square numbers:
    >>> flat_processed = flatten_list(nested, process_item=(lambda x: x**2 if
    ...                                  isinstance(x, int) else x))
    >>> print(flat_processed)
    [1, 4, 9, 16, [25, 36], 49]
    """
    def flatten(current_list: List[Any], current_depth: int) -> List[Any]:
        result = []
        for item in current_list:
            if isinstance(item, list) and current_depth != 0:
                # Decrement depth unless it's infinite (-1)
                new_depth = current_depth - 1 if current_depth > 0 else -1
                result.extend(flatten(item, new_depth))
            else:
                # Apply processing if available
                processed_item = process_item(item) if process_item else item
                result.append(processed_item)
        return result
    
    return flatten(nested_list, depth)

def timeit_decorator(
    logger: Optional[logging.Logger] = None, 
    level: int = logging.INFO
    ):
    """
    A decorator that measures the execution time of a function and optionally
    logs it.

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

def is_valid_if(
    *expected_types: Tuple[type],
    kwarg_types: Optional[Dict[str, type]] = None,
    custom_error: Optional[str] = None,
    skip_check: Optional[Callable[[Any, Any], bool]] = None
):
    """
    A decorator to verify the datatype of positional and keyword parameters 
    of a function.
    
    Allows specifying expected types for both positional and keyword arguments.
    A custom error message and a condition to skip checks can also be provided.

    Parameters
    ----------
    *expected_types : Tuple[type]
        The expected types of the positional parameters.
    kwarg_types : Optional[Dict[str, type]], optional
        A dictionary mapping keyword argument names to their expected types.
    custom_error : Optional[str], optional
        A custom error message to raise in case of a type mismatch. Should
        contain placeholders for positional formatting: {arg_index}, {func_name},
        {expected_type}, and {got_type}.
    skip_check : Optional[Callable[[Any, Any], bool]], optional
        A function that takes the argument list and keyword argument dictionary,
        and returns True if type checks should be skipped.

    Returns
    -------
    Callable
        The decorated function with type verification.

    Raises
    ------
    TypeError
        If the types of the actual parameters do not match the expected types
        and skip_check is False or not provided.

    Examples
    --------
    >>> from gofast.tools.funcutils import is_valid_if
    >>> @is_valid_if(int, float, kwarg_types={'c': str})
    ... def add(a, b, c="default"):
    ...     return a + b
    >>> add(1, 2.5, c="text")
    3.5
    >>> add(1, "2.5", c=100)  # This will raise a TypeError
    TypeError: Argument 2 of 'add' requires <class 'float'> but got <class 'str'>.
    """
    def _construct_error_msg(arg_name, func_name, expected_type, got_type):
        """Constructs a custom or default error message based on provided parameters."""
        if custom_error:
            return custom_error.format(arg_name=arg_name, func_name=func_name,
                                       expected_type=expected_type, got_type=got_type)
        else:
            return (f"Argument '{arg_name}' of '{func_name}' requires {expected_type} "
                    f"but got {got_type}.")

    def _check_arg_types(args, kwargs, func_name):
        """Checks types of positional and keyword arguments."""
        for i, (arg, expected_type) in enumerate(zip(args, expected_types), 1):
            if not isinstance(arg, expected_type):
                error_msg = _construct_error_msg(arg_name=i, func_name=func_name,
                                                 expected_type=expected_type,
                                                 got_type=type(arg))
                raise TypeError(error_msg)

        if kwarg_types:
            for kwarg, expected_type in kwarg_types.items():
                if kwarg in kwargs and not isinstance(kwargs[kwarg], expected_type):
                    error_msg = _construct_error_msg(arg_name=kwarg, func_name=func_name,
                                                     expected_type=expected_type, 
                                                     got_type=type(kwargs[kwarg]))
                    raise TypeError(error_msg)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if skip_check and skip_check(args, kwargs):
                return func(*args, **kwargs)

            _check_arg_types(args, kwargs, func.__name__)

            return func(*args, **kwargs)
        return wrapper
    return decorator

def install_package(
    name: str, extra: str = '', 
    use_conda: bool = False, 
    verbose: bool = True
    ) -> None:
    """
    Install a Python package using either conda or pip, with an option to 
    display installation progress and fallback mechanism.

    This function dynamically chooses between conda and pip for installing 
    Python packages, based on user preference and system configuration. It 
    supports a verbose mode for detailed operation logging and utilizes a 
    progress bar for pip installations.

    Parameters
    ----------
    name : str
        Name of the package to install. Version specification can be included.
    extra : str, optional
        Additional options or version specifier for the package, by default ''.
    use_conda : bool, optional
        Prefer conda over pip for installation, by default False.
    verbose : bool, optional
        Enable detailed output during the installation process, by default True.

    Raises
    ------
    RuntimeError
        If installation fails via both conda and pip, or if the specified installer
        is not available.

    Examples
    --------
    Install a package using pip without version specification:

        >>> install_package('requests', verbose=True)

    Install a specific version of a package using conda:

        >>> install_package('pandas', extra='==1.2.0', use_conda=True, verbose=True)
    
    Notes
    -----
    Conda installations do not display a progress bar due to limitations in capturing
    conda command line output. Pip installations will show a progress bar indicating
    the number of processed output lines from the installation command.
    """
    def execute_command(command: list, progress_bar: bool = False) -> None:
        """
        Execute a system command with optional progress bar for output lines.

        Parameters
        ----------
        command : list
            Command and arguments to execute as a list.
        progress_bar : bool, optional
            Enable a progress bar that tracks the command's output lines, 
            by default False.
        """
        from tqdm import tqdm
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                              text=True, bufsize=1) as process, \
             tqdm(desc="Installing", unit="line", disable=not progress_bar) as pbar:
            for line in process.stdout:
                if verbose:
                    print(line, end='')
                pbar.update(1)
            if process.wait() != 0:  # Non-zero exit code indicates failure
                raise RuntimeError(f"Installation failed for package '{name}{extra}'.")

    conda_available = _check_conda_installed()

    try:
        if use_conda and conda_available:
            if verbose:
                print(f"Attempting to install '{name}{extra}' using conda...")
            execute_command(['conda', 'install', f"{name}{extra}", '-y'], 
                            progress_bar=False)
        elif use_conda and not conda_available:
            if verbose:
                print("Conda is not available. Falling back to pip...")
            execute_command([sys.executable, "-m", "pip", "install", f"{name}{extra}"],
                            progress_bar=True)
        else:
            if verbose:
                print(f"Attempting to install '{name}{extra}' using pip...")
            execute_command([sys.executable, "-m", "pip", "install", f"{name}{extra}"],
                            progress_bar=True)
        if verbose:
            print(f"Package '{name}{extra}' was successfully installed.")
    except Exception as e:
        raise RuntimeError(f"Failed to install '{name}{extra}': {e}") from e

def _check_conda_installed() -> bool:
    """
    Check if conda is installed and available in the system's PATH.

    Returns
    -------
    bool
        True if conda is found, False otherwise.
    """
    try:
        subprocess.check_call(['conda', '--version'], stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def ensure_pkg(
    name: str, 
    extra: str = "",
    errors: str = "raise",
    min_version: str | None = None,
    exception: Exception = None, 
    auto_install: bool = False,
    use_conda: bool = False, 
    partial_check: bool = False,
    condition: Any = None, 
    verbose: bool = False
) -> Callable[[_T], _T]:
    """
    Decorator to ensure a Python package is installed before function execution.

    If the specified package is not installed, or if its installed version does
    not meet the minimum version requirement, this decorator can optionally 
    install or upgrade the package automatically using either pip or conda.

    Parameters
    ----------
    name : str
        The name of the package.
    extra : str, optional
        Additional specification for the package, such as version or extras.
    errors : str, optional
        Error handling strategy if the package is missing: 'raise', 'ignore',
        or 'warn'.
    min_version : str or None, optional
        The minimum required version of the package. If not met, triggers 
        installation.
    exception : Exception, optional
        A custom exception to raise if the package is missing and `errors`
        is 'raise'.
    auto_install : bool, optional
        Whether to automatically install the package if missing. 
        Defaults to False.
    use_conda : bool, optional
        Prefer conda over pip for automatic installation. Defaults to False.
    partial_check : bool, optional
        If True, checks the existence of the package only if the `condition` 
        is met. This allows for conditional package checking based on the 
        function's arguments or other criteria. Defaults to False.
    condition : Any, optional
        A condition that determines whether to check for the package's existence. 
        This can be a callable that takes the same arguments as the decorated function 
        and returns a boolean, a specific argument name to check for truthiness, or 
        any other value that will be evaluated as a boolean. If `None`, the package 
        check is performed unconditionally unless `partial_check` is False.
    verbose : bool, optional
        Enable verbose output during the installation process. Defaults to False.

    Returns
    -------
    Callable
        A decorator that wraps functions to ensure the specified package 
        is installed.

    Examples
    --------
    >>> from gofast.tools.funcutils import ensure_pkg
    >>> @ensure_pkg("numpy", auto_install=True)
    ... def use_numpy():
    ...     import numpy as np
    ...     return np.array([1, 2, 3])

    >>> @ensure_pkg("pandas", min_version="1.1.0", errors="warn", use_conda=True)
    ... def use_pandas():
    ...     import pandas as pd
    ...     return pd.DataFrame([[1, 2], [3, 4]])

    >>> @ensure_pkg("matplotlib", partial_check=True, condition=lambda x, y: x > 0)
    ... def plot_data(x, y):
    ...     import matplotlib.pyplot as plt
    ...     plt.plot(x, y)
    ...     plt.show()
    """
    def decorator(func: _T) -> _T:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if _should_check_condition(
                    partial_check, condition, *args, **kwargs):
                try:
                    # Attempts to import the package, installing 
                    # if necessary and permitted
                    import_optional_dependency(
                        name, extra=extra, errors=errors, 
                        min_version=min_version, exception=exception,
                    )
                except (ModuleNotFoundError, ImportError) as e: #noqa
                    if auto_install:
                        install_package(name, extra=extra, use_conda=use_conda,
                                        verbose=verbose)
                    else:
                        # Reraises the exception with optional custom handling
                        if exception is not None:
                            raise exception
                        raise
                        
            return func(*args, **kwargs)
        return wrapper
    return decorator

def _should_check_condition(
        partial_check: bool, condition: Any, *args, **kwargs) -> bool:
    """
    Determines if the condition for checking the package's existence is met.

    Parameters
    ----------
    partial_check : bool
        Indicates if the package existence check should be conditional.
    condition : Any
        The condition that determines whether to perform the package check.
        This can be a callable, a specific argument name, or any value.
    *args : tuple
        Positional arguments passed to the decorated function.
    **kwargs : dict
        Keyword arguments passed to the decorated function.

    Returns
    -------
    bool
        True if the package check should be performed, False otherwise.
    """
    if not partial_check:
        return True
    
    if callable(condition):
        return condition(*args, **kwargs)
    elif isinstance(condition, str) and condition in kwargs:
        return bool(kwargs[condition])
    else:
        return bool(condition)

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

def make_data_dynamic(
    expected_type: str = 'numeric', 
    capture_columns: bool = False, 
    drop_na: bool = False, 
    na_thresh: Optional[float] = None, 
    na_meth: str = 'drop_rows', 
    reset_index: bool = False, 
    dynamize: bool=True, 
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
    dynamize: bool,
        If True, the preprocessing function is attached as a method to the 
        pandas DataFrame class, allowing it to be called directly on DataFrame
        instances. This enhances the integration and usability of the function
        within pandas workflows, making it more accessible as part of 
        DataFrame's method chain. Defaults to True.

    Examples
    --------
    >>> from gofast.tools.funcutils import make_data_dynamic 
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
        
        if dynamize: 
            if not hasattr(pd.DataFrame, func.__name__):
                setattr(pd.DataFrame, func.__name__, wrapper)
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
        return to_numeric_dtypes (pd.DataFrame(input_data))
    elif isinstance(input_data, dict):
        return to_numeric_dtypes (pd.DataFrame(input_data))
    elif hasattr(input_data, '__iter__'):  # Check if it's iterable
        try:
            return to_numeric_dtypes (pd.DataFrame(np.array(input_data))) 
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

def preserve_input_type(
    keep_columns_intact: bool = False,
    custom_convert: Optional[Callable[[Any, type, Any], Any]] = None,
    fallback_on_error: bool = True
) -> Callable:
    """
    Decorator to preserve the original data type of the first positional 
    argument.
    
    It handles special cases for pandas DataFrame to optionally keep the
    original columns intact.
    
    Parameters
    ----------
    keep_columns_intact : bool, optional
        Attempts to preserve original DataFrame columns in the returned DataFrame.
        Defaults to False.
    custom_convert : Optional[Callable[[Any, type, Any], Any]], optional
        Custom conversion function that takes the result, original type, and 
        original columns (if applicable) as arguments, returning the converted result.
        If None, uses default conversion logic.
    fallback_on_error : bool, optional
        If True, falls back to the unconverted result when conversion fails or
        when `custom_convert` raises an error. Defaults to True.

    Returns
    -------
    Callable
        The decorated function with preserved input data type functionality.
    
    Example
    -------
    >>> from gofast.tools.funcutils import preserve_input_type
    >>> @preserve_input_type(keep_columns_intact=True)
    ... def to_list_then_back(data):
    ...     return pd.DataFrame(list(data.values()))
    ...
    >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> result = to_list_then_back(df)
    >>> print(result)
    >>> print(type(result))
    >>> print(result.columns)
    """
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if not args:
                return func(*args, **kwargs)
            
            original_type, original_columns = _get_original_type_columns(args[0])
            result = func(*args, **kwargs)
            
            if custom_convert:
                try:
                    return custom_convert(result, original_type, original_columns)
                except Exception as e:
                    if not fallback_on_error:
                        raise e
                    # If fallback_on_error=True, simply return the original result
                    return result
                    
            # Conversion back to original type logic, if applicable...    
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
















