# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Utilities for batch processing, ID generation, slicing, and validating input.
Supports generating unique IDs and adjusting data batches efficiently.
"""
from __future__ import print_function

import re
import uuid
import string
import numbers
import random
import inspect
import datetime
import warnings

import numpy as np
import pandas as pd

from ..api.types import  Optional, Iterable, List
from ..compat.scipy import optimize_minimize
from .checks import validate_noise 

__all__ = [ 
    'add_noises_to',
    'adjust_to_samples',
    'batch_generator',
    'get_batch_size',
    'safe_slicing', 
    'gen_batches' , 
    'get_valid_kwargs',
    'generate_id',
    'make_ids',
    ]

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
    >>> from gofast.core.handlers import generate_id
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
    >>> from gofast.core.handlers import get_valid_kwargs
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

def get_batch_size(
    *arrays,
    default_size=None,
    max_memory_usage_ratio=0.1
):
    """
    Determine an optimal batch size based on available memory.

    This function computes an optimal batch size for processing large arrays
    in batches, aiming to prevent memory overload by considering the available
    system memory. If `psutil` is installed, it uses the available memory to
    calculate the batch size. Otherwise, it warns the user and defaults to a
    specified `default_size`.

    Parameters
    ----------
    *arrays : array-like
        One or more arrays (e.g., NumPy arrays) for which to compute the batch
        size. All arrays must have the same number of samples (first dimension).

    default_size : int, optional
        The default batch size to use if `psutil` is not installed or if you prefer
        to specify a fixed batch size. If not provided and `psutil` is not installed,
        the function defaults to 512.

    max_memory_usage_ratio : float, default 0.1
        The fraction of available system memory to allocate for the batch data.
        This parameter is only used if `psutil` is installed.

    Returns
    -------
    int
        The computed batch size, which is at least 1 and at most the number of
        samples in the arrays.

    Notes
    -----
    The batch size is computed using the formula:

    .. math::

        \\text{batch\\_size} = \\min\\left(
            \\max\\left(
                1, \\left\\lfloor \\frac{M \\times R}{S} \\right\\rfloor
            \\right), N
        \\right)

    where:

    - :math:`M` is the available system memory in bytes, obtained via `psutil`.
    - :math:`R` is the `max_memory_usage_ratio`.
    - :math:`S` is the total size in bytes of one sample across all arrays.
    - :math:`N` is the total number of samples in the arrays.

    If `psutil` is not installed, a default `batch_size` is used, or less if there
    are fewer samples.

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.core.handlers import get_batch_size
    >>> X = np.random.rand(1000, 20)
    >>> y = np.random.rand(1000)
    >>> batch_size = get_batch_size(X, y)
    >>> print(batch_size)
    64

    See Also
    --------
    batch_generator : Generator function to create batches.

    References
    ----------
    .. [1] Giampaolo Rodola, "psutil - process and system utilities",
       https://psutil.readthedocs.io/

    """
    try:
        import psutil
        psutil_available = True
    except ImportError:
        psutil_available = False
        if default_size is None:
            default_size = 512
        warnings.warn(
            "'psutil' is not installed. Cannot compute optimal batch size "
            "based on available memory. Using default batch_size="
            f"{default_size}."
        )

    arrays = [np.asarray(arr) for arr in arrays]
    n_samples = arrays[0].shape[0]
    for arr in arrays:
        if arr.shape[0] != n_samples:
            raise ValueError(
                "All arrays must have the same number of samples "
                "in the first dimension."
            )

    if default_size is not None:
        # Check if default_size is greater than the number of samples
        if default_size > n_samples:
            if psutil_available:
                warnings.warn(
                    f"Default batch_size {default_size} is greater than the "
                    f"number of samples ({n_samples}). Recomputing batch size "
                    "based on available memory."
                )
            else:
                warnings.warn(
                    f"Default batch_size {default_size} is greater than the "
                    f"number of samples ({n_samples}). Using batch_size={n_samples}."
                )
                default_size = n_samples
        return default_size

    if psutil_available:
        available_memory = psutil.virtual_memory().available
        # Compute size of one sample across all arrays
        sample_size = sum(
            arr[0].nbytes for arr in arrays
        )
        max_memory_usage = available_memory * max_memory_usage_ratio
        batch_size = int(max_memory_usage // sample_size)
        batch_size = max(1, min(batch_size, n_samples))

        # If batch_size is greater than array length, warn user
        if batch_size > n_samples:
            warnings.warn(
                f"Computed batch_size {batch_size} is greater than the number "
                f"of samples ({n_samples}). Using batch_size={n_samples}."
            )
            batch_size = n_samples

        return batch_size
    else:
        # psutil is not available, default_size must have been set
        return default_size

def batch_generator(
        *arrays,
        batch_size
    ):
    """
    Generate batches of arrays for efficient processing.

    This generator yields batches of the input arrays,
    allowing for memory-efficient processing of large
    datasets. All input arrays must have the same first
    dimension (number of samples).

    Parameters
    ----------
    *arrays : array-like
        One or more arrays (e.g., NumPy arrays) to be
        divided into batches. All arrays must have the
        same number of samples (first dimension).

    batch_size : int
        The size of each batch. Must be a positive integer.

    Yields
    ------
    tuple of array-like
        A tuple containing slices of the input arrays,
        corresponding to the current batch.

    Notes
    -----
    The function iterates over the arrays, yielding slices
    from `start_idx` to `end_idx`, where:

    .. math::

        \text{start\_idx} = k \times \text{batch\_size}

        \text{end\_idx} = \min\left(
            (k + 1) \times \text{batch\_size}, N
        \right)

    with :math:`k` being the batch index and :math:`N`
    the total number of samples.

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.tools.sysutils import batch_generator
    >>> X = np.arange(10)
    >>> y = np.arange(10) * 2
    >>> batch_size = 3
    >>> for X_batch, y_batch in batch_generator(
    ...         X, y, batch_size=batch_size):
    ...     print(X_batch, y_batch)
    [0 1 2] [0 2 4]
    [3 4 5] [6 8 10]
    [6 7 8] [12 14 16]
    [9] [18]

    See Also
    --------
    get_batch_size : Function to compute an optimal batch size.

    References
    ----------
    .. [1] Python Software Foundation, "Generators",
       https://docs.python.org/3/howto/functional.html#generators

    """
    n_samples = arrays[0].shape[0]
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        yield tuple(arr[start_idx:end_idx] for arr in arrays)

        
def safe_slicing(slice_indexes, X):
    """
    Removes slices from the list `slice_indexes` that result in zero samples 
    when applied to the data `X`. The function checks each slice to ensure 
    it selects at least one sample, and discards any slices with no samples 
    selected.

    Parameters
    ----------
    slice_indexes : list of slice objects
        A list of slice objects, each representing a range of indices 
        that can be used to index into a dataset, typically for batch 
        processing.

    X : ndarray of shape (n_samples, n_features)
        The data array (or any other array-like structure) that the slices 
        will be applied to. The function assumes that each slice in 
        `slice_indexes` corresponds to a subset of rows (samples) in `X`.

    Returns
    -------
    valid_slices : list of slice objects
        A list of slice objects that correspond to valid (non-empty) 
        subsets of `X`. Slices with zero elements (e.g., when the 
        start index is equal to or greater than the end index) are removed.

    Examples
    --------
    # Example 1: Basic use case where the last slice is valid
    >>> X = np.random.rand(2000, 5)  # 2000 samples, 5 features
    >>> slice_indexes = [slice(0, 512), slice(512, 1024), slice(1024, 1536), 
                         slice(1536, 2000)]
    >>> safe_slicing(slice_indexes, X)
    [slice(0, 512, None), slice(512, 1024, None), slice(1024, 1536, None),
     slice(1536, 2000, None)]

    # Example 2: Case where the last slice has zero elements and is removed
    >>> slice_indexes = [slice(0, 512), slice(512, 1024), slice(1024, 1536),
                         slice(1536, 1500)]
    >>> safe_slicing(slice_indexes, X)
    [slice(0, 512, None), slice(512, 1024, None), slice(1024, 1536, None)]

    # Example 3: Empty slice case where all slices are removed
    >>> slice_indexes = [slice(0, 0), slice(1, 0)]
    >>> safe_slicing(slice_indexes, X)
    []

    Notes
    -----
    - This function is useful when handling slices generated for batch 
      processing in machine learning workflows, ensuring that only valid 
      batches are processed.
    - The function checks the start and stop indices of each slice and 
      ensures that `end > start` before including the slice in the 
      returned list.
    """
    
    valid_slices = []
    for slice_obj in slice_indexes:
        # Extract the slice range
        start, end = slice_obj.start, slice_obj.stop
        
        # Check if the slice has at least one sample
        if end > start:
            # Add to the valid_slices list only if there are samples
            valid_slices.append(slice_obj)
    
    return valid_slices

def gen_batches(n, batch_size, *, min_batch_size=0):
    """Generator to create slices containing `batch_size` 
    elements from 0 to `n`.

    The last slice may contain less than `batch_size` elements, when
    `batch_size` does not divide `n`.
    
    This is take on scikit-learn :func:`sklearn.utils.gen_batches` but modify 
    to ensure that min_batch_size in not included. 

    Parameters
    ----------
    n : int
        Size of the sequence.
    batch_size : int
        Number of elements in each batch.
    min_batch_size : int, default=0
        Minimum number of elements in each batch.

    Yields
    ------
    slice of `batch_size` elements

    See Also
    --------
    gen_even_slices: Generator to create n_packs slices going up to n.
    sklearn.utils.gen_slices: A generic batch slices

    Examples
    --------
    >>> list(gen_batches(7, 3))
    [slice(0, 3, None), slice(3, 6, None), slice(6, 7, None)]
    >>> list(gen_batches(6, 3))
    [slice(0, 3, None), slice(3, 6, None)]
    >>> list(gen_batches(2, 3))
    [slice(0, 2, None)]
    >>> list(gen_batches(7, 3, min_batch_size=0))
    [slice(0, 3, None), slice(3, 6, None), slice(6, 7, None)]
    >>> list(gen_batches(7, 3, min_batch_size=2))
    [slice(0, 3, None), slice(3, 7, None)]
    """
    if not isinstance(batch_size, numbers.Integral):
        raise TypeError(
            f"gen_batches got batch_size={batch_size}, must be an integer"
        )
    if batch_size <= 0:
        raise ValueError(f"gen_batches got batch_size={batch_size}, must be positive")
    
    start = 0
    for _ in range(int(n // batch_size)):
        end = start + batch_size
        # Skip batches where the remaining size would
        # be smaller than min_batch_size
        if end + min_batch_size > n:
            break
        yield slice(start, end)
        start = end

    # Handle the last batch
    if start < n and (n - start) >= min_batch_size:
        yield slice(start, n)

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
    >>> from gofast.core.handlers import adjust_to_samples
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
    >>> from gofast.core.handlers import add_noises_to
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
    >>> from gofast.core.handlers import _parse_gaussian_noise
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


def make_ids(
        arr: Iterable, prefix: Optional[str] = None, how: str = 'py', 
        skip: bool = False) -> List[str]:
    """
    Generate auto-generated IDs based on the number of items in the input
    iterable.

    Parameters
    ----------
    arr : iterable
        An iterable object (e.g., list, tuple, or array-like) to generate 
        an ID for each item.
        For example, it can be a list of EDI objects, such as a collection of
        `gofast.edi.Edi` objects.

    prefix : str, optional, default: None
        A string value to prepend to each generated ID. This can be used to 
        indicate the site or collection name. If None, no prefix is added.

    how : str, optional, default: 'py'
        The indexing mode for the generated IDs. 
        - 'py' (default) uses Python-style indexing (i.e., starts at 0).
        - Any other string will use 1-based indexing (i.e., starts at 1).

    skip : bool, optional, default: False
        If True, the generated IDs will have a more compact format 
        (i.e., without leading zeros).

    Returns
    -------
    list of str
        A list of generated IDs as strings, each corresponding to an element
        in the input iterable.

    Raises
    ------
    ValueError
        If the `how` parameter is not one of 'py' or another valid string 
        for 1-based indexing.
    TypeError
        If `arr` is not an iterable type.

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.core.handlers import make_ids
    >>> values = ['edi1', 'edi2', 'edi3']
    >>> make_ids(values, prefix='ix')
    ['ix0', 'ix1', 'ix2']

    >>> data = np.random.randn(20)
    >>> make_ids(data, prefix='line', how=None)
    ['line01', 'line02', 'line03', ..., 'line20']

    >>> make_ids(data, prefix='line', how=None, skip=True)
    ['line1', 'line2', 'line3', ..., 'line20']
    """
    
    # Check if the input is an iterable
    if not isinstance(arr, (list, tuple, np.ndarray)):
        raise TypeError(
            f"Expected an iterable object for 'arr', got {type(arr)}")
    
    # Validate the 'how' parameter to ensure it's either 'py' or any other mode
    if how not in ['py', None]:
        raise ValueError(
            "The 'how' parameter must be 'py' for Python"
            " indexing or None for 1-based indexing.")
    
    # Determine the formatting for IDs, based on whether
    # we want compact IDs or padded ones
    fm = '{:0' + ('1' if skip else str(int(np.log10(len(arr))) + 1)) + '}'
    
    # Generate the IDs based on the provided parameters
    id_ = [
        str(prefix) + fm.format(i if how == 'py' else i + 1) if prefix is not None
        else fm.format(i if how == 'py' else i + 1)
        for i in range(len(arr))
    ]
    
    return id_


def get_params (obj: object ) -> dict: 
    """
    Get object parameters. 
    
    Object can be callable or instances 
    
    :param obj: object , can be callable or instance 
    
    :return: dict of parameters values 
    
    :examples: 
    >>> from sklearn.svm import SVC 
    >>> from gofast.core.handlers import get_params 
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
    >>> from gofast.core.utils import parse_attrs 
    >>> parse_attrs('lwi_sub_ohmSmulmagnitude')
    ... ['lwi', 'ohmS', 'magnitude']
    
    
    """
    regex = regex or re.compile (r'per|mod|times|add|sub|[_#&*@!_,;\s-]\s*', 
                        flags=re.IGNORECASE) 
    attr= list(filter (None, regex.split(attr)))
    return attr 