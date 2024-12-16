# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Utilities for array and dataframe manipulation, including data reshaping, 
conversion, and sparse matrix handling to ensure a consistency in array 
formats.
"""

from __future__ import print_function
import re 
import copy
import hashlib
import itertools
import warnings 
import logging

import numpy as np
import pandas as pd

from ..api.types import (
    Any, Union, Series, Tuple, Optional, Set,
    _T,  _F, ArrayLike, List, DataFrame, NDArray
)
from .utils import ( 
    is_iterable, _assert_all_types,sanitize_frame_cols, listing_items_format,  
    )
from .checks import assert_ratio, is_in_if, str2columns, is_numeric_dtype  

__all__ = [ 
    'convert_to_structured_format',
    'denormalize',
    'to_numeric_dtypes',
    'reshape',
    'squeeze_specific_dim',
    'split_list',
    'split_train_test',
    'split_train_test_by_id',
    'make_arr_consistent',
    'process_and_extract_data',
    'to_series_if',
    'test_set_check_id',
    'decode_sparse_data',  
    'map_specific_columns', 
    'reduce_dimensions', 
    ]


def reduce_dimensions(
    arr: np.ndarray,
    z: Union[List, np.ndarray],
    x: Union[List, np.ndarray],
    ops: str = 'reduce',
    axis_names: Tuple[str, str] = ('Z', 'X'),
    error: str ='raise', 
    strict: bool = False,
    logger: Optional[logging.Logger] = None
) -> Union[
    bool, 
    Tuple[np.ndarray, Union[List, np.ndarray], Union[List, np.ndarray]]
]:
    """
    Reduce or Check the Dimensionality of a 2D Data Array.
    
    The ``reduce_dimensions`` function offers a robust mechanism to verify 
    and adjust the dimensionality of a 2D NumPy array ``data`` based on the 
    lengths of the provided axes ``z`` and ``x``. It supports two primary 
    operations:
    
    - **Check Only**: Validates whether the dimensions of ``data`` align with 
      the lengths of ``z`` and ``x``.
    - **Reduce**: Adjusts the dimensions of ``data``, ``z``, and ``x`` to 
      ensure consistency by truncating or padding as necessary.
    
    This utility is particularly beneficial in data preprocessing pipelines 
    where dimensional consistency is crucial for downstream analyses.
    
    .. math::
        \text{Operation} = 
        \begin{cases}
            \text{Check Dimensions} & \text{if } \text{ops} = \text{'check_only'} \\
            \text{Adjust Dimensions} & \text{if } \text{ops} = \text{'reduce'}
        \end{cases}
    
    Parameters
    ----------
    arr : `numpy.ndarray`
        The input 2D array with dimensions corresponding to ``z`` and ``x``.
    
    z : Union[`List`, `numpy.ndarray`]
        Array-like object representing the Z-axis. Should be reduced or checked 
        against the first dimension of ``data``.
    
    x : Union[`List`, `numpy.ndarray`]
        Array-like object representing the X-axis. Should be reduced or checked 
        against the second dimension of ``data``.
    
    ops : `str`, default=`'reduce'`
        Operation mode.
        
        - ``'check_only'``: Validate if ``z`` and ``x`` can be reduced to match 
          ``data``.
        - ``'reduce'``: Perform the reduction of ``z``, ``x``, and ``data`` to 
          match ``data``'s dimensions.
    
    axis_names : `Tuple[str, str]`, default=(``'Z'``, ``'X'``)
        Names of the axes for logging purposes.
    
    error : str, default='raise'
        The strategy for handling validation errors.
        - ``'warn'``: Issues warnings and continues processing.
        - ``'raise'``: Raises exceptions upon encountering errors.
    
    strict : `bool`, default=`False`
        If ``True``, enforces that no padding occurs during reduction. Only 
        truncation is allowed. If padding is required and ``strict`` is ``True``, 
        an error is raised.
        
    logger : `logging.Logger`, optional
        Logger instance for debug messages. If ``None``, a default logger is used.
    
    Returns
    -------
    Union[
        `bool`, 
        Tuple[`numpy.ndarray`, Union[`List`, `numpy.ndarray`], 
              Union[`List`, `numpy.ndarray`]]
    ]
        - If ``ops='check_only'``, returns ``True`` if dimensions match or can 
          be reduced, otherwise ``False``.
        - If ``ops='reduce'``, returns a tuple containing the reduced ``data``, 
          ``z``, and ``x``.
    
    Raises
    ------
    ValueError
        - If ``ops`` is not ``'check_only'`` or ``'reduce'``.
        - If reduction is not possible and ``ops='check_only'``.
        - If ``strict`` is ``True`` and padding is required.
    
    TypeError
        - If ``data`` is not a 2D ``numpy.ndarray``.
        - If ``z`` and ``x`` are not list-like.
    
    Examples
    --------
    >>> import numpy as np
    >>> from gofast.core.array_manager import reduce_dimensions
    >>> 
    >>> # Sample data array
    >>> data = np.array([
    ...     [1, 2, 3],
    ...     [4, 5, 6],
    ...     [7, 8, 9]
    ... ])
    >>> 
    >>> # Corresponding axes
    >>> z = [10, 20, 30]
    >>> x = [100, 200, 300]
    >>> 
    >>> # Check dimensionality
    >>> result = reduce_dimensions(data, z, x, ops='check_only')
    >>> print(result)
    True
    >>> 
    >>> # Reduce dimensions (no change needed in this case)
    >>> reduced_data, reduced_z, reduced_x = reduce_dimensions(
    ...     data, z, x, ops='reduce'
    ... )
    >>> print(reduced_data)
    [[1 2 3]
     [4 5 6]
     [7 8 9]]
    >>> print(reduced_z)
    [10, 20, 30]
    >>> print(reduced_x)
    [100, 200, 300]
    >>> 
    >>> # Dimensionality mismatch
    >>> z_mismatch = [10, 20]
    >>> result = reduce_dimensions(data, z_mismatch, x, ops='check_only')
    >>> print(result)
    False
    >>> 
    >>> # Reduce dimensions with mismatch
    >>> reduced_data, reduced_z, reduced_x = reduce_dimensions(
    ...     data, z_mismatch, x, ops='reduce'
    ... )
    >>> print(reduced_data)
    [[1 2 3]
     [4 5 6]]
    >>> print(reduced_z)
    [10, 20]
    >>> print(reduced_x)
    [100, 200, 300]
    
    Notes
    -----
    - **Logging**: The function utilizes a logger to provide debug information 
      during the dimensionality check and reduction processes. Users can 
      supply their own logger or rely on the default logger.
    
    - **Axis Adjustment**: When reducing dimensions, if the length of an 
      axis exceeds the corresponding dimension in ``data``, the axis is 
      truncated. If it is shorter, the axis is either padded with ``None`` 
      values or an error is raised based on the ``strict`` parameter.
    
    - **Error Handling**: By default, the function raises exceptions for any 
      mismatches or invalid parameters. Users can modify the behavior by 
      adjusting the ``ops`` and ``strict`` parameters.
    
    - **Flexibility**: The function is designed to handle various 
      dimensionality scenarios, making it suitable for preprocessing steps 
      in data analysis workflows.
    
    See Also
    --------
    numpy.ndarray : The primary array object in NumPy.
    logging.Logger : Logger class for debug messages.
    
    References
    ----------
    .. [1] NumPy Documentation: numpy.ndarray.  
       https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html  
    .. [2] Python Documentation: logging.  
       https://docs.python.org/3/library/logging.html  
    .. [3] Freedman, D., & Diaconis, P. (1981). On the histogram as a density 
           estimator: L2 theory. *Probability Theory and Related Fields*, 57(5), 
           453-476.
    """

    # Set up logger
    if logger is None:
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(levelname)s:%(name)s:%(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)
    
    # Validate input types
    if not isinstance(arr, np.ndarray):
        raise TypeError(
            f"'data' must be a numpy.ndarray, got {type(arr).__name__}."
        )
    if arr.ndim != 2:
        raise ValueError(
            f"'data' must be a 2D array, got {arr.ndim}D array."
        )
    if not isinstance(z, (list, np.ndarray)):
        raise TypeError(
            f"'z' must be a list or numpy.ndarray, got {type(z).__name__}."
        )
    if not isinstance(x, (list, np.ndarray)):
        raise TypeError(
            f"'x' must be a list or numpy.ndarray, got {type(x).__name__}."
        )
    
    # Validate 'ops' parameter
    if ops not in {'check_only', 'reduce'}:
        raise ValueError(
            f"Invalid 'ops' value '{ops}'. Expected 'check_only' or 'reduce'."
        )
    
    # Define helper function to adjust axis lengths
    def adjust_axis_length(
        desired_length: int, 
        current_axis: Union[List, np.ndarray], 
        axis_name: str
    ) -> Tuple[int, Union[List, np.ndarray]]:
        """
        Adjust the length of an axis to match the desired length.
        
        Parameters
        ----------
        desired_length : `int`
            The target length for the axis.
        
        current_axis : Union[`List`, `numpy.ndarray`]
            The current axis data to be adjusted.
        
        axis_name : `str`
            The name of the axis for logging purposes.
        
        Returns
        -------
        Tuple[int, Union[`List`, `numpy.ndarray`]]
            A tuple containing the new length and the adjusted axis.
        """
        original_length = len(current_axis)
        if original_length > desired_length:
            adjusted_axis = current_axis[:desired_length]
            logger.debug(
                f"Resizing ``{axis_name}`` from {original_length} to "
                f"{desired_length}."
            )
        elif original_length < desired_length:
            if strict:
                msg = (
                    f"Cannot pad ``{axis_name}`` to match desired length "
                    f"{desired_length}."
                )
                if ops == 'reduce':
                    raise ValueError(msg)
                elif error == 'raise':
                    raise ValueError(msg)
                else:
                    warnings.warn(msg)
                    return desired_length, current_axis
            adjusted_axis = list(current_axis) + [None] * (
                desired_length - original_length
            )
            logger.debug(
                f"Resizing ``{axis_name}`` from {original_length} to "
                f"{desired_length} by padding."
            )
        else:
            adjusted_axis = current_axis
            logger.debug(
                f"No resizing needed for ``{axis_name}``; already at desired "
                f"length {desired_length}."
            )
        return desired_length, adjusted_axis
    
    if ops == 'check_only':
        z_matches = len(z) == arr.shape[0]
        x_matches = len(x) == arr.shape[1]
        if z_matches and x_matches:
            logger.debug(
                "Data dimensions match ``z`` and ``x`` dimensions."
            )
            return True
        else:
            msg = "Data dimensionality does not match ``z`` and/or ``x`` dimensions."
            if not z_matches and not x_matches:
                msg += (
                    f" ``z`` has length {len(z)} vs {arr.shape[0]}, and "
                    f"``x`` has length {len(x)} vs {arr.shape[1]}."
                )
            elif not z_matches:
                msg += f" ``z`` has length {len(z)} vs {arr.shape[0]}."
            else:
                msg += f" ``x`` has length {len(x)} vs {arr.shape[1]}."
            if error == 'raise':
                raise ValueError(msg)
            else:
                warnings.warn(msg)
                return False
    
    elif ops == 'reduce':
        # Determine new lengths based on current data shape and axis lengths
        new_z_length = min(len(z), arr.shape[0])
        new_x_length = min(len(x), arr.shape[1])
        
        # Adjust 'z' axis
        sz0, z_adjusted = adjust_axis_length(
            desired_length=new_z_length, 
            current_axis=z, 
            axis_name=axis_names[0]
        )
        
        # Adjust 'x' axis
        sx0, x_adjusted = adjust_axis_length(
            desired_length=new_x_length, 
            current_axis=x, 
            axis_name=axis_names[1]
        )
        
        # Slice the data accordingly
        data_reduced = arr[:sz0, :sx0]
        logger.debug(
            "Data, ``z``, and ``x`` have been reduced to match dimensions."
        )
        return data_reduced, z_adjusted, x_adjusted
    
def decode_sparse_data(sparse_data: pd.Series) -> pd.DataFrame:
    """
    Decode a sparse matrix represented as strings in a pandas Series 
    back into a dense pandas DataFrame.
    
    Each entry in the `sparse_data` Series should contain multiple lines, 
    where each line represents a non-zero entry in the format:
    `(row_index, column_index)\tvalue`.
    
    **Note:** This function assumes that the row indices within the strings 
    are always `(0, column_index)`. The Series index is used as the actual 
    row index in the decoded DataFrame.
    
    Parameters
    ----------
    sparse_data : pd.Series
        A pandas Series where each element is a string representing a 
        sparse matrix row. Each string contains entries separated by 
        newline characters (`\n`), and each entry is in the format 
        `(row, col)\tvalue`.
    
    Returns
    -------
    pd.DataFrame
        A dense pandas DataFrame reconstructed from the sparse representation.
    
    Raises
    ------
    ValueError
        If an entry in `sparse_data` is not a string or does not follow the 
        expected format.
    
    Examples
    --------
    >>> from gofast.core.array_manager import decode_sparse_data
    >>> import pandas as pd
    >>> 
    >>> # Sample sparse data as a pandas Series
    >>> sparse_data = pd.Series([
    ...     "(0, 0)\t-1.6752467319482305\n(0, 1)\t1.515...",
    ...     "(0, 0)\t-1.5597124745724904\n(0, 1)\t-0.00...",
    ...     "(0, 0)\t-1.4441782171967503\n(0, 1)\t-1.41...",
    ...     "(0, 0)\t-1.3286439598210102\n(0, 1)\t0.912...",
    ...     "(0, 0)\t-1.2131097024452704\n(0, 1)\t-0.41..."
    ... ])
    >>> 
    >>> # Decode the sparse data
    >>> decoded_df = decode_sparse_data(sparse_data)
    >>> print(decoded_df)
               0     1
        0 -1.675  1.515
        1 -1.560 -0.000
        2 -1.444 -1.410
        3 -1.329  0.912
        4 -1.213 -0.410
    
    Notes
    -----
    - **Input Structure:** Each entry in the `sparse_data` Series should be a 
      string containing multiple `(row, column)\tvalue` pairs separated by 
      newline characters.
    - **Row Mapping:** The Series index is used as the row index in the 
      decoded DataFrame. The row index specified within each string is 
      ignored and assumed to be `0`.
    - **Memory Consideration:** Decoding large sparse matrices into dense 
      DataFrames can consume significant memory. Consider using sparse 
      DataFrame representations if memory usage is a concern.
    
    References
    ----------
    - pandas.DataFrame: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
    - scipy.sparse.coo_matrix: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html
    """
    from scipy.sparse import coo_matrix
    
    if isinstance ( sparse_data, pd.DataFrame): 
        # try to squeeze the dataframe if has a single column 
        sparse_data = sparse_data.squeeze() 
        
    if not isinstance(sparse_data, pd.Series):
        raise ValueError("Input `sparse_data` must be a pandas Series.")
    
    rows = []
    cols = []
    data = []
    
    for series_idx, row in sparse_data.items():
        if not isinstance(row, str):
            row = str(row)  # Convert to string if possible
        
        for entry in row.split('\n'):
            entry = entry.strip()
            if entry:
                try:
                    col_row, value = entry.split('\t')
                    # Remove parentheses and split into row and column
                    _, col_idx = map(int, col_row.strip('()').split(','))
                    rows.append(series_idx)  # Use Series index as row index
                    cols.append(col_idx)
                    data.append(float(value))
                except Exception as e:
                    raise ValueError(
                        f"Error parsing entry '{entry}'"
                        f" in Series index {series_idx}: {e}"
                    )
    
    if not rows:
        raise ValueError("No data found to decode.")
    
    # Determine the size of the matrix
    max_row = max(rows) + 1
    max_col = max(cols) + 1
    
    # Create a COO sparse matrix
    sparse_matrix = coo_matrix(
        (data, (rows, cols)), shape=(max_row, max_col)
    )
    
    # Convert to a dense NumPy array
    dense_array = sparse_matrix.toarray()
    
    # Convert to a pandas DataFrame
    df = pd.DataFrame(dense_array)
    
    return df

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
    >>> from gofast.core.array_manager import process_and_extract_data
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
    Constructs a pandas Series from given values, optionally naming 
    the series and its index.

    Parameters
    ----------
    *values : Any
        A variable number of inputs, each can be a scalar, float, int,\
            or array-like object.
    value_names : Optional[List[str]]
        Names to be used for the index of the series. If not provided or 
        if its length doesn't match the number of values, default numeric
        index is used.
    name : Optional[str]
        Name of the series.
    error : str, default 'ignore'
        Error handling strategy ('ignore' or 'raise'). If 'raise', errors 
        during series
        construction lead to an exception.
    **kws : dict
        Additional keyword arguments passed to `pd.Series` constructor.

    Returns
    -------
    pd.Series or original values
        A pandas Series constructed from the inputs if successful, otherwise,
        the original values if the series construction is not applicable.

    Examples
    --------
    >>> from gofast.core.array_manager import to_series_if
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
            raise ValueError("Length of `value_names` does not"
                             " match the number of values.")
        value_names = None  # Reset to default indexing
    # Attempt to construct series
    try:
        # Flatten array-like inputs to avoid creating Series of lists/arrays
        flattened_values = [val[0] if isinstance(
            val, (list,tuple,  np.ndarray, pd.Series)) 
            and len(val) == 1 else val for val in values]
        series = pd.Series(
            flattened_values, index=value_names, name=name, **kws)
    except Exception as e:
        if error == 'raise':
            raise ValueError(f"Failed to construct series due to: {e}")
        return values  # Return the original values if series construction fails

    return series

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
    >>> from gofast.core.array_manager import make_arr_consistent
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
    >>> from gofast.core.array_manager import split_train_test
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
    >>> from gofast.core.array_manager import test_set_check_id
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
    >>> from gofast.core.array_manager import split_train_test_by_id
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
    >>> from gofast.core.array_manager import split_list
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

    >>> from gofast.core.array_manager import squeeze_specific_dim
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
        
def reshape(arr , axis = None) :
    """ Detect the array shape and reshape it accordingly, back to the given axis. 
    
    :param array: array_like with number of dimension equals to 1 or 2 
    :param axis: axis to reshape back array. If 'axis' is None and 
        the number of dimension is greater than 1, it reshapes back array 
        to array-like 
    
    :returns: New reshaped array 
    
    :Example: 
        >>> import numpy as np 
        >>> from gofast.core.array_manager import reshape 
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

def to_numeric_dtypes(
    arr: Union[NDArray, DataFrame], *, 
    columns: Optional[List[str]] = None, 
    return_feature_types: bool = False, 
    missing_values: float = np.nan, 
    pop_cat_features: bool = False, 
    sanitize_columns: bool = False, 
    regex: Optional[re.Pattern] = None, 
    fill_pattern: str = '_', 
    drop_nan_columns: bool = ..., 
    how: str = 'all', 
    reset_index: bool = False, 
    drop_index: bool = ..., 
    verbose: bool = False
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
    >>> from gofast.core.array_manager import to_numeric_dtypes
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

    # pass ellipsis argument to False 
    # ( sanitize_columns, reset_index, 
    #  verbose,return_feature_types, 
    #  pop_cat_features, 
    #     ) = ellipsis2false(
    #         sanitize_columns, 
    #         reset_index, 
    #         verbose,
    #         return_feature_types, 
    #         pop_cat_features
    # )
   
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
        if not is_numeric_dtype(df.columns , to_array=True): 
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
        if is_numeric_dtype(df[serie], to_array =True ): 
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
    >>> from gofast.core.array_manager import denormalize
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
    >>> from gofast.core.array_manager import convert_to_structured_format
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

def map_specific_columns ( 
    data: DataFrame, 
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
    data: dataframe, 
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
    >>> from gofast.utils.plotutils import map_specific_columns 
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
    X = _assert_all_types(data, pd.DataFrame)
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
    
def concat_array_from_list(list_of_array, concat_axis=0):
    """
    Concatenate arrays from a list and replace `None` values with `NaN`.

    Parameters
    ----------
    list_of_array : list of array-like
        A list containing arrays that will be concatenated. If an element 
        is `None`, it will be replaced with a `NaN` array.
    concat_axis : int, optional
        The axis along which to concatenate the arrays. Must be either `0` 
        (rows) or `1` (columns). Default is `0`.

    Returns
    -------
    np.ndarray
        The concatenated array with shape determined by the length of the 
        arrays in the list and the specified `concat_axis`.

    Raises
    ------
    ValueError
        If `concat_axis` is not either `0` or `1`.

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.core.array_manager import concat_array_from_list
    >>> np.random.seed(0)
    >>> ass = np.random.randn(10)
    >>> ass2 = np.linspace(0, 15, 10)
    >>> concat_array_from_list([ass, ass2])
    array([[ 1.76405235,  0.        ],
           [ 0.40015721,  1.66666667],
           [ 0.97873798,  3.33333333],
           [ 2.2408932 ,  5.        ],
           [ 1.86755799,  6.66666667],
           [ -0.97727788,  8.33333333],
           [ 0.95008842, 10.        ],
           [ -0.15135721, 11.66666667],
           [ -0.10321885, 13.33333333],
           [ 0.4105985 , 15.        ]])

    Notes
    -----
    - The function will reshape one-dimensional arrays to two 
      dimensions if necessary, depending on the `concat_axis` value.
    - If the list contains only one array, it will return the reshaped 
      array without concatenation.
    """
    concat_axis = int(_assert_all_types(concat_axis, int, float))
    
    if concat_axis not in (0, 1):
        raise ValueError(f'Unable to understand axis: {str(concat_axis)!r}')
    
    # Replace None with NaN arrays
    list_of_array = list(map(
        lambda e: np.array([np.nan]) if e is None else np.array(e), 
        list_of_array
    ))

    # If the list has only one element, reshape it accordingly
    if len(list_of_array) == 1:
        ar = (list_of_array[0].reshape(
                (1, len(list_of_array[0]))
            ) if concat_axis == 0 else list_of_array[0].reshape(
                (len(list_of_array[0]), 1)
            )
        ) if list_of_array[0].ndim == 1 else list_of_array[0]
        return ar

    # Reshape arrays if necessary before concatenation
    list_of_array = list(map(
        lambda e: e.reshape(e.shape[0], 1) if e.ndim == 1 else e, 
        list_of_array
    )) if concat_axis == 1 else list(map(
        lambda e: e.reshape(1, e.shape[0]) if e.ndim == 1 else e, 
        list_of_array
    ))

    return np.concatenate(list_of_array, axis=concat_axis)

# def _concat_array_from_list (list_of_array , concat_axis = 0) :
#     """ Concat array from list and set the None value in the list as NaN.
    
#     :param list_of_array: List of array elements 
#     :type list of array: list 
    
#     :param concat_axis: axis for concatenation ``0`` or ``1``
#     :type concat_axis: int 
    
#     :returns: Concatenated array with shape np.ndaarry(
#         len(list_of_array[0]), len(list_of_array))
#     :rtype: np.ndarray 
    
#     :Example: 
        
#     >>> import numpy as np 
#     >>> from gofast.core.utils import concat_array_from_list 
#     >>> np.random.seed(0)
#     >>> ass=np.random.randn(10)
#     >>> ass = ass2=np.linspace(0,15,10)
#     >>> concat_array_from_list ([ass, ass]) 
    
#     """
#     concat_axis =int(_assert_all_types(concat_axis, int, float))
#     if concat_axis not in (0 , 1): 
#         raise ValueError(f'Unable to understand axis: {str(concat_axis)!r}')
    
#     list_of_array = list(map(lambda e: np.array([np.nan])
#                              if e is None else np.array(e), list_of_array))
#     # if the list is composed of one element of array, keep it outside
#     # reshape accordingly 
#     if len(list_of_array)==1:
#         ar = (list_of_array[0].reshape ((1,len(list_of_array[0]))
#                  ) if concat_axis==0 else list_of_array[0].reshape(
#                         (len(list_of_array[0]), 1)
#                  )
#              ) if list_of_array[0].ndim ==1 else list_of_array[0]
                     
#         return ar 

#     #if concat_axis ==1: 
#     list_of_array = list(map(
#             lambda e:e.reshape(e.shape[0], 1) if e.ndim ==1 else e ,
#             list_of_array)
#         ) if concat_axis ==1 else list(map(
#             lambda e:e.reshape(1, e.shape[0]) if e.ndim ==1 else e ,
#             list_of_array))
                
#     return np.concatenate(list_of_array, axis = concat_axis)