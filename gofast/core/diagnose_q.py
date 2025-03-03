# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Diagnose and validating quantile-related operations. 
Provides utilities for handling quantile data in various formats 
and ensuring compatibility with expected structures.
"""

import re 
import warnings 
from typing import List, Union, Any, Optional  
import numpy as np 
import pandas as pd 

from .io import to_frame_if
    
__all__= [ 
    'to_iterable',  'validate_quantiles', 
    'validate_quantiles_in', 'validate_q_dict',
    'check_forecast_mode', 'detect_quantiles_in',
    'build_q_column_names',
]

def check_forecast_mode(
    mode, 
    q=None, 
    error="raise", 
    ops="validate",  
    ):
    r"""
    Check consistency between forecast mode and quantile values.

    This function verifies that the provided forecast `mode`
    is consistent with the quantile values (`q`). If the mode
    is ``"point"`` and quantile values are provided, it will
    either warn the user and reset `q` to ``None`` (if
    ``error=="warn"``) or raise a ValueError (if
    ``error=="raise"``). Similarly, if the mode is
    ``"quantile"`` and no quantile values are provided, it will
    either warn the user and set `q` to the default values
    ``[0.1, 0.5, 0.9]`` (if ``error=="warn"``) or raise a
    ValueError (if ``error=="raise"``).
    
    Additionally, if ``ops`` is set to ``"check_only"``, the function 
    only performs the checks without modifying or returning ``q``.

    Parameters
    ----------
    mode : str
        Forecast mode, either ``"point"`` or
        ``"quantile"``.
    q : list of float, optional
        List of quantile values. Defaults to ``None``.
    error : str, optional
        Error handling behavior. If set to ``"raise"``, a
        ValueError is raised when an inconsistency is
        detected. If set to ``"warn"``, a warning is issued
        and a default behavior is applied.
    ops : str, optional
        Operation mode. If set to ``"check_only"``, the function only 
        performs the checks without returning any value. If set to 
        ``"validate"``, the function returns the validated 
       (or updated) quantile values. Default is ``"validate"``.
        
 
    Returns
    -------
    q : list of float or None
        The validated (or updated) quantile values if ``ops`` is 
        ``"validate"``; otherwise, returns ``None``.
    
    Raises
    ------
    ValueError
        If an inconsistency is detected and ``error`` is set to 
        ``"raise"``.
    
    Examples
    --------
    >>> from gofast.utils.diagnose_q impor check_forecast_mode 
    >>> check_forecast_mode("point", q=[0.1, 0.5, 0.9])
    # Raises a ValueError or warns and returns None based on the error flag.
    
    >>> check_forecast_mode("quantile", q=None, error="warn")
    # Issues a warning and returns [0.1, 0.5, 0.9].
    
    """
    # Ensure mode is valid.
    if mode not in ["point", "quantile"]:
        raise ValueError(
            "mode must be either 'point' or 'quantile'."
        )
    
    # Handle the case for "point" mode.
    if mode == "point":
        if q is not None:
            msg = (
                "In point mode, quantile values (q) should be None. "
                "Resetting q to None."
            )
            if error == "warn":
                warnings.warn(msg)
                q = None
            elif error == "raise":
                raise ValueError(msg)
    
    # Handle the case for "quantile" mode.
    elif mode == "quantile":
        if q is None:
            msg = (
                "In quantile mode, quantile values (q) must be provided. "
                "Setting default quantiles to [0.1, 0.5, 0.9]."
            )
            if error == "warn":
                warnings.warn(msg)
                q = [0.1, 0.5, 0.9]
            elif error == "raise":
                raise ValueError(msg)
                
        # then validate quantiles 
        q= validate_quantiles (q )
    # If ops is "check_only", simply return None.
    if ops == "check_only":
        return None
    else:
        return q
    
def to_iterable(
    obj: Any,
    exclude_string: bool = False,
    transform: bool = False,
    parse_string: bool = False,
    flatten: bool = False,
    unique: bool = False,
    delimiter: str = r'[ ,;|\t\n]+'
) -> Union[bool, List[Any]]:
    r"""
    Determines if an object is iterable, with options to transform, parse,
    and modify the input for flexible iterable handling.

    Parameters
    ----------
    obj : Any
        Object to be evaluated or transformed into an iterable.
    exclude_string : bool, default=False
        Excludes strings from being considered as iterable objects.
    transform : bool, default=False
        Transforms `obj` into an iterable if it isn't already. Defaults to
        wrapping `obj` in a list.
    parse_string : bool, default=False
        If `obj` is a string, splits it into a list based on the specified
        `delimiter`. Requires `transform=True`.
    flatten : bool, default=False
        If `obj` is a nested iterable, flattens it into a single list.
    unique : bool, default=False
        Ensures unique elements in the output if `transform=True`.
    delimiter : str, default=r'[ ,;|\t\n]+'
        Regular expression pattern for splitting strings when `parse_string=True`.

    Returns
    -------
    bool or List[Any]
        Returns a boolean if `transform=False`, or an iterable if
        `transform=True`.

    Raises
    ------
    ValueError
        If `parse_string=True` without `transform=True`, or if `delimiter`
        is invalid.

    Notes
    -----
    - When `parse_string` is used, strings are split by `delimiter` to form a
      list of substrings.
    - `flatten` and `unique` apply only when `transform=True`.
    - Using `unique=True` ensures no duplicate values in the output.

    Examples
    --------
    >>> from gofast.core.diagnose_q import to_iterable
    >>> to_iterable("word", exclude_string=True)
    False

    >>> to_iterable(123, transform=True)
    [123]

    >>> to_iterable("parse, this sentence", transform=True, parse_string=True)
    ['parse', 'this', 'sentence']

    >>> to_iterable([1, [2, 3], [4]], transform=True, flatten=True)
    [1, 2, 3, 4]

    >>> to_iterable("a,b,a,b", transform=True, parse_string=True, unique=True)
    ['a', 'b']
    """
    if parse_string and not transform:
        raise ValueError("Set 'transform=True' when using 'parse_string=True'.")

    # Check if object is iterable (excluding strings if specified)
    is_iterable = hasattr(obj, '__iter__') and not (
        exclude_string and isinstance(obj, str))

    # If transformation is not needed, return the boolean check
    if not transform:
        return is_iterable

    # If string parsing is enabled and obj is a string, split it using delimiter
    if isinstance(obj, str) and parse_string:
        obj = re.split(delimiter, obj.strip())

    # Wrap non-iterables into a list if they aren't iterable
    elif not is_iterable:
        obj = [obj]

    # Flatten nested iterables if flatten=True
    if flatten:
        obj = _flatten(obj)

    # Apply unique filtering if requested
    if unique:
        obj = list(dict.fromkeys(obj))  # Preserves order while ensuring uniqueness

    return obj

def _flatten(nested_list: Any) -> List[Any]:
    """ Helper function to recursively flatten a nested list structure. """
    flattened = []
    for element in nested_list:
        if isinstance(element, (list, tuple, set)):
            flattened.extend(_flatten(element))
        else:
            flattened.append(element)
    return flattened

def validate_q_dict(q_dict, recheck=False):
    """
    Converts the keys of a dictionary of quantile columns (`q_dict`) from 
    string representations to numeric values (float) if possible. If the key 
    cannot be converted to a number, it returns the dictionary as is. 

    Optionally validates the quantiles after conversion to ensure that all 
    keys are within the valid range of quantiles [0, 1].

    Parameters
    ----------
    q_dict : dict
        A dictionary where the keys represent quantiles (either as 
        strings like '0.1' or '10%') and the values are lists of 
        column names associated with those quantiles.
        
    recheck : bool, optional, default=False
        If `True`, the keys of the dictionary will be validated as 
        quantiles after the conversion. This validation checks that 
        all keys lie within the range [0, 1].

    Returns
    -------
    dict
        A dictionary with numeric keys if conversion is successful, 
        otherwise the original dictionary. The keys are either floats 
        representing quantiles or the original keys if they cannot 
        be converted.

    Notes
    -----
    The function performs the following steps:
    
    1. Iterates over the dictionary to check whether each key can be 
       converted to a numeric value.
    2. If a key is a string containing a percentage (e.g., '10%'), 
       it removes the '%' sign and divides the value by 100 to 
       convert it to a float.
    3. If a key can be successfully converted, it is stored as a 
       floating-point number in the resulting dictionary.
    4. If the conversion fails (due to a `ValueError`, `TypeError`, or 
       `AttributeError`), the original key is retained in the dictionary.
    5. If `recheck` is `True`, it validates the converted quantiles by 
       ensuring they are in the range [0, 1].

    The function is designed to handle both direct float conversions (e.g., 
    '0.1') and percentage-based representations (e.g., '10%' becomes 0.1).

    Example
    -------
    >>> q_dict = {'0.1': ['subsidence_q10'], '50%': ['subsidence_q50'], 
                  '90%': ['subsidence_q90']}
    >>> validate_q_dict(q_dict)
    {0.1: ['subsidence_q10'], 0.5: ['subsidence_q50'], 0.9: ['subsidence_q90']}

    >>> q_dict = {'0.1': ['subsidence_q10'], '0.5': ['subsidence_q50'], 
                  '0.9': ['subsidence_q90']}
    >>> validate_q_dict(q_dict)
    {'0.1': ['subsidence_q10'], 0.5: ['subsidence_q50'], 'high': ['subsidence_q90']}

    >>> q_dict = {'0.1': ['subsidence_q10'], '200%': ['subsidence_q200']}
    >>> validate_q_dict(q_dict, recheck=True)
    {0.1: ['subsidence_q10'], 2.0: ['subsidence_q200']}

    See Also
    --------
   validate_quantiles`: 
       Validates if the values are valid quantiles in the range [0, 1].

    References
    ----------
    .. [1] Hyndman, R. J., & Fan, Y. (1996). Sample quantiles in 
           statistical packages. The American Statistician, 50(4), 361-365.
    .. [2] Weiss, N. A. (2015). Introductory Statistics. Pearson.
    """
    # Initialize an empty dictionary to store the converted quantiles
    new_q_cols = {}
    
    if not isinstance(q_dict, dict):
        raise TypeError(
            f"Expected a dictionary for `q_dict`, but "
            f"got {type(q_dict).__name__}. Ensure that `q_dict`"
            " is a dictionary where the keys represent quantiles "
            "(either as strings like '0.1' or '10%') and the values"
            " are lists of column names."
        )
    for key, value in q_dict.items():
        try:
            # Check if the key contains a percentage sign and convert it
            if "%" in str(key):
                key_float = float(key.replace('%', '')) / 100.0
            else:
                # Directly convert the key to float
                key_float = float(key)

            new_q_cols[key_float] = value

        except (AttributeError, ValueError, TypeError):
            # If conversion fails, retain the original key
            new_q_cols[key] = value

    if recheck:
        # Validate the quantiles after conversion 
        # (i.e., ensure keys are between 0 and 1)
        validate_quantiles(list(new_q_cols.keys()), dtype='float64')

    return new_q_cols

def validate_quantiles(
    quantiles, 
    asarray=False, 
    round_digits=2, 
    dtype=None, 
    mode="strict", 
    scale_method="uniform"
):
    r"""
    Validate and normalize quantile values with flexible conversion rules.

    Ensures quantile inputs are valid probabilities :math:`q \in [0,1]` while 
    providing mechanisms for automatic value adjustment through different 
    scaling strategies.

    .. math::

        Q_{\text{adj}} = \frac{q_{\text{raw}}}{10^{\lfloor \log_{10}(q_{\text{raw}}) \rfloor + 1}}

    where :math:`q_{\text{raw}}` is the input value requiring adjustment.

    Parameters
    ----------
    quantiles : array-like
        Input values to validate. Accepts:
        - Numeric values in [0,1]
        - Percentages (e.g., "20%")
        - Integers for automatic scaling in ``mode='soft'``
    asarray : bool, default=False
        Determines output format:
        - ``True``: Returns numpy array
        - ``False``: Returns Python list
    round_digits : int, default=2
        Number of decimal places for rounding to mitigate floating-point 
        precision issues
    dtype : str or numpy.dtype, default='float32'
        Output data type. Supported values: 'float32' (TF-compatible) or 
        'float64' (high precision)
    mode : {'strict', 'soft'}, default='strict'
        Validation strictness:
        - ``'strict'``: Rejects values outside [0,1]
        - ``'soft'``: Converts percentages and scales integers using 
           ``scale_method``
    scale_method : {'uniform', 'individual'}, default='uniform'
        Scaling strategy for ``mode='soft'``:
        - ``'uniform'``: Uses maximum digit count from all values for divisor
        - ``'individual'``: Scales each value independently

    Returns
    -------
    list or numpy.ndarray
        Validated quantiles in specified format. Return type matches 
        ``asarray`` parameter.

    Raises
    ------
    TypeError
        For non-numeric inputs in ``mode='strict'`` or invalid types in 
        ``mode='soft'``
    ValueError
        For values outside [0,1] in ``mode='strict'`` or invalid scaling 
        conversions

    Examples
    --------
    Basic validation:
    >>> from gofast.core.diagnose_q import validate_quantiles
    >>> validate_quantiles([0.1, 0.5, 0.9])
    [0.1, 0.5, 0.9]

    Soft mode with percentage conversion:
    >>> validate_quantiles(["20%", 5, 150], mode='soft')
    [0.2, 0.05, 0.15]

    Array output with custom precision:
    >>> validate_quantiles([0.123456, 0.789012], asarray=True, round_digits=3)
    array([0.123, 0.789], dtype=float32)

    Notes
    -----
    1. In ``mode='soft'``:
       - Percentages convert via :math:`\frac{\text{value}}{100}`
       - Integer scaling uses:
         - Uniform: :math:`\frac{\text{value}}{10^{\text{max\_digits}}}`
         - Individual: :math:`\frac{\text{value}}{10^{\text{self\_digits}}}`

    2. Rounding follows banker's rounding (numpy.round behavior) to minimize 
       cumulative errors [1]_.

    See Also
    --------
    gofast.stats.evaluate_quantiles : Evaluates quantile estimation accuracy
    numpy.quantile : Computes quantiles of array values

    References
    ----------
    .. [1] IEEE Standard for Floating-Point Arithmetic. IEEE Std 754-2019.

    .. [2] Hyndman, R.J. & Fan, Y. (1996). Sample Quantiles in Statistical 
           Packages. The American Statistician, 50(4), 361-365.
    """
    quantiles = to_iterable(quantiles, transform=True, flatten=True)
  
    if mode == "soft":
        quantiles = _process_soft_quantiles(
            quantiles, 
            scale_method=scale_method
        )

    if not isinstance(quantiles, (list, np.ndarray)):
        raise TypeError(
            "Quantiles must be list or numpy array. "
            f"Received {type(quantiles).__name__}."
        )

    dtype = _get_valid_dtype(dtype)
    quantiles_np = np.array(quantiles, dtype=dtype)
    
    _validate_quantile_values(quantiles_np)
    quantiles_np = np.round(quantiles_np, decimals=round_digits)
    
    return quantiles_np if asarray else quantiles_np.tolist()

def _process_soft_quantiles(quantiles, scale_method):
    """Process quantiles in soft mode with scaling adjustments."""
    scaled_values = []
    scale_candidates = []
    
    for q in quantiles:
        q_val, needs_scaling = _process_single_quantile(q)
        
        if needs_scaling:
            scale_candidates.append(q_val)
            scaled_values.append(None)
        else:
            scaled_values.append(q_val)

    if scale_candidates:
        scaled = _apply_scaling(
            scale_candidates, 
            scale_method=scale_method
        )
        scaled_values = _merge_scaled_values(scaled_values, scaled)

    return scaled_values

def _process_single_quantile(q):
    """Process individual quantile value for soft mode."""
    original = q
    q = _convert_string_quantile(q)
    
    if not isinstance(q, (int, float)):
        raise TypeError(
            f"Quantile {original} must be numeric. "
            f"Received {type(q).__name__}."
        )

    if q < 0:
        raise ValueError(f"Negative quantile value: {original}")

    if 0 <= q <= 1:
        return q, False

    if not np.isclose(q, int(q)):
        raise ValueError(
            f"Non-integer out-of-range quantile: {original}"
        )
        
    return int(q), True

def _convert_string_quantile(q):
    """Convert string quantiles to numeric values."""
    if isinstance(q, str):
        q = q.strip().rstrip('%')
        try:
            value = float(q)
            if '%' in q:
                value /= 100.0
            return value
        except ValueError:
            raise ValueError(
                f"Could not convert string quantile: {q}"
            ) from None
    return q

def _apply_scaling(scale_candidates, scale_method):
    """Apply scaling strategy to out-of-range quantiles."""
    if scale_method == "uniform":
        max_digits = max(len(str(q)) for q in scale_candidates)
        divisor = 10 ** max_digits
        return [q / divisor for q in scale_candidates]
    
    if scale_method == "individual":
        return [
            q / (10 ** len(str(q))) 
            for q in scale_candidates
        ]
    
    raise ValueError(
        f"Invalid scale_method: {scale_method}. "
        "Choose 'uniform' or 'individual'."
    )

def _merge_scaled_values(values, scaled):
    """Merge scaled values back into original quantile list."""
    result = []
    scale_idx = 0
    
    for val in values:
        if val is None:
            result.append(scaled[scale_idx])
            scale_idx += 1
        else:
            result.append(val)
            
    return result

def _get_valid_dtype(dtype):
    """Validate and return proper numpy dtype."""
    dtype_map = {
        "float32": np.float32, 
        "float64": np.float64
    }
    
    if dtype is None:
        return np.float32
    
    if isinstance(dtype, str) and dtype in dtype_map:
        return dtype_map[dtype]
    
    return dtype if dtype in (np.float32, np.float64) else np.float32

def _validate_quantile_values(quantiles_np):
    """Core validation for quantile value requirements."""
    if not np.issubdtype(quantiles_np.dtype, np.number):
        raise ValueError("All quantiles must be numeric.")
        
    if np.any((quantiles_np < 0) | (quantiles_np > 1)):
        raise ValueError(
            "Quantiles must be in [0, 1] range. "
            "Use 'soft' mode for automatic scaling."
        )

def validate_quantiles_in(
    quantiles, 
    asarray=False, 
    round_digits=1, 
    dtype=None, 
    mode="strict", 

    ):
    """
    Validates the input quantiles and optionally returns the output as a 
    numpy array or list, with an option to round the quantiles to a 
    specified number of decimal places to avoid floating-point precision 
    issues.

    Quantiles are numerical values used in statistical analysis to 
    divide a distribution into intervals. They must lie within the 
    range [0, 1] as they represent proportions of data [1]_.

    Parameters
    ----------
    quantiles : list or numpy.ndarray
        Input array-like containing quantile values to be validated. 
        The values must be numeric and within the range [0, 1].
        
    asarray : bool, optional
        Determines the output format. If `True`, the validated 
        quantiles are returned as a numpy array. If `False`, they 
        are returned as a list. Default is `False`.

    round_digits : int, optional, default=1
        The number of decimal places to which the quantiles should be 
        rounded. This helps avoid floating-point precision errors such as 
        `0.10000000149011612` being displayed as `0.1`. By default, 
        quantiles are rounded to 1 decimal place.
        
    dtype : numpy.dtype, optional, default=np.float32
        The data type for the quantiles array. Use `np.float32`
        for compatibility with TensorFlow or `np.float64` for higher 
        precision. The dtype determines the precision used for quantiles
        during validation and rounding.

    Returns
    -------
    list or numpy.ndarray
        A list or numpy array of validated quantile values, depending 
        on the value of `asarray`.

    Raises
    ------
    TypeError
        If the input `quantiles` is not a list or numpy array.

    ValueError
        If any element of `quantiles` is not numeric or lies outside 
        the range [0, 1].

    Notes
    -----
    Quantiles, denoted as :math:`q \in [0, 1]`, represent the fraction 
    of observations below a certain value in a distribution:
    
    .. math::

        Q(q) = \inf \{ x \in \mathbb{R} : P(X \leq x) \geq q \}

    where :math:`Q(q)` is the quantile function, and :math:`q` is the 
    proportion [2]_.

    This function ensures that all values in `quantiles` adhere to 
    this definition by checking:
    
    1. The type of `quantiles`.
    2. The numerical nature of its elements.
    3. The range of its values.
    4. The optional rounding of the quantiles to a specified number 
       of decimal places.

    Examples
    --------
    >>> from gofast.utils.diagnose_q import validate_quantiles_in
    >>> validate_quantiles([0.1, 0.2, 0.5])
    [0.1, 0.2, 0.5]

    >>> validate_quantiles(np.array([0.3, 0.7, 0.9]), asarray=True)
    array([0.3, 0.7, 0.9])

    >>> validate_quantiles([0.10000000149011612, 0.5, 0.8999999761581421], round_digits=1)
    [0.1, 0.5, 0.9]

    >>> validate_quantiles([0.5, 1.2])
    ValueError: All quantile values must be in the range [0, 1].

    See Also
    --------
    numpy.percentile : Computes the nth percentile of an array.
    numpy.quantile : Computes the qth quantile of an array.

    References
    ----------
    .. [1] Hyndman, R. J., & Fan, Y. (1996). Sample quantiles in 
           statistical packages. The American Statistician, 50(4), 361-365.
    .. [2] Weiss, N. A. (2015). Introductory Statistics. Pearson.

    """
    # Convert quantiles to a list if necessary
    quantiles = to_iterable(quantiles, transform=True, flatten=True)
  
    # Validate input type: must be list or numpy array
    if not isinstance(quantiles, (list, np.ndarray)):
        raise TypeError(
            "Quantiles must be a list or numpy array. Received "
            f"{type(quantiles).__name__!r}."
        )
    
    # Define a dictionary for mapping string dtype names to numpy float types
    dtypes = {"float32": np.float32, "float64": np.float64}

    # Check if dtype is a string, and convert
    # it to the corresponding numpy dtype
    if dtype is None: 
        dtype = 'float32'
    if isinstance(dtype, str):
        if dtype not in dtypes:
            raise ValueError(
                f"Unsupported dtype string: {dtype}."
                " Supported values are 'float32' or 'float64'."
        )
        # Convert string to corresponding numpy dtype
        dtype = dtypes[dtype]  

    # Convert input to numpy array for consistent 
    # processing using the specified dtype
    quantiles = np.array(quantiles, dtype=dtype)
    
    # Validate that all elements are numeric
    if not np.issubdtype(quantiles.dtype, np.number):
        raise ValueError("All quantile values must be numeric.")
    
    # Validate that all values are within the range [0, 1]
    if not np.all((quantiles >= 0) & (quantiles <= 1)):
        raise ValueError("All quantile values must be in the range [0, 1].")
    
    # Round quantiles to the specified number of decimal places
    quantiles = np.round(quantiles, decimals=round_digits)
    
    # Return quantiles in the desired format
    return quantiles if asarray else quantiles.tolist()

def detect_quantiles_in(
    df: pd.DataFrame,
    col_prefix: Optional[str] = None,
    dt_value: Optional[List[str]] = None,
    mode: str = 'soft',
    return_types: str = "columns",
    verbose: int = 0
) -> Union[List[str], List[float], List[np.ndarray], pd.DataFrame, None]:
    r"""
    Detect quantile columns in a DataFrame using naming patterns and value validation.

    Identifies columns containing quantile data through structured naming 
    conventions and value validation. Supports both absolute and normalized 
    quantile representations through mode-based value adjustment.


    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing potential quantile columns. Column names 
        must be strings.
    col_prefix : str, optional
        Column name prefix for targeted search (e.g., ``'price'`` for 
        ``price_q0.25``). If None, scans all columns.
    dt_value : list of str, optional
        Date filters for temporal quantile detection (e.g., ``['2023']`` matches 
        columns like ``price_2023_q0.5``).
    mode : {'soft', 'strict'}, default='soft'
        Value handling strategy:
        - ``'soft'``: Normalizes values >1 to 1.0 using min-max scaling
        - ``'strict'``: Excludes values outside [0,1] range
    return_types : {'columns', 'q_val', 'values', 'frame'}, default='columns'
        Return format specification:
        - ``'columns'``: List of column names
        - ``'q_val'``: Sorted unique quantile values
        - ``'values'``: Column data arrays
        - ``'frame'``: DataFrame subset
    verbose : {0, 1, 2, 3}, default=0
        Output verbosity:
        - 0: Silent
        - 1: Basic scan info
        - 2: Per-column matches
        - 3: Full diagnostic output

    Returns
    -------
    Union[List[str], List[float], List[np.ndarray], pd.DataFrame, None]
        Quantile data in format specified by ``return_types``. Returns None if 
        no quantiles detected.

    Notes 
    --------
    The detection adjustment can be formulated as : 
        
    .. math::

        q_{\text{adj}} = \begin{cases}
        \min(1, \max(0, q_{\text{raw}})) & \text{if } mode=\text{'soft'} \\
        q_{\text{raw}} & \text{if } q \in [0,1] \text{ and } mode=\text{'strict'}
        \end{cases}
            
    Examples
    --------
    >>> from gofast.core.diagnose_q import detect_quantiles_in
    >>> import pandas as pd
    
    # Basic detection
    >>> df = pd.DataFrame({'sales_q0.25': [4.2], 'sales_q0.75': [5.8]})
    >>> detect_quantiles_in(df, col_prefix='sales')
    ['sales_q0.25', 'sales_q0.75']

    # Temporal quantile filtering
    >>> df = pd.DataFrame({'temp_2023_q0.5': [22.1], 'temp_2024_q0.5': [23.4]})
    >>> detect_quantiles_in(df, dt_value=['2023'], return_types='q_val')
    [0.5]

    # Value normalization
    >>> df = pd.DataFrame({'risk_q150': [0.8]})
    >>> detect_quantiles_in(df, mode='soft', return_types='q_val')
    [1.0]

    Notes
    -----
    1. Column name pattern requirements:
       - Requires ``_qX`` suffix where X is numeric
       - Temporal format: ``{prefix}_{date}_q{value}``
       - Non-temporal format: ``{prefix}_q{value}``

    2. Value adjustment in soft mode uses piecewise function:
       - Clips values to [0,1] range
       - Preserves original values within valid range

    See Also
    --------
    gofast.utils.validate_quantiles : For quantile value validation
    pandas.DataFrame.filter : For column selection by pattern

    References
    ----------
    .. [1] Regular Expression HOWTO, Python Documentation
    .. [2] Pandas API Reference: DataFrame operations
    """

    df = to_frame_if(df).copy() 
    df.columns = df.columns.astype(str)
    col_prefix = col_prefix or ''
    
    quantile_columns = []
    found_quantiles = set()
    
    _log_verbose(f"Scanning DataFrame columns with prefix: {col_prefix}",
                 verbose, 1
                )
    
    for col in df.columns:
        result = _process_column(
            col, df, col_prefix, dt_value, mode, verbose
        )
        if result:
            q_val, col_data = result
            found_quantiles.add(q_val)
            _store_results(q_val, col_data, quantile_columns,
                           return_types, col)

    _log_verbose(f"Quantiles detected: {sorted(found_quantiles)}", verbose, 3)
    
    return _format_output(quantile_columns, found_quantiles, return_types, df)

def _process_column(
    col: str,
    df: pd.DataFrame,
    prefix: str,
    dt_values: Optional[List[str]],
    mode: str,
    verbose: int
) -> Optional[tuple]:
    """Process individual column for quantile detection."""
    # Handle both cases: with or without prefix
    if prefix:
        # Remove the prefix part
        if col.startswith(f"{prefix}_"):
            col_match = col[len(prefix)+1:]  # Remove prefix and underscore
        else:
            return None  # If the column does not start with the prefix, skip it
    else:
        col_match = col  # No prefix, use the column name directly
    
    match, q_str = _check_column_match(col_match, dt_values)
    if not match:
        return None

    try:
        q_val = _extract_quantile_value(q_str, mode)
    except ValueError as e:
        _log_verbose(f"Invalid quantile value in {col}: {e}", verbose, 2)
        return None

    _log_verbose(f"Found quantile match: {col} with value: {q_val}", verbose, 2)
    return q_val, df[col].values

def _check_column_match(
    remainder: str,
    dt_values: Optional[List[str]],
    prefix: Optional[str] = None
) -> tuple:
    """Check if column remainder matches date and quantile patterns."""
    # Match quantile pattern like q0.25
    quantile_pattern = re.compile(r'q([\d\.]+)$')  
    # Case 1: If a date filter is provided, look for 
    # columns matching the date and quantile patterns
    if dt_values:
        for d_str in dt_values:
            date_pattern = f"^{d_str}_q"  # Include date check
            # Match column with date filter
            if remainder.startswith(date_pattern):  
                m = quantile_pattern.search(remainder)
                return (True, m.group(1)) if m else (False, None)
    
    # Case 2: If no date filter is provided,
    # look for quantile matches
    else:
        # Check for the quantile pattern in the remainder
        m = quantile_pattern.search(remainder)
        if m:
            # Return the quantile value (e.g., 0.25)
            return (True, m.group(1))  
    
    return (False, None)


def _extract_quantile_value(q_str: str, mode: str) -> float:
    """Extract and validate quantile value with proper error handling."""
    try:
        q_val = float(q_str)
    except ValueError:
        raise ValueError(f"Invalid quantile format: {q_str}")
    
    # Use centralized validation from validate_quantiles
    validated = validate_quantiles(
        [q_val], mode=mode, scale_method='individual', 
        round_digits =2, 
        dtype= np.float64, 
    )
    return validated[0]

def _store_results(
    q_val: float,
    col_data: np.ndarray,
    quantile_columns: list,
    return_types: str, 
    col: str, 
) -> None:
    """Store results based on requested return type."""
    if return_types == 'values':
        quantile_columns.append(col_data)
    elif return_types == 'q_val':
        quantile_columns.append(q_val)
    else:
        quantile_columns.append(col)

def _format_output(
    quantile_columns: list,
    found_quantiles: set,
    return_types: str,
    df: pd.DataFrame
) -> Union[List, pd.DataFrame, None]:
    """Format final output based on return_types."""
    if not quantile_columns:
        return None

    if return_types == 'frame':
        return df[quantile_columns]
    if return_types == 'q_val':
        return sorted(found_quantiles)
    if return_types == 'values':
        return np.vstack(quantile_columns) if quantile_columns else []
    return sorted(quantile_columns)

def _log_verbose(message: str, verbose_level: int, required_level: int) -> None:
    """Centralized verbose logging control."""
    if verbose_level >= required_level:
        print(message)
        
def build_q_column_names(
    df: pd.DataFrame,
    quantiles: List[Union[float, str]],
    value_prefix: Optional[str] = None,
    dt_value: Optional[List[Union[str, int]]] = None,
    strict_match: bool = True
) -> List[str]:
    r"""
    Generate and validate quantile column names following naming conventions.

    Parameters
    ----------
    df : pd.DataFrame
        Target DataFrame containing potential quantile columns
    quantiles : list of float/str
        Quantile values to search for (0 < q < 1). Accepts:
        - Float values (e.g., 0.25)
        - String representations (e.g., "25%")
    value_prefix : str, optional
        Column name prefix for structured naming. If None, 
        looks for unprefixed columns.
    dt_value : list of str/int, optional
        Temporal identifiers for time-aware quantiles. Converts 
        all values to strings.
        
    strict_match : bool, default=True
        Matching strategy:
        - ``True``: Requires exact column name matches
        - ``False``: Uses regex pattern matching for flexible detection

    Returns
    -------
    list
        Valid column names found in the DataFrame matching the 
        quantile naming pattern.

    Notes 
    -------
    Constructs column names using the pattern:
    
    .. math::
        \text{col_name} = \begin{cases}
        \text{value_prefix}\_\text{date}\_q\text{quantile} & \text{if both prefix and date exist} \\
        \text{value_prefix}\_q\text{quantile} & \text{if only prefix exists} \\
        \text{date}\_q\text{quantile} & \text{if only date exists} \\
        q\text{quantile} & \text{otherwise}
        \end{cases}
        
    Examples
    --------
    >>> from gofast.core.diagnose_q import build_q_column_names
    >>> import pandas as pd
    
    # Basic usage with prefix
    >>> df = pd.DataFrame(columns=['price_q0.25', 'price_2023_q0.5'])
    >>> build_q_column_names(df, [0.25, 0.5], 'price')
    # if strict_match ts
    ['price_q0.25', 'price_2023_q0.5']

    # Date-filtered search
    >>> build_q_column_names(df, [0.5], 'price', dt_value=['2023'])
    ['price_2023_q0.5']

    # Unprefixed columns
    >>> df = pd.DataFrame(columns=['q0.75', '2024_q0.9'])
    >>> build_q_column_names(df, [0.75, 0.9])
    ['q0.75', '2024_q0.9']

    See Also
    --------
    gofast.core.diagnose_q.validate_quantiles : For quantile value validation
    pandas.Series.str.contains : For column pattern matching
    """
    df = to_frame_if (df)
    # Validate and normalize inputs
    valid_quantiles = validate_quantiles(
        quantiles, mode='soft', round_digits=2, 
        dtype='float64', 
    )
    date_strings = _process_dt_values(dt_value)
    df.columns = df.columns.astype(str)

    if strict_match:
        candidates = _generate_strict_candidates(
            valid_quantiles, value_prefix, date_strings
        )
        return [col for col in candidates if col in df.columns]

    # Flexible pattern matching
    pattern = _build_flexible_pattern(
        valid_quantiles, value_prefix, date_strings
    )
    return [col for col in df.columns if pattern.search(col)]

def _generate_strict_candidates(
    quantiles: List[float],
    prefix: Optional[str],
    dates: List[str]
) -> List[str]:
    """Generate exact match candidates in all valid formats."""
    candidates = []
    for q in quantiles:
        # Decimal format (q0.25)
        dec_str = f"q{q:.4f}".rstrip('0').rstrip('.')
        # Percentage format (q25)
        pct_str = f"q{int(round(q * 100))}"
        
        for fmt in [dec_str, pct_str]:
            # Temporal candidates
            if dates:
                candidates.extend(
                    f"{prefix}_{d}_{fmt}" if prefix else f"{d}_{fmt}"
                    for d in dates
                )
            # Non-temporal candidates
            candidates.append(
                f"{prefix}_{fmt}" if prefix else fmt
            )
    return list(set(candidates))  # Remove duplicates

def _build_flexible_pattern(
    quantiles: List[float],
    prefix: Optional[str],
    dates: List[str]
) -> re.Pattern:
    """Build regex pattern for flexible quantile matching."""
    # Quantile alternatives (0.25|25)
    q_alternatives = '|'.join(
        f"{q:.4f}".rstrip('0').rstrip('.') + '|' + str(int(round(q * 100)))
        for q in quantiles
    )
    
    # Prefix component
    prefix_part = f"{re.escape(prefix)}_?" if prefix else ''
    
    # Date component
    date_part = (
        f"({'|'.join(map(re.escape, dates))})_+" 
        if dates 
        else r'\d{4}_?|'
    )
    
    return re.compile(
        rf"^{prefix_part}(?:{date_part})?q({q_alternatives})\b",
        flags=re.IGNORECASE
    )

def _process_dt_values(
    dt_values: Optional[List[Union[str, int]]]
) -> List[str]:
    """Normalize temporal values to standardized strings."""
    return (
        [str(v).strip() for v in dt_values] 
        if dt_values 
        else []
    )


    

# # let assume we have a dataframe df with columns below: 
    
#     lon  # spatial columns longitude 
#     lat   # spatial column latitude  
#     <value_prefix>_<dt_value1>_q<qval1> 
#     <value_prefix>_<dt_value1>_q<qval2> 
#     <value_prefix>_<dt_value1>_q<qval3> 
#     <value_prefix>_<dt_value2>_q<qval1> 
#     <value_prefix>_<dt_value2>_q<qval2> 
#     <value_prefix>_<dt_value2>_q<qval3> 
#     <value_prefix>_<dt_value3>_q<qval1> 
#     <value_prefix>_<dt_value3>_q<qval2> 
#     <value_prefix>_<dt_value3>_q<qval3> 
#     .... 
    
# # we want to generate the expected result as long dataframe with the 
# # columns 
# lon  # spatial column longitude, reapeated at each dt_col 
# lat   # spatial_col latitude repeated at each dt_col 
# dt_col # here the value should be <dt_value1> , <dt_value2> , <dt_value3 > and ... 
# <value_prefix>_q<qval1> 
# <value_prefix>_q<qval2>
# <value_prefix>_q<qval3>


# # where value_prefix is the string name  
# # qval is the quantile value for instance qval1 can be =0.1 , qval2=0.5 and qval3 =0.9 

# # if the q parameter is explicitely passed for instance 
# q = [qval1, qval2] 
# # the dataframe returned will have the columns that contain only the qval1 and qval2 
# i.e 
# lon  # spatial column longitude, reapeated at each dt_col 
# lat   # spatial_col latitude repeated at each dt_col 
# dt_col # here the value should be <dt_value1> , <dt_value2> , <dt_value3 > and ... 
# <value_prefix>_q<qval1> 
# <value_prefix>_q<qval2>

# # here the <value_prefix>_q<qval3> is excluded. 

# # if the spatial_cols is not explictly passed the ignore this columns in the dataframe 
# # and contruct the long dataframe instead. 

# # the function should be called : to_long_data_q 
# # to_long_data_q(df, value_prefix , dt_name='dt_col', q=None, error="raise",  verbose=0)
# # if datetime column found in the dataframe columns and then 
# # the output column name with date.time column should be name as 'dt_col', using the dt_name. 
# # if error is raise and dt_value not found in the dataframe , then raise error 
# # warn if error='warn' and return empty dataframe or  ... 
# # verbose should go from 0 to 5 to track bug and issue in the code, to 
# # help debugging. 



    
    
    
    
