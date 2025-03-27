# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Provides additional utilities and tools that extend 
the core capabilities of the gofast package.
"""
from __future__ import annotations 

import re
import itertools
import copy
import warnings 
from typing import Any, Dict, List, Union, Optional, Tuple 
import numpy as np
import pandas as pd

from ..api.types import ArrayLike, DataFrame
from ..compat.pandas import select_dtypes 
from ..compat.sklearn import validate_params
from ..core.array_manager import to_series 
from ..core.checks import assert_ratio, check_numeric_dtype
from ..core.checks import check_params, exist_features
from ..core.generic import vlog 
from ..core.handlers import columns_getter, columns_manager 
from ..core.io import SaveFile, is_data_readable
from ..core.utils import error_policy, smart_format 
from ..decorators import isdf, Dataify 
from .validator import parameter_validator, validate_length_range 
from .validator import  get_estimator_name 


@Dataify(enforce_df=True,)
def compute_pairwise_errors(
    df,
    columns=None,
    method='abs',
    verbose=0,
    pair_strategy='consecutive',
    keep_origin=True,
    epsilon=1e-6,
    custom_error=None,
    error_prefix='error_',
    dropna=False,
    sort_columns=False,
    **kwargs
):
    r"""
    Computes pairwise errors between specified columns in a
    DataFrame. This method `compute_pairwise_errors` can
    operate using different error computation techniques,
    including absolute error, relative error, or squared
    error. Users can also supply a custom error function
    via ``custom_error``.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing numeric data.
        Pairwise errors are computed on the columns
        of this DataFrame.

    columns : list of str, optional
        The list of column names on which to compute
        pairwise errors. If `columns` is None, all
        numeric columns of `df` are used.

    method : {'abs', 'rel', 'squared'}, default='abs'
        The error computation method. Supported
        methods are ``abs`` for absolute error,
        ``rel`` for relative error, and ``squared``
        for squared error.

    verbose : int, default=0
        The verbosity level of log messages:
        * 0 : No messages.
        * 1 : Basic messages (e.g., selected columns,
          dropped rows).
        * 2 : More detailed messages, including pair
          generation.
        * 3 : Most verbose messages (reserved for
          potential future enhancements).

    pair_strategy : {'consecutive', 'all'}, default='consecutive'
        Determines how column pairs are generated:
        * ``consecutive`` : Generates pairs of
          consecutive columns (e.g., (col1,col2),
          (col2,col3), ...).
        * ``all`` : Generates all unique pairs of the
          selected columns.

    keep_origin : bool, default=True
        If True, retains the original columns in the
        output DataFrame alongside the newly computed
        error columns. If False, only the error columns
        remain.

    epsilon : float, default=1e-6
        A small constant to avoid division by zero in the
        relative error calculation.

    custom_error : callable, optional
        A custom function for computing error between
        two columns. It must accept two pandas Series as
        inputs and return a Series of the same shape,
        representing the error. If provided, this
        overrides the built-in methods for error
        computation.

    error_prefix : str, default='error_'
        A prefix for the names of newly created error
        columns.

    dropna : bool, default=False
        If True, rows containing NaN values in the
        selected columns are dropped before error
        computation. If False, all rows are retained.

    sort_columns : bool, default=False
        If True, sorts the selected columns
        lexicographically before generating pairs.
        If False, preserves their current order.

    **kwargs
        Additional keyword arguments for future
        extensibility.

    Returns
    -------
    pd.DataFrame
        A new DataFrame that contains the computed
        pairwise error columns. If `keep_origin`
        is True, both the original and the error
        columns are present; otherwise, only the
        error columns remain.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.utils.ext import compute_pairwise_errors

    >>> df = pd.DataFrame({
    ...     'A': [1.0, 2.0, 3.0],
    ...     'B': [1.5, 2.5, 2.9],
    ...     'C': [2.0, 2.0, 2.0]
    ... })
    >>> # Compute absolute errors for consecutive pairs: (A,B) and (B,C).
    >>> result = compute_pairwise_errors(df, 
    ...                                  columns=['A','B','C'],
    ...                                  method='abs',
    ...                                  pair_strategy='consecutive')
    >>> result

    >>> # Compute relative errors for all pairs and drop original columns.
    >>> result_rel = compute_pairwise_errors(df,
    ...                                      columns=['A','B','C'],
    ...                                      method='rel',
    ...                                      pair_strategy='all',
    ...                                      keep_origin=False)

    Notes
    -----
    The `compute_pairwise_errors` method is especially
    useful when you need to quickly evaluate the
    differences between adjacent (or all) columns in a
    time series, or in a dataset containing related
    measurements. By default, it applies an absolute
    difference, but you can switch to relative or squared
    differences, or provide your own computation by
    setting ``custom_error``.
    
    .. math::
        \begin{aligned}
        &\text{Absolute Error (abs): } 
        E_{\text{abs}}(x, y) = |\,x - y\,|,\\[8pt]
        &\text{Relative Error (rel): }
        E_{\text{rel}}(x, y) 
            = \frac{x - y}{y + \epsilon},\\[8pt]
        &\text{Squared Error (squared): }
        E_{\text{squared}}(x, y) 
            = (x - y)^2,
        \end{aligned}

    where :math:`\epsilon` is a small constant to avoid
    division by zero issues for relative errors.

    See Also
    --------
    pandas.DataFrame.dropna : Method to drop rows with
        missing values.
    pandas.DataFrame.select_dtypes : Select columns
        by dtype.

    References
    ----------
    .. [1] J. D. Hunter et al., *Computational Tools for
       Data Analysis*, 2018.

    """
    # Create a copy of the DataFrame to avoid mutating
    # the original data.
    df_copy = df.copy()

    # Determine columns to process.
    if columns is None:
        columns = df_copy.select_dtypes(
            include='number'
        ).columns.tolist()
        if verbose >= 1:
            print(f"Selected columns: {columns}")
        if not columns:
            raise ValueError("No numeric columns found "
                             "in the DataFrame.")
    else:
        missing = [
            col for col in columns
            if col not in df_copy.columns
        ]
        if missing:
            raise ValueError(f"Columns not found: {missing}")

    # Check for non-numeric columns among selected.
    non_numeric = df_copy[columns].select_dtypes(
        exclude='number'
    ).columns.tolist()
    if non_numeric:
        raise ValueError(f"Non-numeric columns "
                         f"selected: {non_numeric}")

    # Validate custom_error if provided.
    if custom_error is not None \
       and not callable(custom_error):
        raise TypeError("custom_error must be "
                        "a callable function.")

    # Sort columns if requested.
    if sort_columns:
        columns = sorted(columns)
        if verbose >= 1:
            print(f"Sorted columns: {columns}")

    # Drop NA rows if requested.
    if dropna:
        initial_rows = len(df_copy)
        df_copy = df_copy.dropna(subset=columns).copy()
        if verbose >= 1:
            dropped = initial_rows - len(df_copy)
            print(f"Dropped {dropped} rows with NaN in "
                  f"selected columns. Remaining rows: "
                  f"{len(df_copy)}")

    # Validate pair strategy.
    valid_strategies = ['consecutive', 'all']
    if pair_strategy not in valid_strategies:
        raise ValueError(
            f"pair_strategy must be one of "
            f"{valid_strategies}"
        )

    # Generate column pairs.
    if pair_strategy == 'consecutive':
        pairs = list(zip(
            columns[:-1],
            columns[1:]
        ))
        if verbose >= 2:
            print(f"Generated consecutive pairs: {pairs}")
    else:
        pairs = list(
            itertools.combinations(columns, 2)
        )
        if verbose >= 2:
            print(f"Generated all pairs: {pairs}")

    # Compute errors for each pair.
    for col1, col2 in pairs:
        error_col = f"{error_prefix}{col1}_{col2}"
        if verbose >= 1:
            print(f"Computing error between {col1} "
                  f"and {col2} as {error_col}")
        try:
            if custom_error is not None:
                # Use custom function for error.
                df_copy[error_col] = custom_error(
                    df_copy[col1],
                    df_copy[col2]
                )
            else:
                # Built-in error computations.
                if method == 'abs':
                    df_copy[error_col] = (
                        df_copy[col1]
                        - df_copy[col2]
                    ).abs()
                elif method == 'rel':
                    denominator = df_copy[col2].copy()
                    # Avoid zero division by adjusting
                    # values < epsilon.
                    signs = np.sign(denominator).replace(0, 1)
                    mask = denominator.abs() < epsilon
                    denominator[mask] = (
                        signs[mask] * epsilon
                    )
                    df_copy[error_col] = (
                        df_copy[col1]
                        - df_copy[col2]
                    ) / denominator
                    if verbose >= 2:
                        zero_count = mask.sum()
                        if zero_count > 0:
                            print(f"Adjusted "
                                  f"{zero_count} near-zero "
                                  f"denominators in column "
                                  f"'{col2}' using "
                                  f"epsilon={epsilon}")
                elif method == 'squared':
                    df_copy[error_col] = (
                        df_copy[col1]
                        - df_copy[col2]
                    ) ** 2
                else:
                    raise ValueError(
                        f"Unsupported method: {method}"
                    )
        except Exception as e:
            if verbose >= 1:
                print(f"Error computing {error_col}: {e}")
            raise

    # Drop original columns if requested.
    if not keep_origin:
        if verbose >= 1:
            print(f"Dropping original columns: {columns}")
        df_copy.drop(
            columns=columns,
            inplace=True
        )

    return df_copy

@Dataify(enforce_df=False)
def normalize_categorical_column(
    df: Union[pd.DataFrame, pd.Series],
    feature: Optional[str] = None,      
    mapping: Optional[Union[Dict[Any, float], List[Any]]] = None,
    method: str = 'minmax',
    new_name: Optional[str] = None,      
    target_range: Tuple[float, float] = (0.0, 1.0),
    fill_missing: Optional[Any] = None,
    drop_origin: bool = False,           
    smart_distrib: bool = False,         
    spread_factor: float = 0.1,                
    verbose: int = 0                     
) -> Union[pd.DataFrame, pd.Series]:
    """
    Normalize a categorical column from a pandas DataFrame or Series
    by mapping its categories to numeric codes, then applying a
    Min-Max scaling. 
    
    Parameters
    ----------
    df : pandas.DataFrame or pandas.Series
        A DataFrame or Series containing the categorical column
        to normalize. If a DataFrame, <parameter inline>feature`
        must be specified.
    feature : str, optional
        The name of the column to normalize when ``df`` is a
        DataFrame. If ``df`` is a Series, this parameter is
        not required.
    mapping : dict or list, optional
        A dictionary or list defining how each category
        is mapped to numeric codes. If None, a default
        dictionary is created by enumerating unique sorted
        categories. If a list is provided, it is converted
        internally to a dictionary.
    method : str, optional
        Normalization method, currently supports only
        ``"minmax"``. Defaults to ``"minmax"``.
    new_name : str, optional
        Name of the new normalized column. If None, the
        normalized column will have ``_norm`` appended
        to the original column name.
    target_range : tuple of float, optional
        The range for the scaled values, defaulting to
        ``(0.0, 1.0)``.
    fill_missing : any, optional
        Value used to fill missing entries in the numeric
        codes before normalization. If None, missing
        values remain as is.
    drop_origin : bool, optional
        If True, drops the original column from the DataFrame
        after creating the normalized column. Defaults to
        False.
    smart_distrib : bool, optional
        If True, applies a "smart distribution" to spread
        out the highest category values by randomly sampling
        from an interval. Defaults to False.
    spread_factor : float, optional
        The amount of extension for the highest category
        interval under "smart distribution". Defaults to
        0.1.
    verbose : int, optional
        Level of verbosity. 0 = silent, up to 5 for debug
        output. Defaults to 0.
    
    Returns
    -------
    pandas.DataFrame or pandas.Series
        The DataFrame or Series with the normalized numeric
        column. If the input is a DataFrame, a new column is
        added. If a Series was provided, a Series of normalized
        values is returned.
    
    Notes
    -----
    - If ``method="minmax"`` and all mapped values are equal,
      they are replaced by the midpoint of ``target_range``.
    - When <parameter inline>smart_distrib` is True, the top
      category is extended by ``spread_factor`` for random
      sampling, ensuring a spread in the highest values.
    
    The function distributes the highest the category more 
    widely through a "smart distribution" mechanism.
    
    .. math::
       x_{\\text{norm}} = 
       \\frac{x - x_{\\min}}{x_{\\max} - x_{\\min}} 
       \\times 
       (r_{\\max} - r_{\\min}) 
       + r_{\\min}
    
    Here, :math:`x` is the mapped numeric code, 
    :math:`x_{\\min}` is its minimum, and :math:`x_{\\max}` is 
    its maximum. The target range 
    :math:`[r_{\\min}, r_{\\max}]` is defined by 
    ``target_range``.
    
    Examples
    --------
    >>> from gofast.utils.ext import normalize_categorical_column
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "city": ["A", "B", "C", "A", "B", "B", None],
    ...     "val": [10, 20, 15, 12, 30, 25, 18]
    ... })
    >>> # Normalize 'city' column with default min-max and mapping
    >>> df_norm = normalize_categorical_column(
    ...     df,
    ...     feature="city",
    ...     fill_missing=13,
    ...     verbose=3
    ... )
    >>> print(df_norm)
    
    See Also
    --------
    columns_manager : Ensures the specified column(s) in a
        DataFrame exist and handles potential name list input.
    exist_features : Checks whether the required feature
        columns are present in the DataFrame.
    
    References
    ----------
    .. [1] Smith, J., & Doe, A. (2020). Advanced Techniques in
           Data Preprocessing. Data Science Review, 15(4),
           250-269.
    """
    # Determine the Series to work on.
    if isinstance(df, pd.DataFrame):
        if feature is None:
            raise ValueError(
                "When passing a DataFrame, the 'feature'"
                " parameter must be provided"
            )
        # Ensure feature exists and get the first column name if list provided.
        feature = columns_manager(feature, empty_as_none=False)[0]
        exist_features(df, features=feature)
        series = df[feature]
    elif isinstance(df, pd.Series):
        series = df
    else:
        raise TypeError("Input must be a pandas DataFrame or Series")
    
    # Create mapping if not provided.
    if mapping is None:
        unique_vals = sorted(series.dropna().unique())
        mapping = {cat: i for i, cat in enumerate(unique_vals)}
        if verbose >= 3:
            print("Using default mapping:", mapping)
    elif isinstance(mapping, list):
        mapping = {cat: i for i, cat in enumerate(mapping)}
        if verbose >= 3:
            print("Converted mapping list to dict:", mapping)
    elif isinstance(mapping, dict):
        if verbose >= 3:
            print("Using provided mapping dictionary")
    else:
        raise ValueError("Parameter 'mapping' must be a dict, list, or None")
    
    # Map categories to numeric codes.
    numeric_series = series.map(mapping)
    numeric_series = pd.to_numeric(numeric_series, errors='coerce')

    # Fill missing values if requested.
    if fill_missing is not None:
        numeric_series = numeric_series.fillna(fill_missing)
        if verbose >= 3:
            print("Filled missing values with:", fill_missing)
    
    # Normalize using minmax scaling.
    if method == 'minmax':
        min_val = numeric_series.min(skipna=True)
        max_val = numeric_series.max(skipna=True)
        if min_val == max_val:
            # Avoid division by zero; assign midpoint.
            mid_val = (target_range[0] + target_range[1]) / 2.0
            normalized = numeric_series.apply(
                lambda x: mid_val if pd.notnull(x) else x
            )
            if verbose >= 2:
                print("All values equal; using midpoint:", mid_val)
        else:
            normalized = numeric_series.apply(
                lambda x: (
                    ((x - min_val) / (max_val - min_val)) *
                    (target_range[1] - target_range[0]) + target_range[0]
                ) if pd.notnull(x) else x
            )
            if verbose >= 2:
                print("Normalized using minmax scaling")
    else:
        raise ValueError("Only 'minmax' normalization is supported")
    
    # Apply smart distribution if enabled.
    if smart_distrib:
        # Get sorted unique base values from normalized series.
        unique_bases = sorted(normalized.dropna().unique())
        intervals = {}
        for i, base in enumerate(unique_bases):
            if i < len(unique_bases) - 1:
                next_base = unique_bases[i + 1]
                # Interval from current base to next base.
                intervals[base] = (base, next_base)
            else:
                # For the highest category, extend interval using 'spread'.
                intervals[base] = (base, base + spread_factor)
        if verbose >= 2:
            print("Smart intervals:", intervals)
        # Replace each value with a random draw from its interval.
        normalized = normalized.apply(
            lambda x: np.random.uniform(
                low=intervals[x][0], high=intervals[x][1]
            ) if pd.notnull(x) else x
        )
    
    # If input was a DataFrame, assign new column.
    if isinstance(df, pd.DataFrame):
        col_name = new_name if new_name is not None else feature + "_norm"
        df[col_name] = normalized
        if drop_origin:
            df.drop(columns=[feature], inplace=True)
            if verbose >= 2:
                print("Dropped original column:", feature)
        return df
    else:
        return normalized

@check_params ({ 
    "expr": Union[str, List[str]], 
    "op_cols": Optional[Union [str, List[str]]]
 })
@isdf 
def evaluate_df(
    df,
    expr,
    op_cols=None,       
    local_dict=None,    
    global_dict=None,   
    drop_cols=False,    
    engine='python',
    inplace=False,      
    error='raise',      
    **eval_kw
):
    """
    Evaluate one or multiple expressions on a pandas DataFrame
    and store the results in new or existing columns.

    This public function ``evaluate_df`` leverages
    :meth:`pandas.DataFrame.eval` to dynamically compute
    expressions and assign the results back into the DataFrame.
    Given an input :math:`expr`, the function computes:

    .. math::
       \\text{result} = \\text{df.eval}(expr, \\dots)

    The outcome can be in-place or returned as a new DataFrame,
    depending on ``inplace``. Additional dictionaries can be
    passed via ``local_dict`` or ``global_dict`` for variable
    references in expressions [1]_. See notes for best practices.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame on which expressions will be
        evaluated.
    expr : str or list of str
        One or multiple expressions to evaluate. If a list is
        provided, each expression is computed sequentially.
    op_cols : str or list of str, optional
        Names of the columns to store the evaluated results. If
        multiple expressions are given and no ``op_cols`` are
        provided, default column names such as ``eval_0`` will be
        used. If a single expression is supplied without
        ``op_cols``, defaults to ``eval.name``.
    local_dict : dict, optional
        A dictionary of local variables available in the
        expression engine. These override variables in
        ``global_dict``.
    global_dict : dict, optional
        A dictionary of global variables available in the
        expression engine.
    drop_cols : bool or list of str, optional
        If ``True`` or ``original``, any columns used in the
        expressions are dropped after evaluation. If a list of
        column names is provided, those columns are specifically
        dropped. Defaults to ``False``.
    engine : {'python', 'numexpr'}, optional
        Parser engine to use for expression evaluation. Defaults
        to ``'python'``.
    inplace : bool, optional
        If ``True``, modifies the original DataFrame directly.
        Otherwise, it returns a copy with the evaluated results.
        Defaults to ``False``.
    error : {'raise', 'warn', 'ignore'}, optional
        Error-handling strategy. ``'raise'`` (default) raises
        exceptions. ``'warn'`` emits a warning. ``'ignore'``
        silently skips any errors.
    **eval_kw
        Additional keyword arguments passed on to
        :meth:`pandas.DataFrame.eval`.

    Returns
    -------
    pandas.DataFrame
        The modified DataFrame containing the evaluated results.
        If ``inplace`` is ``False``, this is a new DataFrame.

    Examples
    --------
    >>> from gofast.utils.ext import evaluate_df
    >>> import pandas as pd
    >>> data = {
    ...     "colA": [10, 20, 30, 40],
    ...     "colB": [1, 2, 3, 4],
    ...     "colC": [5, 6, 7, 8]
    ... }
    >>> df = pd.DataFrame(data)

    1. Simple usage: create a new column as the sum of ``colA``
       and ``colB``.

    >>> df_result_1 = evaluate_df(
    ...     df=df,
    ...     expr="colA + colB",
    ...     op_cols="sumAB"
    ... )
    >>> df_result_1
       colA  colB  colC  sumAB
    0    10     1     5     11
    1    20     2     6     22
    2    30     3     7     33
    3    40     4     8     44

    2. Overwrite an existing column (modify in place) by
       multiplying ``colA`` and ``colC``:

    >>> evaluate_df(
    ...     df=df,
    ...     expr="colA * colC",
    ...     op_cols="colB",
    ...     inplace=True
    ... )
    >>> df
       colA  colB  colC
    0    10    50     5
    1    20   120     6
    2    30   210     7
    3    40   320     8

    3. Use ``local_dict`` to define a variable ``factor`` and
       multiply ``colA`` by this factor:
           
    >>> df_result_3 = evaluate_df(
    ...     df=df,
    ...     expr="colA * factor",
    ...     op_cols="scaledA",
    ...     local_dict={"factor": 100},
    ...     inplace=False
    ... )
    >>> df_result_3
       colA  colB  colC  scaledA
    0    10    50     5     1000
    1    20   120     6     2000
    2    30   210     7     3000
    3    40   320     8     4000

    4. A more complex expression referencing multiple columns and
       a constant from ``global_dict``:

    >>> MY_CONSTANT = 2.5
    >>> df_result_4 = evaluate_df(
    ...     df=df,
    ...     expr="((colA + colC) * MY_CONSTANT) / colB",
    ...     op_cols="complexCalc",
    ...     global_dict=globals(),
    ...     inplace=False
    ... )
    >>> df_result_4
       colA  colB  colC  complexCalc
    0    10    50     5     0.750000
    1    20   120     6     0.541667
    2    30   210     7     0.440476
    3    40   320     8     0.375000

    Notes
    -----
    Exercise caution when evaluating untrusted expressions.
    Internally, this function uses the pandas
    :meth:`DataFrame.eval` method which depends on Python's
    :math:`eval` mechanism. For advanced usage or optimization,
    consider using the ``'numexpr'`` engine if available.

    See Also
    --------
    pandas.DataFrame.eval : The underlying pandas method used for
        expression parsing and evaluation.

    References
    ----------
    .. [1] McKinney, Wes. *Python for Data Analysis*, 2nd Edition,
       O'Reilly Media, 2017.
    """
    # Apply error policy with base='ignore'
    error = error_policy(
        error,
        base="ignore"
    )

    # Parse expressions and operation columns
    expr = columns_manager(
        expr,
        pattern=None
    )

    op_cols = columns_manager(
        op_cols, 
        pattern=None
    )
    # Decide which DataFrame to work on
    new_df = df if inplace else df.copy()

    # Track columns used in expressions for dropping later
    all_used_cols = []

    # Multiple expressions
    if len(expr) > 1:
        # Default column names if none provided
        if op_cols is None:
            op_cols = [
                f"eval_{i}"
                for i in range(len(expr))
            ]
        else:
            # Ensure op_cols length matches expr length
            if len(op_cols) < len(expr):
                if error == "raise":
                    raise ValueError(
                        "op_cols length must "
                        "match expr length."
                    )
                elif error in ["warn", "ignore"]:
                    if error == "warn":
                        warnings.warn(
                            "op_cols length < expr "
                            "length; extending with "
                            "defaults."
                        )
                    needed = len(expr) - len(op_cols)
                    op_cols += [
                        f"eval_{i+len(op_cols)}"
                        for i in range(needed)
                    ]
            elif len(op_cols) > len(expr):
                if error == "raise":
                    raise ValueError(
                        "op_cols length must "
                        "match expr length."
                    )
                elif error in ["warn", "ignore"]:
                    if error == "warn":
                        warnings.warn(
                            "op_cols length > expr "
                            "length; truncating extra "
                            "names."
                        )
                    op_cols = op_cols[: len(expr)]

        # Evaluate expressions and store results
        for operation, col_name in zip(expr, op_cols):
            used_cols = parse_used_columns(
                new_df,
                operation
            )
            all_used_cols.extend(used_cols)
            try:
                result = new_df.eval(
                    expr=operation,
                    engine=engine,
                    local_dict=local_dict,
                    global_dict=global_dict,
                    **eval_kw
                )
            except Exception as e:
                if error == "raise":
                    raise ValueError(
                        f"Error evaluating "
                        f"'{operation}': {e}"
                    )
                elif error == "warn":
                    warnings.warn(
                        f"Error evaluating "
                        f"'{operation}': {e}"
                    )
                    result = None
                else:
                    result = None

            if result is None:
                continue
            else:
                new_df[col_name] = result

    # Single expression
    else:
        expression = expr[0]
        used_cols = parse_used_columns(
            new_df,
            expression
        )
        all_used_cols.extend(used_cols)

        if op_cols is None:
            op_cols = "eval.name"
        elif isinstance(op_cols, list) and len(op_cols) == 1:
            op_cols = op_cols[0]
        elif isinstance(op_cols, list) and len(op_cols) != 1:
            if error == "raise":
                raise ValueError(
                    "op_cols must have length 1 "
                    "for a single expression."
                )
            elif error == "warn":
                warnings.warn(
                    "op_cols has unexpected length; "
                    "using the first element."
                )
                op_cols = op_cols[0]
            else:
                op_cols = op_cols[0]

        try:
            result = new_df.eval(
                expr=expression,
                engine=engine,
                local_dict=local_dict,
                global_dict=global_dict,
                **eval_kw
            )
        except Exception as e:
            if error == "raise":
                raise ValueError(
                    f"Error evaluating "
                    f"'{expression}': {e}"
                )
            elif error == "warn":
                warnings.warn(
                    f"Error evaluating "
                    f"'{expression}': {e}"
                )
                result = None
            else:
                result = None
        else: 
            
            new_df[op_cols] = result

    # Drop columns if requested
    if drop_cols:
        if drop_cols is True or drop_cols == "original":
            all_used_cols = list(set(all_used_cols))
            new_df.drop(
                columns=all_used_cols,
                inplace=True
            )
        elif isinstance(drop_cols, list):
            new_df.drop(
                columns=drop_cols,
                inplace=True
            )

    return new_df

@check_params ({ 
    "features": Optional[List[str]], 
    "name": str, 
    })
@validate_params({
    "ranking": ['array-like']
    }
)
def to_importances(
    ranking, 
    models=None, 
    features=None, 
    method='linear',
    normalize=False, 
    max_importance=1.0, 
    name="feature", 
    ascending=False, 
    epsilon=1e-8
    ):
    """
    Convert ranking values to importance scores.

    This function converts a ranking dataset (DataFrame or array-
    like) into importance scores using various conversion methods.
    If the input is a DataFrame, its columns and index are used
    unless overridden by the `models` and `features` parameters.
    If the provided features index is numeric, then the index is
    renamed to ``{name}_1``, ``{name}_2``, etc. For numpy arrays,
    a DataFrame is built with default naming if `models` or 
    `features` are not provided or are incomplete.

    See more in :ref:`User Guide <user_guide>`. 

    Parameters
    ----------
    x : array-like or DataFrame
        The ranking dataset. If a numpy array is provided, it is
        converted into a DataFrame.
    models : list-like, optional
        List of model names to use as column labels. If fewer names are
        provided than the number of columns in the dataset, the missing
        names are filled with default names (e.g., ``Model_1``, 
        ``Model_2``, etc.).
    features : list-like, optional
        List of feature names to use as the DataFrame index. If fewer names
        are provided than the number of rows, the missing names are filled
        with default names using the prefix defined by ``name``
        (e.g., ``feature_1``, ``feature_2``, etc.). If the index is 
        numeric, it is renamed accordingly.
    method : str, default ``linear``
        The conversion method. Supported values are ``linear``, 
        ``reciprocal``, ``log``, and ``exponential`` (or ``exp`` as an
        alias).
    normalize : bool, default False
        If True, the computed importance scores for each column are 
        normalized to sum to 1.
    max_importance : float, default 1.0
        The maximum importance score. If this value is between 0 and 1,
        it is treated as a scaling factor; if greater than 1, it is
        considered the absolute maximum importance value.
    name : str, default ``feature``
        The default prefix used for feature naming when no names are
        provided or when the index is numeric.
    ascending : bool or None, default=False
        Determines the sorting order of the importance matrix based on 
        the mean importance scores across all models.

        - `True`  → Computes the mean importance for each feature and
          sorts **from lowest to highest**.
        - `False` (default) → Computes the mean importance for each 
          feature and sorts **from highest to lowest**.
        - `None` → **No sorting is applied**, and the importance
          matrix remains unchanged.

        After sorting, the **mean importance column** used for 
        sorting is automatically removed.
    epsilon : float, default 1e-8
        A small constant added to denominators to avoid division by zero.

    Returns
    -------
    DataFrame
        A DataFrame containing the computed importance scores with the
        same structure as the ranking dataset.

    Notes 
    ------
    The conversion methods available are:

    **Linear:**
    
    .. math::
       I_{ij} = \\frac{\\max(R_j) - R_{ij} + 1}{\\max(R_j)}
       \\times I_{\\max}
       
    **Reciprocal:**
    
    .. math::
       I_{ij} = \\frac{1}{R_{ij} + \\epsilon} \\times I_{\\max}
       
    **Logarithmic:**
    
    .. math::
       I_{ij} = \\frac{\\ln(\\max(R_j)+1) - \\ln(R_{ij}+1)}
       {\\ln(\\max(R_j)+1)} \\times I_{\\max}
       
    **Exponential:**
    
    .. math::
       I_{ij} = I_{\\max} \\times \\exp\\Bigl(
       -\\frac{R_{ij}-1}{\\max(R_j)-1}\\Bigr)

    where :math:`R_{ij}` denotes the ranking of the <parameter inline>
    feature for model j, and :math:`I_{\\max}` is the maximum 
    importance as defined by the ``max_importance`` parameter.
    If ``max_importance`` is between 0 and 1, it is used as a scaling 
    factor; if greater than 1, it is treated as the absolute maximum
    importance value.
    
    Examples
    --------
    >>> import numpy as np
    >>> from gofast.utils.ext import to_importances
    >>> ranking = np.array([[1, 2],
    ...                     [3, 1],
    ...                     [2, 3]])
    >>> imp = to_importances(ranking,
    ...         models=['RF', 'XGBR'],
    ...         features=['feat1', 'feat2', 'feat3'],
    ...         method='linear')
    >>> print(imp)

    Notes
    -----
    The function uses helper functions such as 
    :func:`columns_manager` and 
    :func:`parameter_validator` to process and validate the 
    `models` and `features` inputs. If ``normalize`` is True, each 
    column is scaled so that the importance scores sum to 1. The 
    conversion follows the mathematical formulations above.
    
    See Also
    --------
    to_ranking : Convert importance scores to ranking values.
    pd.DataFrame : Construct a DataFrame from array-like data.

    References
    ----------
    .. [1] Hunter, J. D. (2007). Matplotlib: A 2D Graphics 
           Environment. Computing in Science & Engineering, 9(3), 
           90-95.
    """
    # Define valid conversion methods.
    valid_methods = ['linear', 'reciprocal', 'log', 'exponential']
    
    # Validate the 'method' parameter against valid methods.
    # Allow also 'exp' as shorthand for 'exponential'.
    method = parameter_validator(
        "method",
        target_strs=valid_methods + ['exp'],
        error_msg=(
            f"Method '{method}' not supported. Valid methods are:"
            f" {smart_format(valid_methods)}")
        )(method)
    
    if isinstance (ranking, pd.Series): 
        ranking = ranking.to_frame()

    # If 'ranking' is already a DataFrame, work on its copy.
    if isinstance(ranking, pd.DataFrame):
        df = ranking.copy()
        models = list(df.columns)  # Capture existing column names.
        features = list(df.index)  # Capture existing index values.
    else:
        # Otherwise, convert the ranking
        # array to a DataFrame.
        rank_arr = np.asarray(ranking)
        if rank_arr.ndim == 1:
            rank_arr = rank_arr.reshape(-1, 1)
        df = pd.DataFrame(rank_arr)
    
    # Number of columns in the DataFrame.
    ncols = df.shape[1]  
    # Convert the 'models' parameter 
    # into a list; return None if empty.
    models = columns_manager(models, empty_as_none=True, error="ignore") 
    if models is None:
        # If no models provided, assign default names.
        df.columns = [f"Model_{i}" for i in range(1, ncols + 1)]
    else:
        models= [get_estimator_name (m) for m in models ]
        # If provided, but insufficient in 
        # number, extend with default names.
        if len(models) < ncols:
            models.extend([
                f"Model_{i}" for i in range(len(models) + 1, ncols + 1)])
            # Resize the list to match the number of columns.
            models = models[:ncols]
        # Reassign the columns with the updated model names.
        df.columns = models

    nrows = df.shape[0]  # Number of rows in the DataFrame.
    features = columns_manager(features, empty_as_none=True)
    if features is None:
        # If no features provided and index is numeric, 
        # rename using the default pattern.
        if isinstance(df.index, pd.RangeIndex) or all(
                isinstance(x, (int, np.integer)) for x in df.index):
            df.index = [f"{name}_{i}" for i in range(1, nrows + 1)]
    else:
        # If provided but list is shorter than number of 
        # rows, extend with defaults.
        if len(features) < nrows:
            features.extend([
                f"{name}_{i}" for i in range(len(features) + 1, nrows + 1)])
            features = features[:nrows]
        # Reassign the index with the updated feature names.
        df.index = features

    if method.lower() == 'linear':
        # For linear conversion:
        #   Calculate the maximum rank per column.
        max_rank = df.max(axis=0)
        #   Compute importance scores as:
        #       I = ((max_rank - ranking + 1) / max_rank) * max_importance
        imp = (max_rank - df + 1) / max_rank * max_importance
    elif method.lower() == 'reciprocal':
        # For reciprocal conversion:
        #   Avoid division by zero by adding a small epsilon.
        imp = max_importance * (1 / (df + epsilon))
    elif method.lower() == 'log':
        # For logarithmic conversion:
        #   Compute importance using log transformation.
        #   Add 1 inside the log to prevent log(0).
        max_rank = df.max(axis=0)
        imp = (np.log(max_rank + 1) - np.log(df + 1)) / np.log(
            max_rank + 1) * max_importance
    elif method.lower() == 'exponential':
        # For exponential conversion:
        #   Calculate the maximum rank per column.
        max_rank = df.max(axis=0)
        #   Avoid division by zero by setting denominator to 1 if max_rank is 1.
        denom = np.where(max_rank - 1 == 0, 1, max_rank - 1)
        norm_rank = (df - 1) / denom
        imp = max_importance * np.exp(-norm_rank)

    if normalize:
        # Normalize importance scores so that each column sums to 1.
        imp = imp.div(imp.sum(axis=0), axis=1)
        
    # Sorting Importance Matrix Based on `ascending` Parameter
    if ascending is True:
        # Compute mean importance across features
        # and sort from lowest to highest
        imp["mean_importance"] = imp.mean(axis=1)
        imp = imp.sort_values(
            by="mean_importance", ascending=True
            ).drop(columns=["mean_importance"])
    
    elif ascending is False:
        # Compute mean importance across features
        # and sort from highest to lowest
        imp["mean_importance"] = imp.mean(axis=1)
        imp = imp.sort_values(
            by="mean_importance", ascending=False
            ).drop(columns=["mean_importance"])
    
    # for consistency 
    imp = reorder_by (imp, ascending=ascending )
    
    imp = to_series(imp, handle_2d ="passthrough")
    
    return imp

@check_params ({ 
    "features": Optional[List[str]], 
    "name": str, 
    })
@validate_params({
    "to_ranking": ['array-like']
    }
)
def to_ranking(
    importances, 
    models=None, 
    features=None,
    ascending=False, 
    rank_method='min',
    name="feature",
    strategy="by_data", 
    **kw
):
    """
    Convert importance scores to ranking values.

    This function converts a dataset of importance scores into
    ranking values. The input may be a DataFrame or an array-like
    object. If the input is not a DataFrame, it is converted into
    one, and default column names (models) and index names (features)
    are assigned if they are not provided or are incomplete. For
    instance, if the number of model names is less than the number
    of columns, default names such as ``Model_3`` are appended. If
    the index is numeric, it is renamed using the provided prefix,
    e.g. ``feature_1``, ``feature_2``, etc.

    See more in :ref:`User Guide <user_guide>`.
    
    Parameters
    ----------
    ranking : DataFrame or array-like
        The dataset containing importance scores. If not already a
        DataFrame, the input is converted into one. A 1D array is
        reshaped into a 2D array with one column.
    models  : list-like, optional
        A list of model names to be used as column labels. If the number
        of provided names is less than the number of columns in the
        dataset, the missing names are filled with defaults such as
        ``Model_1``, ``Model_2``, etc.
    features: list-like, optional
        A list of feature names to be used as the DataFrame index.
        If fewer names are provided than rows, the remaining rows
        are named using the prefix specified by ``name``, e.g.
        ``feature_1``, ``feature_2``, etc.
    ascending : bool, default=False
        Controls the order of ranking. By default, higher importance
        scores are ranked as 1 (i.e. highest importance). and sorted 
        by `'rank_strategy'`
    rank_method : str, default ``min``
        The method used to compute the rank. Acceptable values are
        those supported by pandas ``rank`` (e.g., ``min``,
        ``average``, etc.).
    name    : str, default ``feature``
        The prefix used for default feature naming when no feature
        names are provided or when the index is numeric.
    **kw    : dict, optional
        Additional keyword arguments passed to the DataFrame's
        ``rank`` method.

    Returns
    -------
    DataFrame
        A DataFrame with the same structure as the input, containing
        ranking values computed for each importance score.

    Notes 
    ------
    The ranking is computed via the pandas DataFrame ``rank``
    method. By default, higher importance scores receive a rank of
    1, achieved by setting the ranking order to descending (i.e.,
    ``ascending=not ascending``). Additional keyword arguments are
    passed directly to ``DataFrame.rank``.


    The conversion is mathematically defined as follows:

    .. math::
       R_{ij} = \\text{rank}(I_{ij})
       \\quad \\text{for each column } j

    where :math:`I_{ij}` denotes the importance score of the 
    <parameter inline> feature for model j, and :math:`R_{ij}` is 
    its corresponding rank.
    
    Examples
    --------
    >>> import numpy as np
    >>> from gofast.utils.ext import to_ranking
    >>> importance = np.array([[0.8, 0.2],
    ...                        [0.3, 0.9],
    ...                        [0.5, 0.4]])
    >>> rank_df = to_ranking(importance,
    ...          models=['RF', 'XGBR'],
    ...          features=['feat1', 'feat2', 'feat3'],
    ...          ascending=False, rank_method='min')
    >>> print(rank_df)

    Notes
    -----
    This function employs helper functions such as
    :func:`columns_manager` to manage column and index naming, and
    it utilizes the pandas ``rank`` method to compute the ranking.
    The default behavior assigns a rank of 1 to the highest
    importance score, consistent with standard ranking practices
    [1]_.

    See Also
    --------
    to_importances : Convert ranking values to importance scores.
    pd.DataFrame.rank : Rank elements in a DataFrame.

    References
    ----------
    .. [1] Smith, J., & Doe, A. (2020). Ranking Methods in Data
           Analysis. Journal of Data Science, 18(2), 101-110.
    """
    
    # Check if 'importances' is already a DataFrame; if not,
    # convert it to one.
    if isinstance (importances, pd.Series): 
        importances=importances.to_frame()
        
    if isinstance(importances, pd.DataFrame):
        df = importances.copy()
        models= list(df.columns) 
        features= list(df.index)
    else:
        imp_arr = np.asarray(importances)
        # If the array is 1D, reshape to a 2D array with one column.
        if imp_arr.ndim == 1:
            imp_arr = imp_arr.reshape(-1, 1)
        df = pd.DataFrame(imp_arr)

    # Set up the columns (models). The number of columns in df.
    ncols = df.shape[1]
    models = columns_manager(models, empty_as_none=True )
    if models is None: 
        # If models not provided, generate default model names.
        df.columns = [f"Model_{i}" for i in range(1, ncols + 1)]
        
    else: 
        models= [get_estimator_name (m) for m in models ]
        if len(models) < ncols:
            models.extend([
                f"Model_{i}" for i in range(len(models) + 1,ncols + 1)])
            models = models[:ncols]
            df.columns = models

    # Set up the index (features). The number of rows in df.
    nrows = df.shape[0]
    features = columns_manager(features, empty_as_none=True )
    
    if features is  None:
        # If features not provided, generate default feature names.
        df.index = [f"{name}_{i}" for i in range(1, nrows + 1)]
    else: 
        
        # append default feature names.
        if len(features) < nrows:
            features.extend([f"{name}_{i}" for i in range(len(features) + 1,
                                                            nrows + 1)])
        features = features[:nrows]
        df.index = features
    
    # Compute the ranking for each column.
    # By default, higher importance should get rank 1; hence,
    # if ascending is False, we rank in descending order.
    ranking = reorder_by (
        df, 
        ascending =ascending, 
        to_rank=True, 
        rank_strategy=strategy, 
        rank_method=rank_method, 
        **kw
    )
    ranking = to_series (ranking, handle_2d ="passthrough")
    
    return ranking.astype(int)

@check_params ({ 
    "features": Optional[List[str]], 
    "name": str, 
    })
@validate_params({
    "to_ranking": ['array-like']
    }
)
@isdf
def reorder_by(
    df, 
    ascending=True, 
    to_rank=False, 
    rank_strategy='by_col', 
    rank_method='min', 
    **rank_kw
):
    """
    Reorders the rows of a DataFrame based on computed rankings or values.

    The `reorder_by` function provides an efficient way to rearrange a 
    DataFrame based on either ranking the columns individually, ranking 
    based on the sum of row values, or ranking the entire DataFrame by 
    a computed temporary column. The DataFrame can be either sorted 
    or ranked based on different strategies for both column-wise and 
    row-wise calculations. This function allows flexible ranking 
    strategies to cater to various needs for data reorganization.

    Parameters
    ------------
    df : pandas.DataFrame
        The DataFrame to be reordered. It must contain numeric values 
        for the ranking or reordering process.
        
    ascending : bool, default=True
        If `True`, the DataFrame is sorted in ascending order based on 
        either the row-wise computed mean or the rank values. If `False`, 
        it is sorted in descending order.

    to_rank : bool, default=False
        Whether to return the ranked DataFrame instead of the reordered 
        DataFrame. If `True`, the function converts values into ranks 
        instead of sorting them by the computed temporary column. 

    rank_strategy : {'by_col', 'by_data'}, default='by_col'
        Defines the strategy used for ranking the DataFrame. 
        - If `'by_col'`, the DataFrame is ranked individually by each 
          column.
        - If `'by_data'`, the ranking is based on the summed values of 
          each row (calculated via a temporary column), and the entire 
          DataFrame is reordered based on the ranks of these summed values.
    
    rank_method : {'min', 'max', 'average', 'first'}, default='min'
        Specifies the method used to rank the values. Available options:
        - `'min'`: Assigns the smallest rank to tied elements.
        - `'max'`: Assigns the largest rank to tied elements.
        - `'average'`: Assigns the average rank to tied elements.
        - `'first'`: Assigns ranks in the order of appearance of the elements.
        
    rank_kw : keyword arguments, optional
        Additional parameters passed directly to pandas' `rank` method 
        to fine-tune the ranking behavior. These can include parameters 
        like `na_option`, `numeric_only`, etc.

    Returns
    -------
    pandas.DataFrame
        The reordered or ranked DataFrame, depending on the value of 
        the `to_rank` parameter. If `to_rank` is `True`, the output will 
        be a DataFrame with rank values; otherwise, the original DataFrame 
        will be rearranged based on either row-wise sums or individual 
        column rankings.

    Formulation
    -------------
    Given a DataFrame `df` with `n` rows and `m` columns, the temporary 
    column `temp_col` is computed as the mean across each row, i.e.,

    .. math::
       \text{temp\_col}_i = \frac{1}{m} \sum_{j=1}^{m} \text{df}_{ij}

    where :math:`\text{df}_{ij}` denotes the value at row `i` and column `j`.

    If `to_rank=True`, the function then performs the following operations:

    1. **By Column**: Each column is ranked independently.
    2. **By Data**: The sum of each row is used to rank the rows 
       across all columns.

    The final result depends on the sorting behavior defined by 
    `ascending` and `rank_strategy`.

    Example
    -------
    >>> from gofast.utils.ext import reorder_by
    >>> df = pd.DataFrame({
    >>>     'A': [0.1, 0.01, 0.2],
    >>>     'B': [0.6, 0.9, 0.0],
    >>>     'C': [0.33, 0.1, 1.3]
    >>> })
    >>> result = reorder_by(df, ascending=True, to_rank=False)
    >>> print(result)
    >>> result_ranked = reorder_by(df, to_rank=True, rank_strategy='by_col')
    >>> print(result_ranked)

    Notes
    -----
    - When `to_rank=True` and `rank_strategy='by_data'`, the ranking is 
      performed on the sum of each row. The rows are then reordered 
      according to their rank.
    - The temporary column `temp_col` is used in the case of `by_data` 
      strategy to compute the row-wise sums before ranking. This column 
      is removed before returning the result.

    See Also
    --------
    pandas.DataFrame.rank : Rank values in a DataFrame.
    
    References
    ----------
    .. [1] Pandas Documentation: DataFrame Rank Method,
           https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rank.html
    """

    rank_strategy= rank_strategy or 'by_col'
    
    rank_strategy = parameter_validator(
        "rank_strategy",
        target_strs={'by_col', 'by_data'},
        error_msg=(
            f"Ranking strategy '{rank_strategy}'."
            " Choose one of 'by_col' or 'by_data'"
            )
        )(rank_strategy)

    # Compute the temporary column: mean along axis=1 (row-wise)
    df['temp_col'] = df.mean(axis=1)
    
    # Case 1: When to_rank is False - arrange based on mean values.
    if not to_rank:
        # Rank the dataframe based on the sum of temp_col
        df_sorted = df.sort_values(by='temp_col', ascending=ascending)
        # Drop the temporary column after rearranging
        df_sorted = df_sorted.drop(columns=['temp_col'])
        
        return df_sorted

    # Case 2: When to_rank is True - convert data to ranking values.
    else:
        # Rank the dataframe based on the selected strategy
        if rank_strategy == 'by_col':
            # Rank each column individually based on values
            df_ranked = df.rank(
                ascending=ascending, 
                method=rank_method, 
                **rank_kw
            ).astype(int)
            
            # Drop the temporary column after ranking
            df_ranked = df_ranked.drop(columns=['temp_col'])
            
            return df_ranked.astype(int)  
        
        elif rank_strategy == 'by_data':
            # Rank based on the temporary column 'temp_col'
            # Rank each column by data first
            df_ranked = df.rank(
                ascending=False, 
                method=rank_method, 
                **rank_kw
            ).astype(int)
       
            # Rank the temp column and use that to rank the whole dataframe
            df_ranked['temp_col_rank'] = df['temp_col'].rank(
                ascending=False, 
                method=rank_method, 
                **rank_kw
            ).astype(int)
     
            # Sort dataframe based on temp_col_rank
            df_ranked = df_ranked.sort_values(
                by='temp_col_rank', 
                ascending=not ascending
            )
            
            # Drop the temporary ranking column
            # Drop the temporary column used for mean calculation
            df_ranked = df_ranked.drop(columns=['temp_col', 'temp_col_rank'])
        
            # Ensure consistent integer ranking
            return df_ranked.astype(int)

def normalize_in(
    *d: List[ArrayLike], 
    method: str= "01",       
    scaler: str = "math.op",  
    nan_policy: str = "omit",     
    columns: List[str] = None,       
    error: str  = "warn",     
    axis: Optional[int] = None,            
    epsilon=1e-8
):
    """
    Normalize the input data using the specified method and scaler.
    
    This function accepts one or more array-like inputs (NumPy arrays, 
    pandas Series, or pandas DataFrames) and normalizes their numeric 
    data according to the method specified. If the input is a DataFrame 
    containing both numeric and categorical data, only the numeric 
    columns (or those specified in `columns`) are normalized. After 
    normalization, the original DataFrame structure (i.e. column order) 
    is preserved.
    
    Parameters
    ----------
    *d : array-like
        One or more inputs to be normalized. Each input may be a 
        NumPy array, pandas Series, or pandas DataFrame.
    method : ``str``, default ``"01"``
        Normalization method to apply. Options include:
        
        - ``"01"``: Scale data to the range [0, 1] using
          :math:`x_{norm} = \\frac{x - \\min(x)}{\\max(x) - \\min(x)}`.
        - ``"zscore"``: Standardize data to zero mean and unit 
          variance via :math:`x_{norm} = \\frac{x - \\mu}{\\sigma}`.
        - ``"sum"``: Scale data so that the sum of elements is 1,
          i.e. :math:`x_{norm} = \\frac{x}{\\sum x}`.
    scaler : ``str``, default ``"math.op"``
        Specifies the scaling approach. If set to ``"math.op"``,
        normalization is performed using direct mathematical 
        operations. If set to ``"sklearn"``, scikit-learn’s scalers 
        are used (note: the "sum" method is not supported with 
        sklearn).
    nan_policy : ``str``, default ``"omit"``
        How to handle NaN values. Options:
        
        - ``"omit"``: Use functions like `np.nanmin`, `np.nanmax`, 
          and `np.nansum` to ignore NaNs.
        - ``"propagate"``: Allow NaNs to propagate in the result.
        - ``"raise"``: Raise an error if NaNs are present.
    columns : ``list``, optional
        When an input is a DataFrame, only these columns are normalized.
        Other columns remain unchanged. If a specified column is missing,
        behavior is governed by the `error` parameter.
    error : ``str``, default ``"warn"``
        Error handling when specified columns are not found. Options:
        
        - ``"raise"``: Raise an error.
        - ``"warn"``: Issue a warning.
        - ``"ignore"``: Silently ignore missing columns.
    axis : ``int``, optional
        The axis along which to normalize the data for 2D arrays:
        
        - ``None``: Normalize over all elements.
        - ``0``: Normalize each column.
        - ``1``: Normalize each row.
    
    Returns
    -------
    result : {np.ndarray, pandas.Series, pandas.DataFrame} or list
        The normalized data, in the same type as each input. If a single 
        input was provided, returns the normalized version directly; 
        otherwise, returns a list of normalized outputs.
    
    Notes
    -----
    The function first determines whether each input is a DataFrame, 
    Series, or NumPy array. For DataFrames, if `columns` is specified, 
    only those columns are normalized using the specified method. 
    Any categorical data is preserved in its original order. 
    When `scaler` is set to ``"sklearn"``, scikit-learn's 
    MinMaxScaler or StandardScaler is used depending on `method` 
    (except for the "sum" method which is not supported).
    
    Examples
    --------
    1) **Normalize a NumPy array to [0, 1]:**
    
       >>> import numpy as np
       >>> from gofast.utils.ext import normalize_in
       >>> arr = np.array([10, 20, 30, 40])
       >>> norm_arr = normalize_in(arr, method="01")
       >>> norm_arr
       array([0.  , 0.333, 0.667, 1.   ])
    
    2) **Standardize a pandas Series using z-score:**
    
       >>> import pandas as pd
       >>> s = pd.Series([100, 150, 200])
       >>> norm_s = normalize_in(s, method="zscore")
       >>> norm_s
       0   -1.224745
       1    0.000000
       2    1.224745
       dtype: float64
    
    3) **Normalize specific columns in a DataFrame:**
    
       >>> import pandas as pd
       >>> df = pd.DataFrame({
       ...     "A": [1, 2, 3],
       ...     "B": [4, 5, 6],
       ...     "C": ["X", "Y", "Z"]
       ... })
       >>> norm_df = normalize_in(
       ...     df, 
       ...     method="01", 
       ...     columns=["A", "B"],
       ...     axis=0
       ... )
       >>> norm_df
            A    B  C
       0  0.0  0.0  X
       1  0.5  0.5  Y
       2  1.0  1.0  Z

    See Also
    --------
    :func:`numpy.asarray` : Converts inputs to NumPy arrays.
    :func:`~gofast.core.checks.check_numeric_dtype` : Validates numeric data.
    
    References
    ----------
    .. [1] Bishop, C. M. "Pattern Recognition and Machine Learning." 
           Springer, 2006.
    """

    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    # Process each input in *d and store results
    results = []
    
    columns = columns_manager(columns)
    if columns is not None:
        missing_cols= columns_getter(
            *d, error="ignore", 
            return_cols="missing", 
            columns=None
        )
        if missing_cols:
            msg = (f"Columns {missing_cols} not found in DataFrame.")
            if error == "raise":
                raise ValueError(msg)
            elif error == "warn":
                warnings.warn(msg)
                
    error = error_policy (error, base="warn")
    
    method =parameter_validator (
        "method", target_strs={"01","zscore", "sum" },
        error_msg=(f"Unsupported method: {method}"))(method)
    
    for item in d:
        # If item is a DataFrame, process numeric columns separately.
        if isinstance(item, pd.DataFrame):
            orig_df = item.copy()
            # If columns is specified, use only these columns for normalization.
            if columns is not None:
                num_cols = [col for col in columns if col in item.columns]
            else:
                num_cols = select_dtypes (
                    item, incl=[np.number], 
                    return_columns=True,
                    )
                # num_cols = item.select_dtypes(include=[np.number]).columns.tolist()
            # If no numeric columns, warn and leave item unchanged.
            if not num_cols:
                if error in ["warn", "ignore"]:
                    if error=="warn":
                        warnings.warn("No numeric columns found; skipping "
                                      "normalization for this item.")
                    
                    results.append(orig_df)
                    continue
                else:
                    raise ValueError("No numeric columns found.")
            # For each numeric column, apply normalization.
            for col in num_cols:
                col_data = item[col].to_numpy(dtype=float)
                # Handle NaNs based on nan_policy.
                if nan_policy == "omit":
                    min_val = np.nanmin(col_data)
                    max_val = np.nanmax(col_data)
                    total = np.nansum(col_data) if method=="sum" else None
                elif nan_policy == "propagate":
                    min_val = col_data.min()
                    max_val = col_data.max()
                    total = col_data.sum() if method=="sum" else None
                elif nan_policy == "raise":
                    if np.isnan(col_data).any():
                        raise ValueError(f"NaN found in column {col}.")
                    min_val = col_data.min()
                    max_val = col_data.max()
                    total = col_data.sum() if method=="sum" else None
                else:
                    raise ValueError("Invalid nan_policy value.")
                # Use sklearn scalers if specified.
                if scaler == "sklearn" and method in ["01", "zscore"]:
                    if method == "01":
                        sc = MinMaxScaler()
                    else:
                        sc = StandardScaler()
                    norm_data = sc.fit_transform(col_data.reshape(-1, 1)).ravel()
                else:
                    if method == "01":
                        norm_data = (col_data - min_val) / \
                            (max_val - min_val + epsilon)
                    elif method == "zscore":
                        norm_data = (col_data - col_data.mean()) / \
                            (col_data.std() + epsilon)
                    elif method == "sum":
                        if total is None or np.abs(total) < epsilon:
                            raise ValueError(f"Sum of column {col} is too small.")
                        norm_data = col_data / total
            
                item[col] = norm_data
            # Reassemble the DataFrame with original column order.
            results.append(item[orig_df.columns])
        # If item is a pandas Series.
        elif isinstance(item, pd.Series):
            if pd.api.types.is_numeric_dtype(item):
                arr = item.to_numpy(dtype=float)
                if nan_policy == "omit":
                    min_val = np.nanmin(arr)
                    max_val = np.nanmax(arr)
                    total = np.nansum(arr) if method=="sum" else None
                elif nan_policy == "propagate":
                    min_val = arr.min()
                    max_val = arr.max()
                    total = arr.sum() if method=="sum" else None
                elif nan_policy == "raise":
                    if np.isnan(arr).any():
                        raise ValueError("NaN found in Series.")
                    min_val = arr.min()
                    max_val = arr.max()
                    total = arr.sum() if method=="sum" else None
                else:
                    raise ValueError("Invalid nan_policy value.")
                if scaler == "sklearn" and method in ["01", "zscore"]:
                    if method == "01":
                        sc = MinMaxScaler()
                    else:
                        sc = StandardScaler()
                    norm_arr = sc.fit_transform(arr.reshape(-1, 1)).ravel()
                else:
                    if method == "01":
                        norm_arr = (arr - min_val) / (max_val - min_val + epsilon)
                    elif method == "zscore":
                        norm_arr = (arr - arr.mean()) / (arr.std() + epsilon)
                    elif method == "sum":
                        if total is None or np.abs(total) < epsilon:
                            raise ValueError("Sum of series is too small.")
                        norm_arr = arr / total
           
                results.append(pd.Series(norm_arr, index=item.index, name=item.name))
            else:
                if error=="warn": 
                    warnings.warn("Series is not numeric; returning as is.")
                elif error=="raise": 
                    raise ValueError("Series is not numeric")
                    
                results.append(item)
        # Otherwise, assume item is a NumPy array.
        else:
            arr = np.asarray(item, dtype=float)
            if nan_policy == "omit":
                min_val = np.nanmin(arr)
                max_val = np.nanmax(arr)
                total = np.nansum(arr) if method=="sum" else None
            elif nan_policy == "propagate":
                min_val = arr.min()
                max_val = arr.max()
                total = arr.sum() if method=="sum" else None
            elif nan_policy == "raise":
                if np.isnan(arr).any():
                    raise ValueError("NaN found in array.")
                min_val = arr.min()
                max_val = arr.max()
                total = arr.sum() if method=="sum" else None
            else:
                raise ValueError("Invalid nan_policy value.")
            if scaler == "sklearn" and method in ["01", "zscore"]:
                if method == "01":
                    sc = MinMaxScaler()
                else:
                    sc = StandardScaler()
                norm_arr = sc.fit_transform(arr.reshape(-1, 1)).ravel()
            else:
                if method == "01":
                    norm_arr = (arr - min_val) / (max_val - min_val + epsilon)
                elif method == "zscore":
                    norm_arr = (arr - arr.mean()) / (arr.std() + epsilon)
                elif method == "sum":
                    if total is None or np.abs(total) < epsilon:
                        raise ValueError("Sum of array is too small.")
                    norm_arr = arr / total

            results.append(norm_arr)
            
    # Return a single result if only one input was provided.
    if len(results) == 1:
        return results[0]
    return results

def denormalize_in(
    d: DataFrame, 
    from0:Optional[Union[str, ArrayLike]]=None, 
    minmax_vals:Optional[Tuple[float, float]]=None, 
    method: str="01", 
    columns: Optional[str]=None, 
    std_dev_factor: float = 3, 
    epsilon: float=1e-8
):
    """
    Denormalize numeric data in a DataFrame or Series using either a 
    common min-max (from a specified column or tuple) or each column’s 
    own min and max. For DataFrames containing both numeric and 
    categorical data, the function denormalizes the numeric columns 
    (or only those specified by `columns`) and then reconstructs the 
    DataFrame in its original column order.
    """
    
    method =parameter_validator (
        "method", target_strs={"01","zscore", "sum" },
        error_msg=(f"Unsupported method: {method}"))(method)
    
    data= copy.deepcopy(d)
    # If input is a pandas Series, convert to NumPy array.
    if isinstance(data, pd.Series):
        # If the series is numeric, denormalize directly.
        if pd.api.types.is_numeric_dtype(data):
            arr = data.to_numpy(dtype=float)
            if minmax_vals is not None:
                common_min, common_max = minmax_vals
            elif from0 is not None:
                # If from0 is provided for a Series, it must be array-like.
                arr_from0 = (np.asarray(from0, dtype=float) 
                             if not isinstance(from0, pd.Series) 
                             else from0.to_numpy(dtype=float))
                common_min, common_max = arr_from0.min(), arr_from0.max()
            else:
                # Use the series' own min and max.
                common_min, common_max = data.min(), data.max()

            if method == "01":
                denorm = arr * (common_max - common_min) + common_min
            elif method == "zscore":
                mean    = (common_min + common_max) / 2
                std_dev = (common_max - common_min) / (2 * std_dev_factor)
                denorm = arr * std_dev + mean
            elif method == "sum":
                total   = np.sum(arr)
                if np.abs(total) < epsilon:
                    raise ValueError("Sum of series is too small for "
                                     "denormalization.")
                denorm = arr * ((common_max - common_min) / total) + common_min
        
            return pd.Series(denorm, index=data.index, name=data.name)

        else:
            # Non-numeric Series: warn and return as is.
            warnings.warn("Series is not numeric. Returning input as is.")
            return data

    # Process a pandas DataFrame.
    elif isinstance(data, pd.DataFrame):
        orig_df = data.copy()
        # Determine columns to denormalize: if <columns> is provided,
        # use those; otherwise, select all numeric columns.
        if columns is not None:
            num_cols = [col for col in columns if col in data.columns]
            if not num_cols:
                raise ValueError("None of the specified columns were "
                                 "found in the DataFrame.")
        else:
            num_cols = select_dtypes (
                data, incl=[np.number], 
                return_columns=True, 
                )
            # num_cols = data.select_dtypes(include=[np.number]).columns.tolist()

        # If no numeric columns are found, warn and return original.
        if not num_cols:
            warnings.warn("No numeric columns found. Returning data as is.")
            return orig_df

        # Determine common min and max values.
        if minmax_vals is not None:
            common_min, common_max = minmax_vals
        elif from0 is not None:
            if isinstance(from0, str):
                if from0 not in data.columns:
                    raise ValueError(
                        f"Column '{from0}' not found in DataFrame."
                    )
                common_min = data[from0].min()
                common_max = data[from0].max()
            else:
                arr_from0 = (np.asarray(from0, dtype=float) 
                             if not isinstance(from0, pd.Series) 
                             else from0.to_numpy(dtype=float))
                common_min = arr_from0.min()
                common_max = arr_from0.max()
        else:
            # If no common minmax provided, denormalize each column
            # individually.
            common_min = None
            common_max = None

        # Denormalize numeric columns.
        for col in num_cols:
            col_data = data[col].to_numpy(dtype=float)
            # Use common min/max if available; else, use column's own.
            if common_min is not None and common_max is not None:
                min_val = common_min
                max_val = common_max
            else:
                min_val = np.min(col_data)
                max_val = np.max(col_data)

            if method == "01":
                denorm = col_data * (max_val - min_val) + min_val
            elif method == "zscore":
                mean    = (min_val + max_val) / 2
                std_dev = (max_val - min_val) / (2 * std_dev_factor)
                denorm = col_data * std_dev + mean
            elif method == "sum":
                total   = np.sum(col_data)
                if np.abs(total) < epsilon:
                    raise ValueError(f"Sum of column {col} is too small for "
                                     "denormalization.")
                denorm = col_data * ((max_val - min_val) / total) + min_val
    
            # Replace the numeric column with its denormalized values.
            data[col] = denorm

        # Reconstruct DataFrame in original column order.
        # (Categorical columns remain unchanged.)
        final_df = data[orig_df.columns]
        return final_df
    
    # If input is a NumPy array, process it as numeric.
    else: 
        check_numeric_dtype(
            data, coerce=True, param_names={"X": "Data 'd'"}
        )
        arr = np.asarray(data, dtype=float)
        if minmax_vals is not None:
            common_min, common_max = minmax_vals
        elif from0 is not None:
            arr_from0 = np.asarray(from0, dtype=float)
            common_min, common_max = arr_from0.min(), arr_from0.max()
        else:
            common_min, common_max = arr.min(), arr.max()
    
        if method == "01":
            return arr * (common_max - common_min) + common_min
        elif method == "zscore":
            mean    = (common_min + common_max) / 2
            std_dev = (common_max - common_min) / (2 * std_dev_factor)
            return arr * std_dev + mean
        elif method == "sum":
            total   = np.sum(arr)
            if np.abs(total) < epsilon:
                raise ValueError("Sum of array is too small for "
                                 "denormalization.")
            return arr * ((common_max - common_min) / total) + common_min
   
def denormalizer(
    d,
    min_val: float,
    max_val: float,
    method: str = '01',
    std_dev_factor: float = 3,
    columns=None,
    epsilon=1e-8
):
    """
    Denormalize data from a normalized scale back to its
    original scale. This function supports three methods:
    ``'01'``, ``'zscore'``, and ``'sum'``. It can process
    both numeric and categorical data within a pandas
    DataFrame, returning the denormalized numerical part
    merged with any untouched categorical columns.

    Internally, it uses `parameter_validator` to ensure
    the ``method`` is valid. When a pandas or NumPy input
    is purely numeric, `check_numeric_dtype` helps validate
    the data type. In other scenarios, non-numeric columns
    remain unchanged.

    Parameters
    ----------
    d : ndarray, Series or DataFrame
        Numeric data or a mix of numeric and
        categorical columns that need to be
        denormalized. Categorical columns remain
        unchanged.
    min_val : float
        The minimum value used in the normalization
        context. For method ``'01'``, it represents
        the lower bound of the original scale. For
        method ``'zscore'``, it combines with
        ``max_val`` to derive the mean. For method
        ``'sum'``, it is added back after rescaling.
    max_val : float
        The maximum value in the normalization
        context. Similar usage as ``min_val`` but
        acts as the upper bound or helps compute
        the mean, depending on the method.
    method : str, optional
        The normalization mode. Valid options:
        ``'01'``, ``'zscore'``, or ``'sum'``.
        Defaults to ``'01'``. Each choice
        triggers a different denormalization
        formula as shown above.
    std_dev_factor : float, optional
        Used in method ``'zscore'`` to define how
        the standard deviation is derived from
        :math:`(max\_val - min\_val) / (2 \times
        \text{std\_dev\_factor})`. Defaults to 3.
    columns : list of str, optional
        Columns to explicitly denormalize in a
        DataFrame. Other columns are treated as
        categorical and returned as is. If
        unspecified, numeric columns are auto-
        selected.
    epsilon : float, optional
        A small constant to guard against division
        by zero in method ``'sum'``. Defaults to
        1e-8.

    Returns
    -------
    ndarray or Series or DataFrame
        The denormalized data in the same type as
        ``d``. If a DataFrame with numeric and
        categorical columns is passed, it returns
        a DataFrame with denormalized numeric
        columns and untouched categoricals.

    Notes
    -----
    - If no numeric columns are found in a
      DataFrame, a warning is issued, and the
      original DataFrame is returned unchanged.
    - For method ``'sum'``, the sum of the data
      (in normalized space) is used to compute
      the scale. If that sum is below the
      specified ``epsilon``, an error is raised.
      
    .. math::
       \text{For `01`: } x_{\text{denorm}} \;=\;
       x_{\text{norm}} \;\times\; (\text{max\_val} \;-\;
       \text{min\_val})\;+\;\text{min\_val}

    .. math::
       \text{For `zscore`: } x_{\text{denorm}} \;=\;
       x_{\text{norm}} \;\times\; \sigma\;+\;\mu

    where :math:`\mu = \frac{\text{min\_val} +
    \text{max\_val}}{2}` and :math:`\sigma =
    \frac{\text{max\_val} - \text{min\_val}}{2 \times
    \text{std\_dev\_factor}}`.

    .. math::
       \text{For `sum`: } x_{\text{denorm}} \;=\;
       x_{\text{norm}} \;\times\; \left(\frac{\text{max\_val}
       - \text{min\_val}}{\sum x_{\text{norm}}}\right)\;+\;
       \text{min\_val}
       
    Examples
    --------
    >>> from gofast.utils.ext import denormalizer
    >>> import numpy as np
    >>> import pandas as pd
    >>> # Example with NumPy array
    ... arr_norm = np.array([0.2, 0.8])
    >>> denorm_arr = denormalizer(arr_norm,
    ...                           min_val=10,
    ...                           max_val=20,
    ...                           method='01')
    >>> denorm_arr
    array([12., 18.])

    >>> # Example with DataFrame
    ... df = pd.DataFrame({
    ...     'A': [0.25, 0.75],
    ...     'B': ['cat', 'dog']
    ... })
    >>> denorm_df = denormalizer(df, 0, 40, '01')
    >>> denorm_df
         A    B
    0  10.0  cat
    1  30.0  dog

    See Also
    --------
    parameter_validator : Ensures that certain
        string parameters (like ``method``)
        belong to a predefined set of choices.

    check_numeric_dtype : Validates and coerces
        data to numeric type.

    References
    ----------
    .. [1] "Normalization Techniques: A Survey of
           Theoretical and Empirical Analysis."
           *International Journal of Data Science*,
           vol. 5, no. 2, pp. 45-62, 2020.
    """

    method =parameter_validator (
        "method", target_strs={"01","zscore", "sum" },
        error_msg=( f"Unsupported normalization method: {method}"))(method)
    
    # If d is a DataFrame, process numeric and categorical 
    # columns separately.
    if isinstance(d, pd.DataFrame):
        if columns is not None:
            # Select specified columns for denormalization.
            missing = [col for col in columns if col not in d.columns]
            if missing:
                raise ValueError(
                    f"Columns {missing} not found in DataFrame."
                )
            num_df = d[columns]
            cat_df = d.drop(columns=columns)
        else:
            # Automatically select numeric columns.
            num_df = select_dtypes(
                d, incl=[np.number], 
                )
            # num_df = d.select_dtypes(include=[np.number])
            # And categorical columns.
            cat_df = select_dtypes (
                d, excl=[np.number], 
                )
            # cat_df = d.select_dtypes(exclude=[np.number])
        
        # If no numeric data, warn and return d.
        if num_df.empty:
            warnings.warn(
                "Warning: No numeric columns found. Returning data as is.")
            return d
        
        # Convert numeric data to NumPy array.
        data_values = num_df.to_numpy()
        
        # Denormalize based on the specified method.
        if method == '01':
            denorm = data_values * (max_val - min_val) + min_val
        elif method == 'zscore':
            mean   = (min_val + max_val) / 2
            std_dev = (max_val - min_val) / (2 * std_dev_factor)
            denorm = data_values * std_dev + mean
        elif method == 'sum':
            total = np.sum(data_values)
            if np.abs(total) < epsilon:
                raise ValueError(
                    "Sum of data is too small for normalization."
                )
            denorm = data_values * ((max_val - min_val) / total) + min_val
        else:
            raise ValueError(
                f"Unsupported normalization method: {method}"
            )
        
        # Reconstruct numeric DataFrame.
        denorm_df = pd.DataFrame(denorm, 
            index=num_df.index, columns=num_df.columns)
        
        # Reassemble with categorical data, preserving original order.
        if not cat_df.empty:
            orig_cols = d.columns.tolist()
            combined = {}
            for col in orig_cols:
                if col in denorm_df.columns:
                    combined[col] = denorm_df[col]
                else:
                    combined[col] = cat_df[col]
            final_df = pd.DataFrame(combined)
        else:
            final_df = denorm_df
        return final_df

    # If d is a Series, check if numeric.
    elif isinstance(d, pd.Series):
        if pd.api.types.is_numeric_dtype(d):
            arr = d.to_numpy()
            if method == '01':
                denorm = arr * (max_val - min_val) + min_val
            elif method == 'zscore':
                mean   = (min_val + max_val) / 2
                std_dev = (max_val - min_val) / (2 * std_dev_factor)
                denorm = arr * std_dev + mean
            elif method == 'sum':
                total = np.sum(arr)
                if np.abs(total) < epsilon:
                    raise ValueError(
                        "Sum of series is too small for normalization."
                    )
                denorm = arr * ((max_val - min_val) / total) + min_val
            else:
                raise ValueError(
                    f"Unsupported normalization method: {method}"
                )
            return pd.Series(denorm, index=d.index, name=d.name)
        else:
            warnings.warn(
                " Series is not numeric. Returning data as is.")
            return d

    # If d is a NumPy array, process directly.
    else:
        d= check_numeric_dtype(
            d, ops="validate",
            coerce=True, 
            param_names ={"X":"Data 'd'"}
        )
        arr = np.asarray(d, dtype=float)
        if method == '01':
            denorm = arr * (max_val - min_val) + min_val
        elif method == 'zscore':
            mean   = (min_val + max_val) / 2
            std_dev = (max_val - min_val) / (2 * std_dev_factor)
            denorm = arr * std_dev + mean
        elif method == 'sum':
            total = np.sum(arr)
            if np.abs(total) < epsilon:
                raise ValueError(
                    "Sum of array is too small for normalization."
                )
            denorm = arr * ((max_val - min_val) / total) + min_val
        else:
            raise ValueError(
                f"Unsupported normalization method: {method}"
            )
        return denorm


@is_data_readable 
def reorder_importances(
    data,
    axis: int = 0,
    method: str | callable = "sum",
    ascending: bool = False,
    top_n: int | None = None,
    copy: bool = True
) -> pd.DataFrame:
    """
    Reorder the rows or columns of feature importances
    (or any numeric data) in descending (or ascending)
    order, based on an aggregation method such as
    "sum", "mean", "max", or a custom callable.

    Parameters
    ----------
    data : array-like, pd.Series, or pd.DataFrame
        The data whose rows or columns we want
        to reorder. If a NumPy array or Series is
        provided, it will be converted to a
        DataFrame.
    axis : int, default=0
        Axis along which to compute the sorting
        key. If ``0``, reorder rows based on an
        aggregation across columns. If ``1``,
        reorder columns based on an aggregation
        across rows.
    method : {'sum', 'mean', 'max'} or callable, default='sum'
        Aggregation method used to evaluate
        each row (if ``axis=0``) or each column
        (if ``axis=1``). If a callable is given,
        it should accept a 1D array and return a
        scalar, e.g. ``np.median``. 
    ascending : bool, default=False
        If ``False``, sort from largest to
        smallest. If ``True``, sort from smallest
        to largest.
    top_n : int, optional
        If provided, keep only the top ``n``
        rows or columns after sorting (depending
        on ``axis``). For example, ``top_n=3``
        would keep only the 3 best rows when
        ``axis=0`` and discarding the rest.
    copy : bool, default=True
        If ``True``, return a new DataFrame
        without modifying the original. If
        ``False``, sort the data in place.
        (Note that in-place sorting is only
        valid if ``data`` is already a
        DataFrame. Otherwise, a new DataFrame
        will still be returned.)

    Returns
    -------
    pd.DataFrame
        A DataFrame sorted along the given axis
        according to the specified aggregation
        method.

    Examples
    --------
    >>> import numpy as np 
    >>> from gofast.utils.extension import reorder_importances 
    >>> arr = np.array([[0.4, 0.3], [0.2, 0.8], [0.7, 0.1]])
    >>> reorder_importances(arr, axis=0, method="sum")
           0    1
    2    0.7  0.1
    0    0.4  0.3
    1    0.2  0.8

    >>> df = pd.DataFrame({
    ...     "ModelA": [0.2, 0.4, 0.1, 0.3],
    ...     "ModelB": [0.3, 0.1, 0.4, 0.2]},
    ...     index=["f1","f2","f3","f4"])
    >>> reorder_importances(df, axis=0, method="mean", top_n=2)
         ModelA  ModelB
    f2      0.4     0.1
    f1      0.2     0.3

    Notes
    -----
    - When a callable is used for ``method``, it
      is applied to each 1D slice (row or column).
    - By default, sorting is descending to quickly
      see the "largest" importances.
    """

    # 1) Convert data to a DataFrame if needed
    if isinstance(data, pd.DataFrame):
        df = data if not copy else data.copy()
    elif isinstance(data, pd.Series):
        df = data.to_frame()
        if not copy:
            # Series doesn't truly allow "in-place" for sorting by aggregates
            # Return a new DataFrame anyway
            pass
    else:
        # assume NumPy array or array-like
        df = pd.DataFrame(data)

    # 2) Choose aggregator
    if isinstance(method, str):
        if method not in ("sum", "mean", "max"):
            raise ValueError(
                "method must be one of {'sum', 'mean', 'max'} "
                "or a callable."
            )
        agg_fn = getattr(np, method)
    elif callable(method):
        agg_fn = method
    else:
        raise TypeError(
            "method should be a string ('sum', 'mean', 'max') "
            "or a callable."
        )

    # 3) Compute the aggregation on the axis
    #    e.g. if axis=0, we look across columns to get row-based metric
    #    if axis=1, we look across rows to get column-based metric
    #    We'll create a series with that aggregator:
    if axis == 0:
        metric = df.apply(agg_fn, axis=1)
        # Sort row labels by that metric
        sorted_idx = metric.sort_values(ascending=ascending).index
        df = df.loc[sorted_idx]
    elif axis == 1:
        metric = df.apply(agg_fn, axis=0)
        # Sort column labels
        sorted_idx = metric.sort_values(ascending=ascending).index
        df = df[sorted_idx]
    else:
        raise ValueError("axis must be 0 (rows) or 1 (columns).")

    # 4) Optionally keep only the top_n
    if top_n is not None:
        if axis == 0:
            df = df.head(top_n)
        else:
            df = df.iloc[:, :top_n]
    
    df = to_series(df, handle_2d="passthrough") 

    return df

def spread_coverage(
    pred_q,  
    q="50%",  
    cov_col=None,  
    spread=0.2,  
    use_relative=True,  
    abs_margin=None,  
    forecast_err="symmetric",  
    spread_range=None,  
    perturbation=None,  
    sigma_err=0., 
    return_type="df",
    name=None,  
    nloc="prefix",  
    trailer="_",  
    verbose=0  
):
    r"""
    Spread lower (q10) and upper (q90) predictions from a given set 
    of median predictions (q50) in order to create an approximate 
    prediction interval based on spread value. The function supports
    symmetric as well as asymmetric forecast errors, and allows for 
    optional perturbation and renaming of the output series.

    Parameters
    ----------
    pred_q : array-like
        Input predictions. These values are interpreted as the 
        median predictions (i.e. `q50`) when ``q`` is set to 
        ``"50%"`` (or 0.5). If a pandas DataFrame is provided, then 
        the column specified by `cov_col` is used.
    q : {``"50%"``, ``"10%"``, ``"90%"``, 0.5, 0.1, 0.9}, default 
        ``"50%"``
        Indicates which quantile the input `pred_q` represents. For 
        example, if set to ``"50%"`` or 0.5, `pred_q` is considered 
        the median predictions (q50) from which q10 and q90 are 
        generated. If set to ``"10%"`` (or 0.1), then `pred_q` is 
        assumed to be the 10th quantile and the function computes the 
        corresponding q50 and q90. Similarly, ``"90%"`` (or 0.9) 
        indicates that `pred_q` represents the 90th quantile.
    cov_col : ``str``, optional
        When `pred_q` is provided as a DataFrame, this string specifies 
        the column name to extract the numeric predictions.
    spread : ``float``, default 0.2
        The relative spread used to compute the lower and upper 
        predictions when `use_relative` is True. In this case, the 
        lower prediction is computed as 
        :math:`q50 \\times (1 - \\text{spread})` and the upper as 
        :math:`q50 \\times (1 + \\text{spread})`.
    use_relative : ``bool``, default True
        If True, the function uses the relative spread to compute 
        the quantiles; if False, `spread` is treated as an absolute 
        margin.
    abs_margin : ``float``, optional
        Absolute margin to be used in place of `spread`. If provided, 
        the lower and upper predictions are computed as 
        :math:`q50 - \\text{abs_margin}` and 
        :math:`q50 + \\text{abs_margin}`, respectively.
    forecast_err : ``str``, default ``"symmetric"``
        Specifies the forecast error structure. Acceptable values 
        are ``"symmetric"`` and ``"asymmetric"``. For asymmetric 
        errors, `spread_range` should be provided.
    spread_range : tuple or list of two floats, optional
        For asymmetric forecast errors, specifies the lower and 
        upper spread as 
        ``(spread_low, spread_high)``. These values are used to 
        compute q10 and q90 when `forecast_err` is set to 
        ``"asymmetric"``.
    perturbation : ``float`` or list/tuple of two floats, optional
        A factor to perturb the computed q10 and q90. If a list of two 
        values is provided, the first is applied to q10 and the second 
        to q90. If a single number is provided, it is applied to both.
    sigma_err : float, default 0
        Shuffles a fraction of the computed q10 and q90 values to add 
        error. When ``sigma_err`` is ``0``, no shuffling is applied; 
        when ``sigma_err`` is ``1``, all q10 and q90 values are randomly 
        shuffled; when ``sigma_err`` is, for example, ``0.3``, then 
        30% of the predictions are randomly shuffled while the 
        remaining 70% retain their original order.
        
    return_type : ``str``, default ``"df"``
        Specifies the return type. If set to ``"df"``, the function 
        returns a pandas DataFrame with columns `q10`, `q50`, and 
        `q90`. Otherwise, it returns a tuple of NumPy arrays 
        `(q10, q50, q90)`.
    name : ``str``, optional
        An optional string to rename the output columns. For example, 
        if `name` is ``"sale"``, the columns might be renamed to 
        ``sale_q10``, ``sale_q50``, and ``sale_q90`` (depending on 
        `nloc` and `trailer`).
    nloc : ``str``, default ``"prefix"``
        Specifies whether the `name` should be used as a prefix or 
        suffix in the output column names. Valid options are 
        ``"prefix"`` and ``"suffix"``.
    trailer : ``str``, default ``"_"``
        A string used as a separator between `name` and the quantile 
        label in the output column names.
    verbose : ``int``, default 0
        Verbosity level; higher values will print debug information 
        during processing.

    Returns
    -------
    df : pandas.DataFrame
        If `return_type` is set to ``"df"``, a DataFrame with columns 
        `q10`, `q50`, and `q90` is returned.
    or
    tuple of np.ndarray
        A tuple `(q10, q50, q90)` is returned if `return_type` is not 
        ``"df"``.

    Formulation
    ------------
    Let :math:`q_{50,i}` be the median prediction for observation 
    :math:`i`, and let :math:`f_i` be defined by:

    .. math::
       f_i = \\begin{cases}
         \\text{spread} & \\text{if } \\text{forecast_err} = 
         \\text{"symmetric"} \\\\
         \\text{spread_low} & \\text{for asymmetric errors (lower)} \\\\
         \\text{spread_high} & \\text{for asymmetric errors (upper)}
       \\end{cases}

    Then, for the symmetric case,
    
    .. math::
       q_{10,i} = q_{50,i} \\times (1 - f_i) \\quad \\text{and} \\quad 
       q_{90,i} = q_{50,i} \\times (1 + f_i).

    For absolute margins, these become:

    .. math::
       q_{10,i} = q_{50,i} - \\text{abs_margin} \\quad \\text{and} \\quad
       q_{90,i} = q_{50,i} + \\text{abs_margin}.

    If perturbation is applied, then:
    
    .. math::
       q_{10,i} \\text{ is perturbed by } (1 - p_1) \\quad \\text{and} \\quad
       q_{90,i} \\text{ is perturbed by } (1 + p_2),
    
    where :math:`p_1` and :math:`p_2` are the perturbation factors.

    Examples
    --------
    1) **Basic usage with relative spread:**

       >>> from gofast.utils.extension import spread_coverage
       >>> import numpy as np
       >>> q50 = np.array([100, 150, 200])
       >>> df = spread_coverage(q50, q="50%", spread=0.2)
       >>> df
           q10    q50    q90
       0  80.0  100.0  120.0
       1 120.0  150.0  180.0
       2 160.0  200.0  240.0

    2) **Using an absolute margin:**

       >>> df = spread_coverage(q50, q="50%", 
       ...          abs_margin=20, use_relative=False)
       >>> df
           q10    q50    q90
       0  80.0  100.0  120.0
       1 130.0  150.0  170.0
       2 180.0  200.0  220.0

    3) **Returning a tuple and applying perturbation:**

       >>> q10, q50_arr, q90 = spread_coverage(
       ...         q50, q="50%", spread=0.1, 
       ...         perturbation=[0.01, 0.03], 
       ...         return_type="tuple")
       >>> q10
       array([ 89., 134., 179.])
       >>> q50_arr
       array([100., 150., 200.])
       >>> q90
       array([111., 164., 221.])

    Notes
    -----
    This function uses the inline method `assert_ratio` to standardize 
    the input quantile ``q``. The parameters `spread` and 
    `abs_margin` control the width of the prediction interval. The 
    optional `perturbation` parameter allows further fine-tuning of the 
    lower and upper predictions. The output columns can be renamed by 
    specifying `name`, `nloc`, and `trailer`. This implementation 
    assumes a symmetric error distribution by default [1]_.

    See Also
    --------
    gofast.core.checks.check_numeric_dtype` :
        Validates numeric data types in arrays.

    References
    ----------
    .. [1] Armstrong, J. S. "Prediction intervals." 
           *International Journal of Forecasting* 16.4 (2000): 521-530.
    """

    def log(level, msg):
        # Simple logger that prints messages if verbose is high enough.
        if verbose >= level:
            print(msg)

    # If pred_q is a DataFrame, extract the specified column.
    if isinstance(pred_q, pd.DataFrame):
        if cov_col is None:
            raise ValueError(
                "cov_col must be provided when pred_q is a DataFrame."
            )
        pred_q = pred_q[cov_col]

    pred_q= check_numeric_dtype(
        pred_q, ops="validate", 
        coerce=True, 
        param_names ={"X":"pred_q"}
    )
    # Convert pred_q to a NumPy array of floats.
    pred_q = np.asarray(pred_q, dtype=float)
    
    # Standardize q: convert a string like "50%" to 0.5 using the 
    # inline method assert_ratio (assumed imported).
    q = assert_ratio(q, bounds=(0, 1), 
                     exclude_values=[0, 1], 
                     name="Quantile q")
    log(3, f"Processing q={q}, forecast_err={forecast_err}")

    # Determine spread values based on forecast_err.
    if forecast_err == "symmetric":
        spread_low  = spread_high = spread
    elif forecast_err == "asymmetric":
        if spread_range is None:
            log(2, "spread_range not provided, setting default asymmetric spread.")
            spread_low, spread_high = 0.1, 0.2  # Default values.
        elif (isinstance(spread_range, (tuple, list)) and 
              len(spread_range) == 2):
            
            spread_range = validate_length_range(
                spread_range, sorted_values=False, 
                param_name="spread_range" 
            )
            spread_low, spread_high = spread_range
        else:
            raise ValueError(
                "spread_range must be a tuple (spread_low, spread_high) "
                "for asymmetric forecast errors."
            )
    else:
        raise ValueError(
            "Invalid value for `forecast_err`. Choose 'symmetric' or "
            "'asymmetric'."
        )
    log(3, f"Spread values - Low: {spread_low}, High: {spread_high}")

    # Define pred_q50, pred_q10, and pred_q90 based on the input quantile.
    if q == 0.5:
        pred_q50 = pred_q
        pred_q10 = (pred_q50 * (1 - spread_low)
                    if use_relative else pred_q50 - spread_low)
        pred_q90 = (pred_q50 * (1 + spread_high)
                    if use_relative else pred_q50 + spread_high)
    elif q == 0.1:
        pred_q10 = pred_q
        pred_q50 = (pred_q10 / (1 - spread_low)
                    if use_relative else pred_q10 + spread_low)
        pred_q90 = (pred_q50 * (1 + spread_high)
                    if use_relative else pred_q50 + spread_high)
    elif q == 0.9:
        pred_q90 = pred_q
        pred_q50 = (pred_q90 / (1 + spread_high)
                    if use_relative else pred_q90 - spread_high)
        pred_q10 = (pred_q50 * (1 - spread_low)
                    if use_relative else pred_q50 - spread_low)
    else:
        raise ValueError(
            "Invalid value for `q`. Must be '50%', '10%', '90%' or their "
            "numerical equivalents."
        )

    # Override with absolute margin if provided.
    if abs_margin is not None:
        pred_q10 = pred_q50 - abs_margin
        pred_q90 = pred_q50 + abs_margin

    # Apply perturbations if specified.
    if perturbation is not None:
        if isinstance(perturbation, (list, tuple)) and len(perturbation) == 2:
            perturbation = validate_length_range(
                perturbation, param_name="Perturbation", 
                sorted_values=False, 
            )
            p1, p2 = perturbation
        elif isinstance(perturbation, (int, float)):
            p1 = p2 = perturbation
        else:
            raise ValueError(
                "`perturbation` should be a float or a list of two floats."
            )
        pred_q10 *= 1 - p1
        pred_q90 *= 1 + p2

    log(3, f"Final predictions - q10: {pred_q10}, q50: {pred_q50}, q90: {pred_q90}")

    # Apply sigma shuffling
    if sigma_err:
        sigma_err=assert_ratio(sigma_err, bounds=(0, 1), name="sigma_err")
        n = len(pred_q50)
        num_shuffle = int(np.ceil(sigma_err * n))
        indices = np.arange(n)
        shuffle_idx = np.random.choice(
            indices, size=num_shuffle, replace=False
        )
        q10_temp = np.copy(pred_q10[shuffle_idx])
        q90_temp = np.copy(pred_q90[shuffle_idx])
        np.random.shuffle(q10_temp)
        np.random.shuffle(q90_temp)
        pred_q10[shuffle_idx] = q10_temp
        pred_q90[shuffle_idx] = q90_temp
    
    _return_type = return_type 
    if _return_type =="series": 
        return_type ="df"
        
    # Construct output: return a DataFrame or a tuple.
    if return_type == "df":
        output = pd.DataFrame({
            "q10": pred_q10,
            "q50": pred_q50,
            "q90": pred_q90
        })
        # Rename columns if a name is provided.
        if name:
            if nloc == "prefix":
                output.columns = [f"{name}{trailer}{col}" 
                                  for col in output.columns]
            elif nloc == "suffix":
                output.columns = [f"{col}{trailer}{name}" 
                                  for col in output.columns]
                
        if _return_type=="series": 
            # Extract each column as an individual Series and return them.
            return (output.iloc[:, 0], output.iloc[:, 1], output.iloc[:, 2])
    
        return output
    
    return pred_q10, pred_q50, pred_q90

@isdf
def parse_used_columns(
    df,
    expr
):
    """
    Extracts a list of DataFrame columns that appear 
    in the given expression (``expr``).

    For each column name, it constructs a pattern that accounts
    for bracket notation and standalone references. Under the
    hood, it uses Python's built-in regular expressions, matching 
    word boundaries or bracketed occurrences. The extracted 
    columns are returned as a list, ensuring duplicates are 
    removed. If the ``expression`` is invalid or not a string, an 
    empty list is returned.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame whose columns are examined.
    expr : str
        The string expression potentially referencing 
        DataFrame columns.

    Returns
    -------
    list of str
        A list of columns found within the given 
        ``expression``.

    Examples
    --------
    >>> from gofast.utils.ext import parse_used_columns
    >>> import pandas as pd
    >>> data = {'A': [1, 2], 'B': [3, 4]}
    >>> df = pd.DataFrame(data)
    >>> parse_used_columns(df, "A + B")
    ['A', 'B']
    >>> parse_used_columns(df, "df['A'] + df['B']")
    ['A', 'B']

    Notes
    -----
    Make sure to pass a valid string ``expression``. If it is 
    empty or None, an empty list is returned. This function uses 
    regex lookups which might not account for all possible edge 
    cases [1]_.
    
    The approach can be summarized as a pattern-matching problem:

    .. math::
       \\text{pattern} = \\text{Regex}(\\text{col}) 
       \\times \\text{boundaries}

    where :math:`boundaries` are word delimiters or bracket 
    notations. This helps prevent partial matches of column 
    substrings.

    See Also
    --------
    evaluate_df : A function that utilizes column detection 
        during expression parsing.

    References
    ----------
    .. [1] McKinney, Wes. *Python for Data Analysis*, 2nd Edition,
       O'Reilly Media, 2017.
    """
    # Return empty list if expression is not a valid string
    if not expr or not isinstance(expr, str):
        return []

    used_cols = set()
    for col in df.columns:
        # Pattern checks for bracket usage ("col" or 'col')
        # or standalone references with negative/positive
        # lookarounds to avoid partial matches.
        pattern = (
            rf"(\[\"{re.escape(col)}\"\]|"
            rf"\['{re.escape(col)}'\]|"
            rf"(?<![\w\"']){re.escape(col)}(?![\w\"']))"
        )
        if re.search(pattern, expr):
            used_cols.add(col)

    return list(used_cols)

@SaveFile
@check_params ({ 
    "new_names": Optional[Union[str, List[str]]], 
    "columns": Optional[Union [str, List[str]]]
 })
@isdf 
def to_micmic(
    df,
    r=0.8,
    eval_metric='r2',
    columns=None,
    sigma_err=0.0,
    drop_origin=False,
    new_names=None,
    epsilon=1e-8,   
    positive_only=False,  
    savefile=None, 
    verbose=0
):
    """
    Creates a "micmic" version of selected numerical columns in
    the input DataFrame based on a user-defined ratio controlling
    an approximate :math:`R^2` or Mean Absolute Error (MAE) metric.
    Generates additional noise and optional post-processing, such
    as enforcing non-negative values.
    
    Parameters
    ------------
    df : pandas.DataFrame
        The input dataframe containing data to be micmic'ed.
    r : float, default 0.8
        The ratio controlling the desired correlation or error
        metric target. When `eval_metric` is `r2`, `r` stands
        for the approximate :math:`R^2` value to achieve. When
        `eval_metric` is `mae`, it represents the target average
        error.
    eval_metric : str, default 'r2'
        The metric to match. Supported values are `r2` or `mae`.
        If unknown, the function simply copies values.
    columns : str or list of str, optional
        Column name(s) to process. If None, automatically picks
        numeric columns. A single string can be provided for
        a single column.
    sigma_err : float, default 0.0
        Fraction of the resulting micmic values to reshuffle
        within the :math:`Q_{10}`–:math:`Q_{90}` range for added
        unpredictability. For example, ``sigma_err=0.3`` replaces
        30% of the new column values randomly in that range.
    drop_origin : bool, default False
        If True, drops the original columns being micmic'ed and
        retains only the new micmic columns.
    new_names : str or list of str, optional
        The names for the newly created micmic columns. If None,
        defaults to adding a ``mic_`` prefix to each column name.
    epsilon : float, default 1e-8
        A small constant to avoid division-by-zero issues in
        variance or error calculations.
    positive_only : bool, default False
        If True, micmic values are clipped at 0, so no negative
        values appear in the output.
    savefile : str, optional
        If provided, the resulting DataFrame can be saved to
        the specified file path (e.g., CSV). If None, nothing
        is saved.
    verbose : int, default 0
        Controls the verbosity of the console output:
        - `verbose=0` gives no additional messages
        - `verbose=1` prints a summary message
        - `verbose=2` prints intermediate column details
        - `verbose>=3` may offer extra debugging messages
    
    Return
    --------
    new_df: DataFrame 
       New micmic dataframe.
       
    Formulation
    ------------
    When `eval_metric` is `r2`, the function attempts to satisfy
    the relation:
    
    .. math::
       r = \\frac{\\mathrm{Var}(X)}
       {\\mathrm{Var}(X) + \\mathrm{Var}(\\eta)},
    
    where :math:`X` is the original column and
    :math:`\\eta \\sim \\mathcal{N}(0, \\sigma^2)` is the added
    noise. Solving for the variance of the noise
    :math:`\\mathrm{Var}(\\eta)` yields:
    
    .. math::
       \\mathrm{Var}(\\eta) = \\mathrm{Var}(X) \\times
       \\frac{1 - r}{r}.
    
    For `mae`, the function relies on a normal distribution
    where the expected absolute deviation is set to match `r`.
    
    Examples
    --------
    >>> from gofast.utils.ext import to_micmic
    >>> # Create a micmic column for 'price' with ~0.8 R^2
    >>> new_df = to_micmic(
    ...     df,
    ...     r=0.8,
    ...     eval_metric='r2',
    ...     columns='price',
    ...     sigma_err=0.2,
    ...     positive_only=True,
    ...     verbose=2
    ... )
    Column: price, New Column: mic_price
    Micmic transformation complete.
    
    Notes
    -----
    - When `positive_only` is True, all negative results
      are clipped to 0.
    - If the original column is constant (or near-constant
      within ``epsilon``), the function simply copies
      the values, as noise-based transformations do not
      apply well in degenerate variance conditions.
    - Using a large value for `sigma_err` may drastically
      reshuffle the distribution of the micmic column.
    
    See Also
    --------
    No additional public functions in this module reference
    :func:`to_micmic`, but related data transformation routines
    can be built upon this approach.
    
    References
    ----------
    .. [1] Geladi, P. & Kowalski, B. (1986). Partial least-squares
       regression: A tutorial. Analytica Chimica Acta, 185, 1–17.
    """

    r= assert_ratio(r, bounds=(0, 1), name="likehood ratio 'r'")
    
    # pick numeric columns if none provided
    columns = columns_manager(columns, empty_as_none=True)
    if columns is None:
        columns = select_dtypes(
            df, incl= np.number, return_columns=True)
        
    # Safeguard: remove any columns not in df
    columns = [c for c in columns if c in df.columns]
    
    if not columns: 
        warnings.warn (
            "No numeric columns detected, return data as is."
            )
        return df 
    
    # Check whether the columns provided are numerics.
    check_numeric_dtype(
        df[columns], param_names={"X": f"Data: columns {columns}"}
    )
    # Prepare container for new columns
    new_df = df.copy(deep=True)

    # Handle new column naming
    if new_names is None:
        # Default: append "mic_" to each column name
        new_names = [f"mic_{c}" for c in columns]
    elif isinstance(new_names, str):
        # If a single string is provided for one column
        new_names = [new_names]
    new_names = columns_manager(new_names, empty_as_none=False) 
    
    # assert new_names is a list of the correct length
    if len(new_names) < len(columns): 
        new_names +=[f"mic_{c}" for c in columns[len(new_names):]]
    elif len(new_names )> len(columns): 
        new_names = new_names [:len(columns)]
        
    # Loop over each specified column
    for col_idx, col in enumerate(columns):
        # Original values
        original_vals = df[col].values.astype(float)

        # Decide how to build new values based on eval_metric
        if eval_metric.lower() == 'r2':
            # "R²" approach:
            # new_val = original_val + noise, so that theoretical R² ~ r
            var_o = np.var(original_vals)
            # Ensure we have a minimum variance
            if var_o <= epsilon:
                # Degenerate case: near-constant column
                new_col = original_vals.copy()
            else:
                # R² formula derivation:
                # r = var_o / (var_o + var_noise)
                # var_noise = var_o * (1 - r) / r
                if r < 1:
                    var_noise = var_o * (1 - r) / r
                else:
                    var_noise = 0.0
                noise_std = np.sqrt(var_noise) if var_noise > 0 else 0
                noise = np.random.normal(
                    loc=0.0,
                    scale=noise_std,
                    size=len(original_vals)
                )
                new_col = original_vals + noise

        elif eval_metric.lower() == 'mae':
            # "MAE" approach:
            # new_val = original_val + e, E[|e|] ~ r
            # For Normal(0, σ), E[|Z|] = σ * sqrt(2/π)
            if r <= 0:
                new_col = original_vals.copy()
            else:
                sigma = r / np.sqrt(2 / np.pi)
                e = np.random.normal(
                    loc=0.0,
                    scale=sigma,
                    size=len(original_vals)
                )
                new_col = original_vals + e

        else:
            # Fallback: if unknown metric, just copy
            new_col = original_vals.copy()

        # Shuffle a fraction "sigma_err" 
        # in Q10-Q90 range for extra noise
        if sigma_err > 0:
            n_to_shuffle = int(len(new_col) * sigma_err)
            shuffle_idx = np.random.choice(
                len(new_col),
                size=n_to_shuffle,
                replace=False
            )
            # We'll replace those entries with random values
            # in the [q10, q90] range
            q10, q90 = np.percentile(new_col, [10, 90])
            new_col[shuffle_idx] = np.random.uniform(
                low=q10,
                high=q90,
                size=n_to_shuffle
            )

        # Enforce positivity if requested
        if positive_only:
            new_col = np.clip(new_col, 0, None)

        # Insert the new column into the DataFrame
        new_col_name = new_names[col_idx]
        new_df[new_col_name] = new_col

        # Possibly drop the original column
        if drop_origin:
            new_df.drop(columns=[col], inplace=True)

        # Verbose feedback
        if verbose >= 2:
            vlog(f"Column: {col}, New Column: {new_col_name}")

    # Final verbosity
    if verbose >= 1:
        vlog("Micmic transformation complete.")

    return new_df

@SaveFile
@check_params ({ 
    "column": Optional[Union [str, List[str]]]
 })
@isdf 
def spread_uncertainty(
    df,
    column=None,
    sigma_vals=None,
    order='descending',
    epsilon=1e-8,
    sigma_err=0., 
    new_names=None,
    positive_only=False, 
    drop_origin=False,
    savefile=None, 
    verbose=0
):
    """
    Generates a set of "uncertainty" columns from a given numeric
    column based on a list of ratio-like values (``sigma_vals``). The
    function internally calls `to_micmic` with ``eval_metric='r2'``
    to produce new columns that mimic the original column yet
    introduce controlled noise according to user-defined targets.
    
    Parameters
    ----------
    df : pandas.DataFrame or pandas.Series
        The data source. If `df` is a DataFrame, then the
        specified `column` will be used. If `df` is a Series,
        the series name is taken as the reference column
        unless it is ``None``, in which case it defaults to
        ``'reference'``.
    column : str, optional
        The column name to which uncertainties will be
        applied. Not required if `df` is a Series. Must
        be numeric if provided.
    sigma_vals : float or list of floats, optional
        A scalar or list of ratio-like values that define
        the noise intensities to introduce. These ratios,
        :math:`\\rho`, are interpreted as approximate
        :math:`R^2` objectives in the internal call to
        ``to_micmic``. For multiple values, multiple
        new columns are created accordingly.
    order : str, default 'descending'
        Controls the sorting order of `sigma_vals`. Accepted
        values are `'ascending'` or `'descending'`. If set
        to `'descending'`, the highest ratio is processed
        first.
    epsilon : float, default 1e-8
        A small value to avoid division by zero in variance
        calculations and related numeric steps.
    sigma_err : float, default 0.0
        The fraction of generated values to reshuffle within
        the :math:`Q_{10}`–:math:`Q_{90}` range, offering
        further randomness in the final columns. For example,
        a value of ``0.3`` randomly reassigns 30% of the new
        column entries.
    new_names : str or list of str, optional
        Names for the new columns. If insufficient names are
        provided, default naming is applied. If more names
        than `sigma_vals` exist, extras are discarded.
    positive_only : bool, default False
        If True, micmic values are clipped at 0, so no negative
        values appear in the output.
    drop_origin : bool, default False
        If True, removes the source column from the output
        DataFrame after creating the new uncertainty columns.
    savefile : str, optional
        If provided, the resulting DataFrame can be saved to
        the specified file path (e.g., CSV). If None, nothing
        is saved.
    verbose : int, default 0
        Controls how much logging is printed to the console.
        
    Return 
    --------
    result_df: pd.DataFrame 
        New data with uncertainties generated. 
        
    Formulation 
    -------------
    Each value in ``sigma_vals`` targets the relation:
    
    .. math::
       \\rho = \\frac{\\mathrm{Var}(X)}
       {\\mathrm{Var}(X) + \\mathrm{Var}(\\eta)},
    
    where :math:`X` is the original reference column and
    :math:`\\eta` is a noise term drawn from a normal distribution.
    Thus, the variance of the noise is determined by:
    
    .. math::
       \\mathrm{Var}(\\eta) = \\mathrm{Var}(X) \\times
       \\frac{1 - \\rho}{\\rho}.
    
    Examples
    --------
    >>> from gofast.utils.ext import spread_uncertainty
    >>> import pandas as pd 
    >>> df = pd.DataFrame ({"height": [175, 125, 150, 180, 173.5], 
    ...                        "weight": [65, 70, 75, 62, 69.5]})
    >>> result = spread_uncertainty(
    ...     df,
    ...     column='height',
    ...     sigma_vals=[0.7, 0.4, 0.2],
    ...     order='descending',
    ...     drop_origin=False,
    ...     sigma_err=0.2,
    ...     verbose=1
    ... )
    Created 3 new columns with uncertainties.
    
    Notes
    -----
    - For every value in `sigma_vals`, one new column is added
      to the result. Hence, providing multiple values creates
      multiple columns.
    - This function uses `to_micmic` internally with
      ``eval_metric='r2'`` to achieve the approximate ratio
      :math:`\\rho`.
    - If `drop_original=True`, the original column specified
      by `column` is removed from the output.
    
    See Also
    --------
    to_micmic : Generates micmic columns using various metrics
      (like R² or MAE).
    
    References
    ----------
    .. [1] Draper, N.R. and Smith, H. (1998). Applied Regression
       Analysis (3rd ed.). Wiley.
    """

    # Handle default uncertainities
    sigma_vals= columns_manager(sigma_vals, empty_as_none=True)
    if sigma_vals is None:
        sigma_vals = [0.8]
    
    # Ensure it's a list
    sigma_vals = [
        assert_ratio(
            sv, bounds=(0, 1.), 
            name="Uncertainty values 'sigma'"
            )
        for sv in sigma_vals 
    ]
    # Sort the uncertainties according to 'order'
    if order.lower() == 'descending':
        sigma_vals.sort(reverse=True)
    else:
        sigma_vals.sort()
    
    if isinstance (column, (list, tuple)): 
        column = column[0] 
    # Check if df is a Series
    if isinstance(df, pd.Series):
        # Convert to DataFrame if needed
        col_name = df.name if df.name else 'reference'
        temp_df = df.to_frame(name=col_name)
    else:
        temp_df = df.copy(deep=True)
        col_name = column
    
    # Basic checks
    if col_name is None or col_name not in temp_df.columns:
        raise ValueError(
            "A valid column name must be provided, or df "
            "must be a Series with a non-empty name."
        )
    
    # Prepare new DataFrame
    result_df = temp_df.copy(deep=True)
    
    # Build default new column names if needed
    if new_names is None:
        new_names = [f"uncert_{column}_{v}" for v in sigma_vals]
    elif isinstance(new_names, str):
        new_names = [new_names]
    else:
        new_names = list(new_names)
    
    # If new_names is fewer than the uncertainties, fill up
    # If more, truncate
    if len(new_names) < len(sigma_vals):
        # Extend with default naming
        deficit = len(sigma_vals) - len(new_names)
        auto_names = [
            f"uncert_{v}"
            for v in sigma_vals[deficit:]
        ]
        new_names += auto_names
    elif len(new_names) > len(sigma_vals):
        # Truncate
        new_names = new_names[: len(sigma_vals)]
    
    # For each uncertainty, call to_micmic with eval_metric='r2'
    # to generate new columns
    for idx, val in enumerate(sigma_vals):
        mic_df = to_micmic(
            df=result_df[[col_name]],
            r=val,
            eval_metric='r2',
            columns=col_name,
            sigma_err=sigma_err,
            drop_origin=False,
            new_names=new_names[idx],
            epsilon=epsilon,
            positive_only=positive_only,
            verbose=0
        )
        # Extract the newly created column
        result_df[new_names[idx]] = mic_df[new_names[idx]]
    
    # Finally, decide whether to drop the original column
    if drop_origin and (col_name in result_df.columns):
        result_df.drop(columns=[col_name], inplace=True)
    
    # Verbose feedback
    if verbose:
        vlog(f"Created {len(sigma_vals)} new columns "
              f"with uncertainties.", 
              )
    
    return result_df
