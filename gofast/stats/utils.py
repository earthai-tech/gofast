# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""Provides utility functions to support statistical analyses, including 
dataset preprocessing for repeated measures ANOVA."""

import warnings
from itertools import product
import numpy as np
import pandas as pd
from ..api.types import Optional, List, Union, Tuple, Callable, Any
from ..api.types import DataFrame, ArrayLike
from ..core.utils import normalize_string, smart_format
from ..core.io import SaveFile 
from ..decorators import isdf 


__all__=["fix_rm_anova_dataset", "cumulative_ops" ]

def prepare_stats_plot(
    values: ArrayLike,
    data: Optional[ArrayLike] = None,
    axis: Optional[int] = None,
    transform: Optional[Callable[[Any], Any]] = None
) -> Tuple[Any, Any, Optional[Union[pd.Index, range]]]:
    """
    Prepares data and its labels for plotting, handling different data structures
    and orientations based on the specified axis.

    Parameters
    ----------
    values : Union[np.ndarray, pd.Series, pd.DataFrame]
        The data values to be plotted, which can be the result of statistical
        operations like mean, median, etc.
    data : Optional[Union[np.ndarray, pd.Series, pd.DataFrame]], optional
        Additional data related to `values`, used for plotting. Defaults to None.
    axis : Optional[int], optional
        The axis along which to organize the plot data. Can be None, 0, or 1.
        Defaults to None.
    transform : Optional[Callable[[Any], Any]], optional
        A function to apply to the data before plotting. Defaults to None.

    Returns
    -------
    Tuple[Any, Any, Optional[Union[pd.Index, range]]]
        A tuple containing the transformed `values`, `data`, and the labels
        for plotting.

    Examples
    --------
    >>> import pandas as pd 
    >>> import numpy as np 
    >>> from gofast.stats.utils import prepare_stats_plot
    >>> values = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
    >>> prepare_stats_plot(values)
    (Series([1, 2, 3], index=['a', 'b', 'c']), None, Index(['a', 'b', 'c']))

    >>> values = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, index=['x', 'y'])
    >>> prepare_stats_plot(values, axis=1)
    (Transposed DataFrame, Transposed DataFrame, Index(['x', 'y']))

    >>> values = np.array([1, 2, 3])
    >>> prepare_stats_plot(values, transform=np.square)
    (array([1, 4, 9]), None, range(0, 3))
    """
    if transform is not None and callable(transform):
        values = transform(values)
        if data is not None:
            data = transform(data)

    if axis is None:
        return values, data, None

    if isinstance(values, pd.Series):
        data= data.T if axis ==1 else data  
        return values.T, data if data is not None else None, values.index

    elif isinstance(values, pd.DataFrame):
        if axis == 0:
            return values, data, values.columns
        elif axis == 1:
            return values.T, data.T if data is not None else None, values.index

    elif isinstance(values, np.ndarray):
        values = np.squeeze(values)
        if values.ndim == 1:
            return values, data, range(len(values))
        if axis == 0:
            return values, data, range(values.shape[1])
        elif axis == 1:
            return values.T, data.T if data is not None else None, range(values.shape[0])

    return values, data, None

def validate_stats_plot_type(
        type_: str,
        target_strs: List[str],
        match_method: str = 'contains',
        raise_exception: bool = False,
        **kwargs) -> Optional[str]:
    """
    Validates the plot type against a list of acceptable types and returns
    the normalized matching string.

    This function checks if the given plot type matches any of the target strings
    based on the specified match method. If a match is found, the function returns
    the normalized string from the target list. It can optionally raise an exception
    if no match is found.

    Parameters
    ----------
    type_ : str
        The plot type to validate.
    target_strs : List[str]
        A list of acceptable plot type strings to match against.
    match_method : str, default 'contains'
        The method used to match the plot type with the target strings. Options include:
        - 'contains': Checks if `type_` is contained within any of the target strings.
        - 'exact': Checks for an exact match between `type_` and the target strings.
        - 'startswith': Checks if `type_` starts with any of the target strings.
    raise_exception : bool, default False
        If True, raises a ValueError when no match is found. Otherwise, returns None.
    **kwargs : dict
        Additional keyword arguments to be passed to the `normalize_string` function.

    Returns
    -------
    Optional[str]
        The normalized string from `target_strs` that matches `type_`, or None if
        no match is found and `raise_exception` is False.

    Raises
    ------
    ValueError
        If `raise_exception` is True and no match is found.

    Examples
    --------
    >>> from gofast.stats.utils import validate_stats_plot_type
    >>> validate_stats_plot_type('box', ['boxplot', 'histogram', 'density'],
    ...                        match_method='startswith')
    'boxplot'

    >>> validate_stats_plot_type('exact_plot', ['boxplot', 'histogram', 'density'],
    ...                        match_method='exact')
    None

    >>> validate_stats_plot_type('hist', ['boxplot', 'histogram', 'density'], 
    ...                        match_method='contains')
    'histogram'

    >>> validate_stats_plot_type('unknown', ['boxplot', 'histogram', 'density'],
    ...                        raise_exception=True)
    ValueError: Plot type 'unknown' is not supported.

    Note
    ----
    This utility function is designed to help in functions or methods where plot type
    validation is necessary, improving error handling and user feedback for plotting
    functionalities.
    """
    matched_type = normalize_string(
        type_, target_strs=target_strs,
        return_target_str=False, match_method=match_method,
        return_target_only=True, **kwargs)

    if matched_type is None and raise_exception:
        raise ValueError(
            f"Unsupported type '{type_}'. Expect {smart_format(target_strs)}.")

    return matched_type

def fix_rm_anova_dataset(
    data: DataFrame, 
    depvar: str, 
    subject: str, 
    within: List[str], 
    strategy: str = "mean", 
    fill_value: Optional[Union[str, float, int]] = None
) -> DataFrame:
    """
    Generate all possible combinations of within-subject factors and fill missing 
    depvar values based on the specified strategy.

    Parameters
    ----------
    data : DataFrame
        The dataset to be processed.
    depvar : str
        The dependent variable whose missing values need to be filled.
    subject : str
        The subject column in the dataset.
    within : List[str]
        A list of columns representing within-subject factors.
    strategy : str, optional
        The strategy to use for filling missing depvar values. Options are "mean",
        "median", or None. Default is "mean".
    fill_value : Optional[Union[str, float, int]], optional
        A specific value to fill missing depvar values if the strategy is None.
        Default is None, which leaves missing values as None.

    Returns
    -------
    DataFrame
        The modified dataset with missing combinations filled.
    """
    if not isinstance (data, pd.DataFrame): 
        raise TypeError("First positionnal argument `data` expects a DataFrame. Got"
                        f" {type(data).__name__!r}")
    all_combinations = list(product(*[data[factor].unique() for factor in within]))
    fixed_data = []

    for subj in data[subject].unique():
        subj_data = data[data[subject] == subj].copy()
        if strategy == "mean":
            fill = subj_data[depvar].mean()
        elif strategy == "median":
            fill = subj_data[depvar].median()
        elif fill_value is not None:
            fill = fill_value if strategy is None else None
        else:
            fill = None
        for combination in all_combinations:
            if combination not in list(zip(*[subj_data[factor].values for factor in within])):
                new_row = {subject: subj, depvar: fill}
                new_row.update(dict(zip(within, combination)))
                subj_data = pd.concat([subj_data, pd.DataFrame([new_row])], ignore_index=True)

        fixed_data.append(subj_data)

    return pd.concat(fixed_data, ignore_index=True)

@SaveFile (dout ='.csv')
@isdf 
def cumulative_ops(
    df: pd.DataFrame,
    checkpoint: Optional[Union[str, int]]=None,
    ops: str = 'sum', 
    reverse: bool = False, 
    direction: str = 'left_to_right',  
    error: str = 'raise' , 
    savefile: Optional[str]=None, 
) -> pd.DataFrame:
    r"""
    Perform Cumulative and De-cumulative Operations on a DataFrame.

    The `cumulative_ops` function allows users to perform cumulative 
    operations such as `sum` or `product` on numeric columns of a 
    Pandas DataFrame, starting from a specified checkpoint. Additionally, 
    it supports de-cumulative (reverse) operations, which assume that the 
    data is already cumulative and reverses the operation accordingly. 
    The function is flexible, allowing operations to be performed in 
    either left-to-right or right-to-left directions based on the 
    checkpoint provided.

    The categorization and manipulation of the DataFrame are based on 
    the specified `checkpoint`, which can be either a column name or 
    its index. The function intelligently handles numeric and 
    categorical columns, ensuring that only relevant data is 
    processed while maintaining the integrity of categorical data.

    .. math::
        \text{Operation} = \begin{cases}
            \text{Sum}, & \text{if operation is 'sum'} \\
            \text{Product}, & \text{if operation is 'product'}
        \end{cases}

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame on which cumulative or de-cumulative operations 
        will be performed.

    checkpoint : Union[str, int], optional 
        The reference point from which the cumulative operation begins. This 
        can be specified either by the column name (`str`) or by the column 
        index (`int`). All operations will proceed from this checkpoint 
        towards the specified direction. If ``None``, the first column of 
        the dataframe is used instead.

    ops : str, default='sum'
        The type of cumulative operation to perform. 
        
        - ``'sum'``: Performs a cumulative sum across the specified direction.
        - ``'product'``: Performs a cumulative product across the specified
          direction.
        
        Raises a `ValueError` if an unsupported operation is provided.

    reverse : bool, default=False
        Indicates whether to perform a de-cumulative (reverse) operation. 
        
        - If ``False``, the function performs the specified cumulative operation.
        - If ``True``, it assumes that the data is already cumulative and reverses 
          the operation to retrieve the original values.
          
        This parameter effectively toggles between cumulative and de-cumulative 
        processing.

    direction : str, default='left_to_right'
        The direction in which the cumulative operation is applied. 
        
        - ``'left_to_right'``: Operations are performed from left to right 
          (e.g., from column A to column D).
        - ``'right_to_left'``: Operations are performed from right to left 
          (e.g., from column D to column A).
        
        Raises a `ValueError` if an unsupported direction is provided.

    error : str, default='raise'
        Defines how the function handles errors encountered during processing.
        
        - ``'warn'``: Issues a warning and continues execution.
        - ``'raise'``: Raises an exception and halts execution.
        
        Raises a `ValueError` if an unsupported error handling strategy is 
        provided.

    Returns
    -------
    pandas.DataFrame
        The original DataFrame augmented with the results of the cumulative 
        or de-cumulative operations. Numeric columns are modified based on 
        the specified parameters, while categorical columns remain unchanged.

    Raises
    ------
    ValueError
        - If `ops` is not ``'sum'`` or ``'product'``.
        - If `direction` is neither ``'left_to_right'`` nor ``'right_to_left'``.
        - If `error` is neither ``'warn'`` nor ``'raise'``.
        - If the `checkpoint` index is out of range.
        - If the `checkpoint` column does not exist.
        - If the `checkpoint` column is not numeric.
        - If there are no columns to operate on in the specified direction.

    Notes
    -----
    - The function automatically distinguishes between numeric and categorical 
      columns, performing operations only on numeric data.
    - Categorical columns are preserved and reinserted into the DataFrame 
      in their original order after operations.
    - When performing de-cumulative operations, ensure that the data is indeed 
      cumulative to avoid incorrect results.
    - The `checkpoint` parameter is crucial as it determines the starting 
      point for operations. Selecting an inappropriate checkpoint may lead 
      to unexpected outcomes.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from gofast.stats.utils import cumulative_ops
    
    >>> # Sample DataFrame
    >>> data = {
    ...     'A': [1, 0, 10, 0],
    ...     'B': [1.5, 0.7, 0.5, 0.35],
    ...     'C': [3, 8, 7, 1],
    ...     'D': [10, 12, 13, 13]
    ... }
    >>> df = pd.DataFrame(data)
    >>> df

        A     B  C   D
    0   1  1.50  3  10
    1   0  0.70  8  12
    2  10  0.50  7  13
    3   0  0.35  1  13
    
    >>> # Perform cumulative sum from column 'A' to 'D'
    >>> df_cum = cumulative_ops(
    ...     df=df,
    ...     checkpoint='A',
    ...     ops='sum',
    ...     reverse=False,
    ...     direction='left_to_right',
    ...     error='warn'
    ... )
    >>> df_cum

        A      B      C      D
    0   1   2.50   5.50  15.50
    1   0   0.70   8.70  20.70
    2  10  10.50  17.50  30.50
    3   0   0.35   1.35  14.35
    
    >>> # Perform de-cumulative operation assuming data is already cumulative
    >>> df_decum = cumulative_ops(
    ...     df=df_cum,
    ...     checkpoint='A',
    ...     ops='sum',
    ...     reverse=True,
    ...     direction='left_to_right',
    ...     error='warn'
    ... )
    >>> df_decum

        A     B    C     D
    0   1  1.50  3.0  10.0
    1   0  0.70  8.0  12.0
    2  10  0.50  7.0  13.0
    3   0  0.35  1.0  13.0
    
    >>> # Perform cumulative product from column 'D' to 'A' (right to left)
    >>> df_cum_prod = cumulative_ops(
    ...     df=df,
    ...     checkpoint='D',
    ...     ops='product',
    ...     reverse=False,
    ...     direction='right_to_left',
    ...     error='warn'
    ... )
    >>> df_cum_prod

           A      B   C   D
    0   45.0  45.00  30  10
    1    0.0  67.20  96  12
    2  455.0  45.50  91  13
    3    0.0   4.55  13  13
    
    Notes
    -----
    - **Checkpoint Selection**: The `checkpoint` determines the starting 
      point for operations. For instance, selecting column `'A'` and 
      `direction='left_to_right'` means operations will proceed from 
      `'A'` towards `'D'`.
    - **Reverse Operations**: When `reverse=True`, the function 
      assumes that the data is cumulative and reverses the operation to 
      retrieve original values.
    - **Error Handling**: Setting `error='warn'` will issue warnings for 
      any issues encountered without stopping execution, whereas `error='raise'` 
      will halt execution upon encountering an error.
    - **Categorical Data**: Categorical columns are excluded from 
      operations and preserved in their original order within the DataFrame.
    
    See Also
    --------
    pandas.DataFrame.cumsum : Compute the cumulative sum of DataFrame elements.
    pandas.DataFrame.cumprod : Compute the cumulative product of DataFrame elements.
    pandas.cut : Bin values into discrete intervals.
    pandas.qcut : Quantile-based discretization function.
    
    References
    ----------
    .. [1] Pandas Documentation: pandas.DataFrame.cumsum. 
       https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.cumsum.html
    .. [2] Pandas Documentation: pandas.DataFrame.cumprod. 
       https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.cumprod.html
    .. [3] Pandas Documentation: pandas.cut. 
       https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.cut.html
    .. [4] Pandas Documentation: pandas.qcut. 
       https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.qcut.html
    .. [5] Freedman, D., & Diaconis, P. (1981). On the histogram as a density estimator: 
           L2 theory. *Probability Theory and Related Fields*, 57(5), 453-476.
    """
    # for safety make a copy 
    dff = df.copy() 
    
    # Validate operation
    if ops not in {'sum', 'product'}:
        msg = (
            "Operation `ops` must be either 'sum' or 'product'"
        )
        if error== 'warn':
            warnings.warn(msg)
            return dff
        else:
            raise ValueError(msg)
    
    # Validate direction
    if direction not in {'left_to_right', 'right_to_left'}:
        msg = (
            "`direction` must be either 'left_to_right' or 'right_to_left'"
        )
        if error== 'warn':
            warnings.warn(msg)
            return dff
        else:
            raise ValueError(msg)
    
    # Validate error_handling
    if error not in {'warn', 'raise'}:
        msg = (
            "`error_handling` must be either 'warn' or 'raise'"
        )
        raise ValueError(msg)
    
    if checkpoint is None: 
        checkpoint=0 # take the first column. 
        
    # Determine checkpoint column name
    if isinstance(checkpoint, int):
        if checkpoint < 0 or checkpoint >= len(dff.columns):
            msg = (
                f"Checkpoint index {checkpoint} is out of range for "
                f"DataFrame with {len(dff.columns)} columns."
            )
            if error== 'warn':
                warnings.warn(msg)
                return dff
            else:
                raise ValueError(msg)
        checkpoint_col = dff.columns[checkpoint]
    elif isinstance(checkpoint, str):
        if checkpoint not in dff.columns:
            msg = (
                f"Checkpoint column '{checkpoint}' does not exist in the DataFrame."
            )
            if error== 'warn':
                warnings.warn(msg)
                return dff
            else:
                raise ValueError(msg)
        checkpoint_col = checkpoint
    else:
        msg = (
            "`checkpoint` must be a column name (str) or index (int)"
        )
        if error== 'warn':
            warnings.warn(msg)
            return dff
        else:
            raise TypeError(msg)
    
    # Ensure checkpoint column is numeric
    if not pd.api.types.is_numeric_dtype(dff[checkpoint_col]):
        msg = (
            f"Checkpoint column '{checkpoint_col}' must be numeric."
        )
        if error== 'warn':
            warnings.warn(msg)
            return dff
        else:
            raise ValueError(msg)
    
    # Separate numeric and categorical columns
    numeric_cols = dff.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = dff.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Ensure checkpoint is in numeric columns
    if checkpoint_col not in numeric_cols:
        msg = (
            f"Checkpoint column '{checkpoint_col}' is not numeric and cannot "
            f"be used for cumulative operations."
        )
        if error== 'warn':
            warnings.warn(msg)
            return dff
        else:
            raise ValueError(msg)
    
    # Get index of checkpoint in numeric columns
    checkpoint_idx = numeric_cols.index(checkpoint_col)
    
    # Determine columns to operate on based on direction
    if direction == 'left_to_right':
        target_cols = numeric_cols[checkpoint_idx + 1 :]
        if not target_cols:
            msg = (
                "No columns to operate on after the checkpoint when "
                "`direction` is 'left_to_right'."
            )
            if error== 'warn':
                warnings.warn(msg)
                return dff
            else:
                raise ValueError(msg)
    else:  # 'right_to_left'
        target_cols = numeric_cols[:checkpoint_idx][::-1]
        if not target_cols:
            msg = (
                "No columns to operate on before the checkpoint when "
                "`direction` is 'right_to_left'."
            )
            if error== 'warn':
                warnings.warn(msg)
                return dff
            else:
                raise ValueError(msg)
    
    # Separate original numeric data for reverse operations
    original_numeric = dff[numeric_cols].copy()
    
    # Perform cumulative or 'decumulative' operations
    if reverse:
        # De-cumulative: reverse the cumulative operation
        if direction == 'left_to_right':
            prev_col = checkpoint_col
            for col in target_cols:
                dff[col] = dff[col] - original_numeric[prev_col] # dff[prev_col]
                prev_col = col
        else:  # 'right_to_left'
            prev_col = checkpoint_col
            for col in target_cols:
                dff[col] = dff[col] - dff[prev_col]
                prev_col = col
    else:
        # Perform cumulative operations
        if ops == 'sum':
            func = lambda prev, current: prev + current
        elif ops == 'product':
            func = lambda prev, current: prev * current
        
        if direction == 'left_to_right':
            prev_col = checkpoint_col
            for col in target_cols:
                dff[col] = func(dff[prev_col], dff[col])
                prev_col = col
        else:  # 'right_to_left'
            prev_col = checkpoint_col
            for col in target_cols:
                dff[col] = func(dff[prev_col], dff[col])
                prev_col = col
    
    # Re-add categorical columns in original order
    if categorical_cols:
        # Maintain original column order
        cols_order = [
            col for col in dff.columns if col in categorical_cols
        ] + [
            col for col in dff.columns if col in numeric_cols
        ]
        dff = dff[cols_order]
    
    return dff
