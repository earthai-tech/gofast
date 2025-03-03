# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Provides utility functions for quantile extraction and validation.
"""
from __future__ import annotations 

import re 
import warnings 
from typing import List, Optional, Union, Tuple 

import numpy as np # noqa
import pandas as pd 

from ..core.checks import check_empty, check_spatial_columns  
from ..core.diagnose_q import validate_quantiles 
from ..core.handlers import columns_manager 
from ..core.io import to_frame_if 
from ..core.utils import error_policy 
from ..core.io import SaveFile


__all__ =["reshape_quantile_data", "melt_q_data", "pivot_q_data"]

@SaveFile 
@check_empty(params=["df"], allow_none=False, none_as_empty= True) 
def reshape_quantile_data(
    df: pd.DataFrame,
    value_prefix: str,
    spatial_cols: Optional[List[str]] = None,
    dt_col: str = 'year',
    error: str = 'warn',
    savefile: Optional[str] = None, 
    verbose: int = 0, 
) -> pd.DataFrame:
    r"""
    Reshape a wide-format DataFrame with quantile columns into a 
    DataFrame where the quantiles are separated into distinct 
    columns for each quantile value.

    This method transforms columns that follow the naming pattern 
    ``{value_prefix}_{dt_value}_q{quantile}`` into a structured format,
    preserving spatial coordinates and adding the temporal dimension
    based on extracted datetime values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing quantile columns. The columns should 
        follow the pattern ``{value_prefix}_{dt_val}_q{quantile}``, where:
        - `value_prefix` is the base name for the quantile measurement
          (e.g., ``'predicted_subsidence'``)
        - `dt_val` is the datetime value (e.g., year or month)
        - `quantile` is the quantile value (e.g., 0.1, 0.5, 0.9)
    value_prefix : str
        Base name for quantile measurement columns (e.g., 
        ``'predicted_subsidence'``). This is used to identify the 
        quantile columns in the DataFrame.
    spatial_cols : list of str, optional
        List of spatial column names (e.g., ``['longitude', 'latitude']``).
        These columns will be preserved through the reshaping operations.
        If `None`, the default columns (e.g., ``['longitude', 'latitude']``)
        will be used.
    dt_col : str, default='year'
        Name of the column that will contain the extracted temporal 
        information (e.g., 'year'). This will be used as a column in the
        output DataFrame for temporal dimension tracking.
    error : {'raise', 'warn', 'ignore'}, default='warn'
        Specifies how to handle errors when certain columns or data 
        patterns are not found. Options include:
        - ``'raise'``: Raises a ValueError with a message if columns are missing.
        - ``'warn'``: Issues a warning with a message if columns are missing.
        - ``'ignore'``: Silently returns an empty DataFrame when issues are found.
    savefile : str, optional
        Path to save the reshaped DataFrame. If provided, the DataFrame
        will be saved to this location.
    verbose : int, default=0
        Level of verbosity for progress messages. Higher values 
        correspond to more detailed output during processing:
        - 0: Silent
        - 1: Basic progress
        - 2: Column parsing details
        - 3: Metadata extraction
        - 4: Reshaping steps
        - 5: Full debug

    Returns
    -------
    pd.DataFrame
        A reshaped DataFrame with quantiles as separate columns for each 
        quantile value. The DataFrame will have the following columns:
        - Spatial columns (if any)
        - Temporal column (specified by ``dt_col``)
        - ``{value_prefix}_q{quantile}`` value columns for each quantile

    Examples
    --------
    >>> from gofast.utils.q_utils import reshape_quantile_data
    >>> import pandas as pd
    >>> wide_df = pd.DataFrame({
    ...     'lon': [-118.25, -118.30],
    ...     'lat': [34.05, 34.10],
    ...     'subs_2022_q0.1': [1.2, 1.3],
    ...     'subs_2022_q0.5': [1.5, 1.6],
    ...     'subs_2023_q0.1': [1.7, 1.8]
    ... })
    >>> reshaped_df = reshape_quantile_data(wide_df, 'subs')
    >>> reshaped_df.columns
    Index(['lon', 'lat', 'year', 'subs_q0.1', 'subs_q0.5'], dtype='object')

    Notes
    -----
    - The column names must follow the pattern 
      ``{value_prefix}_{dt_value}_q{quantile}`` for proper extraction.
    - The temporal dimension is determined by the ``dt_col`` argument.
    - Spatial columns are automatically detected or can be passed explicitly.
    - The quantiles are pivoted and separated into distinct columns 
      based on the unique quantile values found in the DataFrame.
      
    .. math::

        \mathbf{W}_{m \times n} \rightarrow \mathbf{L}_{p \times k}

    Where:
    - :math:`m` = Original row count
    - :math:`n` = Original columns (quantile + spatial + temporal)
    - :math:`p` = :math:`m \times t` (t = unique temporal values)
    - :math:`k` = Spatial cols + 1 temporal + q quantile cols

    
    See Also
    --------
    pandas.melt : For reshaping DataFrames from wide to long format.
    gofast.utils.validator.melt_q_data : Alternative method for reshaping quantile data.
    gofast.utils.validator.handle_error : Error handling utility for reshaping functions.
    
    References
    ----------
    .. [1] McKinney, W. (2010). "Data Structures for Statistical Computing
           in Python". Proceedings of the 9th Python in Science Conference.
    .. [2] Wickham, H. (2014). "Tidy Data". Journal of Statistical Software,
           59(10), 1-23.
    """
    # Initial validation
    df = to_frame_if(df).copy() 
    if spatial_cols:
        missing_spatial = set(spatial_cols) - set(df.columns)
        if missing_spatial:
            msg = f"Missing spatial columns: {missing_spatial}"
            if error == 'raise':
                raise ValueError(msg)
            elif error == 'warn':
                warnings.warn(msg)
            spatial_cols = list(set(spatial_cols) & set(df.columns))

    # Find quantile columns
    quant_cols = [col for col in df.columns 
                 if col.startswith(value_prefix)]
    
    if not quant_cols:
        msg = f"No columns found with prefix '{value_prefix}'"
        if error == 'raise':
            raise ValueError(msg)
        elif error == 'warn':
            warnings.warn(msg)
        return pd.DataFrame()

    # Extract metadata from column names
    pattern = re.compile(
        rf"{re.escape(value_prefix)}_(\d{{4}})_q([0-9.]+)$"
    )
    
    meta = []
    valid_cols = []
    for col in quant_cols:
        match = pattern.match(col)
        if match:
            year, quantile = match.groups()
            meta.append((col, int(year), float(quantile)))
            valid_cols.append(col)

    if verbose >= 1:
        print(f"Found {len(valid_cols)} valid quantile columns")

    if not valid_cols:
        return pd.DataFrame()

    # Melt dataframe
    id_vars = spatial_cols if spatial_cols else []
    melt_df = df.melt(
        id_vars=id_vars,
        value_vars=valid_cols,
        var_name='column',
        value_name='value'
    )

    # Add metadata columns
    meta_df = pd.DataFrame(
        meta, columns=['column', dt_col, 'quantile']
    )
    melt_df = melt_df.merge(meta_df, on='column')

    # Pivot to wide format
    pivot_df = melt_df.pivot_table(
        index=id_vars + [dt_col],
        columns='quantile',
        values='value',
        aggfunc='first'
    ).reset_index()

    # Clean column names
    pivot_df.columns = [
        f"{value_prefix}_q{col}" if isinstance(col, float) else col 
        for col in pivot_df.columns
    ]

    return pivot_df.sort_values(
        by=dt_col, ascending=True
    ).reset_index(drop=True)


@SaveFile 
@check_empty(params=["df"], allow_none=False, none_as_empty= True) 
def melt_q_data(
    df: pd.DataFrame,
    value_prefix: str,
    dt_name: str = 'dt_col',
    q: Optional[List[Union[float, str]]] = None,
    error: str = 'raise',
    savefile: Optional[str]=None, 
    verbose: int = 0, 
) -> pd.DataFrame:
    r"""
    Reshape wide-format DataFrame with quantile columns to long format 
    with explicit temporal and quantile dimensions.

    This method transforms columns that follow the naming pattern 
    ``{value_prefix}_{dt_value}_q{quantile}`` into a structured long format
    with separated datetime and quantile columns. Handles spatial 
    coordinates preservation through reshaping operations.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing quantile columns. The columns should 
        follow the pattern ``{value_prefix}_{dt_val}_q{quantile}``, where:
        - `value_prefix` is the base name for the quantile measurement
          (e.g., ``'predicted_subsidence'``)
        - `dt_val` is the datetime value (e.g., year or month)
        - `quantile` is the quantile value (e.g., 0.1, 0.5, 0.9)
    value_prefix : str
        Base name for quantile measurement columns (e.g., 
        ``'predicted_subsidence'``). This is used to identify the 
        quantile columns in the DataFrame.
    dt_name : str, default='dt_col'
        Name of the column that will contain the extracted temporal 
        information (e.g., 'year'). This will be used as a column in the
        output DataFrame for temporal dimension tracking.
    q : list of float/str, optional
        Specific quantiles to include. Accepts:
        - Float values (0.1, 0.5, 0.9)
        - Percentage strings ("10%", "90%")
        - None (include all detected quantiles)
    error : {'raise', 'warn', 'ignore'}, default='raise'
        Specifies how to handle errors when certain columns or data 
        patterns are not found. Options include:
        - ``'raise'``: Raises a ValueError with a message if columns are missing.
        - ``'warn'``: Issues a warning with a message if columns are missing.
        - ``'ignore'``: Silently returns an empty DataFrame when issues are found.
    savefile : str, optional
        Path to save the reshaped DataFrame. If provided, the DataFrame
        will be saved to this location.
    verbose : int, default=0
        Level of verbosity for progress messages. Higher values 
        correspond to more detailed output during processing:
        - 0: Silent
        - 1: Basic progress
        - 2: Column parsing details
        - 3: Metadata extraction
        - 4: Reshaping steps
        - 5: Full debug

    Returns
    -------
    pd.DataFrame
        A long-format DataFrame with quantiles as separate columns for 
        each quantile value. The DataFrame will have the following columns:
        - Spatial columns (if any)
        - Temporal column (specified by ``dt_name``)
        - ``{value_prefix}_q{quantile}`` value columns for each quantile

    Examples
    --------
    >>> from gofast.utils.validator import melt_q_data
    >>> import pandas as pd
    >>> wide_df = pd.DataFrame({
    ...     'lon': [-118.25, -118.30],
    ...     'lat': [34.05, 34.10],
    ...     'subs_2022_q0.1': [1.2, 1.3],
    ...     'subs_2022_q0.5': [1.5, 1.6],
    ...     'subs_2023_q0.1': [1.7, 1.8]
    ... })
    >>> long_df = melt_q_data(wide_df, 'subs', dt_name='year')
    >>> long_df.columns
    Index(['lon', 'lat', 'year', 'subs_q0.1', 'subs_q0.5'], dtype='object')

    Notes
    -----
    - The column names must follow the pattern 
      ``{value_prefix}_{dt_value}_q{quantile}`` for proper extraction.
    - The temporal dimension is determined by the ``dt_name`` argument.
    - Spatial columns are automatically detected or can be passed explicitly.
    - The quantiles are pivoted and separated into distinct columns 
      based on the unique quantile values found in the DataFrame.
      
    .. math::

        \mathbf{W}_{m \times n} \rightarrow \mathbf{L}_{p \times k}

    Where:
    - :math:`m` = Original row count
    - :math:`n` = Original columns (quantile + spatial + temporal)
    - :math:`p` = :math:`m \times t` (t = unique temporal values)
    - :math:`k` = Spatial cols + 1 temporal + q quantile cols

    See Also
    --------
    pandas.melt : For reshaping DataFrames from wide to long format.
    gofast.utils.q_utils.reshape_quantile_data : 
        Alternative method for reshaping quantile data.


    References
    ----------
    .. [1] McKinney, W. (2010). "Data Structures for Statistical Computing
           in Python". Proceedings of the 9th Python in Science Conference.
    .. [2] Wickham, H. (2014). "Tidy Data". Journal of Statistical Software,
           59(10), 1-23.
    """

    # Validate error handling parameter
    error = error_policy ( error, base="warn", 
        msg="error must be one of 'raise', 'warn', or 'ignore'" 
    )
    # Create working copy and prepare columns
    df = to_frame_if(df).copy ()
    cols = df.columns.tolist()
    pattern = re.compile(rf"^{re.escape(value_prefix)}_(\d+)_q([0-9.]+)$")

    # Find relevant columns and parse metadata
    meta = []
    quant_cols = []
    for col in cols:
        match = pattern.match(col)
        if match:
            dt_val, q_val = match.groups()
            meta.append((col, dt_val, float(q_val)))
            quant_cols.append(col)
    
    # Handle no columns found
    if not quant_cols:
        msg = f"No columns found with prefix '{value_prefix}'"
        handle_error(msg, error)
        return pd.DataFrame()
    
    # Filter by requested quantiles
    if q is not None:
        q_valid = [float(qq) for qq in validate_quantiles(
            q, mode='soft',dtype='float64')]
        meta = [m for m in meta if m[2] in q_valid]
        quant_cols = [m[0] for m in meta]
        
        if not quant_cols:
            msg = f"No columns match requested quantiles {q}"
            handle_error(msg, error)
            return pd.DataFrame()

    # Extract spatial columns
    spatial_cols = []
    spatial_cols = columns_manager(spatial_cols, empty_as_none=False)
    if spatial_cols: 
        check_spatial_columns(df, spatial_cols )
    # for col in ['longitude', 'latitude']:  # Common default spatial columns
    #     if col in df.columns and col not in quant_cols:
    #         spatial_cols.append(col)
    
    # Melt dataframe
    id_vars = spatial_cols
    melt_df = df.melt(
        id_vars=id_vars,
        value_vars=quant_cols,
        var_name='column',
        value_name=value_prefix
    )

    # Add metadata columns
    meta_df = pd.DataFrame(meta, columns=['column', dt_name, 'quantile'])
    merged_df = melt_df.merge(meta_df, on='column')

    # Pivot quantiles to columns
    pivot_df = merged_df.pivot_table(
        index=id_vars + [dt_name],
        columns='quantile',
        values=value_prefix,
        aggfunc='first'
    ).reset_index()

    # Clean column names
    pivot_df.columns = [
        f"{value_prefix}_q{col:.2f}".rstrip('0').rstrip('.') 
        if isinstance(col, float) else col 
        for col in pivot_df.columns
    ]

    # Sort and finalize
    sort_cols = spatial_cols + [dt_name]
    return pivot_df.sort_values(sort_cols).reset_index(drop=True)

def handle_error(msg: str, error: str) -> None:
    """Centralized error handling."""
    if error == 'raise':
        raise ValueError(msg)
    elif error == 'warn':
        warnings.warn(msg)
    # No action for 'ignore'

@SaveFile 
@check_empty(params=["df"], allow_none=False, none_as_empty= True) 
def pivot_q_data(
    df: pd.DataFrame,
    value_prefix: str,
    dt_col: str = 'dt_col',
    q: Optional[List[Union[float, str]]] = None,
    spatial_cols: Optional[Tuple[str, str]]=None, 
    error: str = 'raise',
    verbose: int = 0
) -> pd.DataFrame:
    r"""
    Convert long-format DataFrame with quantile columns back to wide format
    with temporal quantile measurements.

    Reconstructs columns following the pattern 
    ``{value_prefix}_{dt_value}_q{quantile}`` from separated temporal and
    quantile dimensions. Inverse operation of ``to_long_data_q`` [1]_.

    .. math::

        \mathbf{L}_{p \times k} \rightarrow \mathbf{W}_{m \times n}
        
    Where:
    - :math:`p` = Long format row count
    - :math:`k` = Spatial cols + temporal + quantile columns
    - :math:`m` = :math:`p / t` (t = unique temporal values)
    - :math:`n` = Spatial cols + :math:`t \times q` quantile columns

    Parameters
    ----------
    df : pd.DataFrame
        Long-format DataFrame containing:
        - Spatial columns (e.g., ``'lon'``, ``'lat'``)
        - Temporal column (``dt_col``)
        - Quantile columns (``{value_prefix}_q{quantile}``)
    value_prefix : str
        Base measurement name for column reconstruction
        (e.g., ``'predicted_subsidence'``)
    dt_col : str, default='dt_col'
        Name of temporal dimension column containing dt_values
    q : list of float/str, optional
        Specific quantiles to include in output. If None,
        uses all detected quantiles in columns
    error : {'raise', 'warn', 'ignore'}, default='raise'
        Handling for missing components:
        - ``'raise'``: ValueError on missing data
        - ``'warn'``: Warning with partial DataFrame
        - ``'ignore'``: Return partial DataFrame silently
    verbose : {0, 1, 2, 3, 4, 5}, default=0
        Detail level for processing messages:
        - 0: Silent
        - 1: Basic progress
        - 2: Column detection details
        - 3: Quantile validation
        - 4: Pivoting steps
        - 5: Full shape transitions

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with columns:
        - Spatial columns
        - ``{value_prefix}_{dt_value}_q{quantile}`` columns

    Examples
    --------
    >>> from gofast.utils.q_utils import pivot_q_data
    >>> long_df = pd.DataFrame({
    ...     'lon': [-118.25, -118.25, -118.3],
    ...     'lat': [34.05, 34.05, 34.1],
    ...     'year': [2022, 2023, 2022],
    ...     'subs_q0.1': [1.2, 1.7, 1.3],
    ...     'subs_q0.5': [1.5, 1.9, 1.6]
    ... })
    >>> wide_df = pivot_q_data(long_df, 'subs', dt_col='year')
    >>> wide_df.columns
    Index(['lon', 'lat', 'subs_2022_q0.1', 'subs_2022_q0.5',
           'subs_2023_q0.1', 'subs_2023_q0.5'], dtype='object')

    Notes
    -----
    1. Column requirements:
       - Must contain exactly one temporal column (``dt_col``)
       - Quantile columns must follow ``{prefix}_q{quantile}`` pattern
       - Spatial columns must be unique per location

    2. Pivoting logic:
       - Maintains original spatial coordinates through operations
       - Handles missing quantiles per temporal value based on ``error``
       - Preserves original data types for measurement values

    See Also
    --------
    pandas.pivot_table : Base pandas function for reshaping
    to_long_data_q : Inverse transformation function
    gofast.analysis.validate_spatial_coordinates : Spatial validation

    References
    ----------
    .. [1] VanderPlas, J. (2016). "Python Data Science Handbook".
           O'Reilly Media, Inc.
    .. [2] McKinney, W. (2013). "Python for Data Analysis".
           O'Reilly Media, Inc.
    """
    def handle_error(
        msg: str, 
        error: str, 
        default: pd.DataFrame
    ) -> pd.DataFrame:
        """Centralized error handling."""
        if error == 'raise':
            raise ValueError(msg)
        elif error == 'warn':
            warnings.warn(msg)
        return default

    # Validate input parameters
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    if error not in ['raise', 'warn', 'ignore']:
        raise ValueError("error must be 'raise', 'warn', or 'ignore'")

    # Create working copy and validate structure
    df = df.copy()
    required_cols = {dt_col}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        msg = f"Missing required columns: {missing}"
        return handle_error(msg, error, pd.DataFrame())

    # Detect quantile columns
    quant_pattern = re.compile(rf"^{re.escape(value_prefix)}_q([0-9.]+)$")
    quant_columns = [col for col in df.columns if quant_pattern.match(col)]
    
    if not quant_columns:
        msg = f"No quantile columns found with prefix '{value_prefix}'"
        return handle_error(msg, error, pd.DataFrame())

    # Extract and validate quantile values
    quantiles = sorted(
        [float(quant_pattern.match(col).group(1)) for col in quant_columns],
        key=lambda x: float(x)
    )
    
    if verbose >= 1:
        print(f"Found quantiles: {quantiles}")

    # Filter requested quantiles
    if q is not None:
        valid_q = validate_quantiles(q, mode='soft', dtype='float64')
        quant_columns = [
            col for col in quant_columns
            if float(quant_pattern.match(col).group(1)) in valid_q
        ]
        if not quant_columns:
            msg = f"No columns match filtered quantiles {q}"
            return handle_error(msg, error, pd.DataFrame())

    # Identify spatial columns (non-temporal, non-quantile)
    spatial_cols = columns_manager(spatial_cols, empty_as_none=False)
    if spatial_cols: 
        check_spatial_columns(df, spatial_cols )
        
    # spatial_cols = [
    #     col for col in df.columns
    #     if col not in quant_columns + [dt_col]
    # ]
    # Melt quantile columns to long format
    id_vars = spatial_cols + [dt_col]
    melt_df = df.melt(
        id_vars=id_vars,
        value_vars=quant_columns,
        var_name='quantile',
        value_name='value'
    )

    # Extract numeric quantile values
    melt_df['quantile'] = melt_df['quantile'].str.extract(
        r'q([0-9.]+)$'
    ).astype(float)

    # Pivot to wide format
    try:
        wide_df = melt_df.pivot_table(
            index=spatial_cols,
            columns=[dt_col, 'quantile'],
            values='value',
            aggfunc='first'  # Handle potential duplicates
        )
    except ValueError as e:
        msg = f"Pivoting failed: {str(e)}"
        return handle_error(msg, error, pd.DataFrame())

    # Flatten multi-index columns
    wide_df.columns = [
        f"{value_prefix}_{dt}_q{quantile:.2f}".rstrip('0').rstrip('.')
        for (dt, quantile) in wide_df.columns
    ]

    return wide_df.reset_index()


