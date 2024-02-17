# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Created on Wed Dec 20 10:01:24 2023

These functions provide a comprehensive toolkit for performing basic 
statistical analyses. They can be easily integrated into 
data analysis workflows and are versatile enough to handle a wide range 
of data types and structures.
"""
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt 

from sklearn.cluster import KMeans,SpectralClustering
from sklearn.linear_model import LinearRegression
from sklearn.manifold import MDS

from .._typing import DataFrame, ArrayLike, List, Dict
from .._typing import  Union, Tuple , Optional
from ..decorators import DynamicMethod 
from ..tools.validator import build_data_if , assert_xy_in 
from ..tools.coreutils import to_numeric_dtypes, ellipsis2false 
from ..tools.funcutils import make_data_dynamic , preserve_input_type
from ..tools._dependency import import_optional_dependency 

@preserve_input_type
@make_data_dynamic(expected_type="numeric", capture_columns=True)
def gomean(
        data: Union[ArrayLike, DataFrame], /, 
        columns: List[str] = None,
        **kws):
    """
    Calculates the mean of numeric data provided as an array-like structure 
    or within a pandas DataFrame.

    The function dynamically preprocesses the input data based on the 
    `expected_type` and `capture_columns` parameters specified by the 
    `make_data_dynamic` decorator. For DataFrames, if `columns` are specified,
    only those columns are considered for calculating the mean.

    Parameters
    ----------
    data : ArrayLike or DataFrame
        The input data from which to calculate the mean. Can be a list, 
        numpy array, or pandas DataFrame containing valid numeric values.
    columns : list of str, optional
        List of column names to consider in the calculation if `data` is a
        DataFrame. If not provided, all numeric columns are considered.
    **kws : dict
        Additional keyword arguments passed to the underlying numpy mean 
        function, allowing customization of the mean calculation, such as 
        specifying the axis.

    Returns
    -------
    mean_value : float or np.ndarray
        The calculated mean of the provided data. Returns a single float if 
        the data is one-dimensional or 
        a numpy array if an axis is specified in `kws`.

    Examples
    --------
    >>> from gofast.stats.utils import gomean
    >>> data_array = [1, 2, 3, 4, 5]
    >>> mean(data_array)
    3.0

    >>> data_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> mean(data_df, columns=['A'])
    2.0

    >>> mean(data_df, axis=0)
    array([2., 5.])

    Note
    ----
    The function is wrapped by the `make_data_dynamic` decorator, which 
    preprocesses the input data based on the specified `expected_type` and 
    `capture_columns`. This preprocessing step is crucial for handling pandas 
    DataFrames with mixed data types and for selecting specific columns for
    the calculation.
    """
    if isinstance(data, pd.DataFrame):
        return data.mean(**kws)
    else:
        return np.mean(data, **kws)

@make_data_dynamic(expected_type="numeric", capture_columns=True)
def gomedian(
    data: Union[ArrayLike, DataFrame], 
    columns: List[str] = None, **kws
    ):
    """
    Calculates the median of numeric data provided either as an array-like 
    structure or within a pandas DataFrame.

    Leveraging the `make_data_dynamic` decorator, the function preprocesses 
    the input data to ensure it conforms to the expected numeric type and 
    optionally focuses on specified columns when dealing with DataFrames.

    Parameters
    ----------
    data : ArrayLike or DataFrame
        The input data from which to calculate the median. Can be a list,
        numpy array, or pandas DataFrame containing valid numeric values.
    columns : list of str, optional
        List of column names to consider for the calculation if `data` is a
        DataFrame. If not provided, all numeric columns are considered.
    **kws : dict
        Additional keyword arguments passed to the underlying numpy median 
        function, allowing customization of the median calculation, such as 
        specifying the axis or whether to ignore NaN values.

    Returns
    -------
    median_value : float or np.ndarray
        The calculated median of the provided data. Returns a single float 
        if the data is one-dimensional or  an np.ndarray if an axis is specified in `kws`.

    Examples
    --------
    >>> from gofast.stats.utils import gomedian
    >>> data_array = [3, 1, 4, 1, 5]
    >>> gomedian(data_array)
    3.0

    >>> data_df = pd.DataFrame({'A': [2, 4, 7], 'B': [1, 6, 5]})
    >>> gomedian(data_df, columns=['A'])
    4.0

    >>> gomedian(data_df, axis=0)
    array([4., 5.])

    Note
    ----
    This function is enhanced by the `make_data_dynamic` decorator, which 
    preprocesses the input data to align with the expected numeric type and 
    selectively processes columns if specified. This preprocessing
    is particularly useful for ensuring compatibility with statistical 
    calculations and optimizing performance for DataFrame inputs with mixed
    data types.
    """
    if isinstance(data, pd.DataFrame):
        return np.median(data.to_numpy(), **kws)
    else:
        return np.median(data, **kws)

@make_data_dynamic(expected_type="numeric", capture_columns=True)
def gomode(
        data: Union[ArrayLike, DataFrame], /, 
        columns: List[str] = None, 
        as_frame:bool=..., 
        **kws):
    """
    Calculates the mode(s) of numeric data provided either as an array-like 
    structure or within a pandas DataFrame.

    Utilizes the `make_data_dynamic` decorator to preprocess the input data
    to ensure it conforms to the expected numeric type and optionally focuses
    on specified columns when dealing with DataFrames.

    Parameters
    ----------
    data : ArrayLike or DataFrame
        The input data from which to calculate the mode. Can be a list, 
        numpy array, or pandas DataFrame containing valid numeric values.
    columns : list of str, optional
        List of column names to consider for the calculation if `data` is a 
        DataFrame. If not provided, all 
        numeric columns are considered.
    as_frame : bool, default False
        Indicates whether to convert the input data to a DataFrame before 
        proceeding. This parameter is
        particularly relevant when input data is not already a DataFrame.
    **kws : dict
        Additional keyword arguments for further data processing or passed to 
        underlying statistical functions.

    Returns
    -------
    mode_value : np.ndarray or pd.Series
        The mode(s) of the dataset. Returns a numpy array if the input is 
        ArrayLike or if `as_frame` is False.
        Returns a pandas Series if the input is a DataFrame and 
        `as_frame` is True.

    Examples
    --------
    >>> from gofast.stats.utils import gomode
    >>> data_array = [1, 2, 2, 3, 4]
    >>> gomode(data_array)
    2

    >>> data_df = pd.DataFrame({'A': [1, 2, 2, 3], 'B': [4, 4, 5, 5]})
    >>> gomode(data_df, as_frame=True)
    A    2
    B    4
    dtype: int64

    Note
    ----
    This function is enhanced by the `make_data_dynamic` decorator, which 
    preprocesses the input data to align with the expected numeric type and 
    selectively processes columns if specified. This preprocessing
    ensures that statistical calculations are performed accurately and 
    efficiently on DataFrame inputs with mixed data types.
    """
    as_frame, = ellipsis2false(as_frame)
    if isinstance(data, pd.DataFrame) and columns:
        data = data[columns]
    
    mode_result = stats.mode(data, **kws)
    return mode_result.mode[0] if not as_frame else pd.Series(
        mode_result.mode[0], index=columns or data.columns)

@make_data_dynamic(
    expected_type="numeric", 
    capture_columns=True, 
    reset_index=True
    )
def govar(
    data: Union[ArrayLike, DataFrame], /, 
    columns: List[str] = None, 
    as_frame:bool=..., 
    **kws):
    """
    Calculates the variance of numeric data provided either as an array-like 
    structure or within a pandas DataFrame. This function is designed to be 
    flexible, allowing for the calculation of variance across entire datasets 
    or within specified columns of a DataFrame.

    Utilizing preprocessing parameters such as `expected_type` and 
    `capture_columns`, the input data is first sanitized to ensure it contains 
    only numeric values. This preprocessing step is essential for accurate 
    statistical analysis.

    Parameters
    ----------
    data : ArrayLike or DataFrame
        The input data from which to calculate the variance. Can be a list, 
        numpy array, or pandas DataFrame containing valid numeric values.
    columns : list of str, optional
        List of column names to consider for the calculation if `data` is a 
        DataFrame. If not provided, all numeric columns are considered.
    as_frame : bool, default False
        Indicates whether to convert the input data to a DataFrame before 
        proceeding with the calculation. This parameter is particularly relevant 
        when input data is not already a DataFrame.
    **kws : dict
        Additional keyword arguments passed to the underlying numpy variance 
        function, allowing customization of the variance calculation, such as 
        specifying the axis or whether to use Bessel's correction.

    Returns
    -------
    variance_value : float or np.ndarray or pd.Series
        The calculated variance of the provided data. Returns a single float 
        if the data is one-dimensional, a numpy array if an axis is specified 
        in `kws`, or a pandas Series if the input is a DataFrame and `as_frame`
        is True.

    Examples
    --------
    >>> from gofast.stats.utils import govar
    >>> data_array = [1, 2, 3, 4, 5]
    >>> govar(data_array)
    2.0

    >>> data_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> govar(data_df, columns=['A'], ddof=1)
    1.0

    >>> govar(data_df, as_frame=True)
    A    1.0
    B    1.0
    dtype: float64

    Note
    ----
    The preprocessing steps, controlled by the `make_data_dynamic` decorator,
    ensure that the input data is suitable for variance calculation. This
    includes converting the data to numeric types and filtering based on 
    specified columns, enhancing the function's flexibility and applicability 
    across various data forms.
    """
    as_frame = False if as_frame is ... else as_frame

    # Preprocess DataFrame to focus on specified columns
    if isinstance(data, pd.DataFrame) and columns:
        data = data[columns]

    # Calculate variance
    variance_result = np.var(data, **kws)
    # Return result in the appropriate format
    if as_frame:
        return pd.Series(variance_result, index=columns or data.columns)
    else:
        return variance_result
    
@make_data_dynamic(
    expected_type="numeric", 
    capture_columns=True, 
    reset_index=True
    )    
def gostd(
    data: Union[ArrayLike, DataFrame], /, 
    columns: List[str] = None, 
    as_frame: bool = False, 
    **kws
    ):
    """
    Computes the standard deviation of numeric data provided either as an 
    array-like structure or within a pandas DataFrame. This function leverages 
    preprocessing to ensure the data is numeric and optionally focuses on 
    specified columns when dealing with DataFrames.

    Parameters
    ----------
    data : ArrayLike or DataFrame
        The input data from which to calculate the standard deviation. Can be 
        a list, numpy array, or pandas DataFrame containing valid numeric values.
    columns : list of str, optional
        List of column names to consider for the calculation if `data` is a 
        DataFrame. If not provided, all numeric columns are considered.
    as_frame : bool, default False
        Indicates whether to convert the input data to a DataFrame before 
        proceeding with the calculation. This parameter is relevant when 
        input data is not already a DataFrame.
    **kws : dict
        Additional keyword arguments passed to the underlying numpy standard 
        deviation function, allowing customization of the calculation, such 
        as specifying the axis or whether to use Bessel's correction.

    Returns
    -------
    std_dev_value : float or np.ndarray or pd.Series
        The calculated standard deviation of the provided data. Returns a single 
        float if the data is one-dimensional, an np.ndarray if an axis is specified 
        in `kws`, or a pandas Series if the input is a DataFrame and `as_frame` 
        is True.

    Examples
    --------
    >>> from gofast.stats.utils import gostd
    >>> data_array = [1, 2, 3, 4, 5]
    >>> gostd(data_array)
    1.4142135623730951

    >>> data_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> gostd(data_df, columns=['A'], ddof=1)
    1.0

    >>> gostd(data_df, as_frame=True)
    A    1.0
    B    1.0
    dtype: float64

    Note
    ----
    The preprocessing steps, implied by the `make_data_dynamic` decorator (not shown here but 
    assumed to be applied outside this code snippet), ensure that the input data is suitable 
    for standard deviation calculation. These steps convert the data to numeric types and filter 
    based on specified columns, thus enhancing the function's utility and applicability across 
    diverse data forms.
    """
    as_frame = False if as_frame is ... else as_frame
    
    # Preprocess DataFrame to focus on specified columns
    if isinstance(data, pd.DataFrame) and columns:
        data = data[columns]
    # Calculate standard deviation
    std_dev_result = np.std(data, **kws)
    # Return result in the appropriate format
    if as_frame:
        return pd.Series(std_dev_result, index=columns or data.columns)
    else:
        return std_dev_result

@make_data_dynamic(
    expected_type="numeric", 
    capture_columns=True, 
    reset_index=True
    )   
def get_range(
    data: Union[ArrayLike, DataFrame], /, 
    columns: List[str] = None, 
    as_frame: bool = False,
    **kws
    ):
    """
    Calculates the range of numeric data provided either as an array-like 
    structure or within a pandas DataFrame. This function computes the difference 
    between the maximum and minimum values in the dataset, optionally focusing on 
    specified columns when dealing with DataFrames.

    Parameters
    ----------
    data : ArrayLike or DataFrame
        The input data from which to calculate the range. Can be a list, numpy 
        array, or pandas DataFrame containing valid numeric values.
    columns : list of str, optional
        List of column names to consider for the calculation if `data` is a 
        DataFrame. If not provided, all numeric columns are considered.
    as_frame : bool, default False
        Indicates whether to convert the input data to a DataFrame before 
        proceeding with the calculation. Relevant when input data is not 
        already a DataFrame.
    **kws : dict
        Additional keyword arguments for data preprocessing, not directly 
        used in range calculation but may be used for data sanitization.

    Returns
    -------
    range_value : float or pd.Series
        The calculated range of the provided data. Returns a single float if 
        the data is one-dimensional or a pandas Series if the input is a DataFrame 
        and `as_frame` is True, showing the range for each specified column.

    Examples
    --------
    >>> from gofast.stats.utils import get_range
    >>> data_array = [1, 2, 3, 4, 5]
    >>> get_range(data_array)
    4

    >>> data_df = pd.DataFrame({'A': [2, 5, 8], 'B': [1, 4, 7]})
    >>> get_range(data_df, columns=['A'])
    6

    >>> get_range(data_df, as_frame=True)
    A    6
    B    6
    dtype: int64

    Note
    ----
    The function aims to provide a straightforward way to compute the range 
    of a dataset, enhancing data analysis tasks. For DataFrame inputs, the 
    `as_frame` parameter allows users to maintain DataFrame structure in the 
    output, facilitating further analysis or integration with pandas-based 
    workflows.
    """
    as_frame = False if as_frame is ... else as_frame
    if isinstance(data, pd.DataFrame):
        # When columns are specified, reduce the DataFrame to these columns
        if columns:
            data = data[columns]
    # Calculate the range using numpy functions
    range_calculation = lambda x: np.max(x, **kws) - np.min(x, **kws)
    
    if as_frame and isinstance(data, pd.DataFrame):
        # Calculate range for each column and return as a pandas Series
        return data.apply(range_calculation)
    else:
        # For non-DataFrame or entire DataFrame as input
        return range_calculation(data)
    
@make_data_dynamic(
    expected_type="numeric", 
    capture_columns=True, 
    )  
def quartiles(
    data: Union[ArrayLike, DataFrame], /, 
    columns: List[str] = None, 
    as_frame: bool = ...,
    **kws
    ):
    """
    Calculates the quartiles (25th, 50th, and 75th percentiles) of numeric data 
    provided either as an array-like structure or within a pandas DataFrame. 
    This function is versatile, allowing calculations across entire datasets or 
    within specified columns of a DataFrame.

    Parameters
    ----------
    data : ArrayLike or DataFrame
        The input data from which to calculate quartiles. Can be a list, numpy 
        array, or pandas DataFrame containing valid numeric values.
    columns : list of str, optional
        List of column names to consider for the calculation if `data` is a 
        DataFrame. If not provided, all numeric columns are considered.
    as_frame : bool, default False
        Indicates whether to convert the input data to a DataFrame before 
        proceeding with the calculation. Relevant when input data is not 
        already a DataFrame.
    **kws : dict
        Additional keyword arguments passed to the numpy percentile function, 
        allowing customization of the quartile calculation, such as specifying 
        the interpolation method.

    Returns
    -------
    quartile_values : np.ndarray or pd.DataFrame
        The calculated quartiles of the provided data. Returns a numpy array 
        with the quartiles if the data is one-dimensional or a pandas DataFrame 
        if the input is a DataFrame and `as_frame` is True, showing the quartiles 
        for each specified column.

    Examples
    --------
    >>> from gofast.stats.utils import quartiles
    >>> data_array = [1, 2, 3, 4, 5]
    >>> quartiles(data_array)
    array([2., 3., 4.])

    >>> data_df = pd.DataFrame({'A': [2, 5, 7, 8], 'B': [1, 4, 6, 9]})
    >>> quartiles(data_df, columns=['A'])
    array([4.25, 6. , 7.75])

    >>> quartiles(data_df, as_frame=True)
        25%   50%   75%
    A  4.25  6.0  7.75
    B  2.75  5.0  7.25

    Note
    ----
    Quartiles are a fundamental statistical measure used to understand the spread 
    and center of a dataset. By offering the flexibility to compute quartiles for 
    specific columns or entire datasets, this function aids in comprehensive data 
    analysis, particularly in exploratory data analysis (EDA) and data visualization.
    """
    as_frame = False if as_frame is ... else as_frame

    # Preprocess DataFrame to focus on specified columns
    if isinstance(data, pd.DataFrame) and columns:
        data = data[columns]

    # Calculate quartiles
    quartile_calculation = lambda x: np.percentile(x, [25, 50, 75], **kws)

    if as_frame and isinstance(data, pd.DataFrame):
        # Calculate quartiles for each column and return as a DataFrame
        quartiles_df = pd.DataFrame({col: quartile_calculation(
            data[col]) for col in data.columns}, index=['25%', '50%', '75%'])
        return quartiles_df
    else:
        # For non-DataFrame or entire DataFrame as input
        return quartile_calculation(data)
    
@DynamicMethod ( 
    expected_type="numeric", 
    capture_columns=True
   )
def goquantile(
    data: Union[ArrayLike, DataFrame], /, 
    q: float, 
    columns: List[str] = None, 
    as_frame: bool = False, 
    **kws
    ):
    """
    Computes specified quantiles of numeric data provided either as an array-like 
    structure or within a pandas DataFrame. This function is designed to offer 
    flexibility in calculating quantiles across entire datasets or within specified 
    columns of a DataFrame.

    Parameters
    ----------
    data : ArrayLike or DataFrame
        The input data from which to compute quantiles. Can be a list, numpy 
        array, or pandas DataFrame containing valid numeric values.
    q : float or list of float
        Quantile or sequence of quantiles to compute, which must be between 0 
        and 1 inclusive.
    columns : list of str, optional
        List of column names to consider for the calculation if `data` is a 
        DataFrame. If not provided, all numeric columns are considered.
    as_frame : bool, default False
        Indicates whether to convert the input data to a DataFrame before 
        proceeding with the calculation. Relevant when input data is not 
        already a DataFrame.
    **kws : dict
        Additional keyword arguments passed to the numpy quantile function, 
        allowing customization of the quantile calculation, such as specifying 
        the interpolation method.

    Returns
    -------
    quantile_values : float, np.ndarray, or pd.DataFrame
        The computed quantile(s) of the provided data. Returns a single float if 
        `q` is a single quantile value and the data is one-dimensional, an 
        np.ndarray if `q` is a list and the data is one-dimensional, or a pandas 
        DataFrame if the input is a DataFrame and `as_frame` is True.

    Examples
    --------
    >>> from gofast.stats.utils import goquantile
    >>> data_array = [1, 2, 3, 4, 5]
    >>> goquantile(data_array, q=0.5)
    3.0

    >>> data_df = pd.DataFrame({'A': [2, 5, 7, 8], 'B': [1, 4, 6, 9]})
    >>> goquantile(data_df, q=[0.25, 0.75], columns=['A'])
    array([3.75, 7.25])

    >>> goquantile(data_df, q=0.5, as_frame=True)
       50%
    A  6.0
    B  5.0

    Note
    ----
    The function provides a convenient way to compute quantiles, a critical 
    statistical measure for understanding the distribution of data. The 
    flexibility to compute quantiles for specific columns or entire datasets 
    enhances its utility in exploratory data analysis and data preprocessing.
    """
    as_frame = False if as_frame is ... else as_frame
    if isinstance(data, pd.DataFrame) and columns:
        data = data[columns]

    quantile_calculation = lambda x: np.quantile(x, q, **kws)

    if as_frame and isinstance(data, pd.DataFrame):
        # Compute quantiles for each column and return as a DataFrame
        quantiles_df = pd.DataFrame({col: quantile_calculation(
            data[col]) for col in data.columns}, index=[
                f'{q*100}%' for q in np.atleast_1d(q)])
        return quantiles_df
    else:
        # For non-DataFrame or entire DataFrame as input
        return quantile_calculation(data)

@DynamicMethod ( 
    expected_type="both", 
    capture_columns=True
   )
def gocorr(
    data: Union[ArrayLike, DataFrame], /,  
    columns: List[str] = None, **kws
    ):
    """
    Calculates the correlation matrix for the given dataset. If the dataset is 
    a DataFrame and columns are specified, the correlation is calculated only 
    for those columns.

    Parameters
    ----------
    data : ArrayLike or DataFrame
        The input data from which to calculate the correlation matrix. Can be 
        a list, numpy array, or pandas DataFrame containing valid numeric values.
    columns : list of str, optional
        List of column names for which to calculate the correlation if `data` 
        is a DataFrame. If not provided, the correlation is calculated for all 
        numeric columns in the DataFrame.
    **kws : dict
        Additional keyword arguments for the pandas `corr()` method, allowing 
        customization of the correlation calculation, such as specifying the 
        method (e.g., 'pearson', 'kendall', 'spearman').

    Returns
    -------
    correlation_matrix : DataFrame
        The correlation matrix of the provided data. If `data` is ArrayLike, 
        the function first converts it to a DataFrame.

    Examples
    --------
    >>> from gofast.stats.utils import gocorr
    >>> data = pd.DataFrame({
    ...     'A': [1, 2, 3, 4],
    ...     'B': [4, 3, 2, 1],
    ...     'C': [1, 2, 3, 4]
    ... })
    >>> gocorr(data)
           A         B    C
    A  1.000000 -1.000000  1.000000
    B -1.000000  1.000000 -1.000000
    C  1.000000 -1.000000  1.000000

    >>> gocorr2(data, columns=['A', 'C'])
           A    C
    A  1.000000  1.000000
    C  1.000000  1.000000

    Note
    ----
    The function utilizes pandas' `corr()` method to compute the correlation 
    matrix, offering flexibility through `**kws` to use different correlation 
    computation methods. For non-DataFrame inputs, the data is first converted 
    to a DataFrame, ensuring uniform processing.
    """
    if isinstance(data, (list, np.ndarray)):
        data = pd.DataFrame(data, columns=columns if columns else range(len(data[0])))
    elif isinstance(data, pd.DataFrame) and columns:
        data = data[columns] 
    return data.corr(**kws)

def correlation(
    x: Union[str, ArrayLike], 
    y: Union[str, ArrayLike] = None, 
    data: Optional[pd.DataFrame] = None,
    **kws):
    """
    Computes the correlation between two datasets, or within a DataFrame. 
    This function allows for flexible input types, including direct array-like 
    inputs or specifying column names within a DataFrame.

    Parameters
    ----------
    x : str or ArrayLike
        The first dataset for correlation analysis. If `x` is a string, 
        `data` must be supplied and `x` should be a column name of `data`. 
        Otherwise, `x` can be array-like (list or pd.Series).
    y : str or ArrayLike, optional
        The second dataset for correlation analysis. Similar to `x`, if `y` 
        is a string, `y` should be a column name of `data` and `data` must 
        be supplied. If omitted, calculates the correlation matrix of `data`.
    data : pd.DataFrame, optional
        DataFrame containing valid numeric values if `x` and/or `y` are specified 
        as column names.
    **kws : dict
        Additional keyword arguments passed to the pandas DataFrame corr() method, 
        allowing customization of the correlation calculation, such as specifying 
        the method (e.g., 'pearson', 'kendall', 'spearman').

    Returns
    -------
    correlation_value : float or pd.DataFrame
        The correlation coefficient(s) between `x` and `y` if both are provided, 
        or the correlation matrix of `data` or `x` (if `x` is a DataFrame and `y` 
        is None).

    Examples
    --------
    >>> from gofast.stats.utils import correlation
    >>> import pandas as pd
    >>> data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> gocorr('A', 'B', data=data)
    1.0

    >>> x = [1, 2, 3]
    >>> y = [4, 5, 6]
    >>> gocorr(x, y)
    1.0

    >>> correlation(data)
    DataFrame showing correlation matrix of 'data'

    Note
    ----
    The function is designed to provide a versatile interface for correlation 
    analysis, accommodating different types of input and supporting various 
    correlation methods through keyword arguments. It utilizes pandas' `corr()` 
    method for DataFrame inputs, enabling comprehensive correlation analysis within 
    tabular data.
    """
 
    if isinstance(x, str) and data is not None:
        x = data[x]
    if isinstance(y, str) and data is not None:
        y = data[y]
    if isinstance(x, pd.DataFrame) and y is None:
        # Compute correlation matrix of DataFrame
        return x.corr(**kws)
    elif isinstance(x, pd.Series) and isinstance(y, pd.Series):
        # Compute correlation between two Series
        return x.corr(y, **kws)
    else:
        # Convert array-like inputs to Series and compute correlation
        x_series = pd.Series(x)
        y_series = pd.Series(y) if y is not None else pd.Series()
        return x_series.corr(y_series, **kws) if not y_series.empty else x_series.corr(**kws)

@DynamicMethod ( 
    expected_type="both", 
    capture_columns=True, 
    drop_na=True
   )
def goiqr(
    data: Union[ArrayLike, DataFrame], /, 
    columns: List[str] = None,
    as_frame: bool = False,
    **kws
    ):
    """
    Computes the interquartile range (IQR) of numeric data provided either as 
    an array-like structure or within a pandas DataFrame. 
    
    The IQR is calculated as the difference between the 75th and 25th 
    percentiles of the data, offering insight into the variability and 
    spread of the dataset.

    Parameters
    ----------
    data : ArrayLike or DataFrame
        The input data from which to calculate the IQR. Can be a list, numpy 
        array, or pandas DataFrame containing valid numeric values.
    columns : list of str, optional
        List of column names to consider for the calculation if `data` is a 
        DataFrame. If not provided, the IQR is calculated for all numeric 
        columns in the DataFrame.
    as_frame : bool, default False
        Indicates whether to convert the input data to a DataFrame before 
        proceeding with the calculation. Relevant when input data is not 
        already a DataFrame.
    **kws : dict
        Additional keyword arguments passed to the numpy percentile function, 
        allowing customization of the percentile calculation.

    Returns
    -------
    iqr_value : float or pd.Series
        The calculated IQR of the provided data. Returns a single float if 
        the data is one-dimensional or a pandas Series if the input is a 
        DataFrame and `as_frame` is True, showing the IQR for each specified 
        column.

    Examples
    --------
    >>> from gofast.stats.utils import goiqr
    >>> data_array = [1, 2, 3, 4, 5]
    >>> goiqr(data_array)
    2.0

    >>> data_df = pd.DataFrame({'A': [2, 5, 7, 8], 'B': [1, 4, 6, 9]})
    >>> goiqr(data_df, columns=['A'])
    3.5

    >>> goiqr(data_df, as_frame=True)
    A    3.5
    B    4.0
    dtype: float64

    Note
    ----
    The IQR is a robust measure of spread that is less influenced by outliers 
    than the range. This function simplifies the process of calculating the IQR, 
    especially useful in exploratory data analysis and for identifying potential 
    outliers.
    """
    if isinstance(data, (list, np.ndarray)):
        data = pd.DataFrame(data, columns=columns if columns else range(len(data)))
    elif isinstance(data, pd.DataFrame) and columns:
        data = data[columns]

    if as_frame and isinstance(data, pd.DataFrame):
        # Calculate IQR for each column and return as a pandas Series
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        iqr_values = Q3 - Q1
        return iqr_values
    else:
        # For non-DataFrame or entire DataFrame as input
        Q1, Q3 = np.percentile(data, [25, 75], **kws)
        return Q3 - Q1

@make_data_dynamic('numeric', capture_columns=True)
def z_scores(
    data: Union[ArrayLike, DataFrame], /, 
    columns: List[str] = None, 
    as_frame: bool = False, 
    **kws
    ):
    """
    Computes the Z-scores for each data point in a dataset, indicating how many 
    standard deviations an element is from the mean. This standardization process 
    is crucial for many statistical analyses and data processing tasks.

    Parameters
    ----------
    data : ArrayLike or DataFrame
        The input data from which to calculate Z-scores. Can be a list, numpy 
        array, or pandas DataFrame containing valid numeric values.
    columns : list of str, optional
        List of column names for which to calculate Z-scores if `data` is a 
        DataFrame. If not provided, Z-scores are calculated for all numeric 
        columns in the DataFrame.
    as_frame : bool, default False
        Indicates whether to convert the input data to a DataFrame before 
        proceeding with the calculation. Relevant when input data is not 
        already a DataFrame.
    **kws : dict
        Additional keyword arguments for data sanitization, not directly used 
        in Z-score calculation.

    Returns
    -------
    z_scores_value : np.ndarray or pd.DataFrame
        The calculated Z-scores of the provided data. Returns a numpy array 
        if the data is one-dimensional or a pandas DataFrame if the input is 
        a DataFrame and `as_frame` is True.

    Examples
    --------
    >>> from gofast.stats.utils import z_scores
    >>> data_array = [1, 2, 3, 4, 5]
    >>> z_scores(data_array)
    [-1.41421356, -0.70710678, 0., 0.70710678, 1.41421356]

    >>> data_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> z_scores(data_df, as_frame=True)
             A         B
    0 -1.224745 -1.224745
    1  0.000000  0.000000
    2  1.224745  1.224745

    Note
    ----
    Z-score standardization is widely used in data pre-processing to normalize 
    the distribution of data points. This function facilitates such 
    standardization, making it easier to perform comparative analyses across 
    different datasets or features.
    """
    if isinstance(data, (list, np.ndarray)):
        data = np.array(data)
    elif isinstance(data, pd.DataFrame) and columns:
        data = data[columns]

    if as_frame and isinstance(data, pd.DataFrame):
        # Calculate Z-scores for each column and return as a DataFrame
        mean = data.mean()
        std_dev = data.std()
        z_scores_df = (data - mean) / std_dev
        return z_scores_df
    else:
        # For non-DataFrame or entire DataFrame as input
        mean = np.mean(data, axis=0)
        std_dev = np.std(data, axis=0)
        z_scores_array = (data - mean) / std_dev
        return z_scores_array

@make_data_dynamic('numeric', capture_columns=True)
def godescr_stats_summary(
    data:ArrayLike | DataFrame,
    /,  
    columns: list=None,
    as_frame: bool=..., 
    **kws
    ):
    """ Generates a summary of descriptive statistics for a dataset.
    
    Parameters 
    ----------
    data: ArrayLike, pd.DataFrame 
       Data frame containing valid numeric values.
       
    as_frame: bool, default=False, 
       To convert data as a frame before proceeding. 
    columns: list, 
       Columns to construct the data. 
    kws: dict,  
       Additional keywords arguments for sanitizing the data 
       before proceedings. 
       
    Return 
    ------

    """

    return data.describe(**kws)

@make_data_dynamic('numeric', capture_columns=True)
def skewness(
    data:ArrayLike | DataFrame,
    /,  
    columns: list=None,
    as_frame: bool=..., 
    **kws
    ):
    """ Measures the skewness of the data distribution.
    
    Parameters 
    ----------
    data: ArrayLike, pd.DataFrame 
       Data frame containing valid numeric values. 
    
    columns: list, 
       Columns to construct the data. 
       
    as_frame: bool, default=False, 
       To convert data as a frame before proceeding. 
       
    kws: dict,  
       Additional keywords arguments for sanitizing the data 
       before proceedings. 
       
    Return 
    ------
    skewness_value: skewness value of the data 
    """
    as_frame, = ellipsis2false(as_frame )
    data = build_data_if(data, columns = columns, to_frame= as_frame  )
    data = to_numeric_dtypes(
        data, pop_cat_features= True, return_feature_types= False, **kws )
    return stats.skew(data)

@make_data_dynamic('numeric', capture_columns=True)
def kurtosis(
    data:ArrayLike | DataFrame,
    /,  
    columns: list=None,
    as_frame: bool=..., 
    **kws
    ):
    """ Measures the kurtosis of the data distribution.
    
    Parameters 
    ----------
    data: ArrayLike, pd.DataFrame 
       Data frame containing valid numeric values. 
    
    columns: list, 
       Columns to construct the data. 
    
    as_frame: bool, default=False, 
       To convert data as a frame before proceeding. 
       
    kws: dict,  
       Additional keywords arguments for sanitizing the data 
       before proceedings. 
       
    Return 
    ------
    kurtosis_value: kurtosis value of the data 
    """
    as_frame, = ellipsis2false(as_frame )
    data = build_data_if(data, columns = columns, to_frame= as_frame  )
    data = to_numeric_dtypes(
        data, pop_cat_features= True, return_feature_types= False, **kws )
    
    return stats.kurtosis(data)


def t_test_independent(
    sample1, 
    sample2, 
    alpha=0.05
    ):
    """
    Perform an independent T-test for comparing two samples.
    
    :param sample1: First sample, a list or array of values.
    :param sample2: Second sample, a list or array of values.
    :param alpha: Significance level, default is 0.05.
    :return: T-statistic, p-value, and whether to reject the null hypothesis.
    """
    t_stat, p_value = stats.ttest_ind(sample1, sample2)
    reject_null = p_value < alpha
    return t_stat, p_value, reject_null

def linear_regression(
    x=None , 
    y=None, 
    data=None, 
    columns =None, 
    **kws):
    """
    Perform linear regression analysis between two variables.
    
    Parameters 
    ----------
    x: str, list, Arraylike  
       Independent variable. First array for coorrelation analysis. 
       When `x` is a string `data` must be supplied. `x` should be a 
       column name of the data. 
           
    y: str, list, Arraylike 
       Dependent variable. Second array for coorrelation analysis. 
       When `y` is given as string it should be a column name of the data 
       and `data`must be supplied instead.
       
    data: ArrayLike, pd.DataFrame 
       Data frame containing valid numeric values. 
     
    columns: list, 
      Columns names to construct a pd.DataFrame before operating. 
      This can be useful to specify as `x` and `y` values to 
      retrieve in a large dataset. 
      
    kws: dict,  
       Additional keywords arguments for sanitizing the data 
       before proceedings. 
    Return 
    -------
    LinearRegression model, coefficients, and intercept.
    
    """
    if data is not None: 
        data = build_data_if(data, columns = columns )
        data = to_numeric_dtypes(
            data, pop_cat_features= True, return_feature_types= False,
            **kws )
        
    x, y = assert_xy_in(x, y, data =data, xy_numeric= True )
    
    model = LinearRegression()
    model.fit(np.array(x).reshape(-1, 1), y)
    return model, model.coef_, model.intercept_

@make_data_dynamic('numeric', capture_columns=True)
def chi_squared_test(
    data: ArrayLike | DataFrame,
    /,  
    alpha=0.05, 
    columns: list=None,
    as_frame: bool=True,   
     **kws 
     ):
    """
    Perform a Chi-Squared test for independence between two categorical 
    variables.
    
    Parameters 
    -----------
    
    data: pd.DataFrame
       Contingency table as a pandas DataFrame.
    alpha: float, default=.005 
       Significance level.
       
    columns: list, 
      Columns names to construct a pd.DataFrame before operating. This is 
      usefull when arraylike is given.
      
    as_frame: bool, default=False, 
       To convert data as a frame before proceeding. 
       
    Return
    --------
    Chi-squared statistic, p-value, and whether to reject the null hypothesis.
    
    """
    as_frame, = ellipsis2false(as_frame )
    data = build_data_if(data, columns = columns, to_frame= as_frame  )
    data = to_numeric_dtypes(
        data, return_feature_types= False, **kws )
    chi2_stat, p_value, _, _ = stats.chi2_contingency(data)
    reject_null = p_value < alpha
    return chi2_stat, p_value, reject_null

# XXX todo
@DynamicMethod( 
    'categorical',
    capture_columns=False, 
    treat_int_as_categorical=True, 
    encode_categories= True
  )
def anova_test(
        data: Union[Dict[str, List[float]], ArrayLike, DataFrame], 
        *groups: Union[List[str], ArrayLike], alpha: float = 0.05,
        view: bool = False,
        cmap: str = 'viridis',
        fig_size: Tuple[int, int] = (12, 5), 
        ):
    """
    Perform ANOVA test to compare means across multiple groups.

    Parameters
    ----------
    data : dict, np.ndarray, pd.DataFrame
        The input data from which groups are extracted. Can be a dictionary
        with group names as keys and lists of values as values, a NumPy
        array if `groups` are indices, or a pandas DataFrame with `groups`
        specifying column names.
    groups : List[str] or np.ndarray
        The names or indices of the groups to compare, extracted from `data`.
        If `data` is a dictionary or DataFrame, `groups` should be the keys
        or column names. If `data` is an array, `groups` should be indices
        specifying which slices of the array to compare.
    alpha : float, optional
        The significance level for the ANOVA test. Default is 0.05.

    Returns
    -------
    tuple
        A tuple containing the F-statistic, the p-value, and a boolean indicating
        whether to reject the null hypothesis (True if p-value < alpha).

    Examples
    --------
    >>> data = {'group1': [1, 2, 3], 'group2': [4, 5, 6], 'group3': [7, 8, 9]}
    >>> f_stat, p_value, reject_null = anova_test(data, 'group1', 'group2', 'group3')
    >>> print(f_stat, p_value, reject_null)

    >>> df = pd.DataFrame(data)
    >>> f_stat, p_value, reject_null = anova_test(df, 'group1', 'group2', 'group3')
    >>> print(f_stat, p_value, reject_null)
    """
    # Extract groups based on the type of `data`
    if isinstance(data, dict):
        group_values = [data[group] for group in groups]
    elif isinstance(data, np.ndarray):
        group_values = [data[group] for group in groups]
    elif isinstance(data, pd.DataFrame):
        group_values = [data[group].values for group in groups]
    else:
        raise ValueError("Unsupported data type. `data` must be a dict, np.ndarray, or pd.DataFrame.")

    # Perform ANOVA
    f_stat, p_value = stats.f_oneway(*group_values)
    reject_null = p_value < alpha

    return f_stat, p_value, reject_null


def anova_test(
    data: Union[Dict[str, List[float]], ArrayLike, DataFrame], 
    *groups: Optional[Union[List[str], ArrayLike]], 
    alpha: float = 0.05,
    view: bool = False,
    cmap: str = 'viridis',
    fig_size: Tuple[int, int] = (12, 5)):
    """
    Perform ANOVA test to compare means across multiple groups, with an option to 
    visualize the results via a box plot.

    Parameters
    ----------
    data : dict, np.ndarray, pd.DataFrame
        The input data from which groups are extracted. Can be a dictionary
        with group names as keys and lists of values as values, a NumPy
        array if `groups` are indices, or a pandas DataFrame with `groups`
        specifying column names.
    groups : List[str] or np.ndarray, optional
        The names or indices of the groups to compare, extracted from `data`.
        If `data` is a DataFrame and `groups` is not provided, all columns are used.
    alpha : float, optional
        The significance level for the ANOVA test. Default is 0.05.
    view : bool, optional
        If True, generates a box plot of the group distributions. Default is False.
    cmap : str, optional
        The colormap for the box plot. Default is 'viridis'.
    fig_size : Tuple[int, int], optional
        Figure size for the box plot. Default is (12, 5).

    Returns
    -------
    tuple
        A tuple containing the F-statistic, the p-value, and a boolean indicating
        whether to reject the null hypothesis (True if p-value < alpha).

    Examples
    --------
    >>> from gofast.stats.utils import anova_test
    >>> data = {'group1': [1, 2, 3], 'group2': [4, 5, 6], 'group3': [7, 8, 9]}
    >>> f_stat, p_value, reject_null = anova_test(data, 'group1', 'group2', 'group3')
    >>> print(f_stat, p_value, reject_null)
    """
    if isinstance(data, pd.DataFrame):
        if not groups:
            group_values = [data[col].values for col in data.columns]
        else:
            if not all(group in data.columns for group in groups):
                raise ValueError("All specified groups must be valid column names in the DataFrame.")
            group_values = [data[group].values for group in groups]
    elif isinstance(data, dict):
        group_values = [data[group] for group in groups]
    elif isinstance(data, (np.ndarray, list)):
        group_values = [data[group] for group in groups]
    else:
        raise ValueError("Unsupported data type. `data` must be a dict, np.ndarray, or pd.DataFrame.")

    # Perform ANOVA
    f_stat, p_value = stats.f_oneway(*group_values)
    reject_null = p_value < alpha

    # Optionally generate a box plot
    if view:
        plt.figure(figsize=fig_size)
        plt.boxplot(group_values, patch_artist=True)
        plt.xticks(range(1, len(groups) + 1), groups, rotation=45)
        plt.title('ANOVA Test Box Plot')
        plt.xlabel('Groups')
        plt.ylabel('Values')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

    return f_stat, p_value, reject_null


@make_data_dynamic('numeric', capture_columns=True)
def kmeans(
    data:ArrayLike | DataFrame,
    /, 
    n_clusters=3,  
    columns: list=None,
    as_frame: bool=..., 
    **kws
    ):
    """
    Apply K-Means clustering to the dataset.
    
    Parameters 
    ---------
    data: ArrayLike, pd.DataFrame 
       Multidimensional data, typically a pandas DataFrame or a 2D array.

    n_clusters: int, 
        Number of clusters, default is 3.
        
    kws: dict,  
       Additional keywords arguments for sanitizing the data 
       before proceedings. 
       
    :return: Fitted KMeans model and cluster labels for each data point.
    """
    as_frame, = ellipsis2false(as_frame )
    data = build_data_if(data, columns = columns, to_frame= as_frame  )
    data = to_numeric_dtypes(
        data, pop_cat_features= True, return_feature_types= False, **kws )
    
    km = KMeans(n_clusters=n_clusters, **kws )
    km.fit(data)
    return km, km.labels_


def harmonic_mean(data):
    """
    Calculate the harmonic mean of a data set.

    The harmonic mean is the reciprocal of the arithmetic mean of the reciprocals
    of the data points. It is a measure of central tendency and is used especially
    for rates and ratios.

    Parameters
    ----------
    data : array_like
        An array, any object exposing the array interface, containing
        data for which the harmonic mean is desired. Must be greater than 0.

    Returns
    -------
    h_mean : float
        The harmonic mean of the data set.

    Raises
    ------
    ValueError
        If any data point is less than or equal to zero.

    Examples
    --------
    >>> harmonic_mean([1, 2, 4])
    1.7142857142857142

    >>> harmonic_mean([1, 0, 2])
    ValueError: Data points must be greater than 0.

    >>> harmonic_mean(np.array([2.5, 3.0, 10.0]))
    3.5294117647058822
    """
    data = np.asarray(data)  # Ensure input is array-like
    if np.any(data <= 0):
        raise ValueError("Data points must be greater than 0.")

    return len(data) / np.sum(1.0 / data)

def weighted_median(data, weights):
    """
    Compute the weighted median of a data set.

    The weighted median is a median where each value in the data set
    is assigned a weight. It is the value such that the sum of the weights
    is equal on both sides of the sorted list.

    Parameters
    ----------
    data : array_like
        Data for which the weighted median is desired.
    weights : array_like
        Weights for each element in `data`.

    Returns
    -------
    w_median : float
        The weighted median of the data set.

    Example
    -------
    >>> weighted_median([1, 2, 3], [3, 1, 2])
    2.0
    """
    data, weights = np.array(data), np.array(weights)
    sorted_indices = np.argsort(data)
    sorted_data, sorted_weights = ( 
        data[sorted_indices], weights[sorted_indices]) 
    cumulative_weights = np.cumsum(sorted_weights)
    median_idx = np.where(
        cumulative_weights >= 0.5 * np.sum(sorted_weights))[0][0]
    return sorted_data[median_idx]

def bootstrap(data, /,  n=1000, func=np.mean):
    """
    Perform bootstrapping to estimate the distribution of a statistic.

    Bootstrapping is a resampling technique used to estimate statistics
    on a population by sampling a dataset with replacement.

    Parameters
    ----------
    data : array_like
        The data to bootstrap.
    n : int, optional
        Number of bootstrap samples to generate.
    func : callable, optional
        The statistic to compute from the resampled data.

    Returns
    -------
    bootstrapped_stats : ndarray
        Array of bootstrapped statistic values.

    Example
    -------
    >>> np.random.seed(0)
    >>> bootstrap(np.arange(10), n=100, func=np.mean)
    array([4.5, 4.7, 4.9, ..., 4.4, 4.6, 4.8])
    """
    bootstrapped_stats = []
    for _ in range(n):
        sample = np.random.choice(data, size=len(data), replace=True)
        stat = func(sample)
        bootstrapped_stats.append(stat)
        
    return np.array(bootstrapped_stats)

def kaplan_meier_analysis(durations, event_observed, **kws):
    """
    Perform Kaplan-Meier Survival Analysis.

    Kaplan-Meier Survival Analysis is used to estimate the survival function
    from lifetime data. It is a non-parametric statistic.

    Parameters
    ----------
    durations : array_like
        Observed lifetimes (durations).
    event_observed : array_like
        Boolean array where 1 if the event is observed and 0 is censored.

    kws: dict, 
       Additional keyword arguments passed to 
       :class:`lifelines.KaplanMeierFitter`.
       
    Returns
    -------
    kmf : KaplanMeierFitter
        Fitted Kaplan-Meier estimator.

    Example
    -------
    >>> durations = [5, 6, 6, 2.5, 4, 4]
    >>> event_observed = [1, 0, 0, 1, 1, 1]
    >>> kmf = kaplan_meier_analysis(durations, event_observed)
    >>> kmf.plot_survival_function()
    """
    import_optional_dependency ('lifelines')
    from lifelines import KaplanMeierFitter
    
    kmf = KaplanMeierFitter(**kws)
    kmf.fit(durations, event_observed=event_observed)
    return kmf

def get_gini_coeffs(data, /, ):
    """
    Calculate the Gini coefficient of a data set.

    The Gini coefficient is a measure of inequality of a distribution.
    It is defined as a ratio with values between 0 and 1, where 0
    corresponds to perfect equality and 1 to perfect inequality.

    Parameters
    ----------
    data : array_like
        Data set for which to calculate the Gini coefficient.

    Returns
    -------
    gini : float
        The Gini coefficient.

    Example
    -------
    >>> gini_coefficient([1, 2, 3, 4, 5])
    0.26666666666666666
    """
    # The array is sorted, the index is used for cumulating sums.
    data = np.sort(np.array(data))
    n = data.size
    index = np.arange(1, n+1)
    # The Gini coefficient is then the ratio of two areas.
    return (np.sum((2 * index - n - 1) * data)) / (n * np.sum(data))

def multidim_scaling(data, /,  n_components=2, **kws):
    """
    Perform Multidimensional Scaling (MDS).

    MDS is a means of visualizing the level of similarity of individual cases
    of a dataset in a lower-dimensional space.

    Parameters
    ----------
    data : array_like
        Data set for MDS.
    n_components : int, optional
        Number of dimensions in which to immerse the dissimilarities.

    kws: dict, 
       Additional keyword arguments passed to 
       :class:`sklearn.manifold.MDS 
    Returns
    -------
    mds_result : ndarray
        Coordinates of the data in the MDS space.

    Example
    -------
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> mds_coordinates = multidim_scaling(iris.data, n_components=2)
    >>> mds_coordinates.shape
    (150, 2)
    """
    
    mds = MDS(n_components=n_components, **kws)
    return mds.fit_transform(data)


def dca_analysis(data, /, ):
    """
    Perform Detrended Correspondence Analysis (DCA).

    DCA is widely used in ecology to find the main factors or gradients
    in large, species-rich but usually sparse data matrices.

    Parameters
    ----------
    data : array_like
        Data set for DCA.

    Returns
    -------
    dca_result : OrdinationResults
        Results of DCA, including axis scores and explained variance.

    Example
    -------
    >>> import pandas as pd
    >>> data = pd.DataFrame({'species1': [1, 0, 3], 'species2': [0, 4, 1]})
    >>> dca_result = dca_analysis(data)
    >>> print(dca_result.axes)
    """
    import_optional_dependency('skbio')
    from skbio.stats.ordination import detrended_correspondence_analysis
    
    return detrended_correspondence_analysis(data)

def spectral_clustering(
    data, /, 
    n_clusters=2, 
    assign_labels='discretize',
    random_state=0, 
    **kws 
    ):
    """
    Perform Spectral Clustering.

    Spectral Clustering uses the spectrum of the similarity matrix of the data
    to perform dimensionality reduction before clustering. It's particularly
    useful when the structure of the individual clusters is highly non-convex.

    Parameters
    ----------
    data : array_like
        Data set for clustering.
    n_clusters : int, optional
        The number of clusters to form.
        
    assign_labels : {'kmeans', 'discretize', 'cluster_qr'}, default='kmeans'
        The strategy for assigning labels in the embedding space. There are two
        ways to assign labels after the Laplacian embedding. k-means is a
        popular choice, but it can be sensitive to initialization.
        Discretization is another approach which is less sensitive to random
        initialization [3]_.
        The cluster_qr method [5]_ directly extract clusters from eigenvectors
        in spectral clustering. In contrast to k-means and discretization, cluster_qr
        has no tuning parameters and runs no iterations, yet may outperform
        k-means and discretization in terms of both quality and speed.
        
    random_state : int, RandomState instance, default=None
        A pseudo random number generator used for the initialization
        of the lobpcg eigenvectors decomposition when `eigen_solver ==
        'amg'`, and for the K-Means initialization. Use an int to make
        the results deterministic across calls (See
        :term:`Glossary <random_state>`).

        .. note::
            When using `eigen_solver == 'amg'`,
            it is necessary to also fix the global numpy seed with
            `np.random.seed(int)` to get deterministic results. See
            https://github.com/pyamg/pyamg/issues/139 for further
            information.
            
    kws: dict, 
       Additional keywords argument of 
       :class:`sklearn.cluster.SpectralClustering`
       
    Returns
    -------
    labels : ndarray
        Labels of each point.

    Example
    -------
    >>> from sklearn.datasets import make_moons
    >>> X, _ = make_moons(n_samples=200, noise=0.05, random_state=0)
    >>> labels = spectral_clustering(X, n_clusters=2)
    >>> np.unique(labels)
    array([0, 1])
    """
    clustering = SpectralClustering(n_clusters=n_clusters, 
                                    assign_labels=assign_labels,
                                    random_state=random_state, 
                                    **kws)
    return clustering.fit_predict(data)



def levene_test(*args):
    """
    Perform Levene's test for equal variances.

    Levene's test assesses the homogeneity of variance in different samples.

    Parameters
    ----------
    *args : array_like
        The sample data, possibly with different lengths.

    Returns
    -------
    statistic : float
        The test statistic.
    p_value : float
        The p-value for the test.

    Example
    -------
    >>> levene_test([1, 2, 3], [4, 5, 6], [7, 8, 9])
    (0.0, 1.0)
    """
    return stats.levene(*args)

def kolmogorov_smirnov_test(sample1, sample2):
    """
    Perform the Kolmogorov-Smirnov test for goodness of fit.

    This test compares the distributions of two independent samples.

    Parameters
    ----------
    sample1, sample2 : array_like
        Two arrays of sample observations assumed to be drawn from a continuous
        distribution.

    Returns
    -------
    statistic : float
        KS statistic.
    p_value : float
        Two-tailed p-value.

    Example
    -------
    >>> kolmogorov_smirnov_test([1, 2, 3, 4], [1, 2, 3, 5])
    (0.25, 0.9999999999999999)
    """
    return stats.ks_2samp(sample1, sample2)

def cronbach_alpha(items_scores):
    """
    Calculate Cronbach's Alpha for a set of test items.

    Cronbach's Alpha is a measure of internal consistency, that is, how closely
    related a set of items are as a group.

    Parameters
    ----------
    items_scores : array_like
        A 2D array where rows are items and columns are scoring for each item.

    Returns
    -------
    alpha : float
        Cronbach's Alpha.

    Example
    -------
    >>> scores = [[2, 3, 4], [4, 4, 5], [3, 5, 4]]
    >>> cronbach_alpha(scores)
    0.8964214570007954
    """
    items_scores = np.asarray(items_scores)
    item_variances = items_scores.var(axis=1, ddof=1)
    total_variance = items_scores.sum(axis=0).var(ddof=1)
    n_items = items_scores.shape[0]
    return (n_items / (n_items - 1)) * (1 - item_variances.sum() / total_variance)


def friedman_test(*args, method='auto'):
    """
    Perform a Friedman test to compare more than two related groups,
    with an option to control the test method.

    The Friedman test is a non-parametric test used to detect differences
    in treatments across multiple test attempts. It is particularly useful
    for small sample sizes and for data that do not follow a normal
    distribution.

    Parameters
    ----------
    *args : array_like
        The arrays must have the same number of elements, each representing
        a related group. These groups are typically different treatments
        applied to the same subjects.
    method : {'auto', 'exact', 'asymptotic'}, optional
        The method to use for the test:
        - 'auto' : Use exact method for small sample sizes and asymptotic
          method for larger samples.
        - 'exact' : Use the exact distribution of the test statistic.
        - 'asymptotic' : Use the asymptotic distribution of the test statistic.

    Returns
    -------
    statistic : float
        The test statistic, in this case, the Friedman statistic.
    p_value : float
        The p-value for the test. A p-value less than a significance level
        (commonly 0.05) indicates significant differences among the groups.

    Raises
    ------
    ValueError
        If the input arrays have different lengths.

    Example
    -------
    >>> from gofast.stats import friedman_test
    >>> group1 = [20, 21, 19, 20, 21]
    >>> group2 = [19, 20, 18, 21, 20]
    >>> group3 = [21, 22, 20, 22, 21]
    >>> statistic, p_value = friedman_test(group1, group2, group3, 
                                           method='auto')
    >>> print(f'Friedman statistic: {statistic}, p-value: {p_value}')

    Notes
    -----
    The Friedman test is widely used in scenarios where you want to compare
    the effects of different treatments or conditions on the same subjects,
    especially in medical, psychological, and other scientific research.
    It is an alternative to ANOVA when the normality assumption is not met.

    References
    ----------
    Friedman, Milton. (1937). The use of ranks to avoid the assumption
    of normality implicit in the analysis of variance. Journal of the
    American Statistical Association.
    """
    
    # Check that all input arrays have the same length
    if len(set(map(len, args))) != 1:
        raise ValueError("All input arrays must have the same length.")

    return stats.friedmanchisquare(*args, method=method)

@make_data_dynamic('numeric', capture_columns=True)
def statistical_tests(data, test_type, *args, **kwargs):
    """
    Perform various statistical tests including Repeated Measures ANOVA, 
    Cochran’s Q Test, McNemar’s Test, Kruskal-Wallis H Test, 
    Wilcoxon Signed-Rank Test, and t-Test (Paired or Independent).

    Parameters
    ----------
    data : DataFrame or array_like
        The data to be used in the test. Format and structure depend on the test.
    test_type : str
        Type of the test to perform. Options include 'rm_anova', 'cochran_q', 
        'mcnemar', 'kruskal_wallis', 'wilcoxon', 'ttest_paired', 'ttest_indep'.
        
    *args : additional arguments
        Additional arguments required by the specific test.
    **kwargs : additional keyword arguments
        Additional keyword arguments required by the specific test.

    Returns
    -------
    result : Result object
        The result of the statistical test. Includes test statistic and p-value.

    Test Details
    ------------
    - Repeated Measures ANOVA ('rm_anova'):
        Used for comparing the means of three or more groups on the same subjects.
        Commonly used in experiments where subjects undergo multiple treatments.
        
    - Cochran’s Q Test ('cochran_q'):
        A non-parametric test for comparing three or more matched groups. It is the 
        extension of the McNemar test and is used for binary (two-outcome) data.

    - McNemar’s Test ('mcnemar'):
        Used for binary classification to compare the proportion of misclassified 
        instances between two models on the same dataset.

    - Kruskal-Wallis H Test ('kruskal_wallis'):
        A non-parametric version of ANOVA, used for comparing two or more independent 
        groups. Suitable when the data does not meet ANOVA assumptions.

    - Wilcoxon Signed-Rank Test ('wilcoxon'):
        A non-parametric test to compare two related samples. It's used when the 
        population cannot be assumed to be normally distributed.

    - Paired t-Test ('ttest_paired'):
        Compares the means of two related groups. It's used when the same subjects 
        are used in both groups (e.g., before-after studies).

    - Independent t-Test ('ttest_indep'):
        Compares the means of two independent groups. Used when different subjects 
        are used in each group or condition.
        
    Examples
    --------
    >>> import numpy as np 
    >>> import pandas as pd 
    >>> from gofasts.stats import statistical_tests
    
    For Repeated Measures ANOVA:
    >>> data = pd.DataFrame({'subject': [1, 2, 3, 4, 5],
                             'condition1': [20, 19, 22, 21, 18],
                             'condition2': [22, 20, 24, 23, 19]})
    >>> result = statistical_tests(data, 'rm_anova', subject='subject', 
                                   within=['condition1', 'condition2'])
    
    For Cochran’s Q Test:
    >>> data = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]])
    >>> result = statistical_tests(data, 'cochran_q')

    For McNemar’s Test:
    >>> data = np.array([[10, 2], [3, 5]])
    >>> result = statistical_tests(data, 'mcnemar')

    For Kruskal-Wallis H Test:
    >>> group1 = [20, 21, 19, 20, 21]
    >>> group2 = [19, 20, 18, 21, 20]
    >>> group3 = [21, 22, 20, 22, 21]
    >>> result = statistical_tests([group1, group2, group3], 'kruskal_wallis')

    For Wilcoxon Signed-Rank Test:
    >>> data1 = [20, 21, 19, 20, 21]
    >>> data2 = [19, 20, 18, 21, 20]
    >>> result = statistical_tests((data1, data2), 'wilcoxon')

    For Paired t-Test:
    >>> data1 = [20, 21, 19, 20, 21]
    >>> data2 = [19, 20, 18, 21, 20]
    >>> result = statistical_tests((data1, data2), 'ttest_paired')

    For Independent t-Test:
    >>> data1 = [20, 21, 19, 20, 21]
    >>> data2 = [22, 23, 21, 22, 24]
    >>> result = statistical_tests((data1, data2), 'ttest_indep')

    Notes
    -----
    Ensure that the data is prepared according to the requirements of each test.
    For example, data for Repeated Measures ANOVA should be in long format.
    """
    import_optional_dependency("statsmodels")
    from statsmodels.stats.anova import AnovaRM
    from statsmodels.stats.contingency_tables import mcnemar
    
    test_functions = {
        'rm_anova': lambda: AnovaRM(data, **kwargs).fit(),
        'cochran_q': lambda: stats.cochrans_q(*args),
        'mcnemar': lambda: mcnemar(*args, **kwargs),
        'kruskal_wallis': lambda: stats.kruskal(*args),
        'wilcoxon': lambda: stats.wilcoxon(*args),
        'ttest_paired': lambda: stats.ttest_rel(*args),
        'ttest_indep': lambda: stats.ttest_ind(*args)
    }

    try:
        return test_functions[test_type]()
    except KeyError:
        raise ValueError(f"Invalid test type '{test_type}' specified.")


def data_view ( x=None, y=None, data=None, kind='', **plot_kws):
    """ Visualize data or the groups of columns (`x` and `y`)  in the data 
    
    Parameters 
    ---------
    x: ArrayLike 1d or str, optional 
      if `str` , data must be provided and must exist in the data columns names. 
      Otherwise an error raise 
    y: arraylike 1d or str, optional 
      if `str` 
    
    
    """
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    