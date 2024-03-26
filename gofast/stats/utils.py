# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
:func:`gofast.stats.utils` provides a comprehensive toolkit for performing basic 
statistical analyses. They can be easily integrated into data analysis 
workflows and are versatile enough to handle a wide range 
of data types and structures.
"""
from __future__ import annotations 
from itertools import product
import numpy as np
from scipy import stats
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.linear_model import LinearRegression
from sklearn.manifold import MDS

from .._typing import Optional, List, Dict, Union, Tuple, Callable, Any
from .._typing import NumPyFunction, DataFrame, ArrayLike, Array1D, Series
from ..decorators import DynamicMethod
from ..tools.validator import assert_xy_in, is_frame, check_consistent_length 
from ..tools.validator import _is_arraylike_1d 
from ..tools.coreutils import ensure_visualization_compatibility, ellipsis2false 
from ..tools.coreutils import process_and_extract_data, to_series_if 
from ..tools.coreutils import get_colors_and_alphas, normalize_string 
from ..tools.coreutils import smart_format, check_uniform_type
from ..tools.funcutils import make_data_dynamic, ensure_pkg
from ..tools.funcutils import flatten_data_if, update_series_index 
from ..tools.funcutils import update_index, convert_and_format_data
from ..tools.funcutils import series_naming 

__all__= [
    "anova_test",
    "bootstrap",
    "check_and_fix_rm_anova_data",
    "chi2_test",
    "corr",
    "correlation",
    "cronbach_alpha",
    "dca_analysis",
    "describe",
    "friedman_test",
    "gini_coeffs",
    "get_range",
    "hmean",
    "iqr",
    "kaplan_meier_analysis",
    "kolmogorov_smirnov_test",
    "kruskal_wallis_test",
    "kurtosis",
    "levene_test",
    "mean",
    "mds_similarity",
    "median",
    "mcnemar_test",
    "mixed_effects_model", 
    "mode",
    "perform_kmeans_clustering",
    "perform_linear_regression",
    "perform_spectral_clustering",
    "quantile",
    "quartiles",
    "skew",
    "std",
    "statistical_tests",
    "t_test_independent",
    "var",
    "wmedian",
    "wilcoxon_signed_rank_test",
    "z_scores",
    "paired_t_test"
]

@make_data_dynamic(capture_columns=True)
def mean(
    data: Union[ArrayLike, DataFrame], 
    columns: Optional[List[str]] = None,
    axis: int =None, 
    view: bool = False, 
    cmap: str = 'viridis', 
    as_frame: bool = False, 
    fig_size: Optional[Tuple[int, int]] = None, 
    **kws
):
    """
    Calculates the mean of numeric data provided as an array-like structure 
    or within a pandas DataFrame.
    
    Optionally, the result can be returned as a pandas DataFrame or Series, 
    and the data distribution along with the mean can be visualized.

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
    axis: optional, {index (0), columns {1}}
       Axis for the function to be applied on.  Default is 0.
    view : bool, default=False
        If True, visualizes the distribution of the data and highlights the
        calculated mean values.
    cmap : str, default='viridis'
        Colormap for the visualization. Only applicable if `view` is True.
    as_frame : bool, default=False
        If True, the result is returned as a pandas DataFrame or Series, 
        depending on the dimensionality of the calculated mean. Otherwise, 
        the result is returned in its native format (float or np.ndarray).
    fig_size : Optional[Tuple[int, int]], default=None
        Size of the figure for the visualization. Only applicable if `view` is True.
    **kws : dict
        Additional keyword arguments passed to the underlying numpy mean 
        function, allowing customization of the mean calculation, 
        such as specifying the axis.
         
    **kws : dict
        Additional keyword arguments passed to the underlying numpy mean 
        function, allowing customization of the mean calculation, such as 
        specifying the axis.

    Returns
    -------
    mean_value : float, np.ndarray, pd.Series, or pd.DataFrame
        The calculated mean of the provided data. The format of the return 
        value depends on the input data dimensionality and the 
        `as_frame` parameter.

    Examples
    --------
    >>> from gofast.stats.utils import mean
    >>> data_array = [1, 2, 3, 4, 5]
    >>> mean(data_array)
    3.0

    >>> data_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> mean(data_df, columns=['A'])
    2.0

    >>> mean(data_df, axis=0)
    array([2., 5.])
    
    Calculating mean from a list:

    >>> mean([1, 2, 3, 4, 5])
    3.0

    Calculating mean from a DataFrame and converting to Series:

    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> print(mean(df, as_frame=True))
    A    2.0
    B    5.0
    dtype: float64

    Visualizing data distribution and mean from DataFrame:

    >>> mean(df, view=True)
    Note
    ----
    The function is wrapped by the `make_data_dynamic` decorator, which 
    preprocesses the input data based on the specified `expected_type` and 
    `capture_columns`. This preprocessing step is crucial for handling pandas 
    DataFrames with mixed data types and for selecting specific columns for
    the calculation.
    """
    if isinstance(data, (pd.DataFrame, pd.Series)):
        axis = axis or 0
        mean_values = data.mean(axis =axis, **kws)
    else:
        data = np.array(data)
        mean_values = np.mean(data, axis =axis,  **kws)
    
    if as_frame:
        if isinstance(mean_values, np.ndarray) and mean_values.ndim > 1:
            mean_values = pd.DataFrame(mean_values, columns=columns)
        else:
            mean_values = pd.Series(
                mean_values, index=columns if columns else None, name='Mean')
    
    mean_values, view = ensure_visualization_compatibility(
        mean_values, as_frame, view, mean )
    
    if view:
        _visualize_mean(
            data, mean_values, cmap=cmap, fig_size=fig_size, axis=axis)
        
   
    return mean_values


def _visualize_mean(
        data, mean_values, cmap='viridis', fig_size=None, axis =0 ):
    """
    Visualizes the distribution of the data and highlights the mean values 
    with distinct colors.
    
    Parameters
    ----------
    data : pd.DataFrame
        The data containing the columns to be visualized.
    mean_values : pd.Series or np.ndarray
        The mean values for the columns in the data.
    cmap : str, optional
        The colormap name to be used if `data` is not a DataFrame.
    fig_size : tuple, optional
        The size of the figure to be created.

    Returns
    -------
    None
        This function does not return any value. It shows a plot.
    
    Examples
    --------
    >>> df = pd.DataFrame({'A': np.random.randn(100), 'B': np.random.randn(100) + 1})
    >>> _visualize_mean(df, df.mean())
    """
    
    plt.figure(figsize=fig_size if fig_size else (10, 6))
    mean_values, data, cols= _prepare_plot_data(mean_values, data, axis=axis)
    if isinstance(data, pd.DataFrame):
        # Generate distinct colors for each column from cmap 
        colors, alphas = get_colors_and_alphas( cols, cmap) 
        for ii, col in enumerate(cols):
            # Get the histogram data
            n, bins, patches = plt.hist(data[col], bins='auto', alpha=alphas[ii], 
                                        label=f'{col} Distribution', color=colors[ii])
            # Highlight the mean
            mean_val = mean_values[col]
            plt.axvline(mean_val, color=colors[ii], linestyle='dashed',
                        linewidth=2, label=f'{col} Mean')
            
            # Calculate the height of the mean line for placing the annotation
            mean_bin_index = np.digitize(mean_val, bins) - 1
            mean_height = n[mean_bin_index] if 0 <= mean_bin_index < len(n) else 0
            
            # Add annotation for the mean value
            plt.text(mean_val, mean_height, f'{mean_val:.2f}',
                     color=colors[ii], ha='center', va='bottom')
    else:
        data =flatten_data_if(data, squeeze= True)
        # If `data` is not a DataFrame, use a single color from the colormap
        n, bins, patches = plt.hist(
            data, bins='auto', alpha=0.7, label='Data Distribution')
        # Ensure mean_val is a scalar for annotation
        mean_val =np.mean(data)
        if isinstance(mean_val, np.ndarray):
            mean_val = mean_val.item()
        elif isinstance(mean_val, pd.Series):
            mean_val = mean_val.iloc[0]
        plt.axvline(mean_val, color='red', linestyle='dashed', 
                    linewidth=2, label='Mean')
        # Add annotation for the mean value
        mean_bin_index = np.digitize(mean_val, bins) - 1
        mean_height = n[mean_bin_index] if 0 <= mean_bin_index < len(n) else 0
        plt.text(mean_val, mean_height, f'{mean_val:.2f}',
                 color='red', ha='center', va='bottom')
    
    plt.title('Data Distribution and Mean')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


@make_data_dynamic(capture_columns=True)
def median(
    data: Union[ArrayLike, DataFrame], 
    columns: Optional[List[str]] = None,
    axis: Optional[int]=None, 
    view: bool = False, 
    cmap: str = 'viridis', 
    as_frame: bool = False, 
    fig_size: Optional[Tuple[int, int]] = None, 
    **kws
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
    axis: optional, {index (0), columns {1}}
       Axis for the function to be applied on.  Default is 0.
    view : bool, default=False
        If True, visualizes the distribution of the data and highlights the 
        calculated median values.
    cmap : str, default='viridis'
        Colormap for the visualization. Only applicable if `view` is True.
    as_frame : bool, default=False
        If True, the result is returned as a pandas DataFrame or Series, depending
        on the dimensionality of the calculated median. Otherwise, the result is
        returned in its native format (float or np.ndarray).
    fig_size : Optional[Tuple[int, int]], default=None
        Size of the figure for the visualization. Only applicable if `view` is True.
    **kws : dict
        Additional keyword arguments passed to the underlying numpy median 
        function, allowing customization of the median calculation, such as 
        specifying the axis or whether to ignore NaN values.

    Returns
    -------
    median_value : float or np.ndarray
        median_value : float, np.ndarray, pd.Series, or pd.DataFrame
            The calculated median of the provided data. The format of the return 
            value depends on the input data dimensionality and the
            `as_frame` parameter.

    Examples
    --------
    >>> from gofast.stats.utils import median
    >>> data_array = [3, 1, 4, 1, 5]
    >>> median(data_array)
    3.0

    >>> data_df = pd.DataFrame({'A': [2, 4, 7], 'B': [1, 6, 5]})
    >>> median(data_df, columns=['A'])
    4.0

    >>> median(data_df, axis=0)
    array([4., 5.])
    
    Calculating median from a DataFrame and converting to Series:

    >>> df = pd.DataFrame({'A': [2, 4, 7], 'B': [1, 6, 5]})
    >>> print(median(df, columns=['A'], as_frame=True))
    A    4.0
    dtype: float64

    Visualizing data distribution and median from DataFrame:

    >>> median(df, view=True)
    
    Note
    ----
    This function is enhanced by the `make_data_dynamic` decorator, which 
    preprocesses the input data to align with the expected numeric type and 
    selectively processes columns if specified. This preprocessing
    is particularly useful for ensuring compatibility with statistical 
    calculations and optimizing performance for DataFrame inputs with mixed
    data types.
    """
    # Preprocess data based on input type and selected columns
    # if isinstance(data, pd.DataFrame) and columns is not None:
    #     data = data[columns]
    # Calculate the median
    if isinstance(data, pd.DataFrame):
        axis = axis or 0
        median_values = data.median(axis=axis, **kws)
    else:
        data = np.array(data)
        median_values = np.median(data, axis=axis, **kws)
    
    # Convert the result to DataFrame or Series if requested
    if as_frame:
        if isinstance(median_values, np.ndarray) and median_values.ndim > 1:
            median_values = pd.DataFrame(median_values, columns=columns)
        else:
            median_values = pd.Series(
            median_values,
            index=columns if columns else None, 
            name='Median'
        )

    # Visualization of data distribution and median
    median_values, view = ensure_visualization_compatibility(
        median_values, as_frame, view, median )
    if view:
        _visualize_median(
            data, median_values, cmap=cmap, fig_size=fig_size, axis= axis )

    return median_values

def _visualize_median(
        data, median_values, cmap='viridis', fig_size=None, axis = 0 ):
    """
    Visualizes the distribution of the data and highlights the median values.
    """
    plt.figure(figsize=fig_size if fig_size else (10, 6))
    _, data, cols= _prepare_plot_data(median_values, data, axis=axis)
    if isinstance(data, pd.DataFrame):
        # Generate distinct colors for each column from cmap 
        colors, alphas = get_colors_and_alphas( cols, cmap) 
        for ii, col in enumerate (cols) :
            plt.hist(data[col], bins='auto', alpha=alphas[ii],
                     label=f'{col} Distribution', color=colors [ii])
            plt.axvline(data[col].median(), color='red', linestyle='dashed', 
                        linewidth=2, label=f'{col} Median')
    else:
        plt.hist(data, bins='auto', alpha=0.7, label='Data Distribution',
                 )
        plt.axvline(np.median(data), color='red', linestyle='dashed',
                    linewidth=2, label='Median')
    plt.title('Data Distribution and Median')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


@make_data_dynamic(capture_columns=True)
def mode(
    data: Union[ArrayLike, DataFrame], 
    columns: List[str] = None, 
    axis: Optional[int]=None, 
    as_frame: bool = False, 
    view: bool = False, 
    cmap: str = 'viridis', 
    fig_size: Optional[Tuple[int, int]] = None, 
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
    axis: optional, {index (0), columns {1}}
       Axis for the function to be applied on.  Default is 0.
    as_frame : bool, default False
        Indicates whether to convert the input data to a DataFrame before 
        proceeding. This parameter is particularly relevant when input
        data is not already a DataFrame.
    view : bool, default False
        If True, visualizes the mode(s) of the dataset along with its distribution.
    cmap : str, default 'viridis'
        Colormap for the visualization. Only applicable if `view` is True.
    fig_size : Optional[Tuple[int, int]], default None
        Size of the figure for the visualization. Only applicable if `view` is True.
    **kws : dict
        Additional keyword arguments for further data processing or passed to 
        the underlying statistical functions.
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
    >>> from gofast.stats.utils import mode
    >>> data_array = [1, 2, 2, 3, 4]
    >>> mode(data_array)
    2

    >>> data_df = pd.DataFrame({'A': [1, 2, 2, 3], 'B': [4, 4, 5, 5]})
    >>> mode(data_df, as_frame=True)
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
    as_frame,  = ellipsis2false(as_frame)
    if isinstance(data, pd.DataFrame):
        axis = axis or 0
        mode_result = data.mode(axis=axis, **kws)
    else:
        data = np.asarray(data)
        mode_result = stats.mode(data, axis=axis,  **kws).mode[0]
        mode_result = pd.Series(mode_result) if as_frame else mode_result
    
    mode_result, view = ensure_visualization_compatibility(
        mode_result, as_frame, view, mode )
    if view:
        _visualize_mode(data, mode_result, cmap=cmap, fig_size=fig_size, axis=axis )
    
    return mode_result

def _visualize_mode(data, mode_result, cmap='viridis', fig_size=None, axis=0):
    """
    Visualizes the data distribution and highlights the mode values.
    """
    plt.figure(figsize=fig_size if fig_size else (10, 6))
    _, data, cols= _prepare_plot_data(mode_result, data, axis=axis)
    if isinstance(data, pd.DataFrame):
        colors, alphas = get_colors_and_alphas( cols, cmap)
        for ii, col in enumerate (cols) :
            plt.hist(data[col], bins='auto', alpha=alphas[ii], 
                     label=f'{col} Distribution', color=colors[ii])
            for mode in np.ravel([mode_result[col]]):
                plt.axvline(x=mode, color='red', linestyle='dashed',
                            linewidth=2, label=f'{col} Mode')
    else:
        plt.hist(data, bins='auto', alpha=0.7, label='Data Distribution')
        for mode in np.ravel([mode_result]):
            plt.axvline(x=mode, color='red', linestyle='dashed', 
                        linewidth=2, label='Mode')
    plt.title('Data Distribution and Mode(s)')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    
@make_data_dynamic(capture_columns=True, reset_index=True)
def var(
    data: Union[ArrayLike, DataFrame], 
    columns: Optional[List[str]] = None, 
    axis: Optional[int] = None, 
    ddof: int = 1, 
    as_frame: bool = False, 
    view: bool = False, 
    cmap: str = 'viridis', 
    fig_size: Optional[Tuple[int, int]] = None, 
    **kws):
    r"""
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
    axis : optional, {index (0), columns (1)}
        Axis for the function to be applied on. Default is 0.
    ddof : int, default 1
        Delta Degrees of Freedom. The divisor used in the calculation is 
        N - ddof, where N represents the number of elements. By default, ddof 
        is set to 1, which computes the sample variance. If ddof is set to 0, 
        the population variance is calculated. 
        
        .. math:: \text{Variance} = \frac{\sum (x_i - \bar{x})^2}{N - ddof}
 
    as_frame : bool, default False
        Indicates whether to convert the input data to a DataFrame before 
        proceeding with the calculation. This parameter is particularly relevant 
        when input data is not already a DataFrame.
    view : bool, default False
        If True, visualizes the variance(s) of the dataset along with its distribution.
    cmap : str, default 'viridis'
        Colormap for the visualization. Only applicable if `view` is True.
    fig_size : Optional[Tuple[int, int]], default None
        Size of the figure for the visualization. Only applicable if `view` is True.
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
    >>> from gofast.stats.utils import var
    >>> data_array = [1, 2, 3, 4, 5]
    >>> var(data_array)
    2.0

    >>> data_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> var(data_df, columns=['A'], ddof=1)
    1.0

    >>> var(data_df, as_frame=True)
    A    1.0
    B    1.0
    dtype: float64
    
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> var(df, as_frame=True)
    A    1.0
    B    1.0
    dtype: float64

    >>> var(df, view=True, fig_size=(8, 4))
    Note
    ----
    The preprocessing steps, controlled by the `make_data_dynamic` decorator,
    ensure that the input data is suitable for variance calculation. This
    includes converting the data to numeric types and filtering based on 
    specified columns, enhancing the function's flexibility and applicability 
    across various data forms.
    """
    if isinstance(data, pd.DataFrame):
        axis = axis or 0 # Pandas default ddof=1
        variance_result = data.var(ddof=ddof, axis=axis, **kws)  
    else:
        data = np.asarray(data)
        # Ensure consistency with pandas
        variance_result = np.var(data, ddof=ddof, axis=axis, **kws)  

    if as_frame and not isinstance(data, pd.DataFrame):
        variance_result = pd.Series(
            variance_result, index=columns if columns else ['Variance'])

    variance_result, view = ensure_visualization_compatibility(
        variance_result, as_frame, view, var )
    
    if view:
        _visualize_variance(data, variance_result, columns=columns,
                            cmap=cmap, fig_size=fig_size, axis=axis )
    return variance_result

def _visualize_variance(data, variance_result, columns=None, cmap='viridis',
                        fig_size=None, axis=0 ):
    """
    Visualizes the distribution of the data and highlights the variance values.
    """
    plt.figure(figsize=fig_size if fig_size else (10, 6))
    _, data, cols= _prepare_plot_data(variance_result, data, axis=axis)
    if isinstance(data, pd.DataFrame):
        colors, alphas = get_colors_and_alphas( cols, cmap)
        for ii, col in enumerate(cols):
            plt.hist(data[col], bins='auto', alpha=alphas[ii], 
                     label=f'{col} Distribution', color=colors[ii])
        for col, var in variance_result.items():
            plt.axvline(x=np.sqrt(var), color='red', 
                        linestyle='dashed', linewidth=2, 
                        label=f'{col} StdDev')
    else:
        plt.hist(data, bins='auto', alpha=0.7, label='Data Distribution', 
                 )
        plt.axvline(x=np.sqrt(variance_result), color='red',
                    linestyle='dashed', linewidth=2, label='StdDev')
    plt.title('Data Distribution and Standard Deviation')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

@make_data_dynamic(
    capture_columns=True, 
    reset_index=True
    )    
def std(
    data: Union[ArrayLike, DataFrame], 
    columns: Optional[List[str]] = None, 
    axis: Optional[int] = None, 
    ddof: int = 1, 
    as_frame: bool = False, 
    view: bool = False, 
    cmap: str = 'viridis', 
    fig_size: Optional[Tuple[int, int]] = None, 
    **kws):
    r"""
    Computes the standard deviation of numeric data provided either as an 
    array-like structure or within a pandas DataFrame. This function is 
    designed to be flexible, allowing for the calculation of standard deviation 
    across entire datasets or within specified columns of a DataFrame.

    Utilizing preprocessing parameters such as `expected_type` and 
    `capture_columns`, the input data is first sanitized to ensure it contains 
    only numeric values. This preprocessing step is essential for accurate 
    statistical analysis.

    Parameters
    ----------
    data : ArrayLike or DataFrame
        The input data from which to calculate the standard deviation. Can be 
        a list, numpy array, or pandas DataFrame containing valid numeric values.
    columns : list of str, optional
        List of column names to consider for the calculation if `data` is a 
        DataFrame. If not provided, all numeric columns are considered.
    axis : optional, {index (0), columns (1)}
        Axis for the function to be applied on. Default is 0.
    ddof : int, default 1
        Delta Degrees of Freedom. The divisor used in the calculation is 
        N - ddof, where N represents the number of elements. By default, ddof 
        is set to 1, which computes the sample standard deviation. If ddof is 
        set to 0, the population standard deviation is calculated. The standard 
        deviation is the square root of variance:
            
        .. math:: 
            
           \text{Standard Deviation} = \sqrt{\frac{\sum (x_i - \bar{x})^2}{N - ddof}}
       
    as_frame : bool, default False
        Indicates whether to convert the input data to a DataFrame before 
        proceeding with the calculation. This parameter is relevant when 
        input data is not already a DataFrame.
    view : bool, default False
        If True, visualizes the data distribution and highlights the 
        standard deviation. Applicable only if `view` is True.
    cmap : str, default 'viridis'
        Colormap for the visualization. Applicable only if `view` is True.
    fig_size : Optional[Tuple[int, int]], default None
        Size of the figure for visualization. Applicable only if `view` is True.
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
    >>> from gofast.stats.utils import std
    >>> data_array = [1, 2, 3, 4, 5]
    >>> std(data_array)
    1.4142135623730951

    >>> data_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> std(data_df, columns=['A'], ddof=1)
    1.0

    >>> std(data_df, as_frame=True)
    A    1.0
    B    1.0
    dtype: float64

    >>> std(df, as_frame=True)
    A    1.0
    B    1.0
    dtype: float64

    >>> std(df, view=True, fig_size=(8, 4))
    Note
    ----
    The preprocessing steps, implied by the `make_data_dynamic` decorator 
    (not shown here but assumed to be applied outside this code snippet), 
    ensure that the input data is suitable for standard deviation calculation.
    These steps convert the data to numeric types and filter based on specified 
    columns, thus enhancing the function's utility and applicability across 
    diverse data forms.
    """
    # Handling DataFrame input with captured columns
    if isinstance(data, pd.DataFrame):
        axis = axis or 0
        # Pandas defaults ddof=1
        std_dev_result = data.std(ddof=ddof, axis=axis,  **kws)  
    else:
        # Convert ArrayLike to np.ndarray for consistent processing
        # In the case frame conversion failed, ensure consistency with pandas 
        data = np.array(data)
        std_dev_result = np.std(data, ddof=ddof, axis=axis **kws)  
    
    # Visualization of results if requested
    std_dev_result, view = ensure_visualization_compatibility(
        std_dev_result, as_frame, view, std )
    if view:
        _visualize_std_dev(data, std_dev_result, cmap=cmap, fig_size=fig_size, 
                           axis=axis )
    # Convert result to Series if as_frame=True
    if as_frame and not isinstance(std_dev_result, pd.DataFrame):
        if len(std_dev_result)==1: 
            std_dev_result = to_series_if(
                std_dev_result, value_names =['Standard Deviation']) 
        else:
            std_dev_result = pd.Series(
                np.array(std_dev_result), index=columns )

    return std_dev_result

def _visualize_std_dev(data, std_dev_result, cmap='viridis', fig_size=None,
                       axis=0 ):
    """
    Visualizes the distribution of the data and highlights the standard deviation.
    """
    plt.figure(figsize=fig_size if fig_size else (10, 6))
    _, data, cols= _prepare_plot_data(std_dev_result, data, axis=axis)
    if isinstance(data, pd.DataFrame):
        colors, alphas = get_colors_and_alphas( cols, cmap)
        for ii, col in enumerate(cols):
            plt.hist(data[col], bins='auto', alpha=alphas[ii], 
                     label=f'{col} Distribution', color=colors[ii])
            plt.axvline(x=data[col].mean(), color='red', linestyle='dashed', 
                        linewidth=2, label=f'{col} Mean')
            plt.axvline(x=data[col].mean() + data[col].std(), color='green',
                        linestyle='dashed', linewidth=2, label=f'{col} + Std Dev')
            plt.axvline(x=data[col].mean() - data[col].std(), color='green', 
                        linestyle='dashed', linewidth=2, label=f'{col} - Std Dev')
    else:
        plt.hist(data, bins='auto', alpha=0.7, label='Data Distribution',)
        mean = np.mean(data)
        std_dev = np.std(data)
        plt.axvline(x=mean, color='red', linestyle='dashed', linewidth=2, 
                    label='Mean')
        plt.axvline(x=mean + std_dev, color='green', linestyle='dashed', 
                    linewidth=2, label='+ Std Dev')
        plt.axvline(x=mean - std_dev, color='red',
                    linestyle='dashed', linewidth=2, label='- Std Dev', 
                    )
    plt.title('Data Distribution and Standard Deviation')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
        
@make_data_dynamic(
    capture_columns=True, 
    reset_index=True, 
    dynamize=False, 
    )   
def get_range(
    data: Union[ArrayLike, pd.DataFrame], 
    columns: Optional[List[str]] = None, 
    axis: Optional[int]=None, 
    as_frame: bool = False,
    view: bool = False, 
    cmap: str = 'viridis', 
    fig_size: Optional[Tuple[int, int]] = None, 
    **kws
    ):
    """
    Calculates the range of numeric data provided either as an array-like 
    structure or within a pandas DataFrame. 
    
    This function computes the difference between the maximum and minimum 
    values in the dataset, optionally focusing on specified columns when 
    dealing with DataFrames.

    Parameters
    ----------
    data : ArrayLike or DataFrame
        The input data from which to calculate the range. Can be a list, numpy 
        array, or pandas DataFrame containing valid numeric values.
    columns : list of str, optional
        List of column names to consider for the calculation if `data` is a 
        DataFrame. If not provided, all numeric columns are considered.
    axis : optional, {index (0), columns (1)}
         Axis for the function to be applied on. Default is 0.
    as_frame : bool, default False
        Indicates whether to convert the input data to a DataFrame before 
        proceeding with the calculation. Relevant when input data is not 
        already a DataFrame.
    view : bool, default=False
        If True, visualizes the statistical analysis results or the data 
        distribution.
    cmap : str, default='viridis'
        Colormap for the visualization. Only applicable if `view` is True.
    as_frame : bool, default=False
        If True, the result is returned as a pandas DataFrame or Series, depending
        on the dimensionality of the output. Otherwise, the result is returned in
        its native format (e.g., float, np.ndarray).
    fig_size : Optional[Tuple[int, int]], default=None
        Size of the figure for the visualization. Only applicable if `view` 
        is True.
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
    
    >>> data_array = [1, 2, 3, 4, 5]
    >>> print(get_range(data_array, view=True))

    >>> data_df = pd.DataFrame({'A': [2, 5, 8], 'B': [1, 4, 7]})
    >>> print(get_range(data_df, as_frame=True, view=True, fig_size=(8, 4)))

    Note
    ----
    The function aims to provide a straightforward way to compute the range 
    of a dataset, enhancing data analysis tasks. For DataFrame inputs, the 
    `as_frame` parameter allows users to maintain DataFrame structure in the 
    output, facilitating further analysis or integration with pandas-based 
    workflows.
    """

    data_selected = data.copy() if isinstance(data, pd.DataFrame) else data 
    
    # Compute the range for DataFrame or ArrayLike
    if isinstance(data_selected, pd.DataFrame):
        axis = axis or 0 
        range_values = data_selected.max(axis=axis, **kws) - data_selected.min(
            axis=axis, **kws)
    else:
        range_values = np.max(data_selected, axis=axis,  **kws) - np.min(
            data_selected, axis=axis, **kws)

    range_values, view  = ensure_visualization_compatibility(
        range_values, as_frame, view, get_range )
    
    # Visualization
    if view:
        _visualize_range(data_selected, range_values, columns=columns,
                         cmap=cmap, fig_size=fig_size, axis= axis )
        
    # For DataFrame output
    if as_frame and not isinstance (range_values, pd.DataFrame): 
        if len(range_values) ==1: 
            range_values = to_series_if(
                range_values, 
                value_names=update_series_index (
                    pd.Series (range_values) ,
                    new_indexes =["Range Values"], 
                    allow_replace=True,  
                    condition=lambda s: pd.api.types.is_integer_dtype(s.index.dtype))
                )
        else: 
            range_values = pd.Series(
                range_values, index=columns)
            
    return range_values

def _visualize_range(data, range_values, columns=None, 
                     cmap='viridis', fig_size=None, axis =0):
    """
    Visualizes the data distribution and highlights the range values.
    """
    plt.figure(figsize=fig_size if fig_size else (10, 6))
    _, data, cols = _prepare_plot_data(range_values, data, axis=axis )
    if isinstance(data, pd.DataFrame):
        colors, alphas = get_colors_and_alphas( cols, cmap)
        for ii, col in enumerate(cols):
            values = data[col]
            plt.hist(values, bins=30, alpha=alphas[ii], 
                     label=f'{col} Distribution',
                     color=colors[ii])
            min_val, max_val = values.min(), values.max()
            plt.axvline(x=min_val, color='red', linestyle='dashed', 
                        linewidth=1, label=f'{col} Min')
            plt.axvline(x=max_val, color='green', linestyle='dashed',
                        linewidth=1, label=f'{col} Max')
    else:
        plt.hist(data, bins=30, alpha=0.5, label='Data Distribution',)
        min_val, max_val = np.min(data), np.max(data)
        plt.axvline(x=min_val, color='red', linestyle='dashed', 
                    linewidth=1, label='Min')
        plt.axvline(x=max_val, color='green', linestyle='dashed',
                    linewidth=1, label='Max')
    plt.title('Data Distribution and Range')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


@make_data_dynamic(capture_columns=True)  
def quartiles(
    data: Union[ArrayLike, pd.DataFrame], 
    columns: Optional[List[str]] = None, 
    axis: int =0, 
    as_frame: bool = True,
    view: bool = False, 
    plot_type: str = 'box',  
    cmap: str = 'viridis', 
    fig_size: Optional[Tuple[int, int]] = None, 
    **kws):
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
    axis : optional, {index (0), columns (1)}
         Axis for the function to be applied on. Default is 0.
    as_frame : bool, default False
        Indicates whether to convert the input data to a DataFrame before 
        proceeding with the calculation. Relevant when input data is not 
        already a DataFrame.
    view : bool, default False
        If True, visualizes the quartiles of the dataset along with 
        its distribution.
    plot_type : str, default 'box'
        Type of plot for visualization. Options are 'box' for box plots and 
        'hist' for histograms. Only applicable if `view` is True.
    cmap : str, default 'viridis'
        Colormap for the visualization.
    fig_size : Optional[Tuple[int, int]], default None
        Size of the figure for the visualization.
        
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
    
    >>> data_df = pd.DataFrame({'A': [2, 5, 7, 8], 'B': [1, 4, 6, 9]})
    >>> print(quartiles(data_df, columns=['A'], as_frame=True))
          A
    25%  4.25
    50%  6.00
    75%  7.75

    >>> quartiles(data_df, view=True, fig_size=(8, 4))
    >>> data_df = pd.DataFrame({'A': [2, 5, 7, 8], 'B': [1, 4, 6, 9]})
    >>> quartiles(data_df, view=True, plot_type='box', fig_size=(8, 4))
    >>> quartiles(data_df, view=True, plot_type='hist', fig_size=(8, 4))
    Note
    ----
    Quartiles are a fundamental statistical measure used to understand the spread 
    and center of a dataset. By offering the flexibility to compute quartiles for 
    specific columns or entire datasets, this function aids in comprehensive data 
    analysis, particularly in exploratory data analysis (EDA) and data visualization.
    """

    if isinstance(data, pd.DataFrame):
        data_selected = data.copy() 
        quartiles_result = data_selected.quantile(
            [0.25, 0.5, 0.75], axis = axis,  **kws)
    else:
        data_selected = np.asarray(data)
        quartiles_result = np.percentile(
            data_selected, [25, 50, 75], axis =axis, **kws)

    quartiles_result, view  = ensure_visualization_compatibility(
        quartiles_result, as_frame, view , func_name=quartiles )
    if view:
        _visualize_quartiles(quartiles_result, data, plot_type=plot_type,
                             cmap=cmap, fig_size=fig_size, axis=axis )
    if as_frame: 
        quartiles_result=update_index(
            quartiles_result, 
            new_indexes = ['25%', '50%', '75%'], 
            return_data= True, 
            allow_replace=True
            ).T 
    else: 
        quartiles_result = np.squeeze (np.asarray (quartiles_result)).T

    return quartiles_result 

def _visualize_quartiles(
        quartiles_result, data, plot_type, cmap, fig_size, axis=0):
    """
    Visualizes quartiles using the specified plot type.
    """
    plt.figure(figsize=fig_size if fig_size else (10, 6))
    quartiles_result, data, cols = _prepare_plot_data(
        quartiles_result, data , axis=axis )

    colors, alphas = get_colors_and_alphas(
        cols, cmap, convert_to_named_color=True)
    if plot_type == 'box':
        if isinstance(quartiles_result, pd.DataFrame):
            quartiles_result.boxplot(grid=True, color=colors[0])
        else:
            plt.boxplot(quartiles_result, patch_artist=True, 
                        boxprops=dict(facecolor=colors[0]))
    elif plot_type == 'hist':
        if isinstance(quartiles_result, pd.DataFrame):
            for ii, col in enumerate (cols):
                plt.hist(quartiles_result[col], bins=30, alpha=alphas[ii],
                         label=f'{col} Distribution', color=colors [ii])
        else:
            plt.hist(quartiles_result, bins=30, alpha=0.5, color=cmap)
            
    plt.title('Data Distribution')
    plt.ylabel('Frequency' if plot_type == 'hist' else 'Value')
    plt.legend() if plot_type == 'hist' and isinstance(
        data, pd.DataFrame) else None
    plt.show()

@DynamicMethod (capture_columns=True)
def quantile(
    data: Union[ArrayLike, DataFrame], 
    q: Union[float, List[float]], 
    columns: Optional[List[str]] = None, 
    axis: Optional [int] = None, 
    as_frame: bool = True, 
    view: bool = False, 
    plot_type: Optional[str] = 'box', 
    cmap: str = 'viridis', 
    fig_size: Optional[Tuple[int, int]] = None, 
    **kws):
    """
    Computes specified quantiles of numeric data provided either as an array-like 
    structure or within a pandas DataFrame. 
    
    Function is designed to offer flexibility in calculating quantiles across 
    entire datasets or within specified columns of a DataFrame.

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
    axis : optional, {index (0), columns (1)}
         Axis for the function to be applied on. Default is 0.
    as_frame : bool, default=True
        Indicates whether to convert the input data to a DataFrame before 
        proceeding with the calculation. Relevant when input data is not 
        already a DataFrame.
    view : bool, default False
        If True, visualizes the quartiles of the dataset along with 
        its distribution.
    plot_type : str, default 'box'
        Type of plot for visualization. Options are 'box' for box plots and 
        'hist' for histograms. Only applicable if `view` is True.
    cmap : str, default 'viridis'
        Colormap for the visualization.
        
    fig_size : Optional[Tuple[int, int]], default None
        Size of the figure for the visualization.
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
    >>> import numpy as np
    >>> from gofast.stats.utils import quantile
    >>> data_array = [1, 2, 3, 4, 5]
    >>> quantile(data_array, q=0.5)
    3.0

    >>> data_df = pd.DataFrame({'A': [2, 5, 7, 8], 'B': [1, 4, 6, 9]})
    >>> quantile(data_df, q=[0.25, 0.75], columns=['A'])
    array([3.75, 7.25])

    >>> quantile(data_df, q=0.5, as_frame=True)
       50%
    A  6.0
    B  5.0
    >>> quantile(data_array, q=0.5, view=True, plot_type='hist')

    >>> data_df = pd.DataFrame({'A': [2, 5, 7, 8], 'B': [1, 4, 6, 9]})
    >>> quantile(data_df, q=[0.25, 0.75], as_frame=True, view=True, plot_type='box')
    
    Note
    ----
    The function provides a convenient way to compute quantiles, a critical 
    statistical measure for understanding the distribution of data. The 
    flexibility to compute quantiles for specific columns or entire datasets 
    enhances its utility in exploratory data analysis and data preprocessing.
    """
    axis = axis or 0
    data_selected = data.copy() 
    # Compute the quantile or quantiles
    quantiles_result = data_selected.quantile(q, axis=axis, **kws)
  
    if view: # visualize box/hist quantiles
        _visualize_quantiles(
            data, q, quantiles_result,
            plot_type=plot_type, cmap=cmap, 
            fig_size=fig_size, axis= axis
        )
    # update data frame indexes with string quantiles
    new_indexes= [f'{int(q*100)}%' for q in np.atleast_1d(q)]
    # print(new_indexes)
    quantiles_result=update_index( 
        quantiles_result, 
        new_indexes = new_indexes, 
        return_data= True, 
        allow_replace=True
        ).T 
    # convert to series if applicable 
    quantiles_result = convert_and_format_data(
        quantiles_result,
        as_frame, 
        allow_series_conversion=False if as_frame else True,
        series_name=new_indexes[0], 
        condense=True, 
        condition =lambda x:  {
            "force_array_output": True}  if not as_frame else {} 
        )
    return quantiles_result

def _visualize_quantiles(
    data, q, 
    quantiles_result, 
    plot_type, 
    cmap, 
    fig_size,
    axis=0 
    ):
    """
    Visualizes the data and highlights the quantiles, based on the 
    specified plot type.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        The dataset from which quantiles were computed.
    q : float or list of float
        The quantile(s) that were computed.
    quantiles_result : pd.DataFrame or np.ndarray
        The computed quantiles.
    plot_type : str
        The type of plot to generate ('box' or 'hist').
    cmap : str
        The colormap name to use for coloring the plot.
    fig_size : tuple of int
        The size of the figure to create.
    axis : int, default 0
        The axis along which the quantiles were computed.

    Examples
    --------
    >>> data = pd.DataFrame({'A': np.random.normal(
        size=100), 'B': np.random.normal(size=100)})
    >>> q = [0.25, 0.5, 0.75]
    >>> quantiles_result = data.quantile(q)
    >>> _visualize_quantiles(data, q, quantiles_result, 'box', 'viridis', (10, 6))
    """
    plt.figure(figsize=fig_size if fig_size else (4, 4))
    
    q_list = np.atleast_1d(q)
    _, data, cols = _prepare_plot_data(
        quantiles_result, data , axis=axis )

    colors, alphas = get_colors_and_alphas(
        cols, cmap, convert_to_named_color=True)
    
    if plot_type == 'box':
        try: 
            data.boxplot( grid=True, color=colors[0])
        except: 
            plt.boxplot(data, patch_artist=True, 
                        boxprops=dict(facecolor=colors[0]))
        plt.title('Boxplot and Quantiles')
        
    elif plot_type == 'hist':
        try: 
            plt.hist(data, bins=30, alpha=0.5, color=colors[: len(cols)])
            for quantile in q_list:
                plt.axvline(np.quantile(data, quantile), color='r',
                            linestyle='--', label=f'{quantile*100}% Quantile')
        except: 
            for ii, col in enumerate (cols):
                plt.hist(data[col], 
                          bins=30,
                          alpha=alphas[ii],
                          label=f'{col} Distribution',
                          color=colors[ii])
                for quantile in q_list:
                    plt.axvline(data[col].quantile(quantile),
                                color='r', linestyle='--',
                                label=f'{quantile*100}% Quantile')
          
        plt.title('Histogram and Quantiles')
        
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    # Only add legend if labels are present
    if plt.gca().get_legend_handles_labels()[0]:
        plt.legend()
    plt.show()

@DynamicMethod ( expected_type="both", capture_columns=True)
def corr(
    data: Union[ArrayLike, DataFrame], /,  
    columns: List[str] = None,
    method: str='pearson', 
    view: bool = False, 
    cmap: str = 'viridis', 
    fig_size: Optional[Tuple[int, int]] = None, 
    **kws
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
    >>> from gofast.stats.utils import corr
    >>> data = pd.DataFrame({
    ...     'A': [1, 2, 3, 4],
    ...     'B': [4, 3, 2, 1],
    ...     'C': [1, 2, 3, 4]
    ... })
    >>> corr(data)
           A         B    C
    A  1.000000 -1.000000  1.000000
    B -1.000000  1.000000 -1.000000
    C  1.000000 -1.000000  1.000000

    >>> corr2(data, columns=['A', 'C'])
           A    C
    A  1.000000  1.000000
    C  1.000000  1.000000
    >>> >>> corr(data, view=True)

    >>> corr(data, columns=['A', 'C'], view=True, plot_type='heatmap',
    ...        cmap='coolwarm')
    Note
    ----
    The function utilizes pandas' `corr()` method to compute the correlation 
    matrix, offering flexibility through `**kws` to use different correlation 
    computation methods. For non-DataFrame inputs, the data is first converted 
    to a DataFrame, ensuring uniform processing.
    """
    correlation_matrix = data.corr(method= method, **kws)
    if view:
        plt.figure(figsize=fig_size)
        sns.heatmap(correlation_matrix, annot=True, cmap=cmap, fmt=".2f")
        plt.title("Correlation Matrix")
        plt.show()
    
    return correlation_matrix

def correlation(
   x: Optional[Union[str, Array1D, Series]] = None,  
   y: Optional[Union[str, Array1D, Series]] = None, 
   method: str = 'pearson', 
   data: Optional[DataFrame] = None,
   view: bool = False, 
   plot_type: Optional[str] = None, 
   cmap: str = 'viridis', 
   fig_size: Optional[Tuple[int, int]] = None, 
   **kws):
    """
    Computes and optionally visualizes the correlation between two datasets 
    or within a DataFrame. If both `x` and `y` are provided, calculates the 
    pairwise correlation. If only `x` is provided as a DataFrame and `y` is None,
    computes the correlation matrix for `x`.

    Parameters
    ----------
    x : Optional[Union[str, Array1D, Series]], default None
        The first dataset for correlation analysis or the entire dataset 
        if `y` is None. Can be an array-like object or a column name in `data`.
    y : Optional[Union[str, Array1D, Series]], default None
        The second dataset for correlation analysis. Can be an array-like object
        or a column name in `data`. If omitted, and `x` is a DataFrame, calculates
        the correlation matrix of `x`.
    method : str, default 'pearson'
        The method of correlation ('pearson', 'kendall', 'spearman') or a callable
        with the signature (np.ndarray, np.ndarray) -> float.
    data : Optional[DataFrame], default None
        A DataFrame containing `x` and/or `y` columns if they are specified as 
        column names.
    view : bool, default False
        If True, visualizes the correlation using a scatter plot (for pairwise 
        correlation) or a heatmap (for correlation matrices).
    plot_type : Optional[str], default None
        Type of plot for visualization when `view` is True. Options are 'scatter'
        for pairwise correlation or None for no visualization.
    cmap : str, default 'viridis'
        Colormap for the visualization plot.
    fig_size : Optional[Tuple[int, int]], default None
        Size of the figure for visualization.
    **kws : dict
        Additional keyword arguments for the correlation computation method.

    Returns
    -------
    correlation_value : float or pd.DataFrame
        The correlation coefficient if `x` and `y` are provided, or the correlation
        matrix if `x` is a DataFrame and `y` is None.

    Examples
    --------
    >>> from gofast.stats.utils import correlation
    >>> import pandas as pd
    >>> data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> corr('A', 'B', data=data)
    1.0

    >>> x = [1, 2, 3]
    >>> y = [4, 5, 6]
    >>> corr(x, y)
    1.0

    >>> correlation(data)
    DataFrame showing correlation matrix of 'data'
    
    >>> x = [1, 2, 3, 4, 5]
    >>> y = [5, 4, 3, 2, 1]
    print("Correlation coefficient:", correlation(x, y, view=True))

    >>> # Correlation matrix of a DataFrame
    >>> data = pd.DataFrame({'A': np.random.rand(10), 'B': np.random.rand(10), 
                             'C': np.random.rand(10)})
    print("Correlation matrix:\n", correlation(data))
    correlation(data, view=True)
    
    >>> data = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [4, 3, 2, 1]})
    >>> correlation('A', 'B', data=data, view=True, plot_type='scatter')
    -1.0

    Correlation matrix of a DataFrame:
    >>> correlation(data=data, view=True)
    Outputs a heatmap of the correlation matrix.

    Pairwise correlation between two array-like datasets:
    >>> x = [1, 2, 3, 4]
    >>> y = [4, 3, 2, 1]
    >>> correlation(x, y, view=True, plot_type='scatter')
    -1.0

    Compute pairwise correlation:
    >>> x = [1, 2, 3]
    >>> y = [4, 5, 6]
    >>> correlation(x, y)
    1.0

    Compute and visualize the correlation matrix of a DataFrame:
    >>> data = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [4, 3, 2, 1]})
    >>> correlation(data=data, view=True)

    Compute pairwise correlation with column names from a DataFrame:
    >>> correlation('A', 'B', data=data, view=True, plot_type='scatter')
    -1.0
    Note
    ----
    The function is designed to provide a versatile interface for correlation 
    analysis, accommodating different types of input and supporting various 
    correlation methods through keyword arguments. It utilizes pandas' `corr()` 
    method for DataFrame inputs, enabling comprehensive correlation analysis within 
    tabular data.
    
    """
    if x is None and y is None:
        if data is not None:
            cor_matrix= data.corr(method=method, **kws)
            if view:
                plt.figure(figsize=fig_size if fig_size else (8, 6))
                sns.heatmap(cor_matrix, annot=True, cmap=cmap)
                plt.title("Correlation Matrix")
                plt.show()
            return cor_matrix
        else:
            raise ValueError("At least one of 'x', 'y', or 'data' must be provided.")
            
    # Validate inputs
    if data is None and (isinstance(x, str) or isinstance(y, str)):
        raise ValueError("Data must be provided when 'x' or 'y' are column names.")

    # Extract series from data if x or y are column names
    x_series = data[x] if isinstance(x, str) and data is not None else x
    y_series = data[y] if isinstance(y, str) and data is not None else y

    # If x is a DataFrame and y is None, compute the correlation matrix
    if isinstance(x_series, pd.DataFrame) and y_series is None:
        correlation_matrix = x_series.corr(method=method, **kws)
    # If x and y are defined, compute pairwise correlation
    elif x_series is not None and y_series is not None:
        x_series = pd.Series(x_series) if not isinstance(x_series, pd.Series) else x_series
        y_series = pd.Series(y_series) if not isinstance(y_series, pd.Series) else y_series
        correlation_value = x_series.corr(y_series, method=method, **kws)
    else:
        raise ValueError("Invalid input: 'x' and/or 'y' must be provided.")

    # Visualization
    if view:
        plt.figure(figsize=fig_size if fig_size else (6, 6))
        if 'correlation_matrix' in locals():
            sns.heatmap(correlation_matrix, annot=True, cmap=cmap)
            plt.title("Correlation Matrix")
        elif plot_type == 'scatter':
            colors, _= get_colors_and_alphas(len(x_series), cmap=cmap  )
            plt.scatter(x_series, y_series, color=colors )
            plt.title(f"Scatter Plot: Correlation = {correlation_value:.2f}")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.grid(True)
        plt.show()

    return correlation_matrix if 'correlation_matrix' in locals() else correlation_value

@DynamicMethod ( 
    expected_type="both", 
    capture_columns=True, 
    drop_na=True
   )
def iqr(
    data: Union[ArrayLike, pd.DataFrame], 
    columns: Optional[List[str]] = None,
    axis: Optional[int] =None, 
    as_frame: bool = True,
    view: bool = False, 
    plot_type: Optional[str] = 'boxplot', 
    cmap: str = 'viridis', 
    fig_size: Optional[Tuple[int, int]] = (4, 4), 
    orient: Optional [str] =None, 
    **kws):
    """
    Computes the interquartile range (IQR) for numeric data and optionally 
    visualizes it.

    Parameters
    ----------
    data: Union[ArrayLike, pd.DataFrame]
        The dataset from which the IQR is calculated. This can be directly 
        provided as a list, numpy array, or a pandas DataFrame. When provided 
        as a list or numpy array, the data is internally converted to a pandas 
        DataFrame for calculation.
    
    columns : Optional[List[str]] (default=None)
        Relevant when `data` is a pandas DataFrame. Specifies the columns within 
        the DataFrame for which the IQR is to be calculated. If None, the IQR 
        is calculated for all numeric columns in the DataFrame.
        
    axis: optional, {index (0), columns (1)}
         Axis for the function to be applied on. Default is 0.
         
    as_frame : bool (default=False)
        Determines the format of the return value. When True, the function returns 
        a pandas Series object representing the IQR for each column specified or 
        for all numeric columns by default. When False, the function returns a 
        single float value representing the IQR of the data or the first column 
        if `data` is a DataFrame.

    view : bool (default=False)
        If True, the function generates a plot visualizing the distribution of 
        the data along with the IQR. The type of plot generated is determined 
        by the `plot_type` parameter.

    plot_type : Optional[str] (default='boxplot')
        Specifies the type of plot to use for visualization when `view` is True. 
        Currently supports 'boxplot', which displays a box and whisker plot 
        showing the quartiles of the dataset along with outliers. Additional 
        plot types can be supported by extending the function.

    cmap : str (default='viridis')
        The colormap to use for the visualization when `view` is True. This parameter 
        is passed directly to the plotting function to customize the color scheme 
        of the plot.

    fig_size : Optional[Tuple[int, int]] (default=(8, 6))
        Specifies the size of the figure for the visualization plot when `view` is 
        True. Provided as a tuple (width, height) in inches.

    orient: Optional [str], optional 
       Orientation of box plot either vertical ( default) or horizontal 'h'. 
       
    **kws : dict
        Additional keyword arguments passed directly to the numpy percentile 
        function when calculating the 25th and 75th percentiles. This allows 
        for customization of the percentile calculation, such as specifying 
        the interpolation method.

    Returns
    -------
    iqr_value : float or pd.Series
        The calculated IQR of the provided data. Returns a single float if 
        the data is one-dimensional or if `as_frame` is False. Returns a pandas 
        Series if the input is a DataFrame and `as_frame` is True.

    Examples
    --------
    >>> import numpy as np 
    >>> import pandas as pd 
    >>> from gofast.stats.utils import iqr
    >>> data_array = [1, 2, 3, 4, 5]
    >>> iqr(data_array)
    2.0

    >>> data_df = pd.DataFrame({'A': [2, 5, 7, 8], 'B': [1, 4, 6, 9]})
    >>> iqr(data_df, columns=['A'])
    3.5

    >>> iqr(data_df, as_frame=True)
    A    3.5
    B    4.0
    dtype: float64
    >>> data_array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> print("IQR for array:", iqr(data_array, view=True, plot_type='boxplot'))

    >>> data_df = pd.DataFrame({
        'A': np.random.normal(0, 1, 100),
        'B': np.random.normal(1, 2, 100),
        'C': np.random.normal(2, 3, 100)
    })
    >>> print("IQR for DataFrame:", iqr(data_df, as_frame=True))
    >>> iqr(data_df, view=True, plot_type='boxplot')
    
    Note
    ----
    The IQR is a robust measure of spread that is less influenced by outliers 
    than the range. This function simplifies the process of calculating the IQR, 
    especially useful in exploratory data analysis and for identifying potential 
    outliers.
    """
    # Calculate IQR
    Q1 = data.quantile(0.25, axis = axis or 0 , **kws)
    Q3 = data.quantile(0.75, axis = axis or 0 , **kws)
    iqr_values = Q3 - Q1
 
    # Visualization
    if view:
        _, data, cols = _prepare_plot_data(
            iqr_values, data, axis=axis )
        plt.figure(figsize=fig_size)
        if isinstance(data, pd.DataFrame):
            sns.boxplot(data=data, orient=orient, palette=cmap)
        else:
            sns.boxplot(data=pd.DataFrame(data), orient=orient, palette=cmap)
        plt.title('IQR Visualization with Boxplot')
        plt.show()
        
    iqr_values = convert_and_format_data(
        iqr_values, as_frame, 
        force_array_output= True if not as_frame else False, 
        condense= True, 
        condition= series_naming("IQR")
        )
    return iqr_values

@make_data_dynamic(capture_columns=True)
def z_scores(
    data: Union[ArrayLike, DataFrame], 
    columns: Optional[List[str]] = None,
    as_frame: bool = True,
    view: bool = False, 
    plot_type: Optional[str] = 'hist', 
    cmap: str = 'viridis', 
    orient: Optional [str]=None, 
    fig_size: Optional[Tuple[int, int]] = (8, 6), 
    **kws):
    """
    Computes the Z-scores for each data point in a dataset. 
    
    Optionally visualizes the distribution.
    
    Z-scores, or standard scores, indicate how many standard deviations an 
    element is from the mean. This function can standardize a single array of
    values or specific columns within a pandas DataFrame.
    
    Parameters
    ----------
    data : Union[ArrayLike, pd.DataFrame]
        The dataset for which to calculate Z-scores. Can be a list, 
        numpy array, or pandas DataFrame.
        
    columns : Optional[List[str]] (default=None)
        Specifies the columns within a DataFrame for which to calculate Z-scores. 
        Applicable only when `data` is a DataFrame. If None, Z-scores for 
        all columns are calculated.
        
    as_frame : bool (default=True)
        When True, returns the result as a pandas DataFrame or Series 
        (depending on the input). Useful for maintaining DataFrame structure
        in the output.
        
    view : bool (default=False)
        If True, generates a visualization of the Z-scores distribution. 
        The type of visualization is determined by the `plot_type` parameter.
        
    plot_type : Optional[str] (default='hist')
        Specifies the type of plot for visualization:
        - 'hist': Histogram of the Z-scores distribution.
        - 'boxplot': Box plot showing the quartiles and outliers.
        - 'density': Density plot for a smooth distribution curve.
    
    cmap : str (default='viridis')
        The colormap for the visualization plot. Applicable only if `view` is True.
        
    orient: str, {None,'h'}
       Orientation of the boxplot. Use 'h' for horizontal orientation otherwise 
       the default is vertical (``None``). 
  
    fig_size : Optional[Tuple[int, int]] (default=(8, 6))
        Size of the figure for the visualization plot. Applicable only if `view` is True.
        
    **kws : dict
        Additional keyword arguments for customization, not directly used
        in Z-score calculation.

    Returns
    -------
    Union[np.ndarray, pd.DataFrame]
        The Z-scores of the provided data. Returns a numpy array if the data 
        is one-dimensional or if `as_frame` is False. Returns a pandas DataFrame
        if the input is a DataFrame and `as_frame` is True.

    Examples
    --------
    >>> import numpy as np 
    >>> import pandas as pd 
    >>> from gofast.stats.utils import z_scores
    
    Calculating Z-scores for an array:
    >>> data_array = [1, 2, 3, 4, 5]
    >>> print(z_scores(data_array))
    [-1.41421356, -0.70710678, 0., 0.70710678, 1.41421356]
    
    Calculating and visualizing Z-scores for DataFrame columns:
    >>> data_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> z_scores(data_df, as_frame=True, view=True, plot_type='boxt')
    
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
    The Z-score is a powerful tool for data standardization, making it easier
    to compare measurements across different scales. Visualization options 
    provide quick insights into the distribution and variance of standardized data.
    """

    mean = data.mean()
    std_dev = data.std()
    result = (data - mean) / std_dev 

    # Visualization
    if view:
        plot_type= _validate_plot_type(
            plot_type, target_strs= ['hist', 'box', 'density'],
            raise_exception= True)
        colors,alphas = get_colors_and_alphas(result.columns)

        if plot_type == 'hist':
            if isinstance(result, pd.DataFrame):
                result.plot(kind='hist', alpha=0.5, bins=20, colormap=cmap,
                            legend=True)
            else:
                plt.hist(result, bins=20, alpha=0.5, color=cmap)
            plt.title('Z-Scores Distribution')
            plt.xlabel('Z-Score')
            plt.ylabel('Frequency')
        elif plot_type == 'box':
            if isinstance(result, pd.DataFrame):
                sns.boxplot(data=result, orient=orient, palette=cmap)
            else:
                sns.boxplot(
                    data=pd.DataFrame(result), orient=orient, color=cmap)
            plt.title('Z-Scores Box Plot')
        elif plot_type == 'density':
            if isinstance(result, pd.DataFrame):
                result.plot(kind='density', colormap=cmap, legend=True)
            else:
                sns.kdeplot(result, color=cmap)
            plt.title('Z-Scores Density Plot')
            plt.xlabel('Z-Score')

        plt.show()
    result = convert_and_format_data(
        result, as_frame, 
        force_array_output= True if not as_frame else False, 
        condense= True, 
        condition=series_naming("z_scores")
        )
    
    return result

@make_data_dynamic(capture_columns=True)
def describe(
    data: DataFrame,
    columns: Optional[List[str]] = None,
    include: Union[str, List[str]] = 'all',
    exclude: Union[str, List[str]] = None,
    as_frame: bool = True,
    view: bool = False, 
    orient: Optional[str]=None,
    plot_type: Optional[str] = 'box',   
    cmap: str = 'viridis', 
    fig_size: Optional[Tuple[int, int]] = (10, 6), 
    **kwargs
):
    """
    Generates a comprehensive summary of descriptive statistics for a 
    given dataset.
    
    This function provides an easy way to get an overview of basic descriptive 
    statistics including mean, median, mode, min, max, etc., for numerical data 
    in a DataFrame or an array-like structure. It's particularly useful for 
    preliminary data analysis and understanding the distribution of data points.
    
    Parameters
    ----------
    data : ArrayLike or pd.DataFrame
        The input data for which descriptive statistics are to be calculated.
        Can be a Pandas DataFrame or any array-like structure containing 
        numerical data.
    columns : List[str], optional
        Specific columns to include in the analysis if `data` is a DataFrame.
        If None, all columns are included. Default is None.
    include : 'all', list-like of dtypes or None (default='all'), optional
        A white list of data types to include in the result. 
        Ignored for ArrayLike input.
        - 'all' : All columns of the input will be included in the output.
        - A list-like of dtypes : Limits the data to the provided data types.
          For example, [np.number, 'datetime'] will include only numeric and 
          datetime data.
    exclude : list-like of dtypes or None (default=None), optional
        A black list of data types to exclude from the result. 
        Ignored for ArrayLike input.
    as_frame : bool, default=True
        If True, the input array-like structure is converted to a Pandas DataFrame
        before calculating the descriptive statistics. This is useful if the input
        is a structured array or a sequence of sequences and you want the output
        in the form of a DataFrame.
    view : bool, optional
        If True, generates a visualization of the descriptive statistics 
        based on `plot_type`.  Default is False.
    orient: str, {None,'h'}
       Orientation of the boxplot. Use 'h' for horizontal orientation otherwise 
       the default is vertical (``None``). 
       
    plot_type : Optional[str], optional
        Specifies the type of plot for visualization: 'box', 'hist', or 
        'density'. Ignored if `view` is False. Default is 'box'.
    cmap : str, optional
        The colormap used for plotting. Default is 'viridis'.
    fig_size : Optional[Tuple[int, int]], optional
        The size of the figure for the plot. Default is (10, 6).
    **kwargs : dict
        Additional keyword arguments to be passed to the
        `pd.DataFrame.describe` method.
    
    Returns
    -------
    pd.DataFrame or pd.Series
        The descriptive statistics of the data. Returns a DataFrame if `as_frame` is True
        or if `data` is a DataFrame; otherwise, returns a Series.
    
    Examples
    --------
    
    >>> import numpy as np
    >>> import pandas as pd
    >>> from gofast.stats.utils import describe
    >>> data = np.random.rand(100, 4)
    >>> describe(data, as_frame=True)
    
    >>> import pandas as pd
    >>> df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D'])
    >>> describe(df, columns=['A', 'B'])
    
    >>> df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D'])
    >>> describe(df, columns=['A', 'B'], view=True, plot_type='hist')
    Note
    ----
    This function is a convenient wrapper around `pd.DataFrame.describe`,
    allowing for additional flexibility and ease of use. It's designed to
    work seamlessly with both array-like structures and Pandas DataFrames,
    making it a versatile tool for initial data exploration.
    """
    # Convert array-like input to DataFrame if necessary
    stats_result = data.describe(
        include=include, exclude=exclude, **kwargs)
    
    # Visualization
    if view:
        plot_type= _validate_plot_type(
            plot_type, target_strs= ['hist', 'box', 'density'],
            raise_exception= True)
        colors, alphas = get_colors_and_alphas(
            data.columns, cmap, convert_to_named_color= True )
        if plot_type == 'box':
            sns.boxplot(data=data, orient=orient, palette=cmap)
            plt.title('Box Plot of Descriptive Statistics')
        elif plot_type == 'hist':
            data.plot(kind='hist', bins=30, alpha=0.6, colormap=cmap)
            plt.title('Histogram of Descriptive Statistics')
        elif plot_type == 'density':
            data.plot(kind='density', colormap=cmap)
            plt.title('Density Plot of Descriptive Statistics')
        else:
            raise ValueError(
                f"Unsupported plot_type: {plot_type}."
                "Choose from 'box', 'hist', or 'density'.")
        plt.show()
        
    stats_result= convert_and_format_data(
        stats_result, as_frame, 
        force_array_output= True if not as_frame else False, 
        condense=True,
        condition=series_naming ("descriptive_stats"), 
        )
    return stats_result

@make_data_dynamic(capture_columns=True)
def skew(
    data: Union[ArrayLike, pd.DataFrame],
    columns: Optional[List[str]] = None,
    axis: int =0, 
    as_frame: bool = False,
    view: bool = False, 
    plot_type: Optional[str] = 'density', 
    cmap: str = 'viridis', 
    fig_size: Optional[Tuple[int, int]] = (10, 6), 
    **kwargs
):
    """
    Calculates the skewness of the data distribution, providing insight into
    the asymmetry of the distribution around its mean.

    Skewness is a measure of the symmetry, or lack thereof, of a distribution.
    A skewness value > 0 indicates a distribution that is skewed to the right,
    a value < 0 indicates a distribution that is skewed to the left, and a
    value of 0 indicates a symmetrical distribution.

    Parameters
    ----------
    data : ArrayLike or pd.DataFrame
        The input data for which skewness is to be calculated. Can be a Pandas 
        DataFrame or any array-like structure containing numerical data.
        
    columns : List[str], optional
        Specific columns to include in the analysis if `data` is a DataFrame.
        If None, all columns are included. Default is None.
        
    axis: optional, {index (0), columns (1)}
         Axis for the function to be applied on. Default is 0. 
         
    as_frame : bool, default=False
        If True and `data` is ArrayLike, converts `data` to a Pandas DataFrame
        before calculating skewness. Useful for structured array-like inputs.
        
    view : bool, optional
        If True, visualizes the data distribution and its skewness using the
        specified `plot_type`. Defaults to False.
        
    plot_type : Optional[str], optional
        Type of plot for visualizing the data distribution. Supported values
        are 'density' for a density plot and 'hist' for a histogram. Ignored
        if `view` is False. Defaults to 'density'.
        
    cmap : str, optional
        Colormap for the plot. Defaults to 'viridis'.
    fig_size : Optional[Tuple[int, int]], optional
        Size of the figure for the plot. Defaults to (10, 6).
    **kwargs : dict
        Additional keyword arguments passed to `pd.Series.skew`
        or `pd.DataFrame.skew`.


    Returns
    -------
    float or pd.Series
        The skewness value(s) of the data. Returns a single float value if `data`
        is a single column or ArrayLike without `as_frame` set to True. Returns
        a pd.Series with skewness values for each column if `data` is a DataFrame
        or `as_frame` is True.

    Examples
    --------
    >>> from gofast.stats.utils import skew
    >>> import numpy as np
    >>> data = np.random.normal(loc=0, scale=1, size=1000)
    >>> calculate_skewness(data)
    0.1205147559272991
    
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'A': np.random.normal(loc=0, scale=1, size=1000),
    ...     'B': np.random.normal(loc=1, scale=2, size=1000),
    ...     'C': np.random.lognormal(mean=0, sigma=1, size=1000)
    ... })
    >>> skew(df)
    array([-0.07039989, -0.04687687,  7.20119035])
    
    >>> data = [1, 2, 2, 3, 4, 7, 9]
    >>> skew(data)
    0.9876939667076702

    >>> df = pd.DataFrame({
    ...     'normal': np.random.normal(0, 1, 1000),
    ...     'right_skewed': np.random.exponential(1, 1000)
    ... })
    >>> skew(df, view=True, plot_type='density')

    Note
    ----
    This function is useful for identifying the symmetry of data distributions,
    which can be critical for certain statistical analyses and modeling. Skewness
    can indicate the need for data transformation or the use of non-parametric
    statistical methods.
    """
    # Ensuring numeric data type for calculation # for consisteny 
    data_numeric = data.apply(pd.to_numeric, errors='coerce')
    skewness_value = data_numeric.skew(axis=0, **kwargs) 

    if view:
        colors, alphas = get_colors_and_alphas( data_numeric.columns, cmap)
        plot_type = _validate_plot_type(
            plot_type, target_strs= ['hist', 'box', 'density'],
            raise_exception= True)
        if plot_type == 'density':
            for column in data_numeric.columns:
                sns.kdeplot(data_numeric[column],
                            label=f'Skewness: {skewness_value[column]:.2f}', 
                            fill=True)
            plt.title('Density Plot with Skewness')
        elif plot_type == 'hist':
            for column in data_numeric.columns:
                plt.hist(
                data_numeric[column], alpha=0.5, 
                label=f'{column} (Skewness: {skewness_value[column]:.2f})',
                bins=30)
            plt.title('Histogram with Skewness')
        else: 
            raise ValueError(
                f"Unsupported type '{plot_type}'. Expect 'density' and 'hist'")
        plt.xlabel("Value")
        plt.ylabel ("Frequency")
        if plt.gca().get_legend_handles_labels()[0]:
            plt.legend()
        plt.show()
        
    skewness_value= convert_and_format_data(
        skewness_value, as_frame, 
        force_array_output= False if as_frame else True, 
        condense=False if as_frame else True,
        condition=series_naming ("skewness"),
        )    
    
    return skewness_value


@make_data_dynamic(
    capture_columns=True)
def kurtosis(
    data: Union[ArrayLike, DataFrame],
    columns: List[str] = None,
    axis: int= 0, 
    as_frame: bool = False,
    view: bool = False, 
    plot_type: str = 'density', 
    cmap: str = 'viridis', 
    fig_size: Optional[Tuple[int, int]] = (10, 6), 
    **kwargs
    ):
    """
    Calculates the kurtosis of the data distribution, offering insights into
    the tail heaviness compared to a normal distribution.

    Kurtosis is a measure of whether the data are heavy-tailed or light-tailed
    relative to a normal distribution. Positive kurtosis indicates a distribution
    with heavier tails, while negative kurtosis indicates a distribution with
    lighter tails.

    The formula for kurtosis is defined as:

    .. math::

        Kurtosis = \\frac{N\\sum_{i=1}^{N}(X_i - \\bar{X})^4}{(\\sum_{i=1}^{N}(X_i - \\bar{X})^2)^2} - 3

    where:
    
    - :math:`N` is the number of observations,
    - :math:`X_i` is each individual observation,
    - :math:`\\bar{X}` is the mean of the observations.

    The subtraction by 3 at the end adjusts the result to fit the standard 
    definition of kurtosis, where a normal distribution has a kurtosis of 0.

    Parameters
    ----------
    data : ArrayLike or pd.DataFrame
        The input data for which kurtosis is to be calculated. Can be a Pandas
        DataFrame or any array-like structure containing numerical data.
    columns : List[str], optional
        Specific columns to include in the analysis if `data` is a DataFrame.
        If None, all columns are included. Default is None.
    axis : int, default=0
        Axis along which the kurtosis is calculated.
    as_frame : bool, default=False
        If True and `data` is ArrayLike, converts `data` to a Pandas DataFrame
        before calculating kurtosis. Useful for structured array-like inputs.
    view : bool, optional
        If True, visualizes the data distribution along with its kurtosis
        using the specified `plot_type`. Defaults to False.
    plot_type : str, optional
        Type of plot to visualize the data distribution. Supported values are
        'density' for a density plot and 'hist' for a histogram. Defaults to 'density'.
    cmap : str, optional
        Colormap for the plot. Defaults to 'viridis'.
    fig_size : Optional[Tuple[int, int]], optional
        Size of the figure for the plot. Defaults to (10, 6).
    **kwargs : dict
        Additional keyword arguments passed to `scipy.stats.kurtosis`.

    Returns
    -------
    float or pd.Series
        The kurtosis value(s) of the data. Returns a single float value if `data`
        is a single column or ArrayLike without `as_frame` set to True. Returns
        a pd.Series with kurtosis values for each column if `data` is a DataFrame
        or `as_frame` is True.

    Examples
    --------
    >>> from gofast.stats.utils import kurtosis
    >>> import numpy as np
    >>> data = np.random.normal(0, 1, 1000)
    >>> print(kurtosis(data))
    
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': np.random.normal(0, 1, size=1000),
    ...                    'B': np.random.standard_t(10, size=1000)})
    >>> print(kurtosis(df, as_frame=True))
    
    >>> data = np.random.normal(0, 1, 1000)
    >>> print(kurtosis(data))
    
    >>> df = pd.DataFrame({
    ...     'normal': np.random.normal(0, 1, 1000),
    ...     'leptokurtic': np.random.normal(0, 1, 1000) ** 3,
    ... })
    >>> print(kurtosis(df, as_frame=True))

    Note
    ----
    Kurtosis is useful for understanding the extremity of outliers or the
    propensity of data to produce outliers. A higher kurtosis can indicate
    a higher risk or potential for outlier values in the dataset.
    """
    kurtosis_value = data.kurtosis(axis=axis, **kwargs)
    if view:
        plot_type= _validate_plot_type(
            plot_type, target_strs= ['density', 'hist'],
            raise_exception =True)
        colors, alphas = get_colors_and_alphas(
            len(kurtosis_value), cmap)
        kvalue, data, cols = _prepare_plot_data(kurtosis_value, data, axis = axis )
        if plot_type == 'density':
            for ii, col in enumerate (cols) :
                sns.kdeplot(data[col], label=f'{col} kurtosis={kvalue[ii]:.2f}',
                            color=colors [ii])
        elif plot_type == 'hist':
            data.hist(bins=30, alpha=0.5, color=colors[0], figsize=fig_size)
        plt.title("Data Distribution and Kurtosis")

        plt.xlabel("Value")
        plt.ylabel ("Frequency")
        if plt.gca().get_legend_handles_labels()[0]:
            plt.legend()
        plt.show()

    kurtosis_value= convert_and_format_data(
        kurtosis_value, as_frame, 
        force_array_output= False if as_frame else True, 
        condense=False if as_frame else True,
        condition=series_naming ("kurtosis"),
        )    
    return kurtosis_value

def t_test_independent(
    sample1: Union[List[float], List[int], str],
    sample2: Union[List[float], List[int], str],
    alpha: float = 0.05, 
    data: DataFrame = None, 
    as_frame: bool=True, 
    view: bool = False, 
    plot_type: str = 'box', 
    cmap: str = 'viridis', 
    fig_size: Optional[Tuple[int, int]] = (10, 6),  
    **kws
) -> Tuple[float, float, bool]:
    r"""
    Conducts an independent two-sample t-test to evaluate the difference in
    means between two independent samples. 
    
    This statistical test assesses whether there are statistically significant
    differences between the means of two independent samples.

    The t-statistic is computed as:
    
    .. math:: 
        t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}
    
    
    Where:
    - \(\bar{X}_1\) and \(\bar{X}_2\) are the sample means,
    - \(s_1^2\) and \(s_2^2\) are the sample variances,
    - \(n_1\) and \(n_2\) are the sample sizes.
    
    The function returns the t-statistic, the two-tailed p-value, and a boolean
    indicating if the null hypothesis 
    (the hypothesis that the two samples have identical average values) 
    can be rejected at the given significance level (\(\alpha\)).

    Parameters
    ----------
    sample1 : Union[List[float], List[int], str]
        The first sample or the name of a column in `data` if provided.
    sample2 : Union[List[float], List[int], str]
        The second sample or the name of a column in `data` if provided.
    alpha : float, optional
        The significance level, default is 0.05.
    data : pd.DataFrame, optional
        DataFrame containing the data if column names are provided for 
        `sample1` or `sample2`.
    as_frame : bool, optional
        If True, returns results as a pandas DataFrame/Series.
    view : bool, optional
        If True, generates a plot to visualize the sample distributions.
    plot_type : str, optional
        The type of plot for visualization ('box' or 'hist').
    cmap : str, optional
        Color map for the plot.
    fig_size : Optional[Tuple[int, int]], optional
        Size of the figure for the plot.
    **kwargs : dict
        Additional arguments passed to `stats.ttest_ind`.

    Returns
    -------
    Tuple[float, float, bool]
        t_stat : float
            The calculated t-statistic.
        p_value : float
            The two-tailed p-value.
        reject_null : bool
            A boolean indicating if the null hypothesis can be rejected 
            (True) or not (False).

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.stats.utils import t_test_independent
    >>> sample1 = [22, 23, 25, 27, 29]
    >>> sample2 = [18, 20, 21, 20, 19]
    >>> t_stat, p_value, reject_null = t_test_independent(sample1, sample2)
    >>> print(f"T-statistic: {t_stat}, P-value: {p_value}, Reject Null: {reject_null}")

    >>> df = pd.DataFrame({'Group1': [22, 23, 25, 27, 29], 'Group2': [18, 20, 21, 20, 19]})
    >>> t_stat, p_value, reject_null = t_test_independent('Group1', 'Group2', data=df)
    >>> print(f"T-statistic: {t_stat}, P-value: {p_value}, Reject Null: {reject_null}")
    
    >>> df = pd.DataFrame({'Group1': [22, 23, 25, 27, 29], 'Group2': [18, 20, 21, 20, 19]})
    >>> t_stat, p_value, reject_null = t_test_independent('Group1', 'Group2', data=df)
    >>> print(f"T-statistic: {t_stat}, P-value: {p_value}, Reject Null: {reject_null}")
    Note
    ----
    This function is particularly useful for comparing the means of two 
    independent samples, especially in assessing differences under various 
    conditions or treatments. Ensure a DataFrame is passed when `sample1` a
    nd `sample2` are specified as column names.
    """

    if isinstance(sample1, str) or isinstance (sample2, str): 
        if data is None:
            raise ValueError(
                "Data cannot be None when 'x' or 'y' is specified as a column name.")
        # Validate that data is a DataFrame
        is_frame(data, df_only=True, raise_exception=True)  
        sample1 = data[sample1] if isinstance(sample1, str) else sample1 
        sample2 = data[sample2] if isinstance(sample2, str) else  sample2 

    check_consistent_length(sample1, sample2)
    
    t_stat, p_value = stats.ttest_ind(sample1, sample2, **kws )
    reject_null = p_value < alpha

    if view:
        
        plot_type= _validate_plot_type(
            plot_type, target_strs= ['box', 'hist'],
            raise_exception =True)
        colors, alphas = get_colors_and_alphas(2, cmap)
        if plot_type == 'box':
            sns.boxplot(data=[sample1, sample2], palette=cmap)
            plt.title('Sample Distributions - Boxplot')
        elif plot_type == 'hist':
            sns.histplot(sample1, color=colors[0], alpha=alphas[0], kde=True, 
                         label='Sample 1')
            sns.histplot(sample2, color=colors[1], alpha=alphas[1], kde=True, 
                         label='Sample 2')
            plt.title('Sample Distributions - Histogram')
        
        plt.xlabel("Value")
        plt.ylabel ("Frequency")
        if plt.gca().get_legend_handles_labels()[0]:
            plt.legend()
        plt.show()
    
    if as_frame: 
        return to_series_if(
            t_stat, p_value, reject_null, 
            value_names= ["T-statistic", "P-value","Reject-Null-Hypothesis"],
            name ="t_test_independent")
    
    return t_stat, p_value, reject_null

def perform_linear_regression(
    x: Union[ArrayLike, list, str] = None, 
    y: Union[ArrayLike, list, str] = None, 
    data: DataFrame = None,
    view: bool = False, 
    sample_weight=None, 
    as_frame=False, 
    plot_type: str = 'scatter_line', 
    cmap: str = 'viridis', 
    fig_size: Optional[Tuple[int, int]] = (10, 6), 
    **kwargs
) -> Tuple[LinearRegression, ArrayLike, float]:
    """
    Performs linear regression analysis between an independent variable (x) and
    a dependent variable (y), returning the fitted model, its coefficients,
    and intercept.

    Linear regression is modeled as:

    .. math::
        y = X\beta + \epsilon
    
    where:
    
    - :math:`y` is the dependent variable,
    - :math:`X` is the independent variable(s),
    - :math:`\beta` is the coefficient(s) of the model,
    - :math:`\epsilon` is the error term.

    Parameters
    ----------
    x : str, list, or array-like, optional
        Independent variable(s). If a string is provided, `data` must
        also be supplied, and `x` should refer to a column name within `data`.
    y : str, list, or array-like, optional
        Dependent variable. Similar to `x`, if a string is provided,
        it should refer to a column name within a supplied `data` DataFrame.
    data : pd.DataFrame, optional
        DataFrame containing the variables specified in `x` and `y`.
        Required if `x` or `y` are specified as strings.
    view : bool, optional
        If True, generates a plot to visualize the data points and the
        regression line. Default is False.
    sample_weight : array-like of shape (n_samples,), default=None
        Individual weights for each sample.
    as_frame : bool, optional
        If True, returns the output as a pandas Series. Default is False.
    plot_type : str, optional
        Type of plot for visualization. Currently supports 'scatter_line'.
        Default is 'scatter_line'.
    cmap : str, optional
        Color map for the plot. Default is 'viridis'.
    fig_size : Tuple[int, int], optional
        Figure size for the plot. Default is (10, 6).
    **kwargs : dict
        Additional keyword arguments to pass to the `LinearRegression`
        model constructor.

    Returns
    -------
    model : LinearRegression
        The fitted linear regression model.
    coefficients : np.ndarray
        Coefficients of the independent variable(s) in the model.
    intercept : float
        Intercept of the linear regression model.

    Note
    ----
    This function streamlines the process of performing linear regression analysis,
    making it straightforward to model relationships between two variables and 
    extract useful statistics such as the regression coefficients and intercept.
    
    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn.linear_model import LinearRegression
    >>> x = np.random.rand(100)
    >>> y = 2.5 * x + np.random.normal(0, 0.5, 100)
    >>> model, coefficients, intercept = perform_linear_regression(x, y)
    >>> print(f"Coefficients: {coefficients}, Intercept: {intercept}")

    Using a DataFrame:
    
    >>> df = pd.DataFrame({'X': np.random.rand(100), 
    ...                    'Y': 2.5 * np.random.rand(100) + np.random.normal(0, 0.5, 100)})
    >>> model, coefficients, intercept = perform_linear_regression('X', 'Y', data=df)
    >>> print(f"Coefficients: {coefficients}, Intercept: {intercept}")
    """
    x_values, y_values = assert_xy_in(
        x, y,
        data=data, 
        xy_numeric= True
    )
    
    if _is_arraylike_1d(x_values): 
        x_values = x_values.reshape(-1, 1)
        
    model = LinearRegression(**kwargs)
    model.fit(x_values, y_values, sample_weight=sample_weight) 
    coefficients = model.coef_
    intercept = model.intercept_

    if view:
        plt.figure(figsize=fig_size)
        plt.scatter(x_values, y_values, color='blue', label='Data Points')
        plt.plot(x_values, model.predict(x_values), color='red', label='Regression Line')
        plt.title('Linear Regression Analysis')
        plt.xlabel('Independent Variable')
        plt.ylabel('Dependent Variable')
        plt.legend()
        plt.show()
        
    if as_frame:
        return to_series_if(
            model, coefficients, intercept, 
            value_names=['Linear-model', "Coefficients", "Intercept"], 
            name='linear_regression'
        )
    return model, coefficients, intercept

@make_data_dynamic('numeric', capture_columns=True)
def chi2_test(
    data: Union[ArrayLike, DataFrame],
    alpha: float = 0.05, 
    columns: List[str] = None,
    as_frame=True, 
    view: bool = False,
    plot_type=None, 
    cmap: str = 'viridis',
    fig_size: Tuple[int, int] = (12, 5),
    **kwargs
) -> Tuple[float, float, bool]:
    """
    Performs a Chi-Squared test for independence to assess the relationship 
    between two categorical variables represented in a contingency table.

    The Chi-Squared test evaluates whether there is a significant association 
    between two categorical variables by comparing the observed frequencies in 
    the contingency table with the frequencies that would be expected if there 
    was no association between the variables.

    Parameters
    ----------
    data : array_like or pd.DataFrame
        The contingency table where rows represent categories of one variable
        and columns represent categories of the other variable. If `data` is 
        array_like and `as_frame` is True, `columns` must be provided to convert 
        it into a DataFrame.
    alpha : float, optional
        The significance level for determining if the null hypothesis can be
        rejected, default is 0.05.
    columns : List[str], optional
        Column names when converting `data` from array_like to a DataFrame.
        Required if `data` is array_like and `as_frame` is True.
    as_frame : bool, optional
        If True, returns the results in a pandas Series. Default is False.
    view : bool, optional
        If True, displays a heatmap of the contingency table. Default is False.
    plot_type : str or None, optional
        The type of plot to display. Currently not implemented; reserved for future use.
    cmap : str, optional
        Colormap for the heatmap. Default is 'viridis'.
    fig_size : Tuple[int, int], optional
        Figure size for the heatmap. Default is (12, 5).
    **kwargs : dict
        Additional keyword arguments to pass to `stats.chi2_contingency`.

    Returns
    -------
    chi2_stat : float
        The Chi-squared statistic.
    p_value : float
        The p-value of the test.
    reject_null : bool
        Indicates whether to reject the null hypothesis (True) or not (False),
        based on the comparison between `p_value` and `alpha`.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from scipy import stats
    >>> import seaborn as sns
    >>> import matplotlib.pyplot as plt
    >>> from gofast.stats.utils import chi2_test 
    >>> data = pd.DataFrame({'A': [10, 20, 30], 'B': [20, 15, 30]})
    >>> chi2_stat, p_value, reject_null = chi2_test(data)
    >>> print(f"Chi2 Statistic: {chi2_stat}, P-value: {p_value}, Reject Null: {reject_null}")

    Notes
    -----
    The mathematical formulation for the Chi-Squared test statistic is:

    .. math::
        \chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}

    where :math:`O_i` are the observed frequencies, and :math:`E_i` are the
    expected frequencies under the null hypothesis that the variables are
    independent.

    The Chi-Squared test is a statistical method used to determine if there is a 
    significant association between two categorical variables. It's commonly 
    used in hypothesis testing to analyze the independence of variables in 
    contingency tables.
    """
    chi2_stat, p_value, _, _ = stats.chi2_contingency(data, **kwargs)
    reject_null = p_value < alpha
    
    if view:
        plt.figure(figsize=fig_size)
        sns.heatmap(data, annot=True, cmap=cmap, fmt="g")
        plt.title('Contingency Table')
        plt.show()
        
    if as_frame: 
        return to_series_if(
            chi2_stat, p_value, reject_null, 
            value_names=['Chi2-statistic', 'P-value', "Reject-Null-Hypothesis"], 
            name="chi_squared_test"
            )
    return chi2_stat, p_value, reject_null

@DynamicMethod( 
   'categorical',
    capture_columns=False, 
    treat_int_as_categorical=True, 
    encode_categories= True
  )
def anova_test(
    data: Union[Dict[str, List[float]], ArrayLike], 
    groups: Optional[Union[List[str], np.ndarray]]=None, 
    columns: List[str] = None,
    alpha: float = 0.05,
    view: bool = False,
    as_frame=True, 
    cmap: str = 'viridis',
    fig_size: Tuple[int, int] = (12, 5)
):
    """
    Perform an Analysis of Variance (ANOVA) test to compare means across 
    multiple groups.

    ANOVA is used to determine whether there are any statistically significant 
    differences between the means of three or more independent (unrelated)
    groups.

    .. math::
        F = \frac{\text{Between-group variability}}{\text{Within-group variability}}

    Parameters
    ----------
    data : dict, np.ndarray, pd.DataFrame
        The input data. Can be a dictionary with group names as keys and lists 
        of values as values, a numpy array if `groups` are specified as indices,
        or a pandas DataFrame with `groups` specifying column names.
    groups : optional, List[str] or np.ndarray
        The names or indices of the groups to compare, extracted from `data`.
        If `data` is a DataFrame and `groups` is not provided, all columns 
        are used.
    columns : List[str], optional
        Specifies the column names for converting `data` from an array-like 
        format to a DataFrame. Note that this parameter does not influence 
        the function's behavior; it is included for API consistency.
    alpha : float, optional
        The significance level for the ANOVA test. Default is 0.05.
    view : bool, optional
        If True, generates a box plot of the group distributions. Default is False.
    as_frame : bool, optional
        If True, returns the result as a pandas Series. Default is True.
    cmap : str, optional
        The colormap for the box plot. Not directly used for box plots in current
        implementations but reserved for future compatibility. Default is 'viridis'.
    fig_size : Tuple[int, int], optional
        Figure size for the box plot. Default is (12, 5).

    Returns
    -------
    f_stat : float
        The calculated F-statistic from the ANOVA test.
    p_value : float
        The p-value from the ANOVA test, indicating probability of 
        observing the data if the null hypothesis is true.
    reject_null : bool
        Indicates whether the null hypothesis can be rejected based 
        on the alpha level.

    Examples
    --------
    >>> from scipy import stats
    >>> import seaborn as sns
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from gofast.stats.utils import goanova_test  
    >>> data = {'group1': [1, 2, 3], 'group2': [4, 5, 6], 'group3': [7, 8, 9]}
    >>> f_stat, p_value, reject_null = anova_test(data, alpha=0.05, view=True)
    >>> print(f"F-statistic: {f_stat}, P-value: {p_value}, Reject Null: {reject_null}")

    Notes
    -----
    The F-statistic is calculated as the ratio of the variance between the 
    groups to the variance within the groups. A higher F-statistic indicates a
    greater degree of separation between the group means.
    """

    if isinstance(data, pd.DataFrame):
        if groups:
            groups_data = [data[group].dropna().values for group in groups]
        else:
            groups_data = [data[col].dropna().values for col in data.columns]
    elif isinstance(data, np.ndarray):
        groups_data = [data[group].flatten() for group in groups]  
    else:
        raise ValueError("Unsupported data type for `data` parameter.")

    f_stat, p_value = stats.f_oneway(*groups_data)
    reject_null = p_value < alpha

    if view:
        plt.figure(figsize=fig_size)
        if isinstance(data, (dict, pd.DataFrame)):
            plot_data = pd.DataFrame(
                groups_data, index=groups if groups else data.keys()).T.melt()
            sns.boxplot(x='variable', y='value', data=plot_data, palette=cmap)
        else:
            sns.boxplot(data=np.array(groups_data), palette=cmap)
        plt.title('Group Comparisons via ANOVA')
        plt.show()
        
    if as_frame: 
        return to_series_if(
            f_stat, p_value, reject_null, 
            value_names=['F-statistic', 'P-value', 'Reject-Null-Hypothesis'], 
            name='anova_test')
    return f_stat, p_value, reject_null

@make_data_dynamic(capture_columns=True, dynamize= False )
def perform_kmeans_clustering(
    data: ArrayLike,
    n_clusters: int = 3,
    n_init="auto", 
    columns: list = None,
    view: bool = True,
    cmap='viridis', 
    fig_size: Tuple[int, int] = (8, 8),
    **kwargs
) -> Tuple[KMeans, ArrayLike]:
    r"""
    Applies K-Means clustering to the dataset, returning the fitted model and 
    cluster labels for each data point.

    K-Means clustering aims to partition `n` observations into `k` clusters 
    in which each observation belongs to the cluster with the nearest mean,
    serving as a prototype of the cluster.

    .. math::
        \underset{S}{\mathrm{argmin}}\sum_{i=1}^{k}\sum_{x \in S_i}||x - \mu_i||^2

    Where:
    - :math:`S` is the set of clusters
    - :math:`k` is the number of clusters
    - :math:`x` is each data point
    - :math:`\mu_i` is the mean of points in :math:`S_i`.

    Parameters
    ----------
    data : array_like or pd.DataFrame
        Multidimensional dataset for clustering. Can be a pandas DataFrame 
        or a 2D numpy array. If a DataFrame and `columns` is specified, only
        the selected columns are used for clustering.
    n_clusters : int, optional
        Number of clusters to form. Default is 3.
    n_init : str or int, optional
        Number of time the k-means algorithm will be run with different 
        centroid seeds. The final results will be the best output of `n_init` 
        consecutive runs in terms of inertia. If "auto", it is set to 10 or
        max(1, 2 + log(n_clusters)), whichever is larger.
    columns : list, optional
        Specific columns to use for clustering if `data` is a DataFrame. 
        Ignored if `data` is an array_like. Default is None.
    view : bool, optional
        If True, generates a scatter plot of the clusters with centroids.
        Default is True.
    cmap : str, optional
        Colormap for the scatter plot, default is 'viridis'.
    fig_size : Tuple[int, int], optional
        Figure size for the scatter plot. Default is (10, 6).
    **kwargs : dict
        Additional keyword arguments passed to the `KMeans` constructor.

    Returns
    -------
    model : KMeans
        The fitted KMeans model.
    labels : np.ndarray
        Cluster labels for each point in the dataset.

    Examples
    --------
    >>> from sklearn.datasets import make_blobs
    >>> X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
    >>> model, labels = perform_kmeans_clustering(X, n_clusters=3)
    >>> print(labels)

    Using a DataFrame and selecting specific columns:
    >>> df = pd.DataFrame(X, columns=['feature1', 'feature2'])
    >>> model, labels = perform_kmeans_clustering(
    ...     df, columns=['feature1', 'feature2'], n_clusters=3)
    >>> print(labels)

    See Also
    --------
    sklearn.cluster.KMeans : 
        The KMeans clustering algorithm provided by scikit-learn.
        
    """
    if isinstance(data, pd.DataFrame) and columns is not None:
        data_for_clustering = data[columns]
    else:
        data_for_clustering = data

    km = KMeans(n_clusters=n_clusters, n_init=n_init,  **kwargs)
    labels = km.fit_predict(data_for_clustering)

    if view:
        plt.figure(figsize=fig_size)
        # Scatter plot for clusters
        if isinstance(data_for_clustering, pd.DataFrame):
            plt.scatter(
                data_for_clustering.iloc[:, 0], data_for_clustering.iloc[:, 1],
                c=labels, cmap=cmap, marker='o', edgecolor='k', s=50, alpha=0.6)
        else:
            plt.scatter(data_for_clustering[:, 0], data_for_clustering[:, 1],
                        c=labels, cmap=cmap, marker='o', edgecolor='k',
                        s=50, alpha=0.6)
        
        # Plot centroids
        centers = km.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c='red',
                    s=200, alpha=0.5, marker='+')
        plt.title('K-Means Clustering')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()

    return km, labels

@make_data_dynamic('numeric', capture_columns=True)
def hmean(
    data: ArrayLike,
    columns: List[str] = None,
    as_frame: bool = False,
    view: bool = False,
    cmap: str = 'viridis',
    fig_size: Tuple[int, int] = (10, 6)
):
    """
    Calculate the harmonic mean of a data set and optionally visualize the 
    data distribution through a histogram.

    The harmonic mean, useful for rates and ratios, is less influenced by
    large outliers compared to the arithmetic mean. It is defined as the 
    reciprocal of the arithmetic mean of the reciprocals of the data points.

    .. math::
        H = \frac{n}{\sum_{i=1}^{n} \frac{1}{x_i}}

    Where :math:`H` is the harmonic mean, :math:`n` is the number of observations, 
    and :math:`x_i` are the data points.

    Parameters
    ----------
    data : DataFrame or ArrayLike
        The data for which the harmonic mean is desired. Can be a pandas DataFrame 
        or a numpy array. If a DataFrame and `columns` is specified, only the 
        selected columns are used for the calculation.
    columns : List[str], optional
        Specific columns to use for the calculation if `data` is a DataFrame.
    as_frame : bool, optional
        If True and `data` is ArrayLike, converts `data` into a DataFrame
        using `columns` as column names. Ignored if `data` is already a DataFrame.
    view : bool, optional
        If True, displays a histogram of the data set to visualize its
        distribution. Default is False.
    cmap : str, optional
        Colormap for the histogram. Default is 'viridis'.
    fig_size : Tuple[int, int], optional
        Size of the figure for the histogram. Default is (10, 6).

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
    >>> from gofast.stats.utils import hmean
    >>> hmean([1, 2, 4])
    1.7142857142857142

    >>> hmean(np.array([2.5, 3.0, 10.0]))
    3.5294117647058822
    
    >>> df = pd.DataFrame({'A': [2.5, 3.0, 10.0], 'B': [1.5, 2.0, 8.0]})
    >>> hmean(df, columns=['A'])
    ValueError: Data points must be greater than 0.

    See Also
    --------
    scipy.stats.hmean : Harmonic mean function in SciPy for one-dimensional arrays.
    gofast.stats.mean : Arithmetic mean function.
    """
    data_values = data.to_numpy().flatten()

    if np.any(data_values <= 0):
        raise ValueError("Data points must be greater than 0.")

    h_mean = len(data_values) / np.sum(1.0 / data_values)

    if view:
        plt.figure(figsize=fig_size)
        plt.hist(data_values, bins='auto',color=plt.get_cmap(cmap)(0.7),
                 alpha=0.7, rwidth=0.85)
        plt.title('Data Distribution')
        plt.xlabel('Data Points')
        plt.ylabel('Frequency')
        plt.axvline(h_mean, color='red', linestyle='dashed', linewidth=2)
        plt.text(h_mean, plt.ylim()[1] * 0.9, f'Harmonic Mean: {h_mean:.2f}',
                 rotation=90, verticalalignment='center')
        plt.show()

    if as_frame: 
        return to_series_if (h_mean, value_names= ['H-mean'], name='harmonic_mean') 
    return h_mean

@make_data_dynamic(capture_columns=True)
def wmedian(
    data: ArrayLike,
    weights:Union[str, int, Array1D],
    columns: Union[str, list] = None,
    as_frame: bool = False,
    view: bool = False,
    cmap: str = 'viridis',
    fig_size: Tuple[int, int] = (6, 4)
) -> float | pd.Series:
    """
    Compute the weighted median of a dataset, optionally visualizing the 
    distribution of data points and their weights.

    The weighted median is defined as the value separating the higher half 
    from the lower half of a data sample, a population, or a probability 
    distribution, where each value has a corresponding weight. It is determined 
    such that the sum of weights is equal on both sides of the median in the 
    sorted list of data points.
    
    .. math::
        WM = \text{{value}} \; x \; \text{{such that}} \; \sum_{x_i < WM} w_i < \frac{1}{2} \sum w_i 
        \; \text{{and}} \; \sum_{x_i > WM} w_i \leq \frac{1}{2} \sum w_i

    Where :math:`WM` is the weighted median, :math:`x_i` are the data points, 
    and :math:`w_i` are the weights associated with each data point.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Data for which the weighted median is desired. If a DataFrame 
        and `columns` is specified, the calculation uses only the 
        selected column(s).
    weights : str, int, np.ndarray, or list
        Weights corresponding to each element in `data`. If a string or int, it 
        specifies the column in the `data` DataFrame containing the weights.
    columns : str or list, optional
        Specific column(s) to use for the calculation if `data` is a DataFrame.
    as_frame : bool, optional
        If True, the function returns the result as a pandas Series. 
        Default is False.
    view : bool, optional
        If True, displays a scatter plot of the data points with sizes 
        proportional to their weights. Default is False.
    cmap : str, optional
        Colormap for the scatter plot. Default is 'viridis'.
    fig_size : Tuple[int, int], optional
        Size of the figure for the scatter plot. Default is (8, 8).

    Returns
    -------
    float or pd.Series
        The weighted median of the data set. Returns a pandas Series 
        if `as_frame` is True.

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.stats.utils import wmean 
    >>> data = np.array([1, 2, 3])
    >>> weights = np.array([3, 1, 2])
    >>> wmedian(data, weights)
    2.0

    >>> import pandas as pd
    >>> df = pd.DataFrame({'values': [1, 2, 3], 'weights': [3, 1, 2]})
    >>> wmedian(df, 'weights', columns='values', as_frame=True)
    W-median    2.0
    dtype: float64

    Notes
    -----
    The weighted median is particularly useful in datasets where some values 
    are inherently more important than others and should be more heavily 
    considered in the central tendency measure.

    See Also
    --------
    np.median : Compute the median along the specified axis.
    pd.Series.median : Return the median of the values for the requested axis.
    scipy.stats.weighted_median : Compute the weighted median of a data 
      sample in SciPy.
    """
    # Ensure 'weights' is a column name in the DataFrame.
    if isinstance(weights, str) and weights not in data.columns:
        raise ValueError("The 'weights' parameter, when passed as a string, "
                        "must match the name of a column in the DataFrame.")

    # Extract the relevant columns and convert weights to a numpy array 
    # if specified as a column name
    if isinstance(weights, str):
        weights = data.pop(weights).values

    # re-convert weights to a numpy array for consistency 
    # and check that all weights are positive
    weights= np.asarray(weights)
    if np.any(weights <= 0):
        raise ValueError("All weights must be greater than 0.")
    
    # Convert data to numpy array and flatten if necessary
    data_values = data.to_numpy().flatten() if data.ndim > 1 else data.values
    weights = np.asarray(weights).flatten()

    try:
        check_consistent_length(data_values, weights)
    except ValueError as e:
        raise ValueError(
            "Data and weights must be of the same length. Received lengths:"
            f" data_values={len(data_values)}, weights={len(weights)}") from e

    # Sort data and weights
    sorted_indices = np.argsort(data_values)
    sorted_data = data_values[sorted_indices]
    sorted_weights = weights[sorted_indices]
    
    # Calculate cumulative weights and find the median index
    cumulative_weights = np.cumsum(sorted_weights)
    half_weight = cumulative_weights[-1] / 2
    median_idx = np.searchsorted(cumulative_weights, half_weight)

    # Handle edge case where searchsorted returns an index equal 
    # to the size of data
    if median_idx == len(sorted_data):
        median_idx -= 1

    w_median = sorted_data[median_idx]

    if view:
        visualize_weighted_median(sorted_data, sorted_weights, w_median, cmap, fig_size)

    return to_series_if (w_median, value_names= ['W-median'], name='weighted_median'
                          )  if as_frame else w_median

def visualize_weighted_median(data, weights, w_median, cmap, fig_size):
    plt.figure(figsize=fig_size)
    plt.scatter(range(len(data)), [0] * len(data), 
                s=weights * 100, c=data, 
                cmap=cmap, alpha=0.6)
    plt.colorbar(label='Data Value')
    plt.axvline(w_median, color='red', linestyle='dashed', 
                linewidth=2, label=f'Weighted Median: {w_median}')
    plt.title('Weighted Median Visualization')
    plt.xlabel('Sorted Data Points')
    plt.yticks([])
    plt.legend()
    plt.show()


@make_data_dynamic("numeric", capture_columns=True)
def bootstrap(
    data: ArrayLike,
    n: int = 1000,
    columns: Optional[List[str]] = None,
    func: Callable | NumPyFunction = np.mean,
    as_frame: bool = False, 
    view: bool = True,
    alpha: float = .7, 
    cmap: str = 'viridis',
    fig_size: Tuple[int, int] = (10, 6),
    random_state: Optional[int] = None,
    return_ci: bool = False,
    ci: float = 0.95
) -> Union[Array1D, DataFrame, Tuple[Union[Array1D, DataFrame],
                                     Tuple[float, float]]]:
    """
    Perform bootstrapping to estimate the distribution of a statistic and 
    optionally its confidence interval.
    
    Bootstrapping is a resampling technique used to estimate statistics on a 
    population by sampling a dataset with replacement. This method allows for 
    the estimation of the sampling distribution of almost any statistic.

    Given a dataset :math:`D` of size :math:`N`, bootstrapping generates 
    :math:`B` new datasets :math:`\{D_1, D_2, \ldots, D_B\}`, each of size
    :math:`N` by sampling with replacement from :math:`D`. A statistic :math:`T` 
    is computed on each of these bootstrapped datasets.

    Parameters
    ----------
    data : DataFrame or array_like
        The data to bootstrap. If a DataFrame is provided and `columns` is 
        specified, only the selected columns are used.
    n : int, optional
        Number of bootstrap samples to generate, default is 1000.
    columns : List[str], optional
        Specific columns to use if `data` is a DataFrame.
    func : callable, optional
        The statistic to compute from the resampled data, default is np.mean.
    as_frame : bool, optional
        If True, returns results in a pandas DataFrame. Default is False.
    view : bool, optional
        If True, displays a histogram of the bootstrapped statistics. 
        Default is True.
    alpha : float, optional
        Transparency level of the histogram bars. Default is 0.7.
    cmap : str, optional
        Colormap for the histogram. Default is 'viridis'.
    fig_size : Tuple[int, int], optional
        Size of the figure for the histogram. Default is (10, 6).
    random_state : int, optional
        Seed for the random number generator for reproducibility. 
        Default is None.
    return_ci : bool, optional
        If True, returns a tuple with bootstrapped statistics and their 
        confidence interval. Default is False.
    ci : float, optional
        The confidence level for the interval. Default is 0.95.

    Returns
    -------
    bootstrapped_stats : ndarray or DataFrame
        Array or DataFrame of bootstrapped statistic values. If `return_ci` 
        is True, also returns a tuple containing
        the lower and upper bounds of the confidence interval.

    Examples
    --------
    >>> from gofast.stats.utils import bootstrap
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> data = np.arange(10)
    >>> stats = bootstrap(data, n=100, func=np.mean)
    >>> print(stats[:5])

    Using a DataFrame, returning confidence intervals:
    >>> df = pd.DataFrame({'A': np.random.rand(100), 'B': np.random.rand(100)})
    >>> stats, ci = bootstrap(df, n=1000, func=np.median, columns=['A'],
                              view=True, return_ci=True, ci=0.95)
    >>> print(f"Median CI: {ci}")
    """
    if random_state is not None:
        np.random.seed(random_state)
    data = data.to_numpy().flatten()

    bootstrapped_stats = [
        func(np.random.choice(data, size=len(data), replace=True)
             ) for _ in range(n)]

    if view:
        colors, alphas = get_colors_and_alphas(
            bootstrapped_stats, cmap, convert_to_named_color=True)
        plt.figure(figsize=fig_size)
        plt.hist(bootstrapped_stats, bins='auto', color=colors[0],
                 alpha=alpha, rwidth=0.85)
        plt.title('Distribution of Bootstrapped Statistics')
        plt.xlabel('Statistic Value')
        plt.ylabel('Frequency')
        plt.show()

    if return_ci:
        lower_bound = np.percentile(bootstrapped_stats, (1 - ci) / 2 * 100)
        upper_bound = np.percentile(bootstrapped_stats, (1 + ci) / 2 * 100)
        result = (bootstrapped_stats, (lower_bound, upper_bound))
    else:
        result = bootstrapped_stats

    if as_frame:
        return convert_and_format_data( 
            result if not return_ci else result[0], return_df=True,
            series_name="bootstrap_stats"
            ) if as_frame else np.array(bootstrapped_stats) 
    
    return result

@ensure_pkg(
    "lifelines","The 'lifelines' package is required for this function to run.")
@make_data_dynamic("numeric", capture_columns=True, dynamize=False)
def kaplan_meier_analysis(
    durations: DataFrame | np.ndarray,
    event_observed: Array1D,
    columns: Optional[List[str]]=None, 
    as_frame: bool=False, 
    view: bool = False,
    fig_size: Tuple[int, int] = (10, 6),
    **kws
):
    """
    Perform Kaplan-Meier Survival Analysis and optionally visualize the 
    survival function.

    The Kaplan-Meier estimator, also known as the product-limit estimator, is a 
    non-parametric statistic used to estimate the survival probability from 
    observed lifetimes. It is defined as:

    .. math::
        S(t) = \prod_{i: t_i < t} \left(1 - \frac{d_i}{n_i}\right)

    where \( S(t) \) is the probability of survival until time \( t \), 
    \( d_i \) is the number of death events at time \( t_i \), and \( n_i \) 
    is the number of subjects at risk of death just prior to time \( t_i \).


    Parameters
    ----------
    durations : np.ndarray
        Observed lifetimes (durations).
    event_observed : np.ndarray
        Boolean array, where 1 indicates the event is observed (failure)
        and 0 indicates the event is censored.
    view : bool, optional
        If True, displays the Kaplan-Meier survival function plot.
    columns : List[str], optional
        Specific columns to use for the analysis if `durations` is a DataFrame.
    view : bool, optional
        If True, displays the Kaplan-Meier survival function plot.
    fig_size : Tuple[int, int], optional
        Size of the figure for the Kaplan-Meier plot.
    **kws : dict
        Additional keyword arguments passed to `lifelines.KaplanMeierFitter`.

    Returns
    -------
    kmf : KaplanMeierFitter
        Fitted Kaplan-Meier estimator.

    Returns
    -------
    kmf : KaplanMeierFitter
        Fitted Kaplan-Meier estimator.

    Examples
    --------
    >>> from gofast.stats.utils import kaplan_meier_analysis
    >>> durations = [5, 6, 6, 2.5, 4, 4]
    >>> event_observed = [1, 0, 0, 1, 1, 1]
    >>> kmf = kaplan_meier_analysis(durations, event_observed)

    Using a DataFrame:
    >>> df = pd.DataFrame({'duration': [5, 6, 6, 2.5, 4, 4], 'event': [1, 0, 0, 1, 1, 1]})
    >>> kmf = kaplan_meier_analysis(df['duration'], df['event'], view=True)
    """
    from lifelines import KaplanMeierFitter

    durations = durations.squeeze()  

    kmf = KaplanMeierFitter(**kws)
    kmf.fit(durations, event_observed=event_observed)
    
    if view:
        plt.figure(figsize=fig_size)
        kmf.plot_survival_function()
        plt.title('Kaplan-Meier Survival Curve')
        plt.xlabel('Time')
        plt.ylabel('Survival Probability')
        plt.grid(True)
        plt.show()
        
    if as_frame: 
        return to_series_if(
            kmf , value_names=["KaplanMeier-model"], name="KM_estimate") 
    
    return kmf

@make_data_dynamic(capture_columns=True, dynamize=False)
def gini_coeffs(
    data: Union[pd.DataFrame, np.ndarray],
    columns: Optional[Union[str, List[str]]] = None,
    as_frame: bool = False,
    view: bool = False,
    fig_size: Tuple[int, int] = (10, 6)
):
    """
    Calculate the Gini coefficient of a dataset and optionally visualize
    the Lorenz curve.

    The Gini coefficient is a measure of statistical dispersion intended to 
    represent the income or wealth distribution of a nation's residents, 
    and is the most commonly used measurement of inequality. It is defined 
    mathematically based on the Lorenz curve, which plots the proportion of 
    the total income of a population that is cumulatively earned by the 
    bottom x% of the population.

    The Gini coefficient (G) can be calculated using the formula:

    .. math:: G = \frac{\sum_{i=1}^{n} (2i - n - 1) x_{i}}{n \sum_{i=1}^{n} x_{i}}

    where \( n \) is the number of values, \( x_{i} \) is the value after 
    sorting the data in increasing order, and \( i \) is the rank of values 
    in ascending order.

    Parameters
    ----------
    data : Union[pd.DataFrame, np.ndarray]
        Input data for which to calculate the Gini coefficient. Can be a 
        pandas DataFrame or a numpy ndarray.
    columns : Optional[Union[str, List[str]]], optional
        If provided, specifies the column(s) to use when `data` is a DataFrame.
        If a single string is provided, it will select a single column.
        If a list of strings is provided, it will select multiple columns, 
        and the Gini coefficient will be calculated for each column separately.
    as_frame : bool, optional
        If True, the result will be returned as a pandas DataFrame.
        Default is False.
    view : bool, optional
        If True, displays the Lorenz curve plot. Default is False.
    fig_size : Tuple[int, int], optional
        Size of the figure to display if `view` is True. Default is (10, 6).

    Returns
    -------
    Union[float, pd.DataFrame]
        The Gini coefficient of the data. If `as_frame` is True, returns a 
        pandas DataFrame with the Gini coefficient.

    See Also
    --------
    plot_lorenz_curve : A function to plot the Lorenz curve.

    Notes
    -----
    The Gini coefficient is a widely used measure of inequality. A Gini 
    coefficient of zero expresses perfect equality where all values are the 
    same. A Gini coefficient of one (or 100%) expresses maximal inequality 
    among values (for example, where only one person has all the income).

    Examples
    --------
    >>> import numpy as np 
    >>> import pandas as pd 
    >>> from gofast.stats.utils import gini_coeffs
    >>> gini_coeffs(np.array([1, 2, 3, 4, 5]))
    0.26666666666666666

    >>> df = pd.DataFrame({'income': [1, 2, 3, 4, 5]})
    >>> gini_coeffs(df, columns='income', view=True)
    # This will calculate the Gini coefficient for the 'income' column and 
    # display the Lorenz curve.

    >>> df = pd.DataFrame({'income': [1, 2, 3, 4, 5], 'wealth': [5, 4, 3, 2, 1]})
    >>> gini_coeffs(df, columns=['income', 'wealth'], as_frame=True)
    # This will calculate the Gini coefficient for both 'income' and 'wealth' 
    # columns and return the results in a DataFrame.
    """
    # Ensure data is a 1D numpy array
    data = np.ravel(data)
    
    # Sort data
    data = np.sort(data)
    
    # Calculate Gini coefficient
    n = len(data)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * data) - (n + 1) * np.sum(data)) / (n * np.sum(data))
    
    # Visualize Lorenz curve if requested
    if view:
        plot_lorenz_curve(data, fig_size)

    if as_frame:
        return pd.DataFrame({'Gini-coefficients': [gini]}, index=['gini_coeffs'])

    return gini

def plot_lorenz_curve(data: np.ndarray, fig_size: Tuple[int, int]):
    """
    Plot the Lorenz curve for the given data.

    Parameters
    ----------
    data : np.ndarray
        Sorted 1D data array for which to plot the Lorenz curve.
    fig_size : Tuple[int, int]
        Size of the figure for the Lorenz curve plot.
    """
    lorenz_curve = np.cumsum(data) / np.sum(data)
    lorenz_curve = np.insert(lorenz_curve, 0, 0)  # Start at 0
    
    plt.figure(figsize=fig_size)
    plt.plot(np.linspace(0.0, 1.0, len(lorenz_curve)), lorenz_curve,
             label='Lorenz Curve', color='blue')
    plt.plot([0, 1], [0, 1], label='Line of Equality', 
             linestyle='--', color='red')
    plt.title('Lorenz Curve')
    plt.xlabel('Cumulative share of population')
    plt.ylabel('Cumulative share of wealth')
    plt.legend()
    plt.grid(True)
    plt.show()

@make_data_dynamic(capture_columns=True, dynamize=False)
def mds_similarity(
    data,
    n_components: int = 2,
    columns: Optional[list] = None,
    as_frame: bool = False,
    view: bool = False,
    cmap: str = 'viridis',
    fig_size: Optional[tuple] = (10, 6),
    **kws
):
    """
    Perform Multidimensional Scaling (MDS) to project the dataset into a 
    lower-dimensional space while preserving the pairwise distances between 
    points as much as possible.

    MDS seeks a low-dimensional representation of the data in which the 
    distances respect well the distances in the original high-dimensional space.

    .. math::
        \min_{X} \sum_{i<j} (||x_i - x_j|| - d_{ij})^2

    where :math:`d_{ij}` are the distances in the original space, :math:`x_i` and 
    :math:`x_j` are the coordinates in the lower-dimensional space, and 
    :math:`||\cdot||` denotes the Euclidean norm.

    Parameters
    ----------
    data : DataFrame or ArrayLike
        The dataset to perform MDS on. If a DataFrame and `columns` is specified,
        only the selected columns are used.
    n_components : int, optional
        The number of dimensions in which to immerse the dissimilarities,
        by default 2.
    columns : list, optional
        Specific columns to use if `data` is a DataFrame, by default None.
    as_frame : bool, optional
        If True, the function returns the result as a pandas DataFrame,
        by default False.
    view : bool, optional
        If True, displays a scatter plot of the MDS results, by default False.
    cmap : str, optional
        Colormap for the scatter plot, by default 'viridis'.
    fig_size : tuple, optional
        Size of the figure for the scatter plot, by default (10, 6).
    **kws : dict
        Additional keyword arguments passed to `sklearn.manifold.MDS`.

    Returns
    -------
    mds_result : ndarray or DataFrame
        The coordinates of the data in the MDS space as a NumPy array or
        pandas DataFrame, depending on the `as_frame` parameter.

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from gofast.stats.utils import mds_similarity
    >>> digits = load_digits()
    >>> mds_coordinates = mds_similarity(digits.data, n_components=2, view=True)
    >>> print(mds_coordinates.shape)
    (1797, 2)

    Using with a DataFrame and custom columns:

    >>> import pandas as pd
    >>> df = pd.DataFrame(digits.data, columns=[f'pixel_{i}' for i in range(
    ...    digits.data.shape[1])])
    >>> mds_df = mds_similarity(df, columns=['pixel_0', 'pixel_64'], as_frame=True, view=True)
    >>> print(mds_df.head())

    This function is particularly useful for visualizing high-dimensional 
    data in two or three dimensions, allowing insights into the structure and
    relationships within the data that are not readily apparent in 
    the high-dimensional space.
    """
    # Ensure the data is an array for MDS processing
    data_array = np.asarray(data)

    # Initialize and apply MDS
    mds = MDS(n_components=n_components, **kws)
    mds_result = mds.fit_transform(data_array)
    
    # Visualization
    if view:
        plt.figure(figsize=fig_size)
        scatter = plt.scatter(mds_result[:, 0], mds_result[:, 1], cmap=cmap)
        plt.title('Multidimensional Scaling (MDS) Results')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.colorbar(scatter, label='Data Value')
        plt.show()
    
    # Convert the result to a DataFrame if requested
    if as_frame:
        columns = [f'Component {i+1}' for i in range(n_components)]
        mds_result = pd.DataFrame(mds_result, columns=columns)
    
    return mds_result

@ensure_pkg("skbio", "'scikit-bio' package is required for `dca_analysis` to run.")
@make_data_dynamic(capture_columns=True, dynamize=False)
def dca_analysis(
    data,
    columns: Optional[list] = None,
    as_frame: bool = False,
    view: bool = False,
    cmap: str = 'viridis',
    fig_size: Optional[Tuple[int, int]] = (10, 6),
    **kws
):
    """
    Perform Detrended Correspondence Analysis (DCA) on ecological data to identify
    the main gradients in species abundance or occurrence data across sites.
    Optionally, visualize the species scores in the DCA space.

    DCA is an indirect gradient analysis approach which focuses on non-linear 
    relationships among variables. It's particularly useful in ecology for 
    analyzing species distribution patterns across environmental gradients.

    .. math::
        \\text{DCA is based on the eigen decomposition: } X = U \\Sigma V^T

    Where:
    - :math:`X` is the data matrix,
    - :math:`U` and :math:`V` are the left and right singular vectors,
    - :math:`\\Sigma` is a diagonal matrix containing the singular values.

    Parameters
    ----------
    data : DataFrame or ArrayLike
        Ecological dataset for DCA. If a DataFrame and `columns` is specified,
        only the selected columns are used.
    columns : list, optional
        Specific columns to use if `data` is a DataFrame. Useful for specifying
        subset of data for analysis.
    as_frame : bool, optional
        If True, returns the result as a pandas DataFrame. Useful for further
        data manipulation and analysis.
    view : bool, optional
        If True, displays a scatter plot of species scores in the DCA space. 
        Helpful for visual examination of species distribution patterns.
    cmap : str, optional
        Colormap for the scatter plot. Enhances plot aesthetics.
    fig_size : tuple, optional
        Size of the figure for the scatter plot. Allows customization of the plot size.
    **kws : dict
        Additional keyword arguments passed to the DCA function in `skbio`.

    Returns
    -------
    dca_result : OrdinationResults or DataFrame
        Results of DCA, including axis scores and explained variance. The format
        of the result is determined by the `as_frame` parameter.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from gofast.stats.utils import dca_analysis
    >>> X, y = make_classification(n_samples=100, n_features=5, n_informative=2)
    >>> dca_result = dca_analysis(X, as_frame=True, view=True)
    >>> print(dca_result.head())

    This function is an essential tool for ecologists and environmental scientists 
    looking to explore and visualize complex ecological datasets, revealing patterns 
    and relationships that might not be apparent from raw data alone.
    """
    from skbio.stats.ordination import detrended_correspondence_analysis
    
    # Perform DCA
    dca_result = detrended_correspondence_analysis(data, **kws)
    
    # Visualization
    if view:
        species_scores = dca_result.samples
        plt.figure(figsize=fig_size)
        scatter = plt.scatter(species_scores.iloc[:, 0],
                              species_scores.iloc[:, 1], cmap=cmap)
        plt.title('DCA Species Scores')
        plt.xlabel('DCA Axis 1')
        plt.ylabel('DCA Axis 2')
        plt.colorbar(scatter, label='Species Abundance')
        plt.show()
    
    # Convert to DataFrame if requested
    if as_frame:
        # Assuming 'samples' attribute contains species scores 
        # which are typical in DCA results
        dca_df = pd.DataFrame(dca_result.samples, columns=['DCA Axis 1', 'DCA Axis 2'])
        return dca_df
    
    return dca_result

@make_data_dynamic(capture_columns=True, dynamize=False)
def perform_spectral_clustering(
    data: Union[DataFrame, ArrayLike],
    n_clusters: int = 2,
    assign_labels: str = 'discretize',
    as_frame: bool=False, 
    random_state: int = None,
    columns: Optional[list] = None,
    view: bool = False,
    cmap: str = 'viridis',
    fig_size: Optional[Tuple[int, int]] = (10, 6),
    **kws
):
    """
    Perform Spectral Clustering on a dataset, with an option to visualize
    the clustering results.

    Spectral Clustering uses the eigenvalues of a similarity matrix to perform
    dimensionality reduction before clustering in fewer dimensions. This method
    is particularly effective for identifying clusters that are not necessarily
    globular.

    .. math::
        L = D^{-1/2} (D - W) D^{-1/2} = I - D^{-1/2} W D^{-1/2}

    Where :math:`W` is the affinity matrix, :math:`D` is the diagonal degree matrix,
    and :math:`L` is the normalized Laplacian.

    Parameters
    ----------
    data : DataFrame or ArrayLike
        Dataset for clustering. If a DataFrame and `columns` is specified,
        only the selected columns are used.
    n_clusters : int, optional
        Number of clusters to form, default is 2.
    assign_labels : str, optional
        Strategy for assigning labels in the embedding space: 'kmeans',
        'discretize', or 'cluster_qr', default is 'discretize'.
    assign_labels : {'kmeans', 'discretize', 'cluster_qr'}, default='kmeans'
        The strategy for assigning labels in the embedding space. There are two
        ways to assign labels after the Laplacian embedding. k-means is a
        popular choice, but it can be sensitive to initialization.
        Discretization is another approach which is less sensitive to random
        initialization .

    random_state : int, RandomState instance, default=None
        A pseudo random number generator used for the initialization
        of the lobpcg eigenvectors decomposition when `eigen_solver ==
        'amg'`, and for the K-Means initialization. Use an int to make
        the results deterministic across calls.

    columns : list, optional
        Specific columns to use if `data` is a DataFrame.
    view : bool, optional
        If True, displays a scatter plot of the clustered data.
    cmap : str, optional
        Colormap for the scatter plot.
    fig_size : tuple, optional
        Size of the figure for the scatter plot.
    **kws : dict
        Additional keyword arguments passed to `SpectralClustering`.
    
    See Also
    --------
    sklearn.cluster.SpectralClustering: Spectral clustering
    sklearn.cluster.KMeans : K-Means clustering.
    sklearn.cluster.DBSCAN : Density-Based Spatial Clustering of
        Applications with Noise.
        
    Returns
    -------
    labels : ndarray or DataFrame
        Labels of each point. Returns a DataFrame if `as_frame=True`, 
        containing the original data and a 'cluster' column with labels.

    Examples
    --------
    >>> from sklearn.datasets import make_circles
    >>> from gofast.stats import perform_spectral_analysis 
    >>> X, _ = make_circles(n_samples=300, noise=0.1, factor=0.2, random_state=42)
    >>> labels = perform_spectral_clustering(X, n_clusters=2, view=True)

    Using a DataFrame and returning results as a DataFrame:
    >>> df = pd.DataFrame(X, columns=['x', 'y'])
    >>> results_df = perform_spectral_clustering(df, n_clusters=2, as_frame=True)
    >>> print(results_df.head())
    """
    data_for_clustering = np.asarray(data)

    clustering = SpectralClustering(
        n_clusters=n_clusters,
        assign_labels=assign_labels,
        random_state=random_state,
        **kws
    )
    labels = clustering.fit_predict(data_for_clustering)

    if view:
        _plot_clustering_results(data_for_clustering, labels, cmap, fig_size)

    if as_frame:
        results_df = pd.DataFrame(
            data_for_clustering, columns=columns if columns else [
                'feature_{}'.format(i) for i in range(data_for_clustering.shape[1])])
        results_df['cluster'] = labels
        return results_df

    return labels

def _plot_clustering_results(data, labels, cmap, fig_size):
    """Helper function to plot clustering results."""
    plt.figure(figsize=fig_size if fig_size else (10, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=cmap, alpha=0.6)
    plt.title('Spectral Clustering Results')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Cluster Label')
    plt.show()

def levene_test(
    *samples: Union[List[Array1D], DataFrame], 
    columns: Optional[List[str]] = None,
    center: str = 'median', 
    proportiontocut: float = 0.05, 
    as_frame=True, 
    view: bool = False, 
    cmap: str = 'viridis', 
    fig_size: Optional[Tuple[int, int]] = None,
    **kws
):
    """
    Perform Levene's test for equal variances across multiple samples, with an
    option to visualize the sample distributions.

    Levene's test is used to assess the equality of variances for a variable
    calculated for two or more groups. It is more robust to departures from
    normality than Bartlett's test. The test statistic is based on a measure
    of central tendency (mean, median, or trimmed mean).

    .. math:: 
        
        W = \\frac{(N - k)}{(k - 1)} \\frac{\\sum_{i=1}^{k} n_i (Z_{i\\cdot} - Z_{\\cdot\\cdot})^2}
        \; {\\sum_{i=1}^{k} \\sum_{j=1}^{n_i} (Z_{ij} - Z_{i\\cdot})^2}

    where :math:`W` is the test statistic, :math:`N` is the total number of 
    observations,:math:`k` is the number of groups, :math:`n_i` is the number
    of observations in group :math:`i`, :math:`Z_{ij}` is the deviation from 
    the group mean or median, depending on the centering
    method used.

    Parameters
    ----------
    *samples : Union[List[np.ndarray], pd.DataFrame]
        The sample data, possibly with different lengths. If a DataFrame is
        provided and `columns` are specified, it extracts data for each
        column name to compose samples.
    columns : List[str], optional
        Column names to extract from the DataFrame to compose samples.
        Ignored if direct arrays are provided.
    center : {'median', 'mean', 'trimmed'}, optional
        Specifies which measure of central tendency to use in computing the
        test statistic. Default is 'median'.
    proportiontocut : float, optional
        Proportion (0 to 0.5) of data to cut from each end when center is 
        'trimmed'. Default is 0.05.
    as_frame : bool, optional
        If True, returns the results as a pandas DataFrame. Default is True.
    view : bool, optional
        If True, displays a boxplot of the samples. Default is False.
    cmap : str, optional
        Colormap for the boxplot. Currently unused but included for future 
        compatibility.
    fig_size : Tuple[int, int], optional
        Size of the figure for the boxplot if `view` is True.
    **kws : dict
        Additional keyword arguments passed to `stats.levene`.

    Returns
    -------
    statistic : float
        The test statistic for Levene's test.
    p_value : float
        The p-value for the test.

    Examples
    --------
    Using direct array inputs:

    >>> import numpy as np 
    >>> import pandas as pd 
    >>> from gofast.stats.utils import levene_test
    >>> sample1 = np.random.normal(loc=0, scale=1, size=50)
    >>> sample2 = np.random.normal(loc=0.5, scale=1.5, size=50)
    >>> sample3 = np.random.normal(loc=-0.5, scale=0.5, size=50)
    >>> statistic, p_value = levene_test(sample1, sample2, sample3, view=True)
    >>> print(f"Statistic: {statistic}, p-value: {p_value}")

    Using a DataFrame and specifying columns:

    >>> df = pd.DataFrame({'A': np.random.normal(0, 1, 50),
    ...                    'B': np.random.normal(0, 2, 50),
    ...                    'C': np.random.normal(0, 1.5, 50)})
    >>> statistic, p_value = levene_test(df, columns=['A', 'B', 'C'], view=True)
    >>> print(f"Statistic: {statistic}, p-value: {p_value}")
    """
    # Check if *samples contains a single DataFrame and columns are specified
    samples = process_and_extract_data(
        *samples, columns =columns, allow_split= True ) 
    statistic, p_value = stats.levene(
        *samples, center=center, proportiontocut=proportiontocut, **kws)

    if view:
        _visualize_samples(samples, columns=columns, fig_size=fig_size)

    if as_frame: # return series by default 
        return to_series_if(
            statistic, p_value, value_names=['L-statistic', "P-value"], 
            name ='levene_test'
            )
    return statistic, p_value

def _visualize_samples(samples, columns=None, fig_size=None):
    """
    Visualizes sample distributions using boxplots.
    """
    plt.figure(figsize=fig_size if fig_size else (10, 6))
    plt.boxplot(samples, patch_artist=True, vert=True)
    labels = columns if columns else [f'Sample {i+1}' for i in range(len(samples))]
    plt.xticks(ticks=np.arange(1, len(samples) + 1), labels=labels)
    plt.title('Sample Distributions - Levene’s Test for Equal Variances')
    plt.ylabel('Values')
    plt.xlabel('Samples/Groups')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def kolmogorov_smirnov_test(
    data1: Union[Array1D, str],
    data2: Union[Array1D, str],  
    as_frame: bool = False, 
    alternative: str = 'two-sided',
    data: Optional[DataFrame] = None, 
    method: str = 'auto', 
    view: bool = False, 
    cmap: str = 'viridis', 
    fig_size: Optional[Tuple[int, int]] = None,
):
    """
    Perform the Kolmogorov-Smirnov (KS) test for goodness of fit between two
    samples, with an option to visualize the cumulative distribution functions
    (CDFs).

    The KS test is a nonparametric test that compares the empirical 
    distribution functions of two samples to assess whether they come from 
    the same distribution. It is based on the maximum difference between the 
    two cumulative distributions.

    .. math::
        D_{n,m} = \\sup_x |F_{1,n}(x) - F_{2,m}(x)|

    where :math:`\\sup_x` is the supremum of the set of distances, 
    :math:`F_{1,n}` and :math:`F_{2,m}` are the empirical distribution 
    functions of the first and second sample, respectively, and
    :math:`n` and :math:`m` are the sizes of the first and second sample.

    Parameters
    ----------
    data1, data2 : Union[np.ndarray, str]
        The sample observations, assumed to be drawn from a continuous distribution.
        If strings are provided, they should correspond to column names in `data`.
    as_frame : bool, optional
        If True, returns the test results as a pandas DataFrame if shape is 
        greater than 1 and pandas Series otherwise.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the null hypothesis ('two-sided' default).
    data : pd.DataFrame, optional
        DataFrame containing the columns referred to by `data1` and `data2`.
    method : {'auto', 'exact', 'approx'}, optional
        Method used to compute the p-value.
    view : bool, optional
        If True, visualizes the cumulative distribution functions of both samples.
    cmap : str, optional
        Colormap for the visualization (currently not used).
    fig_size : Tuple[int, int], optional
        Size of the figure for the visualization if `view` is True.

    Returns
    -------
    statistic : float
        The KS statistic.
    p_value : float
        The p-value for the test.

    Examples
    --------
    >>> import numpy as np
    >>> from your_module import kolmogorov_smirnov_test
    >>> data1 = np.random.normal(loc=0, scale=1, size=100)
    >>> data2 = np.random.normal(loc=0.5, scale=1.5, size=100)
    >>> statistic, p_value = kolmogorov_smirnov_test(data1, data2)
    >>> print(f"KS statistic: {statistic}, p-value: {p_value}")

    Using a DataFrame and specifying columns:

    >>> import pandas as pd
    >>> df = pd.DataFrame({'group1': np.random.normal(0, 1, 100),
                           'group2': np.random.normal(0.5, 1, 100)})
    >>> statistic, p_value = kolmogorov_smirnov_test('group1', 'group2', data=df, view=True)
    >>> print(f"KS statistic: {statistic}, p-value: {p_value}")
    """

    data1, data2 = assert_xy_in(data1, data2 , data=data, xy_numeric= True ) 
    statistic, p_value = stats.ks_2samp(
        data1, data2, alternative=alternative, mode=method)

    if view:
        _visualize_cdf_comparison(data1, data2, fig_size)

    return to_series_if(
        statistic, p_value, value_names=['K-statistic', "P-value"], 
        name ='levene_test'
        ) if as_frame else ( statistic, p_value) 

def _visualize_cdf_comparison(data1, data2, fig_size):
    plt.figure(figsize=fig_size if fig_size else (10, 6))
    for sample, label in zip([data1, data2], ['Data1', 'Data2']):
        ecdf = lambda x: np.arange(1, len(x) + 1) / len(x)
        plt.step(sorted(sample), ecdf(sample), label=f'CDF of {label}')
    plt.title('CDF Comparison')
    plt.xlabel('Value')
    plt.ylabel('Cumulative Probability')
    plt.legend()
    plt.grid(True)
    plt.show()

@make_data_dynamic(capture_columns= True, dynamize =False )
def cronbach_alpha(
    items_scores: ArrayLike,
    columns: Optional[list] = None,
    as_frame: bool = False, 
    view: bool = False,
    cmap: str = 'viridis',
    fig_size: Optional[Tuple[int, int]] = None
) -> Union[float, pd.Series]:
    """
    Calculate Cronbach's Alpha for assessing the internal consistency or 
    reliability of a set of test or survey items.

    Cronbach's Alpha is defined as:

    .. math:: 
        
        \\alpha = \\frac{N \\bar{\\sigma_{item}^2}}{\\sigma_{total}^2}
        \; (1 - \\frac{\\sum_{i=1}^{N} \\sigma_{item_i}^2}{\\sigma_{total}^2})

    where :math:`N` is the number of items, :math:`\\sigma_{item}^2` is the variance 
    of each item, and :math:`\\sigma_{total}^2` is the total variance of the scores.
    
    Parameters
    ----------
    items_scores : Union[np.ndarray, pd.DataFrame]
        A 2D array or DataFrame where rows represent scoring for each item and 
        columns represent items.
    columns : Optional[list], default=None
        Specific columns to use if `items_scores` is a DataFrame. If None, 
        all columns are used.
    as_frame : bool, default=False
        If True, returns the results as a pandas DataFrame or Series.
    view : bool, default=False
        If True, displays a bar plot showing the variance of each item.
    cmap : str, default='viridis'
        Colormap for the bar plot. Currently unused but included for future 
        compatibility.
    fig_size : Optional[Tuple[int, int]], default=None
        Size of the figure for the bar plot. If None, defaults to matplotlib's default.

    Returns
    -------
    float or pd.Series
        Cronbach's Alpha as a float or as a pandas Series if `as_frame` is True.

    Notes
    -----
    Cronbach's Alpha values range from 0 to 1, with higher values indicating 
    greater internal consistency of the items. Values above 0.7 are typically 
    considered acceptable, though this threshold may vary depending on the context.
    
    Examples
    --------
    Using a numpy array:
    
    >>> import numpy as np 
    >>> from gofast.stats.utils import cronbach_alpha
    >>> scores = np.array([[2, 3, 4], [4, 4, 5], [3, 5, 4]])
    >>> cronbach_alpha(scores)
    0.75

    Using a pandas DataFrame:
        
    >>> import pandas as pd 
    >>> df_scores = pd.DataFrame({'item1': [2, 4, 3], 'item2': [3, 4, 5], 'item3': [4, 5, 4]})
    >>> cronbach_alpha(df_scores)
    0.75

    Visualizing item variances:

    >>> cronbach_alpha(df_scores, view=True)
    Displays a bar plot of item variances.

    """
    items_scores = np.asarray(items_scores)
    if items_scores.ndim == 1:
        items_scores = items_scores.reshape(-1, 1)

    item_variances = items_scores.var(axis=0, ddof=1)
    total_variance = items_scores.sum(axis=1).var(ddof=1)
    n_items = items_scores.shape[1]

    alpha = (n_items / (n_items - 1)) * (1 - item_variances.sum() / total_variance)
    
    if view:
        _visualize_item_variances(item_variances, n_items, columns, fig_size, cmap)
        
    if as_frame:
        return pd.Series([alpha], index=['Cronbach\'s Alpha'],
                         name="cronbach_alpha")
    return alpha

def _visualize_item_variances(item_variances, n_items, columns, fig_size, cmap):
    plt.figure(figsize=fig_size if fig_size else (10, 6))
    colors = plt.cm.get_cmap(cmap, n_items)
    plt.bar(range(1, n_items + 1), item_variances, color=[
        colors(i) for i in range(n_items)])
    plt.title("Item Variances")
    plt.xlabel("Item")
    plt.ylabel("Variance")
    plt.xticks(ticks=range(1, n_items + 1), labels=columns if columns else [
        f'Item {i}' for i in range(1, n_items + 1)])
    plt.show()

def friedman_test(
    *samples: Union[np.ndarray, pd.DataFrame], 
    columns: Optional[List[str]] = None, 
    method: str = 'auto', 
    as_frame: bool = False, 
    view: bool = False,
    cmap: str = 'viridis',
    fig_size: Optional[Tuple[int, int]] = None
) -> Union[Tuple[float, float], Series]:
    """
    Perform the Friedman test, a non-parametric statistical test used to detect 
    differences between groups on a dependent variable across multiple test 
    attempts.

    The Friedman test [1]_ is used when the data violate the assumptions of parametric 
    tests, such as the normality and homoscedasticity assumptions required for 
    repeated measures ANOVA.

    The test statistic is calculated as follows:

    .. math:: 
        Q = \\frac{12N}{k(k+1)}\\left[\\sum_{j=1}^{k}R_j^2 - \\frac{k(k+1)^2}{4}\\right]

    where :math:`N` is the number of subjects, :math:`k` is the number of groups, 
    and :math:`R_j` is the sum of ranks for the :math:`j`th group.

    Parameters
    ----------
    *samples : Union[np.ndarray, pd.DataFrame]
        The sample groups as separate arrays or a single DataFrame. If a DataFrame 
        is provided and `columns` are specified, it will extract data for each 
        column name to compose samples.
    columns : Optional[List[str]], optional
        Column names to extract from the DataFrame to compose samples, 
        by default None.
    method : {'auto', 'exact', 'asymptotic'}, optional
        The method used for the test:
        - 'auto' : Use exact method for small sample sizes and asymptotic
          method for larger samples.
        - 'exact' : Use the exact distribution of the test statistic.
        - 'asymptotic' : Use the asymptotic distribution of the test statistic.
        The method to use for computing p-values, by default 'auto'.
        The actual friedmanchisquare only supports sample sizes and asymptotic
        method for larger samples.
    as_frame : bool, optional
        If True, returns the test statistic and p-value as a pandas Series, 
        by default False.
    view : bool, optional
        If True, displays a box plot of the sample distributions, by default False.
    cmap : str, optional
        Colormap for the box plot, by default 'viridis'. Currently unused.
    fig_size : Optional[Tuple[int, int]], optional
        Size of the figure for the box plot, by default None.

    Returns
    -------
    Union[Tuple[float, float], pd.Series]
        The Friedman statistic and p-value, either as a tuple or as a pandas 
        Series if `as_frame` is True.

    Notes
    -----
    The Friedman test is widely used in scenarios where you want to compare
    the effects of different treatments or conditions on the same subjects,
    especially in medical, psychological, and other scientific research.
    It is an alternative to ANOVA when the normality assumption is not met.

    References
    ----------
    .. [1] Friedman, Milton. (1937). The use of ranks to avoid the assumption
          of normality implicit in the analysis of variance. Journal of the
          American Statistical Association.
    
    Examples
    --------
    Using array inputs:
     
    >>> from gofast.stats import friedman_test
    >>> group1 = [20, 21, 19, 20, 21]
    >>> group2 = [19, 20, 18, 21, 20]
    >>> group3 = [21, 22, 20, 22, 21]
    >>> statistic, p_value = friedman_test(group1, group2, group3, method='auto')
    >>> print(f'Friedman statistic: {statistic}, p-value: {p_value}')

    Using DataFrame input:

    >>> df = pd.DataFrame({'group1': group1, 'group2': group2, 'group3': group3})
    >>> statistic, p_value = friedman_test(df, columns=['group1', 'group2', 'group3'], view=True)
    >>> print(f'Friedman statistic with DataFrame input: {statistic}, p-value: {p_value}')

    """
    # Check if *samples contains a single DataFrame and columns are specified
    samples = process_and_extract_data(
        *samples, columns =columns, allow_split= True ) 
    # Convert all inputs to numpy arrays for consistency
    samples = [np.asarray(sample) for sample in samples]

    # Perform the Friedman test
    statistic, p_value = stats.friedmanchisquare(*samples)

    if view:
        _visualize_friedman_test_samples(samples, columns, fig_size)

    return to_series_if(
        statistic, p_value, value_names=["F-statistic", "P-value"],name="friedman_test"
        )if as_frame else ( statistic, p_value )

def _visualize_friedman_test_samples(samples, columns, fig_size):
    plt.figure(figsize=fig_size if fig_size else (10, 6))
    # Assume samples are already in the correct format for plotting
    plt.boxplot(samples, patch_artist=True, labels=columns if columns else [
        f'Group {i+1}' for i in range(len(samples))])
    plt.title('Sample Distributions - Friedman Test')
    plt.xlabel('Groups')
    plt.ylabel('Scores')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.show()

@ensure_pkg("statsmodels")
def mcnemar_test(
    *samples: [Array1D, DataFrame, str], 
    data: Optional [DataFrame]=None, 
    as_frame:bool=False, 
    exact:bool=True, 
    correction:bool=True, 
    view:bool=False, 
    cmap: str='viridis', 
    fig_size: Tuple [int, int]=(10, 6)
    ):
    """
    Perform McNemar's test to compare two related samples on categorical data,
    with an option to visualize the contingency table.

    McNemar's test is a non-parametric method used to determine whether there 
    are differences between two related samples. It is suitable for binary
    categorical data to compare the proportion of discrepant observations.

    Parameters
    ----------
    *samples : str or array-like
        Names of columns in `data` DataFrame or two arrays containing the samples.
        When `data` is provided, `samples` must be the names of the columns to compare.
    data : DataFrame, optional
        DataFrame containing the samples if column names are specified in `samples`.
    as_frame : bool, default=False
        If True, returns the result as a pandas Series.
    exact : bool, default=True
        If True, uses the exact binomial distribution for the test. Otherwise, 
        an asymptotic chi-squared approximation is used.
    correction : bool, default=True
        If True, applies continuity correction in the chi-squared approximation
        of the test statistic.
    view : bool, default=False
        If True, visualizes the contingency table as a heatmap.
    cmap : str, default='viridis'
        Colormap for the heatmap visualization.
    fig_size : tuple, default=(10, 6)
        Size of the figure for the heatmap.

    Returns
    -------
    statistic, p_value : float or pd.Series
        The test statistic and p-value of McNemar's test. Returns as a tuple
        or pandas Series based on the value of `as_frame`.
    
    Raises
    ------
    TypeError
        If `data` is not a DataFrame when column names are specified in `samples`.
    ValueError
        If the number of samples provided is not equal to two.

    Notes
    -----
    McNemar's test evaluates the null hypothesis that the row and column marginal
    frequencies are equal. It is commonly used in before-after studies, matched
    pair studies, or repeated measures design where the subjects are the same.

    The test statistic is calculated as follows:

    .. math:: 
        Q = \\frac{(b - c)^2}{b + c}

    where :math:`b` and :math:`c` are the off-diagonal elements of the 2x2 
    contingency table formed by the two samples.

    Examples
    --------
    Performing McNemar's test with array inputs:

    >>> from gofast.stats.utils import mcnemar_test
    >>> sample1 = [0, 1, 0, 1]
    >>> sample2 = [1, 0, 1, 1]
    >>> statistic, p_value = mcnemar_test(sample1, sample2)
    >>> print(statistic, p_value)

    Performing McNemar's test with DataFrame column names:

    >>> df = pd.DataFrame({'before': sample1, 'after': sample2})
    >>> result = mcnemar_test('before', 'after', data=df, view=True, as_frame=True)
    >>> print(result)
    """
    from statsmodels.stats.contingency_tables import mcnemar
    
    # Process input samples
    if isinstance(data, pd.DataFrame) and all(isinstance(s, str) for s in samples):
        samples = [data[col] for col in samples]
    elif not isinstance(data, pd.DataFrame) and len(samples) == 2:
        samples = list(samples)
    else:
        try: 
            samples = process_and_extract_data(*samples, allow_split= True ) 
        except: 
            raise TypeError("Invalid input: `data` must be a DataFrame and `samples`"
                            " must be column names, or `samples` must be two sequences.")

    # Ensure there are exactly two samples
    if len(samples) != 2:
        raise ValueError("McNemar's test requires exactly two related samples.")

    # Create the contingency table and perform McNemar's test
    contingency_table = pd.crosstab(samples[0], samples[1])
    result = mcnemar(contingency_table, exact=exact, correction=correction)

    # Visualization
    if view:
        _visualize_contingency_table(contingency_table, cmap=cmap, fig_size=fig_size)

    # Return results
    if as_frame:
        return pd.Series({"M-statistic": result.statistic, "P-value": result.pvalue},
                         name='McNemar_test')
    
    return result.statistic, result.pvalue

def _visualize_contingency_table(
        contingency_table, cmap='viridis', fig_size=(10, 6)):
    """
    Visualizes the contingency table of McNemar's test as a heatmap.
    """
    plt.figure(figsize=fig_size)
    sns.heatmap(contingency_table, annot=True, cmap=cmap, fmt='d')
    plt.title("McNemar's Test Contingency Table")
    plt.ylabel('Sample 1')
    plt.xlabel('Sample 2')
    plt.show()

def kruskal_wallis_test(
    *samples: Array1D|DataFrame|str, 
    data: Optional [DataFrame]=None, 
    as_frame:bool=False, 
    view:bool=False, 
    cmap: str='viridis', 
    fig_size: Tuple [int, int]=(10, 6),
    **kruskal_kws
    ):
    """
    Perform the Kruskal-Wallis H test for comparing more than two independent samples
    to determine if there are statistically significant differences between their 
    population medians. Optionally, visualize the distribution of each sample.

    The Kruskal-Wallis H test is a non-parametric version of ANOVA. It's used when the 
    assumptions of ANOVA are not met, especially the assumption of normally distributed 
    data. It ranks all data points together and then compares the sums of ranks between 
    groups.

    Parameters
    ----------
    *samples : sequence of array-like or str
        Input data for the test. When `data` is a DataFrame, `samples` can be
        column names.
    data : DataFrame, optional
        DataFrame containing the data if column names are specified in `samples`.
    as_frame : bool, default=False
        If True, returns the result as a pandas Series.
    view : bool, default=False
        If True, generates boxplots of the sample distributions. Default is False.
    cmap : str, default='viridis'
        Colormap for the boxplot visualization.
    fig_size : tuple, default=(10, 6)
        Size of the figure for the boxplot visualization.
    kruskal_kws: dict, 
        Keywords arguments passed to :func:`scipy.stats.kruskal`.
        
    Returns
    -------
    statistic, p_value : float or pd.Series
        The Kruskal-Wallis H statistic and the associated p-value. Returns as a tuple
        or pandas Series based on the value of `as_frame`.
    
    Raises
    ------
    TypeError
        If `data` is not a DataFrame when column names are specified in `samples`.
    ValueError
        If less than two samples are provided.

    Notes
    -----
    The Kruskal-Wallis test evaluates the null hypothesis that the population medians 
    of all groups are equal. It is recommended for use with ordinal data or when the 
    assumptions of one-way ANOVA are not met.

    The test statistic is calculated as follows:

    .. math:: 
        H = \\frac{12}{N(N+1)} \\sum_{i=1}^{g} \\frac{R_i^2}{n_i} - 3(N+1)

    where :math:`N` is the total number of observations across all groups, :math:`g` 
    is the number of groups, :math:`n_i` is the number of observations in the i-th 
    group, and :math:`R_i` is the sum of ranks in the i-th group.

    Examples
    --------
    Performing a Kruskal-Wallis H Test with array inputs:
    
    >>> import numpy as np 
    >>> from gofast.stats.utils import kruskal_wallis_test
    >>> sample1 = np.random.normal(loc=10, scale=2, size=30)
    >>> sample2 = np.random.normal(loc=12, scale=2, size=30)
    >>> sample3 = np.random.normal(loc=11, scale=2, size=30)
    >>> statistic, p_value = kruskal_wallis_test(sample1, sample2, sample3)
    >>> print(statistic, p_value)

    Performing a Kruskal-Wallis H Test with DataFrame column names:

    >>> df = pd.DataFrame({'group1': sample1, 'group2': sample2, 'group3': sample3})
    >>> result = kruskal_wallis_test('group1', 'group2', 'group3', data=df, view=True, as_frame=True)
    >>> print(result)
    """
    # Process input samples
    if isinstance(data, pd.DataFrame) and all(isinstance(s, str) for s in samples):
        samples = [data[col] for col in samples]
    elif not isinstance(data, pd.DataFrame) and len(samples) >= 2:
        samples = list(samples)
    else:
        try: 
            samples = process_and_extract_data(*samples, allow_split= True ) 
        except: 
            raise TypeError("Invalid input: `data` must be a DataFrame and `samples`"
                        " must be column names, or `samples` must be two or more sequences.")

    # Ensure there are at least two samples
    if len(samples) < 2:
        raise ValueError("Kruskal-Wallis H test requires at least two independent samples.")

    # Perform the Kruskal-Wallis H test
    statistic, p_value = stats.kruskal(*samples, **kruskal_kws)

    # Visualization
    if view:
        _visualize_sample_distributions(samples, cmap=cmap, fig_size=fig_size)

    # Return results
    if as_frame:
        return pd.Series({"H-statistic": statistic, "P-value": p_value},
                         name='Kruskal_Wallis_test')
    return statistic, p_value

def _visualize_sample_distributions(samples, cmap='viridis', fig_size=(10, 6)):
    """
    Visualizes the distribution of each sample using boxplots.

    Parameters
    ----------
    samples : list of array-like
        The samples to visualize.
    cmap : str, default='viridis'
        Colormap for the boxplot visualization.
    fig_size : tuple, default=(10, 6)
        Size of the figure for the boxplot visualization.
    """
    plt.figure(figsize=fig_size)
    plt.boxplot(samples, patch_artist=True)
    plt.xticks(range(1, len(samples) + 1), ['Sample ' + str(i) for i in range(
        1, len(samples) + 1)])
    plt.title('Sample Distributions - Kruskal-Wallis H Test')
    plt.ylabel('Values')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def wilcoxon_signed_rank_test(
     *samples: Array1D|DataFrame|str, 
     data: Optional [DataFrame]=None, 
     alternative:str='two-sided', 
     zero_method:str='auto', 
     as_frame:bool=False, 
     view:bool=False, 
     cmap:str='viridis', 
     fig_size:Tuple[int, int]=(10, 6), 
     **wilcoxon_kws
    ):
    """
    Perform the Wilcoxon Signed-Rank Test on two related samples and optionally
    visualize the distribution of differences between pairs.

    The Wilcoxon Signed-Rank Test is a non-parametric test used to compare two
    related samples, matched samples, or repeated measurements on a single
    sample to assess whether their population mean ranks differ. It is a paired
    difference test that can be used as an alternative to the paired Student's
    t-test when the data cannot be assumed to be normally distributed.

    Parameters
    ----------
    *samples : array-like or str
        The two sets of related samples as arrays, column names if `data` is 
        provided, or a single DataFrame.
    data : DataFrame, optional
        DataFrame containing the data if `samples` are specified as column names.
    alternative : {'two-sided', 'greater', 'less'}, optional
        Specifies the alternative hypothesis to test against the null hypothesis
        that there is no difference between the paired samples. The options are:

        - ``two-sided``: Tests for any difference between the pairs without 
          assuming a direction (i.e., it tests for the possibility of the 
                                differences being either positive or negative).
          This is the most common choice for hypothesis testing as it does 
          not require a prior assumption about the direction of the effect.

        - ``greater``: Tests for the possibility that the differences between 
          the pairs are consistently greater than zero. This option is selected
          when there is a theoretical basis or prior evidence to suggest that 
          the first sample is expected to be larger than the second.

        - ``less``: Tests for the possibility that the differences between the 
          pairs are consistently less than zero. This option is appropriate 
          when the first sample is theorized or known to be smaller than the 
          second based on prior knowledge or evidence.

        The choice of the alternative hypothesis affects the interpretation 
        of the test results and should be made based on the specific research 
        question and the directionality of the expected effect. The default is 
        ``two-sided``, which does not assume any direction of the effect and 
        allows for testing differences in both directions.

    zero_method : {'pratt', 'wilcox', 'zsplit', 'auto'}, optional
        Defines how to handle zero differences between paired samples, which 
        can occur when the measurements for both samples are identical. 
        The options are:

        - ``pratt``: Includes zero differences in the ranking process, adjusting
           ranks accordingly.
        - ``wilcox``: Discards all zero differences before the test without 
           considering them for ranking.
        - ``zsplit``: Splits zero differences evenly between positive and 
           negative ranks.
        - ``auto``: Automatically selects between 'zsplit' and 'pratt' based on 
          the presence of zero differences in the data. If zero differences 
          are detected, ``zsplit`` is used to ensure that the test accounts 
          for these observations without excluding them. If no zero differences
          are present, ``pratt`` is used to include all non-zero differences 
          in the ranking process. This option aims to balance sensitivity and 
          specificity by adapting to the data characteristics.

        The choice of method can affect the test's sensitivity to differences
        and is particularly relevant in small samples or when a significant 
        proportion of the data pairs are identical.
        The default method is ``auto``, which provides a data-driven approach 
        to handling zero differences.
    
    as_frame : bool, default=False
        If True, returns the result as a pandas Series.
    view : bool, default=False
        If True, generates a distribution plot of the differences with a zero line
        indicating no difference. Default is False.
    cmap : str, default='viridis'
        Colormap for the distribution plot.
    fig_size : tuple, default=(10, 6)
        Size of the figure for the distribution plot.
    **wilcoxon_kws : keyword arguments
        Additional keyword arguments passed to :func:`scipy.stats.wilcoxon`.

    Returns
    -------
    statistic, p_value : float or pd.Series
        The Wilcoxon Signed-Rank test statistic and the associated p-value. Returns as
        a tuple or pandas Series based on the value of `as_frame`.
    
    Notes
    -----
    The test statistic is the sum of the ranks of the differences between the paired
    samples, where the ranks are taken with respect to the absolute values of the
    differences.

    .. math:: 
        W = \\sum_{i=1}^{n} \\text{sgn}(x_{2i} - x_{1i})R_i

    where :math:`x_{1i}` and :math:`x_{2i}` are the observations in the first and
    second sample respectively, :math:`\\text{sgn}` is the sign function, and
    :math:`R_i` is the rank of the absolute difference :math:`|x_{2i} - x_{1i}|`.

    Examples
    --------
    Performing a Wilcoxon Signed-Rank Test:

    >>> from gofast.stats.utils import wilcoxon_signed_rank_test
    >>> data1 = np.random.normal(loc=10, scale=2, size=30)
    >>> data2 = data1 + np.random.normal(loc=0, scale=1, size=30)
    >>> statistic, p_value = wilcoxon_signed_rank_test(data1, data2)
    >>> print(statistic, p_value)

    Visualizing the distribution of differences:

    >>> wilcoxon_signed_rank_test(data1, data2, view=True, as_frame=True)
    """
    # Extract samples from DataFrame if specified
    if isinstance(data, pd.DataFrame) and all(isinstance(s, str) for s in samples):
        if len(samples) != 2:
            raise ValueError("Two column names must be provided with `data`.")
        data1, data2 = data[samples[0]].values, data[samples[1]].values
    elif len(samples) == 2 and all(isinstance(s, np.ndarray) for s in samples):
        data1, data2 = samples
    else:
        try: 
            data1, data2 = process_and_extract_data(*samples, allow_split= True ) 
        except: 
            raise ValueError(
                "Samples must be two arrays or two column names with `data`.")

    # Check for zero differences and adjust zero_method if 'auto'
    differences = data2 - data1
    if zero_method == 'auto':
        if np.any(differences == 0):
            zero_method = 'zsplit'
            print("Zero differences detected. Using 'zsplit' method for zero_method.")
        else:
            zero_method = 'pratt'

    # Perform the Wilcoxon Signed-Rank Test
    try:
        statistic, p_value = stats.wilcoxon(
            data1, data2, zero_method=zero_method, alternative=alternative, 
            **wilcoxon_kws)
    except ValueError as e:
        raise ValueError(f"An error occurred during the Wilcoxon test: {e}")

    # Visualization
    if view:
        _visualize_differences(data1, data2, cmap, fig_size)

    # Return results
    if as_frame:
        return pd.Series({"W-statistic": statistic, "P-value": p_value},
                         name='Wilcoxon_Signed_Rank_test')
    return statistic, p_value

def _visualize_differences(data1, data2, cmap='viridis', fig_size=(10, 6)):
    """
    Visualizes the distribution of differences between paired samples using a
    distribution plot with a line indicating no difference (zero).

    Parameters
    ----------
    data1, data2 : array-like
        The two sets of related samples.
    cmap : str, default='viridis'
        Colormap for the distribution plot.
    fig_size : tuple, default=(10, 6)
        Size of the figure for the distribution plot.
    """
    differences = data2 - data1
    plt.figure(figsize=fig_size)
    sns.histplot(differences, kde=True, color=cmap)
    plt.axvline(x=0, color='red', linestyle='--')
    plt.title('Distribution of Differences - Wilcoxon Signed-Rank Test')
    plt.xlabel('Difference')
    plt.ylabel('Frequency')
    plt.show()

def paired_t_test(
    *samples: Array1D|DataFrame|str, 
    data: Optional [DataFrame]=None, 
    as_frame: bool=False, 
    alternative:str='two-sided', 
    view:bool=False, 
    cmap:str='viridis', 
    fig_size:Tuple[int, int]=(10, 6), 
    **paired_test_kws
    ):
    """
    Perform the Paired t-Test on two related samples and optionally visualize
    the distribution of differences between pairs.

    The Paired t-Test is a parametric test used to compare two related samples,
    matched samples, or repeated measurements on a single sample to assess
    whether their population mean ranks differ. It assumes that the differences
    between the pairs are normally distributed.

    Parameters
    ----------
    *samples : array-like or str
        The two sets of related samples as arrays, column names if `data` is 
        provided, or a single DataFrame.
    data : DataFrame, optional
        DataFrame containing the data if `samples` are specified as column names.
    as_frame : bool, default=False
        If True, returns the result as a pandas Series.
    alternative : {'two-sided', 'greater', 'less'}, optional
        Defines the alternative hypothesis. The default is 'two-sided'.
    view : bool, default=False
        If True, generates a distribution plot of the differences with a zero line
        indicating no difference. Default is False.
    cmap : str, default='viridis'
        Colormap for the distribution plot.
    fig_size : tuple, default=(10, 6)
        Size of the figure for the distribution plot.
    **paired_test_kws : keyword arguments
        Additional keyword arguments passed to :func:`scipy.stats.ttest_rel`.
    Returns
    -------
    statistic, p_value : float or pd.Series
        The Paired t-Test statistic and the associated p-value. Returns as
        a tuple or pandas Series based on the value of `as_frame`.

    Notes
    -----
    The Paired t-Test is based on the differences between the pairs of 
    observations.The test statistic is computed as follows:

    .. math::
        t = \\frac{\\bar{d}}{s_{d}/\\sqrt{n}}

    where :math:`\\bar{d}` is the mean of the differences between all pairs, 
    :math:`s_{d}` is the standard deviation of these differences, and 
    :math:`n` is the number of pairs. This formula assumes that the differences 
    between pairs are normally distributed.

    The null hypothesis for the test is that the mean difference between the paired 
    samples is zero. Depending on the alternative hypothesis specified, the test can 
    be two-tailed (default), left-tailed, or right-tailed.

    Examples
    --------
    Performing a Paired t-Test:

    >>> from gofast.stats.utils import paired_t_test
    >>> data1 = np.random.normal(loc=10, scale=2, size=30)
    >>> data2 = data1 + np.random.normal(loc=0, scale=1, size=30)
    >>> statistic, p_value = paired_t_test(data1, data2)
    >>> print(statistic, p_value)

    Visualizing the distribution of differences:

    >>> paired_t_test(data1, data2, view=True, as_frame=True)
    """
    # Extract samples from DataFrame if necessary
    if data is not None:
        if len(samples) == 2 and all(isinstance(s, str) for s in samples):
            data1, data2 = data[samples[0]], data[samples[1]]
        else:
            raise ValueError("If `data` is provided, `samples`"
                             " must be two column names.")
    elif len(samples) == 2 and all(isinstance(s, np.ndarray) for s in samples):
        data1, data2 = samples
    else:
        try: 
            data1, data2  = process_and_extract_data(*samples, allow_split= True ) 
        except: 
            raise ValueError("`samples` must be two arrays or "
                             "two column names with `data`.")
    
    # Perform the Paired t-Test
    statistic, p_value = stats.ttest_rel(
        data1, data2, alternative=alternative, **paired_test_kws)

    # Visualization
    if view:
        _visualize_paired_ttest_differences(data1, data2, cmap=cmap, fig_size=fig_size)

    # Return results
    if as_frame:
        return pd.Series({"T-statistic": statistic, "P-value": p_value},
                         name='Paired_T_Test')
    return statistic, p_value

def _visualize_paired_ttest_differences(
        data1, data2, cmap='viridis', fig_size=(10, 6)):
    """
    Visualizes the distribution of differences between paired samples using a
    distribution plot with a line indicating no difference (zero).

    Parameters
    ----------
    data1, data2 : array-like
        The two sets of related samples.
    cmap : str, default='viridis'
        Colormap for the distribution plot. This will select a color from the colormap.
    fig_size : tuple, default=(10, 6)
        Size of the figure for the distribution plot.
    """
    differences = data2 - data1
    plt.figure(figsize=fig_size)
    
    # Select a color from the colormap
    color = plt.get_cmap(cmap)(0.5)  # 0.5 denotes the midpoint of the colormap
    
    sns.histplot(differences, kde=True, color=color)
    plt.axvline(x=0, color='red', linestyle='--')
    plt.title('Distribution of Differences - Paired t-Test')
    plt.xlabel('Difference')
    plt.ylabel('Frequency')
    plt.show()

@ensure_pkg(
    "statsmodels", 
    extra="'rm_anova' and 'mcnemar' tests expect statsmodels' to be installed.",
    partial_check=True,
    condition= lambda *args, **kwargs: kwargs.get("test_type") in [
        "rm_anova", "mcnemar"]
    )
def statistical_tests(
    *args, 
    test_type="mcnemar", 
    data: Optional[DataFrame]=None, 
    error_type: str = 'ci', 
    confidence_interval: float = 0.95, 
    error_bars: bool = True, 
    annot: bool = True, 
    depvar:str=None, 
    subject:str=None, 
    within: List[str] =None, 
    showmeans: bool = True, 
    split: bool = True, 
    trend_line: bool = True, 
    density_overlay: bool = False,
    as_frame: bool=False, 
    view: bool = False, 
    cmap: str = 'viridis', 
    fig_size: Optional[Tuple[int, int]] = None, 
    **kwargs
   ):
    """
    Perform a variety of statistical tests to analyze data and assess hypotheses.
    
    Function supports both parametric and non-parametric tests, catering
    to datasets with different characteristics and research designs.

    Parameters
    ----------
    *args : sequence of array-like or DataFrame
        Input data for performing the statistical test. Each array-like object
        represents a group or condition in the analysis. When `data` is a 
        DataFrame,`args` should be the names of columns in the DataFrame that 
        represent the groups or conditions to be analyzed. This flexible input
        format allows for easy integration of the function within data analysis
        workflows.
    
    test_type : str, optional
        The specific statistical test to be performed. This parameter determines
        the statistical methodology and assumptions underlying the analysis. 
        Supported tests and their applications are as follows:
            
        - ``rm_anova``: Repeated Measures ANOVA, for comparing means across more 
          than two related groups over time or in different conditions.
        - ``cochran_q``: Cochran’s Q Test, for comparing binary outcomes across 
          more than two related groups.
        - ``mcnemar``: McNemar’s Test, for comparing binary outcomes in paired 
          samples.
        - ``kruskal_wallis``: Kruskal-Wallis H Test, a non-parametric test for 
          comparing more than two independent groups.
        - ``wilcoxon``: Wilcoxon Signed-Rank Test, a non-parametric test for 
          comparing paired samples.
        - ``ttest_paired``: Paired t-Test, for comparing means of paired samples.
        - ``ttest_indep``: Independent t-Test, for comparing means of two 
          independent groups.
          
        The default test is ``mcnemar``, which is suitable for categorical data
        analysis.
    
    data : DataFrame, optional
        A pandas DataFrame containing the dataset if column names are specified 
        in `args`. This parameter allows the function to directly interface with
        DataFrame structures, facilitating the extraction and manipulation of 
        specific columns for analysis. If `data` is provided, `args` should 
        correspond to column names within this DataFrame.
        It must not be None when `test_type` is set to ``rm_anova``.
        
    depvar : str
        The name of the dependent variable within the dataset. This variable 
        is what you are trying to predict or explain, and is the main focus 
        of the ANOVA test. It should be numeric and typically represents the 
        outcome or measure that varies across different conditions or groups.
        It must not be None when `test_type` is set to ``rm_anova``.
        
    subject : str
        The name of the variable in the dataset that identifies the subject or
        participant. This variable is used to indicate which observations 
        belong to the same subject, as repeated measures ANOVA assumes multiple
        measurements are taken from the same subjects. Identifying subjects 
        allows the model to account for intra-subject variability, treating it 
        as a random effect.
        It must not be None when `test_type` is set to ``rm_anova``.
        
    within : list of str
        A list of strings where each string is the name of a within-subject 
        factor in the dataset. Within-subject factors are conditions or groups 
        that all subjects are exposed to, allowing the analysis to examine the
        effects of these factors on the dependent variable. Each factor 
        must have two or more levels (e.g., pre-test and post-test), and the 
        analysis will assess how the dependent variable changes in relation to 
        these levels, taking into account the  repeated measures nature of the data.
        
    as_frame: bool, optional 
        Returns a pandas Series or DataFrame based on number of items that 
        may compose the colums. 
       
    view : bool, optional
        Controls the generation of visualizations for the data distributions 
        or test results. If set to ``True``, the function will produce plots 
        that offer graphical representations of the analysis, enhancing 
        interpretability and insight into the data. Default is ``False``.
    
    cmap : str, optional
        Specifies the colormap to be used in the visualizations. This parameter 
        allows for customization of the plot aesthetics, providing flexibility 
        in matching visualizations to the overall theme or style of the analysis.
        Default colormap is ``viridis``.
    
    fig_size : tuple, optional
        Determines the size of the figure for the generated visualizations. 
        This tuple should contain two values representing the width and height 
        of the figure. Specifying `fig_size` allows for control over the 
        appearance of the plots, ensuring that they are appropriately sized for
        the context in which they are presented. Default is None, which will use
        matplotlib's default figure size.
    
    **kwargs : dict
        Additional keyword arguments that are specific to the chosen statistical
        test. These arguments allow for fine-tuning of the test parameters 
        and control over aspects of the analysis that are unique to each 
        statistical method. The availability and effect of these
        parameters vary depending on the `test_type` selected.

    Returns
    -------
    result : Result object
        The result of the statistical test, including the test statistic and the
        p-value. The exact structure of this result object may vary depending on the
        specific test performed, but it generally provides key information needed
        for interpretation of the test outcomes.
    
    Test Details
    ------------
    - Repeated Measures ANOVA ('rm_anova'):
        Used for comparing the means of three or more groups on the same subjects,
        commonly in experiments where subjects undergo multiple treatments. The 
        test statistic is calculated based on the within-subject variability and
        between-group differences [1]_.
        
        .. math::
            F = \\frac{MS_{between}}{MS_{within}}
    
    - Cochran’s Q Test ('cochran_q'):
        A non-parametric test for comparing three or more matched groups on binary
        outcomes. It extends McNemar's test for situations with more than two 
        related groups.
        
        .. math::
            Q = \\frac{12}{nk(k-1)} \\sum_{j=1}^{k} (T_j - \\bar{T})^2
    
    - McNemar’s Test ('mcnemar'):
        Used for binary classification to compare the proportion of misclassified
        instances between two models on the same dataset [2]_.
        
        .. math::
            b + c - |b - c| \\over 2
    
    - Kruskal-Wallis H Test ('kruskal_wallis'):
        A non-parametric version of ANOVA for comparing two or more independent
        groups. Suitable for data that do not meet the assumptions of normality
        required for ANOVA [3]_.
        
        .. math::
            H = \\frac{12}{N(N+1)} \\sum_{i=1}^{k} \\frac{R_i^2}{n_i} - 3(N+1)
    
    - Wilcoxon Signed-Rank Test ('wilcoxon'):
        A non-parametric test to compare two related samples, used when the
        population cannot be assumed to be normally distributed [4]_.
        
        .. math::
            W = \\sum_{i=1}^{n} rank(|x_i - y_i|) \\cdot sign(x_i - y_i)
    
    - Paired t-Test ('ttest_paired'):
        Compares the means of two related groups, such as in before-and-after
        studies, using the same subjects in both groups.
        
        .. math::
            t = \\frac{\\bar{d}}{s_d / \\sqrt{n}}
    
    - Independent t-Test ('ttest_indep'):
        Compares the means of two independent groups, used when different subjects
        are in each group or condition.
        
        .. math::
            t = \\frac{\\bar{X}_1 - \\bar{X}_2}{\\sqrt{\\frac{s_1^2}{n_1} + \\frac{s_2^2}{n_2}}}
    
    Notes:
    - The formulas provided are simplified representations of the test statistics
      used in each respective test. They serve as a conceptual guide to understanding
      the mathematical foundations of the tests.
    - The specific assumptions and conditions under which each test is appropriate
      should be carefully considered when interpreting the results.

    Examples
    --------
    Using the function for a paired t-test:
    
    >>> from gofast.stats.utils import statistical_tests
    >>> data1 = np.random.normal(loc=10, scale=2, size=30)
    >>> data2 = np.random.normal(loc=12, scale=2, size=30)
    >>> result = statistical_tests(data1, data2, test_type='ttest_paired')
    >>> print(result)
    
    Performing a Kruskal-Wallis H Test with DataFrame input:
    
    >>> df = pd.DataFrame({'group1': np.random.normal(10, 2, 30),
    ...                       'group2': np.random.normal(12, 2, 30),
    ...                       'group3': np.random.normal(11, 2, 30)})
    >>> result = statistical_tests(df, test_type='kruskal_wallis', 
                                   columns=['group1', 'group2', 'group3'])
    >>> print(result)
    # Sample dataset
    >>> data = {
    ...     'subject_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
    ...     'score': [5, 3, 8, 4, 6, 7, 6, 5, 8],
    ...     'time': ['pre', 'mid', 'post', 'pre', 'mid', 'post', 'pre', 'mid', 'post'],
    ...     'treatment': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']
    ... }
    >>> df = pd.DataFrame(data)
    # Perform repeated measures ANOVA
    >>> result = statistical_tests(df, depvar='score', subject='subject_id',
    ...                 within=['time', 'treatment'], test_type ='rm_anova')
    # Display the ANOVA table
    >>> print(result)

    Notes
    -----
    - The `rm_anova` and `mcnemar` tests require the `statsmodels` package.
    - Visualization is supported for all tests but is particularly informative
      for distribution-based tests like the Kruskal-Wallis H Test and Wilcoxon 
      Signed-Rank Test.
    
    The choice of test depends on the research question, data characteristics, 
    and assumptions about the data. It is crucial to select the appropriate test 
    to ensure valid and reliable results.
    
    See Also
    --------
    - `scipy.stats.ttest_rel` : For details on the paired t-test.
    - `scipy.stats.kruskal` : For details on the Kruskal-Wallis H Test.
    - `statsmodels.stats.anova.AnovaRM` : For details on Repeated Measures ANOVA.
    - `statsmodels.stats.contingency_tables.mcnemar` : For details on McNemar's Test.
    
    References
    ----------
    .. [1] Friedman, M. (1937). The use of ranks to avoid the assumption of 
           normality implicit in the analysis of variance. 
           *Journal of the American Statistical Association*, 32(200), 675-701.
    .. [2] McNemar, Q. (1947). Note on the sampling error of the difference
          between correlated proportions or percentages. *Psychometrika*, 
          12(2), 153-157.
    .. [3] Kruskal, W. H., & Wallis, W. A. (1952). Use of ranks in one-criterion 
           variance analysis. *Journal of the American Statistical Association*,
           47(260), 583-621.
    .. [4] Wilcoxon, F. (1945). Individual comparisons by ranking methods. 
          *Biometrics Bulletin*, 1(6), 80-83.
    """
    available_tests = ["cochran_q", "kruskal_wallis", "wilcoxon", "ttest_paired", 
        "ttest_indep", "rm_anova", "mcnemar"]
    error_msg = ( 
        f"Invalid test type '{test_type}'. Supported tests"
        f" are: {smart_format(available_tests)}"
        )
    test_type= normalize_string(test_type, target_strs= available_tests, 
                                match_method='contains', return_target_only=True, 
                                raise_exception=True, error_msg= error_msg,
                                deep=True)
    
    # Define test functions
    test_functions = {
        'cochran_q': lambda: stats.cochran.q(*args, **kwargs),
        'kruskal_wallis': lambda: stats.kruskal(*args),
        'wilcoxon': lambda: stats.wilcoxon(*args),
        'ttest_paired': lambda: stats.ttest_rel(*args),
        'ttest_indep': lambda: stats.ttest_ind(*args)
    }
    if isinstance (data, pd.DataFrame): 
        if all ([ isinstance (arg, str) for arg in args]): 
            # use the args as columns of the dataframe and 
            # extract arrays from this datasets 
            args = process_and_extract_data(
                *[data], columns =args, allow_split= True ) 
        else: 
            raise
            
    if test_type in ["rm_anova", "mcnemar"]: 
        from statsmodels.stats.anova import AnovaRM
        from statsmodels.stats.contingency_tables import mcnemar
        if test_type =='rm_anova': 
           test_result= AnovaRM(*args, depvar=depvar, subject=subject, 
                                within=within, **kwargs).fit() 
        elif test_type =="mcnemar": 
            test_result=mcnemar(*args, **kwargs)
    else: 
        # Execute the specified test                       
        try:
            test_result = test_functions[test_type]()
        except KeyError:
            raise ValueError(f"Invalid test type '{test_type}' specified.")
        except Exception as e : 
            raise e 
            
    # Visualization part
    if view:
        plt.figure(figsize=fig_size if fig_size else (10, 6))
        if test_type == 'mcnemar':
            sns.heatmap(test_result.table, annot=annot, cmap=cmap)
            plt.title('McNemar\'s Test Contingency Table')

        elif test_type == 'kruskal_wallis':
            sns.boxplot(data=np.array(args).T, showmeans=showmeans)
            plt.title('Kruskal-Wallis H-test')

        elif test_type == 'wilcoxon':
            if split:
                data = np.array(args[0]) - np.array(args[1])
                sns.violinplot(data=data, split=split)
            plt.title('Wilcoxon Signed-Rank Test')

        elif test_type == 'ttest_paired':
            x, y = args[0], args[1]
            plt.scatter(x, y)
            if trend_line:
                sns.regplot(x=np.array(x), y=np.array(y), ci=confidence_interval)
            plt.title('Paired T-test')

        elif test_type == 'ttest_indep':
            x, y = args[0], args[1]
            sns.histplot(x, kde=density_overlay, color='blue', 
                         label='Group 1', alpha=0.6)
            sns.histplot(y, kde=density_overlay, color='red', 
                         label='Group 2', alpha=0.6)
            plt.legend()
            plt.title('Independent Samples T-test')

        elif test_type == 'rm_anova':
            # Assuming the input data structure is suitable 
            # for this visualization
            sns.lineplot(data=np.array(args[0]),
                         err_style=error_type, 
                         ci=confidence_interval)
            plt.title('Repeated Measures ANOVA')

        elif test_type == 'cochran_q':
            # Assuming args[0] is a binary matrix or similar structure
            positive_responses = np.sum(args[0], axis=0)
            sns.barplot(x=np.arange(len(positive_responses)), 
                        y=positive_responses, 
                        ci=confidence_interval)
            plt.title('Cochran\'s Q Test')

        plt.show()
   
    return _extract_statistical_test_results (test_result, as_frame )

def _extract_statistical_test_results(
        test_result_object, return_as_frame):
    """
    Extracts statistical test results, including the statistic and p-value,
    from a given object.
    
    Parameters
    ----------
    test_result_object : object
        The object containing the statistical test results. It must 
        have attributes for the statistic value (`statistic` or similar) 
        and the p-value (`p_value` or `pvalue`).
    return_as_frame : bool
        Determines whether to return the results as a pandas DataFrame. 
        If False, the results are returned as a tuple.
    
    Returns
    -------
    tuple or pandas.DataFrame
        The statistical test results. If `return_as_frame` is True, returns
        a pandas DataFrame with columns ["Statistic", "P-value"]. Otherwise, 
        returns a tuple containing the statistic and p-value.
    
    Examples
    --------
    >>> from scipy.stats import ttest_1samp
    >>> import numpy as np
    >>> data = np.random.normal(0, 1, size=100)
    >>> test_result = ttest_1samp(data, 0)
    >>> extract_statistical_test_results(test_result, False)
    (statistic_value, p_value)
    
    >>> extract_statistical_test_results(test_result, True)
        Statistic    P-value
    0   statistic_value p_value
    """
    statistic = None
    p_value = None
    # Extract statistic and p-value from the test_result_object
    if hasattr(test_result_object, "statistic"):
        statistic = test_result_object.statistic
    if hasattr(test_result_object, "p_value"):
        p_value = test_result_object.p_value
    elif hasattr(test_result_object, 'pvalue'):
        p_value = test_result_object.pvalue
    
    # Determine the name based on object class or a custom name attribute
    name = getattr(test_result_object, '__class__', None).__name__.lower(
        ) if hasattr(test_result_object, '__class__') else getattr(
            test_result_object, 'name', '').lower()
    name = name.replace("result", "").replace("test", "") + '_test'
    
    if statistic is not None and p_value is not None:
        test_results = (statistic, p_value)
    else:
        test_results = (test_result_object,)
    
    # Convert to pandas DataFrame if requested
    if return_as_frame:
        test_results_df = pd.DataFrame([test_results], columns=["Statistic", "P-value"])
        test_results_df.rename(index={0: name}, inplace=True)
        return test_results_df
    else:
        return test_results

def _prepare_plot_data(
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
    >>> values = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
    >>> _prepare_plot_data(values)
    (Series([1, 2, 3], index=['a', 'b', 'c']), None, Index(['a', 'b', 'c']))

    >>> values = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, index=['x', 'y'])
    >>> _prepare_plot_data(values, axis=1)
    (Transposed DataFrame, Transposed DataFrame, Index(['x', 'y']))

    >>> values = np.array([1, 2, 3])
    >>> _prepare_plot_data(values, transform=np.square)
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

def _validate_plot_type(
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
    >>> _validate_plot_type('box', ['boxplot', 'histogram', 'density'],
    ...                        match_method='startswith')
    'boxplot'

    >>> _validate_plot_type('exact_plot', ['boxplot', 'histogram', 'density'],
    ...                        match_method='exact')
    None

    >>> _validate_plot_type('hist', ['boxplot', 'histogram', 'density'], 
    ...                        match_method='contains')
    'histogram'

    >>> _validate_plot_type('unknown', ['boxplot', 'histogram', 'density'],
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

def check_and_fix_rm_anova_data(
    data: DataFrame, 
    depvar: str, 
    subject: str, 
    within: List[str], 
    fix_issues: bool = False,
    strategy: str ="mean", 
    fill_value: Optional[Union[str, float, int]]=None, 
) -> DataFrame:
    """
    Checks and optionally fixes a DataFrame for repeated measures ANOVA analysis.

    This function verifies if each subject in the dataset has measurements for every 
    combination of within-subject factors. If `fix_issues` is set to True, the dataset 
    will be modified to include missing combinations, assigning `None` to the dependent 
    variable values of these new rows.

    Parameters
    ----------
    data : DataFrame
        The pandas DataFrame containing the data for ANOVA analysis.
    depvar : str
        The name of the dependent variable column in `data`.
    subject : str
        The name of the column identifying subjects in `data`.
    within : List[str]
        A list of column names representing within-subject factors.
    fix_issues : bool, optional
        If True, the dataset will be altered to include missing combinations 
        of within-subject factors for each subject. Default is False.
     strategy : str, optional
         The strategy to use for filling missing depvar values. Options are "mean",
         "median", or None. Default is "mean".
     fill_value : Optional[Union[str, float, int]], optional
         A specific value to fill missing depvar values if the strategy is None.
         Default is None, which leaves missing values as None.
    Returns
    -------
    DataFrame
        The original `data` DataFrame if `fix_issues` is False or no issues are found. 
        A modified DataFrame with issues fixed if `fix_issues` is True.

    Raises
    ------
    TypeError
        If input types for `data`, `depvar`, `subject`, or `within` are incorrect.
    ValueError
        If columns specified by `depvar`, `subject`, or `within` do not exist in `data`.

    Notes
    -----
    The mathematical formulation for identifying missing combinations involves 
    creating a Cartesian product of unique values within each within-subject 
    factor and then verifying these combinations against the existing dataset. 
    
    .. math::
    
        S = \\{s_1, s_2, ..., s_n\\} \quad \\text{(Subjects)}
        
        W_i = \\{w_{i1}, w_{i2}, ..., w_{im}\\} \quad \\text{(Within-subject factors for } i \\text{th factor)}
        
        C = W_1 \\times W_2 \\times ... \\times W_k \quad \\text{(Cartesian product of all within-subject factors)}
        
        \\text{For each subject } s \\text{ in } S, \\text{ verify } (s, c) \\in D \\text{ for all } c \\in C
        
    If combinations are missing, new rows are appended to the dataset to include 
    these missing combinations, ensuring that every subject has measurements 
    across all factor levels.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.stats.utils import check_and_fix_rm_anova_data
    >>> data = {
    ...     'subject_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
    ...     'score': [5, 3, 8, 4, 6, 7, 6, 5, 8],
    ...     'time': ['pre', 'mid', 'post', 'pre', 'mid', 'post', 'pre', 'mid', 'post'],
    ...     'treatment': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C']
    ... }
    >>> df = pd.DataFrame(data)
    >>> fixed_df = check_and_fix_rm_anova_data(
    ...     df, depvar='score', subject='subject_id', within=['time', 'treatment'],
    ...     fix_issues=True)
    >>> fixed_df
    """
    # Validate input types
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected 'data' to be a DataFrame, but got {type(data).__name__}.")
    
    if not isinstance(depvar, str):
        raise TypeError(f"'depvar' should be a string, but got {type(depvar).__name__}.")
    
    if not isinstance(subject, str):
        raise TypeError(f"'subject' should be a string, but got {type(subject).__name__}.")
    
    if not check_uniform_type (within):
        raise TypeError("All items in 'within' should be strings.")

    # Check for necessary columns in the DataFrame
    missing_columns = [col for col in [depvar, subject] + within if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in DataFrame: {', '.join(missing_columns)}.")

    # Check combinations
    combinations_df = data.groupby([subject] + within).size().reset_index(name='counts')
    expected_combinations = len(data[within[0]].unique()) * len(data[within[1]].unique())
    subject_combination_counts = combinations_df.groupby(subject).size()
    
    missing_combinations_subjects = subject_combination_counts[
        subject_combination_counts < expected_combinations]
    
    if missing_combinations_subjects.empty:
        print("All subjects have measurements for every combination of within-subject factors.")
    else:
        missing_info = ", ".join(map(str, missing_combinations_subjects.index.tolist()))
        print(f"Subjects with missing combinations: {missing_info}")
        
        if fix_issues:
            fixed_data = _fix_rm_anova_dataset(data, depvar, subject, within)
            print("Dataset issues fixed.")
            return fixed_data
    
    return data

def _fix_rm_anova_dataset(
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


@ensure_pkg ("statsmodels")
def mixed_effects_model(
    data: DataFrame, 
    formula: str, 
    groups: str, 
    re_formula: Optional[str] = None,
    data_transforms: Optional[List[Union[str, callable]]] = None,
    categorical: Optional[List[str]] = None,
    treatment: Optional[str] = None,
    order: Optional[List[str]] = None,
    summary: bool = True
) :
    """
    Fits a mixed-effects linear model to the data, accommodating both fixed 
    and random effects. 
    
    This approach is particularly useful for analyzing datasets with nested 
    structures or hierarchical levels, such as measurements taken from the 
    same subject over time or data clustered by groups. 
    
    Mixed-effects models account for both within-group (or subject) variance 
    and between-group variance.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset containing the dependent variable, independent variables, 
        subject identifiers, and other covariates.
    formula : str
        A Patsy formula string specifying the fixed effects in the model. 
        E.g., 'score ~ time + treatment'.
    groups : str
        The column name in `data` that identifies the clustering unit or subject.
        Random effects are grouped by this identifier.
    re_formula : Optional[str], default None
        A Patsy formula string defining the structure of the random effects. 
        E.g., '~time' for random slopes for time.
        If None, only random intercepts for `groups` are included.
    data_transforms : Optional[List[Union[str, callable]]], default None
        Transformations to apply to the dataset before fitting the model. 
        This can be a list of column names to convert to categorical or callable
        functions that take the DataFrame as input and return a modified
        DataFrame.
    categorical : Optional[List[str]], default None
        Columns to convert to categorical variables. This is useful for 
        ensuring that categorical predictors use the
        correct data type.
    treatment : Optional[str], default None
        If specified, indicates the column to be treated as an ordered 
        categorical variable, useful for ordinal predictors.
    order : Optional[List[str]], default None
        The order of categories for the `treatment` column, necessary if 
        `treatment` is specified. Defines the levels and their order for an 
        ordered categorical variable.
    summary : bool, default True
        If True, prints a summary of the fitted model. Otherwise, returns 
        the model fit object.

    Returns
    -------
    sm.regression.mixed_linear_model.MixedLMResults
        The results instance for the fitted mixed-effects model.

    Mathematical Formulation
    ------------------------
    The model can be described by the equation:

    .. math:: y = X\\beta + Z\\gamma + \\epsilon

    where :math:`y` is the dependent variable, :math:`X` and :math:`Z` are 
    matrices of covariates for fixed and random effects,
    :math:`\\beta` and :math:`\\gamma` are vectors of fixed and random effects 
    coefficients, and :math:`\\epsilon` is the error term.

    Usage and Application Areas
    ---------------------------
    Mixed-effects models are particularly useful in studies where data are 
    collected in groups or hierarchies, such as longitudinal studies, clustered 
    randomized trials, or when analyzing repeated measures data. They allow for
    individual variation in response to treatments and can handle unbalanced 
    datasets or missing data more gracefully than traditional repeated measures 
    ANOVA.

    Examples
    --------
    Fitting a mixed-effects model to a dataset where scores are measured across 
    different times and treatments for each subject, with subjects as a 
    random effect:
        
    >>> import numpy as np 
    >>> import pandas as pd 
    >>> from gofast.stats.utils import mixed_effects_model
    >>> df = pd.DataFrame({
    ...     'subject_id': [1, 1, 2, 2],
    ...     'score': [5.5, 6.5, 5.0, 6.0],
    ...     'time': ['pre', 'post', 'pre', 'post'],
    ...     'treatment': ['A', 'A', 'B', 'B']
    ... })
    >>> mixed_effects_model(df, 'score ~ time * treatment', 'subject_id',
    ...                            re_formula='~time')
    In this example, 'score' is modeled as a function of time, treatment, 
    and their interaction, with random slopes for time grouped by 'subject_id'.
    """
    
    import statsmodels.formula.api as smf
    # Apply data transformations if specified
    if data_transforms:
        for transform in data_transforms:
            if callable(transform):
                data = transform(data)
            elif transform in data.columns:
                data[transform] = data[transform].astype('category')
    
    # Convert specified columns to categorical if requested
    if categorical:
        for col in categorical:
            data[col] = data[col].astype('category')
    
    # Set treatment column as ordered categorical if requested
    if treatment and order:
        data[treatment] = pd.Categorical(data[treatment], categories=order, ordered=True)
    
    # Fit the model
    model = smf.mixedlm(formula, data, groups=data[groups], re_formula=re_formula)
    model_fit = model.fit()
    
    # Print or return summary
    if summary:
        print(model_fit.summary())
    else:
        return model_fit
