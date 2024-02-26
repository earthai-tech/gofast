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
import seaborn as sns 
import matplotlib.pyplot as plt 

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.linear_model import LinearRegression
from sklearn.manifold import MDS

from .._typing import DataFrame, ArrayLike, List, Dict
from .._typing import _F, Union, Tuple , Optional 
from ..decorators import DynamicMethod, AppendDocFrom # AppendDocSection 
from ..tools.validator import assert_xy_in, is_frame, check_consistent_length 
from ..tools.coreutils import ensure_visualization_compatibility, ellipsis2false 
from ..tools.coreutils import process_and_extract_data, to_series_if 
from ..tools.coreutils import get_colors_and_alphas
from ..tools.funcutils import make_data_dynamic, ensure_pkg 
from ..tools.funcutils import flatten_data_if
__all__= [ 
    "gomean", "gomedian", "gomode",  "govar", "gostd", "get_range", 
    "quartiles", "goquantile","gocorr", "correlation", "goiqr", "z_scores", 
    "descriptive_stats","calculate_skewness", "calculate_kurtosis", 
    "t_test_independent", "perform_linear_regression", "chi_squared_test", 
    "anova_test", "perform_kmeans_clustering", "harmonic_mean", 
    "weighted_median", "bootstrap", "kaplan_meier_analysis", "gini_coeffs",
    "mds_similarity", "dca_analysis", "spectral_clustering", "levene_test",
    "kolmogorov_smirnov_test", "cronbach_alpha", "friedman_test",
    "statistical_tests"
   ]


@make_data_dynamic(capture_columns=True, force_df=True)
def gomean(
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
    >>> from gofast.stats.utils import gomean
    >>> data_array = [1, 2, 3, 4, 5]
    >>> gomean(data_array)
    3.0

    >>> data_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> gomean(data_df, columns=['A'])
    2.0

    >>> gomean(data_df, axis=0)
    array([2., 5.])
    
    Calculating mean from a list:

    >>> gomean([1, 2, 3, 4, 5])
    3.0

    Calculating mean from a DataFrame and converting to Series:

    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> print(gomean(df, as_frame=True))
    A    2.0
    B    5.0
    dtype: float64

    Visualizing data distribution and mean from DataFrame:

    >>> gomean(df, view=True)
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
        mean_values, as_frame, view, gomean )
    
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
    # print("is_dataframe=", isinstance(data, pd.DataFrame))
    # # data = to_series_if(data, )
    # print("is_dataframe=", isinstance(data, pd.DataFrame))
    # print(data)
    # mean_values= mean_values.T ; data=data.T
    mean_values, data, cols= prepare_plot_data(mean_values, data, axis=axis)
    
    print(mean_values, data, cols )
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
def gomedian(
    data: Union[ArrayLike, DataFrame], 
    columns: Optional[List[str]] = None,
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
    >>> from gofast.stats.utils import gomedian
    >>> data_array = [3, 1, 4, 1, 5]
    >>> gomedian(data_array)
    3.0

    >>> data_df = pd.DataFrame({'A': [2, 4, 7], 'B': [1, 6, 5]})
    >>> gomedian(data_df, columns=['A'])
    4.0

    >>> gomedian(data_df, axis=0)
    array([4., 5.])
    
    Calculating median from a DataFrame and converting to Series:

    >>> df = pd.DataFrame({'A': [2, 4, 7], 'B': [1, 6, 5]})
    >>> print(gomedian(df, columns=['A'], as_frame=True))
    A    4.0
    dtype: float64

    Visualizing data distribution and median from DataFrame:

    >>> gomedian(df, view=True)
    
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
        median_values = data.median(**kws)
    else:
        data = np.array(data)
        median_values = np.median(data, **kws)
    
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
        median_values, as_frame, view, gomedian )
    if view:
        _visualize_median(
            data, median_values, cmap=cmap, fig_size=fig_size)

    return median_values

def _visualize_median(
        data, median_values, cmap='viridis', fig_size=None):
    """
    Visualizes the distribution of the data and highlights the median values.
    """
    plt.figure(figsize=fig_size if fig_size else (10, 6))
    
    if isinstance(data, pd.DataFrame):
        # Generate distinct colors for each column from cmap 
        colors, alphas = get_colors_and_alphas( data.columns, cmap) 
        for ii, col in enumerate (data.columns) :
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
def gomode(
    data: Union[ArrayLike, DataFrame], 
    columns: List[str] = None, 
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
    as_frame,  = ellipsis2false(as_frame)
    if isinstance(data, pd.DataFrame):
        mode_result = data.mode(**kws)
    else:
        data = np.asarray(data)
        mode_result = stats.mode(data, **kws).mode[0]
        mode_result = pd.Series(mode_result) if as_frame else mode_result
    
    mode_result, view = ensure_visualization_compatibility(
        mode_result, as_frame, view, gomode )
    if view:
        _visualize_mode(data, mode_result, cmap=cmap, fig_size=fig_size)
    
    return mode_result

def _visualize_mode(data, mode_result, cmap='viridis', fig_size=None):
    """
    Visualizes the data distribution and highlights the mode values.
    """
    plt.figure(figsize=fig_size if fig_size else (10, 6))
    
    if isinstance(data, pd.DataFrame):
        colors, alphas = get_colors_and_alphas( data.columns, cmap)
        for ii, col in enumerate (data.columns) :
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
def govar(
    data: Union[ArrayLike, DataFrame], 
    columns: Optional[List[str]] = None, 
    as_frame: bool = False, 
    view: bool = False, 
    cmap: str = 'viridis', 
    fig_size: Optional[Tuple[int, int]] = None, 
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
    
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> govar(df, as_frame=True)
    A    1.0
    B    1.0
    dtype: float64

    >>> govar(df, view=True, fig_size=(8, 4))
    Note
    ----
    The preprocessing steps, controlled by the `make_data_dynamic` decorator,
    ensure that the input data is suitable for variance calculation. This
    includes converting the data to numeric types and filtering based on 
    specified columns, enhancing the function's flexibility and applicability 
    across various data forms.
    """
    if isinstance(data, pd.DataFrame):
        variance_result = data.var(ddof=1, **kws)  # Pandas default ddof=1
    else:
        data = np.asarray(data)
        # Ensure consistency with pandas
        variance_result = np.var(data, ddof=1, **kws)  

    if as_frame and not isinstance(data, pd.DataFrame):
        variance_result = pd.Series(
            variance_result, index=columns if columns else ['Variance'])

    variance_result, view = ensure_visualization_compatibility(
        variance_result, as_frame, view, govar )
    
    if view:
        _visualize_variance(data, variance_result, columns=columns,
                            cmap=cmap, fig_size=fig_size)
    return variance_result

def _visualize_variance(data, variance_result, columns=None, cmap='viridis',
                        fig_size=None):
    """
    Visualizes the distribution of the data and highlights the variance values.
    """
    plt.figure(figsize=fig_size if fig_size else (10, 6))
    if isinstance(data, pd.DataFrame):
        colors, alphas = get_colors_and_alphas( data.columns, cmap)
        for ii, col in enumerate(data.columns):
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
def gostd(
    data: Union[ArrayLike, DataFrame], 
    columns: Optional[List[str]] = None, 
    as_frame: bool = False, 
    view: bool = False, 
    cmap: str = 'viridis', 
    fig_size: Optional[Tuple[int, int]] = None, 
    **kws):
    """
    Computes the standard deviation of numeric data provided either as an 
    array-like structure or within a pandas DataFrame. 
    
    This function leverages preprocessing to ensure the data is numeric and 
    optionally focuses on specified columns when dealing with DataFrames.

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
    view : bool, default False
        If True, visualizes the data distribution and highlights the 
        standard deviation.
    cmap : str, default 'viridis'
        Colormap for the visualization.
    fig_size : Optional[Tuple[int, int]], default None
        Size of the figure for visualization.
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

    >>> gostd(df, as_frame=True)
    A    1.0
    B    1.0
    dtype: float64

    >>> gostd(df, view=True, fig_size=(8, 4))
    Note
    ----
    The preprocessing steps, implied by the `make_data_dynamic` decorator 
    (not shown here but assumed to be applied outside this code snippet), 
    ensure that the input data is suitable for standard deviation calculation.
    These steps convert the data to numeric types and filter based on specified 
    columns, thus enhancing the function's utility and applicability across 
    diverse data forms.
    """
    as_frame = False if as_frame is ... else as_frame
    # Handling DataFrame input with optional column selection
    if isinstance(data, pd.DataFrame):
        # if columns is not None:
        #     data = data[columns]
        std_dev_result = data.std(ddof=1, **kws)  # Pandas defaults ddof=1
    else:
        # Convert ArrayLike to np.ndarray for consistent processing
        data = np.array(data)
        std_dev_result = np.std(data, ddof=1, **kws)  # Ensure consistency with pandas
    
    # Visualization of results if requested
    std_dev_result, view = ensure_visualization_compatibility(
        std_dev_result, as_frame, view, gostd )
    if view:
        _visualize_std_dev(data, std_dev_result, cmap=cmap, fig_size=fig_size)
    
    # Convert result to Series if as_frame=True
    if as_frame and not isinstance(data, pd.DataFrame):
        std_dev_result = pd.Series(
            std_dev_result, index=columns if columns else ['Standard Deviation'])

    return std_dev_result

def _visualize_std_dev(data, std_dev_result, cmap='viridis', fig_size=None):
    """
    Visualizes the distribution of the data and highlights the standard deviation.
    """
    plt.figure(figsize=fig_size if fig_size else (10, 6))
    if isinstance(data, pd.DataFrame):
        colors, alphas = get_colors_and_alphas( data.columns, cmap)
        for ii, col in enumerate(data.columns):
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
        
def _statistical_function(
    data: Union[ArrayLike, pd.DataFrame], 
    perform_statistical_analysis, 
    convert_to_dataframe_or_series, 
    view: bool = False, 
    cmap: str = 'viridis', 
    as_frame: bool = False, 
    fig_size: Optional[Tuple[int, int]] = None, 
    **kws
):
    """
    Performs a statistical operation on the provided data and optionally 
    visualizes the results.

    Parameters
    ----------
    view : bool, default=False
        If True, visualizes the statistical analysis results or the data distribution.
    cmap : str, default='viridis'
        Colormap for the visualization. Only applicable if `view` is True.
    as_frame : bool, default=False
        If True, the result is returned as a pandas DataFrame or Series, depending
        on the dimensionality of the output. Otherwise, the result is returned in
        its native format (e.g., float, np.ndarray).
    fig_size : Optional[Tuple[int, int]], default=None
        Size of the figure for the visualization. Only applicable if `view` is True.

    Returns
    -------
    The statistical analysis result, formatted according to the `as_frame` parameter.

    Examples
    --------
    >>> data = np.random.normal(0, 1, 100)
    >>> result = statistical_function(data, view=True, as_frame=True)
    """
    # statistical analysis logic
    # Note: The actual implementation of `perform_statistical_analysis` 
    # and `convert_to_dataframe_or_series`
    # functions will depend on the specific statistical 
    # operation being performed.
    result = perform_statistical_analysis(data, **kws)
    
    if as_frame:
        # Convert result to pandas DataFrame or Series if applicable
        result = convert_to_dataframe_or_series(result)
    
    result, view = ensure_visualization_compatibility(
        result, as_frame, view, gostd )
    if view:
        # Visualization logic
        _visualize_data(data, result, cmap=cmap, fig_size=fig_size)
    
    return result

   
@AppendDocFrom (
    _statistical_function,
    "Parameters",
    "Returns",
   )
@make_data_dynamic(
    capture_columns=True, 
    reset_index=True
    )   
def get_range(
    data: Union[ArrayLike, pd.DataFrame], 
    columns: Optional[List[str]] = None, 
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
    as_frame = False if as_frame is ... else as_frame
    if isinstance(data, pd.DataFrame):
        data_selected = data.copy()
    else:
        data_selected = data
    
    # Compute the range for DataFrame or ArrayLike
    if isinstance(data_selected, pd.DataFrame):
        range_values = data_selected.max(**kws) - data_selected.min(**kws)
    else:
        range_values = np.max(data_selected, **kws) - np.min(data_selected, **kws)

    # For DataFrame output
    if as_frame:
        range_values = pd.Series(
            range_values, index=columns if columns else data.columns)
    
    range_values, view  = ensure_visualization_compatibility(
        range_values, as_frame, view, get_range )
    
    # Visualization
    if view:
        _visualize_range(data_selected, range_values, columns=columns,
                         cmap=cmap, fig_size=fig_size)
    
    return range_values

def _visualize_range(data, range_values, columns=None, 
                     cmap='viridis', fig_size=None):
    """
    Visualizes the data distribution and highlights the range values.
    """
    plt.figure(figsize=fig_size if fig_size else (10, 6))
    if isinstance(data, pd.DataFrame):
        colors, alphas = get_colors_and_alphas( data.columns, cmap)
        for ii, col in enumerate(data.columns):
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
    as_frame: bool = False,
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
    as_frame = False if as_frame is ... else as_frame

    if isinstance(data, pd.DataFrame):
        data_selected = data[columns] if columns else data
        quartiles_result = data_selected.quantile([0.25, 0.5, 0.75], **kws)
    else:
        data_selected = np.asarray(data)
        quartiles_result = np.percentile(data_selected, [25, 50, 75], **kws)

    quartiles_result, view  = ensure_visualization_compatibility(
        quartiles_result, as_frame, view , func_name=quartiles )
    
    if view:
        _visualize_quartiles(
            data, plot_type=plot_type, cmap=cmap, fig_size=fig_size, 
            columns=columns)

    if as_frame and not isinstance(data, pd.DataFrame):
        quartiles_result = pd.DataFrame(
            quartiles_result, index=['25%', '50%', '75%']).T

    return quartiles_result 

def _visualize_quartiles(data, plot_type, cmap, fig_size, columns=None):
    """
    Visualizes quartiles using the specified plot type.
    """
    plt.figure(figsize=fig_size if fig_size else (10, 6))
    if plot_type == 'box':
        if isinstance(data, pd.DataFrame):
            if columns:
                data_to_plot = data[columns]
            else:
                data_to_plot = data
            data_to_plot.boxplot(column=columns, grid=False, color=cmap)
        else:
            plt.boxplot(data, patch_artist=True, 
                        boxprops=dict(facecolor=cmap))
    elif plot_type == 'hist':
        if isinstance(data, pd.DataFrame):
            for col in columns or data.columns:
                plt.hist(data[col], bins=30, alpha=0.5,
                         label=f'{col} Distribution', color=cmap)
        else:
            plt.hist(data, bins=30, alpha=0.5, color=cmap)
    plt.title('Data Distribution')
    plt.ylabel('Frequency' if plot_type == 'hist' else 'Value')
    plt.legend() if plot_type == 'hist' and isinstance(data, pd.DataFrame) else None
    plt.show()

@DynamicMethod (capture_columns=True)
def goquantile(
    data: Union[ArrayLike, DataFrame], 
    q: Union[float, List[float]], 
    columns: Optional[List[str]] = None, 
    as_frame: bool = False, 
    view: bool = False, 
    plot_type: Optional[str] = 'box', 
    cmap: str = 'viridis', 
    fig_size: Optional[Tuple[int, int]] = None, 
    **kws):
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
    >>> goquantile(data_array, q=0.5, view=True, plot_type='hist')

    >>> data_df = pd.DataFrame({'A': [2, 5, 7, 8], 'B': [1, 4, 6, 9]})
    >>> goquantile(data_df, q=[0.25, 0.75], as_frame=True, view=True, plot_type='box')
    
    Note
    ----
    The function provides a convenient way to compute quantiles, a critical 
    statistical measure for understanding the distribution of data. The 
    flexibility to compute quantiles for specific columns or entire datasets 
    enhances its utility in exploratory data analysis and data preprocessing.
    """
    if isinstance(data, pd.DataFrame) and columns is not None:
        data_selected = data[columns]
    else:
        data_selected = data
    
    # Compute the quantile or quantiles
    if isinstance(data_selected, pd.DataFrame):
        quantiles_result = data_selected.quantile(q, **kws)
    else:
        data_array = np.asarray(data_selected)
        quantiles_result = np.quantile(data_array, q, **kws)

    if as_frame:
        if not isinstance(quantiles_result, pd.DataFrame):
            quantiles_result = pd.DataFrame(
                quantiles_result, index=[f'{q*100}%' for q in np.atleast_1d(q)]).T

    if view:
        _visualize_quantiles(data_selected, q, quantiles_result,
                             plot_type=plot_type, cmap=cmap, 
                             fig_size=fig_size, columns=columns)

    return quantiles_result

def _visualize_quantiles(data, q, quantiles_result, plot_type, cmap, fig_size,
                         columns=None):
    """
    Visualizes the data and highlights the quantiles, based on the 
    specified plot type.
    """
    plt.figure(figsize=fig_size if fig_size else (10, 6))
    q_list = np.atleast_1d(q)
    if plot_type == 'box':
        if isinstance(data, pd.DataFrame):
            data.boxplot(column=columns, grid=False, color=cmap)
        else:
            plt.boxplot(data, patch_artist=True, 
                        boxprops=dict(facecolor=cmap))
        plt.title('Boxplot and Quantiles')
    elif plot_type == 'hist':
        if isinstance(data, pd.DataFrame):
            for col in columns or data.columns:
                plt.hist(data[col], bins=30, alpha=0.5,
                         label=f'{col} Distribution',
                         color=cmap)
                for quantile in q_list:
                    plt.axvline(data[col].quantile(quantile),
                                color='r', linestyle='--',
                                label=f'{quantile*100}% Quantile')
        else:
            plt.hist(data, bins=30, alpha=0.5, color=cmap)
            for quantile in q_list:
                plt.axvline(np.quantile(data, quantile), color='r',
                            linestyle='--', label=f'{quantile*100}% Quantile')
        plt.title('Histogram and Quantiles')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


@DynamicMethod ( 
    expected_type="both", 
    capture_columns=True
   )
def gocorr(
    data: Union[ArrayLike, DataFrame], /,  
    columns: List[str] = None,
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
    >>> >>> gocorr(data, view=True)

    >>> gocorr(data, columns=['A', 'C'], view=True, plot_type='heatmap',
    ...        cmap='coolwarm')
    Note
    ----
    The function utilizes pandas' `corr()` method to compute the correlation 
    matrix, offering flexibility through `**kws` to use different correlation 
    computation methods. For non-DataFrame inputs, the data is first converted 
    to a DataFrame, ensuring uniform processing.
    """
    if isinstance(data, (list, np.ndarray)):
        data = pd.DataFrame(
            data, columns=columns if columns else range(len(data[0])))
   
    correlation_matrix = data.corr(**kws)
    if view:
        plt.figure(figsize=fig_size)
        sns.heatmap(correlation_matrix, annot=True, cmap=cmap, fmt=".2f")
        plt.title("Correlation Matrix")
        plt.show()

    return correlation_matrix

def correlation(
    x: Union[str, ArrayLike], 
    y: Union[str, ArrayLike] = None, 
    data: Optional[pd.DataFrame] = None,
    view: bool = False, 
    plot_type=None, 
    cmap: str = 'viridis', 
    fig_size: Optional[Tuple[int, int]] = None, 
    **kws):
    """
    Computes the correlation between two datasets, or within a DataFrame. 
    This function allows for flexible input types, including direct array-like 
    inputs or specifying column names within a DataFrame.

    Parameters
    ----------
    x : str or ArrayLike
        The first dataset for correlation analysis. If `x` is a string, `data`
        must be supplied, and `x` should be a column name of `data`. Otherwise,
        `x` can be an array-like object (list, np.ndarray, or pd.Series).
    y : str or ArrayLike, optional
        The second dataset for correlation analysis. Similar to `x`, if `y` 
        is a string, `data`  must be supplied, and `y` should be a column 
        name of `data`. If omitted, and `x` is a DataFrame, calculates the 
        correlation matrix of `x`.
    data : pd.DataFrame, optional
        The DataFrame containing the `x` and/or `y` columns if `x` and/or `y` 
        are specified as column names.
    view : bool, default False
        If True, visualizes the correlation using the specified `plot_type`. 
        Visualization is supported for pairwise correlations (scatter plot) 
        and correlation matrices (heatmap).
    plot_type : str, optional, default 'scatter'
        Specifies the type of plot for visualization. Options:
        - 'scatter': Scatter plot for pairwise correlation.
        - None: No visualization is produced.
    cmap : str, default 'viridis'
        The colormap for the visualization plot. Applicable if `view` is True.
    fig_size : Tuple[int, int], optional
        Specifies the figure size for the visualization plot.
        Applicable if `view` is True.

    **kws : dict
        Additional keyword arguments passed to the correlation calculation 
        method (e.g., pandas DataFrame `corr` method), allowing customization 
        such as specifying the method ('pearson', 'kendall', 'spearman').

    Returns
    -------
    correlation_value : float or pd.DataFrame
        The correlation coefficient between `x` and `y` if both are provided 
        and array-like, or the correlation matrix of `data` or `x` 
        (if `x` is a DataFrame and `y` is None).

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
    Note
    ----
    The function is designed to provide a versatile interface for correlation 
    analysis, accommodating different types of input and supporting various 
    correlation methods through keyword arguments. It utilizes pandas' `corr()` 
    method for DataFrame inputs, enabling comprehensive correlation analysis within 
    tabular data.
    """
    if isinstance(x, str) or isinstance(y, str):
        if data is None:
            raise ValueError(
                "Data cannot be None when 'x' or 'y' is specified as a column name.")
        # Validate that data is a DataFrame
        is_frame(data, df_only=True, raise_exception=True)  
        
        x = data[x] if isinstance(x, str) else x
        y = data[y] if isinstance(y, str) else y

    if isinstance(x, pd.DataFrame) and y is None:
        correlation_matrix = x.corr(**kws)
        if view:
            plt.figure(figsize=fig_size)
            sns.heatmap(correlation_matrix, annot=True, cmap=cmap)
            plt.title("Correlation Matrix")
            plt.show()
        return correlation_matrix
    
    elif isinstance(x, (pd.Series, np.ndarray, list)) and isinstance(
            y, (pd.Series, np.ndarray, list)):
        x_series = pd.Series(x) if not isinstance(x, pd.Series) else x
        y_series = pd.Series(y) if not isinstance(y, pd.Series) else y
        correlation_value = x_series.corr(y_series, **kws)
        if view and plot_type == 'scatter':
            plt.figure(figsize=fig_size)
            plt.scatter(x_series, y_series, color=cmap)
            plt.title(f"Scatter Plot: Correlation = {correlation_value:.2f}")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.grid(True)
            plt.show()
        return correlation_value
    else:
        raise ValueError("Invalid input. x and y must be array-like objects"
                         " or column names in the provided DataFrame.")

@DynamicMethod ( 
    expected_type="both", 
    capture_columns=True, 
    drop_na=True
   )
def goiqr(
    data: Union[ArrayLike, pd.DataFrame], 
    columns: Optional[List[str]] = None,
    as_frame: bool = False,
    view: bool = False, 
    plot_type: Optional[str] = 'boxplot', 
    cmap: str = 'viridis', 
    fig_size: Optional[Tuple[int, int]] = (8, 6), 
    **kws):
    """
    Computes the interquartile range (IQR) for numeric data and optionally 
    visualizes it.

    Parameters
    ----------
    data : Union[ArrayLike, pd.DataFrame]
        The dataset from which the IQR is calculated. This can be directly 
        provided as a list, numpy array, or a pandas DataFrame. When provided 
        as a list or numpy array, the data is internally converted to a pandas 
        DataFrame for calculation.

    columns : Optional[List[str]] (default=None)
        Relevant when `data` is a pandas DataFrame. Specifies the columns within 
        the DataFrame for which the IQR is to be calculated. If None, the IQR 
        is calculated for all numeric columns in the DataFrame.

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
    >>> data_array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> print("IQR for array:", goiqr(data_array, view=True, plot_type='boxplot'))

    >>> data_df = pd.DataFrame({
        'A': np.random.normal(0, 1, 100),
        'B': np.random.normal(1, 2, 100),
        'C': np.random.normal(2, 3, 100)
    })
    >>> print("IQR for DataFrame:", goiqr(data_df, as_frame=True))
    >>> goiqr(data_df, view=True, plot_type='boxplot')
    
    Note
    ----
    The IQR is a robust measure of spread that is less influenced by outliers 
    than the range. This function simplifies the process of calculating the IQR, 
    especially useful in exploratory data analysis and for identifying potential 
    outliers.
    """

    # Handle ArrayLike data by converting it to DataFrame for uniform processing
    if isinstance(data, (list, np.ndarray)):
        data = pd.DataFrame(data, columns=columns if columns else range(len(data[0])))
    # elif isinstance(data, pd.DataFrame) and columns:
    #     data = data[columns]

    # Calculate IQR
    if as_frame and isinstance(data, pd.DataFrame):
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        iqr_values = Q3 - Q1
    else:
        Q1, Q3 = np.percentile(data, [25, 75], **kws)
        iqr_values = Q3 - Q1

    # Visualization
    if view:
        plt.figure(figsize=fig_size)
        if isinstance(data, pd.DataFrame):
            sns.boxplot(data=data, orient='h', palette=cmap)
        else:
            sns.boxplot(data=pd.DataFrame(data), orient='h', palette=cmap)
        plt.title('IQR Visualization with Boxplot')
        # Extend here with additional plot types if necessary
        plt.show()

    return iqr_values

@make_data_dynamic(capture_columns=True)
def z_scores(
    data: Union[ArrayLike, DataFrame], 
    columns: Optional[List[str]] = None,
    as_frame: bool = False,
    view: bool = False, 
    plot_type: Optional[str] = 'hist', 
    cmap: str = 'viridis', 
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
        
    as_frame : bool (default=False)
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
    >>> z_scores(data_df, as_frame=True, view=True, plot_type='boxplot')
    
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
    The Z-score is a powerful tool for data standardization, making it easier
    to compare measurements across different scales. Visualization options 
    provide quick insights into the distribution and variance of standardized data.
    """
    # Calculate Z-scores
    if isinstance(data, (list, np.ndarray)):
        data = np.array(data)
        mean = np.mean(data)
        std_dev = np.std(data)
        result = (data - mean) / std_dev
    elif isinstance(data, pd.DataFrame):
        # if columns:
        #     data = data[columns]
        mean = data.mean()
        std_dev = data.std()
        result = (
            data - mean) / std_dev if as_frame else (
                data - mean) / std_dev.to_numpy()

    # Visualization
    if view:
        plt.figure(figsize=fig_size)
        if plot_type == 'hist':
            if isinstance(result, pd.DataFrame):
                result.plot(kind='hist', alpha=0.5, bins=20, colormap=cmap, legend=True)
            else:
                plt.hist(result, bins=20, alpha=0.5, color=cmap)
            plt.title('Z-Scores Distribution')
            plt.xlabel('Z-Score')
            plt.ylabel('Frequency')
        elif plot_type == 'boxplot':
            if isinstance(result, pd.DataFrame):
                sns.boxplot(data=result, orient='h', palette=cmap)
            else:
                sns.boxplot(data=pd.DataFrame(result), orient='h', color=cmap)
            plt.title('Z-Scores Box Plot')
        elif plot_type == 'density':
            if isinstance(result, pd.DataFrame):
                result.plot(kind='density', colormap=cmap, legend=True)
            else:
                sns.kdeplot(result, color=cmap)
            plt.title('Z-Scores Density Plot')
            plt.xlabel('Z-Score')

        plt.show()

    return result


@make_data_dynamic(capture_columns=True)
def descriptive_stats(
    data: Union[ArrayLike, DataFrame],
    columns: Optional[List[str]] = None,
    include: Union[str, List[str]] = 'all',
    exclude: Union[str, List[str]] = None,
    as_frame: bool = False,
    view: bool = False, 
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
    as_frame : bool, default=False
        If True, the input array-like structure is converted to a Pandas DataFrame
        before calculating the descriptive statistics. This is useful if the input
        is a structured array or a sequence of sequences and you want the output
        in the form of a DataFrame.
    view : bool, optional
        If True, generates a visualization of the descriptive statistics 
        based on `plot_type`.  Default is False.
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
    >>> from gofast.stats.utils import descriptive_stats
    >>> import numpy as np
    >>> import pandas as pd
    >>> data = np.random.rand(100, 4)
    >>> descriptive_stats(data, as_frame=True)
    
    >>> import pandas as pd
    >>> df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D'])
    >>> descriptive_stats(df, columns=['A', 'B'])
    
    >>> df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D'])
    >>> descriptive_stats(df, columns=['A', 'B'], view=True, plot_type='hist')
    Note
    ----
    This function is a convenient wrapper around `pd.DataFrame.describe`,
    allowing for additional flexibility and ease of use. It's designed to
    work seamlessly with both array-like structures and Pandas DataFrames,
    making it a versatile tool for initial data exploration.
    """
    # Convert array-like input to DataFrame if necessary
    # if not isinstance(data, pd.DataFrame):
    #     if as_frame or view:
    #         data = pd.DataFrame(
    #             data, columns=columns if columns else range(len(data[0])))
    #     else:
    #         raise ValueError(
    #             "Data must be a pandas DataFrame for 'view' and 'as_frame' options.")
    
    # Select specified columns if provided
    # if columns is not None:
    #     data = data[columns]
    
    stats_result = data.describe(
        include=include, exclude=exclude, **kwargs)
    
    # Visualization
    if view:
        plt.figure(figsize=fig_size)
        if plot_type == 'box':
            sns.boxplot(data=data, orient='h', palette=cmap)
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
    
    return stats_result

@make_data_dynamic(capture_columns=True)
def calculate_skewness(
    data: Union[ArrayLike, pd.DataFrame],
    columns: Optional[List[str]] = None,
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
        Additional keyword arguments passed to `pd.Series.skew` or `pd.DataFrame.skew`.


    Returns
    -------
    float or pd.Series
        The skewness value(s) of the data. Returns a single float value if `data`
        is a single column or ArrayLike without `as_frame` set to True. Returns
        a pd.Series with skewness values for each column if `data` is a DataFrame
        or `as_frame` is True.

    Examples
    --------
    >>> from gofast.stats.utils import calculate_skewness
    >>> import numpy as np
    >>> data = np.random.normal(loc=0, scale=1, size=1000)
    >>> calculate_skewness(data)
    
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'A': np.random.normal(loc=0, scale=1, size=1000),
    ...     'B': np.random.normal(loc=1, scale=2, size=1000),
    ...     'C': np.random.lognormal(mean=0, sigma=1, size=1000)
    ... })
    >>> calculate_skewness(df)
    
    >>> data = [1, 2, 2, 3, 4, 7, 9]
    >>> calculate_skewness(data)
    0.782

    >>> df = pd.DataFrame({
    ...     'normal': np.random.normal(0, 1, 1000),
    ...     'right_skewed': np.random.exponential(1, 1000)
    ... })
    >>> calculate_skewness(df, view=True, plot_type='density')

    Note
    ----
    This function is useful for identifying the symmetry of data distributions,
    which can be critical for certain statistical analyses and modeling. Skewness
    can indicate the need for data transformation or the use of non-parametric
    statistical methods.
    """
    if isinstance(data, (list, np.ndarray)):
        data = pd.DataFrame(
            data, columns=columns if columns else range(len(data[0])))
    # elif isinstance(data, pd.DataFrame) and columns:
    #     data = data[columns]
    
    # Ensuring numeric data type for calculation
    data_numeric = data.apply(pd.to_numeric, errors='coerce')
    skewness_value = data_numeric.skew(axis=0, **kwargs)
    
    # Visualization
    if view:
        plt.figure(figsize=fig_size)
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
        plt.legend()
        plt.show()
    
    return skewness_value

@make_data_dynamic(capture_columns=True)
def calculate_kurtosis(
    data: Union[ArrayLike, DataFrame],
    columns: List[str] = None,
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

    Parameters
    ----------
    data : ArrayLike or pd.DataFrame
        The input data for which kurtosis is to be calculated.
        Can be a Pandas DataFrame
        or any array-like structure containing numerical data.
    columns : List[str], optional
        Specific columns to include in the analysis if `data` is a DataFrame.
        If None, all columns are included. Default is None.
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
    **kwargs : dict
        Additional keyword arguments to be passed to the `stats.kurtosis` function.

    Returns
    -------
    float or pd.Series
        The kurtosis value(s) of the data. Returns a single float value if `data`
        is a single column or ArrayLike without `as_frame` set to True. Returns
        a pd.Series with kurtosis values for each column if `data` is a DataFrame
        or `as_frame` is True.

    Examples
    --------
    >>> from gofast.stats.utils import calculate_kurtosis
    >>> import numpy as np
    >>> data = np.random.normal(0, 1, 1000)
    >>> calculate_kurtosis(data)
    
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'A': np.random.normal(0, 1, size=1000),
    ...     'B': np.random.standard_t(10, size=1000)
    ... })
    >>> calculate_kurtosis(df, as_frame=True)

    >>> data = np.random.normal(0, 1, 1000)
    >>> print(calculate_kurtosis(data))
    
    >>> df = pd.DataFrame({
    ...     'normal': np.random.normal(0, 1, 1000),
    ...     'leptokurtic': np.random.normal(0, 1, 1000) ** 3,
    ... })
    >>> print(calculate_kurtosis(df, as_frame=True))

    Note
    ----
    Kurtosis is useful for understanding the extremity of outliers or the
    propensity of data to produce outliers. A higher kurtosis can indicate
    a higher risk or potential for outlier values in the dataset.
    """
    # if  isinstance(data, pd.DataFrame):
    #     data = pd.DataFrame(data, columns=columns if columns else range(len(data[0])))
    # elif isinstance(data, pd.DataFrame) and columns:
    #     data = data[columns]

    kurtosis_value = stats.kurtosis(data, axis=0, **kwargs)
    
    if as_frame:
        kurtosis_value = pd.Series(
            kurtosis_value, index=columns or data.columns)

    if view:
        plt.figure(figsize=fig_size)
        if plot_type == 'density':
            for col in data.columns:
                sns.kdeplot(data[col], 
                            label=f'{col} kurtosis={kurtosis_value[col]:.2f}',
                            cmap=cmap)
        elif plot_type == 'hist':
            data.hist(bins=30, alpha=0.5, color=cmap, figsize=fig_size)
        plt.title("Data Distribution and Kurtosis")
        plt.legend()
        plt.show()

    return kurtosis_value

def t_test_independent(
    sample1: Union[List[float], List[int], str],
    sample2: Union[List[float], List[int], str],
    alpha: float = 0.05, 
    data: DataFrame = None, 
    as_frame: bool=False, 
    view: bool = False, 
    plot_type: str = 'box', 
    cmap: str = 'viridis', 
    fig_size: Optional[Tuple[int, int]] = (10, 6),  
    **kws
) -> Tuple[float, float, bool]:
    """
    Performs an independent two-sample t-test to compare the means of two
    independent samples. 
    
    Funtion provides the t-statistic, p-value, and whether the
    null hypothesis (that the samples have identical average values) 
    can be rejected at the specified significance level.

    Parameters
    ----------
    sample1 : array_like or str
        The first sample, must be a list, array of numeric values, or the name
        of a column in `data` if `data` is provided.
    sample2 : array_like or str
        The second sample, must be a list, array of numeric values, or the name
        of a column in `data` if `data` is provided.
    alpha : float, optional
        The significance level used to determine if the null hypothesis can be
        rejected, default is 0.05.
    data : pd.DataFrame, optional
        A DataFrame containing the data for `sample1` and `sample2` if column
        names are provided for these parameters. Required if `sample1` or
        `sample2` is a string.
    as_frame : bool, default=False
        If True and `t_stat`, `p_value` and  `reject_null` results are 
        in pandas Series. 
    view : bool, optional
        If True, generates a plot (boxplot or histogram) to visualize the 
        sample distributions.
    plot_type : str, optional
        Type of plot for visualization (
            'box' for boxplot, 'hist' for histogram). Default is 'box'.
    cmap : str, optional
        Color map for the plot. Default is 'viridis'.
    fig_size : Optional[Tuple[int, int]], optional
        Figure size for the plot. Default is (10, 6).
    **kwargs : dict
        Additional keyword arguments to be passed to the
        `stats.ttest_ind` function.
        
    Returns
    -------
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
    independent samples,especially when assessing the difference in means
    between two groups under different conditions or treatments. When providing
    string arguments for `sample1` and `sample2`, ensure a DataFrame is also 
    passed to the `data` parameter.
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
        plt.figure(figsize=fig_size)
        if plot_type == 'box':
            sns.boxplot(data=[sample1, sample2], palette=cmap)
            plt.title('Sample Distributions - Boxplot')
        elif plot_type == 'hist':
            sns.histplot(sample1, color=cmap, alpha=0.6, kde=True, 
                         label='Sample 1')
            sns.histplot(sample2, color=cmap, alpha=0.6, kde=True, 
                         label='Sample 2')
            plt.title('Sample Distributions - Histogram')
        plt.legend()
        plt.show()
    
    if as_frame: 
        return to_series_if(
            t_stat, p_value, reject_null, 
            ["T-statistic", " P-value","Reject Null" ],
            name ="t_test_independent")
    return t_stat, p_value, reject_null

def perform_linear_regression(
    x: Union[ArrayLike, list, str] = None, 
    y: Union[ArrayLike, list, str] = None, 
    data: DataFrame = None,
    view: bool = False, 
    plot_type: str = 'scatter_line', 
    cmap: str = 'viridis', 
    fig_size: Optional[Tuple[int, int]] = (10, 6), 
    **kwargs
) -> Tuple[LinearRegression, ArrayLike, float]:
    """
    Performs linear regression analysis between an independent variable (x) and 
    a dependent variable (y), returning the fitted model, its coefficients,
    and intercept.

    Parameters
    ----------
    x : str, list, or array-like
        Independent variable(s). If a string is provided, `data` must
        also be supplied, and `x` should refer to a column name within `data`.
    y : str, list, or array-like
        Dependent variable. Similar to `x`, if a string is provided, 
        it should refer to a column name within a supplied `data` DataFrame.
    data : pd.DataFrame, optional
        DataFrame containing the variables specified in `x` and `y`. 
        Required if `x` or `y` are specified as strings.
    view : bool, optional
        If True, generates a plot to visualize the data points and the
        regression line. Default is False.
    plot_type : str, optional
        Type of plot for visualization. Currently supports 'scatter_line'.
        Default is 'scatter_line'.
    cmap : str, optional
        Color map for the plot. Default is 'viridis'.
    fig_size : Optional[Tuple[int, int]], optional
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
    
    Examples
    --------
    >>> import numpy as np
    >>> from gofast.stats.utils import perform_linear_regression
    >>> x = np.random.rand(100)
    >>> y = 2.5 * x + np.random.normal(0, 0.5, 100)
    >>> model, coefficients, intercept = perform_linear_regression(x, y)
    >>> print(f"Coefficients: {coefficients}, Intercept: {intercept}")

    Using a DataFrame:
    >>> import pandas as pd
    >>> df = pd.DataFrame({'X': np.random.rand(100), 
                           'Y': 2.5 * np.random.rand(100) + np.random.normal(0, 0.5, 100)})
    >>> model, coefficients, intercept = perform_linear_regression('X', 'Y', data=df)
    >>> print(f"Coefficients: {coefficients}, Intercept: {intercept}")

    Note
    ----
    This function streamlines the process of performing linear regression analysis,
    making it straightforward to model relationships between two variables and 
    extract useful statistics such as the regression coefficients and intercept.
    """
    x_values, y_values = assert_xy_in(x, y, data=data, xy_numeric= True )
 
    model = LinearRegression(**kwargs)
    model.fit(x_values, y_values)
    coefficients = model.coef_
    intercept = model.intercept_

    if view:
        plt.figure(figsize=fig_size)
        plt.scatter(x_values, y_values, color='blue',
                    label='Data Points')
        plt.plot(x_values, model.predict(x_values), color='red',
                 label='Regression Line')
        plt.title('Linear Regression Analysis')
        plt.xlabel('Independent Variable')
        plt.ylabel('Dependent Variable')
        plt.legend()
        plt.show()
        
    return model, coefficients, intercept

@make_data_dynamic('numeric', capture_columns=True)
def chi_squared_test(
    data: Union[ArrayLike, pd.DataFrame],
    alpha: float = 0.05, 
    columns: List[str] = None,
    view: bool = False,
    plot_type=None, 
    cmap: str = 'viridis',
    fig_size: Tuple[int, int] = (12, 5),
    **kwargs
) -> Tuple[float, float, bool]:
    """
    Performs a Chi-Squared test for independence to assess the relationship 
    between two categorical variables represented in a contingency table.

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

    view : bool, optional
        If True, displays a heatmap of the contingency table, default is False.
    cmap : str, optional
        Colormap for the heatmap, default is 'viridis'.
    fig_size : Tuple[int, int], optional
        Figure size for the heatmap, default is (12, 5).

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
    Using a DataFrame directly:
    
    >>> import numpy as np
    >>> import pandas as pd
    >>> from gofast.stats.utils import perform_chi_squared_test
    >>> data = pd.DataFrame({'A': [10, 20, 30], 'B': [20, 15, 30]})
    >>> chi2_stat, p_value, reject_null = perform_chi_squared_test(data)
    >>> print(f"Chi2 Statistic: {chi2_stat}, P-value: {p_value}, Reject Null: {reject_null}")

    Using array_like with column names:
    >>> data = np.array([[10, 20, 30], [20, 15, 30]])
    >>> columns = ['A', 'B']
    >>> chi2_stat, p_value, reject_null = perform_chi_squared_test(
        data, columns=columns, as_frame=True)
    >>> print(f"Chi2 Statistic: {chi2_stat}, P-value: {p_value}, Reject Null: {reject_null}")

    Note
    ----
    The Chi-Squared test is a statistical method to determine if there is a 
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
        
    return chi2_stat, p_value, reject_null

@DynamicMethod( 
   'categorical',
    capture_columns=False, 
    treat_int_as_categorical=True, 
    encode_categories= True
  )
def anova_test(
    data: Union[Dict[str, List[float]], np.ndarray, pd.DataFrame], 
    groups: Optional[Union[List[str], np.ndarray]]=None, 
    alpha: float = 0.05,
    view: bool = False,
    cmap: str = 'viridis',
    fig_size: Tuple[int, int] = (12, 5)
) -> Tuple[float, float, bool]:
    """
    Perform an ANOVA test to compare means across multiple groups.

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
    alpha : float, optional
        The significance level for the ANOVA test. Default is 0.05.
    view : bool, optional
        If True, generates a box plot of the group distributions.
        Default is False.
    cmap : str, optional
        The colormap for the box plot. This parameter is currently not used 
        as seaborn's boxplot does not support colormap directly but kept 
        for future compatibility. Default is 'viridis'.
    fig_size : Tuple[int, int], optional
        Figure size for the box plot. Default is (12, 5).

    Returns
    -------
    f_stat : float
        The calculated F-statistic.
    p_value : float
        The p-value from the ANOVA test.
    reject_null : bool
        Indicates whether the null hypothesis can be rejected based on the alpha level.

    Examples
    --------
    >>> from gofast.stats.utils import anova_test
    >>> data = {'group1': [1, 2, 3], 'group2': [4, 5, 6], 'group3': [7, 8, 9]}
    >>> f_stat, p_value, reject_null = anova_test(data, alpha=0.05, view=True)
    >>> print(f"F-statistic: {f_stat}, P-value: {p_value}, Reject Null: {reject_null}")
    """
    # Decorator, handle this part. 
    # if isinstance(data, dict):
    #     groups_data = [data[group] for group in groups] 
    #                     if groups else list(data.values())
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

    return f_stat, p_value, reject_null

@make_data_dynamic(capture_columns=True, dynamize= False )
def perform_kmeans_clustering(
    data: Union[np.ndarray, pd.DataFrame],
    n_clusters: int = 3,
    columns: list = None,
    view: bool = True,
    cmap='viridis', 
    fig_size: Tuple[int, int] = (10, 6),
    **kwargs
) -> Tuple[KMeans, np.ndarray]:
    """
    Applies K-Means clustering to the dataset, returning the fitted model and 
    cluster labels for each data point.
    
    Parameters
    ----------
    data : array_like or pd.DataFrame
        Multidimensional dataset for clustering. Can be a pandas DataFrame 
        or a 2D numpy array. If a DataFrame and `columns` is specified, only
        the selected columns are used for clustering.
    n_clusters : int, optional
        Number of clusters to form. Default is 3.
    columns : list, optional
        Specific columns to use for clustering if `data` is a DataFrame. 
        Ignored if `data` is an array_like. Default is None.
    as_frame : bool, optional
        If True and `data` is array_like, converts `data` to a DataFrame 
        before clustering. Requires `columns` to specify column names. 
        Default is False.
    visualize : bool, optional
        If True, generates a scatter plot of the clusters with centroids.
        Default is True.
    cmap : str, optional
        Colormap for the heatmap, default is 'viridis'.
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
        df, columns=['feature1', 'feature2'], n_clusters=3)
    >>> print(labels)
    """
    if isinstance(data, pd.DataFrame) and columns is not None:
        data_for_clustering = data[columns]
    else:
        data_for_clustering = data

    km = KMeans(n_clusters=n_clusters, **kwargs)
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
        plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5, marker='x')
        plt.title('K-Means Clustering')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()

    return km, labels

@make_data_dynamic('numeric', capture_columns=True)
def harmonic_mean(
    data: Union[pd.DataFrame, np.ndarray],
    columns: List[str] = None,
    as_frame: bool = False,
    view: bool = True,
    cmap: str = 'viridis',
    fig_size: Tuple[int, int] = (10, 6)
):
    """
    Calculate the harmonic mean of a data set and optionally visualize the 
    data distribution through a histogram.

    The harmonic mean is the reciprocal of the arithmetic mean of the reciprocals
    of the data points. It is particularly useful for rates and ratios, offering
    a measure of central tendency that is less affected by large outliers compared
    to the arithmetic mean.

    Parameters
    ----------
    data : array_like
        An array, any object exposing the array interface, containing
        data for which the harmonic mean is desired. Values must be greater
        than 0.If a DataFrame is provided and `columns` is specified, the 
        calculation is limited to the selected columns.
    data : DataFrame or ArrayLike
        The data for which the harmonic mean is desired. 
    columns : List[str], optional
        Specific columns to use for the calculation if `data` is a DataFrame.
    as_frame : bool, optional
        If True and `data` is ArrayLike, converts `data` into a DataFrame
        using `columns` as column names. Ignored if `data` is already a DataFrame.
    view : bool, optional
        If True, a histogram of the data set is displayed to visualize its
        distribution. Default is True.
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
    >>> from gofast.stats.utils import harmonic_mean
    >>> harmonic_mean([1, 2, 4])
    1.7142857142857142

    >>> harmonic_mean([1, 0, 2])
    ValueError: Data points must be greater than 0.

    >>> harmonic_mean(np.array([2.5, 3.0, 10.0]))
    3.5294117647058822
    
    >>> df = pd.DataFrame({'A': [2.5, 3.0, 10.0], 'B': [1.5, 2.0, 8.0]})
    >>> harmonic_mean(df, columns=['A'])
    3.5294117647058822
    """
    if isinstance(data, pd.DataFrame):
        data_values = data.to_numpy().flatten()
    else:
        data_values = np.asarray(data).flatten()

    if np.any(data_values <= 0):
        raise ValueError("Data points must be greater than 0.")

    h_mean = len(data_values) / np.sum(1.0 / data_values)

    if view:
        plt.figure(figsize=fig_size)
        plt.hist(data_values, bins='auto', color=cmap, alpha=0.7, rwidth=0.85)
        plt.title('Data Distribution')
        plt.xlabel('Data Points')
        plt.ylabel('Frequency')
        plt.axvline(h_mean, color='red', linestyle='dashed', linewidth=2)
        plt.text(h_mean, plt.ylim()[1] * 0.9, f'Harmonic Mean: {h_mean:.2f}',
                 rotation=90, verticalalignment='center')
        plt.show()

    return h_mean

@make_data_dynamic(capture_columns=True)
def weighted_median(
    data: Union[pd.DataFrame, np.ndarray],
    weights: Union[str, int, np.ndarray, list],
    columns: Union[str, list] = None,
    view: bool = True,
    cmap: str = 'viridis',
    fig_size: Tuple[int, int] = (10, 6)
):
    """
    Compute the weighted median of a data set, with an option to visualize the
    distribution of data points and their weights.

    The weighted median is determined such that the sum of weights is equal on both
    sides of the median in the sorted list of data points.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Data for which the weighted median is desired. If a DataFrame and `columns`
        is specified, calculation uses the selected column.
    weights : str, int, np.ndarray, or list
        Weights for each element in `data`. If a string or int, it specifies the column
        in `data` DataFrame containing the weights.
    columns : str or list, optional
        Specific column(s) to use for the calculation if `data` is a DataFrame.
    view : bool, optional
        If True, displays a scatter plot of the data points with sizes 
        proportional to their weights.
    cmap : str, optional
        Colormap for the scatter plot.
    fig_size : Tuple[int, int], optional
        Size of the figure for the scatter plot.

    Returns
    -------
    w_median : float
        The weighted median of the data set.

    Example
    -------
    >>> import numpy as np 
    >>> from gofast.stats.utils import weighted_median
    >>> weighted_median(np.array([1, 2, 3]), np.array([3, 1, 2]))
    2.0
    """
    if isinstance(data, pd.DataFrame):
        if isinstance(weights, str) or isinstance(weights, int):
            weights = data[weights].values
    else:
        data = np.asarray(data)
    weights = np.asarray(weights)

    sorted_indices = np.argsort(data)
    sorted_data, sorted_weights = data[sorted_indices], weights[sorted_indices]
    cumulative_weights = np.cumsum(sorted_weights)
    median_idx = np.where(cumulative_weights >= 0.5 * np.sum(sorted_weights))[0][0]
    w_median = sorted_data[median_idx]

    if view:
        plt.figure(figsize=fig_size)
        plt.scatter(sorted_data, np.zeros_like(sorted_data), 
                    s=sorted_weights * 100, c=sorted_data, 
                    cmap=cmap, alpha=0.6)
        plt.colorbar(label='Data Value')
        plt.axvline(w_median, color='red', linestyle='dashed', 
                    linewidth=2, label=f'Weighted Median: {w_median}')
        plt.title('Weighted Median Visualization')
        plt.xlabel('Data Points')
        plt.yticks([])
        plt.legend()
        plt.show()

    return w_median

@make_data_dynamic("numeric", capture_columns=True)
def bootstrap(
    data: Union[pd.DataFrame, np.ndarray],
    n: int = 1000,
    columns: Optional[List[str]] = None,
    func: _F = np.mean,
    view: bool = True,
    alpha=.7, 
    cmap: str = 'viridis',
    fig_size: Tuple[int, int] = (10, 6)
):
    """
    Perform bootstrapping to estimate the distribution of a statistic.

    Bootstrapping is a resampling technique used to estimate statistics on a
    population by sampling a dataset with replacement. This method allows for
    the estimation of the sampling distribution of almost any statistic.

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
    view : bool, optional
        If True, displays a histogram of the bootstrapped statistics.
    cmap : str, optional
        Colormap for the histogram.
    fig_size : Tuple[int, int], optional
        Size of the figure for the histogram.

    Returns
    -------
    bootstrapped_stats : ndarray
        Array of bootstrapped statistic values.

    Examples
    --------
    >>> from gofast.stats.utils import bootstrap
    >>> np.random.seed(0)
    >>> data = np.arange(10)
    >>> stats = bootstrap(data, n=100, func=np.mean)
    >>> print(stats[:5])

    Using a DataFrame:
    >>> df = pd.DataFrame({'A': np.random.rand(100), 'B': np.random.rand(100)})
    >>> stats = bootstrap(df, n=1000, func=np.median, columns=['A'], view=True)
    """
    if isinstance(data, pd.DataFrame) and columns is not None:
        data = data[columns].to_numpy().flatten()
    elif isinstance(data, pd.DataFrame):
        data = data.to_numpy().flatten()
    else:
        data = np.asarray(data)

    bootstrapped_stats = [
        func(np.random.choice(data, size=len(data), replace=True)) for _ in range(n)]

    if view:
        plt.figure(figsize=fig_size)
        plt.hist(bootstrapped_stats, bins='auto', color=cmap,
                 alpha=alpha, rwidth=0.85)
        plt.title('Distribution of Bootstrapped Statistics')
        plt.xlabel('Statistic Value')
        plt.ylabel('Frequency')
        plt.show()

    return np.array(bootstrapped_stats)

@ensure_pkg(
    "lifelines",
    "The 'lifelines' package is required for this function to run.")
@make_data_dynamic("numeric", capture_columns=True, dynamize=False)
def kaplan_meier_analysis(
    durations: DataFrame | np.ndarray,
    event_observed: np.ndarray,
    columns=None, 
    view: bool = True,
    fig_size: Tuple[int, int] = (10, 6),
    **kws
):
    """
    Perform Kaplan-Meier Survival Analysis and optionally visualize the 
    survival function.

    Kaplan-Meier Survival Analysis is used to estimate the survival function
    from lifetime data. It is a non-parametric statistic used to estimate the
    survival probability from observed lifetimes.

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
    if isinstance(durations, pd.DataFrame):
        # Ensure it's a Series if only one column
        durations = durations[columns].squeeze()  

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
    
    return kmf

@make_data_dynamic(capture_columns=True, dynamize=False)
def gini_coeffs(
    data: Union[pd.DataFrame, np.ndarray],
    columns: Optional[Union[str, list]] = None,
    view: bool = False,
    cmap: str = 'viridis',
    fig_size: Tuple[int, int] = (10, 6),
    **kws
):
    """
    Calculate the Gini coefficient of a dataset and optionally visualize
    the Lorenz curve.

    The Gini coefficient is a measure of inequality of a distribution,
    ranging from 0 (perfect equality) to 1 (perfect inequality).

    Parameters
    ----------
    data : DataFrame or array_like
        Data set for which to calculate the Gini coefficient. If a DataFrame
        is provided and `columns` is specified, only the selected columns are used.
    columns : str or list, optional
        Specific column(s) to use if `data` is a DataFrame.
    view : bool, optional
        If True, displays the Lorenz curve of the data set.
    cmap : str, optional
        Colormap for the Lorenz curve plot. This parameter is currently unused but
        included for future compatibility.
    fig_size : Tuple[int, int], optional
        Size of the figure for the Lorenz curve plot.
    **kws : dict
        Additional keyword arguments, unused but included for compatibility.

    Returns
    -------
    gini : float
        The Gini coefficient of the data set.

    Examples
    --------
    >>> get_gini_coeffs([1, 2, 3, 4, 5])
    0.26666666666666666

    >>> df = pd.DataFrame({'income': [1, 2, 3, 4, 5]})
    >>> get_gini_coeffs(df, columns='income', view=True)
    """
    if isinstance(data, pd.DataFrame):
        if columns is not None:
            data = data[columns].squeeze()  ## Ensure it's a Series or single column DataFrame
        else:
            raise ValueError("Column name must be provided for DataFrame input.")
    data = np.sort(np.array(data))
    n = data.size
    index = np.arange(1, n + 1)
    gini = (np.sum((2 * index - n - 1) * data)) / (n * np.sum(data))

    if view:
        plt.figure(figsize=fig_size)
        lorenz_curve = np.cumsum(np.sort(data)) / np.sum(data)
        lorenz_curve = np.insert(lorenz_curve, 0, 0)  # Start at 0
        plt.plot(np.linspace(0.0, 1.0, lorenz_curve.size), lorenz_curve,
                 label='Lorenz Curve', color='blue')
        plt.plot([0, 1], [0, 1], label='Line of Equality', 
                 linestyle='--', color='red')
        plt.title('Lorenz Curve')
        plt.xlabel('Cumulative share of population')
        plt.ylabel('Cumulative share of wealth')
        plt.legend()
        plt.grid(True)
        plt.show()

    return gini

@make_data_dynamic(capture_columns=True, dynamize=False)
def mds_similarity(
    data: Union[pd.DataFrame, np.ndarray],
    n_components: int = 2,
    columns: Optional[list] = None,
    view: bool = False,
    cmap: str = 'viridis',
    fig_size: Optional[tuple] = (10, 6),
    **kws
):
    """
    Perform Multidimensional Scaling (MDS) to visualize the level of similarity
    of individual cases of a dataset in a lower-dimensional space.

    Parameters
    ----------
    data : DataFrame or ArrayLike
        Data set for MDS. If a DataFrame and `columns` is specified,
        only the selected columns are used.
    n_components : int, optional
        Number of dimensions in which to immerse the dissimilarities, 
        default is 2.
    columns : list, optional
        Specific columns to use if `data` is a DataFrame.
    view : bool, optional
        If True, displays a scatter plot of the MDS results.
    cmap : str, optional
        Colormap for the scatter plot.
    fig_size : tuple, optional
        Size of the figure for the scatter plot, default is (10, 6).
    **kws : dict
        Additional keyword arguments passed to `sklearn.manifold.MDS`.

    Returns
    -------
    mds_result : ndarray
        Coordinates of the data in the MDS space.

    Examples
    --------
    >>> import gofast.stats.utils import mds_similarity
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> mds_coordinates = mds_similarity(iris.data, n_components=2, view=True)
    >>> print(mds_coordinates.shape)
    (150, 2)
    """
    if isinstance(data, pd.DataFrame) and columns is not None:
        data = data[columns]
    
    mds = MDS(n_components=n_components, **kws)
    mds_result = mds.fit_transform(data)
    
    if view:
        plt.figure(figsize=fig_size)
        plt.scatter(mds_result[:, 0], mds_result[:, 1], cmap=cmap)
        plt.title('Multidimensional Scaling (MDS) Results')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.colorbar(label='Data Value')
        plt.show()
    
    return mds_result

@ensure_pkg ("skbio","The 'skbio' package is required for `dca_analysis` to run." )
@make_data_dynamic(capture_columns=True, dynamize=False)
def dca_analysis(
    data: Union[DataFrame, ArrayLike],
    columns: Optional[list] = None,
    view: bool = False,
    cmap: str = 'viridis',
    fig_size: Optional[Tuple[int, int]] = (10, 6),
    **kws
):
    """
    Perform Detrended Correspondence Analysis (DCA) on ecological data to identify
    the main gradients. Optionally, visualize the species scores in the DCA space.

    Parameters
    ----------
    data : DataFrame or ArrayLike
        Ecological data set for DCA, typically species occurrence or abundance.
        If a DataFrame and `columns` is specified, only the selected columns are used.
    columns : list, optional
        Specific columns to use if `data` is a DataFrame.
    view : bool, optional
        If True, displays a scatter plot of species scores in the DCA space.
    cmap : str, optional
        Colormap for the scatter plot.
    fig_size : tuple, optional
        Size of the figure for the scatter plot.
    **kws : dict
        Additional keyword arguments passed to `detrended_correspondence_analysis`.

    Returns
    -------
    dca_result : OrdinationResults
        Results of DCA, including axis scores and explained variance.

    Examples
    --------
    >>> from gofast.stats.utils import dca_analysis 
    >>> import pandas as pd
    >>> data = pd.DataFrame({'species1': [1, 0, 3], 'species2': [0, 4, 1]})
    >>> dca_result = dca_analysis(data, view=True)
    >>> print(dca_result.axes)
    """

    from skbio.stats.ordination import detrended_correspondence_analysis
    dca_result = detrended_correspondence_analysis(data, **kws)
    
    if view:
        species_scores = dca_result.samples
        plt.figure(figsize=fig_size)
        plt.scatter(species_scores.iloc[:, 0], species_scores.iloc[:, 1], cmap=cmap)
        plt.title('DCA Species Scores')
        plt.xlabel('DCA Axis 1')
        plt.ylabel('DCA Axis 2')
        plt.colorbar(label='Species Abundance')
        plt.show()
    
    return dca_result

@make_data_dynamic(capture_columns=True, dynamize=False)
def spectral_clustering(
    data: Union[DataFrame, ArrayLike],
    n_clusters: int = 2,
    assign_labels: str = 'discretize',
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

    Spectral Clustering leverages the eigenvalues of a similarity matrix to reduce
    dimensionality before applying a conventional clustering technique. It's
    well-suited for identifying non-convex clusters.

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
        initialization [3]_.
        The cluster_qr method [5]_ directly extract clusters from eigenvectors
        in spectral clustering. In contrast to k-means and discretization, cluster_qr
        has no tuning parameters and runs no iterations, yet may outperform
        k-means and discretization in terms of both quality and speed.
        
    random_state : int, RandomState instance, default=None
        A pseudo random number generator used for the initialization
        of the lobpcg eigenvectors decomposition when `eigen_solver ==
        'amg'`, and for the K-Means initialization. Use an int to make
        the results deterministic across calls.

        .. note::
            When using `eigen_solver == 'amg'`,
            it is necessary to also fix the global numpy seed with
            `np.random.seed(int)` to get deterministic results. See
            https://github.com/pyamg/pyamg/issues/139 for further
            information.
            
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

    Returns
    -------
    labels : ndarray
        Labels of each data point.

    Examples
    --------
    >>> from sklearn.datasets import make_moons
    >>> from gofast.stats.utils import spectral_clustering
    >>> X, _ = make_moons(n_samples=200, noise=0.05, random_state=0)
    >>> labels = spectral_clustering(X, n_clusters=2, view=True)
    
    Using a DataFrame:
    >>> df = pd.DataFrame(X, columns=['feature1', 'feature2'])
    >>> labels = spectral_clustering(df, n_clusters=2, columns=['feature1', 'feature2'], view=True)
    """
    clustering = SpectralClustering(
        n_clusters=n_clusters, 
        assign_labels=assign_labels,
        random_state=random_state, 
        **kws)
    labels = clustering.fit_predict(data)

    if view:
        plt.figure(figsize=fig_size if fig_size else (10, 6))
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap=cmap, alpha=0.6)
        plt.title('Spectral Clustering Results')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.colorbar(label='Cluster Label')
        plt.show()
    
    return labels

def levene_test(
    *samples: Union[List[np.ndarray], pd.DataFrame], 
    columns: Optional[List[str]] = None,
    center: str = 'median', 
    proportiontocut: float = 0.05, 
    view: bool = False, 
    cmap: str = 'viridis', 
    fig_size: Optional[Tuple[int, int]] = None,
    **kws
):
    """
    Perform Levene's test for equal variances across multiple samples, with an
    option to visualize the sample distributions.

    This function adapts to input data structures. If a DataFrame and column
    names are provided, it extracts the specified columns to compose samples.

    Parameters
    ----------
    *samples : Union[List[np.ndarray], pd.DataFrame]
        The sample data, possibly with different lengths. If a DataFrame and
        `columns` are specified, extracts data for each column name to compose samples.
    columns : List[str], optional
        Column names to extract from the DataFrame to compose samples.
    center : str, optional
        Specifies which measure of central tendency ('median', 'mean', or 'trimmed')
        to use in computing the test statistic.
    proportiontocut : float, optional
        Proportion of data to cut from each end when center is 'trimmed'.
    view : bool, optional
        If True, displays a boxplot of the samples.
    cmap : str, optional
        Colormap for the boxplot. Currently unused but included for compatibility.
    fig_size : Tuple[int, int], optional
        Size of the figure for the boxplot.
    **kws : dict
        Additional keyword arguments for `stats.levene`.

    Returns
    -------
    statistic : float
        The test statistic for Levene's test.
    p_value : float
        The p-value for the test.

    Examples
    -------
    >>> import numpy as np 
    >>> import pandas as pd 
    >>> from gofast.stats.utils import levene_test
    >>> sample1 = np.random.normal(loc=0, scale=1, size=50)
    >>> sample2 = np.random.normal(loc=0.5, scale=1.5, size=50)
    >>> sample3 = np.random.normal(loc=-0.5, scale=0.5, size=50)
    >>> statistic, p_value = levene_test(sample1, sample2, sample3, view=True)
    >>> print(f"Statistic: {statistic}, p-value: {p_value}")
    
    >>> df = pd.DataFrame({'A': np.random.normal(0, 1, 50),
    ...                    'B': np.random.normal(0, 2, 50),
    ...                    'C': np.random.normal(0, 1.5, 50)})
    >>> statistic, p_value = levene_test(df, columns=['A', 'B', 'C'], view=True)
    """
    # Check if *samples contains a single DataFrame and columns are specified
    samples = process_and_extract_data(
        *samples, columns =columns, allow_split= True ) 
    statistic, p_value = stats.levene(
        *samples, center=center, proportiontocut=proportiontocut, **kws)

    if view:
        plt.figure(figsize=fig_size if fig_size else (10, 6))
        plt.boxplot(samples, patch_artist=True, vert=True)
        plt.xticks(ticks=np.arange(1, len(columns) + 1),
                   labels=columns if columns else range(1, len(samples) + 1))
        plt.title('Sample Distributions - Levene’s Test for Equal Variances')
        plt.ylabel('Values')
        plt.xlabel('Samples/Groups')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    return statistic, p_value

def kolmogorov_smirnov_test(
    data1: Union[ArrayLike, str],
    data2: Union[ArrayLike, str],  
    alternative: str = 'two-sided',
    data: Optional[pd.DataFrame] = None, 
    method: str = 'auto', 
    view: bool = False, 
    cmap: str = 'viridis', 
    fig_size: Optional[Tuple[int, int]] = None,
):
    """
    Perform the Kolmogorov-Smirnov test for goodness of fit and optionally 
    visualize the distribution comparison.

    This test compares the distributions of two independent samples.

    Parameters
    ----------
    data1, data2 : ArrayLike or str
        Two arrays of sample observations assumed to be drawn from a 
        continuous distribution,or column names in the provided DataFrame `data`.
    alternative : str, optional
        Defines the alternative hypothesis ('two-sided', 'less', or 'greater').
    data : pd.DataFrame, optional
        DataFrame containing the columns referred to by `data1` and `data2` 
        if they are strings.
    method : str, optional
        Method used to compute the p-value ('auto', 'exact', or 'approx').
    view : bool, optional
        If True, visualizes the cumulative distribution functions of both samples.
    cmap : str, optional
        Colormap for the visualization.
    fig_size : Tuple[int, int], optional
        Size of the figure for the visualization.

    Returns
    -------
    statistic : float
        KS statistic.
    p_value : float
        Two-tailed p-value.

    Examples
    --------
    >>> from gofast.stats.utils import kolmogorov_smirnov_test
    >>> kolmogorov_smirnov_test([1, 2, 3, 4], [1, 2, 3, 5])
    (0.25, 0.9999999999999999)

    Using DataFrame columns:
    >>> df = pd.DataFrame({'group1': np.random.normal(0, 1, 100),
                           'group2': np.random.normal(0.5, 1, 100)})
    >>> kolmogorov_smirnov_test('group1', 'group2', data=df, view=True)
    """

    data1, data2 = assert_xy_in(data1, data2 , data=data, xy_numeric= True )
    statistic, p_value = stats.ks_2samp(
        data1, data2, alternative=alternative, method=method)

    if view:
        plt.figure(figsize=fig_size if fig_size else (10, 6))
        for sample, label in zip([data1, data2], ['Data1', 'Data2']):
            ecdf = lambda x: np.arange(1, len(x) + 1) / float(len(x))
            plt.step(sorted(sample), ecdf(sample), label=f'CDF of {label}')
        plt.title('CDF Comparison')
        plt.xlabel('Value')
        plt.ylabel('Cumulative Probability')
        plt.legend()
        plt.grid(True)
        plt.show()

    return statistic, p_value


def cronbach_alpha(
    items_scores: Union[ArrayLike, pd.DataFrame],
    columns: Optional[list] = None,
    view: bool = False,
    cmap: str = 'viridis',
    fig_size: Optional[Tuple[int, int]] = None
) -> float:
    """
    Calculate Cronbach's Alpha for a set of test items to assess internal 
    consistency.
    That is, how closely  related a set of items are as a group.

    Parameters
    ----------
    items_scores : Union[ArrayLike, pd.DataFrame]
        A 2D array or DataFrame where rows represent items and columns 
        represent scoring for each item.
    columns : Optional[list], default=None
        Specific columns to use if `items_scores` is a DataFrame. If None, 
        all columns are used.
    view : bool, default=False
        If True, displays a bar plot showing the variance of each item.
    cmap : str, default='viridis'
        Colormap for the bar plot. This parameter is currently unused but 
        included for future compatibility.
    fig_size : Optional[Tuple[int, int]], default=None
        Size of the figure for the bar plot. If None, defaults to matplotlib's default.

    Returns
    -------
    alpha : float
        Cronbach's Alpha, a measure of internal consistency.

    Examples
    --------
    Using a numpy array:
    >>> scores = np.array([[2, 3, 4], [4, 4, 5], [3, 5, 4]])
    >>> print(cronbach_alpha(scores))
    
    Using a pandas DataFrame:
    >>> df_scores = pd.DataFrame({'item1': [2, 4, 3], 'item2': [3, 4, 5], 'item3': [4, 5, 4]})
    >>> print(cronbach_alpha(df_scores))

    Visualizing item variances:
    >>> cronbach_alpha(df_scores, view=True)
    """
    if isinstance(items_scores, pd.DataFrame):
        if columns is not None:
            items_scores = items_scores[columns]
        else:
            items_scores = items_scores.values
    else:
        items_scores = np.asarray(items_scores)

    item_variances = items_scores.var(axis=0, ddof=1)
    total_variance = items_scores.sum(axis=1).var(ddof=1)
    n_items = items_scores.shape[1]

    alpha = (n_items / (n_items - 1)) * (1 - item_variances.sum() / total_variance)

    if view:
        plt.figure(figsize=fig_size if fig_size else (10, 6))
        plt.bar(range(n_items), item_variances, color=cmap)
        plt.title("Item Variances")
        plt.xlabel("Item")
        plt.ylabel("Variance")
        plt.xticks(ticks=range(n_items), 
                   labels=columns if columns else range(1, n_items + 1))
        plt.show()

    return alpha

def friedman_test(
    *samples: ArrayLike, 
    columns: Optional[List[str]] = None, 
    method: str = 'auto',  
    view: bool = False,
    cmap: str = 'viridis',
    fig_size: Optional[Tuple[int, int]] = None
):
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
    columns: List[str]] 
        if columns *args must contain single dataframe where columnames can be 
        extracted as array for operations. 
        
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
    
    Example with DataFrame input
    >>> df = pd.DataFrame({
        'group1': group1,
        'group2': group2,
        'group3': group3
     })
    >>> statistic, p_value = friedman_test(df, columns=['group1', 'group2', 'group3'], view=True)
    print(f'Friedman statistic with DataFrame input: {statistic}, p-value: {p_value}')
    
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
    if len(set(map(len, samples))) != 1:
        raise ValueError("All input arrays must have the same length.")

    # Data extraction if a DataFrame is provided in samples
    if len(samples) == 1 and isinstance(
            samples[0], pd.DataFrame) and columns is not None:
        df = samples[0]
        samples = [df[col].values for col in columns]
    
    # Perform the Friedman test
    statistic, p_value = stats.friedmanchisquare(*samples, method=method)
    
    if view:
        # Visualization logic
        plt.figure(figsize=fig_size if fig_size else (10, 6))
        data_for_plot = np.vstack(samples)
        plt.boxplot(data_for_plot.T, patch_artist=True, 
                    labels=columns if columns 
                    else range(1, len(samples) + 1))
        plt.title('Sample Distributions - Friedman Test')
        plt.xlabel('Groups')
        plt.ylabel('Scores')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.show()

    return to_series_if(
        statistic, p_value, ["statistic", "p-value"],"friedman_test")

    
@ensure_pkg(
    "statsmodels", 
    "'rm_anova' and 'mcnemar' tests expect statsmodels' to be installed.",
    partial_check=True,
    condition=lambda type_test : type_test in ["rm_anova", "mcnemar"]
    )
def statistical_tests(
    *args, 
    test_type="mcnemar", 
    view: bool = False, 
    cmap: str = 'viridis', 
    fig_size: Optional[Tuple[int, int]] = None, 
    error_type: str = 'ci', 
    confidence_interval: float = 0.95, 
    error_bars: bool = True, annot: bool = True, 
    showmeans: bool = True, 
    split: bool = True, 
    trend_line: bool = True, 
    density_overlay: bool = False,
    **kwargs
    ):
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
        
    *args : variable length argument list
        Arguments specific to the statistical test being performed.
    test_type : str, default "mcnemar"
        Type of the statistical test to perform. Options include 'rm_anova', 'cochran_q',
        'mcnemar', 'kruskal_wallis', 'wilcoxon', 'ttest_paired', 'ttest_indep'.
    view : bool, default False
        If True, visualizes the test results or data distributions.
    cmap : str, default 'viridis'
        Colormap for the visualization.
    fig_size : Optional[Tuple[int, int]], default None
        Size of the figure for the visualization.
    **kwargs : dict
        Additional keyword arguments specific to the statistical test 
        being performed.
        
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

    from statsmodels.stats.anova import AnovaRM
    from statsmodels.stats.contingency_tables import mcnemar
    # Define test functions
    test_functions = {
        'rm_anova': lambda: AnovaRM(*args, **kwargs).fit(),
        'cochran_q': lambda: stats.cochran.q(*args, **kwargs),
        'mcnemar': lambda: mcnemar(*args, **kwargs),
        'kruskal_wallis': lambda: stats.kruskal(*args),
        'wilcoxon': lambda: stats.wilcoxon(*args),
        'ttest_paired': lambda: stats.ttest_rel(*args),
        'ttest_indep': lambda: stats.ttest_ind(*args)
    }

    # Execute the specified test                       
    try:
        test_result = test_functions[test_type]()
    except KeyError:
        raise ValueError(f"Invalid test type '{test_type}' specified.")

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
            sns.histplot(x, kde=density_overlay, color='blue', label='Group 1', alpha=0.6)
            sns.histplot(y, kde=density_overlay, color='red', label='Group 2', alpha=0.6)
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

    return test_result

def _visualize_data(data, result, cmap='viridis', fig_size=None):  
    """
    Visualizes the data distribution and highlights the statistical analysis result.
    """
    plt.figure(figsize=fig_size if fig_size else (10, 6))
    # Example visualization: Histogram for data distribution
    plt.hist(data, bins=30, alpha=0.7, color=cmap)
    # Highlighting the statistical result (e.g., mean or median)
    plt.axvline(result, color='red', linestyle='dashed', linewidth=2)
    plt.title('Data Distribution and Statistical Result')
    plt.xlabel('Data Values')
    plt.ylabel('Frequency')
    plt.show()

def get_columns_axis_plot ( values, data =None,  axis=None ): 
    # get the appropriate columns from the values and axis for to iterable . 
    # values if the results of mean, median, var, mode, etc of data applying 
    # on a specific axis. 
    # data can be numpy array, series, dataframe. 
    # axis can be None, 0 and 1. 
    
    # finding the best columns of index to iterate when plotting eleemnt in 
    # axis. 
    if axis is None:
        # we assume is Numpy array, preferably plot 
        return values, data, None 
    if isinstance (values, pd.Series): 
        # columns should be the index 
        # dans transpose data for looping columns 
        # here axis is not so importance 
        
        return values.T, data.T, values.index 
    elif isinstance (values, pd.DataFrame): 
        if axis ==0: 
            return values, data, values.columns 
        elif axis ==1 : 
            # iterate over row and plot values of each rows. 
            # so columns become index 
            return values.T, data.T, values.T.index 
    elif isinstance ( values, np.ndarray): 
        # try to squeeze value to reduce dimensionallity if possible 
        values = np.squeeze ( values )
        if np.ndarray.ndim ==1: 
            return values,  data, range (len(values))
        # for 2D array 
        if axis==0: 
            return values, data, range (values.shape [1])
        elif axis ==1: 
            return values.T, data.T, range ( len(values))
        
    return values, data, None 
    
from typing import Any, Callable, Optional, Tuple, Union
import pandas as pd
import numpy as np

def prepare_plot_data(
    values: Union[np.ndarray, pd.Series, pd.DataFrame],
    data: Optional[Union[np.ndarray, pd.Series, pd.DataFrame]] = None,
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
    >>> prepare_plot_data(values)
    (Series([1, 2, 3], index=['a', 'b', 'c']), None, Index(['a', 'b', 'c']))

    >>> values = pd.DataFrame({'A': [1, 2], 'B': [3, 4]}, index=['x', 'y'])
    >>> prepare_plot_data(values, axis=1)
    (Transposed DataFrame, Transposed DataFrame, Index(['x', 'y']))

    >>> values = np.array([1, 2, 3])
    >>> prepare_plot_data(values, transform=np.square)
    (array([1, 4, 9]), None, range(0, 3))
    """
    if transform is not None and callable(transform):
        values = transform(values)
        if data is not None:
            data = transform(data)

    if axis is None:
        return values, data, None

    if isinstance(values, pd.Series):
        return values.T, data.T if data is not None else None, values.index

    elif isinstance(values, pd.DataFrame):
        if axis == 0:
            print("yes-axis=0")
            return values, data, values.columns
        elif axis == 1:
            print('yes axis=1')
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

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    