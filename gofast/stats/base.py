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

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

from .._typing import ( 
    DataFrame, 
    ArrayLike 
    )
from ..utils.validator import ( 
    build_data_if , 
    assert_xy_in 
    )
from ..utils.funcutils import ( 
    to_numeric_dtypes ,
    ellipsis2false 
    )


def mean(
    data: ArrayLike | DataFrame , 
    /, 
    columns: list = None, 
    as_frame: bool=..., 
    **kws  
    ):
    """ Calculates the average of a list of numbers.
    
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
    mean_value: Mean value of the data 
    
    """
    as_frame, = ellipsis2false(as_frame )
    data = build_data_if(data, columns = columns, to_frame= as_frame  )
    data = to_numeric_dtypes(
        data, pop_cat_features= True, return_feature_types= False, **kws )
    return np.mean(data)

def median(
    data:ArrayLike | DataFrame, /,  
    columns: list=None,
    as_frame: bool=..., 
    **kws 
    ):
    """ Finds the middle value in a list of numbers.
    
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
    median_value: Mean value of the data 
    
    """
    as_frame, = ellipsis2false(as_frame )
    data = build_data_if(data, columns = columns, to_frame= as_frame  )
    data = to_numeric_dtypes(
        data, pop_cat_features= True, return_feature_types= False, **kws )
    return np.median(data)


def mode(
    data:ArrayLike | DataFrame, /,  
    columns: list=None,
    as_frame: bool=..., 
    **kws 
    ):
    """ Determines the most frequently occurring value in a dataset
    
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
    mode_value: Mean value of the data 
    """
    as_frame, = ellipsis2false(as_frame )
    data = build_data_if(data, columns = columns, to_frame= as_frame  )
    data = to_numeric_dtypes(
        data, pop_cat_features= True, return_feature_types= False, **kws )
    mode = stats.mode(data)
    return mode.mode[0]

def variance(
    data:ArrayLike | DataFrame, /,  
    columns: list=None,
    as_frame: bool=..., 
    **kws 
    ):
    """Calculates the variance of a dataset. 
    
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
    variance_value: Mean value of the data 
    
    """
    as_frame, = ellipsis2false(as_frame )
    data = build_data_if(data, columns = columns, to_frame= as_frame  )
    data = to_numeric_dtypes(
        data, pop_cat_features= True, return_feature_types= False, **kws )
    return np.var(data)

def std_dev(
    data:ArrayLike | DataFrame,
    /,  
    columns: list=None,
    as_frame: bool=..., 
    **kws 
    ):
    """Computes the standard deviation of a dataset. 
    
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
    std_dev_value: Mean value of the data 
    
    """
    as_frame, = ellipsis2false(as_frame )
    data = build_data_if(data, columns = columns, to_frame= as_frame  )
    data = to_numeric_dtypes(
        data, pop_cat_features= True, return_feature_types= False, **kws )
    return np.std(data)

def get_range(
    data:ArrayLike | DataFrame,
    /,  
    columns: list=None,
    as_frame: bool=..., 
    **kws):
    """Finds the range (difference between max and min) of a dataset. 
    
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
    range_value: Mean value of the data 
    
    """
    as_frame, = ellipsis2false(as_frame )
    data = build_data_if(data, columns = columns, to_frame= as_frame  )
    data = to_numeric_dtypes(
        data, pop_cat_features= True, return_feature_types= False, **kws )
    return np.max(data) - np.min(data)


def quartiles(
    data:ArrayLike | DataFrame,
    /,  
    columns: list=None,
    as_frame: bool=..., 
    **kws
    ):
    """Determines the quartiles of a dataset. 
    
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
    auartile_values: Quartile values of the data 
    """
    as_frame, = ellipsis2false(as_frame )
    data = build_data_if(data, columns = columns, to_frame= as_frame  )
    data = to_numeric_dtypes(
        data, pop_cat_features= True, return_feature_types= False, **kws )
    return np.percentile(data, [25, 50, 75])

def correlation(
    x=None, 
    y=None, 
    data=None, 
    columns=None, 
    **kws 
    ):
    """Computes the interquartile range of a dataset. 
    
    Parameters 
    ----------
    X: str, Arraylike  
       First array for coorrelation analysis. When `x` is a string 
       `data` must be supplied. `x` should be a column name of the data. \
           
    y: str, Arraylike 
       Second array for coorrelation analysis. When `y` is given as string 
       it should be a column name of the data and `data`must be supplied 
       instead.
    data: ArrayLike, pd.DataFrame 
       Data frame containing valid numeric values. 
    columns: list, 
       Columns to construct the data. 
    kws: dict,  
       Additional keywords arguments for sanitizing the data 
       before proceedings. 
    Return 
    ------
    correlation_value: correlation value of the data 
    
    """
    
    if data is not None: 
        data = build_data_if(data, columns = columns )
        data = to_numeric_dtypes(
            data, pop_cat_features= True, return_feature_types= False, **kws )
        
    x, y = assert_xy_in(x, y, data =data, xy_numeric= True )
    
    return np.corrcoef(x, y)[0, 1]

def iqr(
    data:ArrayLike | DataFrame,
    /,  
    columns: list=None,
    as_frame: bool=..., 
    **kws):
    """ Calculates the Pearson correlation coefficient between two variables.
    
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
    iqr_value: IQR value of the data 
    
    """
    as_frame, = ellipsis2false(as_frame )
    data = build_data_if(data, columns = columns, to_frame= as_frame  )
    data = to_numeric_dtypes(
        data, pop_cat_features= True, return_feature_types= False, **kws )
    Q1, Q3 = np.percentile(data, [25, 75])
    
    return Q3 - Q1

def z_scores(
    data:ArrayLike | DataFrame,
    /,  
    columns: list=None,
    as_frame: bool=..., 
    **kws):
    """ Computes the Z-score for each data point in a dataset.
    
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
    z_scores_value: Z-Score value of the data 
    """
    as_frame, = ellipsis2false(as_frame )
    data = build_data_if(data, columns = columns, to_frame= as_frame  )
    data = to_numeric_dtypes(
        data, pop_cat_features= True, return_feature_types= False, **kws )
    mean = np.mean(data)
    std_dev = np.std(data)
    z_scores = [(x - mean) / std_dev for x in data]
    return z_scores


def descr_stats_summary(
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
    iqr_value: IQR value of the data 
    """
    as_frame, = ellipsis2false(as_frame )
    data = build_data_if(data, columns = columns, to_frame= as_frame  )
    data = to_numeric_dtypes(
        data, pop_cat_features= True, return_feature_types= False, **kws )
    
    series = pd.Series(data)
    return series.describe()

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

def anova_test(*groups, alpha=0.05):
    """
    Perform ANOVA test to compare means across multiple groups.
    
    :param groups: Variable number of lists or arrays, each representing a group.
    :param alpha: Significance level, default is 0.05.
    :return: F-statistic, p-value, and whether to reject the null hypothesis.
    """
    f_stat, p_value = stats.f_oneway(*groups)
    reject_null = p_value < alpha
    return f_stat, p_value, reject_null


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
