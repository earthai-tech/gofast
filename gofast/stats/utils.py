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

from sklearn.cluster import KMeans,SpectralClustering
from sklearn.linear_model import LinearRegression
from sklearn.manifold import MDS

from .._typing import ( 
    DataFrame, 
    ArrayLike 
    )
from ..tools.validator import ( 
    build_data_if , 
    assert_xy_in 
    )
from ..tools.funcutils import ( 
    to_numeric_dtypes ,
    ellipsis2false 
    )

from ..tools._dependency import import_optional_dependency 

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


