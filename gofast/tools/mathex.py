# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Utilities to process and compute parameters. Module for Algebra calculus.
"""
from __future__ import annotations 
import copy 
import inspect 
import warnings 

import numpy as np
import pandas as pd 
from scipy.signal import ( 
    argrelextrema,  )

from scipy.optimize import curve_fit
from scipy.cluster.hierarchy import  linkage 
from scipy.linalg import lstsq
from scipy._lib._util import float_factorial
from scipy.ndimage import convolve1d
from scipy.spatial.distance import pdist, squareform 
import  matplotlib.pyplot as plt

from ._arraytools import axis_slice
from .._gofastlog import gofastlog
from .._docstring import refglossary
from ..decorators import  refAppender, docSanitizer
from ..exceptions import SiteError
from .._typing import (
    _T, 
    _F,
    List, 
    Tuple,
    Union,
    ArrayLike,
    NDArray,
    DType,
    Optional,
    _SP, 
    Series, 
    DataFrame,
)
from .box import Boxspace 
from .funcutils import (
    _assert_all_types, 
    _validate_name_in, 
    assert_ratio,
    concat_array_from_list, 
    remove_outliers, 
    find_close_position,
    to_numeric_dtypes, 
    ellipsis2false, 
    fancy_printer,
    smart_format,
    is_iterable, 
    reshape,
    fillNaN, 
    spi,     
)
from .validator import ( 
    _is_arraylike_1d, 
    _is_numeric_dtype,
    check_consistency_size,
    check_y,
    check_array,
    assert_xy_in, 
    build_data_if
    )
try: import scipy.stats as spstats
except: pass 

_logger =gofastlog.get_gofast_logger(__name__)

mu0 = 4 * np.pi * 1e-7 

def make_mxs(
    y,
    yt,
    threshold=0.5, 
    star_mxs=True, 
    return_ymxs=False,
    mode="strict", 
    include_nan=False, 
    trailer="*"
    ):
    """
    Compute the similarity between labels in arrays true y and predicted yt. 
    
    Function transform yt based on these similarities, and create a new 
    array `ymxs` by filling NaN values in y with corresponding labels from 
    transformed yt. Handles NaN values in `yt` based on the `mode` and
    `include_nan` parameters. See more in [1]_

    Parameters
    ----------
    y : array-like
        The target array containing valid labels and potentially NaN values.
    yt : array-like
        The array containing predicted labels from KMeans.
    threshold : float, optional
        The threshold for considering a label in `y` as similar to a label 
        in `yt` (default is 0.5).
    star_mxs : bool, optional
        If True, appends `trailer` to labels in `yt` when similarity is found 
        (default is True).
    return_ymx : bool, optional
        If True, returns the mixed array `ymx`; otherwise, returns a 
        dictionary of label similarities (default is False).
    mode : str, optional
        "strict" or "soft" handling of NaN values in `yt` (default is "strict").
    include_nan : bool, optional
        If True and `mode` is "soft", includes NaN values in `yt` during 
        similarity computation (default is False).
    trailer : str, optional
        The string to append to labels in `yt` when `star_mxs` is True
        (default is "*").

    Returns
    -------
    array or dict
        Mixed array `ymx` if `return_ymx` is True; otherwise, a 
        dictionary representing similarities of labels in `y` and `yt`.

    Raises
    ------
    ValueError
        If `yt` contains NaN values in "strict" mode or if `trailer` 
        is a number.
        
    References
    -----------
    [1] Kouadio, K.L, Liu R., Liu J., A mixture Learning Strategy for predicting 
        permeability coefficient K (2024). Computers and Geosciences, doi:XXXXX 

    Examples
    --------
    >>> y = np.array([1, 2, np.nan, 4])
    >>> yt = np.array([1, 2, 3, 4])
    >>> make_mxs(y, yt, threshold=0.5, star_mxs=True, return_ymx=True, trailer="#")
    array([1, 2, '3#', '44#'])

    >>> make_mxs(y, yt, threshold=1.5, star_mxs=False, return_ymx=False, mode="soft")
    {1: True, 2: True, np.nan: False, 4: True}
    """
    from sklearn.metrics import pairwise_distances
    
    if not isinstance(trailer, str) or trailer.isdigit():
        raise ValueError("trailer must be a non-numeric string.")

    if mode == "strict" and np.isnan(yt).any():
        raise ValueError("yt should not contain NaN values in 'strict' mode.")

    # Appending trailer to yt if star_mxs is True
    yt_transformed = np.array([f"{label}{trailer}" for label in yt]
                              ) if star_mxs else yt.copy()

    # Computing similarities and transforming yt
    similarities = {}
    for i, label_y in enumerate(y):
        include_label = not np.isnan(label_y) or (include_nan and mode == "soft")
        if include_label:
            similarity = pairwise_distances([[label_y]], [[yt[i]]])[0][0] <= threshold
            similarities[label_y] = similarity
            if similarity and star_mxs:
                # Transform similar labels in yt
                label_yt_trailer = f"{yt[i]}{trailer}"
                yt_transformed[yt_transformed == label_yt_trailer
                               ] = f"{label_y}{label_yt_trailer}"
    # Filling NaN positions in y with corresponding labels from transformed yt
    ymxs = np.where(np.isnan(y), yt_transformed, y)
    
    return ymxs if return_ymxs else similarities

def label_importance(y, include_nan=False):
    """
    Compute the importance of each label in a target array.

    This function calculates the frequency of each unique label 
    in the target array `y`. Importance is defined as the proportion of 
    occurrences of each label in the array.

    Parameters
    ----------
    y : array-like
        The target array containing labels.
    include_nan : bool, optional
        If True, includes NaN values in the calculation, otherwise 
        excludes them (default is False).

    Returns
    -------
    dict
        A dictionary with labels as keys and their corresponding 
        importance as values.

    Notes
    -----
    The mathematical formulation for the importance of a label `l` is given by:

    .. math::

        I(l) = \\frac{\\text{{count of }} l \\text{{ in }} y}{\\text{{total number of elements in }} y}

    Examples
    --------
    >>> y = np.array([1, 2, 2, 3, 3, 3, np.nan])
    >>> label_importance(y)
    {1.0: 0.16666666666666666, 2.0: 0.3333333333333333, 3.0: 0.5}

    >>> label_importance(y, include_nan=True)
    {1.0: 0.14285714285714285, 2.0: 0.2857142857142857, 3.0: 0.42857142857142855,
     nan: 0.14285714285714285}
    """
    y = np.array ( y )
    if not include_nan:
        y = y[~np.isnan(y)]
    labels, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    return {label: count / total for label, count in zip(labels, counts)}

def linear_regression(X, coef, bias=0., noise=0.):
    """
    linear regression.
    
    Generate output for linear regression, modeling a relationship between
    features and a response using a linear approach.

    Linear regression is one of the simplest formss of regression, useful for
    understanding relationships between variables and for making predictions.
    It's widely used in various fields like economics, biology, and engineering.

    Parameters
    ----------
    X : ndarray
        The input samples with shape (n_samples, n_features).
    coef : ndarray
        The coefficients for the linear regression with shape (n_features,).
    bias : float
        The bias term in the linear equation.
    noise : float
        The standard deviation of the Gaussian noise added to the output.

    Returns
    -------
    y : ndarray
        The output values for linear regression with shape (n_samples,).

    Formula
    -------
    y = X \cdot coef + bias + noise
    
    Applications
    ------------
    - Trend analysis in time series data.
    - Predictive modeling in business and finance.
    - Estimating relationships in scientific experiments.
    """
    return np.dot(X, coef) + bias + noise * np.random.randn(X.shape[0])

def quadratic_regression(X, coef, bias=0., noise=0.):
    """
    Quadratic regression.

    Generate output for quadratic regression, which models a parabolic 
    relationship between the dependent variable and independent variables.

    Quadratic regression is suitable for datasets with a non-linear trend. It's 
    often used in areas where the rate of change increases or decreases rapidly.

    Applications
    ------------
    - Modeling acceleration or deceleration patterns in physics.
    - Growth rate analysis in biology and economics.
    - Prediction in financial markets with parabolic trends.
    
    Parameters
    ----------
    X : ndarray
        The input samples with shape (n_samples, n_features).
    coef : ndarray
        The coefficients for the linear regression with shape (n_features,).
    bias : float
        The bias term in the linear equation.
    noise : float
        The standard deviation of the Gaussian noise added to the output.
        
    Formula
    -------
    y = (X^2) \cdot coef + bias + noise
    """
    return np.dot(X**2, coef) + bias + noise * np.random.randn(X.shape[0])

def cubic_regression(X, coef, bias=0., noise=0.):
    """
    Cubic regression.

    Generate output for cubic regression, fitting a cubic polynomial to the data.

    Cubic regression provides a more flexible curve than quadratic models and is 
    beneficial in studying more complex relationships, especially where inflection 
    points are present.

    Applications
    ------------
    - Analyzing drug response curves in pharmacology.
    - Studying the growth patterns of organisms or populations.
    - Complex trend analysis in economic data.
    
    Parameters
    ----------
    X : ndarray
        The input samples with shape (n_samples, n_features).
    coef : ndarray
        The coefficients for the linear regression with shape (n_features,).
    bias : float
        The bias term in the linear equation.
    noise : float
        The standard deviation of the Gaussian noise added to the output.

    Formula
    -------
    y = (X^3) \cdot coef + bias + noise
    """
    return np.dot(X**3, coef) + bias + noise * np.random.randn(X.shape[0])

def exponential_regression(X, coef, bias=0., noise=0.):
    """
    Exponential regression.

    Generate output for exponential regression, ideal for modeling growth or decay.

    Exponential regression is used when data grows or decays at a constant
    percentage rate. It's crucial in fields like biology for population growth 
    studies or in finance for compound interest calculations.

    Applications
    ------------
    - Modeling population growth or decline.
    - Financial modeling for compound interest.
    - Radioactive decay in physics.
    Parameters
    ----------
    X : ndarray
        The input samples with shape (n_samples, n_features).
    coef : ndarray
        The coefficients for the linear regression with shape (n_features,).
    bias : float
        The bias term in the linear equation.
    noise : float
        The standard deviation of the Gaussian noise added to the output.

    Formula
    -------
    y = exp(X \cdot coef) + bias + noise
    """
    return np.exp(np.dot(X, coef)) + bias + noise * np.random.randn(X.shape[0])

def logarithmic_regression(X, coef, bias=0., noise=0.):
    """
    Logarithmic regression.

    Generate output for logarithmic regression, suitable for modeling processes 
    that rapidly increase or decrease and then level off.

    Logarithmic regression is particularly useful in situations where the rate of
    change decreases over time. It's often used in scientific data analysis.

    Applications
    ------------
    - Analyzing diminishing returns in economics.
    - Growth rate analysis in biological processes.
    - Signal processing and sound intensity measurements.
    
    Parameters
    ----------
    X : ndarray
        The input samples with shape (n_samples, n_features).
    coef : ndarray
        The coefficients for the linear regression with shape (n_features,).
    bias : float
        The bias term in the linear equation.
    noise : float
        The standard deviation of the Gaussian noise added to the output.

    Formula
    -------
    y = log(X) \cdot coef + bias + noise
    """
    return np.dot(np.log(X), coef) + bias + noise * np.random.randn(X.shape[0])

def sinusoidal_regression(X, coef, bias=0., noise=0.):
    """
    Sinusoidal regression.

    Generate output for sinusoidal regression, fitting a sinusoidal model to the data.

    This type of regression is useful for modeling cyclical patterns and is commonly
    used in fields like meteorology, seasonal studies, and signal processing.

    Applications
    ------------
    - Seasonal pattern analysis in climatology.
    - Modeling cyclical trends in economics.
    - Signal analysis in electrical engineering.
    
    Parameters
    ----------
    X : ndarray
        The input samples with shape (n_samples, n_features).
    coef : ndarray
        The coefficients for the linear regression with shape (n_features,).
    bias : float
        The bias term in the linear equation.
    noise : float
        The standard deviation of the Gaussian noise added to the output.

    Formula
    -------
    y = sin(X \cdot coef) + bias + noise
    """
    return np.sin(np.dot(X, coef)) + bias + noise * np.random.randn(X.shape[0])

def step_regression(X, coef, bias=0., noise=0.):
    """
    Step regression.

    Step regression is valuable for modeling scenarios where the dependent variable
    changes abruptly at specific thresholds. It's used in quality control and market
    segmentation analysis.

    Applications
    ------------
    - Quality assessment in manufacturing processes.
    - Customer segmentation in marketing.
    - Modeling sudden changes in environmental data.
    
    Parameters
    ----------
    X : ndarray
        The input samples with shape (n_samples, n_features).
    coef : ndarray
        The coefficients for the linear regression with shape (n_features,).
    bias : float
        The bias term in the linear equation.
    noise : float
        The standard deviation of the Gaussian noise added to the output.


    Formula
    -------
    y = step_function(X \cdot coef) + bias + noise

    Note: step_function returns 1 if x >= 0, else 0.
    """
    step_function = np.vectorize(lambda x: 1 if x >= 0 else 0)
    return step_function(np.dot(X, coef)) + bias + noise * np.random.randn(X.shape[0])

def standard_scaler(X, y=None):
    """
    Scales features to have zero mean and unit variance.

    Standard scaling is vital in many machine learning algorithms that are 
    sensitive to the scale of input features. It's commonly used in algorithms
    like Support Vector Machines and k-Nearest Neighbors.

    Applications
    ------------
    - Data preprocessing for machine learning models.
    - Feature normalization in image processing.
    - Standardizing variables in statistical analysis.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The input samples.
    y : ndarray of shape (n_samples,), optional
        The output values. If provided, it will be scaled as well.

    Returns
    -------
    X_scaled : ndarray
        Scaled version of X.
    y_scaled : ndarray, optional
        Scaled version of y, if y is provided.

    Formula
    -------
    For each feature, the Standard Scaler performs the following operation:
        z = \frac{x - \mu}{\sigma}
    where \mu is the mean and \sigma is the standard deviation of the feature.

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> X_scaled = standard_scaler(X)
    """
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_scaled = (X - X_mean) / X_std

    if y is not None:
        y_mean = y.mean()
        y_std = y.std()
        y_scaled = (y - y_mean) / y_std
        return X_scaled, y_scaled

    return X_scaled

def minmax_scaler(X, y=None):
    """
    Scales each feature to a given range, typically [0, 1].

    MinMax scaling is often used when the algorithm requires a bounded interval. 
    It's particularly useful in neural networks and image processing where values 
    need to be normalized.

    Applications
    ------------
    - Data normalization for neural networks.
    - Preprocessing data in computer vision tasks.
    - Scaling features for optimization problems.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The input samples.
    y : ndarray of shape (n_samples,), optional
        The output values. If provided, it will be scaled as well.

    Returns
    -------
    X_scaled : ndarray
        Scaled version of X.
    y_scaled : ndarray, optional
        Scaled version of y, if y is provided.

    Formula
    -------
    The MinMax Scaler performs the following operation for each feature:
        z = \frac{x - \min(x)}{\max(x) - \min(x)}

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> X_scaled = minmax_scaler(X)
    """
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_scaled = (X - X_min) / (X_max - X_min)

    if y is not None:
        y_min = y.min()
        y_max = y.max()
        y_scaled = (y - y_min) / (y_max - y_min)
        return X_scaled, y_scaled

    return X_scaled

def normalize(X, y=None):
    """
    Scales individual samples to have unit norm.

    Normalization is critical for distance-based algorithms like k-Nearest 
    Neighbors and clustering algorithms. It ensures that each feature 
    contributes proportionately to the final distance.

    Applications
    ------------
    - Preprocessing for clustering algorithms.
    - Normalizing data in natural language processing.
    - Feature scaling in bioinformatics.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The input samples.
    y : ndarray of shape (n_samples,), optional
        The output values. If provided, it will be normalized as well.

    Returns
    -------
    X_normalized : ndarray
        Normalized version of X.
    y_normalized : ndarray, optional
        Normalized version of y, if y is provided.

    Formula
    -------
    The Normalize method scales each sample as follows:
        z = \frac{x}{||x||}
    where ||x|| is the Euclidean norm (L2 norm) of the sample.

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> X_normalized = normalize(X)
    """
    X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    X_normalized = X / X_norm

    if y is not None:
        y_norm = np.linalg.norm(y, axis=0, keepdims=True)
        y_normalized = y / y_norm
        return X_normalized, y_normalized

    return X_normalized


def get_azimuth (
    xlon: str | ArrayLike, 
    ylat: str| ArrayLike, 
    *, 
    data: DataFrame =None, 
    utm_zone:str=None, 
    projection:str='ll', 
    isdeg:bool=True, 
    mode:str='soft', 
    extrapolate:bool =...,
    view:bool=..., 
    ): 
    """Compute azimuth from coordinate locations ( latitude,  longitude). 
    
    If `easting` and `northing` are given rather than `longitude` and  
    `latitude`, the projection should explicitely set to ``UTM`` to perform 
    the ideal conversion. However if mode is set to `soft` (default), the type
    of projection is automatically detected . Note that when UTM coordinates 
    are provided, `xlon` and `ylat` fit ``easting`` and ``northing`` 
    respectively.
    
    Parameters
    -----------
    xlon, ylat : Arraylike 1d or str, str 
       ArrayLike of easting/longitude and arraylike of nothing/latitude. They 
       should be one dimensional. In principle if data is supplied, they must 
       be series.  If `xlon` and `ylat` are given as string values, the 
       `data` must be supplied. xlon and ylat names must be included in the  
       dataframe otherwise an error raises. 
       
    data: pd.DataFrame, 
       Data containing x and y names. Need to be supplied when x and y 
       are given as string names. 
       
    utm_zone: Optional, string
       zone number and 'S' or 'N' e.g. '55S'. Default to the centre point
       of coordinates points in the survey area. It should be a string (##N or ##S)
       in the form of number and North or South hemisphere, 10S or 03N
       
    projection: str, ['utm'|'ll'] 
       The coordinate system in which the data points for the profile is collected. 
       when `mode='soft'`,  the auto-detection will be triggered and find the 
       suitable coordinate system. However, it is recommended to explicitly 
       provide projection when data is in UTM coordinates. 
       Note that if `x` and `y` are composed of value greater than 180 degrees 
       for longitude and 90 degrees for latitude, and method is still in 
       the ``soft` mode, it should be considered as  longitude-latitude ``UTM``
       coordinates system. 
       
    isdeg: bool, default=True 
      By default xlon and xlat are in degree coordinates. If both arguments 
      are given in radians, set to ``False`` instead. 
      
    mode: str , ['soft'|'strict']
      ``strict`` mode does not convert any coordinates system to other at least
      it is explicitly set to `projection` whereas the `soft` does.
      
    extrapolate: bool, default=False 
      In principle, the azimuth is compute between two points. Thus, the number
      of values computed for :math:`N` stations should  be  :math:`N-1`. To fit
      values to match the number of size of the array, `extrapolate` should be 
      ``True``. In that case, the first station holds a <<fake>> azimuth as 
      the closer value computed from interpolation of all azimuths. 
      
    view: bool, default=False, 
       Quick view of the azimuth. It is usefull especially when 
       extrapolate is set to ``True``. 
       
    Return 
    --------
    azim: ArrayLike 
       Azimuth computed from locations. 
       
    Examples 
    ----------
    >>> import gofast as gf 
    >>> from gofast.tools.mathex  import get_azimuth 
    >>> # generate a data from ERP 
    >>> data = gf.make_erp (n_stations =7 ).frame 
    >>> get_azimuth ( data.longitude, data.latitude)
    array([54.575, 54.575, 54.575, 54.575, 54.575, 54.575])
    >>> get_azimuth ( data.longitude, data.latitude, view =True, extrapolate=True)
    array([54.57500007, 54.575     , 54.575     , 54.575     , 54.575     ,
           54.575     , 54.575     ])
    
    """
    from ..site import Location 
    
    mode = str(mode).lower() 
    projection= str(projection).lower()
    extrapolate, view = ellipsis2false (extrapolate, view)

    xlon , ylat = assert_xy_in(xlon , ylat , data = data )
    
    if ( 
            xlon.max() > 180.  and ylat.max() > 90.  
            and projection=='ll' 
            and mode=='soft'
            ): 
        warnings.warn("xlon and ylat arguments are greater than 180 degrees."
                     " we assume the coordinates are UTM. Set explicitly"
                     " projection to ``UTM`` to avoid this warning.")
        projection='utm'
        
    if projection=='utm':
        if utm_zone is None: 
            raise TypeError ("utm_zone cannot be None when projection is UTM.")
            
        ylat , xlon = Location.to_latlon_in(
            xlon, ylat, utm_zone= utm_zone)
        
    if len(xlon) ==1 or len(ylat)==1: 
        msg = "Azimuth computation expects at least two points. Got 1"
        if mode=='soft': 
            warnings.warn(msg) 
            return 0. 
        
        raise TypeError(msg )
    # convert to radian 
    if isdeg: 
        xlon = np.deg2rad (xlon ) ; ylat = np.deg2rad ( ylat)
    
    dx = map (lambda ii: np.cos ( ylat[ii]) * np.sin( ylat [ii+1 ]) - 
        np.sin(ylat[ii]) * np.cos( ylat[ii+1]) * np.cos (xlon[ii+1]- xlon[ii]), 
        range (len(xlon)-1)
        )
    dy = map( lambda ii: np.cos (ylat[ii+1])* np.sin( xlon[ii+1]- xlon[ii]), 
                   range ( len(xlon)-1)
                   )
    # to deg 
    z = np.around ( np.rad2deg ( np.arctan2(list(dx) , list(dy) ) ), 3)  
    azim = z.copy() 
    if extrapolate: 
        # use mean azimum of the total area zone and 
        # recompute the position by interpolation 
        azim = np.hstack ( ( [z.mean(), z ]))
        # reset the interpolare value at the first position
        with warnings.catch_warnings():
            #warnings.filterwarnings(action='ignore', category=OptimizeWarning)
            warnings.simplefilter("ignore")
            azim [0] = scalePosition(azim )[0][0] 
        
    if view: 
        x = np.arange ( len(azim )) 
        fig,  ax = plt.subplots (1, 1, figsize = (10, 4))
        # add Nan to the first position of z 
        z = np.hstack (([np.nan], z )) if extrapolate else z 
       
        ax.plot (x, 
                 azim, 
                 c='#0A4CEE',
                 marker = 'o', 
                 label ='extra-azimuth'
                 ) 
        
        ax.plot (x, 
                z, 
                'ok-', 
                label ='raw azimuth'
                )
        ax.legend ( ) 
        ax.set_xlabel ('x')
        ax.set_ylabel ('y') 

    return azim

def linkage_matrix(
    df: DataFrame ,
    columns:List[str] =None,  
    kind:str ='design', 
    metric:str ='euclidean',   
    method:str ='complete', 
    as_frame =False,
    optimal_ordering=False, 
 )->NDArray: 
    r""" Compute the distance matrix from the hierachical clustering algorithm
    
    Parameters 
    ------------ 
    df: dataframe or NDArray of (n_samples, n_features) 
        dataframe of Ndarray. If array is given , must specify the column names
        to much the array shape 1 
    columns: list 
        list of labels to name each columns of arrays of (n_samples, n_features) 
        If dataframe is given, don't need to specify the columns. 
        
    kind: str, ['squareform'|'condense'|'design'], default is {'design'}
        kind of approach to summing up the linkage matrix. 
        Indeed, a condensed distance matrix is a flat array containing the 
        upper triangular of the distance matrix. This is the form that ``pdist`` 
        returns. Alternatively, a collection of :math:`m` observation vectors 
        in :math:`n` dimensions may be passed as  an :math:`m` by :math:`n` 
        array. All elements of the condensed distance matrix must be finite, 
        i.e., no NaNs or infs.
        Alternatively, we could used the ``squareform`` distance matrix to yield
        different distance values than expected. 
        the ``design`` approach uses the complete inpout example matrix  also 
        called 'design matrix' to lead correct linkage matrix similar to 
        `squareform` and `condense``. 
        
    metric : str or callable, default is {'euclidean'}
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by :func:`sklearn.metrics.pairwise.pairwise_distances`.
        If ``X`` is the distance array itself, use "precomputed" as the metric.
        Precomputed distance matrices must have 0 along the diagonal.
        
    method : str, optional, default is {'complete'}
        The linkage algorithm to use. See the ``Linkage Methods`` section below
        for full descriptions.
        
    optimal_ordering : bool, optional
        If True, the linkage matrix will be reordered so that the distance
        between successive leaves is minimal. This results in a more intuitive
        tree structure when the data are visualized. defaults to False, because
        this algorithm can be slow, particularly on large datasets. See
        also :func:`scipy.cluster.hierarchy.linkage`. 
        
        
    Returns 
    --------
    row_clusters: linkage matrix 
        consist of several rows where each rw represents one merge. The first 
        and second columns denotes the most dissimilar members of each cluster 
        and the third columns reports the distance between those members 
        
        
    Linkage Methods 
    -----------------
    The following are methods for calculating the distance between the
    newly formed cluster :math:`u` and each :math:`v`.

    * method='single' assigns

      .. math::
         d(u,v) = \min(dist(u[i],v[j]))

      for all points :math:`i` in cluster :math:`u` and
      :math:`j` in cluster :math:`v`. This is also known as the
      Nearest Point Algorithm.

    * method='complete' assigns

      .. math::
         d(u, v) = \max(dist(u[i],v[j]))

      for all points :math:`i` in cluster u and :math:`j` in
      cluster :math:`v`. This is also known by the Farthest Point
      Algorithm or Voor Hees Algorithm.

    * method='average' assigns

      .. math::
         d(u,v) = \sum_{ij} \\frac{d(u[i], v[j])}{(|u|*|v|)}

      for all points :math:`i` and :math:`j` where :math:`|u|`
      and :math:`|v|` are the cardinalities of clusters :math:`u`
      and :math:`v`, respectively. This is also called the UPGMA
      algorithm.

    * method='weighted' assigns

      .. math::
         d(u,v) = (dist(s,v) + dist(t,v))/2

      where cluster u was formed with cluster s and t and v
      is a remaining cluster in the forest (also called WPGMA).

    * method='centroid' assigns

      .. math::
         dist(s,t) = ||c_s-c_t||_2

      where :math:`c_s` and :math:`c_t` are the centroids of
      clusters :math:`s` and :math:`t`, respectively. When two
      clusters :math:`s` and :math:`t` are combined into a new
      cluster :math:`u`, the new centroid is computed over all the
      original objects in clusters :math:`s` and :math:`t`. The
      distance then becomes the Euclidean distance between the
      centroid of :math:`u` and the centroid of a remaining cluster
      :math:`v` in the forest. This is also known as the UPGMC
      algorithm.

    * method='median' assigns :math:`d(s,t)` like the ``centroid``
      method. When two clusters :math:`s` and :math:`t` are combined
      into a new cluster :math:`u`, the average of centroids s and t
      give the new centroid :math:`u`. This is also known as the
      WPGMC algorithm.

    * method='ward' uses the Ward variance minimization algorithm.
      The new entry :math:`d(u,v)` is computed as follows,

      .. math::

         d(u,v) = \sqrt{\frac{|v|+|s|}{_T}d(v,s)^2 \\
                      + \frac{|v|+|t|}{_T}d(v,t)^2 \\
                      - \frac{|v|}{_T}d(s,t)^2}

      where :math:`u` is the newly joined cluster consisting of
      clusters :math:`s` and :math:`t`, :math:`v` is an unused
      cluster in the forest, :math:`_T=|v|+|s|+|t|`, and
      :math:`|*|` is the cardinality of its argument. This is also
      known as the incremental algorithm.

    Warning: When the minimum distance pair in the forest is chosen, there
    may be two or more pairs with the same minimum distance. This
    implementation may choose a different minimum than the MATLAB
    version.
    
    See Also
    --------
    scipy.spatial.distance.pdist : pairwise distance metrics

    References
    ----------
    .. [1] Daniel Mullner, "Modern hierarchical, agglomerative clustering
           algorithms", :arXiv:`1109.2378v1`.
    .. [2] Ziv Bar-Joseph, David K. Gifford, Tommi S. Jaakkola, "Fast optimal
           leaf ordering for hierarchical clustering", 2001. Bioinformatics
           :doi:`10.1093/bioinformatics/17.suppl_1.S22`

    """
    df = _assert_all_types(df, pd.DataFrame, np.ndarray)
    
    if columns is not None: 
        if isinstance (columns , str):
            columns = [columns]
        if len(columns)!= df.shape [1]: 
            raise TypeError("Number of columns must fit the shape of X."
                            f" got {len(columns)} instead of {df.shape [1]}"
                            )
        df = pd.DataFrame(data = df.values if hasattr(df, 'columns') else df ,
                          columns = columns )
        
    kind= str(kind).lower().strip() 
    if kind not in ('squareform', 'condense', 'design'): 
        raise ValueError(f"Unknown method {method!r}. Expect 'squareform',"
                         " 'condense' or 'design'.")
        
    labels = [f'ID_{i}' for i in range(len(df))]
    if kind =='squareform': 
        row_dist = pd.DataFrame (squareform ( 
        pdist(df, metric= metric )), columns = labels  , 
        index = labels)
        row_clusters = linkage (row_dist, method =method, metric =metric
                                )
    if kind =='condens': 
        row_clusters = linkage (pdist(df, metric =metric), method =method
                                )
    if kind =='design': 
        row_clusters = linkage(df.values if hasattr (df, 'columns') else df, 
                               method = method, 
                               optimal_ordering=optimal_ordering )
        
    if as_frame: 
        row_clusters = pd.DataFrame ( row_clusters, 
                                     columns = [ 'row label 1', 
                                                'row lable 2', 
                                                'distance', 
                                                'no. of items in clust.'
                                                ], 
                                     index = ['cluster %d' % (i +1) for i in 
                                              range(row_clusters.shape[0])
                                              ]
                                     )
    return row_clusters 

def interpolate2d (
        arr2d: NDArray[float] , 
        method:str  = 'slinear', 
        **kws): 
    """ Interpolate the data in 2D dimensional array. 
    
    If the data contains some missing values. It should be replaced by the 
    interpolated values. 
    
    Parameters 
    -----------
    arr2d : np.ndarray, shape  (N, M)
        2D dimensional data 
        
    method: str, default ``linear``
        Interpolation technique to use. Can be ``nearest``or ``pad``. 
    
    kws: dict 
        Additional keywords. Refer to :func:`~.interpolate1d`. 
        
    Returns 
    -------
    arr2d:  np.ndarray, shape  (N, M)
        2D dimensional data interpolated 
    
    Examples 
    ---------
    >>> from gofast.methods.em import EM 
    >>> from gofast.tools.mathex  import interpolate2d 
    >>> # make 2d matrix of frequency
    >>> emObj = EM().fit(r'data/edis')
    >>> freq2d = emObj.make2d (out = 'freq')
    >>> freq2d_i = interpolate2d(freq2d ) 
    >>> freq2d.shape 
    ...(55, 3)
    >>> freq2d 
    ... array([[7.00000e+04, 7.00000e+04, 7.00000e+04],
           [5.88000e+04, 5.88000e+04, 5.88000e+04],
           ...
            [6.87500e+00, 6.87500e+00, 6.87500e+00],
            [        nan,         nan, 5.62500e+00]])
    >>> freq2d_i
    ... array([[7.000000e+04, 7.000000e+04, 7.000000e+04],
           [5.880000e+04, 5.880000e+04, 5.880000e+04],
           ...
           [6.875000e+00, 6.875000e+00, 6.875000e+00],
           [5.625000e+00, 5.625000e+00, 5.625000e+00]])
    
    References 
    ----------
    
    https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.interp2d.html        
        
    """ 
    arr2d = np.array(arr2d)
    
    if len(arr2d.shape) ==1: 
        arr2d = arr2d[:, None] # put on 
    if arr2d.shape[0] ==1: 
        arr2d = reshape (arr2d, axis=0)
    
    if not hasattr (arr2d , '__complex__'): 
        arr2d = check_array(
            arr2d, 
            to_frame = False, 
            input_name ="arr2d",
            force_all_finite="allow-nan" ,
            dtype =arr2d.dtype, 
            )
    arr2d  = np.hstack ([ 
        reshape (interpolate1d(arr2d[:, ii], 
                kind=method, 
                method ='pd', 
                 **kws), 
                 axis=0)
             for ii in  range (arr2d.shape[1])]
        )
    return arr2d 


@refAppender(refglossary.__doc__)
@docSanitizer()    
def scalePosition(
        ydata: ArrayLike | _SP | Series | DataFrame ,
        xdata: ArrayLike| Series = None, 
        func : Optional [_F] = None ,
        c_order: Optional[int|str] = 0,
        show: bool =False, 
        **kws): 
    """ Correct data location or position and return new corrected location 
    
    Parameters 
    ----------
    ydata: array_like, series or dataframe
        The dependent data, a length M array - nominally ``f(xdata, ...)``.
        
    xdata: array_like or object
        The independent variable where the data is measured. Should usually 
        be an M-length sequence or an (k,M)-shaped array for functions with
        k predictors, but can actually be any object. If ``None``, `xdata` is 
        generated by default using the length of the given `ydata`.
        
    func: callable 
        The model function, ``f(x, ...)``. It must take the independent variable 
        as the first argument and the parameters to fit as separate remaining
        arguments. The default `func` is ``linear`` function i.e  for ``f(x)= ax +b``. 
        where `a` is slope and `b` is the intercept value. Setting your own 
        function for better fitting is recommended. 
        
    c_order: int or str
        The index or the column name if ``ydata`` is given as a dataframe to 
        select the right column for scaling.
    show: bool 
        Quick visualization of data distribution. 

    kws: dict 
        Additional keyword argument from  `scipy.optimize_curvefit` parameters. 
        Refer to `scipy.optimize.curve_fit`_.  
        
    Returns 
    --------
    - ydata - array -like - Data scaled 
    - popt - array-like Optimal values for the parameters so that the sum of 
        the squared residuals of ``f(xdata, *popt) - ydata`` is minimized.
    - pcov - array like The estimated covariance of popt. The diagonals provide
        the variance of the parameter estimate. To compute one standard deviation 
        errors on the parameters use ``perr = np.sqrt(np.diag(pcov))``. How the
        sigma parameter affects the estimated covariance depends on absolute_sigma 
        argument, as described above. If the Jacobian matrix at the solution
        doesn’t have a full rank, then ‘lm’ method returns a matrix filled with
        np.inf, on the other hand 'trf' and 'dogbox' methods use Moore-Penrose
        pseudoinverse to compute the covariance matrix.
        
    Examples
    --------
    >>> from gofast.tools import erpSelector, scalePosition 
    >>> df = erpSelector('data/erp/l10_gbalo.xlsx') 
    >>> df.columns 
    ... Index(['station', 'resistivity', 'longitude', 'latitude', 'easting',
           'northing'],
          dtype='object')
    >>> # correcting northing coordinates from easting data 
    >>> northing_corrected, popt, pcov = scalePosition(ydata =df.northing , 
                                               xdata = df.easting, show=True)
    >>> len(df.northing.values) , len(northing_corrected)
    ... (20, 20)
    >>> popt  # by default popt =(slope:a ,intercept: b)
    ...  array([1.01151734e+00, 2.93731377e+05])
    >>> # corrected easting coordinates using the default x.
    >>> easting_corrected, *_= scalePosition(ydata =df.easting , show=True)
    >>> df.easting.values 
    ... array([790284, 790281, 790277, 790270, 790265, 790260, 790254, 790248,
    ...       790243, 790237, 790231, 790224, 790218, 790211, 790206, 790200,
    ...       790194, 790187, 790181, 790175], dtype=int64)
    >>> easting_corrected
    ... array([790288.18571705, 790282.30300999, 790276.42030293, 790270.53759587,
    ...       790264.6548888 , 790258.77218174, 790252.88947468, 790247.00676762,
    ...       790241.12406056, 790235.2413535 , 790229.35864644, 790223.47593938,
    ...       790217.59323232, 790211.71052526, 790205.8278182 , 790199.94511114,
    ...       790194.06240407, 790188.17969701, 790182.29698995, 790176.41428289])
    
    """
    def linfunc (x, a, b): 
        """ Set the simple linear function"""
        return a * x + b 
        
    if str(func).lower() in ('none' , 'linear'): 
        func = linfunc 
    elif not hasattr(func, '__call__') or not inspect.isfunction (func): 
        raise TypeError(
            f'`func` argument is a callable not {type(func).__name__!r}')
        
    ydata = _assert_all_types(ydata, list, tuple, np.ndarray,
                              pd.Series, pd.DataFrame  )
    c_order = _assert_all_types(c_order, int, float, str)
    try : c_order = int(c_order) 
    except: pass 

    if isinstance(ydata, pd.DataFrame): 
        if c_order ==0: 
            warnings.warn("The first column of the data should be considered"
                          " as the `y` target.")
        if c_order is None: 
            raise TypeError('Dataframe is given. The `c_order` argument should '
                            'be defined for column selection. Use column name'
                            ' instead')
        if isinstance(c_order, str): 
            # check whether the value is on the column name
            if c_order.lower() not in list(map( 
                    lambda x :x.lower(), ydata.columns)): 
                raise ValueError (
                    f'c_order {c_order!r} not found in {list(ydata.columns)}'
                    ' Use the index instead.')
                # if c_order exists find the index and get the 
                # right column name 
            ix_c = list(map( lambda x :x.lower(), ydata.columns)
                        ).index(c_order.lower())
            ydata = ydata.iloc [:, ix_c] # series 
        elif isinstance (c_order, (int, float)): 
            c_order =int(c_order) 
            if c_order >= len(ydata.columns): 
                raise ValueError(
                    f"`c_order`'{c_order}' should be less than the number of " 
                    f"given columns '{len(ydata.columns)}'. Use column name instead.")
            ydata= ydata.iloc[:, c_order]
                  
    ydata = check_y (np.array(ydata)  , input_name= "ydata")
    
    if xdata is None: 
        xdata = np.linspace(0, 4, len(ydata))
        
    xdata = check_y (xdata , input_name= "Xdata")
    
    if len(xdata) != len(ydata): 
        raise ValueError(" `x` and `y` arrays must have the same length."
                        "'{len(xdata)}' and '{len(ydata)}' are given.")
        
    popt, pcov = curve_fit(func, xdata, ydata, **kws)
    ydata_new = func(xdata, *popt)
    
    if show:
        plt.plot(xdata, ydata, 'b-', label='data')
        plt.plot(xdata, func(xdata, *popt), 'r-',
             label='fit: a=%5.3f, b=%5.3f' % tuple(popt))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()
        
    return ydata_new, popt, pcov 

def detect_station_position (
        s : Union[str, int] ,
        p: _SP, 
) -> Tuple [int, float]: 
    """ Detect station position and return the index in positions
    
    :param s: str, int - Station location  in the position array. It should 
        be the positionning of the drilling location. If the value given
        is type string. It should be match the exact position to 
        locate the drilling. Otherwise, if the value given is in float or 
        integer type, it should be match the index of the position array. 
         
    :param p: Array-like - Should be the  conductive zone as array of 
        station location values. 
            
    :returns: 
        - `s_index`- the position index location in the conductive zone.  
        - `s`- the station position in distance. 
        
    :Example: 
        
        >>> import numpy as np 
        >>> from gofast.tools.mathex  import detect_station_position 
        >>> pos = np.arange(0 , 50 , 10 )
        >>> detect_station_position (s ='S30', p = pos)
        ... (3, 30.0)
        >>> detect_station_position (s ='40', p = pos)
        ... (4, 40.0)
        >>> detect_station_position (s =2, p = pos)
        ... (2, 20)
        >>> detect_station_position (s ='sta200', p = pos)
        ... WATexError_station: Station sta200 \
            is out of the range; max position = 40
    """
    s = _assert_all_types( s, float, int, str)
    
    p = check_y (p, input_name ="Position array 'p'", to_frame =True )
    
    S=copy.deepcopy(s)
    if isinstance(s, str): 
        s =s.lower().replace('s', '').replace('pk', '').replace('ta', '')
        try : 
            s=int(s)
        except : 
            raise ValueError (f'could not convert string to float: {S}')
            
    p = np.array(p, dtype = np.int32)
    dl = (p.max() - p.min() ) / (len(p) -1) 
    if isinstance(s, (int, float)): 
        if s > len(p): # consider this as the dipole length position: 
            # now let check whether the given value is module of the station 
            if s % dl !=0 : 
                raise SiteError  (
                    f'Unable to detect the station position {S}')
            elif s % dl == 0 and s <= p.max(): 
                # take the index 
                s_index = s//dl
                return int(s_index), s_index * dl 
            else : 
                raise SiteError (
                    f'Station {S} is out of the range; max position = {max(p)}'
                )
        else : 
            if s >= len(p): 
                raise SiteError (
                    'Location index must be less than the number of'
                    f' stations = {len(p)}. {s} is gotten.')
            # consider it as integer index 
            # erase the last variable
            # s_index = s 
            # s = S * dl   # find 
            return s , p[s ]
       
    # check whether the s value is in the p 
    if True in np.isin (p, s): 
        s_index ,  = np.where (p ==s ) 
        s = p [s_index]
        
    return int(s_index) , s 
    
def _manage_colors (c, default = ['ok', 'ob-', 'r-']): 
    """ Manage the ohmic-area plot colors """
    c = c or default 
    if isinstance(c, str): 
        c= [c] 
    c = list(c) +  default 
    
    return c [:3] # return 3colors 
     
@refAppender(refglossary.__doc__)
def plot_ (
    *args : List [Union [str, ArrayLike, ...]],
    fig_size: Tuple[int] = None,
    raw : bool = False, 
    style : str = 'seaborn',   
    dtype: str  ='erp',
    kind: Optional[str] = None , 
    fig_title_kws: dict=None, 
    fbtw:bool=False, 
    fig=None, 
    ax=None, 
    **kws
    ) -> None : 
    """ Quick visualization for fitting model, |ERP| and |VES| curves.
    
    :param x: array-like - array of data for x-axis representation 
    :param y: array-like - array of data for plot y-axis  representation
    :param fig_size: tuple - Matplotlib (MPL) figure size; should be a tuple 
         value of integers e.g. `figsize =(10, 5)`.
    :param raw: bool- Originally the `plot_` function is intended for the 
        fitting |ERP| model i.e. the correct value of |ERP| data. However, 
        when the `raw` is set to ``True``, it plots the both curves: The 
        fitting model as well as the uncorrected model. So both curves are 
        overlaining or supperposed.
    :param style: str - Pyplot style. Default is ``seaborn``
    :param dtype: str - Kind of data provided. Can be |ERP| data or |VES| data. 
        When the |ERP| data are provided, the common plot is sufficient to 
        visualize all the data insight i.e. the default value of `kind` is kept 
        to ``None``. However, when the data collected is |VES| data, the 
        convenient plot for visualization is the ``loglog`` for parameter
        `kind``  while the `dtype` can be set to `VES` to specify the labels 
        into the x-axis. The default value of `dtype` is ``erp`` for common 
        visualization. 
    :param kind:  str - Use to specify the kind of data provided. See the 
        explanation of `dtype` parameters. By default `kind` is set to ``None``
        i.e. its keep the normal plots. It can be ``loglog``, ``semilogx`` and 
        ``semilogy``.
        
    :param fbtw: bool, default=False, 
        Mostly used for |VES| data. If ``True``, filled the computed 
        fractured zone using the parameters computed from 
        :func:`~.gofast.tools.mathex .ohmicArea`. 
    :param fig_title_kws: dict, 
        Additional keywords argument passed in dictionnary to customize 
        the figure title. 
    :param fig: Matplotlib.pyplot.figure
        add plot on the same figure. 
        
    :param kws: dict - Additional `Matplotlib plot`_ keyword arguments. To cus-
        tomize the plot, one can provide a dictionnary of MPL keyword 
        additional arguments like the example below.
    
    :Example: 
        >>> import numpy as np 
        >>> from gofast.tools.mathex  import plot_ 
        >>> x, y = np.arange(0 , 60, 10) ,np.abs( np.random.randn (6)) 
        >>> KWS = dict (xlabel ='Stations positions', ylabel= 'resistivity(ohm.m)', 
                    rlabel = 'raw cuve', rotate = 45 ) 
        >>> plot_(x, y, '-ok', raw = True , style = 'seaborn-whitegrid', 
                  figsize = (7, 7) ,**KWS )
    
    """

    plt.style.use(style)
    # retrieve all the aggregated data from keywords arguments
    if (rlabel := kws.get('rlabel')) is not None : 
        del kws['rlabel']
    if (xlabel := kws.get('xlabel')) is not None : 
        del kws['xlabel']
    if (ylabel := kws.get('ylabel')) is not None : 
        del kws['ylabel']
    if (rotate:= kws.get ('rotate')) is not None: 
        del kws ['rotate']
    if (leg:= kws.get ('leg')) is not None: 
        del kws ['leg']
    if (show_grid:= kws.get ('show_grid')) is not None: 
        del kws ['show_grid']
    if (title:= kws.get ('title')) is not None: 
        del kws ['title']
    x , y, *args = args 
    
    if ( fig is None 
        or ax is None
        ): 
        fig, ax = plt.subplots(1,1, figsize =fig_size)
        # fig = plt.figure(1, figsize =fig_size)
    
    ax.plot (x, y,*args, 
              **kws)
    if raw: 
        kind = kind.lower(
            ) if isinstance(kind, str) else kind 
        if kind =='semilogx': 
            ax.semilogx (x, y, 
                      color = '#9EB3DD',
                      label =rlabel, 
                      )
        elif kind =='semilogy': 
            ax.semilogy (x, y, 
                      color = '#9EB3DD',
                      label =rlabel, 
                      )
        elif kind =='loglog': 
            ax.loglog (x, y, 
                      color = '#9EB3DD',
                      label =rlabel, 
                      )
        else: 
            ax.plot (x, y, 
                      color = '#9EB3DD',
                      label =rlabel, 
                      )
            
        if fbtw and dtype=='ves': 
            # remove colors 
            args = [ag for ag in args if not isinstance (ag, str)] 
            if len(args ) <4 : 
                raise TypeError ("'Fill_between' expects four arguments:"
                                " (x0, y0) for fitting plot and (x1, y1)"
                                " for ohmic area. Got {len(args)}")
            xf, yf , xo, yo,*_ = args  
            # find the index position in xf 
            ixp = list ( find_close_position (xf, xo ) ) 
            ax.fill_between(xo, yf[ixp], y2=yo  )
            
    dtype = dtype.lower() if isinstance(dtype, str) else dtype
    
    if dtype is None:
        dtype ='erp'  
    if dtype not in ('erp', 'ves'): kind ='erp' 
    
    if dtype =='erp':
        ax.set_xticks (x,
                    labels = ['S{:02}'.format(int(i)) for i in x ],
                    rotation = 0. if rotate is None else rotate 
                    )
    elif dtype =='ves': 
        ax.set_xticks (x,
                    rotation = 0. if rotate is None else rotate 
                    )
        
    ax.set_xlabel ('AB/2 (m)' if dtype=='ves' else "Stations"
                ) if xlabel is  None  else plt.xlabel (xlabel)
    ax.set_ylabel ('Resistivity (Ω.m)'
                ) if ylabel is None else plt.ylabel (ylabel)
    
    
    t0= {'erp': 'Plot Electrical Resistivity Profiling', 
         'sfi': 'Pseudo-fracturing index', 
         'ves': 'Vertical Electrical Sounding'
         }

    fig_title_kws = fig_title_kws or dict (
            t = t0.get( dtype) or  title, 
            style ='italic', 
            bbox =dict(boxstyle='round',facecolor ='lightgrey'))
        
    if len(x) >= 20: 
        for kk, label in enumerate ( ax.xaxis.get_ticklabels()) :
            if kk% 10 ==0: 
               label.set_visible(True) 
            else: label.set_visible(False) 
            
 
    if show_grid is not None: 
        # plt.minorticks_on()
        ax.grid (visible =True, which='both')
    plt.tight_layout()
    fig.suptitle(**fig_title_kws)
    plt.legend (leg, loc ='best') if leg  else plt.legend ()
    plt.show ()
   
def quickplot (arr: ArrayLike | List[float], dl:float  =10)-> None: 
    """Quick plot to see the anomaly"""
    
    plt.plot(np.arange(0, len(arr) * dl, dl), arr , ls ='-', c='k')
    plt.show() 
 

def convert_distance_to_m(
        value:_T ,
        converter:float =1e3,
        unit:str ='km'
)-> float: 
    """ Convert distance from `km` to `m` or vice versa even a string 
    value is given.
    
    :param value: value to convert. 
    :paramm converter: Equivalent if given in ``km`` rather than ``m``.
    :param unit: unit to convert to.
    
    """
    
    if isinstance(value, str): 
        try:
            value = float(value.replace(unit, '')
                              )*converter if value.find(
                'km')>=0 else float(value.replace('m', ''))
        except: 
            raise TypeError(f"Expected float not {type(value)!r}."
               )
            
    return value
       
def get_station_number (
        dipole:float,
        distance:float , 
        from0:bool = False,
        **kws
)-> float: 
    """ Get the station number from dipole length and 
    the distance to the station.
    
    :param distance: Is the distance from the first station to `s` in 
        meter (m). If value is given, please specify the dipole length in 
        the same unit as `distance`.
    :param dipole: Is the distance of the dipole measurement. 
        By default the dipole length is in meter.
    :param kws: :func:`convert_distance_to_m` additional arguments
    
    """
    dipole=convert_distance_to_m(dipole, **kws)
    distance =convert_distance_to_m(distance, **kws)

    return  distance/dipole  if from0 else distance/dipole + 1 
    
#FR0: #CED9EF # (206, 217, 239)
#FR1: #9EB3DD # (158, 179, 221)
#FR2: #3B70F2 # (59, 112, 242) #repl rgb(52, 54, 99)
#FR3: #0A4CEE # (10, 76, 238)

def scale_y(
        y: ArrayLike , 
        x: ArrayLike =None, 
        deg: int = None,  
        func:_F =None
        )-> Tuple[ArrayLike, ArrayLike, _F]: 
    """ Scaling value using a fitting curve. 
    
    Create polyfit function from a specifc data points `x` to correct `y` 
    values.  
    
    :param y: array-like of y-axis. Is the array of value to be scaled. 
    
    :param x: array-like of x-axis. If `x` is given, it should be the same 
        length as `y`, otherwise and error will occurs. Default is ``None``. 
    
    :param func: callable - The model function, ``f(x, ...)``. It must take 
        the independent variable as the first argument and the parameters
        to fit as separate remaining arguments.  `func` can be a ``linear``
        function i.e  for ``f(x)= ax +b`` where `a` is slope and `b` is the 
        intercept value. It is recommended according to the `y` value 
        distribution to set up  a custom function for better fitting. If `func`
        is given, the `deg` is not needed.   
        
    :param deg: polynomial degree. If  value is ``None``, it should  be 
        computed using the length of extrema (local and/or global) values.
 
    :returns: 
        - y: array scaled - projected sample values got from `f`.
        - x: new x-axis - new axis  `x_new` generated from the samples.
        - linear of polynomial function `f` 
        
    :references: 
        Wikipedia, Curve fitting, https://en.wikipedia.org/wiki/Curve_fitting
        Wikipedia, Polynomial interpolation, https://en.wikipedia.org/wiki/Polynomial_interpolation
    :Example: 
        >>> import numpy as np 
        >>> import matplotlib.pyplot as plt 
        >>> from gofast.exmath import scale_values 
        >>> rdn = np.random.RandomState(42) 
        >>> x0 =10 * rdn.rand(50)
        >>> y = 2 * x0  +  rnd.randn(50) -1
        >>> plt.scatter(x0, y)
        >>> yc, x , f = scale_values(y) 
        >>> plt.plot(x, y, x, yc) 
        
    """   
    y = check_y( y )
    
    if str(func).lower() != 'none': 
        if not hasattr(func, '__call__') or not inspect.isfunction (func): 
            raise TypeError(
                f'`func` argument is a callable not {type(func).__name__!r}')

    # get the number of local minimum to approximate degree. 
    minl, = argrelextrema(y, np.less) 
    # get the number of degrees
    degree = len(minl) + 1
    if x is None: 
        x = np.arange(len(y)) # np.linspace(0, 4, len(y))
        
    x= check_y (x , input_name="x") 
    
    if len(x) != len(y): 
        raise ValueError(" `x` and `y` arrays must have the same length."
                        f"'{len(x)}' and '{len(y)}' are given.")
        
    coeff = np.polyfit(x, y, int(deg) if deg is not None else degree)
    f = np.poly1d(coeff) if func is  None else func 
    yc = f (x ) # corrected value of y 

    return  yc, x ,  f  

def smooth1d(
    ar, /, 
    drop_outliers:bool=True, 
    ma:bool=True, 
    absolute:bool=False,
    interpolate:bool=False, 
    view:bool=False , 
    x: ArrayLike=None, 
    xlabel:str =None, 
    ylabel:str =None, 
    fig_size:tuple = ( 10, 5) 
    )-> ArrayLike[float]: 
    """ Smooth one-dimensional array. 
    
    Parameters 
    -----------
    ar: ArrayLike 1d 
       Array of one-dimensional 
       
    drop_outliers: bool, default=True 
       Remove the outliers in the data before smoothing 
       
    ma: bool, default=True, 
       Use the moving average for smoothing array value. This seems more 
       realistic.
       
    interpolate: bool, default=False 
       Interpolate value to fit the original data size after NaN filling. 
       
       .. versionadded:: 0.2.8 
       
    absolute: bool, default=False, 
       keep postive the extrapolated scaled values. Indeed, when scaling data, 
       negative value can be appear due to the polyfit function. to absolute 
       this value, set ``absolute=True``. Note that converting to values to 
       positive must be considered as the last option when values in the 
       array must be positive.
       
    view: bool, default =False 
       Display curves 
    x: ArrayLike, optional 
       Abscissa array for visualization. If given, it must be consistent 
       with the given array `ar`. Raises error otherwise. 
    xlabel: str, optional 
       Label of x 
    ylabel:str, optional 
       label of y  
    fig_size: tuple , default=(10, 5)
       Matplotlib figure size
       
    Returns 
    --------
    yc: ArrayLike 
       Smoothed array value. 
       
    Examples 
    ---------
    >>> import numpy as np 
    >>> from gofast.tools.mathex  import smooth1d 
    >>> # add Guassian Noise 
    >>> np.random.seed (42)
    >>> ar = np.random.randn (20 ) * 20 + np.random.normal ( 20 )
    >>> ar [:7 ]
    array([6.42891445e+00, 3.75072493e-02, 1.82905357e+01, 2.92957265e+01,
           6.20589038e+01, 2.26399535e+01, 1.12596434e+01])
    >>> arc = smooth1d (ar, view =True , ma =False )
    >>> arc [:7 ]
    array([12.08603102, 15.29819907, 18.017749  , 20.27968322, 22.11900412,
           23.5707141 , 24.66981557])
    >>> arc = smooth1d (ar, view =True )# ma=True by default 
    array([ 5.0071604 ,  5.90839339,  9.6264018 , 13.94679804, 17.67369252,
           20.34922943, 22.00836725])
    """
    # convert data into an iterable object 
    ar = np.array(
        is_iterable(ar, exclude_string = True , transform =True )) 
    
    if not _is_arraylike_1d(ar): 
        raise TypeError("Expect one-dimensional array. Use `gofast.smoothing`"
                        " for handling two-dimensional array.")
    if not _is_numeric_dtype(ar): 
        raise ValueError (f"{ar.dtype.name!r} is not allowed. Expect a numeric"
                          " array")
        
    arr = ar.copy() 
    if drop_outliers: 
        arr = remove_outliers( 
            arr, fill_value = np.nan , interpolate = interpolate )
    # Nan is not allow so fill NaN if exists in array 
    # is arraylike 1d 
    if not interpolate:
        # fill NaN 
        arr = reshape ( fillNaN( arr , method ='both') ) 
    if ma: 
        arr = moving_average(arr, method ='sma')
    # if extrapolation give negative  values
    # whether to keep as it was or convert to positive values. 
    # note that converting to positive values is 
    arr, *_  = scale_y ( arr ) 
    # if extrapolation gives negative values
    # convert to positive values or keep it intact. 
    # note that converting to positive values is 
    # can be used as the last option when array 
    # data must be positive.
    if absolute: 
        arr = np.abs (arr )
    if view: 
        x = np.arange ( len(ar )) if x is None else np.array (x )

        check_consistency_size( x, ar )
            
        fig,  ax = plt.subplots (1, 1, figsize = fig_size)
        ax.plot (x, 
                 ar , 
                 'ok-', 
                 label ='raw curve'
                 )
        ax.plot (x, 
                 arr, 
                 c='#0A4CEE',
                 marker = 'o', 
                 label ='smooth curve'
                 ) 
        
        ax.legend ( ) 
        ax.set_xlabel (xlabel or '')
        ax.set_ylabel ( ylabel or '') 
        
    return arr 

def smoothing (
    ar, /, 
    drop_outliers = True ,
    ma=True,
    absolute =False,
    interpolate=False, 
    axis = 0, 
    view = False, 
    fig_size =(7, 7), 
    xlabel =None, 
    ylabel =None , 
    cmap ='binary'
    ): 
    """ Smooth data along axis. 
    
    Parameters 
    -----------
    ar: ArrayLike 1d or 2d 
       One dimensional or two dimensional array. 
       
    drop_outliers: bool, default=True 
       Remove the outliers in the data before smoothing along the given axis 
       
    ma: bool, default=True, 
       Use the moving average for smoothing array value along axis. This seems 
       more realistic rather than using only the scaling method. 
       
    absolute: bool, default=False, 
       keep positive the extrapolated scaled values. Indeed, when scaling data, 
       negative value can be appear due to the polyfit function. to absolute 
       this value, set ``absolute=True``. Note that converting to values to 
       positive must be considered as the last option when values in the 
       array must be positive.
       
    axis: int, default=0 
       Axis along with the data must be smoothed. The default is the along  
       the row. 
       
    view: bool, default =False 
       Visualize the two dimensional raw and smoothing grid. 
       
    xlabel: str, optional 
       Label of x 
       
    ylabel:str, optional 
    
       label of y  
    fig_size: tuple , default=(7, 5)
       Matplotlib figure size 
       
    cmap: str, default='binary'
       Matplotlib.colormap to manage the `view` color 
      
    Return 
    --------
    arr0: ArrayLike 
       Smoothed array value. 
    
    Examples 
    ---------
    >>> import numpy as np 
    >>> from gofast.tools.mathex  import smoothing
    >>> # add Guassian Noises 
    >>> np.random.seed (42)
    >>> ar = np.random.randn (20, 7 ) * 20 + np.random.normal ( 20, 7 )
    >>> ar [:3, :3 ]
    array([[ 31.5265026 ,  18.82693352,  34.5459903 ],
           [ 36.94091413,  12.20273182,  32.44342041],
           [-12.90613711,  10.34646896,   1.33559714]])
    >>> arc = smoothing (ar, view =True , ma =False )
    >>> arc [:3, :3 ]
    array([[32.20356863, 17.18624398, 41.22258603],
           [33.46353806, 15.56839464, 19.20963317],
           [23.22466498, 13.8985316 ,  5.04748584]])
    >>> arcma = smoothing (ar, view =True )# ma=True by default
    >>> arcma [:3, :3 ]
    array([[23.96547827,  8.48064226, 31.81490918],
           [26.21374675, 13.33233065, 12.29345026],
           [22.60143346, 16.77242118,  2.07931194]])
    >>> arcma_1 = smoothing (ar, view =True, axis =1 )
    >>> arcma_1 [:3, :3 ]
    array([[18.74017857, 26.91532187, 32.02914421],
           [18.4056216 , 21.81293014, 21.98535213],
           [-1.44359989,  3.49228057,  7.51734762]])
    """
    ar = np.array ( 
        is_iterable(ar, exclude_string = True , transform =True )
        ) 
    if ( 
            str (axis).lower().find('1')>=0 
            or str(axis).lower().find('column')>=0
            ): 
        axis = 1 
    else : axis =0 
    
    if _is_arraylike_1d(ar): 
        ar = reshape ( ar, axis = 0 ) 
    # make a copy
    arr = ar.copy() 
    along_axis = arr.shape [1] if axis == 0 else len(ar) 
    arr0 = np.zeros_like (arr)
    for ix in range (along_axis): 
        value = arr [:, ix ] if axis ==0 else arr[ix , :]
        yc = smooth1d(value, drop_outliers = drop_outliers , 
                      ma= ma, view =False , absolute =absolute , 
                      interpolate= interpolate, 
                      ) 
        if axis ==0: 
            arr0[:, ix ] = yc 
        else : arr0[ix, :] = yc 
        
    if view: 
        fig, ax  = plt.subplots (nrows = 1, ncols = 2 , sharey= True,
                                 figsize = fig_size )
        ax[0].imshow(arr ,interpolation='nearest', label ='Raw Grid', 
                     cmap = cmap )
        ax[1].imshow (arr0, interpolation ='nearest', label = 'Smooth Grid', 
                      cmap =cmap  )
        
        ax[0].set_title ('Raw Grid') 
        ax[0].set_xlabel (xlabel or '')
        ax[0].set_ylabel ( ylabel or '')
        ax[1].set_title ('Smooth Grid') 
        ax[1].set_xlabel (xlabel or '')
        ax[1].set_ylabel ( ylabel or '')
        plt.legend
        plt.show () 
        
    if 1 in ar.shape: 
        arr0 = reshape (arr0 )
        
    return arr0 
    
  
def interpolate1d (
        arr:ArrayLike[DType[_T]], 
        kind:str = 'slinear', 
        method:str=None, 
        order:Optional[int] = None, 
        fill_value:str ='extrapolate',
        limit:Tuple[float] =None, 
        **kws
    )-> ArrayLike[DType[_T]]:
    """ Interpolate array containing invalid values `NaN`
    
    Usefull function to interpolate the missing frequency values in the 
    tensor components. 
    
    Parameters 
    ----------
    arr: array_like 
        Array to interpolate containg invalid values. The invalid value here 
        is `NaN`. 
        
    kind: str or int, optional
        Specifies the kind of interpolation as a string or as an integer 
        specifying the order of the spline interpolator to use. The string 
        has to be one of ``linear``, ``nearest``, ``nearest-up``, ``zero``, 
        ``slinear``,``quadratic``, ``cubic``, ``previous``, or ``next``. 
        ``zero``, ``slinear``, ``quadratic``and ``cubic`` refer to a spline 
        interpolation of zeroth, first, second or third order; ``previous`` 
        and ``next`` simply return the previous or next value of the point; 
        ``nearest-up`` and ``nearest`` differ when interpolating half-integers 
        (e.g. 0.5, 1.5) in that ``nearest-up`` rounds up and ``nearest`` rounds 
        down. If `method` param is set to ``pd`` which refers to pd.interpolate 
        method , `kind` can be set to ``polynomial`` or ``pad`` interpolation. 
        Note that the polynomial requires you to specify an `order` while 
        ``pad`` requires to specify the `limit`. Default is ``slinear``.
        
    method: str, optional, default='mean' 
        Method of interpolation. Can be ``base`` for `scipy.interpolate.interp1d`
        ``mean`` or ``bff`` for scaling methods and ``pd``for pandas interpolation 
        methods. Note that the first method is fast and efficient when the number 
        of NaN in the array if relatively few. It is less accurate to use the 
        `base` interpolation when the data is composed of many missing values.
        Alternatively, the scaled method(the  second one) is proposed to be the 
        alternative way more efficient. Indeed, when ``mean`` argument is set, 
        function replaces the NaN values by the nonzeros in the raw array and 
        then uses the mean to fit the data. The result of fitting creates a smooth 
        curve where the index of each NaN in the raw array is replaced by its 
        corresponding values in the fit results. The same approach is used for
        ``bff`` method. Conversely, rather than averaging the nonzeros values, 
        it uses the backward and forward strategy  to fill the NaN before scaling.
        ``mean`` and ``bff`` are more efficient when the data are composed of 
        lot of missing values. When the interpolation `method` is set to `pd`, 
        function uses the pandas interpolation but ended the interpolation with 
        forward/backward NaN filling since the interpolation with pandas does
        not deal with all NaN at the begining or at the end of the array. Default 
        is ``base``.
        
    fill_value: array-like or (array-like, array_like) or ``extrapolate``, optional
        If a ndarray (or float), this value will be used to fill in for requested
        points outside of the data range. If not provided, then the default is
        NaN. The array-like must broadcast properly to the dimensions of the 
        non-interpolation axes.
        If a two-element tuple, then the first element is used as a fill value
        for x_new < x[0] and the second element is used for x_new > x[-1]. 
        Anything that is not a 2-element tuple (e.g., list or ndarray,
        regardless of shape) is taken to be a single array-like argument meant 
        to be used for both bounds as below, above = fill_value, fill_value.
        Using a two-element tuple or ndarray requires bounds_error=False.
        Default is ``extrapolate``. 
        
    kws: dict 
        Additional keyword arguments from :class:`spi.interp1d`. 
    
    Returns 
    -------
    array like - New interpoolated array. `NaN` values are interpolated. 
    
    Notes 
    ----- 
    When interpolated thoughout the complete frequencies  i.e all the frequency 
    values using the ``base`` method, the missing data in `arr`  can be out of 
    the `arr` range. So, for consistency and keep all values into the range of 
    frequency, the better idea is to set the param `fill_value` in kws argument
    of ``spi.interp1d`` to `extrapolate`. This will avoid an error to raise when 
    the value to  interpolated is extra-bound of `arr`. 
    
    
    References 
    ----------
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
    https://www.askpython.com/python/examples/interpolation-to-fill-missing-entries
    
    Examples 
    --------
    >>> import numpy as np 
    >>> import matplotlib.pyplot as plt 
    >>> from gofast.tools.mathex   import interpolate1d,
    >>> z = np.random.randn(17) *10 # assume 17 freq for 17 values of tensor Z 
    >>> z [[7, 10, 16]] =np.nan # replace some indexes by NaN values 
    >>> zit = interpolate1d (z, kind ='linear')
    >>> z 
    ... array([ -1.97732415, -16.5883156 ,   8.44484348,   0.24032979,
              8.30863276,   4.76437029, -15.45780568,          nan,
             -4.11301794, -10.94003412,          nan,   9.22228383,
            -15.40298253,  -7.24575491,  -7.15149205, -20.9592011 ,
                     nan]),
    >>> zn 
    ...array([ -1.97732415, -16.5883156 ,   8.44484348,   0.24032979,
             8.30863276,   4.76437029, -15.45780568,  -4.11301794,
           -10.94003412,   9.22228383, -15.40298253,  -7.24575491,
            -7.15149205, -20.9592011 , -34.76691014, -48.57461918,
           -62.38232823])
    >>> zmean = interpolate1d (z,  method ='mean')
    >>> zbff = interpolate1d (z, method ='bff')
    >>> zpd = interpolate1d (z,  method ='pd')
    >>> plt.plot( np.arange (len(z)),  zit, 'v--', 
              np.arange (len(z)), zmean, 'ok-',
              np.arange (len(z)), zbff, '^g:',
              np.arange (len(z)), zpd,'<b:', 
              np.arange (len(z)), z,'o', 
              )
    >>> plt.legend(['interp1d', 'mean strategy', 'bff strategy',
                    'pandas strategy', 'data'], loc='best')
    
    """
    method = method or 'mean'; method =str(method).strip().lower() 
    if method in ('pandas', 'pd', 'series', 'dataframe','df'): 
        method = 'pd' 
    elif method in ('interp1d', 'scipy', 'base', 'simpler', 'i1d'): 
        method ='base' 
    
    if not hasattr (arr, '__complex__'): 
        
        arr = check_y(arr, allow_nan= True, to_frame= True ) 
    # check whether there is nan and masked invalid 
    # and take only the valid values 
    t_arr = arr.copy() 
    
    if method =='base':
        mask = ~np.ma.masked_invalid(arr).mask  
        arr = arr[mask] # keep the valid values
        f = spi.interp1d( x= np.arange(len(arr)), y= arr, kind =kind, 
                         fill_value =fill_value, **kws) 
        arr_new = f(np.arange(len(t_arr)))
        
    if method in ('mean', 'bff'): 
        arr_new = arr.copy()
        
        if method =='mean': 
            # use the mean of the valid value
            # and fill the nan value
            mean = t_arr[~np.isnan(t_arr)].mean()  
            t_arr[np.isnan(t_arr)]= mean  
            
        if method =='bff':
            # fill NaN values back and forward.
            t_arr = fillNaN(t_arr, method = method)
            t_arr= reshape(t_arr)
            
        yc, *_= scale_y (t_arr)
        # replace the at NaN positions value in  t_arr 
        # with their corresponding scaled values 
        arr_new [np.isnan(arr_new)]= yc[np.isnan(arr_new)]
        
    if method =='pd': 
        t_arr= pd.Series (t_arr, dtype = t_arr.dtype )
        t_arr = np.array(t_arr.interpolate(
            method =kind, order=order, limit = limit ))
        arr_new = reshape(fillNaN(t_arr, method= 'bff')) # for consistency 
        
    return arr_new 
   

def moving_average (
    y:ArrayLike[DType[_T]],
    *, 
    window_size:int  = 3 , 
    method:str  ='sma',
    mode:str  ='same', 
    alpha: int  =.5 
)-> ArrayLike[DType[_T]]: 
    """ A moving average is  used with time series data to smooth out
    short-term fluctuations and highlight longer-term trends or cycles.
    
    Funtion analyzes data points by creating a series of averages of different
    subsets of the full data set. 
    
    Parameters 
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
        
    window_size : int
        the length of the window. Must be greater than 1 and preferably
        an odd integer number.Default is ``3``
        
    method: str 
        variant of moving-average. Can be ``sma``, ``cma``, ``wma`` and ``ema`` 
        for simple, cummulative, weight and exponential moving average. Default 
        is ``sma``. 
        
    mode: str
        returns the convolution at each point of overlap, with an output shape
        of (N+M-1,). At the end-points of the convolution, the signals do not 
        overlap completely, and boundary effects may be seen. Can be ``full``,
        ``same`` and ``valid``. See :doc:`~np.convole` for more details. Default 
        is ``same``. 
        
    alpha: float, 
        smoothing factor. Only uses in exponential moving-average. Default is 
        ``.5``.
    
    Returns 
    --------
    ya: array like, shape (N,) 
        Averaged time history of the signal
    
    Notes 
    -------
    The first element of the moving average is obtained by taking the average 
    of the initial fixed subset of the number series. Then the subset is
    modified by "shifting forward"; that is, excluding the first number of the
    series and including the next value in the subset.
    
    Examples
    --------- 
    >>> import numpy as np ; import matplotlib.pyplot as plt 
    >>> from gofast.tools.mathex   import moving_average 
    >>> data = np.random.randn (37) 
    >>> # add gaussion noise to the data 
    >>> data = 2 * np.sin( data)  + np.random.normal (0, 1 , len(data))
    >>> window = 5  # fixed size to 5 
    >>> sma = moving_average(data, window) 
    >>> cma = moving_average(data, window, method ='cma' )
    >>> wma = moving_average(data, window, method ='wma' )
    >>> ema = moving_average(data, window, method ='ema' , alpha =0.6)
    >>> x = np.arange(len(data))
    >>> plt.plot (x, data, 'o', x, sma , 'ok--', x, cma, 'g-.', x, wma, 'b:')
    >>> plt.legend (['data', 'sma', 'cma', 'wma'])
    
    References 
    ----------
    .. * [1] https://en.wikipedia.org/wiki/Moving_average
    .. * [2] https://www.sciencedirect.com/topics/engineering/hanning-window
    .. * [3] https://stackoverflow.com/questions/12816011/weighted-moving-average-with-numpy-convolve
    
    """
    y = np.array(y)
    try:
        window_size = np.abs(_assert_all_types(int(window_size), int))
    except ValueError:
        raise ValueError("window_size has to be of type int")
    if window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if  window_size > len(y):
        raise TypeError("window_size is too large for averaging. Window"
                        f" must be greater than 0 and less than {len(y)}")
    
    method =str(method).lower().strip().replace ('-', ' ') 
    
    if method in ('simple moving average',
                  'simple', 'sma'): 
        method = 'sma' 
    elif method  in ('cumulative average', 
                     'cumulative', 'cma'): 
        method ='cma' 
    elif method  in ('weighted moving average',
                     'weight', 'wma'): 
        method = 'wma'
    elif method in('exponential moving average',
                   'exponential', 'ema'):
        method = 'ema'
    else : 
        raise ValueError ("Variant average methods only includes "
                          f" {smart_format(['sma', 'cma', 'wma', 'ema'], 'or')}")
    if  1. <= alpha <= 0 : 
        raise ValueError ('alpha should be less than 1. and greater than 0. ')
        
    if method =='sma': 
        ya = np.convolve(y , np.ones (window_size), mode ) / window_size 
        
    if method =='cma': 
        y = np.cumsum (y) 
        ya = np.array([ y[ii]/ len(y[:ii +1]) for ii in range(len(y))]) 
        
    if method =='wma': 
        w = np.cumsum(np.ones(window_size, dtype = float))
        w /= np.sum(w)
        ya = np.convolve(y, w[::-1], mode ) #/window_size
        
    if method =='ema': 
        ya = np.array ([y[0]]) 
        for ii in range(1, len(y)): 
            v = y[ii] * alpha + ( 1- alpha ) * ya[-1]
            ya = np.append(ya, v)
            
    return ya 


def get_profile_angle (
        easting: float =None, northing: float =None, msg:str ="ignore" ): 
    """
    compute geoprofile angle. 
    Parameters 
    -----------
    * easting : array_like 
            easting coordiantes values 
    * northing : array_like 
            northing coordinates values
    * msg: output a little message if msg is set to "raises"
    
    Returns 
    ---------
    float
         profile_angle 
    float 
        geo_electric_strike 
    """
    msg = (
        "Need to import scipy.stats as a single module. Sometimes import scipy "
        "differently  with stats may not work. Use either `import scipy.stats`"
        " rather than `import scipy as sp`" 
        )
    
    if easting is None or northing is None : 
        raise TypeError('NoneType can not be computed !')
        
        # use the one with the lower standard deviation
    try :
        easting = easting.astype('float')
        northing = northing.astype('float')
    except : 
        raise ValueError('Could not convert input argument to float!')
    try : 
        profile1 = spstats.linregress(easting, northing)
        profile2 =spstats.linregress(northing, easting)
    except:
        warnings.warn(msg)
        
    profile_line = profile1[:2]
    # if the profile is rather E=E(N),
    # the parameters have to converted  into N=N(E) form:
    
    if profile2[4] < profile1[4]:
        profile_line = (1. / profile2[0], -profile2[1] / profile2[0])

    # if self.profile_angle is None:
    profile_angle = (90 - (np.arctan(profile_line[0]) * 180 / np.pi)) % 180
    
    # otherwise: # have 90 degree ambiguity in 
    #strike determination# choose strike which offers larger
    #  angle with profile if profile azimuth is in [0,90].
    if msg=="raises": 
        print("+++ -> Profile angle is {0:+.2f} degrees E of N".format(
                profile_angle
                ) )
    return np.around( profile_angle,2)

        
def savgol_coeffs(window_length, polyorder, deriv=0, delta=1.0, pos=None,
                  use="conv"):
    """Compute the coefficients for a 1-D Savitzky-Golay FIR filter.

    Parameters
    ----------
    window_length : int
        The length of the filter window (i.e., the number of coefficients).
        `window_length` must be an odd positive integer.
    polyorder : int
        The order of the polynomial used to fit the samples.
        `polyorder` must be less than `window_length`.
    deriv : int, optional
        The order of the derivative to compute. This must be a
        nonnegative integer. The default is 0, which means to filter
        the data without differentiating.
    delta : float, optional
        The spacing of the samples to which the filter will be applied.
        This is only used if deriv > 0.
    pos : int or None, optional
        If pos is not None, it specifies evaluation position within the
        window. The default is the middle of the window.
    use : str, optional
        Either 'conv' or 'dot'. This argument chooses the order of the
        coefficients. The default is 'conv', which means that the
        coefficients are ordered to be used in a convolution. With
        use='dot', the order is reversed, so the filter is applied by
        dotting the coefficients with the data set.

    Returns
    -------
    coeffs : 1-D ndarray
        The filter coefficients.

    References
    ----------
    A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of Data by
    Simplified Least Squares Procedures. Analytical Chemistry, 1964, 36 (8),
    pp 1627-1639.

    See Also
    --------
    savgol_filter

    Examples
    --------
    >>> from gofast.exmath.signal import savgol_coeffs
    >>> savgol_coeffs(5, 2)
    array([-0.08571429,  0.34285714,  0.48571429,  0.34285714, -0.08571429])
    >>> savgol_coeffs(5, 2, deriv=1)
    array([ 2.00000000e-01,  1.00000000e-01,  2.07548111e-16, -1.00000000e-01,
           -2.00000000e-01])

    Note that use='dot' simply reverses the coefficients.

    >>> savgol_coeffs(5, 2, pos=3)
    array([ 0.25714286,  0.37142857,  0.34285714,  0.17142857, -0.14285714])
    >>> savgol_coeffs(5, 2, pos=3, use='dot')
    array([-0.14285714,  0.17142857,  0.34285714,  0.37142857,  0.25714286])

    `x` contains data from the parabola x = t**2, sampled at
    t = -1, 0, 1, 2, 3.  `c` holds the coefficients that will compute the
    derivative at the last position.  When dotted with `x` the result should
    be 6.

    >>> x = np.array([1, 0, 1, 4, 9])
    >>> c = savgol_coeffs(5, 2, pos=4, deriv=1, use='dot')
    >>> c.dot(x)
    6.0
    """

    # An alternative method for finding the coefficients when deriv=0 is
    #    t = np.arange(window_length)
    #    unit = (t == pos).astype(int)
    #    coeffs = np.polyval(np.polyfit(t, unit, polyorder), t)
    # The method implemented here is faster.

    # To recreate the table of sample coefficients shown in the chapter on
    # the Savitzy-Golay filter in the Numerical Recipes book, use
    #    window_length = nL + nR + 1
    #    pos = nL + 1
    #    c = savgol_coeffs(window_length, M, pos=pos, use='dot')

    if polyorder >= window_length:
        raise ValueError("polyorder must be less than window_length.")

    halflen, rem = divmod(window_length, 2)

    if rem == 0:
        raise ValueError("window_length must be odd.")

    if pos is None:
        pos = halflen

    if not (0 <= pos < window_length):
        raise ValueError("pos must be nonnegative and less than "
                         "window_length.")

    if use not in ['conv', 'dot']:
        raise ValueError("`use` must be 'conv' or 'dot'")

    if deriv > polyorder:
        coeffs = np.zeros(window_length)
        return coeffs

    # Form the design matrix A. The columns of A are powers of the integers
    # from -pos to window_length - pos - 1. The powers (i.e., rows) range
    # from 0 to polyorder. (That is, A is a vandermonde matrix, but not
    # necessarily square.)
    x = np.arange(-pos, window_length - pos, dtype=float)
    if use == "conv":
        # Reverse so that result can be used in a convolution.
        x = x[::-1]

    order = np.arange(polyorder + 1).reshape(-1, 1)
    A = x ** order

    # y determines which order derivative is returned.
    y = np.zeros(polyorder + 1)
    # The coefficient assigned to y[deriv] scales the result to take into
    # account the order of the derivative and the sample spacing.
    y[deriv] = float_factorial(deriv) / (delta ** deriv)

    # Find the least-squares solution of A*c = y
    coeffs, _, _, _ = lstsq(A, y)

    return coeffs


def _polyder(p, m):
    """Differentiate polynomials represented with coefficients.

    p must be a 1-D or 2-D array.  In the 2-D case, each column gives
    the coefficients of a polynomial; the first row holds the coefficients
    associated with the highest power. m must be a nonnegative integer.
    (numpy.polyder doesn't handle the 2-D case.)
    """

    if m == 0:
        result = p
    else:
        n = len(p)
        if n <= m:
            result = np.zeros_like(p[:1, ...])
        else:
            dp = p[:-m].copy()
            for k in range(m):
                rng = np.arange(n - k - 1, m - k - 1, -1)
                dp *= rng.reshape((n - m,) + (1,) * (p.ndim - 1))
            result = dp
    return result


def _fit_edge(x, window_start, window_stop, interp_start, interp_stop,
              axis, polyorder, deriv, delta, y):
    """
    Given an N-d array `x` and the specification of a slice of `x` from
    `window_start` to `window_stop` along `axis`, create an interpolating
    polynomial of each 1-D slice, and evaluate that polynomial in the slice
    from `interp_start` to `interp_stop`. Put the result into the
    corresponding slice of `y`.
    """

    # Get the edge into a (window_length, -1) array.
    x_edge = axis_slice(x, start=window_start, stop=window_stop, axis=axis)
    if axis == 0 or axis == -x.ndim:
        xx_edge = x_edge
        swapped = False
    else:
        xx_edge = x_edge.swapaxes(axis, 0)
        swapped = True
    xx_edge = xx_edge.reshape(xx_edge.shape[0], -1)

    # Fit the edges.  poly_coeffs has shape (polyorder + 1, -1),
    # where '-1' is the same as in xx_edge.
    poly_coeffs = np.polyfit(np.arange(0, window_stop - window_start),
                             xx_edge, polyorder)

    if deriv > 0:
        poly_coeffs = _polyder(poly_coeffs, deriv)

    # Compute the interpolated values for the edge.
    i = np.arange(interp_start - window_start, interp_stop - window_start)
    values = np.polyval(poly_coeffs, i.reshape(-1, 1)) / (delta ** deriv)

    # Now put the values into the appropriate slice of y.
    # First reshape values to match y.
    shp = list(y.shape)
    shp[0], shp[axis] = shp[axis], shp[0]
    values = values.reshape(interp_stop - interp_start, *shp[1:])
    if swapped:
        values = values.swapaxes(0, axis)
    # Get a view of the data to be replaced by values.
    y_edge = axis_slice(y, start=interp_start, stop=interp_stop, axis=axis)
    y_edge[...] = values

def _fit_edges_polyfit(x, window_length, polyorder, deriv, delta, axis, y):
    """
    Use polynomial interpolation of x at the low and high ends of the axis
    to fill in the halflen values in y.

    This function just calls _fit_edge twice, once for each end of the axis.
    """
    halflen = window_length // 2
    _fit_edge(x, 0, window_length, 0, halflen, axis,
              polyorder, deriv, delta, y)
    n = x.shape[axis]
    _fit_edge(x, n - window_length, n, n - halflen, n, axis,
              polyorder, deriv, delta, y)

def savgol_filter(x, window_length, polyorder, deriv=0, delta=1.0,
                  axis=-1, mode='interp', cval=0.0):
    """ Apply a Savitzky-Golay filter to an array.

    This is a 1-D filter. If `x`  has dimension greater than 1, `axis`
    determines the axis along which the filter is applied.

    Parameters
    ----------
    x : array_like
        The data to be filtered. If `x` is not a single or double precision
        floating point array, it will be converted to type ``numpy.float64``
        before filtering.
    window_length : int
        The length of the filter window (i.e., the number of coefficients).
        `window_length` must be a positive odd integer. If `mode` is 'interp',
        `window_length` must be less than or equal to the size of `x`.
    polyorder : int
        The order of the polynomial used to fit the samples.
        `polyorder` must be less than `window_length`.
    deriv : int, optional
        The order of the derivative to compute. This must be a
        nonnegative integer. The default is 0, which means to filter
        the data without differentiating.
    delta : float, optional
        The spacing of the samples to which the filter will be applied.
        This is only used if deriv > 0. Default is 1.0.
    axis : int, optional
        The axis of the array `x` along which the filter is to be applied.
        Default is -1.
    mode : str, optional
        Must be 'mirror', 'constant', 'nearest', 'wrap' or 'interp'. This
        determines the type of extension to use for the padded signal to
        which the filter is applied.  When `mode` is 'constant', the padding
        value is given by `cval`.  See the Notes for more details on 'mirror',
        'constant', 'wrap', and 'nearest'.
        When the 'interp' mode is selected (the default), no extension
        is used.  Instead, a degree `polyorder` polynomial is fit to the
        last `window_length` values of the edges, and this polynomial is
        used to evaluate the last `window_length // 2` output values.
    cval : scalar, optional
        Value to fill past the edges of the input if `mode` is 'constant'.
        Default is 0.0.

    Returns
    -------
    y : ndarray, same shape as `x`
        The filtered data.

    See Also
    --------
    savgol_coeffs

    Notes
    -----
    Details on the `mode` options:

        'mirror':
            Repeats the values at the edges in reverse order. The value
            closest to the edge is not included.
        'nearest':
            The extension contains the nearest input value.
        'constant':
            The extension contains the value given by the `cval` argument.
        'wrap':
            The extension contains the values from the other end of the array.

    For example, if the input is [1, 2, 3, 4, 5, 6, 7, 8], and
    `window_length` is 7, the following shows the extended data for
    the various `mode` options (assuming `cval` is 0)::

        mode       |   Ext   |         Input          |   Ext
        -----------+---------+------------------------+---------
        'mirror'   | 4  3  2 | 1  2  3  4  5  6  7  8 | 7  6  5
        'nearest'  | 1  1  1 | 1  2  3  4  5  6  7  8 | 8  8  8
        'constant' | 0  0  0 | 1  2  3  4  5  6  7  8 | 0  0  0
        'wrap'     | 6  7  8 | 1  2  3  4  5  6  7  8 | 1  2  3

    .. versionadded:: 0.14.0

    Examples
    --------
    >>> from gofast.tools.mathex  import savgol_filter
    >>> np.set_printoptions(precision=2)  # For compact display.
    >>> x = np.array([2, 2, 5, 2, 1, 0, 1, 4, 9])

    Filter with a window length of 5 and a degree 2 polynomial.  Use
    the defaults for all other parameters.

    >>> savgol_filter(x, 5, 2)
    array([1.66, 3.17, 3.54, 2.86, 0.66, 0.17, 1.  , 4.  , 9.  ])

    Note that the last five values in x are samples of a parabola, so
    when mode='interp' (the default) is used with polyorder=2, the last
    three values are unchanged. Compare that to, for example,
    `mode='nearest'`:

    >>> savgol_filter(x, 5, 2, mode='nearest')
    array([1.74, 3.03, 3.54, 2.86, 0.66, 0.17, 1.  , 4.6 , 7.97])

    """
    if mode not in ["mirror", "constant", "nearest", "interp", "wrap"]:
        raise ValueError("mode must be 'mirror', 'constant', 'nearest' "
                         "'wrap' or 'interp'.")

    x = np.asarray(x)
    # Ensure that x is either single or double precision floating point.
    if x.dtype != np.float64 and x.dtype != np.float32:
        x = x.astype(np.float64)

    coeffs = savgol_coeffs(window_length, polyorder, deriv=deriv, delta=delta)

    if mode == "interp":
        if window_length > x.size:
            raise ValueError("If mode is 'interp', window_length must be less "
                             "than or equal to the size of x.")

        # Do not pad. Instead, for the elements within `window_length // 2`
        # of the ends of the sequence, use the polynomial that is fitted to
        # the last `window_length` elements.
        y = convolve1d(x, coeffs, axis=axis, mode="constant")
        _fit_edges_polyfit(x, window_length, polyorder, deriv, delta, axis, y)
    else:
        # Any mode other than 'interp' is passed on to ndimage.convolve1d.
        y = convolve1d(x, coeffs, axis=axis, mode=mode, cval=cval)

    return y        

def compute_errors (
    arr, /, 
    error ='std', 
    axis = 0, 
    return_confidence=False 
    ): 
    """ Compute Errors ( Standard Deviation ) and standard errors. 
    
    Standard error and standard deviation are both measures of variability:
    - The standard deviation describes variability within a single sample. Its
      formula is given as: 
          
      .. math:: 
          
          SD = \sqrt{ \sum |x -\mu|^2}{N}
          
      where :math:`\sum` means the "sum of", :math:`x` is the value in the data 
      set,:math:`\mu` is the mean of the data set and :math:`N` is the number 
      of the data points in the population. :math:`SD` is the quantity 
      expressing by how much the members of a group differ from the mean 
      value for the group.
      
    - The standard error estimates the variability across multiple 
      samples of a population. Different formulas are used depending on 
      whether the population standard deviation is known.
      
      - when the population standard deviation is known: 
      
        .. math:: 
          
            SE = \frac{SD}{\sqrt{N}} 
            
      - When the population parameter is unknwon 
      
        .. math:: 
            
            SE = \frac{s}{\sqrt{N}} 
            
       where :math:`SE` is the standard error, : math:`s` is the sample
       standard deviation. When the population standard is knwon the 
       :math:`SE` is more accurate. 
    
    Note that the :math:`SD` is  a descriptive statistic that can be 
    calculated from sample data. In contrast, the standard error is an 
    inferential statistic that can only be estimated 
    (unless the real population parameter is known). 
    
    Parameters
    ----------
    arr : array_like , 1D or 2D 
      Array for computing the standard deviation 
      
    error: str, default='std'
      Name of error to compute. By default compute the standard deviation. 
      Can also compute the the standard error estimation if the  argument 
      is passed to ``ste``. 
    return_confidence: bool, default=False, 
      If ``True``, returns the confidence interval with 95% of sample means 
      
    Returns 
    --------
    err: arraylike 1D or 2D 
       Error array. 
       
    Examples
    ---------
    >>> from gofast.tools.mathex  import compute_errors
    >>> from gofast.datasets import make_mining_ops 
    >>> mdata = make_mining_ops ( samples =20, as_frame=True, noises="20%")
    >>> compute_scores (mdata)
    Easting_m                    301.216454
    Northing_m                   301.284073
    Depth_m                      145.343063
    OreConcentration_Percent       5.908375
    DrillDiameter_mm              50.019249
    BlastHoleDepth_m               3.568771
    ExplosiveAmount_kg           142.908481
    EquipmentAge_years             4.537603
    DailyProduction_tonnes      2464.819019
    dtype: float64
    >>> compute_errors ( mdata, return_confidence= True)
    (Easting_m                   -100.015509
     Northing_m                  -181.088446
     Depth_m                      -67.948155
     OreConcentration_Percent      -3.316211
     DrillDiameter_mm              25.820805
     BlastHoleDepth_m               1.733541
     ExplosiveAmount_kg             3.505198
     EquipmentAge_years            -1.581202
     DailyProduction_tonnes      1058.839261
     dtype: float64,
     Easting_m                    1080.752992
     Northing_m                    999.945119
     Depth_m                       501.796651
     OreConcentration_Percent       19.844618
     DrillDiameter_mm              221.896260
     BlastHoleDepth_m               15.723123
     ExplosiveAmount_kg            563.706443
     EquipmentAge_years             16.206202
     DailyProduction_tonnes      10720.929814
     dtype: float64)
    """
    error = _validate_name_in(error , defaults =('error', 'se'),
                              deep =True, expect_name ='se')
    # keep only the numeric values.
    if hasattr (arr, '__array__') and hasattr(arr, 'columns'): 
        arr = to_numeric_dtypes ( arr, pop_cat_features =True )
        
    if not _is_numeric_dtype(arr): 
        raise TypeError("Numeric array is expected for operations.")
    err= np.std (arr) if arr.ndim ==1 else np.std (arr, axis= axis )
                  
    err_lower =  err_upper = None 
    if error =='se': 
        N = len(arr) if arr.ndim ==1 else arr.shape [axis ]
        err =  err / np.sqrt(N)
    if return_confidence: 
        err_lower = arr.mean() - ( 1.96 * err ) 
        err_upper = arr.mean() + ( 1.96 * err )
    return err if not return_confidence else ( err_lower, err_upper)  


def quality_control2(
    ar, 
     /, 
    tol: float= .5 , 
    return_data=False,
    to_log10: bool =False, 
    return_qco:bool=False 
    )->Tuple[float, ArrayLike]: 
    """
    Check the quality control in the collection of Z or EDI objects. 
    
    Analyse the data in the EDI collection and return the quality control value.
    It indicates how percentage are the data to be representative.
   
    Parameters 
    ----------
    
    ar: Arraylike of (m_samples, n_features)
       Arraylike  two dimensional data.
        
    tol: float, default=.5 
        the tolerance parameter. The value indicates the rate from which the 
        data can be consider as meaningful. Preferably it should be less than
        1 and greater than 0.  Default is ``.5`` means 50 %. Analysis becomes 
        soft with higher `tol` values and severe otherwise. 
        
    return_data: bool, default= False, 
        returns the valid data from up to ``1-tol%`` goodness. 
        
    return qco: bool, default=False, 
       retuns quality control object that wraps all usefull informations after 
       control. The following attributes can be fetched as: 
           
       - rate_: the rate of the quality of the data  
       - component_: The selected component where data is selected for analysis 
         By default used either ``xy`` or ``yx``. 
       - mode_: The :term:`EM` mode. Either the ['TE'|'TM'] modes 
       - freqs_: The valid frequency in the data selected according to the 
         `tol` parameters. Note that if ``interpolate_freq`` is ``True``, it 
         is used instead. 
       - invalid_freqs_: Useless frequency dropped in the data during control 
       - data_: Valid tensor data either in TE or TM mode. 
       
    Returns 
    -------
    Tuple (float  )  or (float, array-like, shape (N, )) or QCo
        - return the quality control value and interpolated frequency if  
         `return_freq`  is set to ``True`` otherwise return the
         only the quality control ratio.
        - return the the quality control object. 
        
    Examples 
    -----------
    >>> import gofast as gf 
    >>> data = gf.fetch_data ('huayuan', samples =20, return_data =True ,
                              key='raw')
    >>> r,= gf.qc (data)
    r
    Out[61]: 0.75
    >>> r, = gf.qc (data, tol=.2 )
    0.75
    >>> r, = gf.qc (data, tol=.1 )
    
    """
    tol = assert_ratio(tol , bounds =(0, 1), exclude_value ='use lower bound',
                         name ='tolerance', in_percent =True )
    # by default , we used the resistivity tensor and error at TE mode.
    # force using the error when resistivity or phase tensors are supplied 
    # compute the ratio of NaN in axis =0 
    nan_sum  =np.nansum(np.isnan(ar), axis =1) 

    rr= np.around ( nan_sum / ar.shape[1] , 2) 
    # print(rr); print(nan_sum) 
    # print(rr[0])
    # print(nan_sum[rr[0]].sum())
    # compute the ratio ck
    # ck = 1. -    rr[np.nonzero(rr)[0]].sum() / (
    #     1 if len(np.nonzero(rr)[0])== 0 else len(np.nonzero(rr)[0])) 
    # ck =  (1. * len(rr) - len(rr[np.nonzero(rr)[0]]) )  / len(rr)
    
    # using np.nonzero(rr) seems deprecated 
    ck = 1 - nan_sum[np.nonzero(rr)[0]].sum() / (
        ar.shape [0] * ar.shape [1]) 
    # ck = 1 - nan_sum[rr[0]].sum() / (
    #     ar.shape [0] * ar.shape [1]) 
    # now consider dirty data where the value is higher 
    # than the tol parameter and safe otherwise. 
    index = reshape (np.argwhere (rr > tol))
    # ar_new = np.delete (rr , index , axis = 0 ) 
    # if return QCobj then block all returns  to True 
    if return_qco: 
        return_data = True 
        
    data =[ np.around (ck, 2) ] 

    if return_data :
        data += [ np.delete ( ar, index , axis =0 )] 
        
    data = tuple (data )
    # make QCO object 
    if return_qco: 
        data = Boxspace( **dict (
            tol=tol, 
            rate_= float(np.around (ck, 2)), 
            data_=  np.delete ( ar, index , axis =0 )
            )
        )
    return data
 

def quality_control(
    data, 
    missing_threshold=0.05, 
    outlier_method='IQR', 
    value_ranges=None,
    unique_value_columns=None, 
    string_patterns=None, 
    verbose:bool= ..., 
    polish_and_return:bool=..., 
    columns=None, 
    **kwd 
    ):
    """
    Perform comprehensive data quality checks on a pandas DataFrame.

    In addition to checking, this function now cleans and sanitizes the 
    DataFrame based on identified issues if ``return_cleaned_data=True``. 
    
    This function now includes steps to:

    Drop columns with a high percentage of missing data.
    Remove outliers using either the IQR or Z-score methods.
    For other checks like value ranges, unique value columns, and 
    string patterns, appropriate cleaning steps would depend heavily on the 
    context and requirements of the specific dataset. 
    

    
    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame on which to perform data quality checks.
    missing_threshold : float, optional
        The threshold for missing data percentage to flag a column
        (default is 0.05, i.e., 5%).
    outlier_method : str, optional
        Method to detect outliers. Options are 'IQR' for Interquartile Range 
        (default) and 'Z-score'.
    value_ranges : dict, optional
        A dictionary where keys are column names and values are tuples 
        specifying the acceptable (min, max) range for values in that column.
    unique_value_columns : list, optional
        A list of column names that are expected to have unique values.
    string_patterns : dict, optional
        A dictionary where keys are column names and values are regex patterns 
        that values in the column should match.
        
    polish_and_return : bool, optional
        If True, the function will clean and sanitize the data based on the checks
        and return the polished DataFrame. Default is False.
        
    Returns
    -------
    dict
        A dictionary with keys 'missing_data', 'outliers', 'data_types',
        'value_range_violations', 'unique_value_violations', and 
        'string_pattern_violations', containing information about various data 
        quality issues.

    pandas.DataFrame or dict
        If polish_and_return is True, returns the cleaned and sanitized DataFrame.
        Otherwise, returns a dictionary with data quality check results.
        
    Note 
    ------
    Remember that data cleaning and sanitization can significantly alter 
    your dataset. It's essential to understand the implications of each 
    step and adjust the thresholds and methods according to your data 
    analysis goals.
        
    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.tools.mathex  import quality_control 
    >>> data = {'A': [1, 2, 3, None, 5], 'B': [1, 2, 100, 3, 4], 
                'C': ['abc', 'def', '123', 'xyz', 'ab']}
    >>> data = pd.DataFrame(data)
    >>> quality_control(data, value_ranges={'A': (0, 10)}, unique_value_columns=['B'], 
    ...                    string_patterns={'C': r'^[a-zA-Z]+$'})
    {
        'missing_data': {'A': 20.0},
        'outliers': {'B': [100]},
        'data_types': {'A': dtype('float64'), 'B': dtype('int64'), 'C': dtype('O')},
        'value_range_violations': {'A': [None]},
        'unique_value_violations': {'B': [1, 2, 3, 4]},
        'string_pattern_violations': {'C': ['123']}
    }

    """
    result = {
        'missing_data': {}, 
        'outliers': {}, 
        'data_types': {},
        'value_range_violations': {},
        'unique_value_violations': {},
        'string_pattern_violations': {}
    }
    polish_and_return, verbose = ellipsis2false(polish_and_return, verbose)
    data = build_data_if (data, columns = columns , to_frame=True,  **kwd )
    # for safety and not
    # make a copy for a new data 
    data_ = copy.deepcopy(data ) 
    
    # Handling missing data by dropping columns with 
    # too many missing values if polish_and_return is true. 
    
    for col in data.columns:
        missing_percentage = data[col].isna().mean()
        if missing_percentage > missing_threshold:
            if polish_and_return: 
                data_.drop(col, axis=1, inplace=True)  
            # data.drop(col, axis=1, inplace=True)  # Drop column
            result['missing_data'][col] = missing_percentage * 100
      
    # Detect  and Handling outliers
    for col in data.select_dtypes(include=[np.number]).columns:
        if outlier_method == 'IQR':
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[col][(
                data[col] < lower_bound) | (data[col] > upper_bound)]
            
            if polish_and_return: # Remove outliers
                data_ = data_[
                    (data_[col] >= lower_bound) & (data_[col] <= upper_bound)]  
            
        elif outlier_method == 'Z-score':
            z_score = np.abs((data[col] - data[col].mean()) / data[col].std())
            
            outliers = data[col][z_score > 3]
            
            if polish_and_return: 
                data_ = data_[z_score <= 3]  # Remove outliers

        if not outliers.empty:
            result['outliers'][col] = outliers.tolist()
            
    # Value range validation
    if value_ranges:
        for col, (min_val, max_val) in value_ranges.items():
            if col in data.columns:
                invalid_values = data[col][(data[col] < min_val) | (data[col] > max_val)]
                if not invalid_values.empty:
                    result['value_range_violations'][col] = invalid_values.tolist()

    # Unique value check
    if unique_value_columns:
        for col in unique_value_columns:
            if col in data.columns and data[col].duplicated().any():
                ( result['unique_value_violations'][col]
                 ) = data[col][data[col].duplicated()].tolist()

    # String pattern validation
    if string_patterns:
        for col, pattern in string_patterns.items():
            if col in data.columns and data[col].dtype == 'object':
                invalid_strings = data[col][~data[col].astype(
                    str).str.match(pattern)]
                if not invalid_strings.empty:
                    ( result['string_pattern_violations'][col]
                     )  = invalid_strings.tolist()
                    
    # Data type information
    for col in data.columns:
        result['data_types'][col] = data[col].dtype

    # [Fancy print function as before...]
    if verbose: 
        fancy_printer(result)
    
    return data_ if polish_and_return else result

def get_distance(
    x: ArrayLike, 
    y:ArrayLike , *, 
    return_mean_dist:bool =False, 
    is_latlon= False , 
    **kws
    ): 
    """
    Compute distance between points
    
    Parameters
    ------------
    x, y: ArrayLike 1d, 
       One dimensional arrays. `x` can be consider as the abscissa of the  
       landmark and `y` as ordinates array. 
       
    return_mean_dist: bool, default =False, 
       Returns the average value of the distance between different points. 
       
    is_latlon: bool, default=False, 
        Convert `x` and `y` latitude  and longitude coordinates values 
        into UTM before computing the distance. `x`, `y` should be considered 
        as ``easting`` and ``northing`` respectively. 
        
    kws: dict, 
       Keyword arguments passed to :meth:`gofast.site.Location.to_utm_in`
       
    Returns 
    ---------
    d: Arraylike of shape (N-1) 
      Is the distance between points. 
      
    Examples 
    --------- 
    >>> import numpy as np 
    >>> from gofast.tools.mathex  import get_distance 
    >>> x = np.random.rand (7) *10 
    >>> y = np.abs ( np.random.randn (7) * 12 ) 
    >>> get_distance (x, y) 
    array([ 8.7665511 , 12.47545656,  8.53730212, 13.54998351, 14.0419387 ,
           20.12086781])
    >>> get_distance (x, y, return_mean_dist= True) 
    12.91534996818084
    """
    x, y = _assert_x_y_positions (x, y, is_latlon , **kws  )
    d = np.sqrt( np.diff (x) **2 + np.diff (y)**2 ) 
    
    return d.mean()  if return_mean_dist else d 

def scale_positions (
    x: ArrayLike, 
    y:ArrayLike, 
    *, 
    is_latlon:bool=False, 
    step:float= None, 
    use_average_dist:bool=False, 
    utm_zone:str= None, 
    shift: bool=True, 
    view:bool = False, 
    **kws
    ): 
    """
    Correct the position coordinates. 
     
    By default, it consider `x` and `y` as easting/latitude and 
    northing/longitude coordinates respectively. It latitude and longitude 
    are given, specify the parameter `is_latlon` to ``True``. 
    
    Parameters
    ----------
    x, y: ArrayLike 1d, 
       One dimensional arrays. `x` can be consider as the abscissa of the  
       landmark and `y` as ordinates array. 
       
    is_latlon: bool, default=False, 
       Convert `x` and `y` latitude  and longitude coordinates values 
       into UTM before computing the distance. `x`, `y` should be considered 
       as ``easting`` and ``northing`` respectively. 
           
    step: float, Optional 
       The positions separation. If not given, the average distance between 
       all positions should be used instead. 
    use_average_dist: bool, default=False, 
       Use the distance computed between positions for the correction. 
    utm_zone: str,  Optional (##N or ##S)
       UTM zone in the form of number and North or South hemisphere. For
       instance '10S' or '03N'. Note that if `x` and `y` are UTM coordinates,
       the `utm_zone` must be provide to accurately correct the positions, 
       otherwise the default value ``49R`` should be used which may lead to 
       less accuracy. 
       
    shift: bool, default=True,
       Shift the coordinates from the units of `step`. This is the default 
       behavor. If ``False``, the positions are just scaled. 
    
    view: bool, default=True 
       Visualize the scaled positions 
       
    kws: dict, 
       Keyword arguments passed to :func:`~.get_distance` 
    Returns 
    --------
    xx, yy: Arraylike 1d, 
       The arrays of position correction from `x` and `y` using the 
       bearing. 
       
    See Also 
    ---------
    gofast.tools.mathex .get_bearing: 
        Compute the  direction of one point relative to another point. 
      
    Examples
    ---------
    >>> from gofast.tools.mathex  import scale_positions 
    >>> east = [336698.731, 336714.574, 336730.305] 
    >>> north = [3143970.128, 3143957.934, 3143945.76]
    >>> east_c , north_c= scale_positions (east, north, step =20, view =True  ) 
    >>> east_c , north_c
    (array([336686.69198337, 336702.53498337, 336718.26598337]),
     array([3143986.09866306, 3143973.90466306, 3143961.73066306]))
    """
    from ..site import Location
    
    msg =("x, y are not in longitude/latitude format  while 'utm_zone' is not"
          " supplied. Correction should be less accurate. Provide the UTM"
          " zone to improve the accuracy.")
    
    if is_latlon: 
        xs , ys = np.array(copy.deepcopy(x)) , np.array(copy.deepcopy(y))

    x, y = _assert_x_y_positions( x, y, islatlon = is_latlon , **kws ) 
    
    if step is None: 
        warnings.warn("Step is not given. Average distance between points"
                      " should be used instead.")
        use_average_dist =True 
    else:  
        d = float (_assert_all_types(step, float, int , objname ='Step (m)'))
    if use_average_dist: 
        d = get_distance(x, y, return_mean_dist=use_average_dist,  **kws) 
        
    # compute bearing. 
    utm_zone = utm_zone or '49R'
    if not is_latlon and utm_zone is None: 
        warnings.warn(msg ) 
    if not is_latlon: 
        xs , ys = Location.to_latlon_in(x, y, utm_zone= utm_zone) 
  
    b = get_bearing((xs[0] , ys[0]) , (xs[-1], ys[-1]),
                    to_deg =False ) # return bearing in rad.
 
    xx = x + ( d * np.cos (b))
    yy = y +  (d * np.sin(b))
    if not shift: 
        xx, *_ = scalePosition(x )
        yy, *_ = scalePosition(y)
        
    if view: 
        state = f"{'scaled' if not shift else 'shifted'}"
        plt.plot (x, y , 'ok-', label =f"Un{state} positions") 
        plt.plot (xx , yy , 'or:', label =f"{state.title()} positions")
        plt.xlabel ('x') ; plt.ylabel ('y')
        plt.legend()
        plt.show () 
        
    return xx, yy 

def _assert_x_y_positions (x, y , islatlon = False, is_utm=True,  **kws): 
    """ Assert the position x and y and return array of x and y  """
    from ..site import Location 
    x = np.array(x, dtype = np.float64) 
    y = np.array(y, np.float64)
    for ii, ar in enumerate ([x, y]):
        if not _is_arraylike_1d(ar):
            raise TypeError (
                f"Expect one-dimensional array for {'x' if ii==0 else 'y'!r}."
                " Got {x.ndim}d.")
        if len(ar) <= 1:
            raise ValueError (f"A singleton array {'x' if ii==0 else 'y'!r} is"
                              " not admitted. Expect at least two points"
                              " A(x1, y1) and B(x2, y2)")
    if islatlon: 
        x , y = Location.to_utm_in(x, y, **kws)
    return x, y 

def get_bearing (latlon1, latlon2,  to_deg = True ): 
    """
    Calculate the bearing between two points. 
     
    A bearing can be defined as  a direction of one point relative 
    to another point, usually given as an angle measured clockwise 
    from north.
    The formula of the bearing :math:`\beta` between two points 1(lat1 , lon1)
    and 2(lat2, lon2) is expressed as below: 
        
    .. math:: 
        \beta = atan2(sin(y_2-y_1)*cos(x_2), cos(x_1)*sin(x_2) – \
                      sin(x_1)*cos(x_2)*cos(y_2-y_1))
     
    where: 
       
       - :math:`x_1`(lat1): the latitude of the first coordinate
       - :math:`y_1`(lon1): the longitude of the first coordinate
       - :math:`x_2`(lat2) : the latitude of the second coordinate
       - :math:`y_2`(lon2): the longitude of the second coordinate
    
    Parameters 
    ----------- 
    latlon: Tuple ( latitude, longitude) 
       A latitude and longitude coordinates of the first point in degree. 
    latlon2: Tuple ( latitude, longitude) 
       A latitude and longitude of coordinates of the second point in degree.  
       
    to_deg: bool, default=True 
       Convert the bearing from radians to degree. 
      
    Returns 
    ---------
    b: Value of bearing in degree ( default). 
    
    See More 
    ----------
    See more details by clicking in the link below: 
        https://mapscaping.com/how-to-calculate-bearing-between-two-coordinates/
        
    Examples 
    ---------
    >>> from gofast.tools import get_bearing 
    >>> latlon1 = (28.41196763902007, 109.3328724432221) # (lat, lon) point 1
    >>> latlon2= (28.38756530909265, 109.36931920880758) # (lat, lon) point 2
    >>> get_bearing (latlon1, latlon2 )
    127.26739270447973 # in degree 
    """
    latlon1 = reshape ( np.array ( latlon1, dtype = np.float64)) 
    latlon2 = reshape ( np.array ( latlon2, dtype = np.float64)) 
    
    if len(latlon1) <2 or len(latlon2) <2 : 
        raise ValueError("Wrong coordinates values. Need two coordinates"
                         " (latitude and longitude) of points 1 and 2.")
    lat1 = np.deg2rad (latlon1[0]) ; lon1 = np.deg2rad(latlon1[1])
    lat2 = np.deg2rad (latlon2[0]) ; lon2 = np.deg2rad(latlon2[1])
    
    b = np.arctan2 (
        np.sin(lon2 - lon1 )* np.cos (lat2), 
        np.cos (lat1) * np.sin(lat2) - np.sin (lat1) * np.cos (lat2) * np.cos (lon2 - lon1)
                    )
    if to_deg: 
        # convert bearing to degree and make sure it 
        # is positive between 360 degree 
        b = ( np.rad2deg ( b) + 360 )% 360 
        
    return b 

def find_closest( arr, /, values ): 
    """Get the closest value in array  from given values.
    
    Parameters 
    -----------
    arr : Arraylike  
       Array to find the values 
       
    values: float, arraylike 
    
    Returns
    --------
    closest values in float or array containing in the given array.
    
    Examples
    -----------
    >>> import numpy as np 
    >>> from gofast.tools.mathex  import find_closest
    >>> find_closest (  [ 2 , 3, 4, 5] , ( 2.6 , 5.6 )  )
    array([3., 5.])
    >>> find_closest (  np.array ([[2 , 3], [ 4, 5]]), ( 2.6 , 5.6 ) )
    array([3., 5.])
    array([3., 5.])
    """

    arr = is_iterable(arr, exclude_string=True , transform =True  )
    values = is_iterable(values , exclude_string=True  , transform =True ) 
    
    for ar, v in zip ( [ arr, values ], ['array', 'values']): 
        if not _is_numeric_dtype(arr, to_array= True ) :
            raise TypeError(f"Non-numerical {v} are not allowed.")
        
    arr = np.array (arr, dtype = np.float64 )
    values = np.array (values, dtype = np.float64 ) 
    
    # ravel arr if ndim is not one-dimensional 
    arr  = arr.ravel() if arr.ndim !=1 else arr 
    # Could Find the absolute difference with each value   
    # Get the index of the smallest absolute difference. 
    
    # --> Using map is less faster than list comprehension 
    # close = np.array ( list(
    #     map (lambda v: np.abs ( arr - v).argmin(), values )
    #                   ), dtype = np.float64
    #     )
    return np.array ( [
        arr [ np.abs ( arr - v).argmin()] for v in values ]
        )
  
def gradient_descent(
    z: ArrayLike, 
    s:ArrayLike, 
    alpha:float=.01, 
    n_epochs:int= 100,
    kind:str="linear", 
    degree:int=1, 
    raise_warn:bool=False, 
    ): 
    """ Gradient descent algorithm to  fit the best model parameter.
    
    Model can be changed to polynomial if degree is greater than 1. 
    
    Parameters 
    -----------
    z: arraylike, 
       vertical nodes containing the values of depth V
    s: Arraylike, 
       vertical vector containin the resistivity values 
    alpha: float,
       step descent parameter or learning rate. *Default* is ``0.01`
    n_epochs: int, 
       number of iterations. *Default* is ``100``. Can be changed to other values
    kind: str, {"linear", "poly"}, default= 'linear'
      Type of model to fit. Linear model is selected as the default. 
    degree: int, default=1 
       As the linear model is selected as the default since the degree is set 
       to ``1``
    Returns 
    ---------
    - `_F`: New model values with the best `W` parameters found.
    - `W`: vector containing the parameters fits 
    - `cost_history`: Containing the error at each Itiretaions. 
        
    Examples 
    -----------
    >>> import numpy as np 
    >>> from gofast.tools.mathex  import gradient_descent
    >>> z= np.array([0, 6, 13, 20, 29 ,39, 49, 59, 69, 89, 109, 129, 
                     149, 179])
    >>> res= np.array( [1.59268,1.59268,2.64917,3.30592,3.76168,
                        4.09031,4.33606, 4.53951,4.71819,4.90838,
          5.01096,5.0536,5.0655,5.06767])
    >>> fz, weights, cost_history = gradient_descent(
        z=z, s=res,n_epochs=10,alpha=1e-8,degree=2)
    >>> import matplotlib.pyplot as plt 
    >>> plt.scatter (z, res)
    >>> plt.plot(z, fz)
    """
    
    #Assert degree
    try :degree= abs(int(degree)) 
    except:raise TypeError(f"Degree is integer. Got {type(degree).__name__!r}")
    
    if degree >1 :
        kind='poly'
        
    kind = str(kind).lower()    
    if kind.lower() =='linear': 
        # block degree to one.
        degree = 1 
    elif kind.find('poly')>=0 : 
        if degree <=1 :
            warnings.warn(
                "Polynomial function expects degree greater than 1."
                f" Got {degree!r}. Value is resetting to minimum equal 2."
                      ) if raise_warn else None 
            degree = 2
    # generate function with degree 
    Z, W = _kind_of_model(degree=degree,  x=z, y=s)
    
    # Compute the gradient descent 
    cost_history = np.zeros(n_epochs)
    s=s.reshape((s.shape[0], 1))
    
    for ii in range(n_epochs): 
        with np.errstate(all='ignore'): # rather than divide='warn'
            #https://numpy.org/devdocs/reference/generated/numpy.errstate.html
            W= W - (Z._T.dot(Z.dot(W)-s)/ Z.shape[0]) * alpha 
            cost_history[ii]= (1/ 2* Z.shape[0]) * np.sum((Z.dot(W) -s)**2)
       
    # Model function _F= Z.W where `Z` id composed of vertical nodes 
    # values and `bias` columns and `W` is weights numbers.
    _F= Z.dot(W) # model(Z=Z, W=W)     # generate the new model with the best weights 
             
    return _F,W, cost_history

def _kind_of_model(degree, x, y) :
    """ 
    An isolated part of gradient descent computing. 
    Generate kind of model. If degree is``1`` The linear subset 
    function will use. If `degree` is greater than 2,  Matrix will 
    generate using the polynomail function.
     
    :param x: X values must be the vertical nodes values 
    :param y: S values must be the resistivity of subblocks at node x 
    
    """
    c= []
    deg = degree 
    w = np.zeros((degree+1, 1)) # initialize weights 
    
    def init_weights (x, y): 
        """ Init weights by calculating the scope of the function along 
         the vertical nodes axis for each columns. """
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', 
                                    category=RuntimeWarning)
            for j in range(x.shape[1]-1): 
                a= (y.max()-y.min())/(x[:, j].max()-x[:, j].min())
                w[j]=a
            w[-1] = y.mean()
        return w   # return weights 

    for i in range(degree):
        c.append(x ** deg)
        deg= deg -1 

    if len(c)> 1: 
        x= concat_array_from_list(c, concat_axis=1)
        x= np.concatenate((x, np.ones((x.shape[0], 1))), axis =1)

    else: x= np.vstack((x, np.ones(x.shape)))._T # initialize z to V*2

    w= init_weights(x=x, y=y)
    return x, w  # Return the matrix x and the weights vector w 
    
def adaptive_moving_average(data, /, window_size_factor=0.1):
    """ Adaptative moving average as  smoothing technique. 
 
    Parameters 
    -----------
    data: Arraylike 
       Noise data for smoothing 
       
    window_size_factor: float, default=0.1 
      Parameter to control the adaptiveness of the moving average.
       
    Return 
    --------
    result: Arraylike 
       Smoothed data 
    
    Example 
    ---------
    >>> import matplotlib.pyplot as plt
    >>> from gofast.tools.mathex  import adaptive_moving_average 
    >>> # Sample magnetotelluric data (replace this with your own data)
    >>> # Example data: a sine wave with noise
    >>> time = np.linspace(0, 10, 1000)  # Replace with your actual time values
    >>> mt_data = np.sin(2 * np.pi * 1 * time) + 0.2 * np.random.randn(1000)  # Example data
    >>> # Function to calculate the adaptive moving average
    >>> # Define the window size factor (adjust as needed)
    >>> window_size_factor = 0.1  # Adjust this value based on your data characteristics
    >>> # Apply adaptive moving average to the magnetotelluric data
    >>> smoothed_data = adaptive_moving_average(mt_data, window_size_factor)
    >>> # Plot the original and smoothed data
    >>> plt.figure(figsize=(10, 6))
    >>> plt.plot(time, mt_data, 'b-', label='Original Data')
    >>> plt.plot(time, smoothed_data, 'r-', label='Smoothed Data (AMA)')
    >>> plt.xlabel('Time')
    >>> plt.ylabel('Amplitude')
    >>> plt.title('Adaptive Moving Average (AMA) Smoothing')
    >>> plt.legend()
    >>> plt.grid(True)
    >>> plt.show()
    """
    result = np.zeros_like(data)
    window_size = int(window_size_factor * len(data))
    
    for i in range(len(data)):
        start = max(0, i - window_size)
        end = min(len(data), i + window_size + 1)
        result[i] = np.mean(data[start:end])
    
    return result

def torres_verdin_filter(
    arr, /,  
    weight_factor: float=.1, 
    beta:bool=1., 
    logify:bool=False, 
    axis:int = ..., 
    ):
    """
    Calculates the adaptive moving average of a given data array from 
    Torres and Verdin algorithm [1]_. 
    
    Parameters 
    -----------
    arr: Arraylike 1d 
      List or array-like of data points.  If two-dimensional array 
      is passed, `axis` must be specified to apply the filter onto. 
       
    weight_factor: float, default=.1
      Base smoothing factor for window size which gets adjusted by a factor 
      dependent on the rate of change in the data. 
        
    beta: float, default =1. 
       Scaling factor to adjust `weight_factor` during high volatility. 
       It controls how much the `weight_factor` is adjusted during 
       periods of high volatility.
       
    logify: bool, default=False, 
      By default , Torres uses exponential moving average. So if the 
      values can be logarithmized to ensure the weight be ranged between 
      0 and 1. This is important when data are resistivity or phase. 
      
    axis: int, default=0 
      Axis along which to apply the AMA filter.
    Return 
    -------
    ama: Adaptive moving average
    
    References 
    ------------
    .. [1] Torres-Verdin and Bostick, 1992,  Principles of spatial surface 
        electric field filtering in magnetotellurics: electromagnetic array profiling
        (EMAP), Geophysics, v57, p603-622.https://doi.org/10.1190/1.2400625

    Example
    --------
    >>> import matplotlib.pyplot as plt 
    >>> from gofast.tools.mathex  import torres_verdin_filter 
    >>> data = np.random.randn(100)  
    >>> ama = torres_verdin_filter(data)
    >>> plt.plot (range (len(data)), data, 'k', range(len(data)), ama, '-or')
    >>> # apply on two dimensional array 
    >>> data2d = np.random.randn(7, 10) 
    >>> ama2d = torres_verdin_filter ( data2d, axis =0)
    >>> fig, ax  = plt.subplots (nrows = 1, ncols = 2 , sharey= True,
                             figsize = (7,7) )
    >>> ax[0].imshow(data2d , label ='Raw data', cmap = 'binary' )
    >>> ax[1].imshow (ama2d,  label = 'AMA data', cmap ='binary' )
    >>> ax[0].set_title ('Raw data') 
    >>> ax[1].set_title ('AMA data') 
    >>> plt.legend
    >>> plt.show () 
    
    """
    arr = is_iterable( arr, exclude_string =True, transform =True ) 
    axis, logify= ellipsis2false(axis, logify, default_value =( None , False))
    
    def _filtering_1d_array( ar, wf, b ): 
        if len(ar) < 2:
            return ar
        ama = [ar[0]]  # Initialize the adaptive moving average array
        for i in range(1, len(ar)):
            change = abs(ar[i] - ar[i-1])
            w = wf * (1 + beta * change)
            w = min(w, 1)  # Ensure weight stays between 0 and 1
            ama_value = w * ar[i] + (1 - w) * ama[-1]
            ama.append(ama_value)
            
        return np.array(ama)
    
    arr =np.array (arr )
    #+++++++++++++++++++
    if logify:
        arr = np.log10 ( arr )
    if arr.ndim >=2: 
        if axis is None:
            warnings.warn (f"Array dimension is {arr.ndim}. Axis must be"
                           " specified. Otherwise axis=0 is used .")
            axis =0
        if axis ==0: 
            arr = arr._T 
        for ii in range( len(arr )) : 
            arr [ii] = _filtering_1d_array (
                arr [ii ], wf = weight_factor, b = beta ) 
        # then transpose again 
        if axis ==0: 
            arr = arr._T 
    else: 
        arr = _filtering_1d_array ( arr, wf = weight_factor, b=beta  )
        
    if logify: arr = np.power (10, arr )
    
    return arr 

def binning_statistic(
    data, categorical_column, 
    value_column, 
    statistic='mean'
    ):
    """
    Compute a statistic for each category in a categorical column of a dataset.

    This function categorizes the data into bins based on a categorical variable and then
    applies a statistical function to the values of another column for each category.

    Parameters
    ----------
    data : DataFrame
        Pandas DataFrame containing the dataset.
    categorical_column : str
        Name of the column in `data` which contains the categorical variable.
    value_column : str
        Name of the column in `data` from which the statistic will be calculated.
    statistic : str, optional
        The statistic to compute (default is 'mean'). Other options include 
        'sum', 'count','median', 'min', 'max', etc.

    Returns
    -------
    result : DataFrame
        A DataFrame with each category and the corresponding computed statistic.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'Category': ['A', 'B', 'A', 'C', 'B', 'A', 'C'],
    ...     'Value': [1, 2, 3, 4, 5, 6, 7]
    ... })
    >>> binning_statistic(df, 'Category', 'Value', statistic='mean')
       Category  Mean_Value
    0        A         3.33
    1        B         3.50
    2        C         5.50
    """
    if statistic not in ('mean', 'sum', 'count', 'median', 'min',
                         'max', 'proportion'):
        raise ValueError(
            "Unsupported statistic. Please choose from 'mean',"
            " 'sum', 'count', 'median', 'min', 'max', 'proportion'.")

    grouped_data = data.groupby(categorical_column)[value_column]

    if statistic == 'mean':
        result = grouped_data.mean().reset_index(name=f'Mean_{value_column}')
    elif statistic == 'sum':
        result = grouped_data.sum().reset_index(name=f'Sum_{value_column}')
    elif statistic == 'count':
        result = grouped_data.count().reset_index(name=f'Count_{value_column}')
    elif statistic == 'median':
        result = grouped_data.median().reset_index(name=f'Median_{value_column}')
    elif statistic == 'min':
        result = grouped_data.min().reset_index(name=f'Min_{value_column}')
    elif statistic == 'max':
        result = grouped_data.max().reset_index(name=f'Max_{value_column}')
    elif statistic == 'proportion':
        total_count = data[value_column].count()
        proportion = grouped_data.sum() / total_count
        result = proportion.reset_index(name=f'Proportion_{value_column}')
        
    return result

def category_count(data, /, categorical_column= None):
    """
    Count occurrences of each category in a given categorical 
    column of a dataset.

    This function computes the frequency of each unique category 
    in the specified
    categorical column of a pandas DataFrame.

    Parameters
    ----------
    data : DataFrame
        Pandas DataFrame containing the dataset.
    categorical_column : str
        Name of the column in `data` which contains the categorical variable.

    Returns
    -------
    counts : DataFrame
        A DataFrame with each category and the corresponding count.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'Category': ['A', 'B', 'A', 'C', 'B', 'A', 'C']
    ... })
    >>> category_count(df, 'Category')
       Category  Count
    0        A      3
    1        B      2
    2        C      2
    """
    if categorical_column not in data.columns:
        raise ValueError(f"Column '{categorical_column}' not found in the dataframe.")

    counts = data[categorical_column].value_counts().reset_index()
    counts.columns = [categorical_column, 'Count']
    return counts

def soft_bin_stat(
    data, /, categorical_column, 
    target_column, 
    statistic='mean', 
    update=False, 
    ):
    """
    Compute a statistic for each category in a categorical 
    column based on a binary target.

    This function calculates statistics like mean, sum, or proportion 
    for a binary target variable, grouped by categories in a 
    specified column.

    Parameters
    ----------
    data : DataFrame
        Pandas DataFrame containing the dataset.
    categorical_column : str
        Name of the column in `data` which contains the categorical variable.
    target_column : str
        Name of the column in `data` which contains the binary target variable.
    statistic : str, optional
        The statistic to compute for the binary target (default is 'mean').
        Other options include 'sum' and 'proportion'.

    Returns
    -------
    result : DataFrame
        A DataFrame with each category and the corresponding 
        computed statistic.

    Examples
    --------
    >>> from gofast.tools.mathex import soft_bin_stat
    >>> df = pd.DataFrame({
    ...     'Category': ['A', 'B', 'A', 'C', 'B', 'A', 'C'],
    ...     'Target': [1, 0, 1, 0, 1, 0, 1]
    ... })
    >>> soft_bin_stat(df, 'Category', 'Target', statistic='mean')
       Category  Mean_Target
    0        A     0.666667
    1        B     0.500000
    2        C     0.500000
    """
    if statistic not in ['mean', 'sum', 'proportion']:
        raise ValueError("Unsupported statistic. Please choose from "
                         "'mean', 'sum', 'proportion'.")

    grouped_data = data.groupby(categorical_column)[target_column]

    if statistic == 'mean':
        result = grouped_data.mean().reset_index(name=f'Mean_{target_column}')
    elif statistic == 'sum':
        result = grouped_data.sum().reset_index(name=f'Sum_{target_column}')
    elif statistic == 'proportion':
        total_count = data[target_column].count()
        proportion = grouped_data.sum() / total_count
        result = proportion.reset_index(name=f'Proportion_{target_column}')


    return result

def gradient_boosting_regressor(
        X, y, n_estimators=100, learning_rate=0.1, max_depth=1):
    """
    Implement a simple version of Gradient Boosting Regressor.

    Gradient Boosting builds an additive model in a forward stage-wise fashion. 
    At each stage, regression trees are fit on the negative gradient of the loss function.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The input samples.
    y : ndarray of shape (n_samples,)
        The target values (real numbers).
    n_estimators : int, default=100
        The number of boosting stages to be run.
    learning_rate : float, default=0.1
        Learning rate shrinks the contribution of each tree by `learning_rate`.
    max_depth : int, default=1
        The maximum depth of the individual regression estimators.

    Returns
    -------
    y_pred : ndarray of shape (n_samples,)
        The predicted values.

    Mathematical Formula
    --------------------
    Given a differentiable loss function L(y, F(x)), the general idea is 
    to iteratively construct additive models as follows:
    
    .. math:: 
        F_{m}(x) = F_{m-1}(x) + \\gamma_{m} h_{m}(x)

    where F_{m} is the model at iteration m, \\gamma_{m} is the step size,
    and h_{m} is the weak learner.

    Notes
    -----
    Gradient Boosting is widely used in machine learning for regression and 
    classification problems. It's effective in scenarios where data is not 
    linearly separable.

    References
    ----------
    - J. H. Friedman, "Greedy Function Approximation: A Gradient Boosting Machine," 1999.
    - T. Hastie, R. Tibshirani, and J. Friedman, "The Elements of Statistical Learning," Springer, 2009.

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=100, n_features=1, noise=10)
    >>> y_pred = gradient_boosting_regressor(X, y, n_estimators=100,
                                             learning_rate=0.1)
    >>> print(y_pred[:5])
    """
    from ..estimators import DecisionStumpRegressor
    # Initialize model
    F_m = np.zeros(len(y))
    # for m in range(n_estimators):
        # Compute negative gradient
        # residual = -(y - F_m)

        # # Fit a regression tree to the negative gradient
        # tree = DecisionTreeRegressor(max_depth=max_depth)
        # tree.fit(X, residual)

        # # Update the model
        # F_m += learning_rate * tree.predict(X)

    for m in range(n_estimators):
        # Compute negative gradient
        residual = -(y - F_m)
    
        # Fit a decision stump to the negative gradient
        stump = DecisionStumpRegressor()
        stump.fit(X, residual)
    
        # Update the model
        F_m += learning_rate * stump.predict(X)
    

    return F_m




    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    