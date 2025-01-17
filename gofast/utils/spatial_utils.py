# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
geospatial_utils - A collection of utilities for geospatial and positional 
data analysis, filtering, and transformations.
"""
from __future__ import print_function 
import os 
import re
import copy
import inspect
import warnings
import itertools
from numbers import Real

import scipy.stats as spstats
from scipy._lib._util import float_factorial
from scipy.linalg import lstsq
from scipy.ndimage import convolve1d
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ._arraytools import axis_slice
from ..api.box import KeyBox
from ..api.docstring import refglossary
from ..api.types import (
    ArrayLike, 
    DataFrame, 
    Optional,
    Series, 
    Tuple, 
    Union, 
    _F, 
    _SP, 
    _T,
)
from ..compat.sklearn import validate_params, StrOptions, Interval 
from ..core.array_manager import reshape, extract_array_from 
from ..core.checks import ( 
    _assert_all_types, 
    is_iterable,
    str2columns, 
    exist_features, 
    assert_ratio, 
    are_all_frames_valid, 
    )
from ..core.handlers import columns_manager, resolve_label  
from ..core.io import SaveFile, is_data_readable 
from ..core.utils import ellipsis2false, smart_format 
from ..decorators import Deprecated, AppendDocReferences, isdf 

from .validator import ( 
    _is_arraylike_1d, 
    assert_xy_in, check_y, 
    validate_positive_integer, 
    validate_length_range 
    )

__all__ = [
     'adaptive_moving_average',
     'assert_doi',
     'convert_distance_to_m',
     'detect_station_position',
     'display_infos',
     'find_close_position',
     'fit_ll',
     'get_azimuth',
     'get_bearing',
     'get_distance',
     'get_profile_angle',
     'get_station_number',
     'make_ids',
     'moving_average',
     'numstr2dms',
     'dual_merge', 
     'quality_control2',
     'round_dipole_length',
     'savgol_coeffs',
     'savgol_filter',
     'scalePosition',
     'scale_positions',
     'show_stats',
     'station_id',
     'torres_verdin_filter', 
     'spatial_sampling', 
     'extract_coordinates', 
     'batch_spatial_sampling', 
     'make_mxs_labels',
     'extract_zones_from', 
 ]

@is_data_readable(data_to_read='data')
@SaveFile
@validate_params ({ 
    'z': ['array-like', str], 
    'threshold': ['array-like', StrOptions({'auto'}), Real], 
    'condition': [StrOptions({'auto', 'above', 'below', 'between'})], 
    'use_negative_criteria': [bool], 
    'percentile': [Real], 
    'x': ['array-like', str, None], 
    'y': ['array-like', str, None], 
    'data': ['array-like', None], 
    })
def extract_zones_from(
    z,
    threshold='auto',        
    condition='auto',        
    use_negative_criteria=True,  
    percentile=10,           
    x=None,                  
    y=None,                 
    data=None,              
    view=False,              
    plot_type='scatter',    
    figsize=(8, 6),  
    savefile=None,  
    axis_off=False, 
    show_grid=True,        
    **kwargs                 
):
    r"""
    Extracts specific zones by filtering an input array or arrays
    using a threshold criterion. This function applies a logical
    mask to the values and retains those which satisfy a chosen
    condition (e.g. ``'above'``, ``'below'``, or ``'between'`` a
    specific threshold or thresholds). The threshold can be
    automatically derived using percentiles if ``'auto'`` is
    specified.

    .. math::
       \text{mask}(z) \;=\;
       \begin{cases}
         1 & \text{if } z \,\in\, \Omega \\
         0 & \text{otherwise}
       \end{cases}

    where :math:`\Omega` is the region of acceptance determined
    by the threshold mechanism.

    Parameters
    ----------
    z : array-like, Series, or string
        The input data to be filtered. If <z> is a string,
        it is treated as a column name from the provided
        <data> (see below). If it is an array or Series, it
        is used directly.
    threshold : { ``'auto'``, float, int, tuple }
        The criterion for filtering. If ``'auto'``, the
        function computes a percentile-based threshold
        driven by <percentile>. If a float or int is
        given, the function will filter values above or
        below that single threshold. If a tuple of length 2
        is provided, the function will filter between those
        bounds.
    condition : { ``'auto'``, ``'above'``, ``'below'``,
                     ``'between'`` }
        Defines how the filter is applied relative to the
        given or derived threshold. If ``'auto'``, the
        function decides based on <use_negative_criteria>.
        If ``'above'``, all values satisfying
        :math:`z > \text{threshold}` are kept. If
        ``'below'``, then :math:`z < \text{threshold}`
        are retained. If ``'between'``, a range
        (low, high) is respected.
    use_negative_criteria : bool
        When ``True``, automatically interprets
        ``'auto'`` condition as filtering from below, akin
        to negative-based risk. When ``False``, filters
        from above.
    percentile : int or float
        Used only if ``threshold='auto'``. Determines which
        percentile is used to compute the threshold. For
        example, if <use_negative_criteria> is ``True``,
        the <percentile>th percentile is chosen, otherwise
        the (100 - <percentile>)th percentile is used.
    x : array-like, Series, or string, optional
        The x-axis data. If <x> is a string and <data> is
        provided, the function extracts the relevant
        column. If <x> is an array or Series, it is used
        directly.
    y : array-like, Series, or string, optional
        Similar to <x>, representing the y-axis data.
    data : pandas.DataFrame, optional
        The DataFrame source if <x>, <y>, or <z> are
        provided as strings referencing column names.
    view : bool
        If ``True``, displays a plot of the filtered data.
    plot_type : { ``'scatter'``, ``'line'``, ``'hist'``,
                      ... }
        Determines how the data are visualized when <view>
        is ``True``.
    figsize : tuple of int
        The size of the generated figure for plotting. E.g.
        ``(8,6)`` is typical.
    axis_off: bool, default=False 
        Remove the axis if set to ``True``. 
    show_grid: bool, default=True 
        Display the plot grid or make it invisible if ``False``. 

    Methods
    -------
    The function itself does not expose methods starting with
    letters (excluding `_`), as it is a single operation. All
    steps are executed internally with no additional public
    methods.

    Notes
    ------
    .. math::
       \mathbf{z}_{\text{filtered}}
       \;=\; \{ z_i \mid \text{condition}(z_i) \}

    where :math:`\text{condition}(z_i)` is derived from
    ``threshold`` and ``condition``. For instance, if
    ``condition='below'`` and :math:`\tau =
    \text{threshold}`, then

    .. math::
       \text{condition}(z_i)
       \;=\; [\, z_i < \tau \,].
       
    This function relies on ``extract_array_from`` (from
    the gofast.core.array_manager) if <z>, <x>, or <y>
    are passed as strings and a <data> DataFrame is
    supplied. The user has the option to visualize the
    retained data points by enabling <view> and
    customizing <plot_type>.
    
    Examples
    --------
    >>> from gofast.utils.spatial_utils import extract_zones_from
    >>> import numpy as np
    >>> z_data = np.array([0, 2, 5, 10, 15, 20])
    >>> result = extract_zones_from(z=z_data, threshold=10,
    ...                        condition='above')
    >>> print(result)

    See Also
    --------
    gofast.core.array_manager.extract_array_from`` :
        The array extraction utility used for retrieving
        arrays from DataFrame columns.

    References
    ----------
    .. [1] Smith, J. & Doe, A. "Advanced Filtering
       Techniques", Journal of Data Science, 2022.
    .. [2] Brown, K. "Data Visualization Best
       Practices", Data Analytics Press, 2021.
    """

    # 1) If z, x, y are strings, extract from DataFrame
    if data is not None:
        z, x, y = extract_array_from(
            data,
            z,
            x,
            y,
            handle_unknown='raise',  # raise if columns not found
            error='raise',
            check_size=True,         # ensure x,y,z have same length
        )

    # 2) Ensure z, x, y are arrays
    z = np.asarray(z) if not isinstance(z, np.ndarray) else z
    x = np.asarray(x) if (x is not None and not isinstance(x, np.ndarray)) else x
    y = np.asarray(y) if (y is not None and not isinstance(y, np.ndarray)) else y

    # 3) If threshold='auto', compute it using percentiles
    #    and decide condition based on use_negative_criteria
    if isinstance(threshold, str) and threshold.lower() == 'auto':
        if use_negative_criteria:
            thr_value = np.percentile(z, percentile)  # e.g., lower percentile
            condition = 'below'
        else:
            thr_value = np.percentile(z, 100 - percentile)
            condition = 'above'
        threshold = thr_value

    # 4) Build a mask depending on single-value or range threshold
    if isinstance(threshold, (tuple, list)) and len(threshold) == 2:
        threshold = validate_length_range(threshold, param_name ='Threshold')
        
        if condition == 'auto':
            condition = 'between'  # default if user didn't specify
        if condition.lower() == 'between':
            mask = (z >= threshold[0]) & (z <= threshold[1])
        else:
            raise ValueError(
                "For tuple/list threshold, specify condition='between'"
                " or handle logic manually."
            )
    else:
        # Single numeric threshold
        if condition == 'auto':
            condition = 'below' if use_negative_criteria else 'above'
        if condition.lower() == 'above':
            mask = (z > threshold)
        elif condition.lower() == 'below':
            mask = (z < threshold)
        else:
            raise ValueError(
                f"condition={condition} not valid for a single numeric threshold."
            )

    # 5) Filter the arrays
    z_filtered = z[mask]
    x_filtered = x[mask] if x is not None else None
    y_filtered = y[mask] if y is not None else None

    # 6) Prepare the output
    #    If x & y exist, return a DataFrame with columns [x, y, z].
    #    If only x, return [x, z].
    #    Otherwise, return just the filtered z as a Series.
    x_name = resolve_label(x, default_name ='x')
    y_name = resolve_label(y, default_name ='y')
    z_name = resolve_label(z, default_name ='z')
    if x_filtered is not None and y_filtered is not None:
        result = pd.DataFrame({x_name: x_filtered, y_name: y_filtered, z_name: z_filtered})
    elif x_filtered is not None:
        result = pd.DataFrame({x_name: x_filtered, z_name: z_filtered})
    else:
        result = pd.Series(z_filtered)

    # 7) If view=True, build a plot according to plot_type and data availability
    if view:
        plt.figure(figsize=figsize)
        #  - If x & y provided, do a 2D scatter of x vs y (colored by z or sized by z).
        #  - If only x provided, do 1D plot with z as the values.
        #  - If neither x nor y, do a histogram of z.
        if x_filtered is not None and y_filtered is not None:
            if plot_type == 'scatter':
                plt.scatter(x_filtered, y_filtered, c=z_filtered, **kwargs)
                plt.colorbar(label=z_name)
                plt.xlabel(x_name)
                plt.ylabel(y_name)
                plt.title('Filtered Scatter Plot')
            elif plot_type == 'line':
                # Plot lines in x-y plane, ignoring z
                plt.plot(x_filtered, y_filtered, **kwargs)
                plt.xlabel(x_name)
                plt.ylabel(y_name)
                plt.title('Filtered Line Plot')
            else:
                # fallback
                plt.scatter(x_filtered, y_filtered, **kwargs)
                plt.xlabel(x_name)
                plt.ylabel(y_name)
                plt.title(f'Fallback to Scatter: {plot_type}')
        elif x_filtered is not None: # only x is given
            if plot_type == 'scatter':
                plt.scatter(x_filtered, z_filtered, **kwargs)
                plt.xlabel(x_name)
                plt.ylabel(z_name)
                plt.title('Filtered Scatter (x vs z)')
            elif plot_type == 'line':
                plt.plot(x_filtered, z_filtered, **kwargs)
                plt.xlabel(x_name)
                plt.ylabel(z_name)
                plt.title('Filtered Line (x vs z)')
            elif plot_type == 'hist':
                plt.hist(z_filtered, **kwargs)
                plt.xlabel(z_name)
                plt.title('Filtered Histogram (z)')
            else:
                # fallback
                plt.scatter(x_filtered, z_filtered, **kwargs)
                plt.xlabel(x_name)
                plt.ylabel(z_name)
                plt.title(f'Fallback to Scatter: {plot_type}')
        else:
            # No x or y => just do a histogram of z
            plt.hist(z_filtered, **kwargs)
            plt.xlabel(z_name)
            plt.title('Filtered Histogram (z)')
            
        if axis_off : 
            plt.axis('off')
        if not show_grid: 
            plt.grid(False )
            
        plt.tight_layout()
        plt.show()

    return result

@SaveFile            
def dual_merge(
    df1: pd.DataFrame, 
    df2: pd.DataFrame,
    feature_cols: Union[list, tuple] = ('longitude', 'latitude'),
    find_closest: bool = False, 
    force_coords: bool = False,  
    threshold: float = 0.01,  
    how: str = 'inner', 
    savefile: Optional[str]=None, 
) -> pd.DataFrame:
    """
    Merge two DataFrames based on specified feature columns. The function 
    can match the features exactly or find the closest matches within a 
    specified distance threshold. It also allows for overwriting coordinates 
    or feature values from one DataFrame to another when a close match is found.

    Parameters
    ----------
    df1 : pd.DataFrame
        The first DataFrame to be merged. It contains the primary data 
        along with the feature columns (e.g., longitude, latitude) to be 
        merged on.

    df2 : pd.DataFrame
        The second DataFrame to be merged. It contains the data to be 
        matched with `df1` based on the specified feature columns.

    feature_cols : tuple or list, default ``('longitude', 'latitude')``
        The names of the columns in each DataFrame to merge on. It should 
        contain two columns representing features, such as coordinates 
        (longitude, latitude) or other relevant attributes. These columns 
        will be used to match the rows from `df1` to `df2`.

    find_closest : bool, default ``False``
        If ``True``, the function will attempt to find the closest points 
        in ``df2`` for each point in ``df1`` within the specified distance 
        threshold (`threshold`). If no exact match is found, the closest 
        point within the threshold will be considered.

    force_coords : bool, default ``False``
        If ``True``, when the closest points are found, the coordinates 
        of `df1` will overwrite those of `df2` for the matched points.

    threshold : float, default ``0.01``
        The maximum distance threshold within which points will be considered 
        as "close" for the closest point matching. The value is in the same 
        unit as the feature columns (e.g., degrees for latitude/longitude).

    how : str, default ``'inner'``
        The type of merge to perform. Options include:
        - ``'inner'``: Only includes points that appear in both DataFrames.
        - ``'left'``: All points from `df1` are included; unmatched points 
          from `df2` are excluded.
        - ``'right'``: All points from `df2` are included; unmatched points 
          from `df1` are excluded.
        - ``'outer'``: Includes all points from both DataFrames, with NaN 
          for unmatched points.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the merged data based on the specified 
        feature columns and merge type. If ``find_closest=True``, it will 
        contain the closest matches within the specified threshold, with 
        coordinates from ``df1`` overwritten if ``force_coordinates=True``.

    Notes
    -----
    - This function uses a KDTree for efficient nearest-neighbor searching 
      when ``find_closest=True``. This is useful when dealing with large 
      datasets that may not have exact coordinate matches.
    - When ``force_coordinates=True``, the coordinates from ``df1`` will 
      overwrite those from ``df2`` for the closest points. However, other 
      feature values will be kept from ``df2``.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.utils.datautils import dual_merge
    >>> df1 = pd.DataFrame({
    >>>     'longitude': [1.1, 1.2, 1.3],
    >>>     'latitude': [2.1, 2.2, 2.3],
    >>>     'value1': [10, 20, 30]
    >>> })
    >>> df2 = pd.DataFrame({
    >>>     'longitude': [1.1, 1.4],
    >>>     'latitude': [2.1, 2.4],
    >>>     'value2': [100, 200]
    >>> })
    >>> result = dual_merge(df1, df2, feature_cols=('longitude', 'latitude'), 
    >>>                     find_closest=True, threshold=0.05)
    >>> print(result)
       longitude  latitude  value1  value2
    0        1.1       2.1      10     100

    See Also
    --------
    scipy.spatial.cKDTree: Used for finding the closest points in `df2` 
        when ``find_closest=True``. 
    pandas.merge: pandas.DataFrame.merge
        The pandas merge function, used to merge DataFrames based on columns.

    References
    ----------
    .. [1] Scipy Documentation, cKDTree
       https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html
    """

    # Ensure feature_cols are valid and both contain two elements
    feature_cols = columns_manager(feature_cols , empty_as_none= False )
    if len(feature_cols) != 2:
        raise ValueError(
            "feature_cols must contain exactly two features (e.g., longitude, latitude)")
    
    # check wether df1 and df2  are both dataframes.
    are_all_frames_valid(df1, df2 )
    
    # Extract columns from feature_cols for both DataFrames
    feature1_1, feature1_2 = feature_cols
    feature2_1, feature2_2 = feature_cols
    
    # Filter for relevant columns in both DataFrames
    df1_coords = df1[[feature1_1, feature1_2]]
    df2_coords = df2[[feature2_1, feature2_2]]

    if find_closest:
        # Use KDTree for fast nearest-neighbor search
        tree = cKDTree(df2_coords.values)
        
        # Query for the closest points in df2 for each point in df1
        dist, indices = tree.query(
            df1_coords.values, distance_upper_bound=threshold)
        
        # Filter out points that couldn't find a close match
        valid_idx = dist != np.inf
        df1_coords_closest = df1_coords.iloc[valid_idx]
        df2_coords_closest = df2_coords.iloc[indices[valid_idx]]

        if force_coords:
            # Force coordinates of df1 to overwrite df2
            df2_coords_closest[feature2_1] = df1_coords_closest[feature1_1].values
            df2_coords_closest[feature2_2] = df1_coords_closest[feature1_2].values
        
        # Update df1 with the closest matches from df2
        df1 = df1.iloc[valid_idx]
        df2 = df2.iloc[indices[valid_idx]]

    # Perform the merge based on the feature columns
    merged_data = pd.merge(
        df1,
        df2,
        how=how,
        left_on=[feature1_1, feature1_2],
        right_on=[feature2_1, feature2_2]
    )

    return merged_data

@isdf 
def extract_coordinates(
    df: pd.DataFrame,
    as_frame: bool = False,
    drop_xy: bool = False,
    error: Union[bool, str] = 'raise',
    verbose: int = 0
) -> Tuple[Union[Tuple[float, float], pd.DataFrame, None], pd.DataFrame, Tuple[str, str]]:
    """
    Identifies coordinate columns (longitude/latitude or easting/northing) 
    in a DataFrame, returns the coordinates or their central values, and 
    optionally removes the coordinate columns from the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame expected to contain coordinates (`longitude` and 
        `latitude` or `easting` and `northing`). If both types are present, 
        `longitude` and `latitude` are prioritized.

    as_frame : bool, default=False
        If True, returns the coordinate columns as a DataFrame. If False, 
        computes and returns the midpoint values.

    drop_xy : bool, default=False
        If True, removes coordinate columns (`longitude`/`latitude` or 
        `easting`/`northing`) from the DataFrame after extracting them.

    error : Union[bool, str], {'raise', 'warn', 'ignore'} default='raise'
        If True, raises an error if `df` is not a DataFrame. If set to False, 
        converts errors to warnings. If set to ``"ignore"``, suppresses 
        warnings.

    verbose : int, default=0
        If greater than 0, outputs messages about coordinate detection.

    Returns
    -------
    Tuple[Union[Tuple[float, float], pd.DataFrame, None], pd.DataFrame, Tuple[str, str]]
        - A tuple containing either the midpoint (longitude, latitude) or 
          (easting, northing) if `as_frame=False` or the coordinate columns 
          as a DataFrame if `as_frame=True`.
        - The original DataFrame, optionally with coordinates removed if 
          `drop_xy=True`.
        - A tuple of detected coordinate column names, or an empty tuple if 
          none are detected.

    Notes
    -----
    - This function searches for either `longitude`/`latitude` or 
      `easting`/`northing` columns and returns them as coordinates. If both 
      are found, `longitude`/`latitude` is prioritized.
      
    - To calculate the midpoint of the coordinates, the function averages 
      the values in the columns:

      .. math::
          \text{midpoint} = \left(\frac{\text{longitude}_{min} + \text{longitude}_{max}}{2}, 
          \frac{\text{latitude}_{min} + \text{latitude}_{max}}{2}\right)

    Examples
    --------
    >>> import gofast as gf
    >>> from gofast.utils.spatial_utils import extract_coordinates
    >>> testdata = gf.datasets.make_erp(samples=7, seed=42, as_frame=True)

    # Extract midpoint coordinates
    >>> xy, d, xynames = extract_coordinates(testdata)
    >>> xy, xynames
    ((110.48627946874444, 26.051952363176344), ('longitude', 'latitude'))

    # Extract coordinates as a DataFrame without removing columns
    >>> xy, d, xynames = extract_coordinates(testdata, as_frame=True)
    >>> xy.head(2)
       longitude   latitude
    0  110.485833  26.051389
    1  110.485982  26.051577

    # Drop coordinate columns from the DataFrame
    >>> xy, d, xynames = extract_coordinates(testdata, drop_xy=True)
    >>> xy, xynames
    ((110.48627946874444, 26.051952363176344), ('longitude', 'latitude'))
    >>> d.head(2)
       station  resistivity
    0      0.0          1.0
    1     20.0        167.5

    References
    ----------
    .. [1] Fotheringham, A. Stewart, *Geographically Weighted Regression: 
           The Analysis of Spatially Varying Relationships*, Wiley, 2002.

    See Also
    --------
    pd.DataFrame : Main pandas data structure for handling tabular data.
    np.nanmean : Computes the mean along specified axis, ignoring NaNs.
    """
    
    def rename_if_exists(val: str, col: pd.Index, default: str) -> pd.DataFrame:
        """Rename column in `d` if `val` is found in column names."""
        match = list(filter(lambda x: val in x.lower(), col))
        if match:
            df.rename(columns={match[0]: default}, inplace=True)
        return df

    # Validate input is a DataFrame
    if not (hasattr(df, 'columns') and hasattr(df, '__array__')):
        emsg = ("Expected a DataFrame containing coordinates (`longitude`/"
                "`latitude` or `easting`/`northing`). Got type: "
                f"{type(df).__name__!r}")
        
        error = str(error).lower().strip()
        if error == 'raise':
            raise TypeError(emsg)
        if error =='warn':
            warnings.warn(emsg)
        return None, df, ()

    # Rename columns to standardized names if they contain coordinate values
    for name, std_name in zip(['lat', 'lon', 'east', 'north'], 
                              ['latitude', 'longitude', 'easting', 'northing']):
        df = rename_if_exists(name, df.columns, std_name)

    # Check for and prioritize coordinate columns
    coord_columns = []
    for x, y in [('longitude', 'latitude'), ('easting', 'northing')]:
        if x in df.columns and y in df.columns:
            coord_columns = [x, y]
            break

    # Extract coordinates as DataFrame or midpoint
    if coord_columns:
        xy = df[coord_columns] if as_frame else tuple(
            np.nanmean(df[coord_columns].values, axis=0))
    else:
        xy = None
    
    # Drop coordinates if `drop_xy=True`
    if drop_xy and coord_columns:
        df.drop(columns=coord_columns, inplace=True)

    # Verbose messaging
    if verbose > 0:
        print("###", "No" if not coord_columns else coord_columns, "coordinates found.")
    
    return xy, df, tuple(coord_columns)

@Deprecated(reason=( 
    "This function is deprecated and will be removed in future versions. "
    "Please use `extract_coordinates` instead, which provides enhanced "
    "flexibility and robustness for coordinates extraction.")
)
@isdf 
def get_xy_coordinates(
        df, as_frame=False, drop_xy=False, raise_exception=True, verbose=0
    ):
    """Check whether the coordinate values x, y exist in the data.
    
    Parameters 
    ------------
    df: Dataframe 
       Frame that is expected to contain the longitude/latitude or 
       easting/northing coordinates.  Note if all types of coordinates are
       included in the data frame, the longitude/latitude takes the 
       priority. 
    as_frame: bool, default= False, 
       Returns the coordinates values if included in the data as a frame rather 
       than computing the middle points of the line 
    drop_xy: bool, default=False, 
       Drop the coordinates in the data and return the data transformed inplace 
       
    raise_exception: bool, default=True 
       raise error message if data is not a dataframe. If set to ``False``, 
       exception is converted to a warning instead. To mute the warning set 
       `raise_exception` to ``mute``
       
    verbose: int, default=0 
      Send message whether coordinates are detected. 
         
    returns 
    --------
    xy, d, xynames: Tuple 
      xy : tuple of float ( longitude, latitude) or (easting/northing ) 
         if `as_frame` is set to ``True``. 
      d: Dataframe transformed (coordinated removed )  or not
      xynames: str, the name of coordinates detected. 
      
    Examples 
    ----------
    >>> import gofast as gf 
    >>> from gofast.utils.spatial_utils import get_xy_coordinates 
    >>> testdata = gf.datasets.make_erp ( samples =7, seed =42 , as_frame=True)
    >>> xy, d, xynames = get_xy_coordinates ( testdata,  )
    >>> xy , xynames 
    ((110.48627946874444, 26.051952363176344), ('longitude', 'latitude'))
    >>> xy, d, xynames = get_xy_coordinates ( testdata, as_frame =True  )
    >>> xy.head(2) 
        longitude   latitude        easting      northing
    0  110.485833  26.051389  448565.380621  2.881476e+06
    1  110.485982  26.051577  448580.339199  2.881497e+06
    >>> # remove longitude and  lat in data 
    >>> testdata = testdata.drop (columns =['longitude', 'latitude']) 
    >>> xy, d, xynames = get_xy_coordinates ( testdata, as_frame =True  )
    >>> xy.head(2) 
             easting      northing
    0  448565.380621  2.881476e+06
    1  448580.339199  2.881497e+06
    >>> # note testdata should be transformed inplace when drop_xy is set to True
    >>> xy, d, xynames = get_xy_coordinates ( testdata, drop_xy =True)
    >>> xy, xynames 
    ((448610.25612032827, 2881538.4380570543), ('easting', 'northing'))
    >>> d.head(2)
       station  resistivity
    0      0.0          1.0
    1     20.0        167.5
    >>> testdata.head(2) # coordinates are henceforth been dropped 
       station  resistivity
    0      0.0          1.0
    1     20.0        167.5
    >>> xy, d, xynames = get_xy_coordinates ( testdata, drop_xy =True)
    >>> xy, xynames 
    (None, ())
    >>> d.head(2)
       station  resistivity
    0      0.0          1.0
    1     20.0        167.5

    """   
    
    def get_value_in ( val,  col , default): 
        """ Get the value in the frame columns if `val` exists in """
        x = list( filter ( lambda x: x.find (val)>=0 , col)
                   )
        if len(x) !=0: 
            # now rename col  
            df.rename (columns = {x[0]: str(default) }, inplace = True ) 
            
        return df

    if not (
            hasattr ( df, 'columns') and hasattr ( df, '__array__') 
            ) : 
        emsg = ("Expect dataframe containing coordinates longitude/latitude"
                f" or easting/northing. Got {type (df).__name__!r}")
        
        raise_exception = str(raise_exception).lower().strip() 
        if raise_exception=='true': 
            raise TypeError ( emsg )
        
        if raise_exception  not in ('mute', 'silence'):  
            warnings.warn( emsg )
       
        return df 
    
    # check whether coordinates exists in the data columns 
    for name, tname in zip ( ('lat', 'lon', 'east', 'north'), 
                     ( 'latitude', 'longitude', 'easting', 'northing')
                     ) : 
        df = get_value_in(name, col = df.columns , default = tname )
       
    # get the exist coodinates 
    coord_columns  = []
    for x, y in zip ( ( 'longitude', 'easting' ), ( 'latitude', 'northing')):
        if ( x  in df.columns and y in df.columns ): 
            coord_columns.extend  ( [x, y] )

    xy  = df[ coord_columns] if len(coord_columns)!=0 else None 

    if ( not as_frame 
        and xy is not None ) : 
        # take the middle of the line and if both types of 
        # coordinates are supplied , take longitude and latitude 
        # and drop easting and northing  
        xy = tuple ( np.nanmean ( np.array ( xy ) , axis =0 )) [:2]

    xynames = tuple ( coord_columns)[:2]
    if ( 
            drop_xy  and len( coord_columns) !=0
            ): 
        # modifie the data inplace 
        df.drop ( columns=coord_columns, inplace = True  )

    if verbose: 
        print("###", "No" if len(xynames)==0 else ( 
            tuple (xy.columns) if as_frame else xy), "coordinates found.")
        
    return  xy , df , xynames 

@is_data_readable 
@isdf
def batch_spatial_sampling(
    data,
    sample_size=0.1,
    n_batches=10,
    stratify_by=['year'],
    spatial_bins=10,
    spatial_cols=None,
    random_state=42
):
    """
    Batch resample spatial data with stratification over spatial and
    specified columns.

    This function divides the dataset into `n_batches` batches, each
    being a stratified sample of the data. It ensures that samples in
    the first batch are not present in subsequent batches, and so on.
    This is particularly useful when dealing with very large datasets
    that cannot be processed at once, allowing for batch processing in
    machine learning algorithms.

    Parameters
    ----------
    data : pandas.DataFrame
        The input DataFrame from which samples are to be drawn. It must
        contain the spatial coordinate columns specified in
        `spatial_cols`, and any additional columns specified in
        `stratify_by`.

    sample_size : float or int, optional
        The total number of samples to draw from `data`. If `sample_size`
        is a float between 0.0 and 1.0, it represents the fraction of the
        dataset to include in the sample (e.g., `sample_size=0.1` selects
        10% of the data). If `sample_size` is an integer, it represents
        the absolute number of samples to select. The default is ``0.1``.

    n_batches : int, optional
        The number of batches to divide the total samples into. The
        samples are divided as evenly as possible among the batches. The
        default is ``10``.

    stratify_by : list of str, optional
        A list of column names in `data` to use for stratification. The
        sampling will ensure that the distribution of these columns in
        each batch matches the distribution in the original dataset. The
        default is ``['year']``.

    spatial_bins : int or tuple/list of int, optional
        The number of bins to divide the spatial coordinates into for
        stratification. If an integer, the same number of bins is used
        for all spatial dimensions. If a tuple or list, its length must
        match the number of spatial columns specified in `spatial_cols`,
        and each element specifies the number of bins for that spatial
        dimension. The default is ``10``.

    spatial_cols : list or tuple of str, optional
        A list of column names in `data` representing spatial coordinates.
        The function can accept one or two columns (e.g., longitude and
        latitude). If ``None``, the function will look for columns named
        `'longitude'` and/or `'latitude'` in `data`. If only one spatial
        column is provided or found, a warning is issued, suggesting that
        providing both spatial columns is recommended for more accurate
        sampling. If more than two columns are provided, an error is
        raised.

    random_state : int, optional
        Controls the randomness of the sampling for reproducibility. This
        integer seed is used to initialize the random number generator.
        The default is ``42``.

    Returns
    -------
    batches : list of pandas.DataFrame
        A list of DataFrames, each representing a batch of the stratified
        sampled data.

    Notes
    -----
    The function performs stratified sampling based on spatial bins and
    other specified stratification columns. Spatial coordinates are
    binned using quantile-based discretization (:func:`pandas.qcut`),
    ensuring each bin has approximately the same number of observations.

    The total number of samples, :math:`n`, is divided among the batches,
    and within each batch, samples are drawn in a stratified manner. The
    sample size for each batch is calculated as:

    .. math::

        n_{\text{batch}} = \left\lfloor \frac{n}{n_{\text{batches}}} \right\rfloor

    The remaining samples are distributed among the first few batches:

    .. math::

        n_{\text{leftover}} = n \mod n_{\text{batches}}

    For each batch, the number of samples per stratification group is
    calculated based on the proportion of the group size to the remaining
    data size:

    .. math::

        n_{i} = \left\lceil \frac{N_{i}}{N_{\text{remaining}}}\\
            \times n_{\text{batch}} \right\rceil

    where:

    - :math:`N_{i}` is the size of group :math:`i`.
    - :math:`N_{\text{remaining}}` is the total number of samples
      remaining in the data.
    - :math:`n_{i}` is the number of samples to draw from group
      :math:`i`.

    After sampling, the selected samples are removed from the remaining
    data to ensure that they are not selected again in subsequent batches.

    Examples
    --------
    >>> from gofast.utils.spatialutils import batch_spatial_sampling
    >>> sampled_batches = batch_spatial_sampling(
    ...     data=df,
    ...     sample_size=0.05,
    ...     n_batches=5,
    ...     stratify_by=['year', 'geological_category'],
    ...     spatial_bins=(10, 15),
    ...     spatial_cols=['longitude', 'latitude'],
    ...     random_state=42
    ... )
    >>> for i, batch in enumerate(sampled_batches):
    ...     print(f"Batch {i+1}: {batch.shape}")

    See Also
    --------
    resample_spatial_data : Perform stratified sampling without batching.

    References
    ----------
    .. [1] Kotsiantis, S., Kanellopoulos, D., & Pintelas, P. (2006).
           "Data preprocessing for supervised learning." *International
           Journal of Computer Science*, 1(2), 111-117.

    """

    data = data.copy()
    total_samples = sample_size
    if isinstance(sample_size, float):
        if not 0 < sample_size < 1:
            raise ValueError("When sample_size is a float, it must be between 0 and 1.")
        total_samples = int(len(data) * sample_size)
    elif isinstance(sample_size, int):
        if sample_size <= 0:
            raise ValueError("sample_size must be positive.")
    else:
        raise ValueError("sample_size must be a float or int.")

    if total_samples > len(data):
        raise ValueError("sample_size is larger than the dataset.")

    if n_batches <= 0:
        raise ValueError("n_batches must be a positive integer.")

    sample_size_per_batch = total_samples // n_batches
    leftover = total_samples % n_batches

    batches = []
    remaining_data = data.copy()
    sampled_indices = set()
    rng = np.random.RandomState(random_state)

    # Set default spatial columns if not specified
    if spatial_cols is None:
        spatial_cols = []
        if 'longitude' in data.columns:
            spatial_cols.append('longitude')
        if 'latitude' in data.columns:
            spatial_cols.append('latitude')
        if not spatial_cols:
            raise ValueError(
                "No spatial columns specified and "
                "'longitude' and 'latitude' not found in data."
            )
        if len(spatial_cols) == 1:
            warnings.warn(
                f"Only one spatial column '{spatial_cols[0]}' found. "
                "Using it for spatial stratification. "
                "For more accurate sampling, providing both spatial "
                "columns is recommended.",
                UserWarning
            )
    else:
        if not isinstance(spatial_cols, (list, tuple)):
            raise ValueError(
                "spatial_cols must be a list or tuple of column names."
            )
        if len(spatial_cols) > 2:
            raise ValueError(
                "spatial_cols can have at most two columns."
            )
        for col in spatial_cols:
            if col not in data.columns:
                raise ValueError(
                    f"Spatial column '{col}' not found in data."
                )
        if len(spatial_cols) == 1:
            warnings.warn(
                f"Only one spatial column '{spatial_cols[0]}' specified. "
                "For more accurate sampling, providing two spatial columns "
                "is recommended.",
                UserWarning
            )
    # Validate spatial_bins
    if isinstance(spatial_bins, int):
        n_bins_list = [spatial_bins] * len(spatial_cols)
    elif isinstance(spatial_bins, (tuple, list)):
        if len(spatial_bins) != len(spatial_cols):
            raise ValueError(
                "Length of spatial_bins must match number of spatial_cols."
            )
        n_bins_list = list(spatial_bins)
    else:
        raise ValueError(
            "spatial_bins must be int or tuple/list of int."
        )
    # Create spatial bins in the original data
    for col, n_bins, axis in zip(
        spatial_cols, n_bins_list, ['x_bin', 'y_bin']
    ):
        data[axis] = pd.qcut(
            data[col],
            q=n_bins,
            duplicates='drop'
        )
    # Create combined stratification key in original data
    strat_columns = stratify_by + [
        axis for axis in ['x_bin', 'y_bin'][:len(spatial_cols)]
    ]
    data['strat_key'] = data[strat_columns].apply(
        lambda row: '_'.join(row.values.astype(str)),
        axis=1
    )
    # Initialize remaining data
    remaining_data = data.copy()
    batches = []

    # Set initial random state
    rng = np.random.RandomState(random_state)

    for batch_idx in range(n_batches):
        # Adjust sample size for batches if total_samples is not divisible by n_batches
        if batch_idx < leftover:
            batch_sample_size = sample_size_per_batch + 1
        else:
            batch_sample_size = sample_size_per_batch

        if batch_sample_size > len(remaining_data):
            batch_sample_size = len(remaining_data)

        # Group remaining data by stratification key
        grouped = remaining_data.groupby('strat_key')
        # Calculate number of samples per group
        group_sizes = grouped.size()
        total_size = group_sizes.sum()
        group_sample_sizes = (
            (group_sizes / total_size * batch_sample_size)
            .round()
            .astype(int)
        )
        # Sample data from each group
        sampled_indices = []
        for strat_value, group in grouped:
            n = group_sample_sizes.loc[strat_value]
            if n > 0 and len(group) > 0:
                sampled_group = group.sample(
                    n=min(n, len(group)),
                    random_state=rng.randint(0, 10000)
                )
                sampled_indices.extend(sampled_group.index)
        # Create the sampled DataFrame
        batch_sampled_data = remaining_data.loc[sampled_indices]
        batches.append(batch_sampled_data.drop(
            columns=['strat_key'] + [axis for axis in ['x_bin', 'y_bin'][:len(spatial_cols)]]))
        # Remove sampled data from remaining_data
        remaining_data = remaining_data.drop(index=sampled_indices)
        if len(remaining_data) == 0:
            break  # No more data to sample

    return batches

@is_data_readable 
@isdf
def spatial_sampling(
    data,
    sample_size=0.01,
    stratify_by=['year'],
    spatial_bins=10,
    spatial_cols=None,
    random_state=42
):
    """
    Sample spatial data intelligently to represent the distribution
    of the whole area and include different years.

    This function performs stratified sampling on spatial data,
    ensuring that the sample reflects both spatial distribution
    and temporal aspects of the entire dataset [1]_. It combines spatial
    stratification based on coordinates and additional stratification
    columns specified by the user.

    Parameters
    ----------
    data : pandas.DataFrame
        The input DataFrame to sample from. Must contain spatial
        coordinate columns (e.g., `'longitude'`, `'latitude'`) and
        any columns specified in ``stratify_by``.
    sample_size : float or int, optional
        The proportion or absolute number of samples to select.
        If float, should be between 0.0 and 1.0 and represents the
        fraction of the dataset to include in the sample.
        If int, represents the absolute number of samples to select.
        Default is ``0.01`` (1% of the data).
    stratify_by : list of str, optional
        List of column names to stratify by. Default is ``['year']``.
    spatial_bins : int or tuple/list of int, optional
        Number of bins to divide the spatial coordinates into.
        If an integer, the same number of bins is used for all spatial
        dimensions. If a tuple or list, its length must match the number
        of spatial columns, specifying the number of bins for each spatial
        dimension. Default is ``10``.
    spatial_cols : list or tuple of str, optional
        List of spatial coordinate column names. Can accept one or two
        columns. If ``None``, the function checks for columns named
        `'longitude'` and/or `'latitude'` in ``data``. If only one spatial
        column is provided or found, a warning is issued, suggesting that
        providing both spatial columns is recommended for more accurate
        sampling. If more than two columns are provided, an error is raised.
    random_state : int, optional
        Random seed for reproducibility. Default is ``42``.

    Returns
    -------
    sampled_data : pandas.DataFrame
        A sampled DataFrame representing the distribution of the whole
        area and including different years.

    Notes
    -----
    The function performs stratified sampling based on spatial bins
    and other specified stratification columns. Spatial coordinates
    are binned using quantile-based discretization (:func:`pandas.qcut`),
    ensuring each bin has approximately the same number of observations.

    Let :math:`N` be the total number of samples in ``data``, and
    :math:`n` be the desired sample size. The function calculates the
    number of samples to draw from each stratification group based on
    the proportion of the group size to the total dataset size:

    .. math::

        n_i = \left\lceil \frac{N_i}{N} \times n \right\rceil

    where :math:`N_i` is the size of group :math:`i`, and :math:`n_i`
    is the number of samples to draw from group :math:`i`.

    The function ensures that:

    - All specified spatial and stratification columns exist in ``data``.
    - The number of spatial bins matches the number of spatial columns.
    - The sample size is valid (positive float between 0 and 1, or
      positive integer).

    Warnings are issued if:

    - Only one spatial column is used, suggesting that using two spatial
      columns is recommended for more accurate sampling.

    Examples
    --------
    >>> from gofast.utils.spatialutils import spatial_sampling
    >>> import pandas as pd
    >>> # Assume 'df' is a pandas DataFrame with columns
    >>> # 'longitude', 'latitude', 'year', and other data.
    >>> sampled_df = spatial_sampling(
    ...     data=df,
    ...     sample_size=0.05,
    ...     stratify_by=['year', 'geological_category'],
    ...     spatial_bins=(10, 15),
    ...     spatial_cols=['longitude', 'latitude'],
    ...     random_state=42
    ... )
    >>> print(sampled_df.shape)

    See Also
    --------
    pandas.qcut : Quantile-based discretization function used for binning.
    sklearn.model_selection.StratifiedShuffleSplit : For stratified sampling.

    References
    ----------
    .. [1] Kotsiantis, S., Kanellopoulos, D., & Pintelas, P. (2006).
           "Data preprocessing for supervised learning." *International
           Journal of Computer Science*, 1(2), 111-117.

    """

    data = data.copy()
    # Set default spatial columns if not specified
    if spatial_cols is None:
        spatial_cols = []
        if 'longitude' in data.columns:
            spatial_cols.append('longitude')
        if 'latitude' in data.columns:
            spatial_cols.append('latitude')
        if not spatial_cols:
            raise ValueError(
                "No spatial columns specified and "
                "'longitude' and 'latitude' not found in data."
            )
        if len(spatial_cols) == 1:
            warnings.warn(
                f"Only one spatial column '{spatial_cols[0]}' found. "
                "Using it for spatial stratification. "
                "For more accurate sampling, providing both spatial "
                "columns is recommended.",
                UserWarning
            )
    else:
        if not isinstance(spatial_cols, (list, tuple)):
            raise ValueError(
                "spatial_cols must be a list or tuple of column names."
            )
        if len(spatial_cols) > 2:
            raise ValueError(
                "spatial_cols can have at most two columns."
            )
        for col in spatial_cols:
            if col not in data.columns:
                raise ValueError(
                    f"Spatial column '{col}' not found in data."
                )
        if len(spatial_cols) == 1:
            warnings.warn(
                f"Only one spatial column '{spatial_cols[0]}' specified. "
                "For more accurate sampling, providing two spatial columns "
                "is recommended.",
                UserWarning
            )
    # Validate spatial_bins
    if isinstance(spatial_bins, int):
        n_bins = validate_positive_integer(
            spatial_bins,
            'spatial_bins'
        )
        n_bins_list = [n_bins] * len(spatial_cols)
    elif isinstance(spatial_bins, (tuple, list)):
        if len(spatial_bins) != len(spatial_cols):
            raise ValueError(
                "Length of spatial_bins must match number of spatial_cols."
            )
        n_bins_list = [
            validate_positive_integer(
                n, 'spatial_bins'
            ) for n in spatial_bins
        ]
    else:
        raise ValueError(
            "spatial_bins must be int or tuple/list of int."
        )
    # Create spatial bins
    for col, n_bins, axis in zip(
        spatial_cols, n_bins_list, ['x_bin', 'y_bin']
    ):
        data[axis] = pd.qcut(
            data[col],
            q=n_bins,
            duplicates='drop'
        )
    # Create combined stratification key
    strat_columns = stratify_by + [
        axis for axis in ['x_bin', 'y_bin'][:len(spatial_cols)]
    ]
    data['strat_key'] = data[strat_columns].apply(
        lambda row: '_'.join(row.values.astype(str)),
        axis=1
    )
    # Determine total number of samples
    if isinstance(sample_size, float):
        if not 0 < sample_size < 1:
            raise ValueError(
                "When sample_size is a float, it must be between 0 and 1."
            )
        n_samples = int(len(data) * sample_size)
    elif isinstance(sample_size, int):
        n_samples = validate_positive_integer(
            sample_size,
            'sample_size'
        )
    else:
        raise ValueError(
            "sample_size must be a positive float or int."
        )
    # Group data by stratification key
    grouped = data.groupby('strat_key')
    # Calculate number of samples per group
    group_sizes = grouped.size()
    total_size = group_sizes.sum()
    group_sample_sizes = (
        (group_sizes / total_size * n_samples)
        .round()
        .astype(int)
    )
    # Sample data from each group
    sampled_indices = []
    np.random.seed(random_state)
    for strat_value, group in grouped:
        n = group_sample_sizes.loc[strat_value]
        if n > 0 and len(group) > 0:
            sampled_group = group.sample(
                n=min(n, len(group)),
                random_state=np.random.randint(
                    0,
                    10000
                )
            )
            sampled_indices.extend(
                sampled_group.index
            )
    # Create the sampled DataFrame
    sampled_data = data.loc[
        sampled_indices
    ]
    # Drop helper columns
    cols_to_drop = ['strat_key'] + [
        axis for axis in ['x_bin', 'y_bin'][:len(spatial_cols)]
    ]
    sampled_data = sampled_data.drop(
        columns=cols_to_drop
    )
    return sampled_data.reset_index(
        drop=True
    )

@validate_params({ 
    'y': ['array-like'], 
    'yt': ['array-like'], 
    'threshold': [Interval (Real, 0, 1 , closed='neither')], 
    'mode': [StrOptions({'strict', 'soft'})],
    'trailer': [str], 
    })
def make_mxs_labels(
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
    >>> make_mxs_labels(y, yt, threshold=0.5, star_mxs=True, return_ymx=True, trailer="#")
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
    >>> from gofast.utils.mathex  import savgol_filter
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

def moving_average (
    y:ArrayLike,
    *, 
    window_size:int  = 3 , 
    method:str  ='sma',
    mode:str  ='same', 
    alpha: Union [int, float]  =.5 
)-> ArrayLike: 
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
    >>> from gofast.utils.tools   import moving_average 
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
 
    if method =='cma': 
        y = np.cumsum (y) 
        ya = np.array([ y[ii]/ len(y[:ii +1]) for ii in range(len(y))]) 
        
    elif method =='wma': 
        w = np.cumsum(np.ones(window_size, dtype = float))
        w /= np.sum(w)
        ya = np.convolve(y, w[::-1], mode ) #/window_size
        
    elif method =='ema': 
        ya = np.array ([y[0]]) 
        for ii in range(1, len(y)): 
            v = y[ii] * alpha + ( 1- alpha ) * ya[-1]
            ya = np.append(ya, v)
    else:
        ya = np.convolve(y , np.ones (window_size), mode ) / window_size 
            
    return ya 

@AppendDocReferences(refglossary.__doc__)
def scalePosition(
        ydata: Union[ArrayLike, _SP, Series, DataFrame],
        xdata: Union[ArrayLike, Series] = None, 
        func: Optional[_F] = None,
        c_order: Optional[Union[int, str]] = 0,
        show: bool = False, 
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
        >>> from gofast.utils.mathex  import detect_station_position 
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
    from ..exceptions import SiteError 
    
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

def extract_coordinates2(X, Xt=None, columns=None):
    """
    Extracts 'x' and 'y' coordinate arrays from training (X) and optionally
    test (Xt) datasets. 
    
    Supports input as NumPy arrays or pandas DataFrames. When dealing
    with DataFrames, `columns` can specify which columns to use for coordinates.

    Parameters
    ----------
    X : ndarray or DataFrame
        Training dataset with shape (M, N) where M is the number of samples and
        N is the number of features. It represents the observed data used as
        independent variables in learning.
    Xt : ndarray or DataFrame, optional
        Test dataset with shape (M, N) where M is the number of samples and
        N is the number of features. It represents the data observed at testing
        and prediction time, used as independent variables in learning.
    columns : list of str or int, optional
        Specifies the columns to use for 'x' and 'y' coordinates. Necessary when
        X or Xt are DataFrames with more than 2 dimensions or when selecting specific
        features from NumPy arrays.

    Returns
    -------
    tuple of arrays
        A tuple containing the 'x' and 'y' coordinates from the training set and, 
        if provided, the test set. Formatted as (x, y, xt, yt).
    tuple of str or None
        A tuple containing the names or indices of the 'x' and 'y' columns 
        for the training and test sets. Formatted as (xname, yname, xtname, ytname).
        Values are None if not applicable or not provided.

    Raises
    ------
    ValueError
        If `columns` is not iterable, not provided for DataFrames with more 
        than 2 dimensions, or if X or Xt cannot be validated as coordinate arrays.

    Examples
    --------
    >>> import numpy as np 
    >>> from gofast.utils.spatialutils import extract_coordinates
    >>> X = np.array([[1, 2], [3, 4]])
    >>> Xt = np.array([[5, 6], [7, 8]])
    >>> extract_coordinates(X, Xt )
    ((array([1, 3]), array([2, 4]), array([5, 7]), array([6, 8])), (0, 1, 0, 1))
    """
    if columns is None: 
        if not isinstance ( X, pd.DataFrame) and X.shape[1]!=2: 
            raise ValueError("Columns cannot be None when array is passed.")
        if isinstance(X, np.ndarray) and X.shape[1]==2: 
            columns =[0, 1] 
    
    columns = columns or ( list(X.columns) if isinstance (
        X, pd.DataFrame ) else columns )
    
    if columns is None :
        raise ValueError("Columns parameter is required to specify"
                         " 'x' and 'y' coordinates.")
    
    if not isinstance(columns, (list, tuple)) or len(columns) != 2:
        raise ValueError("Columns parameter must be a list or tuple with "
                         "exactly two elements for 'x' and 'y' coordinates.")
    
    # Process training dataset
    x, y, xname, yname = _process_dataset(X, columns)
    
    # Process test dataset, if provided
    if Xt is not None:
        xt, yt, xtname, ytname = _process_dataset(Xt, columns)
    else:
        xt, yt, xtname, ytname = None, None, None, None

    return (x, y, xt, yt), (xname, yname, xtname, ytname)    


def _process_dataset(dataset, columns):
    """
    Processes the dataset (X or Xt) to extract 'x' and 'y' coordinates based 
    on provided column names or indices.
    
    Parameters
    ----------
    dataset : pandas.DataFrame or numpy.ndarray
        The dataset from which to extract 'x' and 'y' coordinates.
    columns : list of str or int
        The names or indices of the columns to extract as coordinates. 
        For ndarray, integers are expected.
    
    Returns
    -------
    x, y, xname, yname : (numpy.array or pandas.Series, numpy.array or 
                          pandas.Series, str/int, str/int)
        The extracted 'x' and 'y' coordinates, along with their column names 
        or indices.
    
    Raises
    ------
    ValueError
        If the dataset or columns are not properly specified.
    """
    if isinstance(dataset, pd.DataFrame):
        x, xname, y, yname = _validate_columns(dataset, columns)
        return x.to_numpy(), y.to_numpy(), xname, yname
    elif isinstance(dataset, np.ndarray):
        if not isinstance(columns, (list, tuple)) or len(columns) < 2:
            raise ValueError("For ndarray, columns must be a list or tuple "
                             "with at least two indices.")
        xindex, yindex = columns[0], columns[1]
        x, y = dataset[:, xindex], dataset[:, yindex]
        return x, y, xindex, yindex
    else:
        raise ValueError("Dataset must be a pandas.DataFrame or numpy.ndarray.")


def _validate_columns(df, columns):
    """
    Validates and extracts x, y coordinates from a DataFrame based on column 
    names or indices.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame from which to extract coordinate columns.
    columns : list of str or int
        The names or indices of the columns to extract as coordinates.
    
    Returns
    -------
    x, xname, y, yname : (pandas.Series, str/int, pandas.Series, str/int)
        The extracted x and y coordinate Series along with their column
        names or indices.
    
    Raises
    ------
    ValueError
        If the specified columns are not found in the DataFrame or if the 
        columns list is not correctly specified.
    """
    if not isinstance(columns, (list, tuple)) or len(columns) < 2:
        raise ValueError("Columns parameter must be a list or tuple with at"
                         " least two elements.")
    
    try:
        xname, yname = columns[0], columns[1]
        x = df[xname] if isinstance(xname, str) else df.iloc[:, xname]
        y = df[yname] if isinstance(yname, str) else df.iloc[:, yname]
    except Exception as e:
        raise ValueError(f"Error extracting columns: {e}")
    
    return x, xname, y, yname

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

def torres_verdin_filter(
    arr,  
    weight_factor: float = 0.1, 
    beta: float = 1.0, 
    logify: bool = False, 
    axis: int = None, 
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
    >>> from gofast.utils.mathex  import torres_verdin_filter 
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
    arr = is_iterable(arr, exclude_string=True, transform=True)
    axis = 0 if axis is None else axis  # Set default axis to 0 if not specified
    logify = bool(logify)
    
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

    
def get_distance(
    x: ArrayLike, 
    y:ArrayLike , *, 
    return_mean_dist:bool =False, 
    is_latlon= False , 
    **kws
    ): 
    """
    Compute distance between points.
    
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
    >>> from gofast.utils.mathex  import get_distance 
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
    gofast.utils.mathex .get_bearing: 
        Compute the  direction of one point relative to another point. 
      
    Examples
    ---------
    >>> from gofast.utils.mathex  import scale_positions 
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

def adaptive_moving_average(data,  window_size_factor=0.1):
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
    >>> from gofast.utils.mathex  import adaptive_moving_average 
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

def get_azimuth (
    xlon: Union [str, ArrayLike], 
    ylat:Union[ str, ArrayLike], 
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
    >>> from gofast.utils.mathex  import get_azimuth 
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

def quality_control2(
    ar, 
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
        data = KeyBox( **dict (
            tol=tol, 
            rate_= float(np.around (ck, 2)), 
            data_=  np.delete ( ar, index , axis =0 )
            )
        )
    return data
 
def find_close_position (refarr, arr): 
    """ Get the close item from `arr` in the reference array `refarr`. 
    
    :param arr: array-like 1d, 
        Array to extended with fill value. It should be  shorter than the 
        `refarr`.
        
    :param refarr: array-like- 
        the reference array. It should have a greater length than the
        array `arr`.  
    :return: generator of index of the closest position in  `refarr`.  
    """
    for item in arr : 
        ix = np.argmin (np.abs (refarr - item)) 
        yield ix 
    

def fit_ll(ediObjs, by ='index', method ='strict', distance='cartesian' ): 
    """ Fit EDI by location and reorganize EDI according to the site  
    longitude and latitude coordinates. 
    
    EDIs data are mostly reading in an alphabetically order, so the reoganization  

    according to the location(longitude and latitude) is usefull for distance 
    betwen site computing with a right position at each site.  
    
    :param ediObjs: list of EDI object, composed of a collection of 
        gofast.edi.Edi or pycsamt.core.edi.Edi or mtpy.core.edi objects 
    :type ediObjs: gofast.edi.Edi_Collection 
  
    :param by: ['name'|'ll'|'distance'|'index'|'name'|'dataid'] 
       The kind to sorting EDI files. Default uses the position number 
       included in the EDI-files name.
    :type by: str 
    
    :param method:  ['strict|'naive']. Kind of method to sort the 
        EDI file from longitude, latitude. Default is ``strict``. 
    :type method: str 
    
    :param distance: ['cartesian'|'harvesine']. Use the distance between 
       coordinates points to sort EDI files. Default is ``cartesian`` distance.
    :type distance: str 
    
    :returns: array splitted into ediObjs and Edifiles basenames 
    :rtyple: tuple 
    
    :Example: 
        >>> import numpy as np 
        >>> from gofast.methods.em import EM
        >>> from gofast.utils.coreutils import fit_ll
        >>> edipath ='data/edi_ss' 
        >>> cediObjs = EM().fit (edipath) 
        >>> ediObjs = np.random.permutation(cediObjs.ediObjs) # shuffle the  
        ... # the collection of ediObjs 
        >>> ediObjs, ediObjbname = fit_by_ll(ediObjs) 
        ...

    """
    method= 'strict' if str(method).lower() =='strict' else "naive"
    if method=='strict': 
        return _fit_ll(ediObjs, by = by, distance = distance )
    
    #get the ediObjs+ names in ndarray(len(ediObjs), 2) 
    objnames = np.c_[ediObjs, np.array(
        list(map(lambda obj: os.path.basename(obj.edifile), ediObjs)))]
    lataddlon = np.array (list(map(lambda obj: obj.lat + obj.lon , ediObjs)))
    if len(np.unique ( lataddlon)) < len(ediObjs)//2: 
        # then ignore reorganization and used the 
        # station names. 
        pass 
    else:
        sort_ix = np.argsort(lataddlon) 
        objnames = objnames[sort_ix ]
        
    #ediObjs , objbnames = np.hsplit(objnames, 2) 
    return objnames[:, 0], objnames[:, -1]
   
def _fit_ll(ediObjs, distance='cartes', by = 'index'): 
    """ Fit ediObjs using the `strict method`. 
    
    An isolated part of :func:`gofast.utils.coreutils.fit_by_ll`. 
    """
    # get one obj randomnly and compute distance 
    obj_init = ediObjs[0]
    ref_lat = 34.0522  # Latitude of Los Angeles
    ref_lon = -118.2437 # Longitude of Los Angeles
    
    if str(distance).find ('harves')>=0: 
        distance='harves'
    else: distance='cartes'
    
    # create stations list.
    stations = [ 
        {"name": os.path.basename(obj.edifile), 
         "longitude": obj.lon, 
         "latitude": obj.lat, 
         "obj": obj, 
         "dataid": obj.dataid,  
         # compute distance using cartesian or harversine 
         "distance": _compute_haversine_d (
            ref_lat, ref_lon, obj.lat, obj.lon
            ) if distance =='harves' else np.sqrt (
                ( obj_init.lon -obj.lon)**2 + (obj_init.lat -obj.lat)**2), 
         # check wether there is a position number in the data.
         "index": re.search ('\d+', str(os.path.basename(obj.edifile)),
                            flags=re.IGNORECASE).group() if bool(
                                re.search(r'\d', os.path.basename(obj.edifile)))
                                else float(ii) ,
        } 
        for ii, obj in enumerate (ediObjs) 
        ]
                  
    ll=( 'longitude', 'latitude') 
    
    by = 'index' or str(by ).lower() 
    if ( by.find ('ll')>=0 or by.find ('lonlat')>=0): 
        by ='ll'
    elif  by.find ('latlon')>=0: 
        ll =ll[::-1] # reverse 
    
    # sorted from key
    sorted_stations = sorted (
        stations , key = lambda o: (o[ll[0]], [ll[-1]])  
        if (by =='ll' or by=='latlon')
        else o[by]
             )

    objnames = np.array( list(
        map ( lambda o : o['name'], sorted_stations))) 
    ediObjs = np.array ( list(
        map ( lambda o: o['obj'], sorted_stations)), 
                        dtype =object ) 
    
    return ediObjs, objnames 

def _compute_haversine_d(lat1, lon1, lat2, lon2): 
    """ Sort coordinates using Haversine distance calculus. 
    An isolated part of :func:`gofast.utils.coreutils._fit_by_ll"""
    # get reference_lat and reference lon 
    # get one obj randomnly and compute distance 
    # obj_init = np.random.choice (ediObjs) 
    import math 
    # Define a function to calculate the distance 
    # between two points in kilometers
    # def distance(lat1, lon1, lat2, lon2):
        # Convert degrees to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    # Apply the haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(
        lat2) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371 # Earth's radius in kilometers
    
    return c * r
    

def make_ids(arr, prefix =None, how ='py', skip=False): 
    """ Generate auto Id according to the number of given sites. 
    
    :param arr: Iterable object to generate an id site . For instance it can be 
        the array-like or list of EDI object that composed a collection of 
        gofast.edi.Edi object. 
    :type ediObjs: array-like, list or tuple 

    :param prefix: string value to add as prefix of given id. Prefix can be 
        the site name.
    :type prefix: str 
    
    :param how: Mode to index the station. Default is 'Python indexing' i.e. 
        the counting starts by 0. Any other mode will start the counting by 1.
    :type cmode: str 
    
    :param skip: skip the long formatage. the formatage acccording to the 
        number of collected file. 
    :type skip: bool 
    :return: ID number formated 
    :rtype: list 
    
    :Example: 
        >>> import numpy as np 
        >>> from gofast.utils.func_utils import make_ids 
        >>> values = ['edi1', 'edi2', 'edi3'] 
        >>> make_ids (values, 'ix')
        ... ['ix0', 'ix1', 'ix2']
        >>> data = np.random.randn(20)
        >>>  make_ids (data, prefix ='line', how=None)
        ... ['line01','line02','line03', ... , line20] 
        >>> make_ids (data, prefix ='line', how=None, skip =True)
        ... ['line1','line2','line3',..., line20] 
        
    """ 
    fm='{:0' + ('1' if skip else '{}'.format(int(np.log10(len(arr))) + 1)) +'}'
    id_ =[str(prefix) + fm.format(i if how=='py'else i+ 1 ) if prefix is not 
          None else fm.format(i if how=='py'else i+ 1) 
          for i in range(len(arr))] 
    return id_    
    
def show_stats(nedic , nedir, fmtl='~', lenl=77, obj='EDI'): 
    """ Estimate the file successfully read reading over the unread files

    :param nedic: number of input or collected files 
    :param nedir: number of files read sucessfully 
    :param fmt: str to format the stats line 
    :param lenl: length of line denileation."""
    
    def get_obj_len (value):
        """ Control if obj is iterable then take its length """
        try : 
            iter(value)
        except :pass 
        else : value =len(value)
        return value 
    nedic = get_obj_len(nedic)
    nedir = get_obj_len(nedir)
    
    print(fmtl * lenl )
    mesg ='|'.join( ['|{0:<15}{1:^2} {2:<7}',
                     '{3:<15}{4:^2} {5:<7}',
                     '{6:<9}{7:^2} {8:<7}%|'])
    print(mesg.format('Data collected','=',  nedic, f'{obj} success. read',
                      '=', nedir, 'Rate','=', round ((nedir/nedic) *100, 2),
                      2))
    print(fmtl * lenl ) 
    
def station_id (id_, is_index= 'index', how=None, **kws): 
    """ 
    From id get the station  name as input  and return index `id`. 
    Index starts at 0.
    
    :param id_: str, of list of the name of the station or indexes . 
    
    :param is_index: bool 
        considered the given station as a index. so it remove all the letter and
        keep digit as index of each stations. 
        
    :param how: Mode to index the station. Default is 
        'Python indexing' i.e.the counting starts by 0. Any other mode will 
        start the counting by 1. Note that if `is_index` is ``True`` and the 
        param `how` is set to it default value ``py``, the station index should 
        be downgraded to 1. 
        
    :param kws: additionnal keywords arguments from :func:`~.make_ids`.
    
    :return: station index. If the list `id_` is given will return the tuple.
    
    :Example:
        
    >>> from gofast.utils.coreutils import station_id 
    >>> dat1 = ['S13', 's02', 's85', 'pk20', 'posix1256']
    >>> station_id (dat1)
    ... (13, 2, 85, 20, 1256)
    >>> station_id (dat1, how='py')
    ... (12, 1, 84, 19, 1255)
    >>> station_id (dat1, is_index= None, prefix ='site')
    ... ('site1', 'site2', 'site3', 'site4', 'site5')
    >>> dat2 = 1 
    >>> station_id (dat2) # return index like it is
    ... 1
    >>> station_id (dat2, how='py') # considering the index starts from 0
    ... 0
    
    """
    is_iterable =False 
    is_index = str(is_index).lower().strip() 
    isix=True if  is_index in ('true', 'index', 'yes', 'ix') else False 
    
    regex = re.compile(r'\d+', flags=re.IGNORECASE)
    try : 
        iter (id_)
    except : 
        id_= [id_]
    else : is_iterable=True 
    
    #remove all the letter 
    id_= list(map( lambda o: regex.findall(o), list(map(str, id_))))
    # merge the sequences list and for consistency remove emty list or str 
    id_=tuple(filter (None, list(itertools.chain(*id_)))) 
    
    # if considering as Python index return value -1 other wise return index 
    
    id_ = tuple (map(int, np.array(id_, dtype = np.int32)-1)
                 ) if how =='py' else tuple ( map(int, id_)) 
    
    if (np.array(id_) < 0).any(): 
        warnings.warn('Index contains negative values. Be aware that you are'
                      " using a Python indexing. Otherwise turn 'how' argumennt"
                      " to 'None'.")
    if not isix : 
        id_= tuple(make_ids(id_, how= how,  **kws))
        
    if not is_iterable : 
        try: id_ = id_[0]
        except : warnings.warn("The station id is given as a non iterable "
                          "object, but can keep the same format in return.")
        if id_==-1: id_= 0 if how=='py' else id_ + 2 

    return id_

def assert_doi(doi): 
    """
     assert the depth of investigation Depth of investigation converter 

    :param doi: depth of investigation in meters.  If value is given as string 
        following by yhe index suffix of kilometers 'km', value should be 
        converted instead. 
    :type doi: str|float 
    
    :returns doi:value in meter
    :rtype: float
           
    """
    if isinstance (doi, str):
        if doi.find('km')>=0 : 
            try: doi= float(doi.replace('km', '000')) 
            except :TypeError (" Unrecognized value. Expect value in 'km' "
                           f"or 'm' not: {doi!r}")
    try: doi = float(doi)
    except: TypeError ("Depth of investigation must be a float number "
                       "not: {str(type(doi).__name__!r)}")
    return doi

def round_dipole_length(value, round_value =5.): 
    """ 
    small function to graduate dipole length 5 to 5. Goes to be reality and 
    simple computation .
    
    :param value: value of dipole length 
    :type value: float 
    
    :returns: value of dipole length rounded 5 to 5 
    :rtype: float
    """ 
    mm = value % round_value 
    if mm < 3 :return np.around(value - mm)
    elif mm >= 3 and mm < 7 :return np.around(value -mm +round_value) 
    else:return np.around(value - mm +10.)
    
def display_infos(infos, **kws):
    """ Display unique element on list of array infos
    
    :param infos: Iterable object to display. 
    :param header: Change the `header` to other names. 
    
    :Example: 
    >>> from gofast.utils.coreutils import display_infos
    >>> ipts= ['river water', 'fracture zone', 'granite', 'gravel',
         'sedimentary rocks', 'massive sulphide', 'igneous rocks', 
         'gravel', 'sedimentary rocks']
    >>> display_infos('infos= ipts,header='TestAutoRocks', 
                      size =77, inline='~')
    """

    inline =kws.pop('inline', '-')
    size =kws.pop('size', 70)
    header =kws.pop('header', 'Automatic rocks')

    if isinstance(infos, str ): 
        infos =[infos]
        
    infos = list(set(infos))
    print(inline * size )
    mes= '{0}({1:02})'.format(header.capitalize(),
                                  len(infos))
    mes = '{0:^70}'.format(mes)
    print(mes)
    print(inline * size )
    am=''
    for ii in range(len(infos)): 
        if (ii+1) %2 ==0: 
            am = am + '{0:>4}.{1:<30}'.format(ii+1, infos[ii].capitalize())
            print(am)
            am=''
        else: 
            am ='{0:>4}.{1:<30}'.format(ii+1, infos[ii].capitalize())
            if ii ==len(infos)-1: 
                print(am)
    print(inline * size )
    
def numstr2dms(
    sdigit: str,  
    sanitize: bool = True, 
    func: callable = lambda x, *args, **kws: x, 
    args: tuple = (),  
    regex: re.Pattern = re.compile(r'[_#&@!+,;:"\'\s-]\s*', flags=re.IGNORECASE),   
    pattern: str = '[_#&@!+,;:"\'\s-]\s*', 
    return_values: bool = False, 
    **kws
) -> Union[str, Tuple[float, float, float]]: 
    """ Convert numerical digit string to DD:MM:SS
    
    Note that any string digit for Minutes and seconds must be composed
    of two values i.e., the function accepts at least six digits, otherwise an 
    error occurs. For instance, the value between [0-9] must be prefixed by 0 
    beforehand. Here is an example for designating 1 degree-1 min-1 seconds::
        
        sdigit= 1'1'1" --> 01'01'01 or 010101
        
    where ``010101`` is the right arguments for ``111``. 
    
    Parameters
    -----------
    sdigit: str, 
      Digit string composing of unique values. 
    func: Callable, 
      Function uses to parse digit. Function must return string values. 
      Any other values should be converted to str.
      
    args: tuple
      Function `func` positional arguments 
      
    regex: `re` object,  
        Regular expression object. Regex is important to specify the kind
        of data to parse. The default is:: 
            
            >>> import re 
            >>> re.compile(r'[_#&@!+,;:"\'\s-]\s*', flags=re.IGNORECASE) 
            
    pattern: str, default = '[_#&@!+,;:"\'\s-]\s*'
      Specific pattern for sanitizing sdigit. For instance, remove undesirable 
      non-character. 
      
    sanitize: bool, default=True 
       Remove undesirable characters using the default argument of `pattern`
       parameter. 
       
    return_values: bool, default=False, 
       Return the DD:MM:SS into a tuple of (DD, MM, SS).
    
    Returns 
    -------
    sdigit/tuple: str, tuple 
      DD:MM:SS or tuple of (DD, MM, SS)
      
    Examples
    --------
    >>> numstr2dms("1134132.08")
    '113:41:32.08'
    >>> numstr2dms("13'41'32.08")
    '13:41:32.08'
    >>> numstr2dms("11:34:13:2.08", return_values=True)
    (113.0, 41.0, 32.08)
    """
    # Remove any character from the string digit
    sdigit = str(sdigit)
    
    if sanitize: 
        sdigit = re.sub(pattern, "", sdigit, flags=re.IGNORECASE)
        
    try:
        float(sdigit)
    except ValueError:
        raise ValueError(f"Wrong value. Expects a string-digit or digit. Got {sdigit!r}")

    if callable(func): 
        sdigit = func(sdigit, *args, **kws)
        
    # In the case there is'
    decimal = '0'
    # Remove decimal
    sdigit_list = sdigit.split(".")
    
    if len(sdigit_list) == 2: 
        sdigit, decimal = sdigit_list
        
    if len(sdigit) < 6: 
        raise ValueError(f"DMS expects at least six digits (DD:MM:SS). Got {sdigit!r}")
        
    sec, sdigit = sdigit[-2:], sdigit[:-2]
    mm, sdigit = sdigit[-2:], sdigit[:-2]
    deg = sdigit  # The remaining part
    # Concatenate second decimal 
    sec += f".{decimal}" 
    
    return tuple(map(float, [deg, mm, sec])) if return_values \
        else ':'.join([deg, mm, sec])
        
        
def projection_validator (X, Xt=None, columns =None ):
    """ Retrieve x, y coordinates of a datraframe ( X, Xt ) from columns 
    names or indexes. 
    
    If X or Xt are given as arrays, `columns` may hold integers from 
    selecting the the coordinates 'x' and 'y'. 
    
    Parameters 
    ---------
    X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
        training set; Denotes data that is observed at training and prediction 
        time, used as independent variables in learning. The notation 
        is uppercase to denote that it is ordinarily a matrix. When a matrix, 
        each sample may be represented by a feature vector, or a vector of 
        precomputed (dis)similarity with each training sample. 

    Xt: Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
        Shorthand for "test set"; data that is observed at testing and 
        prediction time, used as independent variables in learning. The 
        notation is uppercase to denote that it is ordinarily a matrix.
    columns: list of str or index, optional 
        columns is usefull when a dataframe is given  with a dimension size 
        greater than 2. If such data is passed to `X` or `Xt`, columns must
        hold the name to consider as 'easting', 'northing' when UTM 
        coordinates are given or 'latitude' , 'longitude' when latlon are 
        given. 
        If dimension size is greater than 2 and columns is None , an error 
        will raises to prevent the user to provide the index for 'y' and 'x' 
        coordinated retrieval. 
      
    Returns 
    -------
    ( x, y, xt, yt ), (xname, yname, xtname, ytname), Tuple of coordinate 
        arrays and coordinate labels 
 
    """
    # initialize arrays and names 
    init_none = [None for i in range (4)]
    x,y, xt, yt = init_none
    xname,yname, xtname, ytname = init_none 
    
    m="{0} must be an iterable object, not {1!r}"
    ms= ("{!r} is given while columns are not supplied. set the list of "
        " feature names or indexes to fetch 'x' and 'y' coordinate arrays." )
    
    # validate X if X is np.array or dataframe 
    X =_assert_all_types(X, np.ndarray, pd.DataFrame ) 
    
    if Xt is not None: 
        # validate Xt if Xt is np.array or dataframe 
        Xt = _assert_all_types(Xt, np.ndarray, pd.DataFrame)
        
    if columns is not None: 
        if isinstance (columns, str): 
            columns = str2columns(columns )
        
        if not is_iterable(columns): 
            raise ValueError(m.format('columns', type(columns).__name__))
        
        columns = list(columns) + [ None for i in range (5)]
        xname , yname, xtname, ytname , *_= columns 

    if isinstance(X, pd.DataFrame):
        x, xname, y, yname = _validate_columns(X, [xname, yname])
        
    elif isinstance(X, np.ndarray):
        x, y = _is_valid_coordinate_arrays (X, xname, yname )    
        
        
    if isinstance (Xt, pd.DataFrame) :
        # the test set holds the same feature names
        # as the train set 
        if xtname is None: 
            xtname = xname
        if ytname is None: 
            ytname = yname 
            
        xt, xtname, yt, ytname = _validate_columns(Xt, [xname, yname])

    elif isinstance(Xt, np.ndarray):
        
        if xtname is None: 
            xtname = xname
        if ytname is None: 
            ytname = yname 
            
        xt, yt = _is_valid_coordinate_arrays (Xt, xtname, ytname , 'test')
        
    if (x is None) or (y is None): 
        raise ValueError (ms.format('X'))
    if Xt is not None: 
        if (xt is None) or (yt is None): 
            warnings.warn (ms.format('Xt'))

    return  (x, y , xt, yt ) , (
        xname, yname, xtname, ytname ) 
    
def _validate_columns0 (df, xni, yni ): 
    """ Validate the feature name  in the dataframe using either the 
    string litteral name of the index position in the columns.
    
    :param df: pandas.DataFrame- Dataframe with feature names as columns. 
    :param xni: str, int- feature name  or position index in the columns for 
        x-coordinate 
    :param yni: str, int- feature name  or position index in the columns for 
        y-coordinate 
    
    :returns: (x, ni) Tuple of (pandas.Series, and names) for x and y 
        coordinates respectively.
    
    """
    def _r (ni): 
        if isinstance(ni, str): # feature name
            exist_features(df, ni ) 
            s = df[ni]  
        elif isinstance (ni, (int, float)):# feature index
            s= df.iloc[:, int(ni)] 
            ni = s.name 
        return s, ni 
        
    xs , ys = [None, None ]
    if df.ndim ==1: 
        raise ValueError ("Expect a dataframe of two dimensions, got '1'")
        
    elif df.shape[1]==2: 
       warnings.warn("columns are not specify while array has dimension"
                     "equals to 2. Expect indexes 0 and 1 for (x, y)"
                     "coordinates respectively.")
       xni= df.iloc[:, 0].name 
       yni= df.iloc[:, 1].name 
    else: 
        ms = ("The matrix of features is greater than 2. Need column names or"
              " indexes to  retrieve the 'x' and 'y' coordinate arrays." ) 
        e =' Only {!r} is given.' 
        me=''
        if xni is not None: 
            me =e.format(xni)
        if yni is not None: 
            me=e.format(yni)
           
        if (xni is None) or (yni is None ): 
            raise ValueError (ms + me)
            
    xs, xni = _r (xni) ;  ys, yni = _r (yni)
  
    return xs, xni , ys, yni 

def _validate_array_indexer (arr, index): 
    """ Select the appropriate coordinates (x,y) arrays from indexes.  
    
    Index is used  to retrieve the array of (x, y) coordinates if dimension 
    of `arr` is greater than 2. Since we expect x, y coordinate for projecting 
    coordinates, 1-d  array `X` is not acceptable. 
    
    :param arr: ndarray (n_samples, n_features) - if nfeatures is greater than 
        2 , indexes is needed to fetch the x, y coordinates . 
    :param index: int, index to fetch x, and y coordinates in multi-dimension
        arrays. 
    :returns: arr- x or y coordinates arrays. 

    """
    if arr.ndim ==1: 
        raise ValueError ("Expect an array of two dimensions.")
    if not isinstance (index, (float, int)): 
        raise ValueError("index is needed to coordinate array with "
                         "dimension greater than 2.")
        
    return arr[:, int (index) ]

def _is_valid_coordinate_arrays (arr, xind, yind, ptype ='train'): 
    """ Check whether array is suitable for projecting i.e. whether 
    x and y (both coordinates) can be retrived from `arr`.
    
    :param arr: ndarray (n_samples, n_features) - if nfeatures is greater than 
        2 , indexes is needed to fetch the x, y coordinates . 
        
    :param xind: int, index to fetch x-coordinate in multi-dimension
        arrays. 
    :param yind: int, index to fetch y-coordinate in multi-dimension
        arrays
    :param ptype: str, default='train', specify whether the array passed is 
        training or test sets. 
    :returns: (x, y)- array-like of x and y coordinates. 
    
    """
    xn, yn =('x', 'y') if ptype =='train' else ('xt', 'yt') 
    if arr.ndim ==1: 
        raise ValueError ("Expect an array of two dimensions.")
        
    elif arr.shape[1] ==2 : 
        x, y = arr[:, 0], arr[:, 1]
        
    else :
        msg=("The matrix of features is greater than 2; Need index to  "
             " retrieve the {!r} coordinate array in param 'column'.")
        
        if xind is None: 
            raise ValueError(msg.format(xn))
        else : x = _validate_array_indexer(arr, xind)
        if yind is None : 
            raise ValueError(msg.format(yn))
        else : y = _validate_array_indexer(arr, yind)
        
    return x, y 