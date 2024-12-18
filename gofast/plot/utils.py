# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Miscellanous plot utilities. 
"""
from __future__ import annotations 
import os
import re 
import math
import copy 
import datetime 
import warnings
import itertools 
from numbers import Real, Integral 
import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.axes
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms 
from scipy.interpolate import griddata

from sklearn.metrics import r2_score 
from sklearn.utils import resample

from ..api.types import Optional, Tuple,  Union, List 
from ..api.types import Dict, ArrayLike, DataFrame
from ..api.property import BasePlot
from ..core.array_manager import smart_ts_detector 
from ..core.checks import ( 
    _assert_all_types, is_iterable, str2columns, is_in_if, 
    exist_features, check_features_types, check_spatial_columns, 
)
from ..core.handlers import ( 
    columns_manager, default_params_plot, param_deprecated_message 
)
from ..compat.sklearn import validate_params, StrOptions, Interval 
from ..decorators import isdf
from ..utils.validator import  ( 
    assert_xy_in, build_data_if, validate_positive_integer, 
    validate_quantiles, is_frame, check_consistent_length 
)
from ._d_cms import D_COLORS, D_MARKERS, D_STYLES

__all__=[
    "boxplot", 
    "plot_r_squared", 
    "plot_text", 
    "plot_spatial_features", 
    "plot_categorical_feature", 
    'plot_sensitivity', 
    'plot_spatial_distribution', 
    'plot_dist', 
    'plot_quantile_distributions', 
    'plot_uncertainty', 
    'plot_prediction_intervals',
    'plot_temporal_trends'
]

class PlotUtils(BasePlot):
    def __init__(self, **kwargs):
        """
        Initialize the plotting utility class which extends the BasePlot class,
        allowing for custom plot configurations.

        Parameters
        ----------
        **kwargs : dict
            Arbitrary keyword arguments that are passed to the
            BasePlot's constructor.
        """
        super().__init__(**kwargs)
    
    def save(self, fig):
        """
        Save the figure with the specified attributes if `savefig`
        is set; otherwise, display the figure on screen.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure object to be saved or displayed.

        Notes
        -----
        - If `savefig` is not None, the figure is saved using the
          path specified in `savefig`.
        - The figure is saved with the resolution specified by `fig_dpi`,
          and the orientation can be set with `fig_orientation`.
        - If `savefig` is None, the figure will be displayed using
          `plt.show()` and not saved.

        Examples
        --------
        >>> from matplotlib import pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> ax.plot([0, 1], [0, 1])
        >>> plot_utils = PlotUtils(figsize=(10, 5), savefig='plot.png',
        ...                        fig_dpi=300, fig_orientation='landscape')
        >>> plot_utils.save(fig)  # This will save the figure to 'plot.png'
        """
        if self.savefig is not None:
            fig.savefig(self.savefig, dpi=self.fig_dpi,
                        bbox_inches='tight', 
                        orientation=self.fig_orientation)
            plt.close()
        else:
            plt.show()
    
def create_custom_plotter(base_plotter):
    """
    Dynamically creates a custom plotter class that includes additional 
    methods and attributes from an existing plotter object.

    Parameters:
        base_plotter: An instance of a plotter class from which to inherit 
        properties.

    Returns:
        A new plotting class with combined features from `PlotUtils and `base_plotter`.
    """
    # Creating a dynamic type combining BasePlot and properties from the given plotter
    plot_class = type('CustomPlotter', (PlotUtils,), {**base_plotter.__dict__})
    
    # Update the docstring for the new class
    plot_class.__doc__ = """\
    Custom plotting class that extends :class:`~gofast.api.properties.BasePlot` 
    with dynamic properties.

    Inherits all matplotlib figure properties, allowing modification via 
    object attributes. For example:
        >>> plot_obj = CustomPlotter()
        >>> plot_obj.ls = '-.'  # Set line style
        >>> plot_obj.fig_size = (7, 5)  # Set figure size
        >>> plot_obj.lw = 7  # Set linewidth

    See also:
        Refer to :class:`~gofast.api.property.BasePlot` for details 
        on adjustable parameters.
    """
    
    return plot_class

# #################################################
# Creating a dynamic type combining BasePlot and 
# properties from the given plotter
pobj = type('DynamicPlotter', (PlotUtils,), {})

pobj.__doc__ = """\
Dynamic plotting class that extends :class:`~gofast.api.properties.BasePlot` 
with dynamic properties.

Inherits all matplotlib figure properties, allowing modification via 
object attributes. For example:
    >>> plot_obj = DynamicPlotter()
    >>> plot_obj.ls = '-.'  # Set line style
    >>> plot_obj.fig_size = (7, 5)  # Set figure size
    >>> plot_obj.lw = 7  # Set linewidth

See also:
    Refer to :class:`~gofast.api.property.BasePlot` for details 
    on adjustable parameters.
"""
# ##################################################

@validate_params ({ 
    "sensitivity_values": ['array-like'], 
    "baseline_prediction": ['array-like', Real, None ], 
    "plot_type": [StrOptions({'hist', 'bar', 'line', 'boxplot', 'box'})], 
    "x_ticks_rotation": [Interval( Integral, 0, None, closed="left")], 
    "y_ticks_rotation": [Interval( Integral, 0, None, closed="left")], 
    })
def plot_sensitivity(
    sensitivity_values, *,
    baseline=None, 
    plot_type='line',
    baseline_color='r',
    baseline_linewidth=2,
    baseline_linestyle='--',
    title=None,
    xlabel=None,
    ylabel=None,
    x_ticks_rotation=0,
    y_ticks_rotation=0,
    grid=True,
    legend=True,
    figsize=(10, 6),
    color_palette='muted',
    boxplot_showfliers=False
):
    """
    Plot the feature sensitivity values.

    Parameters
    ----------
    sensitivity_values : pandas.DataFrame
        A DataFrame containing sensitivity values for each feature. Each column 
        represents the sensitivity values for a specific feature. The index 
        represents individual observations or instances.
        
    baseline: array-like or scalar, optional 
        The baseline prediction, either a scalar value or an array-like object 
        (e.g., list, numpy array) representing the baseline prediction to be 
        compared with feature sensitivities.

    plot_type : {'line', 'bar', 'hist', 'boxplot'}, optional, default='line'
        The type of plot to generate. Options include:
        - 'line': Line plot to visualize feature sensitivity trends.
        - 'bar': Bar plot for visualizing feature sensitivity comparisons.
        - 'hist': Histogram to show the distribution of sensitivities for 
          each feature.
        - 'boxplot': Boxplot to summarize the distribution and outliers 
          of sensitivities.

    baseline_color : str, optional, default='r'
        The color for the baseline prediction line. Can be any valid Matplotlib
         color specification (e.g., named color, hex, RGB tuple).

    baseline_linewidth : float, optional, default=2
        The line width for the baseline prediction line.

    baseline_linestyle : {'-', '--', '-.', ':'}, optional, default='--'
        The line style for the baseline prediction line. 
        Options include solid, dashed, dash-dot, and dotted lines.

    title : str, optional, default=None
        The title for the plot. If not provided, a default title is generated 
        based on the plot type.

    xlabel : str, optional, default='Features'
        The label for the x-axis, which typically corresponds to the feature 
        names or identifiers in `sensitivity_values`.

    ylabel : str, optional, default='Sensitivity Value'
        The label for the y-axis, representing the sensitivity value or 
        measure associated with each feature.

    x_ticks_rotation : int, optional, default=0
        The angle in degrees to rotate the x-axis tick labels. Helps in cases 
        where feature names or labels overlap.

    y_ticks_rotation : int, optional, default=0
        The angle in degrees to rotate the y-axis tick labels.

    grid : bool, optional, default=True
        Whether to show gridlines on the plot. True will enable gridlines, 
        False will disable them.

    legend : bool, optional, default=True
        Whether to display the legend in the plot. Set to True to show the 
        legend, False to hide it.

    figsize : tuple of two floats, optional, default=(10, 6)
        The dimensions of the plot as a tuple representing (width, height) 
        in inches.

    color_palette : str, optional, default='muted'
        The seaborn color palette to use for the plot. A string specifying 
        a predefined color palette (e.g., 'deep', 'muted', 'bright').

    boxplot_showfliers : bool, optional, default=False
        Whether to display outliers in the boxplot when `plot_type='boxplot'`. 
        Set to False to hide outliers, True to show them.

    Returns
    -------
    None
        The function generates and displays a plot showing the baseline 
        prediction and feature sensitivity with the specified 
        customization options.


    Notes
    -----
    The function will automatically determine whether to construct the 
    `sensitivity_values` DataFrame if not provided directly as a DataFrame.
    It uses the `build_data_if` helper function to convert the data into a 
    DataFrame before proceeding with plotting.

    Examples
    --------
    >>> from gofast.plot.utils import plot_sensitivity
    1. Basic line plot:
       >>> plot_sensitivity(baseline=0.5, 
                            sensitivity_values=pd.DataFrame({
                                'Feature 1': [0.1, 0.2, 0.3],
                                'Feature 2': [0.05, 0.15, 0.25],
                                'Feature 3': [0.2, 0.3, 0.4]
                            }), 
                            plot_type='line')
       
    2. Bar plot with customized appearance:
       >>> plot_sensitivity(baseline=0.5, 
                            sensitivity_values=pd.DataFrame({
                                'Feature 1': [0.1, 0.2, 0.3],
                                'Feature 2': [0.05, 0.15, 0.25],
                                'Feature 3': [0.2, 0.3, 0.4]
                            }), 
                            plot_type='bar', baseline_color='g', 
                            baseline_linestyle='-', figsize=(8, 5))
       
    3. Histogram plot:
       >>> plot_sensitivity(baseline=0.5, 
                            sensitivity_values=pd.DataFrame({
                                'Feature 1': [0.1, 0.2, 0.3],
                                'Feature 2': [0.05, 0.15, 0.25],
                                'Feature 3': [0.2, 0.3, 0.4]
                            }), 
                            plot_type='hist')
    
    4. Boxplot with outliers:
       >>> plot_sensitivity(baseline=0.5, 
                            sensitivity_values=pd.DataFrame({
                                'Feature 1': [0.1, 0.2, 0.3],
                                'Feature 2': [0.05, 0.15, 0.25],
                                'Feature 3': [0.2, 0.3, 0.4]
                            }), 
                            plot_type='boxplot', boxplot_showfliers=True)
    """

    if not isinstance (sensitivity_values, pd.DataFrame): 
        # build dataframe using the default column name 'feature' 
        sensitivity_values = build_data_if(
            sensitivity_values, 
            force=True, 
            input_name="feature", 
            raise_exception=True 
    ) 

    if len(sensitivity_values) == 1:
        # Transpose if single perturbation
        sensitivity_values = sensitivity_values.T  
    
    sns.set(style="whitegrid", palette=color_palette)
    
    # Default plot title
    if title is None:
        title = 'Feature Sensitivity vs Baseline Prediction'

    plt.figure(figsize=figsize)

    if plot_type == 'line':
        for col in sensitivity_values.columns:
            plt.plot(
                sensitivity_values.index, 
                sensitivity_values[col], 
                label=col, 
                marker='o'
            )
        if baseline is not None:
            if isinstance(baseline, (list, np.ndarray)):
                baseline = baseline[0]  # Use the first element if it's an array-like
            plt.axhline(
                y=baseline, 
                color=baseline_color, 
                linestyle=baseline_linestyle, 
                linewidth=baseline_linewidth, 
                label='Baseline Prediction'
            )
        plt.title(title) 
        plt.xlabel(xlabel or "Pertubations")
        plt.ylabel(ylabel or 'Sensitivity Value')
        plt.xticks(rotation=x_ticks_rotation)
        plt.yticks(rotation=y_ticks_rotation)
        if grid:
            plt.grid(True)
        if legend:
            plt.legend()

    elif plot_type == 'bar':
        sensitivity_values_mean = sensitivity_values.mean(axis=0)
        plt.bar(
            sensitivity_values_mean.index, 
            sensitivity_values_mean.values, 
            label='Feature Sensitivity'
        )
        if baseline is not None:
            if isinstance(baseline, (list, np.ndarray)):
                baseline = baseline[0]  # Use the first element if it's an array-like
            plt.axhline(
                y=baseline, 
                color=baseline_color, 
                linestyle=baseline_linestyle, 
                linewidth=baseline_linewidth, 
                label='Baseline Prediction'
            )
        plt.title(title)
        plt.xlabel(xlabel or "Pertubations")
        plt.ylabel(ylabel or 'Sensitivity Value')
        plt.xticks(rotation=x_ticks_rotation)
        plt.yticks(rotation=y_ticks_rotation)
        if grid:
            plt.grid(True)
        if legend:
            plt.legend()

    elif plot_type == 'hist':
        for col in sensitivity_values.columns:
            sns.histplot(
                sensitivity_values[col], 
                kde=True, 
                label=col, 
                element='step', 
                fill=False
            )
        if baseline is not None:
            if isinstance(baseline, (list, np.ndarray)):
                baseline = baseline[0]  # Use the first element if it's an array-like
            plt.axvline(
                x=baseline, 
                color=baseline_color, 
                linestyle=baseline_linestyle, 
                linewidth=baseline_linewidth, 
                label='Baseline Prediction'
            )
        plt.title(title)
        plt.xlabel(xlabel or 'Sensitivity Value')
        plt.ylabel('Frequency')
        if grid:
            plt.grid(True)
        if legend:
            plt.legend()

    elif plot_type in ['boxplot', 'box']:
        sns.boxplot(
            data=sensitivity_values, 
            showfliers=boxplot_showfliers
        )
        if baseline is not None:
            if isinstance(baseline, (list, np.ndarray)):
                baseline = baseline[0]  # Use the first element if it's an array-like
            plt.axhline(
                y=baseline, 
                color=baseline_color, 
                linestyle=baseline_linestyle, 
                linewidth=baseline_linewidth, 
                label='Baseline Prediction'
            )
        plt.title(title)
        plt.xlabel(xlabel or "Pertubations")
        plt.ylabel(ylabel or 'Sensitivity Value')
        if grid:
            plt.grid(True)
        if legend:
            plt.legend()

    else:
        raise ValueError(
            "Unsupported plot type. Choose from 'line', 'bar', 'hist', or 'boxplot'."
        )
    
    plt.xticks(rotation=x_ticks_rotation)
    plt.yticks(rotation=y_ticks_rotation)
    plt.tight_layout()
    plt.show()

@isdf 
def plot_spatial_features(
    data,
    features,
    dates=None,
    date_col="year",
    x_col='longitude',
    y_col='latitude',
    colormaps=None,
    figsize=None,
    point_size=10,
    marker='o',
    plot_type='scatter',
    colorbar_orientation='vertical',
    cbar_labelsize=10,
    axis_off=True,
    titles=None,
    vmin_vmax=None,
    **kwargs
):
    """
    Plot spatial distribution of specified features over given dates.

    This function creates a grid of subplots, each displaying the
    geographical distribution of a specified feature at particular
    dates or times. It supports multiple plot types including
    ``'scatter'``, ``'hexbin'``, and ``'contour'``, allowing for
    extensive customization of plot appearance. The function leverages
    Matplotlib's plotting capabilities [1]_ to visualize spatial data.

    Parameters
    ----------
    data : pandas.DataFrame
        The input DataFrame containing the data to plot. It must
        include the specified `features`, `x_col`, `y_col`, and
        the `date_col` if `dates` are provided.

    features : list of str
        List of feature names to plot. Each feature must exist
        in `data`. A subplot will be created for each feature.

    dates : str or datetime-like or list of str or datetime-like, optional
        Dates or times to plot. Each date must correspond to an entry
        in the `date_col` of `data`. If `None`, the function plots
        the features without considering dates.

    date_col : str, default ``'year'``
        Name of the column in `data` to use for date or time filtering.
        This column can contain datetime objects, years, or any other
        temporal representation.

    x_col : str, default ``'longitude'``
        Name of the column in `data` to use for the x-axis coordinates.

    y_col : str, default ``'latitude'``
        Name of the column in `data` to use for the y-axis coordinates.

    colormaps : list of str, optional
        List of colormap names to use for each feature. If not
        provided, default colormaps are used.

    figsize : tuple of float, optional
        Figure size in inches, as a tuple ``(width, height)``. If not
        provided, the figure size is determined based on the number of
        features and dates.

    point_size : int, default 10
        Size of the points in the scatter plot.

    marker : str, default ``'o'``
        Marker style for scatter plots.

    plot_type : {'scatter', 'hexbin', 'contour'}, default ``'scatter'``
        Type of plot to create. Supported options are ``'scatter'``,
        ``'hexbin'``, and ``'contour'``.

    colorbar_orientation : {'vertical', 'horizontal'}, default ``'vertical'``
        Orientation of the colorbar.

    cbar_labelsize : int, default 10
        Font size for the colorbar tick labels.

    axis_off : bool, default True
        If ``True``, axes are turned off. If ``False``, axes are shown.

    titles : dict of str, optional
        Dictionary of titles for each feature. Keys are feature names,
        and values are title templates that can include ``{date}``.

    vmin_vmax : dict of tuple, optional
        Dictionary specifying the color scale (vmin and vmax) for each
        feature. Keys are feature names, and values are tuples
        ``(vmin, vmax)``.

    **kwargs
        Additional keyword arguments passed to the plotting functions
        (``scatter``, ``hexbin``, or ``contourf``).

    Notes
    -----
    The function supports different plot types:

    - For ``plot_type='scatter'``, it creates a scatter plot using
      ``matplotlib.pyplot.scatter``.

    - For ``plot_type='hexbin'``, it creates a hexbin plot using
      ``matplotlib.pyplot.hexbin``.

    - For ``plot_type='contour'``, it creates a contour plot by
      interpolating the data onto a grid using
      :func:`scipy.interpolate.griddata` and then plotting using
      ``matplotlib.pyplot.contourf``.

    The color normalization is performed using:

    .. math::

        c_{\text{norm}} = \frac{c - v_{\text{min}}}{v_{\text{max}} - v_{\text{min}}}

    where :math:`c` is the feature value, :math:`v_{\text{min}}` and
    :math:`v_{\text{max}}` are the minimum and maximum values for the
    color scale.

    Examples
    --------
    >>> from gofast.plot.utils import plot_spatial_features
    >>> plot_spatial_features(
    ...     data=df,
    ...     features=['temperature', 'humidity'],
    ...     dates=['2023-01-01', '2023-06-01'],
    ...     date_col='date',
    ...     x_col='lon',
    ...     y_col='lat',
    ...     colormaps=['coolwarm', 'YlGnBu'],
    ...     point_size=15,
    ...     plot_type='scatter',
    ...     axis_off=False,
    ...     titles={'temperature': 'Temp on {date}',
    ...             'humidity': 'Humidity on {date}'},
    ...     alpha=0.7
    ... )

    See Also
    --------
    matplotlib.pyplot.scatter : Create a scatter plot.
    matplotlib.pyplot.hexbin : Make a hexagonal binning plot.
    matplotlib.pyplot.contourf : Create a filled contour plot.
    scipy.interpolate.griddata : Interpolate unstructured D-dimensional data.

    References
    ----------
    .. [1] Hunter, J. D. (2007). Matplotlib: A 2D graphics environment.
       *Computing in Science & Engineering*, 9(3), 90-95.
    """

    # Validate that features exist in data
    exist_features(data, features)
    extra_msg = (
        "If a numeric feature is stored as an 'object' type, "
        "it should be explicitly converted to a numeric type"
        " (e.g., using `pd.to_numeric`). For categorical features,"
        " please use `plot_categorical_feature` instead."
    )
    check_features_types(
        data, features= features, dtype='numeric',
        extra=extra_msg
    )
    # Handle dates parameter
    if dates is not None:
        # Convert single value to list
        if not isinstance(dates, (list, tuple, np.ndarray, pd.Series)):
            dates = [dates]
        else:
            dates = list(dates)

        # Check that 'date_col' exists
        if date_col not in data.columns:
            raise ValueError(f"Column '{date_col}' not found in data.")

        # Depending on the type of 'date_col', process accordingly
        if np.issubdtype(data[date_col].dtype, np.datetime64):
            # If date_col is datetime, convert dates to datetime
            data[date_col] = pd.to_datetime(data[date_col])

            # Convert dates parameter to datetime
            dates = [pd.to_datetime(d) for d in dates]

            # Normalize dates to remove time component
            data_dates = data[date_col].dt.normalize().unique()
            dates_normalized = [d.normalize() for d in dates]

            # Check that dates exist in data
            missing_dates = set(dates_normalized) - set(data_dates)
            if missing_dates:
                missing_dates_str = ', '.join(
                    [d.strftime('%Y-%m-%d') for d in missing_dates]
                )
                raise ValueError(f"Dates {missing_dates_str} not found in data.")

            ncols = len(dates)
        else:
            # date_col is not datetime, treat as categorical or numeric
            data_dates = data[date_col].unique()
            missing_dates = set(dates) - set(data_dates)
            if missing_dates:
                missing_dates_str = ', '.join(map(str, missing_dates))
                raise ValueError(f"Dates {missing_dates_str} not found in data.")

            ncols = len(dates)
    else:
        ncols = 1

    nrows = len(features)
 
    colormaps = columns_manager(colormaps) 
    
    if colormaps is None:
        colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']

    if figsize is None:
        figsize = (5 * ncols, 5 * nrows)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        squeeze=False
    )

    for i, feature in enumerate(features):
        cmap = colormaps[i % len(colormaps)]

        if vmin_vmax and feature in vmin_vmax:
            vmin, vmax = vmin_vmax[feature]
        else:
            vmin = data[feature].min()
            vmax = data[feature].max()

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        if dates is not None:
            for j, date in enumerate(dates):
                ax = axes[i, j]
                if np.issubdtype(data[date_col].dtype, np.datetime64):
                    # Normalize date to remove time component
                    date_normalized = pd.to_datetime(date).normalize()
                    subset = data[
                        (data[date_col].dt.normalize() == date_normalized)
                        & data[feature].notnull()
                    ]
                    date_str = date_normalized.strftime('%Y-%m-%d')
                else:
                    subset = data[
                        (data[date_col] == date)
                        & data[feature].notnull()
                    ]
                    date_str = str(date)

                x = subset[x_col].values
                y = subset[y_col].values
                c = subset[feature].values

                if plot_type == 'scatter':
                    sc = ax.scatter(
                        x,
                        y,
                        c=c,
                        cmap=cmap,
                        norm=norm,
                        s=point_size,
                        marker=marker,
                        **kwargs
                    )
                elif plot_type == 'hexbin':
                    sc = ax.hexbin(
                        x,
                        y,
                        C=c,
                        gridsize=50,
                        cmap=cmap,
                        norm=norm,
                        **kwargs
                    )
                elif plot_type == 'contour':
                    # Create a grid to interpolate data
                    xi = np.linspace(x.min(), x.max(), 100)
                    yi = np.linspace(y.min(), y.max(), 100)
                    xi, yi = np.meshgrid(xi, yi)
                    # Interpolate using griddata
                    zi = griddata((x, y), c, (xi, yi), method='linear')
                    # Plot contour
                    sc = ax.contourf(
                        xi,
                        yi,
                        zi,
                        levels=15,
                        cmap=cmap,
                        norm=norm,
                        **kwargs
                    )
                else:
                    raise ValueError(f"Unsupported plot_type: {plot_type}")

                if titles and feature in titles:
                    title = titles[feature].format(date=date_str)
                else:
                    title = f"{feature} - {date_str}"

                ax.set_title(title)
                if axis_off:
                    ax.axis('off')

                if j == ncols - 1:
                    cbar = fig.colorbar(
                        sc,
                        ax=ax,
                        orientation=colorbar_orientation
                    )
                    cbar.ax.tick_params(labelsize=cbar_labelsize)
        else:
            ax = axes[i, 0]
            subset = data[data[feature].notnull()]
            x = subset[x_col].values
            y = subset[y_col].values
            c = subset[feature].values

            if plot_type == 'scatter':
                sc = ax.scatter(
                    x,
                    y,
                    c=c,
                    cmap=cmap,
                    norm=norm,
                    s=point_size,
                    marker=marker,
                    **kwargs
                )
            elif plot_type == 'hexbin':
                sc = ax.hexbin(
                    x,
                    y,
                    C=c,
                    gridsize=50,
                    cmap=cmap,
                    norm=norm,
                    **kwargs
                )
            elif plot_type == 'contour':
                # Create a grid to interpolate data
                xi = np.linspace(x.min(), x.max(), 100)
                yi = np.linspace(y.min(), y.max(), 100)
                xi, yi = np.meshgrid(xi, yi)
                # Interpolate using griddata
                
                zi = griddata((x, y), c, (xi, yi), method='linear')
                # Plot contour
                sc = ax.contourf(
                    xi,
                    yi,
                    zi,
                    levels=15,
                    cmap=cmap,
                    norm=norm,
                    **kwargs
                )
            else:
                raise ValueError(f"Unsupported plot_type: {plot_type}")

            if titles and feature in titles:
                title = titles[feature]
            else:
                title = f"{feature}"

            ax.set_title(title)
            if axis_off:
                ax.axis('off')

            cbar = fig.colorbar(
                sc,
                ax=ax,
                orientation=colorbar_orientation
            )
            cbar.ax.tick_params(labelsize=cbar_labelsize)

    plt.tight_layout()
    plt.show()

@isdf 
def plot_categorical_feature(
    data,
    feature,
    dates=None,
    date_col='year',
    x_col='longitude',
    y_col='latitude',
    cmap='tab10',
    figsize=None,
    point_size=10,
    marker='o',
    axis_off=True,
    legend_loc='upper left',
    titles=None,
    **kwargs
):
    """
    Plot the geographical distribution of a categorical feature.

    This function creates scatter plots showing the spatial
    distribution of a categorical feature over geographical
    coordinates. It supports plotting multiple dates or times,
    creating subplots for each specified date. The function allows
    extensive customization of the plot's appearance, including
    colormaps, point sizes, markers, and more.

    Parameters
    ----------
    data : pandas.DataFrame
        The input DataFrame containing the data to plot. It must
        include the specified `feature`, `x_col`, `y_col`, and
        `date_col` if `dates` are provided.

    feature : str
        The name of the categorical feature to plot. This feature
        must exist in `data`.

    dates : scalar, list, or array-like, optional
        Dates or times to plot. If provided, the function will
        create subplots for each date specified. If `None`, the
        feature is plotted without considering dates.

    date_col : str, default ``'year'``
        The name of the column in `data` to use for date or time
        filtering.

    x_col : str, default ``'longitude'``
        Name of the column in `data` to use for the x-axis
        coordinates.

    y_col : str, default ``'latitude'``
        Name of the column in `data` to use for the y-axis
        coordinates.

    cmap : str, default ``'tab10'``
        The name of the colormap to use for different categories.

    figsize : tuple, optional
        Figure size in inches, as a tuple ``(width, height)``. If
        not provided, the figure size is determined based on the
        number of dates and default settings.

    point_size : int, default 10
        Size of the points in the scatter plot.

    marker : str, default ``'o'``
        Marker style for scatter plots.

    axis_off : bool, default ``True``
        If ``True``, axes are turned off. If ``False``, axes are
        shown.

    legend_loc : str, default ``'upper left'``
        Location of the legend in the plot. Valid locations are
        strings such as ``'upper right'``, ``'lower left'``, etc.

    titles : dict or str, optional
        Titles for the subplots. If a dictionary, keys should
        correspond to subplot indices or dates, and values are
        title strings. If a string, it is used as a title template
        and can include placeholders like ``{date}`` which will be
        replaced with the actual date.

    **kwargs
        Additional keyword arguments passed to the plotting
        function (`matplotlib.pyplot.scatter`).

    Returns
    -------
    None
        The function displays the plot and does not return any
        value.

    Notes
    -----
    The function plots the spatial distribution of a categorical
    feature over geographical coordinates specified by `x_col` and
    `y_col`. If `dates` are provided, it filters the data for each
    date and creates a subplot for each one.

    The colors for each category are determined using the specified
    `colormap`. The categories are mapped to colors using:

    .. math::

        \\text{color}_i = \\text{colormap}\\left( \\frac{i}{N} \\right)

    where :math:`i` is the category index and :math:`N` is the total
    number of categories.

    Examples
    --------
    >>> from gofast.plot.utils import plot_categorical_feature
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Sample data
    >>> data = pd.DataFrame({
    ...     'longitude': np.random.uniform(-10, 10, 100),
    ...     'latitude': np.random.uniform(-10, 10, 100),
    ...     'category': np.random.choice(['A', 'B', 'C'], 100),
    ...     'year': np.random.choice([2018, 2019, 2020], 100)
    ... })
    >>> # Plotting without dates
    >>> plot_categorical_feature(
    ...     data,
    ...     feature='category',
    ...     x_col='longitude',
    ...     y_col='latitude',
    ...     point_size=20,
    ...     legend_loc='upper right'
    ... )
    >>> # Plotting with dates
    >>> plot_categorical_feature(
    ...     data,
    ...     feature='category',
    ...     dates=[2018, 2019, 2020],
    ...     date_col='year',
    ...     x_col='longitude',
    ...     y_col='latitude',
    ...     point_size=20,
    ...     legend_loc='upper right',
    ...     titles='Category Distribution in {date}'
    ... )

    See Also
    --------
    plot_spatial_features : Function to plot spatial distribution of
        numerical features.

    References
    ----------
    .. [1] Hunter, J. D. (2007). Matplotlib: A 2D graphics environment.
       *Computing in Science & Engineering*, 9(3), 90-95.

    """
    # Validate that the feature exists in data
    exist_features(data, features= feature)
    extra_msg = (
    "To explicitly convert a feature of type 'object' to 'category', "
    "use `data[feature].astype('category')`. For numerical features, "
    "please use `plot_spatial_features` instead."
    )

    check_features_types(
        data, features= feature, dtype='category', extra=extra_msg)
    # Get unique categories
    categories = data[feature].unique()
    num_categories = len(categories)

    # Generate colors for each category
    cmap = plt.get_cmap(cmap, num_categories)
    colors = [cmap(i) for i in range(num_categories)]
    category_color_map = dict(zip(categories, colors))

    # Handle dates parameter
    if dates is not None:
        if not isinstance(dates, (list, tuple, np.ndarray, pd.Series)):
            dates = [dates]
        else:
            dates = list(dates)

        if date_col not in data.columns:
            raise ValueError(f"Column '{date_col}' not found in data.")

        if np.issubdtype(data[date_col].dtype, np.datetime64):
            data[date_col] = pd.to_datetime(data[date_col])
            dates = [pd.to_datetime(d) for d in dates]
            data_dates = data[date_col].dt.normalize().unique()
            dates_normalized = [d.normalize() for d in dates]
            missing_dates = set(dates_normalized) - set(data_dates)
            if missing_dates:
                missing_dates_str = ', '.join(
                    [d.strftime('%Y-%m-%d') for d in missing_dates]
                )
                raise ValueError(f"Dates {missing_dates_str} not found in data.")
            ncols = len(dates)
        else:
            data_dates = data[date_col].unique()
            missing_dates = set(dates) - set(data_dates)
            if missing_dates:
                missing_dates_str = ', '.join(map(str, missing_dates))
                raise ValueError(f"Dates {missing_dates_str} not found in data.")
            ncols = len(dates)
    else:
        ncols = 1

    nrows = 1

    if figsize is None:
        figsize = (5 * ncols, 6)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i in range(ncols):
        ax = axes[i]
        if dates is not None:
            date = dates[i]
            if np.issubdtype(data[date_col].dtype, np.datetime64):
                date_normalized = pd.to_datetime(date).normalize()
                subset = data[data[date_col].dt.normalize() == date_normalized]
                date_str = date_normalized.strftime('%Y-%m-%d')
            else:
                subset = data[data[date_col] == date]
                date_str = str(date)
            title = f"{feature} - {date_str}"
        else:
            subset = data
            title = f"Geographical Distribution of '{feature}'"

        for category in categories:
            cat_subset = subset[subset[feature] == category]
            x = cat_subset[x_col].values
            y = cat_subset[y_col].values
            ax.scatter(
                x, y,
                label=category,
                c=[category_color_map[category]],
                s=point_size,
                marker=marker,
                **kwargs
            )

        if titles:
            if isinstance(titles, dict) and i in titles:
                ax.set_title(titles[i])
            elif isinstance(titles, str):
                ax.set_title(titles)
            else:
                ax.set_title(title)
        else:
            ax.set_title(title)

        if axis_off:
            ax.axis('off')

        if i == ncols - 1:
            # Add legend to the last subplot
            ax.legend(title=feature, bbox_to_anchor=(1.05, 1), loc=legend_loc)

    plt.tight_layout()
    plt.show()

def boxplot(
    data: ArrayLike | DataFrame, 
    labels: list[str],
    title: str, 
    y_label: str, 
    figsize: tuple[int, int]=(8, 8), 
    color: str="lightgreen", 
    showfliers: bool=True, 
    whis: float=1.5, 
    width: float=0.5, 
    linewidth: float=2, 
    flierprops: dict=None, 
    sns_style="whitegrid", 
   ) -> plt.Axes:
    """
    Plots a custom boxplot with the given data and parameters using 
    Seaborn and Matplotlib.

    Parameters
    ----------
    data : np.ndarray
        The input data for each category to plot in the boxplot, 
        organized as a list of arrays.
    labels : list[str]
        The labels for the boxplot categories.
    title : str
        The title of the plot.
    y_label : str
        The label for the Y-axis.
    figsize : tuple[int, int], optional
        Figure dimension (width, height) in inches. Default is (10, 8).
    color : str, optional
        Color for all of the elements, or seed for a gradient palette.
        Default is "lightgreen".
    showfliers : bool, optional
        If True, show the outliers beyond the caps. Default is True.
    whis : float, optional
        Proportion of the IQR past the low and high quartiles to 
        extend the plot whiskers. Default is 1.5.
    width : float, optional
        Width of the full boxplot elements. Default is 0.5.
    linewidth : float, optional
        Width of the lines of the boxplot elements. Default is 2.
    flierprops : dict, optional
        The style of the fliers; if None, then the default is used.
    sns_style: str, defualt='whitegrid'
        The style of seaborn boxplot. 
    Returns
    -------
    matplotlib.axes.Axes
        The Axes object with the plot for further tweaking.

    Examples
    --------
    >>> import numpy as np 
    >>> from gofast.plot.utils import boxplot
    >>> np.random.seed(10)
    >>> d = [np.random.normal(0, std, 100) for std in range(1, 5)]
    >>> labels = ['s1', 's2', 's3', 's4']
    >>> plot_custom_boxplot(d, labels, 
    ...                     title='Class assignment (roc-auc): PsA activity',
    ...                     y_label='roc-auc', 
    ...                     figsize=(12, 7),
    ...                     color="green",
    ...                     showfliers=False, 
    ...                     whis=2,
    ...                     width=0.3, 
    ...                     linewidth=1.5,
    ...                     flierprops=dict(marker='x', color='black', markersize=5))
    Notes
    -----
    Boxplots are a standardized way of displaying the distribution of data 
    based on a five-number summary: minimum, first quartile (Q1), median, 
    third quartile (Q3), and maximum. It can reveal outliers, 
    data symmetry, grouping, and skewness.
    """
    if flierprops is None:
        flierprops = dict(marker='o', color='red', alpha=0.5)
    
    # Create a figure and a set of subplots
    plt.figure(figsize=figsize)
    
    # Create the boxplot
    bplot = sns.boxplot(data=data, width=width, color=color, 
                        showfliers=showfliers, whis=whis, 
                        flierprops=flierprops,
                        linewidth=linewidth
                        )
    
    # Set labels and title
    bplot.set_title(title)
    bplot.set_ylabel(y_label)
    bplot.set_xticklabels(labels)
    
    # Set the style of the plot
    sns.set_style(sns_style)
    
    # Show the plot
    plt.show()
    return bplot

def plot_r_squared(
    y_true, y_pred, 
    model_name="Regression Model",
    figsize=(10, 6), 
    r_color='red', 
    pred_color='blue',
    sns_plot=False, 
    show_grid=True, 
    **scatter_kws
):
    """
    Plot the R-squared value for a regression model's predictions.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values by the regression model.
    model_name : str, optional
        The name of the regression model for display in the plot title. 
        Default is "Regression Model".
    figsize : tuple, optional
        The size of the figure to plot (width, height in inches). 
        Default is (10, 6).
    r_color : str, optional
        The color of the line that represents the actual values.
        Default is 'red'.
    pred_color : str, optional
        The color of the scatter plot points for the predictions.
        Default is 'blue'.
    sns_plot : bool, optional
        If True, use seaborn for plotting. Otherwise, use matplotlib.
        Default is False.
    show_grid : bool, optional
        If True, display the grid on the plot. Default is True.
    scatter_kws : dict
        Additional keyword arguments to be passed to 
        the `scatter` function.

    Returns
    -------
    ax : Axes
        The matplotlib Axes object with the R-squared plot.
         
    Example 
    -------
    >>> import numpy as np 
    >>> from gofast.plot.utils import plot_r_squared 
    >>> # Generate some sample data
    >>> np.random.seed(0)
    >>> y_true_sample = np.random.rand(100) * 100
    >>> # simulated prediction with small random noise
    >>> y_pred_sample = y_true_sample * (np.random.rand(100) * 0.1 + 0.95)  
    # Use the sample data to plot R-squared
    >>> plot_r_squared(y_true_sample, y_pred_sample, "Sample Regression Model")

    """
    # Calculate R-squared
    r_squared = r2_score(y_true, y_pred)
    
    # Create figure and axis
    plt.figure(figsize=figsize)
    ax = plt.gca()
    
    # Plot using seaborn or matplotlib
    if sns_plot:
        sns.scatterplot(x=y_true, y=y_pred, ax=ax, color=pred_color,
                        **scatter_kws)
    else:
        ax.scatter(y_true, y_pred, color=pred_color, **scatter_kws)
    
    # Plot the line of perfect predictions
    ax.plot(y_true, y_true, color=r_color, label='Actual values')
    # Annotate the R-squared value
    ax.legend(labels=[f'Predictions (R² = {r_squared:.2f})', 'Actual values'])
    # Set the title and labels
    ax.set_title(f'{model_name}: R-squared')
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    
    # Display the grid if requested
    if show_grid:
        ax.grid(show_grid)
    
    # Show the plot
    plt.show()

    return ax

def make_plot_colors(
    d , / , 
    colors:str | list[str]=None , axis:int = 0, 
    seed:int  =None, chunk:bool =... 
    ): 
    """ Select colors according to the data size along axis 
    
    Parameters 
    ----------
    d: Arraylike 
       Array data to select colors according to the axis 
    colors: str, list of Matplotlib.colors map, optional 
        The colors for plotting each columns of `X` except the depth. If not
        given, default colors are auto-generated.
        If `colors` is string and 'cs4'or 'xkcd' is included. 
        Matplotlib.colors.CS4_COLORS or Matplotlib.colors.XKCD_COLORS 
        should be used instead. In addition if the `'cs4'` or `'xkcd'` is  
        suffixed by colons and integer value like ``cs4:4`` or ``xkcd:4``, the 
        CS4 or XKCD colors should be used from index equals to ``4``. 
        
 
        Matplotlib.colors.CS4_COLORS or Matplotlib.colors.XKCD_COLORS can 
        be used by setting `colors` to ``'cs4'`` or ``'xkcd'``. To reproduce 
        the same CS4 or XKCD colors, set the `seed` parameter to a 
        specific value. 
           
    axis: int, default=0 
       Axis along with the colors must be generated. By default colors is 
       generated along the row axis 
       
    seed: int, optional 
       Allow to reproduce the Matplotlib.colors.CS4_COLORS if `colors` is 
       set to ``cs4``. 
       
    chunk: bool, default=True 
       Chunk generated colors to fit the exact length of the `d` size 
       
    Returns 
    -------
    colors: list 
       List of new generated colors 
       
    Examples 
    --------
    >>> import numpy as np 
    >>> from gofast.utils.utils import make_plot_colors
    >>> ar = np.random.randn (7, 2) 
    >>> make_plot_colors (ar )
    ['g', 'gray', 'y', 'blue', 'orange', 'purple', 'lime']
    >>> make_plot_colors (ar , axis =1 ) 
    Out[6]: ['g', 'gray']
    >>> make_plot_colors (ar , axis =1 , colors ='cs4')
    ['#F0F8FF', '#FAEBD7']
    >>> len(make_plot_colors (ar , axis =1 , colors ='cs4', chunk=False))
    150
    >>> make_plot_colors (ar , axis =1 , colors ='cs4:4')
    ['#F0FFFF', '#F5F5DC']
    """
    
    # get the data size where colors must be fitted. 
    # note colors should match either the row axis or colurms axis 
    axis = str(axis).lower() 
    if 'columns1'.find (axis)>=0: 
        axis =1 
    else: axis =0
    
    # manage the array 
    d= is_iterable( d, exclude_string=True, transform=True)
    if not hasattr (d, '__array__'): 
        d = np.array(d, dtype =object ) 
    
    axis_length = len(d) if len(d.shape )==1 else d.shape [axis]
    m_cs = make_mpl_properties(axis_length )
    
     #manage colors 
    # we assume the first columns is dedicated for 
    if colors ==...: colors =None 
    if ( 
            isinstance (colors, str) and 
            ( 
                "cs4" in str(colors).lower() 
                 or 'xkcd' in str(colors).lower() 
                 )
            ): 
        #initilize colors infos
        c = copy.deepcopy(colors)
        if 'cs4' in str(colors).lower() : 
            DCOLORS = mcolors.CSS4_COLORS
        else: 
            # remake the dcolors my removing the xkcd: in the keys: 
            DCOLORS = dict(( (k.replace ('xkcd:', ''), c) 
                            for k, c in mcolors.XKCD_COLORS.items()))  
        
        key_colors = list(DCOLORS.keys ())
        colors = list(DCOLORS.values() )
        
        shuffle_cs4=True 
        
        cs4_start= None
        #------
        if ':' in str(c).lower():
            cs4_start = str(c).lower().split(':')[-1]
        #try to converert into integer 
        try: 
            cs4_start= int (cs4_start)
        except : 
            if str(cs4_start).lower() in key_colors: 
                cs4_start= key_colors.index (cs4_start)
                shuffle_cs4=False
            else: 
                pass 
        
        else: shuffle_cs4=False # keep CS4 and dont shuffle 
        
        cs4_start= cs4_start or 0
        
        if shuffle_cs4: 
            np.random.seed (seed )
            colors = list(np.random.choice(colors  , len(m_cs)))
        else: 
            if cs4_start > len(colors)-1: 
                cs4_start = 0 
    
            colors = colors[ cs4_start:]
    
    if colors is not None: 
        if not is_iterable(colors): 
            colors =[colors]
        colors += m_cs 
    else :
        colors = m_cs 
        
    # shrunk data to map the exact colors 
    chunk =True if chunk is ... else False 
    
    return colors[:axis_length] if chunk else colors 

def savefigure(
    fig: object, 
    figname: str = None, 
    ext: str = '.png', 
    **skws
    ):
    """
    Save a matplotlib figure to a file with optional name and extension. 
    
    Parameters
    ----------
    fig : object
        Matplotlib figure object to be saved.
    figname : str, optional
        Name of the output file for the figure. If not provided, the file will
        be named with a timestamp and the extension provided. If a directory
        path is included, ensure the directory exists.
    ext : str, optional
        Extension type for the figure file. Defaults to '.png'. Other common
        types include '.jpg', '.jpeg', '.pdf', and '.svg'.
    **skws : dict
        Additional keyword arguments to pass to `matplotlib.pyplot.savefig`.

    Returns
    -------
    None
        The function saves the figure directly to a file and does not return
        any value.

    Warns
    -----
    UserWarning
        Warns the user if no file name is provided, and a default file name is
        used instead.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [4, 5, 6])
    >>> savefigure(fig, 'my_plot', ext='pdf')
    
    >>> savefigure(fig, 'plot_without_ext')
    
    >>> savefigure(fig)
    """
    ext = '.' + str(ext).lower().strip().replace('.', '')
    
    if figname is None:
        figname = '_' + os.path.splitext(os.path.basename(__file__))[0] +\
                  datetime.datetime.now().strftime('%m-%d-%Y_%H-%M-%S') + '.' + ext
        warnings.warn("No name of figure is given. Figure should be renamed as "
                      f"{figname!r}")
        
    file, ex = os.path.splitext(figname)
    if ex in ('', None):
        ex = ext
        figname = file + '.' + ex

    return fig.savefig(figname, **skws)

def resetting_ticks(get_xyticks, number_of_ticks=None):
    """
    Reset the positions of ticks on the x or y axis modulo 100, returning a
    new array of tick positions.

    Parameters
    ----------
    get_xyticks : list or np.ndarray
        List or ndarray of tick positions obtained via ax.get_x|yticks().
    number_of_ticks : int, optional
        Specifies the number of ticks to set on the x or y axis. If not provided,
        it calculates based on the existing number of ticks or defaults to the
        length of `get_xyticks`.

    Returns
    -------
    list or np.ndarray
        A new list or ndarray of modified tick positions.

    Raises
    ------
    TypeError
        If `get_xyticks` is not a list or ndarray.
    ValueError
        If `number_of_ticks` is not an integer or cannot be converted to an
        integer.

    Examples
    --------
    >>> import numpy as np
    >>> ticks = [10, 20, 30, 40, 50]
    >>> resetting_ticks(ticks)
    >>> resetting_ticks(ticks, 3)
    """
    if not isinstance(get_xyticks, (list, np.ndarray)):
        warnings.warn(
            'Arguments get_xyticks must be a list or ndarray, not <{0}>.'.format(
                type(get_xyticks))
        )
        raise TypeError(
            '<{0}> found. "get_xyticks" must be a list or ndarray.'.format(
                type(get_xyticks))
        )
    
    if number_of_ticks is None:
        if len(get_xyticks) > 2:
            number_of_ticks = int((len(get_xyticks) - 1) / 2)
        else:
            number_of_ticks = len(get_xyticks)
    
    if not isinstance(number_of_ticks, (float, int)):
        try:
            number_of_ticks = int(number_of_ticks)
        except ValueError:
            warnings.warn('"number_of_ticks" must be an integer, not <{0}>.'.format(
                type(number_of_ticks)))
            raise ValueError(f'<{type(number_of_ticks).__name__}> detected. Must be integer.')
        
    number_of_ticks = int(number_of_ticks)
    
    if len(get_xyticks) > 2:
        if get_xyticks[1] % 10 != 0:
            get_xyticks[1] = get_xyticks[1] + (10 - get_xyticks[1] % 10)
        if get_xyticks[-2] % 10 != 0:
            get_xyticks[-2] = get_xyticks[-2] - get_xyticks[-2] % 10
    
        new_array = np.linspace(get_xyticks[1], get_xyticks[-2], number_of_ticks)
    elif len(get_xyticks) < 2:
        new_array = np.array(get_xyticks)

    return new_array

def make_mpl_properties(n: int, prop: str = 'color') -> list:
    """
    Generate a list of matplotlib properties such as colors, markers, or line 
    styles to match the specified number of samples.

    Parameters
    ----------
    n : int
        Number of property items needed. It generates a group of property items.
    prop : str, default='color'
        Name of the property to retrieve. Accepts 'color', 'marker', or 'line'.

    Returns
    -------
    list
        A list of property items with size equal to `n`.

    Raises
    ------
    ValueError
        If the `prop` argument is not one of 'color', 'marker', or 'line'.

    Examples
    --------
    >>> from gofast.utils.utils import make_mpl_properties
    >>> make_mpl_properties(10)
    ['g', 'gray', 'y', 'blue', 'orange', 'purple', 'lime', 'k', 'cyan', (0.6, 0.6, 0.6)]
    >>> make_mpl_properties(100, prop='marker')
    ['o', '^', 'x', 'D', ..., 11, 'None', None, ' ', '']
    >>> make_mpl_properties(50, prop='line')
    ['-', '-', '--', '-.', ..., 'solid', 'dashed', 'dashdot', 'dotted']
    """
    n=int(_assert_all_types(n, int, float, objname ="'n'"))
    prop = str(prop).lower().strip().replace('s', '')
    if prop not in ('color', 'marker', 'line'):
        raise ValueError(f"Property {prop!r} is not available."
                         " Expect 'color', 'marker', or 'line'.")
    # Generate property lists
    props =[]
    if prop=='color': 
        d_colors =  D_COLORS 
        d_colors = mpl.colors.ListedColormap(d_colors[:n]).colors
        if len(d_colors) == n: 
            props= d_colors 
        else:
            rcolors = list(itertools.repeat(
                d_colors , (n + len(d_colors))//len(d_colors))) 
    
            props  = list(itertools.chain(*rcolors))
        
    if prop=='marker': 
        
        d_markers =  D_MARKERS + list(mpl.lines.Line2D.markers.keys()) 
        rmarkers = list(itertools.repeat(
            d_markers , (n + len(d_markers))//len(d_markers))) 
        
        props  = list(itertools.chain(*rmarkers))
    # repeat the lines to meet the number of cv_size 
    if prop=='line': 
        d_lines =  D_STYLES
        rlines = list(itertools.repeat(
            d_lines , (n + len(d_lines))//len(d_lines))) 
        # combine all repeatlines 
        props  = list(itertools.chain(*rlines))
    
    return props [: n ]
  
def resetting_colorbar_bound(
    cbmax: float,
    cbmin: float,
    number_of_ticks: int = 5,
    logscale: bool = False) -> np.ndarray:
    """
    Adjusts the bounds and tick spacing of a colorbar to make the ticks easier
    to read, optionally using a logarithmic scale.

    Parameters
    ----------
    cbmax : float
        Maximum value of the colorbar.
    cbmin : float
        Minimum value of the colorbar.
    number_of_ticks : int, optional
        Number of ticks to be placed on the colorbar. Defaults to 5.
    logscale : bool, optional
        Set to True if the colorbar should use a logarithmic scale. Defaults to False.

    Returns
    -------
    np.ndarray
        Array of tick values for the colorbar.

    Raises
    ------
    ValueError
        If `number_of_ticks` is not an integer or convertible to an integer.

    Examples
    --------
    >>> resetting_colorbar_bound(100, 0)
    array([ 0., 25., 50., 75., 100.])

    >>> resetting_colorbar_bound(100, 0, logscale=True)
    array([  1.,  5.62341,  31.62277,  177.82794,  1000.])
    """
    def round_modulo10(value, mod10):
        """
        Rounds value to nearest multiple of mod10, or to the nearest half multiple.
        """
        if value % mod10 == 0:
            return value
        elif value % (mod10 / 2) == 0:
            return value
        else:
            return value - (value % mod10)

    if not isinstance(number_of_ticks, (float, int)):
        try:
            number_of_ticks = int(number_of_ticks)
        except ValueError:
            warnings.warn('"number_of_ticks" must be an integer, not'
                          f'<{type(number_of_ticks).__name__}>.')
            raise ValueError(f'<{type(number_of_ticks).__name__}> detected.'
                             'Must be an integer.')

    mod10 = np.log10(10) if logscale else 10
    
    if cbmax % cbmin == 0:
        return np.linspace(cbmin, cbmax, number_of_ticks)
    else:
        startpoint = cbmin + (mod10 - cbmin % mod10)
        endpoint = cbmax - (cbmax % mod10)
        return np.array(
            [round_modulo10(ii, mod10) for ii in np.linspace(
                startpoint, endpoint, number_of_ticks)]
        )
    

def plotvec1(u: np.ndarray, z: np.ndarray, v: np.ndarray) -> None:
    """
    Plot three vectors as arrows on a 2D graph to visualize their 
    directions and magnitudes. The vectors should be 2D for proper visualization.

    Parameters
    ----------
    u : np.ndarray
        First vector, displayed in red. Expected to be a 1D array of length 2.
    z : np.ndarray
        Second vector. Expected to be a 1D array of length 2.
    v : np.ndarray
        Third vector, displayed in blue. Expected to be a 1D array of length 2.

    Returns
    -------
    None
        Displays a plot with vectors u, z, and v.

    Raises
    ------
    ValueError
        If any of the vectors are not 1D arrays of length 2.

    Examples
    --------
    >>> import numpy as np
    >>> u = np.array([1, 2])
    >>> z = np.array([2, -1])
    >>> v = np.array([-1, 2])
    >>> plotvec1(u, z, v)
    """
    # Check if all vectors are numpy arrays of length 2
    for vec, name in zip([u, z, v], ['u', 'z', 'v']):
        if not isinstance(vec, np.ndarray) or vec.shape != (2,):
            raise ValueError(f"Vector {name} must be a 1D numpy array of length 2.")

    ax = plt.axes()
    ax.arrow(0, 0, *u, head_width=0.05, color='r', head_length=0.1)
    plt.text(*(u + 0.1), 'u')
    
    ax.arrow(0, 0, *v, head_width=0.05, color='b', head_length=0.1)
    plt.text(*(v + 0.1), 'v')

    ax.arrow(0, 0, *z, head_width=0.05, head_length=0.1)
    plt.text(*(z + 0.1), 'z')

    plt.ylim(-2, 2)
    plt.xlim(-2, 2)
    plt.grid(True)
    plt.show()

def plotvec2(a: np.ndarray, b: np.ndarray) -> None:
    """
    Plot two vectors as arrows on a 2D graph to visualize their directions 
    and orthogonality. The vectors should be 2D for proper visualization.

    Parameters
    ----------
    a : np.ndarray
        First vector, displayed in red. Expected to be a 1D array of length 2.
    b : np.ndarray
        Second vector, displayed in blue. Expected to be a 1D array of length 2.

    Returns
    -------
    None
        Displays a plot with vectors a and b.

    Raises
    ------
    ValueError
        If any of the vectors are not 1D arrays of length 2.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([1, 0])
    >>> b = np.array([0, 1])
    >>> plotvec2(a, b)
    """
    # Validate the input vectors
    for vec, name in zip([a, b], ['a', 'b']):
        if not isinstance(vec, np.ndarray) or vec.shape != (2,):
            raise ValueError(f"Vector {name} must be a 1D numpy array of length 2.")

    ax = plt.axes()
    ax.arrow(0, 0, *a, head_width=0.05, color='r', head_length=0.1)
    plt.text(*(a + 0.1), 'a')
    
    ax.arrow(0, 0, *b, head_width=0.05, color='b', head_length=0.1)
    plt.text(*(b + 0.1), 'b')

    plt.ylim(-2, 2)
    plt.xlim(-2, 2)
    plt.grid(True)
    plt.show()

def plot_errorbar(
    ax,
    x_ar,
    y_ar,
    y_err=None,
    x_err=None,
    color='k',
    marker='x',
    ms=2, 
    ls=':', 
    lw=1, 
    e_capsize=2,
    e_capthick=.5,
    picker=None,
    **kws
 ):
    """
    convinience function to make an error bar instance
    
    Parameters
    ------------
    
    ax: matplotlib.axes 
        instance axes to put error bar plot on

    x_array: np.ndarray(nx)
        array of x values to plot
                  
    y_array: np.ndarray(nx)
        array of y values to plot
                  
    y_error: np.ndarray(nx)
        array of errors in y-direction to plot
    
    x_error: np.ndarray(ns)
        array of error in x-direction to plot
                  
    color: string or (r, g, b)
        color of marker, line and error bar
                
    marker: string
        marker type to plot data as
                 
    ms: float
        size of marker
             
    ls: string
        line style between markers
             
    lw: float
        width of line between markers
    
    e_capsize: float
        size of error bar cap
    
    e_capthick: float
        thickness of error bar cap
    
    picker: float
          radius in points to be able to pick a point. 
        
        
    Returns:
    ---------
    errorbar_object: matplotlib.Axes.errorbar 
           error bar object containing line data, errorbars, etc.
    """
    # this is to make sure error bars 
    #plot in full and not just a dashed line
    eobj = ax.errorbar(
        x_ar,
        y_ar,
        marker=marker,
        ms=ms,
        mfc='None',
        mew=lw,
        mec=color,
        ls=ls,
        xerr=x_err,
        yerr=y_err,
        ecolor=color,
        color=color,
        picker=picker,
        lw=lw,
        elinewidth=lw,
        capsize=e_capsize,
        # capthick=e_capthick
        **kws
         )
    
    return eobj

def get_color_palette (RGB_color_palette): 
    """
    Convert RGB color into matplotlib color palette. In the RGB color 
    system two bits of data are used for each color, red, green, and blue. 
    That means that each color runson a scale from 0 to 255. Black  would be
    00,00,00, while white would be 255,255,255. Matplotlib has lots of
    pre-defined colormaps for us . They are all normalized to 255, so they run
    from 0 to 1. So you need only normalize data, then we can manually  select 
    colors from a color map  

    :param RGB_color_palette: str value of RGB value 
    :type RGB_color_palette: str 
        
    :returns: rgba, tuple of (R, G, B)
    :rtype: tuple
     
    :Example: 
        
        >>> from gofast.utils.utils import get_color_palette 
        >>> get_color_palette (RGB_color_palette ='R128B128')
    """  
    
    def ascertain_cp (cp): 
        if cp >255. : 
            warnings.warn(
                ' !RGB value is range 0 to 255 pixels , '
                'not beyond !. Your input values is = {0}.'.format(cp))
            raise ValueError('Error color RGBA value ! '
                             'RGB value  provided is = {0}.'
                            ' It is larger than 255 pixels.'.format(cp))
        return cp
    if isinstance(RGB_color_palette,(float, int, str)): 
        try : 
            float(RGB_color_palette)
        except : 
              RGB_color_palette= RGB_color_palette.lower()
             
        else : return ascertain_cp(float(RGB_color_palette))/255.
    
    rgba = np.zeros((3,))
    
    if 'r' in RGB_color_palette : 
        knae = RGB_color_palette .replace('r', '').replace(
            'g', '/').replace('b', '/').split('/')
        try :
            _knae = ascertain_cp(float(knae[0]))
        except : 
            rgba[0]=1.
        else : rgba [0] = _knae /255.
        
    if 'g' in RGB_color_palette : 
        knae = RGB_color_palette .replace('g', '/').replace(
            'b', '/').split('/')
        try : 
            _knae =ascertain_cp(float(knae[1]))
        except : 
            rgba [1]=1.
            
        else :rgba[1]= _knae /255.
    if 'b' in RGB_color_palette : 
        knae = knae = RGB_color_palette .replace('g', '/').split('/')
        try : 
            _knae =ascertain_cp(float(knae[1]))
        except :
            rgba[2]=1.
        else :rgba[2]= _knae /255.
        
    return tuple(rgba)       

def _get_xticks_formatage (
        ax,  xtick_range, space= 14 , step=7,
        fmt ='{}',auto = False, ticks ='x', **xlkws):
    """ Skip xticks label at every number of spaces 
    :param ax: matplotlib axes 
    :param xtick_range: list of the xticks values 
    :param space: interval that the label must be shown.
    :param step: the number of label to skip.
    :param fmt: str, formatage type. 
    :param ticks: str, default='x', the ticks axis to format the labels. 
      can be ``'y'``. 
    :param auto: bool , if ``True`` a dynamic tick formatage will start. 
    
    """
    def format_ticks (ind, x):
        """ Format thick parameter with 'FuncFormatter(func)'
        rather than using:: 
            
        axi.xaxis.set_major_locator (plt.MaxNLocator(3))
        
        ax.xaxis.set_major_formatter (plt.FuncFormatter(format_thicks))
        """
        if ind % step ==0: 
            return fmt.format (ind)
        else: None 
        
    # show label every 'space'samples 
    if auto: 
        space = 10.
        step = int (np.ceil ( len(xtick_range)/ space )) 
        
    rotation = xlkws.get('rotation', 90 ) if 'rotation' in xlkws.keys (
        ) else xlkws.get('rotate_xlabel', 90 )
    
    if len(xtick_range) >= space :
        if ticks=='y': 
            ax.yaxis.set_major_formatter (plt.FuncFormatter(format_ticks))
        else: 
            ax.xaxis.set_major_formatter (plt.FuncFormatter(format_ticks))

        plt.setp(ax.get_yticklabels() if ticks=='y' else ax.get_xticklabels(), 
                 rotation = rotation )
    else: 
        
        # ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
        # # ticks_loc = ax.get_xticks().tolist()
        # ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(ticks_loc))
        # ax.set_xticklabels([fmt.format(x) for x in ticks_loc])
        tlst = [fmt.format(item) for item in  xtick_range]
        ax.set_yticklabels(tlst, **xlkws) if ticks=='y' \
            else ax.set_xticklabels(tlst, **xlkws) 
  
def _set_sns_style (s, /): 
    """ Set sns style whether boolean or string is given""" 
    s = str(s).lower()
    s = re.sub(r'true|none', 'darkgrid', s)
    return sns.set_style(s) 

def _is_target_in (X, y=None, target_name=None): 
    """ Create new target name for target_name if given 
    
    :param X: dataframe 
        dataframe containing the data for plotting 
    :param y: array or series
        target data for plotting. Note that multitarget outpout is not 
        allowed yet. Moroever, it `y` is given as a dataframe, 'target_name' must 
        be supplied to retrive y as a pandas series object, otherwise an 
        error will raise. 
    :param target_name: str,  
        target name. If given and `y` is ``None``, Will try to find `target_name`
        in the `X` columns. If 'target_name' does not exist, plot for target is 
        cancelled. 
        
    :return y: Series 
    """
    _assert_all_types(X, pd.DataFrame)
    
    if y is not None: 
        y = _assert_all_types(y , pd.Series, pd.DataFrame, np.ndarray)
        
        if hasattr (y, 'columns'): 
            if target_name not in (y.columns): target_name = None 
            if target_name is None: 
                raise TypeError (
                    "'target_name' must be supplied when y is a dataframe.")
            y = y [target_name ]
        elif hasattr (y, 'name'): 
            target_name = target_name or y.name 
            # reformat inplace the name of series 
            y.name = target_name 
            
        elif hasattr(y, '__array__'): 
            y = pd.Series (y, name = target_name or 'target')
            
    elif y is None: 
        if target_name in X.columns :
            y = X.pop(target_name)

    return X, y 

def _toggle_target_in  (X , y , pos=None): 
    """ Toggle the target in the convenient position. By default the target 
    plot is the last subplots 
    
    :param X: dataframe 
        dataframe containing the data for plotting 
    :param y: array or series
        the target for  plotting. 
    :param pos: int, the position to insert y in the dataframe X 
        By default , `y` is located at the last position 
        
    :return: Dataframe 
        Dataframe containing the target 'y'
        
    """
    
    pos =  0 if pos ==0  else ( pos or X.shape [1])

    pos= int ( _assert_all_types(pos, int, float ) ) 
    ms= ("The positionning of the target is out of the bound."
         "{} position is used instead.")
    
    if pos > X.shape[1] : 
        warnings.warn(ms.format('The last'))
        pos=X.shape[1]
    elif pos < 0: 
        warnings.warn(ms.format(
            " Negative index is not allowed. The first")
                      )
        pos=0 
 
    X.insert (pos, y.name, y )
    
    return X
    
def _skip_log10_columns ( X, column2skip, pattern =None , inplace =True): 
    """ Skip the columns that dont need to put value in logarithms.
    
    :param X: dataframe 
        pandas dataframe with valid columns 
    :param column2skip: list or str , 
        List of columns to skip. If given as string and separed by the default
        pattern items, it should be converted to a list and make sure the 
        columns name exist in the dataframe. Otherwise an error with 
        raise. 
    :param pattern: str, default = '[#&*@!,;\s]\s*'
        The base pattern to split the text in `column2skip` into a columns
        
    :return X: Dataframe
        Dataframe modified inplace with values computed in log10 
        except the skipped columns. 
        
    :example: 
       >>> from gofast.datasets import load_hlogs 
       >>> from gofast.utils.utils import _skip_log10_columns 
       >>> X0, _= load_hlogs (as_frame =True ) 
       >>> # let visualize the  first3 values of `sp` and `resistivity` keys 
       >>> X0['sp'][:3] , X0['resistivity'][:3]  
       ... (0   -1.580000
            1   -1.580000
            2   -1.922632
            Name: sp, dtype: float64,
            0    15.919130
            1    16.000000
            2    24.422316
            Name: resistivity, dtype: float64)
       >>> column2skip = ['hole_id','depth_top', 'depth_bottom', 
                         'strata_name', 'rock_name', 'well_diameter', 'sp']
       >>> _skip_log10_columns (X0, column2skip)
       >>> # now let visualize the same keys values 
       >>> X0['sp'][:3] , X0['resistivity'][:3]
       ... (0   -1.580000
            1   -1.580000
            2   -1.922632
            Name: sp, dtype: float64,
            0    1.201919
            1    1.204120
            2    1.387787
            Name: resistivity, dtype: float64)
      >>> # it is obvious the `resistiviy` values is log10 
      >>> # while `sp` still remains the same 
      
    """
    X0 = X.copy () 
    if not is_iterable( column2skip): 
        raise TypeError ("Columns  to skip expect an iterable object;"
                         f" got {type(column2skip).__name__!r}")
        
    pattern = pattern or r'[#&*@!,;\s]\s*'
    
    if isinstance(column2skip, str):
        column2skip = str2columns (column2skip, pattern=pattern  )
    #assert whether column to skip is in 
    if column2skip:
        cskip = copy.deepcopy (column2skip) 
        column2skip = is_in_if(X.columns, column2skip, return_diff= True)
        if len(column2skip) ==len (X.columns): 
            warnings.warn("Value(s) to skip are not detected.")
        if inplace : 
            X[column2skip] = np.log10 ( X[column2skip] ) 
            X.drop (columns =cskip , inplace =True )
            return  
        else : 
            X0[column2skip] = np.log10 ( X0[column2skip] ) 
            
    return X0
    
def plot_bar(x, y, wh= .8,  kind ='v', fig_size =(8, 6), savefig=None,
             xlabel =None, ylabel=None, fig_title=None, **bar_kws): 
    """
    Make a vertical or horizontal bar plot.

    The bars are positioned at x or y with the given alignment. Their dimensions 
    are given by width and height. The horizontal baseline is left (default 0)
    while the vertical baseline is bottom (default=0)
    
    Many parameters can take either a single value applying to all bars or a 
    sequence of values, one for each bar.
    
    Parameters 
    -----------
    x: float or array-like
        The x coordinates of the bars. is 'x' for vertical bar plot as `kind` 
        is set to ``v``(default) or `y` for horizontal bar plot as `kind` is 
        set to``h``. 
        See also align for the alignment of the bars to the coordinates.
    y: float or array-like
        The height(s) for vertical and width(s) for horizonatal of the bars.
    
    wh: float or array-like, default: 0.8
        The width(s) for vertical or height(s) for horizaontal of the bars.
        
    kind: str, ['vertical', 'horizontal'], default='vertical'
        The kind of bar plot. Can be the horizontal or vertical bar plots. 
    bar_kws: dict, 
        Additional keywords arguments passed to : 
            :func:`~matplotlib.pyplot.bar` or :func:`~matplotlib.pyplot.barh`. 
    """
    
    assert str(kind).lower().strip() in ("vertical", 'v',"horizontal", "h"), (
        "Support only the horizontal 'h' and vertical 'v' bar plots."
        " Got {kind!r}")
    kind =str(kind).lower().strip()
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize =fig_size)
    if kind in ("vertical", "v"): 
        ax.bar (x, height= y, width =  wh , **bar_kws)
    elif kind in ("horizontal", "h"): 
        ax.barh (x , width =y , height =wh, **bar_kws)
        
    ax.set_xlabel (xlabel )
    ax.set_ylabel(ylabel) 
    ax.set_title (fig_title)
    if savefig is not  None: 
        savefigure (fig, savefig, dpi = 300)
        
    plt.close () if savefig is not None else plt.show() 
    

def _format_ticks (value, tick_number, fmt ='S{:02}', nskip =7 ):
    """ Format thick parameter with 'FuncFormatter(func)'
    rather than using `axi.xaxis.set_major_locator (plt.MaxNLocator(3))`
    ax.xaxis.set_major_formatter (plt.FuncFormatter(format_thicks))
    
    :param value: tick range values for formatting 
    :param tick_number: number of ticks to format 
    :param fmt: str, default='S{:02}', kind of tick formatage 
    :param nskip: int, default =7, number of tick to skip 
    
    """
    if value % nskip==0: 
        return fmt.format(int(value)+ 1)
    else: None 
    

def plot_confidence(
    y: Optional[Union[str, ArrayLike]] = None, 
    x: Optional[Union[str, ArrayLike]] = None,  
    data: Optional[DataFrame] = None,  
    ci: float = .95,  
    kind: str = 'line', 
    b_samples: int = 1000, 
    **sns_kws: Dict
) -> plt.Axes:
    """
    Plot confidence interval data using a line plot, regression plot, 
    or the bootstrap method.
    
    A Confidence Interval (CI) is an estimate derived from observed data statistics, 
    indicating a range where a population parameter is likely to be found at a 
    specified confidence level. Introduced by Jerzy Neyman in 1937, CI is a crucial 
    concept in statistical inference. Common types include CI for mean, median, 
    the difference between means, a proportion, and the difference in proportions.

    Parameters 
    ----------
    y : Union[np.ndarray, str], optional
        Dependent variable values. If a string, `y` should be a column name in 
        `data`. `data` cannot be None in this case.
    x : Union[np.ndarray, str], optional
        Independent variable values. If a string, `x` should be a column name in 
        `data`. `data` cannot be None in this case.
    data : pd.DataFrame, optional
        Input data structure. Can be a long-form collection of vectors that can be 
        assigned to named variables or a wide-form dataset that will be reshaped.
    ci : float, default=0.95
        The confidence level for the interval.
    kind : str, default='line'
        The type of plot. Options include 'line', 'reg', or 'bootstrap'.
    b_samples : int, default=1000
        The number of bootstrap samples to use for the 'bootstrap' method.
    sns_kws : Dict
        Additional keyword arguments passed to the seaborn plot function.

    Returns 
    -------
    ax : matplotlib.axes.Axes
        The matplotlib axes containing the plot.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.plot.utils import plot_confidence 
    >>> df = pd.DataFrame({'x': range(10), 'y': np.random.rand(10)})
    >>> ax = plot_confidence(x='x', y='y', data=df, kind='line', ci=0.95)
    >>> plt.show()
    
    >>> ax = plot_confidence(y='y', data=df, kind='bootstrap', ci=0.95, b_samples=500)
    >>> plt.show()
    """

    plot_functions = {
        'line': lambda: sns.lineplot(data=data, x=x, y=y, errorbar=('ci', ci), **sns_kws),
        'reg': lambda: sns.regplot(data=data, x=x, y=y, ci=ci, **sns_kws)
    }
    if isinstance ( data, dict): 
        data = pd.DataFrame ( data )
        
    x, y = assert_xy_in(x, y, data=data, ignore ="x")
    if kind in plot_functions:
        ax = plot_functions[kind]()
    elif kind.lower().startswith('boot'):
        if y is None:
            raise ValueError("y must be provided for bootstrap method.")
        if not isinstance(b_samples, int):
            raise ValueError("`b_samples` must be an integer.")

        medians = [np.median(resample(y, n_samples=len(y))) for _ in range(b_samples)]
        plt.hist(medians)
        plt.show()
        
        p = ((1.0 - ci) / 2.0) * 100
        lower, upper = np.percentile(medians, [p, 100 - p])
        print(f"{ci*100}% confidence interval between {lower} and {upper}")

        ax = plt.gca()
    else:
        raise ValueError(f"Unrecognized plot kind: {kind}")

    return ax
   
def plot_confidence_ellipse(
    x: Union[str, ArrayLike], 
    y: Union[str, ArrayLike], 
    data: Optional[DataFrame] = None,
    figsize: Tuple[int, int] = (6, 6),
    scatter_s: int = 0.5,
    line_colors: Tuple[str, str, str] = ('firebrick', 'fuchsia', 'blue'),
    line_styles: Tuple[str, str, str] = ('-', '--', ':'),
    title: str = 'Different Standard Deviations',
    show_legend: bool = True
) -> plt.Axes:
    """
    Plots the confidence ellipse of a two-dimensional dataset.

    This function visualizes the confidence ellipse representing the covariance 
    of the provided 'x' and 'y' variables. The ellipses plotted represent 1, 2, 
    and 3 standard deviations from the mean.

    Parameters
    ----------
    x : Union[str, np.ndarray, pd.Series]
        The x-coordinates of the data points or column name in DataFrame.
    y : Union[str, np.ndarray, pd.Series]
        The y-coordinates of the data points or column name in DataFrame.
    data : pd.DataFrame, optional
        DataFrame containing x and y data. Required if x and y are column names.
    figsize : Tuple[int, int], optional
        Size of the figure (width, height). Default is (6, 6).
    scatter_s : int, optional
        The size of the scatter plot markers. Default is 0.5.
    line_colors : Tuple[str, str, str], optional
        The colors of the lines for the 1, 2, and 3 std deviation ellipses.
    line_styles : Tuple[str, str, str], optional
        The line styles for the 1, 2, and 3 std deviation ellipses.
    title : str, optional
        The title of the plot. Default is 'Different Standard Deviations'.
    show_legend : bool, optional
        If True, shows the legend. Default is True.

    Returns
    -------
    ax : plt.Axes
        The matplotlib axes containing the plot.

    Note 
    -----
    The approach that is used to obtain the correct geometry 
    is explained and proved here:
      https://carstenschelp.github.io/2018/09/14/Plot_Confidence_Ellipse_001.html
      
    The method avoids the use of an iterative eigen decomposition 
    algorithm and makes use of the fact that a normalized covariance 
    matrix (composed of pearson correlation coefficients and ones) is 
    particularly easy to handle.
    
    Example
    -------
    >>> import numpy as np 
    >>> from gofast.plot.utils import plot_confidence_ellipse
    >>> x = np.random.normal(size=500)
    >>> y = np.random.normal(size=500)
    >>> ax = plot_confidence_ellipse(x, y)
    >>> plt.show()
    """
    x, y= assert_xy_in(x, y, data = data )

    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(c='grey', lw=1)
    ax.axhline(c='grey', lw=1)
    ax.scatter(x, y, s=scatter_s)

    for n_std, color, style in zip([1, 2, 3], line_colors, line_styles):
        confidence_ellipse(x, y, ax, n_std=n_std, label=f'${n_std}\sigma$',
                           edgecolor=color, linestyle=style)

    ax.set_title(title)
    if show_legend:
        ax.legend()
    return ax

def confidence_ellipse(
    x: Union[str, ArrayLike], 
    y: Union[str, ArrayLike], 
    ax: plt.Axes, 
    n_std: float = 3.0, 
    facecolor: str = 'none', 
    data: Optional[DataFrame] = None,
    **kwargs
) -> Ellipse:
    """
    Creates a covariance confidence ellipse of x and y.

    Parameters
    ----------
    x, y : np.ndarray
        Input data arrays with the same size.
    ax : plt.Axes
        The axes object where the ellipse will be plotted.
    n_std : float, optional
        The number of standard deviations to determine the ellipse's radius. 
        Default is 3.
    facecolor : str, optional
        The color of the ellipse's face. Default is 'none' (no fill).
    data : pd.DataFrame, optional
        DataFrame containing x and y data. Required if x and y are column names.

    **kwargs
        Additional arguments passed to the Ellipse patch.

    Returns
    -------
    ellipse : Ellipse
        The Ellipse object added to the axes.

    Raises
    ------
    ValueError
        If 'x' and 'y' are not of the same size.

    Example
    -------
    >>> import numpy as np 
    >>> from gofast.plot.utils import confidence_ellipse
    >>> x = np.random.normal(size=500)
    >>> y = np.random.normal(size=500)
    >>> fig, ax = plt.subplots()
    >>> confidence_ellipse(x, y, ax, n_std=2, edgecolor='red')
    >>> ax.scatter(x, y, s=3)
    >>> plt.show()
    """
    x, y = assert_xy_in(x, y, data = data )

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D().rotate_deg(45).scale(
        scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def plot_text (
    x, y, 
    text=None , 
    data =None, 
    coerce =False, 
    basename ='S', 
    fig_size =( 7, 7 ), 
    show_line =False, 
    step = None , 
    xlabel ='', 
    ylabel ='', 
    color= 'k', 
    mcolor='k', 
    lcolor=None, 
    show_leg =False,
    linelabel='', 
    markerlabel='', 
    ax=None, 
    **text_kws
    ): 
    """ Plot text(s) indicating each position in the line. 
    
    Parameters 
    -----------
    x, y: str, float, Array-like 
        The position to place the text. By default, this is in data 
        coordinates. The coordinate system can be changed using the 
        transform parameter.
        
    text: str, 
        The text
        
    data: pd.DataFrame, 
       Data containing x and y names. Need to be supplied when x and y 
       are given as string names. 
       
    coerce:bool, default=False 
       Force the plot despite the given textes do not match the number of  
       positions `x` and `y`. If ``False``, number of positions must be 
       consistent with x and y, otherwise error raises. 
       
    basename: str, default='S' 
       the text to prefix the position when the text is not given. 
       
    fig_size: tuple, default=(7, 7) 
       Matplotlib figure size.
       
    show_line: bool, default=False 
       Display the line from x, y. 
       
    step: int,Optional 
       The number of intermediate positions to skip in the plotting text. 
       
    xlabel, ylabel: str, Optional, 
       The labels of x and y. 
       
    color: str, default='k', 
       Text color.
       
    mcolor: str, default='k', 
       Marker color. 
       
    lcolor: str, Optional 
       Line color if `show_line` is set to ``True``. 
       
    show_leg: bool, default=False 
       Display the legend of line and marker labels. 
       
    linelabel, markerlabel: str, Optional 
        The labels of the line and marker. 
       
    ax: Matplotlib.Axes, optional 
       Support plot to another axes 
       
       .. versionadded:: 0.2.5 
       
    text_kws: dict, 
       Keyword arguments passed to :meth:`matplotlib.axes.Axes.text`. 

    Return 
    -------
    ax: Matplotlib axes 
    
    Examples 
    --------
    >>> import gofast as gf 
    >>> data =gf.make_erp (as_frame =True, n_stations= 7 )
    >>> x , y =[ 0, 1, 3 ], [2, 3, 6] 
    >>> texto = ['AMT-E1147', 'AMT-E1148',  'AMT-E180']
    >>> plot_text (x, y , text = texto)# no need to set  coerce, same length 
    >>> data =gf.make_erp (as_frame =True, n_stations= 20 )
    >>> x , y = data.easting, data.northing
    >>> text1 = ['AMT-E1147', 'AMT-E1148',  'AMT-E180'] 
    >>> plot_text (x, y , coerce =True , text = text1 , show_leg= True, 
                   show_line=True, linelabel='E1-line', markerlabel= 'Site', 
               basename ='AMT-E0' 
               )
    """
    # assume x, y  series are passed 
    if isinstance(x, str) or hasattr ( x, 'name'): 
        xlabel = x  if isinstance(x, str) else x.name 
        
    if isinstance(y, str) or hasattr ( y, 'name'): 
        ylabel = y  if isinstance(y, str) else y.name 
        
    if x is None and  y is None:
        raise TypeError("x and y are needed for text plot. NoneType"
                        " cannot be plotted.")    
        
    x, y = assert_xy_in(x, y, data = data ) 

    if text is None and not coerce: 
       raise TypeError ("Text cannot be plotted. To force plotting text with"
                        " the basename, set ``coerce=True``.")

    text = is_iterable(text , exclude_string= True , transform =True )
    
    if ( len(text) != len(y) 
        and not coerce) : 
        raise ValueError("In principle text array and x/y must be consistent."
                         f" Got {len(text)} and {len(y)}. To plot anyway,"
                         " set ``coerce=True``.")
    if coerce : 
        basename =str(basename)
        text += [f'{basename}{i+len(text):02}' for i in range (len(y) )]

    if step is not None: 
        step = _assert_all_types(step , float, int , objname ='Step') 
        for ii in range(len(text)): 
            if not ii% step ==0: 
                text[ii]=''

    if ax is None: 
        
        fig, ax = plt.subplots(1,1, figsize =fig_size)
    
    # plot = ax.scatter if show_line else ax.plot 
    ax_m = None 
    if show_line: 
        ax.plot (x, y , label = linelabel, color =lcolor 
                 ) 
        
    for ix, iy , name in zip (x, y, text ): 
        ax.text ( ix , iy , name , color = color,  **text_kws)
        if name !='':
           ax_m  = ax.scatter ( [ix], [iy] , marker ='o', color =mcolor, 
                       )
  
    ax.set_xlabel (xlabel)
    ax.set_ylabel (ylabel) 
    
    ax_m.set_label ( markerlabel) if ax_m is not None else None 
    
    if show_leg : 
        ax.legend () 
        
    return ax 

def _make_axe_multiple ( n, ncols = 3 , fig_size =None, fig =None, ax= ... ): 
    """ Make multiple subplot axes from number of objects. """
    if is_iterable (n): 
       n = len(n) 
     
    nrows = n // ncols + ( n % ncols ) 
    if nrows ==0: 
       nrows =1 
       
    if ax in ( ... , None) : 
        fig, ax = plt.subplots (nrows, ncols, figsize = fig_size )  
    
    return fig , ax 
    

def _manage_plot_kws ( kws, dkws = dict () ): 
    """ Check whether the default values are in plot_kws then pop it"""
    
    kws = dkws or kws 
    for key in dkws.keys(): 
        # if key not in then add it. 
        if key not in kws.keys(): 
            kws[key] = dkws.get(key)
            
    return kws 

def is_colormap(color_name):
    """
    Checks if the given color name is a valid colormap in Matplotlib.

    Parameters:
    - color_name: str, the name of the color or colormap to check.

    Returns:
    - bool, True if the color_name is a colormap, False otherwise.
    """
    # Get the list of all colormaps in Matplotlib
    colormaps = plt.colormaps()
    # Check if the given color_name is in the list of colormaps
    return color_name in colormaps

@validate_params ({ 
    "sensitivity_values": ['array-like'], 
    "features": [str, 'array-like', None ], 
    "plot_type": [StrOptions({'single', 'pair', 'triple'})], 
    })
def plot_distributions(
    data: pd.DataFrame,
    features: Optional[Union[List[str], str]] = None,
    bins: int = 30,
    kde: bool = True,
    hist: bool = True,
    figsize: tuple = (10, 6),
    title: str = "Feature Distributions",
    plot_type: str = 'single', 
    **kwargs
):
    """
    Plots the distribution of numeric columns in the DataFrame. The function 
    allows customization of the plot to include histograms and/or kernel 
    density estimates (KDE). It supports univariate, bivariate, and trivariate 
    distributions.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the data to plot. Only numeric columns 
        are used for plotting the distribution.

    features : list of str or str, optional
        The specific features (columns) to plot. If not provided, all 
        numeric features are plotted. If a single feature is provided as 
        a string, it is automatically treated as a list.

    bins : int, optional, default=30
        The number of bins to use for the histogram. This parameter is 
        ignored if `hist` is set to False.
    
    kde : bool, optional, default=True
        Whether or not to include the Kernel Density Estimate (KDE) plot. 
        If False, only the histogram will be shown.
    
    hist : bool, optional, default=True
        Whether or not to include the histogram in the plot. If False, 
        only the KDE plot will be shown.
    
    figsize : tuple, optional, default=(10, 6)
        The size of the figure (width, height) to create for the plot.
    
    title : str, optional, default="Feature Distributions"
        Title of the plot.
    
    plot_type : str, optional, default='single'
        Type of distribution plot to generate:
        - 'single': Univariate distribution (1D plot for each feature).
        - 'pair': Bivariate distribution (2D plot for two features).
        - 'triple': Trivariate distribution (3D plot for three features).

    **kwargs : additional keyword arguments passed to seaborn or matplotlib 
              functions for further customization.

    Returns
    -------
    None
        Displays the plot of the distributions.

    Examples
    --------
    >>> from gofast.utils.mlutils import plot_distributions
    >>> from gofast.datasets import load_hlogs
    >>> data = load_hlogs().frame  # get the frame
    >>> plot_distributions(data, features=['longitude', 'latitude', 'subsidence'],
                           plot_type='triple')
    """

    # If no specific features provided, select numeric features automatically
    if features is None:
        features = data.select_dtypes(include=np.number).columns.tolist()
    
    features = columns_manager(features) 
    
    # Ensure features are valid
    invalid_features = [f for f in features if f not in data.columns]
    if invalid_features:
        raise ValueError(f"Invalid features: {', '.join(invalid_features)}")

    # Univariate Plot (Single feature distribution)
    if plot_type == 'single':
        plt.figure(figsize=figsize)
        for feature in features:
            plt.subplot(len(features), 1, features.index(feature) + 1)
            if hist:
                sns.histplot(data[feature], kde=kde, bins=bins, **kwargs)
            plt.title(f"{feature} Distribution")
        plt.tight_layout()
        plt.show()

    # Bivariate Plot (Pairwise feature distribution)
    elif plot_type == 'pair':
        if len(features) != 2:
            raise ValueError(
                "For 'pair' plot type, exactly 2 features must be specified.")
        
        plt.figure(figsize=figsize)
        sns.jointplot(data=data, x=features[0], y=features[1], kind="kde", **kwargs)
        plt.suptitle(f"{features[0]} vs {features[1]} Distribution")
        plt.show()

    # Trivariate Plot (3D distribution)
    elif plot_type == 'triple':
        if len(features) != 3:
            raise ValueError(
                "For 'triple' plot type, exactly 3 features must be specified.")

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        x = data[features[0]]
        y = data[features[1]]
        z = data[features[2]]
        
        ax.scatter(x, y, z, c=z, cmap='viridis')
        
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        ax.set_zlabel(features[2])
        plt.title(f"{features[0]} vs {features[1]} vs {features[2]} Distribution")
        plt.show()

@isdf 
def plot_spatial_distribution(
    df: DataFrame,
    category_column: str,  
    continuous_bins: Union[str, List[float]] = 'auto',  
    categories: Optional[List[str]] = None,  
    filter_categories: Optional[List[str]] = None,
    spatial_cols: tuple = ('longitude', 'latitude'), 
    cmap: str = 'coolwarm',  
    plot_type: str = 'scatter', 
    alpha: float = 0.7, 
    show_grid:bool=True, 
    axis_off: bool = False,  
    figsize: tuple = (10, 8)  
) -> None:
    """
    Plot the Spatial Distribution of a Specified Category or Continuous Variable.

    This function visualizes the spatial distribution of geographical data points
    based on a specified categorical or continuous variable. For continuous data,
    it categorizes the values into defined bins, allowing for an intuitive
    representation of data intensity across a geographical area. The visualization
    can be rendered using scatter plots or hexbin plots, facilitating the analysis
    of spatial patterns and concentrations.

    The categorization of continuous variables is performed using either user-defined
    bins or the Freedman-Diaconis rule to determine an optimal bin width:
    
    .. math::
        \text{Bin Width} = 2 \times \frac{\text{IQR}}{n^{1/3}}
    
    where :math:`\text{IQR}` is the interquartile range of the data and :math:`n`
    is the number of observations.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe containing the geographical and categorical or 
        continuous data. It must include at least the following columns:
        
        - `longitude`: Longitude coordinates of the data points.
        - `latitude`: Latitude coordinates of the data points.
        - *category_column*: The column specified by `category_column` parameter.

    category_column : str
        The name of the column in `df` to be used for categorization. This column 
        can be either categorical or continuous. If categorical, the data will be 
        used directly. If continuous, the data will be binned into categories 
        based on `continuous_bins`.

    continuous_bins : Union[str, List[float]], default='auto'
        Defines the bin edges for categorizing continuous data. 
        
        - If set to `'auto'`, the function applies the Freedman-Diaconis rule to 
          determine optimal bin widths.
        - If a list of floats is provided, these values are used as the bin edges. 
          The provided bins must encompass the entire range of the data in 
          `category_column`.
        
        Raises a `ValueError` if the provided bins do not cover the data range.
    
    spatial_cols : tuple, optional, default=('longitude', 'latitude')
            A tuple containing the names of the longitude and latitude columns.
            Must consist of exactly two elements. The function will validate that
            these columns exist in the dataframe and are used as the spatial 
            coordinates for plotting.
            
            .. note::
                Ensure that the specified `spatial_cols` are present in 
                the dataframe and accurately represent geographical coordinates.
                
    categories : Optional[List[str]], default=None
        A list of labels corresponding to each bin when categorizing continuous 
        data. 
        
        - If `None`, the categories are auto-generated based on the bin ranges.
        - If provided, the length of `categories` must match the number of bins 
          minus one.
        
        If the number of categories does not match the number of bins, a warning 
        is issued and categories are auto-generated.

    filter_categories : Optional[List[str]], default=None
        Specifies which categories to include in the visualization. 
        
        - If `None`, all categories are plotted.
        - If a list is provided, only the specified categories are visualized, and 
          others are excluded. The legend reflects only the displayed categories.
        
        Raises a `ValueError` if none of the `filter_categories` are valid.

    cmap : str, default='coolwarm'
        The colormap to use for the visualization. This parameter utilizes matplotlib's 
        colormap names (e.g., `'viridis'`, `'plasma'`, `'inferno'`, `'magma'`, 
        `'cividis'`, etc.).

    plot_type : str, default='scatter'
        The type of plot to generate. 
        
        - `'scatter'`: Generates a scatter plot.
        - `'hexbin'`: Generates a hexbin plot, suitable for large datasets.
        
        Raises a `ValueError` if an unsupported `plot_type` is provided.

    alpha : float, default=0.7
        The transparency level for scatter plots. Ranges from 0 (completely 
        transparent) to 1 (completely opaque).
        
    show_grid : bool, default=True
        If set to `False`, the grid are turned off, providing a cleaner 
        visualization without grid lines.

    axis_off : bool, default=False
        If set to `True`, the plot axes are turned off, providing a cleaner 
        visualization without axis lines and labels.

    figsize : tuple, default=(10, 8)
        Specifies the size of the plot in inches as `(width, height)`.

    Returns
    -------
    None
        This function does not return any value. It renders the plot directly.

    Raises
    ------
    ValueError
        - If `category_column` does not exist in `df`.
        - If `category_column` is neither numeric nor categorical.
        - If `spatial_cols` is not a tuple of two elements or columns 
          are missing.
        - If `continuous_bins` is neither `'auto'` nor a list of numbers.
        - If provided `continuous_bins` do not cover the data range.
        - If the number of `categories` does not match the number of bins.
        - If no valid `filter_categories` are provided after filtering.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from gofast.plot.utils import plot_spatial_distribution

    >>> # Sample DataFrame
    >>> data = {
    ...     'longitude': np.random.uniform(-100, -90, 100),
    ...     'latitude': np.random.uniform(30, 40, 100),
    ...     'subsidence': np.random.choice(['minimal', 'moderate', 'severe'], 100)
    ... }
    >>> df = pd.DataFrame(data)
    
    >>> # Plot only 'severe' subsidence
    >>> plot_spatial_distribution(
    ...     df=df,
    ...     category_column='subsidence',
    ...     categories=['minimal', 'moderate', 'severe'],
    ...     filter_categories=['severe'],
    ...     plot_type='scatter'
    ... )
    
    >>> # Plot 'moderate' and 'severe' subsidence
    >>> plot_spatial_distribution(
    ...     df=df,
    ...     category_column='subsidence',
    ...     categories=['minimal', 'moderate', 'severe'],
    ...     filter_categories=['moderate', 'severe'],
    ...     plot_type='scatter'
    ... )
    
    >>> # Plot all categories
    >>> plot_spatial_distribution(
    ...     df=df,
    ...     category_column='subsidence',
    ...     categories=['minimal', 'moderate', 'severe'],
    ...     filter_categories=None,
    ...     plot_type='scatter'
    ... )

    Notes
    -----
    - The function automatically determines whether the `category_column` is 
      categorical or continuous based on its data type.
    - When categorizing continuous data, ensure that the provided `continuous_bins` 
      comprehensively cover the data range to avoid missing data points.
    - The legend in the plot dynamically adjusts based on the `filter_categories` 
      parameter, displaying only the relevant categories.

    See Also
    --------
    pandas.cut : Function to bin continuous data into discrete intervals.
    seaborn.scatterplot : Function to create scatter plots.
    matplotlib.pyplot.hexbin : Function to create hexbin plots.
    check_spatial_columns : Function to validate spatial columns in the dataframe.

    References
    ----------
    .. [1] Freedman, D., & Diaconis, P. (1981). On the histogram as a density estimator:
       L2 theory. *Probability Theory and Related Fields*, 57(5), 453-476.
    .. [2] Seaborn: Statistical Data Visualization. https://seaborn.pydata.org/
    .. [3] Matplotlib: Visualization with Python. https://matplotlib.org/
    """
    # make a copy for safety 
    df =df.copy() 
    # Check if category_column exists in dataframe
    if category_column not in df.columns:
        raise ValueError(
            f"Column '{category_column}' does not exist in the dataframe."
        )
    # Check whether 
    check_spatial_columns(df )
    
    # Determine if the category_column is categorical
    if pd.api.types.is_categorical_dtype(df[category_column]) or \
       df[category_column].dtype == object:
        is_categorical = True
    elif pd.api.types.is_numeric_dtype(df[category_column]):
        is_categorical = False
    else:
        raise ValueError(
            f"Column '{category_column}' must be either numeric or categorical."
        )

    if not is_categorical:
        # Handle continuous data
        if continuous_bins == 'auto':
            # Use Freedman-Diaconis rule for bin width
            q25, q75 = np.percentile(
                df[category_column].dropna(), [25, 75]
            )
            iqr = q75 - q25
            bin_width = 2 * iqr * len(df) ** (-1 / 3)
            if bin_width == 0:
                bin_width = 1  # Fallback to bin width of 1
            bins = np.arange(
                df[category_column].min(),
                df[category_column].max() + bin_width,
                bin_width
            )
            bins = np.round(bins, decimals=2)
        elif isinstance(continuous_bins, list):
            bins = sorted(continuous_bins)
            if (bins[0] > df[category_column].min()) or \
               (bins[-1] < df[category_column].max()):
                raise ValueError(
                    "Provided continuous_bins do not cover the range of the data."
                )
        else:
            raise ValueError(
                "continuous_bins must be 'auto' or a list of numbers."
            )

        # Categorize the continuous data based on bins
        df['category'] = pd.cut(
            df[category_column],
            bins=bins,
            labels=categories[:len(bins) - 1] if categories else None,
            include_lowest=True,
            right=False
        )

        # Handle category labels if not provided
        if categories is None:
            df['category'] = df['category'].astype(str)
        else:
            if len(categories) != len(bins) - 1:
                warnings.warn(
                    "Number of categories does not match number of bins. "
                    "Categories will be auto-generated.",
                    UserWarning
                )
                df['category'] = df['category'].astype(str)
    else:
        # Handle categorical data
        df['category'] = df[category_column].astype(str)
        if categories:
            missing_categories = set(categories) - set(df['category'].unique())
            if missing_categories:
                warnings.warn(
                    f"The following categories are not present in the data and "
                    f"will be ignored: {missing_categories}",
                    UserWarning
                )
            categories = [
                cat for cat in categories if cat in df['category'].unique()
            ]
            if not categories:
                raise ValueError(
                    "No valid categories to plot after filtering."
                )
        else:
            categories = sorted(df['category'].unique())

    # Filter the data if specified filter categories are provided
    if filter_categories:
        invalid_filters = set(filter_categories) - set(categories)
        if invalid_filters:
            warnings.warn(
                f"The following filter_categories are not in the available "
                f"categories and will be ignored: {invalid_filters}",
                UserWarning
            )
        filter_categories = [
            cat for cat in filter_categories if cat in categories
        ]
        if filter_categories:
            df = df[df['category'].isin(filter_categories)]
            categories = filter_categories  # Update categories to filtered ones
        else:
            raise ValueError(
                "No valid categories to filter after applying filter_categories."
            )

    # Plot the spatial distribution using selected plot type
    plt.figure(figsize=figsize)

    if plot_type == 'scatter':
        # Scatter plot using longitude, latitude as the axes
        sns.scatterplot(
            x='longitude',
            y='latitude',
            hue='category',
            data=df,
            palette=cmap,
            alpha=alpha
        )
    elif plot_type == 'hexbin':
        # Hexbin plot for large number of points
        hb = plt.hexbin(
            df['longitude'],
            df['latitude'],
            gridsize=50,
            cmap=cmap,
            mincnt=1
        )
        plt.colorbar(hb, label='Count')
    else:
        raise ValueError(
            f"Unsupported plot_type: {plot_type}"
        )

    # Labels and title
    plt.title(f"Spatial Distribution of {category_column}")
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    if plot_type == 'scatter':
        plt.legend(
            title='Categories',
            bbox_to_anchor=(1.05, 1),
            loc='upper left'
        )
 
    if not show_grid: 
        plt.grid(False)
        
    if axis_off:
        plt.axis('off')
    
    # Show plot
    plt.tight_layout()
    plt.show()

@default_params_plot(savefig='my_distribution_plot.png')
@validate_params ({ 
    'df': ['array-like'], 
    'x_col': [str], 
    'y_col': [str], 
    'z_cols': ['array-like', str], 
    'plot_type': [StrOptions({'scatter', 'hexbin', 'density'})], 
    'max_cols': [Real]
    })
@isdf 
def plot_dist(
    df: DataFrame,
    x_col: str,
    y_col: str,
    z_cols: List[str],
    plot_type: str = 'scatter',
    axis_off: bool = True,
    max_cols: int = 3,
    savefig=None,
):
    r"""
    Plot multiple distribution datasets on a grid of subplots for 
    comprehensive spatial analysis. This function generates a grid of 
    subplots illustrating the variation of multiple `z`-axis variables 
    (``z_cols``) against spatial coordinates defined by `x_col` and 
    `y_col`. Depending on the chosen `plot_type`, it can create scatter, 
    hexbin, or density plots. Users can visualize how each `z` variable 
    behaves over the spatial domain defined by `x` and `y`, allowing 
    intuitive comparisons and spatial pattern recognition.

    This object aims to project a set of `z` values as a function
    of `x` and `y`, thereby constructing a distribution surface. 
    Mathematically, for each variable :math:`z_i` in ``z_cols``, we 
    consider a mapping:

    .. math::
       z_i = f(x, y)

    where :math:`f: \mathbb{R}^2 \to \mathbb{R}`. The plotting depends 
    on the chosen `plot_type`:
    
    - ``'scatter'``: Directly plots points in the plane colored by their 
      corresponding `z` values.
    - ``'hexbin'``: Aggregates data into hexagonal bins and colors each 
      bin based on the average `z` value.
    - ``'density'``: Estimates a continuous density surface using kernel 
      density estimation, with `z` values acting as weights. This 
      representation assumes non-negative weights [1]_.

    The function arranges these plots in a grid with a maximum of 
    ``max_cols`` columns. If the number of `z_cols` exceeds this, 
    additional rows are added. An optional colorbar provides a unified 
    scale for interpreting the color encoding of `z` values across all 
    subplots.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the data to visualize. Must 
        include at least `x_col`, `y_col`, and all specified `z_cols`.
        The DataFrame should contain numeric values for these columns.
    x_col : str
        Name of the column representing the x-axis coordinate 
        (e.g., longitude). This parameter specifies the independent
        spatial dimension.
    y_col : str
        Name of the column representing the y-axis coordinate 
        (e.g., latitude). Combined with `x_col`, it forms a spatial 
        plane onto which `z` values are projected.
    z_cols : list of str
        A list of column names corresponding to the variables 
        to be visualized along the `z` dimension. Each `z_col` 
        represents a different distribution over the `(x, y)` space.
    plot_type : str, optional
        The type of plot to generate. Supported types are:
        
        - ``'scatter'``: Plots individual data points with colors 
          indicating `z` values.
        - ``'hexbin'``: Uses hexagonal binning to visualize data 
          density and average `z` within each bin.
        - ``'density'``: Creates a kernel density estimate (KDE) 
          surface weighted by `z` values. All `z` values must be 
          non-negative for this method.
        
        Defaults to ``'scatter'``.
    axis_off : bool, optional
        If True, removes axis lines and labels for a cleaner look. 
        Default is True.
    max_cols : int, optional
        The maximum number of columns in the subplot grid. If the 
        total number of `z_cols` exceeds `max_cols`, additional 
        rows are created automatically. Default is 3.
    savefig : str or None, optional
        If provided, specifies the filename or path where the 
        resulting figure should be saved. If None, the figure is 
        displayed interactively.

    Methods
    -------
    This object is a standalone function, therefore it does not 
    provide class methods. No additional callable methods are 
    exposed aside from the function itself. Users interact 
    solely through the function parameters described above.

    Notes
    -----
    - For `density` plot types, ensure no negative values are present 
      in the `z_cols`. Negative weights cause errors in kernel 
      density estimation.
    - Large datasets might benefit from `hexbin` or `density` plots 
      to better visualize overall patterns rather than individual 
      points.

    Examples
    --------
    >>> from gofast.plot.utils import plot_dist
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'longitude': [1,2,3,4],
    ...     'latitude' : [10,10,10,10],
    ...     'subsidence_2018': [5,6,7,8],
    ...     'subsidence_2022': [3,4,2,1]
    ... })
    >>> plot_dist(df, x_col='longitude', y_col='latitude',
    ...           z_cols=['subsidence_2018', 'subsidence_2022'],
    ...           plot_type='scatter', axis_off=True, max_cols=2)

    See Also
    --------
    matplotlib.pyplot.scatter : For basic scatter plot creation.
    matplotlib.pyplot.hexbin : For hexagonal binning visualization.
    seaborn.kdeplot : For kernel density estimation plots.
    gofast.plot.utils.plot_distributions: 
        For the distribution of numeric columns in the DataFrame.

    References
    ----------
    .. [1] Rosenblatt, M. "Remarks on some nonparametric estimates 
           of a density function." Ann. Math. Statist. 27 (1956), 
           832-837.
    """

    # Validate Input Columns
    exist_features(df, features= [x_col, y_col], name ='Columns `x` and `y`')
    z_cols = columns_manager(z_cols, empty_as_none= False, )
    exist_features(df, features=z_cols, name ='Value `z_cols` ')
    
    extra_msg = (
        "If a numeric feature is stored as an 'object' type, "
        "it should be explicitly converted to a numeric type"
        " (e.g., using `pd.to_numeric`)."
    )
    check_features_types(
        df, features= [x_col, y_col] + z_cols , dtype='numeric',
        extra=extra_msg
    )

    # If using 'density', ensure weights are non-negative
    if plot_type == 'density':
        for col in z_cols:
            if (df[col] < 0).any():
                raise ValueError(
                    f"Negative values found in '{col}'. Seaborn kdeplot cannot "
                    "handle negative weights. Please provide non-negative values "
                    "or choose a different plot_type."
                )
    max_cols = validate_positive_integer(max_cols, 'max_cols')
    
    # Determine Subplot Grid Layout
    num_z = len(z_cols)
    n_cols = min(max_cols, num_z)
    n_rows = math.ceil(num_z / max_cols)

    # Create Subplots
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        constrained_layout=True
    )

    # Ensure axes is a 2D array for consistent indexing
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    # Calculate Overall vmin and vmax for Color Normalization
    overall_min = df[z_cols].min().min()
    overall_max = df[z_cols].max().max()


    # Iterate Over z_cols and Plot
    for idx, z_col in enumerate(z_cols):
        row = idx // max_cols
        col = idx % max_cols
        ax = axes[row][col]

        if plot_type == 'scatter':
            # Create a scatter plot
            ax.scatter(
                df[x_col],
                df[y_col],
                c=df[z_col],
                cmap='viridis',
                s=10,
                alpha=0.7,
                norm=plt.Normalize(vmin=overall_min, vmax=overall_max)
            )
        elif plot_type == 'hexbin':
            # Create a hexbin plot
            ax.hexbin(
                df[x_col],
                df[y_col],
                C=df[z_col],
                gridsize=50,
                cmap='viridis',
                reduce_C_function=np.mean,
                norm=plt.Normalize(vmin=overall_min, vmax=overall_max)
            )
        elif plot_type == 'density':
            # Create a density (kde) plot
            # seaborn.kdeplot returns a QuadContourSet, not directly used afterward,
            # but that's fine. We don't need to assign it.
            sns.kdeplot(
                x=df[x_col],
                y=df[y_col],
                weights=df[z_col],
                fill=True,
                cmap='viridis',
                ax=ax,
                thresh=0
            )

        # Set plot title
        ax.set_title(f'{z_col}', fontsize=12)

        # Optionally turn off axes
        if axis_off:
            ax.axis('off')

    # Hide Any Unused Subplots
    total_plots = n_rows * n_cols
    if num_z < total_plots:
        for idx in range(num_z, total_plots):
            row = idx // max_cols
            col = idx % max_cols
            axes[row][col].axis('off')

    # Add a Single Colorbar
    if plot_type in ['scatter', 'hexbin', 'density']:
        # Create a ScalarMappable for the colorbar
        sm = plt.cm.ScalarMappable(
            cmap='viridis',
            norm=plt.Normalize(vmin=overall_min, vmax=overall_max)
        )
        sm.set_array([])  # Only needed for older Matplotlib versions

        # Flatten the axes array to pass a list of Axes to colorbar
        all_axes = [ax for row_axes in axes for ax in row_axes]

        # Add colorbar using all axes to properly position it
        cbar = fig.colorbar(
            sm,
            ax=all_axes,
            orientation='vertical',
            fraction=0.02,
            pad=0.04
        )
        cbar.set_label('Value', fontsize=12)


    plt.show()


@default_params_plot(savefig='my_q.distributions_plot.png')
@validate_params ({ 
    'df': ['array-like'], 
    'x_col': [str], 
    'y_col': [str], 
    'date_col': [ str], 
    'quantiles': ['array-like', None], 
    'date_values':['array-like', None], 
    'value_prefix': [str], 
    'q_cols': [dict, None], 
    })
@isdf 
def plot_quantile_distributions(
    df,
    x_col, # ='longitude'
    y_col, # ='latitude'
    date_col, # ='year'
    quantiles=None,
    date_values=None,
    value_prefix='', # 'predicted_subsidence'
    q_cols=None,
    cmap='viridis',
    s=10,
    alpha=0.8,
    cbar_orientation='vertical',
    cbar_fraction=0.02,
    cbar_pad=0.1,
    figsize=None,
    savefig=None,
    dpi=300,
    axis_off=True, 
    reverse=False, 
):
    r"""
    Plot quantile distributions across spatial and temporal domains,
    visualizing multiple `z`-value distributions defined by quantiles
    over specific date values. This function arranges the resulting 
    plots in a grid, with quantiles along one dimension and time/date 
    values along the other dimension, allowing users to investigate 
    how distributions change over space and time.

    Mathematically, given spatial coordinates :math:`(x,y)` and a set 
    of quantiles :math:`Q = \{q_1, q_2, ..., q_m\}` and date values 
    :math:`D = \{d_1, d_2, ..., d_n\}`, we consider a function 
    :math:`f(x,y,d,q)` that returns a value for each combination. The 
    columns of the DataFrame must represent these values. The function 
    arranges subplots in an :math:`m \times n` grid, where each cell 
    in the grid corresponds to a specific quantile `q` and a date `d`. 
    Each subplot displays:

    .. math::
       z_{q,d}(x,y) = f(x,y; q, d)

    where `z_{q,d}` is extracted from columns based on a naming 
    convention, or directly provided via ``q_cols``.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing spatial and value data. It must 
        include at least the columns corresponding to `<x_col>`, 
        `<y_col>`, `<date_col>`, and either:

        1) Columns named according to the convention 
           ``<value_prefix>_<date>_q<quantile>``, or
        2) Explicit mappings in `q_cols`.
    x_col : str
        The column name representing the x-axis coordinate 
        (e.g., longitude).
    y_col : str
        The column name representing the y-axis coordinate 
        (e.g., latitude).
    date_col : str
        The column representing dates. This can be integer (e.g., year),
        or datetime. If it's datetime, values are filtered by year 
        extracted from that datetime.
    quantiles : list of float, optional
        A list of quantiles (e.g., [0.1, 0.5, 0.9]). If None, the function
        attempts to detect quantiles from columns if ``q_cols`` is also 
        None. If detection fails, a ValueError is raised.
    date_values : list, optional
        List of date values to plot. If None, the function infers date 
        values from `df`. If `date_col` is integer, they are considered 
        as years. If `date_col` is datetime, the year part is extracted.
    value_prefix : str, optional
        The prefix used in column naming format:
        ``<value_prefix>_<date>_q<quantile>``. By default empty, but 
        often something like 'predicted_subsidence'.
    q_cols : dict or None, optional
        If provided, explicitly maps quantiles and dates to columns. 
        Two forms are supported:
        
        - ``{quantile: {date_str: column_name}}``
        - ``{quantile: [list_of_columns_in_same_order_as_dates]}``
        
        If None, the function uses the naming convention or detection.
    cmap : str, optional
        Colormap name used for color encoding values. Default is 'viridis'.
    s : int, optional
        Marker size for scatter plots.
    alpha : float, optional
        Marker transparency.
    cbar_orientation : str, optional
        Orientation of the colorbar. Default is 'vertical'.
    cbar_fraction : float, optional
        Fraction of original axes size occupied by colorbar.
    cbar_pad : float, optional
        Padding between colorbar and the edge of the plot.
    figsize : tuple, optional
        Figure size (width, height) in inches. If None, chosen 
        automatically based on the number of rows and columns.
    savefig : str or None, optional
        If provided, a path to save the figure as an image. If None, 
        displays the figure interactively.
    dpi : int, optional
        Dots-per-inch for the saved figure. Default is 300.
    axis_off : bool, optional
        If True, removes axis ticks and labels for cleaner visualization.
    reverse : bool, optional
        Controls the orientation of the quantile distribution plots.
        By default (`reverse=False`), quantiles are placed along the 
        rows and dates along the columns:

        .. math::
           \text{Rows} \rightarrow \text{Quantiles} \\
           \text{Columns} \rightarrow \text{Dates}

        This results in a grid where each row corresponds to a 
        quantile, and each column corresponds to a date value.

        When `reverse=True`, the layout is inverted so that quantiles 
        are arranged along the columns and dates along the rows:

        .. math::
           \text{Rows} \rightarrow \text{Dates} \\
           \text{Columns} \rightarrow \text{Quantiles}

        In this scenario, each column corresponds to a quantile and 
        each row corresponds to a date value. This can be useful when 
        you want to compare quantiles side-by-side horizontally rather 
        than vertically.

    Methods
    -------
    This object is a standalone function with no associated methods.
    Users interact only through its parameters.

    Notes
    -----
    - If `quantiles` is None and no suitable columns are found, a 
      ValueError is raised.
    - If `q_cols` is provided, it overrides automatic detection or 
      naming conventions.
    - The filtering step for datetime `date_col` extracts the year 
      component. For custom temporal resolutions, users might need 
      to preprocess the data.
    - Negative values in the data might be acceptable for scatter 
      and hexbin plots, but if a density approach is implemented 
      elsewhere, ensure non-negative weights [1]_.

    Examples
    --------
    Consider a DataFrame `df` with columns:
    'longitude', 'latitude', 'year', and 
    'predicted_subsidence_2024_q0.1', 
    'predicted_subsidence_2024_q0.5', 
    'predicted_subsidence_2024_q0.9', etc.
    
    Consider generating a sample DataFrame suitable for testing:

    >>> import pandas as pd
    >>> import numpy as np

    >>> num_points = 10
    >>> years = [2024, 2025]
    >>> quantiles = [0.1, 0.5, 0.9]

    >>> longitudes = np.random.uniform(100.0, 101.0, num_points)
    >>> latitudes  = np.random.uniform(20.0, 21.0,  num_points)

    >>> data = []
    >>> for year in years:
    ...     for i in range(num_points):
    ...         # Base subsidence value
    ...         base_val = np.random.uniform(0, 50)
    ...         row = {
    ...             'longitude': longitudes[i],
    ...             'latitude' : latitudes[i],
    ...             'year'     : year
    ...         }
    ...         # Add columns for each quantile
    ...         for q in quantiles:
    ...             q_col = f'predicted_subsidence_{year}_q{q}'
    ...             row[q_col] = base_val * (1 + np.random.uniform(-0.1,0.1))
    ...         data.append(row)

    >>> df = pd.DataFrame(data)
    >>> df.head()

    This `df` will have columns like:
    `'longitude', 'latitude', 'year', 
    'predicted_subsidence_2024_q0.1', 'predicted_subsidence_2024_q0.5', 
    'predicted_subsidence_2024_q0.9', 'predicted_subsidence_2025_q0.1', 
    'predicted_subsidence_2025_q0.5', 'predicted_subsidence_2025_q0.9'`.


    >>> from gofast.plot.utils import plot_quantile_distributions
    >>> # Automatically detect quantiles and dates:
    >>> plot_quantile_distributions(
    ...     df,
    ...     x_col='longitude',
    ...     y_col='latitude',
    ...     date_col='year',
    ...     value_prefix='predicted_subsidence'
    ... )

    Or specifying quantiles:
    >>> plot_quantile_distributions(
    ...     df,
    ...     x_col='longitude',
    ...     y_col='latitude',
    ...     date_col='year',
    ...     quantiles=[0.1, 0.5, 0.9],
    ...     value_prefix='predicted_subsidence'
    ... )

    If `q_cols` is provided:
    >>> q_map = {
    ...     0.1: ['sub2024_q0.1','sub2025_q0.1'],
    ...     0.5: ['sub2024_q0.5','sub2025_q0.5'],
    ...     0.9: ['sub2024_q0.9','sub2025_q0.9']
    ... }
    >>> plot_quantile_distributions(
    ...     df,
    ...     x_col='longitude',
    ...     y_col='latitude',
    ...     date_col='year',
    ...     quantiles=[0.1,0.5,0.9],
    ...     date_values=[2024,2025],
    ...     q_cols=q_map
    ... )

    See Also
    --------
    matplotlib.pyplot.scatter : Scatter plot generation.
    matplotlib.pyplot.hexbin : Hexbin plot generation.
    seaborn.kdeplot : For density visualization.

    References
    ----------
    .. [1] Rosenblatt, M. "Remarks on some nonparametric estimates 
           of a density function." Ann. Math. Statist. 27 (1956), 
           832-837.
    """

    # Infer date_values if not provided
    if date_values is None:
        if pd.api.types.is_integer_dtype(df[date_col]):
            date_values = sorted(df[date_col].unique())
        elif np.issubdtype(df[date_col].dtype, np.datetime64):
            date_values = pd.to_datetime(df[date_col].unique()).sort_values()
        else:
            date_values = sorted(df[date_col].unique())

    date_values_str = _extract_date_values_if_datetime(df, date_col, date_values)

    # If q_cols is None, we rely on quantiles and naming convention
    # or detect quantiles if quantiles is None.
    quantiles = columns_manager(quantiles, empty_as_none= True )
    if q_cols is None:
        # If quantiles is None, try to detect from columns
        if quantiles is None:
            # Attempt detection
            detected_quantiles = _detect_quantiles_from_columns(
                df, value_prefix, date_values_str)
            if detected_quantiles is None:
                raise ValueError(
                    "No quantiles detected from columns."
                    " Please specify quantiles or q_cols."
                    )
            quantiles = detected_quantiles

        # Now we have quantiles, build column names
        all_cols = _build_column_names(df, value_prefix, quantiles, date_values_str)
        if not all_cols:
            raise ValueError(
                "No matching columns found with given prefix, date, quantiles.")
    else:
        # q_cols provided. Let's assume q_cols is a dict:
        # {quantile: {date_str: column_name}}
        # or {quantile: [cols_in_same_order_as_date_values_str]}
        # We must extract quantiles from q_cols keys
        quantiles_from_qcols = sorted(q_cols.keys())
        if quantiles is None:
            quantiles = quantiles_from_qcols
        else:
            quantiles = validate_quantiles (quantiles )
            # Ensure that quantiles match q_cols keys
            if set(quantiles) != set(quantiles_from_qcols):
                raise ValueError("Quantiles specified do not match q_cols keys.")
        
        # Validate all columns exist
        all_cols = []
        for q in quantiles:
            mapping = q_cols[q]
            if isinstance(mapping, dict):
                # expect {date_str: col_name}
                for d_str in date_values_str:
                    if d_str not in mapping:
                        continue
                    c = mapping[d_str]
                    if c not in df.columns:
                        raise ValueError(f"Column {c} not found in DataFrame.")
                    all_cols.append(c)
            else:
                # assume list parallel to date_values_str
                if len(mapping) != len(date_values_str):
                    raise ValueError("q_cols mapping length does not match date_values.")
                for c in mapping:
                    if c not in df.columns:
                        raise ValueError(f"Column {c} not found in DataFrame.")
                all_cols.extend(mapping)

    # Compute overall min and max for color normalization
    if not all_cols:
        raise ValueError("No columns determined for plotting.")

    overall_min = df[all_cols].min().min()
    overall_max = df[all_cols].max().max()

    if reverse:
        _plot_reversed(
            df, x_col, y_col, date_col, 
            quantiles, date_values_str, 
            q_cols, value_prefix,
            cmap, s, alpha, axis_off,
            overall_min, overall_max,
            cbar_orientation, cbar_fraction, 
            cbar_pad, figsize, dpi, savefig
        )
        return 
     
    # Determine subplot grid size
    n_rows = len(quantiles)
    n_cols = len(date_values)

    if figsize is None:
        figsize = (5 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=figsize, constrained_layout=True
        )

    # Ensure axes is 2D array
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    for i, q in enumerate(quantiles):
        # Extract columns per quantile and date
        for j, d in enumerate(date_values_str):
            ax = axes[i, j]

            if q_cols is None:
                col_name = f'{value_prefix}_{d}_q{q}'
                if col_name not in df.columns:
                    ax.axis('off')
                    ax.set_title(f'No data for {d} (q={q})')
                    continue
                subset_col = col_name
            else:
                mapping = q_cols[q]
                if isinstance(mapping, dict):
                    if d not in mapping:
                        ax.axis('off')
                        ax.set_title(f'No col mapped for {d} (q={q})')
                        continue
                    subset_col = mapping[d]
                else:
                    # list scenario
                    # find index of d in date_values_str
                    idx_date = date_values_str.index(d)
                    subset_col = mapping[idx_date]

            subset = df[[x_col, y_col, date_col, subset_col]].dropna()

            # Filter subset based on date_col and d
            if np.issubdtype(df[date_col].dtype, np.datetime64):
                # convert d to int year
                year_int = int(d)
                subset = subset[subset[date_col].dt.year == year_int]
            else:
                # Attempt to convert d to int if possible
                try:
                    val = int(d)
                except ValueError:
                    val = d
                subset = subset[subset[date_col] == val]

            if subset.empty:
                ax.axis('off')
                ax.set_title(f'No data after filter for {d} (q={q})')
                continue

            scatter = ax.scatter(
                subset[x_col],
                subset[y_col],
                c=subset[subset_col],
                cmap=cmap,
                s=s,
                alpha=alpha,
                norm=plt.Normalize(vmin=overall_min, vmax=overall_max)
            )

            if np.issubdtype(df[date_col].dtype, np.datetime64):
                title_str = f"Quantile={q}, Year={d}"
            else:
                title_str = f"Quantile={q}, {date_col}={d}"
            ax.set_title(title_str, fontsize=10)

            if axis_off:
                ax.axis('off')

        # Add colorbar for the last subplot in each row
        cbar = fig.colorbar(
            scatter,
            ax=axes[i, -1],
            orientation=cbar_orientation,
            fraction=cbar_fraction,
            pad=cbar_pad
        )
        cbar.set_label('Value', fontsize=10)

    if savefig:
        fig.savefig(savefig, dpi=dpi)

    plt.show()

def _extract_date_values_if_datetime(df, date_col, date_values):
    """
    Extract date values as strings suitable for column naming if date_col 
    is datetime. If date_col is datetime, convert them to year strings, 
    otherwise return them as they are.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the date_col.
    date_col : str
        The name of the date column in df.
    date_values : list
        A list of date values inferred or provided.

    Returns
    -------
    list
        A list of string representations of the date values.
    """
    # Convert datetime if needed
    # For datetime, date_values are datetimes; might want to format them
    # as years or something else depending on the scenario
    # Extract date_values as strings if datetime
    
    if np.issubdtype(df[date_col].dtype, np.datetime64):
        # If datetime, convert each date to its year string
        return [str(pd.to_datetime(d).year) for d in date_values]
    else:
        # Otherwise just convert to string
        return [str(d) for d in date_values]

def _build_column_names(df, value_prefix, quantiles, date_values_str):
    """
    Build a list of column names based on a prefix, quantiles, 
    and date values (as strings), checking which exist in the DataFrame.

    Assumes the naming format: f'{value_prefix}_{date}_q{quantile}'

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to check for columns.
    value_prefix : str
        The prefix used in column naming.
    quantiles : list
        A list of quantiles.
    date_values_str : list
        A list of date values as strings.

    Returns
    -------
    list
        A list of column names that exist in df.
    """
    # Build column names dynamically or handle scenario where we must 
    # extract the subset for each date
    # Suppose the columns are of the form value_prefix_date_qquantile
    # If this is not the case, user must supply data differently or 
    # we can handle differently. For now, let's assume columns are named:
    # f"{value_prefix}_{date}_q{quantile}"
    # If date_col is datetime, we might extract year or something else
    
    all_cols = []
    for q in quantiles:
        for d_str in date_values_str:
            col_name = f'{value_prefix}_{d_str}_q{q}'
            if col_name in df.columns:
                all_cols.append(col_name)
    return all_cols

def _detect_quantiles_from_columns(df, value_prefix, date_values_str):
    """
    Attempt to detect quantiles by scanning the DataFrame columns that match
    the pattern f'{value_prefix}_{date}_q...' for each date in date_values_str.
    Extract the quantiles from these column names.

    Parameters
    ----------
    df : pandas.DataFrame
    value_prefix : str
        The value prefix expected in column names.
    date_values_str : list
        A list of date strings.

    Returns
    -------
    quantiles : list
        A list of detected quantiles (floats) sorted.
    """
    quantile_pattern = re.compile(r'_q([\d\.]+)$')
    found_quantiles = set()

    # Check columns that start with value_prefix and contain one of the dates
    # and end with q{quantile}
    for col in df.columns:
        # Check if col fits pattern: value_prefix_date_qquantile
        # First verify value_prefix and date are in it
        # This is simplistic: we check if any date is in col
        # and also if it matches the quantile pattern at the end.
        if col.startswith(value_prefix + "_"):
            # Extract the part after value_prefix_
            remainder = col[len(value_prefix)+1:]
            # remainder should look like date_qquantile
            # Check if it contains a known date_str
            for d_str in date_values_str:
                if remainder.startswith(d_str + "_q"):
                    # match qquantile
                    m = quantile_pattern.search(col)
                    if m:
                        q_val = float(m.group(1))
                        found_quantiles.add(q_val)
                    break

    if not found_quantiles:
        return None
    return sorted(found_quantiles)


def _plot_reversed(
    df, x_col, y_col, date_col,
    quantiles, date_values_str, q_cols, value_prefix,
    cmap, s, alpha, axis_off,
    overall_min, overall_max, cbar_orientation,
    cbar_fraction, cbar_pad,
    figsize, dpi, savefig
):
    # Reversed mode: quantiles in columns, dates in rows
    n_rows = len(date_values_str)
    n_cols = len(quantiles)
    if figsize is None:
        figsize = (5 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=figsize,
        constrained_layout=True
    )

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    for j, d in enumerate(date_values_str):
        for i, q in enumerate(quantiles):
            ax = axes[j, i]

            if q_cols is None:
                col_name = f'{value_prefix}_{d}_q{q}'
                if col_name not in df.columns:
                    ax.axis('off')
                    ax.set_title(f'No data for {d} (q={q})')
                    continue
                subset_col = col_name
            else:
                mapping = q_cols[q]
                if isinstance(mapping, dict):
                    if d not in mapping:
                        ax.axis('off')
                        ax.set_title(f'No col mapped for {d} (q={q})')
                        continue
                    subset_col = mapping[d]
                else:
                    idx_date = date_values_str.index(d)
                    subset_col = mapping[idx_date]

            subset = df[[x_col, y_col, date_col, subset_col]].dropna()

            # Filter subset based on date_col and d
            if np.issubdtype(df[date_col].dtype, np.datetime64):
                year_int = int(d)
                subset = subset[subset[date_col].dt.year == year_int]
            else:
                try:
                    val = int(d)
                except ValueError:
                    val = d
                subset = subset[subset[date_col] == val]

            if subset.empty:
                ax.axis('off')
                ax.set_title(f'No data after filter for {d} (q={q})')
                continue

            scatter = ax.scatter(
                subset[x_col],
                subset[y_col],
                c=subset[subset_col],
                cmap=cmap,
                s=s,
                alpha=alpha,
                norm=plt.Normalize(vmin=overall_min, vmax=overall_max)
            )

            if np.issubdtype(df[date_col].dtype, np.datetime64):
                title_str = f"Quantile={q}, Year={d}"
            else:
                title_str = f"Quantile={q}, {date_col}={d}"
            ax.set_title(title_str, fontsize=10)

            if axis_off:
                ax.axis('off')

        # Add colorbar for the last column in each row
        # Actually, we add one per row: better to add at the end of each row?
        # But here symmetrical to normal, we do once per row
        cbar = fig.colorbar(
            scatter,
            ax=axes[j, -1] if n_cols > 1 else axes[j, 0],
            orientation=cbar_orientation,
            fraction=cbar_fraction,
            pad=cbar_pad
        )
        cbar.set_label('Value', fontsize=10)

    if savefig:
        fig.savefig(savefig, dpi=dpi)
    plt.show()


@default_params_plot(
    savefig='my_uncertainty_plot.png', 
    title ="Distribution of Uncertainties",
    fig_size=None 
 )
@validate_params ({ 
    'df': ['array-like'], 
    'cols': ['array-like', None], 
    'plot_type': [StrOptions({ 'box', 'violin', 'strip', 'swarm'})], 
    'numeric_only': [bool], 
    })
@isdf 
@param_deprecated_message(
    conditions_params_mappings=[
        {
            'param': 'numeric_only',
            'condition': lambda v: v is False,
            'message': ( 
                "Current version only supports 'numeric_only=True'."
                " Resetting numeric_only to True. Note: this parameter"
                " should be removed in the future version."
                ),
            'default': True
        }
    ]
)
def plot_uncertainty(
    df,
    cols=None,
    plot_type='box',
    figsize=None,
    numeric_only=True,
    title=None,
    ylabel="Predicted Value",
    xlabel_rotation=45,
    palette='Set2',
    showfliers=True,
    grid=True,
    savefig=None,
    dpi=300
):
    r"""
    Plot uncertainty distributions from numeric columns in a DataFrame 
    using various plot types. This function helps visualize the spread 
    and variability (uncertainty) of predictions or measurements by 
    generating box plots, violin plots, strip plots, or swarm plots 
    for each selected column.

    Mathematically, given a set of features 
    :math:`X = \{x_1, x_2, \ldots, x_n\}`, each representing a 
    distribution of values, we aim to represent their statistical 
    properties. For box plots, for example, we often visualize: 
    median ( :math:`m` ), quartiles ( :math:`q_1, q_3` ), and 
    outliers [1]_. Such visualizations allow users to quickly 
    assess uncertainty and variability.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data to plot. If `<cols>` is None 
        and `<numeric_only>` is True, all numeric columns in `df` are 
        selected.
    cols : list of str or None, optional
        The columns from `df` to visualize. If None, numeric columns 
        are detected automatically if `<numeric_only>` is True. If 
        `<numeric_only>` is False and no columns are numeric, all 
        columns are chosen.
    plot_type : str, optional
        The type of plot to generate. Supported values:
        
        - ``'box'``: Box plot, showing quartiles and outliers.
        - ``'violin'``: Violin plot, showing the kernel density 
          estimate of the distribution.
        - ``'strip'``: Strip plot, plotting individual data points.
        - ``'swarm'``: Swarm plot, arranging points to avoid 
          overlapping.
        
        Defaults to ``'box'``.
    figsize : tuple or None, optional
        The size of the figure in inches (width, height). If None, 
        determined automatically based on the number of columns.
    numeric_only : bool, optional
        If True, only numeric columns are considered when `<cols>` is 
        None. This helps avoid errors if `df` contains non-numeric 
        data. Default is True.
    title : str, optional
        The title of the plot, displayed above the visualization.
    ylabel : str, optional
        The label for the y-axis. Defaults to "Predicted Value".
    xlabel_rotation : int or float, optional
        The rotation angle (in degrees) for the x-axis labels. 
        Default is 45 degrees.
    palette : str or sequence, optional
        The color palette used for the plot. See seaborn documentation 
        for available palettes.
    showfliers : bool, optional
        If True (for box plots), outliers are shown. For other plot 
        types, ignored. Default is True.
    grid : bool, optional
        If True, a grid is displayed (on the y-axis for typical 
        distributions). Default is True.
    savefig : str or None, optional
        If not None, the figure is saved to the specified path at the 
        given `<dpi>` resolution. If None, the figure is displayed 
        interactively.
    dpi : int, optional
        The resolution of the saved figure in dots per inch. Default 
        is 300.

    Methods
    -------
    This object is a standalone function and does not provide any 
    additional callable methods. Users interact solely through its 
    parameters.

    Notes
    -----
    - By selecting different `<plot_type>` values, users can choose 
      the representation that best suits their data. Box plots show 
      summary statistics, violin plots add density information, 
      strip and swarm plots display raw data points.
    - If `<cols>` is not provided and `<numeric_only>` is True, this 
      function tries to detect numeric columns automatically. If no 
      numeric columns are found, a ValueError is raised.

    Examples
    --------
    >>> from gofast.plot.utils import plot_uncertainty
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'A': np.random.normal(0,1,100),
    ...     'B': np.random.normal(5,2,100),
    ...     'C': np.random.uniform(-1,1,100)
    ... })
    >>> # Automatic numeric detection and box plot:
    >>> plot_uncertainty(df)
    >>> # Use a violin plot and specify columns:
    >>> plot_uncertainty(df, cols=['A','B'], plot_type='violin')

    See Also
    --------
    seaborn.boxplot : For box plots.
    seaborn.violinplot : For violin plots.
    seaborn.stripplot : For strip plots.
    seaborn.swarmplot : For swarm plots.

    References
    ----------
    .. [1] Tukey, J. W. "Exploratory Data Analysis." Addison-Wesley, 
           1977.

    """
    # If no cols provided, select numeric columns if numeric_only=True
    if cols is None:
        if numeric_only:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols = df.columns.tolist()
    cols = columns_manager(cols, empty_as_none= True )
    # If no numeric columns found and numeric_only=True
    if not cols:
        raise ValueError(
            "No columns to plot. Please provide columns or set numeric_only=False.")

    check_features_types(df, features=cols, dtype='numeric')
    # Drop NaN values or handle them
    plot_data = df[cols].dropna()

    # Automatic figsize
    if figsize is None:
        # Simple heuristic: width proportional to number of cols
        # height fixed at 6
        width = max(5, len(cols) * 1.2)
        figsize = (width, 6)

    plt.figure(figsize=figsize)

    # Plot according to type
    if plot_type == 'box':
        sns.boxplot(data=plot_data, showfliers=showfliers, palette=palette)
    elif plot_type == 'violin':
        sns.violinplot(data=plot_data, palette=palette)
    elif plot_type == 'strip':
        sns.stripplot(data=plot_data, palette=palette)
    elif plot_type == 'swarm':
        sns.swarmplot(data=plot_data, palette=palette)
   
    plt.title(title, fontsize=14)
    plt.ylabel(ylabel, fontsize=12)

    # Rotate x-labels if needed
    plt.xticks(rotation=xlabel_rotation)
    
    if grid:
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    if savefig:
        plt.savefig(savefig, dpi=dpi, bbox_inches='tight')

    plt.show()

@default_params_plot(
    savefig='my_prediction_intervals_plot.png', 
    title ="Prediction Intervals",
    fig_size=(10, 6) 
 )
@validate_params ({ 
    'df': ['array-like'], 
    'x': ['array-like', None], 
    'line_colors': ['array-like'], 
    'numeric_only': [bool], 
    })
def plot_prediction_intervals(
    df,
    median_col=None,
    lower_col=None,
    upper_col=None, 
    actual_col=None,
    x=None,
    title=None,
    xlabel="X",
    ylabel="Value",
    legend=True,
    line_colors=('black', 'blue'),
    fill_color='blue',
    fill_alpha=0.2,
    figsize=None,
    grid=True,
    savefig=None,
    dpi=300
):
    r"""
    Plot prediction intervals to visualize forecast uncertainty or 
    variation in predicted values. This function displays a median 
    prediction line, optionally actual observed values, and fills the 
    area between lower and upper quantiles (or prediction bounds) to 
    highlight uncertainty.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing columns for actual and predicted 
        values. Must at least have `<median_col>`, `<lower_col>` and 
        `<upper_col>`.
    median_col : str
        The name of the column in `df` representing the median 
        prediction. This is the central prediction line.
    lower_col : str
        The name of the column in `df` representing the lower bound 
        of the prediction interval.
    upper_col : str
        The name of the column in `df` representing the upper bound 
        of the prediction interval.
    actual_col : str or None, optional
        The name of the column in `df` representing actual observed 
        values. If None, no actual line is plotted.
    x : array-like or None, optional
        The x-coordinates for plotting. If None, the index of `df` is 
        used.
    title : str, optional
        The plot title, describing the scenario or data.
    xlabel: str, optional
        The label for the x-axis.
    ylabel : str, optional
        The label for the y-axis.
    legend : bool, optional
        If True, display a legend showing line and interval labels.
    line_colors : tuple, optional
        A tuple of colors (actual_line_color, median_line_color). 
        Defaults to ('black', 'blue').
    fill_color`: str, optional
        The color used for filling the interval region. Default is 
        'blue'.
    fill_alpha : float, optional
        The transparency level of the filled interval region. 
        Default is 0.2.
    figsize : tuple or None, optional
        The figure size in inches (width, height). If None, defaults 
        to (10, 6).
    grid: bool, optional
        If True, display a grid on the plot. Default is True.
    savefig : str or None, optional
        If provided, a file path to save the figure. If None, the 
        figure is displayed interactively.
    dpi: int, optional
        The resolution of the saved figure in dots-per-inch. 
        Default is 300.

    Methods
    -------
    This object is a standalone function without additional methods. 
    Users interact solely through the given parameters.

    Notes
    -----
    Given a series of predictions indexed by :math:`i`, let 
    :math:`m_i` be the median prediction and 
    :math:`\ell_i`, :math:`u_i` be the lower and upper bounds of the 
    prediction interval at index :math:`i`. The plot represents:

    .. math::
       \ell_i \leq m_i \leq u_i

    as a shaded region from :math:`\ell_i` to :math:`u_i`, with a line 
    at :math:`m_i`. If actual values :math:`a_i` are provided, they are 
    plotted as well to compare predictions against reality.
    
    By visualizing the median prediction along with the lower and 
    upper bounds, users gain insight into the uncertainty or 
    confidence intervals around their forecasts. If actual 
    observations are provided, comparing them against the median 
    and interval can highlight prediction accuracy or deviation.

    Examples
    --------
    >>> from gofast.plot.utils import plot_prediction_intervals
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...    'median_pred': [10, 12, 11, 13],
    ...    'lower_pred':  [9,  11, 10, 12],
    ...    'upper_pred':  [11, 13, 12, 14],
    ...    'actual':      [10.5, 11.5, 11.2, 12.8]
    ... })
    >>> # Plot intervals with actual:
    >>> plot_prediction_intervals(
    ...     df,
    ...     actual_col='actual',
    ...     median_col='median_pred',
    ...     lower_col='lower_pred',
    ...     upper_col='upper_pred',
    ...     xlabel='Time Step',
    ...     ylabel='Value',
    ...     title='Forecast Intervals'
    ... )

    See Also
    --------
    matplotlib.pyplot.fill_between : Used for filling intervals.
    matplotlib.pyplot.plot : For line plotting.

    References
    ----------
    .. [1] Gneiting, T., and Raftery, A. E. "Strictly Proper 
           Scoring Rules, Prediction, and Estimation." J. Amer. 
           Statist. Assoc. 102 (2007): 359–378.
    """
    is_frame(df, df_only=True, raise_exception =True, objname ='df')
    
    # If x is None, use the index
    if x is None:
        x = np.arange(len(df))
        
    check_consistent_length(df, x )
    
    # Validate columns
    if median_col is None or lower_col is None or upper_col is None:
        raise ValueError(
            "median_col, lower_col, and upper_col must be specified.")
    
    exist_features(df, features =[median_col, lower_col, upper_col])
    # Extract data
    median_vals = df[median_col].values
    lower_vals = df[lower_col].values
    upper_vals = df[upper_col].values

    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot actual if provided
    if actual_col is not None:
        exist_features(df, features=actual_col, name ='Actual feature column')
        plt.plot(x, df[actual_col].values, label=actual_col, color=line_colors[0])

    # Plot median line
    plt.plot(x, median_vals, label=median_col, color=line_colors[1])

    # Fill interval
    plt.fill_between(
        x, lower_vals, upper_vals,
        color=fill_color, alpha=fill_alpha,
        label=f'{lower_col} - {upper_col}'
    )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if grid:
        plt.grid(True, linestyle='--', alpha=0.5)

    if legend:
        plt.legend()

    if savefig:
        plt.savefig(savefig, dpi=dpi, bbox_inches='tight')

    plt.show()

@default_params_plot(
    savefig='my_temporal_trends_plot.png', 
    title ="Temporal Trends",
    fig_size=(8, 6) 
 )
@validate_params ({ 
    'df': ['array-like'], 
    'date_col':[str], 
    'value_cols': ['array-like', str], 
    'line_colors': ['array-like'], 
    'to_datetime': [StrOptions({'auto', 'Y','M','W','D','H','min','s'}), None], 
    })
def plot_temporal_trends(
    df,
    date_col,
    value_cols,
    agg_func='mean',
    freq=None,
    to_datetime=None,
    kind='line',
    figsize=None,
    color=None,
    marker='o',
    linestyle='-',
    grid=True,
    title=None,
    xlabel=None,
    ylabel=None,
    legend=True,
    xrotation=0,
    savefig=None,
    dpi=300,
    verbose=0
):
    r"""
    Plot temporal trends of aggregated values over time, allowing 
    flexible input data handling, aggregation, and optional conversion 
    of date columns to a datetime format. By grouping data by `<date_col>` 
    and applying an aggregation function `<agg_func>`, this function 
    derives time-based trends that can be visualized using different 
    plot styles.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing `<date_col>` and `<value_cols>`.
    date_col : str
        The name of the column in `df` representing the temporal axis.
        If `to_datetime` is provided, the column may be converted to 
        a datetime format.
    value_cols : str or list of str
        The column name(s) representing the values to aggregate and 
        plot over time. If a single string is provided, it is internally 
        converted to a list.
    agg_func : str or callable, optional
        The aggregation function applied to the grouped data. If a 
        string (e.g., 'mean'), it must correspond to a valid pandas 
        aggregation method. If callable, it should accept an array 
        and return a scalar.
    freq : None or str, optional
        Reserved for future use or advanced frequency adjustments. 
        Currently not implemented.
    to_datetime : {None, 'auto', 'Y','M','W','D','H','min','s'}, optional
        Controls conversion of `<date_col>` to datetime if not already 
        in datetime format:
        
        - None: No conversion, date_col used as-is.
        - 'auto': Attempt automatic detection and conversion via 
          `smart_ts_detector`.
        - Explicit codes (e.g. 'Y','M','W','min','s') instruct how 
          to interpret and convert numeric values into datetime 
          objects (e.g., integers as years, months, weeks).
    kind : {'line', 'bar'}, optional
        The type of plot:
        
        - 'line': A line plot connecting aggregated points over time.
        - 'bar': A bar plot showing discrete aggregated values.
    figsize : tuple, optional
        Figure size (width, height) in inches. Default is (10,6).
    color : color or sequence of colors, optional
        Color(s) for the plot lines or bars. If None, defaults 
        to matplotlib defaults.
    marker : str, optional
        Marker style for line plots. Default is 'o'.
    linestyle : str, optional
        Line style for line plots. Default is '-'.
    grid : bool, optional
        If True, display a grid. Default is True.
    title : str, optional
        The title of the plot. Default "Temporal Trends".
    xlabel : str or None, optional
        Label for the x-axis. If None, uses `<date_col>`.
    ylabel : str or None, optional
        Label for the y-axis. If None, defaults to "Value".
    legend : bool, optional
        If True, display a legend for multiple `<value_cols>`. 
        Default is True.
    xrotation : int or float, optional
        Rotation angle for x-axis tick labels. Default is 0 (no rotation).
    savefig : str or None, optional
        File path to save the figure. If None, displays interactively.
    dpi: int, optional
        Resolution of saved figure in dots-per-inch. Default is 300.
    verbose : int, optional
        Verbosity level for logging
 
    Returns
    -------
    None
        Displays the plot or saves it to the specified file.

    Notes
    -----
    Consider a DataFrame :math:`D` with a time-related column 
    :math:`d \in D` and value columns :math:`v \in D`. The goal is 
    to produce a plot:

    .. math::
       T(t) = \text{agg_func}(\{v_i | d_i = t\}),

    where `<agg_func>` (e.g. `mean`) is applied to subsets of 
    `<value_cols>` grouped by each unique time unit in `<date_col>`. 
    The resulting series :math:`T(t)` highlights how `<value_cols>` 
    evolve over time.
    
    If `<to_datetime>` is not None, `smart_ts_detector` may be used 
    internally to guess and convert `<date_col>` into a datetime 
    object. For example, if `<to_datetime>='Y'` and the column 
    contains integers like 2020, 2021, they are interpreted as years 
    and converted accordingly.

    If multiple `<value_cols>` are provided and `kind='line'`, each 
    column is plotted as a separate line. If `kind='bar'`, a grouped 
    bar plot is produced automatically using `DataFrame.plot`.

    Examples
    --------
    >>> from gofast.plot.utils import plot_temporal_trends
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'year': [2020,2021,2022,2023],
    ...     'subsidence': [5.4, 5.6, 5.9, 6.1]
    ... })
    >>> # Simple line plot by year
    >>> plot_temporal_trends(df, 'year', 'subsidence')
    >>> # Convert from year int to datetime year and plot
    >>> plot_temporal_trends(df, 'year', 'subsidence', to_datetime='Y')

    See Also
    --------
    pandas.DataFrame.plot : General plotting tool for DataFrames.
    gofast.core.array_manager.smart_ts_detector : 
        Helps infer and convert numeric columns to datetime.
    """
    is_frame(df, df_only=True, raise_exception =True, objname='df')
    
    # If to_datetime is specified, use smart_ts_detector to convert date_col
    if to_datetime is not None:
        # We will call smart_ts_detector with appropriate parameters to 
        # handle the conversion return_types='df' to get a converted df back
        if verbose >= 2:
            print(f"Converting {date_col!r} using smart_ts_detector"
                  f" with to_datetime={to_datetime}")
        df = smart_ts_detector(
            df=df,
            date_col=date_col,
            return_types='df',
            to_datetime=to_datetime,
            error='raise',  # Raise error if something goes wrong
            verbose=verbose
        )

    # Ensure value_cols is a list
    value_cols = columns_manager(value_cols, empty_as_none= False )
    # checks whether columns exist
    exist_features(df, features= value_cols, name ='Value columns')

    # Perform grouping by date_col and aggregation
    grouped = df.groupby(date_col)[value_cols]

    if isinstance(agg_func, str):
        # If a string is given, we assume it's a known aggregation function
        agg_data = grouped.agg(agg_func)
    else:
        # If agg_func is callable, use it directly
        agg_data = grouped.agg(agg_func)

    # Create figure and plot
    plt.figure(figsize=figsize)

    if kind == 'line':
        # Plot each value_col as a separate line
        for c in value_cols:
            plt.plot(
                agg_data.index, agg_data[c],
                marker=marker,
                linestyle=linestyle,
                color=color,
                label=c if legend else None
            )
    elif kind == 'bar':
        # Let pandas handle the bar plot
        agg_data.plot(
            kind='bar', ax=plt.gca(),
            legend=legend, color=color
        )
    else:
        # Fallback to line if unknown kind
        for c in value_cols:
            plt.plot(
                agg_data.index, agg_data[c],
                marker=marker,
                linestyle=linestyle,
                color=color,
                label=c if legend else None
            )

    # Set plot titles and labels
    plt.title(title)
    plt.xlabel(xlabel if xlabel else date_col)
    plt.ylabel(ylabel if ylabel else "Value")
    plt.xticks(rotation=xrotation)

    # Add grid if requested
    if grid:
        plt.grid(True, linestyle='--', alpha=0.7)

    # Add legend if needed
    if legend and kind != 'bar':
        plt.legend()

    # Save figure if a path is provided
    if savefig:
        plt.savefig(savefig, dpi=dpi, bbox_inches='tight')

    # Show the plot
    plt.show()

