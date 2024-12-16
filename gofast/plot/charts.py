# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
The `charts` module provides functions for creating various types of charts. 
It includes tools for plotting pie charts and creating radar 
charts to visually represent data in an informative way.
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from ..api.types import ArrayLike, DataFrame 
from ..api.types import List, Tuple, Optional, Union 
from ..core.checks import is_iterable, exist_features 
from ..utils.validator import is_frame 

__all__=[
    "pie_charts", "radar_chart", "radar_chart_in", "donut_chart"
    ]

def donut_chart(
    data,
    values,
    labels=None,
    aggfunc='sum',
    groupby=None,
    colors=None,
    title=None,
    figsize=(8, 8),
    textprops=None,
    wedgeprops=None,
    explode=None,
    startangle=90,
    counterclock=True,
    pctdistance=0.85,
    labeldistance=1.05,
    inner_radius=0.70,
    outer_radius=1.0,
    legend=True,
    legend_loc='best',
    legend_title=None,
    autopct='%1.1f%%',
    **kwargs
):
    """
    Plot a donut chart from a DataFrame.

    This function creates a donut chart, which is a variation
    of a pie chart with a hollow center. It allows for flexible
    customization of the chart's appearance and supports data
    aggregation through grouping and aggregation functions.

    Parameters
    ----------
    data : pandas.DataFrame
        The input DataFrame containing the data to plot. It must
        include the specified `values` column and optionally
        `labels` and `groupby` columns.

    values : str
        The column name in `data` to use for the values of the
        chart. This column must contain numerical data.

    labels : str or list of str, optional
        The column name(s) in `data` to use for the labels of
        the chart. If `None`, labels are generated from the
        `groupby` columns or the DataFrame index.

    aggfunc : str or callable, default ``'sum'``
        The aggregation function to apply to the `values` column
        if `groupby` is specified. It can be a string such as
        ``'sum'``, ``'mean'``, or a callable function.

    groupby : str or list of str, optional
        Column(s) in `data` to group by before applying the
        aggregation function. If `None`, the data is not grouped.

    colors : list of color, optional
        A list of colors to use for the chart. If not provided,
        the default Matplotlib color cycle is used.

    title : str, optional
        The title of the chart. If `None`, no title is displayed.

    figsize : tuple of float, default ``(8, 8)``
        The size of the figure in inches, as a tuple
        ``(width, height)``.

    textprops : dict, optional
        A dictionary of text properties for the labels. This is
        passed to the `textprops` parameter of
        `matplotlib.pyplot.pie`.

    wedgeprops : dict, optional
        A dictionary of properties for the wedges. This is passed
        to the `wedgeprops` parameter of `matplotlib.pyplot.pie`.

    explode : list of float, optional
        A list of fractions to offset each wedge. This is used to
        "explode" wedges from the center of the chart.

    startangle : float, default ``90``
        The starting angle of the chart in degrees. The default
        ``90`` degrees starts the chart from the top.

    counterclock : bool, default ``True``
        If ``True``, the chart is plotted counterclockwise. If
        ``False``, it is plotted clockwise.

    pctdistance : float, default ``0.85``
        The radial distance at which the numeric labels are drawn,
        relative to the center of the chart.

    labeldistance : float, default ``1.05``
        The radial distance at which the labels are drawn,
        relative to the center of the chart.

    inner_radius : float, default ``0.70``
        The radius of the inner hole of the donut chart, as a
        fraction of the total chart radius.

    outer_radius : float, default ``1.0``
        The radius of the outer edge of the donut chart, as a
        fraction of the total chart radius.

    legend : bool, default ``True``
        If ``True``, a legend is displayed. If ``False``, no
        legend is displayed.

    legend_loc : str, default ``'best'``
        The location of the legend. Valid locations are strings
        such as ``'upper right'``, ``'lower left'``, etc.

    legend_title : str, optional
        The title of the legend. If `None`, no title is displayed.

    autopct : str or callable, optional
        A string or function used to label the wedges with their
        numeric value. If `None`, no numeric labels are displayed.

    **kwargs
        Additional keyword arguments passed to
        `matplotlib.pyplot.pie`.

    Returns
    -------
    None
        The function displays the plot and does not return any
        value.

    Notes
    -----
    The donut chart is a variation of the pie chart, with a hole
    in the center. The size of each wedge is proportional to the
    sum of the `values` in each group, calculated as:

    .. math::

        S_i = \\text{aggfunc}(V_i)

    where :math:`S_i` is the size of the i-th wedge,
    :math:`V_i` is the set of values in the i-th group, and
    :math:`\\text{aggfunc}` is the aggregation function applied.

    The chart is plotted using `matplotlib.pyplot.pie` [1]_ with
    the `wedgeprops` parameter adjusted to create the hole in
    the center.

    Examples
    --------
    >>> from gofast.plot.charts import donut_chart
    >>> import pandas as pd
    >>> # Sample data
    >>> data = pd.DataFrame({
    ...     'year': [2018, 2019, 2020, 2021],
    ...     'rainfall': [800, 950, 700, 850],
    ...     'region': ['North', 'South', 'East', 'West']
    ... })
    >>> # Plot average rainfall per year
    >>> donut_chart(
    ...     data=data,
    ...     values='rainfall',
    ...     labels='year',
    ...     title='Average Rainfall per Year'
    ... )
    >>> # Plot total rainfall per region with custom colors
    >>> donut_chart(
    ...     data=data,
    ...     values='rainfall',
    ...     labels='region',
    ...     colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'],
    ...     title='Total Rainfall by Region',
    ...     aggfunc='sum'
    ... )

    See Also
    --------
    matplotlib.pyplot.pie : Plot a pie chart.
    matplotlib.patches.Wedge : Wedge patch object.

    References
    ----------
    .. [1] Hunter, J. D. (2007). Matplotlib: A 2D graphics
       environment. *Computing in Science & Engineering*, 9(3),
       90-95.

    """
    is_frame (data, df_only=True, raise_exception= True)

    # Aggregate data if groupby is specified
    if groupby is not None:
        # Ensure groupby is a list
        if isinstance(groupby, str):
            groupby = [groupby]
        grouped_data = (
            data
            .groupby(groupby)[values]
            .agg(aggfunc)
            .reset_index()
        )
    else:
        grouped_data = data.copy()

    # Determine labels
    if labels is not None:
        exist_features(data, features= labels, name="Labels")
        if isinstance(labels, str):
            labels = grouped_data[labels].astype(str)
        elif isinstance(labels, list):
            labels = (
                grouped_data[labels]
                .astype(str)
                .agg(' - '.join, axis=1)
            )
        else:
            raise ValueError("labels must be a string or list of strings.")
            
    elif groupby is not None:
        labels = (
            grouped_data[groupby]
            .astype(str)
            .agg(' - '.join, axis=1)
        )
    else:
        labels = grouped_data.index.astype(str)

    # Extract values to plot
    plot_values = grouped_data[values]

    # Use default colors if not specified
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        if len(plot_values) > len(colors):
            colors = colors * (len(plot_values) // len(colors) + 1)
        colors = colors[:len(plot_values)]
    else:
        if len(colors) < len(plot_values):
            raise ValueError(
                "Not enough colors provided for the number of slices."
            )

    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Adjust wedgeprops for donut hole
    if wedgeprops is None:
        wedgeprops = {}
    wedgeprops.setdefault('width', outer_radius - inner_radius)

    # Plot the donut chart
    wedges, texts, autotexts = ax.pie(
        plot_values,
        labels=labels,
        colors=colors,
        startangle=startangle,
        counterclock=counterclock,
        explode=explode,
        autopct=autopct,
        pctdistance=pctdistance,
        labeldistance=labeldistance,
        textprops=textprops,
        wedgeprops=wedgeprops,
        **kwargs
    )

    # Set aspect ratio to be equal
    ax.axis('equal')

    # Set title if specified
    if title is not None:
        ax.set_title(title)

    # Add legend if required
    if legend:
        if legend_title is not None:
            ax.legend(
                wedges,
                labels,
                title=legend_title,
                loc=legend_loc
            )
        else:
            ax.legend(
                wedges,
                labels,
                loc=legend_loc
            )

    # Display the plot
    plt.show()


def pie_charts(
    data: DataFrame, /, 
    columns: Optional[Union[str, List[str]]] = None,
    bin_numerical: bool = True,
    num_bins: int = 4,
    handle_missing: str = 'exclude',
    explode: Optional[Union[Tuple[float, ...], str]] = None,
    shadow: bool = True,
    startangle: int = 90,
    cmap: str = 'viridis',
    autopct: str = '%1.1f%%',
    verbose: int = 0
):
    """
    Plots pie charts for categorical and numerical columns in a DataFrame.
    
    Function automatically detects and appropriately treats each type.
    Numerical columns can be binned into categories.
    
    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame containing the data.
        
    columns : str or list of str, optional
        Specific columns to plot. If `None`, all columns are plotted.
        
    bin_numerical : bool, optional
        If `True`, numerical columns will be binned into categories before
        plotting. Default is `True`.
        
    num_bins : int, optional
        Number of bins to use for numerical columns if `bin_numerical` is 
        `True`. Default is 4.
        
    handle_missing : {'exclude', 'include'}, optional
        How to handle missing values in data. 'exclude' will ignore them,
        while 'include' will treat them as a separate category. Default is 
        'exclude'.
        
    explode : tuple of float, or str, optional
        If not `None`, each value in the tuple indicates how far each wedge 
        is separated from the center of the pie. Can also be a single string
        'all' to apply the same explode value to all wedges. Default is `None`.
        
    shadow : bool, optional
        Draw a shadow beneath the pie chart. Default is `True`.
        
    startangle : int, optional
        Starting angle of the pie chart. Default is 90 degrees.
        
    cmap : str, optional
        Colormap for the pie chart. Default is 'viridis'.
        
    autopct : str, optional
        String used to label the wedges with their numeric value. Default is 
        '%1.1f%%'.
        
    verbose : int, optional
        Verbosity level. Higher values increase the amount of informational 
        output. Default is 0.
        
    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.plot.charts import pie_charts
    >>> df = pd.DataFrame({
    ...     'Category': ['A', 'B', 'A', 'C', 'B', 'A', 'D', 'D'],
    ...     'Values': [1, 2, 3, 4, 5, 6, 7, 8]
    ... })
    >>> pie_charts(df, bin_numerical=True, num_bins=3)
    
    Notes
    -----
    This function helps visualize categorical distributions of data in a 
    DataFrame using pie charts. For numerical columns, data can be binned 
    into a specified number of categories to facilitate categorical plotting.
    
    The mathematical formulation for the binning of numerical data is:
    
    .. math::
        bins = \frac{\max(x) - \min(x)}{n}
    
    where :math:`x` represents the numerical data and :math:`n` is the number
    of bins.
    
    See Also
    --------
    pandas.DataFrame.plot : Basic plotting functionality for DataFrames.
    
    References
    ----------
    .. [1] Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. 
       Computing in Science & Engineering, 9(3), 90-95.
    """

    is_frame (data, df_only=True, raise_exception= True)
    if columns is None:
        columns = data.columns.tolist()
    columns = is_iterable(columns, exclude_string= True, transform=True )
    
    valid_columns = [col for col in columns if col in data.columns]
    n_cols = len(valid_columns)
    rows, cols = (n_cols // 4 + (n_cols % 4 > 0), min(n_cols, 4)
                  ) if n_cols > 1 else (1, 1)
    fig, axs = plt.subplots(rows, cols, figsize=(cols*6, rows*5))
    axs = np.array(axs).flatten()

    for idx, column in enumerate(valid_columns):
        column_data = data[column]
        if column_data.dtype.kind in 'iufc' and bin_numerical:  # Numerical data
            column_data = _handle_numerical_data(column_data, bin_numerical, num_bins)
        column_data = _handle_missing_data(column_data, handle_missing)
        labels, sizes = _compute_sizes_and_labels(column_data)

        _plot_pie_chart(axs[idx], labels, sizes, explode, shadow, startangle, 
                        cmap, autopct, f'{column} Distribution')

    plt.tight_layout()
    plt.show()
    
def _handle_numerical_data(column_data: pd.Series, bin_numerical: bool,
                           num_bins: int) -> pd.Series:
    """Bins numerical data into categories if specified."""
    if bin_numerical:
        return pd.cut(column_data, bins=num_bins, include_lowest=True)
    return column_data

def _handle_missing_data(
        column_data: pd.Series, handle_missing: str  ) -> pd.Series:
    """Handles missing data based on user preference."""
    if handle_missing == 'exclude':
        return column_data.dropna()
    return column_data.fillna('Missing')

def _compute_sizes_and_labels(
        column_data: pd.Series) -> Tuple[List[str], List[float]]:
    """Computes the sizes and labels for the pie chart."""
    sizes = column_data.value_counts(normalize=True)
    labels = sizes.index.astype(str).tolist()
    sizes = sizes.values.tolist()
    return labels, sizes

def _plot_pie_chart(
        ax, labels: List[str], sizes: List[float], 
        explode: Optional[Union[Tuple[float, ...], str]], shadow: bool, 
        startangle: int, cmap: str, autopct: str, title: str):
    """Plots a single pie chart on the given axis."""
    # Adjust explode based on the number of labels if 'auto' is selected
    if explode == 'auto':
        explode = [0.1 if i == sizes.index(max(sizes)) else 0 for i in range(len(labels))]
    elif explode is None:
        explode = (0,) * len(labels)
    else:
        # Ensure explode is correctly sized for the current pie chart
        if len(explode) != len(labels):
            raise ValueError(f"The length of 'explode' ({len(explode)}) does "
                             f"not match the number of categories ({len(labels)}).")
    
    ax.pie(sizes, explode=explode, labels=labels, autopct=autopct,
           shadow=shadow, startangle=startangle, colors=plt.get_cmap(cmap)(
               np.linspace(0, 1, len(labels))))
    ax.set_title(title)
    ax.axis('equal')  # Ensures the pie chart is drawn as a circle

def radar_chart(
    d: ArrayLike, /, categories: List[str], 
    cluster_labels: List[str], 
    title: str = "Radar plot Umatrix cluster properties",
    figsize: Tuple[int, int] = (6, 6), 
    color_map: Union[str, List[str]] = 'Set2', 
    alpha_fill: float = 0.25, 
    linestyle: str = 'solid', 
    linewidth: int = 1,
    yticks: Tuple[float, ...] = (0.5, 1, 1.5), 
    ytick_labels: Union[None, List[str]] = None,
    ylim: Tuple[float, float] = (0, 2),
    legend_loc: str = 'upper right'
   ) -> None:
    """
    Create a radar chart with one axis per variable.

    Parameters
    ----------
    d : array-like
        2D array with shape (n_clusters, n_variables), where each row 
        represents a different cluster and each column represents a 
        different variable.
        
    categories : list of str
        List of variable names corresponding to the columns in the data.
        
    cluster_labels : list of str
        List of labels for the different clusters.
        
    title : str, optional
        The title of the radar chart. Default is "Radar plot Umatrix cluster 
        properties".
        
    figsize : tuple, optional
        The size of the figure to plot (width, height in inches). Default is 
        (6, 6).
        
    color_map : str or list, optional
        Colormap or list of colors for the different clusters. Default is 
        'Set2'.
        
    alpha_fill : float, optional
        Alpha value for the filled area under the plot. Default is 0.25.
        
    linestyle : str, optional
        The style of the line in the plot. Default is 'solid'.
        
    linewidth : int, optional
        The width of the lines. Default is 1.
        
    yticks : tuple, optional
        Tuple containing the y-ticks values. Default is (0.5, 1, 1.5).
        
    ytick_labels : list of str, optional
        List of labels for the y-ticks, must match the length of `yticks`. 
        If `None`, `yticks` will be used as labels. Default is `None`.
        
    ylim : tuple, optional
        Tuple containing the min and max values for the y-axis. Default is 
        (0, 2).
        
    legend_loc : str, optional
        The location of the legend. Default is 'upper right'.

    Returns
    -------
    fig : Figure
        The matplotlib `Figure` object for the radar chart.
        
    ax : Axes
        The matplotlib `Axes` object for the radar chart.

    Examples
    --------
    >>> import numpy as np 
    >>> from gofast.plot.charts import radar_chart
    >>> num_clusters = 5
    >>> num_vars = 10
    >>> data = np.random.rand(num_clusters, num_vars)
    >>> categories = [f"Variable {i}" for i in range(num_vars)]
    >>> cluster_labels = [f"Cluster {i}" for i in range(num_clusters)]
    >>> radar_chart(data, categories, cluster_labels)
    
    Notes
    -----
    This function creates a radar chart (or spider chart) to visualize 
    multivariate data. Each variable has its own axis, and data is plotted 
    radially. The chart helps compare the profiles of different clusters 
    across multiple variables.

    The data should be arranged in a 2D array where each row represents a 
    cluster and each column represents a variable. The angles for the axes 
    are computed using:

    .. math::
        \\theta_i = \\frac{2 \\pi i}{n}
    
    where :math:`i` is the index of the category and :math:`n` is the total 
    number of categories.

    See Also
    --------
    pandas.DataFrame.plot : Basic plotting functionality for DataFrames.
    
    References
    ----------
    .. [1] Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. 
       Computing in Science & Engineering, 9(3), 90-95.
    """
    
    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, len(categories),
                         endpoint=False).tolist()
    # The plot is made in a circular (not polygon) space, so we need to 
    # "complete the loop"and append the start to the end.
    angles += angles[:1]  # complete the loop
    d= np.array(d)
    d = np.concatenate((d, d[:,[0]]), axis=1)
    # Initialize the radar chart
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    plt.title(title, y=1.08)

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], categories)

    # Draw ylabels
    ax.set_rlabel_position(0)
    if ytick_labels is None:
        ytick_labels = [str(ytick) for ytick in yticks]
    plt.yticks(yticks, ytick_labels, color="grey", size=7)
    plt.ylim(*ylim)

    # Plot data and fill with color
    for idx, (row, label) in enumerate(zip(d, cluster_labels)):
        color = plt.get_cmap(color_map)(idx / len(d))
        ax.plot(angles, row, color=color, linewidth=linewidth,
                linestyle=linestyle, label=label)
        ax.fill(angles, row, color=color, alpha=alpha_fill)

    # Add a legend
    plt.legend(loc=legend_loc, bbox_to_anchor=(0.1, 0.1))

    plt.show()
    return fig, ax

def radar_chart_in(
    d: ArrayLike, /, 
    categories: List[str], 
    cluster_labels: List[str], 
    title: str = "Radar plot Umatrix cluster properties"
):
    """
    Create a radar chart with one axis per variable.

    Parameters
    ----------
    d : array-like
        2D array with shape (n_clusters, n_variables), where each row 
        represents a different cluster and each column represents a 
        different variable.
        
    categories : list of str
        List of variable names corresponding to the columns in the data.
        
    cluster_labels : list of str
        List of labels for the different clusters.
        
    title : str, optional
        The title of the radar chart. Default is "Radar plot Umatrix cluster 
        properties".

    Returns
    -------
    fig : Figure
        The matplotlib `Figure` object for the radar chart.
        
    ax : Axes
        The matplotlib `Axes` object for the radar chart.
        
    Examples
    --------
    >>> import numpy as np 
    >>> from gofast.plot.charts import radar_chart_in
    >>> num_clusters = 5
    >>> num_vars = 10
    >>> data = np.random.rand(num_clusters, num_vars)
    >>> categories = [f"Variable {i}" for i in range(num_vars)]
    >>> cluster_labels = [f"Cluster {i}" for i in range(num_clusters)]
    >>> radar_chart_in(data, categories, cluster_labels)
    
    Notes
    -----
    This function creates a radar chart (or spider chart) to visualize 
    multivariate data. Each variable has its own axis, and data is plotted 
    radially. The chart helps compare the profiles of different clusters 
    across multiple variables.

    The data should be arranged in a 2D array where each row represents a 
    cluster and each column represents a variable. The angles for the axes 
    are computed using:

    .. math::
        \\theta_i = \\frac{2 \\pi i}{n}
    
    where :math:`i` is the index of the category and :math:`n` is the total 
    number of categories.

    See Also
    --------
    pandas.DataFrame.plot : Basic plotting functionality for DataFrames.
    
    References
    ----------
    .. [1] Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. 
       Computing in Science & Engineering, 9(3), 90-95.
    """
    
    # Number of variables we're plotting.
    num_vars = len(categories)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # The plot is made in a circular (not polygon) space, so we need to 
    # "complete the loop"and append the start to the end.
    angles += angles[:1]
    d = np.array(d)
    d = np.concatenate((d, d[:,[0]]), axis=1)

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Draw one axe per variable and add labels
    plt.xticks(angles[:-1], categories)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.5, 1, 1.5], ["0.5", "1", "1.5"], color="grey", size=7)
    plt.ylim(0, 2)

    # Plot data
    for i in range(len(d)):
        ax.plot(angles, d[i], linewidth=1, linestyle='solid', 
                label=cluster_labels[i])

    # Fill area
    ax.fill(angles, d[0], color='blue', alpha=0.25)

    # Add a legend and title
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title(title)

    plt.show()
    
    return fig, ax