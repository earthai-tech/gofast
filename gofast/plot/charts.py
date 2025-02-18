# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
The `charts` module provides functions for creating various types of charts. 
It includes tools for plotting pie charts and creating radar 
charts to visually represent data in an informative way.
"""

import math
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from ..api.types import ArrayLike, DataFrame 
from ..api.types import List, Tuple, Optional, Union 
from ..core.checks import is_iterable, exist_features 
from ..core.handlers import extend_values
from ..core.io import is_data_readable, to_frame_if 
from ..decorators import Dataify 
from ..utils.validator import is_frame 
from .utils import make_plot_colors

__all__=[
    "pie_charts", "radar_chart", "radar_chart_in", "donut_chart", 
    "plot_donut_charts"
    ]

@is_data_readable
@Dataify(
    auto_columns=True, 
    prefix="chart_", 
    fail_silently=True 
)
def plot_donut_charts(
    data,
    max_cols=3,
    wedge_width=0.3,
    startangle=0,
    colors=None,
    labels=None,
    label_format=None,
    line_kw=None,
    box_kw=None,
    text_kw=None,
    mode="basic",
    fig_size=(10, 6),
    textprops=None,
    wedgeprops=None,
    explode=None,
    counterclock=True,
    pctdistance=0.85,
    labeldistance=1.05,
    inner_radius=0.70,
    outer_radius=1.0,
    legend=True,
    legend_loc='center',
    legend_title=None,
    autopct='%1.1f%%',
    seed=None, 
    **kw
):
    r"""
    Create one or multiple donut charts from the given
    ``data``, providing either a basic representation or
    an advanced ("expert") mode with extended annotations.

    The donut can be expressed mathematically in terms of
    pie wedges, each wedge :math:`w_i` having radius
    :math:`r_{outer}`, and an inner cutout radius
    :math:`r_{inner}`, forming a ring:

    .. math::
       \text{Area}(w_i) = \pi (r_{outer}^2 - r_{inner}^2)
       \times \frac{\theta_i}{2 \pi}

    where :math:`\theta_i` is the angular extent of wedge
    :math:`w_i`.

    Parameters
    ----------
    data : array-like, pandas.DataFrame, or pandas.Series
        The data used to construct each chart. If multiple
        columns are detected in a DataFrame, one donut chart
        is created per column.
    max_cols : int, optional
        Maximum number of columns in the subplot grid.
        Defaults to 3.
    wedge_width : float, optional
        Thickness of the donut ring if <mode> is ``"basic"``.
        Defaults to 0.3.
    startangle : float, optional
        Starting angle for the first wedge, in degrees.
        Defaults to 0.
    colors: list or None, optional
        List of color specifications or None for automatic
        color generation. Defaults to None.
    labels : list of str, optional
        Explicit labels for each wedge if desired. Typically
        drawn from <data> itself in standard usage.
    label_format : str, optional
        A format string for advanced labeling in "expert"
        mode, e.g. ``"{label}: {value}"``. Defaults to None.
    line_kw : dict, optional
        Dictionary of arrow or line style arguments for
        advanced labeling in "expert" mode (e.g.
        ``{"arrowstyle": "-"}``). Defaults to None.
    box_kw : dict, optional
        Dictionary of bounding box style arguments for
        advanced labeling in "expert" mode, e.g.
        ``{"boxstyle": "square,pad=0.3"}``. Defaults to None.
    text_kw : dict, optional
        Dictionary of text style arguments for advanced
        labeling, e.g. ``{"va": "center"}``. Defaults to None.
    mode : {"basic", "pro", "expert"}, optional
        Donut chart mode. ``"basic"`` is a quick ring around
        each pie wedge. ``"expert"`` allows for variable
        outer radius and line-based annotations. Defaults to
        "basic".
    fig_size : tuple of float, optional
        Size of the figure, in inches. Defaults to ``(10, 6)``.
    textprops : dict, optional
        Dictionary of text properties passed to
        :func:`matplotlib.axes.Axes.pie` for wedge labels
        in "basic" mode. Defaults to None.
    wedgeprops : dict, optional
        Dictionary of wedge properties passed to
        :func:`matplotlib.axes.Axes.pie`, controlling wedge
        styling. Defaults to None.
    explode : array-like, optional
        Offsets each wedge from the center by the specified
        fraction. Defaults to None.
    counterclock : bool, optional
        Whether wedges are drawn counterclockwise.
        Defaults to True.
    pctdistance : float, optional
        The ratio of the wedge radius at which the numeric
        percentage labels are drawn in "basic" mode.
        Defaults to 0.85.
    labeldistance : float, optional
        The radial distance at which wedge labels are drawn.
        Defaults to 1.05.
    inner_radius : float, optional
        Inner cutout radius if <mode> is ``"expert"``.
        Defaults to 0.70.
    outer_radius : float, optional
        Outer radius if <mode> is ``"expert"``. Defaults to 1.0.
    legend : bool, optional
        If True, a legend is displayed. Defaults to True.
    legend_loc : str, optional
        Location of the legend. Defaults to 'center'.
    legend_title : str, optional
        Title of the legend if displayed. Defaults to None.
    autopct : str, optional
        Format string for numeric labels in "basic" mode,
        e.g. ``"%1.1f%%"``. Defaults to ``"%1.1f%%"``.
    seed : int, optional
        Random seed for consistent color generation, if
        <colors> is None. Defaults to None.

    Examples
    --------
    >>> from gofast.plot.charts import plot_donut_charts
    >>> import pandas as pd
    >>> data = pd.DataFrame({"Apples": [10, 20, 30],
    ...                      "Oranges": [15, 5, 40]})
    >>> plot_donut_charts(data, mode="basic", fig_size=(8, 4))

    >>> # Expert mode with external lines
    >>> plot_donut_charts(data, mode="expert",
    ...                   line_kw={"arrowstyle": "->"},
    ...                   box_kw={"fc": "white", "ec": "black"},
    ...                   prefix="Fruit", legend=True)

    >>> # Example data (multiple columns => multiple donuts)
    >>> df = pd.DataFrame({
    ...    "Chart1": {"China": 2899, "US": 55, "France": 152, "Germany": 114},
    ...   "Chart2": {"Brazil": 79, "Argentina": 241, "Canada": 289, "Russia": 266},
    ...    "Chart3": {"Japan": 374, "India": 297, "Mexico": 124, "Italy": 79},
    ...    "Chart4": {"South Korea": 434, "South Africa": 291, "Spain": 197}
    ... })

    >>> plot_donut_charts(
    ...    data=df,
    ...    max_cols=2,
    ...    fig_size=(12, 8),
    ...    wedge_width=0.3,
    ...    startangle=140, 
    ...    mode='expert', 
    ...    explode=0.05, 
    ...    colors="cs4", 
    ... )
    
    Notes
    -----
    This function relies on
    `make_plot_colors` and `extend_values` for color and
    explode handling. By default, it creates as many donut
    subplots as columns in <data>. "Expert" mode uses an
    adjustable :math:`r_{inner}` and :math:`r_{outer}` for
    a more flexible ring display [1]_.

    See Also
    --------
    gofast.plot.charts.donut_chart:
        A simpler version for single-donut plotting
        without subplots or advanced features.

    References
    ----------
    .. [1] Wilkinson, L. (2005). "The Grammar of Graphics"
       (2nd ed.). Springer.
    """

    # Convert Series/list to DataFrame for uniform handling
    if isinstance(data, (pd.Series, np.ndarray, list)):
        data = pd.DataFrame(data)

    n_plots = data.shape[1]
    n_rows = math.ceil(n_plots / max_cols)

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=max_cols,
        figsize=fig_size
    )
    axes = np.atleast_1d(axes).flatten()

    # Defaults for arrow lines, annotation box, etc.
    if line_kw is None:
        line_kw = dict(arrowstyle="-", lw=1, color="black")
    if box_kw is None:
        box_kw = dict(boxstyle="square,pad=0.3",
                      fc="white", ec="black", lw=0.72)
    if text_kw is None:
        text_kw = dict(va="center", ha="center")

    # Generate or use provided colors
    colors = make_plot_colors(data, axis=1, colors=colors, seed=seed)
    
    for i in range(n_plots):
        ax = axes[i]
        ax.set_aspect("equal")
        col_data = data.iloc[:, i].dropna()
        slice_labels = col_data.index.tolist()
        values = col_data.values

        if explode is not None:
            explode = extend_values(explode, target= values)
        
        # BASIC MODE: standard donut with autopct, labeldistance, etc.
        if mode == "basic":
            if wedgeprops is None:
                wedgeprops = dict(width=wedge_width, edgecolor="white")
            wedges, texts, autotexts = ax.pie(
                values,
                labels=slice_labels,
                colors=colors,
                startangle=startangle,
                counterclock=counterclock,
                explode=explode,
                autopct=autopct,
                pctdistance=pctdistance,
                labeldistance=labeldistance,
                textprops=textprops,
                wedgeprops=wedgeprops
            )
            # Optionally add a legend
            if legend:
                if legend_title:
                    ax.legend(
                        wedges,
                        slice_labels,
                        title=legend_title,
                        loc=legend_loc
                    )
                else:
                    ax.legend(
                        wedges,
                        slice_labels,
                        loc=legend_loc
                    )

        # PRO / EXPERT MODE: advanced approach with external lines
        else:
            # Decide wedgeprops
            if wedgeprops is None:
                # In expert mode, allow variable radius
                if mode == "expert":
                    wedgeprops = dict(
                        width=(outer_radius - inner_radius),
                        edgecolor="white"
                    )
                else:
                    wedgeprops = dict(
                        width=wedge_width, edgecolor="white"
                        )

            # In expert mode, specify 'radius' for bigger outer circle
            if mode == "expert":
                radius_val = outer_radius
            else:
                radius_val = 1.0  # default radius

            wedges, _ = ax.pie(
                values,
                radius=radius_val,
                wedgeprops=wedgeprops,
                startangle=startangle,
                labels=None,  # we'll annotate manually
                colors=colors,
                counterclock=counterclock,
                explode=explode
            )

            # Manual labeling
            if label_format is None:
                label_format = "{label}: {value}"
            for w, lbl, val in zip(wedges, slice_labels, values):
                # Midpoint angle of the wedge
                angle = (w.theta2 - w.theta1) / 2.0 + w.theta1
                x = math.cos(math.radians(angle))
                y = math.sin(math.radians(angle))
                label_text = label_format.format(label=lbl, value=val)

                # Annotate with line
                ax.annotate(
                    label_text,
                    xy=(x * radius_val, y * radius_val),
                    xytext=(1.3 * np.sign(x) * radius_val, 1.4 * y * radius_val),
                    horizontalalignment="left" if x > 0 else "right",
                    arrowprops=line_kw,
                    bbox=box_kw,
                    **text_kw
                )

            # Legend if needed
            if legend:
                if legend_title:
                    ax.legend(
                        wedges,
                        slice_labels,
                        title=legend_title,
                        loc=legend_loc
                    )
                else:
                    ax.legend(
                        wedges,
                        slice_labels,
                        loc=legend_loc
                    )

        # Title
        ax.set_title(data.columns[i])

    # Hide unused axes
    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()

@is_data_readable
def donut_chart(
    data,
    value,
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
        include the specified `value` column and optionally
        `labels` and `groupby` columns.

    value : str
        The column name in `data` to use for the values of the
        chart. This column must contain numerical data.

    labels : str or list of str, optional
        The column name(s) in `data` to use for the labels of
        the chart. If `None`, labels are generated from the
        `groupby` columns or the DataFrame index.

    aggfunc : str or callable, default ``'sum'``
        The aggregation function to apply to the `value` column
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
    ...     value='rainfall',
    ...     labels='year',
    ...     title='Average Rainfall per Year'
    ... )
    >>> # Plot total rainfall per region with custom colors
    >>> donut_chart(
    ...     data=data,
    ...     value='rainfall',
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
    data = to_frame_if (data, df_only=True )
    # is_frame (data, df_only=True, raise_exception= True)

    # Aggregate data if groupby is specified
    if groupby is not None:
        # Ensure groupby is a list
        if isinstance(groupby, str):
            groupby = [groupby]
        grouped_data = (
            data
            .groupby(groupby)[value]
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
    plot_values = grouped_data[value]

    # Use default colors if not specified
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        if len(plot_values) > len(colors):
            colors = colors * (len(plot_values) // len(colors) + 1)
        colors = colors[:len(plot_values)]
    else:
        colors = make_plot_colors(plot_values, colors = colors, seed=42 )
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

    if explode is not None:
        explode = extend_values(explode, target= plot_values)
        
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

@is_data_readable
def pie_charts(
    data: DataFrame, 
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
    d: ArrayLike, 
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