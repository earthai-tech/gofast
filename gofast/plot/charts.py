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
import matplotlib.path as mpath
import matplotlib.patches as mpatches

from ..api.types import ArrayLike, DataFrame 
from ..api.types import List, Tuple, Optional, Union 
from ..core.checks import ( 
    is_iterable, 
    exist_features, 
)
from ..core.handlers import extend_values, columns_manager 
from ..core.io import is_data_readable, to_frame_if 
from ..decorators import Dataify 
from ..utils.validator import ( 
    is_frame, 
    parameter_validator, 
    check_donut_inputs
)
from .utils import make_plot_colors

__all__=[
    "pie_charts", "radar_chart", "radar_chart_in", "donut_chart", 
    "plot_donut_charts", "donut_chart_in", "chord_diagram", 
    "multi_level_donut", "two_ring_donuts"
    ]


@is_data_readable 
@Dataify(fail_silently=True)
def chord_diagram(
    data,
    group_names=None,
    colors=None,
    start_angle=0,
    gap=2,
    chord_alpha=0.7,
    chord_colors=None,
    min_value=None,
    show_group_labels=True,
    show_tick_labels=True,
    label_fontsize=10,
    ticks_fontsize=8,
    title=None,
    arc_width=0.1,
    pad_angle=0.02,
    transparency=0.6,
    sort_groups=False,
    ax=None,
    cmap=plt.cm.get_cmap("tab20"),
    fig_size=(8, 8)
):
    """
    Plot a chord diagram from an NxN pandas DataFrame, where df[i, j]
    indicates the flow (or relationship strength) from group i to j.

    Parameters
    ----------
    df : pandas.DataFrame
        N×N matrix of flow magnitudes. Rows are "source" groups,
        columns are "target" groups.
    group_names : list of str, optional
        Names for each group (length N). If None, uses df.index if
        available.    
    colors : list of str or None
        List of colors for each group (length N). If None, the function
        uses the provided colormap (cmap).
    start_angle : float, default 0
        Starting angle in degrees for the first group’s arc.
    gap : float, default 2
        Gap in degrees between group arcs.
    chord_alpha : float, default 0.7
        Alpha (transparency) for the ribbons (flows).
    chord_colors : dict or None
        If you want specific chord (i->j) colors, provide a dict
        {(i, j): color, ...}. Otherwise, chord colors are derived
        from the group colors or a default scheme.
    min_value : float or None
        Threshold below which flows are ignored (no ribbon drawn).
        Useful to hide very small flows that clutter the diagram.
    show_group_labels : bool, default True
        If True, draw group labels (text) around the circle arcs.
    show_tick_labels : bool, default True
        If True, draw numeric ticks on the arcs indicating the
        cumulative flow magnitude.
    label_fontsize : int, default 10
        Font size for group labels.
    ticks_fontsize : int, default 8
        Font size for numeric tick labels.
    title : str, optional
        Title for the chord diagram.
    arc_width : float, default 0.1
        Relative thickness of the group arcs (0.1 means 10% of radius).
    pad_angle : float, default 0.02
        Padding in radians between sub-arc segments when drawing the
        ribbon for a single group. Helps avoid gaps/overlaps in ribbons.
    transparency : float, default 0.6
        General alpha for group arcs if you want them partially
        transparent. Overridden if you provide direct color alpha.
    sort_groups : bool, default False
        If True, sort groups by their total outflow (row sum).
    ax : matplotlib.axes.Axes, optional
        Axes on which to draw the chord diagram. If None, a new figure
        and axes are created.
    cmap : matplotlib.colors.Colormap, default plt.cm.get_cmap("tab20")
        Colormap used if explicit group colors are not provided.
    fig_size : tuple of float, default (8, 8)
        Size of the figure if ax is None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes containing the chord diagram.

    Notes
    -----
    - The arcs around the circle represent each group’s total
      magnitude (sum of row or column, which should match if df is
      symmetrical).
    - Each flow from i to j is drawn as a ribbon connecting the arc
      of group i to the arc of group j. The area (or thickness) of
      that ribbon is proportional to df[i, j].
    - The function tries to space arcs and ribbons to avoid overlaps.
      However, chord diagrams become cluttered for large N.

    Examples
    --------
    Basic usage with an NxN DataFrame:

    >>> from gofast.plot.charts import chord_diagram
    >>> import pandas as pd
    >>> df = pd.DataFrame([
    ...     [0, 5, 2],
    ...     [5, 0, 3],
    ...     [2, 3, 0]
    ... ], columns=["A", "B", "C"], index=["A", "B", "C"])
    >>> fig, ax = chord_diagram(df, title="Simple Chord")

    Hide small flows, set a threshold of 1:

    >>> fig, ax = chord_diagram(df, min_value=1)

    Use explicit group colors:

    >>> group_cols = ["red", "green", "blue"]
    >>> fig, ax = chord_diagram(df, colors=group_cols)

    Formulation
    ------------
    Let :math:`A` be the NxN matrix, with
    :math:`A_{ij}` the flow from group :math:`i` to group
    :math:`j`. We define the total outflow of group
    :math:`i` as

    .. math::
       O_i = \\sum_{j=1}^{N} A_{ij}

    The group arcs are placed around a unit circle, each spanning
    an angle proportional to :math:`O_i`. For each pair
    :math:`(i, j)`, a ribbon is drawn from the sub-arc of :math:`i`
    to the sub-arc of :math:`j`, occupying an angular extent
    proportional to :math:`A_{ij}` within :math:`O_i` and :math:`O_j`.

    See Also
    --------
    - :func:`matplotlib.pyplot.pie` : Basic pie chart
    - :func:`matplotlib.patches.Arc` : Arc patch
    - :func:`matplotlib.path.Path` : Used to draw ribbon shapes

    References
    ----------
    .. [1] Krzywinski, M., et al. "Circos: an information aesthetic
       for comparative genomics." Genome research 19.9 (2009): 1639-1645.
    .. [2] "Chord diagram." *Wikipedia*.
    """

    # 1) Convert data to DataFrame if it's not already
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    mat = data.values
    n = mat.shape[0]

    # 2) Optionally sort groups by descending row sum
    if sort_groups:
        row_sums = mat.sum(axis=1)
        idx_sort = np.argsort(-row_sums)
        mat = mat[idx_sort][:, idx_sort]
        if group_names is None:
            group_names = data.index[idx_sort].tolist()
        else:
            group_names = [group_names[i] for i in idx_sort]
    else:
        if group_names is None:
            group_names = data.index.tolist()

    # 3) Assign colors
    if colors is None:
        # build from colormap
        colors = []
        for i in range(n):
            c = cmap(float(i) / max(1, (n - 1)))
            # add alpha
            c = (c[0], c[1], c[2], transparency)
            colors.append(c)
    else:
        # assume colors is list of length n
        colors = [
            (*plt.colors.to_rgba(c), transparency) for c in colors
        ]

    # 4) Build figure if ax not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig = ax.figure

    ax.set_aspect("equal")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.axis("off")

    # 5) Compute total outflow for each group and overall sum
    row_sum = mat.sum(axis=1)
    total_flow = row_sum.sum()
    deg_per_flow = 360.0 / total_flow

    # group angles
    group_angles = []
    current_angle = start_angle

    # Each group i occupies an arc of size row_sum[i] * deg_per_flow
    for i in range(n):
        size_deg = row_sum[i] * deg_per_flow
        start_deg = current_angle
        end_deg = current_angle + size_deg
        group_angles.append((start_deg, end_deg))
        current_angle = end_deg + gap  # gap between groups

    # 6) Draw group arcs
    for i in range(n):
        start_deg, end_deg = group_angles[i]
        theta1, theta2 = min(start_deg, end_deg), max(start_deg, end_deg)
        arc = plt.matplotlib.patches.Wedge(
            center=(0, 0),
            r=1.0,
            theta1=theta1,
            theta2=theta2,
            width=arc_width,
            facecolor=colors[i],
            edgecolor="white",
            alpha=1.0
        )
        ax.add_patch(arc)

        # optional group label
        if show_group_labels:
            mid_angle = (theta1 + theta2) / 2.0
            rad = 1.0 - arc_width * 0.5
            x_lab = rad * math.cos(math.radians(mid_angle))
            y_lab = rad * math.sin(math.radians(mid_angle))
            ax.text(
                x_lab,
                y_lab,
                str(group_names[i]),
                ha="center",
                va="center",
                rotation=mid_angle - 90 if mid_angle > 180 else mid_angle + 90,
                fontsize=label_fontsize
            )

        # optional tick labels
        if show_tick_labels:
            step = row_sum[i] / 4.0
            for s in np.arange(step, row_sum[i] + step, step):
                if s > row_sum[i]:
                    break
                frac = s / row_sum[i]
                angle_here = theta1 + (theta2 - theta1) * frac
                rad_tick = 1.0
                xt = rad_tick * math.cos(math.radians(angle_here))
                yt = rad_tick * math.sin(math.radians(angle_here))
                ax.text(
                    xt,
                    yt,
                    f"{s:.0f}",
                    ha="center",
                    va="center",
                    fontsize=ticks_fontsize,
                    color="black"
                )

    # Helper to get sub-arc size for flow
    def sub_arc(i, flow):
        a_start, a_end = group_angles[i]
        size_deg = a_end - a_start
        frac = flow / row_sum[i]
        return size_deg * frac

    arc_offsets = [0.0] * n
    arc_offsets_j = [0.0] * n

    # 7) Draw ribbons
    for i in range(n):
        for j in range(n):
            val_ij = mat[i, j]
            if min_value is not None and val_ij < min_value:
                continue
            if val_ij <= 0:
                continue

            # compute arc sub-size
            sub_size_i = sub_arc(i, val_ij)
            a_i_start, a_i_end = group_angles[i]
            arc_start_i = a_i_start + arc_offsets[i]
            arc_end_i = arc_start_i + sub_size_i
            arc_offsets[i] += sub_size_i + pad_angle * deg_per_flow

            sub_size_j = sub_arc(j, val_ij)
            a_j_start, a_j_end = group_angles[j]
            arc_start_j = a_j_start + arc_offsets_j[j]
            arc_end_j = arc_start_j + sub_size_j
            arc_offsets_j[j] += sub_size_j + pad_angle * deg_per_flow

            # chord color
            if chord_colors and (i, j) in chord_colors:
                c_chord = chord_colors[(i, j)]
            else:
                c_chord = colors[i]

            # create a path from sub arc i to sub arc j
            r_inner = 1.0 - arc_width
            xi1 = r_inner * math.cos(math.radians(arc_start_i))
            yi1 = r_inner * math.sin(math.radians(arc_start_i))
            xi2 = r_inner * math.cos(math.radians(arc_end_i))
            yi2 = r_inner * math.sin(math.radians(arc_end_i))

            xj1 = r_inner * math.cos(math.radians(arc_start_j))
            yj1 = r_inner * math.sin(math.radians(arc_start_j))
            xj2 = r_inner * math.cos(math.radians(arc_end_j))
            yj2 = r_inner * math.sin(math.radians(arc_end_j))

            verts = [
                (xi1, yi1),
                (0, 0),
                (xj1, yj1),
                (xj2, yj2),
                (0, 0),
                (xi2, yi2),
                (xi1, yi1)
            ]
            codes = [
                mpath.Path.MOVETO,
                mpath.Path.CURVE3,
                mpath.Path.CURVE3,
                mpath.Path.LINETO,
                mpath.Path.CURVE3,
                mpath.Path.CURVE3,
                mpath.Path.CLOSEPOLY
            ]

            path = mpath.Path(verts, codes)
            patch = mpatches.PathPatch(
                path,
                facecolor=c_chord,
                edgecolor="none",
                alpha=chord_alpha
            )
            ax.add_patch(patch)

    if title:
        ax.set_title(title, fontsize=12)

    return fig, ax

@is_data_readable(data_to_read="data")
@Dataify(
    auto_columns=True, 
    prefix="chart_", 
    fail_silently=True, 
    start_incr_at=1
)
def plot_donut_charts(
    data=None,
    values=None, 
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
    connectionstyle="angle3", 
    value_on_slice=True,
    value_color='white',
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
        Also it corresponds to the data source from which 
        to extract ``values`` and ``labels``. If provided, 
        and if ``values`` or ``labels`` is a double backtick
        string``, the corresponding column is used.
    values : array-like or ``str``, optional
        Numeric values for the donut slices. If ``data`` is 
        provided and ``values`` is a string`` (e.g. ``"Sales"``), 
        then that column is used for numeric data. Otherwise, 
        a numeric array of values.

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
    connectionstyle : ``str``, default ``"angle3"``
        Defines the style of the leader line connection. For example,
        a value of ``"angle3"`` produces a two-segment connector.
    value_on_slice : bool, default True
        If True, the numeric value for each slice is drawn inside the
        donut slice (using the color specified by ``value_color``). If
        False, the value is included in the external label.
    value_color : ``str``, default ``"white"``
        Color used for the numeric values drawn inside the slices when
        ``value_on_slice`` is True.
    seed : int, optional
        Random seed for consistent color generation, if
        <colors> is None. Defaults to None.

    Examples
    --------
    >>> from gofast.datasets._globals import COUNTRY_REGION 
    >>> from gofast.plot.charts import plot_donut_charts
    >>> import pandas as pd
    >>> import numpy as np 
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
    
    >>> # Given total software downloads
    >>> total_downloads = 181512

    >>> # Generate random proportions for each country
    >>> num_countries = len(COUNTRY_REGION)
    >>> random_proportions = np.random.rand(num_countries)
    >>> random_proportions /= random_proportions.sum()  # Normalize to sum to 1

    >>> # Assign proportional downloads
    >>> download_distribution = (random_proportions * total_downloads).astype(int)

    >>> # Create a Series
    >>> download_series = pd.Series(
    ...    download_distribution, 
    ...    index=COUNTRY_REGION.keys(), 
    ...    name="downloads"
    ...    )
    >>> plot_donut_charts(
    ...    download_series,
    ...    fig_size=(10, 8),
    ...    wedge_width=0.3,
    ...    startangle=90, 
    ...    mode='expert', 
    ...    explode=0.05, 
    ...    colors="xkcd:7", 
    ...    legend=False, 
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
    # if isinstance(data, (pd.Series, np.ndarray, list)):
    #     data = pd.DataFrame(data)
    
    data = check_donut_inputs(
        data=data, values=values, 
        labels=labels, 
        ops="build", 
    )

    n_plots = data.shape[1]
    n_rows = math.ceil(n_plots / max_cols)
    
    mode = parameter_validator(
        "mode", target_strs={"basic", "pro", "expert"}, 
        error_msg=( 
            f"Invalid mode {mode!r}. Select one of 'basic',"
            "'pro' or 'expert'."))(mode)
    
    if n_rows==1: 
        # Create figure and axis
        fig, axes = plt.subplots(figsize=fig_size)
    else: 
        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=max_cols,
            figsize=fig_size, 
            # constrained_layout=True if n_rows==1 else False
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
        # text_kw = {'fontsize': 8, 'ha': 'center', 'va': 'center'}
    for i in range(n_plots):
        if n_rows==1: 
            ax = axes 
        else:
            ax = axes[i]
        ax.set_aspect("equal")
        col_data = data.iloc[:, i].dropna()
        slice_labels = col_data.index.tolist()
        values = col_data.values
        
        # Generate or use provided colors
        colors = make_plot_colors(
            values, colors=colors, seed=seed)

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
            
        # PRO: advanced approach with external lines
        elif mode == "pro":
            # Decide wedgeprops if not provided
            if wedgeprops is None:
                wedgeprops = dict(width=wedge_width, edgecolor="white")
        
            # Draw the donut chart once
            wedges, _ = ax.pie(
                values,
                wedgeprops=wedgeprops,
                startangle=startangle,
                labels=None,  # We'll add labels manually
                colors=colors,
                counterclock=counterclock,
                explode=explode
            )
            # If wedge_width < 1.0, add a white circle to create the donut "hole"
            # (inner_radius can be used if you want a specific hole size)
            if wedge_width < 1.0:
                # Example: center circle with radius = outer_radius - wedge_width
                # or just use inner_radius as a parameter if you prefer
                circle_radius = inner_radius
                centre_circle = plt.Circle((0, 0), circle_radius, fc="white")
                ax.add_artist(centre_circle)
        
            # Build default bounding-box (annotation) props from box_kw
            bbox_props = box_kw.copy()
            bbox_props.setdefault("boxstyle", "square,pad=0.3")
            bbox_props.setdefault("fc", "white")
            bbox_props.setdefault("ec", "gray")
            bbox_props.setdefault("lw", 0.5)
        
            # Build default arrow (line) props from line_kw
            arrow_kwargs = line_kw.copy()
            arrow_kwargs.setdefault("arrowstyle", "-")
            arrow_kwargs.setdefault("color", "gray")
            arrow_kwargs.setdefault("lw", 1)
        
            # Manually place labels with leader lines
            for i, wedge in enumerate(wedges):
                # Angle bisector in degrees
                angle = (wedge.theta2 - wedge.theta1) / 2.0 + wedge.theta1
                # Convert to radians
                rad = math.radians(angle)
        
                # (x, y) on the unit circle
                x = math.cos(rad)
                y = math.sin(rad)
        
                # Decide left or right alignment
                horizontal_alignment = "left" if x > 0 else "right"
        
                # Move the label outside the donut:
                # 1.2 is a "padding" factor so labels don't collide with the wedge
                label_x = 1.2 * x
                label_y = 1.2 * y
        
                # Annotate
                ax.annotate(
                    slice_labels[i],
                    xy=(x, y),  # arrow starts at wedge boundary
                    xytext=(label_x, label_y),  # label location
                    ha=horizontal_alignment,    # horizontal alignment
                    arrowprops=arrow_kwargs,
                    bbox=bbox_props
                )
        
            # Optionally adjust the plot limits so labels are not clipped
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
        
            # Optional legend
            if legend:
                if legend_title:
                    ax.legend(
                        wedges, slice_labels, 
                        title=legend_title, 
                        loc=legend_loc
                    )
                else:
                    ax.legend(wedges, slice_labels, loc=legend_loc)

        else:  # "expert"
            # Decide wedgeprops for a thicker donut ring in expert mode
             # 1) Wedge properties (donut) and dynamic explode
            if wedgeprops is None:
                wedgeprops = dict(
                    width=(outer_radius - inner_radius),
                    edgecolor="white"
                )
            # If user did not supply an explode array, 
            # create one based on slice size
            if explode is None:
                total_val = float(sum(values))
                base_explode = 0.02     # base push-out for all slices
                scale_explode = 0.25    # how strongly bigger slices get pushed out
                explode = []
                for val in values:
                    frac = val / total_val
                    explode.append(base_explode + scale_explode * frac)
            else:
                # Use the helper function extend_values 
                # if the user did provide explode
                explode = extend_values(explode, target=values)
        
            # 2) Draw the donut
            wedges, _ = ax.pie(
                values,
                startangle=startangle,
                counterclock=counterclock,
                explode=explode,
                colors=colors,
                labels=None,  # place labels manually
                wedgeprops=wedgeprops
            )
            # If wedge_width < 1.0, add a white circle 
            # to create the donut hole
            if wedge_width < 1.0:
                radius_hole = inner_radius
                circle = plt.Circle((0, 0), radius_hole, fc="white")
                ax.add_artist(circle)
        
            # 3) Build default arrow (line) and box props,
            # using connectionstyle
            if line_kw is None:
                line_kw = {}
            arrow_kwargs = dict(
                arrowstyle=line_kw.get("arrowstyle", "-"),
                lw=line_kw.get("lw", 1),
                color=line_kw.get("color", "black"),
                # The user can set connectionstyle="angle3" or something else
                connectionstyle=connectionstyle
            )
            if box_kw is None:
                box_kw = dict(
                    boxstyle="square,pad=0.3",
                    fc="white",
                    ec="gray",
                    lw=0.6
                )
        
            if text_kw is None:
                text_kw = dict(va="center", ha="center")
        
            # By default, if value_on_slice is False, show "[Label: Value]" outside.
            # If True, show "[Label]" outside and put value inside the wedge.
            if label_format is None:
                if value_on_slice:
                    label_format = "{label}"  # Only label outside
                else:
                    label_format = "{label}: {value}"

        
            def format_label(lbl, val):
                return label_format.format(label=lbl, value=val)
        

            # 4) Gather wedge angles, split left vs. right
            wedge_info = []
            for w, lbl, val in zip(wedges, slice_labels, values):
                # Middle angle of wedge
                angle_deg = (w.theta2 - w.theta1) / 2.0 + w.theta1
                angle_rad = math.radians(angle_deg)
        
                x = math.cos(angle_rad)
                y = math.sin(angle_rad)
        
                wedge_info.append((x, y, lbl, val))
        
            right_side = []
            left_side = []
            for (x, y, lbl, val) in wedge_info:
                if x >= 0:
                    right_side.append((x, y, lbl, val))
                else:
                    left_side.append((x, y, lbl, val))
        
            # Sort each side from top (largest y) to bottom (smallest y)
            right_side.sort(key=lambda t: t[1], reverse=True)
            left_side.sort(key=lambda t: t[1], reverse=True)
        
            # 5) Simple vertical spacing to avoid overlap
            def place_labels(sorted_side, min_gap=0.06, rad_scale=1.3):
                placed = []
                last_y = None
                for (x, y, lbl, val) in sorted_side:
                    # Move label outward beyond the donut boundary
                    new_y = y * rad_scale
                    if last_y is not None and abs(new_y - last_y) < min_gap:
                        new_y = last_y - min_gap
                    placed.append((x, new_y, lbl, val))
                    last_y = new_y
                return placed
        
            placed_right = place_labels(right_side)
            placed_left = place_labels(left_side)
        

            # 6) Annotate each label with a 2-segment angled line
            for (x, final_y, lbl, val) in placed_right + placed_left:
                # Arrow start near the wedge boundary
                arrow_x = x * 1.05
                # reverse the rad_scale factor
                arrow_y = (final_y / 1.3) * 1.05  
        
                # Decide left or right alignment
                if x >= 0:
                    ha = "left"
                    text_x = 1.8
                else:
                    ha = "right"
                    text_x = -1.8
        
                text_kw["ha"] = ha
        
                ax.annotate(
                    format_label(lbl, val),
                    xy=(arrow_x, arrow_y),
                    xytext=(text_x, final_y),
                    arrowprops=arrow_kwargs,
                    bbox=box_kw,
                    **text_kw
                )
                
            if value_on_slice:
                ring_midpoint = (inner_radius + 1.0) / 2.0
                # Place the numeric value near the center of each slice
                for w, val in zip(wedges, values):
                    angle_deg = (w.theta2 - w.theta1)/2 + w.theta1
                    angle_rad = math.radians(angle_deg)
                    # 0.6 ~ 0.7 times radius from center => roughly center of wedge
                    txt_x = math.cos(angle_rad) * ring_midpoint
                    txt_y = math.sin(angle_rad) * ring_midpoint
                    ax.text(
                        txt_x, txt_y,
                        str(val),   # or f"{val}" if you want a format
                        color=value_color,  # from new parameter
                        ha="center", 
                        va="center", 
                        fontsize=9
                    )
        
            # 8) Adjust plot limits & optional legend
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
        
            if legend:
                if legend_title:
                    ax.legend(
                        wedges, slice_labels, title=legend_title, 
                        loc=legend_loc
                    )
                else:
                    ax.legend(wedges, slice_labels, loc=legend_loc)

        # Title
        try:
            ax.set_title(data.columns[i])
        except: 
            pass 

    # Hide unused axes
    if n_rows > 1:
        for j in range(n_plots, len(axes)):
            axes[j].set_visible(False)
 
        plt.tight_layout()
        
    plt.show()

@is_data_readable(data_to_read="data")
def donut_chart_in(
    values=None,  
    data=None,   
    labels=None, 
    ax=None,
    wedge_width=0.3,
    inner_radius=0.7,
    dynamic_explode=True,
    base_explode=0.02,
    explode_scale=0.25,
    user_explode=None,
    max_slices=None,
    group_below_percent=None,
    others_label="Others",
    others_color="lightgray",
    label_format=None,
    value_on_slice=False,
    value_color="white",
    font_size=9,
    avoid_label_overlap=True,
    min_gap=0.06,
    rad_scale=1.3,
    multi_pass=False,
    line_kw=None,
    box_kw=None,
    text_kw=None,
    wedgeprops=None, 
    connectionstyle="angle3",
    startangle=0,
    counterclock=True,
    colors=None,
    title=None, 
    fig_size=(8, 8),
    legend=False,
    legend_loc="best",
    legend_title=None,
    **kw
):
    r"""
    Plot a single donut chart with advanced label placement and
    optional “inner hole.” This function accepts either direct
    arrays or a pandas DataFrame to retrieve <values> and <labels>.
    It can group small slices, limit the maximum number of slices,
    dynamically explode bigger slices, and place numeric values
    inside the donut ring.

    Parameters
    ----------
    values : array-like or ``str``, optional
        Numeric data for the slices. If a double backtick
        string`` is given and <data> is a DataFrame, this
        parameter is interpreted as the column name containing
        numeric values.
    data : pandas.DataFrame or array-like, optional
        If provided, and <values> or <labels> is a double backtick
        string``, the function fetches data from the specified
        column. Otherwise, direct arrays can be used for both
        <values> and <labels>.
    labels : array-like or ``str``, optional
        Labels for the slices. If a double backtick string`` is
        given and <data> is a DataFrame, this parameter is
        interpreted as the column name containing labels. If
        omitted, default labels like “Slice1,” “Slice2,” etc.
        may be generated.
    ax : matplotlib.axes.Axes, optional
        Axes on which to draw the donut. If None, a new figure
        and axes are created.
    wedge_width : float, default 0.3
        Fraction of the total radius occupied by the donut
        slices. If set to 1.0, the result is a pie chart.
    inner_radius : float, default 0.7
        Radius of the donut “hole.” If wedge_width < 1.0, a
        circular hole is drawn at this radius to create the
        donut effect.
    dynamic_explode : bool, default True
        If True, each slice is offset (exploded) proportionally
        to its fraction of the total. This is controlled by
        <base_explode> and <explode_scale>.
    base_explode : float, default 0.02
        Base offset for each slice when <dynamic_explode> is
        True.
    explode_scale : float, default 0.25
        Additional offset factor for bigger slices when
        <dynamic_explode> is True. The offset for a slice of
        fraction :math:`f` is :math:`base\_explode + explode\_scale \times f`.
    user_explode : float or array-like, optional
        If provided, overrides the dynamic explode logic. A
        single float applies to all slices, while an array must
        match the length of <values>.
    max_slices : int, optional
        If specified, only keep the top <max_slices> slices
        (by value) and group the remaining slices into an
        “Others” category.
    group_below_percent : float, optional
        Threshold below which slices are grouped into an
        “Others” category. For instance, 0.05 means any slice
        below 5% of the total is merged.
    others_label : ``str``, default "Others"
        Label for the merged slices if <max_slices> or
        <group_below_percent> is used.
    others_color : ``str``, default "lightgray"
        Special color for the “Others” slice if it appears.
    label_format : ``str``, optional
        Format string for slice labels, e.g. 
        ``"{label}: {value}"``. If None and <value_on_slice> is
        False, defaults to ``"{label}: {value}"``. If
        <value_on_slice> is True, defaults to ``"{label}"``.
    value_on_slice : bool, default False
        If True, numeric values are drawn inside each slice,
        while the slice’s label is placed externally with a
        leader line. If False, both label and value appear
        externally.
    value_color : ``str``, default "white"
        Color of the numeric text drawn inside each slice when
        <value_on_slice> is True.
    font_size : int, default 9
        Base font size for labels. Additional text properties
        can be set in <text_kw>.
    avoid_label_overlap : bool, default True
        Whether to shift labels vertically to reduce collisions.
        This logic separates slices on each side of the chart
        (left vs. right) and sorts them by descending y.
    min_gap : float, default 0.06
        Minimum vertical gap between adjacent labels on the
        same side, used when <avoid_label_overlap> is True.
    rad_scale : float, default 1.3
        Radial scaling factor for label placement. For each
        slice’s midpoint, the label is placed at 
        :math:`(x \times rad\_scale, y \times rad\_scale)`.
    multi_pass : bool, default False
        If True, attempt a second pass of label spacing. This
        can help reduce collisions further.
    line_kw : dict, optional
        Arrow (leader line) properties, including 
        ``arrowstyle``, ``lw``, ``color``, and so on.
    box_kw : dict, optional
        Bounding box properties for each label, including
        ``boxstyle``, ``fc``, ``ec``, etc.
    text_kw : dict, optional
        Additional properties for annotation text, e.g.
        ``fontsize``, alignment, etc.
    wedgeprops : dict, optional
        Dictionary of wedge properties (e.g. ``width``,
        ``edgecolor``) passed to the underlying pie function.
    connectionstyle : ``str``, default "angle3"
        Connection style for the arrow lines. This determines
        the shape of the multi-segment connector. For example,
        ``"angle3"`` draws a two-segment elbow.
    startangle : float, default 0
        Starting angle for the first slice in degrees. Slices
        proceed counterclockwise unless <counterclock> is False.
    counterclock : bool, default True
        If True, slices are laid out counterclockwise from
        <startangle>.
    colors : array-like or ``str``, optional
        If a double backtick string`` is given (e.g. 
        ``"tab20"``), a colormap is used. Otherwise, a list of
        colors can be provided or Matplotlib defaults are used.
    title : ``str``, optional
        Title for the donut chart.
    fig_size : tuple of float, default (8, 8)
        Size of the figure if no Axes is provided.
    legend : bool, default False
        If True, a legend is drawn showing each slice’s label.
    legend_loc : ``str``, default "best"
        Location code for the legend.
    legend_title : ``str``, optional
        Title for the legend.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the donut chart.
    ax : matplotlib.axes.Axes
        The axes on which the donut is drawn.

    Formulation
    ------------
    Let :math:`X = \\{ x_1, x_2, \\dots, x_n \\}` be the numeric
    slice values with total :math:`T = \\sum x_i`. For each
    slice :math:`i`, we compute its fraction 
    :math:`f_i = x_i / T`. The function may group slices with 
    :math:`x_i < \\alpha \\times T` or keep only the top 
    :math:`m` slices, merging the rest into “Others.”

    .. math::
       \\text{explode}_i = 
         \\begin{cases}
           \\text{base\_explode} + 
           \\text{explode\_scale} \\times f_i 
           & \\text{(if dynamic\_explode = True)} \\\\
           \\text{user\_explode}[i]
           & \\text{(if user\_explode is array)} \\\\
           \\text{user\_explode}
           & \\text{(if user\_explode is float)} \\\\
           0 & \\text{otherwise}
         \\end{cases}

    Examples
    --------
    >>> from gofast.plot.charts import donut_chart_in
    >>> # Example 1: Basic usage with arrays
    >>> values = [10, 20, 30, 5]
    >>> labels = ["A", "B", "C", "D"]
    >>> fig, ax = donut_chart_in(
    ...     values=values,
    ...     labels=labels,
    ...     wedge_width=0.4,
    ...     title="Basic Donut"
    ... )

    >>> # Example 2: Group small slices, limit max slices
    >>> values = [1, 2, 5, 50, 10, 3, 2]
    >>> labels = ["A", "B", "C", "Large", "E", "F", "G"]
    >>> fig, ax = donut_chart_in(
    ...     values=values,
    ...     labels=labels,
    ...     group_below_percent=0.05,
    ...     max_slices=4,
    ...     title="Grouped Donut"
    ... )

    >>> # Example 3: DataFrame usage with dynamic explode
    >>> import pandas as pd
    >>> df = pd.DataFrame({"Category": ["X","Y","Z","W"],
    ...                    "Value": [100, 80, 40, 30]})
    >>> fig, ax = donut_chart_in(
    ...     data=df,
    ...     values="Value",
    ...     labels="Category",
    ...     dynamic_explode=True,
    ...     base_explode=0.01,
    ...     explode_scale=0.4,
    ...     title="DataFrame Donut"
    ... )

    Notes
    -----
    The inline method <check_donut_inputs> retrieves and validates
    <values> and <labels> from either direct arrays or a DataFrame
    column [1]_. Leader lines are drawn with a multi-segment 
    connector style if <connectionstyle> is e.g. ``"angle3"``.

    See Also
    --------
    matplotlib.pyplot.pie : Underlying pie function
    gofast.core.checks.check_numeric_dtype: Checks numeric
        dtypes in arrays or DataFrame columns

    """

    # 1. If 'data' is given, parse 'values' and 'labels' from it.
    #    - If 'data' is a Series and 'values' is None, we take the entire
    #      series as numeric and use 'data.index' as labels if labels is None.
    #    - If 'data' is a DataFrame and 'values' is a string, we fetch that column.
    #      If 'labels' is also a string, we fetch that column for labels.
    #    - If 'labels' is None (and not a string), we default to data.index.
    #
    # 2. If 'data' is None, we use the 'values' and 'labels' directly.
    #    - If 'values' is None, we raise an error.
    #    - If 'labels' is None, we can either raise an error or create a default
    #      range of labels (e.g., ["Slice1", "Slice2", ...]).
    values, labels = check_donut_inputs(
        values=values, data=data, 
        labels=labels, 
    )
    
    # Convert labels and values to the final forms
    labels = list(labels)
    values = np.asarray(values, dtype=float)
    total_val = values.sum()

    # Handle grouping of small slices
    # (A) Group slices that are too small
    if group_below_percent is not None:
        # Identify which slices are < threshold
        threshold = group_below_percent * total_val
        keep_idx = [i for i, v in enumerate(values) if v >= threshold]
        group_idx = [i for i, v in enumerate(values) if v < threshold]

        if group_idx:  # we have slices to group
            grouped_val = values[group_idx].sum()
            values = np.concatenate([values[keep_idx], [grouped_val]])
            new_labels = [labels[i] for i in keep_idx] + [others_label]
            labels = new_labels

    # (B) If max_slices is set, keep only top N slices by value
    if max_slices is not None and len(values) > max_slices:
        # sort by descending
        sorted_idx = np.argsort(values)[::-1]
        top_idx = sorted_idx[:max_slices]
        group_idx = sorted_idx[max_slices:]
        grouped_val = values[group_idx].sum()

        new_values = list(values[top_idx]) + [grouped_val]
        new_labels = [labels[i] for i in top_idx] + [others_label]
        values = np.array(new_values, dtype=float)
        labels = new_labels

    # If we introduced an Others slice, color it specially
    # e.g. if we have more slices than original colors

    # 1) Create figure/axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig = ax.figure

    # 2) Decide wedgeprops, explode, etc.
    if wedge_width > 1.0:
        wedge_width = 1.0  # can't exceed total radius
    
    if wedgeprops is None:
        wedgeprops = dict(width=wedge_width, edgecolor="white")

    # Colors
    if isinstance(colors, str):
        # if user passes "cs4", "tab20", etc., we can try a colormap
        try:
            cmap = plt.get_cmap(colors)
            # build a color list
            n = len(values)
            colors = [cmap(i / n) for i in range(n)]
        except: 
            colors = make_plot_colors(values, colors=colors,)
    elif not colors:
        # fallback to default
        colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', None)

    # Explode
    if user_explode is not None:
        # user has a single float or array
        if isinstance(user_explode, (int, float)):
            explode_vals = [user_explode] * len(values)
        else:
            # assume user provided array
            explode_vals = user_explode
    elif dynamic_explode:
        explode_vals = []
        for val in values:
            frac = val / total_val
            explode_vals.append(base_explode + explode_scale * frac)
    else:
        explode_vals = [0] * len(values)

    # 3) Draw the donut
    wedges, _ = ax.pie(
        values,
        labels=None,  # we place labels manually
        explode=explode_vals,
        colors=colors,
        startangle=startangle,
        counterclock=counterclock,
        wedgeprops=wedgeprops,
        **kw
    )

    # If wedge_width < 1, create the donut hole
    if wedge_width < 1.0:
        hole = plt.Circle((0, 0), inner_radius, color="white")
        ax.add_artist(hole)

    # 4) Build arrow / box / text props
    if line_kw is None:
        line_kw = {}
    arrow_kwargs = dict(
        arrowstyle=line_kw.get("arrowstyle", "-"),
        lw=line_kw.get("lw", 1),
        color=line_kw.get("color", "black"),
        connectionstyle=connectionstyle
    )

    if box_kw is None:
        box_kw = dict(
            boxstyle="square,pad=0.3",
            fc="white",
            ec="gray",
            lw=0.6
        )

    if text_kw is None:
        text_kw = dict(va="center", ha="center")

    # unify the font size in text_kw
    text_kw.setdefault("fontsize", font_size)

    # Label format logic
    if label_format is None:
        # default
        if value_on_slice:
            # only label outside
            label_format = "{label}"
        else:
            # label + value outside
            label_format = "{label}: {value}"

    def format_label(lbl, val):
        return label_format.format(label=lbl, value=val)

    # 5) Gather wedge angles, x,y
    wedge_info = []
    for w, lbl, val in zip(wedges, labels, values):
        angle_deg = (w.theta2 - w.theta1)/2 + w.theta1
        angle_rad = math.radians(angle_deg)
        x = math.cos(angle_rad)
        y = math.sin(angle_rad)
        wedge_info.append((x, y, lbl, val))

    # split into left vs. right
    right_side = []
    left_side = []
    for (x, y, lbl, val) in wedge_info:
        if x >= 0:
            right_side.append((x, y, lbl, val))
        else:
            left_side.append((x, y, lbl, val))

    # sort by descending y
    right_side.sort(key=lambda t: t[1], reverse=True)
    left_side.sort(key=lambda t: t[1], reverse=True)


    # 6) Place labels with vertical spacing
    def place_labels(sorted_side):
        placed = []
        last_y = None
        for (x, y, lbl, val) in sorted_side:
            new_y = y * rad_scale
            if avoid_label_overlap and (last_y is not None):
                if abs(new_y - last_y) < min_gap:
                    new_y = last_y - min_gap
            placed.append((x, new_y, lbl, val))
            last_y = new_y
        return placed

    placed_right = place_labels(right_side)
    placed_left = place_labels(left_side)

    # (Optional) multi-pass approach: attempt a second pass
    if multi_pass:
        placed_right = place_labels(placed_right)
        placed_left = place_labels(placed_left)

    # 7) Annotate each label with a 2-segment angled line
    for (x, final_y, lbl, val) in placed_right + placed_left:
        # arrow start near wedge boundary
        arrow_x = x * 1.05
        arrow_y = (final_y / rad_scale) * 1.05  # reverse the rad_scale

        if x >= 0:
            ha = "left"
            text_x = 1.8
        else:
            ha = "right"
            text_x = -1.8

        text_kw["ha"] = ha

        ax.annotate(
            format_label(lbl, val),
            xy=(arrow_x, arrow_y),
            xytext=(text_x, final_y),
            arrowprops=arrow_kwargs,
            bbox=box_kw,
            **text_kw
        )

    # 8) If value_on_slice=True, place numeric values inside slices
    if value_on_slice:
        # place the text in the middle of the ring
        # if outer radius ~ 1.0, inner_radius=0.7 => midpoint=0.85
        ring_midpoint = (inner_radius + 1.0) / 2.0
    
        for w, val in zip(wedges, values):
            angle_deg = (w.theta2 - w.theta1)/2 + w.theta1
            angle_rad = math.radians(angle_deg)
            txt_x = math.cos(angle_rad) * ring_midpoint
            txt_y = math.sin(angle_rad) * ring_midpoint
            ax.text(
                txt_x,
                txt_y,
                str(val),
                color=value_color,
                ha="center",
                va="center",
                fontsize=font_size
            )

    # 9) Adjust plot limits & optional legend
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    

    if legend:
        if legend_title:
            ax.legend(
                wedges, 
                labels, 
                title=legend_title, 
                loc=legend_loc
                )
        else:
            ax.legend(
                wedges, labels, loc=legend_loc
                )
    # Set title if specified
    if title is not None:
        ax.set_title(title)

    # Display the plot
    plt.show()

    return fig, ax


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
    wedges, *_= ax.pie(
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
    # ifautopct
    # texts, autotexts
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
    
    return fig, ax 

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
    
    return fig, axs 
    
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


def multi_level_donut(
    data,
    levels=None,    
    value_col=None, 
    radius=1.0,
    width=0.2,
    gap=0.02,
    startangle=0,
    colors=None,
    cmap="tab20",
    edgecolor="white",
    text_color="black",
    title=None,
    fig_size=(8, 8),
    ax=None,
    label_format=None,
    min_font_size=7,
    max_font_size=12,
    legend=True,
    legend_loc="center left",
    legend_bbox=(1, 0, 0.2, 1),
    **kwargs
):

    r"""
    Plot a multi-level (hierarchical) donut chart from a pandas
    DataFrame or array-like structure. Each ring (or level) is
    subdivided according to the sum of the values in each category
    at that level. The function then recurses to lower levels,
    drawing additional rings inward.

    Parameters
    ----------
    data : ``pd.DataFrame`` or array-like
        The input data, which is converted to a DataFrame if not
        already. Each row should represent one observation, with
        hierarchical columns specified by <levels> and a numeric
        column specified by <value_col>.
    levels : list of ``str``, optional
        Names of the columns in <data> that define the hierarchical
        levels. The first name in the list is the outermost ring,
        and subsequent names define inner rings. If omitted, the
        function attempts to use all but the last column as
        levels.
    value_col : ``str``, optional
        The column in <data> containing the numeric values to sum
        for each ring segment. If omitted, the function infers it
        from columns not in <levels>.
    radius : float, default 1.0
        The outer radius of the outermost ring. Inner rings are
        drawn at successively smaller radii.
    width : float, default 0.2
        Thickness of each ring as a fraction of the radius.
    gap : float, default 0.02
        Small angular gap (in degrees) subtracted from each wedge
        to provide separation between adjacent segments in the
        same ring.
    startangle : float, default 0
        Starting angle (in degrees) for the outermost ring’s first
        segment.
    colors : list of color or None, optional
        Explicit list of colors for segments. If None, the function
        calls the inline method ``pick_color`` to choose colors from
        <cmap>.
    cmap : ``str`` or Colormap, default "tab20"
        Colormap name or object to use if <colors> is None.
    edgecolor : ``str``, default "white"
        The color of the wedge edges.
    text_color : ``str``, default "black"
        Color of any text elements, such as the chart title.
    title : ``str``, optional
        Title displayed at the top of the plot.
    fig_size : tuple of floats, default (8, 8)
        Size of the figure in inches if no <ax> is provided.
    ax : ``matplotlib.axes.Axes``, optional
        Axes on which to draw the donut. If None, a new figure and
        axes are created.
    label_format : ``str``, default "{label} ({value})"
        Format string for labeling each wedge. The placeholders
        {label} and {value} are replaced with the wedge’s name and
        integer value, respectively.
    min_font_size : int, default 7
        Minimum font size for labels (unused in this basic version).
    max_font_size : int, default 12
        Maximum font size for labels (unused in this basic version).
    legend : bool, default True
        Whether to display a legend mapping wedge patches to labels.
    legend_loc : ``str``, default "center left"
        Legend location argument passed to Axes.legend.
    legend_bbox : tuple, default (1, 0, 0.2, 1)
        Bbox_to_anchor for the legend, controlling its position
        outside the main plot.
    **kwargs
        Additional keyword arguments passed to the wedge patches.

    Formulation
    -----------
    Let :math:`L_1, L_2, \dots, L_k` be the hierarchical levels
    (outermost to innermost), and let :math:`V` be the column of
    numeric values. For each unique category :math:`c` in level
    :math:`L_1`, we compute:

    .. math::
       S_c = \sum_{i \in c} V_i

    Then each category :math:`c` at level :math:`L_1` occupies a
    wedge with an angle:

    .. math::
       \theta_c = 360 \times \frac{S_c}{\sum_c S_c}

    At the next level :math:`L_2`, each category in :math:`L_2`
    subdivides the wedge of its parent in :math:`L_1`, proportionally
    to its local sum. This process continues for each subsequent
    level, drawing rings inward.

    Examples
    --------
    >>> from gofast.plot.charts import multi_level_donut
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "Category1": ["A", "A", "B", "B"],
    ...     "Category2": ["X", "Y", "X", "Y"],
    ...     "Value": [10, 20, 5, 25]
    ... })
    >>> # The outer ring is "Category1", inner ring is "Category2"
    >>> # The numeric column is "Value".
    >>> fig, ax = multi_level_donut(
    ...     data=df,
    ...     levels=["Category1", "Category2"],
    ...     value_col="Value",
    ...     title="Multi-level Donut"
    ... )

    Notes
    -----
    The inline method ``pick_color`` is used internally if
    <colors> is None, cycling through the chosen colormap
    <cmap> for each ring segment. Each ring’s wedges are
    subdivided from the wedge of the parent ring in proportion
    to the sum of the numeric values for that subset [1]_.

    """

    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    
    if levels is None:
        # assume all but last col are levels, last col is value
        cols = data.columns.tolist()
        if len(cols) < 2:
            raise ValueError("Need at least 2 columns: (level, value)")
        levels = cols[:-1]
        value_col = cols[-1]
    else:
        levels = columns_manager(levels, empty_as_none= False)
        if value_col is None:
            # assume last is value
            # all_cols = levels[:]
            value_col = data.columns.difference(levels).tolist()
            if len(value_col) != 1:
                raise ValueError(
                    "Cannot infer value column automatically.")
            value_col = value_col[0]

    # ensure numeric
    data[value_col] = pd.to_numeric(
        data[value_col], errors="coerce").fillna(0)
    # group up
    # Example: if levels=["Level1","Level2"] => hierarchical
    # We'll do an iterative approach from top to bottom

    # build figure
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig = ax.figure
    ax.set_aspect("equal")
    ax.axis("off")

    # prepare color scheme
    if colors is not None:
        color_map = None
    else:
        color_map = plt.get_cmap(cmap)

    # track all patches for optional legend
    patch_handles = []
    label_handles = []

    # The idea:
    # - For each ring (level index k), group data by that level's categories
    # - For each category, sum the value
    # - Each category is assigned a wedge in ring k
    # - Then we subdivide that wedge in ring k+1 for its sub-categories
    
    # store arcs info as { (tuple_of_categories_up_to_level) :
        # (start_angle, end_angle, ring_inner, ring_outer) }
    arc_map = {}

    # first level grouping
    group_data = data.groupby(
        levels[0], dropna=False
        )[value_col].sum().reset_index()
    total_value = group_data[value_col].sum()
    if total_value <= 0:
        return fig, ax

    # angles
    current_angle = startangle
    ring_inner = radius - width
    ring_outer = radius

    # function to pick color from a colormap or user list
    def pick_color(i, n, cat):
        if colors is not None:
            idx = i % len(colors)
            return colors[idx]
        else:
            c = color_map(float(i) / max(1, (n - 1)))
            return c

    # draw ring for level 0
    cats0 = group_data[levels[0]].tolist()
    vals0 = group_data[value_col].tolist()
    ncat0 = len(cats0)
    
    if label_format is None: 
        label_format="{label} ({value})" 
        
    for i, (cat, val) in enumerate(zip(cats0, vals0)):
        if val <= 0:
            continue
        frac = val / total_value
        angle_span = 360.0 * frac
        start_ = current_angle
        end_ = current_angle + angle_span - gap
        if end_ < start_:
            end_ = start_
        c = pick_color(i, ncat0, cat)
        wedge = mpatches.Wedge(
            (0, 0),
            ring_outer,
            start_,
            end_,
            width=(ring_outer - ring_inner),
            facecolor=c,
            edgecolor=edgecolor,
            **kwargs
        )
        ax.add_patch(wedge)
        patch_handles.append(wedge)
        lbl = label_format.format(label=str(cat), value=int(val))
        label_handles.append(lbl)
        arc_map[(cat,)] = (start_, end_, ring_inner, ring_outer)
        current_angle += angle_span

    # now handle subsequent levels
    for lvl_index in range(1, len(levels)):
        # ring for level lvl_index
        ring_inner -= width
        ring_outer -= width
        parent_levels = levels[:lvl_index]
        this_level = levels[lvl_index]
        # group data at this level by parent + this
        grp_cols = parent_levels + [this_level]
        sub_data = data.groupby(grp_cols, dropna=False)[value_col].sum().reset_index()
        # for each parent group, we subdivide the arc from arc_map
        parents = sub_data[parent_levels].drop_duplicates().values.tolist()
        for pvals in parents:
            # pvals is a list of length lvl_index => parent categories
            # get sum for that parent
            ptuple = tuple(pvals)
            if ptuple not in arc_map:
                continue
            pstart, pend, pinner, pouter = arc_map[ptuple]
            psum = sub_data[
                (sub_data[parent_levels] == pd.Series(pvals)).all(axis=1)
            ][value_col].sum()
            if psum <= 0:
                continue
            parent_angle_span = pend - pstart
            # now fetch each child row
            child_rows = sub_data[
                (sub_data[parent_levels] == pd.Series(pvals)).all(axis=1)
            ]
            cvals = child_rows[value_col].tolist()
            ccats = child_rows[this_level].tolist()
            # total_p = sum(cvals) 
            current_sub_angle = pstart
            for i, (cc, cv) in enumerate(zip(ccats, cvals)):
                if cv <= 0:
                    continue
                frac = cv / psum
                angle_span = parent_angle_span * frac
                start_ = current_sub_angle
                end_ = current_sub_angle + angle_span - gap
                if end_ < start_:
                    end_ = start_
                c = pick_color(i, len(ccats), cc)
                wedge = mpatches.Wedge(
                    (0, 0),
                    ring_outer,
                    start_,
                    end_,
                    width=(ring_outer - ring_inner),
                    facecolor=c,
                    edgecolor=edgecolor,
                    **kwargs
                )
                ax.add_patch(wedge)
                patch_handles.append(wedge)
                lbl = label_format.format(label=str(cc), value=int(cv))
                label_handles.append(lbl)
                arc_map[ptuple + (cc,)] = (start_, end_, ring_inner, ring_outer)
                current_sub_angle += angle_span

    if title:
        ax.set_title(title, color=text_color)

    if legend:
        ax.legend(
            patch_handles,
            label_handles,
            loc=legend_loc,
            bbox_to_anchor=legend_bbox
        )

    return fig, ax


@is_data_readable(data_to_read="data")
def two_ring_donuts(
    data=None,
    outer_values=None,
    outer_labels=None,
    inner_values=None,
    inner_labels=None,
    # Optional grouping dict for advanced usage:
    group_names=None,  
    reverse=False,     
    wedge_width=0.3,
    gap=0.02,
    startangle=0,
    outer_radius=1.0,
    fig_size=(8, 8),
    ax=None,
    colors_out=None,
    colors_in=None,
    autopct_out='%1.1f%%',
    autopct_in=None,
    labeldistance_out=1.05,
    labeldistance_in=0.7,
    textprops_out=None,
    textprops_in=None,
    wedgeprops_out=None,
    wedgeprops_in=None,
    legend=False,
    legend_loc="best",
    legend_labels_out=None,
    legend_labels_in=None,
    **kwargs
):
    r"""
    Plot two independent donut charts (outer and inner), each 
    summing to 360 degrees, with optional grouping logic via 
    ``group_names`` and reversal of rings via <reverse>. The 
    function accepts either direct arrays or a pandas DataFrame 
    through <data>, allowing flexibility in how values and labels 
    are provided.

    Parameters
    ----------
    data : pd.DataFrame or array-like, optional
        If given, this can be used to fetch numeric columns or 
        labels by name (if <outer_values>, <inner_values>, 
        <outer_labels>, or <inner_labels> are strings). Otherwise, 
        pass arrays directly for those parameters.
    outer_values : array-like or ``str``, optional
        Numeric values for the outer donut slices. If a double 
        backtick string`` is given and <data> is a DataFrame, 
        that column is used.
    outer_labels : array-like or ``str``, optional
        Labels for the outer donut slices. If a double backtick 
        string`` is given and <data> is a DataFrame, that column 
        is used.
    inner_values : array-like or ``str``, optional
        Numeric values for the inner donut slices. Similar usage 
        as <outer_values>.
    inner_labels : array-like or ``str``, optional
        Labels for the inner donut slices. Similar usage as 
        <outer_labels>.
    group_names : dict, optional
        A dictionary mapping an outer-slice name to a list of 
        inner-slice labels. The function sums the values of 
        those inner slices to form a single outer slice. If 
        <reverse> is True, the logic is reversed: the 
        “outer” ring in code becomes the smaller slices, and the 
        aggregated group becomes the “inner” ring.
    reverse : bool, default False
        If True, swap the roles of outer and inner after grouping. 
        This can mimic the style where a single big slice 
        (e.g. “Oils(7)”) is placed on the inner ring, and the 
        smaller slices (“Oil(6)”, “Gas(1)”) appear on the outer 
        ring.
    wedge_width : float, default 0.3
        Thickness of each donut ring as a fraction of its radius.
    gap : float, default 0.02
        Radial gap between the outer and inner rings.
    startangle : float, default 0
        Starting angle in degrees for both donuts.
    outer_radius : float, default 1.0
        Radius for the outer donut. The inner donut is placed at 
        ``outer_radius - wedge_width - gap``.
    fig_size : tuple of float, default (8, 8)
        Size of the figure in inches if no existing Axes is passed.
    ax : matplotlib.axes.Axes, optional
        Axes on which to draw the donuts. If None, a new figure 
        and axes are created.
    colors_out : list of color, optional
        Explicit colors for the outer donut slices. If None, 
        Matplotlib defaults apply.
    colors_in : list of color, optional
        Explicit colors for the inner donut slices.
    autopct_out : str or callable, default '%1.1f%%'
        Format string or function for numeric labels on the outer 
        slices. If None, no numeric labels appear.
    autopct_in : str or callable, optional
        Similar to <autopct_out>, but for the inner donut slices.
    labeldistance_out : float, default 1.05
        Radial distance for outer slice labels, relative to the 
        outer donut radius.
    labeldistance_in : float, default 0.7
        Radial distance for inner slice labels, relative to the 
        inner donut radius.
    textprops_out : dict, optional
        Additional text properties (e.g. fontsize) for the outer 
        slice labels.
    textprops_in : dict, optional
        Additional text properties for the inner slice labels.
    wedgeprops_out : dict, optional
        Additional wedge properties (e.g. edgecolor) for the 
        outer donut.
    wedgeprops_in : dict, optional
        Additional wedge properties for the inner donut.
    legend : bool, default False
        If True, display a combined legend for both donuts.
    legend_loc : str, default "best"
        Legend location code (e.g. "upper right").
    legend_labels_out : list of str, optional
        Custom labels for the outer donut in the legend. If None, 
        uses <outer_labels>.
    legend_labels_in : list of str, optional
        Custom labels for the inner donut in the legend. If None, 
        uses <inner_labels>.
    **kwargs
        Additional keyword arguments passed internally (unused by 
        default).

    Formulation
    -----------
    Let :math:`V_o = \\{ v_{o1}, v_{o2}, ..., v_{on} \\}` be the 
    outer donut slice values, and :math:`V_i = \\{ v_{i1}, 
    v_{i2}, ..., v_{im} \\}` be the inner donut slice values. Each 
    donut is independently normalized to 100%:

    .. math::
       \\text{sum}(V_o) = \\sum_{k=1}^{n} v_{ok} \\quad
       \\text{and}\\quad
       \\text{sum}(V_i) = \\sum_{k=1}^{m} v_{ik}

    The outer ring is drawn at radius :math:`r_o`, and the inner 
    ring at :math:`r_o - w - g`, where :math:`w` is <wedge_width> 
    and :math:`g` is <gap>. If <group_names> is used, we aggregate 
    certain slices by summing their values, effectively merging 
    multiple slices into one.

    Examples
    --------
    1) **Basic usage** with direct arrays:

       >>> from gofast.plot.charts import two_ring_donuts
       >>> out_vals = [26, 12, 10]
       >>> out_labs = ["Single use (26)", "S&P500 (12)", "Indices (10)"]
       >>> in_vals = [2, 7, 7, 10]
       >>> in_labs = ["notes/bonds (2)", "stocks (7)",
       ...            "volatility (7)", "indices (10)"]
       >>> fig, ax = two_ring_donuts(
       ...     outer_values=out_vals,
       ...     outer_labels=out_labs,
       ...     inner_values=in_vals,
       ...     inner_labels=in_labs,
       ...     startangle=140,
       ...     legend=True
       ... )
       >>> fig.show()

    2) **Using group_names** to combine slices:

       >>> group_map = {
       ...   "Market Challenges": ["notes/bonds (2)",
       ...                         "stocks (7)",
       ...                         "volatility (7)",
       ...                         "indices (10)"],
       ...   "Use case": "Single use (26)",
       ...   "Indices&S&P": ["S&P500 (12)", "Indices (10)"]
       ... }
       >>> # The outer ring merges these inner slices
       >>> fig, ax = two_ring_donuts(
       ...     inner_values=in_vals,
       ...     inner_labels=in_labs,
       ...     group_names=group_map,
       ...     legend=True
       ... )
       >>> fig.show()

    3) **Reversed** usage:

       >>> # Suppose we want the big aggregated slices on the inner ring
       >>> # and the smaller slices on the outer ring:
       >>> fig, ax = two_ring_donuts(
       ...     inner_values=in_vals,
       ...     inner_labels=in_labs,
       ...     group_names=group_map,
       ...     reverse=True,
       ...     legend=True
       ... )
       >>> fig.show()

    Notes
    -----
    The inline parameter <group_names> merges multiple slices 
    from the “inner” data into a single slice for the “outer” 
    donut, unless <reverse> is True, in which case the roles are 
    reversed [1]_.

    See Also
    --------
    :func:`matplotlib.pyplot.pie` : Underlying pie chart function
    :func:`pandas.DataFrame` : Data structure for referencing columns

    References
    ----------
    .. [1] Author A., Author B., "Title of Some Paper," *Journal*,
       vol. X, no. Y, pp. 1-10, Year.
    """

    # Convert data if needed
    if data is not None:
        if isinstance(data, pd.DataFrame):
            # If outer_values is a string, fetch that column
            if isinstance(outer_values, str):
                outer_values = data[outer_values].values
            if isinstance(outer_labels, str):
                outer_labels = data[outer_labels].values
            if isinstance(inner_values, str):
                inner_values = data[inner_values].values
            if isinstance(inner_labels, str):
                inner_labels = data[inner_labels].values
        else:
            # fallback, maybe data is array-like
            data = pd.DataFrame(data)

    # If group_names is provided, build outer_values/labels by grouping inner.
    if group_names is not None:
        # The "inner" ring is the raw list of slices
        if (inner_values is None) or (inner_labels is None):
            raise ValueError(
                "group_names requires that inner_values and inner_labels "
                "are provided or can be inferred."
            )

        # Convert to array if not already
        inner_values = np.asarray(inner_values, dtype=float)
        # If user gave them as a Pandas series, convert to array
        if not isinstance(inner_labels, (list, np.ndarray)):
            inner_labels = list(inner_labels)

        # Build a map from label -> value
        label_value_map = dict(zip(inner_labels, inner_values))

        # Summation
        outer_dict = {}
        for gkey, glabs in group_names.items():
            if isinstance(glabs, str):
                # single label
                glabs = [glabs]
            # sum the values
            total_sum = 0
            for lab in glabs:
                total_sum += label_value_map.get(lab, 0)
            outer_dict[gkey] = total_sum

        # Now we have a dict for outer ring
        out_labs = list(outer_dict.keys())
        out_vals = list(outer_dict.values())

        if reverse:
            # In reverse mode, the user wants the "outer ring" in code
            # to be visually the smaller slices, so we swap them.
            # i.e. "outer ring" in final figure is the bigger sums,
            # "inner ring" is the raw slices
            tmp_labs, tmp_vals = out_labs, out_vals
            out_labs, out_vals = inner_labels, inner_values
            inner_labels, inner_values = tmp_labs, tmp_vals

            # Also swap color assignments if needed
            if (colors_out is not None) or (colors_in is not None):
                pass  # advanced logic could swap these if desired

            # So effectively the "outer" ring in code is the raw slices
            # while the "inner" ring is the grouped sums
        else:
            outer_labels, outer_values = out_labs, out_vals

    # fallback if user never provided them
    if outer_labels is None:
        outer_labels = [f"Slice {i+1}" for i in range(len(outer_values))]
    if inner_labels is None:
        inner_labels = [f"Slice {i+1}" for i in range(len(inner_values))]

    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    else:
        fig = ax.figure

    ax.axis("equal")

    # Default wedgeprops
    if wedgeprops_out is None:
        wedgeprops_out = {}
    if wedgeprops_in is None:
        wedgeprops_in = {}

    # Outer donut
    wedges_out, *_text_out = ax.pie(
        outer_values,
        labels=outer_labels,
        radius=outer_radius,
        startangle=startangle,
        labeldistance=labeldistance_out,
        colors=colors_out,
        autopct=autopct_out,
        textprops=textprops_out,
        **wedgeprops_out
    )
    if autopct_out is not None and len(_text_out) == 2:
        # newest versions return 3 arrays from pie
        texts_out, autotexts_out = _text_out

    plt.setp(wedges_out, width=wedge_width)

    # Inner donut
    inner_radius = outer_radius - wedge_width - gap
    wedges_in, *_texts_in = ax.pie(
        inner_values,
        labels=inner_labels,
        radius=inner_radius,
        startangle=startangle,
        labeldistance=labeldistance_in,
        colors=colors_in,
        autopct=autopct_in,
        textprops=textprops_in,
        **wedgeprops_in
    )
    if autopct_in is not None and len(_texts_in) == 2:
        texts_in, autotexts_in = _texts_in

    plt.setp(wedges_in, width=wedge_width)

    if legend:
        all_handles = wedges_out + wedges_in
        if legend_labels_out is None:
            legend_labels_out = outer_labels
        if legend_labels_in is None:
            legend_labels_in = inner_labels
        all_labels = list(legend_labels_out) + list(legend_labels_in)
        ax.legend(all_handles, all_labels, loc=legend_loc)

    return fig, ax
