# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Provides quantile-based plotting functionalities. It includes 
tools to visualize quantile distributions, quantile-based predictions,
and quantile distances.
"""
from __future__ import annotations 
import re 

import warnings
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from ..core.diagnose_q import ( 
    validate_q_dict, 
    validate_quantiles, 
    build_q_column_names, 
    detect_quantiles_in
)
from ..core.handlers import columns_manager
from ..core.plot_manager import ( 
    default_params_plot, 
    return_fig_or_ax
)
from ..compat.sklearn import validate_params
from ..decorators import isdf
from ..utils.spatial_utils import filter_position 
from ..utils.validator import  filter_valid_kwargs
from ._config import PlotConfig
from .utils import _set_defaults, _param_defaults


__all__= ['plot_qbased_preds', 'plot_qdist', 'plot_quantile_distributions']

@return_fig_or_ax(return_type ='ax')
@default_params_plot(
    savefig=PlotConfig.AUTOSAVE('my_q_based_pred_plot.png'), 
    title ="Quantile-based Predictions",
    fig_size=(8, 6) 
 )
@validate_params ({ 
    'df': ['array-like'], 
    'dt_col':[str, None], 
    'q_cols': ['array-like']
    })
@isdf 
def plot_qbased_preds(
    df,
    q_cols, 
    dt_col=None,
    pos_val=None,
    pos_cols=None,
    title=None,  
    xlabel=None,  
    ylabel=None,  
    figsize=(10, 6),
    kind="line",
    linewidth=2,
    fbtw_color='blue',
    fbtw_alpha=0.2,
    fbtw_label=None,
    marker='o',
    linestyle='-',
    color='red',
    label=None,
    show_grid=True,
    grid_props=None,
    show_legend=True,
    fbtw_kws=None,
    rotation=None, 
    **kws
):
    """
    Plots quantile-based predictions, allowing the user to
    visualize multiple quantiles (e.g. q10, q50, q90) along with
    an optional fill region between the lower and upper bounds.
    The function can plot the median or central quantile as a
    line while filling an area for the uncertainty band [1]_.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the quantile columns.
    q_cols : list or dict
        Quantile definitions. If this parameter is a dict
        with keys like ``q10``, ``q50``, ``q90``, their numeric
        portion is parsed. If it is a list, items are assigned
        dummy keys (``q0``, ``q1``, etc.).
    dt_col : str, optional
        Column representing the x-axis (often a date/time).
        If None, the DataFrame index is used.
    pos_val : tuple, optional
        A spatial position (e.g., latitude/longitude) used for
        filtering if <parameter `pos_cols`> is specified.
    pos_cols : tuple, optional
        Column names corresponding to <parameter `pos_val`>.
    title : str, optional
        The plot title. Defaults to ``"Quantile Predictions"``
        if not specified.
    xlabel : str, optional
        X-axis label. Defaults to ``"Date/Time"``.
    ylabel : str, optional
        Y-axis label. Defaults to ``"Values"``.
    figsize : tuple, optional
        Size of the figure (width, height). Defaults to
        (10, 6).
    kind : str, optional
        The type of plot (``"line"``, ``"scatter"``,
        ``"bar"``, or ``"step"``). Defaults to ``"line"``.
    linewidth : float, optional
        Line width for the central quantile. Defaults to 2.
    fbtw_color : str, optional
        Color for the uncertainty band. Defaults to ``"blue"``.
    fbtw_alpha : float, optional
        Alpha transparency for the fill region. Defaults to
        0.2.
    fbtw_label : str, optional
        Label for the fill region. Defaults to
        ``"Uncertainty Band (10%-90%)"`` if none is provided.
    marker : str, optional
        Marker style for plotting the central quantile. Defaults
        to ``"o"``.
    linestyle : str, optional
        Style for the line plot (e.g., ``"-"``, ``"--"``).
        Defaults to ``"-"``.
    color : str, optional
        Color for the central quantile line. Defaults
        to ``"red"``.
    label : str, optional
        Legend label for the central quantile line. Defaults
        to ``"Median Prediction (50%)"`` if no label is given.
    show_grid : bool, optional
        If True, a grid is displayed. Defaults to True.
    grid_props : dict, optional
        Grid customization (e.g., ``{'linestyle': '--',
        'alpha': 0.5}``).
    show_legend : bool, optional
        If True, a legend is displayed. Defaults to True.
    fbtw_kws : dict, optional
        Additional kwargs for the fill region, passed to
        ``matplotlib.axes.Axes.fill_between``.
    rotation : int, optional, default=0
        The angle of rotation for the x-axis tick labels.
    **kws
        Additional keyword arguments for plotting functions
        like ``matplotlib.axes.Axes.plot``, ``matplotlib.axes.
        Axes.scatter``, etc.
    
    Notes
    -----
    This function filters the dataset if <parameter `pos_val`>
    and <parameter `pos_cols`> are specified, using
    :math:`\\mathrm{find\\_closest}` within a certain threshold.
    All missing or invalid position columns are ignored based on
    error-handling logic.
    
    The uncertainty band is derived by the difference between
    the lower quantile :math:`q_{\\text{low}}` and the upper
    quantile :math:`q_{\\text{high}}`:
    
    .. math::
       \\Delta = q_{\\text{high}} - q_{\\text{low}}
    
    while the central quantile (e.g., :math:`q_{50}`) is often
    treated as a median or representative forecast:
    
    .. math::
       q_{50} = \\text{median}
    
    Examples
    --------
    >>> from gofast.plot.q import plot_qbased_preds
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'lat': [113.309998, 113.310001],
    ...     'lon': [22.831362, 22.831364],
    ...     'q10': [0.8, 0.9],
    ...     'q50': [1.0, 1.1],
    ...     'q90': [1.2, 1.3]
    ... })
    >>> # Simple line plot
    >>> plot_qbased_preds(
    ...     df,
    ...     q_cols={'q10': 'q10', 'q50': 'q50', 'q90': 'q90'},
    ...     dt_col=None
    ... )
    
    See Also
    --------
    plot_prediction_intervals:  
        Plots predicted intervals (e.g., lower, median, and upper quantiles) 
        along with an optional reference series.
    plot_with_uncertainty: Plot various uncertainty visualizations.
    
    References
    ----------
    .. [1] Roe, J. & Sage, L. (2020). Methods of statistical
       interval visualization. Journal of Uncertainty
       Analysis, 5(1), 101-115.
    """

    # Update default plot parameters
    _param_defaults.update({
        'title': "Quantile Predictions",
        'xlabel': "Date/Time",
        'ylabel': 'Values',
    })
    params = _set_defaults(title=title, xlabel=xlabel, ylabel=ylabel)

    # Filter dataframe by position if specified
    if pos_val is not None:
        df = filter_position(
            df,
            pos_val,
            pos_cols=pos_cols,
            find_closest=True,
            threshold=0.05, 
            error='warn'
        )
        if df.empty:
            # If no data after filter, warn and return
            warnings.warn(
                f"No data found for position {pos_val}. "
                "Unable to generate plot."
            )
            return
    # Sort dataframe by the datetime/temporal column if given
    if dt_col:
        # Ensure dt_col is datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df[dt_col]):
            df[dt_col] = pd.to_datetime(df[dt_col], errors='coerce')
        #XXX
        df = df.sort_values(by=dt_col)
        x_data = df[dt_col]
    else:
        x_data = df.index  # Use index if no time col is provided

    
    # More robust handling of multiple quantiles
    # Convert q_cols to a dict if it isn't one already
    if not isinstance(q_cols, dict):
        # If it's a list, generate a simple mapping like {'q0': col0, 'q1': col1}
        q_cols = {f'q{i}': col for i, col in enumerate(q_cols)}

    # Extract the numeric portion from keys like 'q10', 'q90', etc.
    # Sort them to identify the min/mid/max or any quantile ordering
    quantile_map = {}
    for k, v in q_cols.items():
        if k.startswith('q'):
            # Attempt to parse numeric part, e.g. 'q10' -> 10
            try:
                num_val = float(k[1:])
                quantile_map[num_val] = v
            except ValueError:
                # If parsing fails, skip silently or warn
                warnings.warn(
                    f"Could not parse quantile '{k}'. Skipped."
                )
        else:
            # If not 'q*', skip or warn
            warnings.warn(
                f"Quantile key '{k}' does not start with 'q'. Skipped."
            )

    # If we have no valid quantile columns, warn and return
    if not quantile_map:
        warnings.warn(
            "No valid quantile columns found in `q_cols`. "
            "Nothing to plot."
        )
        return

    sorted_qs = sorted(quantile_map.keys())
    # Retrieve series from df for each quantile key
    data_map = {q: df[col] for q, col in [(q, quantile_map[q]) for q in sorted_qs]
                if col in df.columns}

    # Identify the min and max quantiles for fill_between if >=2 quantiles
    q_min = sorted_qs[0]
    q_max = sorted_qs[-1]

    # Prepare the figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the uncertainty band if we have at least 2 quantiles
    if len(sorted_qs) >= 2:
        fbtw_kws = filter_valid_kwargs(ax.fill_between, (fbtw_kws or {}))
        ax.fill_between(
            x_data,
            data_map[q_min],
            data_map[q_max],
            color=fbtw_color,
            alpha=fbtw_alpha,
            label=fbtw_label or f"Uncertainty Band "
                                f"({int(q_min)}%-{int(q_max)}%)",
            **fbtw_kws
        )

    # Decide on the "median" quantile:
    #  - If 'q50' exists, use that
    #  - Else pick the middle from the sorted list
    if 50.0 in quantile_map:
        median_key = 50.0
    else:
        # If there's only one, it is effectively the median
        # If multiple, pick the center
        mid_idx = len(sorted_qs) // 2
        median_key = sorted_qs[mid_idx]

    # Prepare plot style for median quantile
    median_data = data_map[median_key]
    median_label = label or f"Median Prediction ({int(median_key)}%)"

    # Determine which function to call for the chosen kind
    if kind == "scatter":
        kws = filter_valid_kwargs(ax.scatter, kws)
        ax.scatter(
            x_data,
            median_data,
            color=color,
            marker=marker,
            label=median_label,
            **kws
        )
    elif kind == "bar":
        kws = filter_valid_kwargs(ax.bar, kws)
        ax.bar(
            x_data,
            median_data,
            color=color,
            alpha=0.7,
            label=median_label,
            **kws
        )
    elif kind == "step":
        kws = filter_valid_kwargs(ax.step, kws)
        ax.step(
            x_data,
            median_data,
            color=color,
            linewidth=linewidth,
            label=median_label,
            **kws
        )
    else:  # default to 'line'
        kws = filter_valid_kwargs(ax.plot, kws)
        ax.plot(
            x_data,
            median_data,
            color=color,
            marker=marker,
            linestyle=linestyle,
            linewidth=linewidth,
            label=median_label,
            **kws
        )

    # Apply axis labels and title
    ax.set_xlabel(params['xlabel'])
    ax.set_ylabel(params['ylabel'])
    ax.set_title(params['title'])

    # Display the plot
    plt.xticks(rotation=rotation or 0.)
    # Toggle grid
    if show_grid:
        ax.grid(True, **(grid_props or dict(linestyle=':', alpha=0.7)))

    # Toggle legend
    if show_legend:
        ax.legend()

    # Display the plot
    plt.show()

@return_fig_or_ax
@default_params_plot(
    savefig=PlotConfig.AUTOSAVE('my_q.dist_plot.png'), 
    fig_size=None, 
  )
@validate_params ({ 
    'df': ['array-like'], 
    'quantiles': ['array-like', None], 
    'dt_values':['array-like', None], 
    'value_prefix': [str, None], 
    'q_cols': [dict, None], 
    })
@isdf 
def plot_qdist(
    df,
    x_col,               
    y_col,               
    dt_name=None,        
    quantiles=None,
    dt_values=None,
    value_prefix=None,   
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
    show_grid=False, 
    grid_props=None, 
):
    r"""
    Plot quantile distributions across spatial (x,y) domains and,
    optionally, temporal or categorical dimensions. This function
    supports four naming configurations for identifying quantile
    columns:
    
    1. ``<value_prefix>_<dt_value>_q<quantile>``
    2. ``<dt_value>_q<quantile>``
    3. ``<value_prefix>_q<quantile>``
    4. ``q<quantile>``
    
    In essence, :func:`plot_qdist` either creates a 2D grid of subplots
    (when date/time values are detected or explicitly provided) or a
    1D arrangement of subplots (only quantiles). By default, rows
    represent quantiles and columns represent date/time steps:
    
    .. math::
       \text{Rows} \rightarrow \text{Quantiles}, \quad
       \text{Columns} \rightarrow \text{Dates}
    
    When ``reverse=True``, the layout is inverted:
    
    .. math::
       \text{Rows} \rightarrow \text{Dates}, \quad
       \text{Columns} \rightarrow \text{Quantiles}
    
    
    Parameters
    ----------
    df: pd.DataFrame
        The input DataFrame containing at least the coordinates
        `<x_col>` and `<y_col>`, plus columns that match one of
        the naming configurations above or a user-supplied mapping
        in ``q_cols``.
    
    x_col : str
        Name of the column representing the x-coordinate (e.g.,
        'longitude').
    
    y_col : str
        Name of the column representing the y-coordinate (e.g.,
        'latitude').
    
    dt_name : str, optional
        If not ``None``, a descriptive label for the date/time
        dimension (e.g., 'year'). Used solely for subplot titles
        and labeling. If omitted, the function treats the data as
        1D in quantiles.
    
    quantiles : list of float, optional
        A user-specified list of quantile values (e.g., ``[0.1,
        0.5, 0.9]``). If not provided, the function attempts to
        detect quantiles from DataFrame columns. If detection
        fails and no mapping is given in ``q_cols``, an error is
        raised.
    
    dt_values : list, optional
        A list of date/time values or identifiers (e.g., years)
        for which columns exist in the DataFrame. When present,
        :func:`plot_qdist` tries to match them against column
        names. If None, the function infers them from the
        detected or matched columns.
    
    value_prefix : str or None, optional
        The prefix used in patterns like ``<value_prefix>_2025_q0.5``.
        If None, the function detects columns without filtering
        by a prefix.
    
    q_cols : dict, optional
        A direct mapping of quantiles to column names or dictionaries.
        For a 1D scenario:
        
        .. code-block:: python
    
           {
             0.1: "my_col_q0.1",
             0.5: "my_col_q0.5",
             0.9: "my_col_q0.9"
           }
    
        For a 2D scenario (quantiles x dt_values):
    
        .. code-block:: python
    
           {
             0.1: { "2023": "subs_2023_q0.1", "2024": "subs_2024_q0.1" },
             0.5: { "2023": "subs_2023_q0.5", "2024": "subs_2024_q0.5" }
           }
    
        This mapping bypasses automatic naming detection. If
        provided, `<quantiles>` and `<dt_values>` can also be
        specified for ordering or cross-verification.
    
    cmap : str, optional
        The colormap used for color encoding (default 'viridis').
    
    s : int, optional
        The marker size for scatter plots.
    
    alpha : float, optional
        The marker transparency in [0, 1].
    
    cbar_orientation : {'vertical', 'horizontal'}, optional
        Orientation of the colorbar. Default 'vertical'.
    
    cbar_fraction : float, optional
        Fraction of original axes to allocate for the colorbar.
    
    cbar_pad : float, optional
        Padding between the colorbar and subplot edge.
    
    figsize : tuple, optional
        The figure size in inches, e.g., ``(12, 6)``. If None,
        a default size is determined by the number of rows and
        columns.
    
    savefig : str, optional
        Path to save the resulting figure (e.g., 'output.png').
        If None, the figure is displayed interactively.
    
    dpi : int, optional
        The dots-per-inch resolution for the figure. Default 300.
    
    axis_off : bool, optional
        If True, hides the axes (no ticks or labels).
    
    reverse : bool, optional
        If True, inverts the subplot layout so that columns
        correspond to quantiles and rows to date/time steps.
        Otherwise, rows represent quantiles and columns
        represent date/time values.
    
    
    Examples
    --------
    >>> from gofast.plot.q import plot_qdist
    >>> import pandas as pd
    >>> import numpy as np
    
    >>> # Example DataFrame with patterns like:
    >>> #    subsidence_2023_q10
    >>> #    subsidence_2023_q50
    >>> #    subsidence_2024_q90
    >>> n = 100
    >>> df = pd.DataFrame({
    ...   'longitude': np.random.uniform(113, 114, n),
    ...   'latitude':  np.random.uniform(22, 23,  n),
    ...   'subsidence_2023_q10': np.random.rand(n)*10,
    ...   'subsidence_2023_q50': np.random.rand(n)*10,
    ...   'subsidence_2024_q90': np.random.rand(n)*10
    ... })
    
    >>> # Simple call that detects date/time and quantiles:
    >>> plot_qdist(
    ...   df,
    ...   x_col='longitude',
    ...   y_col='latitude',
    ...   value_prefix='subsidence'
    ... )
    
    >>> # If no date/time dimension is present, columns might be
    >>> # e.g., 'subsidence_q10', 'subsidence_q50', 'subsidence_q90',
    >>> # or just 'q10','q50','q90' for truly single-dimensional data.
    
    Notes
    -----
    This function processes DataFrame columns that match specific
    naming patterns to identify date/time indices and quantile
    levels. If a date dimension is found, the resulting figure
    forms a 2D grid of subplots arranged by quantile and date.
    Otherwise, a 1D arrangement (only quantiles) is plotted.
    The user may override automatic detection by supplying a
    structured mapping in ``q_cols``.
    
    .. math::
       \mathbf{Plot}:
       \begin{cases}
       \text{2D: } \text{rows} \times \text{columns} = 
       \text{quantiles} \times \text{dates} & 
       \text{(or reversed if } reverse=True)\\
       \text{1D: } \text{rows} = \text{quantiles} & 
       \text{(single column or vice versa)}
       \end{cases}
    
    See Also
    --------
    gofast.plot.q.plot_quantile_distributions: 
        Another quantiles plot with absolute `dt_col`.
    matplotlib.pyplot.scatter : Core scatter plotting.
    matplotlib.pyplot.colorbar : Colorbar configuration.
    
    References
    ----------
    .. [1] Han, J., Kamber, M., & Pei, J. (2011). *Data Mining:
           Concepts and Techniques*, 3rd edition. Elsevier.
    """

    # Function robustly handles four data configurations:
    #   (1) <value_prefix>_<dt_value>_q<quantile>
    #   (2) <dt_value>_q<quantile>
    #   (3) <value_prefix>_q<quantile>
    #   (4) q<quantile>

    # If a `dt_value` is detected, a 2D grid is plotted 
    # (dates x quantiles). Otherwise, a 1D grid is plotted 
    # (just quantiles).
    
    # Init grid props 
    if grid_props is None: 
        grid_props = {'linestyle': ':', 'alpha': .7}

    # If q_cols is provided, it overrides auto-detection logic
    if q_cols is not None:
        _plot_from_qcols(
            df=df, x_col=x_col, y_col=y_col,
            dt_name=dt_name, dt_values=dt_values,
            q_cols=q_cols, cmap=cmap, s=s,
            alpha=alpha, cbar_orientation=cbar_orientation,
            cbar_fraction=cbar_fraction, cbar_pad=cbar_pad,
            figsize=figsize, savefig=savefig, dpi=dpi,
            axis_off=axis_off, reverse=reverse, 
            show_grid=show_grid, 
            grid_props=grid_props
        )
        return

    # Auto-detect columns that match the patterns
    # (1) prefix_dt_qquant  => ^(prefix_)?(\d+)_q([\d.]+)$
    # (2) dt_qquant         => ^(\d+)_q([\d.]+)$
    # (3) prefix_qquant     => ^(prefix_)?q([\d.]+)$
    # (4) qquant            => ^q([\d.]+)$
    # We'll unify them with a single regex capturing optional prefix,
    # optional dt_value, and quantile.
    #   pattern:
    #   ^(?:([^_]+)_)?     # optional prefix (group 1), with underscore
    #      (?:(\d+))?_?    # optional dt_value (group 2), with optional underscore
    #      q([\d.]+)$      # quantile (group 3)
    #
    # Note: We allow prefix or dt_value to be absent. One or both might appear.

    pattern = re.compile(
        r'^(?:([^_]+)_)?(?:(\d+))?_?q([\d\.]+)$'
    )
    # We'll store columns in a structure:
    #   col_info[col_name] = (prefix, dt_val, quant_str)
    # prefix or dt_val can be None if absent
    col_info = {}
    for col in df.columns:
        m = pattern.match(col)
        if m:
            pref, dt_val, q_str = m.groups()
            # If user provided a specific value_prefix, filter by it
            if value_prefix is not None and pref is not None:
                if pref != value_prefix:
                    continue
            col_info[col] = (pref, dt_val, float(q_str))

    if not col_info:
        raise ValueError(
            "No columns match the recognized patterns. "
            "No q_cols provided either."
        )

    # Extract sets of dt_val and quant
    dt_vals_found = set()
    quants_found = set()
    for (pref, dt_val, q_val) in col_info.values():
        if dt_val is not None:
            dt_vals_found.add(dt_val)
        quants_found.add(q_val)

    # If user gave dt_values, we keep only those dt_val in dt_vals_found
    # If dt_name is None or if dt_vals_found is empty => single dimension
    # else => 2D
    if dt_values is not None:
        dt_values = [str(dv) for dv in dt_values]  # unify with string dt_val
        dt_vals_found = dt_vals_found.intersection(dt_values)

    dt_vals_found = sorted(dt_vals_found, key=lambda x: int(x)) \
                    if dt_vals_found else []
    quants_found = sorted(quants_found)

    # If user provides quantiles, we only keep those
    if quantiles is not None:
        qf = set(float(q) for q in quantiles)
        quants_found = [q for q in quants_found if q in qf]
    if not quants_found:
        raise ValueError("No matching quantiles found.")

    # 2D scenario if dt_vals_found has more than 0 elements 
    # and dt_name is not None.
    if dt_name and dt_vals_found:
        _plot_2d(
            df, x_col, y_col,
            col_info, dt_name, dt_vals_found,
            quants_found, value_prefix,
            cmap, s, alpha,
            cbar_orientation, cbar_fraction,
            cbar_pad, figsize, savefig,
            dpi, axis_off, reverse, 
            show_grid, grid_props
        )
    else:
        # 1D scenario for quantiles only
        _plot_1d(
            df, x_col, y_col,
            col_info, quants_found,
            value_prefix, cmap, s, alpha,
            cbar_orientation, cbar_fraction,
            cbar_pad, figsize,
            savefig, dpi, axis_off,
            reverse,show_grid, 
            grid_props
        )


def _plot_1d(df, x_col, y_col, col_info, quants_found, 
             value_prefix, cmap, s, alpha,
             cbar_orientation, cbar_fraction,
             cbar_pad, figsize, savefig, dpi,
             axis_off, reverse, show_grid, grid_props
             ):
    # We have only quantile dimension => one row or one column

    # Build col_name => (prefix, dt_val, q_val) reverse lookup
    # But we only care about q_val here
    q2col = {}
    for c, (p, d, q_val) in col_info.items():
        if q_val in quants_found and (d is None or d == ''):
            q2col[q_val] = c
        # If there's a dt_val, it won't be used in 1D scenario
        if q_val in quants_found and d is not None:
            # This means there's a dt_val, so skip
            pass

    # Some columns may not exist if the user data had dt_val but dt_name=None
    # We'll collect only those that have d is None
    # If none found, fallback to "any dt_val"? 
    valid_cols = [q2col[q] for q in quants_found if q in q2col]

    if not valid_cols:
        # fallback: if all columns have dt_val, we can't do 1D
        # So let's pick dt_val=some minimal or something
        any_col = {}
        for c, (p, d, q_val) in col_info.items():
            if q_val in quants_found:
                # We'll store first occurrence
                if q_val not in any_col:
                    any_col[q_val] = c
        if not any_col:
            raise ValueError("No columns available for 1D scenario.")
        q2col = any_col
        valid_cols = [q2col[q] for q in quants_found if q in q2col]

    n = len(quants_found)
    if reverse:
        n_rows, n_cols = 1, n
    else:
        n_rows, n_cols = n, 1

    if figsize is None:
        if reverse:
            figsize = (4 * n_cols, 4)
        else:
            figsize = (4, 4 * n_rows)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        constrained_layout=True
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    # Compute global min/max
    vals = df[valid_cols].values.ravel()
    overall_min, overall_max = np.nanmin(vals), np.nanmax(vals)

    for i, q in enumerate(quants_found):
        if q not in q2col:
            continue
        col_name = q2col[q]

        sub = df[[x_col, y_col, col_name]].dropna()

        if reverse:
            ax = axes[0, i]
        else:
            ax = axes[i, 0]

        sc = ax.scatter(
            sub[x_col],
            sub[y_col],
            c=sub[col_name],
            cmap=cmap,
            s=s,
            alpha=alpha,
            norm=plt.Normalize(vmin=overall_min, vmax=overall_max)
        )
        ax.set_title(f"q={q}", fontsize=10)
        if show_grid: 
            ax.grid(True, **grid_props)
        else: 
            ax.grid (False) 
            
        if axis_off:
            ax.axis('off')

        cbar = fig.colorbar(
            sc, ax=ax, orientation=cbar_orientation,
            fraction=cbar_fraction, pad=cbar_pad
        )
        cbar.set_label('Value', fontsize=10)

    if savefig:
        fig.savefig(savefig, dpi=dpi)
    else:
        plt.show()


def _plot_2d(df, x_col, y_col, col_info, dt_name, dt_vals_found,
             quants_found, value_prefix, cmap, s, alpha,
             cbar_orientation, cbar_fraction, cbar_pad,
             figsize, savefig, dpi, axis_off, reverse, 
             show_grid, grid_props):
    """
    2D scenario: dt_values x quantiles => grid
    If reverse=False, rows=quantiles, cols=dates
    If reverse=True, rows=dates, cols=quantiles
    """

    # Build dict: (dt_val, q_val) -> col_name
    dtq2col = {}
    for c, (p, d, q_val) in col_info.items():
        if d in dt_vals_found and q_val in quants_found:
            dtq2col[(d, q_val)] = c

    # If user provided dt_values externally, ensure they are strings
    # sorted. We'll only use the intersection dt_vals_found for plotting
    dt_vals_found = sorted(dt_vals_found, key=lambda x: int(x)) \
                    if dt_vals_found else []

    if reverse:
        # rows=dt, cols=q
        n_rows = len(dt_vals_found)
        n_cols = len(quants_found)
    else:
        # rows=q, cols=dt
        n_rows = len(quants_found)
        n_cols = len(dt_vals_found)

    if figsize is None:
        figsize = (4 * n_cols, 4 * n_rows)

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

    # Gather all relevant columns
    used_cols = []
    for dt_val in dt_vals_found:
        for q in quants_found:
            if (dt_val, q) in dtq2col:
                used_cols.append(dtq2col[(dt_val, q)])
    used_cols = list(set(used_cols))
    vals = df[used_cols].values.ravel()
    overall_min, overall_max = np.nanmin(vals), np.nanmax(vals)

    for r_idx, row_key in enumerate(quants_found if not reverse else dt_vals_found):
        for c_idx, col_key in enumerate(dt_vals_found if not reverse else quants_found):
            ax = axes[r_idx, c_idx]

            if not reverse:
                # row_key=quantile, col_key=dt_val
                dt_val = col_key
                q_val = row_key
            else:
                dt_val = row_key
                q_val = col_key

            pair = (dt_val, q_val)
            if pair not in dtq2col:
                ax.axis('off')
                ax.set_title(f"No col for {dt_val}, q={q_val}")
                continue

            col_name = dtq2col[pair]
            sub = df[[x_col, y_col, col_name]].dropna()

            sc = ax.scatter(
                sub[x_col],
                sub[y_col],
                c=sub[col_name],
                cmap=cmap,
                s=s,
                alpha=alpha,
                norm=plt.Normalize(vmin=overall_min, vmax=overall_max)
            )

            if not reverse:
                title_str = f"{dt_name}={dt_val}, q={q_val}"
            else:
                title_str = f"{dt_name}={row_key}, q={col_key}"
            ax.set_title(title_str, fontsize=9)
            
            if show_grid: 
                ax.grid(True, **grid_props)
            else: 
                ax.grid (False) 
                
            if axis_off:
                ax.axis('off')

            # Add colorbar only for last column in the row (non-reverse)
            # or last column if reversed? We'll keep it simple:
            if (not reverse and (c_idx == n_cols - 1)) \
               or (reverse and (c_idx == n_cols - 1)):
                cbar = fig.colorbar(
                    sc, ax=ax, orientation=cbar_orientation,
                    fraction=cbar_fraction, pad=cbar_pad
                )
                cbar.set_label('Value', fontsize=8)

    if savefig:
        fig.savefig(savefig, dpi=dpi)
    else:
        plt.show()


def _plot_from_qcols(
    df, x_col, y_col,
    dt_name, dt_values, q_cols,
    cmap, s, alpha,
    cbar_orientation, cbar_fraction,
    cbar_pad, figsize,
    savefig, dpi,
    axis_off, reverse, 
    show_grid, grid_props 
):
    """
    If q_cols is provided, interpret it. This can be:
      - 1D scenario: {q: col_name}
      - 2D scenario: {q: {date_val: col_name}}
    or {q: [col_names in same order as dt_values]}
    """
    # Distinguish 1D vs 2D by checking if any q maps to a dict or list
    # If all map to single str -> 1D
    # If we find at least one dict or list -> 2D
    is_2d = False
    for q, mapping in q_cols.items():
        if isinstance(mapping, (dict, list)):
            is_2d = True
            break

    # If 2D but dt_name is None, we can skip dt dimension -> error or fallback
    if is_2d and not dt_name:
        raise ValueError(
            "q_cols indicates a 2D structure, but no dt_name was provided."
        )

    # If 1D, handle single dimension
    if not is_2d:
        # 1D scenario
        quants = sorted(q_cols.keys())
        col_names = []
        for q in quants:
            if q_cols[q] not in df.columns:
                raise ValueError(f"Column {q_cols[q]} not found.")
            col_names.append(q_cols[q])

        # Global min/max
        vals = df[col_names].values.ravel()
        overall_min, overall_max = np.nanmin(vals), np.nanmax(vals)
        n = len(quants)
        if reverse:
            n_rows, n_cols = (1, n)
        else:
            n_rows, n_cols = (n, 1)

        if figsize is None:
            if reverse:
                figsize = (4 * n_cols, 4)
            else:
                figsize = (4, 4 * n_rows)

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=figsize,
            constrained_layout=True
        )
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = np.array([axes])
        elif n_cols == 1:
            axes = axes.reshape(n_rows, 1)

        for i, q in enumerate(quants):
            col_name = q_cols[q]
            sub = df[[x_col, y_col, col_name]].dropna()

            ax = axes[0, i] if (reverse) else axes[i, 0]

            sc = ax.scatter(
                sub[x_col],
                sub[y_col],
                c=sub[col_name],
                cmap=cmap,
                s=s,
                alpha=alpha,
                norm=plt.Normalize(vmin=overall_min, vmax=overall_max)
            )
            ax.set_title(f"q={q}", fontsize=9)
            
            if show_grid: 
                ax.grid(True, **grid_props)
            else: 
                ax.grid (False) 
                
            if axis_off:
                ax.axis('off')

            cbar = fig.colorbar(
                sc, ax=ax,
                orientation=cbar_orientation,
                fraction=cbar_fraction,
                pad=cbar_pad
            )
            cbar.set_label('Value', fontsize=9)

        if savefig:
            fig.savefig(savefig, dpi=dpi)
        else:
            plt.show()
        return

    # 2D scenario
    # Expect {q: {dt_val: col_name}} or {q: [col_names in order of dt_values]}
    quants = sorted(q_cols.keys())
    if dt_values is None:
        # glean dt_values from the dictionary keys if any
        dt_set = set()
        for q in quants:
            mapping = q_cols[q]
            if isinstance(mapping, dict):
                dt_set.update(mapping.keys())
        dt_values = sorted(list(dt_set))

    # Validate columns exist
    used_cols = []
    dt_values_str = [str(dv) for dv in dt_values]
    for q in quants:
        mapping = q_cols[q]
        if isinstance(mapping, dict):
            # dt_val => col_name
            for d_str in dt_values_str:
                if d_str in mapping:
                    c = mapping[d_str]
                    if c not in df.columns:
                        raise ValueError(
                            f"Column '{c}' not found in DataFrame."
                        )
                    used_cols.append(c)
        elif isinstance(mapping, list):
            if len(mapping) != len(dt_values_str):
                raise ValueError(
                    "Length of q_cols[q] does not match dt_values."
                )
            for c in mapping:
                if c not in df.columns:
                    raise ValueError(f"Column '{c}' not found in DataFrame.")
            used_cols.extend(mapping)

    vals = df[used_cols].values.ravel()
    overall_min, overall_max = np.nanmin(vals), np.nanmax(vals)

    if reverse:
        # rows = dt, cols = q
        n_rows, n_cols = len(dt_values_str), len(quants)
    else:
        # rows = q, cols = dt
        n_rows, n_cols = len(quants), len(dt_values_str)

    if figsize is None:
        figsize = (4 * n_cols, 4 * n_rows)

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

    for r_idx in range(n_rows):
        for c_idx in range(n_cols):
            ax = axes[r_idx, c_idx]
            if not reverse:
                q = quants[r_idx]
                d = dt_values_str[c_idx]
            else:
                d = dt_values_str[r_idx]
                q = quants[c_idx]

            mapping = q_cols[q]
            if isinstance(mapping, dict):
                if d not in mapping:
                    ax.axis('off')
                    ax.set_title(f"No data for {d}, q={q}")
                    continue
                col_name = mapping[d]
            else:
                # list scenario
                idx = dt_values_str.index(d)
                col_name = mapping[idx]

            sub = df[[x_col, y_col, col_name]].dropna()

            sc = ax.scatter(
                sub[x_col],
                sub[y_col],
                c=sub[col_name],
                cmap=cmap,
                s=s,
                alpha=alpha,
                norm=plt.Normalize(
                    vmin=overall_min,
                    vmax=overall_max
                )
            )
            ax.set_title(f"{dt_name}={d}, q={q}", fontsize=8)
            
            if show_grid: 
                ax.grid(True, **grid_props)
            else: 
                ax.grid (False) 
                
            if axis_off:
                ax.axis('off')

            # colorbar once per subplot or only last col?
            # we'll do once per subplot for clarity
            cbar = fig.colorbar(
                sc, ax=ax, orientation=cbar_orientation,
                fraction=cbar_fraction, pad=cbar_pad
            )
            cbar.set_label('Value', fontsize=8)

    if savefig:
        fig.savefig(savefig, dpi=dpi)
    else:
        plt.show()


@return_fig_or_ax
@default_params_plot(
    savefig=PlotConfig.AUTOSAVE('my_q.distributions_plot.png')
  )
@validate_params ({ 
    'df': ['array-like'], 
    'x_col': [str], 
    'y_col': [str], 
    'dt_col': [ str], 
    'quantiles': ['array-like', None], 
    'dt_values':['array-like', None], 
    'value_prefix': [str], 
    'q_cols': [dict, None], 
    })
@isdf 
def plot_quantile_distributions(
    df,
    x_col, 
    y_col,
    dt_col, 
    quantiles=None,
    dt_values=None,
    value_prefix='',
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
        `<y_col>`, `<dt_col>`, and either:

        1) Columns named according to the convention 
           ``<value_prefix>_<date>_q<quantile>``, or
        2) Explicit mappings in `q_cols`.
    x_col : str
        The column name representing the x-axis coordinate 
        (e.g., longitude).
    y_col : str
        The column name representing the y-axis coordinate 
        (e.g., latitude).
    dt_col : str
        The column representing dates/times. This can be integer (e.g., year),
        or datetime. If it's datetime, values are filtered by year 
        extracted from that datetime.
    quantiles : list of float, optional
        A list of quantiles (e.g., [0.1, 0.5, 0.9]). If None, the function
        attempts to detect quantiles from columns if ``q_cols`` is also 
        None. If detection fails, a ValueError is raised.
    dt_values : list, optional
        List of date values to plot. If None, the function infers date 
        values from `df`. If `dt_col` is integer, they are considered 
        as years. If `dt_col` is datetime, the year part is extracted.
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
    - The filtering step for datetime `dt_col` extracts the year 
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

    >>> # Initialize the DataFrame columns
    >>> data = {
    ...    'longitude': longitudes,
    ...    'latitude': latitudes
    ... }
    >>> # Add the 'year' column, repeating for each point
    >>> data['year'] = np.repeat(years, num_points)  # Repeat years for each point
    >>> for year in years:
    ...     for q in quantiles:
    ...         q_col = f'predicted_subsidence_{year}_q{q}'
    ...         # Generate predicted subsidence value for each quantile
    ...         data[q_col] = np.random.uniform(0, 50, num_points) *\
    ...    (1 + np.random.uniform(-0.1, 0.1))

    >>> df = pd.DataFrame(data)
    >>> df.head()

    This `df` will have columns like:
    `'longitude', 'latitude', 'year', 
    'predicted_subsidence_2024_q0.1', 'predicted_subsidence_2024_q0.5', 
    'predicted_subsidence_2024_q0.9', 'predicted_subsidence_2025_q0.1', 
    'predicted_subsidence_2025_q0.5', 'predicted_subsidence_2025_q0.9'`.


    >>> from gofast.plot.q import plot_quantile_distributions
    >>> # Automatically detect quantiles and dates:
    >>> plot_quantile_distributions(
    ...     df,
    ...     x_col='longitude',
    ...     y_col='latitude',
    ...     dt_col='year',
    ...     value_prefix='predicted_subsidence'
    ... )

    Or specifying quantiles:
    >>> plot_quantile_distributions(
    ...     df,
    ...     x_col='longitude',
    ...     y_col='latitude',
    ...     dt_col='year',
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
    ...     dt_col='year',
    ...     quantiles=[0.1,0.5,0.9],
    ...     dt_values=[2024,2025],
    ...     q_cols=q_map
    ... )

    See Also
    --------
    gofast.plot.q.plot_qdist: More robust quantile plot.
    matplotlib.pyplot.scatter : Scatter plot generation.
    matplotlib.pyplot.hexbin : Hexbin plot generation.
    seaborn.kdeplot : For density visualization.

    References
    ----------
    .. [1] Rosenblatt, M. "Remarks on some nonparametric estimates 
           of a density function." Ann. Math. Statist. 27 (1956), 
           832-837.
    """
    # Infer dt_values if not provided
    if dt_values is None:
        if pd.api.types.is_integer_dtype(df[dt_col]):
            dt_values = sorted(df[dt_col].unique())
        elif np.issubdtype(df[dt_col].dtype, np.datetime64):
            dt_values = pd.to_datetime(df[dt_col].unique()).sort_values()
        else:
            dt_values = sorted(df[dt_col].unique())

    dt_values_str = _extract_date_values_if_datetime(df, dt_col, dt_values)
    # If q_cols is None, we rely on quantiles and naming convention
    # or detect quantiles if quantiles is None.
    quantiles = columns_manager(quantiles, empty_as_none= True )
    if q_cols is None:
        # If quantiles is None, try to detect from columns
        if quantiles is None:
            # Attempt detection
            detected_quantiles = detect_quantiles_in( 
                df, col_prefix= value_prefix, 
                dt_value = dt_values_str , 
                return_types='q_val'
                )
            if not detected_quantiles: 
                # retry 
                detected_quantiles = _detect_quantiles_from_columns(
                    df, value_prefix, dt_values_str)

            if detected_quantiles is None:
                raise ValueError(
                    "No quantiles detected from columns."
                    " Please specify quantiles or q_cols."
                    )
            quantiles = detected_quantiles
        # Now we have quantiles, build column names
   
        all_cols = _build_column_names(
            df, value_prefix, quantiles, dt_values_str
            )
        if not all_cols: 
            #retry:
            all_cols = build_q_column_names(
                df, quantiles=quantiles, 
                dt_value=dt_values_str, 
                strict_match=False, 
                )
        if not all_cols:
            raise ValueError(
                "No matching columns found with given prefix, date, quantiles.")
    else:
        
        # q_cols provided. Let's assume q_cols is a dict:
        # {quantile: {date_str: column_name}}
        # or {quantile: [cols_in_same_order_as_dt_values_str]}
        # We must extract quantiles from q_cols keys
        quantiles_from_qcols = validate_q_dict(q_cols)  
        quantiles_from_qcols = sorted(q_cols.keys())
        if quantiles is None:
            quantiles = quantiles_from_qcols
        else:
            quantiles = validate_quantiles (quantiles, dtype=np.float64 )
            # Ensure that quantiles match q_cols keys
            if set(quantiles) != set(quantiles_from_qcols):
                raise ValueError(
                    "Quantiles specified do not match q_cols keys.")
        
        # Validate all columns exist
        all_cols = []
        for q in quantiles:
            mapping = q_cols[q]
            if isinstance(mapping, dict):
                # expect {date_str: col_name}
                for d_str in dt_values_str:
                    if d_str not in mapping:
                        continue
                    c = mapping[d_str]
                    if c not in df.columns:
                        raise ValueError(f"Column {c} not found in DataFrame.")
                    all_cols.append(c)
            else:
                # assume list parallel to dt_values_str
                if len(mapping) != len(dt_values_str):
                    raise ValueError("q_cols mapping length does not match dt_values.")
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
            df, x_col, y_col, dt_col, 
            quantiles, dt_values_str, 
            q_cols, value_prefix,
            cmap, s, alpha, axis_off,
            overall_min, overall_max,
            cbar_orientation, cbar_fraction, 
            cbar_pad, figsize, dpi, savefig
        )
        return 
     
    # Determine subplot grid size
    n_rows = len(quantiles)
    n_cols = len(dt_values)

    if figsize is None:
        figsize = (5 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=figsize,
        constrained_layout=True
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
        for j, d in enumerate(dt_values_str):
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
                    # find index of d in dt_values_str
                    idx_date = dt_values_str.index(d)
                    subset_col = mapping[idx_date]

            subset = df[[x_col, y_col, dt_col, subset_col]].dropna()

            # Filter subset based on dt_col and d
            if np.issubdtype(df[dt_col].dtype, np.datetime64):
                # convert d to int year
                year_int = int(d)
                subset = subset[subset[dt_col].dt.year == year_int]
            else:
                # Attempt to convert d to int if possible
                try:
                    val = int(d)
                except ValueError:
                    val = d
                subset = subset[subset[dt_col] == val]

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

            if np.issubdtype(df[dt_col].dtype, np.datetime64):
                title_str = f"Quantile={q}, Year={d}"
            else:
                title_str = f"Quantile={q}, {dt_col}={d}"
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

def _extract_date_values_if_datetime(df, dt_col, dt_values):
    """
    Extract date values as strings suitable for column naming if dt_col 
    is datetime. If dt_col is datetime, convert them to year strings, 
    otherwise return them as they are.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the dt_col.
    dt_col : str
        The name of the date column in df.
    dt_values : list
        A list of date values inferred or provided.

    Returns
    -------
    list
        A list of string representations of the date values.
    """
    # Convert datetime if needed
    # For datetime, dt_values are datetimes; might want to format them
    # as years or something else depending on the scenario
    # Extract dt_values as strings if datetime
    
    if np.issubdtype(df[dt_col].dtype, np.datetime64):
        # If datetime, convert each date to its year string
        return [str(pd.to_datetime(d).year) for d in dt_values]
    else:
        # Otherwise just convert to string
        return [str(d) for d in dt_values]


def _build_column_names(df, value_prefix, quantiles, dt_values_str):
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
    dt_values_str : list
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
    # If dt_col is datetime, we might extract year or something else
    
    all_cols = []
    for q in quantiles:
        for d_str in dt_values_str:
            col_name = f'{value_prefix}_{d_str}_q{q}'
            if col_name in df.columns:
                all_cols.append(col_name)
    return all_cols

def _detect_quantiles_from_columns(df, value_prefix, dt_values_str):
    """
    Attempt to detect quantiles by scanning the DataFrame columns that match
    the pattern f'{value_prefix}_{date}_q...' for each date in dt_values_str.
    Extract the quantiles from these column names.

    Parameters
    ----------
    df : pandas.DataFrame
    value_prefix : str
        The value prefix expected in column names.
    dt_values_str : list
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
            for d_str in dt_values_str:
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
    df, x_col, y_col, dt_col,
    quantiles, dt_values_str, q_cols, value_prefix,
    cmap, s, alpha, axis_off,
    overall_min, overall_max, cbar_orientation,
    cbar_fraction, cbar_pad,
    figsize, dpi, savefig
):
    # Reversed mode: quantiles in columns, dates in rows
    n_rows = len(dt_values_str)
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

    for j, d in enumerate(dt_values_str):
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
                    idx_date = dt_values_str.index(d)
                    subset_col = mapping[idx_date]

            subset = df[[x_col, y_col, dt_col, subset_col]].dropna()

            # Filter subset based on dt_col and d
            if np.issubdtype(df[dt_col].dtype, np.datetime64):
                year_int = int(d)
                subset = subset[subset[dt_col].dt.year == year_int]
            else:
                try:
                    val = int(d)
                except ValueError:
                    val = d
                subset = subset[subset[dt_col] == val]

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

            if np.issubdtype(df[dt_col].dtype, np.datetime64):
                title_str = f"Quantile={q}, Year={d}"
            else:
                title_str = f"Quantile={q}, {dt_col}={d}"
            ax.set_title(title_str, fontsize=10)

            if axis_off:
                ax.axis('off')

        # Add colorbar for the last column in each row
        # Actually, we add one per row:
            # better to add at the end of each row?
        # But here symmetrical to normal, we do once per row
        cbar = fig.colorbar(
            scatter,
            ax=axes[j, -1] if n_cols > 1 else axes[j, 0],
            orientation=cbar_orientation,
            fraction=cbar_fraction,
            pad=cbar_pad,
        )
        cbar.set_label('Value', fontsize=10)

    if savefig:
        fig.savefig(savefig, dpi=dpi)
    plt.show()
