# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Provides comparison plots between multiple datasets. 
"""
from itertools import cycle
import warnings 
from typing import List, Optional, Union, Tuple 
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import numpy as np 
import pandas as pd
import seaborn as sns

from sklearn.metrics import mean_squared_error, confusion_matrix

from ..core.checks import are_all_frames_valid
from ..core.handlers import columns_manager 
from ..core.plot_manager import ( 
    set_axis_grid, 
    default_params_plot, 
    is_valid_kind 
)
from ..utils.ts_utils import to_dt 
from ..utils.validator import parameter_validator
from ._config import PlotConfig
from .utils import select_hex_colors 

HAS_TQDM = True
try:
    from tqdm import tqdm
except ImportError:
    HAS_TQDM = False

__all__ = [
    'plot_feature_trend', 
    'plot_density', 
    'plot_prediction_comparison', 
    'plot_error_analysis', 
    'plot_trends', 
    'plot_variability', 
    'plot_factor_contribution', 
    'plot_comparative_bars', 
    'plot_line_graph', 
    ]

     
@default_params_plot(
    savefig=PlotConfig.AUTOSAVE('my_feature_trend_plot.png')
)
def plot_feature_trend(
    *dfs,
    feature: str,
    dt_col: str,
    target_col: str ,
    labels: Optional[List[str]] = None,
    figsize: tuple = (12, 6),
    title: Optional[str] = None,
    xlabel: str = 'Date/Time',
    ylabel_feature: str =None, # 'Groundwater Level (GWL)',
    ylabel_target: str = None, # 'Subsidence',
    colors: Optional[List[str]] = None,
    primary_style: str = '-',
    secondary_style: str = '--',
    show_grid: bool = True,
    grid_props: Optional[dict] = None,
    verbose: int = 0,
    savefig: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """
    Plot a specified `feature` and a corresponding `target_col` trend
    over time from one or more DataFrames, aggregating both by
    ``dt_col`` and visualizing mean values on separate y-axes. 
    
    Parameters
    ----------
    dfs : list of pandas.DataFrame
        One or more DataFrames, each containing columns
        ``feature`` and ``target_col``, as well as the date/time
        column ``dt_col``. If ``dt_col`` is not present, the
        DataFrame index must be a datetime type.
    feature : str
        Name of the column to be plotted on the primary y-axis.
    dt_col : str
        Name of the date/time column used for grouping and
        labeling the x-axis. If absent, the DataFrame index is
        assumed to be datetime.
    target_col : str
        Name of the target column to be plotted on the twin y-axis.
    labels : list of str, optional
        Custom labels for each DataFrame. Defaults to
        ``["Dataset 1", "Dataset 2", ...]`` if not provided.
    figsize : tuple of float, optional
        Figure size in inches (width, height). Default is (12, 6).
    title : str, optional
        Main title of the plot. If None, defaults to
        "Temporal Trend: feature vs target_col".
    xlabel : str, optional
        Label for the x-axis. Default is "Date/Time".
    ylabel_feature : str, optional
        Label for the primary y-axis. If None, defaults to
        the `feature` name.
    ylabel_target : str, optional
        Label for the twin y-axis. If None, defaults to
        the `target_col` name.
    colors : list of str, optional
        List of colors for plotting each DataFrame. If None,
        Matplotlib's default color cycle is used.
    primary_style : str, optional
        Matplotlib linestyle for the feature plot. Default is "-".
    secondary_style : str, optional
        Matplotlib linestyle for the target plot. Default is "--".
    show_grid : bool, optional
        Whether to display grid lines on both y-axes. Default is True.
    grid_props : dict, optional
        Keyword arguments for grid configuration, e.g.
        ``{"linestyle": ":", "alpha": 0.7}``.
    verbose : int, optional
        Verbosity level (0 = silent, 1 = basic info, 2 = details).
        Default is 0.
    savefig : str, optional
        File path for saving the figure. If None, the figure
        is not saved. Default is None.
    
    Returns
    -------
    matplotlib.figure.Figure
        The Matplotlib Figure containing the trend plot.
    
    Notes
    -----
    - The function calls `are_all_frames_valid` internally to ensure
      that each DataFrame has all required columns or a datetime
      index.
    - It groups each DataFrame by the specified ``dt_col`` (or by
      the datetime index if no ``dt_col`` is present) and computes
      the mean of both `feature` and `target_col` within each group.
    - Separate y-axes are used for `feature` (left axis) and
      `target_col` (right axis).
    - `set_axis_grid` is used to manage grid properties across both
      y-axes.
    
    This function can help compare the evolution of a `feature` (e.g.,
    groundwater level) against a `target_col` (e.g., subsidence).
    
    .. math::
       X_{\\text{mean}}(t) = \\frac{1}{n} \\sum_{i=1}^{n} X_i(t)
    
    where :math:`X_{\\text{mean}}(t)` represents the mean value of the
    time-series data points (either the chosen feature or the target)
    at time :math:`t`, and :math:`n` is the number of records within
    that temporal grouping.
    
    Examples
    --------
    >>> from gofast.plot.comparison import plot_feature_trend
    >>> import pandas as pd
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> np.random.seed(42)
    >>>
    >>> # Create sample data for demonstration
    >>> df1 = pd.DataFrame({
    ...     "time": pd.date_range(
    ...         "2020-01-01", periods=50, freq="D"
    ...     ),
    ...     "feature_col": np.random.rand(50) * 10,
    ...     "target_col": np.random.rand(50) * 5,
    ... })
    >>> df2 = pd.DataFrame({
    ...     "time": pd.date_range(
    ...         "2020-02-01", periods=50, freq="D"
    ...     ),
    ...     "feature_col": np.random.rand(50) * 8,
    ...     "target_col": np.random.rand(50) * 3,
    ... })
    >>>
    >>> # Plot the feature vs. target trend
    >>> fig = plot_feature_trend(
    ...     df1, df2,
    ...     feature="feature_col",
    ...     dt_col="time",
    ...     target_col="target_col",
    ...     title="Comparative Feature vs Target Trend",
    ...     figsize=(12, 6)
    ... )
    >>> plt.show()
    
    See Also
    --------
    are_all_frames_valid : Ensures valid columns or a datetime
        index in each DataFrame.
    set_axis_grid : Configures grid lines for the specified axis
        or axes.
    
    References
    ----------
    .. [1] Doe, J., & Smith, A. (2022). Data Visualization
           Techniques for Time-series Analysis. Journal of
           Data Science, 10(3), 45-60.
    """


    # Validate DataFrames
    dfs = are_all_frames_valid(*dfs, ops='validate')

    # Ensure each df has the required columns
    for i, df in enumerate(dfs):
        if feature not in df.columns:
            raise ValueError(
                f"DataFrame {i} missing feature '{feature}'"
            )
        if target_col not in df.columns:
            raise ValueError(
                f"DataFrame {i} missing target '{target_col}'"
            )
        if (dt_col not in df.columns
           and not pd.api.types.is_datetime64_any_dtype(df.index)):
            raise ValueError(
                f"DataFrame {i}: no '{dt_col}' column "
                f"and index not datetime"
            )

    # Prepare default or user grid settings
    default_grid   = {'linestyle': ':', 'alpha': 0.7}
    grid_params    = grid_props or default_grid
    processed_data = []

    # Optionally show progress
    iterator = enumerate(dfs)
    if HAS_TQDM:
        iterator = tqdm(
            enumerate(dfs),
            total = len(dfs),
            desc  = "Processing datasets"
        )
    elif verbose >= 1:
        print(f"Processing {len(dfs)} datasets")

    # Group each df by dt_col if present, or use index
    for i, df in iterator:
        if not HAS_TQDM and verbose >= 2:
            print(f"Processing dataset {i+1}/{len(dfs)}")

        time_ref = (
            df[dt_col]
            if dt_col in df.columns
            else df.index.to_series().dt.year
        )
        grouped = (
            df.groupby(time_ref)
              .agg({feature:'mean', target_col:'mean'})
              .reset_index()
        )
        processed_data.append(grouped)

    # Build default or extended labels if needed
    if labels is None:
        labels = [
            f"Dataset {i+1}" for i in range(len(dfs))
        ]
    else:
        labels = list(labels) + [
            f"Dataset {i+1}" for i in range(len(labels), len(dfs))
        ][:len(dfs)]

    # Create figure + axis, then a twin axis
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    # Cycle colors across plots
    color_cycler = cycle(
        colors or plt.rcParams['axes.prop_cycle'].by_key()['color']
    )

    # Plot each dataset's feature/target trend
    for (data, lbl) in zip(processed_data, labels):
        current_color = next(color_cycler)

        # Plot the feature on primary axis
        ax1.plot(
            data[dt_col],
            data[feature],
            color     = current_color,
            linestyle = primary_style,
            marker    = 'o',
            label     = f"{lbl} {feature}"
        )
        # Plot the target on secondary axis
        ax2.plot(
            data[dt_col],
            data[target_col],
            color     = current_color,
            linestyle = secondary_style,
            marker    = 'x',
            label     = f"{lbl} {target_col}"
        )

    # Configure axis labels
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel_feature or feature, color='k')
    ax2.set_ylabel(ylabel_target or target_col , color='k')

    # If user wants grid lines
    set_axis_grid(
        [ax1, ax2], show_grid, grid_params
    )
    # if show_grid:
    #     ax1.grid(True, **grid_params)
    #     ax2.grid(True, **grid_params)

    # Gather legend labels from both axes
    lines   = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    line_lb = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(
        lines,
        line_lb,
        loc            = 'upper center',
        bbox_to_anchor = (0.5, -0.15),
        ncol           = 2
    )

    # Apply main title
    plt.title(
        title or f"Temporal Trend: {feature} vs {target_col}"
    )
    plt.tight_layout()

    # Optionally save figure
    if savefig:
        plt.savefig(savefig, bbox_inches='tight')

    # Verbosity
    if verbose >= 1:
        print(f"Plot generated with {len(dfs)} datasets")

    return fig

@default_params_plot(
    savefig=PlotConfig.AUTOSAVE('my_compararive_bars_plot.png')
)
def plot_comparative_bars(
    *dfs: pd.DataFrame,
    target_col: str,
    dt_col: Optional[str] = None,
    freq: str = 'Y',
    labels: Optional[List[str]] = None,
    agg_func: str = 'mean',
    error_func: Optional[str] = 'std',
    figsize: tuple = (12, 7),
    title: str = 'Comparative Analysis',
    xlabel: str = 'Study Period',
    ylabel: str = 'Rate',
    colors: Optional[List[str]] = None,
    rotation: int = 45,
    show_grid: bool = True,
    grid_props: dict = None,
    savefig: Optional[str] = None,
    verbose: int = 0,
    **kws
) -> plt.Figure:
    r"""
    Creates comparative bar charts over time for multiple
    datasets. The method `plot_comparative_bars` allows
    aggregating temporal data (resampled by `freq`) and
    displays results side by side in grouped bar charts.

    Parameters
    ----------
    dfs : pd.DataFrame
        One or more DataFrames to be plotted
        comparatively. Each DataFrame must contain the
        `target_col` and optionally a datetime column
        `dt_col` if time-based indexing is desired.

    target_col : str
        The name of the column to aggregate and plot.
        Must be numeric in all DataFrames.

    dt_col : str, optional
        The name of a datetime column, if present.
        If omitted, the index is checked for a datetime
        type; if found, it is used directly.

    freq : str, default='Y'
        The resampling frequency string (e.g., `'M'` for
        monthly, `'Y'` for yearly). Follows pandas'
        resample rules.

    labels : list of str, optional
        Labels corresponding to each DataFrame.
        Automatically generated if fewer than the
        number of DataFrames are provided.

    agg_func : str, default='mean'
        The aggregation function applied to each
        resampling bin. Common examples include
        `'mean'`, `'sum'`, `'max'`, etc.

    error_func : str, optional
        An optional aggregation function for error bars.
        By default, standard deviation (`'std'`) is used.
        Set to None to omit error bars.

    figsize : tuple of (float, float), default=(12, 7)
        The size of the resulting figure
        (width, height) in inches.

    title : str, default='Comparative Analysis'
        The main title displayed above the chart.

    xlabel : str, default='Study Period'
        The x-axis label.

    ylabel : str, default='Rate'
        The y-axis label.

    colors : list of str, optional
        A list of color names or codes used for each
        DataFrame. If not provided, a default cycle
        is used.

    rotation : int, default=45
        Rotation angle (in degrees) for x-axis labels.

    show_grid : bool, default=True
        If True, displays grid lines on the chart.

    grid_props : dict, optional
        Dictionary of grid properties (e.g., `'linestyle'`,
        `'alpha'`). Passed to the function that handles
        axis grid styling.

    savefig : str, optional
        The filepath to save the figure. If None,
        no figure is saved.

    verbose : int, default=0
        The verbosity level. A value > 0 enables a
        progress bar (via `tqdm`) if installed.

    **kws
        Additional keyword arguments passed to
        `ax.bar`, allowing further customization
        of the plotted bars.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object containing the
        comparative bar chart.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from gofast.plot.comparison import plot_comparative_bars
    >>> # Create synthetic data
    >>> np.random.seed(42)
    >>> df1 = pd.DataFrame({
    ...     'date': pd.date_range('2020-01-01', periods=12, freq='M'),
    ...     'value': np.random.randn(12).cumsum() + 10
    ... })
    >>> df2 = pd.DataFrame({
    ...     'date': pd.date_range('2020-01-01', periods=12, freq='M'),
    ...     'value': np.random.randn(12).cumsum() + 5
    ... })
    >>> # Usage of `plot_comparative_bars`
    >>> fig = plot_comparative_bars(
    ...     df1, df2,
    ...     target_col='value',
    ...     dt_col='date',
    ...     freq='M',
    ...     labels=['Region A', 'Region B'],
    ...     agg_func='mean',
    ...     error_func='std',
    ...     title='Monthly Comparative Analysis',
    ...     ylabel='Mean Value',
    ...     rotation=30,
    ...     verbose=1
    ... )

    Notes
    -----
    Internally, `plot_comparative_bars` converts the
    specified `dt_col` into a datetime index (if not
    already), resamples at frequency `freq`, computes
    `agg_func` and optionally `error_func`, then plots
    grouped bars for each dataset side by side. Error
    bars can be omitted by setting ``error_func=None``.

    .. math::
        \text{agg}(t)
        = \text{agg\_func}(\{ x_i : x_i \in \text{period} \})

    where :math:`\text{agg\_func}` is a summary statistic
    (e.g., mean, max) applied to all data points :math:`x_i`
    within each resampling period :math:`t`. An optional
    error function (e.g., standard deviation) can also be
    used to plot error bars.
        
    See Also
    --------
    plot_comparative_bars : This method can be used
        with various resampling intervals for daily,
        monthly, or yearly analyses.
    pandas.DataFrame.resample : Resamples time-series
        data at a specified frequency.
    matplotlib.pyplot.bar : Underlying bar plot
        functionality.

    References
    ----------
    .. [1] J. D. Hunter. *Matplotlib: A 2D Graphics
       Environment.* Computing in Science & Engineering,
       9(3), 90-95, 2007.
    """
    dfs = are_all_frames_valid(
        *dfs, df_only=True, 
        ops='validate'
   )
    # Validate that at least one DataFrame is provided
    if not dfs:
        raise ValueError(
            "At least one DataFrame must be provided"
        )

    # Copy DataFrames to avoid mutation
    dfs_copy = [df.copy() for df in dfs]

    # Ensure target column is in each DataFrame
    for i, df in enumerate(dfs_copy):
        if target_col not in df.columns:
            raise ValueError(
                f"DataFrame {i} missing target "
                f"column: '{target_col}'"
            )

    # Prepare labels for each dataset
    labels = (
        labels or
        [f'Dataset {i+1}' for i in range(len(dfs))]
    )
    labels = (
        labels[:len(dfs)] +
        [f'Dataset {i+1}' for i in range(
            len(labels),
            len(dfs)
        )]
    )

    # Setup the figure and axes
    fig, ax = plt.subplots(
        figsize=figsize
    )
    width      = 0.8 / len(dfs)
    x_ticks    = None
    all_indices = []

    # Create a color cycle for plotting
    default_colors = (
        plt.rcParams['axes.prop_cycle']
        .by_key()['color']
    )
    color_cycle = cycle(
        colors or default_colors
    )

    # Possible progress bar
    iterator = enumerate(
        zip(dfs_copy, labels)
    )
    if HAS_TQDM and verbose > 0:
        iterator = tqdm(
            iterator,
            total=len(dfs),
            desc="Processing datasets"
        )

    # Main loop over DataFrames
    for idx, (df, label) in iterator:
        # Attempt time-based processing
        try:
            # Convert dt_col to datetime if provided
            if dt_col is not None:
                df = to_dt(
                    df,
                    dt_col=dt_col,
                    format=f'%{freq}',
                    error='raise'
                )
            # Determine time series
            time_series = (
                df[dt_col] if dt_col
                else df.index.to_series()
                if df.index.inferred_type
                   == 'datetime'
                else pd.RangeIndex(
                    start=0,
                    stop=len(df)
                )
            )
        except Exception as e:
            raise ValueError(
                f"Time conversion failed "
                f"for {label}: {str(e)}"
            )

        # Set index to time series
        df = df.set_index(
            pd.to_datetime(time_series)
        )

        # Aggregate (resample)
        agg_functions = [agg_func]
        if error_func:
            agg_functions.append(error_func)
        resampled = (
            df.resample(freq)[target_col]
              .agg(agg_functions)
        )

        # Keep track of indices for alignment
        all_indices.append(
            resampled.index
        )

        # Determine positions for bars
        positions = (
            np.arange(len(resampled))
            + idx * width
        )

        # Initialize or keep x_ticks
        positions = (
            np.arange(len(resampled))
            + idx * width
        )
        x_ticks = (
            positions if x_ticks
            is not None else
            ax.get_xticklabels()
        )

        # Plot the bars
        ax.bar(
            positions,
            resampled[agg_func],
            width=width,
            label=label,
            color=next(color_cycle),
            yerr=(
                resampled[error_func]
                if error_func else None
            ),
            error_kw={
                'capsize': 3
            },
            **kws
        )

    # Unite all indices for consistent ticks
    common_index = all_indices[0]
    for idx in all_indices[1:]:
        common_index = common_index.union(idx)

    # Final formatting
    ax.set(
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        xticks=x_ticks,
        xticklabels=[
            ts.strftime('%Y-%m')
            for ts in common_index
        ]
    )
    plt.xticks(rotation=rotation)

    # Place legend
    ax.legend(
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )
    plt.tight_layout()

    # Apply grid or style from manager
    set_axis_grid(
        ax,
        show_grid=show_grid,
        grid_props=grid_props
    )
    ax.set_axisbelow(True)

    # Save figure if path specified
    if savefig:
        plt.savefig(
            savefig,
            bbox_inches='tight',
            dpi=300
        )
        if verbose > 0:
            print(
                f"Saved figure to: {savefig}"
            )

    return fig

@default_params_plot(
    savefig=PlotConfig.AUTOSAVE('my_density_plot.png')
)
def plot_density(
    *dfs,
    density_col: str,
    target_col: str,
    labels: Optional[List[str]] = None,
    figsize: tuple = (10, 6),
    title: Optional[str] = None,
    xlabel: str = 'Density',
    ylabel: str = 'Rate',
    colors: Optional[List[str]] = None,
    markers: Optional[List[str]] = None,
    alpha: float = 0.6,
    edgecolors: str = 'w',
    show_grid: bool = True,
    grid_props: Optional[dict] = None,
    verbose: int = 0,
    savefig: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """
    Plot a scatter of `density_col` vs. `target_col` from one or more
    DataFrames, allowing quick visual assessment of how density values
    relate to the target rates. 
    
    Parameters
    ----------
    dfs : list of pandas.DataFrame
        One or more DataFrames containing columns
        ``density_col`` and ``target_col``. Each DataFrame is
        validated by `are_all_frames_valid`.
    density_col : str
        Name of the column representing density values to be
        plotted on the x-axis.
    target_col : str
        Name of the column representing target values to be
        plotted on the y-axis.
    labels : list of str, optional
        Custom labels for each DataFrame. Defaults to
        ``["Dataset 1", "Dataset 2", ...]`` if not provided.
    figsize : tuple of float, optional
        Width and height of the figure in inches.
        Default is (10, 6).
    title : str, optional
        Main title of the scatter plot. Defaults to
        "density_col vs target_col Correlation".
    xlabel : str, optional
        Label for the x-axis. Defaults to "Density".
    ylabel : str, optional
        Label for the y-axis. Defaults to "Rate".
    colors : list of str, optional
        A list of color names or codes to cycle through for
        each DataFrame. By default, Matplotlib's color cycle
        is used.
    markers : list of str, optional
        A list of marker styles to cycle through for each
        DataFrame. Default is ['o','s','D','^','v'].
    alpha : float, optional
        Opacity level of the scatter points, with 1.0 being fully
        opaque and 0.0 fully transparent. Default is 0.6.
    edgecolors : str, optional
        Color of the marker edges. Defaults to ``'w'`` (white).
    show_grid : bool, optional
        Whether to display grid lines on the plot.
        Defaults to True.
    grid_props : dict, optional
        Dictionary of Matplotlib grid properties (e.g.,
        ``{"linestyle": ":", "alpha": 0.4}``).
    verbose : int, optional
        Level of verbosity. 0 = silent, 1 = basic info,
        2 = detailed info. Default is 0.
    savefig : str, optional
        Path for saving the figure as an image file. If None,
        no file is saved. Default is None.
    
    Returns
    -------
    matplotlib.figure.Figure
        The Matplotlib Figure object containing the scatter plot.
    
    Notes
    -----
    - Each DataFrame is verified via `are_all_frames_valid` to ensure
      the required columns exist.
    - The function calls `set_axis_grid` to optionally apply grid
      settings to the axes.
    
    This function calls `are_all_frames_valid` to ensure each 
    DataFrame contains the necessary columns and leverages 
    `set_axis_grid` to configure grid lines on the plot.
    
    Examples
    --------
    >>> from gofast.plot.comparison import plot_density
    >>> import pandas as pd
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>>
    >>> # Generate random data for demonstration
    >>> np.random.seed(42)
    >>> df1 = pd.DataFrame({
    ...     "density": np.random.rand(50),
    ...     "target": np.random.rand(50) * 5
    ... })
    >>> df2 = pd.DataFrame({
    ...     "density": np.random.rand(50) + 0.5,
    ...     "target": (np.random.rand(50) * 4) + 1
    ... })
    >>>
    >>> # Plot the data
    >>> fig = plot_density(
    ...     df1, df2,
    ...     density_col="density",
    ...     target_col="target",
    ...     title="Density vs Target Demo",
    ...     verbose=1
    ... )
    >>> plt.show()
    
    See Also
    --------
    are_all_frames_valid : Ensures that each DataFrame contains
        the required columns and/or has a datetime index.
    set_axis_grid : Applies consistent grid settings to the
        given axis or axes.
    
    References
    ----------
    .. [1] Doe, J., & Lee, Q. (2021). Fundamentals of Data
           Scatter Visualization. Visualization Journal,
           14(2), 101-115.
    """
    # Validate all frames
    dfs = are_all_frames_valid(*dfs, ops='validate')

    # Ensure each df has the needed columns
    for i, df in enumerate(dfs):
        for col in [density_col, target_col]:
            if col not in df.columns:
                raise ValueError(
                    f"DataFrame {i} missing column '{col}'"
                )

    # Default markers, grid settings
    default_markers = ['o','s','D','^','v']
    default_grid    = {'linestyle':':', 'alpha':0.4}
    grid_params     = grid_props or default_grid

    # Create figure + axis
    fig, ax = plt.subplots(figsize=figsize)

    # Generate color/marker cycles
    color_cycle  = cycle(
        colors or plt.rcParams['axes.prop_cycle'].by_key()['color']
    )
    marker_cycle = cycle(markers or default_markers)

    # Extend or create dataset labels
    if labels is None:
        labels = [
            f"Dataset {i+1}" for i in range(len(dfs))
        ]
    else:
        labels = list(labels) + [
            f"Dataset {i+1}"
            for i in range(len(labels), len(dfs))
        ]

    # Optional progress with tqdm
    iterator = zip(dfs, labels, color_cycle, marker_cycle)
    if HAS_TQDM:
        iterator = tqdm(
            iterator,
            total=len(dfs),
            desc="Plotting datasets"
        )

    # Scatter each dataset
    for df, lbl, color, mark in iterator:
        if verbose >= 2:
            print(
                f"Plotting {lbl} "
                f"({density_col} vs {target_col})"
            )
        ax.scatter(
            x          = df[density_col],
            y          = df[target_col],
            c          = color,
            marker     = mark,
            edgecolors = edgecolors,
            alpha      = alpha,
            label      = lbl,
            **kwargs
        )

    # Axes labeling
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Show or hide grid
    set_axis_grid(ax, show_grid, grid_params)

    # Legend styling
    ax.legend(
        title          = 'Datasets',
        bbox_to_anchor = (1.05, 1),
        loc            = 'upper left'
    )

    # Title and layout
    plt.title(
        title or f"{density_col} vs {target_col} Correlation"
    )
    plt.tight_layout()

    # Optionally save to file
    if savefig:
        plt.savefig(
            savefig,
            bbox_inches='tight',
            dpi=300
        )

    # Verbosity info
    if verbose >= 1:
        print(
            f"\nCreated scatter plot with {len(dfs)} datasets"
        )

    return fig

@default_params_plot(
    savefig=PlotConfig.AUTOSAVE('my_line_graph_plot.png')
)
def plot_line_graph(
    *dfs: pd.DataFrame,
    feature: str,
    target_col: str,
    dt_col: Optional[str] = None,
    freq: str            = 'M',
    labels: Optional[
        List[str]
    ]                   = None,
    figsize: Tuple[
        int, int
    ]                   = (14, 7),
    title: str          = None,
    styles: List[str]   = None,
    markers: List[str]  = None,
    colors: Optional[
        List[str]
    ]                   = None,
    ylabel_feature: str = None,
    ylabel_target: str  = None,
    show_grid: dict     = None,
    grid_props: dict    = None,
    verbose: int        = 0,
    savefig: Optional[
        str
    ]                   = None,
    **kwargs
) -> plt.Figure:
    r"""
    Plots a line graph of two time-series columns using a
    dual-axis approach. The method `plot_line_graph` merges
    multiple DataFrames, resamples them by `freq`, and
    displays `<feature>` on the left y-axis and
    `<target_col>` on the right y-axis.

    Parameters
    ----------
    dfs : list of pd.DataFrame
        One or more DataFrames containing the columns
        `<feature>` and `<target_col>`. Each DataFrame is
        plotted in a distinct style or color.

    feature : str
        The column name of the first time-series to be
        shown on the left y-axis (e.g., a groundwater
        metric).

    target_col : str
        The column name of the second time-series to be
        shown on the right y-axis (e.g., a subsidence
        measure).

    dt_col : str, optional
        The column containing datetime information.
        If None, the index is assumed to be datetime.
        Otherwise, `<dt_col>` is converted to datetime.

    freq : str, default='M'
        A pandas-compatible resampling frequency
        specifying how data is aggregated over time
        (e.g., `'M'`=monthly, `'D'`=daily).

    labels : list of str, optional
        A list of labels for each DataFrame. Additional
        labels are auto-generated if fewer than the
        number of DataFrames are provided.

    figsize : tuple of (int, int), default=(14, 7)
        The figure size in inches (width, height).

    title : str, optional
        The main title displayed at the top of the plot.
        If None, no overall title is shown.

    styles : list of str, optional
        A list of line styles (e.g., `'-', '--', '-.'`)
        used in a cycle for each DataFrame.

    markers : list of str, optional
        A list of marker styles (e.g., `'o', 's', '^'`)
        used in a cycle for each DataFrame.

    colors : list of str, optional
        A list of hex or named colors. If None, default
        package colors are used.

    ylabel_feature : str, optional
        The y-axis label for `<feature>` (left axis).
        Defaults to the name of `<feature>`.

    ylabel_target : str, optional
        The y-axis label for `<target_col>` (right axis).
        Defaults to the name of `<target_col>`.

    show_grid : dict, optional
        Whether and how to show a grid on the main axis.
        For instance, `{'show': True}` or `{'show': True, 
        'axis': 'both'}`.

    grid_props : dict, optional
        Dictionary of grid style properties, e.g.,
        `{'alpha': 0.4, 'linestyle': '--'}`.

    verbose : int, default=0
        Verbosity level. A value > 0 shows progress
        bars (if tqdm is installed) and extra logs.

    savefig : str, optional
        A filepath to which the figure is saved. If None,
        the figure is not saved.

    **kwargs
        Additional keyword arguments passed to the
        underlying matplotlib plotting function(s).

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure containing the dual-axis
        line plot.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.plot.comparison import plot_line_graph
    >>> # Create two DataFrames for demonstration
    >>> df1 = pd.DataFrame({
    ...     'date': pd.date_range('2021-01-01', periods=5, freq='M'),
    ...     'GWL': [10, 9, 8, 7, 7],
    ...     'Subsidence': [0.2, 0.3, 0.4, 0.6, 0.9]
    ... })
    >>> df2 = pd.DataFrame({
    ...     'date': pd.date_range('2021-01-01', periods=5, freq='M'),
    ...     'GWL': [15, 14, 13, 13, 12],
    ...     'Subsidence': [0.1, 0.25, 0.35, 0.55, 0.8]
    ... })
    >>> # Plot them on dual axes
    >>> fig = plot_line_graph(
    ...     df1, df2,
    ...     feature='GWL',
    ...     target_col='Subsidence',
    ...     dt_col='date',
    ...     freq='M',
    ...     labels=['Site A', 'Site B'],
    ...     title='GWL vs. Subsidence'
    ... )

    Notes
    -----
    This function merges two distinct metrics (e.g.,
    a hydrologic parameter and a subsidence parameter)
    over time on separate y-axes. By default, each
    dataset is resampled to a common frequency before
    plotting. Markers, line styles, and colors can all
    be customized to distinguish multiple datasets.
    
    
    .. math::
        \begin{aligned}
        & Y_f(t) = \text{Resample}\bigl(\text{feature}, 
             \text{freq}\bigr), \\
        & Y_t(t) = \text{Resample}\bigl(\text{target\_col}, 
             \text{freq}\bigr),
        \end{aligned}

    where :math:`t` is the time index derived from
    `dt_col` or DataFrame index, and :math:`\text{freq}`
    is the resampling interval (e.g., monthly).
    

    See Also
    --------
    gofast.plot.comparison.plot_feature_trend:
        Plot a specified `feature` and a corresponding 
        `target_col` trend over time
    pandas.DataFrame.resample : Convenience method
        for frequency-based time-series resampling.
    matplotlib.pyplot.plot : Underlying function for
        line plotting.

    References
    ----------
    .. [1] J. D. Hunter. *Matplotlib: A 2D Graphics
       Environment.* Computing in Science &
       Engineering, 9(3), 90-95, 2007.
    """

    # Validate the DataFrames
    dfs = are_all_frames_valid(
        *dfs,
        ops='validate',
        df_only=True
    )
    if not dfs:
        raise ValueError(
            "At least one DataFrame must be provided"
        )

    # Ensure columns exist
    required_cols = [feature, target_col]
    for i, df in enumerate(dfs):
        missing = [
            col for col in required_cols
            if col not in df.columns
        ]
        if missing:
            raise ValueError(
                f"DataFrame {i} missing "
                f"columns: {missing}"
            )

    # Generate or extend labels
    labels = (
        labels or
        [f'Dataset {i+1}' for i in range(len(dfs))]
    )
    labels = (
        labels[:len(dfs)] +
        [f'Dataset {i+1}' for i in range(
            len(labels),
            len(dfs)
        )]
    )

    # Create the figure and dual axes
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    # Default line styles and markers
    styles  = styles  or ['-', '--', '-.', ':']
    markers = markers or ['o', 's', 'D', '^']

    # Prepare color generation
    d_colors = select_hex_colors(
        len(dfs),
        seed=42, 
    )
    # Handle user-provided colors
    colors = columns_manager(
        colors,
        empty_as_none=False
    )
    colors = colors + d_colors

    style_cycle  = cycle(styles)
    marker_cycle = cycle(markers)

    # Possibly show progress bar
    iterator = zip(dfs, labels)
    if HAS_TQDM and verbose > 0:
        iterator = tqdm(
            iterator,
            total=len(dfs),
            desc="Processing datasets"
        )

    for ix, (df, label) in enumerate(iterator):
        # Convert to datetime
        try:
            df = to_dt(
                df,
                dt_col=dt_col,
                format=f"%{freq}"
            )
            time_series = (
                pd.to_datetime(df[dt_col])
                if dt_col else
                pd.to_datetime(df.index)
            )
        except Exception as e:
            raise ValueError(
                f"Datetime conversion failed "
                f"for {label}: {str(e)}"
            )

        df = df.set_index(time_series).sort_index()

        # Resample
        resampled = df.resample(freq).agg({
            feature: 'mean',
            target_col: 'mean'
        })

        linestyle = next(style_cycle)
        marker    = next(marker_cycle)

        # Plot feature on left axis
        ax1.plot(
            resampled.index,
            resampled[feature],
            color=colors[ix],
            linestyle=linestyle,
            marker=marker,
            markersize=6,
            label=f'{label} {feature}',
            **kwargs
        )

        # Plot target_col on right axis
        ax2.plot(
            resampled.index,
            resampled[target_col],
            color=colors[ix],
            linestyle=linestyle,
            marker=marker,
            markersize=6,
            alpha=0.7,
            label=f'{label} {target_col}',
            **kwargs
        )

    ax1.set(
        title=title,
        xlabel='Time',
        ylabel=ylabel_feature or feature
    )
    ax2.set(
        ylabel=ylabel_target or target_col
    )

    # Merge legend entries
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=2
    )

    # Grid styling
    set_axis_grid(
        ax1,
        show_grid=show_grid,
        grid_props=grid_props
    )
    ax2.grid(False)

    fig.autofmt_xdate()

    # Save figure if path is provided
    if savefig:
        plt.savefig(
            savefig,
            bbox_inches='tight',
            dpi=300
        )
        if verbose > 0:
            print(
                f"Saved figure to: {savefig}"
            )

    return fig

@default_params_plot(
    savefig=PlotConfig.AUTOSAVE('my_factor_contrib_plot.png')
)
def plot_factor_contribution(
    *dfs,
    features: List[str],
    target_col: str,
    labels: Optional[List[str]] = None,
    kind: str = 'stacked',
    agg_func: str = 'mean',
    figsize: tuple = (12, 7),
    title: Optional[str] = None,
    colors: Optional[List[str]] = None,
    palette: str = 'viridis',
    show_values: bool = True,
    value_format: str = '.1f',
    show_target_line: bool = True,
    show_grid: bool=True, 
    grid_props: dict = None,
    verbose: int = 0,
    savefig: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    
    """
    Plot factor contributions of specified `features` relative to a
    `target_col` across one or more DataFrames, using a stacked bar or
    heatmap visualization. 
    
    Parameters
    ----------
    dfs : list of pandas.DataFrame
        One or more DataFrames, each containing the columns
        in ``features`` plus the column ``target_col``.
    features : list of str
        The names of the features (factors) to be aggregated.
    target_col : str
        The column name representing the target, used for
        comparison or reference against the sum of the features.
    labels : list of str, optional
        Custom labels for each DataFrame. Defaults to
        ``["Dataset 1", "Dataset 2", ...]`` if not provided.
    kind : {'stacked', 'heatmap'}, optional
        Type of plot to display. If ``'stacked'``, a stacked bar
        chart is produced. If ``'heatmap'``, a heatmap is
        produced. Default is ``'stacked'``.
    agg_func : str, optional
        Aggregation function to apply to features and target.
        Examples include "mean", "sum", etc. Default is "mean".
    figsize : tuple of float, optional
        Width and height of the Matplotlib figure in inches.
        Default is (12, 7).
    title : str, optional
        Main title of the plot. Defaults to
        "Contribution Analysis: <target_col>".
    colors : list of str, optional
        Custom colors for the stacked segments. Only used if
        `plot_type` is ``'stacked'``. If None, a color map is
        generated from the `palette`.
    palette : str, optional
        Name of the Matplotlib colormap to use when generating
        feature colors or a heatmap. Default is ``'viridis'``.
    show_values : bool, optional
        Whether to display numeric annotations of the feature
        contributions (stacked bar or heatmap cells). Default
        is True.
    value_format : str, optional
        Format specifier for numeric annotations (e.g. ``.1f``).
        Default is ``'.1f'``.
    show_target_line : bool, optional
        If True and `plot_type` is ``'stacked'``, overlays a
        dashed line representing the aggregated target.
        Default is True.
    show_grid : bool, optional
        Whether to display a grid on the plot. Default is True.
    grid_props : dict, optional
        Additional properties for the grid, e.g.
        ``{'axis': 'y', 'linestyle': ':', 'alpha': 0.4}``.
    verbose : int, optional
        Verbosity level (0 = silent, 1 = warning messages,
        2 = more info). Default is 0.
    savefig : str, optional
        File path to save the resulting figure. If None, no file
        is saved. Default is None.
    
    Returns
    -------
    matplotlib.figure.Figure
        The Matplotlib Figure object containing the factor
        contribution plot.
    
    Notes
    -----
    - Each input DataFrame is checked by `are_all_frames_valid`
      to ensure it contains the required columns.
    - When `plot_type` is ``'stacked'``, bars are stacked in the
      order of `features`, building from the bottom up.
    - Any discrepancy between the sum of the features and the
      target is signaled if it exceeds 10% of the target.
    
    
    It leverages the method `are_all_frames_valid` to ensure valid
    DataFrames and aggregates the selected features via a specified 
    `agg_func`, then optionally compares the sum of the features 
    to the target to detect large discrepancies.
    
    .. math::
       C_j = \\mathrm{agg\\_func}(X_j)
    
    where :math:`C_j` is the aggregated value of feature
    :math:`X_j`, computed using the user-selected aggregator
    (e.g. :math:`\\text{mean}`, :math:`\\text{sum}`).
    
    Examples
    --------
    >>> from gofast.plot.comparison import plot_factor_contribution
    >>> import pandas as pd
    >>> import numpy as np
    >>>
    >>> # Generate random data for demonstration
    >>> np.random.seed(0)
    >>> df1 = pd.DataFrame({
    ...     "f1": np.random.rand(100),
    ...     "f2": np.random.rand(100) + 0.5,
    ...     "target": (np.random.rand(100) * 2) + 1.5
    ... })
    >>> df2 = pd.DataFrame({
    ...     "f1": np.random.rand(80),
    ...     "f2": np.random.rand(80) * 1.2,
    ...     "target": np.random.rand(80) + 1
    ... })
    >>>
    >>> # Stacked bar example
    >>> fig = plot_factor_contribution(
    ...     df1,
    ...     df2,
    ...     features=["f1", "f2"],
    ...     target_col="target",
    ...     kind="stacked",
    ...     title="Feature Contribution vs Target",
    ...     agg_func="mean",
    ...     verbose=1
    ... )
    >>>
    >>> # Heatmap example
    >>> fig2 = plot_factor_contribution(
    ...     df1,
    ...     df2,
    ...     features=["f1", "f2"],
    ...     target_col="target",
    ...     plot_type="heatmap",
    ...     agg_func="sum",
    ...     title="Feature Contribution Heatmap",
    ...     show_values=True
    ... )
    
    See Also
    --------
    are_all_frames_valid : Confirms each DataFrame has the required
        columns or a datetime index.
    set_axis_grid : Applies consistent grid settings for axes.
    
    References
    ----------
    .. [1] Doe, J. & Wang, H. (2020). Analysis of Factor
           Contributions in Data Aggregates. Journal of
           Data Visualization, 12(3), 34-48.
    """

    dfs = are_all_frames_valid(*dfs, ops='validate')
    # Configure default plot style for uniform
    # visuals. 
    plt.style.use('seaborn-whitegrid')

    # For each DataFrame, ensure it has the specified
    # features plus target_col
    for i, df in enumerate(dfs):
        missing = [
            col for col in features + [target_col]
            if col not in df.columns
        ]
        if missing:
            raise ValueError(
                f"Dataframe {i} missing columns: {missing}"
            )
    
    # Extend or create dataset labels
    if labels is None:
        labels = [
            f"Dataset {i+1}" for i in range(len(dfs))
        ]
    else:
        labels = list(labels) + [
            f"Dataset {i+1}"
            for i in range(len(labels), len(dfs))
        ]

    # Aggregate the data per DataFrame, computing
    # user-chosen agg_func on features and target.
    agg_results = []
    for df in dfs:
        feature_means = df[features].agg(agg_func)
        target_mean   = df[target_col].agg(agg_func)
        agg_results.append({
            'features': feature_means,
            'target': target_mean,
            'total_features': feature_means.sum()
        })

    # Optional logging: warn if total_features
    # significantly differs from target.
    if verbose >= 1:
        for i, res in enumerate(agg_results):
            discrepancy = abs(
                res['target'] - res['total_features']
            )
            if discrepancy > 0.1 * res['target']:
                warnings.warn(
                    f"{labels[i]} feature sum "
                    f"differs from target by "
                    f"{discrepancy:.2f}"
                )

    # Create the Matplotlib figure and axis.
    fig, ax = plt.subplots(figsize=figsize)

    # Default labels if none provided.
    # labels = labels or [
    #     f'Dataset {i+1}' for i in range(len(dfs))
    # ]
    x = np.arange(len(labels))

    # Generate a color map for the features, 
    # based on the chosen palette.
    feature_colors = plt.cm.get_cmap(
        palette,
        len(features)
    )(
        np.linspace(0, 1, len(features))
    )

    # If user wants a stacked bar plot.
    if kind == 'stacked':
        # Build from the bottom up
        bottoms = np.zeros(len(labels))

        for idx, feature in enumerate(features):
            values = [
                res['features'][feature]
                for res in agg_results
            ]
            ax.bar(
                x,
                values,
                bottom=bottoms,
                color=feature_colors[idx],
                edgecolor='white',
                linewidth=0.5,
                label=feature.replace('_',' ').title()
            )
            # Update the "stack bottom" for next feature
            bottoms += values

            # Optionally annotate bar segments with
            # numeric values
            if show_values:
                for i, (v, b) in enumerate(
                    zip(values, bottoms - values)
                ):
                    # Only annotate large slices
                    if v > 0.1 * bottoms[i]:
                        ax.text(
                            x[i],
                            b + v / 2,
                            f"{v:{value_format}}",
                            ha='center',
                            va='center',
                            color=(
                                'white'
                                if idx > len(features)//2 
                                else 'black'
                            )
                        )

        # Overlay line for target if chosen
        if show_target_line:
            targets = [
                res['target'] for res in agg_results
            ]
            ax.plot(
                x,
                targets,
                'k--o',
                markersize=8,
                linewidth=1.5,
                label=(
                    f"Actual "
                    f"{target_col.replace('_',' ').title()}"
                )
            )

        # Label the x-axis with dataset labels
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')

    # If user wants a heatmap instead.
    elif kind == 'heatmap':
        data_matrix = np.array([
            res['features'].values
            for res in agg_results
        ])
        im = ax.imshow(
            data_matrix.T,
            cmap=palette,
            aspect='auto'
        )

        # Configure the axis labels for heatmap
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticks(np.arange(len(features)))
        ax.set_yticklabels(
            [f.title() for f in features]
        )

        # Annotate each cell with numeric values
        if show_values:
            for i in range(len(labels)):
                for j in range(len(features)):
                    val = data_matrix[i,j]
                    ax.text(
                        i,
                        j,
                        f"{val:{value_format}}",
                        ha='center',
                        va='center',
                        color=(
                            'white'
                            if val > 0.5*data_matrix.max() 
                            else 'black'
                        )
                    )
        # Add colorbar to represent the scale
        plt.colorbar(im, ax=ax, label='Contribution Value')

    # Common labeling, grid, etc.
    ax.set(
        title=(
            title or 
            f"Contribution Analysis: "
            f"{target_col.replace('_',' ').title()}"
        ),
        ylabel=(
            'Contribution Value'
            if kind == 'stacked'
            else ''
        )
    )
    if show_grid: 
        if grid_props is None:
            grid_props = {
                'axis': 'y', 
                'linestyle': ':', 
                'alpha': 0.4}
        
        ax.grid(True, **grid_props)
    else: 
        ax.grid(False)

    # If stacked bar, place legend outside
    if kind == 'stacked':
        ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            frameon=True,
            framealpha=0.9
        )

    # Tight layout to reduce overlap
    plt.tight_layout()

    # Save figure if requested
    if savefig:
        plt.savefig(
            savefig,
            dpi=300,
            bbox_inches='tight'
        )
        if verbose >= 1:
            print(f"Saved visualization to {savefig}")

    return fig

@default_params_plot(
    savefig=PlotConfig.AUTOSAVE('my_prediction_comparison_plot.png')
)
def plot_prediction_comparison(
    *dfs: pd.DataFrame,
    actuals: Union[str, List[str]],
    pred_cols: Union[str, List[str]],
    dt_col: Optional[str] = None,
    dates: Optional[List] = None,
    labels: Optional[List[str]] = None,
    figsize: tuple = (16, 6),
    plot_style: str = 'seaborn',
    ylabel: str = "Rate",
    actual_props: dict = None,
    pred_props: dict = None,
    error_metrics: bool = True,
    show_grid: bool = True,
    grid_props: dict = None,
    fig_title: str = None,
    verbose: int = 1,
    savefig: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """
    Plot actual vs predicted values of one or more time-series or
    sequence-based datasets, facilitating side-by-side comparisons of
    different DataFrames. 
    
    Parameters
    ----------
    dfs : list of pandas.DataFrame
        One or more DataFrames containing columns in ``actuals``
        and ``pred_cols``. Each is verified via
        `are_all_frames_valid`.
    actuals : str or list of str
        Column name(s) representing actual values. If a single
        string is provided, it is converted to a list with one
        element.
    pred_cols : str or list of str
        Column name(s) representing predicted values. If a single
        string is provided, it is converted to a list with one
        element.
    dt_col : str, optional
        The date/time column used for x-axis values. If None, the
        function attempts to use a datetime index from the
        DataFrame. Otherwise, integer positions are used.
    dates : list, optional
        Reserved for future date plotting enhancements; if not
        None, it overrides the x-axis with this date array. Not
        implemented yet.
    labels : list of str, optional
        Custom titles for each subplot (one subplot per DataFrame).
        Defaults to ``["DataFrame 1", "DataFrame 2", ...]``.
    figsize : tuple of float, optional
        Width and height of the figure. Default is (16, 6).
    plot_style : str, optional
        Matplotlib style name applied globally (e.g.,
        ``'seaborn'``). Default is ``'seaborn'``.
    ylabel : str, optional
        Label for the y-axis, shared by all subplots.
        Default is "Rate".
    actual_props : dict, optional
        Custom properties (line style, marker, etc.) for actual
        data plotting. If None, defaults are used.
    pred_props : dict, optional
        Custom properties (line style, marker, etc.) for
        predicted data. If None, defaults are used.
    error_metrics : bool, optional
        If True, computes the RMSE for the first actual/pred pair
        and displays it in the subplot. Default is True.
    show_grid : bool, optional
        Whether to display grid lines. Default is True.
    grid_props : dict, optional
        Properties for the grid lines (e.g.,
        ``{'axis': 'both', 'alpha': 0.7, 'linestyle': ':'}``).
    fig_title : str, optional
        Main title displayed above all subplots. Defaults to
        "Actual vs Predicted Subsidence Comparison".
    verbose : int, optional
        Verbosity level. 0 = silent, 1 = basic info.
        Default is 1.
    savefig : str, optional
        Path to save the resulting figure. If None, the figure
        is not saved. Default is None.
    
    Returns
    -------
    matplotlib.figure.Figure
        The Matplotlib Figure object containing one subplot per
        DataFrame, each plotting its actual and predicted values.
    
    Notes
    -----
    - If the length of `actuals` differs from `pred_cols`, only the
      first `min(len(actuals), len(pred_cols))` pairs are plotted.
    - By default, an RMSE annotation is added to each subplot
      (when `error_metrics` is True) based on the first pair of
      `actuals`/`pred_cols`.

    It applies `are_all_frames_valid` to ensure
    each DataFrame contains the required columns and supports an
    optional calculation of RMSE (Root Mean Squared Error) [1]_.
    
    .. math::
       \\mathrm{RMSE}(y, \\hat{y}) =
       \\sqrt{\\frac{1}{N}\\sum_{i=1}^{N}(y_i - \\hat{y}_i)^2}
 
    Examples
    --------
    >>> from gofast.plot.comparison import plot_prediction_comparison
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>>
    >>> # Create sample data
    >>> df1 = pd.DataFrame({
    ...     "time": pd.date_range(
    ...         start="2020-01-01",
    ...         periods=50,
    ...         freq="D"
    ...     ),
    ...     "subsidence_actual": np.random.rand(50) + 1,
    ...     "subsidence_pred": np.random.rand(50) + 0.8
    ... })
    >>> df2 = pd.DataFrame({
    ...     "time": pd.date_range(
    ...         start="2021-01-01",
    ...         periods=50,
    ...         freq="D"
    ...     ),
    ...     "subsidence_actual": np.random.rand(50) + 1.2,
    ...     "subsidence_pred": np.random.rand(50) + 1
    ... })
    >>>
    >>> # Plot comparison
    >>> fig = plot_prediction_comparison(
    ...     df1,
    ...     df2,
    ...     actuals="subsidence_actual",
    ...     pred_cols="subsidence_pred",
    ...     dt_col="time",
    ...     fig_title="Comparing Actual vs Predicted Subsidence",
    ...     error_metrics=True
    ... )
    
    See Also
    --------
    are_all_frames_valid : Validates each DataFrame's columns or
        datetime index requirements.
    set_axis_grid : Adds a consistent grid style to Matplotlib
        axes.
    
    References
    ----------
    .. [1] Willmott, C. J. (1982). Some comments on the evaluation
           of model performance. Bulletin of the American
           Meteorological Society, 63(11), 1309-1313.
    """

    # Ensure DataFrames are valid using the 
    # gofast.core.checks utility function
    dfs = are_all_frames_valid(*dfs, ops='validate')

    # Convert single string inputs into lists for 
    # unified handling
    actual_list = (
        [actuals] if isinstance(actuals, str)
        else actuals
    )
    pred_list = (
        [pred_cols] if isinstance(pred_cols, str)
        else pred_cols
    )
    pair_count = min(len(actual_list), len(pred_list))

    # Warn if actual/pred lists mismatch in length
    if len(actual_list) != len(pred_list):
        warnings.warn(
            f"Actual/Predicted mismatch. "
            f"Using first {pair_count} pairs."
        )

    # Apply style globally
    plt.style.use(plot_style)

    # Create figure with subplots for each DataFrame
    fig, axes = plt.subplots(
        1,
        len(dfs),
        figsize=figsize,
        sharey=True
    )
    # Convert single axis object to a list
    axes = axes if len(dfs) > 1 else [axes]

    # Validate columns in each DataFrame
    for i, df_ in enumerate(dfs):
        missing = [
            c for c in actual_list + pred_list
            if c not in df_.columns
        ]
        if missing:
            raise ValueError(
                f"DataFrame {i} missing columns: {missing}"
            )

    # Prepare time references if dt_col is provided
    time_data = []
    for df_ in dfs:
        if dt_col:
            if dt_col not in df_.columns:
                raise ValueError(
                    f"Missing datetime column: {dt_col}"
                )
            # Convert int-coded years or standard 
            # datetime
            if pd.api.types.is_integer_dtype(
                df_[dt_col]
            ):
                time_ref = pd.to_datetime(
                    df_[dt_col].astype(str),
                    format='%Y'
                )
            else:
                time_ref = pd.to_datetime(df_[dt_col])
        else:
            # If index is datetime, use it; otherwise
            # no time dimension
            time_ref = (
                df_.index
                if pd.api.types.is_datetime64_any_dtype(
                    df_.index
                )
                else None
            )

        # Store grouped data if time_ref is present
        time_data.append({
            'ref': time_ref,
            'grouped': (
                df_.groupby(time_ref).mean()
                if time_ref is not None
                else df_
            )
        })

    # Define defaults for actual/pred line props
    d_colors = select_hex_colors(len(pred_list), seed=42 )
    actual_props = (
        actual_props
        or {'linestyle': '-', 'marker': 'o', 'linewidth': 1.5,}
    )
    if 'color' not in actual_props: 
        actual_colors = d_colors  # plt.cm.tab10  
    else: 
        actual_colors= columns_manager (actual_props.pop('color' )) + d_colors 
    
    pred_props = (
        pred_props
        or {'linestyle': '--', 'marker': 'x', 'alpha': 0.8}
    )
    if 'color' not in pred_props: 
        pred_colors = d_colors 
    else: 
        pred_colors= columns_manager (pred_props.pop('color' )) + d_colors 
    
    grid_props = (
        grid_props
        or {'axis': 'both', 'alpha': 0.7, 'linestyle': ':'}
    )
    # Generate or reuse labels if not provided
    if labels is None:
        labels = [
            f"Dataset {i+1}" for i in range(len(dfs))
        ]
    else:
        labels = list(labels) + [
            f"Dataset {i+1}"
            for i in range(len(labels), len(dfs))
        ]

    # Plot each DataFrame in its own subplot
    for ax, df_, tdata, lbl in zip(
        axes,
        dfs,
        time_data,
        labels
    ):
        # For each pair of actual/pred columns
        for i, (actual, pred) in enumerate(
            zip(
                actual_list[:pair_count],
                pred_list[:pair_count]
            )
        ):
            # Use grouped or naive index for x-values
            x_vals = (
                tdata['grouped'].index
                if tdata['ref'] is not None
                else np.arange(len(df_))
            )
            # Plot actual
            
            ax.plot(
                x_vals,
                tdata['grouped'][actual],
                color =actual_colors[i],
                label = (
                    f"{lbl} Actual"
                    if i == 0
                    else None
                ),
                **actual_props
            )
            # Plot predicted
            ax.plot(
                x_vals,
                tdata['grouped'][pred],
                color = pred_colors[i], 
                # color = plt.cm.tab10(i),
                label = (
                    f"{lbl} Predicted"
                    if i == 0
                    else None
                ),
                **pred_props
            )

        # Label axes
        ax.set(
            title = lbl,
            xlabel = (
                dt_col or
                "Observation Sequence"
                if dates is None else
                "Date/Time"
            ),
            ylabel = ylabel
        )

        # Toggle grid
        if show_grid:
            ax.grid(True, **grid_props)
        else:
            ax.grid(False)

        # If error metrics is True, compute RMSE for
        # the first pair to annotate
        if error_metrics:
            rmse = mean_squared_error(
                df_[actual_list[0]],
                df_[pred_list[0]],
                squared=False
            )
            ax.annotate(
                f"RMSE: {rmse:.2f}",
                xy = (0.05, 0.9),
                xycoords = 'axes fraction',
                fontsize = 10,
                bbox = dict(
                    boxstyle='round',
                    alpha=0.2
                )
            )

        # Rotate x-ticks if vertical orientation
        ax.tick_params(axis='x', rotation=45)
    
    # Add overall title
    fig.suptitle(
        fig_title
        or "Actual vs Predicted Subsidence Comparison",
        y=1.02,
        fontsize=14
    )

    # Gather handles/labels and place a shared legend
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labs,
        loc = 'upper center',
        bbox_to_anchor = (0.5, -0.05),
        ncol = 2
    )

    # Reduce overlapping subplots
    plt.tight_layout()

    # Return the figure for further usage
    return fig

@default_params_plot(
    savefig=PlotConfig.AUTOSAVE('my_error_analysis_plot.png')
)
def plot_error_analysis(
    *dfs: pd.DataFrame,
    actual_col: str,
    pred_col: str,
    dt_col: Optional[str] = None,
    error_type: str = 'abs',    # 'absolute' or 'relative'
    time_bins: Union[str, List] = 'auto',
    target_bins: Union[int, List] = 5,
    matrix_type: str = 'heatmap',    # 'confusion' or 'error'
    figsize: Tuple[int, int] = (14, 6),
    cmap: Union[str, LinearSegmentedColormap] = 'coolwarm',
    annot: bool = True,
    fmt: str = '.1',
    cbar_label: str = 'Error Density',
    normalize: Union[bool, str] = 'true',
    ylabel = None,
    show_grid = True,
    grid_props: dict = None,
    title: Optional[str] = None,
    verbose: int = 1,
    **kwargs
) -> plt.Figure:
    """
    Plot error distributions for one or more DataFrames by binning
    the actual and predicted values over time (and target bins),
    then visualizing them as a heatmap or confusion matrix. 

    Parameters
    ----------
    dfs : list of pandas.DataFrame
        One or more DataFrames containing the columns 
        ``actual_col`` and ``pred_col``. Validated by 
        `are_all_frames_valid`.
    actual_col : str
        Name of the column containing actual values.
    pred_col : str
        Name of the column containing predicted values.
    dt_col : str, optional
        The date/time column. If specified, time-based binning
        can be performed by ``time_bins``. If None, all data are
        treated as a single time-bin.
    error_type : {'absolute', 'relative'}, optional
        Type of error to compute. Default is ``'absolute'``.
    time_bins : str or list, optional
        Time bin specification. If ``'auto'``, the function
        infers a yearly (``'Y'``) or monthly (``'M'``) binning
        based on the total time span. If a list, it should be a
        list of valid periods or boundaries. Default is
        ``'auto'``.
    target_bins : int or list, optional
        Bin specification for the actual values. If an integer,
        the function uses quantile-based binning with that many
        bins. If a list, it is used as boundary edges directly.
        Default is 5.
    matrix_type : {'heatmap', 'confusion'}, optional
        Whether to generate a mean-error heatmap or a confusion
        matrix from binned actual vs. binned predicted.
        Default is ``'heatmap'``.
    figsize : tuple of int, optional
        Size of the figure in inches, e.g. (14, 6).
    cmap : str or matplotlib.colors.LinearSegmentedColormap, optional
        The colormap for the heatmap or confusion matrix.
        Default is ``'coolwarm'``.
    annot : bool, optional
        Whether to annotate heatmap/confusion cells with numeric
        values. Default is True.
    fmt : str, optional
        String format for annotation (e.g. ``'.1'``). Default is
        ``'.1'``.
    cbar_label : str, optional
        Label for the color bar. Default is ``'Error Density'``.
    normalize : bool or str, optional
        Used if ``matrix_type='confusion'``. Determines how
        confusion matrix values are normalized. ``'true'``
        normalizes over each true (row) bin. Default is ``'true'``.
    ylabel : str, optional
        Label for the y-axis. Defaults to ``'Actual Level'``.
    show_grid : bool, optional
        Whether to display grid lines over the plot.
        Default is True.
    grid_props : dict, optional
        Properties for the grid lines. For example:
        ``{'color': 'lightgray', 'linestyle': '--'}``.
    title : str, optional
        Main title for the figure. Defaults to
        "Absolute Error Analysis" or "Relative Error Analysis"
        based on the value of ``error_type``.
    verbose : int, optional
        Verbosity level (0 = silent, 1 = basic progress info).
        Default is 1.
    **kwargs :
        Additional keyword arguments passed to `sns.heatmap`.
    
    Returns
    -------
    matplotlib.figure.Figure
        The Matplotlib Figure object containing the error analysis
        heatmap or confusion matrix for each DataFrame.
    
    Notes
    -----
    - If ``matrix_type='heatmap'``, each cell represents the mean
      error computed for a given time bin (x-axis) and actual bin
      (y-axis).
    - If ``matrix_type='confusion'``, each cell represents the
      frequency of actual bins versus predicted bins, normalized
      as specified by ``normalize``.
    
    The function verifies DataFrame validity through
    `are_all_frames_valid` and can compute either absolute or
    relative errors. This function can facilitate understanding
    of error magnitudes across different temporal segments and
    target value ranges.
    
    .. math::
       e_{\\mathrm{abs}}(i) = \\hat{y}_i - y_i
    
    .. math::
       e_{\\mathrm{rel}}(i) = 
       \\frac{\\hat{y}_i - y_i}{y_i + 10^{-6}}
    
    where :math:`y_i` is the actual value, 
    :math:`\\hat{y}_i` the predicted value, 
    and :math:`e_{\\mathrm{abs}}, e_{\\mathrm{rel}}` the absolute
    and relative errors, respectively.
    
    Examples
    --------
    >>> from gofast.plot.comparison import plot_error_analysis
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(0)
    >>>
    >>> # Create sample data
    >>> df = pd.DataFrame({
    ...     "year": np.random.randint(2015, 2021, 100),
    ...     "actual_val": np.random.rand(100) * 10 + 5,
    ...     "pred_val": np.random.rand(100) * 10 + 4
    ... })
    >>>
    >>> # Plot absolute error heatmap
    >>> fig = plot_error_analysis(
    ...     df,
    ...     actual_col="actual_val",
    ...     pred_col="pred_val",
    ...     dt_col="year",
    ...     error_type="absolute",
    ...     matrix_type="heatmap",
    ...     time_bins="auto",
    ...     target_bins=5,
    ...     title="Absolute Error Heatmap"
    ... )
    
    See Also
    --------
    are_all_frames_valid : Ensures DataFrame columns meet minimum
        requirements.
    set_axis_grid : Applies consistent grid aesthetics.
    
    References
    ----------
    .. [1] Taylor, S. J., & Letham, B. (2018). Forecasting at scale.
           The American Statistician, 72(1), 37-45.
    """

    dfs = are_all_frames_valid(*dfs, ops='validate')
    
    # Validate the chosen error type
    valid_errors = ['absolute', 'relative',]
    error_type = parameter_validator(
        "error_type", 
        target_strs= valid_errors, 
        deep=True, 
        error_msg =  f"Invalid error_type. Choose from {valid_errors}"
        )(error_type)
    # For each DataFrame, confirm required columns
    for df in dfs:
        if (actual_col not in df.columns
           or pred_col not in df.columns):
            missing = [
                c for c in [actual_col, pred_col]
                if c not in df.columns
            ]
            raise ValueError(
                f"Missing columns: {missing} in dataframe"
            )

    # Accumulate pre-processed versions of input data
    processed = []
    for df in dfs:
        df_copy = df.copy()
        # Compute absolute or relative error
        if error_type == 'absolute':
            df_copy['error'] = (
                df_copy[pred_col]
                - df_copy[actual_col]
            )
        else:
            df_copy['error'] = (
                (
                    df_copy[pred_col]
                    - df_copy[actual_col]
                ) 
                / df_copy[actual_col].clip(lower=1e-6)
            )

        # Convert and bin time if dt_col is provided
        if dt_col:
            if dt_col not in df_copy.columns:
                raise ValueError(
                    f"Missing datetime column: {dt_col}"
                )
            # If dt_col is integer-coded years
            if pd.api.types.is_integer_dtype(
                df_copy[dt_col]
            ):
                df_copy[dt_col] = pd.to_datetime(
                    df_copy[dt_col].astype(str),
                    format='%Y'
                )
            else:
                df_copy[dt_col] = pd.to_datetime(
                    df_copy[dt_col]
                )
            # Auto-set bins if specified
            if time_bins == 'auto':
                time_delta = (
                    df_copy[dt_col].max()
                    - df_copy[dt_col].min()
                )
                time_bins = (
                    'Y' if time_delta.days > 365
                    else 'M'
                )
            df_copy['time_bin'] = (
                df_copy[dt_col].dt.to_period(time_bins)
            )
        else:
            # Single bucket if no dt_col
            df_copy['time_bin'] = 'All Time'

        # Bin actual_col to create target_bins
        if isinstance(target_bins, int):
            bins = pd.qcut(
                df_copy[actual_col],
                target_bins,
                duplicates='drop'
            ).cat.categories
        else:
            bins = target_bins
        df_copy['target_bins'] = pd.cut(
            df_copy[actual_col],
            bins=bins,
            include_lowest=True
        )
        processed.append(df_copy)

    # Build either confusion or error (heatmap) matrix
    matrices = []
    # Support optional tqdm for progress logging
    matrix_iterator = processed
    if HAS_TQDM:
        matrix_iterator = tqdm(
            processed,
            desc="Generating matrices",
            leave=False
        )
    elif verbose >= 1:
        print(
            f"Generating {matrix_type} matrices"
        )

    for df_ in matrix_iterator:
        if matrix_type == 'confusion':
            # True label: target_bins, predicted label: 
            # same bins on pred_col
            matrix = confusion_matrix(
                df_['target_bins'].astype(str),
                pd.cut(
                    df_[pred_col],
                    bins=bins,
                    include_lowest=True
                ).astype(str),
                normalize=normalize
            )
        else:
            # Pivot for average error by time_bin and target_bins
            matrix = df_.pivot_table(
                index='target_bins',
                columns='time_bin',
                values='error',
                aggfunc='mean'
            ).fillna(0)
        matrices.append(matrix)

    # Setup default grid props
    grid_props = (
        grid_props
        or {'color': 'lightgray', 'linestyle': '--'}
    )

    # Create subplots for each dataset
    n_plots = len(matrices)
    fig, axes = plt.subplots(
        1,
        n_plots,
        figsize=(figsize[0]*n_plots, figsize[1]),
        squeeze=False
    )
    axes = axes.flatten()

    plot_iterator = zip(axes, matrices, dfs)
    if HAS_TQDM:
        plot_iterator = tqdm(
            plot_iterator,
            total=len(matrices),
            desc="Plotting"
        )

    # Render each matrix in a heatmap
    for ax, matrix, df_ in plot_iterator:
        annotate = (
            matrix.round(2).astype(str)
            if annot
            else None
        )
        sns.heatmap(
            matrix,
            ax=ax,
            cmap=cmap,
            annot=annotate,
            fmt=fmt,
            cbar_kws={'label': cbar_label},
            **kwargs
        )
        # Axis labeling
        ax.set(
            xlabel=(
                'Time Period'
                if dt_col else
                'Predicted Level'
            ),
            ylabel=ylabel or 'Actual Level',
            title=(
                f"{getattr(df_, 'name', 'Dataset')} Profile"
            )
        )
        ax.tick_params(axis='x', rotation=45)

        # Optionally draw grid lines
        set_axis_grid(
            ax,
            show_grid,
            grid_props
        )

    # Common figure title
    fig.suptitle(
        title or
        f"{error_type.title()} Error Analysis: "
        f"{actual_col} vs {pred_col}",
        y=1.05,
        fontsize=14
    )
    plt.tight_layout()

    if verbose >= 1:
        print(
            f"\nGenerated {matrix_type} matrix for {n_plots} datasets"
        )

    return fig

@default_params_plot(
    savefig=PlotConfig.AUTOSAVE('my_trends_plot.png')
)
def plot_trends(
    *dfs: pd.DataFrame,
    target_col: str,
    dt_col: str,
    labels: Optional[List[str]] = None,
    time_freq: Union[str, None] = 'Y',  # 'Y','M','Q', None for raw
    confidence: Union[bool, float] = False,  # CI shading
    colors: Optional[List[str]] = None,
    styles: List[str] = None,
    plot_style: str='seaborn-whitegrid', 
    markers: List[str] = None,
    error_props: dict = None,
    figsize: tuple = (14, 7),
    title: str = 'Comparative Analysis',
    xlabel: str = 'Observation Period',
    ylabel: str = 'Subsidence Rate',
    show_grid: bool=True, 
    grid_props: dict =None,
    savefig: Optional[str]=None, 
    verbose: int = 0,
    **kwargs
) -> plt.Figure:
    """
    Visualize comparative trends of a target column across multiple
    DataFrames, optionally resampled by a given time frequency and
    augmented with confidence intervals. 
    
    Parameters
    ----------
    dfs : list of pandas.DataFrame
        One or more DataFrames, each containing the columns
        ``target_col`` and ``dt_col``. Validated by
        `are_all_frames_valid`.
    target_col : str
        Name of the column to be plotted as the trend.
    dt_col : str
        Name of the date/time column for indexing or resampling.
    labels : list of str, optional
        Custom labels for each dataset. Defaults to
        ``["Dataset 1", "Dataset 2", ...]`` if not provided.
    time_freq : str or None, optional
        Resampling frequency for time-series aggregation.
        Examples include ``'Y'``, ``'M'``, or ``'Q'``. If
        ``None``, data are plotted without resampling.
        Default is ``'Y'`` (yearly).
    confidence : bool or float, optional
        If True, draws a 95% confidence interval region
        (:math:`\\pm 1.96 \\cdot \\sigma/\\sqrt{n}`).
        If a float is provided, it is used as the multiplier
        instead of 1.96. Default is False (no CI).
    colors : list of str, optional
        Custom colors to cycle through for each dataset.
        Defaults to a palette from `matplotlib` or `seaborn`.
    styles : list of str, optional
        A list of line styles (e.g. ``'-', '--', '-.', ':'``)
        to cycle through.
    plot_style : str, optional
        Name of the Matplotlib style (e.g. ``'seaborn-whitegrid'``)
        to apply globally. Default is ``'seaborn-whitegrid'``.
    markers : list of str, optional
        A list of marker styles (e.g. ``'o', 's', 'D', '^'``)
        to cycle through for each dataset's trend line.
    error_props : dict, optional
        Additional properties for the confidence interval shading,
        e.g. ``{'alpha': 0.2, 'linewidth': 0}``.
    figsize : tuple of float, optional
        Width and height of the figure in inches. Default is (14, 7).
    title : str, optional
        Main title of the plot. Default is
        "Comparative Subsidence Analysis".
    xlabel : str, optional
        Label for the x-axis. Default is "Observation Period".
    ylabel : str, optional
        Label for the y-axis. Default is "Subsidence Rate".
    show_grid : bool, optional
        Whether to show grid lines on the plot. Default is True.
    grid_props : dict, optional
        Grid line properties (e.g. ``{'which': 'both', 'alpha': 0.4}``).
    savefig : str, optional
        Path to save the resulting figure. If None, no file is saved.
        Default is None.
    verbose : int, optional
        Verbosity level (0 = silent, 1 = basic info). Default is 0.
    **kwargs :
        Additional keyword arguments passed to the Matplotlib
        plotting function.
    
    Returns
    -------
    matplotlib.figure.Figure
        The Matplotlib Figure object containing the comparative
        trend plot.
    
    Notes
    -----
    - When ``time_freq='auto'``, the function infers a yearly
      frequency if the time span exceeds two years; otherwise,
      it uses monthly resampling.
    - If ``confidence`` is True, the 95% CI is computed using
      the standard error of the mean (SEM) multiplied by 1.96
      (or by the user-specified float).
    
    This function calls `are_all_frames_valid` to validate 
    DataFrame integrity, then groups or resamples data by 
    `the specified `time_freq`` to compute mean and standard 
    deviation. It can also highlight confidence intervals around
    the mean if desired.
    
    .. math::
       \\bar{x}(t) = \\frac{1}{n} \\sum_{i=1}^{n} x_i(t),
       \\quad
       \\sigma(t) = \\sqrt{\\frac{1}{n}\\sum_{i=1}^{n}
       (x_i(t) - \\bar{x}(t))^2},
    
    where :math:`\\bar{x}(t)` is the mean of the target column
    over the period :math:`t` and :math:`\\sigma(t)` is the
    standard deviation.
    
    Examples
    --------
    >>> from gofast.plot.comparison import plot_trends
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>>
    >>> # Generate synthetic data
    >>> df1 = pd.DataFrame({
    ...     "date": pd.date_range(
    ...         start="2020-01-01", 
    ...         periods=50, 
    ...         freq="M"
    ...     ),
    ...     "subsidence": np.random.rand(50) * 2 + 1
    ... })
    >>> df2 = pd.DataFrame({
    ...     "date": pd.date_range(
    ...         start="2021-01-01", 
    ...         periods=50, 
    ...         freq="M"
    ...     ),
    ...     "subsidence": np.random.rand(50) * 2 + 1.5
    ... })
    >>>
    >>> # Plot the trends with confidence intervals
    >>> fig = plot_trends(
    ...     df1,
    ...     df2,
    ...     target_col="subsidence",
    ...     dt_col="date",
    ...     time_freq="M",
    ...     confidence=True,
    ...     title="Monthly Subsidence Trend",
    ...     verbose=1
    ... )
    
    See Also
    --------
    are_all_frames_valid : Checks if DataFrames have necessary columns
        or a datetime index.
    set_axis_grid : Configures grid lines for one or more axes.
    
    References
    ----------
    .. [1] Chatfield, C. (2003). The Analysis of Time Series:
           An Introduction. Chapman and Hall/CRC.
    """
    dfs = are_all_frames_valid(*dfs, ops='validate')
    # --------------------------
    # Validation & Setup
    # --------------------------
    plt.style.use(plot_style)
    fig, ax = plt.subplots(figsize=figsize)
    
    for i, df in enumerate(dfs):
        missing = [c for c in [target_col, dt_col] 
                   if c not in df.columns]
        if missing:
            raise ValueError(f"Dataset {i} missing: {missing}")

    # --------------------------
    # Data Processing
    # --------------------------
    processed = []
    for df in dfs:
        # Temporal handling
        # If dt_col is integer-coded years
        if pd.api.types.is_integer_dtype(
            df[dt_col]
        ):
            time_series= pd.to_datetime(
                df[dt_col].astype(str),
                format='%Y'
            )
        else:
            time_series = pd.to_datetime(
                df[dt_col]
            )
    
        df = df.set_index(time_series).sort_index()
        
        # Frequency detection
        if time_freq == 'auto':
            time_span = df.index.max() - df.index.min()
            time_freq = 'Y' if time_span.days > 730 else 'M'
        
        # Resampling
        agg_df = (df.resample(time_freq)[target_col]
                   .agg(['mean', 'std', 'count'])
                   if time_freq 
                   else df[[target_col]])
        
        processed.append(agg_df)

    # --------------------------
    # Visualization
    # --------------------------
    color_cycler = cycle(colors or sns.color_palette())
    
    styles = styles or ['-', '--', '-.', ':']
    style_cycler = cycle(styles)
    
    markers = markers or ['o', 's', 'D', '^']
    marker_cycler = cycle(markers)

    error_props = error_props or {'alpha': 0.2, 'linewidth': 0}
    grid_props = grid_props or {'visible': True, 'which': 'both', 'alpha': 0.4}
    
    if labels is None:
        labels = [f'Dataset {i+1}' for i in range(len(dfs))]
    else:
        # Extend labels if shorter than dataset count
        labels = list(labels) + [
            f'Dataset {i+1}' 
            for i in range(len(labels), len(dfs))
        ][:len(dfs)]  # Ensure exact match
        
    for i, (data, label) in enumerate(zip(
        processed, 
        labels
    )):
        # Main plot
        ax.plot(
            data.index, data['mean'],
            color     = next(color_cycler),
            linestyle = next(style_cycler),
            marker    = next(marker_cycler),
            label     = label,
            linewidth = 2,
            markersize= 8
        )
        
        # Confidence interval
        if confidence and 'std' in data.columns:
            ci = (data['std'] / data['count']**0.5 * 
                  (1.96 if isinstance(confidence, bool) else confidence))
            ax.fill_between(
                data.index,
                data['mean'] - ci,
                data['mean'] + ci,
                color     = ax.lines[-1].get_color(),
                **error_props
            )

    # --------------------------
    # Aesthetics
    # --------------------------
    ax.set(
        title  = title,
        xlabel = xlabel,
        ylabel = ylabel
    )
    ax.xaxis.set_major_formatter(
        plt.FixedFormatter(data.index.strftime('%Y-%m'))  # Auto-format
        )
    
    # Grid & legend
    set_axis_grid (ax, show_grid, grid_props)

    ax.legend(
        bbox_to_anchor = (1.02, 1),
        loc            = 'upper left',
        borderaxespad  = 0.0
    )

    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

@default_params_plot(
    savefig=PlotConfig.AUTOSAVE('my_variability_plot.png')
)
def plot_variability(
    *dfs: pd.DataFrame,
    target_col: str,
    labels: Optional[List[str]] = None,
    kind: str = 'box', # 'box' or 'violin'
    orient: str = 'v', # orientation: vertical or horizontal
    palette: Union[str, List[str]] = 'tab10',
    show_swarm: bool = False,
    figsize: tuple = (8, 6),
    title: str = 'Distribution Comparison',
    xlabel: str = 'City',
    ylabel: str = 'Rate', # 'Subsidence Rate (mm/year)',
    show_grid: bool = True,
    grid_props: dict = None,
    verbose: int = 0,
    savefig: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """
    Plot the variability in a `target_col` from one or more
    DataFrames using box or violin plots, with an optional swarm
    plot overlay for individual data points. 
    
    Parameters
    ----------
    dfs : list of pandas.DataFrame
        One or more DataFrames containing the column
        ``target_col``. Each DataFrame is validated with
        `are_all_frames_valid`.
    target_col : str
        Column name representing the values to be plotted
        (e.g., subsidence rates).
    labels : list of str, optional
        Custom labels for the datasets. Defaults to
        ``["Dataset 1", "Dataset 2", ...]`` if not provided.
    kind : {'box', 'violin'}, optional
        Type of plot to produce. ``'box'`` displays box plots,
        while ``'violin'`` displays violin plots. Default is
        ``'box'``.
    orient : {'v', 'h'}, optional
        Orientation of the plot. ``'v'`` (vertical) places the
        groups on the x-axis. ``'h'`` (horizontal) places the
        groups on the y-axis. Default is ``'v'``.
    palette : str or list of str, optional
        Color palette or list of colors for the plot.
        Default is ``'tab10'``.
    show_swarm : bool, optional
        Whether to overlay individual data points using a swarm
        plot for additional granularity. Default is False.
    figsize : tuple of float, optional
        Width and height of the figure in inches. Default
        is (8, 6).
    title : str, optional
        Main title of the plot. Default is
        "Distribution Comparison".
    xlabel : str, optional
        Label for the x-axis. Default is "City".
    ylabel : str, optional
        Label for the y-axis. Default is "Rate".
    show_grid : bool, optional
        Whether to display grid lines. Default is True.
    grid_props : dict, optional
        Grid properties (e.g., ``{'axis': 'y', 'alpha': 0.4,
        'linestyle': ':'}``) passed to `set_axis_grid`.
    verbose : int, optional
        Verbosity level (0 = silent, 1 = info). Default is 0.
    savefig : str, optional
        Path to save the figure. If None, no file is saved.
        Default is None.
    **kwargs :
        Additional keyword arguments passed to either Seaborn
        boxplot or violinplot.
    
    Returns
    -------
    matplotlib.figure.Figure
        The Matplotlib Figure object containing the box or violin
        plot of the combined data from the input DataFrames.
    
    Notes
    -----
    - This function concatenates each DataFrame's `target_col`
      into a single structure, labeling each dataset by the
      corresponding entry in `labels`.
    - If `show_swarm` is True, a swarm plot is overlaid to reveal
      individual data points within each group.
      
    This function ensures the presence of `target_col` in each 
    DataFrame through `are_all_frames_valid`, then consolidates 
    data for group-wise visualization. It can visualize either 
    box plots (quartiles and outliers) or violin plots 
    (distribution shapes) via `is_valid_kind`.
    
    .. math::
       Q1, Q2, Q3, IQR
    
    For box plots, :math:`Q1`, :math:`Q2` (median), and
    :math:`Q3` correspond to the first, second, and third
    quartiles, while :math:`IQR = Q3 - Q1` is the interquartile
    range representing the box's height. For violin plots,
    the data distribution is estimated via a kernel density
    function to reveal the shape of the distribution.
      
    
    Examples
    --------
    >>> from gofast.plot.comparison import plot_variability
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>>
    >>> # Generate synthetic data
    >>> df1 = pd.DataFrame({
    ...     "subsidence": np.random.rand(50) + 1
    ... })
    >>> df2 = pd.DataFrame({
    ...     "subsidence": np.random.rand(60) + 0.5
    ... })
    >>>
    >>> # Plot boxplots with swarm overlays
    >>> fig = plot_variability(
    ...     df1, df2,
    ...     target_col="subsidence",
    ...     labels=["City A", "City B"],
    ...     kind="box",
    ...     show_swarm=True,
    ...     title="Subsidence Rate Boxplot Comparison"
    ... )
    
    See Also
    --------
    are_all_frames_valid : Checks that each DataFrame contains
        the required columns.
    set_axis_grid : Configures grid line properties on
        Matplotlib axes.
    is_valid_kind : Validates the requested plot kind.
    
    References
    ----------
    .. [1] McGill, R., Tukey, J. W., & Larsen, W. A. (1978).
           Variations of box plots. The American Statistician,
           32(1), 12-16.
    """

    # Validate plot type
    dfs = are_all_frames_valid(*dfs, ops='validate')
    
    kind = is_valid_kind(
        kind, valid_kinds = {'box', 'violin'} 
    )

    # Confirm each DataFrame has the target column
    for i, df in enumerate(dfs):
        if target_col not in df.columns:
            raise ValueError(
                f"DataFrame {i} missing '{target_col}' column"
            )

    # Build label list if not provided
    if labels is None:
        labels = [f'Dataset {i+1}' for i in range(len(dfs))]
    else:
        # Extend labels if shorter than dataset count
        labels = list(labels) + [
            f'Dataset {i+1}' 
            for i in range(len(labels), len(dfs))
        ][:len(dfs)]  # Ensure exact match
        
    # Combine data from all DataFrames in one structure
    plot_data = []
    for df, lbl in zip(dfs, labels):
        plot_data.append(
            pd.DataFrame({
                'value': df[target_col],
                'group': lbl
            })
        )
    combined_df = pd.concat(plot_data, ignore_index=True)

    # Create figure & axis
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots(figsize=figsize)

    # Generate box or violin plot
    # Main plot
    if kind == 'box':
        sns.boxplot(
            x='group' if orient == 'v' else 'value',
            y='value' if orient == 'v' else 'group',
            data=combined_df,
            palette=palette,
            width=0.6,
            linewidth=1.5,
            ax=ax,
            **kwargs
        )
    else:
        sns.violinplot(
            x='group' if orient == 'v' else 'value',
            y='value' if orient == 'v' else 'group',
            data=combined_df,
            palette=palette,
            inner='quartile',
            cut=0,
            bw=0.2,
            ax=ax,
            **kwargs
        )

    # Overlay swarm plot
    if show_swarm:
        sns.swarmplot(
            x='group' if orient == 'v' else 'value',
            y='value' if orient == 'v' else 'group',
            data=combined_df,
            color='.25',
            size=3,
            alpha=0.7,
            ax=ax
        )
        
    # Axis labeling & optional grid
    ax.set(
        title  = title,
        xlabel = xlabel,
        ylabel = ylabel
    )
    grid_props = (
        grid_props
        or {'axis': 'y', 'alpha': 0.4, 'linestyle': ':'}
    )
    set_axis_grid(ax, show_grid, grid_props)

    # Rotate x-ticks if vertical orientation
    plt.xticks(rotation=45 if orient == 'v' else 0)

    # Save figure if requested
    if savefig:
        plt.savefig(savefig, bbox_inches='tight', dpi=300)
        if verbose >= 1:
            print(f"Saved plot to {savefig}")

    plt.tight_layout()
    
    return fig


# nansha_sample.columns
# Out[205]: 
# Index(['longitude', 'latitude', 'year', 'building_concentration', 'geology',
#        'GWL', 'rainfall_mm', 'normalized_seismic_risk_score', 'soil_thickness',
#        'subsidence'],
#       dtype='object')

# Index(['longitude', 'latitude', 'year', 'GWL', 'seismic_risk_score',
#        'rainfall_mm', 'subsidence', 'geological_category',
#        'normalized_density', 'density_tier', 'subsidence_intensity',
#        'density_concentration', 'normalized_seismic_risk_score',
#        'rainfall_category'],
#       dtype='object')
     
