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
from ..core.plot_manager import ( 
    set_axis_grid, 
    default_params_plot, 
    is_valid_kind 
)
from ._config import PlotConfig

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
    'plot_variability'
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
            f"Created scatter plot with {len(dfs)} datasets"
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
    plot_type: str = 'stacked',
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
                print(
                    f"Warning: Dataset {i} feature sum "
                    f"differs from target by "
                    f"{discrepancy:.2f}"
                )

    # Create the Matplotlib figure and axis.
    fig, ax = plt.subplots(figsize=figsize)

    # Default labels if none provided.
    labels = labels or [
        f'Dataset {i+1}' for i in range(len(dfs))
    ]
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
    if plot_type == 'stacked':
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
    elif plot_type == 'heatmap':
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
            if plot_type == 'stacked'
            else ''
        )
    )
    if show_grid: 
        if grid_props is None:
            grid_props = {
                'axis': 'y', 
                'linestyle': ':', 
                'alpha': 0.4},
        
        ax.grid(True, **grid_props)
    else: 
        ax.grid(False)

    # If stacked bar, place legend outside
    if plot_type == 'stacked':
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
    actual_props = (
        actual_props
        or {'linestyle': '-', 'marker': 'o', 'linewidth': 1.5,}
    )
    pred_props = (
        pred_props
        or {'linestyle': '--', 'marker': 'x', 'alpha': 0.8}
    )
    grid_props = (
        grid_props
        or {'axis': 'both', 'alpha': 0.7, 'linestyle': ':'}
    )

    # Generate or reuse labels if not provided
    labels = labels or [
        f"DataFrame {i+1}" for i in range(len(dfs))
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
                color = plt.cm.tab10(i),
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
                color = plt.cm.tab10(i),
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
    error_type: str = 'absolute',    # 'absolute' or 'relative'
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
    
    dfs = are_all_frames_valid(*dfs, ops='validate')
    
    # Validate the chosen error type
    valid_errors = ['absolute', 'relative']
    if error_type not in valid_errors:
        raise ValueError(
            f"Invalid error_type. Choose from {valid_errors}"
        )

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
            f"Generated {matrix_type} matrix for {n_plots} datasets"
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
    title: str = 'Comparative Subsidence Analysis',
    xlabel: str = 'Observation Period',
    ylabel: str = 'Subsidence Rate',
    show_grid: bool=True, 
    grid_props: dict =None,
    savefig: Optional[str]=None, 
    verbose: int = 0,
    **kwargs
) -> plt.Figure:
    """
    Visualizes comparative subsidence trends across multiple datasets with 
    temporal aggregation and confidence intervals.
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


     
