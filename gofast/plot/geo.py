# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Provides geosciences plotting functionalities.
"""
import warnings 
from typing import List, Optional, Dict 

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import numpy as np 
import pandas as pd 

from ..compat.sklearn import validate_params, StrOptions 
from ..core.checks import check_params, is_iterable, validate_depth 
from ..core.handlers import param_deprecated_message
from ..core.plot_manager import default_params_plot, return_fig_or_ax  
from ..decorators import isdf  
from ._config import PlotConfig

__all__=["plot_well"]

@return_fig_or_ax
@default_params_plot(
    savefig=PlotConfig.AUTOSAVE('my_well_plot.png'), 
    fig_size=None 
 )
@validate_params ({ 
    'df': ['array-like'], 
    'cols': ['array-like', None], 
    'depth_arr': ['array-like', None], 
    'ref_arr' : ['array-like', None], 
    'pred_df': ['array-like', None], 
    'error': [StrOptions({ 'raise', 'warn', 'ignore'})], 
    'numeric_only': [bool], 
    })

@isdf 
@param_deprecated_message(
    warning_category=UserWarning, 
    conditions_params_mappings=[
        {
            'param': 'depth_kind',
            'condition': lambda v: v not in { None, 'log'},
            'message': ( 
                "Current version only supports ``depth_kind='log'``."
                " Resetting depth_kind to None"
                ),
            'default': None
        }, 
        { 
            'param': 'titles',
            'condition': lambda v: v is not None,
            'message': ( 
                "Title for each subplot is unused."
                " Track names overshadow it."
                ),
            'default': None
            
        }
    ]
)
@check_params ({ 
    'ref_col': str, 
    'combined_cols':Optional[Dict[str, List[str]]], 
    'kind_mapping': Optional[Dict[str, str]], 
    })
def plot_well(
    df,
    depth_arr=None,
    ref_arr=None,
    pred_df=None,
    ref_col=None,
    cols=None,
    pred_cols=None,
    kind_mapping=None,
    depth_kind=None,
    combined_cols=None,
    agg_plot=False, 
    ignore_index=False,
    index_as_depth=True,
    error='warn',
    titles=None,
    show_grid=True,
    grid_alpha=0.7,
    grid_ls='--',
    sharey=True,
    fig_size=None,
    savefig=None,
    minorticks_on=True,
    subplot_kws=None
):
    r"""
    Plot well logs from a main DataFrame and optional prediction
    DataFrame with a shared or separate depth axis. The depth can be
    provided explicitly via ``depth_arr`` or derived from `df` index
    if `index_as_depth` is True. This function relies on
    `validate_depth` and can optionally leverage
    `arrange_tracks_for_plotting` to manage columns.

    The well logs are typically arranged as vertical tracks, each
    sharing the same depth axis. This aids in comparing multiple
    logs or predictions across the same interval. If `agg_plot`
    is True, the tracks from both data sources are concatenated
    horizontally with no spacing and only the first track displays
    the depth axis. Otherwise, separate sets of tracks can be
    displayed with configurable space in between.

    .. math::
       y = \alpha x + \beta

    Here, :math:`y` is the log response, and :math:`x` is the
    measured index (e.g., depth). The constants :math:`\alpha` and
    :math:`\beta` represent scaling factors in typical well-log
    transformations [1]_.

    Parameters
    ----------
    df : pandas.DataFrame
        Main DataFrame containing well-log columns. If `cols` is
        provided, only those columns in ``df`` are plotted.
    depth_arr : array-like or pandas.Series, optional
        Depth values aligned to ``df``. If None and
        `index_as_depth` is True, the function uses `df`
        index as depth. Otherwise a simple 0..N range is used.
    ref_arr : array-like or pandas.Series, optional
        Reference or ground truth values aligned with
        predictions in ``pred_df``.
    pred_df : pandas.DataFrame or pandas.Series, optional
        Prediction data (e.g., model outputs) to be plotted
        alongside `df`. If `agg_plot` is True, it is
        appended to df tracks.
    ref_col : str, optional
        Column name in the reference DataFrame if passing
        ``ref_arr`` as a multi-column object. Specifies which
        column to plot as reference.
    cols : list of str, optional
        Subset of columns from `df` to plot. If None,
        all columns from `df` are considered.
    pred_cols : list of str, optional
        Subset or rename columns from ``pred_df``. If
        ``pred_df`` is a single Series and exactly one
        name is given, the Series is renamed accordingly.
    kind_mapping : dict, optional
        A mapping dict of columns or track names to their
        plotting type (e.g. `{'Resistivity': 'log'}`). If a
        track is marked 'log', it is plotted using a semilogx.
    depth_kind : {'log', None}, optional
        If `'log'`, use log-scaling on the depth axis.
    combined_cols : dict or list of str, optional
        Group columns in `df` as combined tracks. A dict like
        ``{'Track1': ['GR','RHOB']}`` merges the columns in
        one subplot. A list merges them under a generated name.
    agg_plot : bool, optional
        If True, merges `df` and ``pred_df`` tracks into one
        continuous set of subplots. If False, plots them
        separately with a configurable gap in between.
    ignore_index : bool, optional
        Whether to reset index for `df`, ``pred_df``, and
        others prior to plotting. If True, indices become
        0..N-1.
    index_as_depth : bool, optional
        Whether to treat `df` index as depth if no
        ``depth_arr`` is given.
    error : {'warn', 'raise', 'ignore'}, optional
        Error policy for mismatches or non-monotonic depth.
    titles : list of str, optional
        Titles for each subplot (unused if overshadowed by
        track names).
    show_grid : bool, default True
        Toggles the grid overlay on each track.
    grid_alpha : float, default 0.7
        Transparency factor for the gridlines.
    grid_ls : str, default '--'
        Line style for the grid.
    sharey : bool, default True
        Whether y-axes (depth axes) are shared among subplots.
        Typically True for well logs.
    fig_size : tuple, optional
        Size of the figure (width, height). If None, a
        reasonable default is chosen.
    savefig : str, optional
        If provided, path to save the resulting figure.
    minorticks_on : bool, default True
        Whether to enable minor ticks for added depth
        readability.
    subplot_kws : dict, optional
        Additional keyword arguments passed to
        ``plt.subplots`` or GridSpec.

    Examples
    --------
    >>> from gofast.plot.geo import plot_well
    >>> import pandas as pd
    >>> # Suppose df is a DataFrame with columns: GR, RHOB, NPHI
    >>> df = pd.DataFrame({
    ...     'GR':   [50, 60, 70],
    ...     'RHOB': [2.3, 2.4, 2.35],
    ...     'NPHI': [0.25, 0.22, 0.20]
    ... })
    >>> # Plot these logs with depth as df.index:
    >>> plot_well(df, index_as_depth=True, agg_plot=True)

    Notes
    -----
    In the mathematical sense, if :math:`D` is the depth axis
    and :math:`L_i` are the log values, then:

    .. math::
       \{(D, L_1), (D, L_2), \dots\} \;\in\; \mathbb{R}^2.

    The function overlays them in vertical subplots
    for quick visual correlation.

    See Also
    --------
    `arrange_tracks_for_plotting` :
        Organizes (track_name, columns) tuples for df and
        pred_df.  
    `validate_depth` :
        Aligns DataFrames and checks monotonic depth.

    References
    ----------
    .. [1] Slatt, R.M. "Stratigraphic reservoir characterization
       for petroleum geologists, geophysicists, and engineers",
       2nd Edition, Elsevier, 2013.
    """

    def arrange_tracks_for_plotting(
            df_tracks, pred_tracks, agg_plot=False):
        # If agg_plot is True, combine both sets of tracks into a single list.
        if agg_plot:
            return df_tracks + pred_tracks
        else:
            # Otherwise, return them separately so the 
            # caller can handle them independently.
            return df_tracks, pred_tracks
    
    # If subplot_kws is None, define an empty dict; 
    # we will handle spacing logic ourselves via GridSpec
    # in this implementation.
    if subplot_kws is None:
        subplot_kws = {}

    # Validate depth and align data (not shown; assume
    # 'validate_depth' is externally defined).
    df, pred_df, ref_arr, depth_arr = validate_depth(
        df                  = df,
        pred_df             = pred_df,
        reference           = ref_arr,
        ref_col             = ref_col,
        depth               = depth_arr,
        new_name            = 'Depth',
        rename_depth        = False,
        reset_index         = ignore_index,
        check_monotonic     = True,
        index_as_depth      = index_as_depth,
        allow_index_mismatch= False,
        error               = error,
        as_series           = True,
        check_size          = False
    )

    # If pred_cols is specified, subset or rename pred_df if needed.
    if pred_cols is not None:
        pred_cols = is_iterable(
            pred_cols, 
            exclude_string=True, 
            transform=True
    )

    if isinstance(pred_df, pd.Series):
        if pred_cols and len(pred_cols) == 1:
            pred_df.name = pred_cols[0]
        else:
            if error == 'warn' and pred_cols and len(pred_cols) > 1:
                warnings.warn(
                    "Multiple pred_cols given, but pred_df is a single Series."
                    " Ignoring rename."
                )
    elif isinstance(pred_df, pd.DataFrame):
        if pred_cols is not None:
            pred_df = pred_df[pred_cols]

    # If cols is given, subset df columns accordingly.
    if cols is not None:
        df = df[list(cols)]

    # Build a dict grouping columns if combined_cols is used.
    track_dict = {}
    used_cols  = set()

    if combined_cols is not None:
        if isinstance(combined_cols, dict):
            for track_name, column_list in combined_cols.items():
                track_dict[track_name] = column_list
                used_cols.update(column_list)
        elif isinstance(combined_cols, list):
            track_name = '-'.join(combined_cols)
            track_dict[track_name] = combined_cols
            used_cols.update(combined_cols)

    # Columns in df that aren't in combined_cols become individual tracks.
    for c in df.columns:
        if c not in used_cols:
            track_dict[c] = [c]

    # Convert to list of (track_name, [columns]) for df.
    df_tracks = list(track_dict.items())

    # Build a list of (track_name, [columns]) for pred_df if provided.
    pred_tracks = []
    if pred_df is not None:
        if isinstance(pred_df, pd.DataFrame):
            for col_name in pred_df.columns:
                pred_tracks.append((col_name, [col_name]))
        elif isinstance(pred_df, pd.Series):
            pred_tracks.append((pred_df.name, [pred_df.name]))

    # Use helper to arrange df and pred tracks based on agg_plot.
    arranged = arrange_tracks_for_plotting(
        df_tracks   = df_tracks,
        pred_tracks = pred_tracks,
        agg_plot    = agg_plot
    )

    # If agg_plot is True, we get a single list of tracks (df+pred). 
    # If False, we get a tuple (df_tracks, pred_tracks).
    if agg_plot:
        all_tracks = arranged
        # No spacing between columns when aggregated.
        # We rely on a single row with wspace=0.0 for all subplots.
        if "wspace" not in subplot_kws:
            subplot_kws["wspace"] = 0.0
        df_total_tracks = len(all_tracks)

        # We'll handle depth labeling by only showing y-label on the 
        # first track. Mask the y-label on subsequent columns.

        if fig_size is None:
            fig_size = (3 * df_total_tracks, 10)

        fig = plt.figure(figsize=fig_size)
        gs  = GridSpec(nrows=1, 
                       ncols=df_total_tracks,
                       figure=fig,
                       **subplot_kws)

        axes = []
        for i in range(df_total_tracks):
            ax = fig.add_subplot(gs[0, i])
            axes.append(ax)

        # Now we have one row of subplots. 
        # We'll plot them in a single pass below.
        ax_list = axes

    else:
        df_tracks_only, pred_tracks_only = arranged
        # We want some spacing between df and pred subplots. 
        # For df alone, we might have wspace=0 between its columns,
        # and for pred alone, also wspace=0, 
        # but a bigger gap between the last df column and first pred column.
        # We'll implement that using gridspec with two sub-grids:

        df_total_tracks   = len(df_tracks_only)
        pred_total_tracks = len(pred_tracks_only)
        # We'll define a figure with df_total_tracks + pred_total_tracks 
        # subplots in one row, but a gap (e.g. wspace=0.3) specifically 
        # between the two sets.

        if fig_size is None:
            fig_size = (3 * (df_total_tracks + pred_total_tracks), 10)

        fig = plt.figure(figsize=fig_size)

        # We'll manually allocate the columns:
        #   0..(df_total_tracks-1) for df
        #   then a gap
        #   then pred_total_tracks columns for pred
        # We'll use the width ratios trick to insert some space.
        # Another approach is two separate subplots calls, 
        # but let's do it in one figure:

        # Example approach:
        # total_ncols = df_total_tracks + pred_total_tracks
        # We define a GridSpec with that many columns. We'll set wspace=0 
        # for everything.
        # Then we force a bigger "gap" column or so. But let's do it simpler:
        # We'll define 2 sub-grids: one for df with wspace=0, 
        # another for pred with wspace=0, 
        # plus a big space in between them by adjusting figure margins
        # or using a big hspace/wspace.

        # We'll do 2 columns in the top-level GridSpec: one for df, one for pred.
        # Then each sub-GridSpec has as many columns as needed for df or pred.
        # In between them, we define wspace=some bigger number, e.g. 0.3 or 0.4.

        # Top-level GridSpec
        top_gs = GridSpec(
            nrows=1,
            ncols=2,
            figure=fig,
            width_ratios=[df_total_tracks, pred_total_tracks],
            wspace=0.3  # Space between the two groups
        )
        
        # Nested GridSpec for df
        gs_df = GridSpecFromSubplotSpec(
            nrows=1,
            ncols=df_total_tracks,
            subplot_spec=top_gs[0, 0],  # Use top-level GridSpec as base
            wspace=0.0  # No space among df columns
        )
        
        # Nested GridSpec for pred
        gs_pred = GridSpecFromSubplotSpec(
            nrows=1,
            ncols=pred_total_tracks,
            subplot_spec=top_gs[0, 1],  # Use top-level GridSpec as base
            wspace=0.0  # No space among pred columns
        )

        # We'll gather the axes in an array so we can handle
        # them in a single pass if we want:
        ax_list = []

        # Create subplots for df tracks
        df_axes = []
        for i in range(df_total_tracks):
            ax = fig.add_subplot(gs_df[0, i])
            df_axes.append(ax)
        # Create subplots for pred tracks
        pred_axes = []
        for j in range(pred_total_tracks):
            ax = fig.add_subplot(gs_pred[0, j])
            pred_axes.append(ax)

        ax_list = df_axes + pred_axes

    # Helper function to set a log scale on depth 
    # if requested, then invert axis.
    # since plot_track firstly invert y so no need 
    # to reinvert y again. 
    def maybe_set_depth_scale(ax_, dkind, show_ylabel):
        if dkind == 'log':
            ax_.set_yscale('log')
        # if show_ylabel:
        #     ax.set_ylabel("Depth")
        # else:
        #     # Hide the y-label if we don't want to see it on this subplot.
        #     ax.set_yticklabels([])  # Hide y-tick labels
        #     ax.set_ylabel("")
            
        # ax_.invert_yaxis () : No Need.
            
    # Function to plot columns in a single track. 
    # We optionally overlay reference data
    # in red for pred tracks.
    def plot_track(
        ax,
        track_name,
        cols_in_track,
        data_source,
        depth_values,
        kmapping,
        dkind,
        do_grid,
        show_ylabel,
        ref_data=None
    ):
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for idx_col, col_name in enumerate(cols_in_track):
            if col_name not in data_source.columns:
                continue

            x_vals = data_source[col_name].values
            col_kind = None
            if kmapping and (col_name in kmapping):
                col_kind = kmapping[col_name]
            else:
                # If the combined_cols dict has a single "track_name" = 'Resistivity'
                # and we put 'log' in kind_mapping for 'Resistivity', we might want
                # to adopt that for all columns in that track. We'll do a fallback:
                if kmapping and (track_name in kmapping):
                    col_kind = kmapping.get(track_name)

            line_color = color_cycle[idx_col % len(color_cycle)]
            if col_kind == 'log':
                ax.semilogx(x_vals, depth_values, color=line_color, label=col_name)
            else:
                ax.plot(x_vals, depth_values, color=line_color, label=col_name)

        if ref_data is not None:
            ax.plot(
                ref_data.values, depth_values, color='red', 
                label= ref_data.name if isinstance(
                    ref_data,pd.Series) else 'Reference'
                )

        ax.set_title(track_name)
        ax.set_xlabel("Value")
        if show_ylabel:
            ax.set_ylabel("Depth")
        else:
            # Hide the y-label if we don't want to see it on this subplot.
            ax.set_yticklabels([])  # Hide y-tick labels
            ax.set_ylabel("")
        ax.invert_yaxis()

        if do_grid:
            ax.grid(True, linestyle=grid_ls, alpha=grid_alpha)
        ax.legend()

        if minorticks_on:
            ax.grid(True, which='minor', linestyle=':', alpha=0.5)  
            ax.minorticks_on()

    # If agg_plot is True, we have all_tracks in a single list. 
    # We'll mask depth labeling except for the first track.
    if agg_plot:
        for i, (track_name, columns_in_track) in enumerate(all_tracks):
            current_ax = ax_list[i]

            # Decide if the columns belong to pred_df or df
            if pred_df is not None:
                if (isinstance(pred_df, pd.DataFrame)
                    and columns_in_track[0] in pred_df.columns):
                    data_source = pred_df
                elif (isinstance(pred_df, pd.Series)
                      and columns_in_track[0] == pred_df.name):
                    data_source = pd.DataFrame(pred_df)
                else:
                    data_source = df
            else:
                data_source = df

            # Determine depth array
            if depth_arr is not None:
                depth_vals = depth_arr.values if isinstance(
                    depth_arr, pd.Series) else depth_arr
            else:
                depth_vals = np.arange(len(data_source))

            # Check if we overlay ref_arr
            if (data_source is not df) and (ref_arr is not None):
                reference_to_plot = ref_arr
            else:
                reference_to_plot = None

            # Only the first column has the y-label for depth 
            show_ylabel = (i == 0)

            plot_track(
                ax           = current_ax,
                track_name   = track_name,
                cols_in_track= columns_in_track,
                data_source  = data_source,
                depth_values = depth_vals,
                kmapping     = kind_mapping,
                dkind        = depth_kind,
                do_grid      = show_grid,
                show_ylabel  = show_ylabel,
                ref_data     = reference_to_plot
            )

            maybe_set_depth_scale(current_ax, depth_kind, show_ylabel )

    else:
        # agg_plot is False, so we have two sets of tracks: df_tracks_only,
        # pred_tracks_only.
        # We'll plot df first, then pred, each in separate sub-GridSpec. 
        # The first column in df has the depth label, likewise the first
        # column in pred has it.
        # No space among df columns or among pred columns, but a bigger
        # space between the two sets.
        
        df_tracks_only, pred_tracks_only = arranged
        # Axes = df_axes + pred_axes in that order
        df_axes   = ax_list[:len(df_tracks_only)]
        pred_axes = ax_list[len(df_tracks_only):]

        # Plot df tracks
        for i, (track_name, columns_in_track) in enumerate(df_tracks_only):
            current_ax = df_axes[i]

            if depth_arr is not None:
                depth_vals = depth_arr.values if isinstance(
                    depth_arr, pd.Series) else depth_arr
            else:
                depth_vals = np.arange(len(df))

            # For df, we do not overlay reference data
            reference_to_plot = None

            # Only first column of df has depth label
            show_ylabel = (i == 0)

            plot_track(
                ax           = current_ax,
                track_name   = track_name,
                cols_in_track= columns_in_track,
                data_source  = df,
                depth_values = depth_vals,
                kmapping     = kind_mapping,
                dkind        = depth_kind,
                do_grid      = show_grid,
                show_ylabel  = show_ylabel,
                ref_data     = reference_to_plot
            )
            maybe_set_depth_scale(current_ax, depth_kind, show_ylabel)

        # Plot pred tracks
        if pred_tracks_only:
            for j, (track_name, columns_in_track) in enumerate(pred_tracks_only):
                current_ax = pred_axes[j]

                if pred_df is not None:
                    if (isinstance(pred_df, pd.DataFrame)
                        and columns_in_track[0] in pred_df.columns):
                        data_source = pred_df
                    elif (isinstance(pred_df, pd.Series)
                          and columns_in_track[0] == pred_df.name):
                        data_source = pd.DataFrame(pred_df)
                    else:
                        data_source = df
                else:
                    data_source = df

                if depth_arr is not None:
                    depth_vals = depth_arr.values if isinstance(
                        depth_arr, pd.Series) else depth_arr
                else:
                    depth_vals = np.arange(len(data_source))

                # If track is from pred_df, we might overlay ref_arr
                if (data_source is not df) and (ref_arr is not None):
                    reference_to_plot = ref_arr
                else:
                    reference_to_plot = None

                # Only first column of pred has the depth label
                show_ylabel = (j == 0)

                plot_track(
                    ax            = current_ax,
                    track_name    = track_name,
                    cols_in_track = columns_in_track,
                    data_source   = data_source,
                    depth_values  = depth_vals,
                    kmapping      = kind_mapping,
                    dkind         = depth_kind,
                    do_grid       = show_grid,
                    show_ylabel   = show_ylabel,
                    ref_data      = reference_to_plot
                )
                maybe_set_depth_scale(current_ax, depth_kind, show_ylabel)

    # Final layout and optional save
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
    plt.show()