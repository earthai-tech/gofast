# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Provides functions for visualizing feature distributions 
in a grid-based layout plots.
"""

from typing import Optional, List
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from ..compat.pandas import select_dtypes 
from ..core.checks import exist_features, check_non_emptiness 
from ..core.plot_manager import ( 
    is_valid_kind, 
    set_axis_grid, 
    default_params_plot
)
from ..decorators import isdf  
from ..utils.generic_utils import check_group_column_validity 
from ..utils.validator import filter_valid_kwargs, is_frame 
from ._config import PlotConfig
from .utils import flex_figsize 

__all__=['plot_feature_dist_grid']

@default_params_plot(
    savefig=PlotConfig.AUTOSAVE('my_feature_distribution_grid_plot.png'), 
    fig_size=None 
 )
@check_non_emptiness 
@isdf
def plot_feature_dist_grid(
    df: pd.DataFrame,
    group_col: str,
    features: Optional[List[str]] = None,
    kind: str = 'violin',
    max_cols: int = 3,
    max_unique: int = 10,
    auto_bin: bool = False,
    bins: int = 4,
    bin_labels: Optional[List[str]] = None,
    figsize: tuple = None, # (18, 10),
    palette: str = "Set2",
    title: Optional[str] = None,
    savefig: Optional[str] = None,
    show_grid: bool = True,
    grid_props: Optional[dict] = None,
    style: str = None,
    fontsize: int = 10,
    rotation: int = 45,
    tight_layout: bool = True,
    dpi: int = 300,
    **kw
):
    """
    Generates a grid of distribution plots (box or violin) for a set of
    numeric features, optionally grouped by a specified column. The
    function leverages `check_group_column_validity` to ensure that
    ``group_col`` is suitably categorical. It also applies Seaborn's
    violin or box plots, organizing them in a grid layout. Internal
    helpers like `set_axis_grid` handle grid lines, while
    `filter_valid_kwargs` passes only recognized arguments to the
    plotting function.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the columns to be plotted.
    group_col : str
        The name of a column in ``df`` to group or partition the
        plots. Must be categorical or have limited unique numeric
        values.
    features : list of str, optional
        A list of numeric feature columns to plot. If None,
        numeric columns from ``df`` are automatically selected
        (excluding `group_col`).
    kind : {'violin', 'box'}, optional
        Specifies the type of plot to use for each feature:
        - ``"violin"`` : Seaborn violinplot
        - ``"box"`` : Seaborn boxplot
    max_cols : int, optional
        Maximum number of plots per row in the grid layout.
    max_unique : int, optional
        Maximum number of unique numeric values allowed in
        ``group_col``. Beyond this threshold, the function will
        attempt or suggest binning.
    auto_bin : bool, optional
        If True, and if ``group_col`` is numeric with too many
        unique values, automatically convert it into quantile
        bins.
    bins : int, optional
        Number of quantile bins to create if ``auto_bin`` is True.
    bin_labels : list of str, optional
        Custom labels for the bins when `group_col` is quantile
        binned. If None, default bin labels (e.g. "Q1", "Q2", ...)
        are used.
    figsize : tuple of float, optional
        The size of the figure (width, height), in inches.
    palette : str, optional
        A Seaborn or matplotlib palette name, e.g. ``"Set2"``
        or ``"viridis"``.
    title : str, optional
        Overall title for the plot grid. If None, no title
        is displayed.
    savefig : str, optional
        File path to save the figure. If None, the figure
        is not saved.
    show_grid : bool, optional
        Whether to show grid lines on each subplot. Handled by
        `set_axis_grid`.
    grid_props : dict, optional
        Additional properties passed to the grid function,
        e.g. ``{"which": "both", "linestyle": "--"}``.
    style : str, optional
        A Seaborn style string (e.g. ``"whitegrid"`` or
        ``"dark"``). If None, uses default.
    fontsize : int, optional
        Font size for subplot titles and optional
        figure title.
    rotation : int, optional
        Degree of rotation for the x-axis tick labels.
    tight_layout : bool, optional
        If True, calls ``plt.tight_layout()`` to reduce
        overlap.
    dpi : int, optional
        Dots-per-inch for the figure, used when saving.
    **kw
        Additional keyword arguments forwarded to the
        respective Seaborn plot function (boxplot or
        violinplot). Filtered by `filter_valid_kwargs`.
    
    Notes
    -----
    - The function automatically checks if `group_col` is suitable
      for categorical grouping via
      `check_group_column_validity`. If numeric with too many
      distinct values, it can optionally bin (quantile-based).
    - Seaborn settings (e.g. figure style) are applied globally.
    
    Examples
    --------
    >>> from gofast.plot.grid import plot_feature_dist_grid
    >>> import pandas as pd
    >>> df_example = pd.DataFrame({
    ...     "group": ["A", "A", "B", "B", "C"],
    ...     "feature1": [5.2, 6.3, 5.9, 7.1, 8.0],
    ...     "feature2": [10.1, 9.8, 10.5, 11.2, 12.0]
    ... })
    >>> # Plot with violin, grouping by 'group'
    >>> plot_feature_dist_grid(
    ...     df_example, group_col="group",
    ...     features=["feature1", "feature2"],
    ...     kind="violin"
    ... )
    
    See Also
    --------
    check_group_column_validity : Verifies if a column can be
        treated as categorical; can auto-bin it if numeric.
    is_valid_kind : Ensures the chosen plot kind is among the
        accepted types.
    set_axis_grid : Toggles the grid lines on axes.
    filter_valid_kwargs : Filters only recognized parameters for
        a given function.
    sns.violinplot : The Seaborn violin plot function used when
        ``kind="violin"``.
    sns.boxplot : The Seaborn box plot function used when
        ``kind="box"``.
    
    References
    ----------
    .. [1] W. S. Cleveland. "The Elements of Graphing Data."
       Wadsworth, 1985.
    
    """
    # Validate that df is indeed a DataFrame.
    # The function 'is_frame' raises if not.
    is_frame(
        df,
        df_only=True,
        objname="Dataframe 'df'"
    )

    # Create a copy to avoid mutating
    # the original DataFrame.
    df_copy = df.copy()

    # Check that group_col is present.
    # 'exist_features' ensures group_col
    # is in df columns.
    exist_features(
        df_copy,
        features=group_col,
        name='Group col'
    )

    # Validate the plot kind. If not box/violin,
    # an error is raised.
    kind = is_valid_kind(
        kind,
        valid_kinds=['box', 'violin']
    )

    # Set Seaborn style if provided.
    sns.set_style(style)

    # Ensure group_col is suitable for grouping.
    # May auto-bin if numeric with too many values.
    df_copy = check_group_column_validity(
        df_copy,
        group_col=group_col,
        max_unique=max_unique,
        auto_bin=auto_bin,
        bins=bins,
        bin_labels=bin_labels,
        ops='validate',
        error='warn'
    )

    # If no specific features are given,
    # pick numeric columns except group_col.
    if features is None:
        features = select_dtypes(
            df_copy,
            incl=np.number,
            return_columns=True
        )
        features = [
            col for col in features
            if col != group_col
        ]

    # Calculate total plots and figure layout.
    n_plots = len(features)
    
    if len(features)< max_cols: 
        # avoid unecessary columns
        max_cols = len(features )
        
    n_rows = int(
        np.ceil(n_plots / max_cols)
    )
    dummy_m = np.zeros ((len(features), len(df_copy[group_col].unique())))
    figsize = flex_figsize(
        dummy_m, figsize=figsize, 
        base =(8, 6), 
        min_base=(4, 2)
    )
    # Create the figure and axes grid.
    fig, axes = plt.subplots(
        n_rows,
        max_cols,
        figsize=figsize
    )
    axes = axes.flatten()

    # Iterate over each feature to plot.
    for i, feature in enumerate(features):
        ax = axes[i]
        # Filter only recognized kwargs
        # for the chosen Seaborn plot.
        if kind == 'violin':
            kw_filtered = filter_valid_kwargs(
                sns.violinplot,
                kw
            )
            sns.violinplot(
                data=df_copy,
                x=group_col,
                y=feature,
                palette=palette,
                ax=ax,
                **kw_filtered
            )
        elif kind == 'box':
            kw_filtered = filter_valid_kwargs(
                sns.violinplot,
                kw
            )
            sns.boxplot(
                data=df_copy,
                x=group_col,
                y=feature,
                palette=palette,
                ax=ax,
                **kw_filtered
            )

        # Title each subplot and rotate x-ticks
        # if needed.
        ax.set_title(
            feature,
            fontsize=fontsize
        )
        ax.tick_params(
            axis='x',
            rotation=rotation
        )

        # Set or remove grid lines.
        set_axis_grid(
            ax,
            show_grid=show_grid,
            grid_props=grid_props
        )

    # Hide unused subplots if the grid is
    # larger than needed.
    for j in range(n_plots, len(axes)):
        axes[j].axis('off')

    # Set a figure title if provided.
    if title:
        fig.suptitle(
            title,
            fontsize=fontsize + 2
        )

    # Use tight_layout to manage spacing
    # if requested.
    if tight_layout:
        plt.tight_layout(
            rect=[0, 0, 1, 0.95]
            if title
            else None
        )

    # Save figure if a path is given.
    if savefig:
        plt.savefig(
            savefig,
            dpi=dpi
        )

    # Finally, show the plots.
    plt.show()

    # Return the figure object for further use.
    return fig

   