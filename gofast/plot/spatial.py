# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Provides spatial plotting functionality and includes 
tools to visualize and analyze spatial data.
"""
from __future__ import annotations 

import math
import warnings
from numbers import Real
import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import griddata
from sklearn.cluster import KMeans

from ..api.types import ( 
    Optional, 
    Union, 
    List, 
    Tuple, 
    Dict, 
    Any 
)
from ..api.types import DataFrame
from ..core.checks import ( 
    check_features_types, 
    check_spatial_columns,
    exist_features,
    are_all_frames_valid, 
)
from ..core.handlers import columns_manager
from ..core.io import is_data_readable, export_data  
from ..core.plot_manager import ( 
    default_params_plot, 
    return_fig_or_ax, 
    set_axis_grid 
)
from ..compat.sklearn import ( 
    validate_params,
    StrOptions, 
)
from ..decorators import isdf
from ..utils.validator import ( 
    validate_positive_integer,
    filter_valid_kwargs
)
from ._config import PlotConfig
from ._d_cms import update_box_kws # noqa 
from .utils import make_plot_colors 

__all__=[
        'plot_categorical_feature',
        'plot_dist',
        'plot_categories_dist',
        'plot_spatial_features', 
        'plot_spatial_clusters', 
        'plot_hotspot_map', 
 ]

@default_params_plot(
    savefig=PlotConfig.AUTOSAVE('my_spatial_clusters_plot.png'))
@isdf 
def plot_spatial_clusters(
    df: DataFrame,
    spatial_cols: Optional[Tuple[str, str]] = None,
    n_clusters: int = 5,
    method: str = "kmeans",
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    show_grid: bool = True,
    grid_props: Optional[Dict[str, Any]] = None,
    cmap: str = "tab10",
    plot_centroids: bool = True,
    legend_position: str = "right",
    export_result: bool=False, 
    figsize=None,
    **scatter_kws: Any
) -> plt.Axes:
    """
    Plot spatial clusters on a 2D coordinate system.

    This function applies ``method`` (currently only
    `'kmeans'`) on the spatial coordinates provided by
    `spatial_cols` in ``df`` to group samples into
    ``n_clusters``. It uses `check_spatial_columns` internally
    to validate columns, and returns a Matplotlib Axes object
    with scatter points colored by cluster assignment. Decorators
    `isdf` and `default_params_plot` are applied for streamlined
    handling of DataFrame inputs and plot configurations.

    Parameters
    ----------
    df : DataFrame
        Input pandas DataFrame containing spatial data.
        Must include the columns specified in
        `spatial_cols`.
    spatial_cols : tuple of str, optional
        Tuple specifying the longitude and latitude
        columns, by default ``("longitude","latitude")``.
    n_clusters : int, default=5
        Number of clusters to form.
    method : str, default="kmeans"
        Clustering algorithm name. Currently, only
        ``"kmeans"`` is supported.
    ax : matplotlib.axes.Axes, optional
        Existing Matplotlib Axes on which the clusters
        are drawn. If ``None``, a new figure and axes
        are created.
    title : str, optional
        Plot title. Defaults to
        ``"Spatial Clustering (Kmeans, k=<n_clusters>)"``.
    show_grid : bool, default=True
        Whether to display a grid on the plot.
    grid_props : dict, optional
        Keyword arguments passed to the Axes grid
        configuration.
    cmap : str, default="tab10"
        Name of the Matplotlib colormap used for
        cluster coloring.
    plot_centroids : bool, default=True
        If True, displays the cluster centroids on the
        scatter plot using a distinctive marker.
    legend_position : str, default="right"
        Position of the colorbar for cluster IDs.
    figsize : tuple, optional
        Figure size in inches, e.g. ``(8,6)``. Used only
        if no existing axes is provided.
    **scatter_kws : Any
        Additional keyword arguments passed to the
        underlying Matplotlib scatter function.

    Returns
    -------
    matplotlib.axes.Axes
        Axes object with the spatial clusters, centroids
        (if requested), colorbar, and optional grid.

    Notes
    -----
    This function is helpful for exploring spatial
    patterns or checking how your data might cluster
    geographically. The underlying `'kmeans'` method
    is powered by scikit-learn's KMeans.
    
    Mathematically, when using KMeans, it solves the following
    objective:

    .. math::
       J = \\sum_{i=1}^{n} \\min_{\\mu_j} \\lVert x_i - \\mu_j \\rVert^2

    where :math:`x_i` is a coordinate point in :math:`\\mathbb{R}^2`,
    and :math:`\\mu_j` is the centroid for cluster :math:`j`. The
    result is a set of clusters that minimizes intra-cluster
    distance [1]_.
    
    Examples
    --------
    >>> from gofast.plot.spatial import plot_spatial_clusters
    >>> import pandas as pd
    >>> # Create a sample DataFrame with columns:
    >>> # ["longitude","latitude"].
    >>> data = {
    ...     "longitude":[0.1,0.2,0.4,2.2,2.3],
    ...     "latitude":[1.0,1.1,0.9,2.1,2.0]
    ... }
    >>> df = pd.DataFrame(data)
    >>> # Example usage:
    >>> ax = plot_spatial_clusters(
    ...     df=df,
    ...     spatial_cols=("longitude","latitude"),
    ...     n_clusters=2,
    ...     method="kmeans",
    ...     plot_centroids=True
    ... )

    See Also
    --------
    check_spatial_columns : Validates that the DataFrame
        has the necessary spatial columns.
    isdf : Decorator that ensures the input is a
        pandas DataFrame.
    default_params_plot : Decorator for applying
        default parameters and saving plots.

    References
    ----------
    .. [1] Pedregosa et al. *Scikit-learn: Machine Learning
       in Python*, JMLR 12, pp. 2825-2830, 2011.
    """

    # Set default spatial columns and validate input
    spatial_cols = spatial_cols or ("longitude", "latitude")
    _validate_inputs(df, spatial_cols, method)
    
    # Unpack spatial column names
    lon_col, lat_col = spatial_cols
    
    # Create axis if not provided
    if ax is None: 
        fig, ax = plt.subplots (figsize = figsize )
    # ax = ax or plt.gca()
    
    # Clean data and handle missing values
    df_clean, coords = _prepare_data(df, lon_col, lat_col)
    
    # Perform clustering
    cluster_labels, centroids = _perform_clustering(
        method=method,
        coords=coords,
        n_clusters=n_clusters
    )
    
    # Create plot with enhanced styling
    ax = _create_plot(
        df=df_clean,
        ax=ax,
        lon_col=lon_col,
        lat_col=lat_col,
        cluster_labels=cluster_labels,
        cmap=cmap,
        centroids=centroids if plot_centroids else None,
        title=title or f"Spatial Clustering ({method.title()}, k={n_clusters})",
        show_grid=show_grid,
        grid_props=grid_props,
        legend_position=legend_position,
        **scatter_kws
    )

    return ax

def _validate_inputs(
    df: DataFrame,
    spatial_cols: Tuple[str, str],
    method: str
) -> None:
    """Validate input data and parameters."""
    check_spatial_columns(df, spatial_cols)

    if method.lower() != "kmeans":
        raise NotImplementedError(
            f"Method '{method}' not supported. "
            "Currently only 'kmeans' available"
        )

def _prepare_data(
    df: DataFrame,
    lon_col: str,
    lat_col: str
) -> Tuple[DataFrame, np.ndarray]:
    """Clean and prepare spatial data for clustering."""
    df_clean = df.dropna(subset=[lon_col, lat_col]).copy()
    if df_clean.empty:
        raise ValueError(
            "No valid coordinates remaining after NaN removal"
        )
    return df_clean, df_clean[[lon_col, lat_col]].values

def _perform_clustering(
    method: str,
    coords: np.ndarray,
    n_clusters: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Execute clustering algorithm and return labels + centroids."""
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init="auto"
    ).fit(coords)
    return kmeans.labels_, kmeans.cluster_centers_

def _create_plot(
    df: DataFrame,
    ax: plt.Axes,
    lon_col: str,
    lat_col: str,
    cluster_labels: np.ndarray,
    cmap: str,
    centroids: Optional[np.ndarray],
    title: str,
    show_grid: bool,
    grid_props: Optional[Dict[str, Any]],
    legend_position: str,
    **scatter_kws: Any
) -> plt.Axes:
    """Generate the actual visualization with enhanced styling."""
    # Configure default plot parameters
    scatter_params = {
        "s": 40,
        "edgecolor": "w",
        "linewidth": 0.5,
        "alpha": 0.8,
        **scatter_kws
    }
    
    # Create scatter plot
    scatter = ax.scatter(
        x=df[lon_col],
        y=df[lat_col],
        c=cluster_labels,
        cmap=cmap,
        **scatter_params
    )
    
    # Add centroids if requested
    if centroids is not None:
        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            marker="X",
            s=200,
            c="red",
            edgecolor="k",
            label="Centroids"
        )
    
    # Configure axis labels and title
    ax.set(
        xlabel=lon_col,
        ylabel=lat_col,
        title=title,
        aspect="equal"
    )
    
    # Configure grid
    grid_props = grid_props or {"linestyle": ":", "alpha": 0.5}
    ax.grid(show_grid, **grid_props)
    
    # Add professional colorbar
    cbar = plt.colorbar(
        scatter,
        ax=ax,
        location=legend_position,
        shrink=0.8
    )
    cbar.set_label("Cluster ID", rotation=270, labelpad=15)
    
    # Add subtle background styling
    sns.despine(ax=ax, offset=5)
    ax.set_facecolor("#f5f5f5")
    
    return ax

@return_fig_or_ax
@default_params_plot(
    savefig=PlotConfig.AUTOSAVE('my_distribution_plot.png'))
@validate_params ({ 
    'df': ['array-like'], 
    'x_col': [str], 
    'y_col': [str], 
    'z_cols': ['array-like', str], 
    'kind': [StrOptions({'scatter', 'hexbin', 'density'})], 
    'max_cols': [Real]
    })
@isdf 
def plot_dist(
    df: DataFrame,
    x_col: str,
    y_col: str,
    z_cols: List[str],
    kind: str = 'scatter',
    axis_off: bool = True,
    max_cols: int = 3,
    cmap='viridis', 
    s=10, 
    savefig=None,
):
    r"""
    Plot multiple distribution datasets on a grid of subplots for 
    comprehensive spatial analysis. This function generates a grid of 
    subplots illustrating the variation of multiple `z`-axis variables 
    (``z_cols``) against spatial coordinates defined by `x_col` and 
    `y_col`. Depending on the chosen `kind`, it can create scatter, 
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
    on the chosen `kind`:
    
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
    kind : str, optional
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
    cmap: str,  
       Matplotlib colormap plot. Default is ``'viridis'``. 
       
    s : int, default 10
        Size of the points in the scatter plot.
       
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
    >>> from gofast.plot.spatial import plot_dist
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'longitude': [1,2,3,4],
    ...     'latitude' : [10,10,10,10],
    ...     'subsidence_2018': [5,6,7,8],
    ...     'subsidence_2022': [3,4,2,1]
    ... })
    >>> plot_dist(df, x_col='longitude', y_col='latitude',
    ...           z_cols=['subsidence_2018', 'subsidence_2022'],
    ...           kind='scatter', axis_off=True, max_cols=2)

    See Also
    --------
    matplotlib.pyplot.scatter : For basic scatter plot creation.
    matplotlib.pyplot.hexbin : For hexagonal binning visualization.
    seaborn.kdeplot : For kernel density estimation plots.
    gofast.plot.spatial.plot_distributions: 
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
    if kind == 'density':
        for col in z_cols:
            if (df[col] < 0).any():
                raise ValueError(
                    f"Negative values found in '{col}'. Seaborn kdeplot cannot "
                    "handle negative weights. Please provide non-negative values "
                    "or choose a different kind."
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

        if kind == 'scatter':
            # Create a scatter plot
            ax.scatter(
                df[x_col],
                df[y_col],
                c=df[z_col],
                cmap=cmap,
                s=s,
                alpha=0.7,
                norm=plt.Normalize(vmin=overall_min, vmax=overall_max)
            )
        elif kind == 'hexbin':
            # Create a hexbin plot
            ax.hexbin(
                df[x_col],
                df[y_col],
                C=df[z_col],
                gridsize=50,
                cmap=cmap,
                reduce_C_function=np.mean,
                norm=plt.Normalize(vmin=overall_min, vmax=overall_max)
            )
        elif kind == 'density':
            # Create a density (kde) plot
            # seaborn.kdeplot returns a QuadContourSet, not directly used afterward,
            # but that's fine. We don't need to assign it.
            sns.kdeplot(
                x=df[x_col],
                y=df[y_col],
                weights=df[z_col],
                fill=True,
                cmap=cmap,
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
    if kind in ['scatter', 'hexbin', 'density']:
        # Create a ScalarMappable for the colorbar
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
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


@return_fig_or_ax(return_type ='ax')
@isdf 
def plot_categories_dist(
    df: DataFrame,
    category_column: str,  
    continuous_bins: Union[str, List[float]] = 'auto',  
    categories: Optional[List[str]] = None,  
    filter_categories: Optional[List[str]] = None,
    spatial_cols: tuple = ('longitude', 'latitude'), 
    cmap: str = 'coolwarm',  
    kind: str = 'scatter', 
    alpha: float = 0.7, 
    show_grid:bool=True, 
    axis_off: bool = False,  
    grid_props: dict =None, 
    export_categories:bool=False, 
    savefile: Optional[str]=None, 
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

    export_categories : bool, optional
        A boolean flag that indicates whether to export data based on categories.
        If set to True, the function will check for a 'category' column in
        the DataFrame. When present, it will filter the data using the 
        'filter_categories' parameter (if provided) before exporting. 
        If absent, a warning is issued and the entire dataset is exported.
        Defaults to False.
    
    savefile : Optional[str], optional
        The file path or name where the exported data will be saved.
        If provided, the data will be written to this specified location.
        If None, the function may use a default path or handle the export
         differently. Defaults to None.

    cmap : str, default='coolwarm'
        The colormap to use for the visualization. This parameter utilizes matplotlib's 
        colormap names (e.g., `'viridis'`, `'plasma'`, `'inferno'`, `'magma'`, 
        `'cividis'`, etc.).

    kind : str, default='scatter'
        The type of plot to generate. 
        
        - `'scatter'`: Generates a scatter plot.
        - `'hexbin'`: Generates a hexbin plot, suitable for large datasets.
        
        Raises a `ValueError` if an unsupported `kind` is provided.

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
    >>> from gofast.plot.spatial import plot_spatial_distribution

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
    ...     kind='scatter'
    ... )
    
    >>> # Plot 'moderate' and 'severe' subsidence
    >>> plot_spatial_distribution(
    ...     df=df,
    ...     category_column='subsidence',
    ...     categories=['minimal', 'moderate', 'severe'],
    ...     filter_categories=['moderate', 'severe'],
    ...     kind='scatter'
    ... )
    
    >>> # Plot all categories
    >>> plot_spatial_distribution(
    ...     df=df,
    ...     category_column='subsidence',
    ...     categories=['minimal', 'moderate', 'severe'],
    ...     filter_categories=None,
    ...     kind='scatter'
    ... )

    Notes
    -----
    - The function automatically determines whether the `category_column` is 
      categorical or continuous based on its data type.
    - When categorizing continuous data, ensure that the provided `continuous_bins` 
      comprehensively cover the data range to avoid missing data points.
    - The legend in the plot dynamically adjusts based on the `filter_categories` 
      parameter, displaying only the relevant categories.

    The categorization of continuous variables is performed using either user-defined
    bins or the Freedman-Diaconis rule to determine an optimal bin width:
    
    .. math::
        \text{Bin Width} = 2 \times \frac{\text{IQR}}{n^{1/3}}
    
    where :math:`\text{IQR}` is the interquartile range of the data and :math:`n`
    is the number of observations.
    
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
    check_spatial_columns(df , spatial_cols=spatial_cols)
    
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
    
    if export_categories:
        # Check if the DataFrame contains the 'category' column.
        if 'category' in df.columns:
            # If specific categories are provided, 
            # filter the DataFrame accordingly.
            if filter_categories:
                ex_data = df[df['category'].isin(filter_categories)]
            else:
                ex_data = df.copy()
        else:
            # Warn the user that no category column was found,
            #so the entire dataset will be exported.
            warnings.warn(
                "'category' column not found in the DataFrame."
                " Exporting the entire dataset.",
                UserWarning
            )
            ex_data = df.copy()
    
        # Export the processed data using the specified file paths.
        export_data(ex_data, file_paths=savefile, overwrite=True, index=False)

    # Plot the spatial distribution using selected plot type
    plt.figure(figsize=figsize)

    if kind == 'scatter':
        # Scatter plot using longitude, latitude as the axes
        sns.scatterplot(
            x='longitude',
            y='latitude',
            hue='category',
            data=df,
            palette=cmap,
            alpha=alpha
        )
    elif kind == 'hexbin':
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
            f"Unsupported kind: {kind}"
        )

    # Labels and title
    plt.title(f"Spatial Distribution of {category_column}")
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    if kind == 'scatter':
        plt.legend(
            title='Categories',
            bbox_to_anchor=(1.05, 1),
            loc='upper left'
        )
 
    
    
    if not show_grid: 
        plt.grid(False)
    else: 
        grid_props = grid_props  or {"linestyle": ':', 'alpha': 0.7}
        plt.grid(True, **grid_props)
        
    if axis_off:
        plt.axis('off')
    
    # Show plot
    plt.tight_layout()
    plt.show()

@return_fig_or_ax(return_type ='ax')
@is_data_readable 
@isdf 
def plot_categorical_feature(
    data,
    feature,
    dates=None,
    dt_col='year',
    x_col='longitude',
    y_col='latitude',
    cmap='tab10',
    figsize=None,
    s=10,
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
        `dt_col` if `dates` are provided.

    feature : str
        The name of the categorical feature to plot. This feature
        must exist in `data`.

    dates : scalar, list, or array-like, optional
        Dates or times to plot. If provided, the function will
        create subplots for each date specified. If `None`, the
        feature is plotted without considering dates.

    dt_col : str, default ``'year'``
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

    s : int, default 10
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
    >>> from gofast.plot.spatial import plot_categorical_feature
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
    ...     s=20,
    ...     legend_loc='upper right'
    ... )
    >>> # Plotting with dates
    >>> plot_categorical_feature(
    ...     data,
    ...     feature='category',
    ...     dates=[2018, 2019, 2020],
    ...     dt_col='year',
    ...     x_col='longitude',
    ...     y_col='latitude',
    ...     s=20,
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
    if isinstance (feature, (list, tuple)): 
        feature = feature[0]
        
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

        if dt_col not in data.columns:
            raise ValueError(f"Column '{dt_col}' not found in data.")

        if np.issubdtype(data[dt_col].dtype, np.datetime64):
            data[dt_col] = pd.to_datetime(data[dt_col])
            dates = [pd.to_datetime(d) for d in dates]
            data_dates = data[dt_col].dt.normalize().unique()
            dates_normalized = [d.normalize() for d in dates]
            missing_dates = set(dates_normalized) - set(data_dates)
            if missing_dates:
                missing_dates_str = ', '.join(
                    [d.strftime('%Y-%m-%d') for d in missing_dates]
                )
                raise ValueError(f"Dates {missing_dates_str} not found in data.")
            ncols = len(dates)
        else:
            data_dates = data[dt_col].unique()
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
            if np.issubdtype(data[dt_col].dtype, np.datetime64):
                date_normalized = pd.to_datetime(date).normalize()
                subset = data[data[dt_col].dt.normalize() == date_normalized]
                date_str = date_normalized.strftime('%Y-%m-%d')
            else:
                subset = data[data[dt_col] == date]
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
                s=s,
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


@return_fig_or_ax
@is_data_readable 
@isdf 
def plot_spatial_features(
    data,
    features,
    dates=None,
    dt_col="year",
    x_col='longitude',
    y_col='latitude',
    colormaps=None,
    figsize=None,
    s=10,
    marker='o',
    kind='scatter',
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
        the `dt_col` if `dates` are provided.

    features : list of str
        List of feature names to plot. Each feature must exist
        in `data`. A subplot will be created for each feature.

    dates : str or datetime-like or list of str or datetime-like, optional
        Dates or times to plot. Each date must correspond to an entry
        in the `dt_col` of `data`. If `None`, the function plots
        the features without considering dates.

    dt_col : str, default ``'year'``
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

    s : int, default 10
        Size of the points in the scatter plot.

    marker : str, default ``'o'``
        Marker style for scatter plots.

    kind : {'scatter', 'hexbin', 'contour'}, default ``'scatter'``
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

    - For ``kind='scatter'``, it creates a scatter plot using
      ``matplotlib.pyplot.scatter``.

    - For ``kind='hexbin'``, it creates a hexbin plot using
      ``matplotlib.pyplot.hexbin``.

    - For ``kind='contour'``, it creates a contour plot by
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
    >>> from gofast.plot.spatial import plot_spatial_features
    >>> plot_spatial_features(
    ...     data=df,
    ...     features=['temperature', 'humidity'],
    ...     dates=['2023-01-01', '2023-06-01'],
    ...     dt_col='date',
    ...     x_col='lon',
    ...     y_col='lat',
    ...     colormaps=['coolwarm', 'YlGnBu'],
    ...     s=15,
    ...     kind='scatter',
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

        # Check that 'dt_col' exists
        if dt_col not in data.columns:
            raise ValueError(f"Column '{dt_col}' not found in data.")

        # Depending on the type of 'dt_col', process accordingly
        if np.issubdtype(data[dt_col].dtype, np.datetime64):
            # If dt_col is datetime, convert dates to datetime
            data[dt_col] = pd.to_datetime(data[dt_col])

            # Convert dates parameter to datetime
            dates = [pd.to_datetime(d) for d in dates]

            # Normalize dates to remove time component
            data_dates = data[dt_col].dt.normalize().unique()
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
            # dt_col is not datetime, treat as categorical or numeric
            data_dates = data[dt_col].unique()
            missing_dates = set(dates) - set(data_dates)
            if missing_dates:
                missing_dates_str = ', '.join(map(str, missing_dates))
                raise ValueError(f"Dates {missing_dates_str} not found in data.")

            ncols = len(dates)
    else:
        ncols = 1

    features = columns_manager(features, empty_as_none =False)
    nrows = len(features)
 
    colormaps = columns_manager(colormaps) 
    
    if colormaps is None:
        colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']

    colormaps = make_plot_colors (
        features, colormaps, cmap_only=True, 
        get_only_names= True 
    )
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
                if np.issubdtype(data[dt_col].dtype, np.datetime64):
                    # Normalize date to remove time component
                    date_normalized = pd.to_datetime(date).normalize()
                    subset = data[
                        (data[dt_col].dt.normalize() == date_normalized)
                        & data[feature].notnull()
                    ]
                    date_str = date_normalized.strftime('%Y-%m-%d')
                else:
                    subset = data[
                        (data[dt_col] == date)
                        & data[feature].notnull()
                    ]
                    date_str = str(date)

                x = subset[x_col].values
                y = subset[y_col].values
                c = subset[feature].values

                if kind == 'scatter':
                    kwargs = filter_valid_kwargs(ax.scatter, kwargs)
                    sc = ax.scatter(
                        x,
                        y,
                        c=c,
                        cmap=cmap,
                        norm=norm,
                        s=s,
                        marker=marker,
                        **kwargs
                    )
                elif kind == 'hexbin':
                    kwargs = filter_valid_kwargs(ax.hexbin, kwargs)
                    sc = ax.hexbin(
                        x,
                        y,
                        C=c,
                        gridsize=50,
                        cmap=cmap,
                        norm=norm,
                        **kwargs
                    )
                elif kind == 'contour':
                    kwargs = filter_valid_kwargs(ax.contourf, kwargs)
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
                    raise ValueError(f"Unsupported kind: {kind}")

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

            if kind == 'scatter':
                kwargs = filter_valid_kwargs(ax.scatter, kwargs)
                sc = ax.scatter(
                    x,
                    y,
                    c=c,
                    cmap=cmap,
                    norm=norm,
                    s=s,
                    marker=marker,
                    **kwargs
                )
            elif kind == 'hexbin':
                kwargs = filter_valid_kwargs(ax.hexbin, kwargs)
                sc = ax.hexbin(
                    x,
                    y,
                    C=c,
                    gridsize=50,
                    cmap=cmap,
                    norm=norm,
                    **kwargs
                )
            elif kind == 'contour':
                kwargs = filter_valid_kwargs(ax.contourf, kwargs)
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
                raise ValueError(f"Unsupported kind: {kind}")

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
    
def plot_hotspot_map(
    *dfs: pd.DataFrame,
    target_col: str,
    spatial_cols: Tuple[str, str] = ('longitude', 'latitude'),
    labels: Optional[List[str]]   = None,
    threshold: Optional[Union[
        str, float
    ]]                           = None,
    cmap: str                     = 'viridis',
    size: Union[float, str]      = 10,
    figsize: Tuple[float, float] = (12, 10),
    title: str                   = 'Hotspot Map',
    thresh_label: str            = None,
    cbar_label: str              = None,
    basemap: bool                = True,
    use_gpd: bool                = "auto",
    crs: str                     = "EPSG:4326",
    epsg                          = 3857,
    axis_off                     = False,
    savefig: Optional[str]       = None,
    show_grid: bool              = True,
    grid_props: dict             = None,
    s                            = 50,
    verbose: int                 = 0,
    **kwargs
) -> plt.Figure:
    r"""
    Generates a spatial visualization of potential
    hotspots using geographic coordinates from one or
    more DataFrames. The method `plot_hotspot_map`
    creates a scatter map of values in `target_col`
    over a specified region, optionally overlaying a
    base map and highlighting areas surpassing a given
    `threshold`.

    Parameters
    ----------
    dfs : list of pd.DataFrame
        The input DataFrames, each containing columns
        for spatial coordinates and the `target_col`.
        A variable number of DataFrames can be provided.

    target_col : str
        The name of the numeric column representing
        the metric to visualize (e.g., subsidence
        or risk score).

    spatial_cols : tuple of str, default=('longitude',
        'latitude')
        A tuple specifying the longitude and latitude
        column names.

    labels : list of str, optional
        Labels corresponding to each DataFrame. If fewer
        labels than DataFrames are provided, the missing
        ones are auto-generated.

    threshold : str or float, optional
        A threshold value for highlighting points
        exceeding a critical limit. If ``'auto'``,
        an internal percentile-based approach is used.
        Set to None to disable highlighting.

    cmap : str, default='viridis'
        The colormap used to color the scatter points
        according to `target_col`.

    size : float or str, default=10
        The marker size. If a numeric value is given,
        all markers have the same size. If a column
        name is given (e.g., ``'size_col'``), marker
        sizes vary by that column's values.

    figsize : tuple of float, default=(12, 10)
        Dimensions of the figure in inches.

    title : str, default='Hotspot Map'
        The main title displayed above the map.

    thresh_label : str, optional
        The legend label used for threshold
        highlight markers.

    cbar_label : str, optional
        The label for the color bar if a basic
        matplotlib scatter is used.

    basemap : bool, default=True
        If True, attempts to overlay a tiled map
        in the background. Requires `contextily`
        if using geopandas.

    use_gpd : bool or str, default="auto"
        Controls whether geopandas-based plotting
        is used (True), or basic matplotlib (False).
        If ``"auto"``, geopandas is used if
        installed, otherwise a fallback occurs.

    crs : str, default="EPSG:4326"
        The initial coordinate reference system of
        the input data (if geopandas is used).

    epsg : int, default=3857
        The projected EPSG code for web mapping (e.g.,
        for contextily backgrounds). Used when
        `basemap` is True.

    axis_off : bool, default=False
        If True, disables axis lines and labels.

    savefig : str, optional
        A path for saving the figure to disk. If None,
        the figure is not saved.

    show_grid : bool, default=True
        If True, shows grid lines on the axis.

    grid_props : dict, optional
        Additional style properties for grid lines,
        passed to an internal method handling axis
        styling.

    s : int, default=50
        The size of markers used for highlighting
        threshold exceedances.

    verbose : int, default=0
        Verbosity level (0-5). Higher values produce
        more console messages.

    **kwargs
        Additional keyword arguments passed to the
        internal plotting function (e.g., scatter
        properties).

    Returns
    -------
    matplotlib.figure.Figure
        The created matplotlib Figure containing the
        plotted map or figure.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.plot.spatial import plot_hotspot_map
    >>> # Create synthetic DataFrame
    >>> df = pd.DataFrame({
    ...     'longitude': [114.05, 114.06, 114.07],
    ...     'latitude': [22.55, 22.56, 22.57],
    ...     'subsidence': [2.5, 3.7, 4.2]
    ... })
    >>> # Basic usage without geopandas
    >>> fig = plot_hotspot_map(
    ...     df,
    ...     target_col='subsidence',
    ...     threshold=3.0,
    ...     title='Subsidence Hotspots'
    ... )

    Notes
    -----
    If geopandas and contextily are installed (and
    `use_gpd` is not False), data points are converted
    into a GeoDataFrame, reprojected to EPSG:3857,
    and overlaid with a web basemap. Otherwise, a
    basic matplotlib scatter is used.
    
    .. math::
        \text{Hotspot} =
        \{\,(x, y)\,\mid\,
        \text{value}(x,y) >
        \text{threshold}\,\}

    where :math:`(x, y)` are the coordinates in
    `spatial_cols`. If `threshold` is set to ``'auto'``,
    the function attempts to infer a threshold via a
    percentile-based approach.


    See Also
    --------
    plot_hotspot_map : Method for creating geographic
        scatter plots with threshold highlighting.
    contextily.add_basemap : Renders a web-based tile
        map behind geospatial data.

    References
    ----------
    .. [1] J. D. Hunter. *Matplotlib: A 2D Graphics
       Environment.* Computing in Science & Engineering,
       9(3), 90-95, 2007.
    """

    # --- Optional geospatial dependencies.
    HAS_GEOPANDAS   = False
    HAS_CONTEXTILY  = False

    try:
        import geopandas as gpd
        from shapely.geometry import Point
        HAS_GEOPANDAS = True
    except ImportError:
        pass

    try:
        import contextily as ctx
        HAS_CONTEXTILY = True
    except ImportError:
        pass

    # If user explicitly sets use_gpd=False,
    # even if geopandas is installed, skip it.
    if not use_gpd:
        HAS_GEOPANDAS = False

    # Validate DataFrames
    dfs = are_all_frames_valid(
        *dfs,
        ops='validate',
        df_only=True
    )
    if not dfs:
        raise ValueError(
            "At least one DataFrame "
            "must be provided"
        )

    # Check all DataFrames have required columns
    [check_spatial_columns(
        df,
        spatial_cols=spatial_cols
    ) for df in dfs]

    lon_col, lat_col = spatial_cols
    required_cols = [
        lon_col,
        lat_col,
        target_col
    ]

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

    # Prepare labels
    labels = (
        labels or
        [f'Area {i+1}' for i in range(len(dfs))]
    )
    labels = (
        labels[:len(dfs)] +
        [f'Area {i+1}' for i in range(
            len(labels),
            len(dfs)
        )]
    )

    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Loop over data sets
    for df, label in zip(dfs, labels):
        # If geopandas is available
        if HAS_GEOPANDAS:
            geometry = [
                Point(xy) for xy
                in zip(df[lon_col], df[lat_col])
            ]
            gdf = gpd.GeoDataFrame(
                df,
                geometry=geometry,
                crs=crs
            )
            if basemap:
                gdf = gdf.to_crs(epsg=epsg)
        else:
            if verbose > 0:
                status ='disabled' if not use_gpd else 'not installed'
                print(
                    f"geopandas {status} - "
                    "using basic matplotlib plot"
                )

        # Plotting args
        plot_args = {
            'cmap': cmap,
            'markersize': (
                size if isinstance(
                    size, (int, float)
                ) else df[size]
            ),
            'alpha': 0.7,
            'edgecolor': 'w',
            'linewidth': 0.3,
            **kwargs
        }

        # If using geopandas + basemap
        if HAS_GEOPANDAS and basemap:
            # geo plotting
            plot = gdf.plot( # Noqa: E504
                column=target_col,
                ax=ax,
                legend=True,
                **plot_args
            )
            if basemap and HAS_CONTEXTILY:
                ctx.add_basemap(
                    ax,
                    source=ctx.providers
                        .OpenStreetMap.Mapnik
                )
        else:
            # fallback scatter
            plot_args = filter_valid_kwargs(
                ax.scatter,
                plot_args
            )
            sc = ax.scatter(
                df[lon_col],
                df[lat_col],
                c=df[target_col],
                **plot_args
            )
        if not HAS_GEOPANDAS:
            plt.colorbar(
                sc,
                label=(
                    cbar_label or
                    f'{target_col.capitalize()}'
                )
            )

        # Threshold highlighting
        if threshold is not None:
            if threshold == "auto":
                from gofast.utils.mathext import (
                    get_threshold_from
                )
                threshold = get_threshold_from(
                    df[target_col],
                    method='percentile'
                )
            hotspots = df[
                df[target_col] > threshold
            ]
            ax.scatter(
                hotspots[lon_col],
                hotspots[lat_col],
                color='red',
                marker='x',
                s=s,
                label=(
                    thresh_label or
                    f'Critical '
                    f'{target_col.capitalize()}'
                )
            )

    # Format axes
    ax.set_title(title)
    ax.set_xlabel(f'{lon_col.capitalize()}')
    ax.set_ylabel(f'{lat_col.capitalize()}')

    set_axis_grid(
        ax,
        show_grid=show_grid,
        grid_props=grid_props
    )

    if basemap and (
        not HAS_CONTEXTILY
    ) and verbose > 0:
        print(
            "contextily not installed "
            "- basemap disabled"
        )

    # Add legend if threshold
    if threshold is not None:
        ax.legend()

    if axis_off:
        ax.set_axis('off')

    if savefig:
        plt.savefig(
            savefig,
            bbox_inches='tight',
            dpi=300
        )
        if verbose > 0:
            print(
                f"Map saved to {savefig}"
            )

    return fig
