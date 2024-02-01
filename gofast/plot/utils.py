# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Additional plot utilities. 
"""
from __future__ import annotations 
import os
import re 
import copy 
import datetime 
import warnings
import itertools 
import scipy.stats
import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib as mpl
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.transforms as transforms 
from matplotlib.collections import EllipseCollection
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from scipy.cluster.hierarchy import dendrogram, ward 

from sklearn.cluster import KMeans 
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix , silhouette_samples, roc_curve 
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import learning_curve, KFold 
from sklearn.utils import resample

from .._typing import Optional, Tuple, Any, List, Union 
from .._typing import Dict, ArrayLike, DataFrame
from ..exceptions import  TipError, PlotError 
from ..tools.funcutils import _assert_all_types, is_iterable, str2columns 
from ..tools.funcutils import make_obj_consistent_if, is_in_if, to_numeric_dtypes 
from ..tools.funcutils import fill_nan_in
from ..tools.validator import _check_array_in , _is_cross_validated
from ..tools.validator import  assert_xy_in, get_estimator_name, check_is_fitted
from ..tools.validator import check_array, check_X_y, check_consistent_length 
from ..tools._dependency import import_optional_dependency 
from ._d_cms import D_COLORS, D_MARKERS, D_STYLES

def plot_regression_diagnostics(
    x: ArrayLike,
    ys: List[ArrayLike],
    titles: List[str],
    xlabel: str = 'X',
    ylabel: str = 'Y',
    figsize: Tuple[int, int] = (15, 5),
    ci: Optional[int] = 95, 
    **reg_kws
) -> plt.Figure:
    """
    Creates a series of plots to diagnose linear regression fits.

    Parameters
    ----------
    x : np.ndarray
        The independent variable data.
    ys : List[np.ndarray]
        A list of dependent variable datasets to be plotted against x.
    titles : List[str]
        Titles for each subplot.
    xlabel : str, default='X'
        Label for the x-axis.
    ylabel : str, default='Y'
        Label for the y-axis.
    figsize : Tuple[int, int], default=(15, 5)
        Size of the entire figure.
    ci : Optional[int], default=95
        Size of the confidence interval for the regression estimate.
    reg_kws: dict, 
        Additional parameters passed to `seaborn.regplot`. 
        
    Returns
    -------
    fig : plt.Figure
        The matplotlib figure object with the plots.

    Example
    -------
    >>> import numpy as np 
    >>> from gofast.plot.utils import plot_regression_diagnostics
    >>> x = np.linspace(160, 170, 100)
    >>> # Homoscedastic noise
    >>> y1 = 50 + 0.6 * x + np.random.normal(size=x.size)  
    >>> # Heteroscedastic noise
    >>> y2 = y1 * np.exp(np.random.normal(scale=0.05, size=x.size))  
    >>> # Larger noise variance
    >>> y3 = y1 + np.random.normal(scale=0.5, size=x.size)  
    >>> titles = ['All assumptions satisfied', 'Nonlinear term in model',
                  'Heteroscedastic noise']
    >>> fig = plot_regression_diagnostics(x, [y1, y2, y3], titles)
    >>> plt.show()
    """
    fig, axes = plt.subplots(1, len(ys), figsize=figsize, sharey=True)
    
    for i, ax in enumerate(axes):
        sns.regplot(x=x, y=ys[i], ax=ax, ci=ci, **reg_kws)
        ax.set_title(titles[i])
        ax.set_xlabel(xlabel)
        if i == 0:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel('')

    plt.tight_layout()
    return fig

def plot_residuals_vs_leverage(
    residuals: ArrayLike,
    leverage: ArrayLike,
    cook_d: Optional[ArrayLike] = None,
    figsize: Tuple[int, int] = (8, 6),
    cook_d_threshold: float = 0.5,
    annotate: bool = True,
    scatter_kwargs: Optional[dict] = None,
    cook_d_kwargs: Optional[dict] = None,
    line_kwargs: Optional[dict] = None,
    annotation_kwargs: Optional[dict] = None
) -> plt.Axes:
    """
    Plots standardized residuals against leverage with Cook's 
    distance contours.

    Parameters
    ----------
    residuals : np.ndarray
        Standardized residuals from the regression model.
    leverage : np.ndarray
        Leverage values calculated from the model.
    cook_d : np.ndarray, optional
        Cook's distance for each observation in the model.
    figsize : Tuple[int, int], default=(8, 6)
        The figure size for the plot.
    cook_d_threshold : float, default=0.5
        The threshold for Cook's distance to draw a contour and potentially 
        annotate points.
    annotate : bool, default=True
        If True, annotate points that exceed the Cook's distance threshold.
    scatter_kwargs : dict, optional
        Additional keyword arguments for the scatter plot.
    cook_d_kwargs : dict, optional
        Additional keyword arguments for the Cook's distance ellipses.
    line_kwargs : dict, optional
        Additional keyword arguments for the horizontal line at 0.
    annotation_kwargs : dict, optional
        Additional keyword arguments for the annotations.

    Returns
    -------
    ax : plt.Axes
        The matplotlib axes containing the plot.

    Example
    -------
    >>> import numpy as np 
    >>> from gofast.plot.utils import plot_residuals_vs_leverage
    >>> residuals = np.random.normal(0, 1, 100)
    >>> leverage = np.random.uniform(0, 0.2, 100)
    >>> # Randomly generated for example purposes
    >>> cook_d = np.random.uniform(0, 1, 100) ** 2  
    >>> ax = plot_residuals_vs_leverage(residuals, leverage, cook_d)
    >>> plt.show()
    """
    if scatter_kwargs is None:
        scatter_kwargs = {'edgecolors': 'k', 'facecolors': 'none'}
    if cook_d_kwargs is None:
        cook_d_kwargs = {'cmap': 'RdYlBu_r'}
    if line_kwargs is None:
        line_kwargs = {'linestyle': '--', 'color': 'grey', 'linewidth': 1}
    if annotation_kwargs is None:
        annotation_kwargs = {'textcoords': 'offset points', 
                             'xytext': (5,5), 'ha': 'right'}

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(leverage, residuals, **scatter_kwargs)
    
    # Add Cook's distance contour if provided
    if cook_d is not None:
        levels = [cook_d_threshold, np.max(cook_d)]
        cook_d_collection = EllipseCollection(
            widths=2 * leverage, heights=2 * np.sqrt(cook_d), 
            angles=0, units='x',
            offsets=np.column_stack([leverage, residuals]),
            transOffset=ax.transData,
            **cook_d_kwargs
        )
        cook_d_collection.set_array(cook_d)
        cook_d_collection.set_clim(*levels)
        ax.add_collection(cook_d_collection)
        fig.colorbar(cook_d_collection, ax=ax, orientation='vertical')

    # Draw the horizontal line at 0 for residuals
    ax.axhline(y=0, **line_kwargs)
    
    # Annotate points with Cook's distance above threshold
    if annotate and cook_d is not None:
        for i, (xi, yi, ci) in enumerate(zip(leverage, residuals, cook_d)):
            if ci > cook_d_threshold:
                ax.annotate(f"{i}", (xi, yi), **annotation_kwargs)

    ax.set_title('Residuals vs Leverage')
    ax.set_xlabel('Leverage')
    ax.set_ylabel('Standardized Residuals')

    plt.show()
    return ax

def plot_residuals_vs_fitted(
    fitted_values: ArrayLike, 
    residuals: ArrayLike, 
    highlight: Optional[Tuple[int, ...]] = None, 
    figsize: Tuple[int, int] = (6, 4),
    title: str = 'Residuals vs Fitted',
    xlabel: str = 'Fitted values',
    ylabel: str = 'Residuals',
    linecolor: str = 'red',
    linestyle: str = '-',
    scatter_kws: Optional[dict] = None,
    line_kws: Optional[dict] = None
) -> plt.Axes:
    """
    Creates a residuals vs fitted values plot.
    
    Function is commonly used in regression analysis to assess the fit of a 
    model. It shows if the residuals have non-linear patterns that could 
    suggest non-linearity in the data or problems with the model.

    Parameters
    ----------
    fitted_values : np.ndarray
        The fitted values from a regression model.
    residuals : np.ndarray
        The residuals from a regression model.
    highlight : Tuple[int, ...], optional
        Indices of points to highlight in the plot.
    figsize : Tuple[int, int], default=(6, 4)
        Size of the figure to be created.
    title : str, default='Residuals vs Fitted'
        Title of the plot.
    xlabel : str, default='Fitted values'
        Label of the x-axis.
    ylabel : str, default='Residuals'
        Label of the y-axis.
    linecolor : str, default='red'
        Color of the line to be plotted.
    linestyle : str, default='-'
        Style of the line to be plotted.
    scatter_kws : dict, optional
        Additional keyword arguments to be passed to the `plt.scatter` method.
    line_kws : dict, optional
        Additional keyword arguments to be passed to the `plt.plot` method.

    Returns
    -------
    ax : plt.Axes
        The matplotlib axes containing the plot.

    See Also 
    ---------
    gofast.tools.mathex.calculate_residuals: 
        Calculate the residuals for regression, binary, or multiclass 
        classification tasks.
        
    Example
    -------
    >>> import numpy as np 
    >>> from gofast.plot.utils import plot_residuals_vs_fitted
    >>> fitted = np.linspace(0, 100, 100)
    >>> residuals = np.random.normal(0, 10, 100)
    >>> ax = plot_residuals_vs_fitted(fitted, residuals)
    >>> plt.show()
    """
    if scatter_kws is None:
        scatter_kws = {}
    if line_kws is None:
        line_kws = {}

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(fitted_values, residuals, **scatter_kws)
    
    if highlight:
        ax.scatter(fitted_values[list(highlight)], 
                   residuals[list(highlight)], color='orange')

    sns.regplot(x=fitted_values, y=residuals, lowess=True, ax=ax,
                line_kws={'color': linecolor, 'linestyle': linestyle, **line_kws})

    # Annotate highlighted points
    if highlight:
        for i in highlight:
            ax.annotate(i, (fitted_values[i], residuals[i]))

    ax.axhline(0, color='grey', lw=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return ax

def plot_feature_interactions(
    data: DataFrame, /, 
    features: Optional[List[str]] = None, 
    histogram_bins: int = 15, 
    scatter_alpha: float = 0.7,
    corr_round: int = 2,
    plot_color: str = 'skyblue',
    edge_color: str = 'black',
    savefig: Optional[str] = None
) -> plt.Figure:
    """
    Visualizes the interactions (distributions and relationships) among 
    various features in a dataset. 
    
    The visualization includes histograms for distribution of features, 
    scatter plots for pairwise relationships, and Pearson correlation 
    coefficients.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing the dataset.
    features : Optional[List[str]]
        A list of feature names to be visualized. If None, all features are used.
    histogram_bins : int, optional
        The number of bins for the histograms. Default is 15.
    scatter_alpha : float, optional
        Alpha blending value for scatter plot, between 0 (transparent) and 
        1 (opaque). Default is 0.7.
    corr_round : int, optional
        The number of decimal places for rounding the correlation coefficient.
        Default is 2.
    plot_color : str, optional
        The color for the plots. Default is 'skyblue'.
    edge_color : str, optional
        The edge color for the histogram bins. Default is 'black'.
    savefig : Optional[str], optional
        The file path to save the figure. If None, the figure is not saved.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.

    Example
    -------
    >>> import numpy as np 
    >>> import pandas as pd 
    >>> from gofast.plot.utils import plot_feature_interactions
    >>> df = pd.DataFrame({
        'Feature1': np.random.randn(100),
        'Feature2': np.random.rand(100),
        'Feature3': np.random.gamma(2., 2., 100)
    })
    >>> fig = plot_feature_interactions(df, histogram_bins=20, scatter_alpha=0.5)
    
    This will create a customized plot with histograms, scatter plots, 
    and correlation coefficients for all features in the DataFrame.
    """
    data = to_numeric_dtypes(data, pop_cat_features=True )
    if features is None:
        features= list( data.columns )
        
    # fill Nan, if exist in data
    data = fill_nan_in(data )
    num_features = len(features)
    
    fig, axs = plt.subplots(num_features, num_features, figsize=(15, 15))

    for i in range(num_features):
        for j in range(num_features):
            if i == j:  # Diagonal - Histogram
                axs[i, j].hist(data[features[i]], bins=histogram_bins,
                               color=plot_color, edgecolor=edge_color)
                axs[i, j].set_title(f'Distribution of {features[i]}')
            elif i < j:  # Upper Triangle - Scatter plot
                axs[i, j].scatter(data[features[j]], data[features[i]],
                                  alpha=scatter_alpha)
                axs[i, j].set_xlabel(features[j])
                axs[i, j].set_ylabel(features[i])
                # Calculate and display Pearson correlation
                corr, _ = scipy.stats.pearsonr(data[features[i]], data[features[j]])
                axs[i, j].annotate(f'ρ = {corr:.{corr_round}f}', xy=(0.1, 0.9),
                                   xycoords='axes fraction')
            else:  # Lower Triangle - Empty
                axs[i, j].axis('off')

    plt.tight_layout()

    if savefig:
        plt.savefig(savefig, format='png', bbox_inches='tight')

    return fig

def create_matrix_representation(
        data: Dict[str, set], 
        cmap: str = 'Blues', 
        savefig: Optional[str] = None
        ) -> plt.Axes:
    """
    Creates and optionally saves a matrix-based representation to visualize 
    intersections among multiple sets.
    
    Rows represent elements and columns represent sets. Cells in the matrix 
    are filled to indicate membership.

    Parameters
    ----------
    data : Dict[str, set]
        A dictionary where each key is the name of a set and each value is 
        the set itself.
    cmap : str, optional
        The colormap used for the matrix plot. Defaults to 'Blues'.
    savefig : Optional[str], optional
        The file path and name to save the figure. If None, the figure is 
        not saved.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object of the plot.

    Example
    -------
    >>> from gofast.plot.utils import create_matrix_representation
    >>> sets = {
        "Set1": {1, 2, 3},
        "Set2": {2, 3, 4},
        "Set3": {3, 4, 5}
    }
    >>> ax = create_matrix_representation(sets, cmap='Greens', savefig='set_matrix.png')

    This will create a matrix plot for three sets, showing which elements 
    belong to which sets and optionally save it to 'set_matrix.png'.
    """
    # Create a DataFrame to represent the sets
    all_elements = sorted(set.union(*data.values()))
    matrix_data = pd.DataFrame(index=all_elements, columns=data.keys(), data=0)

    # Fill the DataFrame
    for set_name, elements in data.items():
        matrix_data.loc[elements, set_name] = 1

    # Create and display the matrix plot
    fig, ax = plt.subplots(figsize=(len(data), len(all_elements)/2))
    cax = ax.imshow(matrix_data, aspect='auto', cmap=cmap, interpolation='nearest')
    fig.colorbar(cax, ax=ax, label='Set Membership')
    ax.set_xticks(np.arange(len(data)))
    ax.set_xticklabels(data.keys())
    ax.set_yticks(np.arange(len(all_elements)))
    ax.set_yticklabels(all_elements)
    ax.set_title('Matrix-Based Representation of Set Intersections')
    ax.set_xlabel('Sets')
    ax.set_ylabel('Elements')
    ax.grid(False)

    # Save the figure if a path is provided
    if savefig:
        plt.savefig(savefig, format='png', bbox_inches='tight')

    return ax

def plot_venn_diagram(
    sets: Union[Dict[str, set], List[ArrayLike]],
    set_labels: Optional[Tuple[str, ...]] = None,
    title: str = 'Venn Diagram',
    figsize: Tuple[int, int] = (8, 8),
    set_colors: Tuple[str, ...] = ('red', 'green', 'blue'),
    alpha: float = 0.5, 
    savefig: Optional[str]=None, 
) -> Axes:
    """
    Create and optionally save a Venn diagram for two sets.

    A Venn diagram is a visual representation of the mathematical or logical
    relationship between two sets of items. It depicts these sets as circles,
    with the overlap representing items common to both sets. It's a useful
    tool for comparing and contrasting groups of elements, especially in
    the field of data analysis and machine learning.
    
    This function supports up to 3 sets. Venn diagrams for more than 3 sets
    are not currently supported due to visualization complexity.

    Parameters
    ----------
    sets : Union[Dict[str, set], List[np.ndarray]]
        Either a dictionary with two items, each being a set of elements, or a 
        list of two arrays. The arrays should be of consistent length, 
        and their elements will be treated as unique identifiers.
    set_labels : Optional[Tuple[str, str]], optional
        A tuple containing two strings that label the sets in the diagram. If None, 
        default labels will be used.
    title : str, optional
        Title of the Venn diagram. Default is 'Venn Diagram'.
    figsize : Tuple[int, int], optional
        Size of the figure (width, height). Default is (8, 8).
    set_colors : Tuple[str, str], optional
        Colors for the two sets. Default is ('red', 'green').
    alpha : float, optional
        Transparency level of the set colors. Default is 0.5.
    savefig : Optional[str], optional
        Path and filename to save the figure. If None, the figure is not saved.
        Example: 'path/to/figure.png'

    Returns
    -------
    ax : Axes
        The axes object of the plot.

    Examples
    --------
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from xgboost import XGBClassifier
    >>> from gofast.plot.utils import plot_venn_diagram
    
    >>> # Example of comparing feature contributions of RandomForest and XGBoost
    >>> X, y = make_classification(n_samples=1000, n_features=20, 
                               n_informative=2, n_redundant=0, 
                               random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.3, 
                                                        random_state=42)

    >>> # Train classifiers
    >>> rf = RandomForestClassifier(random_state=42).fit(X_train, y_train)
    >>> xgb = XGBClassifier(random_state=42).fit(X_train, y_train)

    >>> # Get feature importances
    >>> rf_features = set(np.argsort(rf.feature_importances_)[-5:])  # Top 5 features
    >>> xgb_features = set(np.argsort(xgb.feature_importances_)[-5:])

    >>> # Plot Venn diagram
    >>> ax = plot_venn_diagram(
        sets={'RandomForest': rf_features, 'XGBoost': xgb_features},
        set_labels=('RandomForest Features', 'XGBoost Features'),
        title='Feature Contribution Comparison'
    )
    >>> plt.show()
    
    Using dictionaries
    >>> set1 = {1, 2, 3}
    >>> set2 = {2, 3, 4}
    >>> set3 = {4, 5, 6}
    >>> ax = plot_venn_diagram({'Set1': set1, 'Set2': set2, 'Set3': set3}, 
                           ('Set1', 'Set2', 'Set3'))

    Using arrays
    >>> arr1 = np.array([1, 2, 3])
    >>> arr2 = np.array([2, 3, 4])
    >>> ax = plot_venn_diagram([arr1, arr2], ('Array1', 'Array2'))

    Notes
    -----
    This function can be particularly useful in scenarios like feature analysis
    in machine learning models, where understanding the overlap and uniqueness
    of feature contributions from different models can provide insights into
    model behavior and performance.
    """
    import_optional_dependency("matplotlib_venn")
    from matplotlib_venn import venn2, venn2_circles, venn3, venn3_circles
    num_sets = len(sets)

    # Validate input
    if num_sets not in [2, 3]:
        raise ValueError("Only 2 or 3 sets are supported.")

    if set_labels is not None:
        if len(set_labels) != num_sets:
            raise ValueError("Length of 'set_labels' must match the number of sets.")
    else:
        # Default labels if not provided
        set_labels = tuple(f'Set {i+1}' for i in range(num_sets))

    # Prepare set values for Venn diagram
    if isinstance(sets, dict):
        set_values = tuple(sets.values())
    elif isinstance(sets, list):
        set_values = tuple(set(arr) for arr in sets)

    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)

    # Choose the appropriate Venn function
    if num_sets == 2:
        venn2(subsets=set_values, set_labels=set_labels, 
                     set_colors=set_colors[:2], alpha=alpha)
        venn2_circles(subsets=set_values)
    elif num_sets == 3:
        venn3(subsets=set_values, set_labels=set_labels,
                     set_colors=set_colors[:3], alpha=alpha)
        venn3_circles(subsets=set_values)

    if savefig:
        plt.savefig(savefig, format='png', bbox_inches='tight')

    return ax


def create_upset_plot(
        data: Dict[str, set], sort_by: str = 'degree', 
        show_counts: bool = False):
    """
    Creates an UpSet plot, which is an alternative to Venn diagrams for more 
    than three sets.
    
    The UpSet plot focuses on the intersections of the sets rather than on the 
    sets themselves.

    Parameters
    ----------
    data : Dict[str, set]
        A dictionary where each key is the name of a set and each value is 
        the set itself.
    sort_by : str, optional
        The criteria to sort the bars. Options are 'degree' and 'cardinality'.
        'degree' will sort by the number of sets in the intersection, 
        'cardinality' by the size of the intersection.
    show_counts : bool, optional
        Whether to show the counts at the top of the bars in the UpSet plot.

    Returns
    -------
    None. The function displays the UpSet plot.

    Example
    -------
    >>> sets = {
        "Set1": {1, 2, 3},
        "Set2": {3, 4, 5},
        "Set3": {5, 6, 7},
        "Set4": {7, 8, 9}
    }
    >>> create_upset_plot(sets, sort_by='degree', show_counts=True)

    This will create an UpSet plot for four sets, showing their intersections 
    and the counts of elements in each intersection.
    """
    import_optional_dependency("upsetplot")
    from upsetplot import plot, from_contents 
    # Convert the data into the format required by UpSetPlot
    upset_data = from_contents(data)

    # Create and display the UpSet plot
    plot(upset_data, sort_by=sort_by, show_counts=show_counts)

def plot_euler_diagram(sets: dict):
    """
    Creates an Area-Proportional Euler Diagram using R's venneuler package.

    Euler diagrams are a generalization of Venn diagrams that represent 
    relationships between different sets or groupings. Unlike Venn diagrams, 
    they do not require all possible logical relationships to be shown. This 
    function interfaces with R to create and display an Euler diagram for a 
    given set of data.

    Parameters
    ----------
    sets : dict
        A dictionary where keys are set names and values are the sets.

    Returns
    -------
    None. The function displays the Euler diagram.

    Example
    -------
    >>> sets = {
        "Set1": {1, 2, 3},
        "Set2": {3, 4, 5},
        "Set3": {5, 6, 7}
    }
    >>> plot_euler_diagram(sets)

    This will create an Euler diagram for three sets, showing their overlaps.
    Set1 contains elements 1, 2, and 3; Set2 contains 3, 4, and 5; and Set3 
    contains 5, 6, and 7. The diagram will visually represent the intersections 
    between these sets.

    Notes
    -----
    This function requires a working R environment with the 'venneuler' package 
    installed, as well as 'rpy2' installed in the Python environment. It creates 
    a temporary file to save the plot generated by R, reads this file into Python 
    for display, and then removes the file.
    """
    import_optional_dependency("rpy2")
    
    import rpy2.robjects as robjects
    # from rpy2.robjects.packages import importr
    from rpy2.robjects import pandas2ri
    import matplotlib.image as mpimg
    import tempfile
    
    # Activate automatic conversion between pandas dataframes and R dataframes
    pandas2ri.activate()

    # Define an R script
    r_script = """
    library(venneuler)
    plot_venneuler <- function(sets) {
        v <- venneuler(sets)
        plot(v)
    }
    """

    # Convert the Python dictionary to an R-readable string
    sets_r = 'list(' + ', '.join(f"{k}=c({', '.join(map(str, v))})" 
                                 for k, v in sets.items()) + ')'

    # Run the R script
    robjects.r(r_script)
    plot_venneuler = robjects.globalenv['plot_venneuler']

    # Temporary file to save the plot
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        tmpfile_name = tmpfile.name

    # Execute the R function and save the plot
    robjects.r(f"png('{tmpfile_name}')")
    plot_venneuler(robjects.r(sets_r))
    robjects.r("dev.off()")

    # Display the saved plot in Python
    img = mpimg.imread(tmpfile_name)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    # Remove the temporary file
    os.remove(tmpfile_name)

def plot_sankey(
    data: DataFrame , 
    source_col: str, 
    target_col: str, 
    value_col: str, 
    label_col: Optional[str] = None,
    figsize: Tuple[int, int] = (800, 600), 
    title: Optional[str] = None
 ):
    """
    Creates a Sankey diagram from a pandas DataFrame using Plotly.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the source, target, and value columns 
        for the Sankey diagram.
    source_col : str
        The name of the column in 'data' that contains the source nodes.
    target_col : str
        The name of the column in 'data' that contains the target nodes.
    value_col : str
        The name of the column in 'data' that contains the flow values
        between the nodes.
    label_col : Optional[str], optional
        The name of the column in 'data' that contains the labels of the nodes.
        If None, the nodes will be labeled with unique identifiers.
    figsize : Tuple[int, int], optional
        Figure dimension (width, height) in pixels. Default is (800, 600).
    title : Optional[str], optional
        The title of the plot. If None, no title is set.

    Returns
    -------
    go.Figure
        The Plotly Figure object with the Sankey diagram for further 
        tweaking and rendering.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.plot.utils import plot_sankey 
    >>> df = pd.DataFrame({
    ...     'source': ['A', 'A', 'B', 'B', 'C', 'C'],
    ...     'target': ['C', 'D', 'C', 'D', 'E', 'F'],
    ...     'value': [8, 2, 4, 8, 4, 2],
    ...     'label': ['Node A', 'Node B', 'Node C', 'Node D', 'Node E', 'Node F']
    ... })
    >>> fig = plot_sankey(df, 'source', 'target', 'value', 'label', 
                          title='My Sankey Diagram')
    >>> fig.show()  

    Notes
    -----
    Sankey diagrams are helpful for visualizing flow data between different 
    nodes or stages. The width of the arrows or links is proportional to the 
    flow quantity, allowing for easy comparison of volume or value transfers.
    
    See Also 
    ---------
    gofast.tools.mathex.infer_sankey_columns: 
        Infers source, target, and value columns for a Sankey diagram 
        from a DataFrame.
    """
    import_optional_dependency("plotly")
    import plotly.graph_objects as go

    # Prepare the data for the Sankey diagram
    label_list =  data[label_col].unique().tolist() if label_col else\
        pd.concat([data[source_col], data[target_col]]).unique().tolist()
                  
    source_indices = data[source_col].apply(label_list.index).tolist()
    target_indices = data[target_col].apply(label_list.index).tolist()

    

    # Create the Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color='black', width=0.5),
            label=label_list,
        ),
        link=dict(
            source=source_indices,  # indices correspond to labels
            target=target_indices,
            value=data[value_col].tolist()
        ))])
    
    # Update layout
    fig.update_layout(title_text=title, width=figsize[0], height=figsize[1])
    
    return fig

def plot_sunburst(
    d: List[Dict[str, str]], /, 
    figsize: Tuple[int, int] = (10, 8), 
    color: Optional[List[str]] = None,
    title: Optional[str] = None, 
    savefig: Optional[str] = None
) :
    """
    Plots a sunburst chart with the given data and parameters using Plotly.

    Parameters
    ----------
    d : List[Dict[str, str]]
        The input data for the sunburst chart, where each dict contains 'name',
        'value', and 'parent' keys.
    figsize : Tuple[int, int], optional
        Figure dimension (width, height) in pixels. Default is (1000, 800).
    color : List[str], optional
        A list of color codes for the segments of the sunburst chart. If None,
        default colors are used.
    title : str, optional
        The title of the plot. If None, no title is set.
    savefig : str, optional
        Path and filename where the figure should be saved. If ends with 'html',
        saves an interactive HTML file. Otherwise, saves a static image of the
        plot.

    Returns
    -------
    go.Figure
        The Plotly Figure object with the plot for further tweaking and rendering.

    See Also
    ---------
    gofast.tools.mathex.compute_sunburst_data: 
        Computes the data structure required for generating a sunburst chart 
        from a DataFrame.
        
    Examples
    --------
    >>> from gofast.plot.utils import plot_sunburst
    >>> d = [
    ...     {"name": "Category A", "value": 10, "parent": ""},
    ...     {"name": "Category B", "value": 20, "parent": ""},
    ...     {"name": "Subcategory A1", "value": 5, "parent": "Category A"},
    ...     {"name": "Subcategory A2", "value": 5, "parent": "Category A"}
    ... ]
    >>> plot_sunburst(d, figsize=(8, 8), title='Sunburst Chart Example')

    Notes
    -----
    Sunburst charts are used to visualize hierarchical data spanning outwards
    radially from root to leaves. The parent-child relation is represented by 
    the enclosure in this chart type.

    """
    import_optional_dependency("plotly")
    import plotly.graph_objects as go
    # Convert figsize from inches to pixels
    width_px, height_px = figsize[0] * 100, figsize[1] * 100

    fig = go.Figure(go.Sunburst(
        labels=[item['name'] for item in d],
        parents=[item['parent'] for item in d],
        values=[item['value'] for item in d],
        marker=dict(colors=color) if color else None
    ))

    fig.update_layout(margin=dict(t=0, l=0, r=0, b=0),
                      width=width_px, height=height_px)

    if title:
        fig.update_layout(title=title)

    if savefig:
        if savefig.endswith('.html'):
            fig.write_html(savefig, auto_open=True)
        else:
            fig.write_image(savefig)

    fig.show()

    return fig

def plot_custom_boxplot(
    data: ArrayLike | DataFrame, /, 
    labels: list[str],
    title: str, y_label: str, 
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
    >>> from gofast.plot.utils import plot_custom_boxplot
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

def plot_abc_curve(
    effort: List[float],
    yield_: List[float], 
    title: str = "ABC plot", 
    xlabel: str = "Effort", 
    ylabel: str = "Yield", 
    figsize: Tuple[int, int] = (8, 6),
    abc_line_color: str = "blue", 
    identity_line_color: str = "magenta", 
    uniform_line_color: str = "green",
    abc_linestyle: str = "-", 
    identity_linestyle: str = "--", 
    uniform_linestyle: str = ":",
    linewidth: int = 2,
    legend: bool = True,
    set_annotations: bool = True,
    set_a: Tuple[int, int] = (0, 2),
    set_b: Tuple[int, int] = (0, 0),
    set_c: Tuple[int, str] = (5, '+51 dropped'),
    savefig: Optional[str] = None
) -> None:
    """
    Plot an ABC curve comparing effort vs yield with reference lines for 
    identity and uniform distribution.
    
    Parameters
    ----------
    effort : List[float]
        The effort values (x-axis).
    yield_ : List[float]
        The yield values (y-axis).
    title : str, optional
        The title of the plot.
    xlabel : str, optional
        The x-axis label.
    ylabel : str, optional
        The y-axis label.
    figsize : Tuple[int, int], optional
        The figure size in inches.
    abc_line_color : str, optional
        The color of the ABC line.
    identity_line_color : str, optional
        The color of the identity line.
    uniform_line_color : str, optional
        The color of the uniform line.
    abc_linestyle : str, optional
        The linestyle for the ABC line.
    identity_linestyle : str, optional
        The linestyle for the identity line.
    uniform_linestyle : str, optional
        The linestyle for the uniform line.
    linewidth : int, optional
        The width of the lines.
    legend : bool, optional
        Whether to show the legend.
    set_annotations : bool, optional
        Whether to annotate set A, B, C.
    set_a : Tuple[int, int], optional
        The info for set A annotation.
    set_b : Tuple[int, int], optional
        The info for set B annotation.
    set_c : Tuple[int, str], optional
        The info for set C annotation.
    savefig : Optional[str], optional
        Path to save the figure. If None, the figure is not saved.
    
    Returns
    -------
    ax : Axes
        The matplotlib Axes object for the plot.
    
    See Also
    --------
    gofast.tools.mathex.compute_effort_yield: 
        Compute effort and yield values from importance data. 
        
    Example 
    -------
    >>> import numpy as np 
    >>> from gofast.plot.utils import plot_abc_curve
    >>> effort = np.linspace(0, 1, 100)
    >>> yield_ = effort ** 2  # This is just an example; your data will vary.
    >>> plot_abc_curve(effort, yield_)
    """

    fig, ax = plt.subplots(figsize=figsize)

    # Plot the curves
    ax.plot(effort, yield_, label='ABC', color=abc_line_color,
            linestyle=abc_linestyle, linewidth=linewidth)
    ax.plot([0, 1], [0, 1], label='Identity', color=identity_line_color,
            linestyle=identity_linestyle, linewidth=linewidth)
    ax.hlines(y=yield_[-1], xmin=0, xmax=1, label='Uniform',
              color=uniform_line_color, linestyle=uniform_linestyle,
              linewidth=linewidth)

    # Annotations for sets
    if set_annotations:
        ax.annotate(f'Set A: n = {set_a[1]}', xy=(0.05, 0.95),
                    xycoords='axes fraction', fontsize=9)
        ax.annotate(f'Set B: n = {set_b[1]}', xy=(0.05, 0.90),
                    xycoords='axes fraction', fontsize=9)
        ax.annotate(f'Set C: n = {set_c[0]} {set_c[1]}', xy=(0.05, 0.85), 
                    xycoords='axes fraction', fontsize=9)

    # Additional plot settings
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    if legend:
        ax.legend()

    if savefig:
        plt.savefig(savefig, bbox_inches='tight')

    plt.show()
    
    return ax 

def plot_permutation_importance(
    importances: np.ndarray, 
    feature_names: List[str], 
    title: str = "Permutation feature importance",
    xlabel: str = "RF importance",
    ylabel: str = "Features", 
    figsize: Tuple[int, int] = (10, 8), 
    color: str = "skyblue", 
    edgecolor: str = "black", 
    savefig: Optional[str] = None
) -> None:
    """
    Plot permutation feature importance as a horizontal bar chart.
    
    Parameters
    ----------
    importances : array-like
        The feature importances, typically obtained from a model 
        or permutation test.
    feature_names : list of str
        The names of the features corresponding to the importances.
    title : str, optional
        Title of the plot. Defaults to "Permutation feature importance".
    xlabel : str, optional
        Label for the x-axis. Defaults to "RF importance".
    ylabel : str, optional
        Label for the y-axis. Defaults to "Features".
    figsize : tuple, optional
        Size of the figure (width, height) in inches. Defaults to (10, 8).
    color : str, optional
        Bar color. Defaults to "skyblue".
    edgecolor : str, optional
        Bar edge color. Defaults to "black".
    savefig : str, optional
        Path to save the figure. If None, the figure is not saved. 
        Defaults to None.
    
    Returns
    -------
    fig : Figure
        The matplotlib Figure object for the plot.
    ax : Axes
        The matplotlib Axes object for the plot.
    
    Example
    -------
    >>> import numpy as np 
    >>> from gofast.plot.utils import plot_permutation_importance
    >>> importances = np.random.rand(30)
    >>> feature_names = ['Feature {}'.format(i) for i in range(30)]
    >>> plot_permutation_importance(
        importances, feature_names, title="My Plot", xlabel="Importance",
        ylabel="Features", figsize=(8, 10), color="lightblue",
        edgecolor="gray", savefig="importance_plot.png")
    """

    # Sort the feature importances in ascending order for plotting
    sorted_indices = np.argsort(importances)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(importances)), importances[sorted_indices],
            color=color, edgecolor=edgecolor)
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels(np.array(feature_names)[sorted_indices])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Optionally save the figure to a file
    if savefig is not None:
        plt.savefig(savefig, bbox_inches='tight')

    plt.show()
    return fig, ax

def create_radar_chart(
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
        The size of the figure to plot (width, height in inches). Default is (6, 6).
    color_map : str or list, optional
        Colormap or list of colors for the different clusters. Default is 'Set2'.
    alpha_fill : float, optional
        Alpha value for the filled area under the plot. Default is 0.25.
    linestyle : str, optional
        The style of the line in the plot. Default is 'solid'.
    linewidth : int, optional
        The width of the lines. Default is 1.
    yticks : tuple, optional
        Tuple containing the y-ticks values. Default is (0.5, 1, 1.5).
    ytick_labels : list of str, optional
        List of labels for the y-ticks, must match the length of yticks. 
        If None, yticks will be used as labels. Default is None.
    ylim : tuple, optional
        Tuple containing the min and max values for the y-axis. Default is (0, 2).
    legend_loc : str, optional
        The location of the legend. Default is 'upper right'.

    Returns
    -------
    fig : Figure
        The matplotlib Figure object for the radar chart.
    ax : Axes
        The matplotlib Axes object for the radar chart.

    Example
    -------
    >>> import numpy as np 
    >>> from gofast.plot.utils import create_radar_chart
    >>> num_clusters = 5
    >>> num_vars = 10
    >>> data = np.random.rand(num_clusters, num_vars)
    >>> categories = [f"Variable {i}" for i in range(num_vars)]
    >>> cluster_labels = [f"Cluster {i}" for i in range(num_clusters)]
    >>> create_radar_chart(data, categories, cluster_labels)
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

def create_base_radar_chart(
    d: ArrayLike,/,   categories: List[str], 
    cluster_labels: List[str],
    title:str="Radar plot Umatrix cluster properties"
    ):
    """
    Create a radar chart with one axis per variable.

    Parameters
    ----------
    data : array-like
        2D array with shape (n_clusters, n_variables), where each row 
        represents a different
        cluster and each column represents a different variable.
    categories : list of str
        List of variable names corresponding to the columns in the data.
    cluster_labels : list of str
        List of labels for the different clusters.
    title : str, optional
        The title of the radar chart. Default is "Radar plot
        Umatrix cluster properties".

    Returns
    -------
    fig : Figure
        The matplotlib Figure object for the radar chart.
    ax : Axes
        The matplotlib Axes object for the radar chart.
        
    Example
    -------
    >>> import numpy as np 
    >>> from gofast.plot.utils import create_base_radar_chart
    >>> num_clusters = 5
    >>> num_vars = 10
    >>> data = np.random.rand(num_clusters, num_vars)
    >>> categories = [f"Variable {i}" for i in range(num_vars)]
    >>> cluster_labels = [f"Cluster {i}" for i in range(num_clusters)]
    >>> create_base_radar_chart(data, categories, cluster_labels)
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

def plot_cluster_comparison(
    data,  
    cluster_col, 
    class_col, 
    figsize=(10, 6), 
    palette="RdYlGn", 
    title="Clusters versus Prior Classes"
    ):
    """
    Plots a comparison of clusters versus prior class distributions 
    using a heatmap.

    Parameters
    ----------
    data : DataFrame
        A pandas DataFrame containing at least two columns for clustering 
        and class comparison.
    cluster_col : str
        The name of the column in `data` that contains cluster labels.
    class_col : str
        The name of the column in `data` that contains prior class labels.
    figsize : tuple, optional
        The size of the figure to be created (width, height in inches). 
        Default is (10, 6).
    palette : str, optional
        The color palette to use for differentiating Pearson residuals in the 
        heatmap. Default is "RdYlGn".
    title : str, optional
        The title of the plot. Default is "Clusters versus Prior Classes".

    Returns
    -------
    A matplotlib Axes object with the heatmap.
    
    Examples
    --------
    >>> import numpy as np
    >>> from gofast.plot.utils import plot_cluster_comparison
    >>> # Sample data generation (This should be replaced with the actual data)
    >>> # For illustration purposes, we create a DataFrame with random data
    >>> np.random.seed(0)
    >>> sample_data = pd.DataFrame({
        'Cluster': np.random.choice(['1', '4', '5'], size=100),
        'PriorClass': np.random.choice(['1', '2', '3'], size=100)
    >>> })

    >>> # Call the function with the sample data
    >>> plot_cluster_comparison(sample_data, 'Cluster', 'PriorClass')
    >>> plt.show()
    """

    # Create a contingency table
    contingency_table = pd.crosstab(data[cluster_col], data[class_col])

    # Calculate Pearson residuals
    chi2, _, _, expected = scipy.stats.chi2_contingency(contingency_table)
    pearson_residuals = (contingency_table - expected) / np.sqrt(expected)

    # Create the heatmap
    plt.figure(figsize=figsize)
    ax = sns.heatmap(pearson_residuals, annot=True, fmt=".2f", cmap=palette,
                     cbar_kws={'label': 'Pearson residuals'})
    ax.set_title(title)
    ax.set_xlabel(class_col)
    ax.set_ylabel(cluster_col)

    # Display p-value
    p_value = scipy.stats.chi2.sf(chi2, (contingency_table.shape[0] - 1) 
                                  * (contingency_table.shape[1] - 1))
    plt.text(0.95, 0.05, f'p-value\n{p_value:.2e}', horizontalalignment='right', 
             verticalalignment='bottom', 
             transform=ax.transAxes, color='black', bbox=dict(
                 facecolor='white', alpha=0.5))

    return ax

def plot_shap_summary(
    model: Any, 
    X: Union[ArrayLike, DataFrame], 
    feature_names: Optional[List[str]] = None, 
    plot_type: str = 'dot', 
    color_bar_label: str = 'Feature value', 
    max_display: int = 10, 
    show: bool = True, 
    plot_size: Tuple[int, int] = (15, 10), 
    cmap: str = 'coolwarm'
) -> Optional[Figure]:
    """
    Generate a SHAP (SHapley Additive exPlanations) summary plot for a 
    given model and dataset.

    Parameters
    ----------
    model : model object
        A trained model object that is compatible with SHAP explainer.

    X : array-like or DataFrame
        Input data for which the SHAP values are to be computed. If a DataFrame is
        provided, the feature names are taken from the DataFrame columns.

    feature_names : list, optional
        List of feature names if `X` is an array-like object 
        without feature names.

    plot_type : str, optional
        Type of the plot. Either 'dot' or 'bar'. The default is 'dot'.

    color_bar_label : str, optional
        Label for the color bar. The default is 'Feature value'.

    max_display : int, optional
        Maximum number of features to display on the summary plot. 
        The default is 10.

    show : bool, optional
        Whether to show the plot. The default is True. If False, 
        the function returns the figure object.

    plot_size : tuple, optional
        Size of the plot specified as (width, height). The default is (15, 10).

    cmap : str, optional
        Colormap to use for plotting. The default is 'coolwarm'.

    Returns
    -------
 
    figure : matplotlib Figure object or None
        The figure object if `show` is False, otherwise None.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from gofast.datasets import make_classification
    >>> from gofast.plot import plot_shap_summary
    >>> X, y = make_classification(n_features=5, random_state=42, return_X_y=True)
    >>> model = RandomForestClassifier().fit(X, y)
    >>> plot_shap_summary(model, X, feature_names=['f1', 'f2', 'f3', 'f4', 'f5'])

    """
    import_optional_dependency(
        "shap", extra="'shap' package is need for plot_shap_summary.")

    import shap

    # Compute SHAP values
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # Create a summary plot
    plt.figure(figsize=plot_size)
    shap.summary_plot(
        shap_values, X, 
        feature_names=feature_names, 
        plot_type=plot_type,
        color_bar_label=color_bar_label, 
        max_display=max_display, 
        show=False
    )

    # Customize color bar label
    color_bar = plt.gcf().get_axes()[-1]
    color_bar.set_title(color_bar_label)

    # Set colormap if specified
    if cmap:
        plt.set_cmap(cmap)

    # Show or return the figure
    if show:
        plt.show()
    else:
        return plt.gcf()


def plot_cumulative_variance(
    data: np.ndarray,
    n_components: Optional[int] = None,
    threshold: Optional[float] = None,
    figsize: Tuple[int, int] = (10, 6),
    threshold_color: str = 'red',
    line_color: str = 'teal',
    title: str = None, 
    xlabel: str = None, 
    ylabel: str = None, 
    threshold_label: Optional[str] =None, 
    grid_style: str = ':',
    grid_width: float = 0.5,
    axis_width: float = 2,
    axis_color: str = 'black',
    show_grid: bool = True
) -> plt.Axes:
    """
    Plots the cumulative explained variance ratio by principal components 
    using PCA.
    
    Optionally, a threshold line can be drawn to indicate the desired level 
    of explained variance.

    Parameters
    ----------
    data : np.ndarray
        The input dataset for PCA. Must be 2D (samples x features).
    n_components : int, optional
        Number of principal components to consider. Defaults to min(data.shape).
    threshold : float, optional
        A variance ratio threshold to draw a horizontal line. Defaults to None.
    figsize : Tuple[int, int], optional
        Size of the figure (width, height) in inches. Defaults to (10, 6).
    threshold_color : str, optional
        Color of the threshold line. Defaults to 'red'.
    line_color : str, optional
        Color of the cumulative variance line. Defaults to 'teal'.
    title : str, optional
        Title of the plot. Defaults to 'Cumulative Explained Variance 
        Ratio by Principal Components'.
    xlabel : str, optional
        X-axis label. Defaults to 'Number of Components'.
    ylabel : str, optional
        Y-axis label. Defaults to 'Cumulative Explained Variance Ratio'.
    threshold_label : str, optional
        Label for the threshold line. Defaults to 'Variance Threshold'.
    grid_style : str, optional
        Style of the grid lines (lines, dashes, dots, etc.).
        Defaults to ':' (dotted line).
    grid_width : float, optional
        Width of the grid lines. Defaults to 0.5.
    axis_width : float, optional
        Width of the axes' spines. Defaults to 2.
    axis_color : str, optional
        Color of the axes' spines. Defaults to 'black'.
    show_grid : bool, optional
        If True, display grid lines on the plot. Defaults to True.
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        The Axes object with the plot for further customization.
    
    Raises
    ------
    ValueError
        If 'data' is not a 2D array.
        If 'n_components' is greater than the number of features in 'data'.
        If 'threshold' is not between 0 and 1 when provided.
    
    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> ax = plot_cumulative_variance(iris.data)
    >>> ax.set_title('Updated Plot Title')
    >>> plt.show()
    """
    title= title or  'Cumulative Explained Variance Ratio by Principal Components'
    xlabel = xlabel or 'Number of Components',
    ylabel= ylabel or 'Cumulative Explained Variance Ratio'
    threshold_label =threshold_label or 'Variance Threshold'
    
    if data.ndim != 2:
        raise ValueError("Input 'data' must be a 2D array.")
    
    if n_components is None:
        n_components = min(data.shape)
    elif n_components > data.shape[1]:
        raise ValueError("'n_components' cannot be greater than "
                         f"the number of features ({data.shape[1]}).")
    
    if threshold is not None and (threshold < 0 or threshold > 1):
        raise ValueError("'threshold' must be between 0 and 1.")
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca.fit(data)
    
    # Calculate the cumulative explained variance ratio
    cumulative_variance= np.cumsum(pca.explained_variance_ratio_)
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the cumulative explained variance ratio
    ax.plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
             marker='o', linestyle='-', color=line_color,
             label='Cumulative Variance')
    
    # Plot the threshold line if provided
    if threshold is not None:
        ax.axhline(y=threshold, color=threshold_color, linestyle='--',
                   label=f'{threshold_label} ({threshold:.2f})' 
                   if threshold_label else None)
    
    # Customize the plot
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(1, len(cumulative_variance) + 1))
    ax.legend(loc='best')
    
    # Customize the grid
    if show_grid:
        ax.grid(True, linestyle=grid_style, linewidth=grid_width)
    
    # Customize the axes' appearance
    for spine in ax.spines.values():
        spine.set_linewidth(axis_width)
        spine.set_color(axis_color)
    
    plt.show()
    
    return ax

def plot_cv(
    model_fn, X, y, 
    n_splits=5, 
    epochs=10, 
    metric='accuracy'
    ):
    """
    Performs cross-validation and plots the 
    performance of each fold.

    Parameters
    ----------
    model_fn : function
        A function that returns a compiled neural network model.

    X : np.ndarray
        Training features.

    y : np.ndarray
        Training labels.

    n_splits : int, optional
        Number of splits for cross-validation.

    epochs : int, optional
        Number of epochs for training.

    metric : str, optional
        The performance metric to plot ('accuracy', 'loss', etc.).

    """
    kf = KFold(n_splits=n_splits)
    fold_performance = []

    for fold, (train_index, val_index) in enumerate(kf.split(X), 1):
        # Split data
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Create a new instance of the model
        model = model_fn()

        # Train the model
        history = model.fit(
            X_train, y_train, validation_data=(X_val, y_val),
            epochs=epochs, verbose=0)

        # Store the metrics for this fold
        fold_performance.append(history.history[metric])

    # Plot the results
    plt.figure(figsize=(12, 6))
    for i, performance in enumerate(fold_performance, 1):
        plt.plot(performance, label=f'Fold {i}')

    plt.title(f'Cross-Validation {metric.capitalize()}')
    plt.xlabel('Epochs')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.show()
    
def plot_taylor_diagram(
        models, reference, model_names=None
        ):
    """
    Plots a Taylor Diagram, which is used to display a statistical 
    comparison of models.

    Parameters
    ----------
    models : list of np.ndarray
        A list of arrays, each containing the predictions of a 
        different model.
        Each model's predictions should be the same length as 
        the reference array.

    reference : np.ndarray
        An array containing the reference data against which the 
        models are compared.
        This should have the same length as each model's predictions.

    model_names : list of str, optional
        A list of names for each model. This list should be the same
        length as the 'models' list.
        If not provided, models will be labeled as Model 1, Model 2, etc.

    """

    # Calculate statistics for each model
    correlations = [np.corrcoef(model, reference)[0, 1] for model in models]
    standard_deviations = [np.std(model) for model in models]
    reference_std = np.std(reference)

    # Create polar plot
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(111, polar=True)

    # Convert correlations to angular coordinates in radians
    angles = np.arccos(correlations)
    radii = standard_deviations

    # Plot each model
    for i in range(len(models)):
        ax1.plot([angles[i], angles[i]], [0, radii[i]], 
                 label=f'{model_names[i]}' if model_names else f'Model {i+1}')
        ax1.plot(angles[i], radii[i], 'o')

    # Add reference standard deviation
    ax1.plot([0, 0], [0, reference_std], label='Reference')
    ax1.plot(0, reference_std, 'o')

    # Set labels and title
    ax1.set_thetamax(180)
    ax1.set_theta_direction(-1)
    ax1.set_theta_zero_location('W')
    ax1.set_rlabel_position(90)
    ax1.set_xlabel('Standard Deviation')
    ax1.set_ylabel('Correlation')
    ax1.set_title('Taylor Diagram')

    plt.legend()
    plt.show()
    
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
        
        .. versionadded:: 0.2.3 
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
    >>> from gofast.tools.utils import make_plot_colors
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


def plot_base_silhouette (X, labels, metric ='euclidean',savefig =None , **kwds ):
    r"""Plot quantifying the quality  of clustering silhouette 
    
    Parameters 
    ---------
    X : array-like of shape (n_samples_a, n_samples_a) if metric == \
            "precomputed" or (n_samples_a, n_features) otherwise
        An array of pairwise distances between samples, or a feature array.

    labels : array-like of shape (n_samples,)
        Label values for each sample.

    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by :func:`sklearn.metrics.pairwise.pairwise_distances`.
        If ``X`` is the distance array itself, use "precomputed" as the metric.
        Precomputed distance matrices must have 0 along the diagonal.
        
    savefig: str, default =None , 
        the path to save the figure. Argument is passed to 
        :class:`matplotlib.Figure` class. 
        
    **kwds : optional keyword parameters
        Any further parameters are passed directly to the distance function.
        If using a ``scipy.spatial.distance`` metric, the parameters are still
        metric dependent. See the scipy docs for usage examples.
        
        
    See Also
    --------
    gofast.view.mlplot.plotSilhouette: 
        Gives consistency plot as the use of `prefit` parameter which checks 
        whether`labels` are expected to be passed into the function 
        directly or not. 
    
    Examples
    ---------
    >>> import numpy as np 
    >>> from gofast.exlib.sklearn import KMeans 
    >>> from gofast.datasets import load_iris 
    >>> from gofast.tools.utils import plot_base_silhouette
    >>> d= load_iris ()
    >>> X= d.data [:, 0][:, np.newaxis] # take the first axis 
    >>> km= KMeans (n_clusters =3 , init='k-means++', n_init =10 , 
                    max_iter = 300 , 
                    tol=1e-4, 
                    random_state =0 
                    )
    >>> y_km = km.fit_predict(X) 
    >>> plot_base_silhouette (X, y_km)

    """
    X, labels = check_X_y(
        X, 
        labels, 
        to_frame= True, 
        )
    cluster_labels = np.unique (labels) 
    n_clusters = cluster_labels.shape [0] 
    silhouette_vals = silhouette_samples(
        X, labels= labels, metric = metric ,**kwds)
    y_ax_lower , y_ax_upper = 0, 0 
    yticks =[]
    
    for i, c  in enumerate (cluster_labels ) : 
        c_silhouette_vals = silhouette_vals[labels ==c ] 
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color =mpl.cm.jet (float(i)/n_clusters )
        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, 
                 height =1.0 , 
                 edgecolor ='none', 
                 color =color, 
                 )
        yticks.append((y_ax_lower + y_ax_upper)/2.)
        y_ax_lower += len(c_silhouette_vals)
    silhouette_avg = np.mean(silhouette_vals) 
    plt.axvline (silhouette_avg, 
                 color='red', 
                 linestyle ='--'
                 )
    plt.yticks(yticks, cluster_labels +1 ) 
    plt.ylabel ("Cluster") 
    plt.xlabel ("Silhouette coefficient")
    plt.tight_layout()

    if savefig is not None:
        plt.savefig(savefig, dpi = 300 )
        
    plt.close () if savefig is not None else plt.show() 
    

def plot_sbs_feature_selection(
        sbs_estimator,/, X=None, y=None, fig_size=(8, 5), 
        sns_style=False, savefig=None, verbose=0, **sbs_kws
    ):
    """
    Plot the feature selection process using Sequential Backward Selection (SBS).

    This function visualizes the selection of the best feature subset at each stage 
    in the SBS algorithm. It requires either a fitted SBS estimator or the training
    data (`X` and `y`) to fit the estimator during the plot generation.

    Parameters
    ----------
    sbs_estimator : :class:`~.gofast.transformers.SequentialBackwardSelection`
        The SBS estimator. Can be pre-fitted; if not, `X` and `y` must be provided
        for fitting during the plot generation.

    X : array-like of shape (n_samples, n_features), optional
        Training data, with `n_samples` as the number of samples and `n_features`
        as the number of features. Required if `sbs_estimator` is not pre-fitted.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs), optional
        Target values corresponding to `X`. Required if `sbs_estimator` is not
        pre-fitted.

    fig_size : tuple of (width, height), default=(8, 5)
        Size of the matplotlib figure, specified as a width and height tuple.

    sns_style : bool, default=False
        If True, apply seaborn styling to the plot.

    savefig : str, optional
        File path where the figure is saved. If provided, the plot is saved
        to this path.

    verbose : int, default=0
        If set to a positive number, print feature labels and their importance
        rates.

    sbs_kws : dict, optional
        Additional keyword arguments passed to the
        :class:`~.gofast.base.SequentialBackwardSelection` class.

    Examples
    --------
    # Example 1: Plotting a pre-fitted SBS
    >>> from sklearn.neighbors import KNeighborsClassifier, train_test_split
    >>> from gofast.datasets import fetch_data
    >>> from gofast.transformers import SequentialBackwardSelection
    >>> from gofast.tools.utils import plot_sbs_feature_selection
    >>> X, y = fetch_data('bagoue analysed')  # Data already standardized
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> knn = KNeighborsClassifier(n_neighbors=5)
    >>> sbs = SequentialBackwardSelection(knn)
    >>> sbs.fit(X_train, y_train)
    >>> plot_sbs_feature_selection(sbs, sns_style=True)

    # Example 2: Plotting an SBS estimator without pre-fitting
    >>> plot_sbs_feature_selection(knn, X_train, y_train)  # Same result as above
    """

    from ..transformers import SequentialBackwardSelection as SBS 
    if ( 
        not hasattr (sbs_estimator, 'scores_') 
        and not hasattr (sbs_estimator, 'k_score_')
            ): 
        if ( X is None or y is None ) : 
            clfn = get_estimator_name( sbs_estimator)
            raise TypeError (f"When {clfn} is not a fitted "
                             "estimator, X and y are needed."
                             )
        sbs_estimator = SBS(estimator = sbs_estimator, **sbs_kws)
        sbs_estimator.fit(X, y )
        
    k_feat = [len(k) for k in sbs_estimator.subsets_]
    
    if verbose: 
        flabels =None 
        if  ( not hasattr (X, 'columns') and X is not None ): 
            warnings.warn("None columns name is detected."
                          " Created using index ")
            flabels =[f'{i:>7}' for i in range (X.shape[1])]
            
        elif hasattr (X, 'columns'):
            flabels = list(X.columns)  
        elif hasattr ( sbs_estimator , 'feature_names_in'): 
            flabels = sbs_estimator.feature_names_in 
            
        if flabels is not None: 
            k3 = list (sbs_estimator.subsets_[X.shape[1]])
            print("Smallest feature for subset (k=3) ")
            print(flabels [k3])
            
        else : print("No column labels detected. Can't print the "
                     "smallest feature subset.")
        
    if sns_style: 
        _set_sns_style (sns_style)
        
    plt.figure(figsize = fig_size)
    plt.plot (k_feat , sbs_estimator.scores_, marker='o' ) 
    plt.ylim ([min(sbs_estimator.scores_) -.25 ,
               max(sbs_estimator.scores_) +.2 ])
    plt.ylabel (sbs_estimator.scorer_name_ )
    plt.xlabel ('Number of features')
    plt.tight_layout() 
    
    if savefig is not None:
        plt.savefig(savefig )
        
    plt.close () if savefig is not None else plt.show() 
    

def plot_regularization_path ( 
        X, y , c_range=(-4., 6. ), fig_size=(8, 5), sns_style =False, 
        savefig = None, **kws 
        ): 
    r""" Plot the regularisation path from Logit / LogisticRegression 
    
    Varying the  different regularization strengths and plot the  weight 
    coefficient of the different features for different regularization 
    strength. 
    
    Note that, it is recommended to standardize the data first. 
    
    Parameters 
    -----------
    X : array-like of shape (n_samples, n_features)
        Training vector, where `n_samples` is the number of samples and
        `n_features` is the number of features. X is expected to be 
        standardized. 

    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Target relative to X for classification or regression;
        None for unsupervised learning.
    c_range: list or tuple [start, stop] 
        Regularization strength list. It is a range from the strong  
        strong ( start) to lower (stop) regularization. Note that 'C' is 
        the inverse of the Logistic Regression regularization parameter 
        :math:`\lambda`. 
    fig_size : tuple (width, height), default =(8, 6)
        the matplotlib figure size given as a tuple of width and height
        
    savefig: str, default =None , 
        the path to save the figures. Argument is passed to matplotlib.Figure 
        class. 
    sns_style: str, optional, 
        the seaborn style.
        
    kws: dict, 
        Additional keywords arguments passed to 
        :class:`sklearn.linear_model.LogisticRegression`
    
    Examples
    --------
    >>> from gofast.tools.utils import plot_regularization_path 
    >>> from gofast.datasets import fetch_data
    >>> X, y = fetch_data ('bagoue analysed' ) # data aleardy standardized
    >>> plot_regularization_path (X, y ) 

    """
    X, y = check_X_y(
        X, 
        y, 
        to_frame= True, 
        )
    
    if not is_iterable(c_range): 
        raise TypeError ("'C' regularization strength is a range of C " 
                         " Logit parameter: (start, stop).")
    c_range = sorted (c_range )
    
    if len(c_range) < 2: 
        raise ValueError ("'C' range expects two values [start, stop]")
        
    if len(c_range) >2 : 
        warnings.warn ("'C' range expects two values [start, stop]. Values"
                       f" are shrunk to the first two values: {c_range[:2]} "
                       )
    weights, params = [], []    
    for c in np.arange (*c_range): 
        lr = LogisticRegression(penalty='l1', C= 10.**c, solver ='liblinear', 
                                multi_class='ovr', **kws)
        lr.fit(X,y )
        weights.append (lr.coef_[1])
        params.append(10**c)
        
    weights = np.array(weights ) 
    colors = make_mpl_properties(weights.shape[1])
    if not hasattr (X, 'columns'): 
        flabels =[f'{i:>7}' for i in range (X.shape[1])] 
    else: flabels = X.columns   
    
    # plot
    fig, ax = plt.subplots(figsize = fig_size )
    if sns_style: 
        _set_sns_style (sns_style)

    for column , color in zip( range (weights.shape [1]), colors ): 
        plt.plot (params , weights[:, column], 
                  label =flabels[column], 
                  color = color 
                  )

    plt.axhline ( 0 , color ='black', ls='--', lw= 3 )
    plt.xlim ( [ 10 ** int(c_range[0] -1), 10 ** int(c_range[1]-1) ])
    plt.ylabel ("Weight coefficient")
    plt.xlabel ('C')
    plt.xscale( 'log')
    plt.legend (loc ='upper left',)
    ax.legend(
            loc ='upper right', 
            bbox_to_anchor =(1.38, 1.03 ), 
            ncol = 1 , fancybox =True 
    )
    
    if savefig is not None:
        plt.savefig(savefig, dpi = 300 )
        
    plt.close () if savefig is not None else plt.show() 
    
def plot_rf_feature_importances (
    clf, 
    X=None, 
    y=None, 
    fig_size = (8, 4),
    savefig =None,   
    n_estimators= 500, 
    verbose =0 , 
    sns_style =None,  
    **kws 
    ): 
    """
    Plot features importance with RandomForest.  
    
    Parameters 
    ----------
    clf : estimator object
        The base estimator from which the transformer is built.
        This can be both a fitted (if ``prefit`` is set to True)
        or a non-fitted estimator. The estimator should have a
        ``feature_importances_`` or ``coef_`` attribute after fitting.
        Otherwise, the ``importance_getter`` parameter should be used.
        
    X : array-like of shape (n_samples, n_features)
        Training vector, where `n_samples` is the number of samples and
        `n_features` is the number of features.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Target relative to X for classification or regression;
        None for unsupervised learning.
       
    n_estimators : int, default=500
        The number of trees in the forest.
        
    fig_size : tuple (width, height), default =(8, 6)
        the matplotlib figure size given as a tuple of width and height
        
    savefig: str, default =None , 
        the path to save the figures. Argument is passed to matplotlib.Figure 
        class. 
    sns_style: str, optional, 
        the seaborn style.
    verbose: int, default=0 
        print the feature labels with the rate of their importances. 
    kws: dict, 
        Additional keyyword arguments passed to 
        :class:`sklearn.ensemble.RandomForestClassifier`
    Examples
    ---------
    >>> from gofast.datasets import fetch_data
    >>> from sklearn.ensemble import RandomForestClassifier 
    >>> from gofast.plot.utils import plot_rf_feature_importances 
    >>> X, y = fetch_data ('bagoue analysed' ) 
    >>> plot_rf_feature_importances (
        RandomForestClassifier(), X=X, y=y , sns_style=True)

    """
    if not hasattr (clf, 'feature_importances_'): 
        if ( X is None or y is None ) : 
            clfn = get_estimator_name( clf)
            raise TypeError (f"When {clfn} is not a fitted "
                             "estimator, X and y are needed."
                             )
        clf = RandomForestClassifier(n_estimators= n_estimators , **kws)
        clf.fit(X, y ) 
        
    importances = clf.feature_importances_ 
    indices = np.argsort(importances)[::-1]
    if hasattr( X, 'columns'): 
        flabels = X.columns 
    else : flabels =[f'{i:>7}' for i in range (X.shape[1])]
    
    if verbose : 
        for f in range(X.shape [1]): 
            print("%2d) %-*s %f" %(f +1 , 30 , flabels[indices[f]], 
                                   importances[indices[f]])
                  )
    if sns_style: 
        _set_sns_style (sns_style)

    plt.figure(figsize = fig_size)
    plt.title ("Feature importance")
    plt.bar (range(X.shape[1]) , 
             importances [indices], 
             align='center'
             )
    plt.xticks (range (X.shape[1]), flabels [indices], rotation =90 , 
                ) 
    plt.xlim ([-1 , X.shape[1]])
    plt.ylabel ('Importance rate')
    plt.xlabel ('Feature labels')
    plt.tight_layout()
    
    if savefig is not None:
        plt.savefig(savefig )

    plt.close () if savefig is not None else plt.show() 
    
        
def plot_confusion_matrix (yt, y_pred, view =True, ax=None, annot=True,  **kws ):
    """ plot a confusion matrix for a single classifier model.
    
    :param yt : ndarray or Series of length n
        An array or series of true target or class values. Preferably, 
        the array represents the test class labels data for error evaluation.
    
    :param y_pred: ndarray or Series of length n
        An array or series of the predicted target. 
    :param view: bool, default=True 
        Option to display the matshow map. Set to ``False`` mutes the plot. 
    :param annot: bool, default=True 
        Annotate the number of samples (right or wrong prediction ) in the plot. 
        Set ``False`` to mute the display.
    param kws: dict, 
        Additional keyword arguments passed to the function 
        :func:`sckitlearn.metrics.confusion_matrix`. 
    :returns: mat- confusion matrix bloc matrix 
    
    :example: 
    >>> #Import the required models and fetch a an Ababoost model 
    >>> # for instance then plot the confusion metric 
    >>> import matplotlib.pyplot as plt 
    >>> plt.style.use ('classic')
    >>> from gofast.datasets import fetch_data
    >>> from gofast.exlib.sklearn import train_test_split 
    >>> from gofast.models import pModels 
    >>> from gofast.tools.utils import plot_confusion_matrix
    >>> # split the  data . Note that fetch_data output X and y 
    >>> X, Xt, y, yt  = train_test_split (* fetch_data ('bagoue analysed'),
                                          test_size =.25  )  
    >>> # train the model with the best estimator 
    >>> pmo = pModels (model ='ada' ) 
    >>> pmo.fit(X, y )
    >>> print(pmo.estimator_ )
    >>> #%% 
    >>> # Predict the score using under the hood the best estimator 
    >>> # for adaboost classifier 
    >>> ypred = pmo.predict(Xt) 
    >>> # now plot the score 
    >>> plot_confusion_matrix (yt , ypred )
    """
    check_consistent_length (yt, y_pred)
    mat= confusion_matrix (yt, y_pred, **kws)
    if ax is None: 
        fig, ax = plt.subplots ()
        
    if view: 
        sns.heatmap (
            mat.T, square =True, annot =annot, cbar=False, ax=ax)
        # xticklabels= list(np.unique(ytrue.values)), 
        # yticklabels= list(np.unique(ytrue.values)))
        ax.set_xlabel('true labels' )
        ax.set_ylabel ('predicted labels')
    return mat 

def plot_yb_confusion_matrix (
        clf, Xt, yt, labels = None , encoder = None, savefig =None, 
        fig_size =(6, 6), **kws
        ): 
    """ Confusion matrix plot using the 'yellowbrick' package.  
    
    Creates a heatmap visualization of the sklearn.metrics.confusion_matrix().
    A confusion matrix shows each combination of the true and predicted
    classes for a test data set.

    The default color map uses a yellow/orange/red color scale. The user can
    choose between displaying values as the percent of true (cell value
    divided by sum of row) or as direct counts. If percent of true mode is
    selected, 100% accurate predictions are highlighted in green.

    Requires a classification model.
    
    Be sure 'yellowbrick' is installed before using the function, otherwise an 
    ImportError will raise. 
    
    Parameters 
    -----------
    clf : classifier estimator
        A scikit-learn estimator that should be a classifier. If the model is
        not a classifier, an exception is raised. If the internal model is not
        fitted, it is fit when the visualizer is fitted, unless otherwise specified
        by ``is_fitted``.
        
    Xt : ndarray or DataFrame of shape n x m
        A matrix of n instances with m features. Preferably, matrix represents 
        the test data for error evaluation.  

    yt : ndarray or Series of length n
        An array or series of target or class values. Preferably, the array 
        represent the test class labels data for error evaluation.  

    ax : matplotlib Axes, default: None
        The axes to plot the figure on. If not specified the current axes will be
        used (or generated if required).

    sample_weight: array-like of shape = [n_samples], optional
        Passed to ``confusion_matrix`` to weight the samples.
        
    encoder : dict or LabelEncoder, default: None
        A mapping of classes to human readable labels. Often there is a mismatch
        between desired class labels and those contained in the target variable
        passed to ``fit()`` or ``score()``. The encoder disambiguates this mismatch
        ensuring that classes are labeled correctly in the visualization.
        
    labels : list of str, default: None
        The class labels to use for the legend ordered by the index of the sorted
        classes discovered in the ``fit()`` method. Specifying classes in this
        manner is used to change the class names to a more specific format or
        to label encoded integer classes. Some visualizers may also use this
        field to filter the visualization for specific classes. For more advanced
        usage specify an encoder rather than class labels.
        
    fig_size : tuple (width, height), default =(8, 6)
        the matplotlib figure size given as a tuple of width and height
        
    savefig: str, default =None , 
        the path to save the figures. Argument is passed to matplotlib.Figure 
        class. 
          
    Returns 
    --------
    cmo: :class:`yellowbrick.classifier.confusion_matrix.ConfusionMatrix`
        return a yellowbrick confusion matrix object instance. 
    
    Examples 
    --------
    >>> #Import the required models and fetch a an extreme gradient boosting 
    >>> # for instance then plot the confusion metric 
    >>> import matplotlib.pyplot as plt 
    >>> plt.style.use ('classic')
    >>> from gofast.datasets import fetch_data
    >>> from gofast.exlib.sklearn import train_test_split 
    >>> from gofast.models import pModels 
    >>> from gofast.tools.utils import plot_yb_confusion_matrix
    >>> # split the  data . Note that fetch_data output X and y 
    >>> X, Xt, y, yt  = train_test_split (* fetch_data ('bagoue analysed'),
                                          test_size =.25  )  
    >>> # train the model with the best estimator 
    >>> pmo = pModels (model ='xgboost' ) 
    >>> pmo.fit(X, y )
    >>> print(pmo.estimator_ ) # pmo.XGB.best_estimator_
    >>> #%% 
    >>> # Predict the score using under the hood the best estimator 
    >>> # for adaboost classifier 
    >>> ypred = pmo.predict(Xt) 
    
    >>> # now plot the score 
    >>> plot_yb_confusion_matrix (pmo.XGB.best_estimator_, Xt, yt  )
    """
    import_optional_dependency('yellowbrick', (
        "Cannot plot the confusion matrix via 'yellowbrick' package."
        " Alternatively, you may use ufunc `~.plot_confusion_matrix`,"
        " otherwise install it mannually.")
        )
    from yellowbrick.classifier import ConfusionMatrix 
    fig, ax = plt.subplots(figsize = fig_size )
    cmo= ConfusionMatrix (clf, classes=labels, 
                         label_encoder = encoder, **kws
                         )
    cmo.score(Xt, yt)
    cmo.show()

    if savefig is not None: 
        fig.savefig(savefig, dpi =300)

    plt.close () if savefig is not None else plt.show() 
    
    return cmo 

def plot_confusion_matrices (
    clfs, 
    Xt, 
    yt,  
    annot =True, 
    pkg=None, 
    normalize='true', 
    sample_weight=None,
    encoder=None, 
    fig_size = (22, 6),
    savefig =None, 
    subplot_kws=None,
    **scorer_kws
    ):
    """ 
    Plot inline multiple model confusion matrices using either the sckitlearn 
    or 'yellowbrick'
    
    Parameters 
    -----------
    clfs : list of classifier estimators
        A scikit-learn estimator that should be a classifier. If the model is
        not a classifier, an exception is raised. Note that the classifier 
        must be fitted beforehand.
        
    Xt : ndarray or DataFrame of shape (M X N)
        A matrix of n instances with m features. Preferably, matrix represents 
        the test data for error evaluation.  

    yt : ndarray of shape (M, ) or Series oF length (M, )
        An array or series of target or class values. Preferably, the array 
        represent the test class labels data for error evaluation.  
    
    pkg: str, optional , default ='sklearn'
        the library to handle the plot. It could be 'yellowbrick'. The basic 
        confusion matrix is handled by the scikit-learn package. 

    normalize : {'true', 'pred', 'all'}, default=None
        Normalizes confusion matrix over the true (rows), predicted (columns)
        conditions or all the population. If None, confusion matrix will not be
        normalized.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
        
    encoder : dict or LabelEncoder, default: None
        A mapping of classes to human readable labels. Often there is a mismatch
        between desired class labels and those contained in the target variable
        passed to ``fit()`` or ``score()``. The encoder disambiguates this mismatch
        ensuring that classes are labeled correctly in the visualization.
        
        
    annot: bool, default=True 
        Annotate the number of samples (right or wrong prediction ) in the plot. 
        Set ``False`` to mute the display. 
    
    fig_size : tuple (width, height), default =(8, 6)
        the matplotlib figure size given as a tuple of width and height
        
    savefig: str, default =None , 
        the path to save the figures. Argument is passed to matplotlib.Figure 
        class. 
        
    Examples
    ----------
    >>> import matplotlib.pyplot as plt 
    >>> plt.style.use ('classic')
    >>> from gofast.datasets import fetch_data
    >>> from gofast.exlib.sklearn import train_test_split 
    >>> from gofast.models.premodels import p
    >>> from gofast.tools.utils import plot_confusion_matrices 
    >>> # split the  data . Note that fetch_data output X and y 
    >>> X, Xt, y, yt  = train_test_split (* fetch_data ('bagoue analysed'), test_size =.25  )  
    >>> # compose the models 
    >>> # from RBF, and poly 
    >>> models =[ p.SVM.rbf.best_estimator_,
             p.LogisticRegression.best_estimator_,
             p.RandomForest.best_estimator_ 
             ]
    >>> models 
    [SVC(C=2.0, coef0=0, degree=1, gamma=0.125), LogisticRegression(), 
     RandomForestClassifier(criterion='entropy', max_depth=16, n_estimators=350)]
    >>> # now fit all estimators 
    >>> fitted_models = [model.fit(X, y) for model in models ]
    >>> plot_confusion_matrices(fitted_models , Xt, yt)
    """
    pkg = pkg or 'sklearn'
    pkg= str(pkg).lower() 
    assert pkg in {"sklearn", "scikit-learn", 'yellowbrick', "yb"}, (
        f" Accepts only 'sklearn' or 'yellowbrick' packages, got {pkg!r}") 
    
    if not is_iterable( clfs): 
        clfs =[clfs]

    model_names = [get_estimator_name(name) for name in clfs ]
    # create a figure 
    subplot_kws = subplot_kws or dict (left=0.0625, right = 0.95, 
                                       wspace = 0.12)
    fig, axes = plt.subplots(1, len(clfs), figsize =(22, 6))
    fig.subplots_adjust(**subplot_kws)
    if not is_iterable(axes): 
       axes =[axes] 
    for kk, (model , mname) in enumerate(zip(clfs, model_names )): 
        ypred = model.predict(Xt)
        if pkg in ('sklearn', 'scikit-learn'): 
            plot_confusion_matrix(yt, ypred, annot =annot , ax = axes[kk], 
                normalize= normalize , sample_weight= sample_weight ) 
            axes[kk].set_title (mname)
            
        elif pkg in ('yellowbrick', 'yb'):
            plot_yb_confusion_matrix(
                model, Xt, yt, ax=axes[kk], encoder =encoder )
    if savefig is not None:
        plt.savefig(savefig, dpi = 300 )
        
    plt.close () if savefig is not None else plt.show() 
    
def plot_learning_curves(
    models, 
    X ,
    y, 
    *, 
    cv =None, 
    train_sizes= None, 
    baseline_score =0.4,
    scoring=None, 
    convergence_line =True, 
    fig_size=(20, 6),
    sns_style =None, 
    savefig=None, 
    set_legend=True, 
    subplot_kws=None,
    **kws
    ): 
    """ 
    Horizontally visualization of multiple models learning curves. 
    
    Determines cross-validated training and test scores for different training
    set sizes.
    
    Parameters 
    ----------
    models: list or estimators  
        An estimator instance or not that implements `fit` and `predict` 
        methods which will be cloned for each validation. 
        
    X : array-like of shape (n_samples, n_features)
        Training vector, where `n_samples` is the number of samples and
        `n_features` is the number of features.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Target relative to X for classification or regression;
        None for unsupervised learning.
   
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For int/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        ``cv`` default value if None changed from 3-fold to 4-fold.
        
     train_sizes : array-like of shape (n_ticks,), \
             default=np.linspace(0.1, 1, 50)
         Relative or absolute numbers of training examples that will be used to
         generate the learning curve. If the dtype is float, it is regarded as a
         fraction of the maximum size of the training set (that is determined
         by the selected validation method), i.e. it has to be within (0, 1].
         Otherwise it is interpreted as absolute sizes of the training sets.
         Note that for classification the number of samples usually have to
         be big enough to contain at least one sample from each class.
         
    baseline_score: floatm default=.4 
        base score to start counting in score y-axis  (score)
        
    scoring : str or callable, default=None
        A str (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
        
    convergence_line: bool, default=True 
        display the convergence line or not that indicate the level of bias 
        between the training and validation curve. 
        
    fig_size : tuple (width, height), default =(14, 6)
        the matplotlib figure size given as a tuple of width and height
        
    sns_style: str, optional, 
        the seaborn style . 
        
    set_legend: bool, default=True 
        display legend in each figure. Note the default location of the 
        legend is 'best' from :func:`~matplotlib.Axes.legend`
        
    subplot_kws: dict, default is \
        dict(left=0.0625, right = 0.95, wspace = 0.1) 
        the subplot keywords arguments passed to 
        :func:`matplotlib.subplots_adjust` 
    kws: dict, 
        keyword arguments passed to :func:`sklearn.model_selection.learning_curve`
        
    Examples 
    ---------
    (1) -> plot via a metaestimator already cross-validated. 
    
    >>> import watex # must install watex to get the pretrained model ( pip install watex )
    >>> from watex.models.premodels import p 
    >>> from gofast.datasets import fetch_data 
    >>> from gofast.plot.utils import plot_learning_curves
    >>> X, y = fetch_data ('bagoue prepared') # yields a sparse matrix 
    >>> # let collect 04 estimators already cross-validated from SVMs
    >>> models = [ p.SVM.linear , p.SVM.rbf , p.SVM.sigmoid , p.SVM.poly ]
    >>> plot_learning_curves (models, X, y, cv=4, sns_style = 'darkgrid')
    
    (2) -> plot with  multiples models not crossvalidated yet.
    >>> from sklearn.linear_model import LogisticRegression 
    >>> from sklearn.svm import SVC 
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> models =[LogisticRegression(), RandomForestClassifier(), SVC() ,
                 KNeighborsClassifier() ]
    >>> plot_learning_curves (models, X, y, cv=4, sns_style = 'darkgrid')
    
    """
    if not is_iterable(models): 
        models =[models]
    
    subplot_kws = subplot_kws or  dict(
        left=0.0625, right = 0.95, wspace = 0.1) 
    train_sizes = train_sizes or np.linspace(0.1, 1, 50)
    cv = cv or 4 
    if ( 
        baseline_score >=1 
        and baseline_score < 0 
        ): 
        raise ValueError ("Score for the base line must be less 1 and "
                          f"greater than 0; got {baseline_score}")
    
    if sns_style: 
        _set_sns_style (sns_style)
        
    mnames = [get_estimator_name(n) for n in models]

    fig, axes = plt.subplots(nrows=1, ncols=len(models), figsize =fig_size)
    # for consistency, put axes on list when 
    # a single model is provided 
    if not is_iterable(axes): 
        axes =[axes] 
    fig.subplots_adjust(**subplot_kws)

    for k, (model, name) in enumerate(zip(models,  mnames)):
        cmodel = model.best_estimator_  if _is_cross_validated(
            model ) else model 
        ax = list(axes)[k]

        N, train_lc , val_lc = learning_curve(
            cmodel , 
            X, 
            y, 
            train_sizes = np.linspace(0.1, 1, 50),
            cv=cv, 
            scoring=scoring, 
            **kws
            )
        ax.plot(N, np.mean(train_lc, 1), 
                   color ="blue", 
                   label ="train score"
                   )
        ax.plot(N, np.mean(val_lc, 1), 
                   color ="r", 
                   label ="validation score"
                   )
        if convergence_line : 
            ax.hlines(np.mean([train_lc[-1], 
                                  val_lc[-1]]), 
                                 N[0], N[-1], 
                                 color="k", 
                                 linestyle ="--"
                         )
        ax.set_ylim(baseline_score, 1)
        #ax[k].set_xlim (N[0], N[1])
        ax.set_xlabel("training size")
        ax.set_title(name, size=14)
        if set_legend: 
            ax.legend(loc='best')
    # for consistency
    ax = list(axes)[0]
    ax.set_ylabel("score")
    
    if savefig is not None:
        plt.savefig(savefig, dpi = 300 )
        
    plt.close () if savefig is not None else plt.show() 
        
def plot_base_dendrogram (
        X, 
        *ybounds, 
        fig_size = (12, 5 ), 
        savefig=None,  
        **kws
        ): 
    """ Quick plot dendrogram using the ward clustering function from Scipy.
    
    :param X: ndarray of shape (n_samples, n_features) 
        Array of features 
    :param ybounds: int, 
        integrer values to draw horizontal cluster lines that indicate the 
        number of clusters. 
    :param fig_size: tuple (width, height), default =(12,5) 
        the matplotlib figure size given as a tuple of width and height 
    :param kws: dict , 
        Addditional keyword arguments passed to 
        :func:`scipy.cluster.hierarchy.dendrogram`
    :Examples: 
        >>> from gofast.datasets import fetch_data 
        >>> from gofast.tools.utils import plot_base_dendrogram
        >>> X, _= fetch_data('Bagoue analysed') # data is already scaled 
        >>> # get the two features 'power' and  'magnitude'
        >>> data = X[['power', 'magnitude']]
        >>> plot_base_dendrogram(data ) 
        >>> # add the horizontal line of the cluster at ybounds = (20 , 20 )
        >>> # for a single cluster (cluser 1)
        >>> plot_base_dendrogram(data , 20, 20 ) 
   
    """
    # assert ybounds agument if given
    msg =(". Note that the bounds in y-axis are the y-coordinates for"
          " horizontal lines regarding to the number of clusters that"
          " might be cutted.")
    try : 
        ybounds = [ int (a) for a in ybounds ] 
    except Exception as typerror: 
        raise TypeError  (str(typerror) + msg)
    else : 
        if len(ybounds)==0 : ybounds = None 
    # the scipy ward function returns 
    # an array that specifies the 
    # distance bridged when performed 
    # agglomerate clustering
    linkage_array = ward(X) 
    
    # plot the dendrogram for the linkage array 
    # containing the distances between clusters 
    dendrogram( linkage_array , **kws )
    
    # mark the cuts on the tree that signify two or three clusters
    # change the gca figsize 
    plt.rcParams["figure.figsize"] = fig_size
    ax= plt.gca () 
  
    if ybounds is not None: 
        if not is_iterable(ybounds): 
            ybounds =[ybounds] 
        if len(ybounds) <=1 : 
            warnings.warn(f"axis y bound might be greater than {len(ybounds)}")
        else : 
            # split ybound into sublist of pair (x, y) coordinates
            nsplits = len(ybounds)//2 
            len_splits = [ 2 for i in range (nsplits)]
            # compose the pir list (x,y )
            itb = iter (ybounds)
            ybounds = [list(itertools.islice (itb, it)) for it in len_splits]
            bounds = ax.get_xbound () 
            for i , ( x, y)  in enumerate (ybounds)  : 
                ax.plot(bounds, [x, y], '--', c='k') 
                ax.text ( bounds [1], y , f"cluster {i +1:02}",
                         va='center', 
                         fontdict ={'size': 15}
                         )
    # get xticks and format labels
    xticks_loc = list(ax.get_xticks())
    _get_xticks_formatage(ax, xticks_loc, space =14 )
    
    plt.xlabel ("Sample index ")
    plt.ylabel ("Cluster distance")
            
    if savefig is not None:
        plt.savefig(savefig, dpi = 300 )
        
    plt.close () if savefig is not None else plt.show() 
    
def plot_pca_components (
        components, *, feature_names = None , cmap= 'viridis', 
        savefig=None, **kws
        ): 
    """ Visualize the coefficient of principal component analysis (PCA) as 
    a heatmap  
  
    :param components: Ndarray, shape (n_components, n_features)or PCA object 
        Array of the PCA compoments or object from 
        :class:`gofast.analysis.dimensionality.nPCA`. If the object is given 
        it is not necessary to set the `feature_names`
    :param feature_names: list or str, optional 
        list of the feature names to locate in the map. `Feature_names` and 
        the number of eigen vectors must be the same length. If PCA object is  
        passed as `components` arguments, no need to set the `feature_names`. 
        The name of features is retreived automatically. 
    :param cmap: str, default='viridis'
        the matplotlib color map for matshow visualization. 
    :param kws: dict, 
        Additional keywords arguments passed to 
        :class:`matplotlib.pyplot.matshow`
        
    :Examples: 
    (1)-> with PCA object 
    
    >>> from gofast.datasets import fetch_data
    >>> from gofast.tools.utils import plot_pca_components
    >>> from gofast.analysis import nPCA 
    >>> X, _= fetch_data('bagoue pca') 
    >>> pca = nPCA (X, n_components=2, return_X =False)# to return object 
    >>> plot_pca_components (pca)
    
    (2)-> use the components and features individually 
    
    >>> components = pca.components_ 
    >>> features = pca.feature_names_in_
    >>> plot_pca_components (components, feature_names= features, 
                             cmap='jet_r')
    
    """
    if sp.issparse (components): 
        raise TypeError ("Sparse array is not supported for PCA "
                         "components visualization."
                         )
    # if pca object is given , get the features names
    if hasattr(components, "feature_names_in_"): 
        feature_names = list (getattr (components , "feature_names_in_" ) ) 
        
    if not hasattr (components , "__array__"): 
        components = _check_array_in  (components, 'components_')
        
    plt.matshow(components, cmap =cmap , **kws)
    plt.yticks ([0 , 1], ['First component', 'Second component']) 
    cb=plt.colorbar() 
    cb.set_label('Coeff value')
    if not is_iterable(feature_names ): 
        feature_names = [feature_names ]
        
    if len(feature_names)!= components.shape [1] :
        warnings.warn("Number of features and eigenvectors might"
                      " be consistent, expect {0}, got {1}". format( 
                          components.shape[1], len(feature_names))
                      )
        feature_names=None 
    if feature_names is not None: 
        plt.xticks (range (len(feature_names)), 
                    feature_names , rotation = 60 , ha='left' 
                    )
    plt.xlabel ("Feature") 
    plt.ylabel ("Principal components") 
    
    if savefig is not None:
        plt.savefig(savefig, dpi = 300 )
        
    plt.close () if savefig is not None else plt.show() 
    
        
def plot_clusters (
        n_clusters, X, y_pred, cluster_centers =None , savefig =None, 
        ): 
    """ Visualize the cluster that k-means identified in the dataset 
    
    :param n_clusters: int, number of cluster to visualize 
    :param X: NDArray, data containing the features, expect to be a two 
        dimensional data 
    :param y_pred: array-like, array containing the predicted class labels. 
    :param cluster_centers_: NDArray containg the coordinates of the 
        centroids or the similar points with continous features. 
        
    :Example: 
    >>> from gofast.exlib.sklearn import KMeans, MinMaxScaler
    >>> from gofast.tools.utils import plot_clusters
    >>> from gofast.datasets import fetch_data 
    >>> h= fetch_data('hlogs').frame 
    >>> # collect two features 'resistivity' and gamma-gamma logging values
    >>> h2 = h[['resistivity', 'gamma_gamma']] 
    >>> km = KMeans (n_clusters =3 , init= 'random' ) 
    >>> # scaled the data with MinMax scaler i.e. between ( 0-1) 
    >>> h2_scaled = MinMaxScaler().fit_transform(h2)
    >>> ykm = km.fit_predict(h2_scaled )
    >>> plot_clusters (3 , h2_scaled, ykm , km.cluster_centers_ )
        
    """
    n_clusters = int(
        _assert_all_types(n_clusters, int, float,  objname ="'n_clusters'" )
        )
    X, y_pred = check_X_y(
        X, 
        y_pred, 
        )

    if len(X.shape )!=2 or X.shape[1]==1: 
        ndim = 1 if X.shape[1] ==1 else np.ndim (X )
        raise ValueError(
            f"X is expected to be a two dimensional data. Got {ndim}!")
    # for consistency , convert y to array    
    y_pred = np.array(y_pred)
    
    colors = make_mpl_properties(n_clusters)
    markers = make_mpl_properties(n_clusters, 'markers')
    for n in range (n_clusters):
        plt.scatter (X[y_pred ==n, 0], 
                     X[y_pred ==n , 1],  
                     s= 50 , c= colors [n ], 
                     marker=markers [n], 
                     edgecolors=None if markers [n] =='x' else 'black', 
                     label = f'Cluster {n +1}'
                     ) 
    if cluster_centers is not None: 
        cluster_centers = np.array (cluster_centers)
        plt.scatter (cluster_centers[:, 0 ], 
                     cluster_centers [:, 1], 
                     s= 250. , marker ='*', 
                     c='red', edgecolors='black', 
                     label='centroids' 
                     ) 
    plt.legend (scatterpoints =1 ) 
    plt.grid() 
    plt.tight_layout() 
    
    if savefig is not None:
         savefigure(savefig, savefig )
    plt.close () if savefig is not None else plt.show() 
    
    
def plot_elbow (
        X,  n_clusters , n_init = 10 , max_iter = 300 , random_state=42 ,
        fig_size = (10, 4 ), marker = 'o', savefig= None, 
        **kwd): 
    """ Plot elbow method to find the optimal number of cluster, k', 
    for a given data. 
    
    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training instances to cluster. It must be noted that the data
        will be converted to C ordering, which will cause a memory
        copy if the given data is not C-contiguous.
        If a sparse matrix is passed, a copy will be made if it's not in
        CSR format.

    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.

    n_init : int, default=10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.

    verbose : int, default=0
        Verbosity mode.

    random_state : int, RandomState instance or None, default=42
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        
    savefig: str, default =None , 
        the path to save the figure. Argument is passed to 
        :class:`matplotlib.Figure` class. 
    marker: str, default='o', 
        cluster marker point. 
        
    kwd: dict
        Addionnal keywords arguments passed to :func:`matplotlib.pyplot.plot`
        
    Returns 
    --------
    ax: Matplotlib.pyplot axes objects 
    
    Example
    ---------
    >>> from gofast.datasets import load_hlogs 
    >>> from gofast.tools.utils import plot_elbow 
    >>> # get the only resistivy and gamma-gama values for example
    >>> res_gamma = load_hlogs ().frame[['resistivity', 'gamma_gamma']]  
    >>> plot_elbow(res_gamma, n_clusters=11)
    
    """
    distorsions =[] ; n_clusters = 11
    for i in range (1, n_clusters ): 
        km =KMeans (n_clusters =i , init= 'k-means++', 
                    n_init=n_init , max_iter=max_iter, 
                    random_state =random_state 
                    )
        km.fit(X) 
        distorsions.append(km.inertia_) 
            
    ax = _plot_elbow (distorsions, n_clusters =n_clusters,fig_size = fig_size ,
                      marker =marker , savefig =savefig, **kwd) 

    return ax 
    
def _plot_elbow (distorsions: list  , n_clusters:int ,fig_size = (10 , 4 ),  
               marker='o', savefig =None, **kwd): 
    """ Plot the optimal number of cluster, k', for a given class 
    
    :param distorsions: list - list of values withing the sum-squared-error 
        (SSE) also called  `inertia_` in sckit-learn. 
    
    :param n_clusters: number of clusters. where k starts and end. 
    
    :returns: ax: Matplotlib.pyplot axes objects 
    
    :Example: 
    >>> import numpy as np 
    >>> from sklearn.cluster import KMeans 
    >>> from gofast.datasets import load_iris 
    >>> from gofast.tools.utils import plot_elbow
    >>> d= load_iris ()
    >>> X= d.data [:, 0][:, np.newaxis] # take the first axis 
    >>> # compute distorsiosn for KMeans range 
    >>> distorsions =[] ; n_clusters = 11
    >>> for i in range (1, n_clusters ): 
            km =KMeans (n_clusters =i , 
                        init= 'k-means++', 
                        n_init=10 , 
                        max_iter=300, 
                        random_state =0 
                        )
            km.fit(X) 
            distorsions.append(km.inertia_) 
    >>> plot_elbow (distorsions, n_clusters =n_clusters)
        
    """
    fig, ax = plt.subplots ( nrows=1 , ncols =1 , figsize = fig_size ) 
    
    ax.plot (range (1, n_clusters), distorsions , marker = marker, 
              **kwd )
    plt.xlabel ("Number of clusters") 
    plt.ylabel ("Distorsion")
    plt.tight_layout()
    
    if savefig is not None: 
        savefigure(fig, savefig )
    plt.show() if savefig is None else plt.close () 
    
    return ax 


def plot_cost_vs_epochs(regs, *,  fig_size = (10 , 4 ), marker ='o', 
                     savefig =None, **kws): 
    """ Plot the cost against the number of epochs  for the two different 
    learnings rates 
    
    Parameters 
    ----------
    regs: Callable, single or list of regression estimators 
        Estimator should be already fitted.
    fig_size: tuple , default is (10, 4)
        the size of figure 
    kws: dict , 
        Additionnal keywords arguments passes to :func:`matplotlib.pyplot.plot`
    Returns 
    ------- 
    ax: Matplotlib.pyplot axes objects 
    
    Examples 
    ---------

    >>> from gofast.datasets import load_iris 
    >>> from gofast.base import AdalineGradientDescent
    >>> from gofast.tools.utils import plot_cost_vs_epochs
    >>> X, y = load_iris (return_X_y= True )
    >>> ada1 = AdalineGradientDescent (n_iter= 10 , eta= .01 ).fit(X, y) 
    >>> ada2 = AdalineGradientDescent (n_iter=10 , eta =.0001 ).fit(X, y)
    >>> plot_cost_vs_epochs (regs = [ada1, ada2] ) 
    """
    if not isinstance (regs, (list, tuple, np.array)): 
        regs =[regs]
    s = set ([hasattr(o, '__class__') for o in regs ])

    if len(s) != 1: 
        raise ValueError("All regression models should be estimators"
                         " already fitted.")
    if not list(s) [0] : 
        raise TypeError(f"Needs an estimator, got {type(s[0]).__name__!r}")
    
    fig, ax = plt.subplots ( nrows=1 , ncols =len(regs) , figsize = fig_size ) 
    
    for k, m in enumerate (regs)  : 
        
        ax[k].plot(range(1, len(m.cost_)+ 1 ), np.log10 (m.cost_),
                   marker =marker, **kws)
        ax[k].set_xlabel ("Epochs") 
        ax[k].set_ylabel ("Log(sum-squared-error)")
        ax[k].set_title("%s -Learning rate %.4f" % (m.__class__.__name__, m.eta )) 
        
    if savefig is not None: 
        savefigure(fig, savefig )
    plt.show() if savefig is None else plt.close () 
    
    return ax 

def plot_mlxtend_heatmap (df, columns =None, savefig=None,  **kws): 
    """ Plot correlation matrix array  as a heat map 
    
    :param df: dataframe pandas  
    :param columns: list of features, 
        If given, only the dataframe with that features is considered. 
    :param kws: additional keyword arguments passed to 
        :func:`mlxtend.plotting.heatmap`
    :return: :func:`mlxtend.plotting.heatmap` axes object 
    
    :example: 
        
    >>> from gofast.datasets import load_hlogs 
    >>> from gofast.tools.utils import plot_mlxtend_heatmap
    >>> h=load_hlogs()
    >>> features = ['gamma_gamma', 'sp',
                'natural_gamma', 'resistivity']
    >>> plot_mlxtend_heatmap (h.frame , columns =features, cmap ='PuOr')
    """
    import_optional_dependency('mlxtend', extra=(
        "Can't plot heatmap using 'mlxtend' package."))
  
    from mlxtend.plotting import (  heatmap 
            ) 
    cm = np.corrcoef(df[columns]. values.T)
    ax= heatmap(cm, row_names = columns , column_names = columns, **kws )
    
    if savefig is not None:
         savefigure(savefig, savefig )
    plt.close () if savefig is not None else plt.show() 
    
    return ax 

def plot_mlxtend_matrix(df, columns =None, fig_size = (10 , 8 ),
                        alpha =.5, savefig=None  ):
    """ Visualize the pair wise correlation between the different features in  
    the dataset in one place. 
    
    :param df: dataframe pandas  
    :param columns: list of features, 
        If given, only the dataframe with that features is considered. 
    :param fig_size: tuple of int (width, heigh) 
        Size of the displayed figure 
    :param alpha: figure transparency, default is ``.5``. 
    
    :return: :func:`mlxtend.plotting.scatterplotmatrix` axes object 
    :example: 
    >>> from gofast.datasets import load_hlogs 
    >>> from gofast.tools.utils import plot_mlxtend_matrix
    >>> import pandas as pd 
    >>> import numpy as np 
    >>> h=load_hlogs()
    >>> features = ['gamma_gamma', 'natural_gamma', 'resistivity']
    >>> data = pd.DataFrame ( np.log10 (h.frame[features]), columns =features )
    >>> plot_mlxtend_matrix (data, columns =features)

    """
    import_optional_dependency("mlxtend", extra = (
        "Can't plot the scatter matrix using 'mlxtend' package.") 
                               )
    from mlxtend.plotting import scatterplotmatrix
                                       
    if isinstance (columns, str): 
        columns = [columns ] 
    try: 
        iter (columns)
    except : 
        raise TypeError(" Columns should be an iterable object, not"
                        f" {type (columns).__name__!r}")
    columns =list(columns)
    
    
    if columns is not None: 
        df =df[columns ] 
        
    ax = scatterplotmatrix (
        df[columns].values , figsize =fig_size,names =columns , alpha =alpha 
        )
    plt.tight_layout()

    if savefig is not None:
         savefigure(savefig, savefig )
    plt.close () if savefig is not None else plt.show() 
    
    return ax 

    
def savefigure (fig: object ,
             figname: str = None,
             ext:str ='.png',
             **skws ): 
    """ save figure from the given figure name  
    
    :param fig: Matplotlib figure object 
    :param figname: name of figure to output 
    :param ext: str - extension of the figure 
    :param skws: Matplotlib savefigure keywards additional keywords arguments 
    
    :return: Matplotlib savefigure objects. 
    
    """
    ext = '.' + str(ext).lower().strip().replace('.', '')

    if figname is None: 
        figname =  '_' + os.path.splitext(os.path.basename(__file__)) +\
            datetime.datetime.now().strftime('%m-%d-%Y %H:%M:%S') + ext
        warnings.warn("No name of figure is given. Figure should be renamed as "
                      f"{figname!r}")
        
    file, ex = os.path.splitext(figname)
    if ex in ('', None): 
        ex = ext 
        figname = os.path.join(file, f'{ext}')

    return  fig.savefig(figname, **skws)


def resetting_ticks ( get_xyticks,  number_of_ticks=None ): 
    """
    resetting xyticks  modulo , 100
    
    :param get_xyticks:  xyticks list  , use to ax.get_x|yticks()
    :type  get_xyticks: list 
    
    :param number_of_ticks:  maybe the number of ticks on x or y axis 
    :type number_of_ticks: int
    
    :returns: a new_list or ndarray 
    :rtype: list or array_like 
    """
    if not isinstance(get_xyticks, (list, np.ndarray) ): 
        warnings.warn (
            'Arguments get_xyticks must be a list'
            ' not <{0}>.'.format(type(get_xyticks)))
        raise TipError (
            '<{0}> found. "get_xyticks" must be a '
            'list or (nd.array,1).'.format(type(get_xyticks)))
    
    if number_of_ticks is None :
        if len(get_xyticks) > 2 : 
            number_of_ticks = int((len(get_xyticks)-1)/2)
        else : number_of_ticks  = len(get_xyticks)
    
    if not(number_of_ticks, (float, int)): 
        try : number_of_ticks=int(number_of_ticks) 
        except : 
            warnings.warn('"Number_of_ticks" arguments is the times to see '
                          'the ticks on x|y axis.'\
                          ' Must be integer not <{0}>.'.
                          format(type(number_of_ticks)))
            raise PlotError(f'<{type(number_of_ticks).__name__}> detected.'
                            ' Must be integer.')
        
    number_of_ticks=int(number_of_ticks)
    
    if len(get_xyticks) > 2 :
        if get_xyticks[1] %10 != 0 : 
            get_xyticks[1] =get_xyticks[1] + (10 - get_xyticks[1] %10)
        if get_xyticks[-2]%10  !=0 : 
            get_xyticks[-2] =get_xyticks[-2] -get_xyticks[-2] %10
    
        new_array = np.linspace(get_xyticks[1], get_xyticks[-2],
                                number_of_ticks )
    elif len(get_xyticks)< 2 : 
        new_array = np.array(get_xyticks)
 
    return  new_array
        
def make_mpl_properties(n ,prop ='color'): 
    """ make matplotlib property ('colors', 'marker', 'line') to fit the 
    numer of samples
    
    :param n: int, 
        Number of property that is needed to create. It generates a group of 
        property items. 
    :param prop: str, default='color', name of property to retrieve. Accepts 
        only 'colors', 'marker' or 'line'.
    :return: list of property items with size equals to `n`.
    :Example: 
        >>> from gofast.tools.utils import make_mpl_properties
        >>> make_mpl_properties (10 )
        ... ['g',
             'gray',
             'y',
             'blue',
             'orange',
             'purple',
             'lime',
             'k',
             'cyan',
             (0.6, 0.6, 0.6)]
        >>> make_mpl_properties(100 , prop = 'marker')
        ... ['o',
             '^',
             'x',
             'D',
              .
              .
              .
             11,
             'None',
             None,
             ' ',
             '']
        >>> make_mpl_properties(50 , prop = 'line')
        ... ['-',
             '-',
             '--',
             '-.',
               .
               .
               . 
             'solid',
             'dashed',
             'dashdot',
             'dotted']
        
    """ 
    n=int(_assert_all_types(n, int, float, objname ="'n'"))
    prop = str(prop).lower().strip().replace ('s', '') 
    if prop not in ('color', 'marker', 'line'): 
        raise ValueError ("Property {prop!r} is not availabe yet. , Expect"
                          " 'colors', 'marker' or 'line'.")
    # customize plots with colors lines and styles 
    # and create figure obj
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
       
def resetting_colorbar_bound(cbmax ,
                             cbmin,
                             number_of_ticks = 5, 
                             logscale=False): 
    """
    Function to reset colorbar ticks more easy to read 
    
    :param cbmax: value maximum of colorbar 
    :type cbmax: float 
    
    :param cbmin: minimum data value 
    :type cbmin: float  minimum data value
    
    :param number_of_ticks:  number of ticks should be 
                            located on the color bar . Default is 5.
    :type number_of_ticks: int 
    
    :param logscale: set to True if your data are lograith data . 
    :type logscale: bool 
    
    :returns: array of color bar ticks value.
    :rtype: array_like 
    """
    def round_modulo10(value): 
        """
        round to modulo 10 or logarithm scale  , 
        """
        if value %mod10  == 0 : return value 
        if value %mod10  !=0 : 
            if value %(mod10 /2) ==0 : return value 
            else : return (value - value %mod10 )
    
    if not(number_of_ticks, (float, int)): 
        try : number_of_ticks=int(number_of_ticks) 
        except : 
            warnings.warn('"Number_of_ticks" arguments '
                          'is the times to see the ticks on x|y axis.'
                          ' Must be integer not <{0}>.'.format(
                              type(number_of_ticks)))
            raise TipError('<{0}> detected. Must be integer.')
        
    number_of_ticks=int(number_of_ticks)
    
    if logscale is True :  mod10 =np.log10(10)
    else :mod10 = 10 
       
    if cbmax % cbmin == 0 : 
        return np.linspace(cbmin, cbmax , number_of_ticks)
    elif cbmax% cbmin != 0 :
        startpoint = cbmin + (mod10  - cbmin % mod10 )
        endpoint = cbmax - cbmax % mod10  
        return np.array(
            [round_modulo10(ii) for ii in np.linspace(
                             startpoint,endpoint, number_of_ticks)]
            )
    

            
def controle_delineate_curve(res_deline =None , phase_deline =None ): 
    """
    fonction to controle delineate value given  and return value ceilling .
    
    :param  res_deline:  resistivity  value todelineate. unit of Res in `ohm.m`
    :type  res_deline: float|int|list  
    
    :param  phase_deline:   phase value to  delineate , unit of phase in degree
    :type phase_deline: float|int|list  
    
    :returns: delineate resistivity or phase values 
    :rtype: array_like 
    """
    fmt=['resistivity, phase']
 
    for ii, xx_deline in enumerate([res_deline , phase_deline]): 
        if xx_deline is  not None  : 
            if isinstance(xx_deline, (float, int, str)):
                try :xx_deline= float(xx_deline)
                except : raise TipError(
                        'Value <{0}> to delineate <{1}> is unacceptable.'\
                         ' Please ckeck your value.'.format(xx_deline, fmt[ii]))
                else :
                    if ii ==0 : return [np.ceil(np.log10(xx_deline))]
                    if ii ==1 : return [np.ceil(xx_deline)]
  
            if isinstance(xx_deline , (list, tuple, np.ndarray)):
                xx_deline =list(xx_deline)
                try :
                    if ii == 0 : xx_deline = [
                            np.ceil(np.log10(float(xx))) for xx in xx_deline]
                    elif  ii ==1 : xx_deline = [
                            np.ceil(float(xx)) for xx in xx_deline]
                        
                except : raise TipError(
                        'Value to delineate <{0}> is unacceptable.'\
                         ' Please ckeck your value.'.format(fmt[ii]))
                else : return xx_deline


def fmt_text (data_text, fmt='~', leftspace = 3, return_to_line =77) : 
    """
    Allow to format report with data text , fm and leftspace 

    :param  data_text: a long text 
    :type  data_text: str  
        
    :param fmt:  type of underline text 
    :type fmt: str

    :param leftspae: How many space do you want before starting wrinting report .
    :type leftspae: int 
    
    :param return_to_line: number of character to return to line
    :type return_to_line: int 
    """

    return_to_line= int(return_to_line)
    begin_text= leftspace *' '
    text= begin_text + fmt*(return_to_line +7) + '\n'+ begin_text

    
    ss=0
    
    for  ii, num in enumerate(data_text) : # loop the text 
        if ii == len(data_text)-1 :          # if find the last character of text 
            #text = text + data_text[ss:] + ' {0}\n'.format(fmt) # take the 
            #remain and add return chariot 
            text = text+ ' {0}\n'.format(fmt) +\
                begin_text +fmt*(return_to_line+7) +'\n' 
      
 
            break 
        if ss == return_to_line :                       
            if data_text[ii+1] !=' ' : 
                text = '{0} {1}- \n {2} '.format(
                    text, fmt, begin_text + fmt ) 
            else : 
                text ='{0} {1} \n {2} '.format(
                    text, fmt, begin_text+fmt ) 
            ss=0
        text += num    # add charatecter  
        ss +=1

    return text 

def plotvec1(u, z, v):
    """
    Plot tips function with  three vectors. 
    
    :param u: vector u - a vector 
    :type u: array like  
    
    :param z: vector z 
    :type z: array_like 
    
    :param v: vector v 
    :type v: array_like 
    
    return: plot 
    
    """
    
    ax = plt.axes()
    ax.arrow(0, 0, *u, head_width=0.05, color='r', head_length=0.1)
    plt.text(*(u + 0.1), 'u')
    
    ax.arrow(0, 0, *v, head_width=0.05, color='b', head_length=0.1)
    plt.text(*(v + 0.1), 'v')
    ax.arrow(0, 0, *z, head_width=0.05, head_length=0.1)
    plt.text(*(z + 0.1), 'z')
    plt.ylim(-2, 2)
    plt.xlim(-2, 2)

def plotvec2(a,b):
    """
    Plot tips function with two vectors
    Just use to get the orthogonality of two vector for other purposes 

    :param a: vector u 
    :type a: array like  - a vector 
    :param b: vector z 
    :type b: array_like 
    
    *  Write your code below and press Shift+Enter to execute
    
    :Example: 
        
        >>> import numpy as np 
        >>> from gofast.tools.utils import plotvec2
        >>> a=np.array([1,0])
        >>> b=np.array([0,1])
        >>> Plotvec2(a,b)
        >>> print('the product a to b is =', np.dot(a,b))

    """
    ax = plt.axes()
    ax.arrow(0, 0, *a, head_width=0.05, color ='r', head_length=0.1)
    plt.text(*(a + 0.1), 'a')
    ax.arrow(0, 0, *b, head_width=0.05, color ='b', head_length=0.1)
    plt.text(*(b + 0.1), 'b')
    plt.ylim(-2, 2)
    plt.xlim(-2, 2)  

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
        
        >>> from gofast.tools.utils import get_color_palette 
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

def _is_target_in (X, y=None, tname=None): 
    """ Create new target name for tname if given 
    
    :param X: dataframe 
        dataframe containing the data for plotting 
    :param y: array or series
        target data for plotting. Note that multitarget outpout is not 
        allowed yet. Moroever, it `y` is given as a dataframe, 'tname' must 
        be supplied to retrive y as a pandas series object, otherwise an 
        error will raise. 
    :param tname: str,  
        target name. If given and `y` is ``None``, Will try to find `tname`
        in the `X` columns. If 'tname' does not exist, plot for target is 
        cancelled. 
        
    :return y: Series 
    """
    _assert_all_types(X, pd.DataFrame)
    
    if y is not None: 
        y = _assert_all_types(y , pd.Series, pd.DataFrame, np.ndarray)
        
        if hasattr (y, 'columns'): 
            if tname not in (y.columns): tname = None 
            if tname is None: 
                raise TypeError (
                    "'tname' must be supplied when y is a dataframe.")
            y = y [tname ]
        elif hasattr (y, 'name'): 
            tname = tname or y.name 
            # reformat inplace the name of series 
            y.name = tname 
            
        elif hasattr(y, '__array__'): 
            y = pd.Series (y, name = tname or 'target')
            
    elif y is None: 
        if tname in X.columns :
            y = X.pop(tname)

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
       >>> from gofast.tools.utils import _skip_log10_columns 
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

def plot_voronoi(
    X, y, *, 
    cluster_centers,
    ax= None,
    show_vertices=False, 
    line_colors='k',
    line_width=1. ,
    line_alpha=1.,   
    fig_size = (7, 7), 
    fig_title = ''
    ):
    """Plots the Voronoi diagram of the k-means clusters overlaid with 
    the data
    
    Parameters 
    -----------
    X, y : NDarray, Arraylike 1d 
      Data training X and y. Must have the same length 
    cluster_center: int, 
       Cluster center. Cluster center can be obtain withe KMeans algorithms 
    show_vertices : bool, optional
        Add the Voronoi vertices to the plot.
    line_colors : string, optional
        Specifies the line color for polygon boundaries
    line_width : float, optional
        Specifies the line width for polygon boundaries
    line_alpha : float, optional
        Specifies the line alpha for polygon boundaries
    point_size : float, optional
        Specifies the size of points
    ax: Matplotlib.Axes 
       Maplotlib axes. If `None`, a axis is created instead. 
       
    fig_size: tuple, default = (7, 7) 
       Size of the figures. 
       
    Return
    -------
    ax: Matplotlib.Axes 
       Axes to support the figure
       
    Examples 
    ---------
    >>> from sklearn.datasets import make_moons
    >>> from sklearn.cluster import KMeans 
    >>> from gofast.tools.utils import plot_voronoi
    >>> X, y = make_moons(n_samples=2000, noise=0.2)
    >>> km = KMeans (n_init ='auto').fit(X, y ) 
    >>> plot_voronoi ( X, y , cluster_centers = km.cluster_centers_) 
    """
    X, y = check_X_y(X, y, )
    cluster_centers = check_array(cluster_centers )
    
    if ax is None: 
        fig, ax = plt.subplots(1,1, figsize =fig_size)
        
    from scipy.spatial import Voronoi, voronoi_plot_2d
    
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='Set1', alpha=0.2, 
               label = 'Voronoi plot')
    vor = Voronoi(cluster_centers)
    voronoi_plot_2d(vor, ax=ax, show_vertices=show_vertices, 
                    alpha=0.5, 
                    line_colors=line_colors,
                    line_width=line_width ,
                    line_alpha=line_alpha,  
                    )
    #ax.legend() 
    ax.set_title (fig_title , fontsize=20)
    #fig.suptitle(fig_title, fontsize=20) 
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
    
def plot_roc_curves (
   clfs, /, 
   X, y, 
   names =..., 
   colors =..., 
   ncols = 3, 
   score=False, 
   kind="inone",
   ax = None,  
   fig_size=( 7, 7), 
   **roc_kws ): 
    """ Quick plot of Receiving Operating Characterisctic (ROC) of fitted models 
    
    Parameters 
    ------------
    clfs: list, 
       list of models for ROC evaluation. Model should be a scikit-learn 
       or  XGBoost estimators 
       
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training instances to cluster. It must be noted that the data
        will be converted to C ordering, which will cause a memory
        copy if the given data is not C-contiguous.
        If a sparse matrix is passed, a copy will be made if it's not in
        CSR format.
    
    y : ndarray or Series of length (n_samples, )
        An array or series of target or class values. Preferably, the array 
        represent the test class labels data for error evaluation.  
        
    names: list, 
       List of model names. If not given, a raw name of the model is passed 
       instead.
     
    kind: str, default='inone' 
       If ``['individual'|'2'|'single']``, plot each ROC model separately. 
       Any other value, group of ROC curves into a single plot. 
       
       .. versionchanged:: 0.2.5 
          Parameter `all` is deprecated and replaced by `kind`. It henceforth 
          accepts arguments ``allinone|1|grouped`` or ``individual|2|single``
          for plotting mutliple ROC curves in one or separate each ROC curves 
          respecively. 
          
    colors : str, list 
       Colors to specify each model plot. 
       
    ncols: int, default=3 
       Number of plot to be placed inline before skipping to the next column. 
       This is feasible if `many` is set to ``True``. 
       
    score: bool,default=False
      Append the Area Under the curve score to the legend. 
      
      .. versionadded:: 0.2.4 
      
    kws: dict,
        keyword argument of :func:`sklearn.metrics.roc_curve 
        
    Return
    -------
    ax: Axes.Subplot. 
    
    Examples 
    --------
    >>> from gofast.tools.utils import plot_roc_curves 
    >>> from sklearn.datasets import make_moons 
    >>> from gofast.exlib import ( train_test_split, KNeighborsClassifier, SVC ,
    XGBClassifier, LogisticRegression ) 
    >>> X, y = make_moons (n_samples=2000, noise=0.2)
    >>> X, Xt, y, yt = train_test_split (X, y, test_size=0.2) 
    >>> clfs = [ m().fit(X, y) for m in ( KNeighborsClassifier, SVC , 
                                         XGBClassifier, LogisticRegression)]
    >>> plot_roc_curves(clfs, Xt, yt)
    Out[66]: <AxesSubplot:xlabel='False Positive Rate (FPR)', ylabel='True Positive Rate (FPR)'>
    >>> plot_roc_curves(clfs, Xt, yt,kind='2', ncols = 4 , fig_size = (10, 4))
    """
    from .validator import  get_estimator_name
    
    kind = '2' if str(kind).lower() in 'individual2single' else '1'

    def plot_roc(model, data, labels, score =False ):
        if hasattr(model, "decision_function"):
            predictions = model.decision_function(data)
        else:
            predictions = model.predict_proba(data)[:,1]
            
        fpr, tpr, _ = roc_curve(labels, predictions, **roc_kws )
        auc_score = None 
        if score: 
            auc_score = roc_auc_score ( labels, predictions,)
            
        return fpr, tpr , auc_score
    
    if not is_iterable ( clfs): 
       clfs = is_iterable ( clfs, exclude_string =True , transform =True ) 
       
    # make default_colors 
    colors = make_plot_colors(clfs, colors = colors )
    # save the name of models 
    names = make_obj_consistent_if (
        names , [ get_estimator_name(m) for m in clfs ]) 

    # check whether the model is fitted 
    if kind=='2': 
        fig, ax = _make_axe_multiple ( 
            clfs, ncols = ncols , ax = ax, fig_size = fig_size 
                                  ) 
    else: 
        if ax is None: 
            fig, ax = plt.subplots (1, 1, figsize = fig_size )  
    
    for k, ( model, name)  in enumerate (zip (clfs, names )): 
        check_is_fitted(model )
        fpr, tpr, auc_score = plot_roc(model, X, y, score)

        if hasattr (ax, '__len__'): 
            if len(ax.shape)>1: 
                i, j  =  k // ncols , k % ncols 
                axe = ax [i, j]
            else: axe = ax[k]
        else: axe = ax 

        axe.plot(fpr, tpr, label=name + ('' if auc_score is None 
                                         else f"AUC={round(auc_score, 3) }") , 
                 color = colors[k]  )
        
        if kind=='2': 
            axe.plot([0, 1], [0, 1], 'k--') 
            axe.legend ()
            axe.set_xlabel ("False Positive Rate (FPR)")
            axe.set_ylabel ("True Positive Rate (FPR)")
        # else: 
        #     ax.plot(fpr, tpr, label=name, color = colors[k])
            
    if kind!='2': 
        ax.plot([0, 1], [0, 1], 'k--') # AUC =.5 
        ax.set_xlabel ("False Positive Rate (FPR)")
        ax.set_ylabel ("True Positive Rate (FPR)")
        ax.legend() 
        
    return ax 
        
def plot_rsquared (X , y,  y_pred, **r2_score_kws  ): 
    """ Plot :math:`R^2` squared functions. 
    
    Parameters 
    -----------
    X : array-like of shape (n_samples, n_features)
        Training vector, where `n_samples` is the number of samples and
        `n_features` is the number of features.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Target relative to X for classification or regression;
        None for unsupervised learning.
    
    y_pred: array-like of shape (n_samples,) or (n_samples, n_outputs)
        Predicted target relative to X for classification or regression;
        None for unsupervised learning.
        
    r2_score_kws: dict, optional 
       Additional keyword arguments of :func:`sklearn.metrics.r2_score`. 
    
    """
    from sklearn.metrics import r2_score
    # Calculate R-squared
    r_squared = r2_score(y, y_pred, **r2_score_kws)

    # Plotting the scatter plot
    plt.scatter(X, y, color='blue', label='Actual data')

    # Plotting the regression line
    plt.plot(X, y_pred, color='red', linewidth=2, label='Fitted line')

    # Annotate the R-squared value on the plot
    plt.text(0.5, 0.5, 'R-squared = {:.2f}'.format(r_squared), fontsize=12, ha='center')

    # Adding labels and title
    plt.xlabel('Predictor')
    plt.ylabel('Target')
    plt.title('R-squared Diagram')
    plt.legend()
    # Show the plot
    plt.show()


def plot_l_curve(
    rms, 
    roughness, 
    tau=None, 
    hansen_point=None, 
    rms_target=None,
    view_tline=False,
    hpoint_kws=dict(), 
    fig_size = (10, 4),
    ax=None,
    fig=None, 
    style = 'classic', 
    savefig=None, 
    **plot_kws
    ):
    """
    Plot the Hansen L-curve.
    
    The L-curve criteria is used to determine the suitable model 
    after runing multiple inversions with different :math:`\tau` values. 
    The function plots RMS vs. Roughness with an option to highlight a 
    specific point named Hansen point [1]_.
    
    The :math:`\tau` represents the measure of compromise between data fit and 
    model smoothness. To find out an appropriates-value, the inversion was 
    carried out with differents-values. The RMS error obtained from each 
    inversion is plotted against model roughnes
    
    Plots RMS vs. Roughness with an option to highlight the Hansen point.
    
    Parameters 
    ------------
    
    rms: ArrayLike, list, 
       Corresponding list pr Arraylike of RMS values.
       
    roughness: Arraylike, list, 
       List or ArratLike of roughness values. 
       
    tau: Arraylike or list, optional 
       List of tau values to visualize as text mark in the plot. 

    hansen_point: A tuple (roughness_value, RMS_value) , optional 
       The Hansen point to visualize in the plot. It can be determine 
       automatically if ``highlight_point='auto'``.
       
    rms_target: float, optional 
      The root-mean-squared target. If set, and `view_tline` is ``False``, 
      the target value should be axis limit. 
      
     view_tline: bool, default=False 
       Display the target line should be  displayed.
       
    hpoint_kws: dict, optional 
      Keyword argument to highlight the hansen point in the figure. 
     
    ax: Matplotlib.pyplot.Axes, optional 
       Axe to collect the figure. Could be used to support other axes. 
       
    fig: Matplotlib.pyplot.figure, optional 
        Supply fig to save automatically the plot, otherwise, keep it 
        to ``None``.

    savefig: str, optional 
        Save figure name. The default resolution dot-per-inch is ``300``. 
         
    Return 
    ------
    ax: Matplotlib.pyplot.Axis 
        Return axis  
        
    References
    -----------
    [1] Hansen, P. C., & O'Leary, D. P. (1993). The use of the L-Curve in
        the regularization of discrete ill-posed problems. SIAM Journal
        on Scientific Computing, 14(6), 1487–1503. https://doi.org/10.1137/0914086.
         
    Examples
    ---------
    >>> from gofast.tools.utils import plot_l_curve
    >>> # Test the function with the provided data points and 
    >>> # highlighting point (50, 3.12)
    >>> roughness_data = [0, 50, 100, 150, 200, 250, 300, 350]
    >>> RMS_data = [3.16, 3.12, 3.1, 3.08, 3.06, 3.04, 3.02, 3]
    >>> highlight_data = (50, 3.12)
    >>> plot_l_curve(roughness_data, RMS_data, highlight_data)
    """
    
    rms= np.array (
        is_iterable(rms, exclude_string= True, transform =True ), 
                   dtype =float) 
    roughness = np.array( 
        is_iterable(roughness , exclude_string= True, transform =True 
                            ), dtype = float) 
    
    # Create the plot
    plt.style.use (style )
    
    if ax is None: 
        fig, ax = plt.subplots(1,1, figsize =fig_size)
    
    # manage the plot keyword argument and remove 
    # the default is given.
    plot_kws = _manage_plot_kws (plot_kws, dict(
        marker='o',linestyle='-', color='black')
                    )
    ax.plot(roughness, rms, **plot_kws
             )
    # Highlight the specific hansen point if "auto" 
    # option is provided.
    if str(hansen_point).lower().strip() =="auto": 
        hansen_point = _get_hansen_point(roughness, rms)
        
    if hansen_point is not None:
        if len(hansen_point)!=2: 
            raise ValueError("Hansen knee point needs a tuple of (x, y)."
                             f" Got {hansen_point}")
        hpoint_kws = _manage_plot_kws(hpoint_kws, 
                                         dict(marker='o', color='red'))
        ax.plot(hansen_point[0], hansen_point[1], **hpoint_kws)
        ax.annotate(str(hansen_point[0]), 
                     hansen_point, textcoords="offset points",
                     xytext=(0,10), ha='center'
                     )
    if tau is not None: 
        tau = is_iterable(tau, exclude_string= True, transform =True )
        rough_rms = np.hstack ((roughness, rms))
        for tvalues, text in zip ( rough_rms, tau): 
            if ( 
                    tvalues[0]==hansen_point[0] 
                    and tvalues[1]==hansen_point[1]
                    ): 
                # hansen point is found then skip it
                continue 
            ax.annotate(str(text), tvalues, textcoords="offset points",
                         xytext=(0,10), ha='center'
                         )
    if rms_target: 
        rms_target = float( _assert_all_types(
            rms_target, float, int, objname='RMS target')) 
        
    if view_tline: 
        if rms_target is None: 
            warnings.warn("Missing RMS target value. Could not display"
                          " the target line.")
        else:
            ax.axhline(y=float(rms_target), color='k', linestyle=':')

    if rms_target: 
        # get the extent value from the min 
        extent_value = abs ( rms_target - min(rms )) 
        ax.set_ylim ( [rms_target - extent_value  if view_tline else 0  ,
                       max(rms)+ extent_value]) # .3 extension limit

    # Setting the labels and title
    ax.set_xlabel('Roughness')
    ax.set_ylabel('RMS')
    ax.set_title('RMS vs. Roughness')

    # savefig 
    if savefig is not  None: savefigure (fig, savefig, dpi = 300)
    # Show the plot    
    plt.close () if savefig is not None else plt.show() 
    
    return ax 

def _get_hansen_point ( roughness, RMS): 
    """ Get the Hansen point automatically.
    
    An isolated part of :func:`~plot_l_curve`. 
    
    The L-curve criteria proposed by Hansen and O'Leary (1993)[1]_ and 
    Hansen (1998) [2]_, which suggests that the s value at the knee of 
    the curve is most appropriate, have been adopted.

    References
    -----------
    [1] Hansen, P. C., & O'Leary, D. P. (1993). The use of the L-Curve in
        the regularization of discrete ill-posed problems. SIAM Journal
        on Scientific Computing, 14(6), 1487–1503. https://doi.org/10.1137/0914086.
        
    [2] Hansen, P. C. (1998). Rank deficient and discrete Ill: Posed problems, 
        numerical aspects of linear inversion (p. 247). Philadelphia: SIAM
    """
    # Calculate the curvature of the plot
    # Using a simple method to estimate the 'corner' of the L-curve
    curvature = []
    for i in range(1, len(roughness) - 1):
        # Triangle area method to calculate curvature
        x1, y1 = roughness[i-1], RMS[i-1]
        x2, y2 = roughness[i], RMS[i]
        x3, y3 = roughness[i+1], RMS[i+1]

        # Lengths of the sides of the triangle
        a = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        b = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)
        c = np.sqrt((x3 - x1)**2 + (y3 - y1)**2)

        # Semiperimeter of the triangle
        s = (a + b + c) / 2

        # Area of the triangle
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        # Curvature is 4 * area divided by product of the sides
        # (Heron's formula for the area of a triangle)
        if a * b * c == 0:
            k = 0
        else:
            k = 4 * area / (a * b * c)
        curvature.append(k)

    # Find the index of the point with the maximum curvature
    # +1 due to curvature array being shorter
    max_curvature_index = np.argmax(curvature) + 1  
    return roughness[max_curvature_index], RMS[max_curvature_index]

def _manage_plot_kws ( kws, dkws = dict () ): 
    """ Check whether the default values are in plot_kws then pop it"""
    
    kws = dkws or kws 
    for key in dkws.keys(): 
        # if key not in then add it. 
        if key not in kws.keys(): 
            kws[key] = dkws.get(key)
            
    return kws 



    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
  
    
    
    
    
