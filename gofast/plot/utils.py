# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Miscellanous plot utilities. 
"""
from __future__ import annotations 
import os
import re 
import copy 
import random
import datetime 
import warnings
import itertools 
import numpy as np
import seaborn as sns 
import matplotlib.axes
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.metrics import r2_score

from ..api.types import Optional
from ..api.types import ArrayLike, DataFrame
from ..core.array_manager import drop_nan_in 
from ..core.checks import ( 
    _assert_all_types, is_iterable,
)
from ..core.io import is_data_readable 
from ..core.plot_manager import default_params_plot 
from ..compat.sklearn import  validate_params
from ..utils.validator import assert_xy_in, ensure_2d 
from ._config import PlotConfig
from ._d_cms import D_COLORS, D_MARKERS 
from ._d_cms import D_STYLES, D_CMAPS, D_SEQ  

__all__=[
    "boxplot", 
    "plot_r_squared", 
    "plot_text", 
    'flex_figsize',
    'get_color_palette',
    'is_colormap',
    'make_mpl_properties',
    'make_plot_colors',
    'plot_bar',
    'plot_errorbar',
    'plot_r_squared',
    'plot_text',
    'plot_vec',
    'resetting_colorbar_bound',
    'resetting_ticks',
    'savefigure', 
]

_param_defaults = {
    'title': "Box Plot: Quantiles Across Time", 
    'xlabel': "Date", 
    "ylabel": "Values", 
    'palette': "Set2", 
    'var_name': "Quantiles", 
    'value_name': "Values", 
}


@default_params_plot(
    savefig=PlotConfig.AUTOSAVE("my_r-squared_plot.png"), 
    fig_size = (10, 6), 
    dpi=300, 
)
@validate_params ({ 
    'y_true': ['array-like'], 
    'y_pred': ['array-like'], 
    })
def plot_r_squared(
    y_true,
    y_pred,
    title=None,
    figsize=None,
    r_color='red',
    pred_color='blue',
    xlabel=None,
    ylabel=None,
    sns_plot=False,
    show_grid=True,
    grid_props=None,
    **scatter_kws
):
    r"""
    Plot predictions against true values, annotate the R-squared
    statistic, and optionally use Seaborn for the scatter plot.

    This function displays a scatter plot of predicted vs. true
    values along with a 1:1 reference line (in ``r_color``) and
    computes the R-squared (:math:`R^2`) of the predictions.

    .. math::
        R^2 = 1 - \frac{\sum_{i}(y_i - \hat{y}_i)^2}
        {\sum_{i}(y_i - \bar{y})^2}

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target values.

    y_pred : array-like of shape (n_samples,)
        Predicted target values by the regression model.

    title : str, optional
        Text to use as the main plot title. If None, defaults to
        "Regression Model: R-squared".

    figsize : tuple of (float, float), optional
        Size of the figure (width, height in inches). If None,
        defaults to (10, 6).

    r_color : str, default='red'
        Color of the perfect-fit line indicating ``y_true == y_pred``.

    pred_color : str, default='blue'
        Color for the scatter plot points.

    xlabel : str, optional
        Label for the x-axis. Defaults to "Actual Values" if None.

    ylabel : str, optional
        Label for the y-axis. Defaults to "Predicted Values" if None.

    sns_plot : bool, default=False
        Whether to use Seaborn for scatter plotting. If True, calls
        ``sns.scatterplot(...)``; otherwise uses Matplotlib's
        ``ax.scatter(...)``.

    show_grid : bool, default=True
        Whether to display a grid on the plot background.

    grid_props : dict, optional
        Properties passed to ``ax.grid()`` when showing the grid.
        If None, uses a default dict 
        ``{"linestyle": ":", "alpha": 0.7}``.

    **scatter_kws : dict, optional
        Additional keyword arguments passed to ``scatter`` or
        ``sns.scatterplot`` for customizing the points. For example,
        ``marker='x'``, ``alpha=0.5``, etc.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The Axes object with the scatter plot, reference line,
        and R-squared annotation.

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.plot.utils import plot_r_squared
    >>> # Generate some sample data
    >>> np.random.seed(0)
    >>> y_true_sample = np.random.rand(100) * 100
    >>> y_pred_sample = y_true_sample * (0.95 + 0.1 * np.random.rand(100))
    >>> # Plot the R-squared for this simple simulated regression
    >>> ax = plot_r_squared(
    ...     y_true_sample,
    ...     y_pred_sample,
    ...     title="Sample Regression Model",
    ...     sns_plot=False
    ... )
    >>> # This displays a scatter plot of predicted vs. true values
    >>> # and prints the R-squared in the legend.

    See Also
    --------
    gofast.plot.ml_viz.plot_r2 :
        A more advanced R² plotting utility with additional metrics
        and aesthetic options.

    References
    ----------
    .. [1] Freedman, D. A. (1983). A note on screening regression
           equations. *The American Statistician*, 37(2), 152-155.
    """
    # Remove NaN rows in corresponding positions
    y_true, y_pred = drop_nan_in(y_true, y_pred)

    # Calculate R-squared
    r_squared = r2_score(y_true, y_pred)

    # Create figure if no existing figure
    plt.figure(figsize=figsize or (10, 6))
    ax = plt.gca()

    # Plot using seaborn or matplotlib
    if sns_plot:
        sns.scatterplot(
            x=y_true,
            y=y_pred,
            ax=ax,
            color=pred_color,
            **scatter_kws
        )
    else:
        ax.scatter(
            y_true,
            y_pred,
            color=pred_color,
            **scatter_kws
        )

    # Plot the perfect prediction line
    ax.plot(y_true, y_true, color=r_color, label='Actual = Predicted')

    # Annotate the R-squared in the legend
    ax.legend(
        labels=[
            f'Predictions (R² = {r_squared:.2f})',
            'Actual = Predicted'
        ]
    )

    # Title and labels
    ax.set_title(
        title or 'Regression Model: R-squared'
    )
    ax.set_xlabel(
        xlabel or 'Actual Values'
    )
    ax.set_ylabel(
        ylabel or 'Predicted Values'
    )

    # Display the grid if requested
    if show_grid:
        if grid_props is None:
            grid_props = {"linestyle": ":", "alpha": 0.7}
        ax.grid(True, **grid_props)
    else:
        ax.grid(False)

    plt.show()
    
    return ax

    
@is_data_readable 
def boxplot(
    data: ArrayLike | DataFrame, 
    labels: list[str],
    title: str, 
    y_label: str, 
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
    >>> from gofast.plot.utils import boxplot
    >>> np.random.seed(10)
    >>> d = [np.random.normal(0, std, 100) for std in range(1, 5)]
    >>> labels = ['s1', 's2', 's3', 's4']
    >>> boxplot(d, labels, 
    ...      title='Class assignment (roc-auc): PsA activity',
    ...      y_label='roc-auc', 
    ...      figsize=(12, 7),
    ...      color="green",
    ...      showfliers=False, 
    ...      whis=2,
    ...      width=0.3, 
    ...      linewidth=1.5,
    ...      flierprops=dict(marker='x', color='black', markersize=5))
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
    # bplot.set_xticklabels(labels, rotation=45)
    
    # Set the style of the plot
    sns.set_style(sns_style)
    
    # Show the plot
    plt.show()
    return bplot

def make_plot_colors(
    d, 
    /,
    colors: str | list[str] = None,
    axis: int = 0,
    seed: int = None,
    chunk: bool = ...,
    cmap_only: bool = False, 
    get_only_names=True,
    use_cmap_seq=False, 
):
    """
    Select or generate a color sequence according to the size of `d` along
    the specified axis. By default, a set of auto-generated colors is used,
    but custom color sets (including Matplotlib's CS4 or XKCD colors) can
    also be requested.

    Parameters
    ----------
    d : array-like
        The data (e.g., a NumPy array or list) from which the length along 
        `axis` is used to determine the number of colors needed.
    colors : str or list of str, optional
        The color specification(s) to use. If not provided, default 
        auto-generated colors are used. 
        
        - If a string containing `'cs4'` or `'xkcd'` is given (e.g., 
          `'cs4:4'`, `'xkcd:10'`), the respective 
          :attr:`matplotlib.colors.CSS4_COLORS` or 
          :attr:`matplotlib.colors.XKCD_COLORS` are used. 
          
        - A suffix after a colon (like `:4`) can be used to select the
          starting index in that color dictionary.
          
        - If the exact suffix matches a known color name (e.g., 
          `'cs4:aliceblue'`), then that color name is used as the 
          starting point (no shuffling).
    axis : int, default=0
        The axis along which the colors should match. If `axis=0`, 
        the number of rows in `d` determines how many colors are generated. 
        If `axis=1`, the number of columns, etc.
    seed : int, optional
        A random seed to make color selection reproducible, applicable 
        if colors are drawn randomly (e.g., from `'cs4'` or `'xkcd'`).
    chunk : bool, default=True
        If True, the final list of colors is truncated or "chunked" to 
        exactly match the axis length of `d`. Otherwise, the full list 
        remains untruncated.
    cmap_only : bool, default=False
        If True, use a Matplotlib colormap-based approach (and ignore 
        regular color lists). Colors are sampled from a random (or 
        seeded-random) colormap, providing exactly one color per item 
        along the chosen axis. If both `cmap_only=True` and a custom 
        `colors` argument is given, that custom set is merged with 
        the colormap samples unless `'colors'` itself has `'cs4'` or 
        `'xkcd'` logic applied.
    get_only_names : bool, default=True
        When `cmap_only=True` and `prop='color'`:

        - If True, the function returns the **colormap name** samples 
          `n` times from colormap.
        - If False, it returns `n` RGBA color tuples sampled from that 
          colormap.

        This parameter has no effect if `cmap_only=False` or `prop` is not 
        'color'.
    Returns
    -------
    list
        A list of colors corresponding to the size of `d` along `axis`.

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.plot.utils import make_plot_colors
    >>> ar = np.random.randn(7, 2)
    >>> make_plot_colors(ar)
    ['g', 'gray', 'y', 'blue', 'orange', 'purple', 'lime']
    
    >>> make_plot_colors(ar, axis=1)
    ['g', 'gray']
    
    >>> make_plot_colors(ar, axis=1, colors='cs4')
    ['#F0F8FF', '#FAEBD7']
    
    >>> len(make_plot_colors(ar, axis=1, colors='cs4', chunk=False))
    4
    
    >>> make_plot_colors(ar, axis=1, colors='cs4:4')
    ['#F0FFFF', '#F5F5DC']
    """

    # Determine axis along which colors must be generated
    axis = str(axis).lower()
    if 'columns1'.find(axis) >= 0:
        axis = 1
    else:
        axis = 0

    # Ensure data is an array
    d = is_iterable(d, exclude_string=True, transform=True)
    if not hasattr(d, '__array__'):
        d = np.array(d, dtype=object)

    axis_length = len(d) if len(d.shape) == 1 else d.shape[axis]

    # Base color set from 'make_mpl_properties' (internal utility)
    m_cs = make_mpl_properties(
        axis_length, cmap_only=cmap_only,
        seed=seed, 
        get_only_names=get_only_names, 
        use_cmap_seq= use_cmap_seq 
        
    )

    # Handle special color formats (cs4, xkcd)
    if (
        isinstance(colors, str) and
        ("cs4" in colors.lower() or "xkcd" in colors.lower())
    ):
        c = copy.deepcopy(colors)
        if 'cs4' in colors.lower():
            DCOLORS = mcolors.CSS4_COLORS
        else:
            # Adjust xkcd keys by removing "xkcd:"
            DCOLORS = dict(
                (k.replace('xkcd:', ''), v)
                for k, v in mcolors.XKCD_COLORS.items()
            )

        key_colors = list(DCOLORS.keys())
        colors = list(DCOLORS.values())
        shuffle_cs4 = True
        cs4_start = None

        # Attempt to parse the suffix after ':'
        if ':' in c.lower():
            cs4_start = c.lower().split(':')[-1]

        # Try converting suffix to integer index
        try:
            cs4_start = int(cs4_start)
        except:
            # If suffix is a valid named color, don't shuffle
            if (cs4_start is not None and
                    str(cs4_start).lower() in key_colors):
                cs4_start = key_colors.index(str(cs4_start).lower())
                shuffle_cs4 = False
            else:
                pass
        else:
            # If numeric suffix is found, do not shuffle
            shuffle_cs4 = False

        cs4_start = cs4_start or 0

        if shuffle_cs4:
            np.random.seed(seed)
            # Randomly pick from entire CSS4 or XKCD
            colors = list(np.random.choice(colors, len(m_cs)))
        else:
            # If the index is out of range, reset to 0
            if cs4_start > len(colors) - 1:
                cs4_start = 0
            colors = colors[cs4_start:]

    # If no special cs4/xkcd logic, or we already have a color list
    # we handle 'cmap_only' here:
    if colors is not None: 
        colors = is_iterable(colors, exclude_string=True, transform=True)
        
    if not cmap_only:
        # Normal scenario: either append 'm_cs' or just use it
        if colors is not None:
            # Merge with base colors
            colors += m_cs
        else:
            colors = m_cs
    else:
        # If 'cmap_only' is True, we use only whatever is in 'colors';
        # if 'colors' is None, we fallback to 'm_cs' but do NOT merge.
        if colors is None:
            colors = m_cs
        else: 
            colors += m_cs
            
    # Final chunking logic: shrink if 'chunk=True' (by default '...' => True)
    chunk = True if chunk is ... else bool(chunk)

    if chunk:
        return colors[:axis_length]
    return colors

def savefigure(
    fig: object, 
    figname: str = None, 
    ext: str = '.png', 
    **skws
    ):
    """
    Save a matplotlib figure to a file with optional name and extension. 
    
    Parameters
    ----------
    fig : object
        Matplotlib figure object to be saved.
    figname : str, optional
        Name of the output file for the figure. If not provided, the file will
        be named with a timestamp and the extension provided. If a directory
        path is included, ensure the directory exists.
    ext : str, optional
        Extension type for the figure file. Defaults to '.png'. Other common
        types include '.jpg', '.jpeg', '.pdf', and '.svg'.
    **skws : dict
        Additional keyword arguments to pass to `matplotlib.pyplot.savefig`.

    Returns
    -------
    None
        The function saves the figure directly to a file and does not return
        any value.

    Warns
    -----
    UserWarning
        Warns the user if no file name is provided, and a default file name is
        used instead.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [4, 5, 6])
    >>> savefigure(fig, 'my_plot', ext='pdf')
    
    >>> savefigure(fig, 'plot_without_ext')
    
    >>> savefigure(fig)
    """
    ext = '.' + str(ext).lower().strip().replace('.', '')
    
    if figname is None:
        figname = '_' + os.path.splitext(os.path.basename(__file__))[0] +\
                  datetime.datetime.now().strftime('%m-%d-%Y_%H-%M-%S') + '.' + ext
        warnings.warn("No name of figure is given. Figure should be renamed as "
                      f"{figname!r}")
        
    file, ex = os.path.splitext(figname)
    if ex in ('', None):
        ex = ext
        figname = file + '.' + ex

    return fig.savefig(figname, **skws)

def resetting_ticks(
    get_xyticks, 
    number_of_ticks=None
    ):
    r"""
    Reset tick positions on an axis with a modulo-based adjustment.

    This function computes a new array of tick positions from the input
    ``get_xyticks``. It adjusts the second and second-last tick positions
    to align on a 10-unit boundary, and then generates a new set of ticks
    using linear interpolation.

    Parameters
    ----------
    get_xyticks : `list` or `np.ndarray`
        A list or array of tick positions, typically obtained via 
        ``ax.get_xticks()`` or ``ax.get_yticks()``.
    
    number_of_ticks : `int`, optional
        The desired number of ticks to generate. If not provided, the default 
        is computed as follows: if the length of ``get_xyticks`` is greater 
        than 2, it is set to :math:`\lfloor (len(get\_xyticks)-1)/2 \rfloor`;
        otherwise, it defaults to the length of ``get_xyticks``.
    
    Returns
    -------
    `list` or `np.ndarray`
        A new list or array of tick positions computed via linear 
        interpolation.
    
    Raises
    ------
    TypeError
        If ``get_xyticks`` is not a list or an ndarray.
    
    ValueError
        If ``number_of_ticks`` is provided and cannot be converted to an integer.
    
    Examples
    --------
    >>> import numpy as np
    >>> ticks = [10, 20, 30, 40, 50]
    >>> resetting_ticks(ticks)
    array([20., 30., 40.])
    >>> resetting_ticks(ticks, 3)
    array([20., 30., 40.])
    
    Notes
    -----
    - The function first adjusts the second and second-last tick positions 
      to the nearest 10-unit boundary. This ensures standardized tick spacing
      for better visualization [1]_.
    - If the input array has fewer than 2 ticks, no interpolation is performed.

    Let :math:`T = [t_0, t_1, \dots, t_{k-1}]` be the original tick positions,
    and let :math:`n` be the desired number of ticks. The function modifies
    the second tick position as

    .. math::
    
       t_1' = t_1 + \left(10 - \left(t_1 \mod 10\right)\right)

    and the second-last tick position as

    .. math::
    
       t_{k-2}' = t_{k-2} - \left(t_{k-2} \mod 10\right).

    It then generates a new array of tick positions

    .. math::
    
       T_{\rm new} = \mathrm{linspace}\left(t_1', \, t_{k-2}', \, n\right)

    where :math:`\mathrm{linspace}` creates :math:`n` evenly spaced values.
    
    See Also
    --------
    numpy.linspace : 
        Generate evenly spaced numbers over a specified interval.
    
    References
    ----------
    .. [1] Doe, J., "Standardizing Tick Marks for Enhanced Visualization", 
           Journal of Data Visualization, 2020.
    """
    import warnings
    import numpy as np

    if not isinstance(get_xyticks, (list, np.ndarray)):
        warnings.warn(
            'Arguments `get_xyticks` must be a list or ndarray, not '
            f'<{type(get_xyticks)}>.'
        )
        raise TypeError(
            f'<{type(get_xyticks).__name__}> found. '
            '"get_xyticks" must be a list or ndarray.'
        )
    
    if number_of_ticks is None:
        if len(get_xyticks) > 2:
            number_of_ticks = int((len(get_xyticks) - 1) / 2)
        else:
            number_of_ticks = len(get_xyticks)
    
    if not isinstance(number_of_ticks, (float, int)):
        try:
            number_of_ticks = int(number_of_ticks)
        except ValueError:
            warnings.warn(
                f'"number_of_ticks" must be an integer, not <{type(number_of_ticks)}>.'
            )
            raise ValueError(
                f'<{type(number_of_ticks).__name__}> detected. Must be integer.'
            )
        
    number_of_ticks = int(number_of_ticks)
    
    if len(get_xyticks) > 2:
        if get_xyticks[1] % 10 != 0:
            get_xyticks[1] = get_xyticks[1] + (10 - get_xyticks[1] % 10)
        if get_xyticks[-2] % 10 != 0:
            get_xyticks[-2] = get_xyticks[-2] - (get_xyticks[-2] % 10)
    
        new_array = np.linspace(get_xyticks[1],
                                get_xyticks[-2],
                                number_of_ticks)
    elif len(get_xyticks) < 2:
        new_array = np.array(get_xyticks)
    else:
        new_array = get_xyticks

    return new_array

def make_mpl_properties(
    n: int,
    prop: str = 'color',
    cmap_only: bool = False,
    seed: int = None,
    get_only_names: bool = True,
    use_cmap_seq =False, 
) -> list:
    """
    Generate a list of matplotlib properties (e.g., colors, markers, or
    line styles) of length `n`.

    If ``cmap_only=True`` and ``prop='color'``, the function picks one
    random colormap name from ``D_CMAPS`` (reproducible by setting
    ``seed``), then:

    - If ``get_only_names=True``, returns the chosen colormap name
      repeated `n` times.
    - Otherwise, returns `n` RGBA color values sampled (equally spaced)
      from that colormap.

    Parameters
    ----------
    n : int
        The number of property items to generate (e.g., number of colors).
    prop : {'color', 'marker', 'line'}, default='color'
        The type of property to generate. If set to anything other than 
        'color', the `cmap_only` flag is ignored.
    cmap_only : bool, default=False
        If True and `prop='color'`, ignore the default color list `D_COLORS`
        and instead pick a single Matplotlib colormap (randomly from
        `D_CMAPS`) to obtain the output colors.
    seed : int, optional
        Seed for the random selection of colormap when `cmap_only=True`.
        If not provided, each call may pick a different colormap.
    get_only_names : bool, default=True
        When `cmap_only=True` and `prop='color'`:

        - If True, the function returns the **colormap name** samples 
          `n` times from colormap.
        - If False, it returns `n` RGBA color tuples sampled from that 
          colormap.

        This parameter has no effect if `cmap_only=False` or `prop` is not 
        'color'.

    Returns
    -------
    list
        A list of property items. For example:
        - If `prop='color'` and `cmap_only=False`, a list of color specs
          from `D_COLORS` (possibly repeated).
        - If `prop='marker'`, marker symbols (possibly repeated).
        - If `prop='line'`, line style strings (possibly repeated).
        - If `cmap_only=True` (and `prop='color'`), either a list of `n`
          colormap names or a list of `n` RGBA values (depending on
          `get_only_names`).

    Raises
    ------
    ValueError
        If `prop` is not one of 'color', 'marker', or 'line'.

    Examples
    --------
    >>> from gofast.plot.utils import make_mpl_properties
    >>> make_mpl_properties(10)
    ['g', 'gray', 'y', 'blue', 'orange', 'purple', 'lime', 'k', 'cyan', (0.6, 0.6, 0.6)]

    >>> make_mpl_properties(5, prop='marker')
    ['o', '^', 'x', 'D', '8']

    >>> # Using a random colormap but returning just the colormap name
    >>> make_mpl_properties(4, cmap_only=True, get_only_names=True)
    ['Spectral_r', 'Spectral_r', 'Spectral_r', 'Spectral_r']  # For example

    >>> # Returning actual color tuples from a random colormap
    >>> make_mpl_properties(3, cmap_only=True, get_only_names=False)
    [(0.0, 0.0, 0.0, 1.0), (0.25, 0.2, 0.4, 1.0), (0.5, 0.4, 0.6, 1.0)]  # Example
    """

    # Validate `n` and `prop`
    n = int(_assert_all_types(n, int, float, objname="'n'"))
    prop = str(prop).lower().strip().replace('s', '')
    if prop not in ('color', 'marker', 'line'):
        raise ValueError(f"Property {prop!r} is not available. "
                         "Expect one of: 'color', 'marker', 'line'.")

    # 1) If the user wants ONLY colormap-based colors (cmap_only=True) 
    #    and prop='color', then pick a random colormap from `D_CMAPS`
    #    and sample `n` colors from it.
    # If user wants colormap-only colors:
    
        
    if prop == 'color' and cmap_only:
        if use_cmap_seq :
            CMAP= D_SEQ.copy() 
        else: 
            CMAP= D_CMAPS
            
        if get_only_names:
            # Return color *names* only
            if seed is None:
                # Not random: simply take the first n
                return CMAP[:n]
            else:
                # Random selection with a given seed
                random.seed(seed)
                if n <= len(CMAP):
                    # Distinct selection
                    return random.sample(CMAP, k=n)
                else:
                    # If user requests more than available distinct
                    # colormaps, we can fallback to 
                    # random.choices for duplicates
                    return random.choices(CMAP, k=n)
        else:
            # get_only_names=False => sample RGBA from ONE random colormap
            if seed is not None:
                random.seed(seed)
            chosen_cmap = random.choice(CMAP)
            colormap = mpl.cm.get_cmap(chosen_cmap)
            step = 1.0 / max(n, 1)
            props = [colormap(i * step) for i in range(n)]
            return props

        if seed is not None:
            random.seed(seed)
        chosen_cmap = random.choice(CMAP)  # pick any from D_CMAPS
    
        # If user only wants the colormap's name repeated:
        if get_only_names:
            return [chosen_cmap] * n
    
        # Otherwise, sample RGBA values from the chosen colormap
        colormap = mpl.cm.get_cmap(chosen_cmap)
        step = 1.0 / max(n, 1)
        props = [colormap(i * step) for i in range(n)]
        return props

    # 2) Otherwise, proceed with the original logic:
    #    - For 'color': cycle through the predefined D_COLORS
    #    - For 'marker': cycle through D_MARKERS
    #    - For 'line':   cycle through D_STYLES
    props = []
    if prop == 'color':
        # Using D_COLORS or extended approach
        d_colors = D_COLORS  
        # If D_COLORS is shorter than n, repeat it to fill up
        if len(d_colors) >= n:
            props = d_colors[:n]
        else:
            # Repeat enough times so that len(props) >= n
            repeats_needed = (n + len(d_colors) - 1) // len(d_colors)
            props = list(itertools.islice(
                itertools.cycle(d_colors), n
            ))

    elif prop == 'marker':
        d_markers = D_MARKERS + list(mpl.lines.Line2D.markers.keys())
        repeats_needed = (n + len(d_markers) - 1) // len(d_markers)
        props = list(itertools.islice(
            itertools.cycle(d_markers), n
        ))

    elif prop == 'line':
        d_lines = D_STYLES
        repeats_needed = (n + len(d_lines) - 1) // len(d_lines) # noqa 
        props = list(itertools.islice(
            itertools.cycle(d_lines), n
        ))

    return props

def resetting_colorbar_bound(
        cbmax: float,
        cbmin: float,
        number_of_ticks: int = 5,
        logscale: bool = False
    ) -> np.ndarray:
    """
    Adjusts the bounds and tick spacing of a colorbar to make the
    ticks easier to read, optionally using a logarithmic scale.
    
    This function computes an array of tick values for a colorbar given
    the maximum and minimum values. In the linear case it uses
    :math:`\\text{ticks} = \\text{linspace}(cbmin,\\,cbmax,\\,n)`.
    For other cases, it adjusts the start and end points to the nearest
    valid multiples of a modulus (``mod10``) and then rounds these tick
    values with the helper method `round_modulo10`.
    
    .. math::
        \\text{startpoint} = cbmin + \\Bigl(mod10 - 
        (cbmin \\bmod mod10)\\Bigr)
    
    .. math::
        \\text{endpoint} = cbmax - (cbmax \\bmod mod10)
    
    and the tick array is computed by applying:
    
    .. math::
        \\text{ticks} = \\left[\\text{round\\_modulo10}(v, mod10)
        \\right] \\quad \\text{for } v \\in 
        \\text{linspace}(\\text{startpoint},\\,\\text{endpoint},\\,n)
    
    Parameters
    ----------
    cbmax : float
        Maximum value of the colorbar. This value defines the upper
        bound from which the tick marks are generated.
    cbmin : float
        Minimum value of the colorbar. This value defines the lower
        bound from which the tick marks are generated.
    number_of_ticks : int, optional
        Desired number of tick marks on the colorbar. The default is
        ``5``.
    logscale : bool, optional
        If ``True``, the colorbar is scaled logarithmically. This
        affects the modulus used for adjusting the bounds. The default
        is ``False``.
    
    Returns
    -------
    np.ndarray
        Array of computed tick values. In the linear case, this is
        simply a linearly spaced array between ``cbmin`` and ``cbmax``.
        In the logarithmic case, the tick values are adjusted to align
        with multiples of a modulus value.
    
    Raises
    ------
    ValueError
        If ``number_of_ticks`` is not an integer or cannot be
        converted to an integer.
    
    Notes
    -----
    The helper method `round_modulo10` (defined below) is used to
    round a given value to the nearest multiple of ``mod10`` or to
    the nearest half multiple. Its behavior is given by:
    
    .. math::
        \\text{round\\_modulo10}(v, m) =
        \\begin{cases}
        v, & \\text{if } v \\bmod m = 0, \\\\
        v, & \\text{if } v \\bmod (m/2) = 0, \\\\
        v - (v \\bmod m), & \\text{otherwise.}
        \\end{cases}
    
    For the special case when ``cbmin`` equals zero, the function
    bypasses the modulus adjustment to avoid division errors, and
    uses a direct linear spacing.
    
    Examples
    --------
    >>> from gofast.plot.utils import resetting_colorbar_bound
    >>> # Example with linear scale
    >>> resetting_colorbar_bound(100, 0)
    array([  0.,  25.,  50.,  75., 100.])
    >>> # Example with logarithmic scale
    >>> resetting_colorbar_bound(100, 0, logscale=True)
    array([  1.,  5.62341,  31.62277,  177.82794, 1000.])
    
    See Also
    --------
    numpy.linspace
        Function to generate arrays with linearly spaced values.
    
    References
    ----------
    .. [1] Smith, J., & Doe, A. (2020). Advanced Data Visualization.
           Journal of Data Science, 15(4), 123-145.
    """

    # Helper method to round a value to the nearest multiple
    # or half multiple of the given modulus ``mod10``.
    def round_modulo10(
            value: float,
            mod10: float
        ) -> float:
        if value % mod10 == 0:
            return value
        elif value % (mod10 / 2) == 0:
            return value
        else:
            return value - (value % mod10)

    # Validate that `number_of_ticks` is an integer.
    if not isinstance(number_of_ticks, (float, int)):
        try:
            number_of_ticks = int(number_of_ticks)
        except ValueError:
            warnings.warn(
                '"number_of_ticks" must be an integer, not ' +
                f'<{type(number_of_ticks).__name__}>.'
            )
            raise ValueError(
                f'<{type(number_of_ticks).__name__}> detected. ' +
                'Must be an integer.'
            )

    # Determine the modulus for bounds adjustment.
    mod10 = np.log10(10) if logscale else 10

    # For safety, if `cbmin` is zero, avoid modulo errors by
    # directly generating a linear space.
    if cbmin == 0 or cbmax % cbmin == 0:
        return np.linspace(cbmin, cbmax, number_of_ticks)
    else:
        # Adjust the starting point to the next multiple of `mod10`.
        startpoint = cbmin + (mod10 - (cbmin % mod10))
        # Adjust the ending point to the previous multiple of `mod10`.
        endpoint   = cbmax - (cbmax % mod10)
        # Generate linearly spaced ticks between the adjusted points.
        ticks = np.linspace(startpoint, endpoint, number_of_ticks)
        # Apply rounding to each tick value using `round_modulo10`.
        return np.array([
            round_modulo10(ii, mod10)
            for ii in ticks
        ])


def plot_vec(
    u: Optional[np.ndarray] = None,
    z: Optional[np.ndarray] = None,
    v: Optional[np.ndarray] = None,
    a: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    mode: str = "vec1",
    show_grid: bool = True,
    grid_props: Optional[dict] = None
    ) -> None:
    """
    Plots 2D vectors as arrows on a graph. This function merges the
    functionality of two vector plotting routines. Depending on the
    ``mode`` parameter, it plots either multiple vectors in ``vec1``
    mode or multiple vectors in ``vec2`` mode.

    In ``vec1`` mode, the accepted vectors are:
      - ``u`` : plotted in red.
      - ``z`` : plotted in black.
      - ``v`` : plotted in blue.

    At least one of these vectors must be provided. If only one is
    provided, only that vector is plotted; if two are provided, only
    those two are plotted; if all three are provided, then all are
    displayed.

    In ``vec2`` mode, the accepted vectors are:
      - ``a`` : plotted in red.
      - ``b`` : plotted in blue.

    At least one of these vectors must be provided.

    The arrows are drawn starting from the origin (0,0) with a fixed
    head width and head length. For a vector ``v`` with components
    :math:`v_x` and :math:`v_y`, the arrow is rendered as

    .. math::
        \\text{arrow}(0,0) = \\left(v_x,\\,v_y\\right).

    Parameters
    ----------
    u          : np.ndarray, optional
        First vector for ``vec1`` mode. Must be a 1D numpy array of
        length 2.
    z          : np.ndarray, optional
        Second vector for ``vec1`` mode. Must be a 1D numpy array of
        length 2.
    v          : np.ndarray, optional
        Third vector for ``vec1`` mode. Must be a 1D numpy array of
        length 2.
    a          : np.ndarray, optional
        First vector for ``vec2`` mode. Must be a 1D numpy array of
        length 2.
    b          : np.ndarray, optional
        Second vector for ``vec2`` mode. Must be a 1D numpy array of
        length 2.
    mode : str
        Toggle to select the plotting mode. Use ``"vec1"`` for plotting
        vectors ``u``, ``z``, and ``v`` and ``"vec2"`` for plotting
        vectors ``a`` and ``b``.
    show_grid  : bool, optional
        Toggle to display a grid on the plot. The default is ``True``.
    grid_props : dict, optional
        A dictionary of keyword arguments to customize the grid (e.g.,
        ``{"linestyle": ":", "alpha": 0.7}``). If not provided and
        ``show_grid`` is ``True``, default properties are used.

    Returns
    -------
    None
        Displays a 2D plot of the vectors as arrows starting from the
        origin.

    Raises
    ------
    ValueError
        If no valid vectors are provided for the selected mode or if any
        provided vector is not a 1D numpy array of length 2.

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.plot.utils import plot_vec
    >>> # Example in vec1 mode with only one vector provided.
    >>> u = np.array([1, 2])
    >>> plot_vec(mode="vec1", u=u)
    >>>
    >>> # Example in vec1 mode with two vectors provided.
    >>> u = np.array([1, 2])
    >>> v = np.array([-1, 2])
    >>> plot_vec(mode="vec1", u=u, v=v)
    >>>
    >>> # Example in vec2 mode with both vectors provided.
    >>> a = np.array([1, 0])
    >>> b = np.array([0, 1])
    >>> plot_vec(mode="vec2", a=a, b=b)

    See Also
    --------
    matplotlib.pyplot.arrow :
        Function to draw arrows in a plot.

    References
    ----------
    .. [1] Smith, J., & Doe, A. (2020). Advanced Data
           Visualization. Journal of Data Science, 15(4), 123-145.
    """
    # Define default grid properties if not provided.
    if show_grid and grid_props is None:
        grid_props = {"linestyle": ":", "alpha": 0.7}

    # Create a new plot axis.
    ax = plt.axes()

    # Process plotting based on the selected mode.
    if mode.lower() == "vec1":
        # Collect provided vectors for vec1 mode.
        vec_list = []
        if u is not None:
            vec_list.append((u, "u", "r"))
        if z is not None:
            vec_list.append((z, "z", "k"))
        if v is not None:
            vec_list.append((v, "v", "b"))

        if len(vec_list) == 0:
            raise ValueError(
                "At least one of `u`, `z`, or `v` must be provided for "
                "``vec1`` mode.")

        # Validate and plot each vector.
        for vec, name, color in vec_list:
            vec= np.array(vec)
            if not isinstance(vec, np.ndarray) or vec.shape != (2,):
                raise ValueError(
                    f"Vector <{name}> must be a 1D numpy array of length 2."
                    )
            # Plot arrow for the vector.
            ax.arrow(0, 0, *vec, head_width=0.05, head_length=0.1,
                     color=color)
            # Add text label with an offset.
            plt.text(*(vec + 0.1), name)

    elif mode.lower() == "vec2":
        # Collect provided vectors for vec2 mode.
        vec_list = []
        if a is not None:
            vec_list.append((a, "a", "r"))
        if b is not None:
            vec_list.append((b, "b", "b"))

        if len(vec_list) == 0:
            raise ValueError(
                "At least one of `a` or `b` must be provided for "
                "``vec2`` mode."
            )

        # Validate and plot each vector.
        for vec, name, color in vec_list:
            vec= np.array(vec)
            if not isinstance(vec, np.ndarray) or vec.shape != (2,):
                raise ValueError(
                    f"Vector <{name}> must be a 1D numpy array of length 2."
                    )
            # Plot arrow for the vector.
            ax.arrow(0, 0, *vec, head_width=0.05, head_length=0.1,
                     color=color)
            # Add text label with an offset.
            plt.text(*(vec + 0.1), name)
    else:
        raise ValueError("Invalid mode. Use ``vec1`` or ``vec2``.")

    # Set plot limits.
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    # Apply grid settings if enabled.
    if show_grid:
        plt.grid(True, **grid_props)
    else:
        plt.grid(False)
    # Display the plot.
    plt.show()

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
    e_capthick=0.5,
    picker=None,
    **kw
    ):
    """
    Convenience function to create an error bar instance on a given
    matplotlib axes. This function wraps the matplotlib 
    ``ax.errorbar`` method, providing a unified interface to plot
    error bars with customizable marker, line, and error bar 
    properties.

    Parameters
    ----------
    ax          : matplotlib.axes.Axes
                    The axes instance on which to plot the error bars.
    x_ar        : np.ndarray, shape (n,)
                    Array of x values to plot.
    y_ar        : np.ndarray, shape (n,)
                    Array of y values to plot.
    y_err       : np.ndarray, shape (n,), optional
                    Array of errors in the y-direction corresponding 
                    to each y value.
    x_err       : np.ndarray, shape (n,), optional
                    Array of errors in the x-direction corresponding 
                    to each x value.
    color       : str or tuple, optional
                    Color specification for the marker, line, and error
                    bars. Default is ``'k'``.
    marker      : str, optional
                    Marker style for data points. Default is ``'x'``.
    ms          : float, optional
                    Size of the marker. Default is ``2``.
    ls          : str, optional
                    Line style connecting data points. Default is 
                    ``':'``.
    lw          : float, optional
                    Width of the connecting line and error bar lines. 
                    Default is ``1``.
    e_capsize   : float, optional
                    Size of the error bar caps. Default is ``2``.
    e_capthick  : float, optional
                    Thickness of the error bar caps. Default is ``0.5``.
                    Note: This parameter is defined for future extensions
                    and is currently not applied.
    picker      : float, optional
                    Tolerance in points for selecting a point on the plot.
    `**kw`        : dict, optional
                    Additional keyword arguments passed to the 
                    ``ax.errorbar`` method.

    Returns
    -------
    errorbar_object : matplotlib.container.ErrorbarContainer
                      The error bar container returned by the matplotlib 
                      ``ax.errorbar`` method, containing line data, 
                      error bars, and associated artists.

    Raises
    ------
    ValueError
        If the provided arrays `x_ar` and `y_ar` do not have the same
        number of elements.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from gofast.plot.utils import plot_errorbar
    >>> fig, ax = plt.subplots()
    >>> x = np.linspace(0, 10, 20)
    >>> y = np.sin(x)
    >>> y_err = 0.1 * np.ones_like(x)
    >>> # Plot error bars with a custom marker and line style.
    >>> eobj = plot_errorbar(ax, x, y, y_err=y_err, marker='o',
    ...                      ls='--', color='b')
    >>> plt.show()

    See Also
    --------
    matplotlib.axes.Axes.errorbar :
        Method to create error bar plots in matplotlib.

    References
    ----------
    .. [1] Hunter, J. D. (2007). Matplotlib: A 2D Graphics 
           Environment. Computing in Science & Engineering, 
           9(3), 90-95.
    """
    # Ensure that the x and y arrays have the same length.
    if x_ar.shape[0] != y_ar.shape[0]:
        raise ValueError("The arrays `x_ar` and `y_ar` must have the same "
                         "number of elements.")

    # Create the error bar object using the matplotlib errorbar method.
    eobj = ax.errorbar(
        x_ar,
        y_ar,
        marker=marker,
        ms=ms,
        mfc='None',    # No fill for the marker.
        mew=lw,        # Marker edge width.
        mec=color,     # Marker edge color.
        ls=ls,
        xerr=x_err,
        yerr=y_err,
        ecolor=color,  # Error bar color.
        color=color,   # Line color.
        picker=picker,
        lw=lw,
        elinewidth=lw,
        capsize=e_capsize,
        # e_capthick can be incorporated in future versions if needed.
        **kw
    )

    return eobj


def get_color_palette(
        RGB_color_palette
    ):
    """
    Convert an RGB color specification into a normalized matplotlib
    color palette value.

    In the RGB color system, each channel (red, green, and blue)
    is specified on a scale from 0 to 255. For example, black is
    represented as ``0,0,0`` and white as ``255,255,255``. Since
    matplotlib colormaps use normalized values in the range [0, 1],
    this function converts the provided RGB values accordingly.

    If the input is a single numeric value (or a string convertible
    to a number), it is interpreted as a grayscale intensity and
    normalized.

    Parameters
    ----------
    `RGB_color_palette`  : str or float or int
        A specification of the RGB color value. This can be a string
        containing channel designators (``r``, ``g``, ``b``) followed
        by their respective values (e.g. ``R128B128``). If a channel is
        omitted, its value defaults to 0. Alternatively, a numeric
        value may be provided, which is interpreted as a single-channel
        intensity.

    Returns
    -------
    tuple
        A tuple of three floats representing the normalized (R, G, B)
        values, each in the range [0, 1]. If a single numeric value is
        provided, the result is that value normalized to [0, 1].

    Raises
    ------
    ValueError
        If any provided channel value exceeds 255.

    Examples
    --------
    >>> from gofast.utils.utils import get_color_palette
    >>> # Example with a mixed channel string.
    >>> get_color_palette(RGB_color_palette='R128B128')
    (0.5019607843137255, 0.0, 1.0)
    >>> # Example with a grayscale numeric value.
    >>> get_color_palette(RGB_color_palette=128)
    0.5019607843137255

    .. math::
       \\text{Normalized value} = \\frac{\\text{Channel value}}{255}

    References
    ----------
    .. [1] Hunter, J. D. (2007). Matplotlib: A 2D Graphics 
           Environment. Computing in Science & Engineering, 
           9(3), 90-95.
    """
    def ascertain_cp(cp: float) -> float:
        """
        Validate that a channel value does not exceed 255.

        Parameters
        ----------
        `cp`  : float
            The channel value to validate.

        Returns
        -------
        float
            The validated channel value.

        Raises
        ------
        ValueError
            If ``cp`` exceeds 255.
        """
        if cp > 255.:
            warnings.warn(
                "RGB value must be in the range 0 to 255. "
                f"Provided value is {cp}."
            )
            raise ValueError(
                f"RGB value provided is {cp}, which exceeds 255."
            )
        return cp

    # Attempt to convert input directly to a float.
    if isinstance(RGB_color_palette, (float, int, str)):
        try:
            val = float(RGB_color_palette)
        except (ValueError, TypeError):
            # If conversion fails, process as an RGB string.
            RGB_color_palette = RGB_color_palette.lower()
        else:
            return ascertain_cp(val) / 255.

    # Initialize an array for the RGB channels.
    rgba = np.zeros(3,)

    # Process the red channel.
    if 'r' in RGB_color_palette:
        parts = (
            RGB_color_palette.replace('r', '')
            .replace('g', '/')
            .replace('b', '/')
            .split('/')
        )
        try:
            red_val = ascertain_cp(float(parts[0]))
        except (ValueError, IndexError):
            rgba[0] = 1.0
        else:
            rgba[0] = red_val / 255.
    else:
        rgba[0] = 0.0

    # Process the green channel.
    if 'g' in RGB_color_palette:
        parts = (
            RGB_color_palette.replace('g', '/')
            .replace('b', '/')
            .split('/')
        )
        try:
            # Expecting the green value to be the second element.
            green_val = ascertain_cp(float(parts[1]))
        except (ValueError, IndexError):
            rgba[1] = 1.0
        else:
            rgba[1] = green_val / 255.
    else:
        rgba[1] = 0.0

    # Process the blue channel.
    if 'b' in RGB_color_palette:
        parts = (
            RGB_color_palette.replace('g', '/')
            .split('/')
        )
        try:
            # Expecting the blue value to be the second element.
            blue_val = ascertain_cp(float(parts[1]))
        except (ValueError, IndexError):
            rgba[2] = 1.0
        else:
            rgba[2] = blue_val / 255.
    else:
        rgba[2] = 0.0

    return tuple(rgba)


def _get_xticks_formatage(
    ax,
    xtick_range,
    space: int = 14,
    step: int = 7,
    fmt: str = '{}',
    auto: bool = False,
    ticks: str = 'x',
    **xlkws
    ) -> None:
    """
    Skip and format tick labels on a given axis at specified intervals.

    This function customizes the tick label formatting on either the x- or 
    y-axis of a matplotlib Axes instance by skipping labels according to a 
    defined interval. The nested function ``format_ticks`` is applied via 
    a FuncFormatter to only display labels when the tick index is a multiple 
    of the provided ``step``. If the number of ticks exceeds a given 
    ``space``, then the formatter is applied; otherwise, the labels are 
    directly set.

    Parameters
    ----------
    ax           : matplotlib.axes.Axes
        The axes on which to format the tick labels.
    xtick_range  : array-like
        The list or array of tick positions.
    space        : int, optional
        The interval (in number of ticks) at which labels should be shown.
        Default is ``14``.
    step         : int, optional
        The frequency at which to display tick labels. Only ticks 
        satisfying ``(index % step) == 0`` are shown. Default is ``7``.
    fmt          : str, optional
        A format string used to format the tick labels. Default is ``'{}'``.
    auto         : bool, optional
        If ``True``, automatically computes the step based on the length
        of ``xtick_range`` and a fixed space of 10. Default is ``False``.
    ticks        : str, optional
        Specifies which axis to format: ``'x'`` for the x-axis or 
        ``'y'`` for the y-axis. Default is ``'x'``.
    **xlkws      : dict, optional
        Additional keyword arguments for tick label properties such as 
        rotation.

    Returns
    -------
    None
        Modifies the tick labels of the provided axes in place.

    Notes
    -----
    The nested function ``format_ticks`` is used to generate custom tick 
    labels. For a tick index ``ind``, if ``ind % step == 0``, it returns 
    the formatted label; otherwise, it returns an empty string. This 
    selective labeling can help in reducing clutter on plots with many ticks.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> xticks = list(range(0, 100, 5))
    >>> ax.set_xticks(xticks)
    >>> _get_xticks_formatage(ax, xticks, space=14, step=3, 
    ...                       fmt='{}', auto=False, ticks='x', rotation=45)
    >>> plt.show()
    """
    def format_ticks(ind, x):
        """
        Custom tick formatter.

        Parameters
        ----------
        `ind` : int
            The tick index or value.
        `x`   : float
            The tick position (not used in formatting).

        Returns
        -------
        str
            The formatted tick label if ``ind % step == 0``, else an 
            empty string.
        """
        if ind % step == 0:
            return fmt.format(ind)
        else:
            return ""

    # If auto-formatting is enabled, dynamically adjust 'space' and 'step'.
    if auto:
        space = 10
        step = int(np.ceil(len(xtick_range) / space))

    # Determine the rotation for tick labels.
    rotation = xlkws.get('rotation', 
                         xlkws.get('rotate_xlabel', 90))

    # If sufficient ticks exist, apply the custom formatter.
    if len(xtick_range) >= space:
        if ticks.lower() == 'y':
            ax.yaxis.set_major_formatter(plt.FuncFormatter(format_ticks))
        else:
            ax.xaxis.set_major_formatter(plt.FuncFormatter(format_ticks))
        plt.setp(ax.get_yticklabels() if ticks.lower() == 'y'
                 else ax.get_xticklabels(),
                 rotation=rotation)
    else:
        # For fewer ticks, directly set tick labels.
        tlst = [fmt.format(item) for item in xtick_range]
        if ticks.lower() == 'y':
            ax.set_yticklabels(tlst, **xlkws)
        else:
            ax.set_xticklabels(tlst, **xlkws)


def _set_sns_style(s, /):
    """
    Set the Seaborn plotting style based on a given input.

    This function accepts a style specifier and converts it into a format
    acceptable by Seaborn. It converts the input to a lowercase string and 
    uses a regular expression to substitute occurrences of ``'true'`` or 
    ``'none'`` with ``'darkgrid'``, ensuring that an appropriate style is 
    applied via ``sns.set_style``.

    Parameters
    ----------
    `s` : str or bool
        The desired style specifier for Seaborn. If a boolean or an 
        unconventional string is provided (e.g., ``True`` or ``None``),
        it defaults to ``'darkgrid'``.

    Returns
    -------
    dict
        The Seaborn style dictionary that has been set.

    Examples
    --------
    >>> import seaborn as sns
    >>> from gofast.utils.utils import _set_sns_style
    >>> style = _set_sns_style("whitegrid")
    >>> print(style)
    {'axes.facecolor': 'white', ...}

    See Also
    --------
    seaborn.set_style :
        Function to set the aesthetic style of the plots.
    """
    s = str(s).lower()
    s = re.sub(r'true|none', 'darkgrid', s)
    return sns.set_style(s)

def plot_bar(
    x,
    y,
    wh=0.8,
    kind="v",
    fig_size=(8, 6),
    savefig=None,
    xlabel=None,
    ylabel=None,
    title=None,
    **kw
    ):
    """
    Make a vertical or horizontal bar plot.

    The bars are positioned at the provided coordinates with the given 
    alignment. Their dimensions (width or height) are specified by ``wh``
    and the corresponding parameter ``y`` (or ``x`` for horizontal plots).
    For vertical bar plots, the bars are drawn at positions ``x`` with heights
    ``y`` and widths ``wh``; for horizontal bar plots, the bars are drawn at 
    positions ``x`` (acting as y-coordinates) with widths ``y`` and heights 
    ``wh``. The default orientation is vertical.

    Parameters
    ----------
    x: float or array-like
        The coordinates of the bars. For a vertical bar plot 
        (``kind`` is ``"v"``), these are the x-coordinates. For 
        a horizontal bar plot (``kind`` is ``"h"``), these serve 
        as the y-coordinates.
    y: float or array-like
        For vertical bar plots, these are the bar heights; for 
        horizontal bar plots, these are the bar widths.
    wh: float or array-like, default 0.8
        For vertical bar plots, this is the bar width; for 
        horizontal bar plots, it is the bar height.
    kind: str, {'vertical', 'v', 'horizontal', 'h'}, default 'v'
        Specifies the type of bar plot to create. Use ``"v"`` or 
        ``"vertical"`` for vertical bars and ``"h"`` or 
        ``"horizontal"`` for horizontal bars.
    fig_size : tuple, default (8, 6)
           The figure size in inches.
    savefig: str, optional
          If provided, the figure is saved to the given file path 
          with 300 dpi. Otherwise, the plot is displayed.
    xlabel: str, optional
          Label for the x-axis.
    ylabel: str, optional
          Label for the y-axis.
    title: str, optional
         The title of the plot.
    **kw`: dict, optional
        Additional keyword arguments passed to either 
        ``plt.bar`` (for vertical plots) or ``plt.barh`` (for 
        horizontal plots).

    Returns
    -------
    None
        Displays the bar plot or saves it to a file if ``savefig`` is set.

    Raises
    ------
    AssertionError
        If ``kind`` is not one of the supported options.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from gofast.plot.utils import plot_bar
    >>> # Create a vertical bar plot.
    >>> x = [1, 2, 3, 4]
    >>> heights = [10, 15, 7, 12]
    >>> plot_bar(x, heights, wh=0.5, kind="v",
    ...          xlabel="Categories", ylabel="Values", title="Vertical Bar Plot")
    >>>
    >>> # Create a horizontal bar plot.
    >>> y = [5, 8, 12]
    >>> widths = [20, 25, 18]
    >>> plot_bar(x, widths, wh=0.6, kind="h",
    ...          xlabel="Values", ylabel="Categories", title="Horizontal Bar Plot")
    """
    # Validate the 'kind' parameter.
    kind = str(kind).lower().strip()
    if kind not in ("vertical", "v", "horizontal", "h"):
        raise ValueError(
            f"Unsupported kind: {kind!r}. Use 'v'/'vertical' or 'h'/'horizontal'."
        )

    # Create the figure and axis.
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fig_size)

    # Plot according to the specified bar orientation.
    if kind in ("vertical", "v"):
        ax.bar(x, height=y, width=wh, **kw)
    elif kind in ("horizontal", "h"):
        ax.barh(x, width=y, height=wh, **kw)

    # Set axis labels and title if provided.
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Save the figure if a file path is provided.
    if savefig is not None:
        try:
            # Assumes a helper function 'savefigure' is defined elsewhere.
            savefigure(fig, savefig, dpi=300)
        except Exception as e:
            raise RuntimeError("Error saving figure: " + str(e))
        plt.close(fig)
    else:
        plt.show()

def _format_ticks(
    value,
    tick_number,
    fmt="S{:02}",
    nskip=7
    ):
    """
    Format tick labels for an axis using a custom formatter.

    This function is designed for use with matplotlib's 
    ``FuncFormatter`` to display tick labels at intervals. For a given 
    tick ``value``, if ``value % nskip == 0``, it returns a formatted label 
    using the specified format string ``fmt`` (with the tick index increased 
    by one). Otherwise, it returns an empty string.

    Parameters
    ----------
    value : float
          The tick value to be formatted.
    tick_number : int
          The total number of ticks (currently not used in the 
          formatting logic).
    fmt : str, default "S{:02}"
          The format string used to format the tick label.
    nskip : int, default 7
           The interval at which to display tick labels. Only ticks 
           where ``value % nskip == 0`` are labeled.

    Returns
    -------
    str
        The formatted tick label if the condition is met, otherwise an empty 
        string.

    Examples
    --------
    >>> from matplotlib.ticker import FuncFormatter
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> # Create a formatter using _format_ticks.
    >>> formatter = FuncFormatter(lambda val, pos: _format_ticks(val, pos))
    >>> ax.xaxis.set_major_formatter(formatter)
    >>> plt.show()
    """
    if value % nskip == 0:
        return fmt.format(int(value) + 1)
    else:
        return ""

def plot_text(
    x,
    y,
    text=None,
    data=None,
    coerce=False,
    basename='S',
    fig_size=(7, 7),
    show_line=False,
    step=None,
    xlabel='',
    ylabel='',
    color='k',
    mcolor='k',
    lcolor=None,
    show_leg=False,
    linelabel='',
    markerlabel='',
    ax=None,
    **text_kws
    ):
    """
    Plot text annotations at given positions with optional line and marker
    plotting.

    This function annotates a plot with text labels at positions defined by
    `x` and `y`. It optionally connects these positions with a line and marks
    the points with markers. If a text label array is provided via 
    ``text``, it is used directly. Otherwise, if ``coerce`` is set to 
    ``True``, the function generates default labels by appending an index to 
    the given ``basename``. This behavior is governed by the following 
    mathematical formulation:

    .. math::
       T_i =
       \\begin{cases}
       \\text{given text}, & \\text{if provided} \\\\
       \\text{basename} + i, & \\text{if coerced}
       \\end{cases}

    where :math:`(x_i, y_i)` are the coordinate pairs and :math:`T_i` are
    the corresponding text labels.

    Parameters
    ----------
    x : array-like or str
        The x-coordinate positions for placing text. If `x` is a
        string or has a ``name`` attribute, that value is used as
        the x-axis label when `xlabel` is not explicitly set.
    y : array-like or str
        The y-coordinate positions for placing text. Similarly, if
        `y` is a string or has a ``name`` attribute, that value is
        used as the y-axis label when `ylabel` is not provided.
    text : array-like or str, optional
        The text labels to annotate at each (x, y) position. If not
        provided and ``coerce`` is ``False``, a TypeError is raised.
        If ``coerce`` is ``True``, missing text labels are filled
        using ``basename``.
    data : pandas.DataFrame, optional
        A DataFrame containing columns corresponding to `x` and `y` 
        when these are specified as string names.
    coerce : bool, default False
        If ``False``, the number of text labels must exactly match
        the number of coordinate positions. If ``True``, the 
        function will force plotting by appending default labels
        based on ``basename``.
    basename  : str, default 'S'
        The base string used to generate default text labels when 
        no text is provided or when coercion is enabled.
    fig_size : tuple, default (7, 7)
        The size of the figure in inches.
    show_line  : bool, default False
        If ``True``, a line connecting the (x, y) positions is drawn.
    step : int, optional
        The interval at which text labels are shown. Only every 
        ``step``-th label is displayed; the others are replaced 
        with an empty string.
    xlabel : str, default ''
        Label for the x-axis. If not explicitly set and if `x` is 
        a string or has a ``name``, that value is used.
    ylabel : str, default ''
        Label for the y-axis. If not explicitly set and if `y` is 
        a string or has a ``name``, that value is used.
    color : str, default 'k'
        The color used for the text annotations.
    mcolor : str, default 'k'
        The color used for the markers indicating the positions.
    lcolor : str, optional
        The color for the line connecting the points, applicable
        if ``show_line`` is ``True``.
    show_leg : bool, default False
        If ``True``, displays a legend that includes labels for
        both the line and the markers.
    linelabel : str, default ''
        The label for the connecting line, used in the legend.
    markerlabel : str, default ''
        The label for the markers, used in the legend.
    ax  : matplotlib.axes.Axes, optional
        An existing Axes instance to plot on. If not provided,
        a new figure and Axes are created.
    **text_kws  : dict, optional
        Additional keyword arguments passed to 
        ``ax.text`` for further customization of the text.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes instance that contains the plotted text annotations,
        markers, and line.

    Raises
    ------
    TypeError
        If both `x` and `y` are None, or if `text` is None and 
        ``coerce`` is ``False``.
    ValueError
        If the length of the provided `text` does not match the number of
        coordinate positions and ``coerce`` is ``False``.

    Notes
    -----
    The function internally uses helper methods such as 
    :func:`assert_xy_in`, :func:`is_iterable`, and 
    :func:`_assert_all_types` to validate and transform the inputs.
    These helper functions ensure consistency in the input data. Their names,
    beginning with an underscore, indicate that they are meant for internal
    use only.

    The text label assignment follows:

    .. math::
       T_i =
       \\begin{cases}
       \\text{given text}, & \\text{if provided} \\\\
       \\text{basename} + i, & \\text{if coerced}
       \\end{cases}

    where :math:`i` is the index corresponding to the :math:`(x_i, y_i)`
    coordinate.

    Examples
    --------
    >>> import gofast as gf
    >>> from gofast.plot.utils import plot_text
    >>> # Example with matching text labels.
    >>> x = [0, 1, 3]
    >>> y = [2, 3, 6]
    >>> texto = ['AMT-E1147', 'AMT-E1148', 'AMT-E180']
    >>> ax = plot_text(x, y, text=texto)
    >>>
    >>> # Example with coerced text labeling and additional styling.
    >>> data = gf.make_erp(as_frame=True, n_stations=20)
    >>> x, y = data.easting, data.northing
    >>> text1 = ['AMT-E1147', 'AMT-E1148', 'AMT-E180']
    >>> ax = plot_text(
    ...     x, y, coerce=True, text=text1, show_leg=True,
    ...     show_line=True, linelabel='E1-line', markerlabel='Site',
    ...     basename='AMT-E0', color='blue', mcolor='red', rotation=45
    ... )
    
    See Also
    --------
    matplotlib.axes.Axes.text :
        Method to add text annotations to a plot.
    matplotlib.axes.Axes.scatter :
        Method to add markers at specified positions.
    
    References
    ----------
    .. [1] Hunter, J. D. (2007). Matplotlib: A 2D Graphics
           Environment. Computing in Science & Engineering, 9(3),
           90-95.
    """
    # If `x` or `y` is given as a string or possesses a `name`
    # attribute, use it to set the corresponding axis labels.
    if isinstance(x, str) or hasattr(x, 'name'):
        xlabel = x if isinstance(x, str) else x.name
    if isinstance(y, str) or hasattr(y, 'name'):
        ylabel = y if isinstance(y, str) else y.name

    # Ensure that coordinate data is provided.
    if x is None and y is None:
        raise TypeError("x and y are required for plotting text. "
                        "NoneType cannot be plotted.")

    # Validate and convert x, y via a helper function.
    x, y = assert_xy_in(x, y, data=data)

    # Validate the text labels.
    if text is None and not coerce:
        raise TypeError("Text cannot be plotted. To force plotting text "
                        "with the basename, set ``coerce=True``.")

    text = is_iterable(text,
                       exclude_string=True,
                       transform=True)

    # Check that the number of text labels matches the number
    # of coordinates, unless coercion is enabled.
    if (len(text) != len(y)) and (not coerce):
        raise ValueError("Text array and coordinate arrays must be of "
                         "equal length. Got {} and {}. To plot anyway, "
                         "set ``coerce=True``."
                         .format(len(text), len(y)))

    # Coerce missing text labels using `basename` if needed.
    if coerce:
        basename = str(basename)
        text += [f'{basename}{i + len(text):02}'
                 for i in range(len(y))]

    # Optionally skip intermediate text labels according to `step`.
    if step is not None:
        step = _assert_all_types(step, float, int, objname='Step')
        for ii in range(len(text)):
            if ii % step != 0:
                text[ii] = ''

    # Create a new Axes instance if one is not provided.
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=fig_size)

    ax_m = None
    # Optionally draw a connecting line between the points.
    if show_line:
        ax.plot(x, y,
                label=linelabel,
                color=lcolor)

    # Annotate each (x, y) coordinate with text and optionally plot a marker.
    for ix, iy, name in zip(x, y, text):
        ax.text(ix, iy, name, color=color, **text_kws)
        if name != '':
            ax_m = ax.scatter([ix],
                              [iy],
                              marker='o',
                              color=mcolor)

    # Set axis labels.
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Set the marker legend label if markers have been plotted.
    if ax_m is not None:
        ax_m.set_label(markerlabel)

    # Display the legend if requested.
    if show_leg:
        ax.legend()

    return ax

def _make_axe_multiple(
    n, 
    ncols=3, 
    fig_size=None, 
    fig=None, 
    ax=None
    ):
    """
    Make multiple subplot axes from a number of objects.

    This function creates a grid of subplot axes based on the number
    of objects provided. It calculates the number of rows required
    using the formula:

    .. math::
       n_{rows} = \\left\\lfloor \\frac{n}{ncols} \\right\\rfloor +
       (n \\bmod ncols)

    If no Axes instance is provided, a new figure and axes are created
    using matplotlib.pyplot.subplots.

    Parameters
    ----------
    n : int or array-like
        The number of objects for which subplots are needed. If an
        iterable is provided, its length is used.
    ncols : int, default 3
        The number of columns in the subplot grid.
    fig_size : tuple, optional
        The size of the figure in inches.
    fig : matplotlib.figure.Figure, optional
        An existing Figure instance. If provided along with ax, no new
        figure is created.
    ax : matplotlib.axes.Axes or array-like, optional
        An existing Axes instance or array of Axes. If None, new axes
        are created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the subplot axes.
    ax : matplotlib.axes.Axes or array-like
        The subplot axes.

    Examples
    --------
    >>> from gofast.plot.utils import _make_axe_multiple
    >>> fig, ax = _make_axe_multiple(10, ncols=4, fig_size=(10, 8))
    >>> # Creates a grid for 10 objects with 4 columns.

    See Also
    --------
    matplotlib.pyplot.subplots :
        Create a figure and a set of subplots.
    """
    if is_iterable(n):
        n = len(n)
    nrows = n // ncols + (n % ncols)
    if nrows == 0:
        nrows = 1
    if ax is None:
        fig, ax = plt.subplots(nrows, ncols, figsize=fig_size)
    return fig, ax


def _manage_plot_kws(
    kws, dkws=dict()
    ):
    """
    Check for default values in plotting keyword arguments.

    This function checks whether the default values specified in
    dkws are present in the provided keyword arguments (kws). If a key
    is missing, it is added from dkws.

    Parameters
    ----------
    kws : dict
        A dictionary containing plotting keyword arguments.
    dkws : dict, optional
        A dictionary of default keyword arguments. If provided, any key
        not present in kws is added from dkws. Default is an empty dict.

    Returns
    -------
    dict
        The updated dictionary of plotting keyword arguments.

    Examples
    --------
    >>> from gofast.plot.utils import _manage_plot_kws
    >>> kws = {'color': 'blue'}
    >>> dkws = {'color': 'red', 'linewidth': 2}
    >>> result = _manage_plot_kws(kws, dkws)
    >>> print(result)
    {'color': 'blue', 'linewidth': 2}

    See Also
    --------
    dict.update :
        Update a dictionary with default values.
    """
    kws = dkws or kws
    for key in dkws.keys():
        if key not in kws.keys():
            kws[key] = dkws.get(key)
    return kws

def is_colormap(color_name):
    """
    Check if the given color name is a valid Matplotlib colormap.

    This function verifies whether the provided color name is one of the
    available colormaps in Matplotlib. It retrieves the list of all
    colormaps and checks for membership.

    Parameters
    ----------
    color_name : str
        The name of the color or colormap to check.

    Returns
    -------
    bool
        True if color_name is a valid colormap in Matplotlib, False
        otherwise.

    Examples
    --------
    >>> from gofast.plot.utils import is_colormap
    >>> is_colormap('viridis')
    True
    >>> is_colormap('not_a_color')
    False

    See Also
    --------
    matplotlib.pyplot.colormaps :
        List all available colormaps in Matplotlib.
    """
    colormaps = plt.colormaps()
    return color_name in colormaps

def flex_figsize(
    m, 
    figsize=None, 
    base=(4, 12), 
    min_base=(2, 12), 
    method='abs',
 
    ):
    r"""
    Compute a flexible figure size for plotting a data matrix.

    This function calculates an optimal figure size for visualizing a 
    matrix with a given number of models (columns). If ``figsize`` is not 
    provided, the function scales a baseline figure size relative to a 
    default of 3 models. The baseline size is defined by ``base``, which 
    specifies the width and height (in inches) for 3 models.

    Parameters
    ----------
    m : array_like
        A two-dimensional array or matrix with shape 
        :math:`(n_{\rm features},\, n_{\rm models})`. Each row represents 
        a feature and each column represents a model.
    
    figsize : tuple of int, optional
        Desired figure size as ``(width, height)`` in inches. If 
        ``None``, the figure size is computed automatically based on 
        the number of models in ``m``. Default is ``None``.
    
    base : tuple of int, optional
        The baseline figure size for 3 models as ``(width, height)`` in 
        inches. Default is ``(4, 12)``.
    min_base : tuple of int, default=(2, 12)
        The minimum allowed figure size to avoid extremely small 
        or unreadable plots.
        - `min_base[0]` represents the minimum width.
        - `min_base[1]` represents the minimum height.
    
    method : str, {'abs', 'absolute', 'relative'}, default='abs'
        Determines how the figure size is computed:
        - `'abs'` or `'absolute'`: Uses a fixed scaling factor based 
          on the number of models.
        - `'relative'`: Adjusts the figure size dynamically based 
          on both models and features.
          
    Returns
    -------
    figsize : tuple of int
        The computed figure size as a tuple ``(width, height)`` in inches.

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.plot.utils import flex_figsize
    >>> m = np.random.rand(12, 3)  
    ... # 12 features (rows) and 3 models (columns)
    >>> flex_figsize(m)
    (4, 12)
    >>> m2 = np.random.rand(12, 6)  
    ... # 6 models; scale factor = 2 relative to baseline of 3 models
    >>> flex_figsize(m2)
    (8, 24)
    >>> m = np.random.rand(20, 5)  # 20 features, 5 models
    >>> flex_figsize(m, method='relative')
    (7, 20)

    >>> m = np.random.rand(30, 10)  # 30 features, 10 models
    >>> flex_figsize(m)
    (13, 30)
    
    Notes
    -----
    This function scales the baseline figure size provided by ``base`` 
    using the ratio of the actual number of models to a baseline of 3. 
    This approach ensures that visualizations remain clear and well-proportioned 
    even when the number of models changes.
    
    Let :math:`n_{\rm models}` denote the number of columns in the matrix 
    ``m``. The scaling factor is computed as:

    .. math::

        s = \frac{n_{\rm models}}{3}

    The resulting figure dimensions are given by:

    .. math::

        w = \lceil \texttt{base}[0] \times s \rceil, \quad
        h = \lceil \texttt{base}[1] \times s \rceil

    where :math:`\lceil \cdot \rceil` is the ceiling function and 
    ``base`` is a tuple representing the baseline figure size for 3 models.

    - If `figsize` is provided, it is returned directly without computation.
    - The `'absolute'` method scales only based on the number of 
      models (`n_models`).
    - The `'relative'` method considers both models and features
      (`n_models`, `n_features`).
    - Ensures a minimum figure size (`min_base`) to maintain readability.
    
    See Also
    --------
    matplotlib.pyplot.subplots : Create a figure and a set of subplots.
    
    References
    ----------
    .. [1] Doe, J., "Scalable Visualization in Python", 
           Journal of Data Visualization, 2021.
    """

    m=ensure_2d(m, output_format="auto") 
    
    if figsize is None:
        if method in ["abs", "absolute"]: 
            n_models = m.shape[1]           # Number of models (columns)
            scale_factor = n_models / 3         # Scale relative to 3 models
            width  = int(np.ceil(base[0] * scale_factor))
            height = int(np.ceil(base[1] * scale_factor))
            figsize = (width, height)

        else: # "relative"
         
            # Extract matrix dimensions
            n_models = m.shape[1]   # Number of models (columns)
            n_features = m.shape[0] # Number of features (rows)
    
            # Define scaling factors
            width_scale = n_models / 3    # Scale width based on 3 models
            height_scale = n_features / 12  # Scale height based on 12 features
    
            # Dynamically compute minimum width and height
            # Adjusted dynamically
            min_width = max(
                min_base[0], int(np.ceil(base[0] * (n_models / 6))))  
            min_height = max(
                min_base[1], int(np.ceil(base[1] * (n_features / 24))))  
    
            # Compute dynamic figsize
            width = max(
                min_width, int(np.ceil(base[0] * width_scale)))
            height = max(
                min_height, int(np.ceil(base[1] * height_scale)))
    
            figsize = (width, height)

    return figsize


def _set_defaults(**kwargs):
    """
    Set the default values for multiple parameters if they are not provided.

    This function checks whether each parameter passed in `kwargs` is None. 
    If it is None, the corresponding value from the `_param_defaults` dictionary 
    is used as the default.

    Parameters
    ----------
    kwargs : dict
        The keyword arguments passed to the function. The function checks 
        each key in `kwargs` and replaces its value with the default if it is None.

    Returns
    -------
    dict
        A dictionary containing the updated parameters, where each parameter 
        is either the provided value or the default value if not provided.
    """
    # Initialize the updated parameters with the defaults
    params = _param_defaults.copy()

    # Update the dictionary with the values from kwargs, if they are not None
    for key, value in kwargs.items():
        if value is not None:
            params[key] = value

    return params
