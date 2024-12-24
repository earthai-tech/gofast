# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Plotting utilities for managing color schemes, colormaps, and styles.
Includes functions for color conversion, alpha value generation, and 
ensuring compatibility with visualization libraries.
"""
from __future__ import print_function
import itertools
import inspect 
from functools import wraps 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from ..compat.scipy import ensure_scipy_compatibility

__all__=[ 
    'default_params_plot', 
    'ensure_visualization_compatibility',
    'closest_color',
    'colors_to_names',
    'decompose_colormap',
    'generate_alpha_values',
    'generate_mpl_styles',
    'get_colors_and_alphas',
    'hex_to_rgb', 
    ]

def default_params_plot(
    savefig=None, 
    close=False, 
    fig_size=(10, 6), 
    title=None, 
    dpi=None, 
    new_ax=False, 
    **extra_defaults
):
    """
    Decorator to manage default plotting parameters for Matplotlib functions.
    
    This decorator automates the handling of common plotting parameters, allowing
    functions to focus on plotting logic while ensuring consistency and reducing
    repetition. It provides defaults for saving figures, figure size, titles,
    resolution, and axis management, among others.
    
    Parameters
    ----------
    savefig : `str` or `None`, optional
        **Default:** ``None``
        
        Path to save the figure. If a string is provided, the figure will be saved
        to the specified path using `matplotlib.figure.Figure.savefig`. If set to
        `None`, the figure will not be saved unless overridden during the function
        call.
    
    close : `bool`, optional
        **Default:** ``False``
        
        Determines whether to automatically close the figure after processing.
        Setting this to `True` helps manage memory by closing figures that are no
        longer needed.
    
    fig_size : `tuple` of `float`, optional
        **Default:** ``(10, 6)``
        
        Specifies the default size of the figure in inches as `(width, height)`.
        This can be overridden by passing `fig_size` or `figsize` during the
        function call.
    
    title : `str` or `None`, optional
        **Default:** ``None``
        
        Assigns a default title to the plot. If set to `None`, no title will be
        added unless specified during the function call.
    
    dpi : `int` or `None`, optional
        **Default:** ``None``
        
        Sets the resolution of the figure in dots per inch. If `None`, Matplotlib's
        default DPI is used.
    
    new_ax : `bool`, optional
        **Default:** ``False``
        
        Determines whether to create a new Axes object for the plot. If `True`,
        a new Axes is created and passed to the decorated function. If `False`,
        the decorator will attempt to use an existing Axes object provided as an
        argument.
    
    extra_defaults: `dict`, optional
        Additional keyword arguments to set as default parameters. These are applied
        only if the decorated function accepts them.
    
    Methods
    -------
    No public methods. This decorator modifies the behavior of the decorated
    function by injecting default parameters and managing figure creation, saving,
    and closing.
    
    Formulation
    -----------
    The decorator adjusts the parameters of the decorated function based on the
    following logic:
    
    .. math::
        \text{kwargs}[p] = 
        \begin{cases} 
          \text{user\_value} & \text{if user provides } p \\
          \text{default\_value} & \text{otherwise}
        \end{cases}
    
    Where `p` represents each plotting parameter such as `savefig`, `fig_size`,
    `title`, etc.
    
    This ensures that each parameter is either set to the user-provided value or
    falls back to the decorator's default.
    
    Examples
    --------
    >>> from gofast.core.handlers import default_params_plot
    >>> import matplotlib.pyplot as plt
    
    >>> @default_params_plot(
    ...     savefig='my_plot.png', 
    ...     close=True, 
    ...     fig_size=(8, 6), 
    ...     title='My Plot', 
    ...     dpi=300, 
    ...     new_ax=True
    ... )
    ... def plot_function(ax, x, y, savefig=None):
    ...     ax.plot(x, y)
    ...     ax.set_xlabel('X axis')
    ...     ax.set_ylabel('Y axis')
    
    >>> x = [0, 1, 2, 3]
    >>> y = [0, 1, 4, 9]
    
    >>> # Use decorator's default savefig
    >>> plot_function(ax=None, x=x, y=y)
    {'ax': <AxesSubplot:>, 'x': [0, 1, 2, 3], 'y': [0, 1, 4, 9],
     'savefig': 'my_plot.png'}
    Figure saved as my_plot.png
    Figure closed.
    
    >>> # Override savefig with a different filename
    >>> plot_function(ax=None, x=x, y=y, savefig='override_plot.png')
    {'ax': <AxesSubplot:>, 'x': [0, 1, 2, 3], 'y': [0, 1, 4, 9],
     'savefig': 'override_plot.png'}
    Figure saved as override_plot.png
    Figure closed.
    
    >>> # Disable saving by explicitly setting savefig to None
    >>> plot_function(ax=None, x=x, y=y, savefig=None)
    {'ax': <AxesSubplot:>, 'x': [0, 1, 2, 3], 'y': [0, 1, 4, 9],
     'savefig': 'my_plot.png'}
    Figure saved as my_plot.png
    Figure closed.
    
    Notes
    -----
    - The decorator inspects the signature of the decorated function to ensure
      that only relevant keyword arguments are passed, preventing unexpected
      keyword argument errors.
      
    - If `new_ax` is set to `True`, a new figure and Axes are created regardless of
      whether an Axes is passed to the function. This is useful for generating
      standalone plots.
    
    - When `close` is `True`, the figure is closed after saving to free up
      system resources, which is particularly beneficial when generating multiple
      plots in a loop.
    
    - The decorator can handle both `fig_size` and `figsize` parameters, providing
      flexibility in how figure sizes are specified.
    
    See Also
    --------
    `matplotlib.pyplot.figure` : Create a new figure.
    `matplotlib.axes.Axes` : Class representing an Axes in Matplotlib.
    
    References
    ----------
    .. [1] Hunter, J. D. (2007). *Matplotlib: A 2D Graphics Environment*.
           Computing in Science & Engineering, 9(3), 90–95.
    .. [2] P. (n.d.). *Python Decorators*. Retrieved from https://realpython.com/primer-on-python-decorators/
    """

    def decorator(func):
        # Inspect the signature of the decorated function
        sig = inspect.signature(func)
        func_params = sig.parameters

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Determine which plot-related kwargs the function accepts
            accepted_kwargs = {
                key for key in ['fig_size', 'figsize', 'dpi', 'title', 'savefig'] 
                if key in func_params
            }
            # Include any extra defaults that are accepted
            accepted_kwargs.update(k for k in extra_defaults if k in func_params)

            # Handle fig_size or figsize
            if 'fig_size' in accepted_kwargs or 'figsize' in accepted_kwargs:
                fig_size_param = kwargs.get('fig_size', kwargs.get('figsize', fig_size))
                if 'fig_size' in func_params:
                    kwargs['fig_size'] = fig_size_param
                if 'figsize' in func_params:
                    kwargs['figsize'] = fig_size_param
            else:
                fig_size_param = fig_size  # Default value if not accepted

            # Handle dpi
            if 'dpi' in accepted_kwargs:
                kwargs['dpi'] = kwargs.get('dpi', dpi)
            else:
                dpi_ = kwargs.get('dpi', dpi)

            # Handle title
            if 'title' in accepted_kwargs:
                kwargs['title'] = kwargs.get('title', title)

            # Handle savefig with special logic
            if 'savefig' in accepted_kwargs:
                if 'savefig' in kwargs and kwargs['savefig'] is not None:
                    savefig_param = kwargs['savefig']
                else:
                    savefig_param = savefig
                kwargs['savefig'] = savefig_param
            else:
                savefig_param = None  # Ensure it's defined

            # Handle any extra default parameters
            for key, value in extra_defaults.items():
                if key in accepted_kwargs:
                    kwargs[key] = kwargs.get(key, value)

            fig = None
            ax = None

            # Create new axes if requested
            if new_ax:
                fig, ax = plt.subplots(
                    figsize=fig_size_param, 
                    dpi=kwargs.get('dpi', dpi_)
                )
                if 'ax' in func_params:
                    kwargs['ax'] = ax
            else:
                # Attempt to retrieve ax from args or kwargs
                if args:
                    potential_ax = args[0]
                    if isinstance(potential_ax, plt.Axes):
                        ax = potential_ax
                if not ax and 'ax' in kwargs:
                    ax = kwargs['ax']
                if ax is None and 'ax' in func_params:
                    fig, ax = plt.subplots(
                        figsize=fig_size_param, 
                        dpi=kwargs.get('dpi', dpi)
                    )
                    kwargs['ax'] = ax

            # Call the decorated function
            result = func(*args, **kwargs)

            # If title is to be set and accepted by the function
            if title and 'title' in accepted_kwargs and kwargs.get('title'):
                if isinstance(result, plt.Axes):
                    result.set_title(kwargs['title'])
                elif isinstance(result, plt.Figure):
                    result.suptitle(kwargs['title'])
                    fig = result
                else:
                    # Attempt to get the current Axes and set the title
                    ax = plt.gca()
                    ax.set_title(kwargs['title'])
                    fig = ax.get_figure()
            
            # Save figure if savefig is specified and accepted
            if savefig_param and 'savefig' in accepted_kwargs:
                if isinstance(result, plt.Figure):
                    fig = result
                elif isinstance(result, plt.Axes):
                    fig = result.get_figure()
                if fig is None:
                    fig = plt.gcf()
                fig.savefig(
                    savefig_param, 
                    dpi=kwargs.get('dpi', dpi)
                )
                print(f"Figure saved as {savefig_param}")  # Confirmation
            
            # Close figure if close is True
            if close and fig is not None:
                plt.close(fig)
                print("Figure closed.")  # Confirmation

            return result

        return wrapper
    
    return decorator
def ensure_visualization_compatibility(
        result, as_frame=False, view=False, func_name=None,
        verbose=0, allow_singleton_view=False
        ):
    """
    Evaluates and prepares the result for visualization, adjusting its format
    if necessary and determining whether visualization is feasible based on
    given parameters. If the conditions for visualization are not met, 
    especially for singleton values, it can modify the view flag accordingly.

    Parameters
    ----------
    result : iterable or any
        The result to be checked and potentially modified for visualization.
    as_frame : bool, optional
        If True, the result is intended for frame-based visualization, which 
        may prevent conversion of singleton iterables to a float. Defaults to False.
    view : bool, optional
        Flag indicating whether visualization is intended. This function may 
        modify it to False if visualization conditions aren't met. Defaults to False.
    func_name : callable or str, optional
        The name of the function or a callable from which the name can be derived, 
        used in generating verbose messages. Defaults to None.
    verbose : int, optional
        Controls verbosity level. A value greater than 0 enables verbose messages. 
        Defaults to 0.
    allow_singleton_view : bool, optional
        Allows visualization of singleton values if set to True. If False and a 
        singleton value is encountered, `view` is set to False. Defaults to False.

    Returns
    -------
    tuple
        A tuple containing the potentially modified result and the updated view flag.
        The result is modified if it's a singleton iterable and conditions require it.
        The view flag is updated based on the allowability of visualization.

    Examples
    --------
    >>> from gofast.core.plot_manager import ensure_visualization_compatibility
    >>> result = [100.0]
    >>> modified_result, can_view = ensure_visualization_compatibility(
    ...     result, as_frame=False, view=True, verbose=1, allow_singleton_view=False)
    Visualization is not allowed for singleton value.
    >>> print(modified_result, can_view)
    100.0 False

    >>> result = [[100.0]]
    >>> modified_result, can_view = ensure_visualization_compatibility(
    ...     result, as_frame=True, verbose=1)
    >>> print(modified_result, can_view)
    [[100.0]] True
    """
    if hasattr(result, '__iter__') and len(
            result) == 1 and not allow_singleton_view:
        if not as_frame:
            # Attempt to convert to float value
            try:
                result = float(result[0])
            except ValueError:
                pass  # Keep the result as is if conversion fails

        if view: 
            if verbose > 0:
                # Construct a user-friendly verbose message
                func_name_str = f"{func_name.__name__} visualization" if callable(
                    func_name) else "Visualization"
                # Ensure the first letter is capitalized
                message_start = func_name_str[0].upper() + func_name_str[1:]  
                print(f"{message_start} is not allowed for singleton value.")
            view =False 
    return result, view 

def generate_mpl_styles(n, prop='color'):
    """
    Generates a list of matplotlib property items (colors, markers, or line styles)
    to accommodate a specified number of samples.

    Parameters
    ----------
    n : int
        Number of property items needed. It generates a list of property items.
    prop : str, optional
        Name of the property to retrieve. Accepts 'color', 'marker', or 'line'.
        Defaults to 'color'.

    Returns
    -------
    list
        A list of property items with size equal to `n`.

    Raises
    ------
    ValueError
        If the `prop` argument is not one of the accepted property names.

    Examples
    --------
    Generate 10 color properties:

    >>> from gofast.core.plot_manager import generate_mpl_styles
    >>> generate_mpl_styles(10, prop='color')
    ['g', 'gray', 'y', 'blue', 'orange', 'purple', 'lime', 'k', 'cyan', 'magenta']

    Generate 5 marker properties:

    >>> generate_mpl_styles(5, prop='marker')
    ['o', '^', 's', '*', '+']

    Generate 3 line style properties:

    >>> generate_mpl_styles(3, prop='line')
    ['-', '--', '-.']
    """
    import matplotlib as mpl

    D_COLORS = ["g", "gray", "y", "blue", "orange", "purple", "lime",
                "k", "cyan", "magenta"]
    D_MARKERS = ["o", "^", "s", "*", "+", "x", "D", "H"]
    D_STYLES = ["-", "--", "-.", ":"]
    
    n = int(n)  # Ensure n is an integer
    prop = prop.lower().strip().replace('s', '')  # Normalize the prop string
    if prop not in ('color', 'marker', 'line'):
        raise ValueError(f"Property '{prop}' is not available."
                         " Expect 'color', 'marker', or 'line'.")

    # Mapping property types to their corresponding lists
    properties_map = {
        'color': D_COLORS,
        'marker': D_MARKERS + list(mpl.lines.Line2D.markers.keys()),
        'line': D_STYLES
    }

    # Retrieve the specific list of properties based on the prop parameter
    properties_list = properties_map[prop]

    # Generate the required number of properties, repeating the list if necessary
    repeated_properties = list(itertools.chain(*itertools.repeat(properties_list, (
        n + len(properties_list) - 1) // len(properties_list))))[:n]

    return repeated_properties

def generate_alpha_values(n, increase=True, start=0.1, end=1.0, epsilon=1e-10):
    """
    Generates a list of alpha (transparency) values that either increase or 
    decrease gradually to fit the number of property items.
    
    Incorporates an epsilon to safeguard against division by zero.
    
    Parameters
    ----------
    n : int
        The number of alpha values to generate.
    increase : bool, optional
        If True, the alpha values will increase; if False, they will decrease.
        Defaults to True.
    start : float, optional
        The starting alpha value. Defaults to 0.1.
    end : float, optional
        The ending alpha value. Defaults to 1.0.
    epsilon : float, optional
        Small value to avert division by zero. Defaults to 1e-10.
        
    Returns
    -------
    list
        A list of alpha values of length `n`.
    
    Examples
    --------
    >>> from gofast.core.plot_manager import generate_alpha_values
    >>> generate_alpha_values(5, increase=True)
    [0.1, 0.325, 0.55, 0.775, 1.0]
    
    >>> generate_alpha_values(5, increase=False)
    [1.0, 0.775, 0.55, 0.325, 0.1]
    """
    if not 0 <= start <= 1 or not 0 <= end <= 1:
        raise ValueError("Alpha values must be between 0 and 1.")

    # Calculate the alpha values, utilizing epsilon in the denominator 
    # to prevent division by zero
    alphas = [start + (end - start) * i / max(n - 1, epsilon) for i in range(n)]
    
    if not increase:
        alphas.reverse() # or alphas[::-1] creates new list
    
    return alphas

def decompose_colormap(cmap_name, n_colors=5):
    """
    Decomposes a colormap into a list of individual colors.

    Parameters
    ----------
    cmap_name : str
        The name of the colormap to decompose.
    n_colors : int, default=5
        The number of colors to extract from the colormap.

    Returns
    -------
    list
        A list of RGBA color values from the colormap.

    Examples
    --------
    >>> colors = decompose_colormap('viridis', 5)
    >>> print(colors)
    [(0.267004, 0.004874, 0.329415, 1.0), ..., (0.993248, 0.906157, 0.143936, 1.0)]
    """
    cmap = plt.cm.get_cmap(cmap_name, n_colors)
    colors = [cmap(i) for i in range(cmap.N)]
    return colors


def colors_to_names(*colors, consider_alpha=False, ignore_color_names=False, 
                    color_space='rgb', error='ignore'):
    """
    Converts a sequence of RGB or RGBA colors to their closest named color 
    strings. 
    
    Optionally ignores input color names and handles colors in specified 
    color spaces.
    
    Parameters
    ----------
    *colors : tuple
        A variable number of RGB(A) color tuples or color name strings.
    consider_alpha : bool, optional
        If True, the alpha channel in RGBA colors is considered in the conversion
        process. Defaults to False.
    ignore_color_names : bool, optional
        If True, input strings that are already color names are ignored. 
        Defaults to False.
    color_space : str, optional
        Specifies the color space ('rgb' or 'lab') used for color comparison. 
        Defaults to 'rgb'.
    error : str, optional
        Error handling strategy when encountering invalid colors. If 'raise', 
        errors are raised. Otherwise, errors are ignored. Defaults to 'ignore'.
    
    Returns
    -------
    list
        A list of color name strings corresponding to the input colors.

    Examples
    --------
    >>> from gofast.core.plot_manager import colors_to_names
    >>> colors_to_names((0.267004, 0.004874, 0.329415, 1.0), 
                        (0.127568, 0.566949, 0.550556, 1.0), 
                        consider_alpha=True)
    ['rebeccapurple', 'mediumseagreen']
    
    >>> colors_to_names('rebeccapurple', ignore_color_names=True)
    []
    
    >>> colors_to_names((123, 234, 45), color_space='lab', error='raise')
    ['limegreen']
    """
    color_names = []
    for color in colors:
        if isinstance(color, str):
            if ignore_color_names:
                continue
            else:
                color_names.append(color)  # String color name is found
        else:
            try:
                color_name = closest_color(color, consider_alpha=consider_alpha,
                                           color_space=color_space)
                color_names.append(color_name)
            except Exception as e:
                if error == 'raise':
                    raise e
                
    return color_names

def closest_color(rgb_color, consider_alpha=False, color_space='rgb'):
    """
    Finds the closest named CSS4 color to the given RGB(A) color in the specified
    color space, optionally considering the alpha channel.

    Parameters
    ----------
    rgb_color : tuple
        A tuple representing the RGB(A) color.
    consider_alpha : bool, optional
        Whether to include the alpha channel in the color closeness calculation.
        Defaults to False.
    color_space : str, optional
        The color space to use when computing color closeness. Can be 'rgb' or 'lab'.
        Defaults to 'rgb'.

    Returns
    -------
    str
        The name of the closest CSS4 color.

    Raises
    ------
    ValueError
        If an invalid color space is specified.

    Examples
    --------
    Find the closest named color to a given RGB color:

    >>> from gofast.core.plot_manager import closest_color
    >>> closest_color((123, 234, 45))
    'forestgreen'

    Find the closest named color to a given RGBA color, considering the alpha:

    >>> closest_color((123, 234, 45, 0.5), consider_alpha=True)
    'forestgreen'

    Find the closest named color in LAB color space (more perceptually uniform):

    >>> closest_color((123, 234, 45), color_space='lab')
    'limegreen'
    """
    if color_space not in ['rgb', 'lab']:
        raise ValueError(f"Invalid color space '{color_space}'. Choose 'rgb' or 'lab'.")

    if ensure_scipy_compatibility(): 
        from scipy.spatial import distance 
    # Adjust input color based on consider_alpha flag
    
    # Include alpha channel if consider_alpha is True
    input_color = rgb_color[:3 + consider_alpha]  

    # Convert the color to the chosen color space if needed
    if color_space == 'lab':
        # LAB conversion ignores alpha
        input_color = mcolors.rgb_to_lab(input_color[:3])  
        color_comparator = lambda color: distance.euclidean(
            mcolors.rgb_to_lab(color[:3]), input_color)
    else:  # RGB or RGBA
        color_comparator = lambda color: distance.euclidean(
            color[:len(input_color)], input_color)

    # Compute the closeness of each named color to the given color
    closest_name = None
    min_dist = float('inf')
    for name, hex_color in mcolors.CSS4_COLORS.items():
        # Adjust based on input_color length
        named_color = mcolors.to_rgba(hex_color)[:len(input_color)]  
        dist = color_comparator(named_color)
        if dist < min_dist:
            min_dist = dist
            closest_name = name

    return closest_name
   
   
def get_colors_and_alphas(
    count, 
    cmap=None, 
    alpha_direction='decrease', 
    start_alpha=0.1,
    end_alpha=1.0, 
    convert_to_named_color=True, 
    single_color_as_string=False,
    consider_alpha=False, 
    ignore_color_names=False, 
    color_space='rgb', 
    error="ignore"
):
    """
    Generates a sequence of color codes and alpha (transparency) values. 
    
    Colors can be sourced from a specified Matplotlib colormap or generated 
    using predefined styles. Alpha values can be arranged in ascending or 
    descending order to create a gradient effect.

    The function also supports converting color tuples to named colors and 
    allows for customizing the transparency gradient. Additionally, if only 
    one color is generated, it can return that color directly as a string
    rather than wrapped in a list, for convenience in functions that expect a
    single color string.

    Parameters
    ----------
    count : int or iterable
        Specifies the number of colors and alpha values to generate. If an iterable 
        is provided, its length determines the number of colors and alphas.
    cmap : str, optional
        The name of a Matplotlib colormap to generate colors. If None, colors are
        generated using predefined styles. Defaults to ``None``.
    alpha_direction : str, optional
        Direction to arrange alpha values for creating a gradient effect. ``increase``
        for ascending order, ``decrease`` for descending. Defaults to ``decrease``.
    start_alpha : float, optional
        The starting alpha value (transparency) in the gradient, between 0 (fully
        transparent) and 1 (fully opaque). Defaults to ``0.1``.
    end_alpha : float, optional
        The ending alpha value in the gradient, between 0 and 1. 
        Defaults to ``1.0``.
    convert_to_named_color : bool, optional
        Converts color tuples to the nearest Matplotlib named color. This 
        conversion applies when exactly one color is generated. 
        Defaults to ``True``.
    single_color_as_string : bool, optional
        If True and only one color is generated, returns the color as a string 
        instead of a list. Useful for functions expecting a single color string.
        Defaults to ``False``.
    consider_alpha : bool, optional
        Includes the alpha channel in the conversion process to named colors.
        Applicable only when `convert_to_named_color` is True. This is helpful
        when a human-readable color name is preferred over RGB values.
        Defaults to ``False``.
    ignore_color_names : bool, optional
        When True, any input color names (str) are ignored during conversion 
        to named colors. Useful to exclude specific colors from conversion. 
        Defaults to ``False``.
    color_space : str, optional
        The color space used for computing the closeness of colors. Can be 
        ``rgb`` for RGB color space or ``lab`` for LAB color space, which is more 
        perceptually uniform. Defaults to ``rgb``.
    error : str, optional
        Controls the error handling strategy when an invalid color is 
        encountered during the conversion process. ``raise`` will throw an error,
        while ``ignore`` will proceed without error. Defaults to ``ignore``.

    Returns
    -------
    tuple
        A tuple containing either a list of color codes (RGBA or named color strings) 
        and a corresponding list of alpha values, or a single color code and alpha 
        value if `single_color_as_string` is True and only one color is generated.

    Examples
    --------
    Generate 3 random colors with decreasing alpha values:

    >>> get_colors_and_alphas(3)
    (['#1f77b4', '#ff7f0e', '#2ca02c'], [1.0, 0.55, 0.1])

    Generate 4 colors from the 'viridis' colormap with increasing alpha values:

    >>> get_colors_and_alphas(4, cmap='viridis', alpha_direction='increase')
    (['#440154', '#3b528b', '#21918c', '#5ec962'], [0.1, 0.4, 0.7, 1.0])

    Convert a single generated color to a named color:

    >>> get_colors_and_alphas(1, convert_to_named_color=True)
    ('rebeccapurple', [1.0])

    Get a single color as a string instead of a list:

    >>> get_colors_and_alphas(1, single_color_as_string=True)
    ('#1f77b4', [1.0])
    """
    
    if hasattr(count, '__iter__'):
        count = len(count)
    colors =[]
    if cmap is not None and cmap not in plt.colormaps(): 
        cmap=None 
        colors =[cmap] # add it to generate map
    # Generate colors
    if cmap is not None:
        colors = decompose_colormap(cmap, n_colors=count)
    else:
        colors += generate_mpl_styles(count, prop='color')

    # Generate alphas
    increase = alpha_direction == 'increase'
    alphas = generate_alpha_values(count, increase=increase,
                                   start=start_alpha, end=end_alpha)
    
    # Convert tuple colors to named colors if applicable
    if convert_to_named_color: 
        colors = colors_to_names(
            *colors, consider_alpha= consider_alpha,
            ignore_color_names=ignore_color_names,  
            color_space= color_space, 
            error= error,
            )
    # If a single color is requested as a string, return it directly
    if single_color_as_string and len(colors) == 1:
        if not convert_to_named_color: 
            colors = [closest_color(colors[0], consider_alpha= consider_alpha, 
                                color_space =color_space )]
        colors = colors[0]

    return colors, alphas

def hex_to_rgb (c): 
    """ Convert colors Hexadecimal to RGB """
    c=c.lstrip('#')
    return tuple(int(c[i:i+2], 16) for i in (0, 2, 4)) 
