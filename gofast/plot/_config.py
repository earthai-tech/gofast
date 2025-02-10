# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Plot configuration module for gofast library.
"""
import matplotlib.pyplot as plt
from ..api.property import BasePlot, BaseClass  

class PlotConfig (BaseClass):
    """
    Configuration class managing settings for plot dependencies
    and automated saving logic in the gofast plotting system.

    This class provides both global toggles for installing missing
    dependencies (either via pip or conda) and a mechanism to
    automatically determine if plots should be saved.

    Attributes
    ----------
    install_dependencies : bool
        If ``True``, enables an automatic installation of
        missing packages needed by plot functions. Defaults
        to ``False``.
    use_conda : bool
        If ``True``, uses conda for installation when
        dependencies are missing. Otherwise, uses pip.
        Applies only if ``install_dependencies`` is True.
        Defaults to ``False``.
    auto_save : bool
        If ``True``, automatic saving of plots is enforced
        when using :meth:`AUTOSAVE`. If ``False``, no saving
        is performed by default, unless explicitly specified.

    Examples
    --------
    >>> from gofast.plot._config import PlotConfig
    >>> PlotConfig.install_dependencies = True
    >>> PlotConfig.use_conda = False
    >>> PlotConfig.auto_save = True

    In combination with a decorator or default params:

    >>> @default_params_plot(
    ...     savefig=PlotConfig.AUTOSAVE("ranking_plot.png"),
    ...     figsize=(4, 12),
    ...     dpi=300
    ... )
    ... def plot_ranking(...):
    ...     pass

    If ``PlotConfig.auto_save`` is True, the function
    receives ``"ranking_plot.png"`` as the save path.
    Otherwise, it gets ``None``.

    Notes
    -----
    This pattern can be extended to handle more advanced
    file naming strategies or environment checks if needed.
    """

    install_dependencies = False
    use_conda = False
    auto_save = False

    @classmethod
    def AUTOSAVE(cls, filename: str) -> str:
        """
        Conditionally returns a filename if automatic saving
        is enabled, or returns None otherwise.

        Parameters
        ----------
        filename : str
            The intended file path for saving the plot.

        Returns
        -------
        str or None
            - If ``PlotConfig.auto_save=True``, returns
              the given ``filename``.
            - If ``PlotConfig.auto_save=False``, returns
              ``None``.

        Examples
        --------
        >>> PlotConfig.auto_save = True
        >>> PlotConfig.AUTOSAVE("myplot.png")
        'myplot.png'

        >>> PlotConfig.auto_save = False
        >>> PlotConfig.AUTOSAVE("myplot.png")
        None
        """
        if cls.auto_save:
            return filename
        return None

class PlotUtils(BasePlot):
    def __init__(self, **kwargs):
        """
        Initialize the plotting utility class which extends the BasePlot class,
        allowing for custom plot configurations.

        Parameters
        ----------
        **kwargs : dict
            Arbitrary keyword arguments that are passed to the
            BasePlot's constructor.
        """
        super().__init__(**kwargs)
    
    def save(self, fig):
        """
        Save the figure with the specified attributes if `savefig`
        is set; otherwise, display the figure on screen.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure object to be saved or displayed.

        Notes
        -----
        - If `savefig` is not None, the figure is saved using the
          path specified in `savefig`.
        - The figure is saved with the resolution specified by `fig_dpi`,
          and the orientation can be set with `fig_orientation`.
        - If `savefig` is None, the figure will be displayed using
          `plt.show()` and not saved.

        Examples
        --------
        >>> from matplotlib import pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> ax.plot([0, 1], [0, 1])
        >>> plot_utils = PlotUtils(figsize=(10, 5), savefig='plot.png',
        ...                        fig_dpi=300, fig_orientation='landscape')
        >>> plot_utils.save(fig)  # This will save the figure to 'plot.png'
        """
        if self.savefig is not None:
            fig.savefig(self.savefig, dpi=self.fig_dpi,
                        bbox_inches='tight', 
                        orientation=self.fig_orientation)
            plt.close()
        else:
            plt.show()
    
def create_custom_plotter(base_plotter):
    """
    Dynamically creates a custom plotter class that includes additional 
    methods and attributes from an existing plotter object.

    Parameters:
        base_plotter: An instance of a plotter class from which to inherit 
        properties.

    Returns:
        A new plotting class with combined features from `PlotUtils and `base_plotter`.
    """
    # Creating a dynamic type combining BasePlot and properties from the given plotter
    plot_class = type('CustomPlotter', (PlotUtils,), {**base_plotter.__dict__})
    
    # Update the docstring for the new class
    plot_class.__doc__ = """\
    Custom plotting class that extends :class:`~gofast.api.properties.BasePlot` 
    with dynamic properties.

    Inherits all matplotlib figure properties, allowing modification via 
    object attributes. For example:
        >>> plot_obj = CustomPlotter()
        >>> plot_obj.ls = '-.'  # Set line style
        >>> plot_obj.fig_size = (7, 5)  # Set figure size
        >>> plot_obj.lw = 7  # Set linewidth

    See also:
        Refer to :class:`~gofast.api.property.BasePlot` for details 
        on adjustable parameters.
    """
    
    return plot_class

# #################################################
# Creating a dynamic type combining BasePlot and 
# properties from the given plotter
pobj = type('DynamicPlotter', (PlotUtils,), {})

pobj.__doc__ = """\
Dynamic plotting class that extends :class:`~gofast.api.properties.BasePlot` 
with dynamic properties.

Inherits all matplotlib figure properties, allowing modification via 
object attributes. For example:
    >>> plot_obj = DynamicPlotter()
    >>> plot_obj.ls = '-.'  # Set line style
    >>> plot_obj.fig_size = (7, 5)  # Set figure size
    >>> plot_obj.lw = 7  # Set linewidth

See also:
    Refer to :class:`~gofast.api.property.BasePlot` for details 
    on adjustable parameters.
"""