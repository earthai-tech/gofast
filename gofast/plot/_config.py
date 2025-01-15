
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from ..api.property import BasePlot 

class PlotConfig:
    """
    Configuration class to manage settings for handling plot dependencies
    within the plotting system.

    This class provides static variables to configure global settings for
    automatic installation of dependencies required by the plotting functions,
    and to determine the installation method (pip or conda).

    Attributes
    ----------
    install_dependencies : bool
        If set to True, enables automatic installation of missing packages
        required by plot functions. Defaults to False.
    use_conda : bool
        If set to True, uses conda for automatic installation, otherwise uses pip.
        Applies only if `install_dependencies` is True. Defaults to False.

    Examples
    --------
    To enable automatic installation of dependencies using pip:

    >>> from gofast.plot._config import PlotConfig
    >>> PlotConfig.install_dependencies = True
    >>> PlotConfig.use_conda = False

    To enable automatic installation using conda:

    >>> PlotConfig.install_dependencies = True
    >>> PlotConfig.use_conda = True
    """
    install_dependencies = False
    use_conda = False

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