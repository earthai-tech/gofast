# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Time-Series Plots  
"""
from math import pi
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns

from ..api.property import BasePlot 
from ..exceptions import NotFittedError 
from ..tools.baseutils import smart_rotation  
from ..tools.coreutils import format_to_datetime
from ..tools.funcutils import ensure_pkg 
from ..tools.validator import is_time_series, build_data_if 

 
class TimeSeriesPlotter (BasePlot) :
    def __init__(self,  ax =None, figsize = (10, 6), fontsize=12,  **kws):
        super().__init__(fig_size = figsize , font_size = fontsize, **kws) 
        self.ax = ax 
        self.figsize= figsize 
        self.fontsize= fontsize 

        
    def fit(self, data, date_col=None, value_col=None, **fit_params):
        """
        Fit the TimeSeriesPlotter with a time series dataset, preparing it for
        further plotting and analysis. The method adjusts the DataFrame to
        ensure that date-related data is recognized and properly formatted
        as datetime objects within pandas.
    
        Parameters
        ----------
        data : pandas.DataFrame
            The time series dataset to be processed.
        date_col : str, optional
            The name of the column in `data` that contains the date/time
            information. If `None`, the DataFrame's index is checked and
            converted to a date column, if it is not already a datetime type.
        value_col : str, optional
            The name of the column in `data` that contains the values to
            be analyzed. If `None`, the first column (excluding the date column)
            is selected by default.
        **fit_params : dict
            Additional keyword arguments for internal configuration or processing
            functions, such as `build_data_if`.
    
        Returns
        -------
        self : TimeSeriesPlotter
            The instance of the plotter, configured and ready for further
            operations like plotting.
    
        Raises
        ------
        ValueError
            If the index cannot be converted to datetime or if the specified
            `date_col` does not exist or cannot be converted to datetime.
    
        Examples
        --------
        >>> from gofast.plot.ts import TimeSeriesPlotter
        >>> dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='M')
        >>> values = np.random.rand(len(dates))
        >>> df = pd.DataFrame({'Date': dates, 'Value': values})
        >>> plotter = TimeSeriesPlotter()
        >>> plotter.fit(df, 'Date', 'Value')
        >>> plotter.plotLine()
    
        # Example with DataFrame index as date
        >>> df = pd.DataFrame(values, index=pd.date_range("2020-01-01", periods=12, freq='M'))
        >>> plotter.fit(df)
        >>> plotter.plotHistogram()
    
        Notes
        -----
        - If `date_col` is `None` and the DataFrame's index is not a datetime type,
          an attempt will be made to convert it to datetime and use it as a new
          date column in the DataFrame. If the index has no name, it will be named 'date'.
        - If `date_col` is provided but the column is not in datetime format,
          an attempt will be made to convert this column to datetime using
          `format_to_datetime`.
        - This method does not handle time zones or other more complex datetime
          conversions that may be necessary for certain time series analyses.
          Users should preprocess their data accordingly in such cases.
        """

        columns = fit_params.pop("columns", None)
        data = build_data_if(data, columns=columns, to_frame=True, force=True,
                             input_name='ts', raise_warning="silence")
        if data.empty:
            raise ValueError(
                "The DataFrame is empty. Please provide a DataFrame with data."
                )
        # Handling when date_col is None and ensuring the DataFrame's index is datetime
        if date_col is None:
            if not pd.api.types.is_datetime64_any_dtype(data.index):
                try:
                    data.index = pd.to_datetime(data.index)
                    if data.index.name is None:
                        data.index.name = 'date'
                    date_col = data.index.name
                    data.reset_index(level=0, inplace=True)
                except Exception as e:
                    raise ValueError(
                        "'date_col' is None and the DataFrame index is not in"
                        " datetime format. Ensure that the DataFrame index is"
                        " a datetime index before calling 'fit', or provide a"
                        " 'date_col' name that exists in the DataFrame and can"
                        " be converted to datetime format. Error converting"
                        f" index to datetime: {e}")
            else:
                data.index.name = data.index.name or 'date'
                date_col = data.index.name 
                data.reset_index(level=0, inplace=True)
   
        # Convert date_col to datetime if not already done
        if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
            data = format_to_datetime(data, date_col)
    
        # Verify if data is time series
        is_time_series(data, time_col=date_col)
        
        # Set the class attributes
        self.data = data
        self.date_col = date_col
        self.value_col = value_col or data.columns[data.columns != date_col][0]
        return self

    def plotRollingMean(self, window=12, mean_color='blue',
                         std_color='red', figsize=None, 
                         title='Rolling Mean & Standard Deviation'):
        """
        Generates plots for rolling mean and standard deviation.

        Parameters
        ----------
        window : int, default 12
            Size of the moving window.
        mean_color : str, default 'blue'
            Color for the rolling mean line.
        std_color : str, default 'red'
            Color for the rolling standard deviation line.
        figsize : tuple, default (10, 6)
            Size of the figure.
        title : str, default 'Rolling Mean & Standard Deviation'
            Title of the plot.
        """
        self.inspect 
        self._set_plot_style()
        plt.figure(figsize=self.figsize)
        rolmean = self.data[self.value_col].rolling(window=window).mean()
        rolstd = self.data[self.value_col].rolling(window=window).std()

        plt.plot(self.data[self.date_col], rolmean, label='Rolling Mean', color=mean_color)
        plt.plot(self.data[self.date_col], rolstd, label='Rolling Std', color=std_color)
        plt.title(title, fontsize=self.fontsize + 2 )
        plt.xlabel('Date', fontsize=self.fontsize)
        plt.ylabel('Value', fontsize=self.fontsize)
        plt.legend()
        plt.show()

    def plotAutoCorrelation(self, figsize=None, title='Autocorrelation Plot'):
        """
        Generates an autocorrelation plot for the time series data.

        Parameters
        ----------
        figsize : tuple, default (10, 6)
            Size of the figure.
        title : str, default 'Autocorrelation Plot'
            Title of the plot.
        """
        self.inspect 
        plt.figure(figsize=figsize or self.figsize )
        pd.plotting.autocorrelation_plot(self.data[self.value_col])
        plt.title(title, fontsize=self.fontsize +2 )
        plt.show()
        
    @ensure_pkg ("statsmodels", extra = ( 
        "'plotPACF' expects statsmodels to be installed."
        )
    )
    def plotPACF(
        self, lags=15, 
        figsize=None, 
        title=None, 
        **plot_pacf_kws
        ):
        """
        Generates a partial autocorrelation plot (PACF) for the time series data. 
        This plot is instrumental in time series modeling, especially for determining 
        the order of autoregressive (AR) terms in ARIMA modeling.
    
        Parameters
        ----------
        lags : int, default 15
            Specifies the number of lags to be considered in the PACF plot. This helps 
            in understanding the lagged relationships of the series.
        figsize : tuple, optional
            The dimensions of the figure as `(width, height)` in inches. If not provided, 
            defaults to the class attribute `figsize` or a typical default of (10, 6).
        title : str, optional
            The title of the plot. If None, defaults to 'Partial Autocorrelation Plot'.
        **plot_pacf_kws : dict
            Additional keyword arguments passed to `statsmodels.graphics.tsaplots.plot_pacf`. 
            This allows for further customization of the plot, such as adjusting the confidence 
            intervals or the style of the plot.
    
        Examples
        --------
        >>> import pandas as pd
        >>> from gofast.plot.ts import TimeSeriesPlotter
        >>> data = pd.Series([0.25, 0.5, 0.1, -0.3, 0.0, 0.2, 0.3])
        >>> plotter = TimeSeriesPlotter(data=pd.DataFrame({'Value': data}))
        >>> plotter.plotPACF(lags=10, figsize=(12, 8), title='My PACF Plot')
    
        Notes
        -----
        - PACF plots are essential for identifying how many past points (lags) in 
          the series significantly predict future values.
        - This function leverages the statistical calculations from the `statsmodels` 
          library, which should be installed and available in the environment.
        - The method dynamically creates a figure and axes if not already provided via 
          class attributes, ensuring that the plot is properly sized and displayed.
        - Grid management is configured to visually enhance the plot, making it easier 
          to identify significant lags.
        """
        from statsmodels.graphics.tsaplots import plot_pacf
        
        self.inspect 
        # Create figure and axis if not provided
        fig, ax = self._get_figure_and_axis(figsize)
        
        # Generate the PACF plot
        plot_pacf(self.data[self.value_col], lags=lags, ax=ax, **plot_pacf_kws)
        
        # Setting the plot title
        ax.set_title(title or 'Partial Autocorrelation Plot', fontsize=self.fontsize + 2)
        
        # Configure grid settings
        self._configure_grid(ax)
        
        plt.show()


    @ensure_pkg ( "statsmodels", extra =(
        "'statsmodels' package is expected for 'plotDecomposition' to be feasible.")
        )
    def plotDecomposition(
        self,
        model='additive', 
        freq=12, 
        figsize=None,
        title=None, 
        **decompose_kws
        ):
        """
        Generates a decomposition plot of the time series data, breaking down
        the data into its trend, seasonal, and residual components. This analysis
        helps in understanding the underlying patterns such as seasonality and trends
        in the dataset.
    
        Parameters
        ----------
        model : str, default 'additive'
            Specifies the type of decomposition model to use. Options include:
            - 'additive': suitable when seasonal variations are roughly constant
              throughout the series.
            - 'multiplicative': suitable when seasonal variations change
              proportionally over time.
        freq : int, default 12
            Defines the frequency of the cycle in the time series (e.g., 12 for
            monthly data with an annual cycle).
        figsize : tuple, optional
            The dimensions of the figure as `(width, height)` in inches. If not
            provided, defaults to the class attribute `figsize`.
        title : str, optional
            The title of the plot. If not provided, defaults to 
            'Time Series Decomposition'.
    
        Examples
        --------
        >>> import pandas as pd
        >>> from gofast.plot.ts import TimeSeriesPlotter
        >>> data = pd.Series([1, 2, 3, 4, 5, 6] * 4,
                             index=pd.date_range(start="2020-01-01", periods=24, freq='M'))
        >>> plotter = TimeSeriesPlotter() 
        >>> plotter.fit(data=pd.DataFrame({'Value': data}), date_col='index')
        >>> plotter.plotDecomposition(model='additive', freq=12, 
                                      title='Monthly Data Decomposition')
    
        Notes
        -----
        - This method is particularly useful in identifying whether the underlying
          data is influenced more by seasonal or trend factors.
        - It is critical that the `freq` parameter correctly matches the periodicity
          of the data's cycle for accurate decomposition.
        - The decomposition is performed using the `seasonal_decompose` method from
          the `statsmodels` library, which must be installed in your environment.
        """
        self.inspect 
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Ensure data is appropriately indexed by date
        if not self.data.index.equals(pd.to_datetime(self.data[self.date_col])):
            self.data.set_index(pd.to_datetime(self.data[self.date_col]), inplace=True)
        
        result = seasonal_decompose(self.data[self.value_col], model=model, period=freq, 
                                    **decompose_kws)
    
        # Plot the decomposition results
        result.plot()
        plt.gcf().set_size_inches(figsize or self.figsize)
        plt.suptitle(title or 'Time Series Decomposition', fontsize=self.fontsize + 2)
        
        # self.show_grid=True 
        # self._configure_grid()
        plt.show()

    def _set_plot_style(self):
        """Sets the plot style for aesthetics."""
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = self.figsize  or [10, 6]
        plt.rcParams['font.size'] = self.fontsize

    # document the method and explain the parameters, also add examples 
    # and note of the function. Check the code also. Note all undifined methods 
    
    # are already defined so dont rewrite them . 
    
    def plotCumulativeLine(
        self, 
        figsize=None,
        color='blue', 
        title=None,
        xlabel=None, 
        ylabel=None, 
        rotation=0, 
        **plot_kws
        ):
        """
        Generates a cumulative line plot of the time series data. This type of plot
        is useful for visualizing the accumulation of values over time, providing
        insight into the total growth or decline across the series.
    
        Parameters
        ----------
        figsize : tuple, optional
            The dimensions for the figure as `(width, height)` in inches. If not
            provided, defaults to the class attribute `figsize`.
        color : str, default 'blue'
            The color of the line in the plot. Allows customization of the line color.
        title : str, default 'Cumulative Line Plot'
            The title of the plot. Provides context and description for the plot.
        xlabel : str, optional
            The label for the x-axis. If not provided, defaults to the class attribute
            `date_col` or a generic label if `date_col` is not set.
        ylabel : str, optional
            The label for the y-axis. If not provided, defaults to 'Cumulative Value'
            or the class attribute `value_col` if set.
        rotation : int, default 0
            The rotation angle (in degrees) for x-axis labels. Useful for improving
            readability of date labels or other categorical labels.
        **plot_kws : dict
            Additional keyword arguments to pass to the `ax.plot` function for
            further customization of the line plot.
    
        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from gofast.plot.ts import TimeSeriesPlotter
        >>> data = pd.DataFrame({
        ...     'Date': pd.date_range(start='2020-01-01', periods=100),
        ...     'Value': np.random.rand(100).cumsum()
        ... })
        >>> plotter = TimeSeriesPlotter()
        >>> plotter.fit(data=data, date_col='Date', value_col='Value')
        >>> plotter.plotCumulativeLine(title='Total Growth Over Time', color='green')
    
        Notes
        -----
        This method visually emphasizes the progression or accumulation of values,
        making it particularly useful for time series data where tracking changes
        over time provides valuable insights into trends and patterns.
        """
        self.inspect
        self._set_plot_style()
    
        # Create figure and axis if not provided
        fig, ax = self._get_figure_and_axis(figsize=figsize)
    
        # Plot the cumulative sum of the value column
        ax.plot(self.data[self.date_col], self.data[self.value_col].cumsum(),
                color=color or 'Cumulative Line Plot', **plot_kws)
    
        # Set the plot title and configure grid and labels
        ax.set_title(title, fontsize=self.fontsize + 2)
        self._configure_grid(ax)
        self._configure_axes_and_labels(ax, xlabel, ylabel or 'Cumulative Value', rotation)
    
        plt.show()
        
    def plotDensity(
        self, 
        figsize=None, 
        color='green', 
        title='Density Plot', 
        xlabel=None,
        ylabel=None,
        rotation=0,
        **kde_kws
        ):
        """
        Generates a density plot (Kernel Density Estimate, KDE) of the time series data.
        This plot provides a smooth estimate of the data distribution, which is useful 
        for visualizing the underlying distribution of data points without the need 
        for predefined bins as in histograms.
    
        Parameters
        ----------
        figsize : tuple, optional
            The dimensions for the figure as `(width, height)` in inches. If not
            provided, defaults to the class attribute `figsize`.
            
        color : str, default 'green'
            Specifies the color of the density line in the plot. This allows customization
            of the plot appearance to enhance readability or aesthetic preference.
        title : str, default 'Density Plot'
            The title of the plot. Provides context and description for the density plot.
        xlabel : str, optional
            The label for the x-axis. If not provided, defaults to a generic label 'Value'.
        ylabel : str, optional
            The label for the y-axis. If not provided, defaults to 'Density'.
        rotation : int, default 0
            The rotation angle (in degrees) for x-axis labels, enhancing label readability.
        **kde_kws : dict
            Additional keyword arguments to pass to `sns.kdeplot` for further customization
            of the density plot. This can include parameters such as `bw_adjust` or `fill`.
    
        Examples
        --------
        >>> import numpy as np 
        >>> import pandas as pd
        >>> from gofast.plot.ts import TimeSeriesPlotter
        >>> data = pd.DataFrame({
        ...     'Value': np.random.randn(100)
        ... })
        >>> plotter = TimeSeriesPlotter().fit(data=data, value_col='Value')
        >>> plotter.plotDensity(color='blue', title='Sample Density Plot')
    
        Notes
        -----
        - KDE plots are particularly useful in data analysis for identifying the shape
          of the distribution and potential outliers.
        - Ensure that the `value_col` is properly set in your TimeSeriesPlotter instance
          as this method relies on that column to generate the density plot.
        """
        self.inspect
        self._set_plot_style()
    
        # Create figure and axis if not provided
        fig, ax = self._get_figure_and_axis(figsize=figsize )
    
        # Generate the density plot
        sns.kdeplot(self.data[self.value_col], color=color, shade=True, ax=ax, **kde_kws)
    
        # Set the plot title and configure additional aesthetics
        ax.set_title(title, fontsize=self.fontsize + 2)
    
        # Configure grid settings and labels
        self._configure_grid(ax)
        self._configure_axes_and_labels(
            ax, xlabel or 'Value', ylabel or 'Density', rotation)
    
        plt.show()
    #XXX todo 
    def plotScatterWithTrendline(
        self, 
        figsize=None, 
        color='red', 
        title='Scatter Plot with Trendline', 
        xlabel=None, 
        ylabel=None, 
        rotation=0, 
        **reg_kws
        ):
        """
        Generates a scatter plot with a trendline of the time series data, using
        seaborn's regplot function. This plot is useful for visualizing the 
        relationship between two variables and assessing the linear trend 
        directly from the scatter plot.
    
        Parameters
        ----------
        figsize : tuple, optional
            The dimensions for the figure as `(width, height)` in inches. If not
            provided, defaults to the class attribute `figsize`.
        color : str, default 'red'
            Specifies the color of the scatter points in the plot. This allows
            customization of the plot appearance to enhance readability or 
            aesthetic preference.
        title : str, default 'Scatter Plot with Trendline'
            The title of the plot. Provides context and description for the 
            scatter plot.
        xlabel : str, optional
            The label for the x-axis. If not provided, defaults to the class 
            attribute `date_col`.
        ylabel : str, optional
            The label for the y-axis. If not provided, defaults to the class 
            attribute `value_col`.
        rotation : int, default 0
            The rotation angle (in degrees) for x-axis labels, enhancing label
            readability.
        **reg_kws : dict
            Additional keyword arguments to pass to `sns.regplot` for further 
            customization of the scatter plot. This can include parameters 
            such as `fit_reg` to show or hide the regression line, and 
            `scatter_kws` to modify aspects of the scatter points.
    
        Examples
        --------
        >>> import numpy as np 
        >>> import pandas as pd
        >>> from gofast.plot.ts import TimeSeriesPlotter
        >>> data = pd.DataFrame({
        ...     'Date': pd.date_range(start='2020-01-01', periods=100),
        ...     'Value': np.random.rand(100).cumsum()
        ... })
        >>> plotter = TimeSeriesPlotter() 
        >>> plotter.fit(data=data, date_col='Date', value_col='Value')
        >>> plotter.plotScatterWithTrendline(figsize=(10, 6), title='Growth Trend Analysis')
    
        Notes
        -----
        - This method visually emphasizes the linear relationships between data points,
          making it particularly useful for initial explorations of potential linear 
          patterns or correlations in time series data.
        """
        self.inspect
        self._set_plot_style()
    
        # Create figure and axis if not provided
        fig, ax = self._get_figure_and_axis(figsize=figsize)
    
        # Generate the scatter plot with a regression trendline
        sns.regplot(x=self.date_col, y=self.value_col, data=self.data,
                    color=color, ax=ax, **reg_kws)
    
        # Set the plot title and configure additional aesthetics
        ax.set_title(title, fontsize=self.fontsize + 2)
    
        # Configure grid settings and labels
        self._configure_grid(ax)
        self._configure_axes_and_labels(ax, xlabel, ylabel, rotation)
    
        plt.show()
        
    def plotBar(
        self, 
        figsize=None, 
        color='cyan', 
        title='Bar Plot', 
        xlabel=None, 
        ylabel=None, 
        rotation=0
        ):
        """
        Generates a bar plot of the time series data. This type of plot is useful for 
        visualizing the discrete values in the series across different categories or times.
    
        Parameters
        ----------
        figsize : tuple, optional
            The dimensions for the figure as `(width, height)` in inches. If not
            provided, defaults to the class attribute `figsize`.
        color : str, default 'cyan'
            Specifies the color of the bars in the plot. Allows customization of the plot's
            appearance to enhance readability or aesthetic preference.
        title : str, default 'Bar Plot'
            The title of the plot. Provides context and description for the bar plot.
        xlabel : str, optional
            The label for the x-axis. If not provided, defaults to the class attribute
            `date_col`.
        ylabel : str, optional
            The label for the y-axis. If not provided, defaults to 'Value'.
        rotation : int, default 0
            The rotation angle (in degrees) for x-axis labels, enhancing label readability.
    
        Examples
        --------
        >>> import pandas as pd
        >>> from gofast.plot.ts import TimeSeriesPlotter
        >>> data = pd.DataFrame({
        ...     'Date': pd.date_range(start='2020-01-01', periods=10),
        ...     'Value': [1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
        ... })
        >>> plotter = TimeSeriesPlotter(data=data, date_col='Date', value_col='Value')
        >>> plotter.plotBar(title='Monthly Data Bar Plot', color='blue')
    
        Notes
        -----
        - This method is effective for showing the variation of data across different
          categories or time points, making it easier to spot trends and outliers.
        """
        self.inspect()
        self._set_plot_style()
    
        # Create figure and axis if not provided
        fig, ax = self._get_figure_and_axis(figsize=figsize)
    
        # Generate the bar plot
        self.data.plot(kind='bar', x=self.date_col, y=self.value_col, color=color, ax=ax)
    
        # Set the plot title and configure additional aesthetics
        ax.set_title(title, fontsize=self.fontsize + 2)
    
        # Configure grid settings and labels
        self._configure_grid(ax)
        self._configure_axes_and_labels(ax, xlabel or 'Date', ylabel or 'Value', rotation)
    
        plt.show()


    # def plotScatterWithTrendline(
    #         self, figsize, color='red', title=None, ):
    #     """
    #     Generates a scatter plot with a trendline of the time series data.

    #     Parameters
    #     ----------
    #     color : str, default 'red'
    #         Color of the scatter points in the plot.
    #     title : str, default 'Scatter Plot with Trendline'
    #         Title of the plot.
    #     """
    #     self.inspect 
    #     self._set_plot_style()
    #     # Create figure and axis if not provided
    #     fig, ax = self._get_figure_and_axis(figsize=figsize )
    #     sns.regplot(x=self.date_col, y=self.value_col, data=self.data,
    #                 color=color, **reg_kws)
    #     plt.title(title or 'Scatter Plot with Trendline', fontsize=self.fontsize +2 )
    #     # Configure grid settings and labels
    #     self._configure_grid(ax)
    #     self._configure_axes_and_labels(
    #         ax, xlabel , ylabel, rotation)
    #     plt.show()

    def plotBar(self, color='cyan', title='Bar Plot'):
        """
        Generates a bar plot of the time series data.

        Parameters
        ----------
        color : str, default 'cyan'
            Color of the bars in the plot.
        title : str, default 'Bar Plot'
            Title of the plot.
        """
        self.inspect 
        self._set_plot_style()
        self.data.plot(kind='bar', x=self.date_col, y=self.value_col, color=color)
        plt.title(title, fontsize=self.fontsize+2)
        plt.xlabel('Date', fontsize=self.fontsize)
        plt.ylabel('Value', fontsize=self.fontsize)
        plt.show()

    def plotStackedArea(self, title='Stacked Area Plot'):
        """
        Generates a stacked area plot of the time series data.

        Parameters
        ----------
        title : str, default 'Stacked Area Plot'
            Title of the plot.
        """
        self.inspect 
        self._set_plot_style()
        plt.stackplot(self.data[self.date_col], self.data[self.value_col])
        plt.title(title, fontsize=self.fontsize+2)
        plt.xlabel('Date', fontsize=self.fontsize)
        plt.ylabel('Value', fontsize=self.fontsize)
        plt.show()

    def plotViolin(self, color='purple', title='Violin Plot'):
        """
        Generates a violin plot of the time series data.

        Parameters
        ----------
        color : str, default 'purple'
            Color of the violin in the plot.
        title : str, default 'Violin Plot'
            Title of the plot.
        """
        self.inspect 
        self._set_plot_style()
        sns.violinplot(x=self.data[self.date_col], 
                       y=self.data[self.value_col], color=color)
        plt.title(title, fontsize=self.fontsize +2)
        plt.xlabel('Date', fontsize=self.fontsize)
        plt.ylabel('Value', fontsize=self.fontsize)
        plt.xticks(rotation=45)
        plt.show()

    def pieChart(self, title='Pie Chart'):
        """
        Generates a pie chart of the time series data. Best used with 
        categorical data.

        Parameters
        ----------
        title : str, default 'Pie Chart'
            Title of the chart.
        """
        self.inspect 
        self._set_plot_style()
        self.data[self.value_col].value_counts().plot(kind='pie', autopct='%1.1f%%')
        plt.title(title, fontsize=14)
        plt.ylabel('')
        plt.show()

    def plotLine(
        self, 
        figsize=None, 
        title=None,
        xlabel=None, 
        ylabel=None, 
        color=None, 
        **lineplot_kws
        ):
        """
        Generates a line plot of the time series data using seaborn's lineplot
        function. This method automatically adjusts for overlapping x-axis labels
        and can be customized extensively via additional keyword arguments.
    
        Parameters
        ----------
        figsize : tuple, optional
            The size of the figure to create. If None, defaults to matplotlib's
            figure size.
        title : str, optional
            The title of the plot. If None, defaults to 'Time Series Line Plot'.
        xlabel : str, optional
            The label for the x-axis. If None, defaults to 'Date'.
        ylabel : str, optional
            The label for the y-axis. If None, defaults to 'Value'.
        color : str, optional
            The color of the line in the plot. If None, defaults to 'blue'.
        **lineplot_kws : dict
            Additional keyword arguments to be passed to seaborn.lineplot.
    
        Raises
        ------
        ValueError
            If the DataFrame is empty or does not contain the specified `date_col`
            or `value_col`.
    
        Examples
        --------
        >>> from gofast.plot.ts import TimeSeriesPlotter
        >>> dates = pd.date_range(start="2020-01-01", periods=100, freq='D')
        >>> values = np.random.rand(100)
        >>> df = pd.DataFrame({'Date': dates, 'Value': values})
        >>> plotter = TimeSeriesPlotter().fit(df, 'Date', 'Value')
        >>> plotter.plotLine(title='Random Time Series', color='green')
    
        Notes
        -----
        This method is part of a class that assumes `data`, `date_col`, and `value_col`
        have been predefined as attributes of the instance. The `smart_rotation` function
        is called to handle overlapping x-axis labels automatically, enhancing the 
        readability of the plot, especially when dealing with dense date labels.
        """
        self.inspect 
        if self.data.empty:
            raise ValueError("The DataFrame is empty. Please provide a "
                             "DataFrame with data.")
    
        if self.date_col not in self.data.columns or self.value_col not in self.data.columns:
            raise ValueError(f"DataFrame must contain specified columns"
                             f" '{self.date_col}' and '{self.value_col}'")
    
        if self.ax is None: 
            fig, ax = plt.subplots(figsize=figsize or self.figsize )
        else:
            ax = self.ax
            
        sns.lineplot(data=self.data, x=self.date_col, y=self.value_col,
                     ax=ax, color=color or 'blue', **lineplot_kws)
        ax.set_title(title or 'Time Series Line Plot')
        # Apply smart rotation to adjust tick labels dynamically
        self._configure_axes_and_labels(ax, xlabel, ylabel )

    def plotHistogram(
        self, 
        figsize=None, 
        title=None,
        xlabel=None, 
        ylabel=None, 
        bins=30, 
        **hist_kws
        ):
        """
        Generates a histogram of the time series data using seaborn's histplot
        function. This method allows customization of figure size, labels, title,
        and the number of bins, as well as additional seaborn histplot keyword
        arguments.
    
        Parameters
        ----------
        figsize : tuple, optional
            The size of the figure to create. If None, defaults to matplotlib's
            figure size.
        title : str, optional
            The title of the plot. If None, defaults to 'Time Series Histogram'.
        xlabel : str, optional
            The label for the x-axis. If None, defaults to 'Value'.
        ylabel : str, optional
            The label for the y-axis. If None, defaults to 'Frequency'.
        bins : int, optional
            Number of bins in the histogram. Default is 30.
        **hist_kws : dict
            Additional keyword arguments to be passed to seaborn.histplot.
    
        Raises
        ------
        ValueError
            If the DataFrame is empty or does not contain the specified `value_col`.
    
        Examples
        --------
        >>> from gofast.plot.ts import TimeSeriesPlotter 
        >>> values = np.random.randn(100)
        >>> df = pd.DataFrame({'Value': values})
        >>> plotter = TimeSeriesPlotter().fit(df, value_col='Value')
        >>> plotter.plotHistogram(title='Distribution of Values', bins=20, color='red')
    
        Notes
        -----
        This method is part of a class that assumes `data` and `value_col` have been
        predefined as attributes of the instance. It uses seaborn to create the
        histogram and matplotlib to customize the plot aesthetics.
        """
        self.inspect 
        if self.data.empty:
            raise ValueError("The DataFrame is empty. Please provide a DataFrame with data.")
    
        if self.value_col not in self.data.columns:
            raise ValueError(f"DataFrame must contain the specified column '{self.value_col}'")
    
        if self.ax is None: 
            fig, ax = plt.subplots(figsize=figsize or self.figsize )
        else:
            ax = self.ax
            
        sns.histplot(self.data[self.value_col], bins=bins, kde=True, ax=ax, **hist_kws)
        ax.set_title(title or 'Time Series Histogram', fontsize = self.fontsize +2 )
        self._configure_axes_and_labels(ax, xlabel, ylabel )

    def plotBox(
        self, 
        figsize=None, 
        title=None,
        xlabel=None, 
        ylabel=None, 
        rotation=None, 
        **boxplot_kws
        ):
        """
        Generates a box plot of the time series data using seaborn's boxplot
        function. This method allows customization of figure size, labels, title,
        and rotation of x-axis labels, as well as additional seaborn boxplot keyword
        arguments.
    
        Parameters
        ----------
        figsize : tuple, optional
            The size of the figure to create. If None, defaults to matplotlib's
            figure size.
        title : str, optional
            The title of the plot. If None, defaults to 'Time Series Box Plot'.
        xlabel : str, optional
            The label for the x-axis. If None, defaults to the column name
            specified by `date_col`.
        ylabel : str, optional
            The label for the y-axis. If None, defaults to the column name 
            specified by `value_col`.
        rotation : int, optional
            Rotation angle of x-axis labels. Default is 45 degrees.
        **boxplot_kws : dict
            Additional keyword arguments to be passed to seaborn.boxplot.
    
        Raises
        ------
        ValueError
            If the DataFrame is empty or does not contain the specified `date_col`
            or `value_col`.
    
        Examples
        --------
        >>> from gofast.plot.ts import TimeSeriesPlotter 
        >>> dates = pd.date_range(start="2020-01-01", periods=10, freq='M')
        >>> values = np.random.rand(10)
        >>> df = pd.DataFrame({'Date': dates, 'Value': values})
        >>> plotter = TimeSeriesPlotter().fit(df, date_col='Date', value_col='Value')
        >>> plotter.plotBox(title='Monthly Value Distribution', xlabel='Month',
                            ylabel='Value')
    
        Notes
        -----
        This method is part of a class that assumes `data`, `date_col`, and `value_col`
        have been predefined as attributes of the instance. It uses seaborn to create
        the box plot and matplotlib to customize the plot aesthetics.
        """
        self.inspect 
        if self.data.empty:
            raise ValueError("The DataFrame is empty. Please provide a DataFrame with data.")
    
        if self.date_col not in self.data.columns or self.value_col not in self.data.columns:
            raise ValueError(f"DataFrame must contain specified columns"
                             f" '{self.date_col}' and '{self.value_col}'")
    
        if self.ax is None: 
            fig, ax = plt.subplots(figsize=figsize or self.figsize )
        else:
            ax = self.ax
 
        sns.boxplot(data=self.data, x=self.date_col, y=self.value_col, ax=ax, 
                    **boxplot_kws )
    
        # Set the title and labels
        ax.set_title(title or 'Time Series Box Plot', fontsize = self.fontsize  +2 )
        ax.set_xlabel(xlabel or self.date_col, fontsize = self.fontsize )
        ax.set_ylabel(ylabel or self.value_col, fontsize = self.fontsize )
    
        # Improve date formatting on x-axis
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        # Optionally apply smart rotation if date labels overlap
        # Adjust the rotation of the x-axis labels if specified
        if rotation is None: 
            smart_rotation(ax ) 
        else:  
            plt.xticks(rotation=rotation)
    
        # Improve layout
        plt.gcf().autofmt_xdate()
    
        plt.show()


    def plotArea(
        self, 
        figsize=None, 
        title=None, 
        xlabel=None, 
        ylabel=None,
        color=None, 
        alpha=None
        ):

        """
        Generates an area plot of the time series data. This plot can be useful
        for displaying the volume beneath a line chart, emphasizing the magnitude
        of change over time.

        Parameters
        ----------
        figsize : tuple, optional
            The dimensions for the figure as `(width, height)` in inches.
            Defaults to the value of the class attribute `figsize` if not
            provided.
        title : str, optional
            The title of the plot. Defaults to 'Time Series Area Plot' if 
            not specified.
        xlabel : str, optional
            The label text for the x-axis. Defaults to the class attribute
            `date_col` if not provided.
        ylabel : str, optional
            The label text for the y-axis. Defaults to the class attribute
            `value_col` if not provided.
        color : str, optional
            The fill color for the area under the line. Defaults to 'skyblue'
            if not specified.
        alpha : float, optional
            The transparency level of the fill color. Value should be between
            0 (transparent) and 1 (opaque). Defaults to 0.4 if not provided.

        Raises
        ------
        ValueError
            If `self.data` is empty or if `self.date_col` or `self.value_col`
            are not columns in `self.data`.

        Examples
        --------
        >>> from gofast.plot.ts import TimeSeriesPlotter
        >>> plotter = TimeSeriesPlotter()
        >>> plotter.fit(data=df, date_col='Date', value_col='Value')
        >>> plotter.plotArea(figsize=(12, 6), title='My Area Plot',
                             xlabel='Time', ylabel='Observations',
                             color='lightgreen', alpha=0.5)

        Notes
        -----
        - The method uses `plt.fill_between` from Matplotlib to generate the
          area plot. 
        - The x-axis labels will be automatically rotated if they are
          determined to overlap, enhancing readability.
        - This method should be used after the `fit` method, which prepares
          `self.data`, `self.date_col`, and `self.value_col` for plotting.
        """
        self.inspect 
        if self.data.empty:
            raise ValueError("The DataFrame is empty. Please provide a DataFrame with data.")
    
        if self.date_col not in self.data.columns or self.value_col not in self.data.columns:
            raise ValueError("DataFrame must contain specified columns"
                             f" '{self.date_col}' and '{self.value_col}'")
    
        # Use self.ax if it has been defined, else create a new axis
        if self.ax is None:
            fig, ax = plt.subplots(figsize=figsize or self.figsize)
        else:
            ax = self.ax
    
        # Plot the area chart
        ax.fill_between(self.data[self.date_col], self.data[self.value_col],
                        color=color or 'skyblue', alpha=alpha or 0.4)
    
        # Set the title and labels with a default font size from self.fontsize
        ax.set_title(title or 'Time Series Area Plot', fontsize=self.fontsize + 2)
        ax.set_xlabel(xlabel or self.date_col, fontsize=self.fontsize)
        ax.set_ylabel(ylabel or self.value_col, fontsize=self.fontsize)
        
        # configure grid 
        self._configure_grid(ax )
        
        # Adjust the rotation of the x-axis labels if specified using 
        # smart_rotation function
        smart_rotation(ax)
        # Improve layout
        plt.gcf().autofmt_xdate()
        plt.show()

    def _configure_axes_and_labels(
            self, ax, xlabel=None, ylabel=None, rotation=None):
        """
        Configures the axes labels, their font size, and label rotation for a
        given Axes object. It applies smart rotation to x-axis labels to prevent
        overlap and improve readability, unless a specific rotation is specified.
    
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The Axes object for which to set labels and properties.
        xlabel : str, optional
            The label for the x-axis. If None, defaults to the class attribute
            `date_col`. The font size is set to the class attribute `fontsize`.
        ylabel : str, optional
            The label for the y-axis. If None, defaults to the class attribute
            `value_col`. The font size is set to the class attribute `fontsize`.
        rotation : int, optional
            The rotation angle (in degrees) for the x-axis tick labels.
            If None, applies smart rotation based on label overlap.
    
        Notes
        -----
        This method should be called within other plotting methods after the
        plot has been created but before it is displayed or saved. It relies
        on the `smart_rotation` function to automatically adjust the x-axis
        label rotation if needed.
        """
        # Set axis labels and their font size
        ax.set_xlabel(xlabel or self.date_col, fontsize=self.fontsize)
        ax.set_ylabel(ylabel or self.value_col, fontsize=self.fontsize)
    
        # Adjust x-axis label rotation
        if rotation is None:
            smart_rotation(ax)
        else:
            plt.xticks(rotation=rotation)
    
        # Improve layout and show plot
        plt.gcf().autofmt_xdate()
        plt.show()
        
    def heatmapCorrelation(
        self, 
        method='pearson', 
        min_periods=1, 
        figsize=None, 
        title=None, 
        cmap=None,
        annot=True, 
        fmt=".2f"
        ):
        """
        Generates a heatmap to visualize the correlation matrix of the dataset's
        numerical features. This method provides a color-coded representation
        that highlights the strength of correlations among variables.
    
        Parameters
        ----------
        method : {'pearson', 'kendall', 'spearman'}, default 'pearson'
            Specifies the method of correlation:
            - 'pearson' : standard correlation coefficient
            - 'kendall' : Kendall Tau correlation coefficient
            - 'spearman' : Spearman rank correlation
        min_periods : int, default 1
            Minimum number of observations required per pair of columns
            to have a valid result. If fewer than `min_periods` non-NA values
            are present the result will be NA.
        figsize : tuple, optional
            The dimensions for the figure as `(width, height)` in inches.
            If not provided, defaults to the class attribute `figsize`.
        title : str, optional
            The title of the heatmap. If not specified, defaults to
            'Heatmap of Correlations'.
        cmap : str, optional
            The colormap scheme to use for the heatmap. If None, uses the
            'coolwarm' colormap.
        annot : bool, optional
            Determines whether to annotate each cell with the numeric value.
            Defaults to True.
        fmt : str, optional
            The string formatting code to use when adding annotations.
            Defaults to ".2f", which represents floating-point numbers
            with two decimal places.
    
        Raises
        ------
        ValueError
            If the DataFrame is empty or if `self.data` does not contain
            enough non-NA values to calculate a correlation.

        Examples
        --------
        >>> import pandas as pd
        >>> from gofast.plot.ts import TimeSeriesPlotter
        >>> # Creating a sample DataFrame with random numerical data for the heatmap
        >>> data = {
        ...     'A': np.random.rand(10),
        ...     'B': np.random.rand(10),
        ...     'C': np.random.rand(10),
        ...     'D': np.random.rand(10)
        ... }
        >>> df = pd.DataFrame(data)
        >>> plotter = TimeSeriesPlotter().fit(df)
        >>> plotter.heatmapCorrelation(
        ...     figsize=(8, 8),
        ...     title='Variable Correlation',
        ...     cmap='viridis',
        ...     annot=True,
        ...     fmt=".2f"
        ... )

     
        Notes
        -----
        - The 'method' and 'min_periods' parameters are passed directly to the
          pandas 'corr' function. They determine the type of correlation coefficient
          and the handling of missing data respectively.
        - The heatmap provides a visual representation of the pairwise correlations
          between the columns in `self.data`. High positive correlations are
          typically indicated by darker shades in the heatmap, and high negative
          correlations by lighter shades, with the actual color depending on the
          chosen 'cmap'.
        """
        self.inspect 
        if self.data.empty:
            raise ValueError("The DataFrame is empty. Please provide"
                             " a DataFrame with numerical data.")
     
        # Create figure and axis if not provided
        if self.ax is None:
            fig, ax = plt.subplots(figsize=figsize or self.figsize)
        else:
            ax = self.ax
     
        # Calculate correlation matrix and plot heatmap
        corr_matrix = self.data.corr(method=method, min_periods =min_periods )
        sns.heatmap(corr_matrix, ax=ax, annot=annot, fmt=fmt, cmap=cmap or 'coolwarm')
     
        # Set the title with a default font size from self.fontsize
        ax.set_title(title or 'Heatmap of Correlations', fontsize=self.fontsize + 2)
        
        # Display the plot
        plt.show()
        
    def _get_figure_and_axis(self, figsize=None):
        """
        Retrieves or creates a figure and axis object for plotting. If an axis
        has been predefined in the class instance, it will be used; otherwise,
        a new figure and axis will be created with the specified figure size.
    
        Parameters
        ----------
        figsize : tuple, optional
            The dimensions for the figure as `(width, height)` in inches.
            If not provided, defaults to the class attribute `figsize`.
    
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object for the plot. Only returned if a new figure is created.
        ax : matplotlib.axes.Axes
            The axis object for the plot.
    
        Examples
        --------
        >>> # Assuming this method is part of a class with self.figsize and self.ax
        >>> fig, ax = self._get_figure_and_axis(figsize=(8, 6))
        >>> ax.plot(data)  # Continue with plotting commands
        >>> plt.show()
        
        Notes
        -----
        The `self.ax` attribute is used to allow for subplot reuse. When creating
        new subplots, this function will respect the `figsize` parameter or
        default to `self.figsize` if not provided.
        """
        if self.ax is None:
            fig, ax = plt.subplots(figsize=figsize or self.figsize)
            return fig, ax
        else:
            return None, self.ax
           
    def plotLag(
        self, lag=1, 
        figsize=None, 
        title=None, 
        c='orange', 
        alpha=0.5, 
        **lag_kws
        ):
        """
        Generates a lag plot for the time series data to analyze the relationship 
        between an observation and its lag. Useful for checking randomness or 
        serial correlation in time series datasets.
    
        Parameters
        ----------
        lag : int, default 1
            Specifies the lag of the scatter plot, the number of time units to 
            offset.
        figsize : tuple, optional
            The size of the figure as `(width, height)` in inches.
            If not provided, defaults to the class attribute `figsize`.
        title : str, optional
            The title of the plot. If None, a default title including the lag 
            number is generated (e.g., 'Lag 1 Plot').
        c : str, default 'orange'
            The color of the scatter points in the plot.
        alpha : float, default 0.5
            The transparency level of the scatter points in the plot.
        **lag_kws : dict
            Additional keyword arguments to pass to the `pd.plotting.lag_plot` 
            function.
    
        Examples
        --------
        >>> import pandas as pd
        >>> from gofast.plot.ts import TimeSeriesPlotter
        >>> data = pd.Series([1, 2, 3, 4, 5, 6])
        >>> plotter = TimeSeriesPlotter().fit(data=data)
        >>> plotter.plotLag(lag=1, title='Daily Returns Lag Plot',
        ...                c='blue', alpha=0.6)
    
        Notes
        -----
        This method is intended for time series data analysis, helping identify 
        if there are patterns consistent over time. It plots data against the same 
        data offset by the lag amount, useful for identifying seasonal effects 
        or autocorrelation.
        """
        self.inspect 
    
        # Use self.ax if it has been defined, else create a new axis
        fig, ax = self._get_figure_and_axis(figsize = figsize  )
    
        # Generate the lag plot
        pd.plotting.lag_plot(self.data[self.value_col], 
                             lag=lag, ax=ax, c=c, alpha=alpha, **lag_kws)
    
        # Setting the title if not provided
        ax.set_title(title or f'Lag {lag} Plot', fontsize=self.fontsize + 2)
        # configure grid 
        self._configure_grid(ax )
        
        plt.show()
        
    def plotScatter(
        self, 
        color='blue', 
        figsize=None, 
        title=None, 
        rotation=0,
        xlabel=None, 
        ylabel=None, 
        **scatter_kws
        ):
        """
        Generates a scatter plot of the time series data. This method provides a
        way to visually examine the relationship between two variables.
    
        Parameters
        ----------
        color : str, default 'blue'
            Color of the scatter points in the plot.
        figsize : tuple, optional
            The dimensions for the figure as `(width, height)` in inches.
            If not provided, defaults to the class attribute `figsize`.
        title : str, optional
            The title of the plot. If None, defaults to 'Scatter Plot'.
        rotation : int, optional
            The rotation angle (in degrees) for the x-axis tick labels.
        xlabel : str, optional
            The label for the x-axis. If None, defaults to the class attribute
            `date_col`.
        ylabel : str, optional
            The label for the y-axis. If None, defaults to the class attribute
            `value_col`.
        **scatter_kws : dict
            Additional keyword arguments to pass to the `ax.scatter` function.
    
        Examples
        --------
        >>> import pandas as pd
        >>> from gofast.plot.ts import TimeSeriesPlotter
        >>> data = pd.DataFrame({
        ...     'Date': pd.date_range(start="2020-01-01", periods=100, freq='D'),
        ...     'Value': np.random.rand(100)
        ... })
        >>> plotter = TimeSeriesPlotter().fit(data, 'Date', 'Value')
        >>> plotter.plotScatter(color='red', figsize=(10, 6),
        ...                     title='Daily Values Scatter Plot',
        ...                     xlabel='Date', ylabel='Value')
    
        Notes
        -----
        This method is useful for spotting outliers, trends, or clusters of
        data points. It is particularly effective in time series analysis to
        reveal patterns or data irregularities over time.
        """
        self.inspect 
    
        # Create figure and axis if not provided
        fig, ax = self._get_figure_and_axis(figsize=figsize)
    
        # Plot scatter using additional keyword arguments
        ax.scatter(self.data[self.date_col], self.data[self.value_col], 
                   color=color, **scatter_kws)
    
        # Setting the title and configuring axes labels
        ax.set_title(title or 'Scatter Plot', fontsize=self.fontsize + 2)
        
        # configure grid 
        self._configure_grid(ax )
        
        # Configure axes, labels, and rotation
        self._configure_axes_and_labels(ax, xlabel, ylabel, rotation=rotation)
        
    def plotCumulativeDistribution(
        self, color='green', 
        figsize=None, 
        title=None,
        bins=30, 
        alpha=0.6, 
        density=True, 
        rotation=0,
        xlabel=None, 
        ylabel=None, 
        **hist_kws
        ):
        """
        Generates a cumulative distribution plot of the time series data, highlighting
        how values are distributed across the dataset. This visualization is essential
        for understanding the overall distribution and identifying the presence of
        outliers, skewness, or other anomalies in the data.
    
        Parameters
        ----------
        color : str, default 'green'
            The color of the plot lines. This parameter allows customization of the
            plot's color scheme to enhance readability or thematic presentation.
        figsize : tuple, optional
            Dimensions of the figure as `(width, height)` in inches. If not specified,
            defaults to the class attribute `figsize`.
        title : str, optional
            The title of the plot. If not provided, 'Cumulative Distribution' is used.
        bins : int, default 30
            The number of histogram bins to use. More bins provide a more detailed
            view of the distribution but can lead to overfitting in visualization.
        alpha : float, default 0.6
            Transparency of the histogram bars. Helps in visual overlap and depth
            perception in dense plots.
        density : bool, default True
            If True, the histogram is normalized to form a probability density, where
            the area under the histogram will sum to one. This is useful for comparing
            distributions of datasets of different sizes.
        rotation : int, default 0
            The rotation angle of x-axis labels. Useful for improving label readability
            in cases of closely spaced or long text labels.
        xlabel : str, optional
            Label for the x-axis. If not provided, defaults to the class attribute
            `date_col` if applicable.
        ylabel : str, optional
            Label for the y-axis. Defaults to 'Cumulative Probability' if not provided.
        **hist_kws : dict
            Additional keyword arguments to pass to `plt.hist`. Allows for further
            customization of the histogram plot, such as edge color, line style, etc.
    
        Examples
        --------
        >>> import pandas as pd
        >>> from gofast.plot.ts import TimeSeriesPlotter
        >>> data = pd.Series(np.random.randn(1000))
        >>> plotter = TimeSeriesPlotter().fit(data=pd.DataFrame({'Value': data}),
                                              value_col='Value')
        >>> plotter.plotCumulativeDistribution(
                color='blue', figsize=(12, 8),
                title='Cumulative Distribution of Returns',
                bins=40, alpha=0.5, density=True
            )
    
        Notes
        -----
        Cumulative distribution plots are particularly useful in statistical analysis for
        determining the probability bounds of the dataset. The plot can be used to quickly
        ascertain the percentile ranking of data points, aiding in decision-making processes
        related to risk management or quality control.
        """
        self.inspect 
    
        # Create figure and axis using helper function 
        fig, ax = self._get_figure_and_axis(figsize=figsize)
    
        # Plot the cumulative distribution histogram
        ax.hist(self.data[self.value_col], bins=bins, density=density,
                cumulative=True, color=color, alpha=alpha, **hist_kws)
    
        # Setting plot titles and labels
        ax.set_title(title or 'Cumulative Plot', fontsize=self.fontsize + 2)
        
        # configure grid 
        self._configure_grid(ax )
        
        # Configure axes, labels, and rotation
        self._configure_axes_and_labels(
            ax, xlabel, ylabel or 'Cumulative Probability', rotation=rotation
        )
        
    def plotStackedBar(
        self, 
        secondary_col, 
        title=None, 
        figsize=None,
        xlabel=None, 
        ylabel=None, 
        rotation=0
        ):
        """
        Generates a stacked bar plot of the time series data. This type of plot
        is useful for comparing the proportion of two data series at each date
        point, emphasizing their cumulative totals.
    
        Parameters
        ----------
        secondary_col : str
            The name of the secondary column to stack. This column and the primary
            column (`self.value_col`) will be displayed cumulatively in the stacked bars.
        title : str, optional
            The title of the plot. If None, defaults to 'Stacked Bar Plot'.
        figsize : tuple, optional
            The dimensions for the figure as `(width, height)` in inches.
            If not provided, defaults to the class attribute `figsize` or (10, 6).
        xlabel : str, optional
            The label for the x-axis. If None, defaults to the class attribute `date_col`.
        ylabel : str, optional
            The label for the y-axis. If None, defaults to 'Total'.
        rotation : int, optional
            The rotation angle (in degrees) for the x-axis tick labels. Useful for improving
            label readability, particularly when labels are numerous or lengthy.
    
        Examples
        --------
        >>> import pandas as pd
        >>> from gofast.plot.ts import TimeSeriesPlotter
        >>> data = pd.DataFrame({
        ...     'Date': pd.date_range(start="2021-01-01", periods=4, freq='M'),
        ...     'Sales': [200, 240, 310, 400],
        ...     'Costs': [150, 190, 250, 300]
        ... })
        >>> plotter = TimeSeriesPlotter().fit(data=data, date_col='Date',
        ...                                   value_col='Sales')
        >>> plotter.plotStackedBar(
        ...     secondary_col='Costs',
        ...     title='Monthly Sales and Costs',
        ...     figsize=(8, 6),
        ...     xlabel='Month',
        ...     ylabel='Dollars',
        ...     rotation=45
        ... )
    
        Notes
        -----
        - This method is particularly effective in displaying how two components
          contribute to a total over time.
        - Ensure that the `secondary_col` exists in `self.data` before calling this method.
        - The method uses the `plot` function from pandas with the `kind='bar'` and `stacked=True`
          arguments to generate the plot.
        """
        self.inspect 
    
        # Create figure and axis if not provided
        fig, ax = self._get_figure_and_axis(figsize=figsize)
        
        # Plot the data
        self.data.groupby(self.date_col)[[self.value_col, secondary_col]].sum().plot(
            kind='bar', stacked=True, ax=ax
        )
        
        # Set the title and configure axes and labels
        ax.set_title(title or 'Stacked Bar Plot', fontsize=self.fontsize + 2)
    
        # configure grid 
        self._configure_grid(ax )
        # Configure axes, labels, and rotation
        self._configure_axes_and_labels(ax, xlabel, ylabel, rotation=rotation)
    
        plt.show()


    def plotHexbin(
        self, 
        gridsize=30, 
        title=None, 
        figsize=None,
        xlabel=None, 
        ylabel=None, 
        rotation=0, 
        ):
        """
        Generates a hexbin plot of the time series data.

        Parameters
        ----------
        gridsize : int, default 30
            Size of the hexagons in the grid.
        figsize : tuple, default (10, 6)
            Size of the figure.
        title : str, default 'Hexbin Plot'
            Title of the plot.
        """
        self.inspect 
        # Create figure and axis if not provided
        fig, ax = self._get_figure_and_axis(figsize=figsize)
        plt.hexbin(self.data[self.date_col].apply(lambda x: x.toordinal()),
                   self.data[self.value_col], gridsize=gridsize, cmap='Blues')
        # Setting the title and configuring axes labels
        ax.set_title(title or 'Hexbin Plot', fontsize=self.fontsize + 2)

        # Configure axes, labels, and rotation
        self._configure_axes_and_labels(ax, xlabel, ylabel, rotation=rotation)

    def plotKDE(self, shade=True, figsize=(10, 6), title='KDE Plot'):
        """
        Generates a KDE plot of the time series data.

        Parameters
        ----------
        shade : bool, default True
            If True, fill the area under the KDE curve.
        figsize : tuple, default (10, 6)
            Size of the figure.
        title : str, default 'KDE Plot'
            Title of the plot.
        """
        self.inspect 
        plt.figure(figsize=figsize)
        sns.kdeplot(self.data[self.value_col], shade=shade)
        plt.title(title)
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.show()

    def plotStep(self, color='green', linestyle='-', linewidth=2,
                  figsize=(10, 6), title='Step Plot'):
        """
        Generates a step plot of the time series data.

        Parameters
        ----------
        color : str, default 'green'
            Color of the step line.
        linestyle : str, default '-'
            Style of the line.
        linewidth : int, default 2
            Width of the line.
        figsize : tuple, default (10, 6)
            Size of the figure.
        title : str, default 'Step Plot'
            Title of the plot.
        """
        self.inspect 
        plt.figure(figsize=figsize)
        plt.step(self.data[self.date_col], self.data[self.value_col],
                 color=color, linestyle=linestyle, linewidth=linewidth)
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.show()

    def plotErrorBar(self, yerr=None, color='blue', ecolor='red', 
                       elinewidth=2, capsize=5, figsize=(10, 6),
                       title='Error Bar Plot'):
        """
        Generates an error bar plot of the time series data.

        Parameters
        ----------
        yerr : float or array-like, default None
            Represents the error bar sizes.
        color : str, default 'blue'
            Color of the bars.
        ecolor : str, default 'red'
            Color of the error bars.
        elinewidth : int, default 2
            Width of the error bars.
        capsize : int, default 5
            Size of the end cap of the error bars.
        figsize : tuple, default (10, 6)
            Size of the figure.
        title : str, default 'Error Bar Plot'
            Title of the plot.
        """
        self.inspect 
        plt.figure(figsize=figsize)
        plt.errorbar(self.data[self.date_col], self.data[self.value_col], 
                     yerr=yerr, fmt='o', color=color, ecolor=ecolor, 
                     elinewidth=elinewidth, capsize=capsize)
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.show()

    def plotStackedLine(self, secondary_col, figsize=(10, 6), title='Stacked Line Plot'):
        """
        Generates a stacked line plot of the time series data.
    
        The stacked_line_plot method creates a plot where the values of two 
        columns are stacked on top of each other.
        Parameters
        ----------
        secondary_col : str
            The name of the secondary column to stack with the primary value column.
        figsize : tuple, default (10, 6)
            Size of the figure.
        title : str, default 'Stacked Line Plot'
            Title of the plot.
        """
        self.inspect 
        plt.figure(figsize=figsize)
        plt.plot(self.data[self.date_col], self.data[self.value_col], label='Primary')
        plt.plot(self.data[self.date_col], self.data[secondary_col], label='Secondary', 
                 )
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Values')
        plt.legend()
        plt.show()
 

    def plotBubble(self, bubble_size_col, figsize=(10, 6), title='Bubble Plot'):
        """
        Generates a bubble plot of the time series data.
    
        The bubble_plot visualizes data points with varying sizes based on
        an additional column.
        
        Parameters
        ----------
        bubble_size_col : str
            The name of the column determining the size of each bubble.
        figsize : tuple, default (10, 6)
            Size of the figure.
        title : str, default 'Bubble Plot'
            Title of the plot.
        """
        self.inspect 
        plt.figure(figsize=figsize)
        plt.scatter(self.data[self.date_col], self.data[self.value_col],
                    s=self.data[bubble_size_col] * 1000, alpha=0.5)
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.show()
        
    @ensure_pkg("squarify", extra= ( 
        "`plotSunburst` expects 'squarify' package to be installed."
        )
    )
    def plotSunburst(self, path_col, values_col, figsize=(10, 10),
                      title='Sunburst Plot'):
        """
        Generates a sunburst plot of the time series data.
    
        The sunburst_plot is useful for hierarchical data representation.
        
        Parameters
        ----------
        path_col : str
            The name of the column representing the hierarchical structure.
        values_col : str
            The name of the column representing the values for each segment.
        figsize : tuple, default (10, 10)
            Size of the figure.
        title : str, default 'Sunburst Plot'
            Title of the plot.
        """
        self.inspect 
        import squarify # 

        plt.figure(figsize=figsize)
        squarify.plot(sizes=self.data[values_col],
                      label=self.data[path_col], alpha=0.6)
        plt.axis('off')
        plt.title(title)
        plt.show()
        
    def radarChart(self, categories, figsize=(6, 6), title='Radar Chart'):
        """
        Generates a radar chart of the time series data.
    
        The radar_chart displays multivariate data in the form of a 
        two-dimensional chart with radial axes.
        
        Parameters
        ----------
        categories : list
            List of categories corresponding to each axis in the radar chart.
        figsize : tuple, default (6, 6)
            Size of the figure.
        title : str, default 'Radar Chart'
            Title of the chart.
        """
        self.inspect 
        
        N = len(categories)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
    
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
        plt.xticks(angles[:-1], categories)
    
        values = self.data[categories].iloc[0].tolist()
        values += values[:1]
        ax.plot(angles, values)
        ax.fill(angles, values, 'b', alpha=0.1)
    
        plt.title(title, size=20, y=1.1)
        plt.show()
        
    def plotWaterfall(self, figsize=(10, 6), title='Waterfall Plot'):
        """
        Generates a waterfall plot of the time series data.
        
        The waterfall_plot shows the cumulative effect of sequential 
        positive and negative values.
    
        Parameters
        ----------
        figsize : tuple, default (10, 6)
            Size of the figure.
        title : str, default 'Waterfall Plot'
            Title of the plot.
        """
        self.inspect 
        plt.figure(figsize=figsize)
        increases = self.data[self.value_col] >= 0
        plt.bar(self.data[self.date_col], self.data[self.value_col],
                width=1, color=increases.map({True: 'g', False: 'r'}))
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Change in Value')
        plt.show()

    @property 
    def inspect(self): 
        """ Inspect data and trigger plot after checking the data entry. 
        Raises `NotFittedError` if `ExPlot` is not fitted yet."""
        
        msg = ( "{expobj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )
        
        if self.value_col is None: 
            raise NotFittedError(msg.format(expobj=self))
        return 1 
    
    def _configure_grid(self, ax=None):
        """
        Configures the grid for a given matplotlib Axes object based on the instance's
        grid display preferences. Allows detailed customization of the grid's appearance
        if the grid is enabled.
    
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional 
            The Axes object for which the grid settings are to be configured.
    
        Attributes
        ----------
        show_grid : bool
            A class attribute that determines whether to display the grid.
        gswitch : bool
            Determines whether the grid is turned on or off.
        gls : str
            Grid line style, e.g., '-', '--', '-.', ':'
        glw : float
            Grid line width.
        gaxis : {'both', 'x', 'y'}
            Specifies which grid lines to display.
        galpha : float
            Transparency of grid lines, where 0 is fully transparent and 1 is fully opaque.
        gc : str
            Grid color.
    
        Examples
        --------
        >>> ax = plt.gca()
        >>> self.show_grid = True
        >>> self.gswitch = True
        >>> self.gls = '--'
        >>> self.glw = 0.5
        >>> self.gaxis = 'both'
        >>> self.galpha = 0.7
        >>> self.gc = 'gray'
        >>> self._configure_grid(ax)
    
        Notes
        -----
        - This method should be called after all plotting is done on the ax object
          to ensure the grid settings do not interfere with other plot elements.
        - The grid settings are applied only if `show_grid` is True. Otherwise, the grid
          is removed if previously displayed.
        """
        if not self.show_grid:
            if ax:
                ax.grid(False)
            else:
                plt.grid(False)
            return
    
        # Define common grid settings
        grid_settings = {
            "which": self.gwhich,
            "axis": self.gaxis,  # Controls which grid lines to apply settings on
            "linestyle": self.gls,  # Line style, e.g., '-', '--', '-.', ':'
            "linewidth": self.glw,  # Line width
            "alpha": self.galpha,  # Transparency of grid lines
            "color": self.gc  # Color of the grid lines
        }
    
        # Apply grid settings based on whether an Axes object is provided
        target = ax if ax else plt
        target.grid(True, **grid_settings)

 
TimeSeriesPlotter.__doc__="""\
A class for visualizing time series data in a Pandas DataFrame.

This class provides various methods for plotting time series data,
offering a range of visualizations to analyze and present the data effectively.

Parameters
----------
data : pandas.DataFrame
    The DataFrame containing time series data.
date_col : str
    The name of the column in `data` that contains the date or time information.
value_col : str
    The name of the column in `data` that contains the values to be plotted.

Attributes
----------
data : pandas.DataFrame
    The DataFrame provided by the user.
date_col : str
    The column in `data` representing the time aspect.
value_col : str
    The column in `data` representing the numerical values.

Methods
-------
plotLine(...)
    Generates a line plot of the time series data.
histogram(...)
    Generates a histogram of the values in the time series.
plotBox(...)
    Generates a box plot of the time series data.
plotArea(...)
    Generates an area plot of the time series data.
heatmapCorrelation(...)
    Generates a heatmap showing correlations between columns in the data.
plotLag(...)
    Generates a lag plot to analyze autocorrelation in the time series.
plotRollingMean(...)
    Plots the rolling mean and standard deviation.
plotAutocorrelation(...)
    Generates an autocorrelation plot for the time series.
plotPACF(...)
    Generates a partial autocorrelation plot for the time series.
plotDecomposition(...)
    Decomposes the time series into trend, seasonal, and residual components.
plotScatter(...)
    Generates a scatter plot of the time series data.
plotViolin(...)
    Generates a violin plot of the time series data.
plotCumulativeDistribution(...)
    Generates a cumulative distribution plot of the time series data.
plotStackedBar(...)
    Generates a stacked bar plot of the time series data.
pieChart(...)
    Generates a pie chart of the distribution of values.
plotHexbin(...)
    Generates a hexbin plot of the time series data.
plotkde(...)
    Generates a kernel density estimate plot of the time series data.
plotStep(...)
    Generates a step plot of the time series data.
plotErrorBar(...)
    Generates an error bar plot of the time series data.
plotStackedLine(...)
    Generates a stacked line plot of the time series data.
plotWaterfall(...)
    Generates a waterfall plot of the time series data.
plotBubble(...)
    Generates a bubble plot of the time series data.
plotSunburst(...)
    Generates a sunburst plot of hierarchical data.
radarChart(...)
    Generates a radar chart for multivariate data comparison.

Examples
--------
>>> import pandas as pd 
>>> from gofast.plot.ts import TimeSeriesPlotter 
>>> df = pd.DataFrame({
...     'Date': pd.date_range(start='2021-01-01', periods=5, freq='D'),
...     'Value': [1, 2, 3, 4, 5]
... })
>>> plotter = TimeSeriesPlotter()
>>> plotter.fit(df, 'Date', 'Value')
>>> plotter.plotLine()
"""
if __name__ == "__main__":
    
    from gofast.plot.ts import TimeSeriesPlotter 
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='M')
    values = np.random.rand(len(dates))
    df = pd.DataFrame({'Date': dates, 'Value': values})

    plotter = TimeSeriesPlotter().fit(df, 'Date', 'Value')
    plotter.plotStep()
    plotter.plotErrorBar(yerr=0.1)
    

    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='M')
    values = np.random.rand(len(dates))
    df = pd.DataFrame({'Date': dates, 'Value': values,
                       'SecondaryValue': np.random.rand(len(dates))})

    plotter = TimeSeriesPlotter().fit(df, 'Date', 'Value')
    plotter.plotScatter()

    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='M')
    values = np.random.rand(len(dates))
    df = pd.DataFrame({'Date': dates, 'Value': values})

    plotter = TimeSeriesPlotter().fit(df, 'Date', 'Value')
    plotter.plotRollingMean()
    plotter.plotAutocorrelation()
    plotter.plotPACF()
    plotter.plotDecomposition()
    
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='M')
    values = np.random.rand(len(dates))
    df = pd.DataFrame({'Date': dates, 'Value': values})

    plotter = TimeSeriesPlotter().fit(df, 'Date', 'Value')
    plotter.plotArea()
    plotter.heatmap_correlation()
    plotter.lag_plot()
    
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='M')
    values = np.random.rand(len(dates))
    df = pd.DataFrame({'Date': dates, 'Value': values})

    plotter = TimeSeriesPlotter().fit(df, 'Date', 'Value')
    plotter.plotArea()
    plotter.heatmapCorrelation()
    plotter.plotLag()


    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='M')
    values = np.random.rand(len(dates))
    df = pd.DataFrame({'Date': dates, 'Value': values})

    plotter = TimeSeriesPlotter().fit(df, 'Date', 'Value')
    plotter.plotLine()
    plotter.plotHistogram()
    plotter.plotBox()

    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='M')
    values = np.random.rand(len(dates))
    df = pd.DataFrame({'Date': dates, 'Value': values})

    plotter = TimeSeriesPlotter().fit(df, 'Date', 'Value')
 
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='M')
    values = np.random.rand(len(dates))
    df = pd.DataFrame({'Date': dates, 'Value': values, 
                       'SecondaryValue': np.random.rand(len(dates))})

    plotter = TimeSeriesPlotter().fit(df, 'Date', 'Value')
    plotter.plotArea()
    plotter.heatmapCorrelation()
    plotter = TimeSeriesPlotter().fit(df, 'Date', 'Value')
    plotter.plotStackedLine(secondary_col='SecondaryValue')
    plotter.plotWaterfall()
    plotter.plotBubble(bubble_size_col='BubbleSize')
    categories = ['Category1', 'Category2', 'Category3', 'Category4', 'Category5']
    plotter.radarChart(categories=categories)

