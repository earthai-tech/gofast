# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Time-Series Plots  
"""
from math import pi
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.plotting import lag_plot, autocorrelation_plot
try: 
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.graphics.tsaplots import plot_pacf
except: pass 
try: import squarify  # for sunburst plot
except:pass 

from ..exceptions import NotFittedError 
from ..property import BasePlot 
from ..tools._dependency import import_optional_dependency 
from ..tools.baseutils import extract_target 
from ..tools.coreutils import format_to_datetime
from ..tools.validator import is_time_series , build_data_if 

class TimeSeriesPlotter (BasePlot) :
    def __init__(self,  **kws):
        super().__init__(**kws) 
        
    def fit( self, data, /, date_col, value_col =None, **fit_params): 
        """
        Fit the TimeSeriesPlotter with a time series dataset.

        Parameters
        ----------
        data : pandas DataFrame
            The time series dataset.
        date_col : str
            The name of the column in 'data' that contains the date/time 
            information.
        value_col : str
            The name of the column in 'data' that contains the values to 
            be analyzed.
            
        Return 
        ----------
        self: Object 
           Return self. 
           
        Examples 
        ----------
        dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='M')
        values = np.random.rand(len(dates))
        df = pd.DataFrame({'Date': dates, 'Value': values})
        plotter = TimeSeriesPlotter(df, 'Date', 'Value')
        plotter.line_plot()
        plotter.histogram()
        plotter.box_plot()
        # Example DataFrame
        dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='M')
        values = np.random.rand(len(dates))
        df = pd.DataFrame({'Date': dates, 'Value': values})

        plotter = TimeSeriesPlotter(df, 'Date', 'Value')
        plotter.area_plot()
        plotter.heatmapCorrelation()
        # Other plot methods...
        """
        columns =fit_params.pop("columns", None )
        data = build_data_if(data, columns =columns, to_frame=True, force=True, 
                             input_name='ts', raise_warning="silence")

        self.data = format_to_datetime(data, date_col= date_col )
        is_time_series(data , time_col= date_col )
        
        date_value, self.value_col =extract_target(self.data, target_names= date_col )
        self.date_col = date_col
        self.value_col = value_col
        
        return self 

    def plotRollingMean(self, window=12, mean_color='blue',
                         std_color='red', figsize=(10, 6), 
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
        plt.figure(figsize=figsize)
        rolmean = self.data[self.value_col].rolling(window=window).mean()
        rolstd = self.data[self.value_col].rolling(window=window).std()

        plt.plot(self.data[self.date_col], rolmean, label='Rolling Mean', color=mean_color)
        plt.plot(self.data[self.date_col], rolstd, label='Rolling Std', color=std_color)
        plt.title(title, fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.legend()
        plt.show()

    def plotAutocorrelation(self, figsize=(10, 6), title='Autocorrelation Plot'):
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
        plt.figure(figsize=figsize)
        autocorrelation_plot(self.data[self.value_col])
        plt.title(title, fontsize=14)
        plt.show()

    def plotPACF(self, lags=15, figsize=(10, 6),
                  title='Partial Autocorrelation Plot'):
        """
        Generates a partial autocorrelation plot for the time series data.

        Parameters
        ----------
        lags : int, default 15
            Number of lags to show.
        figsize : tuple, default (10, 6)
            Size of the figure.
        title : str, default 'Partial Autocorrelation Plot'
            Title of the plot.
        """
        self.inspect 
        import_optional_dependency("statsmodels")
        plt.figure(figsize=figsize)
        plot_pacf(self.data[self.value_col], lags=lags)
        plt.title(title, fontsize=14)
        plt.show()

    def plotDecomposition(self, model='additive', freq=12, figsize=(10, 6),
                           title='Time Series Decomposition'):
        """
        Generates a decomposition plot of the time series data.

        Parameters
        ----------
        model : str, default 'additive'
            Type of seasonal component. Options are 'additive' or 'multiplicative'.
        freq : int, default 12
            Frequency of the time series.
        figsize : tuple, default (10, 6)
            Size of the figure.
        title : str, default 'Time Series Decomposition'
            Title of the decomposition plot.
        """
        self.inspect 
        import_optional_dependency("statsmodels")
        plt.figure(figsize=figsize)
        result = seasonal_decompose(self.data.set_index(
            self.date_col)[self.value_col], model=model, period=freq)

        result.plot()
        plt.suptitle(title, fontsize=14)
        plt.show()

    def _set_plot_style(self):
        """Sets the plot style for aesthetics."""
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = [10, 6]
        plt.rcParams['font.size'] = 12


    def plotCumulativeLine(self, color='blue', title='Cumulative Line Plot'):
        """
        Generates a cumulative line plot of the time series data.

        Parameters
        ----------
        color : str, default 'blue'
            Color of the line in the plot.
        title : str, default 'Cumulative Line Plot'
            Title of the plot.
        """
        self.inspect 
        self._set_plot_style()
        plt.plot(self.data[self.date_col], self.data[self.value_col].cumsum(), color=color)
        plt.title(title, fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative Value', fontsize=12)
        plt.show()

    def plotDensity(self, color='green', title='Density Plot'):
        """
        Generates a density plot of the time series data.

        Parameters
        ----------
        color : str, default 'green'
            Color of the density line in the plot.
        title : str, default 'Density Plot'
            Title of the plot.
        """
        self.inspect 
        self._set_plot_style()
        sns.kdeplot(self.data[self.value_col], color=color, shade=True)
        plt.title(title, fontsize=14)
        plt.xlabel('Value', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.show()

    def plotScatterWithTrendline(
            self, color='red', title='Scatter Plot with Trendline'):
        """
        Generates a scatter plot with a trendline of the time series data.

        Parameters
        ----------
        color : str, default 'red'
            Color of the scatter points in the plot.
        title : str, default 'Scatter Plot with Trendline'
            Title of the plot.
        """
        self.inspect 
        self._set_plot_style()
        sns.regplot(x=self.date_col, y=self.value_col, data=self.data, color=color)
        plt.title(title, fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.show()

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
        plt.title(title, fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Value', fontsize=12)
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
        plt.title(title, fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Value', fontsize=12)
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
        plt.title(title, fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Value', fontsize=12)
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

    def plotLine(self, figsize=(10, 6), title='Time Series Line Plot',
                  xlabel='Date', ylabel='Value', color='blue'):
        """
        Generates a line plot of the time series data.

        Parameters
        ----------
        figsize : tuple, default (10, 6)
            Size of the figure.
        title : str, default 'Time Series Line Plot'
            Title of the plot.
        xlabel : str, default 'Date'
            Label for the x-axis.
        ylabel : str, default 'Value'
            Label for the y-axis.
        color : str, default 'blue'
            Color of the line in the plot.
        """
        self.inspect 
        fig, ax = plt.subplots(figsize=figsize)
        sns.lineplot(data=self.data, x=self.date_col,
                     y=self.value_col, ax=ax, color=color)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.show()

    def histogram(self, figsize=(10, 6), title='Time Series Histogram',
                  xlabel='Value', ylabel='Frequency', bins=30):
        """
        Generates a histogram of the time series data.

        Parameters
        ----------
        figsize : tuple, default (10, 6)
            Size of the figure.
        title : str, default 'Time Series Histogram'
            Title of the plot.
        xlabel : str, default 'Value'
            Label for the x-axis.
        ylabel : str, default 'Frequency'
            Label for the y-axis.
        bins : int, default 30
            Number of bins in the histogram.
        """
        self.inspect 
        fig, ax = plt.subplots(figsize=figsize)
        sns.histplot(self.data[self.value_col], bins=bins, kde=True, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.show()

    def plotBox(self, figsize=(10, 6), title='Time Series Box Plot',
                 xlabel='Date', ylabel='Value', rotation=45):
        """
        Generates a box plot of the time series data.

        Parameters
        ----------
        figsize : tuple, default (10, 6)
            Size of the figure.
        title : str, default 'Time Series Box Plot'
            Title of the plot.
        xlabel : str, default 'Date'
            Label for the x-axis.
        ylabel : str, default 'Value'
            Label for the y-axis.
        rotation : int, default 45
            Rotation angle of x-axis labels.
        """
        self.inspect 
        fig, ax = plt.subplots(figsize=figsize)
        sns.boxplot(data=self.data, x=self.date_col, y=self.value_col, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.xticks(rotation=rotation)
        plt.show()


    def plotArea(self, figsize=(10, 6), title='Time Series Area Plot',
                  xlabel='Date', ylabel='Value', color='skyblue', alpha=0.4):
        """
        Generates an area plot of the time series data.

        Parameters
        ----------
        figsize : tuple, default (10, 6)
            Size of the figure.
        title : str, default 'Time Series Area Plot'
            Title of the plot.
        xlabel : str, default 'Date'
            Label for the x-axis.
        ylabel : str, default 'Value'
            Label for the y-axis.
        color : str, default 'skyblue'
            Color of the area plot.
        alpha : float, default 0.4
            Transparency of the area plot.
        """
        self.inspect 
        plt.figure(figsize=figsize)
        plt.fill_between(self.data[self.date_col], self.data[self.value_col],
                         color=color, alpha=alpha)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def heatmapCorrelation(self, figsize=(10, 6),
                            title='Heatmap of Correlations', cmap='coolwarm',
                            annot=True, fmt=".2f"):
        """
        Generates a heatmap showing correlation between different time series.

        Parameters
        ----------
        figsize : tuple, default (10, 6)
            Size of the figure.
        title : str, default 'Heatmap of Correlations'
            Title of the heatmap.
        cmap : str, default 'coolwarm'
            Colormap of the heatmap.
        annot : bool, default True
            If True, write the data value in each cell.
        fmt : str, default ".2f"
            String formatting code to use when adding annotations.
        """
        self.inspect 
        plt.figure(figsize=figsize)
        sns.heatmap(self.data.corr(), annot=annot, fmt=fmt, cmap=cmap)
        plt.title(title, fontsize=14)
        plt.show()

    def plotLag(self, lag=1, figsize=(10, 6), title=None, c='orange',
                 alpha=0.5):
        """
        Generates a lag plot for the time series data.

        Parameters
        ----------
        lag : int, default 1
            Lag number to be used in the plot.
        figsize : tuple, default (10, 6)
            Size of the figure.
        title : str or None, default None
            Title of the plot. If None, defaults to 'Lag {lag} Plot'.
        c : str, default 'orange'
            Color of the scatter points in the plot.
        alpha : float, default 0.5
            Transparency of the scatter points.
        """
        self.inspect 
        plt.figure(figsize=figsize)
        lag_plot(self.data[self.value_col], lag=lag, c=c, alpha=alpha)
        plt.title(title or f'Lag {lag} Plot', fontsize=14)
        plt.show()

    def plotScatter(self, color='blue', figsize=(10, 6), title='Scatter Plot'):
        """
        Generates a scatter plot of the time series data.

        Parameters
        ----------
        color : str, default 'blue'
            Color of the scatter points.
        figsize : tuple, default (10, 6)
            Size of the figure.
        title : str, default 'Scatter Plot'
            Title of the plot.
        """
        self.inspect 
        plt.figure(figsize=figsize)
        plt.scatter(self.data[self.date_col], self.data[self.value_col], color=color)
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.show()

    def plotCumulativeDistribution(self, color='green', figsize=(10, 6),
                                     title='Cumulative Distribution'):
        """
        Generates a cumulative distribution plot of the time series data.

        Parameters
        ----------
        color : str, default 'green'
            Color of the cumulative distribution line.
        figsize : tuple, default (10, 6)
            Size of the figure.
        title : str, default 'Cumulative Distribution'
            Title of the plot.
        """
        self.inspect 
        plt.figure(figsize=figsize)
        plt.hist(self.data[self.value_col], bins=30, density=True,
                 cumulative=True, color=color, alpha=0.6)
        plt.title(title)
        plt.xlabel('Value')
        plt.ylabel('Cumulative Probability')
        plt.show()

    def plotStackedBar(self, secondary_col, figsize=(10, 6),
                         title='Stacked Bar Plot'):
        """
        Generates a stacked bar plot of the time series data.

        Parameters
        ----------
        secondary_col : str
            The name of the secondary column to stack.
        figsize : tuple, default (10, 6)
            Size of the figure.
        title : str, default 'Stacked Bar Plot'
            Title of the plot.
        """
        self.inspect 
        plt.figure(figsize=figsize)
        self.data.groupby(self.date_col)[
            self.value_col, secondary_col].sum().plot(kind='bar', stacked=True)
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Values')
        plt.show()


    def plotHexbin(self, gridsize=30, figsize=(10, 6), title='Hexbin Plot'):
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
        plt.figure(figsize=figsize)
        plt.hexbin(self.data[self.date_col].apply(lambda x: x.toordinal()),
                   self.data[self.value_col], gridsize=gridsize, cmap='Blues')
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.show()

    def plotkde(self, shade=True, figsize=(10, 6), title='KDE Plot'):
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
                 bottom=self.data[self.value_col])
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
        import_optional_dependency("squarify")
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
    plotter.rolling_mean_std()
    plotter.autocorrelation_plot()
    plotter.pacf_plot()
    plotter.decomposition_plot()
    
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='M')
    values = np.random.rand(len(dates))
    df = pd.DataFrame({'Date': dates, 'Value': values})

    plotter = TimeSeriesPlotter().fit(df, 'Date', 'Value')
    plotter.area_plot()
    plotter.heatmap_correlation()
    plotter.lag_plot()
    
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='M')
    values = np.random.rand(len(dates))
    df = pd.DataFrame({'Date': dates, 'Value': values})

    plotter = TimeSeriesPlotter().fit(df, 'Date', 'Value')
    plotter.area_plot()
    plotter.heatmapCorrelation()
    plotter.lag_plot()


    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='M')
    values = np.random.rand(len(dates))
    df = pd.DataFrame({'Date': dates, 'Value': values})

    plotter = TimeSeriesPlotter().fit(df, 'Date', 'Value')
    plotter.line_plot()
    plotter.histogram()
    plotter.box_plot()

    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='M')
    values = np.random.rand(len(dates))
    df = pd.DataFrame({'Date': dates, 'Value': values})

    plotter = TimeSeriesPlotter().fit(df, 'Date', 'Value')
 
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='M')
    values = np.random.rand(len(dates))
    df = pd.DataFrame({'Date': dates, 'Value': values})

    plotter = TimeSeriesPlotter().fit(df, 'Date', 'Value')
    plotter.plotArea()
    plotter.heatmapCorrelation()
    plotter = TimeSeriesPlotter().fit(df, 'Date', 'Value')
    plotter.plotStackedLine(secondary_col='SecondaryValue')
    plotter.plotWaterfall()
    plotter.plotBubble(bubble_size_col='BubbleSize')
    categories = ['Category1', 'Category2', 'Category3', 'Category4', 'Category5']
    plotter.radarChart(categories=categories)

