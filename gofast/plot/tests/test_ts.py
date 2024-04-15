# -*- coding: utf-8 -*-
"""
test_ts.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch # noqa 
import pytest
from gofast.tools.coreutils import is_module_installed 
from gofast.tools.funcutils import install_package 
from gofast.plot.ts import TimeSeriesPlotter  


try:
    import pytest_mock # noqa 
except ImportError:
    if not is_module_installed('pytest_mock'): 
        install_package(
            'pytest_mock', dist_name="pytest-mock", infer_dist_name=True ) 
 
def test_fit_method():
    # Create a sample DataFrame
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='M')
    values = np.random.rand(len(dates))
    df = pd.DataFrame({'Date': dates, 'Value': values})

    # Instantiate the TimeSeriesPlotter
    plotter = TimeSeriesPlotter()

    # Test the fit method
    returned_object = plotter.fit(df, date_col='Date', value_col='Value')

    # Assert that the method returns self for chaining
    assert returned_object is plotter, "Fit method should return self for method chaining."

    # Assert that the internal state is set correctly
    assert plotter.data is not None, "Data should be properly assigned."
    assert plotter.date_col == 'Date', "Date column should be set correctly."
    assert plotter.value_col == 'Value', "Value column should be set correctly."

    # Optionally, check if date conversion and column extraction are correct
    # This might require mocking or further setup to validate properly
    assert pd.to_datetime(plotter.data['Date']).equals(df['Date']), ( 
        "Dates should be correctly converted to datetime."
        )
    
def test_plotRollingMean(mocker):
    mock_figure = mocker.patch('matplotlib.pyplot.figure')
    mocker.patch('matplotlib.pyplot.show')
    mocker.patch('matplotlib.pyplot.plot')
    
    # Create a sample DataFrame
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='D')
    values = np.random.rand(len(dates))
    df = pd.DataFrame({'Date': dates, 'Value': values})

    # Instantiate the TimeSeriesPlotter and fit data
    plotter = TimeSeriesPlotter()
    plotter.fit(data=df, date_col='Date', value_col='Value')

    # Test plotRollingMean
    plotter.plotRollingMean(window=10)

    # Check if figure functions are called the expected number of times
    assert mock_figure.call_count == 5, ( 
        f"Expected 'figure' to have been called 5 times. Called {mock_figure.call_count} times."
        )
    plt.show.assert_called_once()

def test_plotAutocorrelation(mocker):

    mock_figure = mocker.patch('matplotlib.pyplot.figure')
    mocker.patch('matplotlib.pyplot.show')
    mocker.patch('pandas.plotting.autocorrelation_plot', return_value=mocker.Mock())

    # Prepare data
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='D')
    values = np.random.rand(len(dates))
    df = pd.DataFrame({'Date': dates, 'Value': values})

    plotter = TimeSeriesPlotter()
    plotter.fit(data=df, date_col='Date', value_col='Value')

    # Test plotAutocorrelation
    plotter.plotAutoCorrelation()
    #print(mock_figure.call_count )
    assert mock_figure.call_count == 3, ( 
        f"Expected 'figure' to have been called 3 times. Called {mock_figure.call_count} times."
        )
    plt.show.assert_called_once()
    
@pytest.mark.skipif (not is_module_installed('statsmodels'),
                      reason= 'Need statsmodels to be installed for running this code.')
def test_plotPACF(mocker):
    mocker.patch('matplotlib.pyplot.figure')
    mocker.patch('matplotlib.pyplot.show')
    mocker.patch('statsmodels.graphics.tsaplots.plot_pacf', return_value=mocker.Mock())

    # Data setup
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='D')
    values = np.random.rand(len(dates))
    df = pd.DataFrame({'Date': dates, 'Value': values})

    plotter = TimeSeriesPlotter()
    plotter.fit(data=df, date_col='Date', value_col='Value')

    # Test plotPACF
    plotter.plotPACF()

    plt.show.assert_called_once()
    
def test_plotDecomposition(mocker):
    mocker.patch('matplotlib.pyplot.figure')
    mocker.patch('matplotlib.pyplot.show')
    mocker.patch('statsmodels.tsa.seasonal.seasonal_decompose')
    
    # Create a sample DataFrame
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='M')
    values = np.random.rand(len(dates))
    df = pd.DataFrame({'Date': dates, 'Value': values})

    # Instantiate and fit
    plotter = TimeSeriesPlotter()
    plotter.fit(data=df, date_col='Date', value_col='Value')

    # Test plotDecomposition
    plotter.plotDecomposition()

    plt.show.assert_called_once()
    
def test_plotCumulativeLine(mocker):
    mocker.patch('matplotlib.pyplot.figure')
    mocker.patch('matplotlib.pyplot.show')
    mocker.patch('matplotlib.pyplot.plot')

    # Setup DataFrame
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='D')
    values = np.random.rand(len(dates))
    df = pd.DataFrame({'Date': dates, 'Value': values})

    # Instantiate and fit
    plotter = TimeSeriesPlotter()
    plotter.fit(data=df, date_col='Date', value_col='Value')

    # Test
    plotter.plotCumulativeLine()

    plt.show.assert_called_once()
def test_plotDensity(mocker):
    
    mocker.patch('matplotlib.pyplot.figure')
    mocker.patch('matplotlib.pyplot.show')
    mocker.patch('seaborn.kdeplot')

    # Data setup
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='D')
    values = np.random.rand(len(dates))
    df = pd.DataFrame({'Date': dates, 'Value': values})

    # Instantiate and fit
    plotter = TimeSeriesPlotter()
    plotter.fit(data=df, date_col='Date', value_col='Value')

    # Test
    plotter.plotDensity()

    plt.show.assert_called_once()

def test_plotScatterWithTrendline(mocker):
    mocker.patch('matplotlib.pyplot.figure')
    mocker.patch('matplotlib.pyplot.show')
    mocker.patch('seaborn.regplot')

    # Data setup
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq='D')
    values = np.random.rand(len(dates))
    df = pd.DataFrame({'Date': dates, 'Value': values})

    # Instantiate and fit
    plotter = TimeSeriesPlotter()
    plotter.fit(data=df, date_col='Date', value_col='Value')

    # Test
    plotter.plotScatterWithTrendline()

    plt.show.assert_called_once()

if __name__== '__main__': 
    pytest.main( [__file__])