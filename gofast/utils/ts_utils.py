# -*- coding: utf-8 -*-
""" Times-series utilities. """

import warnings
from numbers import Real, Integral 

from scipy.fft import fft
from scipy.stats import pearsonr
from scipy.stats import zscore
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.preprocessing import StandardScaler, MinMaxScaler 
from sklearn.model_selection import TimeSeriesSplit
from sklearn.decomposition import PCA

try: 
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.seasonal import STL, seasonal_decompose

except: 
    pass 

from gofast.api.summary import ResultSummary 
from gofast.compat.sklearn import Interval, StrOptions, validate_params 
from gofast.core.array_manager import smart_ts_detector
from gofast.core.checks import exist_features, validate_ratio, is_in_if
from gofast.core.handlers import columns_manager 
from gofast.core.io import to_frame_if  
from gofast.utils.base_utils import validate_target_in, select_features  
from gofast.utils.deps_utils import ensure_pkg 
from gofast.utils.validator import is_time_series, is_frame

__all__= [ 
    'decompose_ts','infer_decomposition_method',
    'prepare_ts_df','trend_analysis','trend_ops',
    'ts_engineering','ts_validator','visual_inspection', 
    'ts_corr_analysis', 'transform_stationarity','ts_split', 
    'ts_outlier_detector', 'create_lag_features', 
    'select_and_reduce_features'
 ]

def ts_validator(
    df, 
    dt_col=None, 
    to_datetime=None, 
    as_index="auto", 
    error ='raise', 
    return_dt_col=False,
    ensure_order=False,
    verbose=0, 
    ) : 
    """
    Validate and preprocess time series data, ensuring the datetime 
    column is properly handled.
    
    Parameters 
    -----------
    df : pandas.DataFrame
        The DataFrame containing the `<dt_col>`. This column is 
        expected to either be datetime-like, numeric, or convertible 
        to a known temporal format.
    dt_col : str
        The name of the column in `df` representing date or 
        time-related data. If not found, handling depends on `<error>`.
    
    to_datetime : {None, 'auto', 'Y', 'M', 'W', 'D', 'H', 'min', 's'}, optional
        Controls how the column is converted if not already datetime:
        - None: No conversion, only format detection.
        - 'auto': Automatically infer and convert based on rules.
        - Explicit codes like 'Y', 'M', 'W', 'min', 's' attempt to 
          convert according to those units.
    error : {'raise', 'ignore', 'warn'}, optional
        Defines behavior if `<dt_col>` is not found or cannot be 
        interpreted:
        
        - 'raise': Raise a ValueError.
        - 'ignore': Return without modification or raise.
        - 'warn': Issue a warning and proceed (if possible).
    as_index: bool, 
       Whether to return the entire dataset and set as index the `dt_col`. This 
       is done when `return_types='df'. 
       
    
    verbose : int, optional
        Verbosity level for logging:
        
        - 0: No output.
        - 1: Basic info.
        - 2: More details on reasoning steps.
        - 3: Very detailed internal states.
    
    Returns 
    --------
    - df: DataFrame with `dt_col` processed and optionally set as the index.
    
    """

    # Validate input is a DataFrame
    df = to_frame_if(df, df_only=True)
    # If no datetime column is specified, check index
    if dt_col is None:
        # If the index is already datetime
        if pd.api.types.is_datetime64_any_dtype(df.index):
            if verbose >= 1:
                print("Datetime index detected. No further datetime"
                      " conversion needed.")
            # Handle as per the `as_index` setting
            # Index is already set
            dt_col= df.index.name  
            
            if not as_index:
                df.reset_index(inplace=True)
                
            return df if not return_dt_col else ( df, dt_col)
        else:
            # If the index is not datetime, handle conversion
            if verbose >= 1:
                print("Index is not a datetime type. Checking"
                      " datetime conversion settings.")
            
            if to_datetime is not None:
                # Attempt conversion based on provided `to_datetime`
                try:
                    df.index = pd.to_datetime(
                        df.index, errors='coerce', format=to_datetime)
                    if df.index.isnull().any():
                        raise ValueError(
                            "Datetime conversion failed due to"
                            " invalid format in the index.")
                    if verbose >= 2:
                        print("Datetime conversion applied to the"
                              f" index using format: {to_datetime}")
                except Exception as e:
                    if error == 'raise':
                        raise ValueError(
                            f"Failed to convert index to datetime: {e}")
                    elif error == 'warn':
                        warnings.warn(
                            f"Failed to convert index to datetime. {e}")
             
                dt_col=df.index.name 
                
                return (df, dt_col) if return_dt_col else df 
            
            else:
                # Index cannot be converted to datetime and no `to_datetime` specified
                if error == 'raise':
                    raise ValueError(
                        "Index is not a datetime object and no `dt_col`"
                        " or `to_datetime` provided.")
                elif error == 'warn':
                    warnings.warn(
                        "Warning: Index is not a datetime object,"
                        " and no `to_datetime` was provided."
                        )
               
                return (df, dt_col) if return_dt_col else df 

    else:
        # Handle case where `dt_col` is specified
        if dt_col in df.index: 
            df.reset_index (inplace =True, drop=False )  
        exist_features(df, features=dt_col, name="Datetime column ")
        
        # Use smart_ts_detector to process the `dt_col` and convert if necessary
        df = smart_ts_detector(
            df,
            dt_col=dt_col,
            return_types="df",  # Return the full DataFrame with the datetime column processed
            as_index=as_index,
            error=error,
            verbose=verbose,
            to_datetime=to_datetime,
        )
        
    if ensure_order: # ensure temporal order 
    
        is_index_already =False 
        if dt_col in df.index : 
            df.reset_index (inplace =True)
            is_index_already=True 
        df = df.sort_values(by=dt_col)  # Ensure temporal order
        if is_index_already:  # revert back to the index as 
            df.set_index (dt_col, inplace =True )
            
    return (df, dt_col) if return_dt_col else df 


def trend_analysis(
    df, 
    value_col,     
    dt_col=None, 
    view=False, 
    check_stationarity=True, 
    trend_type='both', 
    strategy="adf",  # Can be 'adf' or 'kpss' for stationarity check
    stationnay_color='green', 
    linestyle='--', 
    fig_size=(10, 6), 
    trend_color='red', 
    show_grid=True, 
    error='raise',
    verbose=0,
    **kw
):
    """
    Perform trend analysis on the given time series data, including
    stationarity test and trend detection.
    
    Parameters:
    - df: DataFrame containing the time series data.
    - value_col: Column name or series of the target variable to forecast.
    - dt_col: Column name of the time series data (datetime-related column).
    - view: Whether to visualize the data and trend detection.
    - check_stationarity: Whether to test for stationarity using ADF or KPSS.
    - trend_type: Type of trend detection to use ('both', 'upward', or 'downward').
    - strategy: Stationarity test method ('adf' or 'kpss').
    - stationnay_color: Color for stationary mean line.
    - linestyle: Line style for the stationary mean.
    - fig_size: Figure size for the plot.
    - trend_color: Color for the trend line.
    - show_grid: Whether to display grid lines in the plot.
    - error: Behavior if issues arise ('raise', 'warn', 'ignore').
    - verbose: Verbosity level for logging.
    
    Returns:
    - trend: Detected trend type ('upward', 'downward', 'stationary').
    - p_value: p-value from stationarity test (if applicable).
    """
    # Validate and process the datetime column (set index, format, etc.)
    df, dt_col = ts_validator(
        df, dt_col=dt_col, 
        to_datetime='auto', 
        as_index=False, 
        error=error,
        return_dt_col= True, 
        verbose=verbose
    )
    # Validate and process the time series DataFrame and datetime column
    target,_= validate_target_in(df, value_col, error=error, verbose=verbose) 

    # Step 1: Check if the data is stationary using the chosen test (ADF or KPSS)
    p_value = None
    trend = 'non-stationary'
    
    if check_stationarity:
        if strategy == "adf":
            result = adfuller(df[target.name].dropna())
            p_value = result[1]
            if p_value < 0.05:
                trend = 'stationary'
            else:
                trend = 'non-stationary'
        
        elif strategy == "kpss":
            # Stationarity with constant (level)
            result = kpss(df[target.name].dropna(), regression='c')  
            p_value = result[1]
            if p_value < 0.05:
                trend = 'non-stationary'
            else:
                trend = 'stationary'
    
    # Step 2: Apply trend detection (based on linear regression if necessary)
    if trend == 'non-stationary' or trend_type == 'both':
        # Fit a linear regression model to detect long-term trend (slope)
        X = np.arange(len(df)).reshape(-1, 1)  # Time index
        y = target.values  # Use the processed target variable
        
        model = sm.OLS(y, sm.add_constant(X)).fit()  # Ordinary least squares
        slope = model.params[1]
        
        # Determine trend direction based on the slope
        if slope > 0:
            detected_trend = 'upward'
        elif slope < 0:
            detected_trend = 'downward'
        else:
            detected_trend = 'stationary'

        # Update trend if the detected trend differs from 'stationary'
        if trend == 'stationary' and detected_trend != 'stationary':
            trend = detected_trend
        else:
            trend = 'stationary'
    
    # Step 3: Visualization (optional)
    if view:
        plt.figure(figsize=fig_size)
        
        # Plot the original data
        plt.plot(df[dt_col], label='Original Data', color='blue', **kw)
        plt.title(f"Trend Analysis for {dt_col}", fontsize=14, fontweight='bold')
        plt.xlabel('Time', fontsize=12)
        plt.ylabel(target.name, fontsize=12)
        
        # Add a horizontal line for the mean if stationary
        if trend == 'stationary':
            plt.axhline(y=target.mean(), color=stationnay_color,
                        linestyle=linestyle, label='Mean Line')
        
        # Plot the fitted trend line for non-stationary or detected trend
        else:
            plt.plot(df.index, model.fittedvalues, color=trend_color,
                     label='Fitted Trend Line', linewidth=2)

        # Customize gridlines
        if not show_grid: 
            plt.grid(False)
        else: 
            plt.grid(True, linestyle=':', alpha=0.7)

        # Add annotations for clarity
        plt.text(0.05, 0.95, f"Detected Trend: {trend.capitalize()}", 
                 transform=plt.gca().transAxes, fontsize=12, color=trend_color, 
                 fontweight='bold', ha='left', va='top')
        
        # Add stationarity test result
        plt.text(0.05, 0.85, 
                 f"Stationarity Test ({strategy.upper()}) p-value: {p_value:.3f}", 
                 transform=plt.gca().transAxes, fontsize=10, 
                 color='black', 
                 fontweight='normal',
                 ha='left', va='top'
                 )
        
        # Display the legend
        plt.legend()

        # Show the plot
        plt.tight_layout()
        plt.show()

    return trend, p_value

def trend_ops(
    df, 
    dt_col, 
    value_col,
    ops=None, 
    check_stationarity=True, 
    trend_type='both', 
    error='raise', 
    strategy="adf",  # Use 'adf' or 'kpss' for stationarity test
    verbose=0,  # verbose level: 0 (no output), 1 (minimal), 2 (detailed), 3 (very detailed)
    view=False,  # Plotting flag to view data vs detrended data
    fig_size=(10, 4),  # Figure size for plotting
    show_grid=False,  # Grid option for plotting
    **kw # Additional keyword arguments for customization
):
    """
    Perform operations (removal of trends, detrending, etc.) on the time 
    series data based on detected trends.

    Parameters:
    - df: DataFrame containing the time series data.
    - value_col: Column name or series of the target variable to forecast.
    - dt_col: Column name of the time series data (datetime-related column).
    - ops: Operation to perform. Options: ['remove_upward', 'remove_downward', 'remove_both', 'detrend', 'none']
    - check_stationarity: Whether to check the stationarity of the series using ADF/KPSS.
    - trend_type: Type of trend detection ('both', 'upward', 'downward').
    - error: How to handle errors when no transformation is applied ('raise', 'warn', 'ignore').
    - strategy: The stationarity test to use ('adf' or 'kpss').
    - verbose: Level of logging for tracking the process.
    - view: Whether to visualize the original and transformed data.
    - fig_size: Figure size for plots.
    - show_grid: Whether to show gridlines on the plot.
    
    Returns:
    - Transformed DataFrame based on the operation requested.
    """
    
    # Validate the input DataFrame
    is_frame(df, df_only=True, raise_exception=True, objname="Dataframe 'df'")
    is_time_series(df, time_col=dt_col, check_time_interval=False)

    
    # Step 1: Use trend_analysis to detect trends and stationarity
    trend, p_value = trend_analysis(
        df, value_col=value_col, dt_col=dt_col, 
        check_stationarity=check_stationarity, 
        trend_type=trend_type, view=False, 
        strategy=strategy, 
        **kw
    )
    # Step 2: Validate and process the target using validate_target_in
    target, _ = validate_target_in(df, value_col, error=error, verbose=verbose) 
    tname= target.name 
    
    df.set_index (dt_col, inplace =True)
    detrended_data=pd.DataFrame () # initialize the detrended data 
    # Verbose Logging: Print detected trend and stationarity test result
    if verbose >= 1:
        print(f"Detected Trend: {trend}")
        if check_stationarity:
            print(f"Stationarity Test p-value: {p_value:.4f}")
     
    # Step 2: Depending on the operation requested, transform the data
    if ops == 'remove_upward':
        if trend != 'upward':
            if verbose >= 1:
                print("No upward trend detected. Skipping upward removal.")
        else:
            # Detrend by removing the upward trend (subtract fitted values)
            X = np.arange(len(df)).reshape(-1, 1)  # Time index
            y = df[tname].values
            model = sm.OLS(y, sm.add_constant(X)).fit()
            detrended_data = df[tname] - model.fittedvalues
            #df[tname] = detrended_data
            if verbose >= 1:
                print("Upward trend removed.")
    
    elif ops == 'remove_downward':
        if trend != 'downward':
            if verbose >= 1:
                print("No downward trend detected. Skipping downward removal.")
        else:
            # Detrend by removing the downward trend (subtract fitted values)
            X = np.arange(len(df)).reshape(-1, 1)
            y = df[tname].values
            model = sm.OLS(y, sm.add_constant(X)).fit()
            detrended_data = df[tname] - model.fittedvalues
            #df[tname] = detrended_data
            if verbose >= 1:
                print("Downward trend removed.")
    
    elif ops == 'remove_both':
        if trend == 'stationary':
            if verbose >= 1:
                print("Data is already stationary. Skipping trend removal.")
        else:       
            # Detrend by removing both upward and downward trends
            X = np.arange(len(df)).reshape(-1, 1)
            y = df[tname].values
            model = sm.OLS(y, sm.add_constant(X)).fit()
            detrended_data = df[tname] - model.fittedvalues
            # df[tname] = detrended_data
            if verbose >= 1:
                print("Both upward and downward trends removed.")
    
    elif ops == 'detrend' and trend != 'stationary':
        if trend == 'stationary':
            if verbose >= 1:
                print("Data is already stationary. Skipping differencing.")
        else: 
            # Detrend the series using differencing
            detrended_data = df[tname].diff().dropna()
            if verbose >= 1:
                print("Data detrended using differencing.")
        
    elif ops == 'none':
        if verbose >= 1:
            print("No transformation applied.")
    
    # Step 3: Handle potential errors if the operation results in NaN values
    if ops is not None and df[tname].isnull().all():
        if error == 'raise':
            raise ValueError(f"After {ops}, the data became entirely null.")
        elif error == 'warn':
            warnings.warn(f"After {ops}, the data became entirely null.")
    
    # Step 4: Plotting if view=True (Visualize original vs. transformed data)
    if view:
        if detrended_data.empty: 
            if verbose >= 1:
                print(f"No transformation applied for trend='{trend}'"
                      f" and ops='{ops}'. Skipping visualization.")
        else: 
            fig, axes = plt.subplots(1, 2, figsize=fig_size)
            # Plot the original data on the first subplot
            axes[0].plot(df.index, df[tname], label="Original Data", color='blue')
            axes[0].set_title("Original Data with Trend")
            axes[0].set_xlabel("Time")
            axes[0].set_ylabel(tname)
            axes[0].grid(show_grid)
    
            # Plot the transformed (detrended) data on the second subplot
            axes[1].plot(df.index, detrended_data, label="Transformed Data", color='green')
            axes[1].set_title(f"Transformed Data (After {ops})")
            axes[1].set_xlabel("Time")
            axes[1].set_ylabel(tname)
            axes[1].grid(show_grid)
            
            plt.tight_layout()
            plt.show()

    if detrended_data.empty: 
        return df 
    df[tname]= detrended_data

    # Step 5: Return the transformed DataFrame
    return df


def visual_inspection(
    df, value_col, 
    dt_col=None, 
    window=12,  # Window size for rolling statistics (adjustable)
    seasonal_period=None,  # Specify seasonal period if known (e.g., 12 for monthly data with yearly seasonality)
    figsize=(14, 8),  # Size of the figure for plotting
    show_acf=True,  # Whether to show the autocorrelation plot
    show_decomposition=True,  # Whether to decompose the series and show the decomposition plots
    show_trend=True,  # Whether to plot trend (rolling mean)
    show_seasonal=True,  # Whether to plot seasonal (rolling std)
    show_residual=True,  # Whether to plot residuals (after subtracting trend and seasonal components)
    show_grid=True,  # Show gridlines in the plot
    max_cols=3,  # Max columns in the subplot (adjustable)
    decompose_on_sep=False,  # Whether to plot decomposition in a separate figure
    title="Time Series Visual Inspection",  # Title for the main time series plot
    **kwargs  # Additional keyword arguments for plot customization
):
    # Validate the input
    is_frame(df, df_only=True, raise_exception=True, objname="Dataframe 'df'")
    # is_time_series(df, time_col=dt_col, check_time_interval=False)
    
    # Extract the time series
    ts, _ = validate_target_in(df, value_col, error='raise', verbose=0) 
    tname= ts.name 
    # ts = df[dt_col]

    # Determine number of subplots based on what needs to be shown
    num_plots = 0
    if show_trend:
        num_plots += 1
    if show_seasonal:
        num_plots += 1
    if show_acf:
        num_plots += 1
    if seasonal_period and show_decomposition and not decompose_on_sep:
        num_plots += 4
    if seasonal_period and show_decomposition and show_residual:
        num_plots += 1
    
    # Adjust rows and columns of subplots dynamically
    num_rows = (num_plots // max_cols) + (num_plots % max_cols > 0)  # Round up for odd number of plots
    num_cols = min(max_cols, num_plots)  # Limit columns based on max_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()  # Flatten axes for easy indexing

    plot_idx = 0
    if dt_col is None: 
        dt_col ='Date/Time'
    else: 
        dt_col=str(dt_col).title()
    
    # Plot original time series in the first subplot
    axes[plot_idx].plot(df.index, ts, label="Original Data", color='blue', **kwargs)
    axes[plot_idx].set_title(f"{title}: Original Time Series")
    axes[plot_idx].set_xlabel(dt_col)
    axes[plot_idx].set_ylabel(tname)
    if show_grid:
        axes[plot_idx].grid(True, linestyle=':', alpha=0.7)
    else:
        axes[plot_idx].grid(False)
    plot_idx += 1

    # Plot rolling mean (trend) if needed
    if show_trend:
        rolling_mean = ts.rolling(window=window).mean()
        axes[plot_idx].plot(df.index, ts, label="Original Data", color='blue', alpha=0.5)
        axes[plot_idx].plot(df.index, rolling_mean, label="Rolling Mean (Trend)", color='red')
        axes[plot_idx].set_title(f"Rolling Mean (Trend) - Window={window}")
        axes[plot_idx].set_xlabel(dt_col)
        axes[plot_idx].set_ylabel(tname)
        if show_grid:
            axes[plot_idx].grid(True, linestyle=':', alpha=0.7)
        else:
            axes[plot_idx].grid(False)
        plot_idx += 1
    
    # Plot rolling standard deviation (seasonality) if needed
    if show_seasonal:
        rolling_std = ts.rolling(window=window).std()
        axes[plot_idx].plot(df.index, rolling_std, label="Rolling Std (Seasonality)", color='green')
        axes[plot_idx].set_title(f"Rolling Standard Deviation (Seasonality) - Window={window}")
        axes[plot_idx].set_xlabel(dt_col)
        axes[plot_idx].set_ylabel("Rolling Std")
        if show_grid:
            axes[plot_idx].grid(True, linestyle=':', alpha=0.7)
        else:
            axes[plot_idx].grid(False)
        plot_idx += 1
    
    # Show autocorrelation plot (ACF) if needed
    if show_acf:
        plot_acf(ts, ax=axes[plot_idx], lags=40)
        axes[plot_idx].set_title("Autocorrelation (ACF)")
        plot_idx += 1
    
    # Decompose the series if seasonal period is provided

    if seasonal_period is not None and show_decomposition:
        decomposition = seasonal_decompose(
            ts, model='additive', period=seasonal_period)
        
        if decompose_on_sep:
            # Create separate decomposition plots to avoid 'ax' error
            w= int(np.ceil([figsize[0]])/2 )
            figsize= tuple ([w, w])
            fig_decomp, axes_decomp = plt.subplots(4, 1, figsize=figsize)
            
            # Plot decomposition components
            decomposition.observed.plot(ax=axes_decomp[0], label='Observed', color='blue')
            decomposition.trend.plot(ax=axes_decomp[1], label='Trend', color='red')
            decomposition.seasonal.plot(ax=axes_decomp[2], label='Seasonal', color='green')
            decomposition.resid.plot(ax=axes_decomp[3], label='Residuals', color='purple')
        
            # Set titles for each subplot
            axes_decomp[0].set_title('Observed')
            axes_decomp[1].set_title('Trend')
            axes_decomp[2].set_title('Seasonal')
            axes_decomp[3].set_title('Residuals')
        
            # Set x-labels and y-labels for each subplot
            for ax in axes_decomp:
                ax.set_xlabel(dt_col)
                ax.set_ylabel(tname)  # dt_col is the name of the time series column
            
            # Apply grid to each subplot
            for ax in axes_decomp:
                if show_grid:
                    ax.grid(True, linestyle=':', alpha=0.7)
                else:
                    ax.grid(False)
            
            # Set the overall title for the decomposition
            plt.suptitle("Seasonal Decomposition", fontsize=16)
            
            # Adjust layout to avoid overlap
            plt.tight_layout()
            plt.subplots_adjust(top=0.95)  # Adjust the top margin to avoid title overlap
        
            # Show the decomposition plot
            plt.show()
            
        else:
            # Add decomposition plots to the current figure
            decomposition.observed.plot(ax=axes[plot_idx], label='Observed', color='blue')
            decomposition.trend.plot(ax=axes[plot_idx + 1], label='Trend', color='red')
            decomposition.seasonal.plot(ax=axes[plot_idx + 2], label='Seasonal', color='green')
            decomposition.resid.plot(ax=axes[plot_idx + 3], label='Residuals', color='purple')
            
            # Set titles for each subplot
            axes[plot_idx].set_title('Observed')
            axes[plot_idx + 1].set_title('Trend')
            axes[plot_idx + 2].set_title('Seasonal')
            axes[plot_idx + 3].set_title('Residuals')
        
            # Set x-labels and y-labels for each subplot
            for ax in axes[plot_idx:plot_idx + 4]:  # Loop through the decomposition axes
                ax.set_xlabel("Time")
                ax.set_ylabel(dt_col)  # dt_col is the name of the time series column
            
            # Apply grid to each subplot
            for ax in axes[plot_idx:plot_idx + 4]:  # Loop through the decomposition axes
                if show_grid:
                    ax.grid(True, linestyle=':', alpha=0.7)
                else:
                    ax.grid(False)
            
            plot_idx += 4  # Move the index forward for residual plot
        
        # Plot residuals after removing trend and seasonality 
        # (if decomposition is done)
        if show_residual:
            residual = decomposition.resid.dropna()
            axes[plot_idx].plot(df.index[:len(residual)], 
                                residual, 
                                label="Residuals", color='purple')
            axes[plot_idx].set_title("Residuals (After Trend and Seasonality Removal)")
            axes[plot_idx].set_xlabel("Time")
            axes[plot_idx].set_ylabel("Residuals")
            if show_grid:
                axes[plot_idx].grid(True, linestyle=':', alpha=0.7)
            else:
                axes[plot_idx].grid(False)
            plot_idx += 1

    # If the number of predictions is less than total
    # subplots, hide the unused axes
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')
        
    # Finalize the plot layout and show the plots
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust the top margin to avoid title overlap
    plt.show()


def infer_decomposition_method(
    df, dt_col, 
    period=12, 
    return_components=False , 
    view=False, 
    figsize= (10, 8)
    ):
    """
    Automatically selects the best decomposition method (additive or multiplicative)
    based on the characteristics of the time series in the dataframe.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the time series data.
    dt_col (str): The column name containing datetime information.
    
    Returns:
    str: Best decomposition method ('additive' or 'multiplicative')
    dict: Decomposition components for the selected method
    """
    # Validate the input
    is_frame(df, df_only=True, raise_exception=True, objname="Dataframe 'df'")
    is_time_series(df, time_col=dt_col, check_time_interval=False)
    
    # Ensure the datetime column is in the correct format
    if not pd.api.types.is_datetime64_any_dtype(df[dt_col]):
        df[dt_col] = pd.to_datetime(df[dt_col])

    # Set the datetime column as the index
    df.set_index(dt_col, inplace=True)
    
    # Drop missing values before decomposition (if any)
    df.dropna(inplace=True)

    # Perform both additive and multiplicative decompositions
    def decompose_series(series, method):
        return seasonal_decompose(series, model=method, period=period)  

    additive_decomp = decompose_series(df.iloc[:, 0], 'additive')
    multiplicative_decomp = decompose_series(df.iloc[:, 0], 'multiplicative')
    
    # Calculate the residuals for both decompositions
    residual_additive = additive_decomp.resid.dropna()
    residual_multiplicative = multiplicative_decomp.resid.dropna()

    # Calculate the variance of residuals as a measure of goodness of fit
    var_additive = np.var(residual_additive)
    var_multiplicative = np.var(residual_multiplicative)

    # Visualize the residuals
    if view:
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        sns.histplot(residual_additive, kde=True, ax=axes[0], 
                     color='blue')
        axes[0].set_title("Residuals - Additive Decomposition")
        
        sns.histplot(residual_multiplicative, kde=True, ax=axes[1], 
                     color='green')
        axes[1].set_title("Residuals - Multiplicative Decomposition")
        
        plt.tight_layout()
        plt.show()

    # Choose the best method based on the lower variance of residuals
    if var_additive < var_multiplicative:
        best_method = 'additive'
        best_decomp = additive_decomp
    else:
        best_method = 'multiplicative'
        best_decomp = multiplicative_decomp

    # Return the best method and the components
    if return_components : 
        return best_method, {
            'trend': best_decomp.trend,
            'seasonal': best_decomp.seasonal,
            'residual': best_decomp.resid
        }
    
    return best_method 

def decompose_ts(
    df, value_col, dt_col=None,  
    method='additive',
    strategy='STL', 
    seasonal_period=12,
    robust=True
    ):
    """
    Decompose a time series (dt_col) into trend, seasonal, and residual components,
    while keeping the rest of the features intact.

    Parameters:
    - df: DataFrame containing the time series data.
    - dt_col: Column name of the time series data to decompose.
    - method: Decomposition method, either 'additive' or 'multiplicative'.
    - strategy: Decomposition strategy, either 'STL' or 'SDT'.
    - seasonal_period: Seasonal period (e.g., 12 for monthly data with yearly seasonality).
    - robust: Whether to apply robust STL decomposition (default True).

    Returns:
    - decomposed_df: DataFrame with the components (trend, seasonal, residual) 
    for the dt_col and all other features.
    """
    
    # Validate the input
    # Extract the time series
    ts, _ = validate_target_in(df, value_col, error='raise', verbose=0) 
    tname= ts.name 

    # ts = df[dt_col]

    # Ensure seasonal_period is a valid odd number >= 3
    if seasonal_period % 2 == 0:
        seasonal_period += 1  # Adjust to next odd number if even
    if seasonal_period < 3:
        raise ValueError("The seasonal period must be an odd integer >= 3.")
    
    strategy = str(strategy).lower()
    # Decompose based on the chosen strategy
    if strategy == 'stl':
        # STL decomposition
        stl = STL(ts, seasonal=seasonal_period, 
                  trend=seasonal_period, 
                  robust=robust
                  )
        result = stl.fit()
        decomposed_df = pd.DataFrame({
            'trend': result.trend,
            'seasonal': result.seasonal,
            'residual': result.resid
        })
    
    elif strategy == 'sdt':
        # Seasonal decomposition of time series (additive or multiplicative)
        result = seasonal_decompose(ts, model=method, period=seasonal_period)
        decomposed_df = pd.DataFrame({
            'trend': result.trend,
            'seasonal': result.seasonal,
            'residual': result.resid
        })
    
    else:
        raise ValueError("Invalid strategy. Choose either 'STL' or 'SDT'.")
    
    # Combine the decomposed components with the original features in the DataFrame
    decomposed_df[tname] = df[tname]  # Add the original time series column
    for col in df.columns:
        if col != tname:
            decomposed_df[col] = df[col]  # Retain other features in the DataFrame
    
    return decomposed_df

def ts_engineering(
    df, value_col, 
    dt_col=None, 
    lags=5, 
    window=7, 
    diff_order=1, 
    seasonal_period=None, 
    apply_fourier=False, 
    holiday_df=None, 
    robust_diff=True,
    scaler='z-norm', # norm or 'minmax;scaler ', 'z-norm', for standardScaler 
    
    **kwargs
    ):
    """
    Feature engineering for time series data to create relevant features for ML models.
    
    Parameters:
    - df: DataFrame containing the time series data.
    - dt_col: Column name of the time series data to perform engineering on.
    - lags: Number of lag features to create (e.g., lags=5 creates lag-1, lag-2, ..., lag-5).
    - window: Window size for rolling statistics (e.g., window=7 for 7-day rolling window).
    - diff_order: The order of differencing to apply (0 for no differencing, 1 for first differencing).
    - seasonal_period: Period for seasonal differencing (e.g., 12 for monthly data with yearly seasonality).
    - apply_fourier: Whether to apply Fourier transform to capture periodicity.
    - holiday_df: DataFrame with holiday dates, if relevant, to create a holiday indicator.
    - robust_diff: Whether to apply robust differencing (for non-stationary data).
    - kwargs: Additional parameters for customization, such as handling missing values.
    
    Returns:
    - df: DataFrame with engineered features.
    """
    
    # Extract the time series
    ts, _ = validate_target_in(df, value_col, error='raise', verbose=0) 
    tname= ts.name 

    
    # 1. Create time-based features
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['day_of_week'] = df.index.dayofweek  # 0=Monday, 6=Sunday
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)  # Weekend indicator
    df['quarter'] = df.index.quarter
    df['hour'] = df.index.hour if df.index.freq == 'H' else 0
    df['is_holiday'] = 0  # Default holiday indicator
    if holiday_df is not None:
        df['is_holiday'] = df.index.isin(holiday_df['date']).astype(int)
    
    # 2. Create lag features
    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = ts.shift(lag)
    
    # 3. Create rolling statistics (moving averages, rolling std)
    df[f'rolling_mean_{window}'] = ts.rolling(window=window).mean()
    df[f'rolling_std_{window}'] = ts.rolling(window=window).std()
    
    # 4. Apply differencing (first or seasonal)
    if diff_order > 0:
        df[f'{tname}_diff'] = ts.diff(diff_order)
    
    if seasonal_period and seasonal_period > 0:
        df[f'{tname}_seasonal_diff'] = ts.diff(seasonal_period)
    
    # 5. Apply Fourier Transform (optional, if apply_fourier=True)
    if apply_fourier:
        fft_values = fft(ts.fillna(0))  # Handle missing values for FFT
        fft_features = np.abs(fft_values[:len(ts)//2])  # Take the first half (real part)
        df[['fft_' + str(i) for i in range(1, len(fft_features)+1)]] = pd.DataFrame(fft_features).T
    
    # 6. Handle missing values in features, if needed
    df.fillna(method='ffill', inplace=True)  # Forward fill missing values
    df.dropna(inplace=True)  # Drop any remaining missing values (if applicable)
    
    # 7. Scaling or normalization of features (optional)
    if scaler is not None: 
        scaler = StandardScaler() if scaler=='z-norm' else MinMaxScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(
            df.select_dtypes(include=[np.number])),
            columns=df.select_dtypes(include=[np.number]).columns
            )
        df[df_scaled.columns] = df_scaled
    
    return df


def prepare_ts_df(
    df, dt_col=None, 
    set_index=True, 
    error='raise',
    use_smart_ts_formatter=False, 
    verbose=0, 
    ):
    """
    Prepares the DataFrame for time series operations by ensuring the index
    is a datetime type.
    If the index is not a datetime and no time column is specified, an 
    error or warning is raised.

    Parameters:
    - df: DataFrame containing the data.
    - time_col: Column to be used as the time column. If not specified, 
    the index will be checked.
    - set_index: Whether to set the specified time column as the index (default is True).
    - error: Whether to raise an error or a warning if no valid datetime 
    column is found. Options: 'raise', 'warn', 'ignore'.
    
    Returns:
    - df: DataFrame with the correct index set for time series operations.
    """
    
    # 1. Check if the index is already a datetime
    if pd.api.types.is_datetime64_any_dtype(df.index):
        print("Index is already a datetime object.")
        if not set_index: 
            df.reset_index (inplace=True) 
            
        return df  # No need to change anything
    
    if use_smart_ts_formatter: # mean use_smart_ts_formatter 
        # Step 2: Validate and extract the target column
        df= ts_validator(
            df, dt_col=dt_col, 
            to_datetime='auto', 
            as_index=set_index, 
            error=error,
            return_dt_col=False, 
            verbose=verbose
        )
        return df 
    
    # 2. If the index is not datetime, check if a time column is specified
    if dt_col is not None:
                
        if dt_col not in df.columns:
            raise ValueError(f"Column '{dt_col}' not found in the DataFrame.")
        
        # Set the specified column as the index and convert to datetime
        df[dt_col] = pd.to_datetime(df[dt_col], errors='coerce')
        
        if df[dt_col].isnull().any():
            raise ValueError(f"Column '{dt_col}' contains invalid date"
                             " formats that could not be converted.")
        
        if set_index:
            df.set_index(dt_col, inplace=True)
        
        if verbose>=1:
            print(f"Column '{dt_col}' has been set as the"
                  " index and converted to datetime.")
        return df

    # 3. If no time_col is specified, we need to handle the
    # case based on the error parameter
    if error == 'raise':
        raise ValueError("The index is not a datetime object,"
                         " and no 'time_col' was specified.")
    elif error == 'warn':
        warnings.warn("The index is not a datetime object, and no"
              " 'time_col' was specified. Default behavior will apply.")
    elif error == 'ignore':
        # print("Index is not a datetime, but no action will be taken.")
        return df
    
    # 4. If no valid datetime column found and the user didn’t specify a"
    # time column, return the original df
    if verbose>0:
        print("No datetime index or time column found."
              " Returning the DataFrame as is.")
    return df

def ts_corr_analysis(
    df, 
    dt_col, 
    value_col, 
    lags=40,  # Number of lags for ACF/PACF
    features=None,  # External features for cross-correlation
    view_acf_pacf=True,  # Whether to visualize ACF and PACF
    view_cross_corr=True,  # Whether to visualize cross-correlation
    fig_size=(14, 6),  # Size of ACF/PACF plots
    verbose=0,  # Verbosity level
    show_grid=True,  # Grid option for plots
    cross_corr_on_sep=False  # Whether to plot cross-corr on a separate figure
):
    """
    Perform correlation analysis on a time series dataset, including ACF/PACF
    and cross-correlation analysis with external features.

    Parameters:
    - df: pandas.DataFrame
        Input DataFrame containing the time series data.
    - dt_col: str
        Column name representing the datetime-based series (e.g., 'time').
    - value_col: str
        Column name of the target variable (e.g., 'sales').
    - lags: int, optional (default=40)
        Number of lags for ACF/PACF analysis.
    - features: list, optional
        List of external features for cross-correlation.
    - view_acf_pacf: bool, optional (default=True)
        Whether to visualize ACF and PACF.
    - view_cross_corr: bool, optional (default=True)
        Whether to visualize cross-correlation with external features.
    - fig_size: tuple, optional (default=(14, 6))
        Figure size for ACF/PACF plots.
    - verbose: int, optional (default=0)
        Verbosity level.
    - show_grid: bool, optional (default=True)
        Whether to show gridlines in the plots.
    - cross_corr_on_sep: bool, optional (default=False)
        Whether to plot cross-correlation on a separate figure.

    Returns:
    - results: dict
        Dictionary containing:
        - 'acf_values': ACF values up to the specified lags.
        - 'pacf_values': PACF values up to the specified lags.
        - 'cross_corr': Cross-correlation coefficients with external features.
    """
    # Step 1: Validate input DataFrame
    is_frame(df, df_only=True, raise_exception=True, objname="DataFrame 'df'")
    # Step 2: Validate target column and extract it
    target, df = validate_target_in(df, value_col, verbose=verbose )
    
    # Step 3: Ensure datetime column is valid
    df, dt_col = ts_validator(
        df, dt_col=dt_col, 
        to_datetime='auto', 
        as_index=False, 
        error="raise",
        return_dt_col= True, 
        verbose=verbose
    )
    
    # Step 4: ACF and PACF analysis
    if verbose >= 1:
        print("Performing ACF and PACF analysis...")

    # Step 4: Validate and process features for cross-correlation
    features = columns_manager (features, empty_as_none= True )
    if features is not None:
        exist_features(df, features=features, name="Features for cross-correlation")
    else:
        features = [col for col in df.columns if col != value_col and col != dt_col]

    if verbose >= 1:
        print(f"Target variable: {value_col}")
        print(f"Datetime column: {dt_col}")
        if features:
            print(f"Cross-correlation features: {features}")

    # Step 5: Perform ACF and PACF analysis
    acf_values, pacf_values = None, None
    if view_acf_pacf:
        # Subplot configuration based on cross_corr_on_sep
        if view_cross_corr and not cross_corr_on_sep:
            fig = plt.figure(figsize=(fig_size[0], fig_size[1] * 1.5))
            gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.7])  # GridSpec for flexible layout
            
            # ACF and PACF plots in the first row
            ax_acf = fig.add_subplot(gs[0, 0])
            ax_pacf = fig.add_subplot(gs[0, 1])
            ax_cross_corr = fig.add_subplot(gs[1, :])  # Cross-corr spans the second row
            
        else:
            fig, axes = plt.subplots(1, 2, figsize=fig_size)
            ax_acf, ax_pacf = axes
            ax_cross_corr = None
        
        # ACF plot
        plot_acf(target, lags=lags, ax=ax_acf)
        ax_acf.set_title("Autocorrelation Function (ACF)")
        ax_acf.set_xlabel("Lags")
        ax_acf.set_ylabel("ACF")
        ax_acf.grid(show_grid, linestyle=":", alpha=0.7)

        # PACF plot
        plot_pacf(target, lags=lags, ax=ax_pacf, method='ywm')
        ax_pacf.set_title("Partial Autocorrelation Function (PACF)")
        ax_pacf.set_xlabel("Lags")
        ax_pacf.set_ylabel("PACF")
        ax_pacf.grid(show_grid, linestyle=":", alpha=0.7)

        
        if show_grid: 
            ax_acf.grid(show_grid, linestyle=":", alpha=0.7)
            ax_pacf.grid(show_grid, linestyle=":", alpha=0.7)
        else: 
            ax_acf.grid(False)
            ax_pacf.grid(False)

        # Extract ACF and PACF values
        acf_values = pd.Series(
            np.correlate(target, target, mode='full')[-lags:], index=range(lags)
            )
        pacf_values = None  # PACF values can be separately computed if needed

    # Step 6: Perform cross-correlation analysis
    cross_corr_results = {}
    if features:
        if verbose >= 1:
            print("Performing cross-correlation analysis...")
        
        for feature in features:
            correlation, p_value = pearsonr(target, df[feature])
            cross_corr_results[feature] = {'correlation': correlation, 'p_value': p_value}

            if verbose >= 2:
                print(f"Cross-correlation with {feature}: Correlation"
                      f" = {correlation:.4f}, p-value = {p_value:.4f}")
        
        # Cross-correlation plot
        if view_cross_corr:
            if cross_corr_on_sep:
                # Separate cross-correlation figure
                fig_cross_corr, ax_cross_corr_sep = plt.subplots(
                    figsize=(fig_size[0], fig_size[1] // 2))
                ax_cross_corr_sep.bar(features, [cross_corr_results[f]['correlation'] 
                                                 for f in features], color='skyblue')
                ax_cross_corr_sep.set_title("Cross-Correlation with External Features")
                ax_cross_corr_sep.set_xlabel("Features")
                ax_cross_corr_sep.set_ylabel("Correlation Coefficient")
                ax_cross_corr_sep.grid(show_grid, linestyle=":", alpha=0.7)
                plt.xticks(rotation=45)
                
            elif ax_cross_corr is not None:
                # Cross-correlation on the same figure
                ax_cross_corr.bar(features, [cross_corr_results[f]['correlation'] 
                                             for f in features], color='skyblue')
                ax_cross_corr.set_title("Cross-Correlation with External Features")
                ax_cross_corr.set_xlabel("Features")
                ax_cross_corr.set_ylabel("Correlation Coefficient")
                ax_cross_corr.grid(show_grid, linestyle=":", alpha=0.7)
                plt.xticks(rotation=45)
         
    # Adjust layout and show plot
    if view_acf_pacf:
        plt.tight_layout()
        plt.show()

    # Step 7: Compile results
    results = {
        'acf_values': acf_values,
        'pacf_values': pacf_values,
        'cross_corr': cross_corr_results
    }

    summary = ResultSummary("CrossCorrResults",flatten_nested_dicts=False) 
    summary.add_results(results['cross_corr']) 
    
    if verbose>=1: 
        print(summary)
        
    return results

def transform_stationarity(
    df,
    dt_col=None,  # Datetime column
    value_col=None,  # Target column (time series variable)
    method="differencing",  # Transformation method: 'differencing', 'log', 'sqrt', or 'detrending'
    order=1,  # Order of differencing (1 for first difference, 2 for second, etc.)
    seasonal_period=None,  # Seasonal period for seasonal differencing
    detrend_method="linear",  # Detrending method: 'linear' or 'stl'
    view=True,  # Whether to visualize the transformations
    fig_size=(12, 6),  # Size of the visualization plot
    show_grid=True,  # Grid option for the plots
    drop_original=True, 
    reset_index=False, 
    verbose=0  # Verbosity level
):
    """
    Perform stationarity transformations on a time series dataset, including differencing,
    variance stabilization, and detrending.

    Parameters:
    - df: pandas.DataFrame
        The input DataFrame containing the time series data.
    - dt_col: str, optional
        Column name representing the datetime-based series (e.g., 'time').
    - value_col: str, required
        Column name of the target variable (e.g., 'sales').
    - method: str, optional (default='differencing')
        The transformation method to apply:
        - 'differencing': Apply differencing to stabilize the mean.
        - 'log': Apply logarithmic transformation to stabilize variance.
        - 'sqrt': Apply square root transformation to stabilize variance.
        - 'detrending': Remove the trend using linear regression or STL decomposition.
    - order: int, optional (default=1)
        The order of differencing (only applies if `method='differencing'`).
    - seasonal_period: int, optional
        Seasonal period for seasonal differencing (e.g., 12 for monthly data).
    - detrend_method: str, optional (default='linear')
        Method to use for detrending:
        - 'linear': Fit a linear trend using regression and subtract it.
        - 'stl': Use STL decomposition to remove the trend.
    - view: bool, optional (default=True)
        Whether to visualize the original and transformed data.
    - fig_size: tuple, optional (default=(12, 6))
        Size of the visualization plot.
    - show_grid: bool, optional (default=True)
        Whether to show gridlines on the plots.
    - verbose: int, optional (default=0)
        Verbosity level.

    Returns:
    - transformed_df: pandas.DataFrame
        DataFrame containing the transformed time series data.
    """
    # Step 1: Validate the input DataFrame
    is_frame(df, df_only=True, raise_exception=True, objname="DataFrame 'df'")
    
    # Step 2: Validate and extract the target column
    target, df = validate_target_in(df, value_col )
    tname = target.name  # Get the name of the target variable
    
    # Step 3: Ensure datetime column is valid
    df, dt_col = ts_validator(
        df, dt_col=dt_col, 
        to_datetime='auto', 
        as_index=True, 
        error="raise",
        return_dt_col= True, 
        verbose=verbose
    )
    target.index = df.index 
    
    if verbose >= 1:
        print(f"Target variable: {tname}")
        print(f"Datetime column: {dt_col}")
        print(f"Transformation method: {method}")

    # Step 4: Perform the selected transformation
    if method == "differencing":
        if seasonal_period:
            if verbose >= 1:
                print(f"Applying seasonal differencing with period={seasonal_period}.")
            transformed_data = target.diff(seasonal_period).dropna()
        else:
            if verbose >= 1:
                print(f"Applying first-order differencing with order={order}.")
            transformed_data = target.diff(order).dropna()

    elif method == "log":
        if verbose >= 1:
            print("Applying logarithmic transformation.")
        if (target <= 0).any():
            raise ValueError("Log transformation cannot be applied to non-positive values.")
        transformed_data = np.log(target)

    elif method == "sqrt":
        if verbose >= 1:
            print("Applying square root transformation.")
        if (target < 0).any():
            raise ValueError("Square root transformation cannot be applied to negative values.")
        transformed_data = np.sqrt(target)

    elif method == "detrending":
        if detrend_method == "linear":
            if verbose >= 1:
                print("Applying linear detrending.")
            time_index = np.arange(len(target)).reshape(-1, 1)
            trend = np.polyfit(time_index.flatten(), target.values, 1)  # Linear regression
            trend_line = np.polyval(trend, time_index)
            transformed_data = target - trend_line.flatten()
        elif detrend_method == "stl":
            if verbose >= 1:
                print("Applying STL detrending.")
            stl = STL(target, period=seasonal_period or 7)  # Default weekly seasonality if not specified
            result = stl.fit()
            transformed_data = result.resid
        else:
            raise ValueError(f"Invalid detrend_method: {detrend_method}")
    else:
        raise ValueError(f"Invalid method: {method}")

    # Step 5: Visualize the transformation (if enabled)
    if view:
        plt.figure(figsize=fig_size)
        
        # Plot original data
        plt.subplot(2, 1, 1)
        plt.plot(target, label="Original Data", color="blue")
        plt.title("Original Time Series")
        plt.xlabel("Time")
        plt.ylabel(tname)
        if show_grid:
            plt.grid(True, linestyle=":", alpha=0.7)
        else: 
            plt.grid(False)
        # Plot transformed data
        plt.subplot(2, 1, 2)
        plt.plot(transformed_data, label=f"Transformed Data ({method})", color="green")
        plt.title(f"Transformed Time Series ({method})")
        plt.xlabel("Time")
        plt.ylabel(f"{tname} (Transformed)")
        if show_grid:
            plt.grid(True, linestyle=":", alpha=0.7)
        else: 
            plt.grid(False)
            
        plt.tight_layout()
        plt.show()

    # Step 6: Return the transformed DataFrame
    transformed_df = df.copy()
    
    if not drop_original: 
        transformed_df [tname]=target 
        
    transformed_df[f"{tname}_transformed"] = transformed_data
    
    if reset_index: 
        transformed_df.reset_index (inplace =True )

    return transformed_df

def ts_split(
    df, 
    dt_col=None,  # Column representing the datetime
    value_col=None,  # Target column for splitting
    split_type="simple",  # 'simple' or 'cv' (cross-validation)
    test_ratio=None,  # Size of the test set (number of rows or fraction)
    n_splits=5,  # Number of splits for cross-validation
    gap=0,  # Gap between train and test sets in cross-validation
    train_start=None,  # Start date for training set (only for simple split)
    train_end=None,  # End date for training set (only for simple split)
    verbose=0  # Verbosity level
):
    """
    Perform a time-based split on a time series dataset.

    Parameters:
    - df: pandas.DataFrame
        The input DataFrame containing the time series data.
    - dt_col: str, optional
        Column name representing the datetime (required for simple splits with train_start/train_end).
    - value_col: str, optional
        Column name of the target variable (e.g., sales, temperature).
    - split_type: str, optional (default='simple')
        The type of split to perform:
        - 'simple': Simple time-based train-test split.
        - 'cv': Cross-validation with TimeSeriesSplit.
    - test_ratio: str or float, optional
        - For 'simple' split: Number of rows or fraction of the dataset for the test set.
        - For 'cv': Not applicable (determined by n_splits).
    - n_splits: int, optional (default=5)
        Number of splits for cross-validation (only applicable for 'cv').
    - gap: int, optional (default=0)
        Gap between train and test sets for cross-validation (only applicable for 'cv').
    - train_start: str or None, optional
        Start date for the training set (only for 'simple' split with datetime-based filtering).
    - train_end: str or None, optional
        End date for the training set (only for 'simple' split with datetime-based filtering).
    - verbose: int, optional (default=0)
        Verbosity level for logging.

    Returns:
    - splits: tuple or generator
        - For 'simple': A tuple `(train_df, test_df)`.
        - For 'cv': A generator yielding `(train_idx, test_idx)` for each split.
    """
    # Step 1: Validate the input DataFrame
    is_frame(df, df_only=True, raise_exception=True, objname="DataFrame 'df'")

    # Step 2: Validate and process the datetime column
    df, dt_col = ts_validator(
        df, dt_col=dt_col, 
        to_datetime='auto', 
        as_index=False, 
        error="raise",
        return_dt_col= True, 
        verbose=verbose
    )

    # Step 3: Handle 'simple' time-based split
    if split_type == "simple":
        if train_start or train_end:
            # Perform date-based filtering
            if verbose >= 1:
                print(f"Performing simple split with train_start={train_start}, train_end={train_end}.")
            train_df = df.loc[
                (df[dt_col] >= pd.to_datetime(train_start)) & (df[dt_col] <= pd.to_datetime(train_end))
            ] if train_start and train_end else df.loc[df[dt_col] <= pd.to_datetime(train_end)]

            test_df = df.loc[df[dt_col] > pd.to_datetime(train_end)] if train_end else pd.DataFrame()

        elif test_ratio is not None:
            # Perform row-based split
            test_ratio = validate_ratio(
                test_ratio, bounds=(0, 1), param_name="Test Ratio", exclude=0 ) 
            test_ratio = int(len(df) * test_ratio)
           
            split_idx = len(df) - test_ratio
            if verbose >= 1:
                print(f"Performing simple split: Train size={split_idx}, Test size={test_ratio}.")
            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]
        else:
            raise ValueError("`test_size` or `train_end` must be specified for a simple split.")

        return train_df, test_df

    # Step 4: Handle cross-validation split
    elif split_type == "cv":
        if verbose >= 1:
            print(f"Performing cross-validation split with n_splits={n_splits}, gap={gap}.")
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
        splits = tscv.split(df)

        # Log splits if verbose
        if verbose >= 2:
            for i, (train_idx, test_idx) in enumerate(splits):
                print(f"Split {i}: Train indices={train_idx}, Test indices={test_idx}")

        return splits

    else:
        raise ValueError(f"Invalid split_type: {split_type}. Choose 'simple' or 'cv'.")


def ts_outlier_detector(
    df,
    dt_col=None,  # Datetime column
    value_col=None,  # Target column for outlier detection
    method="zscore",  # Outlier detection method: 'zscore' or 'iqr'
    threshold=3,  # Threshold for Z-Score or IQR multiplier
    view=False,  # Whether to visualize the outliers
    fig_size=(10, 5),  # Size of the visualization plot
    show_grid=True,  # Whether to show gridlines in the plots
    drop=False,  # Whether to drop outliers from the DataFrame
    verbose=0  # Verbosity level
):
    """
    Detect outliers in a time series dataset using Z-Score or IQR method.

    Parameters:
    - df: pandas.DataFrame
        The input DataFrame containing the time series data.
    - dt_col: str, optional
        Column name representing the datetime-based series (e.g., 'time').
    - value_col: str, required
        Column name of the target variable (e.g., sales).
    - method: str, optional (default='zscore')
        The outlier detection method to use:
        - 'zscore': Use Z-Score to detect outliers.
        - 'iqr': Use Interquartile Range (IQR) to detect outliers.
    - threshold: int or float, optional (default=3)
        - For Z-Score: Z-Score threshold to classify outliers.
        - For IQR: Multiplier for the IQR to classify outliers.
    - view: bool, optional (default=True)
        Whether to visualize the original series with outliers highlighted.
    - fig_size: tuple, optional (default=(12, 6))
        Size of the visualization plot.
    - show_grid: bool, optional (default=True)
        Whether to show gridlines in the plots.
    - drop: bool, optional (default=False)
        Whether to drop the outliers from the DataFrame.
    - verbose: int, optional (default=0)
        Verbosity level.

    Returns:
    - result: pandas.DataFrame
        DataFrame containing the original data with an additional column 'is_outlier',
        or the DataFrame without outliers if `drop=True`.
    """
    # Step 1: Validate the input DataFrame
    is_frame(df, df_only=True, raise_exception=True,
             objname="DataFrame 'df'") 
    
    # Step 2: Validate and process the datetime column
    df, dt_col = ts_validator(
        df, dt_col=dt_col, 
        to_datetime='auto', 
        as_index=False, 
        error="raise",
        return_dt_col=True, 
        verbose=verbose
    )
    # Step 2: Validate and extract the target column
    target, _= validate_target_in(df, value_col)
    tname = target.name  # Get the name of the target variable
    
    if verbose >= 1:
        print(f"Target variable: {tname}")
        print(f"Datetime column: {dt_col}")
        print(f"Outlier detection method: {method}, Threshold: {threshold}")

    # Step 4: Detect outliers using the chosen method
    if method == "zscore":
        if verbose >= 1:
            print("Detecting outliers using Z-Score...")
        z_scores = zscore(target)
        outliers = np.abs(z_scores) > threshold

    elif method == "iqr":
        if verbose >= 1:
            print("Detecting outliers using IQR...")
        q1 = target.quantile(0.25)
        q3 = target.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (threshold * iqr)
        upper_bound = q3 + (threshold * iqr)
        outliers = (target < lower_bound) | (target > upper_bound)
    else:
        raise ValueError(f"Invalid method: {method}. Choose 'zscore' or 'iqr'.")

    # Add outlier information to the DataFrame
    df['is_outlier'] = outliers

    if verbose >= 1:
        num_outliers = outliers.sum()
        print(f"Number of outliers detected: {num_outliers}")

    # Step 5: Visualize the outliers (if enabled)
    if view:
        plt.figure(figsize=fig_size)
        plt.plot(df[dt_col], target, label="Original Data", color="blue", alpha=0.8)
        plt.scatter(
            df[dt_col][outliers], target[outliers], 
            color="red", label="Outliers", zorder=5
        )
        plt.title(f"Outlier Detection ({method.capitalize()} Method)",
                  fontsize=14, fontweight="bold")
        plt.xlabel("Time", fontsize=12)
        plt.ylabel(tname, fontsize=12)
        if show_grid:
            plt.grid(True, linestyle=":", alpha=0.7)
        else:
            plt.grid(False)
            
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Step 6: Drop outliers if requested
    if drop:
        df = df[~df['is_outlier']].drop(columns=['is_outlier'])
        if verbose >= 1:
            print(f"Outliers dropped. Remaining data points: {len(df)}")
    else:
        if verbose >= 1:
            print("Outliers retained in the DataFrame.")
    # Step 6: Return the result DataFrame
    return df



def create_lag_features(
    df,value_col,  # Target column (time series variable)
    dt_col=None,  # Datetime column
    lag_features=None,  # List of feature names to create lags for
    lags=[1, 3, 7],  # List of lag intervals to create
    dropna=True,  # Whether to drop rows with NaN values after creating lags
    include_original=True,  # Whether to include the original features in the output
    reset_index=True, 
    verbose=0  # Verbosity level
):
    """
    Create lag features (time-delayed versions) for time series data to 
    capture temporal dependencies.

    Parameters:
    - df: pandas.DataFrame
        The input DataFrame containing the time series data.
    - dt_col: str, optional
        Column name representing the datetime-based series (e.g., 'time').
        # If None expect it is index columns 
    - value_col: str, required
        Column name of the target variable for which lag features will be created.
    - lag_features: list, optional
        List of additional feature names (other than the target) to create lag features for.
        If None, only the target column (`value_col`) is used.
    - lags: list of int, optional (default=[1, 3, 7])
        List of lag intervals to create (e.g., [1, 3, 7] will create lag-1, lag-3, lag-7).
    - dropna: bool, optional (default=True)
        Whether to drop rows with NaN values introduced by lagging.
    - include_original: bool, optional (default=True)
        Whether to include the original features in the output DataFrame.
    - verbose: int, optional (default=0)
        Verbosity level for logging.

    Returns:
    - lagged_df: pandas.DataFrame
        DataFrame containing the lag features and optionally the original features.
    """
    # Step 1: Validate the input DataFrame
    is_frame(df, df_only=True, raise_exception=True, objname="DataFrame 'df'")
    
    # Step 2: Validate and extract the target column
    df, dt_col = ts_validator(
        df, dt_col=dt_col, 
        to_datetime='auto', 
        as_index=True, 
        error="raise",
        return_dt_col=True, 
        verbose=verbose
    )
   # Step 3: Ensure datetime column is valid
    target, _= validate_target_in(df, value_col)
    tname = target.name  # Get the name of the target variable
    if verbose >= 1:
        print(f"Target variable: {tname}")
        print(f"Datetime column: {dt_col}")
        print(f"Lag intervals: {lags}")


    lag = columns_manager(lags)  # if none, return None 
    if lag is None: 
        lag =[1]  # then take the default lag applied to value.  
    
    # Step 4: Determine the columns to create lag features for
    # if none, return empty list 
    lag_features = columns_manager(lag_features, empty_as_none=False ) 
    if value_col not in lag_features:
        lag_features.append(value_col)
    
    exist_features(df, features=lag_features, name="Lag features")

    # Step 5: Create lag features
    lagged_df = pd.DataFrame(index=df.index)
    if dt_col in df.columns:
        lagged_df[dt_col] = df[dt_col]  # Retain datetime column

    for feature in lag_features:
        if verbose >= 1:
            print(f"Creating lag features for: {feature}")
        for lag in lags:
            lagged_df[f"{feature}_lag_{lag}"] = df[feature].shift(lag)

    # Step 6: Optionally include the original features
    if include_original:
        lagged_df = pd.concat([lagged_df, df], axis=1)
        lagged_df = lagged_df[list(set(lagged_df))] # to avoid repeating 
        
    # Step 7: Drop NaN values (if requested)
    if dropna:
        if verbose >= 1:
            num_rows_before = len(lagged_df)
        lagged_df = lagged_df.dropna()
        if verbose >= 1:
            num_rows_after = len(lagged_df)
            print(f"Rows dropped due to NaN values: {num_rows_before - num_rows_after}")
            
    if reset_index: 
        if dt_col == lagged_df.index.name: 
            lagged_df.reset_index(inplace =True)
    # Step 8: Return the lagged DataFrame
    return lagged_df


def select_and_reduce_features(
    df,
    target_col=None,  # Name of the target column
    exclude_cols=None,  # Columns to exclude from feature selection
    method="correlation",  # Method to use: 'correlation' or 'pca'
    corr_threshold=0.9,  # Correlation threshold for CFS
    n_components=None,  # Number of components for PCA
    scale_data=True,  # Whether to scale data for PCA
    return_pca=False, 
    verbose=0,  # Verbosity level
):
    """
    Perform feature selection and dimensionality reduction on a dataset.

    Parameters:
    - df: pandas.DataFrame
        The input DataFrame containing the dataset.
    - target_col: str, optional
        The target column (to exclude from feature selection and reduction).
    - exclude_cols: list, optional
        List of columns to exclude from feature selection and reduction.
    - method: str, optional (default='correlation')
        The method to use:
        - 'correlation': Correlation-based feature selection.
        - 'pca': Principal Component Analysis for dimensionality reduction.
    - corr_threshold: float, optional (default=0.9)
        Threshold for correlation-based feature selection (CFS).
    - n_components: int or float, optional
        Number of components for PCA. If float, it represents the proportion of variance to retain.
    - scale_data: bool, optional (default=True)
        Whether to scale the data before applying PCA.
    - verbose: int, optional (default=0)
        Verbosity level for logging.

    Returns:
    - transformed_df: pandas.DataFrame
        DataFrame with selected or reduced features.
    - pca_model: sklearn.decomposition.PCA or None
        The PCA model (if method='pca'), otherwise None.
    """

    # Step 1: Validate the input DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("`df` must be a pandas DataFrame.")
    
    # return empty list if None or return list if str
    target_col = columns_manager(target_col, empty_as_none=False)
    
    exclude_cols = columns_manager(exclude_cols, empty_as_none=False )
    valid_cols = is_in_if(df.columns, items=exclude_cols, return_diff=True )

    features = select_features (df, features = valid_cols )
    # Step 2: Separate features and target
    target =None 
    if target_col is not None: 
        target, features  = validate_target_in(features, target_col)
    
    pca=None 
    if verbose >= 1:
        print(f"Number of features before selection: {features.shape[1]}")
        print(f"Excluded columns: {exclude_cols}")

    # Step 3: Apply Correlation-based Feature Selection (CFS)
    if method == "correlation":
        if verbose >= 1:
            print("Performing correlation-based feature selection...")
        
        corr_matrix = features.corr().abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), 
                                                   k=1).astype(bool))
        to_drop = [column for column in upper_triangle.columns if any(
            upper_triangle[column] > corr_threshold)]
        
        if verbose >= 2:
            print(f"Correlation matrix:\n{corr_matrix}")
            print(f"Highly correlated features to drop (threshold={corr_threshold}): {to_drop}")
        
        reduced_features = features.drop(columns=to_drop, errors='ignore')

        transformed_df = pd.concat(
            [reduced_features, target], axis=1) if target_col else reduced_features
        if return_pca: 
            warnings.warn ("PCA is not selected as method for dimensionality"
                           " reduction. Return only the transfromed matrix with "
                           " correlation based instead."
                           )
        # columns_no_duplicated = list(set(transformed_df.columns)) 
        
        return transformed_df
    
    # Step 4: Apply Principal Component Analysis (PCA)
    elif method == "pca":
        if verbose >= 1:
            print("Performing Principal Component Analysis (PCA)...")
            if scale_data:
                print("Standardizing data before PCA.")

        # Standardize data if required
        if scale_data:
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
        else:
            scaled_features = features.values

        # Apply PCA
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(scaled_features)

        # Create a DataFrame for PCA components
        if isinstance(n_components, int):
            pca_columns = [f"PC{i+1}" for i in range(n_components)]
        else:  # if n_components is a float (variance proportion)
            pca_columns = [f"PC{i+1}" for i in range(pca.n_components_)]
        pca_df = pd.DataFrame(principal_components, columns=pca_columns, index=df.index)

        if verbose >= 1:
            print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
            print(f"Number of components selected: {pca.n_components_}")

        # Combine PCA components with the target (if applicable)
        transformed_df = pd.concat([pca_df, target], axis=1) if target_col else pca_df
        
        # columns_no_duplicated = list(set (transformed_df.columns)) 
        # transformed_df = transformed_df [columns_no_duplicated]
        if return_pca: 
             return  transformed_df, pca
        
        return transformed_df 

