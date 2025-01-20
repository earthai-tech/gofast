# -*- coding: utf-8 -*-
""" Times-series utilities. """

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
import seaborn as sns 
from sklearn.preprocessing import StandardScaler, MinMaxScaler 

try: 
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.graphics.tsaplots import plot_acf
    from statsmodels.tsa.seasonal import STL, seasonal_decompose
except: 
    pass 

from gofast.core.array_manager import smart_ts_detector
from gofast.utils.validator import is_time_series, is_frame
from gofast.core.checks import exist_features 
from gofast.core.io import to_frame_if  
from gofast.utils.base_utils import validate_target_in 


__all__= [ 
    'decompose_ts','infer_decomposition_method',
    'prepare_time_series','trend_analysis','trend_ops',
    'ts_engineering','ts_validator','visual_inspection' 
 ]

def ts_validator(
    df, 
    dt_col=None, 
    to_datetime=None, 
    as_index="auto", 
    error ='raise', 
    verbose=0, 
    return_dt_col=False,
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

    return (df, dt_col) if return_dt_col else df 


def trend_analysis(
    df, 
    target,     
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
    - target: Column name or series of the target variable to forecast.
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
    target,_= validate_target_in(df, target, error=error, verbose=verbose) 

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
    target,
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
    - target: Column name or series of the target variable to forecast.
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
        df, target=target, dt_col=dt_col, 
        check_stationarity=check_stationarity, 
        trend_type=trend_type, view=False, 
        strategy=strategy, 
        **kw
    )
    # Step 2: Validate and process the target using validate_target_in
    target, _ = validate_target_in(df, target, error=error, verbose=verbose) 
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
    df, target, 
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
    ts, _ = validate_target_in(df, target, error='raise', verbose=0) 
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
    df, target, dt_col=None,  
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
    ts, _ = validate_target_in(df, target, error='raise', verbose=0) 
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
    df, target, 
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
    ts, _ = validate_target_in(df, target, error='raise', verbose=0) 
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


def prepare_time_series(
    df, time_col=None, 
    set_index=True, 
    error='raise', 
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
        return df  # No need to change anything
    
    # 2. If the index is not datetime, check if a time column is specified
    if time_col is not None:
        if time_col not in df.columns:
            raise ValueError(f"Column '{time_col}' not found in the DataFrame.")
        
        # Set the specified column as the index and convert to datetime
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        
        if df[time_col].isnull().any():
            raise ValueError(f"Column '{time_col}' contains invalid date"
                             " formats that could not be converted.")
        
        if set_index:
            df.set_index(time_col, inplace=True)
        print(f"Column '{time_col}' has been set as the"
              " index and converted to datetime.")
        return df

    # 3. If no time_col is specified, we need to handle the
    # case based on the error parameter
    if error == 'raise':
        raise ValueError("The index is not a datetime object,"
                         " and no 'time_col' was specified.")
    elif error == 'warn':
        print("Warning: The index is not a datetime object, and no"
              " 'time_col' was specified. Default behavior will apply.")
    elif error == 'ignore':
        print("Index is not a datetime, but no action will be taken.")
        return df
    
    # 4. If no valid datetime column found and the user didn’t specify a"
    # time column, return the original df
    print("No datetime index or time column found. Returning the DataFrame as is.")
    return df

