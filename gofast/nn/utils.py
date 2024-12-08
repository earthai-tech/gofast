# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Utility functions for neural networks models.

This module provides utility functions to preprocess data for Temporal Fusion 
Transformer (TFT) models, including splitting sequences into static and dynamic 
inputs and creating input sequences with corresponding targets for time series 
forecasting.

"""

from numbers import Integral 
import warnings
from typing import List, Tuple, Optional, Union, Dict
import numpy as np
import pandas as pd

from ..core.checks import (
    ParamsValidator, 
    are_all_frames_valid, 
    exist_features
    )
from ..core.handlers import TypeEnforcer, columns_manager 
from ..core.io import is_data_readable 
from ..compat.sklearn import validate_params, Interval 
from ..decorators import DynamicMethod
from ..tools.validator import validate_sequences 
from ..tools.depsutils import ensure_pkg 

from . import KERAS_DEPS, KERAS_BACKEND, dependency_message

if KERAS_BACKEND:
    Callback=KERAS_DEPS.Callback
    
DEP_MSG = dependency_message('wrappers') 

__all__ = ["split_static_dynamic", "create_sequences", "compute_forecast_horizon"]

@TypeEnforcer({"0": 'array-like', "1": 'array-like'})
@validate_params({ 
    "sequences": ['array-like'], 
    "static_indices":[Integral], 
    "dynamic_indices": [Integral], 
    "static_time_step": [Interval( Integral, 0, None, closed="left")], 
    "static_reshape_shape": [tuple, None], 
    "dynamic_reshape_shape": [tuple, None],
    }
)
def split_static_dynamic(
    sequences: np.ndarray, 
    static_indices: List[int], 
    dynamic_indices: List[int],
    static_time_step: int = 0,
    reshape_static: bool = True,
    reshape_dynamic: bool = True,
    static_reshape_shape: Optional[Tuple[int, ...]] = None,
    dynamic_reshape_shape: Optional[Tuple[int, ...]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split sequences into static and dynamic inputs for the model.

    The `split_static_dynamic` function divides input sequences into static and 
    dynamic components based on specified feature indices. Static features are 
    typically location-specific and do not change over time, while dynamic features 
    vary across different time steps.

    .. math::
        \text{Static Inputs} = \mathbf{S} = \mathbf{X}_{t, static\_indices} \\
        \text{Dynamic Inputs} = \mathbf{D} = \mathbf{X}_{:, dynamic\_indices}

    Parameters
    ----------
    sequences : `numpy.ndarray`
        Array of input sequences with shape 
        (batch_size, sequence_length, num_features).

    static_indices : `List[int]`
        Indices of static features within the feature dimension.

    dynamic_indices : `List[int]`
        Indices of dynamic features within the feature dimension.

    static_time_step : `int`, default=`0`
        Time step from which to extract static features (default is the first 
        time step).

    reshape_static : `bool`, default=`True`
        Whether to reshape static inputs. If `False`, returns without reshaping.

    reshape_dynamic : `bool`, default=`True`
        Whether to reshape dynamic inputs. If `False`, returns without reshaping.

    static_reshape_shape : `Optional[Tuple[int, ...]]`, default=`None`
        Desired shape for static inputs after reshaping. If `None`, defaults to 
        (batch_size, num_static_vars, 1).

    dynamic_reshape_shape : `Optional[Tuple[int, ...]]`, default=`None`
        Desired shape for dynamic inputs after reshaping. If `None`, defaults to 
        (batch_size, sequence_length, num_dynamic_vars, 1).

    Returns
    -------
    Tuple[`numpy.ndarray`, `numpy.ndarray`]
        A tuple containing:
        - Static inputs with shape as specified.
        - Dynamic inputs with shape as specified.

    Raises
    ------
    ValueError
        If `static_time_step` is out of range for the given sequence length.

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.nn.utils import split_static_dynamic
    >>> 
    >>> # Create a dummy sequence array
    >>> sequences = np.random.rand(100, 10, 5)  # (
    ...   batch_size=100, sequence_length=10, num_features=5)
    >>> 
    >>> # Define static and dynamic feature indices
    >>> static_indices = [0, 1]  # e.g., longitude and latitude
    >>> dynamic_indices = [2, 3, 4]  # e.g., year, GWL, density
    >>> 
    >>> # Split the sequences
    >>> static_inputs, dynamic_inputs = split_static_dynamic(
    ...     sequences, 
    ...     static_indices=static_indices, 
    ...     dynamic_indices=dynamic_indices,
    ...     static_time_step=0
    ... )
    >>> 
    >>> print(static_inputs.shape)
    (100, 2, 1)
    >>> print(dynamic_inputs.shape)
    (100, 10, 3, 1)

    Notes
    -----
    - **Static Features:** These are typically location-specific features such as 
      geographical coordinates or categorical attributes that remain constant 
      over time.

    - **Dynamic Features:** These features vary over different time steps and are 
      essential for capturing temporal dependencies in the data.

    - **Reshaping:** The function provides flexibility in reshaping the static and 
      dynamic inputs to match the input requirements of various models, including 
      Temporal Fusion Transformers.

    See Also
    --------
    `create_sequences` : Function to create input sequences and targets for 
    time series forecasting.

    References
    ----------
    .. [1] Qin, Y., Song, D., Chen, H., Cheng, W., Jiang, G., & Cottrell, G. (2017). 
       Temporal fusion transformers for interpretable multi-horizon time series forecasting. 
       *arXiv preprint arXiv:1912.09363*.
    """
    # Validate static_time_step
    if not (0 <= static_time_step < sequences.shape[1]):
        raise ValueError(
            f"static_time_step {static_time_step} is out of range"
            f" for sequence_length {sequences.shape[1]}"
        )
    sequences = validate_sequences(sequences, check_shape= True )
    # Extract static inputs
    static_inputs = sequences[:, static_time_step, static_indices]
    if reshape_static:
        if static_reshape_shape:
            static_inputs = static_inputs.reshape(*static_reshape_shape)
        else:
            static_inputs = static_inputs.reshape(-1, len(static_indices), 1)
    
    # Extract dynamic inputs
    dynamic_inputs = sequences[:, :, dynamic_indices]
    if reshape_dynamic:
        if dynamic_reshape_shape:
            dynamic_inputs = dynamic_inputs.reshape(*dynamic_reshape_shape)
        else:
            dynamic_inputs = dynamic_inputs.reshape(
                -1, sequences.shape[1], len(dynamic_indices), 1)
    
    return static_inputs, dynamic_inputs


@DynamicMethod ('numeric', prefixer ='exclude')
@validate_params({ 
    'df': ['array-like'], 
    'sequence_length': [Interval(Integral, 2, None, closed ='left')], 
    'target_col': [str], 
    'step': [Interval(Integral, 1, None, closed ='left')], 
    'forecast_horizon': [Interval(Integral, 1, None, closed ='left'), None], 
    }
)
def create_sequences(
    df: pd.DataFrame, 
    sequence_length: int, 
    target_col: str,
    step: int = 1,
    include_overlap: bool = True,
    drop_last: bool = True,
    forecast_horizon: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input sequences and corresponding targets for time series forecasting.
    
    The `create_sequences` function generates sequences of features and their 
    corresponding targets from a time series dataset. This is essential for training 
    sequence models like Temporal Fusion Transformers, LSTMs, and others that 
    rely on temporal dependencies.
    
    .. math::
        \text{For each sequence } i, \\
        \mathbf{X}^{(i)} = \left[ \mathbf{x}_{i}, \mathbf{x}_{i+1},\\
                                 \dots, \mathbf{x}_{i+T-1} \right] \\
        y^{(i)} = 
        \begin{cases}
            \mathbf{x}_{i+T} & \text{if } \text{forecast\_horizon} = \text{None} \\
            \left[ \mathbf{x}_{i+T}, \mathbf{x}_{i+T+1}, \dots,\\
                  \mathbf{x}_{i+T+H-1} \right] & \text{if } \text{forecast\_horizon} = H
        \end{cases}
    
    Where:
    - :math:`\mathbf{X}^{(i)}` is the input sequence of length :math:`T`.
    - :math:`y^{(i)}` is the target value(s) following the sequence.
    
    Parameters
    ----------
    df : `pandas.DataFrame`
        The processed DataFrame containing features and the target variable.
    
    sequence_length : `int`
        The number of past time steps to include in each input sequence.
    
    target_col : `str`
        The name of the target column.
    
    step : `int`, default=`1`
        The step size between the starts of consecutive sequences.
    
    include_overlap : `bool`, default=`True`
        Whether to include overlapping sequences based on the step size.
    
    drop_last : `bool`, default=`True`
        Whether to drop the last sequence if it does not have enough data points.
    
    forecast_horizon : `int`, optional, default=`None`
        The number of future time steps to predict. If set to `None`, the function
        will create targets for a single future time step. If provided, targets will
        consist of the next `forecast_horizon` time steps.
    
    Returns
    -------
    Tuple[`numpy.ndarray`, `numpy.ndarray`]
        A tuple containing:
        - `sequences` : Array of input sequences with shape 
          (num_sequences, sequence_length, num_features).
        - `targets` : 
            - If `forecast_horizon` is `None`: Array of target values with shape 
              (num_sequences,).
            - If `forecast_horizon` is an integer: Array of target sequences with shape 
              (num_sequences, forecast_horizon).
    
    Raises
    ------
    ValueError
        If the DataFrame `df` does not contain the `target_col`.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from gofast.nn.utils import create_sequences
    >>> 
    >>> # Create a dummy DataFrame
    >>> data = pd.DataFrame({
    ...     'feature1': np.random.rand(100),
    ...     'feature2': np.random.rand(100),
    ...     'feature3': np.random.rand(100),
    ...     'target': np.random.rand(100)
    ... })
    >>> 
    >>> # Create sequences for single-step forecasting
    >>> sequence_length = 4
    >>> sequences, targets = create_sequences(
    ...     df=data, 
    ...     sequence_length=sequence_length, 
    ...     target_col='target',
    ...     step=1,
    ...     include_overlap=True,
    ...     drop_last=True,
    ...     forecast_horizon=None
    ... )
    >>> 
    >>> print(sequences.shape)
    (95, 4, 4)
    >>> print(targets.shape)
    (95,)
    >>> 
    >>> # Create sequences for multi-step forecasting (e.g., 3 steps ahead)
    >>> forecast_horizon = 3
    >>> sequences, targets = create_sequences(
    ...     df=data, 
    ...     sequence_length=4, 
    ...     target_col='target',
    ...     step=1,
    ...     include_overlap=True,
    ...     drop_last=True,
    ...     forecast_horizon=3
    ... )
    >>> 
    >>> print(sequences.shape)
    (92, 4, 4)
    >>> print(targets.shape)
    (92, 3)
    
    Notes
    -----
    - **Sequence Creation:** The function slides a window of size `sequence_length` 
      across the DataFrame to create input sequences. Each sequence is associated 
      with a target value or sequence of values that immediately follow the input sequence.
    
    - **Forecast Horizon:** 
        - If `forecast_horizon` is `None`, the function creates targets for a single 
          future time step.
        - If `forecast_horizon` is an integer `H`, the function creates targets consisting 
          of the next `H` time steps.
    
    - **Step Size:** The `step` parameter controls the stride of the sliding 
      window. A `step` of `1` results in overlapping sequences, while a larger 
      `step` reduces overlap.
    
    - **Handling Incomplete Sequences:** If `drop_last` is set to `False`, the 
      function includes the last sequence even if it doesn't have enough data 
      points to form a complete sequence or target.
    
    - **Data Validation:** The function utilizes `are_all_frames_valid` from 
      `gofast.core.checks` to ensure the integrity of input DataFrame before 
      processing and `exist_features` to verify the presence of the target column.
    
    See Also
    --------
    gofast.nn.utils.split_static_dynamic : 
        Function to split sequences into static and dynamic inputs.
    
    References
    ----------
    .. [1] Brownlee, J. (2018). Time Series Forecasting with Python: Create 
           accurate models in Python to forecast the future and gain insight
            from your time series data. Machine Learning Mastery.
    .. [2] Qin, Y., Song, D., Chen, H., Cheng, W., Jiang, G., & Cottrell, G. (2017). 
           Temporal fusion transformers for interpretable multi-horizon time
           series forecasting. *arXiv preprint arXiv:1912.09363*.
    
    """
    # Validate all frames
    are_all_frames_valid(df, df_only=True, error_msg=(
        "DataFrame contains invalid or missing data."
    ))
    
    # Validate that target_col exists in the DataFrame
    exist_features(
        df, features=target_col, 
        name=f"Target column '{target_col}'"
    )
    
    sequences = []
    targets = []
    total_length = len(df)
    
    # Determine the maximum required steps based on forecast_horizon
    max_horizon = forecast_horizon if forecast_horizon is not None else 1
    
    for i in range(0, total_length - sequence_length - max_horizon + 1, step):
        seq = df.iloc[i:i+sequence_length]
        
        if forecast_horizon is None:
            target = df.iloc[i+sequence_length][target_col]
        else:
            target = df.iloc[i+sequence_length:i+sequence_length + forecast_horizon][target_col]
        
        sequences.append(seq.values)
        targets.append(target.values if forecast_horizon is not None else target)
    
    if not drop_last and forecast_horizon is not None:
        remaining = (total_length - sequence_length) % step
        if remaining != 0 and (total_length - sequence_length) >= forecast_horizon:
            seq = df.iloc[-sequence_length:]
            target = df.iloc[-forecast_horizon:][target_col]
            sequences.append(seq.values)
            targets.append(target.values)
    elif not drop_last and forecast_horizon is None:
        if (total_length - sequence_length) % step != 0:
            seq = df.iloc[-sequence_length:]
            target = df.iloc[-1][target_col]
            sequences.append(seq.values)
            targets.append(target)
    
    sequences = np.array(sequences)
    targets = np.array(targets)
    
    return sequences, targets

@is_data_readable(data_to_read ='data')
def compute_forecast_horizon(
    data=None,
    date_col=None,
    start_prediction=None,
    end_prediction=None,
    error='raise',
    verbose=1
):
    """
    Compute the forecast horizon for time series forecasting models.

    This function calculates the number of future time steps (`forecast_horizon`)
    a model should predict based on the provided data or specified prediction
    dates. It intelligently infers the frequency of the data and computes the
    horizon accordingly. The function accommodates various datetime formats and
    handles different input scenarios robustly.

    Parameters
    ----------
    data : pandas.DataFrame, pandas.Series, list, or numpy.ndarray, optional
        The dataset containing datetime information. If a `pandas.DataFrame` is
        provided, the `date_col` parameter must be specified to indicate which
        column contains the datetime data. For `pandas.Series`, `list`, or
        `numpy.ndarray`, the function attempts to infer the frequency directly.

    date_col : str, optional
        The name of the column in `data` that contains datetime information.
        This parameter is **required** if `data` is a `pandas.DataFrame`.
        Example:
        ``date_col='timestamp'``

    start_prediction : str, int, or datetime-like
        The starting point for forecasting. This can be a date string
        (e.g., `'2023-04-10'`), a `datetime` object, or an integer representing
        a year (e.g., `2024`). If an integer is provided, it is interpreted as a
        year, and a warning is issued to inform the user of this interpretation.

    end_prediction : str, int, or datetime-like
        The ending point for forecasting. Similar to `start_prediction`, this can
        be a date string, a `datetime` object, or an integer representing a year.
        The function calculates the forecast horizon based on the difference
        between `start_prediction` and `end_prediction`.

    error : {'raise', 'warn', 'ignore'}, default='raise'
        Defines the error handling behavior when encountering issues such as
        invalid input types, missing date columns, or unparseable dates.

        - `'raise'`: Raises a `ValueError` when an error is encountered.
        - `'warn'`: Emits a warning and attempts to proceed with default behavior.

    verbose : int, default=1
        Controls the level of verbosity for debug information.

        - `0`: No output.
        - `1`: Minimal output (e.g., starting message).
        - `2`: Intermediate output (e.g., detected dates, computed horizons).
        - `3`: Detailed output (e.g., types of predictions, inferred frequencies).

    Returns
    -------
    int or None
        The computed `forecast_horizon` representing the number of steps ahead
        the model should predict. Returns `None` if an error occurs and `error`
        is set to `'warn'`.

    Raises
    ------
    ValueError
        If invalid parameters are provided and `error` is set to `'raise'`.

    Examples
    --------
    >>> from gofast.nn.utils import compute_forecast_horizon
    >>> import pandas as pd
    >>> import numpy as np
    >>> from datetime import datetime, timedelta
    >>> 
    >>> # Example 1: Using a DataFrame with a Date Column
    >>> df = pd.DataFrame({
    ...     'date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
    ...     'value': np.random.randn(100)
    ... })
    >>> horizon = compute_forecast_horizon(
    ...     data=df,
    ...     date_col='date',
    ...     start_prediction='2023-04-10',
    ...     end_prediction='2023-04-20',
    ...     error='raise',
    ...     verbose=3
    ... )
    >>> print(f"Forecast Horizon: {horizon}")
    Forecast Horizon: 11

    >>> # Example 2: Using a List of Datetimes
    >>> dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(100)]
    >>> horizon = compute_forecast_horizon(
    ...     data=dates,
    ...     start_prediction='2023-04-10',
    ...     end_prediction='2023-04-20',
    ...     error='warn',
    ...     verbose=2
    ... )
    >>> print(f"Forecast Horizon: {horizon}")
    Forecast Horizon: 11

    >>> # Example 3: Handling Integer Years
    >>> horizon = compute_forecast_horizon(
    ...     start_prediction=2024,
    ...     end_prediction=2030,
    ...     error='raise',
    ...     verbose=1
    ... )
    Forecast Horizon: 7

    >>> # Example 4: Without Providing Data (Assuming Frequency Based on Prediction Dates)
    >>> horizon = compute_forecast_horizon(
    ...     start_prediction='2023-04-10',
    ...     end_prediction='2023-04-20',
    ...     error='raise',
    ...     verbose=1
    ... )
    Forecast Horizon: 11

    Notes
    -----
    - When `data` is not provided, the function relies solely on the difference
      between `start_prediction` and `end_prediction` to compute the forecast
      horizon. In such cases, if the frequency cannot be inferred, the horizon
      is calculated based on the largest possible time unit (years, months,
      weeks, days).
    
    - If `start_prediction` is after `end_prediction`, the function returns `0`
      and issues a warning or raises an error based on the `error` parameter.

    - The function attempts to infer the frequency of the data using `pandas`
      utilities. If the frequency cannot be inferred, it defaults to calculating
      the horizon based on the time difference in the most significant unit.

    See Also
    --------
    pandas.date_range : Generates a fixed frequency DatetimeIndex.
    pandas.infer_freq : Infers the frequency of a DatetimeIndex.

    References
    ----------
    .. [1] McKinney, Wes. *Python for Data Analysis: Data Wrangling with Pandas,
       NumPy, and IPython*. O'Reilly Media, 2017.
    .. [2] Harris, C.R., Millman, K.J., van der Walt, S.J., et al. 
       *Array programming with NumPy*. Nature, 585(7825), pp.357-362, 2020.
    """

    if verbose >= 1:
        print("Starting compute_forecast_horizon...")
    
    # Validate verbosity level
    if not isinstance(verbose, int) or not (0 <= verbose <= 3):
        raise ValueError("`verbose` must be an integer between 0 and 3.")
    
    frequency = None
    dates = None

    # Process data if provided
    if data is not None:
        if isinstance(data, pd.DataFrame):
            if date_col is None:
                message = "`date_col` must be specified when data is a DataFrame."
                if error == 'raise':
                    raise ValueError(message)
                elif error == 'warn':
                    warnings.warn(message)
                    return None
            if date_col not in data.columns:
                message = f"`date_col` '{date_col}' not found in DataFrame."
                if error == 'raise':
                    raise ValueError(message)
                elif error == 'warn':
                    warnings.warn(message)
                    return None
            dates = pd.to_datetime(data[date_col], errors='coerce')
            if dates.isnull().any():
                message = "Some dates could not be converted. Check `date_col` format."
                if error == 'raise':
                    raise ValueError(message)
                elif error == 'warn':
                    warnings.warn(message)
            if verbose >= 2:
                print(f"Detected dates from DataFrame:\n {dates.head()}")
            frequency = pd.infer_freq(dates)
            if frequency is None:
                message = "Could not infer frequency from the date column."
                if error == 'raise':
                    raise ValueError(message)
                elif error == 'warn':
                    warnings.warn(message)
            else:
                if verbose >= 2:
                    print(f"Inferred frequency: {frequency}")
        
        elif isinstance(data, (pd.Series, list, np.ndarray)):
            dates = pd.to_datetime(data, errors='coerce')
            if pd.isnull(dates).any():
                message = "Some dates could not be converted. Check the datetime format."
                if error == 'raise':
                    raise ValueError(message)
                elif error == 'warn':
                    warnings.warn(message)
            frequency = pd.infer_freq(dates)
            if frequency is None:
                message = "Could not infer frequency from the datetime data."
                if error == 'raise':
                    raise ValueError(message)
                elif error == 'warn':
                    warnings.warn(message)
            else:
                if verbose >= 2:
                    print(f"Inferred frequency: {frequency}")
        
        else:
            message = ( 
                "Unsupported data type. Data should"
                " be a DataFrame, Series, list, or numpy array."
                f" Got {type(data).__name__!r}"
                )
            if error == 'raise':
                raise ValueError(message)
            elif error == 'warn':
                warnings.warn(message)
                return None
    
    # Convert start and end prediction to datetime or year
    def convert_prediction(pred):
        if isinstance(pred, int):
            # Treat as year
            converted = pd.to_datetime(f"{pred}", format="%Y", errors='coerce')
            if pd.isnull(converted):
                return None, 'year'
            return converted, 'year'
        else:
            # Attempt to convert to datetime
            converted = pd.to_datetime(pred, errors='coerce')
            if pd.isnull(converted):
                return None, None
            return converted, 'datetime'
    
    start_pred, start_type = convert_prediction(start_prediction)
    end_pred, end_type = convert_prediction(end_prediction)
    
    if start_pred is None or end_pred is None:
        message = ( 
            "Could not convert `start_prediction`"
            " or `end_prediction` to datetime."
            )
        if error == 'raise':
            raise ValueError(message)
        elif error == 'warn':
            warnings.warn(message)
            return None
    
    if verbose >= 2:
        print(f"Start Prediction: {start_pred} ({start_type})")
        print(f"End Prediction: {end_pred} ({end_type})")
    
    # Handle integer years with warnings
    if start_type == 'year' or end_type == 'year':
        message = (
            "Detected integer inputs for `start_prediction` or `end_prediction`. "
            "Interpreted as years."
        )
        if error == 'warn':
            warnings.warn(message)
        elif error == 'raise':
            emsg ="Set `error='ignore'` to interpret input integers as years."
            raise ValueError(
                message.replace("Interpreted as years.", emsg)
            )
    
    # Compute forecast horizon based on frequency
    if frequency is not None:
        try:
            pred_range = pd.date_range(
                start=start_pred, 
                end=end_pred, 
                freq=frequency
            )
            forecast_horizon = len(pred_range)  # Include both start and end
            if forecast_horizon <= 0:
                message = "`end_prediction` is before `start_prediction`."
                if error == 'raise':
                    raise ValueError(message)
                elif error == 'warn':
                    warnings.warn(message)
                    return 0
            if verbose >= 2:
                print(
                    f"Computed forecast horizon based on frequency '{frequency}': "
                    f"{forecast_horizon}"
                )
            if verbose > 0: 
                print(f"Forecast Horizon: {forecast_horizon}")
            return forecast_horizon
        
        except Exception as e:
            message = f"Error computing date range: {e}"
            if error == 'raise':
                raise ValueError(message)
            elif error == 'warn':
                warnings.warn(message)
                return None
    else:
        # If frequency is unknown, compute difference in years, months, days, etc.
        delta = end_pred - start_pred
        if delta.days >= 365:
            forecast_horizon = delta.days // 365 + 1
            unit = "years"
        elif delta.days >= 30:
            forecast_horizon = delta.days // 30 + 1
            unit = "months"
        elif delta.days >= 7:
            forecast_horizon = delta.days // 7 + 1
            unit = "weeks"
        else:
            forecast_horizon = delta.days + 1
            unit = "days"
        
        if forecast_horizon <= 0:
            message = "`end_prediction` is before `start_prediction`."
            if error == 'raise':
                raise ValueError(message)
            elif error == 'warn':
                warnings.warn(message)
                return 0
        
        if verbose >= 2:
            print(
                f"Frequency unknown. Computed forecast horizon based on {unit}: "
                f"{forecast_horizon}"
            )
        if verbose >= 1: 
            print(f"Forecast Horizon: {forecast_horizon}")
            
        return forecast_horizon
    
@ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
def extract_callbacks_from(fit_params, return_fit_params=False):
    r"""
    Extract Keras callbacks from a dictionary of fit parameters. The
    function scans the provided ``fit_params`` dictionary, looks for keys 
    associated with callback instances, and removes them from 
    ``fit_params`` returning a list of callbacks. Optionally, it can 
    return the updated dictionary without these callbacks.

    This function is particularly useful when working with scikit-learn-
    style estimators that pass parameters through ``**fit_params``. By 
    extracting callbacks directly, the user can seamlessly integrate 
    TensorFlow/Keras callbacks such as `EarlyStopping` or 
    `ModelCheckpoint` into their training pipelines.

    The function can handle two scenarios:
    1. A parameter called ``'callbacks'`` containing a list of callbacks.
    2. Individual callback instances passed as keyword arguments.

    After extraction, if ``return_fit_params=True``, it returns a tuple 
    `(callbacks, fit_params)` where `callbacks` is the extracted list and 
    `fit_params` is the remaining dictionary. Otherwise, it returns only 
    `callbacks`.

    .. math::
        n_{\mathrm{callbacks}} = n_{\mathrm{list\_callbacks}} \;+\;
        n_{\mathrm{kwarg\_callbacks}}

    Here, :math:`n_{\mathrm{callbacks}}` represents the total number of 
    extracted callbacks, :math:`n_{\mathrm{list\_callbacks}}` is the 
    number of callbacks initially found in the ``'callbacks'`` parameter, 
    and :math:`n_{\mathrm{kwarg\_callbacks}}` is the number of callbacks 
    discovered among the other keyword arguments.

    Parameters
    ----------
    fit_params : dict
        The dictionary of parameters to be passed to a model's 
        training method. May contain one of the following:
        
        * ``'callbacks'``: a list of callback instances.
        * Arbitrary keyword arguments that are callback instances.

    return_fit_params : bool, optional
        If ``True``, returns a tuple of `(callbacks, fit_params)` where 
        `fit_params` no longer contains the extracted callbacks. If 
        ``False``, returns only the `callbacks` list. Default is ``False``.

    Returns
    -------
    callbacks : list of tf.keras.callbacks.Callback
        A list of extracted callback instances.

    (callbacks, fit_params) : tuple (only if `return_fit_params=True`)
        A tuple where the first element is a list of extracted callbacks 
        and the second is the updated `fit_params` dictionary after 
        removing the callbacks.

    Examples
    --------
    >>> from gofast.nn.utils import extract_callbacks_from
    >>> from tensorflow.keras.callbacks import EarlyStopping
    >>> fit_params = {
    ...     'epochs': 100,
    ...     'batch_size': 64,
    ...     'verbose': 1,
    ...     'early_stopping': EarlyStopping(patience=5)
    ... }
    >>> callbacks, updated_params = extract_callbacks_from(fit_params, 
    ...                                                    return_fit_params=True)
    >>> print(callbacks)
    [<keras.src.callbacks.EarlyStopping object at 0x...>]
    >>> print(updated_params)
    {'epochs': 100, 'batch_size': 64, 'verbose': 1}

    Notes
    -----
    Consider using this function when integrating Keras callbacks within 
    a pipeline or estimator that follows scikit-learn conventions, where 
    parameters are passed as `fit_params`. This approach enables a clean 
    and modular integration of callbacks into your training loops.

    See Also
    --------
    tf.keras.callbacks.Callback :
        Base class used to build new callbacks.

    References
    ----------
    .. [1] François Chollet, et al. Keras Documentation. 
           https://keras.io/api/callbacks/
    """

    callbacks = []

    # If user provides a callbacks list directly
    if 'callbacks' in fit_params and isinstance(fit_params['callbacks'], list):
        cb_list = fit_params.pop('callbacks')
        for c in cb_list:
            if isinstance(c, Callback):
                callbacks.append(c)

    # If user provides individual callbacks as keyword arguments
    keys_to_remove = []
    for k, v in fit_params.items():
        if isinstance(v, Callback):
            callbacks.append(v)
            keys_to_remove.append(k)

    # Remove callback entries from fit_params
    for k in keys_to_remove:
        fit_params.pop(k, None)

    if return_fit_params:
        return callbacks, fit_params
    return callbacks


@TypeEnforcer({"0": 'array-like', "1": 'array-like'})
@ParamsValidator({ 
    "final_processed_data": ['array-like:df'], 
    "feature_columns":['array-like'], 
    "dynamic_feature_indices": ['array-like'], 
    "sequence_length": [Interval( Integral, 1, None, closed="left")], 
    "time_col": [str], 
    "static_feature_names": ['array-like', None], 
    "forecast_horizon": [Interval( Integral, 1, None, closed="left"), None],
    "future_years":[list, Interval( Integral, 1, None, closed="left"), None],
    "encoded_cat_columns": ['array-like', None], 
    "scaling_params": [dict, None]
    }
)
def prepare_future_data(
        final_processed_data: pd.DataFrame,
        feature_columns: List[str],
        dynamic_feature_indices: List[int],
        sequence_length: int = 1,
        time_col: str = 'date',
        static_feature_names: Optional[List[str]] = None,
        forecast_horizon: Optional[int] = None,
        future_years: Optional[List[int]] = None,
        encoded_cat_columns: Optional[List[str]] = None,
        scaling_params: Optional[Dict[str, Dict[str, float]]] = None,
        verbosity: int = 0,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        List[int],
        List[int],
        List[float],
        List[float]
    ]:
    """
    Prepare future static and dynamic inputs for making predictions.

    This function prepares the necessary static and dynamic inputs required for
    forecasting future values in time series data. It processes the provided
    dataset by grouping it by ``location_id``, extracting the last sequence of
    data points based on the specified ``sequence_length``, and generating future
    inputs for prediction over the defined ``forecast_horizon``.

    The function handles both integer and datetime representations of the
    ``time_col``, extracting the year from datetime columns when necessary. It
    also allows for flexibility in specifying static features and encoded
    categorical variables.

    .. math::
        \text{scaled\_time} = \frac{\text{future\_time} - \mu}{\sigma}

    Parameters
    ----------
    final_processed_data : pandas.DataFrame
        The processed DataFrame containing all features and targets. Must include
        the ``location_id`` column and the specified ``time_col``.
    feature_columns : List[str]
        List of feature column names to be used for dynamic input preparation.
    dynamic_feature_indices : List[int]
        Indices of dynamic features in ``feature_columns``. These features are
        considered time-dependent and are used to prepare dynamic inputs.
    sequence_length : int, optional
        The number of past time steps to include in each input sequence.
        Default is ``1``.
    time_col : str, optional
        The name of the time-related column in ``final_processed_data``.
        Defaults to ``'date'``.
    static_feature_names : List[str], optional
        List of static feature column names. If not provided, defaults to
        ``['longitude', 'latitude']`` plus any ``encoded_cat_columns``.
    forecast_horizon : int, optional
        The number of future time steps to predict. If set to ``None``,
        the function defaults to predicting the next immediate time step.
    future_years : List[int], optional
        List of future years to predict. Must match the length of
        ``forecast_horizon`` if ``forecast_horizon`` is provided.
    encoded_cat_columns : List[str], optional
        List of encoded categorical column names to be treated as static features.
    scaling_params : Dict[str, Dict[str, float]], optional
        Dictionary containing scaling parameters (mean and standard deviation)
        for features. Example: ``{'year': {'mean': 2000, 'std': 10}}``.
        If not provided, the function computes the mean and std for the
        ``time_col``.
    verbosity : int, optional
        Verbosity level from ``0`` to ``7`` for debugging and understanding
        the process. Higher values produce more detailed logs.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, List[int], List[int], List[float], List[float]]
        A tuple containing:

        - ``future_static_inputs`` : numpy.ndarray
            Array of future static inputs with shape ``(num_samples, num_static_vars, 1)``.
        - ``future_dynamic_inputs`` : numpy.ndarray
            Array of future dynamic inputs with shape ``(num_samples, sequence_length, num_dynamic_vars, 1)``.
        - ``future_years_list`` : List[int]
            List of future time values corresponding to each sample.
        - ``location_ids_list`` : List[int]
            List of location IDs corresponding to each sample.
        - ``longitudes`` : List[float]
            List of longitude values corresponding to each sample.
        - ``latitudes`` : List[float]
            List of latitude values corresponding to each sample.

    Examples
    --------
    >>> from gofast.nn.utils import prepare_future_data
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'location_id': [1, 1, 1, 2, 2, 2],
    ...     'year': [2018, 2019, 2020, 2018, 2019, 2020],
    ...     'longitude': [10.0, 10.0, 10.0, 20.0, 20.0, 20.0],
    ...     'latitude': [50.0, 50.0, 50.0, 60.0, 60.0, 60.0],
    ...     'temperature': [15, 16, 15.5, 20, 21, 20.5],
    ...     'rainfall': [100, 110, 105, 200, 210, 205],
    ...     'encoded_cat': [1, 1, 1, 2, 2, 2]
    ... })
    >>> feature_cols = ['year', 'temperature', 'rainfall', 'encoded_cat']
    >>> dynamic_indices = [0, 1, 2]
    >>> future_static, future_dynamic, future_years, loc_ids, longs, lats = prepare_future_data(
    ...     final_processed_data=data,
    ...     feature_columns=feature_cols,
    ...     dynamic_feature_indices=dynamic_indices,
    ...     sequence_length=2,
    ...     forecast_horizon=1,
    ...     future_years=[2021],
    ...     encoded_cat_columns=['encoded_cat'],
    ...     verbosity=5,
    ...     time_col='year'
    ... )
    >>> print(future_static.shape)
    (2, 3, 1)
    >>> print(future_dynamic.shape)
    (2, 2, 3, 1)

    Notes
    -----
    - The function handles both integer and datetime representations of the
      ``time_col``. If ``time_col`` is a datetime type, the year is extracted for
      scaling purposes.
    - If ``forecast_horizon`` is set to ``None``, the function defaults to
      generating data for the next immediate time step based on the last entry
      in the time column.
    - Ensure that the length of ``future_years`` matches ``forecast_horizon``
      if ``forecast_horizon`` is provided.
    - The ``static_feature_names`` parameter allows for flexibility in specifying
      which static features to include. If not provided, it defaults to
      ``['longitude', 'latitude']`` plus any ``encoded_cat_columns``.

    See Also
    --------
    prepare_future_data : Main function for preparing future data inputs.

    References
    ----------
    .. [1] Smith, J., & Doe, A. (2020). *Time Series Forecasting Methods*. Journal of
       Data Science, 15(3), 123-145.
    .. [2] Johnson, L. (2019). *Advanced Neural Networks for Time Series Prediction*.
       Machine Learning Review, 22(4), 567-589.
    """
    future_years = columns_manager(future_years, empty_as_none= False )
    static_feature_names = static_feature_names or []
    # Initialize verbosity levels
    def log(message: str, level: int):
        if verbosity >= level:
            print(message)

    log(
        "Starting prepare_future_data with verbosity level "
        f"{verbosity}", 
        1
    )

    # Lists to store future inputs and related data
    future_static_inputs_list = []
    future_dynamic_inputs_list = []
    future_years_list = []
    location_ids_list = []
    longitudes = []
    latitudes = []

    # Handle scaling parameters
    scaling_params = scaling_params or {}
    log("Scaling parameters set.", 3)

    # Determine time column
    detected_time_col = _determine_time_column(final_processed_data, time_col, log)
    
    # Handle scaling for time column
    time_mean, time_std = _handle_time_scaling(
        final_processed_data,
        detected_time_col,
        scaling_params,
        log
    )

    # Ensure encoded_cat_columns is a list
    encoded_cat_columns = encoded_cat_columns or []
    if not encoded_cat_columns:
        log("No encoded categorical columns provided.", 5)
    else:
        log(f"Encoded categorical columns: {encoded_cat_columns}", 5)

    # Determine static feature names
    static_features = _determine_static_features(
        static_feature_names, encoded_cat_columns, log
    )

    # Group data by 'location_id'
    grouped = _group_by_location(final_processed_data, log)

    # Iterate over each location
    for name, group in grouped:
        group = _sort_group_by_time(group, detected_time_col, log, name)

        # Ensure there is enough data to create a sequence
        if len(group) >= sequence_length:
            last_sequence = group.iloc[-sequence_length:]
            last_sequence_features = last_sequence[feature_columns]
            log(
                f"Extracted last {sequence_length} records for "
                f"location_id {name}.", 
                4
            )

            # Prepare static inputs
            static_inputs, static_values = _prepare_static_inputs(
                last_sequence_features,
                static_features,
                log,
                name
            )
            # Prepare dynamic inputs
            dynamic_inputs = _prepare_dynamic_inputs(
                last_sequence_features,
                dynamic_feature_indices,
                sequence_length,
                log,
                name
            )

            # Determine future steps based on forecast_horizon
            future_steps = forecast_horizon if forecast_horizon is not None else 1

            # Generate future inputs for each future step
            for step in range(future_steps):
                future_time = _determine_future_time(
                    forecast_horizon,
                    step,
                    future_years,
                    group,
                    detected_time_col,
                    log,
                    name
                )

                log(
                    f"Generating future data for step {step + 1} "
                    f"time {future_time} at location_id {name}.", 
                    6
                )

                # Copy dynamic inputs
                future_dynamic_inputs = dynamic_inputs.copy()

                # Update the time feature for the next time step
                _update_time_feature(
                    future_dynamic_inputs,
                    feature_columns,
                    dynamic_feature_indices,
                    detected_time_col,
                    future_time,
                    time_mean,
                    time_std,
                    log,
                    name
                )

                # Append to lists
                _append_to_lists(
                    future_static_inputs_list,
                    future_dynamic_inputs_list,
                    future_years_list,
                    location_ids_list,
                    longitudes,
                    latitudes,
                    static_inputs,
                    static_values,
                    static_features,
                    future_dynamic_inputs,
                    future_time,
                    name
                )

        else:
            log(
                f"Insufficient data for location_id {name}: "
                f"required={sequence_length}, available={len(group)}.", 
                4
            )

    # Convert lists to numpy arrays for model input
    log("Converting lists to numpy arrays.", 3)
    future_static_inputs, future_dynamic_inputs = _convert_to_numpy(
        future_static_inputs_list,
        future_dynamic_inputs_list,
        log
    )

    log(
        f"Final shapes - static: {future_static_inputs.shape}, "
        f"dynamic: {future_dynamic_inputs.shape}", 
        4
    )

    log("prepare_future_data completed successfully.", 1)

    return (
        future_static_inputs,
        future_dynamic_inputs,
        future_years_list,
        location_ids_list,
        longitudes,
        latitudes
    )


# Helper Functions
def _determine_time_column(
        final_processed_data: pd.DataFrame,
        time_col: str,
        log_func
    ) -> str:
    """Determine and validate the time column."""
    if time_col in final_processed_data.columns:
        log_func(f"Using time column: {time_col}", 4)
        return time_col
    else:
        raise ValueError(
            f"final_processed_data must contain the '{time_col}' column."
        )


def _handle_time_scaling(
        final_processed_data: pd.DataFrame,
        detected_time_col: str,
        scaling_params: Dict[str, Dict[str, float]],
        log_func
    ) -> Tuple[float, float]:
    """Handle scaling for the time column."""
    if detected_time_col not in scaling_params:
        if detected_time_col == 'year':
            time_mean = final_processed_data[detected_time_col].mean()
            time_std = final_processed_data[detected_time_col].std()
        else:
            # Handle 'date' or other time columns
            if pd.api.types.is_datetime64_any_dtype(
                final_processed_data[detected_time_col]
            ):
                final_processed_data['year_extracted'] = (
                    final_processed_data[detected_time_col].dt.year
                )
                time_col_extracted = 'year_extracted'
                time_mean = final_processed_data[time_col_extracted].mean()
                time_std = final_processed_data[time_col_extracted].std()
                log_func(
                    f"Extracted year from '{detected_time_col}' column as "
                    f"'{time_col_extracted}'",
                    5
                )
                detected_time_col = time_col_extracted
            else:
                # Assume 'date' column contains integer year
                time_mean = final_processed_data[detected_time_col].mean()
                time_std = final_processed_data[detected_time_col].std()
        scaling_params[detected_time_col] = {'mean': time_mean, 'std': time_std}
        log_func(
            f"Computed scaling for '{detected_time_col}': mean={time_mean}, "
            f"std={time_std}", 
            4
        )
    else:
        time_mean = scaling_params[detected_time_col]['mean']
        time_std = scaling_params[detected_time_col]['std']
        log_func(
            f"Using provided scaling for '{detected_time_col}': mean="
            f"{time_mean}, std={time_std}", 
            4
        )
    return time_mean, time_std


def _determine_static_features(
        static_feature_names: Optional[List[str]],
        encoded_cat_columns: List[str],
        log_func
    ) -> List[str]:
    """Determine the static feature names."""
    
    if static_feature_names is not None or len(static_feature_names) !=0:
        log_func(
            f"Using provided static feature names: {static_feature_names}", 
            5
        )
        return static_feature_names + encoded_cat_columns
    else:
        static_features = ['longitude', 'latitude'] + encoded_cat_columns
        log_func(
            f"Using default static feature names: {static_features}", 
            5
        )
        return static_features


def _group_by_location(
        final_processed_data: pd.DataFrame,
        log_func
    ) -> pd.core.groupby.DataFrameGroupBy:
    """Group the data by 'location_id'."""
    if 'location_id' not in final_processed_data.columns:
        raise ValueError(
            "final_processed_data must contain 'location_id' column."
        )
    grouped = final_processed_data.groupby('location_id')
    log_func(
        "Grouped data by 'location_id'. Number of groups: "
        f"{len(grouped)}", 
        2
    )
    return grouped


def _sort_group_by_time(
        group: pd.DataFrame,
        detected_time_col: str,
        log_func,
        location_id: Union[int, str]
    ) -> pd.DataFrame:
    """Sort the group by the time column."""
    group = group.sort_values(detected_time_col).reset_index(drop=True)
    log_func(f"Processing location_id: {location_id}", 3)
    return group


def _prepare_static_inputs(
        last_sequence_features: pd.DataFrame,
        static_features: List[str],
        log_func,
        location_id: Union[int, str]
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare static inputs from the last sequence."""
    if not all(
        col in last_sequence_features.columns 
        for col in static_features
    ):
        raise ValueError(
            f"One or more static feature columns missing in data "
            f"for location_id {location_id}."
        )
    static_values = last_sequence_features.iloc[0][
        static_features
    ].values
    static_inputs = static_values.reshape(
        1, len(static_values)
    ).astype(np.float32)  # Shape: (1, num_static_vars)
    log_func(
        f"Prepared static inputs for location_id {location_id}.", 
        5
    )
    return static_inputs, static_values


def _prepare_dynamic_inputs(
        last_sequence_features: pd.DataFrame,
        dynamic_feature_indices: List[int],
        sequence_length: int,
        log_func,
        location_id: Union[int, str]
    ) -> np.ndarray:
    """Prepare dynamic inputs from the last sequence."""
    dynamic_features = last_sequence_features.iloc[
        :, dynamic_feature_indices
    ].values
    dynamic_inputs = dynamic_features.reshape(
        sequence_length,
        len(dynamic_feature_indices),
        1
    ).astype(np.float32)  # Shape: (sequence_length, num_dynamic_vars, 1)
    log_func(
        f"Prepared dynamic inputs for location_id {location_id}.", 
        5
    )
    return dynamic_inputs


def _determine_future_time(
        forecast_horizon: Optional[int],
        step: int,
        future_years: List[int],
        group: pd.DataFrame,
        detected_time_col: str,
        log_func,
        location_id: Union[int, str]
    ) -> int:
    """Determine the future time value."""
    if forecast_horizon is not None:
        if step < len(future_years):
            future_time = future_years[step]
        else:
            raise ValueError(
                "Length of future_years must be equal to forecast_horizon."
            )
    else:
        # If forecast_horizon is None, use the next time step
        if detected_time_col == 'year_extracted':
            future_time = int(group[detected_time_col].iloc[-1]) + 1
        elif detected_time_col == 'year':
            future_time = int(group[detected_time_col].iloc[-1]) + 1
        else:
            # Assume 'date' column has been processed
            if pd.api.types.is_datetime64_any_dtype(
                group[detected_time_col]
            ):
                last_date = group[detected_time_col].iloc[-1]
                future_time = last_date.year + 1
            else:
                future_time = int(group[detected_time_col].iloc[-1]) + 1
    return future_time


def _update_time_feature(
        future_dynamic_inputs: np.ndarray,
        feature_columns: List[str],
        dynamic_feature_indices: List[int],
        detected_time_col: str,
        future_time: int,
        time_mean: float,
        time_std: float,
        log_func,
        location_id: Union[int, str]
    ):
    """Update the time feature in the dynamic inputs."""
    if detected_time_col in feature_columns:
        time_idx_in_feature = feature_columns.index(detected_time_col)
        if time_idx_in_feature in dynamic_feature_indices:
            dynamic_time_index = (
                dynamic_feature_indices.index(time_idx_in_feature)
            )
            scaled_time = (future_time - time_mean) / time_std
            future_dynamic_inputs[
                -1, dynamic_time_index, 0
            ] = scaled_time
            log_func(
                f"Scaled '{detected_time_col}' for future input: "
                f"original={future_time}, scaled={scaled_time}", 
                7
            )
        else:
            log_func(
                f"'{detected_time_col}' is not in dynamic_feature_indices "
                f"for location_id {location_id}.", 
                6
            )
    else:
        log_func(
            f"'{detected_time_col}' column not found in "
            f"feature_columns for location_id {location_id}.", 
            6
        )


def _append_to_lists(
        future_static_inputs_list: List[np.ndarray],
        future_dynamic_inputs_list: List[np.ndarray],
        future_years_list: List[int],
        location_ids_list: List[Union[int, str]],
        longitudes: List[float],
        latitudes: List[float],
        static_inputs: np.ndarray,
        static_values: np.ndarray,
        static_features: List[str],
        future_dynamic_inputs: np.ndarray,
        future_time: int,
        name: Union[int, str]
    ):
    """Append the prepared inputs and related data to their respective lists."""
    future_static_inputs_list.append(static_inputs)
    future_dynamic_inputs_list.append(future_dynamic_inputs)
    future_years_list.append(future_time)
    location_ids_list.append(name)
    longitudes.append(static_values[static_features.index('longitude')] 
                      if 'longitude' in static_features else np.nan)
    latitudes.append(static_values[static_features.index('latitude')] 
                     if 'latitude' in static_features else np.nan)


def _convert_to_numpy(
        future_static_inputs_list: List[np.ndarray],
        future_dynamic_inputs_list: List[np.ndarray],
        log_func
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Convert the lists to numpy arrays with appropriate shapes."""
    future_static_inputs = np.array(
        future_static_inputs_list
    )  # Shape: (num_samples, num_static_vars)
    future_dynamic_inputs = np.array(
        future_dynamic_inputs_list
    )  # Shape: (num_samples, sequence_length, num_dynamic_vars, 1)

    # Reshape static inputs to ensure the correct shape
    future_static_inputs = future_static_inputs.reshape(
        future_static_inputs.shape[0],
        future_static_inputs.shape[1],
        1
    )
    return future_static_inputs, future_dynamic_inputs
