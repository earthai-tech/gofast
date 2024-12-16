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

from numbers import Integral, Real 
import warnings
from typing import List, Tuple, Optional, Union, Dict, Callable, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ..core.checks import (
    ParamsValidator, 
    are_all_frames_valid, 
    exist_features, 
    check_params 
    )
from ..core.handlers import TypeEnforcer, columns_manager 
from ..core.io import is_data_readable 
from ..compat.sklearn import ( 
    StrOptions, 
    Interval, 
    HasMethods, 
    Hidden, 
   validate_params
)
from ..decorators import DynamicMethod
from ..utils.deps_utils import ensure_pkg 
from ..utils.validator import validate_sequences, check_consistent_length 
from . import KERAS_DEPS, KERAS_BACKEND, dependency_message

if KERAS_BACKEND:
    Callback=KERAS_DEPS.Callback
    
DEP_MSG = dependency_message('utils') 

__all__ = [
    "split_static_dynamic", 
    "create_sequences",
    "compute_forecast_horizon", 
    "prepare_future_data", 
    "compute_anomaly_scores"
   ]

@check_params(
    {
     'domain_func': Optional[Callable]
     }, 
    coerce=False
 )
@ParamsValidator( 
    { 
        'y_true': ['array-like:np:transf'], 
        'y_pred': ['array-like:np:transf', None], 
        'method': [StrOptions(
            {'statistical', 'domain', 'isolation_forest','residual'})
            ], 
        'threshold': [Interval(Real, 1, None, closed ='left')], 
        'contamination': [Interval(Real, 0, None, closed='left')], 
        'epsilon': [Hidden(Interval(Real, 0, 1 , closed ='neither'))], 
        'estimator':[HasMethods(['fit', 'predict']), None], 
        'random_state': ['random_state', None], 
        'residual_metric': [StrOptions({'mse', 'mae','rmse'})], 
        'objective': [str]
    }
)
def compute_anomaly_scores(
    y_true,
    y_pred=None,
    method='statistical',
    threshold=3.0,
    domain_func=None,
    contamination=0.05,
    epsilon=1e-6,
    estimator=None,
    random_state=None,
    residual_metric='mse', 
    objective='ts',
    verbose=1
):
    r"""
    Compute anomaly scores for given true targets using various methods.
    
    This utility function, ``anomaly_scores``, provides a flexible approach
    to compute anomaly scores outside the XTFT model [1]_. Anomaly scores
    serve as indicators of how unusual certain observations are, guiding
    the model towards more robust and stable forecasts. By detecting and
    quantifying anomalies, practitioners can adjust forecasting strategies,
    improve predictive performance, and handle irregular patterns more
    effectively.
    
    Parameters
    ----------
    y_true : np.ndarray
        The ground truth target values with shape ``(B, H, O)``, where:
        - `B`: batch size
        - `H`: number of forecast horizons (time steps ahead)
        - `O`: output dimension (e.g., number of target variables).
        
        Typically, `y_true` corresponds to the same array passed as the
        forecast target to the model. All computations of anomalies
        are relative to these true values or, if provided, their
        predicted counterparts `y_pred`.
    
    y_pred : np.ndarray, optional
        The predicted values with shape ``(B, H, O)``. If provided and
        the `method` is set to `'residual'`, the anomaly scores are derived
        from the residuals between `y_true` and `y_pred`. In this scenario,
        anomalies reflect discrepancies indicating unusual conditions
        or model underperformance.
    
    method : str, optional
        The method used to compute anomaly scores. Supported options:
        - ``"statistical"`` or ``"stats"``:  
          Uses mean and standard deviation of `y_true` to measure deviation
          from the mean. Points far from the mean by a certain factor
          (controlled by `threshold`) yield higher anomaly scores.
          
          Formally, let :math:`\mu` be the mean of `y_true` and 
          :math:`\sigma` its standard deviation. The anomaly score for 
          a point :math:`y` is:  
          .. math::  
             (\frac{y - \mu}{\sigma + \varepsilon})^2
          
          where :math:`\varepsilon` is a small constant for numerical
          stability.
        
        - ``"domain"``:  
          Uses a domain-specific heuristic (provided by `domain_func`)
          to compute scores. If no `domain_func` is provided, a default
          heuristic marks negative values as anomalies.
    
        - ``"isolation_forest"`` or ``"if"``:  
          Employs the IsolationForest algorithm to detect outliers. 
          The model learns a structure to isolate anomalies more quickly
          than normal points. Higher contamination rates allow more
          points to be considered anomalous.
    
        - ``"residual"``:  
          If `y_pred` is provided, anomalies are derived from residuals:
          the difference `(y_true - y_pred)`. By default, mean squared 
          error (`mse`) is used. Other metrics include `mae` and `rmse`,
          offering flexibility in quantifying deviations:
          .. math::
             \text{MSE: }(y_{true} - y_{pred})^2
    
        Default is ``"statistical"``.
    
    threshold : float, optional
        Threshold factor for the `statistical` method. Defines how far
        beyond mean ± (threshold * std) is considered anomalous. Though
        not directly applied as a mask here, it can guide interpretation
        of scores. Default is ``3.0``.
    
    domain_func : callable, optional
        A user-defined function for `domain` method. It takes `y_true`
        as input and returns an array of anomaly scores with the same
        shape. If none is provided, the default heuristic:
        .. math::
           \text{anomaly}(y) = \begin{cases}
           |y| \times 10 & \text{if } y < 0 \\
           0 & \text{otherwise}
           \end{cases}
    
    contamination : float, optional
        Used in the `isolation_forest` method. Specifies the proportion
        of outliers in the dataset. Default is ``0.05``.
    
    epsilon : float, optional
        A small constant :math:`\varepsilon` for numerical stability
        in calculations, especially during statistical normalization.
        Default is ``1e-6``.
    
    estimator : object, optional
        A pre-fitted IsolationForest estimator for the `isolation_forest`
        method. If not provided, a new estimator will be created and fitted
        to `y_true`.
    
    random_state : int, optional
        Sets a random state for reproducibility in the `isolation_forest`
        method.
    
    residual_metric : str, optional
        The metric used to compute anomalies from residuals if `method`
        is set to `'residual'`. Supported metrics:
        - ``"mse"``: mean squared error per point `(residuals**2)`
        - ``"mae"``: mean absolute error per point `|residuals|`
        - ``"rmse"``: root mean squared error `sqrt((residuals**2))`
        
        Default is ``"mse"``.
    
    objective : str, optional
        Specifies the type of objective, for future extensibility.
        Default is ``"ts"`` indicating time series. Could be extended
        for other tasks in the future.
    
    verbose : int, optional
        Controls verbosity. If `verbose=1`, some messages or warnings
        may be printed. Higher values might produce more detailed logs.
    
    Returns
    -------
    anomaly_scores : np.ndarray
        An array of anomaly scores with the same shape as `y_true`.
        Higher values indicate more unusual or anomalous points.
    
    Notes
    -----
    Choosing an appropriate method depends on the data characteristics,
    domain requirements, and model complexity. Statistical methods
    are quick and interpretable but may oversimplify anomalies. Domain
    heuristics leverage expert knowledge, while isolation forest applies
    a more robust, data-driven approach. Residual-based anomalies help
    assess model performance and highlight periods where the model 
    struggles.
    
    Examples
    --------
    >>> from gofast.nn.losses import compute_anomaly_scores
    >>> import numpy as np
    
    >>> # Statistical method example
    >>> y_true = np.random.randn(32, 20, 1)  # (B,H,O)
    >>> scores = compute_anomaly_scores(y_true, method='statistical', threshold=3)
    >>> scores.shape
    (32, 20, 1)
    
    >>> # Domain-specific example
    >>> def my_heuristic(y):
    ...     return np.where(y < -1, np.abs(y)*5, 0.0)
    >>> scores = compute_anomaly_scores(y_true, method='domain',
                                        domain_func=my_heuristic)
    
    >>> # Isolation Forest example
    >>> scores = compute_anomaly_scores(y_true, method='isolation_forest',
                                        contamination=0.1)
    
    >>> # Residual-based example
    >>> y_pred = y_true + np.random.normal(0, 1, y_true.shape)  # Introduce noise
    >>> scores = compute_anomaly_scores(y_true, y_pred=y_pred, method='residual',
                                        residual_metric='mae')
    
    See Also
    --------
    :func:`compute_quantile_loss` : 
        For computing quantile losses.
    :func:`compute_multi_objective_loss` :
        For integrating anomaly scores into a multi-objective loss.
    
    References
    ----------
    .. [1] Wang, X., et al. "Enhanced Temporal Fusion Transformer for Time
           Series Forecasting." International Journal of Forecasting, 37(3),
           2021.
    """
    if objective == 'ts': 
        if y_true.ndim != 3:
            raise ValueError(
                "`y_true` must be a 3-dimensional array with the shape (B, H, O):\n"
                "  - B: Batch size (number of samples per batch).\n"
                "  - H: Number of horizons (time steps ahead).\n"
                "  - O: Output dimension (e.g., number of target variables).\n"
                "Please ensure the input conforms to the specified shape."
            )
    elif y_true.ndim not in [1, 2]:
        raise ValueError(
            "`y_true` must be a 1D or 2D array for non-time-series objectives.\n"
            f"Received shape: {y_true.shape}."
        )
    
    if y_pred is not None:
        check_consistent_length(y_true, y_pred)

    # Normalize method aliases
    method = method.lower()
    method_map = {
        'stats': 'statistical',
        'statistical': 'statistical',
        'if': 'isolation_forest',
        'isolation_forest': 'isolation_forest',
    }
    method = method_map.get(method, method)

    if verbose >= 3:
        print(f"[INFO] Using method: {method}")

    if method == 'statistical':
        mean = y_true.mean()
        std = y_true.std()
        anomaly_scores = ((y_true - mean) / (std + epsilon))**2
        if verbose >= 5:
            print(f"[DEBUG] Mean: {mean}, Std: {std}")

    elif method == 'domain':
        if domain_func is None:
            # Default heuristic: negative values are anomalies
            anomaly_scores = np.where(y_true < 0, np.abs(y_true) * 10, 0.0)
        else:
            anomaly_scores = domain_func(y_true)
            if not isinstance(anomaly_scores, np.ndarray):
                raise ValueError("`domain_func` must return a numpy array.")
            if anomaly_scores.shape != y_true.shape:
                raise ValueError(
                    "`domain_func` output shape must match `y_true` shape.")
        if verbose >= 5:
            print("[DEBUG] Domain-based anomaly scores calculated.")

    elif method == 'isolation_forest':
        flat_data = y_true.reshape(-1, 1)
        if estimator is None:
            iso = IsolationForest(
                contamination=contamination,
                random_state=random_state
            )
            iso.fit(flat_data)
        else:
            iso = estimator

        raw_scores = iso.score_samples(flat_data)
        anomaly_scores = -raw_scores.reshape(y_true.shape)
        if verbose >= 5:
            print("[DEBUG] Isolation Forest raw scores computed.")

    elif method == 'residual':
        if y_pred is None:
            raise ValueError("`y_pred` must be provided if method='residual'.")
        if not isinstance(y_pred, np.ndarray):
            raise ValueError("`y_pred` must be a numpy array.")
        if y_pred.shape != y_true.shape:
            raise ValueError(
                "`y_pred` shape must match `y_true` shape for residual method.")

        residuals = y_true - y_pred
        residual_metric = residual_metric.lower()

        if residual_metric == 'mse':
            # Mean Squared Error per point is just squared residual
            anomaly_scores = residuals**2
        elif residual_metric == 'mae':
            # Mean Absolute Error per point is absolute residual
            anomaly_scores = np.abs(residuals)
        elif residual_metric == 'rmse':
            # Root Mean Squared Error per point: sqrt(residual^2)
            # Essentially absolute value, but we'll respect definition
            anomaly_scores = np.sqrt((residuals**2) + epsilon)
        if verbose >= 5:
            print(f"[DEBUG] Residuals calculated using {residual_metric}.")

    if verbose >= 1:
        print("[INFO] Anomaly scores computed successfully.")
    
    return anomaly_scores



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

@ensure_pkg('tensorflow', extra ="Need 'tensorflow' for this function to proceed")
def validate_anomaly_scores(
    anomaly_config: Optional[Dict[str, Any]], 
    forecast_horizons: int, 
) :
    """
    Validates and processes the 'anomaly_scores' in the provided anomaly_config dictionary.

    Parameters:
    - anomaly_config (Optional[Dict[str, Any]]): 
        Dictionary that may contain:
            - 'anomaly_scores': Precomputed anomaly scores tensor.
            - 'anomaly_loss_weight': Weight for anomaly loss.
    - forecast_horizons (int): 
        The expected number of forecast horizons (second dimension of 'anomaly_scores').

    Returns:
    - Optional[tf.Tensor]: 
        Validated 'anomaly_scores' tensor of shape (batch_size, forecast_horizons), cast to float32.
        Returns None if 'anomaly_scores' is not provided.

    Raises:
    - ValueError: 
        If 'anomaly_scores' is provided but is not a 2D tensor or the second 
        dimension does not match 'forecast_horizons'.
    """
    import tensorflow as tf 
    
    if anomaly_config is None:
        # If anomaly_config is None, no anomaly_scores or anomaly_loss_weight are set
        return None

    # Ensure 'anomaly_scores' key exists in the dictionary
    if 'anomaly_scores' not in anomaly_config:
        anomaly_config['anomaly_scores'] = None

    anomaly_scores = anomaly_config.get('anomaly_scores')

    if anomaly_scores is not None:
        # Convert to tensor if not already a TensorFlow tensor
        if not isinstance(anomaly_scores, tf.Tensor):
            try:
                anomaly_scores = tf.convert_to_tensor(anomaly_scores, dtype=tf.float32)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Failed to convert 'anomaly_scores' to a TensorFlow tensor: {e}")
        else:
            # Cast to float32 if it's already a tensor
            anomaly_scores = tf.cast(anomaly_scores, tf.float32)

        # Validate that anomaly_scores is a 2D tensor
        if anomaly_scores.ndim != 2:
            raise ValueError(
                f"'anomaly_scores' must be a 2D tensor with shape (batch_size, forecast_horizons), "
                f"but got {anomaly_scores.ndim}D tensor."
            )

        # Validate that the second dimension matches forecast_horizons
        if anomaly_scores.shape[1] != forecast_horizons:
            raise ValueError(
                f"'anomaly_scores' second dimension must be {forecast_horizons}, "
                f"but got {anomaly_scores.shape[1]}."
            )

        # Update the anomaly_config with the processed anomaly_scores tensor
        anomaly_config['anomaly_scores'] = anomaly_scores
        return anomaly_scores

    else:
        # If 'anomaly_scores' is not provided, ensure it's set to None
        anomaly_config['anomaly_scores'] = None
        
        return anomaly_scores


def set_anomaly_config(anomaly_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Processes the anomaly_config dictionary to ensure it contains
    'anomaly_scores' and 'anomaly_loss_weight' keys.

    Parameters:
    - anomaly_config (Optional[Dict[str, Any]]): 
        A dictionary that may contain:
            - 'anomaly_scores': Precomputed anomaly scores tensor.
            - 'anomaly_loss_weight': Weight for anomaly loss.

    Returns:
    - Dict[str, Any]: 
        A dictionary with keys 'anomaly_scores' and 'anomaly_loss_weight',
        setting them to None if they were not provided.
    """
    if anomaly_config is None:
        return {'anomaly_loss_weight': None, 'anomaly_scores': None}
    
    # Create a copy to avoid mutating the original dictionary
    config = anomaly_config.copy()

    # Ensure 'anomaly_scores' key exists
    if 'anomaly_scores' not in config:
        config['anomaly_scores'] = None

    # Ensure 'anomaly_loss_weight' key exists
    if 'anomaly_loss_weight' not in config:
        config['anomaly_loss_weight'] = None

    return config


def validate_xtft_inputs(
    inputs: Union[List[Any], Tuple[Any, ...]],
    static_input_dim: int,
    dynamic_input_dim: int,
    future_covariate_dim: Optional[int] = None
):
    """
    Validates and processes the inputs for the XTFT model.

    Parameters
    ----------
    inputs : list or tuple
        A list or tuple containing the inputs to the model in the following order:
        [static_input, dynamic_input, future_covariate_input].

        - `static_input`: TensorFlow tensor or array-like object representing 
          static features.
        - `dynamic_input`: TensorFlow tensor or array-like object representing
          dynamic features.
        - `future_covariate_input`: (Optional) TensorFlow tensor or array-like
          object representing future covariates.
          Can be `None` if not used.

    static_input_dim : int
        The expected dimensionality of the static input features 
        (i.e., number of static features).

    dynamic_input_dim : int
        The expected dimensionality of the dynamic input features
        (i.e., number of dynamic features).

    future_covariate_dim : int, optional
        The expected dimensionality of the future covariate features 
        (i.e., number of future covariate features).
        If `None`, the function expects `future_covariate_input` to be `None`.

    Returns
    -------
    static_input : tf.Tensor
        Validated static input tensor of shape `(batch_size, static_input_dim)`
        and dtype `float32`.

    dynamic_input : tf.Tensor
        Validated dynamic input tensor of shape
        `(batch_size, time_steps, dynamic_input_dim)` and dtype `float32`.

    future_covariate_input : tf.Tensor or None
        Validated future covariate input tensor of shape 
        `(batch_size, time_steps, future_covariate_dim)` and dtype `float32`.
        Returns `None` if `future_covariate_dim` is `None` or if the input was `None`.

    Raises
    ------
    ValueError
        If `inputs` is not a list or tuple with the required number of elements.
        If `future_covariate_dim` is specified but `future_covariate_input` is `None`.
        If the provided inputs do not match the expected dimensionalities.
        If the inputs contain incompatible batch sizes.

    Examples
    --------
    >>> # Example without future covariates
    >>> static_input = tf.random.normal((32, 10))
    >>> dynamic_input = tf.random.normal((32, 20, 45))
    >>> inputs = [static_input, dynamic_input, None]
    >>> validated_static, validated_dynamic, validated_future = validate_xtft_inputs(
    ...     inputs,
    ...     static_input_dim=10,
    ...     dynamic_input_dim=45,
    ...     future_covariate_dim=None
    ... )
    >>> print(validated_static.shape, validated_dynamic.shape, validated_future)
    (32, 10) (32, 20, 45) None

    >>> # Example with future covariates
    >>> future_covariate_input = tf.random.normal((32, 20, 5))
    >>> inputs = [static_input, dynamic_input, future_covariate_input]
    >>> validated_static, validated_dynamic, validated_future = validate_xtft_inputs(
    ...     inputs,
    ...     static_input_dim=10,
    ...     dynamic_input_dim=45,
    ...     future_covariate_dim=5
    ... )
    >>> print(validated_static.shape, validated_dynamic.shape, validated_future.shape)
    (32, 10) (32, 20, 45) (32, 20, 5)
    """
    import tensorflow as tf
    # Step 1: Validate the type and length of inputs
    if not isinstance(inputs, (list, tuple)):
        raise ValueError(
            f"'inputs' must be a list or tuple, but got type {type(inputs).__name__}."
        )
    
    expected_length = 3
    if len(inputs) != expected_length:
        raise ValueError(
            f"'inputs' must contain exactly {expected_length} elements: "
            f"[static_input, dynamic_input, future_covariate_input]. "
            f"Received {len(inputs)} elements."
        )
    
    # Unpack inputs
    static_input, dynamic_input, future_covariate_input = inputs

    # Step 2: Validate static_input
    if static_input is None:
        raise ValueError("'static_input' cannot be None.")
    
    # Convert to tensor if not already
    if not isinstance(static_input, tf.Tensor):
        try:
            static_input = tf.convert_to_tensor(static_input, dtype=tf.float32)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Failed to convert 'static_input' to a TensorFlow tensor: {e}")
    else:
        # Ensure dtype is float32
        static_input = tf.cast(static_input, tf.float32)
    
    # Check static_input dimensions
    if static_input.ndim != 2:
        raise ValueError(
            f"'static_input' must be a 2D tensor with shape (batch_size, static_input_dim), "
            f"but got {static_input.ndim}D tensor."
        )
    
    # Check static_input_dim
    if static_input.shape[1] is not None and static_input.shape[1] != static_input_dim:
        raise ValueError(
            f"'static_input' has incorrect feature dimension. Expected {static_input_dim}, "
            f"but got {static_input.shape[1]}."
        )
    elif static_input.shape[1] is None:
        # Dynamic dimension, cannot validate now
        pass

    # Step 3: Validate dynamic_input
    if dynamic_input is None:
        raise ValueError("'dynamic_input' cannot be None.")
    
    # Convert to tensor if not already
    if not isinstance(dynamic_input, tf.Tensor):
        try:
            dynamic_input = tf.convert_to_tensor(dynamic_input, dtype=tf.float32)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Failed to convert 'dynamic_input' to a TensorFlow tensor: {e}")
    else:
        # Ensure dtype is float32
        dynamic_input = tf.cast(dynamic_input, tf.float32)
    
    # Check dynamic_input dimensions
    if dynamic_input.ndim != 3:
        raise ValueError(
            f"'dynamic_input' must be a 3D tensor with shape (batch_size, time_steps, dynamic_input_dim), "
            f"but got {dynamic_input.ndim}D tensor."
        )
    
    # Check dynamic_input_dim
    if dynamic_input.shape[2] is not None and dynamic_input.shape[2] != dynamic_input_dim:
        raise ValueError(
            f"'dynamic_input' has incorrect feature dimension. Expected {dynamic_input_dim}, "
            f"but got {dynamic_input.shape[2]}."
        )
    elif dynamic_input.shape[2] is None:
        # Dynamic dimension, cannot validate now
        pass
    
    # Step 4: Validate future_covariate_input
    if future_covariate_dim is not None:
        if future_covariate_input is None:
            raise ValueError(
                "'future_covariate_dim' is specified, but 'future_covariate_input' is None."
            )
        
        # Convert to tensor if not already
        if not isinstance(future_covariate_input, tf.Tensor):
            try:
                future_covariate_input = tf.convert_to_tensor(future_covariate_input, dtype=tf.float32)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Failed to convert 'future_covariate_input' to a TensorFlow tensor: {e}")
        else:
            # Ensure dtype is float32
            future_covariate_input = tf.cast(future_covariate_input, tf.float32)
        
        # Check future_covariate_input dimensions
        if future_covariate_input.ndim != 3:
            raise ValueError(
                f"'future_covariate_input' must be a 3D tensor with shape (batch_size, time_steps, future_covariate_dim), "
                f"but got {future_covariate_input.ndim}D tensor."
            )
        
        # Check future_covariate_dim
        if future_covariate_input.shape[2] is not None and future_covariate_input.shape[2] != future_covariate_dim:
            raise ValueError(
                f"'future_covariate_input' has incorrect feature dimension. Expected {future_covariate_dim}, "
                f"but got {future_covariate_input.shape[2]}."
            )
        elif future_covariate_input.shape[2] is None:
            # Dynamic dimension, cannot validate now
            pass
    else:
        if future_covariate_input is not None:
            raise ValueError(
                "'future_covariate_dim' is None, but 'future_covariate_input' is provided."
            )
    
    # Step 5: Validate batch sizes across inputs
    static_batch_size = tf.shape(static_input)[0]
    dynamic_batch_size = tf.shape(dynamic_input)[0]
    
    if future_covariate_dim is not None:
        future_batch_size = tf.shape(future_covariate_input)[0]
        # Check if all batch sizes are equal
        batch_size_cond = tf.reduce_all([
            tf.equal(static_batch_size, dynamic_batch_size),
            tf.equal(static_batch_size, future_batch_size)
        ])
    else:
        # Check only static and dynamic batch sizes
        batch_size_cond = tf.equal(static_batch_size, dynamic_batch_size)
    
    # Ensure batch sizes match
    if not batch_size_cond.numpy():
        raise ValueError(
            f"Batch sizes do not match across inputs: "
            f"'static_input' batch_size={static_batch_size.numpy()}, "
            f"'dynamic_input' batch_size={dynamic_batch_size.numpy()}" +
            (f", 'future_covariate_input' batch_size={future_batch_size.numpy()}" if future_covariate_dim is not None else "")
        )
    
    return static_input, dynamic_input, future_covariate_input

def set_default_params(
    quantiles: Union[str, List[float], None] = None,
    scales: Union[str, List[int], None] = None,
    multi_scale_agg: Union[str, None] = None
) -> Tuple[List[float], List[int], bool]:
    """
    Sets and validates default values for quantiles, scales, and return_sequences parameters.

    Parameters
    ----------
    quantiles : str, list of float, or None, optional
        Specifies the quantiles to be used for probabilistic forecasting.
        
        - If set to `'auto'`, it defaults to `[0.1, 0.5, 0.9]`.
        - If a list is provided, each element must be a float between 0 and 1 (exclusive).
        - If `None`, it remains as `None` (can be used for deterministic forecasting).

    scales : str, list of int, or None, optional
        Specifies the scaling factors to be used in multi-scale processing.
        
        - If set to `'auto'` or `None`, it defaults to `[1]`.
        - If a list is provided, each element must be a positive integer.

    multi_scale_agg : str or None, optional
        Determines the aggregation method for multi-scale features.
        
        - If set to `'auto'` or `None`, `return_sequences` is set to `False`.
        - Otherwise, `return_sequences` is set to `True`.
        - Expected aggregation methods could include `'sum'`, `'concat'`, etc., 
        depending on model requirements.

    Returns
    -------
    Tuple[List[float], List[int], bool]
        A tuple containing:
        
        - `quantiles`: A list of validated quantile floats.
        - `scales`: A list of validated scale integers.
        - `return_sequences`: A boolean indicating whether to return sequences based on `multi_scale_agg`.

    Raises
    ------
    ValueError
        If `quantiles` is neither `'auto'` nor a list of valid floats.
        If `scales` is neither `'auto'` nor a list of valid positive integers.
        If `multi_scale_agg` is provided but not a recognized aggregation method.

    Examples
    --------
    >>> # Example 1: Using default 'auto' settings
    >>> quantiles, scales, return_sequences = set_default_parameters(quantiles='auto', scales='auto', multi_scale_agg='auto')
    >>> print(quantiles)
    [0.1, 0.5, 0.9]
    >>> print(scales)
    [1]
    >>> print(return_sequences)
    False

    >>> # Example 2: Providing custom quantiles and scales
    >>> quantiles, scales, return_sequences = set_default_parameters(
    ...     quantiles=[0.05, 0.5, 0.95],
    ...     scales=[1, 2, 4],
    ...     multi_scale_agg='concat'
    ... )
    >>> print(quantiles)
    [0.05, 0.5, 0.95]
    >>> print(scales)
    [1, 2, 4]
    >>> print(return_sequences)
    True

    >>> # Example 3: Invalid quantiles input
    >>> set_default_parameters(quantiles=[-0.1, 1.2])
    Traceback (most recent call last):
    ...
    ValueError: Each quantile must be a float between 0 and 1 (exclusive). Invalid quantiles: [-0.1, 1.2]

    >>> # Example 4: Invalid scales input
    >>> set_default_parameters(scales=[0, -2])
    Traceback (most recent call last):
    ...
    ValueError: Each scale must be a positive integer. Invalid scales: [0, -2]
    """

    # Set default quantiles if 'auto'
    if quantiles == 'auto':
        quantiles = [0.1, 0.5, 0.9]
    elif quantiles is not None:
        if not isinstance(quantiles, list):
            raise ValueError(
                "'quantiles' must be a list of floats or"
               f" 'auto', but got type {type(quantiles).__name__}.")
        # Validate each quantile
        invalid_quantiles = [q for q in quantiles if not isinstance(q, float) or not (0 < q < 1)]
        if invalid_quantiles:
            raise ValueError(
                f"Each quantile must be a float between 0 and 1 (exclusive). "
                f"Invalid quantiles: {invalid_quantiles}"
            )
    else:
        # quantiles remains None
        pass

    # Set default scales if 'auto' or None
    if scales is None or scales == 'auto':
        scales = [1]
    elif isinstance(scales, list):
        # Validate each scale
        invalid_scales = [s for s in scales if not isinstance(s, int) or s <= 0]
        if invalid_scales:
            raise ValueError(
                f"Each scale must be a positive integer. Invalid scales: {invalid_scales}"
            )
    else:
        raise ValueError(
            "'scales' must be a list of positive integers,"
            f" 'auto', or None, but got type {type(scales).__name__}.")

    # Set return_sequences based on multi_scale_agg
    if multi_scale_agg is None or multi_scale_agg == 'auto':
        return_sequences = False
    else:
        # Optionally, you can validate multi_scale_agg against allowed methods
        allowed_aggregations = {'sum', 'concat', 'average'}  # Example allowed methods
        if not isinstance(multi_scale_agg, str):
            raise ValueError(
                f"'multi_scale_agg' must be a string indicating"
                " the aggregation method, 'auto', or None, "
                f"but got type {type(multi_scale_agg).__name__}."
            )
        if multi_scale_agg.lower() not in allowed_aggregations:
            raise ValueError(
                f"'multi_scale_agg' must be one of {allowed_aggregations}, "
                f"'auto', or None, but got '{multi_scale_agg}'."
            )
        return_sequences = True

    return quantiles, scales, return_sequences
