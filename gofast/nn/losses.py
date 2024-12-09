# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Contains loss functions used in the `gofast.nn` package for neural 
network models, including custom implementations like quantile loss. The loss 
functions are designed to be compatible with Keras and TensorFlow models. 

Importantly, the module checks for required dependencies like Keras backend 
and ensures that necessary packages are available.

Functions:
----------
- quantile_loss: 
    Implements the Pinball (Quantile) loss for quantile regression tasks.

Dependencies:
------------
- KERAS_DEPS: Handles Keras dependency resolution.
- KERAS_BACKEND: Configures backend setup for Keras.

"""
from numbers import Real 
from sklearn.ensemble import IsolationForest
from typing import Callable, Optional, List
import numpy as np

from ..tools.depsutils import ensure_pkg
from ..compat.sklearn import ( 
    StrOptions, 
    Interval, 
    HasMethods, 
    Hidden, 
)
from ..core.checks import ParamsValidator, check_params 
from ..tools.validator import check_consistent_length , validate_quantiles
from . import KERAS_DEPS, KERAS_BACKEND, dependency_message

if KERAS_BACKEND:
    K = KERAS_DEPS.backend

DEP_MSG = dependency_message('loss') 

__all__ = ['quantile_loss', 'quantile_loss_multi', 'anomaly_scores']


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
def anomaly_scores(
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
    >>> from gofast.nn.losses import anomaly_scores
    >>> import numpy as np
    
    >>> # Statistical method example
    >>> y_true = np.random.randn(32, 20, 1)  # (B,H,O)
    >>> scores = anomaly_scores(y_true, method='statistical', threshold=3)
    >>> scores.shape
    (32, 20, 1)
    
    >>> # Domain-specific example
    >>> def my_heuristic(y):
    ...     return np.where(y < -1, np.abs(y)*5, 0.0)
    >>> scores = anomaly_scores(y_true, method='domain', domain_func=my_heuristic)
    
    >>> # Isolation Forest example
    >>> scores = anomaly_scores(y_true, method='isolation_forest', contamination=0.1)
    
    >>> # Residual-based example
    >>> y_pred = y_true + np.random.normal(0, 1, y_true.shape)  # Introduce noise
    >>> scores = anomaly_scores(y_true, y_pred=y_pred, method='residual', residual_metric='mae')
    
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

@check_params({"q": Real})
@ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
def quantile_loss(q):
    """
    Quantile (Pinball) Loss Function for Quantile Regression.

    The ``quantile_loss`` function computes the quantile loss, also known as
    Pinball loss, which is used in quantile regression to predict a specific
    quantile of the target variable's distribution. This loss function
    penalizes over-predictions and under-predictions differently based on the
    quantile parameter, allowing the model to estimate the desired quantile.

    .. math::
        L_q(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^{N} \rho_q(y_i - \hat{y}_i)

    Where:
    - :math:`y_i` is the true value.
    - :math:`\hat{y}_i` is the predicted value.
    - :math:`\rho_q(u)` is the quantile loss function defined as:

    .. math::
        \rho_q(u) = u \cdot (q - \mathbb{I}(u < 0))

    Here, :math:`\mathbb{I}(u < 0)` is the indicator function that is 1
    if :math:`u < 0` and 0 otherwise.

    Parameters
    ----------
    q : float
        The quantile to calculate the loss for. Must be a value between 0 and 1.
        For example, ``q=0.1`` corresponds to the 10th percentile, ``q=0.5``
         is the median, and ``q=0.9`` corresponds to the 90th percentile.

    Returns
    -------
    loss : callable
        A loss function that can be used in Keras models. This function takes
        two arguments, ``y_true`` and ``y_pred``, and returns the computed
        quantile loss.

    Examples
    --------
    >>> from gofast.nn.losses import quantile_loss
    >>> import tensorflow as tf
    >>> from tensorflow.keras.models import Sequential
    >>> from tensorflow.keras.layers import Dense
    >>> import numpy as np
    >>> 
    >>> # Create a simple Keras model
    >>> model = Sequential()
    >>> model.add(Dense(64, input_dim=10, activation='relu'))
    >>> model.add(Dense(1))
    >>> 
    >>> # Compile the model with quantile loss for the 10th percentile
    >>> model.compile(optimizer='adam', loss=quantile_loss(q=0.1))
    >>> 
    >>> # Generate example data
    >>> X_train = np.random.rand(100, 10)
    >>> y_train = np.random.rand(100, 1)
    >>> 
    >>> # Train the model
    >>> model.fit(X_train, y_train, epochs=10, batch_size=32)
    
    Notes
    -----
    - **Usage in Probabilistic Forecasting**:
        The quantile loss function is particularly useful in probabilistic forecasting
        where multiple quantiles are predicted to provide a distribution of possible
        outcomes rather than a single point estimate.

    - **Handling Multiple Quantiles**:
        To predict multiple quantiles, you can create separate output layers
        for each quantile and compile the model with a list of quantile loss
        functions.

    - **Gradient Computation**:
        The quantile loss function is differentiable, allowing it to be used
        seamlessly with gradient-based optimization algorithms in Keras.

    - **Robustness to Outliers**:
        Unlike Mean Squared Error (MSE), the quantile loss function is more
        robust to outliers, especially when predicting lower or higher quantiles.

    See Also
    --------
    tensorflow.keras.losses : A module containing built-in loss functions in Keras.
    sklearn.metrics.mean_pinball_loss : Computes the mean pinball loss, similar to
        quantile loss used here.
    statsmodels.regression.quantile_regression : Provides tools for quantile
        regression analysis.

    References
    ----------
    .. [1] Koenker, R., & Bassett Jr, G. (1978). Regression quantiles. *Econometrica*,
           46(1), 33-50.
    .. [2] Taylor, J. W., Oosterlee, C. W., & Haggerty, K. (2008). A review of quantile
           regression in financial time series forecasting. *Applied Financial Economics*,
           18(12), 955-967.
    .. [3] Koenker, R. (2005). Quantile Regression. *Cambridge University Press*.

    """

    def loss(y_true, y_pred):
        """
        Compute the Quantile Loss (Pinball Loss) for a Given Batch.

        The loss is defined as:

        .. math::
            L_q(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^{N} \rho_q(y_i - \hat{y}_i)

        Where:

        .. math::
            \rho_q(u) = u \cdot (q - \mathbb{I}(u < 0))

        Parameters
        ----------
        y_true : Tensor
            The ground truth values. Shape: ``(batch_size, ...)``.
        y_pred : Tensor
            The predicted values by the model. Shape: ``(batch_size, ...)``.

        Returns
        -------
        loss : Tensor
            The quantile loss value averaged over the batch.
        """
        error = y_true - y_pred
        loss = K.mean(
            K.maximum(q * error, (q - 1) * error),
            axis=-1
        )
        return loss

    return loss

@check_params (
    {
     'quantiles': List[float]
     }
  )
@ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
def quantile_loss_multi(quantiles=[0.1, 0.5, 0.9]):
    """
    Multi-Quantile (Pinball) Loss Function for Quantile Regression.

    The ``quantile_loss_multi`` function computes the average quantile loss across
    multiple quantiles, allowing for the simultaneous prediction of several quantiles
    of the target variable's distribution. This is particularly useful in probabilistic
    forecasting where a range of possible outcomes is desired.

    .. math::
        L_{\text{multi}}(Y, \hat{Y}) = \frac{1}{Q} \sum_{q \in \text{quantiles}} L_q(Y, \hat{Y})

    Where:
    - :math:`L_q(Y, \hat{Y})` is the quantile loss for a specific quantile :math:`q`.
    - :math:`Q` is the total number of quantiles.

    Each individual quantile loss is defined as:

    .. math::
        L_q(Y, \hat{Y}) = \frac{1}{N} \sum_{i=1}^{N} \rho_q(y_i - \hat{y}_i)

    And the pinball loss function :math:`\rho_q(u)` is:

    .. math::
        \rho_q(u) = u \cdot (q - \mathbb{I}(u < 0))

    Here, :math:`\mathbb{I}(u < 0)` is the indicator function that is 1 if
    :math:`u < 0` and 0 otherwise.

    Parameters
    ----------
    quantiles : list of float, default=[0.1, 0.5, 0.9]
        A list of quantiles to calculate the loss for. Each value must be between 0 and 1.
        For example, ``quantiles=[0.1, 0.5, 0.9]`` corresponds to the 10th percentile,
        median, and 90th percentile respectively.

    Returns
    -------
    loss : callable
        A loss function that can be used in Keras models. This function takes
        two arguments, ``y_true`` and ``y_pred``, and returns the averaged
        quantile loss across the specified quantiles.

    Examples
    --------
    >>> from gofast.nn.loss import quantile_loss_multi
    >>> import tensorflow as tf
    >>> from tensorflow.keras.models import Sequential
    >>> from tensorflow.keras.layers import Dense
    >>> import numpy as np
    >>> 
    >>> # Create a simple Keras model
    >>> model = Sequential()
    >>> model.add(Dense(64, input_dim=10, activation='relu'))
    >>> model.add(Dense(1))
    >>> 
    >>> # Compile the model with multi-quantile loss for the 10th, 50th, and 90th percentiles
    >>> model.compile(optimizer='adam', loss=quantile_loss_multi(quantiles=[0.1, 0.5, 0.9]))
    >>> 
    >>> # Generate example data
    >>> X_train = np.random.rand(100, 10)
    >>> y_train = np.random.rand(100, 1)
    >>> 
    >>> # Train the model
    >>> model.fit(X_train, y_train, epochs=10, batch_size=32)
    
    Notes
    -----
    - **Probabilistic Forecasting**:
        The multi-quantile loss function is essential for probabilistic forecasting,
        where multiple quantiles provide a comprehensive view of the possible outcomes
        rather than a single point estimate.
    
    - **Model Output Configuration**:
        When using multiple quantiles, ensure that the model's output layer is
        configured to output predictions for each quantile. For example, the output
        layer should have a number of units equal to the number of quantiles.
    
    - **Handling Multiple Quantiles in Predictions**:
        The model will output a separate prediction for each quantile. It is important
        to interpret these predictions correctly, understanding that each represents a
        specific percentile of the target distribution.
    
    - **Gradient Computation**:
        The quantile loss function is differentiable, allowing it to be used seamlessly
        with gradient-based optimization algorithms in Keras.
    
    - **Robustness to Outliers**:
        Unlike Mean Squared Error (MSE), the quantile loss function is more robust to
        outliers, especially when predicting lower or higher quantiles.
    
    See Also
    --------
    tensorflow.keras.losses : A module containing built-in loss functions in Keras.
    sklearn.metrics.mean_pinball_loss : Computes the mean pinball loss, similar to
        quantile loss used here.
    statsmodels.regression.quantile_regression : Provides tools for quantile
        regression analysis.
    
    References
    ----------
    .. [1] Koenker, R., & Bassett Jr, G. (1978). Regression quantiles. *Econometrica*,
           46(1), 33-50.
    .. [2] Taylor, J. W., Oosterlee, C. W., & Haggerty, K. (2008). A review of quantile
           regression in financial time series forecasting. *Applied Financial Economics*,
           18(12), 955-967.
    .. [3] Koenker, R. (2005). Quantile Regression. *Cambridge University Press*.
    
    """
    def loss(y_true, y_pred):
        """
        Compute the Multi-Quantile Loss (Averaged Pinball Loss) for a Given Batch.

        This function calculates the quantile loss for each specified quantile and
        returns the average loss across all quantiles. It is suitable for models
        that predict multiple quantiles simultaneously.

        Parameters
        ----------
        y_true : Tensor
            The ground truth values. Shape: ``(batch_size, ...)``.
        y_pred : Tensor
            The predicted values by the model. Shape: ``(batch_size, ...)``.

        Returns
        -------
        loss : Tensor
            The averaged quantile loss across all specified quantiles.
        """
        losses = []
        for q in quantiles:
            error = y_true - y_pred
            loss_q = K.mean(K.maximum(q * error, (q - 1) * error), axis=-1)
            losses.append(loss_q)
        
        # Stack the losses for each quantile and compute the mean
        loss_stack = K.stack(losses, axis=0)
        loss_mean = K.mean(loss_stack, axis=0)
        
        return loss_mean
    
    quantiles =validate_quantiles(quantiles)
    
    return loss
