# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Contains loss functions used in the `gofast-nn` package for neural 
network models. The loss functions are designed to be compatible with Keras
and TensorFlow models. 

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
import warnings 
from numbers import Real 
from typing import List, Optional 

from ..compat.sklearn import Interval
from ..core.checks import ParamsValidator, check_params
from ..core.diagnose_q import validate_quantiles_in 
from ..utils.deps_utils import ensure_pkg
from ..utils.validator import check_consistent_length 
from . import KERAS_DEPS, KERAS_BACKEND, dependency_message
from .keras_validator import validate_keras_loss 

if KERAS_BACKEND:
    K = KERAS_DEPS.backend
    Loss=KERAS_DEPS.Loss
    Tensor=KERAS_DEPS.Tensor 
    reduce_mean=KERAS_DEPS.reduce_mean 
    square=KERAS_DEPS.square 
    reshape=KERAS_DEPS.reshape 
    convert_to_tensor=KERAS_DEPS.convert_to_tensor 
    expand_dims=KERAS_DEPS.expand_dims
    maximum=KERAS_DEPS.maximum
    reduce_mean=KERAS_DEPS.reduce_mean 
    rank=KERAS_DEPS.rank 
    register_keras_serializable=KERAS_DEPS.register_keras_serializable
    
    
DEP_MSG = dependency_message('nn.losses') 

__all__ = [
    'quantile_loss', 
    'quantile_loss_multi', 
    'anomaly_loss', 
    'combined_quantile_loss', 
    'combined_total_loss',
    'objective_loss', 
 ]

@ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
def objective_loss(
    multi_obj_loss: Loss,
    anomaly_scores: Optional[Tensor] = None,
):
    """
    Create a multi-objective Keras loss function that wraps
    a `MultiObjectiveLoss` layer, optionally including anomaly
    scores.

    Parameters
    ----------
    multi_obj_loss : MultiObjectiveLoss
        A `MultiObjectiveLoss` instance that combines
        quantile loss and anomaly loss. Typically you
        create it via:
            MultiObjectiveLoss(
                quantile_loss_fn=AdaptiveQuantileLoss(...),
                anomaly_loss_fn=AnomalyLoss(...)
            )
    anomaly_scores : tf.Tensor or None, optional
        Tensor of shape (B, H, D) representing anomaly scores.
        If None, anomaly loss is omitted. Defaults to None.

    Returns
    -------
    callable
        A function `loss_fn(y_true, y_pred) -> scalar`,
        suitable for `model.compile(loss=...)`.

    Notes
    -----
    This function is "Keras-serializable" in that you can
    save and load models using it.  Under the hood, it calls
    `multi_obj_loss(y_true, y_pred, anomaly_scores)`. If
    `anomaly_scores` is None, only the quantile loss is used.

    Examples
    --------
    >>> from gofast.nn.components import (
    ...    MultiObjectiveLoss, AdaptiveQuantileLoss, AnomalyLoss
    ... )
    >>> mo_loss = MultiObjectiveLoss(
    ...     quantile_loss_fn=AdaptiveQuantileLoss([0.1, 0.5, 0.9]),
    ...     anomaly_loss_fn=AnomalyLoss(weight=1.5)
    ... )
    >>> # Suppose anomaly_scores is some Tensor
    >>> anomaly_scores = tf.random.normal((32, 10, 8))
    >>> # Wrap everything as a single Keras loss function
    >>> loss_fn = objective_loss(
    ...     multi_obj_loss=mo_loss,
    ...     anomaly_scores=anomaly_scores
    ... )
    >>> # Now you can do:
    ... model.compile(optimizer="adam", loss=loss_fn)

    See Also
    --------
    gofast.nn.losses.MultiObjectiveLoss : 
        The layer combining quantile + anomaly losses.
    """
    from .components import MultiObjectiveLoss
    # Optional: check if multi_obj_loss has a 'call' method
    # or if it's a valid Keras layer. 
    multi_obj_loss= validate_keras_loss (
        multi_obj_loss, deep_check=True, 
        ops="validate", 
    )
    if not isinstance(multi_obj_loss, MultiObjectiveLoss):
        warnings.warn(
            "Expected a MultiObjectiveLoss instance, got %s" % type(
                multi_obj_loss)
    )

    @register_keras_serializable(
        package="gofast.nn.losses", name="objective_loss"
     )
    @ParamsValidator(
        {
            "y_true": ["array-like:tf:transf"],
            "y_pred": ["array-like:tf:transf"],
        }
    )
    def _loss_fn(y_true, y_pred):
        # If anomaly_scores is not None, we can do a length check:
        if anomaly_scores is not None:
            # Basic length check (optional)
            check_consistent_length(y_true, y_pred, anomaly_scores)

            # Actual call to multi_obj_layer
            return multi_obj_loss(y_true, y_pred, anomaly_scores)
        else:
            check_consistent_length(y_true, y_pred)
            return multi_obj_loss(y_true, y_pred)

    return _loss_fn

@ParamsValidator({'quantiles': [Real, 'array-like']})
@ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
def combined_quantile_loss(quantiles: List[float]):
    """
    Create a quantile loss function for multiple quantiles.

    This top-level function is decorated so that if you do:
        model.compile(loss=combined_quantile_loss([...]))
    Keras can serialize/deserialize it.

    Parameters
    ----------
    quantiles : list of float
        List of quantiles to compute.

    Returns
    -------
    callable
        A loss function: loss_fn(y_true, y_pred).

    Notes
    -----
    - We do not decorate the returned inner function
      with @register_keras_serializable. This avoids
      double registration conflicts.
    """
    # Validate & store quantiles
    quantiles = validate_quantiles_in(quantiles)
    
    @register_keras_serializable(
        "gofast.nn.losses", 
        name="combined_quantile_loss"
    )
    def _cqloss(y_true, y_pred):
        # Expand y_true so it matches y_pred's quantile dimension
        y_true_expanded = expand_dims(y_true, axis=2)  # => (B, H, 1, O)

        if rank(y_pred) == 3:
            # e.g. shape (B, H, Q) => expand last dim
            y_pred = expand_dims(y_pred, axis=-1)  # => (B, H, Q, 1)

        # Broadcast y_true_expanded to match y_pred's shape
        # shape => (B, H, Q, O)
        error = y_true_expanded - y_pred
        # Initialize loss
        loss_val = 0.0

        # Accumulate pinball losses for each quantile
        for i, q in enumerate(quantiles):
            q_loss = maximum(q * error[:, :, i, :],
                              (q - 1) * error[:, :, i, :])
            # Aggregate loss (mean over batch, horizons, and output_dim)
            loss_val += reduce_mean(q_loss)

        # Average loss over all quantiles
        return loss_val / len(quantiles)

    return _cqloss

@ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
def combined_total_loss(
    quantiles: List[float],
    anomaly_layer: Loss,   #  an instance of your AnomalyLoss
    anomaly_scores: Tensor, 
):
    """
    Create a total loss that adds quantile loss + anomaly loss.

    Like above, only this top-level is decorated. The returned
    function is not re-decorated.

    Parameters
    ----------
    quantiles : list of float
        Quantiles for the quantile loss part.
    anomaly_layer : tf.keras.losses.Loss
        A custom Loss or callable implementing anomaly loss.
    anomaly_scores : tf.Tensor or np.ndarray
        The anomaly scores needed for the anomaly loss.

    Returns
    -------
    callable
        A loss function: loss_fn(y_true, y_pred)
    """
    from .components import AnomalyLoss 
    # Re-use the same logic from combined_quantile_loss
    quantile_loss_fn = combined_quantile_loss(quantiles)
    # validate layer
    anomaly_layer =validate_keras_loss(
        anomaly_layer, ops="validate", 
    )
    if not isinstance(anomaly_layer, AnomalyLoss):
        warnings.warn(
            "Expected a AnomalyLoss instance, got %s" % type(
                anomaly_layer)
    )
    
    @register_keras_serializable(
        package="gofast.nn.losses", 
        name="combined_total_loss"
    )
    def _total_loss(y_true, y_pred):
        q_loss = quantile_loss_fn(y_true, y_pred)
        a_loss = anomaly_layer(anomaly_scores)
        return q_loss + a_loss

    return _total_loss

@check_params({"q": Real})
@ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
def quantile_loss(q):
    """
    Quantile (Pinball) Loss Function for Quantile Regression.
    
    The ``quantile_loss`` function computes the quantile loss, also known as
    Pinball loss, which is used in quantile regression to predict a specific
    quantile of the target variable's distribution. This loss function
    penalizes over-predictions and under-predictions differently based on
    the quantile parameter, allowing the model to estimate the desired
    quantile.
    
    .. math::
        L_q(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^{N} \rho_q(y_i -
        \hat{y}_i)
    
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
        The quantile to calculate the loss for. Must be a value between 0
        and 1. For example, ``q=0.1`` corresponds to the 10th percentile,
        ``q=0.5`` is the median, and ``q=0.9`` corresponds to the 90th
        percentile.
    
    Returns
    -------
    loss : callable
        A loss function that can be used in Keras models. This function
        takes two arguments, ``y_true`` and ``y_pred``, and returns the
        computed quantile loss.
    
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
        The quantile loss function is particularly useful in probabilistic
        forecasting where multiple quantiles are predicted to provide a
        distribution of possible outcomes rather than a single point
        estimate.
    
    - **Handling Multiple Quantiles**:
        To predict multiple quantiles, you can create separate output
        layers for each quantile and compile the model with a list of
        quantile loss functions.
    
    - **Gradient Computation**:
        The quantile loss function is differentiable, allowing it to be
        used seamlessly with gradient-based optimization algorithms in
        Keras.
    
    - **Robustness to Outliers**:
        Unlike Mean Squared Error (MSE), the quantile loss function is
        more robust to outliers, especially when predicting lower or
        higher quantiles.
    
    See Also
    --------
    tensorflow.keras.losses : A module containing built-in loss functions
        in Keras.
    sklearn.metrics.mean_pinball_loss : Computes the mean pinball loss,
        similar to quantile loss used here.
    statsmodels.regression.quantile_regression : Provides tools for
        quantile regression analysis.
    
    References
    ----------
    .. [1] Koenker, R., & Bassett Jr, G. (1978). Regression quantiles.
           *Econometrica*, 46(1), 33-50.
    .. [2] Taylor, J. W., Oosterlee, C. W., & Haggerty, K. (2008). A review
           of quantile regression in financial time series forecasting.
           *Applied Financial Economics*, 18(12), 955-967.
    .. [3] Koenker, R. (2005). Quantile Regression. *Cambridge University
           Press*.
    """
    @register_keras_serializable("gofast.nn.losses", name='quantile_loss')
    def _q_loss(y_true, y_pred):
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

    return _q_loss

@check_params ({
     'quantiles': List[float]}
  )
@ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
def quantile_loss_multi(quantiles=[0.1, 0.5, 0.9]):
    """
    Multi-Quantile (Pinball) Loss Function for Quantile Regression.
    
    The ``quantile_loss_multi`` function computes the average quantile loss
    across multiple quantiles, allowing for the simultaneous prediction of
    several quantiles of the target variable's distribution. This is
    particularly useful in probabilistic forecasting where a range of
    possible outcomes is desired.
    
    .. math::
        L_{\text{multi}}(Y, \hat{Y}) = \frac{1}{Q} \sum_{q \in
        \text{quantiles}} L_q(Y, \hat{Y})
    
    Where:
    - :math:`L_q(Y, \hat{Y})` is the quantile loss for a specific quantile
      :math:`q`.
    - :math:`Q` is the total number of quantiles.
    
    Each individual quantile loss is defined as:
    
    .. math::
        L_q(Y, \hat{Y}) = \frac{1}{N} \sum_{i=1}^{N} \rho_q(y_i -
        \hat{y}_i)
    
    And the pinball loss function :math:`\rho_q(u)` is:
    
    .. math::
        \rho_q(u) = u \cdot (q - \mathbb{I}(u < 0))
    
    Here, :math:`\mathbb{I}(u < 0)` is the indicator function that is 1 if
    :math:`u < 0` and 0 otherwise.
    
    Parameters
    ----------
    quantiles : list of float, default=[0.1, 0.5, 0.9]
        A list of quantiles to calculate the loss for. Each value must be
        between 0 and 1. For example, ``quantiles=[0.1, 0.5, 0.9]``
        corresponds to the 10th percentile, median, and 90th percentile
        respectively.
    
    Returns
    -------
    loss : callable
        A loss function that can be used in Keras models. This function
        takes two arguments, ``y_true`` and ``y_pred``, and returns the
        averaged quantile loss across the specified quantiles.
    
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
    >>> # Compile the model with multi-quantile loss for the 10th, 50th,
    >>> # and 90th percentiles
    >>> model.compile(optimizer='adam', loss=quantile_loss_multi(
    ...     quantiles=[0.1, 0.5, 0.9]))
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
        The multi-quantile loss function is essential for probabilistic
        forecasting, where multiple quantiles provide a comprehensive view
        of the possible outcomes rather than a single point estimate.
    
    - **Model Output Configuration**:
        When using multiple quantiles, ensure that the model's output
        layer is configured to output predictions for each quantile. For
        example, the output layer should have a number of units equal to
        the number of quantiles.
    
    - **Handling Multiple Quantiles in Predictions**:
        The model will output a separate prediction for each quantile.
        It is important to interpret these predictions correctly,
        understanding that each represents a specific percentile of the
        target distribution.
    
    - **Gradient Computation**:
        The quantile loss function is differentiable, allowing it to be
        used seamlessly with gradient-based optimization algorithms in
        Keras.
    
    - **Robustness to Outliers**:
        Unlike Mean Squared Error (MSE), the quantile loss function is
        more robust to outliers, especially when predicting lower or
        higher quantiles.
    
    See Also
    --------
    tensorflow.keras.losses : A module containing built-in loss functions
        in Keras.
    sklearn.metrics.mean_pinball_loss : Computes the mean pinball loss,
        similar to quantile loss used here.
    statsmodels.regression.quantile_regression : Provides tools for
        quantile regression analysis.
    
    References
    ----------
    .. [1] Koenker, R., & Bassett Jr, G. (1978). Regression quantiles.
           *Econometrica*, 46(1), 33-50.
    .. [2] Taylor, J. W., Oosterlee, C. W., & Haggerty, K. (2008). A review
           of quantile regression in financial time series forecasting.
           *Applied Financial Economics*, 18(12), 955-967.
    .. [3] Koenker, R. (2005). Quantile Regression. *Cambridge University
           Press*.
    """
    quantiles =validate_quantiles_in(quantiles)
    
    @register_keras_serializable("gofast.nn.losses", name="quantile_loss_multi")
    def _q_loss_multi(y_true, y_pred):
        """
        Compute the Multi-Quantile Loss (Averaged Pinball Loss) for a Given 
        Batch.

        This function calculates the quantile loss for each specified quantile 
        and returns the average loss across all quantiles. It is suitable 
        for models that predict multiple quantiles simultaneously.

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
    
    return _q_loss_multi

@ParamsValidator(
    { 
        'anomaly_scores': ['array-like:tf:transf'], 
        'anomaly_loss_weight': [Interval(Real, 0, None, closed ='neither')]
    }
)
@ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
def anomaly_loss(anomaly_scores, anomaly_loss_weight=1.0):
    """
    Compute the anomaly loss based on given anomaly scores and a 
    scaling weight. 
    
    The function returns a loss function callable that can be directly used 
    in Keras model compilation and  training workflows.
    The anomaly loss penalizes large anomaly scores, thereby guiding
    the model towards producing lower values when data points are considered 
    normal.
     
    Given anomaly scores 
    :math:`a = [a_1, a_2, ..., a_n]`, the anomaly loss 
    :math:`L` is defined as:
    
    .. math::
        L = w \cdot \frac{1}{n} \sum_{i=1}^{n} a_i^{2}
    
    where :math:`w` is the `anomaly_loss_weight`, and 
    :math:`n` is the number of data points.
    
    The model thus aims to reduce these anomaly scores, forcing 
    representations or intermediate outputs to behave more 
    normally according to its learned patterns.
    
    Parameters
    ----------
    anomaly_scores : tf.Tensor or array-like
        The anomaly scores reflecting the degree of abnormality in 
        data points. Higher values indicate more unusual points. 
        If provided as array-like, they will be converted into a 
        :class:`tf.Tensor` of type float32.  
    anomaly_loss_weight : float, optional
        A scaling factor controlling the influence of the anomaly 
        loss on the overall training objective. Default is 
        ``1.0``. Increasing this value places greater emphasis on 
        reducing anomaly scores, encouraging the model to learn 
        representations or predictions that minimize these values.
    
    Returns
    -------
    callable
        A callable loss function with signature 
        ``loss(y_true, y_pred)`` compatible with Keras. This 
        returned function ignores `y_true` and focuses only on 
        `anomaly_scores`, computing the mean of the squared 
        anomaly scores and scaling by ``anomaly_loss_weight``.
    
    Examples
    --------
    >>> from gofast.nn.losses import anomaly_loss
    >>> import tensorflow as tf
    >>> anomaly_scores = tf.constant([0.1, 0.5, 2.0], dtype=tf.float32)
    >>> loss_fn = anomaly_loss(anomaly_scores, anomaly_loss_weight=0.5)
    >>> y_true_dummy = tf.zeros_like(anomaly_scores)
    >>> y_pred_dummy = tf.zeros_like(anomaly_scores)
    >>> loss_value = loss_fn(y_true_dummy, y_pred_dummy)
    >>> print(loss_value.numpy())
    1.4166666
    
    In this example, the anomaly loss encourages the model to 
    reduce the given anomaly scores.
    
    Notes
    -----
    - The `y_true` and `y_pred` parameters are included for 
      compatibility with Keras losses but are not utilized 
      in the anomaly loss computation.
    - If `anomaly_scores` is provided as array-like, it is 
      converted to float32 for consistency. If it is already 
      a tensor, it is cast to float32 if needed.
    
    See Also
    --------
    :func:`tf.keras.losses.Loss` : Base class for all Keras losses.
    :func:`tf.reduce_mean` : TensorFlow method for computing mean.
    :func:`tf.square` : Squares tensor elements.
    
    References
    ----------
    .. [1] Goodfellow, Ian, et al. *Deep Learning.* MIT Press, 2016.
    """

    # if not isinstance(anomaly_scores, tf.Tensor):
    #     anomaly_scores = tf.convert_to_tensor(anomaly_scores, dtype=tf.float32)
    # else:
    #     if anomaly_scores.dtype not in (tf.float16, tf.float32, tf.float64):
    #         anomaly_scores = tf.cast(anomaly_scores, tf.float32)

    if anomaly_scores.shape.rank is None:
        anomaly_scores =reshape(anomaly_scores, [-1])

    anomaly_loss_weight =convert_to_tensor(
        anomaly_loss_weight, dtype=anomaly_scores.dtype
    )
    @register_keras_serializable(
        "gofast.nn.losses", name="anomaly_loss"
    )
    def _a_loss(y_true, y_pred):
        return anomaly_loss_weight * reduce_mean(square(anomaly_scores))

    return _a_loss
