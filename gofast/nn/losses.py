# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
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

from ..tools.depsutils import ensure_pkg
from . import KERAS_DEPS, KERAS_BACKEND, dependency_message

if KERAS_BACKEND:
    K = KERAS_DEPS.backend

DEP_MSG = dependency_message('loss') 

__all__ = ['quantile_loss', 'quantile_loss_multi']


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
        For example, ``q=0.1`` corresponds to the 10th percentile, ``q=0.5`` is the
        median, and ``q=0.9`` corresponds to the 90th percentile.

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

    return loss
