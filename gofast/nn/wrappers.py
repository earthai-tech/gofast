# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Utility wrappers for integrating neural networks models with 
scikit-learn estimators.

This module provides wrapper functions to facilitate the creation and 
compatibility of Temporal Fusion Transformer (TFT) models with scikit-learn 
estimators, such as `KerasRegressor`. These wrappers enable seamless integration 
of TFT models into scikit-learn workflows, including hyperparameter tuning 
and cross-validation.
"""

from textwrap import dedent 
from numbers import Real, Integral 
from typing import Optional, List, Union
from sklearn.base import BaseEstimator, RegressorMixin

import numpy as np 

from ..api.docs import doc , _shared_docs 
from ..compat.sklearn import validate_params, Interval, StrOptions
from ..decorators import Appender
from ..exceptions import NotFittedError 
from ..tools.depsutils import ensure_pkg 

from . import KERAS_DEPS, KERAS_BACKEND, dependency_message
from .transformers import TemporalFusionTransformer 

if KERAS_BACKEND:
    Optimizer=KERAS_DEPS.Optimizer
    Metric=KERAS_DEPS.Metric
    Loss=KERAS_DEPS.Loss
    Model=KERAS_DEPS.Model 
    LSTM = KERAS_DEPS.LSTM
    Dense = KERAS_DEPS.Dense
    Adam= KERAS_DEPS.Adam
    Dropout = KERAS_DEPS.Dropout 
    LayerNormalization = KERAS_DEPS.LayerNormalization 
    MultiHeadAttention = KERAS_DEPS.MultiHeadAttention
    Input = KERAS_DEPS.Input
    Concatenate=KERAS_DEPS.Concatenate 
    

DEP_MSG = dependency_message('wrappers') 

__all__ = ['TFTWrapper', 'TFTRegressor', "create_tft_model"]


@ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
@Appender ( dedent( 
"""
Notes
-----
- **Data Preparation**:
    - The ``TFTRegressor`` expects the input data ``X`` to be a single 2D array where
      the static and dynamic features are concatenated. The expected shape of ``X`` is:
      
      .. math::
          X = 
          \begin{bmatrix}
          X_{\text{static}} & X_{\text{dynamic}}
          \end{bmatrix}
      
      Where:
      - :math:`X_{\text{static}}` has shape :math:`(n_{\text{samples}}, 
        \text{num_static_vars} \times \text{static_input_dim})`
      - :math:`X_{\text{dynamic}}` has shape :math:`(n_{\text{samples}}, 
        \text{forecast_horizon} \times \text{num_dynamic_vars} \times 
        \text{dynamic_input_dim})`
    
    - Ensure that the concatenation order matches the expected split in the
      ``_split_X`` method: static features first, followed by dynamic features.

- **Model Training**:
    - The ``fit`` method trains the underlying Keras model using default parameters.
      For customized training (e.g., different number of epochs, batch size),
      consider modifying the ``build_model`` method or extending the wrapper.

- **Hyperparameter Tuning**:
    - Hyperparameter tuning with ``RandomizedSearchCV`` may be computationally intensive.
      Adjust ``n_iter`` and ``cv`` parameters based on available computational
      resources and the size of the parameter grid.

- **Model Limitations**:
    - The current implementation of ``TFTRegressor`` does not support multi-output
      predictions. For multi-output forecasting, additional modifications to the
      wrapper and underlying model are required.

- **Verbose Levels**:
    - The ``verbose`` parameter controls the level of debug information:
      
      - `0`: No output.
      - `1`: Minimal output (e.g., starting messages).
      - `2`: Intermediate output (e.g., model building steps).
      - `3`: Detailed output (e.g., layer-wise information).
      - `4`: Extensive debugging information.
      - `5`: Highly detailed logs for model construction.
      - `6`: Verbose output for input shapes and transformations.
      - `7`: Maximum verbosity with comprehensive debug information.
    
    - Adjust the ``verbose`` level based on the need for debugging or monitoring
      the training and prediction processes.

See Also
--------
sklearn.model_selection.RandomizedSearchCV : Randomized hyperparameter search.
sklearn.model_selection.GridSearchCV : Exhaustive grid hyperparameter search.
tensorflow.keras.Model : TensorFlow Keras Model API.
sklearn.base.BaseEstimator : Base class for all estimators in Scikit-Learn.
sklearn.base.RegressorMixin : Mixin class for all regressors in Scikit-Learn.
gofast.nn.transformers.TemporalFusionTransformer: 
    Gofast Temporal Fusion Transformer Model API.

References
----------
.. [1] Lim, B., & Zohdy, M. A. (2019). Temporal Fusion Transformers for interpretable
   multi-horizon time series forecasting. *International Journal of Forecasting*.
.. [2] McKinney, W. (2017). *Python for Data Analysis: Data Wrangling with Pandas,
   NumPy, and IPython*. O'Reilly Media.
.. [3] Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is All You Need.
   *Advances in Neural Information Processing Systems*, 30.

""" ), 
 join='\n', 
)
@doc(math_f= dedent ( 
    """
    .. math::
        Y = \text{TFT}(X_{\text{static}}, X_{\text{dynamic}})

    Where:
    - :math:`X_{\text{static}}` represents the static input features.
    - :math:`X_{\text{dynamic}}` represents the dynamic input features.
    - :math:`Y` is the predicted output.
    """
    ),
    tft_params=dedent(
        _shared_docs['tft_params_doc']
        )
 )
class TFTRegressor(BaseEstimator, RegressorMixin):
    """
    A Scikit-Learn Compatible Regressor Wrapper for the Temporal Fusion Transformer.

    The ``TFTRegressor`` class provides seamless integration of the Temporal Fusion
    Transformer (TFT) model with Scikit-Learn's API, enabling the use of Scikit-Learn's
    model selection tools such as ``RandomizedSearchCV`` and ``GridSearchCV`` for
    hyperparameter tuning. This wrapper adheres to Scikit-Learn's estimator
    requirements, facilitating easy training, prediction, and evaluation within
    Scikit-Learn workflows.

    {mathf}

    {tft_params}

    verbose : int, default=0
        Controls the level of verbosity for debug information.
        
        - `0`: No output.
        - `1`: Minimal output (e.g., starting messages).
        - `2`: Intermediate output (e.g., model building steps).
        - `3`: Detailed output (e.g., layer-wise information).
        - `4`: Extensive debugging information.
        - `5`: Highly detailed logs for model construction.
        - `6`: Verbose output for input shapes and transformations.
        - `7`: Maximum verbosity with comprehensive debug information.

    Attributes
    ----------
    model_ : tensorflow.keras.Model
        The underlying TensorFlow Keras model after fitting.

    Methods
    -------
    build_model()
        Constructs and compiles the TensorFlow Keras model based on the initialized
        parameters.

    fit(X, y)
        Trains the TFT model on the provided data.

    predict(X)
        Generates predictions using the trained TFT model.

    See Also
    --------
    sklearn.model_selection.RandomizedSearchCV : Randomized hyperparameter search.
    sklearn.model_selection.GridSearchCV : Exhaustive grid hyperparameter search.
    tensorflow.keras.Model : TensorFlow Keras Model API.
    sklearn.base.BaseEstimator : Base class for all estimators in Scikit-Learn.
    sklearn.base.RegressorMixin : Mixin class for all regressors in Scikit-Learn.

    References
    ----------
    .. [1] Lim, B., & Zohdy, M. A. (2019). Temporal Fusion Transformers for interpretable
       multi-horizon time series forecasting. *International Journal of Forecasting*.
    .. [2] McKinney, W. (2017). *Python for Data Analysis: Data Wrangling with Pandas,
       NumPy, and IPython*. O'Reilly Media.
    .. [3] Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is All You Need.
       *Advances in Neural Information Processing Systems*, 30.

    Examples
    --------
    >>> from gofast.nn.wrappers import TFTRegressor
    >>> import numpy as np
    >>> from sklearn.model_selection import RandomizedSearchCV
    >>> 
    >>> # Sample data
    >>> X_static = np.random.rand(100, 5, 10)  # 100 samples, 5 static vars, 10 dims each
    >>> # 100 samples, 10 time steps, 3 dynamic vars, 15 dims each
    >>> X_dynamic = np.random.rand(100, 10, 3, 15)  
    >>> y = np.random.rand(100, 1)  # 100 samples, 1 target
    >>> 
    >>> # Concatenate static and dynamic features for Scikit-Learn compatibility
    >>> X = np.concatenate([
    ...     X_static.reshape(100, -1), 
    ...     X_dynamic.reshape(100, -1)
    ... ], axis=1)
    >>> 
    >>> # Initialize the wrapper
    >>> tft = TFTRegressor(
    ...     static_input_dim=10,
    ...     dynamic_input_dim=15,
    ...     num_static_vars=5,
    ...     num_dynamic_vars=3,
    ...     hidden_units=64,
    ...     num_heads=4,
    ...     dropout_rate=0.1,
    ...     forecast_horizon=1,
    ...     quantiles=[0.1, 0.5, 0.9],
    ...     activation='elu',
    ...     use_batch_norm=True,
    ...     num_lstm_layers=2,
    ...     lstm_units=128,
    ...     verbose=3
    ... )
    >>> 
    >>> # Define parameter distributions for RandomizedSearchCV
    >>> param_distributions = {
    ...     'hidden_units': [64, 128],
    ...     'num_heads': [4, 8],
    ...     'dropout_rate': [0.1, 0.2],
    ...     'forecast_horizon': [1, 3],
    ...     'num_lstm_layers': [1, 2],
    ...     'lstm_units': [64, 128]
    ... }
    >>> 
    >>> # Initialize RandomizedSearchCV
    >>> random_search = RandomizedSearchCV(
    ...     estimator=tft,
    ...     param_distributions=param_distributions,
    ...     n_iter=10,
    ...     cv=3,
    ...     verbose=2,
    ...     random_state=42
    ... )
    >>> 
    >>> # Fit RandomizedSearchCV
    >>> random_search.fit(X=X, y=y)
    >>> 
    >>> # Access best parameters and estimator
    >>> best_params = random_search.best_params_
    >>> best_estimator = random_search.best_estimator_
    >>> print(f"Best Parameters: {best_params}")
    Best Parameters: {'hidden_units': 128, 'num_heads': 8, 'dropout_rate': 0.2,
                      'forecast_horizon': 3, 'num_lstm_layers': 2, 'lstm_units': 128}
    >>> print(f"Best Estimator: {best_estimator}")
    Best Estimator: TFTRegressor(...)
    
    """
    
    @validate_params({
        "static_input_dim": [Interval(Integral, 1, None, closed='left')], 
        "dynamic_input_dim": [Interval(Integral, 1, None, closed='left')], 
        "num_static_vars": [Interval(Integral, 1, None, closed='left')], 
        "num_dynamic_vars": [Interval(Integral, 1, None, closed='left')],
        "hidden_units": [Interval(Integral, 1, None, closed='left')], 
        "num_heads": [Interval(Integral, 1, None, closed='left')],
        "dropout_rate": [Interval(Real, 0, 1, closed="both")],
        "forecast_horizon": [Interval(Integral, 1, None, closed='left')],
        "quantiles": ['array-like', list,  None],
        "activation": [
            StrOptions({"elu", "relu", "tanh", "sigmoid", "linear", 'gelu'})],
        "use_batch_norm": [bool],
        "num_lstm_layers": [Interval(Integral, 1, None, closed='left')],
        "lstm_units": [list, Interval(Integral, 1, None, closed='left'), None]
        },
    )
    def __init__(
        self,
        static_input_dim,
        dynamic_input_dim,
        num_static_vars,
        num_dynamic_vars,
        hidden_units,
        num_heads=4,  
        dropout_rate=0.1,
        forecast_horizon=1,
        quantiles=None,
        activation='elu',
        use_batch_norm=False,
        num_lstm_layers=1,
        lstm_units=None, 
        verbose=0 
    ):
        self.static_input_dim    = static_input_dim
        self.dynamic_input_dim   = dynamic_input_dim
        self.num_static_vars     = num_static_vars
        self.num_dynamic_vars    = num_dynamic_vars
        self.hidden_units        = hidden_units
        self.num_heads           = num_heads
        self.dropout_rate        = dropout_rate
        self.forecast_horizon    = forecast_horizon
        self.quantiles           = quantiles
        self.activation          = activation
        self.use_batch_norm      = use_batch_norm
        self.num_lstm_layers     = num_lstm_layers
        self.lstm_units          = lstm_units
        self.verbose             = verbose

    def build_model(self):
        if self.verbose >= 5:
            print("Building the TFT model...")
        
        static_input = Input(
            shape=(self.num_static_vars, self.static_input_dim),
            name='static_input'
        )
        if self.verbose >= 6:
            print(f"Static input shape: {static_input.shape}")
        
        dynamic_input = Input(
            shape=(self.forecast_horizon, 
                   self.num_dynamic_vars, 
                   self.dynamic_input_dim),
            name='dynamic_input'
        )
        if self.verbose >= 6:
            print(f"Dynamic input shape: {dynamic_input.shape}")
        
        x = dynamic_input
        for layer_num in range(1, self.num_lstm_layers + 1):
            if self.verbose >= 5:
                print(f"Adding LSTM layer {layer_num}...")
            x = LSTM(
                self.lstm_units or self.hidden_units,
                return_sequences=True,
                activation=self.activation,
                name=f'lstm_{layer_num}'
            )(x)
            if self.use_batch_norm:
                if self.verbose >= 5:
                    print(f"Applying LayerNormalization after LSTM layer {layer_num}...")
                x = LayerNormalization(name=f'layer_norm_{layer_num}')(x)
            if self.verbose >= 5:
                print(f"Applying Dropout after LSTM layer {layer_num}...")
            x = Dropout(self.dropout_rate, name=f'dropout_{layer_num}')(x)
            if self.verbose >= 6:
                print(f"LSTM layer {layer_num} output shape: {x.shape}")
        
        if self.verbose >= 5:
            print("Adding Multi-Head Attention layer...")
        attention = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.hidden_units,
            dropout=self.dropout_rate,
            name='multi_head_attention'
        )(x, x)
        if self.verbose >= 6:
            print(f"Attention output shape: {attention.shape}")
        
        if self.verbose >= 5:
            print("Applying Dropout after Attention...")
        attention = Dropout(self.dropout_rate, name='attention_dropout')(attention)
        
        if self.use_batch_norm:
            if self.verbose >= 5:
                print("Applying LayerNormalization after Attention...")
            attention = LayerNormalization(name='attention_layer_norm')(attention)
        
        if self.verbose >= 5:
            print("Concatenating static inputs with attention output...")
        concatenated = Concatenate(axis=-1, name='concatenate')(
            [static_input, attention]
        )
        if self.verbose >= 6:
            print(f"Concatenated output shape: {concatenated.shape}")
        
        if self.verbose >= 5:
            print("Adding Dense output layer...")
        output = Dense(
            self.static_input_dim,
            activation='linear',
            name='output_dense'
        )(concatenated)
        if self.verbose >= 6:
            print(f"Output layer shape: {output.shape}")
        
        model = Model(
            inputs=[static_input, dynamic_input],
            outputs=output,
            name='TFT_Model'
        )
        
        if self.verbose >= 5:
            print("Compiling the model with Adam optimizer and MSE loss...")
        model.compile(
            optimizer=Adam(),
            loss='mse'
        )
        
        if self.verbose >= 5:
            print("Model built successfully.")
        
        return model

    def fit(self, X, y):
        """
        Fit the TFT model according to the given training data.
        """
        if self.verbose >= 3:
            print("Starting fit process...")
        
        if isinstance(X, (list, tuple)):
            if self.verbose >= 4:
                print("Concatenating static and dynamic inputs for compatibility...")
            try:
                X = np.concatenate(
                    [x.reshape(X[0].shape[0], -1) for x in X], axis=1
                )
                if self.verbose >= 6:
                    print(f"Concatenated X shape: {X.shape}")
            except Exception as e:
                if self.verbose >= 1:
                    print(f"Error concatenating inputs: {e}")
                raise ValueError("Error concatenating X inputs.") from e
        elif isinstance(X, np.ndarray):
            if self.verbose >= 6:
                print(f"Input X shape: {X.shape}")
        else:
            raise ValueError("Input X must be a list, tuple, or numpy array.")
        
        if self.verbose >= 4:
            print("Splitting X into static and dynamic components...")
        X_static, X_dynamic = self._split_X(X)
        
        if self.verbose >= 5:
            print(f"X_static shape: {X_static.shape}")
            print(f"X_dynamic shape: {X_dynamic.shape}")
        
        y = y.reshape(
            -1, 
            self.forecast_horizon, 
            self.static_input_dim
        )
        if self.verbose >= 6:
            print(f"Reshaped y shape: {y.shape}")
        
        if self.verbose >= 3:
            print("Building the model...")
        self.model_ = self.build_model()
        
        if self.verbose >= 3:
            print("Starting model training...")
        self.model_.fit(
            [X_static, X_dynamic],
            y,
            epochs=10,
            batch_size=32,
            verbose=self.verbose >= 4  # Show progress bar if verbose >=4
        )
        if self.verbose >= 3:
            print("Model training completed.")
        
        return self

    def predict(self, X):
        """
        Predict using the TFT model.
        """
        if self.verbose >= 3:
            print("Starting prediction process...")
        
        if self.model_ is None:
            raise NotFittedError(
                "This TFTRegressor instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this estimator."
            )
        
        if isinstance(X, (list, tuple)):
            if self.verbose >= 4:
                print("Concatenating static and dynamic inputs for compatibility...")
            try:
                X = np.concatenate(
                    [x.reshape(X[0].shape[0], -1) for x in X], axis=1
                )
                if self.verbose >= 6:
                    print(f"Concatenated X shape: {X.shape}")
            except Exception as e:
                if self.verbose >= 1:
                    print(f"Error concatenating inputs: {e}")
                raise ValueError("Error concatenating X inputs.") from e
        elif isinstance(X, np.ndarray):
            if self.verbose >= 6:
                print(f"Input X shape: {X.shape}")
        else:
            raise ValueError("Input X must be a list, tuple, or numpy array.")
        
        if self.verbose >= 4:
            print("Splitting X into static and dynamic components...")
        X_static, X_dynamic = self._split_X(X)
        
        if self.verbose >= 5:
            print(f"X_static shape: {X_static.shape}")
            print(f"X_dynamic shape: {X_dynamic.shape}")
        
        if self.verbose >= 3:
            print("Generating predictions...")
        predictions = self.model_.predict(
            [X_static, X_dynamic],
            batch_size=32,
            verbose=self.verbose >= 4  # Show progress bar if verbose >=4
        )
        
        if self.verbose >= 3:
            print("Prediction completed.")
        
        return predictions

    def _split_X(self, X):
        """
        Split the concatenated X into static and dynamic components.
        """
        static_size = self.num_static_vars * self.static_input_dim
        dynamic_size = self.forecast_horizon * self.num_dynamic_vars * self.dynamic_input_dim
        total_size = static_size + dynamic_size
        
        if self.verbose >= 6:
            print(f"Expected total features: {total_size}, got: {X.shape[1]}")
        
        if X.shape[1] != total_size:
            message = (
                f"Expected X to have {total_size} features, "
                f"got {X.shape[1]}."
            )
            if self.verbose >= 1:
                print(message)
            raise ValueError(message)
        
        if self.verbose >= 6:
            print("Splitting X into static and dynamic parts...")
        
        X_static = X[:, :static_size]
        X_static = X_static.reshape(
            (X_static.shape[0], 
             self.num_static_vars, 
             self.static_input_dim)
        )
        if self.verbose >= 6:
            print(f"Reshaped X_static to {X_static.shape}")
        
        X_dynamic = X[:, static_size:]
        X_dynamic = X_dynamic.reshape(
            (X_dynamic.shape[0], 
             self.forecast_horizon, 
             self.num_dynamic_vars, 
             self.dynamic_input_dim)
        )
        if self.verbose >= 6:
            print(f"Reshaped X_dynamic to {X_dynamic.shape}")
        
        return X_static, X_dynamic

@ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
@doc(tft_params=dedent(_shared_docs['tft_params_doc']))
class TFTWrapper(BaseEstimator, RegressorMixin):
    """
    TFTWrapper: A Scikit-Learn Compatible Wrapper for the Temporal Fusion
    Transformer.

    This class provides a seamless integration of the Temporal Fusion 
    Transformer (TFT) model with Scikit-Learn's API, enabling the use of 
    Scikit-Learn's model selection tools such as `RandomizedSearchCV` and 
    `GridSearchCV` for hyperparameter tuning. The wrapper ensures compatibility
    by adhering to Scikit-Learn's estimator requirements, facilitating easy 
    training, prediction, and evaluation within Scikit-Learn workflows.
    
    {tft_params}
    
    optimizer : `Union[str, tf.keras.optimizers.Optimizer]`, default=`'adam'`
        Optimizer to use for compiling the model. Can be a string identifier 
        or an instance of a Keras Optimizer.
    
    loss : `Union[str, tf.keras.losses.Loss]`, default=`'mse'`
        Loss function to use for training the model. Can be a string identifier 
        or an instance of a Keras Loss.
    
    metrics : `Optional[List[Union[str, tf.keras.metrics.Metric]]]`, default=`None`
        List of metrics to be evaluated by the model during training and testing.
        If `None`, no additional metrics are used.
        
    Attributes
    ----------
    model_ : tensorflow.keras.Model
        The underlying TensorFlow Keras model after fitting.
    
    Methods
    -------
    build_model()
        Constructs and compiles the TensorFlow Keras model based on the initialized
        parameters.
    
    fit(X, y)
        Trains the TFT model on the provided data.
    
    predict(X)
        Generates predictions using the trained TFT model.
    
    See Also
    --------
    sklearn.model_selection.RandomizedSearchCV : Randomized hyperparameter search.
    sklearn.model_selection.GridSearchCV : Exhaustive grid hyperparameter search.
    tensorflow.keras.Model : TensorFlow Keras Model API.
    
    References
    ----------
    .. [1] Lim, B., & Zohdy, M. A. (2019). Temporal Fusion Transformers for interpretable
       multi-horizon time series forecasting. *International Journal of Forecasting*.
    .. [2] McKinney, W. (2017). *Python for Data Analysis: Data Wrangling with Pandas,
       NumPy, and IPython*. O'Reilly Media.
    .. [3] Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is All You Need.
       *Advances in Neural Information Processing Systems*, 30.
    
    Examples
    --------
    >>> from gofast.nn.wrappers import TFTWrapper
    >>> import numpy as np
    >>> 
    >>> # Sample data
    >>> X_static = np.random.rand(100, 5, 10)  # 100 samples, 5 static vars, 10 dims each
    >>> # 100 samples, 10 time steps, 3 dynamic vars, 15 dims each
    >>> X_dynamic = np.random.rand(100, 10, 3, 15)  
    >>> y = np.random.rand(100, 1)  # 100 samples, 1 target
    >>> 
    >>> # Initialize the wrapper
    >>> tft = TFTWrapper(
    ...     static_input_dim=10,
    ...     dynamic_input_dim=15,
    ...     num_static_vars=5,
    ...     num_dynamic_vars=3,
    ...     hidden_units=64,
    ...     num_heads=4,
    ...     dropout_rate=0.1,
    ...     forecast_horizon=1,
    ...     quantiles=[0.1, 0.5, 0.9],
    ...     activation='elu',
    ...     use_batch_norm=True,
    ...     num_lstm_layers=2,
    ...     lstm_units=128
    ... )
    >>> 
    >>> # Fit the model
    >>> tft.fit(X=[X_static, X_dynamic], y=y)
    >>> 
    >>> # Make predictions
    >>> predictions = tft.predict(X=[X_static, X_dynamic])
    >>> print(predictions.shape)
    (100, 1)
    
    Notes
    -----
    - The `TFTWrapper` assumes that the input data `X` is a list containing two
      elements: `[X_static, X_dynamic]`. Ensure that the data is preprocessed to
      match the expected shapes:
      
      - `X_static`: (n_samples, num_static_vars, static_input_dim)
      - `X_dynamic`: (n_samples, forecast_horizon, num_dynamic_vars, dynamic_input_dim)
    
    - The `fit` method trains the underlying Keras model using default parameters.
      For customized training, consider modifying the `build_model` method or extending
      the wrapper.
    
    - Hyperparameter tuning can be performed using Scikit-Learn's model selection
      tools. Ensure that all tunable parameters are exposed in the `__init__` method.
    
    - The `predict` method returns deterministic predictions if `quantiles` is `None`.
      If `quantiles` are specified, the model can be extended to provide probabilistic
      forecasts.
    
    See also
    --------
    sklearn.model_selection.RandomizedSearchCV : Randomized hyperparameter search.
    sklearn.model_selection.GridSearchCV : Exhaustive grid hyperparameter search.
    tensorflow.keras.Model : TensorFlow Keras Model API.
    """
    @validate_params({
        "static_input_dim": [Interval(Integral, 1, None, closed='left')], 
        "dynamic_input_dim": [Interval(Integral, 1, None, closed='left')], 
        "num_static_vars": [Interval(Integral, 1, None, closed='left')], 
        "num_dynamic_vars": [Interval(Integral, 1, None, closed='left')],
        "hidden_units": [Interval(Integral, 1, None, closed='left')], 
        "num_heads": [Interval(Integral, 1, None, closed='left')],
        "dropout_rate": [Interval(Real, 0, 1, closed="both")],
        "forecast_horizon": [Interval(Integral, 1, None, closed='left')],
        "quantiles": ['array-like', list,  None],
        "activation": [
            StrOptions({"elu", "relu", "tanh", "sigmoid", "linear", 'gelu'})],
        "use_batch_norm": [bool],
        "num_lstm_layers": [Interval(Integral, 1, None, closed='left')],
        "lstm_units": [list, Interval(Integral, 1, None, closed='left'), None]
        },
    )

    def __init__(
        self,
        static_input_dim,
        dynamic_input_dim,
        num_static_vars,
        num_dynamic_vars,
        hidden_units,
        num_heads=4,
        dropout_rate=0.1,
        forecast_horizon=1,
        quantiles=None,
        activation='elu',
        use_batch_norm=False,
        num_lstm_layers=1,
        lstm_units=None, 
        optimizer='adam',
        loss='mse',
        metrics=None,
        verbose=0,
        **kwargs
    ):
        self.static_input_dim    = static_input_dim
        self.dynamic_input_dim   = dynamic_input_dim
        self.num_static_vars     = num_static_vars
        self.num_dynamic_vars    = num_dynamic_vars
        self.hidden_units        = hidden_units
        self.num_heads           = num_heads
        self.dropout_rate        = dropout_rate
        self.forecast_horizon    = forecast_horizon
        self.quantiles           = quantiles
        self.activation          = activation
        self.use_batch_norm      = use_batch_norm
        self.num_lstm_layers     = num_lstm_layers
        self.lstm_units          = lstm_units
        self.optimizer           = optimizer
        self.loss                = loss
        self.metrics             = metrics if metrics is not None else ['mae']
        self.verbose             = verbose
        self.kwargs              = kwargs

    def build_model(self):
        if self.verbose >= 5:
            print("Building the Temporal Fusion Transformer model...")
        
        # Instantiate the Temporal Fusion Transformer model
        model = TemporalFusionTransformer(
            static_input_dim=self.static_input_dim,
            dynamic_input_dim=self.dynamic_input_dim,
            num_static_vars=self.num_static_vars,
            num_dynamic_vars=self.num_dynamic_vars,
            hidden_units=self.hidden_units,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            forecast_horizon=self.forecast_horizon,
            quantiles=self.quantiles,
            activation=self.activation,
            use_batch_norm=self.use_batch_norm,
            num_lstm_layers=self.num_lstm_layers,
            lstm_units=self.lstm_units,
            **self.kwargs
        )
        
        if self.verbose >= 6:
            print("Model instantiated with the following parameters:")
            print(f"static_input_dim: {self.static_input_dim}")
            print(f"dynamic_input_dim: {self.dynamic_input_dim}")
            print(f"num_static_vars: {self.num_static_vars}")
            print(f"num_dynamic_vars: {self.num_dynamic_vars}")
            print(f"hidden_units: {self.hidden_units}")
            print(f"num_heads: {self.num_heads}")
            print(f"dropout_rate: {self.dropout_rate}")
            print(f"forecast_horizon: {self.forecast_horizon}")
            print(f"quantiles: {self.quantiles}")
            print(f"activation: {self.activation}")
            print(f"use_batch_norm: {self.use_batch_norm}")
            print(f"num_lstm_layers: {self.num_lstm_layers}")
            print(f"lstm_units: {self.lstm_units}")
        
        if self.verbose >= 5:
            print("Compiling the model...")
        model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics
        )
        
        if self.verbose >= 5:
            print("Model compiled successfully.")
        
        self.model_ = model
        
        if self.verbose >= 5:
            print("Model building completed.")

    def fit(self, X, y):
        """
        Fit the TFT model according to the given training data.
        """
        if self.verbose >= 3:
            print("Starting the fit process...")
        
        # Concatenate static and dynamic inputs if provided as list or tuple
        if isinstance(X, (list, tuple)):
            if self.verbose >= 4:
                print("Concatenating static and dynamic inputs for compatibility...")
            try:
                X = np.concatenate([x.reshape(X[0].shape[0], -1) for x in X], axis=1)
                if self.verbose >= 6:
                    print(f"Concatenated X shape: {X.shape}")
            except Exception as e:
                if self.verbose >= 1:
                    print(f"Error concatenating inputs: {e}")
                raise ValueError("Error concatenating X inputs.") from e
        elif isinstance(X, np.ndarray):
            if self.verbose >= 6:
                print(f"Input X shape: {X.shape}")
        else:
            raise ValueError("Input X must be a list, tuple, or numpy array.")
        
        # Split X into static and dynamic components
        if self.verbose >= 4:
            print("Splitting X into static and dynamic components...")
        X_static, X_dynamic = self._split_X(X)
        
        if self.verbose >= 5:
            print(f"X_static shape: {X_static.shape}")
            print(f"X_dynamic shape: {X_dynamic.shape}")
        
        # Reshape y to match model output
        y = y.reshape(
            -1, 
            self.forecast_horizon, 
            self.static_input_dim
        )
        if self.verbose >= 6:
            print(f"Reshaped y shape: {y.shape}")
        
        # Build the model if not already built
        if self.model_ is None:
            if self.verbose >= 3:
                print("Building the model...")
            self.build_model()
        
        # Fit the model
        if self.verbose >= 3:
            print("Starting model training...")
        self.model_.fit(
            [X_static, X_dynamic],
            y,
            epochs=10,
            batch_size=32,
            verbose=1 if self.verbose >= 4 else 0
        )
        if self.verbose >= 3:
            print("Model training completed.")
        
        return self

    def predict(self, X):
        """
        Predict using the TFT model.
        """
        if self.verbose >= 3:
            print("Starting the prediction process...")
        
        if self.model_ is None:
            raise NotFittedError(
                "This TFTWrapper instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this estimator."
            )
        
        # Concatenate static and dynamic inputs if provided as list or tuple
        if isinstance(X, (list, tuple)):
            if self.verbose >= 4:
                print("Concatenating static and dynamic inputs for compatibility...")
            try:
                X = np.concatenate([x.reshape(X[0].shape[0], -1) for x in X], axis=1)
                if self.verbose >= 6:
                    print(f"Concatenated X shape: {X.shape}")
            except Exception as e:
                if self.verbose >= 1:
                    print(f"Error concatenating inputs: {e}")
                raise ValueError("Error concatenating X inputs.") from e
        elif isinstance(X, np.ndarray):
            if self.verbose >= 6:
                print(f"Input X shape: {X.shape}")
        else:
            raise ValueError("Input X must be a list, tuple, or numpy array.")
        
        # Split X into static and dynamic components
        if self.verbose >= 4:
            print("Splitting X into static and dynamic components...")
        X_static, X_dynamic = self._split_X(X)
        
        if self.verbose >= 5:
            print(f"X_static shape: {X_static.shape}")
            print(f"X_dynamic shape: {X_dynamic.shape}")
        
        # Generate predictions
        if self.verbose >= 3:
            print("Generating predictions...")
        predictions = self.model_.predict(
            [X_static, X_dynamic],
            batch_size=32,
            verbose=1 if self.verbose >= 4 else 0
        )
        
        if self.verbose >= 3:
            print("Prediction completed.")
        
        return predictions

    def _split_X(self, X):
        """
        Split the concatenated X into static and dynamic components.
        """
        static_size = self.num_static_vars * self.static_input_dim
        dynamic_size = self.forecast_horizon * self.num_dynamic_vars * self.dynamic_input_dim
        total_size = static_size + dynamic_size
        
        if self.verbose >= 6:
            print(f"Expected total features: {total_size}, got: {X.shape[1]}")
        
        if X.shape[1] != total_size:
            message = (
                f"Expected X to have {total_size} features, "
                f"got {X.shape[1]}."
            )
            if self.verbose >= 1:
                print(message)
            raise ValueError(message)
        
        if self.verbose >= 6:
            print("Splitting X into static and dynamic parts...")
        
        # Split static features
        X_static = X[:, :static_size]
        X_static = X_static.reshape(
            (X_static.shape[0], 
             self.num_static_vars, 
             self.static_input_dim)
        )
        if self.verbose >= 6:
            print(f"Reshaped X_static to {X_static.shape}")
        
        # Split dynamic features
        X_dynamic = X[:, static_size:]
        X_dynamic = X_dynamic.reshape(
            (X_dynamic.shape[0], 
             self.forecast_horizon, 
             self.num_dynamic_vars, 
             self.dynamic_input_dim)
        )
        if self.verbose >= 6:
            print(f"Reshaped X_dynamic to {X_dynamic.shape}")
        
        return X_static, X_dynamic
    
@ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
@validate_params({
    "static_input_dim": [Interval(Integral, 1, None, closed='left')], 
    "dynamic_input_dim": [Interval(Integral, 1, None, closed='left')], 
    "num_static_vars": [Interval(Integral, 1, None, closed='left')], 
    "num_dynamic_vars": [Interval(Integral, 1, None, closed='left')],
    "hidden_units": [Interval(Integral, 1, None, closed='left')], 
    "num_heads": [Interval(Integral, 1, None, closed='left')],
    "dropout_rate": [Interval(Real, 0, 1, closed="both")],
    "forecast_horizon": [Interval(Integral, 1, None, closed='left')],
    "quantiles": ['array-like', list,  None],
    "activation": [
        StrOptions({"elu", "relu", "tanh", "sigmoid", "linear", 'gelu'})],
    "use_batch_norm": [bool],
    "num_lstm_layers": [Interval(Integral, 1, None, closed='left')],
    "lstm_units": [list, Interval(Integral, 1, None, closed='left'), None]
    },
)
@doc(math_f= dedent ( 
    """
    .. math::
        \text{TFT Model} = \text{TemporalFusionTransformer}(
            \text{static\_input\_dim}, 
            \text{dynamic\_input\_dim}, 
            \text{num\_static\_vars}, 
            \text{num\_dynamic\_vars}, 
            \text{hidden\_units}, 
            \text{num\_heads}, 
            \text{dropout\_rate}, 
            \text{forecast\_horizon}, 
            \text{quantiles}, 
            \text{activation}, 
            \text{use\_batch\_norm}, 
            \text{num\_lstm\_layers}, 
            \text{lstm\_units}
        )
    
    """
    ),
    tft_params=dedent(
        _shared_docs['tft_params_doc']
        )
 )
def create_tft_model(
    static_input_dim: int,
    dynamic_input_dim: int,
    num_static_vars: int,
    num_dynamic_vars: int,
    hidden_units: int = 64,
    num_heads: int = 4,
    dropout_rate: float = 0.1,
    forecast_horizon: int = 1,
    quantiles: Optional[List[float]] = None,
    activation: str = 'relu',
    use_batch_norm: bool = True,
    num_lstm_layers: int = 1,
    lstm_units: int = 64,
    optimizer: Union[str, Optimizer] = 'adam',
    loss: Union[str, Loss] = 'mse',
    metrics: Optional[List[Union[str, Metric]]] = None,
    **kwargs
) -> Model:
    """
    Create and compile a Temporal Fusion Transformer (TFT) model.
    
    The `create_tft_model` function initializes a Temporal Fusion Transformer model 
    with the specified architecture and compiles it with the provided optimizer, 
    loss function, and evaluation metrics. This function is designed to be compatible 
    with scikit-learn estimators like `KerasRegressor`, enabling integration 
    into scikit-learn workflows such as hyperparameter tuning and cross-validation.
    
    {math_f}
    
    {tft_params}
    
    optimizer : `Union[str, tf.keras.optimizers.Optimizer]`, default=`'adam'`
        Optimizer to use for compiling the model. Can be a string identifier 
        or an instance of a Keras Optimizer.
    
    loss : `Union[str, tf.keras.losses.Loss]`, default=`'mse'`
        Loss function to use for training the model. Can be a string identifier 
        or an instance of a Keras Loss.
    
    metrics : `Optional[List[Union[str, tf.keras.metrics.Metric]]]`, default=`None`
        List of metrics to be evaluated by the model during training and testing.
        If `None`, no additional metrics are used.
    
    **kwargs
        Additional keyword arguments to pass to the TemporalFusionTransformer 
        constructor.
    
    Returns
    -------
    tf.keras.Model
        A compiled Temporal Fusion Transformer model ready for training.
    
    Raises
    ------
    ValueError
        If `forecast_horizon` is not a positive integer.
    
    TypeError
        If `quantiles` is provided but is not a list of floats between 0 and 1.
    
    Examples
    --------
    >>> from gofast.nn.wrappers import create_tft_model
    >>> 
    >>> # Define model parameters
    >>> static_input_dim = 1
    >>> dynamic_input_dim = 1
    >>> num_static_vars = 5
    >>> num_dynamic_vars = 3
    >>> 
    >>> # Create and compile the TFT model
    >>> model = create_tft_model(
    ...     static_input_dim=static_input_dim,
    ...     dynamic_input_dim=dynamic_input_dim,
    ...     num_static_vars=num_static_vars,
    ...     num_dynamic_vars=num_dynamic_vars,
    ...     hidden_units=128,
    ...     num_heads=8,
    ...     dropout_rate=0.2,
    ...     forecast_horizon=1,
    ...     quantiles=[0.1, 0.5, 0.9],
    ...     activation='tanh',
    ...     use_batch_norm=False,
    ...     num_lstm_layers=2,
    ...     lstm_units=128,
    ...     optimizer='adam',
    ...     loss='mse',
    ...     metrics=['mae']
    ... )
    >>> 
    >>> # Display the model summary
    >>> model.summary()
    
    Notes
    -----
    - **Compatibility with scikit-learn:** This function is designed to work seamlessly 
      with scikit-learn estimators such as `KerasRegressor`, enabling integration 
      into scikit-learn pipelines and facilitating hyperparameter tuning via 
      `RandomizedSearchCV` or `GridSearchCV`.
    
    - **Quantile Forecasting:** Providing a list of quantiles enables the model to 
      perform probabilistic forecasting. Each quantile corresponds to a different 
      prediction, allowing for uncertainty estimation in the forecasts.
    
    - **Customizability:** The function exposes various hyperparameters, allowing users 
      to tailor the model architecture to their specific needs and dataset characteristics.
    
    - **Additional Arguments:** Any additional keyword arguments are forwarded to the 
      `TemporalFusionTransformer` constructor, offering further flexibility in model 
      configuration.
    
    See Also
    --------
    `TemporalFusionTransformer` : The underlying TFT model class.
    `KerasRegressor` : scikit-learn wrapper for Keras models.
    
    References
    ----------
    .. [1] Qin, Y., Song, D., Chen, H., Cheng, W., Jiang, G., & Cottrell, G. (2017). 
       Temporal fusion transformers for interpretable multi-horizon time series forecasting. 
       *arXiv preprint arXiv:1912.09363*.
    
    .. [2] Brownlee, J. (2018). *Time Series Forecasting with Python: Create accurate 
       models in Python to forecast the future and gain insight from your time series 
       data*. Machine Learning Mastery.
    """
    if not isinstance(forecast_horizon, int) or forecast_horizon < 1:
        raise ValueError("`forecast_horizon` must be a positive integer.")
    
    if quantiles is not None:
        if (not isinstance(quantiles, list) or 
            not all(isinstance(q, float) and 0 < q < 1 for q in quantiles)):
            raise TypeError("`quantiles` must be a list of floats between 0 and 1.")
    
    # Instantiate the Temporal Fusion Transformer model
    model = TemporalFusionTransformer(
        static_input_dim=static_input_dim,
        dynamic_input_dim=dynamic_input_dim,
        num_static_vars=num_static_vars,
        num_dynamic_vars=num_dynamic_vars,
        hidden_units=hidden_units,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        forecast_horizon=forecast_horizon,
        quantiles=quantiles,
        activation=activation,
        use_batch_norm=use_batch_norm,
        num_lstm_layers=num_lstm_layers,
        lstm_units=lstm_units,
        **kwargs
    )
    
    # Compile the model with the specified optimizer, loss, and metrics
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )
    
    return model

