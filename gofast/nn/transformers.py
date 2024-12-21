# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Implements the Temporal Fusion Transformer (TFT) and the extended
XTFT architecture for multi-horizon time-series forecasting. These
models integrate static and dynamic covariates and can optionally
use future covariates for predictions several steps ahead. They
employ attention mechanisms and can produce probabilistic forecasts
via quantiles.

These architectures allow flexible configuration, including residual
connections, batch normalization, and scaling strategies. Users can
customize the number of attention heads, LSTM units, and activation
functions.

Below are concrete examples demonstrating how to define and use
XTFT and TemporalFusionTransformer models, including how to prepare
the input data, compile the model with an optimizer and loss
function, fit the model, and make predictions. Users can copy these
examples outside of this script and test them.

Example using XTFT
-------------------
Assume:
- ``static_input_dim = 10`` (static features dimension)
- ``dynamic_input_dim = 45`` (dynamic features dimension)
- ``future_covariate_dim = 5`` (future covariates dimension)
- ``forecast_horizons = 20`` (steps ahead to forecast)
- ``output_dim = 1`` (univariate forecast)
- ``quantiles = [0.1, 0.5, 0.9]`` for probabilistic forecasts

For the sake of demonstration, we create synthetic inputs:
- ``static_input``: shape (batch_size, static_input_dim)
- ``dynamic_input``: shape (batch_size, forecast_horizons, dynamic_input_dim)
- ``future_covariate_input``: shape (batch_size, forecast_horizons, future_covariate_dim)
- ``y_true``: shape (batch_size, forecast_horizons, output_dim)

>>> import numpy as np
>>> from gofast.nn.transformers import XTFT
>>> batch_size = 32
>>> forecast_horizons = 20
>>> static_input_dim = 10
>>> dynamic_input_dim = 45
>>> future_covariate_dim = 5
>>> output_dim = 1
>>> quantiles = [0.1, 0.5, 0.9]

# Create synthetic input data for demonstration:
>>> static_input = np.random.randn(batch_size, static_input_dim).astype(np.float32)
>>> dynamic_input = np.random.randn(batch_size, forecast_horizons, dynamic_input_dim).astype(np.float32)
>>> future_covariate_input = np.random.randn(batch_size, forecast_horizons, future_covariate_dim).astype(np.float32)
>>> y_true = np.random.randn(batch_size, forecast_horizons, output_dim).astype(np.float32)

# Initialize the XTFT model:
>>> xtft_model = XTFT(
...     static_input_dim=static_input_dim,
...     dynamic_input_dim=dynamic_input_dim,
...     future_covariate_dim=future_covariate_dim,
...     embed_dim=32,
...     forecast_horizons=forecast_horizons,
...     quantiles=quantiles,
...     max_window_size=10,
...     memory_size=100,
...     num_heads=4,
...     dropout_rate=0.1,
...     output_dim=output_dim,
...     attention_units=32,
...     hidden_units=64,
...     lstm_units=64,
...     scales=None,
...     multi_scale_agg=None,
...     activation='relu',
...     use_residuals=True,
...     use_batch_norm=False,
...     final_agg='last'
... )

# Compile the model with an optimizer and loss function
>>> xtft_model.compile(optimizer='adam', loss='mse')

# Fit the model to the training data
>>> history = xtft_model.fit(
...     [static_input, dynamic_input, future_covariate_input],
...     y_true,
...     epochs=2,
...     verbose=2
... )

# Make predictions on the input data
>>> y_pred = xtft_model.predict(
...     [static_input, dynamic_input, future_covariate_input]
... )
>>> print(y_pred.shape)
# Expected shape: (batch_size, forecast_horizons, num_quantiles, output_dim)


Example using TemporalFusionTransformer (TFT)
----------------------------------------------
Assume:
- ``static_input_dim = 10``
- ``dynamic_input_dim = 45``
- ``forecast_horizon = 20``
- ``output_dim = 1``
- ``quantiles = [0.1, 0.5, 0.9]``

For TFT, define:
- ``num_static_vars = static_input_dim``
- ``num_dynamic_vars = dynamic_input_dim``

Again, we create synthetic inputs:
- ``static_input``: shape (batch_size, static_input_dim)
- ``dynamic_input``: shape (batch_size, forecast_horizon, dynamic_input_dim)
- ``y_true``: shape (batch_size, forecast_horizon, output_dim)

>>> from gofast.nn.transformers import TemporalFusionTransformer
>>> forecast_horizon = 20
>>> num_static_vars = static_input_dim
>>> num_dynamic_vars = dynamic_input_dim

# Create synthetic input data:
>>> static_input = np.random.randn(batch_size, static_input_dim).astype(np.float32)
>>> dynamic_input = np.random.randn(batch_size, forecast_horizon, dynamic_input_dim).astype(np.float32)
>>> y_true = np.random.randn(batch_size, forecast_horizon, output_dim).astype(np.float32)

# Initialize the TFT model:
>>> tft_model = TemporalFusionTransformer(
...     static_input_dim=static_input_dim,
...     dynamic_input_dim=dynamic_input_dim,
...     num_static_vars=num_static_vars,
...     num_dynamic_vars=num_dynamic_vars,
...     hidden_units=64,
...     num_heads=4,
...     dropout_rate=0.1,
...     forecast_horizon=forecast_horizon,
...     quantiles=quantiles,
...     activation='elu',
...     use_batch_norm=False,
...     num_lstm_layers=1,
...     lstm_units=64
... )

# Compile the TFT model:
>>> tft_model.compile(optimizer='adam', loss='mse')

# Fit the TFT model:
>>> history = tft_model.fit(
...     [static_input, dynamic_input],
...     y_true,
...     epochs=2,
...     verbose=2
... )

# Make predictions:
>>> y_pred = tft_model.predict([static_input, dynamic_input])
>>> print(y_pred.shape)
# Expected shape: (batch_size, forecast_horizon, num_quantiles)

These examples demonstrate the initialization, compilation, fitting,
and prediction steps for both XTFT and TFT models, allowing you to
experiment and adapt them to your own datasets and forecasting tasks.
"""

from ._tft import TemporalFusionTransformer
from ._xtft import XTFT

__all__ = ["TemporalFusionTransformer", "XTFT"]
