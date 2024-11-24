# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""Implements the Temporal Fusion Transformer (TFT), a state-of-the-art 
architecture for multi-horizon time-series forecasting.
"""
from numbers import Real, Integral  
from ..compat.sklearn import validate_params, Interval, StrOptions 
from ..tools.depsutils import ensure_pkg
from ..tools.validator import validate_quantiles
from . import KERAS_DEPS, KERAS_BACKEND, dependency_message

if KERAS_BACKEND:
    LSTM = KERAS_DEPS.LSTM
    reshape = KERAS_DEPS.reshape
    Dense = KERAS_DEPS.Dense
    reduce_sum = KERAS_DEPS.reduce_sum
    Softmax = KERAS_DEPS.Softmax
    Flatten = KERAS_DEPS.Flatten
    Dropout = KERAS_DEPS.Dropout 
    stack = KERAS_DEPS.stack
    Layer = KERAS_DEPS.Layer 
    ELU = KERAS_DEPS.ELU 
    LayerNormalization = KERAS_DEPS.LayerNormalization 
    TimeDistributed = KERAS_DEPS.TimeDistributed
    MultiHeadAttention = KERAS_DEPS.MultiHeadAttention
    expand_dims = KERAS_DEPS.expand_dims
    tile = KERAS_DEPS.tile
    range_=KERAS_DEPS.range 
    concat = KERAS_DEPS.concat
    shape = KERAS_DEPS.shape
    Model = KERAS_DEPS.Model 
    Activation = KERAS_DEPS.Activation
    BatchNormalization = KERAS_DEPS.BatchNormalization
    Input = KERAS_DEPS.Input
    add = KERAS_DEPS.add
    maximum = KERAS_DEPS.maximum
    reduce_mean = KERAS_DEPS.reduce_mean
    add_n = KERAS_DEPS.add_n
    K = KERAS_DEPS.backend
    register_keras_serializable=KERAS_DEPS.register_keras_serializable

DEP_MSG = dependency_message('transformers') 

__all__ = ["TemporalFusionTransformer"]


class _PositionalEncoding(Layer):
    def call(self, inputs):
        batch_size, seq_len, feature_dim = shape(
            inputs)[0], shape(inputs)[1], shape(inputs)[2]
        position_indices = range_(0, seq_len, dtype='float32')
        position_indices = expand_dims(position_indices, axis=0)
        position_indices = expand_dims(position_indices, axis=-1)
        position_encoding = tile(
            position_indices, [batch_size, 1, feature_dim])
        return inputs + position_encoding
    
    def get_config(self):
            config = super().get_config().copy()
            return config
        
_PositionalEncoding.__doc__="""\
Positional Encoding layer for incorporating temporal positions.

The Positional Encoding layer adds information about the positions of
elements in a sequence, which helps the model to capture the order of
time steps. This is especially important in models that rely on attention
mechanisms, as they do not inherently consider the sequence order [1]_.

Methods
-------
call(inputs)
    Forward pass of the positional encoding layer.

    Parameters
    ----------
    inputs : Tensor
        Input tensor of shape ``(batch_size, time_steps, feature_dim)``.

    Returns
    -------
    Tensor
        Output tensor of shape ``(batch_size, time_steps, feature_dim)``.

Notes
-----
This layer adds a positional encoding to the input tensor:

1. Compute position indices:
   .. math::
       \text{Positions} = [0, 1, 2, \dots, T - 1]

2. Expand and tile position indices to match input shape.

3. Add positional encoding to inputs:
   .. math::
       \mathbf{Z} = \mathbf{X} + \text{PositionEncoding}

This simple addition allows the model to be aware of the position of each
time step in the sequence.

Examples
--------
>>> from gofast.nn.transformers import PositionalEncoding
>>> import tensorflow as tf
>>> # Define input tensor
>>> inputs = tf.random.normal((32, 10, 64))
>>> # Instantiate positional encoding layer
>>> pe = PositionalEncoding()
>>> # Forward pass
>>> outputs = pe(inputs)

See Also
--------
TemporalFusionTransformer : Incorporates positional encoding in dynamic features.

References
----------
.. [1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L.,
       Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). "Attention is all
       you need." In *Advances in Neural Information Processing Systems*
       (pp. 5998-6008).
"""

@ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
@register_keras_serializable()
class GatedResidualNetwork(Layer):
    @validate_params({
            "units": [Interval(Integral, 1, None, closed='left')], 
            "dropout_rate": [Interval(Real, 0, 1, closed="both")],
            "use_time_distributed": [bool],
            "activation": [StrOptions({"elu", "relu", "tanh", "sigmoid", "linear"})],
            "use_batch_norm": [bool],
        },
    )
    def __init__(
        self,
        units,
        dropout_rate=0.0,
        use_time_distributed=False,
        activation='elu',
        use_batch_norm=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        self.use_time_distributed = use_time_distributed
        self.activation_name = activation
        self.activation = Activation(activation)
        self.use_batch_norm = use_batch_norm

        self.linear = Dense(units)
        self.linear2 = Dense(units)
        if self.use_batch_norm:
            self.batch_norm = BatchNormalization()
        self.dropout = Dropout(dropout_rate)
        self.layer_norm = LayerNormalization()
        self.gate = Dense(
            units,
            activation='sigmoid'
        )
        self.projection = None

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if input_dim != self.units:
            self.projection = Dense(self.units)
        super().build(input_shape)

    def call(self, x, training=False):
        shortcut = x
        x = self.linear(x)
        x = self.activation(x)
        if self.use_batch_norm:
            x = self.batch_norm(x, training=training)
        x = self.linear2(x)
        x = self.dropout(x, training=training)
        gate = self.gate(x)
        x = x * gate
        if self.projection is not None:
            shortcut = self.projection(shortcut)
        x = x + shortcut
        x = self.layer_norm(x)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'dropout_rate': self.dropout_rate,
            'use_time_distributed': self.use_time_distributed,
            'activation': self.activation_name,
            'use_batch_norm': self.use_batch_norm,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

GatedResidualNetwork.__d0c__="""\
Gated Residual Network (GRN) layer for processing inputs in the TFT.

The Gated Residual Network allows the model to capture complex nonlinear
relationships while controlling information flow via gating mechanisms.
It consists of a nonlinear layer followed by gating and residual
connections, enabling deep feature transformation with controlled
information flow [1]_.

Parameters
----------
units : int
    The number of units in the GRN layer. This defines the dimensionality
    of the output space.
dropout_rate : float, optional
    The dropout rate used in the dropout layers to prevent overfitting.
    Default is ``0.0``.
use_time_distributed : bool, optional
    Whether to apply the layer over the temporal dimension using
    ``TimeDistributed`` wrapper. Default is ``False``.
activation : str, optional
    Activation function to use. Supported values are ``'elu'``, ``'relu'``,
    ``'tanh'``, ``'sigmoid'``, and ``'linear'``. Default is ``'elu'``.
use_batch_norm : bool, optional
    Whether to use batch normalization in the layer. Batch normalization
    can help stabilize and accelerate training. Default is ``False``.

Methods
-------
call(inputs, training=False)
    Forward pass of the GRN layer.

    Parameters
    ----------
    inputs : Tensor
        Input tensor of shape ``(batch_size, ..., input_dim)``.
    training : bool, optional
        Whether the layer is in training mode. Default is ``False``.

    Returns
    -------
    Tensor
        Output tensor of shape ``(batch_size, ..., units)``.

get_config()
    Returns the configuration of the layer for serialization.

from_config(config)
    Instantiates the layer from a configuration dictionary.

Notes
-----
The GRN processes the input through a series of transformations:

1. Linear transformation:
   .. math::
       \mathbf{h} = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1

2. Nonlinear activation:
   .. math::
       \mathbf{h} = \text{Activation}(\mathbf{h})

3. Optional batch normalization:
   .. math::
       \mathbf{h} = \text{BatchNorm}(\mathbf{h})

4. Second linear transformation:
   .. math::
       \mathbf{h} = \mathbf{W}_2 \mathbf{h} + \mathbf{b}_2

5. Dropout:
   .. math::
       \mathbf{h} = \text{Dropout}(\mathbf{h})

6. Gating mechanism:
   .. math::
       \mathbf{g} = \sigma(\mathbf{W}_g \mathbf{h} + \mathbf{b}_g)
       \\
       \mathbf{h} = \mathbf{h} \odot \mathbf{g}

7. Residual connection and layer normalization:
   .. math::
       \mathbf{h} = \text{LayerNorm}(\mathbf{h} + \text{Projection}(\mathbf{x}))

The gating mechanism controls the flow of information, and the residual
connection helps in training deeper networks by mitigating the vanishing
gradient problem.

Examples
--------
>>> from gofast.nn.transformers import GatedResidualNetwork
>>> import tensorflow as tf
>>> # Define input tensor
>>> inputs = tf.random.normal((32, 10, 64))
>>> # Instantiate GRN layer
>>> grn = GatedResidualNetwork(
...     units=64,
...     dropout_rate=0.1,
...     use_time_distributed=True,
...     activation='relu',
...     use_batch_norm=True
... )
>>> # Forward pass
>>> outputs = grn(inputs, training=True)

See Also
--------
VariableSelectionNetwork : Uses GRN for variable processing.
TemporalFusionTransformer : Incorporates GRN in various components.

References
----------
.. [1] Lim, B., & Zohren, S. (2021). "Time-series forecasting with deep
       learning: a survey." *Philosophical Transactions of the Royal
       Society A*, 379(2194), 20200209.
"""

@ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
@register_keras_serializable()
class VariableSelectionNetwork(Layer):
    @validate_params({
            "input_dim": [Interval(Integral, 1, None, closed='left')], 
            "num_inputs": [Interval(Integral, 1, None, closed='left')], 
            "units": [Interval(Integral, 1, None, closed='left')], 
            "dropout_rate": [Interval(Real, 0, 1, closed="both")],
            "use_time_distributed": [bool],
            "activation": [StrOptions({"elu", "relu", "tanh", "sigmoid", "linear"})],
            "use_batch_norm": [bool],
        },
    )
    def __init__(
        self,
        input_dim,
        num_inputs,
        units,
        dropout_rate=0.0,
        use_time_distributed=False,
        activation='elu',
        use_batch_norm=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.num_inputs = num_inputs
        self.units = units
        self.dropout_rate = dropout_rate
        self.use_time_distributed = use_time_distributed
        self.activation_name = activation
        self.use_batch_norm = use_batch_norm

        self.flatten = Flatten()
        self.softmax = Softmax(axis=-2)
        self.single_variable_grns = [
            GatedResidualNetwork(
                units,
                dropout_rate,
                use_time_distributed,
                activation=activation,
                use_batch_norm=use_batch_norm
            )
            for _ in range(num_inputs)
        ]
        self.variable_importance_dense = Dense(1)
        
    def call(self, inputs, training=False):
        variable_outputs = []
        for i in range(self.num_inputs):
            if self.use_time_distributed:
                var_input = inputs[:, :, i, :]
            else:
                var_input = inputs[:, i, :]
            grn_output = self.single_variable_grns[i](var_input, training=training)
            variable_outputs.append(grn_output)

        stacked_outputs = stack(variable_outputs, axis=-2)
        variable_importances = self.variable_importance_dense(stacked_outputs)
        weights = self.softmax(variable_importances)
        outputs = reduce_sum(stacked_outputs * weights, axis=-2)
        return outputs

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'input_dim': self.input_dim,
            'num_inputs': self.num_inputs,
            'units': self.units,
            'dropout_rate': self.dropout_rate,
            'use_time_distributed': self.use_time_distributed,
            'activation': self.activation_name,
            'use_batch_norm': self.use_batch_norm,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

VariableSelectionNetwork.__doc__="""\
Variable Selection Network (VSN) for selecting relevant variables.

The Variable Selection Network applies Gated Residual Networks (GRNs) to
each variable and computes variable importance weights via a softmax
function. This allows the model to focus on the most informative features
by assigning higher weights to them [1]_.

Parameters
----------
input_dim : int
    The input dimension per variable. Typically ``1`` for scalar features
    or higher for embeddings.
num_inputs : int
    The number of input variables.
units : int
    The number of hidden units in the GRNs.
dropout_rate : float, optional
    The dropout rate used in dropout layers to prevent overfitting.
    Default is ``0.0``.
use_time_distributed : bool, optional
    Whether to apply the layer over the temporal dimension using
    ``TimeDistributed`` wrapper. Default is ``False``.
activation : str, optional
    Activation function to use in GRNs. Supported values are ``'elu'``,
    ``'relu'``, ``'tanh'``, ``'sigmoid'``, and ``'linear'``. Default is
    ``'elu'``.
use_batch_norm : bool, optional
    Whether to use batch normalization in GRNs. Default is ``False``.

Methods
-------
call(inputs, training=False)
    Forward pass of the VSN.

    Parameters
    ----------
    inputs : Tensor
        Input tensor of shape:
        - Without time distribution:
          ``(batch_size, num_inputs, input_dim)``
        - With time distribution:
          ``(batch_size, time_steps, num_inputs, input_dim)``
    training : bool, optional
        Whether the layer is in training mode. Default is ``False``.

    Returns
    -------
    Tensor
        Output tensor of shape:
        - Without time distribution:
          ``(batch_size, units)``
        - With time distribution:
          ``(batch_size, time_steps, units)``

get_config()
    Returns the configuration of the layer for serialization.

from_config(config)
    Instantiates the layer from a configuration dictionary.

Notes
-----
The VSN processes each variable individually using GRNs and computes
variable importance weights:

1. Apply GRN to each variable:
   .. math::
       \mathbf{h}_i = \text{GRN}(\mathbf{x}_i), \quad i = 1, \dots, n

2. Stack GRN outputs:
   .. math::
       \mathbf{H} = [\mathbf{h}_1, \mathbf{h}_2, \dots, \mathbf{h}_n]

3. Compute variable importance weights:
   .. math::
       \boldsymbol{\alpha} = \text{Softmax}(\mathbf{W}_v \mathbf{H})

4. Weighted sum of GRN outputs:
   .. math::
       \mathbf{e} = \sum_{i=1}^{n} \alpha_i \mathbf{h}_i

This results in a single representation that emphasizes the most important
variables.

Examples
--------
>>> from gofast.nn.transformers import VariableSelectionNetwork
>>> import tensorflow as tf
>>> # Define input tensor
>>> inputs = tf.random.normal((32, 5, 1))  # 5 variables, scalar features
>>> # Instantiate VSN layer
>>> vsn = VariableSelectionNetwork(
...     input_dim=1,
...     num_inputs=5,
...     units=64,
...     dropout_rate=0.1,
...     activation='relu',
...     use_batch_norm=True
... )
>>> # Forward pass
>>> outputs = vsn(inputs, training=True)

See Also
--------
GatedResidualNetwork : Used within VSN for variable processing.
TemporalFusionTransformer : Incorporates VSN for feature selection.

References
----------
.. [1] Lim, B., & Zohren, S. (2021). "Time-series forecasting with deep
       learning: a survey." *Philosophical Transactions of the Royal
       Society A*, 379(2194), 20200209.
"""


@ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
@register_keras_serializable()
class TemporalAttentionLayer(Layer):
    @validate_params({
            "units": [Interval(Integral, 1, None, closed='left')], 
            "num_heads": [Interval(Integral, 1, None, closed='left')],
            "dropout_rate": [Interval(Real, 0, 1, closed="both")],
            "activation": [StrOptions({"elu", "relu", "tanh", "sigmoid", "linear"})],
            "use_batch_norm": [bool],
        },
    )
    def __init__(
        self,
        units,
        num_heads,
        dropout_rate=0.0,
        activation='elu',
        use_batch_norm=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.activation_name = activation
        self.use_batch_norm = use_batch_norm
        
        self.multi_head_attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=units,
            dropout=dropout_rate
        )
        self.dropout = Dropout(dropout_rate)
        self.layer_norm = LayerNormalization()
        self.grn = GatedResidualNetwork(
            units,
            dropout_rate,
            use_time_distributed=True,
            activation=activation,
            use_batch_norm=use_batch_norm
        )
        self.context_grn = GatedResidualNetwork(
            units,
            dropout_rate,
            activation=activation,
            use_batch_norm=use_batch_norm
        )

    def call(self, inputs, context_vector, training=False):
        context_vector = self.context_grn(context_vector, training=training)
        context_expanded = expand_dims(context_vector, axis=1)
        context_expanded = tile(
            context_expanded,
            [1, shape(inputs)[1], 1]
        )
        query = inputs + context_expanded
        attn_output = self.multi_head_attention(
            query=query,
            value=inputs,
            key=inputs,
            training=training
        )
        attn_output = self.dropout(attn_output, training=training)
        x = self.layer_norm(inputs + attn_output)
        output = self.grn(x, training=training)
        return output
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation_name,
            'use_batch_norm': self.use_batch_norm,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
TemporalAttentionLayer.__doc__="""\
Temporal Attention Layer for focusing on important time steps.

The Temporal Attention Layer applies multi-head attention over the
temporal dimension to focus on important time steps when making
predictions. This mechanism allows the model to weigh different time steps
differently, capturing temporal dependencies more effectively [1]_.

Parameters
----------
units : int
    The number of units in the attention layer.
num_heads : int
    The number of attention heads. Multiple heads allow the model to focus
    on different aspects of the temporal dynamics simultaneously.
dropout_rate : float, optional
    The dropout rate used in the dropout layers to prevent overfitting.
    Default is ``0.0``.
activation : str, optional
    Activation function to use in GRNs. Supported values are ``'elu'``,
    ``'relu'``, ``'tanh'``, ``'sigmoid'``, and ``'linear'``. Default is
    ``'elu'``.
use_batch_norm : bool, optional
    Whether to use batch normalization in GRNs. Default is ``False``.

Methods
-------
call(inputs, context_vector, training=False)
    Forward pass of the temporal attention layer.

    Parameters
    ----------
    inputs : Tensor
        Input tensor of shape ``(batch_size, time_steps, units)``.
    context_vector : Tensor
        Static context vector of shape ``(batch_size, units)`` used to
        enrich the attention mechanism.
    training : bool, optional
        Whether the layer is in training mode. Default is ``False``.

    Returns
    -------
    Tensor
        Output tensor of shape ``(batch_size, time_steps, units)``.

get_config()
    Returns the configuration of the layer for serialization.

from_config(config)
    Instantiates the layer from a configuration dictionary.

Notes
-----
The Temporal Attention Layer performs the following steps:

1. Enrich context vector using GRN:
   .. math::
       \mathbf{c} = \text{GRN}(\mathbf{c})

2. Expand and repeat context vector over time:
   .. math::
       \mathbf{C} = \text{Tile}(\mathbf{c}, T)

3. Compute query by combining inputs and context:
   .. math::
       \mathbf{Q} = \mathbf{X} + \mathbf{C}

4. Apply multi-head attention:
   .. math::
       \mathbf{Z} = \text{MultiHeadAttention}(\mathbf{Q}, \mathbf{X},
       \mathbf{X})

5. Apply dropout and layer normalization:
   .. math::
       \mathbf{Z} = \text{LayerNorm}(\mathbf{Z} + \mathbf{X})

6. Pass through GRN:
   .. math::
       \mathbf{Z} = \text{GRN}(\mathbf{Z})

Examples
--------
>>> from gofast.nn.transformers import TemporalAttentionLayer
>>> import tensorflow as tf
>>> # Define input tensors
>>> inputs = tf.random.normal((32, 10, 64))
>>> context_vector = tf.random.normal((32, 64))
>>> # Instantiate temporal attention layer
>>> tal = TemporalAttentionLayer(
...     units=64,
...     num_heads=4,
...     dropout_rate=0.1,
...     activation='relu',
...     use_batch_norm=True
... )
>>> # Forward pass
>>> outputs = tal(inputs, context_vector, training=True)

See Also
--------
GatedResidualNetwork : Used within the attention layer.
TemporalFusionTransformer : Incorporates the temporal attention layer.

References
----------
.. [1] Lim, B., & Zohren, S. (2021). "Time-series forecasting with deep
       learning: a survey." *Philosophical Transactions of the Royal
       Society A*, 379(2194), 20200209.
"""


@ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
@register_keras_serializable()
class StaticEnrichmentLayer(Layer):
    @validate_params({
            "units": [Interval(Integral, 1, None, closed='left')], 
            "activation": [StrOptions({"elu", "relu", "tanh", "sigmoid", "linear"})],
            "use_batch_norm": [bool],
        },
    )
    def __init__(
            self, units,
            activation='elu', 
            use_batch_norm=False, 
            **kwargs,
            ):
        super().__init__(**kwargs)
        self.units = units
        self.activation_name = activation
        self.use_batch_norm = use_batch_norm
        
    
        self.grn = GatedResidualNetwork(
            units, 
            activation=activation, 
            use_batch_norm=use_batch_norm
        )
    

    def call(self, static_context_vector, temporal_features, training=False):
        static_context_expanded = expand_dims(
            static_context_vector,
            axis=1
        )
        static_context_expanded = tile(
            static_context_expanded,
            [1, shape(temporal_features)[1], 1]
        )
        combined = concat(
            [static_context_expanded, temporal_features],
            axis=-1
        )
        output = self.grn(combined, training=training)
        return output
    
    def get_config(self):
       config = super().get_config().copy()
       config.update({
           'units': self.units,
           'activation': self.activation_name,
           'use_batch_norm': self.use_batch_norm,
       })
       return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
   
StaticEnrichmentLayer.__doc__="""\
Static Enrichment Layer for combining static and temporal features.

The Static Enrichment Layer enriches temporal features with static
context, enabling the model to adjust temporal dynamics based on static
information. This layer combines static embeddings with temporal
representations through a Gated Residual Network (GRN) [1]_.

Parameters
----------
units : int
    The number of units in the GRN used within the layer.
activation : str, optional
    Activation function to use in the GRN. Supported values are ``'elu'``,
    ``'relu'``, ``'tanh'``, ``'sigmoid'``, and ``'linear'``. Default is
    ``'elu'``.
use_batch_norm : bool, optional
    Whether to use batch normalization in the GRN. Default is ``False``.

Methods
-------
call(static_context_vector, temporal_features, training=False)
    Forward pass of the static enrichment layer.

    Parameters
    ----------
    static_context_vector : Tensor
        Static context vector of shape ``(batch_size, units)``.
    temporal_features : Tensor
        Temporal features tensor of shape ``(batch_size, time_steps,
        units)``.
    training : bool, optional
        Whether the layer is in training mode. Default is ``False``.

    Returns
    -------
    Tensor
        Output tensor of shape ``(batch_size, time_steps, units)``.

get_config()
    Returns the configuration of the layer for serialization.

from_config(config)
    Instantiates the layer from a configuration dictionary.

Notes
-----
The Static Enrichment Layer performs the following steps:

1. Expand and repeat static context vector over time:
   .. math::
       \mathbf{C} = \text{Tile}(\mathbf{c}, T)

2. Concatenate static context with temporal features:
   .. math::
       \mathbf{H} = \text{Concat}[\mathbf{C}, \mathbf{X}]

3. Pass through GRN:
   .. math::
       \mathbf{Z} = \text{GRN}(\mathbf{H})

This allows the model to adjust temporal representations based on static
information.

Examples
--------
>>> from gofast.nn.transformers import StaticEnrichmentLayer
>>> import tensorflow as tf
>>> # Define input tensors
>>> static_context_vector = tf.random.normal((32, 64))
>>> temporal_features = tf.random.normal((32, 10, 64))
>>> # Instantiate static enrichment layer
>>> sel = StaticEnrichmentLayer(
...     units=64,
...     activation='relu',
...     use_batch_norm=True
... )
>>> # Forward pass
>>> outputs = sel(static_context_vector, temporal_features, training=True)

See Also
--------
GatedResidualNetwork : Used within the static enrichment layer.
TemporalFusionTransformer : Incorporates the static enrichment layer.

References
----------
.. [1] Lim, B., & Zohren, S. (2021). "Time-series forecasting with deep
       learning: a survey." *Philosophical Transactions of the Royal
       Society A*, 379(2194), 20200209.
"""

@ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
@register_keras_serializable()
class TemporalFusionTransformer(Model):
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
        "activation": [StrOptions({"elu", "relu", "tanh", "sigmoid", "linear"})],
        "use_batch_norm": [bool],
        "num_lstm_layers": [Interval(Integral, 1, None, closed='left')],
        "lstm_units": [Interval(Integral, 1, None, closed='left'), None]
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
        **kwargs
    ):
        super().__init__(**kwargs)
        self.static_input_dim = static_input_dim
        self.dynamic_input_dim = dynamic_input_dim
        self.num_static_vars = num_static_vars
        self.num_dynamic_vars = num_dynamic_vars
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.forecast_horizon = forecast_horizon
        self.quantiles = quantiles
        self.activation_name = activation
        self.use_batch_norm = use_batch_norm
        self.num_lstm_layers = num_lstm_layers
        self.lstm_units = lstm_units
        
        if quantiles is None:
            self.quantiles = None
            self.num_quantiles = 1
        else:
            self.quantiles = validate_quantiles(quantiles)
            self.num_quantiles = len(self.quantiles)

        # Variable Selection Networks
        self.static_var_sel = VariableSelectionNetwork(
            input_dim=static_input_dim,
            num_inputs=num_static_vars,
            units=hidden_units,
            dropout_rate=dropout_rate,
            activation=activation,
            use_batch_norm=use_batch_norm
        )
        self.dynamic_var_sel = VariableSelectionNetwork(
            input_dim=dynamic_input_dim,
            num_inputs=num_dynamic_vars,
            units=hidden_units,
            dropout_rate=dropout_rate,
            use_time_distributed=True,
            activation=activation,
            use_batch_norm=use_batch_norm
        )

        # Positional Encoding
        self.positional_encoding = _PositionalEncoding()

        # Static Context GRNs
        self.static_context_grn = GatedResidualNetwork(
            hidden_units,
            dropout_rate,
            activation=activation,
            use_batch_norm=use_batch_norm
        )
        self.static_context_enrichment_grn = GatedResidualNetwork(
            hidden_units,
            dropout_rate,
            activation=activation,
            use_batch_norm=use_batch_norm
        )

        # LSTM Encoder
        self.lstm_layers = []
        for i in range(num_lstm_layers):
            if lstm_units is not None:
                lstm_units_i = lstm_units[i]
            else:
                lstm_units_i = hidden_units
            self.lstm_layers.append(
                LSTM(
                    lstm_units_i,
                    return_sequences=True,
                    dropout=dropout_rate
                )
            )

        # Static Enrichment Layer
        self.static_enrichment = StaticEnrichmentLayer(
            hidden_units,
            activation=activation,
            use_batch_norm=use_batch_norm
        )

        # Temporal Attention Layer
        self.temporal_attention = TemporalAttentionLayer(
            hidden_units,
            num_heads,
            dropout_rate,
            activation=activation,
            use_batch_norm=use_batch_norm
        )

        # Position-wise Feedforward
        self.positionwise_grn = GatedResidualNetwork(
            hidden_units,
            dropout_rate,
            use_time_distributed=True,
            activation=activation,
            use_batch_norm=use_batch_norm
        )

        # Output Layers for Quantiles
        if self.quantiles is not None:
            self.quantile_outputs = [
                TimeDistributed(Dense(1)) for _ in range(self.num_quantiles)
            ]
        else:
            self.output_layer = TimeDistributed(Dense(1))

    def call(self, inputs, training=False):
        static_inputs, dynamic_inputs = inputs

        # Variable Selection
        static_embedding = self.static_var_sel(
            static_inputs, training=training
        )
        dynamic_embedding = self.dynamic_var_sel(
            dynamic_inputs, training=training
        )

        # Positional Encoding
        dynamic_embedding = self.positional_encoding(dynamic_embedding)

        # Static Context Vectors
        static_context_vector = self.static_context_grn(
            static_embedding, training=training
        )
        static_enrichment_vector = self.static_context_enrichment_grn(
            static_embedding, training=training
        )

        # LSTM Encoder
        x = dynamic_embedding
        for lstm_layer in self.lstm_layers:
            x = lstm_layer(x, training=training)

        # Static Enrichment
        enriched_lstm_output = self.static_enrichment(
            static_enrichment_vector,
            x,
            training=training
        )

        # Temporal Attention
        attention_output = self.temporal_attention(
            enriched_lstm_output,
            context_vector=static_context_vector,
            training=training
        )

        # Position-wise Feedforward
        temporal_feature = self.positionwise_grn(
            attention_output, training=training
        )

        # Final Output
        decoder_steps = self.forecast_horizon
        outputs = temporal_feature[:, -decoder_steps:, :]

        if self.quantiles is not None:
            # Quantile Outputs
            quantile_outputs = [
                quantile_output_layer(outputs)
                for quantile_output_layer in self.quantile_outputs
            ]
            final_output = concat(quantile_outputs, axis=-1)
        else:
            # Single Output for point prediction
            final_output = self.output_layer(outputs)

        return final_output

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'static_input_dim': self.static_input_dim,
            'dynamic_input_dim': self.dynamic_input_dim,
            'num_static_vars': self.num_static_vars,
            'num_dynamic_vars': self.num_dynamic_vars,
            'hidden_units': self.hidden_units,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate,
            'forecast_horizon': self.forecast_horizon,
            'quantiles': self.quantiles,
            'activation': self.activation_name,
            'use_batch_norm': self.use_batch_norm,
            'num_lstm_layers': self.num_lstm_layers,
            'lstm_units': self.lstm_units,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
TemporalFusionTransformer.__doc__=="""\

Temporal Fusion Transformer (TFT) model for time series forecasting.

The Temporal Fusion Transformer combines high-performance multi-horizon
forecasting with interpretable insights into temporal dynamics [1]_.
It integrates several advanced mechanisms, including:

- Variable Selection Networks (VSNs) for static and dynamic features.
- Gated Residual Networks (GRNs) for processing inputs.
- Static Enrichment Layer to incorporate static features into temporal
  processing.
- LSTM Encoder for capturing sequential dependencies.
- Temporal Attention Layer for focusing on important time steps.
- Position-wise Feedforward Layer.
- Final Output Layer for prediction.

Parameters
----------
static_input_dim : int
    The input dimension per static variable. Typically ``1`` for scalar
    features or higher for embeddings. This defines the number of features
    for each static variable. For example, if static variables are
    represented using embeddings of size 16, then ``static_input_dim``
    would be ``16``.

dynamic_input_dim : int
    The input dimension per dynamic variable. This defines the number of
    features for each dynamic variable at each time step. For instance, if
    dynamic variables are represented using embeddings or multiple features,
    specify the appropriate dimension.

num_static_vars : int
    The number of static variables. Static variables are features that do
    not change over time, such as location identifiers, categories, or
    other constants. This parameter indicates how many static variables are
    being used in the model.

num_dynamic_vars : int
    The number of dynamic variables. Dynamic variables are features that
    change over time, such as historical measurements, external
    influences, or other time-varying data. This parameter indicates how
    many dynamic variables are being used at each time step.

hidden_units : int
    The number of hidden units in the model's layers. This parameter
    defines the dimensionality of the hidden layers throughout the model,
    including GRNs, LSTM layers, and fully connected layers. Increasing
    ``hidden_units`` can allow the model to capture more complex patterns
    but may increase computational cost.

num_heads : int, optional
    The number of attention heads used in the temporal attention layer.
    Default is ``4``. Multiple attention heads allow the model to focus on
    different aspects of the temporal dynamics simultaneously, which can
    enhance the model's ability to capture complex temporal relationships.

dropout_rate : float, optional
    The dropout rate used in dropout layers to prevent overfitting.
    Default is ``0.1``. This value should be between ``0`` and ``1``. A
    higher value increases regularization but may lead to underfitting.

forecast_horizon : int, optional
    The number of time steps to forecast. Default is ``1``. This parameter
    defines the number of future time steps the model will predict. For
    multi-step forecasting, set ``forecast_horizon`` to the desired number
    of future steps.

quantiles : list of float or None, optional
    List of quantiles to predict (e.g., ``[0.1, 0.5, 0.9]``). If ``None``,
    the model performs point forecasting and predicts a single value for
    each time step. When quantiles are specified, the model outputs a
    prediction for each quantile, which is useful for estimating prediction
    intervals.

activation : str, optional
    Activation function to use throughout the model. Supported values are
    ``'elu'``, ``'relu'``, ``'tanh'``, ``'sigmoid'``, and ``'linear'``.
    Default is ``'elu'``. The activation function introduces non-linearity
    into the model, allowing it to learn complex patterns.

use_batch_norm : bool, optional
    Whether to use batch normalization in the model layers. Batch
    normalization can help stabilize and accelerate training by
    normalizing layer inputs. Default is ``False``.

num_lstm_layers : int, optional
    Number of LSTM layers in the encoder. Default is ``1``. Adding more
    layers can help the model capture more complex sequential patterns.
    Each additional layer processes the output of the previous LSTM layer.

lstm_units : list of int or None, optional
    List containing the number of units for each LSTM layer. If ``None``,
    all LSTM layers have ``hidden_units`` units. Default is ``None``.
    This parameter allows customizing the size of each LSTM layer. For
    example, to specify different units for each layer, provide a list like
    ``[64, 32]``.

Methods
-------
call(inputs, training=False)
    Forward pass of the model.

    Parameters
    ----------
    inputs : tuple of tensors
        A tuple containing ``(static_inputs, dynamic_inputs)``.

        - ``static_inputs``: Tensor of shape ``(batch_size, num_static_vars,
          static_input_dim)`` representing the static features.
        - ``dynamic_inputs``: Tensor of shape ``(batch_size, time_steps,
          num_dynamic_vars, dynamic_input_dim)`` representing the dynamic
          features.

    training : bool, optional
        Whether the model is in training mode. Default is ``False``.

    Returns
    -------
    Tensor
        The output predictions of the model. The shape depends on the
        ``forecast_horizon`` and whether ``quantiles`` are used.

get_config()
    Returns the configuration of the model for serialization.

from_config(config)
    Instantiates the model from a configuration dictionary.

Notes
-----
The Temporal Fusion Transformer (TFT) model combines the strengths of
sequence-to-sequence models and attention mechanisms to handle complex
temporal dynamics. It provides interpretability by allowing examination
of variable importance and temporal attention weights.

**Variable Selection Networks (VSNs):**

VSNs select relevant variables by applying Gated Residual Networks (GRNs)
to each variable and computing variable importance weights via a softmax
function. This allows the model to focus on the most informative features.

**Gated Residual Networks (GRNs):**

GRNs allow the model to capture complex nonlinear relationships while
controlling information flow via gating mechanisms. They consist of a
nonlinear layer followed by gating and residual connections.

**Static Enrichment Layer:**

Enriches temporal features with static context, enabling the model to
adjust temporal dynamics based on static information. This layer combines
static embeddings with temporal representations.

**Temporal Attention Layer:**

Applies multi-head attention over the temporal dimension to focus on
important time steps. This mechanism allows the model to weigh different
time steps differently when making predictions.

**Mathematical Formulation:**

Let:

- :math:`\mathbf{x}_{\text{static}} \in \mathbb{R}^{n_s \times d_s}` be the
  static inputs,
- :math:`\mathbf{x}_{\text{dynamic}} \in \mathbb{R}^{T \times n_d \times d_d}`
  be the dynamic inputs,
- :math:`n_s` and :math:`n_d` are the numbers of static and dynamic variables,
- :math:`d_s` and :math:`d_d` are their respective input dimensions,
- :math:`T` is the number of time steps.

**Variable Selection Networks (VSNs):**

For static variables:

.. math::

    \mathbf{e}_{\text{static}} = \sum_{i=1}^{n_s} \alpha_i \cdot
    \text{GRN}(\mathbf{x}_{\text{static}, i})

For dynamic variables:

.. math::

    \mathbf{E}_{\text{dynamic}} = \sum_{j=1}^{n_d} \beta_j \cdot
    \text{GRN}(\mathbf{x}_{\text{dynamic}, :, j})

where :math:`\alpha_i` and :math:`\beta_j` are variable importance weights
computed via softmax.

**LSTM Encoder:**

Processes dynamic embeddings to capture sequential dependencies:

.. math::

    \mathbf{H} = \text{LSTM}(\mathbf{E}_{\text{dynamic}})

**Static Enrichment Layer:**

Combines static context with temporal features:

.. math::

    \mathbf{H}_{\text{enriched}} = \text{StaticEnrichment}(
    \mathbf{e}_{\text{static}}, \mathbf{H})

**Temporal Attention Layer:**

Applies attention over time steps:

.. math::

    \mathbf{Z} = \text{TemporalAttention}(\mathbf{H}_{\text{enriched}})

**Position-wise Feedforward Layer:**

Refines the output:

.. math::

    \mathbf{F} = \text{GRN}(\mathbf{Z})

**Final Output:**

For point forecasting:

.. math::

    \hat{y} = \text{OutputLayer}(\mathbf{F}_{T})

For quantile forecasting (if quantiles are specified):

.. math::

    \hat{y}_q = \text{OutputLayer}_q(\mathbf{F}_{T}), \quad q \in \text{quantiles}

where :math:`\mathbf{F}_{T}` is the feature vector at the last time step.

Examples
--------
>>> from gofast.nn.transformers import TemporalFusionTransformer
>>> # Define model parameters
>>> model = TemporalFusionTransformer(
...     static_input_dim=1,
...     dynamic_input_dim=1,
...     num_static_vars=2,
...     num_dynamic_vars=5,
...     hidden_units=64,
...     num_heads=4,
...     dropout_rate=0.1,
...     forecast_horizon=1,
...     quantiles=[0.1, 0.5, 0.9],
...     activation='relu',
...     use_batch_norm=True,
...     num_lstm_layers=2,
...     lstm_units=[64, 32]
... )
>>> model.compile(optimizer='adam', loss='mse')
>>> # Assume `static_inputs`, `dynamic_inputs`, and `labels` are prepared
>>> model.fit(
...     [static_inputs, dynamic_inputs],
...     labels,
...     epochs=10,
...     batch_size=32
... )

Notes
-----
When using quantile regression by specifying the ``quantiles`` parameter,
ensure that your loss function is compatible with quantile prediction,
such as the quantile loss function. Additionally, the model output will
have multiple predictions per time step, corresponding to each quantile.

See Also
--------
VariableSelectionNetwork : Selects relevant variables.
GatedResidualNetwork : Processes inputs with gating mechanisms.
StaticEnrichmentLayer : Enriches temporal features with static context.
TemporalAttentionLayer : Applies attention over time steps.

References
----------
.. [1] Lim, B., & Zohren, S. (2021). "Time-series forecasting with deep
       learning: a survey." *Philosophical Transactions of the Royal
       Society A*, 379(2194), 20200209.
"""


