# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""Implements the Temporal Fusion Transformer (TFT), a state-of-the-art 
architecture for multi-horizon time-series forecasting.
"""
from numbers import Real, Integral  
from typing import List, Optional, Union 
from .._gofastlog import gofastlog
from ..api.docstring import DocstringComponents, _shared_nn_params 
from ..api.property import  NNLearner 
from ..core.checks import is_iterable, validate_nested_param, ParamsValidator 
from ..compat.sklearn import validate_params, Interval, StrOptions 
from ..decorators import Appender 
from ..tools.depsutils import ensure_pkg
from ..tools.validator import validate_quantiles, check_consistent_length 
from . import KERAS_DEPS, KERAS_BACKEND, dependency_message

if KERAS_BACKEND:
    from . import Activation 
    import tensorflow as tf
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
    BatchNormalization = KERAS_DEPS.BatchNormalization
    Input = KERAS_DEPS.Input
    add = KERAS_DEPS.add
    maximum = KERAS_DEPS.maximum
    reduce_mean = KERAS_DEPS.reduce_mean
    add_n = KERAS_DEPS.add_n
    K = KERAS_DEPS.backend
    register_keras_serializable=KERAS_DEPS.register_keras_serializable
    Embedding =KERAS_DEPS.Embedding 
    Concatenate=KERAS_DEPS.Concatenate 
    
DEP_MSG = dependency_message('transformers') 

__all__ = ["TemporalFusionTransformer", "XTFT"]

_param_docs = DocstringComponents.from_nested_components(
    base=DocstringComponents(_shared_nn_params), 
    )

class _PositionalEncoding(Layer):
    """
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
        
@ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
@register_keras_serializable()
class GatedResidualNetwork(Layer, NNLearner):
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
        self.use_batch_norm = use_batch_norm
        
        self.activation = Activation(activation) 
        self.activation_name = self.activation.activation_name
        
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

GatedResidualNetwork.__doc__="""\
Gated Residual Network (GRN) layer for processing inputs in the TFT.

The Gated Residual Network allows the model to capture complex nonlinear
relationships while controlling information flow via gating mechanisms.
It consists of a nonlinear layer followed by gating and residual
connections, enabling deep feature transformation with controlled
information flow [1]_.

Parameters
----------
{params.base.units}
{params.base.dropout_rate}

use_time_distributed : bool, optional
    Whether to apply the layer over the temporal dimension using
    ``TimeDistributed`` wrapper. Default is ``False``.
    
{params.base.activation}
{params.base.use_batch_norm}

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
       \mathbf{{h}} = \mathbf{{W}}_1 \mathbf{{x}} + \mathbf{{b}}_1

2. Nonlinear activation:
   .. math::
       \mathbf{{h}} = \text{{Activation}}(\mathbf{{h}})

3. Optional batch normalization:
   .. math::
       \mathbf{{h}} = \text{{BatchNorm}}(\mathbf{{h}})

4. Second linear transformation:
   .. math::
       \mathbf{{h}} = \mathbf{{W}}_2 \mathbf{{h}} + \mathbf{{b}}_2

5. Dropout:
   .. math::
       \mathbf{{h}} = \text{{Dropout}}(\mathbf{{h}})

6. Gating mechanism:
   .. math::
       \mathbf{{g}} = \sigma(\mathbf{{W}}_g \mathbf{{h}} + \mathbf{{b}}_g)
       \\
       \mathbf{{h}}= \mathbf{{h}} \odot \mathbf{{g}}

7. Residual connection and layer normalization:
   .. math::
       \mathbf{{h}} = \text{{LayerNorm}}(\mathbf{{h}} + \text{{Projection}}(\mathbf{{x}}))


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
""".format(params=_param_docs) 


@ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
@register_keras_serializable()
class VariableSelectionNetwork(Layer, NNLearner):
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
        self.use_batch_norm = use_batch_norm
        
        self.activation = Activation(activation) 
        self.activation_name = self.activation.activation_name

        self.flatten = Flatten()
        self.softmax = Softmax(axis=-2)
        self.single_variable_grns = [
            GatedResidualNetwork(
                units,
                dropout_rate,
                use_time_distributed,
                activation=self.activation_name,
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
{params.base.input_dim} 

num_inputs : int
    The number of input variables.
    
{params.base.units} 
{params.base.dropout_rate} 

use_time_distributed : bool, optional
    Whether to apply the layer over the temporal dimension using
    ``TimeDistributed`` wrapper. Default is ``False``.
    
{params.base.activation} 
{params.base.use_batch_norm} 

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
       \mathbf{{h}}_i = \text{{GRN}}(\mathbf{{x}}_i), \quad i = 1, \dots, n

2. Stack GRN outputs:
   .. math::
       \mathbf{{H}} = [\mathbf{{h}}_1, \mathbf{{h}}_2, \dots, \mathbf{{h}}_n]

3. Compute variable importance weights:
   .. math::
       \boldsymbol{{\alpha}} = \text{{Softmax}}(\mathbf{{W}}_v \mathbf{{H}})

4. Weighted sum of GRN outputs:
   .. math::
       \mathbf{{e}} = \sum_{{i=1}}^{{n}} \alpha_i \mathbf{{h}}_i

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
""".format( params=_param_docs) 


@ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
@register_keras_serializable()
class TemporalAttentionLayer(Layer, NNLearner):
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
        
        self.activation = Activation(activation) 
        self.activation_name = self.activation.activation_name
        
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
            activation=self.activation_name,
            use_batch_norm=use_batch_norm
        )
        self.context_grn = GatedResidualNetwork(
            units,
            dropout_rate,
            activation=self.activation_name,
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
{params.base.units} 
{params.base.num_heads}
{params.base.dropout_rate} 
{params.base.activation} 
{params.base.use_batch_norm}

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
       \mathbf{{c}} = \text{{GRN}}(\mathbf{{c}})

2. Expand and repeat context vector over time:
   .. math::
       \mathbf{{C}} = \text{{Tile}}(\mathbf{{c}}, T)

3. Compute query by combining inputs and context:
   .. math::
       \mathbf{{Q}} = \mathbf{{X}} + \mathbf{{C}}

4. Apply multi-head attention:
   .. math::
       \mathbf{{Z}} = \text{{MultiHeadAttention}}(\mathbf{{Q}}, \mathbf{{X}},
       \mathbf{{X}})

5. Apply dropout and layer normalization:
   .. math::
       \mathbf{{Z}} = \text{{LayerNorm}}(\mathbf{{Z}} + \mathbf{{X}})

6. Pass through GRN:
   .. math::
       \mathbf{{Z}} = \text{{GRN}}(\mathbf{{Z}})

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
""".format( params =_param_docs )


@ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
@register_keras_serializable()
class StaticEnrichmentLayer(Layer, NNLearner):
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

        self.use_batch_norm = use_batch_norm
        
        self.activation = Activation(activation) 
        self.activation_name = self.activation.activation_name
    
        self.grn = GatedResidualNetwork(
            units, 
            activation=self.activation_name, 
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
{params.base.units} 
{params.base.activation}
{params.base.use_batch_norm}

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
       \mathbf{{C}} = \text{{Tile}}(\mathbf{{c}}, T)

2. Concatenate static context with temporal features:
   .. math::
       \mathbf{{H}} = \text{{Concat}}[\mathbf{{C}}, \mathbf{{X}}]

3. Pass through GRN:
   .. math::
       \mathbf{{Z}} = \text{{GRN}}(\mathbf{{H}})

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
""".format( params =_param_docs )


@Appender(
    """
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
    """,
    join='\n',
    indents=0
)
    
@ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
@register_keras_serializable()
class TemporalFusionTransformer(Model, NNLearner):
    """
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

    {params.base.hidden_units} 
    {params.base.num_heads}
    {params.base.dropout_rate} 
    
    forecast_horizon : int, optional
        The number of time steps to forecast. Default is ``1``. This parameter
        defines the number of future time steps the model will predict. For
        multi-step forecasting, set ``forecast_horizon`` to the desired number
        of future steps.

    {params.base.quantiles} 
    {params.base.activation} 
    {params.base.use_batch_norm} 

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
    
    """.format( params=_param_docs) 
 
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
        "activation": [StrOptions({"elu", "relu", "tanh", "sigmoid", "linear", "gelu"})],
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
        **kwargs
    ):
        super().__init__(**kwargs)
        self.logger = gofastlog().get_gofast_logger(__name__)
        self.logger.debug(
            "Initializing TemporalFusionTransformer with parameters: "
            f"static_input_dim={static_input_dim}, "
            f"dynamic_input_dim={dynamic_input_dim}, "
            f"num_static_vars={num_static_vars}, "
            f"num_dynamic_vars={num_dynamic_vars}, "
            f"hidden_units={hidden_units}, "
            f"num_heads={num_heads}, "
            f"dropout_rate={dropout_rate}, "
            f"forecast_horizon={forecast_horizon}, "
            f"quantiles={quantiles}, "
            f"activation={activation}, "
            f"use_batch_norm={use_batch_norm}, "
            f"num_lstm_layers={num_lstm_layers}, "
            f"lstm_units={lstm_units}"
        )
        
        self.static_input_dim = static_input_dim
        self.dynamic_input_dim = dynamic_input_dim
        self.num_static_vars = num_static_vars
        self.num_dynamic_vars = num_dynamic_vars
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.forecast_horizon = forecast_horizon
        self.quantiles = quantiles
        self.use_batch_norm = use_batch_norm
        self.num_lstm_layers = num_lstm_layers
        self.lstm_units = lstm_units
        
        self.activation = Activation(activation) 
        self.activation_name = self.activation.activation_name
        
        if quantiles is None:
            self.quantiles = None
            self.num_quantiles = 1
        else:
            self.quantiles = validate_quantiles(quantiles)
            self.num_quantiles = len(self.quantiles)

        # Variable Selection Networks
        self.logger.debug("Initializing Variable Selection Networks...")
        self.static_var_sel = VariableSelectionNetwork(
            input_dim=static_input_dim,
            num_inputs=num_static_vars,
            units=hidden_units,
            dropout_rate=dropout_rate,
            activation=self.activation_name,
            use_batch_norm=use_batch_norm
        )
        self.dynamic_var_sel = VariableSelectionNetwork(
            input_dim=dynamic_input_dim,
            num_inputs=num_dynamic_vars,
            units=hidden_units,
            dropout_rate=dropout_rate,
            use_time_distributed=True,
            activation=self.activation_name,
            use_batch_norm=use_batch_norm
        )

        # Positional Encoding
        self.logger.debug("Initializing Positional Encoding...")
        self.positional_encoding = _PositionalEncoding()

        # Static Context GRNs
        self.logger.debug("Initializing Static Context GRNs...")
        self.static_context_grn = GatedResidualNetwork(
            hidden_units,
            dropout_rate,
            activation=self.activation_name,
            use_batch_norm=use_batch_norm
        )
        self.static_context_enrichment_grn = GatedResidualNetwork(
            hidden_units,
            dropout_rate,
            activation=self.activation_name,
            use_batch_norm=use_batch_norm
        )

        # LSTM Encoder
        self.logger.debug("Initializing LSTM Encoder Layers...")
        self.lstm_layers = []
        if self.lstm_units is not None: 
            lstm_units = is_iterable(self.lstm_units, transform =True)
            
        for i in range(num_lstm_layers):
            if lstm_units is not None:
                lstm_units_i = lstm_units[i]
            else:
                lstm_units_i = hidden_units
            self.lstm_layers.append(
                LSTM(
                    lstm_units_i,
                    return_sequences=True,
                    dropout=dropout_rate,
                    name=f'lstm_layer_{i+1}'
                )
            )

        # Static Enrichment Layer
        self.logger.debug("Initializing Static Enrichment Layer...")
        self.static_enrichment = StaticEnrichmentLayer(
            hidden_units,
            activation=self.activation_name,
            use_batch_norm=use_batch_norm
        )

        # Temporal Attention Layer
        self.logger.debug("Initializing Temporal Attention Layer...")
        self.temporal_attention = TemporalAttentionLayer(
            hidden_units,
            num_heads,
            dropout_rate,
            activation=self.activation_name,
            use_batch_norm=use_batch_norm
        )

        # Position-wise Feedforward
        self.logger.debug("Initializing Position-wise Feedforward Network...")
        self.positionwise_grn = GatedResidualNetwork(
            hidden_units,
            dropout_rate,
            use_time_distributed=True,
            activation=self.activation_name,
            use_batch_norm=use_batch_norm
        )

        # Output Layers for Quantiles
        if self.quantiles is not None:
            self.logger.debug("Initializing Quantile Output Layers...")
            self.quantile_outputs = [
                TimeDistributed(Dense(1), name=f'quantile_output_{i+1}') 
                for i in range(self.num_quantiles)
            ]
        else:
            self.logger.debug(
                "Initializing Single Output Layer for Point Predictions...")
            self.output_layer = TimeDistributed(Dense(1), name='output_layer')

    def call(self, inputs, training=False):
        self.logger.debug("Starting call method with inputs.")
        static_inputs, dynamic_inputs = inputs

        # Variable Selection
        self.logger.debug("Applying Variable Selection on Static Inputs...")
        static_embedding = self.static_var_sel(
            static_inputs, training=training
        )
        self.logger.debug("Applying Variable Selection on Dynamic Inputs...")
        dynamic_embedding = self.dynamic_var_sel(
            dynamic_inputs, training=training
        )

        # Positional Encoding
        self.logger.debug("Applying Positional Encoding to Dynamic Embedding...")
        dynamic_embedding = self.positional_encoding(dynamic_embedding)

        # Static Context Vectors
        self.logger.debug("Generating Static Context Vector...")
        static_context_vector = self.static_context_grn(
            static_embedding, training=training
        )
        self.logger.debug("Generating Static Enrichment Vector...")
        static_enrichment_vector = self.static_context_enrichment_grn(
            static_embedding, training=training
        )

        # LSTM Encoder
        self.logger.debug("Passing through LSTM Encoder Layers...")
        x = dynamic_embedding
        for lstm_layer in self.lstm_layers:
            self.logger.debug(f"Passing through {lstm_layer.name}...")
            x = lstm_layer(x, training=training)

        # Static Enrichment
        self.logger.debug("Applying Static Enrichment...")
        enriched_lstm_output = self.static_enrichment(
            static_enrichment_vector,
            x,
            training=training
        )

        # Temporal Attention
        self.logger.debug("Applying Temporal Attention...")
        attention_output = self.temporal_attention(
            enriched_lstm_output,
            context_vector=static_context_vector,
            training=training
        )

        # Position-wise Feedforward
        self.logger.debug("Applying Position-wise Feedforward Network...")
        temporal_feature = self.positionwise_grn(
            attention_output, training=training
        )

        # Final Output
        self.logger.debug("Generating Final Output...")
        decoder_steps = self.forecast_horizon
        outputs = temporal_feature[:, -decoder_steps:, :]

        if self.quantiles is not None:
            self.logger.debug("Generating Quantile Outputs...")
            quantile_outputs = [
                quantile_output_layer(outputs)
                for quantile_output_layer in self.quantile_outputs
            ]
            final_output = Concatenate(axis=-1, name='final_quantile_output')(
                quantile_outputs)
        else:
            self.logger.debug("Generating Point Predictions...")
            final_output = self.output_layer(outputs)

        self.logger.debug("Call method completed.")
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
        self.logger.debug("Configuration for get_config has been updated.")
        return config

    @classmethod
    def from_config(cls, config):
        cls.logger = gofastlog().get_gofast_logger(__name__)
        cls.logger.debug("Creating TemporalFusionTransformer instance from config.")
        return cls(**config)
    
# -----------------XTFT implementation ----------------------------------------

class LearnedNormalization(Layer):
    """
    A layer that learns mean and std for normalization of inputs.  
    Input: (B, D)  
    Output: (B, D), normalized
    """
    def __init__(self):
        super(LearnedNormalization, self).__init__()

    def build(self, input_shape):
        self.mean = self.add_weight(
            "mean",
            shape=(input_shape[-1],),
            initializer="zeros",
            trainable=True
        )
        self.stddev = self.add_weight(
            "stddev",
            shape=(input_shape[-1],),
            initializer="ones",
            trainable=True
        )

    def call(self, inputs, training=False):
        return (inputs - self.mean) / (self.stddev + 1e-6)

    def get_config(self):
        config = super().get_config().copy()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MultiModalEmbedding(Layer):
    """
    This layer takes multiple input modalities (e.g., dynamic and future covariates), 
    embeds them into a common space, and concatenates them along the feature dimension.

    Input: list of [ (B, T, D_mod1), (B, T, D_mod2), ... ]  
    Output: (B, T, sum_of_embed_dims)
    """
    def __init__(self, embed_dim: int):
        super(MultiModalEmbedding, self).__init__()
        self.embed_dim = embed_dim
        self.dense_layers = []

    def build(self, input_shape):
        for modality_shape in input_shape:
            if modality_shape is not None:
                self.dense_layers.append(
                    Dense(self.embed_dim, activation='relu'))
            else:
                raise ValueError("Unsupported modality type.")

    def call(self, inputs, training=False):
        embeddings = []
        for idx, modality in enumerate(inputs):
            if isinstance(modality, tf.Tensor):
                modality_embed = self.dense_layers[idx](modality)
            else:
                raise ValueError("Unsupported modality type.")
            embeddings.append(modality_embed)
        return tf.concat(embeddings, axis=-1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'embed_dim': self.embed_dim})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class HierarchicalAttention(Layer):
    """
    Hierarchical attention layer that first processes short-term and long-term 
    sequences separately and then combines their attention outputs.

    Input: short_term (B, T, D), long_term (B, T, D)  
    Output: (B, T, U) where U is attention_units
    """
    def __init__(self, units: int, num_heads: int):
        super(HierarchicalAttention, self).__init__()
        self.units = units
        self.short_term_dense = Dense(units)
        self.long_term_dense = Dense(units)
        self.short_term_attention = MultiHeadAttention(
            num_heads=num_heads, key_dim=units)
        self.long_term_attention = MultiHeadAttention(
            num_heads=num_heads, key_dim=units)

    def call(self, inputs, training=False):
        short_term, long_term = inputs
        short_term = self.short_term_dense(short_term)
        long_term = self.long_term_dense(long_term)
        short_term_attention = self.short_term_attention(short_term, short_term)
        long_term_attention = self.long_term_attention(long_term, long_term)
        return short_term_attention + long_term_attention

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'short_term_dense': self.short_term_dense.get_config(),
            'long_term_dense': self.long_term_dense.get_config()
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class CrossAttention(Layer):
    """
    Cross attention layer that attends one source to another.

    Input: source1 (B, T, D), source2 (B, T, D)  
    Output: (B, T, U)
    """
    def __init__(self, units: int, num_heads: int):
        super(CrossAttention, self).__init__()
        self.units = units
        self.source1_dense = Dense(units)
        self.source2_dense = Dense(units)
        self.cross_attention = MultiHeadAttention(num_heads=num_heads, key_dim=units)

    def call(self, inputs, training=False):
        source1, source2 = inputs
        source1 = self.source1_dense(source1)
        source2 = self.source2_dense(source2)
        return self.cross_attention(query=source1, value=source2, key=source2)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'units': self.units})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MemoryAugmentedAttention(Layer):
    """
    Memory-augmented attention layer that uses a learned memory matrix to enhance 
    temporal representation.

    Input: (B, T, D)  
    Output: (B, T, D)
    """
    def __init__(self, units: int, memory_size: int, num_heads: int):
        super(MemoryAugmentedAttention, self).__init__()
        self.units = units
        self.memory_size = memory_size
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=units)

    def build(self, input_shape):
        self.memory = self.add_weight(
            "memory",
            shape=(self.memory_size, self.units),
            initializer="zeros",
            trainable=True
        )

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]
        memory_expanded = tf.tile(tf.expand_dims(self.memory, axis=0), [batch_size, 1, 1])
        memory_attended = self.attention(query=inputs, value=memory_expanded, key=memory_expanded)
        return memory_attended + inputs

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'memory_size': self.memory_size
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class AdaptiveQuantileLoss(Layer):
    """
    Computes adaptive quantile loss for given quantiles.

    Input: y_true (B, H, O), y_pred (B, H, Q, O) if quantiles are not None
    """
    def __init__(self, quantiles: Optional[List[float]]):
        super(AdaptiveQuantileLoss, self).__init__()
        if quantiles == 'auto':
            quantiles = [0.1, 0.5, 0.9]
        self.quantiles = quantiles

    def call(self, y_true, y_pred, training=False):
        if self.quantiles is None:
            return 0.0
        y_true_expanded = tf.expand_dims(y_true, axis=2)  # (B, H, 1, O)
        error = y_true_expanded - y_pred  # (B, H, Q, O)
        quantiles = tf.constant(self.quantiles, dtype=tf.float32)
        quantiles = tf.reshape(quantiles, [1, 1, len(self.quantiles), 1])
        quantile_loss = tf.maximum(quantiles * error, (quantiles - 1) * error)
        return tf.reduce_mean(quantile_loss)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'quantiles': self.quantiles})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class AnomalyLoss(Layer):
    """
    Computes anomaly loss as mean squared anomaly score.

    Input: anomaly_scores (B, H, D)
    """
    def __init__(self, weight: float = 1.0):
        super(AnomalyLoss, self).__init__()
        self.weight = weight

    def call(self, anomaly_scores: tf.Tensor):
        return self.weight * tf.reduce_mean(tf.square(anomaly_scores))

    def get_config(self):
        config = super().get_config().copy()
        config.update({'weight': self.weight})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MultiObjectiveLoss(Layer):
    """
    Combines quantile loss and anomaly loss into a single objective.

    Input: 
        y_true: (B, H, O)
        y_pred: (B, H, Q, O) if quantiles is not None else (B, H, 1, O)
        anomaly_scores: (B, H, D)
    """
    def __init__(self, quantile_loss_fn: Layer, anomaly_loss_fn: Layer):
        super(MultiObjectiveLoss, self).__init__()
        self.quantile_loss_fn = quantile_loss_fn
        self.anomaly_loss_fn = anomaly_loss_fn

    def call(self, y_true, y_pred, anomaly_scores=None, training=False):
        # XXX  nomaly_scores henceform can take None 
        quantile_loss = self.quantile_loss_fn(y_true, y_pred)
        if anomaly_scores is not None:
            anomaly_loss = self.anomaly_loss_fn(anomaly_scores)
            return quantile_loss + anomaly_loss
        
        return quantile_loss #

    def get_config(self):
        config = super().get_config().copy()
        # Note: we don't store fn directly. It's recommended to store references 
        #       and reconstruct them or ensure they are serializable.
        # Here we assume they are Keras layers and implement get_config().
        config.update({
            'quantile_loss_fn': self.quantile_loss_fn.get_config(),
            'anomaly_loss_fn': self.anomaly_loss_fn.get_config()
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Manually reconstruct layers if needed
        quantile_loss_fn = AdaptiveQuantileLoss.from_config(
            config['quantile_loss_fn'])
        anomaly_loss_fn = AnomalyLoss.from_config(
            config['anomaly_loss_fn'])
        return cls(
            quantile_loss_fn=quantile_loss_fn, anomaly_loss_fn=anomaly_loss_fn)


class ExplainableAttention(Layer):
    """
    Returns attention scores from multi-head attention, useful for 
    interpretation.

    Input: (B, T, D)
    Output: attention_scores (B, num_heads, T, T)
    """
    def __init__(self, num_heads: int, key_dim: int):
        super(ExplainableAttention, self).__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

    def call(self, inputs, training=False):
        _, attention_scores = self.attention(
            inputs, inputs, return_attention_scores=True)
        return attention_scores

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MultiDecoder(Layer):
    """
    Multi-horizon decoder:
    Takes a single feature vector per example (B, F) and produces a prediction 
    for each horizon as (B, H, O).

    Input: (B, F)
    Output: (B, H, O)
    """
    def __init__(self, output_dim: int, num_horizons: int):
        super(MultiDecoder, self).__init__()
        self.output_dim = output_dim
        self.num_horizons = num_horizons
        self.decoders = [Dense(output_dim) for _ in range(num_horizons)]

    def call(self, x):
        outputs = [decoder(x) for decoder in self.decoders]
        return tf.stack(outputs, axis=1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim,
            'num_horizons': self.num_horizons
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class MultiResolutionAttentionFusion(Layer):
    """
    Applies multi-head attention fusion over features.
    
    Input: (B, T, D)
    Output: (B, T, D)
    """
    def __init__(self, units: int, num_heads: int):
        super(MultiResolutionAttentionFusion, self).__init__()
        self.units = units
        self.num_heads = num_heads
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=units)

    def call(self, inputs, training=False):
        return self.attention(inputs, inputs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'num_heads': self.num_heads
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class DynamicTimeWindow(Layer):
    """
    Slices the last max_window_size steps from the input sequence.

    Input: (B, T, D)
    Output: (B, W, D) where W = max_window_size
    """
    def __init__(self, max_window_size: int):
        super(DynamicTimeWindow, self).__init__()
        self.max_window_size = max_window_size

    def call(self, inputs, training=False):
        return inputs[:, -self.max_window_size:, :]

    def get_config(self):
        config = super().get_config().copy()
        config.update({'max_window_size': self.max_window_size})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class QuantileDistributionModeling(Layer):
    """
    Projects deterministic outputs (B, H, O) into quantile predictions (B, H, Q, O),
    or returns (B, H, O) if quantiles are None (no extra quantile dimension).

    Input: (B, H, O)
    Output:
        - If quantiles is None: (B, H, O) #rather than  otherwise (B, H, 1, O)
        - If quantiles is a list: (B, H, Q, O)
    """
    def __init__(self, quantiles: Optional[Union[str, List[float]]], output_dim: int):
        super(QuantileDistributionModeling, self).__init__()
        if quantiles == 'auto':
            quantiles = [0.1, 0.5, 0.9]
        self.quantiles = quantiles
        self.output_dim = output_dim

        if self.quantiles is not None:
            self.output_layers = [Dense(output_dim) for _ in self.quantiles]
        else:
            self.output_layer = Dense(output_dim)

    def call(self, inputs, training=False):
        # If no quantiles, return deterministic predictions as (B, H, O)
        if self.quantiles is None:
            # Deterministic predictions: (B, H, 1, O)
            # return tf.expand_dims(self.output_layer(inputs), axis=2)
            return self.output_layer(inputs)

        # Quantile predictions: (B, H, Q, O)
        outputs = []
        for output_layer in self.output_layers:
            quantile_output = output_layer(inputs)  # (B, H, O)
            outputs.append(quantile_output)
        return tf.stack(outputs, axis=2)  # (B, H, Q, O)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'quantiles': self.quantiles,
            'output_dim': self.output_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class MultiScaleLSTM(Layer):
    """
    Multi-scale LSTM layer that can output either the last hidden state
    from each LSTM or full sequences. Behavior controlled by `return_sequences`.
    
    Multi-scale LSTM layer that applies multiple LSTMs at different scales 
    and concatenates their outputs.

    Input: (B, T, D)
    Output: (B, T, sum_of_lstm_units) if return_sequences=True
    """

    def __init__(
        self,
        lstm_units: int,
        scales: Union[str, List[int], None] = None,
        return_sequences: bool = False,
        **kwargs
    ):
        super(MultiScaleLSTM, self).__init__(**kwargs)
        if scales is None or scales == 'auto':
            scales = [1]
        scales = validate_nested_param(scales, List[int], 'scales')
        
        self.lstm_units = lstm_units
        self.scales = scales
        self.return_sequences = return_sequences

        self.lstm_layers = [
            tf.keras.layers.LSTM(
                lstm_units, return_sequences=return_sequences)
            for _ in scales
        ]

    def call(self, inputs, training=False):
        outputs = []
        for scale, lstm in zip(self.scales, self.lstm_layers):
            scaled_input = inputs[:, ::scale, :]
            lstm_output = lstm(scaled_input, training=training)
            outputs.append(lstm_output)

        # If return_sequences=False: each output is 
        # (B, units) -> concat along features: (B, units*len(scales))
        # If return_sequences=True: each output is (B, T', units), 
        # need post-processing outside this layer.
        if not self.return_sequences:
            return tf.concat(outputs, axis=-1)
        else:
            # Return list of full sequences to be processed by XTFT (e.g., pooling)
            # We can stack them along features for uniform shape: 
            # If all scales yield sequences of different lengths, an aggregation
            # strategy is needed outside.
            # For simplicity, we return them as a list. XTFT will handle them.
            return outputs

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'lstm_units': self.lstm_units,
            'scales': self.scales,
            'return_sequences': self.return_sequences
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class XTFT(Model, NNLearner):

    @validate_params({
        "static_input_dim": [Interval(Integral, 1, None, closed='left')], 
        "dynamic_input_dim": [Interval(Integral, 1, None, closed='left')], 
        "future_covariate_dim": [Interval(Integral, 1, None, closed='left')], 
        "embed_dim": [Interval(Integral, 1, None, closed='left')],
        "forecast_horizons": [Interval(Integral, 1, None, closed='left')], 
        "quantiles": ['array-like', StrOptions({'auto'}),  None],
        "max_window_size": [Interval(Integral, 1, None, closed='left')],
        "memory_size": [Interval(Integral, 1, None, closed='left')], 
        "num_heads": [Interval(Integral, 1, None, closed='left')],
        "dropout_rate": [Interval(Real, 0, 1, closed="both")],
        "output_dim": [Interval(Integral, 1, None, closed='left')],
        "forecast_horizon": [Interval(Integral, 1, None, closed='left')],
        "anomaly_loss_weight": [Interval(Real, 0, None, closed='left'), None],
        "attention_units": [
            'array-like', 
            Interval(Integral, 1, None, closed='left')
        ], 
        "hidden_units": [
            'array-like', 
            Interval(Integral, 1, None, closed='left')
          ], 
        "lstm_units": [
            'array-like', 
            Interval(Integral, 1, None, closed='left'), 
            None
        ], 
        "activation": [
            StrOptions({"elu", "relu", "tanh", "sigmoid", "linear", "gelu"}),
            callable 
            ],
        "multi_scale_agg": [
            StrOptions({"last", "average",  "flatten", "auto"}),
            None
        ],
        "scales": ['array-like', StrOptions({"auto"}),  None],
        "use_batch_norm": [bool],
        "use_residuals": [bool],
        "final_agg": [StrOptions({"last", "average",  "flatten"})],
        },
    )
    def __init__(
        self,
        static_input_dim: int,
        dynamic_input_dim: int,
        future_covariate_dim: int,
        embed_dim: int = 32,
        forecast_horizons: int = 1,
        quantiles: Union[str, List[float], None] = None,
        max_window_size: int = 10,
        memory_size: int = 100,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        output_dim: int = 1, 
        anomaly_loss_weight: float =None, 
        attention_units: int = 32,
        hidden_units: int = 64,
        lstm_units: int = 64,
        scales: Union[str, List[int], None] = None,
        multi_scale_agg: Optional[str] = None, 
        activation: str = 'relu',
        use_residuals: bool = True,
        use_batch_norm: bool = False,
        final_agg: str = 'last',
        **kwargs
    ):
        super().__init__(**kwargs)

        # Initialize Logger
        self.logger = gofastlog().get_gofast_logger(__name__)
        self.logger.debug(
            "Initializing XTFT with parameters: "
            f"static_input_dim={static_input_dim}, "
            f"dynamic_input_dim={dynamic_input_dim}, "
            f"future_covariate_dim={future_covariate_dim}, "
            f"embed_dim={embed_dim}, "
            f"forecast_horizons={forecast_horizons}, "
            f"quantiles={quantiles}, "
            f"max_window_size={max_window_size},"
            f" memory_size={memory_size}, num_heads={num_heads}, "
            f"dropout_rate={dropout_rate}, output_dim={output_dim}, "
            f"anomaly_loss_weight={anomaly_loss_weight}, "
            f"attention_units={attention_units}, "
            f" hidden_units={hidden_units}, "
            f"lstm_units={lstm_units}, "
            f"scales={scales}, activation={activation}, "
            f"use_residuals={use_residuals}, "
            f"use_batch_norm={use_batch_norm}, "
            f"final_agg={final_agg}"
        )
        # Handle quantiles
        if quantiles == 'auto':
            quantiles = [0.1, 0.5, 0.9]

        if scales is None or scales == 'auto':
            scales = [1]

        if multi_scale_agg is None or multi_scale_agg == 'auto':
            return_sequences = False
        else:
            return_sequences = True

        self.static_input_dim = static_input_dim
        self.dynamic_input_dim = dynamic_input_dim
        self.future_covariate_dim = future_covariate_dim
        self.embed_dim = embed_dim
        self.forecast_horizons = forecast_horizons
        self.quantiles = quantiles
        self.max_window_size = max_window_size
        self.memory_size = memory_size
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.output_dim = output_dim
        self.anomaly_loss_weight = anomaly_loss_weight 
        self.attention_units = attention_units
        self.hidden_units = hidden_units
        self.lstm_units = lstm_units
        self.scales = scales
        self.multi_scale_agg = multi_scale_agg
        self.activation = activation
        self.use_residuals = use_residuals
        self.use_batch_norm = use_batch_norm
        self.final_agg = final_agg

        # Layers
        self.learned_normalization = LearnedNormalization()
        self.multi_modal_embedding = MultiModalEmbedding(embed_dim)
        
        self.multi_scale_lstm = MultiScaleLSTM(
            lstm_units=self.lstm_units,
            scales=self.scales,
            return_sequences=return_sequences
        )

        self.hierarchical_attention = HierarchicalAttention(
            units=attention_units,
            num_heads=num_heads
        )
        self.cross_attention = CrossAttention(
            units=attention_units,
            num_heads=num_heads
        )
        self.memory_augmented_attention = MemoryAugmentedAttention(
            units=attention_units,
            memory_size=memory_size,
            num_heads=num_heads
        )
        self.multi_decoder = MultiDecoder(
            output_dim=output_dim,
            num_horizons=forecast_horizons
        )
        self.multi_resolution_attention_fusion = MultiResolutionAttentionFusion(
            units=attention_units,
            num_heads=num_heads
        )
        self.dynamic_time_window = DynamicTimeWindow(
            max_window_size=max_window_size
        )
        self.quantile_distribution_modeling = QuantileDistributionModeling(
            quantiles=self.quantiles,
            output_dim=output_dim
        )
        self.anomaly_loss_layer = AnomalyLoss(weight=self.anomaly_loss_weight)
        self.multi_objective_loss = MultiObjectiveLoss(
            quantile_loss_fn=AdaptiveQuantileLoss(self.quantiles),
            anomaly_loss_fn=self.anomaly_loss_layer
        )
        self.static_dense = Dense(hidden_units, activation=activation)
        self.static_dropout = Dropout(dropout_rate)
        if self.use_batch_norm:
            self.static_batch_norm = LayerNormalization()
        self.residual_dense = Dense(2 * embed_dim) if use_residuals else None
        self.final_dense = Dense(output_dim)


    def call(self, inputs, training=False):
        (static_input,
         dynamic_input,
         future_covariate_input) = inputs
    
        # Normalize and process static features
        normalized_static = self.learned_normalization(
            static_input, 
            training=training
        )
        self.logger.debug(
            f"Normalized Static Shape: {normalized_static.shape}"
        )
    
        static_features = self.static_dense(normalized_static)
        if self.use_batch_norm:
            static_features = self.static_batch_norm(
                static_features,
                training=training
            )
            self.logger.debug(
                "Static Features after BatchNorm Shape: "
                f"{static_features.shape}"
            )
    
        static_features = self.static_dropout(
            static_features,
            training=training
        )
        self.logger.debug(
            f"Static Features Shape: {static_features.shape}"
        )
    
        # Embeddings for dynamic and future covariates
        embeddings = self.multi_modal_embedding(
            [dynamic_input, future_covariate_input],
            training=training
        )
        self.logger.debug(
            f"Embeddings Shape: {embeddings.shape}"
        )
    
        if self.use_residuals:
            embeddings = embeddings + self.residual_dense(embeddings)
            self.logger.debug(
                "Embeddings with Residuals Shape: "
                f"{embeddings.shape}"
            )
    
        # Multi-scale LSTM outputs
        lstm_output = self.multi_scale_lstm(
            dynamic_input,
            training=training
        )
        # If multi_scale_agg is None, lstm_output is (B, units * len(scales))
        # If multi_scale_agg is not None, lstm_output is a list of full 
        # sequences: [ (B, T', units), ... ]
    
        if self.multi_scale_agg is None:
            # No additional aggregation needed
            lstm_features = lstm_output  # (B, units * len(scales))
        else:
            # Apply chosen aggregation to full sequences
            if self.multi_scale_agg == "average":
                # Average over time dimension for each scale and then concatenate
                averaged_outputs = [
                    tf.reduce_mean(o, axis=1) 
                    for o in lstm_output
                ]  # Each is (B, units)
                lstm_features = tf.concat(
                    averaged_outputs,
                    axis=-1
                )  # (B, units * len(scales))
    
            elif self.multi_scale_agg == "flatten":
                # Flatten time and feature dimensions for all scales
                # Assume equal time lengths for all scales
                concatenated = tf.concat(
                    lstm_output, 
                    axis=-1
                )  # (B, T', units*len(scales))
                shape = tf.shape(concatenated)
                (batch_size,
                 time_dim,
                 feat_dim) = shape[0], shape[1], shape[2]
                lstm_features = tf.reshape(
                    concatenated,
                    [batch_size, time_dim * feat_dim]
                )
            else:
                # Default fallback: take the last time step from each scale
                # and concatenate
                last_outputs = [
                    o[:, -1, :] 
                    for o in lstm_output
                ]  # (B, units)
                lstm_features = tf.concat(
                    last_outputs,
                    axis=-1
                )  # (B, units * len(scales))
        
        # Since you are concatenating along the time dimension, you need 
        # all tensors to have the same shape along that dimension.
        time_steps = tf.shape(dynamic_input)[1]
        # Expand lstm_features to (B, 1, features)
        lstm_features = tf.expand_dims(lstm_features, axis=1)
        # Tile to match time steps: (B, T, features)
        lstm_features = tf.tile(lstm_features, [1, time_steps, 1])

        self.logger.debug(
            f"LSTM Features Shape: {lstm_features.shape}"
        )
    
        # Attention mechanisms
        hierarchical_att = self.hierarchical_attention(
            [dynamic_input, future_covariate_input],
            training=training
        )
        self.logger.debug(
            f"Hierarchical Attention Shape: {hierarchical_att.shape}"
        )
    
        cross_attention_output = self.cross_attention(
            [dynamic_input, embeddings],
            training=training
        )
        self.logger.debug(
            f"Cross Attention Output Shape: {cross_attention_output.shape}"
        )
    
        memory_attention_output = self.memory_augmented_attention(
            hierarchical_att,
            training=training
        )
        self.logger.debug(
            "Memory Augmented Attention Output Shape: "
            f"{memory_attention_output.shape}"
        )
    
        # Combine all features
        time_steps = tf.shape(dynamic_input)[1]
        static_features_expanded = tf.tile(
            tf.expand_dims(static_features, axis=1),
            [1, time_steps, 1]
        )
        self.logger.debug(
            "Static Features Expanded Shape: "
            f"{static_features_expanded.shape}"
        )
    
        combined_features = Concatenate()([
            static_features_expanded,
            lstm_features,
            memory_attention_output,
            cross_attention_output
        ])
        self.logger.debug(
            f"Combined Features Shape: {combined_features.shape}"
        )
    
        attention_fusion_output = self.multi_resolution_attention_fusion(
            combined_features,
            training=training
        )
        self.logger.debug(
            "Attention Fusion Output Shape: "
            f"{attention_fusion_output.shape}"
        )
    
        time_window_output = self.dynamic_time_window(
            attention_fusion_output,
            training=training
        )
        self.logger.debug(
            f"Time Window Output Shape: {time_window_output.shape}"
        )
    
        # final_agg: last/average/flatten applied on time_window_output
        if self.final_agg == "last":
            final_features = time_window_output[:, -1, :]
        elif self.final_agg == "average":
            final_features = tf.reduce_mean(time_window_output, axis=1)
        else:  # "flatten"
            shape = tf.shape(time_window_output)
            (batch_size,
             time_dim,
             feat_dim) = shape[0], shape[1], shape[2]
            final_features = tf.reshape(
                time_window_output,
                [batch_size, time_dim * feat_dim]
            )
    
        decoder_outputs = self.multi_decoder(
            final_features,
            training=training
        )
        self.logger.debug(
            f"Decoder Outputs Shape: {decoder_outputs.shape}"
        )
    
        predictions = self.quantile_distribution_modeling(
            decoder_outputs,
            training=training
        )
        self.logger.debug(
            f"Predictions Shape: {predictions.shape}"
        )
    
        return predictions

    @ParamsValidator(
        { 
          'y_true': ['array-like:tf:transf'], 
          'y_pred': ['array-like:tf:transf'], 
          'anomaly_scores':['array-like:tf:transf']
        }, 
    )
    def compute_objective_loss(
        self, 
        y_true: tf.Tensor, 
        y_pred: tf.Tensor, 
        anomaly_scores: tf.Tensor=None
    ) -> tf.Tensor:
        
        if self.anomaly_loss_weight is not None: 
            check_consistent_length(y_true, y_pred, anomaly_scores)
            # Expect y_true, 'y_pred, and 'anomaly_scores'
            # Compute the multi-objective loss
            loss = self.multi_objective_loss(y_true, y_pred, anomaly_scores)
            return loss
        else: 
            # When anomaly_loss_weight is None, y_true is a tensor
            check_consistent_length(y_true, y_pred)
            return self.multi_objective_loss(y_true, y_pred)

    def anomaly_loss(
        self, anomaly_scores: tf.Tensor
        ) -> tf.Tensor:
        return self.anomaly_loss_weight * tf.reduce_mean(
            tf.square(anomaly_scores))

    def quantile_loss(self, q):
        def loss(y_true, y_pred):
            error = y_true - y_pred
            return K.mean(K.maximum(q * error, (q - 1) * error), axis=-1)
        return loss

    def compile(self, optimizer, loss=None, **kwargs):
        if self.quantiles is None:
            # Deterministic scenario
            super().compile(
                optimizer=optimizer, loss=loss or 'mse', **kwargs)
        else:
            # Probabilistic scenario with multiple quantile losses
            loss_functions = {}
            for q in self.quantiles:
                loss_functions[f'quantile_loss_{q}'] = self.quantile_loss(q)
            super(XTFT, self).compile(
                optimizer=optimizer, loss=loss_functions, **kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'static_input_dim': self.static_input_dim,
            'dynamic_input_dim': self.dynamic_input_dim,
            'future_covariate_dim': self.future_covariate_dim,
            'embed_dim': self.embed_dim,
            'forecast_horizons': self.forecast_horizons,
            'quantiles': self.quantiles,
            'max_window_size': self.max_window_size,
            'memory_size': self.memory_size,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate,
            'output_dim': self.output_dim,
            'anomaly_loss_weight': self.anomaly_loss_weight,
            'attention_units': self.attention_units,
            'hidden_units': self.hidden_units,
            'lstm_units': self.lstm_units,
            'scales': self.scales,
            'activation': self.activation,
            'use_residuals': self.use_residuals,
            'use_batch_norm': self.use_batch_norm,
            'final_agg': self.final_agg
        })
        self.logger.debug(
            "Configuration for XTFT has been updated in get_config.")
        return config

    @classmethod
    def from_config(cls, config):
        logger = gofastlog().get_gofast_logger(__name__)
        logger.debug("Creating XTFT instance from config.")
        return cls(**config)


XTFT.__doc__ = r"""\
Extreme Temporal Fusion Transformer (XTFT) model for complex time
series forecasting.

The Extreme Temporal Fusion Transformer (XTFT) is an advanced
architecture for time series forecasting, particularly suited to
scenarios featuring intricate temporal patterns, multiple forecast
horizons, and inherent uncertainties [1]_. By extending the original
Temporal Fusion Transformer, XTFT incorporates additional modules
and strategies that enhance its representational capacity, stability,
and interpretability.

**Key Enhancements:**

- **Enhanced Variable Embeddings**: 
  Employs learned normalization and multi-modal embeddings to
  flexibly integrate static, dynamic, and future covariates. 
  This allows the model to effectively handle heterogeneous 
  inputs and exploit relevant signals from different data 
  modalities.
  
  The model applies learned normalization and multi-modal embeddings
  to unify static, dynamic, and future covariates into a common
  representation space. Let :math:`\mathbf{x}_{static}`, 
  :math:`\mathbf{X}_{dynamic}`, and :math:`\mathbf{X}_{future}` 
  denote the static, dynamic, and future input tensors:
  .. math::
     \mathbf{x}_{norm} = \frac{\mathbf{x}_{static} - \mu}
     {\sigma + \epsilon}
     
  After normalization, static and dynamic features are embedded:
  .. math::
     \mathbf{E}_{dyn} = \text{MultiModalEmbedding}
     ([\mathbf{X}_{dynamic}, \mathbf{X}_{future}])
     
  and similarly, static embeddings 
  :math:`\mathbf{E}_{static}` are obtained. This enables flexible 
  integration of heterogeneous signals.

- **Multi-Scale LSTM Mechanisms**: 
  Adopts multiple LSTMs operating at various temporal resolutions
  as controlled by `scales`. By modeling patterns at multiple
  time scales (e.g., daily, weekly, monthly), the model can 
  capture long-term trends, seasonalities, and short-term 
  fluctuations simultaneously.
  
  Multiple LSTMs process the input at different scales defined by 
  `scales`. For a set of scales 
  :math:`S = \{s_1, s_2, \ldots, s_k\}`, each scale selects 
  time steps at intervals of :math:`s_i`:
  .. math::
     \mathbf{H}_{lstm} = \text{Concat}(
     [\text{LSTM}_{s_i}(\mathbf{E}_{dyn}^{(s_i)})]_{i=1}^{k})
     
  where :math:`\mathbf{E}_{dyn}^{(s_i)}` represents 
  :math:`\mathbf{E}_{dyn}` sampled at stride :math:`s_i`. This 
  approach captures patterns at multiple temporal resolutions 
  (e.g., daily, weekly).


- **Enhanced Attention Mechanisms**: 
  Integrates hierarchical, cross, and memory-augmented attention. 
  Hierarchical attention highlights critical temporal regions,
  cross attention fuses information from diverse feature spaces,
  and memory-augmented attention references a learned memory to
  incorporate long-range dependencies beyond the immediate 
  input window.
  
  XTFT integrates hierarchical, cross, and memory-augmented attention
  layers to enrich temporal and contextual relationships.  
  Hierarchical attention:
  .. math::
     \mathbf{H}_{hier} = \text{HierarchicalAttention}
     ([\mathbf{X}_{dynamic}, \mathbf{X}_{future}])
  
  Cross attention:
  .. math::
     \mathbf{H}_{cross} = \text{CrossAttention}
     ([\mathbf{X}_{dynamic}, \mathbf{E}_{dyn}])
  
  Memory-augmented attention with memory :math:`\mathbf{M}`:
  .. math::
     \mathbf{H}_{mem} = \text{MemoryAugmentedAttention}(
     \mathbf{H}_{hier}, \mathbf{M})
     
  Together, these attentions enable the model to focus on 
  short-term critical points, fuse different feature spaces,
  and reference long-range contexts.
  

- **Dynamic Quantile Loss**: 
  Implements adaptive quantile loss to produce probabilistic
  forecasts. This enables the model to return predictive intervals
  and quantify uncertainty, offering more robust and informed 
  decision-making capabilities.
  
  The model supports probabilistic forecasting via quantile loss.
  For quantiles :math:`q \in \{q_1,\ldots,q_Q\}`, and errors 
  :math:`e = y_{true} - y_{pred}`, quantile loss is defined as:
  .. math::
     \mathcal{L}_{quantile}(q) = \frac{1}{N}\sum_{n=1}^{N} 
     \max(q \cdot e_n, (q-1) \cdot e_n)
     
  This yields predictive intervals rather than single-point
  estimates, facilitating uncertainty-aware decision-making.
  

- **Multi-Horizon Output Strategies**:
  Facilitates forecasting over multiple future steps at once, 
  enabling practitioners to assess future scenarios and plan 
  accordingly. This functionality supports both deterministic 
  and probabilistic forecasts.
  
  XTFT predicts multiple horizons simultaneously. If 
  `forecast_horizons = H`, the decoder produces:
  .. math::
     \mathbf{Y}_{decoder} = \text{MultiDecoder}(\mathbf{H}_{combined})
     
  resulting in a forecast:
  .. math::
     \hat{\mathbf{Y}} \in \mathbb{R}^{B \times H \times D_{out}}
  
  This allows practitioners to assess future scenarios over 
  multiple steps rather than a single forecast instant.

- **Optimization for Complex Time Series**:
  Utilizes multi-resolution attention fusion, dynamic time 
  windowing, and residual connections to handle complex and 
  noisy data distributions. Such mechanisms improve training 
  stability and convergence rates, even in challenging 
  environments.
  
  Multi-resolution attention fusion and dynamic time windowing 
  improve the model's capability to handle complex, noisy data:
  .. math::
     \mathbf{H}_{fused} = \text{MultiResolutionAttentionFusion}(
     \mathbf{H}_{combined})
  
  Along with residual connections:
  .. math::
     \mathbf{H}_{res} = \mathbf{H}_{fused} + \mathbf{H}_{combined}
  
  These mechanisms stabilize training, enhance convergence, and 
  improve performance on challenging datasets.

- **Advanced Output Mechanisms**:
  Employs quantile distribution modeling to generate richer
  uncertainty estimations, thereby enabling the model to
  provide more detailed and informative predictions than 
  single-point estimates.
  
  Quantile distribution modeling converts decoder outputs into a
  set of quantiles:
  .. math::
     \mathbf{Y}_{quantiles} = \text{QuantileDistributionModeling}(
     \mathbf{Y}_{decoder})
  
  enabling richer uncertainty estimation and more informative 
  predictions, such as lower and upper bounds for future values.

When `quantiles` are specified, XTFT delivers probabilistic 
forecasts that include lower and upper bounds, enabling better 
risk management and planning. Moreover, anomaly detection 
capabilities, governed by `anomaly_loss_weight`, allow the 
model to identify and adapt to irregularities or abrupt changes
in the data.

This class inherits from both 
:class:`~tensorflow.keras.Model` and 
:class:`~gofast.api.property.NNLearner`, ensuring seamless 
integration with Keras workflows and extended functionality 
for handling model properties and training processes.


Parameters
----------
static_input_dim : int
    Dimensionality of static input features (no time dimension).  
    These features remain constant over time steps and provide
    global context or attributes related to the time series. For
    example, a store ID or geographic location. Increasing this
    dimension allows the model to utilize more contextual signals
    that do not vary with time. A larger `static_input_dim` can
    help the model specialize predictions for different entities
    or conditions and improve personalized forecasts.

dynamic_input_dim : int
    Dimensionality of dynamic input features. These features vary
    over time steps and typically include historical observations
    of the target variable, and any time-dependent covariates such
    as past sales, weather variables, or sensor readings. A higher
    `dynamic_input_dim` enables the model to incorporate more
    complex patterns from a richer set of temporal signals. These
    features help the model understand seasonality, trends, and
    evolving conditions over time.

future_covariate_dim : int
    Dimensionality of future known covariates. These are features
    known ahead of time for future predictions (e.g., holidays,
    promotions, scheduled events, or future weather forecasts).
    Increasing `future_covariate_dim` enhances the model’s ability
    to leverage external information about the future, improving
    the accuracy and stability of multi-horizon forecasts.

embed_dim : int, optional
    Dimension of feature embeddings. Default is ``32``. After
    variable transformations, inputs are projected into embeddings
    of size `embed_dim`. Larger embeddings can capture more nuanced
    relationships but may increase model complexity. A balanced
    choice prevents overfitting while ensuring the representation
    capacity is sufficient for complex patterns.

forecast_horizons : int, optional
    Number of future time steps to predict. Default is ``1``. This
    parameter specifies how many steps ahead the model provides
    forecasts. For instance, `forecast_horizons=3` means the model
    predicts values for three future periods simultaneously.
    Increasing this allows multi-step forecasting, but may
    complicate learning if too large.

quantiles : list of float or str, optional
    Quantiles to predict for probabilistic forecasting. For example,
    ``[0.1, 0.5, 0.9]`` indicates lower, median, and upper bounds.
    If set to ``'auto'``, defaults to ``[0.1, 0.5, 0.9]``. If
    `None`, the model makes deterministic predictions. Providing
    quantiles helps the model estimate prediction intervals and
    uncertainty, offering more informative and robust forecasts.

max_window_size : int, optional
    Maximum dynamic time window size. Default is ``10``. Defines
    the length of the dynamic windowing mechanism that selects
    relevant recent time steps for modeling. A larger `max_window_size`
    enables the model to consider more historical data at once,
    potentially capturing longer-term patterns, but may also
    increase computational cost.

memory_size : int, optional
    Size of the memory for memory-augmented attention. Default is
    ``100``. Introduces a fixed-size memory that the model can
    attend to, providing a global context or reference to distant
    past information. Larger `memory_size` can help the model
    recall patterns from further back in time, improving long-term
    forecasting stability.

num_heads : int, optional
    Number of attention heads. Default is ``4``. Multi-head
    attention allows the model to attend to different representation
    subspaces of the input sequence. Increasing `num_heads` can
    improve model performance by capturing various aspects of the
    data, but also raises the computational complexity and the
    number of parameters.

dropout_rate : float, optional
    Dropout rate for regularization. Default is ``0.1``. Controls
    the fraction of units dropped out randomly during training.
    Higher values can prevent overfitting but may slow convergence.
    A small to moderate `dropout_rate` (e.g. 0.1 to 0.3) is often
    a good starting point.

output_dim : int, optional
    Dimensionality of the output. Default is ``1``. Determines how
    many target variables are predicted at each forecast horizon.
    For univariate forecasting, `output_dim=1` is typical. For
    multi-variate forecasting, set a larger value to predict
    multiple targets simultaneously.

anomaly_loss_weight : float, optional
    Weight of the anomaly loss term. Default is ``1.0``. Balances
    the contribution of anomaly detection against the primary
    forecasting task. A higher value emphasizes identifying and
    penalizing anomalies, potentially improving robustness to
    irregularities in the data, while a lower value prioritizes
    general forecasting performance.

attention_units : int, optional
    Number of units in attention layers. Default is ``32``.
    Controls the dimensionality of internal representations in
    attention mechanisms. More `attention_units` can allow the
    model to represent more complex dependencies, but may also
    increase risk of overfitting and computation.

hidden_units : int, optional
    Number of units in hidden layers. Default is ``64``. Influences
    the capacity of various dense layers within the model, such as
    those processing static features or for residual connections.
    More units allow modeling more intricate functions, but can
    lead to overfitting if not regularized.

lstm_units : int or None, optional
    Number of units in LSTM layers. Default is ``64``. If `None`,
    LSTM layers may be disabled or replaced with another mechanism.
    Increasing `lstm_units` improves the model’s ability to capture
    temporal dependencies, but also raises computational cost and
    potential overfitting.

scales : list of int, str or None, optional
    Scales for multi-scale LSTM. If ``'auto'``, defaults are chosen
    internally. This parameter configures multiple LSTMs to operate
    at different temporal resolutions. For example, `[1, 7, 30]`
    might represent daily, weekly, and monthly scales. Multi-scale
    modeling can enhance the model’s understanding of hierarchical
    time structures and seasonalities.

multi_scale_agg : str or None, optional
    Aggregation method for multi-scale outputs. Options:
    ``'last'``, ``'average'``, ``'flatten'``, ``'auto'``. If `None`,
    no special aggregation is applied. This parameter determines
    how the multiple scales’ outputs are combined. For instance,
    `average` can produce a more stable representation by averaging
    across scales, while `flatten` preserves all scale information
    in a concatenated form.

activation : str or callable, optional
    Activation function. Default is ``'relu'``. Common choices
    include ``'tanh'``, ``'elu'``, or a custom callable. The choice
    of activation affects the model’s nonlinearity and can influence
    convergence speed and final accuracy. For complex datasets,
    experimenting with different activations may yield better
    results.

use_residuals : bool, optional
    Whether to use residual connections. Default is ``True``.
    Residuals help in stabilizing and speeding up training by
    allowing gradients to flow more easily through the model and
    mitigating vanishing gradients. They also enable deeper model
    architectures without significant performance degradation.

use_batch_norm : bool, optional
    Whether to use batch normalization. Default is ``False``.
    Batch normalization can accelerate training by normalizing
    layer inputs, reducing internal covariate shift. It often makes
    model training more stable and can improve convergence,
    especially in deeper architectures. However, it adds complexity
    and may not always be beneficial.

final_agg : str, optional
    Final aggregation of the time window. Options:
    ``'last'``, ``'average'``, ``'flatten'``. Default is ``'last'``.
    Determines how the time-windowed representations are reduced
    into a final vector before decoding into forecasts. For example,
    `last` takes the most recent time step’s feature vector, while
    `average` merges information across the entire window. Choosing
    a suitable aggregation can influence forecast stability and
    sensitivity to recent or aggregate patterns.

**kwargs : dict
    Additional keyword arguments passed to the model. These may
    include configuration options for layers, optimizers, or
    training routines not covered by the parameters above.


Methods
-------
call(inputs, training=False)
    Perform the forward pass through the model. Given a tuple
    ``(static_input, dynamic_input, future_covariate_input)``,
    it processes all features through embeddings, LSTMs, and
    attention mechanisms before producing final forecasts.
    
    - ``static_input``: 
      A tensor of shape :math:`(B, D_{static})` representing 
      the static features. These do not vary with time.
    - ``dynamic_input``: 
      A tensor of shape :math:`(B, T, D_{dynamic})` representing
      dynamic features across :math:`T` time steps. These include
      historical values and time-dependent covariates.
    - ``future_covariate_input``: 
      A tensor of shape :math:`(B, T, D_{future})` representing
      future-known features, aiding multi-horizon forecasting.

    Depending on the presence of quantiles:
    - If ``quantiles`` is not `None`: 
      The output shape is :math:`(B, H, Q, D_{out})`, where 
      :math:`H` is `forecast_horizons`, :math:`Q` is the number of
      quantiles, and :math:`D_{out}` is `output_dim`.
    - If ``quantiles`` is `None`: 
      The output shape is :math:`(B, H, D_{out})`, providing a 
      deterministic forecast for each horizon.

    Parameters
    ----------
    inputs : tuple of tf.Tensor
        Input tensors `(static_input, dynamic_input, 
        future_covariate_input)`.
    training : bool, optional
        Whether the model is in training mode (default False).
        In training mode, layers like dropout and batch norm
        behave differently.

    Returns
    -------
    tf.Tensor
        The prediction tensor. Its shape and dimensionality depend
        on the `quantiles` setting. In probabilistic scenarios,
        multiple quantiles are returned. In deterministic mode, 
        a single prediction per horizon is provided.

compute_loss(y_true, y_pred, anomaly_scores)
    Compute the total loss, combining both quantile loss (if 
    `quantiles` is not `None`) and anomaly loss. Quantile loss
    measures forecasting accuracy at specified quantiles, while
    anomaly loss penalizes unusual deviations or anomalies.

    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth targets. Shape: :math:`(B, H, D_{out})`.
    y_pred : tf.Tensor
        Model predictions. If quantiles are present:
        :math:`(B, H, Q, D_{out})`. Otherwise:
        :math:`(B, H, D_{out})`.
    anomaly_scores : tf.Tensor
        Tensor indicating anomaly severity. Its shape typically
        matches `(B, H, D_{dynamic})` or a related dimension.

    Returns
    -------
    tf.Tensor
        A scalar tensor representing the combined loss. Lower 
        values indicate better performance, balancing accuracy
        and anomaly handling.

anomaly_loss(anomaly_scores)
    Compute the anomaly loss component. This term encourages the
    model to be robust against abnormal patterns in the data.
    Higher anomaly scores lead to higher loss, prompting the model
    to adjust predictions or representations to reduce anomalies.

    Parameters
    ----------
    anomaly_scores : tf.Tensor
        A tensor reflecting the presence and intensity of anomalies.
        Its shape often corresponds to time steps and dynamic 
        features, e.g., `(B, H, D_{dynamic})`.

    Returns
    -------
    tf.Tensor
        A scalar tensor representing the anomaly loss. Minimizing
        this term encourages the model to learn patterns that 
        mitigate anomalies and produce more stable forecasts.

Notes
------
Consider a batch of time series data. Let:

- :math:`\mathbf{x}_{static} \in \mathbb{R}^{B \times D_{static}}`
  represent the static (time-invariant) features, where
  :math:`B` is the batch size and :math:`D_{static}` is the
  dimensionality of static inputs.
  
- :math:`\mathbf{X}_{dynamic} \in \mathbb{R}^{B \times T \times D_{dynamic}}`
  represent the dynamic (time-varying) features over :math:`T` time steps.
  Here, :math:`D_{dynamic}` corresponds to the dimensionality of
  dynamic inputs (e.g., historical observations).

- :math:`\mathbf{X}_{future} \in \mathbb{R}^{B \times T \times D_{future}}`
  represent the future known covariates, also shaped by
  :math:`T` steps and :math:`D_{future}` features. These may
  include planned events or predicted conditions known ahead of time.

The model first embeds dynamic and future features via multi-modal
embeddings, producing a unified representation:
.. math::
   \mathbf{E}_{dyn} = \text{MultiModalEmbedding}\left(
   [\mathbf{X}_{dynamic}, \mathbf{X}_{future}]\right)

To capture temporal dependencies at various resolutions, multi-scale
LSTMs are applied. These can process data at different temporal scales:
.. math::
   \mathbf{H}_{lstm} = \text{MultiScaleLSTM}(\mathbf{E}_{dyn})

Multiple attention mechanisms enhance the model’s representational
capacity:

1. Hierarchical attention focuses on both short-term and long-term
   interactions between dynamic and future features:
   .. math::
      \mathbf{H}_{hier} = \text{HierarchicalAttention}\left(
      [\mathbf{X}_{dynamic}, \mathbf{X}_{future}]\right)

2. Cross attention integrates information from different modalities
   or embedding spaces, here linking original dynamic inputs and
   their embeddings:
   .. math::
      \mathbf{H}_{cross} = \text{CrossAttention}\left(
      [\mathbf{X}_{dynamic}, \mathbf{E}_{dyn}]\right)

3. Memory-augmented attention incorporates an external memory for
   referencing distant past patterns not directly present in the
   current window:
   .. math::
      \mathbf{H}_{mem} = \text{MemoryAugmentedAttention}(\mathbf{H}_{hier})

Next, static embeddings :math:`\mathbf{E}_{static}` (obtained from
processing static inputs) are combined with the outputs from LSTMs
and attention mechanisms:
.. math::
   \mathbf{H}_{combined} = \text{Concatenate}\left(
   [\mathbf{E}_{static}, \mathbf{H}_{lstm}, \mathbf{H}_{mem},
   \mathbf{H}_{cross}]\right)

The combined representation is decoded into multi-horizon forecasts:
.. math::
   \mathbf{Y}_{decoder} = \text{MultiDecoder}(\mathbf{H}_{combined})

For probabilistic forecasting, quantile distribution modeling

transforms the decoder outputs into quantile predictions:
.. math::
   \mathbf{Y}_{quantiles} = \text{QuantileDistributionModeling}\left(
   \mathbf{Y}_{decoder}\right)

The final predictions are thus:
.. math::
   \hat{\mathbf{Y}} = \mathbf{Y}_{quantiles}

The loss function incorporates both quantile loss for probabilistic
forecasting and anomaly loss for robust handling of irregularities:
.. math::
   \mathcal{L} = \mathcal{L}_{quantile} + \lambda \mathcal{L}_{anomaly}

By adjusting :math:`\lambda`, the model can balance predictive
accuracy against robustness to anomalies.

Furthermore: 
    
- Multi-modal embeddings and multi-scale LSTMs enable the model to
  represent complex temporal patterns at various resolutions.
- Attention mechanisms (hierarchical, cross, memory-augmented)
  enrich the context and allow the model to focus on relevant
  aspects of the data.
- Quantile modeling provides probabilistic forecasts, supplying
  uncertainty intervals rather than single-point predictions.
- Techniques like residual connections, normalization, and
  anomaly loss weighting improve training stability and
  model robustness.

Examples
--------
>>> from gofast.nn.transformers import XTFT
>>> import tensorflow as tf
>>> model = XTFT(
...     static_input_dim=10,
...     dynamic_input_dim=45,
...     future_covariate_dim=5,
...     forecast_horizons=3,
...     quantiles=[0.1, 0.5, 0.9],
...     scales='auto',
...     final_agg='last'
... )
>>> batch_size = 32
>>> time_steps = 20
>>> static_input = tf.random.normal([batch_size, 10])
>>> dynamic_input = tf.random.normal([batch_size, time_steps, 45])
>>> future_covariate_input = tf.random.normal([batch_size, time_steps, 5])
>>> output = model([static_input, dynamic_input, future_covariate_input])
>>> output.shape
TensorShape([32, 3, 3, 1])

See Also
--------
gofast.nn.transformers.TemporalFusionTransformer : 
    The original TFT model for comparison.
MultiHeadAttention : Keras layer for multi-head attention.
LSTM : Keras LSTM layer for sequence modeling.

References
----------
.. [1] Wang, X., et al. (2021). "Enhanced Temporal Fusion Transformer
       for Time Series Forecasting." International Journal of
       Forecasting, 37(3), 1234-1245.
"""
