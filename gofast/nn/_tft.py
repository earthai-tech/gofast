# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""Implements the Temporal Fusion Transformer (TFT), a state-of-the-art 
architecture for multi-horizon time-series forecasting.
"""
from numbers import Real, Integral  

from .._gofastlog import gofastlog
from ..api.docstring import DocstringComponents
from ..api.property import  NNLearner 
from ..core.checks import is_iterable
from ..core.handlers import param_deprecated_message 
from ..compat.sklearn import validate_params, Interval, StrOptions 
from ..decorators import Appender 
from ..utils.deps_utils import ensure_pkg
from ..utils.validator import validate_quantiles
from . import KERAS_DEPS, KERAS_BACKEND, dependency_message
from ._nn_docs import _shared_nn_params, _shared_docs  
 
if KERAS_BACKEND:
    LSTM = KERAS_DEPS.LSTM
    LayerNormalization = KERAS_DEPS.LayerNormalization 
    TimeDistributed = KERAS_DEPS.TimeDistributed
    MultiHeadAttention = KERAS_DEPS.MultiHeadAttention
    Model = KERAS_DEPS.Model 
    BatchNormalization = KERAS_DEPS.BatchNormalization
    Input = KERAS_DEPS.Input
    Softmax = KERAS_DEPS.Softmax
    Flatten = KERAS_DEPS.Flatten
    Dropout = KERAS_DEPS.Dropout 
    Dense = KERAS_DEPS.Dense
    Embedding =KERAS_DEPS.Embedding 
    Concatenate=KERAS_DEPS.Concatenate 
    Layer = KERAS_DEPS.Layer 
    register_keras_serializable=KERAS_DEPS.register_keras_serializable
    
    tf_reduce_sum = KERAS_DEPS.reduce_sum
    tf_stack = KERAS_DEPS.stack
    tf_expand_dims = KERAS_DEPS.expand_dims
    tf_tile = KERAS_DEPS.tile
    tf_range=KERAS_DEPS.range 
    tf_concat = KERAS_DEPS.concat
    tf_shape = KERAS_DEPS.shape
    
    from . import Activation 
    from ._tensor_validation import validate_tft_inputs

DEP_MSG = dependency_message('transformers.tft') 

__all__ = ["TemporalFusionTransformer", "NTemporalFusionTransformer"]

_param_docs = DocstringComponents.from_nested_components(
    base=DocstringComponents(_shared_nn_params), 
)

# ------------------- TFT components ------------------------------------------

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
        batch_size, seq_len, feature_dim = tf_shape(
            inputs)[0], tf_shape(inputs)[1], tf_shape(inputs)[2]
        position_indices = tf_range(0, seq_len, dtype='float32')
        position_indices = tf_expand_dims(position_indices, axis=0)
        position_indices = tf_expand_dims(position_indices, axis=-1)
        position_encoding = tf_tile(
            position_indices, [batch_size, 1, feature_dim])
        return inputs + position_encoding
    
    def get_config(self):
            config = super().get_config().copy()
            return config
        
@register_keras_serializable('Gofast')
class GatedResidualNetwork(Layer, NNLearner):
    @validate_params({
            "units": [Interval(Integral, 1, None, closed='left')], 
            "dropout_rate": [Interval(Real, 0, 1, closed="both")],
            "use_time_distributed": [bool],
            "activation": [StrOptions({"elu", "relu", "tanh", "sigmoid", "linear"})],
            "use_batch_norm": [bool],
        },
    )
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
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


@register_keras_serializable('Gofast')
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
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(
        self,
        # XXX TODO: remove input_dim
        # input_dim,
        num_inputs,
        units,
        dropout_rate=0.0,
        use_time_distributed=False,
        activation='elu',
        use_batch_norm=False,
        **kwargs
    ):
        super().__init__(**kwargs)
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

        stacked_outputs = tf_stack(variable_outputs, axis=-2)
        self.variable_importances_ = self.variable_importance_dense(stacked_outputs)
        weights = self.softmax(self._variable_importances_)
        outputs = tf_reduce_sum(stacked_outputs * weights, axis=-2)
        return outputs

    def get_config(self):
        config = super().get_config().copy()
        config.update({
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

@register_keras_serializable('Gofast')
class TemporalAttentionLayer(Layer, NNLearner):
    @validate_params({
            "units": [Interval(Integral, 1, None, closed='left')], 
            "num_heads": [Interval(Integral, 1, None, closed='left')],
            "dropout_rate": [Interval(Real, 0, 1, closed="both")],
            "activation": [StrOptions({"elu", "relu", "tanh", "sigmoid", "linear"})],
            "use_batch_norm": [bool],
        },
    )
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
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
        context_expanded = tf_expand_dims(context_vector, axis=1)
        context_expanded = tf_tile(
            context_expanded,
            [1, tf_shape(inputs)[1], 1]
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


@register_keras_serializable('Gofast')
class StaticEnrichmentLayer(Layer, NNLearner):
    @validate_params({
            "units": [Interval(Integral, 1, None, closed='left')], 
            "activation": [StrOptions({"elu", "relu", "tanh", "sigmoid", "linear"})],
            "use_batch_norm": [bool],
        },
    )
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
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
        static_context_expanded = tf_expand_dims(
            static_context_vector,
            axis=1
        )
        static_context_expanded = tf_tile(
            static_context_expanded,
            [1, tf_shape(temporal_features)[1], 1]
        )
        combined = tf_concat(
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

# ------------------- TFT implementation --------------------------------------

@Appender(_shared_docs['tft_math_doc'], join='\n', indents=0)
@param_deprecated_message(
    conditions_params_mappings=[
        {
            'param': 'quantiles',
            'condition': lambda v: v is not None,
            'message': ( 
                "Current version only supports 'quantiles=None'."
                " Resetting quantiles to None."
                ),
            'default': None
        }
    ]
)
@register_keras_serializable('Gofast')
class TemporalFusionTransformer(Model, NNLearner):
    @validate_params({
        "static_input_dim": [Interval(Integral, 1, None, closed='left')], 
        "dynamic_input_dim": [Interval(Integral, 1, None, closed='left')], 
        "hidden_units": [Interval(Integral, 1, None, closed='left')], 
        "num_heads": [Interval(Integral, 1, None, closed='left')],
        "dropout_rate": [Interval(Real, 0, 1, closed="both")],
        "forecast_horizon": [Interval(Integral, 1, None, closed='left')],
        "quantiles": ['array-like', None],
        "activation": [StrOptions({"elu", "relu", "tanh", "sigmoid", "linear", "gelu"})],
        "use_batch_norm": [bool],
        "num_lstm_layers": [Interval(Integral, 1, None, closed='left')],
        "lstm_units": ['array-like', Interval(Integral, 1, None, closed='left'), None]
        },
    )
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(
        self,
        dynamic_input_dim,
        static_input_dim,
        hidden_units=32,
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
            num_inputs=static_input_dim,
            units=hidden_units,
            dropout_rate=dropout_rate,
            activation=self.activation_name,
            use_batch_norm=use_batch_norm
        )
        self.dynamic_var_sel = VariableSelectionNetwork(
            num_inputs=dynamic_input_dim,
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
            self.lstm_units = is_iterable(self.lstm_units, transform =True)
            
        for i in range(num_lstm_layers):
            if self.lstm_units is not None:
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
     
        static_inputs, dynamic_inputs = validate_tft_inputs(
            inputs,
            dynamic_input_dim= self.dynamic_input_dim,
            static_input_dim=self.static_input_dim, 
          )

        # Variable Selection
        self.logger.debug("Applying Variable Selection on Static Inputs...")
        static_embedding = self.static_var_sel(
            static_inputs, training=training
        )
        self.static_variable_importances_=self.static_var_sel.variable_importances_ 
        
        self.logger.debug("Applying Variable Selection on Dynamic Inputs...")
        dynamic_embedding = self.dynamic_var_sel(
            dynamic_inputs, training=training
        )
        
        self.dynamic_variable_importances_= self.dynamic_var_sel.variable_importances_ 
        
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
    
@register_keras_serializable('Gofast')
class NTemporalFusionTransformer(Model, NNLearner):
    """
    TemporalFusionTransformer model implementation for multi-horizon 
    forecasting, with optional static, past, and future inputs.

    This class extends Keras `Model` and integrates with the gofast 
    NNLearner interface. It supports dynamic (past) inputs, optional 
    static inputs, and newly added optional future inputs 
    (``future_input_dim``). By including the future covariates, the 
    TemporalFusionTransformer can account for known future features 
    (e.g., events, planned discount rates, etc.) in its predictions.

    Parameters
    ----------
    dynamic_input_dim: int
        Dimensionality of the dynamic (past) inputs. This is mandatory 
        for the TFT model.
    static_input_dim : int, optional
        Dimensionality of static inputs. If not `None`, the call method
        will expect static inputs.
    future_input_dim : int, optional
        Dimensionality of future (known) inputs. If not `None`, the call
        method will expect future inputs to handle exogenous covariates
        known in the future (e.g., events, planned promotions, etc.).
    hidden_units : int, default=32
        Number of hidden units for the layers that do not have a distinct 
        specification (e.g., GRNs, variable selection networks).
    num_heads : int, default=4
        Number of attention heads in the multi-head attention layer.
    dropout_rate : float, default=0.1
        Dropout rate for various layers (GRNs, attention, etc.).
    forecast_horizons : int, default=1
        Number of timesteps to forecast into the future.
    quantiles : list of float, optional
        List of quantiles for probabilistic forecasting. If `None`, a 
        single deterministic output is produced.
    activation : str, default='elu'
        Activation function. Must be one of ``{'elu', 'relu', 'tanh', 
        'sigmoid', 'linear', 'gelu'}``.
    use_batch_norm : bool, default=False
        Whether to apply batch normalization in various sub-layers.
    num_lstm_layers : int, default=1
        Number of LSTM layers in the encoder.
    lstm_units : list of int or None, default=None
        If provided, each index corresponds to the number of LSTM 
        units for that layer. If `None`, uses ``hidden_units`` for 
        each layer.

    Examples
    --------
    >>> from gofast.nn._tensor_validation import validate_tft_inputs
    >>> from gofast.nn.tft import TemporalFusionTransformer
    >>> model = TemporalFusionTransformer(
    ...     dynamic_input_dim=10,
    ...     static_input_dim=5,
    ...     future_input_dim=8,
    ...     hidden_units=32,
    ...     num_heads=4,
    ...     dropout_rate=0.1,
    ...     forecast_horizons=7,
    ...     quantiles=[0.1, 0.5, 0.9],
    ...     activation='elu',
    ...     use_batch_norm=True,
    ...     num_lstm_layers=2,
    ...     lstm_units=[64, 32]
    ... )

    Notes
    -----
    The newly added ``future_input_dim`` allows the model to incorporate 
    future covariates known at forecast time. In the ``call`` method, if 
    ``future_input_dim`` is not `None`, the model expects three inputs:
    ``(static_inputs, dynamic_inputs, future_inputs)``. Otherwise, it 
    expects only ``(static_inputs, dynamic_inputs)``.

    See Also
    --------
    VariableSelectionNetwork : For feature selection and embedding.
    GatedResidualNetwork : A GRN used in various sub-layers.
    LSTM : Keras LSTM layers for sequence processing.

    References
    ----------
    .. [1] Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2019).
           Temporal Fusion Transformers for Interpretable
           Multi-horizon Time Series Forecasting.
           https://arxiv.org/abs/1912.09363
    """

    @validate_params({
        "dynamic_input_dim": [Interval(Integral, 1, None, closed='left')],
        "static_input_dim" : [Interval(Integral, 1, None, closed='left'), None],
        "hidden_units"     : [Interval(Integral, 1, None, closed='left'), None],
        "num_heads"        : [Interval(Integral, 1, None, closed='left')],
        "dropout_rate"     : [Interval(Real, 0, 1, closed="both")],
        "forecast_horizons": [Interval(Integral, 1, None, closed='left')],
        "quantiles"        : ['array-like', None],
        "activation"       : [StrOptions({"elu", "relu", "tanh", "sigmoid",
                                          "linear", "gelu"})],
        "use_batch_norm"   : [bool],
        "num_lstm_layers"  : [Interval(Integral, 1, None, closed='left')],
        "lstm_units"       : ['array-like', Interval(Integral, 1, None, 
                                     closed='left'), None]
    })
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(
        self,
        dynamic_input_dim,
        static_input_dim=None,
        future_input_dim=None,
        hidden_units=32,
        num_heads=4,
        dropout_rate=0.1,
        forecast_horizons=1,
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
            "Initializing NTemporalFusionTransformer with parameters: "
            f"static_input_dim={static_input_dim}, "
            f"dynamic_input_dim={dynamic_input_dim}, "
            f"future_input_dim={future_input_dim}, "
            f"hidden_units={hidden_units}, "
            f"num_heads={num_heads}, "
            f"dropout_rate={dropout_rate}, "
            f"forecast_horizons={forecast_horizons}, "
            f"quantiles={quantiles}, "
            f"activation={activation}, "
            f"use_batch_norm={use_batch_norm}, "
            f"num_lstm_layers={num_lstm_layers}, "
            f"lstm_units={lstm_units}"
        )

        self.static_input_dim  = static_input_dim
        self.dynamic_input_dim = dynamic_input_dim
        self.future_input_dim  = future_input_dim
        self.hidden_units      = hidden_units
        self.num_heads         = num_heads
        self.dropout_rate      = dropout_rate
        self.forecast_horizons = forecast_horizons
        self.quantiles         = quantiles
        self.use_batch_norm    = use_batch_norm
        self.num_lstm_layers   = num_lstm_layers
        self.lstm_units        = lstm_units

        # Convert string activation to a Keras Activation layer 
        # for uniform usage across sub-layers.
        self.activation      = Activation(activation)
        self.activation_name = self.activation.activation_name

        # If quantiles are not provided, interpret as single output 
        # (deterministic). Otherwise, parse and store them.
        if quantiles is None:
            self.quantiles = None
            self.num_quantiles = 1
        else:
            self.quantiles = validate_quantiles(quantiles)
            self.num_quantiles = len(self.quantiles)

        # Initialize variable selection networks for static, dynamic, 
        # and optionally future inputs.
        self.logger.debug("Initializing Variable Selection Networks...")

        # For static inputs (metadata)
        self.static_var_sel = (
            VariableSelectionNetwork(
                num_inputs=static_input_dim,
                units=hidden_units,
                dropout_rate=dropout_rate,
                activation=self.activation_name,
                use_batch_norm=use_batch_norm
            ) if static_input_dim else None
        )

        # For dynamic (past) inputs
        self.dynamic_var_sel = VariableSelectionNetwork(
            num_inputs=dynamic_input_dim,
            units=hidden_units,
            dropout_rate=dropout_rate,
            use_time_distributed=True,
            activation=self.activation_name,
            use_batch_norm=use_batch_norm
        )

        # For future inputs (if needed)
        if self.future_input_dim is not None:
            self.logger.debug("Initializing Future Variable Selection Network...")
            self.future_var_sel = VariableSelectionNetwork(
                num_inputs=future_input_dim,
                units=hidden_units,
                dropout_rate=dropout_rate,
                use_time_distributed=True,
                activation=self.activation_name,
                use_batch_norm=use_batch_norm
            )
        else:
            self.future_var_sel = None

        # Positional Encoding handles the time step embedding 
        # for the dynamic/future sequences.
        self.logger.debug("Initializing Positional Encoding...")
        self.positional_encoding = _PositionalEncoding()

        # Static Context GRNs are used to derive static context vectors
        # and enrichment vectors from static inputs (if present).
        self.logger.debug("Initializing Static Context GRNs...")
        self.static_context_grn = (
            GatedResidualNetwork(
                hidden_units,
                dropout_rate,
                activation=self.activation_name,
                use_batch_norm=use_batch_norm
            ) if static_input_dim else None
        )
        self.static_context_enrichment_grn = (
            GatedResidualNetwork(
                hidden_units,
                dropout_rate,
                activation=self.activation_name,
                use_batch_norm=use_batch_norm
            ) if static_input_dim else None
        )

        # LSTM Encoder: multi-layer LSTMs to encode the historical 
        # (plus possibly future) time series embeddings.
        self.logger.debug("Initializing LSTM Encoder Layers...")
        self.lstm_layers = []
        if self.lstm_units is not None:
            # Convert user-supplied LSTM units to a list if not already.
            self.lstm_units = is_iterable(self.lstm_units, transform=True)
        for i in range(num_lstm_layers):
            lstm_units_i = hidden_units
            if self.lstm_units is not None and i < len(self.lstm_units):
                lstm_units_i = self.lstm_units[i]
            self.lstm_layers.append(
                LSTM(
                    lstm_units_i,
                    return_sequences=True,
                    dropout=dropout_rate,
                    name=f'lstm_layer_{i+1}'
                )
            )

        # Static Enrichment: merges static context with the LSTM-encoded 
        # dynamic embeddings prior to attention.
        self.logger.debug("Initializing Static Enrichment Layer...")
        self.static_enrichment = (
            StaticEnrichmentLayer(
                hidden_units,
                activation=self.activation_name,
                use_batch_norm=use_batch_norm
            ) if static_input_dim else None
        )

        # Temporal Attention Layer for interpretability and weighting 
        # various time steps. 
        self.logger.debug("Initializing Temporal Attention Layer...")
        self.temporal_attention = TemporalAttentionLayer(
            hidden_units,
            num_heads,
            dropout_rate,
            activation=self.activation_name,
            use_batch_norm=use_batch_norm
        )

        # Position-wise Feedforward (GRN) for final transformation 
        # after attention.
        self.logger.debug("Initializing Position-wise Feedforward Network...")
        self.positionwise_grn = GatedResidualNetwork(
            hidden_units,
            dropout_rate,
            use_time_distributed=True,
            activation=self.activation_name,
            use_batch_norm=use_batch_norm
        )

        # Output Layers for either multiple quantiles or a single 
        # deterministic point forecast.
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
        """
        The main forward pass for NTemporalFusionTransformer.

        1. Validate and unpack `inputs` using `validate_tft_inputs`.
        2. Apply variable selection to static, dynamic, and future inputs.
        3. Perform positional encoding on dynamic+future sequences.
        4. Compute static context vectors if static is present.
        5. Pass through LSTM encoders.
        6. Optionally enrich dynamic with static context.
        7. Temporal attention for interpretable weighting of time steps.
        8. Position-wise feedforward (GRN).
        9. Final slicing (forecast horizon) and output (quantiles or single).

        Parameters
        ----------
        inputs : tuple
            Should contain up to three elements:
            (dynamic_inputs, future_inputs, static_inputs)
            or fewer if not all are provided.
        training : bool, default=False
            Whether in training mode (affects dropout, BN, etc.).

        Returns
        -------
        tf.Tensor
            Final predicted sequences of shape 
            (batch_size, forecast_horizons, num_quantiles or 1).
        """
        self.logger.debug("Starting call method with inputs.")

        # Use the validation function to unify shapes and optionally
        # convert them to tf.float32. The function returns 
        # dynamic_inputs, future_inputs, static_inputs in that order.
        dynamic_inputs, future_inputs, static_inputs = validate_tft_inputs(
            inputs= inputs,
            dynamic_input_dim = self.dynamic_input_dim,
            future_covariate_dim= self.future_input_dim,
            static_input_dim = self.static_input_dim
        )

        # 1. Apply Variable Selection on static (if present).
        self.logger.debug("Applying Variable Selection on Static Inputs (if any).")
        if self.static_input_dim and self.static_var_sel:
            static_embedding = self.static_var_sel(static_inputs, training=training)
            self.static_variable_importances_ = self.static_var_sel.variable_importances_
        else:
            static_embedding = None

        # 2. Variable Selection on dynamic (past) inputs.
        self.logger.debug("Applying Variable Selection on Dynamic Inputs...")
        dynamic_embedding = self.dynamic_var_sel(dynamic_inputs, training=training)
        self.dynamic_variable_importances_ = self.dynamic_var_sel.variable_importances_

        # 3. If future inputs exist, apply future var selection and 
        #    concatenate with dynamic.
        if self.future_input_dim and self.future_var_sel:
            self.logger.debug("Applying Variable Selection on Future Inputs...")
            fut_embed = self.future_var_sel(future_inputs, training=training)
            self.future_variable_importances_ = self.future_var_sel.variable_importances_
            dynamic_embedding = tf.concat([dynamic_embedding, fut_embed], axis=1)

        # 4. Positional encoding for combined sequence embedding
        #    (dynamic + future).
        self.logger.debug("Applying Positional Encoding to dynamic/future Embedding...")
        dynamic_embedding = self.positional_encoding(dynamic_embedding)

        # 5. Compute static context if static_embedding is available.
        if static_embedding is not None:
            self.logger.debug("Generating Static Context Vector & Enrichment...")
            static_context_vector = self.static_context_grn(
                static_embedding, training=training
                )
            static_enrichment_vector = self.static_context_enrichment_grn(
                static_embedding, training=training
                )
        else:
            static_context_vector = None
            static_enrichment_vector = None

        # 6. Pass embeddings through multi-layer LSTM encoders.
        self.logger.debug("Passing through LSTM Encoder Layers...")
        x = dynamic_embedding
        for lstm_layer in self.lstm_layers:
            x = lstm_layer(x, training=training)

        # 7. Static enrichment merges the static context with LSTM output 
        #    if static is present.
        if self.static_enrichment and static_enrichment_vector is not None:
            self.logger.debug("Applying Static Enrichment...")
            enriched_lstm_output = self.static_enrichment(
                static_enrichment_vector, x, training=training
            )
        else:
            enriched_lstm_output = x

        # 8. Temporal Attention for interpretable weighting of time steps, 
        #    possibly using static context as well.
        self.logger.debug("Applying Temporal Attention...")
        attention_output = self.temporal_attention(
            enriched_lstm_output,
            context_vector=static_context_vector,
            training=training
        )

        # 9. Position-wise feedforward transforms the attended output.
        self.logger.debug("Applying Position-wise Feedforward...")
        temporal_feature = self.positionwise_grn(
            attention_output, training=training
            )

        # 10. Slice the last `forecast_horizons` steps for final prediction
        #     and apply output projection (either quantiles or point).
        self.logger.debug("Generating Final Output...")
        decoder_steps = self.forecast_horizons
        outputs = temporal_feature[:, -decoder_steps:, :]

        if self.quantiles:
            self.logger.debug("Generating Quantile Outputs...")
            quantile_outs = []
            for i, qout in enumerate(self.quantile_outputs):
                out_i = qout(outputs)
                quantile_outs.append(out_i)
            final_output = Concatenate(
                axis=-1, name='final_quantile_output')(quantile_outs)
        else:
            self.logger.debug("Generating Point Predictions...")
            final_output = self.output_layer(outputs)

        self.logger.debug("Call method completed.")
        
        return final_output

    def get_config(self):
        """
        Return the model configuration for serialization. 
        Includes all hyperparameters that define the structure 
        of the NTemporalFusionTransformer.
        """
        config = super().get_config().copy()
        config.update({
            'dynamic_input_dim' : self.dynamic_input_dim,
            'static_input_dim'  : self.static_input_dim,
            'future_input_dim'  : self.future_input_dim,
            'hidden_units'      : self.hidden_units,
            'num_heads'         : self.num_heads,
            'dropout_rate'      : self.dropout_rate,
            'forecast_horizons' : self.forecast_horizons,
            'quantiles'         : self.quantiles,
            'activation'        : self.activation_name,
            'use_batch_norm'    : self.use_batch_norm,
            'num_lstm_layers'   : self.num_lstm_layers,
            'lstm_units'        : self.lstm_units,
        })
        self.logger.debug("Configuration for get_config has been updated.")
        return config

    @classmethod
    def from_config(cls, config):
        """
        Recreate NTemporalFusionTransformer instance from config.
        This classmethod is invoked by Keras to deserialize the model.
        """
        cls.logger = gofastlog().get_gofast_logger(__name__)
        cls.logger.debug("Creating NTemporalFusionTransformer instance from config.")
        return cls(**config)

TemporalFusionTransformer.__doc__="""\
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


{params.base.hidden_units} 
{params.base.num_heads}
{params.base.dropout_rate} 

forecast_horizons : int, optional
    The number of time steps to forecast. Default is ``1``. This parameter
    defines the number of future time steps the model will predict. For
    multi-step forecasting, set ``forecast_horizons`` to the desired number
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
# num_static_vars : int
#     The number of static variables. Static variables are features that do
#     not change over time, such as location identifiers, categories, or
#     other constants. This parameter indicates how many static variables are
#     being used in the model.

# num_dynamic_vars : int
#     The number of dynamic variables. Dynamic variables are features that
#     change over time, such as historical measurements, external
#     influences, or other time-varying data. This parameter indicates how
#     many dynamic variables are being used at each time step.



# We assume these classes/functions come from your gofast package imports
# (VariableSelectionNetwork, GatedResidualNetwork, etc.) and thus
# do not redefine them here.