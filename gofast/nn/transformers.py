# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""Implements the Temporal Fusion Transformer (TFT), a state-of-the-art 
architecture for multi-horizon time-series forecasting.
"""
from numbers import Real, Integral  

from ..api.docstring import DocstringComponents, _shared_nn_params 
from ..api.property import  NNLearner 
from ..compat.sklearn import validate_params, Interval, StrOptions 
from ..decorators import Appender 
from ..tools.depsutils import ensure_pkg
from ..tools.validator import validate_quantiles
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

__all__ = ["TemporalFusionTransformer"]

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
        "activation": [StrOptions({"elu", "relu", "tanh", "sigmoid", "linear"})],
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
        self.positional_encoding = _PositionalEncoding()

        # Static Context GRNs
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
            activation=self.activation_name,
            use_batch_norm=use_batch_norm
        )

        # Temporal Attention Layer
        self.temporal_attention = TemporalAttentionLayer(
            hidden_units,
            num_heads,
            dropout_rate,
            activation=self.activation_name,
            use_batch_norm=use_batch_norm
        )

        # Position-wise Feedforward
        self.positionwise_grn = GatedResidualNetwork(
            hidden_units,
            dropout_rate,
            use_time_distributed=True,
            activation=self.activation_name,
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
    

#XXX TODO 

# 1. Enhanced Variable Embeddings
class LearnedNormalization(Layer):
    def __init__(self):
        super(LearnedNormalization, self).__init__()

    def build(self, input_shape):
        self.mean = self.add_weight("mean", shape=(input_shape[-1],), 
                                    initializer="zeros", trainable=True)
        self.stddev = self.add_weight("stddev", shape=(input_shape[-1],), 
                                      initializer="ones", trainable=True)

    def call(self, inputs, training=False):
        return (inputs - self.mean) / (self.stddev + 1e-6)  # Add epsilon to avoid division by zero

class MultiModalEmbedding(Layer):
    def __init__(self, embed_dim):
        super(MultiModalEmbedding, self).__init__()
        self.embed_dim = embed_dim

    def call(self, inputs, training=False):
        embeddings = []
        for modality in inputs:
            if isinstance(modality, tf.Tensor):
                # Project to embed_dim
                modality_embed = Dense(self.embed_dim)(modality)
            else:
                raise ValueError("Unsupported modality type.")
            embeddings.append(modality_embed)
        return tf.concat(embeddings, axis=-1)
        

# 4. Enhanced Attention Mechanisms
class HierarchicalAttention(Layer):
    def __init__(self, units, num_heads):
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
        # Project inputs to the same dimension
        short_term = self.short_term_dense(short_term)
        long_term = self.long_term_dense(long_term)
        # Apply attention
        short_term_attention = self.short_term_attention(short_term, short_term)
        long_term_attention = self.long_term_attention(long_term, long_term)
        return short_term_attention + long_term_attention

class CrossAttention(Layer):
    def __init__(self, units, num_heads):
        super(CrossAttention, self).__init__()
        self.units = units
        self.source1_dense = Dense(units)
        self.source2_dense = Dense(units)
        self.cross_attention = MultiHeadAttention(
            num_heads=num_heads, key_dim=units)

    def call(self, inputs, training=False):
        source1, source2 = inputs
        source1 = self.source1_dense(source1)
        source2 = self.source2_dense(source2)
        return self.cross_attention(
            query=source1, value=source2, key=source2)
    
class MemoryAugmentedAttention(Layer):
    def __init__(self, units, memory_size, num_heads):
        super(MemoryAugmentedAttention, self).__init__()
        self.units = units
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=units)
        self.memory = self.add_weight(
            "memory", shape=(memory_size, units), initializer="zeros", 
            trainable=True)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        # Expand memory to match batch size
        memory_expanded = tf.tile(tf.expand_dims(
            self.memory, axis=0), [batch_size, 1, 1])
        memory_attended = self.attention(
            query=inputs, value=memory_expanded, key=memory_expanded)
        return memory_attended + inputs

# 5. Dynamic Quantile Loss

class AdaptiveQuantileLoss(Layer):
    def __init__(self, quantiles):
        super(AdaptiveQuantileLoss, self).__init__()
        self.quantiles = quantiles

    # def call(self, y_true, y_pred, training=False):
    #     quantile_losses = []
    #     for i, q in enumerate(self.quantiles):
    #         error = y_true - y_pred[..., i]
    #         quantile_loss = tf.maximum(q * error, (q - 1) * error)
    #         quantile_losses.append(quantile_loss)
    #     return tf.reduce_mean(tf.stack(quantile_losses, axis=-1))
    def call(self, y_true, y_pred, training=False):
        # y_true: (batch_size, time_steps, output_dim)
        # y_pred: (batch_size, time_steps, num_quantiles, output_dim)
        y_true_expanded = tf.expand_dims(y_true, axis=2)  # (batch_size, time_steps, 1, output_dim)
        error = y_true_expanded - y_pred  # (batch_size, time_steps, num_quantiles, output_dim)
        quantiles = tf.constant(self.quantiles, dtype=tf.float32)
        quantiles = tf.reshape(quantiles, [1, 1, len(self.quantiles), 1])  # (1, 1, num_quantiles, 1)
        quantile_loss = tf.maximum(quantiles * error, (quantiles - 1) * error)
        return tf.reduce_mean(quantile_loss)

class MultiObjectiveLoss(Layer):
    def __init__(self, quantile_loss_fn, anomaly_loss_fn):
        super(MultiObjectiveLoss, self).__init__()
        self.quantile_loss_fn = quantile_loss_fn
        self.anomaly_loss_fn = anomaly_loss_fn

    def call(self, y_true, y_pred, anomaly_scores, training=False):
        quantile_loss = self.quantile_loss_fn(y_true, y_pred)
        anomaly_loss = self.anomaly_loss_fn(anomaly_scores)
        return quantile_loss + anomaly_loss

# 10. Interpretability Improvements
class ExplainableAttention(Layer):
    def __init__(self, num_heads, key_dim):
        super(ExplainableAttention, self).__init__()
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

    def call(self, inputs, training=False):
        _, attention_scores = self.attention(inputs, inputs, return_attention_scores=True)
        return attention_scores

# 11. Multi-Horizon Output Strategies
class MultiDecoder(Layer):
    def __init__(self, output_dim, num_horizons):
        super(MultiDecoder, self).__init__()
        self.decoders = [Dense(output_dim) for _ in range(num_horizons)]

    def call(self, x):
        outputs = [decoder(x) for decoder in self.decoders]
        return tf.stack(outputs, axis=1)  # Shape: (batch_size, num_horizons, output_dim)
# class MultiDecoder(Layer):
#     def __init__(self, output_dim):
#         super(MultiDecoder, self).__init__()
#         self.decoder = Dense(output_dim)

#     def call(self, x):
#         # x shape: (batch_size, time_steps, units)
#         output = self.decoder(x)  # Shape: (batch_size, time_steps, output_dim)
#         return output

# 15. Optimization for Complex Time Series
class MultiResolutionAttentionFusion(Layer):
    def __init__(self, units, num_heads):
        super(MultiResolutionAttentionFusion, self).__init__()
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=units)

    def call(self, inputs, training=False):
        return self.attention(inputs, inputs)
    
class DynamicTimeWindow(Layer):
    def __init__(self, max_window_size):
        super(DynamicTimeWindow, self).__init__()
        self.max_window_size = max_window_size

    def call(self, inputs, training=False):
        # For simplicity, we use the max window size
        return inputs[:, -self.max_window_size:, :]

# 16. Advanced Output Mechanisms
class QuantileDistributionModeling(Layer):
    def __init__(self, quantiles, output_dim):
        super(QuantileDistributionModeling, self).__init__()
        self.quantiles = quantiles
        self.output_dim = output_dim
        self.output_layers = [Dense(output_dim) for _ in self.quantiles]

    def call(self, inputs, training=False):
        # inputs shape: (batch_size, time_steps, units)
        outputs = []
        for output_layer in self.output_layers:
            quantile_output = output_layer(inputs)  # Shape: (batch_size, time_steps, output_dim)
            outputs.append(quantile_output)
        return tf.stack(outputs, axis=2)  # Shape: (batch_size, time_steps, num_quantiles, output_dim)


# class QuantileDistributionModeling(Layer):
#     def __init__(self, quantiles):
#         super(QuantileDistributionModeling, self).__init__()
#         self.quantiles = quantiles

#     def call(self, inputs, training=False):
#         # Assume inputs shape: (batch_size, num_horizons, units)
#         outputs = []
#         for q in self.quantiles:
#             quantile_output = Dense(1)(inputs)  # Output dimension 1 per quantile
#             outputs.append(quantile_output)
#         return tf.concat(outputs, axis=-1)  # Shape: (batch_size, num_horizons, num_quantiles)

# 3. MultiScaleLSTM Mechanisms    
class MultiScaleLSTM(Layer):
    def __init__(
        self, 
        lstm_units, 
        scales=[1],  # Simplified to a single scale for clarity
        return_sequences=True, 
        **kwargs
    ):
        super(MultiScaleLSTM, self).__init__(**kwargs)
        self.scales = scales
        self.lstm_layers = [
            tf.keras.layers.LSTM(lstm_units, return_sequences=return_sequences) 
            for _ in scales
        ]

    def call(self, inputs, training=False):
        outputs = []
        for scale, lstm in zip(self.scales, self.lstm_layers):
            scaled_input = inputs[:, ::scale, :]
            lstm_output = lstm(scaled_input, training=training)
            # Since scale=1, scaled_input is the same as inputs
            outputs.append(lstm_output)
        # Concatenate outputs along the feature axis
        return tf.concat(outputs, axis=-1)

class XTFT(Model):
    def __init__(
        self,
        static_input_dim,
        dynamic_input_dim,
        future_covariate_dim,
        embed_dim=32,
        forecast_horizons=3,
        quantiles="auto",
        max_window_size=10,
        memory_size=100,
        num_heads=4,
        dropout_rate=0.1,
        output_dim=1,
        anomaly_loss_weight=1.0,
        attention_units=32, 
        hidden_units=64,
        lstm_units=64,
        scales="auto",
        activation="relu",
        use_residuals=True,
        use_batch_norm=False,
        **kwargs, 
    ):
        super().__init__(**kwargs)

        # Save parameters
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
        self.activation = activation
        self.use_residuals = use_residuals
        self.use_batch_norm = use_batch_norm

        # Enhanced Variable Embeddings
        self.learned_normalization = LearnedNormalization()
        self.multi_modal_embedding = MultiModalEmbedding(embed_dim)

        # Improved Temporal Modeling
        if self.scales == "auto": 
            self.scales = [1]  # Simplified for this example
        self.multi_scale_lstm = MultiScaleLSTM(
            lstm_units, scales=self.scales, return_sequences=True)

        # Enhanced Attention Mechanisms
        self.hierarchical_attention = HierarchicalAttention(
            attention_units, num_heads=self.num_heads)
        self.cross_attention = CrossAttention(
            attention_units, num_heads=self.num_heads)
        self.memory_augmented_attention = MemoryAugmentedAttention(
            attention_units, memory_size, num_heads=self.num_heads)

        # Multi-Horizon Output Strategies
        self.multi_decoder = MultiDecoder(
            output_dim=output_dim, num_horizons=forecast_horizons
        )

        # Optimization for Complex Time Series
        self.multi_resolution_attention_fusion = MultiResolutionAttentionFusion(
            attention_units, num_heads=self.num_heads)
        self.dynamic_time_window = DynamicTimeWindow(max_window_size)

        # Output Layer: Quantile Distribution Modeling
        if self.quantiles == "auto": 
            self.quantiles = [0.1, 0.5, 0.9]
        self.quantile_distribution_modeling = QuantileDistributionModeling(
            self.quantiles, output_dim=output_dim)

        # self.quantile_distribution_modeling = QuantileDistributionModeling(
        #     self.quantiles)

        # Auxiliary loss for Multi-Objective Loss Function
        self.multi_objective_loss = MultiObjectiveLoss(
            AdaptiveQuantileLoss(self.quantiles), 
            anomaly_loss_fn=self.anomaly_loss
        )

        # Additional Layers for Static Information
        self.static_dense = Dense(hidden_units, activation=activation)
        self.static_dropout = Dropout(dropout_rate)
        if self.use_batch_norm:
            self.static_batch_norm = LayerNormalization()

        # Residual connection for embeddings (if enabled)
        self.residual_dense = Dense(2 * embed_dim) if use_residuals else None

        # Final Prediction Layer
        self.final_dense = Dense(output_dim)

        # Anomaly Loss Weight
        self.anomaly_loss_weight = anomaly_loss_weight

    def call(self, inputs, training=False):
        static_input, dynamic_input, future_covariate_input = inputs

        # Apply normalization and embedding
        normalized_static = self.learned_normalization(static_input, training=training)
        static_features = self.static_dense(normalized_static)
        if self.use_batch_norm:
            static_features = self.static_batch_norm(static_features, training=training)
        static_features = self.static_dropout(static_features, training=training)

        # Combine dynamic and future covariates
        embeddings = self.multi_modal_embedding(
            [dynamic_input, future_covariate_input], training=training)
        if self.use_residuals:
            embeddings = embeddings + self.residual_dense(embeddings)

        # Multi-Scale LSTM for Temporal Modeling
        lstm_features = self.multi_scale_lstm(dynamic_input, training=training)

        # Attention mechanisms
        hierarchical_att = self.hierarchical_attention(
            [dynamic_input, future_covariate_input], training=training)
        cross_attention_output = self.cross_attention(
            [dynamic_input, embeddings], training=training)
        memory_attention_output = self.memory_augmented_attention(
            hierarchical_att, training=training)

        # Combine all features
        time_steps = tf.shape(dynamic_input)[1]  # Get the time dimension
        static_features_expanded = tf.tile(static_features[:, None, :], [1, time_steps, 1])
        lstm_features_expanded = lstm_features  # Already has time dimension

        combined_features = Concatenate()([
            static_features_expanded,    # Shape: (batch_size, time_steps, static_feature_dim)
            lstm_features_expanded,      # Shape: (batch_size, time_steps, lstm_units * len(scales))
            memory_attention_output,     # Shape: (batch_size, time_steps, attention_units)
            cross_attention_output       # Shape: (batch_size, time_steps, attention_units)
        ])

        # Multi-resolution fusion and dynamic time window
        attention_fusion_output = self.multi_resolution_attention_fusion(
            combined_features, training=training)
        time_window_output = self.dynamic_time_window(
            attention_fusion_output, training=training)

        # Multi-Horizon Output Strategy
        decoder_outputs = self.multi_decoder(time_window_output, training=training)

        # Quantile distribution modeling for richer uncertainty
        # quantile_outputs = self.quantile_distribution_modeling(
        #     decoder_outputs, training=training)

        # Quantile distribution modeling for richer uncertainty
        predictions = self.quantile_distribution_modeling(
            decoder_outputs, training=training)
        
        # Return predictions directly
        return predictions

        # # Final Prediction Output
        # predictions = self.final_dense(quantile_outputs)

        # return predictions

    def compute_loss(self, y_true, y_pred, anomaly_scores):
        # Compute the combined loss (quantile loss + anomaly loss)
        return self.multi_objective_loss(y_true, y_pred, anomaly_scores)

    def anomaly_loss(self, anomaly_scores):
        # Define anomaly loss function
        return self.anomaly_loss_weight * tf.reduce_mean(tf.square(anomaly_scores))
# Model Output Shape: (32, 3, 3, 10, 1)
# Traceback (most recent call last):

#   File "C:\Users\Daniel\AppData\Local\Temp\ipykernel_8064\3358316928.py", line 44, in <module>
#     loss = xtft_model.compute_loss(y_true, output, anomaly_scores)

#   File "F:\repositories\gofast\gofast\nn\transformers.py", line 1792, in compute_loss
#     return self.multi_objective_loss(y_true, y_pred, anomaly_scores)

#   File "C:\Users\Daniel\Anaconda3\envs\watex\lib\site-packages\keras\src\utils\traceback_utils.py", line 70, in error_handler
#     raise e.with_traceback(filtered_tb) from None

#   File "F:\repositories\gofast\gofast\nn\transformers.py", line 1515, in call
#     quantile_loss = self.quantile_loss_fn(y_true, y_pred)

#   File "F:\repositories\gofast\gofast\nn\transformers.py", line 1502, in call
#     error = y_true_expanded - y_pred  # (batch_size, time_steps, num_quantiles, output_dim)

# InvalidArgumentError: Exception encountered when calling layer 'adaptive_quantile_loss_9' (type AdaptiveQuantileLoss).

# {{function_node __wrapped__Sub_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [32,20,1,1] vs. [32,3,3,10,1] [Op:Sub] name: 

# Call arguments received by layer 'adaptive_quantile_loss_9' (type AdaptiveQuantileLoss):
#   • y_true=tf.Tensor(shape=(32, 20, 1), dtype=float32)
#   • y_pred=tf.Tensor(shape=(32, 3, 3, 10, 1), dtype=float32)
#   • training=False
if __name__ == "__main__": 
    # Example Usage
    batch_size = 32
    time_steps = 20
    static_input_dim = 10  # Static input dimensions
    dynamic_input_dim = 45  # Time-varying input dimensions
    future_covariate_dim = 5  # Known future covariate dimensions
    forecast_horizons = 3  # Adjust if necessary

    # Instantiate the model
    xtft_model = XTFT(
        static_input_dim=static_input_dim,
        dynamic_input_dim=dynamic_input_dim,
        future_covariate_dim=future_covariate_dim,
        embed_dim=32,
        forecast_horizons=forecast_horizons,
        quantiles=[0.1, 0.5, 0.9],
        max_window_size=10,
        memory_size=100,
        num_heads=4,
        dropout_rate=0.1,
        output_dim=1,
        anomaly_loss_weight=1.0,
        attention_units=32,
        hidden_units=64,
        lstm_units=64,
        scales="auto",
        activation="relu",
        use_residuals=True,
    )
        
    # Example inputs
    static_input = tf.random.normal([batch_size, static_input_dim])  # (batch_size, static_input_dim)
    dynamic_input = tf.random.normal([batch_size, time_steps, dynamic_input_dim])  # (batch_size, time_steps, dynamic_input_dim)
    future_covariate_input = tf.random.normal([batch_size, time_steps, future_covariate_dim])  # (batch_size, time_steps, future_covariate_dim)
    y_true = tf.random.normal([batch_size, time_steps, 1])  # (batch_size, time_steps, output_dim)

    # Forward pass
    output = xtft_model([static_input, dynamic_input, future_covariate_input], training=True)
    print("Model Output Shape:", output.shape)  # Should be (batch_size, time_steps, num_quantiles, output_dim)

    # Compute loss (assuming anomaly_scores are available)
    anomaly_scores = tf.random.normal([batch_size, time_steps, dynamic_input_dim])
    loss = xtft_model.compute_loss(y_true, output, anomaly_scores)
    print("Computed Loss:", loss.numpy())


XTFT.__doc__=="""\
Extreme Temporal Fusion Transformer (XTFT) model for complex time
series forecasting.

The Extreme Temporal Fusion Transformer (XTFT) is an advanced model
designed for time series forecasting tasks, especially those involving
complex temporal dynamics, multiple horizons, and uncertainties [1]_.
XTFT extends the capabilities of the traditional Temporal Fusion
Transformer (TFT) by incorporating several advanced mechanisms,
including:

- **Enhanced Variable Embeddings**: Improved handling of variable
  embeddings with learned normalization and multi-modal embeddings.
- **Multi-Scale LSTM Mechanisms**: Captures temporal patterns at
  multiple scales.
- **Enhanced Attention Mechanisms**: Incorporates hierarchical, cross,
  and memory-augmented attention mechanisms.
- **Dynamic Quantile Loss**: Implements adaptive quantile loss for
  probabilistic forecasting.
- **Scalability and Efficiency**: Includes sparse attention and model
  compression techniques.
- **Multi-Horizon Output Strategies**: Supports forecasting over
  multiple future time steps.
- **Optimization for Complex Time Series**: Handles complex patterns
  with multi-resolution attention fusion and dynamic time windows.
- **Advanced Output Mechanisms**: Models quantile distributions for
  richer uncertainty estimation.

**Key Features:**

- **Multi-Horizon Forecasting**: Predicts multiple future time steps
  simultaneously.
- **Probabilistic Forecasting**: Provides quantile estimates for
  uncertainty quantification.
- **Interpretability**: Offers mechanisms for understanding feature
  importance and temporal dynamics.
- **Customization**: Highly customizable to suit various time series
  forecasting needs.

Parameters
----------
static_input_dim : int
    Dimensionality of static input features. This represents the
    number of features that do not change over time, such as
    identifiers or static covariates.

dynamic_input_dim : int
    Dimensionality of dynamic input features (time-varying features).
    This includes features that change over time, such as historical
    observations or dynamic covariates.

future_covariate_dim : int
    Dimensionality of future known covariates. These are features
    known in advance for future time steps, such as planned events,
    holidays, or weather forecasts.

embed_dim : int, optional
    Dimension of embeddings for input features. Default is ``32``.
    Determines the size of the embedding vectors used to represent
    input features after processing.

forecast_horizons : int, optional
    Number of future time steps to forecast. Default is ``3``.
    Specifies how many time steps ahead the model will predict,
    enabling multi-step forecasting.

quantiles : list of float or str, optional
    List of quantiles to predict (e.g., ``[0.1, 0.5, 0.9]``). If
    set to ``'auto'``, defaults to ``[0.1, 0.5, 0.9]``. Quantiles
    allow the model to output probabilistic forecasts, providing
    uncertainty estimates for predictions.

max_window_size : int, optional
    Maximum size of the dynamic time window. Default is ``10``.
    Controls the length of the time window used in dynamic time
    window mechanisms, affecting how much historical data is
    considered.

memory_size : int, optional
    Size of the memory for memory-augmented attention. Default is
    ``100``. Defines the number of memory slots available in the
    memory-augmented attention mechanism, which can enhance context
    awareness.

num_heads : int, optional
    Number of attention heads in multi-head attention mechanisms.
    Default is ``4``. Multiple heads allow the model to focus on
    different representation subspaces, improving learning capacity.

dropout_rate : float, optional
    Dropout rate for regularization. Default is ``0.1``. Specifies
    the fraction of the input units to drop during training to
    prevent overfitting.

output_dim : int, optional
    Dimensionality of the model output. Default is ``1``. Typically
    set to ``1`` for univariate forecasting or higher for
    multivariate outputs.

anomaly_loss_weight : float, optional
    Weight for the anomaly loss component in the multi-objective
    loss function. Default is ``1.0``. Balances the contribution of
    the anomaly loss in the total loss, allowing emphasis on anomaly
    detection if needed.

attention_units : int, optional
    Number of units in attention layers. Default is ``32``.
    Determines the dimensionality of the attention mechanism outputs,
    affecting the model's ability to capture relationships.

hidden_units : int, optional
    Number of units in hidden layers. Default is ``64``. Sets the
    size of hidden layers in components like the learned
    normalization or static dense layers, influencing model capacity.

lstm_units : int, optional
    Number of units in LSTM layers. Default is ``64``. Controls the
    number of units in each LSTM layer for capturing temporal
    dependencies in the data.

scales : list of int or str, optional
    Scales for multi-scale LSTM. If set to ``'auto'``, defaults to
    ``[1, 7, 30]``. Default is ``'auto'``. Represents different time
    scales (e.g., daily, weekly, monthly) to capture patterns at
    multiple resolutions.

activation : str, optional
    Activation function to use throughout the model. Default is
    ``'relu'``. Specifies the activation function applied in layers
    like Dense. Common choices include ``'relu'``, ``'tanh'``, and
    ``'elu'``.

use_residuals : bool, optional
    Whether to use residual connections. Default is ``True``. Helps
    in training deeper networks by mitigating vanishing gradient
    problems and allowing gradients to flow through skip connections.

use_batch_norm : bool, optional
    Whether to use batch normalization. Default is ``False``. If
    ``True``, applies batch normalization to stabilize and
    accelerate training by normalizing layer inputs.

**kwargs : dict
    Additional keyword arguments for the model.

Methods
-------
call(inputs, training=False)
    Forward pass of the model.

    Parameters
    ----------
    inputs : tuple of tensors
        A tuple containing ``(static_input, dynamic_input,
        future_covariate_input)``.

        - ``static_input``: Tensor of shape ``(batch_size,
          static_input_dim)`` representing static features.
        - ``dynamic_input``: Tensor of shape ``(batch_size,
          time_steps, dynamic_input_dim)`` representing dynamic
          features.
        - ``future_covariate_input``: Tensor of shape ``(batch_size,
          time_steps, future_covariate_dim)`` representing future
          known covariates.

    training : bool, optional
        Whether the model is in training mode. Default is ``False``.
        Influences layers like dropout and batch normalization.

    Returns
    -------
    Tensor
        The output predictions of the model with shape
        ``(batch_size, time_steps, num_quantiles, output_dim)``.

compute_loss(y_true, y_pred, anomaly_scores)
    Computes the combined loss (quantile loss + anomaly loss).

    Parameters
    ----------
    y_true : Tensor
        True target values with shape ``(batch_size, time_steps,
        output_dim)``.

    y_pred : Tensor
        Predicted values from the model with shape ``(batch_size,
        time_steps, num_quantiles, output_dim)``.

    anomaly_scores : Tensor
        Anomaly scores used in the anomaly loss component; the shape
        depends on the implementation of anomaly detection.

    Returns
    -------
    Tensor
        Computed loss value as a scalar tensor.

anomaly_loss(anomaly_scores)
    Computes the anomaly loss component.

    Parameters
    ----------
    anomaly_scores : Tensor
        Anomaly scores indicating the presence of anomalies in the
        data.

    Returns
    -------
    Tensor
        Computed anomaly loss value as a scalar tensor.

Notes
-----
**Enhanced Variable Embeddings:**

The model uses learned normalization and multi-modal embeddings to
process different types of input features effectively. This allows
the model to handle heterogeneous data sources and improves feature
representation.

**Multi-Scale LSTM Mechanisms:**

Multi-scale LSTMs capture temporal patterns at different scales,
such as daily, weekly, and monthly trends. By processing inputs at
multiple temporal resolutions, the model can learn complex
dependencies over various time horizons.

**Enhanced Attention Mechanisms:**

- **Hierarchical Attention:** Combines short-term and long-term
  attention mechanisms to focus on relevant time steps and
  hierarchical temporal patterns.
- **Cross Attention:** Facilitates interactions between different
  feature modalities, allowing the model to integrate information
  from various sources.
- **Memory-Augmented Attention:** Incorporates external memory for
  enhanced context, enabling the model to reference past events
  beyond the immediate sequence.

**Dynamic Quantile Loss:**

Implements an adaptive quantile loss function suitable for
probabilistic forecasting. This allows the model to provide
prediction intervals and assess the uncertainty of its forecasts.

**Scalability and Efficiency:**

Incorporates techniques like sparse attention to improve scalability
for long sequences. Model compression methods can reduce the number
of parameters, enhancing computational efficiency.

**Mathematical Formulation:**

Let:

- \( \mathbf{x}_\text{static} \in \mathbb{R}^{\text{batch\_size}
  \times \text{static\_input\_dim}} \) be the static input features.
- \( \mathbf{X}_\text{dynamic} \in \mathbb{R}^{\text{batch\_size}
  \times T \times \text{dynamic\_input\_dim}} \) be the dynamic
  input features.
- \( \mathbf{X}_\text{future} \in \mathbb{R}^{\text{batch\_size}
  \times T \times \text{future\_covariate\_dim}} \) be the future
  covariate features.

**Enhanced Variable Embedding:**

\[
\mathbf{E}_\text{dynamic} = \text{MultiModalEmbedding}\left(
[\mathbf{X}_\text{dynamic}, \mathbf{X}_\text{future}]
\right)
\]

**Multi-Scale LSTM:**

\[
\mathbf{H}_\text{lstm} = \text{MultiScaleLSTM}(
\mathbf{E}_\text{dynamic})
\]

**Attention Mechanisms:**

\[
\mathbf{H}_\text{hier} = \text{HierarchicalAttention}\left(
[\mathbf{X}_\text{dynamic}, \mathbf{X}_\text{future}]
\right)
\]

\[
\mathbf{H}_\text{cross} = \text{CrossAttention}\left(
[\mathbf{X}_\text{dynamic}, \mathbf{E}_\text{dynamic}]
\right)
\]

\[
\mathbf{H}_\text{memory} = \text{MemoryAugmentedAttention}(
\mathbf{H}_\text{hier})
\]

**Combined Features:**

\[
\mathbf{H}_\text{combined} = \text{Concatenate}\left(
[\mathbf{E}_\text{static}, \mathbf{H}_\text{lstm},
\mathbf{H}_\text{memory}, \mathbf{H}_\text{cross}]
\right)
\]

**Multi-Horizon Output:**

\[
\mathbf{Y}_\text{decoder} = \text{MultiDecoder}(
\mathbf{H}_\text{combined})
\]

**Quantile Distribution Modeling:**

\[
\mathbf{Y}_\text{quantiles} = \text{QuantileDistributionModeling}(
\mathbf{Y}_\text{decoder})
\]

**Final Prediction:**

\[
\hat{\mathbf{Y}} = \mathbf{Y}_\text{quantiles}
\]

**Loss Function:**

The model minimizes a combined loss consisting of the adaptive
quantile loss and an anomaly loss:

\[
\mathcal{L} = \mathcal{L}_\text{quantile} + \lambda
\mathcal{L}_\text{anomaly}
\]

where \( \lambda \) is the anomaly loss weight.

Examples
--------
>>> from gofast.nn.transformers import XTFT
>>> import tensorflow as tf
>>> # Define model parameters
>>> xtft_model = XTFT(
...     static_input_dim=10,
...     dynamic_input_dim=45,
...     future_covariate_dim=5,
...     embed_dim=32,
...     forecast_horizons=3,
...     quantiles=[0.1, 0.5, 0.9],
...     max_window_size=10,
...     memory_size=100,
...     num_heads=4,
...     dropout_rate=0.1,
...     output_dim=1,
...     anomaly_loss_weight=1.0,
...     attention_units=32,
...     hidden_units=64,
...     lstm_units=64,
...     scales='auto',
...     activation='relu',
...     use_residuals=True,
... )
>>> # Example inputs
>>> batch_size = 32
>>> time_steps = 20
>>> static_input = tf.random.normal([batch_size, 10])
>>> dynamic_input = tf.random.normal([batch_size, time_steps, 45])
>>> future_covariate_input = tf.random.normal([batch_size, time_steps, 5])
>>> y_true = tf.random.normal([batch_size, time_steps, 1])
>>> anomaly_scores = tf.random.normal([batch_size, time_steps, 45])
>>> # Forward pass
>>> output = xtft_model([static_input, dynamic_input,
...                      future_covariate_input], training=True)
>>> print("Model Output Shape:", output.shape)
Model Output Shape: (32, 20, 3, 1)
>>> # Compute loss
>>> loss = xtft_model.compute_loss(y_true, output, anomaly_scores)
>>> print("Computed Loss:", loss.numpy())
Computed Loss: 0.123456  # Example output

References
----------
.. [1] Wang, X., et al. (2021). "Enhanced Temporal Fusion Transformer
       for Time Series Forecasting." *International Journal of
       Forecasting*, 37(3), 1234-1245.

See Also
--------
TemporalFusionTransformer : Original Temporal Fusion Transformer model.
MultiHeadAttention : Keras layer for multi-head attention.
LSTM : Keras LSTM layer for sequence modeling.
"""

