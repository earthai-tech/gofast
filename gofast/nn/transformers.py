# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""Implements the Temporal Fusion Transformer (TFT), a state-of-the-art 
architecture for multi-horizon time-series forecasting.
"""
from numbers import Real 
from ..compat.sklearn import validate_params, Interval  
from ..tools.depsutils import ensure_pkg
from . import KERAS_DEPS, KERAS_BACKEND, dependency_message

if KERAS_BACKEND:
    LSTM=KERAS_DEPS.LSTM
    reshape = KERAS_DEPS.reshape
    Dense = KERAS_DEPS.Dense
    reduce_sum = KERAS_DEPS.reduce_sum
    Softmax = KERAS_DEPS.Softmax
    Flatten = KERAS_DEPS.Flatten
    Dropout=KERAS_DEPS.Dropout 
    stack=KERAS_DEPS.stack
    Layer=KERAS_DEPS.Layer 
    ELU=KERAS_DEPS.ELU 
    LayerNormalization=KERAS_DEPS.LayerNormalization 
    TimeDistributed=KERAS_DEPS.TimeDistributed
    MultiHeadAttention=KERAS_DEPS.MultiHeadAttention
    expand_dims=KERAS_DEPS.expand_dims
    tile=KERAS_DEPS.tile
    concat=KERAS_DEPS.concat
    shape=KERAS_DEPS.shape
    Model=KERAS_DEPS.Model 
    
DEP_MSG=dependency_message('tune') 
    
__all__= ["TemporalFusionTransformer"]


class GatedResidualNetwork(Layer):
    def __init__(
        self,
        units,
        dropout_rate=0.0,
        use_time_distributed=False
    ):
        super(GatedResidualNetwork, self).__init__()
        self.use_time_distributed = use_time_distributed
        self.units = units

        self.linear = Dense(units)
        self.elu = ELU()
        self.linear2 = Dense(units)
        self.dropout = Dropout(dropout_rate)
        self.layer_norm = LayerNormalization()
        self.gate = Dense(
            units,
            activation='sigmoid'
        )

    def call(self, x):
        shortcut = x
        if self.use_time_distributed:
            x = TimeDistributed(self.linear)(x)
            x = TimeDistributed(self.elu)(x)
            x = TimeDistributed(self.linear2)(x)
        else:
            x = self.linear(x)
            x = self.elu(x)
            x = self.linear2(x)

        x = self.dropout(x)
        gate = self.gate(x)
        x = x * gate
        x = x + shortcut
        x = self.layer_norm(x)
        return x

class VariableSelectionNetwork(Layer):
    def __init__(
        self,
        input_dim,
        num_inputs,
        units,
        dropout_rate=0.0,
        use_time_distributed=False
    ):
        super(VariableSelectionNetwork, self).__init__()
        self.use_time_distributed = use_time_distributed
        self.num_inputs = num_inputs
        self.units = units

        self.flatten = Flatten()
        self.softmax = Softmax(axis=-1)
        self.single_variable_grns = [
            GatedResidualNetwork(
                units,
                dropout_rate,
                use_time_distributed
            )
            for _ in range(num_inputs)
        ]

    def call(self, inputs):
        # inputs shape: (batch_size, [time_steps,] num_inputs, input_dim)
        variable_outputs = []
        for i in range(self.num_inputs):
            if self.use_time_distributed:
                var_input = inputs[:, :, i, :]
            else:
                var_input = inputs[:, i, :]
            grn_output = self.single_variable_grns[i](var_input)
            variable_outputs.append(grn_output)

        # Stack and compute variable importance
        stacked_outputs = stack(variable_outputs, axis=-2)
        # Compute variable weights
        weights = self.softmax(self.flatten(stacked_outputs))
        weights = reshape(
            weights,
            (-1, self.num_inputs, self.units)
        )
        # Apply weights
        outputs = reduce_sum(
            stacked_outputs * weights,
            axis=-2
        )
        return outputs

class TemporalAttentionLayer(Layer):
    def __init__(
        self,
        units,
        num_heads,
        dropout_rate=0.0
    ):
        super(TemporalAttentionLayer, self).__init__()
        self.units = units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.multi_head_attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=units
        )
        self.dropout = Dropout(dropout_rate)
        self.layer_norm = LayerNormalization()
        self.grn = GatedResidualNetwork(
            units,
            dropout_rate,
            use_time_distributed=True
        )

    def call(self, inputs):
        # inputs shape: (batch_size, time_steps, units)
        attn_output = self.multi_head_attention(
            query=inputs,
            value=inputs,
            key=inputs
        )
        attn_output = self.dropout(attn_output)
        # Residual connection and layer normalization
        x = self.layer_norm(inputs + attn_output)
        # Apply GRN
        output = self.grn(x)
        return output

class StaticEnrichmentLayer(Layer):
    def __init__(self, units):
        super(StaticEnrichmentLayer, self).__init__()
        self.units = units
        self.grn = GatedResidualNetwork(units)

    def call(self, static_context_vector, temporal_features):
        # static_context_vector shape: (batch_size, units)
        # temporal_features shape: (batch_size, time_steps, units)
        # Expand static context to time dimension
        static_context_expanded = expand_dims(
            static_context_vector,
            axis=1
        )
        static_context_expanded = tile(
            static_context_expanded,
            [1, shape(temporal_features)[1], 1]
        )
        # Combine and pass through GRN
        combined = concat(
            [static_context_expanded, temporal_features],
            axis=-1
        )
        output = self.grn(combined)
        return output

@ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
class TemporalFusionTransformer(Model):
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
        The input dimension for static variables. This is typically the
        embedding size for static categorical variables or 1 for continuous
        variables.
    dynamic_input_dim : int
        The input dimension for dynamic variables.
    num_static_vars : int
        The number of static variables.
    num_dynamic_vars : int
        The number of dynamic variables.
    hidden_units : int
        The number of hidden units in the model.
    num_heads : int
        The number of attention heads used in the temporal attention layer.
    dropout_rate : float, optional
        The dropout rate used in the dropout layers. Default is ``0.1``.

    Methods
    -------
    call(inputs)
        Forward pass of the model.

    Notes
    -----
    The TFT model combines the strengths of sequence-to-sequence models and
    attention mechanisms to handle complex temporal dynamics. It provides
    interpretability by allowing the examination of variable importance and
    temporal attention weights.

    **Variable Selection Networks (VSNs):**

    VSNs select relevant variables by applying Gated Residual Networks (GRNs)
    to each variable and computing variable importance weights.

    **Gated Residual Networks (GRNs):**

    GRNs allow the model to capture complex nonlinear relationships while
    controlling information flow via gating mechanisms.

    **Static Enrichment Layer:**

    Enriches temporal features with static context, enabling the model to
    adjust temporal dynamics based on static information.

    **Temporal Attention Layer:**

    Applies multi-head attention over the temporal dimension to focus on
    important time steps.

    **Mathematical Formulation:**

    Let :math:`\mathbf{x}_{\text{static}} \in \mathbb{R}^{n_s \times d_s}` be the
    static inputs and :math:`\mathbf{x}_{\text{dynamic}} \in\\
        \mathbb{R}^{T \times n_d \times d_d}` be the
    dynamic inputs, where :math:`n_s` and :math:`n_d` are the numbers of static and
    dynamic variables, :math:`d_s` and :math:`d_d` are their respective input
    dimensions, and :math:`T` is the number of time steps.

    The VSNs compute:

    .. math::
        \mathbf{e}_{\text{static}} = \sum_{i=1}^{n_s} \alpha_i \cdot
        \text{GRN}(\mathbf{x}_{\text{static}, i})

    .. math::
        \mathbf{E}_{\text{dynamic}} = \sum_{j=1}^{n_d} \beta_j \cdot
        \text{GRN}(\mathbf{x}_{\text{dynamic}, :, j})

    where :math:`\alpha_i` and :math:`\beta_j` are variable importance weights
    computed via softmax.

    The LSTM Encoder processes :math:`\mathbf{E}_{\text{dynamic}}` to capture
    sequential dependencies:

    .. math::
        \mathbf{H} = \text{LSTM}(\mathbf{E}_{\text{dynamic}})

    The Static Enrichment Layer combines static context with temporal features:

    .. math::
        \mathbf{H}_{\text{enriched}} = \text{StaticEnrichment}(
        \mathbf{e}_{\text{static}}, \mathbf{H})

    Temporal Attention is applied to focus on important time steps:

    .. math::
        \mathbf{Z} = \text{TemporalAttention}(\mathbf{H}_{\text{enriched}})

    The Position-wise Feedforward Layer refines the output:

    .. math::
        \mathbf{F} = \text{GRN}(\mathbf{Z})

    The final output is produced:

    .. math::
        \hat{y} = \text{OutputLayer}(\mathbf{F}_{T})

    where :math:`\mathbf{F}_{T}` is the feature vector at the last time step.

    Examples
    --------
    >>> from gofast.nn.transformers import TemporalFusionTransformer
    >>> model = TemporalFusionTransformer(
    ...     static_input_dim=1,
    ...     dynamic_input_dim=1,
    ...     num_static_vars=2,
    ...     num_dynamic_vars=5,
    ...     hidden_units=64,
    ...     num_heads=4,
    ...     dropout_rate=0.1
    ... )
    >>> model.compile(optimizer='adam', loss='mse')
    >>> # Assume `static_inputs` and `dynamic_inputs` are prepared
    >>> model.fit(
    ...     [static_inputs, dynamic_inputs],
    ...     labels,
    ...     epochs=10,
    ...     batch_size=32
    ... )

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
    @validate_params({
        "static_input_dim": [int], 
        "dynamic_input_dim": [int], 
        "num_static_vars": [int], 
        "hidden_units": [int], 
        "num_heads": [int], 
        "dropout_rate": [Interval( Real, 0, 1, closed="both")]
        }
    )
    def __init__(
        self,
        static_input_dim,
        dynamic_input_dim,
        num_static_vars,
        num_dynamic_vars,
        hidden_units,
        num_heads,
        dropout_rate=0.1
    ):
        super(TemporalFusionTransformer, self).__init__()
        self.hidden_units = hidden_units

        # Variable Selection Networks
        self.static_var_sel = VariableSelectionNetwork(
            input_dim=static_input_dim,
            num_inputs=num_static_vars,
            units=hidden_units,
            dropout_rate=dropout_rate
        )
        self.dynamic_var_sel = VariableSelectionNetwork(
            input_dim=dynamic_input_dim,
            num_inputs=num_dynamic_vars,
            units=hidden_units,
            dropout_rate=dropout_rate,
            use_time_distributed=True
        )

        # Static Context GRNs
        self.static_context_grn = GatedResidualNetwork(
            hidden_units,
            dropout_rate
        )
        self.static_context_enrichment_grn = GatedResidualNetwork(
            hidden_units,
            dropout_rate
        )

        # LSTM Encoder
        self.lstm_encoder = LSTM(
            hidden_units,
            return_sequences=True
        )

        # Static Enrichment Layer
        self.static_enrichment = StaticEnrichmentLayer(
            hidden_units
        )

        # Temporal Attention Layer
        self.temporal_attention = TemporalAttentionLayer(
            hidden_units,
            num_heads,
            dropout_rate
        )

        # Position-wise Feedforward
        self.positionwise_grn = GatedResidualNetwork(
            hidden_units,
            dropout_rate,
            use_time_distributed=True
        )

        # Output Layer
        self.output_layer = Dense(1)

    def call(self, inputs):
        """
        Forward pass of the Temporal Fusion Transformer model.

        Parameters
        ----------
        inputs : tuple of tensors
            A tuple ``(static_inputs, dynamic_inputs)`` where:
            - ``static_inputs`` is a tensor of shape
              ``(batch_size, num_static_vars, static_input_dim)``.
            - ``dynamic_inputs`` is a tensor of shape
              ``(batch_size, time_steps, num_dynamic_vars, dynamic_input_dim)``.

        Returns
        -------
        output : tensor
            The model output tensor of shape ``(batch_size, 1)``.

        """
        static_inputs, dynamic_inputs = inputs

        # Variable Selection
        static_embedding = self.static_var_sel(
            static_inputs
        )  # Shape: (batch_size, hidden_units)
        dynamic_embedding = self.dynamic_var_sel(
            dynamic_inputs
        )  # Shape: (batch_size, time_steps, hidden_units)

        # Static Context Vectors
        static_context_vector = self.static_context_grn(
            static_embedding
        )
        static_enrichment_vector = self.static_context_enrichment_grn(
            static_embedding
        )

        # LSTM Encoder
        lstm_output = self.lstm_encoder(
            dynamic_embedding
        )  # Shape: (batch_size, time_steps, hidden_units)

        # Static Enrichment
        enriched_lstm_output = self.static_enrichment(
            static_enrichment_vector,
            lstm_output
        )

        # Temporal Attention
        attention_output = self.temporal_attention(
            enriched_lstm_output
        )

        # Position-wise Feedforward
        temporal_feature = self.positionwise_grn(
            attention_output
        )

        # Final Output
        output = self.output_layer(
            temporal_feature[:, -1, :]
        )  # Use last time step for prediction

        return output





