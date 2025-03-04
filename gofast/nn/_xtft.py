# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""Implements the Extreme Temporal Fusion Transformer (XTFT), a state-of-the-art 
architecture for multi-horizon time-series forecasting.
"""
from textwrap import dedent 
from numbers import Real, Integral  
from typing import List, Optional, Union, Dict, Any  

from .._gofastlog import gofastlog
from ..api.docs import doc 
from ..api.property import NNLearner 
from ..compat.sklearn import validate_params, Interval, StrOptions 
from ..core.handlers import param_deprecated_message 
from ..utils.deps_utils import ensure_pkg
from ..decorators import Appender , Deprecated

from . import KERAS_DEPS, KERAS_BACKEND, dependency_message
from ._nn_docs import _shared_docs 

if KERAS_BACKEND:
    LSTM = KERAS_DEPS.LSTM
    Dense = KERAS_DEPS.Dense
    Flatten = KERAS_DEPS.Flatten
    Dropout = KERAS_DEPS.Dropout 
    Layer = KERAS_DEPS.Layer 
    LayerNormalization = KERAS_DEPS.LayerNormalization 
    MultiHeadAttention = KERAS_DEPS.MultiHeadAttention
    Model = KERAS_DEPS.Model 
    Input = KERAS_DEPS.Input
    Concatenate=KERAS_DEPS.Concatenate 
    Tensor=KERAS_DEPS.Tensor
    register_keras_serializable=KERAS_DEPS.register_keras_serializable
    
    tf_reduce_sum = KERAS_DEPS.reduce_sum
    tf_stack = KERAS_DEPS.stack
    tf_expand_dims = KERAS_DEPS.expand_dims
    tf_tile = KERAS_DEPS.tile
    tf_range_=KERAS_DEPS.range 
    tf_concat = KERAS_DEPS.concat
    tf_shape = KERAS_DEPS.shape
    tf_reshape=KERAS_DEPS.reshape
    tf_add = KERAS_DEPS.add
    tf_maximum = KERAS_DEPS.maximum
    tf_reduce_mean = KERAS_DEPS.reduce_mean
    tf_add_n = KERAS_DEPS.add_n
    tf_float32=KERAS_DEPS.float32
    tf_constant=KERAS_DEPS.constant 
    tf_square=KERAS_DEPS.square 
    tf_autograph=KERAS_DEPS.autograph
    
    tf_autograph.set_verbosity(0)

    
    from ._tensor_validation import validate_anomaly_scores 
    from ._tensor_validation import validate_xtft_inputs 
    
    from .losses import combined_quantile_loss, combined_total_loss
    from .utils import set_default_params, set_anomaly_config 
    from .components import ( 
            AdaptiveQuantileLoss,
            AnomalyLoss,
            CrossAttention,
            DynamicTimeWindow,
            GatedResidualNetwork,
            HierarchicalAttention,
            LearnedNormalization,
            MemoryAugmentedAttention,
            MultiDecoder,
            MultiModalEmbedding,
            MultiObjectiveLoss,
            MultiResolutionAttentionFusion,
            MultiScaleLSTM,
            QuantileDistributionModeling,
            VariableSelectionNetwork,
            PositionalEncoding, 
            aggregate_multiscale 
        )
    
DEP_MSG = dependency_message('transformers') 


@register_keras_serializable('gofast.nn.transformers', name="XTFT")
@doc (
    key_improvements= dedent(_shared_docs['xtft_key_improvements']), 
    key_functions= dedent(_shared_docs['xtft_key_functions']), 
    methods= dedent( _shared_docs['xtft_methods']
    )
 )
@param_deprecated_message(
    conditions_params_mappings=[
        {
            'param': 'multi_scale_agg',
            'condition': lambda v: v == "concat",
            'message': (
                "The 'concat' mode for multi-scale aggregation requires identical "
                "time dimensions across scales, which is rarely practical. "
                "This mode will fall back to the robust last-timestep approach "
                "in real applications. For true multi-scale handling, use 'last' "
                "mode instead (automatically set).\n"
                "Why change?\n"
                "- 'concat' mixes features across scales at the same timestep\n"
                "- Requires manual time alignment between scales\n" 
                "- 'last' preserves scale independence & handles variable lengths"
            ),
            'default': "last"
        }
    ],
    warning_category=UserWarning
)
class XTFT(Model, NNLearner):
    @validate_params({
        "static_input_dim": [Interval(Integral, 1, None, closed='left')], 
        "dynamic_input_dim": [Interval(Integral, 1, None, closed='left')], 
        "future_input_dim": [Interval(Integral, 1, None, closed='left')], 
        "embed_dim": [Interval(Integral, 1, None, closed='left')],
        "forecast_horizon": [Interval(Integral, 1, None, closed='left')], 
        "quantiles": ['array-like', StrOptions({'auto'}),  None],
        "max_window_size": [Interval(Integral, 1, None, closed='left')],
        "memory_size": [Interval(Integral, 1, None, closed='left')], 
        "num_heads": [Interval(Integral, 1, None, closed='left')],
        "dropout_rate": [Interval(Real, 0, 1, closed="both")],
        "output_dim": [Interval(Integral, 1, None, closed='left')],
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
            StrOptions({"last", "average",  "flatten", "auto", "sum", "concat"}),
            None
        ],
        "scales": ['array-like', StrOptions({"auto"}),  None],
        "use_batch_norm": [bool],
        "use_residuals": [bool],
        "final_agg": [StrOptions({"last", "average",  "flatten"})],
        },
    )
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(
        self,
        static_input_dim: int,
        dynamic_input_dim: int,
        future_input_dim: int,
        embed_dim: int = 32,
        forecast_horizon: int = 1,
        quantiles: Union[str, List[float], None] = None,
        max_window_size: int = 10,
        memory_size: int = 100,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        output_dim: int = 1, 
        anomaly_config: Optional[Dict[str, Any]] = None,  
        attention_units: int = 32,
        hidden_units: int = 64,
        lstm_units: int = 64,
        scales: Union[str, List[int], None] = None,
        multi_scale_agg: Optional[str] = None, 
        activation: str = 'relu',
        use_residuals: bool = True,
        use_batch_norm: bool = False,
        final_agg: str = 'last',
        **kw, 
    ):
        super().__init__(**kw)

        # Initialize Logger
        self.logger = gofastlog().get_gofast_logger(__name__)
        
        #self.activation = Activation(activation) 
        self.activation = activation #.activation
        
        self.logger.debug(
            "Initializing XTFT with parameters: "
            f"static_input_dim={static_input_dim}, "
            f"dynamic_input_dim={dynamic_input_dim}, "
            f"future_input_dim={future_input_dim}, "
            f"embed_dim={embed_dim}, "
            f"forecast_horizon={forecast_horizon}, "
            f"quantiles={quantiles}, "
            f"max_window_size={max_window_size},"
            f" memory_size={memory_size}, num_heads={num_heads}, "
            f"dropout_rate={dropout_rate}, output_dim={output_dim}, "
            f"anomaly_config={None if anomaly_config is None else anomaly_config.keys()}, "
            f"attention_units={attention_units}, "
            f" hidden_units={hidden_units}, "
            f"lstm_units={lstm_units}, "
            f"scales={scales}, "
            f"activation={self.activation}, "
            f"use_residuals={use_residuals}, "
            f"use_batch_norm={use_batch_norm}, "
            f"final_agg={final_agg}"
        )
        # Handle default quantiles, scales and multi_scale_agg 
        quantiles, scales, return_sequences = set_default_params(
            quantiles, scales, multi_scale_agg ) 
        
        self.static_input_dim = static_input_dim
        self.dynamic_input_dim = dynamic_input_dim
        self.future_input_dim = future_input_dim
        self.embed_dim = embed_dim
        self.forecast_horizon = forecast_horizon
        self.quantiles = quantiles
        self.max_window_size = max_window_size
        self.memory_size = memory_size
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.output_dim = output_dim
        self.anomaly_config=set_anomaly_config(anomaly_config)
        self.attention_units = attention_units
        self.hidden_units = hidden_units
        self.lstm_units = lstm_units
        self.scales = scales
        self.multi_scale_agg = multi_scale_agg
        self.use_residuals = use_residuals
        self.use_batch_norm = use_batch_norm
        self.final_agg = final_agg

        # Layers
        self.learned_normalization = LearnedNormalization()
        self.multi_modal_embedding = MultiModalEmbedding(embed_dim)
        
        # Add PositionalEncoding layer
        self.positional_encoding = PositionalEncoding()
        
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
            num_horizons=forecast_horizon
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
        
        self.anomaly_loss_layer = AnomalyLoss(
            weight=self.anomaly_config.get('anomaly_loss_weight', 1.)
            )
        # ---------------------------------------------------------------------
        # The MultiObjectiveLoss encapsulates both quantile and anomaly losses
        # to allow simultaneous training on multiple objectives. While this 
        # functionality can currently be bypassed, note that it may be removed 
        # in a future release. Users who rely on multi-objective training 
        # strategies should keep an eye on upcoming changes.
        # 
        # Here, we instantiate the MultiObjectiveLoss with an adaptive quantile 
        # loss function, which adjusts quantile estimates dynamically based on 
        # the provided quantiles, and an anomaly loss function that penalizes 
        # predictions deviating from expected anomaly patterns.
        # ---------------------------------------------------------------------
        self.multi_objective_loss = MultiObjectiveLoss(
            quantile_loss_fn=AdaptiveQuantileLoss(self.quantiles),
            anomaly_loss_fn=self.anomaly_loss_layer
        )
        # ---------------------------------------------------------------------
 
        self.static_dense = Dense(hidden_units, activation=self.activation)
        self.static_dropout = Dropout(dropout_rate)
        if self.use_batch_norm:
            self.static_batch_norm = LayerNormalization()
            
        # Initialize Gated Residual Networks (GRNs) for attention outputs
        self.grn_static = GatedResidualNetwork(
            units=hidden_units,
            dropout_rate=dropout_rate,
            use_time_distributed=False,
            activation=self.activation,
            use_batch_norm=self.use_batch_norm
        )
    
        self.residual_dense = Dense(2 * embed_dim) if use_residuals else None
        # self.final_dense = Dense(output_dim)
        
    @tf_autograph.experimental.do_not_convert
    def call(self, inputs, training=False, **kwargs):
        static_input , dynamic_input, future_input = validate_xtft_inputs (
            inputs =inputs,
            static_input_dim=self.static_input_dim, 
            dynamic_input_dim= self.dynamic_input_dim, 
            future_covariate_dim= self.future_input_dim, 
        )
  
        # Normalize and process static features
        normalized_static = self.learned_normalization(
            static_input, 
            training=training
        )
        self.logger.debug(
            f"Normalized Static Shape: {normalized_static.shape}"
        )
        # Apply -> GRN pipeline to cross attention
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
        # XXX TODO # check apply --> GRN
        static_features = self.grn_static(
            static_features, 
            training=training
        ) 
        
        # Embeddings for dynamic and future covariates
        embeddings = self.multi_modal_embedding(
            [dynamic_input, future_input],
            training=training
        )
        self.logger.debug(
            f"Embeddings Shape: {embeddings.shape}"
        )
    
        # Add positional encoding to embeddings
        # before attention mechanisms. 
        embeddings = self.positional_encoding(
            embeddings, 
            training=training 
        )  
        
        self.logger.debug(
            f"Embeddings with Positional Encoding Shape: {embeddings.shape}"
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
        lstm_features = aggregate_multiscale(
            lstm_output, mode= self.multi_scale_agg 
        )
        # Since we are concatenating along the time dimension, we need 
        # all tensors to have the same shape along that dimension.
        time_steps = tf_shape(dynamic_input)[1]
        # Expand lstm_features to (B, 1, features)
        lstm_features = tf_expand_dims(lstm_features, axis=1)
        # Tile to match tf_time steps: (B, T, features)
        lstm_features = tf_tile(lstm_features, [1, time_steps, 1])

        self.logger.debug(
            f"LSTM Features Shape: {lstm_features.shape}"
        )
    
        # Attention mechanisms
        hierarchical_att = self.hierarchical_attention(
            [dynamic_input, future_input],
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
        time_steps = tf_shape(dynamic_input)[1]
        static_features_expanded = tf_tile(
            tf_expand_dims(static_features, axis=1),
            [1, time_steps, 1]
        )
        self.logger.debug(
            "Static Features Expanded Shape: "
            f"{static_features_expanded.shape}"
        )
    
        combined_features = Concatenate()([
            static_features_expanded,
            lstm_features,
            cross_attention_output, 
            hierarchical_att,
            memory_attention_output,
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
            final_features = tf_reduce_mean(time_window_output, axis=1)
        else:  # "flatten"
            shape = tf_shape(time_window_output)
            (batch_size,
             time_dim,
             feat_dim) = shape[0], shape[1], shape[2]
            final_features = tf_reshape(
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
        
        # Compute anomaly scores
        self.anomaly_scores= validate_anomaly_scores(
            self.anomaly_config, forecast_horizon= self.forecast_horizon)
        
        self.anomaly_loss_weight= self.anomaly_config.get('anomaly_loss_weight')
        
        if self.anomaly_scores is not None:
            # Use anomaly_scores from anomaly_config
            self.logger.debug(
                "Using Anomaly Scores from anomaly_config"
                f" Shape: {self.anomaly_scores.shape}")
        
            if self.anomaly_loss_weight is None: 
                # Use provided anomaly_scores from anomaly_config
                self.logger.debug(
                    "Using Anomaly Scores from anomaly_config is None."
                    " Ressetting to 1.")
                
                self.anomaly_loss_weight = 1. 

        # compute anomaly scores 
        # Handle anomaly_scores exclusively via anomaly_config
        if self.anomaly_loss_weight is not None:
            # Compute anomaly loss and add it to the total loss
            anomaly_loss = self.anomaly_loss_layer(self.anomaly_scores)
            self.add_loss(self.anomaly_loss_weight * anomaly_loss)
            self.logger.debug(
                f"Anomaly Loss Computed and Added: {anomaly_loss}")
            
        return predictions

    def compile(self, optimizer, loss=None, **kws):
        """
        Compile the XTFT model, allowing an explicit user-specified loss
        to override the defaults.

        If the user provides a loss (loss=...), it is used regardless of
        quantiles or anomaly scores. Otherwise, the method uses the
        following logic:

        - If ``self.quantiles`` is None, defaults to "mse"(or the
          user-supplied ``loss``).
        - If ``self.quantiles`` is not None, uses a quantile-based loss.
          If ``anomaly_scores`` is present, a total loss is used that
          adds anomaly loss on top.
          
        See also:
        --------
        gofast.nn.losses.combined_quantile_loss
        gofast.nn.losses.combined_total_loss
        """
        # 1) If user explicitly provides a loss, respect that and skip defaults
        if loss is not None:
            super().compile(
                optimizer=optimizer,
                loss=loss,
                **kws
            )
            return
    
        # 2) Otherwise, we handle the default logic
        if self.quantiles is None:
            # Deterministic scenario
            super().compile(
                optimizer=optimizer,
                loss="mean_squared_error",
                **kws
            )
            return
    
        # Probabilistic scenario with quantiles
        quantile_loss_fn = combined_quantile_loss(self.quantiles)
    
        if not hasattr(self, "anomaly_scores"):
            self.anomaly_scores = self.anomaly_config.get("anomaly_scores")
    
        # If anomaly scores exist, add anomaly loss to the quantile loss
        if self.anomaly_scores is not None:
            total_loss_fn = combined_total_loss(
                quantiles=self.quantiles,
                anomaly_layer=self.anomaly_loss_layer,
                anomaly_scores=self.anomaly_scores
            )
            super().compile(
                optimizer=optimizer,
                loss=total_loss_fn,
                **kws
            )
        else:
            # Only quantile loss
            super().compile(
                optimizer=optimizer,
                loss=quantile_loss_fn,
                **kws
            )
            
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'static_input_dim': self.static_input_dim,
            'dynamic_input_dim': self.dynamic_input_dim,
            'future_input_dim': self.future_input_dim,
            'embed_dim': self.embed_dim,
            'forecast_horizon': self.forecast_horizon,
            'quantiles': self.quantiles,
            'max_window_size': self.max_window_size,
            'memory_size': self.memory_size,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate,
            'output_dim': self.output_dim,
            'anomaly_config': {
                'anomaly_scores': ( 
                    self.anomaly_scores.numpy() 
                    if self.anomaly_scores is not None else None
                    ),
                'anomaly_loss_weight': self.anomaly_loss_weight
            },
            'attention_units': self.attention_units,
            'hidden_units': self.hidden_units,
            'lstm_units': self.lstm_units,
            'scales': self.scales,
            'activation': self.activation,
            'use_residuals': self.use_residuals,
            'use_batch_norm': self.use_batch_norm,
            'final_agg': self.final_agg, 
            'multi_scale_agg': self.multi_scale_agg, 
        })
        
        self.logger.debug(
            "Configuration for XTFT has been updated in get_config.")
        return config

    @classmethod
    def from_config(cls, config):
        logger = gofastlog().get_gofast_logger(__name__)
        logger.debug("Creating XTFT instance from config.")
        return cls(**config)

@Deprecated(
    "SuperXTFT is currently under maintenance and will be released soon. " 
    "Please stay updated for the upcoming release. For now, use the "
    "standard XTFT instead."
)
@Appender ( dedent( 
    XTFT.__doc__.replace ('XTFT', 'SuperXTFT'),
    ), 
    join='\n', 
)
@register_keras_serializable('gofast.nn.transformers', name="SuperXTFT")
class SuperXTFT(XTFT):
    """
    SuperXTFT: An enhanced version of XTFT with Variable Selection Networks (VSNs) 
    and integrated Gate → Add & Norm → GRN pipeline in attention layers.
    
    """
    def __init__(
        self,
        static_input_dim: int,
        dynamic_input_dim: int,
        future_input_dim: int,
        embed_dim: int = 32,
        forecast_horizon: int = 1,
        quantiles: Union[str, List[float], None] = None,
        max_window_size: int = 10,
        memory_size: int = 100,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        output_dim: int = 1, 
        anomaly_config: Optional[Dict[str, Any]] = None,  
        attention_units: int = 32,
        hidden_units: int = 64,
        lstm_units: int = 64,
        scales: Union[str, List[int], None] = None,
        multi_scale_agg: Optional[str] = 'auto', 
        activation: str = 'relu',
        use_residuals: bool = True,
        use_batch_norm: bool = False,
        final_agg: str = 'last',
        **kw
    ):
        super().__init__(
            static_input_dim=static_input_dim,
            dynamic_input_dim=dynamic_input_dim,
            future_input_dim=future_input_dim,
            embed_dim=embed_dim,
            forecast_horizon=forecast_horizon,
            quantiles=quantiles,
            max_window_size=max_window_size,
            memory_size=memory_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            output_dim=output_dim,
            anomaly_config=anomaly_config,
            attention_units=attention_units,
            hidden_units=hidden_units,
            lstm_units=lstm_units,
            scales=scales,
            multi_scale_agg=multi_scale_agg,
            activation=activation,
            use_residuals=use_residuals,
            use_batch_norm=use_batch_norm,
            final_agg=final_agg,
            **kw, 
        )
        
        self.logger = gofastlog().get_gofast_logger(__name__)
        
        # Initialize Variable Selection Networks (VSNs)
        self.variable_selection_static = VariableSelectionNetwork(
            num_inputs=static_input_dim,  
            units=hidden_units,
            dropout_rate=dropout_rate,
            use_time_distributed=False,
            activation=activation,
            use_batch_norm=use_batch_norm
        )
        
        self.variable_selection_dynamic = VariableSelectionNetwork(
            num_inputs=dynamic_input_dim,  
            units=hidden_units,
            dropout_rate=dropout_rate,
            use_time_distributed=True,
            activation=activation,
            use_batch_norm=use_batch_norm
        )
        
        self.variable_future_covariate = VariableSelectionNetwork(
            num_inputs=future_input_dim,  
            units=hidden_units,
            dropout_rate=dropout_rate,
            use_time_distributed=True,
            activation=activation,
            use_batch_norm=use_batch_norm
        )
        # Add positional encoding 
        self.positional_encoding = PositionalEncoding()
        
        # Initialize Gated Residual Networks (GRNs) for attention outputs
        self.grn_attention_hierarchical = GatedResidualNetwork(
            units=attention_units,
            dropout_rate=dropout_rate,
            use_time_distributed=False,
            activation=activation,
            use_batch_norm=use_batch_norm
        )
        
        self.grn_attention_cross = GatedResidualNetwork(
            units=attention_units,
            dropout_rate=dropout_rate,
            use_time_distributed=False,
            activation=activation,
            use_batch_norm=use_batch_norm
        )
        
        self.grn_memory_attention= GatedResidualNetwork(
            units=attention_units,
            dropout_rate=dropout_rate,
            use_time_distributed=False,
            activation=activation,
            use_batch_norm=use_batch_norm
        )
        
        # Initialize Gate -> Add & Norm -> GRN pipeline for decoder outputs
        self.grn_decoder = GatedResidualNetwork(
            units=output_dim,
            dropout_rate=dropout_rate, 
            use_time_distributed=False,
            activation=activation,
            use_batch_norm=use_batch_norm
        )
        
    @tf_autograph.experimental.do_not_convert
    def call(self, inputs, training=False, **kwargs):
        static_input, dynamic_input, future_input = validate_xtft_inputs(
            inputs=inputs,
            static_input_dim=self.static_input_dim, 
            dynamic_input_dim=self.dynamic_input_dim, 
            future_covariate_dim=self.future_input_dim, 
        )

        # Variable Selection for static, dynamic
        # inputs and future covariate
        selected_static = self.variable_selection_static(
            static_input, training=training)
        selected_dynamic = self.variable_selection_dynamic(
            dynamic_input, training=training)
        selected_future = self.variable_future_covariate(
            future_input, training=training)
        
        self.logger.debug(
            f"Selected Static Features Shape: {selected_static.shape}"
        )
        self.logger.debug(
            f"Selected Dynamic Features Shape: {selected_dynamic.shape}"
        )
        self.logger.debug(
            f"Selected Covariate Features Shape: {selected_future.shape}"
        )
        
        # Proceed with the original XTFT forward pass using selected features
        # Normalize and process static features
        normalized_static = self.learned_normalization(
            selected_static, 
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
        # Embeddings for dynamic and future covariates using selected_dynamic
        embeddings = self.multi_modal_embedding(
            [selected_dynamic, selected_future],
            training=training
        )
        self.logger.debug(
            f"Embeddings Shape: {embeddings.shape}"
        )
        
        # Positional info
        embeddings = self.positional_encoding(
            embeddings, 
            training=training
        )  
        
        self.logger.debug(
            f"Embeddings Shape after Positional Encoding: {embeddings.shape}"
        )
        
        if self.use_residuals:
            embeddings = embeddings + self.residual_dense(embeddings)
            self.logger.debug(
                "Embeddings with Residuals Shape: "
                f"{embeddings.shape}"
            )
        
        # Multi-scale LSTM outputs
        lstm_output = self.multi_scale_lstm(
            selected_dynamic,
            training=training
        )
        # Handle multi_scale_agg as in XTFT
        lstm_features = aggregate_multiscale(
            lstm_output, mode= self.multi_scale_agg 
        )
        
        # Expand and tile lstm_features to match time steps
        time_steps = tf_shape(dynamic_input)[1]
        lstm_features = tf_expand_dims(lstm_features, axis=1)  # (B, 1, features)
        lstm_features = tf_tile(lstm_features, [1, time_steps, 1])  # (B, T, features)
        
        self.logger.debug(
            f"LSTM Features Shape: {lstm_features.shape}"
        )
        
        # Attention mechanisms with integrated GRNs
        hierarchical_att = self.hierarchical_attention(
            [selected_dynamic, future_input],
            training=training
        )
        self.logger.debug(
            f"Hierarchical Attention Shape: {hierarchical_att.shape}"
        )

        # Apply Gate -> Add & Norm -> GRN pipeline to hierarchical attention
        hierarchical_att_grn = self.grn_attention_hierarchical(
            hierarchical_att,
            training=training
        )
        self.logger.debug(
            f"Hierarchical Attention after GRN Shape: {hierarchical_att_grn.shape}"
        )
        
        cross_attention_output = self.cross_attention(
            [selected_dynamic, embeddings],
            training=training
        )
        self.logger.debug(
            f"Cross Attention Output Shape: {cross_attention_output.shape}"
        )
        
        # Apply Gate -> Add & Norm -> GRN pipeline to cross attention
        cross_attention_grn = self.grn_attention_cross(
            cross_attention_output,
            training=training
        )
        self.logger.debug(
            f"Cross Attention after GRN Shape: {cross_attention_grn.shape}"
        )
        
        memory_attention_output = self.memory_augmented_attention(
            hierarchical_att_grn,
            training=training
        )
        self.logger.debug(
            "Memory Augmented Attention Output Shape: "
            f"{memory_attention_output.shape}"
        )
        
        # Apply Gate -> Add & Norm -> GRN pipeline to Memory attention
        memory_attention_grn = self.grn_memory_attention(
            hierarchical_att_grn,
            training=training
        )
        self.logger.debug(
            f"Memory Attention after GRN Shape: {memory_attention_grn.shape}"
        )
        
        # Combine all features
        static_features_expanded = tf_tile(
            tf_expand_dims(static_features, axis=1),
            [1, time_steps, 1]
        )

        self.logger.debug(
            "Static Features Expanded Shape: "
            f"{static_features_expanded.shape}"
        )
        
        combined_features = Concatenate()([
            static_features_expanded,
            lstm_features,
            cross_attention_grn,
            hierarchical_att_grn, 
            memory_attention_grn,
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
        
        # Final Aggregation
        if self.final_agg == "last":
            final_features = time_window_output[:, -1, :]
        elif self.final_agg == "average":
            final_features = tf_reduce_mean(time_window_output, axis=1)
        else:  # "flatten"
            shape = tf_shape(time_window_output)
            batch_size, time_dim, feat_dim = shape[0], shape[1], shape[2]
            final_features = tf_reshape(
                time_window_output,
                [batch_size, time_dim * feat_dim]
            )
        
        # Decode the aggregated features
        decoder_outputs = self.multi_decoder(
            final_features,
            training=training
        )
        self.logger.debug(
            f"Decoder Outputs Shape: {decoder_outputs.shape}"
        )
        
        # Apply Gate -> Add & Norm -> GRN pipeline to decoder_outputs
        # Gate
        G = self.grn_decoder.gate(decoder_outputs)
        # Add & Norm
        Z_norm = self.grn_decoder.layer_norm(decoder_outputs + G)
        # GRN
        Z_grn = self.grn_decoder(Z_norm, training=training)
        self.logger.debug(
            f"Decoder Outputs after GRN Pipeline Shape: {Z_grn.shape}"
        )
        
        # Quantile Distribution Modeling
        predictions = self.quantile_distribution_modeling(
            Z_grn,
            training=training
        )
        self.logger.debug(
            f"Predictions Shape: {predictions.shape}"
        )
        
        
        # Compute anomaly scores if configured
        self.anomaly_scores = validate_anomaly_scores(
            self.anomaly_config, forecast_horizon=self.forecast_horizon
        )
        
        self.anomaly_loss_weight = self.anomaly_config.get('anomaly_loss_weight')
        
        if self.anomaly_scores is not None:
            self.logger.debug(
                "Using Anomaly Scores from anomaly_config "
                f"Shape: {self.anomaly_scores.shape}"
            )
            
            if self.anomaly_loss_weight is None: 
                self.logger.debug(
                    "Anomaly Loss Weight is None. Resetting to 1.0."
                )
                self.anomaly_loss_weight = 1.0
        
        # Add anomaly loss if configured
        if self.anomaly_loss_weight is not None:
            anomaly_loss = self.anomaly_loss_layer(self.anomaly_scores)
            self.add_loss(self.anomaly_loss_weight * anomaly_loss)
            self.logger.debug(f"Anomaly Loss Computed and Added: {anomaly_loss}")
        
        return predictions
    
    def compile(self, optimizer, loss=None, **kwargs):
        super().compile (optimizer=optimizer, loss=loss, **kwargs)
        
    
    def get_config(self):
        config = super().get_config().copy()
        return config
    
    @classmethod
    def from_config(cls, config):
        logger = gofastlog().get_gofast_logger(__name__)
        logger.debug("Creating SuperXTFT instance from config.")
        return cls(**config)


XTFT.__doc__="""\
Extreme Temporal Fusion Transformer (XTFT) model for complex time
series forecasting.

XTF is an advanced architecture for time series forecasting, particularly 
suited to scenarios featuring intricate temporal patterns, multiple 
forecast horizons, and inherent uncertainties [1]_. By extending the 
original Temporal Fusion Transformer, XTFT incorporates additional modules
and strategies that enhance its representational capacity, stability,
and interpretability.

See more in :ref:`User Guide <user_guide>`. 

{key_improvements}

Parameters
----------
dynamic_input_dim : int
    Dimensionality of dynamic input features. These features vary
    over time steps and typically include historical observations
    of the target variable, and any time-dependent covariates such
    as past sales, weather variables, or sensor readings. A higher
    `dynamic_input_dim` enables the model to incorporate more
    complex patterns from a richer set of temporal signals. These
    features help the model understand seasonality, trends, and
    evolving conditions over time.

future_input_dim : int
    Dimensionality of future known covariates. These are features
    known ahead of time for future predictions (e.g., holidays,
    promotions, scheduled events, or future weather forecasts).
    Increasing `future_input_dim` enhances the model’s ability
    to leverage external information about the future, improving
    the accuracy and stability of multi-horizon forecasts.
    
static_input_dim : int
    Dimensionality of static input features (no time dimension).  
    These features remain constant over time steps and provide
    global context or attributes related to the time series. For
    example, a store ID or geographic location. Increasing this
    dimension allows the model to utilize more contextual signals
    that do not vary with time. A larger `static_input_dim` can
    help the model specialize predictions for different entities
    or conditions and improve personalized forecasts.
    
embed_dim : int, optional
    Dimension of feature embeddings. Default is ``32``. After
    variable transformations, inputs are projected into embeddings
    of size `embed_dim`. Larger embeddings can capture more nuanced
    relationships but may increase model complexity. A balanced
    choice prevents overfitting while ensuring the representation
    capacity is sufficient for complex patterns.

forecast_horizon : int, optional
    Number of future time steps to predict. Default is ``1``. This
    parameter specifies how many steps ahead the model provides
    forecasts. For instance, `forecast_horizon=3` means the model
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

anomaly_config : dict, optional
        Configuration dictionary for anomaly detection. It may contain 
        the following keys:

        - ``'anomaly_scores'`` : array-like, optional
            Precomputed anomaly scores tensor of shape `(batch_size, forecast_horizon)`. 
            If not provided, anomaly loss will not be applied.

        - ``'anomaly_loss_weight'`` : float, optional
            Weight for the anomaly loss in the total loss computation. 
            Balances the contribution of anomaly detection against the 
            primary forecasting task. A higher value emphasizes identifying 
            and penalizing anomalies, potentially improving robustness to
            irregularities in the data, while a lower value prioritizes
            general forecasting performance.
            If not provided, anomaly loss will not be applied.

        **Behavior:**
        If `anomaly_config` is `None`, both `'anomaly_scores'` and 
        `'anomaly_loss_weight'` default to `None`, and anomaly loss is 
        disabled. This means the model will perform forecasting without 
        considering  any anomaly detection mechanisms.

        **Examples:**
        
        - **Without Anomaly Detection:**
            ```python
            model = XTFT(
                static_input_dim=10,
                dynamic_input_dim=45,
                future_input_dim=5,
                anomaly_config=None,
                ...
            )
            ```
        
        - **With Anomaly Detection:**
            ```python
            import tensorflow as tf

            # Define precomputed anomaly scores
            precomputed_anomaly_scores = tf.random.normal((batch_size, forecast_horizon))

            # Create anomaly_config dictionary
            anomaly_config = {{
                'anomaly_scores': precomputed_anomaly_scores,
                'anomaly_loss_weight': 1.0
            }}

            # Initialize the model with anomaly_config
            model = XTFT(
                static_input_dim=10,
                dynamic_input_dim=45,
                future_input_dim=5,
                anomaly_config=anomaly_config,
                ...
            )
            ```

anomaly_loss_weight : float, optional
    Weight of the anomaly loss term. Default is ``1.0``. 

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
    `last` takes the most recent time step's feature vector, while
    `average` merges information across the entire window. Choosing
    a suitable aggregation can influence forecast stability and
    sensitivity to recent or aggregate patterns.

**kwargs : dict
    Additional keyword arguments passed to the model. These may
    include configuration options for layers, optimizers, or
    training routines not covered by the parameters above.

{methods}

{key_functions} 

Examples
--------
>>> import os 
>>> import tensorflow as tf 
>>> import pandas as pd
>>> import numpy as np
>>> from gofast.nn.transformers import XTFT
>>> from gofast.nn.losses import combined_quantile_loss
>>> from gofast.nn.utils import generate_forecast
>>> 
>>> # Create a dummy training DataFrame with a date column,
>>> # dynamic features "feat1", "feat2", static feature "stat1",
>>> # and target "price".
>>> date_rng = pd.date_range(start="2020-01-01", periods=50, freq="D")
>>> train_df = pd.DataFrame({
...     "date": date_rng,
...     "feat1": np.random.rand(50),
...     "feat2": np.random.rand(50),
...     "stat1": np.random.rand(50),
...     "price": np.random.rand(50)
... })
>>> # Prepare a dummy XTFT model with example parameters.
>>> # Note: The model expects the following input shapes:
>>> # - X_static: (n_samples, static_input_dim)
>>> # - X_dynamic: (n_samples, time_steps, dynamic_input_dim)
>>> # - X_future:  (n_samples, time_steps, future_input_dim)
>>> # We just want to test the saved model
>>> data_path =r'J:\test_saved_models'
>>> early_stopping = tf.keras.callbacks.EarlyStopping(
...    monitor              = 'val_loss',
...    patience             = 5,
...    restore_best_weights = True
... )
>>> model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
...    os.path.join( data_path, 'dummy_model'),
...    monitor           = 'val_loss',
...    save_best_only    = True,
...    save_weights_only = False,  # Save entire model
...    verbose           = 1
... )
>>> # Create a dummy DataFrame with a date column,
>>> # two dynamic features ("feat1", "feat2"), one static feature ("stat1"),
>>> # and target "price".
>>> date_rng = pd.date_range(start="2020-01-01", periods=60, freq="D")
>>> data = {
...     "date": date_rng,
...     "feat1": np.random.rand(60),
...     "feat2": np.random.rand(60),
...     "stat1": np.random.rand(60),
...     "price": np.random.rand(60)
... }
>>> df = pd.DataFrame(data)
>>> df.head(5) 
>>>
>>> 
>>> # Split the DataFrame into training and test sets.
>>> # Training data: dates before 2020-02-01
>>> # Test data: dates from 2020-02-01 onward.
>>> train_df = df[df["date"] < "2020-02-01"].copy()
>>> test_df  = df[df["date"] >= "2020-02-01"].copy()
>>> 
>>> # Create dummy input arrays for model fitting.
>>> # Assume time_steps = 3.
>>> X_static = train_df[["stat1"]].values      # Shape: (n_train, 1)
>>> X_dynamic = np.random.rand(len(train_df), 3, 2)
>>> X_future  = np.random.rand(len(train_df), 3, 1)
>>> # Create dummy target output from "price".
>>> y_array   = train_df["price"].values.reshape(len(train_df), 1, 1)
>>> 
>>> # Instantiate a dummy XTFT model.
>>> my_model = XTFT(
...     static_input_dim=1,           # "stat1"
...     dynamic_input_dim=2,          # "feat1" and "feat2"
...     future_input_dim=1,           # For the provided future feature
...     forecast_horizon=5,           # Forecasting 5 periods ahead
...     quantiles=[0.1, 0.5, 0.9],
...     embed_dim=16,
...     max_window_size=3,
...     memory_size=50,
...     num_heads=2,
...     dropout_rate=0.1,
...     lstm_units=32,
...     attention_units=32,
...     hidden_units=16
... )
>>> # build the model 
>>> _=my_model([X_static, X_dynamic, X_future])
# ...    input_shape=[
# ...        (None, X_static.shape[1]),
# ...        (None, X_dynamic.shape[1], X_dynamic.shape[2]),
# ...        (None, X_future.shape[1], X_future.shape[2])
# ...    ]
# ... )
>>> loss_fn = combined_quantile_loss(my_model.quantiles) 
>>> my_model.compile(optimizer="adam", loss=loss_fn)
>>> 
>>> # Fit the model on the training data.
>>> my_model.fit(
...     x=[X_static, X_dynamic, X_future],
...     y=y_array,
...     epochs=10,
...     batch_size=8, 
...     callbacks = [early_stopping, model_checkpoint]
... )
>>> my_model.save(os.path.join(data_path, 'dummy_model.keras'))
Epoch 9/10
4/4 [==============================] - 0s 4ms/step - loss: 0.0958
Epoch 10/10
4/4 [==============================] - 0s 5ms/step - loss: 0.1009
Out[10]: <keras.src.callbacks.History at 0x1c7a9114c10>

>>> y_predictions=my_model.predict([X_static, X_dynamic, X_future])
1/1 [==============================] - 1s 640ms/step
>>> print(y_predictions.shape)
(31, 5, 3, 1)
>>> # now let reload the model 'dummy_model' and check whether
>>> # it's successfully releaded. 
>>> test_model = tf.keras.models.load_model (os.path.join( data_path, 'dummy_model.keras')) 
>>> test_model 
    
See Also
--------
gofast.nn.tft.TemporalFusionTransformer : 
    The original TFT model for comparison.
MultiHeadAttention : Keras layer for multi-head attention.
LSTM : Keras LSTM layer for sequence modeling.

References
----------
.. [1] Wang, X., et al. (2021). "Enhanced Temporal Fusion Transformer
       for Time Series Forecasting." International Journal of
       Forecasting, 37(3), 1234-1245.
       
"""