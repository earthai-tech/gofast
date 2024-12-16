# -*- coding: utf-8 -*-
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
from ..api.property import  NNLearner 
from ..core.checks import validate_nested_param, ParamsValidator 
from ..compat.sklearn import validate_params, Interval, StrOptions 

from ..utils.deps_utils import ensure_pkg
from ..utils.validator import check_consistent_length 

from . import KERAS_DEPS, KERAS_BACKEND, dependency_message
from ._nn_docs import _shared_docs 
from ._tensor_validation import validate_anomaly_scores, validate_xtft_inputs 
from .losses import combined_quantile_loss
from .utils import set_default_params, set_anomaly_config 

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
    
    from . import Activation 

DEP_MSG = dependency_message('transformers.xtft') 


# -------------------- XTFT components ----------------------------------------

@register_keras_serializable('Gofast')
class LearnedNormalization(Layer, NNLearner):
    """
    A layer that learns mean and std for normalization of inputs.  
    Input: (B, D)  
    Output: (B, D), normalized
    """
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(self):
        super().__init__()

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

@register_keras_serializable('Gofast')
class MultiModalEmbedding(Layer, NNLearner):
    """
    This layer takes multiple input modalities (e.g., dynamic and future covariates), 
    embeds them into a common space, and concatenates them along the feature dimension.

    Input: list of [ (B, T, D_mod1), (B, T, D_mod2), ... ]  
    Output: (B, T, sum_of_embed_dims)
    """
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(self, embed_dim: int):
        super().__init__()
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
            if isinstance(modality, Tensor):
                modality_embed = self.dense_layers[idx](modality)
            else:
                raise ValueError("Unsupported modality type.")
            embeddings.append(modality_embed)
        return tf_concat(embeddings, axis=-1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'embed_dim': self.embed_dim})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
@register_keras_serializable('Gofast')
class HierarchicalAttention(Layer, NNLearner):
    """
    Hierarchical attention layer that first processes short-term and long-term 
    sequences separately and then combines their attention outputs.

    Input: short_term (B, T, D), long_term (B, T, D)  
    Output: (B, T, U) where U is attention_units
    """
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(self, units: int, num_heads: int):
        super().__init__()
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

@register_keras_serializable('Gofast')
class CrossAttention(Layer, NNLearner):
    """
    Cross attention layer that attends one source to another.

    Input: source1 (B, T, D), source2 (B, T, D)  
    Output: (B, T, U)
    """
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(self, units: int, num_heads: int):
        super().__init__()
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


@register_keras_serializable('Gofast')
class MemoryAugmentedAttention(Layer, NNLearner):
    """
    Memory-augmented attention layer that uses a learned memory matrix to enhance 
    temporal representation.

    Input: (B, T, D)  
    Output: (B, T, D)
    """
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(self, units: int, memory_size: int, num_heads: int):
        super().__init__()
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
        batch_size = tf_shape(inputs)[0]
        memory_expanded = tf_tile(tf_expand_dims(
            self.memory, axis=0), [batch_size, 1, 1])
        memory_attended = self.attention(
            query=inputs, value=memory_expanded, key=memory_expanded)
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

@register_keras_serializable('Gofast')
class AdaptiveQuantileLoss(Layer, NNLearner):
    """
    Computes adaptive quantile loss for given quantiles.

    Input: y_true (B, H, O), y_pred (B, H, Q, O) if quantiles are not None
    """
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(self, quantiles: Optional[List[float]]):
        super().__init__()
        if quantiles == 'auto':
            quantiles = [0.1, 0.5, 0.9]
        self.quantiles = quantiles

    def call(self, y_true, y_pred, training=False):
        if self.quantiles is None:
            return 0.0
        y_true_expanded = tf_expand_dims(y_true, axis=2)  # (B, H, 1, O)
        error = y_true_expanded - y_pred  # (B, H, Q, O)
        quantiles = tf_constant(self.quantiles, dtype=tf_float32)
        quantiles = tf_reshape(quantiles, [1, 1, len(self.quantiles), 1])
        quantile_loss = tf_maximum(quantiles * error, (quantiles - 1) * error)
        return tf_reduce_mean(quantile_loss)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'quantiles': self.quantiles})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register_keras_serializable('Gofast')
class AnomalyLoss(Layer, NNLearner):
    """
    Computes anomaly loss as mean squared anomaly score.

    Input: anomaly_scores (B, H, D)
    """
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def call(self, anomaly_scores: Tensor):
        return self.weight * tf_reduce_mean(tf_square(anomaly_scores))

    def get_config(self):
        config = super().get_config().copy()
        config.update({'weight': self.weight})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@register_keras_serializable('Gofast')
class MultiObjectiveLoss(Layer, NNLearner):
    """
    Combines quantile loss and anomaly loss into a single objective.

    Input: 
        y_true: (B, H, O)
        y_pred: (B, H, Q, O) if quantiles is not None else (B, H, 1, O)
        anomaly_scores: (B, H, D)
    """
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(self, quantile_loss_fn: Layer, anomaly_loss_fn: Layer):
        super().__init__()
        self.quantile_loss_fn = quantile_loss_fn
        self.anomaly_loss_fn = anomaly_loss_fn

    def call(self, y_true, y_pred, anomaly_scores=None, training=False):
        # XXX :MARK: anomaly_scores henceforth can take None 
        quantile_loss = self.quantile_loss_fn(y_true, y_pred)
        if anomaly_scores is not None:
            anomaly_loss = self.anomaly_loss_fn(anomaly_scores)
            return quantile_loss + anomaly_loss
        
        return quantile_loss # retuns quantile loss only

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


@register_keras_serializable('Gofast')
class ExplainableAttention(Layer, NNLearner):
    """
    Returns attention scores from multi-head attention, useful for 
    interpretation.

    Input: (B, T, D)
    Output: attention_scores (B, num_heads, T, T)
    """
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(self, num_heads: int, key_dim: int):
        super().__init__()
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


@register_keras_serializable('Gofast')
class MultiDecoder(Layer, NNLearner):
    """
    Multi-horizon decoder:
    Takes a single feature vector per example (B, F) and produces a prediction 
    for each horizon as (B, H, O).

    Input: (B, F)
    Output: (B, H, O)
    """
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(self, output_dim: int, num_horizons: int):
        super().__init__()
        self.output_dim = output_dim
        self.num_horizons = num_horizons
        self.decoders = [Dense(output_dim) for _ in range(num_horizons)]

    def call(self, x, training=False):
        outputs = [decoder(x) for decoder in self.decoders]
        return tf_stack(outputs, axis=1)

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

@register_keras_serializable('Gofast')
class MultiResolutionAttentionFusion(Layer, NNLearner):
    """
    Applies multi-head attention fusion over features.
    
    Input: (B, T, D)
    Output: (B, T, D)
    """
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(self, units: int, num_heads: int):
        super().__init__()
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

@register_keras_serializable('Gofast')
class DynamicTimeWindow(Layer, NNLearner):
    """
    Slices the last max_window_size steps from the input sequence.

    Input: (B, T, D)
    Output: (B, W, D) where W = max_window_size
    """
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(self, max_window_size: int):
        super().__init__()
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

@register_keras_serializable('Gofast')
class QuantileDistributionModeling(Layer, NNLearner):
    """
    Projects deterministic outputs (B, H, O) into quantile predictions (B, H, Q, O),
    or returns (B, H, O) if quantiles are None (no extra quantile dimension).

    Input: (B, H, O)
    Output:
        - If quantiles is None: (B, H, O) #rather than  otherwise (B, H, 1, O)
        - If quantiles is a list: (B, H, Q, O)
    """
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(self, quantiles: Optional[Union[str, List[float]]], output_dim: int):
        super().__init__()
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
        return tf_stack(outputs, axis=2)  # (B, H, Q, O)

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

@register_keras_serializable('Gofast')
class MultiScaleLSTM(Layer, NNLearner):
    """
    Multi-scale LSTM layer that can output either the last hidden state
    from each LSTM or full sequences. Behavior controlled by `return_sequences`.
    
    Multi-scale LSTM layer that applies multiple LSTMs at different scales 
    and concatenates their outputs.

    Input: (B, T, D)
    Output: (B, T, sum_of_lstm_units) if return_sequences=True
    """
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
    def __init__(
        self,
        lstm_units: int,
        scales: Union[str, List[int], None] = None,
        return_sequences: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        if scales is None or scales == 'auto':
            scales = [1]
        scales = validate_nested_param(scales, List[int], 'scales')
        
        self.lstm_units = lstm_units
        self.scales = scales
        self.return_sequences = return_sequences

        self.lstm_layers = [
            LSTM(
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
            return tf_concat(outputs, axis=-1)
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
    
# -----------------XTFT implementation ----------------------------------------

@register_keras_serializable('Gofast')
@doc (
    key_improvements= dedent(_shared_docs['xtft_key_improvements']), 
    key_functions= dedent(_shared_docs['xtft_key_functions']), 
    methods= dedent( _shared_docs['xtft_methods']
    )
 )
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
    @ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
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
        **kwargs
    ):
        super().__init__(**kwargs)

        # Initialize Logger
        self.logger = gofastlog().get_gofast_logger(__name__)
        
        self.activation = Activation(activation) 
        self.activation_name = self.activation.activation_name
        
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
            f"anomaly_config={None if anomaly_config is None else anomaly_config.keys()}, "
            f"attention_units={attention_units}, "
            f" hidden_units={hidden_units}, "
            f"lstm_units={lstm_units}, "
            f"scales={scales}, "
            f"activation={self.activation_name}, "
            f"use_residuals={use_residuals}, "
            f"use_batch_norm={use_batch_norm}, "
            f"final_agg={final_agg}"
        )
        # Handle default quantiles, scales and multi_scale_agg 
        quantiles, scales, return_sequences = set_default_params(
            quantiles, scales, multi_scale_agg ) 
        
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
        # self.anomaly_loss_weight = anomaly_loss_weight 
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
        self.static_dense = Dense(hidden_units, activation=self.activation_name)
        self.static_dropout = Dropout(dropout_rate)
        if self.use_batch_norm:
            self.static_batch_norm = LayerNormalization()
        self.residual_dense = Dense(2 * embed_dim) if use_residuals else None
        self.final_dense = Dense(output_dim)
        

    def call(self, inputs, training=False):
        static_input, dynamic_input, future_covariate_input = validate_xtft_inputs (
            inputs =inputs,
            static_input_dim=self.static_input_dim, 
            dynamic_input_dim= self.dynamic_input_dim, 
            future_covariate_dim= self.future_covariate_dim, 
        )
    
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
                    tf_reduce_mean(o, axis=1) 
                    for o in lstm_output
                ]  # Each is (B, units)
                lstm_features = tf_concat(
                    averaged_outputs,
                    axis=-1
                )  # (B, units * len(scales))
    
            elif self.multi_scale_agg == "flatten":
                # Flatten time and feature dimensions for all scales
                # Assume equal time lengths for all scales
                concatenated = tf_concat(
                    lstm_output, 
                    axis=-1
                )  # (B, T', units*len(scales))
                shape = tf_shape(concatenated)
                (batch_size,
                 time_dim,
                 feat_dim) = shape[0], shape[1], shape[2]
                lstm_features = tf_reshape(
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
                lstm_features = tf_concat(
                    last_outputs,
                    axis=-1
                )  # (B, units * len(scales))
        
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
            self.anomaly_config, forecast_horizons= self.forecast_horizons)
        
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
            self.logger.debug(f"Anomaly Loss Computed and Added: {anomaly_loss}")
            
        return predictions
    
    def compile(self, optimizer, loss=None, **kwargs):
        if self.quantiles is None:
            # Deterministic scenario
            super().compile(
                optimizer=optimizer, loss=loss or 'mse', **kwargs)
        else:
            # Probabilistic scenario with combined quantile loss
            quantile_loss_fn = combined_quantile_loss(self.quantiles)
    
            if not hasattr (self, 'anomaly_scores'): 
                self.anomaly_scores = self.anomaly_config.get('anomaly_scores')
                
            if self.anomaly_scores is not None:
                # Define a total loss that includes both quantile loss and anomaly loss
                def total_loss(y_true, y_pred):
                    # Compute quantile loss
                    q_loss = quantile_loss_fn(y_true, y_pred)
    
                    # Compute anomaly loss
                    # anomaly_scores = self._get_anomaly_scores()
                    a_loss = self.anomaly_loss_layer(self.anomaly_scores)
    
                    # Combine losses
                    return q_loss + a_loss
    
                super().compile(
                    optimizer=optimizer, loss=total_loss, **kwargs)
            else:
                # Only quantile loss
                super().compile(
                    optimizer=optimizer, loss=quantile_loss_fn, **kwargs)

    @ParamsValidator(
        { 
          'y_true': ['array-like:tf:transf'], 
          'y_pred': ['array-like:tf:transf'], 
          'anomaly_scores':['array-like:tf:transf']
        }, 
    )
    def objective_loss(
        self, 
        y_true: Tensor, 
        y_pred: Tensor, 
        anomaly_scores: Tensor=None
    ) -> Tensor:
        if not hasattr (self, 'anomaly_scores'): 
            self.anomaly_scores = self.anomaly_config.get('anomaly_scores')
            
        if self.anomaly_scores is not None: 
            check_consistent_length(y_true, y_pred, self.anomaly_scores)
            # Expect y_true, 'y_pred, and 'anomaly_scores'
            # Compute the multi-objective loss
            loss = self.multi_objective_loss(y_true, y_pred, self.anomaly_scores)
            return loss
        else: 
            # When anomaly_loss_weight is None, y_true is a tensor
            check_consistent_length(y_true, y_pred)
            return self.multi_objective_loss(y_true, y_pred)


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
            'anomaly_config': {
                'anomaly_scores': self.anomaly_scores.numpy() if self.anomaly_scores is not None else None,
                'anomaly_loss_weight': self.anomaly_loss_weight
            },
            'attention_units': self.attention_units,
            'hidden_units': self.hidden_units,
            'lstm_units': self.lstm_units,
            'scales': self.scales,
            'activation': self.activation_name,
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
            Precomputed anomaly scores tensor of shape `(batch_size, forecast_horizons)`. 
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
                future_covariate_dim=5,
                anomaly_config=None,
                ...
            )
            ```
        
        - **With Anomaly Detection:**
            ```python
            import tensorflow as tf

            # Define precomputed anomaly scores
            precomputed_anomaly_scores = tf.random.normal((batch_size, forecast_horizons))

            # Create anomaly_config dictionary
            anomaly_config = {{
                'anomaly_scores': precomputed_anomaly_scores,
                'anomaly_loss_weight': 1.0
            }}

            # Initialize the model with anomaly_config
            model = XTFT(
                static_input_dim=10,
                dynamic_input_dim=45,
                future_covariate_dim=5,
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
>>> from gofast.nn.transformers import XTFT
>>> import tensorflow as tf
>>> model = XTFT(
...     static_input_dim=10,
...     dynamic_input_dim=45,
...     future_covariate_dim=5,
...     forecast_horizons=3,
...     quantiles=None# [0.1, 0.5, 0.9],

...     scales='auto',
...     final_agg='last'
... )
>>> batch_size = 32
>>> time_steps = 20
>>> output_dim=1
>>> static_input = tf.random.normal([batch_size, 10])
>>> dynamic_input = tf.random.normal([batch_size, time_steps, 45])
>>> future_covariate_input = tf.random.normal([batch_size, time_steps, 5])
>>> output = model([static_input, dynamic_input, future_covariate_input])
>>> output.shape
TensorShape([32, 3, 3, 1])

>>>  # True targets
>>> y_true_forecast = tf.random.normal([batch_size, 3, output_dim])
>>> model.compile(optimizer ='adam', loss=['mse'])
>>> output = model.fit(
    x=[static_input, dynamic_input, future_covariate_input], 
    y=y_true_forecast
    )
>>> output
1/1 [==============================] - 2s 2s/step - loss: 1.0534
Out[8]: <keras.callbacks.History at 0x20474300640>


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
 