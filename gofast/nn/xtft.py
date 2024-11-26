# -*- coding: utf-8 -*-
"""
Implement Extreme Temporal Fusion Transformer (XTFT)
"""

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Embedding, MultiHeadAttention, Add
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Dropout

#
# 1. Enhanced Variable Embeddings
class LearnedNormalization(Layer):
    def __init__(self):
        super(LearnedNormalization, self).__init__()

    def build(self, input_shape):
        self.mean = self.add_weight("mean", shape=input_shape[1:], 
                                    initializer="zeros", trainable=True)
        self.stddev = self.add_weight(
            "stddev", shape=input_shape[1:], 
             initializer="ones", trainable=True
              )

    def call(self, inputs, training=False):
        return (inputs - self.mean) / self.stddev

class MultiModalEmbedding(Layer):
    def __init__(self, embed_dim):
        super(MultiModalEmbedding, self).__init__()
        self.embed_dim = embed_dim

    def call(self, inputs, training=False):
        embeddings = []
        for modality in inputs:
            if isinstance(modality, tf.Tensor):
                modality_embed = Dense(self.embed_dim)(modality)
            elif isinstance(modality, str):
                modality_embed = Embedding(
                    input_dim=1000, 
                    output_dim=self.embed_dim)(modality)
            embeddings.append(modality_embed)
        return tf.concat(embeddings, axis=-1)
    
# 4. MultiScaleLSTM Mechanisms    
class MultiScaleLSTM(tf.keras.layers.Layer):
    def __init__(
        self, 
        lstm_units, 
        scales=[1, 7, 30], 
        return_sequences=False, 
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
            # Subsample the input based on the scale (e.g., daily, weekly, monthly)
            scaled_input = inputs[:, ::scale, :]
            lstm_output = lstm(scaled_input)
            outputs.append(lstm_output)

        # Concatenate outputs along the feature axis
        return tf.concat(outputs, axis=-1)

# 4. Enhanced Attention Mechanisms
class HierarchicalAttention(Layer):
    def __init__(self, units):
        super(HierarchicalAttention, self).__init__()
        self.short_term_attention = MultiHeadAttention(num_heads=4, key_dim=units)
        self.long_term_attention = MultiHeadAttention(num_heads=4, key_dim=units)

    def call(self, inputs, training=False):
        short_term, long_term = inputs
        short_term_attention = self.short_term_attention(short_term, short_term)
        long_term_attention = self.long_term_attention(long_term, long_term)
        return short_term_attention + long_term_attention


class CrossAttention(Layer):
    def __init__(self, units):
        super(CrossAttention, self).__init__()
        self.cross_attention = MultiHeadAttention(num_heads=4, key_dim=units)

    def call(self, inputs, training=False):
        source1, source2 = inputs
        return self.cross_attention(source1, source2)

class MemoryAugmentedAttention(Layer):
    def __init__(self, units, memory_size):
        super(MemoryAugmentedAttention, self).__init__()
        self.attention = MultiHeadAttention(num_heads=4, key_dim=units)
        self.memory = tf.Variable(tf.zeros([memory_size, units]))

    def call(self, inputs):
        memory_attended = self.attention(self.memory, inputs)
        return memory_attended + inputs

# 5. Dynamic Quantile Loss
class AdaptiveQuantileLoss(Layer):
    def __init__(self, quantiles):
        super(AdaptiveQuantileLoss, self).__init__()
        self.quantiles = quantiles

    def call(self, y_true, y_pred, training=False):
        quantile_losses = []
        for q in self.quantiles:
            error = y_true - y_pred
            quantile_loss = tf.maximum(q * error, (q - 1) * error)
            quantile_losses.append(quantile_loss)
        return tf.reduce_mean(tf.stack(quantile_losses, axis=-1))

class MultiObjectiveLoss(Layer):
    def __init__(self, quantile_loss_fn, anomaly_loss_fn):
        super(MultiObjectiveLoss, self).__init__()
        self.quantile_loss_fn = quantile_loss_fn
        self.anomaly_loss_fn = anomaly_loss_fn

    def call(self, y_true, y_pred, anomaly_scores, training=False):
        quantile_loss = self.quantile_loss_fn(y_true, y_pred)
        anomaly_loss = self.anomaly_loss_fn(anomaly_scores)
        return quantile_loss + anomaly_loss

# 7. Scalability and Efficiency
class SparseAttention(Layer):
    def __init__(self, num_heads, key_dim):
        super(SparseAttention, self).__init__()
        self.attention = MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim)

    def call(self, inputs, training=False):
        return self.attention(
            inputs, inputs)

class ModelCompression(Layer):
    def __init__(self, prune_percent):
        super(ModelCompression, self).__init__()
        self.prune_percent = prune_percent

    def call(self, model):
        # Example: apply pruning to the model weights
        for layer in model.layers:
            if isinstance(layer, Dense):
                num_params = layer.get_weights()[0].shape[0]
                pruning_mask = tf.random.uniform([num_params]) < self.prune_percent
                pruned_weights = layer.get_weights()[0] * tf.expand_dims(pruning_mask, axis=-1)
                layer.set_weights([pruned_weights, layer.get_weights()[1]])
        return model

# 10. Interpretability Improvements
class ExplainableAttention(Layer):
    def __init__(self, num_heads, key_dim):
        super(ExplainableAttention, self).__init__()
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

    def call(self, inputs, training=False):
        attention_map = self.attention(inputs, inputs, return_attention_scores=True)
        return attention_map

# 11. Multi-Horizon Output Strategies
class MultiDecoder(Layer):
    def __init__(self, output_dim, num_horizons):
        super(MultiDecoder, self).__init__()
        self.decoders = [Dense(output_dim) for _ in range(num_horizons)]

    def call(self, x):
        return [decoder(x) for decoder in self.decoders]

class TransferLearningAdapter(Layer):
    def __init__(self, adapter_dim):
        super(TransferLearningAdapter, self).__init__()
        self.adapter = Dense(adapter_dim)

    def call(self, x, training=False):
        return self.adapter(x)

# 15. Optimization for Complex Time Series
class MultiResolutionAttentionFusion(Layer):
    def __init__(self, units):
        super(MultiResolutionAttentionFusion, self).__init__()
        self.attention = MultiHeadAttention(num_heads=4, key_dim=units)

    def call(self, inputs, training=False):
        return self.attention(inputs, inputs)

class DynamicTimeWindow(Layer):
    def __init__(self, max_window_size):
        super(DynamicTimeWindow, self).__init__()
        self.max_window_size = max_window_size

    def call(self, inputs, training=False):
        # Dynamically adjust window size based on input characteristics
        window_size = tf.random.uniform([], 1, self.max_window_size, dtype=tf.int32)
        return inputs[:, :window_size, :]

# 16. Advanced Output Mechanisms
class QuantileDistributionModeling(Layer):
    def __init__(self, quantiles):
        super(QuantileDistributionModeling, self).__init__()
        self.quantiles = quantiles

    def call(self, inputs, training=False):
        quantile_outputs = []
        for q in self.quantiles:
            quantile_output = tf.quantile(inputs, q, axis=0)
            quantile_outputs.append(quantile_output)
        return tf.stack(quantile_outputs, axis=-1)

# implement this completly 
class CustomizableOutputLayer(Layer):
    def __init__(self, output_config):
        super(CustomizableOutputLayer, self).__init__()
        self.output_config = output_config

    def call(self, x, training=False):
        if self.output_config == "autoregressive":
            return x  # Example: autoregressive output
        elif self.output_config == "rolling":
            return x  # Example: rolling forecast
        else:
            return x  # Example: direct multi-step forecast


class XTFT(Model):
    def __init__(
        self,
        static_input_dim,
        dynamic_input_dim,
        future_covariate_dim,
        num_static_vars,
        num_dynamic_vars,
        embed_dim=32,
        forecast_horizons=3,# 1 by default
        quantiles="auto",  # If None passed, default to [0.1, 0.5, 0.9] if 'auto', # fix it 
        adapter_dim=16,
        max_window_size=10,
        memory_size=100,
        num_heads=4,
        dropout_rate=0.1,
        output_dim=1,
        anomaly_loss_weight=1.0,
        decoder_units=16,
        attention_units=32, 
        hidden_units=64,
        lstm_units=64,
        scales="auto",  # If None , skip LSTM or if "auto", use defaults to [1, 7, 30] for daily, weekly, monthly
        activation="relu",
        use_residuals=True,
        use_batch_norm=False, # apply this also, to accelerate the training 
        **kwargs, 
    ):
        super(XTFT, self).__init__(**kwargs)

        # Save parameters and list all the parameters 
        self.static_input_dim = static_input_dim
        self.dynamic_input_dim = dynamic_input_dim
        self.future_covariate_dim = future_covariate_dim
        self.num_static_vars = num_static_vars
        self.num_dynamic_vars = num_dynamic_vars
        self.embed_dim=embed_dim
        self.forecast_horizons=forecast_horizons
        self.quantiles=quantiles
        self.adapter_dim=adapter_dim
        self.max_window_size=max_window_size
        self.memory_size=memory_size
        self.num_heads=num_heads
        self.dropout_rate=dropout_rate
        self.output_dim=output_dim
        self.anomaly_loss_weight=anomaly_loss_weight
        self.decoder_units=decoder_units
        self.attention_units=attention_units
        self.hidden_units=hidden_units
        self.lstm_units=lstm_units
        self.scales=scales  
        self.activation=activation
        self.use_residuals=use_residuals
        self.use_batch_norm=use_batch_norm

        # Enhanced Variable Embeddings
        self.learned_normalization = LearnedNormalization()
        self.multi_modal_embedding = MultiModalEmbedding(embed_dim)

        # Improved Temporal Modeling
        if scales =="auto": 
            scales =[1, 7, 30] 
        # 
        self.multi_scale_lstm = MultiScaleLSTM(
            lstm_units, scales=scales)

        # Enhanced Attention Mechanisms
        self.hierarchical_attention = HierarchicalAttention(
            attention_units)
        self.cross_attention = CrossAttention(attention_units)
        self.memory_augmented_attention = MemoryAugmentedAttention(
            attention_units, memory_size)

        # Multi-Horizon Output Strategies
        self.multi_decoder = MultiDecoder(
            output_dim=output_dim, num_horizons=forecast_horizons
            )

        # Optimization for Complex Time Series
        self.multi_resolution_attention_fusion = MultiResolutionAttentionFusion(
            attention_units)
        self.dynamic_time_window = DynamicTimeWindow(max_window_size)

        # Output Layer: Quantile Distribution Modeling
        if quantiles =="auto": 
            quantiles=[0.1, 0.5, 0.9]
        self.quantile_distribution_modeling = QuantileDistributionModeling(
            quantiles)

        # Auxiliary loss for Multi-Objective Loss Function
        self.multi_objective_loss = MultiObjectiveLoss(
            AdaptiveQuantileLoss(quantiles), 
            anomaly_loss_fn=self.anomaly_loss
            )

        # Additional Layers for Static Information
        self.static_dense = Dense(hidden_units, activation=activation)
        self.static_dropout = Dropout(dropout_rate)

        # Residual connection for embeddings (if enabled)
        self.use_residuals = use_residuals
        self.residual_dense = Dense(embed_dim) if use_residuals else None

        # Final Prediction Layer
        self.final_dense = Dense(output_dim)

        # Anomaly Loss Weight
        self.anomaly_loss_weight = anomaly_loss_weight

    def call(self, inputs, training=False):
        static_input, dynamic_input, future_covariate_input = inputs

        # Apply normalization and embedding
        normalized_static = self.learned_normalization(static_input)
        static_features = self.static_dense(normalized_static)
        static_features = self.static_dropout(static_features)

        # Combine dynamic and future covariates
        embeddings = self.multi_modal_embedding(
            [dynamic_input, future_covariate_input])
        if self.use_residuals:
            embeddings = embeddings + self.residual_dense(embeddings)

        
        # Multi-Scale LSTM for Temporal Modeling
        lstm_features = self.multi_scale_lstm(dynamic_input)

        # Attention mechanisms
        hierarchical_att = self.hierarchical_attention(
            [dynamic_input, future_covariate_input])
        cross_attention_output = self.cross_attention([dynamic_input, embeddings])
        memory_attention_output = self.memory_augmented_attention(hierarchical_att)

        # Combine all features: static, LSTM, cross-attention, and memory-augmented attention
        combined_features = Concatenate()([static_features, lstm_features, 
                                           memory_attention_output, 
                                           cross_attention_output]
                                          )

        # Multi-resolution fusion and dynamic time window
        attention_fusion_output = self.multi_resolution_attention_fusion(
            combined_features)
        time_window_output = self.dynamic_time_window(attention_fusion_output)

        # Multi-Horizon Output Strategy
        decoder_outputs = self.multi_decoder(time_window_output)

        # Quantile distribution modeling for richer uncertainty
        quantile_outputs = self.quantile_distribution_modeling(
            decoder_outputs)

        # Final Prediction Output
        predictions = self.final_dense(quantile_outputs)

        return predictions

    def compute_loss(self, y_true, y_pred, anomaly_scores):
        # Compute the combined loss (quantile loss + anomaly loss)
        return self.multi_objective_loss(y_true, y_pred, anomaly_scores)

    def anomaly_loss(self, anomaly_scores):
        # Define anomaly loss function (e.g., based on reconstruction error or outlier detection)
        return self.anomaly_loss_weight * tf.reduce_mean(tf.square(anomaly_scores))
    

if __name__=="__main__": 
    # Example Usage
    static_input_dim = 10  # Static input dimensions
    dynamic_input_dim = 50  # Time-varying input dimensions
    future_covariate_dim = 5  # Known future covariate dimensions
    
    # Instantiate the model
    xtft_model = XTFT(
        static_input_dim=static_input_dim,
        dynamic_input_dim=dynamic_input_dim,
        future_covariate_dim=future_covariate_dim,
        num_static_vars=5,
        num_dynamic_vars=45,
        embed_dim=32,
        num_horizons=3,
        quantiles=[0.1, 0.5, 0.9],
        adapter_dim=16,
        max_window_size=10,
        memory_size=100,
        num_heads=4,
        dropout_rate=0.1,
        output_dim=1,
        anomaly_loss_weight=1.0,
        intermediate_units=64,
        hidden_units=32,
        activation="relu",
        use_residuals=True,
    )
    
    # Example inputs
    static_input = tf.random.normal([32, static_input_dim])  # Batch size of 32
    dynamic_input = tf.random.normal([32, dynamic_input_dim])
    future_covariate_input = tf.random.normal([32, future_covariate_dim])
    y_true = tf.random.normal([32, 3, 1])  # True values for 3 horizons
    anomaly_scores = tf.random.normal([32, dynamic_input_dim])
    
    # Forward pass with loss calculation
    output = xtft_model([static_input, dynamic_input, 
                              future_covariate_input])
    print("Model Output Shape:", output.shape)
 

# runcell(0, 'J:/zhongshan_project/zongshan_codes/data_process_new/ls_features_processing/xtft.py')
# Traceback (most recent call last):

#   File "C:\Users\Daniel\Anaconda3\envs\watex\lib\site-packages\spyder_kernels\py3compat.py", line 356, in compat_exec
#     exec(code, globals, locals)

#   File "j:\zhongshan_project\zongshan_codes\data_process_new\ls_features_processing\xtft.py", line 533, in <module>
#     output, loss = xtft_model([static_input, dynamic_input,

#   File "C:\Users\Daniel\Anaconda3\envs\watex\lib\site-packages\keras\src\utils\traceback_utils.py", line 70, in error_handler
#     raise e.with_traceback(filtered_tb) from None

#   File "j:\zhongshan_project\zongshan_codes\data_process_new\ls_features_processing\xtft.py", line 463, in call
#     embeddings = embeddings + residual

# InvalidArgumentError: Exception encountered when calling layer 'xtft_4' (type XTFT).

# {{function_node __wrapped__AddV2_device_/job:localhost/replica:0/task:0/device:CPU:0}} Incompatible shapes: [32,64] vs. [32,32] [Op:AddV2] name: 

# Call arguments received by layer 'xtft_4' (type XTFT):
#   • inputs=['tf.Tensor(shape=(32, 10), dtype=float32)', 'tf.Tensor(shape=(32, 50), dtype=float32)', 'tf.Tensor(shape=(32, 5), dtype=float32)']
#   • training=False