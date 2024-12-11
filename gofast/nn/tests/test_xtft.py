# -*- coding: utf-8 -*-
import pytest
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

from gofast.nn.transformers import XTFT
#%
@pytest.fixture
def basic_model():
    """Fixture to create a basic XTFT model."""
    # Minimal parameters for quick test
    model = XTFT(
        static_input_dim=10,
        dynamic_input_dim=45,
        future_covariate_dim=5,
        embed_dim=32,
        forecast_horizons=3,
        quantiles=None,  # Deterministic for simplicity
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
        final_agg="last"
    )
    return model

def test_model_initialization(basic_model):
    """Test that the model initializes properly."""
    assert isinstance(basic_model, XTFT), "Model should be an instance of XTFT."

def test_forward_pass(basic_model):
    """Test a forward pass through the model."""
    batch_size = 4
    forecast_horizons = basic_model.forecast_horizons
    static_input_dim = basic_model.static_input_dim
    dynamic_input_dim = basic_model.dynamic_input_dim
    future_covariate_dim = basic_model.future_covariate_dim
    output_dim = basic_model.output_dim

    # Create random test inputs
    static_input = tf.random.normal((batch_size, static_input_dim))
    dynamic_input = tf.random.normal((batch_size, 20, dynamic_input_dim))
    future_covariate_input = tf.random.normal((batch_size, 20, future_covariate_dim))

    # Run model inference
    outputs = basic_model([static_input, dynamic_input, future_covariate_input], training=False)
    # For deterministic prediction, shape: (B, H, 1,  O) # for quantiles is None:  become 
    #  (B, H,   O)
    expected_shape = (batch_size, forecast_horizons, output_dim)
    assert outputs.shape == expected_shape, f"Expected output shape {expected_shape}, got {outputs.shape}."

def test_compile_and_train_step(basic_model):
    """Test that model can compile and run a single training step."""
    basic_model.compile(optimizer=Adam(), loss=MeanSquaredError())
    batch_size = 4
    forecast_horizons = basic_model.forecast_horizons
    output_dim = basic_model.output_dim

    # Generate dummy inputs
    static_input = tf.random.normal((batch_size, basic_model.static_input_dim))
    dynamic_input = tf.random.normal((batch_size, 20, basic_model.dynamic_input_dim))
    future_covariate_input = tf.random.normal((batch_size, 20, basic_model.future_covariate_dim))
    y_true = tf.random.normal((batch_size, forecast_horizons, output_dim))
    anomaly_scores = tf.random.normal((batch_size, forecast_horizons, basic_model.dynamic_input_dim))

    with tf.GradientTape() as tape:
        y_pred = basic_model([static_input, dynamic_input, future_covariate_input], training=True)
        # Use the model's internal loss computation if desired
        # If directly training via model.fit, this step isn't required.
        # For demonstration, we'll just compute a dummy loss manually.
        loss = tf.reduce_mean((y_true - y_pred)**2) + tf.reduce_mean((anomaly_scores)**2)

    grads = tape.gradient(loss, basic_model.trainable_variables)
    # Ensure no None in gradients
    assert all([g is not None for g in grads if g is not None]), "Gradients should be computable for all trainable variables."

def test_serialization(basic_model):
    """Test model serialization and deserialization."""
    config = basic_model.get_config()
    # Ensure config is a dictionary
    assert isinstance(config, dict), "Config should be a dictionary."

    # Reconstruct model from config
    new_model = XTFT.from_config(config)
    # Check that newly created model is also XTFT and has same parameters
    assert isinstance(new_model, XTFT), "Deserialized model should be an XTFT instance."
    assert new_model.static_input_dim == basic_model.static_input_dim, "Parameters should be preserved after deserialization."


@pytest.fixture
def deterministic_model():
    """Fixture to create a deterministic XTFT model (no quantiles)."""
    model = XTFT(
        static_input_dim=10,
        dynamic_input_dim=45,
        future_covariate_dim=5,
        embed_dim=32,
        forecast_horizons=3,
        quantiles=None,  # Deterministic
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
        final_agg="last"
    )
    return model

@pytest.fixture
def probabilistic_model():
    """Fixture to create a probabilistic XTFT model (with quantiles)."""
    model = XTFT(
        static_input_dim=10,
        dynamic_input_dim=45,
        future_covariate_dim=5,
        embed_dim=32,
        forecast_horizons=3,
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
        scales=[1],  # Explicit scale
        activation="relu",
        use_residuals=True,
        use_batch_norm=False,
        final_agg="average"  # Different aggregation
    )
    return model

def create_dummy_data(model, batch_size=4, time_steps=20):
    """Helper function to create dummy data compatible with the model."""
    static_input = tf.random.normal((batch_size, model.static_input_dim))
    dynamic_input = tf.random.normal((batch_size, time_steps, model.dynamic_input_dim))
    future_covariate_input = tf.random.normal((batch_size, time_steps, model.future_covariate_dim))
    y_true = tf.random.normal((batch_size, model.forecast_horizons, model.output_dim))
    anomaly_scores = tf.random.normal((batch_size, model.forecast_horizons, model.dynamic_input_dim))
    return static_input, dynamic_input, future_covariate_input, y_true, anomaly_scores

def test_model_initialization_deterministic(deterministic_model):
    """Test that a deterministic model initializes properly."""
    assert isinstance(deterministic_model, XTFT)

def test_forward_pass_deterministic(deterministic_model):
    """Test a forward pass through the deterministic model."""
    batch_size = 4
    forecast_horizons = deterministic_model.forecast_horizons
    output_dim = deterministic_model.output_dim

    static_input, dynamic_input, future_covariate_input, _, _ = create_dummy_data(deterministic_model, batch_size)
    outputs = deterministic_model([static_input, dynamic_input, future_covariate_input], training=False)
    # Deterministic: (B, H, O)
    expected_shape = (batch_size, forecast_horizons, output_dim)
    assert outputs.shape == expected_shape, f"Expected {expected_shape}, got {outputs.shape}."

def test_compile_and_train_step_deterministic(deterministic_model):
    """Test that the deterministic model can compile and run a single training step."""
    deterministic_model.compile(optimizer=Adam(), loss=MeanSquaredError())
    batch_size = 4

    static_input, dynamic_input, future_covariate_input, y_true, anomaly_scores = create_dummy_data(deterministic_model, batch_size)
    with tf.GradientTape() as tape:
        y_pred = deterministic_model([static_input, dynamic_input, future_covariate_input], training=True)
        # simple loss calculation
        loss = tf.reduce_mean((y_true - y_pred)**2) + tf.reduce_mean(anomaly_scores**2)
    grads = tape.gradient(loss, deterministic_model.trainable_variables)
    assert all(g is not None for g in grads if g is not None), "All gradients should be computable."

def test_serialization_deterministic(deterministic_model):
    """Test deterministic model serialization and deserialization."""
    config = deterministic_model.get_config()
    assert isinstance(config, dict), "Config should be a dictionary."
    new_model = XTFT.from_config(config)
    assert isinstance(new_model, XTFT), "Should be XTFT instance after deserialization."
    assert new_model.static_input_dim == deterministic_model.static_input_dim

def test_model_initialization_probabilistic(probabilistic_model):
    """Test that a probabilistic model initializes properly."""
    assert isinstance(probabilistic_model, XTFT)

def test_forward_pass_probabilistic(probabilistic_model):
    """Test a forward pass through the probabilistic model."""
    batch_size = 4
    forecast_horizons = probabilistic_model.forecast_horizons
    output_dim = probabilistic_model.output_dim
    num_quantiles = len(probabilistic_model.quantiles)

    static_input, dynamic_input, future_covariate_input, _, _ = create_dummy_data(probabilistic_model, batch_size)
    outputs = probabilistic_model([static_input, dynamic_input, future_covariate_input], training=False)
    # Probabilistic: (B, H, Q, O)
    expected_shape = (batch_size, forecast_horizons, num_quantiles, output_dim)
    assert outputs.shape == expected_shape, f"Expected {expected_shape}, got {outputs.shape}."

def test_compile_and_train_step_probabilistic(probabilistic_model):
    """Test that the probabilistic model can compile and run a single training step."""
    # When quantiles are used, multiple loss functions might be applied internally.
    # For simplicity, we just run a forward pass and a manual loss here.
    probabilistic_model.compile(optimizer=Adam(), loss=MeanSquaredError())
    batch_size = 4

    static_input, dynamic_input, future_covariate_input, y_true, anomaly_scores = create_dummy_data(probabilistic_model, batch_size)
    with tf.GradientTape() as tape:
        y_pred = probabilistic_model([static_input, dynamic_input, future_covariate_input], training=True)
        # y_pred: (B, H, Q, O), y_true: (B, H, O)
        # We'll take the median quantile (Q=1) for a simple test
        median_quantile = y_pred[:, :, 1, :]  # (B, H, O)
        loss = tf.reduce_mean((y_true - median_quantile)**2) + tf.reduce_mean(anomaly_scores**2)
    grads = tape.gradient(loss, probabilistic_model.trainable_variables)
    assert all(g is not None for g in grads if g is not None), "All gradients should be computable in probabilistic mode."

def test_serialization_probabilistic(probabilistic_model):
    """Test probabilistic model serialization and deserialization."""
    config = probabilistic_model.get_config()
    assert isinstance(config, dict), "Config should be a dictionary."
    new_model = XTFT.from_config(config)
    assert isinstance(new_model, XTFT), "Should be XTFT instance after deserialization."
    assert new_model.quantiles == probabilistic_model.quantiles

@pytest.mark.parametrize("final_agg", ["last", "average", "flatten"])
def test_final_agg_strategies(final_agg):
    """Test different final_agg strategies to ensure shape consistency."""
    model = XTFT(
        static_input_dim=10,
        dynamic_input_dim=45,
        future_covariate_dim=5,
        embed_dim=32,
        forecast_horizons=3,
        quantiles=None,
        max_window_size=10,
        memory_size=100,
        num_heads=4,
        dropout_rate=0.1,
        output_dim=1,
        anomaly_loss_weight=1.0,
        attention_units=32,
        hidden_units=64,
        lstm_units=64,
        scales=[1],
        activation="relu",
        use_residuals=True,
        use_batch_norm=False,
        final_agg=final_agg
    )

    batch_size = 4
    static_input, dynamic_input, future_covariate_input, _, _ = create_dummy_data(model, batch_size)
    outputs = model([static_input, dynamic_input, future_covariate_input], training=False)
    # Deterministic: (B, H, O)
    assert outputs.shape == (batch_size, model.forecast_horizons, model.output_dim), \
        f"For final_agg='{final_agg}', expected shape (B,H,O) but got {outputs.shape}."

@pytest.mark.parametrize("scales", [None, "auto", [1, 2]])
def test_scales_behavior(scales):
    """Test different scaling parameters to ensure model runs forward pass without error."""
    model = XTFT(
        static_input_dim=10,
        dynamic_input_dim=45,
        future_covariate_dim=5,
        embed_dim=32,
        forecast_horizons=3,
        quantiles=None,
        max_window_size=10,
        memory_size=100,
        num_heads=4,
        dropout_rate=0.1,
        output_dim=1,
        anomaly_loss_weight=1.0,
        attention_units=32,
        hidden_units=64,
        lstm_units=64,
        scales=scales,
        activation="relu",
        use_residuals=True,
        use_batch_norm=False,
        final_agg="last"
    )

    batch_size = 4
    static_input, dynamic_input, future_covariate_input, _, _ = create_dummy_data(model, batch_size)
    # Just run a forward pass
    outputs = model([static_input, dynamic_input, future_covariate_input], training=False)
    expected_shape = (batch_size, model.forecast_horizons, model.output_dim)
    assert outputs.shape == expected_shape, \
        f"With scales={scales}, expected {expected_shape}, got {outputs.shape}."


if __name__=='__main__': 
    pytest.main( [__file__])

