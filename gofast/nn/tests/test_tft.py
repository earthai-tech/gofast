# -*- coding: utf-8 -*-
import pytest
import numpy as np
import warnings 
from gofast.compat.tf import HAS_TF 
from gofast.nn.transformers import TemporalFusionTransformer
from gofast.nn.transformers import ( 
    GatedResidualNetwork, VariableSelectionNetwork, 
    StaticEnrichmentLayer, TemporalAttentionLayer
)
from gofast.utils.deps_utils import ensure_module_installed 

if not HAS_TF: 
    try:
        HAS_TF=ensure_module_installed("tensorflow", auto_install=True)
    except  Exception as e: 
        warnings.warn(f"Fail to install `tensorflow` library: {e}")
   
if HAS_TF: 
    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam
        
#%
@pytest.fixture
def model_params():
    """
    Fixture to provide model parameters for testing.
    """
    return {
        "static_input_dim": 10,
        "dynamic_input_dim": 5,
        "num_static_vars": 3,
        "num_dynamic_vars": 7,
        "hidden_units": 32,
        "num_heads": 4,
        "dropout_rate": 0.1
    }


@pytest.fixture
def dummy_data(model_params):
    """
    Fixture to create dummy input data for testing.
    """
    batch_size = 8
    time_steps = 20

    static_input_dim = model_params["static_input_dim"]
    dynamic_input_dim = model_params["dynamic_input_dim"]
    num_static_vars = model_params["num_static_vars"]
    num_dynamic_vars = model_params["num_dynamic_vars"]

    # Create random static inputs: (batch_size, num_static_vars,
    # static_input_dim)
    static_inputs = np.random.rand(
        batch_size,
        num_static_vars,
        static_input_dim
    ).astype(np.float32)

    # Create random dynamic inputs: (batch_size, time_steps, num_dynamic_vars, dynamic_input_dim)
    dynamic_inputs = np.random.rand(
        batch_size,
        time_steps,
        num_dynamic_vars,
        dynamic_input_dim
    ).astype(np.float32)

    # Create dummy targets: (batch_size, 1)
    targets = np.random.rand(batch_size, 1).astype(np.float32)

    return (static_inputs, dynamic_inputs), targets


def test_model_instantiation(model_params):
    """
    Test that the TemporalFusionTransformer can be instantiated with given parameters.
    """
    try:
        model = TemporalFusionTransformer(
            static_input_dim=model_params["static_input_dim"],
            dynamic_input_dim=model_params["dynamic_input_dim"],
            num_static_vars=model_params["num_static_vars"],
            num_dynamic_vars=model_params["num_dynamic_vars"],
            hidden_units=model_params["hidden_units"],
            num_heads=model_params["num_heads"],
            dropout_rate=model_params["dropout_rate"]
        )
    except Exception as e:
        pytest.fail(f"Model instantiation failed with exception: {e}")


def test_forward_pass(model_params, dummy_data):
    """
    Test that the model can perform a forward pass and
    produces output with the correct shape.
    """
    model = TemporalFusionTransformer(
        static_input_dim=model_params["static_input_dim"],
        dynamic_input_dim=model_params["dynamic_input_dim"],
        num_static_vars=model_params["num_static_vars"],
        num_dynamic_vars=model_params["num_dynamic_vars"],
        hidden_units=model_params["hidden_units"],
        num_heads=model_params["num_heads"],
        dropout_rate=model_params["dropout_rate"]
    )

    inputs, _ = dummy_data

    try:
        outputs = model(inputs, training=False)
    except Exception as e:
        pytest.fail(f"Forward pass failed with exception: {e}")

    # Check output shape: (batch_size, 1)
    expected_shape = (inputs[0].shape[0], 1, 1)
    assert outputs.shape == expected_shape, (
        f"Expected output shape {expected_shape}, but got {outputs.shape}"
    )


def test_training_step(model_params, dummy_data):
    """
    Test that the model can be compiled and perform a training step.
    """
    model = TemporalFusionTransformer(
        static_input_dim=model_params["static_input_dim"],
        dynamic_input_dim=model_params["dynamic_input_dim"],
        num_static_vars=model_params["num_static_vars"],
        num_dynamic_vars=model_params["num_dynamic_vars"],
        hidden_units=model_params["hidden_units"],
        num_heads=model_params["num_heads"],
        dropout_rate=model_params["dropout_rate"]
    )

    # Compile the model
    try:
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse'
        )
    except Exception as e:
        pytest.fail(f"Model compilation failed with exception: {e}")

    inputs, targets = dummy_data

    # Perform a single training step
    try:
        history = model.fit(
            inputs,
            targets,
            epochs=1,
            batch_size=8,
            # verbose=0, 
        )
    except Exception as e:
        pytest.fail(f"Training step failed with exception: {e}")

    # Optionally, check that loss is a float
    loss = history.history.get('loss', [None])[0]
    assert isinstance(loss, float), f"Expected loss to be float, got {type(loss)}"


def test_output_consistency(model_params, dummy_data):
    """
    Test that the model produces consistent output shapes across multiple forward passes.
    """
    model = TemporalFusionTransformer(
        static_input_dim=model_params["static_input_dim"],
        dynamic_input_dim=model_params["dynamic_input_dim"],
        num_static_vars=model_params["num_static_vars"],
        num_dynamic_vars=model_params["num_dynamic_vars"],
        hidden_units=model_params["hidden_units"],
        num_heads=model_params["num_heads"],
        dropout_rate=model_params["dropout_rate"]
    )

    inputs, _ = dummy_data

    outputs1 = model(inputs, training=False)
    outputs2 = model(inputs, training=False)

    assert outputs1.shape == outputs2.shape, (
        f"Inconsistent output shapes: {outputs1.shape} vs {outputs2.shape}"
    )

def test_zero_batch_size(model_params):
    """
    Test the model's behavior when batch size is zero.
    """
    model = TemporalFusionTransformer(
        static_input_dim=model_params["static_input_dim"],
        dynamic_input_dim=model_params["dynamic_input_dim"],
        num_static_vars=model_params["num_static_vars"],
        num_dynamic_vars=model_params["num_dynamic_vars"],
        hidden_units=model_params["hidden_units"],
        num_heads=model_params["num_heads"],
        dropout_rate=model_params["dropout_rate"]
    )

    # Zero batch size
    static_inputs = np.random.rand(
        0,
        model_params["num_static_vars"],
        model_params["static_input_dim"]
    ).astype(np.float32)

    dynamic_inputs = np.random.rand(
        0,
        20,
        model_params["num_dynamic_vars"],
        model_params["dynamic_input_dim"]
    ).astype(np.float32)

    try:
        outputs = model(
            (static_inputs, dynamic_inputs),
            training=False
        )
        expected_shape = (0, 1, 1)
        assert outputs.shape == expected_shape, (
            f"Expected output shape {expected_shape}, got {outputs.shape}"
        )
    except Exception as e:
        pytest.fail(f"Model failed with zero batch size: {e}")


def test_dropout_behavior(model_params, dummy_data):
    """
    Test that dropout is active during training and inactive during inference.
    """
    model = TemporalFusionTransformer(
        static_input_dim=model_params["static_input_dim"],
        dynamic_input_dim=model_params["dynamic_input_dim"],
        num_static_vars=model_params["num_static_vars"],
        num_dynamic_vars=model_params["num_dynamic_vars"],
        hidden_units=model_params["hidden_units"],
        num_heads=model_params["num_heads"],
        dropout_rate=model_params["dropout_rate"]
    )

    inputs, _ = dummy_data

    # Set dropout rate to a high value for testing
    model.dropout_rate = 0.99
    for layer in model.layers:
        if hasattr(layer, 'rate'):
            layer.rate = 0.99

    # Forward pass with training=True
    outputs_training = model(inputs, training=True)

    # Forward pass with training=False
    outputs_inference = model(inputs, training=False)

    # Since dropout is active during training, outputs should differ
    tf.random.set_seed(42)
    outputs_training_2 = model(inputs, training=True)

    # It's possible (though unlikely) for outputs to be the same due to high dropout,
    # but generally, they should differ
    assert not np.array_equal(
        outputs_training.numpy(),
        outputs_inference.numpy()
    ), "Outputs should differ between training and inference due to dropout"

@pytest.mark.skip ("loading falls an issue based on the temporay save file."
                   " But work fine outside the tempdir.")
def test_model_serialization(model_params, dummy_data):
    """
    Test that the model can be saved and loaded correctly.
    """
    model = TemporalFusionTransformer(
        static_input_dim=model_params["static_input_dim"],
        dynamic_input_dim=model_params["dynamic_input_dim"],
        num_static_vars=model_params["num_static_vars"],
        num_dynamic_vars=model_params["num_dynamic_vars"],
        hidden_units=model_params["hidden_units"],
        num_heads=model_params["num_heads"],
        dropout_rate=model_params["dropout_rate"]
    )

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='mse'
    )
    
    inputs, _ = dummy_data

    # Save the model to a temporary directory
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, 'tft_model')
        
        try:
            model.save(save_path, save_format='tf')
        except Exception as e:
            pytest.fail(f"Model serialization failed with exception: {e}")

        # Load the model
        try:
            loaded_model = tf.keras.models.load_model(
                save_path,
                custom_objects={
                    'TemporalFusionTransformer': TemporalFusionTransformer,
                    'GatedResidualNetwork':GatedResidualNetwork,
                    'VariableSelectionNetwork':VariableSelectionNetwork,
                    'TemporalAttentionLayer':TemporalAttentionLayer,
                    'StaticEnrichmentLayer':StaticEnrichmentLayer
                }
            )
        except Exception as e:
            pytest.fail(f"Model deserialization failed with exception: {e}")

        # Test that the loaded model produces the same output
        original_output = model(dummy_data[0], training=False)
        loaded_output = loaded_model(dummy_data[0], training=False)

        assert np.allclose(
            original_output.numpy(),
            loaded_output.numpy(),
            atol=1e-5
        ), "Loaded model outputs do not match the original model outputs"

if __name__=='__main__': 
    pytest.main( [__file__])