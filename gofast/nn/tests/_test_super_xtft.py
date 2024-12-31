# -*- coding: utf-8 -*-
# test_super_xtft.py

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Import the SuperXTFT model and necessary components
from gofast.nn.transformers import SuperXTFT
from gofast.nn._tft import VariableSelectionNetwork, GatedResidualNetwork

# Suppress TensorFlow warnings for cleaner test output
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# ------------------------ Fixtures ------------------------ #

@pytest.fixture(scope="module")
def synthetic_data():
    """
    Generates synthetic data for testing the SuperXTFT model.
    
    Returns:
        tuple: Contains training and testing datasets.
    """
    num_samples = 100
    time_steps = 20
    num_static_vars = 5
    num_dynamic_vars = 3
    dynamic_input_dim = 10
    static_input_dim = 8
    future_covariate_dim = 4
    forecast_horizons = 5  # Number of future time steps to predict
    output_dim = 1  # Number of target variables
    quantiles = [0.1, 0.5, 0.9]

    # Generate synthetic static features: (num_samples, num_static_vars, static_input_dim)
    static_features = np.random.rand(num_samples, num_static_vars, static_input_dim).astype(np.float32)

    # Generate synthetic dynamic features: (num_samples, num_dynamic_vars, time_steps, dynamic_input_dim)
    dynamic_features = np.random.rand(num_samples, num_dynamic_vars, time_steps, dynamic_input_dim).astype(np.float32)

    # Generate synthetic future covariates: (num_samples, num_dynamic_vars, forecast_horizons, future_covariate_dim)
    future_covariates = np.random.rand(num_samples, num_dynamic_vars, forecast_horizons, future_covariate_dim).astype(np.float32)

    # Generate synthetic targets: (num_samples, forecast_horizons, output_dim)
    targets = np.random.rand(num_samples, forecast_horizons, output_dim).astype(np.float32)

    # Split into training and testing sets
    train_size = int(0.8 * num_samples)
    X_train = {
        'static_input': static_features[:train_size],
        'dynamic_input': dynamic_features[:train_size],
        'future_covariate_input': future_covariates[:train_size]
    }
    y_train = targets[:train_size]

    X_test = {
        'static_input': static_features[train_size:],
        'dynamic_input': dynamic_features[train_size:],
        'future_covariate_input': future_covariates[train_size:]
    }
    y_test = targets[train_size:]

    return (X_train, y_train), (X_test, y_test)

@pytest.fixture(scope="module")
def super_xtft_model():
    """
    Instantiates the SuperXTFT model with predefined parameters.
    
    Returns:
        SuperXTFT: An instance of the SuperXTFT model.
    """
    model_params = {
        'static_input_dim': 8,
        'dynamic_input_dim': 10,
        'future_covariate_dim': 4,
        'num_static_vars': 5,
        'num_dynamic_vars': 3,
        'embed_dim': 64,
        'forecast_horizons': 5,
        'quantiles': [0.1, 0.5, 0.9],
        'max_window_size': 15,
        'memory_size': 50,
        'num_heads': 4,
        'dropout_rate': 0.2,
        'output_dim': 1,
        'anomaly_config': {
            'anomaly_scores': None,  # No anomaly scores for this test
            'anomaly_loss_weight': None
        },
        'attention_units': 64,
        'hidden_units': 128,
        'lstm_units': 64,
        'scales': [1, 2, 3],
        'multi_scale_agg': 'average',
        'activation': 'relu',
        'use_residuals': True,
        'use_batch_norm': True,
        'final_agg': 'last'
    }

    model = SuperXTFT(**model_params)
    return model

# ------------------------ Test Cases ------------------------ #

def test_model_instantiation(super_xtft_model):
    """
    Test that the SuperXTFT model is instantiated correctly.
    """
    assert super_xtft_model is not None, "Model instantiation failed."
    assert isinstance(super_xtft_model, SuperXTFT), "Model is not an instance of SuperXTFT."
    
    # Check that VariableSelectionNetworks are correctly integrated
    assert hasattr(super_xtft_model, 'variable_selection_static'), "Missing variable_selection_static attribute."
    assert hasattr(super_xtft_model, 'variable_selection_dynamic'), "Missing variable_selection_dynamic attribute."
    
    # Check that GatedResidualNetworks are correctly integrated
    assert hasattr(super_xtft_model, 'grn_attention_hierarchical'), "Missing grn_attention_hierarchical attribute."
    assert hasattr(super_xtft_model, 'grn_attention_cross'), "Missing grn_attention_cross attribute."
    assert hasattr(super_xtft_model, 'grn_decoder'), "Missing grn_decoder attribute."

def test_model_compile(super_xtft_model):
    """
    Test that the SuperXTFT model compiles successfully.
    """
    optimizer = Adam(learning_rate=0.001)
    try:
        super_xtft_model.compile(optimizer=optimizer)
    except Exception as e:
        pytest.fail(f"Model compilation failed with error: {e}")
    
    # Check if optimizer is set correctly
    assert super_xtft_model.optimizer is not None, "Optimizer not set after compilation."

def test_model_training(super_xtft_model, synthetic_data):
    """
    Test that the SuperXTFT model can train on synthetic data without errors.
    """
    (X_train, y_train), _ = synthetic_data
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    try:
        history = super_xtft_model.fit(
            x=X_train,
            y=y_train,
            validation_split=0.1,
            epochs=10,
            batch_size=16,
            callbacks=[early_stopping],
            verbose=0  # Set to 1 for detailed logs
        )
    except Exception as e:
        pytest.fail(f"Model training failed with error: {e}")
    
    # Check that training reduced the loss
    assert 'val_loss' in history.history, "Validation loss not recorded."
    assert history.history['val_loss'][-1] < history.history['loss'][-1], \
        "Validation loss did not decrease."

def test_model_prediction(super_xtft_model, synthetic_data):
    """
    Test that the SuperXTFT model can make predictions with the correct output shape.
    """
    _, (X_test, _) = synthetic_data
    
    try:
        predictions = super_xtft_model.predict(X_test, verbose=0)
    except Exception as e:
        pytest.fail(f"Model prediction failed with error: {e}")
    
    # Expected shape: (num_test_samples, forecast_horizons, Q, output_dim)
    num_test_samples = X_test['static_input'].shape[0]
    forecast_horizons = 5
    Q = 3  # Number of quantiles
    output_dim = 1
    
    assert predictions.shape == (num_test_samples, forecast_horizons, Q, output_dim), \
        f"Unexpected prediction shape: {predictions.shape}"
    
    # Check that predictions contain valid numerical values
    assert not np.isnan(predictions).any(), "Predictions contain NaN values."
    assert not np.isinf(predictions).any(), "Predictions contain infinite values."

def test_invalid_parameters():
    """
    Test that the SuperXTFT model raises errors with invalid parameters.
    """
    with pytest.raises(ValueError):
        # Attempt to instantiate with invalid static_input_dim
        model = SuperXTFT(
            static_input_dim=-1,  # Invalid
            dynamic_input_dim=10,
            future_covariate_dim=4,
            num_static_vars=5,
            num_dynamic_vars=3,
            embed_dim=64,
            forecast_horizons=5,
            quantiles=[0.1, 0.5, 0.9],
            max_window_size=15,
            memory_size=50,
            num_heads=4,
            dropout_rate=0.2,
            output_dim=1,
            anomaly_config={
                'anomaly_scores': None,
                'anomaly_loss_weight': None
            },
            attention_units=64,
            hidden_units=128,
            lstm_units=64,
            scales=[1, 2, 3],
            multi_scale_agg='average',
            activation='relu',
            use_residuals=True,
            use_batch_norm=True,
            final_agg='last'
        )

def test_anomaly_loss_handling(super_xtft_model, synthetic_data):
    """
    Test that the SuperXTFT model correctly handles anomaly loss when anomaly_scores are provided.
    """
    (X_train, y_train), (X_test, y_test) = synthetic_data
    
    # Modify anomaly_config to include anomaly_scores
    # For testing, we'll use random anomaly scores
    anomaly_scores_train = np.random.rand(X_train['static_input'].shape[0], 5, 64).astype(np.float32)
    anomaly_scores_test = np.random.rand(X_test['static_input'].shape[0], 5, 64).astype(np.float32)
    
    # Update the model's anomaly_config
    super_xtft_model.anomaly_config['anomaly_scores'] = anomaly_scores_train
    super_xtft_model.anomaly_config['anomaly_loss_weight'] = 1.0
    
    # Recompile the model to include the anomaly loss
    optimizer = Adam(learning_rate=0.001)
    try:
        super_xtft_model.compile(optimizer=optimizer)
    except Exception as e:
        pytest.fail(f"Model compilation with anomaly loss failed with error: {e}")
    
    # Train the model
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    try:
        history = super_xtft_model.fit(
            x=X_train,
            y=y_train,
            validation_split=0.1,
            epochs=10,
            batch_size=16,
            callbacks=[early_stopping],
            verbose=0  # Set to 1 for detailed logs
        )
    except Exception as e:
        pytest.fail(f"Model training with anomaly loss failed with error: {e}")
    
    # Check that training reduced the loss
    assert 'val_loss' in history.history, "Validation loss not recorded for anomaly loss."
    assert history.history['val_loss'][-1] < history.history['loss'][-1], \
        "Validation loss did not decrease when anomaly loss was included."
    
    # Make predictions and ensure anomaly loss is handled
    try:
        predictions = super_xtft_model.predict(X_test, verbose=0)
    except Exception as e:
        pytest.fail(f"Model prediction with anomaly loss failed with error: {e}")
    
    assert predictions.shape == (X_test['static_input'].shape[0], 5, 3, 1), \
        f"Unexpected prediction shape with anomaly loss: {predictions.shape}"
    assert not np.isnan(predictions).any(), "Predictions contain NaN values with anomaly loss."
    assert not np.isinf(predictions).any(), "Predictions contain infinite values with anomaly loss."

def test_model_serialization(super_xtft_model, synthetic_data):
    """
    Test that the SuperXTFT model can be saved and loaded correctly.
    """
    # Save the model
    try:
        super_xtft_model.save('super_xtft_test_model.h5')
    except Exception as e:
        pytest.fail(f"Model saving failed with error: {e}")
    
    # Load the model
    try:
        loaded_model = tf.keras.models.load_model('super_xtft_test_model.h5', custom_objects={'SuperXTFT': SuperXTFT})
    except Exception as e:
        pytest.fail(f"Model loading failed with error: {e}")
    
    # Check that the loaded model has the same configuration
    assert loaded_model.get_config() == super_xtft_model.get_config(), "Loaded model config does not match original."
    
    # Make predictions with the loaded model
    _, (X_test, _) = synthetic_data
    try:
        loaded_predictions = loaded_model.predict(X_test, verbose=0)
    except Exception as e:
        pytest.fail(f"Loaded model prediction failed with error: {e}")
    
    # Ensure predictions from original and loaded models are identical
    original_predictions = super_xtft_model.predict(X_test, verbose=0)
    assert np.allclose(original_predictions, loaded_predictions, atol=1e-5), \
        "Predictions from loaded model do not match original model."
    
    # Clean up saved model file
    import os
    if os.path.exists('super_xtft_test_model.h5'):
        os.remove('super_xtft_test_model.h5')

# ------------------------ Run Tests ------------------------ #

# To execute the tests, run the following command in your terminal:
# pytest test_super_xtft.py

# Alternatively, within a Jupyter notebook or interactive environment, you can run:
# !pytest test_super_xtft.py

if __name__=='__main__': 
    pytest.main([__file__])