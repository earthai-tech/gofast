# -*- coding: utf-8 -*-

import warnings
import pytest
import numpy as np
from gofast.compat.tf import HAS_TF 
from gofast.utils.deps_utils import ensure_module_installed 

if not HAS_TF: 
    try:
        HAS_TF=ensure_module_installed("tensorflow", auto_install=True)
    except  Exception as e: 
        warnings.warn(f"Fail to install `tensorflow` library: {e}")
   
if HAS_TF: 
    import tensorflow as tf
    from gofast.nn.transformers import XTFT


@pytest.fixture
def sample_data():
    """Generate synthetic test data"""
    batch_size = 32
    time_steps = 24
    horizon = 12
    
    return {
        'static': np.random.randn(batch_size, 5),
        'dynamic': np.random.randn(batch_size, time_steps, 10),
        'future': np.random.randn(batch_size, time_steps, 3),
        'target': np.random.randn(batch_size, horizon, 1)
    }

def test_feature_based_anomaly_scores(sample_data):
    """Test feature-based anomaly score generation"""
    model = XTFT(
        static_input_dim=5,
        dynamic_input_dim=10,
        future_input_dim=3,
        forecast_horizon=12,
        anomaly_detection_strategy='feature_based',
        anomaly_loss_weight=0.3,
        output_dim=1,
        embed_dim=32,
        scales=[3, 6],
        final_agg='last'
    )
    # Test forward pass
    inputs = [sample_data['static'], sample_data['dynamic'], sample_data['future']]
    predictions = model(inputs) # noqa
    
    # Verify anomaly scores generation
    assert hasattr(model, 'anomaly_scores'), "Model should have anomaly_scores attribute"
    assert model.anomaly_scores.shape == (32, 24, 1), \
        f"Unexpected anomaly scores shape: {model.anomaly_scores.shape}"
        
    # Verify loss components
    model.compile(optimizer='adam')
    loss = model.test_on_batch(inputs, sample_data['target'])
    assert isinstance(loss, float), "Loss should be computed"
    assert loss > 0, "Loss should be positive"

def test_prediction_based_anomaly_training(sample_data):
    """Test prediction-based anomaly scoring through training"""
    model = XTFT(
        static_input_dim=5,
        dynamic_input_dim=10,
        future_input_dim=3,
        forecast_horizon=12,
        anomaly_detection_strategy='prediction_based',
        anomaly_loss_weight=0.5,
        output_dim=1,
        embed_dim=32
    )
    
    model.compile(optimizer='adam', loss='mse')
    
    # Train for a few epochs
    initial_loss = None
    for _ in range(3):
        loss = model.train_on_batch(
            [sample_data['static'], sample_data['dynamic'], sample_data['future']],
            sample_data['target']
        )
        if initial_loss is None:
            initial_loss = loss
        else:
            assert loss < initial_loss * 2, "Loss should decrease or stabilize"
    
    # Verify prediction-based scoring
    predictions = model.predict(
        [sample_data['static'], sample_data['dynamic'], 
          sample_data['future']]
    )
    errors = np.abs(sample_data['target'] - predictions)
    assert errors.shape == (32, 12, 1), "Prediction errors should match target shape"

def test_config_based_anomaly_scores(sample_data):
    """Test externally provided anomaly scores"""
    # Generate mock anomaly scores
    anomaly_scores = np.random.rand(32, 12)
    
    model = XTFT(
        static_input_dim=5,
        dynamic_input_dim=10,
        future_input_dim=3,
        forecast_horizon=12,
        anomaly_detection_strategy='from_config',
        anomaly_config={
            'anomaly_scores': anomaly_scores,
            'anomaly_loss_weight': 0.8
        },
        output_dim=1
    )
    
    # Verify score loading
    assert model.anomaly_scores.shape == (32, 12), \
        "Should load external anomaly scores"
    
    # Test training integration
    model.compile(optimizer='adam')
    loss = model.test_on_batch(
        [sample_data['static'], sample_data['dynamic'], sample_data['future']],
        sample_data['target']
    )
    assert loss > np.mean(anomaly_scores**2) * 0.7, \
        "Loss should reflect external anomaly scores"
        
@pytest.mark.skip (
    "Work perfectly but allucinating sometimes"
    " for catching the UserWarnings.")
def test_invalid_anomaly_config(sample_data):
    """Test invalid anomaly configurations"""
    with pytest.raises(ValueError):
        # Wrong anomaly scores shape
        XTFT(
            static_input_dim=5,
            dynamic_input_dim=10,
            future_input_dim=3,
            forecast_horizon=12,
            anomaly_detection_strategy='from_config',
            anomaly_config={
                'anomaly_scores': np.random.rand(32, 10)  # Should be 12 horizon
            }
        )
    
    # Test missing config warning
    with pytest.warns(UserWarning):
        model = XTFT(
            static_input_dim=5,
            dynamic_input_dim=10,
            future_input_dim=3,
            forecast_horizon=12,
            anomaly_detection_strategy='from_config'
        )
        assert model.anomaly_detection_strategy is None, \
            "Should disable invalid strategy"

def test_mixed_strategy_behavior(sample_data):
    """Test interaction between different strategies"""
    # Test feature-based + config combination
    model = XTFT(
        static_input_dim=5,
        dynamic_input_dim=10,
        future_input_dim=3,
        forecast_horizon=12,
        anomaly_detection_strategy='feature_based',
        anomaly_config={
            'anomaly_scores': np.random.rand(32, 12),
            'anomaly_loss_weight': 0.5
        }
    )
    
    # Should prioritize strategy over config
    inputs = [sample_data['static'], sample_data['dynamic'], sample_data['future']]
    _ = model(inputs)
    assert not hasattr(model, 'external_anomaly_scores'), \
        "Should ignore external scores in feature-based mode"

def test_anomaly_loss_calculation(sample_data):
    """Verify anomaly loss component calculations"""
    # Setup model with known weights
    model = XTFT(
        static_input_dim=5,
        dynamic_input_dim=10,
        future_input_dim=3,
        forecast_horizon=12,
        anomaly_detection_strategy='feature_based',
        anomaly_loss_weight=0.5,
        output_dim=1
    )
    
    dummy_input = tf.zeros((1, 64))  # assuming the expected input has shape (batch, 64)
    _ = model.anomaly_scorer(dummy_input)
    model.anomaly_scorer.set_weights([
        np.zeros((64, 1)),  # Weights (assuming 64-dim attention output)
        np.zeros(1)         # Biases
    ])
    
    # Calculate predictable anomaly scores
    inputs = [sample_data['static'], sample_data['dynamic'], sample_data['future']]
    _ = model(inputs)
    expected_scores = np.zeros((32, 12, 1))
    
    # Verify loss calculation
    model.compile(optimizer='adam')
    loss = model.test_on_batch(inputs, sample_data['target'])
    
    # Calculate expected loss components
    pred_loss = np.mean((model(inputs).numpy() - sample_data['target'])**2)
    anomaly_loss = np.mean(expected_scores**2) * 0.5
    expected_total = pred_loss + anomaly_loss
    
    assert np.isclose(loss, expected_total, rtol=1e-3), \
        "Loss calculation mismatch in feature-based mode"

def test_training_with_all_strategies(sample_data):
    """End-to-end training test for all strategies"""
    strategies = ['feature_based', 'prediction_based', 'from_config']
    
    for strategy in strategies:
        model = XTFT(
            static_input_dim=5,
            dynamic_input_dim=10,
            future_input_dim=3,
            forecast_horizon=12,
            anomaly_detection_strategy=strategy,
            anomaly_config={
                'anomaly_scores': np.random.rand(32, 12),
                'anomaly_loss_weight': 0.3
            } if strategy == 'from_config' else None,
            output_dim=1
        )
        
        model.compile(optimizer='adam', loss='mse')
        
        # Train for 2 epochs
        losses = []
        for _ in range(2):
            loss = model.train_on_batch(
                [sample_data['static'], sample_data['dynamic'], sample_data['future']],
                sample_data['target']
            )
            losses.append(loss)
        
        # Verify training progression
        assert losses[1] <= losses[0] * 1.5, \
            f"Loss should stabilize/decrease in {strategy} strategy"
            
            
if __name__=='__main__': 
    pytest.main([__file__])