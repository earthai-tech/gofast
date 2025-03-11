# -*- coding: utf-8 -*-
import warnings
import pytest
from gofast.compat.tf import HAS_TF 
from gofast.utils.deps_utils import ensure_module_installed 

if not HAS_TF: 
    try:
        HAS_TF=ensure_module_installed("tensorflow", auto_install=True)
    except  Exception as e: 
        warnings.warn(f"Fail to install `tensorflow` library: {e}")
   
if HAS_TF: 
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


if __name__=='__main__': 
    pytest.main( [__file__])

