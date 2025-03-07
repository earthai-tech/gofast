# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-
# nansha_tune_params.py
"""
This script fine-tunes the XTFT model using Keras Tuner (Bayesian
Optimization) for various forecasting cases of land subsidence.
Depending on the case selected, the model is configured either for 
quantile forecasts (e.g., q10, q50, q90) or for point forecasts (i.e.
quantiles = None) and for different forecast horizons (2022 or 
2023–2026).

Workflow:
----------
1. Define a hyperparameter search space for the XTFT model.
2. Perform hyperparameter tuning with Bayesian Optimization.
3. Select the best model based on validation loss.
4. Train the best model with the optimal hyperparameters.
5. Evaluate the model using R² score (and coverage score for quantile cases).
6. Save the predictions and visualize actual vs. predicted subsidence.

Author: LKouadio
"""

import os
import warnings

from typing import List, Any, Optional, Dict, List, Optional, Callable, Tuple 
import keras_tuner as kt
import tensorflow as tf
import pandas as pd
import numpy as np

import joblib
from tensorflow.keras.optimizers import Adam
from gofast.nn.transformers import XTFT
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import r2_score
from gofast.nn.transformers import XTFT
from gofast.metrics_special import coverage_score
from gofast.utils.data_utils import mask_by_reference
from gofast.plot import plot_spatial_features

# =============================================================================
# Global Definitions & Case Dictionary
# =============================================================================
DATA_PATH = r'E:\zhongshan_data'
DATA_FILE = 'zhongshan_final_dataset.csv'

# Define available cases with associated saved joblib file and quantiles setting.
CASE_DICT = {
    "QT_2022": {
         "data_file": "qt.xtft_data_2022.joblib",
         "quantiles": [0.1, 0.5, 0.9],
         "forecast_horizons": 1,
         "description": "Quantile forecast for 2022"
    },
    "PF_2022": {
         "data_file": "pf.xtft_data_2022.joblib",
         "quantiles": None,
         "forecast_horizons": 1,
         "description": "Point forecast for 2022"
    },
    "QT_2023_2026": {
         "data_file": "qt.xtft_data_2023_2026.joblib",
         "quantiles": [0.1, 0.5, 0.9],
         "forecast_horizons": 4,
         "description": "Quantile forecast for 2023–2026"
    },
    "PF_2023_2026": {
         "data_file": "pf.xtft_data_2023_2026.joblib",
         "quantiles": None,
         "forecast_horizons": 4,
         "description": "Point forecast for 2023–2026"
    }
}

# Specify the tuning case here. Options: "QT_2022", "PF_2022", "QT_2023_2026", "PF_2023_2026"
TUNE_CASE = "PF_2023_2026"

# =============================================================================
# Load the preprocessed data based on the selected case
# =============================================================================
case_info = CASE_DICT[TUNE_CASE]
data_dict = joblib.load(os.path.join(DATA_PATH, case_info["data_file"]))

X_static= data_dict['X_static']
y = data_dict['y']

X_static_train  = data_dict['X_static_train']
X_dynamic_train = data_dict['X_dynamic_train']
X_future_train  = data_dict['X_future_train']
y_train         = data_dict['y_train']

X_static_test   = data_dict.get('X_static_test', None)
X_dynamic_test  = data_dict.get('X_dynamic_test', None)
X_future_test   = data_dict.get('X_future_test', None)
y_test          = data_dict.get('y_test', None)

# Benchmark metric (if needed)
BENCHMARK_COV_SCORE = 0.5015992464550291

# =============================================================================
# Step 1: Define the Model-Building Function for Keras Tuner
# =============================================================================
def build_xtft_model(hp):
    """
    Build an XTFT model with hyperparameters for Keras Tuner for forecasting.
    
    Parameters
    ----------
    hp : keras_tuner.engine.hyperparameters.HyperParameters
        Hyperparameters object provided by Keras Tuner.
    
    Returns
    -------
    model : XTFT
        Compiled XTFT model instance configured per the selected case.
    """
    model = XTFT(
        static_input_dim  = X_static_train.shape[1],
        dynamic_input_dim = X_dynamic_train.shape[2],
        future_input_dim  = X_future_train.shape[2],
        forecast_horizons = case_info["forecast_horizons"],
        quantiles         = case_info["quantiles"],
        embed_dim         = hp.Choice('embed_dim', [16, 32, 64]),
        max_window_size   = hp.Choice('max_window_size', [3, 5, 10]),
        memory_size       = hp.Choice('memory_size', [50, 100, 200]),
        num_heads         = hp.Choice('num_heads', [2, 4, 8]),
        dropout_rate      = hp.Choice('dropout_rate', [0.1, 0.2, 0.3]),
        lstm_units        = hp.Choice('lstm_units', [32, 64, 128]),
        attention_units   = hp.Choice('attention_units', [32, 64, 128]),
        hidden_units      = hp.Choice('hidden_units', [32, 64, 128])
    )
    
    model.compile(
        optimizer = Adam(hp.Choice('learning_rate', [0.0001, 0.001, 0.01])),
        loss      = 'mse'
    )
    return model

# =============================================================================
# Step 2: Initialize Keras Tuner for Hyperparameter Search
# =============================================================================
tuner = kt.BayesianOptimization(
    build_xtft_model,
    objective    = 'val_loss',
    max_trials   = 5,
    directory    = os.path.join(DATA_PATH, 'z_tuning_results'),
    project_name = f'z_XTFT_{TUNE_CASE}_tuning'
)

# =============================================================================
# Step 3: Run Hyperparameter Search over Different Batch Sizes
# =============================================================================
batch_sizes = [16, 32, 64]
try:
    for bs in batch_sizes:
        print(f"Running tuner with Batch Size: {bs}")
        tuner.search(
            x              = [X_static_train, X_dynamic_train, X_future_train],
            y              = y_train,
            epochs         = 50,
            batch_size     = bs,
            validation_split = 0.2,
            callbacks      = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5
                )
            ]
        )
except Exception as e:
    warnings.warn(str(e))
    # Proceed with partial tuning results if error occurs.

# =============================================================================
# Step 4: Retrieve Best Hyperparameters and Train Best Model
# =============================================================================
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best Hyperparameters <{TUNE_CASE}: {best_hps.values}")

# Export the best hyperparameters to a text file.
output_txt_path = os.path.join(DATA_PATH, "{TUNE_CASE.lower()}_best_hps.txt")
with open(output_txt_path, "w") as f:
    f.write("Best Hyperparameters from Tuning {TUNE_CASE}:\n")
    for param, value in best_hps.values.items():
        f.write(f"{param}: {value}\n")

print(f"Best hyperparameters saved to: {output_txt_path}")


best_model = tuner.hypermodel.build(best_hps)
best_batch_size = min(batch_sizes, key=lambda x: abs(x - 32))

best_model.fit(
    x              = [X_static_train, X_dynamic_train, X_future_train],
    y              = y_train,
    epochs         = 50,
    batch_size     = best_batch_size,
    validation_split = 0.2
)


def xtft_tuner(
    inputs: List [np.array], 
    # X_static: np.ndarray,
    # X_dynamic: np.ndarray,
    # X_future: np.ndarray,
    y: np.ndarray,
    forecast_horizon: int=1, 
    quantiles=None, 
    param_grid: Dict[str, Any]=None, 
    case_info: Dict[str, Any]=None,
    max_trials: int = 10,
    objective: str = 'val_loss',
    epochs: int = 50,
    batch_sizes: List[int] = [16, 32, 64],
    validation_split: float = 0.2,
    tuner_dir: Optional[str] = None,
    project_name: Optional[str] = None,
    tuner_type: str = 'bayesian',
    callbacks: Optional[List[Any]] = None,
    model_builder: Optional[Callable] = None,
    verbose: int = 1
) -> Tuple[Dict[str, Any], tf.keras.Model, kt.Tuner]:
    r"""
    Fine-tunes the XTFT forecasting model using Keras Tuner with flexible
    hyperparameter search settings.

    This function sets up a hyperparameter tuning workflow using Keras Tuner,
    specifically employing Bayesian Optimization (by default) to search over
    a defined space of hyperparameters for the XTFT model. It supports multiple
    batch sizes to accommodate varying memory and training conditions, and
    it returns the best hyperparameter configuration, the best model, and the
    tuner instance for further inspection.

    The hyperparameter search is formulated as:

    .. math::
       \min_{\theta \in \Theta} \, L\big(\theta; \mathbf{X}, y\big)

    where :math:`\Theta` is the hyperparameter space, and
    :math:`L(\theta; \mathbf{X}, y)` is the validation loss computed over
    training data.

    Parameters
    ----------
    X_static_train    : np.ndarray
        Training array for static features; shape should be
        ``(n_samples, n_static_features)``.
    X_dynamic_train   : np.ndarray
        Training array for dynamic features; shape should be
        ``(n_samples, time_steps, n_dynamic_features)``.
    X_future_train    : np.ndarray
        Training array for future features; shape should be
        ``(n_samples, time_steps, n_future_features)``.
    y_train           : np.ndarray
        Target values for training; shape depends on forecast type.
    case_info         : dict
        Dictionary containing configuration for the tuning case,
        including keys such as ``"forecast_horizon"`` and ``"quantile"``.
    max_trials        : int, default=10
        Maximum number of hyperparameter trials to conduct during tuning.
    objective         : str, default='val_loss'
        The metric to optimize during hyperparameter tuning.
    epochs            : int, default=50
        Number of training epochs for each hyperparameter trial.
    batch_sizes       : list of int, default=[16, 32, 64]
        A list of batch sizes to try during the tuning process.
    validation_split  : float, default=0.2
        Fraction of training data to use for validation.
    tuner_dir         : str, optional
        Directory to save tuner results; if not provided, a temporary
        directory is used.
    project_name      : str, optional
        A name for the tuning project; used to organize tuner output files.
    tuner_type        : str, default='bayesian'
        The type of Keras Tuner to use. Currently only ``'bayesian'`` is
        supported.
    callbacks         : list, optional
        A list of Keras callbacks to use during model training (e.g.,
        EarlyStopping). If not provided, defaults to an EarlyStopping
        callback with patience 5.
    model_builder     : callable, optional
        A function that builds and compiles the XTFT model. If ``None``,
        a default builder is used which defines a hyperparameter search
        space over parameters such as ``embed_dim``, ``max_window_size``,
        ``memory_size``, ``num_heads``, ``dropout_rate``, ``lstm_units``,
        ``attention_units``, and ``hidden_units``. The model is compiled
        with the Adam optimizer and mean squared error loss.
    verbose`          : int, default=1
        Verbosity level; higher values produce more detailed output.

    Returns
    -------
    tuple
        A tuple containing:
            - dict: Best hyperparameters found.
            - tf.keras.Model: The best trained XTFT model.
            - kt.Tuner: The tuner instance used for hyperparameter search.

    Examples
    --------
    >>> from gofast.nn.temporal_tuner import tune_xtft_model
    >>> best_hps, best_model, tuner = tune_xtft_model(
    ...     X_static_train, X_dynamic_train, X_future_train, y_train,
    ...     case_info={'forecast_horizons': 4, 'quantiles': None},
    ...     max_trials=5,
    ...     epochs=50,
    ...     batch_sizes=[16, 32],
    ...     validation_split=0.2,
    ...     tuner_dir="tuning_results",
    ...     project_name="XTFT_Tuning",
    ...     verbose=1
    ... )
    >>> print("Best hyperparameters:", best_hps)
    >>> best_model.summary()

    Notes
    -----
    This function is designed to be robust and flexible. It iterates over
    a list of batch sizes to identify the one that best matches a target
    batch size (here, 32 is used as a default target). If errors occur during
    tuning for any batch size, a warning is issued and tuning continues with
    remaining values. The function utilizes Keras Tuner's Bayesian
    Optimization to efficiently search the hyperparameter space.

    See Also
    --------
    kt.BayesianOptimization : Keras Tuner's Bayesian Optimization tuner.
    tensorflow.keras.optimizers.Adam : Optimizer used for model training.
    XTFT : The transformer model used for forecasting in gofast.nn.
    """
    # assert inputs as expect three X_static, X_dynamic, X_future, 
    # and if numpy array, then force all dtypes to be np.float32 
    # and return , inputs 
    # if deep_check (True)(default) then 
    # 
    X_static, X_dynamic, X_future = inputs 
    
    # Define default model builder if none is provided.
    if model_builder is None:
        def default_model_builder(hp: kt.HyperParameters) -> tf.keras.Model:
            # Build XTFT model with hyperparameter search space.
            model = XTFT(
                static_input_dim  = X_static.shape[1],
                dynamic_input_dim = X_dynamic.shape[2],
                future_input_dim  = X_future.shape[2],
                forecast_horizon = case_info.get("forecast_horizon", 1),
                quantiles         = case_info.get("quantiles", None),
                embed_dim         = hp.Choice('embed_dim', [16, 32, 64]),
                max_window_size   = hp.Choice('max_window_size', [3, 5, 10]),
                memory_size       = hp.Choice('memory_size', [50, 100, 200]),
                num_heads         = hp.Choice('num_heads', [2, 4, 8]),
                dropout_rate      = hp.Choice('dropout_rate', [0.1, 0.2, 0.3]),
                lstm_units        = hp.Choice('lstm_units', [32, 64, 128]),
                attention_units   = hp.Choice('attention_units', [32, 64, 128]),
                hidden_units      = hp.Choice('hidden_units', [32, 64, 128])
            )
            model.compile(
                optimizer = Adam(
                    hp.Choice('learning_rate', [0.0001, 0.001, 0.01])
                ),
                loss = 'mse'
            )
            return model
        
        model_builder = default_model_builder

    # Use default callback if none provided.
    if callbacks is None:
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ]

    # Use a default tuner directory if not provided.
    tuner_dir = tuner_dir or os.path.join(os.getcwd(), "tuning_results")
    project_name = project_name or f"XTFT_Tuning_{case_info.get('description', '')}"

    # Initialize the Keras Tuner (Bayesian Optimization).
    tuner = kt.BayesianOptimization(
        model_builder,
        objective=objective,
        max_trials=max_trials,
        directory=tuner_dir,
        project_name=project_name
    )

    # Iterate over provided batch sizes; choose best one based on closeness to target.
    best_model = None
    best_hps = None
    best_val_loss = np.inf
    best_batch = None

    for bs in batch_sizes:
        if verbose:
            print(f"[INFO] Tuning with batch size: {bs}")
        try:
            tuner.search(
                x=[X_static, X_dynamic, X_future],
                y=y,
                epochs=epochs,
                batch_size=bs,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=verbose
            )
            # Retrieve the best hyperparameters for this batch size.
            current_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            # Build and train the model with these hyperparameters.
            model = tuner.hypermodel.build(current_hps)
            history = model.fit(
                x=[X_static, X_dynamic, X_future],
                y=y,
                epochs=epochs,
                batch_size=bs,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=verbose
            )
            current_val_loss = min(history.history['val_loss'])
            if verbose:
                print(
                    f"[INFO] Batch Size {bs}: Best val_loss = {current_val_loss:.4f}"
                )
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_hps = current_hps.values
                best_model = model
                best_batch = bs
        except Exception as e:
            warnings.warn(f"Tuning failed for batch size {bs}: {e}")
            continue

    if best_model is None:
        raise RuntimeError("Hyperparameter tuning failed for all batch sizes.")

    if verbose:
        print(f"[INFO] Best Batch Size: {best_batch}")
        print(f"[INFO] Best Hyperparameters: {best_hps}")

    return best_hps, best_model, tuner
