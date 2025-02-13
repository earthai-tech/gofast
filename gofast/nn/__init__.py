# -*- coding: utf-8 -*-
"""
Provides classes and functions for advanced model training, hyperparameter
tuning, and architecture search using deep learning models. It is designed to 
work with TensorFlow and offers utilities for efficient model evaluation, 
tuning strategies like Hyperband and Population-Based Training (PBT), and 
various model-building utilities.

Note: This module requires TensorFlow to be installed. If TensorFlow is not 
available, the module will raise an ImportError with instructions to install 
TensorFlow.

"""
import warnings
from .generate import create_sequences, data_generator
from ..compat.tf import import_keras_dependencies, check_keras_backend, standalone_keras
from ._config import configure_dependencies, Config as config

# Set default configuration
config.INSTALL_DEPS = False
config.WARN_STATUS = 'warn'

# Custom message for missing dependencies
EXTRA_MSG = ( 
    "`nn` sub-package expects the `tensorflow` or"
    " `keras` library to be installed."
    )

# Configure and install dependencies if needed
configure_dependencies(install_dependencies=config.INSTALL_DEPS)

# Lazy-load Keras dependencies
KERAS_DEPS={}
try:
    KERAS_DEPS = import_keras_dependencies(
        extra_msg=EXTRA_MSG, error='ignore')
except BaseException as e:
    warnings.warn(f"{EXTRA_MSG}: {e}")

# Check if TensorFlow or Keras is installed
KERAS_BACKEND = check_keras_backend(error='ignore')

def dependency_message(module_name):
    """
    Generate a custom message for missing dependencies.

    Parameters
    ----------
    module_name : str
        The name of the module that requires the dependencies.

    Returns
    -------
    str
        A message indicating the required dependencies.
    """
    return (
        f"`{module_name}` needs either the `tensorflow` or `keras` package to be "
        "installed. Please install one of these packages to use this function."
    )

if KERAS_BACKEND:
    
    from .build_models import (
        build_lstm_model, build_mlp_model, create_attention_model, 
        create_autoencoder_model, create_cnn_model, create_lstm_model
    )
    from .train import (
        calculate_validation_loss, cross_validate_lstm, evaluate_model, 
        make_future_predictions, plot_errors, plot_history, plot_predictions, 
        train_and_evaluate, train_and_evaluate2, train_epoch, train_model
    )
    from .tune import (
        Hyperband, PBTTrainer, base_tuning, custom_loss, deep_cv_tuning, 
        fair_neural_tuning, find_best_lr, lstm_ts_tuner, robust_tuning
    )
    __all__ = [
        "plot_history",
        "base_tuning",
        "robust_tuning",
        "build_mlp_model",
        "fair_neural_tuning",
        "deep_cv_tuning",
        "train_and_evaluate2",
        "train_and_evaluate",
        "Hyperband",
        'PBTTrainer',
        "custom_loss",
        "train_epoch",
        "calculate_validation_loss",
        "data_generator",
        "evaluate_model",
        "train_model",
        "create_lstm_model",
        "create_cnn_model",
        "create_autoencoder_model",
        "create_attention_model",
        "plot_errors",
        "plot_predictions",
        "find_best_lr",
        "create_sequences",
        "make_future_predictions",
        "build_lstm_model",
        "lstm_ts_tuner",
        "cross_validate_lstm",
    ]

    # Get necessary classes and functions from Keras dependencies
    Layer = KERAS_DEPS.Layer 
    # Equivalent to: from tensorflow.keras import activations
    try:
        activations = KERAS_DEPS.activations  
    except (ImportError, AttributeError) as e: 
        try: 
            activations = standalone_keras('activations')
        except: 
            raise ImportError (str(e))
    except: 
        raise ImportError(
                "Module 'activations' could not be imported from either "
                "tensorflow.keras or standalone keras. Ensure that TensorFlow "
                "or standalone Keras is installed and the module exists."
            )

    class Activation(Layer):
        """
        Custom Activation layer that wraps a Keras activation function
        and captures its name.
        """
        def __init__(self, activation='relu', **kwargs):
            super(Activation, self).__init__(**kwargs)
            # Get the activation function; Keras will raise an error if invalid
            self.activation= activations.get(activation)
            # self.activation = activation  # Store the original activation parameter

            # Assign activation name
            if isinstance(activation, str):
                self.activation_name = activation
            elif callable(activation):
                # Try to get the name from the activation function
                self.activation_name = getattr(
                    activation, '__name__', activation.__class__.__name__)
            else:
                # Fallback to string representation
                self.activation_name = str(activation)

        def call(self, inputs):
            return self.activation(inputs)

        def get_config(self):
            config = super(Activation, self).get_config()
            # Serialize the activation function properly
            config.update({
                'activation': activations.serialize(self.activation)
            })
            return config

        def __repr__(self):
            return f"{self.__class__.__name__}(activation={self.activation_name!r})"

    # Add Activation to the list of public objects if __all__ is defined
    __all__.extend(["Activation"])

    