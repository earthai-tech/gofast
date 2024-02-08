# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
The `deep_search` module provides classes and functions for advanced model 
training, hyperparameter tuning, and architecture search using deep learning 
models. It is designed to work with TensorFlow and offers utilities for 
efficient model evaluation, tuning strategies like Hyperband and Population-Based 
Training (PBT), and various model-building utilities.

Note: This module requires TensorFlow to be installed. If TensorFlow is not 
available, the module will raise an ImportError with instructions to install TensorFlow.


"""
import os
import datetime
import warnings 
import copy 
import itertools
import json
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold

from .._typing import List, Optional, Union, Dict, Tuple
from .._typing import ArrayLike , Callable, Any, Generator
from ..tools._dependency import import_optional_dependency 
from ..tools.funcutils import is_iterable, denormalize
try: 
    extra_msg = "`deep_search` module expects the `tensorflow` library to be installed."
    import_optional_dependency('tensorflow', extra=extra_msg)
    import tensorflow as tf
except BaseException as e : 
    warnings.warn(str(e) )
else: 
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.layers import LSTM, Input
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
    from tensorflow.keras.layers import Attention, Concatenate
    from tensorflow.keras.callbacks import Callback, History, TensorBoard
    from tensorflow.keras.callbacks import EarlyStopping 
    from tensorflow.keras import regularizers
    from tensorflow.keras.optimizers import Optimizer, Adam, SGD, RMSprop
    from tensorflow.keras.losses import Loss
    from tensorflow.keras.metrics import Metric
    from tensorflow.keras.models import load_model
    
__all__=["plot_history", "base_tuning", "robust_tuning","build_mlp_model", 
         "fair_neural_tuning", "deep_cv_tuning", "train_and_evaluate2", 
         "train_and_evaluate", "Hyperband",'PBTTrainer', "custom_loss", 
         "train_epoch", "calculate_validation_loss","data_generator",
         "evaluate_model", "train_model","create_lstm_model","create_cnn_model",
         "create_autoencoder_model" ,"create_attention_model", "plot_errors", 
         "plot_predictions"]

def plot_history(
    history: History, 
    title: str = 'Model Learning Curve',
    color_scheme: Optional[Dict[str, str]] = None,
    xlabel: str = 'Epoch',
    ylabel_acc: str = 'Accuracy',
    ylabel_loss: str = 'Loss',
    figsize: Tuple[int, int] = (12, 6)
    ) -> Tuple[plt.Axes, plt.Axes]:
    """
    Plots the learning curves of the training process for both accuracy and loss,
    and returns the axes for further customization if needed.

    Parameters
    ----------
    history : History object
        This is the output from the fit function of the Keras model, which contains
        the history of accuracy and loss values during training.
    title : str, optional
        Title of the plot. Default is 'Model Learning Curve'.
    color_scheme : dict of str, optional
        Dictionary containing color settings for the plot. Keys should include
        'train_acc', 'val_acc', 'train_loss', and 'val_loss'. Default is None.
    xlabel : str, optional
        Label for the x-axis. Default is 'Epoch'.
    ylabel_acc : str, optional
        Label for the y-axis of the accuracy plot. Default is 'Accuracy'.
    ylabel_loss : str, optional
        Label for the y-axis of the loss plot. Default is 'Loss'.
    figsize : tuple of int, optional
        Size of the figure (width, height) in inches. Default is (12, 6).

    Returns
    -------
    ax1, ax2 : tuple of matplotlib.axes._subplots.AxesSubplot
        Axes objects for the accuracy and loss plots, respectively.

    Examples
    --------
    >>> history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)
    >>> ax_acc, ax_loss = plot_learning_curve(history)
    >>> ax_acc.set_title('Updated Accuracy Title')
    >>> plt.show()
    """

    if color_scheme is None:
        color_scheme = {
            'train_acc': 'blue', 'val_acc': 'green',
            'train_loss': 'orange', 'val_loss': 'red'
        }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy', 
             color=color_scheme.get('train_acc', 'blue'))
    if 'val_accuracy' in history.history:
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy',
                 color=color_scheme.get('val_acc', 'green'))
    ax1.set_title(title + ' - Accuracy')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel_acc)
    ax1.legend()

    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss',
             color=color_scheme.get('train_loss', 'orange'))
    if 'val_loss' in history.history:
        ax2.plot(history.history['val_loss'], label='Validation Loss',
                 color=color_scheme.get('val_loss', 'red'))
    ax2.set_title(title + ' - Loss')
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel_loss)
    ax2.legend()
    return ax1, ax2

def build_mlp_model(
    input_dim: int, 
    output_classes: int = 6,
    hidden_units: List[int] = [14, 10], 
    dropout_rate: float = 0.5, 
    learning_rate: float = 0.001,
    activation_functions: Optional[List[str]] = None,
    optimizer: Union[str, Optimizer] = 'adam',
    loss_function: Union[str, Loss] = 'categorical_crossentropy',
    regularizer: Optional[regularizers.Regularizer] = None,
    include_batch_norm: bool = False,
    custom_metrics: List[Union[str, Metric]] = ['accuracy'],
    initializer: str = 'glorot_uniform'
    ) -> Sequential:
    """
    Build and compile a Multilayer Perceptron (MLP) model.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    output_classes : int, default 6
        Number of classes for the output layer.
    hidden_units : List[int], default [14, 10]
        List specifying the number of neurons in each hidden layer.
    dropout_rate : float, default 0.5
        Dropout rate for regularization to prevent overfitting.
    learning_rate : float, default 0.001
        Learning rate for the optimizer.
    activation_functions : Optional[List[str]], default None
        List of activation functions for each hidden layer. If None,
        'relu' is used for all layers.
    optimizer : Union[str, Optimizer], default 'adam'
        Instance of Optimizer or string specifying the optimizer to use.
    loss_function : Union[str, Loss], default 'categorical_crossentropy'
        Loss function for the model.
    regularizer : Optional[regularizers.Regularizer], default None
        Regularizer function applied to the kernel weights matrix.
    include_batch_norm : bool, default False
        Whether to include batch normalization layers after each hidden layer.
    custom_metrics : List[Union[str, Metric]], default ['accuracy']
        Metrics to be evaluated by the model during training and testing.
    initializer : str, default 'glorot_uniform'
        Initializer for the kernel weights matrix.

    Returns
    -------
    model : Sequential
        A compiled Keras Sequential model.
    """

    if activation_functions is None:
        activation_functions = ['relu'] * len(hidden_units)

    # Initialize the Sequential model
    model = Sequential()
    
    # Add the input layer and first hidden layer
    model.add(Dense(hidden_units[0], input_shape=(input_dim,), 
                    activation=activation_functions[0], kernel_initializer=initializer,
                    kernel_regularizer=regularizer))
    if include_batch_norm:
        model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # Add additional hidden layers
    for units, activation in zip(hidden_units[1:], activation_functions[1:]):
        model.add(Dense(units, activation=activation, kernel_initializer=initializer,
                        kernel_regularizer=regularizer))
        if include_batch_norm:
            model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    # Add the output layer with softmax activation for classification
    model.add(Dense(output_classes, activation='softmax'))

    # Configure the optimizer
    if isinstance(optimizer, str):
        if optimizer.lower() == 'adam':
            opt = Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            opt = SGD(learning_rate=learning_rate)
        elif optimizer.lower() == 'rmsprop':
            opt = RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
    else:
        opt = optimizer

    # Compile the model
    model.compile(optimizer=opt, loss=loss_function, metrics=custom_metrics)

    return model

def base_tuning(
    model: Model,
    train_data: Tuple[ArrayLike, ArrayLike],
    val_data: Tuple[ArrayLike, ArrayLike],
    test_data: Tuple[ArrayLike, ArrayLike],
    learning_rates: List[float],
    batch_sizes: List[int],
    epochs: int,
    optimizer: Union[Optimizer, Callable[[float], Optimizer]],
    loss: Union[str, Callable],
    metrics: List[str],
    callbacks: List[Callback] = None
) -> Tuple[Model, float, float]:
    """
    Fine-tunes a neural network model based on different hyperparameters,
    offering extensive flexibility through customizable options.

    Parameters
    ----------
    model : Model
        The neural network model to be fine-tuned.
    train_data : Tuple[ndarray, ndarray]
        Training dataset, including features and labels.
    val_data : Tuple[ndarray, ndarray]
        Validation dataset, including features and labels.
    test_data : Tuple[ndarray, ndarray]
        Test dataset, including features and labels.
    learning_rates : List[float]
        List of learning rates to try.
    batch_sizes : List[int]
        List of batch sizes to try.
    epochs : int
        Number of epochs for training.
    optimizer : Optimizer or Callable
        Optimizer or a function returning an optimizer instance to be used 
        for training.
    loss : str or Callable
        Loss function to be used for training.
    metrics : List[str]
        List of metrics to be evaluated by the model during training and testing.
    callbacks : List[Callback], optional
        List of callbacks to apply during training.

    Returns
    -------
    Tuple[Model, float, float]
        best_model: The best model after fine-tuning.
        best_accuracy: The best accuracy achieved on the validation set.
        test_accuracy: The accuracy on the test set.

    Example
    -------
    >>> from tensorflow.keras.models import Sequential
    >>> from tensorflow.keras.layers import Dense
    >>> from tensorflow.keras.optimizers import Adam
    >>> from tensorflow.keras.callbacks import EarlyStopping
    >>> import numpy as np
    >>> from gofast.models.deep_search import base_tuning

    # Create a simple model
    >>> model = Sequential([
    ...     Dense(64, activation='relu', input_shape=(100,)),
    ...     Dense(3, activation='softmax')
    ... ])

    # Dummy data
    >>> X_train, y_train = np.random.rand(1000, 100), np.random.randint(0, 3, (1000,))
    >>> X_val, y_val = np.random.rand(200, 100), np.random.randint(0, 3, (200,))
    >>> X_test, y_test = np.random.rand(200, 100), np.random.randint(0, 3, (200,))

    # Hyperparameters
    >>> learning_rates = [0.001, 0.0001]
    >>> batch_sizes = [32, 64]
    >>> epochs = 10

    # Fine-tuning the model with additional options
    >>> best_model, best_accuracy, test_accuracy = base_tuning(
    ...     model=model,
    ...     train_data=(X_train, y_train),
    ...     val_data=(X_val, y_val),
    ...     test_data=(X_test, y_test),
    ...     learning_rates=learning_rates,
    ...     batch_sizes=batch_sizes,
    ...     epochs=epochs,
    ...     optimizer=Adam,
    ...     loss='sparse_categorical_crossentropy',
    ...     metrics=['accuracy'],
    ...     callbacks=[EarlyStopping(monitor='val_loss', patience=3)]
    ... )

    # Output: The best model and its validation and test accuracies.
    """
    best_accuracy = 0
    best_model = None
    
    for lr in learning_rates:
        for batch_size in batch_sizes:
            # Configure the model with the current hyperparameters
            model.compile(optimizer=optimizer(learning_rate=lr),
                          loss=loss,
                          metrics=metrics)
    
            # Train the model
            model.fit(train_data, batch_size=batch_size,
                      epochs=epochs, validation_data=val_data,
                      callbacks=callbacks)
    
            # Evaluate the model on validation data
            accuracy = model.evaluate(val_data)[1]  # Assuming accuracy is the second metric
    
            # Update the best model if current model is better
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
    
    # Evaluate the best model on the test set
    test_accuracy = best_model.evaluate(test_data)[1]
    
    return best_model, best_accuracy, test_accuracy

def robust_tuning(
    model_fn: Callable[..., tf.keras.Model],
    dataset: Tuple[ArrayLike, ArrayLike],
    param_grid: Dict[str, List[Union[float, int]]],
    n_splits: int = 5,
    epochs: int = 50,
    patience: int = 5,
    log_dir: str = "logs/fit",
    loss: Union[str, Callable] = 'sparse_categorical_crossentropy',
    metrics: List[str] = 'accuracy',
    custom_callbacks: List[tf.keras.callbacks.Callback] = None
) -> Tuple[tf.keras.Model, Dict[str, Union[float, int]], float]:
    """
    Fine-tunes a neural network model using cross-validation and grid 
    search for hyperparameter optimization, with support for custom loss 
    functions, metrics, and callbacks.

    Parameters
    ----------
    model_fn : Callable[..., tf.keras.Model]
        A function that returns a compiled neural network model, which
        can accept hyperparameters as arguments.
    dataset : Tuple[np.ndarray, np.ndarray]
        Tuple (X, y) of training data and labels.
    param_grid : Dict[str, List[Union[float, int]]]
        Dictionary with hyperparameters to search (e.g., 
        {'learning_rate': [0.01, 0.001], 'batch_size': [32, 64]}).
    n_splits : int, optional
        Number of splits for cross-validation.
    epochs : int, optional
        Number of epochs for training.
    patience : int, optional
        Number of epochs to wait for improvement before early stopping.
    log_dir : str, optional
        Directory for TensorBoard logs.
    loss : str or Callable, optional
        Loss function to be used for training.
    metrics : List[str], optional
        List of metrics to be evaluated by the model during training and testing.
    custom_callbacks : List[tf.keras.callbacks.Callback], optional
        List of additional callbacks to apply during training, beyond early
        stopping and TensorBoard.

    Returns
    -------
    Tuple[tf.keras.Model, Dict[str, Union[float, int]], float]
        best_model: The best model after fine-tuning.
        best_params: Best hyperparameter set.
        best_score: The best score achieved during cross-validation.

    Example
    -------
    >>> from tensorflow.keras.models import Sequential
    >>> from tensorflow.keras.layers import Dense
    >>> import numpy as np
    >>> from gofast.models.deep_search import robust_tuning

    # Define a model function
    >>> def model_fn(learning_rate):
    ...     model = Sequential([
    ...         Dense(64, activation='relu', input_shape=(100,)),
    ...         Dense(3, activation='softmax')
    ...     ])
    ...     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    ...                   loss='sparse_categorical_crossentropy',
    ...                   metrics=['accuracy'])
    ...     return model

    # Dummy data
    >>> X, y = np.random.rand(1000, 100), np.random.randint(0, 3, size=(1000,))

    # Hyperparameters
    >>> param_grid = {
    ...     'learning_rate': [0.001, 0.0001]
    ... }

    # Execute the tuning
    >>> best_model, best_params, best_score = robust_tuning(
    ...     model_fn=model_fn,
    ...     dataset=(X, y),
    ...     param_grid=param_grid,
    ...     n_splits=5,
    ...     epochs=10,
    ...     patience=3
    ... )

    # Output: The best model, its parameters, and score.
    """
    X, y = dataset
    kf = KFold(n_splits=n_splits)
    best_score = 0
    best_params = None
    best_model = None
    
    metrics= is_iterable(metrics, exclude_string=True, transform =True )

    # Prepare grid search
    param_combinations = [dict(zip(param_grid, v))
                          for v in itertools.product(*param_grid.values())]

    for params in param_combinations:
        scores = []
        for train_index, val_index in kf.split(X):
            # Split data
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            # Create a new instance of the model with the current set of parameters
            model = model_fn(**params)
            model.compile(optimizer=model.optimizer, loss=loss, metrics=metrics)

            # Prepare callbacks
            early_stop = EarlyStopping(monitor='val_loss', patience=patience)
            log_path = os.path.join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            tensorboard_callback = TensorBoard(log_dir=log_path, histogram_freq=1)
            callbacks = [early_stop, tensorboard_callback] + (
                custom_callbacks if custom_callbacks else [])

            # Train the model
            model.fit(X_train, y_train, validation_data=(X_val, y_val),
                      epochs=epochs, batch_size=params.get('batch_size', 32),
                      callbacks=callbacks)

            # Evaluate the model 
            # Assuming the second metric is the score
            score = model.evaluate(X_val, y_val, verbose=0)[1]  
            scores.append(score)

        # Compute the average score over all folds
        avg_score = np.mean(scores)

        # Update the best model if the current set of parameters is better
        if avg_score > best_score:
            best_score = avg_score
            best_params = params
            best_model = tf.keras.models.clone_model(model)
            best_model.set_weights(model.get_weights())

    return best_model, best_params, best_score

def fair_neural_tuning(
    model_fn: Callable[..., Model],
    train_data: Tuple[ArrayLike, ArrayLike],
    val_data: Tuple[ArrayLike, ArrayLike],
    test_data: Tuple[ArrayLike, ArrayLike],
    param_grid: Dict[str, List[Union[float, int, str]]],
    epochs: int = 50,
    loss: Union[str, Callable] = 'sparse_categorical_crossentropy',
    metrics: List[str] = 'accuracy',
    callbacks: List[Callback] = None,
    verbose: int = 0
) -> Tuple[Model, Dict[str, Union[float, int, str]], float]:
    """
    Fine-tunes a neural network model based on different hyperparameters using 
    a progress bar for each configuration, with support for custom loss functions, 
    metrics, and callbacks.

    Parameters
    ----------
    model_fn : Callable[..., Model]
        A function that returns a compiled neural network model, which
        can accept hyperparameters as arguments.
    train_data : Tuple[ndarray, ndarray]
        Training dataset, including features and labels.
    val_data : Tuple[ndarray, ndarray]
        Validation dataset, including features and labels.
    test_data : Tuple[ndarray, ndarray]
        Test dataset, including features and labels.
    param_grid : Dict[str, List[Union[float, int, str]]]
        Dictionary with hyperparameters to search (e.g., 
        {'learning_rate': [0.01, 0.001], 'batch_size': [32, 64]}).
    epochs : int, optional
        Number of epochs for training.
    loss : str or Callable, optional
        Loss function to be used for training.
    metrics : List[str], optional
        List of metrics to be evaluated by the model during training and testing.
    callbacks : List[Callback], optional
        List of additional callbacks to apply during training.
    verbose : int, optional
        Verbosity mode, 0 or 1.

    Returns
    -------
    Tuple[Model, Dict[str, Union[float, int, str]], float]
        best_model: The best model after fine-tuning.
        best_params: Best hyperparameter set.
        best_score: The best accuracy achieved on the validation set.

    Example
    -------
    >>> from tensorflow.keras.models import Sequential
    >>> from tensorflow.keras.layers import Dense
    >>> from tensorflow.keras.optimizers import Adam
    >>> import numpy as np
    >>> from gofast.models.deep_search import fair_neural_tuning

    # Define a model function that accepts hyperparameters
    >>> def model_fn(learning_rate):
    ...     model = Sequential([
    ...         Dense(64, activation='relu', input_shape=(10,)),
    ...         Dense(1, activation='sigmoid')
    ...     ])
    ...     model.compile(optimizer=Adam(learning_rate=learning_rate),
    ...                  loss='binary_crossentropy',
    ...                  metrics=['accuracy'])
    ...    return model

    # Dummy data
    >>> train_data = (np.random.rand(100, 10), np.random.rand(100))
    >>> val_data = (np.random.rand(20, 10), np.random.rand(20))
    >>> test_data = (np.random.rand(20, 10), np.random.rand(20))

    # Hyperparameters
    >>> param_grid = {
    ...    'learning_rate': [0.01, 0.001]
    ... }

    # Execute the tuning
    >>> best_model, best_params, best_score = fair_neural_tuning(
    ...    model_fn=model_fn,
    ...    train_data=train_data,
    ...    val_data=val_data,
    ...    test_data=test_data,
    ...    param_grid=param_grid,
    ...    epochs=10,
    ...    loss='binary_crossentropy',
    ...    metrics=['accuracy']
    ... )

    # Output: The best model, its parameters, and score.
    """
    best_score = 0
    best_params = None
    best_model = None
    metrics= is_iterable(metrics, exclude_string=True, transform =True )
    param_combinations = [dict(zip(param_grid, v)) for v in itertools.product(
        *param_grid.values())]

    for params in tqdm(param_combinations, desc='Hyperparameter combinations', 
                       ascii=True, ncols=100):
        model = model_fn(**params)
        model.compile(optimizer=model.optimizer, loss=loss, metrics=metrics)
        
        current_callbacks = callbacks if callbacks else []
        
        model.fit(
            train_data[0], train_data[1], 
            batch_size=params.get('batch_size', 32),
            epochs=epochs, 
            validation_data=val_data, 
            verbose=verbose, 
            callbacks=current_callbacks
        )
        
        accuracy = model.evaluate(val_data[0], val_data[1], verbose=0)[1]
        
        if accuracy > best_score:
            best_score = accuracy
            best_params = params
            best_model = tf.keras.models.clone_model(model)
            best_model.set_weights(model.get_weights())

    best_model.evaluate(test_data[0], test_data[1], verbose=0)[1]
    
    return best_model, best_params, best_score

def deep_cv_tuning(
    model_fn: Callable[..., tf.keras.Model],
    dataset: Tuple[ArrayLike, ArrayLike],
    param_grid: Dict[str, List[Union[float, int, str]]],
    n_splits: int = 5,
    epochs: int = 50,
    patience: int = 5,
    log_dir: str = "logs/fit",
    loss: Union[str, Callable] = 'binary_crossentropy',
    metrics: List[str] = 'accuracy',
    verbose: int = 0,
    callbacks: List[tf.keras.callbacks.Callback] = None
) -> Tuple[tf.keras.Model, Dict[str, Union[float, int, str]], float]:
    """
    Performs hyperparameter optimization on a neural network model using 
    cross-validation and grid search, including progress tracking for each 
    parameter combination.

    Parameters
    ----------
    model_fn : Callable[..., tf.keras.Model]
        Function that returns a compiled neural network model, accepting 
        hyperparameters as arguments.
    dataset : Tuple[np.ndarray, np.ndarray]
        Training data and labels (X, y).
    param_grid : Dict[str, List[Union[float, int, str]]]
        Hyperparameters to search (e.g., {'learning_rate': [0.01, 0.001]}).
    n_splits : int, optional
        Number of splits for cross-validation.
    epochs : int, optional
        Training epochs for each model.
    patience : int, optional
        Early stopping patience.
    log_dir : str, optional
        TensorBoard logs directory.
    loss : str or Callable, optional
        Loss function for model compilation.
    metrics : List[str], optional
        List of metrics for model evaluation.
    verbose : int, optional
        Verbosity mode.
    callbacks : List[tf.keras.callbacks.Callback], optional
        Additional callbacks for model training.

    Returns
    -------
    Tuple[tf.keras.Model, Dict[str, Union[float, int, str]], float]
        best_model: The model with the highest cross-validation score.
        best_params: Parameters of the best model.
        best_score: Best cross-validation score.

    Example
    -------
    >>> import tensorflow as tf
    >>> import numpy as np
    >>> from gofast.models.deep_search import deep_cv_tuning
    >>> def model_fn(learning_rate):
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(10, activation='relu', input_shape=(10,)),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                          loss='binary_crossentropy', metrics=['accuracy'])
            return model
    >>> dataset = (np.random.rand(100, 10), np.random.rand(100))
    >>> param_grid = {'learning_rate': [0.01, 0.001], 'batch_size': [32, 64]}
    >>> best_model, best_params, best_score = deep_cv_tuning(
            model_fn, dataset, param_grid, n_splits=5, epochs=10, patience=3)
    """
    X, y = dataset
    kf = KFold(n_splits=n_splits)
    best_score = -np.inf
    best_params = None
    best_model = None
    metrics= is_iterable(metrics, exclude_string=True, transform =True )

    param_combinations = [dict(zip(param_grid, v)) for v in itertools.product(
        *param_grid.values())]

    for params in tqdm(param_combinations, desc='Hyperparameter Grid', ascii=True, 
                       ncols=100):
        scores = []
        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            model = model_fn(**params)
            model.compile(optimizer=model.optimizer, loss=loss, metrics=metrics)

            early_stop = EarlyStopping(monitor='val_loss', patience=patience, verbose=verbose)
            log_path = os.path.join(log_dir, f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}", str(params))
            tensorboard_callback = TensorBoard(log_dir=log_path, histogram_freq=1)

            model.fit(X_train, y_train, validation_data=(X_val, y_val),
                      epochs=epochs, batch_size=params.get('batch_size', 32), verbose=verbose,
                      callbacks=[early_stop, tensorboard_callback] + (callbacks if callbacks else []))
            # Assuming accuracy is the target metric
            score = model.evaluate(X_val, y_val, verbose=verbose)[1] 
            scores.append(score)

        avg_score = np.mean(scores)
        if avg_score > best_score:
            best_score = avg_score
            best_params = params
            best_model = tf.keras.models.clone_model(model)
            best_model.set_weights(model.get_weights())

    return best_model, best_params, best_score

def train_and_evaluate2(
    model_config: Dict[str, Any], 
    resource: int, 
    dataset: Optional[tuple] = None, 
    batch_size: int = 32, 
    optimizer: str = 'Adam', 
    callbacks: List[tf.keras.callbacks.Callback] = None
    ) -> float:
    """
    Trains and evaluates a TensorFlow/Keras model given a configuration, 
    resources, and additional training parameters.

    Parameters
    ----------
    model_config : Dict[str, Any]
        Configuration of the model including hyperparameters such as the number 
        of units in each layer.
    resource : int
        Allocated resource for the model, typically the number of epochs.
    dataset : tuple, optional
        The dataset to use for training and validation, structured as 
        ((x_train, y_train), (x_val, y_val)).
        If None, the MNIST dataset is loaded by default.
    batch_size : int, optional
        The size of the batches to use during training. Default is 32.
    optimizer : str, optional
        The name of the optimizer to use. Default is 'Adam'. Other examples 
        include 'sgd', 'rmsprop'.
    callbacks : List[tf.keras.callbacks.Callback], optional
        A list of callbacks to use during training. Default is None.

    Returns
    -------
    float
        The best validation accuracy achieved during training.

    Examples
    --------
    >>> model_config = {'units': 64, 'learning_rate': 0.001}
    >>> resource = 10  # Number of epochs
    >>> best_val_accuracy = train_and_evaluate(model_config, resource)
    >>> print(f"Best Validation Accuracy: {best_val_accuracy}")

    Note: The function defaults to using the MNIST dataset if no dataset is provided.
    """
    # Load and prepare dataset
    if dataset is None:
        (x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.reshape(-1, 784).astype('float32') / 255
        x_val = x_val.reshape(-1, 784).astype('float32') / 255
    else:
        (x_train, y_train), (x_val, y_val) = dataset

    # Define model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(model_config['units'], activation='relu', 
                              input_shape=(784,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Select optimizer
    if optimizer.lower() == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=model_config['learning_rate'])
    elif optimizer.lower() == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=model_config['learning_rate'])
    elif optimizer.lower() == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=model_config['learning_rate'])
    else:
        raise ValueError(f"Optimizer '{optimizer}' is not supported.")
    
    # Compile model
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train model
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=resource, 
                        validation_data=(x_val, y_val), verbose=0, callbacks=callbacks)
    
    # Extract and return the best validation accuracy
    best_val_accuracy = max(history.history['val_accuracy'])
    return best_val_accuracy

def train_and_evaluate(model_config: Dict[str, Any], resource: int) -> float:
    """
    Trains and evaluates a Keras model given a configuration and resources.

    Parameters:
    model_config (Dict[str, Any]) 
        Configuration of the model including hyperparameters.
    resource (int): 
        Allocated resource for the model, e.g., number of epochs.

    Returns:
    float: The validation accuracy of the model.
    """
    # Define a simple model based on the configuration
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(model_config['units'], 
                              activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=model_config['learning_rate']),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Load and prepare the MNIST dataset
    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype('float32') / 255
    x_val = x_val.reshape(-1, 784).astype('float32') / 255
    
    # Train the model
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), 
                        epochs=resource, verbose=0)
    
    # Extract and return the best validation accuracy
    best_val_accuracy = max(history.history['val_accuracy'])
    
    return best_val_accuracy

from typing import Callable, Dict, Any, List, Tuple


class Hyperband:
    """
    Hyperband is an advanced hyperparameter optimization algorithm that 
    efficiently identifies the best hyperparameters for a given model using 
    a bandit-based approach. It dynamically allocates computational resources 
    to different model configurations by balancing exploration of the 
    hyperparameter space with exploitation of promising configurations through 
    successive halving.

    The algorithm operates in rounds, where each round consists of training a 
    set of model configurations with an increasing amount of resources, and 
    only the most promising configurations proceed to the next round until 
    the best configuration is identified.

    Parameters
    ----------
    model_fn : Callable
        A callable that takes a hyperparameter configuration as input and 
        returns a compiled Keras model. The function is responsible for model 
        instantiation, compilation, and setting hyperparameters.
    max_resource : int
        The maximum amount of resource (e.g., number of epochs) that can be 
        allocated to train a single model configuration.
    eta : float, optional, default=3
        The factor by which the allocated resource is reduced in each round of
        successive halving. A higher `eta` value results in more aggressive 
        reduction.

    Attributes
    ----------
    best_model_ : tf.keras.Model
        The model instance that achieved the highest performance score after the
        completion of the Hyperband optimization process.
    best_params_ : Dict[str, Any]
        The set of hyperparameters associated with the `best_model_`.
    best_score_ : float
        The performance score of the `best_model_`, typically validation accuracy.
    model_results_ : List[Dict[str, Any]]
        A list containing the hyperparameters and performance scores for all evaluated
        model configurations.

    Examples
    --------
    >>> from gofast.models.deep_search import Hyperband
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.model_selection import train_test_split
    >>> from tensorflow.keras.utils import to_categorical

    >>> def model_fn(params):
    ...     model = Sequential([
    ...         Dense(params['units'], activation='relu', input_shape=(64,)),
    ...         Dense(10, activation='softmax')])
    ...     model.compile(optimizer=Adam(learning_rate=params['learning_rate']),
    ...                   loss='categorical_crossentropy',
    ...                   metrics=['accuracy'])
    ...     return model

    >>> digits = load_digits()
    >>> X = digits.data / 16.0  # Normalizing
    >>> y = to_categorical(digits.target, num_classes=10)
    >>> X_train, X_val, y_train, y_val = train_test_split(
    ...    X, y, test_size=0.2,random_state=42)

    >>> hyperband = Hyperband(model_fn=model_fn, max_resource=81, eta=3)
    >>> hyperband.run(train_data=(X_train, y_train), val_data=(X_val, y_val))
    >>> print(f"Best Hyperparameters: {hyperband.best_params_},
    ... Best Score: {hyperband.best_score_}")

    Note
    ----
    The `model_fn` should handle both the instantiation and compilation of the
    Keras model, including setting any relevant hyperparameters based on the 
    input configuration. The Hyperband algorithm assumes that higher resource 
    allocations (e.g., more epochs) will generally lead to better model 
    performance, which may not always hold true for all models or datasets.

    The mathematical formulation of Hyperband involves determining the number 
    of configurations (`n`) and the amount of resource (`r`) for each round 
    based on the inputs `max_resource` and `eta`. It can be described by the 
    following equations:

    - `s_max = floor(log_{eta}(max_resource))`
    - `B = (s_max + 1) * max_resource`

    where `s_max` is the maximum number of iterations, and `B` is the total 
    budget across all brackets. The algorithm then iteratively selects a subset 
    of configurations to train with increasing resources until the best 
    configuration is found.
    
    """

    def __init__(self, model_fn: Callable, max_resource: int, eta: float = 3 ):
        self.max_resource = max_resource
        self.eta = eta
        self.model_fn = model_fn
        
    def _train_and_evaluate(
        self, model_config: Dict[str, Any], resource: int, 
        train_data: Tuple, val_data: Tuple
        ) -> float:
        """
        Trains and evaluates a model for a specified configuration and resource.
        
        Parameters
        ----------
        model_config : Dict[str, Any]
            Hyperparameter configuration for the model.
        resource : int
            Allocated resource for the model, typically the number of epochs.
        train_data : Tuple[np.ndarray, np.ndarray]
            Training data and labels.
        val_data : Tuple[np.ndarray, np.ndarray]
            Validation data and labels.

        Returns
        -------
        float
            The performance metric of the model, e.g., validation accuracy.
        """
        model = self.model_fn(model_config)
        X_train, y_train = train_data
        X_val, y_val = val_data
        history = model.fit(X_train, y_train, epochs=resource, 
                            validation_data=(X_val, y_val), verbose=self.verbose )
        val_accuracy = history.history['val_accuracy'][-1]
        return val_accuracy

    def get_hyperparameter_configuration(self, n: int) -> List[Dict[str, Any]]:
        """
        Generates a list of `n` random hyperparameter configurations.
        
        Parameters
        ----------
        n : int
            Number of configurations to generate.
        
        Returns
        -------
        List[Dict[str, Any]]
            A list of hyperparameter configurations.
        """
        configurations = [{'learning_rate': np.random.uniform(1e-4, 1e-2),
                           'units': np.random.randint(50, 500)} 
                          for _ in range(n)]
        return configurations

    def run(self, train_data: Tuple[ArrayLike, ArrayLike],
            val_data: Tuple[ArrayLike, ArrayLike]) -> 'Hyperband':
        """
        Executes the Hyperband optimization process on the given dataset.

        Parameters
        ----------
        train_data : Tuple[np.ndarray, np.ndarray]
            Training data and labels.
        val_data : Tuple[np.ndarray, np.ndarray]
            Validation data and labels.

        Returns
        -------
        self : Hyperband
            The Hyperband instance with updated attributes, including the best model
            and its hyperparameters.
        """
        for s in reversed(range(int(np.log(self.max_resource) / np.log(self.eta)) + 1)):
            n = int(np.ceil(self.max_resource / self.eta ** s / (s + 1)))
            resource = self.max_resource * self.eta ** (-s)
            configurations = self.get_hyperparameter_configuration(n)
            for i in range(s + 1):
                n_i = n * self.eta ** (-i)
                r_i = resource * self.eta ** i
                
                val_scores = [self._train_and_evaluate(
                    config, int(np.floor(r_i)), train_data, val_data) 
                    for config in configurations]
                
                if self.verbose:
                   print(f"Generation {i+1}/{s+1}, Configurations evaluated:"
                         f" {len(configurations)}")
                # Select top-k configurations based on validation scores
                if i < s:
                    top_k_indices = np.argsort(val_scores)[-max(int(n_i / self.eta), 1):]
                    configurations = [configurations[j] for j in top_k_indices]
                else:
                    best_index = np.argmax(val_scores)
                    self.best_score_ = val_scores[best_index]
                    self.best_params_ = configurations[best_index]
                    self.best_model_ = self.model_fn(self.best_params_)
                    self.model_results_.append({'config': self.best_params_, 
                                                'score': self.best_score_})
        return self

class PBTTrainer:
    """
    Population Based Training (PBT) Trainer.

    PBT is a hyperparameter optimization technique that dynamically adjusts 
    hyperparameters during training. It trains a population of models in parallel
    and periodically applies "exploit and explore" phases. Underperforming models
    are replaced with perturbed versions of better-performing models, allowing
    hyperparameters to evolve over time.

    Parameters
    ----------
    model_fn : callable
        Function that returns a new instance of the model to be trained. This 
        function should not accept any arguments and should return a 
        compiled Keras model.
    param_space : dict
        Dictionary defining the hyperparameter space for exploration. Each key 
        is a hyperparameter name, and its value is a tuple specifying the 
        minimum and maximum range (min, max) from which values are sampled.
    population_size : int, optional, default=10
        Number of models in the population. Represents the diversity of the
        hyperparameter space exploration.
    exploit_method : str, optional, default='truncation'
        Strategy used for the "exploit" phase. Currently, only 'truncation' 
        is implemented, where underperforming models are replaced by copies of 
        better-performing models.
    perturb_factor : float, optional, default=0.2
        Factor used to perturb the hyperparameters of a model during the 
        "explore" phase, introducing variation and enabling the exploration 
        of the hyperparameter space.
    num_generations : int, optional, default=5
        Number of generations to evolve the population. Each generation 
        consists of training, evaluation, and the "exploit and explore" phase.
    epochs_per_step : int, optional, default=5
        Number of epochs for which to train each model before applying the 
        "exploit and explore" phase. Controls the frequency of hyperparameter 
        adjustments.
    verbose : int, optional, default=0
        Verbosity level. Higher values lead to more messages being printed 
        to monitor the training process and PBT progress.

    Attributes
    ----------
    best_params_ : dict
        Hyperparameters of the best-performing model after the last generation.
    best_score_ : float
        Highest performance score achieved by any model in the population.
    best_model_ : tf.keras.Model
        The model instance that achieved the best_score_.
    model_results_ : list
        List of dictionaries, each containing the hyperparameters and 
        performance score of a model at each generation.
        
    Examples
    --------
    >>> import tensorflow as tf 
    >>> from gofast.models.deep_search import PBTTrainer
    >>> def model_fn():
    ...     model = tf.keras.Sequential([
    ...         tf.keras.layers.Dense(10, activation='relu', input_shape=(10,)),
    ...         tf.keras.layers.Dense(1, activation='sigmoid')
    ...     ])
    ...     return model
    >>> param_space = {'learning_rate': (0.001, 0.01), 'batch_size': (16, 64)}
    >>> trainer = PBTTrainer(model_fn=model_fn, param_space=param_space, 
    ...                         population_size=5,num_generations=10, 
    ...                         epochs_per_step=2, verbose=1)
    >>> trainer.run(train_data=(X_train, y_train), val_data=(X_val, y_val))
    >>> trainer.run(train_data, val_data)
    >>> trainer.best_params_
    """
    def __init__(
        self, 
        model_fn, 
        param_space, 
        population_size=10, 
        exploit_method='truncation',
        perturb_factor=0.2, 
        num_generations=5, 
        epochs_per_step=5, 
        verbose=0
        ):
        self.model_fn = model_fn
        self.param_space = param_space
        self.population_size = population_size
        self.exploit_method = exploit_method
        self.perturb_factor = perturb_factor
        self.num_generations = num_generations
        self.epochs_per_step = epochs_per_step
        self.verbose=verbose 
        
    def run(self, train_data: Tuple[ArrayLike, ArrayLike],
            val_data: Tuple[ArrayLike, ArrayLike])-> 'PBTTrainer':
        """
        Runs the PBT optimization process on the given dataset.

        Parameters
        ----------
        train_data : tuple
            A tuple (X_train, y_train) containing the training data and labels.
        val_data : tuple
            A tuple (X_val, y_val) containing the validation data and labels.

        Returns
        -------
        self : PBTTrainer
            The PBTTrainer instance, allowing access to the best model, 
            parameters, and scores.
        """
        self.population = self._init_population()
        if self.exploit_method.lower() !="truncation": 
            warnings.warn("Currently, supported only 'truncation' method.")
            self.exploit_method='truncation'
            
        for generation in range(self.num_generations):
            if self.verbose: 
                print(f"Generation {generation + 1}/{self.num_generations}")
            for model, hyperparams in self.population:
                self._train_model(model, train_data, hyperparams, self.epochs_per_step)
                performance = self._evaluate_model(model, val_data)
                self.model_results_.append({'hyperparams': hyperparams, 
                                            'performance': performance})
                if performance > self.best_score_:
                    self.best_score_ = performance
                    self.best_params_ = hyperparams
                    self.best_model_ = copy.deepcopy(model)
            self._exploit_and_explore()

        return self
    
    def _init_population(self):
        """
        Initializes the population with models and random hyperparameters.
        """
        population = []
        for _ in range(self.population_size):
            hyperparams = {k: np.random.uniform(low=v[0], high=v[1]) 
                           for k, v in self.param_space.items()}
            model = self.model_fn()
            population.append((model, hyperparams))
        return population
    
    def _train_model(self, model, train_data, hyperparams, epochs):
        """
        Trains a single model instance using TensorFlow.

        Parameters
        ----------
        model : tf.keras.Model
            The TensorFlow model to train.
        train_data : tuple
            A tuple (X_train, y_train) containing the training data and labels.
        hyperparams : dict
            Hyperparameters to use for training, including 'learning_rate'.
        epochs : int
            Number of epochs to train the model.
        """
        X_train, y_train = train_data
        optimizer = Adam(learning_rate=hyperparams['learning_rate'])
        model.compile(optimizer=optimizer, loss='binary_crossentropy',
                      metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=epochs, batch_size=int(
            hyperparams.get('batch_size', 32)), verbose=0)

    def _evaluate_model(self, model, val_data):
        """
        Evaluates a single model instance using TensorFlow.

        Parameters
        ----------
        model : tf.keras.Model
            The TensorFlow model to evaluate.
        val_data : tuple
            A tuple (X_val, y_val) containing the validation data and labels.

        Returns
        -------
        performance : float
            The performance metric of the model, typically accuracy.
        """
        X_val, y_val = val_data
        _, performance = model.evaluate(X_val, y_val, verbose=0)
        return performance
    
    def _exploit_and_explore(self):
        """
        Apply the exploit and explore strategy to evolve the population.
        """
        # Sort models based on performance
        self.population.sort(key=lambda x: x[0].performance, reverse=True)

        # Exploit: Replace bottom half with top performers
        top_performers = self.population[:len(self.population) // 2]
        for i in range(len(self.population) // 2, len(self.population)):
            if self.exploit_method == 'truncation':
                # Clone a top performer's model and hyperparameters
                model, hyperparams = copy.deepcopy(top_performers[i % len(top_performers)])
                self.population[i] = (model, hyperparams)

        # Explore: Perturb the hyperparameters
        for i in range(len(self.population) // 2, len(self.population)):
            _, hyperparams = self.population[i]
            perturbed_hyperparams = {k: v * np.random.uniform(
                1 - self.perturb_factor, 1 + self.perturb_factor) 
                for k, v in hyperparams.items()}
            self.population[i] = (self.model_fn(), perturbed_hyperparams)  
            # Reinitialize model
            
def custom_loss(
    y_true: tf.Tensor, 
    y_pred: tf.Tensor, 
    y_estimated: tf.Tensor, 
    lambda_value: float, 
    reduction: str = 'auto', 
    loss_name: str = 'custom_loss'
    ) -> Callable:
    """
    Computes a custom loss value which is a combination of mean squared error between 
    true and predicted values, and an additional term weighted by a lambda value.

    Parameters
    ----------
    y_true : tf.Tensor
        The ground truth values.
    y_pred : tf.Tensor
        The predicted values.
    y_estimated : tf.Tensor
        An estimated version of the ground truth values, used for an additional
        term in the loss.
    lambda_value : float
        The weight of the additional term in the loss calculation.
    reduction : str, optional
        Type of `tf.keras.losses.Reduction` to apply to loss. Default value is 'auto',
        which means the reduction option will be determined by the current Keras backend.
        Other possible values include 'sum_over_batch_size', 'sum', and 'none'.
    loss_name : str, optional
        Name to use for the loss.

    Returns
    -------
    Callable
        A callable that takes `y_true` and `y_pred` as inputs and returns the loss value
        as an output.

    Examples
    --------
    >>> from gofast.models.deep_search import custom_loss
    >>> import tensorflow as tf
    >>> model = tf.keras.models.Sequential([
    ...     tf.keras.layers.Dense(10, activation='relu', input_shape=(20,)),
    ...     tf.keras.layers.Dense(1)
    ... ])
    >>> model.compile(optimizer='adam', loss=lambda y_true, y_pred: custom_loss(
    ...     y_true, y_pred, y_estimated=tf.zeros_like(y_pred), lambda_value=0.5))
    >>> # Assume X_train, y_train are prepared data
    >>> # model.fit(X_train, y_train, epochs=10)

    Note
    ----
    The `custom_loss` function is designed to be used with TensorFlow and Keras models. 
    The `y_estimated` parameter allows for incorporating additional domain-specific knowledge 
    into the loss, beyond what is captured by comparing `y_true` and `y_pred` alone.
    """
    def loss(y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
        additional_term = tf.reduce_mean(tf.square(y_true - y_estimated), axis=-1)
        return mse + lambda_value * additional_term
    
    return tf.keras.losses.Loss(name=loss_name, reduction=reduction)(loss)

def train_epoch(
    model: tf.keras.Model, 
    optimizer: tf.keras.optimizers.Optimizer, 
    x_train: np.ndarray, 
    y_train_actual: np.ndarray, 
    y_train_estimated: Optional[np.ndarray] = None, 
    lambda_value: float = 0.0, 
    batch_size: int = 32, 
    use_custom_loss: bool = False
) -> Tuple[float, float]:
    """
    Trains the model for one epoch over the provided training data and calculates
    the training and validation loss.

    Parameters
    ----------
    model : tf.keras.Model
        The neural network model to train.
    optimizer : tf.keras.optimizers.Optimizer
        The optimizer to use for training.
    x_train : np.ndarray
        Input features for training.
    y_train_actual : np.ndarray
        Actual target values for training.
    y_train_estimated : Optional[np.ndarray], default=None
        Estimated target values for training, used with custom loss.
    lambda_value : float, default=0.0
        The weighting factor for the custom loss component, if used.
    batch_size : int, default=32
        The size of the batches to use for training.
    use_custom_loss : bool, default=False
        Whether to use a custom loss function or default to MSE.

    Returns
    -------
    Tuple[float, float]
        The mean training loss and validation loss for the epoch.

    Examples
    --------
    >>> from gofast.models.deep_search import train_epoch
    >>> model = tf.keras.models.Sequential([
    ...     tf.keras.layers.Dense(10, activation='relu', input_shape=(20,)),
    ...     tf.keras.layers.Dense(1)
    ... ])
    >>> optimizer = tf.keras.optimizers.Adam()
    >>> # Assume x_train, y_train_actual, and y_train_estimated are prepared
    >>> train_loss, val_loss = train_epoch(
    ...     model, optimizer, x_train, y_train_actual,
    ...     y_train_estimated=None, lambda_value=0.5, batch_size=64,
    ...     use_custom_loss=True)
    >>> print(f"Train Loss: {train_loss}, Validation Loss: {val_loss}")
    """
    epoch_train_loss = []
    for x_batch, y_actual_batch, y_estimated_batch in data_generator(
        x_train, y_train_actual, y_train_estimated, batch_size):
        
        with tf.GradientTape() as tape:
            y_pred_batch = model(x_batch, training=True)
            if use_custom_loss and y_train_estimated is not None:
                loss = custom_loss(
                    y_actual_batch, y_pred_batch, 
                    y_estimated_batch, lambda_value)
            else:
                loss = tf.keras.losses.mean_squared_error(
                    y_actual_batch, y_pred_batch)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        epoch_train_loss.append(tf.reduce_mean(loss).numpy())

    # Assume calculate_validation_loss is a predefined function
    val_loss = calculate_validation_loss(
        model, x_train, y_train_actual, 
        y_train_estimated, lambda_value, use_custom_loss)
    
    return np.mean(epoch_train_loss), val_loss

def calculate_validation_loss(
    model: tf.keras.Model, 
    x_val: np.ndarray, 
    y_val_actual: np.ndarray, 
    y_val_estimated: Optional[np.ndarray] = None, 
    lambda_value: float = 0.0, 
    use_custom_loss: bool = False
) -> float:
    """
    Calculates the loss on the validation dataset using either a custom loss function
    or mean squared error.

    Parameters
    ----------
    model : tf.keras.Model
        The trained model for which to calculate the validation loss.
    x_val : np.ndarray
        The input features of the validation dataset.
    y_val_actual : np.ndarray
        The actual target values of the validation dataset.
    y_val_estimated : Optional[np.ndarray], default=None
        The estimated target values for the validation dataset, used with custom loss.
    lambda_value : float, default=0.0
        The weighting factor for the custom loss component, if used.
    use_custom_loss : bool, default=False
        Indicates whether to use the custom loss function or default to MSE.

    Returns
    -------
    float
        The mean validation loss calculated over all validation samples.

    Examples
    --------
    >>> from gofast.models.deep_search import calculate_validation_loss
    >>> model = tf.keras.models.Sequential([
    ...     tf.keras.layers.Dense(10, activation='relu', input_shape=(20,)),
    ...     tf.keras.layers.Dense(1)
    ... ])
    >>> # Assume x_val, y_val_actual, and y_val_estimated are prepared
    >>> val_loss = calculate_validation_loss(
    ...     model, x_val, y_val_actual, 
    ...     y_val_estimated=None, lambda_value=0.5, 
    ...     use_custom_loss=True)
    >>> print(f"Validation Loss: {val_loss}")
    """
    val_preds = model.predict(x_val)
    if use_custom_loss and y_val_estimated is not None:
        loss = custom_loss(y_val_actual, val_preds, y_val_estimated, lambda_value)
    else:
        loss = tf.keras.losses.mean_squared_error(y_val_actual, val_preds)
    return np.mean(loss.numpy())

def data_generator(
    x_data: np.ndarray, 
    y_data_actual: np.ndarray, 
    y_data_estimated: Optional[np.ndarray], 
    batch_size: int
) -> Generator[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]], None, None]:
    """
    Yields batches of data for training or validation.

    Parameters
    ----------
    x_data : np.ndarray
        The input features.
    y_data_actual : np.ndarray
        The actual target values.
    y_data_estimated : Optional[np.ndarray]
        The estimated target values, if applicable.
    batch_size : int
        The number of samples per batch.

    Yields
    ------
    Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]
        A tuple containing a batch of input features, actual target values, 
        and optionally estimated target values.
    """
    num_samples = x_data.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]

        x_batch = x_data[batch_indices]
        y_actual_batch = y_data_actual[batch_indices]
        y_estimated_batch = y_data_estimated[batch_indices] if y_data_estimated is not None else None

        yield (x_batch, y_actual_batch, y_estimated_batch)
        

def evaluate_model(
    model_path: str, 
    x_test: ArrayLike, 
    y_test: ArrayLike, 
    batch_size: int = 32, 
    denorm_range: Tuple[float, float] = (0.0, 1.0), 
    save_metrics: bool = False, 
    metrics: Optional[List[str]] = None
) -> dict:
    """
    Evaluates a saved model on a test dataset, denormalizes predictions and
    actual values using the provided denormalization range, calculates 
    specified metrics, and optionally saves the metrics to a JSON file.

    Parameters
    ----------
    model_path : str
        Path to the saved model.
    x_test : np.ndarray
        Test dataset features.
    y_test : np.ndarray
        Actual target values for the test dataset.
    batch_size : int, default=32
        Batch size to use for prediction.
    denorm_range : Tuple[float, float], default=(0.0, 1.0)
        A tuple containing the minimum and maximum values used for 
        denormalizing the dataset.
    save_metrics : bool, default=False
        Whether to save the evaluation metrics to a JSON file.
    metrics : Optional[List[str]], default=None
        List of evaluation metrics to calculate. Supported metrics are 'mse',
        'mae', and 'r2_score'. If None, all supported metrics are calculated.

    Returns
    -------
    dict
        A dictionary containing the calculated metrics.

    Examples
    --------
    >>> from gofast.models.deep_search import evaluate_model
    >>> model_path = 'path/to/your/model.h5'
    >>> x_test, y_test = load_your_test_data()  # Assume this loads your test data
    >>> test_metrics = evaluate_model(model_path, x_test, y_test, 
    ...                               denorm_range=(0, 100), 
    ...                               save_metrics=True, metrics=['mse', 'mae'])
    >>> print(test_metrics)

    Note
    ----
    The function assumes that `y_test` and model predictions need to be denormalized
    before calculating metrics. The `denorm_range` parameter should match the 
    normalization applied to your dataset.
    """
    min_value, max_value = denorm_range
    model = load_model(model_path)
    predictions = model.predict(x_test, batch_size=batch_size)
    predictions_denormalized = denormalize(predictions, min_value, max_value)
    y_test_denormalized = denormalize(y_test, min_value, max_value)
    
    test_metrics = {}
    if metrics is None or 'mse' in metrics:
        test_metrics['mse'] = mean_squared_error(y_test_denormalized, predictions_denormalized)
    if metrics is None or 'mae' in metrics:
        test_metrics['mae'] = mean_absolute_error(y_test_denormalized, predictions_denormalized)
    if metrics is None or 'r2_score' in metrics:
        test_metrics['r2_score'] = r2_score(y_test_denormalized, predictions_denormalized)
    
    test_metrics['model_size_bytes'] = os.path.getsize(model_path)
    
    if save_metrics:
        with open(model_path + '_test_metrics.json', 'w') as json_file:
            json.dump(test_metrics, json_file)
    
    return test_metrics

def train_model(
    model: 'tf.keras.Model', 
    x_train: ArrayLike, 
    y_train_actual: ArrayLike, 
    y_train_estimated: Optional[ArrayLike] = None, 
    lambda_value: float = 0.1, 
    num_epochs: int = 100, 
    batch_size: int = 32, 
    initial_lr: float = 0.001, 
    checkpoint_dir: str = './checkpoints', 
    patience: int = 10, 
    use_custom_loss: bool = True
) -> Tuple[List[float], List[float], str]:
    """
    Trains a TensorFlow/Keras model using either a custom loss function or mean squared
    error, with early stopping based on validation loss improvement.

    Parameters
    ----------
    model : tf.keras.Model
        The model to be trained.
    x_train : np.ndarray
        Input features for training.
    y_train_actual : np.ndarray
        Actual target values for training.
    y_train_estimated : Optional[np.ndarray], default=None
        Estimated target values for training, used with custom loss. If `None`,
        the custom loss will ignore this component.
    lambda_value : float, default=0.1
        Weighting factor for the custom loss component.
    num_epochs : int, default=100
        Total number of epochs to train the model.
    batch_size : int, default=32
        Number of samples per batch of computation.
    initial_lr : float, default=0.001
        Initial learning rate for the Adam optimizer.
    checkpoint_dir : str, default='./checkpoints'
        Directory where the model checkpoints will be saved.
    patience : int, default=10
        Number of epochs with no improvement after which training will be stopped.
    use_custom_loss : bool, default=True
        Indicates whether to use the custom loss function.

    Returns
    -------
    Tuple[List[float], List[float], str]
        A tuple containing the list of training losses, validation losses, and
        the checkpoint directory path.

    Examples
    --------
    >>> from gofast.models.deep_search import train_model
    >>> model = build_your_model()  # Assume a function to build your model
    >>> x_train, y_train_actual = load_your_data()  # Assume a function to load your data
    >>> train_losses, val_losses, checkpoint_dir = train_model(
    ...     model, x_train, y_train_actual, num_epochs=50, batch_size=64,
    ...     initial_lr=0.01, checkpoint_dir='/tmp/model_checkpoints',
    ...     patience=5, use_custom_loss=False)
    >>> print(f"Training complete. Checkpoints saved to {checkpoint_dir}")
    """
    optimizer = Adam(learning_rate=initial_lr)
    best_loss = np.inf
    patience_counter = 0
    train_losses, val_losses = [], []

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    for epoch in range(num_epochs):
        # Assume train_epoch is a function that trains the model for one epoch and returns
        # the training and validation loss
        train_loss, val_loss = train_epoch(
            model, optimizer, x_train, y_train_actual, y_train_estimated, 
            lambda_value, batch_size, use_custom_loss)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            model.save(os.path.join(checkpoint_dir, 'best_model'), save_format='tf')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return train_losses, val_losses, checkpoint_dir

def create_lstm_model(
    input_shape: Tuple[int, int], 
    n_units_list: List[int], 
    n_future: int, 
    activation: str = "relu", 
    dropout_rate: float = 0.2, 
    dense_activation: Optional[str] = None
) -> 'tf.keras.Model':
    """
    Creates an LSTM model dynamically based on the specified architecture 
    parameters.

    Parameters
    ----------
    input_shape : Tuple[int, int]
        The shape of the input data, excluding the batch size.
    n_units_list : List[int]
        A list containing the number of units in each LSTM layer.
    n_future : int
        The number of units in the output dense layer, often corresponding
        to the prediction horizon.
    activation : str, default='relu'
        Activation function to use in the LSTM layers.
    dropout_rate : float, default=0.2
        Dropout rate for the dropout layers following each LSTM layer.
    dense_activation : Optional[str], default=None
        Activation function for the output dense layer. If `None`, a linear
        activation is applied.

    Returns
    -------
    tf.keras.Model
        The constructed LSTM model.

    Examples
    --------
    >>> from gofast.models.deep_search import create_lstm_model
    >>> input_shape = (10, 1)  # 10 time steps, 1 feature
    >>> n_units_list = [50, 25]  # Two LSTM layers with 50 and 25 units
    >>> n_future = 1  # Predicting a single future value
    >>> model = create_lstm_model(input_shape, n_units_list, n_future,
    ...                           activation='tanh', dropout_rate=0.1,
    ...                           dense_activation='linear')
    >>> model.summary()
    
    This function constructs an LSTM model suitable for time series forecasting or
    sequence prediction tasks, allowing customization of the network architecture,
    activation functions, and regularization techniques.
    """
    model = Sequential()
    for i, n_units in enumerate(n_units_list):
        is_return_sequences = i < len(n_units_list) - 1
        model.add(LSTM(units=n_units, activation=activation, 
                       return_sequences=is_return_sequences, 
                       input_shape=input_shape if i == 0 else None))
        model.add(Dropout(dropout_rate))
    model.add(Dense(units=n_future, activation=dense_activation))
    return model

def create_cnn_model(
    input_shape: Tuple[int, int, int],
    conv_layers: List[Tuple[int, Tuple[int, int], Optional[str], float]],
    dense_units: List[int],
    dense_activation: str = "relu",
    output_units: int = 1,
    output_activation: str = "sigmoid"
) -> 'tf.keras.Model':
    """
    Creates a CNN model dynamically based on the specified architecture parameters.
    
    Parameters
    ----------
    input_shape : Tuple[int, int, int]
        The shape of the input data, including the channels.
    conv_layers : List[Tuple[int, Tuple[int, int], Optional[str], float]]
        A list of tuples for configuring the convolutional layers. Each tuple should
        contain the number of filters, kernel size, optional activation function (default
        to 'relu' if None), and dropout rate after the layer.
    dense_units : List[int]
        A list containing the number of units in each dense layer before the output.
    dense_activation : str, default='relu'
        Activation function to use in the dense layers.
    output_units : int, default=1
        The number of units in the output layer.
    output_activation : str, default='sigmoid'
        Activation function for the output layer.

    Returns
    -------
    tf.keras.Model
        The constructed CNN model.

    Examples
    --------
    >>> from gofast.models.deep_search import create_cnn_model
    >>> input_shape = (28, 28, 1)  # Example for MNIST
    >>> conv_layers = [
    ...     (32, (3, 3), 'relu', 0.2),
    ...     (64, (3, 3), 'relu', 0.2)
    ... ]
    >>> dense_units = [128]
    >>> model = create_cnn_model(input_shape, conv_layers, dense_units,
    ...                           output_units=10, output_activation='softmax')
    >>> model.summary()
    
    This function constructs a CNN model suitable for image classification or
    other tasks requiring spatial hierarchy extraction, allowing customization
    of the network architecture, activation functions, and regularization techniques.
    """
    model = Sequential()
    for i, (filters, kernel_size, activation, dropout) in enumerate(conv_layers):
        if i == 0:
            model.add(Conv2D(filters, kernel_size, activation=activation or 'relu', input_shape=input_shape))
        else:
            model.add(Conv2D(filters, kernel_size, activation=activation or 'relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        if dropout > 0:
            model.add(Dropout(dropout))
    model.add(Flatten())
    for units in dense_units:
        model.add(Dense(units, activation=dense_activation))
    model.add(Dense(output_units, activation=output_activation))
    return model

def create_autoencoder_model(
    input_dim: int,
    encoder_layers: List[Tuple[int, Optional[str], float]],
    decoder_layers: List[Tuple[int, Optional[str], float]],
    code_activation: Optional[str] = None,
    output_activation: str = "sigmoid"
) -> Model:
    """
    Creates an Autoencoder model dynamically based on the specified architecture
    parameters for the encoder and decoder.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input data.
    encoder_layers : List[Tuple[int, Optional[str], float]]
        A list of tuples defining the encoder layers. Each tuple should contain
        the number of units, an optional activation function (default to 'relu' if None),
        and dropout rate after the layer.
    decoder_layers : List[Tuple[int, Optional[str], float]]
        A list of tuples defining the decoder layers. Each tuple mirrors the structure
        of `encoder_layers`.
    code_activation : Optional[str], default=None
        Activation function for the bottleneck (code) layer. If `None`, no activation
        is applied (linear activation).
    output_activation : str, default='sigmoid'
        Activation function for the output layer of the decoder.

    Returns
    -------
    Model
        The constructed Autoencoder model as a Keras Model instance.

    Examples
    --------
    >>> from gofast.models.deep_search import create_autoencoder_model
    >>> input_dim = 784  # For MNIST images flattened to a vector
    >>> encoder_layers = [(128, 'relu', 0.2), (64, 'relu', 0.2)]
    >>> decoder_layers = [(128, 'relu', 0.2), (784, 'sigmoid', 0.0)]
    >>> autoencoder = create_autoencoder_model(input_dim, encoder_layers,
    ...                                        decoder_layers, code_activation='relu',
    ...                                        output_activation='sigmoid')
    >>> autoencoder.summary()

    Note
    ----
    This function constructs an Autoencoder model that is suitable for dimensionality
    reduction, feature learning, or unsupervised pretraining of a neural network. The
    flexibility in specifying layer configurations allows for easy experimentation
    with different architectures.
    """
    # Input layer
    input_layer = Input(shape=(input_dim,))

    # Encoder
    x = input_layer
    for units, activation, dropout in encoder_layers:
        x = Dense(units, activation=activation)(x)
        if dropout > 0:
            x = Dropout(dropout)(x)

    # Bottleneck (code layer)
    code_layer = Dense(decoder_layers[0][0], activation=code_activation)(x)

    # Decoder
    x = code_layer
    for units, activation, dropout in decoder_layers:
        x = Dense(units, activation=activation)(x)
        if dropout > 0:
            x = Dropout(dropout)(x)

    # Output layer
    output_layer = Dense(input_dim, activation=output_activation)(x)

    # Model
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    
    return autoencoder

def create_attention_model(
    input_dim: int,
    seq_length: int,
    lstm_units: List[int],
    attention_units: int,
    dense_units: List[int],
    dropout_rate: float = 0.2,
    output_units: int = 1,
    output_activation: str = "sigmoid"
) -> Model:
    """
    Creates a sequence model with an attention mechanism dynamically based on the
    specified architecture parameters.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features per time step.
    seq_length : int
        Length of the input sequences.
    lstm_units : List[int]
        A list of units for each LSTM layer in the encoder part of the model.
    attention_units : int
        The number of units in the Attention layer.
    dense_units : List[int]
        A list of units for each Dense layer following the attention mechanism.
    dropout_rate : float, default=0.2
        Dropout rate applied after each LSTM and Dense layer.
    output_units : int, default=1
        The number of units in the output layer.
    output_activation : str, default='sigmoid'
        Activation function for the output layer.

    Returns
    -------
    Model
        The constructed attention-based sequence model as a Keras Model instance.

    Examples
    --------
    >>> from gofast.models.deep_search import create_attention_model
    >>> input_dim = 128  # Feature dimensionality
    >>> seq_length = 100  # Length of the sequence
    >>> lstm_units = [64, 64]  # Two LSTM layers
    >>> attention_units = 32  # Attention layer units
    >>> dense_units = [64, 32]  # Two Dense layers after attention
    >>> model = create_attention_model(input_dim, seq_length, lstm_units,
    ...                                attention_units, dense_units,
    ...                                output_units=10, output_activation='softmax')
    >>> model.summary()

    Note
    ----
    This function constructs an attention-based model suitable for handling
    sequence data, enhancing the model's ability to focus on relevant parts
    of the input for making predictions. It's particularly useful for tasks
    such as sequence classification, time series forecasting, or any scenario
    where the importance of different parts of the input sequence may vary.
    """
    inputs = Input(shape=(seq_length, input_dim))

    # Encoder (LSTM layers)
    x = inputs
    for units in lstm_units:
        x = LSTM(units, return_sequences=True)(x)
        x = Dropout(dropout_rate)(x)

    # Attention layer
    query = Dense(attention_units)(x)
    value = Dense(attention_units)(x)
    attention = Attention()([query, value])
    attention = Concatenate()([x, attention])

    # Dense layers after attention
    x = attention
    for units in dense_units:
        x = Dense(units, activation='relu')(x)
        x = Dropout(dropout_rate)(x)

    # Output layer
    outputs = Dense(output_units, activation=output_activation)(x)

    # Construct model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def plot_predictions(
    predicted_custom: np.ndarray, 
    predicted_mse: np.ndarray, 
    actual: np.ndarray, 
    filename: Optional[str] = None, 
    figsize: tuple = (12, 6), 
    title: str = 'Actual vs Predicted',
    xlabel: Optional[str] = None, 
    ylabel: Optional[str] = None, 
    **kws
) -> 'matplotlib.axes.Axes':
    """
    Plots predicted values using custom loss and MSE against actual values,
    optionally saving the plot to a file.

    Parameters
    ----------
    predicted_custom : np.ndarray
        Predicted values using a custom loss function.
    predicted_mse : np.ndarray
        Predicted values using mean squared error (MSE).
    actual : np.ndarray
        Actual values.
    filename : Optional[str], default=None
        Path and filename where the plot will be saved. If None, the plot is not saved.
    figsize : tuple, default=(12, 6)
        Figure size.
    title : str, default='Actual vs Predicted'
        Title of the plot.
    xlabel : Optional[str],
        Label for the X-axis. If None, the X-axis will not be labeled.
    ylabel : Optional[str]
        Label for the Y-axis. If None, the Y-axis will not be labeled.
    **kws
        Additional keyword arguments to pass to `plt.plot`.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object of the plot.

    Examples
    --------
    >>> from gofast.models.deep_search import plot_predictions
    >>> # Assume predicted_custom, predicted_mse, and actual are numpy arrays
    >>> axes = plot_predictions(predicted_custom, predicted_mse, actual,
    ...                         filename='predictions.png',
    ...                         title='Wind Speed Prediction Comparison',
    ...                         xlabel='Time', ylabel='Speed',
    ...                         linestyle='--')
    >>> plt.show()
    """
    plt.figure(figsize=figsize)
    plt.plot(predicted_custom, label='Predicted (Custom Loss)', color='blue', **kws)
    plt.plot(predicted_mse, label='Predicted (MSE)', color='green', **kws)
    plt.plot(actual, label='Actual', color='orange', **kws)
    plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.legend()
    if filename:
        plt.savefig(filename)
    return plt.gca()

def plot_errors(
    predicted_custom: np.ndarray, 
    predicted_mse: np.ndarray, 
    actual: np.ndarray, 
    filename: Optional[str] = None, 
    figsize: tuple = (12, 6), 
    title: str = 'Error in Predictions', 
    xlabel: Optional[str] = 'Time Steps', 
    ylabel: Optional[str] = 'Error',
    **kws
) -> 'matplotlib.axes.Axes':
    """
    Plots the absolute errors in predictions using custom loss and MSE against
    actual values, optionally saving the plot to a file.

    Parameters
    ----------
    predicted_custom : np.ndarray
        Predicted values using a custom loss function.
    predicted_mse : np.ndarray
        Predicted values using mean squared error (MSE).
    actual : np.ndarray
        Actual values to compare against predictions.
    filename : Optional[str], default=None
        Path and filename where the plot will be saved. If None, the plot is not saved.
    figsize : tuple, default=(12, 6)
        Dimensions of the figure.
    title : str, default='Error in Predictions'
        Title of the plot.
    xlabel : Optional[str], default='Time Steps'
        Label for the X-axis. If None, the X-axis will not be labeled.
    ylabel : Optional[str], default='Error'
        Label for the Y-axis. If None, the Y-axis will not be labeled.
    **kws
        Additional keyword arguments to pass to `plt.plot`.

    Returns
    -------
    matplotlib.axes.Axes
        The axes object of the plot.

    Examples
    --------
    >>> from gofast.models.deep_search import plot_errors
    >>> # Assume predicted_custom, predicted_mse, and actual are numpy arrays
    >>> axes = plot_errors(predicted_custom, predicted_mse, actual,
    ...                    filename='errors_plot.png',
    ...                    title='Prediction Error Comparison',
    ...                    xlabel='Time', ylabel='Prediction Error',
    ...                    linestyle='--')
    >>> plt.show()
    """
    error_custom = np.abs(actual - predicted_custom)
    error_mse = np.abs(actual - predicted_mse)
    
    plt.figure(figsize=figsize)
    plt.plot(error_custom, label='Error (Custom Loss)', color='blue', **kws)
    plt.plot(error_mse, label='Error (MSE)', color='green', **kws)
    plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.legend()
    if filename:
        plt.savefig(filename)
    return plt.gca()

