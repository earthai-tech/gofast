# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio
"""
Created on Thu Dec 21 17:06:51 2023
@author: Daniel
"""
import os
import datetime
import warnings 
import numpy as np
import itertools
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.model_selection import KFold

from .._typing import List, Optional, Union, Dict, Tuple
from ..tools._dependency import import_optional_dependency 
try: 
    extra_msg = ("`deep_search` module expects the `tensorflow` library"
                 "to be installed.")
    import_optional_dependency('tensorflow', extra= extra_msg)
    import tensorflow as tf
    from tensorflow.keras.callbacks import ( 
        EarlyStopping, 
        # ModelCheckpoint, 
        History, 
        TensorBoard
        )
except BaseException as e : 
    warnings.warn(str(e) )
else: 
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras import regularizers
    from tensorflow.keras.optimizers import Optimizer, Adam, SGD, RMSprop
    from tensorflow.keras.losses import Loss
    from tensorflow.keras.metrics import Metric


__all__=["plot_history", "base_tuning", "robust_tuning","build_mlp_model", 
         "neural_tuning", "deep_tuning"]

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
        model, train_data, 
        val_data, test_data, 
        learning_rates, 
        batch_sizes, 
        epochs, 
        optimizer
        ):
    """
    Fine-tunes a neural network model based on different hyperparameters.

    Parameters
    ----------
    model : keras.Model
        The neural network model to be fine-tuned.

    train_data : (np.ndarray, np.ndarray)
        Training dataset, including features and labels.

    val_data : (np.ndarray, np.ndarray)
        Validation dataset, including features and labels.

    test_data : (np.ndarray, np.ndarray)
        Test dataset, including features and labels.

    learning_rates : list of float
        List of learning rates to try.

    batch_sizes : list of int
        List of batch sizes to try.

    epochs : int
        Number of epochs for training.

    optimizer : keras.optimizers
        Optimizer to be used for training.

    Returns
    -------
    tuple
        best_model: The best model after fine-tuning.
        best_accuracy: The best accuracy achieved on the validation set.
        test_accuracy: The accuracy on the test set.

    """
    best_accuracy = 0
    best_model = None
    
    for lr in learning_rates:
        for batch_size in batch_sizes:
            # Configure the model with the current hyperparameters
            model.compile(optimizer=optimizer(learning_rate=lr),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
    
            # Train the model => history =
            model.fit(train_data, batch_size=batch_size,
                                epochs=epochs, validation_data=val_data)
    
            # Evaluate the model
            # ->  Assuming the second return value is accuracy
            accuracy = model.evaluate(val_data)[1] 
    
            # Update the best model if current model is better
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
    
    # Optionally, evaluate the best model on the test set
    test_accuracy = best_model.evaluate(test_data)[1]
    
    return best_model, best_accuracy, test_accuracy

def robust_tuning(
    model_fn, dataset, 
    param_grid, n_splits=5, 
    epochs=50, patience=5, 
    log_dir="logs/fit"
    ):
    """
    Fine-tunes a neural network model using cross-validation and grid 
    search for hyperparameter optimization.

    Parameters
    ----------
    model_fn : function
        A function that returns a compiled neural network model.

    dataset : tuple
        Tuple (X, y) of training data and labels.

    param_grid : dict
        Dictionary with hyperparameters to search (
            e.g., {'learning_rate': [0.01, 0.001], 'batch_size': [32, 64]}).

    n_splits : int, optional
        Number of splits for cross-validation.

    epochs : int, optional
        Number of epochs for training.

    patience : int, optional
        Number of epochs to wait for improvement before early stopping.

    log_dir : str, optional
        Directory for TensorBoard logs.

    Returns
    -------
    tuple
        best_model: The best model after fine-tuning.
        best_params: Best hyperparameter set.
        best_score: The best score achieved during cross-validation.

    """
    X, y = dataset
    kf = KFold(n_splits=n_splits)
    best_score = 0
    best_params = None
    best_model = None

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

            # Callbacks
            early_stop = EarlyStopping(monitor='val_loss', patience=patience)
            log_path = os.path.join(
                log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            tensorboard_callback = TensorBoard(log_dir=log_path, 
                                               histogram_freq=1)

            # Train the model
            model.fit(X_train, y_train, validation_data=(X_val, y_val),
                      epochs=epochs, batch_size=params.get('batch_size', 32),
                      callbacks=[early_stop, tensorboard_callback])

            # Evaluate the model
            score = model.evaluate(X_val, y_val, verbose=0)
            scores.append(score)

        # Compute the average score over all folds
        avg_score = np.mean(scores)

        # Update the best model if the current set of parameters is better
        if avg_score > best_score:
            best_score = avg_score
            best_params = params
            best_model = tf.keras.models.clone_model(model)

    return best_model, best_params, best_score

def neural_tuning(
        model, train_data, 
        val_data, test_data, 
        learning_rates, 
        batch_sizes, 
        epochs, optimizer
        ):
    """
    Fine-tunes a neural network model based on different hyperparameters, 
    with a progress bar for each configuration.

    Parameters
    ----------
    model : keras.Model
        The neural network model to be fine-tuned.

    train_data : (np.ndarray, np.ndarray)
        Training dataset, including features and labels.

    val_data : (np.ndarray, np.ndarray)
        Validation dataset, including features and labels.

    test_data : (np.ndarray, np.ndarray)
        Test dataset, including features and labels.

    learning_rates : list of float
        List of learning rates to try.

    batch_sizes : list of int
        List of batch sizes to try.

    epochs : int
        Number of epochs for training.

    optimizer : keras.optimizers
        Optimizer to be used for training.

    Returns
    -------
    tuple
        best_model: The best model after fine-tuning.
        best_accuracy: The best accuracy achieved on the validation set.
        test_accuracy: The accuracy on the test set.

    Example
    -------
    >>> from keras.models import Sequential
    >>> from keras.layers import Dense
    >>> from keras.optimizers import Adam
    >>> model = Sequential([Dense(10, activation='relu'), Dense(1, activation='sigmoid')])
    >>> train_data = (np.random.rand(100, 10), np.random.rand(100))
    >>> val_data = (np.random.rand(20, 10), np.random.rand(20))
    >>> test_data = (np.random.rand(20, 10), np.random.rand(20))
    >>> learning_rates = [0.01, 0.001]
    >>> batch_sizes = [32, 64]
    >>> epochs = 10
    >>> optimizer = Adam
    >>> best_model, best_accuracy, test_accuracy = fair_robust_tuning(
        model, train_data, val_data, test_data, learning_rates, 
        batch_sizes, epochs, optimizer)

    """
    best_accuracy = 0
    best_model = None

    for lr in tqdm(learning_rates, desc='Learning Rates'):
        for batch_size in tqdm(batch_sizes, desc='Batch Sizes'):
            model.compile(optimizer=optimizer(learning_rate=lr),
                          loss='categorical_crossentropy', metrics=['accuracy'])
            model.fit(train_data[0], train_data[1], batch_size=batch_size,
                      epochs=epochs, validation_data=val_data, verbose=0)
            accuracy = model.evaluate(val_data[0], val_data[1], verbose=0)[1]
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = tf.keras.models.clone_model(model)

    test_accuracy = best_model.evaluate(test_data[0], test_data[1], verbose=0)[1]
    return best_model, best_accuracy, test_accuracy

def deep_tuning(
        model_fn, 
        dataset, 
        param_grid, 
        n_splits=5, 
        epochs=50, 
        patience=5, 
        log_dir="logs/fit"
        ):
    """
    Fine-tunes a neural network model using cross-validation and grid search 
    for hyperparameter optimization,
    with a progress bar for each parameter combination.

    Parameters
    ----------
    model_fn : function
        A function that returns a compiled neural network model.

    dataset : tuple
        Tuple (X, y) of training data and labels.

    param_grid : dict
        Dictionary with hyperparameters to search (
            e.g., {'learning_rate': [0.01, 0.001], 'batch_size': [32, 64]}).

    n_splits : int, optional
        Number of splits for cross-validation.

    epochs : int, optional
        Number of epochs for training.

    patience : int, optional
        Number of epochs to wait for improvement before early stopping.

    log_dir : str, optional
        Directory for TensorBoard logs.

    Returns
    -------
    tuple
        best_model: The best model after fine-tuning.
        best_params: Best hyperparameter set.
        best_score: The best score achieved during cross-validation.

    Example
    -------
    >>> import tensorflow as tf
    >>> def model_fn(learning_rate):
    >>>     model = tf.keras.Sequential(
        [tf.keras.layers.Dense(10, activation='relu'),
         tf.keras.layers.Dense(1, activation='sigmoid')])
    >>>     model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy', metrics=['accuracy'])
    >>>     return model
    >>> dataset = (np.random.rand(100, 10), np.random.rand(100))
    >>> param_grid = {'learning_rate': [0.01, 0.001], 'batch_size': [32, 64]}
    >>> best_model, best_params, best_score = deep_tuning(
        model_fn, dataset, param_grid)
    """
    X, y = dataset
    kf = KFold(n_splits=n_splits)
    best_score = 0
    best_params = None
    best_model = None

    param_combinations = [dict(zip(param_grid, v)) 
                          for v in itertools.product(*param_grid.values())]

    for params in tqdm(param_combinations, desc='Parameter Grid'):
        scores = []
        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            model = model_fn(**params)
            model.fit(X_train, y_train, validation_data=(X_val, y_val),
                      epochs=epochs, batch_size=params.get('batch_size', 32), 
                      verbose=0)
            score = model.evaluate(X_val, y_val, verbose=0)[1]
            scores.append(score)

        avg_score = np.mean(scores)
        if avg_score > best_score:
            best_score = avg_score
            best_params = params
            best_model = tf.keras.models.clone_model(model)

    return best_model, best_params, best_score




