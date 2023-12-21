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
import pandas as pd
from joblib import Parallel, delayed
import numpy as np
from sklearn.model_selection import KFold
import itertools
from tqdm import tqdm

from ..tools._dependency import import_optional_dependency 

try: 
    extra_msg = ("Use `deep_search` module implies the `tensorflow` library"
                 "to be installed.")
    import_optional_dependency('tensorflow', extra= extra_msg)
    import tensorflow as tf
    from tensorflow.keras.callbacks import ( 
        EarlyStopping, 
        # ModelCheckpoint, 
        TensorBoard
        )
except BaseException as e : 
    warnings.warn(str(e) )

def fine_tune_model(
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

def complex_fine_tune_model(
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

def fine_tune_model2(
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
    >>> best_model, best_accuracy, test_accuracy = fine_tune_model(
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

def complex_fine_tune_model2(
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
    >>> best_model, best_params, best_score = complex_fine_tune_model(
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


def clean_and_format_dataset(
    dataset, 
    dropna_threshold=0.5, 
    categorical_threshold=10, 
    standardize=True
    ):
    """
    Cleans and formats a dataset for analysis. 
    This includes handling missing values,
    converting data types, and standardizing numerical columns.

    Parameters
    ----------
    dataset : pd.DataFrame
        The dataset to be cleaned and formatted.

    dropna_threshold : float, optional
        The threshold for dropping columns with missing values. 
        Columns with a fraction of missing values greater than 
        this threshold will be dropped.
        Default is 0.5 (50%).

    categorical_threshold : int, optional
        The maximum number of unique values in a column for it to 
        be considered categorical.
        Columns with unique values fewer than or equal to this 
        number will be converted to 
        categorical type. Default is 10.

    standardize : bool, optional
        If True, standardize numerical columns to have a mean of 0 
        and a standard deviation of 1.
        Default is True.

    Returns
    -------
    pd.DataFrame
        The cleaned and formatted dataset.

    Example
    -------
    >>> data = pd.DataFrame({
    >>>     'A': [1, 2, np.nan, 4, 5],
    >>>     'B': ['x', 'y', 'z', 'x', 'y'],
    >>>     'C': [1, 2, 3, 4, 5]
    >>> })
    >>> clean_data = clean_and_format_dataset(data)

    """
    # Drop columns with too many missing values
    dataset = dataset.dropna(thresh=int(
        dropna_threshold * len(dataset)), axis=1)

    # Fill missing values
    for col in dataset.columns:
        if dataset[col].dtype == 'object':
            # For categorical columns, fill with the mode
            dataset[col] = dataset[col].fillna(dataset[col].mode()[0])
        else:
            # For numerical columns, fill with the median
            dataset[col] = dataset[col].fillna(dataset[col].median())

    # Convert columns to categorical if they have fewer unique values than the threshold
    for col in dataset.columns:
        if dataset[col].dtype == 'object' or dataset[col].nunique() <= categorical_threshold:
            dataset[col] = dataset[col].astype('category')

    # Standardize numerical columns
    if standardize:
        num_cols = dataset.select_dtypes(include=['number']).columns
        dataset[num_cols] = (
            dataset[num_cols] - dataset[num_cols].mean()) / dataset[num_cols].std()

    return dataset


def process_large_dataset(data, func, n_jobs=-1):
    """
    Processes a large dataset by applying a complex function to each row, 
    utilizing parallel processing to optimize for speed.

    Parameters
    ----------
    data : pd.DataFrame
        The large dataset to be processed. Assumes the 
        dataset is a Pandas DataFrame.

    func : function
        A complex function to apply to each row of the dataset. 
        This function should take a row of the DataFrame as 
        input and return a processed row.

    n_jobs : int, optional
        The number of jobs to run in parallel. -1 means using 
        all processors. Default is -1.

    Returns
    -------
    pd.DataFrame
        The processed dataset.

    Example
    -------
    >>> def complex_calculation(row):
    >>>     # Example of a complex row-wise calculation
    >>>     return row * 2  # This is a simple placeholder for demonstration.
    >>>
    >>> large_data = pd.DataFrame(np.random.rand(10000, 10))
    >>> processed_data = process_large_dataset(large_data, complex_calculation)

    """
    # Function to apply `func` to each row in parallel
    def process_row(row):
        return func(row)

    # Using Joblib's Parallel and delayed to apply the function in parallel
    results = Parallel(n_jobs=n_jobs)(delayed(process_row)(row) 
                                      for row in data.itertuples(index=False))

    # Converting results back to DataFrame
    processed_data = pd.DataFrame(results, columns=data.columns)

    return processed_data
