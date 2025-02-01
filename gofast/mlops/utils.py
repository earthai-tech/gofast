# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Contains utility functions and classes for machine learning operations 
(MLOps), including configuration management, experiment tracking,
logging setup, hyperparameter management, testing utilities, pipeline
management, and metadata handling.

These utilities are designed to assist in the development, deployment, and
maintenance of machine learning models in production environments.
"""

import os
import json
import logging
import random
import yaml
import pickle
import hashlib
from itertools import product
from typing import Any, Dict, Optional, List, Tuple 
from datetime import datetime
from contextlib import contextmanager

from sklearn.model_selection import ( 
    cross_val_score, 
    StratifiedKFold, 
    KFold, 
    train_test_split
)
import numpy as np

from .._gofastlog import gofastlog 
from ..api.property import BaseClass
from ..decorators import EnsureMethod 
from ..core.io import EnsureFileExists 
from ..utils.validator import parameter_validator 

logger=gofastlog.get_gofast_logger(__name__)


__all__ = [
     'ConfigManager',
     'CrossValidator',
     'DataVersioning',
     'EarlyStopping',
     'ExperimentTrackerIn',
     'ParameterGrid',
     'PipelineBuilder',
     'Timer',
     'TrainTestSplitter',
     'calculate_metrics',
     'get_model_metadata',
     'load_model',
     'load_pipeline',
     'log_model_summary',
     'save_model',
     'save_pipeline',
     'set_random_seed',
     'setup_logging',
]

class ConfigManager(BaseClass):
    """
    Manages configuration files in JSON or YAML format.

    This class provides methods to load, save, and update configuration
    settings for machine learning projects. It supports both JSON and YAML
    formats.

    Parameters
    ----------
    config_file : str
        Path to the configuration file.
    config_format : {'json', 'yaml'}, optional
        Format of the configuration file. Defaults to ``'json'``.

    Attributes
    ----------
    config : dict
        Dictionary containing the configuration settings.

    Examples
    --------
    >>> from gofast.mlops.utils import ConfigManager
    >>> config_manager = ConfigManager('config.yaml', config_format='yaml')
    >>> config = config_manager.load_config()
    >>> config['learning_rate'] = 0.001
    >>> config_manager.save_config()

    """
    
    @EnsureFileExists
    def __init__(self, config_file: str, config_format: str = 'json'):

        self.config_format = config_format
        self.config: Dict[str, Any] = {}
        
        self.config_file =parameter_validator(
            config_format, target_strs=['json', 'yaml', 'yml'],  
            return_target_str=True, error_msg=(
                "config_format must be 'json' or 'yaml'")
            )
        
    def load_config(self) -> Dict[str, Any]:
        """
        Loads the configuration from the file.

        Returns
        -------
        config : dict
            The loaded configuration settings.

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist.

        """
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Configuration file '{self.config_file}' not found.")

        with open(self.config_file, 'r') as f:
            if self.config_format == 'json':
                self.config = json.load(f)
            elif self.config_format == 'yaml':
                self.config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from '{self.config_file}'.")
        return self.config

    def save_config(self):
        """
        Saves the current configuration to the file.

        Raises
        ------
        IOError
            If the configuration file cannot be written.

        """
        with open(self.config_file, 'w') as f:
            if self.config_format == 'json':
                json.dump(self.config, f, indent=4)
            elif self.config_format == 'yaml':
                yaml.safe_dump(self.config, f)
        logger.info(f"Configuration saved to '{self.config_file}'.")

    def update_config(self, updates: Dict[str, Any]):
        """
        Updates the configuration with new settings.

        Parameters
        ----------
        updates : dict
            Dictionary containing the configuration updates.

        Examples
        --------
        >>> config_manager.update_config({'batch_size': 64})

        """
        self.config.update(updates)
        logger.info("Configuration updated.")

class ExperimentTrackerIn(BaseClass):
    """
    Tracks experiments, logging hyperparameters, metrics, and artifacts.

    This class provides methods to log experiment details, save artifacts,
    and maintain a record of different runs.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.
    base_dir : str, optional
        Base directory to store experiment logs and artifacts.
        Defaults to ``'experiments'``.

    Attributes
    ----------
    experiment_dir : str
        Directory where experiment logs and artifacts are stored.

    Examples
    --------
    >>> from gofast.mlops.utils import ExperimentTracker
    >>> tracker = ExperimentTracker('my_experiment')
    >>> tracker.log_params({'learning_rate': 0.001, 'batch_size': 32})
    >>> tracker.log_metrics({'accuracy': 0.95})
    >>> tracker.save_artifact('model.pkl')

    """

    def __init__(self, experiment_name: str, base_dir: str = 'experiments'):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
        os.makedirs(self.experiment_dir, exist_ok=True)
        self.params_file = os.path.join(self.experiment_dir, 'params.json')
        self.metrics_file = os.path.join(self.experiment_dir, 'metrics.json')
        self.artifacts_dir = os.path.join(self.experiment_dir, 'artifacts')
        os.makedirs(self.artifacts_dir, exist_ok=True)
        logger.info(f"Experiment directory created at '{self.experiment_dir}'.")

    def log_params(self, params: Dict[str, Any]):
        """
        Logs hyperparameters for the experiment.

        Parameters
        ----------
        params : dict
            Dictionary of hyperparameters.

        """
        with open(self.params_file, 'w') as f:
            json.dump(params, f, indent=4)
        logger.info("Parameters logged.")

    def log_metrics(self, metrics: Dict[str, Any]):
        """
        Logs metrics for the experiment.

        Parameters
        ----------
        metrics : dict
            Dictionary of metric names and values.

        """
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info("Metrics logged.")

    @EnsureFileExists(file_param="artifact_path", action="ignore")
    def save_artifact(self, artifact_path: str, artifact_name: Optional[str] = None):
        """
        Saves an artifact file to the experiment directory.

        Parameters
        ----------
        artifact_path : str
            Path to the artifact file.
        artifact_name : str, optional
            Name to save the artifact as. If not provided, uses the original file name.

        """
        if artifact_name is None:
            artifact_name = os.path.basename(artifact_path)
        dest_path = os.path.join(self.artifacts_dir, artifact_name)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(artifact_path, 'rb') as src, open(dest_path, 'wb') as dst:
            dst.write(src.read())
        logger.info(f"Artifact '{artifact_name}' saved.")

class EarlyStopping(BaseClass):
    """
    Implements early stopping mechanism to halt training when performance degrades.

    Parameters
    ----------
    patience : int, optional
        Number of epochs with no improvement after which training will be stopped.
        Defaults to ``5``.
    delta : float, optional
        Minimum change in the monitored metric to qualify as an improvement.
        Defaults to ``0.0``.
    monitor : str, optional
        Metric to monitor. Defaults to ``'loss'``.

    Attributes
    ----------
    best_score : float
        Best score achieved so far.
    counter : int
        Number of epochs since the last improvement.
    early_stop : bool
        Flag indicating whether early stopping should trigger.

    Examples
    --------
    >>> from gofast.mlops.utils import EarlyStopping
    >>> early_stopper = EarlyStopping(patience=3)
    >>> for epoch in range(epochs):
    ...     train_loss = train(...)
    ...     val_loss = validate(...)
    ...     early_stopper(val_loss)
    ...     if early_stopper.early_stop:
    ...         break

    """

    def __init__(self, patience: int = 5, delta: float = 0.0, monitor: str = 'loss'):
        self.patience = patience
        self.delta = delta
        self.monitor = monitor
        self.best_score: Optional[float] = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, value: float):
        """
        Updates the early stopping mechanism with the latest metric value.

        Parameters
        ----------
        value : float
            The latest value of the monitored metric.

        """
        score = -value if self.monitor == 'loss' else value
        if self.best_score is None:
            self.best_score = score
            logger.info("Initial best score set.")
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(f"No improvement. Early stopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info("Early stopping triggered.")
        else:
            self.best_score = score
            self.counter = 0
            logger.info("Improvement detected. Counter reset.")


class DataVersioning(BaseClass):
    """
    Manages data versioning for datasets using checksums and metadata.

    Parameters
    ----------
    data_dir : str
        Directory containing the data files.
    metadata_file : str, optional
        Path to the metadata file for storing checksums.
        Defaults to ``'data_metadata.json'``.

    Attributes
    ----------
    data_dir : str
        Directory containing the data files.
    metadata_file : str
        Path to the metadata file.

    Examples
    --------
    >>> from gofast.mlops.utils import DataVersioning
    >>> data_versioning = DataVersioning('data/')
    >>> data_versioning.generate_checksums()
    >>> data_versioning.check_for_changes()

    """

    def __init__(self, data_dir: str, metadata_file: str = 'data_metadata.json'):
        self.data_dir = data_dir
        self.metadata_file = metadata_file
        self.metadata: Dict[str, str] = {}

    def generate_checksums(self):
        """
        Generates checksums for all files in the data directory and saves them
        to the metadata file.

        """
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                checksum = self._calculate_checksum(file_path)
                relative_path = os.path.relpath(file_path, self.data_dir)
                self.metadata[relative_path] = checksum
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=4)
        logger.info("Data checksums generated and saved.")

    def check_for_changes(self) -> bool:
        """
        Checks for changes in the data files by comparing current checksums with
        those stored in the metadata file.

        Returns
        -------
        changes_detected : bool
            True if changes are detected, False otherwise.

        """

        if not os.path.exists(self.metadata_file):
            raise FileNotFoundError(f"Metadata file '{self.metadata_file}' not found.")

        with open(self.metadata_file, 'r') as f:
            stored_metadata = json.load(f)

        current_metadata = {}
        changes_detected = False

        for root, _, files in os.walk(self.data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                checksum = self._calculate_checksum(file_path)
                relative_path = os.path.relpath(file_path, self.data_dir)
                current_metadata[relative_path] = checksum
                if relative_path not in stored_metadata or stored_metadata[relative_path] != checksum:
                    logger.warning(f"Change detected in file '{relative_path}'.")
                    changes_detected = True

        if changes_detected:
            logger.info("Data changes detected.")
        else:
            logger.info("No data changes detected.")
        return changes_detected

    def _calculate_checksum(self, file_path: str) -> str:
        """
        Calculates the MD5 checksum of a file.

        Parameters
        ----------
        file_path : str
            Path to the file.

        Returns
        -------
        checksum : str
            MD5 checksum of the file.

        """

        hash_md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

class ParameterGrid(BaseClass):
    """
    Generates a grid of parameter combinations for hyperparameter tuning.

    Parameters
    ----------
    param_grid : dict
        Dictionary with parameters names (`str`) as keys and lists of parameter
        settings to try as values.

    Examples
    --------
    >>> from gofast.mlops.utils import ParameterGrid
    >>> param_grid = {
    ...     'learning_rate': [0.001, 0.01],
    ...     'batch_size': [16, 32],
    ...     'optimizer': ['adam', 'sgd']
    ... }
    >>> grid = ParameterGrid(param_grid)
    >>> for params in grid:
    ...     print(params)

    """

    def __init__(self, param_grid: Dict[str, List[Any]]):
        self.param_grid = param_grid
        self._grid = self._generate_grid()

    def _generate_grid(self) -> List[Dict[str, Any]]:
        

        items = sorted(self.param_grid.items())
        keys, values = zip(*items)
        experiments = [dict(zip(keys, v)) for v in product(*values)]
        logger.info(f"Generated {len(experiments)} parameter combinations.")
        return experiments

    def __iter__(self):
        return iter(self._grid)

    def __len__(self):
        return len(self._grid)

    def __getitem__(self, idx):
        return self._grid[idx]

@EnsureMethod(error='ignore', mode='soft')
class TrainTestSplitter(BaseClass):
    """
    Utility class for splitting data into training, validation, 
    and testing sets with extended functionality.

    This class provides methods to split datasets using various 
    strategies. It supports the standard train-test split, splitting 
    into train, validation, and test sets, k-fold cross-validation, 
    and time series splitting. The basic train-test split is defined 
    mathematically as:

    .. math::
       (X_{train}, X_{test}, y_{train}, y_{test}) =
       f(X, y, \\text{test\_size}, \\text{random\_state}, \\text{stratify})

    where :math:`X` is the feature matrix, :math:`y` is the target 
    vector, and the parameters control the proportions and randomness 
    of the split.

    Parameters
    ----------
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split. For 
        example, ``test_size=0.2`` implies that 20% of the data is used 
        for testing.
    random_state : int, default=42
        Seed for the random number generator to ensure reproducible 
        splits. For example, ``random_state=42`` guarantees that the same 
        split is produced on each run.
    stratify : bool or array-like, default=False
        If set to ``True`` or provided as an array-like, the split is 
        performed in a stratified manner based on the target variable.
    verbose : int, default=0
        Controls the verbosity of the class. A higher value produces 
        more detailed debugging output.

    Attributes
    ----------
    test_size : float
        See parameter `test_size`.
    random_state : int
        See parameter `random_state`.
    stratify : bool or array-like
        See parameter `stratify`.
    verbose : int
        See parameter `verbose`.

    Methods
    -------
    split(*arrays, test_size: float, random_state: int, stratify)
        Splits the input arrays into training and testing sets.
    split_train_validate_test(*arrays, test_size: float, 
        val_size: float, random_state: int, stratify)
        Splits the data into training, validation, and testing sets.
    k_fold_split(*arrays, n_splits: int, random_state: int, 
        shuffle: bool)
        Generates indices for k-fold cross-validation splits.
    time_series_split(*arrays, test_size: int)
        Splits time series data while preserving temporal order.

    Examples
    --------
    Create a splitter for a dataset with stratification:

    >>> from gofast.mlops.metadata import TrainTestSplitter
    >>> splitter = TrainTestSplitter(
    ...     test_size=0.3, 
    ...     random_state=42, 
    ...     stratify=True,
    ...     verbose=1
    ... )
    >>> X_train, X_test, y_train, y_test = splitter.split(X, y)
    >>> print(X_train.shape, X_test.shape)

    Notes
    -----
    The splitting methods leverage scikit-learn's utilities (e.g., 
    :py:func:`sklearn.model_selection.train_test_split`) to perform 
    the splits. In addition, methods like `k_fold_split` and 
    `time_series_split` provide advanced functionality for cross- 
    validation and sequential data splitting.

    See Also
    --------
    sklearn.model_selection.train_test_split : Function to split data.
    sklearn.model_selection.KFold : K-fold cross-validation iterator.

    References
    ----------
    .. [1] Pedregosa, F. et al. "Scikit-learn: Machine Learning in Python." 
           Journal of Machine Learning Research, 2011.
    """

    def __init__(
        self,
        test_size   : float = 0.2,
        random_state: int   = 42,
        stratify    : bool or Any = False,
        verbose     : int   = 0,
    ):
        # Initialize splitting parameters.
        self.test_size    = test_size
        self.random_state = random_state
        self.stratify     = stratify
        self.verbose      = verbose

    @staticmethod
    def split(
        *arrays,
        test_size   : float = 0.2,
        random_state: int   = 42,
        stratify    : bool or Any = False
    ):
        """
        Splits the input arrays into training and testing sets.

        This method uses scikit-learn's 
        :py:func:`sklearn.model_selection.train_test_split` to perform 
        the split. It returns training and testing subsets for each 
        provided array.

        Parameters
        ----------
        arrays : array-like
            The data arrays to be split, e.g. features `<X>` and targets `<y>`.
        test_size : float, default=0.2
            Proportion of the dataset allocated to the test split.
        random_state : int, default=42
            Seed for random shuffling.
        stratify : bool or array-like, default=False
            If provided, performs stratified splitting based on the target 
            variable.

        Returns
        -------
        tuple
            A tuple containing the training and testing splits in the 
            following order: ``(X_train, X_test, y_train, y_test)``.

        Examples
        --------
        >>> from gofast.mlops.metadata import TrainTestSplitter
        >>> X_train, X_test, y_train, y_test = TrainTestSplitter.split(X, y)
        """
        from sklearn.model_selection import train_test_split
        # Validate test_size if necessary via a helper function.
        splitted = train_test_split(
            *arrays,
            test_size   = test_size,
            random_state= random_state,
            stratify    = stratify
        )
        if TrainTestSplitter().verbose:
            print("Data split into training and testing sets.")
        return splitted

    def split_train_validate_test(
        self,
        *arrays,
        test_size   : float = 0.2,
        val_size    : float = 0.1,
        random_state: int   = 42,
        stratify    : bool or Any = False
    ):
        """
        Splits data into training, validation, and testing sets.

        First, splits the dataset into training and testing sets. Then, 
        splits the training set further into training and validation sets.

        Parameters
        ----------
        arrays : array-like
            The data arrays to split (e.g. features `<X>` and targets `<y>`).
        test_size : float, default=0.2
            Proportion of the data allocated to the test set.
        val_size : float, default=0.1
            Proportion of the training data allocated to the validation set.
        random_state : int, default=42
            Seed for reproducibility.
        stratify : bool or array-like, default=False
            If provided, performs stratified splitting based on the target.

        Returns
        -------
        tuple
            A tuple with splits: 
            ``(X_train, X_val, X_test, y_train, y_val, y_test)``.

        Examples
        --------
        >>> from gofast.mlops.metadata import TrainTestSplitter
        >>> splits = TrainTestSplitter().split_train_validate_test(X, y)
        >>> X_train, X_val, X_test, y_train, y_val, y_test = splits
        """
  
        # First split: train+validation and test
        train_val, X_test, y_train, y_test = train_test_split(
            *arrays,
            test_size    = test_size,
            random_state = random_state,
            stratify     = stratify
        )
        # Second split: train and validation from train_val.
        val_relative_size = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            *train_val,
            test_size    = val_relative_size,
            random_state = random_state,
            stratify     = stratify
        )
        if self.verbose:
            print("Data split into training, validation, and testing sets.")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def k_fold_split(
        self,
        *arrays,
        n_splits    : int = 5,
        random_state: int = 42,
        shuffle     : bool = True
    ):
        """
        Generates k-fold cross-validation indices for the input data.

        This method leverages scikit-learn's 
        :py:class:`sklearn.model_selection.KFold` to yield indices 
        for k-fold splitting.

        Parameters
        ----------
        arrays : array-like
            The data arrays to split.
        n_splits : int, default=5
            Number of folds.
        random_state : int, default=42
            Seed for random shuffling of data before splitting.
        shuffle : bool, default=True
            If ``True``, shuffles data prior to splitting.

        Returns
        -------
        generator
            A generator yielding tuples of train and test indices for each fold.

        Examples
        --------
        >>> from gofast.mlops.metadata import TrainTestSplitter
        >>> splitter = TrainTestSplitter()
        >>> for train_idx, test_idx in splitter.k_fold_split(X, y, n_splits=5):
        ...     print(train_idx, test_idx)
        """
        from sklearn.model_selection import KFold
        kf = KFold(
            n_splits    = n_splits,
            random_state= random_state,
            shuffle     = shuffle
        )
        return kf.split(arrays[0])

    def time_series_split(
        self,
        *arrays,
        test_size: int = 100
    ):
        """
        Splits time series data while preserving temporal order.

        Unlike random splits, this method divides data sequentially,
        assigning the last `<test_size>` samples to the test set and
        the remaining to the training set.

        Parameters
        ----------
        arrays : array-like
            The data arrays to split, where the first axis represents 
            time.
        test_size : int, default=100
            The number of data points to assign to the test set.

        Returns
        -------
        tuple
            A tuple containing training and testing splits for each input array.

        Examples
        --------
        >>> from gofast.mlops.metadata import TrainTestSplitter
        >>> X_train, X_test, y_train, y_test = TrainTestSplitter(
            ).time_series_split(X, y, test_size=50)
        >>> print(len(X_test))
        50
        """
        # Determine split index based on test_size
        split_index = -test_size
        results = []
        for array in arrays:
            # Slice array: training set is all elements except the last test_size
            # and test set is the last test_size elements.
            train = array[:split_index]
            test  = array[split_index:]
            results.extend([train, test])
        if self.verbose:
            print("Time series split completed; test set size:", test_size)
        return tuple(results)
    

@EnsureMethod(error='ignore', mode='soft')
class CrossValidator(BaseClass):
    """
    Utility class for performing and extending cross-validation 
    on machine learning models.

    This class provides methods for standard k-fold cross-validation, 
    stratified splits, grid search with cross-validation, retrieving 
    fold indices, and plotting cross-validation scores. The overall 
    evaluation of a model using cross-validation can be formulated as:

    .. math::
        S = f(X, y, \\text{cv}, \\text{scoring})

    where :math:`S` is the set of scores computed over `cv` folds for 
    input feature matrix :math:`X` and target vector :math:`y`.

    Parameters
    ----------
    model : Any
        The machine learning model to evaluate, e.g. ``<model>``.
    cv : int, default=5
        Number of folds in the cross-validation. For example, 
        ``<cv>`` = ``5``.
    stratified : bool, default=False
        If ``True``, performs stratified splitting based on the target.
    verbose : int, default=0
        Verbosity level for debugging; valid values are from 
        ``0`` (silent) to ``3`` (most verbose).

    Attributes
    ----------
    model : Any
        See parameter `model`.
    cv : int
        See parameter `cv`.
    stratified : bool
        See parameter `stratified`.
    verbose : int
        See parameter `verbose`.

    Methods
    -------
    run() -> CrossValidator
        Activates the cross-validator, ensuring it is ready for use.
    evaluate(X: np.ndarray, y: np.ndarray, scoring: str) -> dict
        Evaluates the model using cross-validation and returns 
        the mean and standard deviation of the scores.
    get_fold_indices(X: np.ndarray, y: np.ndarray = None) -> list
        Returns indices for each fold, using stratified or regular 
        K-Fold splitting.
    grid_search_evaluate(X: np.ndarray, y: np.ndarray, param_grid: dict, scoring: str) -> dict
        Performs grid search cross-validation and returns the best 
        parameters and score.
    plot_scores(X: np.ndarray, y: np.ndarray, scoring: str) -> None
        Plots the distribution of cross-validation scores.

    Examples
    --------
    >>> from gofast.mlops.metadata import CrossValidator
    >>> cv = CrossValidator(
    ...     model=my_model,
    ...     cv=5,
    ...     stratified=True,
    ...     verbose=2
    ... )
    >>> results = cv.evaluate(X, y, scoring="accuracy")
    >>> print(results)
    {'mean_score': 0.92, 'std_score': 0.03, 'scores': [0.90, 0.93, 0.92, 0.91, 0.94]}
    >>> fold_indices = cv.get_fold_indices(X, y)
    >>> for train_idx, test_idx in fold_indices:
    ...     print(train_idx, test_idx)
    >>> grid_results = cv.grid_search_evaluate(
    ...     X, y, param_grid={"C": [0.1, 1, 10]}, scoring="accuracy"
    ... )
    >>> cv.plot_scores(X, y, scoring="accuracy")

    See Also
    --------
    sklearn.model_selection.train_test_split :
        Splitting data into training and test sets.
    sklearn.model_selection.KFold : 
        K-fold cross-validation iterator.
    sklearn.model_selection.GridSearchCV : 
        Exhaustive search over specified parameter values.
    
    References
    ----------
    .. [1] Pedregosa, F. et al. "Scikit-learn: Machine Learning in Python." 
           Journal of Machine Learning Research, 2011.
    """
    def __init__(
            self,
            model         : Any,
            cv            : int  = 5,
            stratified    : bool = False,
            verbose       : int  = 0
    ):
        # Initialize the cross-validator with the given model and parameters.
        self.model      = model
        self.cv         = cv
        self.stratified = stratified
        self.verbose    = verbose

    def run(self) -> "CrossValidator":
        # Set up the cross-validator (if any pre-run steps are needed)
        if self.verbose >= 1:
            logger.info("CrossValidator activated.")
        return self

    def evaluate(
        self,
        X         : np.ndarray,
        y         : np.ndarray,
        scoring   : str = "accuracy"
    ) -> Dict[str, Any]:
        
        # Perform cross-validation and compute scores.
        scores = cross_val_score(
                    self.model, X, y,
                    cv         = self.cv,
                    scoring    = scoring
                 )
        results = {
            "mean_score" : np.mean(scores),
            "std_score"  : np.std(scores),
            "scores"     : scores.tolist()
        }
        if self.verbose >= 1:
            logger.info("Cross-validation completed.")
        return results

    def get_fold_indices(
            self,
            X : np.ndarray,
            y : Optional[np.ndarray] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        # Use StratifiedKFold if stratification is enabled and y is provided.
        if self.stratified and y is not None:
            kf = StratifiedKFold(
                     n_splits    = self.cv,
                     shuffle     = True,
                     random_state= 42
                 )
        else:
            kf = KFold(
                     n_splits    = self.cv,
                     shuffle     = True,
                     random_state= 42
                 )
        indices = list(kf.split(X, y))
        if self.verbose >= 2:
            logger.info("Fold indices generated.")
        return indices

    def grid_search_evaluate(
            self,
            X          : np.ndarray,
            y          : np.ndarray,
            param_grid : Dict[str, List[Any]],
            scoring    : str = "accuracy"
    ) -> Dict[str, Any]:
        from sklearn.model_selection import GridSearchCV
        # Execute grid search with cross-validation.
        grid = GridSearchCV(
                    estimator   = self.model,
                    param_grid  = param_grid,
                    cv          = self.cv,
                    scoring     = scoring,
                    verbose     = self.verbose
                )
        grid.fit(X, y)
        if self.verbose >= 1:
            logger.info("Grid search completed.")
        return {
            "best_params" : grid.best_params_,
            "best_score"  : grid.best_score_,
            "cv_results"  : grid.cv_results_
        }

    def plot_scores(
            self,
            X         : np.ndarray,
            y         : np.ndarray,
            scoring   : str = "accuracy"
    ) -> None:
        import matplotlib.pyplot as plt
        from sklearn.model_selection import cross_val_score
        # Compute cross-validation scores.
        scores = cross_val_score(
                    self.model, X, y,
                    cv       = self.cv,
                    scoring  = scoring
                 )
        plt.figure(figsize=(8, 5))
        plt.plot(scores, marker="o", linestyle="-")
        plt.title("Cross-Validation Scores")
        plt.xlabel("Fold")
        plt.ylabel(f"Score ({scoring})")
        plt.grid(True)
        plt.show()
        if self.verbose >= 1:
            logger.info("Cross-validation scores plotted.")

class PipelineBuilder(BaseClass):
    """
    Utility class for building machine learning 
    pipelines.

    This class simplifies the construction of 
    machine learning pipelines by chaining 
    together preprocessing steps and 
    estimators. The pipeline is defined as a 
    sequence of (name, transform) tuples, where 
    each `<name>` is a string identifier and 
    each ``<transform>`` implements the fit/ 
    transform interface.

    The pipeline is represented as:

    .. math::
       P = T_1 \\circ T_2 \\circ \\cdots \\circ T_n

    where :math:`T_i` is the i-th step in the pipeline.

    Parameters
    ----------
    steps : list of tuple
        List of (name, transform) tuples. Each 
        tuple consists of a `<name>` and a 
        ``<transform>`` object. For example, 
        ``steps=[("scaler", StandardScaler()),
        ("classifier", LogisticRegression())]``.

    Attributes
    ----------
    steps : list of tuple
        The pipeline steps provided during 
        initialization.

    Methods
    -------
    build() -> Pipeline
        Constructs and returns a scikit-learn 
        Pipeline object composed of the specified 
        steps.

    Examples
    --------
    >>> from gofast.mlops.metadata import PipelineBuilder
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.linear_model import LogisticRegression
    >>> builder = PipelineBuilder([
    ...     ('scaler', StandardScaler()),
    ...     ('classifier', LogisticRegression())
    ... ])
    >>> pipeline = builder.build()
    >>> pipeline.fit(X_train, y_train)

    See Also
    --------
    sklearn.pipeline.Pipeline : Class for creating ML 
        pipelines.

    References
    ----------
    .. [1] Pedregosa, F. et al. "Scikit-learn: Machine Learning 
           in Python." Journal of Machine Learning Research, 
           2011.
    """

    def __init__(
        self,
        steps: List[tuple], 
        verbose: int=0, 
        ):
        self.steps = steps
        self.verbose=verbose
        self._is_built =False 

    def add_step(
        self,
        name : str,
        transformer : Any,
        position : Optional[int] = None
        ) -> None:
        # Add a new step to the pipeline.
        # If position is specified, insert at that index; otherwise, append.
        if position is None:
            self.steps.append((name, transformer))
        else:
            self.steps.insert(position, (name, transformer))
        logger.info(f"Added step ``{name}`` at position " +
                    (f"``{position}``" if position is not None else "end"))

    def remove_step(
         self,
         name : str
         ) -> None:
        # Remove a step identified by its name.
        original_len = len(self.steps)
        self.steps = [step for step in self.steps if step[0] != name]
        if len(self.steps) < original_len:
            logger.info(f"Removed step ``{name}`` from pipeline.")
        else:
            logger.warning(f"Step ``{name}`` not found in pipeline.")

    def update_step(
         self,
        name  : str,
        transformer   : Any
        ) -> None:
        # Update an existing step identified by name with a new transformer.
        updated = False
        new_steps = []
        for step_name, step_transformer in self.steps:
            if step_name == name:
                new_steps.append((name, transformer))
                updated = True
            else:
                new_steps.append((step_name, step_transformer))
        if not updated:
            # If the step is not found, add it as a new step.
            new_steps.append((name, transformer))
            logger.warning(
                f"Step ``{name}`` not found; added as new step.")
        else:
            logger.info(f"Updated step ``{name}`` in pipeline.")
        self.steps = new_steps

    def get_steps(self) -> List[tuple]:
        # Return the list of steps in the pipeline.
        return self.steps

    def build(self, memory=None) -> Any:
        # Build and return a scikit-learn Pipeline object.
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline(self.steps, memory=memory, verbose=self.verbose)
        logger.info("Pipeline constructed successfully.")
        self.pipeline_= pipeline 
        
        self._is_built=True 
        return pipeline

    def save(
        self,
        filename : Optional[str]=None
        ) -> None:
        # Save the constructed pipeline to a file using joblib.
        import joblib
        
        filename = filename or '_my_pipeline.joblib'
        
        if not self._is_built: 
            self.pipeline_ = self.build()
        joblib.dump(self.pipeline_, filename)
        logger.info(f"Pipeline saved to file: ``{filename}``.")
        if self.verbose: 
            print(f"My pipeline saved to: {filename}")

    @classmethod
    def load(cls, filename : str) -> Any:
        # Load and return a pipeline from a file using joblib.
        import joblib
        pipeline = joblib.load(filename)
        logger.info(f"Pipeline loaded from file: ``{filename}``.")
   
        return pipeline

    def visualize(self) -> None:
        # Visualize the pipeline structure by printing the steps.
        print("Pipeline Structure:")
        for idx, (name, transformer) in enumerate(self.steps):
            # Print step index, name, and transformer class.
            print(f"  {idx + 1}. ``{name}`` -> ``{transformer.__class__.__name__}``")
        logger.info("Pipeline structure visualized.")



def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO):
    """
    Sets up logging configuration.

    Parameters
    ----------
    log_file : str, optional
        Path to the log file. If not provided, logs are output to the console.
    level : int, optional
        Logging level. Defaults to ``logging.INFO``.

    Examples
    --------
    >>> from gofast.mlops.utils import setup_logging
    >>> setup_logging('training.log')

    """
    handlers = [logging.StreamHandler()]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    logger.info("Logging setup complete.")

def save_model(model: Any, model_path: str):
    """
    Saves a machine learning model to the specified path using pickle.

    Parameters
    ----------
    model : object
        The machine learning model to save.
    model_path : str
        Path where the model will be saved.

    Examples
    --------
    >>> from gofast.mlops.utils import save_model
    >>> save_model(model, 'models/my_model.pkl')

    """
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to '{model_path}'.")

def load_model(model_path: str) -> Any:
    """
    Loads a machine learning model from the specified path.

    Parameters
    ----------
    model_path : str
        Path to the saved model file.

    Returns
    -------
    model : object
        The loaded machine learning model.

    Examples
    --------
    >>> from gofast.mlops.utils import load_model
    >>> model = load_model('models/my_model.pkl')

    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    logger.info(f"Model loaded from '{model_path}'.")
    return model

@contextmanager
def Timer(name: str):
    """
    Context manager for timing code execution.

    Parameters
    ----------
    name : str
        Name to associate with the timing block.

    Examples
    --------
    >>> from gofast.mlops.utils import Timer
    >>> with Timer('data_loading'):
    ...     # Code block to time
    ...     data = load_data()

    """
    start_time = datetime.now()
    logger.info(f"Starting '{name}'...")
    yield
    elapsed_time = datetime.now() - start_time
    logger.info(f"'{name}' completed in {elapsed_time}.")

def set_random_seed(seed: int = 42):
    """
    Sets the random seed for reproducibility.

    Parameters
    ----------
    seed : int, optional
        The seed value to use. Defaults to ``42``.

    Notes
    -----
    This function sets the seed for the `random`, `numpy`, and `torch` (if available)
    modules to ensure reproducible results.

    Examples
    --------
    >>> from gofast.mlops.utils import set_random_seed
    >>> set_random_seed(123)

    """
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Random seed set to {seed}.")
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        logger.info("Random seed set for PyTorch.")
    except ImportError:
        logger.warning("PyTorch not installed; skipping torch seed setting.")



def calculate_metrics(
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        metrics: Optional[List[str]] = None
        ) -> Dict[str, float]:
    """
    Calculates evaluation metrics for model predictions.

    Parameters
    ----------
    y_true : ndarray
        Ground truth labels.
    y_pred : ndarray
        Predicted labels or probabilities.
    metrics : list of str, optional
        List of metrics to calculate. Supported metrics are:
        'accuracy', 'precision', 'recall', 'f1', 'roc_auc'.

    Returns
    -------
    results : dict
        Dictionary of metric names and their calculated values.

    Examples
    --------
    >>> from gofast.mlops.utils import calculate_metrics
    >>> metrics = calculate_metrics(y_true, y_pred, metrics=['accuracy', 'f1'])

    """
    from sklearn.metrics import ( 
        accuracy_score, 
        precision_score, 
        recall_score, 
        f1_score, 
        roc_auc_score
        )

    if metrics is None:
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    results = {}
    for metric in metrics:
        if metric == 'accuracy':
            results['accuracy'] = accuracy_score(y_true, y_pred)
        elif metric == 'precision':
            results['precision'] = precision_score(y_true, y_pred, average='binary')
        elif metric == 'recall':
            results['recall'] = recall_score(y_true, y_pred, average='binary')
        elif metric == 'f1':
            results['f1'] = f1_score(y_true, y_pred, average='binary')
        elif metric == 'roc_auc':
            if len(np.unique(y_true)) == 2:
                results['roc_auc'] = roc_auc_score(y_true, y_pred)
            else:
                logger.warning("ROC AUC is not defined for multi-class classification.")
        else:
            logger.warning(f"Unsupported metric: {metric}")
    logger.info("Metrics calculated.")
    return results



def save_pipeline(pipeline: Any, pipeline_path: str):
    """
    Saves a machine learning pipeline to the specified path using joblib.

    Parameters
    ----------
    pipeline : object
        The pipeline object to save.
    pipeline_path : str
        Path where the pipeline will be saved.

    Examples
    --------
    >>> from gofast.mlops.utils import save_pipeline
    >>> save_pipeline(pipeline, 'pipelines/my_pipeline.pkl')

    """
    from joblib import dump

    dump(pipeline, pipeline_path)
    logger.info(f"Pipeline saved to '{pipeline_path}'.")

def load_pipeline(pipeline_path: str) -> Any:
    """
    Loads a machine learning pipeline from the specified path.

    Parameters
    ----------
    pipeline_path : str
        Path to the saved pipeline file.

    Returns
    -------
    pipeline : object
        The loaded pipeline object.

    Examples
    --------
    >>> from gofast.mlops.utils import load_pipeline
    >>> pipeline = load_pipeline('pipelines/my_pipeline.pkl')

    """
    from joblib import load

    pipeline = load(pipeline_path)
    logger.info(f"Pipeline loaded from '{pipeline_path}'.")
    return pipeline

def log_model_summary(model: Any, model_name: str = 'Model'):
    """
    Logs a summary of the model structure.

    Parameters
    ----------
    model : object
        The machine learning model.
    model_name : str, optional
        Name of the model. Defaults to ``'Model'``.

    Examples
    --------
    >>> from gofast.mlops.utils import log_model_summary
    >>> log_model_summary(model, model_name='Neural Network')

    """
    from sklearn.utils import estimator_html_repr

    try:
        summary = estimator_html_repr(model)
        logger.info(f"{model_name} summary:\n{summary}")
    except Exception as e:
        logger.warning(f"Could not generate model summary: {e}")

def get_model_metadata(model: Any) -> Dict[str, Any]:
    """
    Extracts metadata from a scikit-learn model.

    Parameters
    ----------
    model : object
        The scikit-learn model.

    Returns
    -------
    metadata : dict
        Dictionary containing model metadata, such as parameters and estimator type.

    Examples
    --------
    >>> from gofast.mlops.utils import get_model_metadata
    >>> metadata = get_model_metadata(model)
    >>> print(metadata)

    """
    metadata = {
        'class_name': model.__class__.__name__,
        'module': model.__module__,
        'params': model.get_params()
    }
    logger.info("Model metadata extracted.")
    return metadata
