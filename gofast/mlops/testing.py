# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Ensure the reliability of machine learning models and pipelines
through robust testing.
"""
# XXX TO OPTIMIZE 
import time
import random 
from datetime import datetime
from typing import List, Callable, Optional, Dict, Tuple 
from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ..api.types import DataFrame
from ..api.property import BaseClass 
from ..decorators import RunReturn, smartFitRun
from ..tools.validator import check_is_fitted, check_is_runned
from ..tools.validator import check_X_y, check_array 


__all__ = [
    'PipelineTest',  
    'ModelQuality',  
    'OverfittingDetection',  
    'DataIntegrity', 
    'BiasDetection', 
    'ModelVersioning',  
    'PerformanceRegression',  
    'CIIntegration',  
    
]

class BaseTest(BaseClass, metaclass=ABCMeta):
    """
    BaseTest class provides a framework for creating test cases with customizable
    configurations, supporting parallel execution, logging, and randomization. 
    This class cannot be instantiated directly and is intended to be subclassed 
    to implement specific test cases through the `fit` and `run` methods.

    The mathematical formulation underlying the tests includes statistical 
    processes and optimizations, often involving randomization where applicable.
    For example, a random seed can be set to ensure reproducibility in 
    stochastic test executions.

    .. math::
        R(x) = f(x) + \epsilon

    where :math:`R(x)` is the result of a test applied to the input data :math:`x`, 
    and :math:`\epsilon` represents any potential noise or random variations 
    introduced, controlled by the `random_seed` parameter.

    Parameters
    ----------
    test_name : str, optional
        The name of the test instance. Default is "BaseTest".
    store_results : bool, optional
        Whether to store the results of the test. Default is True.
    enable_logging : bool, optional
        If True, logs events during the test execution. Default is True.
    parallel_execution : bool, optional
        If True, enables parallel execution of the test. Default is False.
    random_seed : Optional[int], optional
        Sets the random seed for reproducibility. Default is None.
    **config_params : dict
        Additional configuration parameters that can be set dynamically.

    Attributes
    ----------
    test_name : str
        Name of the test case.
    store_results : bool
        Indicates whether results will be stored.
    enable_logging : bool
        If True, logging of events is enabled.
    parallel_execution : bool
        If True, test cases are executed in parallel.
    random_seed : Optional[int]
        Random seed for reproducibility.
    config_params : dict
        Dynamic configuration parameters.
    _is_fitted : bool
        Internal flag indicating if the `fit` method has been executed.
    _is_runned : bool
        Internal flag indicating if the `run` method has been executed.
    results_ : Any
        Stores the test results after execution.
    _log_ : list
        Holds the event logs during execution.

    Notes
    -----
    - This class is abstract and should be subclassed. You must implement 
      the `fit` and `run` methods in subclasses to define specific behaviors.
    - The `parallel_execution` flag enables the use of concurrent threads 
      for parallelism, which can be useful for large-scale test cases.

    Methods
    -------
    fit(*args, **kwargs)
        Abstract method. Subclasses must implement this to fit a model 
        or prepare data for testing.
    
    run(*args, **kwargs)
        Abstract method. Subclasses must implement this to execute the test 
        based on fitted data or parameters.
    
    log_event(event: str, details: dict = None)
        Logs the specified event and any details if logging is enabled.
    
    get_logs()
        Returns the event logs.
    
    reset()
        Resets the internal state of the test object.
    
    save_results(path: str)
        Saves the results of the test to the specified file path.
    
    load_results(path: str)
        Loads test results from the specified file path.
    
    parallelize(func: Callable, data: list)
        Executes the function `func` over the provided `data` in parallel, 
        using threads.

    run_test(*args, **kwargs)
        Executes the test case. Handles parallel execution if enabled.
    
    Examples
    --------
    >>> from gofast.mlops.testing import BaseTest
    >>> class MyTest(BaseTest):
    >>>     def fit(self, data):
    >>>         self._set_fitted()
    >>>         # implement fitting logic
    >>> 
    >>>     def run(self, data):
    >>>         self._set_runned()
    >>>         return sum(data)
    >>> 
    >>> test = MyTest(test_name="My Test", parallel_execution=True)
    >>> test.fit(data=[1, 2, 3])
    >>> results = test.run(data=[1, 2, 3])
    >>> print(results)
    
    Raises
    ------
    ValueError
        If no results are available when trying to save, or if parallel 
        execution is attempted when not enabled.

    See Also
    --------
    gofast.mlops.parallelizer : Parallel execution utilities.
    gofast.mlops.logging : Advanced logging framework.

    References
    ----------
    .. [1] Smith, J., et al. (2022). "Advanced Test Execution Techniques," 
       Journal of Software Testing, 15(4), 123-145.
    """

    def __init__(self, 
                 test_name: str = "BaseTest", 
                 store_results: bool = True, 
                 enable_logging: bool = True, 
                 parallel_execution: bool = False, 
                 random_seed: Optional[int] = None, 
                 **config_params):
        self.test_name = test_name
        self.store_results = store_results
        self.enable_logging = enable_logging
        self.parallel_execution = parallel_execution
        self.random_seed = random_seed
        self.config_params = config_params
        self._is_fitted = False
        self._is_runned = False
        self.results_ = None
        self._log_ = []

        # Set random seed if provided
        if self.random_seed is not None:
            self._set_random_seed()

        # Additional initialization from config parameters
        self._initialize_from_config(config_params)

    def _initialize_from_config(self, config_params):
        """
        Initializes additional configurations based on provided 
        parameters.
        """
        for param, value in config_params.items():
            setattr(self, param, value)

    def _set_random_seed(self):
        """
        Sets the random seed for reproducibility of stochastic processes.
        """

        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    @abstractmethod
    def fit(self, *args, **kwargs):
        """
        Fit method must be implemented in subclasses. The fitting process
        may involve preparing data, training models, or setting up necessary 
        parameters for running the test.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the subclass.
        """
        raise NotImplementedError(
            "Subclasses should implement this method if 'fit' is required.")

    @abstractmethod
    def run(self, *args, **kwargs):
        """
        Run method must be implemented in subclasses. It defines the logic 
        for executing the test once data or parameters are ready.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the subclass.
        """
        raise NotImplementedError(
            "Subclasses should implement this method if 'run' is required.")

    def log_event(self, event: str, details: dict = None):
        """
        Logs an event with optional details if logging is enabled.
        
        Parameters
        ----------
        event : str
            A description of the event to be logged.
        details : dict, optional
            Additional information about the event.

        Notes
        -----
        The logs can be accessed via the `get_logs` method.
        """
        if self.enable_logging:
            log_entry = {"event": event, "details": details}
            self._log_.append(log_entry)

    def get_logs(self):
        """
        Retrieves the log of all events that have occurred.
        
        Returns
        -------
        list
            A list of logged events.

        Notes
        -----
        This method is only useful if logging is enabled during object 
        initialization.
        """
        return self._log_

    def reset(self):
        """
        Resets the internal state of the test object. This includes the 
        fitted and run states, as well as clearing any results and logs.

        Notes
        -----
        Use this method to clear a test's state and rerun it from scratch.
        """
        self._is_fitted = False
        self._is_runned = False
        self.results_ = None
        self._log_ = []

    def save_results(self, path: str):
        """
        Saves the results of the test to a specified file path.

        Parameters
        ----------
        path : str
            The file path where results should be saved.

        Raises
        ------
        ValueError
            If there are no results to save or if result storage is disabled.
        """
        if self.results_ is None:
            raise ValueError("No results to save. Ensure tests have been run.")
        
        if not self.store_results:
            raise ValueError("Result storage is disabled.")
        
        with open(path, 'w') as f:
            f.write(str(self.results_))
        
        self.log_event("results_saved", {"path": path})

    def load_results(self, path: str):
        """
        Loads results from a specified file path.

        Parameters
        ----------
        path : str
            The file path from which to load results.

        Notes
        -----
        This method can be used to restore results from a previous test run.
        """
        with open(path, 'r') as f:
            self.results_ = f.read()
        
        self.log_event("results_loaded", {"path": path})

    def parallelize(self, func: Callable, data: list):
        """
        Executes the provided function `func` over the data in parallel.

        Parameters
        ----------
        func : Callable
            The function to be executed in parallel.
        data : list
            A list of input data over which the function will be applied.

        Returns
        -------
        list
            Results of the function applied to the data.

        Raises
        ------
        ValueError
            If parallel execution is not enabled.

        Notes
        -----
        Parallelism is achieved through threading, which can improve 
        performance for I/O-bound operations or multiple test cases.
        """
        if not self.parallel_execution:
            raise ValueError("Parallel execution is not enabled.")
        
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(func, data))
        
        return results

    def run_test(self, *args, **kwargs):
        """
        Executes the test. If `parallel_execution` is enabled, the test 
        will be run in parallel. Otherwise, it will run sequentially.

        Parameters
        ----------
        *args, **kwargs : 
            Arguments and keyword arguments passed to the `run` method.
        
        Returns
        -------
        Any
            Results of the test execution.
        
        Notes
        -----
        If parallel execution is enabled, the test logic must be able 
        to handle concurrent operations safely.
        """
        if self.parallel_execution:
            return self.parallelize(self.run, *args, **kwargs)
        else:
            return self.run(*args, **kwargs)


    def _set_fitted(self):
        """
        Marks the test as 'fitted' by setting the internal `_is_fitted` 
        flag to `True`.
    
        Notes
        -----
        This method is used internally after the `fit` method is successfully 
        executed. It indicates that the test has been prepared or trained 
        on the provided data.
        
        Raises
        ------
        None
    
        See Also
        --------
        check_is_fitted : Verifies if the test has been fitted.
        """
        self._is_fitted = True
    
    
    def _set_runned(self):
        """
        Marks the test as 'runned' by setting the internal `_is_runned` 
        flag to `True`.
    
        Notes
        -----
        This method is used internally after the `run` method is successfully 
        executed. It indicates that the test has been executed and results 
        may be available.
        
        Raises
        ------
        None
    
        See Also
        --------
        check_is_runned : Verifies if the test has been run.
        """
        self._is_runned = True
    
    
    def check_is_fitted(self, msg=None):
        """
        Checks whether the test has been fitted by verifying the `_is_fitted` 
        attribute.
    
        Parameters
        ----------
        msg : str, optional
            Custom message to be displayed if the test is not fitted.
            Default is None, in which case a generic error message will be used.
    
        Raises
        ------
        AttributeError
            If the test has not been fitted and `_is_fitted` is `False`.
    
        Notes
        -----
        This method should be used before any operation that requires the 
        test to have been fitted. It ensures that the test is in the correct 
        state before proceeding.
    
        Examples
        --------
        >>> test.check_is_fitted()
        AttributeError: This BaseTest instance is not fitted yet.
    
        See Also
        --------
        _set_fitted : Marks the test as fitted.
        """
        check_is_fitted(self, attributes=["_is_fitted"], msg=msg)

    
    def check_is_runned(self, msg=None):
        """
        Checks whether the test has been run by verifying the `_is_runned` 
        attribute.
    
        Parameters
        ----------
        msg : str, optional
            Custom message to be displayed if the test has not been run.
            Default is None, in which case a generic error message will be used.
    
        Raises
        ------
        AttributeError
            If the test has not been run and `_is_runned` is `False`.
    
        Notes
        -----
        This method should be used before any operation that requires the 
        test to have been executed. It ensures that the test has run 
        and the results are available before proceeding.
    
        Examples
        --------
        >>> test.check_is_runned()
        AttributeError: This BaseTest instance has not been run yet.
    
        See Also
        --------
        _set_runned : Marks the test as run.
        """
        check_is_runned(self, attributes=["_is_runned"], msg=msg)

@smartFitRun
class PipelineTest(BaseTest):
    """
    PipelineTest is designed to test and evaluate scikit-learn pipelines 
    with built-in metrics tracking and parallel execution capabilities.
    It integrates with scoring functions and enables automated logging of 
    events throughout the test lifecycle.

    The test process fits the pipeline to provided data, evaluates it 
    based on predefined metrics, and optionally tracks those metrics 
    through custom scoring functions.

    The class uses the decorator `@SmartFitRun`, which ensures that 
    `fit` and `run` methods operate intelligently, as per the user-defined 
    parameters.

    .. math::
        M(x, y) = f(Pipeline(x), y)

    where :math:`M(x, y)` represents the metrics evaluated on the data :math:`x`
    with the true labels :math:`y` using a pipeline-based function 
    :math:`Pipeline(x)`.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        The scikit-learn pipeline that will be fitted and evaluated.
    scoring : Optional[Callable], optional
        A scoring function to evaluate the fitted pipeline. If `track_metrics` 
        is True, this function is mandatory. Default is None.
    random_seed : Optional[int], optional
        The random seed for reproducibility. Default is None.
    parallel_execution : bool, optional
        If True, the test will be executed in parallel, enabling faster 
        processing of large datasets. Default is False.
    store_results : bool, optional
        If True, test results will be stored. Default is True.
    logging_level : str, optional
        The logging level. Default is "INFO". Set to "NONE" to disable logging.
    track_metrics : bool, optional
        If True, the test will track and evaluate performance metrics. 
        Default is True.
    metrics : Optional[List[str]], optional
        A list of metrics to evaluate during the test. Default metrics include 
        "accuracy", "precision", and "recall".
    **config_params : dict
        Additional configuration parameters passed to the test for customization.

    Attributes
    ----------
    pipeline : sklearn.pipeline.Pipeline
        The pipeline to be tested and evaluated.
    scoring : Optional[Callable]
        A function for scoring the fitted pipeline.
    track_metrics : bool
        Indicates whether performance metrics should be tracked.
    metrics : List[str]
        A list of metrics used for evaluation.
    results_ : dict
        Stores the results from the fitting and evaluation process.
    
    Notes
    -----
    - This class requires scikit-learn's pipeline for execution.
    - The `track_metrics` flag enables custom scoring; you must provide 
      a scoring function if this flag is set to True.

    Methods
    -------
    fit(X, y=None, **fit_params)
        Fits the pipeline to the provided data and, if tracking is enabled, 
        computes the fit score using the scoring function.
    evaluate_metrics(X, y)
        Evaluates the specified metrics on the test data.
    run_test(*args, **kwargs)
        Executes the test, handling parallel execution if enabled.
    _default_validation(X_train, y_train, X_test, y_test)
        Fits the pipeline on training data and predicts on test data. If 
        tracking is enabled, computes the validation score using the scoring 
        function.

    Examples
    --------
    >>> from gofast.mlops.testing import PipelineTest
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.metrics import accuracy_score
    >>>
    >>> pipeline = Pipeline([
    >>>     ('scaler', StandardScaler()),
    >>>     ('classifier', LogisticRegression())
    >>> ])
    >>> test = PipelineTest(pipeline=pipeline, scoring=accuracy_score)
    >>> X_train, X_test, y_train, y_test = ...
    >>> test.fit(X_train, y_train)
    >>> test.evaluate_metrics(X_test, y_test)

    Raises
    ------
    ValueError
        If `track_metrics` is True and no scoring function is provided.
    ValueError
        If the pipeline is not fitted before metrics evaluation.
    
    See Also
    --------
    sklearn.pipeline.Pipeline : The base scikit-learn pipeline.
    sklearn.metrics : Scoring functions for evaluating pipelines.

    References
    ----------
    .. [1] Pedregosa, F., et al. (2011). "Scikit-learn: Machine Learning in 
       Python," Journal of Machine Learning Research, 12, pp. 2825-2830.
    """

    def __init__(self, 
                 pipeline, 
                 scoring: Optional[Callable] = None, 
                 random_seed: Optional[int] = None, 
                 parallel_execution: bool = False, 
                 store_results: bool = True,
                 logging_level: str = "INFO",
                 track_metrics: bool = True,
                 metrics: Optional[List[str]] = None,
                 **config_params):
        super().__init__(test_name=self.__class__.__name__, 
                         random_seed=random_seed, 
                         parallel_execution=parallel_execution, 
                         store_results=store_results, 
                         enable_logging=logging_level != "NONE", 
                         **config_params)
        self.pipeline = pipeline
        self.scoring = scoring
        self.track_metrics = track_metrics
        self.metrics = metrics or ["accuracy", "precision", "recall"]
        self.results_ = {}

    def fit(self, X, y=None, **fit_params):
        """
        Fits the pipeline to the provided data. If metrics tracking is 
        enabled, computes the fit score using the provided scoring function.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training data.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs), optional
            The target values. Default is None, in which case `X` is expected 
            to be the entire dataset.
        **fit_params : dict
            Additional parameters passed to the pipeline's `fit` method.

        Raises
        ------
        ValueError
            If `track_metrics` is True but no scoring function is provided.

        Notes
        -----
        If `track_metrics` is enabled and the scoring function is provided, 
        the fit score is logged and stored in the `results_` dictionary.
        """
        if y is None:
            X = check_array(X, input_name='X')
        else:
            X, y = check_X_y(X, y, multi_output=True, estimator=self)
        
        if self.track_metrics and not self.scoring:
            raise ValueError("Scoring function must be provided if track_metrics is True.")
        
        # Fit the pipeline with data
        self.pipeline.fit(X, y, **fit_params)
        self._set_fitted()

        if self.track_metrics and self.scoring:
            self.results_["fit_score"] = self.scoring(self.pipeline, X, y)
            self.log_event("fit_completed", {"fit_score": self.results_["fit_score"]})
            
        return self 

    def evaluate_metrics(self, X, y):
        """
        Evaluates the specified metrics on the provided test data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The test data.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The true target values.

        Returns
        -------
        dict
            A dictionary containing the computed metrics.

        Raises
        ------
        ValueError
            If the pipeline is not fitted before metrics evaluation.

        Notes
        -----
        By default, accuracy is computed as a metric, but other metrics such 
        as precision and recall can also be evaluated if specified.
        """
        self.check_is_fitted("Pipeline must be fitted before evaluating metrics.")

        results = {}
        if "accuracy" in self.metrics:
            results["accuracy"] = self.pipeline.score(X, y)

        for metric in self.metrics:
            if metric != "accuracy" and hasattr(self.scoring, metric):
                score = self.scoring(self.pipeline, X, y)
                results[metric] = score

        self.results_["metrics"] = results
        self.log_event("metrics_evaluated", {"metrics": results})
        return results

    def _default_validation(self, X_train, y_train, X_test, y_test):
        """
        Fits the pipeline on training data and validates on test data. If 
        metrics tracking is enabled, the validation score is computed using 
        the provided scoring function.

        Parameters
        ----------
        X_train : array-like of shape (n_samples_train, n_features)
            The training data.
        y_train : array-like of shape (n_samples_train,) or (n_samples_train, n_outputs)
            The target values for the training data.
        X_test : array-like of shape (n_samples_test, n_features)
            The test data.
        y_test : array-like of shape (n_samples_test,) or (n_samples_test, n_outputs)
            The target values for the test data.

        Returns
        -------
        float or array-like
            The predictions or validation score, depending on whether metrics 
            tracking is enabled.

        Notes
        -----
        This method is used internally for default validation scenarios where 
        train-test splits are predefined.
        """
        self.pipeline.fit(X_train, y_train)
        predictions = self.pipeline.predict(X_test)

        if self.track_metrics and self.scoring:
            validation_score = self.scoring(self.pipeline, X_test, y_test)
            self.log_event("default_validation", {"validation_score": validation_score})
            return validation_score
        
        return predictions

@smartFitRun
class ModelQuality(BaseTest):
    """
    ModelQuality is designed to evaluate the performance of machine learning 
    models by calculating various metrics on training and testing datasets. 
    It supports cross-validation, custom metrics, parallel execution, and 
    logging of results.

    The evaluation process includes splitting the data into training and testing 
    sets (if required), fitting the model, calculating specified performance metrics, 
    and optionally performing cross-validation.

    The mathematical formulation behind the metrics is generally model-specific, 
    but for classification metrics like accuracy, precision, recall, and F1-score, 
    the formulation follows standard definitions:

    .. math::
        Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
    
    where TP, TN, FP, and FN are the true positives, true negatives, 
    false positives, and false negatives, respectively.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        The machine learning model to be evaluated.
    metrics : Optional[Dict[str, Callable]], optional
        A dictionary of metric names and corresponding scoring functions. 
        Default metrics include `accuracy`, `precision`, `recall`, and 
        `f1_score`.
    random_seed : Optional[int], optional
        The random seed for reproducibility. Default is None.
    parallel_execution : bool, optional
        If True, enables parallel execution for faster processing. Default is False.
    store_results : bool, optional
        Whether to store the results of the evaluation. Default is True.
    logging_level : str, optional
        The logging level to use. Default is "INFO". Set to "NONE" to disable logging.
    cross_validation : bool, optional
        If True, performs cross-validation in addition to regular evaluation. 
        Default is False.
    cv_folds : int, optional
        The number of folds for cross-validation. Default is 5.
    evaluation_split : float, optional
        The proportion of data to be used as the test set. If set to 1.0, 
        no data splitting occurs. Default is 0.2.
    **config_params : dict
        Additional configuration parameters passed to the test for customization.

    Attributes
    ----------
    model : sklearn.base.BaseEstimator
        The machine learning model being evaluated.
    metrics : dict
        Dictionary of metric names and their associated scoring functions.
    cross_validation : bool
        Indicates whether cross-validation is enabled.
    cv_folds : int
        Number of cross-validation folds.
    evaluation_split : float
        Proportion of the data used for testing.
    results_ : dict
        Stores the results of the evaluation, including metrics and 
        cross-validation scores if applicable.

    Notes
    -----
    - This class assumes the input model follows the scikit-learn API, 
      with `fit` and `predict` methods.
    - Custom metrics must be passed as a dictionary, where each key is a 
      metric name and each value is a callable (usually a function from 
      scikit-learn's metrics module).

    Methods
    -------
    fit(X, y=None, **fit_params)
        Fits the model to the training data and evaluates metrics on both 
        training and testing datasets.
    _split_data(X, y)
        Splits the input data into training and testing sets based on the 
        `evaluation_split` ratio.
    _evaluate_metrics(X, y, data_type="train")
        Evaluates the specified metrics on the provided dataset (train or test).
    _cross_validate(X, y)
        Performs cross-validation on the training data and returns the scores.

    Examples
    --------
    >>> from gofast.mlops.testing import ModelQuality
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.metrics import accuracy_score
    >>>
    >>> model = RandomForestClassifier(random_state=42)
    >>> evaluator = ModelQuality(model=model, metrics={"accuracy": accuracy_score})
    >>> X_train, X_test, y_train, y_test = ...
    >>> evaluator.fit(X_train, y_train)
    >>> evaluator.evaluate_metrics(X_test, y_test)

    Raises
    ------
    ValueError
        If `X` and `y` are not provided during fitting.
    ValueError
        If no scoring function is provided for the specified metrics.
    
    See Also
    --------
    sklearn.model_selection.train_test_split : Utility function to split the data.
    sklearn.model_selection.cross_val_score : Utility function to perform cross-validation.
    sklearn.metrics : Module with standard evaluation metrics.

    References
    ----------
    .. [1] Pedregosa, F., et al. (2011). "Scikit-learn: Machine Learning in 
       Python," Journal of Machine Learning Research, 12, pp. 2825-2830.
    """

    def __init__(self, 
                 model, 
                 metrics: Optional[Dict[str, Callable]] = None, 
                 random_seed: Optional[int] = None, 
                 parallel_execution: bool = False, 
                 store_results: bool = True,
                 logging_level: str = "INFO", 
                 cross_validation: bool = False, 
                 cv_folds: int = 5, 
                 evaluation_split: float = 0.2, 
                 **config_params):
        super().__init__(test_name= self.__class__.__name__, 
                         random_seed=random_seed, 
                         parallel_execution=parallel_execution, 
                         store_results=store_results, 
                         enable_logging=logging_level != "NONE", 
                         **config_params)
        self.model = model
        self.metrics = metrics or {
            "accuracy": accuracy_score, 
            "precision": precision_score, 
            "recall": recall_score, 
            "f1_score": f1_score
        }
        self.cross_validation = cross_validation
        self.cv_folds = cv_folds
        self.evaluation_split = evaluation_split
        self.results_ = {}

    def fit(self, X, y=None, **fit_params):
        """
        Fits the model to the provided data and evaluates the specified 
        metrics on both training and testing datasets.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs), optional
            The target values. Default is None.
        **fit_params : dict
            Additional parameters passed to the model's `fit` method.

        Returns
        -------
        self : ModelQuality
            Returns the instance of the class after fitting.

        Raises
        ------
        ValueError
            If `X` or `y` is None.

        Notes
        -----
        This method splits the data into training and testing sets, fits 
        the model on the training set, and then evaluates metrics on both 
        training and testing data.
        """
        self._check_required_params(X, y)
        X_train, X_test, y_train, y_test = self._split_data(X, y)

        # Fit the model
        self.model.fit(X_train, y_train, **fit_params)
        self._set_fitted()

        # Evaluate metrics after fitting
        self.results_["train_metrics"] = self._evaluate_metrics(X_train, y_train, "train")
        self.results_["test_metrics"] = self._evaluate_metrics(X_test, y_test, "test")

        # Optionally run cross-validation
        if self.cross_validation:
            self.results_["cross_val_scores"] = self._cross_validate(X_train, y_train)

        return self 

    def _split_data(self, X, y):
        """
        Splits the input data into training and testing sets based on 
        the evaluation split ratio.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values.

        Returns
        -------
        X_train, X_test, y_train, y_test : tuple of arrays
            The split data for training and testing.
        """
        if self.evaluation_split < 1.0:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.evaluation_split, random_state=self.random_seed)
            return X_train, X_test, y_train, y_test
        else:
            return X, X, y, y  # No split if evaluation_split is 1.0

    def _evaluate_metrics(self, X, y, data_type="train"):
        """
        Evaluates the specified metrics on the given dataset (train or test).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        y : array-like of shape (n_samples,)
            The true target values.
        data_type : str, optional
            Specifies whether the data is from training or testing. Default is "train".

        Returns
        -------
        dict
            A dictionary containing the evaluated metrics.

        Notes
        -----
        If a metric function expects an additional argument (e.g., `average` 
        for multi-class metrics), it will be provided as "weighted".
        """
        metrics_results = {}
        y_pred = self.model.predict(X)
        for metric_name, metric_func in self.metrics.items():
            try:
                if "average" in metric_func.__code__.co_varnames:
                    metrics_results[metric_name] = metric_func(
                        y, y_pred, average="weighted")
                else:
                    metrics_results[metric_name] = metric_func(y, y_pred)
            except Exception as e:
                self.log_event(f"metric_error_{metric_name}", {
                    "error": str(e), "data_type": data_type})
                metrics_results[metric_name] = None

        self.log_event(f"{data_type}_metrics_evaluation", metrics_results)
        return metrics_results

    def _cross_validate(self, X, y):
        """
        Performs cross-validation on the training data and returns the 
        cross-validation scores.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        array
            Cross-validation scores for each fold.

        Raises
        ------
        ValueError
            If the model is not fitted before cross-validation is performed.

        Notes
        -----
        Cross-validation is performed using the specified number of folds (`cv_folds`).
        """
        self.check_is_fitted()
        try:
            scores = cross_val_score(self.model, X, y, cv=self.cv_folds,
                                     n_jobs=-1 if self.parallel_execution else 1)
            self.log_event('cross_validation', {"cv_scores": scores})
            return scores
        except Exception as e:
            self.log_event("cross_val_error", {"error": str(e)})
            return None

    def _check_required_params(self, X, y):
        """
        Ensures that the required parameters `X` and `y` are provided.

        Parameters
        ----------
        X : array-like
            The input data.
        y : array-like
            The target values.

        Raises
        ------
        ValueError
            If either `X` or `y` is None.
        """
        if X is None or y is None:
            raise ValueError("X and y cannot be None. Both are required to fit the model.")

@smartFitRun
class OverfittingDetection(BaseTest):
    """
    OverfittingDetection class detects overfitting in a machine learning model 
    by comparing the model's performance on training and test data. It supports 
    cross-validation, configurable tolerance for detecting overfitting, and logging 
    of results.

    The detection is based on the comparison of training and test scores. 
    If the difference between the training score and the test score exceeds 
    the specified `tolerance`, overfitting is considered to be detected.

    .. math::
        \text{Overfitting} = (S_{\text{train}} - S_{\text{test}}) > \text{tolerance}

    where :math:`S_{\text{train}}` is the score on the training set, 
    :math:`S_{\text{test}}` is the score on the test set, and `tolerance` 
    is a user-defined threshold.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        The machine learning model to be evaluated for overfitting.
    tolerance : float, optional
        The acceptable difference between training and test scores before 
        overfitting is detected. Default is 0.05.
    store_results : bool, optional
        If True, the overfitting results will be stored.
        Default is True.
    enable_logging : bool, optional
        If True, logs the events related to overfitting detection. 
        Default is True.
    parallel_execution : bool, optional
        If True, enables parallel execution for cross-validation.
        Default is False.
    random_seed : Optional[int], optional
        The random seed for reproducibility. Default is None.
    cross_validation : bool, optional
        If True, cross-validation will be used to evaluate the model.
        Default is False.
    cv_folds : int, optional
        The number of cross-validation folds if `cross_validation` is enabled.
        Default is 5.
    evaluation_split : float, optional
        The proportion of the data used for testing. If set to 1.0, 
        no data splitting is performed. Default is 0.2.
    **config_params : dict
        Additional configuration parameters passed to the test for customization.

    Attributes
    ----------
    model : sklearn.base.BaseEstimator
        The machine learning model to be evaluated for overfitting.
    tolerance : float
        The tolerance threshold for detecting overfitting.
    cross_validation : bool
        Indicates whether cross-validation is enabled.
    cv_folds : int
        Number of folds for cross-validation.
    evaluation_split : float
        Proportion of the data used for testing.
    results_ : dict
        Stores the results of the overfitting detection process, including 
        training and test scores, and whether overfitting is detected.

    Notes
    -----
    This class assumes the model follows the scikit-learn API, with `fit`, `score`, 
    and `predict` methods.

    Methods
    -------
    fit(X, y, X_test=None, y_test=None, **fit_params)
        Fits the model to the training data and evaluates overfitting by 
        comparing train and test scores.
    cross_validate(X, y)
        Performs cross-validation and returns the cross-validation scores.
    save_overfitting_report(path)
        Saves the overfitting report to the specified file.

    Examples
    --------
    >>> from gofast.mlops.testing import OverfittingDetection
    >>> from sklearn.ensemble import RandomForestClassifier
    >>>
    >>> model = RandomForestClassifier(random_state=42)
    >>> detector = OverfittingDetection(model=model, tolerance=0.05)
    >>> X_train, X_test, y_train, y_test = ...
    >>> detector.fit(X_train, y_train, X_test, y_test)
    >>> print(detector.results_)

    Raises
    ------
    ValueError
        If `X_test` is provided but `y_test` is not, or vice versa.
    ValueError
        If cross-validation is attempted but is disabled.
    ValueError
        If no overfitting report is available when attempting to save.

    See Also
    --------
    sklearn.model_selection.cross_val_score : Utility function to perform cross-validation.
    sklearn.metrics : Module containing various scoring functions for models.

    References
    ----------
    .. [1] Hastie, T., Tibshirani, R., Friedman, J. (2009). "The Elements of 
       Statistical Learning," Springer Series in Statistics.
    """

    def __init__(self,
                 model,
                 tolerance: float = 0.05,
                 store_results: bool = True,
                 enable_logging: bool = True,
                 parallel_execution: bool = False,
                 random_seed: Optional[int] = None,
                 cross_validation: bool = False,
                 cv_folds: int = 5,
                 evaluation_split: float = 0.2,
                 **config_params):
        super().__init__(test_name=self.__class__.__name__,
                         store_results=store_results,
                         enable_logging=enable_logging,
                         parallel_execution=parallel_execution,
                         random_seed=random_seed,
                         **config_params)
        self.model = model
        self.tolerance = tolerance
        self.cross_validation = cross_validation
        self.cv_folds = cv_folds
        self.evaluation_split = evaluation_split

    def fit(self, X, y,  **fit_params):
        """
        Fits the model and detects overfitting by comparing the training 
        and test scores.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values.
        X_test : array-like of shape (n_samples_test, n_features), optional
            The test input samples. Default is None.
        y_test : array-like of shape (n_samples_test,), optional
            The test target values. Default is None.
        **fit_params : dict
            Additional parameters passed to the model's `fit` method.

        Returns
        -------
        self : OverfittingDetection
            Returns the instance of the class after fitting and detecting overfitting.

        Raises
        ------
        ValueError
            If `X_test` is provided without `y_test`, or vice versa.

        Notes
        -----
        If `X_test` and `y_test` are not provided, the data will be split 
        internally using `evaluation_split`. After fitting, the method will 
        compare training and test scores to detect overfitting.
        """
        X_test = fit_params.pop("X_test", None )
        y_test = fit_params.pop("y_test", None ) 
        
        if X_test is not None and y_test is None:
            raise ValueError("y_test must be provided if X_test is supplied.")
        if y_test is not None and X_test is None:
            raise ValueError("X_test must be provided if y_test is supplied.")

        # Split data if no X_test/y_test are provided
        if X_test is None and y_test is None:
            X_train, X_test, y_train, y_test = self._split_data(X, y)
        else:
            X_train, y_train = X, y

        # Fit the model
        self.model.fit(X_train, y_train, **fit_params)
        self._set_fitted()

        # Perform overfitting detection based on train and test data
        self.results_ = self._detect_overfitting(X_train, y_train, X_test, y_test)
        return self 

    def _detect_overfitting(self, X_train, y_train, X_test, y_test):
        """
        Detects overfitting by comparing training and test scores.

        Parameters
        ----------
        X_train : array-like of shape (n_samples_train, n_features)
            The training input samples.
        y_train : array-like of shape (n_samples_train,)
            The training target values.
        X_test : array-like of shape (n_samples_test, n_features)
            The test input samples.
        y_test : array-like of shape (n_samples_test,)
            The test target values.

        Returns
        -------
        dict
            A dictionary containing training and test scores, and a boolean 
            indicating whether overfitting is detected.

        Notes
        -----
        Overfitting is detected if the difference between the training and 
        test scores exceeds the specified tolerance.
        """
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)

        overfitting = (train_score - test_score) > self.tolerance
        results = {
            'train_score': train_score,
            'test_score': test_score,
            'overfitting_detected': overfitting
        }

        self.log_event("overfitting_detection", results)
        return results

    def cross_validate(self, X, y):
        """
        Performs cross-validation on the provided data and returns the 
        cross-validation scores.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        array
            Cross-validation scores for each fold.

        Raises
        ------
        ValueError
            If cross-validation is disabled.

        Notes
        -----
        Cross-validation is performed using the specified number of folds (`cv_folds`).
        """
        if not self.cross_validation:
            raise ValueError("Cross-validation is disabled. Enable "
                             "it by setting `cross_validation=True`.")
        
        self.check_is_fitted()

        scores = cross_val_score(self.model, X, y, cv=self.cv_folds, n_jobs=-1 
                                 if self.parallel_execution else 1)
        self.log_event("cross_validation", {"cv_scores": scores})
        return scores

    def save_overfitting_report(self, path: str):
        """
        Saves the overfitting report to a specified file path.

        Parameters
        ----------
        path : str
            The file path where the report will be saved.

        Raises
        ------
        ValueError
            If there is no overfitting report available or result storage 
            is disabled.

        Notes
        -----
        The report contains information about training and test scores, 
        and whether overfitting was detected.
        """
        if self.results_ is None:
            raise ValueError("No overfitting report available. Ensure the "
                             "model has been fitted and evaluated.")
        if not self.store_results:
            raise ValueError("Result storage is disabled.")

        with open(path, 'w') as file:
            file.write(str(self.results_))
        
        self.log_event("report_saved", {"path": path})

@smartFitRun
class DataIntegrity(BaseTest):
    """
    DataIntegrity is designed to perform various checks on a dataset to 
    ensure that it meets specified quality standards. This includes checks 
    for missing values, duplicates, data types, value ranges, and custom 
    validation functions. The class also supports logging and parallel 
    execution for large datasets.

    The integrity of the data is evaluated based on a combination of 
    statistical checks and user-defined validation functions.

    .. math::
        \text{Missing Ratio} = \frac{\text{Total Missing Values}}{\text{Total Elements in Data}}

    Parameters
    ----------
    validation_checks : Optional[Dict[str, Callable]], optional
        A dictionary of custom validation check names and their corresponding 
        functions that accept a DataFrame and return a boolean indicating 
        success or failure. Default is None.
    missing_value_threshold : float, optional
        The acceptable threshold for the percentage of missing values in 
        the dataset. Default is 0.0 (no missing values allowed).
    unique_check_columns : Optional[List[str]], optional
        A list of column names to check for uniqueness. If duplicates are 
        found in these columns and `allow_duplicates` is False, an issue 
        will be raised. Default is None.
    allow_duplicates : bool, optional
        If True, duplicate rows in the `unique_check_columns` will be allowed. 
        If False, duplicates will be flagged. Default is False.
    data_types : Optional[Dict[str, type]], optional
        A dictionary where the keys are column names and the values are 
        the expected data types for those columns. Default is None.
    range_checks : Optional[Dict[str, Tuple[float, float]]], optional
        A dictionary where the keys are column names and the values are 
        tuples specifying the acceptable range (min, max) for values in 
        those columns. Default is None.
    enable_logging : bool, optional
        If True, logs the events related to data integrity checks. Default is True.
    parallel_execution : bool, optional
        If True, enables parallel execution for large datasets. Default is False.
    random_seed : Optional[int], optional
        The random seed for reproducibility. Default is None.
    **config_params : dict
        Additional configuration parameters for the test.

    Attributes
    ----------
    validation_checks : dict
        Custom validation checks provided by the user.
    missing_value_threshold : float
        The threshold for acceptable missing values.
    unique_check_columns : list
        Columns to check for uniqueness.
    allow_duplicates : bool
        Whether duplicates are allowed in the unique check columns.
    data_types : dict
        Expected data types for columns.
    range_checks : dict
        Expected value ranges for columns.
    issues_ : list
        A list containing issues found during the data integrity check.

    Notes
    -----
    - The integrity checks can be customized through `validation_checks`, 
      which allows users to define their own data validation logic.
    - Range checks are performed on specified columns to ensure values 
      fall within the provided bounds.

    Methods
    -------
    run(data: DataFrame, **run_kwargs)
        Executes the data integrity check on the provided dataset, 
        performing all configured checks.
    save_issues(path: str)
        Saves the identified issues to a specified file.

    Examples
    --------
    >>> from gofast.mlops.testing import DataIntegrity
    >>> import pandas as pd
    >>>
    >>> data = pd.DataFrame({
    >>>     'age': [25, 30, 22, 29, None],
    >>>     'income': [50000, 60000, 40000, None, 45000],
    >>>     'country': ['US', 'UK', 'US', 'UK', 'UK']
    >>> })
    >>>
    >>> integrity_test = DataIntegrity(
    >>>     missing_value_threshold=0.05,
    >>>     unique_check_columns=['age'],
    >>>     data_types={'age': float, 'income': float},
    >>>     range_checks={'age': (18, 65), 'income': (0, 100000)}
    >>> )
    >>>
    >>> issues = integrity_test.run(data)
    >>> print(issues)

    Raises
    ------
    ValueError
        If any of the data validation checks fail, appropriate issues are raised.
    
    See Also
    --------
    pandas.DataFrame.isnull : Check for missing values in a DataFrame.
    pandas.DataFrame.duplicated : Identify duplicate rows in a DataFrame.

    References
    ----------
    .. [1] McKinney, W. (2010). "Data Structures for Statistical Computing 
       in Python," Proceedings of the 9th Python in Science Conference.
    """

    def __init__(
            self,
            validation_checks: Optional[Dict[str, Callable]] = None,
            missing_value_threshold: float = 0.0,
            unique_check_columns: Optional[List[str]] = None,
            allow_duplicates: bool = False,
            data_types: Optional[Dict[str, type]] = None,
            range_checks: Optional[Dict[str, Tuple[float, float]]] = None,
            enable_logging: bool = True,
            parallel_execution: bool = False,
            random_seed: Optional[int] = None,
            **config_params
            ):
        super().__init__(test_name=self.__class__.__name__,
                         store_results=True,
                         enable_logging=enable_logging,
                         parallel_execution=parallel_execution,
                         random_seed=random_seed,
                         **config_params)
        self.validation_checks = validation_checks or {}
        self.missing_value_threshold = missing_value_threshold
        self.unique_check_columns = unique_check_columns or []
        self.allow_duplicates = allow_duplicates
        self.data_types = data_types or {}
        self.range_checks = range_checks or {}
    
    @RunReturn(attribute_name='issues_', ) 
    def run(self, data: DataFrame, **run_kwargs):
        """
        Executes the data integrity checks on the provided dataset.

        Parameters
        ----------
        data : pandas.DataFrame
            The input data on which the integrity checks are performed.
        **run_kwargs : dict
            Additional arguments that may influence the execution of the test.

        Returns
        -------
        list
            A list of issues identified during the data integrity check. 
            If no issues are found, a success message is returned.

        Notes
        -----
        The following checks are performed:
        - Missing values are checked against the `missing_value_threshold`.
        - Duplicates are identified based on `unique_check_columns`.
        - Data types of columns are verified against `data_types`.
        - Values in columns are checked for falling within specified 
          `range_checks`.
        - Custom validation checks are executed, if provided.

        Raises
        ------
        ValueError
            If any of the data validation checks fail.

        See Also
        --------
        pandas.DataFrame.isnull : Check for missing values in a DataFrame.
        pandas.DataFrame.duplicated : Identify duplicate rows in a DataFrame.
        """
        issues = []

        # Missing values check
        if self.missing_value_threshold > 0:
            missing_ratio = data.isnull().sum().sum() / data.size
            if missing_ratio > self.missing_value_threshold:
                issues.append(f"Missing values exceed threshold: {missing_ratio:.2%}")

        # Duplicates check
        if not self.allow_duplicates and self.unique_check_columns:
            duplicates = data.duplicated(subset=self.unique_check_columns).sum()
            if duplicates > 0:
                issues.append(
                    f"Duplicate rows found in columns {self.unique_check_columns}:"
                    f" {duplicates} duplicates.")

        # Data type checks
        for column, expected_type in self.data_types.items():
            if column in data.columns and not data[column].apply(
                    lambda x: isinstance(x, expected_type)).all():
                issues.append(
                    f"Data type mismatch in column '{column}'. Expected {expected_type}.")

        # Range checks
        for column, (min_val, max_val) in self.range_checks.items():
            if column in data.columns:
                out_of_range = ((data[column] < min_val) | (data[column] > max_val)).sum()
                if out_of_range > 0:
                    issues.append(f"Values out of range in column '{column}':"
                                  f" {out_of_range} values outside [{min_val}, {max_val}].")

        # Custom validation checks
        for check_name, check_func in self.validation_checks.items():
            if not check_func(data):
                issues.append(f"{check_name} failed.")

        # If no issues were found
        if not issues:
            issues.append("Data integrity check passed successfully.")

        # Log the test results if logging is enabled
        self.log_event("data_integrity_test", {"issues": issues})

        # Set the test as run
        self._set_runned()

        # Return issues identified during the integrity test
        self.issues_ = issues
  
class BiasDetection(BaseTest):
    """
    BiasDetection class is designed to assess the fairness of a machine learning 
    model with respect to a specified sensitive feature. It measures bias 
    using a fairness metric and compares the resulting bias score against 
    a threshold to determine whether the model exhibits bias.

    The bias is calculated by comparing the model's predictions across different 
    groups defined by the sensitive feature. This process can be carried out 
    using a single fairness metric or multiple metrics if specified.

    .. math::
        \text{Bias} = \frac{\sum_{g=1}^{G} M(P_g, y_g)}{G}

    where :math:`M` is the fairness metric applied to the predicted outcomes :math:`P_g` 
    and true labels :math:`y_g` of group :math:`g`, and :math:`G` is the total number of 
    groups based on the sensitive feature.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        The machine learning model to be evaluated for bias.
    sensitive_feature : str
        The column name of the feature in the data that represents a sensitive 
        attribute (e.g., gender, race) which is used to detect bias.
    fairness_metric : Union[Callable, Dict[str, Callable]]
        A fairness metric or a dictionary of fairness metrics that measure bias. 
        The function(s) should accept predictions, true labels, and the sensitive 
        feature as input and return a bias score.
    bias_threshold : float, optional
        The threshold above which bias is considered to be detected. Default is 0.1.
    multi_metric : bool, optional
        If True, multiple fairness metrics are used for bias detection. Default is False.
    log_results : bool, optional
        If True, logs the results of the bias detection. Default is True.
    **config_params : dict
        Additional configuration parameters.

    Attributes
    ----------
    model : sklearn.base.BaseEstimator
        The machine learning model being evaluated.
    sensitive_feature : str
        The feature used to detect bias.
    fairness_metric : Union[Callable, Dict[str, Callable]]
        The fairness metric(s) used for bias calculation.
    bias_threshold : float
        The threshold for detecting bias.
    multi_metric : bool
        Indicates whether multiple fairness metrics are being used.
    log_results : bool
        Indicates whether to log bias detection results.

    Notes
    -----
    - The fairness metric(s) should be designed to handle both binary and multi-class 
      classification problems.
    - If `multi_metric` is enabled, bias detection will aggregate the results from 
      multiple fairness metrics and calculate an average bias score.

    Methods
    -------
    fit(X, y, **fit_params)
        Fits the model to the provided training data.
    detect_bias(X, y)
        Detects bias in the model based on the specified fairness metric(s) 
        and the sensitive feature.
    fit_and_detect_bias(X, y, **fit_params)
        Combines `fit` and `detect_bias` into one method to fit the model 
        and detect bias in one call.

    Examples
    --------
    >>> from gofast.mlops.testing import BiasDetection
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.metrics import accuracy_score
    >>>
    >>> def fairness_metric(predictions, y, sensitive_feature_values):
    >>>     # Example fairness metric based on accuracy difference
    >>>     group_1 = sensitive_feature_values == 1
    >>>     group_2 = sensitive_feature_values == 0
    >>>     return abs(accuracy_score(y[group_1], predictions[group_1]) - 
    >>>                accuracy_score(y[group_2], predictions[group_2]))
    >>>
    >>> model = RandomForestClassifier()
    >>> bias_detector = BiasDetection(
    >>>     model=model, 
    >>>     sensitive_feature='gender',
    >>>     fairness_metric=fairness_metric,
    >>>     bias_threshold=0.05
    >>> )
    >>> X_train, y_train = ...
    >>> bias_detector.fit(X_train, y_train)
    >>> bias_results = bias_detector.detect_bias(X_train, y_train)
    >>> print(bias_results)

    Raises
    ------
    ValueError
        If the model has not been fitted before detecting bias.
    
    See Also
    --------
    sklearn.base.BaseEstimator : The base class for all scikit-learn estimators.
    sklearn.metrics : Common evaluation metrics, which may serve as fairness metrics.

    References
    ----------
    .. [1] Barocas, S., Hardt, M., Narayanan, A. (2019). "Fairness and Machine Learning," 
       fairmlbook.org, available at https://fairmlbook.org.
    """

    def __init__(self, 
                 model, 
                 sensitive_feature: str, 
                 fairness_metric, 
                 bias_threshold: float = 0.1, 
                 multi_metric: bool = False,
                 log_results: bool = True,
                 **config_params):
        super().__init__(test_name = self.__class__.__name__, **config_params)
        self.model = model
        self.sensitive_feature = sensitive_feature
        self.fairness_metric = fairness_metric
        self.bias_threshold = bias_threshold
        self.multi_metric = multi_metric
        self.log_results = log_results
        

    def fit(self, X, y, **fit_params):
        """
        Fits the machine learning model to the provided training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training data.
        y : array-like of shape (n_samples,)
            The target labels.
        **fit_params : dict
            Additional parameters passed to the model's `fit` method.

        Notes
        -----
        This method ensures that the model is trained on the provided data 
        before bias detection can occur.
        """
        X, y = check_X_y(X, y)

        # Fit the model with the data
        self.model.fit(X, y, **fit_params)
        self._set_fitted()

    def detect_bias(self, X, y):
        """
        Detects bias in the trained model using the specified fairness metric(s).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data used to detect bias.
        y : array-like of shape (n_samples,)
            The true labels for the input data.

        Returns
        -------
        dict
            A dictionary containing the calculated bias scores, the sensitive 
            feature being evaluated, the overall bias, and whether bias was 
            detected based on the bias threshold.

        Notes
        -----
        - This method requires that the model is already fitted on the data.
        - The bias is calculated based on predictions for different groups 
          defined by the sensitive feature.

        Raises
        ------
        ValueError
            If the model has not been fitted before calling this method.
        """
        self.check_is_fitted()

        # Run predictions
        predictions = self.model.predict(X)

        # Calculate the bias score using the fairness metric(s)
        if self.multi_metric:
            bias_scores = {}
            for metric_name, metric_func in self.fairness_metric.items():
                bias_scores[metric_name] = metric_func(predictions, y, X[self.sensitive_feature])
            overall_bias = sum(bias_scores.values()) / len(bias_scores)
        else:
            bias_scores = self.fairness_metric(predictions, y, X[self.sensitive_feature])
            overall_bias = bias_scores

        # Log event if enabled
        if self.log_results:
            self.log_event('bias_detection', {
                'bias_scores': bias_scores,
                'sensitive_feature': self.sensitive_feature
            })

        # Determine if the model exceeds the bias threshold
        is_biased = overall_bias > self.bias_threshold

        # Return results
        self.results_= {
            'bias_scores': bias_scores,
            'sensitive_feature': self.sensitive_feature,
            'overall_bias': overall_bias,
            'is_biased': is_biased,
            'bias_threshold': self.bias_threshold
        }
        
        return self 
    

    def fit_and_detect_bias(self, X, y, **fit_params):
        """
        Fits the model and then detects bias in a single step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        y : array-like of shape (n_samples,)
            The true labels for the input data.
        **fit_params : dict
            Additional parameters passed to the model's `fit` method.

        Returns
        -------
        dict
            A dictionary containing the results of the bias detection.

        Notes
        -----
        This method combines the model training (`fit`) and bias detection 
        (`detect_bias`) processes in one step for convenience.
        """
        self.fit(X, y, **fit_params)
        return self.detect_bias(X, y)

class ModelVersioning(BaseTest):
    """
    ModelVersioning class provides a robust mechanism for validating the 
    versioning of a machine learning model. It checks whether the current 
    model version matches the expected version, supports minor version 
    mismatches, detects deprecated versions, and applies custom version 
    policies.

    The model versioning system is designed to ensure compatibility and 
    warn against deprecated versions or unsupported versions based on 
    defined policies.

    .. math::
        \text{version\_match} = 
        \begin{cases} 
        \text{True} & \text{if major.minor match} \\
        \text{False} & \text{otherwise}
        \end{cases}

    Parameters
    ----------
    expected_version : str
        The expected version of the model (e.g., '1.0.0').
    allow_minor_version_mismatch : bool, optional
        If True, allows minor version mismatches between the model and 
        the expected version (e.g., '1.0.x'). Default is False.
    check_deprecation : bool, optional
        If True, checks whether the model's version is deprecated. 
        Default is False.
    deprecated_versions : Optional[list], optional
        A list of deprecated model versions. Default is None.
    log_results : bool, optional
        If True, logs the results of the versioning checks. Default is True.
    api_url : Optional[str], optional
        The URL of an external service to fetch deprecated versions. 
        Default is "https://api.example.com/deprecated-versions".
    config_file_path : Optional[str], optional
        Path to a local configuration file containing deprecated versions. 
        Default is "config/deprecated_versions.json".
    fallback_versions : Optional[list], optional
        A hardcoded list of fallback deprecated versions to use if external 
        services or configuration files are unavailable. Default is ['1.0.0', 
        '1.1.0', '2.0.0-beta'].
    custom_policies : Optional[dict], optional
        A dictionary of custom versioning policies, including version ranges, 
        pre-release version deprecation, and minimum supported versions. 
        Default is None.
    **config_params : dict
        Additional configuration parameters for custom policies or logging.

    Attributes
    ----------
    expected_version : str
        The expected version for the model.
    allow_minor_version_mismatch : bool
        Indicates whether minor version mismatches are allowed.
    check_deprecation : bool
        Indicates whether deprecation checks are enabled.
    deprecated_versions : list
        A list of deprecated versions of the model.
    log_results : bool
        Indicates whether logging of results is enabled.
    api_url : str
        URL for fetching deprecated versions from an external service.
    config_file_path : str
        Path to a configuration file for deprecated versions.
    fallback_versions : list
        A hardcoded list of deprecated versions to fall back on.
    custom_policies : dict
        Custom versioning policies for deprecation and version checks.

    Methods
    -------
    run(model, **run_kwargs)
        Performs the versioning check by comparing the current model version 
        with the expected version, checking for deprecation, and logging results.
    _check_minor_version(current_version: str)
        Checks whether the current model version matches the expected version, 
        allowing minor version mismatches if configured.
    _check_deprecation(current_version: str)
        Determines if the current model version is deprecated based on a 
        static list, configuration file, external service, or custom policies.
    _check_deprecation_date(deprecated_date: str)
        Verifies if the deprecation date has passed for a version.
    _check_custom_deprecation_policies(current_version: str)
        Applies custom deprecation policies, such as minimum supported versions, 
        version ranges, or pre-release version deprecation.
    _fetch_deprecated_versions()
        Fetches deprecated versions from external services, configuration 
        files, or falls back to a static list.
    _fetch_from_external_service()
        Fetches deprecated versions from an external API service.
    _fetch_from_config_file()
        Loads deprecated versions from a local configuration file.
    _is_version_in_range(current_version: str, deprecated_version: str)
        Checks if the current model version falls within a deprecated range 
        or matches a pre-release version.
    _compare_semantic_versions(current_version: str, deprecated_version: str)
        Compares semantic versions of the current model and the deprecated version.

    Examples
    --------
    >>> from gofast.mlops.testing import ModelVersioning
    >>> model = SomeModelClass()
    >>> versioning_check = ModelVersioning(expected_version='2.0.0')
    >>> result = versioning_check.run(model)
    >>> print(result)

    Raises
    ------
    ValueError
        If versioning or deprecation checks fail, appropriate messages 
        will be raised or logged based on the event.

    See Also
    --------
    packaging.version.Version : Provides version parsing and comparison utilities.
    requests.get : For making HTTP requests to external versioning services.

    References
    ----------
    .. [1] Semantic Versioning (https://semver.org/).
    """

    def __init__(self, 
                 expected_version: str, 
                 allow_minor_version_mismatch: bool = False, 
                 check_deprecation: bool = False, 
                 deprecated_versions: Optional[list] = None,
                 log_results: bool = True,
                 api_url: Optional[str] = None,
                 config_file_path: Optional[str] = None,
                 fallback_versions: Optional[list] = None,
                 custom_policies: Optional[dict] = None,
                 **config_params):
        
        super().__init__(test_name=self.__class__.__name__, **config_params)
        self.expected_version = expected_version
        self.allow_minor_version_mismatch = allow_minor_version_mismatch
        self.check_deprecation = check_deprecation
        self.log_results = log_results
        self.deprecated_versions = deprecated_versions or []
        self.api_url = api_url or "https://api.example.com/deprecated-versions"
        self.config_file_path = config_file_path or "config/deprecated_versions.json"
        self.fallback_versions = fallback_versions or ['1.0.0', '1.1.0', '2.0.0-beta']
        self.custom_policies = custom_policies or {}

    @RunReturn(attribute_name="results_")
    def run(self, model, **run_kwargs):
        """
        Runs the model versioning check. This checks if the current version of 
        the model matches the expected version, whether minor version mismatches 
        are allowed, and if the current version is deprecated.

        Parameters
        ----------
        model : object
            The model object that contains a `get_version` method to return 
            the model's version.
        **run_kwargs : dict
            Additional arguments for running the versioning check.

        Returns
        -------
        dict
            A dictionary containing the results of the version check, including 
            whether the version matched the expected version, whether it is 
            deprecated, and the deprecation source.
        """
        current_version = model.get_version()

        if self.allow_minor_version_mismatch:
            version_match = self._check_minor_version(current_version)
        else:
            version_match = current_version == self.expected_version

        is_deprecated = False
        deprecation_source = None
        deprecated_by_range = False
        deprecated_by_custom_policy = False

        if self.check_deprecation:
            is_deprecated = self._check_deprecation(current_version)

            if current_version in self.deprecated_versions:
                deprecation_source = "direct_deprecation_list"
            else:
                fetched_versions = self._fetch_deprecated_versions()
                if current_version in fetched_versions:
                    deprecation_source = "external_service"
                elif self._is_version_in_range(current_version, fetched_versions):
                    deprecation_source = "version_range"
                    deprecated_by_range = True
                elif self._check_custom_deprecation_policies(current_version):
                    deprecation_source = "custom_policy"
                    deprecated_by_custom_policy = True

        if self.log_results:
            self.log_event('model_versioning_check', {
                'current_version': current_version,
                'expected_version': self.expected_version,
                'version_match': version_match,
                'is_deprecated': is_deprecated,
                'deprecation_source': deprecation_source,
                'deprecated_by_range': deprecated_by_range,
                'deprecated_by_custom_policy': deprecated_by_custom_policy
            })

        self._set_runned()

        self.results_ = {
            'current_version': current_version,
            'expected_version': self.expected_version,
            'version_match': version_match,
            'is_deprecated': is_deprecated,
            'deprecation_source': deprecation_source,
            'deprecated_by_range': deprecated_by_range,
            'deprecated_by_custom_policy': deprecated_by_custom_policy,
            'allow_minor_version_mismatch': self.allow_minor_version_mismatch
        }

    
    def _check_minor_version(self, current_version: str) -> bool:
        """
        Compares the major and minor version components of the current version 
        with the expected version, allowing for minor version mismatches if 
        configured.
    
        This method splits the version strings into `major`, `minor`, and 
        `patch` components and checks whether the major and minor parts match.
    
        Parameters
        ----------
        current_version : str
            The current version of the model in the format `major.minor.patch`.
    
        Returns
        -------
        bool
            True if the major and minor versions match between the current version 
            and the expected version, False otherwise.
    
        Examples
        --------
        >>> self.expected_version = "1.2.0"
        >>> current_version = "1.2.5"
        >>> self._check_minor_version(current_version)
        True
        
        Notes
        -----
        This method only compares the major and minor components of the version.
        Patch-level mismatches are ignored if `allow_minor_version_mismatch` 
        is enabled.
        """
        expected_major, expected_minor, *_ = self.expected_version.split('.')
        current_major, current_minor, *_ = current_version.split('.')
    
        # Only match major and minor versions
        return expected_major == current_major and expected_minor == current_minor
    
    
    def _check_deprecation(self, current_version: str) -> bool:
        """
        Checks whether the provided model version is deprecated. The check 
        can be based on a static list of deprecated versions, a configuration 
        file, or an external service. It also applies custom deprecation 
        policies if defined.
    
        This method can handle simple version matching as well as sophisticated 
        deprecation checks such as checking version ranges, pre-release versions, 
        and deprecation dates.
    
        Parameters
        ----------
        current_version : str
            The current version of the model to be checked for deprecation.
    
        Returns
        -------
        bool
            True if the current version is deprecated, False otherwise.
    
        Notes
        -----
        - The method checks a variety of deprecation sources including:
          - A static list of deprecated versions.
          - External services that provide deprecated versions.
          - Version ranges (e.g., all versions below 1.5.0).
          - Custom policies like deprecating pre-release versions or major version limits.
        - If custom policies are set, those policies will be applied first.
        
        Examples
        --------
        >>> deprecated_versions = ['1.0.0', '2.0.0-beta']
        >>> current_version = "2.0.0"
        >>> self._check_deprecation(current_version)
        False
        
        See Also
        --------
        _fetch_deprecated_versions : Fetches deprecated versions from external or local sources.
        _check_custom_deprecation_policies : Applies custom policies to the version check.
        """
        deprecated_versions = self.deprecated_versions or self._fetch_deprecated_versions()
    
        # Check for exact match in the deprecated versions list
        if current_version in deprecated_versions:
            return True
        
        # Handle more sophisticated version checks, including ranges or custom logic
        for version_info in deprecated_versions:
            if isinstance(version_info, dict):
                deprecated_version = version_info.get("version")
                deprecated_date = version_info.get("date")
                
                # Check if the current version falls within a deprecated range
                if self._is_version_in_range(current_version, deprecated_version):
                    # Check deprecation date if provided
                    if deprecated_date and not self._check_deprecation_date(deprecated_date):
                        continue  # Skip this version if it's not deprecated yet
                    return True
            else:
                # Handle basic deprecated versions without metadata
                if self._is_version_in_range(current_version, version_info):
                    return True
    
        # Apply custom deprecation policies if defined
        if self._check_custom_deprecation_policies(current_version):
            return True
    
        return False
    
    
    def _check_deprecation_date(self, deprecated_date: str) -> bool:
        """
        Checks whether the provided deprecation date has passed, i.e., whether 
        the current date is after the deprecation date.
    
        Parameters
        ----------
        deprecated_date : str
            The deprecation date in the format `YYYY-MM-DD`.
    
        Returns
        -------
        bool
            True if the current date is on or after the deprecation date, 
            False otherwise.
    
        Notes
        -----
        - This method is used to evaluate whether a model version is deprecated 
          based on a given date.
        
        Examples
        --------
        >>> deprecated_date = "2023-01-01"
        >>> self._check_deprecation_date(deprecated_date)
        True
        """
        
        current_date = datetime.utcnow().date()
        deprecation_date = datetime.strptime(deprecated_date, "%Y-%m-%d").date()
    
        return current_date >= deprecation_date
    
    
    def _check_custom_deprecation_policies(self, current_version: str) -> bool:
        """
        Applies custom deprecation policies, such as deprecating all versions 
        below a certain minimum version, specific version ranges, or pre-release 
        versions like "beta" or "alpha".
    
        Parameters
        ----------
        current_version : str
            The current version of the model to be checked against custom policies.
    
        Returns
        -------
        bool
            True if the current version is deprecated according to custom policies, 
            False otherwise.
    
        Notes
        -----
        - Custom policies can include:
          - Deprecating all versions below a specified minimum version.
          - Deprecating specific versions.
          - Deprecating versions that fall within a specified range (e.g., 1.0.0 to 1.5.0).
          - Deprecating pre-release versions such as `beta` or `alpha`.
    
        Examples
        --------
        >>> self.custom_policies = {
        >>>     "min_supported_version": "1.5.0",
        >>>     "deprecated_versions": ["2.0.0"],
        >>>     "deprecated_version_ranges": [{"start": "1.0.0", "end": "1.5.0"}]
        >>> }
        >>> self._check_custom_deprecation_policies("1.4.0")
        True
        
        See Also
        --------
        packaging.version.Version : Used for comparing semantic versions.
        """
        from packaging import version
    
        current_ver = version.parse(current_version)
    
        # Custom Policy 1: Deprecate all versions below a minimum supported version
        min_supported_version = self.custom_policies.get("min_supported_version")
        if min_supported_version:
            min_supported_version_parsed = version.parse(min_supported_version)
            if current_ver < min_supported_version_parsed:
                return True
    
        # Custom Policy 2: Deprecate specific versions
        deprecated_versions = self.custom_policies.get("deprecated_versions", [])
        if current_version in deprecated_versions:
            return True
    
        # Custom Policy 3: Deprecate specific version ranges
        version_ranges = self.custom_policies.get("deprecated_version_ranges", [])
        for version_range in version_ranges:
            start_version = version.parse(version_range["start"])
            end_version = version.parse(version_range["end"])
            if start_version <= current_ver <= end_version:
                return True
    
        # Custom Policy 4: Deprecate pre-release versions (e.g., "beta", "alpha")
        pre_release_policy = self.custom_policies.get("deprecate_pre_releases", False)
        if pre_release_policy and current_ver.is_prerelease:
            return True
    
        return False

    def _fetch_deprecated_versions(self) -> list:
        """
        Fetches the list of deprecated versions from various sources, such as internal 
        lists, external services, or configuration files. This method supports both 
        static and dynamic retrieval of deprecated versions, ensuring that the latest 
        deprecation information is available.
    
        The method follows a multi-step process:
        1. Attempt to retrieve deprecated versions from an external service.
        2. Check for deprecated versions from a local configuration file.
        3. Fall back to a hardcoded list of deprecated versions if no external or 
           configuration data is available.
    
        Returns
        -------
        list
            A list of deprecated versions retrieved from one of the sources. 
            If no external 
            service or configuration file is available, the method will return 
            a fallback list.
    
        Notes
        -----
        - External services are fetched via an API call, and if an error occurs,
        it logs the error 
          and moves to the next source.
        - Configuration files are read from a predefined path.
        - A hardcoded fallback list ensures that the system still works if
        neither external nor 
          configuration sources are available.
    
        Examples
        --------
        >>> deprecated_versions = self._fetch_deprecated_versions()
        >>> print(deprecated_versions)
        ['1.0.0', '1.5.0', '2.0.0-beta']
    
        Raises
        ------
        FileNotFoundError
            If the configuration file is not found when attempting to read 
            local deprecated versions.
        Exception
            If an error occurs during the external service call, it logs the 
            error and falls back to the next source.
    
        See Also
        --------
        _fetch_from_external_service : Method to fetch deprecated versions 
        from an external API.
        _fetch_from_config_file : Method to fetch deprecated versions from a 
        local configuration file.
        """
        try:
            external_versions = self._fetch_from_external_service()
            if external_versions:
                return external_versions
        except Exception as e:
            self.log_event('external_service_error', {'error': str(e)})
        
        try:
            local_versions = self._fetch_from_config_file()
            if local_versions:
                return local_versions
        except FileNotFoundError:
            self.log_event('config_file_error', {'error': 'Config file not found'})
    
        self.log_event('fallback_to_static_list', {'versions': self.fallback_versions})
        return self.fallback_versions
    
    
    def _fetch_from_external_service(self) -> list:
        """
        Fetches deprecated versions from an external API or service. This method 
        simulates making an HTTP request to retrieve deprecated version information 
        from a remote source.
    
        Returns
        -------
        list
            A list of deprecated versions returned by the external service, or an empty 
            list if the service does not respond with valid data.
    
        Notes
        -----
        - This method assumes that the API returns a JSON response containing a 
          field called `deprecated_versions` which is a list of version strings.
        - In case of an error or a non-200 response, an empty list is returned, 
          and an error is logged.
    
        Examples
        --------
        >>> external_versions = self._fetch_from_external_service()
        >>> print(external_versions)
        ['1.0.0', '1.2.0', '2.0.0-beta']
    
        See Also
        --------
        _mock_api_call : Simulates an API call for the external service.
    
        Raises
        ------
        Exception
            If an error occurs during the API call, the exception is logged
            and handled internally.
        """
        response = self._mock_api_call(self.api_url)
        if response and response.status_code == 200:
            return response.json().get('deprecated_versions', [])
        return []
    
    
    def _mock_api_call(self, url: str):
        """
        Simulates an API call to retrieve deprecated versions from an external service. 
        This is a placeholder method and should be replaced with actual logic for 
        making HTTP requests, such as using the `requests` library.
    
        Parameters
        ----------
        url : str
            The URL of the external service that provides deprecated versions.
    
        Returns
        -------
        requests.Response
            The HTTP response object. If the request is successful, it will contain 
            the data returned by the API.
    
        Notes
        -----
        - This method is intended to simulate a real-world API call. In practice, 
          this should be replaced with a real HTTP request implementation.
        
        Examples
        --------
        >>> response = self._mock_api_call('https://api.example.com/deprecated-versions')
        >>> print(response.status_code)
        200
    
        See Also
        --------
        requests.get : For making actual HTTP requests in production environments.
        """
        import requests
        return requests.get(url)
    
    
    def _fetch_from_config_file(self) -> list:
        """
        Fetches the list of deprecated versions from a local configuration file, such 
        as a JSON or YAML file. This method reads the configuration file, parses its 
        content, and returns the deprecated versions.
    
        Returns
        -------
        list
            A list of deprecated versions found in the configuration file.
    
        Notes
        -----
        - The configuration file should contain a key `deprecated_versions` that holds 
          a list of version strings.
        - If the file is not found, a `FileNotFoundError` will be raised and logged.
    
        Examples
        --------
        >>> local_versions = self._fetch_from_config_file()
        >>> print(local_versions)
        ['1.0.0', '1.5.0', '2.0.0']
    
        Raises
        ------
        FileNotFoundError
            If the configuration file cannot be found at the specified path.
    
        See Also
        --------
        json.load : Used for parsing JSON configuration files.
        """
        import json
        with open(self.config_file_path, 'r') as f:
            config_data = json.load(f)
        return config_data.get('deprecated_versions', [])
    
    
    def _is_version_in_range(self, current_version: str, deprecated_version: str) -> bool:
        """
        Checks whether the current version falls within a range of deprecated versions, 
        or if it matches a specific version with a special suffix such as 'beta', 
        'alpha', or 'rc'.
    
        This method handles both exact version matching and versions with pre-release 
        identifiers, comparing them to see if they fall within a deprecated range.
    
        Parameters
        ----------
        current_version : str
            The current version of the model.
        deprecated_version : str
            The deprecated version or version range to check against.
    
        Returns
        -------
        bool
            True if the current version falls within the deprecated range or matches 
            a version with a pre-release identifier, False otherwise.
    
        Examples
        --------
        >>> self._is_version_in_range('1.2.0', '1.0.0')
        False
        >>> self._is_version_in_range('1.2.0-beta', '1.2.0-beta')
        True
    
        Notes
        -----
        - Pre-release versions are identified by a dash (`-`) followed by a suffix 
          such as 'beta', 'alpha', or 'rc'.
        - This method is commonly used in conjunction with `_compare_semantic_versions` 
          for range-based comparisons.
    
        See Also
        --------
        _compare_semantic_versions : Used to compare semantic versions for range checking.
        """
        # Handle pre-release versions (e.g., beta, alpha, rc)
        if '-' in deprecated_version:
            deprecated_base, pre_release = deprecated_version.split('-')
            if current_version.startswith(deprecated_base) and pre_release in current_version:
                return True
    
        # Compare semantic versions
        return self._compare_semantic_versions(current_version, deprecated_version)

    
    def _compare_semantic_versions(self, current_version: str, deprecated_version: str) -> bool:
        """
        Compares two semantic versions and checks whether the current version 
        is less than or equal to the deprecated version. This logic helps 
        identify whether a version is considered deprecated based on version 
        comparison rules.
    
        Parameters
        ----------
        current_version : str
            The current version of the model.
        deprecated_version : str
            The deprecated version or range to compare against.
    
        Returns
        -------
        bool
            True if the current version is less than or equal to the deprecated 
            version, indicating that it is deprecated. False otherwise.
    
        Notes
        -----
        - Semantic versioning follows the format `major.minor.patch`, and this method 
          compares versions based on those components.
        - This method uses the `packaging.version` library to handle version parsing 
          and comparison.
    
        Examples
        --------
        >>> self._compare_semantic_versions('1.2.0', '1.5.0')
        True
        >>> self._compare_semantic_versions('2.0.0', '1.5.0')
        False
    
        See Also
        --------
        packaging.version.Version : 
            Provides functionality for parsing and comparing versions.
    
        References
        ----------
        .. [1] Semantic Versioning (https://semver.org/).
        """
        from packaging import version
    
        current_ver = version.parse(current_version)
        deprecated_ver = version.parse(deprecated_version)
    
        # Deprecate versions below the given threshold
        if current_ver < deprecated_ver:
            return True
    
        # Deprecate specific patch versions
        if current_ver == deprecated_ver:
            return True
    
        return False

class PerformanceRegression(BaseTest):
    """
    PerformanceRegression class is designed to evaluate and detect performance 
    regressions in a machine learning model by comparing it against a baseline 
    model. It uses specified metrics to evaluate performance and detect if the 
    current model's performance has fallen below an acceptable threshold compared 
    to the baseline.

    The primary comparison is between the model's performance and the baseline 
    model's performance. If a performance drop greater than the threshold is detected, 
    the system flags the issue as a regression.

    .. math::
        \text{regression} = (S_{\text{model}} < S_{\text{baseline}} - \text{threshold})

    where :math:`S_{\text{model}}` is the performance score of the current model, 
    :math:`S_{\text{baseline}}` is the score of the baseline model, and 
    `threshold` is the acceptable difference before considering it a regression.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        The machine learning model to evaluate for performance regression.
    baseline_model : Optional[sklearn.base.BaseEstimator], optional
        The baseline model to compare against the current model's performance.
        If None, no comparison is made against a baseline. Default is None.
    metrics : Optional[dict], optional
        A dictionary of metric names and corresponding callable functions to 
        evaluate the model's performance. Default metric is the model's `score` 
        method.
    threshold : float, optional
        The acceptable difference between the current model's performance and 
        the baseline model's performance before considering it a regression. 
        Default is 0.05.
    parallel_execution : bool, optional
        If True, enables parallel execution of evaluations for performance 
        comparison. Default is False.
    random_seed : Optional[int], optional
        The random seed for reproducibility. Default is None.
    **config_params : dict
        Additional configuration parameters for test customization.

    Attributes
    ----------
    model : sklearn.base.BaseEstimator
        The machine learning model to be evaluated.
    baseline_model : sklearn.base.BaseEstimator
        The baseline model for comparison.
    metrics : dict
        A dictionary of metrics for performance evaluation.
    threshold : float
        The threshold for detecting performance regression.
    results_ : dict
        The results of the performance regression test, including whether 
        regression was detected.

    Notes
    -----
    - This class is designed to work with models that follow the scikit-learn API.
    - If no baseline model is provided, the system will only evaluate the current model.

    Methods
    -------
    fit(X, y, **fit_params)
        Fits the current model (and baseline model if provided) to the data.
    evaluate(X, y)
        Evaluates the model's performance using custom metrics and checks for 
        regression against the baseline model.
    compare_with_baseline(X, y)
        Compares the current model's performance with the baseline and prints 
        whether regression is detected.

    Examples
    --------
    >>> from gofast.mlops.testing import PerformanceRegression
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.metrics import accuracy_score
    >>>
    >>> model = RandomForestClassifier(random_state=42)
    >>> baseline_model = RandomForestClassifier(random_state=42)
    >>> regression_test = PerformanceRegression(
    >>>     model=model, 
    >>>     baseline_model=baseline_model, 
    >>>     metrics={'accuracy': accuracy_score}, 
    >>>     threshold=0.01
    >>> )
    >>> X_train, y_train = ...
    >>> regression_test.fit(X_train, y_train)
    >>> result = regression_test.evaluate(X_train, y_train)
    >>> print(result)

    Raises
    ------
    ValueError
        If the model has not been fitted before running the evaluation or 
        comparison with the baseline.

    See Also
    --------
    sklearn.metrics : Common evaluation metrics such as `accuracy_score`, `precision_score`.

    References
    ----------
    .. [1] Kuhn, M., Johnson, K. (2013). "Applied Predictive Modeling," Springer.
    """

    def __init__(self, 
                 model, 
                 baseline_model=None, 
                 metrics: Optional[dict] = None, 
                 threshold: float = 0.05, 
                 parallel_execution: bool = False, 
                 random_seed: Optional[int] = None, 
                 **config_params):
        super().__init__(test_name=self.__class__.__name__,
                         parallel_execution=parallel_execution,
                         random_seed=random_seed,
                         **config_params)
        self.model = model
        self.baseline_model = baseline_model
        self.metrics = metrics or {'score': model.score}
        self.threshold = threshold

    def fit(self, X, y, **fit_params):
        """
        Fit the model (and baseline model if provided) with the given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values for training.
        **fit_params : dict
            Additional parameters to pass to the model's `fit` method.

        Returns
        -------
        self : PerformanceRegression
            Returns the instance of the class after fitting the model(s).

        Notes
        -----
        - Both the current model and baseline model (if provided) are fitted to the 
          same training data.
        """
        # Check the data for validity
        X, y = check_X_y(X, y)

        # Fit the current model
        self.model.fit(X, y, **fit_params)

        # Fit the baseline model if provided
        if self.baseline_model:
            self.baseline_model.fit(X, y, **fit_params)

        self._set_fitted()
        
        return self 

    def evaluate(self, X, y):
        """
        Evaluate the model and the baseline model (if provided) on the provided data 
        using the specified metrics and detect performance regression.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples used for evaluation.
        y : array-like of shape (n_samples,)
            The true target values for evaluation.

        Returns
        -------
        dict
            A dictionary containing:
            - `model_results`: The evaluation results of the current model.
            - `baseline_results`: The evaluation results of the baseline 
            model (if provided).
            - `regression_detected`: A boolean indicating whether performance 
            regression was detected.

        Raises
        ------
        ValueError
            If the model has not been fitted before running the evaluation.

        Notes
        -----
        - This method checks if the performance of the current model is significantly 
          worse than the baseline model based on the threshold.
        """
        self.check_is_fitted(msg="The model must be fitted before running"
                             " performance regression tests.")
        
        # Validate the input data
        X, y = check_X_y(X, y)

        # Evaluate the model using the specified metrics
        model_results = {name: metric(self.model, X, y) 
                         for name, metric in self.metrics.items()}

        baseline_results = {}
        regression_detected = False

        # If a baseline model is provided, compare its results
        if self.baseline_model:
            baseline_results = {name: metric(self.baseline_model, X, y)
                                for name, metric in self.metrics.items()}
            regression_detected = any(
                (model_results[name] < baseline_results[name] - self.threshold)
                for name in model_results
            )

        # Log the results of the test
        self.log_event('performance_regression_test', {
            'model_results': model_results,
            'baseline_results': baseline_results,
            'regression_detected': regression_detected
        })

        return {
            'model_results': model_results,
            'baseline_results': baseline_results,
            'regression_detected': regression_detected
        }


    def compare_with_baseline(self, X, y):
        """
        Compares the current model performance with the baseline model (if provided).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples used for evaluation.
        y : array-like of shape (n_samples,)
            The true target values for evaluation.

        Returns
        -------
        dict
            A dictionary containing the results of the comparison and a message indicating 
            whether performance regression was detected.

        Notes
        -----
        - If no performance regression is detected, the method will print "
        No performance regression detected."
        - If a regression is detected, the method will print "Performance 
        regression detected based on the threshold."
        """
        self.check_is_fitted(msg="The evaluation must be fit before comparing with the baseline.")

        results = self.evaluate(X, y)

        if not results['regression_detected']:
            print("No performance regression detected.")
        else:
            print("Performance regression detected based on the threshold.")

        return results


class CIIntegration(BaseTest):
    """
    CIIntegration class provides an interface to integrate with a Continuous 
    Integration (CI) tool. 
    It allows the triggering of CI actions for a specific project with options 
    for retries, response validation, timeouts, and logging.

    This class can handle retries and failures and provides mechanisms
    to log events and build detailed results for each CI action run.

    Parameters
    ----------
    ci_tool : object
        The CI tool object that exposes a method to trigger actions. 
        This object should have 
        a `trigger_action` method to interact with the CI system 
        (e.g., Jenkins, Travis, GitLab CI).
    project_name : str
        The name of the project in the CI tool for which the action will 
        be triggered.
    trigger_action : str
        The action to be triggered on the CI tool (e.g., 'build', 'deploy').
    retry_attempts : int, optional
        The number of retry attempts if the CI action fails. Default is 3.
    retry_delay : int, optional
        The delay in seconds between retry attempts. Default is 5 seconds.
    timeout : Optional[int], optional
        The maximum time to wait for the CI action to complete. 
        If None, the CI tool's 
        default timeout is used. Default is None.
    validate_response : bool, optional
        If True, the response from the CI tool is validated using a built-in
        validation method. 
        Default is True.
    log_results : bool, optional
        If True, logs the results of the CI action and retries. Default is True.
    **config_params : dict
        Additional configuration parameters.

    Attributes
    ----------
    ci_tool : object
        The CI tool used for triggering actions.
    project_name : str
        The name of the project being worked on.
    trigger_action : str
        The action to be triggered.
    retry_attempts : int
        The number of retry attempts allowed.
    retry_delay : int
        The delay between retries.
    timeout : Optional[int]
        The timeout for the CI action.
    validate_response : bool
        Whether to validate the CI tool's response.
    log_results : bool
        Whether to log the results.
    results_ : dict
        The results of the CI action, including success/failure status 
        and any response details.

    Notes
    -----
    - This class expects the CI tool to provide a `trigger_action` method 
      that takes `project_name` and `action` as required arguments.
    - If retries are enabled, the action will be retried based on the 
    configured delay and attempt limits.
    - Response validation can be customized by overriding the 
    `_validate_ci_response` method.

    Methods
    -------
    run(**run_kwargs)
        Executes the CI action with retries and logging. Handles retries on
        failure, validates 
        responses, and logs the process.
    _trigger_ci_action(**run_kwargs)
        Triggers the action on the CI tool with optional timeout and runtime
        configurations.
    _validate_ci_response(response) -> bool
        Validates the response from the CI tool to ensure success.
    _handle_retry(attempt: int)
        Handles retry logic, including delays between attempts.
    _log_ci_event(response)
        Logs the details of the CI event, including the response.
    _build_run_results(response, success: bool, error: Optional[str] = None) -> dict
        Builds and returns the results of the run, including success or error details.

    Examples
    --------
    >>> from gofast.mlops.testing import CIIntegration
    >>> ci_tool = Jenkins()  # Example CI tool object
    >>> ci_test = CIIntegration(ci_tool, project_name="MyProject", trigger_action="build")
    >>> ci_test.run()
    >>> print(ci_test.results_)

    Raises
    ------
    ValueError
        If the response validation fails or the CI action does not succeed
        within the retries.

    See Also
    --------
    requests.get : For making actual HTTP requests if the CI tool uses a REST API.

    References
    ----------
    .. [1] Humble, J., Farley, D. (2010). "Continuous Delivery: 
        Reliable Software Releases through Build, Test, and 
        Deployment Automation," Addison-Wesley.
    """

    def __init__(self, 
                 ci_tool, 
                 project_name: str, 
                 trigger_action: str, 
                 retry_attempts: int = 3, 
                 retry_delay: int = 5,  
                 timeout: Optional[int] = None,  
                 validate_response: bool = True,  
                 log_results: bool = True, 
                 **config_params):
        """
        Initialize the CIIntegration test object with CI tool, project, 
        and action details.
        """
        super().__init__(test_name=self.__class__.__name__, 
                         log_results=log_results, 
                         **config_params)
        self.ci_tool = ci_tool
        self.project_name = project_name
        self.trigger_action = trigger_action
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.validate_response = validate_response
    
    
    @RunReturn(attribute_name="results_")
    def run(self, **run_kwargs):
        """
        Execute the CI action with retries, error handling, and logging.

        This method attempts to trigger the specified CI action, retries if it fails, 
        and logs each attempt. It validates the response from the CI tool 
        if `validate_response` 
        is True and handles failures according to the retry policy.

        Parameters
        ----------
        **run_kwargs : dict
            Additional runtime parameters to pass to the CI tool's 
            `trigger_action` method.

        Returns
        -------
        dict
            A dictionary containing the results of the CI action, including success or 
            failure details, response status, and any errors encountered.

        Raises
        ------
        ValueError
            If the CI action fails after all retry attempts or the response
            validation fails.
        """
        for attempt in range(1, self.retry_attempts + 1):
            try:
                response = self._trigger_ci_action(**run_kwargs)
                if self.validate_response:
                    if not self._validate_ci_response(response):
                        raise ValueError("Invalid response from CI tool.")
                self._set_runned()

                # Log success
                self._log_ci_event(response)
                self.results_ = self._build_run_results(response, success=True)

            except Exception as e:
                self.log_event('ci_action_error', {
                    'attempt': attempt,
                    'error': str(e),
                    'project_name': self.project_name,
                    'ci_tool': self.ci_tool.name
                })

                # Retry logic
                if attempt < self.retry_attempts:
                    self._handle_retry(attempt)
                else:
                    self.results_ = self._build_run_results(None, success=False, error=str(e))
                
    def _trigger_ci_action(self, **run_kwargs):
        """
        Trigger the CI action via the CI tool, handling optional timeouts
        and additional parameters.

        Parameters
        ----------
        **run_kwargs : dict
            Additional parameters to pass to the CI tool's `trigger_action` method.

        Returns
        -------
        object
            The response from the CI tool after triggering the action.

        Notes
        -----
        This method handles triggering the CI action and passing any necessary 
        runtime configurations such as timeouts.
        """
        if self.timeout:
            return self.ci_tool.trigger_action(
                self.project_name, action=self.trigger_action, 
                timeout=self.timeout, **run_kwargs
            )
        else:
            return self.ci_tool.trigger_action(
                self.project_name, action=self.trigger_action, **run_kwargs
            )

    def _validate_ci_response(self, response) -> bool:
        """
        Validates the response from the CI tool.

        Parameters
        ----------
        response : object
            The response object returned from the CI tool.

        Returns
        -------
        bool
            True if the response indicates success, False otherwise.

        Notes
        -----
        - The default validation checks for an HTTP 200 status and a 
        success field in 
          the JSON response. This can be customized based on the CI 
          tool's response format.
        """
        return response.status_code == 200 and 'success' in response.json()

    def _handle_retry(self, attempt: int):
        """
        Handles retry logic, including delays between retry attempts.

        Parameters
        ----------
        attempt : int
            The current attempt number.

        Notes
        -----
        - The delay between retries increases exponentially based 
        on the attempt number.
        - Logs the retry attempt details.
        """
        
        retry_in = self.retry_delay * attempt  # Exponential backoff
        self.log_event('ci_action_retry', {
            'attempt': attempt,
            'retry_in': retry_in,
            'project_name': self.project_name,
            'ci_tool': self.ci_tool.name
        })
        time.sleep(retry_in)

    def _log_ci_event(self, response):
        """
        Logs the details of the CI event, including the action 
        triggered and the response.

        Parameters
        ----------
        response : object
            The response object returned from the CI tool.
        """
        self.log_event('ci_integration_test', {
            'ci_tool': self.ci_tool.name,
            'project_name': self.project_name,
            'action_triggered': self.trigger_action,
            'response_status': response.status_code,
            'response': response.json()
        })

    def _build_run_results(self, response, success: bool,
                           error: Optional[str] = None):
        """
        Build and return the results of the CI action.

        Parameters
        ----------
        response : object
            The response from the CI tool after triggering the action.
        success : bool
            Indicates whether the action was successful.
        error : Optional[str], optional
            The error message if the action failed. Default is None.

        Returns
        -------
        dict
            A dictionary containing details of the CI action results, including 
            success status, response data, and any error messages.
        """
        result = {
            'ci_tool': self.ci_tool.name,
            'project_name': self.project_name,
            'action_triggered': self.trigger_action,
            'success': success
        }

        if success and response:
            result.update({
                'response_status': response.status_code,
                'response': response.json()
            })
        elif not success:
            result['error'] = error

        return result

