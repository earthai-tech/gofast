# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Dynamic system implements various dynamic system models for classification 
and regression tasks within the gofast library. These models are designed to 
handle complex, time-dependent data by combining dynamic system theory with 
machine learning techniques.
"""
from collections import defaultdict
from numbers import Real
# import warnings

import numpy as np
from sklearn.base import (
    ClassifierMixin,
    RegressorMixin,
)
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.metrics import (
    log_loss,
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
)
from sklearn.utils._param_validation import StrOptions
from sklearn.utils import check_random_state # , gen_batches 

from ..api.types import Any, Optional,Tuple
from ..compat.sklearn import Interval, get_sgd_loss_param
from ..decorators import TrainingProgressBar
from ..tools.coreutils import gen_X_y_batches 
from ..metrics import twa_score, prediction_stability_score
from ..tools.validator import (
    check_is_fitted,
    check_X_y,
    check_array,
    validate_length_range
)
from ._dynamic_system import BaseHammersteinWiener
from .util import activator

# Attempt to import 'type_of_target' from sklearn;
# fallback to local if error
try:
    from sklearn.utils.multiclass import type_of_target
except:
    from ..tools.coreutils import type_of_target

__all__= ["HammersteinWienerClassifier","HammersteinWienerRegressor" ]

class HammersteinWienerRegressor(BaseHammersteinWiener, RegressorMixin):
    """
    Hammerstein-Wiener Regressor.
    
    The Hammerstein-Wiener (HW) model is a block-structured nonlinear model
    that consists of three main components: a nonlinear input block, a linear
    dynamic block, and a nonlinear output block. This structure allows the HW
    model to capture complex nonlinear relationships in data while maintaining
    interpretability and computational efficiency.
    
    .. math::
        \mathbf{y} = f_{\text{output}} \left( \mathbf{H} f_{\text{input}}
        \left( \mathbf{X} \right) \right)
    
    where:
    :math:`f_{\text{input}}` is the nonlinear input estimator,
    :math:`\mathbf{H}` represents the linear dynamic block (e.g., regression
    coefficients), and
    :math:`f_{\text{output}}` is the nonlinear output estimator.
    
    Parameters
    ----------
    nonlinear_input_estimator : estimator, default=None
        The estimator to model the nonlinear relationship at the input.
        It must implement the methods ``fit`` and either ``transform`` or
        ``predict``. If ``None``, no nonlinear transformation is applied
        to the input data.
    
    nonlinear_output_estimator : estimator, default=None
        The estimator to model the nonlinear relationship at the output.
        It must implement the methods ``fit`` and either ``transform`` or
        ``predict``. If ``None``, no nonlinear transformation is applied
        to the output data.
    
    p : int, default=1
        The number of lagged observations to include in the model. This
        determines the number of past time steps used to predict the
        current output.
    
    loss : str, default="mse"
        The loss function to use for training. Supported options are:
        
        - ``"mse"``: Mean Squared Error
        - ``"mae"``: Mean Absolute Error
        - ``"huber"``: Huber Loss
        - ``"time_weighted_mse"``: Time-Weighted Mean Squared Error
    
    output_scale : tuple or None, default=None
        The desired range for scaling the output predictions. If provided,
        predictions are scaled to fit within the specified range using
        min-max scaling. For example, ``output_scale=(0, 1)`` scales the
        outputs to the range [0, 1]. If ``None``, no scaling is applied.
    
    time_weighting : str or None, default="linear"
        Method for applying time-based weights to the loss function.
        Supported options are:
        
        - ``"linear"``: Linearly increasing weights over time.
        - ``"exponential"``: Exponentially increasing weights over time.
        - ``"inverse"``: Inversely proportional weights over time.
        - ``None``: No time-based weighting (equal weights).
    
    feature_engineering : str, default='auto'
        Method for feature engineering. Currently supports only ``'auto'``,
        which enables automatic feature creation based on the number of
        lagged observations.
    
    delta : float, default=1.0
        The threshold parameter for the Huber loss function. Determines the
        point where the loss function transitions from quadratic to linear.
    
    epsilon : float, default=1e-8
        A small constant added to avoid division by zero during scaling.
    
    shuffle : bool, default=True
        Whether to shuffle the training data before each epoch.
    
    batch_size : int or str, default='auto'
        The number of samples per gradient update. If set to ``'auto'``,
        the batch size is determined automatically based on the dataset
        size.
    
    optimizer : str, default='adam'
        Optimization algorithm to use for training the linear dynamic
        block. Supported options are:
        
        - ``'sgd'``: Stochastic Gradient Descent
        - ``'adam'``: Adaptive Moment Estimation
        - ``'adagrad'``: Adaptive Gradient Algorithm
        
    learning_rate : float, default=0.001
        The initial learning rate for the optimizer. Controls the step size
        during gradient descent updates.
    
    max_iter : int, default=1000
        Maximum number of iterations (epochs) for training the linear
        dynamic block.
    
    tol : float, default=1e-3
        Tolerance for the optimization. Training stops when the loss
        improvement is below this threshold.
    
    early_stopping : bool, default=False
        Whether to stop training early if the validation loss does not
        improve after a certain number of iterations.
    
    validation_fraction : float, default=0.1
        The proportion of the training data to set aside as validation data
        for early stopping.
    
    n_iter_no_change : int, default=5
        Number of iterations with no improvement to wait before stopping
        training early.
        
    random_state : int, RandomState instance, default=None
        Determines random number generation for weights and bias
        Pass an int for reproducible results across multiple function calls.
    
    n_jobs : int or None, default=None
        Number of CPU cores to use during training. ``-1`` means using all
        available cores. If ``None``, the number of jobs is determined
        automatically.
    
    verbose : int, default=0
        Controls the verbosity of the training process. Higher values
        result in more detailed logs.
    
    Attributes
    ----------
    linear_model_ : SGDRegressor
        The linear dynamic block trained using stochastic gradient descent.
    
    best_loss_ : float or None
        The best validation loss observed during training. Used for early
        stopping.
    
    initial_loss_ : float
        The loss computed on the entire dataset after initial training.
    
    is_fitted_ : bool
        Indicates whether the model has been fitted.
    
    Methods
    -------
    fit(X, y, **fit_params)
        Fit the Hammerstein-Wiener regressor model to data.
    
    predict(X)
        Predict target values for input samples.
    
    score(X, y)
        Return the coefficient of determination R^2 of the prediction.
    
    transform(X)
        Apply the nonlinear input transformation followed by the linear
        dynamic block.
    
    inverse_transform(y)
        Apply the inverse of the nonlinear output transformation.
    
    Examples
    --------
    >>> from gofast.estimators.dynamic_system import HammersteinWienerRegressor
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.linear_model import SGDRegressor
    >>> # Initialize the Hammerstein-Wiener regressor with a linear scaler
    >>> hw_regressor = HammersteinWienerRegressor(
    ...     nonlinear_input_estimator=StandardScaler(),
    ...     nonlinear_output_estimator=StandardScaler(),
    ...     p=2,
    ...     loss="huber",
    ...     output_scale=(0, 1),
    ...     time_weighting="linear",
    ...     optimizer='adam',
    ...     learning_rate=0.01,
    ...     batch_size=64,
    ...     max_iter=500,
    ...     tol=1e-4,
    ...     early_stopping=True,
    ...     validation_fraction=0.2,
    ...     n_iter_no_change=10,
    ...     shuffle=True,
    ...     delta=1.0,
    ...     epsilon=1e-10,
    ...     n_jobs=-1,
    ...     verbose=1
    ... )
    >>> # Fit the model on training data
    >>> hw_regressor.fit(X_train, y_train)
    >>> # Make predictions on new data
    >>> predictions = hw_regressor.predict(X_test)
    
    Notes
    -----
    - The Hammerstein-Wiener model is particularly effective for systems
      where the input-output relationship can be decomposed into distinct
      nonlinear and linear components. This structure allows the model to
      capture complex dynamics while maintaining interpretability.
    
    - Proper selection of the number of lagged observations (`p`) is
      crucial for capturing the temporal dependencies in the data. A higher
      value of `p` allows the model to consider more past observations but may
      increase computational complexity.
    
    - Time-based weighting can be used to emphasize recent observations
      more than older ones, which is useful in time series forecasting where
      recent data points may be more indicative of future trends.
    
    - The choice of optimizer (`optimizer`) and learning rate
      (`learning_rate`) significantly impacts the convergence and performance
      of the linear dynamic block. It is advisable to experiment with
      different optimizers and learning rates based on the specific dataset
      and problem.
    
    See Also
    --------
    scikit-learn :py:mod:`sklearn.base.BaseEstimator`  
        The base class for all estimators in scikit-learn, providing
        basic parameter management and utility methods.
    
    HammersteinModel :class:`~gofast.estimators.HammersteinWienerClassifier`  
        A concrete implementation of the Hammerstein-Wiener classification 
        model.
    
    SGDRegressor :class:`~sklearn.linear_model.SGDRegressor`  
        An estimator for linear regression with stochastic gradient descent.
    
    References
    ----------
    .. [1] Hammerstein, W. (1950). "Beiträge zum Problem der adaptiven
       Regelung". *Zeitschrift für angewandte Mathematik und Mechanik*,
       30(3), 345-367.
    .. [2] Wiener, N. (1949). "Extrapolation, Interpolation, and Smoothing
       of Stationary Time Series". *The MIT Press*.
    .. [3] Ljung, L. (1999). *System Identification: Theory for the
       User*. Prentice Hall.
    """
    
    _parameter_constraints: dict = {
        **BaseHammersteinWiener._parameter_constraints,
        "output_scale": [None, tuple],
        "delta": [Interval(Real, 0, None, closed='left')],
        "loss": [StrOptions({
            "mse", "mae", "huber", "time_weighted_mse", 
        }), None],
    }
    
    def __init__(
        self,
        nonlinear_input_estimator=None,
        nonlinear_output_estimator=None,
        p=1,
        loss="mse",
        output_scale=None,
        time_weighting="linear",
        feature_engineering='auto',
        delta=1.0,
        epsilon=1e-8,
        shuffle=True, 
        batch_size="auto", 
        optimizer='adam',
        learning_rate=0.001,
        max_iter=1000,
        tol=1e-3,
        early_stopping=False,
        validation_fraction=0.1, 
        n_iter_no_change=5,
        random_state=None, 
        n_jobs=None,
        verbose=0
    ):
        super().__init__(
            nonlinear_input_estimator=nonlinear_input_estimator,
            nonlinear_output_estimator=nonlinear_output_estimator,
            p=p,
            feature_engineering=feature_engineering,
            n_jobs=n_jobs,
            verbose=verbose,
            optimizer=optimizer,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_iter=max_iter,
            tol=tol,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            shuffle=shuffle,
            epsilon=epsilon,
            time_weighting=time_weighting,
            random_state=random_state 
        )

        self.output_scale = output_scale
        self.delta = delta
        self.loss = loss

        
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **fit_params: Any
    ) -> 'HammersteinWienerRegressor':
        """
        Fit the Hammerstein-Wiener regressor model.
        
        This method trains the Hammerstein-Wiener regressor by performing the 
        following steps:
        - Validating parameters and input data.
        - Applying nonlinear input transformations and creating lagged features.
        - Initializing the linear model.
        - Splitting the data into training and validation sets.
        - Determining the batch size and number of batches.
        - Initializing metrics for tracking performance.
        - Performing the training loop with optional early stopping.
        - Computing the initial loss on the entire dataset after training.
        
        Parameters
        ----------
        X : np.ndarray
            The input features, shape (n_samples, n_features).
        y : np.ndarray
            The target values, shape (n_samples,).
        fit_params : dict, optional
            Additional parameters for fitting the model.
        
        Returns
        -------
        self : HammersteinWienerRegressor
            Fitted regressor instance.
        """
        if self.verbose > 0:
            print("Starting HammersteinWienerRegressor fit method.")
    
        # Initialize metrics
        # Loss should start at infinity, to be minimized
        # Validation loss should also start at infinity
        # PSS (Prediction Stability Score) at infinity, reflecting potential
        # instability
        metrics = {
            'loss': float('inf'),
            'PSS': float('inf'),
            'val_loss': float("inf"),
            'val_PSS': float('inf'),
        }
    
        # Initialize early stopping parameters
        self.best_loss_ = np.inf if self.early_stopping else None
        self._no_improvement_count = 0
    
        # Validate model parameters and preprocess input data
        self._validate_params()
        X, y = self._validate_input_data(X, y)
        X, y = check_X_y(X, y, multi_output=True)
        X_transformed = self._apply_nonlinear_input(X, y)
        X_lagged = self._create_lagged_features(X_transformed)
        
        self._random_state= check_random_state(self.random_state)
        
        if self.verbose > 0:
            print("Fitting linear model with batch training.")
    
        # Initialize the linear dynamic model (SGDRegressor)
        self._initialize_model()
    
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = self._split_data(X_lagged, y)
    
        # Determine the number of samples in the dataset
        
        # n_samples = X_lagged.shape[0]
    
        # # Determine and validate batch size
        # if self.batch_size == "auto":
        #     batch_size = min(200, n_samples)
        # else:
        #     if self.batch_size > n_samples:
        #         warnings.warn(
        #             "Got `batch_size` less than 1 or larger than "
        #             "sample size. It is going to be clipped."
        #         )
        #     batch_size = np.clip(self.batch_size, 1, n_samples)
        
        # batches = list(gen_batches(
        #     n_samples, 
        #     self.batch_size, 
        #     min_batch_size=1
        #     )
        # )
    
        # Generating `X_y_batches` prior to validation ensures that batch
        # processing is consistent throughout the training epoch. However,
        # when batches are generated using index-based slicing, shuffling
        # can occasionally result in some batches having zero samples.
        # This scenario lead to Progress Bar Misalignment:
        # - Metrics Inaccuracy: If empty batches are skipped, the
        #   metrics may not be updated correctly, leading to inaccurate
        #   performance evaluations.
        # Indeed, The progress bar may not accurately reflect the actual 
        # number of batches, often falling short by one or two batches,
        # which disrupts the user’s perception of training progress.
        #   
        # To mitigate these issues, the optimal approach is to remove any
        # empty batches and include only valid batches. This ensures that
        # the number of batches aligns perfectly with the progress bar
        # status when verbosity is enabled. By doing so, metrics are
        # consistently updated, and the progress bar accurately represents
        # the training progress.
        #
        # Additionally, using `gen_X_y_batches` is more stable compared to
        # inline indexing within the training loop. Pre-generating a list
        # of `(X_batch, y_batch)` tuples before the training epoch
        # prevents inconsistencies and potential errors that might arise
        # from dynamic batch generation during training. This approach
        # enhances the overall stability and reliability of the training
        # process.
        X_y_batches= gen_X_y_batches (
            X_train, y_train,
            batch_size=self.batch_size , 
            min_batch_size= 1, 
            shuffle= self.shuffle,
            random_state= self._random_state 
        ) 
        
        # XXX
        # Initialize early stopping parameters and track best loss
        self.best_loss_ = np.inf if self.early_stopping else None
        self._no_improvement_count = 0
        
        # Begin the training loop
        if self.verbose == 0:
            with TrainingProgressBar(
                epochs=self.max_iter,
                steps_per_epoch=len(X_y_batches),
                metrics=metrics
            ) as progress_bar:
                for epoch in range(self.max_iter):
                    print(f"Epoch {epoch + 1}/{self.max_iter}")
                    
                    # Initialize epoch metrics
                    epoch_metrics = defaultdict(list)
                    
                    # Train the model for the current epoch
                    self._train_epoch(
                        # X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val,
                        Xy_batches=X_y_batches,
                        metrics=metrics,
                        epoch=epoch,
                        bar=progress_bar,
                        epoch_metrics=epoch_metrics
                    )
                    print("\n")  
                    
                    # Check for early stopping condition
                    if self.early_stopping and (
                        self._no_improvement_count >= self.n_iter_no_change
                    ):
                        print(
                            f"Early stopping triggered after "
                            f"{epoch + 1} epochs."
                        )
                        break
        else:
            for epoch in range(self.max_iter):
                # Initialize epoch metrics
                epoch_metrics = defaultdict(list)
                
                if self.verbose > 0:
                    print(f"Epoch {epoch + 1}/{self.max_iter}")
                
                # Train the model for the current epoch
                self._train_epoch(
                    # X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    Xy_batches=X_y_batches,
                    metrics=metrics,
                    epoch=epoch,
                    epoch_metrics=epoch_metrics
                )
                
                # Check for early stopping condition
                if self.early_stopping and (
                    self._no_improvement_count >= self.n_iter_no_change
                ):
                    if self.verbose > 0:
                        print(
                            f"Early stopping triggered after "
                            f"{epoch + 1} epochs."
                        )
                    break
    
        # Compute initial loss on the entire dataset
        y_linear = self.linear_model_.predict(X_lagged)
        self._apply_nonlinear_output(y_linear, y)
    
        # Predict and compute the initial loss
        y_pred_initial = self.predict(X)
        self.initial_loss_ = self._compute_loss(
            y, y_pred_initial
        )
    
        if self.verbose > 0:
            print(f"Initial loss: {self.initial_loss_}")
            print("Fit method completed.")
        
        return self

    def _initialize_model(self) -> None:
        """
        Initialize the SGDRegressor for the linear dynamic block.
        
        This method configures and initializes the linear dynamic model using
        Scikit-learn's SGDRegressor. The learning rate type and loss function
        are determined based on the optimizer and loss parameters specified.
        """
        # Determine the learning rate schedule based on the optimizer
        learning_rate_type = self._get_learning_rate_type()
        
        # Determine the loss function based on the loss parameter
        self.loss_function_ = self._get_loss_function()
        
        # Initialize the SGDRegressor with specified parameters
        self.linear_model_ = SGDRegressor(
            loss=self.loss_function_,           # Loss function parameter
            learning_rate=learning_rate_type,   # Learning rate schedule
            eta0=self.learning_rate,            # Initial learning rate
            max_iter=1,                         # Maximum iterations
            tol=None,                           # Disable internal tolerance
            shuffle=False,                      # Manual shuffling handled
            verbose=0,                          # Verbosity level- suppress
            epsilon=self.delta,                 # Epsilon parameter
            random_state=None                   # Random state for reproducibility
        )

    def _get_learning_rate_type(self) -> str:
        """
        Determine the learning rate type based on the optimizer.
        
        Maps the optimizer to the corresponding learning rate schedule used by
        the SGDRegressor.
        
        Returns
        -------
        str
            The learning rate schedule ('optimal', 'adaptive', 'invscaling').
        """
        return {
            'sgd': 'optimal',
            'adam': 'adaptive',
            'adagrad': 'invscaling'
            # Default to 'optimal' if not specified
        }.get(self.optimizer, 'optimal')  

    def _get_loss_function(self) -> str:
        """
        Determine the loss function based on the loss parameter.
        
        Maps the specified loss to the corresponding loss function used by
        the SGDRegressor.
        
        Returns
        -------
        str
            The loss function parameter (
                'squared_error', 'epsilon_insensitive', 'huber').
        """
        return {
            "mse": 'squared_error',
            "mae": 'epsilon_insensitive',
            "huber": 'huber'
            # Default to 'squared_error' if not specified
        }.get(self.loss, 'squared_error')  

    def _update_metrics(
        self,
        y_batch: np.ndarray,
        y_pred: np.ndarray,
        metrics: dict[str, float],
        batch_idx: int,
        step_metrics: dict[str, float],
        epoch_metrics: dict[str, list[float]]
    ) -> Tuple[dict[str, float], dict[str, list[float]]]:
        """
        Update metrics for progress bar and stability calculation.
        
        This method calculates and updates various performance metrics based on
        the true and predicted values for the current batch. Metrics are aggregated
        both for the current step and across the entire epoch.
        
        Parameters
        ----------
        y_batch : np.ndarray
            The true target values for the current batch.
        y_pred : np.ndarray
            The predicted target values for the current batch.
        metrics : dict[str, float]
            Dictionary to store aggregated metrics.
        batch_idx : int
            The index of the current batch within the epoch.
        step_metrics : dict[str, float]
            Dictionary to store metrics for the current step/batch.
        epoch_metrics : dict[str, list[float]]
            Dictionary to collect metrics across all batches in the epoch.
        
        Returns
        -------
        Tuple[dict[str, float], dict[str, list[float]]]
            Updated step_metrics and epoch_metrics after calculation.
        """
        try:
            # Calculate batch loss using the specified loss function
            batch_loss = self._compute_pred_loss(y_batch, y_pred)
            
            # Calculate Prediction Stability Score (PSS)
            batch_pss = prediction_stability_score(y_pred)
            
            if batch_idx == 0:
                # Initialize metrics with the first batch's results
                metrics['loss'] = batch_loss
                metrics['PSS'] = batch_pss
            else:
                # Update metrics by averaging with previous values
                metrics['loss'] = (
                    metrics['loss'] * batch_idx + batch_loss
                ) / (batch_idx + 1)
                metrics['PSS'] = (
                    metrics['PSS'] * batch_idx + batch_pss
                ) / (batch_idx + 1)
            
            # Update step_metrics with current batch's results
            step_metrics['loss'] = batch_loss
            step_metrics['PSS'] = batch_pss
            
            # Append current batch's metrics to epoch_metrics for tracking
            epoch_metrics['loss'].append(batch_loss)
            epoch_metrics['PSS'].append(batch_pss)
        
        except ValueError:
            # Ignore errors in metrics calculation (e.g., empty batch)
            pass
        
        if self.verbose > 0:
            # Print aggregated metrics if verbosity is set
            print(
                f"loss: {metrics['loss']:.4f} - PSS: {metrics['PSS']:.4f}"
            )
        
        return step_metrics, epoch_metrics
    
    def _evaluate_batch(
        self,
        X_batch: np.ndarray,
        y_batch: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        metrics: dict[str, float],
        batch_idx: int,
        step_metrics: dict[str, float],
        epoch_metrics: dict[str, list[float]]
    ) -> Tuple[dict[str, float], dict[str, list[float]]]:
        """
        Evaluate the performance of the model on the current batch.
        
        This method predicts on the current training batch and updates both
        step-specific and epoch-wide metrics. If validation data is provided,
        it also evaluates the model on the validation set and updates relevant
        metrics. Early stopping is checked based on validation loss.
        
        Parameters
        ----------
        X_batch : np.ndarray
            The input features for the current training batch.
        y_batch : np.ndarray
            The target labels for the current training batch.
        X_val : Optional[np.ndarray]
            The input features for the validation set.
        y_val : Optional[np.ndarray]
            The target labels for the validation set.
        metrics : dict[str, float]
            Dictionary to store aggregated metrics.
        batch_idx : int
            The index of the current batch within the epoch.
        step_metrics : dict[str, float]
            Dictionary to store metrics for the current step/batch.
        epoch_metrics : dict[str, list[float]]
            Dictionary to collect metrics across all batches in the epoch.
        
        Returns
        -------
        Tuple[dict[str, float], dict[str, list[float]]]
            Updated step_metrics and epoch_metrics after evaluation.
        """
        # Predict target values for the current batch
        y_pred = self.linear_model_.predict(X_batch)
        
        # Update metrics based on the current batch predictions
        step_metrics, epoch_metrics = self._update_metrics(
            y_batch=y_batch,
            y_pred=y_pred,
            metrics=metrics,
            batch_idx=batch_idx,
            step_metrics=step_metrics,
            epoch_metrics=epoch_metrics
        )
        
        if X_val is not None and y_val is not None:
            # Predict target values for the validation set
            y_val_pred = self.linear_model_.predict(X_val)
            
            # Compute validation loss using the specified loss function
            val_loss = self._compute_pred_loss(y_val, y_val_pred)
            
            # Compute Prediction Stability Score (PSS) for validation predictions
            val_pss = prediction_stability_score(y_val_pred)
            
            if batch_idx == 0:
                # Initialize validation metrics with the first batch's results
                metrics['val_loss'] = val_loss
                metrics['val_PSS'] = val_pss
            
            # Update step_metrics with current validation results
            step_metrics['val_loss'] = val_loss
            step_metrics['val_PSS'] = val_pss
            
            # Append validation metrics to epoch_metrics for tracking
            epoch_metrics['val_loss'].append(val_loss)
            epoch_metrics['val_PSS'].append(val_pss)
            
            if self.verbose > 1:
                # Print validation metrics if verbosity is high
                print(
                    f"val_loss: {val_loss:.4f} - "
                    f"val_PSS: {val_pss:.4f}"
                )
            
            if self.early_stopping:
                # Handle early stopping based on validation loss
                self._handle_early_stopping(val_loss)
        
        return step_metrics, epoch_metrics


    def _compute_pred_loss(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Compute loss value from batch prediction.
        
        This method calculates the loss between the true target values and
        the predicted values using the specified loss function.
        
        Parameters
        ----------
        y_true : np.ndarray
            The true target values, shape (n_samples,).
        y_pred : np.ndarray
            The predicted target values, shape (n_samples,).
        
        Returns
        -------
        float
            The computed loss value.
        """
        if self.loss_function_ == 'mae':
            # Calculate Mean Absolute Error
            loss = mean_absolute_error(y_true, y_pred)
        else:
            # Default to Mean Squared Error
            loss = mean_squared_error(y_true, y_pred)
        
        return loss

    
    def predict(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Predict target values for input samples.
        
        This method generates predictions for the input samples by performing
        the following steps:
        - Ensuring the model is fitted.
        - Validating and preprocessing input data.
        - Applying nonlinear input transformations.
        - Creating lagged features.
        - Getting predictions from the linear dynamic block.
        - Applying nonlinear output transformations.
        - Applying optional scaling to constrain the output range.
        
        Parameters
        ----------
        X : np.ndarray
            Input features, shape (n_samples, n_features).
        
        Returns
        -------
        np.ndarray
            Predicted target values, shape (n_samples,).
        """
        if self.verbose > 0:
            print("Starting HammersteinWienerRegressor predict method.")
        
        # Ensure the model is fitted before making predictions
        check_is_fitted(self, 'linear_model_')
        
        # Validate and preprocess input data
        X = check_array(X)
        
        # Apply nonlinear input transformation to capture nonlinear input relationships
        X_transformed = self._apply_nonlinear_input(X)
        
        # Create lagged features for the linear dynamic block
        X_lagged = self._create_lagged_features(X_transformed)
        
        # Get predictions from the linear dynamic block
        y_linear = self.linear_model_.predict(X_lagged)
        
        # Apply nonlinear output transformation to the intermediate predictions
        y_pred = self._apply_nonlinear_output(y_linear)
        
        # Apply optional scaling to constrain the output range
        y_pred_scaled = self._scale_output(y_pred)
        
        if self.verbose > 0:
            print("Predict method completed.")
        
        return y_pred_scaled

    def _compute_loss(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Compute loss value based on the specified loss function.
        
        This method calculates the loss between the true target values and the
        predicted values using the specified loss function. Supported loss
        functions include Mean Squared Error (MSE), Mean Absolute Error (MAE),
        Huber loss, and Time-Weighted Mean Squared Error (time_weighted_mse).
        
        Parameters
        ----------
        y_true : np.ndarray
            True target values, shape (n_samples, [n_outputs]).
        y_pred : np.ndarray
            Predicted target values, shape (n_samples, [n_outputs]).
        
        Returns
        -------
        float
            Computed loss value.
        
        Raises
        ------
        ValueError
            If an unsupported loss function is specified.
        """
        if self.verbose > 0:
            print(f"Computing loss using {self.loss} loss function.")
        
        # Compute loss based on the specified function
        if self.loss == "mse":
            # Compute mean squared error over all samples and outputs
            loss = np.mean((y_true - y_pred) ** 2)
        
        elif self.loss == "mae":
            # Compute mean absolute error over all samples and outputs
            loss = np.mean(np.abs(y_true - y_pred))
        
        elif self.loss == "huber":
            # Compute Huber loss
            residual = y_true - y_pred
            loss = np.mean(np.where(
                np.abs(residual) < self.delta,
                0.5 * residual ** 2,
                self.delta * (np.abs(residual) - 0.5 * self.delta)
            ))
        
        elif self.loss == "time_weighted_mse":
            # Compute time-weighted mean squared error
            weights = self._compute_time_weights(len(y_true))
            # Expand weights if y_true is multi-output
            if y_true.ndim > 1:
                weights = weights[:, np.newaxis]  # Shape (n_samples, 1)
            loss = np.mean(weights * (y_true - y_pred) ** 2)
        
        else:
            # Unsupported loss function
            raise ValueError(f"Unsupported loss function: {self.loss}")
        
        if self.verbose > 0:
            print(f"Computed loss: {loss}")
        
        return loss

    
    def _compute_time_weights(
        self,
        n: int
    ) -> np.ndarray:
        """
        Compute time-based weights based on the specified weighting method.
        
        This method generates weights for each sample based on the chosen
        time weighting method: linear, exponential, inverse, or equal weighting.
        These weights can be used for time-weighted loss calculations.
        
        Parameters
        ----------
        n : int
            The number of samples to compute weights for.
        
        Returns
        -------
        np.ndarray
            Array of computed weights, shape (n_samples,).
        """
        if self.verbose > 0:
            print(
                f"Computing time weights using {self.time_weighting} method."
            )
        
        # Compute linear, exponential, or inverse weights based on user input
        if self.time_weighting == "linear":
            # Linear weights increasing from 0.1 to 1.0
            weights = np.linspace(0.1, 1.0, n)
        
        elif self.time_weighting == "exponential":
            # Exponential weights increasing rapidly
            weights = np.exp(np.linspace(0, 1, n)) - 1
            weights /= weights.max()  # Normalize to [0, 1]
        
        elif self.time_weighting == "inverse":
            # Inverse weights decreasing over time
            weights = 1 / np.arange(1, n + 1)
            weights /= weights.max()  # Normalize to [0, 1]
        
        else:
            # Equal weighting if method is unrecognized or None
            weights = np.ones(n)
        
        if self.verbose > 0:
            print(f"Time weights: {weights}")
        
        return weights

    def _scale_output(
        self,
        y: np.ndarray
    ) -> np.ndarray:
        """
        Apply optional scaling to constrain the output range.
        
        This method applies min-max scaling to the predicted outputs based on
        the specified output range. It ensures that the predictions fall within
        the desired bounds.
        
        Parameters
        ----------
        y : np.ndarray
            Predicted target values, shape (n_samples, [n_outputs]).
        
        Returns
        -------
        np.ndarray
            Scaled predicted target values, shape (n_samples, [n_outputs]).
        """
        if self.output_scale is not None:
            if self.verbose > 0:
                print("Scaling output predictions.")
            
            self.output_scale = validate_length_range(
                self.output_scale, param_name="Output scale"
                ) 
            # Apply min-max scaling based on specified output range
            y_min, y_max = self.output_scale
            
            # Compute min and max per output
            y_min_per_output = y.min(axis=0)
            y_max_per_output = y.max(axis=0)
            
            # Avoid division by zero
            denom = y_max_per_output - y_min_per_output + self.epsilon
            
            # Normalize to [0, 1] per output
            y_norm = (y - y_min_per_output) / denom
            
            # Scale to [y_min, y_max]
            y_scaled = y_norm * (y_max - y_min) + y_min
            
            if self.verbose > 0:
                print(f"Scaled output range: [{y_min}, {y_max}]")
            
            return y_scaled
        
        return y

class HammersteinWienerClassifier(BaseHammersteinWiener, ClassifierMixin):
    """
    Hammerstein-Wiener Classifier.
    
    The Hammerstein-Wiener (HW) model is a block-structured nonlinear model
    that consists of three main components: a nonlinear input block, a linear
    dynamic block, and a nonlinear output block. This structure allows the HW
    model to capture complex nonlinear relationships in data while maintaining
    interpretability and computational efficiency.
    
    .. math::
        \mathbf{y} = f_{\text{output}} \left( \mathbf{H} f_{\text{input}}
        \left( \mathbf{X} \right) \right)
    
    where:
    :math:`f_{\text{input}}` is the nonlinear input estimator,
    :math:`\mathbf{H}` represents the linear dynamic block (e.g., regression
    coefficients), and
    :math:`f_{\text{output}}` is the nonlinear output estimator.
    
    The `HammersteinWienerClassifier` extends the base HW model to handle
    classification tasks. It incorporates a loss function tailored for
    classification, such as cross-entropy, enabling the model to predict
    categorical outcomes effectively.
    
    Parameters
    ----------
    nonlinear_input_estimator : estimator, default=None
        The estimator to model the nonlinear relationship at the input.
        It must implement the methods ``fit`` and either ``transform`` or
        ``predict``. If ``None``, no nonlinear transformation is applied
        to the input data.
    
    nonlinear_output_estimator : estimator, default=None
        The estimator to model the nonlinear relationship at the output.
        It must implement the methods ``fit`` and either ``transform`` or
        ``predict``. If ``None``, no nonlinear transformation is applied
        to the output data.
    
    p : int, default=1
        The number of lagged observations to include in the model. This
        determines the number of past time steps used to predict the
        current output.
    
    loss : str, default="cross_entropy"
        The loss function to use for training. Supported options are:
        
        - ``"cross_entropy"``: Cross-Entropy Loss
        - ``"time_weighted_cross_entropy"``: Time-Weighted Cross-Entropy Loss
    
    time_weighting : str or None, default="linear"
        Method for applying time-based weights to the loss function.
        Supported options are:
        
        - ``"linear"``: Linearly increasing weights over time.
        - ``"exponential"``: Exponentially increasing weights over time.
        - ``"inverse"``: Inversely proportional weights over time.
        - ``None``: No time-based weighting (equal weights).
    
    feature_engineering : str, default='auto'
        Method for feature engineering. Currently supports only ``'auto'``,
        which enables automatic feature creation based on the number of
        lagged observations.
    
    epsilon : float, default=1e-8
        A small constant added to avoid division by zero during scaling.
    
    shuffle : bool, default=True
        Whether to shuffle the training data before each epoch.
    
    batch_size : int or str, default='auto'
        The number of samples per gradient update. If set to ``'auto'``,
        the batch size is determined automatically based on the dataset
        size.
    
    optimizer : str, default='adam'
        Optimization algorithm to use for training the linear dynamic
        block. Supported options are:
        
        - ``'sgd'``: Stochastic Gradient Descent
        - ``'adam'``: Adaptive Moment Estimation
        - ``'adagrad'``: Adaptive Gradient Algorithm
        
    learning_rate : float, default=0.001
        The initial learning rate for the optimizer. Controls the step size
        during gradient descent updates.
    
    max_iter : int, default=1000
        Maximum number of iterations (epochs) for training the linear
        dynamic block.
    
    tol : float, default=1e-3
        Tolerance for the optimization. Training stops when the loss
        improvement is below this threshold.
    
    early_stopping : bool, default=False
        Whether to stop training early if the validation loss does not
        improve after a certain number of iterations.
    
    validation_fraction : float, default=0.1
        The proportion of the training data to set aside as validation data
        for early stopping.
    
    n_iter_no_change : int, default=5
        Number of iterations with no improvement to wait before stopping
        training early.
        
    random_state : int, RandomState instance, default=None
        Determines random number generation for weights and bias
        Pass an int for reproducible results across multiple function calls.
    
    n_jobs : int or None, default=None
        Number of CPU cores to use during training. ``-1`` means using all
        available cores. If ``None``, the number of jobs is determined
        automatically.
    
    verbose : int, default=0
        Controls the verbosity of the training process. Higher values
        result in more detailed logs.
    
    Attributes
    ----------
    linear_model_ : SGDRegressor
        The linear dynamic block trained using stochastic gradient descent.
    
    best_loss_ : float or None
        The best validation loss observed during training. Used for early
        stopping.
    
    initial_loss_ : float
        The loss computed on the entire dataset after initial training.
    
    is_fitted_ : bool
        Indicates whether the model has been fitted.
    
    Methods
    -------
    fit(X, y, **fit_params)
        Fit the Hammerstein-Wiener classifier model to data.
    
    predict(X)
        Predict class labels for input samples.
    
    predict_proba(X)
        Predict class probabilities for input samples.
    
    score(X, y)
        Return the mean accuracy on the given test data and labels.
    
    transform(X)
        Apply the nonlinear input transformation followed by the linear
        dynamic block.
    
    inverse_transform(y)
        Apply the inverse of the nonlinear output transformation.
    
    Examples
    --------
    >>> from gofast.estimators.dynamic_system import HammersteinWienerClassifier
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.linear_model import SGDRegressor
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> # Generate synthetic classification data
    >>> X, y = make_classification(n_samples=1000, n_features=20, 
    ...                            n_informative=15, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.2, random_state=42
    ... )
    >>> # Initialize the Hammerstein-Wiener classifier with a linear scaler
    >>> hw_classifier = HammersteinWienerClassifier(
    ...     nonlinear_input_estimator=StandardScaler(),
    ...     nonlinear_output_estimator=StandardScaler(),
    ...     p=2,
    ...     loss="cross_entropy",
    ...     time_weighting="linear",
    ...     optimizer='adam',
    ...     learning_rate=0.01,
    ...     batch_size=64,
    ...     max_iter=500,
    ...     tol=1e-4,
    ...     early_stopping=True,
    ...     validation_fraction=0.2,
    ...     n_iter_no_change=10,
    ...     shuffle=True,
    ...     delta=1.0,
    ...     epsilon=1e-10,
    ...     n_jobs=-1,
    ...     verbose=1
    ... )
    >>> # Fit the model on training data
    >>> hw_classifier.fit(X_train, y_train)
    >>> # Make predictions on new data
    >>> predictions = hw_classifier.predict(X_test)
    >>> # Predict class probabilities on new data
    >>> probabilities = hw_classifier.predict_proba(X_test)
    
    Notes
    -----
    - The Hammerstein-Wiener model is particularly effective for classification
      tasks where the input-output relationship can be decomposed into distinct
      nonlinear and linear components. This structure allows the model to
      capture complex dynamics while maintaining interpretability.
    
    - Proper selection of the number of lagged observations (`p`) is
      crucial for capturing the temporal dependencies in the data. A higher
      value of `p` allows the model to consider more past observations but may
      increase computational complexity.
    
    - Time-based weighting can be used to emphasize recent observations
      more than older ones, which is useful in time series classification
      where recent data points may be more indicative of future trends.
    
    - The choice of optimizer (`optimizer`) and learning rate
      (`learning_rate`) significantly impacts the convergence and performance
      of the linear dynamic block. It is advisable to experiment with
      different optimizers and learning rates based on the specific dataset
      and problem.
    
    See Also
    --------
    scikit-learn :py:mod:`sklearn.base.BaseEstimator`  
        The base class for all estimators in scikit-learn, providing
        basic parameter management and utility methods.
    
    HammersteinModel :class:`~gofast.estimators.HammersteinWienerRegressor`  
        A concrete implementation of the Hammerstein-Wiener regression model.
    
    SGDRegressor :class:`~sklearn.linear_model.SGDRegressor`  
        An estimator for linear regression with stochastic gradient descent.
    
    LogisticRegression :class:`~sklearn.linear_model.LogisticRegression`  
        A logistic regression classifier.
    
    References
    ----------
    .. [1] Hammerstein, W. (1950). "Beiträge zum Problem der adaptiven
       Regelung". *Zeitschrift für angewandte Mathematik und Mechanik*,
       30(3), 345-367.
    .. [2] Wiener, N. (1949). "Extrapolation, Interpolation, and Smoothing
       of Stationary Time Series". *The MIT Press*.
    .. [3] Ljung, L. (1999). *System Identification: Theory for the
       User*. Prentice Hall.
    .. [4] Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep
       Learning*. MIT Press.
    
    """
    
    _parameter_constraints: dict = {
        **BaseHammersteinWiener._parameter_constraints,
        "loss": [StrOptions({
            "cross_entropy", "time_weighted_cross_entropy"
        })],
    }
    
    def __init__(
        self,
        nonlinear_input_estimator=None,
        nonlinear_output_estimator=None,
        p=1,
        loss="cross_entropy",
        time_weighting="linear",
        feature_engineering='auto',
        epsilon=1e-8,
        shuffle=True, 
        batch_size="auto", 
        optimizer='adam',
        learning_rate=0.001,
        max_iter=1000,
        tol=1e-3,
        early_stopping=False,
        validation_fraction=0.1, 
        n_iter_no_change=5,
        random_state=None,
        n_jobs=None,
        verbose=0
    ):
        super().__init__(
            nonlinear_input_estimator=nonlinear_input_estimator,
            nonlinear_output_estimator=nonlinear_output_estimator,
            p=p,
            feature_engineering=feature_engineering,
            n_jobs=n_jobs,
            verbose=verbose,
            optimizer=optimizer,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_iter=max_iter,
            tol=tol,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change,
            shuffle=shuffle,
            epsilon=epsilon,
            time_weighting=time_weighting,
            random_state=random_state
        )

        self.loss = loss
 
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **fit_params: Any
    ) -> 'HammersteinWienerClassifier':
        """
        Fit the Hammerstein-Wiener classifier model.
        
        This method trains the Hammerstein-Wiener classifier by performing the 
        following steps:
        - Validating parameters and input data.
        - Determining if the classification problem is multilabel.
        - Applying nonlinear input transformations and creating lagged features.
        - Initializing the linear model.
        - Splitting the data into training and validation sets.
        - Determining the batch size and number of batches.
        - Initializing metrics for tracking performance.
        - Performing the training loop with optional early stopping.
        - Computing the initial loss on the entire dataset after training.
        
        Parameters
        ----------
        X : np.ndarray
            The input features, shape (n_samples, n_features).
        y : np.ndarray
            The target labels, shape (n_samples,).
        fit_params : dict, optional
            Additional parameters for fitting the model.
        
        Returns
        -------
        self : HammersteinWienerClassifier
            Fitted classifier instance.
        """
        if self.verbose > 0:
            print("Starting HammersteinWienerClassifier fit method.")
    
        # Validate model parameters
        self._validate_params()
    
        # Validate and preprocess input data
        X, y = self._validate_input_data(X, y)
        X, y = check_X_y(X, y, multi_output=True)
        self._random_state = check_random_state(self.random_state)
        
        # Determine if the classification problem is multilabel
        self.is_multilabel_ = type_of_target(y) in (
            'multilabel-indicator', 'multiclass-multioutput'
        )
    
        # Apply nonlinear input transformation
        X_transformed = self._apply_nonlinear_input(X)
    
        # Create lagged features from transformed data
        X_lagged = self._create_lagged_features(X_transformed)
    
        # Initialize the linear dynamic model
        self._initialize_model()
    
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = self._split_data(X_lagged, y)
    
        # Determine the number of samples in training data
        # n_samples = X_train.shape[0]

        #XXX
        # The `gen_X_y_batches` function performs several critical operations:
        # - Determines the total number of samples (`n_samples`) if not
        #   explicitly provided.
        # - Ensures that the `batch_size` is valid by clipping it to be
        #   at least 1 and no larger than the total number of samples.
        #   This prevents errors related to invalid batch sizes.
        # - Calculates the number of batches per epoch based on the
        #   validated `batch_size` using the `gen_batches` utility
        #   function.
        #
        # Generating `X_y_batches` prior to validation ensures that batch
        # processing is consistent throughout the training epoch. However,
        # when batches are generated using index-based slicing, shuffling
        # can occasionally result in some batches having zero samples.
        # This scenario lead to Progress Bar Misalignment:
        # - Metrics Inaccuracy: If empty batches are skipped, the
        #   metrics may not be updated correctly, leading to inaccurate
        #   performance evaluations.
        # Indeed, The progress bar may not accurately reflect the actual 
        # number of batches, often falling short by one or two batches,
        # which disrupts the user’s perception of training progress.
        #   
        # To mitigate these issues, the optimal approach is to remove any
        # empty batches and include only valid batches. This ensures that
        # the number of batches aligns perfectly with the progress bar
        # status when verbosity is enabled. By doing so, metrics are
        # consistently updated, and the progress bar accurately represents
        # the training progress.
        #
        # Additionally, using `gen_X_y_batches` is more stable compared to
        # inline indexing within the training loop. Pre-generating a list
        # of `(X_batch, y_batch)` tuples before the training epoch
        # prevents inconsistencies and potential errors that might arise
        # from dynamic batch generation during training. This approach
        # enhances the overall stability and reliability of the training
        # process.
        
        X_y_batches = gen_X_y_batches(
            X_train, y_train,
            batch_size=self.batch_size,
            min_batch_size=1,
            shuffle=self.shuffle,
            random_state=self._random_state
        )
        # if self.batch_size == "auto":
        #     batch_size = min(32, n_samples)
        # else:
        #     if self.batch_size > n_samples:
        #         warnings.warn(
        #             "Got `batch_size` less than 1 or larger than "
        #             "sample size. It is going to be clipped."
        #         )
        #     batch_size = np.clip(self.batch_size, 1, n_samples)
        
        # # Validate and set the batch size
        # # And calculate the number of batches per epoch
        # self.batch_size = validate_batch_size(batch_size, n_samples)
        # batches = list(gen_batches(
        #     n_samples, 
        #     self.batch_size, 
        #     min_batch_size=1
        #     )
        # )
    
        # Initialize metrics for tracking model performance
        metrics = {
            'loss': float('inf'),
            'accuracy': 0.0,
            'TWA': 0.0,
            'val_loss': float("inf"),
            'val_accuracy': 0.0
        }
    
        # Initialize early stopping parameters
        self.best_loss_ = np.inf if self.early_stopping else None
        self._no_improvement_count = 0
    
        # Begin the training loop
        if self.verbose == 0:
            with TrainingProgressBar(
                epochs=self.max_iter,
                steps_per_epoch=len(X_y_batches),
                metrics=metrics
            ) as progress_bar:
                for epoch in range(self.max_iter):
                    print(f"Epoch {epoch + 1}/{self.max_iter}")
                    
                    # Initialize epoch metrics
                    epoch_metrics = defaultdict(list)
                    
                    # Train the model for the current epoch
                    self._train_epoch(
                        # X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val,
                        # batches=batches,
                        Xy_batches= X_y_batches, 
                        metrics=metrics,
                        epoch=epoch,
                        bar=progress_bar,
                        epoch_metrics=epoch_metrics
                    )
                    print("\n")  
                    
                    # Check for early stopping condition
                    if self.early_stopping and (
                        self._no_improvement_count >= self.n_iter_no_change
                    ):
                        print(
                            f"Early stopping triggered after "
                            f"{epoch + 1} epochs."
                        )
                        break
        else:
            for epoch in range(self.max_iter):
                # Initialize epoch metrics
                epoch_metrics = defaultdict(list)
                
                if self.verbose > 0:
                    print(f"Epoch {epoch + 1}/{self.max_iter}")
                
                # Train the model for the current epoch
                self._train_epoch(
                    # X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    # batches=batches,
                    Xy_batches= X_y_batches, 
                    metrics=metrics,
                    epoch=epoch, 
                    epoch_metrics=epoch_metrics
                )
                
                # Check for early stopping condition
                if self.early_stopping and (
                    self._no_improvement_count >= self.n_iter_no_change
                ):
                    if self.verbose > 0:
                        print(
                            f"Early stopping triggered after "
                            f"{epoch + 1} epochs."
                        )
                    break
    
        # Compute initial loss on the entire dataset
        y_linear = self._apply_linear_dynamic_block(X_lagged)
        self._apply_nonlinear_output(y_linear, y)
        self.initial_loss_ = self._compute_loss(
            y, self.predict_proba(X)
        )
    
        if self.verbose > 0:
            print(f"Initial loss: {self.initial_loss_}")
            print("Fit method completed.")
        
        return self

    def _initialize_model(self) -> None:
        """
        Initialize the SGDClassifier for the linear dynamic block.
        
        This method configures and initializes the linear model using
        Scikit-learn's SGDClassifier. The learning rate type is determined
        based on the optimizer specified. The model is set to perform
        one iteration per call, with internal early stopping disabled
        as epochs are managed externally.
        """
        # Determine the learning rate schedule based on the optimizer
        if self.optimizer == 'sgd':
            learning_rate_type = 'optimal'
        elif self.optimizer == 'adam':
            learning_rate_type = 'adaptive'
        else:
            learning_rate_type = 'invscaling'
        
        # Initialize the linear dynamic model with specified parameters
        self.linear_model_ = SGDClassifier(
            loss=get_sgd_loss_param(),          # Loss function parameter
            learning_rate=learning_rate_type,   # Learning rate schedule
            eta0=self.learning_rate,            # Initial learning rate
            max_iter=1,                         # Manual epoch handling
            tol=None,                           # Disable internal tolerance
            shuffle=False,                      # Manual shuffling handled
            verbose=0,                          # Suppress internal logs
            n_jobs=self.n_jobs,                 # Number of parallel jobs
            n_iter_no_change=self.n_iter_no_change  # Early stopping
        )

    def _evaluate_batch(
        self,
        X_batch: np.ndarray,
        y_batch: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        metrics: dict[str, float],
        batch_idx: int,
        step_metrics: dict[str, float],
        epoch_metrics: dict[str, list[float]]
    ) -> Tuple[dict[str, float], dict[str, list[float]]]:
        """
        Evaluate the performance of the model on the current batch.
        
        This method predicts on the current training batch and updates both
        step-specific and epoch-wide metrics. If validation data is provided,
        it also evaluates the model on the validation set and updates relevant
        metrics. Early stopping is checked based on validation loss.
        
        Parameters
        ----------
        X_batch : np.ndarray
            The input features for the current training batch.
        y_batch : np.ndarray
            The target labels for the current training batch.
        X_val : Optional[np.ndarray]
            The input features for the validation set.
        y_val : Optional[np.ndarray]
            The target labels for the validation set.
        metrics : dict[str, float]
            Dictionary to store aggregated metrics.
        batch_idx : int
            The index of the current batch within the epoch.
        step_metrics : dict[str, float]
            Dictionary to store metrics for the current step/batch.
        epoch_metrics : dict[str, list[float]]
            Dictionary to collect metrics across all batches in the epoch.
        
        Returns
        -------
        Tuple[dict[str, float], dict[str, list[float]]]
            Updated step_metrics and epoch_metrics after evaluation.
        """
        # Predict class labels and probabilities for the current batch
        y_pred = self.linear_model_.predict(X_batch)
        y_pred_proba = self.linear_model_.predict_proba(X_batch)
        
        # Update metrics based on the current batch predictions
        step_metrics, epoch_metrics = self._update_metrics(
            y_true=y_batch,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            metrics=metrics,
            batch_idx=batch_idx,
            step_metrics=step_metrics,
            epoch_metrics=epoch_metrics
        )
        
        if X_val is not None and y_val is not None:
            # Predict probabilities on the validation set
            y_val_pred_proba = self.linear_model_.predict_proba(X_val)
            
            # Compute validation loss and accuracy
            val_loss = log_loss(y_val, y_val_pred_proba)
            val_accuracy = accuracy_score(
                y_val, self.linear_model_.predict(X_val)
            )
            
            if batch_idx == 0:
                # Initialize validation metrics with the first batch's results
                metrics['val_loss'] = val_loss
                metrics['val_accuracy'] = val_accuracy
            
            # Update step-specific validation metrics
            step_metrics['val_loss'] = val_loss
            step_metrics['val_accuracy'] = val_accuracy
            
            # Append validation metrics to epoch_metrics for tracking
            epoch_metrics['val_loss'].append(val_loss)
            epoch_metrics['val_accuracy'].append(val_accuracy)
            
            if self.verbose > 1:
                # Log validation metrics if verbosity is high
                print(
                    f"val_loss: {val_loss:.4f} - "
                    f"val_accuracy: {val_accuracy:.4f}"
                )
            
            if self.early_stopping:
                # Handle early stopping based on validation loss
                self._handle_early_stopping(val_loss)
        
        return step_metrics, epoch_metrics

    def _update_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        metrics: dict[str, float],
        batch_idx: int,
        step_metrics: dict[str, float],
        epoch_metrics: dict[str, list[float]]
    ) -> Tuple[dict[str, float], dict[str, list[float]]]:
        """
        Update metrics for progress bar and accuracy calculation.
        
        This method calculates and updates various performance metrics based
        on the true labels and predicted values for the current batch. Metrics
        are aggregated both for the current step and across the entire epoch.
        
        Parameters
        ----------
        y_true : np.ndarray
            The true target labels.
        y_pred : np.ndarray
            The predicted target labels.
        y_pred_proba : np.ndarray
            The predicted target probabilities.
        metrics : dict[str, float]
            Dictionary to store aggregated metrics.
        batch_idx : int
            The index of the current batch within the epoch.
        step_metrics : dict[str, float]
            Dictionary to store metrics for the current step/batch.
        epoch_metrics : dict[str, list[float]]
            Dictionary to collect metrics across all batches in the epoch.
        
        Returns
        -------
        Tuple[dict[str, float], dict[str, list[float]]]
            Updated step_metrics and epoch_metrics after calculation.
        """
        try:
            # Calculate batch loss using log loss
            batch_loss = log_loss(y_true, y_pred_proba)
            
            # Calculate batch accuracy
            batch_accuracy = accuracy_score(y_true, y_pred)
            
            # Calculate TWA (Time-Weighted Accuracy) metric
            batch_twa_accuracy = twa_score(y_true, y_pred)
            
            if batch_idx == 0:
                # Initialize metrics with the first batch's results
                metrics['loss'] = batch_loss
                metrics['accuracy'] = batch_accuracy
                metrics['TWA'] = batch_twa_accuracy
            else:
                # Update metrics by averaging with previous values
                metrics['loss'] = (
                    metrics['loss'] * batch_idx + batch_loss
                ) / (batch_idx + 1)
                metrics['accuracy'] = (
                    metrics['accuracy'] * batch_idx + batch_accuracy
                ) / (batch_idx + 1)
                metrics['TWA'] = (
                    metrics['TWA'] * batch_idx + batch_twa_accuracy
                ) / (batch_idx + 1)
            
            # Update step_metrics with current batch's results
            step_metrics['loss'] = batch_loss
            step_metrics['accuracy'] = batch_accuracy
            step_metrics['TWA'] = batch_twa_accuracy
            
            # Append current batch's metrics to epoch_metrics
            epoch_metrics['loss'].append(batch_loss)
            epoch_metrics['accuracy'].append(batch_accuracy)
            epoch_metrics['TWA'].append(batch_twa_accuracy)
        
        except ValueError:
            # Ignore errors in metrics calculation (e.g., empty batch)
            pass
        
        if self.verbose > 0:
            # Print aggregated metrics if verbosity is set
            print(
                f"loss: {metrics['loss']:.4f} - "
                f"accuracy: {metrics['accuracy']:.4f} - "
                f"TWA: {metrics['TWA']:.4f}"
            )
        
        return step_metrics, epoch_metrics

    def predict_proba(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Predict class probabilities for input samples.
        
        This method generates probability estimates for each class for the
        input samples. It applies nonlinear input transformations, creates
        lagged features, computes the linear dynamic block output, and then
        applies a nonlinear output transformation to obtain probabilities.
        
        Parameters
        ----------
        X : np.ndarray
            Input features, shape (n_samples, n_features).
        
        Returns
        -------
        np.ndarray
            Predicted class probabilities, shape (n_samples, n_classes).
        """
        if self.verbose > 0:
            print("Starting predict_proba method.")
        
        # Ensure the model has been fitted
        check_is_fitted(self, 'linear_model_')
        
        # Validate and preprocess input data
        X = check_array(X)
        
        # Apply nonlinear input transformation
        X_transformed = self._apply_nonlinear_input(X)
        
        # Create lagged features for the linear dynamic block
        X_lagged = self._create_lagged_features(X_transformed)
        
        # Get the decision function output from the linear model
        y_linear = self._apply_linear_dynamic_block(X_lagged)
        
        # Apply nonlinear output transformation to obtain transformed output
        y_transformed = self._apply_nonlinear_output(y_linear)
        
        # Convert transformed output to probabilities based on the problem type
        if self.is_multilabel_:
            # For multilabel classification, apply sigmoid activation
            y_pred_proba = activator(
                y_transformed, activation="sigmoid"
            )
        else:
            if len(self.linear_model_.classes_) == 2:
                # For binary classification, apply sigmoid activation
                y_pred_proba = activator(
                    y_transformed, activation="sigmoid"
                )
                # Ensure the output has two columns representing class probabilities
                y_pred_proba = np.hstack([
                    1 - y_pred_proba, y_pred_proba
                ])
            else:
                # For multiclass classification, apply softmax activation
                y_pred_proba = activator(
                    y_transformed, activation="softmax"
                )
        
        if self.verbose > 0:
            print("predict_proba method completed.")
        
        return y_pred_proba

    def predict(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Predict class labels for input samples.
        
        This method generates class predictions for the input samples by first
        obtaining class probabilities and then converting these probabilities
        into discrete class labels. It handles both multilabel and multiclass
        classification scenarios.
        
        Parameters
        ----------
        X : np.ndarray
            Input features, shape (n_samples, n_features).
        
        Returns
        -------
        np.ndarray
            Predicted class labels, shape (n_samples,).
        """
        if self.verbose > 0:
            print("Starting predict method.")
        
        # Obtain class probabilities
        y_pred_proba = self.predict_proba(X)
        
        if self.is_multilabel_:
            # For multilabel classification, apply threshold to probabilities
            y_pred = (y_pred_proba >= 0.5).astype(int)
        else:
            # For binary and multiclass classification, select class
            # with highest probability
            y_pred = np.argmax(y_pred_proba, axis=1)
        
        if self.verbose > 0:
            print("Predict method completed.")
        
        return y_pred

    def _compute_loss(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> float:
        """
        Compute the loss based on the specified loss function.
        
        This method calculates the loss between the true labels and the predicted
        probabilities using the specified loss function. It supports standard
        cross-entropy loss and time-weighted cross-entropy loss.
        
        Parameters
        ----------
        y_true : np.ndarray
            True target labels, shape (n_samples,).
        y_pred_proba : np.ndarray
            Predicted class probabilities, shape (n_samples, n_classes).
        
        Returns
        -------
        float
            Computed loss value.
        
        Raises
        ------
        ValueError
            If an unsupported loss function is specified.
        """
        if self.verbose > 0:
            print(f"Computing loss using {self.loss} loss function.")
        
        # Clip probabilities to prevent log of zero
        y_pred_proba = np.clip(
            y_pred_proba, self.epsilon, 1 - self.epsilon
        )
        
        # Compute loss based on the specified loss function
        if self.loss == "cross_entropy":
            loss = log_loss(y_true, y_pred_proba)
        elif self.loss == "time_weighted_cross_entropy":
            # Compute time-based weights
            weights = self._compute_time_weights(len(y_true))
            loss = log_loss(
                y_true, y_pred_proba, sample_weight=weights
            )
        else:
            raise ValueError("Unsupported loss function.")
        
        if self.verbose > 0:
            print(f"Computed loss: {loss}")
        
        return loss

