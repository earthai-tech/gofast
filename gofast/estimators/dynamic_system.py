# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Dynamic system implements various dynamic system models for classification 
and regression tasks within the gofast library. These models are designed to 
handle complex, time-dependent data by combining dynamic system theory with 
machine learning techniques.
"""
from numbers import Integral, Real
import numpy as np

from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.utils._param_validation import StrOptions
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.metrics import ( 
    log_loss, accuracy_score, mean_squared_error, mean_absolute_error
    )
from sklearn.model_selection import train_test_split 
try:
    from sklearn.utils.multiclass import type_of_target
except: 
    from ..tools.coreutils import type_of_target
    
from ..metrics import twa_score, prediction_stability_score 
from ..compat.sklearn import Interval, get_sgd_loss_param
from ..decorators import TrainingProgressBar 
from ..tools.validator import check_is_fitted, check_X_y, check_array, validate_batch_size 
from ._dynamic_system import BaseHammersteinWiener
from .util import activator

__all__= ["HammersteinWienerClassifier","HammersteinWienerRegressor" ]

class HammersteinWienerClassifier(BaseHammersteinWiener, ClassifierMixin):
    """
    Hammerstein-Wiener model for classification tasks.

    The Hammerstein-Wiener model is a block-oriented model used to
    represent nonlinear dynamic systems. It consists of a cascade of
    a static nonlinear input block, a linear dynamic block, and a
    static nonlinear output block. This model is suitable for capturing
    systems with input and output nonlinearities surrounding a linear
    dynamic system.

    The general structure of the Hammerstein-Wiener model is as follows:

    .. math::
        y(t) = f_{\\text{out}}\\left( L\\left( f_{\\text{in}}\\left( x(t) \\right) \\right) \\right)

    where:

    - :math:`x(t)` is the input vector at time :math:`t`.
    - :math:`f_{\\text{in}}` is the nonlinear input function.
    - :math:`L` is the linear dynamic block.
    - :math:`f_{\\text{out}}` is the nonlinear output function.
    - :math:`y(t)` is the output at time :math:`t`.

    See more in :ref:`User Guide`.
    
    Parameters
    ----------
    nonlinear_input_estimator : estimator object or None, default=None
        Estimator for the nonlinear input function :math:`f_{\\text{in}}`.
        This should be an object that implements ``fit`` and either
        ``transform`` or ``predict`` methods. If ``None``, no nonlinear
        transformation is applied to the input.

    nonlinear_output_estimator : estimator object or None, default=None
        Estimator for the nonlinear output function :math:`f_{\\text{out}}`.
        This should be an object that implements ``fit`` and either
        ``transform`` or ``predict`` methods. If ``None``, no nonlinear
        transformation is applied to the output.

    p : int, default=1
        Order of the linear dynamic block. This specifies the number of
        lagged inputs to include in the linear dynamic model.

    loss : {'cross_entropy', 'time_weighted_cross_entropy'}, default='cross_entropy'
        Loss function to use for evaluating the model performance.

        - ``'cross_entropy'``: Standard cross-entropy loss.
        - ``'time_weighted_cross_entropy'``: Time-weighted cross-entropy loss,
          giving more importance to recent misclassifications.

    time_weighting : {'linear', 'exponential', 'inverse'}, default='linear'
        Method for computing time weights when using time-weighted loss
        functions.

        - ``'linear'``: Linearly increasing weights over time.
        - ``'exponential'``: Exponentially increasing weights over time.
        - ``'inverse'``: Inversely decreasing weights over time.

    feature_engineering : {'auto'}, default='auto'
        Method for feature engineering. Currently, only ``'auto'`` is
        supported.

    epsilon : float, default=1e-15
        A small constant used to prevent division by zero or log of zero 
        errors during calculations  in probability estimates. Clipping 
        predictions within the range defined by `epsilon` ensures numerical
        stability by constraining values to avoid extremes.

    n_jobs : int or None, default=None
        The number of jobs to run in parallel. ``None`` means 1 unless in
        a ``joblib.parallel_backend`` context.
        
        batch_size : int, default=32
        The number of samples per batch during training. Smaller batch sizes 
        may provide more granular updates to the model, while larger batch 
        sizes can result in faster training times but require more memory.
        
    optimizer : {'sgd', 'adam', 'adagrad'}, default='adam'
        The optimization algorithm to use for training the linear dynamic 
        block. The choice of optimizer affects the model's convergence 
        rate and stability:
        
        - ``'sgd'``: Stochastic Gradient Descent, a basic optimizer that 
          updates weights based on a single sample, providing faster 
          updates but potentially noisier convergence.
        - ``'adam'``: Adaptive Moment Estimation, an advanced optimizer 
          that adjusts learning rates for each parameter dynamically, 
          improving convergence speed and stability.
        - ``'adagrad'``: Adaptive Gradient Algorithm, which adapts learning 
          rates for each parameter individually, helpful for sparse data 
          but may decay learning rates too aggressively.
          
    learning_rate : float, default=0.001
        The initial learning rate applied by the chosen optimizer. A higher 
        learning rate can speed up training but risks overshooting minima, 
        while a lower rate can lead to slower convergence but more precise 
        optimization. Recommended values often range from 0.0001 to 0.01 
        depending on model complexity.
    
    max_iter : int, default=1000
        The maximum number of training iterations (epochs) for model 
        training. Each iteration represents one pass through the entire 
        dataset. Increasing this value allows the model more time to learn 
        patterns but risks overfitting if too large.
        
    tol : float, default=1e-3
        The tolerance for the optimization criterion. This value sets a 
        threshold for determining when the optimization process should stop. 
        If the change in the loss function or model parameters falls below 
        this tolerance level, training will be halted. A smaller `tol` value 
        can lead to more precise convergence but may require more iterations, 
        while a larger value can speed up the process but risks not finding 
        the optimal solution.
    
    early_stopping : bool, default=False
        A flag that indicates whether to enable early stopping during training. 
        When set to `True`, the training process will terminate if the 
        performance of the model does not improve for a specified number of 
        iterations, as defined by `n_iter_no_change`. This feature helps 
        prevent overfitting and saves computational resources. However, it 
        should be used judiciously, as premature stopping may lead to 
        suboptimal model performance.
    
    validation_fraction : float, default=0.1
        The proportion of the training data to set aside for validation. This 
        parameter specifies the fraction of the training dataset to be used 
        for validating the model's performance during training. A common 
        choice is to set this value between 0.1 and 0.2. Using a validation 
        set helps monitor the model's generalization ability and enables the 
        application of early stopping if the validation performance does not 
        improve.
    
    n_iter_no_change : int, default=5
        The number of iterations with no improvement on the training 
        score after which the training will be stopped early. This 
        parameter helps prevent overfitting by terminating the training 
        process when the model's performance ceases to improve, thus 
        saving computational resources. Setting this value too low 
        may result in premature stopping, while a higher value could 
        lead to unnecessary computation. Typical values range from 
        5 to 20, depending on the model's convergence behavior 
        and the complexity of the dataset.

    verbose : int, default=0
        The verbosity level for logging messages during fitting and 
        prediction. A value greater than 0 enables message output, providing 
        insights into the training process. If set to ``None``, message 
        outputs will be suppressed. Adjusting this parameter can help with 
        debugging or monitoring the model's progress.


    Attributes
    ----------
    linear_model_ : object
        The linear model used in the linear dynamic block. This model 
        captures the linear relationship between lagged features and 
        the target classes and is fitted during the training process.
    
    initial_loss_ : float
        Initial loss computed on the training data after fitting. This 
        value indicates the model's performance on the training data 
        immediately following training and serves as a baseline loss.
    
    Methods
    -------
    fit(X, y)
        Fit the Hammerstein-Wiener classifier to the provided training 
        data (`X`, `y`). Applies input transformations, creates lagged 
        features, fits the linear dynamic block with a classification 
        model, and optionally applies a nonlinear output transformation.
    
    predict_proba(X)
        Predict class probabilities for the input samples (`X`) using 
        the fitted Hammerstein-Wiener classifier. Outputs the probability 
        for each class based on transformations and linear dynamics.
    
    predict(X)
        Predict binary class labels for the input samples (`X`) using 
        the fitted Hammerstein-Wiener classifier. Classifies samples 
        based on the computed class probabilities.
    

    See Also
    --------
    HammersteinWienerRegressor : Hammerstein-Wiener model for regression tasks.

    Notes
    -----
    The Hammerstein-Wiener model combines static nonlinear blocks with
    a linear dynamic block to model complex systems that exhibit both
    dynamic and nonlinear behaviors [1]_.

    The classifier is especially suited for time-series classification, 
    signal processing, and systems identification tasks where current outputs 
    are significantly influenced by historical inputs. It finds extensive 
    applications in fields like telecommunications, control systems, and 
    financial modeling, where understanding dynamic behaviors is crucial.
    
    References
    ----------
    .. [1] Schoukens, J., & Ljung, L. (2019). Nonlinear System
           Identification: A User-Oriented Roadmap. IEEE Control
           Systems Magazine, 39(6), 28-99.

    Examples
    --------
    >>> from gofast.estimators.dynamic_system import HammersteinWienerClassifier
    >>> from sklearn.datasets import make_classification
    >>> import numpy as np
    >>> # Generate synthetic data
    >>> X, y = make_classification(n_samples=200, n_features=5,
    ...                            n_informative=3, n_redundant=1,
    ...                            random_state=42)
    >>> # Instantiate the classifier
    >>> model = HammersteinWienerClassifier(p=2, verbose=1)
    >>> # Fit the model
    >>> model.fit(X, y)
    Starting HammersteinWienerClassifier fit method.
    Applying nonlinear input transformation.
    No nonlinear input estimator provided; using original X.
    Creating lagged features.
    Creating lag 1 features.
    Creating lag 2 features.
    Lagged features shape: (200, 10)
    Fitting linear model for classification.
    Applying nonlinear output transformation.
    No nonlinear output estimator provided; using linear output.
    Computed loss: 0.35
    Initial loss: 0.35
    Fit method completed.
    >>> # Predict
    >>> y_pred = model.predict(X)
    Starting predict method.
    Starting predict_proba method.
    Applying nonlinear input transformation.
    No nonlinear input estimator provided; using original X.
    Creating lagged features.
    Creating lag 1 features.
    Creating lag 2 features.
    Lagged features shape: (200, 10)
    predict_proba method completed.
    Predict method completed.
    >>> # Evaluate
    >>> from sklearn.metrics import accuracy_score
    >>> acc = accuracy_score(y, y_pred)
    >>> print(f"Accuracy: {acc:.2f}")
    Accuracy: 0.85
    """
    _parameter_constraints: dict = {
        **BaseHammersteinWiener._parameter_constraints,
        "loss": [StrOptions({"cross_entropy", "time_weighted_cross_entropy"})],
        "time_weighting": [StrOptions({"linear", "exponential", "inverse"})],
        "epsilon": [Interval(Real, 1e-15, None, closed='left')],
        "batch_size": [Interval(Integral, 1, None, closed='left')],
        "optimizer": [StrOptions({"sgd", "adam", "adagrad"})],
        "learning_rate": [Interval(Real, 0, None, closed='neither')],
        "max_iter": [Interval(Integral, 1, None, closed='left')],
        "tol": [Interval(Real, 1e-5, None, closed='neither')],
        "early_stopping": [bool],
        "validation_fraction": [Interval(Real, 0, 1, closed='neither')],
        "n_iter_no_change": [Interval(Integral, 1, None, closed='left')],
        "verbose": [Interval(Integral, 0, None, closed='left'), None]  
    }

    def __init__(
        self, 
        nonlinear_input_estimator=None, 
        nonlinear_output_estimator=None,
        p=1, 
        loss="cross_entropy", 
        time_weighting="linear", 
        feature_engineering='auto',
        n_jobs=None,
        epsilon=1e-15, 
        batch_size=32, 
        optimizer='adam',
        learning_rate=0.001, 
        max_iter=1000, 
        tol=1e-3, 
        early_stopping=False,
        validation_fraction=0.1, 
        n_iter_no_change=5, 
        verbose=0
        ):
        super().__init__(
            nonlinear_input_estimator=nonlinear_input_estimator,
            nonlinear_output_estimator=nonlinear_output_estimator,
            p=p,
            feature_engineering=feature_engineering,
            n_jobs=n_jobs,
            verbose=verbose
        )

        self.loss = loss
        self.time_weighting = time_weighting
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change

        
    def fit(self, X, y, **fit_params):
        """
        Fit the Hammerstein-Wiener classifier to the training data.
        
        This method applies transformations to the input data, constructs 
        lagged features for time-based dynamics, and fits a logistic 
        regression model on these features to capture linear relationships 
        for classification. If a nonlinear output transformation is provided, 
        it is applied to the decision function output.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples, where `n_samples` is the number of samples 
            and `n_features` is the number of features.
    
        y : array-like of shape (n_samples,)
            Target binary class labels for each sample. Class labels are 
            typically 0 or 1, representing the two classes in binary 
            classification.
    
        Returns
        -------
        self : object
            Returns the fitted `HammersteinWienerClassifier` instance.
        
        Notes
        -----
        - Logistic Regression:
          The linear model is fitted using logistic regression, which aims 
          to maximize the likelihood of correctly predicting class labels. 
          Given lagged features, the logistic model’s decision function 
          output :math:`f(X_lagged)` is given by:
    
          .. math::
              f(X_{lagged}) = X_{lagged} \\beta + b
    
          where :math:`X_{lagged}` are the lagged features, :math:`\\beta` 
          represents the regression coefficients, and :math:`b` is the bias 
          term.
    
        - Nonlinear Output Transformation:
          If provided, a nonlinear transformation function :math:`g` is 
          applied to the decision function output, resulting in:
    
          .. math::
              y_{linear} = g(f(X_{lagged}))
    

        - This method uses logistic regression to fit the linear model.
        - Nonlinear output transformations allow the model to capture more 
          complex patterns beyond the linear decision boundary.
    
        Examples
        --------
        >>> from gofast.estimators.dynamic_system import HammersteinWienerClassifier
        >>> model = HammersteinWienerClassifier()
        >>> X = np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]])
        >>> y = np.array([0, 1, 0])
        >>> model.fit(X, y)
        >>> print("Initial loss:", model.initial_loss_)
    
        See Also
        --------
        predict_proba : Predict class probabilities for input data.
        predict       : Predict binary class labels for input data.
    
        References
        ----------
        .. [1] Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements 
               of Statistical Learning. Springer.
        """
        """Fit the Hammerstein-Wiener classifier to the data."""
        if self.verbose > 0:
            print("Starting HammersteinWienerClassifier fit method.")

        self._validate_params()
        X, y = self._validate_input_data(X, y)
        X, y = check_X_y(X, y, multi_output=True)
        
        self.is_multilabel_ = type_of_target(y) in (
            'multilabel-indicator', 'multiclass-multioutput')
        
        # Apply non-linear transformation and create lagged features
        # then initialize the linear model
        X_transformed = self._apply_nonlinear_input(X)
        X_lagged = self._create_lagged_features(X_transformed)
        self._initialize_model()

        # Split data into training and validation sets
        X_train, X_val, y_train,  y_val = self._split_data(X, y, X_lagged)
        
        n_samples = X_train.shape[0]
        n_batches = int(np.floor(n_samples / self.batch_size))
        
        # Loss should start at infinity, as you're looking to minimize it.
        # TWA should start at 0, indicating no predictions made yet.
        # Validation loss should also start at infinity.
        # Validation accuracy starts at 0%.
        metrics = {
            'loss': 1.,          
            'accuracy': 0.0,               
            'TWA': 0.0,                    
            'val_loss': float('inf'),      
            'val_accuracy': 0.0             
        }


        # Early stopping initialization and Track best loss
        self.best_loss_ = np.inf if self.early_stopping else None
        self._no_improvement_count = 0
        
        if self.verbose == 0:
            with TrainingProgressBar(
                    epochs= self.max_iter, steps_per_epoch= n_batches, 
                    metrics=metrics, ) as progress_bar:
                for epoch in range(self.max_iter):
                    print(f"Epoch {epoch + 1 }/{self.max_iter}")
      
                    self._train_epoch(X_train, y_train, 
                        X_val, y_val, y,  n_batches, metrics, epoch, 
                        bar=progress_bar, 
                    )
                    print("\n")  
                    if self.early_stopping and ( 
                            self._no_improvement_count >= self.n_iter_no_change
                        ):
                        print(f"Early stopping triggered after {epoch + 1} epochs.")
                        break
        else:
            for epoch in range(self.max_iter):
                if self.verbose > 0: 
                    print(f"Epoch {epoch + 1}/{self.max_iter}")
                self._train_epoch(X_train, y_train, 
                    X_val, y_val, y, n_batches, metrics, epoch
                )

                if self.early_stopping and ( 
                        self._no_improvement_count >= self.n_iter_no_change
                        ):
                    if self.verbose > 0: 
                        print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break

        # Finalize and compute initial loss
        # Assess its performance on the data by calculating the linear 
        # predictions using the decision function of the linear model.
        # apply the nonlinear transformation to the linear outputs.
        # This combines the output from the linear model with 
        # the nonlinear component to produce the final predictions.
        # Thus, we compare the true labels (y) with the predicted probabilities
        # obtained from the model `predict_proba` method to give the 
        # probability estimates for each class,
        # which are then used to calculate the loss.
        y_linear = self.linear_model_.decision_function(X_lagged)
        self._apply_nonlinear_output(y_linear, y)
        self.initial_loss_ = self._compute_loss(y, self.predict_proba(X))

        if self.verbose > 0:
            print(f"Initial loss: {self.initial_loss_}")
            print("Fit method completed.")
        return self
    
    def _initialize_model(self):
        """Initialize SGDClassifier for the linear dynamic block."""
        learning_rate_type = 'optimal' if self.optimizer == 'sgd' else (
            'adaptive' if self.optimizer == 'adam' else 'invscaling'
        )
        self.linear_model_ = SGDClassifier(
            loss=get_sgd_loss_param(),
            learning_rate=learning_rate_type,
            eta0=self.learning_rate,
            max_iter=self.max_iter,
            tol=self.tol,
            shuffle=True,
            verbose=self.verbose,
            n_jobs=self.n_jobs,
            n_iter_no_change=self.n_iter_no_change,
        )
        
    def _split_data(self, X, y, X_lagged):
        """Split data into training and validation sets based on validation_fraction."""
        if self.validation_fraction < 1.0:
            return train_test_split(X_lagged, y, test_size=self.validation_fraction,
                                    random_state=42, stratify=y)
        return X_lagged, y, None, None
    

    def _update_metrics(self, y_batch, y_pred, y_pred_proba, metrics, batch_idx):
        """Update metrics for progress bar and accuracy calculation."""
        try:
            # Calculate batch loss and accuracy metrics
            batch_loss = log_loss(y_batch, y_pred_proba)
            batch_accuracy = accuracy_score(y_batch, y_pred)
            batch_twa_accuracy = twa_score(y_batch, y_pred)

            metrics['loss'] = (
                metrics['loss'] * batch_idx + batch_loss
                ) / (batch_idx + 1)
            metrics['accuracy'] = (
                metrics['accuracy'] * batch_idx + batch_accuracy
                ) / (batch_idx + 1)
            
            metrics['TWA'] = (
                metrics['TWA'] * batch_idx + batch_twa_accuracy
            ) / (batch_idx + 1)
         
              
        except ValueError:
            pass  # Ignore errors in metrics calculation

        # Print current metrics if verbosity is set
        if self.verbose > 0: 
            print(
                f"loss: {metrics['loss']:.4f} - accuracy: {metrics['accuracy']:.4f}"
                f" - TWA: {metrics['TWA']:.4f}"
            )
 
    def _handle_early_stopping(self, val_loss):
        """Manage early stopping logic."""
        if val_loss < self.best_loss_ - self.tol:
            self._no_improvement_count = 0
            self.best_loss_ = val_loss
        else:
            self._no_improvement_count += 1
            
    def _train_epoch(
        self, X_train, y_train, X_val, y_val, y, 
        n_batches, metrics, epoch, bar=None
        ):
        """Train the model for one epoch."""
        
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        for batch_idx in range(n_batches):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, X_train.shape[0])
            X_batch = X_train_shuffled[start:end]
            y_batch = y_train_shuffled[start:end]

            if epoch == 0 and batch_idx == 0:
                self.linear_model_.partial_fit(
                    X_batch, y_batch, classes=np.unique(y))
            else:
                self.linear_model_.partial_fit(X_batch, y_batch)

            if batch_idx == n_batches - 1:
                self._evaluate_batch(
                    X_batch, y_batch, 
                    X_val, y_val, 
                    metrics, batch_idx
                )
            if bar is not None:
                bar.update(batch_idx + 1, epoch)
              

    def _evaluate_batch(self, X_batch, y_batch, X_val, y_val, metrics, batch_idx):
        """Evaluate the performance of the model on the current batch."""
        y_pred = self.linear_model_.predict(X_batch)
        y_pred_proba = self.linear_model_.predict_proba(X_batch)
        self._update_metrics(y_batch, y_pred, y_pred_proba, metrics, batch_idx)
        
        if X_val is not None:
            y_val_pred_proba = self.linear_model_.predict_proba(X_val)
            val_loss = log_loss(y_val, y_val_pred_proba)
            val_accuracy = accuracy_score(y_val, self.linear_model_.predict(X_val))
            
            metrics['val_loss'] = (
                metrics['accuracy'] * batch_idx + val_loss
                ) / (batch_idx + 1)
            
            metrics['val_accuracy'] = (
                metrics['val_accuracy'] * batch_idx + val_accuracy
            ) / (batch_idx + 1)
            # metrics['val_loss'] = val_loss
            # metrics['val_accuracy'] = val_accuracy
            
            
            if self.verbose > 1:
                print(f"val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f}")
                
            if self.early_stopping:
                self._handle_early_stopping(val_loss)

    def predict_proba(self, X):
        """
        Predict class probabilities for each input sample in `X`.
    
        This method generates class probabilities by applying the nonlinear 
        input transformation, creating lagged features, and then using the 
        logistic regression model’s decision function. A logistic sigmoid 
        transformation is applied to the decision function output, yielding 
        class probabilities.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict probabilities for, where `n_samples` is the 
            number of samples and `n_features` is the number of features.
    
        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The first column 
            represents the probability of class 0, and the second column 
            represents the probability of class 1.
    
        Concept
        -------
        - Sigmoid Transformation:
          The logistic sigmoid function is applied to the decision function 
          output to convert it into probabilities:
    
          .. math::
              \\hat{p}(y = 1 | X) = \\sigma(f(X_{lagged})) = 
                                    \\frac{1}{1 + e^{-f(X_{lagged})}}
    
          where :math:`\\sigma` is the sigmoid function, and :math:`f(X_{lagged})` 
          is the output of the decision function.
    
        Notes
        -----
        - This method requires that the model is already fitted.
        - Lagged features allow the model to incorporate historical dependencies 
          in probability estimation.
    
        Examples
        --------
        >>> from gofast.estimators.dynamic_system import HammersteinWienerClassifier
        >>> model = HammersteinWienerClassifier()
        >>> model.fit(X, y)
        >>> probabilities = model.predict_proba(X)
        >>> print("Class probabilities:", probabilities)
    
        See Also
        --------
        predict : Predict binary class labels for input samples.
        """
        if self.verbose > 0:
            print("Starting predict_proba method.")
    
        # Check if the model is fitted
        check_is_fitted(self, 'linear_model_')
    
        # Validate input data
        X = check_array(X)
    
        # Apply nonlinear input transformation
        X_transformed = self._apply_nonlinear_input(X)
    
        # Create lagged features for linear dynamic block
        X_lagged = self._create_lagged_features(X_transformed)
    
        # Get the decision function output
        y_linear = self.linear_model_.decision_function(X_lagged)
    
        # Apply nonlinear output transformation
        y_transformed = self._apply_nonlinear_output(y_linear)
    
        # Convert to probabilities
        if self.is_multilabel_:
            # For multilabel, apply sigmoid function
            y_pred_proba = activator(y_transformed, activation="sigmoid")
        else:
            if len(self.linear_model_.classes_) == 2:
                # Binary classification
                y_pred_proba = activator(y_transformed, activation="sigmoid")
                # Ensure the output has two columns
                y_pred_proba = np.hstack([1 - y_pred_proba, y_pred_proba])
            else:
                # Multiclass classification, apply softmax
                y_pred_proba = activator(y_transformed, activation="softmax")
    
        if self.verbose > 0:
            print("predict_proba method completed.")
    
        return y_pred_proba

    def predict(self, X):
        """
        Predict class labels for each input sample in `X`.
    
        This method generates class labels by first computing class 
        probabilities and then applying a threshold of 0.5 to determine 
        the predicted class. Samples with predicted probability greater 
        than or equal to 0.5 are classified as class 1, while the rest 
        are classified as class 0.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict, where `n_samples` is the number of samples 
            and `n_features` is the number of features.
    
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted binary class labels for each sample. The labels are 
            either 0 or 1, representing the two classes in binary 
            classification.
    
        Notes
        -----
        - Probability Thresholding:
          This method applies a threshold to the predicted probabilities 
          to determine class labels:
    
          .. math::
              \\hat{y} = \\begin{cases} 
                            1 & \\text{if } \\hat{p}(y = 1 | X) \\geq 0.5 \\\\
                            0 & \\text{otherwise}
                         \\end{cases}
    
          where :math:`\\hat{p}(y = 1 | X)` is the probability of class 1.
    
        Notes
        -----
        - This method requires `predict_proba` to be executed first.
        - A threshold of 0.5 is commonly used for binary classification tasks, 
          though this can be adjusted if needed.
    
        Examples
        --------
        >>> from gofast.estimators.dynamic_system import HammersteinWienerClassifier
        >>> model = HammersteinWienerClassifier()
        >>> model.fit(X, y)
        >>> labels = model.predict(X)
        >>> print("Predicted labels:", labels)
    
        See Also
        --------
        predict_proba : Predict class probabilities for input samples.
        """
        if self.verbose > 0:
            print("Starting predict method.")

        y_pred_proba = self.predict_proba(X)

        if self.is_multilabel_:
            # For multilabel, threshold probabilities at 0.5
            y_pred = (y_pred_proba >= 0.5).astype(int)
        else:
            # For binary and multiclass, take argmax
            y_pred = np.argmax(y_pred_proba, axis=1)

        if self.verbose > 0:
            print("Predict method completed.")

        return y_pred

    def _compute_loss(self, y_true, y_pred_proba):
        """
        Computes the classification loss based on the specified loss function.
        This method calculates the average error between the true labels 
        (`y_true`) and predicted probabilities (`y_pred_proba`) by applying 
        either the cross-entropy or time-weighted cross-entropy loss function. 
        The latter assigns greater penalty to recent errors, which can be 
        essential in dynamic classification systems where recent predictions 
        carry higher importance.
    
        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True binary labels for each sample, where values are either 0 or 1.
            These labels are used as the ground truth for computing the loss.
    
        y_pred_proba : array-like of shape (n_samples,)
            Predicted probabilities for each sample. These values range between 
            0 and 1, representing the probability of the positive class. Values 
            are clipped between a small epsilon (1e-15) and 1 - epsilon to 
            prevent computational issues when taking the log of zero.
    
        Returns
        -------
        loss : float
            Computed loss value based on the specified loss function. The loss 
            is averaged across all samples.
    
        Notes
        -----
        - Cross-Entropy Loss:
          The cross-entropy loss measures the error between the true and 
          predicted probabilities. The formula is:
    
          .. math::
              L_{CE} = - \\frac{1}{n} \\sum_{i=1}^{n} 
                      \\left[ y_i \\log(\\hat{y}_i) 
                      + (1 - y_i) \\log(1 - \\hat{y}_i) \\right]
    
          where :math:`y_i` is the true label, and :math:`\\hat{y}_i` is the 
          predicted probability for sample :math:`i`.
    
        - Time-Weighted Cross-Entropy Loss:
          In time-weighted loss, a weight vector is applied, emphasizing recent 
          samples. The formula becomes:
    
          .. math::
              L_{TWCE} = - \\frac{1}{n} \\sum_{i=1}^{n} 
                         w_i \\left[ y_i \\log(\\hat{y}_i) 
                         + (1 - y_i) \\log(1 - \\hat{y}_i) \\right]
    
          where :math:`w_i` are time weights calculated by `_compute_time_weights`.
    

        - Cross-entropy is commonly used for binary classification problems.
        - Time-weighted cross-entropy can improve performance in time-sensitive 
          applications by focusing on recent prediction accuracy.

        """
        if self.verbose > 0:
            print(f"Computing loss using {self.loss} loss function.")

        # Clip probabilities to prevent log of zero
        y_pred_proba = np.clip(y_pred_proba, self.epsilon, 1 - self.epsilon)

        # Compute loss using sklearn's log_loss
        if self.loss == "cross_entropy":
            loss = log_loss(y_true, y_pred_proba)
        elif self.loss == "time_weighted_cross_entropy":
            weights = self._compute_time_weights(len(y_true))
            loss = log_loss(y_true, y_pred_proba, sample_weight=weights)
        else:
            raise ValueError("Unsupported loss function.")

        if self.verbose > 0:
            print(f"Computed loss: {loss}")

        return loss

    def _compute_time_weights(self, n):
        """
        Computes time-based weights to prioritize recent predictions. The 
        weight scheme is based on the `time_weighting` parameter, which defines 
        how the weights are distributed across samples. Supported options 
        include `linear`, `exponential`, and `inverse` weighting [1]_.
    
        Parameters
        ----------
        n : int
            The number of samples or time points for which weights are computed.
    
        Returns
        -------
        weights : array-like of shape (n,)
            Computed time weights, normalized to sum to one for consistent 
            scaling across different weighting schemes.
    
        Notes
        -----
        Time weights are calculated as follows, based on the `time_weighting` 
        method:
    
        - Linear Weighting:
          The weights increase linearly, giving a low weight to the first 
          observation and the highest weight to the last.
    
          .. math::
              w_i = \\frac{0.1 + \\frac{i}{n-1}}{1.0}
    
        - Exponential Weighting:
          Weights follow an exponential increase, prioritizing recent samples 
          more aggressively.
    
          .. math::
              w_i = \\frac{e^{\\frac{i}{n}} - 1}{e^{1} - 1}
    
        - Inverse Weighting:
          The weights decrease inversely with the time index, placing 
          more importance on recent samples.
    
          .. math::
              w_i = \\frac{1}{i}
    
        - Weights are normalized to sum to one across all samples.
        - Different weighting schemes can be used to fine-tune the model's 
          sensitivity to time-based dependencies.

        """
        if self.verbose > 0:
            print(f"Computing time weights using {self.time_weighting} method.")

        # Compute weights based on the selected method
        if self.time_weighting == "linear":
            weights = np.linspace(0.1, 1.0, n)
        elif self.time_weighting == "exponential":
            weights = np.exp(np.linspace(0, 1, n)) - 1
            weights /= weights.max()  # Normalize to [0, 1]
        elif self.time_weighting == "inverse":
            weights = 1 / np.arange(1, n + 1)
            weights /= weights.max()  # Normalize to [0, 1]
        else:
            # Default to equal weights if unrecognized weighting scheme
            weights = np.ones(n)

        if self.verbose > 0:
            print(f"Time weights: {weights}")

        return weights            
            

class HammersteinWienerRegressor(BaseHammersteinWiener, RegressorMixin):
    """
    Hammerstein-Wiener model for regression tasks.

    The Hammerstein-Wiener model is a block-oriented model used to
    represent nonlinear dynamic systems. It consists of a cascade of
    a static nonlinear input block, a linear dynamic block, and a
    static nonlinear output block. This model is suitable for capturing
    systems with input and output nonlinearities surrounding a linear
    dynamic system.

    The general structure of the Hammerstein-Wiener model is as follows:

    .. math::
        y(t) = f_{\\text{out}}\\left( L\\left( f_{\\text{in}}\\left( x(t) \\right) \\right) \\right)

    where:

    - :math:`x(t)` is the input vector at time :math:`t`.
    - :math:`f_{\\text{in}}` is the nonlinear input function.
    - :math:`L` is the linear dynamic block.
    - :math:`f_{\\text{out}}` is the nonlinear output function.
    - :math:`y(t)` is the output at time :math:`t`.

    See more in :ref:`User Guide`.
    
    Parameters
    ----------
    nonlinear_input_estimator : estimator object or None, default=None
        Estimator for the nonlinear input function :math:`f_{\\text{in}}`.
        This should be an object that implements ``fit`` and either
        ``transform`` or ``predict`` methods. If ``None``, no nonlinear
        transformation is applied to the input.

    nonlinear_output_estimator : estimator object or None, default=None
        Estimator for the nonlinear output function :math:`f_{\\text{out}}`.
        This should be an object that implements ``fit`` and either
        ``transform`` or ``predict`` methods. If ``None``, no nonlinear
        transformation is applied to the output.

    p : int, default=1
        Order of the linear dynamic block. This specifies the number of
        lagged inputs to include in the linear dynamic model.

    loss : {'mse', 'mae', 'huber', 'time_weighted_mse'}, default='mse'
        Loss function to use for evaluating the model performance.

        - ``'mse'``: Mean Squared Error.
        - ``'mae'``: Mean Absolute Error.
        - ``'huber'``: Huber loss.
        - ``'time_weighted_mse'``: Time-weighted Mean Squared Error, giving
          more importance to recent errors.

    output_scale : tuple or None, default=None
        Tuple specifying the minimum and maximum values to scale the
        output predictions. If ``None``, no scaling is applied.

    time_weighting : {'linear', 'exponential', 'inverse'}, default='linear'
        Method for computing time weights when using time-weighted loss
        functions.

        - ``'linear'``: Linearly increasing weights over time.
        - ``'exponential'``: Exponentially increasing weights over time.
        - ``'inverse'``: Inversely decreasing weights over time.

    feature_engineering : {'auto'}, default='auto'
        Method for feature engineering. Currently, only ``'auto'`` is
        supported.
 
    delta : float, default=1.0
        Threshold parameter for the Huber loss function. It controls the 
        point at which the Huber loss transitions from a quadratic function 
        (for small residuals) to a linear function (for large residuals), 
        thus reducing the impact of outliers. Setting `delta` to a higher 
        value makes the model less sensitive to outliers.

    epsilon : float, default=1e-8
        A small constant added to prevent division by zero or other 
        numerical instabilities in calculations. It ensures stability by 
        limiting values that approach zero, allowing for more robust 
        computations.
        
    batch_size : int, default=32
        The number of samples per batch during training. Smaller batch sizes 
        may provide more granular updates to the model, while larger batch 
        sizes can result in faster training times but require more memory.
        
    optimizer : {'sgd', 'adam', 'adagrad'}, default='adam'
        The optimization algorithm to use for training the linear dynamic 
        block. The choice of optimizer affects the model's convergence 
        rate and stability:
        
        - ``'sgd'``: Stochastic Gradient Descent, a basic optimizer that 
          updates weights based on a single sample, providing faster 
          updates but potentially noisier convergence.
        - ``'adam'``: Adaptive Moment Estimation, an advanced optimizer 
          that adjusts learning rates for each parameter dynamically, 
          improving convergence speed and stability.
        - ``'adagrad'``: Adaptive Gradient Algorithm, which adapts learning 
          rates for each parameter individually, helpful for sparse data 
          but may decay learning rates too aggressively.
          
    learning_rate : float, default=0.001
        The initial learning rate applied by the chosen optimizer. A higher 
        learning rate can speed up training but risks overshooting minima, 
        while a lower rate can lead to slower convergence but more precise 
        optimization. Recommended values often range from 0.0001 to 0.01 
        depending on model complexity.
    
    max_iter : int, default=1000
        The maximum number of training iterations (epochs) for model 
        training. Each iteration represents one pass through the entire 
        dataset. Increasing this value allows the model more time to learn 
        patterns but risks overfitting if too large.

    tol : float, default=1e-3
        The tolerance for the optimization criterion. This value sets a 
        threshold for determining when the optimization process should stop. 
        If the change in the loss function or model parameters falls below 
        this tolerance level, training will be halted. A smaller `tol` value 
        can lead to more precise convergence but may require more iterations, 
        while a larger value can speed up the process but risks not finding 
        the optimal solution.
    
    early_stopping : bool, default=False
        A flag that indicates whether to enable early stopping during training. 
        When set to `True`, the training process will terminate if the 
        performance of the model does not improve for a specified number of 
        iterations, as defined by `n_iter_no_change`. This feature helps 
        prevent overfitting and saves computational resources. However, it 
        should be used judiciously, as premature stopping may lead to 
        suboptimal model performance.
    
    validation_fraction : float, default=0.1
        The proportion of the training data to set aside for validation. This 
        parameter specifies the fraction of the training dataset to be used 
        for validating the model's performance during training. A common 
        choice is to set this value between 0.1 and 0.2. Using a validation 
        set helps monitor the model's generalization ability and enables the 
        application of early stopping if the validation performance does not 
        improve.
    
    n_iter_no_change : int, default=5
        The number of iterations with no improvement on the training 
        score after which the training will be stopped early. This 
        parameter helps prevent overfitting by terminating the training 
        process when the model's performance ceases to improve, thus 
        saving computational resources. Setting this value too low 
        may result in premature stopping, while a higher value could 
        lead to unnecessary computation. Typical values range from 
        5 to 20, depending on the model's convergence behavior 
        and the complexity of the dataset.
        
    n_jobs : int or None, default=None
        The number of jobs to run in parallel. ``None`` means 1 unless in
        a ``joblib.parallel_backend`` context.

    verbose : int, default=0
        Determines the level of verbosity for logging messages during 
        the fitting and prediction phases. Setting this value to greater 
        than 0 will enable message outputs, giving visibility into the 
        training process. If set to ``None``, all output messages will be 
        suppressed. 


    Attributes
    ----------
    linear_coefficients_ : ndarray of shape (n_features * p,)
        Coefficients of the linear dynamic block. These coefficients 
        represent the linear relationship between the lagged features 
        and the target values and are computed during the fit process.
    
    initial_loss_ : float
        Initial loss computed on the training data after fitting. This 
        value represents the performance of the model immediately after 
        training and provides a baseline loss value.
    
    Methods
    -------
    fit(X, y)
        Fit the Hammerstein-Wiener regressor to the provided training 
        data (`X`, `y`). Applies input transformations, creates lagged 
        features, fits the linear dynamic block, and optionally applies 
        a nonlinear output transformation.
    
    predict(X)
        Predict values for the input samples (`X`) using the fitted 
        Hammerstein-Wiener regressor. Generates predictions by applying 
        transformations and linear dynamics, followed by optional output 
        scaling if specified.

    See Also
    --------
    HammersteinWienerClassifier : Hammerstein-Wiener model for classification tasks.

    Notes
    -----
    The Hammerstein-Wiener model combines static nonlinear blocks with
    a linear dynamic block to model complex systems that exhibit both
    dynamic and nonlinear behaviors [1]_.
    
    `HammersteinWienerRegressor` is especially suited for applications that 
    require detailed analysis and prediction of systems where the output 
    behavior is influenced by historical input and output data. Its ability to 
    model both linear and nonlinear dynamics makes it indispensable in advanced 
    fields like adaptive control, nonlinear system analysis, and complex signal 
    processing, providing insights and predictive capabilities critical for 
    effective system management.

    References
    ----------
    .. [1] Schoukens, J., & Ljung, L. (2019). Nonlinear System
           Identification: A User-Oriented Roadmap. IEEE Control
           Systems Magazine, 39(6), 28-99.

    Examples
    --------
    >>> from gofast.estimators.dynamic_system import HammersteinWienerRegressor
    >>> from sklearn.datasets import make_regression
    >>> import numpy as np
    >>> # Generate synthetic data
    >>> X, y = make_regression(n_samples=200, n_features=5, noise=0.1, random_state=42)
    >>> y += 0.5 * np.sin(X[:, 0])  # Introduce nonlinearity
    >>> # Instantiate the regressor
    >>> model = HammersteinWienerRegressor(p=2, verbose=1)
    >>> # Fit the model
    >>> model.fit(X, y)
    Starting HammersteinWienerRegressor fit method.
    Applying nonlinear input transformation.
    No nonlinear input estimator provided; using original X.
    Creating lagged features.
    Creating lag 1 features.
    Creating lag 2 features.
    Lagged features shape: (200, 10)
    Calculating linear coefficients.
    Applying linear dynamic block.
    Applying nonlinear output transformation.
    No nonlinear output estimator provided; using linear output.
    Starting HammersteinWienerRegressor predict method.
    Applying nonlinear input transformation.
    No nonlinear input estimator provided; using original X.
    Creating lagged features.
    Creating lag 1 features.
    Creating lag 2 features.
    Lagged features shape: (200, 10)
    Applying linear dynamic block.
    Applying nonlinear output transformation.
    No nonlinear output estimator provided; using linear output.
    Computed loss: 94.123456
    Fit method completed.
    >>> # Predict
    >>> y_pred = model.predict(X)
    Starting HammersteinWienerRegressor predict method.
    Applying nonlinear input transformation.
    No nonlinear input estimator provided; using original X.
    Creating lagged features.
    Creating lag 1 features.
    Creating lag 2 features.
    Lagged features shape: (200, 10)
    Applying linear dynamic block.
    Applying nonlinear output transformation.
    No nonlinear output estimator provided; using linear output.
    Predict method completed.
    >>> # Evaluate
    >>> from sklearn.metrics import mean_squared_error
    >>> mse = mean_squared_error(y, y_pred)
    >>> print(f"Mean Squared Error: {mse:.2f}")
    Mean Squared Error: 94.12
    """
    _parameter_constraints: dict = {
        **BaseHammersteinWiener._parameter_constraints,
        "loss": [StrOptions({"mse", "mae", "huber", "time_weighted_mse"})],
        "output_scale": [None, tuple],
        "time_weighting": [StrOptions({"linear", "exponential", "inverse"}), None],
        "delta": [Interval(Real, 0, None, closed='left')],
        "epsilon": [Interval(Real, 1e-15, None, closed='left')],
        "batch_size": [Interval(Integral, 1, None, closed='left')],
        "optimizer": [StrOptions({"sgd", "adam", "adagrad"})],
        "learning_rate": [Interval(Real, 0, None, closed='neither')],
        "max_iter": [Interval(Integral, 1, None, closed='left')],
        "tol": [Interval(Real, 0, None, closed='neither')],
        "early_stopping": [bool],
        "validation_fraction": [Interval(Real, 0, 1, closed='both')],
        "n_iter_no_change": [Interval(Integral, 1, None, closed='left')],
        "verbose": [Interval(Integral, 0, None, closed='left'), None]  
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
        batch_size=32,
        optimizer='adam',
        learning_rate=0.001,
        max_iter=1000,
        tol=1e-3,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        n_jobs=None,
        verbose=0
    ):
        super().__init__(
            nonlinear_input_estimator=nonlinear_input_estimator,
            nonlinear_output_estimator=nonlinear_output_estimator,
            p=p,
            feature_engineering=feature_engineering,
            n_jobs=n_jobs,
            verbose=verbose
        )

        self.loss = loss
        self.output_scale = output_scale
        self.time_weighting = time_weighting
        self.delta = delta
        self.epsilon = epsilon
        self.batch_size = batch_size      
        self.optimizer = optimizer        
        self.learning_rate = learning_rate  
        self.max_iter = max_iter          
        self.tol = tol
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        

    def fit(self, X, y, **fit_params):
        """
        Fit the Hammerstein-Wiener regressor to the training data.
    
        This method transforms the input data with a nonlinear function, 
        constructs lagged features to capture temporal dependencies, and 
        then fits a linear model to approximate the relationship between 
        input and target values. If a nonlinear output transformation 
        function is specified, it is applied to the intermediate linear 
        predictions to refine the final model output.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples, where `n_samples` is the number of 
            samples and `n_features` is the number of features in each sample.
    
        y : array-like of shape (n_samples,)
            Target values for regression. These values represent the output 
            associated with each input sample in `X`.
    
        Returns
        -------
        self : object
            Returns the fitted `HammersteinWienerRegressor` instance.
    
        Notes
        -----
        - Nonlinear Input Transformation:
          This transformation allows the model to capture nonlinear effects 
          at the input level by applying a function :math:`f_{NL_{in}}` to 
          each feature vector :math:`x`:
    
          .. math::
              \\tilde{x} = f_{NL_{in}}(x)
    
          where :math:`\\tilde{x}` represents the transformed input feature 
          vector.
    
        - Linear Dynamic Block with Lagged Features:
          Lagged versions of :math:`\\tilde{x}` are generated to capture 
          dynamic behavior over time. The linear relationship is then 
          estimated by solving for :math:`\\beta` in the equation:
    
          .. math::
              y_{linear} = X_{lagged} \\beta + b
    
          where :math:`X_{lagged}` represents the matrix of lagged features, 
          :math:`\\beta` is the vector of coefficients, and :math:`b` is the 
          intercept.
    
        - Nonlinear Output Transformation:
          If a nonlinear output transformation function :math:`f_{NL_{out}}` 
          is provided, it is applied to :math:`y_{linear}` to capture 
          additional nonlinear effects in the output:
    
          .. math::
              \\hat{y} = f_{NL_{out}}(y_{linear})
    
        Notes
        -----
        - This method solves for the linear coefficients using the pseudo-inverse.
        - The `fit` method computes the initial loss to assess the model's 
          performance after training.
    
        Examples
        --------
        >>> from gofast.estimators.dynamic_system import HammersteinWienerRegressor
        >>> model = HammersteinWienerRegressor()
        >>> X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        >>> y = np.array([1.0, 2.0, 3.0])
        >>> model.fit(X, y)
        >>> print("Initial loss:", model.initial_loss_)
    
        See Also
        --------
        predict : Predict values for input data.
    
        References
        ----------
        .. [1] Ljung, L. (1999). System Identification: Theory for the User. 
               Prentice Hall.
        """
        if self.verbose > 0:
            print("Starting HammersteinWienerRegressor fit method.")

        # Initialize private attributes
        # Loss should start at infinity, to be minimized.
        # Validation loss should also start at infinity.
        # # Initialize PSS at infinity, reflecting potential instability., 
        metrics = {
            'loss': float('inf'),      
            'PSS': float('inf'),               
            'val_loss': float("inf"),  
            'val_PSS': float('inf'),            
        }

        self.best_loss_ = np.inf if self.early_stopping else None
        self._no_improvement_count = 0

        # Validate parameters and input data
        self._validate_params()
        X, y = self._validate_input_data(X, y)
        X, y = check_X_y(X, y, multi_output=True)
        X_transformed = self._apply_nonlinear_input(X, y)
        X_lagged = self._create_lagged_features(X_transformed)

        if self.verbose > 0:
            print("Fitting linear model with batch training.")

        # Initialize the SGDRegressor
        self._initialize_model()
  
        # Split data into training and validation sets
        X_train, X_val, y_train,  y_val = self._split_data(X_lagged, y)
 
        n_samples = X_lagged.shape[0]
  
        self.batch_size = validate_batch_size(self.batch_size, n_samples)
        n_batches = max(1, int(np.floor(n_samples / self.batch_size)))

        # Early stopping initialization and Track best loss
        self.best_loss_ = np.inf if self.early_stopping else None
        self._no_improvement_count = 0
        
        if self.verbose == 0:
            with TrainingProgressBar(
                    epochs= self.max_iter, steps_per_epoch= n_batches, 
                    metrics=metrics, ) as progress_bar:
                for epoch in range(self.max_iter):
                    print(f"Epoch {epoch + 1 }/{self.max_iter}")
      
                    self._train_epoch(X_train, y_train, 
                        X_val, y_val, n_batches, metrics, epoch, 
                        bar=progress_bar, 
                    )
                    print("\n")  
                    if self.early_stopping and ( 
                            self._no_improvement_count >= self.n_iter_no_change
                        ):
                        print(f"Early stopping triggered after {epoch + 1} epochs.")
                        break
        else:
            for epoch in range(self.max_iter):
                if self.verbose > 0: 
                    print(f"Epoch {epoch + 1}/{self.max_iter}")
                
                self._train_epoch(X_train, y_train, 
                    X_val, y_val, n_batches, metrics, epoch
                )

                if self.early_stopping and ( 
                        self._no_improvement_count >= self.n_iter_no_change
                        ):
                    if self.verbose > 0: 
                        print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break
                
        y_linear = self.linear_model_.predict(X_lagged)
        self._apply_nonlinear_output(y_linear, y)

        y_pred_initial = self.predict(X)
        self.initial_loss_ = self._compute_loss(y, y_pred_initial)

        if self.verbose > 0:
            print(f"Initial loss: {self.initial_loss_}")
            print("Fit method completed.")

        return self
    
    def _initialize_model(self):
        """Initialize SGDRegressor for the linear dynamic block."""
        learning_rate_type = self._get_learning_rate_type()
        self.loss_function_ = self._get_loss_function()

        # Initialize the SGDRegressor
        self.linear_model_ = SGDRegressor(
            loss=self.loss_function_ ,
            learning_rate=learning_rate_type,
            eta0=self.learning_rate,
            max_iter=self.max_iter,
            tol=self.tol,
            shuffle=True,
            verbose=self.verbose,
            epsilon=self.delta,
            random_state=None
        )

    def _split_data(self, X_lagged, y):
        """Split data into training and validation sets based on validation_fraction."""
        if self.validation_fraction < 1.0:
            return train_test_split(X_lagged, y, test_size=self.validation_fraction,
                                    random_state=42)
        return X_lagged, y, None, None
    
    def _train_epoch(
        self, X_train, y_train, X_val, y_val, 
        n_batches, metrics, epoch, bar=None
        ):
        """Train the model for one epoch."""
   
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        print("nbatches =", n_batches)
        for batch_idx in range(n_batches):

            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, X_train.shape[0])
            X_batch = X_train_shuffled[start:end]
            y_batch = y_train_shuffled[start:end]
            # # Check if the batch is empty
            # if X_batch.size == 0 or y_batch.size == 0:
            #     continue  # Skip the iteration or handle it accordingly
            
            if self.verbose > 2:
                print(f"Batch {batch_idx + 1}/{n_batches}:"
                      f" X_batch shape {X_batch.shape},"
                      f" y_batch shape {y_batch.shape}")

            # Partial fit on the batch
            try: 
                self.linear_model_.partial_fit(X_batch, y_batch)
    
                if batch_idx == n_batches - 1:
                    self._evaluate_batch(
                        X_batch, y_batch, 
                        X_val, y_val, 
                        metrics, batch_idx
                    )
            except: pass 
            finally:
                if bar is not None:
                    bar.update(batch_idx + 1, epoch)

    def _get_learning_rate_type(self):
        return {
            'sgd': 'optimal',
            'adam': 'adaptive',
            'adagrad': 'invscaling'
        }.get(self.optimizer, 'optimal')

    def _get_loss_function(self):
        return {
            "mse": 'squared_error',
            "mae": 'epsilon_insensitive',
            "huber": 'huber'
        }.get(self.loss, 'squared_error')


    def _update_metrics(self, y_batch, y_pred, metrics, batch_idx):
        """Update metrics for progress bar and accuracy calculation."""
        try:
            # Calculate batch loss and accuracy metrics
            batch_loss = self._compute_pred_loss(y_batch, y_pred)
            batch_pss = prediction_stability_score( y_pred)
            metrics['loss'] = batch_loss
            metrics['PSS'] = batch_pss
            
            metrics['loss'] = (
                metrics['loss'] * batch_idx + batch_loss
                ) / (batch_idx + 1)
            metrics['PSS'] = (
                metrics['PSS'] * batch_idx + batch_pss
                ) / (batch_idx + 1)
            
        except ValueError:
            pass  # Ignore errors in metrics calculation
   
        # Print current metrics if verbosity is set
        if self.verbose > 0: 
            print(
                f"loss: {metrics['loss']:.4f} - PSS: {metrics['PSS']:.4f}"
            )
    def _evaluate_batch(self, X_batch, y_batch, X_val, y_val, metrics, batch_idx):
        """Evaluate the performance of the model on the current batch."""
        y_pred = self.linear_model_.predict(X_batch)
        self._update_metrics(y_batch, y_pred, metrics, batch_idx)
        
        if X_val is not None:
            y_val_pred = self.linear_model_.predict(X_val)
            val_loss = self._compute_pred_loss(y_val, y_val_pred)
            val_pss = prediction_stability_score(y_val_pred)
            
            metrics['val_loss'] = (
                metrics['val_loss'] * batch_idx + val_loss
                ) / (batch_idx + 1)
            metrics['val_PSS'] = (
                metrics['val_PSS'] * batch_idx + val_pss
                ) / (batch_idx + 1)
            
      
            if self.verbose > 1:
                print(f"val_loss: {val_loss:.4f} - val_PSS: {val_pss:.4f}")
                
            if self.early_stopping:
                self._handle_early_stopping(val_loss)
    
    def _handle_early_stopping(self, val_loss):
        """Manage early stopping logic."""
        if val_loss < self.best_loss_ - self.tol:
            self._no_improvement_count = 0
            self.best_loss_ = val_loss
        else:
            self._no_improvement_count += 1
    
    def _compute_pred_loss(self, y_true, y_pred): 
        """ Compute loss value from batch prediction"""
        if self.loss_function_ =='mae': 
            loss = mean_absolute_error(y_true, y_pred)
        else: 
            loss = mean_squared_error(y_true, y_pred)
            
        return loss 
            
    def predict(self, X):
        """
        Predict using the Hammerstein-Wiener regressor.
    
        This method predicts output values by applying the nonlinear input 
        transformation, creating lagged features, and passing them through 
        the linear dynamic block. If a nonlinear output transformation is 
        provided, it is applied to the intermediate predictions, followed 
        by an optional scaling to constrain the output range.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict, where `n_samples` is the number of samples 
            and `n_features` is the number of features in each sample.
    
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values for each input sample, potentially scaled to 
            the specified output range.
    
        Notes
        -----
        - Nonlinear Input Transformation:
          Similar to `fit`, this method applies the transformation 
          :math:`f_{NL_{in}}` to the input:
    
          .. math::
              \\tilde{x} = f_{NL_{in}}(x)
    
        - Lagged Features and Linear Dynamic Block:
          The lagged features are created from :math:`\\tilde{x}` and passed 
          through the linear dynamic block, yielding:
    
          .. math::
              y_{linear} = X_{lagged} \\beta + b
    
        - Nonlinear Output Transformation:
          If provided, the output function :math:`f_{NL_{out}}` is applied to 
          :math:`y_{linear}`, and scaling is optionally applied to ensure 
          predictions fall within a specified range:
    
          .. math::
              \\hat{y} = \\text{scale}(f_{NL_{out}}(y_{linear}))

        - The `predict` method requires that the model is already fitted.
        - Output scaling is applied only if specified in `output_scale`.
    
        Examples
        --------
        >>> from gofast.estimators.dynamic_system import HammersteinWienerRegressor
        >>> model = HammersteinWienerRegressor()
        >>> X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        >>> model.fit(X, y)
        >>> predictions = model.predict(X)
        >>> print("Predictions:", predictions)
    
        See Also
        --------
        fit : Fit the Hammerstein-Wiener regressor to training data.
        """

        if self.verbose > 0:
            print("Starting HammersteinWienerRegressor predict method.")

        # Ensure the model is fitted before making predictions
        check_is_fitted(self, 'linear_model_')

        # Validate input data
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

    def _compute_loss(self, y_true, y_pred):
        """
        Compute the specified loss function between true and predicted values.
    
        This method calculates the average error between the true values 
        (`y_true`) and the predicted values (`y_pred`) using the specified 
        loss function, which can be Mean Squared Error (MSE), Mean Absolute 
        Error (MAE), Huber loss, or a time-weighted MSE. The time-weighted 
        MSE option uses weights that emphasize recent observations.
    
        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True values for each sample. These values are the ground truth 
            for calculating the error between predicted and actual values.
    
        y_pred : array-like of shape (n_samples,)
            Predicted values for each sample as generated by the model.
    
        Returns
        -------
        loss : float
            Computed loss value based on the specified loss function, averaged 
            across all samples.
            
        Notes
        -----
        - Mean Squared Error (MSE):
          The MSE is calculated as the mean of the squared differences between 
          the true and predicted values. It is given by:
    
          .. math::
              L_{MSE} = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2
    
          where :math:`y_i` is the true value and :math:`\\hat{y}_i` is the 
          predicted value for each sample :math:`i`.
    
        - Mean Absolute Error (MAE):
          The MAE is the mean of the absolute differences between true and 
          predicted values:
    
          .. math::
              L_{MAE} = \\frac{1}{n} \\sum_{i=1}^{n} |y_i - \\hat{y}_i|
    
        - Huber Loss:
          Huber loss is less sensitive to outliers than MSE, defined by a 
          threshold :math:`\\delta`:
    
          .. math::
              L_{Huber} = \\frac{1}{n} \\sum_{i=1}^{n} \\begin{cases} 
                            0.5 (y_i - \\hat{y}_i)^2 & \\text{if } |y_i - 
                            \\hat{y}_i| < \\delta \\\\
                            \\delta (|y_i - \\hat{y}_i| - 0.5 \\delta) & 
                            \\text{otherwise}
                         \\end{cases}
    
        - Time-Weighted MSE:
          A weight vector :math:`w_i` is applied to each squared error term 
          to emphasize recent observations:
    
          .. math::
              L_{TW-MSE} = \\frac{1}{n} \\sum_{i=1}^{n} w_i (y_i - \\hat{y}_i)^2
    

        - Huber loss is recommended when outliers are present in the data.
        - Time-weighted MSE is useful in dynamic systems where recent 
          predictions are more important.
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
            residual = y_true - y_pred
            loss = np.mean(np.where(
                np.abs(residual) < self.delta,
                0.5 * residual ** 2,
                self.delta * (np.abs(residual) - 0.5 * self.delta)
            ))

        elif self.loss == "time_weighted_mse":
            weights = self._compute_time_weights(len(y_true))
            # weights shape: (n_samples,)
            # weights need to be expanded to (n_samples, n_outputs)
            # if y_true is multi-output
            if y_true.ndim > 1:
                weights = weights[:, np.newaxis]  # Shape (n_samples, 1)
            loss = np.mean(weights * (y_true - y_pred) ** 2)

        if self.verbose > 0:
            print(f"Computed loss: {loss}")

        return loss

    def _compute_time_weights(self, n):
        """
        Compute time-based weights for emphasizing recent observations.
    
        This method generates a vector of weights for each sample based on 
        the specified time-weighting method. Time weights allow the model to 
        focus more on recent data points, which is particularly useful in 
        dynamic regression contexts.
    
        Parameters
        ----------
        n : int
            The number of samples for which weights need to be computed.
    
        Returns
        -------
        weights : array-like of shape (n,)
            Time weights for each sample, scaled to sum to 1 for consistent 
            application across various weighting methods.
    
        Notes
        -----
        Time weights are computed as follows depending on the weighting method:
    
        - Linear Weighting:
          Weights increase linearly, with the smallest weight assigned to the 
          earliest sample and the largest to the most recent:
    
          .. math::
              w_i = \\frac{0.1 + \\frac{i}{n - 1}}{1.0}
    
        - Exponential Weighting:
          Exponential weights place even greater importance on recent samples:
    
          .. math::
              w_i = \\frac{e^{\\frac{i}{n}} - 1}{e^{1} - 1}
    
        - Inverse Weighting:
          Weights decrease inversely with the index, with greater emphasis 
          on the most recent samples:
    
          .. math::
              w_i = \\frac{1}{i}
    
        - Weights are normalized to ensure they sum to 1, preserving a stable 
          loss scale across samples.
        - Different weighting methods allow control over the model's temporal 
          sensitivity.

        """

        if self.verbose > 0:
            print(f"Computing time weights using {self.time_weighting} method.")

        # Compute linear, exponential, or inverse weights based on user input
        if self.time_weighting == "linear":
            weights = np.linspace(0.1, 1.0, n)

        elif self.time_weighting == "exponential":
            weights = np.exp(np.linspace(0, 1, n)) - 1
            weights /= weights.max()  # Normalize to [0, 1]

        elif self.time_weighting == "inverse":
            weights = 1 / np.arange(1, n + 1)
            weights /= weights.max()  # Normalize to [0, 1]

        # Default to equal weights if the method is not recognized
        else:
            # Equal weighting if method is unrecognized i.e. None
            weights = np.ones(n)

        if self.verbose > 0:
            print(f"Time weights: {weights}")

        return weights

    def _scale_output(self, y):
        """
        Scale output predictions to a specified range if defined.
    
        This method scales the output predictions to a specified range, 
        which is helpful when model predictions need to be constrained to 
        known limits, as in physical measurements or probability scores.
    
        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Model output predictions to be scaled to the specified range.
    
        Returns
        -------
        y_scaled : array-like of shape (n_samples,)
            Scaled predictions, with values transformed to the specified 
            range if `output_scale` is defined.
    
        Notes
        -----
        Scaling transforms the predicted values to the specified range 
        :math:`[y_{min}, y_{max}]` by first normalizing the predictions 
        to :math:`[0, 1]` and then applying the scale transformation:
    
        .. math::
            y_{scaled} = y_{min} + y * (y_{max} - y_{min})
    
        where :math:`y` represents the normalized prediction values.
    
        - Scaling is applied only if `output_scale` is defined. By default, 
          scaling range is set to None, meaning no scaling is applied.
        - This is especially useful in regression tasks where the output 
          must be within a certain range.
        """

        if self.output_scale is not None:
            if self.verbose > 0:
                print("Scaling output predictions.")

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
 
   