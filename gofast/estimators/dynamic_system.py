# -*- coding: utf-8 -*-

from __future__ import annotations 
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import LogisticRegression, LinearRegression 

try:from sklearn.utils import type_of_target
except: from ..tools.coreutils import type_of_target 

from ..exceptions import  EstimatorError 
from ..tools.coreutils import smart_format, is_iterable
from ..tools.validator import check_X_y, get_estimator_name, check_array 
from ..tools.validator import check_is_fitted, parameter_validator 

__all__= [
    "HammersteinWienerClassifier","HammersteinWienerRegressor",
    "EnsembleHWClassifier", "EnsembleHWRegressor",
    ]

class HammersteinWienerClassifier(BaseEstimator, ClassifierMixin):
    r"""
    Hammerstein-Wiener Classifier for Dynamic Classification Tasks.

    The Hammerstein-Wiener Classifier is designed for modeling and predicting 
    outcomes in dynamic systems where the response depends on past inputs. It 
    is particularly useful in scenarios where the system's behavior exhibits 
    both linear and nonlinear characteristics. This model uniquely combines 
    the properties of Hammerstein and Wiener systems to effectively capture 
    complex input-output relationships.

    The Hammerstein model component consists of a nonlinear input function 
    followed by a linear dynamic block, whereas the Wiener model component 
    features a linear dynamic block followed by a nonlinear output function. 
    This configuration allows the classifier to capture both linear dynamics 
    and nonlinear transformations inherent in the data.

    The Hammerstein-Wiener model for classification is mathematically 
    expressed as:

    .. math::
        y(t) = g_2\\left( \\text{{classifier}}\\left( \\sum_{i=1}^{n} g_1(X_{t-i}) \\right) \\right)

    where:
    - :math:`g_1` is the nonlinear function applied to inputs.
    - :math:`g_2` is the nonlinear output function, typically a logistic 
      function for classification.
    - 'classifier' denotes a linear model, such as logistic regression, 
      applied within the construct.
    - :math:`X_{t-i}` represents the input features at time step :math:`t-i`, 
      highlighting the memory effect essential for dynamic systems.

    The classifier is especially suited for time-series classification, 
    signal processing, and systems identification tasks where current outputs 
    are significantly influenced by historical inputs. It finds extensive 
    applications in fields like telecommunications, control systems, and 
    financial modeling, where understanding dynamic behaviors is crucial.

    Parameters
    ----------
    classifier : object (default=LogisticRegression())
        Linear classifier model for the dynamic block. Should support fit 
        and predict methods. For multi-class classification, it can be set to 
        use softmax regression (e.g., 
        LogisticRegression with multi_class='multinomial').

    nonlinearity_in : callable (default=np.tanh)
        Nonlinear function applied to inputs. It transforms the input data 
        before feeding it into the linear dynamic block.

    nonlinearity_out : callable (default=lambda x: 1 / (1 + np.exp(-x)))
        Nonlinear function applied to the output of the classifier. It models 
        the nonlinear transformation at the output stage.

    memory_depth : int (default=5)
        The number of past time steps to consider in the model. This parameter
        defines the 'memory' of the system, enabling the model to use past 
        information for current predictions.

    Attributes
    ----------
    fitted_ : bool
        Indicates whether the classifier has been fitted to the data.

    Examples
    --------
    >>> from gofast.estimators.dynamic_system import HammersteinWienerClassifier
    >>> from sklearn.linear_model import LogisticRegression
    >>> hw = HammersteinWienerClassifier(
    ...     classifier=LogisticRegression(),
    ...     nonlinearity_in=np.tanh,
    ...     nonlinearity_out=lambda x: 1 / (1 + np.exp(-x)),
    ...     memory_depth=5
    ... )
    >>> X, y = np.random.rand(100, 1), np.random.randint(0, 2, 100)
    >>> hw.fit(X, y)
    >>> y_pred = hw.predict(X)

    Notes
    -----
    The choice of nonlinear functions (g_1 and g_2) and the memory depth are 
    crucial in capturing the dynamics of the system accurately. They should 
    be chosen based on the specific characteristics of the data and the 
    underlying system behavior.

    References
    ----------
    - Hammerstein, A. (1930). Nichtlineare Systeme und Regelkreise.
    - Wiener, N. (1958). Nonlinear Problems in Random Theory.

    See Also
    --------
    LogisticRegression : Standard logistic regression classifier from 
       Scikit-Learn.
    TimeSeriesSplit : Time series cross-validator for Scikit-Learn.
    """


    def __init__(
        self, 
        classifier="LogisticRegression", 
        nonlinearity_in=np.tanh, 
        nonlinearity_out="sigmoid", 
        memory_depth=5
        ):
        self.classifier = classifier
        self.nonlinearity_in = nonlinearity_in
        self.nonlinearity_out = nonlinearity_out
        self.memory_depth = memory_depth

    def _preprocess_data(self, X):
        """
        Preprocess the input data by applying the input nonlinearity and
        incorporating memory depth.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        X_transformed : array-like
            The transformed input data.
        """
        X_transformed = self.nonlinearity_in(X)

        n_samples, n_features = X_transformed.shape
        X_lagged = np.zeros((n_samples - self.memory_depth, 
                             self.memory_depth * n_features))
        for i in range(self.memory_depth, n_samples):
            X_lagged[i - self.memory_depth, :] = ( 
                X_transformed[i - self.memory_depth:i, :].flatten()
                )

        return X_lagged

    def fit(self, X, y):
        """
        Fit the Hammerstein-Wiener model to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data input.
        y : array-like of shape (n_samples,)
            Target classification labels.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, estimator=get_estimator_name(self))
        
        if isinstance(self.nonlinearity_out, str):
            self.nonlinearity_out = parameter_validator(
                "nonlinearity_out", target_strs={"sigmoid", "softmax"}, 
                error_msg=(
                    "The 'nonlinearity_out' parameter must be either 'sigmoid'"
                    " or 'softmax', or it should be a callable object that"
                    " performs a custom transformation. Please ensure you"
                    " provide one of the supported strings or a valid callable."
                )
            )(self.nonlinearity_out)
            self.nonlinearity_out_name_=self.nonlinearity_out
            
        elif callable (self.nonlinearity_out): 
            self.nonlinearity_out_name_=self.nonlinearity_out.__name__ 
            
        if self.nonlinearity_out=='sigmoid': 
            self.nonlinearity_out = lambda x: 1 / (1 + np.exp(-x))
        elif self.nonlinearity_out =='softmax': 
            self.nonlinearity_out=lambda x: np.exp(x) / np.sum(np.exp(x), axis=0)
            
        if self.classifier=='LogisticRegression': 
            if type_of_target(y)=='binary': 
                self.classifier = LogisticRegression()
            else: 
                self.classifier=LogisticRegression(
                    multi_class='multinomial')
     
        X_lagged = self._preprocess_data(X)
        y = y[self.memory_depth:]

        self.classifier.fit(X_lagged, y)
        self.fitted_ = True
        return self
    

    def predict(self, X):
        """
        Predict using the Hammerstein-Wiener model.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict for.
    
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted classification labels.
        """
        check_is_fitted(self, 'fitted_') 
        X = check_array(X)
        X_lagged = self._preprocess_data(X)
    
        y_linear = self.classifier.predict(X_lagged)
        y_pred = self.nonlinearity_out(y_linear)
        
        # Adjust for truncated samples
        # Generate default predictions for the first 'memory_depth' samples
        default_prediction = np.array([self.classifier.classes_[0]] * self.memory_depth)
        
        # Concatenate the default predictions with the actual predictions
        y_pred_full = np.concatenate((default_prediction, y_pred))
        
        # Get the predicted class based on the probability threshold of 0.5
        y_pred_full = np.where(y_pred_full >= 0.5, 1, 0)
    
        return y_pred_full

    def predict_proba(self, X):
        """
        Probability estimates for the Hammerstein-Wiener model.

        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict for.

        Returns
        -------
        proba : array-like of shape (n_samples, n_classes)
            Probability of the sample for each class in the model.
        """
        check_is_fitted (self, 'fitted_') 
        X = check_array(X)
        X_lagged = self._preprocess_data(X)

        # Get the probability estimates from the linear classifier
        proba_linear = self.classifier.predict_proba(X_lagged)

        # Apply the output nonlinearity (if necessary)
        return np.apply_along_axis(self.nonlinearity_out, 1, proba_linear)

class HammersteinWienerRegressor(BaseEstimator, RegressorMixin):
    r"""
    Hammerstein-Wiener Estimator for Nonlinear Dynamic System Identification.

    The Hammerstein-Wiener Estimator is designed to model dynamic systems 
    where outputs are nonlinear functions of both past inputs and outputs. 
    It combines a Hammerstein model (nonlinear input followed by linear dynamics)
    with a Wiener model (linear dynamics followed by nonlinear output), 
    making it exceptionally adept at capturing the complexities inherent in 
    nonlinear dynamic systems. This estimator is particularly valuable in 
    fields such as control systems, signal processing, and physiological 
    data analysis, where the behavior of systems often exhibits both linear 
    and nonlinear dynamics over time.

    Mathematical Formulation:
    The estimator's operation is mathematically represented as:

    .. math::
        y(t) = g_2\left( \sum_{i} a_i y(t-i) + \sum_{j} b_j g_1(u(t-j)) \right)

    where:
    - :math:`g_1` and :math:`g_2` are the nonlinear functions applied to the input 
      and output, respectively.
    - :math:`a_i` and :math:`b_j` are the coefficients representing the linear 
      dynamic components of the system.
    - :math:`u(t-j)` and :math:`y(t-i)` denote the system inputs and outputs at 
      various time steps, capturing the past influence on the current output.

    Parameters
    ----------
    linear_model : object
        A linear model for the dynamic block. This should be an estimator with fit 
        and predict methods (e.g., LinearRegression from scikit-learn). The choice 
        of linear model influences how the system's linear dynamics are captured.

    nonlinearity_in : callable (default=np.tanh)
        Nonlinear function applied to system inputs. This function should be chosen 
        based on the expected nonlinear behavior of the input data (e.g., np.tanh, 
        np.sin).

    nonlinearity_out : callable (default=np.tanh)
        Nonlinear function applied to the output of the linear dynamic block. It 
        shapes the final output of the system and should reflect the expected 
        output nonlinearity (e.g., np.tanh, logistic function).

    memory_depth : int (default=5)
        The number of past time steps considered in the model. This parameter is 
        crucial for capturing the memory effect in dynamic systems. A higher value 
        means more past data points are used, which can enhance model accuracy but 
        increase computational complexity.

    Attributes
    ----------
    fitted_ : bool
        Indicates whether the estimator has been fitted to data.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> from gofast.estimators.dynamic_system import HammersteinWienerRegressor
    >>> hw = HammersteinWienerRegressor(
    ...     nonlinearity_in=np.tanh,
    ...     nonlinearity_out=np.tanh,
    ...     linear_model=LinearRegression(),
    ...     memory_depth=5
    ... )
    >>> X, y = np.random.rand(100, 1), np.random.rand(100)
    >>> hw.fit(X, y)
    >>> y_pred = hw.predict(X)

    Notes
    -----
    Selecting appropriate nonlinear functions and memory depth is key to effectively 
    modeling the system. The estimator's performance can be significantly impacted 
    by these choices, and they should be tailored to the specific characteristics 
    of the data and the system being modeled.
    
    `HammersteinWienerRegressor` is especially suited for applications that 
    require detailed analysis and prediction of systems where the output behavior
    is influenced by historical input and output data. Its ability to model both
    linear and nonlinear dynamics makes it indispensable in advanced fields 
    like adaptive control, nonlinear system analysis, and complex signal 
    processing, providing insights and predictive capabilities critical for 
    effective system management.

    References
    ----------
    - Hammerstein, A. (1930). Nichtlineare Systeme und Regelkreise.
    - Wiener, N. (1958). Nonlinear Problems in Random Theory.
    - Narendra, K.S., and Parthasarathy, K. (1990). Identification and Control of 
      Dynamical Systems Using Neural Networks. IEEE Transactions on Neural Networks.

    See Also
    --------
    LinearRegression : Ordinary least squares Linear Regression.
    ARIMA : AutoRegressive Integrated Moving Average model for time-series 
         forecasting.
    """

    def __init__(self,
        linear_model="LinearRegression", 
        nonlinearity_in= np.tanh, 
        nonlinearity_out=np.tanh, 
        memory_depth=5
        ):
        self.nonlinearity_in = nonlinearity_in
        self.nonlinearity_out = nonlinearity_out
        self.linear_model = linear_model
        self.memory_depth = memory_depth

    def _preprocess_data(self, X):
        """
        Preprocess the input data by applying the input nonlinearity and
        incorporating memory depth.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        X_transformed : array-like
            The transformed input data.
        """
        # Apply the input nonlinearity
        X_transformed = self.nonlinearity_in(X)

        # Incorporate memory depth
        n_samples, n_features = X_transformed.shape
        X_lagged = np.zeros((n_samples - self.memory_depth, 
                             self.memory_depth * n_features))
        for i in range(self.memory_depth, n_samples):
            X_lagged[i - self.memory_depth, :] = ( 
                X_transformed[i - self.memory_depth:i, :].flatten()
                )

        return X_lagged
    
    def fit(self, X, y):
        """
        Fit the Hammerstein-Wiener model to the data.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data input.
        y : array-like of shape (n_samples,)
            Target values.
    
        Returns
        -------
        self : object
            Returns self, an instance of HammersteinWienerRegressor.
        """
        # Validate input arrays and ensure they meet the required 
        # shape and type specifications
        X, y = check_X_y(X, y, estimator=self)
        
        # Preprocess the data to include past information based on memory depth
        X_lagged = self._preprocess_data(X)
        
        # Adjust target array to align with the truncated input data due to memory depth
        y_adjusted = y[self.memory_depth:]
        
        # Validate if linear_model is instantiated or needs instantiation
        if isinstance(self.linear_model, str):
            if self.linear_model == "LinearRegression":
                self.linear_model = LinearRegression()
            else:
                raise ValueError(
                    "The specified linear_model name is not supported."
                    " Please provide an instance of a linear model.")
        
        # Ensure the linear_model has a fit method
        if not hasattr(self.linear_model, 'fit'):
            raise TypeError(
                f"The provided linear_model does not have a 'fit' method."
                f" Please ensure that {type(self.linear_model).__name__}"
                " is a valid estimator.")
        
        # Fit the linear model to the preprocessed data
        self.linear_model.fit(X_lagged, y_adjusted)
        
        # Set the fitted_ attribute to True to indicate that the model 
        # has been successfully fitted
        self.fitted_ = True
        
        return self
    
    def predict(self, X):
        """
        Predict using the Hammerstein-Wiener model.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict for.
    
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted values.
        """
        check_is_fitted(self, 'fitted_')
        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse=True,
            to_frame=False,
        )
    
        # Apply preprocessing to input data
        X_lagged = self._preprocess_data(X)
    
        # Predict using the linear model
        y_linear = self.linear_model.predict(X_lagged)
    
        # Apply the output nonlinearity if available
        if callable(self.nonlinearity_out):
            y_pred_transformed = self.nonlinearity_out(y_linear)
        else:
            y_pred_transformed = y_linear
    
        # Prepare final predictions array
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples)  # Initialize with zeros or another default value
    
        # Fill predictions from memory_depth onwards
        y_pred[self.memory_depth:] = y_pred_transformed
    
        # Generate default predictions for the first 'memory_depth' samples
        if self.memory_depth > 0:
            # You can customize this logic as needed. For example,
            # you could use a simple mean, median,
            # or the mode of `y_pred_transformed`, or even another model's predictions.
            default_prediction = np.mean(y_pred_transformed) if len(y_pred_transformed) > 0 else 0
            y_pred[:self.memory_depth] = default_prediction
    
        return y_pred

class EnsembleHWClassifier(BaseEstimator, ClassifierMixin):
    r"""
    Hammerstein-Wiener Ensemble Classifier.

    `EnsembleHWClassifier` combines the Hammerstein-Wiener model with ensemble 
    learning, effectively managing both linear and nonlinear dynamics within data. 
    It is particularly suited for dynamic systems where outputs depend on 
    historical inputs and outputs, making it ideal for applications in time-series
    forecasting, control systems, and complex scenarios in signal processing 
    or economics.

    Parameters
    ----------
    n_estimators : int, default=50
        The number of base classifiers in the ensemble.
    learning_rate : float, default=0.1
        The learning rate for gradient boosting, influencing how base classifier 
        weights are adjusted.

    Attributes
    ----------
    base_classifiers_ : list
        List of the instantiated base classifiers.
    weights_ : list
        Weights of each base classifier, determining their influence on the 
        final outcome.

    Mathematical Formulation
    ------------------------
    The Hammerstein-Wiener Ensemble Classifier utilizes the following models and 
    computations:

    1. Hammerstein-Wiener Model:
       .. math::
           y(t) = g_2\left( \sum_{i} a_i y(t-i) + \sum_{j} b_j g_1(u(t-j)) \right)

       where:
       - :math:`g_1` and :math:`g_2` are nonlinear functions applied to the 
         inputs and outputs, respectively.
       - :math:`a_i` and :math:`b_j` are the coefficients of the linear dynamic
         blocks.
       - :math:`u(t-j)` represents the input at time \(t-j\).
       - :math:`y(t-i)` denotes the output at time \(t-i\).

    2. Weight Calculation for Base Classifiers:
       The influence of each classifier is determined by its performance, using:
       
       .. math::
           \text{Weight} = \text{learning\_rate} \cdot \frac{1}{1 + \text{Weighted Error}}

       where :math:`\text{Weighted Error}` is typically evaluated based on the 
       mean squared error or a similar metric.

    Usage and Applications:
    This classifier excels in scenarios requiring modeling of dynamics involving 
    delays or historical dependencies. It is especially effective in time-series 
    forecasting, control systems, and complex signal processing applications.

    See Also 
    ----------
    gofast.estimators.dynamic_system.HammersteinWienerClassifier: 
        Hammerstein-Wiener Classifier for Dynamic Classification Tasks. 
        
    Examples
    --------
    >>> # Import necessary libraries
    >>> import numpy as np
    >>> from sklearn.model_selection import train_test_split
    >>> from gofast.estimators.dynamic_system import EnsembleHWClassifier
    >>> # Define your own data and labels
    >>> X = np.array(...)  # Input data
    >>> y = np.array(...)  # Target labels

    >>> X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

    >>> # Create a Hammerstein-Wiener Ensemble Classifier with default parameters
    >>> hammerstein_wiener_classifier = HWEnsembleClassifier()
    >>> hammerstein_wiener_classifier.fit(X_train, y_train)
    >>> y_pred = hammerstein_wiener_classifier.predict(X_test)
    >>> accuracy = np.mean(y_pred == y_test)
    >>> print('Accuracy:', accuracy)

    Notes
    -----
    - The Hammerstein-Wiener Ensemble Classifier combines the power of the
      Hammerstein-Wiener model with ensemble learning to make accurate
      predictions.
    - The number of base classifiers and the learning rate can be adjusted
      to control the ensemble's behavior.
    - This ensemble is particularly effective when dealing with complex,
      dynamic systems where individual classifiers may struggle.
    - It provides a powerful tool for classification tasks with challenging
      dynamics and nonlinearities.
    """

    def __init__(self, n_estimators=50, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object
        """
        self.base_classifiers_ = []
        self.weights_ = []
        y_pred = np.zeros(len(y))

        for _ in range(self.n_estimators):
            # Create and fit a base classifier. 
            # can replace this with your specific base classifier)
            base_classifier = HammersteinWienerClassifier()
            base_classifier.fit(X, y)
            
            # Calculate weighted error
            weighted_error = np.sum((y - base_classifier.predict(X)) ** 2)
            
            # Calculate weight for this base classifier
            weight = self.learning_rate / (1 + weighted_error)
            
            # Update predictions
            y_pred += weight * base_classifier.predict(X)
            
            # Store the base classifier and its weight
            self.base_classifiers_.append(base_classifier)
            self.weights_.append(weight)
        
        return self

    def predict(self, X):
        """Return class labels.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Samples.

        Returns
        -------
        y_pred : array-like, shape = [n_samples]
            Predicted class labels.
        """
        check_is_fitted(self, 'base_classifiers_')
        y_pred = np.zeros(X.shape[0])
        for i, base_classifier in enumerate(self.base_classifiers_):
            y_pred += self.weights_[i] * base_classifier.predict(X)
        
        return np.where(y_pred >= 0.0, 1, -1)

    def predict_proba(self, X):
        """
        Predict class probabilities using the Hammerstein-Wiener Ensemble 
        Classifier model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        proba : array-like of shape (n_samples, 2)
            The class probabilities of the input samples.
        """
        check_is_fitted(self, 'base_classifiers_')
        X = check_array(X, accept_sparse=True)

        cumulative_prediction = np.zeros(X.shape[0])

        for weight, classifier in zip(self.weights_, self.base_classifiers_):
            cumulative_prediction += weight * classifier.predict(X)

        # Sigmoid function to convert predictions to probabilities
        proba_positive_class = 1 / (1 + np.exp(-cumulative_prediction))
        proba_negative_class = 1 - proba_positive_class

        return np.vstack((proba_negative_class, proba_positive_class)).T

class EnsembleHWRegressor(BaseEstimator, RegressorMixin):
    r"""
    Hammerstein-Wiener Ensemble (HWE) Regressor.

    This ensemble regressor combines multiple Hammerstein-Wiener models to 
    perform regression tasks. Each model in the ensemble is a dynamic system 
    model where the output is a nonlinear function of past inputs and outputs. 
    By averaging the predictions of these models, the ensemble aims to enhance 
    robustness and accuracy, potentially capturing various dynamics of the data.

    The :class:`EnsembleHWRegressor` assumes that each component model 
    (HammersteinWienerEstimator) is pre-implemented with its fit and predict 
    methods. It fits each individual model to the training data and then 
    averages their predictions to produce the final output.

    This approach can potentially improve the performance by capturing different 
    aspects or dynamics of the data with each Hammerstein-Wiener model, and then 
    combining these to form a more robust overall prediction. However, the 
    effectiveness of this ensemble would heavily depend on the diversity and 
    individual accuracy of the included Hammerstein-Wiener models.

    Parameters
    ----------
    hweights_estimators : list of HammersteinWienerEstimator objects
        List of pre-configured Hammerstein-Wiener estimators. Each estimator 
        should be set with its specific hyperparameters, including nonlinearity 
        settings, linear dynamics configurations, and memory depths.

    Attributes
    ----------
    fitted_ : bool
        Indicates whether the ensemble model has been fitted.

    Notes
    -----
    The HWEnsembleRegressor averages the predictions of multiple 
    Hammerstein-Wiener models according to the following mathematical formula:

    .. math::
        y_{\text{pred}} = \frac{1}{N} \sum_{i=1}^{N} y_{\text{pred}_i}

    where:
    - :math:`N` is the number of Hammerstein-Wiener models in the ensemble.
    - :math:`y_{\text{pred}_i}` is the prediction from the \(i\)-th model.

    This method of averaging helps to mitigate any single model's potential 
    overfitting or bias, making the ensemble more reliable for complex 
    regression tasks.

    See Also
    ---------
    HammersteinWienerRegressor:
        For further details on individual Hammerstein-Wiener model implementations.

    Examples
    --------
    >>> # Import necessary libraries
    >>> import numpy as np
    >>> import gofast.estimators.dynamic_system import EnsembleHWRegressor
    >>> # Define two Hammerstein-Wiener models with different configurations
    >>> hw1 = HammersteinWienerRegressor(nonlinearity_in=np.tanh, 
    ...                                  nonlinearity_out=np.tanh, 
    ...                                  linear_model=LinearRegression(),
    ...                                  memory_depth=5)
    >>> hw2 = HammersteinWienerRegressor(nonlinearity_in=np.sin, 
    ...                                  nonlinearity_out=np.sin, 
    ...                                  linear_model=LinearRegression(),
    ...                                  memory_depth=5)
    
    >>> # Create a Hammerstein-Wiener Ensemble Regressor with the models
    >>> ensemble = HWEnsembleRegressor([hw1, hw2])

    >>> # Generate random data for demonstration
    >>> X, y = np.random.rand(100, 1), np.random.rand(100)
    
    >>> # Fit the ensemble on the data and make predictions
    >>> ensemble.fit(X, y)
    >>> y_pred = ensemble.predict(X)

    Notes
    -----
    - The Hammerstein-Wiener Ensemble Regressor combines the power of multiple
      Hammerstein-Wiener models to provide a more accurate and robust regression
      prediction.
    - Each Hammerstein-Wiener model in the ensemble should be pre-configured
      with its own settings, including nonlinearities and linear models.
    - Ensemble models are particularly useful when the underlying dynamics of
      the data can be captured by different model configurations.
    """
    
    def __init__(self, hweights_estimators):
        self.hweights_estimators = hweights_estimators


    def fit(self, X, y):
        """
        Fit the Hammerstein-Wiener Ensemble model to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data input.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y( X, y, estimator = get_estimator_name(self ))
        self.hweights_estimators = is_iterable(
            self.hweights_estimators, exclude_string=True, transform =True) 
        estimator_names = [ get_estimator_name(estimator) for estimator in 
                           self.hweights_estimators ]
        if list( set (estimator_names)) [0] !="HammersteinWienerRegressor": 
            raise EstimatorError("Expect `HammersteinWienerRegressor` estimators."
                                 f" Got {smart_format(estimator_names)}")
            
        for estimator in self.hweights_estimators:
            estimator.fit(X, y)
        self.fitted_ = True
        return self

    def predict(self, X):
        """
        Predict using the Hammerstein-Wiener Ensemble model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict for.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted values, averaged across all Hammerstein-Wiener estimators.
        """
        check_is_fitted(self, 'fitted_')

        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )
                
        predictions = np.array([estimator.predict(X) 
                                for estimator in self.hweights_estimators])
        return np.mean(predictions, axis=0)


