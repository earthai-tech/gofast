# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <eta0noyau@gmail.com>

from __future__ import annotations 
import numpy as np
from tqdm import tqdm

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils import shuffle

try:from sklearn.utils import type_of_target
except: from ..tools.coreutils import type_of_target 
from .._gofastlog import  gofastlog
from ..tools.validator import check_X_y, get_estimator_name, check_array 
from ..tools.validator import check_is_fitted
from ..tools.funcutils import ensure_pkg 
from ._base import FuzzyNeuralNetBase, NeuroFuzzyBase
from ._base import GradientDescentBase 


_logger = gofastlog().get_gofast_logger(__name__)


__all__=[
    "Perceptron", 
    "GradientDescentClassifier",
    "GradientDescentRegressor", 
    "NeuroFuzzyRegressor", 
    "NeuroFuzzyClassifier", 
    "FuzzyNeuralNetClassifier", 
    "FuzzyNeuralNetRegressor", 
    ]

class Perceptron(BaseEstimator, RegressorMixin, ClassifierMixin):
    """
    Perceptron model for both regression and classification tasks.

    This class implements a perceptron model capable of handling regression, 
    binary classification, and multi-class classification problems. The 
    perceptron is based on the MCP (McCulloch-Pitts) neuron model and follows 
    the perceptron learning rule as proposed by Rosenblatt.

    The update rule for the perceptron can be formalized as follows:
    For each training example :math:`x^{(i)}` with target :math:`y^{(i)}` and
    prediction :math:`\hat{y}^{(i)}`, the weights are updated as:

    .. math::
        w := w + \eta0 (y^{(i)} - \hat{y}^{(i)}) x^{(i)}

    where :math:`\eta0` is the learning rate, :math:`w` is the weight vector, and
    :math:`x^{(i)}` is the feature vector of the :math:`i`-th example. The update
    occurs if the prediction :math:`\hat{y}^{(i)}` is incorrect, thereby gradually
    moving the decision boundary towards an optimal position.

    Parameters
    ----------
    eta0 : float, default=0.01
        The learning rate, a value between 0.0 and 1.0. It controls the
        magnitude of weight updates and hence the speed of convergence.

    max_iter : int, default=50
        The number of passes over the training data (also known as epochs).
        It determines how many times the algorithm iterates through the entire
        dataset.

    tol : float, default=1e-4
        The tolerance for stopping criteria. If the number of updates is 
        less than or equal to `tol`, the training stops early.

    early_stopping : bool, default=False
        Whether to use early stopping to terminate training when the 
        validation score is not improving.

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for 
        early stopping. It is used only if `early_stopping` is set to `True`.

    n_iter_no_change : int, default=5
        Maximum number of epochs to not meet `tol` improvement.
        
    problem : {'auto', 'regression', 'classification'}, default='auto'
        Specifies the problem type. If 'auto', the problem type will be 
        detected based on the target values.
        
    random_state : int, default=None
        Seed for the random number generator for weight initialization. A
        consistent `random_state` ensures reproducibility of results.
        
    verbose : bool, default=False
        Whether to print progress messages to stdout.

    Attributes
    ----------
    weights_ : ndarray of shape (n_features + 1, n_outputs)
        Weights after fitting the model. Each weight corresponds to a feature.

    errors_ : list of int
        The number of misclassifications (updates) in each epoch. It can be
        used to evaluate the performance of the classifier over iterations.

    Notes
    -----
    The perceptron algorithm does not converge if the data is not linearly
    separable. In such cases, the number of iterations (`max_iter`) controls 
    the termination of the algorithm.

    This implementation initializes the weights to small random numbers for
    better convergence behavior.

    The `predict_proba` method is only available for classification problems. 
    It raises a `ValueError` if called for regression problems.

    Examples
    --------
    >>> from gofast.estimators.perceptron import Perceptron
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.preprocessing import StandardScaler

    # Load data
    >>> X, y = load_iris(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.3, random_state=42)

    # Standardize features
    >>> sc = StandardScaler()
    >>> X_train_std = sc.fit_transform(X_train)
    >>> X_test_std = sc.transform(X_test)

    # Create and fit the model
    >>> ppn = Perceptron(eta0=0.01, max_iter=40, problem='classification')
    >>> ppn.fit(X_train_std, y_train)

    # Predict and evaluate
    >>> y_pred = ppn.predict(X_test_std)
    >>> print('Misclassified samples: %d' % (y_test != y_pred).sum())

    See Also
    --------
    SGDClassifier : Linear classifiers (SVM, logistic regression, etc.) with 
        SGD training.
    LogisticRegression : Logistic Regression classifier.
    LinearSVC : Linear Support Vector Classification.
    
    References
    ----------
    .. [1] Rosenblatt, F. (1957). The Perceptron: A Perceiving and Recognizing
           Automaton. Cornell Aeronautical Laboratory.
    .. [2] McCulloch, W.S., and Pitts, W. (1943). A Logical Calculus of the
           Ideas Immanent in Nervous Activity. Bulletin of Mathematical
           Biophysics, 5(4), 115-133.
    """
    def __init__(
        self, 
        eta0=0.01, 
        max_iter=50, 
        tol=1e-4,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        problem='auto',
        random_state=None, 
        verbose=False
    ):
        self.eta0 = eta0
        self.max_iter = max_iter
        self.random_state = random_state
        self.problem = problem
        self.tol = tol
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.verbose = verbose
        
    def fit(self, X, y, sample_weight=None):
        """
        Fit the Perceptron model according to the given training data.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data where `n_samples` is the number of samples and 
            `n_features` is the number of features.
    
        y : array-like of shape (n_samples,)
            Target values. For classification tasks, these should be class labels.
            For regression tasks, these should be continuous values.
    
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample. If not provided, all samples 
            are given equal weight.
    
        Returns
        -------
        self : object
            Returns self.
    
        Notes
        -----
        The fit process involves the following steps:
    
        1. **Data Validation**:
           Ensures that `X` and `y` have correct shapes and types using `check_X_y`.
           
        2. **Problem Type Determination**:
           Determines the problem type (regression or classification) based on 
           the `problem` parameter. If `problem` is set to 'auto', the type is 
           determined based on the target values using `type_of_target`.
    
        3. **Target Transformation**:
           For classification tasks, the target values are binarized using 
           `LabelBinarizer`. For binary classification, the target is converted 
           to two columns.
    
        4. **Weight Initialization**:
           Initializes the weights to small random numbers using a random number 
           generator with the specified `random_state`.
    
        5. **Model Training**:
           Iteratively adjusts the weights based on the prediction errors. The 
           update rule for the perceptron is:
    
           .. math::
               w := w + \eta0 (y^{(i)} - \hat{y}^{(i)}) x^{(i)}
    
           where :math:`\eta0` is the learning rate, :math:`w` is the weight 
           vector, and :math:`x^{(i)}` is the feature vector of the :math:`i`-th 
           example. The update occurs if the prediction :math:`\hat{y}^{(i)}` is 
           incorrect.
    
        If `early_stopping` is enabled, training stops early if the number of 
        errors falls below `tol`.
    
        If `verbose=True`, a progress bar is displayed during the training process.
    
        Examples
        --------
        >>> from gofast.estimators.perceptron import Perceptron
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.preprocessing import StandardScaler
    
        # Load data
        >>> X, y = load_iris(return_X_y=True)
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X, y, test_size=0.3, random_state=42)
    
        # Standardize features
        >>> sc = StandardScaler()
        >>> X_train_std = sc.fit_transform(X_train)
        >>> X_test_std = sc.transform(X_test)
    
        # Create and fit the model
        >>> ppn = Perceptron(eta0=0.01, max_iter=40, problem='classification')
        >>> ppn.fit(X_train_std, y_train)
    
        # Predict and evaluate
        >>> y_pred = ppn.predict(X_test_std)
        >>> print('Misclassified samples: %d' % (y_test != y_pred).sum())
    
        See Also
        --------
        SGDClassifier : Linear classifiers (SVM, logistic regression, etc.) with 
            SGD training.
        LogisticRegression : Logistic Regression classifier.
        LinearSVC : Linear Support Vector Classification.
        
        References
        ----------
        .. [1] Rosenblatt, F. (1957). The Perceptron: A Perceiving and Recognizing
               Automaton. Cornell Aeronautical Laboratory.
        .. [2] McCulloch, W.S., and Pitts, W. (1943). A Logical Calculus of the
               Ideas Immanent in Nervous Activity. Bulletin of Mathematical
               Biophysics, 5(4), 115-133.
        """
        X, y = check_X_y(X, y, estimator=self)
        
        self.problem = str(self.problem).lower()
        
        if self.problem == 'auto':
            self.problem = type_of_target(y)
            
        if self.problem in ["binary", "multiclass", "classification"]: 
            self.problem = 'classification'
        elif self.problem in ["continuous", "regression"]:
            self.problem = 'regression'
        
        if self.problem not in ["classification", "regression"]:
            raise ValueError(f"Unsupported task type: {self.problem}")
            
        if self.problem == 'classification':
            self.label_binarizer_ = LabelBinarizer()
            y = self.label_binarizer_.fit_transform(y)
            if y.shape[1] == 1:
                # Convert to two columns for binary classification
                y = np.hstack([y, 1 - y])  
        
        rgen = np.random.RandomState(self.random_state)
        self.weights_ = rgen.normal(loc=0., scale=0.01, size=(
            1 + X.shape[1], y.shape[1] if self.problem == 'classification' else 1))
        self.errors_ = []
        
        if self.verbose:
            progress_bar = tqdm(
                total=self.max_iter, ascii=True, ncols=100,
                desc=f'Fitting {self.__class__.__name__}', 
            )
        for _ in range(self.max_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta0 * (target - self.net_input(
                    xi.reshape(1, -1)).ravel())
                self.weights_[1:] += np.outer(xi, update)
                self.weights_[0] += update
                errors += int(np.any(update != 0.0))
            self.errors_.append(errors)
            if self.early_stopping and errors <= self.tol:
                break
            
            if self.verbose: 
                progress_bar.update(1)
                
        if self.verbose: 
            progress_bar.close() 
            
        return self

    def net_input(self, X):
        """ Compute the net input """
        return np.dot(X, self.weights_[1:]) + self.weights_[0]
    
    def predict(self, X):
        """
        Predict the class label or regression value for the input samples.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
    
        Returns
        -------
        y_pred : array of shape (n_samples,)
            The predicted values. For classification tasks, these are the class
            labels. For regression tasks, these are the continuous target values.
    
        Notes
        -----
        For classification tasks, the prediction is based on the net input to 
        the perceptron:
    
        .. math::
            \hat{y} = \begin{cases}
            1 & \text{if } w \cdot x + b \ge 0 \\
            0 & \text{otherwise}
            \end{cases}
    
        For binary classification, if the net input is greater than or equal to 
        zero, the sample is classified as the positive class; otherwise, it is 
        classified as the negative class.
    
        For multi-class classification, the class with the highest net input 
        value is selected.
    
        For regression tasks, the predicted value is the net input to the 
        perceptron:
    
        .. math::
            \hat{y} = w \cdot x + b
    
        Examples
        --------
        >>> from gofast.estimators.perceptron import Perceptron
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.preprocessing import StandardScaler
    
        # Load data
        >>> X, y = load_iris(return_X_y=True)
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X, y, test_size=0.3, random_state=42)
    
        # Standardize features
        >>> sc = StandardScaler()
        >>> X_train_std = sc.fit_transform(X_train)
        >>> X_test_std = sc.transform(X_test)
    
        # Create and fit the model
        >>> ppn = Perceptron(eta0=0.01, max_iter=40, problem='classification')
        >>> ppn.fit(X_train_std, y_train)
    
        # Predict class labels
        >>> y_pred = ppn.predict(X_test_std)
        >>> y_pred
    
        See Also
        --------
        predict_proba : Probability estimates for classification tasks.
        
        References
        ----------
        .. [1] Rosenblatt, F. (1957). The Perceptron: A Perceiving and Recognizing
               Automaton. Cornell Aeronautical Laboratory.
        .. [2] McCulloch, W.S., and Pitts, W. (1943). A Logical Calculus of the
               Ideas Immanent in Nervous Activity. Bulletin of Mathematical
               Biophysics, 5(4), 115-133.
        """
        check_is_fitted(self, 'weights_')
        X = check_array(X, accept_sparse=True, estimator=self )
        if self.problem == 'classification':
            net_input = self.net_input(X)
            if net_input.shape[1] == 1:  # Binary classification
                return np.where(net_input >= 0.0, self.label_binarizer_.classes_[1],
                                self.label_binarizer_.classes_[0])
            else:  # Multi-class classification
                return self.label_binarizer_.inverse_transform(net_input)
        else:
            return self.net_input(X)
        
    def predict_proba(self, X):
        """
        Probability estimates using the Perceptron model.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples for which to predict the class probabilities.
    
        Returns
        -------
        y_proba : array of shape (n_samples, n_classes)
            The predicted class probabilities. The columns correspond to the 
            classes in sorted order, as they appear in the attribute `classes_`.
    
        Raises
        ------
        NotImplementedError
            If the problem type is regression.
    
        Notes
        -----
        The probability estimates are calculated using the logistic sigmoid 
        function:
    
        .. math::
            P(y = 1 | x) = \frac{1}{1 + \exp(-w \cdot x + b)}
    
        For multi-class classification, the softmax function is used:
    
        .. math::
            P(y = k | x) = \frac{\exp(\hat{y}_k)}{\sum_{j=1}^{C} \exp(\hat{y}_j)}
    
        where :math:`\hat{y}_k` is the raw model output for class `k`, and 
        :math:`C` is the number of classes.
    
        Examples
        --------
        >>> from gofast.estimators.perceptron import Perceptron
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.preprocessing import StandardScaler
    
        # Load data
        >>> X, y = load_iris(return_X_y=True)
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X, y, test_size=0.3, random_state=42)
    
        # Standardize features
        >>> sc = StandardScaler()
        >>> X_train_std = sc.fit_transform(X_train)
        >>> X_test_std = sc.transform(X_test)
    
        # Create and fit the model
        >>> ppn = Perceptron(eta0=0.01, max_iter=40, problem='classification')
        >>> ppn.fit(X_train_std, y_train)
    
        # Predict class probabilities
        >>> y_proba = ppn.predict_proba(X_test_std)
        >>> y_proba
    
        See Also
        --------
        predict : Predict class labels or regression values.
        
        """
        if self.problem != 'classification':
            raise NotImplementedError(
                "Probability estimates are not available for regression.")
        
        check_is_fitted(self, 'weights_')
        X = check_array(X, accept_sparse=True)
        net_input = self.net_input(X)
        proba = 1 / (1 + np.exp(-net_input))
        if proba.shape[1] == 1:  # Binary classification
            return np.vstack([1 - proba.ravel(), proba.ravel()]).T
        else:  # Multi-class classification
            return proba / proba.sum(axis=1, keepdims=True)

    def score(self, X, y, sample_weight=None):
        """
        Returns the mean accuracy on the given test data and labels for 
        classification tasks, and the coefficient of determination R^2 for 
        regression tasks.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. These are the input samples for which predictions are
            to be made.
    
        y : array-like of shape (n_samples,)
            True labels for classification tasks, or continuous target values 
            for regression tasks. These are the ground truth values against which
            predictions will be compared.
    
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If provided, these weights are used to calculate 
            weighted accuracy for classification tasks or weighted R^2 score for 
            regression tasks. If `None`, all samples are given equal weight.
    
        Returns
        -------
        score : float
            Mean accuracy for classification tasks, and R^2 score for 
            regression tasks. The mean accuracy is the fraction of correctly
            classified samples for classification tasks, while the R^2 score
            represents the proportion of the variance in the dependent variable
            that is predictable from the independent variables for regression tasks.
    
        Notes
        -----
        For classification tasks, the accuracy score is calculated as:
    
        .. math::
            \text{accuracy} = \frac{\text{number of correct predictions}}\\
                {\text{total number of predictions}}
    
        For regression tasks, the R^2 score is calculated as:
    
        .. math::
            R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
    
        where :math:`y_i` are the true values, :math:`\hat{y}_i` are the predicted values,
        and :math:`\bar{y}` is the mean of the true values.
    
        Examples
        --------
        >>> from gofast.estimators.perceptron import Perceptron
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.preprocessing import StandardScaler
    
        # Load data
        >>> X, y = load_iris(return_X_y=True)
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X, y, test_size=0.3, random_state=42)
    
        # Standardize features
        >>> sc = StandardScaler()
        >>> X_train_std = sc.fit_transform(X_train)
        >>> X_test_std = sc.transform(X_test)
    
        # Create and fit the model
        >>> ppn = Perceptron(eta0=0.01, max_iter=40, problem='classification')
        >>> ppn.fit(X_train_std, y_train)
    
        # Evaluate the model
        >>> score = ppn.score(X_test_std, y_test)
        >>> print('Accuracy: %.2f' % score)
    
        See Also
        --------
        sklearn.metrics.accuracy_score : Accuracy classification score.
        sklearn.metrics.r2_score : R^2 (coefficient of determination) regression score.

        """
        check_is_fitted(self, 'weights_')
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        
        if self.problem == 'classification':
            return accuracy_score(y, y_pred, sample_weight=sample_weight)
        else:
            return r2_score(y, y_pred, sample_weight=sample_weight)
        
    def __repr__(self):
        """
        Provide a string representation of the Perceptron object, displaying 
        its parameters in a formatted manner for better readability.
    
        Returns
        -------
        repr : str
            String representation of the Perceptron object with its parameters.
        """
        params = ",\n    ".join(f"{key}={val}" for key, val in self.get_params().items())
        return f"{self.__class__.__name__}(\n    {params}\n)"
    
@ensure_pkg("skfuzzy", extra="The `skfuzzy` package is required for the"
            " `NeuroFuzzyRegressor` to function correctly. Please install"
            " it to proceed."
    )
class NeuroFuzzyRegressor(NeuroFuzzyBase, RegressorMixin):
    """
    NeuroFuzzyRegressor is a neuro-fuzzy network-based regressor that
    integrates fuzzy logic and neural networks to perform regression tasks.
    
    See more in :ref:`User Guide`.
    
    Parameters
    ----------
    n_clusters : int, default=3
        The number of fuzzy clusters for each feature in the input data. 
        This parameter controls the granularity of the fuzzy logic 
        partitioning.
    
    eta0 : float, default=0.001
        The initial learning rate for training the `MLPRegressor`. A smaller 
        learning rate makes the training process more stable but slower.
    
    max_iter : int, default=200
        Maximum number of iterations for training the `MLPRegressor`. This 
        parameter determines how long the training will continue if it does 
        not converge before reaching this limit.
    
    hidden_layer_sizes : tuple, default=(100,)
        The number of neurons in each hidden layer of the `MLPRegressor`. 
        For example, (100,) means there is one hidden layer with 100 neurons.
    
    activation : {'identity', 'logistic', 'tanh', 'relu'}, default='relu'
        Activation function for the hidden layer in the `MLPRegressor`.
        - 'identity': no-op activation, useful for linear bottleneck, 
          returns `f(x) = x`
        - 'logistic': the logistic sigmoid function, returns 
          `f(x) = 1 / (1 + exp(-x))`
        - 'tanh': the hyperbolic tan function, returns 
          `f(x) = tanh(x)`
        - 'relu': the rectified linear unit function, returns 
          `f(x) = max(0, x)`
    
    solver : {'lbfgs', 'sgd', 'adam'}, default='adam'
        The solver for weight optimization in the `MLPRegressor`.
        - 'lbfgs': optimizer in the family of quasi-Newton methods
        - 'sgd': stochastic gradient descent
        - 'adam': stochastic gradient-based optimizer proposed by Kingma, 
          Diederik, and Jimmy Ba
    
    alpha : float, default=0.0001
        L2 penalty (regularization term) parameter for the `MLPRegressor`. 
        Regularization helps to prevent overfitting by penalizing large weights.
    
    batch_size : int, default='auto'
        Size of minibatches for stochastic optimizers. If 'auto', batch size 
        is set to `min(200, n_samples)`.
    
    learning_rate : {'constant', 'invscaling', 'adaptive'}, default='constant'
        Learning rate schedule for weight updates.
        - 'constant': learning rate remains constant
        - 'invscaling': gradually decreases the learning rate at each time step
        - 'adaptive': keeps the learning rate constant to `eta0`  as long as 
        training loss keeps decreasing
    
    power_t : float, default=0.5
        The exponent for inverse scaling learning rate. It is used only 
        when `learning_rate='invscaling'`.
    
    tol : float, default=1e-4
        Tolerance for the optimization. Training will stop when the 
        validation score is not improving by at least `tol` for two consecutive 
        iterations.
    
    momentum : float, default=0.9
        Momentum for gradient descent update. Should be between 0 and 1. 
        It helps to accelerate gradients vectors in the right directions, 
        thus leading to faster converging.
    
    nesterovs_momentum : bool, default=True
        Whether to use Nesterov’s momentum. Nesterov’s momentum is an 
        improvement over standard momentum by considering the gradient ahead 
        of the current point in time.
    
    early_stopping : bool, default=False
        Whether to use early stopping to terminate training when validation 
        score is not improving. If set to `True`, it will automatically set 
        aside a fraction of training data as validation and terminate 
        training when validation score is not improving by at least `tol` for 
        `n_iter_no_change` consecutive epochs.
    
    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for 
        early stopping. It is used only if `early_stopping` is set to `True`.
    
    beta_1 : float, default=0.9
        Exponential decay rate for estimates of first moment vector in adam, 
        should be in the range `[0, 1)`.
    
    beta_2 : float, default=0.999
        Exponential decay rate for estimates of second moment vector in adam, 
        should be in the range `[0, 1)`.
    
    epsilon : float, default=1e-8
        Value for numerical stability in adam. It is used to prevent any 
        division by zero in the implementation.
    
    n_iter_no_change : int, default=10
        Maximum number of epochs to not meet `tol` improvement. The training 
        will stop if the validation score is not improving by at least `tol` 
        for `n_iter_no_change` consecutive epochs.
    
    max_fun : int, default=15000
        Only used when solver='lbfgs'. Maximum number of function calls.
    
    random_state : int, RandomState instance, default=None
        Determines random number generation for weights and bias 
        initialization, train-test split if early stopping is used, and batch 
        sampling when solver='sgd' or 'adam'. Pass an int for reproducible 
        results across multiple function calls.
    
    verbose : bool, default=False
        Whether to print progress messages to stdout. If `True`, progress 
        messages are printed during training.
    
    Notes
    -----
    The NeuroFuzzyRegressor combines the strengths of fuzzy logic and neural 
    networks to capture nonlinear relationships in data. Fuzzy logic helps in 
    dealing with uncertainty and imprecision, while neural networks provide 
    learning capabilities.

    The fuzzy clustering is performed using c-means clustering. The resulting 
    fuzzy sets are used to transform the input features, which are then fed 
    into an `MLPRegressor` for learning.

    The objective function for fuzzy c-means clustering is:
    
    .. math::
        \min \sum_{i=1}^{n} \sum_{j=1}^{c} u_{ij}^m \|x_i - v_j\|^2

    where :math:`u_{ij}` is the degree of membership of :math:`x_i` in the 
    cluster :math:`j`, :math:`v_j` is the cluster center, and :math:`m` is a 
    weighting exponent.
    
    The `MLPRegressor` uses backpropagation to learn the weights of the 
    network by minimizing the mean squared error between the predicted and 
    actual target values:
    
    .. math::
        MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2

    where :math:`n` is the number of samples, :math:`y_i` is the actual target 
    value, and :math:`\hat{y}_i` is the predicted target value.

    Examples
    --------
    >>> from gofast.estimators.perceptron import NeuroFuzzyRegressor
    >>> X = [[0., 0.], [1., 1.]]
    >>> y = [0, 1]
    >>> regr = NeuroFuzzyRegressor(n_clusters=2, max_iter=1000, verbose=True)
    >>> regr.fit(X, y)
    NeuroFuzzyRegressor(...)
    >>> regr.predict([[2., 2.]])
    array([1])

    See Also
    --------
    sklearn.neural_network.MLPRegressor : Multi-layer Perceptron regressor.
    
    References
    ----------
    .. [1] J.-S. R. Jang, "ANFIS: Adaptive-Network-based Fuzzy Inference 
           System," IEEE Transactions on Systems, Man, and Cybernetics, vol. 
           23, no. 3, pp. 665-685, 1993.
    """
    def __init__(
        self, 
        n_clusters=3, 
        eta0=0.001, 
        max_iter=200, 
        hidden_layer_sizes=(100,), 
        activation='relu', 
        solver='adam', 
        alpha=0.0001, 
        batch_size='auto', 
        learning_rate='constant', 
        power_t=0.5, 
        tol=1e-4, 
        momentum=0.9, 
        nesterovs_momentum=True, 
        early_stopping=False, 
        validation_fraction=0.1, 
        beta_1=0.9, 
        beta_2=0.999, 
        epsilon=1e-8, 
        n_iter_no_change=10, 
        max_fun=15000, 
        random_state=None, 
        verbose=False
    ):
        super().__init__(
            n_clusters=n_clusters, 
            eta0=eta0, 
            max_iter=max_iter, 
            hidden_layer_sizes=hidden_layer_sizes, 
            activation=activation, 
            solver=solver, 
            alpha=alpha, 
            batch_size=batch_size, 
            learning_rate=learning_rate, 
            power_t=power_t, 
            tol=tol, 
            momentum=momentum, 
            nesterovs_momentum=nesterovs_momentum, 
            early_stopping=early_stopping, 
            validation_fraction=validation_fraction, 
            beta_1=beta_1, 
            beta_2=beta_2, 
            epsilon=epsilon, 
            n_iter_no_change=n_iter_no_change, 
            max_fun=max_fun, 
            random_state=random_state, 
            verbose=verbose, 
            is_classifier=False
        )

    def fit(self, X, y, sample_weight=None):
        """
        Fit the NeuroFuzzyRegressor model according to the given training data.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data where `n_samples` is the number of samples and
            `n_features` is the number of features.
    
        y : array-like of shape (n_samples,)
            Target values.
    
        Returns
        -------
        self : object
            Returns self.
    
        Notes
        -----
        The fit process involves the following steps:
        
        1. **Data Validation and Preprocessing**:
           Ensures that `X` and `y` have correct shapes and types.
           Standard scaling is applied to `X`, and `y` is one-hot encoded.
    
        2. **Fuzzification**:
           The input features are transformed into fuzzy sets using fuzzy
           c-means clustering:
           
           .. math::
               \min \sum_{i=1}^{n} \sum_{j=1}^{c} u_{ij}^m \|x_i - v_j\|^2
    
           where :math:`u_{ij}` is the degree of membership of :math:`x_i` in
           cluster :math:`j`, :math:`v_j` is the cluster center, and :math:`m`
           is a weighting exponent.
    
        3. **MLP Training**:
           The scaled and fuzzified features are used to train an
           `MLPRegressor` to map inputs to the encoded target values.
        
        If `verbose=True`, a progress bar will be displayed during the MLP
        training process.
    
        Examples
        --------
        >>> from gofast.estimators.perceptron import NeuroFuzzyRegressor
        >>> X = [[0., 0.], [1., 1.]]
        >>> y = [0, 1]
        >>> regr = NeuroFuzzyRegressor(n_clusters=2, max_iter=1000, verbose=True)
        >>> regr.fit(X, y)
        NeuroFuzzyRegressor(...)
    
        See Also
        --------
        sklearn.neural_network.MLPRegressor : Multi-layer Perceptron regressor.
        
        References
        ----------
        .. [1] J.-S. R. Jang, "ANFIS: Adaptive-Network-based Fuzzy Inference
               System," IEEE Transactions on Systems, Man, and Cybernetics, vol.
               23, no. 3, pp. 665-685, 1993.
        """
        return self._fit(X, y, sample_weight)
    
    def predict(self, X):
        """
        Predict using the NeuroFuzzyRegressor model.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
    
        Returns
        -------
        y_pred_classes : array of shape (n_samples,)
            The predicted classes.
    
        Notes
        -----
        The predict process involves the following steps:
        
        1. **Data Scaling**:
           The input features are scaled using the same scaler fitted during
           the training process.
    
        2. **MLP Prediction**:
           The scaled features are fed into the trained `MLPRegressor` to
           obtain predicted probabilities for each class.
    
        3. **Inverse Transform**:
           The predicted class probabilities are transformed back to class
           labels using the fitted `OneHotEncoder`.
        
        Examples
        --------
        >>> from gofast.estimators.perceptron import NeuroFuzzyRegressor
        >>> X = [[2., 2.]]
        >>> regr = NeuroFuzzyRegressor(n_clusters=2, max_iter=1000, verbose=True)
        >>> regr.fit([[0., 0.], [1., 1.]], [0, 1])
        NeuroFuzzyRegressor(...)
        >>> regr.predict(X)
        array([1])
    
        See Also
        --------
        sklearn.neural_network.MLPRegressor : Multi-layer Perceptron regressor.
        
        References
        ----------
        .. [1] J.-S. R. Jang, "ANFIS: Adaptive-Network-based Fuzzy Inference
               System," IEEE Transactions on Systems, Man, and Cybernetics, vol.
               23, no. 3, pp. 665-685, 1993.
        """
 
        return self._predict(X)
    
class NeuroFuzzyClassifier(NeuroFuzzyBase, ClassifierMixin):
    """
    NeuroFuzzyClassifier is a neuro-fuzzy network-based classifier that 
    integrates fuzzy logic and neural networks to perform classification tasks.

    Parameters
    ----------
    n_clusters : int, default=3
        The number of fuzzy clusters for each feature in the input data. 
        This parameter controls the granularity of the fuzzy logic 
        partitioning.

    learning_rate_init : float, default=0.001
        The initial learning rate for training the `MLPClassifier`. A smaller 
        learning rate makes the training process more stable but slower.

    max_iter : int, default=200
        Maximum number of iterations for training the `MLPClassifier`. This 
        parameter determines how long the training will continue if it does 
        not converge before reaching this limit.

    hidden_layer_sizes : tuple, default=(100,)
        The number of neurons in each hidden layer of the `MLPClassifier`. 
        For example, (100,) means there is one hidden layer with 100 neurons.

    activation : {'identity', 'logistic', 'tanh', 'relu'}, default='relu'
        Activation function for the hidden layer in the `MLPClassifier`.
        - 'identity': no-op activation, useful for linear bottleneck, 
          returns `f(x) = x`
        - 'logistic': the logistic sigmoid function, returns 
          `f(x) = 1 / (1 + exp(-x))`
        - 'tanh': the hyperbolic tan function, returns 
          `f(x) = tanh(x)`
        - 'relu': the rectified linear unit function, returns 
          `f(x) = max(0, x)`

    solver : {'lbfgs', 'sgd', 'adam'}, default='adam'
        The solver for weight optimization in the `MLPClassifier`.
        - 'lbfgs': optimizer in the family of quasi-Newton methods
        - 'sgd': stochastic gradient descent
        - 'adam': stochastic gradient-based optimizer proposed by Kingma, 
          Diederik, and Jimmy Ba

    alpha : float, default=0.0001
        L2 penalty (regularization term) parameter for the `MLPClassifier`. 
        Regularization helps to prevent overfitting by penalizing large weights.

    batch_size : int, default='auto'
        Size of minibatches for stochastic optimizers. If 'auto', batch size 
        is set to `min(200, n_samples)`.

    learning_rate : {'constant', 'invscaling', 'adaptive'}, default='constant'
        Learning rate schedule for weight updates.
        - 'constant': learning rate remains constant
        - 'invscaling': gradually decreases the learning rate at each time step
        - 'adaptive': keeps the learning rate constant to `learning_rate_init` 
          as long as training loss keeps decreasing

    power_t : float, default=0.5
        The exponent for inverse scaling learning rate. It is used only 
        when `learning_rate='invscaling'`.

    tol : float, default=1e-4
        Tolerance for the optimization. Training will stop when the 
        validation score is not improving by at least `tol` for two consecutive 
        iterations.

    momentum : float, default=0.9
        Momentum for gradient descent update. Should be between 0 and 1. 
        It helps to accelerate gradients vectors in the right directions, 
        thus leading to faster converging.

    nesterovs_momentum : bool, default=True
        Whether to use Nesterov’s momentum. Nesterov’s momentum is an 
        improvement over standard momentum by considering the gradient ahead 
        of the current point in time.

    early_stopping : bool, default=False
        Whether to use early stopping to terminate training when validation 
        score is not improving. If set to `True`, it will automatically set 
        aside a fraction of training data as validation and terminate 
        training when validation score is not improving by at least `tol` for 
        `n_iter_no_change` consecutive epochs.

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for 
        early stopping. It is used only if `early_stopping` is set to `True`.

    beta_1 : float, default=0.9
        Exponential decay rate for estimates of first moment vector in adam, 
        should be in the range `[0, 1)`.

    beta_2 : float, default=0.999
        Exponential decay rate for estimates of second moment vector in adam, 
        should be in the range `[0, 1)`.

    epsilon : float, default=1e-8
        Value for numerical stability in adam. It is used to prevent any 
        division by zero in the implementation.

    n_iter_no_change : int, default=10
        Maximum number of epochs to not meet `tol` improvement. The training 
        will stop if the validation score is not improving by at least `tol` 
        for `n_iter_no_change` consecutive epochs.

    max_fun : int, default=15000
        Only used when solver='lbfgs'. Maximum number of function calls.

    random_state : int, RandomState instance, default=None
        Determines random number generation for weights and bias 
        initialization, train-test split if early stopping is used, and batch 
        sampling when solver='sgd' or 'adam'. Pass an int for reproducible 
        results across multiple function calls.

    verbose : bool, default=False
        Whether to print progress messages to stdout. If `True`, progress 
        messages are printed during training.

    Notes
    -----
    The NeuroFuzzyClassifier combines the strengths of fuzzy logic and neural 
    networks to capture nonlinear relationships in data. Fuzzy logic helps in 
    dealing with uncertainty and imprecision, while neural networks provide 
    learning capabilities.

    The fuzzy clustering is performed using c-means clustering. The resulting 
    fuzzy sets are used to transform the input features, which are then fed 
    into an `MLPClassifier` for learning.

    The objective function for fuzzy c-means clustering is:
    
    .. math::
        \min \sum_{i=1}^{n} \sum_{j=1}^{c} u_{ij}^m \|x_i - v_j\|^2

    where :math:`u_{ij}` is the degree of membership of :math:`x_i` in the 
    cluster :math:`j`, :math:`v_j` is the cluster center, and :math:`m` is a 
    weighting exponent.
    
    The `MLPClassifier` uses cross-entropy loss to learn the weights of the 
    network by minimizing the difference between the predicted and actual 
    class probabilities:
    
    .. math::
        \text{Cross-Entropy} = - \sum_{i=1}^{n} \sum_{j=1}^{c} y_{ij} \log(\hat{y}_{ij})

    where :math:`n` is the number of samples, :math:`c` is the number of 
    classes, :math:`y_{ij}` is the actual class label (one-hot encoded), and 
    :math:`\hat{y}_{ij}` is the predicted class probability.

    Examples
    --------
    >>> from gofast.estimators.perceptron import NeuroFuzzyClassifier
    >>> X = [[0., 0.], [1., 1.]]
    >>> y = [0, 1]
    >>> clf = NeuroFuzzyClassifier(n_clusters=2, max_iter=1000, verbose=True)
    >>> clf.fit(X, y)
    NeuroFuzzyClassifier(...)
    >>> clf.predict([[2., 2.]])
    array([1])

    See Also
    --------
    sklearn.neural_network.MLPClassifier : Multi-layer Perceptron classifier.
    
    References
    ----------
    .. [1] J.-S. R. Jang, "ANFIS: Adaptive-Network-based Fuzzy Inference 
           System," IEEE Transactions on Systems, Man, and Cybernetics, vol. 
           23, no. 3, pp. 665-685, 1993.
    """
    def __init__(
        self, 
        n_clusters=3, 
        eta0=0.001, 
        max_iter=200, 
        hidden_layer_sizes=(100,), 
        activation='relu', 
        solver='adam', 
        alpha=0.0001, 
        batch_size='auto', 
        learning_rate='constant', 
        power_t=0.5, 
        tol=1e-4, 
        momentum=0.9, 
        nesterovs_momentum=True, 
        early_stopping=False, 
        validation_fraction=0.1, 
        beta_1=0.9, 
        beta_2=0.999, 
        epsilon=1e-8, 
        n_iter_no_change=10, 
        max_fun=15000, 
        random_state=None, 
        verbose=False
    ):
        super().__init__(
            n_clusters=n_clusters, 
            eta0=eta0, 
            max_iter=max_iter, 
            hidden_layer_sizes=hidden_layer_sizes, 
            activation=activation, 
            solver=solver, 
            alpha=alpha, 
            batch_size=batch_size, 
            learning_rate=learning_rate, 
            power_t=power_t, 
            tol=tol, 
            momentum=momentum, 
            nesterovs_momentum=nesterovs_momentum, 
            early_stopping=early_stopping, 
            validation_fraction=validation_fraction, 
            beta_1=beta_1, 
            beta_2=beta_2, 
            epsilon=epsilon, 
            n_iter_no_change=n_iter_no_change, 
            max_fun=max_fun, 
            random_state=random_state, 
            verbose=verbose, 
            is_classifier=True
        )

    def fit(self, X, y, sample_weight=None):
        """
        Fit the NeuroFuzzyClassifier model according to the given training data.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data where `n_samples` is the number of samples and
            `n_features` is the number of features.
    
        y : array-like of shape (n_samples,)
            Target values (class labels).
            
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. Currently, this parameter is not used by the method 
            but is included for API consistency.
    
        Returns
        -------
        self : object
            Returns self.
    
        Notes
        -----
        The fit process involves the following steps:
        
        1. **Data Validation and Preprocessing**:
           Ensures that `X` and `y` have correct shapes and types.
           Standard scaling is applied to `X`, and `y` is one-hot encoded if 
           there are multiple classes.
    
        2. **Fuzzification**:
           The input features are transformed into fuzzy sets using fuzzy
           c-means clustering:
           
           .. math::
               \min \sum_{i=1}^{n} \sum_{j=1}^{c} u_{ij}^m \|x_i - v_j\|^2
    
           where :math:`u_{ij}` is the degree of membership of :math:`x_i` in
           cluster :math:`j`, :math:`v_j` is the cluster center, and :math:`m`
           is a weighting exponent.
    
        3. **MLP Training**:
           The scaled and fuzzified features are used to train an
           `MLPClassifier` to map inputs to the encoded target values.
    
        If `verbose=True`, a progress bar will be displayed during the MLP
        training process.
    
        Examples
        --------
        >>> from gofast.estimators.perceptron import NeuroFuzzyClassifier
        >>> X = [[0., 0.], [1., 1.]]
        >>> y = [0, 1]
        >>> clf = NeuroFuzzyClassifier(n_clusters=2, max_iter=1000, verbose=True)
        >>> clf.fit(X, y)
        NeuroFuzzyClassifier(...)
    
        See Also
        --------
        sklearn.neural_network.MLPClassifier : Multi-layer Perceptron classifier.
        
        References
        ----------
        .. [1] J.-S. R. Jang, "ANFIS: Adaptive-Network-based Fuzzy Inference
               System," IEEE Transactions on Systems, Man, and Cybernetics, vol.
               23, no. 3, pp. 665-685, 1993.
        """
        return self._fit(X, y, sample_weight)

    def predict(self, X):
        """
        Predict using the NeuroFuzzyClassifier model.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
    
        Returns
        -------
        y_pred_classes : array of shape (n_samples,)
            The predicted class labels.
    
        Notes
        -----
        The predict process involves the following steps:
        
        1. **Data Scaling**:
           The input features are scaled using the same scaler fitted during
           the training process.
    
        2. **MLP Prediction**:
           The scaled features are fed into the trained `MLPClassifier` to
           obtain predicted class probabilities.
    
        3. **Inverse Transform**:
           The predicted class probabilities are transformed back to class
           labels using the fitted `OneHotEncoder`.
    
        Examples
        --------
        >>> from gofast.estimators.perceptron import NeuroFuzzyClassifier
        >>> X = [[2., 2.]]
        >>> clf = NeuroFuzzyClassifier(n_clusters=2, max_iter=1000, verbose=True)
        >>> clf.fit([[0., 0.], [1., 1.]], [0, 1])
        NeuroFuzzyClassifier(...)
        >>> clf.predict(X)
        array([1])
    
        See Also
        --------
        sklearn.neural_network.MLPClassifier : Multi-layer Perceptron classifier.
        
        References
        ----------
        .. [1] J.-S. R. Jang, "ANFIS: Adaptive-Network-based Fuzzy Inference
               System," IEEE Transactions on Systems, Man, and Cybernetics, vol.
               23, no. 3, pp. 665-685, 1993.
        """
        return self._predict(X)
    
    def predict_proba(self, X):
        """
        Probability estimates using the NeuroFuzzyClassifier model.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples for which to predict the class probabilities.
    
        Returns
        -------
        y_proba : array of shape (n_samples, n_classes)
            The predicted class probabilities. The columns correspond to the 
            classes in sorted order, as they appear in the attribute `classes_`.
    
        Notes
        -----
        The predict_proba process involves the following steps:
        
        1. **Data Scaling**:
           The input features are scaled using the same scaler fitted during
           the training process.
    
        2. **MLP Probability Prediction**:
           The scaled features are fed into the trained `MLPClassifier` to
           obtain predicted class probabilities.
    
        The predicted probabilities represent the likelihood of each class
        for the input samples. The `MLPClassifier` computes these probabilities
        using the softmax function in the output layer:
        
        .. math::
            P(y = k | x) = \frac{\exp(\hat{y}_k)}{\sum_{j=1}^{C} \exp(\hat{y}_j)}
    
        where :math:`\hat{y}_k` is the raw model output for class `k`, and 
        :math:`C` is the number of classes.
    
        Examples
        --------
        >>> from gofast.estimators.perceptron import NeuroFuzzyClassifier
        >>> X = [[2., 2.]]
        >>> clf = NeuroFuzzyClassifier(n_clusters=2, max_iter=1000, verbose=True)
        >>> clf.fit([[0., 0.], [1., 1.]], [0, 1])
        NeuroFuzzyClassifier(...)
        >>> clf.predict_proba(X)
        array([[0.1, 0.9]])
    
        See Also
        --------
        sklearn.neural_network.MLPClassifier : Multi-layer Perceptron classifier.
        
        References
        ----------
        .. [1] J.-S. R. Jang, "ANFIS: Adaptive-Network-based Fuzzy Inference
               System," IEEE Transactions on Systems, Man, and Cybernetics, vol.
               23, no. 3, pp. 665-685, 1993.
        """
        return self._predict_proba(X)


class GradientDescentClassifier(GradientDescentBase, ClassifierMixin):
    def __init__(
            self, 
            eta0=0.001, 
            max_iter=1000, 
            tol=1e-4, 
            early_stopping=False, 
            validation_fraction=0.1, 
            n_iter_no_change=5,
            learning_rate='constant', 
            power_t=0.5, 
            alpha=0.0001, 
            batch_size='auto', 
            clipping_threshold=250,
            shuffle=True, 
            random_state=None, 
            verbose=False,
            ):
        super().__init__(
            eta0=eta0, 
            max_iter=max_iter, 
            tol=tol, 
            early_stopping=early_stopping, 
            validation_fraction=validation_fraction, 
            n_iter_no_change=n_iter_no_change, 
            learning_rate=learning_rate, 
            power_t=power_t, 
            alpha=alpha, 
            batch_size=batch_size, 
            clipping_threshold=clipping_threshold,
            shuffle=shuffle, 
            random_state=random_state, 
            verbose=verbose, 
            )

    def fit(self, X, y, sample_weight=None):
        return self._fit(X, y, is_classifier=True)

    def predict(self, X):
        check_is_fitted(self, 'weights_')
        X = check_array(X, accept_sparse=True)
        return self._predict(X)

    def predict_proba(self, X):
        check_is_fitted(self, 'weights_')
        X = check_array(X, accept_sparse=True)
        return self._predict_proba(X)

    def score(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred, sample_weight=sample_weight)

class GradientDescentRegressor(GradientDescentBase, RegressorMixin):
    def __init__(
        self, 
        eta0=0.001, 
        max_iter=1000, 
        shuffle=True, 
        random_state=None, 
        verbose=False,
        tol=1e-4, 
        early_stopping=False, 
        validation_fraction=0.1, 
        n_iter_no_change=5,
        learning_rate='constant', 
        power_t=0.5, 
        alpha=0.0001, 
        batch_size='auto', 
        clipping_threshold=250
        ):
        super().__init__(
            eta0=eta0, 
            max_iter=max_iter, 
            tol=tol, 
            early_stopping=early_stopping, 
            validation_fraction=validation_fraction, 
            n_iter_no_change=n_iter_no_change, 
            learning_rate=learning_rate, 
            power_t=power_t, 
            alpha=alpha, 
            batch_size=batch_size, 
            clipping_threshold=clipping_threshold,
            shuffle=shuffle, 
            random_state=random_state, 
            verbose=verbose, 
            )

    def fit(self, X, y):
        return self._fit(X, y, is_classifier=False)

    def predict(self, X):
        check_is_fitted(self, 'weights_')
        X = check_array(X, accept_sparse=True)
        return self._predict(X)

    def score(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y)
        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight)


class GradientDescentClassifier0(BaseEstimator, ClassifierMixin):
    r"""
    Gradient Descent Classifier for Binary and Multi-Class Classification.

    This classifier leverages the gradient descent optimization algorithm to 
    efficiently train binary classifiers for each class using a One-vs-Rest (OvR) 
    approach for multi-class scenarios. In this framework, a distinct binary 
    classifier is trained for each class, where the specific class is treated 
    as the positive class, and all other classes are aggregated into a single 
    negative class. This method simplifies complex multi-class problems into 
    multiple binary classification problems, making it easier to apply the 
    strengths of binary classifiers.

    Mathematical Formulation:
    The principle of gradient descent involves iteratively updating the model 
    parameters to minimize a predefined cost function. The weight update rule 
    in the context of gradient descent is mathematically formulated as follows:

    .. math::
        w := w - \eta0 \nabla J(w)

    where:
    - :math:`w` represents the weight vector of the model.
    - :math:`\eta0` is the learning rate, which controls the step size during the 
      iterative weight update process.
    - :math:`\nabla J(w)` denotes the gradient of the cost function \(J\), 
      calculated with respect to the weights. This gradient points in the direction 
      of the steepest ascent of the cost function, and subtracting it leads to 
      movement towards the minimum.

    This model's effectiveness hinges on the choice of the cost function, which 
    typically aims to quantify the discrepancy between the predicted labels and 
    the actual labels in a dataset, and the appropriate setting of the learning 
    rate to ensure convergence without overshooting the minimum.

    Parameters
    ----------
    eta0 : float
        Learning rate, between 0.0 and 1.0. It controls the step size at each
        iteration while moving toward a minimum of the cost function.

    max_iter : int
        Number of epochs, i.e., complete passes over the entire training dataset.

    shuffle : bool
        If True, shuffles the training data before each epoch to prevent cycles
        and ensure better convergence.

    Attributes
    ----------
    weights_ : 2d-array
        Weights for each binary classifier, one row per classifier.

    classes_ : array
        Unique class labels identified in the training data.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from gofast.estimators.perceptron import GradientDescentClassifier
    >>> iris = load_iris()
    >>> X = iris.data
    >>> y = iris.target
    >>> gd_clf = GradientDescentClassifier(eta0=0.01, max_iter=50)
    >>> gd_clf.fit(X, y)
    >>> y_pred = gd_clf.predict(X)
    >>> print('Accuracy:', np.mean(y_pred == y))

    Notes
    -----
    The learning rate (eta0) is a critical parameter that affects the convergence
    of the algorithm. A small learning rate can lead to slow convergence, while a
    large learning rate can cause oscillations or divergence in the cost function.
    The number of epochs controls the number of times the algorithm iterates
    over the entire dataset.

    See Also
    --------
    LogisticRegression : Logistic Regression classifier from Scikit-Learn.
    SGDClassifier : Linear classifier with Stochastic Gradient Descent.
    
    """

    def __init__(self, eta0=0.01, max_iter=50, shuffle=True):
        self.eta0 = eta0
        self.max_iter = max_iter
        self.shuffle = shuffle

    def fit(self, X, y, sample_weight =None ):
        """
        Fit the model to the training data.
    
        This method fits a binary classifier for each class using a 
        One-vs-Rest approach when dealing with multiple classes. It initializes
        weights, applies binarization to the target values, and iteratively 
        updates weights using gradient descent.
    
        Parameters:
        - X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        - y : array-like, shape (n_samples,)
            Target values (class labels).
    
        Returns:
        - self : object
            Returns an instance of self.
        """
        X, y = check_X_y(X, y, estimator=self)
        self.classes_ = np.unique(y)
        n_features = X.shape[1]
        n_classes = len(self.classes_)
        
        # Initialize weights
        self.weights_ = np.zeros((n_classes, n_features + 1))
        
        # Ensure binary classification is handled properly
        if n_classes == 2:
            self.label_binarizer_ = LabelBinarizer(pos_label=1, neg_label=0)
        else:
            self.label_binarizer_ = LabelBinarizer()
        
        Y_bin = self.label_binarizer_.fit_transform(y)
        
        # Adjust for binary case with single output column from LabelBinarizer
        if n_classes == 2 and Y_bin.shape[1] == 1:
            Y_bin = np.hstack([1 - Y_bin, Y_bin])
        
        for i in range(n_classes):
            y_bin = Y_bin[:, i]
            for _ in range(self.max_iter):
                if self.shuffle:
                    X, y_bin = shuffle(X, y_bin)
                errors = y_bin - self._predict_proba(X, i)
                self.weights_[i, 1:] += self.eta0 * X.T.dot(errors)
                self.weights_[i, 0] += self.eta0 * errors.sum()
        return self
    
    def _predict_proba(self, X, class_idx):
        """
        Calculate the class probabilities for a given class index using 
        the model's net input.
    
        Parameters:
        - X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Samples for which to predict the probabilities.
        - class_idx : int
            Index of the class for which the probability is predicted.
    
        Returns:
        - proba : array-like, shape (n_samples,)
            Probability estimates for the specified class.
        """
        net_input = self.net_input(X, class_idx)
        return np.where(net_input >= 0.0, 1, 0)

    def net_input(self, X, class_idx):
        """
        Compute the net input (linear combination plus bias) for a 
        specified class index.
    
        Parameters:
        - X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Samples for which to calculate the net input.
        - class_idx : int
            Index of the class for which the net input is calculated.
    
        Returns:
        - net_input : array-like, shape (n_samples,)
            Calculated net input for the specified class.
        """
        return np.dot(X, self.weights_[class_idx, 1:]) + self.weights_[class_idx, 0]

    def predict(self, X):
        """
        Predict class labels for samples in X.
    
        The predicted class label for each sample in X is the class that yields 
        the highest net input
        value (before activation function).
    
        Parameters:
        - X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Samples to predict for.
    
        Returns:
        - labels : array-like, shape (n_samples,)
            Predicted class label for each sample.
        """
        check_is_fitted(self, 'weights_')
        X = check_array(X, accept_large_sparse=True, accept_sparse=True, to_frame=False)
        probas = np.array([self.net_input(X, i) for i in range(len(self.classes_))]).T
        return self.classes_[np.argmax(probas, axis=1)]

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.
    
        In multi-label classification, this is a subset accuracy where all labels
        for each sample must be correctly predicted to count as a correct prediction.
    
        Parameters:
        - X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Test samples.
        - y : array-like, shape (n_samples,)
            True labels for X.
    
        Returns:
        - score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        X = check_array(X, accept_large_sparse=True, accept_sparse=True,
                        to_frame=False)
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

class GradientDescentRegressor0(BaseEstimator, RegressorMixin):
    r"""
    Gradient Descent Regressor for Linear Regression.

    This regressor employs the gradient descent optimization algorithm to
    perform linear regression tasks. The GradientDescentRegressor predicts 
    outcomes using a linear combination of input features and optimizes the 
    weights by minimizing the cost function through gradient descent. During 
    training, the model's weights are updated by calculating the gradient of 
    the cost function and adjusting the weights to minimize the error.

    Mathematical Formulation:
    The core of the gradient descent for linear regression involves iteratively
    updating the model weights to reduce the cost function, using the following rule:

    .. math::
        w := w - \eta0 \sum_{i} (y^{(i)} - \phi(z^{(i)})) x^{(i)}

    Here:
    - :math:`\eta0` is the learning rate, which controls the step size in the 
      weight update process.
    - :math:`y^{(i)}` represents the true target value for the \(i\)-th sample.
    - :math:`\phi(z^{(i)})` is the predicted target value, where :math:`\phi(z)` 
      is typically a linear function of the form \(w^T x\).
    - :math:`x^{(i)}` is the feature vector of the \(i\)-th sample.

    By continuously adjusting the weights in the direction that reduces the cost 
    function, this regressor effectively fits the linear model to the data, 
    improving prediction accuracy over iterations.

    Parameters
    ----------
    eta0 : float, default=0.0001
        Learning rate (between 0.0 and 1.0). Controls the step size for weight
        updates during training.

    max_iter : int, default=1000
        Number of passes over the training dataset (epochs). Specifies how many
        times the algorithm iterates over the entire dataset during training.

    random_state : int or None, default=None
        Seed for the random number generator. If provided, it ensures
        reproducibility of results. Set to None for non-deterministic behavior.

    Attributes
    ----------
    weights_ : 1d-array
        Weights after fitting. These weights represent the coefficients for
        the linear combination of features.

    cost_ : list
        List containing the value of the cost function at each epoch during
        training. Useful for monitoring the convergence of the algorithm.

    Examples
    --------
    >>> from sklearn.datasets import load_boston
    >>> from sklearn.model_selection import train_test_split
    >>> from gofast.estimators.perceptron import GradientDescentRegressor
    >>> boston = load_boston()
    >>> X = boston.data
    >>> y = boston.target
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.3, random_state=0)
    >>> gd_reg = GradientDescentRegressor(eta0=0.0001, max_iter=1000)
    >>> gd_reg.fit(X_train, y_train)
    >>> y_pred = gd_reg.predict(X_test)
    >>> mse = np.mean((y_pred - y_test) ** 2)
    >>> print('Mean Squared Error:', mse)

    Notes
    -----
    Gradient Descent is a widely used optimization technique for training
    linear regression models. The learning rate (eta0) and the number of
    iterations (max_iter) are crucial hyperparameters that impact the training
    process. Careful tuning of these hyperparameters is necessary for
    achieving optimal results.

    See Also
    --------
    LinearRegression : Linear regression from Scikit-Learn.
    """
    def __init__(self, 
                 eta0=0.0001, 
                 max_iter=1000, 
                 # random_state and verbose should be placed as last parameters 
                 random_state =None,
                 verbose=False 
                 ):
        self.eta0 = eta0
        self.max_iter = max_iter
        self.random_state = random_state 

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
        X, y = check_X_y( X, y, estimator = get_estimator_name(self ))
        self.weights_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.max_iter):
            errors = y - self.predict(X)
            self.weights_[1:] += self.eta0 * X.T.dot(errors)
            self.weights_[0] += self.eta0 * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.weights_[1:]) + self.weights_[0]

    def predict(self, X):
        """Return continuous output"""
        check_is_fitted (self, 'weights_') 
        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )
        return self.net_input(X)


class FuzzyNeuralNetClassifier(FuzzyNeuralNetBase, ClassifierMixin):
    """
    FuzzyNeuralNetClassifier is an ensemble neuro-fuzzy network-based 
    classifier that integrates fuzzy logic and neural networks to perform 
    classification tasks.

    Parameters
    ----------
    n_clusters : int, default=3
        The number of fuzzy clusters for each feature in the input data. 
        This parameter controls the granularity of the fuzzy logic 
        partitioning.

    n_estimators : int, default=10
        The number of neural network estimators in the ensemble. This 
        parameter determines how many individual MLP models will be trained 
        and their predictions aggregated.

    learning_rate_init : float, default=0.001
        The initial learning rate for training the `MLPClassifier`. A smaller 
        learning rate makes the training process more stable but slower.

    max_iter : int, default=200
        Maximum number of iterations for training each `MLPClassifier`. This 
        parameter determines how long the training will continue if it does 
        not converge before reaching this limit.

    hidden_layer_sizes : tuple, default=(100,)
        The number of neurons in each hidden layer of the `MLPClassifier`. 
        For example, (100,) means there is one hidden layer with 100 neurons.

    activation : {'identity', 'logistic', 'tanh', 'relu'}, default='relu'
        Activation function for the hidden layer in the `MLPClassifier`.
        - 'identity': no-op activation, useful for linear bottleneck, 
          returns `f(x) = x`
        - 'logistic': the logistic sigmoid function, returns 
          `f(x) = 1 / (1 + exp(-x))`
        - 'tanh': the hyperbolic tan function, returns 
          `f(x) = tanh(x)`
        - 'relu': the rectified linear unit function, returns 
          `f(x) = max(0, x)`

    solver : {'lbfgs', 'sgd', 'adam'}, default='adam'
        The solver for weight optimization in the `MLPClassifier`.
        - 'lbfgs': optimizer in the family of quasi-Newton methods
        - 'sgd': stochastic gradient descent
        - 'adam': stochastic gradient-based optimizer proposed by Kingma, 
          Diederik, and Jimmy Ba

    alpha : float, default=0.0001
        L2 penalty (regularization term) parameter for the `MLPClassifier`. 
        Regularization helps to prevent overfitting by penalizing large weights.

    batch_size : int, default='auto'
        Size of minibatches for stochastic optimizers. If 'auto', batch size 
        is set to `min(200, n_samples)`.

    learning_rate : {'constant', 'invscaling', 'adaptive'}, default='constant'
        Learning rate schedule for weight updates.
        - 'constant': learning rate remains constant
        - 'invscaling': gradually decreases the learning rate at each time step
        - 'adaptive': keeps the learning rate constant to `learning_rate_init` 
          as long as training loss keeps decreasing

    power_t : float, default=0.5
        The exponent for inverse scaling learning rate. It is used only 
        when `learning_rate='invscaling'`.

    tol : float, default=1e-4
        Tolerance for the optimization. Training will stop when the 
        validation score is not improving by at least `tol` for two consecutive 
        iterations.

    momentum : float, default=0.9
        Momentum for gradient descent update. Should be between 0 and 1. 
        It helps to accelerate gradients vectors in the right directions, 
        thus leading to faster converging.

    nesterovs_momentum : bool, default=True
        Whether to use Nesterov’s momentum. Nesterov’s momentum is an 
        improvement over standard momentum by considering the gradient ahead 
        of the current point in time.

    early_stopping : bool, default=False
        Whether to use early stopping to terminate training when validation 
        score is not improving. If set to `True`, it will automatically set 
        aside a fraction of training data as validation and terminate 
        training when validation score is not improving by at least `tol` for 
        `n_iter_no_change` consecutive epochs.

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for 
        early stopping. It is used only if `early_stopping` is set to `True`.

    beta_1 : float, default=0.9
        Exponential decay rate for estimates of first moment vector in adam, 
        should be in the range `[0, 1)`.

    beta_2 : float, default=0.999
        Exponential decay rate for estimates of second moment vector in adam, 
        should be in the range `[0, 1)`.

    epsilon : float, default=1e-8
        Value for numerical stability in adam. It is used to prevent any 
        division by zero in the implementation.

    n_iter_no_change : int, default=10
        Maximum number of epochs to not meet `tol` improvement. The training 
        will stop if the validation score is not improving by at least `tol` 
        for `n_iter_no_change` consecutive epochs.

    max_fun : int, default=15000
        Only used when solver='lbfgs'. Maximum number of function calls.

    random_state : int, RandomState instance, default=None
        Determines random number generation for weights and bias 
        initialization, train-test split if early stopping is used, and batch 
        sampling when solver='sgd' or 'adam'. Pass an int for reproducible 
        results across multiple function calls.

    verbose : bool, default=False
        Whether to print progress messages to stdout. If `True`, progress 
        messages are printed during training.

    Notes
    -----
    The FuzzyNeuralNetClassifier combines the strengths of fuzzy logic and 
    neural networks to capture nonlinear relationships in data. Fuzzy logic 
    helps in dealing with uncertainty and imprecision, while neural networks 
    provide learning capabilities.

    The fuzzy clustering is performed using c-means clustering. The resulting 
    fuzzy sets are used to transform the input features, which are then fed 
    into an ensemble of `MLPClassifier` models for learning.

    The objective function for fuzzy c-means clustering is:
    
    .. math::
        \min \sum_{i=1}^{n} \sum_{j=1}^{c} u_{ij}^m \|x_i - v_j\|^2

    where :math:`u_{ij}` is the degree of membership of :math:`x_i` in the 
    cluster :math:`j`, :math:`v_j` is the cluster center, and :math:`m` is a 
    weighting exponent.
    
    The ensemble of neural networks helps to improve the model's robustness 
    and performance by averaging the predictions of multiple estimators, 
    which reduces the variance and the risk of overfitting.

    Examples
    --------
    >>> from gofast.estimators.perceptron import FuzzyNeuralNetClassifier
    >>> X = [[0., 0.], [1., 1.]]
    >>> y = [0, 1]
    >>> clf = FuzzyNeuralNetClassifier(n_clusters=2, n_estimators=5, 
                                       max_iter=1000, verbose=True)
    >>> clf.fit(X, y)
    FuzzyNeuralNetClassifier(...)
    >>> clf.predict([[2., 2.]])
    array([1])

    See Also
    --------
    sklearn.neural_network.MLPClassifier : Multi-layer Perceptron classifier.
    sklearn.neural_network.MLPRegressor : Multi-layer Perceptron regressor.
    
    References
    ----------
    .. [1] J.-S. R. Jang, "ANFIS: Adaptive-Network-based Fuzzy Inference 
           System," IEEE Transactions on Systems, Man, and Cybernetics, vol. 
           23, no. 3, pp. 665-685, 1993.
    """
    def __init__(
        self, 
        n_estimators=10,
        n_clusters=3, 
        eta0=0.001, 
        max_iter=200, 
        hidden_layer_sizes=(100,), 
        activation='relu', 
        solver='adam', 
        alpha=0.0001, 
        batch_size='auto', 
        learning_rate='constant', 
        power_t=0.5, 
        tol=1e-4, 
        momentum=0.9, 
        nesterovs_momentum=True, 
        early_stopping=False, 
        validation_fraction=0.1, 
        beta_1=0.9, 
        beta_2=0.999, 
        epsilon=1e-8, 
        n_iter_no_change=10, 
        max_fun=15000, 
        random_state=None, 
        verbose=False): 
        super().__init__( 
            n_clusters=n_clusters, 
            n_estimators=n_estimators,
            eta0=eta0, 
            max_iter=max_iter, 
            hidden_layer_sizes=hidden_layer_sizes, 
            activation=activation, 
            solver=solver, 
            alpha=alpha, 
            batch_size=batch_size, 
            learning_rate=learning_rate, 
            power_t=power_t, 
            tol=tol, 
            momentum=momentum, 
            nesterovs_momentum=nesterovs_momentum, 
            early_stopping=early_stopping, 
            validation_fraction=validation_fraction, 
            beta_1=beta_1, 
            beta_2=beta_2, 
            epsilon=epsilon, 
            n_iter_no_change=n_iter_no_change, 
            max_fun=max_fun, 
            random_state=random_state, 
            verbose=verbose
        )
    def fit(self, X, y, sample_weight=None):
        """
        Fit the FuzzyNeuralNetClassifier model according to the given training 
        data.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data where `n_samples` is the number of samples and
            `n_features` is the number of features.
    
        y : array-like of shape (n_samples,)
            Target values (class labels).
    
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample. If not provided, all samples are 
            given equal weight.
    
        Returns
        -------
        self : object
            Returns self.
    
        Notes
        -----
        The fit process involves the following steps:
        
        1. **Data Validation and Preprocessing**:
           Ensures that `X` and `y` have correct shapes and types.
           Standard scaling is applied to `X`, and `y` is one-hot encoded if 
           there are multiple classes.
    
        2. **Fuzzification**:
           The input features are transformed into fuzzy sets using fuzzy
           c-means clustering:
           
           .. math::
               \min \sum_{i=1}^{n} \sum_{j=1}^{c} u_{ij}^m \|x_i - v_j\|^2
    
           where :math:`u_{ij}` is the degree of membership of :math:`x_i` in 
           cluster :math:`j`, :math:`v_j` is the cluster center, and :math:`m` 
           is a weighting exponent.
    
        3. **Ensemble Training**:
           An ensemble of `n_estimators` neural network models is trained. 
           Each model in the ensemble is an `MLPClassifier`.
    
           The scaled and fuzzified features are used to train each model in 
           the ensemble. The models are trained independently, and their 
           predictions will be aggregated to form the final prediction.
    
        If `verbose=True`, a progress bar will be displayed during the training 
        process for each model in the ensemble.
    
        Examples
        --------
        >>> from gofast.estimators.perceptron import FuzzyNeuralNetClassifier
        >>> X = [[0., 0.], [1., 1.]]
        >>> y = [0, 1]
        >>> clf = FuzzyNeuralNetClassifier(n_clusters=2, n_estimators=5, 
                                           max_iter=1000, verbose=True)
        >>> clf.fit(X, y)
        FuzzyNeuralNetClassifier(...)
    
        See Also
        --------
        sklearn.neural_network.MLPClassifier : Multi-layer Perceptron classifier.
        
        References
        ----------
        .. [1] J.-S. R. Jang, "ANFIS: Adaptive-Network-based Fuzzy Inference
               System," IEEE Transactions on Systems, Man, and Cybernetics, vol.
               23, no. 3, pp. 665-685, 1993.
        """
        return self._fit(
            X, y, 
            is_classifier=True, 
            sample_weight=sample_weight
        )

    def predict(self, X):
        """
        Predict using the FuzzyNeuralNetClassifier model.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
    
        Returns
        -------
        y_pred_classes : array of shape (n_samples,)
            The predicted class labels.
    
        Notes
        -----
        The predict process involves the following steps:
        
        1. **Data Scaling**:
           The input features are scaled using the same scaler fitted during
           the training process.
    
        2. **Ensemble Prediction**:
           The scaled features are fed into each trained `MLPClassifier` in 
           the ensemble to obtain individual predictions. The final prediction 
           is made by taking the majority vote among the ensemble's predictions.
    
        Examples
        --------
        >>> from gofast.estimators.perceptron import FuzzyNeuralNetClassifier
        >>> X = [[2., 2.]]
        >>> clf = FuzzyNeuralNetClassifier(n_clusters=2, n_estimators=5, 
                                           max_iter=1000, verbose=True)
        >>> clf.fit([[0., 0.], [1., 1.]], [0, 1])
        FuzzyNeuralNetClassifier(...)
        >>> clf.predict(X)
        array([1])
    
        See Also
        --------
        sklearn.neural_network.MLPClassifier : Multi-layer Perceptron classifier.
        
        References
        ----------
        .. [1] J.-S. R. Jang, "ANFIS: Adaptive-Network-based Fuzzy Inference
               System," IEEE Transactions on Systems, Man, and Cybernetics, vol.
               23, no. 3, pp. 665-685, 1993.
        """
        check_is_fitted (self, 'ensemble_') 
        X = check_array(X,accept_sparse= True, to_frame=False, input_name="X" )
        X_scaled = self.scaler.transform(X)
        y_pred_ensemble = np.array(
            [estimator.predict(X_scaled) for estimator in self.ensemble_])
        y_pred_majority = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x)), axis=0, arr=y_pred_ensemble)
        y_pred_classes = self.encoder.inverse_transform(
            y_pred_majority.reshape(-1, 1)).flatten()
        
        return y_pred_classes

    def predict_proba(self, X):
        """
        Probability estimates using the FuzzyNeuralNetClassifier model.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples for which to predict the class probabilities.
    
        Returns
        -------
        y_proba : array of shape (n_samples, n_classes)
            The predicted class probabilities. The columns correspond to the 
            classes in sorted order, as they appear in the attribute `classes_`.
    
        Notes
        -----
        The predict_proba process involves the following steps:
        
        1. **Data Scaling**:
           The input features are scaled using the same scaler fitted during
           the training process.
    
        2. **Ensemble Probability Prediction**:
           The scaled features are fed into each trained `MLPClassifier` in 
           the ensemble to obtain individual class probability predictions. The 
           final probability prediction is made by averaging the probabilities 
           predicted by all models in the ensemble.
    
        The predicted probabilities represent the likelihood of each class
        for the input samples. The `MLPClassifier` computes these probabilities
        using the softmax function in the output layer:
        
        .. math::
            P(y = k | x) = \frac{\exp(\hat{y}_k)}{\sum_{j=1}^{C} \exp(\hat{y}_j)}
    
        where :math:`\hat{y}_k` is the raw model output for class `k`, and 
        :math:`C` is the number of classes.
    
        Examples
        --------
        >>> from gofast.estimators.perceptron import FuzzyNeuralNetClassifier
        >>> X = [[2., 2.]]
        >>> clf = FuzzyNeuralNetClassifier(n_clusters=2, n_estimators=5, 
                                           max_iter=1000, verbose=True)
        >>> clf.fit([[0., 0.], [1., 1.]], [0, 1])
        FuzzyNeuralNetClassifier(...)
        >>> clf.predict_proba(X)
        array([[0.1, 0.9]])
    
        See Also
        --------
        sklearn.neural_network.MLPClassifier : Multi-layer Perceptron classifier.
        
        References
        ----------
        .. [1] J.-S. R. Jang, "ANFIS: Adaptive-Network-based Fuzzy Inference
               System," IEEE Transactions on Systems, Man, and Cybernetics, vol.
               23, no. 3, pp. 665-685, 1993.
        """
        check_is_fitted (self, 'ensemble_') 
        X = check_array(X,accept_sparse= True, to_frame=False, input_name="X" )
        
        X_scaled = self.scaler.transform(X)
        y_proba_ensemble = np.array(
            [estimator.predict_proba(X_scaled) for estimator in self.ensemble_])
        y_proba_avg = np.mean(y_proba_ensemble, axis=0)
        
        return y_proba_avg

class FuzzyNeuralNetRegressor(FuzzyNeuralNetBase, RegressorMixin):
    """
    FuzzyNeuralNetRegressor is an ensemble neuro-fuzzy network-based 
    regressor that integrates fuzzy logic and neural networks to perform 
    regression tasks.

    Parameters
    ----------
    n_clusters : int, default=3
        The number of fuzzy clusters for each feature in the input data. 
        This parameter controls the granularity of the fuzzy logic 
        partitioning.

    n_estimators : int, default=10
        The number of neural network estimators in the ensemble. This 
        parameter determines how many individual MLP models will be trained 
        and their predictions aggregated.

    learning_rate_init : float, default=0.001
        The initial learning rate for training the `MLPRegressor`. A smaller 
        learning rate makes the training process more stable but slower.

    max_iter : int, default=200
        Maximum number of iterations for training each `MLPRegressor`. This 
        parameter determines how long the training will continue if it does 
        not converge before reaching this limit.

    hidden_layer_sizes : tuple, default=(100,)
        The number of neurons in each hidden layer of the `MLPRegressor`. 
        For example, (100,) means there is one hidden layer with 100 neurons.

    activation : {'identity', 'logistic', 'tanh', 'relu'}, default='relu'
        Activation function for the hidden layer in the `MLPRegressor`.
        - 'identity': no-op activation, useful for linear bottleneck, 
          returns `f(x) = x`
        - 'logistic': the logistic sigmoid function, returns 
          `f(x) = 1 / (1 + exp(-x))`
        - 'tanh': the hyperbolic tan function, returns 
          `f(x) = tanh(x)`
        - 'relu': the rectified linear unit function, returns 
          `f(x) = max(0, x)`

    solver : {'lbfgs', 'sgd', 'adam'}, default='adam'
        The solver for weight optimization in the `MLPRegressor`.
        - 'lbfgs': optimizer in the family of quasi-Newton methods
        - 'sgd': stochastic gradient descent
        - 'adam': stochastic gradient-based optimizer proposed by Kingma, 
          Diederik, and Jimmy Ba

    alpha : float, default=0.0001
        L2 penalty (regularization term) parameter for the `MLPRegressor`. 
        Regularization helps to prevent overfitting by penalizing large weights.

    batch_size : int, default='auto'
        Size of minibatches for stochastic optimizers. If 'auto', batch size 
        is set to `min(200, n_samples)`.

    learning_rate : {'constant', 'invscaling', 'adaptive'}, default='constant'
        Learning rate schedule for weight updates.
        - 'constant': learning rate remains constant
        - 'invscaling': gradually decreases the learning rate at each time step
        - 'adaptive': keeps the learning rate constant to `learning_rate_init` 
          as long as training loss keeps decreasing

    power_t : float, default=0.5
        The exponent for inverse scaling learning rate. It is used only 
        when `learning_rate='invscaling'`.

    tol : float, default=1e-4
        Tolerance for the optimization. Training will stop when the 
        validation score is not improving by at least `tol` for two consecutive 
        iterations.

    momentum : float, default=0.9
        Momentum for gradient descent update. Should be between 0 and 1. 
        It helps to accelerate gradients vectors in the right directions, 
        thus leading to faster converging.

    nesterovs_momentum : bool, default=True
        Whether to use Nesterov’s momentum. Nesterov’s momentum is an 
        improvement over standard momentum by considering the gradient ahead 
        of the current point in time.

    early_stopping : bool, default=False
        Whether to use early stopping to terminate training when validation 
        score is not improving. If set to `True`, it will automatically set 
        aside a fraction of training data as validation and terminate 
        training when validation score is not improving by at least `tol` for 
        `n_iter_no_change` consecutive epochs.

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for 
        early stopping. It is used only if `early_stopping` is set to `True`.

    beta_1 : float, default=0.9
        Exponential decay rate for estimates of first moment vector in adam, 
        should be in the range `[0, 1)`.

    beta_2 : float, default=0.999
        Exponential decay rate for estimates of second moment vector in adam, 
        should be in the range `[0, 1)`.

    epsilon : float, default=1e-8
        Value for numerical stability in adam. It is used to prevent any 
        division by zero in the implementation.

    n_iter_no_change : int, default=10
        Maximum number of epochs to not meet `tol` improvement. The training 
        will stop if the validation score is not improving by at least `tol` 
        for `n_iter_no_change` consecutive epochs.

    max_fun : int, default=15000
        Only used when solver='lbfgs'. Maximum number of function calls.

    random_state : int, RandomState instance, default=None
        Determines random number generation for weights and bias 
        initialization, train-test split if early stopping is used, and batch 
        sampling when solver='sgd' or 'adam'. Pass an int for reproducible 
        results across multiple function calls.

    verbose : bool, default=False
        Whether to print progress messages to stdout. If `True`, progress 
        messages are printed during training.

    Notes
    -----
    The FuzzyNeuralNetRegressor combines the strengths of fuzzy logic and 
    neural networks to capture nonlinear relationships in data. Fuzzy logic 
    helps in dealing with uncertainty and imprecision, while neural networks 
    provide learning capabilities.

    The fuzzy clustering is performed using c-means clustering. The resulting 
    fuzzy sets are used to transform the input features, which are then fed 
    into an ensemble of `MLPRegressor` models for learning.

    The objective function for fuzzy c-means clustering is:
    
    .. math::
        \min \sum_{i=1}^{n} \sum_{j=1}^{c} u_{ij}^m \|x_i - v_j\|^2

    where :math:`u_{ij}` is the degree of membership of :math:`x_i` in the 
    cluster :math:`j`, :math:`v_j` is the cluster center, and :math:`m` is a 
    weighting exponent.
    
    The ensemble of neural networks helps to improve the model's robustness 
    and performance by averaging the predictions of multiple estimators, 
    which reduces the variance and the risk of overfitting.

    Examples
    --------
    >>> from gofast.estimators.perceptron import FuzzyNeuralNetRegressor
    >>> X = [[0., 0.], [1., 1.]]
    >>> y = [0.0, 1.0]
    >>> regr = FuzzyNeuralNetRegressor(n_clusters=2, n_estimators=5, 
                                       max_iter=1000, verbose=True)
    >>> regr.fit(X, y)
    FuzzyNeuralNetRegressor(...)
    >>> regr.predict([[2., 2.]])
    array([1.0])

    See Also
    --------
    sklearn.neural_network.MLPRegressor : Multi-layer Perceptron regressor.
    
    References
    ----------
    .. [1] J.-S. R. Jang, "ANFIS: Adaptive-Network-based Fuzzy Inference 
           System," IEEE Transactions on Systems, Man, and Cybernetics, vol. 
           23, no. 3, pp. 665-685, 1993.
    """
    def __init__(
        self, 
        n_estimators=10,
        n_clusters=3, 
        eta0=0.001, 
        max_iter=200, 
        hidden_layer_sizes=(100,), 
        activation='relu', 
        solver='adam', 
        alpha=0.0001, 
        batch_size='auto', 
        learning_rate='constant', 
        power_t=0.5, 
        tol=1e-4, 
        momentum=0.9, 
        nesterovs_momentum=True, 
        early_stopping=False, 
        validation_fraction=0.1, 
        beta_1=0.9, 
        beta_2=0.999, 
        epsilon=1e-8, 
        n_iter_no_change=10, 
        max_fun=15000, 
        random_state=None, 
        verbose=False): 
        super().__init__( 
            n_clusters=n_clusters, 
            n_estimators=n_estimators,
            eta0=eta0, 
            max_iter=max_iter, 
            hidden_layer_sizes=hidden_layer_sizes, 
            activation=activation, 
            solver=solver, 
            alpha=alpha, 
            batch_size=batch_size, 
            learning_rate=learning_rate, 
            power_t=power_t, 
            tol=tol, 
            momentum=momentum, 
            nesterovs_momentum=nesterovs_momentum, 
            early_stopping=early_stopping, 
            validation_fraction=validation_fraction, 
            beta_1=beta_1, 
            beta_2=beta_2, 
            epsilon=epsilon, 
            n_iter_no_change=n_iter_no_change, 
            max_fun=max_fun, 
            random_state=random_state, 
            verbose=verbose
        )
        
    def fit(self, X, y, sample_weight=None):
        """
        Fit the FuzzyNeuralNetRegressor model according to the given training 
        data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target values (continuous).

        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample. If not provided, all samples 
            are given equal weight.

        Returns
        -------
        self : object
            Returns self.

        Notes
        -----
        The fit process involves the following steps:
        
        1. **Data Validation and Preprocessing**:
           Ensures that `X` and `y` have correct shapes and types.
           Standard scaling is applied to `X`.

        2. **Fuzzification**:
           The input features are transformed into fuzzy sets using fuzzy
           c-means clustering:
           
           .. math::
               \min \sum_{i=1}^{n} \sum_{j=1}^{c} u_{ij}^m \|x_i - v_j\|^2

           where :math:`u_{ij}` is the degree of membership of :math:`x_i` in 
           cluster :math:`j`, :math:`v_j` is the cluster center, and :math:`m` 
           is a weighting exponent.

        3. **Ensemble Training**:
           An ensemble of `n_estimators` neural network models is trained. 
           Each model in the ensemble is an `MLPRegressor`.

           The scaled and fuzzified features are used to train each model in 
           the ensemble. The models are trained independently, and their 
           predictions will be aggregated to form the final prediction.

        If `verbose=True`, a progress bar will be displayed during the training 
        process for each model in the ensemble.

        Examples
        --------
        >>> from gofast.estimators.perceptron import FuzzyNeuralNetRegressor
        >>> X = [[0., 0.], [1., 1.]]
        >>> y = [0.0, 1.0]
        >>> regr = FuzzyNeuralNetRegressor(n_clusters=2, n_estimators=5, 
                                           max_iter=1000, verbose=True)
        >>> regr.fit(X, y)
        FuzzyNeuralNetRegressor(...)

        See Also
        --------
        sklearn.neural_network.MLPRegressor : Multi-layer Perceptron regressor.
        
        References
        ----------
        .. [1] J.-S. R. Jang, "ANFIS: Adaptive-Network-based Fuzzy Inference
               System," IEEE Transactions on Systems, Man, and Cybernetics, vol.
               23, no. 3, pp. 665-685, 1993.
        """
        return self._fit(X, y, is_classifier=False, sample_weight=sample_weight)
    
    def predict(self, X):
        """
        Predict using the FuzzyNeuralNetRegressor model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred_avg : array of shape (n_samples,)
            The predicted continuous values.

        Notes
        -----
        The predict process involves the following steps:
        
        1. **Data Scaling**:
           The input features are scaled using the same scaler fitted during
           the training process.

        2. **Ensemble Prediction**:
           The scaled features are fed into each trained `MLPRegressor` in 
           the ensemble to obtain individual predictions. The final prediction 
           is made by averaging the predictions of all models in the ensemble.

        Examples
        --------
        >>> from gofast.estimators.perceptron import FuzzyNeuralNetRegressor
        >>> X = [[2., 2.]]
        >>> regr = FuzzyNeuralNetRegressor(n_clusters=2, n_estimators=5, 
                                           max_iter=1000, verbose=True)
        >>> regr.fit([[0., 0.], [1., 1.]], [0.0, 1.0])
        FuzzyNeuralNetRegressor(...)
        >>> regr.predict(X)
        array([1.0])

        See Also
        --------
        sklearn.neural_network.MLPRegressor : Multi-layer Perceptron regressor.
        
        References
        ----------
        .. [1] J.-S. R. Jang, "ANFIS: Adaptive-Network-based Fuzzy Inference
               System," IEEE Transactions on Systems, Man, and Cybernetics, vol.
               23, no. 3, pp. 665-685, 1993.
        """
        check_is_fitted(self, 'ensemble')
        X = check_array(X, accept_sparse=True, to_frame=False, input_name="X")
        X_scaled = self.scaler.transform(X)
        y_pred_ensemble = np.array(
            [estimator.predict(X_scaled) for estimator in self.ensemble])
        y_pred_avg = np.mean(y_pred_ensemble, axis=0)
        
        return y_pred_avg











