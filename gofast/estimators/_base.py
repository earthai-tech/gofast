# -*- coding: utf-8 -*-

from __future__ import annotations
from collections import defaultdict 
import inspect 
import numpy as np 
from tqdm import tqdm

from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelBinarizer, StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPRegressor,MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from .._gofastlog import  gofastlog
from ..tools.validator import check_X_y, check_array 
from ..tools.validator import check_is_fitted
from ..tools.funcutils import ensure_pkg 
from .util import activator 
try: 
    from skfuzzy.control import Antecedent
    import skfuzzy as fuzz
except: pass 

class GradientDescentBase(BaseEstimator):
    """
    Base class for Gradient Descent optimization.

    This class implements the core functionality of gradient descent, which is
    used for both regression and classification tasks. It includes parameters
    for controlling the learning rate, stopping criteria, and other aspects
    of the training process.

    Parameters
    ----------
    eta0 : float, default=0.01
        The initial learning rate for weight updates. A smaller learning rate
        makes the training process more stable but slower.

    max_iter : int, default=1000
        Maximum number of iterations over the training data (epochs).

    tol : float, default=1e-4
        Tolerance for the optimization. Training will stop when the validation
        score is not improving by at least `tol` for `n_iter_no_change` consecutive
        epochs.

    early_stopping : bool, default=False
        Whether to use early stopping to terminate training when validation score
        is not improving.

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for early
        stopping. It is used only if `early_stopping` is set to `True`.

    n_iter_no_change : int, default=5
        Maximum number of epochs to not meet `tol` improvement. The training will
        stop if the validation score is not improving by at least `tol` for `n_iter_no_change`
        consecutive epochs.

    learning_rate : {'constant', 'invscaling', 'adaptive'}, default='constant'
        Learning rate schedule for weight updates.
        - 'constant': learning rate remains constant.
        - 'invscaling': gradually decreases the learning rate at each time step.
        - 'adaptive': keeps the learning rate constant to `eta0` as long as training
          loss keeps decreasing.

    power_t : float, default=0.5
        The exponent for inverse scaling learning rate. It is used only when
        `learning_rate='invscaling'`.

    alpha : float, default=0.0001
        L2 penalty (regularization term) parameter. Regularization helps to prevent
        overfitting by penalizing large weights.

    batch_size : int, default='auto'
        Size of minibatches for stochastic optimizers. If 'auto', batch size is set
        to `min(200, n_samples)`.

    clipping_threshold : float, default=250
        Threshold for clipping gradients to avoid overflow during the computation of
        the sigmoid function. This prevents numerical instability in the training process.

    shuffle : bool, default=True
        Whether to shuffle the training data before each epoch.

    random_state : int, RandomState instance, default=None
        Seed for the random number generator for weight initialization and shuffling
        the training data.

    verbose : bool, default=False
        Whether to print progress messages to stdout. If `True`, progress messages are
        printed during training.

    Attributes
    ----------
    weights_ : ndarray of shape (n_features + 1, n_outputs)
        Coefficients of the features in the decision function.

    n_iter_ : int
        Number of iterations run.

    best_loss_ : float
        Best loss achieved during training.

    no_improvement_count_ : int
        Number of iterations with no improvement in validation score.

    Examples
    --------
    >>> from gofast.estimators.perceptron import GradientDescentBase
    >>> import numpy as np
    >>> X = np.array([[0, 0], [1, 1], [2, 2]])
    >>> y = np.array([0, 1, 2])
    >>> gd = GradientDescentBase(eta0=0.01, max_iter=100)
    >>> gd._fit(X, y, is_classifier=False)
    >>> gd._predict(X)

    Notes
    -----
    Gradient Descent is a fundamental optimization technique used in various
    machine learning algorithms. This implementation allows for flexibility
    and control over the learning process through its numerous parameters.

    See Also
    --------
    SGDClassifier : Linear classifiers (SVM, logistic regression, etc.) with SGD training.
    SGDRegressor : Linear model fitted by minimizing a regularized empirical loss with SGD.

    References
    ----------
    .. [1] Bottou, L. (2010). "Large-Scale Machine Learning with Stochastic Gradient Descent".
           Proceedings of COMPSTAT'2010.
    .. [2] Ruder, S. (2016). "An overview of gradient descent optimization algorithms".
           arXiv preprint arXiv:1609.04747.
    """

    def __init__(
        self, 
        eta0=0.01, 
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
        activation='sigmoid',
        shuffle=True, 
        random_state=None, 
        verbose=False
    ):
        self.eta0 = eta0
        self.max_iter = max_iter
        self.tol = tol
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.learning_rate = learning_rate
        self.power_t = power_t
        self.alpha = alpha
        self.batch_size = batch_size
        self.clipping_threshold = clipping_threshold
        self.activation=activation
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose

    def _initialize_weights(self, n_features, n_outputs):
        self.weights_ = np.zeros((n_features + 1, n_outputs))
        self.n_iter_ = 0
        self.best_loss_ = np.inf
        self.no_improvement_count_ = 0

    def _update_weights(self, X_batch, y_batch):
        errors = y_batch - self._net_input(X_batch)
        if self.learning_rate == 'invscaling':
            eta = self.eta0 / (self.n_iter_ ** self.power_t)
        elif self.learning_rate == 'adaptive':
            eta = (
                self.eta0 if self.no_improvement_count_ < self.n_iter_no_change 
                else self.eta0 / 10
            )
        else:
            eta = self.eta0

        self.weights_[1:] += eta * X_batch.T.dot(errors) - self.alpha * self.weights_[1:]
        self.weights_[0] += eta * errors.sum(axis=0)
        
    def _fit(self, X, y, is_classifier):
        """
        Fit the model according to the given training data.
    
        This method performs the core training process using gradient descent. It
        handles both regression and classification tasks, with different loss functions
        and weight updates for each case. The method iterates over the training data,
        updates the model weights, and optionally uses early stopping based on the
        validation score.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data where `n_samples` is the number of samples and
            `n_features` is the number of features.
    
        y : array-like of shape (n_samples,)
            Target values. For classification tasks, these are the class labels. For
            regression tasks, these are the continuous target values.
    
        is_classifier : bool
            Indicates whether the task is classification (`True`) or regression (`False`).
    
        Returns
        -------
        self : object
            Returns self.
    
        Notes
        -----
        The fitting process involves several steps:
        
        1. **Data Validation and Preprocessing**:
           Ensures that `X` and `y` have correct shapes and types. For classification,
           the target values `y` are binarized if there are multiple classes.
    
        2. **Initialization**:
           Initializes the model weights and various parameters needed for the training
           loop.
    
        3. **Gradient Descent Optimization**:
           Iteratively updates the weights to minimize the loss function. The update
           rule for the weights depends on the learning rate schedule and regularization
           term.
    
        4. **Loss Computation**:
           Computes the loss function value for each iteration. For classification, the
           loss function is the cross-entropy loss, while for regression, it is the mean
           squared error.
    
        5. **Early Stopping**:
           Optionally stops the training early if the validation score does not improve
           for a specified number of iterations.
    
        The weight update rule for gradient descent can be formalized as:
    
        .. math::
            w := w - \eta \nabla J(w)
    
        where:
        - :math:`w` represents the weight vector of the model.
        - :math:`\eta` is the learning rate.
        - :math:`\nabla J(w)` denotes the gradient of the loss function with respect
          to the weights.
    
        Examples
        --------
        >>> from gofast.estimators.perceptron import GradientDescentBase
        >>> import numpy as np
        >>> X = np.array([[0, 0], [1, 1], [2, 2]])
        >>> y = np.array([0, 1, 2])
        >>> gd = GradientDescentBase(eta0=0.01, max_iter=100)
        >>> gd._fit(X, y, is_classifier=False)
        >>> gd._predict(X)
    
        See Also
        --------
        SGDClassifier : Linear classifiers (SVM, logistic regression, etc.) with SGD training.
        SGDRegressor : Linear model fitted by minimizing a regularized empirical loss with SGD.
    
        """
        X, y = check_X_y(X, y, estimator=self)
        self.is_classifier = is_classifier
        if is_classifier:
            self.classes_ = np.unique(y)
            n_classes = len(self.classes_)
            self.label_binarizer_ = LabelBinarizer()
            y = self.label_binarizer_.fit_transform(y)
            if n_classes == 2:
                y = np.hstack([1 - y, y])
        else:
            y = y.reshape(-1, 1)
    
        n_samples, n_features = X.shape
        self._initialize_weights(n_features, y.shape[1])
    
        if self.verbose:
            progress_bar = tqdm(total=self.max_iter, ascii=True, ncols=100,
                                desc=f'Fitting {self.__class__.__name__}')
    
        for i in range(self.max_iter):
            if self.shuffle:
                X, y = shuffle(X, y, random_state=self.random_state)
    
            self._update_weights(X, y)
            if is_classifier:
                net_input = self._net_input(X)
                proba = activator(net_input, self.activation)
                # proba = self._sigmoid(net_input)
                loss = -np.mean(y * np.log(proba + 1e-9) + (1 - y) * np.log(1 - proba + 1e-9))
            else:
                loss = ((y - self._net_input(X)) ** 2).mean()
    
            if loss < self.best_loss_ - self.tol:
                self.best_loss_ = loss
                self.no_improvement_count_ = 0
            else:
                self.no_improvement_count_ += 1
    
            if self.early_stopping and self.no_improvement_count_ >= self.n_iter_no_change:
                if self.verbose:
                    print(f"\nEarly stopping at iteration {i+1}")
                    progress_bar.update(self.max_iter - i)
                break
    
            if self.verbose:
                progress_bar.update(1)
            self.n_iter_ += 1
    
        if self.verbose:
            progress_bar.close()
    
        return self

    def _net_input(self, X):
        """ Compute the Net Input """
        return np.dot(X, self.weights_[1:]) + self.weights_[0]

    def _sigmoid(self, z):
        """ Use sigmoid for probaiblity estimation"""
        # Use clipping_threshold to avoid overflow
        z = np.clip(z, -self.clipping_threshold, self.clipping_threshold)  
        return 1 / (1 + np.exp(-z))
    
    def _predict(self, X):
        """
        Predict class labels or continuous values for samples in X.
    
        This method predicts class labels for classification tasks and continuous 
        values for regression tasks based on the net input computed from the model 
        weights and input features.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples for which to predict the class labels or continuous values.
    
        Returns
        -------
        y_pred : array of shape (n_samples,)
            The predicted class labels for classification tasks, or continuous target 
            values for regression tasks.
    
        Notes
        -----
        For classification tasks, the predicted class label for each sample in `X` is 
        determined based on the net input values. If there are two classes, the prediction 
        is based on whether the net input is greater than or equal to zero. For multi-class 
        classification, the predicted class is determined by the class with the highest 
        net input value.
    
        Examples
        --------
        >>> from gofast.estimators.perceptron import GradientDescentBase
        >>> import numpy as np
        >>> X = np.array([[0, 0], [1, 1], [2, 2]])
        >>> y = np.array([0, 1, 2])
        >>> gd = GradientDescentBase(eta0=0.01, max_iter=100)
        >>> gd._fit(X, y, is_classifier=True)
        >>> gd._predict(X)
    
        See Also
        --------
        sklearn.linear_model.SGDClassifier : 
            Linear classifiers (SVM, logistic regression, etc.) with SGD training.
        sklearn.linear_model.SGDRegressor : 
            Linear model fitted by minimizing a regularized empirical loss with SGD.
        
        """
        net_input = self._net_input(X)
        if self.is_classifier:
            if net_input.shape[1] == 1:
                return np.where(net_input >= 0.0, self.label_binarizer_.classes_[1], 
                                self.label_binarizer_.classes_[0])
            else:
                return self.label_binarizer_.inverse_transform(net_input)
        else:
            return net_input

    def _predict_proba(self, X):
        """
        Predict class probabilities for samples in X for classification tasks.
    
        This method returns the probability estimates for each class for classification 
        tasks based on the net input computed from the model weights and input features. 
        For regression tasks, this method is not implemented.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples for which to predict the class probabilities.
    
        Returns
        -------
        y_proba : array of shape (n_samples, n_classes)
            The predicted class probabilities for each sample. The columns correspond to 
            the classes in sorted order, as they appear in the attribute `classes_`.
    
        Notes
        -----
        The predicted probabilities represent the likelihood of each class for the input 
        samples. The probabilities are computed using the sigmoid function applied to the 
        net input values for binary classification tasks. For multi-class classification, 
        the softmax function is used to normalize the probabilities across all classes.
    
        The sigmoid function is defined as:
    
        .. math::
            \sigma(z) = \frac{1}{1 + \exp(-z)}
    
        where :math:`z` is the net input.
    
        For multi-class classification, the softmax function is defined as:
    
        .. math::
            P(y = k | x) = \frac{\exp(\hat{y}_k)}{\sum_{j=1}^{C} \exp(\hat{y}_j)}
    
        where :math:`\hat{y}_k` is the raw model output for class `k`, and :math:`C` is the 
        number of classes.
    
        Examples
        --------
        >>> from gofast.estimators.perceptron import GradientDescentBase
        >>> import numpy as np
        >>> X = np.array([[0, 0], [1, 1], [2, 2]])
        >>> y = np.array([0, 1, 2])
        >>> gd = GradientDescentBase(eta0=0.01, max_iter=100)
        >>> gd._fit(X, y, is_classifier=True)
        >>> gd._predict_proba(X)
    
        See Also
        --------
        sklearn.linear_model.SGDClassifier :
            Linear classifiers (SVM, logistic regression, etc.) with SGD training.
        
        References
        ----------
        .. [1] Bottou, L. (2010). "Large-Scale Machine Learning with Stochastic
          Gradient Descent".Proceedings of COMPSTAT'2010.
        .. [2] Ruder, S. (2016). "An overview of gradient descent optimization
          algorithms".arXiv preprint arXiv:1609.04747.
        """
        if not self.is_classifier:
            raise NotImplementedError(
                "Probability estimates are not available for regression.")
    
        net_input = self._net_input(X)
        proba = activator(net_input, self.activation)
        # proba = self._sigmoid(net_input)
        if proba.shape[1] == 1:
            return np.vstack([1 - proba.ravel(), proba.ravel()]).T
        else:
            return proba / proba.sum(axis=1, keepdims=True)
    
class NeuroFuzzyBase(BaseEstimator):
    """
    NeuroFuzzyBase is a base class for neuro-fuzzy network-based models that 
    integrate fuzzy logic and neural networks to perform classification or 
    regression tasks.

    Parameters
    ----------
    n_clusters : int, default=3
        The number of fuzzy clusters for each feature in the input data. 
        This parameter controls the granularity of the fuzzy logic 
        partitioning.

    eta0 : float, default=0.001
        The initial learning rate for training the `MLPClassifier` or 
        `MLPRegressor`. A smaller learning rate makes the training process 
        more stable but slower.

    max_iter : int, default=200
        Maximum number of iterations for training each `MLPClassifier` or 
        `MLPRegressor`. This parameter determines how long the training 
        will continue if it does not converge before reaching this limit.

    hidden_layer_sizes : tuple, default=(100,)
        The number of neurons in each hidden layer of the `MLPClassifier` or 
        `MLPRegressor`. For example, (100,) means there is one hidden layer 
        with 100 neurons.

    activation : {'identity', 'logistic', 'tanh', 'relu'}, default='relu'
        Activation function for the hidden layer in the `MLPClassifier` or 
        `MLPRegressor`.
        - 'identity': no-op activation, useful for linear bottleneck, 
          returns `f(x) = x`
        - 'logistic': the logistic sigmoid function, returns 
          `f(x) = 1 / (1 + exp(-x))`
        - 'tanh': the hyperbolic tan function, returns 
          `f(x) = tanh(x)`
        - 'relu': the rectified linear unit function, returns 
          `f(x) = max(0, x)`

    solver : {'lbfgs', 'sgd', 'adam'}, default='adam'
        The solver for weight optimization in the `MLPClassifier` or 
        `MLPRegressor`.
        - 'lbfgs': optimizer in the family of quasi-Newton methods
        - 'sgd': stochastic gradient descent
        - 'adam': stochastic gradient-based optimizer proposed by Kingma, 
          Diederik, and Jimmy Ba

    alpha : float, default=0.0001
        L2 penalty (regularization term) parameter for the `MLPClassifier` or 
        `MLPRegressor`. Regularization helps to prevent overfitting by 
        penalizing large weights.

    batch_size : int, default='auto'
        Size of minibatches for stochastic optimizers. If 'auto', batch size 
        is set to `min(200, n_samples)`.

    learning_rate : {'constant', 'invscaling', 'adaptive'}, default='constant'
        Learning rate schedule for weight updates.
        - 'constant': learning rate remains constant
        - 'invscaling': gradually decreases the learning rate at each time step
        - 'adaptive': keeps the learning rate constant to `eta0` as long as 
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

    is_classifier : bool, default=False
        Whether the model is a classifier (`True`) or a regressor (`False`).

    Notes
    -----
    The NeuroFuzzyBase class combines the strengths of fuzzy logic and 
    neural networks to capture nonlinear relationships in data. Fuzzy logic 
    helps in dealing with uncertainty and imprecision, while neural networks 
    provide learning capabilities.

    The fuzzy clustering is performed using c-means clustering. The resulting 
    fuzzy sets are used to transform the input features, which are then fed 
    into an `MLPClassifier` or `MLPRegressor` for learning.

    The objective function for fuzzy c-means clustering is:
    
    .. math::
        \min \sum_{i=1}^{n} \sum_{j=1}^{c} u_{ij}^m \|x_i - v_j\|^2

    where :math:`u_{ij}` is the degree of membership of :math:`x_i` in the 
    cluster :math:`j`, :math:`v_j` is the cluster center, and :math:`m` is a 
    weighting exponent.

    For classification tasks, the `MLPClassifier` uses cross-entropy loss to 
    learn the weights of the network by minimizing the difference between the 
    predicted and actual class probabilities:
    
    .. math::
        \text{Cross-Entropy} = - \sum_{i=1}^{n} \sum_{j=1}^{c} y_{ij} \log(\hat{y}_{ij})

    where :math:`n` is the number of samples, :math:`c` is the number of 
    classes, :math:`y_{ij}` is the actual class label (one-hot encoded), and 
    :math:`\hat{y}_{ij}` is the predicted class probability.

    For regression tasks, the `MLPRegressor` uses the mean squared error (MSE) 
    loss function:
    
    .. math::
        MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2

    where :math:`n` is the number of samples, :math:`y_i` is the actual target 
    value, and :math:`\hat{y}_i` is the predicted target value.

    Examples
    --------
    >>> from gofast.estimators._base import NeuroFuzzyBase
    >>> X = [[0., 0.], [1., 1.]]
    >>> y = [0, 1]
    >>> model = NeuroFuzzyBase(n_clusters=2, max_iter=1000, verbose=True, is_classifier=True)
    >>> model._fit(X, y)
    NeuroFuzzyBase(...)
    >>> model._predict([[2., 2.]])
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
        verbose=False,
        is_classifier=False
    ):
        self.n_clusters = n_clusters
        self.eta0 = eta0
        self.max_iter = max_iter
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.power_t = power_t
        self.tol = tol
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change
        self.max_fun = max_fun
        self.random_state = random_state
        self.verbose = verbose
        self.is_classifier = is_classifier
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder()
        
    def _fuzzify(self, X):
        """
        Transform the input features into fuzzy sets using fuzzy c-means clustering.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to be fuzzified.
    
        Returns
        -------
        antecedents : list of `Antecedent`
            A list of fuzzy antecedents for each feature in the input data. Each 
            antecedent represents the fuzzy clusters for a corresponding feature.
    
        Notes
        -----
        The fuzzification process involves the following steps:
    
        1. **Fuzzy c-means Clustering**:
           For each feature in `X`, fuzzy c-means clustering is applied to 
           partition the feature values into `n_clusters` fuzzy clusters. The 
           objective function for fuzzy c-means clustering is:
           
           .. math::
               \min \sum_{i=1}^{n} \sum_{j=1}^{c} u_{ij}^m \|x_i - v_j\|^2
    
           where :math:`u_{ij}` is the degree of membership of :math:`x_i` in 
           cluster :math:`j`, :math:`v_j` is the cluster center, and :math:`m` 
           is a weighting exponent.
    
        2. **Antecedent Creation**:
           For each feature, an `Antecedent` object is created representing the 
           universe of discourse for that feature. The fuzzy clusters obtained 
           from c-means clustering are then used to define the membership 
           functions for each cluster within the antecedent.
    
        This process transforms the input features into fuzzy sets, which can 
        then be used in the fuzzy inference system.
    
        If `verbose=True`, a progress bar will be displayed during the fuzzification 
        process for each feature.
    
        Examples
        --------
        >>> from gofast.estimators._base import NeuroFuzzyBase
        >>> X = [[0., 0.], [1., 1.]]
        >>> model = NeuroFuzzyBase(n_clusters=2, max_iter=1000, verbose=True, 
                                   is_classifier=True)
        >>> antecedents = model._fuzzify(X)
        >>> len(antecedents)
        2
    
        See Also
        --------
        skfuzzy.cluster.cmeans : Fuzzy c-means clustering algorithm.
        skfuzzy.control.Antecedent : Class representing a fuzzy antecedent.
    
        References
        ----------
        .. [1] J.-S. R. Jang, "ANFIS: Adaptive-Network-based Fuzzy Inference
               System," IEEE Transactions on Systems, Man, and Cybernetics, vol.
               23, no. 3, pp. 665-685, 1993.
        """
        n_features = X.shape[1]
        antecedents = []
        if self.verbose:
            progress_bar = tqdm(
                total=n_features, ascii=True, ncols=100,
                desc="{:<30}".format(f'Fuzzify {self.estimator_name_}'), 
            )
        for i in range(n_features):
            antecedent = Antecedent(np.linspace(0, 1, 100), f'feature_{i}')
            clusters = fuzz.cluster.cmeans(
                X[:, [i]].T, self.n_clusters, 2, error=0.005, maxiter=1000)[0]
            for j, cluster in enumerate(clusters):
                antecedent[f'cluster_{j}'] = fuzz.trimf(antecedent.universe, cluster)
            antecedents.append(antecedent)
            
            if self.verbose: 
                progress_bar.update(1)
        if self.verbose: 
            progress_bar.close()
            
        return antecedents

    
    def _fit(self, X, y, sample_weight=None):
        """
        Fit the NeuroFuzzyBase model according to the given training data.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data where `n_samples` is the number of samples and 
            `n_features` is the number of features.
    
        y : array-like of shape (n_samples,)
            Target values. For classification tasks, these should be class labels.
            For regression tasks, these should be continuous values.
    
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
           Standard scaling is applied to `X`. If `is_classifier` is `True`, 
           `y` is one-hot encoded.
    
        2. **Fuzzification**:
           The input features are transformed into fuzzy sets using fuzzy
           c-means clustering:
           
           .. math::
               \min \sum_{i=1}^{n} \sum_{j=1}^{c} u_{ij}^m \|x_i - v_j\|^2
    
           where :math:`u_{ij}` is the degree of membership of :math:`x_i` in 
           cluster :math:`j`, :math:`v_j` is the cluster center, and :math:`m` 
           is a weighting exponent.
    
        3. **MLP Training**:
           The scaled and fuzzified features are used to train an `MLPClassifier`
           or `MLPRegressor` to map inputs to the encoded target values.
    
        If `verbose=True`, a progress bar will be displayed during the MLP
        training process.
    
        Examples
        --------
        >>> from gofast.estimators._base import NeuroFuzzyBase
        >>> X = [[0., 0.], [1., 1.]]
        >>> y = [0, 1]
        >>> model = NeuroFuzzyBase(n_clusters=2, max_iter=1000, verbose=True, 
                                   is_classifier=True)
        >>> model._fit(X, y)
        NeuroFuzzyBase(...)
    
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

        X, y = check_X_y(X, y, estimator=self)
        
        self.estimator_name_= ( "NeuroFuzzyClassifier" if self.classifier 
                               else "NeuroFuzzyRegressor")
        
        if self.verbose: 
            print(f"Fitting {self.estimator_name_}....")
        
        if self.is_classifier:
            self.mlp_ = MLPClassifier(
                learning_rate_init=self.eta0, 
                max_iter=self.max_iter,
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation, 
                solver=self.solver, 
                alpha=self.alpha, 
                batch_size=self.batch_size, 
                learning_rate=self.learning_rate, 
                power_t=self.power_t, 
                tol=self.tol, 
                momentum=self.momentum, 
                nesterovs_momentum=self.nesterovs_momentum, 
                early_stopping=self.early_stopping, 
                validation_fraction=self.validation_fraction, 
                beta_1=self.beta_1, 
                beta_2=self.beta_2, 
                epsilon=self.epsilon, 
                n_iter_no_change=self.n_iter_no_change, 
                max_fun=self.max_fun, 
                random_state=self.random_state, 
                verbose=self.verbose
            )
        else:
            self.mlp_ = MLPRegressor(
                learning_rate_init=self.eta0, 
                max_iter=self.max_iter,
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation, 
                solver=self.solver, 
                alpha=self.alpha, 
                batch_size=self.batch_size, 
                learning_rate=self.learning_rate, 
                power_t=self.power_t, 
                tol=self.tol, 
                momentum=self.momentum, 
                nesterovs_momentum=self.nesterovs_momentum, 
                early_stopping=self.early_stopping, 
                validation_fraction=self.validation_fraction, 
                beta_1=self.beta_1, 
                beta_2=self.beta_2, 
                epsilon=self.epsilon, 
                n_iter_no_change=self.n_iter_no_change, 
                max_fun=self.max_fun, 
                random_state=self.random_state, 
                verbose=self.verbose
            )
        
        X_scaled = self.scaler.fit_transform(X)
        y_encoded = self.encoder.fit_transform(y.reshape(-1, 1)).toarray(
            ) if self.is_classifier else y
        
        
        self.antecedents_ = self._fuzzify(X_scaled)
        
        if self.verbose:
            for _ in tqdm(range(1), ncols=100, desc="{:<30}".format(
                    f"Fitting {self.estimator_name_}"), ascii=True):
                self.mlp_.fit(X_scaled, y_encoded)
        else:
            self.mlp_.fit(X_scaled, y_encoded)
        
        self.fitted_ = True 
        
        if self.verbose: 
            print(f"Fitting {self.estimator_name_} completed.")
        
        return self

    def _defuzzify(self, y_fuzzy):
        """ The _defuzzify method was designed to convert fuzzy outputs back
        into crisp values, but since we are using MLPRegressor, which provides
        crisp predictions directly, the _defuzzify method is not needed. 
        Instead, the predictions from the MLPRegressor are handled using 
        standard techniques to get the final output classes.
        """
        y = []
        for yf in y_fuzzy:
            max_membership = max(yf, key=lambda k: yf[k])
            y.append(max_membership)
        return np.array(y)
    
    def _predict(self, X):
        """
        Predict using the NeuroFuzzyBase model.
    
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
        The predict process involves the following steps:
    
        1. **Data Scaling**:
           The input features are scaled using the same scaler fitted during
           the training process.
    
        2. **MLP Prediction**:
           The scaled features are fed into the trained `MLPClassifier` or 
           `MLPRegressor` to obtain predicted values.
    
        For classification tasks, the predicted class probabilities are transformed 
        back to class labels using the fitted `OneHotEncoder`.
    
        Examples
        --------
        >>> from gofast.estimators._base import NeuroFuzzyBase
        >>> X = [[2., 2.]]
        >>> model = NeuroFuzzyBase(n_clusters=2, max_iter=1000, verbose=True, 
                                   is_classifier=True)
        >>> model._fit([[0., 0.], [1., 1.]], [0, 1])
        NeuroFuzzyBase(...)
        >>> model._predict(X)
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
        check_is_fitted(self, "fitted_")
        X = check_array(X, accept_sparse=True, estimator=self)
        X_scaled = self.scaler.transform(X)
        y_pred = self.mlp_.predict(X_scaled)
        
        if self.is_classifier:
            y_pred_labels = np.argmax(y_pred, axis=1)
            y_pred_classes = self.encoder.inverse_transform(
                y_pred_labels.reshape(-1, 1)).flatten()
            return y_pred_classes
        else:
            return y_pred
        
    def _predict_proba(self, X):
        """
        Probability estimates using the NeuroFuzzyBase model.

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
        >>> from gofast.estimators._base import NeuroFuzzyBase
        >>> X = [[2., 2.]]
        >>> model = NeuroFuzzyBase(n_clusters=2, max_iter=1000, verbose=True, 
                                   is_classifier=True)
        >>> model._fit([[0., 0.], [1., 1.]], [0, 1])
        NeuroFuzzyBase(...)
        >>> model._predict_proba(X)
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
        if not self.is_classifier:
            raise NotImplementedError(
                "Probability estimates are not available for regressors.")
        
        check_is_fitted(self, "fitted_")
        X = check_array(X, accept_sparse=True, estimator=self)
        
        X_scaled = self.scaler.transform(X)
        y_proba = self.mlp_.predict_proba(X_scaled)
        return y_proba

class FuzzyNeuralNetBase(BaseEstimator):
    """
    FuzzyNeuralNetBase is a base class for ensemble neuro-fuzzy network-based 
    models that integrates fuzzy logic and neural networks to perform 
    classification or regression tasks.

    Parameters
    ----------
    n_estimators : int, default=10
        The number of neural network estimators in the ensemble. This 
        parameter determines how many individual MLP models will be trained 
        and their predictions aggregated.
    
    n_clusters : int, default=3
        The number of fuzzy clusters for each feature in the input data. 
        This parameter controls the granularity of the fuzzy logic 
        partitioning.

    eta0 : float, default=0.001
        The initial learning rate for training the `MLPClassifier` or 
        `MLPRegressor`. A smaller learning rate makes the training process 
        more stable but slower.

    max_iter : int, default=200
        Maximum number of iterations for training each `MLPClassifier` or 
        `MLPRegressor`. This parameter determines how long the training 
        will continue if it does not converge before reaching this limit.

    hidden_layer_sizes : tuple, default=(100,)
        The number of neurons in each hidden layer of the `MLPClassifier` or 
        `MLPRegressor`. For example, (100,) means there is one hidden layer 
        with 100 neurons.

    activation : {'identity', 'logistic', 'tanh', 'relu'}, default='relu'
        Activation function for the hidden layer in the `MLPClassifier` or 
        `MLPRegressor`.
        - 'identity': no-op activation, useful for linear bottleneck, 
          returns `f(x) = x`
        - 'logistic': the logistic sigmoid function, returns 
          `f(x) = 1 / (1 + exp(-x))`
        - 'tanh': the hyperbolic tan function, returns 
          `f(x) = tanh(x)`
        - 'relu': the rectified linear unit function, returns 
          `f(x) = max(0, x)`

    solver : {'lbfgs', 'sgd', 'adam'}, default='adam'
        The solver for weight optimization in the `MLPClassifier` or 
        `MLPRegressor`.
        - 'lbfgs': optimizer in the family of quasi-Newton methods
        - 'sgd': stochastic gradient descent
        - 'adam': stochastic gradient-based optimizer proposed by Kingma, 
          Diederik, and Jimmy Ba

    alpha : float, default=0.0001
        L2 penalty (regularization term) parameter for the `MLPClassifier` or 
        `MLPRegressor`. Regularization helps to prevent overfitting by 
        penalizing large weights.

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
    The FuzzyNeuralNetBase class combines the strengths of fuzzy logic and 
    neural networks to capture nonlinear relationships in data. Fuzzy logic 
    helps in dealing with uncertainty and imprecision, while neural networks 
    provide learning capabilities.

    The fuzzy clustering is performed using c-means clustering. The resulting 
    fuzzy sets are used to transform the input features, which are then fed 
    into an ensemble of `MLPClassifier` or `MLPRegressor` models for learning.

    The objective function for fuzzy c-means clustering is:
    
    .. math::
        \min \sum_{i=1}^{n} \sum_{j=1}^{c} u_{ij}^m \|x_i - v_j\|^2

    where :math:`u_{ij}` is the degree of membership of :math:`x_i` in the 
    cluster :math:`j`, :math:`v_j` is the cluster center, and :math:`m` is a 
    weighting exponent.

    The ensemble of neural networks helps to improve the model's robustness 
    and performance by averaging the predictions of multiple estimators, 
    which reduces the variance and the risk of overfitting.
    
    References
    ----------
    .. [1] J.-S. R. Jang, "ANFIS: Adaptive-Network-based Fuzzy Inference 
           System," IEEE Transactions on Systems, Man, and Cybernetics, vol. 
           23, no. 3, pp. 665-685, 1993.
    """

    def __init__(
        self, 
        n_clusters=3, 
        n_estimators=10,
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
        self.n_clusters = n_clusters
        self.n_estimators = n_estimators
        self.learning_rate_init = eta0
        self.max_iter = max_iter
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.power_t = power_t
        self.tol = tol
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change
        self.max_fun = max_fun
        self.random_state = random_state
        self.verbose = verbose
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder()

    def _fuzzify(self, X):
        """
        Transform the input features into fuzzy sets using fuzzy c-means clustering.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to be fuzzified.
    
        Returns
        -------
        antecedents : list of `Antecedent`
            A list of fuzzy antecedents for each feature in the input data. Each 
            antecedent represents the fuzzy clusters for a corresponding feature.
    
        Notes
        -----
        The fuzzification process involves the following steps:
    
        1. **Fuzzy c-means Clustering**:
           For each feature in `X`, fuzzy c-means clustering is applied to 
           partition the feature values into `n_clusters` fuzzy clusters. The 
           objective function for fuzzy c-means clustering is:
           
           .. math::
               \min \sum_{i=1}^{n} \sum_{j=1}^{c} u_{ij}^m \|x_i - v_j\|^2
    
           where :math:`u_{ij}` is the degree of membership of :math:`x_i` in 
           cluster :math:`j`, :math:`v_j` is the cluster center, and :math:`m` 
           is a weighting exponent.
    
        2. **Antecedent Creation**:
           For each feature, an `Antecedent` object is created representing the 
           universe of discourse for that feature. The fuzzy clusters obtained 
           from c-means clustering are then used to define the membership 
           functions for each cluster within the antecedent.
    
        This process transforms the input features into fuzzy sets, which can 
        then be used in the fuzzy inference system.
    
        Examples
        --------
        >>> from gofast.estimators.perceptron import FuzzyNeuralNetClassifier
        >>> X = [[0., 0.], [1., 1.]]
        >>> clf = FuzzyNeuralNetClassifier(n_clusters=2, n_estimators=5, 
                                           max_iter=1000, verbose=True)
        >>> antecedents = clf._fuzzify(X)
        >>> len(antecedents)
        2
        """

        n_features = X.shape[1]
        antecedents = []
        if self.verbose:
            progress_bar = tqdm(
                total=n_features, ascii=True, ncols=100,
                desc='{:<30}'.format(f'Fuzzyfying {self.estimator_name_}'), 
            )
        for i in range(n_features):
            antecedent = Antecedent(np.linspace(0, 1, 100), f'feature_{i}')
            clusters = fuzz.cluster.cmeans(
                X[:, [i]].T, self.n_clusters, 2, error=0.005, maxiter=1000)[0]
            for j, cluster in enumerate(clusters):
                antecedent[f'cluster_{j}'] = fuzz.trimf(antecedent.universe, cluster)
            antecedents.append(antecedent)
            
            if self.verbose: 
                progress_bar.update (1)
        if self.verbose: 
            progress_bar.close () 
        return antecedents

    def _fit(self, X, y, is_classifier, sample_weight=None):
        """
        Fit the ensemble of neural network models according to the given 
        training data.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data where `n_samples` is the number of samples and
            `n_features` is the number of features.
    
        y : array-like of shape (n_samples,)
            Target values. For classification tasks, these should be class labels.
            For regression tasks, these should be continuous values.
    
        is_classifier : bool
            Flag indicating whether the task is classification (`True`) or 
            regression (`False`).
    
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
           Standard scaling is applied to `X`. If `is_classifier` is `True`, 
           `y` is one-hot encoded.
    
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
           Each model in the ensemble is either an `MLPClassifier` or 
           `MLPRegressor`, depending on the value of `is_classifier`.
    
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
        >>> clf._fit(X, y, is_classifier=True)
        FuzzyNeuralNetClassifier(...)
  
        """
        
        X, y = check_X_y(X, y, estimator=self)
        self.ensemble_ = []
        
        for _ in range(self.n_estimators):
            if is_classifier:
                estimator = MLPClassifier(
                    learning_rate_init=self.eta0, 
                    max_iter=self.max_iter,
                    hidden_layer_sizes=self.hidden_layer_sizes,
                    activation=self.activation, 
                    solver=self.solver, 
                    alpha=self.alpha, 
                    batch_size=self.batch_size, 
                    learning_rate=self.learning_rate, 
                    power_t=self.power_t, 
                    tol=self.tol, 
                    momentum=self.momentum, 
                    nesterovs_momentum=self.nesterovs_momentum, 
                    early_stopping=self.early_stopping, 
                    validation_fraction=self.validation_fraction, 
                    beta_1=self.beta_1, 
                    beta_2=self.beta_2, 
                    epsilon=self.epsilon, 
                    n_iter_no_change=self.n_iter_no_change, 
                    max_fun=self.max_fun, 
                    random_state=self.random_state, 
                    verbose=self.verbose)
            else:
                estimator = MLPRegressor(
                    learning_rate_init=self.eta0, 
                    max_iter=self.max_iter,
                    hidden_layer_sizes=self.hidden_layer_sizes,
                    activation=self.activation, 
                    solver=self.solver, 
                    alpha=self.alpha, 
                    batch_size=self.batch_size, 
                    learning_rate=self.learning_rate, 
                    power_t=self.power_t, 
                    tol=self.tol, 
                    momentum=self.momentum, 
                    nesterovs_momentum=self.nesterovs_momentum, 
                    early_stopping=self.early_stopping, 
                    validation_fraction=self.validation_fraction, 
                    beta_1=self.beta_1, 
                    beta_2=self.beta_2, 
                    epsilon=self.epsilon, 
                    n_iter_no_change=self.n_iter_no_change, 
                    max_fun=self.max_fun, 
                    random_state=self.random_state, 
                    verbose=self.verbose
                    )
            self.ensemble_.append(estimator)
            
        X_scaled = self.scaler.fit_transform(X)
        if is_classifier:
            y_encoded = self.encoder.fit_transform(y.reshape(-1, 1)).toarray()
        else:
            y_encoded = y
        
        self.estimator_name_= ( "FuzzyNeuralNetClassifier" if is_classifier 
                               else "FuzzyNeuralNetRegressor" ) 
        self.antecedents_ = self._fuzzify(X_scaled)
        
        if self.verbose:
            progress_bar = tqdm(
                total=self.n_estimators, ascii=True, ncols=100,
                desc='{:<30}'.format(f'Fitting {self.estimator_name_}'), 
            )
        for estimator in self.ensemble_:
            estimator.fit(X_scaled, y_encoded)
            
            if self.verbose: 
                progress_bar.update (1)
        if self.verbose: 
            progress_bar.close () 
                
        return self
    
class StandardEstimator:
    """Base class for all classes in gofast for parameters retrievals

    Notes
    -----
    All class defined should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "gofast classes should always "
                    "specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                    " %s with constructor %s doesn't "
                    " follow this convention." % (cls, init_signature)
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this class and
            contained sub-objects.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple classes as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                local_valid_params = self._get_param_names()
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self}. "
                    f"Valid parameters are: {local_valid_params!r}."
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self

class _GradientBoostingClassifier:
    r"""
    A simple gradient boosting classifier for binary classification.

    Gradient boosting is a machine learning technique for regression and 
    classification problems, which produces a prediction model in the form 
    of an ensemble of weak prediction models, typically decision trees. It 
    builds the model in a stage-wise fashion like other boosting methods do, 
    and it generalizes them by allowing optimization of an arbitrary 
    differentiable loss function.

    Attributes
    ----------
    n_estimators : int
        The number of boosting stages to be run.
    learning_rate : float
        Learning rate shrinks the contribution of each tree.
    estimators_ : list of DecisionStumpRegressor
        The collection of fitted sub-estimators.

    Methods
    -------
    fit(X, y)
        Build the gradient boosting model from the training set (X, y).
    predict(X)
        Predict class labels for samples in X.
    predict_proba(X)
        Predict class probabilities for X.

    Mathematical Formula
    --------------------
    The model is built in a stage-wise fashion as follows:
    .. math:: 
        F_{m}(x) = F_{m-1}(x) + \\gamma_{m} h_{m}(x)

    where F_{m} is the model at iteration m, \\gamma_{m} is the step size, 
    and h_{m} is the weak learner.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=20, n_classes=2)
    >>> model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
    >>> model.fit(X, y)
    >>> print(model.predict(X)[:5])
    >>> print(model.predict_proba(X)[:5])

    References
    ----------
    - J. H. Friedman, "Greedy Function Approximation: A Gradient Boosting Machine," 1999.
    - T. Hastie, R. Tibshirani, and J. Friedman, "The Elements of Statistical
    Learning," Springer, 2009.

    Applications
    ------------
    Gradient Boosting can be used for both regression and classification problems. 
    It's particularly effective in scenarios where the relationship between 
    the input features and target variable is complex and non-linear. It's 
    widely used in applications like risk modeling, classification of objects,
    and ranking problems.
    """

    def __init__(self, n_estimators=100, learning_rate=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators_ = []

    def fit(self, X, y):
        """
        Fit the gradient boosting classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).
        """
        from .tree import DecisionStumpRegressor 
        X, y = check_X_y(X, y, estimator= self )
        # Convert labels to 0 and 1
        y = np.where(y == np.unique(y)[0], -1, 1)

        F_m = np.zeros(len(y))

        for m in range(self.n_estimators):
            # Compute pseudo-residuals
            residuals = -1 * y * self._sigmoid(-y * F_m)

            # Fit a decision stump to the pseudo-residuals
            stump = DecisionStumpRegressor()
            stump.fit(X, residuals)

            # Update the model
            F_m += self.learning_rate * stump.predict(X)
            self.estimators_.append(stump)

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        The predicted class probabilities are the model's confidence
        scores for the positive class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        proba : array-like of shape (n_samples, 2)
            The class probabilities for the input samples. The columns 
            correspond to the negative and positive classes, respectively.
            
        Examples 
        ----------
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=100, n_features=20, n_classes=2)
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
        model.fit(X, y)
        print(model.predict_proba(X)[:5])
        """
        F_m = sum(self.learning_rate * estimator.predict(X) for estimator in self.estimators_)
        proba_positive_class = self._sigmoid(F_m)
        return np.vstack((1 - proba_positive_class, proba_positive_class)).T

    def _sigmoid(self, z):
        """Compute the sigmoid function."""
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted class labels.
        """
        F_m = sum(self.learning_rate * estimator.predict(X) for estimator in self.estimators_)
        return np.where(self._sigmoid(F_m) > 0.5, 1, 0)
    
class _GradientBoostingRegressor:
    r"""
    A simple gradient boosting regressor for regression tasks.

    Gradient Boosting builds an additive model in a forward stage-wise fashion. 
    At each stage, regression trees are fit on the negative gradient of the loss function.

    Parameters
    ----------
    n_estimators : int, optional (default=100)
        The number of boosting stages to be run. This is essentially the 
        number of decision trees in the ensemble.

    eta0 : float, optional (default=1.0)
        Learning rate shrinks the contribution of each tree. There is a 
        trade-off between eta0 and n_estimators.
        
    max_depth : int, default=1
        The maximum depth of the individual regression estimators.
        
    Attributes
    ----------
    estimators_ : list of DecisionStumpRegressor
        The collection of fitted sub-estimators.

    Methods
    -------
    fit(X, y)
        Fit the gradient boosting model to the training data.
    predict(X)
        Predict continuous target values for samples in X.
    decision_function(X)
        Compute the raw decision scores for the input data.

    Mathematical Formula
    --------------------
    Given a differentiable loss function L(y, F(x)), the model is 
    constructed as follows:
    
    .. math:: 
        F_{m}(x) = F_{m-1}(x) + \\gamma_{m} h_{m}(x)

    where F_{m} is the model at iteration m, \\gamma_{m} is the step size, 
    and h_{m} is the weak learner.

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=100, n_features=1, noise=10)
    >>> model = GradientBoostingRegressor(n_estimators=100, eta0=0.1)
    >>> model.fit(X, y)
    >>> print(model.predict(X)[:5])
    >>> print(model.decision_function(X)[:5])

    References
    ----------
    - J. H. Friedman, "Greedy Function Approximation: A Gradient Boosting Machine," 1999.
    - T. Hastie, R. Tibshirani, and J. Friedman, "The Elements of Statistical Learning," Springer, 2009.

    See Also
    --------
    DecisionTreeRegressor, RandomForestRegressor, AdaBoostRegressor

    Applications
    ------------
    Gradient Boosting Regressor is commonly used in various regression tasks where the relationship 
    between features and target variable is complex and non-linear. It is particularly effective 
    in predictive modeling and risk assessment applications.
    """

    def __init__(self, n_estimators=100, eta0=1.0, max_depth=1):
        self.n_estimators = n_estimators
        self.eta0 = eta0
        self.max_depth=max_depth

    def fit(self, X, y):
        """
        Fit the gradient boosting regressor to the training data.

        The method sequentially adds decision stumps to the ensemble, each one 
        correcting its predecessor. The fitting process involves finding the best 
        stump at each stage that reduces the overall prediction error.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples used for training. Each row in X is a sample and each 
            column is a feature.

        y : array-like of shape (n_samples,)
            The target values (continuous). The regression targets are continuous 
            values which the model will attempt to predict.

        Raises
        ------
        ValueError
            If input arrays X and y have incompatible shapes.

        Notes
        -----
        - The fit process involves computing pseudo-residuals which are the gradients 
          of the loss function with respect to the model's predictions. These are used 
          as targets for the subsequent weak learner.
        - The model complexity increases with each stage, controlled by the learning rate.

        Examples
        --------
        >>> from sklearn.datasets import make_regression
        >>> X, y = make_regression(n_samples=100, n_features=1, noise=10)
        >>> reg = GradientBoostingRegressor(n_estimators=50, eta0=0.1)
        >>> reg.fit(X, y)
        """
        from .tree import DecisionStumpRegressor 
        X, y = check_X_y(X, y, estimator= self )
        if X.shape[0] != y.shape[0]:
            raise ValueError("Mismatched number of samples between X and y")

        # Initialize the prediction to zero
        F_m = np.zeros(y.shape)

        for m in range(self.n_estimators):
            # Compute residuals
            residuals = y - F_m

            # # Fit a regression tree to the negative gradient
            tree = DecisionStumpRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)

            # Update the model predictions
            F_m += self.eta0 * tree.predict(X)

            # Store the fitted estimator
            self.estimators_.append(tree)

        return self

    def predict(self, X):
        """
        Predict continuous target values for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted target values.
        """
        F_m = sum(self.eta0 * estimator.predict(X)
                  for estimator in self.estimators_)
        return F_m

    def decision_function(self, X):
        """
        Compute the raw decision scores for the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        decision_scores : array-like of shape (n_samples,)
            The raw decision scores for each sample.
        """
        return sum(self.eta0 * estimator.predict(X)
                   for estimator in self.estimators_)

