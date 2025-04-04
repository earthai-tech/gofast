# -*- coding: utf-8 -*-

from abc import abstractmethod
from numbers import Real, Integral 
import numpy as np 

from sklearn.base import BaseEstimator

from ..api.property import LearnerMeta 
from ..compat.sklearn import ( 
    Interval, 
    StrOptions, 
    Hidden
)
from .util import activator 

class BaseAdalineStochastic(BaseEstimator, metaclass=LearnerMeta):
    """
    Base class for Adaline Stochastic Gradient Descent.

    This base class implements the core functionalities of the Adaline 
    Stochastic Gradient Descent algorithm. It serves as a foundation for 
    both regression and classification tasks by updating model weights 
    incrementally for each training instance.

    Parameters
    ----------
    eta0 : float, default=0.001
        Initial learning rate. It controls the step size in the weight 
        update. Must be between 0.0 and 1.0. The learning rate governs 
        how much the weights are adjusted with respect to the error 
        between the predicted and true target values.

    max_iter : int, default=1000
        Maximum number of passes over the training dataset (epochs). 
        Determines how many times the algorithm will iterate over the 
        entire training data to update the weights. The value should be 
        large enough to ensure convergence but not too large to avoid 
        overfitting.

    early_stopping : bool, default=False
        Whether to stop training early when the validation error is not 
        improving. If `True`, it will terminate training when the validation 
        error does not improve for `tol` for two consecutive epochs. 
        This can help prevent overfitting and speed up training.

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as a validation set 
        for early stopping. Only used if `early_stopping` is `True`. 
        It determines the fraction of the training data that will be 
        used to evaluate the validation error.

    tol : float, default=1e-4
        Tolerance for the optimization. Training stops when the validation 
        error does not improve by at least `tol` for two consecutive epochs. 
        This helps in stopping the training process when the improvement 
        becomes negligible, preventing unnecessary computations.

    warm_start : bool, default=False
        When set to `True`, reuse the solution of the previous call to 
        `fit` and add more iterations to the estimator, otherwise, just 
        erase the previous solution. This option allows for incremental 
        learning by continuing from a previously trained model.

    learning_rate : {'constant', 'adaptive'}, default='constant'
        Learning rate schedule:
        - 'constant': The learning rate remains the same throughout 
          training.
        - 'adaptive': The learning rate decreases by `eta0_decay` factor 
          after each epoch to ensure more refined updates as the training 
          progresses.

    eta0_decay : float, default=0.99
        Factor by which the learning rate `eta0` is multiplied at each 
        epoch if `learning_rate` is set to 'adaptive'. This factor allows 
        for gradual reduction of the learning rate, improving convergence 
        towards the optimal weights.

    shuffle : bool, default=True
        Whether to shuffle the training data before each epoch to prevent 
        cycles. If `True`, the training data will be randomly shuffled at 
        the start of each epoch, promoting more diverse updates and helping 
        to break any patterns that might occur in the data.

    random_state : int or None, default=None
        Seed used by the random number generator for shuffling and 
        initializing weights. This ensures reproducibility of the training 
        process. If `None`, the seed is determined by the system time.

    verbose : bool, default=False
        Whether to print progress messages to stdout. If `True`, the 
        algorithm will print out the current status of the training process, 
        including details such as the epoch number and current error.

    Attributes
    ----------
    weights_ : ndarray of shape (n_features + 1,) or (n_features + 1, n_outputs)
        Weights after fitting. The weights represent the learned parameters 
        of the model after training. It includes the bias term as well, 
        which is handled separately from the feature weights.

    cost_ : list of float
        Average cost (mean squared error) per epoch. This list contains 
        the value of the cost function at each epoch, which is used to 
        evaluate the model's performance during training.

    Notes
    -----
    Adaline (ADAptive LInear NEuron) is a single-layer artificial neural 
    network and one of the earliest forms of a neural network. The 
    algorithm is sensitive to feature scaling, so it is often beneficial 
    to standardize the features before training.

    The weight update rule in stochastic gradient descent for Adaline is 
    given by:

    .. math::
        \Delta w = \eta_0 (y^{(i)} - \phi(z^{(i)})) x^{(i)}

    where:
    - :math:`\Delta w` is the change in weights.
    - :math:`\eta_0` is the learning rate.
    - :math:`y^{(i)}` is the target value for the \(i\)-th training instance.
    - :math:`\phi(z^{(i)})` is the predicted value using the linear 
      activation function.
    - :math:`x^{(i)}` is the feature vector of the \(i\)-th training 
      instance.

    This method ensures that the model is incrementally adjusted to 
    reduce the mean squared error (MSE) across training instances, 
    promoting convergence to an optimal set of weights.

    Examples
    --------
    >>> from gofast.estimators._adaline import BaseAdalineStochastic
    >>> import numpy as np
    >>> from sklearn.datasets import fetch_california_housing
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.preprocessing import StandardScaler

    >>> # Load the California Housing dataset
    >>> housing = fetch_california_housing()
    >>> X, y = housing.data, housing.target
    >>> X_train, X_test, y_train, y_test = train_test_split(
    >>>     X, y, test_size=0.3, random_state=0)

    >>> # Standardize features
    >>> sc = StandardScaler()
    >>> X_train_std = sc.fit_transform(X_train)
    >>> X_test_std = sc.transform(X_test)

    >>> # Initialize and fit the regressor
    >>> ada_sgd_reg = BaseAdalineStochastic(
    >>>     eta0=0.0001, max_iter=1000, early_stopping=True, 
    >>>     validation_fraction=0.1, tol=1e-4, verbose=True)
    >>> ada_sgd_reg.fit(X_train_std, y_train)
    >>> y_pred = ada_sgd_reg.predict(X_test_std)
    >>> print('Mean Squared Error:', np.mean((y_pred - y_test) ** 2))

    See Also
    --------
    SGDRegressor : Linear regression model fitted by SGD with a variety of 
        loss functions.
    LinearRegression : Ordinary least squares Linear Regression.
    Ridge : Ridge regression.

    References
    ----------
    .. [1] Widrow, B., Hoff, M.E., 1960. Adaptive switching circuits. IRE 
           WESCON Convention Record, New York, 96-104.

    .. [2] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., 
           Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., 
           Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., 
           and Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. 
           Journal of Machine Learning Research, 12, 2825-2830.
    """
    
    _parameter_constraints: dict = {
        "eta0": [Interval(Real, 0., 1., closed="neither")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "early_stopping": [bool],
        "validation_fraction": [Interval(Real, 0., 1., closed="neither")],
        "tol": [Interval(Real, 0, None, closed="neither")],
        "warm_start": [bool],
        "learning_rate": [StrOptions({"constant", "adaptive"})],
        "eta0_decay": [Interval(Real, 0., 1., closed="neither")],
        "shuffle": [bool],
        "random_state": [Integral, None],
        "activation": [StrOptions(
            {'sigmoid', 'relu', 'leaky_relu', 'identity', 'elu', 
             'tanh', 'softmax'}), callable],
        "epsilon": [Hidden(Interval(Real, 0, 1, closed='neither'))], 
        "verbose": [bool, int],
    }

    @abstractmethod
    def __init__(
        self, 
        eta0=0.001, 
        max_iter=1000, 
        early_stopping=False, 
        validation_fraction=0.1, 
        tol=1e-4, 
        warm_start=False, 
        learning_rate='constant', 
        eta0_decay=0.99, 
        shuffle=True,
        random_state=None, 
        activation="sigmoid", 
        epsilon=1e-8, 
        verbose=False
    ):
        self.eta0 = eta0
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.tol = tol
        self.warm_start = warm_start
        self.learning_rate = learning_rate
        self.eta0_decay = eta0_decay
        self.shuffle = shuffle
        self.random_state = random_state
        self.activation= activation 
        self.epsilon=epsilon 
        self.verbose = verbose
 
    def _initialize_weights(self, n_features, n_outputs=1):
        """
        Initialize the weights for the model.

        Parameters
        ----------
        n_features : int
            Number of features in the input data.

        n_outputs : int, default=1
            Number of outputs. This is typically 1 for regression and the 
            number of classes for classification.

        Notes
        -----
        The weights are initialized using a normal distribution with mean 0 
        and standard deviation 0.01. This initialization includes an extra 
        weight for the bias term.
        """
        rgen = np.random.RandomState(self.random_state)
        self.weights_ = rgen.normal(loc=0.0, scale=0.01, size=1 + n_features)

    def _update_weights(self, xi, target, idx):
        """
        Update the weights using a single training example.

        Parameters
        ----------
        xi : array-like, shape (n_features,)
            Training vector of the current example.

        target : float
            Target value of the current example.

        idx : int
            Index of the output. Used for multi-output tasks.

        Returns
        -------
        error : float
            The error of the prediction for the current example.

        Notes
        -----
        The weights are updated according to the Adaline rule:

        .. math::
            \Delta w = \eta_0 (y - \phi(z)) x

        where :math:`\eta_0` is the learning rate, :math:`y` is the target 
        value, :math:`\phi(z)` is the predicted value, and :math:`x` is the 
        input feature vector.
        """
        error = target - self.net_activation(xi, idx)
        self.weights_[1:] += self.eta0 * xi * error
        self.weights_[0] += self.eta0 * error
        return error

    def net_input(self, X, idx=None):
        """
        Calculate the net input.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        idx : int, optional
            Index of the output. Used for multi-output tasks in classification.

        Returns
        -------
        net_input : array-like, shape (n_samples,)
            Net input values.

        Notes
        -----
        The net input is calculated as the dot product of the input features 
        and the weights, plus the bias term.
        """
        if self._estimator_type=="classifier":
            return np.dot(X, self.weights_[1:, idx]) + self.weights_[0, idx]
        else:
            return np.dot(X, self.weights_[1:]) + self.weights_[0]

    def net_activation(self, X, idx):
        """
        Compute the linear activation.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        idx : int
            Index of the output. Used for multi-output tasks.

        Returns
        -------
        activation : array-like, shape (n_samples,)
            Activation values.

        Notes
        -----
        This method computes the linear activation, which is the same as the 
        net input for Adaline.
        """
        return self.net_input(X, idx)
    

    def _activator(self, z):
        """
        Apply activation function to net inputs (internal helper).
    
        Parameters
        ----------
        z : array-like
            Net input calculated as :math:`z = X \cdot w + b`
    
        Returns
        -------
        activated : ndarray
            Activation output transformed through specified function
    
        Notes
        -----
        Supported activation functions:
        - Sigmoid: :math:`1/(1 + e^{-z})`
        - Softmax: :math:`e^{z_i}/\sum_j e^{z_j}`
        - ReLU: :math:`\max(0, z)`
        - Tanh: :math:`\tanh(z)`
        - Identity: Linear passthrough
    
        Custom functions must maintain numerical stability and
        differentiability.
        """
        return activator(z, activation=self.activation)

class BaseAdaline(BaseEstimator, metaclass=LearnerMeta):
    """
    Base class for Adaline Regressor and Adaline Classifier.

    This class implements the core functionalities of the Adaline algorithm 
    using Stochastic Gradient Descent (SGD). Both regression and classification 
    tasks inherit from this class to use the common methods, while each subclass 
    implements its task-specific logic.

    Parameters
    ----------
    eta0 : float, default=0.01
        The learning rate, determining the step size at each iteration while
        moving toward a minimum of the loss function. The value must be between
        0.0 and 1.0.

    max_iter : int, default=1000
        The number of passes over the training dataset (epochs).

    early_stopping : bool, default=False
        Whether to stop training early when validation error is not improving.

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for 
        early stopping. Only used if `early_stopping` is True.

    tol : float, default=1e-4
        The tolerance for the optimization. Training stops when the validation 
        error does not improve by at least `tol` for two consecutive epochs.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to `fit` and 
        add more iterations to the estimator, otherwise, just erase the previous 
        solution.

    shuffle : bool, default=True
        Whether to shuffle the training data before each epoch. Shuffling helps 
        in preventing cycles and ensures that individual samples are encountered 
        in different orders.

    random_state : int, default=None
        The seed of the pseudo-random number generator for shuffling the data 
        and initializing the weights.

    verbose : int, default=0
        The level of verbosity. Ranges from 0 to 7, where:
        - 0: No output
        - 1: Displays progress bar
        - 2-7: Detailed output with increasing levels of verbosity.
    
    epsilon : float, default=1e-8
        Small constant to prevent numerical instability during weight updates.

    Attributes
    ----------
    weights_ : array-like, shape (n_features,)
        Weights assigned to the features after fitting the model.

    cost_ : list
        The sum of squared errors (cost) accumulated over the training epochs.
    """
    
    _parameter_constraints: dict ={
        "eta0": [Interval(Real, 0.0, 1.0, closed="neither")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "early_stopping": [bool],
        "validation_fraction": [Interval(Real, 0.0, 1.0, closed="neither")],
        "tol": [Interval(Real, 0.0, None, closed="neither")],
        "warm_start": [bool],
        "shuffle": [bool],
        "random_state": [Integral, None],
        "epsilon": [Hidden(Interval(Real, 0.0, None, closed="neither"))]
    }
    
    @abstractmethod 
    def __init__(
        self, eta0=0.01, max_iter=1000, early_stopping=False, 
        validation_fraction=0.1, tol=1e-4, warm_start=False, 
        shuffle=True, random_state=None, epsilon=1e-8, 
        verbose=0
    ):
        self.eta0 = eta0
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.tol = tol
        self.warm_start = warm_start
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.epsilon = epsilon
        
    def _initialize_weights(self, n_features, n_outputs=1):
        """
        Initialize weights for the model.
        """
        rgen = np.random.RandomState(self.random_state)
        self.weights_ = rgen.normal(
            loc=0.0, scale=0.01, size=(n_features + 1, n_outputs))

    def _update_weights(self, xi, target, idx):
        """
        Update weights using a single training example.
        """
        error = target - self.activation(xi, idx)
        
        # Prevent overflow by limiting the magnitude of updates
        update = self.eta0 * xi * error
        self.weights_[1:, idx] += np.clip(
            update, -self.epsilon, self.epsilon)  
        self.weights_[0, idx] += np.clip(
            self.eta0 * error, -self.epsilon, self.epsilon)  
        return error

    def net_input(self, X, idx):
        """
        Calculate net input.
        """
        return np.dot(X, self.weights_[1:, idx]) + self.weights_[0, idx]

    def activation(self, X, idx):
        """
        Compute the activation.
        """
        return self.net_input(X, idx)

    def _is_classifier(self):
        """
        Flag to indicate that this is a regressor.
        """
        return False
    
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

  