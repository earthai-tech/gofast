# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <eta0noyau@gmail.com>

from __future__ import annotations 
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import shuffle

try:from sklearn.utils import type_of_target
except: from ..tools.coreutils import type_of_target 
from ..tools.validator import check_X_y, get_estimator_name, check_array 
from ..tools.validator import check_is_fitted

__all__= [
    
        "AdalineClassifier","AdalineMixte","AdalineRegressor",
        "AdalineStochasticRegressor","AdalineStochasticClassifier",
    ]

class AdalineStochasticRegressor(BaseEstimator, RegressorMixin):
    r"""
    Adaline Stochastic Gradient Descent Regressor.

    This regressor implements the Adaptive Linear Neuron (Adaline) algorithm 
    using Stochastic Gradient Descent (SGD) for linear regression tasks. It 
    is particularly well-suited for large datasets due to its stochastic 
    nature, updating the model weights incrementally for each training instance.

    In SGD, the cost function is approximated for each instance, rather than
    being computed over the entire dataset. The Adaline algorithm updates weights
    based on a linear activation function and a continuous cost function 
    (Mean Squared Error, MSE).

    The weight update rule in SGD for Adaline is given by:

    .. math::
        w := w + \eta0 (y^{(i)} - \phi(z^{(i)})) x^{(i)}

    where:
    - :math:`w` is the weight vector.
    - :math:`\eta0` is the learning rate.
    - :math:`y^{(i)}` is the true value for the \(i\)-th training instance.
    - :math:`\phi(z^{(i)})` is the predicted value using the linear activation function.
    - :math:`x^{(i)}` is the feature vector of the \(i\)-th training instance.

    This method ensures that the model is incrementally adjusted to reduce the MSE 
    across training instances, promoting convergence to an optimal set of weights.

    Parameters
    ----------
    eta0 : float, default=0.0001
        Learning rate (between 0.0 and 1.0).

    max_iter : int, default=10
        Number of passes over the training dataset (epochs).

    shuffle : bool, default=True
        Whether to shuffle training data before each epoch to prevent cycles.

    random_state : int, default=None
        Seed used by the random number generator for shuffling and initializing 
        weights.

    Attributes
    ----------
    weights_ : 1d-array
        Weights after fitting.

    cost_ : list
        Average cost (mean squared error) per epoch.

    Notes
    -----
    - Adaline SGD is sensitive to feature scaling and it often beneficial
      to standardize the features before training.
    - Since the algorithm uses a random shuffle, setting a `random_state`
      ensures reproducibility.

    See Also
    --------
    SGDRegressor : Linear regression model fitted by SGD with a variety of 
        loss functions.
    LinearRegression : Ordinary least squares Linear Regression.
    Ridge : Ridge regression.

    Examples
    --------
    >>> from sklearn.datasets import load_boston
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.preprocessing import StandardScaler
    >>> from gofast.estimators.adaline import AdalineStochasticRegressor
    >>> boston = load_boston()
    >>> X, y = boston.data, boston.target
    >>> X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

    # Standardize features
    >>> sc = StandardScaler()
    >>> X_train_std = sc.fit_transform(X_train)
    >>> X_test_std = sc.transform(X_test)

    >>> ada_sgd_reg = AdalineStochasticRegressor(eta0=0.0001, max_iter=1000)
    >>> ada_sgd_reg.fit(X_train_std, y_train)
    >>> y_pred = ada_sgd_reg.predict(X_test_std)
    >>> print('Mean Squared Error:', np.mean((y_pred - y_test) ** 2))
    """

    def __init__(
        self, 
        eta0=0.0001, 
        max_iter=10, 
        shuffle=True,
        random_state=None 
        ):
        self.eta0 = eta0
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state=random_state 

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
        rgen = np.random.RandomState(self.random_state)
        self.weights_ = rgen.normal(loc=0. , scale =.01 , size = 1 + X.shape[1])
        # self.weights_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.max_iter):
            if self.shuffle:
                X, y = shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                error = target - self.predict(xi.reshape (1, -1))
                self.weights_[1:] += self.eta0 * xi * error
                self.weights_[0] += self.eta0 * error
                cost.append(error**2 / 2.0)
            self.cost_.append(np.mean(cost))
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

class AdalineStochasticClassifier(BaseEstimator, ClassifierMixin):
    r"""
    Adaptive Linear Neuron Classifier with Stochastic Gradient Descent.

    This classifier implements a stochastic gradient descent algorithm for 
    adaptive linear neurons. Stochastic Gradient Descent (SGD) is an efficient 
    approach to discriminative learning of linear classifiers under convex loss 
    functions such as (linear) Support Vector Machines and Logistic Regression. 
    SGD has been successfully applied to large-scale and sparse machine learning 
    problems often encountered in text classification and natural language 
    processing.

    The principle behind SGD is to update the model parameters (weights) 
    incrementally for each training example. In the context of this classifier, 
    the weight update is performed as follows:

    .. math::
        \Delta w = \sum_{i} (y^{(i)} - \phi(z^{(i)})) x^{(i)}

    Here, :math:`\Delta w` represents the change in weights, :math:`y^{(i)}` is the 
    true label, :math:`\phi(z^{(i)})` is the predicted label, and :math:`x^{(i)}` is 
    the input feature vector.

    The weights are updated incrementally for each training example:

    .. math::
        w := w + \eta0 (y^{(i)} - \phi(z^{(i)})) x^{(i)}

    where :math:`\eta0` is the learning rate. This incremental update approach 
    helps in adapting the classifier more robustly to large and varying datasets, 
    ensuring that each training instance directly influences the model's learning.

    Parameters
    ----------
    eta0 : float, optional (default=0.01)
        The learning rate, determining the step size at each iteration while 
        moving toward a minimum of a loss function. The value must be between 
        0.0 and 1.0.

    max_iter : int, optional (default=10)
        The number of passes over the training data (aka epochs).

    shuffle : bool, optional (default=True)
        Whether to shuffle the training data before each epoch. Shuffling helps 
        in preventing cycles and ensures that individual samples are encountered 
        in different orders.

    random_state : int, optional (default=42)
        The seed of the pseudo random number generator to use when shuffling the 
        data and initializing the weights.

    Attributes
    ----------
    weights_ : array-like, shape (n_features,)
        Weights assigned to the features after fitting the model.

    cost_ : list
        The sum of squared errors (cost) accumulated over the training epochs. 
        This can be used to evaluate how the model's performance has improved 
        over time.

    Notes
    -----
    Stochastic Gradient Descent is sensitive to feature scaling, so it is 
    highly recommended to scale your data. For example, use 
    `sklearn.preprocessing.StandardScaler` for standardization.

    References
    ----------
    [1] Widrow, B., Hoff, M.E., 1960. Adaptive switching circuits. IRE WESCON 
        Convention Record, New York, 96-104.

    Examples
    --------
    >>> from gofast.estimators.adaline import AdalineStochasticClassifier
    >>> X = [[0., 0.], [1., 1.]]
    >>> y = [0, 1]
    >>> clf = AdalineStochasticClassifier(eta0=0.01, max_iter=10)
    >>> clf.fit(X, y)
    AdalineStochasticClassifier(eta0=0.01, max_iter=10)

    See Also
    --------
    AdalineGradientDescent : Gradient Descent variant of Adaline.
    SGDClassifier : Scikit-learn's SGD classifier.

    """

    def __init__(self, eta0:float = .01 , max_iter: int = 50 , shuffle=True, 
                 random_state:int = None ) :
        self.eta0=eta0 
        self.max_iter=max_iter 
        self.shuffle=shuffle 
        self.random_state=random_state 
        
        
    def fit(self , X, y ): 
        """ Fit the training data 
        
        Parameters 
        ----------
        X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
            Training set; Denotes data that is observed at training and 
            prediction time, used as independent variables in learning. 
            When a matrix, each sample may be represented by a feature vector, 
            or a vector of precomputed (dis)similarity with each training 
            sample. :code:`X` may also not be a matrix, and may require a 
            feature extractor or a pairwise metric to turn it into one  before 
            learning a model.
        y: array-like, shape (M, ) ``M=m-samples``, 
            train target; Denotes data that may be observed at training time 
            as the dependent variable in learning, but which is unavailable 
            at prediction time, and is usually the target of prediction. 
        
        Returns 
        --------
        self: `Perceptron` instance 
            returns ``self`` for easy method chaining.
        """ 
        self.weights_initialized_ =False 
        X, y = check_X_y(
            X, 
            y, 
            estimator = get_estimator_name(self), 
            )
        self._init_weights (X.shape[1])
        self.cost_=list() 
        for i in range(self.max_iter ): 
            if self.shuffle: 
                X, y = self._shuffle (X, y) 
            cost =[] 
            for xi , target in zip(X, y) :
                cost.append(self._update_weights(xi, target)) 
            avg_cost = sum(cost)/len(y) 
            self.cost_.append(avg_cost) 
        
        return self 
    
    def partial_fit(self, X, y):
        """
        Fit training data without reinitialising the weights 
        
        Parameters
        ----------
        X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
            Training set; Denotes data that is observed at training and 
            prediction time, used as independent variables in learning. 
            When a matrix, each sample may be represented by a feature vector, 
            or a vector of precomputed (dis)similarity with each training 
            sample. :code:`X` may also not be a matrix, and may require a 
            feature extractor or a pairwise metric to turn it into one  before 
            learning a model.
        y: array-like, shape (M, ) ``M=m-samples``, 
            train target; Denotes data that may be observed at training time 
            as the dependent variable in learning, but which is unavailable 
            at prediction time, and is usually the target of prediction. 
        
        Returns 
        --------
        self: `Perceptron` instance 
            returns ``self`` for easy method chaining.

        """
        X, y = check_X_y(
            X, 
            y, 
            estimator = get_estimator_name(self),  
            )
        
        if not self.weights_initialized_ : 
           self._init_weights (X.shape[1])
          
        if y.ravel().shape [0]> 1: 
            for xi, target in zip(X, y):
                self._update_weights (xi, target) 
        else: 
            self._update_weights (X, y)
                
        return self 
    
    def _shuffle (self, X, y):
        """
        Shuffle training data 
        
        Parameters
        ----------
        X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
            Training set; Denotes data that is observed at training and 
            prediction time, used as independent variables in learning. 
            When a matrix, each sample may be represented by a feature vector, 
            or a vector of precomputed (dis)similarity with each training 
            sample. :code:`X` may also not be a matrix, and may require a 
            feature extractor or a pairwise metric to turn it into one  before 
            learning a model.
        y: array-like, shape (M, ) ``M=m-samples``, 
            train target; Denotes data that may be observed at training time 
            as the dependent variable in learning, but which is unavailable 
            at prediction time, and is usually the target of prediction. 

        Returns
        -------
        Training and target data shuffled  

        """
        r= self.rgen_.permutation(len(y)) 
        return X[r], y[r]
    
    def _init_weights (self, m): 
        """
        Initialize weights with small random numbers 

        Parameters
        ----------
        m : int 
           random number for weights initialization .

        """
        self.rgen_ =  np.random.RandomState(self.random_state)
        self.weights_ = self.rgen_.normal(loc=.0 , scale=.01, size = 1+ m) 
        self.weights_initialized_ = True 
        
    def _update_weights (self, X, y):
        """
        Adeline learning rules to update the weights 

        Parameters
        ----------
        X : Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
            Training set for initializing
        y :array-like, shape (M, ) ``M=m-samples``, 
            train target for initializing 

        Returns
        -------
        cost: list,
            sum-squared errors 

        """
        output = self.activation (self.net_input(X))
        errors =(y - output ) 
        self.weights_[1:] += self.eta0 * X.dot(errors) 
        cost = errors **2 /2. 
        
        return cost 
    
    def net_input (self, X):
        """
        Compute the net input X 
        
        Parameters
        ----------
        X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
            Training set; Denotes data that is observed at training and 
            prediction time, used as independent variables in learning. 
            When a matrix, each sample may be represented by a feature vector, 
            or a vector of precomputed (dis)similarity with each training 
            sample. :code:`X` may also not be a matrix, and may require a 
            feature extractor or a pairwise metric to turn it into one  before 
            learning a model.

        Returns
        -------
        weight net inputs 

        """
        check_is_fitted (self, 'weights_') 
        return np.dot (X, self.weights_[1:]) + self.weights_[0] 

    def activation (self, X):
        """
        Compute the linear activation 

        Parameters
        ----------
        X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
            Training set; Denotes data that is observed at training and 
            prediction time, used as independent variables in learning. 
            When a matrix, each sample may be represented by a feature vector, 
            or a vector of precomputed (dis)similarity with each training 
            sample. :code:`X` may also not be a matrix, and may require a 
            feature extractor or a pairwise metric to turn it into one  before 
            learning a model.

        Returns
        -------
        X: activate NDArray 

        """
        return X 
    
    def predict (self, X):
        """
        Predict the  class label after unit step
        
        Parameters
        ----------
        X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
            Training set; Denotes data that is observed at training and 
            prediction time, used as independent variables in learning. 
            When a matrix, each sample may be represented by a feature vector, 
            or a vector of precomputed (dis)similarity with each training 
            sample. :code:`X` may also not be a matrix, and may require a 
            feature extractor or a pairwise metric to turn it into one  before 
            learning a model.

        Returns
        -------
        ypred: predicted class label after the unit step  (1, or -1)
        """
        check_is_fitted (self, 'weights_') 
        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )
        return np.where (self.activation(self.net_input(X))>=0. , 1, -1)
    
    def predict_proba(self, X):
        """
        Predict class probabilities using the Adaline Stochastic 
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
        check_is_fitted(self, 'weights_')
        X = check_array(X, accept_sparse=True)

        # Apply the linear model and logistic sigmoid function
        net_input = self.net_input(X)
        proba_positive_class = 1 / (1 + np.exp(-net_input))
        proba_negative_class = 1 - proba_positive_class

        return np.vstack((proba_negative_class, proba_positive_class)).T

    def __repr__(self): 
        """ Represent the output class """
        
        tup = tuple (f"{key}={val}".replace ("'", '') for key, val in 
                     self.get_params().items() )
        
        return self.__class__.__name__ + str(tup).replace("'", "") 

class AdalineRegressor(BaseEstimator, RegressorMixin):
    r"""
    Adaline Gradient Descent Regressor.

    `AdalineRegressor` is based on the principles of Adaptive Linear Neurons (Adaline),
    employing the gradient descent optimization algorithm for regression tasks.
    The AdalineRegressor fits a linear model to the data by minimizing the sum
    of squared errors between the observed targets in the dataset and the targets
    predicted by the linear approximation.

    The weight update in the gradient descent step is performed using the following
    rule:

    .. math::
        w := w + \eta0 \sum_{i} (y^{(i)} - \phi(z^{(i)})) x^{(i)}

    where:
    - :math:`w` represents the weight vector.
    - :math:`\eta0` is the learning rate.
    - :math:`y^{(i)}` is the actual observed value for the \(i\)-th sample.
    - :math:`\phi(z^{(i)})` is the predicted value obtained from the linear model.
    - :math:`x^{(i)}` is the feature vector of the \(i\)-th sample.

    This weight update mechanism allows the model to iteratively adjust the weights
    to minimize the overall prediction error, thereby refining the model's accuracy
    over training iterations.

    Parameters
    ----------
    eta0 : float, optional (default=0.01)
        The learning rate, determining the step size at each iteration while
        moving toward a minimum of the loss function. The value should be between
        0.0 and 1.0.

    max_iter : int, optional (default=50)
        The number of passes over the training dataset (epochs).

    random_state : int, optional (default=None)
        The seed of the pseudo random number generator for shuffling the data 
        and initializing the weights.

    Attributes
    ----------
    weights_ : array-like, shape (n_features,)
        Weights assigned to the features after fitting the model.

    errors_ : list
        The sum of squared errors (residuals) after each epoch. It can be used to
        analyze the convergence behavior of the algorithm during training.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import train_test_split
    >>> from gofast.estimators.adaline import AdalineRegressor
    >>> diabetes = load_diabetes()
    >>> X = diabetes.data
    >>> y = diabetes.target
    >>> X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)
    >>> ada = AdalineRegressor(eta0=0.01, max_iter=50)
    >>> ada.fit(X_train, y_train)
    >>> y_pred = ada.predict(X_test)
    >>> print('Mean Squared Error:', np.mean((y_pred - y_test) ** 2))

    Notes
    -----
    This implementation is suitable for learning linear relationships but might
    not be optimal for non-linear datasets. Feature scaling (e.g., using
    `StandardScaler` from sklearn.preprocessing) is recommended with gradient 
    descent algorithms.

    See Also
    --------
    LinearRegression : Ordinary least squares Linear Regression from Scikit-Learn.
    SGDRegressor : Linear model fitted by minimizing a regularized empirical loss 
                   with SGD from Scikit-Learn.

    """

    def __init__(self, eta0=0.01, max_iter=50, random_state =None):
        self.eta0 = eta0
        self.max_iter = max_iter
        self.random_state=random_state 

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
        rgen = np.random.RandomState(self.random_state)
        self.weights_ = rgen.normal(loc=0. , scale =.01 , size = 1 + X.shape[1]
                              )
        # self.weights_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.max_iter):
            errors = 0
            for xi, target in zip(X, y):
                error = target - self.predict(xi.reshape ( 1, -1))
                update = self.eta0 * error
                self.weights_[1:] += update * xi
                self.weights_[0] += update
                errors += error ** 2
            self.errors_.append(errors)
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
            # ensure_2d=False
            )
        return self.net_input(X)

class AdalineClassifier(BaseEstimator, ClassifierMixin):
    r"""
    Adaline Gradient Descent Classifier.

    This classifier is based on the principles of Adaptive Linear Neurons 
    (Adaline), employing the gradient descent optimization algorithm for 
    binary classification tasks. The AdalineClassifier fits a linear decision 
    boundary to the data by minimizing the sum of squared errors between 
    the observed labels in the dataset and the labels predicted by the 
    linear approximation.

    The weight update in the gradient descent step is performed using the 
    following rule:

    .. math::
        w := w + \eta0 \sum_{i} (y^{(i)} - \phi(z^{(i)})) x^{(i)}

    where:
    - :math:`w` represents the weight vector.
    - :math:`\eta0` is the learning rate.
    - :math:`y^{(i)}` is the actual label for the \(i\)-th sample.
    - :math:`\phi(z^{(i)})` is the predicted label, obtained from the
      linear model activation.
    - :math:`x^{(i)}` is the feature vector of the \(i\)-th sample.

    This update mechanism incrementally adjusts the weights to minimize 
    the prediction error, refining the model's ability to classify new 
    samples accurately.

    Parameters
    ----------
    eta0 : float, optional (default=0.01)
        The learning rate, determining the step size at each iteration while
        moving toward a minimum of the loss function. The value should be between
        0.0 and 1.0.

    max_iter : int, optional (default=50)
        The number of passes over the training dataset (epochs).

    random_state : int, optional (default=None)
        The seed of the pseudo random number generator for shuffling the data 
        and initializing the weights.

    Attributes
    ----------
    weights_ : array-like, shape (n_features,)
        Weights assigned to the features after fitting the model.

    errors_ : list
        The number of misclassifications in each epoch. This can be used to 
        analyze the convergence behavior of the algorithm during training.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> from gofast.estimator import AdalineClassifier
    >>> iris = load_iris()
    >>> X = iris.data[:, :2]
    >>> y = iris.target
    >>> X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)
    >>> ada = AdalineClassifier(eta0=0.01, max_iter=50)
    >>> ada.fit(X_train, y_train)
    >>> y_pred = ada.predict(X_test)
    >>> print('Accuracy:', np.mean(y_pred == y_test))

    Notes
    -----
    This implementation is intended for binary classification tasks. For multi-class
    classification, one could use strategies such as One-vs-Rest (OvR). Feature scaling
    (e.g., using `StandardScaler` from sklearn.preprocessing) is recommended with 
    gradient descent algorithms.

    See Also
    --------
    LogisticRegression : Logistic Regression classifier from Scikit-Learn.
    SGDClassifier : Linear model fitted by minimizing a regularized empirical loss 
                    with SGD from Scikit-Learn.

    """

    def __init__(self, eta0=0.01, max_iter=50, random_state=None ):
        self.eta0 = eta0
        self.max_iter = max_iter
        self.random_state=random_state  
        
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
        rgen = np.random.RandomState(self.random_state)
        self.weights_ = rgen.normal(loc=0. , scale =.01 , size = 1 + X.shape[1]
                              )
        # self.weights_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.max_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta0 * (target - self.predict(xi.reshape(1,-1)))
                self.weights_[1:] += update * xi
                self.weights_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.weights_[1:]) + self.weights_[0]

    def predict(self, X):
        """Predict  class label after unit step
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted class label from the input samples.
        
        """
        check_is_fitted (self, 'weights_') 
        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    
    def predict_proba(self, X):
        """
        Predict class probabilities using the Adaline Classifier model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        proba : array-like of shape (n_samples, 2)
            The class probabilities of the input samples.
        """
        check_is_fitted(self, 'weights_')
        X = check_array(X, accept_sparse=True)
        # Apply the linear model
        net_input = self.net_input(X)

        # Use the logistic sigmoid function to estimate probabilities
        proba_positive_class = 1 / (1 + np.exp(-net_input))
        proba_negative_class = 1 - proba_positive_class

        return np.vstack((proba_negative_class, proba_positive_class)).T

class AdalineMixte(BaseEstimator, ClassifierMixin, RegressorMixin):
    r"""
    Adaline Mixte for Dual Regression and Classification Tasks.

    The ADAptive LInear NEuron (Adaline) Mixte is a foundational model in 
    machine learning, capable of performing both regression and classification.
    This model is particularly noted for its historical significance, having 
    been developed by Bernard Widrow and his doctoral student Ted Hoff in the 
    early 1960s. It extends the concepts introduced by Frank Rosenblatt's 
    perceptron by utilizing a continuous, linear activation function for 
    weight adjustment, instead of the step function used in the perceptron.

    Adaline is critical in demonstrating the importance of defining and 
    minimizing a continuous cost function, paving the way for the development 
    of advanced machine learning algorithms such as Logistic Regression, 
    Support Vector Machines, and various regression models.

    The Widrow-Hoff learning rule, or the Adaline rule, differs from Rosenblatt's 
    perceptron by updating weights based on a linear activation function rather 
    than a binary step function. This approach allows for the use of gradient 
    descent  optimization methods due to the differentiability of the cost 
    function.

    Key Contributions and Innovations:
    - Adaline's primary innovation is its cost function, which is continuous 
      and differentiable. This allows for the application of gradient descent,
      leading to more efficient and robust optimization.
    - The introduction of the Widrow-Hoff learning rule with Adaline minimizes
      the mean squared error between actual outputs and predictions, fostering
      a stable and convergent learning process.

    Mathematical Formulation:
    The weight update rule for Adaline is defined as follows:

    .. math::
        w := w + \eta0 \sum_{i} (y^{(i)} - \phi(w^T x^{(i)})) x^{(i)}

    where:
    - :math:`\eta0` is the learning rate.
    - :math:`y^{(i)}` is the true value or label for the \(i\)-th sample.
    - :math:`\phi(w^T x^{(i)})` is the predicted value, where :math:`\phi` 
      is the identity function (i.e., :math:`\phi(z) = z`), reflecting the 
      continuous nature of the linear activation function.
    - :math:`x^{(i)}` is the feature vector of the \(i\)-th sample.

    Parameters
    ----------
    eta0 : float
        Learning rate (between 0.0 and 1.0). Determines the step size at each 
        iteration  while moving toward a minimum of the loss function.

    max_iter : int
        The number of passes over the training dataset (epochs).

    random_state : int, optional (default=42)
        The seed of the pseudo random number generator for shuffling the 
        data and initializing the weights.

    Attributes
    ----------
    weights_ : array-like, shape (n_features,)
        Weights assigned to the features after fitting the model.

    cost_ : list
        Sum of squares cost function value at each epoch, useful for evaluating 
        the performance improvement over time.

    Examples
    --------
    >>> # Example for regression
    >>> from sklearn.datasets import load_boston
    >>> from gofast.estimators.adaline import AdalineMixte
    >>> boston = load_boston()
    >>> X = boston.data
    >>> y = boston.target
    >>> model = AdalineMixte(eta0=0.0001, max_iter=1000)
    >>> model.fit(X, y)
    >>> y_pred = model.predict(X)
    >>> print('Mean Squared Error:', np.mean((y_pred - y) ** 2))

    >>> # Example for classification
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> X = iris.data
    >>> y = iris.target
    >>> model = AdalineMixte(eta0=0.0001, max_iter=1000)
    >>> model.fit(X, y)
    >>> y_pred = model.predict(X)
    >>> print('Accuracy:', np.mean(y_pred == y))

    Notes
    -----
    Adaline's sensitivity to feature scaling implies that preprocessing steps 
    like normalization or standardization are often necessary for optimal 
    performance. Its linear nature makes it best suited for datasets where 
    the relationship between features and target variable is linear or 
    nearly linear.

    References
    ----------
    [1] Widrow, B., Hoff, M.E., 1960. An Adaptive "Adaline" Neuron Using Chemical 
        "Memristors". Technical Report 1553-2, Stanford Electron Labs, Stanford, CA, 
        October 1960.

    See Also
    --------
    LogisticRegression : Logistic Regression classifier for binary 
       classification tasks.
    LinearRegression : Ordinary least squares Linear Regression.
    SGDClassifier : Linear models (SVM, logistic regression, etc.) fitted 
       by SGD.

    """

    def __init__(self, eta0:float = .01 , max_iter: int = 50 , 
                 random_state:int = 42 ) :
        self.eta0=eta0 
        self.max_iter=max_iter 
        self.random_state=random_state 
        
    def fit(self , X, y ): 
        """ Fit the training data 
        
        Parameters 
        ----------
        X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
            Training set; Denotes data that is observed at training and 
            prediction time, used as independent variables in learning. 
            When a matrix, each sample may be represented by a feature vector, 
            or a vector of precomputed (dis)similarity with each training 
            sample. :code:`X` may also not be a matrix, and may require a 
            feature extractor or a pairwise metric to turn it into one  before 
            learning a model.
        y: array-like, shape (M, ) ``M=m-samples``, 
            train target; Denotes data that may be observed at training time 
            as the dependent variable in learning, but which is unavailable 
            at prediction time, and is usually the target of prediction. 
        
        Returns 
        --------
        self: `Perceptron` instance 
            returns ``self`` for easy method chaining.
        """
        X, y = check_X_y( X, y, 
            estimator = get_estimator_name(self), 
            )
        
        self.task_type = type_of_target(y)
        rgen = np.random.RandomState(self.random_state)
        self.weights_ = rgen.normal(loc=0. , scale =.01 , size = 1 + X.shape[1]
                              )
        self.cost_ =list()    
        for i in range (self.max_iter): 
            net_input = self.net_input (X) 
            output = self.activation (net_input) 
            errors =  ( y -  output ) 
            self.weights_[1:] += self.eta0 * X.T.dot(errors)
            self.weights_[0] += self.eta0 * errors.sum() 
            
            if self.task_type == "continuous":
                cost = (errors**2).sum() / 2.0
            else:
                cost = errors[errors != 0].size
            # cost = (errors **2 ).sum() / 2. 
            self.cost_.append(cost) 
        
        return self 
    
    def net_input (self, X):
        """
        Compute the net input X 
        
        Parameters
        ----------
        X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
            Training set; Denotes data that is observed at training and 
            prediction time, used as independent variables in learning. 
            When a matrix, each sample may be represented by a feature vector, 
            or a vector of precomputed (dis)similarity with each training 
            sample. :code:`X` may also not be a matrix, and may require a 
            feature extractor or a pairwise metric to turn it into one  before 
            learning a model.

        Returns
        -------
        weight net inputs 
        """
        return np.dot (X, self.weights_[1:]) + self.weights_[0] 

    def activation (self, X):
        """
        Compute the linear activation 

        Parameters
        ----------
        X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
            Training set; Denotes data that is observed at training and 
            prediction time, used as independent variables in learning. 
            When a matrix, each sample may be represented by a feature vector, 
            or a vector of precomputed (dis)similarity with each training 
            sample. :code:`X` may also not be a matrix, and may require a 
            feature extractor or a pairwise metric to turn it into one  before 
            learning a model.

        Returns
        -------
        X: activate NDArray 

        """
        return X 
    
    def predict (self, X):
        """
        Predict the  class label after unit step
        
        Parameters
        ----------
        X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
            Training set; Denotes data that is observed at training and 
            prediction time, used as independent variables in learning. 
            When a matrix, each sample may be represented by a feature vector, 
            or a vector of precomputed (dis)similarity with each training 
            sample. :code:`X` may also not be a matrix, and may require a 
            feature extractor or a pairwise metric to turn it into one  before 
            learning a model.

        Returns
        -------
        ypred: predicted class label after the unit step  (1, or -1)
        """
        check_is_fitted (self, 'weights_') 
        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )
        
        if self.task_type == "continuous":
            return self.net_input(X)
        else:
            #return np.where(self.net_input(X) >= 0.0, 1, -1)
            return np.where (self.activation(self.net_input(X))>=0. , 1, -1)
    
    def predict_proba(self, X):
        """
        Predict class probabilities using the AdalineMixte model for 
        classification tasks.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        proba : array-like of shape (n_samples, 2)
            The class probabilities of the input samples.

        Raises
        ------
        ValueError
            If the model is used for regression tasks.
        """
        check_is_fitted(self, 'weights_')
        X = check_array(X, accept_sparse=True)
        if self.task_type != "binary":
            raise ValueError("predict_proba is not supported for regression"
                             " tasks with AdalineMixte.")
        # Apply the linear model and the logistic sigmoid function
        net_input = self.net_input(X)
        proba_positive_class = 1 / (1 + np.exp(-net_input))
        proba_negative_class = 1 - proba_positive_class

        return np.vstack((proba_negative_class, proba_positive_class)).T

    def __repr__(self): 
        """ Represent the output class """
        
        tup = tuple (f"{key}={val}".replace ("'", '') for key, val in 
                     self.get_params().items() )
        
        return self.__class__.__name__ + str(tup).replace("'", "") 
    

