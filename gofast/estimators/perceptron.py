# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations 
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from .._gofastlog import  gofastlog
from ..tools.validator import check_X_y, get_estimator_name, check_array 
from ..tools.validator import check_is_fitted

_logger = gofastlog().get_gofast_logger(__name__)


__all__=[
    "BasePerceptron", "GradientDescentClassifier",
    "GradientDescentRegressor",
    ]

class BasePerceptron(BaseEstimator, ClassifierMixin):
    r"""Perceptron classifier

    This class implements a perceptron classifier following the principles
    of the perceptron learning rule as proposed by Rosenblatt. The perceptron
    is a fundamental algorithm in neural network theory, based on the MCP
    (McCulloch-Pitts) neuron model.

    The perceptron rule is a binary classification algorithm that operates
    on linearly separable data. It iteratively adjusts the weights of the
    features based on the classification errors made in previous iterations.
    The algorithm converges when it finds a hyperplane that completely
    separates the two classes.

    The update rule for the perceptron can be formalized as follows:
    For each training example :math:`x^{(i)}` with target :math:`y^{(i)}` and
    prediction :math:`\hat{y}^{(i)}`, the weights are updated as:

    .. math::
        w := w + \eta (y^{(i)} - \hat{y}^{(i)}) x^{(i)}

    where :math:`\eta` is the learning rate, :math:`w` is the weight vector, and
    :math:`x^{(i)}` is the feature vector of the :math:`i`-th example. The update
    occurs if the prediction :math:`\hat{y}^{(i)}` is incorrect, thereby gradually
    moving the decision boundary towards an optimal position.

    Parameters
    ----------
    eta : float, default=0.01
        The learning rate, a value between 0.0 and 1.0. It controls the
        magnitude of weight updates and hence the speed of convergence.

    n_iter : int, default=50
        The number of passes over the training data (also known as epochs).
        It determines how many times the algorithm iterates through the entire
        dataset.

    random_state : int, default=None
        Seed for the random number generator for weight initialization. A
        consistent random_state ensures reproducibility of results.

    Attributes
    ----------
    weights_ : ndarray of shape (n_features,)
        Weights after fitting the model. Each weight corresponds to a feature.

    errors_ : list of int
        The number of misclassifications (updates) in each epoch. It can be
        used to evaluate the performance of the classifier over iterations.
        
    Notes
    -----
    The perceptron algorithm does not converge if the data is not linearly
    separable. In such cases, the number of iterations (n_iter) controls the
    termination of the algorithm.

    This implementation initializes the weights to zero but can be modified
    to initialize with small random numbers for alternative convergence behavior.

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

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.preprocessing import StandardScaler
    >>> from gofast.estimators.perceptron import BasePerceptron

    # Load data
    >>> X, y = load_iris(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split
    (X, y, test_size=0.3, random_state=42)

    # Standardize features
    >>> sc = StandardScaler()
    >>> X_train_std = sc.fit_transform(X_train)
    >>> X_test_std = sc.transform(X_test)

    # Create and fit the model
    >>> ppn = BasePerceptron(eta=0.01, n_iter=40)
    >>> ppn.fit(X_train_std, y_train)

    # Predict and evaluate
    >>> y_pred = ppn.predict(X_test_std)
    >>> print('Misclassified samples: %d' % (y_test != y_pred).sum())
    """

    def __init__(
        self, 
        eta:float = .01 , 
        n_iter: int = 50 , 
        random_state:int = None 
        ) :
        self.eta=eta 
        self.n_iter=n_iter 
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
        X, y = check_X_y(
            X, 
            y, 
            estimator = get_estimator_name(self ), 
            to_frame= True, 
            )
        
        rgen = np.random.RandomState(self.random_state)
        self.weights_ = rgen.normal(loc=0. , scale =.01 , size = 1 + X.shape[1])
        self.errors_ =list() 
        for _ in range (self.n_iter):
            errors =0 
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi.reshape(1, -1)))
                self.weights_[1:] += update * xi 
                self.weights_[0] += update 
                errors  += int(update !=0.) 
            self.errors_.append(errors)
        
        return self 
    
    def net_input(self, X) :
        """ Compute the net input """
        return np.dot (X, self.weights_[1:]) + self.weights_[0] 

    def predict (self, X): 
        """
       Predict the  class label after unit step
        
        Parameters
        ----------
        X : Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
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
        return np.where (self.net_input(X) >=.0 , 1 , -1 )
    
    def __repr__(self): 
        """ Represent the output class """
        
        tup = tuple (f"{key}={val}".replace ("'", '') for key, val in 
                     self.get_params().items() )
        
        return self.__class__.__name__ + str(tup).replace("'", "") 

class GradientDescentClassifier(BaseEstimator, ClassifierMixin):
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
        w := w - \eta \nabla J(w)

    where:
    - :math:`w` represents the weight vector of the model.
    - :math:`\eta` is the learning rate, which controls the step size during the 
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
    eta : float
        Learning rate, between 0.0 and 1.0. It controls the step size at each
        iteration while moving toward a minimum of the cost function.

    n_iter : int
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
    >>> gd_clf = GradientDescentClassifier(eta=0.01, n_iter=50)
    >>> gd_clf.fit(X, y)
    >>> y_pred = gd_clf.predict(X)
    >>> print('Accuracy:', np.mean(y_pred == y))

    Notes
    -----
    The learning rate (eta) is a critical parameter that affects the convergence
    of the algorithm. A small learning rate can lead to slow convergence, while a
    large learning rate can cause oscillations or divergence in the cost function.
    The number of epochs controls the number of times the algorithm iterates
    over the entire dataset.

    See Also
    --------
    LogisticRegression : Logistic Regression classifier from Scikit-Learn.
    SGDClassifier : Linear classifier with Stochastic Gradient Descent.
    
    """

    def __init__(self, eta=0.01, n_iter=50, shuffle=True):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle

    def fit(self, X, y):
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
            for _ in range(self.n_iter):
                if self.shuffle:
                    X, y_bin = shuffle(X, y_bin)
                errors = y_bin - self._predict_proba(X, i)
                self.weights_[i, 1:] += self.eta * X.T.dot(errors)
                self.weights_[i, 0] += self.eta * errors.sum()
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

class GradientDescentRegressor(BaseEstimator, RegressorMixin):
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
        w := w - \eta \sum_{i} (y^{(i)} - \phi(z^{(i)})) x^{(i)}

    Here:
    - :math:`\eta` is the learning rate, which controls the step size in the 
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
    eta : float, default=0.0001
        Learning rate (between 0.0 and 1.0). Controls the step size for weight
        updates during training.

    n_iter : int, default=1000
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
    >>> gd_reg = GradientDescentRegressor(eta=0.0001, n_iter=1000)
    >>> gd_reg.fit(X_train, y_train)
    >>> y_pred = gd_reg.predict(X_test)
    >>> mse = np.mean((y_pred - y_test) ** 2)
    >>> print('Mean Squared Error:', mse)

    Notes
    -----
    Gradient Descent is a widely used optimization technique for training
    linear regression models. The learning rate (eta) and the number of
    iterations (n_iter) are crucial hyperparameters that impact the training
    process. Careful tuning of these hyperparameters is necessary for
    achieving optimal results.

    See Also
    --------
    LinearRegression : Linear regression from Scikit-Learn.
    """
    def __init__(self, eta=0.0001, n_iter=1000, random_state =None):
        self.eta = eta
        self.n_iter = n_iter
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

        for i in range(self.n_iter):
            errors = y - self.predict(X)
            self.weights_[1:] += self.eta * X.T.dot(errors)
            self.weights_[0] += self.eta * errors.sum()
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









