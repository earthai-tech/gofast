# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations 
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import StackingRegressor, StackingClassifier

from .._gofastlog import  gofastlog
from ..tools.validator import check_X_y, get_estimator_name
from ..tools.validator import check_is_fitted

_logger = gofastlog().get_gofast_logger(__name__)

__all__=[ 
            "BenchmarkRegressor", "BenchmarkClassifier"   
    ]

class BenchmarkRegressor(BaseEstimator, RegressorMixin):
    r"""
    Benchmark Regressor for combining various regression.estimators.benchmark.

    This regressor operates as a stacking model, combining multiple base
    regression.estimators.benchmark to improve prediction accuracy. It benchmarks
    individual regressors and utilizes a meta-regressor for final predictions.
    The BenchmarkRegressor fits various base regressors on the dataset and uses
    their predictions as input for the meta-regressor to make final predictions.

    `BenchmarkRegressor` provides a versatile way to combine different 
    regression.estimators.benchmark using a stacking strategy. It's designed to be 
    flexible, allowing users to experiment with various combinations of base 
   .estimators.benchmark and meta-regressors.
    
    The stacking strategy can be mathematically represented as:

    .. math::
        \hat{y} = f_{\text{meta}}(\hat{y}_1, \hat{y}_2, ..., \hat{y}_m)

    where :math:`\hat{y}` is the final prediction, :math:`f_{\text{meta}}` 
    represents the meta-regressor's prediction function, and :math:`\hat{y}_i` 
    are the predictions from the \(i\)-th base regressor. Each :math:`\hat{y}_i`
    is obtained by applying the corresponding base regressor to the input 
    features. This method allows for a refined prediction output by leveraging 
    the strengths of each base estimator and effectively combining their 
    individual predictive capabilities.

    Parameters
    ----------
    base_estimators: list of (str, estimator) tuples
        Base regressors to be used in the ensemble.

    meta_regressor : estimator
        The meta-regressor to combine the base regressors' predictions.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy for base regressors.

    Attributes
    ----------
    stacked_model_ : StackingRegressor
        The underlying StackingRegressor model.

    Examples
    --------
    >>> from sklearn.datasets import load_boston
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> from sklearn.neighbors import KNeighborsRegressor
    >>> from gofast.estimators.benchmark import BenchmarkRegressor
    >>> boston = load_boston()
    >>> X, y = boston.data, boston.target
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Define base.estimators.benchmark and meta regressor
    >>> base_estimators = [
            ('lr', LinearRegression()),
            ('dt', DecisionTreeRegressor()),
            ('knn', KNeighborsRegressor())
        ]
    >>> meta_regressor = LinearRegression()

    # Create and fit the BenchmarkRegressor
    >>> benchmark_reg = BenchmarkRegressor(base_estimators=base_estimators,
                                           meta_regressor=meta_regressor)
    >>> benchmark_reg.fit(X_train, y_train)
    >>> y_pred = benchmark_reg.predict(X_test)
    >>> print('R^2 Score:', benchmark_reg.score(X_test, y_test))

    Notes
    -----
    - The choice of base.estimatorsand the meta-regressor can significantly affect
      the performance. It's recommended to experiment with different combinations.
    - The cross-validation strategy for base regressors can be tuned to balance
      between overfitting and underfitting.

    See Also
    --------
    StackingRegressor : Stacking regressor for combining estimators for 
       regression.
    RandomForestRegressor : A random forest regressor.
    GradientBoostingRegressor : Gradient Boosting for regression.
    
    """
    def __init__(self, base_estimators, meta_regressor, cv=None):
        self.base_estimators= base_estimators
        self.meta_regressor = meta_regressor
        self.cv = cv

    def fit(self, X, y):
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
        self: `BenchmarkRegressor` instance 
            returns ``self`` for easy method chaining.
        """
        # Create and fit the stacking model
        self.stacked_model_ = StackingRegressor(
           estimators=self.base.estimators,
            final_estimator=self.meta_regressor,
            cv=self.cv
        )
        self.stacked_model_.fit(X, y)
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
        check_is_fitted (self, 'stacked_model_')
        return self.stacked_model_.predict(X)

    def score(self, X, y):
        """ Compute the score from the `stacked_model_`
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict for.

        Returns
        -------
        y: array-like of shape (n_samples,)
            Predicted values.
        
        """
        check_is_fitted (self, 'stacked_model_')
        return self.stacked_model_.score(X, y)

class BenchmarkClassifier(BaseEstimator, ClassifierMixin):
    r"""
    Benchmark Classifier for combining various classification.estimators.benchmark.

    This classifier operates as a stacking model, combining multiple base
    classification.estimators.benchmark to improve prediction accuracy. It benchmarks
    individual classifiers and utilizes a meta-classifier for final predictions.
    The BenchmarkClassifier fits various base classifiers on the dataset and uses
    their predictions as input for the meta-classifier to make final predictions.

    `BenchmarkClassifier` provides a flexible way to combine different 
    classification.estimators.benchmark using a stacking strategy. The included 
    `predict_proba` method allows for the estimation of class probabilities, 
    which is crucial in many classification scenarios.
    
    The stacking strategy can be mathematically represented as:

    .. math::
        \hat{y} = f_{\text{meta}}(C_1(x), C_2(x), \ldots, C_m(x))

    where :math:`\hat{y}` is the final prediction, :math:`f_{\text{meta}}` 
    represents the meta-classifier's prediction function, and 
    :math:`C_i(x)` are the predictions from the \(i\)-th base classifier. 
    Each :math:`C_i(x)` computes the predicted output for the input \(x\), 
    contributing to the overall decision made by the meta-classifier.

    Parameters
    ----------
    base_classifiers : list of (str, estimator) tuples
        Base classifiers to be used in the ensemble.

    meta_classifier : estimator
        The meta-classifier to combine the base classifiers' predictions.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy for base classifiers.

    Attributes
    ----------
    stacked_model_ : StackingClassifier
        The underlying StackingClassifier model.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from gofast.estimators.benchmark import BenchmarkClassifier
    >>> iris = load_iris()
    >>> X, y = iris.data, iris.target
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Define base classifiers and meta classifier
    >>> base_classifiers = [
            ('lr', LogisticRegression()),
            ('dt', DecisionTreeClassifier()),
            ('knn', KNeighborsClassifier())
        ]
    >>> meta_classifier = LogisticRegression()

    # Create and fit the BenchmarkClassifier
    >>> benchmark_clf = BenchmarkClassifier(base_classifiers=base_classifiers,
                                            meta_classifier=meta_classifier)
    >>> benchmark_clf.fit(X_train, y_train)
    >>> y_pred = benchmark_clf.predict(X_test)
    >>> print('Accuracy:', benchmark_clf.score(X_test, y_test))

    Notes
    -----
    - The choice of base classifiers and the meta-classifier can significantly affect
      the performance. It's recommended to experiment with different combinations.
    - The cross-validation strategy for base classifiers can be tuned to balance
      between overfitting and underfitting.

    See Also
    --------
    StackingClassifier : Stacking classifier for combining.estimators.benchmark for
        classification.
    RandomForestClassifier : A random forest classifier.
    GradientBoostingClassifier : Gradient Boosting for classification.
    """
    def __init__(self, base_classifiers, meta_classifier, cv=None):
        self.base_classifiers = base_classifiers
        self.meta_classifier = meta_classifier
        self.cv = cv

    def fit(self, X, y):
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
        self: `BenchmarkClassifier` instance 
            returns ``self`` for easy method chaining.
        """
        X, y = check_X_y(
            X, 
            y, 
            estimator = get_estimator_name(self ), 
            to_frame= True, 
            )
        # Create and fit the stacking model
        self.stacked_model_ = StackingClassifier(
           estimators=self.base_classifiers,
            final_estimator=self.meta_classifier,
            cv=self.cv
        )
        self.stacked_model_.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict the class label of X 
        
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
        maj_vote:{array_like}, shape (n_examples, )
            Predicted class label array 
        """
        check_is_fitted (self, 'stacked_model_')
        return self.stacked_model_.predict(X)

    def predict_proba(self, X):
        """
        Predict the class probabilities an return average probabilities which 
        is usefull when computing the the receiver operating characteristic 
        area under the curve (ROC AUC ). 
        
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
        avg_proba: {array_like }, shape (n_examples, n_classes) 
            weights average probabilities for each class per example. 
        """
        check_is_fitted (self, 'stacked_model_')
        return self.stacked_model_.predict_proba(X)

    def score(self, X, y):
        """ Compute the score of from the stacked model."""
        return self.stacked_model_.score(X, y)
    