# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations 
import re  
import inspect 
import numpy as np
from collections import defaultdict
from scipy import stats

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import StackingRegressor, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import _name_estimators
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.utils import shuffle
try:from sklearn.utils import type_of_target
except: from .tools.funcutils import type_of_target 

from ._gofastlog import  gofastlog
from .exceptions import  EstimatorError 
from .tools.funcutils import smart_format, is_iterable
from .tools.validator import check_X_y, get_estimator_name, check_array 
from .tools.validator import check_is_fitted
    
_logger = gofastlog().get_gofast_logger(__name__)

__all__=[
   "AdalineClassifier",
   "AdalineMixte",
   "AdalineRegressor",
   "AdalineStochasticRegressor",
   "AdalineStochasticClassifier",
   "BasePerceptron",
   "BenchmarkRegressor", 
   "BenchmarkClassifier", 
   "BoostedRegressionTree",
   "BoostedClassifierTree",
   "DecisionTreeBasedRegressor",
   "DecisionTreeBasedClassifier",
   "GradientDescentClassifier",
   "GradientDescentRegressor",
   "HammersteinWienerClassifier",
   "HammersteinWienerRegressor",
   "HBTEnsembleRegressor",
   "HBTEnsembleClassifier",
   "HWEnsembleClassifier",
   "HWEnsembleRegressor",
   "HybridBoostedTreeClassifier",
   "HybridBoostedTreeRegressor",
   "MajorityVoteClassifier",
   "NeuroFuzzyEnsemble",
   "SimpleAverageRegressor",
   "SimpleAverageClassifier",
   "WeightedAverageRegressor",
   "WeightedAverageClassifier",
 ]


class BenchmarkRegressor(BaseEstimator, RegressorMixin):
    """
    Benchmark Regressor for combining various regression estimators.

    This regressor operates as a stacking model, combining multiple base
    regression estimators to improve prediction accuracy. It benchmarks
    individual regressors and utilizes a meta-regressor for final predictions.
    The BenchmarkRegressor fits various base regressors on the dataset and uses
    their predictions as input for the meta-regressor to make final predictions.

    `BenchmarkRegressor` provides a versatile way to combine different 
    regression estimators using a stacking strategy. It's designed to be 
    flexible, allowing users to experiment with various combinations of base 
    estimators and meta-regressors.
    
    The stacking strategy can be mathematically represented as:
    
    .. math::
        \hat{y} = f_{meta}(\hat{y}_1, \hat{y}_2, ..., \hat{y}_m)

    where \( \hat{y} \) is the final prediction, \( f_{meta} \) represents the
    meta-regressor's prediction function, and \( \hat{y}_i \) are the predictions
    from the \( i \)-th base regressor.
    
    Parameters
    ----------
    base_estimators : list of (str, estimator) tuples
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
    >>> from gofast.estimators import BenchmarkRegressor
    >>> boston = load_boston()
    >>> X, y = boston.data, boston.target
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Define base estimators and meta regressor
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
    - The choice of base estimators and the meta-regressor can significantly affect
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
        self.base_estimators = base_estimators
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
            estimators=self.base_estimators,
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
    """
    Benchmark Classifier for combining various classification estimators.

    This classifier operates as a stacking model, combining multiple base
    classification estimators to improve prediction accuracy. It benchmarks
    individual classifiers and utilizes a meta-classifier for final predictions.
    The BenchmarkClassifier fits various base classifiers on the dataset and uses
    their predictions as input for the meta-classifier to make final predictions.

    `BenchmarkClassifier` provides a flexible way to combine different 
    classification estimators using a stacking strategy. The included 
    `predict_proba` method allows for the estimation of class probabilities, 
    which is crucial in many classification scenarios.
    
    The stacking strategy can be mathematically represented as:
    
    .. math::
        \hat{y} = f_{meta}(C_1(x), C_2(x), \ldots, C_m(x))

    where \( \hat{y} \) is the final prediction, \( f_{meta} \) represents the
    meta-classifier's prediction function, and \( C_i(x) \) are the predictions
    from the \( i \)-th base classifier.

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
    >>> from your_module import BenchmarkClassifier
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
    StackingClassifier : Stacking classifier for combining estimators for
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
    For each training example \( x^{(i)} \) with target \( y^{(i)} \) and
    prediction \( \hat{y}^{(i)} \), the weights are updated as:

    .. math::

        w := w + \eta (y^{(i)} - \hat{y}^{(i)}) x^{(i)}

    where \(\eta\) is the learning rate, \(w\) is the weight vector, and
    \(x^{(i)}\) is the feature vector of the \(i\)-th example.

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
    >>> from gofast.estimators import BasePerceptron

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
                update = self.eta * (target - self.predict(xi))
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
    

class MajorityVoteClassifier (BaseEstimator, ClassifierMixin ): 
    r"""
    A majority vote ensemble classifier.

    This classifier combines different classification algorithms, each associated
    with individual weights for confidence. The aim is to create a stronger
    meta-classifier that balances out individual classifiers' weaknesses on
    specific datasets. The majority vote considers the weighted contribution of
    each classifier in making the final decision.

    Mathematically, the weighted majority vote is expressed as:

    .. math::

        \hat{y} = \arg\max_{i} \sum_{j=1}^{m} weights_j \chi_{A}(C_j(x)=i)

    Here, :math:`weights_j` is the weight associated with the base classifier :math:`C_j`;
    :math:`\hat{y}` is the predicted class label by the ensemble; :math:`A` is
    the set of unique class labels; :math:`\chi_A` is the characteristic
    function or indicator function, returning 1 if the predicted class of
    the j-th classifier matches :math:`i`. For equal weights, the equation 
    simplifies to:

    .. math::

        \hat{y} = \text{mode} \{ C_1(x), C_2(x), \ldots, C_m(x) \}

    Parameters
    ----------
    clfs : array-like, shape (n_classifiers)
        Different classifiers for the ensemble.

    vote : str, {'classlabel', 'probability'}, default='classlabel'
        If 'classlabel', prediction is based on the argmax of class labels.
        If 'probability', prediction is based on the argmax of the sum of
        probabilities. Recommended for calibrated classifiers.

    weights : array-like, shape (n_classifiers,), optional, default=None
        Weights for each classifier. Uniform weights are used if 'weights' 
        is None.

    Attributes
    ----------
    classes_ : array-like, shape (n_classes)
        Unique class labels.

    classifiers_ : list
        List of fitted classifiers.

    Notes
    -----
    - This classifier assumes that all base classifiers are capable of producing
      probability estimates if 'vote' is set to 'probability'.
    - In case of a tie in 'classlabel' voting, the class label is determined
      randomly.

    See Also
    --------
    VotingClassifier : Ensemble voting classifier in scikit-learn.
    BaggingClassifier : A Bagging ensemble classifier.
    RandomForestClassifier : A random forest classifier.
    
    Examples 
    ---------
    >>> from sklearn.linear_model import LogisticRegression 
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from sklearn.pipeline import Pipeline 
    >>> from sklearn.model_selection import cross_val_score, train_test_split
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.impute import SimpleImputer
    >>> from gofast.datasets import fetch_data 
    >>> from gofast.estimators import MajorityVoteClassifier 
    >>> from gofast.tools.funcutils import select_features 
    >>> data = fetch_data('bagoue original').frame
    >>> X0 = data.iloc [:, :-1]; y0 = data ['flow'].values  
    >>> # exclude the categorical value for demonstration 
    >>> # binarize the target y 
    >>> y = np.asarray (list(map (lambda x: 0 if x<=1 else 1, y0))) 
    >>> X = selectfeatures (X0, include ='number')
    >>> X = SimpleImputer().fit_transform (X) 
    >>> X, Xt , y, yt = train_test_split(X, y)
    >>> clf1 = LogisticRegression(penalty ='l2', solver ='lbfgs') 
    >>> clf2= DecisionTreeClassifier(max_depth =1 ) 
    >>> clf3 = KNeighborsClassifier( p =2 , n_neighbors=1) 
    >>> pipe1 = Pipeline ([('sc', StandardScaler()), 
                           ('clf', clf1)])
    >>> pipe3 = Pipeline ([('sc', StandardScaler()), 
                           ('clf', clf3)])
    
    (1) -> Test the each classifier results taking individually 
    
    >>> clf_labels =['Logit', 'DTC', 'KNN']
    >>> # test the results without using the MajorityVoteClassifier
    >>> for clf , label in zip ([pipe1, clf2, pipe3], clf_labels): 
            scores = cross_val_score(clf, X, y , cv=10 , scoring ='roc_auc')
            print("ROC AUC: %.2f (+/- %.2f) [%s]" %(scores.mean(), 
                                                     scores.std(), 
                                                     label))
    ... ROC AUC: 0.91 (+/- 0.05) [Logit]
        ROC AUC: 0.73 (+/- 0.07) [DTC]
        ROC AUC: 0.77 (+/- 0.09) [KNN]
    
    (2) _> Implement the MajorityVoteClassifier
    
    >>> # test the resuls with Majority vote  
    >>> mv_clf = MajorityVoteClassifier(clfs = [pipe1, clf2, pipe3])
    >>> clf_labels += ['Majority voting']
    >>> all_clfs = [pipe1, clf2, pipe3, mv_clf]
    >>> for clf , label in zip (all_clfs, clf_labels): 
            scores = cross_val_score(clf, X, y , cv=10 , scoring ='roc_auc')
            print("ROC AUC: %.2f (+/- %.2f) [%s]" %(scores.mean(), 
                                                     scores.std(), label))
    ... ROC AUC: 0.91 (+/- 0.05) [Logit]
        ROC AUC: 0.73 (+/- 0.07) [DTC]
        ROC AUC: 0.77 (+/- 0.09) [KNN]
        ROC AUC: 0.92 (+/- 0.06) [Majority voting] # give good score & less errors 
    """     
    
    def __init__(self, clfs, weights = None , vote ='classlabel'):
        
        self.clfs=clfs 
        self.weights=weights
        self.vote=vote 
        
        self.classifier_names_={}
  
    def fit(self, X, y):
        """
        Fit classifiers 
        
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
        y: array-like, shape (M, ) ``M=m-samples``
            train target; Denotes data that may be observed at training time 
            as the dependent variable in learning, but which is unavailable 
            at prediction time, and is usually the target of prediction. 
        
        Returns 
        --------
        self: `MajorityVoteClassifier` instance 
            returns ``self`` for easy method chaining.
        """
        X, y = check_X_y(
            X, 
            y, 
            estimator = get_estimator_name(self ), 
            to_frame= True, 
            )
        
        self._check_clfs_vote_and_weights ()
        
        # use label encoder to ensure that class start by 0 
        # which is important for np.argmax call in predict 
        self._labenc = LabelEncoder () 
        self._labenc.fit(y)
        self.classes_ = self._labenc.classes_ 
        
        self.classifiers_ = list()
        for clf in self.clfs: 
            fitted_clf= clone (clf).fit(X, self._labenc.transform(y))
            self.classifiers_.append (fitted_clf ) 
            
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
        check_is_fitted (self, 'classifiers_')
        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )
        if self.vote =='proba': 
            maj_vote = np.argmax (self.predict_proba(X), axis =1 )
        if self.vote =='label': 
            # collect results from clf.predict 
            preds = np.asarray(
                [clf.predict(X) for clf in self.classifiers_ ]).T 
            maj_vote = np.apply_along_axis(
                lambda x : np.argmax( 
                    np.bincount(x , weights = self.weights )), 
                    axis = 1 , 
                    arr= preds 
                    
                    )
            maj_vote = self._labenc.inverse_transform(maj_vote )
        
        return maj_vote 
    
    def predict_proba (self, X): 
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
        check_is_fitted (self, 'classifiers_')
        probas = np.asarray (
            [ clf.predict_proba(X) for clf in self.classifiers_ ])
        avg_proba = np.average (probas , axis = 0 , weights = self.weights ) 
        
        return avg_proba 
    
    def get_params( self , deep = True ): 
        """ Overwrite the get params from `Base` class  and get 
        classifiers parameters from GridSearch . """
        
        if not deep : 
            return super().get_params(deep =False )
        if deep : 
            out = self.classifier_names_.copy() 
            for name, step in self.classifier_names_.items() : 
                for key, value in step.get_params (deep =True).items (): 
                    out['%s__%s'% (name, key)]= value 
        
        return out 
        
    def _check_clfs_vote_and_weights (self): 
        """ assert the existence of classifiers, vote type and the 
         classfifers weigths """
        l = "https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html"
        if self.clfs is None: 
            raise TypeError( "Expect at least one classifiers. ")

        if hasattr(self.clfs , '__class__') and hasattr(
                self.clfs , '__dict__'): 
            self.clfs =[self.clfs ]
      
        s = set ([ (hasattr(o, '__class__') and hasattr(o, '__dict__')) for o 
                  in self.clfs])
        
        if  not list(s)[0] or len(s)!=1:
            raise TypeError(
                "Classifier should be a class object, not {0!r}. Please refer"
                " to Scikit-Convention to write your own estimator <{1!r}>."
                .format('type(self.clfs).__name__', l)
                )
        self.classifier_names_ = {
            k : v for k, v  in _name_estimators(self.clfs)
            }
        
        regex= re.compile(r'(class|label|target)|(proba)')
        v= regex.search(self.vote)
        if v  is None : 
            raise ValueError ("Vote argument must be 'probability' or "
                              "'classlabel', got %r"%self.vote )
        if v is not None: 
            if v.group (1) is not None:  
                self.vote  ='label'
            elif v.group(2) is not None: 
                self.vote  ='proba'
           
        if self.weights and len(self.weights)!= len(self.clfs): 
           raise ValueError(" Number of classifier must be consistent with "
                            " the weights. got {0} and {1} respectively."
                            .format(len(self.clfs), len(self.weights))
                            )
            
class AdalineStochasticRegressor(BaseEstimator, RegressorMixin):
    """
    Adaline Stochastic Gradient Descent Regressor.

    This regressor implements the Adaptive Linear Neuron (Adaline) algorithm 
    using Stochastic Gradient Descent (SGD) for linear regression tasks. It 
    is particularly well-suited for large datasets due to its stochastic 
    nature, updating the model weights incrementally for each training instance.

    In SGD, the cost function is approximated for each instance, rather than
    being computed over the entire dataset. The Adaline algorithm updates weights
    based on a linear activation function and a continuous cost function 
    (Mean Squared Error).

    The weight update rule in SGD for Adaline is given by:

    .. math::

        w := w + \eta (y^{(i)} - \phi(z^{(i)})) x^{(i)}

    where \( w \) is the weight vector, \( \eta \) is the learning rate,
    \( y^{(i)} \) is the true value, \( \phi(z^{(i)}) \) is the predicted value, and
    \( x^{(i)} \) is the feature vector of the \( i \)-th training instance.

    Parameters
    ----------
    eta : float, default=0.0001
        Learning rate (between 0.0 and 1.0).

    n_iter : int, default=10
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
    >>> from gofast.estimators import AdalineStochasticRegressor
    >>> boston = load_boston()
    >>> X, y = boston.data, boston.target
    >>> X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

    # Standardize features
    >>> sc = StandardScaler()
    >>> X_train_std = sc.fit_transform(X_train)
    >>> X_test_std = sc.transform(X_test)

    >>> ada_sgd_reg = AdalineStochasticRegressor(eta=0.0001, n_iter=1000)
    >>> ada_sgd_reg.fit(X_train_std, y_train)
    >>> y_pred = ada_sgd_reg.predict(X_test_std)
    >>> print('Mean Squared Error:', np.mean((y_pred - y_test) ** 2))
    """

    def __init__(
        self, 
        eta=0.0001, 
        n_iter=10, 
        shuffle=True,
        random_state=None 
        ):
        self.eta = eta
        self.n_iter = n_iter
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

        for i in range(self.n_iter):
            if self.shuffle:
                X, y = shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                error = target - self.predict(xi)
                self.weights_[1:] += self.eta * xi * error
                self.weights_[0] += self.eta * error
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
    """
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

    Here, \( \Delta w \) represents the change in weights, \( y^{(i)} \) is the 
    true label, \( \phi(z^{(i)}) \) is the predicted label, and \( x^{(i)} \) is 
    the input feature vector.

    The weights are updated incrementally for each training example:

    .. math::

        w := w + \eta (y^{(i)} - \phi(z^{(i)})) x^{(i)}

    where \( \eta \) is the learning rate.

    Parameters
    ----------
    eta : float, optional (default=0.01)
        The learning rate, determining the step size at each iteration while 
        moving toward a minimum of a loss function. The value must be between 
        0.0 and 1.0.

    n_iter : int, optional (default=10)
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
    >>> from gofast.estimators import AdalineStochasticClassifier
    >>> X = [[0., 0.], [1., 1.]]
    >>> y = [0, 1]
    >>> clf = AdalineStochasticClassifier(eta=0.01, n_iter=10)
    >>> clf.fit(X, y)
    AdalineStochasticClassifier(eta=0.01, n_iter=10)

    See Also
    --------
    AdalineGradientDescent : Gradient Descent variant of Adaline.
    SGDClassifier : Scikit-learn's SGD classifier.

    """

    def __init__(self, eta:float = .01 , n_iter: int = 50 , shuffle=True, 
                 random_state:int = None ) :
        self.eta=eta 
        self.n_iter=n_iter 
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
        for i in range(self.n_iter ): 
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
        self.weights_[1:] += self.eta * X.dot(errors) 
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
        self.inspect 
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
    """
    Adaline Gradient Descent Regressor.

    This regressor is based on the principles of Adaptive Linear Neurons (Adaline),
    employing the gradient descent optimization algorithm for regression tasks.
    The AdalineRegressor fits a linear model to the data by minimizing the sum
    of squared errors between the observed targets in the dataset and the targets
    predicted by the linear approximation.

    The weight update in the gradient descent step is performed using the following
    rule:

    .. math::
        w := w + \eta \sum_{i} (y^{(i)} - \phi(z^{(i)})) x^{(i)}

    Here, \( w \) represents the weights, \( \eta \) is the learning rate, 
    \( y^{(i)} \) is the actual value, \( \phi(z^{(i)}) \) is the predicted value,
    and \( x^{(i)} \) is the feature vector for the ith sample.

    Parameters
    ----------
    eta : float, optional (default=0.01)
        The learning rate, determining the step size at each iteration while
        moving toward a minimum of the loss function. The value should be between
        0.0 and 1.0.

    n_iter : int, optional (default=50)
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
    >>> from gofast.estimators import AdalineRegressor
    >>> diabetes = load_diabetes()
    >>> X = diabetes.data
    >>> y = diabetes.target
    >>> X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)
    >>> ada = AdalineRegressor(eta=0.01, n_iter=50)
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

    def __init__(self, eta=0.01, n_iter=50, random_state =None):
        self.eta = eta
        self.n_iter = n_iter
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

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                error = target - self.predict(xi)
                update = self.eta * error
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
    """
    Adaline Gradient Descent Classifier.

    This classifier is based on the principles of Adaptive Linear Neurons (Adaline),
    employing the gradient descent optimization algorithm for binary classification tasks.
    The AdalineClassifier fits a linear decision boundary to the data by minimizing
    the sum of squared errors between the observed labels in the dataset and the labels
    predicted by the linear approximation.

    The weight update in the gradient descent step is performed using the following
    rule:

    .. math::
        w := w + \eta \sum_{i} (y^{(i)} - \phi(z^{(i)})) x^{(i)}

    Here, \( w \) represents the weights, \( \eta \) is the learning rate, 
    \( y^{(i)} \) is the actual label, \( \phi(z^{(i)}) \) is the predicted label,
    and \( x^{(i)} \) is the feature vector for the ith sample.

    Parameters
    ----------
    eta : float, optional (default=0.01)
        The learning rate, determining the step size at each iteration while
        moving toward a minimum of the loss function. The value should be between
        0.0 and 1.0.

    n_iter : int, optional (default=50)
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
    >>> ada = AdalineClassifier(eta=0.01, n_iter=50)
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

    def __init__(self, eta=0.01, n_iter=50, random_state=None ):
        self.eta = eta
        self.n_iter = n_iter
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

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
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
    """
    Adaline Mixte for Dual Regression and Classification Tasks.

    The ADAptive LInear NEuron (Adaline) Mixte is a foundational model in 
    machine learning, capable of performing both regression and classification.
    This versatility stems from  the Adaline's theoretical underpinnings, 
    initially developed by Bernard Widrow and his doctoral student Ted Hoff 
    in the early 1960s. Adaline builds upon the concepts introduced in Frank 
    Rosenblatt's perceptron, primarily through the use of a continuous, 
    linear activation function for weight adjustment, rather than the step 
    function used in the perceptron.

    Adaline illustrates the key concepts of defining and minimizing continuous
    cost function. This lays the groundwork for understanding more advanced 
    machine learning algorithm for classification, such as Logistic Regression, 
    Support Vector Machines,and Regression models.  

    The key difference between Adaline rule (also know as the WIdrow-Hoff rule) 
    and Rosenblatt's perceptron is that the weights are updated based on linear 
    activation function rather than unit step function like in the perceptron. 
    In Adaline, this linear activation function :math:`\phi(z)` is simply 
    the identify function of the net input so that:
        
    Key Contributions and Innovations:
    - Adaline's primary innovation lies in its cost function, which is 
      continuous and differentiable. This aspect allows the use of more 
      efficient optimization algorithms, particularly gradient descent.
      
    - The Widrow-Hoff learning rule, a fundamental concept in neural network 
      training, was first introduced with Adaline. This rule minimizes the 
      mean squared error between the actual outputs and the predictions, 
      leading to a more stable and convergent learning process than the 
      perceptron's rule.

    Mathematical Formulation:
    The update rule for Adaline's weights (w) can be expressed as:

    .. math::
        w := w + \eta \sum_{i} (y^{(i)} - \phi(w^T x^{(i)})) x^{(i)}

    where:
    - \( \eta \) is the learning rate.
    - \( y^{(i)} \) is the true value or label.
    - \( \phi(w^T x^{(i)}) \) is the predicted value or label, with \( \phi \) 
      being the identity function in the case of Adaline, hence \( \phi(z) = z \).
    - \( x^{(i)} \) is the feature vector of the ith sample.

    Parameters
    ----------
    eta : float
        Learning rate (between 0.0 and 1.0). Determines the step size at each 
        iteration  while moving toward a minimum of the loss function.

    n_iter : int
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
    >>> from gofast.estimators import AdalineMixte
    >>> boston = load_boston()
    >>> X = boston.data
    >>> y = boston.target
    >>> model = AdalineMixte(eta=0.0001, n_iter=1000)
    >>> model.fit(X, y)
    >>> y_pred = model.predict(X)
    >>> print('Mean Squared Error:', np.mean((y_pred - y) ** 2))

    >>> # Example for classification
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> X = iris.data
    >>> y = iris.target
    >>> model = AdalineMixte(eta=0.0001, n_iter=1000)
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

    def __init__(self, eta:float = .01 , n_iter: int = 50 , 
                 random_state:int = 42 ) :
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
        X, y = check_X_y( X, y, 
            estimator = get_estimator_name(self), 
            )
        
        self.task_type = type_of_target(y)
        rgen = np.random.RandomState(self.random_state)
        self.weights_ = rgen.normal(loc=0. , scale =.01 , size = 1 + X.shape[1]
                              )
        self.cost_ =list()    
        for i in range (self.n_iter): 
            net_input = self.net_input (X) 
            output = self.activation (net_input) 
            errors =  ( y -  output ) 
            self.weights_[1:] += self.eta * X.T.dot(errors)
            self.weights_[0] += self.eta * errors.sum() 
            
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
    
class HammersteinWienerClassifier(BaseEstimator, ClassifierMixin):
    """
    Hammerstein-Wiener Classifier for Dynamic Classification Tasks.

    The Hammerstein-Wiener Classifier is designed for modeling and predicting 
    outcomes in dynamic systems where the response depends on past inputs. It 
    is particularly useful in scenarios where the system's behavior exhibits 
    both linear and nonlinear characteristics. This model combines the 
    properties of Hammerstein and Wiener systems to capture complex 
    input-output relationships.

    The Hammerstein model in this classifier consists of a nonlinear input 
    function followed by a linear dynamic block. Conversely, the Wiener model 
    comprises a linear dynamic block followed by a nonlinear output function. 
    These combined models allow for capturing both the linear dynamics and 
    nonlinear transformations in the data.

    Mathematical Representation:
    The Hammerstein-Wiener model for classification is expressed as:

    .. math::
        y(t) = g_2\\left( \\text{classifier}\\left( \\sum_{i=1}^{n} g_1(X_{t-i}) \\right) \\right)

    where:
    - \( g_1 \) is the nonlinear function applied to inputs.
    - \( g_2 \) is the nonlinear function applied to the output, typically a 
      logistic function for classification purposes.
    - The 'classifier' is a linear model such as logistic regression.
    - \( X_{t-i} \) denotes the input features at time step \( t-i \), capturing 
      the memory effect in the dynamic system.

    Usage and Applications:
    This classifier is particularly suited for time-series classification, 
    signal processing, and systems identification where the current output 
    is influenced by past inputs. It's useful in fields like telecommunications,
    control systems, and financial modeling where dynamic behaviors are prevalent.

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
    >>> from gofast.estimators import HammersteinWienerClassifier
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
        
        if self.nonlinearity_out=='sigmoid': 
            self.nonlinearity_out = lambda x: 1 / (1 + np.exp(-x))
        elif self.nonlinearity_out =='softmax': 
            self.nonlinearity_out=lambda x: np.exp(x) / np.sum(np.exp(x), axis=0)
            
        if self.classfifer=='LogisticRegression': 
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
        check_is_fitted (self, 'fitted_') 
        X = check_array(X)
        X_lagged = self._preprocess_data(X)

        y_linear = self.classifier.predict(X_lagged)
        y_pred = self.nonlinearity_out(y_linear)
        return y_pred

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
    """
    Hammerstein-Wiener Estimator for Nonlinear Dynamic System Identification.

    The Hammerstein-Wiener Estimator is a powerful tool for modeling dynamic 
    systems where the output is a nonlinear function of both past inputs and outputs. 
    It integrates a Hammerstein model (nonlinear input, linear dynamics) and a Wiener 
    model (linear dynamics, nonlinear output) to effectively capture the complexities 
    of nonlinear dynamic systems. This model is particularly useful in fields like 
    control systems, signal processing, and physiological data analysis, where systems 
    exhibit both linear and nonlinear behaviors over time.

    Mathematical Formulation:
    The estimator is mathematically represented as:

    .. math::
        y(t) = g_2\\left( \\sum_{i} a_i y(t-i) + \\sum_{j} b_j g_1(u(t-j)) \\right)

    where:
    - \( g_1 \) and \( g_2 \) are nonlinear functions for input and output, respectively.
    - \( a_i \) and \( b_j \) are coefficients of the linear dynamic block.
    - \( u(t) \) and \( y(t) \) denote the system input and output at time \( t \).

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
    >>> from gofast.estimators import HammersteinWienerRegressor
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
        linear_model, 
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
            Returns self.
        """
        X, y = check_X_y( X, y, estimator = get_estimator_name(self ))
        X_lagged = self._preprocess_data(X)
        y = y[self.memory_depth:]

        # Fit the linear model
        self.linear_model.fit(X_lagged, y)
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
        check_is_fitted (self, 'fitted_') 
        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )
        X_lagged = self._preprocess_data(X)

        # Predict using the linear model
        y_linear = self.linear_model.predict(X_lagged)

        # Apply the output nonlinearity
        y_pred = self.nonlinearity_out(y_linear)
        return y_pred

class GradientDescentClassifier(BaseEstimator, ClassifierMixin):
    """
    Gradient Descent Classifier for Binary and Multi-Class Classification.

    This classifier utilizes the gradient descent optimization algorithm to train
    binary classifiers for each class using a One-vs-Rest (OvR) approach for
    multi-class classification. In this method, a separate binary classifier is
    trained for each class, where the particular class is treated as the positive
    class and all other classes are combined into a single negative class.

    Mathematical Formulation:
    The core of gradient descent lies in iteratively updating the model parameters
    to minimize a cost function. The update rule can be expressed as:

    .. math::
        w := w - \eta \nabla J(w)

    Here, \( w \) denotes the weight vector, \( \eta \) is the learning rate,
    and \( \nabla J(w) \) is the gradient of the cost function with respect to
    the weights.

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
    >>> from gofast.estimators import GradientDescentClassifier
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
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors.
        y : array-like, shape = [n_samples]
            Target values (class labels).

        Returns
        -------
        self : object
        """
        X, y = check_X_y( X, y, estimator = get_estimator_name(self ))
        self.classes_ = np.unique(y)
        self.weights_ = np.zeros((self.classes_.size, X.shape[1] + 1))
        self.label_binarizer_ = LabelBinarizer().fit(y)
        Y_bin = self.label_binarizer_.transform(y)

        for i, class_ in enumerate(self.classes_):
            y_bin = Y_bin[:, i]
            for _ in range(self.n_iter):
                if self.shuffle:
                    X, y_bin = shuffle(X, y_bin)
                errors = y_bin - self._predict_proba(X, class_)
                self.weights_[i, 1:] += self.eta * X.T.dot(errors)
                self.weights_[i, 0] += self.eta * errors.sum()

        return self

    def net_input(self, X, class_):
        """Calculate net input for a specific class"""
        w = self.weights_[class_]
        return np.dot(X, w[1:]) + w[0]

    def _predict_proba(self, X, class_):
        """Predict class probabilities for a specific class"""
        return np.where(self.net_input(X, class_) >= 0.0, 1, 0)

    def predict(self, X):
        """Predict class labels for samples in X"""
        check_is_fitted (self, 'weights_') 
        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )
        probas = np.array([self.net_input(X, class_)
                           for class_ in range(len(self.classes_))]).T
        return self.classes_[np.argmax(probas, axis=1)]

class GradientDescentRegressor(BaseEstimator, RegressorMixin):
    """
    Gradient Descent Regressor for Linear Regression.

    This regressor employs the gradient descent optimization algorithm to
    perform linear regression tasks.

    GradientDescentRegressor utilizes a linear combination of features to make
    predictions and minimizes the cost function using gradient descent. During
    training, the fit method updates the model's weights by calculating the
    gradient of the cost function and adjusting the weights accordingly.

    Mathematical Formulation:
    The core of gradient descent for linear regression involves iteratively
    updating the model weights using the following rule:

    .. math::
        w := w + \eta \sum_i (y^{(i)} - \phi(z^{(i)}))x^{(i)}

    Here:
    - \( \eta \) is the learning rate.
    - \( y^{(i)} \) represents the true target value.
    - \( \phi(z^{(i)}) \) is the predicted target value.
    - \( x^{(i)} \) is the feature vector for the \( i \)-th sample.

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
    >>> from gofast.estimators import GradientDescentRegressor
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
        check_is_fitted (self, 'fitted_') 
        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )
        return self.net_input(X)

class SimpleAverageRegressor(BaseEstimator, RegressorMixin):
    """
    Simple Average Ensemble Regressor.

    This ensemble model performs regression tasks by averaging the predictions
    of multiple base regression models. It is a straightforward and effective
    technique to reduce the variance of individual model predictions, improving
    the overall robustness and accuracy of the predictions.

    Mathematical Formulation:
    The predictions of the Simple Average Ensemble are calculated as the
    arithmetic mean of the predictions made by each base estimator:

    .. math::
        y_{\\text{pred}} = \\frac{1}{N} \\sum_{i} y_{\\text{pred}_i}

    where:
    - \( N \) is the number of base estimators.
    - \( y_{\\text{pred}_i} \) is the prediction of the i-th base estimator.

    Parameters
    ----------
    base_estimators : list of objects
        A list of base regression estimators. Each estimator in the list should
        support fit and predict methods. These base estimators will be
        combined through averaging to form the ensemble model.

    Attributes
    ----------
    fitted_ : bool
        True if the model has been fitted.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> from gofast.estimators import SimpleAverageRegressor
    >>> # Create an ensemble with Linear Regression and Decision Tree Regressor
    >>> ensemble = SimpleAverageRegressor([
    ...     LinearRegression(),
    ...     DecisionTreeRegressor()
    ... ])
    >>> X, y = np.random.rand(100, 1), np.random.rand(100)
    >>> ensemble.fit(X, y)
    >>> y_pred = ensemble.predict(X)

    Notes
    -----
    - The Simple Average Ensemble is a basic ensemble technique that can be
      used to combine the predictions of multiple regression models.
    - It is particularly effective when the base estimators have diverse
      strengths and weaknesses, as it can balance out their individual
      performances.
    - This ensemble technique is computationally efficient and easy to
      implement, making it a practical choice for many regression problems.
    - While it provides robustness and improved generalization, it may not
      capture complex relationships between features and target variables as
      advanced ensemble methods like Random Forest or Gradient Boosting.

    See Also
    --------
    VotingRegressor : A similar ensemble method provided by Scikit-Learn.
    Random Forest : An ensemble of decision trees with improved performance.
    Gradient Boosting : An ensemble technique for boosting model performance.
    """

    def __init__(self, base_estimators):
        self.base_estimators = base_estimators

    def fit(self, X, y):
        """
        Fit the Simple Average Ensemble model to the data.

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
        for estimator in self.base_estimators:
            estimator.fit(X, y)
        self.fitted_ = True
        return self

    def predict(self, X):
        """
        Predict using the Simple Average Ensemble model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict for.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted values, averaged across all base estimators.
        """
        check_is_fitted (self, 'fitted_') 

        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )
        predictions = [estimator.predict(X) for estimator in self.base_estimators]
        return np.mean(predictions, axis=0)

class WeightedAverageRegressor(BaseEstimator, RegressorMixin):
    """
    Weighted Average Ensemble Regressor.

    This ensemble model calculates the weighted average of the predictions of
    multiple base regression models. Each base model's prediction is multiplied
    by a weight, allowing for differential contributions of each model to the
    final prediction.
    
    In this `WeightedAverageEnsemble` class, each base estimator's prediction is 
    multiplied by its corresponding weight. The weights parameter should be 
    an array-like object where each weight corresponds to a base estimator 
    in base_estimators. The weights can be determined based on the individual 
    performance of the base models or set according to domain knowledge or 
    other criteria.
     
    The model can effectively balance the contributions of various models, 
    potentially leading to better overall performance than a simple average 
    ensemble, especially when the base models have significantly different 
    levels of accuracy or are suited to different parts of the input space.

    Mathematical Formulation:
    The weighted average predictions of the ensemble are calculated as follows:

    .. math::
        y_{\\text{pred}} = \\sum_{i} (weights_i \\times y_{\\text{pred}_i})

    where:
    - \( weights_i \) is the weight of the i-th base estimator.
    - \( y_{\\text{pred}_i} \) is the prediction of the i-th base estimator.

    Parameters
    ----------
    base_estimators : list of objects
        A list of base regression estimators. Each estimator in the list should
        support fit and predict methods. These base estimators will be
        combined through weighted averaging to form the ensemble model.
    weights : array-like of shape (n_estimators,)
        An array-like object containing the weights for each estimator in the
        `base_estimators` list. The weights determine the importance of each
        base estimator's prediction in the final ensemble prediction.

    Attributes
    ----------
    fitted_ : bool
        True if the model has been fitted.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> from gofast.estimators import WeightedAverageRegressor
    >>> # Create an ensemble with Linear Regression and Decision Tree Regressor
    >>> # Weighted with 0.7 for Linear Regression and 0.3 for Decision Tree
    >>> ensemble = WeightedAverageRegressor([
    ...     LinearRegression(),
    ...     DecisionTreeRegressor()
    ... ], [0.7, 0.3])
    >>> X, y = np.random.rand(100, 1), np.random.rand(100)
    >>> ensemble.fit(X, y)
    >>> y_pred = ensemble.predict(X)

    Notes
    -----
    - The Weighted Average Ensemble is a versatile ensemble technique that
      allows for fine-grained control over the contribution of each base
      estimator to the final prediction.
    - The weights can be determined based on the performance of the base
      models, domain knowledge, or other criteria.
    - It is particularly effective when the base models have significantly
      different levels of accuracy or are suited to different parts of the
      input space.
    - This ensemble technique provides flexibility and can improve overall
      performance compared to simple averaging or individual estimators.

    See Also
    --------
    SimpleAverageRegressor : An ensemble that equally weights base model \
        predictions.
    VotingRegressor : A similar ensemble method provided by Scikit-Learn.
    Random Forest : An ensemble of decision trees with improved performance.
    Gradient Boosting : An ensemble technique for boosting model performance.
    """


    def __init__(self, base_estimators, weights):
        self.base_estimators = base_estimators
        self.weights = weights


    def fit(self, X, y):
        """
        Fit the Weighted Average Ensemble model to the data.

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
        for estimator in self.base_estimators:
            estimator.fit(X, y)
        self.fitted_ = True
        
        return self

    def predict(self, X):
        """
        Predict using the Weighted Average Ensemble model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict for.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Weighted predicted values, averaged across all base estimators.
        """
        check_is_fitted (self, 'fitted_') 

        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )
                
        predictions = np.array([estimator.predict(X) for estimator 
                                in self.base_estimators])
        weighted_predictions = np.average(predictions, axis=0, 
                                          weights=self.weights)
        return weighted_predictions

class HWEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Hammerstein-Wiener Ensemble Classifier.

    This classifier combines the Hammerstein-Wiener model with ensemble learning.

    Parameters
    ----------
    n_estimators : int, default=50
        The number of base classifiers in the ensemble.
    learning_rate : float, default=0.1
        The learning rate for gradient boosting (between 0.0 and 1.0).

    Attributes
    ----------
    base_classifiers_ : list
        List of base classifiers.
    weights_ : list
        Weight of each base classifier.

    Mathematical Formulation
    ------------------------
    The Hammerstein-Wiener Ensemble Classifier combines the Hammerstein-Wiener 
    model with ensemble learning. It uses the following formulas:

    1. Hammerstein-Wiener Model:
       The Hammerstein-Wiener model is a dynamic system model that describes the
       output as a nonlinear function of past inputs and outputs. It consists
       of a Hammerstein model (nonlinear input followed by a linear dynamic block)
       and a Wiener model (linear dynamic block followed by a nonlinear output).
       Mathematically, it is represented as:
       
       \[ y(t) = g_2( \sum a_i * y(t-i) + \sum b_j * g_1(u(t-j)) ) \]

       where:
       - \( y(t) \) is the predicted label.
       - \( g_1 \) and \( g_2 \) are nonlinear functions.
       - \( a_i \) and \( b_j \) are coefficients of the linear dynamic model.
       - \( u(t) \) is the input.

    2. Weight Calculation for Base Classifiers:
       The ensemble assigns weights to each base classifier to determine their
       influence on the final prediction. The weight calculation is given by:
       
       \[ \text{Weight} = \text{learning\_rate} \cdot \frac{1}{1 + \text{Weighted Error}} \]

       where:
       - \( \text{learning\_rate} \) is the learning rate for gradient boosting.
       - Weighted Error represents the error of the base classifier.

    See Also 
    ----------
    gofast.estimators.HammersteinWienerClassifier: 
        Hammerstein-Wiener Classifier for Dynamic Classification Tasks. 
        
    Examples
    --------
    >>> # Import necessary libraries
    >>> import numpy as np
    >>> from sklearn.model_selection import train_test_split
    >>> from gofast.estimators import HWEnsembleClassifier
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


class HWEnsembleRegressor(BaseEstimator, RegressorMixin):
    """
    Hammerstein-Wiener Ensemble (HWE) for regression tasks.

    This ensemble model combines multiple Hammerstein-Wiener models, each of 
    which is a dynamic system model where the output is a nonlinear function 
    of past inputs and outputs. The ensemble averages the predictions of 
    these models.

    HWEnsembleRegressor class assumes that each Hammerstein-Wiener model
    (HammersteinWienerEstimator) is already implemented with its fit and 
    predict methods. The ensemble model fits each Hammerstein-Wiener estimator 
    on the training data and then averages their predictions.

    This approach can potentially improve the performance by capturing different 
    aspects or dynamics of the data with each Hammerstein-Wiener model, and then 
    combining these to form a more robust overall prediction. However, the 
    effectiveness of this ensemble would heavily depend on the diversity and 
    individual accuracy of the included Hammerstein-Wiener models.
    
    Parameters
    ----------
    hweights_estimators : list of HammersteinWienerEstimator objects
        List of Hammerstein-Wiener model estimators. Each estimator in the list
        should be pre-configured with its own hyperparameters, including
        nonlinearities, linear models, and memory depth.

    Attributes
    ----------
    fitted_ : bool
        True if the ensemble model has been fitted.

    Mathematical Formulation
    ------------------------
    The Hammerstein-Wiener Ensemble Regressor combines the predictions of
    multiple Hammerstein-Wiener models to obtain the final regression result.
    The ensemble averages the predicted outputs from each model, which is
    represented mathematically as:

    .. math::
        y_{\\text{pred}} = \\frac{1}{N} \\sum_{i=1}^{N} y_{\\text{pred}_i}

    where:
    - \(N\) is the number of Hammerstein-Wiener models in the ensemble.
    - \(y_{\\text{pred}_i}\) is the prediction of the \(i\)-th Hammerstein-Wiener model.

    See Also 
    ---------
    gofast.estimators.HammersteinWienerRegressor: 
        Hammerstein-Wiener Regressor for Dynamic Regression Tasks. 
        
    Examples
    --------
    >>> # Import necessary libraries
    >>> import numpy as np
    >>> import gofast.estimators import HWEnsembleRegressor
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

class NeuroFuzzyEnsemble(BaseEstimator, RegressorMixin):
    """
    Neuro-Fuzzy Ensemble for regression tasks.

    This ensemble model combines multiple neuro-fuzzy models, which integrate
    neural network learning capabilities with fuzzy logic's qualitative 
    reasoning. The ensemble averages the predictions of these models.

    In this NeuroFuzzyEnsemble class, each NeuroFuzzyModel is assumed to be 
    a pre-defined class that encapsulates a neuro-fuzzy system with fit and 
    predict methods. The ensemble model fits each neuro-fuzzy estimator on 
    the training data and then averages their predictions.
    
    Neuro-fuzzy models are particularly useful for capturing complex and 
    non-linear relationships in data by combining the qualitative reasoning of 
    fuzzy logic with the learning capabilities of neural networks.
    
    Mathematical Formulas
    ---------------------
    The Neuro-Fuzzy Ensemble combines the predictions of multiple neuro-fuzzy models
    and calculates the ensemble prediction as follows:

    1. Neuro-Fuzzy Model Prediction:
       \[ \hat{y}_i = f_{\text{NF}_i}(X) \]

    2. Ensemble Prediction (Averaging):
       \[ \hat{y}_{\text{ensemble}} = \frac{1}{N} \sum_{i=1}^{N} \hat{y}_i \]

    where:
    - \( \hat{y}_i \) is the prediction of the i-th neuro-fuzzy model.
    - \( N \) is the number of neuro-fuzzy models in the ensemble.
    - \( f_{\text{NF}_i} \) represents the prediction function of the i-th 
       neuro-fuzzy model.
       
    The Neuro-Fuzzy Ensemble combines the predictions of multiple neuro-fuzzy
     models and calculates the ensemble prediction as follows:

    1. Neuro-Fuzzy Model Prediction:
       \[ \hat{y}_i = f_{\text{NF}_i}(X) \]

    2. Ensemble Prediction (Averaging):
       \[ \hat{y}_{\text{ensemble}} = \frac{1}{N} \sum_{i=1}^{N} \hat{y}_i \]

    where:
    - \( \hat{y}_i \) is the prediction of the i-th neuro-fuzzy model.
    - \( N \) is the number of neuro-fuzzy models in the ensemble.
    - \( f_{\text{NF}_i} \) represents the prediction function of the i-th 
    neuro-fuzzy model.

 
    Parameters
    ----------
    nf_estimators : list of NeuroFuzzyModel objects
        List of neuro-fuzzy model estimators. Each estimator in the list
        should be pre-configured with its own settings and hyperparameters.

    Attributes
    ----------
    fitted_ : bool
        True if the ensemble model has been fitted.

    Examples
    --------
    >>> # Import necessary libraries
    >>> import numpy as np
    >>> from gofast.estimators import NeuroFuzzyEnsemble
    
    >>> # Define two Neuro-Fuzzy models with different configurations
    >>> nf1 = NeuroFuzzyModel(...)
    >>> nf2 = NeuroFuzzyModel(...)
    
    >>> # Create a Neuro-Fuzzy Ensemble Regressor with the models
    >>> ensemble = NeuroFuzzyEnsemble([nf1, nf2])

    >>> # Generate random data for demonstration
    >>> X, y = np.random.rand(100, 1), np.random.rand(100)
    
    >>> # Fit the ensemble on the data and make predictions
    >>> ensemble.fit(X, y)
    >>> y_pred = ensemble.predict(X)

    Notes
    -----
    - The Neuro-Fuzzy Ensemble model combines the predictions of multiple
      neuro-fuzzy estimators to provide a more accurate and robust regression
      prediction.
    - Each NeuroFuzzyModel in the ensemble should be pre-configured with its
      own settings, architecture, and hyperparameters.
    """

    def __init__(self, nf_estimators):
        self.nf_estimators = nf_estimators

    def fit(self, X, y):
        """
        Fit the Neuro-Fuzzy Ensemble model to the data.

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
        for estimator in self.nf_estimators:
            estimator.fit(X, y)
        self.fitted_ = True
        return self

    def predict(self, X):
        """
        Predict using the Neuro-Fuzzy Ensemble model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict for.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted values, averaged across all Neuro-Fuzzy estimators.
        """
        check_is_fitted(self, 'fitted_')
        
        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )

        predictions = np.array([estimator.predict(X)
                                for estimator in self.nf_estimators])
        return np.mean(predictions, axis=0)

class BoostedRegressionTree(BaseEstimator, RegressorMixin):
    """
    Enhanced Boosted Regression Tree (BRT) for regression tasks.

    The Enhanced Boosted Regression Tree (BRT) is an advanced implementation
    of the Boosted Regression Tree algorithm, designed to provide better
    performance and prevent overfitting. It incorporates various features
    including support for different loss functions, stochastic boosting,
    and tree pruning.

    The BRT algorithm is based on the principle of ensemble learning, where
    multiple decision trees (base learners) are combined to make more accurate
    predictions. The process is mathematical and iterative, focusing on areas
    where previous models have performed poorly.
    
    ``Different Loss Functions``: supports different loss functions 
    ('linear', 'square', 'exponential'). The derivative of the loss function 
    is used to update the residuals.
    
    ``Stochastic Boosting``: The model includes an option for stochastic 
    boosting, controlled by the subsample parameter, which dictates the 
    fraction of samples used for fitting each base learner. This can 
    help in reducing overfitting.
    
    ``Tree Pruning``: While explicit tree pruning isn't detailed here, it can 
    be managed via the max_depth parameter. Additional pruning techniques can 
    be implemented within the DecisionTreeRegressor fitting process.

    Mathematically, the process is represented as follows:

    .. math::
        F_{k}(x) = \\text{Prediction of the ensemble at iteration } k.

        r = y - F_{k}(x) \\text{ (Residual calculation)}

        F_{k+1}(x) = F_{k}(x) + \\text{learning_rate} \\times h_{k+1}(x)

    where:
    - \( F_{k}(x) \) is the prediction of the ensemble at iteration k.
    - \( r \) is the residual.
    - \( F_{k+1}(x) \) is the updated ensemble prediction.
    - \( h_{k+1}(x) \) is the prediction of the new tree added at iteration k+1.
    - 'learning_rate' is a hyperparameter controlling the contribution of each 
       tree.

    This method incrementally builds the ensemble, focusing on areas where 
    previous models have performed poorly.
   
    Parameters
    ----------
    n_estimators : int
        The number of trees in the ensemble.
    learning_rate : float
        The rate at which the boosting algorithm adapts from previous 
        trees' errors.
    max_depth : int
        The maximum depth of each regression tree.
    loss : str
        The loss function to use. Supported values are 'linear', 
        'square', and 'exponential'.
    subsample : float
        The fraction of samples to be used for fitting each individual base 
        learner .If smaller than 1.0, it enables stochastic boosting.

    Attributes
    ----------
    estimators_ : list of DecisionTreeRegressor
        The collection of fitted sub-estimators.

    Examples
    --------
    >>> from gofast.estimators import BoostedRegressionTree
    >>> brt = BoostedRegressionTree(n_estimators=100, learning_rate=0.1, 
                                    max_depth=3, loss='linear', subsample=0.8)
    >>> X, y = np.random.rand(100, 4), np.random.rand(100)
    >>> brt.fit(X, y)
    >>> y_pred = brt.predict(X)
    
    See Also
    --------
    - `sklearn.ensemble.GradientBoostingRegressor`: The scikit-learn library's
      implementation of gradient boosting for regression tasks.
    - `sklearn.tree.DecisionTreeRegressor`: Decision tree regressor used as
      base learners in ensemble methods.
    - `sklearn.metrics.mean_squared_error`: A common metric for evaluating
      regression models.
  
    Notes
    -----
    - The Boosted Regression Tree model is built iteratively, focusing on
      minimizing errors and improving predictions.
    - Different loss functions can be selected, allowing flexibility in
      modeling.
    - Stochastic boosting can help in reducing overfitting by using a fraction
      of samples for fitting each tree.
    - Tree depth can be controlled to avoid overly complex models.

    """

    def __init__(
        self, 
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=3,
        loss='linear', 
        subsample=1.0
        ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.loss = loss
        self.subsample = subsample
        
    def _loss_derivative(self, y, y_pred):
        """
        Compute the derivative of the loss function.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            True values.
        y_pred : array-like of shape (n_samples,)
            Predicted values.

        Returns
        -------
        loss_derivative : array-like of shape (n_samples,)
        """
        if self.loss == 'linear':
            return y - y_pred
        elif self.loss == 'square':
            return (y - y_pred) * 2
        elif self.loss == 'exponential':
            return np.exp(y_pred - y)
        else:
            raise ValueError("Unsupported loss function")

    def fit(self, X, y):
        """
        Fit the Boosted Regression Tree model to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (real numbers).

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y( X, y, estimator = get_estimator_name(self ))
        n_samples = X.shape[0]
        residual = y.copy()
        
        self.estimators_ = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth)

            # Stochastic boosting (sub-sampling)
            sample_size = int(self.subsample * n_samples)
            indices = np.random.choice(n_samples, sample_size, replace=False)
            X_subset, residual_subset = X[indices], residual[indices]

            tree.fit(X_subset, residual_subset)
            prediction = tree.predict(X)

            # Update residuals
            residual -= self.learning_rate * self._loss_derivative(y, prediction)

            self.estimators_.append(tree)

        return self

    def predict(self, X):
        """
        Predict using the Boosted Regression Tree model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted values.
        """
        check_is_fitted(self, 'estimators_')
        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )
        
        y_pred = np.zeros(X.shape[0])

        for tree in self.estimators_:
            y_pred += self.learning_rate * tree.predict(X)

        return y_pred

class DecisionTreeBasedRegressor(BaseEstimator, RegressorMixin):
    """
    Decision Tree-based Regression for regression tasks.

    `DecisionTreeBasedRegressor` is an ensemble model that combines multiple
    Decision Regression Trees. Each tree in the ensemble contributes to the final
    prediction, which is typically the average of the predictions made by
    each tree.

    Mathematical Formulation
    ------------------------
    The ensemble prediction is obtained by averaging the predictions of each
    individual Regression Tree in the ensemble:

    .. math::
        y_{\\text{pred}} = \\frac{1}{N} \\sum_{i} y_{\\text{tree}_i}

    where:
    - \( N \) is the number of trees in the ensemble.
    - \( y_{\\text{pred}} \) is the final predicted value.
    - \( y_{\\text{tree}_i} \) is the prediction of the i-th Regression Tree.

    Parameters
    ----------
    n_estimators : int
        The number of trees in the ensemble.
    max_depth : int
        The maximum depth of each regression tree.
    random_state : int
        Controls the randomness of the estimator.

    Attributes
    ----------
    estimators_ : list of DecisionTreeRegressor
        The collection of fitted sub-estimators.

    Examples
    --------
    >>> from gofast.estimators import DecisionTreeBasedRegressor
    >>> rte = DecisionTreeBasedRegressor(
    ...     n_estimators=100, max_depth=3, random_state=42)
    >>> X, y = np.random.rand(100, 4), np.random.rand(100)
    >>> rte.fit(X, y)
    >>> y_pred = rte.predict(X)

    See Also
    --------
    - sklearn.ensemble.RandomForestRegressor: A popular ensemble method
      based on decision trees for regression tasks.
    - sklearn.tree.DecisionTreeRegressor: Decision tree regressor used as
      base learners in ensemble methods.
    - sklearn.metrics.mean_squared_error: A common metric for evaluating
      regression models.
     - gofast.estimators.BoostedRegressionTree: An enhanced BRT

    Notes
    -----
    - The Regression Tree Ensemble is built by fitting multiple Regression
      Tree models.
    - Each tree is trained on the entire dataset, and their predictions are
      averaged to obtain the final prediction.
    """

    def __init__(self, n_estimators=100, max_depth=3, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        
    def fit(self, X, y):
        """
        Fit the Regression Tree Ensemble model to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (real numbers).

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y( X, y, estimator = get_estimator_name(self ))
        self.estimators_ = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                         random_state=self.random_state)
            tree.fit(X, y)
            self.estimators_.append(tree)

        return self

    def predict(self, X):
        """
        Predict using the Regression Tree Ensemble model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted values.
        """
        check_is_fitted(self, 'estimators_')
        
        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )
        
        predictions = np.array([tree.predict(X) for tree in self.estimators_])
        return np.mean(predictions, axis=0)

class HBTEnsembleRegressor(BaseEstimator, RegressorMixin):
    """
    Hybrid Boosted Tree Ensemble Regressor.

    The Hybrid Boosted Tree Ensemble Regressor is a powerful ensemble learning
    model that combines decision trees with gradient boosting. It is designed
    for regression tasks and can provide accurate predictions for a wide range
    of applications.

    Parameters
    ----------
    n_estimators : int, default=50
        The number of decision trees in the ensemble.
    learning_rate : float, default=0.1
        The learning rate for gradient boosting (between 0.0 and 1.0).
    max_depth : int, default=3
        The maximum depth of individual decision trees.

    Attributes
    ----------
    base_estimators_ : list
        List of DecisionTreeRegressors (base learners).
    weights_ : list
        Weight of each base learner.

    Mathematical Formulation
    ------------------------
    The Hybrid Booster Tree Ensemble Regressor combines decision trees with
    gradient boosting. It uses the following formulas:

    1. Residual Calculation:
       \[ \text{Residual} = y - F_k(x) \]

    2. Weighted Error Calculation:
       \[ \text{Weighted Error} = \sum_{i=1}^{n} (weights_i \cdot \text{Residual}_i)^2 \]

    3. Weight Calculation for Base Learners:
       \[ \text{Weight} = \text{learning\_rate} \cdot \frac{1}{1 + \text{Weighted Error}} \]

    4. Update Predictions:
       \[ F_{k+1}(x) = F_k(x) + \text{Weight} \cdot \text{Residual} \]

    where:
    - \( n \) is the number of samples
    - \( weights_i \) is the weight of each sample
    - \( y \) is the true target values
    - \( F_k(x) \) is the ensemble prediction at iteration \( k \)
    - \(\text{learning\_rate}\) is the learning rate for gradient boosting

    Example
    -------
    Here's an example of how to use the `HBTEnsembleRegressor`
    on the Boston Housing dataset:

    >>> from sklearn.datasets import load_boston
    >>> from sklearn.model_selection import train_test_split
    >>> from gofast.estimators import HBTEnsembleRegressor

    >>> # Load the dataset
    >>> boston = load_boston()
    >>> X = boston.data
    >>> y = boston.target

    >>> # Split the data into training and testing sets
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.3, random_state=0)

    >>> # Create and fit the HBTEnsembleRegressor
    >>> hybrid_regressor = HBTEnsembleRegressor(
    ...     n_estimators=50, learning_rate=0.01, max_depth=3)
    >>> hybrid_regressor.fit(X_train, y_train)

    >>> # Make predictions and evaluate the model
    >>> y_pred = hybrid_regressor.predict(X_test)
    >>> mse = np.mean((y_pred - y_test) ** 2)
    >>> print('Mean Squared Error:', mse)

    Applications and Performance
    ----------------------------
    The Hybrid Boosted Tree Ensemble Regressor is a versatile model suitable
    for various regression tasks, including real estate price prediction,
    financial forecasting, and more. Its performance depends on the quality
    of the data, the choice of hyperparameters, and the number of estimators.

    When tuned correctly, it can achieve high accuracy and robustness, making
    it a valuable tool for predictive modeling.

    See Also
    --------
    - `sklearn.ensemble.GradientBoostingRegressor`: Scikit-learn's Gradient
      Boosting Regressor for comparison.
    - `sklearn.tree.DecisionTreeRegressor`: Decision tree regressor used as
      base learners in ensemble methods.
    - `sklearn.metrics.mean_squared_error`: A common metric for evaluating
      regression models.
    - gofast.estimators.HybridBoostedTreeRegressor: Hybrid Boosted Regression 
      Tree for regression tasks.
    """

    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth


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
        self.base_estimators_ = []
        self.weights_ = []
        F_k = np.zeros(len(y))

        for _ in range(self.n_estimators):
            # Calculate residuals
            residual = y - F_k
            
            # Fit a decision tree on the residuals
            base_estimator = DecisionTreeRegressor(max_depth=self.max_depth)
            base_estimator.fit(X, residual)
            
            # Calculate weighted error
            weighted_error = np.sum((residual - base_estimator.predict(X)) ** 2)
            
            # Calculate weight for this base estimator
            weight = self.learning_rate / (1 + weighted_error)
            
            # Update predictions
            F_k += weight * base_estimator.predict(X)
            
            # Store the base estimator and its weight
            self.base_estimators_.append(base_estimator)
            self.weights_.append(weight)
        
        return self

    def predict(self, X):
        """Return regression predictions.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Samples.

        Returns
        -------
        y_pred : array-like, shape = [n_samples]
            Predicted target values.
        """
        check_is_fitted(self, 'weights_')
        y_pred = np.zeros(X.shape[0])
        for i, base_estimator in enumerate(self.base_estimators_):
            y_pred += self.weights_[i] * base_estimator.predict(X)
        
        return y_pred

class HybridBoostedTreeClassifier(BaseEstimator, ClassifierMixin):
    """
    Hybrid Boosted Tree Classifier.

    The Hybrid Boosted Tree Classifier is a powerful ensemble learning model
    that combines decision trees with gradient boosting for binary
    classification tasks. It can provide accurate classification results
    by creating an ensemble of decision trees with carefully tuned weights.

    Parameters
    ----------
    n_estimators : int, default=50
        The number of decision trees in the ensemble.
    learning_rate : float, default=0.1
        The learning rate for gradient boosting (between 0.0 and 1.0).
    max_depth : int, default=3
        The maximum depth of individual decision trees.

    Attributes
    ----------
    base_estimators_ : list
        List of DecisionTreeClassifiers (base learners).
    weights_ : list
        Weight of each base learner.

    Mathematical Formulation
    ------------------------
    The Hybrid Boosted Tree Classifier combines decision trees with gradient
    boosting. It uses the following formulas:

    1. Weighted Error Calculation:
       \[ \text{Weighted Error} = \sum_{i=1}^{n} (weights_i \cdot (y_i \neq y_{\text{pred}_i})) \]

    2. Weight Calculation for Base Learners:
       \[ \text{Weight} = \text{learning\_rate} \cdot \log\left(\frac{1 - \text{Weighted Error}}{\text{Weighted Error}}\right) \]

    3. Update Sample Weights:
       \[ \text{Sample\_Weights} = \exp(-\text{Weight} \cdot y \cdot y_{\text{pred}}) \]

    where:
    - \( n \) is the number of samples
    - \( weights_i \) is the weight of each sample
    - \( y_i \) is the true label
    - \( y_{\text{pred}_i} \) is the predicted label
    - \(\text{learning\_rate}\) is the learning rate for gradient boosting

    Example
    -------
    Here's an example of how to use the `HybridBoostedTreeClassifier` on the
    Iris dataset for binary classification:

    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> from gofast.estimators import HybridBoostedTreeClassifier

    >>> # Load the Iris dataset
    >>> iris = load_iris()
    >>> X = iris.data[:, :2]
    >>> y = (iris.target != 0) * 1  # Converting to binary classification

    >>> # Split the data into training and testing sets
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.3, random_state=0)

    >>> # Create and fit the HybridBoostedTreeClassifier
    >>> hybrid_boosted_tree = HybridBoostedTreeClassifier(
    ...     n_estimators=50, learning_rate=0.01, max_depth=3)
    >>> hybrid_boosted_tree.fit(X_train, y_train)

    >>> # Make predictions and evaluate the model
    >>> y_pred = hybrid_boosted_tree.predict(X_test)
    >>> accuracy = np.mean(y_pred == y_test)
    >>> print('Accuracy:', accuracy)

    Applications and Performance
    ----------------------------
    The Hybrid Boosted Tree Classifier is suitable for binary classification
    tasks where you want to combine the strengths of decision trees and
    gradient boosting. It can be used in various applications, such as
    fraud detection, spam classification, and more.

    The model's performance depends on the quality of the data, the choice of
    hyperparameters, and the number of estimators. With proper tuning, it can
    achieve high classification accuracy.

    See Also
    --------
    - `sklearn.ensemble.GradientBoostingClassifier`: Scikit-learn's Gradient
      Boosting Classifier for comparison.
    - `sklearn.tree.DecisionTreeClassifier`: Decision tree classifier used as
      base learners in ensemble methods.
    - `sklearn.metrics.accuracy_score`: A common metric for evaluating
      classification models.

    """
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

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
        self.base_estimators_ = []
        self.weights_ = []

        for _ in range(self.n_estimators):
            # Fit a decision tree on the weighted dataset
            base_estimator = DecisionTreeClassifier(max_depth=self.max_depth)
            base_estimator.fit(X, y, sample_weight=self._compute_sample_weights(y))
            
            # Calculate weighted error
            y_pred = base_estimator.predict(X)
            weighted_error = np.sum(self._compute_sample_weights(y) * (y != y_pred))
            
            # Calculate weight for this base estimator
            weight = self.learning_rate * np.log((1 - weighted_error) / weighted_error)
            
            # Update sample weights
            weight = self._update_sample_weights(y, y_pred, weight)
            
            # Store the base estimator and its weight
            self.base_estimators_.append(base_estimator)
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
        check_is_fitted (self, 'base_estimators_')  
        y_pred = np.zeros(X.shape[0])
        for i, base_estimator in enumerate(self.base_estimators_):
            y_pred += self.weights_[i] * base_estimator.predict(X)
        
        return np.sign(y_pred)
    
    def predict_proba(self, X):
        """
        Predict class probabilities using the Hybrid Boosted Tree Classifier 
        model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        proba : array-like of shape (n_samples, 2)
            The class probabilities of the input samples.
        """
        check_is_fitted(self, 'base_estimators_')
        X = check_array(X, accept_sparse=True)
        # Compute weighted sum of predictions from all base estimators
        weighted_predictions = sum(weight * estimator.predict(X) 
                                   for weight, estimator in zip(
                                           self.weights_, self.base_estimators_))

        # Convert to probabilities using the sigmoid function
        proba_positive_class = 1 / (1 + np.exp(-weighted_predictions))
        proba_negative_class = 1 - proba_positive_class

        return np.vstack((proba_negative_class, proba_positive_class)).T

    def _compute_sample_weights(self, y):
        """Compute sample weights."""
        return np.ones_like(y) / len(y)

    def _update_sample_weights(self, y, y_pred, weight):
        """Update sample weights."""
        return np.exp(-weight * y * y_pred)

class HybridBoostedTreeRegressor(BaseEstimator, RegressorMixin):
    """
    Hybrid Boosted Regression Tree (BRT) for regression tasks.

    The Hybrid Boosted Tree Regressor is a powerful ensemble learning model
    that combines multiple Boosted Regression Tree (BRT) models. Each BRT
    model is itself an ensemble created using boosting principles.
    
    This ensemble model combines multiple Boosted Regression Tree models,
    each of which is an ensemble in itself, created using the 
    principles of boosting.

    In `HybridBoostedTreeRegressor` class, the `n_estimators` parameter 
    controls the number of individual Boosted Regression Trees in the ensemble,
    and `brt_params` is a dictionary of parameters to be passed to each Boosted 
    Regression Tree model. The `GradientBoostingRegressor` from scikit-learn 
    is used as the individual BRT model. This class's fit method trains 
    each BRT model on the entire dataset, and the predict method averages 
    their predictions for the final output.

    Parameters
    ----------
    n_estimators : int, default=50
        The number of Boosted Regression Tree models in the ensemble.
    brt_params : dict, default=None
        Parameters to be passed to each Boosted Regression Tree model. If None,
        default values for BRT parameters are used.

    Attributes
    ----------
    brt_ensembles_ : list of GradientBoostingRegressor
        The collection of fitted Boosted Regression Tree ensembles.

    Mathematical Formulation
    ------------------------
    The Hybrid Boosted Regression Tree Ensemble combines multiple BRT models.
    It uses boosting principles for each BRT model:

    1. Calculate Residuals:
       \[ \text{Residuals} = y - F_k(x) \]

    2. Update Predictions:
       \[ F_{k+1}(x) = F_k(x) + \text{learning_rate} \cdot h_k(x) \]

    where:
    - \( F_k(x) \) is the prediction of the ensemble at iteration k.
    - \( y \) is the true target values.
    - \(\text{learning\_rate}\) is the learning rate for gradient boosting.
    - \( h_k(x) \) is the prediction of the new tree added at iteration k.

    Example
    -------
    Here's an example of how to use the `HybridBoostedTreeRegressor`:

    >>> brt_params = {'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.1}
    >>> hybrid_brt = HybridBoostedTreeRegressor(
    ...     n_estimators=10, brt_params=brt_params)
    >>> X, y = np.random.rand(100, 4), np.random.rand(100)
    >>> hybrid_brt.fit(X, y)
    >>> y_pred = hybrid_brt.predict(X)

    Applications and Performance
    ----------------------------
    The Hybrid Boosted Regression Tree Ensemble is suitable for regression tasks
    where combining the strengths of multiple Boosted Regression Trees can
    lead to improved predictive performance. It can be used in various
    applications, such as predicting house prices, stock prices, and more.

    The model's performance depends on the quality of the data and the choice
    of hyperparameters, including the number of estimators and BRT parameters.

    See Also
    --------
    - `sklearn.ensemble.GradientBoostingRegressor`: Scikit-learn's Gradient
      Boosting Regressor for comparison.
    - `sklearn.tree.DecisionTreeRegressor`: Decision tree regressor used as
      base learners in ensemble methods.

    """

    def __init__(self, n_estimators=10, brt_params=None):
        self.n_estimators = n_estimators
        self.brt_params = brt_params or {}

    def fit(self, X, y):
        """
        Fit the Hybrid Boosted Regression Tree Ensemble model to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (real numbers).

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y( X, y, estimator = get_estimator_name(self ))
        self.brt_ensembles_ = []
        for _ in range(self.n_estimators):
            brt = GradientBoostingRegressor(**self.brt_params)
            brt.fit(X, y)
            self.brt_ensembles_.append(brt)

        return self

    def predict(self, X):
        """
        Predict using the Hybrid Boosted Regression Tree Ensemble model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted values.
        """
        check_is_fitted(self, "brt_ensembles_")
        
        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )
        
        predictions = np.array([brt.predict(X) for brt in self.brt_ensembles_])
        return np.mean(predictions, axis=0)

class BoostedClassifierTree(BaseEstimator, ClassifierMixin):
    """
    Boosted Decision Tree Classifier.

    This classifier employs an ensemble boosting method using decision trees. 
    It builds the model by sequentially adding trees, where each tree is trained 
    to correct the errors made by the previous ones. The final model's prediction 
    is the weighted aggregate of all individual trees' predictions, where weights 
    are adjusted by the learning rate.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the ensemble. More trees can lead to better 
        performance but increase computational complexity.

    max_depth : int, default=3
        The maximum depth of each decision tree. Controls the complexity of the 
        model. Deeper trees can capture more complex patterns but may overfit.

    learning_rate : float, default=0.1
        The rate at which the boosting process adapts to the errors of the 
        previous trees. A lower rate requires more trees to achieve similar 
        performance levels but can yield a more robust model.

    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators. Available after fitting.

    Notes
    -----
    The BoostedClassifierTree combines weak learners (decision trees) into a 
    stronger model. The boosting process iteratively adjusts the weights of 
    observations based on the previous trees' errors. The final prediction is 
    made based on a weighted majority vote (or sum) of the weak learners' predictions.

    The boosting procedure is mathematically formulated as:

    .. math::
        F_t(x) = F_{t-1}(x) + \\alpha_t h_t(x)

    where \( F_t(x) \) is the model at iteration \( t \), \( \\alpha_t \) is 
    the learning rate, and \( h_t(x) \) is the weak learner.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from gofast.estimators import BoostedClassifierTree
    >>> X, y = make_classification(n_samples=100, n_features=4)
    >>> clf = BoostedClassifierTree(n_estimators=100, max_depth=3, learning_rate=0.1)
    >>> clf.fit(X, y)
    >>> print(clf.predict(X))

    References
    ----------
    1. Y. Freund, R. E. Schapire, "A Decision-Theoretic Generalization of 
       On-Line Learning and an Application to Boosting", 1995.
    2. J. H. Friedman, "Greedy Function Approximation: A Gradient Boosting Machine", 
       Annals of Statistics, 2001.
    3. T. Hastie, R. Tibshirani, J. Friedman, "The Elements of Statistical Learning",
       Springer, 2009.

    See Also
    --------
    DecisionTreeClassifier : A decision tree classifier.
    AdaBoostClassifier : An adaptive boosting classifier.
    GradientBoostingClassifier : A gradient boosting machine for classification.
    """

    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

    def fit(self, X, y):
        """
        Fit the Boosted Decision Tree Classifier model to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target class labels.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y( X, y, estimator = get_estimator_name(self ))
        residual = y.copy(); self.estimators_ = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X, residual)
            prediction = tree.predict(X)
            # Update residuals
            residual -= self.learning_rate * (prediction - residual)
            self.estimators_.append(tree)

        return self

    def predict(self, X):
        """
        Predict using the Boosted Decision Tree Classifier model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted class labels.
        """
        check_is_fitted(self, 'estimators_')
        
        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )
        
        cumulative_prediction = np.zeros(X.shape[0])

        for tree in self.estimators_:
            cumulative_prediction += self.learning_rate * tree.predict(X)

        # Thresholding to convert to binary classification
        return np.where(cumulative_prediction > 0.5, 1, 0)
    
    def predict_proba(self, X):
        """
        Predict class probabilities using the Boosted Decision Tree 
        Classifier model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        proba : array-like of shape (n_samples, 2)
            The class probabilities of the input samples. Column 0 contains the
            probabilities of the negative class, and column 1 contains 
            the probabilities of the positive class.
            
        Example
        -------
        >>> from sklearn.datasets import make_classification
        >>> from gofast.estimators import BoostedClassifierTree
        >>> # Create a synthetic dataset
        >>> X, y = make_classification(n_samples=100, n_features=4, n_classes=2,
                                       random_state=42)
        >>> clf = BoostedClassifierTree(n_estimators=10, max_depth=3, learning_rate=0.1)
        >>> clf.fit(X, y)
        >>> # Predict probabilities
        >>> print(clf.predict_proba(X))
        """
        check_is_fitted(self, 'estimators_')
        X = check_array(X, accept_sparse=True)
        
        cumulative_prediction = np.zeros(X.shape[0])

        for tree in self.estimators_:
            cumulative_prediction += self.learning_rate * tree.predict(X)
        # Convert cumulative predictions to probabilities
        proba_positive_class = 1 / (1 + np.exp(-cumulative_prediction))  # Sigmoid function
        proba_negative_class = 1 - proba_positive_class

        return np.vstack((proba_negative_class, proba_positive_class)).T

class DecisionTreeBasedClassifier(BaseEstimator, ClassifierMixin):
    """
    Decision Tree -based Ensemble Classifier.

    The Decision Tree-Based Classifier is an ensemble model that combines
    multiple decision trees to make predictions. It aggregates the predictions
    from multiple decision tree classifiers, typically using majority voting
    to determine the final classification.

    TreeBasedClassifier class, the fit method trains multiple 
    DecisionTreeClassifier models on the entire dataset, and the predict 
    method aggregates their predictions using majority voting to determine 
    the final class labels. The `n_estimators` parameter controls the number 
    of trees in the ensemble, and `max_depth` controls the depth of each tree. 
    The `random_state` parameter can be set for reproducibility of results.
    
    Mathematical Formulation
    ------------------------
    The Decision Tree-Based Classifier combines `n` decision tree models, 
    each represented as \(T_i\), where \(i\) varies from 1 to \(n\).
    
    The ensemble uses majority voting to make the final classification.
    
    For each input \(x\), each decision tree \(T_i\) predicts a class label \(C_i\).
    
    The final predicted class label \(C_{\text{final}}\) for \(x\) is 
    determined by a majority vote among the predictions of all decision trees:

    \[
    C_{\text{final}}(x) = \text{argmax}\left(\sum_{i=1}^{n}\delta(C_i(x))\right)
    \]

    where:
    - \(C_{\text{final}}(x)\) is the final predicted class label for input \(x\).
    - \(\delta(C_i(x))\) is a function that returns 1 if \(C_i(x)\) is the
      predicted class label by tree \(T_i\) and 0 otherwise.
    - \(n\) is the number of decision trees in the ensemble.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of decision trees in the ensemble.
    max_depth : int, default=3
        The maximum depth of each decision tree.
    random_state : int or None, default=None
        Controls the randomness of the estimator for reproducibility.

    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    Example
    -------
    Here's an example of how to use the `DecisionTreeBasedClassifier`:

    >>> from gofast.estimators import DecisionTreeBasedClassifier
    >>> rtec = DecisionTreeBasedClassifier(n_estimators=100, max_depth=3,
    ...                                     random_state=42)
    >>> X, y = np.random.rand(100, 4), np.random.randint(0, 2, 100)
    >>> rtec.fit(X, y)
    >>> y_pred = rtec.predict(X)

    Applications
    ------------
    The Decision Tree-Based Classifier is suitable for various classification
    tasks, such as spam detection, sentiment analysis, and medical diagnosis.

    Performance
    -----------
    The model's performance depends on the quality of the data, the number of
    estimators (trees), and the depth of each tree. Hyperparameter tuning may
    be necessary to optimize performance.

    See Also
    --------
    - `sklearn.ensemble.RandomForestClassifier`: Scikit-learn's Random Forest
      Classifier for ensemble-based classification.
    - `sklearn.tree.DecisionTreeClassifier`: Decision tree classifier used as
      base learners in ensemble methods.

    """

    def __init__(self, n_estimators=100, max_depth=3, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state 
        
        
    def fit(self, X, y):
        """
        Fit the Regression Tree Ensemble Classifier model to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target class labels.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y( X, y, estimator = get_estimator_name(self ))
        self.estimators_ = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(max_depth=self.max_depth, 
                                          random_state=self.random_state)
            tree.fit(X, y)
            self.estimators_.append(tree)

        return self

    def predict(self, X):
        """
        Predict using the Regression Tree Ensemble Classifier model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted class labels.
        """
        check_is_fitted(self, 'estimators_')
        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )
        predictions = np.array([tree.predict(X) for tree in self.estimators_])
        # Majority voting
        y_pred = stats.mode(predictions, axis=0).mode[0]
        return y_pred
    
    def predict_proba(self, X):
        """
        Predict class probabilities using the Decision Tree Based Ensemble 
        Classifier model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        proba : array-like of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        check_is_fitted(self, 'estimators_')
        X = check_array(X, accept_sparse=True)

        # Collect probabilities from each estimator
        all_proba = np.array([tree.predict_proba(X) for tree in self.estimators_])
        # Average probabilities across all estimators
        avg_proba = np.mean(all_proba, axis=0)
        return avg_proba
    
class HBTEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Hybrid Boosted Regression Tree Ensemble Classifier.

    This classifier combines multiple Boosted Decision Tree classifiers. 
    Each member of the ensemble is a complete Boosted Decision Tree model, 
    with boosting applied to decision trees.

    In `HBTEnsembleClassifier`, each GradientBoostingClassifier  
    in `gb_ensembles_` represents an individual Boosted Decision Tree model. 
    The fit method trains each Boosted Decision Tree model on the entire 
    dataset, and the predict method uses majority voting among all the 
    models to determine the final class labels. The `gb_params` parameter 
    allows customization of the individual Gradient Boosting models.
    
    Mathematical Formulation
    ------------------------
    The Hybrid Boosted Tree Ensemble Classifier combines multiple Boosted 
    Decision Tree models, each represented as \(T_i\), where \(i\) varies 
    from 1 to \(n\).

    The ensemble uses majority voting to make the final classification.

    For each input \(x\), each Boosted Decision Tree \(T_i\) predicts a 
    class label \(C_i\).

    The final predicted class label \(C_{\text{final}}\) for \(x\) is determined 
    by a majority vote among the predictions of all Boosted Decision Trees:

    \[
    C_{\text{final}}(x) = \text{argmax}\left(\sum_{i=1}^{n}\delta(C_i(x))\right)
    \]

    where:
    - \(C_{\text{final}}(x)\) is the final predicted class label for input \(x\).
    - \(\delta(C_i(x))\) is a function that returns 1 if \(C_i(x)\) is the 
      predicted class label by Boosted Tree \(T_i\) and 0 otherwise.
    - \(n\) is the number of Boosted Decision Trees in the ensemble.
    
    Parameters
    ----------
    n_estimators : int
        The number of Boosted Decision Tree models in the ensemble.
    gb_params : dict
        Parameters to be passed to each GradientBoostingClassifier model.

    Attributes
    ----------
    gb_ensembles_ : list of GradientBoostingClassifier
        The collection of fitted Boosted Decision Tree ensembles.

    Mathematical Formulation
    ------------------------
    The Hybrid Boosted Tree Ensemble Classifier combines multiple Boosted 
    Decision Tree models, each represented as \(T_i\), where \(i\) varies 
    from 1 to \(n\).

    The ensemble uses majority voting to make the final classification.

    For each input \(x\), each Boosted Decision Tree \(T_i\) predicts a 
    class label \(C_i\).

    The final predicted class label \(C_{\text{final}}\) for \(x\) is determined 
    by a majority vote among the predictions of all Boosted Decision Trees:

    \[
    C_{\text{final}}(x) = \text{argmax}\left(\sum_{i=1}^{n}\delta(C_i(x))\right)
    \]

    where:
    - \(C_{\text{final}}(x)\) is the final predicted class label for input \(x\).
    - \(\delta(C_i(x))\) is a function that returns 1 if \(C_i(x)\) is the 
      predicted class label by Boosted Tree \(T_i\) and 0 otherwise.
    - \(n\) is the number of Boosted Decision Trees in the ensemble.

    Examples
    --------
    >>> from gofast.estimators import HBTEnsembleClassifier
    >>> gb_params = {'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.1}
    >>> hybrid_gb = HBTEnsembleClassifier(n_estimators=10,
                                              gb_params=gb_params)
    >>> X, y = np.random.rand(100, 4), np.random.randint(0, 2, 100)
    >>> hybrid_gb.fit(X, y)
    >>> y_pred = hybrid_gb.predict(X)
    """

    def __init__(self, n_estimators=10, gb_params=None):
        self.n_estimators = n_estimators
        self.gb_params = gb_params or {}
        

    def fit(self, X, y):
        """
        Fit the Hybrid Boosted Regression Tree Ensemble Classifier 
        model to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target class labels.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y( X, y, estimator = get_estimator_name(self ))
        self.gb_ensembles_ = []
        for _ in range(self.n_estimators):
            gb = GradientBoostingClassifier(**self.gb_params)
            gb.fit(X, y)
            self.gb_ensembles_.append(gb)

        return self

    def predict(self, X):
        """
        Predict using the Hybrid Boosted Regression Tree Ensemble 
        Classifier model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted class labels.
        """
        check_is_fitted(self, 'gb_ensembles_')
        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )
        predictions = np.array([gb.predict(X) for gb in self.gb_ensembles_])
        # Majority voting for classification
        y_pred = stats.mode(predictions, axis=0).mode[0]
        return y_pred

class WeightedAverageClassifier(BaseEstimator, ClassifierMixin):
    """
    Weighted Average Ensemble Classifier.

    This classifier averages the predictions of multiple base classifiers,
    each weighted by its assigned importance. The ensemble prediction for 
    each class is the weighted average of the predicted probabilities from 
    all classifiers.

    `WeightedAverageClassifier` class, `base_classifiers` are the 
    individual classifiers in the ensemble, and `weights` are their 
    corresponding importance weights. The `fit` method trains each 
    classifier in the ensemble, and the `predict` method calculates the 
    weighted average of the predicted probabilities from all classifiers
    to determine the final class labels.
    
    This implementation assumes that each base classifier can output 
    class probabilities (i.e., has a `predict_proba` method), which is 
    common in many scikit-learn classifiers. The effectiveness of this 
    ensemble classifier depends on the diversity and performance of the 
    base classifiers, as well as the appropriateness of their assigned 
    weights. Fine-tuning these elements is crucial for optimal performance.
    
    Parameters
    ----------
    base_classifiers : list of objects
        List of base classifiers. Each classifier should support `fit`, 
        `predict`, and `predict_proba` methods.
    weights : array-like of shape (n_classifiers,)
        Weights for each classifier in the `base_classifiers` list.

    Attributes
    ----------
    classifiers_ : list of fitted classifiers
        The collection of fitted base classifiers.

    Mathematical Formulation
    ------------------------
    The Weighted Average Ensemble Classifier combines multiple base classifiers
    \(C_i\) each with its assigned weight \(weights_i\). For a given input \(x\), each 
    base classifier \(C_i\) predicts a class label \(y_i\) and class probabilities 
    \(\mathbf{P}_i(x)\).

    The final predicted class label \(C_{\text{final}}(x)\) for input \(x\) 
    is determined by the weighted average of the predicted probabilities 
    from all classifiers:

    \[
    C_{\text{final}}(x) = \text{argmax}\left(\sum_{i=1}^{n} weights_i \cdot \mathbf{P}_i(x)\right)
    \]

    where:
    - \(C_{\text{final}}(x)\) is the final predicted class label for input \(x\).
    - \(n\) is the number of base classifiers.
    - \(weights_i\) is the weight assigned to base classifier \(C_i\).
    - \(\mathbf{P}_i(x)\) is the predicted class probabilities by base
      classifier \(C_i\) for input \(x\).

    Notes
    -----
    - It's important to carefully select and fine-tune the base classifiers 
      and their weights for optimal performance.
    - The `WeightedAverageClassifier` assumes that each base classifier 
    supports `fit`, `predict`, and `predict_proba` methods.

    See Also
    --------
    - sklearn.ensemble.VotingClassifier: Scikit-learn's ensemble classifier 
      that supports various voting strategies.
    - sklearn.ensemble.StackingClassifier: Scikit-learn's stacking ensemble 
      classifier that combines multiple base estimators.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from gofast.estimators import WeightedAverageClassifier
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> ensemble = WeightedAverageClassifier(
    ...     base_classifiers=[LogisticRegression(), DecisionTreeClassifier()],
    ...     weights=[0.7, 0.3]
    ... )
    >>> X, y = np.random.rand(100, 4), np.random.randint(0, 2, 100)
    >>> ensemble.fit(X, y)
    >>> y_pred = ensemble.predict(X)
    """

    def __init__(self, base_classifiers, weights):
        self.base_classifiers = base_classifiers
        self.weights = weights
        
    def fit(self, X, y):
        """
        Fit the Weighted Average Ensemble Classifier model to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target class labels.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y( X, y, estimator = get_estimator_name(self ))
        self.classifiers_ = []
        for clf in self.base_classifiers:
            fitted_clf = clf.fit(X, y)
            self.classifiers_.append(fitted_clf)

        return self

    def predict(self, X):
        """
        Predict using the Weighted Average Ensemble Classifier model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted class labels.
        """
        check_is_fitted(self, 'classifiers_')
        
        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )
        
        weighted_sum = np.zeros(
            (X.shape[0], len(np.unique(self.classifiers_[0].classes_))))

        for clf, weight in zip(self.classifiers_, self.weights):
            probabilities = clf.predict_proba(X)
            weighted_sum += weight * probabilities

        return np.argmax(weighted_sum, axis=1)
    
    def predict_proba(self, X):
        """
        Predict class probabilities using the Weighted Average 
        Ensemble Classifier model.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
    
        Returns
        -------
        y_proba : array-like of shape (n_samples, n_classes)
            The predicted class probabilities.
        """
        # Check if the classifier is fitted
        check_is_fitted(self, 'classifiers_')
    
        # Check input and ensure it's compatible with base classifiers
        X = check_array(X)
    
        # Get predicted probabilities from each base classifier
        probas = [clf.predict_proba(X) for clf in self.classifiers_]
    
        # Calculate the weighted average of predicted probabilities
        weighted_avg_proba = np.average(probas, axis=0, weights=self.weights)
    
        return weighted_avg_proba

class SimpleAverageClassifier(BaseEstimator, ClassifierMixin):
    """
    Simple Average Ensemble Classifier.

    This classifier combines the predictions of multiple base classifiers
    using a simple average of their predicted class probabilities.

    `SimpleAverageClassifier` aggregates predictions from `base_classifiers`, 
    each of which is a separate classifier. The `fit` method trains each 
    classifier in the ensemble, and the `predict` method calculates the 
    simple average of their predicted class probabilities to determine the 
    final class labels.
    
    Parameters
    ----------
    base_classifiers : list of objects
        List of base classifiers. Each classifier should support `fit`,
        `predict`, and `predict_proba` methods.

    Attributes
    ----------
    classifiers_ : list of fitted classifiers
        The collection of fitted base classifiers.

    Mathematical Formulation
    ------------------------
    The Simple Average Ensemble Classifier combines predictions from multiple
    base classifiers, each with its assigned weight. For a given input `x`, 
    each base classifier predicts class probabilities. The final predicted class
    label for input `x` is determined by the simple average of the predicted 
    class probabilities from all classifiers.

    \[
    C_{\text{final}}(x) = \text{argmax}\left(\frac{1}{n}\sum_{i=1}^{n} \mathbf{P}_i(x)\right)
    \]

    where:
    - \(C_{\text{final}}(x)\) is the final predicted class label for input `x`.
    - \(n\) is the number of base classifiers.
    - \(\mathbf{P}_i(x)\) is the predicted class probabilities by base 
    classifier \(C_i\) for input `x`.

    Notes
    -----
    - The `SimpleAverageClassifier` assumes that each base classifier supports
    `fit`, `predict`, and `predict_proba` methods.

    See Also
    --------
    - sklearn.ensemble.VotingClassifier: Scikit-learn's ensemble classifier 
      that supports various voting strategies.
    - sklearn.ensemble.StackingClassifier: Scikit-learn's stacking ensemble 
      classifier that combines multiple base estimators.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from gofast.estimators import SimpleAverageClassifier
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> ensemble = SimpleAverageClassifier(
    ...     base_classifiers=[LogisticRegression(), DecisionTreeClassifier()]
    ... )
    >>> X, y = np.random.rand(100, 4), np.random.randint(0, 2, 100)
    >>> ensemble.fit(X, y)
    >>> y_pred = ensemble.predict(X)
    """

    def __init__(self, base_classifiers):
        self.base_classifiers = base_classifiers
        
    def fit(self, X, y):
        """
        Fit the Simple Average Ensemble Classifier model to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target class labels.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y( X, y, estimator = get_estimator_name(self ))
        self.classifiers_ = []
        for clf in self.base_classifiers:
            fitted_clf = clf.fit(X, y)
            self.classifiers_.append(fitted_clf)

        return self

    def predict(self, X):
        """
        Predict using the Simple Average Ensemble Classifier model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted class labels.
        """
        check_is_fitted(self, 'classifiers_')
        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )
        # Calculate the average predicted class probabilities
        avg_proba = self._average_proba(X)

        # Determine the class with the highest average probability 
        #as the predicted class
        y_pred = np.argmax(avg_proba, axis=1)
        return y_pred
    
    def predict_proba(self, X):
        """
        Predict class probabilities using the Simple Average 
        Ensemble Classifier model.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
    
        Returns
        -------
        y_proba : array-like of shape (n_samples, n_classes)
            The predicted class probabilities.
        """
        self.inspect 
        
        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )
        # Calculate the average predicted class probabilities
        avg_proba = self._average_proba(X)
        return avg_proba
    
    def _average_proba(self, X):
       """
       Calculate the average predicted class probabilities from base classifiers.

       Parameters
       ----------
       X : array-like of shape (n_samples, n_features)
           The input samples.

       Returns
       -------
       avg_proba : array-like of shape (n_samples, n_classes)
           The average predicted class probabilities.
       """
       # Get predicted probabilities from each base classifier
       probas = [clf.predict_proba(X) for clf in self.classifiers_]

       # Calculate the average predicted class probabilities
       avg_proba = np.mean(probas, axis=0)
       return avg_proba

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
  




