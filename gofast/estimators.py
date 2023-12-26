# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations 
import re  
import inspect 
import numpy as np
from collections import defaultdict
from scipy import stats

from sklearn.base import BaseEstimator, ClassifierMixin, clone, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.pipeline import _name_estimators
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.utils import shuffle
try:from sklearn.utils import type_of_target
except: from .tools.funcutils import type_of_target 

from ._docstring import DocstringComponents,_core_docs
from ._gofastlog import  gofastlog
from .exceptions import NotFittedError, EstimatorError 
from .tools.funcutils import smart_format, is_iterable
from .tools.validator import check_X_y, get_estimator_name, check_array 
    

__all__=[
    'AdalineClassifier', 
    'AdalineRegressor', 
    'AdalineMixte',
    'AdalineStochasticClassifier',
    'BasePerceptron',
    'BoostedClassifierTree',
    'BoostedRegressionTree',
    'HammersteinWienerRegressor',
    'HammersteinWienerEnsemble',
    'HybridBRTEnsembleClassifier',
    'HybridBRTRegressor',
    'MajorityVoteClassifier',
    'NeuroFuzzyEnsemble',
    'RegressionTreeBasedClassifier',
    'RegressionTreeEnsemble',
    'SimpleAverageClassifier',
    'SimpleAverageRegressor',
    'WeightedAverageClassifier',
    'WeightedAverageRegressor'
	 ]

# +++ add base documentations +++
_base_params = dict ( 
    axis="""
axis: {0 or 'index', 1 or 'columns'}, default 0
    Determine if rows or columns which contain missing values are 
    removed.
    * 0, or 'index' : Drop rows which contain missing values.
    * 1, or 'columns' : Drop columns which contain missing value.
    Changed in version 1.0.0: Pass tuple or list to drop on multiple 
    axes. Only a single axis is allowed.    
    """, 
    columns="""
columns: str or list of str 
    columns to replace which contain the missing data. Can use the axis 
    equals to '1'.
    """, 
    name="""
name: str, :attr:`pandas.Series.name`
    A singluar column name. If :class:`pandas.Series` is given, 'name'  
    denotes the attribute of the :class:`pandas.Series`. Preferably `name`
    must correspond to the label name of the target. 
    """, 
    sample="""
sample: int, Optional, 
    Number of row to visualize or the limit of the number of sample to be 
    able to see the patterns. This is usefull when data is composed of 
    many rows. Skrunked the data to keep some sample for visualization is 
    recommended.  ``None`` plot all the samples ( or examples) in the data     
    """, 
    kind="""
kind: str, Optional 
    type of visualization. Can be ``dendrogramm``, ``mbar`` or ``bar``. 
    ``corr`` plot  for dendrogram , :mod:`msno` bar,  :mod:`plt`
    and :mod:`msno` correlation  visualization respectively: 
        * ``bar`` plot counts the  nonmissing data  using pandas
        *  ``mbar`` use the :mod:`msno` package to count the number 
            of nonmissing data. 
        * dendrogram`` show the clusterings of where the data is missing. 
            leaves that are the same level predict one onother presence 
            (empty of filled). The vertical arms are used to indicate how  
            different cluster are. short arms mean that branch are 
            similar. 
        * ``corr` creates a heat map showing if there are correlations 
            where the data is missing. In this case, it does look like 
            the locations where missing data are corollated.
        * ``None`` is the default vizualisation. It is useful for viewing 
            contiguous area of the missing data which would indicate that 
            the missing data is  not random. The :code:`matrix` function 
            includes a sparkline along the right side. Patterns here would 
            also indicate non-random missing data. It is recommended to limit 
            the number of sample to be able to see the patterns. 
    Any other value will raise an error. 
    """, 
    inplace="""
inplace: bool, default False
    Whether to modify the DataFrame rather than creating a new one.    
    """
 )

_param_docs = DocstringComponents.from_nested_components(
    core=_core_docs["params"],
    base = DocstringComponents(_base_params)
    )
# +++ end base documentations +++

_logger = gofastlog().get_gofast_logger(__name__)


class BasePerceptron (BaseEstimator, ClassifierMixin): 
    r""" Perceptron classifier 
    
    Inspired from Rosenblatt concept of perceptron rules. Indeed, Rosenblatt 
    published the first concept of perceptron learning rule based on the MCP 
    (McCulloth-Pitts) neuron model. With the perceptron rule, Rosenblatt 
    proposed an algorithm thar would automatically learn the optimal weights 
    coefficients that would them be multiplied by the input features in order 
    to make the decision of whether a neuron fires (transmits a signal) or not. 
    In the context of supervised learning and classification, such algirithm 
    could them be used to predict whether a new data points belongs to one 
    class or the other. 
    
    Rosenblatt initial perceptron rule and the perceptron algorithm can be 
    summarized by the following steps: 
        - initialize the weights at 0 or small random numbers. 
        - For each training examples, :math:`x^{(i)}`:
            - Compute the output value :math:`\hat{y}`. 
            - update the weighs. 
    the weights :math:`w` vector can be fromally written as:
        
    .. math:: 
        
        w := w_j + \delta w_j
            
    Parameters 
    -----------
    eta: float, 
        Learning rate between (0. and 1.) 
    n_iter: int , 
        number of iteration passes over the training set 
    random_state: int, default is 42
        random number generator seed for random weight initialization.
        
    Attributes 
    ----------
    w_: Array-like, 
        Weight after fitting 
    errors_: list 
        Number of missclassification (updates ) in each epoch
    
        
    References
    ------------
    .. [1] Rosenblatt F, 1957, The perceptron:A perceiving and Recognizing
        Automaton,Cornell Aeoronautical Laboratory 1957
    .. [2] McCulloch W.S and W. Pitts, 1943. A logical calculus of Idea of 
        Immanent in Nervous Activity, Bulleting of Mathematical Biophysics, 
        5(4): 115-133, 1943.
    
    """
    def __init__(self, eta:float = .01 , n_iter: int = 50 , 
                 random_state:int = None ) :
        super().__init__()
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
        
        self.w_ = rgen.normal(loc=0. , scale =.01 , size = 1 + X.shape[1]
                              )
        self.errors_ =list() 
        for _ in range (self.n_iter):
            errors =0 
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi 
                self.w_[0] += update 
                errors  += int(update !=0.) 
            self.errors_.append(errors)
        
        return self 
    
    def net_input(self, X) :
        """ Compute the net input """
        return np.dot (X, self.w_[1:]) + self.w_[0] 

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
        self.inspect 
        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )
        return np.where (self.net_input(X) >=.0 , 1 , -1 )
    
    @property 
    def inspect (self): 
        """ Inspect object whether is fitted or not"""
        msg = ( "{obj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )
        
        if not hasattr (self, 'w_'): 
            raise NotFittedError(msg.format(
                obj=self)
            )
        return 1 
    
    def __repr__(self): 
        """ Represent the output class """
        
        tup = tuple (f"{key}={val}".replace ("'", '') for key, val in 
                     self.get_params().items() )
        
        return self.__class__.__name__ + str(tup).replace("'", "") 
    

class MajorityVoteClassifier (BaseEstimator, ClassifierMixin ): 
    r"""
    A majority vote Ensemble classifier 
    
    Combine different classification algorithms associate with individual 
    weights for confidence. The goal is to build a stronger meta-classifier 
    that balance out of the individual classifiers weaknes on a particular  
    datasets. In more precise in mathematical terms, the weighs majority 
    vote can be expressed as follow: 
        
    .. math:: 
        
        \hat{y} = arg \max{i} \sum {j=1}^{m} w_j\chi_A (C_j(x)=1)
    
    where :math:`w_j` is a weight associated with a base classifier, :math:`C_j`; 
    :math:`\hat{y}` is the predicted class label of the ensemble. :math:`A` is 
    the set of the unique class label; :math:`\chi_A` is the characteristic 
    function or indicator function which returns 1 if the predicted class of 
    the jth clasifier matches :math:`i(C_j(x)=1)`. For equal weights, the equation 
    is simplified as follow: 
        
    .. math:: 
        
        \hat{y} = mode {{C_1(x), C_2(x), ... , C_m(x)}}
            
    Parameters 
    ------------
    
    clfs: {array_like}, shape (n_classifiers)
        Differents classifier for ensembles 
        
    vote: str , ['classlabel', 'probability'], default is {'classlabel'}
        If 'classlabel' the prediction is based on the argmax of the class 
        label. Otherwise, if 'probability', the argmax of the sum of the 
        probabilities is used to predict the class label. Note it is 
        recommended for calibrated classifiers. 
        
    weights:{array-like}, shape (n_classifiers, ), Optional, default=None 
        If a list of `int` or `float`, values are provided, the classifier 
        are weighted by importance; it uses the uniform weights if 'weights' is
        ``None``.
        
    Attributes 
    ------------
    classes_: array_like, shape (n_classifiers) 
        array of classifiers withencoded classes labels 
    
    classifiers_: list, 
        list of fitted classifiers 
        
    Examples 
    ---------
    >>> from gofast.exlib.sklearn import (
        LogisticRegression,DecisionTreeClassifier ,KNeighborsClassifier, 
         Pipeline , cross_val_score , train_test_split , StandardScaler , 
         SimpleImputer )
    >>> from gofast.datasets import fetch_data 
    >>> from gofast.base import MajorityVoteClassifier 
    >>> from gofast.base import selectfeatures 
    >>> data = fetch_data('bagoue original').get('data=dfy1')
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
    
    @property 
    def inspect (self): 
        """ Inspect object whether is fitted or not"""
        msg = ( "{obj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )
        
        if not hasattr (self, 'classifiers_'): 
            raise NotFittedError(msg.format(
                obj=self)
            )
        return 1 
    
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
        self.inspect 
        
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
        self.inspect 
        probas = np.asarray (
            [ clf.predict_proba(X) for clf in self.classifiers_ ])
        avg_proba = np.average (probas , axis = 0 , weights = self.weights ) 
        
        return avg_proba 
    
    def get_params( self , deep = True ): 
        """ Overwrite the get params from `_Base` class  and get 
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

    This regressor uses the Adaline algorithm with Stochastic Gradient Descent
    for linear regression tasks.

    AdalineStochasticRegressor updates the model weights for each training 
    instance within each epoch. The shuffle parameter, when set to True, 
    ensures that the training data is shuffled before each epoch to prevent 
    cycles. The cost is computed as the average of the squared errors for 
    each instance, providing a measure of the model's performance during 
    training
    
    In Stochastic Gradient Descent, the weights are updated incrementally for
    each training instance.

    Parameters
    ----------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Number of passes over the training dataset (epochs).
    shuffle : bool
        Whether to shuffle training data before each epoch.

    Attributes
    ----------
    w_ : 1d-array
        Weights after fitting.
    cost_ : list
        Average cost per epoch.

    Examples
    --------
    >>> from sklearn.datasets import load_boston
    >>> from sklearn.model_selection import train_test_split
    >>> boston = load_boston()
    >>> X = boston.data
    >>> y = boston.target
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    >>> ada_sgd_reg = AdalineStochasticRegressor(eta=0.0001, n_iter=1000)
    >>> ada_sgd_reg.fit(X_train, y_train)
    >>> y_pred = ada_sgd_reg.predict(X_test)
    >>> print('Mean Squared Error:', np.mean((y_pred - y_test) ** 2))
    """

    def __init__(self, eta=0.0001, n_iter=10, shuffle=True,
                 random_state=None ):
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
        self.w_ = rgen.normal(loc=0. , scale =.01 , size = 1 + X.shape[1])
        # self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            if self.shuffle:
                X, y = shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                error = target - self.predict(xi)
                self.w_[1:] += self.eta * xi * error
                self.w_[0] += self.eta * error
                cost.append(error**2 / 2.0)
            self.cost_.append(np.mean(cost))
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return continuous output"""
        self.inspect 
        
        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )
        return self.net_input(X)

    @property 
    def inspect (self): 
        """ Inspect object whether is fitted or not"""
        msg = ( "{obj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )
        
        if not hasattr (self, 'w_'): 
            raise NotFittedError(msg.format(
                obj=self)
            )
        return 1 
    
class AdalineStochasticClassifier (BaseEstimator, ClassifierMixin) :
    r""" Adaptative Linear Neuron Classifier  with batch  (stochastic) 
    gradient descent 
    
    A stochastic gradient descent is a popular alternative algorithm which is  
    sometimes also called iterative or online gradient descent [1]_. It updates
    the weights based on the sum of accumulated errors over all training 
    examples :math:`x^{(i)}`: 
        
    .. math:: 
        
        \delta w: \sum{i} (y^{(i)} -\phi( z^{(i)}))x^(i)
            
    the weights are updated incremetally for each training examples: 
        
    .. math:: 
        
        \eta(y^{(i)} - \phi(z^{(i)})) x^{(i)}
            
    Parameters 
    -----------
    eta: float, 
        Learning rate between (0. and 1.) 
    n_iter: int, 
        number of iteration passes over the training set 
    suffle: bool, 
        shuffle training data every epoch if True to prevent cycles. 

    random_state: int, default is 42
        random number generator seed for random weight initialization.
        
    Attributes 
    ----------
    w_: Array-like, 
        Weight after fitting 
    cost_: list 
        Sum of squares cost function (updates ) in each epoch
        
    See also 
    ---------
    AdelineGradientDescent: :class:`~gofast.base.AdalineGradientDescent` 
    
    References 
    -----------
    .. [1] Windrow and al., 1960. An Adaptative "Adaline" Neuron Using Chemical
        "Memistors", Technical reports Number, 1553-2,B Windrow and al., 
        standford Electron labs, Standford, CA,October 1960. 
            
    """
    def __init__(self, eta:float = .01 , n_iter: int = 50 , shuffle=True, 
                 random_state:int = None ) :
        super().__init__()
        self.eta=eta 
        self.n_iter=n_iter 
        self.shuffle=shuffle 
        self.random_state=random_state 
        self.w_initialized =False 
        
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
    
    @property 
    def inspect (self): 
        """ Inspect object whether is fitted or not"""
        msg = ( "{obj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )
        
        if not hasattr (self, 'w_'): 
            raise NotFittedError(msg.format(
                obj=self)
            )
        return 1 
    
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
        
        if not self.w_initialized : 
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
        r= self.rgen.permutation(len(y)) 
        return X[r], y[r]
    
    def _init_weights (self, m): 
        """
        Initialize weights with small random numbers 

        Parameters
        ----------
        m : int 
           random number for weights initialization .

        """
        self.rgen =  np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=.0 , scale=.01, size = 1+ m) 
        self.w_initialized = True 
        
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
        self.w_[1:] += self.eta * X.dot(errors) 
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
        return np.dot (X, self.w_[1:]) + self.w_[0] 

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
        self.inspect 
        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )
        return np.where (self.activation(self.net_input(X))>=0. , 1, -1)
    
    def __repr__(self): 
        """ Represent the output class """
        
        tup = tuple (f"{key}={val}".replace ("'", '') for key, val in 
                     self.get_params().items() )
        
        return self.__class__.__name__ + str(tup).replace("'", "") 


class AdalineRegressor(BaseEstimator, RegressorMixin):
    """
    Adaline Gradient Descent regressor.

    This regressor follows the principles of Adaptive Linear Neurons,
    using the gradient descent optimization algorithm for regression tasks.

    In `AdalineRegressor`, the predict method returns a 
    continuous output, and the example at the bottom demonstrates how 
    to use this regressor with the Diabetes dataset. The performance is 
    evaluated using the mean squared error, which is a common metric 
    for regression tasks. Remember, this is a simple implementation, and 
    real-world applications might require additional features or fine-tuning.
    
    The update rule for weights w is given by:
    
    .. math::
        w := w + \eta \sum_i (y^{(i)} - \phi(z^{(i)}))x^{(i)}

    where:
    - \( \eta \) is the learning rate
    - \( y^{(i)} \) is the true value
    - \( \phi(z^{(i)}) \) is the predicted value
    - \( x^{(i)} \) is the feature vector

    Parameters
    ----------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    ----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Sum of squared errors in every epoch.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import train_test_split
    >>> diabetes = load_diabetes()
    >>> X = diabetes.data
    >>> y = diabetes.target
    >>> X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)
    >>> ada = AdalineRegressor(eta=0.01, n_iter=50)
    >>> ada.fit(X_train, y_train)
    >>> y_pred = ada.predict(X_test)
    >>> print('Mean Squared Error:', np.mean((y_pred - y_test) ** 2))
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
        self.w_ = rgen.normal(loc=0. , scale =.01 , size = 1 + X.shape[1]
                              )
        # self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                error = target - self.predict(xi)
                update = self.eta * error
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += error ** 2
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return continuous output"""
        self.inspect 
        
        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            # ensure_2d=False
            )
        return self.net_input(X)
    
    @property 
    def inspect (self): 
        """ Inspect object whether is fitted or not"""
        msg = ( "{obj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )
        
        if not hasattr (self, 'w_'): 
            raise NotFittedError(msg.format(
                obj=self)
            )
        return 1 

class AdalineClassifier(BaseEstimator, ClassifierMixin):
    """
    Adaline Gradient Descent classifier.

    This classifier follows the principles of Adaptive Linear Neurons,
    using the gradient descent optimization algorithm.

    Adaline classifier using gradient descent. The fit method updates 
    the weights based on the training data, and the predict method 
    classifies new samples. The example usage at the bottom demonstrates 
    how to use this classifier with the Iris dataset for a binary 
    classification problem. Remember that this is a simple implementation 
    and might need adjustments or improvements for real-world applications.
    
    The update rule for weights w is given by:
    
    .. math::
        w := w + \eta \sum_i (y^{(i)} - \phi(z^{(i)}))x^{(i)}

    where:
    - \( \eta \) is the learning rate
    - \( y^{(i)} \) is the true label
    - \( \phi(z^{(i)}) \) is the predicted label
    - \( x^{(i)} \) is the feature vector

    Parameters
    ----------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    ----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications in every epoch.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> iris = load_iris()
    >>> X = iris.data[:, :2]
    >>> y = iris.target
    >>> X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)
    >>> ada = AdalineClassifier(eta=0.01, n_iter=50)
    >>> ada.fit(X_train, y_train)
    >>> y_pred = ada.predict(X_test)
    >>> print('Accuracy:', np.mean(y_pred == y_test))
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
        self.w_ = rgen.normal(loc=0. , scale =.01 , size = 1 + X.shape[1]
                              )
        # self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        self.inspect 
        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    @property 
    def inspect (self): 
        """ Inspect object whether is fitted or not"""
        msg = ( "{obj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )
        
        if not hasattr (self, 'w_'): 
            raise NotFittedError(msg.format(
                obj=self)
            )
        return 1 

class AdalineMixte(BaseEstimator, ClassifierMixin, RegressorMixin): 
    r"""Mixe Adaptative Linear Neuron performs dual for 
    regression and classifier tasks. 
    
    ADAptative LInear NEuron (Adaline) was published by Bernard Widrow and 
    his doctoral studentTeed Hoff only a few uears after Rosenblatt's 
    perceptron algorithm. It can be  considered as impovrment of the latter 
    Windrow and al., 1960.
    
    Adaline illustrates the key concepts of defining and minimizing continuous
    cost function. This lays the groundwork for understanding more advanced 
    machine learning algorithm for classification, such as Logistic Regression, 
    Support Vector Machines,and Regression models.  
    
    The key difference between Adaline rule (also know as the WIdrow-Hoff rule) 
    and Rosenblatt's perceptron is that the weights are updated based on linear 
    activation function rather than unit step function like in the perceptron. 
    In Adaline, this linear activation function :math:`\phi(z)` is simply 
    the identifu function of the net input so that:
        
        .. math:: 
            
            \phi (w^Tx)= w^Tx 
    
    while the linear activation function is used for learning the weights. 
    
    Parameters 
    -----------
    eta: float, 
        Learning rate between (0. and 1.) 
    n_iter: int , 
        number of iteration passes over the training set 
    random_state: int, default is 42
        random number generator seed for random weight initialization.
        
    Attributes 
    ----------
    w_: Array-like, 
        Weight after fitting 
    cost_: list 
        Sum of squares cost function (updates ) in each epoch
        
    
    References 
    -----------
    .. [1] Windrow and al., 1960. An Adaptative "Adeline" Neuron Using Chemical
        "Memistors", Technical reports Number, 1553-2,B Windrow and al., 
        standford Electron labs, Standford, CA,October 1960. 
        
    Examples 
    ---------
    >>> # For regression
    >>> from sklearn.datasets import load_boston
    >>> boston = load_boston()
    >>> X = boston.data
    >>> y = boston.target
    >>> model = AdalineMixte(eta=0.0001, n_iter=1000)
    >>> model.fit(X, y)
    >>> y_pred = model.predict(X)
    >>> print('Mean Squared Error:', np.mean((y_pred - y) ** 2))

    >>> # For classification
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> X = iris.data
    >>> y = iris.target
    >>> model = AdalineMixte(eta=0.0001, n_iter=1000)
    >>> model.fit(X, y)
    >>> y_pred = model.predict(X)
    >>> print('Accuracy:', np.mean(y_pred == y))
    """
    def __init__(self, eta:float = .01 , n_iter: int = 50 , 
                 random_state:int = 42 ) :
        super().__init__()
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
            estimator = get_estimator_name(self), 
            )
        
        self.task_type = type_of_target(y)
        
        rgen = np.random.RandomState(self.random_state)
        
        self.w_ = rgen.normal(loc=0. , scale =.01 , size = 1 + X.shape[1]
                              )
        self.cost_ =list()    
        
        for i in range (self.n_iter): 
            net_input = self.net_input (X) 
            output = self.activation (net_input) 
            errors =  ( y -  output ) 
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum() 
            
            if self.task_type == "continuous":
                cost = (errors**2).sum() / 2.0
            else:
                cost = errors[errors != 0].size
            # cost = (errors **2 ).sum() / 2. 
            self.cost_.append(cost) 
        
        return self 
    
    @property 
    def inspect (self): 
        """ Inspect object whether is fitted or not"""
        msg = ( "{obj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )
        
        if not hasattr (self, 'w_'): 
            raise NotFittedError(msg.format(
                obj=self)
            )
        return 1 
    
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
        return np.dot (X, self.w_[1:]) + self.w_[0] 

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
        self.inspect 
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
    
    def __repr__(self): 
        """ Represent the output class """
        
        tup = tuple (f"{key}={val}".replace ("'", '') for key, val in 
                     self.get_params().items() )
        
        return self.__class__.__name__ + str(tup).replace("'", "") 
    

class HammersteinWienerRegressor(BaseEstimator, RegressorMixin):
    """
    Hammerstein-Wiener Estimator for dynamic system identification.

    This estimator models a dynamic system where the output is a nonlinear
    function of past inputs and outputs. It consists of a Hammerstein model
    (nonlinear input followed by a linear dynamic block) and a Wiener model
    (linear dynamic block followed by a nonlinear output).
    
    The Hammerstein-Wiener model is a nonlinear dynamic model that can be 
    mathematically represented as follows:

    .. math::
        y(t) = g_2\\left( \\sum_{i} a_i y(t-i) + \\sum_{j} b_j g_1(u(t-j)) \\right)

    where:
    - \( g_1 \) and \( g_2 \) are the input and output nonlinear functions,
      respectively.
    - \( a_i \) and \( b_j \) are the coefficients of the linear dynamic model.
    - \( u(t) \) and \( y(t) \) are the input and output of the system at 
      time \( t \).

    This model is used in various applications, including control systems 
    and signal processing.

    Parameters
    ----------
    nonlinearity_in : callable, default=hyperbolictangent ``tanh``
        Nonlinear function at the input.
    nonlinearity_out : callable, default=hyperbolictangent ``tanh``
        Nonlinear function at the output.
    linear_model : object
        Linear model for the dynamic block. Should support fit and predict methods.
    memory_depth : int
        The number of past time steps to include in the model.

    Attributes
    ----------
    fitted_ : bool
        True if the model has been fitted.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> hw = HammersteinWienerRegressor(
    ...     nonlinearity_in=np.tanh,
    ...     nonlinearity_out=np.tanh,
    ...     linear_model=LinearRegression(),
    ...     memory_depth=5
    ... )
    >>> X, y = np.random.rand(100, 1), np.random.rand(100)
    >>> hw.fit(X, y)
    >>> y_pred = hw.predict(X)

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
        self.inspect 
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

    @property 
    def inspect (self): 
        """ Inspect object whether is fitted or not"""
        msg = ( "{obj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )
        
        if not hasattr (self, 'fitted_'): 
            raise NotFittedError(msg.format(
                obj=self)
            )
        return 1 

class GradientDescentClassifier(BaseEstimator, ClassifierMixin):
    """
    Gradient Descent Classifier for binary and multi-class 
    classification tasks.

    This classifier uses the One-vs-Rest (OvR) strategy for multi-class 
    classification,training a separate binary classifier for each class.

    GradientDescentClassifier uses the One-vs-Rest strategy to train a binary 
    classifier for each class. The fit method trains these classifiers, and 
    the predict method uses them to predict the class with the highest 
    probability for each sample. 
    
    Parameters
    ----------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Number of passes over the training dataset (epochs).
    shuffle : bool
        Whether to shuffle training data before each epoch to prevent cycles.

    Attributes
    ----------
    w_ : 2d-array
        Weights after fitting. Each row corresponds to a binary classifier.
    classes_ : array
        Class labels.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> iris = load_iris()
    >>> X = iris.data
    >>> y = iris.target
    >>> gd_clf = GradientDescentClassifier(eta=0.01, n_iter=50)
    >>> gd_clf.fit(X, y)
    >>> y_pred = gd_clf.predict(X)
    >>> print('Accuracy:', np.mean(y_pred == y))
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
        self.w_ = np.zeros((self.classes_.size, X.shape[1] + 1))
        self.label_binarizer_ = LabelBinarizer().fit(y)
        Y_bin = self.label_binarizer_.transform(y)

        for i, class_ in enumerate(self.classes_):
            y_bin = Y_bin[:, i]
            for _ in range(self.n_iter):
                if self.shuffle:
                    X, y_bin = shuffle(X, y_bin)
                errors = y_bin - self._predict_proba(X, class_)
                self.w_[i, 1:] += self.eta * X.T.dot(errors)
                self.w_[i, 0] += self.eta * errors.sum()

        return self

    def net_input(self, X, class_):
        """Calculate net input for a specific class"""
        w = self.w_[class_]
        return np.dot(X, w[1:]) + w[0]

    def _predict_proba(self, X, class_):
        """Predict class probabilities for a specific class"""
        return np.where(self.net_input(X, class_) >= 0.0, 1, 0)

    def predict(self, X):
        """Predict class labels for samples in X"""
        self.inspect 
        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )
        probas = np.array([self.net_input(X, class_)
                           for class_ in range(len(self.classes_))]).T
        return self.classes_[np.argmax(probas, axis=1)]

    @property 
    def inspect (self): 
        """ Inspect object whether is fitted or not"""
        msg = ( "{obj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )
        
        if not hasattr (self, 'w_'): 
            raise NotFittedError(msg.format(
                obj=self)
            )
        return 1 

class GradientDescentRegressor(BaseEstimator, RegressorMixin):
    """
    Gradient Descent Regressor.

    This regressor uses the gradient descent optimization algorithm for
    linear regression tasks.

    GradientDescentRegressor class uses a linear combination of features 
    for predictions and minimizes the cost function using gradient descent. 
    The fit method updates the weights by calculating the gradient of the 
    cost function and adjusting the weights accordingly. 
    
    The weight update rule in gradient descent is given by:

    .. math::
        w := w + \eta \sum_i (y^{(i)} - \phi(z^{(i)}))x^{(i)}

    where:
    - \( \eta \) is the learning rate
    - \( y^{(i)} \) is the true value
    - \( \phi(z^{(i)}) \) is the predicted value
    - \( x^{(i)} \) is the feature vector

    Parameters
    ----------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    ----------
    w_ : 1d-array
        Weights after fitting.
    cost_ : list
        Value of the cost function at each epoch.

    Examples
    --------
    >>> from sklearn.datasets import load_boston
    >>> from sklearn.model_selection import train_test_split
    >>> boston = load_boston()
    >>> X = boston.data
    >>> y = boston.target
    >>> X_train, X_test, y_train, y_test = train_test_split(
        X, y,test_size=0.3, random_state=0)
    >>> gd_reg = GradientDescentRegressor(eta=0.0001, n_iter=1000)
    >>> gd_reg.fit(X_train, y_train)
    >>> y_pred = gd_reg.predict(X_test)
    >>> print('Mean Squared Error:', np.mean((y_pred - y_test) ** 2))
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
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            errors = y - self.predict(X)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return continuous output"""
        self.inspect 
        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )
        return self.net_input(X)

    @property 
    def inspect (self): 
        """ Inspect object whether is fitted or not"""
        msg = ( "{obj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )
        
        if not hasattr (self, 'w_'): 
            raise NotFittedError(msg.format(
                obj=self)
            )
        return 1 
 
class SimpleAverageRegressor(BaseEstimator, RegressorMixin):
    """
    Simple Average Ensemble for regression tasks.

    This ensemble model averages the predictions of multiple base 
    regression models.  It is a straightforward approach to reduce 
    the variance of individual model predictions by averaging their results.
    
    `SimpleAverageEnsemble` class accepts a list of base estimators and 
    averages their predictions. Each base estimator must be compatible 
    with scikit-learn, i.e., they should have fit and predict methods. 
    This ensemble model can be particularly effective when combining models 
    of different types (e.g., linear models, tree-based models) as it can 
    balance out their individual strengths and weaknesses.
    
    The predictions of the Simple Average Ensemble are given by the following 
    mathematical formula:

    .. math::
        y_{\\text{pred}} = \\frac{1}{N} \\sum_{i} y_{\\text{pred}_i}

    where:
    - \( N \) is the number of base estimators.
    - \( y_{\\text{pred}_i} \) is the prediction of the i-th base estimator.

    This ensemble method averages the predictions of several base estimators 
    to improve robustness over a single estimator.

    Parameters
    ----------
    base_estimators : list of objects
        List of base regression estimators. Each estimator in the list should
        support fit and predict methods.

    Attributes
    ----------
    fitted_ : bool
        True if the model has been fitted.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> ensemble = SimpleAverageRegressor([
    ...     LinearRegression(),
    ...     DecisionTreeRegressor()
    ... ])
    >>> X, y = np.random.rand(100, 1), np.random.rand(100)
    >>> ensemble.fit(X, y)
    >>> y_pred = ensemble.predict(X)

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
        self.inspect 

        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )
        predictions = [estimator.predict(X) for estimator in self.base_estimators]
        return np.mean(predictions, axis=0)

    @property 
    def inspect (self): 
        """ Inspect object whether is fitted or not"""
        msg = ( "{obj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )
        
        if not hasattr (self, 'fitted_'): 
            raise NotFittedError(msg.format(
                obj=self)
            )
        return 1 

class WeightedAverageRegressor(BaseEstimator, RegressorMixin):
    """
    Weighted Average Ensemble for regression tasks.

    This ensemble model calculates the weighted average of the predictions of
    multiple base regression models. Each model's prediction is multiplied by 
    a weight, allowing for differential contributions of each model to the 
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
     
    The weighted average predictions of the ensemble are given by the following
    mathematical formula:
    
    .. math::
        y_{\\text{pred}} = \\sum_{i} (w_i \\times y_{\\text{pred}_i})
    
    where:
    - \( w_i \) is the weight of the i-th base estimator.
    - \( y_{\\text{pred}_i} \) is the prediction of the i-th base estimator.
    
    This ensemble method combines the predictions of several base estimators with assigned
    weights to improve robustness and performance over single estimators or simple averages.

    Parameters
    ----------
    base_estimators : list of objects
        List of base regression estimators. Each estimator in the list should
        support fit and predict methods.
    weights : array-like of shape (n_estimators,)
        Weights for each estimator in the base_estimators list.

    Attributes
    ----------
    fitted_ : bool
        True if the model has been fitted.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> ensemble = WeightedAverageRegressor([
    ...     LinearRegression(),
    ...     DecisionTreeRegressor()
    ... ], [0.7, 0.3])
    >>> X, y = np.random.rand(100, 1), np.random.rand(100)
    >>> ensemble.fit(X, y)
    >>> y_pred = ensemble.predict(X)

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
        self.inspect 

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
    
    @property 
    def inspect (self): 
        """ Inspect object whether is fitted or not"""
        msg = ( "{obj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )
        
        if not hasattr (self, 'fitted_'): 
            raise NotFittedError(msg.format(
                obj=self)
            )
        return 1 

class HammersteinWienerEnsemble(BaseEstimator, RegressorMixin):
    """
    Hammerstein-Wiener Ensemble (HWE) for regression tasks.

    This ensemble model combines multiple Hammerstein-Wiener models, each of 
    which is a dynamic system model where the output is a nonlinear function 
    of past inputs and outputs. The ensemble averages the predictions of 
    these models.

    HammersteinWienerEnsemble class assumes that each Hammerstein-Wiener model
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
    hw_estimators : list of HammersteinWienerEstimator objects
        List of Hammerstein-Wiener model estimators.

    Attributes
    ----------
    fitted_ : bool
        True if the ensemble model has been fitted.

    Examples
    --------
    >>> hw1 = HammersteinWienerRegressor(nonlinearity_in=np.tanh, 
                                         nonlinearity_out=np.tanh, 
                                         linear_model=LinearRegression(),
                                         memory_depth=5)
    >>> hw2 = HammersteinWienerRegressor(nonlinearity_in=np.sin, 
                                         nonlinearity_out=np.sin, 
                                         linear_model=LinearRegression(),
                                         memory_depth=5)
    >>> ensemble = HammersteinWienerEnsemble([hw1, hw2])
    >>> X, y = np.random.rand(100, 1), np.random.rand(100)
    >>> ensemble.fit(X, y)
    >>> y_pred = ensemble.predict(X)

    Notes
    -----
    The Hammerstein-Wiener Ensemble model averages the predictions of multiple
    Hammerstein-Wiener estimators to obtain a final prediction.
    """

    def __init__(self, hw_estimators):
        self.hw_estimators = hw_estimators

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
        self.hw_estimators = is_iterable(
            self.hw_estimators, exclude_string=True, transform =True) 
        estimator_names = [ get_estimator_name(estimator) for estimator in 
                           self.hw_estimators ]
        if list( set (estimator_names)) [0] !="HammersteinWiener": 
            raise EstimatorError("Expect `HammersteinWiener` estimators."
                                 f" Got {smart_format(estimator_names)}")
            
        for estimator in self.hw_estimators:
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
        self.inspect 

        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )
                
        predictions = np.array([estimator.predict(X) for estimator in self.hw_estimators])
        return np.mean(predictions, axis=0)

    @property 
    def inspect (self): 
        """ Inspect object whether is fitted or not"""
        msg = ( "{obj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )
        
        if not hasattr (self, 'fitted_'): 
            raise NotFittedError(msg.format(
                obj=self)
            )
        return 1 
    
class NeuroFuzzyEnsemble(BaseEstimator, RegressorMixin):
    """
    Neuro-Fuzzy Ensemble for regression tasks.

    This ensemble model combines multiple neuro-fuzzy models, which integrate
    neural network learning capabilities with fuzzy logic's qualitative reasoning.
    The ensemble averages the predictions of these models.

    In this NeuroFuzzyEnsemble class, each NeuroFuzzyModel is assumed to be 
    a pre-defined class that encapsulates a neuro-fuzzy system with fit and 
    predict methods. The ensemble model fits each neuro-fuzzy estimator on 
    the training data and then averages their predictions.
    
    Parameters
    ----------
    nf_estimators : list of NeuroFuzzyModel objects
        List of neuro-fuzzy model estimators.

    Attributes
    ----------
    fitted_ : bool
        True if the ensemble model has been fitted.

    Examples
    --------
    >>> nf1 = NeuroFuzzyModel(...)
    >>> nf2 = NeuroFuzzyModel(...)
    >>> ensemble = NeuroFuzzyEnsemble([nf1, nf2])
    >>> X, y = np.random.rand(100, 1), np.random.rand(100)
    >>> ensemble.fit(X, y)
    >>> y_pred = ensemble.predict(X)

    Notes
    -----
    The Neuro-Fuzzy Ensemble model averages the predictions of multiple
    neuro-fuzzy estimators to obtain a final prediction. Neuro-fuzzy models
    are expected to capture complex relationships in data by combining the
    fuzzy qualitative approach with the learning ability of neural networks.
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
        self.inspect 
        
        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )

        predictions = np.array([estimator.predict(X)
                                for estimator in self.nf_estimators])
        return np.mean(predictions, axis=0)
    
    @property 
    def inspect (self): 
        """ Inspect object whether is fitted or not"""
        msg = ( "{obj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )
        
        if not hasattr (self, 'fitted_'): 
            raise NotFittedError(msg.format(obj=self))
        return 1 

class BoostedRegressionTree(BaseEstimator, RegressorMixin):
    """
    Enhanced Boosted Regression Tree (BRT) for regression tasks.

    This model is an advanced implementation of the Boosted Regression 
    Tree algorithm, featuring tree pruning, handling different loss functions,
    and incorporating strategies to prevent overfitting.
    
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
    >>> brt = BoostedRegressionTree(n_estimators=100, learning_rate=0.1, 
                                    max_depth=3, loss='linear', subsample=0.8)
    >>> X, y = np.random.rand(100, 4), np.random.rand(100)
    >>> brt.fit(X, y)
    >>> y_pred = brt.predict(X)

    Notes
    -----
    The Boosted Regression Tree model is built iteratively. It includes 
    advanced features such as different loss functions and stochastic boosting
    to improve model performance and reduce the risk of overfitting.

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
        self.inspect 
        
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


    @property 
    def inspect (self): 
        """ Inspect object whether is fitted or not"""
        msg = ( "{obj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )
        
        if not hasattr (self, 'estimators_'): 
            raise NotFittedError(msg.format(
                obj=self)
            )
        return 1 
    
class RegressionTreeEnsemble(BaseEstimator, RegressorMixin):
    """
    Regression Tree Ensemble for regression tasks.

    This ensemble model combines multiple Regression Trees. Each 
    tree in the ensemble contributes to the final prediction, which is 
    typically the average of the predictions made by each tree.

    RegressionTreeEnsemble class fits multiple DecisionTreeRegressor 
    models on the entire dataset and averages their predictions for the 
    final output. The n_estimators parameter controls the number of trees 
    in the ensemble, and max_depth controls the depth of each tree. 
    The random_state can be set for reproducibility.
    
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
    >>> rte = RegressionTreeEnsemble(
        n_estimators=100, max_depth=3, random_state=42)
    >>> X, y = np.random.rand(100, 4), np.random.rand(100)
    >>> rte.fit(X, y)
    >>> y_pred = rte.predict(X)

    Notes
    -----
    The Regression Tree Ensemble is built by fitting multiple Regression 
    Tree models. Each tree is trained on the entire dataset, and their 
    predictions are averaged.
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
        self.inspect 
        
        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )
        
        predictions = np.array([tree.predict(X) for tree in self.estimators_])
        return np.mean(predictions, axis=0)

    @property 
    def inspect (self): 
        """ Inspect object whether is fitted or not"""
        msg = ( "{obj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )
        
        if not hasattr (self, 'estimators_'): 
            raise NotFittedError(msg.format(
                obj=self)
            )
        return 1 

class HybridBRTRegressor(BaseEstimator, RegressorMixin):
    """
    Hybrid Boosted Regression Tree (BRT) Ensemble for regression tasks.

    This ensemble model combines multiple Boosted Regression Tree models,
    each of which is an ensemble in itself, created using the 
    principles of boosting.

    HybridBRTRegressorEnsemble class, the n_estimators parameter 
    controls the number of individual Boosted Regression Trees in the ensemble,
    and brt_params is a dictionary of parameters to be passed to each Boosted 
    Regression Tree model. The GradientBoostingRegressor from scikit-learn 
    is used as the individual BRT model. This class's fit method trains 
    each BRT model on the entire dataset, and the predict method averages 
    their predictions for the final output.
    
    Parameters
    ----------
    n_estimators : int
        The number of Boosted Regression Tree models in the ensemble.
    brt_params : dict
        Parameters to be passed to each Boosted Regression Tree model.

    Attributes
    ----------
    brt_ensembles_ : list of GradientBoostingRegressor
        The collection of fitted Boosted Regression Tree ensembles.

    Examples
    --------
    >>> brt_params = {'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.1}
    >>> hybrid_brt = HybridBRTRegressor(
        n_estimators=10, brt_params=brt_params)
    >>> X, y = np.random.rand(100, 4), np.random.rand(100)
    >>> hybrid_brt.fit(X, y)
    >>> y_pred = hybrid_brt.predict(X)

    Notes
    -----
    The Hybrid Boosted Regression Tree Ensemble model aims to combine the 
    strengths of boosting with ensemble learning. Each member of the ensemble 
    is a complete Boosted Regression Tree model, trained to focus on 
    different aspects of the data.
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
        self.inspect 
        
        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )
        
        predictions = np.array([brt.predict(X) for brt in self.brt_ensembles_])
        return np.mean(predictions, axis=0)

    @property 
    def inspect (self): 
        """ Inspect object whether is fitted or not"""
        msg = ( "{obj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )
        
        if not hasattr (self, 'brt_ensembles_'): 
            raise NotFittedError(msg.format(
                obj=self)
            )
        return 1 

class BoostedClassifierTree(BaseEstimator, ClassifierMixin):
    """
    Boosted Decision Tree Classifier.

    BoostedClassifierTree applies boosting techniques to Decision Tree 
    classifiers. Its implementation, each tree in the ensemble is a
    DecisionTreeClassifier. The model is trained iteratively, where each 
    tree tries to correct the mistakes of the previous trees in the ensemble. 
    The final prediction is determined by a thresholding mechanism applied 
    to the cumulative prediction of all trees. The learning_rate controls 
    the contribution of each tree to the final prediction.

    Parameters
    ----------
    n_estimators : int
        The number of trees in the ensemble.
    max_depth : int
        The maximum depth of each decision tree.
    learning_rate : float
        The rate at which the boosting algorithm adapts from 
        previous trees' errors.

    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    Examples
    --------
    >>> bdtc = BoostedClassifierTree(n_estimators=100, 
                                             max_depth=3, learning_rate=0.1)
    >>> X, y = np.random.rand(100, 4), np.random.randint(0, 2, 100)
    >>> bdtc.fit(X, y)
    >>> y_pred = bdtc.predict(X)
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
        self.inspect 
        
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
    
    @property 
    def inspect (self): 
        """ Inspect object whether is fitted or not"""
        msg = ( "{obj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )
        
        if not hasattr (self, 'estimators_'): 
            raise NotFittedError(msg.format(
                obj=self)
            )
        return 1 

class RegressionTreeBasedClassifier(BaseEstimator, ClassifierMixin):
    """
    Regression Tree Ensemble Classifier.

    This classifier is an ensemble of decision trees. It aggregates 
    predictions from multiple decision tree classifiers, typically using
    majority voting for the final classification.

    RegressionTreeBasedClassifier class, the fit method trains multiple 
    DecisionTreeClassifier models on the entire dataset, and the predict 
    method aggregates their predictions using majority voting to determine 
    the final class labels. The n_estimators parameter controls the number 
    of trees in the ensemble, and max_depth controls the depth of each tree. 
    The random_state parameter can be set for reproducibility of results.
    
    Parameters
    ----------
    n_estimators : int
        The number of trees in the ensemble.
    max_depth : int
        The maximum depth of each decision tree.
    random_state : int
        Controls the randomness of the estimator.

    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    Examples
    --------
    >>> rtec = RegressionTreeBasedClassifier(n_estimators=100, max_depth=3,
                                                random_state=42)
    >>> X, y = np.random.rand(100, 4), np.random.randint(0, 2, 100)
    >>> rtec.fit(X, y)
    >>> y_pred = rtec.predict(X)
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
        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )
        
        self.inspect 
        predictions = np.array([tree.predict(X) for tree in self.estimators_])
        # Majority voting
        y_pred = stats.mode(predictions, axis=0).mode[0]
        return y_pred

    @property 
    def inspect (self): 
        """ Inspect object whether is fitted or not"""
        msg = ( "{obj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )
        
        if not hasattr (self, 'estimators_'): 
            raise NotFittedError(msg.format(
                obj=self)
            )
        return 1 
    
class HybridBRTEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Hybrid Boosted Regression Tree Ensemble Classifier.

    This classifier combines multiple Boosted Decision Tree classifiers. 
    Each member of the
    ensemble is a complete Boosted Decision Tree model, with boosting 
    applied to decision trees.

    In `HybridBRTEnsembleClassifier`, each GradientBoostingClassifier in 
    `gb_ensembles_` represents an individual Boosted Decision Tree model. 
    The fit method trains each Boosted Decision Tree model on the entire 
    dataset, and the predict method uses majority voting among all the 
    models to determine the final class labels. The gb_params parameter 
    allows customization of the individual Gradient Boosting models.
    
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

    Examples
    --------
    >>> gb_params = {'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.1}
    >>> hybrid_gb = HybridBRTEnsembleClassifier(n_estimators=10,
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
        self.inspect
        
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
    
    @property 
    def inspect (self): 
        """ Inspect object whether is fitted or not"""
        msg = ( "{obj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )
        
        if not hasattr (self, 'gb_ensembles_'): 
            raise NotFittedError(msg.format(
                obj=self)
            )
        return 1 
    
class WeightedAverageClassifier(BaseEstimator, ClassifierMixin):
    """
    Weighted Average Ensemble Classifier.

    This classifier averages the predictions of multiple base classifiers,
    each weighted by its assigned importance. The ensemble prediction for each class
    is the weighted average of the predicted probabilities from all classifiers.

    ``WeightedAverageEnsembleClassifier`` class, base_classifiers are the 
    individual classifiers in the ensemble, and weights are their 
    corresponding importance weights. The fit method trains each 
    classifier in the ensemble, and the predict method calculates the 
    weighted average of the predicted probabilities from all classifiers
    to determine the final class labels.
    
    This implementation assumes that each base classifier can output 
    class probabilities (i.e., has a predict_proba method), which is 
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
        Weights for each classifier in the base_classifiers list.

    Attributes
    ----------
    classifiers_ : list of fitted classifiers
        The collection of fitted base classifiers.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
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
        self.inspect 
        
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

    @property 
    def inspect (self): 
        """ Inspect object whether is fitted or not"""
        msg = ( "{obj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )
        
        if not hasattr (self, 'classifiers_'): 
            raise NotFittedError(msg.format(
                obj=self)
            )
        return 1 
    
class SimpleAverageClassifier(BaseEstimator, ClassifierMixin):
    """
    Simple Average Ensemble Classifier.

    This classifier averages the predictions of multiple base classifiers.
    The ensemble prediction for each class is the simple average of the predicted
    probabilities from all classifiers.

    ``SimpleAverageEnsembleClassifier`` class, base_classifiers are the 
    individual classifiers in the ensemble. The fit method trains each 
    classifier in the ensemble, and the predict method calculates the 
    simple average of the predicted probabilities from all classifiers to 
    determine the final class labels.
    
    Parameters
    ----------
    base_classifiers : list of objects
        List of base classifiers. Each classifier should support `fit`,
        `predict`, and `predict_proba` methods.

    Attributes
    ----------
    classifiers_ : list of fitted classifiers
        The collection of fitted base classifiers.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
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
        self.inspect 
        
        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )
        
        avg_proba = np.mean([clf.predict_proba(X) for clf in self.classifiers_], axis=0)
        y_pred = np.argmax(avg_proba, axis=1)
        return y_pred

    @property 
    def inspect (self): 
        """ Inspect object whether is fitted or not"""
        msg = ( "{obj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )
        
        if not hasattr (self, 'classifiers_'): 
            raise NotFittedError(msg.format(
                obj=self)
            )
        return 1 

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
            contained subobjects.

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
  




