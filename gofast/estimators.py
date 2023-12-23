# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations 
import re  
import inspect 
import itertools
import numpy as np
from collections import defaultdict

from sklearn.base import BaseEstimator, ClassifierMixin, clone 
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import accuracy_score,  roc_auc_score
from sklearn.model_selection import  train_test_split
from sklearn.pipeline import _name_estimators
from sklearn.preprocessing import LabelEncoder

from ._docstring import DocstringComponents,_core_docs
from ._gofastlog import  gofastlog
from ._typing import List, DataFrame 
from .exceptions import NotFittedError
from .tools.funcutils import _assert_all_types, smart_format
from .tools.validator import check_X_y, get_estimator_name
    

__all__=[ "AdalineGradientDescent", "AdalineStochasticGradientDescent",
    "SequentialBackwardSelection","MajorityVoteClassifier", "GreedyPerceptron", 
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

class _Base:
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
    

class SequentialBackwardSelection (_Base ):
    r"""
    Sequential Backward Selection (SBS) is a feature selection algorithm which 
    aims to reduce dimensionality of the initial feature subspace with a 
    minimum decay  in the performance of the classifier to improve upon 
    computationan efficiency. In certains cases, SBS can even improve the 
    predictive power of the model if a model suffers from overfitting. 
    
    The idea behind the SBS is simple: it sequentially removes features 
    from the full feature subset until the new feature subspace contains the 
    desired number of features. In order to determine which feature is to be 
    removed at each stage, the criterion fonction :math:`J` is needed for 
    minimization [1]_. 
    Indeed, the criterion calculated from the criteria function can simply be 
    the difference in performance of the classifier before and after the 
    removal of this particular feature. Then, the feature to be remove at each 
    stage can simply be the defined as the feature that maximizes this 
    criterion; or in more simple terms, at each stage, the feature that causes 
    the least performance is eliminated loss after removal. Based on the 
    preceding definition of SBS, the algorithm can be outlibe with a few steps:
        
        - Initialize the algorithm with :math:`k=d`, where :math:`d` is the 
            dimensionality of the full feature space, :math:`X_d`. 
        - Determine the feature :math:`x^{-}`,that maximizes the criterion: 
            :math:`x^{-}= argmax J(X_k-x)`, where :math:`x\in X_k`. 
        - Remove the feature :math:`x^{-}` from the feature set 
            :math:`X_{k+1}= X_k -x^{-}; k=k-1`.
        -Terminate if :math:`k` equals to the number of desired features; 
            otherwise go to the step 2. [2]_ 
            
    Parameters 
    -----------
    estimator: callable or instanciated object,
        callable or instance object that has a fit method. 
    k_features: int, default=1 
        the number of features from where starting the selection. It must be 
        less than the number of feature in the training set, otherwise it 
        does not make sense. 
    scoring: callable or str , default='accuracy'
        metric for scoring. availabe metric are 'precision', 'recall', 
        'roc_auc' or 'accuracy'. Any other metric with raise an errors. 
    test_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. If ``train_size`` is also None, it will
        be set to 0.25. 
        
    random_state : int, RandomState instance or None, default=None
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.

    References 
    -----------
    .. [1] Raschka, S., Mirjalili, V., 2019. Python Machine Learning, 3rd ed. Packt.
    .. [2] Ferri F., Pudil F., Hatef M., and Kittler J., Comparative study of 
        the techniques for Large-scale feature selection, pages 403-413, 1994.
    
    Attributes 
    -----------
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
        
    indices_: tuple of dimensionnality X
        Collect the indices of subset of the best validated models 
        
    subsets_: list, 
        list of `indices_` 
        
    scores_: list, 
        Collection of the scores of the best model got during the
        cross-validating 
        
    k_score_: float, 
        The score of the desired feature. 
        
    Examples
    --------
    >>> from gofast.exlib.sklearn import KNeighborsClassifier , train_test_split
    >>> from gofast.datasets import fetch_data
    >>> from gofast.base import SequentialBackwardSelection
    >>> X, y = fetch_data('bagoue analysed') # data already standardized
    >>> Xtrain, Xt, ytrain,  yt = train_test_split(X, y)
    >>> knn = KNeighborsClassifier(n_neighbors=5)
    >>> sbs= SequentialBackwardSelection (knn)
    >>> sbs.fit(Xtrain, ytrain )

    """
    _scorers = dict (accuracy = accuracy_score , recall = recall_score , 
                   precision = precision_score, roc_auc= roc_auc_score 
                   )
    def __init__ (self, estimator=None , k_features=1 , 
                  scoring ='accuracy', test_size = .25 , 
                  random_state = 42 ): 
        self.estimator=estimator 
        self.k_features=k_features 
        self.scoring=scoring 
        self.test_size=test_size
        self.random_state=random_state 
        
    def fit(self, X, y) :
        """  Fit the training data 
        
        Note that SBS splits the datasets into a test and training insite the 
        fit function. :math:`X` is still fed to the algorithm. Indeed, SBS 
        will then create a new training subsets for testing (validation) and 
        training , which is why this test set is also called the validation 
        dataset. This approach is necessary to prevent our original test set 
        to becoming part of the training data. 
        
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
        self: `SequentialBackwardSelection` instance 
            returns ``self`` for easy method chaining.
        
        """
        X, y = check_X_y(
            X, 
            y, 
            estimator = get_estimator_name(self ), 
            to_frame= True, 
            )
        
        self._check_sbs_args(X)
        
        if hasattr(X, 'columns'): 
            self.feature_names_in = list(X.columns )
            X = X.values 
            
        Xtr, Xt,  ytr, yt = train_test_split(X, y , test_size=self.test_size, 
                                            random_state=self.random_state 
                                            )
        dim = Xtr.shape [1] 
        self.indices_= tuple (range (dim))
        self.subsets_= [self.indices_]
        score = self._compute_score(Xtr, Xt,  ytr, yt, self.indices_)
        self.scores_=[score]
        # compute the score for p indices in 
        # list indices in dimensions 
        while dim > self.k_features: 
            scores , subsets = [], []
            for p in itertools.combinations(self.indices_, r=dim-1):
                score = self._compute_score(Xtr, Xt,  ytr, yt, p)
                scores.append (score) 
                subsets.append (p)
            
            best = np.argmax (scores) 
            self.indices_= subsets [best]
            self.subsets_.append(self.indices_)
            dim -=1 # go back for -1 
            
            self.scores_.append (scores[best])
            
        # set  the k_feature score 
        self.k_score_= self.scores_[-1]
        
        return self 
        
    def transform (self, X): 
        """ Transform the training set 
        
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
        X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
            New transformed training set with selected features columns 
        
        """
        if not hasattr (self, 'indices_'): 
            raise NotFittedError(
                "Can't call transform with estimator not fitted yet."
                " Fit estimator by calling the 'fit' method with appropriate"
                " arguments.")
        return X[:, self.indices_]
    
    def _compute_score (self, Xtr, Xt,  ytr, yt, indices):
        """ Compute score from splitting `X` and indices """
        self.estimator.fit(Xtr[:, indices], ytr)
        y_pred = self.estimator.predict (Xt [:, indices])
        score = self.scoring (yt, y_pred)
        
        return score 

    def _check_sbs_args (self, X): 
        """ Assert SBS main arguments  """
        
        if not hasattr(self.estimator, 'fit'): 
            raise TypeError ("Estimator must have a 'fit' method.")
        try : 
            self.k_features = int (self.k_features)
        except  Exception as err: 
            raise TypeError ("Expect an integer for number of feature k,"
                             f" got {type(self.k_features).__name__!r}"
                             ) from err
        if self.k_features > X.shape [1] :
            raise ValueError ("Too many number of features."
                              f" Expect max-features={X.shape[1]}")
        if  ( 
            callable(self.scoring) 
            or inspect.isfunction ( self.scoring )
            ): 
            self.scoring = self.scoring.__name__.replace ('_score', '')
        
        if self.scoring not in self._scorers.keys(): 
            raise ValueError (
                f"Accept only scorers {list (self._scorers.keys())}"
                f"for scoring, not {self.scoring!r}")
            
        self.scoring = self._scorers[self.scoring] 
        
        self.scorer_name_ = self.scoring.__name__.replace (
            '_score', '').title ()
        
    def __repr__(self): 
        """ Represent the  Sequential Backward Selection class """
        get_params = self.get_params()  
        get_params.pop('scoring')
        if hasattr (self, 'scorer_name_'): 
            get_params ['scoring'] =self.scorer_name_ 
        
        tup = tuple (f"{key}={val}".replace ("'", '') for key, val in 
                     get_params.items() )
        
        return self.__class__.__name__ + str(tup).replace("'", "") 
    
class GreedyPerceptron (_Base): 
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
        if not hasattr (self, 'w_'): 
            raise NotFittedError("Can't call 'predict' method with estimator"
                                 " not fitted yet. Fit estimator by calling"
                                 " the 'fit' method first.")
        return np.where (self.net_input(X) >=.0 , 1 , -1 )
    
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
            
        
class AdalineStochasticGradientDescent (_Base) :
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
                 random_state:int = 42 ) :
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
        return np.where (self.activation(self.net_input(X))>=0. , 1, -1)
    
    def __repr__(self): 
        """ Represent the output class """
        
        tup = tuple (f"{key}={val}".replace ("'", '') for key, val in 
                     self.get_params().items() )
        
        return self.__class__.__name__ + str(tup).replace("'", "") 
    
class AdalineGradientDescent (_Base): 
    r"""Adaptative Linear Neuron Classifier 
    
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
            cost = (errors **2 ).sum() / 2. 
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
        return np.where (self.activation(self.net_input(X))>=0. , 1, -1)
    
    def __repr__(self): 
        """ Represent the output class """
        
        tup = tuple (f"{key}={val}".replace ("'", '') for key, val in 
                     self.get_params().items() )
        
        return self.__class__.__name__ + str(tup).replace("'", "") 
        
def get_params (obj: object 
                ) -> dict: 
    """
    Get object parameters. 
    
    Object can be callable or instances 
    
    :param obj: object , can be callable or instance 
    
    :return: dict of parameters values 
    
    :examples: 
    >>> from sklearn.svm import SVC 
    >>> from gofast.base import get_params 
    >>> sigmoid= SVC (
        **{
            'C': 512.0,
            'coef0': 0,
            'degree': 1,
            'gamma': 0.001953125,
            'kernel': 'sigmoid',
            'tol': 1.0 
            }
        )
    >>> pvalues = get_params( sigmoid)
    >>> {'decision_function_shape': 'ovr',
         'break_ties': False,
         'kernel': 'sigmoid',
         'degree': 1,
         'gamma': 0.001953125,
         'coef0': 0,
         'tol': 1.0,
         'C': 512.0,
         'nu': 0.0,
         'epsilon': 0.0,
         'shrinking': True,
         'probability': False,
         'cache_size': 200,
         'class_weight': None,
         'verbose': False,
         'max_iter': -1,
         'random_state': None
     }
    """
    if hasattr (obj, '__call__'): 
        cls_or_func_signature = inspect.signature(obj)
        PARAMS_VALUES = {k: None if v.default is (inspect.Parameter.empty 
                         or ...) else v.default 
                    for k, v in cls_or_func_signature.parameters.items()
                    # if v.default is not inspect.Parameter.empty
                    }
    elif hasattr(obj, '__dict__'): 
        PARAMS_VALUES = {k:v  for k, v in obj.__dict__.items() 
                         if not (k.endswith('_') or k.startswith('_'))}
    
    return PARAMS_VALUES


def _existfeatures (df, features, error='raise'): 
    """Control whether the features exists or not  
    
    :param df: a dataframe for features selections 
    :param features: list of features to select. Lits of features must be in the 
        dataframe otherwise an error occurs. 
    :param error: str - raise if the features don't exist in the dataframe. 
        *default* is ``raise`` and ``ignore`` otherwise. 
        
    :return: bool 
        assert whether the features exists 
    """
    isf = False  
    
    error= 'raise' if error.lower().strip().find('raise')>= 0  else 'ignore' 

    if isinstance(features, str): 
        features =[features]
        
    features = _assert_all_types(features, list, tuple, np.ndarray)
    set_f =  set (features).intersection (set(df.columns))
    if len(set_f)!= len(features): 
        nfeat= len(features) 
        msg = f"Feature{'s' if nfeat >1 else ''}"
        if len(set_f)==0:
            if error =='raise':
                raise ValueError (f"{msg} {smart_format(features)} "
                                  f"{'does not' if nfeat <2 else 'dont'}"
                                  " exist in the dataframe")
            isf = False 
        # get the difference 
        diff = set (features).difference(set_f) if len(
            features)> len(set_f) else set_f.difference (set(features))
        nfeat= len(diff)
        if error =='raise':
            raise ValueError(f"{msg} {smart_format(diff)} not found in"
                             " the dataframe.")
        isf = False  
    else : isf = True 
    
    return isf  
    
def _selectfeatures (
        df: DataFrame,
        features: List[str] =None, 
        include = None, 
        exclude = None,
        coerce: bool=False,
        **kwd
        ): 
    """ Select features  and return new dataframe.  
    
    :param df: a dataframe for features selections 
    :param features: list of features to select. Lits of features must be in the 
        dataframe otherwise an error occurs. 
    :param include: the type of data to retrieved in the dataframe `df`. Can  
        be ``number``. 
    :param exclude: type of the data to exclude in the dataframe `df`. Can be 
        ``number`` i.e. only non-digits data will be keep in the data return.
    :param coerce: return the whole dataframe with transforming numeric columns.
        Be aware that no selection is done and no error is raises instead. 
        *default* is ``False``
    :param kwd: additional keywords arguments from `pd.astype` function 
    
    :ref: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.astype.html
    """
    
    if features is not None: 
        _existfeatures(df, features, error ='raise')
    # change the dataype 
    df = df.astype (float, errors ='ignore', **kwd) 
    # assert whether the features are in the data columns
    if features is not None: 
        return df [features] 
    # raise ValueError: at least one of include or exclude must be nonempty
    # use coerce to no raise error and return data frame instead.
    return df if coerce else df.select_dtypes (include, exclude) 

