# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations 

import inspect
import warnings  
from pprint import pprint 
import numpy as np 

from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold
from sklearn.model_selection import cross_val_score, StratifiedKFold, LeaveOneOut
from sklearn.pipeline import Pipeline  

from .._docstring import DocstringComponents, _core_docs 
from .._gofastlog import gofastlog
from .._typing import _F, List,ArrayLike, NDArray, Dict, Any 
from .._typing import DataFrame, Series,Optional, Union
from ..exceptions import EstimatorError, NotFittedError
from ..tools.box import Boxspace 
from ..tools.funcutils import _assert_all_types, get_params, save_job
from ..tools.funcutils import listing_items_format, pretty_printer
from ..tools.validator import check_X_y, check_array, check_consistent_length 
from ..tools.validator import get_estimator_name

from .utils import get_scorers, naive_evaluation
 
_logger = gofastlog().get_gofast_logger(__name__)

__all__=["BaseEvaluation", "GridSearch", "GridSearchMultiple",
         "get_best_kPCA_params", "CrossValidator", 
    ]

_param_docs = DocstringComponents.from_nested_components(
    core=_core_docs["params"], 
    )

class CrossValidator:
    """
    A class for handling cross-validation of machine learning models.

    This class provides functionalities for performing cross-validation on
    various classifiers or regressors using different scoring strategies.

    Attributes
    ----------
    clf : BaseEstimator
        The machine learning model to be used for cross-validation.
    cv : int, optional
        The number of folds for cross-validation (default is 5).
    scoring : str, optional
        The scoring strategy to evaluate the model (default is 'accuracy').
    results : Optional[Tuple[np.ndarray, float]]
        Stored results of the cross-validation, including scores and mean score.

    Methods
    -------
    fit(X, y=None):
        Fits the model to the data (X, y) and performs cross-validation.
    calculate_mean_score():
        Calculates and returns the mean of the cross-validation scores.
    display_results():
        Prints the cross-validation scores and their mean.
        
    Examples
    ---------
    >>> from gofast.models.search import CrossValidator
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> clf = DecisionTreeClassifier()
    >>> cross_validator = CrossValidator(clf, cv=5, scoring='accuracy')
    >>> X, y = # Load your dataset here
    >>> cross_validator.fit(X, y)
    >>> cross_validator.displayResults()
    """

    def __init__(
            self, clf: BaseEstimator,
            cv: int = 5, 
            scoring: str = 'accuracy'
            ):
        """
        Initializes the CrossValidator with a classifier and evaluation parameters.
        """
        self.clf = clf
        self.cv = cv
        self.scoring = scoring

    def fit(self, X: Union[ArrayLike, list],
            y: Optional[Union[ArrayLike, list]] = None):
        """
        Fits the model to the data (X, y) and performs cross-validation.

        Parameters
        ----------
        X : np.ndarray or list
            The feature dataset for training the model.
        y : np.ndarray or list, optional
            The target labels for the dataset (default is None).

        Raises
        ------
        ValueError
            If the target labels `y` are not provided for supervised 
            learning models.
        """
        if y is None and issubclass(self.clf.__class__, BaseEstimator):
            raise ValueError("Target labels `y` must be provided for"
                             " supervised learning models.")
        
        scores = cross_val_score(self.clf, X, y, cv=self.cv, scoring=self.scoring)
        mean_score = scores.mean()
        self.results_ = (scores, mean_score)

        return self 
    
    def calculateMeanScore(self) -> float:
        """
        Calculate and return the mean score from the cross-validation results.

        Returns
        -------
        float
            Mean of the cross-validation scores.

        Raises
        ------
        ValueError
            If cross-validation has not been performed yet.
        """
        self.inspect 
        if self.results_ is None:
            raise ValueError("Cross-validation has not been performed yet.")
        return self.results_[1]
    
        return self 

    def displayResults(self):
        """
        Display the cross-validation scores and their mean.

        Raises
        ------
        ValueError
            If cross-validation has not been performed yet.
        """
        self.inspect 
        if self.results_ is None:
            raise ValueError("Cross-validation has not been performed yet.")
        
        scores, mean_score = self.results_
        clf_name = self.clf.__class__.__name__
        print(f'Classifier/Regressor: {clf_name}')
        print(f'Cross-validation scores: {scores}')
        print(f'Mean score: {mean_score:.4f}')

    def setCVStrategy(self, cv_strategy: Union[int, str, object], 
                        n_splits: int = 5, random_state: int = None):
        """
        Sets the cross-validation strategy for the model.

        Parameters
        ----------
        cv_strategy : int, str, or cross-validation generator object
            The cross-validation strategy to be used. If an integer is provided,
            KFold is assumed with that many splits. If a string is provided,
            it must be one of 'kfold', 'stratified', or 'leaveoneout'.
            Alternatively, a custom cross-validation generator object can be provided.
        n_splits : int, optional
            The number of splits for KFold or StratifiedKFold (default is 5).
        random_state : int, optional
            Random state for reproducibility (default is None).
        
        Returns
        -------
        cv : cross-validation generator object
            The configured cross-validation generator instance.
            
        Notes
        -----
        - 'kfold': KFold divides all the samples into 'n_splits' number of groups,
           called folds, of equal sizes (if possible). Each fold is then used as
           a validation set once while the remaining folds form the training set.
        - 'stratified': StratifiedKFold is a variation of KFold that returns
           stratified folds. Each set contains approximately the same percentage
           of samples of each target class as the complete set.
        - 'leaveoneout': LeaveOneOut (LOO) is a simple cross-validation. Each
           learning set is created by taking all the samples except one, the test
           set being the sample left out.
           
        Raises
        ------
        ValueError
            If an invalid cross-validation strategy or type is provided.
            
        Examples 
        ---------
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> clf = DecisionTreeClassifier()
        >>> cross_validator = CrossValidator(clf, cv=5, scoring='accuracy')
        >>> cross_validator.setCVStrategy('stratified', n_splits=5)
        >>> X, y = # Load your dataset here
        >>> cross_validator.fit(X, y)
        >>> cross_validator.display_results()
        """
        if isinstance(cv_strategy, int):
            self.cv = KFold(n_splits=cv_strategy, random_state=random_state, shuffle=True)
        elif isinstance(cv_strategy, str):
            cv_strategy = cv_strategy.lower()
            if cv_strategy == 'kfold':
                self.cv = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
            elif cv_strategy == 'stratified':
                self.cv = StratifiedKFold(n_splits=n_splits, random_state=random_state, 
                                          shuffle=True)
            elif cv_strategy == 'leaveoneout':
                self.cv = LeaveOneOut()
            else:
                raise ValueError(f"Invalid cross-validation strategy: {cv_strategy}")
        elif hasattr(cv_strategy, 'split'):
            self.cv = cv_strategy
        else:
            raise ValueError("cv_strategy must be an integer, a string,"
                             " or a cross-validation generator object.")
        
        return self.cv 

    def applyCVStrategy(
        self, 
        X: ArrayLike, 
        y: ArrayLike, 
        metrics: Optional[Dict[str, _F[[ArrayLike, ArrayLike], float]]] = None, 
        display_results: bool = False
        ):
        """
        Applies the configured cross-validation strategy to the given dataset
        and evaluates the model.

        This method performs cross-validation using the strategy set by the
        `setCVStrategy` method. It fits the model on each training set and
        evaluates it on each corresponding test set. The results, including
        scores and other metrics, are stored in the `cv_results_` attribute.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data to be used for cross-validation.
        y : array-like of shape (n_samples,)
            Target values corresponding to X.
        metrics : dict, optional
            A dictionary where keys are metric names and values are functions
            that compute the metric. Each function should accept two arguments:
            true labels and predicted labels. Default is None.
            
        display_results : bool, optional
            Whether to print the cross-validation results. Default is False.

        Stores
        -------
        cv_results_ : dict
            Dictionary containing the results of cross-validation. This includes
            scores for each fold, mean score, and standard deviation.

        Notes
        -----
        - KFold: Suitable for large datasets and is not stratified.
        - StratifiedKFold: Suitable for imbalanced datasets, as it preserves
          the percentage of samples for each class.
        - LeaveOneOut: Suitable for small datasets but computationally expensive.
        - Ensure that the cross-validation strategy is appropriately chosen
          for your dataset and problem. For example, StratifiedKFold is more
          suited for datasets with imbalanced class distributions.
        - The metrics functions should match the signature of sklearn.metrics
          functions, i.e., they accept two arguments: y_true and y_pred.
          
        Raises
        ------
        ValueError
            If the cross-validation strategy has not been set prior to calling
            this method.
            
        Examples
        --------
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> from sklearn.datasets import load_iris
        >>> X, y = load_iris(return_X_y=True)
        >>> clf = DecisionTreeClassifier()
        >>> cross_validator = CrossValidator(clf)
        >>> cross_validator.setCVStrategy('kfold', n_splits=5)
        >>> cross_validator.applyCVStrategy(X, y, display_results=True)
        >>> from sklearn.metrics import accuracy_score, precision_score
        >>> additional_metrics = {
        ...     'accuracy': accuracy_score,
        ...     'precision': precision_score
        ... }
        >>> cross_validator.applyCVStrategy(X, y, metrics=additional_metrics)
        """
        if not hasattr(self, 'cv'):
            raise ValueError("Cross-validation strategy not set."
                             " Please call setCVStrategy first.")

        self.cv_results_ = {'scores': [], 'additional_metrics': {}}
        if metrics:
            for metric in metrics:
                self.cv_results_['additional_metrics'][metric] = []

        for train_index, test_index in self.cv.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model_clone = clone(self.model)
            model_clone.fit(X_train, y_train)
            y_pred = model_clone.predict(X_test)

            score = model_clone.score(X_test, y_test)
            self.cv_results_['scores'].append(score)

            if metrics:
                for metric_name, metric_func in metrics.items():
                    metric_score = metric_func(y_test, y_pred)
                    self.cv_results_['additional_metrics'][metric_name].append(metric_score)

        self.cv_results_['mean_score'] = np.mean(self.cv_results_['scores'])
        self.cv_results_['std_dev'] = np.std(self.cv_results_['scores'])

        if metrics:
            for metric in self.cv_results_['additional_metrics']:
                mean_metric = np.mean(self.cv_results_['additional_metrics'][metric])
                self.cv_results_['additional_metrics'][metric] = mean_metric

        if display_results:
            self._display_results()

        return self

    def _display_results(self):
        """
       Displays the results of cross-validation.

        This method prints the mean score, standard deviation, and individual
        fold scores from the cross-validation.

        Raises
        ------
        ValueError
            If this method is called before applying cross-validation.
        """
        if not hasattr(self, 'cv_results_'):
            raise ValueError("Cross-validation not applied. Please call apply_cv_strategy.")

        print(f"CV Mean Score: {self.cv_results_['mean_score']:.3f}")
        print(f"CV Standard Deviation: {self.cv_results_['std_dev']:.3f}")
        print("Scores per fold: ", self.cv_results_['scores'])


    @property 
    def inspect(self): 
        """ Inspect data and trigger plot after checking the data entry. 
        Raises `NotFittedError` if `ExPlot` is not fitted yet."""
        
        msg = ( "{expobj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )
        
        if not hasattr (self, "results_" ): 
            raise NotFittedError(msg.format(expobj=self))
        return 1 

class GridSearch: 
    __slots__=(
        '_base_estimator',
        'grid_params', 
        'scoring',
        'cv', 
        '_kind', 
        'grid_kws', 
        'best_params_',
        'cv_results_',
        'feature_importances_',
        'best_estimator_',
        'verbose',
        'grid_kws',
        )
    def __init__(
        self,
        base_estimator:_F,
        grid_params:Dict[str,Any],
        cv:int =4,
        kind:str ='GridSearchCV',
        scoring:str = 'nmse',
        savejob:bool=False, 
        filename:str=None, 
        verbose:int=0, 
        **grid_kws
        ): 
        
        self._base_estimator = base_estimator 
        self.grid_params = grid_params 
        self.scoring = scoring 
        self.cv = cv 
        self.best_params_ =None 
        self.cv_results_= None
        self.feature_importances_= None
        self.grid_kws = grid_kws 
        self._kind = kind 
        self.verbose=verbose
        self.filename=filename

    @property 
    def base_estimator (self): 
        """ Return the base estimator class"""
        return self._base_estimator 
    
    @base_estimator.setter 
    def base_estimator (self, base_est): 
        if not hasattr (base_est, 'fit'): 
            raise EstimatorError(
                f"Wrong estimator {get_estimator_name(base_est)!r}. Each"
                " estimator must have a fit method. Refer to scikit-learn"
                " https://scikit-learn.org/stable/modules/classes.html API"
                " reference to build your own estimator.") 

        self._base_estimator =base_est 
        
    @property 
    def kind(self): 
        """ Kind of searched. `RandomizedSearchCV` or `GridSearchCV`."""
        return self._kind 
    
    @kind.setter 
    def kind (self, ksearch): 
        """`kind attribute checker"""
        if 'gridsearchcv1'.find( str(ksearch).lower())>=0: 
            ksearch = 'GridSearchCV' 
        elif 'randomizedsearchcv2'.find( str(ksearch).lower())>=0:
            ksearch = 'RandomizedSearchCV'
        else: raise ValueError (
            " Unkown the kind of parameter search {ksearch!r}."
            " Supports only 'GridSearchCV' and 'RandomizedSearchCV'.")
        self._kind = ksearch 

    def fit(self,  X, y): 
        """ Fit method using base Estimator and populate gridSearch 
        attributes.
 
        Parameters
        ----------
        X:  Ndarray ( M x N) matrix where ``M=m-samples``, & ``N=n-features``)
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
        ----------
        ``self``: `GridSearch`
            Returns :class:`~.GridSearch` 
    
        """
        if callable (self.base_estimator): 
            self.base_estimator= self.base_estimator () 
            parameters = get_params (self.base_estimator.__init__)
            
            if self.verbose > 0: 
                msg = ("Estimator {!r} is cloned with default arguments{!r}"
                       " for cross validation search.".format(
                           get_estimator_name (self.base_estimator), parameters)
                       )
                warnings.warn(msg)
        
        self.kind =self._kind 
        
        if self.kind =='GridSearchCV': 
            searchGridMethod = GridSearchCV 
        elif self.kind=='RandomizedSearchCV': 
            searchGridMethod= RandomizedSearchCV 
            
        if self.scoring in ( 'nmse', None): 
            self.scoring ='neg_mean_squared_error'
        # assert scoring values 
        get_scorers(scorer= self.scoring , check_scorer= True, error ='raise' ) 
         
        gridObj = searchGridMethod(
            self.base_estimator, 
            self.grid_params,
            scoring = self.scoring , 
            cv = self.cv,
            **self.grid_kws
            )
        gridObj.fit(X, y)
        
        #make_introspection(self,  gridObj)
        params = ('best_params_','best_estimator_','cv_results_')
        params_values = [getattr (gridObj , param, None) for param in params] 
        
        for param , param_value in zip(params, params_values ):
            setattr(self, param, param_value)
        # set feature_importances if exists 
        try : 
            attr_value = gridObj.best_estimator_.feature_importances_
        except AttributeError: 
            setattr(self,'feature_importances_', None )
        else : 
            setattr(self,'feature_importances_', attr_value)
        
        self.data_=  { f"{get_estimator_name (self.base_estimator)}":  params_values} 
        if self.savejob: 
            self.filename = self.filename or get_estimator_name ( self.base_estimator)+ '.results'
            save_job ( job= self.data_ , savefile = self.filename ) 

        return self
    
GridSearch.__doc__="""\
Fine-tune hyperparameters using grid search methods. 

Search Grid will be able to  fiddle with the hyperparameters until to 
      
Parameters 
------------
base_estimator: Callable,
    estimator for trainset and label evaluating; something like a 
    class that implements a fit method. Refer to 
    https://scikit-learn.org/stable/modules/classes.html

grid_params: list of dict, 
    list of hyperparameters params  to be fine-tuned.For instance::
    
        param_grid=[dict(
            kpca__gamma=np.linspace(0.03, 0.05, 10),
            kpca__kernel=["rbf", "sigmoid"]
            )]

pipeline: Callable or :class:`~sklearn.pipeline.Pipeline` object 
    If `pipeline` is given , `X` is transformed accordingly, Otherwise 
    evaluation is made using purely the base estimator with the given `X`. 

prefit: bool, default=False, 
    If ``False``, does not need to compute the cross validation score once 
    again and ``True`` otherwise.
    
savejob: bool, default=False
    Save your model parameters to external file using 'joblib' or Python 
    persistent 'pickle' module. Default sorted to 'joblib' format. 
    
filename: str, 
    Name of the file to collect the cross-validation results. It is composed 
    of a dictionnary of the best parameters and  the results of CV. If name is 
    not given, it shoul be created using the estimator name. 

{params.core.cv}
    The default is ``4``.
kind:str, default='GridSearchCV' or '1'
    Kind of grid parameter searches. Can be ``1`` for ``GridSearchCV`` or
    ``2`` for ``RandomizedSearchCV``. 
{params.core.scoring} 
{params.core.random_state}

Examples
-----------
>>> from pprint import pprint 
>>> from gofast.datasets import fetch_data 
>>> from gofast.models.search import GridSearch
>>> from sklearn.ensemble import RandomForestClassifier
>>> X_prepared, y_prepared =fetch_data ('bagoue prepared')
>>> grid_params = [ dict(
...        n_estimators=[3, 10, 30], max_features=[2, 4, 6, 8]), 
...        dict(bootstrap=[False], n_estimators=[3, 10], 
...                             max_features=[2, 3, 4])
...        ]
>>> forest_clf = RandomForestClassifier()
>>> grid_search = GridSearch(forest_clf, grid_params)
>>> grid_search.fit(X= X_prepared,y =  y_prepared,)
>>> pprint(grid_search.best_params_ )
{{'max_features': 8, 'n_estimators': 30}}
>>> pprint(grid_search.cv_results_)
""".format (params=_param_docs,
)
    
class GridSearchMultiple:
    def __init__ (
        self, 
        estimators: _F, 
        scoring:str,  
        grid_params: Dict[str, Any],
        *, 
        kind:str ='GridSearchCV', 
        cv: int =7, 
        random_state:int =42,
        savejob:bool =False,
        filename: str=None, 
        verbose:int =0,
        **grid_kws, 
        ):
        self.estimators = estimators 
        self.scoring=scoring 
        self.grid_params=grid_params
        self.kind=kind 
        self.cv=cv
        self.savejob=savejob
        self.filename=filename 
        self.verbose=verbose 
        self.grid_kws=grid_kws
        
    def fit(
            self, 
            X: NDArray, 
            y:ArrayLike, 
        ):
        """ Fit methods, evaluate each estimator and store models results.
        
        Parameters 
        -----------
        {params.core.X}
        {params.core.y}
        
        Returns 
        --------
        {returns.self}

        """.format( 
            params =_param_docs , 
            returns = _core_docs['returns'] 
        ) 
        err_msg = (" Each estimator must have its corresponding grid params,"
                   " i.e estimators and grid params must have the same length."
                   " Please provide the appropriate arguments.")
        try: 
            check_consistent_length(self.estimators, self.grid_params)
        except ValueError as err : 
            raise ValueError (str(err) +f". {err_msg}")

        self.best_estimators_ =[] 
        self.data_ = {} 
        models_= {}
        msg =''
        
        self.filename = self.filename or '__'.join(
            [get_estimator_name(b) for b in self.estimators ])
        
        for j, estm in enumerate(self.estimators):
            estm_name = get_estimator_name(estm)
            msg = f'{estm_name} is evaluated with {self.kind}.'
            searchObj = GridSearch(base_estimator=estm, 
                                    grid_params= self.grid_params[j], 
                                    cv = self.cv, 
                                    kind=self.kind, 
                                    scoring=self.scoring, 
                                    **self.grid_kws
                                      )
            searchObj.fit(X, y)
            best_model_clf = searchObj.best_estimator_ 
            
            if self.verbose > 7 :
                msg += ( 
                    f"\End {self.kind} search. Set estimator {estm_name!r}"
                    " best parameters, cv_results and other importances" 
                    " attributes\n'"
                 )
            self.data_[estm_name]= {
                                'best_model_':searchObj.best_estimator_ ,
                                'best_params_':searchObj.best_params_ , 
                                'cv_results_': searchObj.cv_results_,
                                'grid_params':self.grid_params[j],
                                'scoring':self.scoring, 
                                "grid_kws": self.grid_kws
                                    }
            
            models_[estm_name] = searchObj
            
            
            msg += ( f"Cross-evaluatation the {estm_name} best model."
                    f" with KFold ={self.cv}"
                   )
            bestim_best_scores, _ = naive_evaluation(
                best_model_clf, 
                X,
                y,
                cv = self.cv, 
                scoring = self.scoring,
                display ='on' if self.verbose > 7 else 'off'
                )
            # store the best scores 
            self.data_[f'{estm_name}']['best_scores']= bestim_best_scores
    
            self.best_estimators_.append((estm, searchObj.best_estimator_,
                          searchObj.best_params_, 
                          bestim_best_scores) 
                        )
            
        # save models into a Box 
        d = {**models_, ** dict( 
            keys_ = list (models_.values() ), 
            values_ = list (models_.values() ), 
            models_= models_, 
            )
            
            }
        self.models= Boxspace(**d) 
        
        if self.verbose:
            msg += ('\Pretty print estimators results using'
                    f' scoring ={self.scoring!r}')
            pretty_printer(clfs=self.best_estimators_, scoring =self.scoring, 
                          clf_scores= None)
        if self.savejob:
            msg += ('\Serialize the dict of fine-tuned '
                    f'parameters to `{self.filename}`.')
            save_job (job= self.data_ , savefile = self.filename )
            _logger.info(f'Dumping models `{self.filename}`!')
            
            if self.verbose: 
                pprint(msg)
                bg = ("Job is successfully saved. Try to fetch your job from "
                       f"{self.filename!r} using")
                lst =[ "{}.load('{}') or ".format('joblib', self.filename ),
                      "{}.load('{}')".format('pickle', self.filename)]
                
                listing_items_format(lst, bg )
    
        if self.verbose:  
            pprint(msg)    

        return self 

GridSearchMultiple.__doc__="""\
Search and find multiples best parameters from differents
estimators.

Parameters
----------
estimators: list of callable obj 
    list of estimator objects to fine-tune their hyperparameters 
    For instance::
        
    random_state=42
    # build estimators
    logreg_clf = LogisticRegression(random_state =random_state)
    linear_svc_clf = LinearSVC(random_state =random_state)
    sgd_clf = SGDClassifier(random_state = random_state)
    svc_clf = SVC(random_state =random_state) 
               )
    estimators =(svc_clf,linear_svc_clf, logreg_clf, sgd_clf )
 
grid_params: list 
    list of parameters Grids. For instance:: 
        
        grid_params= ([
        dict(C=[1e-2, 1e-1, 1, 10, 100], gamma=[5, 2, 1, 1e-1, 1e-2, 1e-3],
                     kernel=['rbf']), 
        dict(kernel=['poly'],degree=[1, 3,5, 7], coef0=[1, 2, 3], 
         'C': [1e-2, 1e-1, 1, 10, 100])], 
        [dict(C=[1e-2, 1e-1, 1, 10, 100], loss=['hinge'])], 
        [dict()], [dict()]
        )
{params.core.cv} 

{params.core.scoring}
   
kind:str, default='GridSearchCV' or '1'
    Kind of grid parameter searches. Can be ``1`` for ``GridSearchCV`` or
    ``2`` for ``RandomizedSearchCV``. 
    
{params.core.random_state} 

savejob: bool, default=False
    Save your model parameters to external file using 'joblib' or Python 
    persistent 'pickle' module. Default sorted to 'joblib' format. 
    
filename: str, 
    Name of the file to collect the cross-validation results. It is composed 
    of a dictionnary of the best parameters and  the results of CV. If name is 
    not given, it shoul be created using the estimator name. 

{params.core.verbose} 

grid_kws: dict, 
    Argument passed to `grid_method` additional keywords. 
    
Examples
--------
>>> from gofast.search import GridSearchMultiple , displayFineTunedResults
>>> from sklearn.svm import SVC, LinearSVC 
>>> from sklearn.linear_model import SGDClassifier,LogisticRegression
>>> X, y  = gf.fetch_data ('bagoue prepared') 
>>> X
... <344x18 sparse matrix of type '<class 'numpy.float64'>'
... with 2752 stored elements in Compressed Sparse Row format>
>>> # As example, we can build 04 estimators and provide their 
>>> # grid parameters range for fine-tuning as ::
>>> random_state=42
>>> logreg_clf = LogisticRegression(random_state =random_state)
>>> linear_svc_clf = LinearSVC(random_state =random_state)
>>> sgd_clf = SGDClassifier(random_state = random_state)
>>> svc_clf = SVC(random_state =random_state) 
>>> estimators =(svc_clf,linear_svc_clf, logreg_clf, sgd_clf )
>>> grid_params= ([dict(C=[1e-2, 1e-1, 1, 10, 100], 
                        gamma=[5, 2, 1, 1e-1, 1e-2, 1e-3],kernel=['rbf']), 
                   dict(kernel=['poly'],degree=[1, 3,5, 7], coef0=[1, 2, 3],
                        C= [1e-2, 1e-1, 1, 10, 100])],
                [dict(C=[1e-2, 1e-1, 1, 10, 100], loss=['hinge'])], 
                [dict()], # we just no provided parameter for demo
                [dict()]
                )
>>> #Now  we can call :class:`gofast.models.GridSearchMultiple` for
>>> # training and self-validating as:
>>> gobj = GridSearchMultiple(estimators = estimators, 
                       grid_params = grid_params ,
                       cv =4, 
                       scoring ='accuracy', 
                       verbose =1,   #> 7 put more verbose 
                       savejob=False ,  # set true to save job in binary disk file.
                       kind='GridSearchCV').fit(X, y)
>>> # Once the parameters are fined tuned, we can display the fined tuning 
>>> # results using displayFineTunedResults`` function
>>> displayFineTunedResults (gobj.models.values_) 
MODEL NAME = SVC
BEST PARAM = {{'C': 100, 'gamma': 0.01, 'kernel': 'rbf'}}
BEST ESTIMATOR = SVC(C=100, gamma=0.01, random_state=42)

MODEL NAME = LinearSVC
BEST PARAM = {{'C': 100, 'loss': 'hinge'}}
BEST ESTIMATOR = LinearSVC(C=100, loss='hinge', random_state=42)

MODEL NAME = LogisticRegression
BEST PARAM = {{}}
BEST ESTIMATOR = LogisticRegression(random_state=42)

MODEL NAME = SGDClassifier
BEST PARAM = {{}}
BEST ESTIMATOR = SGDClassifier(random_state=42)

Notes
--------
Call :func:`~.get_scorers` or use `sklearn.metrics.SCORERS.keys()` to get all
the metrics used to evaluate model errors. Can be any others metrics  in 
`~metrics.metrics.SCORERS.keys()`. Furthermore if `scoring` is set to ``None``
``nmse`` is used as default value for 'neg_mean_squared_error'`.
 
""".format (params=_param_docs,
)
    
class BaseEvaluation: 
    def __init__(
        self, 
        estimator: _F,
        cv: int = 4,  
        pipeline: List[_F]= None, 
        prefit:bool=False, 
        scoring: str ='nmse',
        random_state: int=42, 
        verbose: int=0, 
        ): 
        self._logging =gofastlog().get_gofast_logger(self.__class__.__name__)
        
        self.estimator = estimator
        self.cv = cv 
        self.pipeline =pipeline
        self.prefit =prefit 
        self.scoring = scoring
        self.random_state=random_state
        self.verbose=verbose 

    def _check_callable_estimator (self, base_est ): 
        """ Check wether the estimator is callable or not.
        
        If callable use the default parameter for initialization. 
        """
        if not hasattr (base_est, 'fit'): 
            raise EstimatorError(
                f"Wrong estimator {get_estimator_name(base_est)!r}. Each"
                " estimator must have a fit method. Refer to scikit-learn"
                " https://scikit-learn.org/stable/modules/classes.html API"
                " reference to build your own estimator.") 
            
        self.estimator  = base_est 
        if callable (base_est): 
            self.estimator  = base_est () # use default initialization 
            
        return self.estimator 
        
    def fit(self, X, y, sample_weight= .75 ): 
        
        """ Quick methods used to evaluate eastimator, display the 
        error results as well as the sample model_predictions.
        
        Parameters 
        -----------
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
        
        sample_weight: float,default = .75 
            The ratio to sample X and y. The default sample 3/4 percent of the 
            data. 
            If given, will sample the `X` and `y`.  If ``None``, will sample the 
            half of the data.
            
        Returns 
        ---------
        `self` : :class:`~.BaseEvaluation` 
            :class:`~.BaseEvaluation` object. 
        """ 
        # pass when pipeline is supplied. 
        # we expect data be transform into numeric dtype 
        dtype = object if self.pipeline is not None else "numeric"
        X, y = check_X_y ( X,y, to_frame =True, dtype =dtype, 
            estimator= get_estimator_name(self.estimator), 
            )
        
        self.estimator = self._check_callable_estimator(self.estimator )
        
        self._logging.info (
            'Quick estimation using the %r estimator with config %r arguments %s.'
                %(repr(self.estimator),self.__class__.__name__, 
                inspect.getfullargspec(self.__init__)))

        sample_weight = float(
            _assert_all_types(sample_weight, int, float, 
                              objname ="Sample weight"))
        if sample_weight <= 0 or sample_weight >1: 
            raise ValueError ("Sample weight must be range between 0 and 1,"
                              f" got {sample_weight}")
            
        # sampling train data. 
        # use 75% by default of among data 
        n = int ( sample_weight * len(X)) 
        if hasattr (X, 'columns'): X = X.iloc [:n] 
        else : X=X[:n, :]
        y= y[:n]
 
        if self.pipeline is not None: 
            X =self.pipeline.fit_transform(X)
            
        if not self.prefit: 
            #for consistency 
            if self.scoring is None: 
                warnings.warn("'neg_mean_squared_error' scoring is used when"
                              " scoring parameter is ``None``.")
                self.scoring ='neg_mean_squared_error'
            self.scoring = "neg_mean_squared_error" if self.scoring in (
                None, 'nmse') else self.scoring 
            
            self.mse_, self.rmse_ , self.cv_scores_ = self._fit(
                X, y, 
                self.estimator, 
                cv_scores=True,
                scoring = self.scoring
                )
            
        return self 
    
    def _fit(self, 
        X, 
        y, 
        estimator,  
        cv_scores=True, 
        scoring ='neg_mean_squared_error' 
        ): 
        """Fit data once verified and compute the ``rmse`` scores.
        
        Parameters 
        ----------
        X: array-like of shape (n_samples, n_features) 
            training data for fitting 
        y: arraylike of shape (n_samples, ) 
            target for training 
        estimator: callable or scikit-learn estimator 
            Callable or something that has a fit methods. Can build your 
            own estimator following the API reference via 
            https://scikit-learn.org/stable/modules/classes.html 
   
        cv_scores: bool,default=True 
            compute the cross validations scores 
       
        scoring: str, default='neg_mean_squared_error' 
            metric dfor scores evaluation. 
            Type of scoring for cross validation. Please refer to  
            :doc:`~.slkearn.model_selection.cross_val_score` for further 
            details.
            
        Returns 
        ----------
        (mse, rmse, scores): Tuple 
            - mse: Mean Squared Error  
            - rmse: Root Meam Squared Error 
            - scores: Cross validation scores 

        """
        mse = rmse = None  
        def display_scores(scores): 
            """ Display scores..."""
            n=("scores:", "Means:", "RMSE scores:", "Standard Deviation:")
            p=(scores, scores.mean(), np.sqrt(scores), scores.std())
            for k, v in zip (n, p): 
                pprint(k, v )
                
        self._logging.info(
            "Fit data with a supplied pipeline or using purely estimator")

        estimator.fit(X, y)
 
        y_pred = estimator.predict(X)
        
        if self.scoring !='accuracy': # if regression task
            mse = mean_squared_error(y , y_pred)
            rmse = np.sqrt(mse)
        scores = None 
        if cv_scores: 
            scores = cross_val_score(
                estimator, X, y, cv=self.cv, scoring=self.scoring
                                     )
            if self.scoring == 'neg_mean_squared_error': 
                rmse= np.sqrt(-scores)
            else: 
                rmse= np.sqrt(scores)
            if self.verbose:
                if self.scoring =='neg_mean_squared_error': 
                    scores = -scores 
                display_scores(scores)   
                
        return mse, rmse, scores 
    
    def predict (self, X ): 
        """ Quick prediction and get the scores.
        
        Parameters 
        -----------
        X:  Ndarray ( M x N matrix where ``M=m-samples``, & ``N=n-features``)
            Test set; Denotes data that is observed at testing and 
            prediction time, used as independent variables in learning. 
            When a matrix, each sample may be represented by a feature vector, 
            or a vector of precomputed (dis)similarity with each training 
            sample. :code:`X` may also not be a matrix, and may require a 
            feature extractor or a pairwise metric to turn it into one  before 
            learning a model.
            
        Returns 
        -------
        y: array-like, shape (M, ) ``M=m-samples``, 
            test predicted target. 
        """
        self.inspect 
        
        dtype = object if self.pipeline is not None else "numeric"
        
        X = check_array(X, accept_sparse= False, 
                        input_name ='X', dtype= dtype, 
                        estimator=get_estimator_name(self.estimator), 
                        )
        
        if self.pipeline is not None: 
            X= self.pipeline.fit_transform (X) 

        return self.estimator.predict (X ) 
    
    @property 
    def inspect (self): 
        """ Inspect object whether is fitted or not"""
        msg = ( "{obj.__class__.__name__} instance is not fitted yet."
               " Call 'fit' with appropriate arguments before using"
               " this method"
               )
        
        if not hasattr (self, 'cv_scores_'): 
            raise NotFittedError(msg.format(
                obj=self)
            )
        return 1 
        
BaseEvaluation.__doc__="""\
Evaluation of dataset using a base estimator.

Quick evaluation of the data after preparing and pipeline constructions. 

Parameters 
-----------
estimator: Callable,
    estimator for trainset and label evaluating; something like a 
    class that implements a fit methods. Refer to 
    https://scikit-learn.org/stable/modules/classes.html

{params.core.cv}
    The default is ``4``.
{params.core.scoring} 

pipeline: Callable or :class:`~sklearn.pipeline.Pipeline` object 
    If `pipeline` is given , `X` is transformed accordingly, Otherwise 
    evaluation is made using purely the base estimator with the given `X`. 
    Refer to https://scikit-learn.org/stable/modules/classes.html#module-sklearn.pipeline
    for further details. 
    
kind: str, default ='GridSearchCV'
    Kind of grid search method. Could be ``GridSearchCV`` or 
    ``RandomizedSearchCV``.

prefit: bool, default=False, 
    If ``False``, does not need to compute the cross validation score once 
    again and ``True`` otherwise.
{params.core.random_state}
        
Examples 
-------- 
>>> import gofast as gf 
>>> from gofast.datasets import load_bagoue 
>>> from gofast.models import BaseEvaluation 
>>> from sklearn.ensemble import RandomForestClassifier
>>> from sklearn.model_selection import train_test_split
>>> X, y = load_bagoue (as_frame =True ) 
>>> # categorizing the labels 
>>> yc = gf.smart_label_classifier (y , values = [1, 3, 10 ], 
                                 # labels =['FR0', 'FR1', 'FR2', 'FR4'] 
                                 ) 
>>> # drop the subjective columns ['num', 'name'] 
>>> X = X.drop (columns = ['num', 'name']) 
>>> # X = gf.cleaner (X , columns = 'num name', mode='drop') 
>>> X.columns 
Index(['shape', 'type', 'geol', 'east', 'north', 'power', 'magnitude', 'sfi',
       'ohmS', 'lwi'],
      dtype='object')
>>> X =  gf.naive_imputer ( X, mode ='bi-impute') # impute data 
>>> # create a pipeline for X 
>>> pipe = gf.make_naive_pipe (X) 
>>> Xtrain, Xtest, ytrain, ytest = train_test_split(X, yc) 
>>> b = BaseEvaluation (estimator= RandomForestClassifier, 
                        scoring = 'accuracy', pipeline = pipe)
>>> b.fit(Xtrain, ytrain ) # accepts only array 
>>> b.cv_scores_ 
Out[174]: array([0.75409836, 0.72131148, 0.73333333, 0.78333333])
>>> ypred = b.predict(Xtest)
>>> scores = gf.sklearn.accuracy_score (ytest, ypred) 
0.7592592592592593
""".format (params=_param_docs,
)

def get_best_kPCA_params(
    X:NDArray | DataFrame,
    n_components: float | int =2,
    *,
    y: ArrayLike | Series=None,
    param_grid: Dict[str, Any] =None, 
    clf: _F =None,
    cv: int =7,
    **grid_kws
    )-> Dict[str, Any]: 

    from ..analysis.dimensionality import ( 
        get_component_with_most_variance, KernelPCA) 
    if n_components is None: 
        n_components= get_component_with_most_variance(X)
    if clf is None: 

        clf =Pipeline([
            ('kpca', KernelPCA(n_components=n_components)),
            ('log_reg', LogisticRegression())
            ])
    gridObj =GridSearch(base_estimator= clf,
                        grid_params= param_grid, 
                        cv=cv,
                        **grid_kws
                        ) 
    gridObj.fit(X, y)
    
    return gridObj.best_params_

get_best_kPCA_params.__doc__="""\
Select the Kernel and hyperparameters using GridSearchCV that lead 
to the best performance.

As kPCA( unsupervised learning algorithm), there is obvious performance
measure to help selecting the best kernel and hyperparameters values. 
However dimensionality reduction is often a preparation step for a 
supervised task(e.g. classification). So we can use grid search to select
the kernel and hyperparameters that lead the best performance on that 
task. By default implementation we create two steps pipeline. First reducing 
dimensionality to two dimension using kPCA, then applying the 
`LogisticRegression` for classification. AFter use Grid searchCV to find 
the best ``kernel`` and ``gamma`` value for kPCA in oder to get the best 
clasification accuracy at the end of the pipeline.

Parameters
----------
{params.core.X} 
{params.core.y}

n_components:int, 
     Number of dimension to preserve. If `n_components` is ranged between 
     0. to 1., it indicated the number of variance ratio to preserve. 
    
param_grid: list 
    list of parameters grids. For instance::
    
        param_grid=[dict(
            kpca__gamma=np.linspace(0.03, 0.05, 10),
            kpca__kernel=["rbf", "sigmoid"]
            )]
    
{params.core.clf} 
    It can also be a base estimator or a composite estimor with pipeline. For 
    instance::
    clf =Pipeline([
    ('kpca', KernelPCA(n_components=2))
    ('log_reg', LogisticRegression())
    ])
    
{params.core.cv}

grid_kws: dict, 
    Additional keywords arguments passed to Grid parameters from 
    :class:`~gofast.models.search.GridSearch`

Examples
---------
>>> from gofast.analysis.dimensionality import get_best_kPCA_params
>>> from gofast.datasets import fetch_data 
>>> X, y=fetch_data('Bagoue analysis data')
>>> param_grid=[dict(
    kpca__gamma=np.linspace(0.03, 0.05, 10),
    kpca__kernel=["rbf", "sigmoid"]
    )]
>>> clf =Pipeline([
    ('kpca', KernelPCA(n_components=2)), 
    ('log_reg', LogisticRegression())
     ])
>>> kpca_best_params =get_best_kPCA_params(
            X,y=y,scoring = 'accuracy',
            n_components= 2, clf=clf, 
            param_grid=param_grid)
>>> kpca_best_params
... {{'kpca__gamma': 0.03, 'kpca__kernel': 'rbf'}}

""".format(
    params=_param_docs,
    )
   
   