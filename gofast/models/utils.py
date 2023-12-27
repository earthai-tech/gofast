# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations 

import numpy as np 
from sklearn.covariance import ShrunkCovariance
from sklearn.model_selection import cross_val_score, GridSearchCV 
from sklearn.svm import SVC, SVR
from sklearn.utils.multiclass import type_of_target

from .._typing import (
    Tuple,
    _F, 
    ArrayLike, 
    NDArray, 
    Dict,
    )

from ..tools.validator import get_estimator_name, check_X_y 
from .._gofastlog import gofastlog
_logger = gofastlog().get_gofast_logger(__name__)

__all__= [
    'find_best_C', 
    'get_cv_mean_std_scores',  
    'get_split_best_scores', 
    'display_model_max_details',
    'display_fine_tuned_results', 
    'display_fine_tuned_results',
    'display_cv_tables', 
    'get_scorers', 
    'naive_evaluation'
  ]


def find_best_C(X, y, C_range, cv=5, scoring='accuracy', 
                scoring_reg='neg_mean_squared_error'):
    """
    Find the best C regularization parameter for an SVM, automatically determining
    whether the task is classification or regression based on the target variable.

     Mathematically, the formula can be expressed as: 

     .. math::
         \\text{Regularization Path: } C_i \\in \\{C_1, C_2, ..., C_n\\}
         \\text{For each } C_i:\\
             \\text{Evaluate } \\frac{1}{k} \\sum_{i=1}^{k} \\text{scoring}(\\text{SVM}(C_i, \\text{fold}_i))
         \\text{Select } C = \\arg\\max_{C_i} \\text{mean cross-validated score}
         
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training vectors, where n_samples is the number of samples and 
        n_features is the number of features.
    y : array-like, shape (n_samples,)
        Target values, used to determine if the task is classification or 
        regression.
    C_range : array-like
        The range of C values to explore.
    cv : int, default=5
        Number of folds in cross-validation.
    scoring : str, default='accuracy'
        A string to determine the cross-validation scoring metric 
        for classification.
    scoring_reg : str, default='neg_mean_squared_error'
        A string to determine the cross-validation scoring metric 
        for regression.

    Returns
    -------
    best_C : float
        The best C parameter found in C_range.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> X, y = iris.data, iris.target
    >>> C_range = np.logspace(-4, 4, 20)
    >>> best_C = find_best_C(X, y, C_range)
    >>> print(f"Best C value: {best_C}")
    """

    X, y = check_X_y(
        X, 
        y, 
        to_frame= True, 
        )
    task_type = type_of_target(y)
    best_score = ( 0 if task_type == 'binary' or task_type == 'multiclass'
                  else float('inf') )
    best_C = None

    for C in C_range:
        if task_type == 'binary' or task_type == 'multiclass':
            model = SVC(C=C)
            score_function = scoring
        else:  # regression
            model = SVR(C=C)
            score_function = scoring_reg

        scores = cross_val_score(model, X, y, cv=cv, scoring=score_function)
        mean_score = np.mean(scores)
        if (task_type == 'binary' or task_type == 'multiclass' and mean_score > best_score) or \
           (task_type != 'binary' and task_type != 'multiclass' and mean_score < best_score):
            best_score = mean_score
            best_C = C

    return best_C

def get_cv_mean_std_scores (
        cvres : Dict[str, ArrayLike] 
        ) -> Tuple [float]: 
    """ Retrieve the global mean and standard deviation score  from the 
    cross validation containers. 
    
    Parameters
    ------------
    cvres: dict of (str, Array-like) 
        cross validation results after training the models of number 
        of parameters equals to N. The `str` fits the each parameter stored 
        during the cross-validation while the value is stored in Numpy array.
    
    Returns 
    ---------
    ( mean_test_scores', 'std_test_scores') 
         scores on CV test data and standard deviation 
        
    """
    return  ( cvres.get('mean_test_score').mean() ,
             cvres.get('std_test_score').mean()) 

def get_split_best_scores(cvres:Dict[str, ArrayLike], 
                       split:int=0)->Dict[str, float]: 
    """ Get the best score at each split from cross-validation results
    
    Parameters 
    -----------
    cvres: dict of (str, Array-like) 
        cross validation results after training the models of number 
        of parameters equals to N. The `str` fits the each parameter stored 
        during the cross-validation while the value is stored in Numpy array.
    split: int, default=1 
        The number of split to fetch parameters. 
        The number of split must be  the number of cross-validation (cv) 
        minus one.
        
    Returns
    -------
    bests: Dict, 
        Dictionnary of the best parameters at the corresponding `split` 
        in the cross-validation. 
        
    """
    #if split ==0: split =1 
    # get the split score 
    split_score = cvres[f'split{split}_test_score'] 
    # take the max score of the split 
    max_sc = split_score.max() 
    ix_max = split_score.argmax()
    mean_score= split_score.mean()
    # get parm and mean score 
    bests ={'param': cvres['params'][ix_max], 
        'accuracy_score':cvres['mean_test_score'][ix_max], 
        'std_score':cvres['std_test_score'][ix_max],
        f"CV{split}_score": max_sc , 
        f"CV{split}_mean_score": mean_score,
        }
    return bests 

def display_model_max_details(cvres:Dict[str, ArrayLike], cv:int =4):
    """ Display the max details of each stored model from cross-validation.
    
    Parameters 
    -----------
    cvres: dict of (str, Array-like) 
        cross validation results after training the models of number 
        of parameters equals to N. The `str` fits the each parameter stored 
        during the cross-validation while the value is stored in Numpy array.
    cv: int, default=1 
        The number of KFlod during the fine-tuning models parameters. 

    """
    for k in range (cv):
        print(f'split = {k}:')
        b= get_split_best_scores(cvres, split =k)
        print( b)

    globalmeansc , globalstdsc= get_cv_mean_std_scores(cvres)
    print("Global split scores:")
    print('mean=', globalmeansc , 'std=',globalstdsc)


def display_fine_tuned_results ( cvmodels: list[_F] ): 
    """Display fined -tuning results 
    
    Parameters 
    -----------
    cvmnodels: list
        list of fined-tuned models.
    """
    bsi_bestestimators = [model.best_estimator_ for model in cvmodels ]
    mnames = ( get_estimator_name(n) for n in bsi_bestestimators)
    bsi_bestparams = [model.best_params_ for model in cvmodels]

    for nam, param , estimator in zip(mnames, bsi_bestparams, 
                                      bsi_bestestimators): 
        print("MODEL NAME =", nam)
        print('BEST PARAM =', param)
        print('BEST ESTIMATOR =', estimator)
        print()

def display_cv_tables(cvres:Dict[str, ArrayLike],  cvmodels:list[_F] ): 
    """ Display the cross-validation results from all models at each 
    k-fold. 
    
    Parameters 
    -----------
    cvres: dict of (str, Array-like) 
        cross validation results after training the models of number 
        of parameters equals to N. The `str` fits the each parameter stored 
        during the cross-validation while the value is stored in Numpy array.
    cvmnodels: list
        list of fined-tuned models.
        
    Examples 
    ---------
    >>> from gofast.datasets import fetch_data
    >>> from gofast.models import GridSearchMultiple, displayCVTables
    >>> X, y  = fetch_data ('bagoue prepared') 
    >>> gobj =GridSearchMultiple(estimators = estimators, 
                                 grid_params = grid_params ,
                                 cv =4, scoring ='accuracy', 
                                 verbose =1,  savejob=False , 
                                 kind='GridSearchCV')
    >>> gobj.fit(X, y) 
    >>> displayCVTables (cvmodels=[gobj.models.SVC] ,
                         cvres= [gobj.models.SVC.cv_results_ ])
    ... 
    """
    modelnames = (get_estimator_name(model.best_estimator_ ) 
                  for model in cvmodels  )
    for name,  mdetail, model in zip(modelnames, cvres, cvmodels): 
        print(name, ':')
        display_model_max_details(cvres=mdetail)
        
        print('BestParams: ', model.best_params_)
        try:
            print("Best scores:", model.best_score_)
        except: pass 
        finally: print()
        
        
def get_scorers (*, scorer:str=None, check_scorer:bool=False, 
                 error:str='ignore')-> Tuple[str] | bool: 
    """ Fetch the list of available metrics from scikit-learn or verify 
    whether the scorer exist in that list of metrics. 
    This is prior necessary before  the model evaluation. 
    
    :param scorer: str, 
        Must be an metrics for model evaluation. Refer to :mod:`sklearn.metrics`
    :param check_scorer:bool, default=False
        Returns bool if ``True`` whether the scorer exists in the list of 
        the metrics for the model evaluation. Note that `scorer`can not be 
        ``None`` if `check_scorer` is set to ``True``.
    :param error: str, ['raise', 'ignore']
        raise a `ValueError` if `scorer` not found in the list of metrics 
        and `check_scorer `is ``True``. 
        
    :returns: 
        scorers: bool, tuple 
            ``True`` if scorer is in the list of metrics provided that 
            ` scorer` is not ``None``, or the tuple of scikit-metrics. 
            :mod:`sklearn.metrics`
    """
    from sklearn import metrics
    try:
        scorers = tuple(metrics.SCORERS.keys()) 
    except: scorers = tuple (metrics.get_scorer_names()) 
    
    if check_scorer and scorer is None: 
        raise ValueError ("Can't check the scorer while the scorer is None."
                          " Provide the name of the scorer or get the list of"
                          " scorer by setting 'check_scorer' to 'False'")
    if scorer is not None and check_scorer: 
        scorers = scorer in scorers 
        if not scorers and error =='raise': 
            raise ValueError(
                f"Wrong scorer={scorer!r}. Supports only scorers:"
                f" {tuple(metrics.SCORERS.keys())}")
            
    return scorers 
              
def naive_evaluation(
        clf: _F,
        X:NDArray,
        y:ArrayLike,
        cv:int =7,
        scoring:str  ='accuracy', 
        display: str ='off', 
        **kws
        ): 
    scores = cross_val_score(clf , X, y, cv = cv, scoring=scoring, **kws)
                         
    if display is True or display =='on':
        print('clf=:', clf.__class__.__name__)
        print('scores=:', scores )
        print('scores.mean=:', scores.mean())
    
    return scores , scores.mean()

naive_evaluation.__doc__="""\
Quick scores evaluation using cross validation. 

Parameters
----------
clf: callable 
    Classifer for testing default data. 
X: ndarray
    trainset data 
    
y: array_like 
    label data 
cv: int 
    KFold for data validation.
    
scoring: str 
    type of error visualization. 
    
display: str or bool, 
    show the show on the stdout
kws: dict, 
    Additional keywords arguments passed to 
    :func:`gofast.exlib.slearn.cross_val_score`.
Returns 
---------
scores, mean_core: array_like, float 
    scaore after evaluation and mean of the score
    
Examples 
---------
>>> import gofast as gf 
>>> from gofast.models.search import naive_evaluation
>>> X,  y = gf.fetch_data ('bagoue data prepared') 
>>> clf = gf.sklearn.DecisionTreeClassifier() 
>>> naive_evaluation(clf, X, y , cv =4 , display ='on' )
clf=: DecisionTreeClassifier
scores=: [0.6279 0.7674 0.7093 0.593 ]
scores.mean=: 0.6744186046511629
Out[57]: (array([0.6279, 0.7674, 0.7093, 0.593 ]), 0.6744186046511629)
"""

def shrink_covariance_cv_score(X, skrink_space =( -2, 0, 30 )):
    shrinkages = np.logspace(*skrink_space)  # Fit the models
    cv = GridSearchCV(ShrunkCovariance(), {'shrinkage': shrinkages})
    return np.mean(cross_val_score(cv.fit(X).best_estimator_, X))

shrink_covariance_cv_score.__doc__="""\
shrunk the covariance scores from validating X using 
GridSearchCV.
 
Parameters 
-----------
X : array_like, pandas.DataFrame 
    Input data where rows represent samples and 
    columns represent features.

Returns
-----------
score: score of covariance estimator (best ) with shrinkage

"""