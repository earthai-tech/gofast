#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Optimizing searches helper functions
"""

import joblib
import concurrent 
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import numpy as np 
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import ParameterGrid

from ..api.box import KeyBox 
from ..api.types import Any, Dict, List,Union, Tuple, Optional, ArrayLike
from ..api.types import _F, Array1D, NDArray, Callable 
from ..api.summary import ModelSummary
from ..tools.coreutils import ellipsis2false , smart_format
from ..tools.validator import get_estimator_name , check_X_y 
from ..tools.validator import filter_valid_kwargs
from .utils import get_optimizer_method, align_estimators_with_params
from .utils import params_combinations # noqa

__all__=[
    "Optimizer", 
    "optimize_search", 
    "optimize_search2", 
    "parallelize_search", 
    "optimize_hyperparams",
    ]

class Optimizer:
    """
    Optimizer class for hyperparameter optimization of multiple estimators.

    This class facilitates the process of hyperparameter optimization for 
    multiple machine learning estimators using various optimization techniques 
    such as Grid Search Cross Validation (GSCV). It allows the user to specify 
    the estimators, parameter grids, and other optimization settings, and 
    provides functionalities to perform the optimization and save the results.

    Parameters
    ----------
    estimators : dict or list
        A dictionary of estimator names to estimator instances, or a list of 
        estimator instances. Each estimator is an instance of a model to be 
        optimized. If a dictionary is provided, the keys are used as the names 
        of the estimators.

    param_grids : dict or list
        A dictionary of estimator names to parameter grids, or a list of 
        parameter grids. Each parameter grid is a dictionary where the keys 
        are parameter names and the values are lists of parameter settings to 
        try for that parameter.

    optimizer : str, default='GSCV'
        The optimization technique to apply. 'GSCV' refers to Grid Search 
        Cross Validation. Supported optimizers include:
        
        - 'GSCV', 'GridSearchCV' for Grid Search Cross Validation.
        - 'RSCV', 'RandomizedSearchCV' for Randomized Search Cross Validation.
        - 'BSCV', 'BayesSearchCV' for Bayesian Optimization.
        - 'ASCV', 'AnnealingSearchCV' for Simulated Annealing-based Search.
        - 'SWCV', 'PSOSCV', 'SwarmSearchCV' for Particle Swarm Optimization.
        - 'SSCV', 'SequentialSearchCV' for Sequential Model-Based Optimization.
        - 'ESCV', 'EvolutionarySearchCV' for Evolutionary Algorithms-based Search.
        - 'GBSCV', 'GradientSearchCV' for Gradient-Based Optimization.
        - 'GESCV', 'GeneticSearchCV' for Genetic Algorithms-based Search.
    
    save_results : bool, default=False
        If True, the optimization results (best parameters and scores) for 
        each estimator are saved to a file.

    n_jobs : int, default=-1
        The number of jobs to run in parallel for the optimization process. 
        `-1` means using all processors.

    scoring : str, callable, list/tuple, or dict, default=None
        A string (see model evaluation documentation), a callable (see 
        defining your scoring strategy from metric functions), a list/tuple 
        of strings or callables, or a dictionary mapping scorer names to 
        strings/callables.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy. Possible inputs 
        for `cv` are:
          - None, to use the default 5-fold cross-validation,
          - int, to specify the number of folds in a (Stratified)KFold,
          - CV splitter,
          - An iterable yielding (train, test) splits as arrays of indices.

    search_kwargs : dict, optional
        Additional keyword arguments to pass to the optimizer function.

    Examples
    --------
    >>> from sklearn.svm import SVC
    >>> from sklearn.linear_model import SGDClassifier
    >>> from gofast.models.optimize import Optimizer
    >>> estimators = {'SVC': SVC(), 'SGDClassifier': SGDClassifier()}
    >>> param_grids = {'SVC': {'C': [1, 10], 'kernel': ['linear', 'rbf']}, 
    ...                'SGDClassifier': {'max_iter': [50, 100], 'alpha': [0.0001, 0.001]}}
    >>> optimizer = Optimizer(estimators, param_grids, optimizer='GSCV', n_jobs=1)
    >>> results = optimizer.fit(X_train, y_train)
    >>> print(results)
                      Optimized Results                       
    ==============================================================
    |                            SVC                             |
    --------------------------------------------------------------
                            Model Results                         
    ==============================================================
    Best estimator       : SVC
    Best parameters      : {'C': 1, 'kernel': 'rbf'}
    Best score           : 0.9625
    nCV                  : 5
    Params combinations  : 4
    ==============================================================
    
                       Tuning Results (*=score)                   
    ==============================================================
            Params   Mean*  Std.* Overall Mean* Overall Std.* Rank
    --------------------------------------------------------------
    0   (1, linear) 0.9375 0.0395        0.9375        0.0395    3
    1      (1, rbf) 0.9625 0.0500        0.9625        0.0500    1
    2  (10, linear) 0.9250 0.0468        0.9250        0.0468    4
    3     (10, rbf) 0.9625 0.0306        0.9625        0.0306    1
    ==============================================================
    
    Notes
    -----
    The learning rate (`eta0`) and the number of iterations (`max_iter`) are 
    crucial hyperparameters that impact the training process. Careful tuning 
    of these hyperparameters is necessary for achieving optimal results.

    The function uses joblib for parallel processing. Ensure that the 
    objects passed to the function are pickleable.

    The progress bars are displayed using tqdm to show the progress of each 
    estimator's optimization process.

    The optimization technique can be extended to include other methods by 
    implementing additional optimizers and specifying them in the `optimizer` 
    parameter.

    See Also
    --------
    sklearn.model_selection.GridSearchCV : Exhaustive search over specified 
        parameter values for an estimator.
    sklearn.model_selection.RandomizedSearchCV : Randomized search on hyper 
        parameters.

    References
    ----------
    .. [1] Bergstra, J., & Bengio, Y. (2012). Random Search for Hyper-Parameter 
           Optimization. Journal of Machine Learning Research, 13, 281-305.
    .. [2] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., 
           Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine 
           Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.
    """
    def __init__(
        self, 
        estimators, 
        param_grids, 
        optimizer='GSCV', 
        save_results=False,
        n_jobs=-1, 
        scoring=None, 
        cv=None, 
        **search_kwargs
    ):
        self.estimators = estimators
        self.param_grids = param_grids
        self.optimizer = optimizer
        self.save_results = save_results
        self.n_jobs = n_jobs
        self.scoring = scoring
        self.cv = cv
        self.search_kwargs = search_kwargs
    
    def fit(self, X, y):
        """
        Perform hyperparameter optimization for a list of estimators.
    
        This method applies a specified optimization technique (e.g., 
        Grid Search) to a range of estimators and their associated 
        parameter grids. It allows for the simultaneous tuning of multiple 
        models, facilitating the selection of the best model and parameters 
        based on the provided data.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, with `n_samples` as the number of samples and 
            `n_features` as the number of features. It supports both dense and 
            sparse matrix formats.
        
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Target values corresponding to `X`.
    
        Returns
        -------
        result_dict : ModelSummary
            A ModelSummary object containing the optimization results for each 
            estimator. The object includes information on the best estimator, 
            best parameters, best score, and cross-validation results for each 
            model.
    
        Notes
        -----
        The function leverages parallel processing using joblib to expedite the 
        hyperparameter search process. The progress of the optimization is 
        displayed using tqdm progress bars.
    
        The hyperparameter optimization is flexible and can be extended to use 
        different optimization techniques by specifying the `optimizer` parameter.
    
        Examples
        --------
        >>> from sklearn.svm import SVC
        >>> from sklearn.linear_model import SGDClassifier
        >>> from gofast.models.optimize import Optimizer
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import train_test_split
        >>> X, y = make_classification(n_samples=100, n_features=7, random_state=42)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        >>> estimators = {'SVC': SVC(), 'SGDClassifier': SGDClassifier()}
        >>> param_grids = {'SVC': {'C': [1, 10], 'kernel': ['linear', 'rbf']}, 
        ...                'SGDClassifier': {'max_iter': [50, 100], 'alpha': [0.0001, 0.001]}}
        >>> optimizer = Optimizer(estimators, param_grids, optimizer='GSCV', n_jobs=1)
        >>> results = optimizer.fit(X_train, y_train)
        >>> print(results)
                          Optimized Results                       
        ==============================================================
        |                            SVC                             |
        --------------------------------------------------------------
                                Model Results                         
        ==============================================================
        Best estimator       : SVC
        Best parameters      : {'C': 1, 'kernel': 'rbf'}
        Best score           : 0.9625
        nCV                  : 5
        Params combinations  : 4
        ==============================================================
        
                           Tuning Results (*=score)                   
        ==============================================================
                Params   Mean*  Std.* Overall Mean* Overall Std.* Rank
        --------------------------------------------------------------
        0   (1, linear) 0.9375 0.0395        0.9375        0.0395    3
        1      (1, rbf) 0.9625 0.0500        0.9625        0.0500    1
        2  (10, linear) 0.9250 0.0468        0.9250        0.0468    4
        3     (10, rbf) 0.9625 0.0306        0.9625        0.0306    1
        ==============================================================
 
        """
        X, y = check_X_y(X, y, accept_sparse=True, accept_large_sparse=True)
        estimators, param_grids = _validate_parameters(
            self.param_grids, self.estimators)
        max_length = max([len(str(estimator)) for estimator in estimators])
    
        results = Parallel(n_jobs=self.n_jobs)(delayed(_perform_search)(
            name, estimators[i], param_grids[i], 
            self.optimizer, X, y, self.scoring, self.cv, self.search_kwargs,
            f"Optimizing {get_estimator_name(name):<{max_length}}") for i, 
            name in enumerate(estimators))
    
        result_dict = {get_estimator_name(name): {
            'best_estimator': best_est, 'best_params': best_params, 
            'best_score': best_sc, 'cv_results': cv_res
            } for name, best_est, best_params, best_sc, cv_res in results
        }
    
        if self.save_results:
            joblib.dump(result_dict, "optimization_results.joblib")
    
        return ModelSummary(
            descriptor="Optimizer", **result_dict).summary(result_dict)

def optimize_search(
    estimators: Dict[str, BaseEstimator], 
    param_grids: Dict[str, Any], 
    X: Union [NDArray, ArrayLike], 
    y: Union [Array1D, ArrayLike], 
    optimizer: str = 'RSCV', 
    save_results: bool = False, 
    n_jobs: int = -1, 
    **search_kwargs: Any
) -> Dict[str, Dict[str, Any]]:
    """
    Perform hyperparameter optimization for multiple estimators in parallel.
    
    Function supports Grid Search, Randomized Search, and Bayesian Search. This 
    parallel processing can significantly expedite the hyperparameter tuning process.

    Parameters
    ----------
    estimators : dict
        A dictionary where keys are estimator names and values are estimator instances.
    param_grids : dict
        A dictionary where keys are estimator names (matching those in 'estimators') 
        and values are parameter grids.
    X : ndarray or DataFrame
        Input features for the model.
    y : ndarray or Series
        Target variable for the model.
    optimizer : str, optional
        Type of search to perform. Default is 'RSCV'.
    save_results : bool, optional
        If True, saves the results of the search to a joblib file. Default is False.
    n_jobs : int, optional
        Number of jobs to run in parallel. Default is -1 (all available processors).
    **search_kwargs : dict
        Additional keyword arguments to pass to the search constructor.

    Returns
    -------
    dict
        A dictionary with keys as estimator names and values as dictionaries 
        containing 'best_estimator', 'best_params', and 'cv_results' for each estimator.

    Raises
    ------
    ValueError
        If the keys in 'estimators' and 'param_grids' do not match.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.svm import SVC
    >>> from sklearn.datasets import load_iris 
    >>> from gofast.models.optimize import optimize_search
    >>> X, y = load_iris(return_X_y=True)
    >>> estimators = {'rf': RandomForestClassifier(), 'svc': SVC()}
    >>> param_grids = {'rf': {'n_estimators': [10, 100], 'max_depth': [None, 10]},
    ...                'svc': {'C': [1, 10], 'kernel': ['linear', 'rbf']}}
    >>> results = optimize_search(estimators, param_grids, X, y, optimizer='RSCV',
    ...                          save_results=False, n_jobs=4)
    """

    if set(estimators.keys()) != set(param_grids.keys()):
        raise ValueError("The keys in 'estimators' and 'param_grids' must match.")

    optimizer_class = get_optimizer_method(optimizer)

    def perform_search(estimator_name, estimator, param_grid):
        search = optimizer_class(estimator, param_grid, n_jobs=n_jobs, **search_kwargs)
        search.fit(X, y)
        return (estimator_name, search.best_estimator_, search.best_params_,
                search.best_score_, search.cv_results_)
    # Parallel execution of the search for each estimator
    results = Parallel(n_jobs=n_jobs)(delayed(perform_search)(
        name, est, param_grids[name])
        for name, est in tqdm(estimators.items(), desc="Optimizing Estimators",
                              ncols=100, ascii=True))

    result_dict = {name: {'best_estimator_': best_est, 
                          'best_params_': best_params,
                          'best_score_': best_score, 
                          'cv_results_': cv_res, 
                          }
                  for name, best_est, best_params, best_score,  cv_res in results}

    # Optionally save results to a joblib file
    if save_results:
        filename = "optimization_results.joblib"
        joblib.dump(result_dict, filename)
        print(f"Results saved to {filename}")
        
    summary= ModelSummary(**result_dict)
    summary.summary(result_dict)
    return summary

def optimize_search2(
    estimators: Dict[str, BaseEstimator], 
    param_grids: Dict[str, Any],
    X: ArrayLike, 
    y: ArrayLike, 
    optimizer: str='GSCV',
    scoring: str | Callable=None, 
    cv:int|Callable =None, 
    save_results: bool=False, 
    n_jobs: int=-1, 
    **search_kwargs: Any 
    ):
    """
    Perform hyperparameter optimization for a list of estimators.

    This function applies a specified optimization technique (e.g., 
    Grid Search) to a range of estimators and their associated 
    parameter grids. It allows for the simultaneous tuning of multiple 
    models, facilitating the selection of the best model and parameters 
    based on the provided data.

    Parameters
    ----------
    estimators : list of estimator objects or tuples (str, estimator)
        A list of estimators or (name, estimator) tuples. Each estimator 
        is an instance of a model to be optimized. If a tuple is provided, 
        the first element is used as the name of the estimator.

    param_grids : list of dicts
        A list of dictionaries, where each dictionary contains the 
        parameters to be searched for the corresponding estimator in 
        `estimators`. Each key in the dictionary is a parameter name, and 
        the associated value is a list of values to try for that parameter.

    X : array-like of shape (n_samples, n_features)
        Training data, with `n_samples` as the number of samples and 
        `n_features` as the number of features.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Target values corresponding to `X`.

    optimizer : str, default='GSCV'
        The optimization technique to apply. 'GSCV' refers to Grid Search 
        Cross Validation. Additional optimizers can be implemented and 
        specified here.

    save_results : bool, default=False
        If True, the optimization results (best parameters and scores) for 
        each estimator are saved to a file.

    n_jobs : int, default=-1
        The number of jobs to run in parallel for `optimizer`. `-1` means 
        using all processors.

    search_kwargs : dict, optional
        Additional keyword arguments to pass to the optimizer function.

    Returns
    -------
    results : dict
        A dictionary containing the optimization results for each estimator. 
        The keys are the estimator names, and the values are the results 
        returned by the optimizer for that estimator.

    Examples
    --------
    >>> from sklearn.svm import SVC
    >>> from sklearn.linear_model import SGDClassifier
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from gofast.models.optimize import optimize_search2
    >>> X, y = make_classification(n_samples=100, n_features=7, 
                                   random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size=0.2, 
                                                            random_state=42)
    >>> estimators = [SVC(), SGDClassifier()]
    >>> param_grids = [{'C': [1, 10], 'kernel': ['linear', 'rbf']}, 
                       {'max_iter': [50, 100], 'alpha': [0.0001, 0.001]}]
    >>> result = optimize_search2(estimators, param_grids, X_train, y_train, 
                          n_jobs=1, n_iter=10)
    >>> print(result)
                      Optimized Results                       
    ==============================================================
    |                            SVC                             |
    --------------------------------------------------------------
                            Model Results                         
    ==============================================================
    Best estimator       : SVC
    Best parameters      : {'C': 1, 'kernel': 'rbf'}
    Best score           : 0.9625
    nCV                  : 5
    Params combinations  : 4
    ==============================================================
    
                       Tuning Results (*=score)                   
    ==============================================================
            Params   Mean*  Std.* Overall Mean* Overall Std.* Rank
    --------------------------------------------------------------
    0   (1, linear) 0.9375 0.0395        0.9375        0.0395    3
    1      (1, rbf) 0.9625 0.0500        0.9625        0.0500    1
    2  (10, linear) 0.9250 0.0468        0.9250        0.0468    4
    3     (10, rbf) 0.9625 0.0306        0.9625        0.0306    1
    ==============================================================
    
    
    ==============================================================
    |                       SGDClassifier                        |
    --------------------------------------------------------------
                            Model Results                         
    ==============================================================
    Best estimator       : SGDClassifier
    Best parameters      : {'alpha': 0.0001, 'max_iter': 100}
    Best score           : 0.9750
    nCV                  : 5
    Params combinations  : 4
    ==============================================================
    
                       Tuning Results (*=score)                   
    ==============================================================
            Params   Mean*  Std.* Overall Mean* Overall Std.* Rank
    --------------------------------------------------------------
    0 (0.0001, 100) 0.9750 0.0306        0.9750        0.0306    1
    1 (0.0001, 300) 0.9625 0.0500        0.9625        0.0500    3
    2  (0.001, 100) 0.9750 0.0306        0.9750        0.0306    1
    3  (0.001, 300) 0.9500 0.0468        0.9500        0.0468    4
    ==============================================================

    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.model_selection import train_test_split
    >>> from gofast.datasets import make_classification 
    >>> from gofast.estimators.adaline import AdalineStochasticClassifier 
    >>> X, y = make_classification (n_samples =100, n_features=7, return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> estimators = [RandomForestClassifier()]
    >>> param_grids = [{'n_estimators': [100, 200], 'max_depth': [10, 20]}]
    >>> result_dict=optimize_search2(estimators, param_grids, X_train, y_train)
    
    Notes
    -----
    - The function uses joblib for parallel processing. Ensure that the 
      objects passed to the function are pickleable.
    - The progress bars are displayed using tqdm to show the progress of 
      each estimator's optimization process.
    - The optimization technique can be extended to include other methods 
      by implementing additional optimizers and specifying them in the 
      `optimizer` parameter.

    """
    def make_estimator_name (estimator): 
        return get_estimator_name(estimator ) if not isinstance ( 
            estimator, str) else estimator
    
    estimators, param_grids = _validate_parameters(param_grids, estimators,)
    max_length = max([len(str(estimator)) for estimator in estimators])
    
    # try: 
    results = Parallel(n_jobs=n_jobs)(delayed(_perform_search)(
        name, estimators[i], param_grids[i], optimizer, X, y, 
        scoring, cv, search_kwargs,
        f"Optimizing {make_estimator_name(name):<{max_length}}")
        for i, name in enumerate(estimators))
    # except: 
    #     result_dict= _optimize_search2(
    #         X, y, param_grids=param_grids, estimators=estimators, 
    #          **search_kwargs)
    # else: 
    result_dict = {make_estimator_name(name): {
        'best_estimator': best_est, 'best_params': best_params, 
        'best_score': best_sc, 'cv_results': cv_res
        }
       for name, best_est, best_params, best_sc, cv_res in results
    }
    
    if save_results:
        joblib.dump(result_dict, "optimization_results.joblib")
    
    return ModelSummary(**result_dict).summary(result_dict)

def optimize_hyperparams(
    estimator: BaseEstimator, 
    param_grid: Dict[str, Any], 
    X: ArrayLike, y: ArrayLike, 
    cv: int=5, 
    scoring: str | _F=None, 
    optimizer: str= 'GridSearchCV', 
    n_jobs: int=-1, 
    savejob: bool= ..., 
    savefile: str=None, 
    **kws 
    ):
    """
    Optimize hyperparameters for a given estimator.

    Parameters
    ----------
    estimator : estimator object
        The object to use to fit the data.
    param_grid : dict or list of dictionaries
        Dictionary with parameters names (`str`) as keys and lists of parameter 
        settings to try as values.
    X : array-like of shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and n_features 
        is the number of features.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Target relative to X for classification or regression.
    cv : int, default=5
        Determines the cross-validation splitting strategy.
    scoring : str or callable, default=None
        A str (see model evaluation documentation) or a scorer callable 
        object / function with signature scorer(estimator, X, y).
    n_jobs : int, default=-1
        Number of jobs to run in parallel. `-1` means using all processors.
    savejob: bool, default=False, 
        Save model into a binary files. 
    savefile: str, optional 
       model binary file name. If ``None``, the estimator name is 
       used instead.
       
    Returns
    -------
    best_estimator : estimator object
        Estimator that was chosen by the search, i.e. estimator 
        which gave highest score.
    best_params : dict
        Parameter setting that gave the best results on the hold 
        out data.
    cv_results: dict, 
        Cross-validation results  
        
    """
    savejob, = ellipsis2false(savejob) 
    optimizer_class = get_optimizer_method(optimizer) 
    optimizer = optimizer_class (estimator, param_grid, cv=cv, scoring=scoring,
                                 n_jobs=n_jobs, **kws)
    optimizer.fit(X, y)

    results_dict= {"best_estimator_":optimizer.best_estimator_ , 
                   "best_params_":optimizer.best_params_  , 
                   "cv_results_": optimizer.cv_results_}
    # try to save file 
    if savejob: 
        savefile = savefile or get_estimator_name(estimator)
        # remove joblib if extension is appended.
        savefile= str(savefile).replace ('.joblib', '')
        joblib.dump ( results_dict,filename = f'{savefile}.joblib' )
    
    summary=ModelSummary(**results_dict).summary(results_dict)
    return summary

def parallelize_search(
    estimators, 
    param_grids, 
    X, y, 
    file_prefix="models", 
    cv:int=5, 
    scoring:str=None, 
    optimizer="RandomizedSearchCV", 
    n_jobs=-1, 
    pack_models: bool=...,
    **kws
   ):
    """
    Parallelizes the hyperparameter optimization for multiple estimators.

    Parameters
    ----------
    estimators : list of estimator objects
        List of estimators for which to optimize hyperparameters.
    param_grids : list of dicts
        List of parameter grids to search for each estimator.
    X : array-like of shape (n_samples, n_features)
        Training data.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Target data.
    file_prefix : str, default="estimator"
        Prefix for the filename to save the estimators.
    cv : int, default=5
        Number of folds in cross-validation.
    scoring : str or callable, default=None
        Scoring method to use.
    n_jobs : int, default=-1
        Number of jobs to run in parallel for GridSearchCV.
    pack_models: bool, default=False, 
       Aggregate multiples models results and save it into a single 
       binary file. 
       
    Returns
    -------
    o: gofast.api.summary.ModelSummary 
        The function saves the best estimator and parameters, and 
        cv results for each input estimator to disk
        returns object where `best_params_`, `best_estimators_` and `cv_results_`
        can be retrieved as an object. 
        Visualization is possible using Print since ModelSummary operated 
        similarly as BunchObject :class:`gofast.api.box.KeyBox`

    Note 
    -----
    When parallelizing tasks that are already CPU-intensive 
    (like GridSearchCV with n_jobs=-1), it's important to manage the 
    overall CPU load to avoid overloading your system. Adjust the n_jobs 
    parameter based on your system's capabilities
    
    Examples 
    ---------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.svm import SVC
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from gofast.models.optimize import parallelize_search
    >>> X, y = load_iris(return_X_y=True)
    >>> estimators = [SVC(), DecisionTreeClassifier()]
    >>> param_grids = [{'C': [1, 10], 'kernel': ['linear', 'rbf']}, 
                       {'max_depth': [3, 5, None], 'criterion': ['gini', 'entropy']}
                       ]
    >>> o= parallelize_search(estimators, param_grids, X, y)
    >>> o.SVC.best_estimator_
    Out[294]: SVC(C=1, kernel='linear')
    >>> o.DecisionTreeClassifier.best_params_
    Out[296]: {'max_depth': None, 'criterion': 'gini'}
    """
    pack_models, = ellipsis2false( pack_models )

    o={}; pack ={} # save models in dict/object.
    with ThreadPoolExecutor() as executor:
        futures = []
        for idx, (estimator, param_grid) in enumerate(zip(estimators, param_grids)):
            futures.append(executor.submit(
                optimize_hyperparams, estimator, 
                param_grid, X, y, cv, scoring, optimizer, 
                n_jobs, **kws))

        for idx, (future, estimator)in enumerate (zip (
                tqdm(concurrent.futures.as_completed(futures),
                           total=len(futures), desc="Optimizing Estimators", 
                           ncols=100, ascii=True,
                           ), estimators)
                                                 ):
            est_name = get_estimator_name(estimator)
            summary= future.result()
            best_estimator = summary.best_estimator_ 
            best_params = summary.best_params_ 
            cv_results= summary.cv_results_
            
            # save model results into a large object that can be return 
            # as an object . 
            pack [f"{est_name}"]= {"best_params_": best_params, 
                                   "best_estimator_": best_estimator, 
                                   "cv_results_": cv_results
                                   }
            o[f"{est_name}"]= KeyBox ( ** pack [f"{est_name}"])
            
            if  not pack_models: 
                # save all model individualy and append index 
                # to differential wether muliple 
                file_name = f"{est_name}_{idx}.joblib"
                joblib.dump((best_estimator, best_params), file_name)
                
        if pack_models: 
            joblib.dump(pack , filename= f"{file_prefix}.joblib")
            
    summary= ModelSummary(**o)
    summary.summary(o)
    return summary

def _get_optimizer_method(optimizer: str) -> Any:
    """
    Returns the correct optimizer class based on the input optimizer string,
    ignoring case sensitivity.

    Parameters
    ----------
    optimizer : str
        The name or abbreviation of the optimizer.

    Returns
    -------
    Any
        The optimizer class corresponding to the provided optimizer string.

    Raises
    ------
    ValueError
        If no matching optimizer is found.

    Examples
    --------
    >>> optimizer_class = get_optimizer_method('RSCV')
    >>> print(optimizer_class)
    <class 'sklearn.model_selection.RandomizedSearchCV'>
    """

    # Mapping of optimizer names to their possible abbreviations and variations
    opt_dict = { 
        'RandomizedSearchCV': ['RSCV', 'RandomizedSearchCV'], 
        'GridSearchCV': ['GSCV', 'GridSearchCV'], 
        'BayesSearchCV': ['BSCV', 'BayesSearchCV'], 
        'AnnealingSearchCV': ['ASCV',"AnnealingSearchCV" ], 
        'SwarmSearchCV': ['SWCV', 'PSOSCV', 'SwarmSearchCV'], 
        'SequentialSearchCV': ['SSCV', 'SMBOSearchCV'], 
        'EvolutionarySearchCV': ['ESCV', 'EvolutionarySearchCV'], 
        'GradientSearchCV':['GBSCV', 'GradientBasedSearchCV'], 
        'GeneticSearchCV': ['GASCV', 'GeneticSearchCV']
    }

    # Mapping of optimizer names to their respective classes
    optimizer_dict = {
        'GridSearchCV': GridSearchCV,
        'RandomizedSearchCV': RandomizedSearchCV,
    }
    try: from skopt import BayesSearchCV
    except: pass 
    else : optimizer_dict["BayesSearchCV"]= BayesSearchCV
    # Normalize the input optimizer string to ignore case
    optimizer_lower = optimizer.lower()

    # Search for the corresponding optimizer class
    for key, aliases in opt_dict.items():
        if optimizer_lower in [alias.lower() for alias in aliases]:
            return optimizer_dict[key]

    # Raise an error if no matching optimizer is found
    raise ValueError(f"Invalid 'optimizer' parameter '{optimizer}'."
                     f" Choose from {smart_format(opt_dict.keys(), 'or')}.") 

def _process_estimators_and_params(
    param_grids: List[Union[Dict[str, List[Any]], Tuple[BaseEstimator, Dict[str, List[Any]]]]],
    estimators: Optional[List[BaseEstimator]] = None
) -> Tuple[List[BaseEstimator], List[Dict[str, List[Any]]]]:
    """
    Process and separate estimators and their corresponding parameter grids.

    This function handles two cases:
    1. `param_grids` contains tuples of estimators and their parameter grids.
    2. `param_grids` only contains parameter grids, and `estimators` are 
    provided separately.

    Parameters
    ----------
    param_grids : List[Union[Dict[str, List[Any]],
                             Tuple[BaseEstimator, Dict[str, List[Any]]]]]
        A list containing either parameter grids or tuples of estimators and 
        their parameter grids.

    estimators : List[BaseEstimator], optional
        A list of estimator objects. Required if `param_grids` only contains 
        parameter grids.

    Returns
    -------
    Tuple[List[BaseEstimator], List[Dict[str, List[Any]]]]
        Two lists: the first containing the estimators, and the second containing
        the corresponding parameter grids.

    Raises
    ------
    ValueError
        If `param_grids` does not contain estimators and `estimators` is None.

    Examples
    --------
    >>> from sklearn.svm import SVC
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> param_grids = [
    ...     (SVC(), {'C': [1, 10, 100], 'kernel': ['linear', 'rbf']}),
    ...     (RandomForestClassifier(), {'n_estimators': [10, 50, 100],
                                        'max_depth': [5, 10, None]})
    ... ]
    >>> estimators, grids = process_estimators_and_params(param_grids)
    >>> print(estimators)
    [SVC(), RandomForestClassifier()]
    >>> print(grids)
    [{'C': [1, 10, 100], 'kernel': ['linear', 'rbf']}, {'n_estimators': [10, 50, 100],
                                                        'max_depth': [5, 10, None]}]
    """
    
    if all(isinstance(grid, (tuple, list)) for grid in param_grids):
        # Extract estimators and parameter grids from tuples
        estimators, param_grids = zip(*param_grids)
        return list(estimators), list(param_grids)
    elif estimators is not None:
        # Use provided estimators and param_grids
        return estimators, param_grids
    else:
        raise ValueError("Estimators are missing. They must be provided either "
                         "in param_grids or as a separate list.")

def _optimize_search2(
    X: ArrayLike,
    y: ArrayLike,
    param_grids: List[Union[Dict[str, List[Any]], Tuple[BaseEstimator, Dict[str, List[Any]]]]],
    estimators: Optional[List[BaseEstimator]] = None,
    optimizer: str = 'GridSearchCV',
    n_jobs: int = -1,
    **search_params: Any
) -> Dict[str, Any]:
    """
    Perform hyperparameter optimization across multiple estimators using a 
    specified optimizer.

    Parameters
    ----------
    X : np.ndarray
        Training vectors of shape (n_samples, n_features).
    y : np.ndarray
        Target values (labels) of shape (n_samples,).
    param_grids : List[Union[Dict, Tuple[BaseEstimator, Dict]]]
        Parameter grids to explore for each estimator. Can include tuples of
        estimators and their respective parameter grids.
    estimators : Optional[List[BaseEstimator]], default=None
        List of estimator objects, required if param_grids does not include them.
    optimizer : str, default='GridSearchCV'
        Optimization technique to apply ('GridSearchCV', 'RandomizedSearchCV', 'BayesSearchCV').
    n_jobs : int, default=-1
        Number of jobs to run in parallel. `-1` means using all processors.
    search_params : Any
        Additional parameters to pass to the optimizer.

    Returns
    -------
    Dict[str, Any]
        Optimization results for each estimator, with estimator names as keys.

    Raises
    ------
    ValueError
        If invalid optimizer is specified.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = np.random.rand(100, 10), np.random.randint(0, 2, 100)
    >>> param_grids = [{'n_estimators': [100, 200], 'max_depth': [10, 20]}]
    >>> results = optimize_search(X, y, param_grids, [RandomForestClassifier()],
                                  optimizer='RandomizedSearchCV')
    """
    estimators, param_grids = _process_estimators_and_params(param_grids, estimators)
    OptimizeMethod= get_optimizer_method(optimizer )
    def calculate_grid_length(param_grid):
        # n_combinations = len(list(itertools.product(*param_grid.values())))
        return np.prod([len(v) for v in param_grid.values()])
        # return n_combinations 
    def run_search(estimator, param_grid, X, y, name):
        total_combinations = calculate_grid_length(param_grid)
        with tqdm(total=total_combinations, desc=f"{name:<30}", unit= "it", #"cfg",
                  leave=True, ncols =100,
                  ) as pbar:
            search = OptimizeMethod(estimator, param_grid, n_jobs=1, **search_params)
            search.fit(X, y)
            pbar.update(total_combinations)
            return search

    results = Parallel(n_jobs=n_jobs)(
        delayed(run_search)(est, grid, X, y, est.__class__.__name__)
        for est, grid in zip(estimators, param_grids)
    )
    results_dict = {est.__class__.__name__: {'best_estimator_': result.best_estimator_, 
                          'best_params_': result.best_params_,
                          'best_score_': result.best_score_, 
                          'cv_results_': result.cv_results_, 
                          }
                  for est, result in zip(estimators, results)}
    
    return results_dict


def _validate_parameters(param_grids, estimators):
    """
    Align estimators with their corresponding parameter grids.

    This function ensures that the estimators and their parameter grids are 
    correctly aligned and compatible for optimization.

    Parameters
    ----------
    param_grids : dict
        Dictionary of parameter grids for each estimator.
    
    estimators : dict
        Dictionary of estimator names to estimator instances.

    Returns
    -------
    tuple
        A tuple containing aligned lists of estimators and parameter grids.
    """
    return align_estimators_with_params(param_grids, estimators)


def _initialize_search(optimizer, estimator, param_grid, scoring, cv, **search_kwargs):
    """
    Initialize the optimizer search method.

    This function initializes the specified optimizer with the given estimator 
    and parameter grid, filtering valid keyword arguments for the optimizer class.

    Parameters
    ----------
    optimizer : str
        The optimization technique to apply (e.g., 'GridSearchCV').
    
    estimator : estimator instance
        The estimator instance to be optimized.
    
    param_grid : dict
        Dictionary of parameter grids for the estimator.
    
    scoring : str or callable
        Scoring method to evaluate the predictions on the test set.
    
    cv : int, cross-validation generator or iterable
        Determines the cross-validation splitting strategy.

    search_kwargs : dict
        Additional keyword arguments for the optimizer.

    Returns
    -------
    search_instance : optimizer class instance
        An instance of the optimizer class initialized with the provided parameters.
    """
    optimizer_class = get_optimizer_method(optimizer)
    search_kwargs = filter_valid_kwargs(optimizer_class, search_kwargs)
    return optimizer_class(
        estimator, param_grid, scoring=scoring, cv=cv, **search_kwargs)


def _perform_search(
        name, estimator, param_grid, optimizer, X, y, scoring, cv, 
        search_kwargs, progress_bar_desc):
    """
    Perform the hyperparameter search.

    This function performs the hyperparameter optimization for a given estimator 
    and parameter grid using the specified optimizer.

    Parameters
    ----------
    name : str
        Name of the estimator.
    
    estimator : estimator instance
        The estimator instance to be optimized.
    
    param_grid : dict
        Dictionary of parameter grids for the estimator.
    
    optimizer : str
        The optimization technique to apply (e.g., 'GridSearchCV').
    
    X : array-like of shape (n_samples, n_features)
        Training data.
    
    y : array-like of shape (n_samples,)
        Target values.
    
    scoring : str or callable
        Scoring method to evaluate the predictions on the test set.
    
    cv : int, cross-validation generator or iterable
        Determines the cross-validation splitting strategy.
    
    search_kwargs : dict
        Additional keyword arguments for the optimizer.
    
    progress_bar_desc : str
        Description for the progress bar.

    Returns
    -------
    tuple
        A tuple containing the name of the estimator, best estimator, best parameters, 
        best score, and cross-validation results.
    """
    search = _initialize_search(optimizer, estimator, param_grid, scoring, cv, **search_kwargs)
    search.fit(X, y)
    n_combinations = len(list(ParameterGrid(param_grid)))
    pbar = tqdm(total=n_combinations, desc=progress_bar_desc, ncols=103, ascii=True, position=0, leave=True)
    for _ in range(n_combinations):
        pbar.update(1)
    pbar.close()

    return (
        name,
        search.best_estimator_,
        search.best_params_,
        search.best_score_,
        search.cv_results_
    )







