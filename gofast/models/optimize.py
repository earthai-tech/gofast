#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Created on Wed Dec 20 11:48:55 2023

This script defines two functions: optimize_hyperparameters for optimizing a 
single estimator and parallelize_estimators for handling multiple estimators 
in parallel. The latter function also saves the best estimator and parameters 
to disk using joblib.
"""

# import numpy as np
import joblib
import concurrent 
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import ( 
    GridSearchCV, 
    RandomizedSearchCV, 
    )

from ..tools.funcutils import smart_format, ellipsis2false 
from ..tools.validator import get_estimator_name 
from ..tools._dependency import import_optional_dependency 
from ..tools.box import Boxspace 

opt_dict = { 
    'RandomizedSearchCV': [ 'RSCV', 'RandomizedSearchCV'], 
    'GridSearchCV': ['GSCV', 'GridSearchCV'], 
    'BayesSearchCV': ['BSCV', 'BayesSearchCV']
    }

def optimize_search(
    estimators, 
    param_grids, 
    X, 
    y, 
    optimizer='RSCV', 
    save_results=False, 
    n_jobs=-1, 
    **search_kwargs
    ):
    """
    Perform hyperparameter optimization for multiple estimators in parallel. 
    
    Function supports Grid Search, Randomized Search, and Bayesian Search. This 
    parallel processing can significantly expedite the hyperparameter tuning process.

    Parameters
    ----------
    estimators : dict
        A dictionary where keys are estimator names and values are estimator 
        instances.
    param_grids : dict
        A dictionary where keys are estimator names (matching those in 'estimators') 
        and values are parameter grids.
    X : ndarray or DataFrame
        Input features for the model.
    y : ndarray or Series
        Target variable for the model.
    optimizer : str, optional
        Type of search to perform: 'GSCV' or 'GridSearchCV', 'RSCV' or 
        'RandomizedSearchCV', 'BSCV' or 'BayesianSearchCV'. Default is 'RSCV'.
    save_results : bool, optional
        If True, saves the results of the search to a joblib file. 
        Default is False.
    n_jobs : int, optional
        Number of jobs to run in parallel. Default is -1, which uses all 
        available processors.
    **search_kwargs : dict
        Additional keyword arguments to pass to the search constructor.

    Returns
    -------
    dict
        A dictionary with keys as estimator names and values as dictionaries 
        containing 'best_estimator', 'best_params', and 'cv_results' for 
        each estimator.

    Raises
    ------
    ValueError
        If 'kind' is not a recognized option or if the keys in 'estimators' 
        and 'param_grids' do not match.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.svm import SVC
    >>> from sklearn.datasets import load_iris 
    >>> X, y = load_iris (return_X_y=True)
    >>> from gofast.models.optimize import optimize_search
    >>> estimators = {'rf': RandomForestClassifier(), 'svc': SVC()}
    >>> param_grids = {'rf': {'n_estimators': [10, 100], 'max_depth': [None, 10]},
    ...                'svc': {'C': [1, 10], 'kernel': ['linear', 'rbf']}}
    >>> results = optimize_search(estimators, param_grids, X, y, optimizer='RSCV',
                                  save_results=False, n_jobs=4)
    """
    if optimizer not in ['GSCV', 'GridSearchCV', 'RSCV', 'RandomizedSearchCV',
                    'BSCV', 'BayesianSearchCV']:
        raise ValueError("Invalid 'optimizer' parameter. Choose from 'GSCV', 'RSCV', or 'BSCV'.")

    if estimators.keys() != param_grids.keys():
        raise ValueError("The keys in 'estimators' and 'param_grids' must match.")
    if optimizer in ['BSCV', 'BayesianSearchCV']: 
        extra_msg = ("'BayesSearchCV' expects `skopt` to be installed."
                     " Skopt is the shorthand of `scikit-optimize` library.")
        import_optional_dependency("skopt", extra =extra_msg )
        from skopt import BayesSearchCV  # Assuming skopt for Bayesian Search
        
    def perform_search(estimator_name, estimator, param_grid):
        if optimizer in ['GSCV', 'GridSearchCV']:
            search = GridSearchCV(estimator, param_grid, **search_kwargs)
        elif optimizer in ['RSCV', 'RandomizedSearchCV']:
            search = RandomizedSearchCV(estimator, param_grid, **search_kwargs)
        elif optimizer in ['BSCV', 'BayesianSearchCV']:
            search = BayesSearchCV(estimator, param_grid, **search_kwargs)
        search.fit(X, y)
        return estimator_name, search.best_estimator_, search.best_params_, search.cv_results_

    # Parallel execution of the search for each estimator
    results = Parallel(n_jobs=n_jobs)(delayed(perform_search)(name, est, param_grids[name])
            for name, est in tqdm(estimators.items(), desc="Optimizing Estimators",
                                  ncols=77, ascii=True ))
    
    result_dict = {name: {'best_estimator_': best_est, 'best_params_': best_params,
                          'cv_results_': cv_res}
                   for name, best_est, best_params, cv_res in results}
    
    # Optionally save results to a joblib file
    if save_results:
        filename = "optimization_results.joblib"
        joblib.dump(result_dict, filename)
        print(f"Results saved to {filename}")
    
    return result_dict

def optimize_search2(estimators, param_grids, X, y, optimizer='GSCV', 
                    save_results=False, n_jobs=-1, **search_kwargs):
    
    if optimizer in ['BSCV', 'BayesianSearchCV']: 
        extra_msg = ("'BayesSearchCV' expects `skopt` to be installed."
                     " Skopt is the shorthand of `scikit-optimize` library.")
        import_optional_dependency("skopt", extra =extra_msg )
        from skopt import BayesSearchCV  # Assuming skopt for Bayesian Search
        
    def validate_parameters():
        if optimizer not in ['GSCV', 'GridSearchCV', 'RSCV', 'RandomizedSearchCV',
                             'BSCV', 'BayesianSearchCV']:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        if estimators.keys() != param_grids.keys():
            raise ValueError("The keys in 'estimators' and 'param_grids' must match.")

    def initialize_search(optimizer, estimator, param_grid):
        if optimizer in ['GSCV', 'GridSearchCV']:
            return GridSearchCV(estimator, param_grid, **search_kwargs)
        elif optimizer in ['RSCV', 'RandomizedSearchCV']:
            return RandomizedSearchCV(estimator, param_grid, **search_kwargs)
        elif optimizer in ['BSCV', 'BayesianSearchCV']:
            
            return BayesSearchCV(estimator, param_grid, **search_kwargs)

    def perform_search(name, estimator, param_grid, pbar):
        search = initialize_search(optimizer, estimator, param_grid)
        # tem_r={}
        for _ in tqdm(range(search_kwargs.get('n_iter', 1)), position=0,
                      leave=False, desc="{:<30}".format(f"Optimizing {name}"),
                      ncols=103, ascii=True):
            search.fit(X, y)
            pbar.update(1)
        return name, search.best_estimator_, search.best_params_, search.cv_results_

    validate_parameters()
    progress_bars = [tqdm(total=search_kwargs.get('n_iter', 1), position=i + 1,
                          desc="{:<30}".format(f"Optimizing {name}"),
                          ncols=103, ascii=True) for i, name in enumerate(estimators)
                     ]
    results = Parallel(n_jobs=n_jobs)(delayed(perform_search)(
        name, estimators[name], param_grids[name], progress_bars[i])
              for i, name in enumerate(estimators))

    for pbar in progress_bars:
        pbar.close()
    result_dict = {name: {'best_estimator': best_est, 'best_params': best_params,
                          'cv_results': cv_res}
                   for name, best_est, best_params, cv_res in results}
    if save_results:
        joblib.dump(result_dict, "optimization_results.joblib")

    return result_dict

def optimize_hyperparameters(
    estimator, 
    param_grid, 
    X, y, 
    cv=5, 
    scoring=None, 
    optimizer= 'RandomisedSearchCV', 
    n_jobs=-1, 
    savejob: bool= ..., 
    savefile: str=None, 
    **kws 
    ):
    """
    Optimize hyperparameters for a given estimator using GridSearchCV, 
    with parallelization.

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
    savejob, = ellipsis2false(savejob )

    if isinstance (optimizer, str): 
        for key, values in opt_dict.items(): 
               if str(optimizer).lower() in  [ s.lower() for s in values]: 
                   optimizer = key 
    else: 
        optimizer = get_estimator_name(optimizer) 
    
    if optimizer not in opt_dict.keys(): 
        raise ValueError(
            f"Unknow Optimizer. Expects {smart_format(opt_dict.keys())}."
            f" Got {optimizer!r}.")
    if optimizer =='BayesSearchCV': 
        extra_msg = ("'BayesSearchCV' expects `skopt` to be installed."
                     " Skopt is the shorthand of `scikit-optimize` library.")
        import_optional_dependency("skopt", extra =extra_msg )
        # ++++++++++++++++++++++++++++++++++++++++
        from skopt.searchcv import BayesSearchCV 
        # ++++++++++++++++++++++++++++++++++++++++
        optimizer =  BayesSearchCV ( estimator, search_spaces = param_grid, 
                                    cv=cv, scoring=scoring, **kws)
        
    elif optimizer =='RandomizedSearchCV': 
        optimizer = RandomizedSearchCV(estimator, param_distributions= param_grid, 
                                       scoring=scoring, cv=cv, **kws) 
    else: 
        optimizer = GridSearchCV ( estimator, param_grid, cv=cv, 
                                   scoring=scoring, n_jobs=n_jobs, 
                                   **kws)
    optimizer.fit(X, y)
    
    # try to save file 
    if savejob: 
        savefile = savefile or get_estimator_name(estimator)
        # remove joblib if extension is appended.
        savefile= str(savefile).replace ('.joblib', '')
        joblib.dump ( dict ( optimizer.best_estimator_,
                            optimizer.best_params_, 
                            optimizer.cv_results_
                            ),
                     filename = f'{savefile}.joblib' 
                     )
    return ( optimizer.best_estimator_,
            optimizer.best_params_, 
            optimizer.cv_results_
            )

def parallelize_estimators(
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
    o: gofast.tools.boxspace
        The function saves the best estimator and parameters, and 
        cv results for each input estimator to disk
        returns object where `best_params_`, `best_estimators_` and `cv_results_`
        can be retrieved as an object.

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
    >>> X, y = load_iris(return_X_y=True)
    >>> estimators = [SVC(), DecisionTreeClassifier()]
    >>> param_grids = [{'C': [1, 10], 'kernel': ['linear', 'rbf']}, 
                       {'max_depth': [3, 5, None], 'criterion': ['gini', 'entropy']}
                       ]

    >>> o= parallelize_estimators(estimators, param_grids, X, y)
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
                optimize_hyperparameters, estimator, 
                param_grid, X, y, cv, scoring, optimizer, 
                n_jobs, **kws))

        for idx, (future, estimator)in enumerate (zip (
                tqdm(concurrent.futures.as_completed(futures),
                           total=len(futures), desc="Optimizing Estimators", 
                           ncols=77, ascii=True,
                           ), estimators)
                                                 ):
            est_name = get_estimator_name(estimator)
            best_estimator, best_params, cv_results = future.result()
            # save model results into a large object that can be return 
            # as an object . 
            
            pack [f"{est_name}"]= {"best_params_": best_params, 
                                   "best_estimator_": best_estimator, 
                                   "cv_results_": cv_results
                                   }
            o[f"{est_name}"]= Boxspace ( ** pack [f"{est_name}"])
            
            if  not pack_models: 
                # save all model individualy and append index 
                # to differential wether muliple 
                file_name = f"{est_name}_{idx}.joblib"
                joblib.dump((best_estimator, best_params), file_name)
                
        if pack_models: 
            joblib.dump(pack , filename= f"{file_prefix}.joblib")

    return Boxspace( **o)
