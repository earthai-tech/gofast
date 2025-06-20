# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Provides classes and functions designed to optimize machine learning models, 
featuring methods for hyperparameter tuning and strategies for executing 
searches in parallel."""

from __future__ import annotations 
import joblib
import concurrent 
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from joblib import Parallel, delayed

from sklearn.base import BaseEstimator 

from ..api.box import KeyBox 
from ..api.types import Any, Dict, List,Union, Optional, ArrayLike
from ..api.types import _F, Array1D, NDArray, Callable 
from ..api.summary import ModelSummary
from ..core.utils import ellipsis2false 
from ..decorators import smartFitRun 
from ..utils.validator import get_estimator_name , check_X_y 
from ._optimize import BaseOptimizer, _perform_search, _validate_parameters
from .utils import get_strategy_method, params_combinations # noqa
from .utils import prepare_estimators_and_param_grids
from .utils import fix_batch_size_for_cv 


__all__=[
    "Optimizer", "ThreadedOptimizer" , "OptimizeSearch", "ParallelizeSearch", 
    "OptimizeHyperparams",  "ParallelOptimizer", "optimize_search", 
    "parallelize_search", "optimize_hyperparams", "optimize_search_in", 
    "BatchTuner"
    ]

class BatchTuner(BaseOptimizer):
    """
    Provides batch-wise hyperparameter optimization for multiple
    estimators by splitting data into chunks. The class
    `BatchTuner` inherits from `BaseOptimizer` and runs each
    chunk in parallel, aggregating the best results.
    
    .. math::
       \text{arg\,minimize}_{\Theta} \,
       \mathbb{E}[\text{Loss}(f(X; \Theta), y)]
    
    Here, :math:`\Theta` denotes the hyperparameter space,
    and data is divided into batches to accelerate the
    search process.
    
    Parameters
    ----------
    estimators : dict
        A mapping of custom names to estimator objects.
        Each key in `estimators` is a string uniquely
        identifying the estimator (e.g., `'svc'`,
        `'rf'`). The value is an instance of a scikit-learn
        compatible estimator (i.e., an object with `.fit`,
        `.predict`, and optionally `.score` methods). For
        example::
    
            estimators = {
                'svc': SVC(),
                'random_forest': RandomForestClassifier()
            }
    
        This dictionary may contain any number of estimators,
        each to be tuned with its own hyperparameter grid.
    
    param_grids : dict
        A dictionary mapping the same estimator names in
        `estimators` to their respective parameter grids.
        For each entry, the key must precisely match one of
        the keys in `<estimators>`, and the value is another
        dictionary describing the parameter search space.
        For example::
    
            param_grids = {
                'svc': {
                    'C': [1, 10],
                    'kernel': ['rbf', 'linear']
                },
                'random_forest': {
                    'n_estimators': [50, 100],
                    'max_depth': [2, 5, 10]
                }
            }
    
        Each parameter grid indicates which hyperparameters
        will be systematically explored by the chosen
        strategy (e.g., `'GSCV'`). For instance, with
        `'GSCV'`, every combination of listed parameter
        values is tested.
    
    strategy : str, default='GSCV'
        The name of the optimization search strategy.
        Determines which internal search method is used
        to explore hyperparameters. Valid values typically
        include (but are not limited to):
    
        * ``'GSCV'`` or ``'GridSearchCV'`` for exhaustive
          grid search,
        * ``'RSCV'`` or ``'RandomizedSearchCV'`` for random
          search,
        * ``'BSCV'`` or ``'BayesSearchCV'`` for Bayesian
          optimization,
        - ``'ASCV'``, ``'AnnealingSearchCV'`` for Simulated
          Annealing-based Search.
        - ``'SWSCV'``, ``'PSOSCV'`` for Swarm Optimization.
        - ``'SQSCV'``, ``'SequentialSearchCV'`` for 
          Sequential Model-Based Optimization.
        - ``'EVSCV'``, ``'EvolutionarySearchCV'`` for 
          Evolutionary Algorithms-based Search.
        - ``'GBSCV'``, ``'GradientSearchCV'`` for Gradient-Based
          Optimization.
        - ``'GENSCV'``, 'GeneticSearchCV' for Genetic 
          Algorithms-based Search.
 
        By default, it is set to ``'GSCV'``, which means a
        typical grid search approach is used (i.e., every
        combination of values in `param_grids` is tested).
        For specialized or advanced searching (like random
        sampling, Bayesian optimization, or evolutionary
        algorithms), you can switch to a different strategy
        (e.g., `'RSCV'`, `'BSCV'`,`'SWSCV'` etc.) as 
        supported by gofast package.
    
    scoring : str or callable, optional
        Metric for evaluating parameter sets. If None,
        uses the default scorer of the estimator.
        Accepts any scikit-learn-compatible scorer.
    
    cv : int or cross-validation generator, optional
        Cross-validation strategy. If None, defaults
        to 5-fold. An integer (e.g., 3) triggers
        k-fold with that many folds.
    
    save_results : bool, default=False
        If True, invokes the method
        `save_results_to_file` to persist results
        upon completion.
    
    n_jobs : int, default=-1
        Number of parallel jobs. `-1` uses all
        available CPUs. This parameter influences
        the parallelism during batch fits.
    
    batch_size : int, default=2
        Number of samples per batch. The input data
        is split into chunks of this size, each
        chunk is fit in parallel using the chosen
        strategy.
    
    **search_kwargs : dict
        Additional keyword arguments passed to the
        search constructor. For instance, when
        using random search, you could include
        ``n_iter=50`` in ``search_kwargs``.
    
    Methods
    -------
    fit(X, y)
        Perform parallel hyperparameter optimization
        over chunked data. It divides X, y into
        batches of size `batch_size` and runs
        the search in parallel. Returns a summary
        containing best parameters and scores.
    
    Examples
    --------
    >>> from gofast.models.optimize import BatchTuner
    >>> import pandas as pd
    >>> from sklearn.svm import SVC
    >>> # Create synthetic data
    >>> X = pd.DataFrame({
    ...     'feat1': [0.1, 0.3, 0.7, 1.2]*10,
    ...     'feat2': [1.0, 1.5, 0.9, 0.3]*10
    ... })
    >>> y = pd.Series([0, 1, 0, 1]*10)
    >>> # Define estimators & param grids
    >>> ests = {'svc': SVC()}
    >>> pgrids = {'svc': {'C': [1, 10], 'kernel': ['rbf','linear']}}
    >>> # Create and run tuner
    >>> tuner = BatchTuner(
    ...     estimators=ests,
    ...     param_grids=pgrids,
    ...     strategy='RSCV',
    ...     batch_size=2, 
    ...     cv=2, 
    ... )
    >>> summary = tuner.fit(X, y)
    >>> print(summary)
    
    Notes
    -----
    `BatchTuner` can be especially helpful when
    datasets are large or when data is naturally
    segmented. It harnesses parallel processing
    to evaluate different portions of the data
    simultaneously. After collecting results from
    each batch, it identifies the best overall
    parameters.
    
    See Also
    --------
    BaseOptimizer : Core class for optimization logic.
    sklearn.model_selection.GridSearchCV : Exhaustive
        parameter value search.
    sklearn.model_selection.RandomizedSearchCV :
        Random hyperparameter search.
    
    References
    ----------
    .. [1] Pedregosa, F. et al. *Scikit-learn: Machine
       Learning in Python.* *J. Mach. Learn. Res.*, 12,
       2825-2830, 2011.
    """

    def __init__(
        self,
        estimators,
        param_grids,
        strategy='GSCV',
        scoring=None,
        cv=None,
        save_results=False,
        n_jobs=-1,
        batch_size: int = 2,
        **search_kwargs
    ):
        # Initialize the parent class with standard arguments.
        super().__init__(
            estimators=estimators,
            param_grids=param_grids,
            strategy=strategy,
            scoring=scoring,
            cv=cv,
            save_results=save_results,
            n_jobs=n_jobs,
            **search_kwargs
        )
        # Store user-defined batch size.
        self.batch_size = batch_size

    def fit(self, X, y):
        # Validate search parameters and strategy.
        self._validate_search_params()

        # Total number of samples.
        n_samples = len(X)
        # fix batch_size for cv 
        self.batch_size = fix_batch_size_for_cv(
            X, y, 
            batch_size=self.batch_size, 
            cv= 5 if self.cv is None else self.cv, 
            classification="auto"
        ) 
        # Number of full batches
        n_batches = n_samples // self.batch_size
        # Leftover samples if not perfectly divisible.
        remainder = n_samples % self.batch_size

        # Build list of (start_idx, end_idx) for each batch.
        index_splits = []
        start_idx = 0
        for _ in range(n_batches):
            end_idx = start_idx + self.batch_size
            index_splits.append((start_idx, end_idx))
            start_idx = end_idx
        if remainder > 0:
            index_splits.append((start_idx, start_idx + remainder))

        # Store results across all estimators.
        aggregator = {}

        # Helper function to process one batch (parallelizable).
        def process_batch(estimator, param_grid, start_idx, end_idx):
            # Slice data for current batch.
            X_chunk = X[start_idx:end_idx]
            y_chunk = y[start_idx:end_idx]
            # Instantiate chosen search method (GridSearch, etc.).
            search_obj = self.strategy_(
                estimator,
                param_grid,
                scoring=self.scoring,
                cv=self.cv,
                n_jobs=self.n_jobs,
                **self.search_kwargs
            )
            # Fit the search object on the chunk.
            search_obj.fit(X_chunk, y_chunk)
        
            # Return best results from that batch.
            return {
                "best_estimator_": search_obj.best_estimator_,
                "best_params_":    search_obj.best_params_,
                "best_score_":     search_obj.best_score_,
                "cv_results_":     search_obj.cv_results_
            }
  
        # For each estimator, run parallel search across batches.
        names = list(self.generate_estimator_shortnames().keys())
        for name,  est, param_grid in zip(
                names,  self.estimators, self.param_grids
                ):
            # Execute each batch in parallel.
            with Parallel(n_jobs=self.n_jobs, prefer="threads") as parallel:
                batch_results = parallel(
                    delayed(process_batch)(
                        est,
                        param_grid,
                        start_idx,
                        end_idx
                    )
                    for (start_idx, end_idx) in tqdm(
                        index_splits,
                        desc=f"[BatchTuner] {name}",
                        ascii=True
                    )
                )

            # Determine the best batch by highest score.
            best_batch = max(batch_results, key=lambda r: r["best_score_"])
            aggregator[name] = best_batch

        # Optionally save results to file if requested.
        if self.save_results:
            self.save_results_to_file(aggregator)

        # Build summary from aggregator.
        summary = self.construct_model_summary(
            aggregator,
            descriptor="BatchTuner"
        )

        return summary


@smartFitRun
class OptimizeHyperparams(BaseOptimizer):
    """
    OptimizeHyperparams class for hyperparameter optimization of a single 
    estimator.

    This class facilitates the process of hyperparameter optimization for 
    a single machine learning estimator using various optimization 
    techniques such as Grid Search Cross Validation (GSCV). It allows the 
    user to specify the estimator, parameter grid, and other optimization 
    settings, and provides functionalities to perform the optimization and 
    save the results.

    Parameters
    ----------
    estimator : estimator object
        The object to use to fit the data. It should be an instance of a 
        model that implements the scikit-learn estimator interface.

    param_grid : dict or list of dicts
        Dictionary with parameters names (`str`) as keys and lists of 
        parameter settings to try as values. This can also be a list of such 
        dictionaries, in which case the grids spanned by each dictionary in 
        the list are explored. This enables searching over any sequence of 
        parameter settings.

    cv : int, default=5
        Determines the cross-validation splitting strategy. Possible inputs 
        for `cv` are:
          - None, to use the default 5-fold cross-validation,
          - int, to specify the number of folds in a (Stratified)KFold,
          - CV splitter,
          - An iterable yielding (train, test) splits as arrays of indices.

    scoring : str or callable, default=None
        A string (see model evaluation documentation) or a scorer callable 
        object / function with signature `scorer(estimator, X, y)`.

    strategy : str, default='GridSearchCV'
        The search strategy to apply for hyperparameter optimization. 
        Supported strategies include:
        
        - 'GSCV', 'GridSearchCV' for Grid Search Cross Validation.
        - 'RSCV', 'RandomizedSearchCV' for Randomized Search Cross 
          Validation.
        - 'BSCV', 'BayesSearchCV' for Bayesian Optimization.
        - 'ASCV', 'AnnealingSearchCV' for Simulated Annealing-based Search.
        - 'SWCV', 'PSOSCV', 'SwarmSearchCV' for Particle Swarm Optimization.
        - 'SQCV', 'SequentialSearchCV' for Sequential Model-Based 
          Optimization.
        - 'EVSCV', 'EvolutionarySearchCV' for Evolutionary Algorithms-based 
          Search.
        - 'GBSCV', 'GradientSearchCV' for Gradient-Based Optimization.
        - 'GENSCV', 'GeneticSearchCV' for Genetic Algorithms-based Search.

    n_jobs : int, default=-1
        The number of jobs to run in parallel for the optimization process. 
        `-1` means using all processors.

    savejob : bool, default=False
        If True, the optimization results (best parameters and scores) for 
        the estimator are saved to a file.

    savefile : str, optional
        File name to save the model binary. If `None`, the estimator name is 
        used instead.

    **search_kws : dict, optional
        Additional keyword arguments to pass to the search constructor.

    Methods
    -------
    fit(X, y)
        Perform hyperparameter optimization for the specified estimator on 
        the given data.

    Examples
    --------
    >>> from sklearn.svm import SVC
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> from gofast.models.optimize import OptimizeHyperparams
    >>> X, y = load_iris(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, 
    ...                                                     test_size=0.2, 
    ...                                                     random_state=42)
    >>> estimator = SVC()
    >>> param_grid = {'C': [1, 10], 'kernel': ['linear', 'rbf']}
    >>> optimizer = OptimizeHyperparams(estimator, param_grid, strategy='SWSCV', 
    ...                                 n_jobs=1)
    >>> results = optimizer.fit(X_train, y_train)
    SwarmSearch - SVC: 100%|##############################################| ...

    print(results) 
                                   Model Results                                
    ============================================================================
    Best estimator       : SVC
    Best parameters      : {'C': 4.583176637829093, 'kernel': 'linear'}
    Best score           : 0.9667
    nCV                  : 10
    Params combinations  : 100
    ============================================================================

                              Tuning Results (*=score)                          
    ============================================================================
                          Params   Mean*  Std.* Overall Mean* Overall Std.* Rank
    ----------------------------------------------------------------------------
    0 (4.583176637829093, linear) 0.9667 0.0667        0.9667        0.0667    1
    ============================================================================

    Notes
    -----
    The `OptimizeHyperparams` class uses various optimization strategies to 
    find the best hyperparameters for a given estimator. The progress of the 
    optimization is displayed using tqdm progress bars.

    The results of the optimization can be saved to disk using the `savejob` 
    and `savefile` parameters.
    
    The optimization process involves searching for the best set of 
    hyperparameters that minimizes or maximizes a given scoring function. 
    Given a set of hyperparameters :math:`\theta`, the goal is to find:

    .. math::
        \theta^* = \arg\min_{\theta \in \Theta} \mathbb{E}_{(X, y) \sim D} 
        [L(f(X; \theta), y)]

    where :math:`\Theta` represents the hyperparameter space, :math:`D` 
    denotes the data distribution, :math:`L` is the loss function, and 
    :math:`f(X; \theta)` is the model's prediction function.

    See Also
    --------
    sklearn.model_selection.GridSearchCV : Exhaustive search over specified 
        parameter values for an estimator.
    sklearn.model_selection.RandomizedSearchCV : Randomized search on hyper 
        parameters.
    ModelSummary : Class for summarizing model results.

    References
    ----------
    .. [1] Bergstra, J., & Bengio, Y. (2012). Random Search for Hyper-Parameter 
           Optimization. Journal of Machine Learning Research, 13, 281-305.
    .. [2] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., 
           Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine 
           Learning in Python. Journal of Machine Learning Research, 12, 
           2825-2830.
    """

    def __init__(
        self, 
        estimator: BaseEstimator, 
        param_grid: Dict[str, Any], 
        cv: int = 5, 
        scoring: Union[str, Callable] = None, 
        strategy: str = 'GSCV', 
        n_jobs: int = -1, 
        savejob: bool = False, 
        savefile: str = None, 
        **search_kwargs
        ):
        super().__init__(
            estimators={get_estimator_name(estimator): estimator}, 
            param_grids={get_estimator_name(estimator): param_grid}, 
            strategy=strategy, 
            scoring=scoring, 
            cv=cv, 
            save_results=savejob, 
            n_jobs=n_jobs, 
            **search_kwargs
        )
        self.estimator = estimator
        self.param_grid = param_grid
        self.savejob = savejob
        self.savefile = savefile
    
    def fit(self, X: ArrayLike, y: ArrayLike):
        """
        Perform hyperparameter optimization for the specified estimator on the 
        given data.
    
        This method applies a specified optimization technique (e.g., Grid Search 
        Cross Validation) to the estimator and its associated parameter grid. 
        It tunes the model's hyperparameters to find the best performing model 
        based on the provided data and scoring method.
    
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
        summary : ModelSummary
            A `ModelSummary` object containing the optimization results for the 
            estimator. The object includes information on the best estimator, 
            best parameters, best score, and cross-validation results.
    
        Notes
        -----
        The `fit` method leverages the selected optimization strategy to search 
        for the best hyperparameters. The progress of the optimization is 
        displayed using `tqdm` progress bars.
    
        If `savejob` is True, the results of the optimization (best parameters 
        and scores) are saved to a file specified by `savefile`. If `savefile` is 
        not provided, the estimator name is used instead.
    
        The optimization process involves searching for the best set of 
        hyperparameters that minimizes or maximizes a given scoring function. 
        Given a set of hyperparameters :math:`\theta`, the goal is to find:
    
        .. math::
            \theta^* = \arg\min_{\theta \in \Theta} \mathbb{E}_{(X, y) \sim D} 
            [L(f(X; \theta), y)]
    
        where :math:`\Theta` represents the hyperparameter space, :math:`D` 
        denotes the data distribution, :math:`L` is the loss function, and 
        :math:`f(X; \theta)` is the model's prediction function.
    
        Examples
        --------
        >>> from sklearn.svm import SVC
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> from gofast.models.optimize import OptimizeHyperparams
        >>> X, y = load_iris(return_X_y=True)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, 
        ...                                                     test_size=0.2, 
        ...                                                     random_state=42)
        >>> estimator = SVC()
        >>> param_grid = {'C': [1, 10], 'kernel': ['linear', 'rbf']}
        >>> optimizer = OptimizeHyperparams(estimator, param_grid, 
        ...                                 strategy='GSCV', n_jobs=1)
        >>> results = optimizer.fit(X_train, y_train)
        >>> print(results)
    
        See Also
        --------
        sklearn.model_selection.GridSearchCV : Exhaustive search over specified 
            parameter values for an estimator.
        sklearn.model_selection.RandomizedSearchCV : Randomized search on hyper 
            parameters.
        ModelSummary : Class for summarizing model results.
    
        References
        ----------
        .. [1] Bergstra, J., & Bengio, Y. (2012). Random Search for Hyper-Parameter 
               Optimization. Journal of Machine Learning Research, 13, 281-305.
        .. [2] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., 
               Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine 
               Learning in Python. Journal of Machine Learning Research, 12, 
               2825-2830.
        """
        self._control_strategy()
        strategy = self.strategy_(self.estimator, self.param_grid, cv=self.cv, 
                                  scoring=self.scoring, n_jobs=self.n_jobs,
                                  **self.search_kwargs)
        cv_search = f"{get_estimator_name(strategy)}".replace ("CV", "")
        desc=f"{cv_search} - {get_estimator_name(self.estimator)}"
        for _ in tqdm(range(1), desc=f"{desc}", ncols=100, ascii=True):
            strategy.fit(X, y)

            results_dict = {
                "best_estimator_": strategy.best_estimator_, 
                "best_params_": strategy.best_params_,
                "best_score_": strategy.best_score_,
                "cv_results_": strategy.cv_results_
            }

        if self.savejob:
            savefile = self.savefile or get_estimator_name(self.estimator)
            savefile = str(savefile).replace('.joblib', '')
            joblib.dump(results_dict, f'{savefile}.joblib')
            print(f"Results saved to {savefile}.joblib")

        self.summary = self.construct_model_summary(
            results_dict, descriptor="OptimizeHyperparams")
        
        return self.summary

    def __repr__(self):
        return super().__repr__()

    def __str__(self):
        return super().__str__()

@smartFitRun
class Optimizer(BaseOptimizer):
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

    strategy : str, default='GSCV'
        The search strategy to apply for hyperparameter optimization. 
        'GSCV' refers to Grid Search Cross Validation. Supportedstrategys
        include:
        
        - 'GSCV', 'GridSearchCV' for Grid Search Cross Validation.
        - 'RSCV', 'RandomizedSearchCV' for Randomized Search Cross Validation.
        - 'BSCV', 'BayesSearchCV' for Bayesian Optimization.
        - 'ASCV', 'AnnealingSearchCV' for Simulated Annealing-based Search.
        - 'SWCV', 'PSOSCV', 'SwarmSearchCV' for Particle Swarm Optimization.
        - 'SQCV', 'SequentialSearchCV' for Sequential Model-Based Optimization.
        - 'EVSCV', 'EvolutionarySearchCV' for Evolutionary Algorithms-based Search.
        - 'GBSCV', 'GradientSearchCV' for Gradient-Based Optimization.
        - 'GENSCV', 'GeneticSearchCV' for Genetic Algorithms-based Search.
    
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
        Additional keyword arguments to pass to thestrategy function.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.svm import SVC
    >>> from sklearn.linear_model import SGDClassifier
    >>> from gofast.models.optimize import Optimizer
    >>> X, y = load_iris(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
    ...                                                     random_state=42)
    >>> estimators = {'SVC': SVC(), 'SGDClassifier': SGDClassifier()}
    >>> param_grids = {'SVC': {'C': [1, 10], 'kernel': ['linear', 'rbf']}, 
    ...                'SGDClassifier': {'max_iter': [50, 100], 'alpha': [0.0001, 0.001]}}
    >>> optimizer = Optimizer(estimators, param_grids,strategy='GSCV', n_jobs=1)
    >>> results =optimizer.fit(X_train, y_train)
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
    implementing additionalstrategys and specifying them in the `strategy` 
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
        strategy='GSCV', 
        save_results=False,
        n_jobs=-1, 
        scoring=None, 
        cv=None, 
        **search_kwargs
    ):
        super().__init__(
            estimators=estimators, 
            param_grids=param_grids, 
            strategy=strategy, 
            scoring=scoring, 
            cv=cv, 
            save_results=save_results, 
            n_jobs=n_jobs, 
            **search_kwargs
            )

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
        different optimization techniques by specifying the `strategy` parameter.
    
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
        >>> optimizer = Optimizer(estimators, param_grids, strategy='GSCV', n_jobs=1)
        >>> optimizer.fit(X_train, y_train)
        >>> print(optimizer)
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
        self._validate_search_params() 
        X, y = check_X_y(X, y, accept_sparse=True, accept_large_sparse=True,
                         estimator=self )

        max_length = max([len(str(estimator)) for estimator in self.estimators])
  
        results = Parallel(n_jobs=self.n_jobs)(delayed(_perform_search)(
            name, self.estimators[i], self.param_grids[i], 
            self.strategy, X, y, self.scoring, self.cv, self.search_kwargs,
            f"Optimizing {get_estimator_name(name):<{max_length}}") for i, 
            name in enumerate(self.estimators))
    
        result_dict = {get_estimator_name(name): {
            'best_estimator_': best_est, 'best_params_': best_params, 
            'best_score_': best_sc, 'cv_results_': cv_res
            } for name, best_est, best_params, best_sc, cv_res in results
        }
    
        self.save_results_to_file(result_dict)
        return self.construct_model_summary(result_dict, descriptor="Optimizer")

@smartFitRun
class OptimizeSearch(BaseOptimizer):
    """
    OptimizeSearch class for hyperparameter optimization of multiple 
    estimators.

    This class facilitates the process of hyperparameter optimization for 
    multiple machine learning estimators using various optimization 
    techniques such as Grid Search Cross Validation (GSCV). It allows the 
    user to specify the estimators, parameter grids, and other optimization 
    settings, and provides functionalities to perform the optimization and 
    save the results.

    Parameters
    ----------
    estimators : dict of str, estimator objects
        A dictionary where keys are estimator names and values are estimator 
        instances. Each estimator is an instance of a model to be optimized.

    param_grids : dict of str, dict
        A dictionary where keys are estimator names (matching those in 
        `estimators`) and values are parameter grids. Each parameter grid is 
        a dictionary where the keys are parameter names and the values are 
        lists of parameter settings to try for that parameter.

    strategy : str, default='RSCV'
        The search strategy to apply for hyperparameter optimization. 
        Supported strategies include:
        
        - 'GSCV', 'GridSearchCV' for Grid Search Cross Validation.
        - 'RSCV', 'RandomizedSearchCV' for Randomized Search Cross 
          Validation.
        - 'BSCV', 'BayesSearchCV' for Bayesian Optimization.
        - 'ASCV', 'AnnealingSearchCV' for Simulated Annealing-based Search.
        - 'SWCV', 'PSOSCV', 'SwarmSearchCV' for Particle Swarm Optimization.
        - 'SQCV', 'SequentialSearchCV' for Sequential Model-Based 
          Optimization.
        - 'EVSCV', 'EvolutionarySearchCV' for Evolutionary Algorithms-based 
          Search.
        - 'GBSCV', 'GradientSearchCV' for Gradient-Based Optimization.
        - 'GENSCV', 'GeneticSearchCV' for Genetic Algorithms-based Search.

    save_results : bool, default=False
        If True, the optimization results (best parameters and scores) for 
        each estimator are saved to a file.

    n_jobs : int, default=-1
        The number of jobs to run in parallel for the optimization process. 
        `-1` means using all processors.

    **search_kwargs : dict, optional
        Additional keyword arguments to pass to the search constructor.

    Methods
    -------
    fit(X, y)
        Perform hyperparameter optimization for the specified estimators on 
        the given data.

    Examples
    --------
    >>> from sklearn.svm import SVC
    >>> from sklearn.linear_model import SGDClassifier
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> from gofast.models.optimize import OptimizeSearch
    >>> X, y = load_iris(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, 
    ...                                                     test_size=0.2, 
    ...                                                     random_state=42)
    >>> estimators = {'SVC': SVC(), 'SGDClassifier': SGDClassifier()}
    >>> param_grids = {'SVC': {'C': [1, 10], 'kernel': ['linear', 'rbf']}, 
    ...                'SGDClassifier': {'max_iter': [50, 100], 'alpha': 
    ...                                  [0.0001, 0.001]}}
    >>> optimizer = OptimizeSearch(estimators, param_grids, strategy='GSCV', 
    ...                            n_jobs=1)
    >>> results = optimizer.fit(X_train, y_train)
    >>> print(results)

    Notes
    -----
    The `OptimizeSearch` class uses parallel processing to expedite the 
    hyperparameter search process. The progress of the optimization is 
    displayed using tqdm progress bars.

    The hyperparameter optimization can be extended to use different 
    optimization techniques by specifying the `strategy` parameter.
    
    The optimization process involves searching for the best set of 
    hyperparameters that minimizes or maximizes a given scoring function. 
    Given a set of hyperparameters :math:`\theta`, the goal is to find:

    .. math::
        \theta^* = \arg\min_{\theta \in \Theta} \mathbb{E}_{(X, y) \sim D} 
        [L(f(X; \theta), y)]

    where :math:`\Theta` represents the hyperparameter space, :math:`D` 
    denotes the data distribution, :math:`L` is the loss function, and 
    :math:`f(X; \theta)` is the model's prediction function.

    The `GridSearchCV` and `RandomizedSearchCV` strategies perform an 
    exhaustive search over a specified parameter grid and a randomized search 
    over a parameter grid, respectively. Bayesian optimization techniques, 
    such as `BayesSearchCV`, model the objective function and select the 
    next hyperparameters to evaluate based on this model.

    See Also
    --------
    sklearn.model_selection.GridSearchCV : Exhaustive search over specified 
        parameter values for an estimator.
    sklearn.model_selection.RandomizedSearchCV : Randomized search on hyper 
        parameters.
    ModelSummary : Class for summarizing model results.

    References
    ----------
    .. [1] Bergstra, J., & Bengio, Y. (2012). Random Search for Hyper-Parameter 
           Optimization. Journal of Machine Learning Research, 13, 281-305.
    .. [2] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., 
           Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine 
           Learning in Python. Journal of Machine Learning Research, 12, 
           2825-2830.

    """
    def __init__(
        self, 
        estimators: Dict[str, BaseEstimator], 
        param_grids: Dict[str, Any], 
        strategy: str = 'RSCV', 
        scoring: Optional[Union[str, Callable]] = None, 
        cv: Optional[Union[int, Callable]] = None, 
        save_results: bool = False, 
        n_jobs: int = -1, 
        **search_kwargs: Any
        ):
        super().__init__(
            estimators=estimators, 
            param_grids=param_grids, 
            strategy=strategy, 
            scoring=scoring, 
            cv=cv, 
            save_results=save_results, 
            n_jobs=n_jobs, 
            **search_kwargs
            )

    def fit(self, X: Union[NDArray, ArrayLike], y: Union[Array1D, ArrayLike]):
        """
        Perform hyperparameter optimization for a list of estimators.
    
        This method applies a specified optimization technique (e.g., Grid Search 
        Cross Validation) to a range of estimators and their associated parameter 
        grids. It allows for the simultaneous tuning of multiple models, 
        facilitating the selection of the best model and parameters based on the 
        provided data.
    
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
        summary : ModelSummary
            A `ModelSummary` object containing the optimization results for each 
            estimator. The object includes information on the best estimator, 
            best parameters, best score, and cross-validation results for each 
            model.
    
        Notes
        -----
        The `fit` method leverages parallel processing using joblib to expedite 
        the hyperparameter search process. The progress of the optimization is 
        displayed using tqdm progress bars.
    
        The hyperparameter optimization is flexible and can be extended to use 
        different optimization techniques by specifying the `strategy` parameter.
    
        The optimization process involves searching for the best set of 
        hyperparameters that minimizes or maximizes a given scoring function. 
        Given a set of hyperparameters :math:`\theta`, the goal is to find:
    
        .. math::
            \theta^* = \arg\min_{\theta \in \Theta} \mathbb{E}_{(X, y) \sim D} 
            [L(f(X; \theta), y)]
    
        where :math:`\Theta` represents the hyperparameter space, :math:`D` 
        denotes the data distribution, :math:`L` is the loss function, and 
        :math:`f(X; \theta)` is the model's prediction function.
    
        Examples
        --------
        >>> from sklearn.svm import SVC
        >>> from sklearn.linear_model import SGDClassifier
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> from gofast.models.optimize import OptimizeSearch
        >>> X, y = load_iris(return_X_y=True)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, 
        ...                                                     test_size=0.2, 
        ...                                                     random_state=42)
        >>> estimators = {'SVC': SVC(), 'SGDClassifier': SGDClassifier()}
        >>> param_grids = {'SVC': {'C': [1, 10], 'kernel': ['linear', 'rbf']}, 
        ...                'SGDClassifier': {'max_iter': [50, 100], 'alpha': 
        ...                                  [0.0001, 0.001]}}
        >>> optimizer = OptimizeSearch(estimators, param_grids, strategy='GSCV', 
        ...                            n_jobs=1)
        >>> results = optimizer.fit(X_train, y_train)
        >>> print(results)
    
        See Also
        --------
        sklearn.model_selection.GridSearchCV : Exhaustive search over specified 
            parameter values for an estimator.
        sklearn.model_selection.RandomizedSearchCV : Randomized search on hyper 
            parameters.
        ModelSummary : Class for summarizing model results.
    
        References
        ----------
        .. [1] Bergstra, J., & Bengio, Y. (2012). Random Search for Hyper-Parameter 
               Optimization. Journal of Machine Learning Research, 13, 281-305.
        .. [2] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., 
               Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine 
               Learning in Python. Journal of Machine Learning Research, 12, 
               2825-2830.
        """
        self._validate_search_params()

        def perform_search(estimator_name, estimator, param_grid):
            search = self.strategy_(estimator, param_grid, n_jobs=self.n_jobs,
                                    scoring=self.scoring, cv=self.cv, 
                                    **self.search_kwargs)
            search.fit(X, y)
            return (estimator_name, search.best_estimator_, search.best_params_,
                    search.best_score_, search.cv_results_)
        
        estimators = {get_estimator_name(est): est for est in self.estimators}
        results = Parallel(n_jobs=self.n_jobs)(delayed(perform_search)(
            name, est, self.param_grids[ii])
            for ii, (name, est) in enumerate(tqdm(estimators.items(), 
                                  desc="Optimizing Estimators",
                                  ncols=100, ascii=True)))

        result_dict = {name: {'best_estimator_': best_est, 
                              'best_params_': best_params,
                              'best_score_': best_score, 
                              'cv_results_': cv_res}
                      for name, best_est, best_params, best_score, cv_res in results}

        self.save_results_to_file(result_dict)
        return self.construct_model_summary(result_dict, descriptor="OptimizeSearch")

@smartFitRun
class ParallelizeSearch(BaseOptimizer):
    """
    ParallelizeSearch class for hyperparameter optimization of multiple 
    estimators in parallel.

    This class facilitates the process of hyperparameter optimization for 
    multiple machine learning estimators using various optimization 
    techniques. It allows the user to specify the estimators, parameter 
    grids, and other optimization settings, and provides functionalities to 
    perform the optimization in parallel and save the results.

    Parameters
    ----------
    estimators : list of estimator objects
        List of estimators for which to optimize hyperparameters.

    param_grids : list of dicts
        List of parameter grids to search for each estimator. Each parameter 
        grid is a dictionary where the keys are parameter names and the 
        values are lists of parameter settings to try for that parameter.

    file_prefix : str, default='models'
        Prefix for the filename to save the estimators.

    cv : int, default=5
        Number of folds in cross-validation.

    scoring : str or callable, default=None
        Scoring method to use. It can be a string or a callable object/function 
        with the signature `scorer(estimator, X, y)`.

    strategy : str, default='RandomizedSearchCV'
        The search strategy to apply for hyperparameter optimization. 
        Supported strategies include:
        
        - 'GSCV', 'GridSearchCV' for Grid Search Cross Validation.
        - 'RSCV', 'RandomizedSearchCV' for Randomized Search Cross 
          Validation.
        - 'BSCV', 'BayesSearchCV' for Bayesian Optimization.
        - 'ASCV', 'AnnealingSearchCV' for Simulated Annealing-based Search.
        - 'SWCV', 'PSOSCV', 'SwarmSearchCV' for Particle Swarm Optimization.
        - 'SQCV', 'SequentialSearchCV' for Sequential Model-Based 
          Optimization.
        - 'EVSCV', 'EvolutionarySearchCV' for Evolutionary Algorithms-based 
          Search.
        - 'GBSCV', 'GradientSearchCV' for Gradient-Based Optimization.
        - 'GENSCV', 'GeneticSearchCV' for Genetic Algorithms-based Search.

    n_jobs : int, default=-1
        The number of jobs to run in parallel for the optimization process. 
        `-1` means using all processors.
        
    save_results : bool, default=False
        If True, the optimization results (best parameters and scores) for 
        each estimator are saved to a file.
        
    pack_models : bool, default=False
        If True, aggregate multiple models' results and save them into a single 
        binary file.

    **kws : dict, optional
        Additional keyword arguments to pass to the search constructor.

    Methods
    -------
    fit(X, y)
        Perform hyperparameter optimization for the specified estimators on 
        the given data.

    Examples
    --------
    >>> from sklearn.svm import SVC
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> from gofast.models.optimize import ParallelizeSearch
    >>> X, y = load_iris(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, 
    ...                                                     test_size=0.2, 
    ...                                                     random_state=42)
    >>> estimators = [SVC(), DecisionTreeClassifier()]
    >>> param_grids = [{'C': [1, 10], 'kernel': ['linear', 'rbf']}, 
    ...                {'max_depth': [3, 5, None], 'criterion': ['gini', 
    ...                                                         'entropy']}]
    >>> optimizer = ParallelizeSearch(estimators, param_grids, 
    ...                               strategy='RSCV', n_jobs=4)
    >>> results = optimizer.fit(X_train, y_train)
    >>> print(results)

    Notes
    -----
    When parallelizing tasks that are already CPU-intensive (like 
    GridSearchCV with `n_jobs=-1`), it's important to manage the overall CPU 
    load to avoid overloading your system. Adjust the `n_jobs` parameter 
    based on your system's capabilities.

    The `ParallelizeSearch` class uses `ThreadPoolExecutor` for parallel 
    processing and `tqdm` for progress display.

    The optimization process involves searching for the best set of 
    hyperparameters that minimizes or maximizes a given scoring function. 
    Given a set of hyperparameters :math:`\theta`, the goal is to find:

    .. math::
        \theta^* = \arg\min_{\theta \in \Theta} \mathbb{E}_{(X, y) \sim D} 
        [L(f(X; \theta), y)]

    where :math:`\Theta` represents the hyperparameter space, :math:`D` 
    denotes the data distribution, :math:`L` is the loss function, and 
    :math:`f(X; \theta)` is the model's prediction function.

    See Also
    --------
    sklearn.model_selection.GridSearchCV : Exhaustive search over specified 
        parameter values for an estimator.
    sklearn.model_selection.RandomizedSearchCV : Randomized search on hyper 
        parameters.
    ModelSummary : Class for summarizing model results.

    References
    ----------
    .. [1] Bergstra, J., & Bengio, Y. (2012). Random Search for Hyper-Parameter 
           Optimization. Journal of Machine Learning Research, 13, 281-305.
    .. [2] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., 
           Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine 
           Learning in Python. Journal of Machine Learning Research, 12, 
           2825-2830.
    """

    def __init__(
        self, 
        estimators: List[BaseEstimator], 
        param_grids: List[Dict[str, Any]], 
        file_prefix: str = "models", 
        cv: int = 5, 
        scoring: Union[str, Callable] = None, 
        strategy: str = "RSCV", 
        n_jobs: int = -1, 
        save_results: bool=False, 
        pack_models: bool = False, 
        **search_kwargs
        ):
        super().__init__(
            estimators=estimators, 
            param_grids=param_grids, 
            strategy=strategy, 
            scoring=scoring, 
            cv=cv, 
            save_results=save_results, 
            n_jobs=n_jobs, 
            **search_kwargs
            )
        self.file_prefix = file_prefix
        self.pack_models = pack_models


    def fit(self, X: ArrayLike, y: ArrayLike):
        """
        Perform hyperparameter optimization for a list of estimators in parallel.
    
        This method applies a specified optimization technique (e.g., Randomized 
        Search Cross Validation) to a range of estimators and their associated 
        parameter grids. It allows for the simultaneous tuning of multiple models, 
        facilitating the selection of the best model and parameters based on the 
        provided data. The optimization is performed in parallel to speed up the 
        process.
    
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
        summary : ModelSummary
            A `ModelSummary` object containing the optimization results for each 
            estimator. The object includes information on the best estimator, 
            best parameters, best score, and cross-validation results for each 
            model.
    
        Notes
        -----
        The `fit` method leverages parallel processing using 
        `concurrent.futures.ThreadPoolExecutor` to expedite the hyperparameter 
        search process. The progress of the optimization is displayed using 
        `tqdm` progress bars.
    
        The results of the optimization can be saved to disk, either as individual 
        files for each estimator or as a single aggregated file, depending on the 
        `pack_models` parameter.
    
        The optimization process involves searching for the best set of 
        hyperparameters that minimizes or maximizes a given scoring function. 
        Given a set of hyperparameters :math:`\theta`, the goal is to find:
    
        .. math::
            \theta^* = \arg\min_{\theta \in \Theta} \mathbb{E}_{(X, y) \sim D} 
            [L(f(X; \theta), y)]
    
        where :math:`\Theta` represents the hyperparameter space, :math:`D` 
        denotes the data distribution, :math:`L` is the loss function, and 
        :math:`f(X; \theta)` is the model's prediction function.
    
        Examples
        --------
        >>> from sklearn.svm import SVC
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> from gofast.models.optimize import ParallelizeSearch
        >>> X, y = load_iris(return_X_y=True)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, 
        ...                                                     test_size=0.2, 
        ...                                                     random_state=42)
        >>> estimators = [SVC(), DecisionTreeClassifier()]
        >>> param_grids = [{'C': [1, 10], 'kernel': ['linear', 'rbf']}, 
        ...                {'max_depth': [3, 5, None], 'criterion': ['gini', 
        ...                                                         'entropy']}]
        >>> optimizer = ParallelizeSearch(estimators, param_grids, 
        ...                               strategy='RSCV', n_jobs=4)
        >>> results = optimizer.fit(X_train, y_train)
        >>> print(results)
    
        See Also
        --------
        sklearn.model_selection.GridSearchCV : Exhaustive search over specified 
            parameter values for an estimator.
        sklearn.model_selection.RandomizedSearchCV : Randomized search on hyper 
            parameters.
        ModelSummary : Class for summarizing model results.
    
        References
        ----------
        .. [1] Bergstra, J., & Bengio, Y. (2012). Random Search for Hyper-Parameter 
               Optimization. Journal of Machine Learning Research, 13, 281-305.
        .. [2] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., 
               Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine 
               Learning in Python. Journal of Machine Learning Research, 12, 
               2825-2830.
        """
        o = {}
        pack = {}
        
        self._validate_search_params()

        if self.pack_models: 
            self.save_results =True 

        with ThreadPoolExecutor() as executor:
            futures = []
            for estimator, param_grid in zip(self.estimators, self.param_grids):
                futures.append(executor.submit(
                    optimize_hyperparams, estimator, 
                    param_grid, X, y, self.cv, self.scoring, self.strategy, 
                    self.n_jobs, **self.search_kwargs))

            for future, estimator in zip(
                    tqdm(concurrent.futures.as_completed(futures),
                         total=len(futures),
                         desc=f"Optimizing {self.estimators_nickname_} Estimators", 
                         ncols=100, ascii=True),
                    self.estimators):
                est_name = get_estimator_name(estimator)
                summary = future.result()
                best_estimator = summary.best_estimator_
                best_params = summary.best_params_
                cv_results = summary.cv_results_

                pack[f"{est_name}"] = {
                    "best_params_": best_params,
                    "best_estimator_": best_estimator,
                    
                    "cv_results_": cv_results
                }
                o[f"{est_name}"] = KeyBox(**pack[f"{est_name}"])

                if self.save_results and not self.pack_models:
                    file_name = f"{est_name}_{self.estimators.index(estimator)}.joblib"
                    joblib.dump((best_estimator, best_params), file_name)
                    print(f"Results saved to {file_name}")

            if self.pack_models:
                joblib.dump(pack, filename=f"{self.file_prefix}.joblib")
                print(f"Aggregated results saved to {self.file_prefix}.joblib")

        self.summary = ModelSummary(descriptor="ParallelizeSearch", **o)
        self.summary.summary(o)
        return self.summary

@smartFitRun
class ThreadedOptimizer(BaseOptimizer):
    """
    ThreadedOptimizer class for hyperparameter optimization of multiple estimators 
    separately.

    This class facilitates the process of hyperparameter optimization for 
    multiple machine learning estimators using various optimization 
    techniques such as Grid Search Cross Validation (GSCV). Each estimator 
    is optimized separately, with its progress displayed using a tqdm 
    progress bar.

    Parameters
    ----------
    estimators : dict of str, estimator objects
        A dictionary where keys are estimator names and values are estimator 
        instances. Each estimator is an instance of a model to be optimized.

    param_grids : dict of str, dict
        A dictionary where keys are estimator names (matching those in 
        `estimators`) and values are parameter grids. Each parameter grid is 
        a dictionary where the keys are parameter names and the values are 
        lists of parameter settings to try for that parameter.

    strategy : str, default='GSCV'
        The search strategy to apply for hyperparameter optimization. 
        Supported strategies include:
        
        - 'GSCV', 'GridSearchCV' for Grid Search Cross Validation.
        - 'RSCV', 'RandomizedSearchCV' for Randomized Search Cross 
          Validation.
        - 'BSCV', 'BayesSearchCV' for Bayesian Optimization.
        - 'ASCV', 'AnnealingSearchCV' for Simulated Annealing-based Search.
        - 'SWCV', 'PSOSCV', 'SwarmSearchCV' for Particle Swarm Optimization.
        - 'SQCV', 'SequentialSearchCV' for Sequential Model-Based 
          Optimization.
        - 'EVSCV', 'EvolutionarySearchCV' for Evolutionary Algorithms-based 
          Search.
        - 'GBSCV', 'GradientSearchCV' for Gradient-Based Optimization.
        - 'GENSCV', 'GeneticSearchCV' for Genetic Algorithms-based Search.
        
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
    save_results : bool, default=False
        If True, the optimization results (best parameters and scores) for 
        each estimator are saved to a file.

    n_jobs : int, default=-1
        The number of jobs to run in parallel for the optimization process. 
        `-1` means using all processors.

    **search_kwargs : dict, optional
        Additional keyword arguments to pass to the search constructor.

    Methods
    -------
    fit(X, y)
        Perform hyperparameter optimization for the specified estimators on 
        the given data.

    Examples
    --------
    >>> from sklearn.svm import SVC
    >>> from sklearn.linear_model import SGDClassifier
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> from gofast.models.optimize import ThreadedOptimizer
    >>> X, y = load_iris(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, 
    ...                                                     test_size=0.2, 
    ...                                                     random_state=42)
    >>> estimators = {'SVC': SVC(), 'SGDClassifier': SGDClassifier()}
    >>> param_grids = {'SVC': {'C': [1, 10], 'kernel': ['linear', 'rbf']}, 
    ...                'SGDClassifier': {'max_iter': [50, 100], 'alpha': 
    ...                                  [0.0001, 0.001]}}
    >>> optimizer = ThreadedOptimizer(estimators, param_grids, strategy='GSCV', 
    ...                        n_jobs=1)
    >>> results = optimizer.fit(X_train, y_train)
    >>> print(results)

    Notes
    -----
    The `ThreadedOptimizer` class uses parallel processing to expedite the 
    hyperparameter search process. Each estimator's optimization progress is 
    displayed using tqdm progress bars.

    The hyperparameter optimization can be extended to use different 
    optimization techniques by specifying the `strategy` parameter.
    
    The optimization process involves searching for the best set of 
    hyperparameters that minimizes or maximizes a given scoring function. 
    Given a set of hyperparameters :math:`\theta`, the goal is to find:

    .. math::
        \theta^* = \arg\min_{\theta \in \Theta} \mathbb{E}_{(X, y) \sim D} 
        [L(f(X; \theta), y)]

    where :math:`\Theta` represents the hyperparameter space, :math:`D` 
    denotes the data distribution, :math:`L` is the loss function, and 
    :math:`f(X; \theta)` is the model's prediction function.

    See Also
    --------
    sklearn.model_selection.GridSearchCV : Exhaustive search over specified 
        parameter values for an estimator.
    sklearn.model_selection.RandomizedSearchCV : Randomized search on hyper 
        parameters.
    ModelSummary : Class for summarizing model results.

    References
    ----------
    .. [1] Bergstra, J., & Bengio, Y. (2012). Random Search for Hyper-Parameter 
           Optimization. Journal of Machine Learning Research, 13, 281-305.
    .. [2] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., 
           Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine 
           Learning in Python. Journal of Machine Learning Research, 12, 
           2825-2830.

    """
    def __init__(
        self, 
        estimators: Dict[str, BaseEstimator], 
        param_grids: Dict[str, Any], 
        strategy: str='GSCV', 
        scoring: Union[str, Callable] = None, 
        cv: Union [int, Callable] =None, 
        save_results: bool=False, 
        n_jobs: int=-1, 
        **search_kwargs:Any
        ):
        super().__init__(
            estimators=estimators, 
            param_grids=param_grids, 
            strategy=strategy, 
            scoring=scoring, 
            cv=cv, 
            save_results=save_results, 
            n_jobs=n_jobs, 
            **search_kwargs
        )

    def fit(self, X: ArrayLike, y: ArrayLike):
        """
        Perform hyperparameter optimization for a list of estimators.

        This method applies a specified optimization technique (e.g., Grid Search 
        Cross Validation) to each estimator and its associated parameter grid 
        separately. It allows for the simultaneous tuning of multiple models, 
        facilitating the selection of the best model and parameters based on 
        the provided data.

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
        summary : ModelSummary
            A `ModelSummary` object containing the optimization results for each 
            estimator. The object includes information on the best estimator, 
            best parameters, best score, and cross-validation results for each 
            model.

        Notes
        -----
        The `fit` method leverages parallel processing using joblib to expedite 
        the hyperparameter search process. The progress of the optimization is 
        displayed using `tqdm` progress bars.

        The hyperparameter optimization is flexible and can be extended to use 
        different optimization techniques by specifying the `strategy` parameter.

        The optimization process involves searching for the best set of 
        hyperparameters that minimizes or maximizes a given scoring function. 
        Given a set of hyperparameters :math:`\theta`, the goal is to find:

        .. math::
            \theta^* = \arg\min_{\theta \in \Theta} \mathbb{E}_{(X, y) \sim D} 
            [L(f(X; \theta), y)]

        where :math:`\Theta` represents the hyperparameter space, :math:`D` 
        denotes the data distribution, :math:`L` is the loss function, and 
        :math:`f(X; \theta)` is the model's prediction function.
        
        
        Examples
        --------
        >>> from sklearn.svm import SVC
        >>> from sklearn.linear_model import SGDClassifier
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> from gofast.models.optimize import ThreadedOptimizer
        >>> X, y = load_iris(return_X_y=True)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, 
        ...                                                     test_size=0.2, 
        ...                                                     random_state=42)
        >>> estimators = {'SVC': SVC(), 'SGDClassifier': SGDClassifier()}
        >>> param_grids = {'SVC': {'C': [1, 10], 'kernel': ['linear', 'rbf']}, 
        ...                'SGDClassifier': {'max_iter': [50, 100], 'alpha': 
        ...                                  [0.0001, 0.001]}}
        >>> optimizer = ThreadedOptimizer(estimators, param_grids, strategy='GSCV', 
        ...                        n_jobs=1)
        >>> results = optimizer.fit(X_train, y_train)
        >>> print(results)

        See Also
        --------
        sklearn.model_selection.GridSearchCV : Exhaustive search over specified 
            parameter values for an estimator.
        sklearn.model_selection.RandomizedSearchCV : Randomized search on hyper 
            parameters.
        ModelSummary : Class for summarizing model results.

        References
        ----------
        .. [1] Bergstra, J., & Bengio, Y. (2012). Random Search for Hyper-Parameter 
               Optimization. Journal of Machine Learning Research, 13, 281-305.
        .. [2] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., 
               Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine 
               Learning in Python. Journal of Machine Learning Research, 12, 
               2825-2830.

        """
        def make_estimator_name (estimator): 
            return get_estimator_name(estimator ) if not isinstance ( 
                estimator, str) else estimator
        
        o = {}
        self._validate_search_params() 
        
        def perform_search(name, estimator, param_grid):
            strategy = self.strategy_(estimator, param_grid, cv=self.cv, 
                                      scoring=self.scoring, n_jobs=self.n_jobs,
                                      **self.search_kwargs)
            strategy.fit(X, y)
            return ( 
                name, 
                strategy.best_estimator_, 
                strategy.best_params_, 
                strategy.best_score_, 
                strategy.cv_results_
            )
        estimators = {make_estimator_name(est): est for est in self.estimators}
       
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(perform_search, name, est, self.param_grids[ii]): name 
                       for ii, (name, est) in enumerate (estimators.items())}

            for future in tqdm(concurrent.futures.as_completed(futures), 
                               total=len(futures), 
                               desc=f"Optimizing {self.estimators_nickname_} Estimators", 
                               ncols=100, ascii=True):
                name = futures[future]
                _, best_est, best_params, best_score, cv_results = future.result()
                o[name] = {
                    'best_estimator': best_est,
                    'best_params': best_params,
                    'best_score': best_score,
                    'cv_results': cv_results
                }

                if self.save_results:
                    file_name = f"{name}.joblib"
                    joblib.dump(o[name], file_name)
                    print(f"Results saved to {file_name}")

        self.summary = ModelSummary(descriptor="ThreadedOptimizer", **o)
        return self.summary.summary(o)
   
@smartFitRun
class ParallelOptimizer(BaseOptimizer):
    """
    ParallelOptimizer for hyperparameter optimization of multiple estimators.

    This class performs parallel hyperparameter optimization on a list of 
    estimators using various search strategies. It inherits from the 
    BaseOptimizer class, leveraging its common functionality and adding 
    parallel processing capabilities.

    Parameters
    ----------
    estimators : dict of str, estimator objects
        A dictionary where keys are estimator names and values are estimator 
        instances. Each estimator is an instance of a model to be optimized.

    param_grids : dict of str, dict
        A dictionary where keys are estimator names (matching those in 
        `estimators`) and values are parameter grids. Each parameter grid is 
        a dictionary where the keys are parameter names and the values are 
        lists of parameter settings to try for that parameter.

    strategy : str, default='GSCV'
        The search strategy to apply for hyperparameter optimization. 
        Supported strategies include:
        
        - 'GSCV', 'GridSearchCV' for Grid Search Cross Validation.
        - 'RSCV', 'RandomizedSearchCV' for Randomized Search Cross 
          Validation.
        - 'BSCV', 'BayesSearchCV' for Bayesian Optimization.
        - 'ASCV', 'AnnealingSearchCV' for Simulated Annealing-based Search.
        - 'SWSCV', 'PSOSCV' for Swarm Optimization.
        - 'SQSCV', 'SequentialSearchCV' for Sequential Model-Based 
          Optimization.
        - 'EVSCV', 'EvolutionarySearchCV' for Evolutionary Algorithms-based 
          Search.
        - 'GBSCV', 'GradientSearchCV' for Gradient-Based Optimization.
        - 'GENSCV', 'GeneticSearchCV' for Genetic Algorithms-based Search.

    scoring : str or callable, default=None
        A string (see model evaluation documentation) or a scorer callable 
        object / function with signature `scorer(estimator, X, y)`.

    cv : int or cross-validation generator or iterable, default=None
        Determines the cross-validation splitting strategy. Possible inputs 
        for `cv` are:
          - None, to use the default 5-fold cross-validation,
          - int, to specify the number of folds in a (Stratified)KFold,
          - CV splitter,
          - An iterable yielding (train, test) splits as arrays of indices.

    save_results : bool, default=False
        If True, the optimization results (best parameters and scores) for 
        each estimator are saved to a file.

    n_jobs : int, default=-1
        The number of jobs to run in parallel for the optimization process. 
        `-1` means using all processors.

    **search_kwargs : dict, optional
        Additional keyword arguments to pass to the search constructor.

    Methods
    -------
    _optimize(estimator_name, estimator, param_grid, X, y)
        Perform the optimization for a single estimator and return the results.

    fit(X, y)
        Run the optimization for all estimators in parallel and return the 
        summary of results.

    Notes
    -----
    This class leverages parallel processing to efficiently optimize 
    hyperparameters for multiple estimators. It supports various search 
    strategies, making it flexible for different optimization needs.

    The optimization process aims to find the best hyperparameters 
    :math:`\theta^*` that minimize the objective function :math:`L(\theta)` 
    over the given parameter grid :math:`\Theta`:

    .. math:: 
        \theta^* = \arg\min_{\theta \in \Theta} L(\theta)

    Examples
    --------
    >>> from gofast.models.optimize import ParallelOptimizer
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.svm import SVC
    >>> from sklearn.ensemble import RandomForestClassifier

    >>> data = load_iris()
    >>> X, y = data.data, data.target

    >>> estimators = {
    >>>     'SVC': SVC(),
    >>>     'RandomForest': RandomForestClassifier()
    >>> }

    >>> param_grids = {
    >>>     'SVC': {'kernel': ['linear', 'rbf'], 'C': [1, 10]},
    >>>     'RandomForest': {'n_estimators': [50, 100], 'max_depth': [5, 10]}
    >>> }

    >>> optimizer = ParallelOptimizer(estimators, param_grids, strategy='GSCV'
                                      , cv=5, save_results=True)
    >>> results = optimizer.fit(X, y)
    Optimizing SVC                   : 100%|##############################| 4/4 
    Optimizing RandomForestClassifier: 100%|##############################| 4/4 
    
    >>> print(results) 
                          Optimized Results                      
    =============================================================
    |                            SVC                            |
    -------------------------------------------------------------
                            Model Results                        
    =============================================================
    Best estimator       : SVC
    Best parameters      : {'C': 1, 'kernel': 'linear'}
    Best score           : 0.9800
    nCV                  : 5
    Params combinations  : 4
    =============================================================

                       Tuning Results (*=score)                  
    =============================================================
           Params   Mean*  Std.* Overall Mean* Overall Std.* Rank
    -------------------------------------------------------------
    0  (1, linear) 0.9800 0.0163        0.9800        0.0163    1
    1     (1, rbf) 0.9667 0.0211        0.9667        0.0211    4
    2 (10, linear) 0.9733 0.0389        0.9733        0.0389    3
    3    (10, rbf) 0.9800 0.0163        0.9800        0.0163    1
    =============================================================


    =============================================================
    |                   RandomForestClassifier                  |
    -------------------------------------------------------------
                            Model Results                        
    =============================================================
    Best estimator       : RandomForestClassifier
    Best parameters      : {'max_depth': 5, 'n_estimators': 100}
    Best score           : 0.9667
    nCV                  : 5
    Params combinations  : 4
    =============================================================

                       Tuning Results (*=score)                  
    =============================================================
           Params   Mean*  Std.* Overall Mean* Overall Std.* Rank
    -------------------------------------------------------------
    0      (5, 50) 0.9533 0.0340        0.9533        0.0340    4
    1     (5, 100) 0.9667 0.0211        0.9667        0.0211    1
    2     (10, 50) 0.9600 0.0249        0.9600        0.0249    3
    3    (10, 100) 0.9667 0.0211        0.9667        0.0211    1
    =============================================================

    See Also
    --------
    BaseOptimizer : Base class for hyperparameter optimization.
    sklearn.model_selection.GridSearchCV : Exhaustive search over specified 
        parameter values for an estimator.
    sklearn.model_selection.RandomizedSearchCV : Randomized search on hyper 
        parameters.
    ModelSummary : Class for summarizing model results.

    References
    ----------
    .. [1] Bergstra, J., & Bengio, Y. (2012). Random Search for Hyper-Parameter 
           Optimization. Journal of Machine Learning Research, 13, 281-305.
    .. [2] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., 
           Grisel, O., ... & Duchesnay, E. (2011). Scikit-learn: Machine 
           Learning in Python. Journal of Machine Learning Research, 12, 
           2825-2830.
    """
    def __init__(
        self, 
        estimators, 
        param_grids, 
        strategy='GSCV', 
        scoring=None, 
        cv=None, 
        n_jobs=-1, 
        save_results=False, 
        **search_kwargs
        ):
        super().__init__( 
            estimators = estimators, 
            param_grids= param_grids, 
            strategy=strategy, 
            scoring=scoring, 
            cv=cv, 
            save_results=save_results, 
            n_jobs=n_jobs, 
            **search_kwargs)

    def _optimize(self, estimator_name, estimator, param_grid, X, y):
        search = self.strategy_(
            estimator, 
            param_grid, 
            scoring=self.scoring, 
            cv=self.cv, 
            n_jobs=self.n_jobs, 
            **self.search_kwargs
            )
        desc = f"Optimizing {estimator_name:<{self._max_name_length_}}"
        n_combinations = len(list(params_combinations(param_grid)))
        with tqdm(total=n_combinations, desc=desc, ascii=True, ncols=100 ) as pbar:
            search.fit(X, y)
           
            best_estimator = search.best_estimator_
            best_params = search.best_params_
            best_score = search.best_score_
            cv_results = search.cv_results_
    
            result = {
                "best_estimator_": best_estimator,
                "best_params_": best_params,
                "best_score_": best_score,
                "cv_results_": cv_results
            }
    
            if self.save_results:
                self.save_results_to_file(result, f"{estimator_name}_results")
            
            pbar.update(n_combinations)

        return estimator_name, result

    def fit(self, X, y):
        """
        Run the hyperparameter optimization for all estimators in parallel.

        This method performs the optimization process for each estimator 
        using the specified strategy, running the tasks in parallel to 
        expedite the process. The results are summarized and returned.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data to fit.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target variable to try to predict in supervised learning.

        Returns
        -------
        summary : ModelSummary
            A summary object containing the results of the hyperparameter 
            optimization for each estimator.

        Notes
        -----
        This method first validates the search parameters to ensure they are 
        correctly specified. It then constructs a dictionary of estimators 
        and their names, and calculates the maximum length of estimator 
        names for display purposes. The optimization is performed in parallel 
        using joblib's `Parallel` and `delayed` functions. The results are 
        collected in a dictionary and summarized using a `ModelSummary` 
        object.

        The optimization process aims to find the best hyperparameters 
        :math:`\theta^*` for each estimator that minimize the objective 
        function :math:`L(\theta)` over the given parameter grid 
        :math:`\Theta`:

        .. math:: 
            \theta^* = \arg\min_{\theta \in \Theta} L(\theta)

        Examples
        --------
        >>> from gofast.models.optimize import ParallelOptimizer
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.svm import SVC
        >>> from sklearn.ensemble import RandomForestClassifier

        >>> data = load_iris()
        >>> X, y = data.data, data.target

        >>> estimators = {
        >>>     'SVC': SVC(),
        >>>     'RandomForest': RandomForestClassifier()
        >>> }

        >>> param_grids = {
        >>>     'SVC': {'kernel': ['linear', 'rbf'], 'C': [1, 10]},
        >>>     'RandomForest': {'n_estimators': [50, 100], 'max_depth': [5, 10]}
        >>> }

        >>> optimizer = ParallelOptimizer(estimators, param_grids, strategy='GSCV',
                                          cv=5, save_results=True)
        >>> results = optimizer.fit(X, y)

        >>> for estimator_name, result in results.items():
        >>>     print(f"{estimator_name}:")
        >>>     print(f"Best Estimator: {result['best_estimator_']}")
        >>>     print(f"Best Params: {result['best_params_']}")
        >>>     print(f"Best Score: {result['best_score_']}")
        SVC:
        Best Estimator: SVC(C=1, kernel='linear')
        Best Params: {'C': 1, 'kernel': 'linear'}
        Best Score: 0.9800000000000001
        RandomForestClassifier:
        Best Estimator: RandomForestClassifier(max_depth=5)
        Best Params: {'max_depth': 5, 'n_estimators': 100}
        Best Score: 0.9666666666666668

        See Also
        --------
        BaseOptimizer : Base class for hyperparameter optimization.
        sklearn.model_selection.GridSearchCV : Exhaustive search over specified 
            parameter values for an estimator.
        sklearn.model_selection.RandomizedSearchCV : Randomized search on hyper 
            parameters.
        ModelSummary : Class for summarizing model results.

        References
        ----------
        .. [1] Bergstra, J., & Bengio, Y. (2012). Random Search for 
               Hyper-Parameter Optimization. Journal of Machine Learning 
               Research, 13, 281-305.
        .. [2] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., 
               Thirion, B., Grisel, O., ... & Duchesnay, E. (2011). 
               Scikit-learn: Machine Learning in Python. Journal of Machine 
               Learning Research, 12, 2825-2830.
        """

        self._validate_search_params() 
        estimators = {get_estimator_name(estimator): estimator for estimator 
                      in self.estimators}
        self._max_name_length_= max ( len(name) for name in estimators.keys())
        
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._optimize)(name, estimator, self.param_grids[ii], X, y)
            for ii, (name, estimator)  in enumerate (zip(
                    estimators.keys(), estimators.values())
        )
            )

        results = dict(results)
        self.summary = ModelSummary(descriptor="ParallelOptimizer", **results)
        return self.summary.summary(results)
    

def optimize_search(
    estimators: Dict[str, BaseEstimator], 
    param_grids: Dict[str, Any], 
    X: Union [NDArray, ArrayLike], 
    y: Union [Array1D, ArrayLike], 
    strategy: str = 'RSCV', 
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
    strategy : str, optional
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
        containing 'best_estimator', 'best_params', and 'cv_results' for each 
        estimator.

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
    >>> results = optimize_search(estimators, param_grids, X, y,strategy='RSCV',
    ...                          save_results=False, n_jobs=4)
    """
    estimators, param_grids = prepare_estimators_and_param_grids(
        estimators, param_grids)
    strategy_class = get_strategy_method(strategy)

    def perform_search(estimator_name, estimator, param_grid):
        search =strategy_class(estimator, param_grid, n_jobs=n_jobs, **search_kwargs)
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

def optimize_search_in(
    estimators: Dict[str, BaseEstimator], 
    param_grids: Dict[str, Any],
    X: ArrayLike, 
    y: ArrayLike, 
    strategy: str='GSCV',
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

    strategy : str, default='GSCV'
        The optimization technique to apply. 'GSCV' refers to Grid Search 
        Cross Validation. Additionalstrategys can be implemented and 
        specified here.

    save_results : bool, default=False
        If True, the optimization results (best parameters and scores) for 
        each estimator are saved to a file.

    n_jobs : int, default=-1
        The number of jobs to run in parallel for `strategy`. `-1` means 
        using all processors.

    search_kwargs : dict, optional
        Additional keyword arguments to pass to thestrategy function.

    Returns
    -------
    results : dict
        A dictionary containing the optimization results for each estimator. 
        The keys are the estimator names, and the values are the results 
        returned by thestrategy for that estimator.

    Examples
    --------
    >>> from sklearn.svm import SVC
    >>> from sklearn.linear_model import SGDClassifier
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from gofast.models.optimize import optimize_search_in
    >>> X, y = make_classification(n_samples=100, n_features=7, 
                                   random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size=0.2, 
                                                            random_state=42)
    >>> estimators = [SVC(), SGDClassifier()]
    >>> param_grids = [{'C': [1, 10], 'kernel': ['linear', 'rbf']}, 
                       {'max_iter': [50, 100], 'alpha': [0.0001, 0.001]}]
    >>> result = optimize_search_in(estimators, param_grids, X_train, y_train, 
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
    >>> result_dict=optimize_search_in(estimators, param_grids, X_train, y_train)
    
    Notes
    -----
    - The function uses joblib for parallel processing. Ensure that the 
      objects passed to the function are pickleable.
    - The progress bars are displayed using tqdm to show the progress of 
      each estimator's optimization process.
    - The optimization technique can be extended to include other methods 
      by implementing additionalstrategys and specifying them in the 
      `strategy` parameter.

    """
    def make_estimator_name (estimator): 
        return get_estimator_name(estimator ) if not isinstance ( 
            estimator, str) else estimator
    
    estimators, param_grids = _validate_parameters(param_grids, estimators,)
    max_length = max([len(str(estimator)) for estimator in estimators])
    
    # try: 
    results = Parallel(n_jobs=n_jobs)(delayed(_perform_search)(
        name, estimators[i], param_grids[i],strategy, X, y, 
        scoring, cv, search_kwargs,
        f"Optimizing {make_estimator_name(name):<{max_length}}")
        for i, name in enumerate(estimators))
    # except: 
    #     result_dict= _optimize_search_in(
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
    strategy: str= 'GridSearchCV', 
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
    strategy_class = get_strategy_method(strategy) 
    strategy =strategy_class (estimator, param_grid, cv=cv, scoring=scoring,
                                  n_jobs=n_jobs, **kws)
    strategy.fit(X, y)

    results_dict= {"best_estimator_":strategy.best_estimator_ , 
                   "best_params_":strategy.best_params_  , 
                   "cv_results_":strategy.cv_results_}
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
    strategy="RandomizedSearchCV", 
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
                param_grid, X, y, cv, scoring,strategy, 
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









