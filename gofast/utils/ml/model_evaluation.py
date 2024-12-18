# -*- coding: utf-8 -*-
#   Licence:BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Offers tools and functions to evaluate and assess the performance of
machine learning models using various metrics and validation techniques.
"""

import numpy as np
from sklearn.metrics import ( 
    mean_absolute_error, mean_squared_error, accuracy_score
 ) 
from ..._gofastlog import gofastlog
from ...api.docstring import DocstringComponents, _core_docs 
from ...api.types import Tuple, Any, Dict, Optional, Union, Series
from ...api.types import _F, ArrayLike, NDArray, DataFrame
from ...api.formatter import MetricFormatter

from ...compat.sklearn import  validate_params, HasMethods
from ...core.checks import is_classification_task
from ..validator import check_consistent_length, check_is_fitted

# Logger Configuration
_logger = gofastlog().get_gofast_logger(__name__)
# Parametrize the documentation 
_param_docs = DocstringComponents.from_nested_components(
    core=_core_docs["params"],
)

__all__=[
    'evaluate_model',
    'format_model_score',
    'get_global_score',
    'stats_from_prediction', 
]

@validate_params({"y_true": ['array-like'], 'y_pred': ['array-like']})
def stats_from_prediction(y_true, y_pred, verbose=False):
    """
    Generate statistical summaries and accuracy metrics from actual values (y_true)
    and predicted values (y_pred).

    Parameters
    ----------
    y_true : list or numpy.array
        Actual values.
    y_pred : list or numpy.array
        Predicted values.
    verbose : bool, optional
        If True, print the statistical summary and accuracy metrics.
        Default is False.

    Returns
    -------
    dict
        A dictionary containing statistical measures such 
        as MAE, MSE, RMSE, 
        and accuracy (if applicable).

    Examples
    --------
    >>> from gofast.utils.mlutils import stats_from_prediction 
    >>> y_true = [0, 1, 1, 0, 1]
    >>> y_pred = [0, 1, 0, 0, 1]
    >>> stats_from_prediction(y_true, y_pred, verbose=True)
    """
    
    # Calculating statistics
    check_consistent_length(y_true, y_pred )
    stats = {
        'mean': np.mean(y_pred),
        'median': np.median(y_pred),
        'std_dev': np.std(y_pred),
        'min': np.min(y_pred),
        'max': np.max(y_pred)
    }
    # add the metric stats 
    stats =dict ({
            'MAE': mean_absolute_error(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        }, **stats, 
        )
    # Adding accuracy for classification tasks
    # Check if y_true and y_pred are categories task 
    if is_classification_task(y_true, y_pred ): 
    # if all(map(lambda x: x in [0, 1], y_true + y_pred)): #binary 
        stats['Accuracy'] = accuracy_score(y_true, y_pred)

    # Printing the results if verbose is True
    summary = MetricFormatter(
        title="Prediction Summary", descriptor="PredictStats", 
        **stats)
    if verbose:
        print(summary)
       
    return summary

def get_global_score(
    cvres: Dict[str, ArrayLike],
    ignore_convergence_problem: bool = False
) -> Tuple[float, float]:
    """
    Retrieve the global mean and standard deviation of test scores from 
    cross-validation results.

    This function computes the overall mean and standard deviation of test 
    scores from the results of cross-validation. It can also handle situations 
    where convergence issues might have occurred during model training, 
    depending on the `ignore_convergence_problem` flag.

    Parameters
    ----------
    cvres : Dict[str, np.ndarray]
        A dictionary containing the cross-validation results. Expected to have 
        keys 'mean_test_score' and 'std_test_score', with each key mapping to 
        an array of scores.
    ignore_convergence_problem : bool, default=False
        If True, ignores NaN values that might have resulted from convergence 
        issues during model training while calculating the mean. If False, NaN 
        values contribute to the final mean as NaN.

    Returns
    -------
    Tuple[float, float]
        A tuple containing two float values:
        - The first element is the mean of the test scores across all 
          cross-validation folds.
        - The second element is the mean of the standard deviations of the 
          test scores across all cross-validation folds.

    Examples
    --------
    >>> from sklearn.model_selection import cross_val_score
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> clf = DecisionTreeClassifier()
    >>> scores = cross_val_score(clf, iris.data, iris.target, cv=5,
    ...                          scoring='accuracy', return_train_score=True)
    >>> cvres = {'mean_test_score': scores, 'std_test_score': np.std(scores)}
    >>> mean_score, mean_std = get_global_score(cvres)
    >>> print(f"Mean score: {mean_score}, Mean standard deviation: {mean_std}")

    Notes
    -----
    - The function is primarily designed to be used with results obtained from 
      scikit-learn's cross-validation functions like `cross_val_score`.
    - It is assumed that `cvres` contains keys 'mean_test_score' and 
      'std_test_score'.
    """
    if ignore_convergence_problem:
        mean_score = np.nanmean(cvres.get('mean_test_score'))
        mean_std = np.nanmean(cvres.get('std_test_score'))
    else:
        mean_score = np.mean( cvres.get('mean_test_score'))
        mean_std = np.mean(cvres.get('std_test_score'))

    return mean_score, mean_std


def format_model_score(
    model_score: Union[float, Dict[str, float]] = None,
    selected_estimator: Optional[str] = None
) -> None:
    """
    Formats and prints model scores.

    Parameters
    ----------
    model_score : float or Dict[str, float], optional
        The model score or a dictionary of model scores with estimator 
        names as keys.
    selected_estimator : str, optional
        Name of the estimator to format the score for. Used only if 
        `model_score` is a float.

    Example
    -------
    >>> from gofast.utils.mlutils import format_model_score
    >>> format_model_score({'DecisionTreeClassifier': 0.26, 'BaggingClassifier': 0.13})
    >>> format_model_score(0.75, selected_estimator='RandomForestClassifier')
    """

    print('-' * 77)
    if isinstance(model_score, dict):
        for estimator, score in model_score.items():
            formatted_score = round(score * 100, 3)
            print(f'> {estimator:<30}:{"Score":^10}= {formatted_score:^10} %')
    elif isinstance(model_score, float):
        estimator_name = selected_estimator if selected_estimator else 'Unknown Estimator'
        formatted_score = round(model_score * 100, 3)
        print(f'> {estimator_name:<30}:{"Score":^10}= {formatted_score:^10} %')
    else:
        print('Invalid model score format. Please provide a float or'
              ' a dictionary of scores.')
    print('-' * 77)
    
@validate_params ({
    'model': [HasMethods (['fit', 'predict']), None], 
    'X': ['array-like', None], 
    'Xt':['array-like', None], 
    'y': ['array-like', None], 
    'yt':['array-like', None], 
    'y_pred':['array-like', None], 
    'scorer': [ str, callable ], 
    'eval': [bool] 
    }
 )
def evaluate_model(
    model: Optional[_F] = None,
    X: Optional[Union[NDArray, DataFrame]] = None,
    Xt: Optional[Union[NDArray, DataFrame]] = None,
    y: Optional[Union[NDArray, Series]] = None, 
    yt: Optional[Union[NDArray, Series]] = None,
    y_pred: Optional[Union[NDArray, Series]] = None,
    scoring: Union[str, _F] = 'accuracy_score',
    eval: bool = False,
    **kws: Any
) -> Union[Tuple[Optional[Union[NDArray, Series]], Optional[float]],
           Optional[Union[NDArray, Series]]]:
    """
    Evaluates a predictive model's performance or the effectiveness of 
    predictions using a specified scoring metric.

    Parameters
    ----------
    {params.core.model} 
    {params.core.X}
    {params.core.Xt}
    {params.core.y} 
    {params.core.yt}

    y_pred : array-like, optional
        The predicted labels or values generated by the model, or provided 
        directly as input. This can be a one-dimensional array or series of 
        predicted values for classification or regression tasks. If not provided, 
        the model's predictions will be computed using the `model.predict` method. 

        If `eval=True`, `y_pred` is compared against the true labels (`yt`) using 
        the specified scoring metric (e.g., accuracy, precision, etc.).

        Note that if `y_pred` is provided directly, the `model`'s `fit` and `
        predict` methods will not be called. This is useful when you want to 
        evaluate pre-computed predictions, such as when predictions are already 
        available from another process or dataset.

    {params.core.scoring}
    
    eval : bool, optional, default=False
        If True, the function will evaluate the model's predictions (`y_pred`) 
        against the true labels (`yt`) using the specified scoring function 
        (`scorer`). The score, computed by the `scorer`, is returned along
        with the predictions.
        
        If False, only the predictions are returned without evaluation. 
        This is useful when you want to obtain the model's predictions without
        performing any evaluation or scoring.

    **kws : Any
        Additional keyword arguments to pass to the scoring function.

    Returns
    -------
    predictions : np.ndarray or pd.Series
        The predicted labels or probabilities.
    score : float, optional
        The score of the predictions based on `scorer`. Only returned if 
        `eval` is True.

    Raises
    ------
    ValueError
        If required arguments are missing or if the provided arguments are invalid.
    TypeError
        If `scorer` is not a recognized scoring function.

    Examples
    --------
    >>> import numpy as np 
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.linear_model import LogisticRegression
    >>> from gofast.utils.mlutils import evaluate_model
    >>> iris = load_iris()
    >>> X, y = iris.data, iris.target
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    >>> model = LogisticRegression()
    
    # Case 1: Model predicts y_pred using model.fit and model.predict
    >>> y_pred, score = evaluate_model(model=model, X=X_train, Xt=X_test,
    ...                                y=y_train, yt=y_test, eval=True)
    >>> print(f'Score: {{score:.2f}}')

    # Case 2: Provide y_pred directly (e.g., pre-computed predictions)
    >>> y_pred = np.array([0, 1, 2, 2, 0])  # Pre-computed predictions
    >>> y_test_2 = y_test [: len(y_pred)]
    >>> y_pred, score = evaluate_model(y_pred=y_pred, yt=y_test_2,
                                       scoring='accuracy',
    ...                                eval=True)
    >>> print(f'Accuracy: {{score:.2f}}')
    
    >>> y_pred, score = evaluate_model(model=model, X=X_train, Xt=X_test,
    ...                                y=y_train, yt=y_test, eval=True)
    >>> print(f'Score: {{score:.2f}}')
    
    >>> # Providing predictions directly
    >>> y_pred, _ = evaluate_model(y_pred=y_pred, yt=y_test, 
                                   scoring='accuracy',
    ...                            eval=True)
    >>> print(f'Accuracy: {{score:.2f}}')
    
    # Case 3: Perform evaluation using the specified scorer
    >>> y_pred, score = evaluate_model(model=model, X=X_train, Xt=X_test,
    ...                                y=y_train, yt=y_test, eval=True)
    >>> print(f'Score: {{score:.2f}}')

    # Case 4: Get predictions without evaluation
    >>> y_pred = evaluate_model(model=model, X=X_train, Xt=X_test,
    ...                         y=y_train, eval=False)
    >>> print(f'Predictions: {{y_pred}}')
    
    """.format (params= _param_docs )
    
    from ..metrics import fetch_scorers
    
    if y_pred is None:
        if model is None or X is None or y is None or Xt is None:
            raise ValueError("Model, X, y, and Xt must be provided when y_pred"
                             " is not provided.")
        if not hasattr(model, 'fit') or not hasattr(model, 'predict'):
            raise TypeError("The provided model does not implement fit and "
                            "predict methods.")
        
        # Check model if is fitted
        try: check_is_fitted(model)
        except: 
            # If the model is not fitted, then fit it with X and y
            if X is not None and y is not None:
                if hasattr(X, 'ndim') and X.ndim == 1:
                    X = X.reshape(-1, 1)
                model.fit(X, y)
            else:
                raise ValueError("Model is not fitted, and no training data"
                                 " (X, y) were provided.")
        y_pred = model.predict(Xt)

    if eval:
        if yt is None:
            raise ValueError("yt must be provided when eval is True.")
        if not isinstance(scoring, (str, callable)):
            raise TypeError("scorer must be a string or a callable,"
                            f" got {type(scoring).__name__}.")
        if isinstance (scoring, str): 
            scoring= fetch_scorers (scoring) 
        # score_func = get_scorer(scorer, include_sklearn= True )
        score = scoring(yt, y_pred, **kws)
        return y_pred, score

    return y_pred
