# -*- coding: utf-8 -*-

from packaging.version import Version, parse
import sklearn
from sklearn.utils import resample
from sklearn.utils.validation import check_is_fitted as sklearn_check_is_fitted
from sklearn.metrics import get_scorer

# Determine the installed scikit-learn version
SKLEARN_VERSION = parse(sklearn.__version__)

# Feature and compatibility flags
SKLEARN_LT_0_22 = SKLEARN_VERSION < Version("0.22.0")
SKLEARN_LT_0_23 = SKLEARN_VERSION < Version("0.23.0")
SKLEARN_LT_0_24 = SKLEARN_VERSION < Version("0.24.0")

__all__ = [
    "resample",
    "train_test_split",
    "get_scorer",
    "get_feature_names",
    "get_feature_names_out", 
    "get_transformers_from_column_transformer",
    "check_is_fitted",
    "adjusted_mutual_info_score", 
    "SKLEARN_LT_0_22", 
    "SKLEARN_LT_0_23", 
    "SKLEARN_LT_0_24"
]

def get_feature_names(estimator, *args, **kwargs):
    """
    Compatibility function for fetching feature names from an estimator.
    
    Parameters:
    - estimator : estimator object
        Scikit-learn estimator from which to get feature names.
    - *args : Additional positional arguments for the get_feature_names method.
    - **kwargs : Additional keyword arguments for the get_feature_names method.

    Returns:
    - feature_names : list
        List of feature names.
    """
    if hasattr(estimator, 'get_feature_names_out'):
        # For versions of scikit-learn that support get_feature_names_out
        return estimator.get_feature_names_out(*args, **kwargs)
    elif hasattr(estimator, 'get_feature_names'):
        # For older versions of scikit-learn
        return estimator.get_feature_names(*args, **kwargs)
    else:
        raise AttributeError(
            "The estimator does not have a method to get feature names.")

def get_feature_names_out(estimator, *args, **kwargs):
    """
    Compatibility function for fetching feature names from an estimator, using
    get_feature_names_out for scikit-learn versions that support it.
    
    Parameters:
    - estimator : estimator object
        Scikit-learn estimator from which to get feature names.
    - *args : Additional positional arguments for the get_feature_names_out method.
    - **kwargs : Additional keyword arguments for the get_feature_names_out method.

    Returns:
    - feature_names_out : list
        List of feature names.
    """
    return get_feature_names(estimator, *args, **kwargs)

def get_transformers_from_column_transformer(ct):
    """
    Compatibility function to get transformers from a ColumnTransformer object.
    
    Parameters:
    - ct : ColumnTransformer
        A fitted ColumnTransformer instance.

    Returns:
    - transformers : list of tuples
        List of (name, transformer, column(s)) tuples.
    """
    if hasattr(ct, 'transformers_'):
        return ct.transformers_
    else:
        raise AttributeError(
            "The ColumnTransformer instance does not have a 'transformers_' attribute.")

def check_is_fitted(estimator, attributes=None, msg=None, all_or_any=all):
    """
    Compatibility wrapper for scikit-learn's check_is_fitted function.
    
    Parameters:
    - estimator : estimator instance
        The estimator to check.
    - attributes : str or list of str, optional
        The attributes to check for.
    - msg : str, optional
        The message to display on failure.
    - all_or_any : callable, optional
        all or any; whether all or any of the given attributes must be present.

    Returns:
    - None
    """
    return sklearn_check_is_fitted(estimator, attributes, msg, all_or_any)

def adjusted_mutual_info_score(
        labels_true, labels_pred, average_method='arithmetic'):
    """
    Compatibility function for adjusted_mutual_info_score with the 
    average_method parameter.

    Parameters:
    - labels_true : array-like of shape (n_samples,)
        Ground truth class labels.
    - labels_pred : array-like of shape (n_samples,)
        Cluster labels to evaluate.
    - average_method : str, default='arithmetic'
        The method to average the mutual information scores. Versions of
        scikit-learn before 0.22.0 do not have this parameter and use 'arithmetic'
        by default.

    Returns:
    - ami : float
       Adjusted Mutual Information Score.
    """
    from sklearn.metrics import adjusted_mutual_info_score as ami_score
    if SKLEARN_LT_0_22:
        return ami_score(labels_true, labels_pred)
    else:
        return ami_score(labels_true, labels_pred, average_method=average_method)
    
def fetch_openml(*args, **kwargs):
    """
    Compatibility function for fetch_openml to ensure consistent return type.

    Parameters:
    - args, kwargs: Arguments and keyword arguments for 
    sklearn.datasets.fetch_openml.

    Returns:
    - data : Bunch
        Dictionary-like object with all the data and metadata.
    """
    from sklearn.datasets import fetch_openml
    if 'as_frame' not in kwargs and not SKLEARN_LT_0_24:
        kwargs['as_frame'] = True
    return fetch_openml(*args, **kwargs)

def plot_confusion_matrix(estimator, X, y_true, *args, **kwargs):
    """
    Compatibility function for plot_confusion_matrix across scikit-learn versions.

    Parameters:
    - estimator : estimator instance
        Fitted classifier.
    - X : array-like of shape (n_samples, n_features)
        Input values.
    - y_true : array-like of shape (n_samples,)
        True labels for X.

    Returns:
    - display : ConfusionMatrixDisplay
        Object that stores the confusion matrix display.
    """
    try:
        from sklearn.metrics import plot_confusion_matrix
    except ImportError:
        # Assume older version without plot_confusion_matrix
        # Implement fallback or raise informative error
        raise NotImplementedError(
            "plot_confusion_matrix not available in your sklearn version.")
    return plot_confusion_matrix(estimator, X, y_true, *args, **kwargs)

def train_test_split(*args, **kwargs):
    """
    Compatibility wrapper for train_test_split to ensure consistent behavior.

    Parameters:
    - args, kwargs: Arguments and keyword arguments for 
    sklearn.model_selection.train_test_split.
    """
    from sklearn.model_selection import train_test_split
    if 'shuffle' not in kwargs:
        kwargs['shuffle'] = True
    return train_test_split(*args, **kwargs)

__all__.extend([
    "fetch_openml",
    "plot_confusion_matrix",
])

