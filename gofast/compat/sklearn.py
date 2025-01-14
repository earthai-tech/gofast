# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Provides compatibility utilities for different versions of
scikit-learn (sklearn). It includes functions and feature flags that
ensure smooth operation across various sklearn versions, handling
breaking changes and deprecated features. The module includes 
resampling utilities, scorer functions, and compatibility checks.

Key functionalities include:
- Resampling with sklearn's `resample`
- Validation with `check_is_fitted`
- Scorer retrieval with `get_scorer`
- Feature and compatibility flags for sklearn versions

The module ensures compatibility with sklearn versions less than 
0.22.0, 0.23.0, and 0.24.0.

Attributes
----------
SKLEARN_VERSION : packaging.version.Version
    The installed scikit-learn version.
SKLEARN_LT_0_22 : bool
    True if the installed scikit-learn version is less than 0.22.0.
SKLEARN_LT_0_23 : bool
    True if the installed scikit-learn version is less than 0.23.0.
SKLEARN_LT_0_24 : bool
    True if the installed scikit-learn version is less than 0.24.0.

Functions
---------
resample
    Resample arrays or sparse matrices in a consistent way.
get_scorer
    Get a scorer from string.
check_is_fitted
    Perform is_fitted validation for sklearn models.
"""
import warnings
from packaging.version import Version, parse
import inspect
import numpy as np
import sklearn
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
from sklearn.utils._param_validation import validate_params as sklearn_validate_params
from sklearn.utils._param_validation import Interval as sklearn_Interval 
from sklearn.utils._param_validation import StrOptions, HasMethods, Hidden
from sklearn.utils._param_validation import InvalidParameterError 
from sklearn.utils import resample
from sklearn.utils.validation import check_is_fitted as sklearn_check_is_fitted
from sklearn.metrics import get_scorer

# Determine the installed scikit-learn version
SKLEARN_VERSION = parse(sklearn.__version__)

# Feature and compatibility flags
SKLEARN_LT_0_22 = SKLEARN_VERSION < Version("0.22.0")
SKLEARN_LT_0_23 = SKLEARN_VERSION < Version("0.23.0")
SKLEARN_LT_0_24 = SKLEARN_VERSION < Version("0.24.0")
SKLEARN_LT_1_3 = SKLEARN_VERSION < parse("1.3.0")

__all__ = [
    "Interval", 
    "resample",
    "train_test_split",
    "get_scorer",
    "type_of_target", 
    "get_feature_names",
    "get_feature_names_out", 
    "get_transformers_from_column_transformer",
    "check_is_fitted",
    "adjusted_mutual_info_score", 
    "get_sgd_loss_param", 
    "validate_params", 
    "InvalidParameterError", 
    "StrOptions", 
    "HasMethods", 
    "Hidden", 
    "OneHotEncoder", 
    "SKLEARN_LT_0_22", 
    "SKLEARN_LT_0_23", 
    "SKLEARN_LT_0_24"
]


class OneHotEncoder(SklearnOneHotEncoder):
    """
    A compatibility wrapper around scikit-learn's OneHotEncoder
    that manages 'sparse' vs. 'sparse_output' parameters across
    different scikit-learn versions.

    For scikit-learn < 1.2:
      - The 'sparse' parameter is recognized, while 'sparse_output'
        is not. If a user supplies ``sparse_output``, it is mapped
        to ``sparse``. If both are supplied, we prioritize
        ``sparse_output``.

    For scikit-learn >= 1.2:
      - The 'sparse' parameter is deprecated, replaced by
        ``sparse_output``. If a user supplies ``sparse``, it is
        mapped to ``sparse_output``, preventing deprecation warnings.

    Notes
    -----
    This wrapper helps avoid warnings such as:

    .. code-block:: none

       FutureWarning: `sparse` was renamed to `sparse_output` in version 1.2
       and will be removed in 1.4. `sparse_output` is ignored unless you
       leave `sparse` to its default value.

    Usage is identical to the scikit-learn OneHotEncoder, with the same
    parameters. The wrapper internally adjusts parameters to be compatible
    with the detected scikit-learn version. For full parameter details,
    see the official scikit-learn OneHotEncoder documentation.
    """

    def __init__(
        self,
        categories = "auto",
        drop = None,
        sparse = "deprecated",
        sparse_output = None,
        dtype = np.float64,
        handle_unknown = "error",
        min_frequency = None,
        max_categories = None
        # we can add other parameters introduced in scikit-learn
        # for future or past versions if needed.
    ):
        # Determine scikit-learn version at runtime
        sk_version = parse(sklearn.__version__)

        # If scikit-learn < 1.2: rename 'sparse_output' -> 'sparse'
        # if user provided it. If both are set, prioritize 'sparse_output'.
        if sk_version < parse("1.2"):
            if sparse == "deprecated":
                sparse = sparse_output
            if sparse_output is not None and sparse != "deprecated":
                warnings.warn(
                    "Both 'sparse' and 'sparse_output' are set. "
                    "Using 'sparse_output' as the final value for older "
                    "scikit-learn versions (<1.2)."
                )
                sparse = sparse_output
            if sparse == "deprecated":
                sparse = True

            super().__init__(
                categories = categories,
                drop = drop,
                sparse = sparse,
                dtype = dtype,
                handle_unknown = handle_unknown,
                min_frequency = min_frequency,
                max_categories = max_categories
            )

        # If scikit-learn >= 1.2: rename 'sparse' -> 'sparse_output'
        # if user provided it. If both are set, prioritize 'sparse_output'.
        else:
            if sparse_output is None and sparse != "deprecated":
                sparse_output = sparse
            if sparse_output is None and sparse == "deprecated":
                sparse_output = True
            elif sparse_output is not None and sparse != "deprecated":
                warnings.warn(
                    "Both 'sparse' and 'sparse_output' are set. "
                    "Using 'sparse_output' as the final value for newer "
                    "scikit-learn versions (>=1.2)."
                )

            super().__init__(
                categories = categories,
                drop = drop,
                sparse_output = sparse_output,
                dtype = dtype,
                handle_unknown = handle_unknown,
                min_frequency = min_frequency,
                max_categories = max_categories
            )


class Interval:
    """
    Compatibility wrapper for scikit-learn's `Interval` class to handle 
    versions that do not include the `inclusive` argument.
    
    Parameters
    ----------
    *args : tuple
        Positional arguments passed to the `Interval` class, typically 
        the expected data types and the range boundaries for the validation 
        interval.
    
    inclusive : bool, optional
        Specifies whether the interval includes its bounds. Only supported 
        in scikit-learn versions that accept the `inclusive` parameter. If 
        `True`, the interval includes the bounds. Default is `None` for 
        older versions where this argument is not available.
    
    closed : str, optional
        Defines how the interval is closed. Can be "left", "right", "both", 
        or "neither". This argument is accepted by both older and newer 
        scikit-learn versions. Default is "left" (includes the left bound, 
        but excludes the right bound).
    
    kwargs : dict
        Additional keyword arguments passed to the `Interval` class for 
        compatibility, including any additional arguments required by the 
        current scikit-learn version.

    Returns
    -------
    Interval
        A compatible `Interval` object based on the scikit-learn version, 
        with or without the `inclusive` argument.
    
    Raises
    ------
    ValueError
        If an unsupported version of scikit-learn is used or the parameters 
        are not valid for the given version.
    
    Notes
    -----
    This class provides a compatibility layer for creating `Interval` 
    objects in different versions of scikit-learn. The `inclusive` argument 
    was introduced in newer versions, so this class removes it if not 
    supported in older versions. 
    
    If you are using scikit-learn versions that support the `inclusive` 
    argument (e.g., version 1.2 or later), it will be included in the call 
    to `Interval`. Otherwise, the argument will be excluded.
    
    Examples
    --------
    In newer scikit-learn versions (e.g., >=1.2), you can include the 
    `inclusive` parameter:
    
    >>> from numbers import Integral
    >>> from gofast.compat.sklearn import Interval
    >>> interval = Interval(Integral, 1, 10, closed="left", inclusive=True)
    >>> interval
    
    In older versions of scikit-learn that don't support `inclusive`, it 
    will automatically be removed:
    
    >>> interval = Interval(Integral, 1, 10, closed="left")
    >>> interval
    
    See Also
    --------
    sklearn.utils._param_validation.Interval : Original scikit-learn `Interval` 
        class used for parameter validation.
    
    References
    ----------
    .. [1] Pedregosa, F. et al. (2011). "Scikit-learn: Machine Learning in 
       Python." *Journal of Machine Learning Research*, 12, 2825-2830.
    
    .. [2] Buitinck, L., Louppe, G., Blondel, M., et al. (2013). "API design 
       for machine learning software: experiences from the scikit-learn 
       project." *arXiv preprint arXiv:1309.0238*.
    """
    
    def __new__(cls, *args, **kwargs):
        """
        Creates a compatible `Interval` object based on the scikit-learn 
        version.
        
        Parameters
        ----------
        *args : tuple
            Positional arguments for the `Interval` class.
        kwargs : dict
            Keyword arguments, including `inclusive` if supported by the 
            scikit-learn version.
        
        Returns
        -------
        sklearn.utils._param_validation.Interval
            A compatible `Interval` object.
        """
        # Check if 'inclusive' is a parameter in the __init__ method of 
        # sklearn_Interval
        signature = inspect.signature(sklearn_Interval.__init__)
        if 'inclusive' in signature.parameters:
            # 'inclusive' is supported, use kwargs as is
            return sklearn_Interval(*args, **kwargs)
        else:
            # 'inclusive' not supported, remove it from kwargs if present
            kwargs.pop('inclusive', None)
            return sklearn_Interval(*args, **kwargs)


def type_of_target(y):
    """
    Determine the type of the target variable.

    This function identifies the nature of the target variable ``y`` and
    returns a string indicating its type. It leverages scikit-learn's
    `type_of_target` if available; otherwise, it falls back to gofast's
    implementation.

    Parameters
    ----------
    y : array-like
        The target variable to classify. It can be one of the following
        types:
        - List
        - Tuple
        - NumPy array
        - Pandas Series
        - Pandas DataFrame
        - Other array-like structures.

        The function assesses the structure and contents of ``y`` to
        determine its type, such as binary, multiclass, multilabel-indicator,
        continuous, or continuous-multioutput.

    Returns
    -------
    target_type : str
        A string representing the type of the target variable. Possible
        return values include:
        - `'binary'`: Binary classification.
        - `'multiclass'`: Multiclass classification.
        - `'multilabel-indicator'`: Multilabel classification with
          binary indicators.
        - `'continuous'`: Continuous target (regression).
        - `'continuous-multioutput'`: Multioutput regression.
        - `'unknown'`: Unknown or unsupported target type.

    Notes
    -----
    The `type_of_target` function classifies the target variable based on its
    structure and the number of unique classes. It is essential for determining
    the appropriate machine learning algorithms to apply.

    The classification rules are as follows:

    .. math::
        \text{If } y \text{ is 2D and each column has at most two unique values,} \\
        \text{then it is 'multilabel-indicator'}. \\
        \text{Else, if } y \text{ has two unique values, it is 'binary'}. \\
        \text{Else, if } y \text{ has more than two unique values, it is 'multiclass'}. \\
        \text{If } y \text{ contains continuous values, it is 'continuous'}. \\
        \text{For multioutput regression, it is 'continuous-multioutput'}.

    Examples
    --------
    >>> from gofast.compat.sklearn import type_of_target

    Binary classification:

    >>> y_binary = [0, 1, 0, 1, 1]
    >>> type_of_target(y_binary)
    'binary'

    Multiclass classification:

    >>> y_multiclass = [0, 1, 2, 1, 0]
    >>> type_of_target(y_multiclass)
    'multiclass'

    Multilabel-indicator:

    >>> y_multilabel = [[1, 0], [0, 1], [1, 1], [0, 0]]
    >>> type_of_target(y_multilabel)
    'multilabel-indicator'

    Continuous (regression):

    >>> y_continuous = [0.5, 1.2, 3.3, 2.1, 5.5]
    >>> type_of_target(y_continuous)
    'continuous'

    Continuous multioutput (multioutput regression):

    >>> y_multioutput = [[0.5, 1.0], [1.2, 0.8], [3.3, 2.2],
    ...                 [2.1, 1.5], [5.5, 3.3]]
    >>> type_of_target(y_multioutput)
    'continuous-multioutput'

    Unknown type:

    >>> y_unknown = ['a', 'b', 'a', 'c', 'b']
    >>> type_of_target(y_unknown)
    'unknown'

    Empty target:

    >>> y_empty = []
    >>> type_of_target(y_empty)
    'unknown'

    Notes
    -----
    The function prioritizes scikit-learn's implementation for its robustness
    and comprehensive type classification. The fallback to gofast's
    ``type_of_target`` ensures compatibility in environments where scikit-learn
    is unavailable.

    See also
    --------
    `sklearn.utils.multiclass.type_of_target` : scikit-learn's implementation
    `gofast.core.utils.type_of_target` : gofast's fallback implementation

    References
    ----------
    .. [1] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B.,
           Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V.,
           Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M.,
           & Duchesnay, É. (2011). Scikit-learn: Machine Learning in Python.
           Journal of Machine Learning Research, 12, 2825–2830.

    .. [2] Gofast Documentation. Available at https://gofast.readthedocs.io/en/latest/

    """
    # Attempt to import type_of_target from scikit-learn
    try:
        from sklearn.utils.multiclass import type_of_target as skl_type_of_target
        return skl_type_of_target(y)
    except ImportError:
        # Fallback to gofast's type_of_target if scikit-learn is not available
        try:
            from gofast.core.utils import type_of_target as gofast_type_of_target
            return gofast_type_of_target(y)
        except ImportError:
            # If both imports fail, raise an ImportError
            raise ImportError(
                "Neither scikit-learn nor gofast.core.utils could provide "
                "'type_of_target'. Please ensure that scikit-learn is installed "
                "or gofast.core.utils contains 'type_of_target'."
            )


def get_sgd_loss_param():
    """Get the correct argument of loss parameter for SGDClassifier 
    based on scikit-learn version.

    This function determines which loss parameter to use for 
    the SGDClassifier depending on the installed version of 
    scikit-learn. In versions 0.24 and newer, the loss parameter 
    should be set to 'log_loss'. In older versions, it should 
    be set to 'log'.

    Returns
    -------
    str
        The appropriate loss parameter for the SGDClassifier.

    Examples
    --------
    >>> loss_param = get_sgd_loss_param()
    >>> print(loss_param)
    'log_loss'  # If using scikit-learn 0.24 or newer

    >>> # Example usage with SGDClassifier
    >>> from sklearn.linear_model import SGDClassifier
    >>> clf = SGDClassifier(loss=get_sgd_loss_param(), max_iter=1000)
    
    Notes
    -----
    This function is useful for maintaining compatibility 
    with different versions of scikit-learn, ensuring that 
    the model behaves as expected regardless of the library 
    version being used.
    """
    
    # Use 'log' for older versions if SKLEARN_LT_1_3
    return 'log' if SKLEARN_LT_1_3 else 'log_loss'


def validate_params(params, *args, prefer_skip_nested_validation=True, **kwargs):
    """
    Compatibility wrapper for scikit-learn's `validate_params` function
    to handle versions that require the `prefer_skip_nested_validation` argument,
    with a default value that can be overridden by the user.

    Parameters
    ----------
    params : dict
        A dictionary that defines the validation rules for the parameters.
        Each key in the dictionary should represent the name of a parameter
        that requires validation, and its associated value should be a list 
        of expected types (e.g., ``[int, str]``). 
        The function will validate that the parameters passed to the 
        decorated function match the specified types.
        
        For example, if `params` is:
        
        .. code-block:: python

            params = {
                'step_name': [str],
                'n_trials': [int]
            }

        Then, the `step_name` parameter must be of type `str`, and 
        `n_trials` must be of type `int`.
    
    prefer_skip_nested_validation : bool, optional
        If ``True`` (the default), the function will attempt to skip 
        nested validation of complex objects (e.g., dictionaries or 
        lists), focusing only on the top-level structure. This option 
        can be useful for improving performance when validating large, 
        complex objects where deep validation is unnecessary.
        
        Set to ``False`` to enable deep validation of nested objects.

    *args : list
        Additional positional arguments to pass to `validate_params`.

    **kwargs : dict
        Additional keyword arguments to pass to `validate_params`. 
        These can include options such as `prefer_skip_nested_validation` 
        and other custom behavior depending on the context of validation.
    
    Returns
    -------
    function
        Returns the `validate_params` function with appropriate argument 
        handling for scikit-learn's internal parameter validation. This 
        function can be used as a decorator to ensure type safety and 
        parameter consistency in various machine learning pipelines.

    Notes
    -----
    The `validate_params` function provides a robust way to enforce 
    type and structure validation on function arguments, especially 
    in the context of machine learning workflows. By ensuring that 
    parameters adhere to a predefined structure, the function helps 
    prevent runtime errors due to unexpected types or invalid argument 
    configurations.
    
    In the case where a user sets `prefer_skip_nested_validation` to 
    ``True``, the function optimizes the validation process by skipping 
    nested structures (e.g., dictionaries or lists), focusing only on 
    validating the top-level parameters. When set to ``False``, a deeper 
    validation process occurs, checking every element within nested 
    structures.

    The validation process can be represented mathematically as:

    .. math::

        V(p_i) = 
        \begin{cases}
        1, & \text{if} \, \text{type}(p_i) \in T(p_i) \\
        0, & \text{otherwise}
        \end{cases}

    where :math:`V(p_i)` is the validation function for parameter :math:`p_i`,
    and :math:`T(p_i)` represents the set of expected types for :math:`p_i`. 
    The function returns 1 if the parameter matches the expected type, 
    otherwise 0.

    Examples
    --------
    >>> from gofast.compat.sklearn import validate_params
    >>> @validate_params({
    ...     'step_name': [str],
    ...     'param_grid': [dict],
    ...     'n_trials': [int],
    ...     'eval_metric': [str]
    ... }, prefer_skip_nested_validation=False)
    ... def tune_hyperparameters(step_name, param_grid, n_trials, eval_metric):
    ...     print(f"Hyperparameters tuned for step: {step_name}")
    ... 
    >>> tune_hyperparameters(
    ...     step_name='TrainModel', 
    ...     param_grid={'learning_rate': [0.01, 0.1]}, 
    ...     n_trials=5, 
    ...     eval_metric='accuracy'
    ... )
    Hyperparameters tuned for step: TrainModel

    See Also
    --------
    sklearn.utils.validate_params : Original scikit-learn function for parameter 
        validation. Refer to scikit-learn documentation for more detailed information.

    References
    ----------
    .. [1] Pedregosa, F. et al. (2011). "Scikit-learn: Machine Learning in Python."
       *Journal of Machine Learning Research*, 12, 2825-2830.

    .. [2] Buitinck, L., Louppe, G., Blondel, M., et al. (2013). "API design for 
       machine learning software: experiences from the scikit-learn project."
       *arXiv preprint arXiv:1309.0238*.
    """
    # Check if `prefer_skip_nested_validation` is required by inspecting the signature
    sig = inspect.signature(sklearn_validate_params)
    if 'prefer_skip_nested_validation' in sig.parameters:
        # Pass the user's choice or default for `prefer_skip_nested_validation`
        kwargs['prefer_skip_nested_validation'] = prefer_skip_nested_validation
    
    # Call the actual validate_params with appropriate arguments
    return sklearn_validate_params(params, *args, **kwargs)


def get_column_transformer_feature_names(column_transformer, input_features=None):
    """
    Get feature names from a ColumnTransformer.
    
    Parameters:
    - column_transformer : ColumnTransformer
        The ColumnTransformer object.
    - input_features : list of str, optional
        List of input feature names.
    
    Returns:
    - feature_names : list of str
        List of feature names generated by the transformers in the ColumnTransformer.
    """
    output_features = []

    # Ensure input_features is a list; if not provided, assume numerical column indices
    if input_features is None:
        input_features = list(range(column_transformer._n_features))
    
    for transformer_name, transformer, column in column_transformer.transformers_:
        if transformer == 'drop' or (
                hasattr(transformer, 'remainder') and transformer.remainder == 'drop'):
            continue

        # Resolve actual column names/indices
        actual_columns = [input_features[c] for c in column] if isinstance(
            column, (list, slice)) else [input_features[column]]

        if hasattr(transformer, 'get_feature_names_out'):
            # For transformers that support get_feature_names_out
            if hasattr(transformer, 'feature_names_in_'):
                transformer.feature_names_in_ = actual_columns
            transformer_features = transformer.get_feature_names_out()
        elif hasattr(transformer, 'get_feature_names'):
            # For transformers that support get_feature_names
            transformer_features = transformer.get_feature_names()
        else:
            # Default behavior for transformers without get_feature_names methods
            transformer_features = [f"{transformer_name}__{i}" for i in range(
                transformer.transform(column).shape[1])]
        
        output_features.extend(transformer_features)

    return output_features

def get_column_transformer_feature_names2(column_transformer, input_features=None):
    """
    Get feature names from a ColumnTransformer.
    
    Parameters:
    - column_transformer : ColumnTransformer
        The ColumnTransformer object.
    - input_features : list of str, optional
        List of input feature names.
    
    Returns:
    - feature_names : list of str
        List of feature names generated by the transformers in the ColumnTransformer.
    """
    output_features = []

    for transformer_name, transformer, column in column_transformer.transformers_:
        if transformer == 'drop' or (
                hasattr(transformer, 'remainder') and transformer.remainder == 'drop'):
            continue

        if hasattr(transformer, 'get_feature_names_out'):
            # For transformers that support get_feature_names_out
            if input_features is not None and hasattr(transformer, 'feature_names_in_'):
                # Adjust for the case where column is a list of column names or indices
                transformer_feature_names_in = [input_features[col] if isinstance(
                    column, list) else input_features[column] for col in column] if isinstance(
                        column, list) else [input_features[column]]
                transformer.feature_names_in_ = transformer_feature_names_in
            transformer_features = transformer.get_feature_names_out()
        elif hasattr(transformer, 'get_feature_names'):
            # For transformers that support get_feature_names
            transformer_features = transformer.get_feature_names()
        else:
            # Default behavior for transformers without get_feature_names methods
            transformer_features = [f"{transformer_name}__{i}" for i in range(
                transformer.transform(column).shape[1])]
        
        output_features.extend(transformer_features)

    return output_features

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


def get_transformer_feature_names(transformer, input_features=None):
    """
    Compatibility function to get feature names from transformers like OneHotEncoder
    in scikit-learn, taking into account changes in method names across versions.

    Parameters:
    - transformer : sklearn transformer instance
        The transformer instance from which to get feature names.
    - input_features : list of str, optional
        List of input feature names to the transformer. Required for transformers
        that support `get_feature_names` method which requires input feature names.

    Returns:
    - feature_names : list of str
        List of feature names generated by the transformer.
    """
    if hasattr(transformer, 'get_feature_names_out'):
        # Use get_feature_names_out if available (preferable in newer versions)
        return transformer.get_feature_names_out(input_features)
    elif hasattr(transformer, 'get_feature_names'):
        # Fallback to get_feature_names for compatibility with older versions
        if input_features is not None:
            return transformer.get_feature_names(input_features)
        else:
            return transformer.get_feature_names()
    else:
        # Raise error if neither method is available
        raise AttributeError(
            f"{transformer.__class__.__name__} does not support feature name extraction.")

def get_pipeline_feature_names(pipeline, input_features=None):
    """
    Compatibility function to safely extract feature names from a pipeline,
    especially when it contains transformers like SimpleImputer that do not
    support get_feature_names_out directly.

    Parameters:
    - pipeline : sklearn Pipeline instance
        The pipeline instance from which to extract feature names.
    - input_features : list of str, optional
        List of input feature names to the pipeline. Required for transformers
        that support `get_feature_names` or `get_feature_names_out` methods which
        require input feature names.

    Returns:
    - feature_names : list of str
        List of feature names generated by the pipeline.
    """
    import numpy as np 
    if input_features is None:
        input_features = []

    # Initialize with input features
    current_features = np.array(input_features)
    
    # Iterate through transformers in the pipeline
    for name, transformer in pipeline.steps:
        if hasattr(transformer, 'get_feature_names_out'):
            # Transformer supports get_feature_names_out
            current_features = transformer.get_feature_names_out(current_features)
        elif hasattr(transformer, 'get_feature_names'):
            # Transformer supports get_feature_names and requires current feature names
            current_features = transformer.get_feature_names(current_features)
        elif hasattr(transformer, 'categories_'):
            # Handle OneHotEncoder separately
            current_features = np.concatenate(transformer.categories_)
        else:
            # For transformers that do not modify feature names 
            # or do not provide a method to get feature names
            continue
    
    # Ensure output is a list of strings
    feature_names = list(map(str, current_features))
    return feature_names


__all__.extend([
    "fetch_openml",
    "plot_confusion_matrix",
    "get_transformer_feature_names", 
    "get_pipeline_feature_names"
])

