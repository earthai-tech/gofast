# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Learning utilities for data transformation, model learning, and inspections.
This module provides tools for data preprocessing, feature engineering, 
and utilities for handling machine learning workflows efficiently.
"""

import re
import copy
import warnings
from collections import Counter
from numbers import Real

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import (
    OneHotEncoder, RobustScaler, OrdinalEncoder, StandardScaler,
    MinMaxScaler, LabelBinarizer, LabelEncoder, Normalizer,
    PolynomialFeatures
)
from sklearn.utils import resample

from ..._gofastlog import gofastlog
from ...api.types import List, Tuple, Any, Dict, Optional, Union, Series
from ...api.types import _F, ArrayLike, NDArray, DataFrame
from ...api.summary import ResultSummary
from ...compat.sklearn import ( 
    get_feature_names, train_test_split, validate_params, 
    Interval, HasMethods, StrOptions, type_of_target
    ) 
from ...compat.pandas import select_dtypes 
from ...core.array_manager import ( 
    to_numeric_dtypes, 
    to_array, to_series, 
    array_preserver, 
    return_if_preserver_failed, 
    )
from ...core.checks import (
    is_in_if, is_iterable, validate_feature, exist_features, 
    check_empty, check_numeric_dtype, validate_ratio
)
from ...core.handlers import get_valid_kwargs, columns_manager
from ...core.io import is_data_readable, to_frame_if 
from ...core.utils import smart_format, contains_delimiter, error_policy 
from ...decorators import SmartProcessor, Dataify
from ..base_utils import select_features, extract_target
from ..deps_utils import ensure_pkg
from ..data_utils import nan_to_na 
from ..validator import (
    _is_numeric_dtype, _is_arraylike_1d, get_estimator_name, check_array,
    check_consistent_length, is_frame, build_data_if, 
    check_mixed_data_types, validate_strategy,
    validate_numeric 
)
from .feature_selection import bi_selector 

# Logger Configuration
_logger = gofastlog().get_gofast_logger(__name__)

__all__ = [
    'bin_counting',
    'build_data_preprocessor',
    'discretize_categories',
    'generate_dirichlet_features', 
    'generate_proxy_feature', 
    'handle_imbalance',
    'make_pipe',
    'one_click_prep',
    'process_data_types', 
    'resampling',
    'soft_encoder',
    'soft_imputer',
    'soft_scaler',
    'handle_minority_classes', 
    'encode_target', 
]


@is_data_readable(data_to_read= 'data') 
@validate_params ({
    'target': [str, 'array-like'], 
    'data': ['array-like', None ], 
    'tech': [StrOptions({
            'drop',
            'minimum_samples',
            'oversampling',
            'undersampling' , 
            'combine',
            'augment', 
            'anomaly_detection',
            'tomeklins',
            'enn' 
        }),
        None ], 
    'strategy': [StrOptions({
        'random_over', 
         'random_under', 
         "adasyn", "smote", 
         "cluster_centroids"
         }),
        None ], 
    'test_ratio': [ str, Interval(Real, 0, 1, closed="neither")], 
    'contamination': [str,  Interval(Real, 0, 1, closed="neither")], 
    'anomaly_detector': [HasMethods(['predict']), None], 
    'random_state': ['random_state'], 
    'min_count': [Real, None], 
   })

@ensure_pkg (
    'imblearn', 
    dist_name=' imbalanced-learn', 
    partial_check=True, # check partially based on 'techn' condition 
    condition = lambda *args, **kwargs: kwargs.get('techn') in {
        'oversampling', 'undersampling', 'tomeklins', 'enn'}, 
    min_version='0.8.0'
 )
@check_empty ( ['target', 'data'])
def handle_minority_classes(
    target,
    data=None,
    target_col=None,
    techn=None,
    strategy=None,
    test_ratio="20%",
    augment_fn=None,
    contamination=0.01,
    anomaly_detector=None,
    random_state=42,
    min_count=None,
    error='raise',
    verbose=1,
    **kwargs
):
    """
    Manage imbalanced classification targets by dropping, resampling,
    combining, or augmenting minority classes.

    The `<handle_minority_classes>` function identifies classes in
    the provided ``target`` whose frequency is below a certain
    cutoff (`min_count`) and applies various user-specified
    techniques (e.g., `'oversampling'`, `'undersampling'`,
    `'combine'`, `'augment'`, `'anomaly_detection'`, etc.) to address
    imbalances. When no technique is specified, it defaults to
    `'minimum_samples'`, removing underrepresented classes.

    Parameters
    ----------
    target : str, DataFrame, Series, or ndarray
        The target variable, which can be:
        - A column name if `<data>` is a DataFrame.
        - A single-column DataFrame or Series.
        - A 1D NumPy array. If multiple columns exist, specify
          `<target_col>`.

    data : DataFrame or None, optional
        Feature matrix aligning with `target`. If `target` is a
        column name, `data` must be provided to extract the target.

    target_col : str or None, optional
        The specific column to use as a target if `<target>` is
        a multi-column DataFrame.

    techn : {'drop', 'minimum_samples', 'oversampling', 'undersampling',
             'combine', 'augment', 'anomaly_detection', 'tomeklins', 'enn'}
            or None, optional
        The imbalance handling approach:
        - ``'drop'`` or ``'minimum_samples'``: Removes underrepresented
          classes.
        - ``'oversampling'``: Uses `'random_over'`, `'smote'`, or
          `'adasyn'` strategies to balance.
        - ``'undersampling'``: Uses `'random_under'` or
          `'cluster_centroids'`.
        - ``'combine'``: Merges minority classes as `'Other'`.
        - ``'augment'``: Invokes a user-defined function for data
          augmentation.
        - ``'anomaly_detection'``: Treats minority classes as
          anomalies using an outlier detector.
        - ``'tomeklins'`` or ``'enn'``: Cleans ambiguous samples.
        If None, defaults to `'minimum_samples'`.

    strategy : str or None, optional
        Further refines how oversampling or undersampling is
        performed, such as `'random_over'`, `'smote'`, `'adasyn'`,
        `'random_under'`, or `'cluster_centroids'`. If None,
        defaults are `'random_over'` or `'random_under'`.

    test_ratio : str or float, default='20%'
        A ratio used to estimate a default cutoff if `<min_count>`
        is not provided. If given as a percentage string (e.g.,
        ``'20%'``), it is converted to a float fraction. Formally:
        .. math::
           \\text{test\\_size} = \\begin{cases}
           \\frac{20}{100} &\\text{if '20\\%'} \\\\
           0.2 &\\text{if 0.2} 
           \\end{cases}
        Used to compute ``min_count = \\lceil
        (\\text{test\\_size} * \\text{n\\_samples})\\rceil``.

    augment_fn : callable or None, optional
        A user-defined function for generating new samples in the
        `'augment'` technique. Must accept
        ``(X_minority, y_minority, **kwargs)`` and return
        ``(X_new, y_new)``.

    contamination : float, default=0.01
        The amount of contamination assumed in `'anomaly_detection'`.
        A fraction of outliers in the data. Passed to the chosen or
        default anomaly detector.

    anomaly_detector : object or None, optional
        A scikit-learn-like anomaly detection estimator. If None,
        `IsolationForest` is used by default.

    random_state : int, default=42
        A random seed ensuring reproducible resampling or detection
        where applicable.

    min_count : int or None, optional
        Minimum required samples per class. If not specified, it
        is derived from ``test_ratio * len(target)``.

    error : {'raise', 'warn', 'ignore'}, default='raise'
        The error handling policy for mismatches (e.g., if the
        length of `data` differs from `target`), or for regression
        targets:
        - ``'raise'``: Raises a `ValueError`.
        - ``'warn'``: Issues a `Warning`.
        - ``'ignore'``: Continues silently.

    verbose : int, default=1
        Controls the verbosity level for messages. Higher values
        print more details on performed steps.

    **kwargs
        Additional keyword arguments passed to the internal
        resampling or anomaly detection methods.

    Returns
    -------
    Series or tuple
        If `<data>` is None, returns the processed target only.
        Otherwise, returns a tuple ``(y_processed, data_processed)``.

    Notes
    -----
    The function interprets `target` via flexible inputs (column
    name, array, Series, or DataFrame). Once identified, it
    evaluates class frequencies in :math:`y`:
    .. math::
       \\text{counts}(c) < \\text{min\\_count} \\Longrightarrow
       \\text{class } c \\text{ is underrepresented}
    Then, depending on `<techn>`, it applies the corresponding
    balancing or augmentation method, leveraging internal
    functions like `oversample_data` or `undersample_data`.
    If `<techn>` is `'anomaly_detection'`, minority classes are
    labeled as `'Anomaly'` after fitting an outlier detector.

    Examples
    --------
    >>> from gofast.utils.ml.preprocessing import handle_minority_classes
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...    'feature1': [1,2,3,4,5,6],
    ...    'target':   [0,0,1,0,1,1]
    ... })
    >>> # Drop minority classes if they have fewer than 'min_count' samples
    >>> y_out, X_out = handle_minority_classes(
    ...    target='target', data=df, techn='drop', min_count=2
    ... )
    >>> y_out
    0    0
    1    0
    3    0
    dtype: int64

    See also
    --------
    `oversample_data` : Oversampling approaches like SMOTE, ADASYN,
    or random duplication.
    `undersample_data` : Undersampling approaches like
    random removal or cluster centroids.
    `augment_data` : Domain-specific generation of additional samples.

    References
    ----------
    .. [1] Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning
           in Python. Journal of Machine Learning Research, 12,
           2825–2830.
    .. [2] Gofast Documentation. Available at:
           https://gofast.readthedocs.io/en/latest/
    """

    # Resolve the error policy with default = 'raise' if unrecognized
    error = error_policy(error, base='raise')

    # Step 1) Interpret `target` parameter
    if isinstance(target, str):
        # If target is a column name, ensure data is provided
        if data is None:
            raise ValueError(
                "If 'target' is a string, 'data' must be provided."
            )
        # Check if target column is in data
        if target not in data.columns:
            raise ValueError(
                f"Column '{target}' is not found in the provided data."
            )
        # Extract the target as Series, drop from data
        y, data = extract_target(data, target_names=target, return_y_X=True )
        try: 
            y = to_series (y)
        except Exception as e: 
            raise ValueError(f"Multioutput target is not allowded. {e}")
            
    else:
        # Handle non-string target
        if isinstance(target, np.ndarray):
            # Convert to 1D array, then to Series
            target = to_array(target, accept='only_1d', as_frame=True)
            target = to_series(target, name='target')

        # Check target type
        if isinstance(target, pd.DataFrame):
            # If multiple columns, need to specify target_col
            if target.shape[1] == 1:
                y = target.iloc[:, 0].copy()
            else:
                if target_col is None:
                    raise ValueError(
                        "When 'target' has multiple columns, 'target_col' "
                        "must be specified."
                    )
                if target_col not in target.columns:
                    raise ValueError(
                        f"Column '{target_col}' is missing in the DataFrame."
                    )
                y = target[target_col].copy()
        elif isinstance(target, pd.Series):
            y = target.copy()
        else:
            raise TypeError(
                "Invalid 'target' type. Expect str, DataFrame,"
                " Series, or ndarray."
            )

        # If data is None but target was a DataFrame with extra columns
        if data is None and isinstance(target, pd.DataFrame):
            if target_col is not None and target_col in target.columns:
                data = target.drop(columns=[target_col])
            else:
                data = None

    # Step 2) Check length alignment if data is provided
    if data is not None:
        if len(data) != len(y):
            msg = (
                f"Length mismatch: data has {len(data)} rows, but "
                f"target has {len(y)} entries."
            )
            if error == 'raise':
                raise ValueError(msg)
            elif error == 'warn':
                warnings.warn(msg)
            # if 'ignore', do nothing

    # Detect target type
    target_type = type_of_target(y)

    # Reject continuous target if error policy is 'raise'
    if target_type in ['continuous', 'continuous-multioutput']:
        if error == 'raise':
            raise ValueError(
                "Regression target detected in a classification context. "
                "Please provide a classification target."
            )
        elif error in ['warn', 'ignore']:
            warnings.warn(
                "Regression target found in classification context. "
                "Continuing may produce unexpected results."
            )

    # Step 3) Identify minority classes
    class_counts = y.value_counts()
    total = len(y)

    # If no technique is specified, default to 'minimum_samples'
    if techn is None:
        if verbose > 0:
            print("No technique provided. Defaulting to 'minimum_samples'.")
        techn = 'minimum_samples'

    # Validate test_ratio => used as default min_count if needed
    test_size = validate_ratio(test_ratio, bounds=(0, 1),
                               param_name='Test ratio')

    # If we need to drop minority classes but min_count is not specified
    if techn in ('drop', 'minimum_samples') and min_count is None:
        min_count = int(np.ceil(test_size * total))

    # Identify classes below min_count
    if min_count is not None:
        minority_classes = class_counts[class_counts < min_count].index.tolist()
    else:
        minority_classes = []

    if verbose >= 2 and minority_classes:
        print(f"Detected minority classes: {minority_classes}")

    # Step 5) Apply technique
    # 5) Based on `techn`, we apply the chosen approach:
    #    - drop or minimum_samples
    #    - oversampling
    #    - undersampling
    #    - combine
    #    - augment
    #    - anomaly_detection
    #    - tomeklins or enn
    
    if techn in ('drop', 'minimum_samples'):
        if minority_classes:
            mask = ~y.isin(minority_classes)
            y_new = y[mask]
            data_new = data[mask] if data is not None else None
            if verbose >= 2:
                print(
                    f"Removing {len(y) - len(y_new)} samples from "
                    f"minority classes {minority_classes}."
                )
            y, data = y_new, data_new
        else:
            if verbose >= 2:
                print("No minority classes found to drop.")
    elif techn == 'combine':
        if minority_classes:
            y_new = y.where(~y.isin(minority_classes), other='Other')
            data_new = data.copy() if data is not None else None
            if verbose >= 2:
                print(
                    f"Combined minority classes {minority_classes} into 'Other'."
                )
            y, data = y_new, data_new
        else:
            if verbose >= 2:
                print("No minority classes to combine.")
    elif techn == 'oversampling':
        y, data = _oversample_data(
            y,
            data,
            strategy=strategy if strategy else 'random_over',
            random_state=random_state,
            **kwargs
        )
        if verbose >= 2:
            print(f"Oversampling with strategy='{strategy}'.")
    elif techn == 'undersampling':
        y, data = _undersample_data(
            y,
            data,
            strategy=strategy if strategy else 'random_under',
            random_state=random_state,
            **kwargs
        )
        if verbose >= 2:
            print(f"Undersampling with strategy='{strategy}'.")
    elif techn == 'augment':
        y, data = _augment_data(
            y,
            data,
            augmentation_fn=augment_fn,
            **kwargs
        )
        if verbose >= 2:
            print("Applied augmentation via user-defined function.")
    elif techn == 'anomaly_detection':
        y, data = _anomaly_detection_handle(
            y,
            data,
            anomaly_detector=anomaly_detector,
            contamination=contamination,
            **kwargs
        )
        if verbose >= 2:
            print("Anomaly detection performed on minority classes.")
    elif techn in ('tomeklins', 'enn'):
        y, data = _clean_data_tomek_enn(
            y,
            data,
            techn=techn,
            **kwargs
        )
        if verbose >= 2:
            print(f"Applied data cleaning with '{techn}'.")
    else:
        if techn is not None and verbose >= 1:
            warnings.warn(
                f"Technique '{techn}' not recognized. "
                "No action performed."
            )

    # Step 6) Return final results (target + data if available)
    if data is not None:
        return y, data
    else:
        return y

def _oversample_data(
        y, data, strategy='random_over', 
        random_state=None, **kwargs
    ):
    """
    Apply oversampling to balance the minority classes in the dataset.

    Parameters:
    - y: pandas Series, target variable.
    - data: pandas DataFrame or None, feature matrix.
    - strategy: str, oversampling strategy ('random_over', 'smote', 'adasyn').
    - random_state: int or None, random state for reproducibility.
    - **kwargs: additional keyword arguments for the oversampler.

    Returns:
    - y_resampled: pandas Series, resampled target variable.
    - data_resampled: pandas DataFrame or None, resampled feature matrix.
    """
    from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
    # Initialize the oversampler based on the strategy
    if strategy == 'random_over':
        sampler = RandomOverSampler(random_state=random_state, **kwargs)
    elif strategy == 'smote':
        sampler = SMOTE(random_state=random_state, **kwargs)
    elif strategy == 'adasyn':
        sampler = ADASYN(random_state=random_state, **kwargs)
    else:
        raise ValueError(
            f"Oversampling strategy '{strategy}' is not recognized.")

    if data is not None:
        # Fit and resample the data and target
        X_resampled, y_resampled = sampler.fit_resample(data, y)
        data_resampled = pd.DataFrame(X_resampled, columns=data.columns)
    else:
        # If data is None, only resample y
        y_resampled = sampler.fit_resample(y.values.reshape(-1, 1), y)[0].ravel()
        y_resampled = pd.Series(y_resampled, name=y.name)

    return y_resampled, data_resampled if data is not None else None

def _undersample_data(
        y, data, strategy='random_under', 
        random_state=None, **kwargs
    ):
    """
    Apply undersampling to balance the majority classes in the dataset.

    Parameters:
    - y: pandas Series, target variable.
    - data: pandas DataFrame or None, feature matrix.
    - strategy: str, undersampling strategy ('random_under', 'cluster_centroids').
    - random_state: int or None, random state for reproducibility.
    - **kwargs: additional keyword arguments for the undersampler.

    Returns:
    - y_resampled: pandas Series, resampled target variable.
    - data_resampled: pandas DataFrame or None, resampled feature matrix.
    """
    from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
    
    # Initialize the undersampler based on the strategy
    if strategy == 'random_under':
        sampler = RandomUnderSampler(random_state=random_state, **kwargs)
    elif strategy == 'cluster_centroids':
        sampler = ClusterCentroids(random_state=random_state, **kwargs)
    else:
        raise ValueError(
            f"Undersampling strategy '{strategy}' is not recognized.")

    if data is not None:
        # Fit and resample the data and target
        X_resampled, y_resampled = sampler.fit_resample(data, y)
        data_resampled = pd.DataFrame(X_resampled, columns=data.columns)
    else:
        # If data is None, only resample y
        y_resampled = sampler.fit_resample(y.values.reshape(-1, 1), y)[0].ravel()
        y_resampled = pd.Series(y_resampled, name=y.name)

    return y_resampled, data_resampled if data is not None else None


def _augment_data(y, data, augmentation_fn=None, **kwargs):
    """
    Augment minority classes using a domain-specific augmentation function.

    Parameters:
    - y: pandas Series, target variable.
    - data: pandas DataFrame or None, feature matrix.
    - augmentation_fn: function, user-defined function to generate new samples.
                      It should accept (X_minority, y_minority, **kwargs) 
                      and return (X_new, y_new).
    - **kwargs: additional keyword arguments for the augmentation function.

    Returns:
    - y_augmented: pandas Series, augmented target variable.
    - data_augmented: pandas DataFrame or None, augmented feature matrix.
    """
    if augmentation_fn is None:
        raise ValueError(
            "An augmentation function must be provided for 'augment' technique.")

    # Identify minority classes based on the current class distribution
    class_counts = y.value_counts()
    minority_classes = class_counts[class_counts < class_counts.mean()].index.tolist()

    if not minority_classes:
        # No minority classes to augment
        return y, data

    # Initialize lists to collect augmented data
    X_augmented_list = []
    y_augmented_list = []

    for cl in minority_classes:
        # Extract samples of the current minority class
        X_minority = data[y == cl] if data is not None else None
        y_minority = y[y == cl]

        # Generate new samples using the augmentation function
        X_new, y_new = augmentation_fn(X_minority, y_minority, **kwargs)

        if X_new is not None and y_new is not None:
            X_augmented_list.append(X_new)
            y_augmented_list.append(y_new)

    if X_augmented_list:
        # Concatenate all augmented samples
        if data is not None:
            X_augmented = pd.concat(X_augmented_list, ignore_index=True)
            y_augmented = pd.concat(y_augmented_list, ignore_index=True)
            data_augmented = pd.concat([data, X_augmented], ignore_index=True)
            y_augmented = pd.concat([y, y_augmented], ignore_index=True)
        else:
            y_augmented = pd.concat([y] + y_augmented_list, ignore_index=True)
            data_augmented = None

        return y_augmented, data_augmented
    else:
        # No augmentation was performed
        return y, data

def _anomaly_detection_handle(
        y, data, anomaly_detector=None, 
        contamination=0.01, **kwargs
    ):
    """
    Treat minority classes as anomalies and apply anomaly detection algorithms.

    Parameters:
    - y: pandas Series, target variable.
    - data: pandas DataFrame or None, feature matrix.
    - anomaly_detector: sklearn-like anomaly detection estimator.
                        If None, IsolationForest is used by default.
    - contamination: float, the amount of contamination of the data set.
    - **kwargs: additional keyword arguments for the anomaly detector.

    Returns:
    - y_anomaly: pandas Series, target variable with anomalies labeled.
    - data_anomaly: pandas DataFrame or None, feature matrix with anomalies handled.
    """
    check_numeric_dtype(data, y, param_names={"X":"Data", "y": "Target"})
    
    from sklearn.ensemble import IsolationForest

    # Initialize the anomaly detector
    if anomaly_detector is None:
        detector = IsolationForest(
            contamination=contamination, random_state=42, 
            **kwargs)
    else:
        detector = anomaly_detector

    if data is not None:
        # Fit the detector on the data
        detector.fit(data)

        # Predict anomalies
        anomaly_labels = detector.predict(data)
        # -1 for anomalies, 1 for normal instances

        # Map anomaly labels to the target
        y_anomaly = y.copy()
        y_anomaly[anomaly_labels == -1] = 'Anomaly'

        return y_anomaly, data
    else:
        # If data is None, cannot perform anomaly detection
        raise ValueError(
            "Data must be provided for anomaly detection.")


def _clean_data_tomek_enn(
        y, data, techn='tomeklins', error ='raise',  **kwargs):
    """
    Clean the dataset by removing ambiguous samples using Tomek Links or ENN.

    Parameters:
    - y: pandas Series, target variable.
    - data: pandas DataFrame or None, feature matrix.
    - techn: str, cleaning technique ('tomeklins', 'enn').
    - **kwargs: additional keyword arguments for the cleaner.

    Returns:
    - y_cleaned: pandas Series, cleaned target variable.
    - data_cleaned: pandas DataFrame or None, cleaned feature matrix.
    """
    check_numeric_dtype(data, y, param_names={"X":"Data", "y": "Target"})
        
    if techn == 'tomeklins':
        from imblearn.under_sampling import TomekLinks
        cleaner = TomekLinks(**kwargs)
    elif techn == 'enn':
        try:
            from imblearn.neighbors import EditedNearestNeighbours
        except ImportError as e:
            raise ImportError(
                "The 'imbalanced-learn' (version >=0.8.0) library is required"
                " for removing ambiguous samples using Tomek Links, but is not"
                " installed. Please install it using 'pip install imbalanced-learn'"
                " or 'conda install -c conda-forge imbalanced-learn' and try again."
            ) from e
            
        cleaner = EditedNearestNeighbours(**kwargs)
    else:
        raise ValueError(
            f"Cleaning technique '{techn}' is not recognized.")

    if data is not None:
        # Fit and resample the data and target
        X_cleaned, y_cleaned = cleaner.fit_resample(data, y)
        data_cleaned = pd.DataFrame(X_cleaned, columns=data.columns)
    else:
        # If data is None, cannot perform cleaning
        warnings.warn(
            "Data must be provided for cleaning with Tomek Links or ENN."
            "Cannot perform cleaning. Skipping!"
            )
        data_cleaned=None 

    return y_cleaned, data_cleaned


@validate_params ({
    'target': ['array-like'], 
    'categorize_strategy': [StrOptions({'quantile','uniform', 'auto'}), None], 
    })
def encode_target(
    target: Union[np.ndarray, pd.Series, pd.DataFrame],
    to_categorical: Optional[bool] = None,
    to_continuous: Optional[bool] = None,
    bins: Optional[int] = None,
    return_cat_codes: bool = False,
    categorize_strategy: str = 'auto',
    show_cat_codes: bool = False,
    verbose: int = 0
) -> Union[
    np.ndarray,
    Tuple[np.ndarray, Dict]
]:
    r"""
    Encode a target variable (time series or label column)
    by transforming numeric values to categorical bins or vice
    versa. The function can also optionally return mappings of
    the new categories or numeric codes for reference.

    Specifically, let :math:`X \in \mathbb{R}^n` be a numeric
    target. If :math:`\text{to\_categorical} = True`, it
    partitions :math:`X` into bins, returning integer-coded
    categories. Conversely, a categorical target can be mapped
    to numeric codes if :math:`\text{to\_continuous} = True`.

    Parameters
    ----------
    target : {ndarray, Series, DataFrame}
        The target data to encode. If multiple columns exist
        (DataFrame), each column is processed separately.
    to_categorical : bool, optional
        If ``True``, convert numeric data into categories via
        binning (Q-cut or uniform by default). If left
        ``None``, no forced numeric->categorical conversion
        is applied unless it is explicitly set or the data is
        already categorical.
    to_continuous : bool, optional
        If ``True``, convert categorical data into numeric
        codes. Typically, each unique category becomes an
        integer in [0, k-1], where k is the number of
        categories in the column.
    bins : int, optional
        Number of bins if converting numeric data to
        categorical. Default is 5 if none specified.
    return_cat_codes : bool, optional
        If ``True``, returns a second object containing a
        dictionary mapping integer codes to category labels
        for each column.
    categorize_strategy : {'auto', 'quantile', 'uniform'}, optional
        The strategy for binning numeric data:

        * ``'quantile'``: Use quantiles, ensuring an
          (approximately) equal count in each bin.
        * ``'uniform'``: Use uniform intervals over the
          data range.
        * ``'auto'``: Defaults to ``'quantile'``.
    show_cat_codes : bool, optional
        If ``True``, prints a summary of the category codes
        per column using :class:`ResultSummary`.
    verbose : int, optional
        Verbosity level:

        * ``0``: No console messages.
        * ``1``: Basic steps logged.
        * ``2``: Additional details on transformations.
        * ``3``: Very detailed logs (debug-level).

    Returns
    -------
    encoded_target : ndarray
        The transformed target data. If multiple columns, it
        returns an array of shape (n_samples, n_cols).
    cat_codes : dict, optional
        Returned only if ``return_cat_codes=True``. Maps each
        column name to a dictionary of integer code => category
        label.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from gofast.utils.ml.preprocessing import encode_target
    >>> # Numeric to categorical example
    >>> arr = np.array([10, 20, 30, 40, 50])
    >>> encoded, codes = encode_target(
    ...     arr,
    ...     to_categorical=True,
    ...     bins=2,
    ...     return_cat_codes=True,
    ...     categorize_strategy='quantile',
    ...     verbose=2
    ... )
    Bining strategy not specified. Reset to 5 (default). # Example logs
    Processing column '0' for numeric-to-categorical ...
    ...
    >>> # The array is now mapped to 2 bins, e.g. [0, 0, 0, 1, 1]

    Notes
    -----
    * If both ``to_categorical=True`` and ``to_continuous=True``
      are set, numeric->categorical has priority for numeric
      columns, and categorical->numeric for object/categorical
      columns.  
    * This function attempts to preserve the shape and index
      structure of the original input through the
      ``array_preserver`` mechanism [1]_.

    See Also
    --------
    pandas.cut : Bin values into discrete intervals (uniform).
    pandas.qcut : Bin values into intervals with equal counts.
    pandas.Categorical : Convert object data to categorical
        codes.

    References
    ----------
    .. [1] Cleveland, W.S. (1979). Robust Locally Weighted
           Regression and Smoothing Scatterplots.
           *Journal of the American Statistical Association*,
           74(368), 829-836. (Though more about robust
           smoothing, the array_preserver concept is somewhat
           similar in usage of shapes/structure).
    """

    # Preserve the structure of the input array/Series/DataFrame.
    collected = array_preserver(target, action='collect')

    # Convert input to a DataFrame if needed (for uniform processing).
    target = to_frame_if(target, df_only=True)

    # Helper to check if a Series is numeric
    def is_numeric(series: pd.Series) -> bool:
        return pd.api.types.is_numeric_dtype(series)

    # Prepare a dictionary to store category mappings
    map_codes = {}

    # List to accumulate processed columns
    encoded_columns = []

    # For each column in the target, apply the transformation
    for col in target.columns:
        col_data = target[col]

        # Case 1: Numeric -> Categorical
        if to_categorical and is_numeric(col_data):
            # If user hasn't specified bins, default to 5
            if bins is None:
                bins = 5
                if verbose >= 2:
                    print(
                        "Bining strategy not specified. "
                        "Reset to 5 (default)."
                    )

            if verbose >= 3:
                print(
                    f"Processing column '{col}' for numeric-"
                    f"to-categorical transformation..."
                )
                print(
                    f"Binning strategy: {categorize_strategy}, "
                    f"Bins: {bins}"
                )

            # Decide on quantile or uniform binning
            if categorize_strategy == 'quantile':
                categorized, bin_edges = pd.qcut(
                    col_data,
                    bins,
                    retbins=True,
                    duplicates='drop'
                )
            elif categorize_strategy == 'uniform':
                categorized, bin_edges = pd.cut(
                    col_data,
                    bins,
                    retbins=True
                )
            else:
                # 'auto' defaults to 'quantile'
                categorized, bin_edges = pd.qcut(
                    col_data,
                    bins,
                    retbins=True,
                    duplicates='drop'
                )

            # Convert categories to integer codes
            encoded = categorized.cat.codes
            map_codes[col] = {
                k: v
                for k, v in enumerate(categorized.cat.categories)
            }

            if verbose >= 2:
                print(f"Categorized column '{col}': {map_codes[col]}")

        # Case 2: Categorical -> Numeric
        elif to_continuous and not is_numeric(col_data):
            if verbose >= 3:
                print(
                    f"Processing column '{col}' for "
                    f"categorical-to-numeric transformation..."
                )

            # Ensure it's a pd.Categorical for .cat.codes
            col_data = col_data.astype('category')
            encoded = col_data.cat.codes
            map_codes[col] = {
                k: v
                for k, v in enumerate(col_data.cat.categories)
            }

            if verbose >= 2:
                print(f"Encoded column '{col}': {map_codes[col]}")

        # Case 3: No transformation needed or already appropriate
        else:
            # If already a categorical type, map codes
            if col_data.dtype.name == 'category':
                encoded = col_data.cat.codes
                map_codes[col] = {
                    k: v
                    for k, v in enumerate(col_data.cat.categories)
                }
            else:
                encoded = col_data

            if verbose >= 2:
                print(
                    f"Column '{col}' requires no transformation."
                )

        encoded_columns.append(encoded)

    # Reconstruct a DataFrame from processed columns
    encoded_target = pd.concat(encoded_columns, axis=1)

    # Attempt to restore original structure (index, shape, etc.)
    collected['processed'] = [encoded_target]
    try:
        encoded_target = array_preserver(
            collected,
            solo_return=True,
            action='restore',
            deep_restore=True
        )
    except Exception:
        # If it fails, fallback to raw DataFrame, optional warning
        encoded_target = return_if_preserver_failed(
            encoded_target,
            warn="ignore",
            verbose=verbose
        )

    # If user wants to see category codes, print them nicely
    if show_cat_codes:
        summary = ResultSummary(
            name="CatCodes",
            flatten_nested_dicts=False
        ).add_results(map_codes)
        print(summary)

    # If user wants category mappings returned, do so
    if return_cat_codes:
        return encoded_target, map_codes
    return encoded_target


@is_data_readable
@validate_params({
    'data': ['array-like'], 
    'target_columns': ['array-like', str, None], 
    'columns': ['array-like', str, None], 
    'impute_strategy':[dict, None], 
    })
def one_click_prep (
    data: DataFrame, 
    target_columns=None,
    columns=None, 
    impute_strategy=None,
    coerce_datetime=False, 
    seed=None, 
    **process_kws
    ):
    """
    Perform all-in-one preprocessing for beginners on a pandas DataFrame,
    simplifying common tasks like scaling, encoding, and imputation.

    This function is designed to be user-friendly, particularly for those new
    to data analysis, by automating complex preprocessing tasks with just a
    few parameters. It supports selective processing through specification
    of target and other columns and incorporates flexibility with custom
    imputation strategies and random seeds for reproducibility.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame to preprocess. This should be a structured DataFrame
        with any combination of numeric and categorical variables.
    target_columns : str or list of str, optional
        Column(s) designated as target(s) for modeling. These columns will
        not be scaled or imputed to avoid data leakage. Defaults to None,
        implying no columns are treated as targets.
    columns : str or list of str, optional
        Column names to be specifically included in preprocessing. If None
        (default), all columns are processed.
    impute_strategy : dict, optional
        Defines specific strategies for imputing missing values, separately
        for 'numeric' and 'categorical' data types. The default strategy is
        {'numeric': 'median', 'categorical': 'constant'}.
    coerce_datetime : bool, default=False
        If True, tries to convert object columns to datetime data types when 
        `data` is a numpy array. 
    seed : int, optional
        Random seed for operations that involve randomization, ensuring
        reproducibility of results. Defaults to None.
    
    **process_kws : keyword arguments, optional
        Additional arguments that can be passed to preprocessing steps, such
        as parameters for scaling or encoding methods.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with preprocessed data, where numeric columns have been
        scaled, categorical columns encoded, and missing values imputed.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np 
    >>> from gofast.utils import one_click_preprocess
    >>> data = pd.DataFrame({
    ...     'Age': [25, np.nan, 37, 59],
    ...     'City': ['New York', 'Paris', 'Berlin', np.nan],
    ...     'Income': [58000, 67000, np.nan, 120000]
    ... })
    >>> processed_data = one_click_preprocess(
    ...     data, seed=42)
    >>> processed_data.head()
            Age    Income  City_Berlin  City_New York  City_Paris  City_missing
    0 -1.180971 -0.815478            0              1           0             0
    1 -0.203616 -0.448513            0              0           1             0
    2 -0.203616 -0.448513            1              0           0             0
    3  1.588203  1.712504            0              0           0             1

    After preprocessing, the 'Age' and 'Income' columns will be scaled,
    the 'City' column will be one-hot encoded, and missing values in
    'Age' and 'City' will be imputed based on the specified strategies.

    Notes
    -----
    The function relies on scikit-learn's ColumnTransformer to apply
    different preprocessing steps to numeric and categorical columns.
    It is important to note that while this function aims to simplify
    preprocessing for beginners, understanding the underlying transformations
    and their implications is beneficial for more advanced data analysis.
    """
    # Convert input data to a DataFrame if it is not one already,
    # handling any issues silently.
    data = build_data_if(data, to_frame=True, force=True, input_name='col',
                         raise_warning='silence',
                         coerce_datetime=coerce_datetime 
                         )
    # Set a seed for reproducibility in operations that involve randomness.
    np.random.seed(seed)

    # Set default imputation strategies if none are provided.
    # Ensure impute_strategy is a dictionary with required keys.
    impute_strategy=validate_strategy(
        impute_strategy, error ='raise'
    )
    # Pop keyword arguments or set defaults for handling missing categories,
    # fill values, and behavior of additional columns not specified
    # in transformers.
    handle_unknown = process_kws.pop("handle_unknown", 'ignore')
    fill_value = process_kws.pop("fill_value", 'missing')
    remainder = process_kws.pop("remainder", 'passthrough')

    # If specific columns are specified, reduce the DataFrame to these columns only.
    columns = columns_manager(columns, pattern=r'[@&;#]')
    if columns is not None:
        data = data[columns] if isinstance(columns, list) else data[[columns]]

    # Convert target_columns to a list if it's a single column passed as string.
    if target_columns is not None:
        target_columns = [target_columns] if isinstance(
            target_columns, str) else target_columns

    # Identify numeric and categorical features based on their data type.
    numeric_features = data.select_dtypes(['int64', 'float64']).columns.tolist()
    categorical_features = data.select_dtypes(['object']).columns.tolist()
    
    # Exclude target columns from the numeric features list if specified.
    if target_columns is not None:
        numeric_features = [col for col in numeric_features 
                            if col not in target_columns]

    # Define transformation pipeline for numeric features: imputation and scaling.
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=impute_strategy['numeric'])),
        ('scaler', StandardScaler())
    ])

    # Define transformation pipeline for categorical 
    # features: imputation and one-hot encoding.
    steps = [('imputer', SimpleImputer(
        strategy=impute_strategy['categorical'], fill_value=fill_value))
        ] 
    categorical_step = ('onehot', OneHotEncoder(handle_unknown=handle_unknown))
    if categorical_features: 
        steps += [categorical_step]
        
    categorical_transformer = Pipeline(steps=steps)

    # Combine the numeric and categorical transformations with ColumnTransformer.
    transformer_steps = [('num', numeric_transformer, numeric_features)] 
    if categorical_features: 
        transformer_steps +=[('cat', categorical_transformer, categorical_features)]
        
    preprocessor = ColumnTransformer(
        transformers=transformer_steps, remainder=remainder
    )

    # Fit and transform the data using the defined ColumnTransformer.
    data_processed = preprocessor.fit_transform(data)

    # Attempt to retrieve processed column names for creating
    # a DataFrame from the transformed data.
    try:
        if categorical_features: 
            processed_columns = numeric_features + list(get_feature_names(
                preprocessor.named_transformers_['cat']['onehot'], categorical_features)
                ) + [col for col in data.columns 
                     if col not in numeric_features + categorical_features]
        else: 
            processed_columns = numeric_features  + [
                col for col in data.columns if col not in numeric_features]
    except:
        # Fallback for older versions of scikit-learn or other compatibility issues.
        if categorical_features: 
            cat_features_names = get_feature_names(preprocessor.named_transformers_['cat'])
            processed_columns = numeric_features + list(cat_features_names)
        else: 
            processed_columns = numeric_features

    # Check if the transformed data is a sparse matrix and convert to dense if necessary.
    if sparse.issparse(data_processed):
        data_processed = data_processed.toarray()

    # Create a new DataFrame with the processed data.
    data_processed = pd.DataFrame(
        data_processed, columns=processed_columns, index=data.index)

    # Attempt to use a custom transformer if available.
    try:
        from ..transformers import FloatCategoricalToInt
        data_processed = FloatCategoricalToInt(
            ).fit_transform(data_processed)
    except:
        pass
    # Return the preprocessed DataFrame.
    return data_processed

@is_data_readable
def soft_encoder(
    data: Union[DataFrame, ArrayLike], 
    columns: List[str] = None, 
    func: _F = None, 
    categories: Dict[str, List] = None, 
    get_dummies: bool = False, 
    parse_cols: bool = False, 
    return_cat_codes: bool = False, 
) -> DataFrame:
    """
    Encode multiple categorical variables in a dataset.

    Function facilitates the encoding of categorical variables by applying 
    specified transformations, mapping categories to integers, or performing 
    one-hot encoding. It accepts input data in the form of pandas DataFrames, 
    array-like structures convertible to DataFrame, or dictionaries that are 
    directly convertible to DataFrame.

    Parameters
    ----------
    data : DataFrame | ArrayLike | dict
        The input data to process. Accepts a pandas DataFrame, any array-like 
        structure that can be converted to a DataFrame, or a dictionary that will
        be converted to a DataFrame. In the case of array-like or dictionary
        inputs, the structure must be suitable for DataFrame transformation.
    columns : list, optional
        A list of column names from the data that are to be encoded. If not 
        provided, all columns within the DataFrame are considered for encoding.
    func : callable, optional
        A function to apply to the data for encoding purposes. The function 
        should accept a single value and return a transformed value. It is 
        applied to each element of the columns specified by `columns`, or to all
        elements in the DataFrame if `columns` is None.
    categories : dict, optional
        A dictionary mapping column names (keys) to lists of categories (values)
        that should be used for encoding. This dictionary explicitly defines 
        the categories corresponding to each column that should be transformed.
    get_dummies : bool, default False
        When set to True, enables one-hot encoding for the specified `columns` 
        or for all columns if `columns` is unspecified. This parameter converts
        each categorical variable into multiple binary columns, one for each 
        category, which indicates the presence of the category.
    parse_cols : bool, default False
        If True and `columns` is a string, this parameter will interpret the 
        string as a list of column names, effectively parsing a single comma-
        separated string into separate column names.
    return_cat_codes : bool, default False
        When True, the function returns a tuple. The first element of the tuple 
        is the DataFrame with transformed data, and the second element is a 
        dictionary that maps the original categorical values to the new 
        numerical codes. This is useful for retaining a reference to the original
        categorical data.

    Returns
    -------
    DataFrame or (DataFrame, dict)
        The primary return is the encoded DataFrame. If `return_cat_codes` is 
        True, a dictionary mapping original categories to their new numerical 
        codes is also returned.
        
    Raises
    ------
    TypeError
        If `func` is provided but is not callable, or if `categories` 
        is not a dictionary.
        
    Examples
    --------
    >>> from gofast.utils.mlutils import soft_encoder
    >>> # Sample dataset with categorical variables
    >>> data = {'Height': [152, 175, 162, 140, 170],
    ...         'Color': ['Red', 'Blue', 'Green', 'Red', 'Blue'],
    ...         'Size': ['Small', 'Large', 'Medium', 'Medium', 'Small'],
    ...         'Shape': ['Circle', 'Square', 'Triangle', 'Circle', 'Triangle'],
    ...         'Weight': [80, 75, 55, 61, 70]
    ...        }
    >>> # Basic encoding without additional parameters
    >>> df_encoded = soft_encoder(data)
    >>> df_encoded.head(2)
    Out[1]:
       Height  Weight  Color  Size  Shape
    0     152      80      2     2      0
    1     175      75      0     0      1
    
    >>> # Returning a map of categorical codes
    >>> df_encoded, map_codes = soft_encoder(data, return_cat_codes=True)
    >>> map_codes
    Out[2]:
    {'Color': {2: 'Red', 0: 'Blue', 1: 'Green'},
     'Size': {2: 'Small', 0: 'Large', 1: 'Medium'},
     'Shape': {0: 'Circle', 1: 'Square', 2: 'Triangle'}}
    
    >>> # Custom function to manually map categories
    >>> def cat_func(x):
    ...     if x == 'Red':
    ...         return 2
    ...     elif x == 'Blue':
    ...         return 0
    ...     elif x == 'Green':
    ...         return 1
    ...     else:
    ...         return x
    >>> df_encoded = soft_encoder(data, func=cat_func)
    >>> df_encoded.head(3)
    Out[3]:
       Height  Color    Size     Shape  Weight
    0     152      2   Small    Circle      80
    1     175      0   Large    Square      75
    2     162      1  Medium  Triangle      55
    
    >>> # Perform one-hot encoding
    >>> df_encoded = soft_encoder(data, get_dummies=True)
    >>> df_encoded.head(3)
    Out[4]:
       Height  Weight  Color_Blue  Color_Green  Color_Red  Size_Large  Size_Medium  \
    0     152      80           0            0          1           0            0   
    1     175      75           1            0          0           1            0   
    2     162      55           0            1          0           0            1   
    
       Size_Small  Shape_Circle  Shape_Square  Shape_Triangle
    0           1             1             0               0
    1           0             0             1               0
    2           0             0             0               1
    
    >>> # Specifying explicit categories
    >>> df_encoded = soft_encoder(data, categories={'Size': ['Small', 'Large', 'Medium']})
    >>> df_encoded.head()
    Out[5]:
       Height  Color     Shape  Weight  Size
    0     152    Red    Circle      80     0
    1     175   Blue    Square      75     1
    2     162  Green  Triangle      55     2
    3     140    Red    Circle      61     2
    4     170   Blue  Triangle      70     0
    
    Notes
    -----
    - The function handles various forms of input data, applying default 
      encoding if no specific encoding function or categories are provided.
    - Custom encoding functions allow for flexibility in mapping categories 
      manually.
    - One-hot encoding transforms each categorical attribute into multiple 
      binary attributes, enhancing model interpretability but increasing 
      dimensionality.
    - Specifying explicit categories helps ensure consistency in encoding, 
      especially when some categories might not appear in the training set but 
      could appear in future data.

    """

    # Convert input data to DataFrame if not already a DataFrame
    df = build_data_if(data, to_frame=True, force=True, input_name='col',
                       raise_warning='silence')

    # Recheck and convert data to numeric dtypes if possible
    # and handle NaN to fit it specified types. 
    df = nan_to_na(to_numeric_dtypes(df)) 
    # Ensure columns are iterable and parse them if necessary
    if columns is not None:
        columns = columns_manager(
            columns, pattern=r'[@&,;#]' if parse_cols else r'[@&;#]',
            to_string=parse_cols
        )
        # Select only the specified features from the DataFrame
        df = select_features(df, features=columns)
    
    # Initialize map_codes to store mappings of categorical codes to labels
    map_codes = {}
    # Create a CategoryMap code object for nested dict collection
    mapresult=ResultSummary(name="CategoryMap", flatten_nested_dicts=False)
    
    # Perform one-hot encoding if requested
    if get_dummies:
        # Use pandas get_dummies for one-hot encoding and handle
        # return type based on return_cat_codes
        mapresult.add_results(map_codes)
        return (pd.get_dummies(df, columns=columns), mapresult
                ) if return_cat_codes else pd.get_dummies(df, columns=columns)

    # Automatically select numeric and categorical columns
    # if not manually specified
    num_columns, cat_columns = bi_selector(df)

    # Apply provided function to categorical columns if func is given
    if func is not None:
        if not callable(func):
            raise TypeError(f"Provided func is not callable. Received: {type(func)}")
        if len(cat_columns) == 0:
            # Warn if no categorical data were found
            warnings.warn("No categorical data were detected. To transform"
                          " numeric values into categorical labels, consider"
                          " using either `gofast.utils.smart_label_classifier`"
                          " or `gofast.utils.categorize_target`.")
            return df
        
        # Apply the function to each categorical column
        for col in cat_columns:
            df[col] = df[col].apply(func)

        # Return DataFrame and mappings if required
        mapresult.add_results(map_codes)
        return (df, mapresult) if return_cat_codes else df

    # Handle automatic categorization if categories are not provided
    if categories is None:
        categories = {}
        for col in cat_columns:
            # Drop NaN values before finding unique categories
            unique_values = np.unique(df[col].dropna())
            categories[col] =  list(unique_values) 

    # Ensure categories is a dictionary
    if not isinstance(categories, dict):
        raise TypeError("Expected a dictionary with the format"
                        " {'column name': 'labels'} to categorize data.")

    # Map categories for each column and adjust DataFrame accordingly
    for col, values in categories.items():
        if col not in df.columns:
            continue
        values = is_iterable(values, exclude_string=True, transform=True)
        df[col] = pd.Categorical(df[col], categories=values, ordered=True)
        val = df[col].cat.codes
        temp_col = col + '_col'
        df[temp_col] = val
        map_codes[col] = dict(zip(val, df[col]))
        df.drop(columns=[col], inplace=True)  # Drop original column
        # Rename the temp column to original column name
        df.rename(columns={temp_col: col}, inplace=True) 

    # Return DataFrame and mappings if required
    mapresult.add_results(map_codes)
    return (df, mapresult) if return_cat_codes else df

@ensure_pkg ("imblearn", extra= (
    "`imblearn` is actually a shorthand for ``imbalanced-learn``.")
   )
def resampling( 
    X, 
    y, 
    kind ='over', 
    strategy ='auto', 
    random_state =None, 
    verbose: bool=..., 
    **kws
    ): 
    """ Combining Random Oversampling and Undersampling 
    
    Resampling involves creating a new transformed version of the training 
    dataset in which the selected examples have a different class distribution.
    This is a simple and effective strategy for imbalanced classification 
    problems.

    Applying re-sampling strategies to obtain a more balanced data 
    distribution is an effective solution to the imbalance problem. There are 
    two main approaches to random resampling for imbalanced classification; 
    they are oversampling and undersampling.

    - Random Oversampling: Randomly duplicate examples in the minority class.
    - Random Undersampling: Randomly delete examples in the majority class.
        
    Parameters 
    -----------
    X : array-like of shape (n_samples, n_features)
        Training vector, where `n_samples` is the number of samples and
        `n_features` is the number of features.
        
    y: array-like of shape (n_samples, ) 
        Target vector where `n_samples` is the number of samples.
    kind: str, {"over", "under"} , default="over"
      kind of sampling to perform. ``"over"`` and ``"under"`` stand for 
      `oversampling` and `undersampling` respectively. 
      
    strategy : float, str, dict, callable, default='auto'
        Sampling information to sample the data set.

        - When ``float``, it corresponds to the desired ratio of the number of
          samples in the minority class over the number of samples in the
          majority class after resampling. Therefore, the ratio is expressed as
          :math:`\\alpha_{us} = N_{m} / N_{rM}` where :math:`N_{m}` is the
          number of samples in the minority class and
          :math:`N_{rM}` is the number of samples in the majority class
          after resampling.

          .. warning::
             ``float`` is only available for **binary** classification. An
             error is raised for multi-class classification.

        - When ``str``, specify the class targeted by the resampling. The
          number of samples in the different classes will be equalized.
          Possible choices are:

            ``'majority'``: resample only the majority class;

            ``'not minority'``: resample all classes but the minority class;

            ``'not majority'``: resample all classes but the majority class;

            ``'all'``: resample all classes;

            ``'auto'``: equivalent to ``'not minority'``.

        - When ``dict``, the keys correspond to the targeted classes. The
          values correspond to the desired number of samples for each targeted
          class.

        - When callable, function taking ``y`` and returns a ``dict``. The keys
          correspond to the targeted classes. The values correspond to the
          desired number of samples for each class.
          
    random_state : int, RandomState instance, default=None
            Control the randomization of the algorithm.

            - If int, ``random_state`` is the seed used by the random number
              generator;
            - If ``RandomState`` instance, random_state is the random number
              generator;
            - If ``None``, the random number generator is the ``RandomState``
              instance used by ``np.random``.
              
    verbose: bool, default=False 
      Display the counting samples 
      
    Returns 
    ---------
    X, y : NDarray, Arraylike 
        Arraylike sampled 
    
    Examples 
    --------- 
    >>> import gofast as gf 
    >>> from gofast.utils.mlutils import resampling 
    >>> data, target = gf.fetch_data ('bagoue analysed', as_frame =True, return_X_y=True) 
    >>> data.shape, target.shape 
    >>> data_us, target_us = resampling (data, target, kind ='under',verbose=True)
    >>> data_us.shape, target_us.shape 
    Counters: Auto      
                         Raw counter y: Counter({0: 232, 1: 112})
               UnderSampling counter y: Counter({0: 112, 1: 112})
    Out[43]: ((224, 8), (224,))
    
    """
    kind = str(kind).lower() 
    if kind =='under': 
        from imblearn.under_sampling import RandomUnderSampler
        rsampler = RandomUnderSampler(sampling_strategy=strategy, 
                                      random_state = random_state ,
                                      **kws)
    else:  
        from imblearn.over_sampling import RandomOverSampler 
        rsampler = RandomOverSampler(sampling_strategy=strategy, 
                                     random_state = random_state ,
                                     **kws
                                     )
    Xs, ys = rsampler.fit_resample(X, y)
    
    if verbose: 
        print("{:<20}".format(f"Counters: {strategy.title()}"))
        print( "{:>35}".format( "Raw counter y:") , Counter (y))
        print( "{:>35}".format(f"{kind.title()}Sampling counter y:"), Counter (ys))
        
    return Xs, ys 


@is_data_readable
def bin_counting(
    data: DataFrame, 
    bin_columns: Union[str, List[str]], 
    tname: Union[str, Series[int]], 
    odds: str = "N+", 
    return_counts: bool = False, 
    tolog: bool = False, 
    encode_categorical: bool = False, 
) -> None:
    """ Bin counting categorical variable and turn it into probabilistic 
    ratio.
    
    Bin counting is one of the perennial rediscoveries in machine learning. 
    It has been reinvented and used in a variety of applications, from ad 
    click-through rate prediction to hardware branch prediction [1]_, [2]_ 
    and [3]_.
    
    Given an input variable X and a target variable Y, the odds ratio is 
    defined as:
        
    .. math:: 
        
        odds ratio = \frac{ P(Y = 1 | X = 1)/ P(Y = 0 | X = 1)}{
            P(Y = 1 | X = 0)/ P(Y = 0 | X = 0)}
          
    Probability ratios can easily become very small or very large. The log 
    transform again comes to our rescue. Anotheruseful property of the 
    logarithm is that it turns a division into a subtraction. To turn 
    bin statistic probability value to log, set ``uselog=True``.
    
    Parameters 
    -----------
    data: dataframe 
       Data containing the categorical values. 
       
    bin_columns: str or list , 
       The columns to apply the bin_counting. If 'auto', the columns are 
       extracted and categorized before applying the `bin_counting`. 
       
    tname: str, pd.Series
      The target name for which the counting is operated. If series, it 
      must have the same length as the data. 
      
    odds: str , {"N+", "N-", "log_N+"}: 
        The odds ratio of bin counting to fill the categorical. ``N+`` and  
        ``N-`` are positive and negative probabilistic computing. Whereas the
        ``log_N+`` is the logarithm odds ratio useful when value are smaller 
        or larger. 
        
    return_counts: bool, default=True 
      return the bin counting dataframes. 
  
    tolog: bool, default=False, 
      Apply the logarithm to the output data ratio. Indeed, Probability ratios 
      can easily  become very small or very large. For instance, there will be 
      users who almost never click on ads, and perhaps users who click on ads 
      much more frequently than not.) The log transform again comes to our  
      rescue. Another useful property of the logarithm is that it turns a 
      division 
      
    encode_categorical : bool, optional
        If `True`, encode categorical variables. Default is `False`.

    Returns 
    --------
    d: dataframe 
       Dataframe transformed or bin-counting data
       
    Examples 
    ---------
    >>> import gofast as gf 
    >>> from gofast.utils.mlutils import bin_counting 
    >>> X, y = gf.fetch_data ('bagoue analysed', as_frame =True) 
    >>> # target binarize 
    >>> y [y <=1] = 0;  y [y > 0]=1 
    >>> X.head(2) 
    Out[7]: 
          power  magnitude       sfi      ohmS       lwi  shape  type  geol
    0  0.191800  -0.140799 -0.426916  0.386121  0.638622    4.0   1.0   3.0
    1 -0.430644  -0.114022  1.678541 -0.185662 -0.063900    3.0   2.0   2.0
    >>>  bin_counting (X , bin_columns= 'geol', tname =y).head(2)
    Out[8]: 
          power  magnitude       sfi      ohmS  ...  shape  type      geol  bin_target
    0  0.191800  -0.140799 -0.426916  0.386121  ...    4.0   1.0  0.656716           1
    1 -0.430644  -0.114022  1.678541 -0.185662  ...    3.0   2.0  0.219251           0
    [2 rows x 9 columns]
    >>>  bin_counting (X , bin_columns= ['geol', 'shape', 'type'], tname =y).head(2)
    Out[10]: 
          power  magnitude       sfi  ...      type      geol  bin_target
    0  0.191800  -0.140799 -0.426916  ...  0.267241  0.656716           1
    1 -0.430644  -0.114022  1.678541  ...  0.385965  0.219251           0
    [2 rows x 9 columns]
    >>> df = pd.DataFrame ( pd.concat ( [X, pd.Series ( y, name ='flow')],
                                       axis =1))
    >>> bin_counting (df , bin_columns= ['geol', 'shape', 'type'], 
                      tname ="flow", tolog=True).head(2)
    Out[12]: 
          power  magnitude       sfi      ohmS  ...     shape      type      geol  flow
    0  0.191800  -0.140799 -0.426916  0.386121  ...  0.828571  0.364706  1.913043     1
    1 -0.430644  -0.114022  1.678541 -0.185662  ...  0.364865  0.628571  0.280822     0
    >>> bin_counting (df , bin_columns= ['geol', 'shape', 'type'],odds ="N-", 
                      tname =y, tolog=True).head(2)
    Out[13]: 
          power  magnitude       sfi  ...      geol  flow  bin_target
    0  0.191800  -0.140799 -0.426916  ...  0.522727     1           1
    1 -0.430644  -0.114022  1.678541  ...  3.560976     0           0
    [2 rows x 10 columns]
    >>> bin_counting (df , bin_columns= "geol",tname ="flow", tolog=True,
                      return_counts= True )
    Out[14]: 
         flow  no_flow  total_flow        N+        N-     logN+     logN-
    3.0    44       23          67  0.656716  0.343284  1.913043  0.522727
    2.0    41      146         187  0.219251  0.780749  0.280822  3.560976
    0.0    18       43          61  0.295082  0.704918  0.418605  2.388889
    1.0     9       20          29  0.310345  0.689655  0.450000  2.222222

    References 
    -----------
    .. [1] Yeh, Tse-Yu, and Yale N. Patt. Two-Level Adaptive Training Branch 
           Prediction. Proceedings of the 24th Annual International 
           Symposium on Microarchitecture (1991):51–61
           
    .. [2] Li, Wei, Xuerui Wang, Ruofei Zhang, Ying Cui, Jianchang Mao, and 
           Rong Jin.Exploitation and Exploration in a Performance Based Contextual 
           Advertising System. Proceedings of the 16th ACM SIGKDD International
           Conference on Knowledge Discovery and Data Mining (2010): 27–36
           
    .. [3] Chen, Ye, Dmitry Pavlov, and John _F. Canny. “Large-Scale Behavioral 
           Targeting. Proceedings of the 15th ACM SIGKDD International 
           Conference on Knowledge Discovery and Data Mining (2009): 209–218     
    """
  
    # assert everything
    if not is_frame (data, df_only =True ):
        raise TypeError(f"Expect dataframe. Got {type(data).__name__!r}")
        
    if isinstance (bin_columns, str) and bin_columns=='auto': 
        ttname = tname if isinstance (tname, str) else None # pass 
        _, bin_columns = process_data_types(
            data, target_name= ttname,
            exclude_target=True if ttname else False 
        )
        
    if encode_categorical: 
        data = soft_encoder(data) 
        
    if not _is_numeric_dtype(data, to_array= True): 
        raise TypeError ("Expect data with encoded categorical variables."
                         " Please check your data.")
    if hasattr ( tname, '__array__'): 
        check_consistent_length( data, tname )
        if not _is_arraylike_1d(tname): 
            raise TypeError (
                 "Only one dimensional array is allowed for the target.")
        # create fake bin target 
        if not hasattr ( tname, 'name'): 
            tname = pd.Series (tname, name ='bin_target')
        # concatenate target 
        data= pd.concat ( [ data, tname], axis = 1 )
        tname = tname.name  # take the name 
    
    bin_columns= is_iterable(bin_columns, exclude_string= True, transform =True )
    tname = str(tname) ; 
    target_all_counts =[]
    
    validate_feature(data, features =bin_columns + [tname] )
    d= data.copy() 
    # -convert all features dtype to float for consistency
    # except the binary target 
    feature_cols = is_in_if (d.columns , tname, return_diff= True ) 
    d[feature_cols] = d[feature_cols].astype ( float)
    # -------------------------------------------------
    for bin_column in bin_columns: 
        d, tc  = _single_counts(d , bin_column, tname, 
                           odds =odds, 
                           tolog=tolog, 
                           return_counts= return_counts
                           )
    
        target_all_counts.append (tc) 
    # lowering the computation time 
    if return_counts: 
        d = ( target_all_counts if len(target_all_counts) >1 
                 else target_all_counts [0]
                 ) 

    return d



def _single_counts ( 
        d,  bin_column, tname, odds = "N+",
        tolog= False, return_counts = False ): 
    """ An isolated part of bin counting. 
    Compute single bin_counting. """
    # polish pos_label 
    od = copy.deepcopy( odds) 
    # reconvert log and removetrailer
    odds= str(odds).upper().replace ("_", "")
    # just separted for 
    keys = ('N-', 'N+', 'lOGN+')
    msg = ("Odds ratio or log Odds ratio expects"
           f" {smart_format(('N-', 'N+', 'logN+'), 'or')}. Got {od!r}")
    # check wther log is included 
    if odds.find('LOG')>=0: 
        tolog=True # then remove log 
        odds= odds.replace ("LOG", "")

    if odds not in keys: 
        raise ValueError (msg) 
    # If tolog, then reconstructs
    # the odds_labels
    if tolog: 
        odds= f"log{odds}"
    
    target_counts= _target_counting(
        d.filter(items=[bin_column, tname]),
    bin_column , tname =tname, 
    )
    target_all, target_bin_counts = _bin_counting(target_counts, tname, odds)
    # Check to make sure we have all the devices
    target_all.sort_values(by = f'total_{tname}', ascending=False)  
    if return_counts: 
        return d, target_all 
   
    # zip index with ratio 
    lr = list(zip (target_bin_counts.index, target_bin_counts[odds])
         )
    ybin = np.array ( d[bin_column])# replace value with ratio 
    for (value , ratio) in lr : 
        ybin [ybin ==value] = ratio 
        
    d[bin_column] = ybin 
    
    return d, target_all

def _target_counting(d, bin_column, tname ):
    """ An isolated part of counting the target. 
    
    :param d: DataFrame 
    :param bin_column: str, columns to appling bincounting strategy 
    :param tname: str, target name. 

    """
    pos_action = pd.Series(d[d[tname] > 0][bin_column].value_counts(),
        name=tname)
    
    neg_action = pd.Series(d[d[tname] < 1][bin_column].value_counts(),
    name=f'no_{tname}')
     
    counts = pd.DataFrame([pos_action,neg_action]).T.fillna('0')
    counts[f'total_{tname}'] = counts[tname].astype('int64') +\
    counts[f'no_{tname}'].astype('int64')
    
    return counts

def _bin_counting (counts, tname, odds="N+" ):
    """ Bin counting application to the target. 
    :param counts: pd.Series. Target counts 
    :param tname: str target name. 
    :param odds: str, label to bin-compute
    """
    counts['N+'] = ( counts[tname]
                    .astype('int64')
                    .divide(counts[f'total_{tname}'].astype('int64')
                            )
                    )
    counts['N-'] = ( counts[f'no_{tname}']
                    .astype('int64')
                    .divide(counts[f'total_{tname}'].astype('int64'))
                    )
    
    items2filter= ['N+', 'N-']
    if str(odds).find ('log')>=0: 
        counts['logN+'] = counts['N+'].divide(counts['N-'])
        counts ['logN-'] = counts ['N-'].divide ( counts['N+'])
        items2filter.extend (['logN+', 'logN-'])
    # If we wanted to only return bin-counting properties, 
    # we would filter here
    bin_counts = counts.filter(items= items2filter)

    return counts, bin_counts  

@is_data_readable
def process_data_types(
    data,
    target_name=None, 
    exclude_target=False, 
    return_frame=False, 
    include_datetime=True,
    is_numeric_datetime=True, 
    return_target=False, 
    encode_cat_columns=False, 
    error='warn', 
    ):
    """
    Processes a DataFrame to separate numeric and categorical data.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame.
    target_name : str or list of str, optional
        The name(s) of the target column(s) to exclude. If a single 
        target column is provided, it can be given as a string. 
        If multiple target columns are provided, they should be given 
        as a list of strings.
    exclude_target : bool, optional
        If `True`, exclude the target column(s) from the returned 
        numeric and categorical data. Default is `False`.
    return_frame : bool, optional
        If `True`, return DataFrames for numeric and categorical data. 
        If `False`, return lists of column names. Default is `False`.
    include_datetime : bool, optional
        If `True`, include datetime columns in the processing. 
        Default is `True`.
    is_numeric_datetime : bool, optional
        If `True`, consider datetime columns as numeric. Default is `True`.
    return_target : bool, optional
        If `True`, return the target data along with numeric and 
        categorical data. Default is `False`.
    encode_cat_columns : bool, optional
        If `True`, encode categorical variables. Default is `False`.
    error : str, optional
        How to handle errors when target columns are not found. Options 
        are ``'warn'``, ``'raise'``, ``'ignore'``. Default is ``'warn'``.

    Returns
    -------
    tuple
        If `return_frame` is `True`, returns a tuple of DataFrames 
        (`numeric_df`, `categorical_df`, `target_df`). 
        If `return_frame` is `False`, returns a tuple of lists 
        (`numeric_columns`, `categorical_columns`, `target_columns`).

    Raises
    ------
    ValueError
        If `exclude_target` is `True` and `target_name` is not provided.
        If specified target columns do not exist in the DataFrame 
        and ``error`` is set to ``'raise'``.

    Notes
    -----
    The function processes a DataFrame to separate numeric and 
    categorical data, with optional inclusion of datetime columns. 
    If `target_name` is provided, it can be excluded from the data 
    or returned separately. The function handles missing target 
    columns based on the ``error`` parameter.

    Mathematically, the separation of numeric and categorical columns 
    can be represented as:

    .. math:: 
        X_{\text{numeric}} = \{ x_i \mid x_i \in \mathbb{R}, \forall i \}
    
    .. math:: 
        X_{\text{categorical}} = \{ x_i \mid x_i \notin \mathbb{R}, \forall i \}

    where :math:`X` is the set of all columns in the DataFrame, 
    :math:`X_{\text{numeric}}` is the set of numeric columns, and 
    :math:`X_{\text{categorical}}` is the set of categorical columns.

    Examples
    --------
    >>> from gofast.utils.mlutils import process_data_types 
    >>> df = pd.DataFrame({'A': [1, 2, 3], 
    ...                    'B': ['a', 'b', 'c'], 
    ...                    'C': pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03'])})
    >>> numeric_columns, categorical_columns = process_data_types(df)
    >>> numeric_columns
    ['A']
    >>> categorical_columns
    ['B']

    >>> numeric_df, categorical_df = process_data_types(df, return_frame=True)
    >>> numeric_df
       A
    0  1
    1  2
    2  3
    >>> categorical_df
       B
    0  a
    1  b
    2  c

    >>> numeric_columns, categorical_columns, target_columns = process_data_types(
    ...     df, target_name='A', exclude_target=True, return_target=True)
    >>> numeric_columns
    []
    >>> categorical_columns
    ['B']
    >>> target_columns
    ['A']

    See Also
    --------
    pandas.DataFrame.select_dtypes : Select columns by data type.
    pandas.concat : Concatenate pandas objects along a particular axis.

    References
    ----------
    .. [1] McKinney, Wes. "pandas: a foundational Python library for data 
       analysis and statistics." Python for High Performance and Scientific 
       Computing (2011): 1-9.
    """

    is_frame(data, df_only =True, raise_exception=True, objname='data')
    if exclude_target and not target_name:
        raise ValueError("If exclude_target is True, target_name must be provided.")
    
    if isinstance(target_name, str):
        target_name = [target_name]
    
    if target_name:
        missing_targets = [target for target in target_name if target not in data.columns]
        if missing_targets:
            if error == 'warn':
                warnings.warn(f"Target columns {missing_targets} not found in DataFrame.")
            elif error == 'raise':
                raise ValueError(f"Target columns {missing_targets} not found in DataFrame.")
            # Filter out missing targets
            target_name = [target for target in target_name if target in data.columns]
    
    numeric_df = data.select_dtypes([np.number])
    categorical_df = data.select_dtypes(None, [np.number])
    
    if include_datetime:
        datetime_df = data.select_dtypes(['datetime'])
        if is_numeric_datetime:
            numeric_df = pd.concat([numeric_df, datetime_df], axis=1)
        else:
            categorical_df = pd.concat([categorical_df, datetime_df], axis=1)
    
    target_df = pd.DataFrame()
    if target_name:
        target_df = data[target_name]
    
    if exclude_target and target_name:
        numeric_df = numeric_df.drop(columns=target_name, errors='ignore')
        categorical_df = categorical_df.drop(columns=target_name, errors='ignore')
    
    if encode_cat_columns and not categorical_df.empty: 
        categorical_df = soft_encoder ( categorical_df )
        
    if return_frame:
        if return_target:
            return numeric_df, categorical_df, target_df
        else:
            return numeric_df, categorical_df
    else:
        if return_target:
            return ( 
                numeric_df.columns.tolist(), categorical_df.columns.tolist(),
                target_df.columns.tolist()
                )
        else:
            return numeric_df.columns.tolist(), categorical_df.columns.tolist()

@is_data_readable
def discretize_categories(
    data: Union[DataFrame, Series],
    in_cat: str,
    new_cat: Optional[str] = None,
    divby: float = 1.5,
    higherclass: int = 5
) -> DataFrame:
    """
    Discretizes a numerical column in the DataFrame into categories. 
    
    Creating a new categorical column based on ceiling division and 
    an upper class limit.

    Parameters
    ----------
    data : DataFrame or Series
        Input data containing the column to be discretized.
    in_cat : str
        Column name in `data` used for generating the new categorical attribute.
    new_cat : str, optional
        Name for the newly created categorical column. If not provided, 
        a default name 'new_category' is used.
    divby : float, default=1.5
        The divisor used in the ceiling division to discretize the column values.
    higherclass : int, default=5
        The upper bound for the discretized categories. Values reaching this 
        class or higher are grouped into this single upper class.

    Returns
    -------
    DataFrame
        A new DataFrame including the newly created categorical column.

    Examples
    --------
    >>> from gofast.utils.mlutils import discretize_categories
    >>> df = pd.DataFrame({'age': [23, 45, 18, 27]})
    >>> discretized_df = discretize_categories(df, 'age', 'age_cat', divby=10, higherclass=3)
    >>> print(discretized_df)
       age  age_cat
    0   23      3.0
    1   45      3.0
    2   18      2.0
    3   27      3.0

    Note: The 'age_cat' column contains discretized categories based on the 
    'age' column.
    """
    if isinstance (data, pd.Series): 
        data = data.to_frame() 
    is_frame(data, df_only =True, raise_exception=True, objname='data')
    
    if new_cat is None:
        new_cat = 'new_category'
    
    # Discretize the specified column
    data[new_cat] = np.ceil(data[in_cat] / divby)
    # Apply upper class limit
    data[new_cat] = data[new_cat].where(data[new_cat] < higherclass, other=higherclass)
    
    return data

def _assert_sl_target (target,  df=None, obj=None): 
    """ Check whether the target name into the dataframe for supervised 
    learning.
    
    :param df: dataframe pandas
    :param target: str or index of the supervised learning target name. 
    
    :Example: 
        
        >>> from gofast.utils.mlutils import _assert_sl_target
        >>> from gofast.datasets import fetch_data
        >>> data = fetch_data('Bagoue original').get('data=df')  
        >>> _assert_sl_target (target =12, obj=prepareObj, df=data)
        ... 'flow'
    """
    is_dataframe = isinstance(df, pd.DataFrame)
    is_ndarray = isinstance(df, np.ndarray)
    if is_dataframe :
        targets = smart_format(
            df.columns if df.columns is not None else [''])
    else:targets =''
    
    if target is None:
        nameObj=f'{obj.__class__.__name__}'if obj is not None else 'Base class'
        msg =''.join([
            f"{nameObj!r} {'basically' if obj is not None else ''}"
            " works with surpervised learning algorithms so the",
            " input target is needed. Please specify the target", 
            f" {'name' if is_dataframe else 'index' if is_ndarray else ''}", 
            " to take advantage of the full functionalities."
            ])
        if is_dataframe:
            msg += f" Select the target among {targets}."
        elif is_ndarray : 
            msg += f" Max columns size is {df.shape[1]}"

        warnings.warn(msg, UserWarning)
        _logger.warning(msg)
        
    if target is not None: 
        if is_dataframe: 
            if isinstance(target, str):
                if not target in df.columns: 
                    msg =''.join([
                        f"Wrong target value {target!r}. Please select "
                        f"the right column name: {targets}"])
                    warnings.warn(msg, category= UserWarning)
                    _logger.warning(msg)
                    target =None
            elif isinstance(target, (float, int)): 
                is_ndarray =True 
  
        if is_ndarray : 
            _len = len(df.columns) if is_dataframe else df.shape[1] 
            m_=f"{'less than' if target >= _len  else 'greater than'}" 
            if not isinstance(target, (float,int)): 
                msg =''.join([f"Wrong target value `{target}`!"
                              f" Object type is {type(df)!r}. Target columns", 
                              " index should be given instead."])
                warnings.warn(msg, category= UserWarning)
                _logger.warning(msg)
                target=None
            elif isinstance(target, (float,int)): 
                target = int(target)
                if not 0 <= target < _len: 
                    msg =f" Wrong target index. Should be {m_} {str(_len-1)!r}."
                    warnings.warn(msg, category= UserWarning)
                    _logger.warning(msg) 
                    target =None
                    
            if df is None: 
                wmsg = ''.join([
                    f"No data found! `{target}` does not fit any data set.", 
                      "Could not fetch the target name.`df` argument is None.", 
                      " Need at least the data `numpy.ndarray|pandas.dataFrame`",
                      ])
                warnings.warn(wmsg, UserWarning)
                _logger.warning(wmsg)
                target =None
                
            target = list(df.columns)[target] if is_dataframe else target
            
    return target

def handle_imbalance(
    X, y=None, strategy='oversample', 
    random_state=42, 
    target_col='target'
    ):
    """
    Handles imbalanced datasets by either oversampling the minority class or 
    undersampling the majority class.
    
    It supports inputs as pandas DataFrame/Series, numpy arrays, and allows 
    specifying the target variable either as a separate argument or as part 
    of the DataFrame.

    Parameters
    ----------
    X : pd.DataFrame, np.ndarray
        The features of the dataset. If `y` is None, `X` is expected to include 
        the target variable.
    y : pd.Series, np.ndarray, optional
        The target variable of the dataset. If None, `target_col` must be 
        specified, and `X` must be a DataFrame containing the target.
    strategy : str, optional
        The strategy to apply for handling imbalance: 'oversample' or 
        'undersample'. Default is 'oversample'.
    random_state : int, optional
        The random state for reproducible results. Default is 42.
    target_col : str, optional
        The name of the target column in `X` if `X` is a DataFrame and 
        `y` is None. Default is 'target'.
    
    Returns
    -------
    X_resampled, y_resampled : Resampled features and target variable.
        The types of `X_resampled` and `y_resampled` match the input types.

    Examples
    --------
    Using with numpy arrays:
    
    >>> import numpy as np 
    >>> import pandas as pd 
    >>> from gofast.utils.mlutils import handle_imbalance
    >>> X = np.array([[1, 2], [2, 3], [3, 4]])
    >>> y = np.array([0, 1, 0])
    >>> X_resampled, y_resampled = handle_imbalance(X, y)
    >>> print(X_resampled.shape, y_resampled.shape)
    (3, 2) (3,)
    Using with pandas DataFrame (including target column):
    
    >>> df = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [2, 3, 4], 'target': [0, 1, 0]})
    >>> X_resampled, y_resampled = handle_imbalance(df, target_col='target')
    >>> print(X_resampled.shape, y_resampled.value_counts())

    Using with pandas DataFrame and Series:
    
    >>> X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [2, 3, 4]})
    >>> y = pd.Series([0, 1, 0], name='target')
    >>> X_resampled, y_resampled = handle_imbalance(X, y)
    >>> print(X_resampled.shape, y_resampled.value_counts())
    """
    if y is None:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("`X` must be a DataFrame when `y` is None.")
        exist_features(X, target_col, name =target_col)
        y = X[target_col]
        X = X.drop(target_col, axis=1)

    if not _is_arraylike_1d(y): 
        raise TypeError ("Check `y`. Expect one-dimensional array.")

    if not isinstance (y, pd.Series) : 
        # squeeze y to keep 1d array for skipping value error 
        # when constructing pd.Series and 
        # ensure `y` is a Series with the correct name for 
        # easy concatenation and manipulation
        y =pd.Series (np.squeeze (y), name =target_col ) 
    else : target_col = y.name # reset the default target_col 
 
    # Check consistent length 
    check_consistent_length(X, y )
    
    if isinstance(X, pd.DataFrame):
        data = pd.concat([X, y], axis=1)
    elif isinstance (X, np.ndarray ): 
        # Ensure `data` from `X` is a DataFrame with correct 
        # column names for subsequent operations
        data = pd.DataFrame(
            np.column_stack([X, y]), columns=[*X.columns, y.name]
            if isinstance(X, pd.DataFrame) else [
                    *[f"feature_{i}" for i in range(X.shape[1])], y.name])
    else: 
        TypeError("Unsupported type for X. Must be np.ndarray or pd.DataFrame.")
        
    # Identify majority and minority classes
    majority_class = y.value_counts().idxmax()
    minority_class = y.value_counts().idxmin()

    # Correctly determine the number of samples for resampling
    num_majority = y.value_counts()[majority_class]
    num_minority = y.value_counts()[minority_class]

    # Apply resampling strategy
    if strategy == 'oversample':
        minority_upsampled = resample(
            data[data[target_col] == minority_class],
            replace=True,
            n_samples=num_majority,
            random_state=random_state
            )
        resampled = pd.concat(
            [data[data[target_col] == majority_class], minority_upsampled])
    elif strategy == 'undersample':
        majority_downsampled = resample(
            data[data[target_col] == majority_class],
            replace=False,
            n_samples=num_minority,
            random_state=random_state
            )
        resampled = pd.concat(
            [data[data[target_col] == minority_class], majority_downsampled])

    # Prepare the output
    X_resampled = resampled.drop(target_col, axis=1)
    y_resampled = resampled[target_col]

    # Convert back to the original input type if necessary
    if isinstance(X, np.ndarray):
        X_resampled = X_resampled.to_numpy()
        y_resampled = y_resampled.to_numpy()

    return X_resampled, y_resampled

def make_pipe(
    X, 
    y =None, *,   
    num_features = None, 
    cat_features=None, 
    label_encoding='LabelEncoder', 
    scaler = 'StandardScaler' , 
    missing_values =np.nan, 
    impute_strategy = 'median', 
    sparse_output=True, 
    for_pca =False, 
    transform =False, 
    ): 
    """ make a pipeline to transform data at once. 
    
    make a quick pipeline is usefull to fast preprocess the data at once 
    for quick prediction. 
    
    Work with a pandas dataframe. If `None` features is set, the numerical 
    and categorial features are automatically retrieved. 
    
    Parameters
    ---------
    X : pandas dataframe of shape (n_samples, n_features)
        The input samples. Use ``dtype=np.float32`` for maximum
        efficiency. Sparse matrices are also supported, use sparse
        ``csc_matrix`` for maximum efficiency.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Target relative to X for classification or regression;
        None for unsupervised learning.
    num_features: list or str, optional 
        Numerical features put on the list. If `num_features` are given  
        whereas `cat_features` are ``None``, `cat_features` are figured out 
        automatically.
    cat_features: list of str, optional 
        Categorial features put on the list. If `num_features` are given 
        whereas `num_features` are ``None``, `num_features` are figured out 
        automatically.
    label_encoding: callable or str, default='sklearn.preprocessing.LabelEncoder'
        kind of encoding used to encode label. This assumes 'y' is supplied. 
    scaler: callable or str , default='sklearn.preprocessing.StandardScaler'
        kind of scaling used to scaled the numerical data. Note that for 
        the categorical data encoding, 'sklearn.preprocessing.OneHotEncoder' 
        is implemented  under the hood instead. 
    missing_values : int, float, str, np.nan, None or pandas.NA, default=np.nan
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For pandas' dataframes with
        nullable integer dtypes with missing values, `missing_values`
        can be set to either `np.nan` or `pd.NA`.
    
    impute_strategy : str, default='mean'
        The imputation strategy.
    
        - If "mean", then replace missing values using the mean along
          each column. Can only be used with numeric data.
        - If "median", then replace missing values using the median along
          each column. Can only be used with numeric data.
        - If "most_frequent", then replace missing using the most frequent
          value along each column. Can be used with strings or numeric data.
          If there is more than one such value, only the smallest is returned.
        - If "constant", then replace missing values with fill_value. Can be
          used with strings or numeric data.
    
           strategy="constant" for fixed value imputation.
           
    sparse_output : bool, default=False
        Is used when label `y` is given. Binarize labels in a one-vs-all 
        fashion. If ``True``, returns array from transform is desired to 
        be in sparse CSR format.
        
    for_pca:bool, default=False, 
        Transform data for principal component ( PCA) analysis. If set to 
        ``True``, :class:`sklearn.preprocessing.OrdinalEncoder`` is used insted 
        of :class:sklearn.preprocessing.OneHotEncoder``. 
        
    transform: bool, default=False, 
        Tranform data inplace rather than returning the naive pipeline. 
        
    Returns
    ---------
    full_pipeline: :class:`gofast.exlib.sklearn.FeatureUnion`
        - Full pipeline composed of numerical and categorical pipes 
    (X_transformed &| y_transformed):  {array-like, sparse matrix} of \
        shape (n_samples, n_features)
        - Transformed data. 
        
        
    Examples 
    ---------
    >>> from gofast.utils.mlutils import make_naive_pipe 
    >>> from gofast.datasets import load_hlogs 
    
    (1) Make a naive simple pipeline  with RobustScaler, StandardScaler 
    >>> from gofast.exlib.sklearn import RobustScaler 
    >>> X_, y_ = load_hlogs (as_frame=True )# get all the data  
    >>> pipe = make_naive_pipe(X_, scaler =RobustScaler ) 
    
    (2) Transform X in place with numerical and categorical features with 
    StandardScaler (default). Returned CSR matrix 
    
    >>> make_naive_pipe(X_, transform =True )
    ... <181x40 sparse matrix of type '<class 'numpy.float64'>'
    	with 2172 stored elements in Compressed Sparse Row format>

    """
    
    from ..transformers import DataFrameSelector
    
    sc= {"StandardScaler": StandardScaler ,"MinMaxScaler": MinMaxScaler , 
         "Normalizer":Normalizer , "RobustScaler":RobustScaler}

    if not hasattr (X, '__array__'):
        raise TypeError(f"'make_naive_pipe' not supported {type(X).__name__!r}."
                        " Expects X as 'pandas.core.frame.DataFrame' object.")
    X = check_array (
        X, 
        dtype=object, 
        force_all_finite="allow-nan", 
        to_frame=True, 
        input_name="Array for transforming X or making naive pipeline"
        )
    if not hasattr (X, "columns"):
        # create naive column for 
        # Dataframe selector 
        X = pd.DataFrame (
            X, columns = [f"naive_{i}" for i in range (X.shape[1])]
            )
    #-> Encode y if given
    if y is not None: 
        # if (label_encoding =='labelEncoder'  
        #     or get_estimator_name(label_encoding) =='LabelEncoder'
        #     ): 
        #     enc =LabelEncoder()
        if  ( label_encoding =='LabelBinarizer' 
                or get_estimator_name(label_encoding)=='LabelBinarizer'
               ): 
            enc =LabelBinarizer(sparse_output=sparse_output)
        else: 
            label_encoding =='labelEncoder'
            enc =LabelEncoder()
            
        y= enc.fit_transform(y)
    #set features
    if num_features is not None: 
        cat_features, num_features  = bi_selector(
            X, features= num_features 
            ) 
    elif cat_features is not None: 
        num_features, cat_features  = bi_selector(
            X, features= cat_features 
            )  
    if ( cat_features is None 
        and num_features is None 
        ): 
        num_features , cat_features = bi_selector(X ) 
    # assert scaler value 
    if get_estimator_name (scaler)  in sc.keys(): 
        scaler = sc.get (get_estimator_name(scaler )) 
    elif ( any ( [v.lower().find (str(scaler).lower()) >=0
                  for v in sc.keys()])
          ):  
        for k, v in sc.items () :
            if k.lower().find ( str(scaler).lower() ) >=0: 
                scaler = v ; break 
    else : 
        msg = ( f"Supports {smart_format( sc.keys(), 'or')} or "
                "other scikit-learn scaling objects, got {!r}" 
                )
        if hasattr (scaler, '__module__'): 
            name = getattr (scaler, '__module__')
            if getattr (scaler, '__module__') !='sklearn.preprocessing._data':
                raise ValueError (msg.format(name ))
        else: 
            name = scaler.__name__ if callable (scaler) else (
                scaler.__class__.__name__ ) 
            raise ValueError (msg.format(name ))
    # make pipe 
    npipe = [
            ('imputerObj',SimpleImputer(missing_values=missing_values , 
                                    strategy=impute_strategy)),                
            ('scalerObj', scaler() if callable (scaler) else scaler ), 
            ]
    
    if len(num_features)!=0 : 
       npipe.insert (
            0,  ('selectorObj', DataFrameSelector(columns= num_features))
            )

    num_pipe=Pipeline(npipe)
    
    if for_pca : 
        encoding=  ('OrdinalEncoder', OrdinalEncoder())
    else:  encoding =  (
        'OneHotEncoder', OneHotEncoder())
        
    cpipe = [
        encoding
        ]
    if len(cat_features)!=0: 
        cpipe.insert (
            0, ('selectorObj', DataFrameSelector(columns= cat_features))
            )

    cat_pipe = Pipeline(cpipe)
    # make transformer_list 
    transformer_list = [
        ('num_pipeline', num_pipe),
        ('cat_pipeline', cat_pipe), 
        ]

    #remove num of cat pipe if one of them is 
    # missing in the data 
    if len(cat_features)==0: 
        transformer_list.pop(1) 
    if len(num_features )==0: 
        transformer_list.pop(0)
        
    full_pipeline =FeatureUnion(transformer_list=transformer_list) 
    
    return  ( full_pipeline.fit_transform (X) if y is None else (
        full_pipeline.fit_transform (X), y ) 
             ) if transform else full_pipeline

@ensure_pkg (
    "imblearn", 
    partial_check=True, 
    condition="balance_classes", 
    extra= (
        "Synthetic Minority Over-sampling Technique (SMOTE) cannot be used."
        " Note,`imblearn` is actually a shorthand for ``imbalanced-learn``."
        ), 
   )
def build_data_preprocessor(
    X: Union [NDArray, DataFrame], 
    y: Optional[ArrayLike] = None, *,  
    num_features: Optional[List[str]] = None, 
    cat_features: Optional[List[str]] = None, 
    custom_transformers: Optional[List[Tuple[str, TransformerMixin]]] = None,
    label_encoding: Union[str, TransformerMixin] = 'LabelEncoder', 
    scaler: Union[str, TransformerMixin] = 'StandardScaler', 
    missing_values: Union[int, float, str, None] = np.nan, 
    impute_strategy: str = 'median', 
    feature_interaction: bool = False,
    dimension_reduction: Optional[Union[str, TransformerMixin]] = None,
    feature_selection: Optional[Union[str, TransformerMixin]] = None,
    balance_classes: bool = False,
    advanced_imputation: Optional[TransformerMixin] = None,
    verbose: bool = False,
    output_format: str = 'array',
    transform: bool = False,
    **kwargs: Any
) -> Any:
    """
    Create a preprocessing pipeline for data transformation and feature engineering.

    Function constructs a pipeline to preprocess data for machine learning tasks, 
    accommodating a variety of transformations including scaling, encoding, 
    and dimensionality reduction. It supports both numerical and categorical data, 
    and can incorporate custom transformations.

    Parameters
    ----------
    X : np.ndarray or DataFrame
        Input features dataframe or arraylike. Must be two dimensional array.
    y : array-like, optional
        Target variable. Required for supervised learning tasks.
    num_features : list of str, optional
        List of numerical feature names. If None, determined automatically.
    cat_features : list of str, optional
        List of categorical feature names. If None, determined automatically.
    custom_transformers : list of tuples, optional
        Custom transformers to be included in the pipeline. Each tuple should 
        contain ('name', transformer_instance).
    label_encoding : str or transformer, default 'LabelEncoder'
        Encoder for the target variable. Accepts standard scikit-learn encoders 
        or custom encoder objects.
    scaler : str or transformer, default 'StandardScaler'
        Scaler for numerical features. Accepts standard scikit-learn scalers 
        or custom scaler objects.
    missing_values : int, float, str, np.nan, None, default np.nan
        Placeholder for missing values for imputation.
    impute_strategy : str, default 'median'
        Imputation strategy. Options: 'mean', 'median', 'most_frequent', 'constant'.
    feature_interaction : bool, default False
        If True, generate polynomial and interaction features.
    dimension_reduction : str or transformer, optional
        Dimensionality reduction technique. Accepts 'PCA', 't-SNE', or custom object.
    feature_selection : str or transformer, optional
        Feature selection method. Accepts 'SelectKBest', 'SelectFromModel', or custom object.
    balance_classes : bool, default False
        If True, balance classes in classification tasks.
    advanced_imputation : transformer, optional
        Advanced imputation technique like KNNImputer or IterativeImputer.
    verbose : bool, default False
        Enable verbose output.
    output_format : str, default 'array'
        Desired output format: 'array' or 'dataframe'.
    transform : bool, default False
        If True, apply the pipeline to the data immediately and return transformed data.

    Returns
    -------
    full_pipeline : Pipeline or (X_transformed, y_transformed)
        The constructed preprocessing pipeline, or transformed data if `transform` is True.

    Examples
    --------
    >>> from gofast.utils.mlutils import build_data_preprocessor
    >>> from gofast.datasets import load_hlogs
    >>> X, y = load_hlogs(as_frame=True, return_X_y=True)
    >>> pipeline = build_data_preprocessor(X, y, scaler='RobustScaler')
    >>> X_transformed = pipeline.fit_transform(X)
    
    """
    sc= {"StandardScaler": StandardScaler ,"MinMaxScaler": MinMaxScaler , 
         "Normalizer":Normalizer , "RobustScaler":RobustScaler}

    if not isinstance (X, pd.DataFrame): 
        # create fake dataframe for handling columns features 
        X= pd.DataFrame(X)
    # assert scaler value 
    if get_estimator_name (scaler) in sc.keys(): 
        scaler = sc.get (get_estimator_name(scaler ))() 
        
    # Define numerical and categorical pipelines
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy=impute_strategy, missing_values=missing_values)),
        ('scaler', StandardScaler() if scaler == 'StandardScaler' else scaler)
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent', 
                                  missing_values=missing_values)),
        ('encoder', OneHotEncoder() if label_encoding in ('LabelEncoder', 'onehot')
         else label_encoding)
    ])

    # Determine automatic feature selection if not provided
    if num_features is None and cat_features is None:
        num_features = make_column_selector(dtype_include=['int', 'float'])(X)
        cat_features = make_column_selector(dtype_include='object')(X)

    # Feature Union for numerical and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, num_features),
            ('cat', categorical_pipeline, cat_features)
        ])

    # Add custom transformers if any
    if custom_transformers:
        for name, transformer in custom_transformers:
            preprocessor.named_transformers_[name] = transformer

    # Feature interaction, selection, and dimension reduction
    steps = [('preprocessor', preprocessor)]
    
    if feature_interaction:
        steps.append(('interaction', PolynomialFeatures()))
    if feature_selection:
        steps.append(('feature_selection', SelectKBest() 
                      if feature_selection == 'SelectKBest' else feature_selection))
    if dimension_reduction:
        steps.append(('dim_reduction', PCA() if dimension_reduction == 'PCA' 
                      else dimension_reduction))

    # Final pipeline
    pipeline = Pipeline(steps)

   # Advanced imputation logic if required
    if advanced_imputation:
        from sklearn.experimental import enable_iterative_imputer # noqa
        from sklearn.impute import IterativeImputer
        if advanced_imputation == 'IterativeImputer':
            steps.insert(0, ('advanced_imputer', IterativeImputer(
                estimator=RandomForestClassifier(), random_state=42)))
        else:
            steps.insert(0, ('advanced_imputer', advanced_imputation))

    # Final pipeline before class balancing
    pipeline = Pipeline(steps)

    # Class balancing logic if required
    if balance_classes and y is not None:
        if str(balance_classes).upper() == 'SMOTE':
            from imblearn.over_sampling import SMOTE
            # Note: SMOTE works on numerical data, so it's applied after initial pipeline
            pipeline = Pipeline([('preprocessing', pipeline), (
                'smote', SMOTE(random_state=42))])

    # Transform data if transform flag is set
    # if transform:
    output_format = output_format or 'array' # force none to hold array
    if str(output_format) not in ('array', 'dataframe'): 
        raise ValueError(f"Invalid '{output_format}', expect 'array' or 'dataframe'.")
        
    return _execute_transformation(
        pipeline, X, y, transform, output_format, label_encoding)

def _execute_transformation(
        pipeline, X, y, transform, output_format, label_encoding):
    """ # Transform data if transform flag is set or return pipeline"""
    if transform:
        X_transformed = pipeline.fit_transform(X)
        if y is not None:
            y_transformed = _transform_target(y, label_encoding) if label_encoding else y
            return (X_transformed, y_transformed) if output_format == 'array' else (
                pd.DataFrame(X_transformed), pd.Series(y_transformed))
        
        return X_transformed if output_format == 'array' else pd.DataFrame(X_transformed)
    
    return pipeline

def _transform_target(y, label_encoding:Union [BaseEstimator, TransformerMixin] ):
    if label_encoding == 'LabelEncoder':
        encoder = LabelEncoder()
        return encoder.fit_transform(y)
    elif isinstance(label_encoding, (BaseEstimator, TransformerMixin)):
        return label_encoding.fit_transform(y)
    else:
        raise ValueError("Unsupported label_encoding value: {}".format(label_encoding)) 


@SmartProcessor(fail_silently=True, param_name ="skip_columns")
def soft_imputer(
    X, 
    strategy='mean', 
    missing_values=np.nan, 
    fill_value=None, 
    drop_features=False, 
    mode=None, 
    copy=True, 
    verbose=0, 
    add_indicator=False,
    keep_empty_features=False, 
    skip_columns=None, 
    **kwargs
    ):
    """
    Impute missing values in a dataset, optionally dropping features and handling 
    both numerical and categorical data.

    This function extends the functionality of scikit-learn's SimpleImputer to 
    support dropping specified features and a ``bi-impute`` mode for handling 
    both numerical and categorical data. It ensures API consistency with 
    scikit-learn's transformers and allows for flexible imputation strategies.

    Parameters
    ----------
    X : array-like or sparse matrix of shape (n_samples, n_features)
        The input data to impute.
        
    strategy : str, default='mean'
        The imputation strategy:
        - 'mean': Impute using the mean of each column. Only for numeric data.
        - 'median': Impute using the median of each column. Only for numeric data.
        - 'most_frequent': Impute using the most frequent value of each column. 
          For numeric and categorical data.
        - 'constant': Impute using the specified `fill_value`.
        
    missing_values : int, float, str, np.nan, None, or pd.NA, default=np.nan
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed.
        
    fill_value : str or numerical value, default=None
        When `strategy` == 'constant', `fill_value` is used to replace all
        occurrences of `missing_values`. If left to the default, `fill_value` 
        will be 0 when imputing numerical data and 'missing_value' for strings 
        or object data types.
        For 'constant' strategy, specifies the value used for replacement.
        In 'bi-impute' mode, allows specifying separate fill values for
        numerical and categorical data using a delimiter from the set
        {"__", "--", "&", "@", "!"}. For example, "0.5__missing" indicates
        0.5 as fill value for numerical data and "missing" for categorical.
        
    drop_features : bool or list, default=False
        If True, drops all categorical features before imputation. If a list, 
        drops specified features.
        
    mode : str, optional
        If set to 'bi-impute', imputes both numerical and categorical features 
        and returns a single imputed dataframe. Only 'bi-impute' is supported.
        
    copy : bool, default=True
        If True, a copy of X will be created. If False, imputation will
        be done in-place whenever possible.
        
    verbose : int, default=0
        Controls the verbosity of the imputer.
        
    add_indicator : bool, default=False
        If True, a `MissingIndicator` transform will be added to the output 
        of the imputer's transform.
        
    keep_empty_features : bool, default=False
        If True, features that are all missing when `fit` is called are 
        included in the transform output.
        
    skip_columns : list of str or int, optional
        Specifies the columns to exclude from processing when the decorator is
        applied to a function. If the input data `X` is a pandas DataFrame, 
        `skip_columns` should contain the names of the columns to be skipped. 
        If `X` is a numpy array, `skip_columns`  should be a list of column 
        indices (integers) indicating the positions of the columns to be excluded.
        This allows selective processing of data, avoiding alterations to the
        specified columns.
        
    **kwargs : dict
        Additional fitting parameters.

    Returns
    -------
    Xi : array-like or sparse matrix of shape (n_samples, n_features)
        The imputed dataset.

    Examples
    --------
    >>> import numpy as np 
    >>> from gofast.utils.mlutils import soft_imputer
    >>> X = np.array([[1, np.nan, 3], [4, 5, np.nan], [np.nan, np.nan, 9]])
    >>> soft_imputer(X, strategy='mean')
    array([[ 1. ,  5. ,  3. ],
           [ 4. ,  5. ,  6. ],
           [ 2.5,  5. ,  9. ]])
    
    >>> df = pd.DataFrame({'A': [1, 2, np.nan], 'B': ['a', np.nan, 'b']})
    >>> soft_imputer(df, strategy='most_frequent', mode='bi-impute')
               A  B
        0    1.0  a
        1    2.0  a
        2    1.5  b

    Notes
    -----
    The 'bi-impute' mode requires categorical features to be explicitly indicated
    as such by using pandas Categorical dtype or by specifying features to drop.
    """
    X, is_frame  = _convert_to_dataframe(X)
    X = _drop_features(X, drop_features)
    
    if mode == 'bi-impute':
        fill_values, strategies = _enabled_bi_impute_mode (strategy, fill_value)
        try: 
            num_imputer = SimpleImputer(
                strategy=strategies[0], missing_values=missing_values,
                fill_value=fill_values[0])
        except ValueError as e: 
            msg= (" Consider using the {'__', '--', '&', '@', '!'} delimiter"
                  " for mixed numeric and categorical fill values.")
            # Improve the error message 
            raise ValueError(
                f"Imputation failed due to: {e}." +
                msg if check_mixed_data_types(X) else ''
                )
        cat_imputer = SimpleImputer(strategy= strategies[1], 
                                    missing_values=missing_values,
                                    fill_value = fill_values [1]
                                    )
        num_imputed, cat_imputed, num_columns, cat_columns = _separate_and_impute(
            X, num_imputer, cat_imputer)
        Xi = np.hstack((num_imputed, cat_imputed))
        new_columns = num_columns + cat_columns
        Xi = pd.DataFrame(Xi, index=X.index, columns=new_columns)
    else:
        try:
            Xi, imp = _impute_data(
                X, strategy, missing_values, fill_value, add_indicator, copy)
        except Exception as e : 
            raise ValueError("Imputation failed. Consider using the"
                             " 'bi-impute' mode for mixed data types.") from e 
        if isinstance(X, pd.DataFrame):
            Xi = pd.DataFrame(Xi, index=X.index, columns=imp.feature_names_in_)
            
    if not is_frame: # revert back to array
        Xi = np.array ( Xi )
    return Xi

def _convert_to_dataframe(X):
    """Ensure input is a pandas DataFrame."""
    is_frame=True 
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(check_array(X, dtype=None, force_all_finite='allow-nan'), 
                         columns=[f'feature_{i}' for i in range(X.shape[1])])
        is_frame=False 
    return X, is_frame 

def _drop_features(X, drop_features):
    """Drop specified features from the DataFrame."""
    if isinstance(drop_features, bool) and drop_features:
        X = X.select_dtypes(exclude=['object', 'category'])
    elif isinstance(drop_features, list):
        X = X.drop(columns=drop_features, errors='ignore')
    return X

def _impute_data(X, strategy, missing_values, fill_value, add_indicator, copy):
    """Impute the dataset using SimpleImputer."""
    imp = SimpleImputer(strategy=strategy, missing_values=missing_values, 
                        fill_value=fill_value, add_indicator=add_indicator, 
                        copy=copy)
    Xi = imp.fit_transform(X)
    return Xi, imp

def _separate_and_impute(X, num_imputer, cat_imputer):
    """Separate and impute numerical and categorical features."""
    X, num_columns, cat_columns= to_numeric_dtypes(X, return_feature_types=True )

    if len(num_columns) > 0:
        num_imputed = num_imputer.fit_transform(X[num_columns])
    else:
        num_imputed = np.array([]).reshape(X.shape[0], 0)
    
    if len(cat_columns) > 0:
        cat_imputed = cat_imputer.fit_transform(X[cat_columns])
    else:
        cat_imputed = np.array([]).reshape(X.shape[0], 0)
    return num_imputed, cat_imputed, num_columns, cat_columns

def _enabled_bi_impute_mode(
    strategy: str, fill_value: Union[str, float, None]
     ) -> Tuple[List[Union[None, float, str]], List[str]]:
    """
    Determines strategies and fill values for numerical and categorical data
    in bi-impute mode based on the provided strategy and fill value.

    Parameters
    ----------
    strategy : str
        The imputation strategy to use.
    fill_value : Union[str, float, None]
        The fill value to use for imputation, which can be a float, string, 
        or None. When a string containing delimiters is provided, it indicates
        separate fill values for numerical and categorical data.

    Returns
    -------
    Tuple[List[Union[None, float, str]], List[str]]
        A tuple containing two lists: the first with fill values for numerical
        and categorical data, and the second with strategies for numerical and
        categorical data.

    Examples
    --------
    >>> from gofast.utils.mlutils import _enabled_bi_impute_mode
    >>> enabled_bi_impute_mode('mean', None)
    ([None, None], ['mean', 'most_frequent'])

    >>> _enabled_bi_impute_mode('constant', '0__missing')
    ([0.0, 'missing'], ['constant', 'constant'])
    
    >>> _enabled_bi_impute_mode (strategy='constant', fill_value="missing")
    ([0.0, 'missing'], ['constant', 'constant'])
    
    >>> _enabled_bi_impute_mode('constant', 9.) 
    ([9.0, None], ['constant', 'most_frequent'])
    
    >>> _enabled_bi_impute_mode(strategy='constant', fill_value="mean__missing",)
    ([None, 'missing'], ['mean', 'constant'])
    """
    num_strategy, cat_strategy = 'most_frequent', 'most_frequent'
    fill_values = [None, None]
    
    if fill_value is None or isinstance(fill_value, (float, int)):
        if strategy in ["mean", 'median', 'constant']:
            num_strategy = strategy
            fill_values[0] = ( 
                0.0 if strategy == 'constant' 
                and fill_value is None else fill_value
                )
        return fill_values, [num_strategy, cat_strategy]

    if contains_delimiter(fill_value,{"__", "--", "&", "@", "!"} ):
        fill_values, strategies = _manage_fill_value(fill_value, strategy)
    else:
        fill_value = (
            f"{strategy}__{fill_value}" if strategy in ['mean', 'median'] 
            else ( f"0__{fill_value}" if strategy =="constant" else fill_value )
        )
        fill_values, strategies = _manage_fill_value(fill_value, strategy)
    
    return fill_values, strategies

def _manage_fill_value(
    fill_value: str, strategy: str
    ) -> Tuple[List[Union[None, float, str]], List[str]]:
    """
    Parses and manages fill values for bi-impute mode, supporting mixed types.

    Parameters
    ----------
    fill_value : str
        The fill value string potentially containing mixed types for numerical
        and categorical data.
    strategy : str
        The imputation strategy to determine how to handle numerical fill values.

    Returns
    -------
    Tuple[List[Union[None, float, str]], List[str]]
        A tuple containing two elements: the first is a list with numerical and
        categorical fill values, and the second is a list of strategies for 
        numerical and categorical data.

    Raises
    ------
    ValueError
        If the fill value does not contain a proper separator or if the numerical
        fill value is incompatible with the specified strategy.

    Examples
    --------
    >>> from gofast.utils.mlutils import _manage_fill_value
    >>> _manage_fill_value("0__missing", "constant")
    ([0.0, 'missing'], ['constant', 'constant'])

    >>> _manage_fill_value("mean__missing", "mean")
    ([None, 'missing'], ['mean', 'constant'])
    """
    regex = re.compile(r'(__|--|&|@|!)')
    parts = regex.split(fill_value)
    if len(parts) < 3:
        raise ValueError("Fill value must contain a separator (__|--|&|@|!)"
                         " between numerical and categorical fill values.")

    num_fill, cat_fill = parts[0], parts[-1]
    num_fill_value = None if strategy in ['mean', 'median'] and num_fill.lower(
        ) in ['mean', 'median'] else num_fill

    try:
        num_fill_value = float(num_fill) if num_fill.replace('.', '', 1).isdigit() else num_fill
    except ValueError:
        raise ValueError(f"Numerical fill value '{num_fill}' must be a float"
                         f" for strategy '{strategy}'.")
    strategies =[ strategy if strategy in ['mean', 'median', 'constant']
                 else 'most_frequent', 'constant'] 
    if num_fill.lower() in ['mean', 'median'] and strategy=='constant': 
        # Permutate the strategy and fill value. 
        strategies [0]= num_fill.lower()
        num_fill_value =None ; 
    
    return [num_fill_value, cat_fill], strategies

@SmartProcessor(fail_silently= True, param_name="skip_columns")
def soft_scaler(
    X, *, 
    kind=StandardScaler, 
    copy=True, 
    with_mean=True, 
    with_std=True, 
    feature_range=(0, 1), 
    clip=False, 
    norm='l2',
    skip_columns=None, 
    verbose=0, 
    **kwargs
    ):
    """
    Scale data using specified scaling strategy from scikit-learn. 
    
    Function excludes categorical features from scaling and provides 
    feedback via verbose parameter.

    Parameters
    ----------
    X : DataFrame or array-like of shape (n_samples, n_features)
        The data to scale, can contain both numerical and categorical features.
    kind : str, default='StandardScaler'
        The kind of scaling to apply to numerical features. One of 'StandardScaler', 
        'MinMaxScaler', 'Normalizer', or 'RobustScaler'.
    copy : bool, default=True
        If False, avoid a copy and perform inplace scaling instead.
    with_mean : bool, default=True
        If True, center the data before scaling. Only applicable when kind is
        'StandardScaler' or 'RobustScaler'.
    with_std : bool, default=True
        If True, scale the data to unit variance. Only applicable when kind is
        'StandardScaler' or 'RobustScaler'.
    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data. Only applicable when kind 
        is 'MinMaxScaler'.
    clip : bool, default=False
        Set to True to clip transformed values to the provided feature range.
        Only applicable when kind is 'MinMaxScaler'.
    norm : {'l1', 'l2', 'max'}, default='l2'
        The norm to use to normalize each non-zero sample or feature.
        Only applicable when kind is 'Normalizer'.
    skip_columns : list of str or int, optional
        Specifies the columns to exclude from processing when the decorator is
        applied to a function. If the input data `X` is a pandas DataFrame, 
        `skip_columns` should contain the names of the columns to be skipped. 
        If `X` is a numpy array, `skip_columns`  should be a list of column 
        indices (integers) indicating the positions of the columns to be excluded.
        This allows selective processing of data, avoiding alterations to the
        specified columns.
    verbose : int, default=0
        If > 0, warning messages about the processing.
        
    **kwargs : additional keyword arguments
        Additional fitting parameters to pass to the scaler.
        
    Returns
    -------
    X_scaled : {ndarray, sparse matrix, dataframe} of shape (n_samples, n_features)
        The scaled data. The scaled data with numerical features scaled according
        to the specified kind, and categorical features returned unchanged. 
        The return type matches the input type.

    Examples
    --------
    >>> import numpy as np 
    >>> import pandas as pd 
    >>> from gofast.utils.mlutils import soft_scaler
    >>> X = np.array([[1, -1, 2], [2, 0, 0], [0, 1, -1]])
    >>> X_scaled = soft_scaler(X, kind='StandardScaler')
    >>> print(X_scaled)
    [[ 0.  -1.22474487  1.33630621]
     [ 1.22474487  0.  -0.26726124]
     [-1.22474487  1.22474487 -1.06904497]]

    >>> df = pd.DataFrame(X, columns=['a', 'b', 'c'])
    >>> df_scaled = soft_scaler(df, kind='RobustScaler', with_centering=True,
                                with_scaling=True)
    >>> print(df_scaled)
              a    b    c
        0 -0.5 -1.0  1.0
        1  0.5  0.0  0.0
        2 -0.5  1.0 -1.0
    """
    X= to_numeric_dtypes(X)
    input_is_dataframe = isinstance(X, pd.DataFrame)
    cat_features = X.select_dtypes(
        exclude=['number']).columns if input_is_dataframe else []

    if verbose > 0 and len(cat_features) > 0:
        warnings.warn(
            "Note: Categorical data detected and excluded from scaling.")

    kind= kind if isinstance(kind, str) else kind.__name__
    scaler = _determine_scaler(
        kind, copy=copy, with_mean=with_mean, with_std=with_std, norm=norm, 
        feature_range=feature_range, clip=clip, **kwargs)

    if input_is_dataframe:
        num_features = X.select_dtypes(['number']).columns
        X_scaled_numeric = _scale_numeric_features(X, scaler, num_features)
        X_scaled = _concat_scaled_numeric_with_categorical(
            X_scaled_numeric, X, cat_features)
    else:
        X_scaled = scaler.fit_transform(X)
    
    return X_scaled

def _determine_scaler(kind, **kwargs):
    """
    Determines the scaler based on the kind parameter.
    """
    scaler_classes = {
        'StandardScaler': StandardScaler,
        'MinMaxScaler': MinMaxScaler,
        'Normalizer': Normalizer,
        'RobustScaler': RobustScaler
    }
    scaler_class = scaler_classes.get(kind, None)
    if scaler_class is None:
        raise ValueError(f"Unsupported scaler kind: {kind}. Supported scalers"
                         f" are: {', '.join(scaler_classes.keys())}.")
    kwargs= get_valid_kwargs(scaler_class, **kwargs)
    return scaler_class(**kwargs)

def _scale_numeric_features(X, scaler, num_features):
    """
    Scales numerical features of the DataFrame X using the provided scaler.
    """
    return scaler.fit_transform(X[num_features])

def _concat_scaled_numeric_with_categorical(X_scaled_numeric, X, cat_features):
    """
    Concatenates scaled numerical features with original categorical features.
    """
    X_scaled = pd.concat([pd.DataFrame(X_scaled_numeric, index=X.index, 
                        columns=X.select_dtypes(['number']).columns),
                          X[cat_features]], axis=1)
    return X_scaled[X.columns]  # Maintain original column order

@is_data_readable
@Dataify(auto_columns=True, prefix='feature_')
@validate_params ({ 
    "data": ['array-like'], 
    "target": [str, 'array-like'], 
    "model": [HasMethods(['fit', 'predict']), None], 
    "columns": [str, list, None], 
    "proxy_name": [str, None], 
    "infer_data": [bool], 
    "random_state": ['random_state', None], 
    'verbose': [bool]
    })
def generate_proxy_feature(
    data,
    target,
    model=None,
    columns=None,
    proxy_name=None,
    infer_data=False,
    random_state=None, 
    verbose=False
):
    """
    Generate a proxy feature based on the available features in the dataset 
    (both numeric and categorical).

    The function generates a proxy feature by training a machine learning 
    model (default: RandomForestRegressor) on the input dataset, using a 
    subset of features (numeric and categorical) and the specified target. 
    The model's predictions on the input data are returned as the proxy 
    feature.

    Parameters:
    ----------
    data : pd.DataFrame
        The input dataset containing both numeric and categorical features.

        - `data` is a pandas DataFrame that includes both numeric and 
          categorical features, with the target column specified separately.
        
    target : str or array-like
        The target column or an array-like object containing the target values.

        - `target` refers to the name of the column in `data` that holds the 
          target values to be predicted by the model.

    model : estimator object, default=None
        The machine learning model to use for generating the proxy feature.
        If None, defaults to RandomForestRegressor.

        - `model` is a scikit-learn estimator used to fit the model on the data. 
          If no model is provided, a default RandomForestRegressor will be used.
        
    columns : list of str, optional
        A list of feature names to use for generating the proxy. 
        If None, all features except the target column are used.

        - `columns` is an optional parameter specifying which features from 
          `data` will be used in model training. If not specified, all 
          columns except the target column are included.

    proxy_name : str, default="Proxy_Feature"
        The name for the generated proxy feature column.

        - `proxy_name` is the name that will be assigned to the new proxy 
          feature. The default is "Proxy_Feature".

    infer_data : bool, default=False
        If True, the proxy feature is added to the original dataframe. 
        If False, only the generated proxy feature is returned.

        - `infer_data` controls whether the proxy feature is added to the 
          original dataset (`True`) or returned as a separate series (`False`).

    random_state : int, default=42
        Random seed for reproducibility of results.

        - `random_state` ensures that the random processes (like train-test 
          splitting) are reproducible across runs.

    verbose : bool, default=False
        If True, additional output is printed for model performance and 
        feature importance.

        - `verbose` controls whether the Mean Squared Error (MSE) on the 
          test set and feature importances are printed for model evaluation.

    Returns
    ---------
    pd.Series or pd.DataFrame
        - If `infer_data=True`, returns the original dataframe with the new 
          proxy feature added.
        - If `infer_data=False`, returns only the generated proxy feature 
          as a pd.Series.

    Notes
    ------
    The proxy feature is generated using the trained model's predictions, 
    which are made based on the input features. This proxy feature can serve 
    as a useful representation or summarization of the underlying data, 
    especially when trying to approximate or predict the target variable based 
    on existing features. The model is trained on a subset of data (numeric and 
    categorical features) and then applied to the entire dataset to create the 
    proxy feature.

    Example:
    --------
    >>> from gofast.utils.mlutils import generate_proxy_feature
    >>> import pandas as pd

    # Create a sample dataframe
    >>> data = pd.DataFrame({
    >>>     'col1': [1, 2, 3], 'col2': [4, 5, 6], 'target': [0, 1, 0]
    >>> })
    
    # Generate the proxy feature
    >>> result = generate_proxy_feature(
    >>>     data, target='target', proxy_name="Generated_Proxy", infer_data=True
    >>> )
    
    >>> print(result)
       col1  col2  target  Generated_Proxy
    0     1     4       0         0.123456
    1     2     5       1         0.789012
    2     3     6       0         0.456789

    See Also
    ---------
    sklearn.ensemble.RandomForestRegressor : Default machine learning 
         model used for regression tasks.
    pandas.DataFrame.join` : Method to combine the original data with the 
        generated proxy feature.
    gofast.utils.mlutils.generate_dirichlet_features`: Generate 
        synthetic features using the Dirichlet distribution.
    
    References
    -----------
    .. [1] Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
    """

    from sklearn.metrics import mean_squared_error

    proxy_name = proxy_name or "Proxy_Feature"
    # Handle model parameter (defaults to RandomForestRegressor if None)
    if model is None:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(random_state=random_state)

    # Handle data and target through the helper function
    data, target_column = _manage_target(data, target)

    # Handle columns parameter (defaults to all columns except target_column)
    if columns is None:
        columns = data.drop(columns=[target_column]).columns.tolist()

    # For consistency, make sure that columns exist in the dataframe
    exist_features(data, features=columns)

    # Separate target column from features
    X = data[columns]
    y = data[target_column]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Define preprocessor for both numeric and categorical features
    numeric_features = select_dtypes(X, dtypes=['int64', 'float64']).columns  
    categorical_features= select_dtypes(X, dtypes=['object']).columns 

    # Create a column transformer with imputation, scaling for numeric,
    # and encoding for categorical features
    # Handle missing values & Encode categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values
                ('scaler', StandardScaler())  # Standardize the numeric features
            ]), numeric_features),
            
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))  
            ]), categorical_features)
        ]
    )

    # Create the model pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = pipeline.predict(X_test)

    # Optionally print the model performance (e.g., Mean Squared Error)
    mse = mean_squared_error(y_test, y_pred)
    if verbose: 
        print(f"Mean Squared Error on test set: {mse:.4f}")
        if hasattr (model, 'feature_importances_'): 
            print("Feature importances:\n", model.feature_importances_)
        if hasattr (model, 'feature_names_in_'): 
            print("Feature names:\n", model.feature_names_in_)

    # Apply the trained model to the entire dataset to generate the proxy feature
    proxy_feature = pipeline.predict(X)

    # Convert the proxy feature to a DataFrame or Series
    proxy_series = pd.Series(proxy_feature, index=data.index, name=proxy_name)

    # If infer_data is True, add the proxy feature to the original dataframe
    if infer_data:
        data[proxy_name] = proxy_series
        return data
    else:
        return proxy_series
        

def _manage_target(data, target):
    """
    Helper function to manage the target argument, ensuring that it's in a
    consistent format (either as a column name or as an array-like object).
    
    Parameters:
    ----------
    data : pd.DataFrame
        The input dataset containing features.

    target : str, pd.Series, np.ndarray, or pd.DataFrame
        The target column name, a pandas Series, or a numpy array containing
        the target values. If a DataFrame with a single column is passed, 
        it will be treated as a Series.

    Returns:
    -------
    tuple
        The processed dataframe and the target column name.
    """
    
    # If target is a DataFrame with a single column, squeeze it into a Series
    if isinstance(target, pd.DataFrame):
        if target.shape[1] == 1:  # Ensure it's a single column DataFrame
            target = target.squeeze()
            target_column = target.name if target.name else 'target_column'
        else:
            raise ValueError("DataFrame target must have exactly one column.")

    # If target is a string (assumed to be the name of a column)
    elif isinstance(target, str):
        # Ensure the column exists in the dataframe
        exist_features(data, features=target, error='raise')
        target_column = target

    # If target is array-like (Series or ndarray)
    elif isinstance(target, (np.ndarray, pd.Series)):
        if isinstance(target, pd.Series):
            target_column = target.name or 'target_column'
        elif isinstance(target, np.ndarray):
            target_column = 'target_column'

        # Ensure the target is one-dimensional (flatten if necessary)
        if target.ndim > 1:
            target = target.ravel()

        # Check that the target and data have the same length
        check_consistent_length(data, target)
        # Ensure that the target is aligned with the dataframe
        data[target_column] = target

    else:
        raise ValueError(
            "Target must be a column name (str), a Series, or an ndarray.")

    return data, target_column
   
@is_data_readable 
@Dataify(auto_columns=True, prefix='feature_')
@validate_params ({ 
    "data": ['array-like'], 
    "num_categories": [Interval(Real, 1, None, closed="left")], 
    "concentration_params": ['array-like', None], 
    "proxy_name": [str, None], 
    "infer_data": [bool], 
    "random_state": ['random_state', None], 
    })
def generate_dirichlet_features(
    data, 
    num_categories=3, 
    concentration_params=None, 
    proxy_name=None, 
    infer_data=False, 
    random_state=None
    ):
    """
    Generate synthetic features using the Dirichlet distribution and 
    add them to the given dataset.

    The function generates synthetic features using the Dirichlet 
    distribution. The Dirichlet distribution is often used to model 
    categorical data or proportions, and the function produces synthetic 
    features representing the proportions of `num_categories` categories. 
    The concentration parameters control how the probability mass is 
    allocated across the categories, with larger values leading to more 
    concentrated or skewed distributions, and smaller values leading to 
    more uniform distributions.

    Parameters
    ------------
    data : pd.DataFrame
        The input dataset to which the synthetic feature will be added.
        
        - `data` is a pandas DataFrame containing the existing dataset 
          to which synthetic features will be appended.
        - The index of `data` will be retained in the generated features.

    num_categories : int, default=3
        The number of categories (or features) to generate for the Dirichlet 
        distribution.
        
        - `num_categories` specifies how many synthetic features (categories) 
          should be generated using the Dirichlet distribution. The default 
          value is 3.

    concentration_params : list of float, default=None
        A list of positive values that determine the concentration of the 
        Dirichlet distribution. If None, defaults to an equal concentration 
        for each category (uniform distribution).
        
        - `concentration_params` specifies the concentration parameters for 
          each category. Larger values make the distribution more concentrated, 
          while smaller values yield more uniform distributions. 
        - If set to None, the function will use a uniform concentration where 
          each category has an equal weight.
        
    proxy_name : str, default="Feature"
        The name of the generated synthetic features.
        
        - `proxy_name` is a string prefix used to name the generated features. 
          By default, the synthetic features will be named `Feature_1`, 
          `Feature_2`, ..., up to `Feature_n`. You can change this prefix 
          with the `proxy_name` parameter.

    infer_data : bool, default=False
        If True, the synthetic features are added to the original dataframe. 
        If False, only the generated features are returned.
        
        - When `infer_data` is set to `True`, the original `data` DataFrame 
          will be updated to include the synthetic features. Otherwise, 
          only the generated synthetic features are returned.

    random_state : int, default=42
        The random seed for reproducibility.
        
        - `random_state` controls the random seed used for generating random 
          numbers. This ensures reproducibility when running the function 
          multiple times with the same input data.

    Returns
    ---------
    pd.DataFrame or pd.Series
        - If `infer_data=True`, the original dataframe with the new features 
          added. 
        - If `infer_data=False`, only the synthetic features are returned 
          as a new DataFrame.

    Notes
    ------
    The Dirichlet distribution is a multivariate distribution often used to 
    model categorical data where the sum of the random variables equals 1. 
    The synthetic features are generated using the Dirichlet distribution:

    .. math::
        \mathbf{x} = \text{Dirichlet}(\boldsymbol{\alpha})

    Where:
    - \( \mathbf{x} \) represents the generated synthetic feature vector.
    - \( \boldsymbol{\alpha} \) is the vector of concentration parameters, 
      specifying how the probability mass is distributed across categories.
    - The resulting feature vector is normalized such that the sum of the 
      elements is 1.


    The Dirichlet distribution is used in a variety of fields to generate 
    probabilities or proportions that sum to 1, such as modeling topic 
    distributions in text or proportions of resources in economics. In this 
    function, it is used to generate synthetic features that can be treated as 
    proportions across different categories. The concentration parameters 
    control how strongly the probability mass is concentrated on particular 
    categories. For instance, if the concentration parameters are large, 
    the synthetic features will tend to have values close to a single 
    category, while if they are small, the values will be more uniformly 
    distributed across categories.

    Example:
    --------
    >>> from gofast.utils.ml.preprocessing import generate_dirichlet_features
    >>> import pandas as pd
    
    # Create a sample dataframe
    >>> data = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
    
    # Generate synthetic Dirichlet features and add them to the dataframe
    >>> result = generate_dirichlet_features(
    >>>     data, num_categories=3, concentration_params=[1, 2, 3], 
    >>>     proxy_name="Synthetic", infer_data=True
    >>> )
    
    >>> print(result)
       col1  col2  Synthetic_1  Synthetic_2  Synthetic_3
    0     1     4     0.259634     0.379842     0.360524
    1     2     5     0.123573     0.289129     0.587298
    2     3     6     0.355040     0.279367     0.365593

    See Also:
    ---------
    numpy.random.dirichlet : The function used to generate Dirichlet-
          distributed random variables.
    pandas.DataFrame.join : The method used to join the generated features 
          with the original data.
    gofast.utils.ml.preprocessing.generate_proxy_feature: Generate a proxy feature 
         based on the available features. 
         
    References
    -----------
    [1]_ Kingma, D.P., & Welling, M. (2013). Auto-Encoding Variational 
    Bayes. ICLR.
    """
    from scipy.stats import dirichlet

    # Set random seed for reproducibility
    np.random.seed(random_state)
    proxy_name = proxy_name or "Feature" 
    # If no concentration parameters are provided, default to uniform concentration
    if concentration_params is None:
        concentration_params = [1.0] * num_categories  # Equal concentration for each category
    
    concentration_params = [
        validate_numeric(v, allow_negative=False, ) 
        for v in concentration_params
    ]
    # Ensure the concentration parameters are positive values
    if any(param <= 0 for param in concentration_params):
        raise ValueError("Concentration parameters must be positive.")
    
    # Generate synthetic features using the Dirichlet distribution
    synthetic_features = dirichlet.rvs(concentration_params, size=len(data))
    
    # Create a DataFrame from the generated Dirichlet samples
    synthetic_features_df = pd.DataFrame(
        synthetic_features, 
        columns=[f"{proxy_name}_{i+1}" for i in range(num_categories)], 
        index=data.index
    )
    
    # If infer_data is True, add the generated features to the original dataframe
    if infer_data:
        data = data.join(synthetic_features_df)
        return data
    else:
        return synthetic_features_df
