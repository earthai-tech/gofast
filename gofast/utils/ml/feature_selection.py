# -*- coding: utf-8 -*-
"""
Utilities for features selections, and inspections.
"""
import copy
import warnings
from numbers import Integral, Real 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
 
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.model_selection import cross_val_score

from ..._gofastlog import gofastlog
from ...api.docstring import DocstringComponents, _core_docs 
from ...api.types import ( 
    Any, List, Dict, 
    Tuple, Union, 
    Optional, DataFrame, 
    ArrayLike, Series
)
from ...api.summary import ReportFactory, ResultSummary 
from ...compat.sklearn import ( 
    validate_params, 
    Interval, 
    StrOptions, 
    HasMethods 
)
from ...core.array_manager import to_numeric_dtypes, is_array_like
from ...core.checks import ( 
    is_in_if, 
    is_iterable, 
    exist_features, 
    is_numeric_dtype
)
from ...core.handlers import columns_manager 
from ...core.io import is_data_readable
from ...core.utils import type_of_target 
from ...decorators import Dataify
from ..base_utils import select_features, extract_target
from ..deps_utils import ensure_pkg
from ..generic_utils import vlog  
from ..io_utils import to_txt
from ..validator import ( 
    build_data_if, 
    validate_data_types, 
    check_consistent_length, 
    get_estimator_name, 
    filter_valid_kwargs
)

# Logger Configuration
_logger = gofastlog().get_gofast_logger(__name__)
# Parametrize the documentation 
_param_docs = DocstringComponents.from_nested_components(
    core=_core_docs["params"],
)

__all__= [ 
    'bi_selector',
    'get_correlated_features',
    'select_feature_importances', 
    'get_feature_contributions',
    'display_feature_contributions', 
    'select_relevant_features', 
    'validate_feature_utility'
    ]

@Dataify(auto_columns= True, prefix ='feat_') 
def validate_feature_utility(
    df: DataFrame,
    target: Union[ArrayLike, Series, str],
    feature_set_a: list,
    feature_set_b: list,
    task: str = 'auto',
    n_folds: int = 5,
    view: bool = False,
    save_result: bool=False, 
    filename:Optional[str]=None, 
    box_kws=None,
    figsize=None,
    verbose: int = 1
):
    r"""
    Compare two feature sets by cross-validating a predictive
    model for each set and computing the performance improvement
    of one set over the other.

    This function uses `exist_features`, `columns_manager`,
    `extract_target`, `check_consistent_length`,
    `type_of_target`, `ResultSummary`, and scikit-learn's
    `cross_val_score`. It trains either a random forest
    classifier or regressor on both feature sets, evaluates
    each set via cross-validation, and estimates the mean
    performance difference.

    .. math::
       R_{improvement} = \\mu_B - \\mu_A

    where :math:`\\mu_B` is the mean performance (accuracy or
    MAE) of ``feature_set_b`` and :math:`\\mu_A` is the mean
    performance of ``feature_set_a``.

    Parameters
    ----------
    df : DataFrame
        The input pandas DataFrame containing all the features
        and possibly the target column.
    target : Union[ArrayLike, Series, str]
        The target used for model training. If `target` is a
        string, it is assumed to be a column name in `df`.
    feature_set_a : list
        List of feature column names forming the first set.
    feature_set_b : list
        List of feature column names forming the second set.
    task : str, default='auto'
        The task type for modeling (e.g. `binary`,
        `multiclass`, or `regression`). If ``'auto'`` is
        provided, the function detects the task automatically
        by analyzing `target`.
    n_folds : int, default=5
        Number of folds to use in cross-validation.
    view : bool, default=False
        If True, draws boxplots showing the distribution of
        cross-validation scores for each feature set.
    save_result: bool, False 
        Export the result into a txt file. 
    filename: str, optional 
       Name of the file to export. 
    box_kws : dict, optional
        Dictionary of keyword arguments passed to the
        seaborn boxplot function.
    figsize : tuple, optional
        Figure size for the boxplots, e.g. (width, height).
    verbose : int, default=1
        Verbosity level (0 to 5). Controls how much internal
        information is printed to console.

    Returns
    -------
    dict
        A dictionary containing the performance metrics for
        both feature sets and the improvement.

    Notes
    -----
    This approach can be used to determine whether additional
    features or an alternative feature representation yields a
    better predictive performance.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from gofast.utils.ml.feature_selection import \
    ...     validate_feature_utility
    >>> # Generate a random dataset with columns
    >>> # 'feat1','feat2','feat3','target'.
    >>> np.random.seed(42)
    >>> df = pd.DataFrame({
    ...     'feat1': np.random.rand(10),
    ...     'feat2': np.random.rand(10),
    ...     'feat3': np.random.rand(10),
    ...     'target': np.random.randint(0, 2, 10)
    ... })
    
    >>> # Example 1: Minimal usage with defaults
    >>> result_1 = validate_feature_utility(
    ...     df=df,
    ...     target='target',
    ...     feature_set_a=['feat1', 'feat2'],
    ...     feature_set_b=['feat3']
    ... )
    >>> print(result_1)
    
    >>> # Example 2: Classification task with higher verbosity
    >>> result_2 = validate_feature_utility(
    ...     df=df,
    ...     target='target',
    ...     feature_set_a=['feat1'],
    ...     feature_set_b=['feat1', 'feat2', 'feat3'],
    ...     task='binary',
    ...     n_folds=5,
    ...     verbose=5
    ... )
    >>> print(result_2)
    
    >>> # Example 3: Regression task, fewer folds, display boxplots
    >>> result_3 = validate_feature_utility(
    ...     df=df,
    ...     target='target',
    ...     feature_set_a=['feat1', 'feat2'],
    ...     feature_set_b=['feat1', 'feat2', 'feat3'],
    ...     task='regression',
    ...     n_folds=3,
    ...     view=True,
    ...     verbose=2
    ... )
    >>> print(result_3)

    See Also
    --------
    exist_features : Checks if requested features exist in a
        DataFrame.
    columns_manager : Cleans or preprocesses columns in a list.
    extract_target : Extracts the target from a DataFrame.
    check_consistent_length : Ensures consistency among input
        lengths.
    type_of_target : Detects whether the task is classification
        or regression.
    cross_val_score : Performs cross-validation score.
    ResultSummary : Stores the evaluation results in a summary
        object.

    References
    ----------
    .. [1] Pedregosa et al. *Scikit-learn: Machine Learning in
       Python.*, JMLR 12, pp. 2825-2830, 2011.
    """

    # Check if each feature in feature_set_a and feature_set_b
    # exists in the provided DataFrame
    vlog("[INFO] Checking if features exist in DataFrame.", 
         verbose
         )
    exist_features(
        df,
        features = feature_set_a,
        name= "Feature set A",
    )
    exist_features(
        df,
        features = feature_set_b,
        name= "Feature set B",
    )

    # Clean and prepare feature sets
    if verbose >= 3:
        vlog("[INFO] Cleaning columns for feature sets.", 
             verbose)
    
    # Create feature matrices for each set
    # and extract the target vector y
    feature_set_a = columns_manager(feature_set_a)
    feature_set_b = columns_manager(feature_set_b)

    # Create feature matrices
    X_a = df[feature_set_a]
    X_b = df[feature_set_b]

    # Extract target vector y
    if isinstance(target, str):
        vlog(f"[INFO] Extracting target as column: {target}", 
             verbose)
        y = extract_target(
            df,
            target_names = target,
            drop= False,
        )
    else:
     
        vlog(
            "  [TRACE] Using provided target array-like.", 
            verbose
        )
        y = target

    # Ensure consistent length
    check_consistent_length(df, y)

    # Automatically detect task type (classification vs. regression)
    # if user set task='auto'
    if task == 'auto':
        vlog(" [DEBUG] Auto-detecting task type.", verbose)
        task = type_of_target(y)

    # Initialize model and scoring
    if task in ['classification', 'multiclass', 'binary']:
        vlog("[INFO] Task identified as classification.", verbose)
        model= RandomForestClassifier(random_state = 42)
        scoring = 'accuracy'
    else:
        vlog("[INFO] Task identified as regression.", verbose)
        model = RandomForestRegressor(random_state = 42)
        scoring = 'neg_mean_absolute_error'

    # Cross-validate feature set A
    vlog("[INFO] Cross-validating Feature Set A.",
         verbose
    )
    scores_a = cross_val_score(
        estimator = model,
        X = X_a,
        y = y,
        cv = n_folds,
        scoring = scoring,
    )

    # Cross-validate feature set B
    vlog("[INFO] Cross-validating Feature Set B.",
         verbose)
    scores_b = cross_val_score(
        estimator = model,
        X = X_b,
        y = y,
        cv = n_folds,
        scoring = scoring,
    )

    # Convert scores to positive if regression
    if task == 'regression':
        vlog("[INFO] Converting negative MAE to positive.",
            verbose
        )
        scores_a = -scores_a
        scores_b = -scores_b

    # Compile results
    mean_a = np.mean(scores_a)
    mean_b = np.mean(scores_b)
    std_a  = np.std(scores_a)
    std_b  = np.std(scores_b)

    result = {
        'feature_set_a': {
            'features': feature_set_a,
            f'mean_{scoring}': mean_a,
            'std_dev': std_a,
        },
        'feature_set_b': {
            'features': feature_set_b,
            f'mean_{scoring}': mean_b,
            'std_dev': std_b,
        },
        'improvement': mean_b - mean_a,
    }

    # Add results to custom summary object
    summary = ResultSummary(
        name="ABTest"
    ).add_results(result)

    # Print summary if verbosity >= 1
    if verbose >= 1:
        print("[INFO] Summary of results:\n", summary)

    # Draw boxplots if requested
    if view:
        _plot_box ( 
            box_kws=box_kws, 
            figsize=figsize, 
            scores_a=scores_a, 
            scores_b=scores_b, 
            scoring=scoring, 
            task=task, 
            verbose=verbose  
        ) 
        
    if save_result: 
        to_txt(result, filename) 
        
    # Return dictionary with comparison metrics
    return result

def _plot_box ( 
    box_kws, 
    figsize, 
    scores_a, 
    scores_b,
    scoring, 
    task, 
    verbose  
    ): 
    from ...plot._d_cms import update_box_kws 
    
    vlog("  [TRACE] Displaying performance boxplots.", verbose)
    d_box_kws = update_box_kws(
        "medianprops__color_0.8", 'whiskerprops__color_#137', 
        )
    # d_box_kws = box_kws or dict(
    #     width     = 15,
    #     whis      = 2,
    #     color     = ".8",
    #     linecolor = "#137",
    # )
    box_kws = box_kws or d_box_kws 
    
    plt.figure(figsize = figsize or (10, 5))

    box_kws= filter_valid_kwargs ( sns.boxplot, box_kws)
    sns.boxplot(
        data=[scores_a, scores_b],
        palette='Set2',
        **box_kws
    )
    plt.xticks(
        ticks=[0, 1],
        labels=['Feature Set A', 'Feature Set B']
    )
    plt.ylabel(
        scoring.upper()
        if task == 'classification'
        else 'MAE'
    )
    plt.title(
        'A/B Test: Model Performance Comparison'
    )
    plt.show()
    
    
@is_data_readable 
@validate_params({
    'data': ['array-like'], 
    'target': ['array-like', str, None], 
    'threshold': [Interval(Real, 0, 1 , closed ='both')], 
    'method': [StrOptions({'pearson','spearman', 'kendall'})], 
    'top_n': [Integral, None]
    })
def select_relevant_features(
    data: DataFrame,
    target: Optional[Union[Any, str]] = None,
    threshold: float = 0.1,
    remove_target: bool = True,
    method: str = 'pearson',
    top_n: Optional[int] = None,
    ascending: bool = False,
    return_frame: bool = False,
    view: bool = False,
    show_grid: bool=True, 
    verbose: int = 1
) -> Union[List[str], DataFrame]:
    """
    Select features with correlation to the target variable above a 
    specified threshold.
    
    The ``select_relevant_features`` function identifies and selects 
    features from a dataset that exhibit a moderate or strong correlation 
    with a specified target variable. If the target is not provided, it 
    selects features based on their variance. This is useful for feature 
    selection in predictive modeling and exploratory data analysis.
    
    .. math::
        \text{Relevant Features} = \left\{ 
            x_i \in X \mid | \rho(x_i, y) | > \text{threshold} 
        \right\}
    
    Parameters
    ----------
    data : pd.DataFrame
        The dataset containing features and the target variable.
    
    target : Optional[Union[Any, str]], default=None
        The target variable for which correlations are computed. It can be 
        specified as either the name of the target column (``str``) or 
        as an array-like object representing the target values. If ``None``, 
        features are selected based on their variance.
    
    threshold : float, optional, default=0.1
        The minimum absolute correlation required for a feature to be 
        considered relevant when ``target`` is specified, or the minimum 
        variance required when ``target`` is ``None``. Must be between 
        0 and 1.
    
    remove_target : bool, optional, default=True
        Whether to exclude the target variable from the list of relevant 
        features when ``target`` is specified.
    
    method : str, optional, default='pearson'
        The method to compute correlation. Options are:
        
        - ``'pearson'``: Standard Pearson correlation coefficient.
        - ``'spearman'``: Spearman rank-order correlation.
        - ``'kendall'``: Kendall Tau correlation.
        
        Applicable only when ``target`` is specified.
    
    top_n : Optional[int], optional, default=None
        If specified, selects the top_n features with the highest absolute 
        correlation with the target variable or the highest variance when 
        ``target`` is ``None``. If ``None``, all features meeting the 
        threshold are selected.
    
    ascending : bool, optional, default=False
        If ``True``, sorts features in ascending order of correlation or 
        variance; otherwise, in descending order. Relevant only when 
        ``top_n`` is specified.
    
    view : bool, optional, default=False
        If ``True``, visualizes the correlations or variances of the relevant 
        features using a horizontal bar plot.
    
    return_frame : bool, optional, default=False
        If ``True``, returns a pandas DataFrame containing only the 
        relevant features instead of a list of feature names.
    
    verbose : int, optional, default=1
        Controls the verbosity level of informational messages:
        
        - ``0``: No messages.
        - ``1``: Basic messages about the selection process.
        - ``2``: Detailed messages including intermediate steps.
        - ``3``: Debug-level messages with extensive details.
    
    Returns
    -------
    Union[List[str], pd.DataFrame]
        - If ``return_frame=False``: Returns a list of relevant feature 
          names that have a correlation with the target above the specified 
          threshold or variance above the threshold when ``target`` is 
          ``None``.
        - If ``return_frame=True``: Returns a pandas DataFrame containing 
          only the relevant features.
    
    Raises
    ------
    ValueError
        If the target variable is not present in the DataFrame when ``target`` 
        is specified.
        If the threshold is not between 0 and 1.
        If the specified correlation method is unsupported.
        If the target array-like object length does not match the 
        number of rows in the DataFrame.
    
    Examples
    --------
    >>> from gofast.utils.ml.feature_selection import select_relevant_features
    >>> import pandas as pd
    >>> 
    >>> # Sample data with target as a column name
    >>> data = pd.DataFrame({
    ...     'toc': [1, 2, 3, 4, 5],
    ...     'feature1': [2, 4, 6, 8, 10],
    ...     'feature2': [5, 3, 6, 2, 7],
    ...     'feature3': [1, 2, 1, 2, 1]
    ... })
    >>> 
    >>> # Select features with correlation above 0.5
    >>> relevant = select_relevant_features(
    ...     data, 
    ...     target='toc', 
    ...     threshold=0.5, 
    ...     view=True, 
    ...     verbose=2
    ... )
    >>> print(relevant)
    ['feature1', 'feature2']
    
    >>> # Sample data with target as an array-like object
    >>> target_array = [1, 2, 3, 4, 5]
    >>> data = pd.DataFrame({
    ...     'feature1': [2, 4, 6, 8, 10],
    ...     'feature2': [5, 3, 6, 2, 7],
    ...     'feature3': [1, 2, 1, 2, 1]
    ... })
    >>> 
    >>> # Select features based on target array correlation above 0.5
    >>> relevant = select_relevant_features(
    ...     data, 
    ...     target=target_array, 
    ...     threshold=0.5, 
    ...     view=True, 
    ...     verbose=3
    ... )
    >>> print(relevant)
    ['feature1', 'feature2']
    
    >>> # Select top 1 feature with highest correlation
    >>> top_feature = select_relevant_features(
    ...     data, 
    ...     target='toc', 
    ...     top_n=1, 
    ...     verbose=1
    ... )
    >>> print(top_feature)
    ['feature1']
    
    >>> # Return a DataFrame of relevant features
    >>> relevant_df = select_relevant_features(
    ...     data, 
    ...     target='toc', 
    ...     threshold=0.5, 
    ...     return_frame=True
    ... )
    >>> print(relevant_df)
       feature1  feature2
    0         2         5
    1         4         3
    2         6         6
    3         8         2
    4        10         7
    
    >>> # Select features based on variance when target is not provided
    >>> relevant_variance = select_relevant_features(
    ...     data, 
    ...     threshold=1.0, 
    ...     view=True, 
    ...     verbose=2
    ... )
    >>> print(relevant_variance)
    ['feature1', 'feature2']
    
    Notes
    -----
    - When ``target`` is specified, the function computes the correlation matrix 
      for the provided DataFrame and identifies features that have an absolute 
      correlation with the target variable exceeding the specified threshold.
    - When ``target`` is an array-like object, it should have the same length 
      as the number of rows in ``data``.
    - When ``target`` is ``None``, the function selects features based on their 
      variance exceeding the specified threshold.
    - When ``top_n`` is specified, the function selects the top_n features based 
      on the absolute correlation values or variance.
    - The ``view`` parameter enables visualization of the relevant feature 
      correlations or variances using matplotlib and seaborn.
    - Verbosity controlled by the ``verbose`` parameter provides feedback 
      during execution, aiding in debugging and process tracking.
    
    See Also
    --------
    pandas.DataFrame.corr : Compute pairwise correlation of columns.
    seaborn.barplot : Show point estimates and confidence intervals as 
                      rectangular bars.
    
    References
    ----------
    .. [1] McKinney, W. (2010). Data Structures for Statistical Computing in 
           Python. In *Proceedings of the 9th Python in Science Conference*, 
           51-56.
    .. [2] Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). 
           Array programming with NumPy. *Nature*, 585(7825), 357-362.
    .. [3] Pandas Development Team. (2023). *pandas documentation*. 
           https://pandas.pydata.org/pandas-docs/stable/
    """
    # Validate input DataFrame and build if necessary
    data = build_data_if(
        data, 
        to_frame=True, 
        input_name='col_', 
        raise_exception=True
    )
    
    # Validate target variable
    if target is not None:
        tmsg = "The 'target' parameter must be a numeric array-like object."
        if isinstance(target, str):
            exist_features(data, target, name='Target')
            if not is_numeric_dtype(data[target], to_array=True): 
                raise TypeError(tmsg)
            target_correlations = data.corr(method=method)[target]
            if remove_target:
                target_correlations = target_correlations.drop(labels=target)
        elif is_array_like(target):
            check_consistent_length(data, target)
            
            if not is_numeric_dtype(target, to_array=True): 
                raise TypeError(tmsg)
            temp_target_column = 'temp_target_column'
            data[temp_target_column] = target
            correlation_matrix = data.corr(method=method)
            target_correlations = correlation_matrix[
                temp_target_column].drop(labels=temp_target_column)
        else:
            raise ValueError(
                "The 'target' parameter must be a string or array-like object.")
    else:
        # Strategy when target is not provided: select based on variance
        if verbose >= 1:
            print("No target provided. Selecting features based on variance.")
        variances = data.var()
        target_correlations = variances  # Using variance as the metric
        # Remove target if remove_target is True and target is a column
        if remove_target and 'target' in data.columns:
            target_correlations = variances.drop(labels='target')
    
    # Select features based on threshold
    if target is not None:
        relevant_features = target_correlations[
            abs(target_correlations) > threshold
        ].index.tolist()
    else:
        relevant_features = target_correlations[
            target_correlations > threshold
        ].index.tolist()
    
    # If top_n is specified, select top_n features
    if top_n is not None:
        if top_n > len(relevant_features):
            warnings.warn(
                f"Requested top_n={top_n} exceeds the number of available features="
                f"{len(relevant_features)}. Returning all relevant features.",
                UserWarning
            )
            top_n = len(relevant_features)
        if target is not None:
            relevant_features = target_correlations.abs().sort_values(
                ascending=ascending, 
                kind='quicksort'
            ).head(top_n).index.tolist()
        else:
            relevant_features = target_correlations.sort_values(
                ascending=ascending, 
                kind='quicksort'
            ).head(top_n).index.tolist()
    
    # Remove temporary target column if added
    if target is not None and not isinstance(
            target, str) and 'temp_target_column' in data.columns:
        data.drop(columns=['temp_target_column'], inplace=True)
    
    # Visualization
    if view:
        if verbose >= 1:
            if not relevant_features:
                warnings.warn(
                    "No features meet the threshold for visualization.",
                    UserWarning)
            else:
                print(f"Visualizing {'correlations' if target is not None else 'variances'}"
                      f" of {len(relevant_features)} relevant feature(s).")
        
        if relevant_features:
            plt.figure(figsize=(10, 6))
            if target is not None:
                sns.barplot(
                    x=abs(target_correlations[relevant_features]), 
                    y=relevant_features, 
                    orient='h'
                )
                plt.xlabel(f'Absolute Correlation with {target}')
                plt.title(f'Relevant Features Correlated with {target}')
            else:
                sns.barplot(
                    x=target_correlations[relevant_features], 
                    y=relevant_features, 
                    orient='h'
                )
                plt.xlabel('Variance')
                plt.title('Relevant Features Based on Variance')
                
            plt.grid (show_grid)
            plt.tight_layout()
            plt.show()
    
    # Verbosity
    if verbose >= 1:
        if target is not None:
            print(f"Selected {len(relevant_features)} relevant feature(s)"
                  f" based on correlation threshold={threshold}.")
        else:
            print(f"Selected {len(relevant_features)} relevant feature(s)"
                  f" based on variance threshold={threshold}.")
        if top_n:
            print(f"Top {top_n} feature(s) selected.")
    
    # Return as DataFrame if requested
    if return_frame:
        if not relevant_features:
            if verbose >= 1:
                print("No relevant features to return as DataFrame.")
            return pd.DataFrame()
        return data[relevant_features]
    
    return relevant_features


@is_data_readable
@Dataify(auto_columns=True)
def bi_selector(
    data,  
    features=None, 
    return_frames=False,
    parse_features: bool=False 
):
    """
    Automatically differentiates numerical and categorical attributes 
    in a dataset.

    This function is useful for efficiently selecting categorical features from 
    numerical features, and vice versa, when dealing with a large number 
    of features. Manually selecting features can be tedious and prone to errors, 
    especially in large datasets.

    Parameters
    ----------
    data : pandas.DataFrame
        The input DataFrame containing the data.
    
    features : list of str, optional
        A list of feature names (column names) to be considered for selection. 
        If any feature does not exist in the DataFrame, an error is raised. 
        If `features` is `None`, the function will return the categorical and 
        numerical features from the entire DataFrame.
    
    return_frames : bool, default=False
        If `True`, the function will return two DataFrames: one containing the 
        specified features and the other containing the remaining features. 
        If `False`, it returns the features as a list.
    
    parse_features : bool, default=False
        If `True`, the function will parse and construct a list of features 
        from a string input, using regular expressions to handle special 
        characters like  commas (`,`) and at symbols (`@`).
    
    Returns
    -------
    tuple
        - If `return_frames=False`, returns a tuple of two lists:
          - A list of the selected features.
          - A list of the remaining features in the DataFrame.
        - If `return_frames=True`, returns a tuple of two DataFrames:
          - A DataFrame containing the selected features.
          - A DataFrame containing the remaining features.
    
    Example
    -------
    >>> from gofast.utils.mlutils import bi_selector 
    >>> from gofast.datasets import load_hlogs 
    >>> data = load_hlogs().frame # get the frame 
    >>> data.columns 
    >>> Index(['hole_id', 'depth_top', 'depth_bottom', 'strata_name', 'rock_name',
           'layer_thickness', 'resistivity', 'gamma_gamma', 'natural_gamma', 'sp',
           'short_distance_gamma', 'well_diameter', 'aquifer_group',
           'pumping_level', 'aquifer_thickness', 'hole_depth_before_pumping',
           'hole_depth_after_pumping', 'hole_depth_loss', 'depth_starting_pumping',
           'pumping_depth_at_the_end', 'pumping_depth', 'section_aperture', 'k',
           'kp', 'r', 'rp', 'remark'],
          dtype='object')
    >>> num_features, cat_features = bi_selector (data)
    >>> num_features
    ...['gamma_gamma',
         'depth_top',
         'aquifer_thickness',
         'pumping_depth_at_the_end',
         'section_aperture',
         'remark',
         'depth_starting_pumping',
         'hole_depth_before_pumping',
         'rp',
         'hole_depth_after_pumping',
         'hole_depth_loss',
         'depth_bottom',
         'sp',
         'pumping_depth',
         'kp',
         'resistivity',
         'short_distance_gamma',
         'r',
         'natural_gamma',
         'layer_thickness',
         'k',
         'well_diameter']
    >>> cat_features 
    ... ['hole_id', 'strata_name', 'rock_name', 'aquifer_group', 
         'pumping_level']
    """
    if features is None: 
        data, diff_features, features = to_numeric_dtypes(
            data,  return_feature_types= True ) 
    if features is not None: 
        features = is_iterable(features, exclude_string= True, transform =True, 
                               parse_string=parse_features )
        diff_features = is_in_if( data.columns, items =features, return_diff= True )
        if diff_features is None: diff_features =[]
    return  ( diff_features, features ) if not return_frames else  (
        data [diff_features] , data [features ] ) 

@is_data_readable
def get_correlated_features(
    data:DataFrame ,
    corr:str ='pearson', 
    threshold: float=.80 , 
    fmt: bool= False 
    )-> DataFrame: 
    """Find the correlated features/columns in the dataframe. 
    
    Indeed, highly correlated columns don't add value and can throw off 
    features importance and interpretation of regression coefficients. If we  
    had correlated columns, choose to remove either the columns from  
    level_0 or level_1 from the features data is a good choice. 
    
    Parameters 
    -----------
    data: Dataframe or shape (M, N) from :class:`pandas.DataFrame` 
        Dataframe containing samples M  and features N
    corr: str, ['pearson'|'spearman'|'covariance']
        Method of correlation to perform. Note that the 'person' and 
        'covariance' don't support string value. If such kind of data 
        is given, turn the `corr` to `spearman`. *default* is ``pearson``
        
    threshold: int, default is ``0.95``
        the value from which can be considered as a correlated data. Should not 
        be greater than 1. 
        
    fmt: bool, default {``False``}
        format the correlated dataframe values 
        
    Returns 
    ---------
    df: `pandas.DataFrame`
        Dataframe with cilumns equals to [level_0, level_1, pearson]
        
    Examples
    --------
    >>> from gofast.utils.mlutils import get_correlated_features 
    >>> df_corr = get_correlated_features (data , corr='spearman',
                                     fmt=None, threshold=.95
                                     )
    """
    data = build_data_if(data, to_frame=True, raise_exception= True, 
                         input_name="col")
    
    th= copy.deepcopy(threshold) 
    threshold = str(threshold)  
    try : 
        threshold = float(threshold.replace('%', '')
                          )/1e2  if '%' in threshold else float(threshold)
    except: 
        raise TypeError (
            f"Threshold should be a float value, got: {type(th).__name__!r}")
          
    if threshold >= 1 or threshold <= 0 : 
        raise ValueError (
            f"threshold must be ranged between 0 and 1, got {th!r}")
      
    if corr not in ('pearson', 'covariance', 'spearman'): 
        raise ValueError (
            f"Expect ['pearson'|'spearman'|'covariance'], got{corr!r} ")
        
    # collect numerical values and exclude cat values
    df = select_features(data, features =None, dtypes_inc ='number')
        
    # use pipe to chain different func applied to df 
    c_df = ( 
        df.corr()
        .pipe(
            lambda df1: pd.DataFrame(
                np.tril (df1, k=-1 ), # low triangle zeroed 
                columns = df.columns, 
                index =df.columns, 
                )
            )
            .stack ()
            .rename(corr)
            .pipe(
                lambda s: s[
                    s.abs()> threshold 
                    ].reset_index()
                )
                .query("level_0 not in level_1")
        )

    return  c_df.style.format({corr :"{:2.f}"}) if fmt else c_df 
          
@validate_params({ 
    'estimator': [HasMethods(['fit', 'predict'])], 
    'X': ['array-like'], 
    'y': ['array-like', None],  
    'threshold': [Interval(Real, 0, 1, closed ='both'),
                  StrOptions({"mean","median" })], 
    })
def select_feature_importances(
    estimator: BaseEstimator, 
    X: Union[pd.DataFrame, np.ndarray], 
    y: Optional[Union[pd.Series, np.ndarray]] = None, 
    *,
    threshold: Union[float, str] = 0.1, 
    prefit: bool = True, 
    verbose: int = 0, 
    return_selector: bool = False, 
    view: bool=False, 
    **kwargs
) -> Union[List[str], pd.DataFrame, SelectFromModel, VarianceThreshold]:
    """
    Select features based on importance thresholds after model fitting or variance.

    The ``select_feature_importances`` function selects relevant features from 
    a dataset based on feature importances derived from a supervised estimator 
    or based on feature variance for unsupervised scenarios. This is useful for 
    feature selection in both predictive modeling and exploratory data analysis.

    .. math::
        \text{Relevant Features} = 
        \begin{cases}
            \{ x_i \in X \mid | \rho(x_i, y) | > \text{threshold} \}\\
                & \text{if } y \text{ is provided} \\
            \{ x_i \in X \mid \text{Var}(x_i) > \text{threshold} \}\\
                & \text{if } y \text{ is None}
        \end{cases}

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator from which the feature importances are derived. Must have
        either `feature_importances_` or `coef_` attributes after fitting, unless
        `importance_getter` is specified in `kwargs`.
        
    X : {array-like, sparse matrix, pd.DataFrame} of shape (n_samples, n_features)
        The input samples.
        
    y : Optional[array-like] of shape (n_samples,), default=None
        The target values (class labels) as integers or strings. If ``None``,
        the function performs unsupervised feature selection based on feature variance.
        
    threshold : float or str, optional, default=0.1
        The threshold value to use for feature selection. 
        - If ``y`` is provided, features with importance greater than or equal 
          to this value are retained. Can also be a string such as `"mean"` or 
          `"median"` to use the mean or median feature importance as the threshold.
        - If ``y`` is ``None``, features with variance greater than this value are retained.
        
    prefit : bool, default=True
        Whether the estimator is expected to be prefit. If ``True``, 
        ``estimator`` should already be fitted; otherwise, it will be fitted 
        on ``X`` and ``y`` (if ``y`` is provided).
        
    verbose : int, default=0
        Controls the verbosity: the higher, the more messages.
        
    return_selector : bool, default=False
        Whether to return the selector object instead of the transformed data.
        
    **kwargs : additional keyword arguments
        Additional arguments passed to `SelectFromModel` or `VarianceThreshold`.
    
    Returns
    -------
    Union[List[str], pd.DataFrame, SelectFromModel, VarianceThreshold]
        - If ``return_selector=False``:
            - Returns a list of relevant feature names if ``X`` is a DataFrame.
            - Returns a NumPy array with selected features if ``X`` is an array-like.
        - If ``return_selector=True``:
            - Returns the selector object (`SelectFromModel` or `VarianceThreshold`).
    
    Raises
    ------
    ValueError
        If the estimator does not have `feature_importances_` or `coef_` attributes 
        when ``y`` is provided.
        If the threshold is not valid (e.g., not between 0 and 1 for float values).
        If ``y`` is provided but its length does not match the number of samples in ``X``.
        If the specified correlation method is unsupported.
    
    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from gofast.utils.mlutils import select_feature_importances
    >>> X, y = make_classification(n_samples=1000, n_features=10, n_informative=3)
    >>> clf = RandomForestClassifier()
    >>> X_selected = select_feature_importances(
    ...     estimator=clf, 
    ...     X=X, 
    ...     y=y, 
    ...     threshold="mean", 
    ...     prefit=False
    ... )
    >>> X_selected.shape
    (1000, n_selected_features)
    
    Using `return_selector=True` to get the selector object:
    
    >>> selector = select_feature_importances(
    ...     estimator=clf, 
    ...     X=X, 
    ...     y=y, 
    ...     threshold="mean", 
    ...     prefit=False, 
    ...     return_selector=True
    ... )
    >>> selector.get_support()
    array([ True, False, ..., True])
    
    Performing unsupervised feature selection based on variance:
    
    >>> from sklearn.datasets import make_blobs
    >>> X, _ = make_blobs(n_samples=500, n_features=8, centers=3, cluster_std=1.0)
    >>> X_selected_variance = select_feature_importances(
    ...     estimator=RandomForestClassifier(),  # Estimator is ignored when y=None
    ...     X=X, 
    ...     y=None, 
    ...     threshold=1.0, 
    ...     prefit=False, 
    ...     verbose=2
    ... )
    >>> print(X_selected_variance)
    ['feature1', 'feature3', 'feature5']
    
    Using `return_selector=True` for variance-based selection:
    
    >>> selector_variance = select_feature_importances(
    ...     estimator=RandomForestClassifier(),
    ...     X=X, 
    ...     y=None, 
    ...     threshold=1.0, 
    ...     prefit=False, 
    ...     return_selector=True
    ... )
    >>> selector_variance.get_support()
    array([ True, False, True, False, True, False, False, False])
    ```
    # noqa: E501
    
    Notes
    -----
    - **Supervised vs. Unsupervised:**
        - When ``y`` is provided, the function performs supervised feature 
          selection using the feature importances or coefficients from the estimator.
        - When ``y`` is ``None``, the function performs unsupervised feature 
          selection based on feature variance.
          
    - **Estimator Requirements:**
        - For supervised selection, the estimator must have either `feature_importances_` 
          or `coef_` attributes after fitting. This can be overridden by specifying 
          `importance_getter` in ``kwargs``.
        - For unsupervised selection, the estimator is not used; instead,
           feature variance is utilized.
          
    - **Threshold Interpretation:**
        - In supervised selection, the threshold can be a float or a string 
          (`"mean"`, `"median"`, etc.) to dynamically set the threshold based
          on feature importances.
        - In unsupervised selection, the threshold represents the minimum 
          variance required for a feature to be retained.
          
    - **Verbose Levels:**
        - The ``verbose`` parameter provides feedback on the selection process:
            - `0`: No messages.
            - `1`: Basic messages about the number of features selected.
            - `2`: Detailed messages including intermediate steps.
            - `3`: Debug-level messages with extensive details.
            
    - **Return Selector:**
        - Setting ``return_selector=True`` allows users to access the
          underlying selector object, which can be useful for inspecting
          feature support or integrating into pipelines.
          
    - **Estimator Prefit:**
        - If ``prefit=True``, the estimator must already be fitted. Otherwise,
           it will be fitted within the function.
          
    - **Data Integrity:**
        - The function does not modify the original ``X`` data. The selection 
          is performed and returned as a new array or DataFrame.
          
    See Also
    --------
    sklearn.feature_selection.SelectFromModel :
        Meta-transformer for feature selection based on importance weights.
    sklearn.feature_selection.VarianceThreshold :
        Feature selector that removes all low-variance features.
    sklearn.base.BaseEstimator : Base class for all estimators in scikit-learn.
    
    References
    ----------
    .. [1] McKinney, W. (2010). Data Structures for Statistical Computing in 
           Python. In *Proceedings of the 9th Python in Science Conference*, 
           51-56.
    .. [2] Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). 
           Scikit-learn: Machine Learning in Python. *Journal of Machine 
           Learning Research*, 12, 2825-2830.
    .. [3] Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). 
           Array programming with NumPy. *Nature*, 585(7825), 357-362.
    .. [4] Pandas Development Team. (2023). *pandas documentation*. 
           https://pandas.pydata.org/pandas-docs/stable/
    """
    # Validate input DataFrame and build if necessary
    data = build_data_if(
        data=X, 
        to_frame=True, 
        input_name='feature_', 
        raise_exception=True
    )
    
    # Validate target variable
    if y is not None:
        if isinstance(y, str):
            raise ValueError(
                "When y is provided, it should be an array-like object, not a string.")
        elif is_array_like(y):
            if len(y) != data.shape[0]:
                raise ValueError(
                    "The length of y does not match the number of samples in X.")
        else:
            raise ValueError(
                "The 'y' parameter must be an array-like object or None.")
    
    # Validate threshold
    if isinstance(threshold, float):
        if not (0 <= threshold <= 1) and not isinstance(threshold, str):
            raise ValueError(
                "The 'threshold' must be between 0 and 1 or"
                " a valid string for dynamic thresholds.")
    elif isinstance(threshold, str):
        if threshold not in ["mean", "median", "auto"]:
            raise ValueError(
                "Invalid string for 'threshold'. Choose from 'mean', 'median', or 'auto'.")
    else:
        raise ValueError(
            "The 'threshold' must be a float or a string ('mean', 'median', 'auto').")
    
    # Select appropriate feature selector based on whether y is provided
    if y is not None:
        # Supervised feature selection using SelectFromModel
        selector = SelectFromModel(
            estimator=estimator, 
            threshold=threshold, 
            prefit=prefit, 
            **kwargs
        )
        
        if not prefit:
            selector.fit(data, y)
            if verbose >= 2:
                print(f"Estimator {estimator.__class__.__name__} fitted on data.")
        
        # Extract feature importances or coefficients
        if hasattr(estimator, 'feature_importances_'):
            importances = estimator.feature_importances_
            if verbose >= 2:
                print(f"Feature importances: {importances}")
        elif hasattr(estimator, 'coef_'):
            importances = estimator.coef_
            # Handle multi-output estimators
            if importances.ndim > 1:
                importances = np.mean(np.abs(importances), axis=0)
            else:
                importances = np.abs(importances)
            if verbose >= 2:
                print(f"Feature coefficients: {importances}")
        else:
            raise ValueError(
                f"The estimator {estimator.__class__.__name__} does not have "
                "`feature_importances_` or `coef_` attributes."
            )
        
        if verbose >= 1:
            n_selected = selector.transform(data).shape[1]
            print(
                f"Number of features meeting the threshold={threshold}: {n_selected}")
    else:
        # Unsupervised feature selection using VarianceThreshold
        if verbose >= 1:
            print(
                "No target provided. Selecting features based on variance.")
        
        selector = VarianceThreshold(
            threshold=threshold, 
            **kwargs
        )
        selector.fit(data)
        
        # Extract variances
        variances = data.var()
        if verbose >= 2:
            print(f"Feature variances: {variances.values}")
        
        # Select features based on variance threshold
        relevant_features = variances[variances > threshold].index.tolist()
        if verbose >= 1:
            n_selected = len(relevant_features)
            print("Number of features meeting the"
                  f" variance threshold={threshold}: {n_selected}")
    
    # Transform the data
    X_selected = selector.transform(data)
    
    # Handle verbose messaging for unsupervised selection
    if y is None and verbose >= 1:
        n_selected = X_selected.shape[1]
        print(f"Selected {n_selected} feature(s)"
              " based on variance threshold={threshold}.")
    
    # Visualization
    if view:
        if y is not None:
            if verbose >= 1:
                if X_selected.shape[1] == 0:
                    warnings.warn(
                        "No features meet the threshold for visualization.",
                        UserWarning)
                else:
                    print(f"Visualizing correlations of {X_selected.shape[1]}"
                          " relevant feature(s).")
            
            if X_selected.shape[1] > 0:
                feature_indices = selector.get_support(indices=True)
                selected_features = data.columns[feature_indices]
                importances_selected = importances[feature_indices]
                
                plt.figure(figsize=(10, 6))
                sns.barplot(
                    x=np.abs(importances_selected), 
                    y=selected_features, 
                    orient='h'
                )
                plt.xlabel('Absolute Feature Importance')
                plt.title('Selected Features Based on Feature Importances')
                plt.tight_layout()
                plt.show()
        else:
            if verbose >= 1:
                if X_selected.shape[1] == 0:
                    warnings.warn(
                        "No features meet the variance threshold"
                        " for visualization.", UserWarning)
                else:
                    print(f"Visualizing variances of {X_selected.shape[1]}"
                          " relevant feature(s).")
            
            if X_selected.shape[1] > 0:
                feature_indices = selector.get_support(indices=True)
                selected_features = data.columns[feature_indices]
                variances_selected = variances[selected_features]
                
                plt.figure(figsize=(10, 6))
                sns.barplot(
                    x=variances_selected.values, 
                    y=selected_features, 
                    orient='h'
                )
                plt.xlabel('Variance')
                plt.title('Selected Features Based on Variance')
                plt.tight_layout()
                plt.show()
    
    # Verbosity messaging
    if verbose >= 1 and y is not None:
        print(f"Selected {X_selected.shape[1]} relevant feature(s)"
              f" based on correlation threshold={threshold}.")

    # Return the selector object if requested
    if return_selector:
        return selector
    
    # Return the selected features as a DataFrame or array
    if isinstance(X, pd.DataFrame):
        selected_columns = data.columns[selector.get_support(indices=True)]
        return data[selected_columns]
    
    return X_selected

@ensure_pkg ("shap", extra = ( 
    "`get_feature_contributions` needs SHapley Additive exPlanations (SHAP)"
    " package to be installed. Instead, you can use"
    " `gofast.utils.display_feature_contributions` for contribution scores" 
    " and `gofast.analysis.get_feature_importances` for PCA quick evaluation."
    )
 )
def get_feature_contributions(X, model=None, view=False):
    """
    Calculate the SHAP (SHapley Additive exPlanations) values to determine 
    the contribution of each feature to the model's predictions for each 
    instance in the dataset and optionally display a visual summary.

    Parameters
    ----------
    X : ndarray or DataFrame
        The feature matrix for which to calculate feature contributions.
    model : sklearn.base.BaseEstimator, optional
        A pre-trained tree-based machine learning model from scikit-learn (e.g.,
        RandomForest). If None, a new RandomForestClassifier will be trained on `X`.
    view : bool, optional
        If True, displays a visual summary of feature contributions using SHAP's
        visualization tools. Default is False.

    Returns
    -------
    ndarray
        A matrix of SHAP values where each row corresponds to an instance and
        each column corresponds to a feature's contribution to that instance's 
        prediction.

    Notes
    -----
    The function defaults to creating and using a RandomForestClassifier if no 
    model is provided. It is more efficient to pass a pre-trained model if 
    available.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from gofast.utils.mlutils import get_feature_contributions
    >>> data = load_iris()
    >>> X = data['data']
    >>> model = RandomForestClassifier(random_state=42)
    >>> model.fit(X, data['target'])
    >>> contributions = get_feature_contributions(X, model, view=True)
    """
    import shap
    
    # If no model is provided, train a RandomForestClassifier
    if model is None:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        # Dummy target, assuming unsupervised setup for example
        model.fit(X, np.zeros(X.shape[0]))  

    # Create the Tree explainer and calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # SHAP returns a list for multi-class, we sum across all classes for overall importance
    if isinstance(shap_values, list):
        shap_values = np.sum(np.abs(shap_values), axis=0)

    # Visualization if view is True
    if view:
        shap.summary_plot(shap_values, X, feature_names=model.feature_names_in_ if hasattr(
            model, 'feature_names_in_') else None)

    return shap_values


@ensure_pkg(
    "shap", extra="SHapley Additive exPlanations (SHAP) is needed.",
    partial_check=True,
    condition=lambda *args, **kwargs: kwargs.get("pkg") == "shap"
)
def display_feature_contributions(
    X, 
    y: Optional[Union[pd.Series, np.ndarray]] = None, 
    estimator: Optional[BaseEstimator] = None,
    threshold: Union[float, str] = 0.1,
    prefit: bool = True,
    pkg: Optional[str] = None,
    view: bool = False, 
    show_grid: bool=True, 
    return_selector: bool=False, 
    verbose: int = 0,
    **kwargs
) -> Union[Dict[str, float], Tuple[Dict[str, float], Union[
    SelectFromModel, VarianceThreshold]]]:
    """
    Trains a model to determine the importance of features in the dataset 
    and optionally displays these importances visually using SHAP 
    or Matplotlib.
    
    The ``display_feature_contributions`` function selects relevant features 
    from a dataset based on feature importances derived from a supervised 
    estimator or based on feature variance for unsupervised scenarios. 
    This is useful for feature selection in both predictive modeling and 
    exploratory data analysis.
    
    .. math::
        \text{Relevant Features} = 
        \begin{cases}
            \{ x_i \in X \mid | \rho(x_i, y) | > \text{threshold} \}\\
                & \text{if } y \text{ is provided} \\
            \{ x_i \in X \mid \text{Var}(x_i) > \text{threshold} \}\\
                & \text{if } y \text{ is None}
        \end{cases}
    
    Parameters
    ----------
    X : ndarray or DataFrame
        The feature matrix from which to determine feature importances. This 
        should not include the target variable.
    y : array-like, optional
        The target variable array. If provided, it will be used for supervised 
        learning. If None, an unsupervised approach will be used 
        (feature importances based on feature variance).
    view : bool, optional
        If True, display feature importances using SHAP's summary plot or 
        a Matplotlib bar plot.
        Defaults to False.
    pkg : str, optional
        The visualization package to use. Options are 'shap' or 'matplotlib'.
        If 'shap' is selected, SHAP summary plot is used.
        If 'matplotlib' is selected, a bar plot with feature importance values
        on top of each bar is used.
        Defaults to 'matplotlib'.
    estimator : BaseEstimator, optional
        The estimator to use for supervised feature selection. If None, a 
        RandomForestClassifier or RandomForestRegressor is used based on 
        whether y is classification or regression.
    threshold : float or str, optional, default=0.1
        The threshold value to use for feature selection. 
        - If y is provided, features with importance greater than or equal 
          to this value are retained. Can also be a string such as "mean" or 
          "median" to use the mean or median feature importance as the threshold.
        - If y is None, features with variance greater than this value are retained.
    prefit : bool, default=True
        Whether the estimator is expected to be prefit. If True, 
        estimator should already be fitted; otherwise, it will be fitted 
        on X and y.
    return_selector: 
        If ``return_selector=True`` allows users to access the 
        underlying selector object, which can be useful for inspecting
        feature support or integrating into pipelines.
          
    verbose : int, default=0
        Controls the verbosity: the higher, the more messages.
        
    **kwargs : additional keyword arguments
        Additional arguments passed to `SelectFromModel` or `VarianceThreshold`.
    
    Returns
    -------
    gofast.api.summary.Summary object 
        Collect the selector and feature importance dictionnary
        - summary.feature_importances_:
            - Returns a dictionary where keys are feature names and values 
              are their corresponding importances (if y is provided) or variances
              (if y is None).
        - summary.selector:
            - Returns a selector object.
    
    Raises
    ------
    ValueError
        If the estimator does not have `feature_importances_` or `coef_` attributes 
        when ``y`` is provided.
        If the threshold is not valid (e.g., not between 0 and 1 for float values).
        If ``y`` is provided but its length does not match the number of samples in ``X``.
    
    Examples
    --------
    >>> from sklearn.datasets import load_iris 
    >>> from gofast.utils.ml.feature_selection import display_feature_contributions
    >>> data = load_iris()
    >>> X = data['data']
    >>> y = data['target']
    >>> feature_names = data['feature_names']
    >>> summary = display_feature_contributions(X, y=y, view=True, return_summary=True)
    >>> print(summary.feature_importances_)
    {'feature_0': 0.10612761987750428,
     'feature_1': 0.02167809317736852,
     'feature_2': 0.4361295069034437,
     'feature_3': 0.43606478004168353}
    
    >>> # Using `return_selector=True` to get the selector object:
    >>> feature_importances, selector = display_feature_contributions(
    ...     X, y=y, view=True, prefit=False, return_selector=True)
    >>> selector.get_support()
    array([ True, False, ..., True])
    
    >>> # Performing unsupervised feature selection based on variance:
    >>> from sklearn.datasets import make_blobs
    >>> X_unsup, _ = make_blobs(n_samples=500, n_features=8, centers=3, cluster_std=1.0)
    >>> display_feature_contributions(
    ...     X=X_unsup, y=None, view=True, pkg='matplotlib', threshold=1.0, verbose=2)
    >>> print(feature_variances)
    {'feature1': 1.2, 'feature3': 1.5, 'feature5': 1.3}
    
    >>> # Using `return_selector=True` for variance-based selection:
    >>> feature_variances, selector_variance = display_feature_contributions(
    ...     X=X_unsup, y=None, view=True, pkg='matplotlib', threshold=1.0, 
      verbose=2, return_selector=True)
    >>> selector_variance.get_support()
    array([ True, False, True, False, True, False, False, False])

    
    Notes
    -----
    - **Supervised vs. Unsupervised:**
        - When ``y`` is provided, the function performs supervised feature selection 
          using the feature importances or coefficients from the estimator.
        - When ``y`` is ``None``, the function performs unsupervised feature selection 
          based on feature variance.
          
    - **Estimator Requirements:**
        - For supervised selection, the estimator must have either 
          `feature_importances_` 
          or `coef_` attributes after fitting. This can be overridden by 
          specifying `estimator` in the function parameters.
        - For unsupervised selection, the estimator is not utilized; instead,
           feature variance is used as the selection metric.
          
    - **Threshold Interpretation:**
        - In supervised selection, the threshold can be a float or a string (`"mean"`, 
          `"median"`, etc.) to dynamically set the threshold based on feature importances.
        - In unsupervised selection, the threshold represents the minimum variance required 
          for a feature to be retained.
          
    - **Verbose Levels:**
        - The ``verbose`` parameter provides feedback on the selection process:
            - `0`: No messages.
            - `1`: Basic messages about the number of features selected.
            - `2`: Detailed messages including intermediate steps like feature
              importances or variances.
            - `3`: Debug-level messages with extensive details.
            
    - **Return Selector Option:**
        - Setting ``return_selector=True`` allows users to access the 
          underlying selector object, which can be useful for inspecting
          feature support or integrating into pipelines.
          
    - **Estimator Prefit:**
        - If ``prefit=True``, the estimator must already be fitted.
           Otherwise, it will be fitted within the function.
          
    - **Data Integrity:**
        - The function does not modify the original ``X`` data. The selection 
          is performed  and returned as a new dictionary or alongside 
          the selector object.
          
    - **Visualization Enhancements:**
        - When ``pkg='matplotlib'``, the feature importance or variance value
           is placed on top of each bar for better interpretability.
          
    See Also
    --------
    sklearn.feature_selection.SelectFromModel :
        Meta-transformer for feature selection based on importance weights.
    sklearn.feature_selection.VarianceThreshold : 
        Feature selector that removes all low-variance features.
    sklearn.base.BaseEstimator : Base class for all estimators in scikit-learn.
    
    References
    ----------
    .. [1] McKinney, W. (2010). Data Structures for Statistical Computing in 
           Python. In *Proceedings of the 9th Python in Science Conference*, 
           51-56.
    .. [2] Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). 
           Scikit-learn: Machine Learning in Python. *Journal of Machine 
           Learning Research*, 12, 2825-2830.
    .. [3] Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). 
           Array programming with NumPy. *Nature*, 585(7825), 357-362.
    .. [4] Pandas Development Team. (2023). *pandas documentation*. 
           https://pandas.pydata.org/pandas-docs/stable/
    """
    # Set default pkg to 'matplotlib' if not provided
    pkg = 'matplotlib' if pkg in [ "matplotlib", "mpl", None
                                  ] else str(pkg).lower()
    
    # Validate data types (assumed to be imported from gofast)
    validate_data_types(X, nan_policy="raise", error="raise")

    # Initialize the estimator if not provided and y is given
    if y is not None:
        if estimator is None:
            target_type = type_of_target(y)
            if target_type in ['continuous', 'continuous-multioutput']:
                estimator = RandomForestRegressor(
                    n_estimators=100, random_state=42)
                if verbose >= 2:
                    print("Initialized RandomForestRegressor for continuous target.")
            else:
                estimator = RandomForestClassifier(
                    n_estimators=100, random_state=42)
                if verbose >= 2:
                    print("Initialized RandomForestClassifier"
                          " for classification target.")
    else:
        # For unsupervised, use VarianceThreshold; estimator is ignored
        estimator = None
    
    # Supervised feature selection
    if y is not None:
        if not prefit:
            estimator.fit(X, y)
            if verbose >= 2:
                print(f"Estimator {estimator.__class__.__name__} fitted on data.")
        
        # Extract feature importances or coefficients
        if hasattr(estimator, 'feature_importances_'):
            importances = estimator.feature_importances_
            if verbose >= 2:
                print(f"Feature importances: {importances}")
        elif hasattr(estimator, 'coef_'):
            importances = estimator.coef_
            # Handle multi-output estimators
            if importances.ndim > 1:
                importances = np.mean(np.abs(importances), axis=0)
            else:
                importances = np.abs(importances)
            if verbose >= 2:
                print(f"Feature coefficients: {importances}")
        else:
            extra_msg =''
            if ( 
                estimator is not None and  
                'RandomForest' in get_estimator_name(estimator)
                ): 
                extra_msg = (
                    "This error occurs because you did not provided a"
                    " fitted estimator. To use the default estimator,"
                    " set ``prefit=False``."
                )
            raise ValueError(
                f"The estimator {estimator.__class__.__name__} does not have "
                f"`feature_importances_` or `coef_` attributes. {extra_msg}"
            )
        
        # Create a dict of feature importances
        feature_names = X.columns if isinstance(X, pd.DataFrame) else [
            f'feature_{i}' for i in range(X.shape[1])]
        feature_importance_dict = dict(zip(feature_names, importances))

        
    else:
        # Unsupervised feature selection using VarianceThreshold
        from sklearn.feature_selection import VarianceThreshold
        
        selector = VarianceThreshold(
            threshold=threshold, 
            **kwargs
        )
        selector.fit(X)
        
        # Extract variances
        variances = X.var() if isinstance(X, pd.DataFrame) else np.var(X, axis=0)
        feature_names = X.columns if isinstance(X, pd.DataFrame) else [
            f'feature_{i}' for i in range(X.shape[1])]
        
        if verbose >= 2:
            print(f"Feature variances: {variances}")
        
        # Create a dict of feature variances
        feature_importance_dict = dict(zip(feature_names, variances))
        
        if verbose >= 1:
            n_selected = selector.transform(X).shape[1]
            print("Number of features meeting the variance"
                  f" threshold={threshold}: {n_selected}")
    
    # Transform the data
    # X_selected = selector.transform(X)
    # Visualization
    if view:
        if y is not None:
            if pkg == "shap":
                import shap 
                explainer = shap.TreeExplainer(estimator)
                shap_values = explainer.shap_values(X)
                shap.summary_plot(shap_values, X, feature_names=feature_names)
            elif pkg =="matplotlib":
                plt.figure(figsize=(10, 5))
                
                # Sort features by importance
                sorted_features = sorted(feature_importance_dict.items(), 
                                         key=lambda x: x[1], reverse=True)
                features, importances_sorted = zip(*sorted_features)
                
                bars = plt.bar(features, importances_sorted, color='skyblue')
                plt.title('Feature Importances')
                plt.xlabel('Feature')
                plt.ylabel('Importance')
                plt.xticks(rotation=45, ha='right')
                
                # Place the importance value on top of each bar
                for bar, importance in zip(bars, importances_sorted):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2.0, height, 
                             f'{importance:.3f}', 
                             ha='center', va='bottom')
                plt.grid(show_grid)
                plt.tight_layout()
                plt.show()
            else:
                warnings.warn(
                    f"Unsupported visualization package: {pkg}. "
                    "Supported packages are 'shap' and 'matplotlib'.",
                    UserWarning)
        else:
            if pkg =="matplotlib":

                plt.figure(figsize=(10, 5))
                
                # Sort features by variance
                sorted_features = sorted(feature_importance_dict.items(), 
                                         key=lambda x: x[1], reverse=True)
                features, variances_sorted = zip(*sorted_features)
                
                bars = plt.bar(features, variances_sorted, color='skyblue')
                plt.title('Feature Variances')
                plt.xlabel('Feature')
                plt.ylabel('Variance')
                plt.xticks(rotation=45, ha='right')
                
                # Place the variance value on top of each bar
                for bar, variance in zip(bars, variances_sorted):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2.0, height, 
                             f'{variance:.3f}', 
                             ha='center', va='bottom')
                
                plt.grid(show_grid)
                plt.tight_layout()
                plt.show()
            else:
                warnings.warn(
                    f"Unsupported visualization package: {pkg}."
                    " Supported package is 'matplotlib' for unsupervised"
                    " selection.", UserWarning)
    
    # Create and print summary using ReportFactory 
    summary = ReportFactory(title="Feature Contributions Table").add_mixed_types(
        feature_importance_dict)
    summary.feature_importances_=feature_importance_dict
    summary.selector = selector if y is None else estimator 
    
    if verbose >0:
        print(summary)

    if return_selector:
        return summary.selector
    
    return summary 

    

