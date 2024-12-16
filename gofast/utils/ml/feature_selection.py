# -*- coding: utf-8 -*-
"""
Utilities for features selections, and inspections.
"""
import copy
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

from ..._gofastlog import gofastlog
from ...api.docstring import DocstringComponents, _core_docs 
from ...api.types import DataFrame
from ...api.summary import ReportFactory 
from ...core.array_manager import to_numeric_dtypes
from ...core.checks import is_in_if, is_iterable 
from ...core.io import is_data_readable
from ...decorators import Dataify
from ..base_utils import select_features
from ..deps_utils import ensure_pkg
from ..validator import build_data_if, validate_data_types

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
    'display_feature_contributions'
    ]


@Dataify(auto_columns=True)
@is_data_readable
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
    threshold: float=.95 , 
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
    
    df = select_features(data, None, 'number')
        
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
                      

def select_feature_importances(
        clf, X, y=None, *, threshold=0.1, prefit=True, 
        verbose=0, return_selector=False, **kwargs
        ):
    """
    Select features based on importance thresholds after model fitting.
    
    Parameters
    ----------
    clf : estimator object
        The estimator from which the feature importances are derived. Must have
        either `feature_importances_` or `coef_` attributes after fitting, unless
        `importance_getter` is specified in `kwargs`.
        
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The training input samples.
        
    y : array-like of shape (n_samples,), default=None
        The target values (class labels) as integers or strings.
        
    threshold : float, default=0.1
        The threshold value to use for feature selection. Features with importance
        greater than or equal to this value are retained.
        
    prefit : bool, default=True
        Whether the estimator is expected to be prefit. If `True`, `clf` should
        already be fitted; otherwise, it will be fitted on `X` and `y`.
        
    verbose : int, default=0
        Controls the verbosity: the higher, the more messages.
        
    return_selector : bool, default=False
        Whether to return the selector object instead of the transformed data.
        
    **kwargs : additional keyword arguments
        Additional arguments passed to `SelectFromModel`.
    
    Returns
    -------
    X_selected or selector : array or SelectFromModel object
        The selected features in `X` if `return_selector` is False, or the
        selector object itself if `return_selector` is True.
        
    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from gofast.utils.mlutils import select_feature_importances
    >>> X, y = make_classification(n_samples=1000, n_features=10, n_informative=3)
    >>> clf = RandomForestClassifier()
    >>> X_selected = select_feature_importances(clf, X, y, threshold="mean", prefit=False)
    >>> X_selected.shape
    (1000, n_selected_features)
    
    Using `return_selector=True` to get the selector object:
    
    >>> selector = select_feature_importances(
        clf, X, y, threshold="mean", prefit=False, return_selector=True)
    >>> selector.get_support()
    array([True, False, ..., True])
    """
    # Check if the classifier is fitted based on the presence of attributes
    if not prefit and (hasattr(clf, 'feature_importances_') or hasattr(clf, 'coef_')):
        warnings.warn(f"The estimator {clf.__class__.__name__} appears to be fitted. "
                      "Consider setting `prefit=True` or refit the estimator.",UserWarning)
    try:threshold = float(threshold ) 
    except: pass 

    selector = SelectFromModel(clf, threshold=threshold, prefit=prefit, **kwargs)
    
    if not prefit:
        selector.fit(X, y)
    
    if verbose:
        n_features = selector.transform(X).shape[1]
        print(f"Number of features meeting the threshold={threshold}: {n_features}")
    
    return selector if return_selector else selector.transform(X)


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
    partial_check= True,
    condition= lambda *args, **kwargs: kwargs.get("pkg")=="shap"
    )
def display_feature_contributions(
        X, y=None, view=False, pkg=None):
    """
    Trains a RandomForest model to determine the importance of features in
    the dataset and optionally displaysthese importances visually using SHAP.

    Parameters
    ----------
    X : ndarray or DataFrame
        The feature matrix from which to determine feature importances. This 
        should not include the target variable.
    y : ndarray, optional
        The target variable array. If provided, it will be used for supervised 
        learning. If None, an unsupervised approach will be used 
        (feature importances based on feature permutation).
    view : bool, optional
        If True, display feature importances using SHAP's summary plot.
        Defaults to False.

    Returns
    -------
    dict
        A dictionary where keys are feature names and values are their 
        corresponding importances as determined by the RandomForest model.

    Examples
    --------
    >>> from sklearn.datasets import load_iris 
    >>> from gofast.utils.mlutils import display_feature_contributions
    >>> data = load_iris()
    >>> X = data['data']
    >>> feature_names = data['feature_names']
    >>> display_feature_contributions(X, view=True)
    {'sepal length (cm)': 0.112, 'sepal width (cm)': 0.032, 
     'petal length (cm)': 0.423, 'petal width (cm)': 0.433}
    """
    pkg ='shap' if str(pkg).lower() =='shap' else 'matplotlib'
    
    validate_data_types(X, nan_policy="raise", error ="raise")
    
    # Initialize the RandomForest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Fit the model on the provided features, with or without a target variable
    if y is not None:
        model.fit(X, y)
    else:
        # Fit a dummy target if y is None, assuming unsupervised setup
        model.fit(X, range(X.shape[0]))

    # Extract feature importances
    importances = model.feature_importances_
    
    # Optionally, display the feature importances using the chosen visualization package
    feature_names = model.feature_names_in_ if hasattr(
            model, 'feature_names_in_') else [f'feature_{i}' for i in range(X.shape[1])]
    if view:
        if pkg.lower() == "shap":
            import shap
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            shap.summary_plot(shap_values, X, feature_names=feature_names)
            
        elif pkg.lower() == "matplotlib":
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            indices = range(len(importances))
            plt.title('Feature Importances')
            plt.bar(indices, importances, color='skyblue', align='center')
            plt.xticks(indices, feature_names, rotation=45)
            plt.xlabel('Feature')
            plt.ylabel('Importance')
            plt.show()

    # Map feature names to their importances
    feature_importance_dict = dict(zip(feature_names, importances))
    summary = ReportFactory(title="Feature Contributions Table",).add_mixed_types(
        feature_importance_dict)
    
    print(summary)
