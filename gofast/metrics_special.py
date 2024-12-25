# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

""" 
Specialized metrics and utility functions for model evaluation.
This includes functions for assessing classifier and regression performance, 
visualizing metrics (e.g., confusion matrix, ROC curve, precision-recall), 
and performing sensitivity analysis. These utilities extend the functionality 
of scikit-learn's metrics by offering more flexible and domain-specific tools.
"""

from __future__ import annotations
from numbers import Real, Integral 
import itertools
import warnings
from textwrap import dedent

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
    
from sklearn.metrics import (
    accuracy_score,  confusion_matrix, f1_score,
    jaccard_score, mean_absolute_error, mean_squared_error,
    precision_recall_curve, precision_score, recall_score, r2_score,
    roc_auc_score, roc_curve
)
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import label_binarize

from .api.formatter import MetricFormatter, DescriptionFormatter 
from .api.summary import TW, ReportFactory, assemble_reports
from .api.summary import ResultSummary 
from .core.checks import exist_features, is_iterable
from .core.utils import normalize_string
from .compat.sklearn import StrOptions, HasMethods, Interval, validate_params 
from .decorators import Substitution, Appender
from .utils.validator import (
    _ensure_y_is_valid, _is_numeric_dtype, check_epsilon, check_is_fitted, 
    validate_multioutput, validate_nan_policy, check_array, build_data_if
)

__all__= [
     'analyze_target',
     'assess_classifier_metrics',
     'assess_regression_metrics',
     'display_confusion_matrix',
     'display_precision_recall',
     'display_roc',
     'evaluate_confusion_matrix',
     'geo_information_value',
     'jaccard_flex',
     'madev_flex',
     'mae_flex',
     'mse_flex',
     'precision_recall_tradeoff',
     'prediction_stability_score',
     'r2_flex',
     'relative_sensitivity_score',
     'relative_sensitivity_scores',
     'rmse_flex',
     'roc_tradeoff', 
     'coverage_score', 
 ]

# --------- doc templates -----------------------------------------------------
_shared_doc_kwargs= { 
    "model_desc": ( 
        "A trained model with a `.predict()` method. The model"
        " should accept `X` as input and return predictions for each instance."
        ), 
    "X_desc": ( 
        "The input data to the model. Each row represents an instance, and each"
        "column corresponds to a feature. The features should correspond to the"
        "names provided in `feature_names`."
        ), 
    "feature_names_desc": ( 
        "The feature(s) for which the relative sensitivity"
        " is to be computed. Each feature name should correspond to a column"
        " in `X`. If `None`, all features will be used."
        ), 
    "plot_type_desc": ( 
        "The type of plot to generate for the sensitivity results. If `None`,"
        "no plot is generated. Available plot styles include:"
        "- `'hist'`: A histogram of relative sensitivities with a KDE."
        "- `'line'`: A line plot of relative sensitivities for each feature."
        "- `'bar'`: A bar plot of relative sensitivities."
        "- `'scatter'`: A scatter plot showing relative sensitivities."
        "- `'box'`: A box plot to show the distribution of relative sensitivities."
        ), 
    "interpret_desc": ( 
        "If `True`, additional analysis details are printed to"
        " the console. This includes a ranking of features by their relative "
        "sensitivity and key insights into which features are the most sensitive."
        )
    }

_shared_params= """\
Parameters
----------
model : object
    %(model_desc)s
X : array-like, shape (n_samples, n_features)
    %(X_desc)s
feature_names : str or list of str, Optional
    %(feature_names_desc)s
plot_type : {'hist', 'line', 'bar', 'scatter', 'boxplot', 'box'}, optional
    %(plot_type_desc)s
interpret : bool, default=False
    %(interpret_desc)s
"""

# ----------------------- end doc templates ---------------------------------

def coverage_score(
    y_true,
    y_lower,
    y_upper,
    nan_policy='propagate',
    fill_value=np.nan,
    verbose=1
):
    r"""
    Compute the coverage score of prediction intervals, measuring
    the fraction of instances where the true value lies within a
    provided lower and upper bound. This metric is useful for
    evaluating uncertainty estimates in probabilistic forecasts,
    resembling a probabilistic analog to traditional accuracy.

    Formally, given observed true values 
    :math:`y = \{y_1, \ldots, y_n\}`, and corresponding interval 
    bounds :math:`\{l_1, \ldots, l_n\}` and 
    :math:`\{u_1, \ldots, u_n\}`, the coverage score is defined
    as:

    .. math::
       \text{coverage} = \frac{1}{n}\sum_{i=1}^{n}
       \mathbf{1}\{ l_i \leq y_i \leq u_i \},

    where :math:`\mathbf{1}\{\cdot\}` is an indicator function 
    that equals 1 if :math:`y_i` falls within the interval 
    :math:`[l_i, u_i]` and 0 otherwise.

    Parameters
    ----------
    y_true : array-like
        The true observed values. Must be array-like and numeric.
    y_lower : array-like
        The lower bound predictions for each instance, matching 
        `<y_true>` in shape and alignment.
    y_upper : array-like
        The upper bound predictions, aligned with `<y_true>` and 
        `<y_lower>`.
    nan_policy: {'omit', 'propagate', 'raise'}, optional
        Defines how to handle NaN values in `<y_true>`, `<y_lower>`, 
        or `<y_upper>`:
        
        - ``'propagate'``: NaNs remain, potentially affecting the 
          result or causing it to be NaN.
        - ``'omit'``: NaNs lead to omission of those samples from 
          coverage calculation.
        - ``'raise'``: Encountering NaNs raises a ValueError.
    fill_value: scalar, optional
        The value used to fill missing entries if `<allow_missing>` 
        is True. Default is `np.nan`. If `nan_policy='omit'`, these 
        filled values may be omitted.
    verbose: int, optional
        Controls the level of verbosity for internal logging:
        
        - 0: No output.
        - 1: Basic info (e.g., final coverage).
        - 2: Additional details (e.g., handling NaNs).
        - 3: More internal state details (shapes, conversions).
        - 4: Very detailed output (e.g., sample masks).
    
    Returns
    -------
    float
        The coverage score, a number between 0 and 1. A value closer 
        to 1.0 indicates that the provided intervals successfully 
        capture a large portion of the true values.

    Notes
    -----
    The `<nan_policy>` or `<allow_missing>` parameters control how 
    missing values are handled. If `nan_policy='raise'` and NaNs 
    are found, an error is raised. If `nan_policy='omit'`, these 
    samples are excluded from the calculation. If `nan_policy` is 
    'propagate', NaNs remain, potentially influencing the result 
    (e.g., coverage might become NaN if the fraction cannot be 
    computed).

    When `<allow_missing>` is True, missing values are filled with 
    `<fill_value>`. This can interact with `nan_policy`. For 
    instance, if `fill_value` is NaN and `nan_policy='omit'`, 
    those samples are omitted anyway.

    By adjusting these parameters, users can adapt the function 
    to various data cleanliness scenarios and desired behaviors.

    Examples
    --------
    >>> from gofast.metrics_special import coverage_score
    >>> import numpy as np
    >>> y_true = np.array([10, 12, 11, 9])
    >>> y_lower = np.array([9, 11, 10, 8])
    >>> y_upper = np.array([11, 13, 12, 10])
    >>> cov = coverage_score(y_true, y_lower, y_upper)
    >>> print(f"Coverage: {cov:.2f}")
    Coverage: 1.00

    See Also
    --------
    numpy.isnan : Identify missing values in arrays.

    References
    ----------
    .. [1] Gneiting, T. & Raftery, A. E. (2007). "Strictly Proper 
           Scoring Rules, Prediction, and Estimation." J. Amer. 
           Statist. Assoc., 102(477):359–378.
    """
    y_true_arr = np.asarray(y_true)
    y_lower_arr = np.asarray(y_lower)
    y_upper_arr = np.asarray(y_upper)
    
    if verbose >= 3:
        print("Converting inputs to arrays...")
        print("Shapes:", y_true_arr.shape, y_lower_arr.shape, y_upper_arr.shape)

    if y_true_arr.shape != y_lower_arr.shape or y_true_arr.shape != y_upper_arr.shape:
        if verbose >= 2:
            print("Shapes not matching:")
            print("y_true:", y_true_arr.shape)
            print("y_lower:", y_lower_arr.shape)
            print("y_upper:", y_upper_arr.shape)
        raise ValueError(
            "All inputs (y_true, y_lower, y_upper) must have the same shape."
        )

    mask_missing = np.isnan(y_true_arr) | np.isnan(y_lower_arr) | np.isnan(y_upper_arr)

    if np.any(mask_missing):
        if nan_policy == 'raise':
            if verbose >= 2:
                print("Missing values detected and nan_policy='raise'. Raising error.")
            raise ValueError(
                "Missing values detected. To allow missing values, change nan_policy."
            )
        elif nan_policy == 'omit':
            if verbose >= 2:
                print("Missing values detected. Omitting these samples.")
            # omit those samples
            valid_mask = ~mask_missing
            y_true_arr = y_true_arr[valid_mask]
            y_lower_arr = y_lower_arr[valid_mask]
            y_upper_arr = y_upper_arr[valid_mask]
        elif nan_policy == 'propagate':
            if verbose >= 2:
                print("Missing values detected and nan_policy='propagate'."
                      "No special handling. Result may be NaN.")
            # do nothing
      
    coverage_mask = (y_true_arr >= y_lower_arr) & (y_true_arr <= y_upper_arr)
    coverage = np.mean(coverage_mask) if coverage_mask.size > 0 else np.nan

    if verbose >= 4:
        print("Coverage mask (sample):",
              coverage_mask[:10] if coverage_mask.size > 10 else coverage_mask)
    if verbose >= 1:
        print(f"Coverage computed: {coverage:.4f}")

    return coverage

@validate_params({
    "model": [HasMethods(["predict"])],
    "X": ['array-like'],
    "feature_names": ['array-like', None],
    "perturbation": [Interval(Real, 0, 1, closed='both')],
    "plot_type": [
        StrOptions({'hist', 'bar', 'line', 'boxplot', 'box'}), None],
    "interpret": [bool]
   }, 
)
@Appender(dedent( 
"""
perturbation : float, default=0.1
    The amount by which to perturb each feature when calculating sensitivity.
    Should be a small fraction between 0 and 1. For instance, 0.1 indicates
    a 10% increase in the feature value during the sensitivity analysis.

Returns
-------
gofast.api.summary.ResultSummary
    A summary object containing the computed relative sensitivity for each 
    feature in the model:

    - `relative_sensitivity_scores` : pd.DataFrame
        The RS values computed for each feature perturbation.

    - `ranked_features` : pd.DataFrame
        Features ranked from highly sensitive to less sensitive.

    - `relative_sensitivity_by_feature` : pd.DataFrame
        Detailed sensitivity scores grouped by feature.

    - `baseline_predictions` : array-like
        The model's baseline predictions before any perturbations.

Notes
-----
The relative sensitivity (RS) of a feature is computed by perturbing 
the feature value and observing the impact on the model's output. The RS 
for each feature is calculated using the formula:

.. math::
    RS = \\frac{\\Delta \\text{Forecast}}{\\Delta \\text{Input}}

where:
    - :math:`\\Delta \\text{Forecast}` is the change in the predicted output
      after perturbing the feature.
    - :math:`\\Delta \\text{Input}` is the change in the input feature 
      after perturbation.

If the change in input is zero (i.e., no perturbation), the relative 
sensitivity is set to zero [2]_.

The function uses a simple "one-at-a-time" (OAT) sensitivity analysis 
approach. For each feature, the function perturbs its value, computes 
the model's prediction with the perturbed feature, and then calculates
the relative sensitivity score.

If `interpret=True`, the function prints a detailed analysis
of the sensitivity scores, including feature rankings and insights on
which features have the greatest impact on model predictions.

Examples
--------
>>> import numpy as np 
>>> from gofast.metrics_special import relative_sensitivity_score
>>> from gofast.estimators.tree import DTBClassifier 
>>> np.random.seed(123)
>>> X = np.random.randn(100, 19) 
>>> y = np.random.randint(0, 2, size=len(X))
>>> dtb_model = DTBClassifier().fit(X, y) 
>>> 
>>> sensitivity_df = relative_sensitivity_score(
...     dtb_model, X, feature_names=['feature_0', 'feature_14'],
...     perturbation=0.05, plot_type='line', interpret=True
... )
>>> print(sensitivity_df)
    Feature  RS (Relative Sensitivity)
0  feature_0                     0.0000
1 feature_14                     0.1079

References
----------
.. [1] Saltelli, A., Tarantola, S., & Campolongo, F. (2000). Sensitivity analysis 
    as an ingredient of modeling. *Statistical Science, 15*(4), 377-395.
.. [2] Sobol, I. M. (1993). Sensitivity analysis for nonlinear mathematical 
    models. *Mathematical Modelling and Computation*, 4(6), 247-278.    
    
"""
    ), 
    join ='\n', 
)

@Substitution ( **_shared_doc_kwargs) 
@Appender( _shared_params)  
def relative_sensitivity_score(
    model, X, *, 
    perturbation=0.1,
    feature_names=None,  
    plot_type=None, 
    interpret=False, 
):
    """
    Compute the Relative Sensitivity (RS) for each feature in the model 
    predictions.
    
    This function calculates the relative sensitivity of input features in
    a model by perturbing each feature and measuring the resulting changes
    in predictions. This is useful in understanding how sensitive a model is
    to variations in its input features [1]_. It computes the sensitivity score 
    for each feature and optionally visualizes the results with various plots.
    
    See more in :ref:`User Guide <user_guide>`.
    
    """
    # build dataframe if numpy array is passed
    X = build_data_if(
        X, 
        input_name ='feature', 
        raise_exception=True, 
        force=True  
        )
    
    # Ensure feature_names is always a list
    if isinstance(feature_names, str):
        feature_names = [feature_names]
    
    if feature_names is None: 
        feature_names = X.columns.tolist() 
    
    # Validate that the features exist in the DataFrame X
    exist_features(X, feature_names, error="raise")

    # Store baseline predictions
    baseline_predictions = model.predict(X.values)
    
    # Initialize a results dictionary
    sensitivity_results = {
        "Feature": [],
        "RS (Relative Sensitivity)": []
    }
    
    # Iterate over each feature to compute its sensitivity
    for feature in feature_names:
        # Perturb the selected feature
        X_perturbed = X.copy()
        # Increase feature by `perturbation`
        X_perturbed[feature] *= (1 + perturbation)  
        
        # Get predictions with perturbed feature
        perturbed_predictions = model.predict(X_perturbed.values)
        
        # Calculate changes in the predictions and input features
        delta_forecast = np.abs(perturbed_predictions - baseline_predictions)
        delta_input = np.abs(X_perturbed[feature] - X[feature]).mean()
        
        # Compute Relative Sensitivity (RS)
        rs = (
            delta_forecast.mean() / delta_input
            if delta_input != 0 else 0
        )
        # Store results
        sensitivity_results["Feature"].append(feature)
        sensitivity_results["RS (Relative Sensitivity)"].append(round(rs, 4))
    
    # Convert results to DataFrame
    sensitivity_df = pd.DataFrame(sensitivity_results)
    
    # Rank features by RS to highlight the most important ones
    ranked_features = sensitivity_df.sort_values(
        "RS (Relative Sensitivity)", ascending=False
    )
    # Show analysis of results
    if interpret:
        _interpretRS(ranked_features)

    sensitivity_values = sensitivity_df.copy()
    rs_result=sensitivity_df.set_index ('Feature')[
        'RS (Relative Sensitivity)'].to_dict()
    
    result = ResultSummary (
        "RSByFeature",
        pad_keys="auto", 
        relative_sensitivity_scores= sensitivity_df,
        ranked_features= ranked_features,
        relative_sensitivity_by_feature=rs_result, 
        baseline_predictions= baseline_predictions
        
        ).add_results(rs_result)
    
    # Plot results if requested
    if plot_type is not None:
        from .plot.utils import plot_sensitivity 
        plot_sensitivity(
            sensitivity_values, 
            baseline= baseline_predictions, 
            plot_type = plot_type, 
            title = "Sensitivity Analysis: Relative Sensitivity by Feature", 
            xlabel="Feature", 
            ylabel="Relative Sensitivity (RS)", 
            x_ticks_rotation= 90, 
            )

    return result

@Appender(dedent( 
"""
perturbations : list of float, optional
    A list of perturbation amounts to use when calculating sensitivity.
    Each value should be a small fraction between 0 and 1.
    
Returns
-------
gofast.api.summary.ResultSummary
    A summary object containing the computed relative sensitivity for each 
    perturbation performed on each feature in the model:
    
    - `relative_sensitivity_score` : pd.DataFrame
        The RS values computed for each perturbation of each feature.
    
    - `ranked_features` : pd.DataFrame
        The impactful features ranked from highly sensitive to less 
        sensitive.

Example
-------
>>> import numpy as np 
>>> from gofast.metrics_special import relative_sensitivity_scores
>>> from gofast.estimators.tree import DTBClassifier 
>>> np.random.seed(123)
>>> X = np.random.randn(100, 19) 
>>> y = np.random.randint(0, 2, size=len(X))
>>> dtb_model = DTBClassifier().fit(X, y) 
>>> 
>>> sensitivity_df = relative_sensitivity_scores(
...     dtb_model, X, feature_names=['feature_11', 'feature_15'],
...     perturbations=[0.5, 0.8, 0.7], plot_type='line', interpret=True
... )
>>> print(sensitivity_df)

See Also
--------
`gofast.metrics_special.relative_sensitivity_score` :
    Compute the Relative Sensitivity (RS) for each feature in the model 
    predictions.
`gofast.plot.utils.plot_sensitivity` :
    Plot the feature sensitivity values.
    """
    ), 
    join ='\n', 
 )
@Substitution ( **_shared_doc_kwargs)
@Appender( _shared_params, join='\n')
def relative_sensitivity_scores(
    model, X, *, 
    perturbations=None, 
    feature_names=None,  
    plot_type=None, 
    interpret=False, 
): 
    """
    Compute the Relative Sensitivity (RS) for multiple perturbations
    for each feature in the model predictions. 

    See more in :ref:`User Guide <user_guide>`. 
    
    """
    # If perturbations are not provided, 
    # use a default value of 10% perturbation
    if perturbations is None: 
        # Compute only single perturbation by default
        perturbations = [0.10]  
        
    # for singe pertubation use 
    # `relative_sensitivity_score` instead.
    if len(perturbations) ==1: 
        return relative_sensitivity_score( 
            model=model, 
            X=X, 
            feature_names=feature_names, 
            perturbation=perturbations[0], 
            plot_type=plot_type,  
            interpret=interpret  
        )
    
    # Ensure perturbations is iterable
    perturbations = is_iterable(perturbations, transform=True)

    # Initialize a dictionary to store
    # the sensitivity results for each feature
    sensitivities = {}

    for idx, perturbation in enumerate(perturbations): 
        result = relative_sensitivity_score( 
            model=model, 
            X=X, 
            feature_names=feature_names, 
            perturbation=perturbation, 
            plot_type=None,  # No need for plot in this function
            interpret=False  # No need for interpretation 
        )

        # Extract the relative sensitivity results from the result dictionary
        rs_dict = result.relative_sensitivity_by_feature
        
        # Initialize the sensitivities dictionary with the first set of results
        if idx == 0:
            sensitivities = {k: [v] for k, v in rs_dict.items()}
            continue
        
        # Append the results from each perturbation
        # to the sensitivities dictionary
        for feature in sensitivities.keys(): 
            sensitivities[feature].append(rs_dict[feature])
    
    # Once all perturbations are processed,
    # construct a DataFrame to hold the sensitivity values
    sensitivities_values = pd.DataFrame(sensitivities)
    
    # Extract baseline predictions from the result
    baseline_predictions = result.baseline_predictions
    
    # Rank features based on the mean
    # relative sensitivity across all perturbations
    mean_values = sensitivities_values.mean() 
    ranked_features = pd.DataFrame({
        'Feature': mean_values.index,
        'RS (Relative Sensitivity)': mean_values.values
    })
    
    # Rank features by RS to highlight the most important ones
    ranked_features = ranked_features.sort_values(
        "RS (Relative Sensitivity)", ascending=False
    )
    # Interpret the results if requested
    if interpret: 
        _interpretRS(ranked_features)

    # Create a result summary
    result = ResultSummary(
        "RSByPertubation",
        pad_keys="auto", 
        relative_sensitivity_score=sensitivities_values,
        ranked_features=ranked_features,
    ).add_results(sensitivities)  

    # Plot results if requested
    if plot_type is not None:
        from .plot.utils import plot_sensitivity 
        
        plot_sensitivity(
            sensitivities_values, 
            baseline=baseline_predictions, 
            plot_type=plot_type, 
            title="Sensitivity Analysis: Relative Sensitivity by Feature", 
            xlabel= 'Sensitivity Value' if plot_type =='hist' else (
                "Pertubations" if plot_type in ("line", ) else "Features"), 
            ylabel ="Frequency" if plot_type in (
                'hist', ) else "Relative Sensitivity (RS)" , 
            x_ticks_rotation=90, 
            boxplot_showfliers=True
        )

    return result

@validate_params({ 
    "y_pred": ['array-like'], 
    "y_true": ['array-like', None], 
    "sample_weight":['array-like', None], 
    "multioutput": [
        StrOptions({"uniform_average","raw_values"})
        ]
    }
  )
def prediction_stability_score(
    y_pred,
    y_true=None,      
    sample_weight=None,
    multioutput='uniform_average'
    ):
    """
    Calculate the Prediction Stability Score (PSS), which assesses the temporal
    stability of predictions across consecutive time steps [1]_.

    The Prediction Stability Score is defined as:

    .. math::
        \\text{PSS} = \\frac{1}{T - 1} \\sum_{t=1}^{T - 1}
        \\left| \\hat{y}_{t+1} - \\hat{y}_t \\right|

    where:

    - :math:`T` is the number of time steps.
    - :math:`\\hat{y}_t` is the prediction at time :math:`t`.
    
    See more in :ref:`user guide <user_guide>`.

    Parameters
    ----------
    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Predicted values.

    y_true : None
        Not used, present for API consistency by convention.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights. If provided, these are used to weight the differences
        between consecutive predictions.

    multioutput : {'raw_values', 'uniform_average'} or array-like of shape \
            (n_outputs,), default='uniform_average'
        Defines how to aggregate multiple output values.

        - ``'raw_values'`` :
          Returns a full set of scores in case of multioutput input.
        - ``'uniform_average'`` :
          Scores of all outputs are averaged with uniform weight.
        - array-like :
          Weighted average of the output scores.

    Returns
    -------
    score : float or ndarray of floats
        Prediction Stability Score. If ``multioutput`` is ``'raw_values'``,
        then an array of scores is returned. Otherwise, a single float is
        returned.

    Examples
    --------
    >>> from gofast.metrics_special import prediction_stability_score
    >>> import numpy as np
    >>> y_pred = np.array([3, 3.5, 4, 5, 5.5])
    >>> prediction_stability_score(y_pred)
    0.625

    Notes
    -----
    The Prediction Stability Score measures the average absolute difference
    between consecutive predictions. A lower score indicates more stable
    predictions over time.

    See Also
    --------
    twa_score : Time-weighted accuracy for classification tasks.

    References
    ----------
    .. [1] Schoukens, J., & Ljung, L. (2019). Nonlinear System Identification:
           A User-Oriented Roadmap. IEEE Control Systems Magazine, 39(6),
           28-99.
    """
    # Ensure y_pred is a numpy array
    y_pred = check_array(y_pred, ensure_2d=False, dtype=None)

    # For multi-output regression, y_pred can be 2D
    if y_pred.ndim == 1:
        # Reshape to 2D array with one output
        y_pred = y_pred.reshape(-1, 1)

    # Number of samples and outputs
    n_samples, n_outputs = y_pred.shape

    # Compute absolute differences between consecutive predictions
    # diff shape: (n_samples - 1, n_outputs)
    diff = np.abs(y_pred[1:] - y_pred[:-1])

    # Adjust sample_weight for differences
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        if sample_weight.ndim > 1:
            sample_weight = sample_weight.squeeze()
        # Ensure sample_weight has correct length
        if len(sample_weight) != n_samples:
            raise ValueError(
                "sample_weight must have the same length as y_pred"
            )
        # Use weights from t=1 to T-1
        sample_weight = sample_weight[1:]

    # Compute mean absolute difference per output
    if sample_weight is not None:
        # Weighted average over time steps
        score = np.average(
            diff,
            weights=sample_weight[:, np.newaxis],
            axis=0
        )
    else:
        # Unweighted average over time steps
        score = np.mean(diff, axis=0)

    # Handle multioutput parameter
    if multioutput == 'raw_values':
        # Return array of shape (n_outputs,)
        return score
    elif multioutput == 'uniform_average':
        # Average over outputs
        return np.mean(score)
    elif isinstance(multioutput, (list, np.ndarray)):
        # Weighted average over outputs
        output_weights = np.asarray(multioutput)
        if output_weights.shape[0] != n_outputs:
            raise ValueError(
                "output_weights must have the same length as n_outputs"
            )
        return np.average(score, weights=output_weights)
   
@validate_params ({ 
    "y_train": ['array-like'], 
    "y_test": ['array-like', None], 
    "plot_type": [StrOptions({"hist", "box", "hist-box"}), None], 
    "scaling_threshold": [Interval(Real, 0, 1, closed ='both')], 
    "figsize": [tuple, list], 
    "bins": [Interval(Integral, 1, None, closed="left")], 
    "kde": [bool], 
    "color": [str], 
    "font_scale":[Interval( Real, 0, None, closed="left")], 
    "acceptable_relative_mae": [Interval( Real, 0, None, closed="left")], 
    "show_recommendations": [bool]
    }
  )
def analyze_target(
    y_train,
    y_test=None,
    plot_type=None,
    scaling_threshold=1.0,
    figsize=(14, 6),
    bins=30,
    kde=True,
    color='blue',
    font_scale=1.2,
    acceptable_relative_mae=10.0,
    show_recommendations=True
):
    """
    Analyzes the target variable to help determine if scaling is needed.

    Parameters
    ----------
    y_train: array-like
        The training target variable.
    y_test: array-like, optional
        The test target variable. Default is ``None``.
    plot_type : str, optional
        Type of plot to display. Options are ``'hist'``, ``'box'``, or
        ``'hist-box'``. If set to ``None``, no plot is displayed. Default
        is ``None``.
    scaling_threshold : float, optional
        Threshold for recommending scaling based on the range of the target
        variable. Default is ``1.0``.
    figsize : tuple, optional
        Figure size for the plots. Default is ``(14, 6)``.
    bins : int, optional
        Number of bins for the histogram. Default is ``30``.
    kde : bool, optional
        Whether to display the kernel density estimate on the histogram.
        Default is ``True``.
    color : str, optional
        Color for the plots. Default is ``'blue'``.
    font_scale : float, optional
        Font scale for the plots. Default is ``1.2``.
    acceptable_relative_mae : float, optional
        Threshold for acceptable relative MAE percentage. Default is
        ``10.0``.
    show_recommendations : bool, optional
        Whether to print the recommendations based on the statistics of the
        target variable. Default is ``True``.

    Returns
    -------
    stats : dict
        A dictionary containing key statistics of the target variable.

    Notes
    -----
    This function computes key statistics of the target variable, such as
    minimum, maximum, mean, standard deviation, median, and range. It also
    provides a recommendation on whether scaling is needed based on the
    range of the target variable.

    If ``plot_style`` is specified, it visualizes the distribution of the
    target variable using the specified plot type.

    Examples
    --------
    >>> from gofast.metrics_special import analyze_target
    >>> y_train = [1, 2, 3, 4, 5]
    >>> stats = analyze_target(y_train, plot_type='hist-box')

    See Also
    --------
    interpret_loss

    References
    ----------
    .. [1] "Understanding Data Distributions", Data Science Handbook.

    """

    # Convert input to numpy array
    y = np.concatenate([y_train, y_test]) if y_test is not None else y_train
    y = np.array(y)

    # Compute statistics
    stats = {
        'min': np.min(y),
        'max': np.max(y),
        'mean': np.mean(y),
        'std': np.std(y),
        'median': np.median(y),
        'range': np.max(y) - np.min(y)
    }

    # Plotting if plot_style is specified
    if plot_type is not None:
        sns.set(font_scale=font_scale)
        if plot_type == 'hist':
            # Histogram plot
            plt.figure(figsize=figsize)
            sns.histplot(y, bins=bins, kde=kde, color=color)
            plt.title('Histogram of Target Variable')
            plt.xlabel('Target Value')
            plt.ylabel('Frequency')
            plt.show()
        elif plot_type == 'box':
            # Boxplot
            plt.figure(figsize=figsize)
            sns.boxplot(x=y, color=color)
            plt.title('Boxplot of Target Variable')
            plt.xlabel('Target Value')
            plt.show()
        elif plot_type == 'hist-box':
            # Histogram and boxplot side by side
            plt.figure(figsize=figsize)
            plt.subplot(1, 2, 1)
            sns.histplot(y, bins=bins, kde=kde, color=color)
            plt.title('Histogram of Target Variable')
            plt.xlabel('Target Value')
            plt.ylabel('Frequency')
            plt.subplot(1, 2, 2)
            sns.boxplot(x=y, color=color)
            plt.title('Boxplot of Target Variable')
            plt.xlabel('Target Value')
            plt.tight_layout()
            plt.show()
     
    if show_recommendations: 
        key = "Range={} {} threshold={}".format(
            stats["range"],
            '>' if stats['range'] > scaling_threshold else '<=', 
            scaling_threshold
        )
        s={'results': ' * '.join( [f"{k}={v:.4f}" for k, v in stats.items()])}
        stats_report = ReportFactory(
            "Statistics", descriptor ='stat_results').add_mixed_types(
                s, table_width=TW )
        # Recommendation on scaling
        recommendations = ReportFactory(
            "Recommendations", descriptor="recommendations").add_mixed_types(
             {key:("The target variable is within a small"
                   " range. Scaling may not be necessary.")}, 
             table_width=TW
            )
    
        if stats['range'] > scaling_threshold:
            recommendations = ReportFactory("Recommendations").add_mixed_types(
                { key:"The target variable has a wide range. Consider scaling."},
                table_width= TW, 
                )
 
        assemble_reports(stats_report, recommendations, display=True  )
        
    metric_stats = MetricFormatter(
        descriptor="Metric", 
        title="Results",
        **stats ) 
    
    return metric_stats

@validate_params({
    'y_true': ['array-like'],  
    'y_scores': ['array-like', None], 
    'X': ['array-like', None], 
    'estimator': [HasMethods (['fit', 'predict']), None],  
    'cv': ['cv_object', None],  
    'label': [int, None],  
    'scoring_method': [StrOptions({"decision_function", "predict_proba"})], 
    'pos_label': [int, None], 
    'sample_weight': ['array-like', None],  
    'threshold': [Interval(Real, 0, 1, closed='right'), None],  
    'return_scores': [bool],  
    'display_chart': [bool], 
})
def precision_recall_tradeoff(
    y_true, 
    y_scores=None, 
    X=None, 
    estimator=None, 
    *,
    cv=None,
    label=None,
    scoring_method="decision_function",
    pos_label=None, 
    sample_weight=None, 
    threshold=None,
    return_scores=False, 
    display_chart=False, 
    **cv_kwargs
):
    """
    Evaluate and visualize the precision-recall tradeoff for classification models.

    This function computes precision, recall, and F1 scores for binary classification
    tasks. It supports direct score inputs or model predictions. If `display_chart`
    is True, it also plots the precision-recall curve.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_scores : array-like, optional
        Scores or probabilities for the positive class. If None, scores are 
        computed using the `estimator` and `X`.
    X : {array-like, sparse matrix}, shape (n_samples, n_features), optional
        Input features matrix. Required if `y_scores` is None.
    estimator : estimator object implementing 'fit', optional
        A trained classifier instance. Required if `y_scores` is None.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy. 
        Only used when `y_scores` is None.
    label : int or str, optional
        The label of the positive class. Required for multiclass classification.
    scoring_method : {'decision_function', 'predict_proba'}, default='decision_function'
        The method to call on `estimator` to obtain scores.
    pos_label : int or str, optional
        The label of the positive class. Required if `y_true` contains more 
        than two classes.
    sample_weight : array-like, shape (n_samples,), optional
        Optional array of weights for the samples.
    threshold : float, optional
        A threshold to determine the binary classification from scores. If not set,
        the threshold that maximizes the F1 score is used.
    return_scores : bool, default=False
        If True, return the scores array along with the metrics.
    display_chart : bool, default=False
        If True, display the precision-recall curve.

    Returns
    -------
    scores : Bunch
        A `Bunch` object with the precision, recall, F1 score, and optionally the
        scores array if `return_scores` is True.

    Notes
    -----
    The precision-recall curve plots the tradeoff between precision and recall for
    different threshold values. Precision is defined as the number of true positives
    divided by the number of true positives plus the number of false positives. Recall
    is defined as the number of true positives divided by the number of true positives
    plus the number of false negatives [1]_.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.linear_model import LogisticRegression
    >>> from gofast.metrics_special import precision_recall_tradeoff
    >>> X, y = make_classification(n_samples=1000, n_features=20,
    ...                            n_classes=2, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    >>> clf = LogisticRegression()
    >>> clf.fit(X_train, y_train)
    >>> b=precision_recall_tradeoff(y_test, X=X_test, estimator=clf,
    ...                           scoring_method='predict_proba',
    ...                           display_chart=True)
    >>> print(b) 
    
    References
    ----------
    .. [1] Davis, J. & Goadrich, M. (2006). The relationship between Precision-Recall
       and ROC curves. Proceedings of the 23rd International Conference on Machine
       learning (pp. 233-240).
    """
    # Validate scoring method
    scoring_method = normalize_string(
        scoring_method, target_strs= ['decision_function', 'predict_proba'], 
        match_method= "contains", return_target_only= True, 
        error_msg= (f"Invalid scoring method '{scoring_method}'. Expected"
                    " 'decision_function' or 'predict_proba'.")
        )
    
    _assert_binary_classification_args(y_true, label)
    if label is not None: 
        # Ensure binary classification
        y_true = np.asarray(y_true == label, dtype=int)  
    
    # Validate and predict scores if not provided
    if y_scores is None:
        if X is None or estimator is None:
            raise ValueError(
                "When 'y_scores' is None, both 'X' and 'estimator' must be provided.")
        if not isinstance (X, np.ndarray): 
            X = np.asarray(X )
        y_scores = cross_val_predict(
            estimator, X, y_true, cv=cv, method=scoring_method, **cv_kwargs)
    
    #y_true, y_scores = _ensure_y_is_valid(y_true, y_scores, y_numeric =True )
    
    # Handle 'predict_proba' scoring for classifiers
    if scoring_method == 'predict_proba' and y_scores.ndim > 1:
        y_scores = y_scores[:, 1]  # Use probabilities for the positive class
    
    # Calculate precision, recall, and F1 scores
    metrics = MetricFormatter(
        f1_score=f1_score(y_true, y_scores >= (threshold or 0.5)),
        precision_score=precision_score(y_true, y_scores >= (threshold or 0.5)),
        recall_score=recall_score(y_true, y_scores >= (threshold or 0.5))
    )

    # Calculate precision-recall curve
    metrics.precisions, metrics.recalls, metrics.thresholds = precision_recall_curve(
        y_true, y_scores, pos_label=pos_label, sample_weight=sample_weight)
    
    # Display precision-recall chart if requested
    if display_chart:
        display_precision_recall(metrics.precisions, metrics.recalls,
                                       metrics.thresholds)
    
    return y_scores if return_scores else metrics

@validate_params(
    {"precisions": ['array-like'],
     "recalls":['array-like'],
     "thresholds": ['array-like']
     }
  )
def display_precision_recall(precisions, recalls, thresholds):
    """
    Displays a precision-recall tradeoff chart for given precision, recall,
    and threshold values, aiding in the visualization of the tradeoff between
    precision and recall across different threshold settings.

    Parameters
    ----------
    precisions : array-like
        An array of precision scores corresponding to various 
        threshold levels.
    recalls : array-like
        An array of recall scores corresponding to various threshold levels.
    thresholds : array-like
        An array of decision thresholds corresponding to the precision 
        and recall scores.

    Notes
    -----
    The precision-recall tradeoff chart is a useful tool for understanding the
    balance between the true positive rate (recall) and the positive 
    predictive value (precision) at various threshold levels. High precision
    relates to a low false positive rate, while high recall relates to a low 
    false negative rate. Adjusting the decision threshold can control the 
    balance between precision and recall.

    This function plots precision and recall as functions of the decision 
    threshold. The precision array must have one more element than the 
    thresholds array, representing the precision at threshold set to 
    `-inf` (recall at 100%).

    Examples
    --------
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.metrics import precision_recall_curve
    >>> from gofast.metrics_special import display_precision_recall
    >>> X, y = make_classification(n_samples=1000, n_classes=2, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    >>> clf = LogisticRegression()
    >>> clf.fit(X_train, y_train)
    >>> y_scores = clf.decision_function(X_test)
    >>> precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)
    >>> display_precision_recall(precisions, recalls, thresholds)

    This function produces a line plot with the decision thresholds on the x-axis
    and precision and recall scores on the y-axis, showcasing how precision and recall
    vary with different threshold settings.

    References
    ----------
    .. [1] Davis, J. and Goadrich, M., 2006, June. The relationship between 
       Precision-Recall and ROC curves. In Proceedings of the 23rd international
       conference on Machine learning (pp. 233-240).
    """

    # Convert inputs to numpy arrays and validate their dimensions
    precisions = np.asarray(precisions)
    recalls = np.asarray(recalls)
    thresholds = np.asarray(thresholds)

    if not (precisions.ndim == recalls.ndim == thresholds.ndim == 1):
        raise ValueError("All inputs must be 1-dimensional arrays.")

    if not (len(precisions) == len(recalls) == len(thresholds) + 1):
        raise ValueError("Length of precisions and recalls must be equal"
                         " and one more than the length of thresholds.")

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)

    plt.xlabel("Threshold", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.legend(loc="best", fontsize=12)
    plt.title("Precision-Recall Tradeoff", fontsize=16)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

@validate_params({
    'y_true': ['array-like'],  
    'y_scores': ['array-like', None], 
    'X': ['array-like', None], 
    'model': [HasMethods (['fit', 'predict']), None],  
    'cv': ['cv_object', None],  
    'pos_label': [int, None], 
    'sample_weight': ['array-like', None],  
    'return_scores': [bool],  
    'display_chart': [bool], 
})
def roc_tradeoff(
    y_true, 
    y_scores=None, 
    X=None, 
    model=None, 
    *,
    cv=None,
    pos_label=None, 
    sample_weight=None, 
    return_scores=False, 
    display_chart=False, 
    **cv_kwargs
):
    """
    Evaluates and visualizes the Receiver Operating Characteristic (ROC) curve and
    computes the Area Under the Curve (AUC) for classification models. This
    function is flexible, allowing for direct input of scores or computation
    from a provided estimator and features.

    Parameters
    ----------
    y_true : array-like
        True labels of the classification targets.
    y_scores : array-like, optional
        Scores predicted by the classifier. Required unless `X` and `estimator`
        are provided.
    X : {array-like, sparse matrix}, shape (n_samples, n_features), optional
        Input features, required if `y_scores` is not provided.
    model : estimator object implementing 'fit', optional
        The classification estimator. Required if `y_scores` is not provided.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
    pos_label : int or str, optional
        The label of the positive class.
    sample_weight : array-like, optional
        Optional array of weights to be applied to the samples.
    return_scores : bool, default=False
        Whether to return the computed scores along with the ROC metrics.
    display_chart : bool, default=False
        If True, the ROC curve will be plotted.
    cv_kwargs : dict, optional
        Additional parameters to pass to `cross_val_predict`.

    Returns
    -------
    scores : Bunch
        A dictionary-like object with keys `auc_score`, `fpr`, `tpr`, `thresholds`,
        and optionally `y_scores` (if `return_scores` is True). 

        - `auc_score` : float
            The area under the ROC curve.
        - `fpr` : array
            False Positive Rates.
        - `tpr` : array
            True Positive Rates.
        - `thresholds` : array
            Thresholds on the decision function used to compute `fpr` and `tpr`.

    Notes
    -----
    The ROC curve is a graphical plot that illustrates the diagnostic ability of a
    binary classifier system as its discrimination threshold is varied. The curve
    is created by plotting the True Positive Rate (TPR) against the False Positive
    Rate (FPR) at various threshold settings. The AUC score represents the measure
    of separability. It tells how much the model is capable of distinguishing
    between classes.
    
    The ROC curve is defined by:

    .. math::
        \text{TPR} = \frac{TP}{TP + FN}

    .. math::
        \text{FPR} = \frac{FP}{FP + TN}

    Where:
    - TP is the number of true positives
    - FN is the number of false negatives
    - FP is the number of false positives
    - TN is the number of true negatives
    
    Examples
    --------
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> from gofast.metrics_special import roc_tradeoff
    >>> X, y = make_classification(n_samples=1000, n_features=20,
    ...                            n_classes=2, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
    ...                                                     random_state=42)
    >>> clf = RandomForestClassifier(random_state=42)
    >>> clf.fit(X_train, y_train)
    >>> results = roc_tradeoff(y_test, estimator=clf, X=X_test, display_chart=True)
    >>> print(results)
    
    This will display the ROC curve for the RandomForestClassifier on the test data.
    """
    if y_scores is None:
        if X is None or model is None:
            raise ValueError("When 'y_scores' is None, both 'X' and"
                             " 'estimator' must be provided.")
        # Check if the estimator is fitted and predict the scores
        check_is_fitted(model)
        if hasattr(model, "decision_function"):
            y_scores = cross_val_predict(
                model, X, y_true, cv=cv, method="decision_function", **cv_kwargs)
        elif hasattr(model, "predict_proba"):
            y_scores = cross_val_predict(
                model, X, y_true, cv=cv, method="predict_proba", **cv_kwargs)[:, 1]
        elif hasattr ( model, 'predict'): 
            y_scores = cross_val_predict(
                model, X, y_true, cv=cv, **cv_kwargs)
        else:
            raise ValueError("Model must have a 'decision_function',"
                             " 'predict_proba' or 'predict' method .")

    y_true, y_scores = _ensure_y_is_valid(y_true, y_scores, y_numeric=True)

    fpr, tpr, thresholds = roc_curve(
        y_true, y_scores, pos_label=pos_label, sample_weight=sample_weight)
    auc_score = roc_auc_score(y_true, y_scores, sample_weight=sample_weight)

    if display_chart:
        # Example of plotting
        display_roc (fpr, tpr, auc_score)

    scores = MetricFormatter(
        auc_score=auc_score, fpr=fpr, tpr=tpr, thresholds=thresholds, 
        y_scores= y_scores )

    return y_scores if return_scores else scores

@validate_params({
    'fpr': ['array-like'],  
    'tpr': ['array-like'], 
    'auc_score': [Interval(Real, 0, 1, closed ='both')],  
})
def display_roc(fpr, tpr, auc_score, *, title=None, figsize=None):
    """
    Visualize the Receiver Operating Characteristic (ROC) curve along with the
    area under the ROC curve (AUC) for a classification model's performance.

    The ROC curve is a graphical representation of the tradeoff between 
    the true positive rate (TPR) and false positive rate (FPR) across a 
    series of thresholds. The AUC provides a scalar measure of the model's 
    ability to distinguish between classes, with a higher value indicating 
    better performance [1]_.

    Parameters
    ----------
    fpr : array-like
        False positive rates for each threshold.
    tpr : array-like
        True positive rates for each threshold.
    auc_score : float
        The area under the ROC curve.
    title : str, optional
        The title for the plot. Defaults to "Receiver Operating Characteristic (ROC)"
        if not specified.
    figsize : tuple, optional
        Figure size as a tuple (width, height). Defaults to (10, 8).

    Raises
    ------
    ValueError
        If `fpr` and `tpr` are not 1-dimensional arrays or if `auc_score` 
        is not a scalar.

    Notes
    -----
    The ROC curve is defined by:

    .. math::
        \text{TPR} = \frac{TP}{TP + FN}

    .. math::
        \text{FPR} = \frac{FP}{FP + TN}

    Where:
    - TP is the number of true positives
    - FN is the number of false negatives
    - FP is the number of false positives
    - TN is the number of true negatives

    Examples
    --------
    >>> from sklearn.metrics import roc_curve, roc_auc_score
    >>> y_true = [0, 1, 1, 0, 1]
    >>> y_scores = [0.1, 0.4, 0.35, 0.8, 0.7]
    >>> fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    >>> auc_score = roc_auc_score(y_true, y_scores)
    >>> display_roc(fpr, tpr, auc_score)

    References
    ----------
    .. [1] Fawcett, T. (2006). An introduction to ROC analysis. Pattern 
       Recognition Letters,27(8), 861-874.
    """
    # Validate inputs
    fpr, tpr, auc_score = map(np.asarray, [fpr, tpr, auc_score])
    if not (fpr.ndim == tpr.ndim == 1):
        raise ValueError("fpr and tpr must be 1-dimensional arrays.")
    try: 
        auc_score= float(auc_score)
    except: 
        if not np.isscalar(auc_score):
            raise ValueError("auc_score must be a scalar value.")

    plt.figure(figsize=figsize or (10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (area = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(title or "Receiver Operating Characteristic (ROC)", fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

@validate_params({
    'y_true': ['array-like'],  
    'y_pred': ['array-like', None], 
    'X': ['array-like', None], 
    'model': [HasMethods (['fit', 'predict_proba']), None],  
    'cv': ['cv_object', None],  
    'labels': ['array-like', None],  
    'sample_weight': ['array-like', None],  
})
def evaluate_confusion_matrix(
    y_true, 
    y_pred=None, 
    model=None, 
    X=None, 
    *, 
    cv=None, 
    labels=None, 
    sample_weight=None, 
    normalize=False, 
    display=False, 
    cmap='viridis', 
    **cv_kwargs
):
    """
    Evaluates the confusion matrix for a classification model, optionally using
    cross-validation. This function can also normalize and display the confusion
    matrix for visual analysis.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True labels for the classification tasks.
    y_pred : array-like of shape (n_samples,), optional
        Predicted labels. If None, predictions will be made using the
        classifier and X.
    model : object implementing 'fit', optional
        The classifier instance to use if `y_pred` is None.
    X : array-like of shape (n_samples, n_features), optional
        Input features, required if `y_pred` is None.
    cv : int, cross-validation generator or iterable, optional
        Cross-validation strategy, used if `y_pred` is None.
    labels : array-like, optional
        The list of labels to index the matrix. This may be used to reorder
        or select a subset of labels.
    sample_weight : array-like of shape (n_samples,), optional
        Weights of samples.
    normalize : bool, default=False
        If True, the confusion matrix will be normalized.
    display : bool, default=False
        If True, the confusion matrix will be displayed using matplotlib.
    cmap : str or matplotlib Colormap, default='viridis'
        The colormap for displaying the confusion matrix.
    **cv_kwargs : additional keyword arguments
        Additional arguments passed to `cross_val_predict` if `y_pred` is None.

    Returns
    -------
    Bunch
        A Bunch object with the confusion matrix, and optionally the normalized
        confusion matrix if 'normalize' is True.

    Notes
    -----
    The confusion matrix \(C\) is defined as:
    
    .. math:: C_{i, j} = \text{number of observations known to be in group } i
             \text{ and predicted to be in group } j.
    
    For normalized confusion matrix, each element of \(C\) is divided by the sum
    of its row.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from gofast.metrics_special import evaluate_confusion_matrix 
    >>> X, y = make_classification(
    ...    n_samples=1000, n_features=4, n_classes=2, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...    X, y, test_size=0.25, random_state=42)
    >>> clf = RandomForestClassifier(random_state=42)
    >>> clf.fit(X_train, y_train)
    >>> results = evaluate_confusion_matrix(
    ...    y_test, model=clf, X=X_test, display=True, normalize=True)
    >>> print(results)
    This will output a Bunch object containing the confusion matrix and display
    the normalized confusion matrix.

    See Also
    --------
    display_confusion_matrix : Function to visualize the confusion matrix.
    sklearn.metrics.confusion_matrix : The confusion matrix computation.
    """

    if y_pred is None:
        if not (X is not None and model is not None):
            raise ValueError("Provide 'y_pred' or both 'X' and 'classifier'.")
        check_is_fitted(model, 'predict')
        y_pred = cross_val_predict(model, X, y_true, cv=cv, **cv_kwargs)
    
    y_true, y_pred =_ensure_y_is_valid(y_true, y_pred, y_numeric=True )
    cm = confusion_matrix(y_true, y_pred, labels=labels, sample_weight=sample_weight)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        np.fill_diagonal(cm, 0)

    if display:
        display_confusion_matrix(cm, labels=labels, cmap=cmap, normalize=normalize)

    return MetricFormatter(confusion_matrix=cm)

@validate_params ({
    "cm": ['array-like'], 
    "labels": ['array-like', None], 
    })
def display_confusion_matrix(cm, labels=None, cmap='viridis', normalize=False):
    """
    Displays a confusion matrix using matplotlib, providing options for 
    normalization and color mapping.

    Parameters
    ----------
    cm : array-like of shape (n_classes, n_classes)
        The confusion matrix to be displayed.
    labels : list of strings, optional
        The labels for the classes corresponding to the indices of the 
        confusion matrix. If None, labels will be inferred from the indices 
        of the confusion matrix.
    cmap : str or matplotlib.colors.Colormap, default='viridis'
        The colormap for the matrix visualization. Can be any valid colormap 
        recognized by matplotlib.
    normalize : bool, default=False
        If True, the confusion matrix will be normalized before display.
        Normalization is performed by dividing each element by the sum of its row.

    Notes
    -----
    The confusion matrix \(C\) is defined as:
    
    .. math:: C_{i, j} = \text{number of observations known to be in group } i
             \text{ and predicted to be in group } j.
    
    For a normalized confusion matrix, each element of \(C\) is divided by 
    the sum of its row to represent the proportion of predictions.

    Examples
    --------
    >>> from sklearn.metrics import confusion_matrix
    >>> from gofast.metrics_special import display_confusion_matrix
    >>> y_true = [2, 0, 2, 2, 0, 1]
    >>> y_pred = [0, 0, 2, 2, 0, 2]
    >>> cm = confusion_matrix(y_true, y_pred)
    >>> display_confusion_matrix(cm, labels=['Class 0', 'Class 1', 'Class 2'], 
    ...                          normalize=True)
    
    This will display a normalized confusion matrix for the provided true 
    and predicted labels with custom class labels.

    See Also
    --------
    evaluate_confusion_matrix : 
        Function to compute and optionally display a confusion matrix.
    matplotlib.pyplot.imshow : 
        Used to display the confusion matrix as an image.

    """

    # Validate cm is a square matrix
    if not isinstance(cm, np.ndarray) or cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("cm must be a square matrix (2D numpy array).")
    
    # Validate labels, if provided
    if labels is not None:
        if not isinstance(labels, (list, np.ndarray)) or len(labels) != cm.shape[0]:
            raise ValueError(
                "labels must be a list or array of length matching cm dimensions.")

    title = 'Normalized Confusion Matrix' if normalize else 'Confusion Matrix'
    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels)) if labels is not None else np.arange(cm.shape[0])
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

@validate_params ({ 
    'Xp': ['array-like'],
    'Np': ['array-like'], 
    'Sp': ['array-like'], 
    'clip_value': [Interval(Real, 0, None, closed="left"), None], 
    'epsilon': [Interval(Real, 0, 1, closed="neither"), 
                StrOptions({"auto"})], 
    }
  )
def geo_information_value(
        Xp, Np, Sp, *, aggregate=True, epsilon=1e-15, clip_value=None):
    """
    Calculate the Geographic Information Value (Geo-IV) for assessing the influence
    of various factors on landslide occurrences. Geo-IV quantifies the sensitivity 
    and relevance of each factor in predicting spatial events such as landslides.

    This metric implements the Information Value (InV) method, a statistical 
    approach that evaluates the predictive power of spatial parameters and their 
    relationships with landslide occurrences. The method is particularly useful in 
    geosciences for landslide susceptibility mapping and risk assessment [1]_.

    Parameters
    ----------
    Xp : array-like
        An array representing causative factors, with shape (n_samples,).
    Np : array-like
        The number of pixels associated with each causative factor `Xp`,
        with shape (n_samples,).
    Sp : array-like
        The number of landslide pixels in the presence of causative factor `Xp`,
        with shape (n_samples,).
    aggregate : bool, optional, default=True
        If True, returns the overall Geo-IV value by summing up individual
        information values.
    epsilon : float, optional, default=1e-15
        A small value added to avoid division by zero during calculations.
    clip_value : float, optional
        An upper bound for clipping `Np` values to prevent excessively high
        ratios. If None, clipping is only applied at the lower bound (`epsilon`).

    Returns
    -------
    array-like or float
        The calculated Geo-IV for each causative factor, or the overall Geo-IV
        if `aggregate` is True.

    Notes
    -----
    The Geo-IV of each causative factor `X_i` is calculated using the formula:

    .. math::
        I_i = \log\left(\frac{S_i / N_i}{S / N}\right)

    where `S_i` is the landslide pixel number in the presence of `X_i`, `N_i` is
    the number of pixels associated with `X_i`, `S` is the total number of
    landslide pixels, and `N` is the total number of pixels in the study area.

    Positive values of Geo-IV indicate a factor's relevance in the occurrence
    of landslides, while negative values suggest irrelevance. The magnitude
    of the value reflects the strength of the correlation [1]_.

    References
    ----------
    .. [1] Sarkar S, Kanungo D, Patra A (2006). GIS-Based Landslide Susceptibility
           Mapping: A Case Study in the Indian Himalayas. In: Disaster Mitigation
           of Debris Flows, Slope Failures, and Landslides. Universal Academic
           Press, Tokyo, pp. 617-624.
           DOI: https://doi.org/10.1007/s11629-019-5839-3

    Examples
    --------
    >>> import numpy as np 
    >>> from gofast.metrics_special import geo_information_value
    >>> Xp = np.array([1, 2, 3])  # Example causative factors
    >>> Np = np.array([100, 200, 300])  # Pixels associated with each factor
    >>> Sp = np.array([10, 20, 30])  # Landslide pixels for each factor
    >>> geo_iv = geo_information_value(Xp, Np, Sp, aggregate=True)
    >>> print(f"Overall Geographic Information Value: {geo_iv}")
    """
    # Ensure inputs are numpy arrays and convert single numbers to arrays
    
    try: 
        Xp, Np, Sp = map(lambda x: np.asarray(x, dtype=float), [Xp, Np, Sp])
    except ValueError:
        raise ValueError(
            "All inputs (Xp, Np, Sp) must be convertible to float arrays.")
    # Ensure all inputs have compatible shapes
    if not (Xp.shape == Np.shape == Sp.shape):
        raise ValueError(
            "Shapes of Xp, Np, and Sp must be identical. Received shapes"
            f" Xp: {Xp.shape}, Np: {Np.shape}, Sp: {Sp.shape}.")
    
    # Validate inputs are not empty
    if Xp.size == 0 or Np.size == 0 or Sp.size == 0:
        raise ValueError("Inputs Xp, Np, and Sp must not be empty.")
  
    epsilon= check_epsilon(epsilon, Np, base_epsilon = 1e-15 )
    # Apply clipping with epsilon as lower bound and clip_upper_bound as upper bound
    Np = np.clip(Np, epsilon, clip_value if clip_value is not None else np.max(Np))
    total_N = np.sum(Np)
    if total_N <= 0:
        raise ValueError("Sum of Np must be greater than zero after applying epsilon.")
        
    total_S = np.sum(Sp)
    if total_S <= 0:
        raise ValueError("Sum of Sp must be greater than zero.")

    # Calculate the Geo-IV for each element
    Iv = np.log10((Sp / Np) / (total_S / total_N))
    
    # Aggregate the Geo-IV values if requested
    return np.sum(Iv) if aggregate else Iv

@validate_params({
    'y_true': ['array-like'],  
    'y_pred': ['array-like', None], 
    'X': ['array-like', None], 
    'model': [HasMethods (['fit', 'predict']), None],  
    'sample_weight': ['array-like', None],  
    "multioutput": [
        StrOptions({"uniform_average","raw_values"})
        ],
    'clip_value': [Interval(Real, 0, None, closed="left"), None], 
    'epsilon': [Interval(Real, 0, 1, closed="neither"), 
                StrOptions({"auto"})], 
})
def assess_regression_metrics(
    y_true, 
    y_pred=None, 
    X=None, 
    model=None, *,
    sample_weight=None, 
    multioutput='uniform_average',
    force_finite=True, 
    clip_value=0, 
    epsilon=1e-15
    ):
    """
    Assess a comprehensive set of metrics for evaluating the performance of
    regression models. 
    
    Function simplifies the process of model evaluation by computing common 
    regression metrics including Mean Absolute Error (MAE), Mean Squared Error
    (MSE), Root Mean Squared Error (RMSE), :math:`R^2` (coefficient of determination),
    adjusted :math:`R^2`, Mean Squared Logarithmic Error (MSLE), and
    Median Absolute Error (MedAE) [1]_.
    
    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like, optional
        Predicted target values. If None, `model` and `X` must be provided to
        generate predictions.
    X : array-like, optional
        Feature dataset. Required if `y_pred` is None.
    model : estimator object implementing 'fit', optional
        The object to use to fit the data and generate predictions. Required if
        `y_pred` is None.
    sample_weight : array-like, optional
        Sample weights.
    multioutput : str, optional
        Defines aggregating of multiple output scores. Default is
        'uniform_average'.
    force_finite : bool, optional
        Forces the output scores to be finite by clipping. Default is True.
    clip_value : float, optional
        Minimum log value to use for mean_squared_log_error to avoid taking log
        of zero. Default is 0.
    epsilon : float, optional
        Small offset to add to input values to avoid division by zero or taking
        log of zero in metrics calculation. Default is 1e-15.

    Returns
    -------
    scores : Bunch
        A dictionary-like object containing the calculated metrics.

    Notes
    -----
    - The function assumes `y_true` and `y_pred` are already validated for
      their shapes and data types.
    - Adjusted R² is calculated only if `X` is provided and has more than one
      feature, otherwise, it's set to None.
    - Mean Squared Logarithmic Error (MSLE) uses `clip_value` and `epsilon` to
      ensure stability in its calculation.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.datasets import fetch_california_housing
    >>> from gofast.metrics_special import assess_regression_metrics
    >>> X, y = fetch_california_housing(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    >>> model = LinearRegression()
    >>> model.fit(X_train, y_train)
    >>> metrics = assess_regression_metrics(y_test, model=model, X=X_test)
    >>> print(metrics)

    References
    ----------
    .. [1] James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). 
       An Introduction to Statistical Learning. Springer. 
       DOI: 10.1007/978-1-4614-7138-7

    See Also
    --------
    sklearn.metrics.mean_absolute_error : 
        Compute the mean absolute error regression loss.
    sklearn.metrics.mean_squared_error : 
        Compute the mean squared error regression loss.
    sklearn.metrics.r2_score : 
        Compute R², the coefficient of determination.
    sklearn.metrics.mean_squared_log_error :
        Compute the mean squared logarithmic error regression loss.
    sklearn.metrics.median_absolute_error : 
        Compute the median absolute error regression loss.
    sklearn.metrics.make_scorer :
        Make a scorer from a performance metric or loss function.

    Examples of other libraries that provide similar functionality:
    - scikit-learn (sklearn.metrics): Offers a comprehensive set of metrics 
      for evaluating regression models.
    - Statsmodels: Provides classes and functions for the estimation of many 
      different statistical models, as well as for conducting statistical tests,
      and statistical data exploration which includes some regression metrics.
    
    """
    from .metrics import ( 
        adjusted_r2_score , mean_squared_log_error, median_absolute_error
        )
    
    y_true = np.asarray(y_true)
    if y_pred is None:
        if X is None or model is None:
            raise ValueError("When 'y_pred' is None, both 'X' and 'model' must be provided.")
        check_is_fitted(model)
        y_pred = model.predict(X)
    
    # Ensure y_true and y_pred are valid.
    # Assuming _ensure_y_is_valid is defined to validate the inputs
    y_true, y_pred = _ensure_y_is_valid(y_true, y_pred, y_numeric=True)
    # Calculate common regression evaluation scores
    scores = MetricFormatter(
        mean_absolute_error=mean_absolute_error(
            y_true, y_pred, sample_weight=sample_weight, multioutput=multioutput),
        mean_squared_error=mean_squared_error(y_true, y_pred),
        rmse=np.sqrt(mean_squared_error(y_true, y_pred)),
        r2_score=r2_score(
            y_true, y_pred, sample_weight=sample_weight, multioutput= multioutput,
            force_finite=force_finite) ,
        # Adjusted R2 can be calculated if X is provided and has more than one feature
        adjusted_r2_score=adjusted_r2_score(y_true, y_pred, X) if X is not None else None,
        mean_squared_log_error=mean_squared_log_error(
            y_true, y_pred, clip_value=clip_value, epsilon =epsilon
            ) if not np.any (y_true < 0 ) else None,
        median_absolute_error=median_absolute_error(y_true, y_pred),
    )
    return scores

@validate_params({
    'y_true': ['array-like'],  
    'y_pred': ['array-like', None], 
    'X': ['array-like', None], 
    'model': [HasMethods (['fit', 'predict']), None],  
    'average': [
        StrOptions({'micro', 'macro', 'samples', 'weighted', 'binary'}), None], 
    'multi_class': [StrOptions({'raise', 'ovr', 'ovo'})], 
    'normalize': [bool], 
    'sample_weight': ['array-like', None],  
    }
 )
def assess_classifier_metrics(
    y_true, 
    y_pred=None, 
    X=None, 
    model=None, *,
    average="binary", 
    multi_class="raise", 
    normalize=True, 
    sample_weight=None, 
    **scorer_kws
    ):
    """
    Evaluate classification model performance through a comprehensive 
    set of metrics, including accuracy, recall, precision, F1 score, and 
    ROC-AUC score. This function provides a unified approach to model 
    evaluation across binary, multiclass, and multilabel classification 
    scenarios.
    
    Parameters
    ----------
    y_true : array-like
        True labels for classification.
    y_pred : array-like, optional
        Predicted labels or probabilities. Required unless `X` and `model` 
        are provided.
    X : array-like, optional
        Feature set for making predictions using `model`. Required if `y_pred` 
        is None.
    model : object
        A fitted classifier object that has a `predict` or `predict_proba` method.
        Required if `y_pred` is None.
    average : str, optional
        Strategy for averaging binary metrics in multiclass/multilabel 
        scenarios. Default is "binary".
    multi_class : str, optional
        Strategy for treating multiclass data for ROC-AUC score calculation. 
        Default is "raise".
    normalize : bool, optional
        Whether to normalize the accuracy score. Default is True.
    sample_weight : array-like, optional
        Sample weights for metrics calculation.
    **scorer_kws : dict
        Additional keyword arguments for metric functions.

    Returns
    -------
    scores : Bunch
        Dictionary-like object containing evaluation metrics: accuracy, 
        recall, precision, F1 score, and ROC-AUC score.

    Notes
    -----
    - The ROC-AUC score calculation depends on the `predict_proba` method. 
      If it is not available or `multi_class` is "raise", ROC-AUC score 
      calculation will be skipped with a warning.
    - This function automatically validates input shapes and data types 
      for `y_true` and `y_pred`.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from gofast.metrics_special import assess_classifier_metrics
    >>> X, y = make_classification(n_samples=1000, n_features=20,
    ...                            n_classes=2, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ... X, y, test_size=0.25, random_state=42)
    >>> model = RandomForestClassifier(random_state=42)
    >>> model.fit(X_train, y_train)
    >>> metrics = assess_classifier_metrics(
    ...    y_test, model=model, X=X_test,average='macro')
    >>> print(metrics)

    See Also
    --------
    sklearn.metrics.accuracy_score : 
        Compute the accuracy classification score.
    sklearn.metrics.recall_score : 
        Compute the recall, the ability of the classifier to find all positive samples.
    sklearn.metrics.precision_score : 
        Compute the precision, the ability of the classifier not to label as 
        positive a sample that is negative.
    sklearn.metrics.f1_score : 
        Compute the F1 score, the weighted average of precision and recall.
    sklearn.metrics.roc_auc_score : 
        Compute the Area Under the Receiver Operating Characteristic 
        Curve (ROC AUC) from prediction scores.
    sklearn.metrics.classification_report :
        Build a text report showing the main classification metrics.
    
    Examples of other libraries that provide similar functionality:
    - scikit-learn (sklearn.metrics): Offers a wide range of performance metrics
      for classification, including those used here.
    - Yellowbrick: A suite of visual diagnostic tools built on Scikit-Learn 
      and Matplotlib that also offers model evaluation metrics visualization.
      
    References
    ----------
    - Powers, D.M.W. (2011). "Evaluation: From Precision, Recall and F-Score 
      to ROC, Informedness, Markedness & Correlation." Journal of Machine 
      Learning Technologies.
    """
    # Convert y_true to numpy array for consistency
    y_true = np.asarray(y_true)
    
    # Validate model and prediction logic
    if y_pred is None:
        if X is None or model is None:
            raise ValueError("When 'y_pred' is None, both 'X' and 'model' must be provided.")
        # Check if the model is fitted
        check_is_fitted(model)
        y_pred = model.predict(X)
        
    # Ensure y_true and y_pred are valid. 
    y_true, y_pred = _ensure_y_is_valid(y_true, y_pred, y_numeric =True )
    # Calculate evaluation scores
    scores = MetricFormatter(
        accuracy_score=accuracy_score(
            y_true, y_pred, normalize=normalize, sample_weight=sample_weight),
        recall_score=recall_score(
            y_true, y_pred, average=average, sample_weight=sample_weight, 
            **scorer_kws),
        precision_score=precision_score(
            y_true, y_pred, average=average, sample_weight=sample_weight,
            **scorer_kws),
        f1_score=f1_score(
            y_true, y_pred, average=average, sample_weight=sample_weight,
            **scorer_kws),
    )
    # Attempt to compute ROC-AUC score if possible
    try:
        if multi_class != 'raise':
            y_score = model.predict_proba(X)
        else:
            # Use y_pred as y_score if predict_proba is not 
            # available or not applicable
            y_score = y_pred 
        scores.roc_auc_score = roc_auc_score(
            y_true, y_score, average=average, multi_class=multi_class, 
            sample_weight=sample_weight, **scorer_kws)
    except Exception as e:
        scores.roc_auc_score = None
        warnings.warn(f"Unable to compute ROC-AUC score: {e}")

    return scores

@validate_params({
    'y_true': ['array-like'],  
    'y_pred': ['array-like'], 
    'detailed': ['boolean'], 
    'scale_errors': ['boolean'],  
    'epsilon': [Interval(Real, 0, 1, closed="neither"), 
                StrOptions({"auto"})], 
    'zero_division': [StrOptions({'warn', 'ignore'})],  
    }
 )
def mse_flex(
    y_true, y_pred, *, 
    detailed=False, 
    scale_errors=False, 
    epsilon='auto', 
    zero_division='warn'
    ):
    """
    Compute the Mean Squared Error (MSE) with options for detailed error 
    analysis, error scaling, and handling of insignificant errors, 
    encapsulated in a Bunch object. This flexible version allows for more 
    nuanced error analysis by providing additional controls over the 
    computation.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.
    detailed : bool, optional
        If True, return detailed error statistics (minimum, maximum, and 
        standard deviation of squared errors) in addition to MSE.
    scale_errors : bool, optional
        If True, scale squared errors based on the magnitude of y_true.
    epsilon : {'auto'} or float, optional
        Threshold for considering a squared error significant. If 'auto', an
        appropriate epsilon is determined automatically. Must be a positive 
        value or 'auto'.
    zero_division : {'warn', 'ignore'}, optional
        How to handle division by zero during error scaling.

    Returns
    -------
    Bunch
        An object containing the MSE and, optionally, detailed error statistics
        and scaled MSE.

    Notes
    -----
    The MSE is defined as the average of the squared differences between the 
    predicted values and the true values. This implementation allows for the 
    exclusion of insignificant errors (smaller than `epsilon` squared) from 
    the MSE calculation, and for scaling errors relative to the true values, 
    providing a more nuanced error analysis.

    The mathematical expression of the modified MSE when considering `epsilon` 
    is:

    .. math::

        \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} \max(0, (y_{\text{true},i} - 
        y_{\text{pred},i})^2 - \epsilon^2)

    References
    ----------
    .. [1] Gneiting, T., Raftery, A.E. (2007). Strictly Proper Scoring Rules, 
           Prediction, and Estimation. Journal of the American Statistical 
           Association, 102(477), 359-378.

    See Also
    --------
    mae_flex : A flexible version of Mean Absolute Error.
    mean_squared_log_error : Mean Squared Logarithmic Error metric.
    mean_absolute_error : Mean Absolute Error metric.

    Examples
    --------
    >>> from gofast.metrics_special import mse_flex
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> result = mse_flex(y_true, y_pred)
    >>> print(result.score)
    0.375
    >>> result_detailed = mse_flex(y_true, y_pred, detailed=True)
    >>> print(result_detailed.score, result_detailed.min_error)
    0.375 0.0
    """
    # Ensure y_true and y_pred are valid and have the same shape
    y_true, y_pred = _ensure_y_is_valid(y_true, y_pred, y_numeric=True)
    
    # Assuming this function determines a suitable epsilon value
    epsilon= check_epsilon(epsilon, y_pred )
    
    # Validation for epsilon
    if not isinstance(epsilon, (int, float)):
        raise ValueError("epsilon must be 'auto' or a numeric value.")
    
    # Calculate the squared errors
    squared_errors = (y_true - y_pred) ** 2
    
    # Update squared errors based on epsilon, if necessary
    # Only consider errors significant if they are greater
    # than epsilon squared
    significant_squared_errors = np.where(
        squared_errors > epsilon**2, squared_errors, 0)
    
    # Compute the modified MSE using significant squared errors
    mse = np.mean(significant_squared_errors)
    
    result = MetricFormatter(score=mse)
    
    if detailed:
        result.min_error = np.min(significant_squared_errors)
        result.max_error = np.max(significant_squared_errors)
        result.std_error = np.std(significant_squared_errors)
    
    if scale_errors:
        if zero_division == 'warn' and np.any(y_true == 0):
            warnings.warn("Division by zero encountered in scale_errors computation.")
        # Handle zero division according to the zero_division parameter
        with np.errstate(divide='ignore' if zero_division == 'ignore' else 'warn'):
            scaled_squared_errors = np.divide(
                significant_squared_errors, (
                    np.abs(y_true) + epsilon)**2, where=y_true != 0)
        
        result.scaled_score = np.mean(scaled_squared_errors)
        if detailed:
            result.min_scaled_error = np.min(
                scaled_squared_errors, initial=np.inf, where=y_true != 0)
            result.max_scaled_error = np.max(
                scaled_squared_errors, initial=-np.inf, where=y_true != 0)
            result.std_scaled_error = np.std(
                scaled_squared_errors, where=y_true != 0)
    return result

@validate_params({
    'y_true': ['array-like'],  
    'y_pred': ['array-like'], 
    'detailed': ['boolean'], 
    'scale_errors': ['boolean'],  
    'epsilon': [Interval(Real, 0, 1, closed="neither"), 
                StrOptions({"auto"})], 
    'zero_division': [StrOptions({'warn', 'ignore'})],  
    }
 )
def rmse_flex(
    y_true, y_pred, *, 
    detailed=False, 
    scale_errors=False, 
    epsilon='auto', 
    zero_division='warn'
    ):
    """
    Compute the Root Mean Squared Error (RMSE) with options for detailed error
    analysis, error scaling, and handling of insignificant errors, 
    encapsulated in a Bunch object. This flexible version allows for more
    nuanced error analysis by providing additional controls over the
    computation.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.
    detailed : bool, optional
        If True, return detailed error statistics (minimum, maximum, and 
        standard deviation of squared errors) in addition to RMSE.
    scale_errors : bool, optional
        If True, scale squared errors based on the magnitude of y_true.
    epsilon : {'auto'} or float, optional
        Threshold for considering a squared error significant. If 'auto', an
        appropriate epsilon is determined automatically. Must be a positive 
        value or 'auto'.
    zero_division : {'warn', 'ignore'}, optional
        How to handle division by zero during error scaling.

    Returns
    -------
    Bunch
        An object containing the RMSE and, optionally, detailed error statistics
        and scaled RMSE.

    Notes
    -----
    The RMSE is defined as the square root of the average of the squared 
    differences between the predicted values and the true values. This 
    implementation allows for the exclusion of insignificant errors 
    (smaller than `epsilon` squared) from the RMSE calculation, and for 
    scaling errors relative to the true values, providing a more nuanced 
    error analysis [1]_.

    The mathematical expression of the modified RMSE when considering 
    `epsilon` is:

    .. math::

        \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \max(0, (y_{\text{true},i} - 
        y_{\text{pred},i})^2 - \epsilon^2)}

    References
    ----------
    .. [1] Chai, T. and Draxler, R.R. (2014). Root mean square error (RMSE) or mean 
          absolute error (MAE)? – Arguments against avoiding RMSE in the literature.
          Geoscientific Model Development, 7, 1247–1250.

    See Also
    --------
    mae_flex : A flexible version of Mean Absolute Error.
    mse_flex : A flexible version of Mean Squared Error.
    sklearn.metrics.mean_squared_error : Mean Squared Error metric.

    Examples
    --------
    >>> from gofast.metrics_special import rmse_flex
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> result = rmse_flex(y_true, y_pred)
    >>> print(result.score)
    0.612...
    >>> result_detailed = rmse_flex(y_true, y_pred, detailed=True)
    >>> print(result_detailed.score, result_detailed.min_error)
    0.612... 0.0
    """
    # Ensure y_true and y_pred are valid and have the same shape
    y_true, y_pred = _ensure_y_is_valid(y_true, y_pred, y_numeric=True)
    
    # Assuming this function determines a suitable epsilon value
    epsilon= check_epsilon(epsilon, y_pred )
    
    # Validation for epsilon
    if not isinstance(epsilon, (int, float)):
        raise ValueError("epsilon must be 'auto' or a numeric value.")
    
    # Calculate the squared errors
    squared_errors = (y_true - y_pred) ** 2
    
    # Update squared errors based on epsilon, if necessary
    # Only consider errors significant if they are greater than epsilon squared
    significant_squared_errors = np.where(
        squared_errors > epsilon**2, squared_errors, 0)
    
    # Compute the RMSE using significant squared errors
    rmse = np.sqrt(np.mean(significant_squared_errors))
    
    result = MetricFormatter(title="RMSE Results",  score=rmse)
    
    if detailed:
        result.min_error = np.min(significant_squared_errors)
        result.max_error = np.max(significant_squared_errors)
        result.std_error = np.std(significant_squared_errors)
    
    if scale_errors:
        if zero_division == 'warn' and np.any(y_true == 0):
            warnings.warn("Division by zero encountered in scale_errors computation.")
        # Handle zero division according to the zero_division parameter
        with np.errstate(divide='ignore' if zero_division == 'ignore' else 'warn'):
            scaled_squared_errors = np.divide(
                significant_squared_errors, (
                    np.abs(y_true) + epsilon)**2, where=y_true != 0)
        
        result.scaled_RMSE = np.sqrt(np.mean(scaled_squared_errors))
        if detailed:
            result.min_scaled_error = np.min(
                scaled_squared_errors, initial=np.inf, where=y_true != 0)
            result.max_scaled_error = np.max(
                scaled_squared_errors, initial=-np.inf, where=y_true != 0)
            result.std_scaled_error = np.std(
                scaled_squared_errors, where=y_true != 0)
    
    return result

@validate_params({
    'y_true': ['array-like'],  
    'y_pred': ['array-like'], 
    'adjust_for_n': ['boolean'],  
    'epsilon': [Interval(Real, 0, 1, closed="neither"), StrOptions({"auto"})], 
    'n_predictions': [Interval(Integral, 1, None, closed='left'), None],  
    }
 )
def r2_flex(
    y_true, y_pred, *, 
    epsilon=1e-15, 
    adjust_for_n=False,
    n_predictors=None  
    ):
    """
    Compute the R-squared (Coefficient of Determination) with options to handle
    edge cases and to adjust for the number of predictors. 
    
    Funtion offers enhanced flexibility by allowing for a minimum variance 
    threshold (epsilon) and providing an option to calculate an adjusted 
    R-squared value, making it more robust against overfitting when using 
    models with multiple predictors.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.
    epsilon : float, default=1E-15
        A small value to prevent division by zero in the total sum of squares
        calculation, ensuring numerical stability. This value is used as the
        minimum allowable variance.
    adjust_for_n : bool, default=False
        If True, the adjusted R-squared value is computed to account for the
        number of predictors in the model. This can provide a more accurate
        measure of the model's explanatory power when comparing models with
        different numbers of predictors.
    n_predictors : int, optional
        The number of predictors used in the model. This parameter is required
        if `adjust_for_n` is True to properly calculate the adjusted R-squared.

    Returns
    -------
    Bunch
        An object containing the R-squared value and, if `adjust_for_n` is True,
        the adjusted R-squared value.

    Notes
    -----
    The standard R-squared is calculated as 1 minus the ratio of the residual
    sum of squares (RSS) to the total sum of squares (TSS). The adjusted
    R-squared additionally considers the number of predictors in the model to
    adjust for the penalty of model complexity.

    Adjusted R-squared is especially useful in multiple regression models to
    prevent the R-squared value from artificially inflating when additional
    predictors are added to the model.

    R-squared is defined as:

    .. math::
        R^2 = 1 - \frac{SS_{res}}{SS_{tot}}

    Where:

    - \(SS_{res}\) is the sum of squares of residuals.
    - \(SS_{tot}\) is the total sum of squares.
    
    Adjusted R-squared is calculated as:

    .. math::
        \text{Adjusted } R^2 = 1 - (1-R^2)\frac{n-1}{n-p-1}

    Where:

    - \(n\) is the number of samples.
    - \(p\) is the number of predictors.

    The `epsilon` parameter helps to ensure \(SS_{tot}\) is not zero by providing
    a minimum value, thereby preventing undefined R-squared values.

    References
    ----------
    .. [1] James, G., Witten, D., Hastie, T., and Tibshirani, R. (2013). 
           An Introduction to Statistical Learning. New York: Springer.

    See Also
    --------
    mae_flex : Mean Absolute Error with flexibility.
    mse_flex : Mean Squared Error with flexibility.
    rmse_flex : Root Mean Squared Error with flexibility.

    Examples
    --------
    >>> from gofast.metrics_special import r2_flex
    >>> y_true = np.array([3, 5, 2, 7])
    >>> y_pred = np.array([2.5, 0.5, 2, 8])
    >>> result = r2_flex(y_true, y_pred)
    >>> print(result.score)
    -0.4576271186440677...
    >>> result_adjusted = r2_flex(y_true, y_pred, adjust_for_n=True, n_predictors=1)
    >>> print(result_adjusted.adjusted_score)
    -1.1864406779661016...
    """
    # Ensure y_true and y_pred are valid and have the same shape
    y_true, y_pred = _ensure_y_is_valid(y_true, y_pred, y_numeric=True)
    
    # Validation for epsilon
    epsilon= check_epsilon(epsilon, y_pred )
  
    # Calculate R-squared
    ssr = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ssr / max (sst, epsilon)) # Prevent division by zero
    
    result = MetricFormatter(title="R2 Results", score=r2)
    
    if n_predictors is None: 
        n_predictors = max (1, epsilon)
        
    # Check if n_predictors is a numeric type, including NumPy numeric types
    if isinstance(n_predictors, (int, float, np.integer, np.floating)):
        n_predictors = int(n_predictors)  # Convert to int (handles float and np.floating)
    else:
        raise ValueError("n_predictors must be a numeric value.")

    # Adjusted R-squared calculation if requested 
    # and if n_predictors is specified
    if adjust_for_n:
        n = len(y_true)
        p = n_predictors  # Use the specified number of predictors
        if n == 1 or n - p - 1 == 0:
            warnings.warn ("Adjustment for n is not meaningful with only"
                           " one sample or one predictor.")
        elif n <= p:
            warnings.warn("Cannot compute adjusted R2 with number of"
                          " samples <= number of predictors.")
        else:
            result.adjusted_score = 1 - (1 - r2) * (n - 1) / (n - p - 1)
            
    elif adjust_for_n and n_predictors is None:
        warnings.warn("Number of predictors must be specified to adjust R2.")
    
    return result

def _assert_binary_classification_args(y, target_class=None):
    """
    Ensures that the arguments passed for evaluating a binary classification 
    are valid. It checks if 'y' is binary and if the 'target_class' specified 
    exists within 'y'.

    Parameters
    ----------
    y : array-like
        Target labels. Expected to be binary for precision-recall metrics.
    target_class : int, optional
        The class of interest for which to evaluate metrics. If 'y' is binary,
        'target_class' is optional. If specified, it must exist in 'y'.

    Raises
    ------
    ValueError
        - If 'y' is not binary and 'target_class' is not specified.
        - If 'target_class' is specified but does not exist in 'y'.
    TypeError
        - If 'target_class' cannot be converted to an integer.
    """
    unique_labels_= np.unique(y)

    # Validate binary nature of 'y'
    if len(unique_labels_) != 2:
        if target_class is None:
            raise ValueError(
                "Precision-recall metrics require binary classification, but "
                f"{len(unique_labels_)} unique labels were found. Please specify "
                "'target_class' for multiclass data."
            )

    # Validate 'target_class'
    if target_class is not None:
        try:
            target_class = int(target_class)
        except (ValueError, TypeError) as e:
            raise TypeError(
                "Expected 'target_class' to be an integer,"
                f" got {type(target_class).__name__}: {e}"
            )

        if target_class not in unique_labels_:
            raise ValueError(
                f"Specified 'target_class' ({target_class}) does"
                " not exist in the target labels."
            )

@validate_params({
    'y_true': ['array-like'],  
    'y_pred': ['array-like'], 
    'detailed': ['boolean'], 
    'scale_errors': ['boolean'],  
    'epsilon': [Interval(Real, 0, 1, closed="neither"), StrOptions({"auto"})], 
    'zero_division': [StrOptions({'warn', 'ignore'})],  
    }
 )
def mae_flex(
    y_true, y_pred, *, 
    detailed=False, 
    scale_errors=False, 
    epsilon='auto', 
    zero_division='warn'
    ):
    """
    Compute the Mean Absolute Error (MAE) with options for detailed error 
    analysis, error scaling, and handling of insignificant errors, encapsulated
    in a Bunch object.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.
    detailed : bool, optional
        If True, return detailed error statistics in addition to MAE.
    scale_errors : bool, optional
        If True, scale errors based on the magnitude of y_true.
    epsilon : {'auto'} or float, optional
        Threshold for considering an error significant. If 'auto', an
        appropriate epsilon is determined automatically. Must be a positive 
        value or 'auto'.
    zero_division : {'warn', 'ignore'}, optional
        How to handle division by zero during error scaling.

    Returns
    -------
    Bunch
        An object containing the MAE and, optionally, detailed error statistics
        and scaled MAE.

    Notes
    -----
    The MAE is defined as the average of the absolute differences between the 
    predicted values and the true values. It provides a straightforward measure 
    of prediction accuracy for regression models. This implementation allows for 
    the exclusion of insignificant errors (smaller than `epsilon`) from the MAE 
    calculation, and for scaling errors relative to the true values, providing
    a more nuanced error analysis [1]_.

    The mathematical expression of the modified MAE when considering `epsilon` is:

    .. math::

        \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} \max(0, |y_{\text{true},i} - 
        y_{\text{pred},i}| - \epsilon)

    References
    ----------
    .. [1] Hyndman, R.J., Koehler, A.B. (2006). Another look at measures of forecast 
          accuracy. International Journal of Forecasting, 22(4), 679-688.
    
    See Also
    --------
    mean_squared_error : Mean Squared Error metric.
    mean_absolute_percentage_error : Mean Absolute Percentage Error metric.

    Examples
    --------
    >>> from gofast.metrics_special import mae_flex
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> result = mae_flex(y_true, y_pred)
    >>> print(result.MAE)
    0.5
    >>> result_detailed = mae_flex(y_true, y_pred, detailed=True)
    >>> print(result_detailed.MAE, result_detailed.min_error)
    0.5 0.0
    """
    # Ensure y_true and y_pred are valid and have the same shape
    y_true, y_pred = _ensure_y_is_valid(y_true, y_pred, y_numeric=True)
    
    # Function determines a suitable 
    # epsilon value if set to ``auto``.
    epsilon= check_epsilon(epsilon, y_pred )

    # Validation for epsilon
    if not isinstance(epsilon, (int, float)):
        raise ValueError("epsilon must be 'auto' or a numeric value.")
    
    # Calculate the absolute errors
    errors = np.abs(y_true - y_pred)
    
    # Update errors based on epsilon, if necessary
    # Only consider errors significant if they are greater than epsilon
    significant_errors = np.where(errors > epsilon, errors, 0)
    
    # Compute the modified MAE using significant errors
    mae = np.mean(significant_errors)
    
    result = MetricFormatter(score=mae)
    
    if detailed:
        result.min_error = np.min(significant_errors)
        result.max_error = np.max(significant_errors)
        result.std_error = np.std(significant_errors)
    
    if scale_errors:
        if zero_division == 'warn' and np.any(y_true == 0):
            # Implement warning for zero division if needed
            warnings.warn("Division by zero encountered in scale_errors computation.")
        # Handle zero division according to the zero_division parameter
        with np.errstate(divide='ignore' if zero_division == 'ignore' else 'warn'):
            scaled_errors = np.divide(
                significant_errors, np.abs(y_true) + epsilon, where=y_true != 0)
        
        result.scaled_MAE = np.mean(scaled_errors)
        if detailed:
            result.min_scaled_error = np.min(
                scaled_errors, initial=np.inf, where=y_true != 0)
            result.max_scaled_error = np.max(
                scaled_errors, initial=-np.inf, where=y_true != 0)
            result.std_scaled_error = np.std(
                scaled_errors, where=y_true != 0)
    
    return result

@validate_params({
    'data': ['array-like'],  
    'sample_weight': ['array-like', None], 
    'nan_policy': [StrOptions({'omit', 'propagate', 'raise'})], 
    'detailed': ['boolean'], 
    'axis': [Interval(Integral, 0, 1, closed="both"), None], 
    'scale_errors': ['boolean'],  
    'epsilon': [Interval(Real, 0, 1, closed="neither"), StrOptions({"auto"})], 
    'zero_division': [StrOptions({'warn', 'ignore'})],  
    }
 )
def madev_flex(
    data, *, 
    sample_weight=None,
    nan_policy='propagate',
    epsilon=1e-8, 
    axis=None, 
    detailed=False, 
    scale_errors=False, 
    zero_division='warn'
):
    """
    Compute the flexible Mean Absolute Deviation (MADev) with options for
    handling NaN values, scaling errors, and providing detailed statistics.
    This version allows for more nuanced analysis and robust handling of
    data, encapsulated in a Bunch object.

    Parameters
    ----------
    data : array-like
        The dataset for which the Mean Absolute Deviation (MADev) is to be 
        computed. This can be a list, numpy array, or a pandas DataFrame/Series.
        The data should represent a sample or population from which to measure
        central dispersion.

    sample_weight : array-like of shape (n_samples,), optional
        Weights associated with the elements of `data`, representing the importance
        or frequency of each data point. Useful in weighted statistical analyses
        where some observations are considered more significant than others. 
        Default is None, where each data point has equal weight.

    nan_policy : str, {'omit', 'propagate', 'raise'}, defaut = 'propagate'
        If set to 'omit', the function will ignore NaN (Not a Number) values 
        in the `data`. This is particularly useful when dealing with datasets
        that have missing values, ensuring they do not affect the computation 
        of the MADev. Default is ``propagate`` which returns the input data 
        without changes. 'raise' throws an error if NaNs are detected.

    epsilon : float, optional
        A small positive constant added to the denominator to prevent division
        by zero when calculating scaled deviations or when the mean of 
        `data` is zero. Default value is ``1e-8``, which is sufficiently small
        to not distort the calculations.

    axis : int, optional
        The axis along which to compute the mean absolute deviation. For a 2D 
        array, ``axis=0`` computes the deviation column-wise, while ``axis=1``
        row-wise. Default is None, flattening the `data` and computing the 
        deviation for the entire dataset.

    detailed : bool, optional
        If True, the function returns detailed statistics including the minimum,
        maximum, and standard deviation of the absolute deviations. This provides
        a deeper insight into the variability and dispersion of the dataset.
        Default is False.

    scale_errors : bool, optional
        If True, the absolute deviations are scaled by the mean value of `data`,
        resulting in a relative measure of dispersion. This is useful for comparing
        the variability of datasets with different scales. Default is False.

    zero_division : {'warn', 'ignore'}, optional
        Specifies how to handle situations where division by zero might occur
        during the computation. 'warn' will issue a warning, and 'ignore' will 
        suppress the warning and proceed with the calculation. 
        Default is ``'warn'``.

    Returns
    -------
    Bunch
        A Bunch object containing the computed Mean Absolute Deviation (MADev) 
        and,optionally, detailed and scaled error statistics based on the 
        specified parameters. This structured output allows for easy access 
        to the results of the computation.

    Notes
    -----
    The Mean Absolute Deviation (MADev) is a robust measure of variability that
    quantifies the average absolute dispersion of a dataset around its mean.
    Unlike variance or standard deviation, MADev is not influenced as heavily
    by extreme values, making it a reliable measure of spread, especially in
    datasets with outliers [1]_.

    This flexible implementation of MADev allows for detailed and scaled analysis,
    handling of missing values, and application to multi-dimensional datasets,
    enhancing its utility for a broad range of statistical and data analysis tasks.

    The Mean Absolute Deviation is defined as:

    .. math:: \text{MADev} = \frac{1}{n} \sum_{i=1}^{n} |x_i - \bar{x}|

    where :math:`\bar{x}` is the mean of the data and :math:`n` is the number
    of observations.

    Examples
    --------
    >>> from gofast.metrics_special import madev_flex
    >>> data = [1, 2, 3, 4, 5]
    >>> result=madev_flex(data)
    >>> result.score
    1.2

    >>> result= madev_flex(data, detailed=True)
    >>> print(result)
    MADev         : 1.2
    max_deviation : 2.0
    min_deviation : 0.0
    std_deviation : 0.7483314773547883

    >>> result=madev_flex(data, scale_errors=True)
    >>> result.scaled_MADev 
    0.4

    References
    ----------
    .. [1] Lehmann, E. L. (1998). Nonparametrics: Statistical Methods Based on Ranks.
           Prentice Hall.

    See Also
    --------
    numpy.mean : Compute the arithmetic mean along the specified axis.
    numpy.nanmean : Compute the arithmetic mean along the specified axis,
                    ignoring NaNs.
    """
    data = np.asarray(data)
    if not _is_numeric_dtype(data): 
        raise TypeError("MADev computation expects `data` to be numeric.")
        
    # Handle NaN values by filtering them out
    data, *opt_sample_weight= validate_nan_policy(
        nan_policy, data, sample_weights= sample_weight ) 
    sample_weight = opt_sample_weight[0] if opt_sample_weight else sample_weight
    
    # Calculate the mean, taking into account whether NaN values should be ignored
    mean = np.nanmean(data, axis=axis) if nan_policy=='omit' else np.mean(data, axis=axis)
    
    # Calculate absolute deviation from the mean
    abs_deviation = np.abs(data - mean)
    
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        # Compute weighted absolute deviation
        weighted_abs_deviation = abs_deviation * sample_weight
        sum_weights = np.sum(sample_weight, axis=axis)
        
        # Ensure that division by zero is handled according to 
        # zero_division parameter
        with np.errstate(divide=zero_division, invalid=zero_division):
            madev = np.sum(weighted_abs_deviation, axis=axis) / np.maximum(
                sum_weights, epsilon)
    else:
        madev = np.mean(abs_deviation, axis=axis)
    
    result = MetricFormatter(title="MADev Results", score=madev)
    
    if detailed:
        # Add detailed statistics if requested
        result.min_deviation = np.min(abs_deviation, axis=axis)
        result.max_deviation = np.max(abs_deviation, axis=axis)
        result.std_deviation = np.std(abs_deviation, axis=axis)
    
    if scale_errors:
        # Scale errors relative to the mean, with protection
        # against division by zero
        scaled_abs_deviation = abs_deviation / np.maximum(mean, epsilon)
        scaled_madev = np.mean(scaled_abs_deviation, axis=axis)
        result.scaled_score = scaled_madev
        
        if detailed:
            # Include detailed statistics for scaled deviations if requested
            result.min_scaled_deviation = np.min(scaled_abs_deviation, axis=axis)
            result.max_scaled_deviation = np.max(scaled_abs_deviation, axis=axis)
            result.std_scaled_deviation = np.std(scaled_abs_deviation, axis=axis)
    
    return result

@validate_params({
    'y_true': ['array-like'],  
    'y_pred': ['array-like'],  
    'labels': ['array-like', None], 
    'pos_label': [Interval(Integral, 0, None, closed='left')], 
    'average': [StrOptions({'binary', 'micro', 'macro', 'samples', 'weighted'})], 
    'detailed': ['boolean'], 
    'scale_errors': ['boolean'],  
    'epsilon': [Interval(Real, 0, 1, closed="neither"), StrOptions({"auto"})], 
    'zero_division': [StrOptions({'warn', 'ignore'}), int], 
    'multioutput': [StrOptions({'average_uniform', 'raw_values'})]
   }
 )
def jaccard_flex(
    y_true, y_pred, *, 
    labels=None,
    pos_label=1, 
    average='binary', 
    detailed=False, 
    scale_errors=False, 
    epsilon='auto', 
    zero_division='warn', 
    multioutput='average_uniform'
):
    """
    Compute the Jaccard Similarity Coefficient, extending functionality to 
    support detailed output and handling of multi-class/multi-label cases.

    This metric measures the similarity and diversity between sample sets, 
    making it useful in various fields including ecology, gene sequencing 
    analysis, and machine learning for clustering accuracy evaluation [1]_.

    Parameters
    ----------
    y_true : array-like
        True labels or binary indicators of membership.
    y_pred : array-like
        Predicted labels or binary indicators of membership.
    labels : array-like, optional
        The set of labels to include when `average` is not 'binary', 
        and their order if `average` is None.
    pos_label : int or str, default=1
        The label of the positive class.
    average : {'binary', 'micro', 'macro', 'samples', 'weighted'}, default='binary'
        Determines the type of averaging performed on the data.
    detailed : bool, default=False
        If True, return detailed metrics in the output.
    scale_errors : bool, default=False
        If True, scale intersection and union by the mean value of `y_true`.
    epsilon : float or 'auto', default='auto'
        A small value to avoid division by zero. Automatically determined if 'auto'.
    zero_division : str or int, default='warn'
        Sets the value to return when there is a zero division.
    multioutput : {'average_uniform', 'raw_values'}, default='average_uniform'
        Defines how to aggregate scores for multiple outputs.

    Returns
    -------
    Bunch
        A bunch containing the Jaccard score, and if `detailed` is True, additional
        metrics like intersection and union sizes, potentially scaled by `scale_errors`.

    Notes
    -----
    The Jaccard Similarity Coefficient (J) is defined as:

    .. math::
        J = \\frac{|y_{\\text{true}} \\cap y_{\\text{pred}}|}{|y_{\\text{true}} \\cup y_{\\text{pred}}|}

    This function supports both binary and multi-class/multi-label tasks by binarizing 
    labels in a multi-class/multi-label setting [2]_.

    Examples
    --------
    >>> from gofast.metrics_special import jaccard_flex 
    >>> y_true = [0, 1, 2, 1, 2, 0]
    >>> y_pred = [0, 2, 1, 0, 0, 1]
    >>> jaccard_flex(y_true, y_pred, average='macro')
    Bunch(score=0.333...)

    References
    ----------
    .. [1] Scikit-learn documentation on Jaccard score:
          https://scikit-learn.org/stable/modules/generated/sklearn.metrics.jaccard_score.html
    .. [2] Wikipedia entry for the Jaccard index:
           https://en.wikipedia.org/wiki/Jaccard_index

    See Also
    --------
    sklearn.metrics.jaccard_score : Jaccard similarity coefficient score.
    
    """
    # Ensure inputs are numpy arrays for consistency
    y_true, y_pred = _ensure_y_is_valid(
        y_true, y_pred, y_numeric=True, allow_nan=True, multi_output =True
    )
    # Initialize Bunch object to store results
    b = MetricFormatter(title="Jaccard Results" )
    # Determine epsilon dynamically if required
    epsilon = check_epsilon(epsilon, y_pred, scale_factor= np.finfo(float).eps)

    # Handle multi-class/multi-label cases
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    
    # Binarize labels in a multi-class/multi-label setting
    if average != 'binary' or len(labels) > 2:
        y_true = label_binarize(y_true, classes=labels)
        y_pred = label_binarize(y_pred, classes=labels)
    
    # Calculate Jaccard score
    b.score = jaccard_score(y_true, y_pred, labels=labels, pos_label=pos_label,
                          average=average, zero_division=zero_division)
    
    # Store basic result
    # Calculate detailed statistics if requested
    if detailed:
        intersect = np.logical_and(y_true, y_pred)
        union = np.logical_or(y_true, y_pred)
        b.intersection = np.sum(intersect, axis=0)
        b.union = np.sum(union, axis=0)
        if scale_errors:
            # Scale the intersection and union by the mean value of y_true
            mean_val = np.mean(y_true, axis=0) + epsilon  # Avoid division by zero
            b.scaled_intersection = b.intersection / mean_val
            b.scaled_union = b.union / mean_val
    
    # Handle multioutput
    multioutput = validate_multioutput(multioutput)
    if multioutput == 'raw_values':
        b.raw_scores = jaccard_score(
            y_true, y_pred, labels=labels, pos_label=pos_label, 
            average=None, zero_division=zero_division)
    return b

# -- private utilities---

def _interpretRS (ranked_features): 
    """ An isolate part of Relative sensitivity Interpretation  """
    analyses={}
    for _, row in ranked_features.iterrows():
        
        feature = row["Feature"]
        rs_score = row["RS (Relative Sensitivity)"]
        
        if rs_score > 1:
            analyses [f"{feature}"]= ( 
                f"rs = {rs_score:.2f} -> Highly sensitive >>> Small"
                " changes in {feature} result in significant changes"
                " in model predictions."
                )

        elif rs_score > 0.5:
            analyses [f"{feature}"]= ( 
                f"rs = {rs_score:.2f} -> Moderately sensitive >>> Variations"
                " in {feature} cause noticeable but not drastic changes." 
                )
        else:
            analyses [f"{feature}"]= ( 
                f"rs = {rs_score:.2f} -> Low sensitivity >>> Changes"
                f" in {feature} have minimal effect on predictions."
                )
  
    description = DescriptionFormatter(
        title="Relative Sensitivity (RS) Analyses",
        content=analyses,
        header_cols = ("Feature", "Interpretation")
        ) 
    print(description)
    
    # Optional: Highlight possible outliers or unexpected results
    high_sensitivity = ranked_features[
        ranked_features["RS (Relative Sensitivity)"] > 1
    ]
   
    if not high_sensitivity.empty:
        dict_hs=high_sensitivity[["Feature", "RS (Relative Sensitivity)"]
                                 ].set_index ('Feature')[
            'RS (Relative Sensitivity)'].to_dict()
    
        hsummary = DescriptionFormatter(
            title ="Highly Sensitive Features (RS > 1)", 
            content= dict_hs, 
            header_cols = ("Feature", "Relative Sensitivity score ")
            )
        print()
        print(hsummary)
  
    # Optional: Provide a summary of the general trends
    if (
        len(ranked_features) > 1
        and (ranked_features['RS (Relative Sensitivity)'
                             ].iloc[0].round(4).all() != 0)
    ):
        most_impactful = ranked_features.iloc[0]
        
        key_insights = (
            f"The most impactful feature is '{most_impactful['Feature']}' with "
            f"an RS of {most_impactful['RS (Relative Sensitivity)']:.2f}."
        )
    else:
        key_insights= "No significant sensitivity observed across features."
     

    impact_doc = DescriptionFormatter ( 
        content = key_insights, title ="Impactful Feature", 
        )
    print() 
    print(impact_doc)

