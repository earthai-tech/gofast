# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Metrics are measures of quantitative assessment commonly used for 
estimating, comparing, and tracking performance or production. Generally,
a group of metrics will typically be used to build a dashboard that
management or analysts review on a regular basis to maintain performance
assessments, opinions, and business strategies.
"""
from __future__ import annotations 
import copy
import warnings  
import numpy as np 
from scipy.stats import spearmanr

from sklearn import metrics 
from sklearn.metrics import (  
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_curve, 
    roc_auc_score,
    accuracy_score, 
    confusion_matrix, # as cfsmx ,
    classification_report, 
    mean_squared_error, 
    log_loss
    )

from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import cross_val_predict 
from sklearn.preprocessing import label_binarize

from ._docstring import DocstringComponents,_core_docs
from ._gofastlog import gofastlog
from ._typing import _F, List, Optional, ArrayLike , NDArray 
from .exceptions import LearningError 
from .tools.box import Boxspace 
from .tools.coreutils import normalize_string 
from .tools.mathex import determine_epsilon, calculate_binary_iv
from .tools.validator import get_estimator_name, _is_numeric_dtype
from .tools.validator import check_consistent_length, check_y  
from .tools.validator import check_classification_targets 

from .tools.coreutils import is_iterable, _assert_all_types 

_logger = gofastlog().get_gofast_logger(__name__)

_SCORERS = {
    "classification_report": classification_report,
    'precision_recall': precision_recall_curve,
    "confusion_matrix": confusion_matrix,
    'precision': precision_score,
    "accuracy": accuracy_score,
    "mse": mean_squared_error,
    "recall": recall_score,
    'auc': roc_auc_score,
    'roc': roc_curve,
    'f1': f1_score,
}
__all__=[
    "precision_recall_tradeoff",
    "roc_curve_",
    "confusion_matrix_", 
    "get_eval_scores", 
    "mean_squared_log_error",
    "balanced_accuracy",
    "information_value", 
    "mean_absolute_error_b",
    "mean_squared_error", 
    "root_mean_squared_error",
    "r_squared", 
    "mean_absolute_percentage_error", 
    "explained_variance_score", 
    "median_absolute_error",
    "max_error",
    "mean_squared_log_error",
    "mean_poisson_deviance", 
    "mean_gamma_deviance",
    "mean_absolute_deviation", 
    "dice_similarity_coeff", 
    "gini_coeff",
    "hamming_loss", 
    "fowlkes_mallows_index",
    "rmse_log_error", 
    "mean_percentage_error",
    "percentage_bias", 
    "spearmans_rank_correlation",
    "precision_at_k", 
    "ndcg_at_k", 
    "mean_reciprocal_rank", 
    "average_precision",
    "jaccard_similarity_coeff", 
    "geo_iv", 
    
    ]

#----add metrics docs 
_metrics_params =dict (
    label="""
label: float, int 
    Specific class to evaluate the tradeoff of precision 
    and recall. If `y` is already a binary classifer (0 & 1), `label` 
    does need to specify.     
    """, 
    method="""
method: str
    Method to get scores from each instance in the trainset. 
    Could be a ``decison_funcion`` or ``predict_proba``. When using the  
    scikit-Learn classifier, it generally has one of the method. 
    Default is ``decision_function``.   
    """, 
    tradeoff="""
tradeoff: float
    check your `precision score` and `recall score`  with a 
    specific tradeoff. Suppose  to get a precision of 90%, you 
    might specify a tradeoff and get the `precision score` and 
    `recall score` by setting a `y-tradeoff` value.
    """
    )
_param_docs = DocstringComponents.from_nested_components(
    core=_core_docs["params"],
    metric=DocstringComponents(_metrics_params ), 
    )

def mean_squared_log_error(y_true, y_pred, clip_value=0, epsilon=1e-15):
    """
    Compute the Mean Squared Logarithmic Error (MSLE) between true and
    predicted values. 
    
    This metric is useful for regression tasks where the
    focus is on the percentage differences rather than absolute differences.
    It penalizes underestimates more than overestimates. The function allows
    for clipping predictions to a minimum value for numerical stability and
    includes an epsilon to ensure values are strictly positive before
    logarithmic transformation.

    Parameters
    ----------
    y_true : array-like
        True target values. Must be non-negative.
    y_pred : array-like
        Predicted target values. Can contain any real numbers.
    clip_value : float, optional
        The value to clip `y_pred` values at minimum. This ensures that
        the logged values are not negative, by default 0.
    epsilon : float, optional
        A small value added to `y_pred` after clipping and to `y_true`, to
        ensure they are strictly positive before applying the logarithm, by
        default 1e-15.

    Returns
    -------
    float
        The Mean Squared Logarithmic Error between `y_true` and `y_pred`.

    Notes
    -----
    The MSLE is computed as:

    .. math::
        \\frac{1}{n} \\sum_{i=1}^{n} (\\log(p_i + 1) - \\log(a_i + 1))^2

    Where:
    - \(p_i\) is the \(i\)th predicted value,
    - \(a_i\) is the \(i\)th actual value,
    - \(\\log\) is the natural logarithm,
    - \(n\) is the total number of observations in the dataset.

    It is important that `y_true` contains non-negative values only, as the
    logarithm of negative values is undefined. The function enforces `y_pred`
    values to be at least `clip_value` to avoid taking the logarithm of
    negative numbers or zero, potentially leading to `-inf` or `NaN`. The
    addition of `epsilon` ensures that even after clipping, no value is exactly
    zero before the logarithm is applied, providing a buffer against numerical
    instability.
    
    Examples
    --------
    >>> from gofast.metrics import mean_squared_log_error
    >>> y_true = [3, 5, 2.5, 7]
    >>> y_pred = [2.5, 5, 4, 8]
    >>> mean_squared_log_error(y_true, y_pred)
    0.03973

    >>> mean_squared_log_error(y_true, [2, -5, 4, 8], clip_value=0.01)
    0.8496736598821342
    # Example output with clipping and epsilon adjustment
    """
    y_true, y_pred = _ensure_y_is_valid(y_true, y_pred, y_numeric= True )
    # If y_numeric is True, check if y_true contains only non-negative values
    if np.any(y_true < 0):
        raise ValueError("y_true must contain non-negative values only.")
        
    y_pred = np.clip(y_pred, clip_value, None) + epsilon  # Clip and adjust `y_pred`
    y_true += epsilon  # Adjust `y_true` to avoid log(0)
    return np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)

def balanced_accuracy(
    y_true, y_pred, 
    normalize=False, 
    sample_weight=None, 
    strategy='ovr', 
    epsilon=1e-15, 
    zero_division=0, 
    ):
    """
    Compute the Balanced Accuracy for binary and multiclass classification
    tasks. This metric is especially useful in situations with imbalanced
    classes. It allows choosing between One-vs-Rest (OVR) and One-vs-One
    (OVO) strategies for multiclass scenarios, providing flexibility based
    on the specific characteristics of the dataset and the problem at hand.

    Parameters
    ----------
    y_true : array-like
        True labels of the data.
    y_pred : array-like
        Predicted labels by the classifier.
    normalize : bool, optional
        If True, normalize the confusion matrix before computing metrics, by
        default False.
    sample_weight : array-like, optional
        Array of weights that are assigned to individual samples, by default None.
    strategy : str, optional
        Strategy for multiclass metrics calculation - either 'ovr' for One-vs-Rest
        or 'ovo' for One-vs-One, by default 'ovr'.
    epsilon : float, optional
        Small constant to avoid division by zero, by default 1e-15.
    zero_division : int, optional
        Value to return when there is a division by zero, by default 0.
        
    Returns
    -------
    float
        The Balanced Accuracy score.

    Notes
    -----
    The Balanced Accuracy in the binary classification scenario is defined as the
    average of sensitivity (true positive rate) and specificity (true negative rate):

    .. math:: BA = \frac{TPR + TNR}{2}

    Where:
    - TPR (True Positive Rate) = \frac{TP}{TP + FN}
    - TNR (True Negative Rate) = \frac{TN}{TN + FP}

    In multiclass scenarios, the metric is computed as the average of sensitivity
    for each class, considering each class as the positive class in turn, which
    corresponds to the One-vs-Rest (OVR) strategy. The One-vs-One (OVO) strategy
    involves averaging the metric over all pairs of classes.

    Examples
    --------
    Binary classification example:

    >>> from gofast.metrics import balanced_accuracy
    >>> y_true = [0, 1, 0, 1]
    >>> y_pred = [0, 1, 0, 0]
    >>> balanced_accuracy(y_true, y_pred)
    0.75

    Multiclass classification example with OVR strategy:

    >>> y_true = [0, 1, 2, 2, 1]
    >>> y_pred = [0, 2, 2, 1, 1]
    >>> balanced_accuracy(y_true, y_pred, strategy='ovr')
    0.666

    Multiclass classification example with OVO strategy:

    >>> balanced_accuracy(y_true, y_pred, strategy='ovo')
    0.666
    """
    # Ensure y_true and y_pred are valid and have consistent lengths
    y_true, y_pred = _ensure_y_is_valid(y_true, y_pred)

    # Check that y_true and y_pred are suitable for classification
    y_true, y_pred = check_classification_targets(
        y_true, y_pred, strategy="custom logic")
    labels = unique_labels(y_true, y_pred)
    
    if len(labels) == 2:  # Binary classification scenario
        cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Calculate sensitivity (True Positive Rate) and specificity (True Negative Rate)
        sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0] + epsilon)
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1] + epsilon)
        
        # Return the Balanced Accuracy
        return (sensitivity + specificity) / 2
    
    else:  # Multiclass classification scenario
        strategy =normalize_string(
            strategy, target_strs=['ovr', 'ovo'], match_method='contains', 
            return_target_only= True, raise_exception= True, 
            error_msg=("strategy parameter must be either 'ovr' or 'ovo'") 
            ) 
        if strategy == 'ovr':
            sensitivities = []
            for label in labels:
                binary_y_true = (y_true == label).astype(int)
                binary_y_pred = (y_pred == label).astype(int)
                cm = confusion_matrix(binary_y_true, binary_y_pred, sample_weight=sample_weight)
                
                if normalize:
                    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                
                tp = cm[1, 1]
                fn = cm[1, 0]
                sensitivity = tp / (tp + fn + epsilon) if (tp + fn) > 0 else zero_division
                sensitivities.append(sensitivity)
            # Compute average sensitivity across all classes
            return np.mean(sensitivities)
        
        elif strategy == 'ovo':
            # For OVO, use a simplified approach as an example, like averaging pairwise AUC scores
            # Note: This is a simplification and may not align with all 
            # interpretations of OVO strategy
            y_bin = label_binarize(y_true, classes=labels)
            auc_scores = []
            for i, label_i in enumerate(labels):
                for j, label_j in enumerate(labels):
                    if i >= j:
                        continue  # Avoid duplicate pairs and self-comparison
                    specific_sample_weight = sample_weight[(y_true == label_i) | (
                        y_true == label_j)] if sample_weight is not None else None
                    specific_y_true = y_bin[:, i][(y_true == label_i) | (y_true == label_j)]
                    specific_y_pred = y_bin[:, i][(y_true == label_i) | (y_true == label_j)]
                    auc_score = roc_auc_score(specific_y_true, specific_y_pred,
                                              sample_weight=specific_sample_weight)
                    auc_scores.append(auc_score)
            return np.mean(auc_scores)

def information_value(
    y_true, y_pred, 
    problem_type='binary', 
    epsilon="auto", 
    scale='binary_scale', 
    method='binning', 
    bins='auto', 
    bins_method='freedman_diaconis',
    data_range=None, 
    ):
    """
    Calculate the Information Value (IV) for various types of classification 
    and regression problems. 
    
    The :term:`IV` quantifies the predictive power of a feature or model, with  
    higher values indicating greater predictive power. This function supports 
    ``binary``, ``multiclass``, ``multilabel`` classifications, and ``regression``
     problems, allowing for flexibility in its application across different data 
    science tasks.

    Parameters
    ----------
    y_true : array-like
        True labels or values. For classification problems, these should be class
        labels. For regression problems, these are continuous target values.
        
    y_pred : array-like
        Predicted probabilities or values. For classification, these are typically
        the probabilities of belonging to the positive class (in binary classification)
        or class probabilities (in multiclass classification). For regression 
        problems, these are the predicted continuous values.
        
    problem_type : str, optional
        Specifies the type of problem for which the IV is being calculated. Valid
        options are ``'binary'``, ``'multiclass'``, ``'multilabel'``, and 
        ``'regression'``. This parameter determines how the IV calculation is 
        adapted to fit the problem context. Default is 'binary'.
        
    epsilon : float or "auto", optional
        A small epsilon value added to probabilities to prevent division by zero
        in the logarithmic calculation. If "auto", an appropriate epsilon value
        is dynamically determined based on the predicted values. Default is "auto".
        
    scale : str or float, optional
        For multiclass and multilabel problems, defines how the IV should be scaled
        or normalized. The default ``'binary_scale'`` scales these problems 
        to a binary IV scale for easier interpretation. If a float is provided,
        it custom scales the IV by this factor.
        
    method : str, optional
        Specifies the calculation method for Information Value (IV). Two methods 
        are supported:
        - 'base': A straightforward calculation based on the overall distribution 
          of events and non-events in the dataset. Suitable for a high-level 
          overview of predictive power.
        - 'binning': A more detailed analysis that divides the predicted probabilities
          (`y_pred`) into bins and calculates IV within each bin. This method is 
          valuable for examining how the model's predictive power varies across 
          different probability ranges. It requires specifying `bins` and can be 
          fine-tuned further with `bins_method` and `data_range`. 
          Default is ``'binning'``.
        
    bins : int, 'auto', optional
        Defines the number of bins used for the 'binning' method. This parameter 
        is crucial for segmenting the predicted probabilities (`y_pred`) into 
        discrete bins, allowing for a more granular analysis of the model's 
        predictive power. If set to 'auto', the number of bins is determined 
        using the method specified by `bins_method`. Providing an integer directly 
        specifies the fixed number of bins to use. Default is 'auto'.

    bins_method : str, optional
        Specifies the method to determine the optimal number of bins when 
        `bins` is set to 'auto'. Available methods are:
        - 'freedman_diaconis': Employs the Freedman-Diaconis rule, which bases 
          the bin width on the data's interquartile range and the cube root of 
          data size. Ideal for a wide range of distributions.
          .. math:: \text{bin width} = 2 \cdot \frac{IQR}{\sqrt[3]{n}}
        - 'sturges': Uses Sturges' formula, which is a function of the data size. 
          Suitable for normal distributions but may perform poorly for large, 
          skewed datasets.
          .. math:: \text{bins} = \log_2(n) + 1
        - 'sqrt': Calculates the square root of the data size to determine the 
          number of bins. A simple heuristic useful for smaller datasets.
          .. math:: \text{bins} = \sqrt{n}
        The chosen method can significantly impact the IV calculation, especially 
        in the 'binning' approach, influencing the resolution of the predictive 
        power analysis across different probability ranges.
        Default method is 'freedman_diaconis'.
        
    data_range : tuple, optional
        A tuple specifying the minimum and maximum values to consider for binning 
        when `method` is set to 'binning'. This parameter allows focusing the IV 
        analysis on a specific range of predicted probabilities, which can be 
        useful for excluding outliers or focusing on a region of interest. If None,
        the entire range of `y_pred` is used. Default is None.
        
    Returns
    -------
    float
        The calculated Information Value (IV) score. Positive values indicate
        predictive power, with higher values being preferable. For regression
        and multiclass/multilabel classifications with `scale='binary_scale'`,
        the IV is adjusted to fit a binary classification context.

    Raises
    ------
    ValueError
        If an invalid `problem_type` is specified.
        
    Notes
    -----
    The Information Value (IV) quantifies the predictive power of a feature or model, 
    illustrating its ability to distinguish between classes or predict outcomes. It is 
    a measure of the effectiveness of a variable or model in predicting the target.

    Mathematical formulations for IV across different problem types are as follows:

    - For binary classification:
      The IV is computed as the sum of differences between the proportions of 
      non-events and events, each multiplied by the logarithm of the ratio of
      these proportions, adjusted by a small epsilon (`\epsilon`) to prevent 
      division by zero:
          
      .. math::
        IV = \sum \left((\% \text{{ of non-events}} - \% \text{{ of events}}) \times 
        \ln\left(\frac{\% \text{{ of non-events}} + \epsilon}
        {\% \text{{ of events}} + \epsilon}\right)\right)

    - For multiclass classification:
      The IV is adapted from the average log loss for multiclass classification, 
      normalized to the binary IV scale. This normalization is essential for 
      comparing the predictive power of multiclass models to binary models:
      The log loss for multiclass classification is given by:
      
      .. math::
        \text{{log_loss}} = - \frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{M} 
        y_{ij} \log(p_{ij})
        
        IV = -\frac{1}{\log(2)} \cdot \text{{log_loss}}(y\_true, y\_pred)
        
      where \(N\) is the number of samples, \(M\) is the number of classes, 
      \(y_{ij}\) is a binary indicator of whether class \(j\) is the correct 
      classification for sample \(i\), and \(p_{ij}\) is the model probability 
      of assigning class \(j\) to sample \(i\).

    - For multilabel classification:
      Similar to multiclass, the IV for multilabel classification uses the 
      negative average binary cross-entropy loss, normalized to the binary scale. 
      This approach allows direct comparison between the predictive powers 
      of multilabel and binary scale. The binary cross-entropy for multilabel
      classification is given as:
      
      .. math::
        \text{{binary\_crossentropy}} = - \frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{M} 
        \left( y_{ij} \log(p_{ij}) + (1 - y_{ij}) \log(1 - p_{ij}) \right)
        
        IV = -\frac{1}{\log(2)} \cdot \text{{binary\_crossentropy}}(y\_true, y\_pred)
        
      where \(N\) is the number of samples, \(M\) is the number of labels, 
      \(y_{ij}\) indicates whether label \(j\) is relevant to sample \(i\), 
      and \(p_{ij}\) is the predicted probability of label \(j\) for sample \(i\).

    - For regression:
      In regression problems, the IV is determined by the negative mean squared error 
      (MSE) between the true and predicted values. Lower (more negative) MSE values, 
      indicating closer predictions to the actual values, translate into higher IV 
      scores, signifying better predictive capability:
      
      .. math::
        IV = -\frac{1}{N} \sum_{i=1}^{N} (y_{i} - \hat{y_{i}})^2
        
      where \(N\) is the number of samples, \(y_{i}\) is the true value for 
      sample \(i\), and \(\hat{y_{i}}\) is the predicted value for sample \(i\).

    These formulations adjust the traditional concept of IV for a broad spectrum 
    of applications beyond binary classification, enhancing its versatility as a 
    metric for assessing model performance across various types of predictive 
    modeling tasks.

    IV is invaluable in fields such as credit scoring and risk modeling, aiding in 
    the evaluation of a model's or variable's predictive power. By analyzing IV, 
    data scientists can pinpoint the most informative features, leading to the 
    development of superior predictive models.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.metrics import log_loss, mean_squared_error
    >>> from gofast.metrics import information_value 
    
    Binary classification with automatically determined epsilon:

    >>> y_true = [0, 1, 1, 0]
    >>> y_pred = [0.1, 0.9, 0.8, 0.2]
    >>> print(information_value(y_true, y_pred, problem_type='binary'))
    1.3219280948873623
    >>> y_true_binary = np.array([0, 1, 1, 0, 1, 0])
    >>> y_pred_binary = np.array([0.2, 0.7, 0.6, 0.3, 0.8, 0.1])
    >>> iv_binary = information_value(y_true_binary, y_pred_binary, problem_type='binary')
    >>> iv_binary
    0.7621407247492977
    
    Multiclass classification with specified scale:

    >>> y_true = [2, 1, 0, 1, 2]
    >>> y_pred = [[0.1, 0.2, 0.7], [0.2, 0.7, 0.1], [0.7, 0.2, 0.1],
    ...           [0.2, 0.7, 0.1], [0.1, 0.3, 0.6]]
    >>> print(information_value(y_true, y_pred, problem_type='multiclass'))
    -0.6365141682948128

    >>> y_true_multiclass = np.array([0, 1, 2, 1, 0])
    >>> y_pred_multiclass = np.array([[0.2, 0.4, 0.4], [0.7, 0.1, 0.2],
    ...                                  [0.1, 0.2, 0.7], [0.3, 0.3, 0.4], 
    ...                                  [0.8, 0.1, 0.1]])
    >>> iv_multiclass = information_value(y_true_multiclass, y_pred_multiclass,
    ...                                      problem_type='multiclass')
    >>> iv_multiclass
    -0.6729845552217573
    
    Multilabel classification with specified scale:
 
    >>> y_true_multilabel = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]])
    >>> y_pred_multilabel = np.array([[0.8, 0.2, 0.7], [0.1, 0.9, 0.3], 
    ...                                  [0.9, 0.7, 0.1], [0.2, 0.1, 0.8]])
    >>> iv_multilabel = information_value(y_true_multilabel, y_pred_multilabel,
    ...                                      problem_type='multilabel')
    >>> iv_multilabel
    -0.4750837692748695
 
    >>> y_true_regression = np.array([10.5, 12.1, 9.8, 11.2, 10.0])
    >>> y_pred_regression = np.array([10.2, 12.3, 9.5, 11.0, 10.1])
    >>> iv_regression = information_value(y_true_regression, y_pred_regression,
    ...                                      problem_type='regression')
    >>> iv_regression
    -0.04239999999999994
    
    Regression with custom scale factor:

    >>> y_true = [3.5, 2.5, 4.0, 5.5]
    >>> y_pred = [3.0, 2.7, 4.1, 5.0]
    >>> print(information_value(y_true, y_pred, problem_type='regression', scale=1.0))
    -0.0475
    
    Using the 'binning' method with automatically determined bins:
    
    >>> y_true = np.array([0, 1, 1, 0, 1])
    >>> y_pred = np.array([0.1, 0.6, 0.8, 0.05, 0.9])
    >>> iv_auto_bins = information_value(y_true, y_pred, 
    ...                                  problem_type='binary', 
    ...                                  method='binning', 
    ...                                  bins='auto', 
    ...                                  bins_method='freedman_diaconis')
    >>> print(f"IV with auto bins: {iv_auto_bins}")

    Specifying a fixed number of bins for the 'binning' method:
    
    >>> iv_fixed_bins = information_value(y_true, y_pred, 
    ...                                   problem_type='binary', 
    ...                                   method='binning', 
    ...                                   bins=5)
    >>> print(f"IV with fixed bins: {iv_fixed_bins}")
    
    Using the 'binning' method with a specified data range:

    >>> y_true = np.array([0, 1, 0, 1, 1, 0])
    >>> y_pred = np.array([0.05, 0.95, 0.2, 0.85, 0.9, 0.1])
    >>> iv_data_range = information_value(y_true, y_pred, 
    ...                                   problem_type='binary', 
    ...                                   method='binning', 
    ...                                   bins='auto', 
    ...                                   bins_method='freedman_diaconis',
    ...                                   data_range=(0.1, 0.9))
    >>> print(f"IV with specified data range: {iv_data_range}")

    The 'binning' method allows for a nuanced understanding of model performance, 
    especially useful in scenarios where predictive power might vary significantly 
    across the probability spectrum. Specifying `data_range` can refine this 
    analysis, offering insights into specific segments of the prediction range.
    
    See Also 
    ----------
    gofast.tools.calculate_binary_iv: 
        Calculate the Information Value (IV) for binary classification problems
        using a base or binning approach.
    """
    # Implementation goes here...
    y_true, y_pred = _ensure_y_is_valid(y_true, y_pred, y_numeric =True, 
                                        multi_output= True )
    # Determine an appropriate epsilon value if set to "auto"
    if str(epsilon).lower() == "auto":
        epsilon = determine_epsilon(y_pred)
    # Initialize IV calculation
    iv = None
    if problem_type == 'binary':
        iv = calculate_binary_iv(
            y_true, y_pred, method=method,
            bins=bins, epsilon= epsilon, 
            bins_method=bins_method,
            data_range=data_range, 
            )
        
    elif problem_type == 'multiclass':
        # Normalize to binary IV scale if requested
        iv = -log_loss(y_true, y_pred, eps=epsilon) / np.log(
            2) if scale == 'binary_scale' else -log_loss(y_true, y_pred, eps=epsilon)
        
    elif problem_type == 'multilabel':
        iv = -np.mean(y_true * np.log(y_pred + epsilon) + (
            1 - y_true) * np.log(1 - y_pred + epsilon)) / np.log(
                2) if scale == 'binary_scale' else -np.mean(
                    y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(
                        1 - y_pred + epsilon))
        
    elif problem_type == 'regression':
        iv = -mean_squared_error(y_true, y_pred)
        
    else:
        raise ValueError("Invalid 'problem_type'. Use 'binary', 'multiclass',"
                         " 'multilabel', or 'regression'.")

    # Apply custom scale if specified
    if isinstance(scale, float):
        iv *= scale
 
    return iv

def geo_iv (Xp:ArrayLike, /,  Np:ArrayLike, Sp:ArrayLike, *, 
                       return_tot :bool = ... ): 
    """The geo-information value (geo-InV) constructs with the influencing factors 
    landslide areas and calculates the sensitivity of each influencing 
    factor.
    
    InV method is a statistical approach for the prediction of spatial events
    on the basis of associated parameters and landslide relationships [1]_. 
    The inV :math:`I_i` of each causative factor :math:`X_i` can be 
    expressed as: 
        
    .. math:: 
        
        I_i= \log \frac{\frac{S_i}{N_i}{\frac{S}{N}} 
                        
    where :math:`S_i`  is the landslide pixel number in the presence of a 
    causative factor ܺ:math:`X_i`, :math:`N_i` is the number of pixels 
    associated with causative factor ܺ:math:`X_i` . :math:`S` is the total 
    number of landslide pixels, and:math:`N` is the total number of pixels in 
    the study area. Then, the overall information value :math:`I` can be 
    calculated by:
        
    .. math:: 
        
        I_i= \sum{i=1}{n} \log \frac{\frac{S_i}{N_i}}{\frac{S}{N}}
        
    Thus Negative and positive values of :math:\I_i` irrepresent the relevant  
    and relevant correlation between the presence of a certain causative factor 
    and landslide event, respectively. The stronger the correlation is, the 
    higher the value of :math:`I_i`. See more details in the paper of Chen 
    Tao with the following :doi:`https://doi.org/10.1007/s11629-019-5839-3` .
    
    
    Parameters 
    ----------
    
    Xp: Arraylike, Shape (n_samples ) of pixels 
       Causative factor. 
       
    Sp: Arraylike ,Shape of (n_samples) of pixels  
        The landslide pixel number in the presence of a causative factor 
        :math:`X_i`.
    Np: Arraylike, Shape of n_samples of pixels 
       Number of pixels associated to the causative factor :math:`X_i`. 
       
    return_tot: bool, default=False 
      Returns the overall information value. 

    Returns 
    ---------
    I: Arraylike 
       Information value of caustive factor :math:`X_p`. 
       
    
    References 
    ------------
    .. [1] Sarkar S, Kanungo D, Patra A (2006) GIS Based Landslide 
           Susceptibility Mapping - A Case Study in Indian Himalaya in 
           Disaster Mitigation of Debris Flows, Slope Failures and 
           Landslides, Universal Academic Press, Tokyo, 617-624 
    """
    if not _is_numeric_dtype(Xp, to_array= True ): 
        raise TypeError ("Causative factor expect a numeric array."
                         f" Got {np.array(Xp).dtype!r}")
    Xp = np.array ( is_iterable(Xp , transform =True ))
    
    if Np: Np = _assert_all_types(Np , float , int, 
                                  objname= "Number of pixel 'Ni'")

    I = np.zeros_like ( Xp , dtype = float )

    for i in len(Xp ): 
        Ii = _get_geo_iv_value ( Sp[i], Np[i], Sp, Np )
        # Ii = np.log10 ( ( Sp[i] / Np[i])/( len(Sp)/len(Np) ))
        I[i]= Ii 
    return I if not return_tot else np.sum (I )
    

def _get_geo_iv_value ( Si, Ni, S, N ): 
    """ Compute the information value :math:`I_i` of each causative factor
    :math:`X_i` """
    return np.log10 ( Si/Ni * ( S/ N )) 

def get_metrics(): 
    """
    Get the list of  available metrics. 
    
    Metrics are measures of quantitative assessment commonly used for 
    assessing, comparing, and tracking performance or production. Generally,
    a group of metrics will typically be used to build a dashboard that
    management or analysts review on a regular basis to maintain performance
    assessments, opinions, and business strategies.
    """
    return tuple(metrics.SCORERS.keys())

def get_eval_scores (
    model, 
    Xt, 
    yt, 
    *,average="binary", 
    multi_class="raise", 
    normalize=True, 
    sample_weight=None,
    verbose = False, 
    **scorer_kws, 
    ): 
    ypred = model.predict(Xt) 
    acc_scores = accuracy_score(yt, ypred, normalize=normalize, 
                                sample_weight= sample_weight) 
    rec_scores = recall_score(
        yt, ypred, average =average, sample_weight = sample_weight, 
        **scorer_kws)
    prec_scores = precision_score(
        yt, ypred, average =average,sample_weight = sample_weight, 
        **scorer_kws)
    try:
        #compute y_score when predict_proba is available 
        # or  when  probability=True
        ypred = model.predict_proba(Xt) if multi_class !='raise'\
            else model.predict(Xt) 
    except: rocauc_scores=None 
    else :
        rocauc_scores= roc_auc_score (
            yt, ypred, average=average, multi_class=multi_class, 
            sample_weight = sample_weight, **scorer_kws)

    scores= Boxspace (**dict ( 
        accuracy = acc_scores , recall = rec_scores, 
        precision= prec_scores, auc = rocauc_scores 
        ))
    if verbose: 
        mname=get_estimator_name(model)
        print(f"{mname}:\n")
        print("accuracy -score = ", acc_scores)
        print("recall -score = ", rec_scores)
        print("precision -score = ", prec_scores)
        print("ROC AUC-score = ", rocauc_scores)
    return scores 

get_eval_scores.__doc__ ="""\
Compute the `accuracy`,  `precision`, `recall` and `AUC` scores.

Parameters 
------------
{params.core.model}
{params.core.Xt} 
{params.core.yt}

average : {{'micro', 'macro', 'samples', 'weighted', 'binary'}} or None, \
        default='binary'
    This parameter is required for multiclass/multilabel targets.
    If ``None``, the scores for each class are returned. Otherwise, this
    determines the type of averaging performed on the data:

    ``'binary'``:
        Only report results for the class specified by ``pos_label``.
        This is applicable only if targets (``y_{{true,pred}}``) are binary.
    ``'micro'``:
        Calculate metrics globally by counting the total true positives,
        false negatives and false positives.
    ``'macro'``:
        Calculate metrics for each label, and find their unweighted
        mean.  This does not take label imbalance into account.
    ``'weighted'``:
        Calculate metrics for each label, and find their average weighted
        by support (the number of true instances for each label). This
        alters 'macro' to account for label imbalance; it can result in an
        _F-score that is not between precision and recall. Weighted recall
        is equal to accuracy.
    ``'samples'``:
        Calculate metrics for each instance, and find their average (only
        meaningful for multilabel classification where this differs from
        :func:`accuracy_score`).
        Will be ignored when ``y_true`` is binary.
        Note: multiclass ROC AUC currently only handles the 'macro' and
        'weighted' averages.
        
multi_class : {{'raise', 'ovr', 'ovo'}}, default='raise'
    Only used for multiclass targets. Determines the type of configuration
    to use. The default value raises an error, so either
    ``'ovr'`` or ``'ovo'`` must be passed explicitly.

    ``'ovr'``:
        Stands for One-vs-rest. Computes the AUC of each class
        against the rest [1]_ [2]_. This
        treats the multiclass case in the same way as the multilabel case.
        Sensitive to class imbalance even when ``average == 'macro'``,
        because class imbalance affects the composition of each of the
        'rest' groupings.
    ``'ovo'``:
        Stands for One-vs-one. Computes the average AUC of all
        possible pairwise combinations of classes [3]_.
        Insensitive to class imbalance when
        ``average == 'macro'``.
        
normalize : bool, default=True
    If ``False``, return the number of correctly classified samples.
    Otherwise, return the fraction of correctly classified samples.

sample_weight : array-like of shape (n_samples,), default=None
    Sample weights.
    
{params.core.verbose}

scorer_kws: dict, 
    Additional keyword arguments passed to the scorer metrics: 
    :func:`~sklearn.metrics.accuracy_score`, 
    :func:`~sklearn.metrics.precision_score`, 
    :func:`~sklearn.metrics.recall_score`, 
    :func:`~sklearn.metrics.roc_auc_score`
    
Returns 
--------
scores: :class:`gofast.tools.box.Boxspace`. , 
    A dictionnary object to retain all the scores from metrics evaluation such as 
    - accuracy , 
    - recall 
    - precision 
    - ROC AUC ( Receiving Operating Characteric Area Under the Curve)
    Each score can be fetch as an attribute. 
    
Notes 
-------
Note that if `yt` is given, it computes `y_score` known as array-like of 
shape (n_samples,) or (n_samples, n_classes)Target scores following the 
scheme below: 

* In the binary case, it corresponds to an array of shape
  `(n_samples,)`. Both probability estimates and non-thresholded
  decision values can be provided. The probability estimates correspond
  to the **probability of the class with the greater label**,
  i.e. `estimator.classes_[1]` and thus
  `estimator.predict_proba(X, y)[:, 1]`. The decision values
  corresponds to the output of `estimator.decision_function(X, y)`.
  See more information in the :ref:`User guide <roc_auc_binary>`;
* In the multiclass case, it corresponds to an array of shape
  `(n_samples, n_classes)` of probability estimates provided by the
  `predict_proba` method. The probability estimates **must**
  sum to 1 across the possible classes. In addition, the order of the
  class scores must correspond to the order of ``labels``,
  if provided, or else to the numerical or lexicographical order of
  the labels in ``y_true``. See more information in the
  :ref:`User guide <roc_auc_multiclass>`;
* In the multilabel case, it corresponds to an array of shape
  `(n_samples, n_classes)`. Probability estimates are provided by the
  `predict_proba` method and the non-thresholded decision values by
  the `decision_function` method. The probability estimates correspond
  to the **probability of the class with the greater label for each
  output** of the classifier. See more information in the
  :ref:`User guide <roc_auc_multilabel>`.
      
References
----------

.. [1] Provost, _F., Domingos, P. (2000). Well-trained PETs: Improving
       probability estimation trees (Section 6.2), CeDER Working Paper
       #IS-00-04, Stern School of Business, New York University.

.. [2] `Fawcett, T. (2006). An introduction to ROC analysis. Pattern
        Recognition Letters, 27(8), 861-874.
        <https://www.sciencedirect.com/science/article/pii/S016786550500303X>`_
         
.. [3] `Hand, D.J., Till, R.J. (2001). A Simple Generalisation of the Area
        Under the ROC Curve for Multiple Class Classification Problems.
        Machine Learning, 45(2), 171-186.
        <http://link.springer.com/article/10.1023/A:1010920819831>`_
See Also
--------
average_precision_score : Area under the precision-recall curve.
roc_curve : Compute Receiver operating characteristic (ROC) curve.
RocCurveDisplay.from_estimator : Plot Receiver Operating Characteristic
    (ROC) curve given an estimator and some data.
RocCurveDisplay.from_predictions : Plot Receiver Operating Characteristic
    (ROC) curve given the true and predicted values.
    
Examples
--------
Binary case:

>>> from sklearn.datasets import load_breast_cancer
>>> from sklearn.linear_model import LogisticRegression
>>> from sklearn.metrics import roc_auc_score
>>> X, y = load_breast_cancer(return_X_y=True)
>>> clf = LogisticRegression(solver="liblinear", random_state=0).fit(X, y)
>>> roc_auc_score(y, clf.predict_proba(X)[:, 1])
0.99...
>>> roc_auc_score(y, clf.decision_function(X))
0.99...

Multiclass case:

>>> from sklearn.datasets import load_iris
>>> X, y = load_iris(return_X_y=True)
>>> clf = LogisticRegression(solver="liblinear").fit(X, y)
>>> roc_auc_score(y, clf.predict_proba(X), multi_class='ovr')
0.99...

Multilabel case:

>>> import numpy as np
>>> from sklearn.datasets import make_multilabel_classification
>>> from sklearn.multioutput import MultiOutputClassifier
>>> X, y = make_multilabel_classification(random_state=0)
>>> clf = MultiOutputClassifier(clf).fit(X, y)
>>> # get a list of n_output containing probability arrays of shape
>>> # (n_samples, n_classes)
>>> y_pred = clf.predict_proba(X)
>>> # extract the positive columns for each output
>>> y_pred = np.transpose([pred[:, 1] for pred in y_pred])
>>> roc_auc_score(y, y_pred, average=None)
array([0.82..., 0.86..., 0.94..., 0.85... , 0.94...])
>>> from sklearn.linear_model import RidgeClassifierCV
>>> clf = RidgeClassifierCV().fit(X, y)
>>> roc_auc_score(y, clf.decision_function(X), average=None)
array([0.81..., 0.84... , 0.93..., 0.87..., 0.94...])
""".format(params =_param_docs
)
    
def _assert_metrics_args(y, label): 
    """ Assert metrics argument 
    
    :param y: array-like, 
        label for prediction. `y` is binary label by default. 
        If `y` is composed of multilabel, specify  the `classe_` 
        argumentto binarize the label(`True` ot `False`). ``True``  
        for `classe_`and ``False`` otherwise. 
    :param label:float, int 
        Specific class to evaluate the tradeoff of precision 
        and recall. If `y` is already a binary classifer, `classe_` 
        does need to specify.     
    """
    # check y if value to plot is binarized ie.True of false 
    msg = ("Precision-recall metrics are fundamentally metrics for"
           " binary classification. ")
    y_unik = np.unique(y)
    if len(y_unik )!=2 and label is None: 
        warnings.warn( msg + f"Classes values of 'y' is '{len(y_unik )}', "
                      "while expecting '2'. Can not set the tradeoff for "
                      " non-binarized classifier ",  UserWarning
                       )
        _logger.warning('Expect a binary classifier(2), but %s are given'
                              %len(y_unik ))
        raise LearningError(f'Expect a binary labels but {len(y_unik )!r}'
                         f' {"are" if len(y_unik )>1 else "is"} given')
        
    if label is not None: 
        try : 
            label= int(label)
        except ValueError: 
            raise ValueError('Need integer value; Could not convert to Float.')
        except TypeError: 
            raise TypeError(f'Could not convert {type(label).__name__!r}') 
    
    if label not in y: 
        raise ValueError("Value '{}' must be a label of a binary target"
                         .format(label))
  
def precision_recall_tradeoff(
    clf:_F, 
    X:NDArray,
    y:ArrayLike,
    *,
    cv:int =7,
    label: str | Optional[List[str]]=None,
    method:Optional[str] =None,
    cvp_kws: Optional[dict]  =None,
    tradeoff: Optional[float] =None,
    **prt_kws
)-> object:
    pass 
    
    # mc= copy.deepcopy(method)
    # method = method or "decision_function"
    # method =str(method).lower().strip() 
    # if method not in ('decision_function', 'predict_proba'): 
    #     raise ValueError (f"Invalid method {mc!r}.Expect 'decision_function'"
    #                       " or 'predict_proba'.")
        
    # #create a object to hold attributes 
    # obj = type('Metrics', (), {})
    
    # _assert_metrics_args(y, label)
    # y=(y==label) # set boolean 
    
    # if cvp_kws is None: 
    #     cvp_kws = dict()
        
    # obj.y_scores = cross_val_predict(clf,X,y,cv =cv,
    #                                  method= method,**cvp_kws )
    # y_scores = cross_val_predict(clf,X,y, cv =cv,**cvp_kws )
    
    # obj.confusion_matrix =cfsmx(y, y_scores )
    
    # obj.f1_score = f1_score(y,y_scores)
    # obj.precision_score = precision_score(y, y_scores)
    # obj.recall_score= recall_score(y, y_scores)
        
    # if method =='predict_proba': 
    #     # if classifier has a `predict_proba` method like 
    #     # `Random_forest` then use the positive class
    #     # probablities as score  score = proba of positive 
    #     # class 
    #     obj.y_scores =obj.y_scores [:, 1] 
        
    # if tradeoff is not None:
    #     try : 
    #         float(tradeoff)
    #     except ValueError: 
    #         raise ValueError(f"Could not convert {tradeoff!r} to float.")
    #     except TypeError: 
    #         raise TypeError(f'Invalid type `{type(tradeoff)}`')
            
    #     y_score_pred = (obj.y_scores > tradeoff) 
    #     obj.precision_score = precision_score(y, y_score_pred)
    #     obj.recall_score = recall_score(y, y_score_pred)
        
    # obj.precisions, obj.recalls, obj.thresholds =\
    #     precision_recall_curve(y, obj.y_scores,**prt_kws)
        
    # obj.y =y
    
    # return obj

precision_recall_tradeoff.__doc__ ="""\
Precision-recall Tradeoff computes a score based on the decision function. 

Is assign the instance to the positive class if that score on 
the left is greater than the `threshold` else it assigns to negative 
class. 

Parameters
----------
{params.core.clf}
{params.core.X}
{params.core.y}
{params.core.cv}

label: float, int 
    Specific class to evaluate the tradeoff of precision 
    and recall. If `y` is already a binary classifer, `classe_` 
    does need to specify. 
method: str
    Method to get scores from each instance in the trainset. 
    Ciuld be ``decison_funcion`` or ``predict_proba`` so 
    Scikit-Learn classifier generally have one of the method. 
    Default is ``decision_function``.
tradeoff: float, optional,
    check your `precision score` and `recall score`  with a 
    specific tradeoff. Suppose  to get a precision of 90%, you 
    might specify a tradeoff and get the `precision score` and 
    `recall score` by setting a `y-tradeoff` value.

Notes
------
    
Contreverse to the `confusion matrix`, a precision-recall 
tradeoff is very interesting metric to get the accuracy of the 
positive prediction named ``precison`` of the classifier with 
equation is:

.. math:: precision = TP/(TP+FP)
    
where ``TP`` is the True Positive and ``FP`` is the False Positive
A trival way to have perfect precision is to make one single 
positive precision (`precision` = 1/1 =100%). This would be usefull 
since the calssifier would ignore all but one positive instance. So 
`precision` is typically used along another metric named `recall`,
 also `sensitivity` or `true positive rate(TPR)`:This is the ratio of 
positive instances that are corectly detected by the classifier.  
Equation of`recall` is given as:

.. math:: recall = TP/(TP+FN)
    
where ``FN`` is of couse the number of False Negatives. 
It's often convenient to combine `preicion`and `recall` metrics into
a single metric call the `F1 score`, in particular if you need a 
simple way to compared two classifiers. The `F1 score` is the harmonic 
mean of the `precision` and `recall`. Whereas the regular mean treats 
all  values equaly, the harmony mean gives much more weight to low 
values. As a result, the classifier will only get the `F1 score` if 
both `recalll` and `preccion` are high. The equation is given below:

.. math::
    
    F1 &= 2/((1/precision)+(1/recall))= 2* precision*recall /(precision+recall) \\ 
       &= TP/(TP+ (FN +FP)/2)
    
The way to increase the precion and reduce the recall and vice versa
is called `preicionrecall tradeoff`.

Returns 
--------
obj: object, an instancied metric tying object 
    The metric object is composed of the following attributes:
        
    * `confusion_matrix` 
    * `f1_score`
    * `precision_score`
    * `recall_score`
    * `precisions` from `precision_recall_curve` 
    * `recalls` from `precision_recall_curve` 
    * `thresholds` from `precision_recall_curve` 
    * `y` classified 
    
    and can be retrieved for plot purpose.    
  
Examples
--------
>>> from gofast.exlib import SGDClassifier
>>> from gofast.metrics import precision_recall_tradeoff
>>> from gofast.datasets import fetch_data 
>>> X, y= fetch_data('Bagoue analysed')
>>> sgd_clf = SGDClassifier()
>>> mObj = precision_recall_tradeoff (clf = sgd_clf, X= X, y = y,
                                label=1, cv=3 , y_tradeoff=0.90) 
>>> mObj.confusion_matrix
""".format(
    params =_param_docs
)
    
def roc_curve_( 
    roc_kws:dict =None, 
    **tradeoff_kws
)-> object: 

    obj= precision_recall_tradeoff(**tradeoff_kws)
    # for key in obj.__dict__.keys():
    #     setattr(mObj, key, obj.__dict__[key])
    if roc_kws is None: roc_kws =dict()
    obj.fpr , obj.tpr , thresholds = roc_curve(obj.y, 
                                       obj.y_scores,
                                       **roc_kws )
    obj.roc_auc_score = roc_auc_score(obj.y, obj.y_scores)

    return obj 

roc_curve_.__doc__ ="""\
The Receiving Operating Characteric (ROC) curve is another common
tool  used with binary classifiers. 

It's very similar to precision/recall , but instead of plotting 
precision versus recall, the ROC curve plots the `true positive rate`
(TNR)another name for recall) against the `false positive rate`(FPR). 
The FPR is the ratio of negative instances that are correctly classified 
as positive.It is equal to one minus the TNR, which is the ratio 
of  negative  isinstance that are correctly classified as negative.
The TNR is also called `specify`. Hence the ROC curve plot 
`sensitivity` (recall) versus 1-specifity.

Parameters 
----------
{params.core.clf}
{params.core.X}
{params.core.y}
{params.core.cv}
{params.metric.label}
{params.metric.method}
{params.metric.tradeoff}

roc_kws: dict 
    roc_curve additional keywords arguments
    
See also
---------
gofast.view.mlplot.MLPlot.precisionRecallTradeoff:  
    plot consistency precision recall curve. 
    
    
Returns 
---------
obj: object, an instancied metric tying object 
    The metric object hold the following attributes additional to the return
    attributes from :func:~.precision_recall_tradeoff`:: 
        * `roc_auc_score` for area under the curve
        * `fpr` for false positive rate 
        * `tpr` for true positive rate 
        * `thresholds` from `roc_curve` 
        * `y` classified 
    and can be retrieved for plot purpose.    
    
Note 
-------
:func:`~roc_curve_` returns a ROC object for plotting purpose. ``_`` is used 
to differentiate it with the `roc_curve` metric provided by scikit-learn.
To get the `roc_curve` score, use  ``obj.<obj.roc_auc_score>`` instead.

Examples
--------
>>> from gofast.exlib import SGDClassifier
>>> from gofast.metrics import ROC_curve
>>> from gofast.datasets import fetch_data 
>>> X, y= fetch_data('Bagoue prepared')
>>> rocObj =ROC_curve(clf = sgd_clf,  X= X, 
               y = y, classe_=1, cv=3 )                                
>>> rocObj.__dict__.keys()
>>> rocObj.roc_auc_score 
>>> rocObj.fpr

""".format(
    params =_param_docs
)   

def confusion_matrix_(
    clf:_F, 
    X:NDArray, 
    y:ArrayLike,
    *, 
    cv:int =7, 
    plot_conf_max:bool =False, 
    crossvalp_kws:dict=dict(), 
    **conf_mx_kws 
)->object: 
    
    pass 
    # #create a object to hold attributes 
    # obj = type('Metrics', (), dict())
    # obj.y_pred =cross_val_predict(clf, X, y, cv=cv, **crossvalp_kws )
    
    # if obj.y_pred.ndim ==1 : 
    #     obj.y_pred.reshape(-1, 1)
    # obj.conf_mx = cfsmx(y, obj.y_pred, **conf_mx_kws)

    # # statement to plot confusion matrix errors rather than values 
    # row_sums = obj.conf_mx.sum(axis=1, keepdims=True)
    # norm_conf_mx = obj.conf_mx / row_sums 
    # # now let fill the diagonal with zeros to keep only the errors
    # # and let's plot the results 
    # np.fill_diagonal(norm_conf_mx, 0)
    # obj.norm_conf_mx= norm_conf_mx

    # fp =0
    # if plot_conf_max =='map': 
    #     confmax = obj.conf_mx
    #     fp=1
    # if plot_conf_max =='error':
    #     confmax= norm_conf_mx
    #     fp =1
    # if fp: 
    #     import matplotlib.pyplot as plt 
    #     plt.matshow(confmax, cmap=plt.cm.gray)
    #     plt.show ()
        
    # return obj  
  
confusion_matrix_.__doc__ ="""\
Evaluate the preformance of the model or classifier by counting 
the number of the times instances of class A are classified in class B. 

To compute a confusion matrix, you need first to have a set of 
prediction, so they can be compared to the actual targets. You could 
make a prediction using the test set, but it's better to keep it 
untouch since you are not ready to make your final prediction. Remember 
that we use the test set only at very end of the project, once you 
have a classifier that you are ready to lauchn instead. 
The confusion metric give a lot of information but sometimes we may 
prefer a more concise metric like `precision` and `recall` metrics. 

Parameters 
----------
{params.core.clf}
{params.core.X}
{params.core.y}
{params.core.cv}
{params.metric.label}
{params.metric.method}
{params.metric.tradeoff}

plot_conf_max: bool, str 
    can be `map` or `error` to visualize the matshow of prediction 
    and errors 
crossvalp_kws: dict 
    crossvalpredict additional keywords arguments 
conf_mx_kws: dict 
    Additional confusion matrix keywords arguments.

Returns 
---------
obj: object, an instancied metric tying object 
    The metric object hold the following attributes additional to the return
    attributes from :func:~.confusion_matrix_`:: 
        * `conf_mx` returns the score computed between `y_true` and `y_pred`. 
        * `norm_conf_mx` returns the normalized values. 
    and can be retrieved for plot purpose.    
    
Note 
-------
:func:`~confusion_matrix_` returns a ROC object for plotting purpose. ``_`` 
is used to differentiate it with the `confusion_matrix` metric provided 
by scikit-learn. To get the `confusion_matrix` score, use  
``obj.<obj.conf_mx>`` instead.


Examples
--------
>>> from sklearn.svm import SVC 
>>> from gofast.tools.metrics import Metrics 
>>> from gofast.datasets import fetch_data 
>>> X,y = fetch_data('Bagoue dataset prepared') 
>>> svc_clf = SVC(C=100, gamma=1e-2, kernel='rbf',
...              random_state =42) 
>>> confObj =confusion_matrix_(svc_clf,X=X,y=y,
...                        plot_conf_max='error')
>>> confObj.norm_conf_mx
>>> confObj.conf_mx
>>> confObj.__dict__.keys()  
""".format(
    params =_param_docs
)


def mean_absolute_error_b(y_true, y_pred):
    """
    Calculate the Mean Absolute Error (MAE).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    Returns
    -------
    float
        The MAE value.

    Formula (ASciimath):
    -------------------
    MAE = (1 / n) * Σ |y_true - y_pred|

    Example:
    --------
    >>> y_true = np.array([3, 5, 2, 7])
    >>> y_pred = np.array([2, 6, 3, 8])
    >>> mean_absolute_error(y_true, y_pred)
    1.0
    """
    y_true, y_pred = _ensure_y_is_valid (y_true, y_pred ) 
    return np.mean(np.abs(y_true - y_pred))


def mean_squared_error(y_true, y_pred):
    """
    Calculate the Mean Squared Error (MSE).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    Returns
    -------
    float
        The MSE value.

    Formula (ASciimath):
    -------------------
    MSE = (1 / n) * Σ (y_true - y_pred)^2

    Example:
    --------
    >>> y_true = np.array([3, 5, 2, 7])
    >>> y_pred = np.array([2, 6, 3, 8])
    >>> mean_squared_error(y_true, y_pred)
    1.25
    """
    y_true, y_pred = _ensure_y_is_valid (y_true, y_pred ) 
    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    Returns
    -------
    float
        The RMSE value.

    Formula (ASciimath):
    -------------------
    RMSE = √(MSE)

    Example:
    --------
    >>> y_true = np.array([3, 5, 2, 7])
    >>> y_pred = np.array([2, 6, 3, 8])
    >>> root_mean_squared_error(y_true, y_pred)
    1.118033988749895
    """
    y_true, y_pred = _ensure_y_is_valid (y_true, y_pred ) 
    return np.sqrt(mean_squared_error(y_true, y_pred))

    
def r_squared(y_true, y_pred):
    r"""
    Calculate the Coefficient of Determination (R-squared).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    Returns
    -------
    float
        The R-squared value.

    Formula (ASciimath):
    -------------------
    R^2 = 1 - (Σ (y_true - y_pred)^2) / (Σ (y_true - mean(y_true))^2)

    Example:
    --------
    >>> y_true = np.array([3, 5, 2, 7])
    >>> y_pred = np.array([2, 6, 3, 8])
    >>> r_squared(y_true, y_pred)
    0.4210526315789472
    """
    y_true, y_pred = _ensure_y_is_valid (y_true, y_pred )
    ssr = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ssr / sst)

     
def mean_absolute_percentage_error(y_true, y_pred):
    r"""
    Calculate the Mean Absolute Percentage Error (MAPE).
    
    Measures the average of the percentage errors by which forecasts of a 
    model differ from actual values of the quantity being forecasted.
    
    Applicability: Common in various forecasting models, particularly in finance
    and operations management.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    Returns
    -------
    float
        The MAPE value.

    Formula (ASciimath):
    -------------------
    MAPE = (1 / n) * Σ |(y_true - y_pred) / y_true| * 100

    Example:
    --------
    >>> y_true = np.array([3, 5, 2, 7])
    >>> y_pred = np.array([2, 6, 3, 8])
    >>> mean_absolute_percentage_error(y_true, y_pred)
    26.190476190476193
    """
    y_true, y_pred = _ensure_y_is_valid (y_true, y_pred ) 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def explained_variance_score(y_true, y_pred):
    """
    Calculate the Explained Variance Score.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    Returns
    -------
    float
        The Explained Variance Score.

    Formula (ASciimath):
    -------------------
    Explained Variance Score = 1 - (Var(y_true - y_pred) / Var(y_true))

    Example:
    --------
    >>> y_true = np.array([3, 5, 2, 7])
    >>> y_pred = np.array([2, 6, 3, 8])
    >>> explained_variance_score(y_true, y_pred)
    0.576923076923077
    """
    y_true, y_pred = _ensure_y_is_valid (y_true, y_pred ) 
    return 1 - (np.var(y_true - y_pred) / np.var(y_true))


def median_absolute_error(y_true, y_pred):
    """
    Calculate the Median Absolute Error (MedAE).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    Returns
    -------
    float
        The Median Absolute Error (MedAE).

    Formula (ASciimath):
    -------------------
    MedAE = median(|y_true - y_pred|)

    Example:
    --------
    >>> y_true = np.array([3, 5, 2, 7])
    >>> y_pred = np.array([2, 6, 3, 8])
    >>> median_absolute_error(y_true, y_pred)
    0.5
    """
    y_true, y_pred = _ensure_y_is_valid (y_true, y_pred ) 
    return np.median(np.abs(y_true - y_pred))


def max_error(y_true, y_pred):
    """
    Calculate the Maximum Error.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    Returns
    -------
    float
        The Maximum Error.

    Formula (ASciimath):
    -------------------
    Max Error = max(|y_true - y_pred|)

    Example:
    --------
    >>> y_true = np.array([3, 5, 2, 7])
    >>> y_pred = np.array([2, 6, 3, 8])
    >>> max_error(y_true, y_pred)
    1
    """
    y_true, y_pred = _ensure_y_is_valid (y_true, y_pred ) 
    return np.max(np.abs(y_true - y_pred))


def mean_poisson_deviance(y_true, y_pred):
    """
    Calculate the Mean Poisson Deviance.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    Returns
    -------
    float
        The Mean Poisson Deviance.

    Formula (ASciimath):
    -------------------
    Mean Poisson Deviance = (1 / n) * Σ (2 * (y_pred - y_true * log(y_pred / y_true)))

    Example:
    --------
    >>> y_true = np.array([3, 5, 2, 7])
    >>> y_pred = np.array([2, 6, 3, 8])
    >>> mean_poisson_deviance(y_true, y_pred)
    0.3504668404698445
    """

    return np.mean(2 * (y_pred - y_true * np.log(y_pred / y_true)))


def mean_gamma_deviance(y_true, y_pred):
    """
    Calculate the Mean Gamma Deviance.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.

    Returns
    -------
    float
        The Mean Gamma Deviance.

    Formula (ASciimath):
    -------------------
    Mean Gamma Deviance = (1 / n) * Σ (2 * (log(y_true / y_pred) - (y_true / y_pred)))

    Example:
    --------
    >>> y_true = np.array([3, 5, 2, 7])
    >>> y_pred = np.array([2, 6, 3, 8])
    >>> mean_gamma_deviance(y_true, y_pred)
    0.09310805868940843
    """
    y_true, y_pred = _ensure_y_is_valid (y_true, y_pred ) 
    return np.mean(2 * (np.log(y_true / y_pred) - (y_true / y_pred)))


def mean_absolute_deviation(data):
    """
    Compute the Mean Absolute Deviation of a dataset.

    Parameters
    ----------
    data : array-like
        The data for which the mean absolute deviation is to be computed.

    Returns
    -------
    float
        The mean absolute deviation of the data.

    Examples
    --------
    >>> data = [1, 2, 3, 4, 5]
    >>> mean_absolute_deviation(data)
    1.2

    Notes
    -----
    MAD = \frac{1}{n} \sum_{i=1}^n |x_i - \bar{x}|
    where \bar{x} is the mean of the data, and n is the number of observations.
    """
    data = np.asarray(data)
    mean = np.mean(data)
    return np.mean(np.abs(data - mean))

  
def dice_similarity_coeff(y_true, y_pred):
    """
    Compute the Dice Similarity Coefficient between two boolean 1D arrays.

    Measures the similarity between two sets, often used in image segmentation 
    and binary classification tasks.
    
    Particularly useful in medical image analysis for comparing the pixel-wise 
    agreement between a ground truth segmentation and a predicted segmentation.

    Parameters
    ----------
    y_true : array-like of bool
        True labels of the data.
    y_pred : array-like of bool
        Predicted labels of the data.

    Returns
    -------
    float
        Dice Similarity Coefficient.

    Examples
    --------
    >>> y_true = [True, False, True, False, True]
    >>> y_pred = [True, True, True, False, False]
    >>> dice_similarity_coefficient(y_true, y_pred)
    0.6

    Notes
    -----
    DSC = 2 * (|y_true ∩ y_pred|) / (|y_true| + |y_pred|)
    """
    y_true, y_pred = _ensure_y_is_valid (y_true, y_pred ) 
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    intersection = np.sum(y_true & y_pred)
    return 2. * intersection / (np.sum(y_true) + np.sum(y_pred))

def gini_coeff(y_true, y_pred):
    """
    Compute the Gini Coefficient, a measure of inequality among values.

    A measure of statistical dispersion intended to represent the income or 
    wealth distribution of a nation's residents.
    
    Widely used in economics for inequality measurement, but also applicable 
    in machine learning for assessing inequality in error distribution
    
    Parameters
    ----------
    y_true : array-like
        Observed values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        Gini Coefficient.

    Examples
    --------
    >>> y_true = [1, 2, 3, 4, 5]
    >>> y_pred = [2, 2, 3, 4, 4]
    >>> gini_coefficient(y_true, y_pred)
    0.2

    Notes
    -----
    G = \frac{\sum_i \sum_j |y_{\text{true},i} - y_{\text{pred},j}|}{2n\sum_i y_{\text{true},i}}
    """
    y_true, y_pred = _ensure_y_is_valid (y_true, y_pred ) 
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    abs_diff = np.abs(np.subtract.outer(y_true, y_pred))
    return np.sum(abs_diff) / (2 * len(y_true) * np.sum(y_true))

def hamming_loss(y_true, y_pred):
    """
    Compute the Hamming loss, the fraction of labels that are 
    incorrectly predicted.

    Measures the fraction of wrong labels to the total number of labels.
    Useful in multi-label classification problems, such as text 
    categorization or image classification where each instance might 
    have multiple labels.

    Parameters
    ----------
    y_true : array-like
        True labels of the data.
    y_pred : array-like
        Predicted labels of the data.

    Returns
    -------
    float
        Hamming loss.

    Examples
    --------
    >>> y_true = [1, 2, 3, 4]
    >>> y_pred = [2, 2, 3, 4]
    >>> hamming_loss(y_true, y_pred)
    0.25

    Notes
    -----
    HL = \frac{1}{n} \sum_{i=1}^n 1(y_{\text{true},i} \neq y_{\text{pred},i})
    """
    y_true, y_pred = _ensure_y_is_valid (y_true, y_pred ) 
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(y_true != y_pred)

def fowlkes_mallows_index(y_true, y_pred):
    """
    Compute the Fowlkes-Mallows Index for clustering 
    performance.
    
    A measure of similarity between two sets of clusters.
    
    Used in clustering and image segmentation to evaluate the similarity 
    between the actual and predicted clusters.

    Parameters
    ----------
    y_true : array-like
        True cluster labels.
    y_pred : array-like
        Predicted cluster labels.

    Returns
    -------
    float
        Fowlkes-Mallows Index.

    Examples
    --------
    >>> y_true = [1, 1, 2, 2, 3, 3]
    >>> y_pred = [1, 1, 1, 2, 3, 3]
    >>> fowlkes_mallows_index(y_true, y_pred)
    0.7717

    Notes
    -----
    FMI = \sqrt{\frac{TP}{TP + FP} \times \frac{TP}{TP + FN}}
    """
    y_true, y_pred = _ensure_y_is_valid (y_true, y_pred ) 
    cm = confusion_matrix(y_true, y_pred)
    tp = np.sum(np.diag(cm))  # True Positives
    fp = np.sum(cm, axis=0) - np.diag(cm)  # False Positives
    fn = np.sum(cm, axis=1) - np.diag(cm)  # False Negatives
    return np.sqrt(np.sum(tp / (tp + fp)) * np.sum(tp / (tp + fn)))

def rmse_log_error(y_true, y_pred):
    """
    Compute the Root Mean Squared Logarithmic Error.

    Provides a measure of accuracy in predicting quantitative data where 
    the emphasis is on the relative rather than the absolute difference.
    
    Often used in forecasting and regression problems, especially when 
    dealing with exponential growth, like in population studies or viral 
    growth modeling.
    
    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        Root Mean Squared Logarithmic Error.

    Examples
    --------
    >>> y_true = [3, 5, 2.5, 7]
    >>> y_pred = [2.5, 5, 4, 8]
    >>> root_mean_squared_log_error(y_true, y_pred)
    0.1993

    Notes
    -----
    RMSLE = \sqrt{\frac{1}{n} \sum_{i=1}^n (\log(y_{\text{pred},i} + 1) - \log(y_{\text{true},i} + 1))^2}
    """
    y_true, y_pred = _ensure_y_is_valid (y_true, y_pred ) 
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))

def mean_percentage_error(y_true, y_pred):
    """
    Compute the Mean Percentage Error.
    
    Measures the average of the percentage errors by which forecasts of a 
    model differ from actual values of the quantity being forecasted.
    
    Applicability: Common in various forecasting models, particularly in 
    finance and operations management.
    

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        Mean Percentage Error.

    Examples
    --------
    >>> y_true = [100, 200, 300]
    >>> y_pred = [110, 190, 295]
    >>> mean_percentage_error(y_true, y_pred)
    -1.6667

    Notes
    -----
    MPE = \frac{100}{n} \sum_{i=1}^n \frac{y_{\text{pred},i} - y_{\text{true},i}}{y_{\text{true},i}}
    """
    y_true, y_pred = _ensure_y_is_valid (y_true, y_pred ) 
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_pred - y_true) / y_true) * 100


def percentage_bias(y_true, y_pred):
    """
    Compute the Percentage Bias between true and predicted values.

    Indicates the average tendency of the predictions to overestimate or 
    underestimate against actual values.
    
    Used in forecasting models, such as in weather forecasting,
    economic forecasting, or any model where the direction 
    of bias is crucial.
    
    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        The percentage bias of the predictions.

    Examples
    --------
    >>> y_true = [100, 150, 200, 250, 300]
    >>> y_pred = [110, 140, 210, 230, 310]
    >>> percentage_bias(y_true, y_pred)
    1.3333

    Notes
    -----
    Percentage Bias = \frac{100}{n} \sum_{i=1}^n \frac{y_{\text{pred},i} - y_{\text{true},i}}{y_{\text{true},i}}
    """
    y_true, y_pred = _ensure_y_is_valid (y_true, y_pred ) 
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return 100 * np.sum((y_pred - y_true) / y_true) / len(y_true)

def spearmans_rank_correlation(y_true, y_pred):
    """
    Compute Spearman's Rank Correlation Coefficient, a nonparametric 
    measure of rank correlation.

    Parameters
    ----------
    y_true : array-like
        True rankings.
    y_pred : array-like
        Predicted rankings.

    Returns
    -------
    float
        Spearman's Rank Correlation Coefficient.

    Examples
    --------
    >>> y_true = [1, 2, 3, 4, 5]
    >>> y_pred = [5, 6, 7, 8, 7]
    >>> spearmans_rank_correlation(y_true, y_pred)
    0.8208

    Notes
    -----
    \rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}
    where d_i is the difference between the two ranks of each observation, 
    and n is the number of observations.
    """
    y_true, y_pred = _ensure_y_is_valid (y_true, y_pred ) 
    return spearmanr(y_true, y_pred)[0]

def precision_at_k(y_true, y_pred, k):
    """
    Compute Precision at K for ranking problems.
    
    Measures the proportion of relevant items found in the top-k recommendations.
    
    Widely used in information retrieval and recommendation systems, like in 
    search engine result ranking or movie recommendation.
    
    Parameters
    ----------
    y_true : list of list of int
        List of lists containing the true relevant items.
    y_pred : list of list of int
        List of lists containing the top-k predicted items.
    k : int
        The rank at which precision is evaluated.

    Returns
    -------
    float
        Precision at K.

    Examples
    --------
    >>> y_true = [[1, 2], [1, 2, 3]]
    >>> y_pred = [[2, 3, 4], [2, 3, 5]]
    >>> k = 2
    >>> precision_at_k(y_true, y_pred, k)
    0.75

    Notes
    -----
    P@K = \frac{1}{|U|} \sum_{u=1}^{|U|} \frac{|{ \text{relevant items at k for user } u } \cap { \text{recommended items at k for user } u }|}{k}
    """
    assert len(y_true) == len(y_pred),(
        "Length of true and predicted lists must be equal.")
    y_true, y_pred = _ensure_y_is_valid (y_true, y_pred ) 
    precision_scores = []
    for true, pred in zip(y_true, y_pred):
        num_relevant = len(set(true) & set(pred[:k]))
        precision_scores.append(num_relevant / k)
    
    return np.mean(precision_scores)

def ndcg_at_k(y_true, y_pred, k):
    """
    Compute Normalized Discounted Cumulative Gain at K for 
    ranking problems.
    
    Evaluates the quality of rankings by considering the position of the 
    relevant items.
    
    Crucial in search engines, recommender systems, and any other system 
    where the order of predictions is important.

    Parameters
    ----------
    y_true : list of list of int
        List of lists containing the true relevant items with their grades.
    y_pred : list of list of int
        List of lists containing the predicted items.
    k : int
        The rank at which NDCG is evaluated.

    Returns
    -------
    float
        NDCG at K.

    Examples
    --------
    >>> y_true = [[3, 2, 3], [2, 1, 2]]
    >>> y_pred = [[1, 2, 3], [1, 2, 3]]
    >>> k = 3
    >>> ndcg_at_k(y_true, y_pred, k)
    0.9203

    Notes
    -----
    DCG@K = \sum_{i=1}^k \frac{2^{rel_i} - 1}{\log_2(i + 1)}
    NDCG@K = \frac{DCG@K}{IDCG@K}
    where rel_i is the relevance of the item at position i.
    """
    def dcg_at_k(rel, k):
        rel = np.asfarray(rel)[:k]
        discounts = np.log2(np.arange(2, rel.size + 2))
        return np.sum(rel / discounts)
    
    y_true, y_pred = _ensure_y_is_valid (y_true, y_pred ) 
    ndcg_scores = []
    for true, pred in zip(y_true, y_pred):
        idcg = dcg_at_k(sorted(true, reverse=True), k)
        dcg = dcg_at_k([true[pred.index(d)] if d in pred else 0 for d in pred], k)
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0)
    
    return np.mean(ndcg_scores)

def mean_reciprocal_rank(y_true, y_pred):
    """
    Compute Mean Reciprocal Rank for ranking problems.

    Averages the reciprocal ranks of the first correct answer in a list of 
    predictions.
    
    Applicability: Commonly used in information retrieval and natural 
    language processing, particularly for evaluating query response 
    systems
    
    Parameters
    ----------
    y_true : list of int
        List containing the true relevant item.
    y_pred : list of list of int
        List of lists containing the predicted ranked items.

    Returns
    -------
    float
        Mean Reciprocal Rank.

    Examples
    --------
    >>> y_true = [1, 2]
    >>> y_pred = [[1, 2, 3], [1, 3, 2]]
    >>> mean_reciprocal_rank(y_true, y_pred)
    0.75

    Notes
    -----
    MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank of first relevant item for query } i}
    """
    y_true, y_pred = _ensure_y_is_valid (y_true, y_pred ) 
    reciprocal_ranks = []
    for true, preds in zip(y_true, y_pred):
        rank = next((1 / (i + 1) for i, pred in enumerate(preds) if pred == true), 0)
        reciprocal_ranks.append(rank)
    
    return np.mean(reciprocal_ranks)

def average_precision(y_true, y_pred):
    """
    Compute Average Precision for binary classification problems.

    Measures the average precision of a classifier at different threshold 
    levels.
    
    Widely used in binary classification tasks and ranking problems in 
    information retrieval, like document retrieval and object 
    detection in images.
    
    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred : array-like
        Predicted probabilities.

    Returns
    -------
    float
        Average Precision.

    Examples
    --------
    >>> y_true = [0, 1, 0, 1]
    >>> y_pred = [0.1, 0.4, 0.35, 0.8]
    >>> average_precision(y_true, y_pred)
    0.8333

    Notes
    -----
    AP = \sum_{k=1}^n P(k) \Delta r(k)
    where P(k) is the precision at cutoff k, and \Delta r(k) is the change 
    in recall from items k-1 to k.
    """
    y_true, y_pred = _ensure_y_is_valid (y_true, y_pred ) 
    sorted_indices = np.argsort(y_pred)[::-1]
    y_true_sorted = np.asarray(y_true)[sorted_indices]

    tp = np.cumsum(y_true_sorted)
    fp = np.cumsum(~y_true_sorted)
    precision_at_k = tp / (tp + fp)

    return np.sum(precision_at_k * y_true_sorted) / np.sum(y_true_sorted)

def jaccard_similarity_coeff(y_true, y_pred):
    """
    Compute the Jaccard Similarity Coefficient for binary 
    classification.
    
    Measures the similarity and diversity of sample sets.
    
    Useful in many fields including ecology, gene sequencing analysis, and 
    also in machine learning for evaluating the accuracy of clustering.
    
    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred : array-like
        Predicted binary labels.

    Returns
    -------
    float
        Jaccard Similarity Coefficient.

    Examples
    --------
    >>> y_true = [1, 1, 0, 0]
    >>> y_pred = [1, 0, 0, 1]
    >>> jaccard_similarity_coefficient(y_true, y_pred)
    0.3333

    Notes
    -----
    J = \frac{|y_{\text{true}} \cap y_{\text{pred}}|}{|y_{\text{true}} \cup y_{\text{pred}}|}
    """
    y_true, y_pred = _ensure_y_is_valid (y_true, y_pred ) 
    
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    return intersection.sum() / float(union.sum())


def _ensure_y_is_valid(y_true, y_pred, **kwargs):
    """
    Validates that the true and predicted target arrays are suitable for further
    processing. This involves ensuring that both arrays are non-empty, of the
    same length, and meet any additional criteria specified by keyword arguments.

    Parameters
    ----------
    y_true : array-like
        The true target values.
    y_pred : array-like
        The predicted target values.
    **kwargs : dict
        Additional keyword arguments to pass to the check_y function for any
        extra validation criteria.

    Returns
    -------
    y_true : array-like
        Validated true target values.
    y_pred : array-like
        Validated predicted target values.

    Raises
    ------
    ValueError
        If the validation checks fail, indicating that the input arrays do not
        meet the required criteria for processing.

    Examples
    --------
    Suppose `check_y` validates that the input is a non-empty numpy array and
    `check_consistent_length` ensures the arrays have the same number of elements.
    Then, usage could be as follows:

    >>> y_true = np.array([1, 2, 3])
    >>> y_pred = np.array([1.1, 2.1, 3.1])
    >>> y_true_valid, y_pred_valid = _ensure_y_is_valid(y_true, y_pred)
    >>> print(y_true_valid, y_pred_valid)
    [1 2 3] [1.1 2.1 3.1]
    """
    # Convert y_true and y_pred to numpy arrays if they are not already
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Ensure individual array validity
    y_true = check_y(y_true, **kwargs)
    y_pred = check_y(y_pred, **kwargs)

    # Check if the arrays have consistent lengths
    check_consistent_length(y_true, y_pred)

    return y_true, y_pred

    
    