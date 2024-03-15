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
import itertools 
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
    log_loss, 
    mean_absolute_error, 
    r2_score
    )

from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import cross_val_predict 
from sklearn.preprocessing import label_binarize

from ._docstring import DocstringComponents,_core_docs
from ._gofastlog import gofastlog
# from ._typing import _F, List, Optional, ArrayLike , NDArray 
# from .exceptions import LearningError 
from .tools.box import Bunch 
from .tools.coreutils import normalize_string 
from .tools.mathex import determine_epsilon, calculate_binary_iv
# from .tools.validator import get_estimator_name, _is_numeric_dtype
from .tools.validator import check_consistent_length, check_y  
from .tools.validator import check_classification_targets, check_is_fitted

# from .tools.coreutils import is_iterable, _assert_all_types 

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
    "roc_tradeoff",
    "evaluate_confusion_matrix", 
    "mean_squared_log_error",
    "balanced_accuracy",
    "information_value", 
    "flexible_mae",
    "flexible_mse", 
    "flexible_rmse",
    "flexible_r2", 
    "mean_absolute_percentage_error", 
    "explained_variance_score", 
    "median_absolute_error",
    "max_error",
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
    "geo_information_value", 
    "assess_regression_metrics", 
    "assess_classifier_metrics", 
    "adjusted_r2_score", 
    "display_confusion_matrix", 
    "display_roc", 
    "display_precision_recall"
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

def get_metrics(): 
    """
    Get the list of  available gofast metrics. 
    
    Metrics are measures of quantitative assessment commonly used for 
    assessing, comparing, and tracking performance or production. Generally,
    a group of metrics will typically be used to build a dashboard that
    management or analysts review on a regular basis to maintain performance
    assessments, opinions, and business strategies.
    """
    return tuple(metrics.SCORERS.keys())

def mean_squared_log_error(y_true, y_pred, *,  clip_value=0, epsilon=1e-15):
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
    epsilon='auto', 
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
        specifies the fixed number of bins to use. Default is ``'auto'``.

    bins_method : str, optional
        Specifies the method to determine the optimal number of bins when 
        `bins` is set to ``'auto'``. Available methods are:
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

def geo_information_value(
        Xp, Np, Sp, *, aggregate=True, epsilon=1e-15, clip_upper_bound=None):
    """
    Calculate the Geographic Information Value (Geo-IV) for assessing the influence
    of various factors on landslide occurrences. Geo-IV quantifies the sensitivity 
    and relevance of each factor in predicting spatial events such as landslides.

    This metric implements the Information Value (InV) method, a statistical 
    approach that evaluates the predictive power of spatial parameters and their 
    relationships with landslide occurrences. The method is particularly useful in 
    geosciences for landslide susceptibility mapping and risk assessment.

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
    clip_upper_bound : float, optional
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
    >>> from gofast.metrics import geo_information_value
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
    
    # Apply clipping with epsilon as lower bound and clip_upper_bound as upper bound
    Np = np.clip(Np, epsilon, clip_upper_bound if clip_upper_bound is not None else np.max(Np))
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
    >>> from gofast.metrics import assess_regression_metrics
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
    scores = Bunch(
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

def assess_classifier_metrics(
    y_true, y_pred=None, X=None, model=None, *,
    average="binary", multi_class="raise", 
    normalize=True, sample_weight=None, 
    **scorer_kws):
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
    >>> from gofast.metrics import assess_classifier_metrics
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
    scores = Bunch(
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

def adjusted_r2_score(
        y_true, y_pred, X, sample_weight=None, epsilon=1e-7, zero_division="warn"):
    """
    Calculate the adjusted R-squared score, a modification of the R-squared
    score that accounts for the number of predictors in the model.

    The adjusted R-squared increases only if the new term improves the model
    more than would be expected by chance. It decreases when a predictor improves
    the model by less than expected by chance. It is thus especially useful for
    comparing models with different numbers of independent variables.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.
    y_pred : array-like of shape (n_samples,)
        Estimated target values.
    X : array-like of shape (n_samples, n_features)
        Input features matrix.
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    epsilon : float, default=1e-7
        A small value added to the denominator to prevent division by zero.
    zero_division : {"warn", "error"} or numeric, default="warn"
        Defines how to handle the case when the calculation denominator is zero:
        - "warn": issues a warning and returns `np.nan`
        - "error": raises a `ZeroDivisionError`
        - numeric: returns this value

    Returns
    -------
    float
        The adjusted R-squared score.

    Notes
    -----
    The adjusted R-squared is calculated as:

    .. math::
        1 - (1-R^2)\\frac{n-1}{n-p-1}

    where:
    - :math:`R^2` is the coefficient of determination,
    - :math:`n` is the number of samples,
    - :math:`p` is the number of independent variables.

    Examples
    --------
    >>> from gofast.metrics import adjusted_r2_score
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> X = [[1], [2], [3], [4]]
    >>> adjusted_r2_score(y_true, y_pred, X)
    0.948...

    References
    ----------
    - Wikipedia entry for the Coefficient of determination:
      https://en.wikipedia.org/wiki/Coefficient_of_determination

    See Also
    --------
    r2_score : R^2 (coefficient of determination) regression score function.
    mean_squared_error : Mean squared error regression loss.
    """
    # Implementation details skipped for brevity

    # Ensure X is a numeric numpy array
    if not isinstance(X, np.ndarray):
        X = np.asarray(X, dtype=float)
    
    n, p = X.shape  # Number of observations and independent variables
    y_true, y_pred = _ensure_y_is_valid(y_true, y_pred, y_numeric=True)
    r2 = r2_score(y_true, y_pred, sample_weight=sample_weight)
    
    denominator = n - p - 1
    
    if denominator <= 0:
        if zero_division == "warn":
            warnings.warn(
                "The calculation of the adjusted R-squared involves division"
                " by zero or a negative number, which can lead to overfitting."
                " Result may not be reliable.", UserWarning)
            adjusted_r2 = np.nan # Return NaN to indicate the calculation isn't reliable
        elif zero_division == "error":
            raise ZeroDivisionError(
                "The calculation of the adjusted R-squared involves division"
                " by zero or a negative number.")
        elif isinstance(zero_division, (int, float)):
            adjusted_r2 = zero_division # Custom Fallback Value:
        else:
            raise ValueError("zero_division must be 'warn', 'error', or a numeric value.")
    else:
        # Adjusted R-squared calculation
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / max(denominator, epsilon)
    
    return adjusted_r2

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
    unique_labels = np.unique(y)

    # Validate binary nature of 'y'
    if len(unique_labels) != 2:
        if target_class is None:
            raise ValueError(
                "Precision-recall metrics require binary classification, but "
                f"{len(unique_labels)} unique labels were found. Please specify "
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

        if target_class not in unique_labels:
            raise ValueError(
                f"Specified 'target_class' ({target_class}) does"
                " not exist in the target labels."
            )
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
    >>> from gofast.metrics import precision_recall_tradeoff
    >>> X, y = make_classification(n_samples=1000, n_features=20,
    ...                            n_classes=2, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    >>> clf = LogisticRegression()
    >>> clf.fit(X_train, y_train)
    >>> precision_recall_tradeoff(y_test, X=X_test, estimator=clf,
    ...                           scoring_method='predict_proba',
    ...                           display_chart=True)

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
    metrics = Bunch(
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
    >>> from gofast.metrics import display_precision_recall
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
    import matplotlib.pyplot as plt

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

def roc_tradeoff(
    y_true, 
    y_scores=None, 
    X=None, 
    estimator=None, 
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
    estimator : estimator object implementing 'fit', optional
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
    >>> X, y = make_classification(n_samples=1000, n_features=20,
    ...                            n_classes=2, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
    ...                                                     random_state=42)
    >>> clf = RandomForestClassifier(random_state=42)
    >>> clf.fit(X_train, y_train)
    >>> evaluate_roc(y_test, estimator=clf, X=X_test, display_chart=True)
    
    This will display the ROC curve for the RandomForestClassifier on the test data.
    """
    if y_scores is None:
        if X is None or estimator is None:
            raise ValueError("When 'y_scores' is None, both 'X' and"
                             " 'estimator' must be provided.")
        # Check if the estimator is fitted and predict the scores
        check_is_fitted(estimator)
        if hasattr(estimator, "decision_function"):
            y_scores = cross_val_predict(
                estimator, X, y_true, cv=cv, method="decision_function", **cv_kwargs)
        elif hasattr(estimator, "predict_proba"):
            y_scores = cross_val_predict(
                estimator, X, y_true, cv=cv, method="predict_proba", **cv_kwargs)[:, 1]
        elif hasattr ( estimator, 'predict'): 
            y_scores = cross_val_predict(
                estimator, X, y_true, cv=cv, **cv_kwargs)
        else:
            raise ValueError("Estimator must have a 'decision_function',"
                             " 'predict_proba' or 'predict' method .")

    y_true, y_scores = _ensure_y_is_valid(y_true, y_scores, y_numeric=True)

    fpr, tpr, thresholds = roc_curve(
        y_true, y_scores, pos_label=pos_label, sample_weight=sample_weight)
    auc_score = roc_auc_score(y_true, y_scores, sample_weight=sample_weight)

    if display_chart:
        # Example of plotting
        display_roc (fpr, tpr, auc_score)

    scores = Bunch(auc_score=auc_score, fpr=fpr, tpr=tpr, thresholds=thresholds)

    return y_scores if return_scores else scores

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
    import matplotlib.pyplot as plt

    # Validate inputs
    fpr, tpr, auc_score = map(np.asarray, [fpr, tpr, auc_score])
    if not (fpr.ndim == tpr.ndim == 1):
        raise ValueError("fpr and tpr must be 1-dimensional arrays.")
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

def evaluate_confusion_matrix(
    y_true, 
    y_pred=None, 
    classifier=None, 
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
    classifier : object implementing 'fit', optional
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
    >>> X, y = make_classification(
    ...    n_samples=1000, n_features=4, n_classes=2, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...    X, y, test_size=0.25, random_state=42)
    >>> clf = RandomForestClassifier(random_state=42)
    >>> clf.fit(X_train, y_train)
    >>> evaluate_confusion_matrix(
    ...    y_test, classifier=clf, X=X_test, display=True, normalize=True)
    
    This will output a Bunch object containing the confusion matrix and display
    the normalized confusion matrix.

    See Also
    --------
    display_confusion_matrix : Function to visualize the confusion matrix.
    sklearn.metrics.confusion_matrix : The confusion matrix computation.
    """

    if y_pred is None:
        if not (X is not None and classifier is not None):
            raise ValueError("Provide 'y_pred' or both 'X' and 'classifier'.")
        check_is_fitted(classifier, 'predict')
        y_pred = cross_val_predict(classifier, X, y_true, cv=cv, **cv_kwargs)
    
    y_true, y_pred =_ensure_y_is_valid(y_true, y_pred, y_numeric=True )
    cm = confusion_matrix(y_true, y_pred, labels=labels, sample_weight=sample_weight)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        np.fill_diagonal(cm, 0)

    if display:
        display_confusion_matrix(cm, labels=labels, cmap=cmap, normalize=normalize)

    return Bunch(confusion_matrix=cm)

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
    import matplotlib.pyplot as plt
 
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


def flexible_mae(
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
    >>> from gofast.metrics import flexible_mae
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> result = flexible_mae(y_true, y_pred)
    >>> print(result.MAE)
    0.5
    >>> result_detailed = flexible_mae(y_true, y_pred, detailed=True)
    >>> print(result_detailed.MAE, result_detailed.min_error)
    0.5 0.0
    """
    # Ensure y_true and y_pred are valid and have the same shape
    y_true, y_pred = _ensure_y_is_valid(y_true, y_pred, y_numeric=True)
    
    if str(epsilon).lower() == 'auto':
        # Assuming this function determines a suitable epsilon value
        epsilon = determine_epsilon(y_pred)  
    
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
    
    result = Bunch(MAE=mae)
    
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

def flexible_mse(
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
    flexible_mae : A flexible version of Mean Absolute Error.
    mean_squared_log_error : Mean Squared Logarithmic Error metric.
    mean_absolute_error : Mean Absolute Error metric.

    Examples
    --------
    >>> from gofast.metrics import flexible_mse
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> result = flexible_mse(y_true, y_pred)
    >>> print(result.MSE)
    0.375
    >>> result_detailed = flexible_mse(y_true, y_pred, detailed=True)
    >>> print(result_detailed.MSE, result_detailed.min_error)
    0.375 0.0
    """
    # Ensure y_true and y_pred are valid and have the same shape
    y_true, y_pred = _ensure_y_is_valid(y_true, y_pred, y_numeric=True)
    
    if str(epsilon).lower() == 'auto':
        # Assuming this function determines a suitable epsilon value
        epsilon = determine_epsilon(y_pred)  
    
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
    
    result = Bunch(MSE=mse)
    
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
        
        result.scaled_MSE = np.mean(scaled_squared_errors)
        if detailed:
            result.min_scaled_error = np.min(
                scaled_squared_errors, initial=np.inf, where=y_true != 0)
            result.max_scaled_error = np.max(
                scaled_squared_errors, initial=-np.inf, where=y_true != 0)
            result.std_scaled_error = np.std(
                scaled_squared_errors, where=y_true != 0)
    return result

def flexible_rmse(
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
    flexible_mae : A flexible version of Mean Absolute Error.
    flexible_mse : A flexible version of Mean Squared Error.
    sklearn.metrics.mean_squared_error : Mean Squared Error metric.

    Examples
    --------
    >>> from gofast.metrics import flexible_rmse
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> result = flexible_rmse(y_true, y_pred)
    >>> print(result.RMSE)
    0.612...
    >>> result_detailed = flexible_rmse(y_true, y_pred, detailed=True)
    >>> print(result_detailed.RMSE, result_detailed.min_error)
    0.612... 0.0
    """
    # Ensure y_true and y_pred are valid and have the same shape
    y_true, y_pred = _ensure_y_is_valid(y_true, y_pred, y_numeric=True)
    
    if str(epsilon).lower() == 'auto':
        # Assuming this function determines a suitable epsilon value
        epsilon = determine_epsilon(y_pred)  
    
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
    
    result = Bunch(RMSE=rmse)
    
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

def flexible_r2(
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
    flexible_mae : Mean Absolute Error with flexibility.
    flexible_mse : Mean Squared Error with flexibility.
    flexible_rmse : Root Mean Squared Error with flexibility.

    Examples
    --------
    >>> from gofast.metrics import flexible_r2
    >>> y_true = np.array([3, 5, 2, 7])
    >>> y_pred = np.array([2.5, 0.5, 2, 8])
    >>> result = flexible_r2(y_true, y_pred)
    >>> print(result.R2)
    -0.4576271186440677...
    >>> result_adjusted = flexible_r2(y_true, y_pred, adjust_for_n=True, n_predictors=1)
    >>> print(result_adjusted.adjusted_R2)
    -1.1864406779661016...
    """
    # Ensure y_true and y_pred are valid and have the same shape
    y_true, y_pred = _ensure_y_is_valid(y_true, y_pred, y_numeric=True)
    
    # Validation for epsilon
    if not isinstance(epsilon, (int, float)):
        raise ValueError("epsilon must be 'auto' or a numeric value.")
    
    # Calculate R-squared
    ssr = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ssr / max (sst, epsilon)) # Prevent division by zero
    
    result = Bunch(R2=r2)
    
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
            result.adjusted_R2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
            
    elif adjust_for_n and n_predictors is None:
        warnings.warn("Number of predictors must be specified to adjust R2.")
    
    return result

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

    
    