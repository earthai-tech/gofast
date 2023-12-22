# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
# metric_bases.pyx 

"""
@author: Daniel
"""
import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, pow

cpdef double rmse(double[:] predictions, double[:] targets):
    """
    Compute the Root Mean Square Error (RMSE) between predictions and targets.

    Parameters
    ----------
    predictions : ndarray
        Predicted values.
    targets : ndarray
        Actual target values.

    Returns
    -------
    double
        The computed RMSE value.
    """
    cdef int i
    cdef double error_sum = 0.0
    cdef int n = predictions.shape[0]

    for i in range(n):
        error_sum += (predictions[i] - targets[i]) ** 2

    return (error_sum / n) ** 0.5

cpdef double precision(double[:] true_positives, double[:] false_positives):
    """
    Compute the precision of a classification model.

    Parameters
    ----------
    true_positives : ndarray
        True positive counts for each class.
    false_positives : ndarray
        False positive counts for each class.

    Returns
    -------
    double
        The computed precision value.
    """
    cdef double total_true_positives = np.sum(true_positives)
    cdef double total_false_positives = np.sum(false_positives)

    if total_true_positives + total_false_positives == 0:
        return 0

    return total_true_positives / (total_true_positives + total_false_positives)

cpdef double recall(double[:] true_positives, double[:] false_negatives):
    """
    Compute the recall of a classification model.

    Parameters
    ----------
    true_positives : ndarray
        True positive counts for each class.
    false_negatives : ndarray
        False negative counts for each class.

    Returns
    -------
    double
        The computed recall value.
    """
    cdef double total_true_positives = np.sum(true_positives)
    cdef double total_false_negatives = np.sum(false_negatives)

    if total_true_positives + total_false_negatives == 0:
        return 0

    return total_true_positives / (total_true_positives + total_false_negatives)

cpdef double f1_score(double precision, double recall):
    """
    Compute the F1 score from precision and recall.

    Parameters
    ----------
    precision : double
        The precision of the model.
    recall : double
        The recall of the model.

    Returns
    -------
    double
        The computed F1 score.
    """
    if precision + recall == 0:
        return 0

    return 2 * (precision * recall) / (precision + recall)

cpdef double mean_absolute_error(double[:] predictions, double[:] targets):
    """
    Compute the Mean Absolute Error (MAE) between predictions and targets.

    Parameters
    ----------
    predictions : ndarray
        Predicted values.
    targets : ndarray
        Actual target values.

    Returns
    -------
    double
        The computed MAE value.
    """
    cdef int i
    cdef double error_sum = 0.0
    cdef int n = predictions.shape[0]

    for i in range(n):
        error_sum += abs(predictions[i] - targets[i])

    return error_sum / n

cpdef double log_loss(double[:] predictions, double[:] targets):
    """
    Compute the logistic loss (log loss) between predictions and targets.

    Parameters
    ----------
    predictions : ndarray
        Predicted probabilities.
    targets : ndarray
        Actual binary target values.

    Returns
    -------
    double
        The computed log loss.
    """
    cdef int i
    cdef double loss_sum = 0.0
    cdef int n = predictions.shape[0]
    cdef double epsilon = 1e-15

    for i in range(n):
        pred = max(min(predictions[i], 1 - epsilon), epsilon)
        loss_sum -= (targets[i] * np.log(pred) + (1 - targets[i]) * np.log(1 - pred))

    return loss_sum / n


