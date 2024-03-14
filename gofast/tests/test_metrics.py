# -*- coding: utf-8 -*-
"""
test_metrics.py

@author: LKouadio ~@Daniel
"""
import numpy as np
import pytest
from gofast.metrics import mean_squared_log_error, balanced_accuracy
from gofast.metrics import information_value 

def test_mean_squared_log_error_basic():
    y_true = np.array([3, 5, 2.5, 7])
    y_pred = np.array([2.5, 5, 4, 8])
    expected_msle = 0.03973  
    msle = mean_squared_log_error(y_true, y_pred)
    assert np.isclose(msle, expected_msle, atol=1e-4), f"Expected {expected_msle}, got {msle}"

def test_mean_squared_log_error_with_clip():
    y_true = [3, 5, 2.5, 1]
    y_pred = [2.5, 5, 4, -8] # this is simulation that not fit the reality for non-negative values. 
    msle = mean_squared_log_error(y_true, y_pred, clip_value=0.01)
    expected_msle = 0.15295
    assert np.isclose(msle, expected_msle, atol=1e-4), f"Expected {expected_msle}, got {msle}"

def test_mean_squared_log_error_negative_values():
    y_true = [3, 5, 2.5, -7]
    y_pred = [2, -5, 4, 8]
    with pytest.raises(ValueError):
        mean_squared_log_error(y_true, y_pred)

def test_balanced_accuracy_binary():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 0]
    expected_ba = 0.75
    ba = balanced_accuracy(y_true, y_pred)
    assert np.isclose(ba, expected_ba, atol=1e-4), f"Expected {expected_ba}, got {ba}"

def test_balanced_accuracy_multiclass_ovr():
    y_true = [0, 1, 2, 2, 1]
    y_pred = [0, 2, 2, 1, 1]
    expected_ba = 0.666  
    ba = balanced_accuracy(y_true, y_pred, strategy='ovr')
    assert np.isclose(ba, expected_ba, atol=1e-2), f"Expected {expected_ba}, got {ba}"

def test_balanced_accuracy_invalid_strategy():
    y_true = [0, 1, 2, 2, 1]
    y_pred = [0, 2, 2, 1, 1]
    with pytest.raises(ValueError):
        balanced_accuracy(y_true, y_pred, strategy='invalid')

def test_information_value_binary_classification():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0.1, 0.8, 0.7, 0.2])
    iv = information_value(y_true, y_pred, problem_type='binary', scale=None)
    print(iv)
    assert isinstance(iv, float), "IV should be a float"
    assert iv > 0, "IV should be positive for binary classification with these inputs"

def test_information_value_multiclass_classification():
    y_true = np.array([0, 1, 2, 1])
    y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3], [0.2, 0.2, 0.6], [0.1, 0.7, 0.2]])
    iv = information_value(y_true, y_pred, problem_type='multiclass', scale='binary_scale')
    assert isinstance(iv, float), "IV should be a float"
    assert iv < 0, "IV should be negative for multiclass classification with these inputs and binary_scale"

def test_information_value_multilabel_classification():
    y_true = np.array([[1, 0], [0, 1]])
    y_pred = np.array([[0.8, 0.2], [0.4, 0.6]])
    iv = information_value(y_true, y_pred, problem_type='multilabel', scale='none')
    print(iv)
    assert isinstance(iv, float), "IV should be a float"
    assert iv > 0, "IV should be positive for multilabel classification with"
    " these inputs and binary_scale"


def test_information_value_binary_classification2():
    y_true_binary = np.array([0, 1, 0, 1, 1, 0])
    y_pred_binary = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3])
    iv = information_value(y_true_binary, y_pred_binary, problem_type='binary', )
    assert isinstance(iv, float), "IV should be a float"
    assert iv > 0, "IV should be positive for binary classification with these inputs"

def test_information_value_multilabel_classification2():
    y_true_multilabel = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1],
        [0, 0, 1]
    ])
    y_pred_multilabel = np.array([
        [0.8, 0.1, 0.7],
        [0.2, 0.9, 0.2],
        [0.6, 0.2, 0.8],
        [0.1, 0.3, 0.9]
    ])
    iv = information_value(y_true_multilabel, y_pred_multilabel,
                           problem_type='multilabel', scale='binary_scale')
    assert isinstance(iv, float), "IV should be a float"
    # Here, the expectation for IV being positive or negative depends 
    # on  interpretation.
    # If we expect IV to reflect predictive power similarly across problem
    # types, consider if the IV calculation
    # logic inversely correlates the IV value with predictive power
    # for multilabel classification.
    assert iv > 0, ( "IV should be positive for multilabel classification"
                    " with these inputs and binary_scale")


def test_information_value_regression():
    y_true = np.array([2.5, 0.0, 2.0, 8.0])
    y_pred = np.array([3.0, 0.5, 2.0, 7.5])
    iv = information_value(y_true, y_pred, problem_type='regression')
    assert isinstance(iv, float), "IV should be a float"
    assert iv < 0, "IV should be negative for regression with these inputs"

def test_information_value_invalid_problem_type():
    y_true = [0, 1]
    y_pred = [0.5, 0.5]
    with pytest.raises(ValueError):
        information_value(y_true, y_pred, problem_type='invalid_type')


if __name__=='__main__': 
    pytest.main([__file__])