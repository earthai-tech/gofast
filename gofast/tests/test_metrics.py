# -*- coding: utf-8 -*-
"""
test_metrics.py

@author: LKouadio ~@Daniel
"""
import numpy as np
import pytest
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import precision_recall_curve, confusion_matrix
from sklearn.linear_model import LogisticRegression

from gofast.metrics import assess_classifier_metrics 
from gofast.metrics import assess_regression_metrics  
from gofast.metrics import mean_squared_log_error, balanced_accuracy
from gofast.metrics import information_value, geo_information_value 
from gofast.metrics import mae_flex, mse_flex, rmse_flex, r2_flex
from gofast.metrics import adjusted_r2_score  
from gofast.metrics import precision_recall_tradeoff, roc_tradeoff
from gofast.metrics import evaluate_confusion_matrix, display_precision_recall
from gofast.metrics import display_confusion_matrix  

@pytest.fixture
def binary_classification_data():
    X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)

@pytest.fixture
def trained_model(binary_classification_data):
    X_train, _, y_train, _ = binary_classification_data
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def test_precision_recall_tradeoff(binary_classification_data, trained_model):
    _, X_test, _, y_test = binary_classification_data
    # Assuming y_scores are provided
    y_scores = trained_model.decision_function(X_test)
    result = precision_recall_tradeoff(y_test, y_scores=y_scores, display_chart=False)
    assert 'f1_score' in result and 'precision_score' in result and 'recall_score' in result,( 
        "Required metrics are missing")

def test_roc_tradeoff(binary_classification_data, trained_model):
    _, X_test, _, y_test = binary_classification_data
    y_scores = trained_model.decision_function(X_test)
    result = roc_tradeoff(y_test, y_scores=y_scores, display_chart=False)
    assert 'auc_score' in result, "AUC score missing from results"

def test_evaluate_confusion_matrix(binary_classification_data, trained_model):
    _, X_test, _, y_test = binary_classification_data
    result = evaluate_confusion_matrix(y_test, classifier=trained_model, X=X_test, display=False)
    assert 'confusion_matrix' in result, "Confusion matrix missing from results"

def test_display_precision_recall_does_not_raise_error():
    precisions, recalls, thresholds = precision_recall_curve([0, 1], [0.1, 0.9])
    try:
        display_precision_recall(precisions, recalls, thresholds)
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")

def test_display_confusion_matrix_does_not_raise_error():
    cm = confusion_matrix([0, 1], [1, 1])
    try:
        display_confusion_matrix(cm, labels=['Neg', 'Pos'])
    except Exception as e:
        pytest.fail(f"Unexpected error occurred: {e}")

def test_geo_information_value_basic():
    Xp = np.array([1, 2, 3])
    Np = np.array([100, 200, 300])
    Sp = np.array([10, 20, 30])
    geo_iv = geo_information_value(Xp, Np, Sp, aggregate=True)
    assert isinstance(geo_iv, float)
    assert geo_iv >= 0, "Geo-IV should be positive for this input"

def test_geo_information_value_aggregate_false():
    Xp = np.array([1, 2, 3])
    Np = np.array([100, 200, 300])
    Sp = np.array([10, 20, 30])
    geo_iv = geo_information_value(Xp, Np, Sp, aggregate=False)
    assert len(geo_iv) == len(Xp), "The output should match the number of causative factors"
    assert all(iv >= 0 for iv in geo_iv), "All Geo-IV values should be positive for this input"

def test_geo_information_value_clip_upper_bound():
    Xp = np.array([1, 2])
    Np = np.array([1e-16, 200])  # Extremely small value that should be clipped
    Sp = np.array([1, 20])
    geo_iv = geo_information_value(Xp, Np, Sp, clip_upper_bound=100, aggregate=True)
    assert isinstance(geo_iv, float), "Output should be a float when aggregate is True"

def test_adjusted_r2_score_basic():
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    X = np.random.rand(4, 2)  # Assuming 2 predictors
    adjusted_r2 = adjusted_r2_score(y_true, y_pred, X)
    basic_r2 = r2_score(y_true, y_pred)
    assert adjusted_r2 < basic_r2, "Adjusted R² should be less than R² for this example"

def test_adjusted_r2_score_negative_denominator():
    y_true = np.array([1, 2])
    y_pred = np.array([1.1, 1.9])
    X = np.random.rand(2, 5)  # More predictors than samples
    with pytest.warns(UserWarning):
        adjusted_r2 = adjusted_r2_score(y_true, y_pred, X)
    assert np.isnan(adjusted_r2), "Adjusted R² should be NaN when denominator is negative or zero"

def test_assess_regression_metrics():
    X, y = make_regression(n_samples=100, n_features=4, noise=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    metrics = assess_regression_metrics(y_test, model=model, X=X_test)
    # return Bunch object 
    assert 'mean_absolute_error' in metrics, "Dictionary should contain MAE"
    assert 'mean_squared_error' in metrics, "Dictionary should contain MSE"
    assert 'rmse' in metrics, "Dictionary should contain RMSE"
    assert 'r2_score' in metrics, "Dictionary should contain R² score"
    assert metrics.mean_absolute_error >= 0, "MAE should be non-negative"
    assert metrics.mean_squared_error >= 0, "MSE should be non-negative"
    assert metrics.rmse >= 0, "RMSE should be non-negative"
    assert -1 <= metrics.r2_score <= 1, "R² score should be within [-1, 1]"

def test_assess_classifier_metrics():
    X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    metrics = assess_classifier_metrics(y_test, model=model, X=X_test, average='macro')

    assert 'accuracy_score' in metrics, "Dictionary should contain accuracy score"
    assert 'recall_score' in metrics, "Dictionary should contain recall score"
    assert 'precision_score' in metrics, "Dictionary should contain precision score"
    assert 'f1_score' in metrics, "Dictionary should contain F1 score"
    # ROC AUC might be skipped in multiclass scenarios without predict_proba
    # assert 'roc_auc_score' in metrics, "Dictionary should contain ROC-AUC score"
    assert 0 <= metrics.accuracy_score <= 1, "Accuracy score should be within [0, 1]"
    assert 0 <= metrics.recall_score <= 1, "Recall score should be within [0, 1]"
    assert 0 <= metrics.precision_score <= 1, "Precision score should be within [0, 1]"
    assert 0 <= metrics.f1_score <= 1, "F1 score should be within [0, 1]"

# Sample data for testing
y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])
def test_flexible_mae():
    result = mae_flex(y_true, y_pred)
    assert 'MAE' in result
    assert result.MAE >= 0  # MAE should always be non-negative

def test_flexible_mse():
    result = mse_flex(y_true, y_pred)
    assert 'MSE' in result
    assert result.MSE>= 0  # MSE should always be non-negative

def test_flexible_rmse():
    result = rmse_flex(y_true, y_pred)
    assert 'RMSE' in result
    assert result.RMSE >= 0  # RMSE should always be non-negative

def test_flexible_r2():
    result = r2_flex(y_true, y_pred)
    assert  'R2' in result
    assert -1 <= result.R2 <= 1  # R2 should be in the range [-1, 1]

    # Test with adjustment for the number of predictors
    result_adjusted = r2_flex(y_true, y_pred, adjust_for_n=True, n_predictors=1)
    assert 'adjusted_R2' in result_adjusted
    # adjusted R2 should also be in the range [-1, 1]
    assert -1 <= result_adjusted.adjusted_R2 <= 1 

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