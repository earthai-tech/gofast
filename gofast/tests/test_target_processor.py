# -*- coding: utf-8 -*-
import pytest
import numpy as np
import pandas as pd
import importlib.util
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from gofast.base import TargetProcessor
from gofast.tools.funcutils import reshape 
# Sample data for testing
y_binary = np.array([0, 1, 0, 1])
y_multiclass = np.array(['A', 'B', 'C', 'A'])
y_multilabel = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
X_sample = pd.DataFrame({'feature1': np.random.rand(4), 
                         'feature2': np.random.rand(4)})

# Setup a fixture for repeated use of a processor instance, if needed
@pytest.fixture
def sample_processor():
    return TargetProcessor()
# Function to check if skmultilearn is installed
def is_skmultilearn_installed():
    return importlib.util.find_spec("skmultilearn") is not None

# Test initialization
def test_init():
    processor = TargetProcessor()
    assert processor.tnames is None
    assert processor.verbose is False

# Test fit method
def test_fit():
    processor = TargetProcessor()
    processor.fit(y_binary)
    assert np.array_equal(processor.target_, y_binary)

# Test label encoding
def test_label_encode():
    processor = TargetProcessor()
    processor.fit(y_multiclass).label_encode()
    assert np.array_equal(processor.target_, LabelEncoder().fit_transform(y_multiclass))

# Test one hot encoding
def test_one_hot_encode():
    processor = TargetProcessor()
    processor.fit(y_multiclass).one_hot_encode()
    assert np.array_equal(processor.target_, OneHotEncoder(
        sparse_output=False).fit_transform(y_multiclass.reshape(-1, 1)))

# Test binarization
def test_binarize():
    processor = TargetProcessor()
    threshold = 0.5
    processor.fit(y_binary).binarize(threshold)

    expected_output = np.array([0, 1, 0, 1])  # Based on threshold
    assert np.array_equal(reshape (processor.target_), expected_output)

# Test calculating metrics
def test_calculate_metrics():
    processor = TargetProcessor()
    processor.fit(y_binary)
    y_pred = np.array([0, 1, 1, 0])
    metrics = processor.calculate_metrics(y_pred)
    # Check if metrics contain expected keys and values
    assert 'accuracy' in metrics and 'precision' in metrics and 'recall' in metrics and 'f1' in metrics


# Test cost-sensitive learning adjustment
def test_cost_sensitive_learning():
    processor = TargetProcessor()
    processor.fit(y_binary)
    # Assuming there's a method to adjust for cost-sensitive learning
    # This would require specific implementation details to test effectively
    processor.adjust_for_cost_sensitive_learning()
    # Verify the effect of cost-sensitive learning (e.g., class weights adjusted)

# Test feature correlation with target
def test_feature_correlation_with_target():
    processor = TargetProcessor()
    X = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'target': np.random.choice([0, 1], 100)
    })
    processor.fit('target', X=X)
    # Assuming there's a method to calculate feature correlation with target
    correlations = processor.analyze_feature_correlation_with_target(X)
    assert isinstance(correlations, dict)
    assert 'feature1' in correlations and 'feature2' in correlations

# Test multi-label transformation techniques
@pytest.mark.skipif(not is_skmultilearn_installed(), 
                    reason="skmultilearn is required for this test")
def test_multilabel_transformation():
    processor = TargetProcessor()
    processor.fit(y_multilabel, X=X_sample)
    # Assuming there's a method for binary relevance transformation
    processor.transform_multi_label()
    # Validate the transformed target
    # Similar tests can be written for classifier chains and label powerset
    

# Test threshold tuning
def test_threshold_tuning():
    processor = TargetProcessor()
    processor.fit(y_binary, X=X_sample)
    new_threshold = 0.6
    # Assuming there's a method to adjust the decision threshold
    yscores= np.abs ( np.random.normal(size = len(y_binary)))
    processor.threshold_tuning(yscores, new_threshold)
    # Validate if the adjustment has been made correctly

# Test visualization
def test_visualization():
    processor = TargetProcessor()
    processor.fit(y_binary, X=X_sample)
    # Assuming there's a method to plot target distribution
    plot = processor.visualization()
    assert plot is not None  # A more detailed check might be needed based on the plotting library used


# Sample data for testing
y_regression = np.random.rand(100)  # continuous target for regression
y_imbalanced = np.random.choice([0, 1], 100, p=[0.1, 0.9])  # imbalanced binary target

# Test normalization/standardization (for regression targets)
def test_normalize_standardize():
    processor = TargetProcessor()
    processor.fit(y_regression)
    # Assuming there are methods for normalize and standardize
    processor.scale_target(method='normalize')
    # Check if normalization was applied correctly
    assert np.allclose(processor.target_.min(), 0)
    assert np.allclose(processor.target_.max(), 1)

    processor.scale_target()
    # Check if standardization was applied correctly
    assert np.isclose(processor.target_.mean(), 0, atol=0.1)
    assert np.isclose(processor.target_.std(), 1, atol=0.1)

# Test handling imbalanced data
def test_balance_data():
    processor = TargetProcessor()
    processor.fit(y_imbalanced)
    X_sample = pd.DataFrame({'feature1': np.random.rand(100), 'feature2': np.random.rand(100)})

    # Assuming there's a method balance_data to handle imbalanced target
    processor.balance_data(X_sample, method='smote')  # Example method: SMOTE
    # Verify that the class distribution is more balanced
    unique, counts = np.unique(processor.target_, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    # Check if the class distribution is more balanced after applying SMOTE
    assert len(class_distribution.keys()) > 1  # More than one class exists
    balance_ratio = min(class_distribution.values()) / max(class_distribution.values())
    assert balance_ratio > 0.5  # Assuming a more balanced distribution


if __name__ == "__main__":
    pytest.main([__file__])
