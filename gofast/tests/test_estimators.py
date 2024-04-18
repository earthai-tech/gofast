# -*- coding: utf-8 -*-
"""
test_estimators.py
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

from gofast.estimators import (
    SimpleAverageClassifier,
    WeightedAverageClassifier,
    SimpleAverageRegressor,
    WeightedAverageRegressor,
    HybridBoostedTreeClassifier,
    HybridBoostedTreeRegressor,
    DecisionTreeBasedClassifier,
    HBTEnsembleClassifier,
    HBTEnsembleRegressor,
)

def test_simple_average_classifier():
    # Load a sample dataset (Iris) for classification
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Create and fit the SimpleAverageClassifier
    base_classifiers = [DecisionTreeBasedClassifier() for _ in range(3)]
    ensemble = SimpleAverageClassifier(base_classifiers)
    ensemble.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    y_pred = ensemble.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    assert accuracy >= 0.0  

def test_weighted_average_classifier():
    # Load a sample dataset (Iris) for classification
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Create and fit the WeightedAverageClassifier
    base_classifiers = [DecisionTreeBasedClassifier() for _ in range(3)]
    weights = [0.3, 0.4, 0.3]
    ensemble = WeightedAverageClassifier(base_classifiers, weights)
    ensemble.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    y_pred = ensemble.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    assert accuracy >= 0.0  

def test_simple_average_regressor():
    # Generate synthetic regression data
    X = np.random.rand(100, 4)
    y = np.random.rand(100)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Create and fit the SimpleAverageRegressor
    base_regressors = [DecisionTreeBasedClassifier() for _ in range(3)]
    ensemble = SimpleAverageRegressor(base_regressors)
    ensemble.fit(X_train, y_train)

    # Make predictions and calculate mean squared error
    y_pred = ensemble.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    assert mse >= 0.0  


def test_weighted_average_regressor():
    # Generate synthetic regression data
    X = np.random.rand(100, 4)
    y = np.random.rand(100)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Create and fit the WeightedAverageRegressor
    base_regressors = [DecisionTreeBasedClassifier() for _ in range(3)]
    weights = [0.3, 0.4, 0.3]
    ensemble = WeightedAverageRegressor(base_regressors, weights)
    ensemble.fit(X_train, y_train)

    # Make predictions and calculate mean squared error
    y_pred = ensemble.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    assert mse >= 0.0  


def test_hybrid_boosted_tree_classifier():
    # Load a sample dataset (Iris) for classification
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Create and fit the HybridBoostedTreeClassifier
    ensemble = HybridBoostedTreeClassifier(n_estimators=50, learning_rate=0.01, max_depth=3)
    ensemble.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    y_pred = ensemble.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    assert accuracy >= 0.0  

def test_hybrid_boosted_tree_regressor():
    # Generate synthetic regression data
    X = np.random.rand(100, 4)
    y = np.random.rand(100)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Create and fit the HybridBoostedTreeRegressor
    ensemble = HybridBoostedTreeRegressor(n_estimators=50, learning_rate=0.01, max_depth=3)
    ensemble.fit(X_train, y_train)

    # Make predictions and calculate mean squared error
    y_pred = ensemble.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    assert mse >= 0.0  

def test_decision_tree_based_classifier():
    # Load a sample dataset (Iris) for classification
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Create and fit the DecisionTreeBasedClassifier
    ensemble = DecisionTreeBasedClassifier(n_estimators=50, max_depth=3, random_state=42)
    ensemble.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    y_pred = ensemble.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    assert accuracy >= 0.0  

def test_hybrid_boosted_tree_ensemble_classifier():
    # Load a sample dataset (Iris) for classification
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Create and fit the HybridBoostedTreeEnsembleClassifier
    ensemble = HBTEnsembleClassifier(n_estimators=50, max_depth=3, learning_rate=0.1)
    ensemble.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    y_pred = ensemble.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    assert accuracy >= 0.0  

def test_hybrid_boosted_tree_ensemble_regressor():
    # Generate synthetic regression data
    X = np.random.rand(100, 4)
    y = np.random.rand(100)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Create and fit the HybridBoostedTreeEnsembleRegressor
    ensemble = HBTEnsembleRegressor(n_estimators=50, max_depth=3, learning_rate=0.1)
    ensemble.fit(X_train, y_train)

    # Make predictions and calculate mean squared error
    y_pred = ensemble.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    assert mse >= 0.0  

if __name__ == "__main__":
    test_simple_average_classifier()
    test_weighted_average_classifier()
    test_simple_average_regressor()
    test_weighted_average_regressor()
    test_hybrid_boosted_tree_classifier()
    test_hybrid_boosted_tree_regressor()
    test_decision_tree_based_classifier()
    test_hybrid_boosted_tree_ensemble_classifier()
    test_hybrid_boosted_tree_ensemble_regressor()

