# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Provides benchmark estimators for both regression
and classification tasks. These estimators follow the
scikit-learn API, allowing you to systematically compare
different base estimators and optionally combine them with a
meta-estimator to improve performance.

Examples
--------
Below is an example demonstrating the usage of the
BenchmarkRegressor with various base estimators:

    >>> from gofast.benchmark import BenchmarkRegressor
    >>> from sklearn.datasets import fetch_california_housing
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> from sklearn.neighbors import KNeighborsRegressor

    >>> # Load the California Housing dataset
    >>> housing = fetch_california_housing()
    >>> X, y = housing.data, housing.target
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.3
    ... )

    >>> # Define base estimators and meta-estimator
    >>> base_estimators = [
    ...     ('lr', LinearRegression()),
    ...     ('dt', DecisionTreeRegressor()),
    ...     ('knn', KNeighborsRegressor())
    ... ]
    >>> meta_regressor = LinearRegression()

    >>> # Initialize and fit the benchmark model
    >>> benchmark_reg = BenchmarkRegressor(
    ...     base_estimators=base_estimators,
    ...     meta_regressor=meta_regressor
    ... )
    >>> benchmark_reg.fit(X_train, y_train)

    >>> # Predict and evaluate
    >>> y_pred = benchmark_reg.predict(X_test)
    >>> print('R^2 Score:', benchmark_reg.score(X_test, y_test))
"""

from gofast.estimators.benchmark import (
    BenchmarkRegressor, BenchmarkClassifier
)
from gofast.estimators.base import (
    StumpClassifier, StumpRegressor
)

__all__ = [
    "BenchmarkRegressor",
    "BenchmarkClassifier",
    "StumpRegressor",
    "StumpClassifier",
]
