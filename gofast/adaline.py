# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Provides Adaline (Adaptive Linear Neuron) estimators
that follow the scikit-learn API. These estimators are grouped
under the 'estimators' subpackage and can be used for both
classification and regression tasks. The new import strategy
allows you to import them conveniently at a higher module level,
for instance:

    >>> from gofast.adaline import SGDAdalineRegressor

or

    >>> from gofast.adaline import AdalineClassifier

This approach avoids the extra subpackage nesting and makes
imports more concise and readable.

Examples
--------
Below is an example demonstrating the usage of the new import
strategy to train and evaluate a regression model on the
California Housing dataset:

    >>> import numpy as np
    >>> from sklearn.datasets import fetch_california_housing
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.preprocessing import StandardScaler
    >>> from gofast.adaline import SGDAdalineRegressor

    >>> # Load the California Housing dataset
    >>> housing = fetch_california_housing()
    >>> X, y = housing.data, housing.target
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.3, random_state=0
    ... )

    >>> # Standardize features
    >>> sc = StandardScaler()
    >>> X_train_std = sc.fit_transform(X_train)
    >>> X_test_std = sc.transform(X_test)

    >>> # Initialize and fit the regressor
    >>> ada_sgd_reg = SGDAdalineRegressor(
    ...     eta0=0.0001, max_iter=1000, early_stopping=True,
    ...     validation_fraction=0.1, tol=1e-4, verbose=True
    ... )
    >>> ada_sgd_reg.fit(X_train_std, y_train)

    >>> # Predict and evaluate
    >>> y_pred = ada_sgd_reg.predict(X_test_std)
    >>> print('Mean Squared Error:',
    ...       np.mean((y_pred - y_test) ** 2))

References
----------
* Widrow, B., and Hoff, M. E. (1960s). Adaptive switching circuits.
  1960 IRE WESCON Convention Record, 4, 96–104.
"""

from gofast.estimators.adaline import AdalineClassifier, AdalineRegressor
from gofast.estimators.adaline import SGDAdalineRegressor, SGDAdalineClassifier


__all__ = [
    "AdalineClassifier",
    "AdalineRegressor",
    "SGDAdalineRegressor",
    "SGDAdalineClassifier",
]

