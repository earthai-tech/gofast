# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Provides boosting estimators that adhere to the scikit-learn API,
enabling the creation of powerful ensemble models by sequentially building
strong learners from weaker ones. Boosting techniques focus on converting
weak learners into strong learners by emphasizing the training of subsequent
models based on the errors of previous ones. This module supports both
regression and classification tasks through various specialized estimators.

The available boosting estimators include:
- `BoostedTreeRegressor`: A gradient boosting regressor using decision trees.
- `BoostedTreeClassifier`: A gradient boosting classifier using decision trees.
- `HybridBoostClassifier`: Combines multiple classifiers in a boosting framework.
- `HybridBoostRegressor`: Combines multiple regressors in a boosting framework.

Importing Estimators
--------------------
One can directly import boosting estimatorsfrom the `gofast` package, 
simplifying the import paths and improving code clarity.

Examples
--------
Below is an example demonstrating the usage of the `BoostedTreeRegressor`:

    >>> import numpy as np
    >>> from gofast.boosting import BoostedTreeRegressor
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.metrics import mean_squared_error

    >>> # Generate synthetic regression data
    >>> X, y = make_regression(n_samples=1000, n_features=20, noise=0.1, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    >>> # Initialize and fit the BoostedTreeRegressor
    >>> brt = BoostedTreeRegressor(
    ...     n_estimators=100,
    ...     learning_rate=0.1,
    ...     max_depth=3,
    ...     loss='squared_error',
    ...     subsample=0.8,
    ...     random_state=42,
    ...     verbose=True
    ... )
    >>> brt.fit(X_train, y_train)

    >>> # Make predictions
    >>> y_pred = brt.predict(X_test)

    >>> # Evaluate the model
    >>> mse = mean_squared_error(y_test, y_pred)
    >>> print('Mean Squared Error:', mse)

Additional functionalities include:
- Custom loss functions for tailored boosting objectives.
- Support for various base learners beyond decision trees.
- Early stopping based on validation performance to prevent overfitting.

References
----------
* Friedman, J. H. (2001). Greedy function approximation: A gradient boosting
  machine. Annals of Statistics, 29(5), 1189-1232.
* Schapire, R. E. (1990). The strength of weak learnability. Machine Learning, 5(2), 197-227.
"""

from gofast.estimators.boosting import (
    BoostedTreeRegressor,
    BoostedTreeClassifier,
    HybridBoostClassifier,
    HybridBoostRegressor
)

__all__ = [
    "BoostedTreeRegressor",
    "BoostedTreeClassifier",
    "HybridBoostClassifier",
    "HybridBoostRegressor",
]
