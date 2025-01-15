# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Provides tree-based estimators for both classification
and regression tasks. Tree-based methods are powerful and versatile
machine learning algorithms that can capture complex patterns in data.
By adhering to the scikit-learn API, these estimators ensure compatibility
and ease of integration into existing machine learning workflows.

The available tree-based estimators include:
- `DTBRegressor`: A Decision Tree Boosting regressor for robust regression tasks.
- `DTBClassifier`: A Decision Tree Boosting classifier for robust classification tasks.
- `WeightedTreeClassifier`: A classifier that assigns weights to individual trees.
- `WeightedTreeRegressor`: A regressor that assigns weights to individual trees.

Importing Estimators
--------------------
One can import these estimators directly from the
`gofast` package without navigating through the `estimators` subpackage. This
simplifies the import statements and enhances code readability.

Examples
--------
Below is an example demonstrating the usage of the `DTBRegressor`:

    >>> import numpy as np
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.model_selection import train_test_split
    >>> from gofast.tree import DTBRegressor
    >>> from sklearn.metrics import mean_squared_error

    >>> # Generate synthetic regression data
    >>> X, y = make_regression(
    ...     n_samples=1000,
    ...     n_features=20,
    ...     noise=0.1,
    ...     random_state=42
    ... )

    >>> # Split the dataset into training and testing sets
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.3, random_state=42
    ... )

    >>> # Initialize the DTBRegressor with desired parameters
    >>> dtb_regressor = DTBRegressor(
    ...     n_estimators=50,
    ...     max_depth=3,
    ...     learning_rate=0.1,
    ...     random_state=42,
    ...     verbose=True
    ... )

    >>> # Fit the regressor to the training data
    >>> dtb_regressor.fit(X_train, y_train)

    >>> # Make predictions on the test set
    >>> y_pred = dtb_regressor.predict(X_test)

    >>> # Evaluate the regressor's performance
    >>> mse = mean_squared_error(y_test, y_pred)
    >>> print('Mean Squared Error:', mse)
    Mean Squared Error: 0.0123

Additional Functionalities
--------------------------
- **Customizable Number of Estimators**: Adjust the `n_estimators` parameter to control
  the number of trees in the ensemble.
- **Learning Rate Adjustment**: Modify the `learning_rate` to balance the contribution
  of each tree.
- **Depth Control**: Set the `max_depth` parameter to prevent overfitting by limiting
  the depth of individual trees.
- **Weighted Trees**: Assign different weights to trees to emphasize their importance
  in the ensemble.
- **Verbose Output**: Enable verbose mode to monitor the training progress and
  convergence of the boosting process.

References
----------
* Friedman, J. H. (2001). Greedy function approximation: A gradient boosting
  machine. Annals of Statistics, 29(5), 1189-1232.
* Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.
* Schapire, R. E. (1990). The strength of weak learnability. Machine Learning,
  5(2), 197-227.
"""

from gofast.tree import (
    DTBRegressor,
    DTBClassifier,
    WeightedTreeClassifier,
    WeightedTreeRegressor
)

__all__ = [
    "DTBRegressor",
    "DTBClassifier",
    "WeightedTreeClassifier",
    "WeightedTreeRegressor",
]
