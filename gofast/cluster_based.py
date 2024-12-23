# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Provides cluster-based estimators for both classification
and regression tasks. These estimators leverage clustering algorithms
to enhance predictive performance by capturing underlying data
structures [1]_. By adhering to the scikit-learn API, these estimators
ensure compatibility and ease of integration into existing machine
learning workflows.

The available cluster-based estimators include:
- `KMFClassifier`: A classifier that incorporates K-Means clustering
  to improve classification accuracy by modeling cluster centroids.
- `KMFRegressor`: A regressor that utilizes K-Means clustering to
  enhance regression performance by considering cluster-specific
  patterns.

Importing Estimators
--------------------
With the new import strategy, you can import these estimators directly
from the `gofast` package without navigating through the `estimators`
subpackage. This simplifies the import statements and enhances code
readability.

Examples
--------
Below is an example demonstrating the usage of the `KMFClassifier`:

    >>> from gofast.cluster_based import KMFClassifier
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.metrics import accuracy_score

    >>> # Generate a synthetic classification dataset
    >>> X, y = make_classification(
    ...     n_samples=1000,
    ...     n_features=20,
    ...     n_informative=15,
    ...     n_redundant=5,
    ...     random_state=42
    ... )

    >>> # Split the dataset into training and testing sets
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.3, random_state=42
    ... )

    >>> # Initialize the KMFClassifier with desired parameters
    >>> kmf_classifier = KMFClassifier(
    ...     n_clusters=5,
    ...     max_iter=300,
    ...     tol=1e-4,
    ...     random_state=42,
    ...     verbose=True
    ... )

    >>> # Fit the classifier to the training data
    >>> kmf_classifier.fit(X_train, y_train)

    >>> # Make predictions on the test set
    >>> y_pred = kmf_classifier.predict(X_test)

    >>> # Evaluate the classifier's performance
    >>> accuracy = accuracy_score(y_test, y_pred)
    >>> print('Accuracy:', accuracy)

Additional Functionalities
--------------------------
- **Customizable Number of Clusters**: Adjust the `n_clusters` parameter
  to capture varying data complexities.
- **Integration with Different Distance Metrics**: Utilize various
  distance metrics to suit specific clustering needs.
- **Support for Feature Scaling and Preprocessing**: Incorporate
  feature scaling within the estimator to enhance clustering performance.
- **Verbose Mode**: Enable verbose output to monitor the clustering
  process and convergence.

References
----------
.. [1] Kouadio, K.L., Liu, J., Liu, R., Wang, Y., Liu, W., 2024.
       K-Means Featurizer: A booster for intricate datasets. Earth Sci.
       Informatics 17, 1203–1228. https://doi.org/10.1007/s12145-024-01236-3
.. [2] MacQueen, J. (1967). Some Methods for Classification and Analysis of
      Multivariate Observations. Proceedings of the Fifth Berkeley
      Symposium on Mathematical Statistics and Probability, 1, 281-297.
.. [3] Kaufman, L., & Rousseeuw, P. J. (2009). Finding Groups in Data:
      An Introduction to Cluster Analysis. Wiley Series in Probability
      and Statistics.
"""

from gofast.estimators.cluster_based import (
    KMFClassifier,
    KMFRegressor
)

__all__ = [
    "KMFClassifier",
    "KMFRegressor"
]
