# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Provides a collection of ensemble estimators for both classification
and regression tasks. Ensemble methods combine multiple individual models to
produce a more robust and accurate overall model. By following the scikit-learn
API, these estimators ensure compatibility and ease of integration into existing
machine learning workflows.

The available ensemble estimators include:
- `MajorityVoteClassifier`: Combines multiple classifiers using majority voting.
- `SimpleAverageClassifier`: A classifier that averages the predictions of base classifiers.
- `SimpleAverageRegressor`: A regressor that averages the predictions of base regressors.
- `WeightedAverageClassifier`: A classifier that averages predictions with assigned weights.
- `WeightedAverageRegressor`: A regressor that averages predictions with assigned weights.
- `EnsembleClassifier`: A flexible ensemble classifier supporting various combination strategies.
- `EnsembleRegressor`: A flexible ensemble regressor supporting various combination strategies.

Importing Estimators
--------------------
With the new import strategy, you can import these estimators directly from the
`gofast` package without navigating through the `estimators` subpackage. This
simplifies the import statements and enhances code readability.

Examples
--------
Below is an example demonstrating the usage of the `MajorityVoteClassifier`:

    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.model_selection import cross_val_score, train_test_split
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.impute import SimpleImputer
    >>> from gofast.datasets import fetch_data
    >>> from gofast.ensemble import MajorityVoteClassifier
    >>> from gofast.utils.base_utils import select_features
    >>> from sklearn.metrics import roc_auc_score

    >>> # Load and preprocess the dataset
    >>> data = fetch_data('bagoue original').frame
    >>> X0 = data.iloc[:, :-1]
    >>> y0 = data['flow'].values

    >>> # Binarize the target variable
    >>> y = np.asarray([0 if x <= 1 else 1 for x in y0])

    >>> # Select numerical features and handle missing values
    >>> X = select_features(X0, include='number')
    >>> X = SimpleImputer().fit_transform(X)

    >>> # Split the dataset into training and testing sets
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.3, random_state=42
    ... )

    >>> # Initialize individual classifiers with pipelines
    >>> clf1 = LogisticRegression(penalty='l2', solver='lbfgs', random_state=42)
    >>> clf2 = DecisionTreeClassifier(max_depth=1, random_state=42)
    >>> clf3 = KNeighborsClassifier(p=2, n_neighbors=1)

    >>> pipe1 = Pipeline([
    ...     ('scaler', StandardScaler()),
    ...     ('classifier', clf1)
    ... ])
    >>> pipe3 = Pipeline([
    ...     ('scaler', StandardScaler()),
    ...     ('classifier', clf3)
    ... ])

    >>> # Define classifier labels
    >>> clf_labels = ['Logistic Regression', 'Decision Tree', 'K-Nearest Neighbors']

    >>> # Evaluate individual classifiers using cross-validation
    >>> for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
    ...     scores = cross_val_score(clf, X, y, cv=10, scoring='roc_auc')
    ...     print("ROC AUC: %.2f (+/- %.2f) [%s]" % (scores.mean(), scores.std(), label))
    ROC AUC: 0.91 (+/- 0.05) [Logistic Regression]
    ROC AUC: 0.73 (+/- 0.07) [Decision Tree]
    ROC AUC: 0.77 (+/- 0.09) [K-Nearest Neighbors]

    >>> # Implement and evaluate the MajorityVoteClassifier
    >>> mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])
    >>> clf_labels += ['Majority Voting']
    >>> all_classifiers = [pipe1, clf2, pipe3, mv_clf]

    >>> for clf, label in zip(all_classifiers, clf_labels):
    ...     scores = cross_val_score(clf, X, y, cv=10, scoring='roc_auc')
    ...     print("ROC AUC: %.2f (+/- %.2f) [%s]" % (scores.mean(), scores.std(), label))
    ROC AUC: 0.91 (+/- 0.05) [Logistic Regression]
    ROC AUC: 0.73 (+/- 0.07) [Decision Tree]
    ROC AUC: 0.77 (+/- 0.09) [K-Nearest Neighbors]
    ROC AUC: 0.92 (+/- 0.06) [Majority Voting]

Additional Functionalities
--------------------------
- **Customizable Classifier Sets**: 
    Combine any number of classifiers to form an ensemble.
- **Flexible Voting Strategies**:
    Implement different voting mechanisms such as majority voting 
    or weighted voting.
- **Support for Both Classification and Regression**: 
    Utilize ensemble methods for a variety of predictive tasks.
- **Ease of Integration**: 
    Designed to seamlessly integrate with scikit-learn pipelines and utilities.

References
----------
* Breiman, L. (1996). Bagging predictors. Machine Learning, 24(2), 123-140.
* Dietterich, T.G. (2000). Ensemble Methods in Machine Learning. In
  Multiple Classifier Systems, Lecture Notes in Computer Science, vol 1857,
  pp. 1-15.
"""

from gofast.estimators.ensemble import (
    MajorityVoteClassifier,
    SimpleAverageClassifier,
    SimpleAverageRegressor,
    WeightedAverageClassifier,
    WeightedAverageRegressor,
    EnsembleClassifier,
    EnsembleRegressor
)

__all__ = [
    "MajorityVoteClassifier",
    "SimpleAverageRegressor",
    "SimpleAverageClassifier",
    "WeightedAverageRegressor",
    "WeightedAverageClassifier",
    "EnsembleClassifier",
    "EnsembleRegressor"
]

