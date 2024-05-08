# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations
from collections import defaultdict 
import inspect 
from scipy import stats
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from ..tools.validator import check_X_y, get_estimator_name, check_array 
from ..tools.validator import check_is_fitted


__all__=[ 
    "DecisionStumpRegressor","DecisionTreeBasedRegressor",
    "DecisionTreeBasedClassifier"
    ]

class DecisionTreeBasedRegressor(BaseEstimator, RegressorMixin):
    r"""
    Decision Tree-based Regression for Regression Tasks.

    The `DecisionTreeBasedRegressor` employs an ensemble approach, combining 
    multiple Decision Regression Trees to form a more robust regression model. 
    Each tree in the ensemble independently predicts the outcome, and the final 
    prediction is derived by averaging these individual predictions. This method 
    is effective in reducing variance and improving prediction accuracy over a 
    single decision tree.

    Mathematical Formulation:
    The ensemble prediction is computed by averaging the predictions from each 
    individual Regression Tree within the ensemble, as follows:

    .. math::
        y_{\text{pred}} = \frac{1}{N} \sum_{i=1}^{N} y_{\text{tree}_i}

    where:
    - :math:`N` is the number of trees in the ensemble.
    - :math:`y_{\text{pred}}` is the final predicted value aggregated from all
      trees.
    - :math:`y_{\text{tree}_i}` represents the prediction made by the \(i\)-th 
      Regression Tree.

    This ensemble approach leverages the strength of multiple learning estimators 
    to achieve better performance than what might be obtained from any single tree, 
    especially in the presence of complex data relationships and high variance in 
    the training data.

    Parameters
    ----------
    n_estimators : int
        The number of trees in the ensemble.
    max_depth : int
        The maximum depth of each regression tree.
    random_state : int
        Controls the randomness of the estimator.

    Attributes
    ----------
    estimators_ : list of DecisionTreeRegressor
        The collection of fitted sub-estimators.

    Examples
    --------
    >>> from gofast.estimators.tree.tree.tree import DecisionTreeBasedRegressor
    >>> rte = DecisionTreeBasedRegressor(
    ...     n_estimators=100, max_depth=3, random_state=42)
    >>> X, y = np.random.rand(100, 4), np.random.rand(100)
    >>> rte.fit(X, y)
    >>> y_pred = rte.predict(X)

    See Also
    --------
    - sklearn.ensemble.RandomForestRegressor: A popular ensemble method
      based on decision trees for regression tasks.
    - sklearn.tree.DecisionTreeRegressor: Decision tree regressor used as
      base learners in ensemble methods.
    - sklearn.metrics.mean_squared_error: A common metric for evaluating
      regression models.
     - gofast.estimators.tree.tree.tree.BoostedRegressionTree: An enhanced BRT

    Notes
    -----
    - The Regression Tree Ensemble is built by fitting multiple Regression
      Tree models.
    - Each tree is trained on the entire dataset, and their predictions are
      averaged to obtain the final prediction.
    """

    def __init__(self, n_estimators=100, max_depth=3, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        
    def fit(self, X, y):
        """
        Fit the Regression Tree Ensemble model to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (real numbers).

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y( X, y, estimator = get_estimator_name(self ))
        self.estimators_ = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                         random_state=self.random_state)
            tree.fit(X, y)
            self.estimators_.append(tree)

        return self

    def predict(self, X):
        """
        Predict using the Regression Tree Ensemble model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted values.
        """
        check_is_fitted(self, 'estimators_')
        
        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )
        
        predictions = np.array([tree.predict(X) for tree in self.estimators_])
        return np.mean(predictions, axis=0)

class DecisionTreeBasedClassifier(BaseEstimator, ClassifierMixin):
    r"""
    Decision Tree-Based Classifier.

    This classifier leverages an ensemble of decision trees to enhance prediction 
    accuracy and robustness over individual tree performance. It aggregates the 
    predictions from multiple DecisionTreeClassifier instances, using majority 
    voting to determine the final classification. This method is particularly 
    effective at reducing overfitting and increasing the generalization ability 
    of the model.

    The fit method trains multiple DecisionTreeClassifier models on the entire dataset. 
    The predict method then aggregates their predictions through majority voting 
    to determine the final class labels. The ensemble's behavior can be customized 
    through parameters controlling the number and depth of trees.

    Parameters
    ----------
    n_estimators : int
        The number of decision trees in the ensemble.
    max_depth : int
        The maximum depth of each tree in the ensemble.
    random_state : int, optional
        Controls the randomness of the tree building process and the bootstrap 
        sampling of the data points (if bootstrapping is used).

    Attributes
    ----------
    tree_classifiers_ : list of DecisionTreeClassifier
        A list containing the fitted decision tree classifiers.

    Notes
    -----
    The final classification decision is made by aggregating the predictions 
    from all the decision trees and selecting the class with the majority vote:

    .. math::
        C_{\text{final}}(x) = \text{argmax}\left(\sum_{i=1}^{n} \delta(C_i(x), \text{mode})\right)

    where:
    - :math:`C_{\text{final}}(x)` is the final predicted class label for input \(x\).
    - :math:`\delta(C_i(x), \text{mode})` is an indicator function that counts the 
      occurrence of the most frequent class label predicted by the \(i\)-th tree.
    - :math:`n` is the number of decision trees in the ensemble.
    - :math:`C_i(x)` is the class label predicted by the \(i\)-th decision tree.

    This ensemble approach significantly reduces the likelihood of overfitting and 
    increases the predictive performance by leveraging the diverse predictive 
    capabilities of multiple models.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of decision trees in the ensemble.
    max_depth : int, default=3
        The maximum depth of each decision tree.
    random_state : int or None, default=None
        Controls the randomness of the estimator for reproducibility.

    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    Example
    -------
    Here's an example of how to use the `DecisionTreeBasedClassifier`:

    >>> from gofast.estimators.tree.tree.tree import DecisionTreeBasedClassifier
    >>> rtec = DecisionTreeBasedClassifier(n_estimators=100, max_depth=3,
    ...                                     random_state=42)
    >>> X, y = np.random.rand(100, 4), np.random.randint(0, 2, 100)
    >>> rtec.fit(X, y)
    >>> y_pred = rtec.predict(X)

    Applications
    ------------
    The Decision Tree-Based Classifier is suitable for various classification
    tasks, such as spam detection, sentiment analysis, and medical diagnosis.

    Performance
    -----------
    The model's performance depends on the quality of the data, the number of
    estimators (trees), and the depth of each tree. Hyperparameter tuning may
    be necessary to optimize performance.

    See Also
    --------
    - `sklearn.ensemble.RandomForestClassifier`: Scikit-learn's Random Forest
      Classifier for ensemble-based classification.
    - `sklearn.tree.DecisionTreeClassifier`: Decision tree classifier used as
      base learners in ensemble methods.

    """

    def __init__(self, n_estimators=100, max_depth=3, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state 
        
        
    def fit(self, X, y):
        """
        Fit the Regression Tree Ensemble Classifier model to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target class labels.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y( X, y, estimator = get_estimator_name(self ))
        self.estimators_ = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(max_depth=self.max_depth, 
                                          random_state=self.random_state)
            tree.fit(X, y)
            self.estimators_.append(tree)

        return self

    def predict(self, X):
        """
        Predict using the Regression Tree Ensemble Classifier model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted class labels.
        """
        check_is_fitted(self, 'estimators_')
        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )
        predictions = np.array([tree.predict(X) for tree in self.estimators_])
        # Majority voting
        y_pred = stats.mode(predictions, axis=0).mode[0]
        return y_pred
    
    def predict_proba(self, X):
        """
        Predict class probabilities using the Decision Tree Based Ensemble 
        Classifier model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        proba : array-like of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        check_is_fitted(self, 'estimators_')
        X = check_array(X, accept_sparse=True)

        # Collect probabilities from each estimator
        all_proba = np.array([tree.predict_proba(X) for tree in self.estimators_])
        # Average probabilities across all estimators
        avg_proba = np.mean(all_proba, axis=0)
        return avg_proba
   
class StandardEstimator:
    """Base class for all classes in gofast for parameters retrievals

    Notes
    -----
    All class defined should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator"""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = inspect.signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "gofast classes should always "
                    "specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                    " %s with constructor %s doesn't "
                    " follow this convention." % (cls, init_signature)
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this class and
            contained sub-objects.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple classes as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                local_valid_params = self._get_param_names()
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self}. "
                    f"Valid parameters are: {local_valid_params!r}."
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self
    
class DecisionStumpRegressor (StandardEstimator):
    r"""
    A simple decision stump regressor for use in gradient boosting.

    This class implements a basic decision stump, which is a decision tree 
    with only one decision node and two leaves. The stump splits the data 
    based on the feature and threshold that minimizes the error.
    
    Mathematical Formulation
    ------------------------
    For each feature, the stump examines all possible thresholds
    (unique values of the feature). The MSE for a split at a given 
    threshold t is calculated as follows:

    .. math::
        MSE = \sum_{i \in \text{left}(t)} (y_i - \overline{y}_{\text{left}})^2 +
              \sum_{i \in \text{right}(t)} (y_i - \overline{y}_{\text{right}})^2

    where :math:`\text{left}(t)` and :math:`\text{right}(t)` are the sets of indices 
    of samples that fall to the left and right of the threshold t, respectively. 
    :math:`\overline{y}_{\text{left}}` and :math:`\overline{y}_{\text{right}}` 
    are the mean values of the target variable for the samples in each of these 
    two sets.

    The algorithm selects the feature and threshold that yield the lowest MSE.

    Attributes
    ----------
    split_feature_ : int
        Index of the feature used for the split.
    split_value_ : float
        Threshold value used for the split.
    left_value_ : float
        The value predicted for samples where the feature value is less than 
        or equal to the split value.
    right_value_ : float
        The value predicted for samples where the feature value is greater
        than the split value.

    Methods
    -------
    fit(X, y)
        Fits the decision stump to the data.
    predict(X)
        Predicts the target values for the given data.
    decision_function(X)
       Compute the raw decision scores for the given input data.
       
    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=100, n_features=1, noise=10)
    >>> stump = DecisionStumpRegressor()
    >>> stump.fit(X, y)
    >>> predictions = stump.predict(X)
    """

    def __init__(self ):
        self.split_feature=None
        self.split_value=None
        self.left_value=None
        self.right_value=None
        
    def fit(self, X, y):
        """
        Fits the decision stump to the data.

        The method iterates over all features and their unique values to 
        find the split 
        that minimizes the mean squared error.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input samples.
        y : ndarray of shape (n_samples,)
            The target values.

        Return 
        -------
        self: object 
           Return self.
        Notes
        -----
        The decision stump is a weak learner and is primarily used in
        ensemble methods like AdaBoost and Gradient Boosting.
        """
        min_error = float('inf')
        for feature in range(X.shape[1]):
            possible_values = np.unique(X[:, feature])
            for value in possible_values:
                # Create a split and calculate the mean value for each leaf
                left_mask = X[:, feature] <= value
                right_mask = X[:, feature] > value
                left_value = np.mean(y[left_mask]) if np.any(left_mask) else 0
                right_value = np.mean(y[right_mask]) if np.any(right_mask) else 0

                # Calculate the total error for this split
                error = np.sum((y[left_mask] - left_value) ** 2
                               ) + np.sum((y[right_mask] - right_value) ** 2)

                # Update the stump's split if this split is better
                if error < min_error:
                    min_error = error
                    self.split_feature = feature
                    self.split_value= value
                    self.left_value= left_value
                    self.right_value= right_value
                    
        self.fitted_=True 
        
        return self 
       
    def predict(self, X):
        """
        Predict target values for the given input data using the trained 
        decision stump.

        The prediction is based on the split learned during the fitting 
        process. Each sample in X is assigned a value based on which side 
        of the split it falls on.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input samples for which predictions are to be made.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted target values for each sample in X.

        Examples
        --------
        >>> from sklearn.datasets import make_regression
        >>> X, y = make_regression(n_samples=100, n_features=1, noise=10)
        >>> stump = DecisionStumpRegressor()
        >>> stump.fit(X, y)
        >>> predictions = stump.predict(X)
        >>> print(predictions[:5])

        Notes
        -----
        The `predict` method should be called only after the `fit` method has 
        been called. It uses the `split_feature`, `split_value`, `left_value`,
        and `right_value` attributes set by the `fit` method to make predictions.
        """
        check_is_fitted(self, "fitted_")
        # Determine which side of the split each sample falls on
        left_mask = X[:, self.split_feature] <= self.split_value
        y_pred = np.zeros(X.shape[0])
        y_pred[left_mask] = self.left_value
        y_pred[~left_mask] = self.right_value
        return y_pred

    def decision_function(self, X):
        """
        Compute the raw decision scores for the given input data.

        The decision function calculates the continuous value for each sample in X
        based on the trained decision stump. It reflects the model's degree of 
        certainty about the classification.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input samples for which decision scores are to be computed.

        Returns
        -------
        decision_scores : ndarray of shape (n_samples,)
            The raw decision scores for each sample in X.

        Examples
        --------
        >>> from sklearn.datasets import make_regression
        >>> X, y = make_regression(n_samples=100, n_features=1, noise=10)
        >>> stump = DecisionStumpRegressor()
        >>> stump.fit(X, y)
        >>> decision_scores = stump.decision_function(X)
        >>> print(decision_scores[:5])

        Notes
        -----
        This method should be called after the model has been fitted. It uses the
        attributes set during the `fit` method (i.e., `split_feature`, `split_value`,
        `left_value`, `right_value`) to compute the decision scores.
        """
        check_is_fitted(self, "fitted_")
        # Calculate the decision scores based on the split
        decision_scores = np.where(X[:, self.split_feature] <= self.split_value,
                                   self.left_value, self.right_value)
        return decision_scores

class _GradientBoostingClassifier:
    r"""
    A simple gradient boosting classifier for binary classification.

    Gradient boosting is a machine learning technique for regression and 
    classification problems, which produces a prediction model in the form 
    of an ensemble of weak prediction models, typically decision trees. It 
    builds the model in a stage-wise fashion like other boosting methods do, 
    and it generalizes them by allowing optimization of an arbitrary 
    differentiable loss function.

    Attributes
    ----------
    n_estimators : int
        The number of boosting stages to be run.
    learning_rate : float
        Learning rate shrinks the contribution of each tree.
    estimators_ : list of DecisionStumpRegressor
        The collection of fitted sub-estimators.

    Methods
    -------
    fit(X, y)
        Build the gradient boosting model from the training set (X, y).
    predict(X)
        Predict class labels for samples in X.
    predict_proba(X)
        Predict class probabilities for X.

    Mathematical Formula
    --------------------
    The model is built in a stage-wise fashion as follows:
    .. math:: 
        F_{m}(x) = F_{m-1}(x) + \\gamma_{m} h_{m}(x)

    where F_{m} is the model at iteration m, \\gamma_{m} is the step size, 
    and h_{m} is the weak learner.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=100, n_features=20, n_classes=2)
    >>> model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
    >>> model.fit(X, y)
    >>> print(model.predict(X)[:5])
    >>> print(model.predict_proba(X)[:5])

    References
    ----------
    - J. H. Friedman, "Greedy Function Approximation: A Gradient Boosting Machine," 1999.
    - T. Hastie, R. Tibshirani, and J. Friedman, "The Elements of Statistical
    Learning," Springer, 2009.

    Applications
    ------------
    Gradient Boosting can be used for both regression and classification problems. 
    It's particularly effective in scenarios where the relationship between 
    the input features and target variable is complex and non-linear. It's 
    widely used in applications like risk modeling, classification of objects,
    and ranking problems.
    """

    def __init__(self, n_estimators=100, learning_rate=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators_ = []

    def fit(self, X, y):
        """
        Fit the gradient boosting classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values (class labels).
        """
        # Convert labels to 0 and 1
        y = np.where(y == np.unique(y)[0], -1, 1)

        F_m = np.zeros(len(y))

        for m in range(self.n_estimators):
            # Compute pseudo-residuals
            residuals = -1 * y * self._sigmoid(-y * F_m)

            # Fit a decision stump to the pseudo-residuals
            stump = DecisionStumpRegressor()
            stump.fit(X, residuals)

            # Update the model
            F_m += self.learning_rate * stump.predict(X)
            self.estimators_.append(stump)

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        The predicted class probabilities are the model's confidence
        scores for the positive class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        proba : array-like of shape (n_samples, 2)
            The class probabilities for the input samples. The columns 
            correspond to the negative and positive classes, respectively.
            
        Examples 
        ----------
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=100, n_features=20, n_classes=2)
        model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
        model.fit(X, y)
        print(model.predict_proba(X)[:5])
        """
        F_m = sum(self.learning_rate * estimator.predict(X) for estimator in self.estimators_)
        proba_positive_class = self._sigmoid(F_m)
        return np.vstack((1 - proba_positive_class, proba_positive_class)).T

    def _sigmoid(self, z):
        """Compute the sigmoid function."""
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted class labels.
        """
        F_m = sum(self.learning_rate * estimator.predict(X) for estimator in self.estimators_)
        return np.where(self._sigmoid(F_m) > 0.5, 1, 0)
    




















