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
from sklearn.utils import resample

from ..api.property import BaseClass 
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
    >>> from gofast.estimators.tree import DecisionTreeBasedRegressor
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
    """
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
    
    criterion : str, default="squared_error"
        The function to measure the quality of a split. Supported criteria 
        are "squared_error" for the mean squared error, which is equal to 
        variance reduction as feature selection criterion, and "friedman_mse", 
        which uses mean squared error with Friedman's improvement score for 
        potential splits.
        
    splitter : str, default="best"
        The strategy used to choose the split at each node. Supported strategies 
        are "best" to choose the best split and "random" to choose the best 
        random split.
        
    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:
        - If int, then consider min_samples_split as the minimum number.
        - If float, then min_samples_split is a fraction and ceil
        (min_samples_split * n_samples) are the minimum number of samples 
        for each split.
        
    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node. A split 
        point at any depth will only be considered if it leaves at least 
        min_samples_leaf training samples in each of the left and right branches.
        - If int, then consider min_samples_leaf as the minimum number.
        - If float, then min_samples_leaf is a fraction and ceil
        (min_samples_leaf * n_samples) are the minimum number of samples for 
        each node.
        
    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all the input 
        samples) required to be at a leaf node. Samples have equal weight when 
        sample_weight is not provided.
        
    max_features : int, float, str or None, default=None
        The number of features to consider when looking for the best split:
        - If int, then consider max_features features at each split.
        - If float, then max_features is a fraction and int(max_features * n_features) 
          features are considered at each split.
        - If "auto", then max_features=n_features.
        - If "sqrt", then max_features=sqrt(n_features).
        - If "log2", then max_features=log2(n_features).
        - If None, then max_features=n_features.
        
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator. The features are always 
        randomly permuted at each split. When max_features < n_features, 
        the algorithm will select max_features at random at each split 
        before finding the best split among them. 
        - Pass an int for reproducible output across multiple function calls.
        
    max_leaf_nodes : int or None, default=None
        Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are
        defined as relative reduction in impurity. If None, then unlimited 
        number of leaf nodes.
        
    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity 
        greater than or equal to this value.
        
    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The 
        subtree with the largest cost complexity that is smaller than ccp_alpha
        will be chosen.
    
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

    >>> from gofast.estimators.tree import DecisionTreeBasedClassifier
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
    def __init__(
        self, n_estimators=10, 
        max_depth=None, 
        criterion='gini', 
        splitter='best', 
        min_samples_split=2, 
        min_samples_leaf=1, 
        min_weight_fraction_leaf=0.0, 
        max_features=None, 
        random_state=None, 
        max_leaf_nodes=None, 
        min_impurity_decrease=0.0, 
        class_weight=None, 
        ccp_alpha=0.0
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.criterion = criterion
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha

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
        X, y = check_X_y(
            X, y, 
            accept_sparse= True, 
            accept_large_sparse= True, 
            estimator = get_estimator_name(self )
            )
        self.estimators_ = []
        for _ in range(self.n_estimators):
            X_resampled, y_resampled = resample(
                X, y, random_state=self.random_state)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth, 
                criterion=self.criterion, 
                splitter=self.splitter, 
                min_samples_split=self.min_samples_split, 
                min_samples_leaf=self.min_samples_leaf, 
                min_weight_fraction_leaf=self.min_weight_fraction_leaf, 
                max_features=self.max_features, 
                random_state=self.random_state, 
                max_leaf_nodes=self.max_leaf_nodes, 
                min_impurity_decrease=self.min_impurity_decrease,
                class_weight=self.class_weight, 
                ccp_alpha=self.ccp_alpha
            )
            tree.fit(X_resampled, y_resampled)
            self.estimators_.append(tree)

        return self

    def predict(self, X):
        """
        Predict class for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.
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
        majority_vote = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
        # y_pred = stats.mode(predictions, axis=0).mode[0]
        return majority_vote

    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples.
        """
        check_is_fitted(self, 'estimators_')
        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )
        # Collect probabilities from each estimator
        probas = np.array([tree.predict_proba(X) for tree in self.estimators_])
        # Average probabilities across all estimators
        avg_proba = np.mean(probas, axis=0)
        return avg_proba

class WeightedTreeClassifier(BaseEstimator, ClassifierMixin):
    r"""
    Weighted Tree Classifier.

    The Weighted Tree Classifier is an ensemble learning model that 
    combines decision trees with gradient boosting techniques to tackle binary 
    classification tasks.
    
    By integrating multiple decision trees with varying weights, this classifier
    achieves high accuracy and reduces the risk of overfitting.

    Parameters
    ----------
    n_estimators : int, default=50
        The number of decision trees in the ensemble.
    eta0 : float, default=0.1
        The learning rate for gradient boosting, controlling how much each tree
        influences the overall prediction.
    max_depth : int, default=3
        The maximum depth of each decision tree, determining the complexity of
        the model.
        
    criterion : {"gini", "entropy"}, default="gini"
        The function to measure the quality of a split. Supported criteria 
        are "gini" for the Gini impurity and "entropy" for the information 
        gain.

    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node. Supported 
        strategies are "best" to choose the best split and "random" to 
        choose the best random split.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:
        - If int, then consider min_samples_split as the minimum number.
        - If float, then min_samples_split is a fraction and 
          `ceil(min_samples_split * n_samples)` are the minimum number of 
          samples for each split.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node. A 
        split point at any depth will only be considered if it leaves at 
        least `min_samples_leaf` training samples in each of the left and 
        right branches.
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and 
          `ceil(min_samples_leaf * n_samples)` are the minimum number of 
          samples for each node.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all 
        the input samples) required to be at a leaf node. Samples have 
        equal weight when sample_weight is not provided.

    max_features : int, float, str or None, default=None
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and 
          `int(max_features * n_features)` features are considered at each 
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator. The features are always 
        randomly permuted at each split. When `max_features` < n_features, 
        the algorithm will select `max_features` at random at each split 
        before finding the best split among them. 
        - Pass an int for reproducible output across multiple function calls.

    max_leaf_nodes : int or None, default=None
        Grow a tree with `max_leaf_nodes` in best-first fashion. Best nodes 
        are defined as relative reduction in impurity. If None, then 
        unlimited number of leaf nodes.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity 
        greater than or equal to this value.

    class_weight : dict, list of dicts, "balanced" or None, default=None
        Weights associated with classes in the form `{class_label: weight}`. 
        If None, all classes are supposed to have weight one. If "balanced", 
        the class weights will be adjusted inversely proportional to class 
        frequencies in the input data as `n_samples / (n_classes * np.bincount(y))`.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The 
        subtree with the largest cost complexity that is smaller than 
        `ccp_alpha` will be chosen.

    Attributes
    ----------
    base_estimators_ : list of DecisionTreeClassifier
        List of base learners, each a DecisionTreeClassifier.
    weights_ : list
        Weights associated with each base learner, influencing their contribution
        to the final prediction.

    Example
    -------
    Here's an example of how to use the `HybridBoostingTreeClassifier` on the
    Iris dataset for binary classification:

    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> from gofast.estimators.tree import HybridBoostingTreeClassifier

    >>> # Load the Iris dataset
    >>> iris = load_iris()
    >>> X = iris.data[:, :2]
    >>> y = (iris.target != 0) * 1  # Converting to binary classification

    >>> # Split the data into training and testing sets
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.3, random_state=0)

    >>> # Create and fit the WeightedTreeClassifier
    >>> hybrid_boosted_tree = WeightedTreeClassifier(
    ...     n_estimators=50, eta0=0.01, max_depth=3)
    >>> hybrid_boosted_tree.fit(X_train, y_train)

    >>> # Make predictions and evaluate the model
    >>> y_pred = hybrid_boosted_tree.predict(X_test)
    >>> accuracy = np.mean(y_pred == y_test)
    >>> print('Accuracy:', accuracy)

    Notes
    -----
    The Hybrid Boosted Tree Classifier uses a series of mathematical steps to refine
    the predictions iteratively:
    
    1. Weighted Error Calculation:
       .. math::
           \text{Weighted Error} = \sum_{i=1}^{n} (weights_i \cdot (y_i \neq y_{\text{pred}_i}))
    
    2. Weight Calculation for Base Learners:
       .. math::
           \text{Weight} = \text{learning\_rate} \cdot\\
               \log\left(\frac{1 - \text{Weighted Error}}{\text{Weighted Error}}\right)
    
    3. Update Sample Weights:
       .. math::
           \text{Sample\_Weights} = \exp(-\text{Weight} \cdot y \cdot y_{\text{pred}})
    
    where:
    - :math:`n` is the number of samples in the training data.
    - :math:`weights_i` are the weights associated with each sample.
    - :math:`y_i` is the true label of each sample.
    - :math:`y_{\text{pred}_i}` is the predicted label by the classifier.
    - :math:`\text{learning\_rate}` is a parameter that controls the rate at 
      which the model learns.
    
    This model effectively combines the predictive power of multiple trees 
    through boosting, adjusting errors from previous iterations to 
    enhance overall accuracy.

    The Hybrid Boosted Tree Classifier is suitable for binary classification
    tasks where you want to combine the strengths of decision trees and
    gradient boosting. It can be used in various applications, such as
    fraud detection, spam classification, and more.

    The model's performance depends on the quality of the data, the choice of
    hyperparameters, and the number of estimators. With proper tuning, it can
    achieve high classification accuracy.

    See Also
    --------
    - `sklearn.ensemble.GradientBoostingClassifier`: Scikit-learn's Gradient
      Boosting Classifier for comparison.
    - `sklearn.tree.DecisionTreeClassifier`: Decision tree classifier used as
      base learners in ensemble methods.
    - `sklearn.metrics.accuracy_score`: A common metric for evaluating
      classification models.

    """
    def __init__(
        self, 
        n_estimators=50, 
        eta0=0.1, 
        max_depth=3, 
        criterion="gini", 
        splitter="best", 
        min_samples_split=2, 
        min_samples_leaf=1, 
        min_weight_fraction_leaf=0., 
        max_features=None, 
        random_state=None, 
        max_leaf_nodes=None, 
        min_impurity_decrease=0.,
        class_weight=None, 
        ccp_alpha=0. 
        ):
        self.n_estimators = n_estimators
        self.eta0 = eta0
        self.max_depth = max_depth
        self.criterion = criterion 
        self.splitter = splitter 
        self.min_samples_split = min_samples_split 
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf 
        self.max_features = max_features 
        self.random_state = random_state 
        self.max_leaf_nodes = max_leaf_nodes 
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight 
        self.ccp_alpha = ccp_alpha 

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object
        
        """
        self.base_estimators_ = []
        self.weights_ = []
        # Initialize sample weights
        sample_weights = self._compute_sample_weights(y) 
    
        for _ in range(self.n_estimators):
            # Fit a decision tree on the weighted dataset
            base_estimator = DecisionTreeClassifier(
                max_depth=self.max_depth,
                criterion=self.criterion, 
                splitter=self.splitter, 
                min_samples_split=self.min_samples_split, 
                min_samples_leaf=self.min_samples_leaf, 
                min_weight_fraction_leaf=self.min_weight_fraction_leaf, 
                max_features=self.max_features, 
                random_state=self.random_state, 
                max_leaf_nodes=self.max_leaf_nodes, 
                min_impurity_decrease=self.min_impurity_decrease,
                class_weight=self.class_weight, 
                ccp_alpha=self.ccp_alpha
            )
                
            base_estimator.fit(X, y, sample_weight=sample_weights)
            
            # Calculate weighted error
            y_pred = base_estimator.predict(X)
            errors = (y != y_pred)
            weighted_error = np.sum(sample_weights * errors) / np.sum(sample_weights)
    
            if weighted_error == 0:
                # Prevent log(0) scenario
                continue
    
            # Weight calculation for this base estimator
            weight = self.eta0 * np.log((1 - weighted_error) / weighted_error)
            # Update sample weights for next iteration
            sample_weights = self._update_sample_weights(y, y_pred, weight)
            
            # Store the base estimator and its weight
            self.base_estimators_.append(base_estimator)
            self.weights_.append(weight)
    
        return self

    def predict(self, X):
        check_is_fitted(self, 'base_estimators_')
        y_pred = np.zeros(X.shape[0])
    
        for base_estimator, weight in zip(self.base_estimators_, self.weights_):
            y_pred += weight * base_estimator.predict(X)
    
        # Normalize predictions before applying sign
        y_pred = y_pred / np.sum(self.weights_) if self.weights_ else y_pred
        return np.sign(y_pred)

    def predict_proba(self, X):
        """
        Predict class probabilities using the Hybrid Boosted Tree Classifier 
        model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        proba : array-like of shape (n_samples, 2)
            The class probabilities of the input samples.
        """
        check_is_fitted(self, 'base_estimators_')
        X = check_array(X, accept_sparse=True)
        # Compute weighted sum of predictions from all base estimators
        weighted_predictions = sum(weight * estimator.predict(X) 
                                   for weight, estimator in zip(
                                           self.weights_, self.base_estimators_))

        # Convert to probabilities using the sigmoid function
        proba_positive_class = 1 / (1 + np.exp(-weighted_predictions))
        proba_negative_class = 1 - proba_positive_class

        return np.vstack((proba_negative_class, proba_positive_class)).T

    def _compute_sample_weights(self, y):
        """Compute sample weights."""
        return np.ones_like(y) / len(y)

    def _update_sample_weights(self, y, y_pred, weight):
        """Update sample weights."""
        return np.exp(-weight * y * y_pred)
    

class WeightedTreeRegressor(BaseEstimator, RegressorMixin):
    """
    Weighted Tree Regressor.

    The Weighted Tree Regressor is an ensemble learning model that 
    combines decision trees with gradient boosting techniques to tackle regression tasks.
    
    By integrating multiple decision trees with varying weights, this regressor
    achieves high accuracy and reduces the risk of overfitting.

    Parameters
    ----------
    n_estimators : int, default=50
        The number of decision trees in the ensemble.
    eta0 : float, default=0.1
        The learning rate for gradient boosting, controlling how much each tree
        influences the overall prediction.
    max_depth : int, default=3
        The maximum depth of each decision tree, determining the complexity of
        the model.
    criterion : {"squared_error", "friedman_mse", "absolute_error", "poisson"},\
        default="squared_error"
        The function to measure the quality of a split.
    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node.
    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node.
    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all the input samples)
        required to be at a leaf node.
    max_features : int, float, str or None, default=None
        The number of features to consider when looking for the best split.
    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator.
    max_leaf_nodes : int or None, default=None
        Grow a tree with `max_leaf_nodes` in best-first fashion.
    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity 
        greater than or equal to this value.

    Attributes
    ----------
    base_estimators_ : list of DecisionTreeRegressor
        List of base learners, each a DecisionTreeRegressor.
    weights_ : list
        Weights associated with each base learner, influencing their contribution
        to the final prediction.

    Example
    -------
    Here's an example of how to use the `WeightedTreeRegressor` on a dataset:

    >>> from sklearn.datasets import make_regression
    >>> from sklearn.model_selection import train_test_split
    >>> from gofast.estimators.tree import WeightedTreeRegressor

    >>> X, y = make_regression(n_samples=100, n_features=4, noise=0.1)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                            random_state=0)

    >>> reg = WeightedTreeRegressor(n_estimators=50, eta0=0.1, max_depth=3)
    >>> reg.fit(X_train, y_train)
    >>> y_pred = reg.predict(X_test)

    Notes
    -----
    This model effectively combines the predictive power of multiple trees 
    through boosting, adjusting errors from previous iterations to 
    enhance overall accuracy.

    The model's performance depends on the quality of the data, the choice of
    hyperparameters, and the number of estimators. With proper tuning, it can
    achieve high regression accuracy.
    """

    def __init__(
        self, 
        n_estimators=50, 
        eta0=0.1, 
        max_depth=3, 
        criterion="squared_error", 
        splitter="best", 
        min_samples_split=2, 
        min_samples_leaf=1, 
        min_weight_fraction_leaf=0., 
        max_features=None, 
        random_state=None, 
        max_leaf_nodes=None, 
        min_impurity_decrease=0.
        ):
        self.n_estimators = n_estimators
        self.eta0 = eta0
        self.max_depth = max_depth
        self.criterion = criterion 
        self.splitter = splitter 
        self.min_samples_split = min_samples_split 
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf 
        self.max_features = max_features 
        self.random_state = random_state 
        self.max_leaf_nodes = max_leaf_nodes 
        self.min_impurity_decrease = min_impurity_decrease

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y, accept_sparse=True, accept_large_sparse=True)
        self.base_estimators_ = []
        self.weights_ = []
        residuals = y

        for _ in range(self.n_estimators):
            base_estimator = DecisionTreeRegressor(
                max_depth=self.max_depth,
                criterion=self.criterion, 
                splitter=self.splitter, 
                min_samples_split=self.min_samples_split, 
                min_samples_leaf=self.min_samples_leaf, 
                min_weight_fraction_leaf=self.min_weight_fraction_leaf, 
                max_features=self.max_features, 
                random_state=self.random_state, 
                max_leaf_nodes=self.max_leaf_nodes, 
                min_impurity_decrease=self.min_impurity_decrease
            )
                
            base_estimator.fit(X, residuals)
            predictions = base_estimator.predict(X)

            residuals -= self.eta0 * predictions
            self.base_estimators_.append(base_estimator)
            self.weights_.append(self.eta0)
    
        return self

    def predict(self, X):
        """
        Predict using the Weighted Tree Regressor model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted values.
        """
        check_is_fitted(self, 'base_estimators_')
        X = check_array(X, accept_sparse=True)
        y_pred = np.zeros(X.shape[0])
    
        for base_estimator, weight in zip(self.base_estimators_, self.weights_):
            y_pred += weight * base_estimator.predict(X)
    
        return y_pred

    def _compute_sample_weights(self, y):
        """Compute sample weights."""
        return np.ones_like(y) / len(y)

    def _update_sample_weights(self, y, y_pred, weight):
        """Update sample weights."""
        return np.exp(-weight * y * y_pred)


class WeightedTreeRegressor0(BaseEstimator, RegressorMixin):
    r"""
    Weighted Regression Tree (BRT) for regression tasks.

    The Hybrid Boosted Tree Regressor is a powerful ensemble learning model
    that combines multiple Boosted Regression Tree (BRT) models. Each BRT
    model is itself an ensemble created using boosting principles.
    
    This ensemble model combines multiple Boosted Regression Tree models,
    each of which is an ensemble in itself, created using the 
    principles of boosting.
    
    In `HybridBoostingTreeRegressor` class, the `n_estimators` parameter 
    controls the number of individual Boosted Regression Trees in the ensemble,
    and `brt_params` is a dictionary of parameters to be passed to each Boosted 
    Regression Tree model. The `GradientBoostingRegressor` from scikit-learn 
    is used as the individual BRT model. This class's fit method trains 
    each BRT model on the entire dataset, and the predict method averages 
    their predictions for the final output.

    Parameters
    ----------
    n_estimators : int, default=50
        The number of Boosted Regression Tree models in the ensemble.
    brt_params : dict, default=None
        Dictionary of parameters for configuring each Boosted Regression Tree model. 
        If None, default parameters are used.

    Attributes
    ----------
    brt_ensembles_ : list of GradientBoostingRegressor
        A list containing the fitted Boosted Regression Tree models.

    Example
    -------
    Here's an example of how to initialize and use the `HybridBoostingTreeRegressor`:
    ```python
    from gofast.estimators.boosting import HybridBoostingTreeRegressor
    import numpy as np

    brt_params = {'n_estimators': 100, 'max_depth': 3, 'eta0': 0.1}
    hybrid_brt = HybridBoostingTreeRegressor(n_estimators=10, brt_params=brt_params)
    X, y = np.random.rand(100, 4), np.random.rand(100)
    hybrid_brt.fit(X, y)
    y_pred = hybrid_brt.predict(X)
    ```

    Notes
    -----
    The Hybrid Boosted Tree Regressor employs an iterative process to refine 
    predictions:

    1. Calculate Residuals:
       .. math::
           \text{Residuals} = y - F_k(x)

    2. Update Predictions:
       .. math::
           F_{k+1}(x) = F_k(x) + \text{eta0} \cdot h_k(x)

    where:
    - :math:`F_k(x)` is the prediction of the ensemble at iteration \(k\).
    - :math:`y` is the true target values.
    - :math:`\text{eta0}` is the learning rate, influencing the impact
      of each tree.
    - :math:`h_k(x)` is the prediction update contributed by the new tree at 
      iteration \(k\).

    The Hybrid Boosted Regression Tree Ensemble is particularly effective for
    regression tasks requiring accurate modeling of complex relationships 
    within data, such as in financial markets, real estate, or any predictive 
    modeling that benefits from robust and precise forecasts.

    The model's performance significantly depends on the quality of the data, 
    the setting of hyperparameters, and the adequacy of the training process.

    See Also
    --------
    - `sklearn.ensemble.GradientBoostingRegressor`: Compare to Scikit-learn's 
      Gradient Boosting Regressor.
    - `sklearn.tree.DecisionTreeRegressor`: The type of regressor used as base 
      learners in this ensemble method.
    """
    def __init__(self, n_estimators=10, brt_params=None):
        self.n_estimators = n_estimators
        self.brt_params = brt_params or {}

    def fit(self, X, y):
        """
        Fit the Hybrid Boosted Regression Tree Ensemble model to the data.

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
        self.brt_ensembles_ = []
        for _ in range(self.n_estimators):
            brt = DecisionTreeRegressor(**self.brt_params)
            brt.fit(X, y)
            self.brt_ensembles_.append(brt)

        return self

    def predict(self, X):
        """
        Predict using the Hybrid Boosted Regression Tree Ensemble model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted values.
        """
        check_is_fitted(self, "brt_ensembles_")
        
        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )
        
        predictions = np.array([brt.predict(X) for brt in self.brt_ensembles_])
        return np.mean(predictions, axis=0)
    
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
    
class DecisionStumpRegressor (BaseClass, StandardEstimator):
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
    >>> from gofast.estimators.tree import DecisionStumpRegressor
    >>> X, y = make_regression(n_samples=100, n_features=1, noise=10)
    >>> stump = DecisionStumpRegressor()
    >>> stump.fit(X, y)
    >>> predictions = stump.predict(X)
    """

    def __init__(self ):
        super().__init__()
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
    




















