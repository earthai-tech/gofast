# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

from __future__ import annotations

import numpy as np
from tqdm import tqdm
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.utils import resample
from sklearn.metrics import r2_score , accuracy_score

from ._base import StandardEstimator 
from ..api.property import BaseClass 
from ..tools.validator import check_X_y, get_estimator_name, check_array 
from ..tools.validator import check_is_fitted, validate_fit_weights 

__all__=[ 
    "DecisionStumpRegressor", "DecisionStumpClassifier",
    "DecisionTreeBasedRegressor", "DecisionTreeBasedClassifier",
    "WeightedTreeClassifier", "WeightedTreeRegressor", 
    ]

class DecisionTreeBasedRegressor(BaseEstimator, RegressorMixin):
    """
    Decision Tree-based Regressor for Regression Tasks.

    The `DecisionTreeBasedRegressor` employs an ensemble approach, 
    combining multiple Decision Regression Trees to form a more 
    robust regression model. Each tree in the ensemble independently 
    predicts the outcome, and the final prediction is derived by averaging 
    these individual predictions. This method is effective in reducing 
    variance and improving prediction accuracy over a single decision tree.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the ensemble.
    max_depth : int, default=3
        The maximum depth of each regression tree.
    criterion : {"squared_error", "friedman_mse", "absolute_error", "poisson"},\
        default="squared_error"
        The function to measure the quality of a split. Supported criteria 
        are "squared_error" for mean squared error, "friedman_mse" for 
        mean squared error with improvement score by Friedman, "absolute_error" 
        for mean absolute error, and "poisson" for Poisson deviance.
        
    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node. Supported 
        strategies are "best" to choose the best split and "random" to 
        choose the best random split.
        
    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and 
          `ceil(min_samples_split * n_samples)` are the minimum number of 
          samples for each split.
          
    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node:
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and 
          `ceil(min_samples_leaf * n_samples)` are the minimum number of 
          samples for each node.
          
    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all 
        the input samples) required to be at a leaf node.
        
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
        before finding the best split among them. Pass an int for reproducible 
        output across multiple function calls.
        
    max_leaf_nodes : int or None, default=None
        Grow a tree with `max_leaf_nodes` in best-first fashion. Best nodes 
        are defined as relative reduction in impurity. If None, then unlimited 
        number of leaf nodes.
        
    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity 
        greater than or equal to this value.
    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The 
        subtree with the largest cost complexity that is smaller than 
        `ccp_alpha` will be chosen.
        
    subsample : float, default=1.0
        The fraction of samples to be used for fitting each base learner.
        
    bootstrap : bool, default=True
        Whether samples are drawn with replacement. If False, sampling 
        without replacement is performed.
        
    verbose : int, default=0
        Controls the verbosity when fitting and predicting.

    Attributes
    ----------
    estimators_ : list of DecisionTreeRegressor
        The collection of fitted sub-estimators.
    oob_score_ : float
        Out-of-bag score for the training dataset.

    Examples
    --------
    Here's an example of how to use the `DecisionTreeBasedRegressor` on a 
    dataset:

    >>> from sklearn.datasets import make_regression
    >>> from sklearn.model_selection import train_test_split
    >>> from gofast.estimators.tree import DecisionTreeBasedRegressor

    >>> X, y = make_regression(n_samples=100, n_features=4, noise=0.1)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, 
    ...                                                     test_size=0.3, 
    ...                                                     random_state=0)
    
    >>> reg = DecisionTreeBasedRegressor(n_estimators=50, 
    ...                                  max_depth=3, verbose=1)
    >>> reg.fit(X_train, y_train)
    >>> y_pred = reg.predict(X_test)

    Notes
    -----
    This model effectively combines the predictive power of multiple trees 
    through averaging, reducing variance and improving accuracy over a 
    single decision tree. The ensemble prediction is computed by averaging 
    the predictions from each individual Regression Tree within the ensemble, 
    as follows:

    .. math::
        y_{\text{pred}} = \frac{1}{N} \sum_{i=1}^{N} y_{\text{tree}_i}

    where:
    - :math:`N` is the number of trees in the ensemble.
    - :math:`y_{\text{pred}}` is the final predicted value aggregated from 
      all trees.
    - :math:`y_{\text{tree}_i}` represents the prediction made by the 
      :math:`i`-th Regression Tree.

    The use of bootstrap sampling and subsampling enhances model robustness 
    and provides an estimate of model performance through out-of-bag (OOB) 
    score when subsampling is enabled.

    See Also
    --------
    sklearn.ensemble.RandomForestRegressor : A popular ensemble method 
        based on decision trees for regression tasks.
    sklearn.tree.DecisionTreeRegressor : Decision tree regressor used as 
        base learners in ensemble methods.
    sklearn.metrics.mean_squared_error : A common metric for evaluating 
        regression models.
    gofast.estimators.tree.BoostedRegressionTree : An enhanced BRT.
    """

    def __init__(
        self, 
        n_estimators=100, 
        max_depth=3, 
        criterion="squared_error", 
        splitter="best", 
        min_samples_split=2, 
        min_samples_leaf=1, 
        min_weight_fraction_leaf=0., 
        max_features=None, 
        random_state=None, 
        max_leaf_nodes=None, 
        min_impurity_decrease=0.,
        ccp_alpha=0., 
        subsample=1.0, 
        bootstrap=True,
        verbose=0
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
        self.ccp_alpha = ccp_alpha
        self.subsample = subsample
        self.bootstrap = bootstrap
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):
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
        X, y = check_X_y(X, y, accept_sparse=True)
        self.estimators_ = []
        self.oob_score_ = None
        n_samples = X.shape[0]
        sample_indices = np.arange(n_samples)
        sample_weight = validate_fit_weights(y, sample_weight=sample_weight)

        if self.verbose > 0:
            progress_bar = tqdm(
                range(self.n_estimators), ascii=True, ncols= 100,
                desc=f'Fitting {self.__class__.__name__}', 
                )
        for i in range(self.n_estimators):
            if self.bootstrap:
                subsample_indices = np.random.choice(
                    sample_indices, size=int(self.subsample * n_samples), replace=True
                )
            else:
                subsample_indices = sample_indices[:int(self.subsample * n_samples)]

            tree = DecisionTreeRegressor(
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
                ccp_alpha=self.ccp_alpha
            )

            tree.fit(X[subsample_indices], y[subsample_indices],
                     sample_weight=sample_weight[subsample_indices])
            self.estimators_.append(tree)

            if self.verbose > 0:
                progress_bar.update(1)

        if self.verbose > 0:
            progress_bar.close()

        if self.bootstrap and self.subsample < 1.0:
            oob_indices = np.setdiff1d(sample_indices, subsample_indices)
            if len(oob_indices) > 0:
                oob_predictions = np.mean(
                    [tree.predict(X[oob_indices]) for tree in self.estimators_], axis=0
                )
                self.oob_score_ = np.mean((y[oob_indices] - oob_predictions) ** 2)

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
        X = check_array(X, accept_sparse=True)

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
        ccp_alpha=0.0, 
        verbose=0 
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
        self.verbose=verbose

    def fit(self, X, y, sample_weight =None):
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
        if self.verbose > 0:
            progress_bar = tqdm(
                range(self.n_estimators), ascii=True, ncols= 100,
                desc=f'Fitting {self.__class__.__name__}', 
                )
            
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
            tree.fit(X_resampled, y_resampled, sample_weight= sample_weight ) 
            self.estimators_.append(tree)
            
            if self.verbose > 0:
                progress_bar.update(1)

        if self.verbose > 0:
            progress_bar.close()

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
        ccp_alpha=0., 
        verbose=False
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
        
        if self.verbose > 0:
            progress_bar = tqdm(range(self.n_estimators), ascii=True, ncols= 100,
                desc=f'Fitting {self.__class__.__name__}' )
            
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
            
            if self.verbose > 0:
                progress_bar.update(1)

        if self.verbose > 0:
            progress_bar.close()
            
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
    
    The model updates residuals in each iteration to improve accuracy. The 
    residuals are calculated as:

    .. math:: r_i = y_i - \eta \cdot f_m(X_i)

    where:
    - :math:`r_i` is the residual for sample :math:`i`.
    - :math:`y_i` is the true value for sample :math:`i`.
    - :math:`\eta` is the learning rate.
    - :math:`f_m` is the prediction of the :math:`m`-th tree.

    The final prediction is the sum of weighted predictions of all trees:

    .. math:: \hat{y} = \sum_{m=1}^{M} \eta \cdot f_m(X)

    See Also
    --------
    sklearn.ensemble.GradientBoostingRegressor : Scikit-learn's Gradient 
        Boosting Regressor for comparison.
    sklearn.tree.DecisionTreeRegressor : Decision tree regressor used as 
        base learners in ensemble methods.
    sklearn.metrics.mean_squared_error : A common metric for evaluating 
        regression models.
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
        min_impurity_decrease=0.,
        verbose=False
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
        self.verbose=verbose

    def fit(self, X, y, sample_weight=None):
        """
        Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        sample_weight : array-like, shape = [n_samples], optional
            Individual weights for each sample.

        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y, accept_sparse=True, accept_large_sparse=True)
        self.base_estimators_ = []
        self.weights_ = []
        residuals = y
        
        if self.verbose:
            progress_bar = tqdm(range(self.n_estimators), ascii=True, ncols= 100,
                desc=f'Fitting {self.__class__.__name__}' )
        
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
                
            base_estimator.fit(X, residuals, sample_weight=sample_weight)
            predictions = base_estimator.predict(X)

            residuals -= self.eta0 * predictions
            self.base_estimators_.append(base_estimator)
            self.weights_.append(self.eta0)
            
            if self.verbose:
                progress_bar.update(1)

        if self.verbose:
            progress_bar.close()
            
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


class DecisionStumpRegressor(BaseClass, StandardEstimator):
    r"""
    A simple decision stump regressor for use in gradient boosting.

    This class implements a basic decision stump, which is a decision tree 
    with only one decision node and two leaves. The stump splits the data 
    based on the feature and threshold that minimizes the error.
    
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

    Parameters
    ----------
    min_samples_split : int, default=2
        The minimum number of samples required to consider a split at a node.

    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node. A split point 
        at any depth will only be considered if it leaves at least this many training 
        samples in each of the left and right branches.
        
    verbose : int, default=False
        Controls the verbosity when fitting and predicting.
        
    Attributes
    ----------
    split_feature_ : int
        Index of the feature used for the split.

    split_value_ : float
        Threshold value used for the split.

    left_value_ : float
        The value predicted for samples where the feature value is less than or 
        equal to the split value.

    right_value_ : float
        The value predicted for samples where the feature value is greater than 
        the split value.

    feature_importances_ : ndarray
        The feature importances (if calculated), reflecting the reduction in MSE 
        contributed by each feature.

    Methods
    -------
    fit(X, y, sample_weight=None)
        Fit the decision stump to the data.

    predict(X)
        Predict target values for the given input data using the trained 
        decision stump.

    decision_function(X)
        Compute the raw decision scores for the given input data.

    score(X, y, sample_weight=None)
        Return the coefficient of determination R^2 of the prediction.
       
    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> from gofast.estimators.tree import DecisionStumpRegressor
    >>> X, y = make_regression(n_samples=100, n_features=1, noise=10)
    >>> stump = DecisionStumpRegressor()
    >>> stump.fit(X, y)
    >>> predictions = stump.predict(X)
    """

    def __init__(self, min_samples_split=2, min_samples_leaf=1, verbose=False):
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.verbose=verbose
        
        self.split_feature_ = None
        self.split_value_ = None
        self.left_value_ = None
        self.right_value_ = None
        self.feature_importances_ = None
        

    def fit(self, X, y, sample_weight=None):
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
        X, y = check_X_y(X, y )
        sample_weight = validate_fit_weights(np.ones(X.shape[0]), sample_weight)
        self.feature_importances_ = np.zeros(X.shape[1])
        min_error = float('inf')
        n_samples, n_features = X.shape
        
        if self.verbose:
            progress_bar = tqdm(range(n_features), ascii=True, ncols= 100,
                desc=f'Fitting {self.__class__.__name__}', 
                )
        for feature in range(n_features):
            sorted_idx = np.argsort(X[:, feature])
            X_sorted, y_sorted, weights_sorted = ( 
                X[sorted_idx, feature], y[sorted_idx], sample_weight[sorted_idx]
                )
            for i in range(self.min_samples_leaf, n_samples - self.min_samples_leaf):
                if i < self.min_samples_split or X_sorted[i] == X_sorted[i - 1]:
                    continue
                left_mask = sorted_idx[:i]
                right_mask = sorted_idx[i:]
                if len(left_mask) < self.min_samples_leaf or len(
                        right_mask) < self.min_samples_leaf:
                    continue

                left_value = np.average(y_sorted[:i], weights=weights_sorted[:i])
                right_value = np.average(y_sorted[i:], weights=weights_sorted[i:])
                error = (np.sum(weights_sorted[:i] * ((y_sorted[:i] - left_value) ** 2)) +
                         np.sum(weights_sorted[i:] * ((y_sorted[i:] - right_value) ** 2)))

                if error < min_error:
                    min_error = error
                    self.split_feature_ = feature
                    self.split_value_ = X_sorted[i - 1]
                    self.left_value_ = left_value
                    self.right_value_ = right_value
                    self.feature_importances_[feature] += min_error - error
            
            if self.verbose:
                progress_bar.update(1)

        if self.verbose:
            progress_bar.close()
                
        self.fitted_ = True

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
        left_mask = X[:, self.split_feature_] <= self.split_value_
        y_pred = np.where(left_mask, self.left_value_, self.right_value_)
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
        # # Calculate the decision scores based on the split
        decision_scores = np.where(
            X[:, self.split_feature_] <= self.split_value_,
            self.left_value_, self.right_value_
            )
        
        return decision_scores

    def score(self, X, y, sample_weight=None):
        """
        Calculate the coefficient of determination :math:`R^2` of the 
        prediction.
    
        The :math:`R^2` score is a statistical measure of how well the predictions 
        approximate the true data points. An :math:`R^2` of 1 indicates that 
        the model perfectly predicts the target values, whereas an :math:`R^2` 
        of 0 indicates that the model predicts as well as a model that always 
        predicts the mean of the target  values, regardless of the input 
        features.
    
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input samples. For each feature in the dataset, the method 
            predicts a response based on the decision stump model.
        
        y : ndarray of shape (n_samples,)
            True values for the target variable. These are the values that 
            the model attempts to predict.
    
        sample_weight : ndarray of shape (n_samples,), default=None
            Individual weights for each sample. If provided, the :math:`R^2` score 
            will be calculated with these weights, emphasizing the importance of 
            certain samples more than others.
    
        Returns
        -------
        score : float
            The :math:`R^2` score that indicates the proportion of variance in the 
            dependent variable that is predictable from the independent variables. 
            Values range from -∞ (a model that performs infinitely worse than 
            the mean model) to 1. A model that always predicts the exact true 
            value would achieve a score of 1.
    
        Examples
        --------
        >>> from sklearn.datasets import make_regression
        >>> X, y = make_regression(n_samples=100, n_features=1, noise=0.1)
        >>> stump = DecisionStumpRegressor()
        >>> stump.fit(X, y)
        >>> score = stump.score(X, y)
        >>> print(f"R^2 score: {score:.2f}")
        """
        check_is_fitted(self, "fitted_") 
        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight)

class DecisionStumpClassifier(BaseClass, StandardEstimator):
    r"""
    A simple decision stump classifier that uses a single-level decision tree for
    binary classification. This classifier identifies the best feature and
    threshold to split the data into two groups, aiming to minimize impurity in 
    each node.

    The decision stump makes a binary decision: it assigns every sample in one
    subset to one class and all samples in the other subset to another class. The
    choice of subset is based on a threshold applied to one feature.

    .. math::
        I(node) = 1 - \max(p, 1-p)

    Where \( I(node) \) is the impurity of the node, and \( p \) is the proportion
    of the samples in the node that belong to the most frequent class.

    Parameters
    ----------
    min_samples_split : int, default=2
        The minimum number of samples required to consider a split at a node.

    min_samples_leaf : int, default=1
        The minimum number of samples required to be at a leaf node. This parameter
        ensures that each leaf has at least `min_samples_leaf` samples, which
        helps prevent the tree from overfitting.
        
    verbose : int, default=False
        Controls the verbosity when fitting.

    Attributes
    ----------
    split_feature_ : int
        The index of the feature used for the best split.

    split_value_ : float
        The threshold value used for the split at `split_feature_`.

    left_class_ : int
        The class label assigned to samples where the feature value is less than
        or equal to the threshold.

    right_class_ : int
        The class label assigned to samples where the feature value is greater
        than the threshold.

    See Also
    --------
    DecisionTreeClassifier : A classifier that uses multiple levels of decision
                             nodes, providing more complex classification boundaries.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from gofast.estimators.tree import DecisionStumpClassifier
    >>> X, y = make_classification(n_samples=100, n_features=2, random_state=42)
    >>> stump = DecisionStumpClassifier(min_samples_split=4, min_samples_leaf=2)
    >>> stump.fit(X, y)
    >>> print(stump.predict([[0, 0], [1, 1]]))
    [0 1]

    Notes
    -----
    This implementation is simplified and intended for binary classification only.
    It does not handle multi-class classification and does not support more advanced
    tree-building methods that consider information gain or Gini impurity.
    """
    def __init__(self, min_samples_split=2, min_samples_leaf=1, verbose=False):
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.verbose=verbose
        self.split_feature_ = None
        self.split_value_ = None
        self.left_class_ = None
        self.right_class_ = None
        
        
    def fit(self, X, y, sample_weight=None):
        r"""
        Fit the decision stump model to the binary classification data.
    
        The fitting process involves finding the feature and threshold that 
        minimize the impurity of the resulting binary split. This is achieved by 
        calculating the error as the negative of the weighted sum of correct 
        classifications in both left and right groups formed by the threshold. 
        The goal is to maximize the number of correct predictions by selecting 
        the optimal split.
    
        .. math::
            error = - \left( \sum_{i \in \text{{left}}}w_i(y_i = \text{{left\_class}}) + 
            \sum_{j \in \text{{right}}}w_j(y_j = \text{{right\_class}}) \right)
    
        Where `left_class` and `right_class` are the classes predicted for the 
        left and right groups, respectively, \( y_i \) is the actual class of the 
        ith sample, and \( w_i \) is the sample weight of the ith sample.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
    
        y : array-like of shape (n_samples,)
            Target values (class labels in classification).
        
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. The length 
            of `sample_weight` must match the number of samples.
    
        Returns
        -------
        self : object
            Returns self with the fitted model. The following attributes are set after 
            fitting:
                - `split_feature_` : Index of the feature used for the best split.
                - `split_value_` : Threshold value at the best split.
                - `left_class_` : Class label for samples on the left of the threshold.
                - `right_class_` : Class and data handling, this method modifies internal
                                   state and prepares the model for prediction.
    
        Examples
        --------
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=100, n_features=2, random_state=42)
        >>> stump = DecisionStumpClassifier()
        >>> stump.fit(X, y)
        >>> stump.predict([[0.1, -0.1], [0.3, 0.3]])
        array([0, 1])
    
        Notes
        -----
        The method checks the input data using `check_X_y` from sklearn, which verifies
        that the data is correctly formatted and that the number of samples and labels
        matches. It also requires the `get_estimator_name` function to properly identify
        the estimator in error messages and validations.
        """
        X, y = check_X_y(X, y, estimator=self)
        min_error = float('inf')
        n_samples, n_features = X.shape
    
        sample_weight = validate_fit_weights(y, sample_weight ) 
        if self.verbose:
            progress_bar = tqdm(range(n_features), ascii=True, ncols= 100,
                desc=f'Fitting {self.__class__.__name__}' )
            
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
    
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(
                        right_mask) < self.min_samples_leaf:
                    continue
    
                left_class = np.bincount(y[left_mask], weights=sample_weight[left_mask]).argmax()
                right_class = np.bincount(y[right_mask], weights=sample_weight[right_mask]).argmax()
    
                error = - (
                    np.sum(sample_weight[left_mask] * (y[left_mask] == left_class)) + 
                    np.sum(sample_weight[right_mask] * (y[right_mask] == right_class))
                )
    
                if error < min_error:
                    min_error = error
                    self.split_feature_ = feature
                    self.split_value_ = threshold
                    self.left_class_ = left_class
                    self.right_class_ = right_class
            
            if self.verbose:
                progress_bar.update(1)

        if self.verbose:
            progress_bar.close()
            
        self.fitted_ = True
        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.
    
        The predictions are based on the simple rule using the `split_feature_` and 
        `split_value_` determined during the fitting process. Depending on whether 
        the feature value for a given sample is less than or equal to `split_value_`, 
        the prediction is either `left_class_` or `right_class_`.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
    
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted class labels for each sample in X.
    
        Examples
        --------
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=100, n_features=2, random_state=42)
        >>> stump = DecisionStumpClassifier()
        >>> stump.fit(X, y)
        >>> print(stump.predict([[0.1, -0.1], [0.3, 0.3]]))
        [0 1]
        """
        check_is_fitted(self, "fitted_") 
        left_mask = X[:, self.split_feature_] <= self.split_value_
        y_pred = np.where(left_mask, self.left_class_, self.right_class_)
        return y_pred
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
    
        The probabilities are computed as the frequencies of the `left_class_` and 
        `right_class_` in the training data. For each sample in X, this method 
        returns the probability of the sample belonging to the `left_class_` or 
        `right_cap` class based on the position relative to the `split_value_`.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
    
        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            The probability of the sample belonging to each class.
    
        Examples
        --------
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        >>> stump = DecisionStumpClassifier()
        >>> stump.fit(X, y)
        >>> print(stump.predict_proba([[0.1, -0.1], [0.3, 0.3]]))
        [[0.7 0.3]
         [0.2 0.8]]
        """
        check_is_fitted(self, "fitted_") 
        left_mask = X[:, self.split_feature_] <= self.split_value_
        proba_left = np.mean(self.left_class_ == 1)
        proba_right = np.mean(self.right_class_ == 1)
        proba = np.zeros((X.shape[0], 2))
        proba[:, 0] = np.where(left_mask, 1 - proba_left, 1 - proba_right)
        proba[:, 1] = np.where(left_mask, proba_left, proba_right)
        return proba
    
    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.
    
        This method computes the accuracy, which is the fraction of correctly predicted
        labels to the total number of observations. It uses the `predict` method to obtain 
        the class labels for the input samples and compares them with the true labels.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
    
        y : array-like of shape (n_samples,)
            True labels for X.
    
        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y. The score is a float in the range [0, 1]
            where 1 indicates perfect accuracy.
    
        Examples
        --------
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=100, n_features=2, random_state=42)
        >>> stump = DecisionStumpClassifier()
        >>> stump.fit(X, y)
        >>> accuracy = stump.score(X, y)
        >>> print(f"Accuracy: {accuracy:.2f}")
        
        Notes
        -----
        This method checks if the estimator is fitted by using `check_is_fitted` before
        proceeding with predictions and scoring. If the model is not fitted, it will raise
        a `NotFittedError`. The inputs are validated using `check_X_y` from sklearn, which
        ensures that the dimensions and types of X and y are compatible and appropriate for
        the estimator.
        """
        check_is_fitted(self, "fitted_") 
        X, y = check_X_y(X, y, estimator=get_estimator_name(self))
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)


















