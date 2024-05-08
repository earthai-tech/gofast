# -*- coding: utf-8 -*-

from __future__ import annotations 
from scipy import stats
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from ..tools.validator import check_X_y, get_estimator_name, check_array 
from ..tools.validator import check_is_fitted

__all__=[
    "BoostedTreeRegressor","BoostedTreeClassifier",
    "HybridBoostedTreeClassifier","HybridBoostedTreeRegressor",
    "EnsembleHBTRegressor", "EnsembleHBTClassifier",
    ]

class BoostedTreeRegressor(BaseEstimator, RegressorMixin):
    r"""
    Enhanced Boosted Regression Tree (BRT) for Regression Tasks.

    The Enhanced Boosted Regression Tree (BRT) is an advanced implementation
    of the Boosted Regression Tree algorithm, aimed at improving performance 
    and reducing overfitting. This model incorporates features like support for 
    multiple loss functions, stochastic boosting, and controlled tree depth for 
    pruning.

    BRT builds on ensemble learning principles, combining multiple decision trees 
    (base learners) to enhance prediction accuracy. It focuses on improving areas 
    where previous iterations of the model underperformed.

    Features:
    - ``Different Loss Functions``: Supports 'linear', 'square', and 'exponential' 
      loss functions, utilizing their derivatives to update residuals.
      
    ``Stochastic Boosting``: The model includes an option for stochastic 
    boosting, controlled by the subsample parameter, which dictates the 
    fraction of samples used for fitting each base learner. This can 
    help in reducing overfitting.
    
    ``Tree Pruning``: While explicit tree pruning isn't detailed here, it can 
    be managed via the max_depth parameter. Additional pruning techniques can 
    be implemented within the DecisionTreeRegressor fitting process.
    
    The iterative updating process of the ensemble is mathematically
    represented as:

    .. math::
        F_{k}(x) = \text{Prediction of the ensemble at iteration } k.

        r = y - F_{k}(x) \text{ (Residual calculation)}

        F_{k+1}(x) = F_{k}(x) + \text{learning_rate} \cdot h_{k+1}(x)

    where:
    - :math:`F_{k}(x)` is the prediction of the ensemble at iteration \(k\).
    - \(r\) represents the residuals, calculated as the difference between the 
      actual values \(y\) and the ensemble's predictions.
    - :math:`F_{k+1}(x)` is the updated prediction after adding the new tree.
    - :math:`h_{k+1}(x)` is the contribution of the newly added tree at 
      iteration \(k+1\).
    - `learning_rate` is a hyperparameter that controls the influence of each 
      new tree on the final outcome.

    The Boosted Regression Tree method effectively harnesses the strengths 
    of multiple trees to achieve lower bias and variance, making it highly 
    effective for complex regression tasks with varied data dynamics.

    Parameters
    ----------
    n_estimators : int
        The number of trees in the ensemble.
    learning_rate : float
        The rate at which the boosting algorithm adapts from previous 
        trees' errors.
    max_depth : int
        The maximum depth of each regression tree.
    loss : str
        The loss function to use. Supported values are 'linear', 
        'square', and 'exponential'.
    subsample : float
        The fraction of samples to be used for fitting each individual base 
        learner .If smaller than 1.0, it enables stochastic boosting.

    Attributes
    ----------
    estimators_ : list of DecisionTreeRegressor
        The collection of fitted sub-estimators.

    Examples
    --------
    >>> from gofast.estimators.boosting import BoostedTreeRegressor
    >>> brt = BoostedTreeRegressor(n_estimators=100, learning_rate=0.1, 
                                    max_depth=3, loss='linear', subsample=0.8)
    >>> X, y = np.random.rand(100, 4), np.random.rand(100)
    >>> brt.fit(X, y)
    >>> y_pred = brt.predict(X)
    
    See Also
    --------
    - `sklearn.ensemble.GradientBoostingRegressor`: The scikit-learn library's
      implementation of gradient boosting for regression tasks.
    - `sklearn.tree.DecisionTreeRegressor`: Decision tree regressor used as
      base learners in ensemble methods.
    - `sklearn.metrics.mean_squared_error`: A common metric for evaluating
      regression models.
  
    Notes
    -----
    - The Boosted Regression Tree model is built iteratively, focusing on
      minimizing errors and improving predictions.
    - Different loss functions can be selected, allowing flexibility in
      modeling.
    - Stochastic boosting can help in reducing overfitting by using a fraction
      of samples for fitting each tree.
    - Tree depth can be controlled to avoid overly complex models.
    """

    def __init__(
        self, 
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=3,
        loss='linear', 
        subsample=1.0
        ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.loss = loss
        self.subsample = subsample
        
    def _loss_derivative(self, y, y_pred):
        """
        Compute the derivative of the loss function.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            True values.
        y_pred : array-like of shape (n_samples,)
            Predicted values.

        Returns
        -------
        loss_derivative : array-like of shape (n_samples,)
        """
        if self.loss == 'linear':
            return y - y_pred
        elif self.loss == 'square':
            return (y - y_pred) * 2
        elif self.loss == 'exponential':
            return np.exp(y_pred - y)
        else:
            raise ValueError("Unsupported loss function")

    def fit(self, X, y):
        """
        Fit the Boosted Regression Tree model to the data.

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
        n_samples = X.shape[0]
        residual = y.copy()
        
        self.estimators_ = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeRegressor(max_depth=self.max_depth)

            # Stochastic boosting (sub-sampling)
            sample_size = int(self.subsample * n_samples)
            indices = np.random.choice(n_samples, sample_size, replace=False)
            X_subset, residual_subset = X[indices], residual[indices]

            tree.fit(X_subset, residual_subset)
            prediction = tree.predict(X)

            # Update residuals
            residual -= self.learning_rate * self._loss_derivative(y, prediction)

            self.estimators_.append(tree)

        return self

    def predict(self, X):
        """
        Predict using the Boosted Regression Tree model.

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
        
        y_pred = np.zeros(X.shape[0])

        for tree in self.estimators_:
            y_pred += self.learning_rate * tree.predict(X)

        return y_pred
    
class BoostedTreeClassifier(BaseEstimator, ClassifierMixin):
    r"""
    Boosted Decision Tree Classifier.

    This classifier employs an ensemble boosting method using decision trees. 
    It builds the model by sequentially adding trees, where each tree is trained 
    to correct the errors made by the previous ones. The final model's prediction 
    is the weighted aggregate of all individual trees' predictions, where weights 
    are adjusted by the learning rate.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the ensemble. More trees can lead to better 
        performance but increase computational complexity.

    max_depth : int, default=3
        The maximum depth of each decision tree. Controls the complexity of the 
        model. Deeper trees can capture more complex patterns but may overfit.

    learning_rate : float, default=0.1
        The rate at which the boosting process adapts to the errors of the 
        previous trees. A lower rate requires more trees to achieve similar 
        performance levels but can yield a more robust model.

    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators. Available after fitting.

    Notes
    -----
    The BoostedClassifierTree combines weak learners (decision trees) into a 
    stronger model. The boosting process iteratively adjusts the weights of 
    observations based on the previous trees' errors. The final prediction is 
    made based on a weighted majority vote (or sum) of the weak learners' predictions.

    The boosting procedure is mathematically formulated as:

    .. math::
        F_t(x) = F_{t-1}(x) + \\alpha_t h_t(x)

    where \( F_t(x) \) is the model at iteration \( t \), \( \\alpha_t \) is 
    the learning rate, and \( h_t(x) \) is the weak learner.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from gofast.estimators.boosting import BoostedTreeClassifier
    >>> X, y = make_classification(n_samples=100, n_features=4)
    >>> clf = BoostedTreeClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
    >>> clf.fit(X, y)
    >>> print(clf.predict(X))

    References
    ----------
    1. Y. Freund, R. E. Schapire, "A Decision-Theoretic Generalization of 
       On-Line Learning and an Application to Boosting", 1995.
    2. J. H. Friedman, "Greedy Function Approximation: A Gradient Boosting Machine", 
       Annals of Statistics, 2001.
    3. T. Hastie, R. Tibshirani, J. Friedman, "The Elements of Statistical Learning",
       Springer, 2009.

    See Also
    --------
    DecisionTreeClassifier : A decision tree classifier.
    AdaBoostClassifier : An adaptive boosting classifier.
    GradientBoostingClassifier : A gradient boosting machine for classification.
    """

    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

    def fit(self, X, y):
        """
        Fit the Boosted Decision Tree Classifier model to the data.

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
        residual = y.copy(); self.estimators_ = []
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X, residual)
            prediction = tree.predict(X)
            # Update residuals
            residual -= self.learning_rate * (prediction - residual)
            self.estimators_.append(tree)

        return self

    def predict(self, X):
        """
        Predict using the Boosted Decision Tree Classifier model.

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
        
        cumulative_prediction = np.zeros(X.shape[0])

        for tree in self.estimators_:
            cumulative_prediction += self.learning_rate * tree.predict(X)

        # Thresholding to convert to binary classification
        return np.where(cumulative_prediction > 0.5, 1, 0)
    
    def predict_proba(self, X):
        """
        Predict class probabilities using the Boosted Decision Tree 
        Classifier model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        proba : array-like of shape (n_samples, 2)
            The class probabilities of the input samples. Column 0 contains the
            probabilities of the negative class, and column 1 contains 
            the probabilities of the positive class.
            
        Example
        -------
        >>> from sklearn.datasets import make_classification
        >>> from gofast.estimators.boosting import BoostedClassifierTree
        >>> # Create a synthetic dataset
        >>> X, y = make_classification(n_samples=100, n_features=4, n_classes=2,
                                       random_state=42)
        >>> clf = BoostedClassifierTree(n_estimators=10, max_depth=3, learning_rate=0.1)
        >>> clf.fit(X, y)
        >>> # Predict probabilities
        >>> print(clf.predict_proba(X))
        """
        check_is_fitted(self, 'estimators_')
        X = check_array(X, accept_sparse=True)
        
        cumulative_prediction = np.zeros(X.shape[0])

        for tree in self.estimators_:
            cumulative_prediction += self.learning_rate * tree.predict(X)
        # Convert cumulative predictions to probabilities
        proba_positive_class = 1 / (1 + np.exp(-cumulative_prediction))  # Sigmoid function
        proba_negative_class = 1 - proba_positive_class

        return np.vstack((proba_negative_class, proba_positive_class)).T


class HybridBoostedTreeClassifier(BaseEstimator, ClassifierMixin):
    r"""
    Hybrid Boosted Tree Classifier.

    The Hybrid Boosted Tree Classifier is an ensemble learning model that 
    combines decision trees with gradient boosting techniques to tackle binary 
    classification tasks.
    By integrating multiple decision trees with varying weights, this classifier
    achieves high accuracy and reduces the risk of overfitting.

    Parameters
    ----------
    n_estimators : int, default=50
        The number of decision trees in the ensemble.
    learning_rate : float, default=0.1
        The learning rate for gradient boosting, controlling how much each tree
        influences the overall prediction.
    max_depth : int, default=3
        The maximum depth of each decision tree, determining the complexity of
        the model.

    Attributes
    ----------
    base_estimators_ : list of DecisionTreeClassifier
        List of base learners, each a DecisionTreeClassifier.
    weights_ : list
        Weights associated with each base learner, influencing their contribution
        to the final prediction.

    Example
    -------
    Here's an example of how to use the `HybridBoostedTreeClassifier` on the
    Iris dataset for binary classification:

    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> from gofast.estimators.boosting import HybridBoostedTreeClassifier

    >>> # Load the Iris dataset
    >>> iris = load_iris()
    >>> X = iris.data[:, :2]
    >>> y = (iris.target != 0) * 1  # Converting to binary classification

    >>> # Split the data into training and testing sets
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.3, random_state=0)

    >>> # Create and fit the HybridBoostedTreeClassifier
    >>> hybrid_boosted_tree = HybridBoostedTreeClassifier(
    ...     n_estimators=50, learning_rate=0.01, max_depth=3)
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
    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

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
    
        sample_weights = self._compute_sample_weights(y)  # Initialize sample weights
    
        for _ in range(self.n_estimators):
            # Fit a decision tree on the weighted dataset
            base_estimator = DecisionTreeClassifier(max_depth=self.max_depth)
            base_estimator.fit(X, y, sample_weight=sample_weights)
            
            # Calculate weighted error
            y_pred = base_estimator.predict(X)
            errors = (y != y_pred)
            weighted_error = np.sum(sample_weights * errors) / np.sum(sample_weights)
    
            if weighted_error == 0:
                # Prevent log(0) scenario
                continue
    
            # Weight calculation for this base estimator
            weight = self.learning_rate * np.log((1 - weighted_error) / weighted_error)
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
    
    def score(self, X, y):
        """
        Returns the mean accuracy on the given test data and labels.
    
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Test samples.
    
        y : array-like, shape (n_samples,)
            True labels for X.
    
        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        
        # Obtain predictions
        y_pred = self.predict(X)
        # Calculate and return the accuracy score
        return accuracy_score(y, y_pred)
    
class HybridBoostedTreeRegressor(BaseEstimator, RegressorMixin):
    r"""
    Hybrid Boosted Regression Tree (BRT) for regression tasks.

    The Hybrid Boosted Tree Regressor is a powerful ensemble learning model
    that combines multiple Boosted Regression Tree (BRT) models. Each BRT
    model is itself an ensemble created using boosting principles.
    
    This ensemble model combines multiple Boosted Regression Tree models,
    each of which is an ensemble in itself, created using the 
    principles of boosting.
    
    In `HybridBoostedTreeRegressor` class, the `n_estimators` parameter 
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
    Here's an example of how to initialize and use the `HybridBoostedTreeRegressor`:
    ```python
    from gofast.estimators.boosting import HybridBoostedTreeRegressor
    import numpy as np

    brt_params = {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1}
    hybrid_brt = HybridBoostedTreeRegressor(n_estimators=10, brt_params=brt_params)
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
           F_{k+1}(x) = F_k(x) + \text{learning_rate} \cdot h_k(x)

    where:
    - :math:`F_k(x)` is the prediction of the ensemble at iteration \(k\).
    - :math:`y` is the true target values.
    - :math:`\text{learning_rate}` is the learning rate, influencing the impact
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
            brt = GradientBoostingRegressor(**self.brt_params)
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

class EnsembleHBTClassifier(BaseEstimator, ClassifierMixin):
    r"""
    Hybrid Boosted Regression Tree Ensemble Classifier.

    This classifier leverages an ensemble of Boosted Decision Tree classifiers,
    each being a full implementation of the GradientBoostingClassifier. This 
    ensemble approach enhances prediction accuracy and robustness by combining
    the strengths of multiple boosted trees.

    In the `EnsembleHBTClassifier`, each classifier in the `gb_ensembles_` list 
    is an independent Boosted Decision Tree model. The `fit` method trains each 
    model on the entire dataset, while the `predict` method applies majority 
    voting among all models to determine the final class labels. The `gb_params` 
    parameter allows for customization of each individual Gradient Boosting model.

    Parameters
    ----------
    n_estimators : int
        The number of Boosted Decision Tree models in the ensemble.
    gb_params : dict
        Parameters to be passed to each GradientBoostingClassifier model, such
        as the number of boosting stages, learning rate, and tree depth.

    Attributes
    ----------
    gb_ensembles_ : list of GradientBoostingClassifier
        A list containing the fitted Boosted Decision Tree models.

    Notes
    -----
    The Hybrid Boosted Tree Ensemble Classifier uses majority voting based on 
    the predictions from multiple Boosted Decision Tree models. Mathematically,
    the ensemble's decision-making process is formulated as follows:

    .. math::
        C_{\text{final}}(x) = \text{argmax}\left(\sum_{i=1}^{n} \delta(C_i(x), \text{mode})\right)

    where:
    - :math:`C_{\text{final}}(x)` is the final predicted class label for input \(x\).
    - :math:`\delta(C_i(x), \text{mode})` is an indicator function that counts 
      the occurrence of the most frequent class label predicted by the \(i\)-th 
      Boosted Tree.
    - :math:`n` is the number of Boosted Decision Trees in the ensemble.
    - :math:`C_i(x)` is the class label predicted by the \(i\)-th Boosted Decision Tree.

    This ensemble method significantly reduces the likelihood of overfitting and 
    increases the predictive performance by leveraging the diverse predictive 
    capabilities of multiple models.
    
    Examples
    --------
    >>> from gofast.estimators.boosting import EnsembleHBTClassifier
    >>> gb_params = {'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.1}
    >>> hybrid_gb = HBTEnsembleClassifier(n_estimators=10,
                                              gb_params=gb_params)
    >>> X, y = np.random.rand(100, 4), np.random.randint(0, 2, 100)
    >>> hybrid_gb.fit(X, y)
    >>> y_pred = hybrid_gb.predict(X)
    """

    def __init__(self, n_estimators=10, gb_params=None):
        self.n_estimators = n_estimators
        self.gb_params = gb_params or {}
        

    def fit(self, X, y):
        """
        Fit the Hybrid Boosted Regression Tree Ensemble Classifier 
        model to the data.

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
        self.gb_ensembles_ = []
        for _ in range(self.n_estimators):
            gb = GradientBoostingClassifier(**self.gb_params)
            gb.fit(X, y)
            self.gb_ensembles_.append(gb)

        return self

    def predict(self, X):
        """
        Predict using the Hybrid Boosted Regression Tree Ensemble 
        Classifier model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted class labels.
        """
        check_is_fitted(self, 'gb_ensembles_')
        X = check_array(
            X,
            accept_large_sparse=True,
            accept_sparse= True,
            to_frame=False, 
            )
        predictions = np.array([gb.predict(X) for gb in self.gb_ensembles_])
        # Majority voting for classification
        y_pred = stats.mode(predictions, axis=0).mode[0]
        return y_pred
    
class EnsembleHBTRegressor(BaseEstimator, RegressorMixin):
    r"""
    Hybrid Boosted Tree Ensemble Regressor.

    This ensemble model combines decision trees with gradient boosting for 
    regression tasks. Designed to enhance prediction accuracy, the Hybrid 
    Boosted Tree Ensemble Regressor adjusts the weight of each decision tree 
    based on its performance, optimizing predictions across a wide range of 
    applications.

    Parameters
    ----------
    n_estimators : int, default=50
        The number of decision trees in the ensemble.
    learning_rate : float, default=0.1
        The learning rate for gradient boosting, affecting how rapidly the 
        model adapts to the problem.
    max_depth : int, default=3
        The maximum depth allowed for each decision tree, controlling complexity.

    Attributes
    ----------
    base_estimators_ : list of DecisionTreeRegressor
        List of base learners, each a DecisionTreeRegressor.
    weights_ : list
        Weights assigned to each base learner, influencing their impact on the 
        final prediction.

    Notes
    -----
    The Hybrid Boosted Tree Ensemble Regressor employs gradient boosting to
    enhance and correct predictions iteratively:

    1. Residual Calculation:
       .. math::
           \text{Residual} = y - F_k(x)

    2. Weighted Error Calculation:
       .. math::
           \text{Weighted Error} = \sum_{i=1}^{n} (weights_i \cdot \text{Residual}_i)^2

    3. Weight Calculation for Base Learners:
       .. math::
           \text{Weight} = \text{learning\_rate} \cdot \frac{1}{1 + \text{Weighted Error}}

    4. Update Predictions:
       .. math::
           F_{k+1}(x) = F_k(x) + \text{Weight} \cdot \text{Residual}

    where:
    - :math:`F_k(x)` is the prediction of the ensemble at iteration \(k\).
    - :math:`y` represents the true target values.
    - :math:`\text{Residual}` is the difference between the true values and the
      ensemble prediction.
    - :math:`\text{Weighted Error}` is the squared residuals weighted by their
      respective sample weights.
    - :math:`\text{Weight}` is the weight assigned to the predictions from the 
      new tree in the boosting process.
    - :math:`n` is the number of samples.
    - :math:`weights_i` is the weight of each sample.
    - :math:`\text{learning\_rate}` is a hyperparameter controlling the 
      influence of each new tree.

    Example
    -------
    Here's an example of how to use the `EnsembleHBTRegressor`
    on the Boston Housing dataset:

    >>> from sklearn.datasets import load_boston
    >>> from sklearn.model_selection import train_test_split
    >>> from gofast.estimators.boosting import EnsembleHBTRegressor

    >>> # Load the dataset
    >>> boston = load_boston()
    >>> X = boston.data
    >>> y = boston.target

    >>> # Split the data into training and testing sets
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.3, random_state=0)

    >>> # Create and fit the EnsembleHBTRegressor
    >>> hybrid_regressor = HBTEnsembleRegressor(
    ...     n_estimators=50, learning_rate=0.01, max_depth=3)
    >>> hybrid_regressor.fit(X_train, y_train)

    >>> # Make predictions and evaluate the model
    >>> y_pred = hybrid_regressor.predict(X_test)
    >>> mse = np.mean((y_pred - y_test) ** 2)
    >>> print('Mean Squared Error:', mse)

    Applications and Performance
    ----------------------------
    The Hybrid Boosted Tree Ensemble Regressor is a versatile model suitable
    for various regression tasks, including real estate price prediction,
    financial forecasting, and more. Its performance depends on the quality
    of the data, the choice of hyperparameters, and the number of estimators.

    When tuned correctly, it can achieve high accuracy and robustness, making
    it a valuable tool for predictive modeling.

    See Also
    --------
    - `sklearn.ensemble.GradientBoostingRegressor`: Scikit-learn's Gradient
      Boosting Regressor for comparison.
    - `sklearn.tree.DecisionTreeRegressor`: Decision tree regressor used as
      base learners in ensemble methods.
    - `sklearn.metrics.mean_squared_error`: A common metric for evaluating
      regression models.
    - gofast.estimators.boosting.HybridBoostedTreeRegressor: Hybrid Boosted Regression 
      Tree for regression tasks.
    """

    def __init__(self, n_estimators=50, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth


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
        F_k = np.zeros(len(y))

        for _ in range(self.n_estimators):
            # Calculate residuals
            residual = y - F_k
            
            # Fit a decision tree on the residuals
            base_estimator = DecisionTreeRegressor(max_depth=self.max_depth)
            base_estimator.fit(X, residual)
            
            # Calculate weighted error
            weighted_error = np.sum((residual - base_estimator.predict(X)) ** 2)
            
            # Calculate weight for this base estimator
            weight = self.learning_rate / (1 + weighted_error)
            
            # Update predictions
            F_k += weight * base_estimator.predict(X)
            
            # Store the base estimator and its weight
            self.base_estimators_.append(base_estimator)
            self.weights_.append(weight)
        
        return self

    def predict(self, X):
        """Return regression predictions.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Samples.

        Returns
        -------
        y_pred : array-like, shape = [n_samples]
            Predicted target values.
        """
        check_is_fitted(self, 'weights_')
        y_pred = np.zeros(X.shape[0])
        for i, base_estimator in enumerate(self.base_estimators_):
            y_pred += self.weights_[i] * base_estimator.predict(X)
        
        return y_pred
    

class _GradientBoostingRegressor:
    r"""
    A simple gradient boosting regressor for regression tasks.

    Gradient Boosting builds an additive model in a forward stage-wise fashion. 
    At each stage, regression trees are fit on the negative gradient of the loss function.

    Parameters
    ----------
    n_estimators : int, optional (default=100)
        The number of boosting stages to be run. This is essentially the 
        number of decision trees in the ensemble.

    learning_rate : float, optional (default=1.0)
        Learning rate shrinks the contribution of each tree. There is a 
        trade-off between learning_rate and n_estimators.
        
    max_depth : int, default=1
        The maximum depth of the individual regression estimators.
        
    Attributes
    ----------
    estimators_ : list of DecisionStumpRegressor
        The collection of fitted sub-estimators.

    Methods
    -------
    fit(X, y)
        Fit the gradient boosting model to the training data.
    predict(X)
        Predict continuous target values for samples in X.
    decision_function(X)
        Compute the raw decision scores for the input data.

    Mathematical Formula
    --------------------
    Given a differentiable loss function L(y, F(x)), the model is 
    constructed as follows:
    
    .. math:: 
        F_{m}(x) = F_{m-1}(x) + \\gamma_{m} h_{m}(x)

    where F_{m} is the model at iteration m, \\gamma_{m} is the step size, 
    and h_{m} is the weak learner.

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=100, n_features=1, noise=10)
    >>> model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
    >>> model.fit(X, y)
    >>> print(model.predict(X)[:5])
    >>> print(model.decision_function(X)[:5])

    References
    ----------
    - J. H. Friedman, "Greedy Function Approximation: A Gradient Boosting Machine," 1999.
    - T. Hastie, R. Tibshirani, and J. Friedman, "The Elements of Statistical Learning," Springer, 2009.

    See Also
    --------
    DecisionTreeRegressor, RandomForestRegressor, AdaBoostRegressor

    Applications
    ------------
    Gradient Boosting Regressor is commonly used in various regression tasks where the relationship 
    between features and target variable is complex and non-linear. It is particularly effective 
    in predictive modeling and risk assessment applications.
    """

    def __init__(self, n_estimators=100, learning_rate=1.0, max_depth=1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth=max_depth

    def fit(self, X, y):
        """
        Fit the gradient boosting regressor to the training data.

        The method sequentially adds decision stumps to the ensemble, each one 
        correcting its predecessor. The fitting process involves finding the best 
        stump at each stage that reduces the overall prediction error.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples used for training. Each row in X is a sample and each 
            column is a feature.

        y : array-like of shape (n_samples,)
            The target values (continuous). The regression targets are continuous 
            values which the model will attempt to predict.

        Raises
        ------
        ValueError
            If input arrays X and y have incompatible shapes.

        Notes
        -----
        - The fit process involves computing pseudo-residuals which are the gradients 
          of the loss function with respect to the model's predictions. These are used 
          as targets for the subsequent weak learner.
        - The model complexity increases with each stage, controlled by the learning rate.

        Examples
        --------
        >>> from sklearn.datasets import make_regression
        >>> X, y = make_regression(n_samples=100, n_features=1, noise=10)
        >>> reg = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1)
        >>> reg.fit(X, y)
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("Mismatched number of samples between X and y")

        # Initialize the prediction to zero
        F_m = np.zeros(y.shape)

        for m in range(self.n_estimators):
            # Compute residuals
            residuals = y - F_m

            # # Fit a regression tree to the negative gradient
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)

            # Update the model predictions
            F_m += self.learning_rate * tree.predict(X)

            # Store the fitted estimator
            self.estimators_.append(tree)

        return self

    def predict(self, X):
        """
        Predict continuous target values for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted target values.
        """
        F_m = sum(self.learning_rate * estimator.predict(X)
                  for estimator in self.estimators_)
        return F_m

    def decision_function(self, X):
        """
        Compute the raw decision scores for the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        decision_scores : array-like of shape (n_samples,)
            The raw decision scores for each sample.
        """
        return sum(self.learning_rate * estimator.predict(X)
                   for estimator in self.estimators_)
    












