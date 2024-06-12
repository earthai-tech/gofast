# -*- coding: utf-8 -*-

from __future__ import annotations
from collections import defaultdict 
from abc import ABCMeta
from abc import abstractmethod
import inspect 
import numpy as np 
from tqdm import tqdm

from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelBinarizer, StandardScaler, OneHotEncoder
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.utils import shuffle as skl_shuffle

from ..tools.validator import check_X_y, check_array, validate_fit_weights 
from ..tools.validator import check_is_fitted, get_estimator_name
from .util import activator 
try: 
    from skfuzzy.control import Antecedent
    import skfuzzy as fuzz
except: pass 


class BaseDTB(BaseEstimator, metaclass=ABCMeta):
    """
    Base class for Decision Tree-Based Ensembles.

    This class serves as a foundational base for creating ensemble
    models using decision trees. It encapsulates common functionality
    and parameters for both classification and regression tasks.
    This class is intended to be inherited by specific ensemble
    classes such as `DTBClassifier` and `DTBRegressor`.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the ensemble. This parameter controls
        how many decision trees will be fitted and aggregated to form
        the final model.

    max_depth : int, default=3
        The maximum depth of each decision tree in the ensemble. Limiting
        the depth of the trees helps prevent overfitting by restricting
        the complexity of the model.

    criterion : str, default="squared_error"
        The function to measure the quality of a split. Supported criteria
        for regression are:
        - "squared_error": mean squared error, which is equal to variance
          reduction as feature selection criterion.
        - "friedman_mse": mean squared error with improvement score by Friedman.
        - "absolute_error": mean absolute error.
        - "poisson": Poisson deviance.

        For classification, supported criteria are:
        - "gini": Gini impurity.
        - "entropy": Information gain (entropy).

    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node. Supported
        strategies are:
        - "best": Choose the best split.
        - "random": Choose the best random split.

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
        The minimum weighted fraction of the sum total of weights
        (of all the input samples) required to be at a leaf node.

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

    verbose : int, default=0
        Controls the verbosity when fitting and predicting. Higher values
        indicate more messages.

    Methods
    -------
    fit(X, y, sample_weight=None)
        Fit the ensemble model to the data.

    predict(X)
        Predict using the fitted ensemble model.

    Notes
    -----
    The `BaseDTB` class combines the predictive power of multiple decision
    trees by averaging their predictions (for regression) or using majority
    voting (for classification). This reduces variance and improves accuracy
    over a single decision tree. The ensemble prediction is computed by
    aggregating the predictions from each individual tree within the ensemble.

    The mathematical formulation for regression can be described as follows:

    .. math::
        y_{\text{pred}} = \frac{1}{N} \sum_{i=1}^{N} y_{\text{tree}_i}

    where:
    - :math:`N` is the number of trees in the ensemble.
    - :math:`y_{\text{pred}}` is the final predicted value aggregated from
      all trees.
    - :math:`y_{\text{tree}_i}` represents the prediction made by the
      :math:`i`-th tree.

    For classification, the final class label is determined by majority voting:

    .. math::
        C_{\text{final}}(x) = \text{argmax}\left(\sum_{i=1}^{n} \delta(C_i(x), \text{mode})\right)

    where:
    - :math:`C_{\text{final}}(x)` is the final predicted class label for input :math:`x`.
    - :math:`\delta(C_i(x), \text{mode})` is an indicator function that counts the
      occurrence of the most frequent class label predicted by the :math:`i`-th tree.
    - :math:`n` is the number of decision trees in the ensemble.
    - :math:`C_i(x)` is the class label predicted by the :math:`i`-th decision tree.

    Examples
    --------
    Here's an example of how to use the `DTBClassifier` and `DTBRegressor`:

    >>> from gofast.estimators.tree import DTBClassifier, DTBRegressor
    >>> from sklearn.datasets import make_classification, make_regression
    >>> from sklearn.model_selection import train_test_split

    Classification Example:

    >>> X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, 
    ...                                                     test_size=0.3, 
    ...                                                     random_state=42)
    >>> clf = DTBClassifier(n_estimators=50, max_depth=3, random_state=42)
    >>> clf.fit(X_train, y_train)
    >>> y_pred = clf.predict(X_test)
    >>> print("Predicted class labels:", y_pred)

    Regression Example:

    >>> X, y = make_regression(n_samples=100, n_features=20, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, 
    ...                                                     test_size=0.3, 
    ...                                                     random_state=42)
    >>> reg = DTBRegressor(n_estimators=50, max_depth=3, random_state=42)
    >>> reg.fit(X_train, y_train)
    >>> y_pred = reg.predict(X_test)
    >>> print("Predicted values:", y_pred)

    See Also
    --------
    sklearn.ensemble.RandomForestClassifier : A popular ensemble method
        based on decision trees for classification tasks.
    sklearn.ensemble.RandomForestRegressor : A popular ensemble method
        based on decision trees for regression tasks.
    sklearn.tree.DecisionTreeClassifier : Decision tree classifier used as
        base learners in ensemble methods.
    sklearn.tree.DecisionTreeRegressor : Decision tree regressor used as
        base learners in ensemble methods.

    References
    ----------
    .. [1] Breiman, L. "Bagging predictors." Machine learning 24.2 (1996): 123-140.
    .. [2] Friedman, J., Hastie, T., & Tibshirani, R. "The Elements of Statistical
           Learning." Springer Series in Statistics. (2001).
    """

    @abstractmethod
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
        self.verbose = verbose
        
    def fit(self, X, y, sample_weight=None):
        """
        Fit the ensemble model to the data.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples. It can be a dense or sparse matrix.
    
        y : array-like of shape (n_samples,)
            The target values (class labels or continuous values).
    
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample. If `None`, all samples are given
            equal weight. This parameter is used to account for different
            importance of samples during the training process.
    
        Returns
        -------
        self : object
            Returns self.
    
        Notes
        -----
        The `fit` method trains the ensemble model on the input data `X` and
        target labels `y`. The process involves the following steps:
    
        1. **Input Validation**: The input data `X` and target values `y` are
           checked for validity and conformance to the expected format and type
           using `check_X_y`.
    
        2. **Estimator Initialization**: An empty list `estimators_` is created
           to store the individual decision trees that will be trained.
    
        3. **Sample Indices Generation**: An array of sample indices is created
           to facilitate the selection of samples for training each tree.
    
        4. **Progress Bar Setup**: If `verbose` is greater than 0, a progress bar
           is initialized to provide visual feedback during the fitting process.
    
        5. **Bootstrap Sampling**: For each estimator, a subset of samples is
           randomly selected with replacement (if `bootstrap` is `True`) or without
           replacement (if `bootstrap` is `False`).
    
        6. **Tree Training**: A new decision tree is created and trained on the
           selected subset of samples.
    
        7. **Out-of-Bag Score Calculation**: If `bootstrap` is `True` and
           `subsample` is less than 1.0, the out-of-bag (OOB) score is calculated
           as the mean squared error on the samples not used for training (for
           regression tasks).
    
        The mathematical formulation of the ensemble prediction process is as follows:
    
        .. math::
            y_{\text{pred}} = \frac{1}{N} \sum_{i=1}^{N} y_{\text{tree}_i}
    
        where:
        - :math:`N` is the number of trees in the ensemble.
        - :math:`y_{\text{pred}}` is the final predicted value aggregated from
          all trees.
        - :math:`y_{\text{tree}_i}` represents the prediction made by the
          :math:`i`-th tree.
    
        Examples
        --------
        Here's an example of how to use the `fit` method with `DTBClassifier` and
        `DTBRegressor`:
    
        >>> from gofast.estimators.ensemble import DTBClassifier, DTBRegressor
        >>> from sklearn.datasets import make_classification, make_regression
        >>> from sklearn.model_selection import train_test_split
    
        Classification Example:
    
        >>> X, y = make_classification(n_samples=100, n_features=20, random_state=42)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, 
        ...                                                     test_size=0.3, 
        ...                                                     random_state=42)
        >>> clf = DTBClassifier(n_estimators=50, max_depth=3, random_state=42)
        >>> clf.fit(X_train, y_train)
        >>> y_pred = clf.predict(X_test)
        >>> print("Predicted class labels:", y_pred)
    
        Regression Example:
    
        >>> X, y = make_regression(n_samples=100, n_features=20, random_state=42)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, 
        ...                                                     test_size=0.3, 
        ...                                                     random_state=42)
        >>> reg = DTBRegressor(n_estimators=50, max_depth=3, random_state=42)
        >>> reg.fit(X_train, y_train)
        >>> y_pred = reg.predict(X_test)
        >>> print("Predicted values:", y_pred)
    
        See Also
        --------
        sklearn.utils.validation.check_X_y : Utility function to check the input
            data and target values.
        sklearn.utils.validation.check_array : Utility function to check the input
            array.
        sklearn.ensemble.BaggingClassifier : A bagging classifier.
        sklearn.ensemble.BaggingRegressor : A bagging regressor.
        sklearn.ensemble.GradientBoostingClassifier : A gradient boosting classifier.
        sklearn.ensemble.GradientBoostingRegressor : A gradient boosting regressor.
    
        References
        ----------
        .. [1] Breiman, L. "Bagging predictors." Machine learning 24.2 (1996): 123-140.
        .. [2] Friedman, J., Hastie, T., & Tibshirani, R. "The Elements of Statistical
               Learning." Springer Series in Statistics. (2001).
    
        """
        X, y = check_X_y(X, y, accept_sparse=True)
        self.estimators_ = []
        n_samples = X.shape[0]
        sample_indices = np.arange(n_samples)
        sample_weight = validate_fit_weights(y, sample_weight=sample_weight)
        if self.verbose > 0:
            progress_bar = tqdm(
                range(self.n_estimators), ascii=True, ncols=100,
                desc=f'Fitting {self.__class__.__name__}', 
            )
    
        for i in range(self.n_estimators):
            if self.bootstrap:
                subsample_indices = np.random.choice(
                    sample_indices, size=int(self.subsample * n_samples), replace=True
                )
            else:
                subsample_indices = sample_indices[:int(self.subsample * n_samples)]
    
            tree = self._create_tree()
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
        Predict using the ensemble model.
    
        This method generates predictions for the input samples `X`
        using the fitted ensemble model. It checks if the model is fitted
        and then uses the base estimators within the ensemble to make
        predictions.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. It can be a dense or sparse matrix.
    
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted values (class labels or continuous values).
    
        Notes
        -----
        The `predict` method performs the following steps:
    
        1. **Check if Fitted**: The method checks if the model has been fitted
           by verifying the presence of the `estimators_` attribute. If the model
           is not fitted, it raises a `NotFittedError`.
    
        2. **Input Validation**: The input samples `X` are validated using
           `check_array` to ensure they conform to the expected format and type.
           This includes handling both dense and sparse matrices.
    
        3. **Prediction**: The method collects predictions from each individual
           base estimator in the ensemble. These predictions are then aggregated
           to form the final output. For regression tasks, the predictions are
           averaged. For classification tasks, majority voting is used to determine
           the final class label.
    
        The mathematical formulation of the prediction process is as follows:
    
        For regression:
    
        .. math::
            y_{\text{pred}} = \frac{1}{N} \sum_{i=1}^{N} y_{\text{tree}_i}
    
        where:
        - :math:`N` is the number of trees in the ensemble.
        - :math:`y_{\text{pred}}` is the final predicted value aggregated from
          all trees.
        - :math:`y_{\text{tree}_i}` represents the prediction made by the
          :math:`i`-th tree.
    
        For classification:
    
        .. math::
            C_{\text{final}}(x) = \text{argmax}\left(\sum_{i=1}^{n} \delta(C_i(x), \text{mode})\right)
    
        where:
        - :math:`C_{\text{final}}(x)` is the final predicted class label for input :math:`x`.
        - :math:`\delta(C_i(x), \text{mode})` is an indicator function that counts the
          occurrence of the most frequent class label predicted by the :math:`i`-th tree.
        - :math:`n` is the number of decision trees in the ensemble.
        - :math:`C_i(x)` is the class label predicted by the :math:`i`-th decision tree.
    
        Examples
        --------
        Here's an example of how to use the `predict` method with `DTBClassifier` and
        `DTBRegressor`:
    
        >>> from gofast.estimators.ensemble import DTBClassifier, DTBRegressor
        >>> from sklearn.datasets import make_classification, make_regression
        >>> from sklearn.model_selection import train_test_split
    
        Classification Example:
    
        >>> X, y = make_classification(n_samples=100, n_features=20, random_state=42)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, 
        ...                                                     test_size=0.3, 
        ...                                                     random_state=42)
        >>> clf = DTBClassifier(n_estimators=50, max_depth=3, random_state=42)
        >>> clf.fit(X_train, y_train)
        >>> y_pred = clf.predict(X_test)
        >>> print("Predicted class labels:", y_pred)
    
        Regression Example:
    
        >>> X, y = make_regression(n_samples=100, n_features=20, random_state=42)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, 
        ...                                                     test_size=0.3, 
        ...                                                     random_state=42)
        >>> reg = DTBRegressor(n_estimators=50, max_depth=3, random_state=42)
        >>> reg.fit(X_train, y_train)
        >>> y_pred = reg.predict(X_test)
        >>> print("Predicted values:", y_pred)
    
        See Also
        --------
        sklearn.utils.validation.check_array : Utility function to check the input
            array.
        sklearn.utils.validation.check_is_fitted : Utility function to check if the
            estimator is fitted.
        sklearn.ensemble.BaggingClassifier : A bagging classifier.
        sklearn.ensemble.BaggingRegressor : A bagging regressor.
        sklearn.ensemble.GradientBoostingClassifier : A gradient boosting classifier.
        sklearn.ensemble.GradientBoostingRegressor : A gradient boosting regressor.
    
        References
        ----------
        .. [1] Breiman, L. "Bagging predictors." Machine learning 24.2 (1996): 123-140.
        .. [2] Friedman, J., Hastie, T., & Tibshirani, R. "The Elements of Statistical
               Learning." Springer Series in Statistics. (2001).
    
        """
        check_is_fitted(self, 'estimators_')
        X = check_array(X, accept_sparse=True)
        predictions = np.array([tree.predict(X) for tree in self.estimators_])
        return self._aggregate_predictions(predictions)


    @abstractmethod
    def _create_tree(self):
        """
        Create a new decision tree instance.
    
        This abstract method should be implemented by subclasses to return
        an instance of a decision tree model (`DecisionTreeClassifier` or
        `DecisionTreeRegressor`) configured with the appropriate parameters.
    
        Returns
        -------
        tree : DecisionTreeClassifier or DecisionTreeRegressor
            A new instance of a decision tree model.
        """
        pass
    
    @abstractmethod
    def _aggregate_predictions(self, predictions):
        """
        Aggregate predictions from multiple trees.
    
        This abstract method should be implemented by subclasses to combine
        the predictions from all individual trees in the ensemble. For
        regression, this might involve averaging the predictions. For
        classification, this might involve majority voting.
    
        Parameters
        ----------
        predictions : array-like of shape (n_estimators, n_samples)
            The predictions from all individual trees.
    
        Returns
        -------
        aggregated_predictions : array-like of shape (n_samples,)
            The aggregated predictions.
        """
        pass


class GradientDescentBase(BaseEstimator, metaclass=ABCMeta):
    """
    Base class for Gradient Descent optimization.

    This class implements the core functionality of gradient descent, which is
    used for both regression and classification tasks. It includes parameters
    for controlling the learning rate, stopping criteria, and other aspects
    of the training process.

    Parameters
    ----------
    eta0 : float, default=0.01
        The initial learning rate for weight updates. A smaller learning rate
        makes the training process more stable but slower.

    max_iter : int, default=1000
        Maximum number of iterations over the training data (epochs).

    tol : float, default=1e-4
        Tolerance for the optimization. Training will stop when the validation
        score is not improving by at least `tol` for `n_iter_no_change` consecutive
        epochs.

    early_stopping : bool, default=False
        Whether to use early stopping to terminate training when validation score
        is not improving.

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for early
        stopping. It is used only if `early_stopping` is set to `True`.

    n_iter_no_change : int, default=5
        Maximum number of epochs to not meet `tol` improvement. The training will
        stop if the validation score is not improving by at least `tol` for `n_iter_no_change`
        consecutive epochs.

    learning_rate : {'constant', 'invscaling', 'adaptive'}, default='constant'
        Learning rate schedule for weight updates.
        - 'constant': learning rate remains constant.
        - 'invscaling': gradually decreases the learning rate at each time step.
        - 'adaptive': keeps the learning rate constant to `eta0` as long as training
          loss keeps decreasing.

    power_t : float, default=0.5
        The exponent for inverse scaling learning rate. It is used only when
        `learning_rate='invscaling'`.

    alpha : float, default=0.0001
        L2 penalty (regularization term) parameter. Regularization helps to prevent
        overfitting by penalizing large weights.

    batch_size : int, default='auto'
        Size of minibatches for stochastic optimizers. If 'auto', batch size is set
        to `min(200, n_samples)`.

    clipping_threshold : float, default=250
        Threshold for clipping gradients to avoid overflow during the computation of
        the sigmoid function. This prevents numerical instability in the training process.

    shuffle : bool, default=True
        Whether to shuffle the training data before each epoch.

    random_state : int, RandomState instance, default=None
        Seed for the random number generator for weight initialization and shuffling
        the training data.

    verbose : bool, default=False
        Whether to print progress messages to stdout. If `True`, progress messages are
        printed during training.

    Attributes
    ----------
    weights_ : ndarray of shape (n_features + 1, n_outputs)
        Coefficients of the features in the decision function.

    n_iter_ : int
        Number of iterations run.

    best_loss_ : float
        Best loss achieved during training.

    no_improvement_count_ : int
        Number of iterations with no improvement in validation score.

    Examples
    --------
    >>> from gofast.estimators.perceptron import GradientDescentBase
    >>> import numpy as np
    >>> X = np.array([[0, 0], [1, 1], [2, 2]])
    >>> y = np.array([0, 1, 2])
    >>> gd = GradientDescentBase(eta0=0.01, max_iter=100)
    >>> gd._fit(X, y, is_classifier=False)
    >>> gd._predict(X)

    Notes
    -----
    Gradient Descent is a fundamental optimization technique used in various
    machine learning algorithms. This implementation allows for flexibility
    and control over the learning process through its numerous parameters.

    See Also
    --------
    SGDClassifier : Linear classifiers (SVM, logistic regression, etc.) with SGD training.
    SGDRegressor : Linear model fitted by minimizing a regularized empirical loss with SGD.

    References
    ----------
    .. [1] Bottou, L. (2010). "Large-Scale Machine Learning with Stochastic Gradient Descent".
           Proceedings of COMPSTAT'2010.
    .. [2] Ruder, S. (2016). "An overview of gradient descent optimization algorithms".
           arXiv preprint arXiv:1609.04747.
    """
    @abstractmethod
    def __init__(
        self, 
        *, 
        eta0=0.01, 
        max_iter=1000, 
        tol=1e-4, 
        early_stopping=False, 
        validation_fraction=0.1, 
        n_iter_no_change=5,
        learning_rate='constant', 
        power_t=0.5, 
        alpha=0.0001, 
        batch_size='auto',
        clipping_threshold=250, 
        activation='sigmoid',
        shuffle=True, 
        random_state=None, 
        verbose=False
    ):
        self.eta0 = eta0
        self.max_iter = max_iter
        self.tol = tol
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.learning_rate = learning_rate
        self.power_t = power_t
        self.alpha = alpha
        self.batch_size = batch_size
        self.clipping_threshold = clipping_threshold
        self.activation=activation
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose

    def _initialize_weights(self, n_features, n_outputs):
        self.weights_ = np.zeros((n_features + 1, n_outputs))
        self.n_iter_ = 0
        self.best_loss_ = np.inf
        self.no_improvement_count_ = 0

    def _update_weights(self, X_batch, y_batch):
        errors = y_batch - self._net_input(X_batch)
        if self.learning_rate == 'invscaling':
            eta = self.eta0 / (self.n_iter_ ** self.power_t)
        elif self.learning_rate == 'adaptive':
            eta = (
                self.eta0 if self.no_improvement_count_ < self.n_iter_no_change 
                else self.eta0 / 10
            )
        else:
            eta = self.eta0

        self.weights_[1:] += eta * X_batch.T.dot(errors) - self.alpha * self.weights_[1:]
        self.weights_[0] += eta * errors.sum(axis=0)
        
    def _fit(self, X, y, is_classifier):
        """
        Fit the model according to the given training data.
    
        This method performs the core training process using gradient descent. It
        handles both regression and classification tasks, with different loss functions
        and weight updates for each case. The method iterates over the training data,
        updates the model weights, and optionally uses early stopping based on the
        validation score.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data where `n_samples` is the number of samples and
            `n_features` is the number of features.
    
        y : array-like of shape (n_samples,)
            Target values. For classification tasks, these are the class labels. For
            regression tasks, these are the continuous target values.
    
        is_classifier : bool
            Indicates whether the task is classification (`True`) or regression (`False`).
    
        Returns
        -------
        self : object
            Returns self.
    
        Notes
        -----
        The fitting process involves several steps:
        
        1. **Data Validation and Preprocessing**:
           Ensures that `X` and `y` have correct shapes and types. For classification,
           the target values `y` are binarized if there are multiple classes.
    
        2. **Initialization**:
           Initializes the model weights and various parameters needed for the training
           loop.
    
        3. **Gradient Descent Optimization**:
           Iteratively updates the weights to minimize the loss function. The update
           rule for the weights depends on the learning rate schedule and regularization
           term.
    
        4. **Loss Computation**:
           Computes the loss function value for each iteration. For classification, the
           loss function is the cross-entropy loss, while for regression, it is the mean
           squared error.
    
        5. **Early Stopping**:
           Optionally stops the training early if the validation score does not improve
           for a specified number of iterations.
    
        The weight update rule for gradient descent can be formalized as:
    
        .. math::
            w := w - \eta \nabla J(w)
    
        where:
        - :math:`w` represents the weight vector of the model.
        - :math:`\eta` is the learning rate.
        - :math:`\nabla J(w)` denotes the gradient of the loss function with respect
          to the weights.
    
        Examples
        --------
        >>> from gofast.estimators.perceptron import GradientDescentBase
        >>> import numpy as np
        >>> X = np.array([[0, 0], [1, 1], [2, 2]])
        >>> y = np.array([0, 1, 2])
        >>> gd = GradientDescentBase(eta0=0.01, max_iter=100)
        >>> gd._fit(X, y, is_classifier=False)
        >>> gd._predict(X)
    
        See Also
        --------
        SGDClassifier : Linear classifiers (SVM, logistic regression, etc.) with SGD training.
        SGDRegressor : Linear model fitted by minimizing a regularized empirical loss with SGD.
    
        """
        X, y = check_X_y(X, y, estimator=self)
        self.is_classifier = is_classifier
        if is_classifier:
            self.classes_ = np.unique(y)
            n_classes = len(self.classes_)
            self.label_binarizer_ = LabelBinarizer()
            y = self.label_binarizer_.fit_transform(y)
            if n_classes == 2:
                y = np.hstack([1 - y, y])
        else:
            y = y.reshape(-1, 1)
    
        n_samples, n_features = X.shape
        self._initialize_weights(n_features, y.shape[1])
    
        if self.verbose:
            progress_bar = tqdm(total=self.max_iter, ascii=True, ncols=100,
                                desc=f'Fitting {self.__class__.__name__}')
    
        for i in range(self.max_iter):
            if self.shuffle:
                X, y = skl_shuffle(X, y, random_state=self.random_state)
    
            self._update_weights(X, y)
            if is_classifier:
                net_input = self._net_input(X)
                proba = activator(net_input, self.activation,
                                  clipping_threshold=self.clipping_threshold)
                # proba = self._sigmoid(net_input)
                loss = -np.mean(y * np.log(proba + 1e-9) + (1 - y) * np.log(1 - proba + 1e-9))
            else:
                loss = ((y - self._net_input(X)) ** 2).mean()
    
            if loss < self.best_loss_ - self.tol:
                self.best_loss_ = loss
                self.no_improvement_count_ = 0
            else:
                self.no_improvement_count_ += 1
    
            if self.early_stopping and self.no_improvement_count_ >= self.n_iter_no_change:
                if self.verbose:
                    print(f"\nEarly stopping at iteration {i+1}")
                    progress_bar.update(self.max_iter - i)
                break
    
            if self.verbose:
                progress_bar.update(1)
            self.n_iter_ += 1
    
        if self.verbose:
            progress_bar.close()
    
        return self

    def _net_input(self, X):
        """ Compute the Net Input """
        return np.dot(X, self.weights_[1:]) + self.weights_[0]

    def _sigmoid(self, z):
        """ Use sigmoid for probaiblity estimation"""
        # Use clipping_threshold to avoid overflow
        z = np.clip(z, -self.clipping_threshold, self.clipping_threshold)  
        return 1 / (1 + np.exp(-z))
    
    def _predict(self, X):
        """
        Predict class labels or continuous values for samples in X.
    
        This method predicts class labels for classification tasks and continuous 
        values for regression tasks based on the net input computed from the model 
        weights and input features.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples for which to predict the class labels or continuous values.
    
        Returns
        -------
        y_pred : array of shape (n_samples,)
            The predicted class labels for classification tasks, or continuous target 
            values for regression tasks.
    
        Notes
        -----
        For classification tasks, the predicted class label for each sample in `X` is 
        determined based on the net input values. If there are two classes, the prediction 
        is based on whether the net input is greater than or equal to zero. For multi-class 
        classification, the predicted class is determined by the class with the highest 
        net input value.
    
        Examples
        --------
        >>> from gofast.estimators.perceptron import GradientDescentBase
        >>> import numpy as np
        >>> X = np.array([[0, 0], [1, 1], [2, 2]])
        >>> y = np.array([0, 1, 2])
        >>> gd = GradientDescentBase(eta0=0.01, max_iter=100)
        >>> gd._fit(X, y, is_classifier=True)
        >>> gd._predict(X)
    
        See Also
        --------
        sklearn.linear_model.SGDClassifier : 
            Linear classifiers (SVM, logistic regression, etc.) with SGD training.
        sklearn.linear_model.SGDRegressor : 
            Linear model fitted by minimizing a regularized empirical loss with SGD.
        
        """
        net_input = self._net_input(X)
        if self.is_classifier:
            if net_input.shape[1] == 1:
                return np.where(net_input >= 0.0, self.label_binarizer_.classes_[1], 
                                self.label_binarizer_.classes_[0])
            else:
                return self.label_binarizer_.inverse_transform(net_input)
        else:
            return net_input

    def _predict_proba(self, X):
        """
        Predict class probabilities for samples in X for classification tasks.
    
        This method returns the probability estimates for each class for classification 
        tasks based on the net input computed from the model weights and input features. 
        For regression tasks, this method is not implemented.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples for which to predict the class probabilities.
    
        Returns
        -------
        y_proba : array of shape (n_samples, n_classes)
            The predicted class probabilities for each sample. The columns correspond to 
            the classes in sorted order, as they appear in the attribute `classes_`.
    
        Notes
        -----
        The predicted probabilities represent the likelihood of each class for the input 
        samples. The probabilities are computed using the sigmoid function applied to the 
        net input values for binary classification tasks. For multi-class classification, 
        the softmax function is used to normalize the probabilities across all classes.
    
        The sigmoid function is defined as:
    
        .. math::
            \sigma(z) = \frac{1}{1 + \exp(-z)}
    
        where :math:`z` is the net input.
    
        For multi-class classification, the softmax function is defined as:
    
        .. math::
            P(y = k | x) = \frac{\exp(\hat{y}_k)}{\sum_{j=1}^{C} \exp(\hat{y}_j)}
    
        where :math:`\hat{y}_k` is the raw model output for class `k`, and :math:`C` is the 
        number of classes.
    
        Examples
        --------
        >>> from gofast.estimators.perceptron import GradientDescentBase
        >>> import numpy as np
        >>> X = np.array([[0, 0], [1, 1], [2, 2]])
        >>> y = np.array([0, 1, 2])
        >>> gd = GradientDescentBase(eta0=0.01, max_iter=100)
        >>> gd._fit(X, y, is_classifier=True)
        >>> gd._predict_proba(X)
    
        See Also
        --------
        sklearn.linear_model.SGDClassifier :
            Linear classifiers (SVM, logistic regression, etc.) with SGD training.
        
        References
        ----------
        .. [1] Bottou, L. (2010). "Large-Scale Machine Learning with Stochastic
          Gradient Descent".Proceedings of COMPSTAT'2010.
        .. [2] Ruder, S. (2016). "An overview of gradient descent optimization
          algorithms".arXiv preprint arXiv:1609.04747.
        """
        if not self.is_classifier:
            raise NotImplementedError(
                "Probability estimates are not available for regression.")
    
        net_input = self._net_input(X)
        proba = activator(net_input, self.activation,
                          clipping_threshold=self.clipping_threshold)
        # proba = self._sigmoid(net_input)
        if proba.shape[1] == 1:
            return np.vstack([1 - proba.ravel(), proba.ravel()]).T
        else:
            return proba / proba.sum(axis=1, keepdims=True)

class EnsembleBase(BaseEstimator, metaclass=ABCMeta):
    """
    EnsembleBase

    The `EnsembleBase` class serves as a foundational abstract base
    class for creating ensemble learning models. This class
    encapsulates the common functionality and parameters required
    for implementing both ensemble classifiers and regressors.
    
    Parameters
    ----------
    base_estimator : estimator object, default=None
        The base estimator to fit on random subsets of the dataset. 
        If `None`, the default base estimator will be used, which is 
        defined in the subclasses.
        
    n_estimators : int, default=50
        The number of base estimators in the ensemble. For the 
        hybrid strategy, this is the total number of base estimators 
        combined across bagging and boosting.
        
    eta0 : float, default=0.1
        Learning rate that shrinks the contribution of each base 
        estimator by `eta0`. This parameter is only used for the 
        boosting and hybrid strategies.
        
    max_depth : int, default=3
        The maximum depth of the individual estimators. This controls 
        the complexity of each base estimator.
        
    strategy : {'hybrid', 'bagging', 'boosting'}, default='hybrid'
        The strategy to use for the ensemble. Options are:
        - 'bagging': Use Bagging strategy.
        - 'boosting': Use Boosting strategy.
        - 'hybrid': Combine Bagging and Boosting strategies.
        
    random_state : int or RandomState, default=None
        Controls the randomness of the estimator for reproducibility. 
        Pass an int for reproducible output across multiple function 
        calls.
        
    max_samples : float or int, default=1.0
        The number of samples to draw from `X` to train each base 
        estimator. If float, then draw `max_samples * n_samples` samples.
        
    max_features : int or float, default=1.0
        The number of features to draw from `X` to train each base 
        estimator. If float, then draw `max_features * n_features` features.
        
    bootstrap : bool, default=True
        Whether samples are drawn with replacement. If False, sampling 
        without replacement is performed.
        
    bootstrap_features : bool, default=False
        Whether features are drawn with replacement.
        
    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate the 
        generalization error.
        
    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to 
        fit and add more estimators to the ensemble.
        
    n_jobs : int, default=None
        The number of jobs to run in parallel for both `fit` and `predict`. 
        `None` means 1 unless in a `joblib.parallel_backend` context.
        
    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the 
        impurity greater than or equal to this value. Used to control 
        tree growth.
        
    init : estimator object, default=None
        An estimator object that is used to compute the initial predictions. 
        Used only for boosting.
        
    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node.
        - If int, consider `min_samples_split` as the minimum number.
        - If float, `min_samples_split` is a fraction and 
          `ceil(min_samples_split * n_samples)` is the minimum number 
          of samples for each split.
        
    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node. 
        A split point at any depth will only be considered if it leaves 
        at least `min_samples_leaf` training samples in each of the left 
        and right branches.
        - If int, consider `min_samples_leaf` as the minimum number.
        - If float, `min_samples_leaf` is a fraction and 
          `ceil(min_samples_leaf * n_samples)` is the minimum number 
          of samples for each node.
        
    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights 
        required to be at a leaf node.
        
    max_leaf_nodes : int, default=None
        Grow trees with `max_leaf_nodes` in best-first fashion. Best 
        nodes are defined as relative reduction in impurity. If None, 
        unlimited number of leaf nodes.
        
    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set 
        for early stopping. Used only for boosting.
        
    n_iter_no_change : int, default=None
        Used to decide if early stopping will be used to terminate 
        training when validation score is not improving. Used only for 
        boosting.
        
    tol : float, default=1e-4
        Tolerance for the early stopping. Used only for boosting.
        
    ccp_alpha : float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning.
    
    verbose : int, default=0
        Controls the verbosity when fitting and predicting. Higher values 
        indicate more messages.
        
    Attributes
    ----------
    model_ : object
        The fitted ensemble model.
    
    Methods
    -------
    fit(X, y, sample_weight=None)
        Fit the ensemble model to the data.
        
    predict(X)
        Predict using the fitted ensemble model.
    
    Notes
    -----
    This model combines the predictive power of multiple base estimators 
    through bagging, boosting, or a hybrid approach, effectively reducing 
    variance and improving accuracy over a single estimator.
    
    The hybrid strategy uses a combination of bagging and boosting, 
    where the bagging model contains boosting models as its base estimators. 
    This leverages the strengths of both approaches to achieve better 
    performance.
    
    .. math::
        y_{\text{pred}} = \frac{1}{N} \sum_{i=1}^{N} y_{\text{tree}_i}
    
    where:
    - :math:`N` is the number of base estimators in the ensemble.
    - :math:`y_{\text{pred}}` is the final predicted value aggregated from 
      all base estimators.
    - :math:`y_{\text{tree}_i}` represents the prediction made by the 
      :math:`i`-th base estimator.
    
    Examples
    --------
    Here's an example of how to use the `EnsembleBase` class on a dataset:
    
    >>> from gofast.estimators._base import EnsembleBase
    >>> from sklearn.datasets import make_classification, make_regression
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.metrics import accuracy_score, mean_squared_error
    
    Classification Example:
    
    >>> X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, 
    ...                                                     test_size=0.3, 
    ...                                                     random_state=42)
    >>> clf = EnsembleClassifier(n_estimators=50, strategy='hybrid', 
    ...                          random_state=42)
    >>> clf.fit(X_train, y_train)
    >>> y_pred = clf.predict(X_test)
    >>> print("Classification accuracy:", accuracy_score(y_test, y_pred))
    
    Regression Example:
    
    >>> X, y = make_regression(n_samples=100, n_features=20, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, 
    ...                                                     test_size=0.3, 
    ...                                                     random_state=42)
    >>> reg = EnsembleRegressor(n_estimators=50, strategy='hybrid', 
    ...                         random_state=42)
    >>> reg.fit(X_train, y_train)
    >>> y_pred = reg.predict(X_test)
    >>> print("Regression MSE:", mean_squared_error(y_test, y_pred))
    
    See Also
    --------
    sklearn.ensemble.BaggingClassifier : A bagging classifier.
    sklearn.ensemble.GradientBoostingClassifier : A gradient boosting classifier.
    sklearn.ensemble.BaggingRegressor : A bagging regressor.
    sklearn.ensemble.GradientBoostingRegressor : A gradient boosting regressor.
    sklearn.metrics.accuracy_score : A common metric for evaluating classification models.
    sklearn.metrics.mean_squared_error : A common metric for evaluating regression models.
    
    References
    ----------
    .. [1] Breiman, L. "Bagging predictors." Machine learning 24.2 (1996): 123-140.
    .. [2] Freund, Y., & Schapire, R. E. "Experiments with a new boosting algorithm."
           icml. Vol. 96. 1996.
    .. [3] Friedman, J., Hastie, T., & Tibshirani, R. "The Elements of Statistical 
           Learning." Springer Series in Statistics. (2001).
    """

    @abstractmethod
    def __init__(
        self,
        base_estimator=None,
        n_estimators=50,
        eta0=0.1,
        max_depth=3,
        strategy='hybrid',
        random_state=None,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=True,
        bootstrap_features=False,
        oob_score=False,
        warm_start=False,
        n_jobs=None,
        min_impurity_decrease=0.0,
        init=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_leaf_nodes=None,
        validation_fraction=0.1,
        n_iter_no_change=None,
        tol=1e-4,
        ccp_alpha=0.0,
        verbose=0
    ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.eta0 = eta0
        self.max_depth = max_depth
        self.strategy = strategy
        self.random_state = random_state
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.min_impurity_decrease = min_impurity_decrease
        self.init = init
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.ccp_alpha = ccp_alpha
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):
        """
        Fit the ensemble model to the data.
    
        This method trains the ensemble model on the input data `X`
        and target labels `y` using the specified ensemble strategy.
        The available strategies are `bagging`, `boosting`, and `hybrid`.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples. It can be a dense or sparse matrix.
    
        y : array-like of shape (n_samples,)
            The target values (class labels or continuous values).
    
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample. If `None`, all samples are
            given equal weight.
    
        Notes
        -----
        The `fit` method initializes the base estimator and then trains
        the ensemble model based on the specified strategy. The process
        can be summarized as follows:
    
        1. **Input Validation**: The input data `X` and target values `y`
           are checked for validity and conformance to the expected format
           and type using `check_X_y`.
    
        2. **Base Estimator Initialization**: If no base estimator is
           provided, the default base estimator is initialized with the
           specified `max_depth`.
    
        3. **Ensemble Strategy**: The chosen ensemble strategy is converted
           to lowercase for consistency. Based on the value of `strategy`,
           the corresponding `_fit_bagging`, `_fit_boosting`, or `_fit_hybrid`
           method is called to train the ensemble model.
    
        The mathematical formulation of the ensemble model can be described
        as follows:
    
        .. math::
            y_{\text{pred}} = \frac{1}{N} \sum_{i=1}^{N} y_{\text{est}_i}
    
        where:
        - :math:`N` is the number of base estimators in the ensemble.
        - :math:`y_{\text{pred}}` is the final predicted value aggregated from
          all base estimators.
        - :math:`y_{\text{est}_i}` represents the prediction made by the
          :math:`i`-th base estimator.
    
        Examples
        --------
        Here's an example of how to use the `fit` method with both
        `EnsembleClassifier` and `EnsembleRegressor`:
    
        >>> from gofast.estimators.ensemble import EnsembleClassifier, EnsembleRegressor
        >>> from sklearn.datasets import make_classification, make_regression
        >>> from sklearn.model_selection import train_test_split
    
        Classification Example:
    
        >>> X, y = make_classification(n_samples=100, n_features=20, random_state=42)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, 
        ...                                                     test_size=0.3, 
        ...                                                     random_state=42)
        >>> clf = EnsembleClassifier(n_estimators=50, strategy='hybrid', 
        ...                          random_state=42)
        >>> clf.fit(X_train, y_train)
        >>> print("Fitted ensemble classifier model:", clf)
    
        Regression Example:
    
        >>> X, y = make_regression(n_samples=100, n_features=20, random_state=42)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, 
        ...                                                     test_size=0.3, 
        ...                                                     random_state=42)
        >>> reg = EnsembleRegressor(n_estimators=50, strategy='hybrid', 
        ...                         random_state=42)
        >>> reg.fit(X_train, y_train)
        >>> print("Fitted ensemble regressor model:", reg)
    
        See Also
        --------
        sklearn.utils.validation.check_X_y : Utility function to check the input
            data and target values.
        sklearn.ensemble.BaggingClassifier : A bagging classifier.
        sklearn.ensemble.GradientBoostingClassifier : A gradient boosting classifier.
        sklearn.ensemble.BaggingRegressor : A bagging regressor.
        sklearn.ensemble.GradientBoostingRegressor : A gradient boosting regressor.
    
        References
        ----------
        .. [1] Breiman, L. "Bagging predictors." Machine learning 24.2 (1996): 123-140.
        .. [2] Freund, Y., & Schapire, R. E. "Experiments with a new boosting algorithm."
               ICML. Vol. 96. 1996.
        .. [3] Friedman, J., Hastie, T., & Tibshirani, R. "The Elements of Statistical
               Learning." Springer Series in Statistics. (2001).
    
        """
        X, y = check_X_y(
            X, y, accept_sparse=True,
            accept_large_sparse=True,
            estimator=get_estimator_name(self),
        )
        if self.base_estimator is None:
            self.base_estimator = self.default_base_estimator(max_depth=self.max_depth)

        self.strategy = str(self.strategy).lower()
        if self.strategy == 'bagging':
            self._fit_bagging(X, y, sample_weight, self.is_classifier)
        elif self.strategy == 'boosting':
            self._fit_boosting(X, y, sample_weight, self.is_classifier)
        elif self.strategy == 'hybrid':
            self._fit_hybrid(X, y, sample_weight, self.is_classifier)
        else:
            raise ValueError(
                "Invalid strategy, choose from 'hybrid', 'bagging', 'boosting'")

        return self
    
    def predict(self, X):
        """
        Predict using the fitted ensemble model.
    
        This method generates predictions for the input samples `X`
        using the fitted ensemble model. It checks if the model
        is fitted and then uses the base estimators within the
        ensemble to make predictions.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. It can be a dense or sparse matrix.
    
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted values (class labels or continuous values).
    
        Notes
        -----
        The `predict` method performs the following steps:
    
        1. **Check if Fitted**: The method checks if the model has been
           fitted by verifying the presence of the `model_` attribute.
           If the model is not fitted, it raises a `NotFittedError`.
    
        2. **Input Validation**: The input samples `X` are validated
           using `check_array` to ensure they conform to the expected
           format and type. This includes handling both dense and sparse
           matrices.
    
        3. **Prediction**: The method delegates the prediction task to
           the `predict` method of the fitted `model_` (which can be
           a `BaggingClassifier`, `BaggingRegressor`, `GradientBoostingClassifier`,
           or `GradientBoostingRegressor` depending on the strategy used).
    
        The mathematical formulation of the prediction process can be
        described as follows:
    
        .. math::
            \hat{y}_i = f(X_i)
    
        where:
        - :math:`\hat{y}_i` is the predicted value for the i-th sample.
        - :math:`f` is the ensemble model, which aggregates the predictions
          of the base estimators.
        - :math:`X_i` is the i-th input sample.
    
        Examples
        --------
        Here's an example of how to use the `predict` method with both
        `EnsembleClassifier` and `EnsembleRegressor`:
    
        >>> from gofast.estimators._base import EnsembleClassifier, EnsembleRegressor
        >>> from sklearn.datasets import make_classification, make_regression
        >>> from sklearn.model_selection import train_test_split
    
        Classification Example:
    
        >>> X, y = make_classification(n_samples=100, n_features=20, random_state=42)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, 
        ...                                                     test_size=0.3, 
        ...                                                     random_state=42)
        >>> clf = EnsembleClassifier(n_estimators=50, strategy='hybrid', 
        ...                          random_state=42)
        >>> clf.fit(X_train, y_train)
        >>> y_pred = clf.predict(X_test)
        >>> print("Predictions:", y_pred)
    
        Regression Example:
    
        >>> X, y = make_regression(n_samples=100, n_features=20, random_state=42)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, 
        ...                                                     test_size=0.3, 
        ...                                                     random_state=42)
        >>> reg = EnsembleRegressor(n_estimators=50, strategy='hybrid', 
        ...                         random_state=42)
        >>> reg.fit(X_train, y_train)
        >>> y_pred = reg.predict(X_test)
        >>> print("Predictions:", y_pred)
    
        See Also
        --------
        sklearn.utils.validation.check_array : Utility function to check the input
            array.
        sklearn.utils.validation.check_is_fitted : Utility function to check if the
            estimator is fitted.
        sklearn.ensemble.BaggingClassifier : A bagging classifier.
        sklearn.ensemble.GradientBoostingClassifier : A gradient boosting classifier.
        sklearn.ensemble.BaggingRegressor : A bagging regressor.
        sklearn.ensemble.GradientBoostingRegressor : A gradient boosting regressor.
    
        References
        ----------
        .. [1] Breiman, L. "Bagging predictors." Machine learning 24.2 (1996): 123-140.
        .. [2] Freund, Y., & Schapire, R. E. "Experiments with a new boosting algorithm."
               ICML. Vol. 96. 1996.
        .. [3] Friedman, J., Hastie, T., & Tibshirani, R. "The Elements of Statistical
               Learning." Springer Series in Statistics. (2001).
    
        """
        check_is_fitted(self, 'model_')
        X = check_array(
            X, accept_sparse=True,
            accept_large_sparse=True,
            estimator=get_estimator_name(self)
        )
        return self.model_.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities using the fitted ensemble model.
    
        This method generates probability estimates for the input samples
        `X` using the fitted ensemble model. This method is only applicable
        for classification tasks. If used with a regressor, it raises a
        `NotImplementedError`.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples. It can be a dense or sparse matrix.
    
        Returns
        -------
        p : array-like of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
    
        Notes
        -----
        The `predict_proba` method performs the following steps:
    
        1. **Check if Fitted**: The method checks if the model has been
           fitted by verifying the presence of the `model_` attribute.
           If the model is not fitted, it raises a `NotFittedError`.
    
        2. **Input Validation**: The input samples `X` are validated
           using `check_array` to ensure they conform to the expected
           format and type. This includes handling both dense and sparse
           matrices.
    
        3. **Probability Prediction**: The method delegates the probability
           prediction task to the `predict_proba` method of the fitted
           `model_` (which should be a `BaggingClassifier` or 
           `GradientBoostingClassifier`).
    
        If the ensemble model is a regressor, this method raises a
        `NotImplementedError` with the message "Probability estimates are
        not available for regressors."
    
        The mathematical formulation of the probability prediction process
        can be described as follows:
    
        .. math::
            P_{\text{class}} = \frac{1}{N} \sum_{i=1}^{N} P_{\text{est}_i}
    
        where:
        - :math:`N` is the number of base estimators in the ensemble.
        - :math:`P_{\text{class}}` is the final predicted probability
          aggregated from all base estimators.
        - :math:`P_{\text{est}_i}` represents the predicted probability
          made by the :math:`i`-th base estimator.
    
        Examples
        --------
        Here's an example of how to use the `predict_proba` method with
        `EnsembleClassifier`:
    
        >>> from gofast.estimators._base import EnsembleClassifier
        >>> from sklearn.datasets import make_classification
        >>> from sklearn.model_selection import train_test_split
    
        Classification Example:
    
        >>> X, y = make_classification(n_samples=100, n_features=20, random_state=42)
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, 
        ...                                                     test_size=0.3, 
        ...                                                     random_state=42)
        >>> clf = EnsembleClassifier(n_estimators=50, strategy='hybrid', 
        ...                          random_state=42)
        >>> clf.fit(X_train, y_train)
        >>> probas = clf.predict_proba(X_test)
        >>> print("Predicted probabilities:", probas)

        """
        raise NotImplementedError(
            "Probability estimates are not available for regressors.")
        
    def _fit_bagging(self, X, y, sample_weight, is_classifier):
        """
        Fit the ensemble model using the Bagging strategy.
    
        This method trains the ensemble model on the input data `X`
        and target labels `y` using the Bagging strategy. Depending
        on the value of `is_classifier`, it uses either a `BaggingClassifier`
        or a `BaggingRegressor`.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples. It can be a dense or sparse matrix.
    
        y : array-like of shape (n_samples,)
            The target values (class labels or continuous values).
    
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample. If `None`, all samples are
            given equal weight.
    
        is_classifier : bool
            A flag indicating whether the task is classification or regression.
            - If `True`, a `BaggingClassifier` is used.
            - If `False`, a `BaggingRegressor` is used.
    
        Notes
        -----
        The `Bagging` strategy, short for Bootstrap Aggregating, involves
        training multiple base estimators on random subsets of the original
        dataset and aggregating their predictions. This approach reduces
        variance and improves the robustness of the model.
    
        For a given dataset :math:`(X, y)`, bagging works as follows:
        
        1. Randomly draw `n_estimators` subsets from the dataset with
           replacement (bootstrap sampling).
        2. Train a base estimator on each subset.
        3. Aggregate the predictions from all base estimators.
    
        The final prediction for classification is determined by majority
        voting (for discrete labels) or averaging probabilities (for
        probabilistic outputs). For regression, the final prediction is
        the average of all base estimators' predictions.
    
        The base estimator used can be specified by the `base_estimator`
        parameter. If not provided, a default estimator is used.
    
        Examples
        --------
        Here's an example of how to use the `_fit_bagging` method for
        both classification and regression tasks:
    
        >>> from gofast.estimators.ensemble import EnsembleClassifier
        >>> from gofast.estimators.ensemble import EnsembleRegressor
        >>> from sklearn.datasets import make_classification, make_regression
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.utils import check_X_y
    
        Classification Example:
    
        >>> X, y = make_classification(n_samples=100, n_features=20, random_state=42)
        >>> X, y = check_X_y(X, y, accept_sparse=True, accept_large_sparse=True)
        >>> sample_weight = None
        >>> is_classifier = True
        >>> base_estimator = DecisionTreeClassifier(max_depth=3)
        >>> obj = EnsembleClassifier(base_estimator=base_estimator)
        >>> obj._fit_bagging(X, y, sample_weight, is_classifier)
        >>> print("Fitted bagging classifier model:", obj.model_)
    
        Regression Example:
    
        >>> X, y = make_regression(n_samples=100, n_features=20, random_state=42)
        >>> X, y = check_X_y(X, y, accept_sparse=True, accept_large_sparse=True)
        >>> sample_weight = None
        >>> is_classifier = False
        >>> base_estimator = DecisionTreeRegressor(max_depth=3)
        >>> obj = EnsembleRegressor(base_estimator=base_estimator)
        >>> obj._fit_bagging(X, y, sample_weight, is_classifier)
        >>> print("Fitted bagging regressor model:", obj.model_)
    
        See Also
        --------
        sklearn.ensemble.BaggingClassifier : A bagging classifier.
        sklearn.ensemble.BaggingRegressor : A bagging regressor.
    
        References
        ----------
        .. [1] Breiman, L. "Bagging predictors." Machine learning 24.2 (1996): 123-140.
    
        """
        if is_classifier:
            self.model_ = BaggingClassifier(
                estimator=self.base_estimator,
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                max_samples=self.max_samples,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                bootstrap_features=self.bootstrap_features,
                oob_score=self.oob_score,
                warm_start=self.warm_start,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
            )
        else:
            self.model_ = BaggingRegressor(
                estimator=self.base_estimator,
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                max_samples=self.max_samples,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                bootstrap_features=self.bootstrap_features,
                oob_score=self.oob_score,
                warm_start=self.warm_start,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
            )
        self.model_.fit(X, y, sample_weight)

    def _fit_boosting(self, X, y, sample_weight, is_classifier):
        """
        Fit the ensemble model using the Boosting strategy.
    
        This method trains the ensemble model on the input data `X`
        and target labels `y` using the Boosting strategy. Depending
        on the value of `is_classifier`, it uses either a 
        `GradientBoostingClassifier` or a `GradientBoostingRegressor`.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples. It can be a dense or sparse matrix.
    
        y : array-like of shape (n_samples,)
            The target values (class labels or continuous values).
    
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample. If `None`, all samples are
            given equal weight.
    
        is_classifier : bool
            A flag indicating whether the task is classification or regression.
            - If `True`, a `GradientBoostingClassifier` is used.
            - If `False`, a `GradientBoostingRegressor` is used.
    
        Notes
        -----
        The `Boosting` strategy involves training multiple base estimators
        sequentially. Each estimator attempts to correct the errors of the
        previous one, making the model robust to overfitting and improving
        its predictive performance.
    
        For a given dataset :math:`(X, y)`, boosting works as follows:
    
        1. Initialize the model with a base estimator and a learning rate.
        2. Train the base estimator on the data.
        3. Adjust the weights of the training samples based on the error
           of the previous estimator.
        4. Train the next base estimator on the adjusted data.
        5. Repeat steps 3-4 for `n_estimators` iterations.
        6. Aggregate the predictions of all base estimators.
    
        The final prediction for classification is determined by majority
        voting (for discrete labels) or averaging probabilities (for
        probabilistic outputs). For regression, the final prediction is
        the weighted sum of all base estimators' predictions.
    
        The base estimator used can be specified by the `base_estimator`
        parameter. If not provided, a default estimator is used.
    
        Examples
        --------
        Here's an example of how to use the `_fit_boosting` method for
        both classification and regression tasks:
    
        >>> from gofast.estimators.ensemble import EnsembleClassifier
        >>> from gofast.estimators.ensemble import EnsembleRegressor 
        >>> from sklearn.datasets import make_classification, make_regression
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.utils import check_X_y
    
        Classification Example:
    
        >>> X, y = make_classification(n_samples=100, n_features=20, random_state=42)
        >>> X, y = check_X_y(X, y, accept_sparse=True, accept_large_sparse=True)
        >>> sample_weight = None
        >>> is_classifier = True
        >>> base_estimator = GradientBoostingClassifier(max_depth=3)
        >>> obj = EnsembleClassifier(base_estimator=base_estimator)
        >>> obj._fit_boosting(X, y, sample_weight, is_classifier)
        >>> print("Fitted boosting classifier model:", obj.model_)
    
        Regression Example:
    
        >>> X, y = make_regression(n_samples=100, n_features=20, random_state=42)
        >>> X, y = check_X_y(X, y, accept_sparse=True, accept_large_sparse=True)
        >>> sample_weight = None
        >>> is_classifier = False
        >>> base_estimator = GradientBoostingRegressor(max_depth=3)
        >>> obj = EnsembleRegressor(base_estimator=base_estimator)
        >>> obj._fit_boosting(X, y, sample_weight, is_classifier)
        >>> print("Fitted boosting regressor model:", obj.model_)
    
        See Also
        --------
        sklearn.ensemble.GradientBoostingClassifier : A gradient boosting classifier.
        sklearn.ensemble.GradientBoostingRegressor : A gradient boosting regressor.
    
        References
        ----------
        .. [1] Freund, Y., & Schapire, R. E. "Experiments with a new boosting algorithm."
               ICML. Vol. 96. 1996.
        .. [2] Friedman, J., Hastie, T., & Tibshirani, R. "The Elements of Statistical 
               Learning." Springer Series in Statistics. (2001).
    
        """
        if is_classifier:
            self.model_ = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.eta0,
                max_depth=self.max_depth,
                random_state=self.random_state,
                min_impurity_decrease=self.min_impurity_decrease,
                init=self.init,
                max_features=self.max_features,
                verbose=self.verbose,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_leaf_nodes=self.max_leaf_nodes,
                validation_fraction=self.validation_fraction,
                n_iter_no_change=self.n_iter_no_change,
                tol=self.tol,
                ccp_alpha=self.ccp_alpha
            )
        else:
            self.model_ = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.eta0,
                max_depth=self.max_depth,
                random_state=self.random_state,
                min_impurity_decrease=self.min_impurity_decrease,
                init=self.init,
                max_features=self.max_features,
                verbose=self.verbose,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_leaf_nodes=self.max_leaf_nodes,
                validation_fraction=self.validation_fraction,
                n_iter_no_change=self.n_iter_no_change,
                tol=self.tol,
                ccp_alpha=self.ccp_alpha
            )
        self.model_.fit(X, y, sample_weight)
    
    def _fit_hybrid(self, X, y, sample_weight, is_classifier):
        """
        Fit the ensemble model using the Hybrid strategy.
    
        This method trains the ensemble model on the input data `X`
        and target labels `y` using the Hybrid strategy, which combines
        bagging and boosting approaches. Depending on the value of
        `is_classifier`, it uses either a `GradientBoostingClassifier`
        or a `GradientBoostingRegressor` as the base estimator within
        a `BaggingClassifier` or `BaggingRegressor`.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples. It can be a dense or sparse matrix.
    
        y : array-like of shape (n_samples,)
            The target values (class labels or continuous values).
    
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample. If `None`, all samples
            are given equal weight.
    
        is_classifier : bool
            A flag indicating whether the task is classification or regression.
            - If `True`, a `GradientBoostingClassifier` is used within a 
              `BaggingClassifier`.
            - If `False`, a `GradientBoostingRegressor` is used within a 
              `BaggingRegressor`.
    
        Notes
        -----
        The `Hybrid` strategy combines the strengths of both bagging and 
        boosting. Bagging reduces variance by training multiple base estimators 
        on random subsets of the dataset, while boosting reduces bias by 
        sequentially training base estimators, each correcting the errors of 
        the previous ones.
    
        For a given dataset :math:`(X, y)`, the hybrid strategy works as follows:
    
        1. Initialize the model with a bagging estimator and a boosting base 
           estimator.
        2. Use the boosting estimator to train on subsets of the data, with 
           each boosting iteration attempting to correct the errors of the 
           previous iteration.
        3. Use the bagging estimator to aggregate the boosting models.
        4. Repeat the process for `n_estimators` iterations.
    
        The final prediction for classification is determined by majority voting 
        (for discrete labels) or averaging probabilities (for probabilistic outputs). 
        For regression, the final prediction is the average of all base estimators' 
        predictions.
    
        Examples
        --------
        Here's an example of how to use the `_fit_hybrid` method for both 
        classification and regression tasks:
    
        >>> from gofast.estimators.ensemble import EnsembleClassifier
        >>> from gofast.estimators.ensemble import EnsembleRegressor 
        >>> from sklearn.datasets import make_classification, make_regression
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.utils import check_X_y
    
        Classification Example:
    
        >>> X, y = make_classification(n_samples=100, n_features=20, random_state=42)
        >>> X, y = check_X_y(X, y, accept_sparse=True, accept_large_sparse=True)
        >>> sample_weight = None
        >>> is_classifier = True
        >>> obj = EnsembleClassifier()
        >>> obj._fit_hybrid(X, y, sample_weight, is_classifier)
        >>> print("Fitted hybrid classifier model:", obj.model_)
    
        Regression Example:
    
        >>> X, y = make_regression(n_samples=100, n_features=20, random_state=42)
        >>> X, y = check_X_y(X, y, accept_sparse=True, accept_large_sparse=True)
        >>> sample_weight = None
        >>> is_classifier = False
        >>> obj = EnsembleRegressor()
        >>> obj._fit_hybrid(X, y, sample_weight, is_classifier)
        >>> print("Fitted hybrid regressor model:", obj.model_)
    
        See Also
        --------
        sklearn.ensemble.BaggingClassifier : A bagging classifier.
        sklearn.ensemble.GradientBoostingClassifier : A gradient boosting classifier.
        sklearn.ensemble.BaggingRegressor : A bagging regressor.
        sklearn.ensemble.GradientBoostingRegressor : A gradient boosting regressor.
    
        References
        ----------
        .. [1] Breiman, L. "Bagging predictors." Machine learning 24.2 (1996): 123-140.
        .. [2] Freund, Y., & Schapire, R. E. "Experiments with a new boosting algorithm."
               ICML. Vol. 96. 1996.
        .. [3] Friedman, J., Hastie, T., & Tibshirani, R. "The Elements of Statistical 
               Learning." Springer Series in Statistics. (2001).
    
        """
        if is_classifier:
            base_estimator = GradientBoostingClassifier(
                n_estimators=self.n_estimators // 2,
                learning_rate=self.eta0,
                max_depth=self.max_depth,
                random_state=self.random_state,
                min_impurity_decrease=self.min_impurity_decrease,
                init=self.init,
                max_features=self.max_features,
                verbose=self.verbose,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_leaf_nodes=self.max_leaf_nodes,
                validation_fraction=self.validation_fraction,
                n_iter_no_change=self.n_iter_no_change,
                tol=self.tol,
                ccp_alpha=self.ccp_alpha
            )
        else:
            base_estimator = GradientBoostingRegressor(
                n_estimators=self.n_estimators // 2,
                learning_rate=self.eta0,
                max_depth=self.max_depth,
                random_state=self.random_state,
                min_impurity_decrease=self.min_impurity_decrease,
                init=self.init,
                max_features=self.max_features,
                verbose=self.verbose,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_leaf_nodes=self.max_leaf_nodes,
                validation_fraction=self.validation_fraction,
                n_iter_no_change=self.n_iter_no_change,
                tol=self.tol,
                ccp_alpha=self.ccp_alpha
            )
    
        if is_classifier:
            self.model_ = BaggingClassifier(
                estimator=base_estimator,
                n_estimators=2,  # number of boosting models in the bagging
                random_state=self.random_state,
                max_samples=self.max_samples,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                bootstrap_features=self.bootstrap_features,
                oob_score=self.oob_score,
                warm_start=self.warm_start,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
            )
        else:
            self.model_ = BaggingRegressor(
                estimator=base_estimator,
                n_estimators=2,  # number of boosting models in the bagging
                random_state=self.random_state,
                max_samples=self.max_samples,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                bootstrap_features=self.bootstrap_features,
                oob_score=self.oob_score,
                warm_start=self.warm_start,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
            )
        self.model_.fit(X, y, sample_weight)

class NeuroFuzzyBase(BaseEstimator, metaclass=ABCMeta):
    """
    NeuroFuzzyBase is a base class for neuro-fuzzy network-based models that 
    integrate fuzzy logic and neural networks to perform classification or 
    regression tasks.

    Parameters
    ----------
    n_clusters : int, default=3
        The number of fuzzy clusters for each feature in the input data. 
        This parameter controls the granularity of the fuzzy logic 
        partitioning.

    eta0 : float, default=0.001
        The initial learning rate for training the `MLPClassifier` or 
        `MLPRegressor`. A smaller learning rate makes the training process 
        more stable but slower.

    max_iter : int, default=200
        Maximum number of iterations for training each `MLPClassifier` or 
        `MLPRegressor`. This parameter determines how long the training 
        will continue if it does not converge before reaching this limit.

    hidden_layer_sizes : tuple, default=(100,)
        The number of neurons in each hidden layer of the `MLPClassifier` or 
        `MLPRegressor`. For example, (100,) means there is one hidden layer 
        with 100 neurons.

    activation : {'identity', 'logistic', 'tanh', 'relu'}, default='relu'
        Activation function for the hidden layer in the `MLPClassifier` or 
        `MLPRegressor`.
        - 'identity': no-op activation, useful for linear bottleneck, 
          returns `f(x) = x`
        - 'logistic': the logistic sigmoid function, returns 
          `f(x) = 1 / (1 + exp(-x))`
        - 'tanh': the hyperbolic tan function, returns 
          `f(x) = tanh(x)`
        - 'relu': the rectified linear unit function, returns 
          `f(x) = max(0, x)`

    solver : {'lbfgs', 'sgd', 'adam'}, default='adam'
        The solver for weight optimization in the `MLPClassifier` or 
        `MLPRegressor`.
        - 'lbfgs': optimizer in the family of quasi-Newton methods
        - 'sgd': stochastic gradient descent
        - 'adam': stochastic gradient-based optimizer proposed by Kingma, 
          Diederik, and Jimmy Ba

    alpha : float, default=0.0001
        L2 penalty (regularization term) parameter for the `MLPClassifier` or 
        `MLPRegressor`. Regularization helps to prevent overfitting by 
        penalizing large weights.

    batch_size : int, default='auto'
        Size of minibatches for stochastic optimizers. If 'auto', batch size 
        is set to `min(200, n_samples)`.

    learning_rate : {'constant', 'invscaling', 'adaptive'}, default='constant'
        Learning rate schedule for weight updates.
        - 'constant': learning rate remains constant
        - 'invscaling': gradually decreases the learning rate at each time step
        - 'adaptive': keeps the learning rate constant to `eta0` as long as 
          training loss keeps decreasing

    power_t : float, default=0.5
        The exponent for inverse scaling learning rate. It is used only 
        when `learning_rate='invscaling'`.

    tol : float, default=1e-4
        Tolerance for the optimization. Training will stop when the 
        validation score is not improving by at least `tol` for two consecutive 
        iterations.

    momentum : float, default=0.9
        Momentum for gradient descent update. Should be between 0 and 1. 
        It helps to accelerate gradients vectors in the right directions, 
        thus leading to faster converging.

    nesterovs_momentum : bool, default=True
        Whether to use Nesterov’s momentum. Nesterov’s momentum is an 
        improvement over standard momentum by considering the gradient ahead 
        of the current point in time.

    early_stopping : bool, default=False
        Whether to use early stopping to terminate training when validation 
        score is not improving. If set to `True`, it will automatically set 
        aside a fraction of training data as validation and terminate 
        training when validation score is not improving by at least `tol` for 
        `n_iter_no_change` consecutive epochs.

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for 
        early stopping. It is used only if `early_stopping` is set to `True`.

    beta_1 : float, default=0.9
        Exponential decay rate for estimates of first moment vector in adam, 
        should be in the range `[0, 1)`.

    beta_2 : float, default=0.999
        Exponential decay rate for estimates of second moment vector in adam, 
        should be in the range `[0, 1)`.

    epsilon : float, default=1e-8
        Value for numerical stability in adam. It is used to prevent any 
        division by zero in the implementation.

    n_iter_no_change : int, default=10
        Maximum number of epochs to not meet `tol` improvement. The training 
        will stop if the validation score is not improving by at least `tol` 
        for `n_iter_no_change` consecutive epochs.

    max_fun : int, default=15000
        Only used when solver='lbfgs'. Maximum number of function calls.

    random_state : int, RandomState instance, default=None
        Determines random number generation for weights and bias 
        initialization, train-test split if early stopping is used, and batch 
        sampling when solver='sgd' or 'adam'. Pass an int for reproducible 
        results across multiple function calls.

    verbose : bool, default=False
        Whether to print progress messages to stdout. If `True`, progress 
        messages are printed during training.

    is_classifier : bool, default=False
        Whether the model is a classifier (`True`) or a regressor (`False`).

    Notes
    -----
    The NeuroFuzzyBase class combines the strengths of fuzzy logic and 
    neural networks to capture nonlinear relationships in data. Fuzzy logic 
    helps in dealing with uncertainty and imprecision, while neural networks 
    provide learning capabilities.

    The fuzzy clustering is performed using c-means clustering. The resulting 
    fuzzy sets are used to transform the input features, which are then fed 
    into an `MLPClassifier` or `MLPRegressor` for learning.

    The objective function for fuzzy c-means clustering is:
    
    .. math::
        \min \sum_{i=1}^{n} \sum_{j=1}^{c} u_{ij}^m \|x_i - v_j\|^2

    where :math:`u_{ij}` is the degree of membership of :math:`x_i` in the 
    cluster :math:`j`, :math:`v_j` is the cluster center, and :math:`m` is a 
    weighting exponent.

    For classification tasks, the `MLPClassifier` uses cross-entropy loss to 
    learn the weights of the network by minimizing the difference between the 
    predicted and actual class probabilities:
    
    .. math::
        \text{Cross-Entropy} = - \sum_{i=1}^{n} \sum_{j=1}^{c} y_{ij} \log(\hat{y}_{ij})

    where :math:`n` is the number of samples, :math:`c` is the number of 
    classes, :math:`y_{ij}` is the actual class label (one-hot encoded), and 
    :math:`\hat{y}_{ij}` is the predicted class probability.

    For regression tasks, the `MLPRegressor` uses the mean squared error (MSE) 
    loss function:
    
    .. math::
        MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2

    where :math:`n` is the number of samples, :math:`y_i` is the actual target 
    value, and :math:`\hat{y}_i` is the predicted target value.

    Examples
    --------
    >>> from gofast.estimators._base import NeuroFuzzyBase
    >>> X = [[0., 0.], [1., 1.]]
    >>> y = [0, 1]
    >>> model = NeuroFuzzyBase(n_clusters=2, max_iter=1000, verbose=True, is_classifier=True)
    >>> model._fit(X, y)
    NeuroFuzzyBase(...)
    >>> model._predict([[2., 2.]])
    array([1])

    See Also
    --------
    sklearn.neural_network.MLPClassifier : Multi-layer Perceptron classifier.
    sklearn.neural_network.MLPRegressor : Multi-layer Perceptron regressor.
    
    References
    ----------
    .. [1] J.-S. R. Jang, "ANFIS: Adaptive-Network-based Fuzzy Inference 
           System," IEEE Transactions on Systems, Man, and Cybernetics, vol. 
           23, no. 3, pp. 665-685, 1993.
    """
    @abstractmethod
    def __init__(
        self, 
        *, 
        n_clusters=3, 
        eta0=0.001, 
        max_iter=200, 
        hidden_layer_sizes=(100,), 
        activation='relu', 
        solver='adam', 
        alpha=0.0001, 
        batch_size='auto', 
        learning_rate='constant', 
        power_t=0.5, 
        tol=1e-4, 
        momentum=0.9, 
        nesterovs_momentum=True, 
        early_stopping=False, 
        validation_fraction=0.1, 
        beta_1=0.9, 
        beta_2=0.999, 
        epsilon=1e-8, 
        n_iter_no_change=10, 
        max_fun=15000, 
        random_state=None, 
        verbose=False,
        is_classifier=False
    ):
        self.n_clusters = n_clusters
        self.eta0 = eta0
        self.max_iter = max_iter
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.power_t = power_t
        self.tol = tol
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change
        self.max_fun = max_fun
        self.random_state = random_state
        self.verbose = verbose
        self.is_classifier = is_classifier
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder()
        
    def _fuzzify(self, X):
        """
        Transform the input features into fuzzy sets using fuzzy c-means clustering.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to be fuzzified.
    
        Returns
        -------
        antecedents : list of `Antecedent`
            A list of fuzzy antecedents for each feature in the input data. Each 
            antecedent represents the fuzzy clusters for a corresponding feature.
    
        Notes
        -----
        The fuzzification process involves the following steps:
    
        1. **Fuzzy c-means Clustering**:
           For each feature in `X`, fuzzy c-means clustering is applied to 
           partition the feature values into `n_clusters` fuzzy clusters. The 
           objective function for fuzzy c-means clustering is:
           
           .. math::
               \min \sum_{i=1}^{n} \sum_{j=1}^{c} u_{ij}^m \|x_i - v_j\|^2
    
           where :math:`u_{ij}` is the degree of membership of :math:`x_i` in 
           cluster :math:`j`, :math:`v_j` is the cluster center, and :math:`m` 
           is a weighting exponent.
    
        2. **Antecedent Creation**:
           For each feature, an `Antecedent` object is created representing the 
           universe of discourse for that feature. The fuzzy clusters obtained 
           from c-means clustering are then used to define the membership 
           functions for each cluster within the antecedent.
    
        This process transforms the input features into fuzzy sets, which can 
        then be used in the fuzzy inference system.
    
        If `verbose=True`, a progress bar will be displayed during the fuzzification 
        process for each feature.
    
        Examples
        --------
        >>> from gofast.estimators._base import NeuroFuzzyBase
        >>> X = [[0., 0.], [1., 1.]]
        >>> model = NeuroFuzzyBase(n_clusters=2, max_iter=1000, verbose=True, 
                                   is_classifier=True)
        >>> antecedents = model._fuzzify(X)
        >>> len(antecedents)
        2
    
        See Also
        --------
        skfuzzy.cluster.cmeans : Fuzzy c-means clustering algorithm.
        skfuzzy.control.Antecedent : Class representing a fuzzy antecedent.
    
        References
        ----------
        .. [1] J.-S. R. Jang, "ANFIS: Adaptive-Network-based Fuzzy Inference
               System," IEEE Transactions on Systems, Man, and Cybernetics, vol.
               23, no. 3, pp. 665-685, 1993.
        """
        n_features = X.shape[1]
        antecedents = []
        if self.verbose:
            progress_bar = tqdm(
                total=n_features, ascii=True, ncols=100,
                desc="{:<30}".format(f'Fuzzify {self.estimator_name_}'), 
            )
        for i in range(n_features):
            antecedent = Antecedent(np.linspace(0, 1, 100), f'feature_{i}')
            clusters = fuzz.cluster.cmeans(
                X[:, [i]].T, self.n_clusters, 2, error=0.005, maxiter=1000)[0]
            for j, cluster in enumerate(clusters):
                antecedent[f'cluster_{j}'] = fuzz.trimf(antecedent.universe, cluster)
            antecedents.append(antecedent)
            
            if self.verbose: 
                progress_bar.update(1)
        if self.verbose: 
            progress_bar.close()
            
        return antecedents

    
    def _fit(self, X, y, sample_weight=None):
        """
        Fit the NeuroFuzzyBase model according to the given training data.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data where `n_samples` is the number of samples and 
            `n_features` is the number of features.
    
        y : array-like of shape (n_samples,)
            Target values. For classification tasks, these should be class labels.
            For regression tasks, these should be continuous values.
    
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample. If not provided, all samples are 
            given equal weight.
    
        Returns
        -------
        self : object
            Returns self.
    
        Notes
        -----
        The fit process involves the following steps:
    
        1. **Data Validation and Preprocessing**:
           Ensures that `X` and `y` have correct shapes and types.
           Standard scaling is applied to `X`. If `is_classifier` is `True`, 
           `y` is one-hot encoded.
    
        2. **Fuzzification**:
           The input features are transformed into fuzzy sets using fuzzy
           c-means clustering:
           
           .. math::
               \min \sum_{i=1}^{n} \sum_{j=1}^{c} u_{ij}^m \|x_i - v_j\|^2
    
           where :math:`u_{ij}` is the degree of membership of :math:`x_i` in 
           cluster :math:`j`, :math:`v_j` is the cluster center, and :math:`m` 
           is a weighting exponent.
    
        3. **MLP Training**:
           The scaled and fuzzified features are used to train an `MLPClassifier`
           or `MLPRegressor` to map inputs to the encoded target values.
    
        If `verbose=True`, a progress bar will be displayed during the MLP
        training process.
    
        Examples
        --------
        >>> from gofast.estimators._base import NeuroFuzzyBase
        >>> X = [[0., 0.], [1., 1.]]
        >>> y = [0, 1]
        >>> model = NeuroFuzzyBase(n_clusters=2, max_iter=1000, verbose=True, 
                                   is_classifier=True)
        >>> model._fit(X, y)
        NeuroFuzzyBase(...)
    
        See Also
        --------
        sklearn.neural_network.MLPClassifier : Multi-layer Perceptron classifier.
        sklearn.neural_network.MLPRegressor : Multi-layer Perceptron regressor.
        
        References
        ----------
        .. [1] J.-S. R. Jang, "ANFIS: Adaptive-Network-based Fuzzy Inference
               System," IEEE Transactions on Systems, Man, and Cybernetics, vol.
               23, no. 3, pp. 665-685, 1993.
        """

        X, y = check_X_y(X, y, estimator=self)
        
        self.estimator_name_= ( "NeuroFuzzyClassifier" if self.classifier 
                               else "NeuroFuzzyRegressor")
        
        if self.verbose: 
            print(f"Fitting {self.estimator_name_}....")
        
        if self.is_classifier:
            self.mlp_ = MLPClassifier(
                learning_rate_init=self.eta0, 
                max_iter=self.max_iter,
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation, 
                solver=self.solver, 
                alpha=self.alpha, 
                batch_size=self.batch_size, 
                learning_rate=self.learning_rate, 
                power_t=self.power_t, 
                tol=self.tol, 
                momentum=self.momentum, 
                nesterovs_momentum=self.nesterovs_momentum, 
                early_stopping=self.early_stopping, 
                validation_fraction=self.validation_fraction, 
                beta_1=self.beta_1, 
                beta_2=self.beta_2, 
                epsilon=self.epsilon, 
                n_iter_no_change=self.n_iter_no_change, 
                max_fun=self.max_fun, 
                random_state=self.random_state, 
                verbose=self.verbose
            )
        else:
            self.mlp_ = MLPRegressor(
                learning_rate_init=self.eta0, 
                max_iter=self.max_iter,
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation, 
                solver=self.solver, 
                alpha=self.alpha, 
                batch_size=self.batch_size, 
                learning_rate=self.learning_rate, 
                power_t=self.power_t, 
                tol=self.tol, 
                momentum=self.momentum, 
                nesterovs_momentum=self.nesterovs_momentum, 
                early_stopping=self.early_stopping, 
                validation_fraction=self.validation_fraction, 
                beta_1=self.beta_1, 
                beta_2=self.beta_2, 
                epsilon=self.epsilon, 
                n_iter_no_change=self.n_iter_no_change, 
                max_fun=self.max_fun, 
                random_state=self.random_state, 
                verbose=self.verbose
            )
        
        X_scaled = self.scaler.fit_transform(X)
        y_encoded = self.encoder.fit_transform(y.reshape(-1, 1)).toarray(
            ) if self.is_classifier else y
        
        
        self.antecedents_ = self._fuzzify(X_scaled)
        
        if self.verbose:
            for _ in tqdm(range(1), ncols=100, desc="{:<30}".format(
                    f"Fitting {self.estimator_name_}"), ascii=True):
                self.mlp_.fit(X_scaled, y_encoded)
        else:
            self.mlp_.fit(X_scaled, y_encoded)
        
        self.fitted_ = True 
        
        if self.verbose: 
            print(f"Fitting {self.estimator_name_} completed.")
        
        return self

    def _defuzzify(self, y_fuzzy):
        """ The _defuzzify method was designed to convert fuzzy outputs back
        into crisp values, but since we are using MLPRegressor, which provides
        crisp predictions directly, the _defuzzify method is not needed. 
        Instead, the predictions from the MLPRegressor are handled using 
        standard techniques to get the final output classes.
        """
        y = []
        for yf in y_fuzzy:
            max_membership = max(yf, key=lambda k: yf[k])
            y.append(max_membership)
        return np.array(y)
    
    def _predict(self, X):
        """
        Predict using the NeuroFuzzyBase model.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
    
        Returns
        -------
        y_pred : array of shape (n_samples,)
            The predicted values. For classification tasks, these are the class
            labels. For regression tasks, these are the continuous target values.
    
        Notes
        -----
        The predict process involves the following steps:
    
        1. **Data Scaling**:
           The input features are scaled using the same scaler fitted during
           the training process.
    
        2. **MLP Prediction**:
           The scaled features are fed into the trained `MLPClassifier` or 
           `MLPRegressor` to obtain predicted values.
    
        For classification tasks, the predicted class probabilities are transformed 
        back to class labels using the fitted `OneHotEncoder`.
    
        Examples
        --------
        >>> from gofast.estimators._base import NeuroFuzzyBase
        >>> X = [[2., 2.]]
        >>> model = NeuroFuzzyBase(n_clusters=2, max_iter=1000, verbose=True, 
                                   is_classifier=True)
        >>> model._fit([[0., 0.], [1., 1.]], [0, 1])
        NeuroFuzzyBase(...)
        >>> model._predict(X)
        array([1])
    
        See Also
        --------
        sklearn.neural_network.MLPClassifier : Multi-layer Perceptron classifier.
        sklearn.neural_network.MLPRegressor : Multi-layer Perceptron regressor.
        
        References
        ----------
        .. [1] J.-S. R. Jang, "ANFIS: Adaptive-Network-based Fuzzy Inference
               System," IEEE Transactions on Systems, Man, and Cybernetics, vol.
               23, no. 3, pp. 665-685, 1993.
        """
        check_is_fitted(self, "fitted_")
        X = check_array(X, accept_sparse=True, estimator=self)
        X_scaled = self.scaler.transform(X)
        y_pred = self.mlp_.predict(X_scaled)
        
        if self.is_classifier:
            y_pred_labels = np.argmax(y_pred, axis=1)
            y_pred_classes = self.encoder.inverse_transform(
                y_pred_labels.reshape(-1, 1)).flatten()
            return y_pred_classes
        else:
            return y_pred
        
    def _predict_proba(self, X):
        """
        Probability estimates using the NeuroFuzzyBase model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples for which to predict the class probabilities.

        Returns
        -------
        y_proba : array of shape (n_samples, n_classes)
            The predicted class probabilities. The columns correspond to the 
            classes in sorted order, as they appear in the attribute `classes_`.

        Notes
        -----
        The predict_proba process involves the following steps:

        1. **Data Scaling**:
           The input features are scaled using the same scaler fitted during
           the training process.

        2. **MLP Probability Prediction**:
           The scaled features are fed into the trained `MLPClassifier` to
           obtain predicted class probabilities.

        The predicted probabilities represent the likelihood of each class
        for the input samples. The `MLPClassifier` computes these probabilities
        using the softmax function in the output layer:
        
        .. math::
            P(y = k | x) = \frac{\exp(\hat{y}_k)}{\sum_{j=1}^{C} \exp(\hat{y}_j)}

        where :math:`\hat{y}_k` is the raw model output for class `k`, and 
        :math:`C` is the number of classes.

        Examples
        --------
        >>> from gofast.estimators._base import NeuroFuzzyBase
        >>> X = [[2., 2.]]
        >>> model = NeuroFuzzyBase(n_clusters=2, max_iter=1000, verbose=True, 
                                   is_classifier=True)
        >>> model._fit([[0., 0.], [1., 1.]], [0, 1])
        NeuroFuzzyBase(...)
        >>> model._predict_proba(X)
        array([[0.1, 0.9]])

        See Also
        --------
        sklearn.neural_network.MLPClassifier : Multi-layer Perceptron classifier.
        
        References
        ----------
        .. [1] J.-S. R. Jang, "ANFIS: Adaptive-Network-based Fuzzy Inference
               System," IEEE Transactions on Systems, Man, and Cybernetics, vol.
               23, no. 3, pp. 665-685, 1993.
        """
        if not self.is_classifier:
            raise NotImplementedError(
                "Probability estimates are not available for regressors.")
        
        check_is_fitted(self, "fitted_")
        X = check_array(X, accept_sparse=True, estimator=self)
        
        X_scaled = self.scaler.transform(X)
        y_proba = self.mlp_.predict_proba(X_scaled)
        return y_proba

class FuzzyNeuralNetBase(BaseEstimator, metaclass=ABCMeta):
    """
    FuzzyNeuralNetBase is a base class for ensemble neuro-fuzzy network-based 
    models that integrates fuzzy logic and neural networks to perform 
    classification or regression tasks.

    Parameters
    ----------
    n_estimators : int, default=10
        The number of neural network estimators in the ensemble. This 
        parameter determines how many individual MLP models will be trained 
        and their predictions aggregated.
    
    n_clusters : int, default=3
        The number of fuzzy clusters for each feature in the input data. 
        This parameter controls the granularity of the fuzzy logic 
        partitioning.

    eta0 : float, default=0.001
        The initial learning rate for training the `MLPClassifier` or 
        `MLPRegressor`. A smaller learning rate makes the training process 
        more stable but slower.

    max_iter : int, default=200
        Maximum number of iterations for training each `MLPClassifier` or 
        `MLPRegressor`. This parameter determines how long the training 
        will continue if it does not converge before reaching this limit.

    hidden_layer_sizes : tuple, default=(100,)
        The number of neurons in each hidden layer of the `MLPClassifier` or 
        `MLPRegressor`. For example, (100,) means there is one hidden layer 
        with 100 neurons.

    activation : {'identity', 'logistic', 'tanh', 'relu'}, default='relu'
        Activation function for the hidden layer in the `MLPClassifier` or 
        `MLPRegressor`.
        - 'identity': no-op activation, useful for linear bottleneck, 
          returns `f(x) = x`
        - 'logistic': the logistic sigmoid function, returns 
          `f(x) = 1 / (1 + exp(-x))`
        - 'tanh': the hyperbolic tan function, returns 
          `f(x) = tanh(x)`
        - 'relu': the rectified linear unit function, returns 
          `f(x) = max(0, x)`

    solver : {'lbfgs', 'sgd', 'adam'}, default='adam'
        The solver for weight optimization in the `MLPClassifier` or 
        `MLPRegressor`.
        - 'lbfgs': optimizer in the family of quasi-Newton methods
        - 'sgd': stochastic gradient descent
        - 'adam': stochastic gradient-based optimizer proposed by Kingma, 
          Diederik, and Jimmy Ba

    alpha : float, default=0.0001
        L2 penalty (regularization term) parameter for the `MLPClassifier` or 
        `MLPRegressor`. Regularization helps to prevent overfitting by 
        penalizing large weights.

    batch_size : int, default='auto'
        Size of minibatches for stochastic optimizers. If 'auto', batch size 
        is set to `min(200, n_samples)`.

    learning_rate : {'constant', 'invscaling', 'adaptive'}, default='constant'
        Learning rate schedule for weight updates.
        - 'constant': learning rate remains constant
        - 'invscaling': gradually decreases the learning rate at each time step
        - 'adaptive': keeps the learning rate constant to `learning_rate_init` 
          as long as training loss keeps decreasing

    power_t : float, default=0.5
        The exponent for inverse scaling learning rate. It is used only 
        when `learning_rate='invscaling'`.

    tol : float, default=1e-4
        Tolerance for the optimization. Training will stop when the 
        validation score is not improving by at least `tol` for two consecutive 
        iterations.

    momentum : float, default=0.9
        Momentum for gradient descent update. Should be between 0 and 1. 
        It helps to accelerate gradients vectors in the right directions, 
        thus leading to faster converging.

    nesterovs_momentum : bool, default=True
        Whether to use Nesterov’s momentum. Nesterov’s momentum is an 
        improvement over standard momentum by considering the gradient ahead 
        of the current point in time.

    early_stopping : bool, default=False
        Whether to use early stopping to terminate training when validation 
        score is not improving. If set to `True`, it will automatically set 
        aside a fraction of training data as validation and terminate 
        training when validation score is not improving by at least `tol` for 
        `n_iter_no_change` consecutive epochs.

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for 
        early stopping. It is used only if `early_stopping` is set to `True`.

    beta_1 : float, default=0.9
        Exponential decay rate for estimates of first moment vector in adam, 
        should be in the range `[0, 1)`.

    beta_2 : float, default=0.999
        Exponential decay rate for estimates of second moment vector in adam, 
        should be in the range `[0, 1)`.

    epsilon : float, default=1e-8
        Value for numerical stability in adam. It is used to prevent any 
        division by zero in the implementation.

    n_iter_no_change : int, default=10
        Maximum number of epochs to not meet `tol` improvement. The training 
        will stop if the validation score is not improving by at least `tol` 
        for `n_iter_no_change` consecutive epochs.

    max_fun : int, default=15000
        Only used when solver='lbfgs'. Maximum number of function calls.

    random_state : int, RandomState instance, default=None
        Determines random number generation for weights and bias 
        initialization, train-test split if early stopping is used, and batch 
        sampling when solver='sgd' or 'adam'. Pass an int for reproducible 
        results across multiple function calls.

    verbose : bool, default=False
        Whether to print progress messages to stdout. If `True`, progress 
        messages are printed during training.

    Notes
    -----
    The FuzzyNeuralNetBase class combines the strengths of fuzzy logic and 
    neural networks to capture nonlinear relationships in data. Fuzzy logic 
    helps in dealing with uncertainty and imprecision, while neural networks 
    provide learning capabilities.

    The fuzzy clustering is performed using c-means clustering. The resulting 
    fuzzy sets are used to transform the input features, which are then fed 
    into an ensemble of `MLPClassifier` or `MLPRegressor` models for learning.

    The objective function for fuzzy c-means clustering is:
    
    .. math::
        \min \sum_{i=1}^{n} \sum_{j=1}^{c} u_{ij}^m \|x_i - v_j\|^2

    where :math:`u_{ij}` is the degree of membership of :math:`x_i` in the 
    cluster :math:`j`, :math:`v_j` is the cluster center, and :math:`m` is a 
    weighting exponent.

    The ensemble of neural networks helps to improve the model's robustness 
    and performance by averaging the predictions of multiple estimators, 
    which reduces the variance and the risk of overfitting.
    
    References
    ----------
    .. [1] J.-S. R. Jang, "ANFIS: Adaptive-Network-based Fuzzy Inference 
           System," IEEE Transactions on Systems, Man, and Cybernetics, vol. 
           23, no. 3, pp. 665-685, 1993.
    """

    @abstractmethod
    def __init__(
        self, 
        *, 
        n_clusters=3, 
        n_estimators=10,
        eta0=0.001, 
        max_iter=200, 
        hidden_layer_sizes=(100,), 
        activation='relu', 
        solver='adam', 
        alpha=0.0001, 
        batch_size='auto', 
        learning_rate='constant', 
        power_t=0.5, 
        tol=1e-4, 
        momentum=0.9, 
        nesterovs_momentum=True, 
        early_stopping=False, 
        validation_fraction=0.1, 
        beta_1=0.9, 
        beta_2=0.999, 
        epsilon=1e-8, 
        n_iter_no_change=10, 
        max_fun=15000, 
        random_state=None, 
        verbose=False
        ):
        self.n_clusters = n_clusters
        self.n_estimators = n_estimators
        self.learning_rate_init = eta0
        self.max_iter = max_iter
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.power_t = power_t
        self.tol = tol
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change
        self.max_fun = max_fun
        self.random_state = random_state
        self.verbose = verbose
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder()

    def _fuzzify(self, X):
        """
        Transform the input features into fuzzy sets using fuzzy c-means clustering.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples to be fuzzified.
    
        Returns
        -------
        antecedents : list of `Antecedent`
            A list of fuzzy antecedents for each feature in the input data. Each 
            antecedent represents the fuzzy clusters for a corresponding feature.
    
        Notes
        -----
        The fuzzification process involves the following steps:
    
        1. **Fuzzy c-means Clustering**:
           For each feature in `X`, fuzzy c-means clustering is applied to 
           partition the feature values into `n_clusters` fuzzy clusters. The 
           objective function for fuzzy c-means clustering is:
           
           .. math::
               \min \sum_{i=1}^{n} \sum_{j=1}^{c} u_{ij}^m \|x_i - v_j\|^2
    
           where :math:`u_{ij}` is the degree of membership of :math:`x_i` in 
           cluster :math:`j`, :math:`v_j` is the cluster center, and :math:`m` 
           is a weighting exponent.
    
        2. **Antecedent Creation**:
           For each feature, an `Antecedent` object is created representing the 
           universe of discourse for that feature. The fuzzy clusters obtained 
           from c-means clustering are then used to define the membership 
           functions for each cluster within the antecedent.
    
        This process transforms the input features into fuzzy sets, which can 
        then be used in the fuzzy inference system.
    
        Examples
        --------
        >>> from gofast.estimators.perceptron import FuzzyNeuralNetClassifier
        >>> X = [[0., 0.], [1., 1.]]
        >>> clf = FuzzyNeuralNetClassifier(n_clusters=2, n_estimators=5, 
                                           max_iter=1000, verbose=True)
        >>> antecedents = clf._fuzzify(X)
        >>> len(antecedents)
        2
        """

        n_features = X.shape[1]
        antecedents = []
        if self.verbose:
            progress_bar = tqdm(
                total=n_features, ascii=True, ncols=100,
                desc='{:<30}'.format(f'Fuzzyfying {self.estimator_name_}'), 
            )
        for i in range(n_features):
            antecedent = Antecedent(np.linspace(0, 1, 100), f'feature_{i}')
            clusters = fuzz.cluster.cmeans(
                X[:, [i]].T, self.n_clusters, 2, error=0.005, maxiter=1000)[0]
            for j, cluster in enumerate(clusters):
                antecedent[f'cluster_{j}'] = fuzz.trimf(antecedent.universe, cluster)
            antecedents.append(antecedent)
            
            if self.verbose: 
                progress_bar.update (1)
        if self.verbose: 
            progress_bar.close () 
        return antecedents

    def _fit(self, X, y, is_classifier, sample_weight=None):
        """
        Fit the ensemble of neural network models according to the given 
        training data.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data where `n_samples` is the number of samples and
            `n_features` is the number of features.
    
        y : array-like of shape (n_samples,)
            Target values. For classification tasks, these should be class labels.
            For regression tasks, these should be continuous values.
    
        is_classifier : bool
            Flag indicating whether the task is classification (`True`) or 
            regression (`False`).
    
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample. If not provided, all samples are 
            given equal weight.
    
        Returns
        -------
        self : object
            Returns self.
    
        Notes
        -----
        The fit process involves the following steps:
    
        1. **Data Validation and Preprocessing**:
           Ensures that `X` and `y` have correct shapes and types.
           Standard scaling is applied to `X`. If `is_classifier` is `True`, 
           `y` is one-hot encoded.
    
        2. **Fuzzification**:
           The input features are transformed into fuzzy sets using fuzzy
           c-means clustering:
           
           .. math::
               \min \sum_{i=1}^{n} \sum_{j=1}^{c} u_{ij}^m \|x_i - v_j\|^2
    
           where :math:`u_{ij}` is the degree of membership of :math:`x_i` in
           cluster :math:`j`, :math:`v_j` is the cluster center, and :math:`m`
           is a weighting exponent.
    
        3. **Ensemble Training**:
           An ensemble of `n_estimators` neural network models is trained. 
           Each model in the ensemble is either an `MLPClassifier` or 
           `MLPRegressor`, depending on the value of `is_classifier`.
    
           The scaled and fuzzified features are used to train each model in 
           the ensemble. The models are trained independently, and their 
           predictions will be aggregated to form the final prediction.
    
        If `verbose=True`, a progress bar will be displayed during the training 
        process for each model in the ensemble.
    
        Examples
        --------
        >>> from gofast.estimators.perceptron import FuzzyNeuralNetClassifier
        >>> X = [[0., 0.], [1., 1.]]
        >>> y = [0, 1]
        >>> clf = FuzzyNeuralNetClassifier(n_clusters=2, n_estimators=5, 
                                           max_iter=1000, verbose=True)
        >>> clf._fit(X, y, is_classifier=True)
        FuzzyNeuralNetClassifier(...)
  
        """
        
        X, y = check_X_y(X, y, estimator=self)
        self.ensemble_ = []
        
        for _ in range(self.n_estimators):
            if is_classifier:
                estimator = MLPClassifier(
                    learning_rate_init=self.eta0, 
                    max_iter=self.max_iter,
                    hidden_layer_sizes=self.hidden_layer_sizes,
                    activation=self.activation, 
                    solver=self.solver, 
                    alpha=self.alpha, 
                    batch_size=self.batch_size, 
                    learning_rate=self.learning_rate, 
                    power_t=self.power_t, 
                    tol=self.tol, 
                    momentum=self.momentum, 
                    nesterovs_momentum=self.nesterovs_momentum, 
                    early_stopping=self.early_stopping, 
                    validation_fraction=self.validation_fraction, 
                    beta_1=self.beta_1, 
                    beta_2=self.beta_2, 
                    epsilon=self.epsilon, 
                    n_iter_no_change=self.n_iter_no_change, 
                    max_fun=self.max_fun, 
                    random_state=self.random_state, 
                    verbose=self.verbose)
            else:
                estimator = MLPRegressor(
                    learning_rate_init=self.eta0, 
                    max_iter=self.max_iter,
                    hidden_layer_sizes=self.hidden_layer_sizes,
                    activation=self.activation, 
                    solver=self.solver, 
                    alpha=self.alpha, 
                    batch_size=self.batch_size, 
                    learning_rate=self.learning_rate, 
                    power_t=self.power_t, 
                    tol=self.tol, 
                    momentum=self.momentum, 
                    nesterovs_momentum=self.nesterovs_momentum, 
                    early_stopping=self.early_stopping, 
                    validation_fraction=self.validation_fraction, 
                    beta_1=self.beta_1, 
                    beta_2=self.beta_2, 
                    epsilon=self.epsilon, 
                    n_iter_no_change=self.n_iter_no_change, 
                    max_fun=self.max_fun, 
                    random_state=self.random_state, 
                    verbose=self.verbose
                    )
            self.ensemble_.append(estimator)
            
        X_scaled = self.scaler.fit_transform(X)
        if is_classifier:
            y_encoded = self.encoder.fit_transform(y.reshape(-1, 1)).toarray()
        else:
            y_encoded = y
        
        self.estimator_name_= ( "FuzzyNeuralNetClassifier" if is_classifier 
                               else "FuzzyNeuralNetRegressor" ) 
        self.antecedents_ = self._fuzzify(X_scaled)
        
        if self.verbose:
            progress_bar = tqdm(
                total=self.n_estimators, ascii=True, ncols=100,
                desc='{:<30}'.format(f'Fitting {self.estimator_name_}'), 
            )
        for estimator in self.ensemble_:
            estimator.fit(X_scaled, y_encoded)
            
            if self.verbose: 
                progress_bar.update (1)
        if self.verbose: 
            progress_bar.close () 
                
        return self
    
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
        from .tree import DecisionStumpRegressor 
        X, y = check_X_y(X, y, estimator= self )
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

    eta0 : float, optional (default=1.0)
        Learning rate shrinks the contribution of each tree. There is a 
        trade-off between eta0 and n_estimators.
        
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
    >>> model = GradientBoostingRegressor(n_estimators=100, eta0=0.1)
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

    def __init__(self, n_estimators=100, eta0=1.0, max_depth=1):
        self.n_estimators = n_estimators
        self.eta0 = eta0
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
        >>> reg = GradientBoostingRegressor(n_estimators=50, eta0=0.1)
        >>> reg.fit(X, y)
        """
        from .tree import DecisionStumpRegressor 
        X, y = check_X_y(X, y, estimator= self )
        if X.shape[0] != y.shape[0]:
            raise ValueError("Mismatched number of samples between X and y")

        # Initialize the prediction to zero
        F_m = np.zeros(y.shape)

        for m in range(self.n_estimators):
            # Compute residuals
            residuals = y - F_m

            # # Fit a regression tree to the negative gradient
            tree = DecisionStumpRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)

            # Update the model predictions
            F_m += self.eta0 * tree.predict(X)

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
        F_m = sum(self.eta0 * estimator.predict(X)
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
        return sum(self.eta0 * estimator.predict(X)
                   for estimator in self.estimators_)

