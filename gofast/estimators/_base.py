# -*- coding: utf-8 -*-

from __future__ import annotations
from collections import defaultdict 
import inspect 
from scipy.sparse import issparse
import numpy as np 

from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler

from ..tools.validator import check_array, check_X_y
from ..tools.validator import validate_fit_weights

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
    

def get_default_meta_estimator(
        meta_estimator: BaseEstimator = None, problem: str = 'regression'
        ) -> BaseEstimator:
    """
    Select and return a default meta estimator if none is provided.

    Parameters
    ----------
    meta_estimator : BaseEstimator, optional
        The meta estimator provided by the user. If None, a default estimator
        will be selected based on the problem type.
    problem : str, default='regression'
        The type of problem. Options are 'regression' for regression problems
        and 'classification' for classification problems.

    Returns
    -------
    meta_estimator : BaseEstimator
        The selected meta estimator. Default is LinearRegression for regression
        problems and LogisticRegression for classification problems.

    Raises
    ------
    ValueError
        If an invalid problem type is provided.

    Examples
    --------
    >>> from gofast.estimators._base import get_default_meta_estimator
    >>> regressor = get_default_meta_estimator(problem='regression')
    >>> classifier = get_default_meta_estimator(problem='classification')
    """
    if meta_estimator is not None:
        if not isinstance(meta_estimator, BaseEstimator):
            raise ValueError("The meta_estimator must be an instance of sklearn.base.BaseEstimator")
        return meta_estimator

    if problem.lower() in ('regression', 'reg'):
        return LinearRegression()
    elif problem.lower() in ('classification', 'class', 'clf'):
        return LogisticRegression()
    else:
        raise ValueError("Invalid problem type. Expected 'regression' or 'classification'.")
        
def fit_with_estimator(
        estimator, X, y, sample_weight=None, apply_weighted_y=False, 
        return_weighted_y=False):
    """
    Fit an estimator with the given data and handle sample weights appropriately.

    Parameters
    ----------
    estimator : estimator object
        The estimator to fit.
    X : array-like of shape (n_samples, n_features)
        The training input samples.
    y : array-like of shape (n_samples,)
        The target values.
    sample_weight : array-like of shape (n_samples,), default=None
        Individual weights for each sample.
    apply_weighted_y : bool, default=False
        Whether to adjust `y` by `sample_weight` before fitting if the estimator
        does not accept `sample_weight` directly.
    return_weighted_y : bool, default=False
        Whether to return the modified `y` values along with the fitted estimator.

    Returns
    -------
    estimator : estimator object
        The fitted estimator.
    y : array-like of shape (n_samples,), optional
        The modified target values if `return_weighted_y` is True.

    Notes
    -----
    This function checks if the estimator accepts `sample_weight`. If not, and
    `apply_weighted_y` is True, it adjusts the target values `y` using the
    sample weights before fitting the estimator. This is useful for estimators
    that do not natively support sample weights.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.datasets import make_regression
    >>> from gofast.estimators._base import fit_with_estimator
    >>> X, y = make_regression(n_samples=100, n_features=2, noise=0.1)
    >>> estimator = LinearRegression()
    >>> fitted_estimator = fit_with_estimator(estimator, X, y)
    """
    

    # Validate inputs
    X, y = check_X_y(X, y)
    # Validate and process sample weights
    sample_weight = validate_fit_weights(y, sample_weight)

    # Check if the estimator accepts sample_weight parameter
    from inspect import signature
    sig = signature(estimator.fit)
    if 'sample_weight' in sig.parameters:
        estimator.fit(X, y, sample_weight=sample_weight)
    else:
        if apply_weighted_y:
            _, y_weighted = validate_fit_weights(y, sample_weight, weighted_y=True)
            estimator.fit(X, y_weighted)
        else:
            estimator.fit(X, y)

    return (estimator, y) if return_weighted_y else estimator

def determine_weights(
    base_estimators, 
    X, y, 
    cv=None,
    scoring='neg_mean_squared_error', 
    problem='regression'
    ):
    """
    Determine weights based on the performance of base models.

    Parameters
    ----------
    base_estimators : list of estimators
        List of base estimators to be evaluated. Each estimator should 
        support `fit` and `predict` methods. These estimators will be 
        combined through weighted averaging to form the ensemble model.

    X : array-like of shape (n_samples, n_features)
        The training input samples. These represent the features of the 
        dataset used to train the model. Each row corresponds to a sample,
        and each column corresponds to a feature.

    y : array-like of shape (n_samples,)
        The target values for each sample. These are the true values that
        the model aims to predict. Each element corresponds to the target
        value for a respective sample in `X`.

    cv : int, cross-validation generator or an iterable, default=5
        Determines the cross-validation splitting strategy. It specifies 
        the number of folds in cross-validation or provides an iterable 
        yielding train/test splits. If None, 5-fold cross-validation is used.

    scoring : str, callable, list/tuple, or dict, default='neg_mean_squared_error'
        A string (see model evaluation documentation) or a scorer callable 
        object / function with signature `scorer(estimator, X, y)`. It defines
        the metric used to evaluate the performance of the base estimators.
        Default is 'neg_mean_squared_error', suitable for regression tasks.
        For classification tasks, it is recommended to use 'accuracy' or 
        another appropriate classification metric.

    problem : str, {'regression', 'classification'}, default='regression'
        Defines the type of problem being solved. If 'regression', it uses 
        the default scoring for regression problems. If 'classification', 
        it changes the default scoring to 'accuracy'.

    Returns
    -------
    weights : array-like of shape (n_estimators,)
        Weights for each base estimator, determined based on performance.
        The weights are proportional to the mean cross-validation scores
        of the estimators. They sum to 1.

    Notes
    -----
    This function evaluates each base estimator using cross-validation and
    assigns weights proportional to their performance (e.g., mean cross-
    validation score). The resulting weights are normalized to sum to 1.

    The weighting strategy can be mathematically represented as:

    .. math::
        w_i = \frac{\bar{s}_i}{\sum_{j=1}^{N} \bar{s}_j}

    where:
    - :math:`w_i` is the weight assigned to the \(i\)-th base estimator.
    - :math:`\bar{s}_i` is the mean cross-validation score of the \(i\)-th 
      base estimator.
    - \(N\) is the number of base estimators in the ensemble.

    Example
    -------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from gofast.estimators._base import determine_weights
    >>> import numpy as np
    >>> base_estimators = [LogisticRegression(), DecisionTreeClassifier()]
    >>> X = np.random.rand(100, 4)
    >>> y = np.random.randint(0, 2, size=100)
    >>> weights = determine_weights(base_estimators, X, y, problem='classification')
    >>> print(weights)

    See Also
    --------
    - cross_val_score : Evaluate a score by cross-validation.
    - make_scorer : Make a scorer from a performance metric or loss function.
    - WeightedAverageRegressor : Ensemble regressor using weighted average.

    """
    scores = []
    cv = cv or 5 
    if isinstance(scoring, str) and problem.lower() == 'classification':
        scoring = 'accuracy'
        
    for estimator in base_estimators:
        # Perform cross-validation and get the mean score
        score = cross_val_score(estimator, X, y, cv=cv, scoring=scoring)
        scores.append(np.mean(score))
    
    # Convert scores to weights by normalizing them
    scores = np.array(scores)
    # Ensure no negative weights (e.g., converting scores if using negative error metrics)
    if np.any(scores < 0):
        scores = scores - np.min(scores)
    weights = scores / np.sum(scores)
    
    return weights

def apply_scaling(scaler, X, return_keys=False):
    """
    Apply the specified scaler to the data.

    Parameters
    ----------
    scaler : str or object
        The scaler to use for scaling the input data. Supported scalers include:
        - 'minmax' or '01': Uses `MinMaxScaler` to scale features to [0, 1].
        - 'standard' or 'zscore': Uses `StandardScaler` to scale features to have
          zero mean and unit variance.
        - 'robust': Uses `RobustScaler` to scale features using statistics that
          are robust to outliers.
        - 'sum': Normalizes features by dividing each element by the sum of its
          respective row.
        If `scaler` is already a fitted scaler object, it will be used directly.

    X : array-like of shape (n_samples, n_features)
        The input data to scale.

    return_keys : bool, default=False
        If True, returns  the scaler and the key of the scaler used along 
        with the scaled data.

    Returns
    -------
    X_scaled : array-like of shape (n_samples, n_features)
        The scaled input data.

    Notes
    -----
    This method applies the scaler specified by the `scaler` attribute to the
    input data `X`. If `scaler` is a string, it maps to a corresponding scaler
    class and applies it to `X`. If `scaler` is already a fitted scaler object,
    it is used directly to transform `X`.

    Examples
    --------
    >>> from sklearn.preprocessing import StandardScaler
    >>> from gofast.estimators._base import apply_scaling
    >>> X = [[1, 2], [3, 4], [5, 6]]
    >>> X_scaled = apply_scaling('standard', X)
    >>> print(X_scaled)

    >>> scaler = StandardScaler()
    >>> X_scaled = apply_scaling(scaler, X)
    >>> print(X_scaled)

    See Also
    --------
    - sklearn.preprocessing.MinMaxScaler
    - sklearn.preprocessing.StandardScaler
    - sklearn.preprocessing.RobustScaler

    References
    ----------
    .. [1] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., 
           Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., 
           Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., 
           and Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. 
           Journal of Machine Learning Research, 12, 2825-2830.

    """
    # Validate input array
    X = check_array(X, accept_sparse=True)

    # Define a mapping from string identifiers to scaler classes
    scaler_map = {
        'minmax': MinMaxScaler,
        '01': MinMaxScaler,
        'standard': StandardScaler,
        'zscore': StandardScaler,
        'robust': RobustScaler
    }

    # If scaler is a string, map to corresponding scaler class
    if isinstance(scaler, str):
        scaler_key = scaler.lower()
        if scaler_key in scaler_map:
            scaler = scaler_map[scaler_key]()
        elif scaler_key == 'sum':
            X = normalize_sum(X)
        else:
            raise ValueError(f"Unknown scaler type: {scaler}")

    # Ensure the scaler has a fit_transform method
    if callable(getattr(scaler, "fit_transform", None)):
        X = scaler.fit_transform(X)
    else:
        raise TypeError("Scaler must be a string or a fitted scaler object"
                        f" with fit_transform method, got {type(scaler)}")

    # Return scaled data, and optionally the scaler keys used
    return (X, scaler, scaler_key) if return_keys else X

def normalize_sum(X, axis =1 ):
    """
    Normalize the data by dividing each element by the sum of its respective row.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data to normalize. Can be a dense array or sparse matrix.
        
    axis: int, default=1 
       Axis along with the sum is applied. 
       
    Returns
    -------
    X_normalized : array-like of shape (n_samples, n_features)
        The sum-normalized input data.

    Notes
    -----
    This method normalizes the input data `X` by dividing each element by the 
    sum of its respective row. This ensures that the sum of the elements in each 
    row is equal to 1. This is useful for various machine learning algorithms 
    where normalized inputs are required.

    For a dense matrix, each element is divided by the sum of its row directly.
    For a sparse matrix, the non-zero elements are divided by the sum of their 
    respective rows.

    Mathematically, the normalization for each element \( x_{ij} \) in row \( i \) 
    and column \( j \) of matrix \( X \) is given by:

    .. math::
        x_{ij}^{\text{normalized}} = \frac{x_{ij}}{\sum_{k=1}^{n} x_{ik}}

    where:
    - \( x_{ij}^{\text{normalized}} \) is the normalized value.
    - \( x_{ij} \) is the original value.
    - \( n \) is the number of columns (features).
    - \( \sum_{k=1}^{n} x_{ik} \) is the sum of all elements in row \( i \).

    Examples
    --------
    >>> from gofast.estimators.ensemble import normalize_sum
    >>> import numpy as np
    >>> from scipy.sparse import csr_matrix
    >>> 
    >>> # Example with dense matrix
    >>> X_dense = np.array([[1, 2], [3, 4], [5, 6]])
    >>> X_normalized_dense = normalize_sum(X_dense)
    >>> print("Normalized Dense Matrix:\n", X_normalized_dense)
    >>> 
    >>> # Example with sparse matrix
    >>> X_sparse = csr_matrix([[1, 2], [3, 4], [5, 6]])
    >>> X_normalized_sparse = normalize_sum(X_sparse)
    >>> print("Normalized Sparse Matrix:\n", X_normalized_sparse.toarray())

    See Also
    --------
    sklearn.preprocessing.normalize : Utility function to normalize samples.

    References
    ----------
    .. [1] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., 
           Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., 
           Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., 
           and Duchesnay, E. (2011). Scikit-learn: Machine Learning in Python. 
           Journal of Machine Learning Research, 12, 2825-2830.
    """
    if issparse(X):
        # Sum each row for sparse matrix
        row_sums = np.array(X.sum(axis=axis)).flatten()
        row_indices, col_indices = X.nonzero()
        normalized_data = X.data / row_sums[row_indices]
        return X.__class__((normalized_data, (row_indices, col_indices)), shape=X.shape)
    else:
        # Sum each row for dense matrix
        row_sums = X.sum(axis=axis)
        return X / row_sums[:, np.newaxis]
    
def optimize_hyperparams(
        estimator, X, y, param_grid=None, optimizer='RSCV', cv=None):
    """
    Optimize hyperparameters for the given estimator using the specified
    optimizer and cross-validation strategy.

    Parameters
    ----------
    estimator : estimator object
        The base estimator to optimize. The estimator should have methods 
        `fit` and `predict` to be compatible with the optimization process.

    X : array-like of shape (n_samples, n_features)
        Training data input. This represents the features of the dataset 
        used to train the model. Each row corresponds to a sample, and each 
        column corresponds to a feature.

    y : array-like of shape (n_samples,)
        Target values for each sample. These are the true values that the 
        model aims to predict. Each element corresponds to the target value 
        for a respective sample in `X`.

    optimizer : str, default='RSCV'
        The optimization method to use for hyperparameter tuning. Supported 
        options include:
        - 'RSCV' : RandomizedSearchCV for random search over hyperparameter 
          distributions.
        - 'GSCV' : GridSearchCV for exhaustive search over specified 
          hyperparameter values.
    param_grid: dict, 
       List of parameters for estimator fine-tuning. If ``None``, none 
       grid param is passed and return default tunning. 
       
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy. It specifies the 
        number of folds in cross-validation or provides an iterable yielding 
        train/test splits. If None, 5-fold cross-validation is used.

    Returns
    -------
    estimator : estimator object
        The optimized estimator. If the optimizer was used, the returned 
        estimator will be the best estimator found during the optimization 
        process. Otherwise, the original estimator is returned.

    Examples
    --------
    >>> from gofast.estimators._base import optimize_hyperparams
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> X = np.random.rand(100, 10)
    >>> y = np.random.rand(100)
    >>> estimator = RandomForestRegressor()
    >>> optimized_estimator = optimize_hyperparams(
        estimator, X, y, optimizer='GSCV')

    Notes
    -----
    Hyperparameter optimization is a critical step in the machine learning 
    pipeline to improve the performance of a model. It involves searching for 
    the best combination of hyperparameters that yield the highest performance 
    on a given dataset.

    The choice of optimizer (`optimizer` parameter) can significantly impact 
    the efficiency and effectiveness of the hyperparameter search:
    - 'RSCV' (RandomizedSearchCV) is suitable for large hyperparameter spaces 
      as it samples a fixed number of hyperparameter settings from specified 
      distributions.
    - 'GSCV' (GridSearchCV) performs an exhaustive search over a specified 
      parameter grid, which can be computationally expensive but thorough.

    See Also
    --------
    sklearn.model_selection.RandomizedSearchCV : Randomized search on hyperparameters.
    sklearn.model_selection.GridSearchCV : Exhaustive search over specified parameter values.

    References
    ----------
    .. [1] Bergstra, J., Bardenet, R., Bengio, Y., and Kegl, B. (2011). 
           Algorithms for Hyper-Parameter Optimization. In Advances in Neural 
           Information Processing Systems (pp. 2546-2554).
    .. [2] Bergstra, J., and Bengio, Y. (2012). Random Search for Hyper-Parameter 
           Optimization. Journal of Machine Learning Research, 13(Feb), 281-305.
    """
    from ..models.utils import get_optimizer_method
    
    param_grid = param_grid or {}
    # Validate and retrieve the optimizer method
    optimizer = get_optimizer_method(optimizer)
    
    # Initialize the search object (RandomizedSearchCV or GridSearchCV)
    search = optimizer(estimator, param_grid, cv=cv) if optimizer else estimator

    # Fit the search object to the data
    search.fit(X, y)
    # Return the best estimator if available, otherwise return the search object
    return search.best_estimator_ if hasattr(search, 'best_estimator_') else search

# Example usage
if __name__ == "__main__":
    regressor = get_default_meta_estimator(problem='regression')
    print(f"Selected meta regressor: {regressor}")

    classifier = get_default_meta_estimator(problem='classification')
    print(f"Selected meta classifier: {classifier}")
