# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
`adaline` module implements various versions of the Adaptive Linear Neuron 
(Adaline) algorithm for both regression and classification tasks.
The Adaline algorithm is a foundational model in machine 
learning that uses a linear activation function and is trained using gradient 
descent. This module includes different variants of Adaline to handle 
stochastic and batch training as well as mixed tasks that can perform both 
regression and classification. 
"""

from __future__ import annotations 
import numpy as np
from tqdm import tqdm 

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle

from ..compat.sklearn import type_of_target 
from ..decorators import Deprecated 
from ..utils.validator import check_X_y, check_array 
from ..utils.validator import check_is_fitted
from ._adaline import BaseAdaline, BaseAdalineStochastic 

__all__= [
        "AdalineClassifier","AdalineRegressor",
        "SGDAdalineRegressor","SGDAdalineClassifier",
    ]

class SGDAdalineRegressor(BaseAdalineStochastic, RegressorMixin):
    """
    Adaline Stochastic Gradient Descent Regressor for linear regression.

    Implements Adaptive Linear Neuron algorithm with stochastic gradient 
    descent optimization. Suitable for large datasets due to incremental 
    weight updates. Minimizes Mean Squared Error (MSE) through iterative 
    parameter adjustments.

    The weight update rule follows:

    .. math:: 
        w^{(t+1)} = w^{(t)} + \eta_0 \\cdot (y^{(i)} - \\phi(z^{(i)})) 
        \\cdot x^{(i)}

    where:
        - :math:`w^{(t)}`: Weight vector at iteration t
        - :math:`\eta_0`: Learning rate
        - :math:`y^{(i)}`: True value for i-th sample
        - :math:`\\phi(z^{(i)})`: Linear activation output
        - :math:`x^{(i)}`: Feature vector for i-th sample

    Parameters
    ----------
    eta0 : float, default=0.0001
        Initial learning rate (0 < eta0 <= 1). Controls update step size. 
        Smaller values improve stability but slow convergence. Values >0.1 
        may cause divergence.

    max_iter : int, default=1000
        Maximum training epochs (full dataset passes). Defines upper 
        iteration limit if convergence not reached earlier.

    early_stopping : bool, default=False
        Enable early termination when validation loss plateaus. Requires 
        ``validation_fraction`` >0.

    validation_fraction : float, default=0.1
        Proportion of training data (0-1) reserved for validation when 
        ``early_stopping=True``. Larger values reduce training data but 
        improve stopping reliability.

    tol : float, default=1e-4
        Minimum absolute improvement threshold for early stopping. Training 
        stops if validation MSE decreases by less than ``tol`` for two 
        consecutive epochs.

    warm_start : bool, default=False
        Reuse previous fit results for incremental training. When False, 
        resets weights at each fit call. Useful for online learning 
        scenarios.

    learning_rate : {'constant', 'adaptive'}, default='constant'
        Learning schedule type:
        - ``'constant'``: Fixed rate throughout training
        - ``'adaptive'``: Geometrically decayed rate via 
          ``eta0_decay`` factor

    eta0_decay : float, default=0.99
        Decay multiplier (0-1) for adaptive learning rate. Applied as:
        :math:`\eta_0^{(t+1)} = \eta_0^{(t)} \\cdot \\text{eta0\_decay}`

    shuffle : bool, default=True
        Shuffle training data each epoch. Prevents cyclical update patterns 
        and improves generalization. Disable for deterministic behavior.

    random_state : int, default=None
        Seed for random number generation (weight initialization & 
        shuffling). Enables reproducible results across runs.

    verbose : int, default=0
        Output verbosity control:
        - 0: Silent mode
        - 1: Progress bar display
        - 2: Per-epoch metrics
        - >=3: Detailed debug logging

    epsilon : float, default=1e-8
        Numerical stability constant. Clips weight updates to 
        :math:`[-1/\\epsilon, 1/\\epsilon]` to prevent overflow. Lower 
        values allow larger weight magnitudes.

    Attributes
    ----------
    weights_ : ndarray of shape (n_features + 1,)
        Learned weight vector with bias term (w0) as first element. 
        Initialized via normal distribution with scale=0.01.

    cost_ : list of float
        Average MSE values per epoch. Monitors training progress and 
        convergence.

    Notes
    -----
    - **Convergence**: Plot ``cost_`` to diagnose learning issues - flat 
      curves suggest too low ``eta0``, oscillations indicate high ``eta0``
    - **Feature Scaling**: Critical for performance - center and scale 
      features to zero mean/unit variance
    - **Sparse Data**: Supports CSC sparse matrices for memory-efficient 
      processing of high-dimensional data
    - **Adaptive Learning**: Combine ``learning_rate='adaptive'`` with 
      ``eta0_decay=0.95-0.99`` for noisy or non-stationary targets

    Examples
    --------
    >>> from gofast.estimators.adaline import SGDAdalineRegressor
    >>> from sklearn.preprocessing import StandardScaler
    >>> X, y = fetch_california_housing(return_X_y=True)
    >>> X = StandardScaler().fit_transform(X)
    >>> reg = SGDAdalineRegressor(eta0=0.005, max_iter=500, 
    ...                          early_stopping=True)
    >>> reg.fit(X, y)
    >>> predictions = reg.predict(X[:5])

    See Also
    --------
    SGDRegressor : Scikit-learn's SGD implementation with various losses
    LinearRegression : Ordinary least squares linear model
    HuberRegressor : Robust regression with SGD training

    References
    ----------
    .. [1] Rosenblatt, F. (1958). The perceptron: A probabilistic model for 
       information storage and organization in the brain. Psychological 
       Review, 65(6), 386-408.
    .. [2] Bottou, L. (2010). Large-scale machine learning with stochastic 
       gradient descent. Proceedings of COMPSTAT'2010, 177-186.
    """
    
    def __init__(
        self, 
        eta0=0.001, 
        max_iter=1000, 
        early_stopping=False, 
        validation_fraction=0.1, 
        tol=1e-4, 
        warm_start=False, 
        learning_rate='constant', 
        eta0_decay=0.99, 
        shuffle=True, 
        random_state=None,  
        epsilon=1e-8,
        verbose=0
        ):
        super().__init__(
            eta0=eta0, 
            max_iter=max_iter, 
            early_stopping=early_stopping, 
            validation_fraction=validation_fraction, 
            tol=tol, 
            warm_start=warm_start, 
            learning_rate=learning_rate, 
            eta0_decay=eta0_decay, 
            shuffle=shuffle, 
            random_state=random_state, 
            epsilon=epsilon, 
            verbose=verbose
        )

    def fit(self, X, y, sample_weight=None):
        """
        Fit training data.
    
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.
    
        y : array-like, shape (n_samples,)
            Target values.
    
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. Currently, this parameter is not used by the method 
            but is included for API consistency.
    
        Returns
        -------
        self : object
            Returns self.
    
        Notes
        -----
        This method fits the Adaline Stochastic Gradient Descent Regressor to the 
        training data `X` and target values `y`. The model weights are updated 
        incrementally for each training instance based on the learning rate `eta0`.
    
        The weight update rule in SGD for Adaline is given by:
    
        .. math::
            w := w + \eta0 (y^{(i)} - \phi(z^{(i)})) x^{(i)}
    
        where:
        - :math:`w` is the weight vector.
        - :math:`\eta0` is the learning rate.
        - :math:`y^{(i)}` is the true value for the \(i\)-th training instance.
        - :math:`\phi(z^{(i)})` is the predicted value using the linear activation 
          function.
        - :math:`x^{(i)}` is the feature vector of the \(i\)-th training instance.
    
        If `early_stopping` is enabled, a portion of the training data is set aside 
        as a validation set. Training stops if the validation error does not improve 
        by at least `tol` for two consecutive epochs.
    
        The learning rate can be adjusted adaptively by setting `learning_rate` to 
        'adaptive'. In this case, the learning rate `eta0` is multiplied by `eta0_decay`
        after each epoch.
    
        Examples
        --------
        >>> from gofast.estimators._base import AdalineStochasticRegressor
        >>> import numpy as np
        >>> from sklearn.datasets import fetch_california_housing
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.preprocessing import StandardScaler
    
        >>> # Load the California Housing dataset
        >>> housing = fetch_california_housing()
        >>> X, y = housing.data, housing.target
        >>> X_train, X_test, y_train, y_test = train_test_split(
        >>>     X, y, test_size=0.3, random_state=0)
    
        >>> # Standardize features
        >>> sc = StandardScaler()
        >>> X_train_std = sc.fit_transform(X_train)
        >>> X_test_std = sc.transform(X_test)
    
        >>> # Initialize and fit the regressor
        >>> ada_sgd_reg = AdalineStochasticRegressor(
        >>>     eta0=0.0001, max_iter=1000, early_stopping=True, 
        >>>     validation_fraction=0.1, tol=1e-4, verbose=True)
        >>> ada_sgd_reg.fit(X_train_std, y_train)
        >>> y_pred = ada_sgd_reg.predict(X_test_std)
        >>> print('Mean Squared Error:', np.mean((y_pred - y_test) ** 2))
    
        See Also
        --------
        sklearn.linear_model.SGDRegressor : Linear regression model fitted by SGD 
            with a variety of loss functions.
        sklearn.linear_model.LinearRegression : Ordinary least squares Linear 
            Regression.
        sklearn.linear_model.Ridge : Ridge regression.
    
        """
        self._validate_params() 
        check_params = dict(
            estimator=self, 
            accept_sparse="csc", 
            ensure_2d=False, 
            dtype=None, 
        )
        X, y = self._validate_data(X, y, **check_params)

        if not self.warm_start or not hasattr(self, 'weights_'):
            self._initialize_weights(X.shape[1])

        self.cost_ = []

        if self.early_stopping:
            X, X_val, y, y_val = train_test_split(
                X, y, test_size=self.validation_fraction, 
                random_state=self.random_state
                )
        
        if self.verbose > 0:
            progress_bar = tqdm(
                total=self.max_iter, ascii=True, ncols=100,
                desc=f'Fitting {self.__class__.__name__}', 
            )

        for i in range(self.max_iter):
            if self.shuffle:
                X, y = shuffle(X, y, random_state=self.random_state)
            cost = []
            for xi, target in zip(X, y):
                error = target - self.predict(xi.reshape(1, -1))
                # Prevent overflow in weight updates using epsilon
                self.weights_[1:] += self.eta0 * xi * error
                self.weights_[0] += self.eta0 * error
                # Apply epsilon to prevent weights from growing too large
                self.weights_ = np.clip(
                    self.weights_, -1/self.epsilon, 1/self.epsilon
                    )
                cost.append(error ** 2 / 2.0)
            self.cost_.append(np.mean(cost))
            
            if self.early_stopping:
                y_val_pred = self.predict(X_val)
                val_error = mean_squared_error(y_val, y_val_pred)
                if val_error < self.tol:
                    if self.verbose > 1:
                        print(f'Early stopping at epoch {i+1}')
                        progress_bar.update(self.max_iter - i)
                    break

            if self.learning_rate == 'adaptive':
                self.eta0 *= self.eta0_decay

            if self.verbose > 1:
                progress_bar.update(1)

        if self.verbose > 0:
            progress_bar.close()
        
        return self

    def predict(self, X):
        """
        Predict continuous output.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features. Can be a dense array
            or sparse matrix.

        Returns
        -------
        y_pred : array, shape (n_samples,)
            Predicted continuous values.
        
        Notes
        -----
        This method predicts the continuous output values for the input data `X` 
        using the weights learned during training. The prediction is based on 
        calculating the net input:

        .. math::
            z = X \cdot w + b

        where:
        - :math:`X` is the input feature matrix.
        - :math:`w` is the weight vector.
        - :math:`b` is the bias term.

        The predicted value :math:`z` is the linear combination of the input 
        features and the learned weights.
        
        Examples
        --------
        >>> from gofast.estimators.adaline import AdalineStochasticRegressor
        >>> import numpy as np
        >>> from sklearn.datasets import fetch_california_housing
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.preprocessing import StandardScaler
    
        >>> # Load the California Housing dataset
        >>> housing = fetch_california_housing()
        >>> X, y = housing.data, housing.target
        >>> X_train, X_test, y_train, y_test = train_test_split(
        >>>     X, y, test_size=0.3, random_state=0)
    
        >>> # Standardize features
        >>> sc = StandardScaler()
        >>> X_train_std = sc.fit_transform(X_train)
        >>> X_test_std = sc.transform(X_test)
    
        >>> # Initialize and fit the regressor
        >>> ada_sgd_reg = AdalineStochasticRegressor(
        >>>     eta0=0.0001, max_iter=1000)
        >>> ada_sgd_reg.fit(X_train_std, y_train)
    
        >>> # Predict using the trained model
        >>> y_pred = ada_sgd_reg.predict(X_test_std)
        >>> print('Predicted values:', y_pred)
        """
        check_is_fitted(self, 'weights_')
        X = check_array(X, accept_large_sparse=True, accept_sparse=True)
        return self.net_input(X)


class SGDAdalineClassifier(BaseAdalineStochastic, ClassifierMixin):
    """
    Adaline Classifier with Stochastic Gradient Descent optimization.

    Implements Adaptive Linear Neuron algorithm for classification tasks 
    using SGD. Suitable for both binary and multiclass classification 
    through one-vs-rest strategy. Applies activation function to linear 
    outputs for probabilistic predictions.

    The weight update rule follows:

    .. math:: 
        w^{(t+1)}_k = w^{(t)}_k + \eta_0 \\cdot (y^{(i)}_k - \\phi(z^{(i)}_k)) 
        \\cdot x^{(i)}

    where:
        - :math:`w^{(t)}_k`: Weight vector for class k at iteration t
        - :math:`\eta_0`: Learning rate controlling update magnitude
        - :math:`y^{(i)}_k`: Binary indicator for class k membership
        - :math:`\\phi(z^{(i)}_k)`: Activation function output
        - :math:`x^{(i)}`: Feature vector of i-th sample

    Parameters
    ----------
    eta0 : float, default=0.01
        Initial learning rate (0 < eta0 <= 1). Larger values increase 
        convergence speed but risk overshooting optima. Typical range 
        0.0001-0.1.

    max_iter : int, default=1000
        Maximum number of training epochs. Each epoch processes all 
        samples once. Lower values prevent overfitting but may underfit.

    early_stopping : bool, default=False
        Enable early termination when validation accuracy plateaus. 
        Monitors ``validation_fraction`` split for convergence detection.

    validation_fraction : float, default=0.1
        Proportion of training data (0-1) reserved for validation when 
        ``early_stopping=True``. Balance between training data size and 
        validation reliability.

    tol : float, default=1e-4
        Minimum relative improvement threshold for early stopping. 
        Training stops if validation accuracy improvement < ``tol`` 
        for 2 consecutive epochs.

    warm_start : bool, default=False
        Reuse weights from previous fit calls for incremental training. 
        Enables model refinement with new data without reinitialization.

    learning_rate : {'constant', 'adaptive'}, default='constant'
        Learning schedule strategy:
        - ``'constant'``: Fixed learning rate throughout training
        - ``'adaptive'``: Geometrically decayed rate via 
          ``eta0_decay`` factor per epoch

    eta0_decay : float, default=0.99
        Decay factor (0-1) for adaptive learning rate. Applied as:
        :math:`\eta_0^{(t+1)} = \eta_0^{(t)} \\cdot \\text{eta0\_decay}`

    activation : {'sigmoid', 'relu', 'leaky_relu', 'identity', 'elu', 
                 'tanh', 'softmax'} or callable, default='sigmoid'
        Activation function mapping net input to class probabilities:
        - ``'sigmoid'``: Logistic function for binary classification
        - ``'softmax'``: Normalized exponentials for multiclass
        - Custom functions should accept array input and return 
          same-shaped array

    shuffle : bool, default=True
        Shuffle training data each epoch. Breaks sample order 
        dependencies and improves generalization. Disable for 
        deterministic behavior.

    random_state : int, default=None
        Seed for reproducible weight initialization and shuffling. 
        Enables consistent results across multiple runs.
        
    epsilon : float, default=1e-8
        Numerical stability constant. Clips weights to 
        :math:`[-1/\\epsilon, 1/\\epsilon]` to prevent overflow 
        during updates.    

    verbose : int, default=0
        Verbosity control:
        - 0: Silent mode
        - 1: Progress bar display
        - 2: Per-epoch metrics
        - >=3: Detailed debug logging

    Attributes
    ----------
    weights_ : ndarray of shape (n_features + 1, n_classes)
        Learned weight matrix with bias terms. Each column represents 
        weights for one class in one-vs-rest configuration.

    cost_ : list of float
        Average cross-entropy loss per epoch. Monitors training 
        convergence.

    label_binarizer_ : LabelBinarizer
        Internal label binarizer handling multiclass conversions.

    Methods
    -------
    fit(X, y, sample_weight=None)
        Train classifier on input data X and target labels y.

    predict(X)
        Predict class labels for samples in X.

    predict_proba(X)
        Estimate class probabilities for samples in X.

    Notes
    -----
    - For multiclass problems, uses one-vs-rest strategy with 
      separate weight vectors per class
    - Sigmoid/softmax activations enable probabilistic outputs
    - Binary classification requires 0/1 labels, multiclass 
      needs integer class labels
    - Prefer ``learning_rate='adaptive'`` for noisy or non-stationary 
      data streams

    Examples
    --------
    >>> from gofast.estimators.adaline import SGDAdalineClassifier
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.preprocessing import StandardScaler
    >>> X, y = load_iris(return_X_y=True)
    >>> X = StandardScaler().fit_transform(X)
    >>> clf = SGDAdalineClassifier(activation='softmax', eta0=0.01,
    ...                           max_iter=500, random_state=42)
    >>> clf.fit(X, y)
    >>> predictions = clf.predict(X[:5])
    >>> probabilities = clf.predict_proba(X[:5])

    See Also
    --------
    SGDClassifier : Scikit-learn's SGD implementation for classification
    LogisticRegression : Linear model with logistic loss
    MLPClassifier : Neural network-based classifier

    References
    ----------
    .. [1] Bishop, C.M. (2006). Pattern Recognition and Machine Learning.
       Springer.
    .. [2] Zhang, T. (2004). Solving large scale linear prediction problems 
       using stochastic gradient descent algorithms. ICML.
    """
    
    def __init__(
        self, 
        eta0=0.001, 
        max_iter=1000, 
        early_stopping=False, 
        validation_fraction=0.1, 
        tol=1e-4, 
        warm_start=False, 
        learning_rate='constant', 
        eta0_decay=0.99, 
        activation='sigmoid', 
        shuffle=True, 
        random_state=None, 
        epsilon=1e-8,
        verbose=0, 
        ):
        super().__init__(
            eta0=eta0, 
            max_iter=max_iter, 
            early_stopping=early_stopping, 
            validation_fraction=validation_fraction, 
            tol=tol, 
            warm_start=warm_start, 
            learning_rate=learning_rate, 
            eta0_decay=eta0_decay, 
            shuffle=shuffle, 
            random_state=random_state, 
            activation=activation, 
            epsilon=epsilon, 
            verbose=verbose
        )

    def fit(self, X, y, sample_weight=None):
        """
        Train the Adaline classifier using stochastic gradient descent.
    
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data matrix. Accepts both dense arrays and CSC sparse 
            matrices for memory-efficient processing of high-dimensional data.
    
        y : array-like of shape (n_samples,)
            Target class labels. For binary classification, expects 0/1 labels. 
            For multiclass, requires integer class labels.
    
        sample_weight : array-like of shape (n_samples,), default=None
            Currently unused, present for scikit-learn API compatibility.
    
        Returns
        -------
        self : object
            Fitted classifier instance with updated ``weights_`` and ``cost_`` 
            attributes.
    
        Notes
        -----
        Implements the following key steps:
        1. Label binarization for multiclass handling using one-vs-rest strategy
        2. Weight initialization from :math:`\mathcal{N}(0, 0.01^2)` distribution
        3. Per-epoch processing:
           - Data shuffling (if enabled)
           - Per-sample weight updates for each class
           - Weight clipping to :math:`[-1/\\epsilon, 1/\\epsilon]` range
           - Cost calculation using MSE
        4. Early stopping monitoring (if enabled)
        5. Adaptive learning rate decay (if configured)
    
        The weight update rule for class k is:
        .. math:: 
            \Delta w_k = \eta_0 \cdot (y_k - \phi(z_k)) \cdot \mathbf{x}
    
        Examples
        --------
        >>> from gofast.estimators.adaline import SGDAdalineClassifier
        >>> clf = SGDAdalineClassifier(eta0=0.01, max_iter=100)
        >>> clf.fit(X_train, y_train)
        """

        self._validate_params() 
        check_params = dict(
            estimator=self, 
            accept_sparse="csc", 
            ensure_2d=False, dtype=None, 
            multi_output=True
            )
        X, y = self._validate_data(X, y, **check_params)
        self.label_binarizer_ = LabelBinarizer()
        y = self.label_binarizer_.fit_transform(y)
        
        if y.ndim == 1:
            y = y[:, np.newaxis]
        
        rgen = np.random.RandomState(self.random_state)
        
        if not self.warm_start or not hasattr(self, 'weights_'):
            self.weights_ = rgen.normal(
                loc=0.0, scale=0.01, size=(X.shape[1] + 1, y.shape[1]))
        
        self.cost_ = []

        if self.early_stopping:
            X, X_val, y, y_val = train_test_split(
                X, y, test_size=self.validation_fraction, 
                random_state=self.random_state)
         
        if self.verbose > 0:
            progress_bar = tqdm(
                total=self.max_iter, ascii=True, ncols=100,
                desc=f'Fitting {self.__class__.__name__}', 
            )

        for i in range(self.max_iter):
            if self.shuffle:
                X, y = shuffle(X, y, random_state=self.random_state)
            cost = []
            for xi, target in zip(X, y):
                for idx in range(self.weights_.shape[1]):
                    error = target[idx] - self.net_activation(xi, idx)
                    self.weights_[1:, idx] += self.eta0 * xi * error
                    self.weights_[0, idx] += self.eta0 * error
                    # Prevent overflow in weight updates using epsilon
                    self.weights_ = np.clip(
                        self.weights_, -1/self.epsilon, 1/self.epsilon)
                    cost.append(error ** 2 / 2.0) 
            self.cost_.append(np.mean(cost))
            if self.early_stopping:
                y_val_pred = self.predict(X_val).reshape(-1, 1)
                val_error = np.mean((y_val - y_val_pred) ** 2)
                if val_error < self.tol:
                    if self.verbose > 1:
                        print(f'Early stopping at epoch {i+1}')
                        # Force the progress bar to complete
                        progress_bar.n = self.max_iter  
                        progress_bar.last_print_n = self.max_iter
                        # Refresh the progress bar display
                        progress_bar.update(0)  
                    break

            if self.learning_rate == 'adaptive':
                self.eta0 *= self.eta0_decay
            
            if self.verbose > 1:
                progress_bar.update(1)
                
        if self.verbose > 0:
            progress_bar.close()

        return self

    def predict(self, X):
        """
        Predict class labels for input samples.
    
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input data matrix. Accepts both dense arrays and CSR/CSC sparse 
            matrices.
    
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels:
            - Binary classification: 0 or 1 based on sigmoid threshold
            - Multiclass: Class with highest activation score
    
        Notes
        -----
        Decision process:
        - Binary case: Thresholds activation output at 0.5
        - Multiclass: Selects class with maximum activation using argmax
    
        Uses inverse transform of :attr:`label_binarizer_` to convert 
        binary/multiclass outputs to original label space.
    
        Examples
        --------
        >>> y_pred = clf.predict(X_test)
        >>> accuracy = (y_pred == y_test).mean()
        """
        check_is_fitted(self, 'weights_')
        X = check_array(X, accept_large_sparse=True, accept_sparse=True)
    
        if len(self.label_binarizer_.classes_) == 2:
            return self.label_binarizer_.inverse_transform(
                np.where(self._activator(self.net_activation(X, 0)) >= 0.0, 1, 0))
        else:
            activations = np.column_stack(
                [self.net_activation(X, idx) for idx in range(self.weights_.shape[1])])
            indices = np.argmax(activations, axis=1)
            binary_arr = np.zeros_like(activations, dtype=int)
            binary_arr[np.arange(activations.shape[0]), indices] = 1
            return self.label_binarizer_.inverse_transform(binary_arr)
        
    def predict_proba(self, X):
        """
        Estimate class probability estimates.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input data matrix. Format should match training data (dense/sparse).

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probability estimates:
            - Binary: [1-p, p] where p = σ(z)
            - Multiclass: Softmax-normalized activations

        Notes
        -----
        Probability derivation:
        - Binary: Uses sigmoid activation :math:`\sigma(z) = 1/(1+e^{-z})`
        - Multiclass: Applies softmax normalization across classes

        For custom activation functions, ensure outputs are in [0,1] range 
        and sum to 1 for multiclass cases.

        Examples
        --------
        >>> proba = clf.predict_proba(X_test)
        >>> print("Class confidence scores:", proba[:3])
        """
        check_is_fitted(self, 'weights_')
        X = check_array(X, accept_large_sparse=True, accept_sparse=True)
        if len(self.label_binarizer_.classes_) == 2:
            proba = self._activator(self.net_activation(X, 0))
            return np.vstack([1 - proba, proba]).T
        else:
            proba = [self._activator(self.net_activation(X, idx)
                                     ) for idx in range(
                self.weights_.shape[1])]
            return np.column_stack(proba)
    
class AdalineRegressor(BaseAdaline, RegressorMixin):
    r"""
    Adaline (Adaptive Linear Neuron) Regressor with Batch Gradient Descent.

    Implements linear regression through batch gradient descent optimization, 
    minimizing the mean squared error (MSE) between predictions and targets. 
    Suitable for datasets where linear relationships exist between features 
    and continuous targets.

    The weight update rule per epoch is:

    .. math:: 
        \mathbf{w} := \mathbf{w} + \eta_0 \sum_{i=1}^n\\
            (y^{(i)} - \phi(z^{(i)})) \mathbf{x}^{(i)}

    where:
        - :math:`\mathbf{w}`: Weight vector (including bias term)
        - :math:`\eta_0`: Learning rate controlling update magnitude
        - :math:`y^{(i)}`: True value for i-th sample
        - :math:`\phi(z^{(i)})`: Linear prediction 
          :math:`\mathbf{x}^{(i)} \cdot \mathbf{w}`
        - :math:`\mathbf{x}^{(i)}`: Feature vector of i-th sample

    Parameters
    ----------
    eta0 : float, default=0.01
        Learning rate (0 < eta0 <= 1). Controls gradient descent step size. 
        Higher values accelerate convergence but risk overshooting minima. 
        Typical range: 0.0001-0.1.

    max_iter : int, default=1000
        Maximum training epochs (full dataset passes). Acts as stopping 
        criterion if convergence not achieved earlier.

    early_stopping : bool, default=False
        Enable early termination when validation loss plateaus. Requires 
        ``validation_fraction`` >0 for validation set creation.

    validation_fraction : float, default=0.1
        Proportion of training data (0-1) reserved for validation when 
        ``early_stopping=True``. Balances training data utilization and 
        validation reliability.

    tol : float, default=1e-4
        Minimum absolute improvement threshold for early stopping. Training 
        halts if validation MSE reduction < ``tol`` for two consecutive 
        epochs.

    warm_start : bool, default=False
        Reuse weights from previous ``fit`` calls for incremental training. 
        When False, reinitializes weights at each fit call.

    shuffle : bool, default=True
        Shuffle training data before each epoch. Breaks sample order 
        dependencies and improves convergence stability.

    random_state : int, default=None
        Seed for reproducible weight initialization and data shuffling. 
        Ensures deterministic behavior across runs when set.

    verbose : int, default=0
        Verbosity levels (0-7):
        - 0: Silent mode
        - 1: Progress bar display
        - 2: Epoch-level metrics
        - >=3: Gradient/weight debugging

    epsilon : float, default=1e-8
        Numerical stability constant. Prevents overflow by clipping weights 
        to :math:`[-1/\epsilon, 1/\epsilon]`.

    Attributes
    ----------
    weights_ : ndarray of shape (n_features + 1,)
        Learned weight vector with bias term as first element. Initialized 
        via normal distribution (:math:`\mu=0, \sigma=0.01`).

    cost_ : list of float
        Mean squared error values per epoch. Tracks training progress and 
        convergence behavior.

    Methods
    -------
    fit(X, y, sample_weight=None)
        Train model on feature matrix X and target vector y

    predict(X)
        Generate predictions for input samples X

    Examples
    --------
    >>> from gofast.estimators.adaline import AdalineRegressor
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.preprocessing import StandardScaler
    >>> X, y = load_diabetes(return_X_y=True)
    >>> X = StandardScaler().fit_transform(X)
    >>> reg = AdalineRegressor(eta0=0.01, max_iter=1000).fit(X, y)
    >>> predictions = reg.predict(X[:5])

    Notes
    -----
    - **Feature Scaling**: Critical for performance - center and scale 
      features to zero mean/unit variance using ``StandardScaler``
    - **Convergence Monitoring**: Plot ``cost_`` to diagnose issues - flat 
      curves suggest low ``eta0``, oscillations indicate high ``eta0``
    - **Early Stopping**: Use with ``validation_fraction=0.1-0.2`` to 
      prevent overfitting

    See Also
    --------
    SGDRegressor : Stochastic gradient descent regression
    LinearRegression : Ordinary least squares regression
    Ridge : L2-regularized linear regression

    References
    ----------
    .. [1] Widrow, B. & Hoff, M.E. (1960). Adaptive Switching Circuits. 
       IRE WESCON Convention Record, 96-104.
    .. [2] Bishop, C.M. (2006). Pattern Recognition and Machine Learning. 
       Springer.
    """

    def __init__(
        self, 
        eta0=0.01, 
        max_iter=1000, 
        early_stopping=False, 
        validation_fraction=0.1, 
        tol=1e-4, 
        warm_start=False, 
        shuffle=True, 
        random_state=None,  
        epsilon=1e-8, 
        verbose=0
        ):
        super().__init__(
            eta0=eta0, 
            max_iter=max_iter, 
            early_stopping=early_stopping, 
            validation_fraction=validation_fraction, 
            tol=tol, 
            warm_start=warm_start, 
            shuffle=shuffle, 
            random_state=random_state, 
            verbose=verbose, 
            epsilon=epsilon
        )

    def fit(self, X, y, sample_weight=None):
        """
        Fit training data.
    
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.
    
        y : array-like, shape (n_samples,)
            Target values.
    
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. Currently, this parameter is not used by the method 
            but is included for API consistency.
    
        Returns
        -------
        self : object
            Returns self.
    
        Notes
        -----
        This method fits the Adaline Gradient Descent Regressor to the training 
        data `X` and target values `y`. The model weights are updated incrementally 
        for each training instance based on the learning rate `eta0`.
    
        The weight update rule in batch gradient descent for Adaline is given by:
    
        .. math::
            w := w + \eta0 \sum_{i} (y^{(i)} - \phi(z^{(i)})) x^{(i)}
    
        where:
        - :math:`w` is the weight vector.
        - :math:`\eta0` is the learning rate.
        - :math:`y^{(i)}` is the actual observed value for the \(i\)-th sample.
        - :math:`\phi(z^{(i)})` is the predicted value obtained from the linear model.
        - :math:`x^{(i)}` is the feature vector of the \(i\)-th sample.
    
        If `early_stopping` is enabled, a portion of the training data is set aside 
        as a validation set. Training stops if the validation error does not improve 
        by at least `tol` for two consecutive epochs.
    
        Examples
        --------
        >>> from gofast.estimators.adaline import AdalineRegressor
        >>> import numpy as np
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.preprocessing import StandardScaler
    
        >>> # Load the diabetes dataset
        >>> diabetes = load_diabetes()
        >>> X, y = diabetes.data, diabetes.target
        >>> X_train, X_test, y_train, y_test = train_test_split(
        >>>     X, y, test_size=0.3, random_state=0)
    
        >>> # Standardize features
        >>> sc = StandardScaler()
        >>> X_train_std = sc.fit_transform(X_train)
        >>> X_test_std = sc.transform(X_test)
    
        >>> # Initialize and fit the regressor
        >>> ada_reg = AdalineRegressor(
        >>>     eta0=0.01, max_iter=1000, early_stopping=True, 
        >>>     validation_fraction=0.1, tol=1e-4, verbose=True)
        >>> ada_reg.fit(X_train_std, y_train)
    
        >>> # Predict continuous values
        >>> y_pred = ada_reg.predict(X_test_std)
        >>> print('Mean Squared Error:', np.mean((y_pred - y_test) ** 2))
    
        See Also
        --------
        sklearn.linear_model.SGDRegressor : 
            Linear model fitted by minimizing a regularized empirical loss with SGD.
        sklearn.linear_model.LinearRegression : 
            Ordinary least squares Linear Regression.
        sklearn.linear_model.Ridge : Ridge regression.
    
        References
        ----------
        .. [1] Widrow, B., Hoff, M.E., 1960. Adaptive switching circuits. IRE WESCON 
               Convention Record, New York, 96-104.
        """
        self._validate_params() 
        check_params = dict(
            estimator=self, 
            accept_sparse="csc", 
            ensure_2d=False, dtype=None, 
            multi_output=True
            )
        X, y = self._validate_data(X, y, **check_params)

        if not hasattr(self, 'weights_'):
            self._initialize_weights(X.shape[1], 1)

        self.cost_ = []

        if self.early_stopping:
            X, X_val, y, y_val = train_test_split(
                X, y, test_size=self.validation_fraction, 
                random_state=self.random_state
        )

        if self.verbose >= 1:
            progress_bar = tqdm(
                total=self.max_iter, ascii=True, ncols=100,
                desc=f'Fitting {self.__class__.__name__}', 
            )

        for i in range(self.max_iter):
            if self.shuffle:
                X, y = shuffle(X, y, random_state=self.random_state)
            cost = []
            for xi, target in zip(X, y):
                error = self._update_weights(xi, target, 0)  
                cost.append(error ** 2 / 2.0)
            self.cost_.append(np.mean(cost))

            if self.early_stopping:
                y_val_pred = self.predict(X_val)
                val_error = np.mean((y_val - y_val_pred) ** 2)
                if val_error < self.tol:
                    if self.verbose >= 2:
                        print(f'Early stopping at epoch {i+1}')
                        progress_bar.update(self.max_iter - i)
                    break

            if self.verbose >= 3:
                print(f'Epoch {i+1}/{self.max_iter}, Cost: {self.cost_[-1]}')

            if self.verbose >= 1:
                progress_bar.update(1)

        if self.verbose >= 1:
            progress_bar.close()

        return self


    def predict(self, X):
        """
        Return continuous output.
    
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Input data, where `n_samples` is the number of samples and
            `n_features` is the number of features. Can be a dense array 
            or sparse matrix.
    
        Returns
        -------
        y_pred : array, shape (n_samples,)
            Predicted continuous values.
    
        Notes
        -----
        This method predicts the continuous output values for the input data `X` 
        using the weights learned during training. The prediction is based on 
        calculating the net input:
    
        .. math::
            z = X \cdot w + b
    
        where:
        - :math:`X` is the input feature matrix.
        - :math:`w` is the weight vector.
        - :math:`b` is the bias term.
    
        The predicted value :math:`z` is the linear combination of the input 
        features and the learned weights.
    
        Examples
        --------
        >>> from gofast.estimators.adaline import AdalineRegressor
        >>> import numpy as np
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.preprocessing import StandardScaler
    
        >>> # Load the diabetes dataset
        >>> diabetes = load_diabetes()
        >>> X, y = diabetes.data, diabetes.target
        >>> X_train, X_test, y_train, y_test = train_test_split(
        >>>     X, y, test_size=0.3, random_state=0)
    
        >>> # Standardize features
        >>> sc = StandardScaler()
        >>> X_train_std = sc.fit_transform(X_train)
        >>> X_test_std = sc.transform(X_test)
    
        >>> # Initialize and fit the regressor
        >>> ada_reg = AdalineRegressor(eta0=0.01, max_iter=1000)
        >>> ada_reg.fit(X_train_std, y_train)
    
        >>> # Predict continuous values
        >>> y_pred = ada_reg.predict(X_test_std)
        >>> print('Predicted values:', y_pred)
    
        See Also
        --------
        sklearn.linear_model.SGDRegressor : 
            Linear model fitted by minimizing a regularized empirical loss with SGD.
        sklearn.linear_model.LinearRegression : 
            Ordinary least squares Linear Regression.
        sklearn.linear_model.Ridge : Ridge regression.
    
        References
        ----------
        .. [1] Widrow, B., Hoff, M.E., 1960. Adaptive switching circuits. IRE WESCON 
               Convention Record, New York, 96-104.
        """
        check_is_fitted(self, 'weights_')
        X = check_array(X, accept_large_sparse=True, accept_sparse=True)
        return self.net_input(X, 0)
       
class AdalineClassifier(BaseAdaline, ClassifierMixin):
    r"""
    Adaline (Adaptive Linear Neuron) Classifier with Batch Gradient Descent.

    Implements binary and multiclass classification through batch gradient 
    descent optimization, minimizing mean squared error between predictions 
    and targets. Uses one-vs-rest strategy for multiclass problems.

    The weight update rule per epoch is:

    .. math:: 
        \mathbf{w}_k := \mathbf{w}_k + \eta_0 \sum_{i=1}^n\\
            (y^{(i)}_k - \phi(z^{(i)}_k)) \mathbf{x}^{(i)}

    where:
        - :math:`\mathbf{w}_k`: Weight vector for class k
        - :math:`\eta_0`: Learning rate controlling update magnitude
        - :math:`y^{(i)}_k`: Binary indicator for class k membership
        - :math:`\phi(z^{(i)}_k)`: Activation function output
        - :math:`\mathbf{x}^{(i)}`: Feature vector of i-th sample

    Parameters
    ----------
    eta0 : float, default=0.01
        Learning rate (0 < eta0 <= 1). Controls gradient descent step size. 
        Higher values increase convergence speed but risk overshooting 
        minima. Typical range: 0.001-0.1.

    max_iter : int, default=1000
        Maximum training epochs (full dataset passes). Acts as fail-safe 
        stopping criterion if convergence not achieved earlier.

    early_stopping : bool, default=False
        Enable early termination when validation accuracy plateaus. 
        Requires ``validation_fraction`` >0 for validation set creation.

    validation_fraction : float, default=0.1
        Proportion of training data (0-1) reserved for validation when 
        ``early_stopping=True``. Larger values reduce training data but 
        improve validation reliability.

    tol : float, default=1e-4
        Minimum absolute accuracy improvement threshold for early 
        stopping. Training stops if validation accuracy improvement 
        < ``tol`` for two consecutive epochs.

    warm_start : bool, default=False
        When True, retains weights from previous ``fit`` calls for 
        incremental training. When False, reinitializes weights.

    shuffle : bool, default=True
        Shuffle training data before each epoch. Breaks sample order 
        patterns and improves convergence stability.

    random_state : int, default=None
        Seed for reproducible weight initialization and data shuffling. 
        Ensures deterministic behavior across runs when set.

    verbose : int, default=0
        Verbosity levels (0-3):
        - 0: Silent mode
        - 1: Progress bar display
        - 2: Epoch-level metrics
        - 3: Gradient/weight debugging

    Attributes
    ----------
    weights_ : ndarray of shape (n_features + 1, n_classes)
        Learned weight matrix with bias terms. Each column represents 
        weights for one class in one-vs-rest configuration.

    cost_ : list of float
        Mean squared error values per epoch. Tracks training progress 
        and convergence.

    label_binarizer_ : LabelBinarizer
        Internal label encoder handling multiclass target conversions.

    Methods
    -------
    fit(X, y)
        Train model on feature matrix X and target labels y

    predict(X)
        Predict class labels for input samples X

    predict_proba(X)
        Estimate class probabilities using sigmoid/softmax activations

    Examples
    --------
    >>> from gofast.estimators.adaline import AdalineClassifier
    >>> from sklearn.datasets import load_iris
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = AdalineClassifier(eta0=0.01, max_iter=500).fit(X, y)
    >>> predictions = clf.predict(X[:5])

    Notes
    -----
    - **Binary Classification**: Expects 0/1 labels, uses sigmoid activation
    - **Multiclass Handling**: Implements one-vs-rest strategy with 
      argmax prediction
    - **Feature Scaling**: Critical for performance - standardize features 
      using ``StandardScaler``
    - **Convergence Monitoring**: Plot ``cost_`` to diagnose learning 
      issues - flat curves suggest low ``eta0``

    See Also
    --------
    LogisticRegression : Linear model for logistic classification
    SGDClassifier : Stochastic gradient descent classifier
    Perceptron : Simple threshold-based linear classifier

    References
    ----------
    .. [1] Widrow, B. & Hoff, M.E. (1960). Adaptive Switching Circuits. 
       IRE WESCON Convention Record, 96-104.
    .. [2] Bishop, C.M. (2006). Pattern Recognition and Machine Learning. 
       Springer.
    """

    def __init__(
        self, 
        eta0=0.01,
        max_iter=1000, 
        early_stopping=False,
        validation_fraction=0.1, 
        tol=1e-4, 
        warm_start=False,
        shuffle=True,
        random_state=None, 
        epsilon=1e-8, 
        verbose=0
        ):
        super().__init__(
            eta0=eta0, 
            max_iter=max_iter, 
            early_stopping=early_stopping, 
            validation_fraction=validation_fraction, 
            tol=tol, 
            warm_start=warm_start, 
            shuffle=shuffle, 
            random_state=random_state,
            epsilon=epsilon, 
            verbose=verbose
        )


    def fit(self, X, y, sample_weight=None):
        """
        Fit training data.
    
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.
    
        y : array-like, shape (n_samples,)
            Target values.
    
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. Currently, this parameter is not used by the method 
            but is included for API consistency.
    
        Returns
        -------
        self : object
            Returns self.
    
        Notes
        -----
        This method fits the Adaline Gradient Descent Classifier to the training 
        data `X` and target values `y`. The model weights are updated incrementally 
        for each training instance based on the learning rate `eta0`.
    
        The weight update rule in gradient descent for Adaline is given by:
    
        .. math::
            w := w + \eta0 \sum_{i} (y^{(i)} - \phi(z^{(i)})) x^{(i)}
    
        where:
        - :math:`w` is the weight vector.
        - :math:`\eta0` is the learning rate.
        - :math:`y^{(i)}` is the actual label for the \(i\)-th sample.
        - :math:`\phi(z^{(i)})` is the predicted label, obtained from the linear 
          model activation.
        - :math:`x^{(i)}` is the feature vector of the \(i\)-th sample.
    
        If `early_stopping` is enabled, a portion of the training data is set aside 
        as a validation set. Training stops if the validation error does not improve 
        by at least `tol` for two consecutive epochs.
    
        Examples
        --------
        >>> from gofast.estimators.adaline import AdalineClassifier
        >>> import numpy as np
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.preprocessing import StandardScaler
    
        >>> # Load the iris dataset
        >>> iris = load_iris()
        >>> X = iris.data[:, :2]  # we only take the first two features.
        >>> y = (iris.target != 0) * 1  # binary classification: Iris Setosa vs. others
        >>> X_train, X_test, y_train, y_test = train_test_split(
        >>>     X, y, test_size=0.3, random_state=0)
    
        >>> # Standardize features
        >>> sc = StandardScaler()
        >>> X_train_std = sc.fit_transform(X_train)
        >>> X_test_std = sc.transform(X_test)
    
        >>> # Initialize and fit the classifier
        >>> ada_clf = AdalineClassifier(
        >>>     eta0=0.01, max_iter=1000, early_stopping=True, 
        >>>     validation_fraction=0.1, tol=1e-4, verbose=True)
        >>> ada_clf.fit(X_train_std, y_train)
    
        >>> # Predict class labels
        >>> y_pred = ada_clf.predict(X_test_std)
        >>> print('Accuracy:', np.mean(y_pred == y_test))
    
        See Also
        --------
        LogisticRegression : Logistic Regression classifier from Scikit-Learn.
        SGDClassifier : Linear model fitted by minimizing a regularized empirical 
                        loss with SGD from Scikit-Learn.
    
        References
        ----------
        .. [1] Widrow, B., Hoff, M.E., 1960. Adaptive switching circuits. IRE WESCON 
               Convention Record, New York, 96-104.
        """
        self._validate_params() 
        check_params = dict(
            estimator=self, 
            accept_sparse="csc", 
            ensure_2d=True, 
            dtype=None, 
            multi_output=True
            )
        X, y = self._validate_data(X, y, **check_params)

        self.label_binarizer_ = LabelBinarizer()
        y = self.label_binarizer_.fit_transform(y)
        
        if y.ndim == 1:
            y = y[:, np.newaxis]
        
        rgen = np.random.RandomState(self.random_state)

        if not self.warm_start or not hasattr(self, 'weights_'):
            self.weights_ = rgen.normal(
                loc=0.0, scale=0.01, size=(X.shape[1] + 1, y.shape[1])
                )
        
        self.errors_ = []

        if self.early_stopping:
            X, X_val, y, y_val = train_test_split(
                X, y, test_size=self.validation_fraction,
                random_state=self.random_state)

        if self.verbose > 0:
            progress_bar = tqdm(
                total=self.max_iter, ascii=True, ncols=100,
                desc=f'Fitting {self.__class__.__name__}', 
            )

        for i in range(self.max_iter):
            if self.shuffle:
                X, y = shuffle(X, y, random_state=self.random_state)
            cost = []
            for xi, target in zip(X, y):
                for idx in range(self.weights_.shape[1]):
                    error = target[idx] - self.activation(xi, idx)
                    # Update weights with a clipped gradient to prevent overflow
                    self.weights_[1:, idx] += np.clip(
                        self.eta0 * xi * error, -self.epsilon, self.epsilon
                        )
                    self.weights_[0, idx] += np.clip(
                        self.eta0 * error, -self.epsilon, self.epsilon)
                    cost.append(error ** 2 / 2.0)
                    
                    # error = target[idx] - self.activation(xi, idx)
                    # self.weights_[1:, idx] += self.eta0 * xi * error
                    # self.weights_[0, idx] += self.eta0 * error
                    # cost.append(error ** 2 / 2.0)
            self.errors_.append(np.mean(cost))
            
            if self.early_stopping:
                y_val_pred = self.predict(X_val)
                val_error = np.mean((y_val - y_val_pred) ** 2)
                if val_error < self.tol:
                    if self.verbose:
                        print(f'Early stopping at epoch {i+1}')
                        progress_bar.update(self.max_iter - i )
                    break

            if self.verbose:
                progress_bar.update(1)

        if self.verbose:
            progress_bar.close()

        return self

    def predict(self, X):
        """
        Predict class label after unit step.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
    
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted class labels for the input samples.
    
        Notes
        -----
        This method predicts the class labels for the input data `X` by 
        calculating the net input and applying the unit step function. The 
        unit step function maps the net input to class labels:
    
        .. math::
            y^{\text{pred}} = 
            \begin{cases}
            1 & \text{if } \phi(z) \geq 0 \\
            -1 & \text{if } \phi(z) < 0
            \end{cases}
    
        where :math:`\phi(z)` is the activation function (net input).
    
        Examples
        --------
        >>> from gofast.estimators.adaline import AdalineClassifier
        >>> import numpy as np
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.preprocessing import StandardScaler
    
        >>> # Load the iris dataset
        >>> iris = load_iris()
        >>> X = iris.data[:, :2]  # we only take the first two features.
        >>> y = (iris.target != 0) * 1  # binary classification: Iris Setosa vs. others
        >>> X_train, X_test, y_train, y_test = train_test_split(
        >>>     X, y, test_size=0.3, random_state=0)
    
        >>> # Standardize features
        >>> sc = StandardScaler()
        >>> X_train_std = sc.fit_transform(X_train)
        >>> X_test_std = sc.transform(X_test)
    
        >>> # Initialize and fit the classifier
        >>> ada_clf = AdalineClassifier(eta0=0.01, max_iter=1000)
        >>> ada_clf.fit(X_train_std, y_train)
    
        >>> # Predict class labels
        >>> y_pred = ada_clf.predict(X_test_std)
        >>> print('Predicted class labels:', y_pred)
    
        See Also
        --------
        LogisticRegression : Logistic Regression classifier from Scikit-Learn.
        SGDClassifier : Linear model fitted by minimizing a regularized empirical 
                        loss with SGD from Scikit-Learn.
    
        References
        ----------
        .. [1] Widrow, B., Hoff, M.E., 1960. Adaptive switching circuits. IRE WESCON 
               Convention Record, New York, 96-104.
        """
        check_is_fitted(self, 'weights_')
        X = check_array(X, accept_large_sparse=True, accept_sparse=True)

        if len(self.label_binarizer_.classes_) == 2:
            return self.label_binarizer_.inverse_transform(
                np.where(self.activation(X, 0) >= 0.0, 1, 0))
        else:
            activations = np.column_stack(
                [self.activation(X, idx) for idx in range(self.weights_.shape[1])])
            indices = np.argmax(activations, axis=1)
            binary_arr = np.zeros_like(activations, dtype=int)
            binary_arr[np.arange(activations.shape[0]), indices] = 1
            return self.label_binarizer_.inverse_transform(binary_arr)
    
    def predict_proba(self, X):
        """
        Predict class probabilities using the Adaline Classifier model.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
    
        Returns
        -------
        proba : array-like of shape (n_samples, 2)
            The class probabilities of the input samples.
    
        Notes
        -----
        This method predicts the probability of each class for the input data `X` 
        using the logistic sigmoid function. The sigmoid function maps the net 
        input to probabilities:
    
        .. math::
            P(y=1|x) = \frac{1}{1 + \exp(-\phi(z))}
    
        where :math:`\phi(z)` is the activation function (net input).
    
        The method returns a 2D array where each row corresponds to a sample and 
        each column corresponds to a class probability. The first column contains 
        the probabilities of the negative class and the second column contains the 
        probabilities of the positive class.
    
        Examples
        --------
        >>> from gofast.estimators.adaline import AdalineClassifier
        >>> import numpy as np
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.preprocessing import StandardScaler
    
        >>> # Load the iris dataset
        >>> iris = load_iris()
        >>> X = iris.data[:, :2]  # we only take the first two features.
        >>> y = (iris.target != 0) * 1  # binary classification: Iris Setosa vs. others
        >>> X_train, X_test, y_train, y_test = train_test_split(
        >>>     X, y, test_size=0.3, random_state=0)
    
        >>> # Standardize features
        >>> sc = StandardScaler()
        >>> X_train_std = sc.fit_transform(X_train)
        >>> X_test_std = sc.transform(X_test)
    
        >>> # Initialize and fit the classifier
        >>> ada_clf = AdalineClassifier(eta0=0.01, max_iter=1000)
        >>> ada_clf.fit(X_train_std, y_train)
    
        >>> # Predict class probabilities
        >>> y_proba = ada_clf.predict_proba(X_test_std)
        >>> print('Class probabilities:', y_proba)
    
        See Also
        --------
        LogisticRegression : Logistic Regression classifier from Scikit-Learn.
        SGDClassifier : Linear model fitted by minimizing a regularized empirical 
                        loss with SGD from Scikit-Learn.
    
        References
        ----------
        .. [1] Widrow, B., Hoff, M.E., 1960. Adaptive switching circuits. IRE WESCON 
               Convention Record, New York, 96-104.
        """
        def _sigmoid(z):
            # Use the logistic sigmoid function to estimate probabilities
            return 1 / (1 + np.exp(-z))
 
        check_is_fitted(self, 'weights_')
        X = check_array(X, accept_large_sparse=True, accept_sparse=True)
        # Apply the linear model
        if len(self.label_binarizer_.classes_) == 2:
            proba = _sigmoid(self.activation(X, 0))
            return np.vstack([1 - proba, proba]).T
        else:
            proba = [_sigmoid(self.activation(X, idx)) for idx in range(
                self.weights_.shape[1])]
            return np.column_stack(proba)

    
   
@Deprecated (
    "AdalineMixte should be removed in next realease."
    " Use 'AdalineRegressor' or 'AdalineClassifier' for"
    " regression and classification task respectively." 
)
class AdalineMixte(BaseEstimator, ClassifierMixin, RegressorMixin):
    r"""
    Adaline Mixte for Dual Regression and Classification Tasks.

    The ADAptive LInear NEuron (Adaline) Mixte model is capable of performing
    both regression and classification tasks. This model extends the classical
    Adaline by incorporating both tasks, allowing it to handle various types
    of targets including binary, multiclass, and continuous values.

    Parameters
    ----------
    eta0 : float, default=0.01
        The learning rate, determining the step size at each iteration while
        moving toward a minimum of the loss function. The value should be between
        0.0 and 1.0.

    max_iter : int, default=1000
        The number of passes over the training dataset (epochs).

    early_stopping : bool, default=False
        Whether to stop training early when validation error is not improving.

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for
        early stopping. Only used if `early_stopping` is True.

    tol : float, default=1e-4
        The tolerance for the optimization. Training stops when the validation
        error does not improve by at least `tol` for two consecutive epochs.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to `fit` and
        add more iterations to the estimator, otherwise, just erase the previous
        solution.

    shuffle : bool, default=True
        Whether to shuffle the training data before each epoch. Shuffling helps
        in preventing cycles and ensures that individual samples are encountered
        in different orders.

    random_state : int, default=None
        The seed of the pseudo random number generator for shuffling the data
        and initializing the weights.

    verbose : bool, default=False
        Whether to print progress messages to stdout.

    Attributes
    ----------
    weights_ : array-like of shape (n_features,)
        Weights assigned to the features after fitting the model.

    errors_ : list
        The number of misclassifications (for classification tasks) or the
        sum of squared errors (for regression tasks) in each epoch. This can be
        used to analyze the convergence behavior of the algorithm during training.

    Notes
    -----
    This implementation is suitable for both regression and classification
    tasks. For multi-class classification, the model uses a one-vs-rest strategy.
    Feature scaling (e.g., using `StandardScaler` from `sklearn.preprocessing`)
    is recommended with gradient descent algorithms.
    
    Adaline is critical in demonstrating the importance of defining and 
    minimizing a continuous cost function, paving the way for the development 
    of advanced machine learning algorithms such as Logistic Regression, 
    Support Vector Machines, and various regression models.

    The Widrow-Hoff learning rule, or the Adaline rule, differs from Rosenblatt's 
    perceptron by updating weights based on a linear activation function rather 
    than a binary step function. This approach allows for the use of gradient 
    descent  optimization methods due to the differentiability of the cost 
    function.

    Key Contributions and Innovations:
    - Adaline's primary innovation is its cost function, which is continuous 
      and differentiable. This allows for the application of gradient descent,
      leading to more efficient and robust optimization.
    - The introduction of the Widrow-Hoff learning rule with Adaline minimizes
      the mean squared error between actual outputs and predictions, fostering
      a stable and convergent learning process.


    The weight update rule in the gradient descent step for Adaline is:

    .. math::
        w := w + \eta0 \sum_{i} (y^{(i)} - \phi(w^T x^{(i)})) x^{(i)}

    where:
    - :math:`\eta0` is the learning rate.
    - :math:`y^{(i)}` is the true value or label for the \(i\)-th sample.
    - :math:`\phi(w^T x^{(i)})` is the predicted value, where :math:`\phi`
      is the identity function (i.e., :math:`\phi(z) = z`), reflecting the
      continuous nature of the linear activation function.
    - :math:`x^{(i)}` is the feature vector of the \(i\)-th sample.

    Examples
    --------
    >>> from gofast.estimators.adaline import AdalineMixte
    >>> from sklearn.datasets import load_boston, load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.preprocessing import StandardScaler

    >>> # Example for regression
    >>> boston = load_boston()
    >>> X, y = boston.data, boston.target
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.3, random_state=0)
    >>> sc = StandardScaler()
    >>> X_train_std = sc.fit_transform(X_train)
    >>> X_test_std = sc.transform(X_test)
    >>> model = AdalineMixte(eta0=0.01, max_iter=1000)
    >>> model.fit(X_train_std, y_train)
    >>> y_pred = model.predict(X_test_std)
    >>> print('Mean Squared Error:', np.mean((y_pred - y_test) ** 2))

    >>> # Example for classification
    >>> iris = load_iris()
    >>> X, y = iris.data, iris.target
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.3, random_state=0)
    >>> sc = StandardScaler()
    >>> X_train_std = sc.fit_transform(X_train)
    >>> X_test_std = sc.transform(X_test)
    >>> model = AdalineMixte(eta0=0.01, max_iter=1000)
    >>> model.fit(X_train_std, y_train)
    >>> y_pred = model.predict(X_test_std)
    >>> print('Accuracy:', np.mean(y_pred == y_test))

    See Also
    --------
    LogisticRegression : Logistic Regression classifier for binary
        classification tasks.
    LinearRegression : Ordinary least squares Linear Regression.
    SGDClassifier : Linear models (SVM, logistic regression, etc.) fitted
        by SGD.

    References
    ----------
    .. [1] Widrow, B., Hoff, M.E., 1960. An Adaptive "Adaline" Neuron Using Chemical
           "Memristors". Technical Report 1553-2, Stanford Electron Labs, Stanford, CA,
           October 1960.
    """

    def __init__(
        self, 
        eta0=0.01, 
        max_iter=1000, 
        early_stopping=False,
        validation_fraction=0.1, 
        tol=1e-4, 
        warm_start=False,
        shuffle=True, 
        random_state=None, 
        epsilon=1e-8, 
        verbose=False
        ):
    
        self.eta0 = eta0
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.tol = tol
        self.warm_start = warm_start
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.epsilon =epsilon 
        
    def fit(self, X, y, sample_weight=None):
        """
        Fit training data.
    
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.
    
        y : array-like, shape (n_samples,)
            Target values. For classification tasks, these should be discrete class
            labels. For regression tasks, these should be continuous values.
    
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. Currently, this parameter is not used by the method
            but is included for API consistency.
    
        Returns
        -------
        self : object
            Returns self.
    
        Notes
        -----
        This method fits the Adaline Mixte model to the training data `X` and
        target values `y`. The model can handle both regression and classification
        tasks, and the type of task is determined based on the target values.
    
        For classification tasks, the target values are binarized. The weight update
        rule in gradient descent for Adaline is given by:
    
        .. math::
            w := w + \eta0 \sum_{i} (y^{(i)} - \phi(w^T x^{(i)})) x^{(i)}
    
        where:
        - :math:`\eta0` is the learning rate.
        - :math:`y^{(i)}` is the true value or label for the \(i\)-th sample.
        - :math:`\phi(w^T x^{(i)})` is the predicted value, where :math:`\phi`
          is the identity function (i.e., :math:`\phi(z) = z`), reflecting the
          continuous nature of the linear activation function.
        - :math:`x^{(i)}` is the feature vector of the \(i\)-th sample.
    
        If `early_stopping` is enabled, a portion of the training data is set aside
        as a validation set. Training stops if the validation error does not improve
        by at least `tol` for two consecutive epochs.
    
        Examples
        --------
        >>> from gofast.estimators.adaline import AdalineMixte
        >>> from sklearn.datasets import load_boston, load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.preprocessing import StandardScaler
    
        >>> # Example for regression
        >>> boston = load_boston()
        >>> X, y = boston.data, boston.target
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X, y, test_size=0.3, random_state=0)
        >>> sc = StandardScaler()
        >>> X_train_std = sc.fit_transform(X_train)
        >>> X_test_std = sc.transform(X_test)
        >>> model = AdalineMixte(eta0=0.01, max_iter=1000)
        >>> model.fit(X_train_std, y_train)
        >>> y_pred = model.predict(X_test_std)
        >>> print('Mean Squared Error:', np.mean((y_pred - y_test) ** 2))
    
        >>> # Example for classification
        >>> iris = load_iris()
        >>> X, y = iris.data, iris.target
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X, y, test_size=0.3, random_state=0)
        >>> sc = StandardScaler()
        >>> X_train_std = sc.fit_transform(X_train)
        >>> X_test_std = sc.transform(X_test)
        >>> model = AdalineMixte(eta0=0.01, max_iter=1000)
        >>> model.fit(X_train_std, y_train)
        >>> y_pred = model.predict(X_test_std)
        >>> print('Accuracy:', np.mean(y_pred == y_test))
    
        See Also
        --------
        LogisticRegression : Logistic Regression classifier for binary
            classification tasks.
        LinearRegression : Ordinary least squares Linear Regression.
        SGDClassifier : Linear models (SVM, logistic regression, etc.) fitted
            by SGD.
    
        References
        ----------
        .. [1] Widrow, B., Hoff, M.E., 1960. An Adaptive "Adaline" Neuron Using Chemical
               "Memristors". Technical Report 1553-2, Stanford Electron Labs, Stanford, CA,
               October 1960.
        """

        X, y = check_X_y(X, y, estimator=self, ensure_2d=True, multi_output=True)
        self.task_type = type_of_target(y)
        
        if self.task_type not in [
                "binary", "multiclass", "continuous", "multilabel-indicator"]:
            raise ValueError(f"Unsupported task type: {self.task_type}")
        
        if self.task_type in ["binary", "multiclass"]:
            self.label_binarizer_ = LabelBinarizer()
            y = self.label_binarizer_.fit_transform(y)
            if y.ndim == 1:
                y = y[:, np.newaxis]
        
        rgen = np.random.RandomState(self.random_state)

        if not self.warm_start or not hasattr(self, 'weights_'):
            self.weights_ = rgen.normal(loc=0.0, scale=0.01, size=(
                X.shape[1] + 1, y.shape[1] if self.task_type != "continuous" else 1))
        
        self.errors_ = []

        if self.early_stopping and self.task_type != "continuous":
            X, X_val, y, y_val = train_test_split(
                X, y, test_size=self.validation_fraction,
                random_state=self.random_state
                )

        if self.verbose:
            progress_bar = tqdm(
                total=self.max_iter, ascii=True, ncols=100,
                desc=f'Fitting {self.__class__.__name__}', 
            )

        for i in range(self.max_iter):
            if self.shuffle:
                X, y = shuffle(X, y, random_state=self.random_state)
            cost = []
            for xi, target in zip(X, y):
                if self.task_type == "continuous":
                    error = target - self.activation(xi, 0)
                    self.weights_[1:, 0] += self.eta0 * xi * error
                    self.weights_[0, 0] += self.eta0 * error
                    cost.append(error ** 2 / 2.0)
                else:
                    for idx in range(self.weights_.shape[1]):
                        error = target[idx] - self.activation(xi, idx)
                        self.weights_[1:, idx] += self.eta0 * xi * error
                        self.weights_[0, idx] += self.eta0 * error
                        
                        cost.append(error ** 2 / 2.0)
            self.errors_.append(np.mean(cost))
            if self.early_stopping and self.task_type != "continuous":
                y_val_pred = self.predict(X_val)
                if y_val_pred.ndim == 1:
                    y_val_pred = y_val_pred[:, np.newaxis]
                val_error = np.mean((y_val - y_val_pred) ** 2)
                if val_error < self.tol:
                    if self.verbose:
                        print(f'Early stopping at epoch {i+1}')
                        progress_bar.update(self.max_iter - i )
                    break

            if self.verbose:
                progress_bar.update(1)

        if self.verbose:
            progress_bar.close()

        return self

    def net_input(self, X, idx=0):
        return np.dot(X, self.weights_[1:, idx]) + self.weights_[0, idx]

    def activation(self, X, idx=0):
        return self.net_input(X, idx)
    
    def predict(self, X):
        """
        Predict class labels or continuous values for the input samples.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
    
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels for classification tasks or continuous values
            for regression tasks.
    
        Notes
        -----
        This method predicts the class labels for classification tasks or continuous
        values for regression tasks based on the input samples `X`. For classification,
        the model uses the net input to determine the class labels. For regression,
        the continuous values are predicted directly from the net input.
    
        For classification tasks:
        - If the task is binary classification, the method uses a threshold of 0.0
          on the activation function to determine class labels.
        - If the task is multi-class classification, the method determines the class
          with the highest activation value.
    
        Examples
        --------
        >>> from gofast.estimators.adaline import AdalineMixte
        >>> import numpy as np
        >>> from sklearn.datasets import load_iris, load_boston
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.preprocessing import StandardScaler
    
        >>> # Example for regression
        >>> boston = load_boston()
        >>> X, y = boston.data, boston.target
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X, y, test_size=0.3, random_state=0)
        >>> sc = StandardScaler()
        >>> X_train_std = sc.fit_transform(X_train)
        >>> X_test_std = sc.transform(X_test)
        >>> model = AdalineMixte(eta0=0.01, max_iter=1000)
        >>> model.fit(X_train_std, y_train)
        >>> y_pred = model.predict(X_test_std)
        >>> print('Mean Squared Error:', np.mean((y_pred - y_test) ** 2))
    
        >>> # Example for classification
        >>> iris = load_iris()
        >>> X, y = iris.data, iris.target
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X, y, test_size=0.3, random_state=0)
        >>> sc = StandardScaler()
        >>> X_train_std = sc.fit_transform(X_train)
        >>> X_test_std = sc.transform(X_test)
        >>> model = AdalineMixte(eta0=0.01, max_iter=1000)
        >>> model.fit(X_train_std, y_train)
        >>> y_pred = model.predict(X_test_std)
        >>> print('Accuracy:', np.mean(y_pred == y_test))
    
        See Also
        --------
        sklearn.linear_model.LogisticRegression : Logistic regression classifier.
        sklearn.linear_model.LinearRegression : Linear regression model.
        sklearn.linear_model.SGDClassifier : Linear models (SVM, logistic regression, etc.)
            fitted by SGD.
    
        References
        ----------
        .. [1] Widrow, B., Hoff, M.E., 1960. An Adaptive "Adaline" Neuron Using Chemical
               "Memristors". Technical Report 1553-2, Stanford Electron Labs, Stanford, CA,
               October 1960.
        """
        check_is_fitted(self, 'weights_')
        X = check_array(X, accept_large_sparse=True, accept_sparse=True)
    
        if self.task_type == "continuous":
            return self.net_input(X)
        else:
            if len(self.label_binarizer_.classes_) == 2:
                return self.label_binarizer_.inverse_transform(
                    np.where(self.activation(X, 0) >= 0.0, 1, 0))
            else:
                activations = np.column_stack(
                    [self.activation(X, idx) for idx in range(self.weights_.shape[1])])
                indices = np.argmax(activations, axis=1)
                binary_arr = np.zeros_like(activations, dtype=int)
                binary_arr[np.arange(activations.shape[0]), indices] = 1
                return self.label_binarizer_.inverse_transform(binary_arr)
    
    def predict_proba(self, X):
        """
        Predict class probabilities using the Adaline Mixte model for 
        classification tasks.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
    
        Returns
        -------
        proba : array-like of shape (n_samples, 2)
            The class probabilities of the input samples.
    
        Raises
        ------
        ValueError
            If the model is used for regression tasks.
    
        Notes
        -----
        This method predicts the probability of each class for the input data `X`
        using the logistic sigmoid function. The sigmoid function maps the net
        input to probabilities:
    
        .. math::
            P(y=1|x) = \frac{1}{1 + \exp(-\phi(z))}
    
        where :math:`\phi(z)` is the activation function (net input).
    
        The method returns a 2D array where each row corresponds to a sample and
        each column corresponds to a class probability. The first column contains
        the probabilities of the negative class and the second column contains the
        probabilities of the positive class.
    
        Examples
        --------
        >>> from gofast.estimators.adaline import AdalineMixte
        >>> import numpy as np
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.model_selection import train_test_split
        >>> from sklearn.preprocessing import StandardScaler
    
        >>> # Load the iris dataset
        >>> iris = load_iris()
        >>> X, y = iris.data, iris.target
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X, y, test_size=0.3, random_state=0)
        >>> sc = StandardScaler()
        >>> X_train_std = sc.fit_transform(X_train)
        >>> X_test_std = sc.transform(X_test)
        >>> model = AdalineMixte(eta0=0.01, max_iter=1000)
        >>> model.fit(X_train_std, y_train)
        >>> y_proba = model.predict_proba(X_test_std)
        >>> print('Class probabilities:', y_proba)
    
        See Also
        --------
        sklearn.linear_model.LogisticRegression :
            Logistic regression classifier.
        sklearn.linear_model.SGDClassifier : 
            Linear models (SVM, logistic regression, etc.)
            fitted by SGD.
    
        References
        ----------
        .. [1] Widrow, B., Hoff, M.E., 1960. An Adaptive "Adaline" Neuron Using Chemical
               "Memristors". Technical Report 1553-2, Stanford Electron Labs, Stanford, CA,
               October 1960.
        """
        check_is_fitted(self, 'weights_')
        X = check_array(X, accept_large_sparse=True, accept_sparse=True)
        if self.task_type == "continuous":
            raise ValueError("predict_proba is not supported for regression tasks"
                             " with AdalineMixte. Use AdalineRegresssor instead.")
        net_input = self.net_input(X)
        proba_positive_class = 1 / (1 + np.exp(-net_input))
        proba_negative_class = 1 - proba_positive_class
    
        return np.vstack((proba_negative_class, proba_positive_class)).T

    def score(self, X, y, sample_weight=None):
        """
        Returns the mean accuracy on the given test data and labels for 
        classification tasks, and the coefficient of determination R^2 for 
        regression tasks.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
    
        y : array-like of shape (n_samples,)
            True labels for classification tasks, or continuous target values 
            for regression tasks.
    
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
    
        Returns
        -------
        score : float
            Mean accuracy for classification tasks, and R^2 score for 
            regression tasks.
        """
    
        check_is_fitted(self, 'weights_')
        X, y = check_X_y(X, y, estimator= self ) 
        y_pred = self.predict(X)
        
        if self.task_type != "continuous":
            return accuracy_score(y, y_pred, sample_weight=sample_weight)
        else:
            return r2_score(y, y_pred, sample_weight=sample_weight)
        
    def __repr__(self):
        """
        Provide a string representation of the Perceptron object, displaying 
        its parameters in a formatted manner for better readability.
    
        Returns
        -------
        repr : str
            String representation of the Perceptron object with its parameters.
        """
        params = ",\n    ".join(f"{key}={val}" for key, val in self.get_params().items())
        return f"{self.__class__.__name__}(\n    {params}\n)"
    

