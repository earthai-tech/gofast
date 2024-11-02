# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

from abc import ABCMeta, abstractmethod
from numbers import Integral
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils._param_validation import HasMethods, StrOptions

from ..tools.validator import check_is_fitted
from ..compat.sklearn import Interval 
from ..exceptions import NotFittedError

class BaseHammersteinWiener(BaseEstimator, metaclass=ABCMeta):
    """
    Base class for Hammerstein-Wiener models.

    The Hammerstein-Wiener model is a block-oriented model used to
    represent nonlinear dynamic systems. It consists of a cascade of
    a static nonlinear block, a dynamic linear block, and another
    static nonlinear block. This model is suitable for capturing
    systems with input and output nonlinearities surrounding a linear
    dynamic system.

    The general structure of the Hammerstein-Wiener model is as
    follows:

    .. math::
        y(t) = f_{\\text{out}}\\left( L\\left( f_{\\text{in}}\\left( x(t) \\right) \\right) \\right)

    where:

    - :math:`x(t)` is the input vector at time :math:`t`.
    - :math:`f_{\\text{in}}` is the nonlinear input function.
    - :math:`L` is the linear dynamic block (often represented by an
      autoregressive model).
    - :math:`f_{\\text{out}}` is the nonlinear output function.
    - :math:`y(t)` is the output at time :math:`t`.

    Parameters
    ----------
    nonlinear_input_estimator : estimator object or None, default=None
        Estimator for the nonlinear input function :math:`f_{\\text{in}}`.
        This should be an object that implements ``fit`` and either
        ``transform`` or ``predict`` methods. If ``None``, no nonlinear
        transformation is applied to the input.

    nonlinear_output_estimator : estimator object or None, default=None
        Estimator for the nonlinear output function :math:`f_{\\text{out}}`.
        This should be an object that implements ``fit`` and either
        ``transform`` or ``predict`` methods. If ``None``, no nonlinear
        transformation is applied to the output.

    p : int, default=1
        Order of the linear dynamic block. This specifies the number of
        lagged inputs to include in the linear dynamic model.

    feature_engineering : {'auto'}, default='auto'
        Method for feature engineering. Currently, only ``'auto'`` is
        supported.

    n_jobs : int or None, default=None
        The number of jobs to run in parallel. ``None`` means 1 unless in
        a ``joblib.parallel_backend`` context.

    verbose : int, default=0
        The verbosity level. If greater than 0, prints messages during
        fitting and prediction.

    Attributes
    ----------
    linear_coefficients_ : ndarray of shape (n_features * p,)
        Coefficients of the linear dynamic block.

    See Also
    --------
    HammersteinWienerRegressor : Hammerstein-Wiener model for regression tasks.
    HammersteinWienerClassifier : Hammerstein-Wiener model for classification tasks.

    Notes
    -----
    The Hammerstein-Wiener model combines static nonlinear blocks with
    a linear dynamic block to model complex systems that exhibit both
    dynamic and nonlinear behaviors [1]_.

    References
    ----------
    .. [1] Schoukens, J., & Ljung, L. (2019). Nonlinear System
           Identification: A User-Oriented Roadmap. IEEE Control
           Systems Magazine, 39(6), 28-99.

    Examples
    --------
    >>> from gofast.estimators._dynamic_system import HammersteinWienerRegressor
    >>> # Instantiate the regressor
    >>> model = HammersteinWienerRegressor()
    >>> # Fit the model (example data X and y required)
    >>> model.fit(X, y)
    >>> # Predict
    >>> y_pred = model.predict(X)
    """
    _parameter_constraints: dict = {
        "nonlinear_input_estimator": [
            HasMethods(['fit', 'transform']), HasMethods(['fit', 'predict']), None
        ],
        "nonlinear_output_estimator": [
            HasMethods(['fit', 'transform']), HasMethods(['fit', 'predict']), None
        ],
        "p": [Interval(Integral, 1, None, closed='left')],
        "feature_engineering": [StrOptions({'auto'})],
        "n_jobs": [Integral, None],
        "verbose": [Interval(Integral, 0, None, closed='left'), None],
    }

    @abstractmethod
    def __init__(
        self,
        nonlinear_input_estimator=None,
        nonlinear_output_estimator=None,
        p=1,
        feature_engineering='auto',
        n_jobs=None,
        verbose=0
    ):
        self.nonlinear_input_estimator = nonlinear_input_estimator
        self.nonlinear_output_estimator = nonlinear_output_estimator
        self.p = p
        self.feature_engineering = feature_engineering
        self.n_jobs = n_jobs
        self.verbose = verbose


    def _apply_nonlinear_input(self, X, y=None):
        """Apply the nonlinear input transformation."""
        if self.verbose > 0:
            print("Applying nonlinear input transformation.")
        if self.nonlinear_input_estimator is not None:
            # Check if the estimator is fitted
            try:
                check_is_fitted(self.nonlinear_input_estimator)
            except NotFittedError:
                if self.verbose > 0:
                    print("Fitting nonlinear input estimator.")
                if y is not None:
                    self.nonlinear_input_estimator.fit(X, y)
                else:
                    self.nonlinear_input_estimator.fit(X)
            # Use transform or predict based on estimator type
            if hasattr(self.nonlinear_input_estimator, 'transform'):
                X_transformed = self.nonlinear_input_estimator.transform(X)
            elif hasattr(self.nonlinear_input_estimator, 'predict'):
                X_transformed = self.nonlinear_input_estimator.predict(X)
                # Ensure the output is 2D
                if X_transformed.ndim == 1:
                    X_transformed = X_transformed.reshape(-1, 1)
            else:
                raise ValueError(
                    "The nonlinear_input_estimator must have either a 'transform' "
                    "or 'predict' method."
                )
        else:
            # No nonlinear input estimator provided; use original X
            X_transformed = X
            if self.verbose > 0:
                print("No nonlinear input estimator provided; using original X.")
        return X_transformed

    def _create_lagged_features(self, X_transformed):
        """Create lagged features for the linear dynamic block."""
        if self.verbose > 0:
            print("Creating lagged features.")
        n_samples, n_features = X_transformed.shape
        lagged_features = []
        for lag in range(self.p):
            if self.verbose > 0:
                print(f"Creating lag {lag + 1} features.")
            # Shift the data to create lagged features
            lagged = np.roll(X_transformed, shift=lag, axis=0)
            # Zero padding for initial rows
            lagged[:lag, :] = 0
            lagged_features.append(lagged)
        # Concatenate all lagged features
        X_lagged = np.hstack(lagged_features)
        if self.verbose > 0:
            print(f"Lagged features shape: {X_lagged.shape}")
        return X_lagged


    def _apply_linear_dynamic_block(self, X_lagged):
        """Apply the linear dynamic block."""
        if self.verbose > 0:
            print("Applying linear dynamic block.")
        return X_lagged @ self.linear_coefficients_

    def _apply_nonlinear_output(self, y_linear, y=None):
        """Apply the nonlinear output transformation."""
        if self.verbose > 0:
            print("Applying nonlinear output transformation.")
        if self.nonlinear_output_estimator is not None:
            # Check if the estimator is fitted
            try:
                check_is_fitted(self.nonlinear_output_estimator)
            except NotFittedError:
                if self.verbose > 0:
                    print("Fitting nonlinear output estimator.")
                if y is not None:
                    self.nonlinear_output_estimator.fit(
                        y_linear.reshape(-1, 1), y
                    )
                else:
                    self.nonlinear_output_estimator.fit(
                        y_linear.reshape(-1, 1), y_linear
                    )
            
            # Use transform or predict based on estimator type
            if hasattr(self.nonlinear_output_estimator, 'transform'):
                y_pred = self.nonlinear_output_estimator.transform(
                    y_linear.reshape(-1, 1)
                )
            elif hasattr(self.nonlinear_output_estimator, 'predict'):
                y_pred = self.nonlinear_output_estimator.predict(
                    y_linear.reshape(-1, 1)
                )
            else:
                raise ValueError(
                    "The nonlinear_output_estimator must have either a 'transform' "
                    "or 'predict' method."
                )
            # Ensure the output is 2D
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)
        else:
            # No nonlinear output estimator provided; use linear output
            y_pred = y_linear
            if y_pred.ndim == 1:
                y_pred = y_pred.reshape(-1, 1)
            if self.verbose > 0:
                print("No nonlinear output estimator provided; using linear output.")
        return y_pred
    
    @abstractmethod
    def fit(self, X, y):
        """
        Fit the model to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predict using the Hammerstein-Wiener model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        pass
