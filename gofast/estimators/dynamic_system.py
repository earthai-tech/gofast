# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

"""
Dynamic system implements various dynamic system models for classification 
and regression tasks within the gofast library. These models are designed to 
handle complex, time-dependent data by combining dynamic system theory with 
machine learning techniques.
"""
from numbers import Real
import numpy as np

from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.utils._param_validation import StrOptions
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
try:
    from sklearn.utils.multiclass import type_of_target
except: 
    from ..tools.coreutils import type_of_target
    
from ..compat.sklearn import Interval 
from ..tools.validator import check_is_fitted, check_X_y, check_array
from ._dynamic_system import BaseHammersteinWiener
from .util import activator

__all__= ["HammersteinWienerClassifier","HammersteinWienerRegressor" ]


class HammersteinWienerRegressor(BaseHammersteinWiener, RegressorMixin):
    """
    Hammerstein-Wiener model for regression tasks.

    The Hammerstein-Wiener model is a block-oriented model used to
    represent nonlinear dynamic systems. It consists of a cascade of
    a static nonlinear input block, a linear dynamic block, and a
    static nonlinear output block. This model is suitable for capturing
    systems with input and output nonlinearities surrounding a linear
    dynamic system.

    The general structure of the Hammerstein-Wiener model is as follows:

    .. math::
        y(t) = f_{\\text{out}}\\left( L\\left( f_{\\text{in}}\\left( x(t) \\right) \\right) \\right)

    where:

    - :math:`x(t)` is the input vector at time :math:`t`.
    - :math:`f_{\\text{in}}` is the nonlinear input function.
    - :math:`L` is the linear dynamic block.
    - :math:`f_{\\text{out}}` is the nonlinear output function.
    - :math:`y(t)` is the output at time :math:`t`.

    See more in :ref:`User Guide`.
    
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

    loss : {'mse', 'mae', 'huber', 'time_weighted_mse'}, default='mse'
        Loss function to use for evaluating the model performance.

        - ``'mse'``: Mean Squared Error.
        - ``'mae'``: Mean Absolute Error.
        - ``'huber'``: Huber loss.
        - ``'time_weighted_mse'``: Time-weighted Mean Squared Error, giving
          more importance to recent errors.

    output_scale : tuple or None, default=None
        Tuple specifying the minimum and maximum values to scale the
        output predictions. If ``None``, no scaling is applied.

    time_weighting : {'linear', 'exponential', 'inverse'}, default='linear'
        Method for computing time weights when using time-weighted loss
        functions.

        - ``'linear'``: Linearly increasing weights over time.
        - ``'exponential'``: Exponentially increasing weights over time.
        - ``'inverse'``: Inversely decreasing weights over time.

    feature_engineering : {'auto'}, default='auto'
        Method for feature engineering. Currently, only ``'auto'`` is
        supported.
 
    delta : float, default=1.0
        Threshold parameter for the Huber loss function. It controls the 
        point at which the Huber loss transitions from a quadratic function 
        (for small residuals) to a linear function (for large residuals), 
        thus reducing the impact of outliers. Setting `delta` to a higher 
        value makes the model less sensitive to outliers.

    epsilon : float, default=1e-8
        A small constant added to prevent division by zero or other 
        numerical instabilities in calculations. It ensures stability by 
        limiting values that approach zero, allowing for more robust 
        computations.

    n_jobs : int or None, default=None
        The number of jobs to run in parallel. ``None`` means 1 unless in
        a ``joblib.parallel_backend`` context.

    verbose : int, default=0
        The verbosity level. If greater than 0, prints messages during
        fitting and prediction.

    Attributes
    ----------
    linear_coefficients_ : ndarray of shape (n_features * p,)
        Coefficients of the linear dynamic block. These coefficients 
        represent the linear relationship between the lagged features 
        and the target values and are computed during the fit process.
    
    initial_loss_ : float
        Initial loss computed on the training data after fitting. This 
        value represents the performance of the model immediately after 
        training and provides a baseline loss value.
    
    Methods
    -------
    fit(X, y)
        Fit the Hammerstein-Wiener regressor to the provided training 
        data (`X`, `y`). Applies input transformations, creates lagged 
        features, fits the linear dynamic block, and optionally applies 
        a nonlinear output transformation.
    
    predict(X)
        Predict values for the input samples (`X`) using the fitted 
        Hammerstein-Wiener regressor. Generates predictions by applying 
        transformations and linear dynamics, followed by optional output 
        scaling if specified.

    See Also
    --------
    HammersteinWienerClassifier : Hammerstein-Wiener model for classification tasks.

    Notes
    -----
    The Hammerstein-Wiener model combines static nonlinear blocks with
    a linear dynamic block to model complex systems that exhibit both
    dynamic and nonlinear behaviors [1]_.
    
    `HammersteinWienerRegressor` is especially suited for applications that 
    require detailed analysis and prediction of systems where the output 
    behavior is influenced by historical input and output data. Its ability to 
    model both linear and nonlinear dynamics makes it indispensable in advanced 
    fields like adaptive control, nonlinear system analysis, and complex signal 
    processing, providing insights and predictive capabilities critical for 
    effective system management.

    References
    ----------
    .. [1] Schoukens, J., & Ljung, L. (2019). Nonlinear System
           Identification: A User-Oriented Roadmap. IEEE Control
           Systems Magazine, 39(6), 28-99.

    Examples
    --------
    >>> from gofast.estimators.dynamic_system import HammersteinWienerRegressor
    >>> from sklearn.datasets import make_regression
    >>> import numpy as np
    >>> # Generate synthetic data
    >>> X, y = make_regression(n_samples=200, n_features=5, noise=0.1, random_state=42)
    >>> y += 0.5 * np.sin(X[:, 0])  # Introduce nonlinearity
    >>> # Instantiate the regressor
    >>> model = HammersteinWienerRegressor(p=2, verbose=1)
    >>> # Fit the model
    >>> model.fit(X, y)
    Starting HammersteinWienerRegressor fit method.
    Applying nonlinear input transformation.
    No nonlinear input estimator provided; using original X.
    Creating lagged features.
    Creating lag 1 features.
    Creating lag 2 features.
    Lagged features shape: (200, 10)
    Calculating linear coefficients.
    Applying linear dynamic block.
    Applying nonlinear output transformation.
    No nonlinear output estimator provided; using linear output.
    Starting HammersteinWienerRegressor predict method.
    Applying nonlinear input transformation.
    No nonlinear input estimator provided; using original X.
    Creating lagged features.
    Creating lag 1 features.
    Creating lag 2 features.
    Lagged features shape: (200, 10)
    Applying linear dynamic block.
    Applying nonlinear output transformation.
    No nonlinear output estimator provided; using linear output.
    Computed loss: 94.123456
    Fit method completed.
    >>> # Predict
    >>> y_pred = model.predict(X)
    Starting HammersteinWienerRegressor predict method.
    Applying nonlinear input transformation.
    No nonlinear input estimator provided; using original X.
    Creating lagged features.
    Creating lag 1 features.
    Creating lag 2 features.
    Lagged features shape: (200, 10)
    Applying linear dynamic block.
    Applying nonlinear output transformation.
    No nonlinear output estimator provided; using linear output.
    Predict method completed.
    >>> # Evaluate
    >>> from sklearn.metrics import mean_squared_error
    >>> mse = mean_squared_error(y, y_pred)
    >>> print(f"Mean Squared Error: {mse:.2f}")
    Mean Squared Error: 94.12
    """
    _parameter_constraints: dict = {
        **BaseHammersteinWiener._parameter_constraints,
        "loss": [StrOptions({"mse", "mae", "huber", "time_weighted_mse"})],
        "output_scale": [None, tuple],
        "time_weighting": [StrOptions({"linear", "exponential", "inverse"}), None],
        "delta": [Interval(Real, 0, None, closed='left')],
        "epsilon": [Interval(Real, 1e-15, None, closed='left')],
    }

    def __init__(
            self,
            nonlinear_input_estimator=None,
            nonlinear_output_estimator=None,
            p=1,
            loss="mse",
            output_scale=None,
            time_weighting="linear",
            feature_engineering='auto',
            n_jobs=None,
            delta=1.0,
            epsilon=1e-8,
            verbose=0
    ):
        super().__init__(
            nonlinear_input_estimator=nonlinear_input_estimator,
            nonlinear_output_estimator=nonlinear_output_estimator,
            p=p,
            feature_engineering=feature_engineering,
            n_jobs=n_jobs,
            verbose=verbose
        )
        self.loss = loss
        self.output_scale = output_scale
        self.time_weighting = time_weighting
        self.delta = delta
        self.epsilon = epsilon
        
    def fit(self, X, y):
        """
        Fit the Hammerstein-Wiener regressor to the training data.
    
        This method transforms the input data with a nonlinear function, 
        constructs lagged features to capture temporal dependencies, and 
        then fits a linear model to approximate the relationship between 
        input and target values. If a nonlinear output transformation 
        function is specified, it is applied to the intermediate linear 
        predictions to refine the final model output.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples, where `n_samples` is the number of 
            samples and `n_features` is the number of features in each sample.
    
        y : array-like of shape (n_samples,)
            Target values for regression. These values represent the output 
            associated with each input sample in `X`.
    
        Returns
        -------
        self : object
            Returns the fitted `HammersteinWienerRegressor` instance.
    
        Notes
        -----
        - Nonlinear Input Transformation:
          This transformation allows the model to capture nonlinear effects 
          at the input level by applying a function :math:`f_{NL_{in}}` to 
          each feature vector :math:`x`:
    
          .. math::
              \\tilde{x} = f_{NL_{in}}(x)
    
          where :math:`\\tilde{x}` represents the transformed input feature 
          vector.
    
        - Linear Dynamic Block with Lagged Features:
          Lagged versions of :math:`\\tilde{x}` are generated to capture 
          dynamic behavior over time. The linear relationship is then 
          estimated by solving for :math:`\\beta` in the equation:
    
          .. math::
              y_{linear} = X_{lagged} \\beta + b
    
          where :math:`X_{lagged}` represents the matrix of lagged features, 
          :math:`\\beta` is the vector of coefficients, and :math:`b` is the 
          intercept.
    
        - Nonlinear Output Transformation:
          If a nonlinear output transformation function :math:`f_{NL_{out}}` 
          is provided, it is applied to :math:`y_{linear}` to capture 
          additional nonlinear effects in the output:
    
          .. math::
              \\hat{y} = f_{NL_{out}}(y_{linear})
    
        Notes
        -----
        - This method solves for the linear coefficients using the pseudo-inverse.
        - The `fit` method computes the initial loss to assess the model's 
          performance after training.
    
        Examples
        --------
        >>> from gofast.estimators.dynamic_system import HammersteinWienerRegressor
        >>> model = HammersteinWienerRegressor()
        >>> X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        >>> y = np.array([1.0, 2.0, 3.0])
        >>> model.fit(X, y)
        >>> print("Initial loss:", model.initial_loss_)
    
        See Also
        --------
        predict : Predict values for input data.
    
        References
        ----------
        .. [1] Ljung, L. (1999). System Identification: Theory for the User. 
               Prentice Hall.
        """
        if self.verbose > 0:
            print("Starting HammersteinWienerRegressor fit method.")

        # Validate parameter constraints and input data
        self._validate_params()
        # Allow multi-output y
        X, y = check_X_y(X, y, multi_output=True)

        # Apply nonlinear input transformation to capture non-linear input effects
        X_transformed = self._apply_nonlinear_input(X, y)

        # Create lagged features for the linear dynamic block to capture temporal dependencies
        X_lagged = self._create_lagged_features(X_transformed)

        if self.verbose > 0:
            print("Calculating linear coefficients.")

        # Solve for linear coefficients using pseudo-inverse
        # For multi-output y, linear_coefficients_ will be of shape (n_features_total, n_outputs)
        self.linear_coefficients_ = np.linalg.pinv(X_lagged) @ y

        # Apply the linear dynamic block to get intermediate output
        y_linear = self._apply_linear_dynamic_block(X_lagged)

        # Fit nonlinear output estimator if specified
        self._apply_nonlinear_output(y_linear, y)

        # Compute initial loss for reporting model performance
        y_pred_initial = self.predict(X)
        self.initial_loss_ = self._compute_loss(y, y_pred_initial)

        if self.verbose > 0:
            print(f"Initial loss: {self.initial_loss_}")
            print("Fit method completed.")

        return self
    
    def predict(self, X):
        """
        Predict using the Hammerstein-Wiener regressor.
    
        This method predicts output values by applying the nonlinear input 
        transformation, creating lagged features, and passing them through 
        the linear dynamic block. If a nonlinear output transformation is 
        provided, it is applied to the intermediate predictions, followed 
        by an optional scaling to constrain the output range.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict, where `n_samples` is the number of samples 
            and `n_features` is the number of features in each sample.
    
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values for each input sample, potentially scaled to 
            the specified output range.
    
        Notes
        -----
        - Nonlinear Input Transformation:
          Similar to `fit`, this method applies the transformation 
          :math:`f_{NL_{in}}` to the input:
    
          .. math::
              \\tilde{x} = f_{NL_{in}}(x)
    
        - Lagged Features and Linear Dynamic Block:
          The lagged features are created from :math:`\\tilde{x}` and passed 
          through the linear dynamic block, yielding:
    
          .. math::
              y_{linear} = X_{lagged} \\beta + b
    
        - Nonlinear Output Transformation:
          If provided, the output function :math:`f_{NL_{out}}` is applied to 
          :math:`y_{linear}`, and scaling is optionally applied to ensure 
          predictions fall within a specified range:
    
          .. math::
              \\hat{y} = \\text{scale}(f_{NL_{out}}(y_{linear}))

        - The `predict` method requires that the model is already fitted.
        - Output scaling is applied only if specified in `output_scale`.
    
        Examples
        --------
        >>> from gofast.estimators.dynamic_system import HammersteinWienerRegressor
        >>> model = HammersteinWienerRegressor()
        >>> X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        >>> model.fit(X, y)
        >>> predictions = model.predict(X)
        >>> print("Predictions:", predictions)
    
        See Also
        --------
        fit : Fit the Hammerstein-Wiener regressor to training data.
        """
        if self.verbose > 0:
            print("Starting HammersteinWienerRegressor predict method.")

        # Ensure the model is fitted before making predictions
        check_is_fitted(self, 'linear_coefficients_')

        # Validate input data
        X = check_array(X)

        # Apply nonlinear input transformation to capture nonlinear input relationships
        X_transformed = self._apply_nonlinear_input(X)

        # Create lagged features for the linear dynamic block
        X_lagged = self._create_lagged_features(X_transformed)

        # Apply the linear dynamic block to get intermediate output
        y_linear = self._apply_linear_dynamic_block(X_lagged)

        # Apply nonlinear output transformation to the intermediate predictions
        y_pred = self._apply_nonlinear_output(y_linear)

        # Apply optional scaling to constrain the output range
        y_pred_scaled = self._scale_output(y_pred)

        if self.verbose > 0:
            print("Predict method completed.")

        return y_pred_scaled
    
    def _compute_loss(self, y_true, y_pred):
        """
        Compute the specified loss function between true and predicted values.
    
        This method calculates the average error between the true values 
        (`y_true`) and the predicted values (`y_pred`) using the specified 
        loss function, which can be Mean Squared Error (MSE), Mean Absolute 
        Error (MAE), Huber loss, or a time-weighted MSE. The time-weighted 
        MSE option uses weights that emphasize recent observations.
    
        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True values for each sample. These values are the ground truth 
            for calculating the error between predicted and actual values.
    
        y_pred : array-like of shape (n_samples,)
            Predicted values for each sample as generated by the model.
    
        Returns
        -------
        loss : float
            Computed loss value based on the specified loss function, averaged 
            across all samples.
            
        Notes
        -----
        - Mean Squared Error (MSE):
          The MSE is calculated as the mean of the squared differences between 
          the true and predicted values. It is given by:
    
          .. math::
              L_{MSE} = \\frac{1}{n} \\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2
    
          where :math:`y_i` is the true value and :math:`\\hat{y}_i` is the 
          predicted value for each sample :math:`i`.
    
        - Mean Absolute Error (MAE):
          The MAE is the mean of the absolute differences between true and 
          predicted values:
    
          .. math::
              L_{MAE} = \\frac{1}{n} \\sum_{i=1}^{n} |y_i - \\hat{y}_i|
    
        - Huber Loss:
          Huber loss is less sensitive to outliers than MSE, defined by a 
          threshold :math:`\\delta`:
    
          .. math::
              L_{Huber} = \\frac{1}{n} \\sum_{i=1}^{n} \\begin{cases} 
                            0.5 (y_i - \\hat{y}_i)^2 & \\text{if } |y_i - 
                            \\hat{y}_i| < \\delta \\\\
                            \\delta (|y_i - \\hat{y}_i| - 0.5 \\delta) & 
                            \\text{otherwise}
                         \\end{cases}
    
        - Time-Weighted MSE:
          A weight vector :math:`w_i` is applied to each squared error term 
          to emphasize recent observations:
    
          .. math::
              L_{TW-MSE} = \\frac{1}{n} \\sum_{i=1}^{n} w_i (y_i - \\hat{y}_i)^2
    

        - Huber loss is recommended when outliers are present in the data.
        - Time-weighted MSE is useful in dynamic systems where recent 
          predictions are more important.
    
        Examples
        --------
        >>> from gofast.estimators.dynamic_system import HammersteinWienerRegressor
        >>> model = HammersteinWienerRegressor(loss="huber")
        >>> y_true = np.array([1.0, 2.0, 1.5])
        >>> y_pred = np.array([1.1, 1.9, 1.6])
        >>> loss = model._compute_loss(y_true, y_pred)
        >>> print(f"Computed loss: {loss:.4f}")
    
        See Also
        --------
        _compute_time_weights : Computes weights for time-weighted MSE loss.
        """
        if self.verbose > 0:
            print(f"Computing loss using {self.loss} loss function.")

        # Compute loss based on the specified function
        if self.loss == "mse":
            # Compute mean squared error over all samples and outputs
            loss = np.mean((y_true - y_pred) ** 2)

        elif self.loss == "mae":
            # Compute mean absolute error over all samples and outputs
            loss = np.mean(np.abs(y_true - y_pred))

        elif self.loss == "huber":
            residual = y_true - y_pred
            loss = np.mean(np.where(
                np.abs(residual) < self.delta,
                0.5 * residual ** 2,
                self.delta * (np.abs(residual) - 0.5 * self.delta)
            ))

        elif self.loss == "time_weighted_mse":
            weights = self._compute_time_weights(len(y_true))
            # weights shape: (n_samples,)
            # weights need to be expanded to (n_samples, n_outputs) if y_true is multi-output
            if y_true.ndim > 1:
                weights = weights[:, np.newaxis]  # Shape (n_samples, 1)
            loss = np.mean(weights * (y_true - y_pred) ** 2)

        if self.verbose > 0:
            print(f"Computed loss: {loss}")

        return loss

    def _compute_time_weights(self, n):
        """
        Compute time-based weights for emphasizing recent observations.
    
        This method generates a vector of weights for each sample based on 
        the specified time-weighting method. Time weights allow the model to 
        focus more on recent data points, which is particularly useful in 
        dynamic regression contexts.
    
        Parameters
        ----------
        n : int
            The number of samples for which weights need to be computed.
    
        Returns
        -------
        weights : array-like of shape (n,)
            Time weights for each sample, scaled to sum to 1 for consistent 
            application across various weighting methods.
    
        Notes
        -----
        Time weights are computed as follows depending on the weighting method:
    
        - Linear Weighting:
          Weights increase linearly, with the smallest weight assigned to the 
          earliest sample and the largest to the most recent:
    
          .. math::
              w_i = \\frac{0.1 + \\frac{i}{n - 1}}{1.0}
    
        - Exponential Weighting:
          Exponential weights place even greater importance on recent samples:
    
          .. math::
              w_i = \\frac{e^{\\frac{i}{n}} - 1}{e^{1} - 1}
    
        - Inverse Weighting:
          Weights decrease inversely with the index, with greater emphasis 
          on the most recent samples:
    
          .. math::
              w_i = \\frac{1}{i}
    
        Notes
        -----
        - Weights are normalized to ensure they sum to 1, preserving a stable 
          loss scale across samples.
        - Different weighting methods allow control over the model's temporal 
          sensitivity.
    
        Examples
        --------
        >>> from gofast.estimators.dynamic_system import HammersteinWienerRegressor
        >>> model = HammersteinWienerRegressor(time_weighting="linear")
        >>> weights = model._compute_time_weights(5)
        >>> print("Computed time weights:", weights)
    
        See Also
        --------
        _compute_loss : Uses time weights to compute time-weighted MSE loss.
        """
        if self.verbose > 0:
            print(f"Computing time weights using {self.time_weighting} method.")

        # Compute linear, exponential, or inverse weights based on user input
        if self.time_weighting == "linear":
            weights = np.linspace(0.1, 1.0, n)

        elif self.time_weighting == "exponential":
            weights = np.exp(np.linspace(0, 1, n)) - 1
            weights /= weights.max()  # Normalize to [0, 1]

        elif self.time_weighting == "inverse":
            weights = 1 / np.arange(1, n + 1)
            weights /= weights.max()  # Normalize to [0, 1]

        # Default to equal weights if the method is not recognized
        else:
            # Equal weighting if method is unrecognized i.e. None
            weights = np.ones(n)

        if self.verbose > 0:
            print(f"Time weights: {weights}")

        return weights
   
    def _scale_output(self, y):
        """
        Scale output predictions to a specified range if defined.
    
        This method scales the output predictions to a specified range, 
        which is helpful when model predictions need to be constrained to 
        known limits, as in physical measurements or probability scores.
    
        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Model output predictions to be scaled to the specified range.
    
        Returns
        -------
        y_scaled : array-like of shape (n_samples,)
            Scaled predictions, with values transformed to the specified 
            range if `output_scale` is defined.
    
        Notes
        -----
        Scaling transforms the predicted values to the specified range 
        :math:`[y_{min}, y_{max}]` by first normalizing the predictions 
        to :math:`[0, 1]` and then applying the scale transformation:
    
        .. math::
            y_{scaled} = y_{min} + y * (y_{max} - y_{min})
    
        where :math:`y` represents the normalized prediction values.
    
        - Scaling is applied only if `output_scale` is defined. By default, 
          scaling range is set to None, meaning no scaling is applied.
        - This is especially useful in regression tasks where the output 
          must be within a certain range.
    
        Examples
        --------
        >>> from gofast.estimators.dynamic_system import HammersteinWienerRegressor
        >>> model = HammersteinWienerRegressor(output_scale=(0, 10))
        >>> predictions = np.array([0.2, 0.5, 0.9])
        >>> scaled_output = model._scale_output(predictions)
        >>> print("Scaled output predictions:", scaled_output)
    
        """
        if self.output_scale is not None:
            if self.verbose > 0:
                print("Scaling output predictions.")

            # Apply min-max scaling based on specified output range
            y_min, y_max = self.output_scale

            # Compute min and max per output
            y_min_per_output = y.min(axis=0)
            y_max_per_output = y.max(axis=0)

            # Avoid division by zero
            denom = y_max_per_output - y_min_per_output + self.epsilon

            # Normalize to [0, 1] per output
            y_norm = (y - y_min_per_output) / denom

            # Scale to [y_min, y_max]
            y_scaled = y_norm * (y_max - y_min) + y_min

            if self.verbose > 0:
                print(f"Scaled output range: [{y_min}, {y_max}]")

            return y_scaled
        return y
    
class HammersteinWienerClassifier(BaseHammersteinWiener, ClassifierMixin):
    """
    Hammerstein-Wiener model for classification tasks.

    The Hammerstein-Wiener model is a block-oriented model used to
    represent nonlinear dynamic systems. It consists of a cascade of
    a static nonlinear input block, a linear dynamic block, and a
    static nonlinear output block. This model is suitable for capturing
    systems with input and output nonlinearities surrounding a linear
    dynamic system.

    The general structure of the Hammerstein-Wiener model is as follows:

    .. math::
        y(t) = f_{\\text{out}}\\left( L\\left( f_{\\text{in}}\\left( x(t) \\right) \\right) \\right)

    where:

    - :math:`x(t)` is the input vector at time :math:`t`.
    - :math:`f_{\\text{in}}` is the nonlinear input function.
    - :math:`L` is the linear dynamic block.
    - :math:`f_{\\text{out}}` is the nonlinear output function.
    - :math:`y(t)` is the output at time :math:`t`.

    See more in :ref:`User Guide`.
    
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

    loss : {'cross_entropy', 'time_weighted_cross_entropy'}, default='cross_entropy'
        Loss function to use for evaluating the model performance.

        - ``'cross_entropy'``: Standard cross-entropy loss.
        - ``'time_weighted_cross_entropy'``: Time-weighted cross-entropy loss,
          giving more importance to recent misclassifications.

    time_weighting : {'linear', 'exponential', 'inverse'}, default='linear'
        Method for computing time weights when using time-weighted loss
        functions.

        - ``'linear'``: Linearly increasing weights over time.
        - ``'exponential'``: Exponentially increasing weights over time.
        - ``'inverse'``: Inversely decreasing weights over time.

    feature_engineering : {'auto'}, default='auto'
        Method for feature engineering. Currently, only ``'auto'`` is
        supported.

    epsilon : float, default=1e-15
        A small constant used to prevent division by zero or log of zero 
        errors during calculations  in probability estimates. Clipping 
        predictions within the range defined by `epsilon` ensures numerical
        stability by constraining values to avoid extremes.

    n_jobs : int or None, default=None
        The number of jobs to run in parallel. ``None`` means 1 unless in
        a ``joblib.parallel_backend`` context.
        
        
    verbose : int, default=0
        The verbosity level. If greater than 0, prints messages during
        fitting and prediction.

    Attributes
    ----------
    linear_model_ : object
        The linear model used in the linear dynamic block. This model 
        captures the linear relationship between lagged features and 
        the target classes and is fitted during the training process.
    
    initial_loss_ : float
        Initial loss computed on the training data after fitting. This 
        value indicates the model's performance on the training data 
        immediately following training and serves as a baseline loss.
    
    Methods
    -------
    fit(X, y)
        Fit the Hammerstein-Wiener classifier to the provided training 
        data (`X`, `y`). Applies input transformations, creates lagged 
        features, fits the linear dynamic block with a classification 
        model, and optionally applies a nonlinear output transformation.
    
    predict_proba(X)
        Predict class probabilities for the input samples (`X`) using 
        the fitted Hammerstein-Wiener classifier. Outputs the probability 
        for each class based on transformations and linear dynamics.
    
    predict(X)
        Predict binary class labels for the input samples (`X`) using 
        the fitted Hammerstein-Wiener classifier. Classifies samples 
        based on the computed class probabilities.
    

    See Also
    --------
    HammersteinWienerRegressor : Hammerstein-Wiener model for regression tasks.

    Notes
    -----
    The Hammerstein-Wiener model combines static nonlinear blocks with
    a linear dynamic block to model complex systems that exhibit both
    dynamic and nonlinear behaviors [1]_.

    The classifier is especially suited for time-series classification, 
    signal processing, and systems identification tasks where current outputs 
    are significantly influenced by historical inputs. It finds extensive 
    applications in fields like telecommunications, control systems, and 
    financial modeling, where understanding dynamic behaviors is crucial.
    
    References
    ----------
    .. [1] Schoukens, J., & Ljung, L. (2019). Nonlinear System
           Identification: A User-Oriented Roadmap. IEEE Control
           Systems Magazine, 39(6), 28-99.

    Examples
    --------
    >>> from gofast.estimators.dynamic_system import HammersteinWienerClassifier
    >>> from sklearn.datasets import make_classification
    >>> import numpy as np
    >>> # Generate synthetic data
    >>> X, y = make_classification(n_samples=200, n_features=5,
    ...                            n_informative=3, n_redundant=1,
    ...                            random_state=42)
    >>> # Instantiate the classifier
    >>> model = HammersteinWienerClassifier(p=2, verbose=1)
    >>> # Fit the model
    >>> model.fit(X, y)
    Starting HammersteinWienerClassifier fit method.
    Applying nonlinear input transformation.
    No nonlinear input estimator provided; using original X.
    Creating lagged features.
    Creating lag 1 features.
    Creating lag 2 features.
    Lagged features shape: (200, 10)
    Fitting linear model for classification.
    Applying nonlinear output transformation.
    No nonlinear output estimator provided; using linear output.
    Computed loss: 0.35
    Initial loss: 0.35
    Fit method completed.
    >>> # Predict
    >>> y_pred = model.predict(X)
    Starting predict method.
    Starting predict_proba method.
    Applying nonlinear input transformation.
    No nonlinear input estimator provided; using original X.
    Creating lagged features.
    Creating lag 1 features.
    Creating lag 2 features.
    Lagged features shape: (200, 10)
    predict_proba method completed.
    Predict method completed.
    >>> # Evaluate
    >>> from sklearn.metrics import accuracy_score
    >>> acc = accuracy_score(y, y_pred)
    >>> print(f"Accuracy: {acc:.2f}")
    Accuracy: 0.85
    """

    _parameter_constraints: dict = {
        **BaseHammersteinWiener._parameter_constraints,
        "loss": [StrOptions({"cross_entropy", "time_weighted_cross_entropy"})],
        "time_weighting": [StrOptions({"linear", "exponential", "inverse"})],
        "epsilon": [Interval(Real, 1e-15, None, closed='left')]
    }

    def __init__(
            self,
            nonlinear_input_estimator=None,
            nonlinear_output_estimator=None,
            p=1,
            loss="cross_entropy",
            time_weighting="linear",
            feature_engineering='auto',
            n_jobs=None,
            epsilon=1e-15,
            verbose=0
    ):
        super().__init__(
            nonlinear_input_estimator=nonlinear_input_estimator,
            nonlinear_output_estimator=nonlinear_output_estimator,
            p=p,
            feature_engineering=feature_engineering,
            n_jobs=n_jobs,
            verbose=verbose
        )
        self.loss = loss
        self.time_weighting = time_weighting
        self.epsilon = epsilon
        
    def fit(self, X, y):
        """
        Fit the Hammerstein-Wiener classifier to the training data.
        
        This method applies transformations to the input data, constructs 
        lagged features for time-based dynamics, and fits a logistic 
        regression model on these features to capture linear relationships 
        for classification. If a nonlinear output transformation is provided, 
        it is applied to the decision function output.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input samples, where `n_samples` is the number of samples 
            and `n_features` is the number of features.
    
        y : array-like of shape (n_samples,)
            Target binary class labels for each sample. Class labels are 
            typically 0 or 1, representing the two classes in binary 
            classification.
    
        Returns
        -------
        self : object
            Returns the fitted `HammersteinWienerClassifier` instance.
        
        Notes
        -----
        - Logistic Regression:
          The linear model is fitted using logistic regression, which aims 
          to maximize the likelihood of correctly predicting class labels. 
          Given lagged features, the logistic model’s decision function 
          output :math:`f(X_lagged)` is given by:
    
          .. math::
              f(X_{lagged}) = X_{lagged} \\beta + b
    
          where :math:`X_{lagged}` are the lagged features, :math:`\\beta` 
          represents the regression coefficients, and :math:`b` is the bias 
          term.
    
        - Nonlinear Output Transformation:
          If provided, a nonlinear transformation function :math:`g` is 
          applied to the decision function output, resulting in:
    
          .. math::
              y_{linear} = g(f(X_{lagged}))
    

        - This method uses logistic regression to fit the linear model.
        - Nonlinear output transformations allow the model to capture more 
          complex patterns beyond the linear decision boundary.
    
        Examples
        --------
        >>> from gofast.estimators.dynamic_system import HammersteinWienerClassifier
        >>> model = HammersteinWienerClassifier()
        >>> X = np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]])
        >>> y = np.array([0, 1, 0])
        >>> model.fit(X, y)
        >>> print("Initial loss:", model.initial_loss_)
    
        See Also
        --------
        predict_proba : Predict class probabilities for input data.
        predict       : Predict binary class labels for input data.
    
        References
        ----------
        .. [1] Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements 
               of Statistical Learning. Springer.
        """
        if self.verbose > 0:
            print("Starting HammersteinWienerClassifier fit method.")

        # Validate parameter constraints
        self._validate_params()

        # Validate input and output data
        X, y = check_X_y(X, y, multi_output=True)

        # Determine the type of target (binary, multiclass, multilabel)
        target_type = type_of_target(y)
        if target_type in ('binary', 'multiclass'):
            self.is_multilabel_ = False
        elif target_type in ('multilabel-indicator', 'multiclass-multioutput'):
            self.is_multilabel_ = True
        else:
            raise ValueError(f"Unsupported target type: {target_type}")

        # Apply nonlinear input transformation
        X_transformed = self._apply_nonlinear_input(X, y)

        # Create lagged features for linear dynamic block
        X_lagged = self._create_lagged_features(X_transformed)

        if self.verbose > 0:
            print("Fitting linear model for classification.")

        # Use logistic regression for the linear dynamic block
        self.linear_model_ = LogisticRegression()
        self.linear_model_.fit(X_lagged, y)

        # Get the decision function output
        y_linear = self.linear_model_.decision_function(X_lagged)

        # Fit nonlinear output estimator if provided
        self._apply_nonlinear_output(y_linear, y)

        # Compute initial loss for reporting
        y_pred_proba_initial = self.predict_proba(X)
        self.initial_loss_ = self._compute_loss(y, y_pred_proba_initial)

        if self.verbose > 0:
            print(f"Initial loss: {self.initial_loss_}")
            print("Fit method completed.")
    
    def predict_proba(self, X):
        """
        Predict class probabilities for each input sample in `X`.
    
        This method generates class probabilities by applying the nonlinear 
        input transformation, creating lagged features, and then using the 
        logistic regression model’s decision function. A logistic sigmoid 
        transformation is applied to the decision function output, yielding 
        class probabilities.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict probabilities for, where `n_samples` is the 
            number of samples and `n_features` is the number of features.
    
        Returns
        -------
        y_proba : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The first column 
            represents the probability of class 0, and the second column 
            represents the probability of class 1.
    
        Concept
        -------
        - Sigmoid Transformation:
          The logistic sigmoid function is applied to the decision function 
          output to convert it into probabilities:
    
          .. math::
              \\hat{p}(y = 1 | X) = \\sigma(f(X_{lagged})) = 
                                    \\frac{1}{1 + e^{-f(X_{lagged})}}
    
          where :math:`\\sigma` is the sigmoid function, and :math:`f(X_{lagged})` 
          is the output of the decision function.
    
        Notes
        -----
        - This method requires that the model is already fitted.
        - Lagged features allow the model to incorporate historical dependencies 
          in probability estimation.
    
        Examples
        --------
        >>> from gofast.estimators.dynamic_system import HammersteinWienerClassifier
        >>> model = HammersteinWienerClassifier()
        >>> model.fit(X, y)
        >>> probabilities = model.predict_proba(X)
        >>> print("Class probabilities:", probabilities)
    
        See Also
        --------
        predict : Predict binary class labels for input samples.
        """
        if self.verbose > 0:
            print("Starting predict_proba method.")

        # Check if the model is fitted
        check_is_fitted(self, 'linear_model_')

        # Validate input data
        X = check_array(X)

        # Apply nonlinear input transformation
        X_transformed = self._apply_nonlinear_input(X)

        # Create lagged features for linear dynamic block
        X_lagged = self._create_lagged_features(X_transformed)

        # Get the decision function output
        y_linear = self.linear_model_.decision_function(X_lagged)

        # Apply nonlinear output transformation
        y_transformed = self._apply_nonlinear_output(y_linear)

        # Convert to probabilities
        if self.is_multilabel_:
            # For multilabel, apply sigmoid function
            y_pred_proba= activator(y_transformed, activation="sigmoid")
            # y_pred_proba = expit(y_transformed)
        else:
            if len(self.linear_model_.classes_) == 2:
                # Binary classification
                # y_pred_proba = expit(y_transformed)
                y_pred_proba= activator(y_transformed, activation=self.activation)
                # Ensure the output has two columns
                y_pred_proba = np.hstack([1 - y_pred_proba, y_pred_proba])
            else:
                # Multiclass classification, apply softmax
                y_pred_proba= activator(y_transformed, activation="softmax")
                # y_pred_proba = softmax(y_transformed, axis=1)

        if self.verbose > 0:
            print("predict_proba method completed.")

        return y_pred_proba

    def predict(self, X):
        """
        Predict class labels for each input sample in `X`.
    
        This method generates class labels by first computing class 
        probabilities and then applying a threshold of 0.5 to determine 
        the predicted class. Samples with predicted probability greater 
        than or equal to 0.5 are classified as class 1, while the rest 
        are classified as class 0.
    
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict, where `n_samples` is the number of samples 
            and `n_features` is the number of features.
    
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted binary class labels for each sample. The labels are 
            either 0 or 1, representing the two classes in binary 
            classification.
    
        Notes
        -----
        - Probability Thresholding:
          This method applies a threshold to the predicted probabilities 
          to determine class labels:
    
          .. math::
              \\hat{y} = \\begin{cases} 
                            1 & \\text{if } \\hat{p}(y = 1 | X) \\geq 0.5 \\\\
                            0 & \\text{otherwise}
                         \\end{cases}
    
          where :math:`\\hat{p}(y = 1 | X)` is the probability of class 1.
    
        Notes
        -----
        - This method requires `predict_proba` to be executed first.
        - A threshold of 0.5 is commonly used for binary classification tasks, 
          though this can be adjusted if needed.
    
        Examples
        --------
        >>> from gofast.estimators.dynamic_system import HammersteinWienerClassifier
        >>> model = HammersteinWienerClassifier()
        >>> model.fit(X, y)
        >>> labels = model.predict(X)
        >>> print("Predicted labels:", labels)
    
        See Also
        --------
        predict_proba : Predict class probabilities for input samples.
        """
        if self.verbose > 0:
            print("Starting predict method.")

        y_pred_proba = self.predict_proba(X)

        if self.is_multilabel_:
            # For multilabel, threshold probabilities at 0.5
            y_pred = (y_pred_proba >= 0.5).astype(int)
        else:
            # For binary and multiclass, take argmax
            y_pred = np.argmax(y_pred_proba, axis=1)

        if self.verbose > 0:
            print("Predict method completed.")

        return y_pred
    
    def _compute_loss(self, y_true, y_pred_proba):
        """
        Computes the classification loss based on the specified loss function.
        This method calculates the average error between the true labels 
        (`y_true`) and predicted probabilities (`y_pred_proba`) by applying 
        either the cross-entropy or time-weighted cross-entropy loss function. 
        The latter assigns greater penalty to recent errors, which can be 
        essential in dynamic classification systems where recent predictions 
        carry higher importance.
    
        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True binary labels for each sample, where values are either 0 or 1.
            These labels are used as the ground truth for computing the loss.
    
        y_pred_proba : array-like of shape (n_samples,)
            Predicted probabilities for each sample. These values range between 
            0 and 1, representing the probability of the positive class. Values 
            are clipped between a small epsilon (1e-15) and 1 - epsilon to 
            prevent computational issues when taking the log of zero.
    
        Returns
        -------
        loss : float
            Computed loss value based on the specified loss function. The loss 
            is averaged across all samples.
    
        Notes
        -----
        - Cross-Entropy Loss:
          The cross-entropy loss measures the error between the true and 
          predicted probabilities. The formula is:
    
          .. math::
              L_{CE} = - \\frac{1}{n} \\sum_{i=1}^{n} 
                      \\left[ y_i \\log(\\hat{y}_i) 
                      + (1 - y_i) \\log(1 - \\hat{y}_i) \\right]
    
          where :math:`y_i` is the true label, and :math:`\\hat{y}_i` is the 
          predicted probability for sample :math:`i`.
    
        - Time-Weighted Cross-Entropy Loss:
          In time-weighted loss, a weight vector is applied, emphasizing recent 
          samples. The formula becomes:
    
          .. math::
              L_{TWCE} = - \\frac{1}{n} \\sum_{i=1}^{n} 
                         w_i \\left[ y_i \\log(\\hat{y}_i) 
                         + (1 - y_i) \\log(1 - \\hat{y}_i) \\right]
    
          where :math:`w_i` are time weights calculated by `_compute_time_weights`.
    

        - Cross-entropy is commonly used for binary classification problems.
        - Time-weighted cross-entropy can improve performance in time-sensitive 
          applications by focusing on recent prediction accuracy.
    
        Examples
        --------
        >>> from gofast.estimators.dynamic_system import HammersteinWienerClassifier
        >>> model = HammersteinWienerClassifier(loss="time_weighted_cross_entropy")
        >>> y_true = np.array([0, 1, 1, 0])
        >>> y_pred_proba = np.array([0.1, 0.8, 0.6, 0.2])
        >>> loss = model._compute_loss(y_true, y_pred_proba)
        >>> print(f"Computed loss: {loss:.4f}")
    
        See Also
        --------
        _compute_time_weights : Computes weights based on temporal relevance.
    
        References
        ----------
        .. [1] Bishop, C. M. (2006). Pattern Recognition and Machine Learning.
               Springer.
        """
        if self.verbose > 0:
            print(f"Computing loss using {self.loss} loss function.")
    
        # Clip probabilities to prevent log of zero
        y_pred_proba = np.clip(y_pred_proba, self.epsilon, 1 - self.epsilon)
    
        # Compute loss using sklearn's log_loss
        if self.loss == "cross_entropy":
            loss = log_loss(y_true, y_pred_proba, eps=self.epsilon)
        elif self.loss == "time_weighted_cross_entropy":
            weights = self._compute_time_weights(len(y_true))
            loss = log_loss(y_true, y_pred_proba, sample_weight=weights, eps=self.epsilon)
        else:
            raise ValueError("Unsupported loss function.")
    
        if self.verbose > 0:
            print(f"Computed loss: {loss}")
    
        return loss
     

    def _compute_time_weights(self, n):
        """
        Computes time-based weights to prioritize recent predictions. The 
        weight scheme is based on the `time_weighting` parameter, which defines 
        how the weights are distributed across samples. Supported options 
        include `linear`, `exponential`, and `inverse` weighting [1]_.
    
        Parameters
        ----------
        n : int
            The number of samples or time points for which weights are computed.
    
        Returns
        -------
        weights : array-like of shape (n,)
            Computed time weights, normalized to sum to one for consistent 
            scaling across different weighting schemes.
    
        Mathematical Concept
        --------------------
        Time weights are calculated as follows, based on the `time_weighting` 
        method:
    
        - Linear Weighting:
          The weights increase linearly, giving a low weight to the first 
          observation and the highest weight to the last.
    
          .. math::
              w_i = \\frac{0.1 + \\frac{i}{n-1}}{1.0}
    
        - Exponential Weighting:
          Weights follow an exponential increase, prioritizing recent samples 
          more aggressively.
    
          .. math::
              w_i = \\frac{e^{\\frac{i}{n}} - 1}{e^{1} - 1}
    
        - Inverse Weighting:
          The weights decrease inversely with the time index, placing 
          more importance on recent samples.
    
          .. math::
              w_i = \\frac{1}{i}
    
        Notes
        -----
        - Weights are normalized to sum to one across all samples.
        - Different weighting schemes can be used to fine-tune the model's 
          sensitivity to time-based dependencies.
    
        Examples
        --------
        >>> from gofast.estimators.dynamic_system import HammersteinWienerClassifier
        >>> model = HammersteinWienerClassifier(time_weighting="exponential")
        >>> weights = model._compute_time_weights(5)
        >>> print("Time weights:", weights)
    
        See Also
        --------
        _compute_loss : Uses weights to compute time-weighted cross-entropy loss.
    
        References
        ----------
        .. [1] Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: 
               An Introduction. MIT Press.
        """
        if self.verbose > 0:
            print(f"Computing time weights using {self.time_weighting} method.")

        # Compute weights based on the selected method
        if self.time_weighting == "linear":
            weights = np.linspace(0.1, 1.0, n)
        elif self.time_weighting == "exponential":
            weights = np.exp(np.linspace(0, 1, n)) - 1
            weights /= weights.max()  # Normalize to [0, 1]
        elif self.time_weighting == "inverse":
            weights = 1 / np.arange(1, n + 1)
            weights /= weights.max()  # Normalize to [0, 1]
        else:
            # Default to equal weights if unrecognized weighting scheme
            weights = np.ones(n)

        if self.verbose > 0:
            print(f"Time weights: {weights}")

        return weights
    