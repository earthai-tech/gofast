# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Provides perceptron-based estimators for both classification
and regression tasks. These estimators implement various perceptron
algorithms, including traditional Perceptron, Light Gradient Descent
(Classifier and Regressor), and Neuro-Fuzzy Neural Networks. By adhering
to the scikit-learn API, these estimators ensure compatibility and ease
of integration into existing machine learning workflows.

The available perceptron-based estimators include:
- `Perceptron`: A basic perceptron classifier for binary classification
  tasks.
- `LightGDClassifier`: A perceptron classifier utilizing light gradient
  descent.
- `LightGDRegressor`: A perceptron regressor utilizing light gradient
  descent.
- `NeuroFuzzyClassifier`: A neuro-fuzzy classifier integrating fuzzy
  logic.
- `NeuroFuzzyRegressor`: A neuro-fuzzy regressor integrating fuzzy
  logic.
- `FuzzyNeuralNetClassifier`: A fuzzy neural network classifier for
  enhanced classification performance.
- `FuzzyNeuralNetRegressor`: A fuzzy neural network regressor for
  enhanced regression performance.

Importing Estimators
--------------------
One can import these estimators directly fromthe `gofast` package without 
navigating through the `estimators` subpackage.
This simplifies the import statements and enhances code readability.

Examples
--------
Below is an example demonstrating the usage of the `LightGDClassifier`:

    >>> import numpy as np
    >>> from sklearn.datasets import load_iris
    >>> from gofast.perceptron import LightGDClassifier
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.metrics import accuracy_score

    >>> # Load the Iris dataset
    >>> iris = load_iris()
    >>> X = iris.data
    >>> y = iris.target

    >>> # Split the dataset into training and testing sets
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.3, random_state=42
    ... )

    >>> # Initialize the LightGDClassifier with desired parameters
    >>> lightgd_clf = LightGDClassifier(
    ...     eta0=0.01,
    ...     max_iter=50,
    ...     tol=1e-4,
    ...     random_state=42,
    ...     verbose=True
    ... )

    >>> # Fit the classifier to the training data
    >>> lightgd_clf.fit(X_train, y_train)
    LightGDClassifier(...)

    >>> # Make predictions on the test set
    >>> y_pred = lightgd_clf.predict(X_test)

    >>> # Evaluate the classifier's performance
    >>> accuracy = accuracy_score(y_test, y_pred)
    >>> print('Accuracy:', accuracy)
    Accuracy: 0.97

Additional Functionalities
--------------------------
- **Customizable Learning Rate**: Adjust the `eta0` parameter to control the
  learning rate of the gradient descent.
- **Maximum Iterations**: Set the `max_iter` parameter to limit the number
  of training iterations.
- **Tolerance for Stopping Criteria**: Use the `tol` parameter to define
  the tolerance for the optimization.
- **Neuro-Fuzzy Integration**: Enhance classifiers and regressors with
  fuzzy logic for better handling of uncertainty and imprecision.
- **Verbose Mode**: Enable verbose output to monitor the training progress.

References
----------
.. [1] Kouadio, K.L., Liu, J., Liu, R., Wang, Y., Liu, W., 2024.
       K-Means Featurizer: A booster for intricate datasets. Earth Sci.
       Informatics 17, 1203–1228. https://doi.org/10.1007/s12145-024-01236-3
.. [2] Rosenblatt, F. (1958). The perceptron: a probabilistic model for
      information storage and organization in the brain. Psychological
      Review, 65(6), 386–408.
.. [3] Minsky, M., & Papert, S. (1969). Perceptrons: An Introduction to
      Computational Geometry. MIT Press.
.. [4] Widrow, B., & Hoff, M. E. (1960). Adaptive Switching Circuits.
      1960 IRE WESCON Convention Record, 4, 96–104.
"""

import warnings
import importlib

from gofast.utils.deps_utils import import_optional_dependency

from gofast.estimators.perceptron import Perceptron
from gofast.estimators.perceptron import LightGDClassifier, LightGDRegressor

__all__ = ["Perceptron", "LightGDClassifier", "LightGDRegressor"]


def __getattr__(name):
    """
    Dynamically import optional perceptron-based classes and issue warnings
    if required dependencies are missing.

    Parameters
    ----------
    name : str
        Name of the attribute being accessed.

    Returns
    -------
    class
        The requested class if available.

    Warns
    -----
    UserWarning
        If the required dependencies for the class are not installed.
    """
    optional_classes = {
        "NeuroFuzzyRegressor",
        "NeuroFuzzyClassifier",
        "FuzzyNeuralNetClassifier",
        "FuzzyNeuralNetRegressor",
    }

    if name in optional_classes:
        try:
            import_optional_dependency("skfuzzy")
        except ImportError:
            warnings.warn(
                f"'{name}' requires the 'scikit-fuzzy' package. "
                "Please install it via 'pip install scikit-fuzzy' "
                "to use this estimator.",
                UserWarning,
                stacklevel=2
            )
            # raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
        # Dynamically import the class from the perceptron module
        try:
            module = importlib.import_module("gofast.estimators.perceptron")
            cls = getattr(module, name)
            globals()[name] = cls  # Cache the class in the globals
            __all__.append(name)    # Extend the __all__ list with the new class
            return cls
        except AttributeError:
            raise AttributeError( f"'{name}' is not found in 'gofast.perceptron'.")

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

