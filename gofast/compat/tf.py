# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Provides utilities for managing compatibility between TensorFlow's Keras and 
standalone Keras. It includes functions and classes for dynamically importing 
Keras dependencies and checking the availability of TensorFlow or Keras.
"""

import logging
import warnings 
import importlib
from functools import wraps
from contextlib import contextmanager 
from typing import Callable 

# Attempt to import TensorFlow and set a flag based on availability
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False
    
__all__ = [
    'KerasDependencies',
    'check_keras_backend',
    'import_keras_dependencies',
    'import_keras_function',
    'standalone_keras', 
    'optional_tf_function', 
    'suppress_tf_warnings'
]

class KerasDependencies:
    def __init__(
        self,
        extra_msg=None,
        error='warn'
    ):
        self.extra_msg = extra_msg
        self.error = error
        self._dependencies = {}
        self._check_tensorflow()

    def _check_tensorflow(self):
        try:
            import_optional_dependency(
                'tensorflow',
                extra=self.extra_msg
            )
            if self.error == 'warn':
                warnings.warn(
                    "TensorFlow is installed.",
                    UserWarning
                )
        except ImportError as e:
            warnings.warn(f"{self.extra_msg}: {e}")

    def __getattr__(
        self,
        name
    ):
        if name not in self._dependencies:
            self._dependencies[name] = self._import_dependency(name)
        return self._dependencies[name]

    def _import_dependency(
        self,
        name
    ):
        standalone_mapping = {
            'reduce_mean': ('tensorflow', 'reduce_mean'),
            'reduce_sum': ('tensorflow', 'reduce_sum'),
            'rank': ('tensorflow', 'rank'), 
            'stack': ('tensorflow', 'stack'), 
            'reshape': ('tensorflow', 'reshape'), 
            'tile': ('tensorflow', 'tile'), 
            'concat': ('tensorflow', 'concat'),
            'expand_dims': ('tensorflow', 'expand_dims'), 
            'shape': ('tensorflow', 'shape'), 
            'square': ('tensorflow.math', 'square'),
            'GradientTape': ('tensorflow', 'GradientTape'),
            'Dataset': ('tensorflow.data', 'Dataset'),
            'add_n': ('tensorflow.math', 'add_n'), 
            'maximum':('tensorflow.math', 'maximum'), 
            'backend': ('tensorflow.keras', 'backend'), 
            'activations': ('tensorflow.keras', 'activations'), 
            'add': ('tensorflow.math', 'add'), 
            'range':('tensorflow', 'range'), 
            'convert_to_tensor': ('tensorflow', 'convert_to_tensor'), 
            'Tensor': ('tensorflow', 'Tensor'), 
            'cast': ('tensforflow', 'cast'), 
            'float32': ('tensorflow', 'float32'), 
            # 'constant': ('tensorflow', 'constant')
        }

        mapping = {
            **standalone_mapping,
            'Model': ('models', 'Model'),
            'Add': ('layers', 'Add'), 
            'Activation': ('layers', 'Activation'), 
            'Layer': ('layers', 'Layer'), 
            'ELU': ('layers', 'ELU'), 
            'LayerNormalization': ('layers', 'LayerNormalization'), 
            'TimeDistributed': ('layers', 'TimeDistributed'),
            'Softmax': ('layers', 'Softmax'), 
            'MultiHeadAttention': ('layers', 'MultiHeadAttention'), 
            'Sequential': ('models', 'Sequential'),
            'Dense': ('layers', 'Dense'),
            'Dropout': ('layers', 'Dropout'),
            'BatchNormalization': ('layers', 'BatchNormalization'),
            'LSTM': ('layers', 'LSTM'),
            'Input': ('layers', 'Input'),
            'Conv2D': ('layers', 'Conv2D'),
            'Optimizer':('optimizers', 'Optimizer'), 
            'Metric': ('metrics', 'Metric'), 
            'MaxPooling2D': ('layers', 'MaxPooling2D'),
            'Flatten': ('layers', 'Flatten'),
            'Attention': ('layers', 'Attention'),
            'Concatenate': ('layers', 'Concatenate'),
            'Adam': ('optimizers', 'Adam'),
            'SGD': ('optimizers', 'SGD'),
            'RMSprop': ('optimizers', 'RMSprop'),
            'mnist': ('datasets', 'mnist'),
            'cifar10': ('datasets', 'cifar10'),
            'Loss': ('losses', 'Loss'),
            'mean_squared_error': (
                'losses',
                'mean_squared_error'
            ),
            'categorical_crossentropy': (
                'losses',
                'categorical_crossentropy'
            ),
            'binary_crossentropy': (
                'losses',
                'binary_crossentropy'
            ),
            'K': ('backend', 'K'), 
            'sum': ('backend', 'sum'),
            'ones': ('backend', 'ones'),
            'zeros': ('backend', 'zeros'),
            'constant': ('backend', 'constant'),
            'random_normal': ('initializers', 'random_normal'),
            'glorot_uniform': ('initializers', 'glorot_uniform'),
            'EarlyStopping': ('callbacks', 'EarlyStopping'),
            'ModelCheckpoint': (
                'callbacks',
                'ModelCheckpoint'
            ),
            'Callback': ('callbacks', 'Callback'), 
            'TensorBoard': ('callbacks', 'TensorBoard'),
            'LearningRateScheduler': (
                'callbacks',
                'LearningRateScheduler'
            ),
            'Embedding': ('layers', 'Embedding'), 
            'clone_model': ('models', 'clone_model'),
            'load_model': ('models', 'load_model'),
            'register_keras_serializable': ('utils', 'register_keras_serializable')
         
        }

        if name in mapping:
            module_name, function_name = mapping[name]
            return import_keras_function(
                module_name,
                function_name,
                error=self.error
            )
        raise AttributeError(
            f"'KerasDependencies' object has no attribute '{name}'"
        )
        
# XXX TODO 
# WARNING:tensorflow:From C:\Users\Daniel\Anaconda3\envs\watex\lib\site-packages
# \keras\src\losses.py:2976: The name tf.losses.sparse_softmax_cross_entropy is
#  deprecated. Please use tf.compat.v1.losses.sparse_softmax_cross_entropy instead.

def import_keras_function(
    module_name,
    function_name,
    error='warn'
):
    try:
        # Import directly from TensorFlow for TensorFlow-specific functions
        if module_name.startswith('tensorflow'):
            tf_module = importlib.import_module(module_name)
            function = getattr(tf_module, function_name)
            if error == 'warn':
                warnings.warn(
                    f"Using TensorFlow for {function_name} from {module_name}"
                )
            return function

        # Import from tensorflow.keras
        tf_keras_module = __import__(
            'tensorflow.keras.' + module_name,
            fromlist=[function_name]
        )
        function = getattr(tf_keras_module, function_name)
        if error == 'warn':
            warnings.warn(
                f"Using TensorFlow Keras for {function_name} from {module_name}"
            )
        return function
    except ImportError:
        try:
            # Fallback to standalone Keras
            keras_module = __import__(
                'keras.' + module_name,
                fromlist=[function_name]
            )
            function = getattr(keras_module, function_name)
            if error == 'warn':
                warnings.warn(
                    f"Using Keras for {function_name} from {module_name}"
                )
            return function
        except ImportError:
            raise ImportError(
                f"Cannot import {function_name} from {module_name}. "
                "Ensure TensorFlow or Keras is installed."
            )

def import_optional_dependency(
    package_name,
    error='warn',
    extra=None
):
    try:
        module = importlib.import_module(package_name)
        return module
    except ImportError as e:
        message = f"{package_name} is not installed"
        if extra:
            message = f"{extra}: {e}"
        else:
            message = (
                f"{message}: {e}. Use pip or conda to install it."
            )
        if error == 'warn':
            warnings.warn(message)
            return None
        elif error == 'raise':
            raise ImportError(message)
        else:
            raise ValueError(
                "Parameter 'error' must be either 'warn' or 'raise'."
            )

def import_keras_dependencies(
    extra_msg=None,
    error='warn'
):
    return KerasDependencies(extra_msg, error)

def check_keras_backend(
    error='warn'
):
    try:
        importlib.import_module('tensorflow')
        return 'tensorflow'
    except ImportError:
        try:
            importlib.import_module('keras')
            return 'keras'
        except ImportError as e:
            message = (
                "Neither TensorFlow nor Keras is installed."
            )
            if error == 'warn':
                warnings.warn(message)
            elif error == 'raise':
                raise ImportError(message) from e
            return None

def standalone_keras(module_name):
    """
    Tries to import the specified module from tensorflow.keras or 
    standalone keras.

    Parameters
    ----------
    module_name : str
        The name of the module to import (e.g., 'activations', 'layers', etc.).

    Returns
    -------
    module
        The imported module from tensorflow.keras or keras.

    Raises
    ------
    ImportError
        If neither tensorflow.keras nor standalone keras is installed or if
        the specified module does not exist in both frameworks.
        
    Examples
    ---------
    # Usage example
    try:
        activations = import_keras_module("activations")
        print("Successfully loaded activations module from:", activations)
    except ImportError as e:
        print(e)
            
    """
    try:
        # Try importing from tensorflow.keras
        import tensorflow.keras as tf_keras
        return getattr(tf_keras, module_name)
    except (ImportError, AttributeError):
        try:
            # Fallback to standalone keras
            import keras
            return getattr(keras, module_name)
        except (ImportError, AttributeError):
            raise ImportError(
                f"Module '{module_name}' could not be imported from either "
                f"tensorflow.keras or standalone keras. Ensure that TensorFlow "
                f"or standalone Keras is installed and the module exists."
            )

def optional_tf_function(func: Callable) -> Callable:
    """
    A decorator that applies @tf.function if TensorFlow is available.
    Otherwise, it wraps the function to raise an ImportError when called.

    Parameters:
    - ``func`` (`Callable`): The function to decorate.

    Returns:
    - `Callable`: The decorated function.
    """
    if HAS_TF:
        return tf.function(func)
    else:
        @wraps(func)
        def wrapper(*args, **kwargs):
            raise ImportError(
                "TensorFlow is required to use this function. "
                "Please install TensorFlow to proceed."
            )
        return wrapper

@contextmanager
def suppress_tf_warnings():
    """
    A context manager to temporarily suppress TensorFlow warnings.
    
    Usage:
        with suppress_tf_warnings():
            # TensorFlow operations that may generate warnings
            ...
    """
    if HAS_TF: 
        tf_logger = tf.get_logger()
        original_level = tf_logger.level
        tf_logger.setLevel(logging.ERROR)  # Suppress WARNING and INFO messages
        try:
            yield
        finally:
            tf_logger.setLevel(original_level)  # Restore original logging level
            
# ---------------------- class and func documentations ----------------------

KerasDependencies.__doc__="""\ 
Lazy-loads Keras dependencies from `tensorflow.keras` or `keras`.

Parameters
----------
extra_msg : str, optional
    Additional message to display in the warning if TensorFlow is 
    not found. This message will be appended to the default warning 
    message. Default is `None`.
error : str, optional
    Defines the error handling strategy. If `'warn'` (default), a 
    warning message will be displayed if TensorFlow is not found. 
    If `'raise'`, an `ImportError` will be raised. Default is 
    `'warn'`.

Notes
-----
This class attempts to dynamically import necessary Keras 
dependencies from either `tensorflow.keras` or `keras` on demand.

Mathematically, the import of each function or class can be 
represented as:

.. math::

    f = 
    \begin{cases} 
    f_{\text{tf.keras}} & \text{if TensorFlow is installed} \\
    f_{\text{keras}} & \text{if only Keras is installed} \\
    \end{cases}

where :math:`f` is the function or class to be imported, 
:math:`f_{\text{tf.keras}}` is the function or class from 
`tensorflow.keras`, and :math:`f_{\text{keras}}` is the function or 
class from `keras`.

Examples
--------
>>> from gofast.compat.tf import KerasDependencies
>>> deps = KerasDependencies(extra_msg="Custom warning message")
>>> Adam = deps.Adam  # Only imports Adam optimizer when accessed
>>> Sequential = deps.Sequential  # Only imports Sequential model when accessed

See Also
--------
importlib.import_module : Import a module programmatically.
keras.models.Model : The `Model` class from `keras.models`.
keras.models.Sequential : The `Sequential` class from `keras.models`.
tensorflow.keras.models.Model : The `Model` class from `tensorflow.keras.models`.
tensorflow.keras.models.Sequential : The `Sequential` class from 
                                      `tensorflow.keras.models`.

References
----------
.. [1] Chollet, François. *Deep Learning with Python*. Manning, 2017.
.. [2] Abadi, Martín et al. *TensorFlow: Large-Scale Machine Learning 
        # on Heterogeneous Systems*. 2015.    
    
    
"""

import_optional_dependency.__doc__="""\
Checks if a package is installed and optionally imports it.

Parameters
----------
package_name : str
    The name of the package to check and optionally import.
error : str, optional
    Defines the error handling strategy. If `'warn'` (default), a warning 
    message will be displayed if the package is not found. If `'raise'`,
    an ImportError will be raised.
extra : str, optional
    Additional message to display in the warning or error if the package 
    is not found.

Returns
-------
module or None
    The imported module if available, otherwise None if `error='warn'` is used.

Raises
------
ImportError
    If the package is not installed and `error='raise'` is used.

Notes
-----
This function attempts to check for the presence of a specified package and 
optionally import it. 
If the package is not found, the behavior depends on the `error` parameter.

Examples
--------
>>> import_optional_dependency('numpy')
Using numpy
>>> import_optional_dependency('nonexistent_package',
                                extra="Custom warning message", error='warn')
Custom warning message: No module named 'nonexistent_package'
>>> import_optional_dependency('nonexistent_package', error='raise')
Traceback (most recent call last):
  ...
ImportError: nonexistent_package is not installed. Use pip or conda to install it.

See Also
--------
importlib.import_module : Import a module programmatically.

References
----------
.. [1] Python documentation on importlib: https://docs.python.org/3/library/importlib.html
"""


import_keras_function.__doc__="""\
Tries to import a function from `tensorflow.keras`, falling back to 
`keras` if necessary.

Parameters
----------
module_name : str
    The name of the module from which to import the function. 
    This could be a submodule within `tensorflow.keras` or `keras`. 
    For example, `'optimizers'` or `'models'`.
function_name : str
    The name of the function to import from the specified module. 
    For example, `'Adam'` or `'load_model'`.
error : str, optional
    Defines the error handling strategy. If `'warn'` (default), 
    a warning message will be displayed when the function is 
    imported. If `'ignore'`, no warnings will be shown.

Returns
-------
function
    The imported function from either `tensorflow.keras` or `keras`, 
    depending on what is available in the environment.

Raises
------
ImportError
    If neither `tensorflow.keras` nor `keras` is installed, or if 
    the specified function cannot be found in the respective modules.

Notes
-----
This function attempts to dynamically import a specified function 
from either `tensorflow.keras` or `keras`. It first tries to import 
from `tensorflow.keras`, and if that fails (due to `tensorflow` not 
being installed), it falls back to `keras`.

Mathematically, the import can be considered as a selection 
operation:

.. math::

    f = 
    \begin{cases} 
    f_{\text{tf.keras}} & \text{if TensorFlow is installed} \\
    f_{\text{keras}} & \text{if only Keras is installed} \\
    \end{cases}

where :math:`f` is the function to be imported, :math:`f_{\text{tf.keras}}` 
is the function from `tensorflow.keras`, and :math:`f_{\text{keras}}` 
is the function from `keras`.

Examples
--------
>>> from gofast.compat.nn import import_keras_function
>>> Adam = import_keras_function('optimizers', 'Adam')
>>> load_model = import_keras_function('models', 'load_model')

See Also
--------
keras.optimizers.Adam
keras.models.load_model
tensorflow.keras.optimizers.Adam
tensorflow.keras.models.load_model

References
----------
.. [1] Chollet, François. *Deep Learning with Python*. Manning, 2017.
.. [2] Abadi, Martn et al. *TensorFlow: Large-Scale Machine Learning 
        on Heterogeneous Systems*. 2015.
"""


import_keras_dependencies.__doc__="""\
Create a `KerasDependencies` instance for lazy-loading Keras dependencies.

Parameters
----------
extra_msg : str, optional
    Additional message to display in the warning if TensorFlow is 
    not found. This message will be appended to the default warning 
    message. Default is `None`.
error : str, optional
    Defines the error handling strategy. If `'warn'` (default), a 
    warning message will be displayed if TensorFlow is not found. 
    If `'raise'`, an `ImportError` will be raised. Default is 
    `'warn'`.

Returns
-------
KerasDependencies
    An instance of the `KerasDependencies` class for lazy-loading 
    Keras dependencies.

Notes
-----
This function returns an instance of the `KerasDependencies` class, 
which attempts to dynamically import necessary Keras dependencies 
from either `tensorflow.keras` or `keras` on demand.

Mathematically, the import of each function or class can be 
represented as:

.. math::

    f = 
    \begin{cases} 
    f_{\text{tf.keras}} & \text{if TensorFlow is installed} \\
    f_{\text{keras}} & \text{if only Keras is installed} \\
    \end{cases}

where :math:`f` is the function or class to be imported, 
:math:`f_{\text{tf.keras}}` is the function or class from 
`tensorflow.keras`, and :math:`f_{\text{keras}}` is the function or 
class from `keras`.

Examples
--------
>>> from gofast.compat.tf import import_keras_dependencies
>>> deps = import_keras_dependencies(extra_msg="Custom warning message")
>>> Adam = deps.Adam  # Only imports Adam optimizer when accessed
>>> Sequential = deps.Sequential  # Only imports Sequential model when accessed

See Also
--------
importlib.import_module : Import a module programmatically.
keras.models.Model : The `Model` class from `keras.models`.
keras.models.Sequential : The `Sequential` class from `keras.models`.
tensorflow.keras.models.Model : The `Model` class from 
                                `tensorflow.keras.models`.
tensorflow.keras.models.Sequential : The `Sequential` class from 
                                      `tensorflow.keras.models`.

References
----------
.. [1] Chollet, François. *Deep Learning with Python*. Manning, 2017.
.. [2] Abadi, Martín et al. *TensorFlow: Large-Scale Machine Learning 
        on Heterogeneous Systems*. 2015.
"""



check_keras_backend.__doc__="""\
Check if `tensorflow` or `keras` is installed.

Parameters
----------
error : str, optional
    Defines the error handling strategy. If `'warn'` (default), a 
    warning message will be displayed if neither `tensorflow` nor 
    `keras` is installed. If `'raise'`, an `ImportError` will be 
    raised. If `'ignore'`, no action will be taken. Default is `'warn'`.

Returns
-------
str or None
    Returns 'tensorflow' if TensorFlow is installed, 'keras' if Keras 
    is installed, or None if neither is installed.

Examples
--------
>>> backend = check_keras_backend()
>>> if backend:
>>>     print(f"{backend} is installed.")
>>> else:
>>>     print("Neither tensorflow nor keras is installed.")

Notes
-----
This function first checks for TensorFlow installation and then for 
Keras. The behavior on missing packages depends on the `error` 
parameter.
"""

