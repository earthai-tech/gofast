# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Provides utilities to check and validate Keras models,
including optional deep inspection of essential Keras
capabilities.
"""
import warnings 
from typing import Any, Callable, Optional

from ..utils.validator import has_required_attributes
from ..utils.deps_utils import ensure_pkg
from . import KERAS_DEPS,  KERAS_BACKEND, dependency_message

if KERAS_BACKEND: 
    Model= KERAS_DEPS.Model 
    Sequential=KERAS_DEPS.Sequential
    
DEP_MSG = dependency_message('nn.validator') 
    
__all__=[
    "is_keras_model", "validate_keras_model",
    "check_keras_model_status"
 ]

@ensure_pkg(KERAS_BACKEND or "keras", extra=DEP_MSG)
def is_keras_model(model: Any) -> bool:
    """
    Check if the provided object is an instance of a Keras model.

    This function imports TensorFlow (if installed) and returns
    True if ``model`` is an instance of ``tf.keras.models.Model``
    or ``tf.keras.Sequential``, False otherwise.

    Parameters
    ----------
    model : Any
        The object to check.

    Returns
    -------
    bool
        True if the object is a recognized Keras model,
        False otherwise.

    Raises
    ------
    ImportError
        If TensorFlow is not installed.

    Notes
    -----
    - Internally performs a lazy import of TensorFlow
      to check model types.
    """
    return isinstance(model, (Model, Sequential))

def validate_keras_model(
    model: Any,
    custom_check: Optional[Callable[[Any], bool]] = None,
    deep_check: bool = False,
    raise_exception: bool = False
) -> bool:
    """
    Validate an object as a Keras model, optionally with a custom check
    and/or deep inspection of Keras functionalities.

    This function ensures the object is recognized as a Keras model
    and can optionally:
      1. Check essential methods (``fit``, ``predict``, ``compile``,
         ``summary``) if `deep_check` is True.
      2. Run a user-supplied `custom_check` callback.
      3. Raise exceptions instead of returning False, if
         `raise_exception` is True.

    Parameters
    ----------
    model : Any
        The object to validate as a Keras model.
    custom_check : Callable[[Any], bool], optional
        A callback that returns True if the model passes the custom
        criteria, or False (or raises) if it fails. If None, no custom
        check is done. Defaults to None.
    deep_check : bool, optional
        If True, verifies that `model` has key methods associated with
        Keras (`fit`, `predict`, `compile`, and `summary`). Defaults
        to False.
    raise_exception : bool, optional
        If True, raises an exception upon validation failure instead
        of returning False. Defaults to False.

    Returns
    -------
    bool or Model 
        True if the model passes all checks, False otherwise. If
        `raise_exception` is True and a check fails, this function
        raises instead of returning a model.

    Raises
    ------
    TypeError
        If the object is not a Keras model, or if the deep check
        fails (when `raise_exception` is True).
    ValueError
        If the `custom_check` is provided and fails
        (when `raise_exception` is True).
    ImportError
        If TensorFlow is not installed.

    Examples
    --------
    >>> from tensorflow.keras.layers import Dense
    >>> from tensorflow.keras.models import Sequential
    >>> from gofast.nn.validator import validate_keras_model
    >>> # Suppose we have:
    ... model = Sequential([Dense(2)])
    >>> # Basic validation:
    ... validate_keras_model(model)
    True

    >>> # Custom check: model must have more than 1 layer
    ... custom_layer_check = lambda m: len(m.layers) > 1
    >>> validate_keras_model(model, custom_check=custom_layer_check)
    False

    >>> # Enforce deep inspection:
    ... validate_keras_model(model, deep_check=True)
    True
    """
    if not is_keras_model(model):
        msg = "Provided object is not a Keras model."
        if raise_exception:
            raise TypeError(msg)
        return False

    if deep_check:
        # Ensure essential Keras methods exist
        required = ['fit', 'predict', 'compile', 'summary']
        if not has_required_attributes(model, required):
            msg = (
                "Model lacks one or more essential Keras functionalities: "
                f"{required}."
            )
            if raise_exception:
                raise TypeError(msg)
            return False

    if custom_check is not None:
        try:
            if not custom_check(model):
                # The custom check returned False
                msg = "Custom check returned False."
                if raise_exception:
                    raise ValueError(msg)
                return False
        except Exception as exc:
            # The custom check raised an Exception
            msg = f"Custom check raised an exception: {exc}"
            if raise_exception:
                raise ValueError(msg) from exc
            return False

    return model

def check_keras_model_status(
    model,
    mode: str = "fit",
    ops: str = "check_only",
    error: str = "warn",
    err_msg: str = None
) -> bool:
    """
    Check whether a Keras model has been compiled, built, or trained,
    and optionally validate that status, raising a warning or error
    if not satisfied.

    Parameters
    ----------
    model : tf.keras.Model
        The Keras model to check.
    mode : {'fit', 'build', 'compile'}, optional
        The status to verify:
          - 'compile': Checks if the model is compiled
                       (i.e., has a non-None optimizer).
          - 'build':   Checks if the model was built
                       (i.e., ``model.built`` is True).
          - 'fit':     Checks if the model has been
                       trained (i.e., ``optimizer.iterations > 0``).
        Defaults to ``'fit'``.
    ops : {'check_only', 'validate'}, optional
        The action to take with the result:
          - 'check_only': Return True/False. (No side effects.)
          - 'validate':   If the check fails, either raise an error
                          or issue a warning, depending on ``error``.
        Defaults to ``'check_only'``.
    error : {'warn', 'raise'}, optional
        If ``ops='validate'`` and the check fails:
          - 'warn':  issue a warning
          - 'raise': raise a ValueError
        Defaults to ``'warn'``.
    err_msg : str, optional
        A custom message used if the status fails
        under validation. If None, a default message is used.

    Returns
    -------
    bool
        If ``ops='check_only'``, returns True if the model
        passes the requested status check, False otherwise.
        If ``ops='validate'``:
          - If the check fails, it either warns or raises,
            so this function won't return. 
          - If the check passes, returns True.

    Notes
    -----
    1. For ``mode='compile'``, we check that
       ``model.optimizer`` is not None, which typically
       means the model was compiled with an optimizer.
    2. For ``mode='build'``, we check the boolean
       ``model.built`` property. This becomes True when the
       model has a known input shape (e.g., from explicit
       build or a forward pass).
    3. For ``mode='fit'``, we look at
       ``model.optimizer.iterations``. If it's greater
       than zero, at least one training step occurred.
       This may not detect custom training loops or custom
       optimizers that do not increment that variable.

    Examples
    --------
    >>> import tensorflow as tf
    >>> from tensorflow.keras import layers
    >>> from gofast.nn.validator import check_keras_model_status
    >>> # Minimal model
    ... model = tf.keras.Sequential([
    ...     layers.Dense(1, input_shape=(3,))
    ... ])
    >>> # Check 'build' status
    ... check_keras_model_status(model, mode='build')
    False

    >>> # Build the model by calling it once
    ... model(tf.random.normal((1, 3)))
    >>> check_keras_model_status(model, mode='build')
    True

    >>> # Not compiled yet
    ... check_keras_model_status(model, mode='compile')
    False

    >>> # Compile
    ... model.compile(optimizer='adam', loss='mse')
    >>> check_keras_model_status(model, mode='compile')
    True

    >>> # Not fit yet
    ... check_keras_model_status(model, mode='fit')
    False

    >>> # Train it
    ... model.fit(tf.random.normal((10, 3)),
    ...           tf.random.normal((10, 1)),
    ...           epochs=1, verbose=0)
    >>> check_keras_model_status(model, mode='fit')
    True
    """
    # Try importing TensorFlow (if not already imported)
    if not KERAS_BACKEND: 
    # try:
    #     import tensorflow as tf  # noqa: F401
    # except ImportError as e:
        raise ImportError(
            "TensorFlow(prefered)/Keras is not installed. Please install"
            " one of them to proceed."
        ) # from e

    # Validate mode
    mode = mode.lower().strip()
    if mode not in {"compile", "build", "fit"}:
        raise ValueError(
            "`mode` must be one of {'compile', 'build', 'fit'}."
        )

    # Determine the model's status according to mode
    if mode == "compile":
        # Compiled if there's a non-None optimizer
        status = (model.optimizer is not None)

    elif mode == "build":
        # 'built' if model.built is True
        status = bool(model.built)

    else:  # mode == "fit"
        # Must have an optimizer and show iteration > 0
        if getattr(model, "optimizer", None) is None:
            status = False
        else:
            try:
                iterations = model.optimizer.iterations.numpy()
                status = (iterations > 0)
            except AttributeError:
                # If a custom optimizer has no .iterations
                status = False

    # Handle validation if ops == 'validate'
    if ops == "validate":
        if not status:
            # Construct default or custom error/warn message
            if err_msg is None:
                err_msg = f"Model does not satisfy '{mode}' status."

            if error == "warn":
                warnings.warn(err_msg)
    
            elif error == "raise":
                raise ValueError(err_msg)
                
        return model  # If passes, or if we didn't fail, return the model 

    # If ops == 'check_only', just return the boolean
    return status