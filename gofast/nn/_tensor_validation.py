# -*- coding: utf-8 -*-
# gofast.nn._tensor_validation.py 

from typing import List, Tuple, Optional, Union, Dict, Any
from ..utils.deps_utils import ensure_pkg 
from ..compat.tf import optional_tf_function, suppress_tf_warnings  

try:
    import tensorflow as tf
except ImportError:
    # Warn the user that TensorFlow is required for this module
    import warnings
    warnings.warn(
        "TensorFlow is not installed. Please install TensorFlow to use "
        "this module.",
        ImportWarning
    )

@ensure_pkg(
    'tensorflow',
    extra="Need 'tensorflow' for this function to proceed"
)
def validate_anomaly_scores(
    anomaly_config: Optional[Dict[str, Any]],
    forecast_horizons: int,
) -> Optional[tf.Tensor]:
    """
    Validates and processes the ``anomaly_scores`` in the provided 
    `anomaly_config` dictionary.

    Parameters:
    - ``anomaly_config`` (Optional[`Dict[str, Any]`]): 
        Dictionary that may contain:
            - 'anomaly_scores': Precomputed anomaly scores tensor.
            - 'anomaly_loss_weight': Weight for anomaly loss.
    - ``forecast_horizons`` (int): 
        The expected number of forecast horizons (second dimension 
        of `anomaly_scores`).

    Returns:
    - Optional[`tf.Tensor`]: 
        Validated `anomaly_scores` tensor of shape 
        (batch_size, forecast_horizons), cast to float32.
        Returns None if `anomaly_scores` is not provided.

    Raises:
    - ValueError: 
        If `anomaly_scores` is provided but is not a 2D tensor or the 
        second dimension does not match `forecast_horizons`.
    """

    if anomaly_config is None:
        # If `anomaly_config` is None, no `anomaly_scores` or 
        # `anomaly_loss_weight` are set
        return None

    if isinstance(anomaly_config, dict):
        # Ensure 'anomaly_scores' key exists in the dictionary
        if 'anomaly_scores' not in anomaly_config:
            anomaly_config['anomaly_scores'] = None

        anomaly_scores = anomaly_config.get('anomaly_scores')
    else:
        # Assume `anomaly_scores` is passed directly as `anomaly_config`
        anomaly_scores = anomaly_config
        anomaly_config = {}

    if anomaly_scores is not None:
        # Convert to tensor if not already a TensorFlow tensor
        if not isinstance(anomaly_scores, tf.Tensor):
            try:
                anomaly_scores = tf.convert_to_tensor(
                    anomaly_scores,
                    dtype=tf.float32
                )
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Failed to convert `anomaly_scores` to a TensorFlow tensor: {e}"
                )
        else:
            # Cast to float32 if it's already a tensor
            anomaly_scores = tf.cast(anomaly_scores, tf.float32)

        # Validate that `anomaly_scores` is a 2D tensor
        if anomaly_scores.ndim != 2:
            raise ValueError(
                f"`anomaly_scores` must be a 2D tensor with shape "
                f"(batch_size, forecast_horizons), but got "
                f"{anomaly_scores.ndim}D tensor."
            )

        # Validate that the second dimension matches `forecast_horizons`
        if anomaly_scores.shape[1] != forecast_horizons:
            raise ValueError(
                f"`anomaly_scores` second dimension must be "
                f"{forecast_horizons}, but got "
                f"{anomaly_scores.shape[1]}."
            )

        # Update the `anomaly_config` with the processed 
        # `anomaly_scores` tensor
        anomaly_config['anomaly_scores'] = anomaly_scores
        return anomaly_scores

    else:
        # If `anomaly_scores` is not provided, ensure it's set to None
        anomaly_config['anomaly_scores'] = None
        return anomaly_scores

@optional_tf_function
def validate_xtft_inputs(
    inputs: Union[List[Any], Tuple[Any, ...]],
    static_input_dim: int,
    dynamic_input_dim: int,
    future_covariate_dim: Optional[int] = None
) -> Tuple[tf.Tensor, tf.Tensor, Optional[tf.Tensor]]:
    """
    Validates and processes the ``inputs`` for the XTFT model.
    
    Parameters:
    - ``inputs`` (Union[List[Any], Tuple[Any, ...]]): 
        A list or tuple containing the inputs to the model in the following 
        order: [static_input, dynamic_input, future_covariate_input].
        
        - `static_input`: TensorFlow tensor or array-like object 
          representing static features.
        - `dynamic_input`: TensorFlow tensor or array-like object 
          representing dynamic features.
        - `future_covariate_input`: (Optional) TensorFlow tensor or 
          array-like object representing future covariates.
          Can be `None` if not used.
    - ``static_input_dim`` (int): 
        The expected dimensionality of the static input features 
        (i.e., number of static features).
    - ``dynamic_input_dim`` (int): 
        The expected dimensionality of the dynamic input features 
        (i.e., number of dynamic features).
    - ``future_covariate_dim`` (Optional[int], optional): 
        The expected dimensionality of the future covariate features 
        (i.e., number of future covariate features).
        If `None`, the function expects `future_covariate_input` to be 
        `None`.
    
    Returns:
    - ``static_input`` (`tf.Tensor`): 
        Validated static input tensor of shape 
        `(batch_size, static_input_dim)` and dtype `float32`.
    - ``dynamic_input`` (`tf.Tensor`): 
        Validated dynamic input tensor of shape 
        `(batch_size, time_steps, dynamic_input_dim)` and dtype `float32`.
    - ``future_covariate_input`` (`tf.Tensor` or `None`): 
        Validated future covariate input tensor of shape 
        `(batch_size, time_steps, future_covariate_dim)` and dtype `float32`.
        Returns `None` if `future_covariate_dim` is `None` or if the input 
        was `None`.
    
    Raises:
    - ValueError: 
        If ``inputs`` is not a list or tuple with the required number of 
        elements.
        If ``future_covariate_dim`` is specified but 
        ``future_covariate_input`` is `None`.
        If the provided inputs do not match the expected dimensionalities.
        If the inputs contain incompatible batch sizes.
    
    Examples:
    ---------
    >>> # Example without future covariates
    >>> import tensorflow as tf 
    >>> from gofast.nn._tensor_validation import validate_xtft_inputs 
    >>> static_input = tf.random.normal((32, 10))
    >>> dynamic_input = tf.random.normal((32, 20, 45))
    >>> inputs = [static_input, dynamic_input, None]
    >>> validated_static, validated_dynamic, validated_future = validate_xtft_inputs(
    ...     inputs,
    ...     static_input_dim=10,
    ...     dynamic_input_dim=45,
    ...     future_covariate_dim=None
    ... )
    >>> print(validated_static.shape, validated_dynamic.shape, validated_future)
    (32, 10) (32, 20, 45) None
    
    >>> # Example with future covariates
    >>> future_covariate_input = tf.random.normal((32, 20, 5))
    >>> inputs = [static_input, dynamic_input, future_covariate_input]
    >>> validated_static, validated_dynamic, validated_future = validate_xtft_inputs(
    ...     inputs,
    ...     static_input_dim=10,
    ...     dynamic_input_dim=45,
    ...     future_covariate_dim=5
    ... )
    >>> print(validated_static.shape, validated_dynamic.shape, validated_future.shape)
    (32, 10) (32, 20, 45) (32, 20, 5)
    """
    # Step 1: Validate the type and length of inputs
    if not isinstance(inputs, (list, tuple)):
        raise ValueError(
            f"'inputs' must be a list or tuple, but got type {type(inputs).__name__}."
        )
    
    expected_length = 3
    if len(inputs) != expected_length:
        raise ValueError(
            f"'inputs' must contain exactly {expected_length} elements: "
            f"[static_input, dynamic_input, future_covariate_input]. "
            f"Received {len(inputs)} elements."
        )
    
    # Unpack inputs
    static_input, dynamic_input, future_covariate_input = inputs

    # Step 2: Validate static_input
    if static_input is None:
        raise ValueError("``static_input`` cannot be None.")
    
    # Convert to tensor if not already
    if not isinstance(static_input, tf.Tensor):
        try:
            static_input = tf.convert_to_tensor(
                static_input,
                dtype=tf.float32
            )
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Failed to convert ``static_input`` to a TensorFlow tensor: {e}"
            )
    else:
        # Ensure dtype is float32
        static_input = tf.cast(static_input, tf.float32)
    
    # Check static_input dimensions
    if static_input.ndim != 2:
        raise ValueError(
            f"``static_input`` must be a 2D tensor with shape "
            f"(batch_size, static_input_dim), but got {static_input.ndim}D tensor."
        )
    
    # Check static_input_dim
    if static_input.shape[1] is not None and static_input.shape[1] != static_input_dim:
        raise ValueError(
            f"``static_input`` has incorrect feature dimension. Expected "
            f"{static_input_dim}, but got {static_input.shape[1]}."
        )
    elif static_input.shape[1] is None:
        # Dynamic dimension, cannot validate now
        pass

    # Step 3: Validate dynamic_input
    if dynamic_input is None:
        raise ValueError("``dynamic_input`` cannot be None.")
    
    # Convert to tensor if not already
    if not isinstance(dynamic_input, tf.Tensor):
        try:
            dynamic_input = tf.convert_to_tensor(
                dynamic_input,
                dtype=tf.float32
            )
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Failed to convert ``dynamic_input`` to a TensorFlow tensor: {e}"
            )
    else:
        # Ensure dtype is float32
        dynamic_input = tf.cast(dynamic_input, tf.float32)
    
    # Check dynamic_input dimensions
    if dynamic_input.ndim != 3:
        raise ValueError(
            f"``dynamic_input`` must be a 3D tensor with shape "
            f"(batch_size, time_steps, dynamic_input_dim), but got "
            f"{dynamic_input.ndim}D tensor."
        )
    
    # Check dynamic_input_dim
    if dynamic_input.shape[2] is not None and dynamic_input.shape[2] != dynamic_input_dim:
        raise ValueError(
            f"``dynamic_input`` has incorrect feature dimension. Expected "
            f"{dynamic_input_dim}, but got {dynamic_input.shape[2]}."
        )
    elif dynamic_input.shape[2] is None:
        # Dynamic dimension, cannot validate now
        pass

    # Step 4: Validate future_covariate_input
    if future_covariate_dim is not None:
        if future_covariate_input is None:
            raise ValueError(
                "``future_covariate_dim`` is specified, but "
                "``future_covariate_input`` is None."
            )
        
        # Convert to tensor if not already
        if not isinstance(future_covariate_input, tf.Tensor):
            try:
                future_covariate_input = tf.convert_to_tensor(
                    future_covariate_input,
                    dtype=tf.float32
                )
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Failed to convert ``future_covariate_input`` to a TensorFlow tensor: {e}"
                )
        else:
            # Ensure dtype is float32
            future_covariate_input = tf.cast(future_covariate_input, tf.float32)
        
        # Check future_covariate_input dimensions
        if future_covariate_input.ndim != 3:
            raise ValueError(
                f"``future_covariate_input`` must be a 3D tensor with shape "
                f"(batch_size, time_steps, future_covariate_dim), but got "
                f"{future_covariate_input.ndim}D tensor."
            )
        
        # Check future_covariate_dim
        if (future_covariate_input.shape[2] is not None and 
            future_covariate_input.shape[2] != future_covariate_dim):
            raise ValueError(
                f"``future_covariate_input`` has incorrect feature dimension. "
                f"Expected {future_covariate_dim}, but got "
                f"{future_covariate_input.shape[2]}."
            )
        elif future_covariate_input.shape[2] is None:
            # Dynamic dimension, cannot validate now
            pass
    else:
        if future_covariate_input is not None:
            raise ValueError(
                "``future_covariate_dim`` is None, but "
                "``future_covariate_input`` is provided."
            )
    
    # Step 5: Validate batch sizes across inputs
    static_batch_size = tf.shape(static_input)[0]
    dynamic_batch_size = tf.shape(dynamic_input)[0]
    
    with suppress_tf_warnings():
        if future_covariate_dim is not None:
            future_batch_size = tf.shape(future_covariate_input)[0]
            # Check if all batch sizes are equal
            batch_size_cond = tf.reduce_all([
                tf.equal(static_batch_size, dynamic_batch_size),
                tf.equal(static_batch_size, future_batch_size)
            ])
        else:
            # Check only static and dynamic batch sizes
            batch_size_cond = tf.equal(static_batch_size, dynamic_batch_size)
        
        # Ensure batch sizes match
        tf.debugging.assert_equal(
            batch_size_cond, True,
            message=(
                "Batch sizes do not match across inputs: "
                f"``static_input`` batch_size={static_batch_size}, "
                f"``dynamic_input`` batch_size={dynamic_batch_size}" +
                (f", ``future_covariate_input`` batch_size={future_batch_size}" 
                 if future_covariate_dim is not None else "")
            )
        )

    return static_input, dynamic_input, future_covariate_input

@optional_tf_function
def validate_batch_sizes(
    static_batch_size: tf.Tensor,
    dynamic_batch_size: tf.Tensor,
    future_batch_size: Optional[tf.Tensor] = None
) -> None:
    """
    Validates that the batch sizes of static, dynamic, and future 
    covariate inputs match.
    
    Parameters:
    - ``static_batch_size`` (`tf.Tensor`): 
        Batch size of the static input.
    - ``dynamic_batch_size`` (`tf.Tensor`): 
        Batch size of the dynamic input.
    - ``future_batch_size`` (`Optional[tf.Tensor]`, optional): 
        Batch size of the future covariate input.
        Defaults to `None`.
    
    Raises:
    - tf.errors.InvalidArgumentError: 
        If the batch sizes do not match.
    """
    tf.debugging.assert_equal(
        static_batch_size, dynamic_batch_size,
        message=(
            "Batch sizes do not match across inputs: "
            f"``static_input`` batch_size={static_batch_size.numpy()}, "
            f"``dynamic_input`` batch_size={dynamic_batch_size.numpy()}" +
            (f", ``future_covariate_input`` batch_size={future_batch_size.numpy()}" 
             if future_batch_size is not None else "")
        )
    )
    if future_batch_size is not None:
        tf.debugging.assert_equal(
            static_batch_size, future_batch_size,
            message=(
                "Batch sizes do not match between static and future covariate inputs: "
                f"``static_input`` batch_size={static_batch_size.numpy()}, "
                f"``future_covariate_input`` batch_size={future_batch_size.numpy()}."
            )
        )

@optional_tf_function
def check_batch_sizes(
    static_batch_size: tf.Tensor,
    dynamic_batch_size: tf.Tensor,
    future_batch_size: Optional[tf.Tensor] = None
) -> None:
    """
    Checks that the batch sizes of static, dynamic, and future covariate 
    inputs are equal.
    
    Parameters:
    - ``static_batch_size`` (`tf.Tensor`): 
        Batch size of the static input.
    - ``dynamic_batch_size`` (`tf.Tensor`): 
        Batch size of the dynamic input.
    - ``future_batch_size`` (`Optional[tf.Tensor]`, optional): 
        Batch size of the future covariate input.
        Defaults to `None`.
    
    Raises:
    - tf.errors.InvalidArgumentError: 
        If the batch sizes do not match.
    """
    tf.assert_equal(
        static_batch_size, dynamic_batch_size,
        message=(
            "Batch sizes do not match across inputs: "
            f"``static_input`` batch_size={static_batch_size.numpy()}, "
            f"``dynamic_input`` batch_size={dynamic_batch_size.numpy()}" +
            (f", ``future_covariate_input`` batch_size={future_batch_size.numpy()}" 
             if future_batch_size is not None else "")
        )
    )
    if future_batch_size is not None:
        tf.assert_equal(
            static_batch_size, future_batch_size,
            message=(
                "Batch sizes do not match between static and future covariate inputs: "
                f"``static_input`` batch_size={static_batch_size.numpy()}, "
                f"``future_covariate_input`` batch_size={future_batch_size.numpy()}."
            )
        )


def validate_batch_sizes_eager(
    static_batch_size: int,
    dynamic_batch_size: int,
    future_batch_size: Optional[int] = None
) -> None:
    """
    Validates that the batch sizes of static, dynamic, and future covariate 
    inputs match in eager execution mode.
    
    Parameters:
    - ``static_batch_size`` (int): 
        Batch size of the static input.
    - ``dynamic_batch_size`` (int): 
        Batch size of the dynamic input.
    - ``future_batch_size`` (`Optional[int]`, optional): 
        Batch size of the future covariate input.
        Defaults to `None`.
    
    Raises:
    - AssertionError: 
        If the batch sizes do not match.
    """
    assert static_batch_size == dynamic_batch_size, (
        "Batch sizes do not match across inputs: "
        f"``static_input`` batch_size={static_batch_size}, "
        f"``dynamic_input`` batch_size={dynamic_batch_size}" +
        (f", ``future_covariate_input`` batch_size={future_batch_size}" 
         if future_batch_size is not None else "")
    )
    if future_batch_size is not None:
        assert static_batch_size == future_batch_size, (
            "Batch sizes do not match between static and future covariate inputs: "
            f"``static_input`` batch_size={static_batch_size}, "
            f"``future_covariate_input`` batch_size={future_batch_size}."
        )







