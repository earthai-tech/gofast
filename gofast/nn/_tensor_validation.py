# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

from typing import List, Tuple, Optional, Union, Dict, Any
from ..utils.deps_utils import ensure_pkg 
from ..compat.tf import optional_tf_function, suppress_tf_warnings
from ..compat.tf import TFConfig  

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
else:
    config = TFConfig()
    # Enable compatibility mode for ndim
    config.compat_ndim_enabled = True  

@ensure_pkg(
    'tensorflow',
    extra="Need 'tensorflow' for this function to proceed"
)
def validate_anomaly_scores(
    anomaly_config: Optional[Dict[str, Any]],
    forecast_horizon: int,
) -> Optional[tf.Tensor]:
    """
    Validates and processes the ``anomaly_scores`` in the provided 
    `anomaly_config` dictionary.

    Parameters:
    - ``anomaly_config`` (Optional[`Dict[str, Any]`]): 
        Dictionary that may contain:
            - 'anomaly_scores': Precomputed anomaly scores tensor.
            - 'anomaly_loss_weight': Weight for anomaly loss.
    - ``forecast_horizon`` (int): 
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
        if len(anomaly_scores.shape) != 2:
            raise ValueError(
                f"`anomaly_scores` must be a 2D tensor with shape "
                f"(batch_size, forecast_horizons), but got "
                f"{len(anomaly_scores.shape)}D tensor."
            )

        # Validate that the second dimension matches `forecast_horizons`
        if anomaly_scores.shape[1] != forecast_horizon:
            raise ValueError(
                f"`anomaly_scores` second dimension must be "
                f"{forecast_horizon}, but got "
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
def validate_tft_inputs(
    inputs : Union[List[Any], Tuple[Any, ...]],
    dynamic_input_dim : int,
    static_input_dim : Optional[int] = None,
    future_covariate_dim : Optional[int] = None,
    error: str = 'raise'
) -> Tuple[tf.Tensor, Optional[tf.Tensor], Optional[tf.Tensor]]:
    """
    Validate and process the input tensors for TFT (Temporal Fusion
    Transformer) models in a consistent manner.

    The function enforces that ``dynamic_input_dim`` (past inputs)
    is always provided, while ``static_input_dim`` and 
    ``future_covariate_dim`` can be `None`. Depending on how many 
    items are in `inputs`, this function decides which item 
    corresponds to which tensor (past, static, or future). It also 
    converts each valid item to a :math:`\\text{tf.float32}` tensor, 
    verifying shapes and optionally raising or warning if invalid 
    conditions occur.

    Parameters
    ----------
    inputs : 
        list or tuple of input items. 
        - If length is 1, interpret as only dynamic inputs.
        - If length is 2, interpret second item either as static 
          or future inputs, depending on whether 
          ``static_input_dim`` or ``future_covariate_dim`` is set.
        - If length is 3, interpret them as 
          (past_inputs, future_inputs, static_inputs) in order.
    dynamic_input_dim : int
        Dimensionality of the dynamic (past) inputs. This is 
        mandatory for the TFT model.
    static_input_dim : int, optional
        Dimensionality of static inputs. If not `None`, expects 
        a second or third item in ``inputs`` to be assigned 
        as static inputs.
    future_covariate_dim : int, optional
        Dimensionality of future covariates. If not `None`, 
        expects a second or third item in ``inputs`` to be 
        assigned as future inputs.
    error : str, default='raise'
        Error-handling strategy if invalid conditions arise.
        - `'raise'` : Raise a `ValueError` upon invalid usage.
        - `'warn'`  : Issue a warning and proceed.
        - `'ignore'`: Silence the issue and proceed (not 
          recommended).

    Returns
    -------
    tuple of tf.Tensor
        Returns a three-element tuple 
        (past_inputs, future_inputs, static_inputs). 
        - `past_inputs` is always present.
        - `future_inputs` or `static_inputs` may be `None` if 
          not provided or `None` in shape.

    Notes
    -----
    If the length of `inputs` is three but one of 
    ``static_input_dim`` or ``future_covariate_dim`` is `None`, 
    then based on ``error`` parameter, a `ValueError` is raised, 
    a warning is issued, or the issue is silently ignored.
    
    .. math::
        \\text{past\\_inputs} \\in 
            \\mathbb{R}^{B \\times T \\times \\text{dynamic\\_input\\_dim}}
        \\quad
        \\text{future\\_inputs} \\in 
            \\mathbb{R}^{B \\times T' \\times \\text{future\\_covariate\\_dim}}
        \\quad
        \\text{static\\_inputs} \\in 
            \\mathbb{R}^{B \\times \\text{static\\_input\\_dim}}
            
    Examples
    --------
    >>> from gofast.nn._tensor_validation import validate_tft_inputs
    >>> import tensorflow as tf
    >>> # Example with only past (dynamic) inputs
    >>> single_input = tf.random.normal([8, 24, 10])  # batch=8, time=24
    >>> past, fut, stat = validate_tft_inputs(
    ...     [single_input], dynamic_input_dim=10
    ... )
    >>> print(past.shape)
    (8, 24, 10)
    >>> print(fut, stat)  # None, None

    >>> # Example with two inputs: dynamic past and static
    >>> dynamic_in = tf.random.normal([8, 24, 20])
    >>> static_in  = tf.random.normal([8, 5])
    >>> past, fut, stat = validate_tft_inputs(
    ...     [dynamic_in, static_in],
    ...     dynamic_input_dim=20,
    ...     static_input_dim=5
    ... )
    >>> print(past.shape, stat.shape)
    (8, 24, 20) (8, 5)
    >>> print(fut)  # None

    See Also
    --------
    Other internal functions that manipulate or validate 
    TFT inputs.

    References
    ----------
    .. [1] Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2019).
           Temporal Fusion Transformers for Interpretable
           Multi-horizon Time Series Forecasting.
    """

    # 1) Basic checks and shape verifications.
    if not isinstance(inputs, (list, tuple)):
        inputs= [inputs] # When single input is provided
        msg = ("`inputs` must be a list or tuple, got "
               f"{type(inputs)} instead.")
        if error == 'raise':
            raise ValueError(msg)
        elif error == 'warn':
            warnings.warn(msg)
        # if error=='ignore', do nothing

    num_inputs = len(inputs)

    # 2) Convert each item to tf.float32 and gather shapes
    def to_float32_tensor(x: Any) -> tf.Tensor:
        """Convert x to tf.float32 tensor."""
        tensor = tf.convert_to_tensor(x)
        if tensor.dtype != tf.float32:
            tensor = tf.cast(tensor, tf.float32)
        return tensor

    # Initialize placeholders
    past_inputs: Optional[tf.Tensor] = None
    future_inputs : Optional[tf.Tensor] = None
    static_inputs : Optional[tf.Tensor] = None

    # 3) Assign based on how many items are in `inputs`
    if num_inputs == 1:
        # Only dynamic/past inputs
        past_inputs = to_float32_tensor(inputs[0])

    elif num_inputs == 2:
        # We have past + either static or future
        # Decide based on static_input_dim / future_covariate_dim
        past_inputs = to_float32_tensor(inputs[0])
        second_data = to_float32_tensor(inputs[1])

        if static_input_dim is not None and future_covariate_dim is None:
            # second_data is static
            static_inputs = second_data
        elif static_input_dim is None and future_covariate_dim is not None:
            # second_data is future
            future_inputs = second_data
        else:
            # ambiguous or invalid
            msg = ("With two inputs, must have either "
                   "`static_input_dim` or `future_covariate_dim` "
                   "set, but not both or neither.")
            if error == 'raise':
                raise ValueError(msg)
            elif error == 'warn':
                warnings.warn(msg)
            # if error == 'ignore', do nothing

    elif num_inputs == 3:
        # We have past + future + static
        # Check if both static_input_dim and future_covariate_dim
        # are defined
        if (static_input_dim is None or future_covariate_dim is None):
            msg = ("Expect three inputs for past, future, "
                   "and static. But one of `static_input_dim` "
                   "or `future_covariate_dim` is None.")
            if error == 'raise':
                raise ValueError(msg)
            elif error == 'warn':
                warnings.warn(msg)
            # if error == 'ignore', do nothing

        past_inputs   = to_float32_tensor(inputs[0])
        future_inputs = to_float32_tensor(inputs[1])
        static_inputs = to_float32_tensor(inputs[2])

    else:
        # Invalid length
        msg = (f"`inputs` has length {num_inputs}, but only 1, 2, or 3 "
               "items are supported.")
        if error == 'raise':
            raise ValueError(msg)
        elif error == 'warn':
            warnings.warn(msg)
        # if error == 'ignore', do nothing

    # 4) Additional shape checks (e.g., batch size consistency).
    non_null_tensors = [
        x for x in [past_inputs, future_inputs, static_inputs] 
        if x is not None
    ]

    # If we have at least one non-None tensor, let's define a reference
    # batch size from the first. We'll do a static shape check if 
    # possible. If shape[0] is None, we do a dynamic check with tf.shape().
    if non_null_tensors:
        # For simplicity, let's define a function to get batch size.
        # If static shape is None, we fallback to tf.shape(x)[0].
        def get_batch_size(t: tf.Tensor) -> Union[int, tf.Tensor]:
            """Return the first-dim batch size, static if available."""
            if t.shape.rank and t.shape[0] is not None:
                return t.shape[0]  # static shape
            return tf.shape(t)[0]  # fallback to dynamic

        # Reference batch size
        ref_batch_size = get_batch_size(non_null_tensors[0])

        # Check all other non-null items
        for t in non_null_tensors[1:]:
            batch_size = get_batch_size(t)
            # We compare them in a consistent manner. If either
            # is a Tensor, we rely on tf.equal or a python check 
            # if both are python ints. We'll do a python approach 
            # if they're both int, else a tf.cond approach if needed.
            if (isinstance(ref_batch_size, int) and 
                isinstance(batch_size, int)):
                # Both are static
                if ref_batch_size != batch_size:
                    msg = (f"Inconsistent batch sizes among inputs. "
                           f"Got {ref_batch_size} vs {batch_size}.")
                    if error == 'raise':
                        raise ValueError(msg)
                    elif error == 'warn':
                        warnings.warn(msg)
                    # if error=='ignore', do nothing
            else:
                # At least one is dynamic. We'll do a tf-level check.
                # In eager mode, we can still evaluate it directly. 
                # Let's do so carefully.
                are_equal = tf.reduce_all(
                    tf.equal(ref_batch_size, batch_size)
                )
                if not bool(are_equal.numpy()):
                    msg = ("Inconsistent batch sizes among inputs. "
                           "Got a mismatch in dynamic shapes.")
                    if error == 'raise':
                        raise ValueError(msg)
                    elif error == 'warn':
                        warnings.warn(msg)
                    # if error=='ignore', do nothing

    # 5) Return the triple (past_inputs, future_inputs, static_inputs)
    return past_inputs, future_inputs, static_inputs

@optional_tf_function
def validate_xtft_inputs(
    inputs: Union[List[Any], Tuple[Any, ...]],
    dynamic_input_dim: int,
    static_input_dim: int,
    future_covariate_dim: Optional[int] = None, 
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
    if len(static_input.shape) != 2:
        raise ValueError(
            f"``static_input`` must be a 2D tensor with shape "
            f"(batch_size, static_input_dim), but got {len(static_input.shape)}D tensor."
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
    if len(dynamic_input.shape) != 3:
        raise ValueError(
            f"``dynamic_input`` must be a 3D tensor with shape "
            f"(batch_size, time_steps, dynamic_input_dim), but got "
            f"{len(dynamic_input.shape)}D tensor."
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
        if len(future_covariate_input.shape) != 3:
            raise ValueError(
                f"``future_covariate_input`` must be a 3D tensor with shape "
                f"(batch_size, time_steps, future_covariate_dim), but got "
                f"{len(future_covariate_input.shape)}D tensor."
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







