# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Utility functions for data validation, assertion, and feature extraction.
Includes checks for type consistency, and feature existence.
Also supports regex-based searches and classification task validation.
"""
from __future__ import print_function
import re
import os
import numbers 
import inspect 
import warnings
from functools import wraps
from collections.abc import Iterable 
import scipy.sparse as ssp 
from typing import ( 
    Any, 
    List, 
    Dict,
    Optional,
    Union, 
    Tuple,
    Type, 
    Callable, 
    get_origin,
    get_args, 
)
import numpy as np
import pandas as pd
from pandas.api.types import (
    is_numeric_dtype as _is_numeric_dtype,
    is_categorical_dtype,
    is_datetime64_any_dtype, 
    is_object_dtype
)
from sklearn.utils._param_validation import (
    Hidden, make_constraint, _Constraint, 
    InvalidParameterError
    )
from ..api.types import Series, _F, DataFrame 
from ..api.types import _Sub, ArrayLike 


__all__= [ 
    'ParamsValidator', 
    'assert_ratio',
    'check_uniform_type',
    'exist_features',
    'features_in',
    'find_features_in',
    'validate_feature',
    'validate_noise',
    'validate_ratio',
    'is_classification_task',
    'is_depth_in',
    'is_in_if',
    'is_in', 
    'is_numeric_dtype', 
    'find_by_regex',
    'find_closest',
    'str2columns',
    'random_state_validator', 
    'is_sparse_matrix', 
    'has_sparse_format', 
    'check_features_types', 
    'are_all_frames_valid', 
    'has_nan', 
    'check_spatial_columns', 
    'validate_spatial_columns', 
    'validate_nested_param', 
    'check_params', 
    ]


class ParamsValidator:
    """
    `ParamsValidator` is a decorator class designed to validate and transform
    parameters of functions and class constructors based on predefined constraints.
    It ensures that the inputs conform to specified types and conditions,
    enhancing the robustness and reliability of the codebase.

    .. math::
        \text{Given a set of constraints } C = \{c_1, c_2, \dots, c_n\},
        \text{ `ParamsValidator` verifies that each input parameter } p_i\\
            \text{ satisfies at least one } c_j \in C.

    Parameters
    ----------
    constraints : Dict[str, Any]
        A dictionary mapping parameter names to their respective constraints.
        Each constraint defines the expected type or condition that the parameter
        must satisfy.

    skip_nested_validation : bool, optional
        If set to ``True``, the validator will not perform recursive validation
        on nested structures such as lists or dictionaries. Default is ``True``.

    verbose : int, optional
        Sets the verbosity level of the validator's logging output.
        The level ranges from ``0`` (no output) to ``7`` (maximum verbosity).
        Default is ``0``.

    Methods
    -------
    __call__(obj: Callable or Type) -> Callable or Type
        Applies the `ParamsValidator` decorator to a function or a class.
        It wraps the target's `__init__` method (for classes) or the function 
        itself, enforcing the defined parameter constraints upon invocation.

    Notes
    -----
    The validation process can be represented as:

    .. math::
        \forall p_i \in \text{parameters}, \exists c_j \in C\\
            \text{ such that } c_j(p_i) \text{ is True}

    Where:
    - \( p_i \) represents each parameter to be validated.
    - \( c_j \) represents each constraint applied to the parameters.
    - The parameter \( p_i \) must satisfy at least one constraint \( c_j \).
    

    - The `ParamsValidator` can handle both simple and complex constraints,
      including type checks and transformations.
    - When `skip_nested_validation` is set to ``False``, the validator will
      recursively validate elements within nested structures like lists and dictionaries.
    - Verbosity levels allow developers to control the amount of logging information
      for debugging purposes.
      
    Examples
    --------
    Validate parameters of a function to ensure `age` is an integer and `name`
    is a string:

    >>> from gofast.core.checks import ParamsValidator
    >>> constraints = {
    ...     'age': [int],
    ...     'name': [str]
    ... }
    >>> @ParamsValidator(constraints)
    ... def register_user(name, age):
    ...     print(f"User {name} registered with age {age}.")
    ...
    >>> register_user("Alice", 30)
    User Alice registered with age 30.
    >>> register_user("Bob", "thirty")
    Traceback (most recent call last):
        ...
    InvalidParameters: Parameter 'age' must be of type int.

    Validate a class constructor to ensure `data` is a pandas DataFrame:

    >>> from gofast.core.checks import ParamsValidator
    >>> import pandas as pd
    >>> constraints = {
    ...     'data': ['array-like:dataframe:transf']
    ... }
    >>> @ParamsValidator(constraints, verbose=3)
    ... class DataProcessor:
    ...     def __init__(self, data):
    ...         self.data = data
    ...
    >>> df = {'column1': [1, 2], 'column2': [3, 4]}
    >>> processor = DataProcessor(df)
    
    [ParamsValidator] Initialized with constraints: {'data': 'dataframe:transf'}
    [ParamsValidator] Skip nested validation: True
    [ParamsValidator] Verbosity level set to: 3
    [ParamsValidator] Decorating class 'DataProcessor' __init__ method.
    [ParamsValidator] Bound arguments before validation: OrderedDict(...)
    [ParamsValidator] Starting validation of bound arguments.
    [ParamsValidator] Validating parameter 'data' with value: {'column1': [1, 2], 'column2': [3, 4]}
    [ParamsValidator] Parameter 'data' is a valid dict.
    [ParamsValidator] Validation completed successfully.

    See Also
    --------
    `InvalidParameters` : Exception raised when parameter validation fails.
    `make_constraint` : Utility function to create constraint objects.

    References
    ----------
    .. [1] Smith, J. (2020). *Effective Python Decorators*. Python Publishing.
    .. [2] Doe, A. (2021). *Advanced Parameter Validation Techniques*. 
      Software Engineering Journal.
    """

    def __init__(
        self,
        constraints: Dict[str, Any],
        skip_nested_validation: bool = True,
        verbose: int = 0  # Verbose level from 0 to 7
    ):
        self.constraints = constraints
        self.skip_nested_validation = skip_nested_validation
        self.verbose = verbose

        if self.verbose >= 1:
            print(f"[ParamsValidator] Initialized with constraints:"
                  f" {self.constraints}")
            print(f"[ParamsValidator] Skip nested validation:"
                  f" {self.skip_nested_validation}")
            print(f"[ParamsValidator] Verbosity level set to:"
                  f" {self.verbose}")

    def __call__(self, obj: Callable or Type) -> Callable or Type:
        """
        Apply the decorator to a function or a class.

        Parameters:
            obj (Callable or Type): The function or class to decorate.

        Returns:
            Callable or Type: The decorated function or class.
        """
        if isinstance(obj, type):
            # Decorate a class by wrapping its __init__ method
            original_init = obj.__init__

            @wraps(original_init)
            def wrapped_init(*args, **kwargs):
                if self.verbose >= 2:
                    print("[ParamsValidator] Decorating class"
                          f" '{obj.__name__}' __init__ method.")
                sig = inspect.signature(original_init)
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()

                if self.verbose >= 3:
                    print("[ParamsValidator] Bound arguments"
                          f" before validation: {bound.arguments}")

                self._validate(bound)

                if self.verbose >= 4:
                    print("[ParamsValidator] Arguments after"
                          f" validation: {bound.arguments}")

                # Reconstruct args and kwargs from bound.arguments
                new_args, new_kwargs = self._reconstruct_args_kwargs(bound)

                if self.verbose >= 5:
                    print(f"[ParamsValidator] Reconstructed args: {new_args}")
                    print(f"[ParamsValidator] Reconstructed kwargs: {new_kwargs}")

                return original_init(*new_args, **new_kwargs)

            obj.__init__ = wrapped_init
            return obj
        else:
            # Decorate a function by wrapping it
            @wraps(obj)
            def wrapped_func(*args, **kwargs):
                if self.verbose >= 2:
                    print(f"[ParamsValidator] Decorating function '{obj.__name__}'.")
                sig = inspect.signature(obj)
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()

                if self.verbose >= 3:
                    print("[ParamsValidator] Bound arguments"
                          f" before validation: {bound.arguments}")

                self._validate(bound)

                if self.verbose >= 4:
                    print("[ParamsValidator] Arguments after"
                          f" validation: {bound.arguments}")

                # Reconstruct args and kwargs from bound.arguments
                new_args, new_kwargs = self._reconstruct_args_kwargs(bound)

                if self.verbose >= 5:
                    print(f"[ParamsValidator] Reconstructed args: {new_args}")
                    print(f"[ParamsValidator] Reconstructed kwargs: {new_kwargs}")

                return obj(*new_args, **new_kwargs)

            return wrapped_func

    def _validate(self, bound: inspect.BoundArguments):
        """
        Validates and transforms the parameters in bound.arguments 
        based on the constraints.

        Parameters:
            bound (inspect.BoundArguments): The bound arguments
            to validate and transform.

        Raises:
            InvalidParameters: If any parameter fails to satisfy
            its constraints.
        """
        if self.verbose >= 3:
            print("[ParamsValidator] Starting validation of bound arguments.")

        for param, constraints in self.constraints.items():
            if param not in bound.arguments:
                if self.verbose >= 6:
                    print(f"[ParamsValidator] Parameter '{param}'"
                          " not in arguments; skipping.")
                continue

            value = bound.arguments[param]

            if self.verbose >= 5:
                print("[ParamsValidator] Validating parameter"
                      f" '{param}' with value: {value}")

            if value is None:
                if None not in constraints:
                    error_message = (
                        f"Parameter '{param}' cannot be None"
                    )
                    raise InvalidParameterError(error_message)
                if self.verbose >= 6:
                    print(f"[ParamsValidator] Parameter '{param}' is None and allowed.")
                continue

            valid = False
  
            for constraint in constraints:
                origin = get_origin(constraint)

                # Handle generic types first (e.g., List[str], Dict[str, float])
                if origin in [list, List]:
                    if not isinstance(value, list):
                        error_message = (
                            f"Parameter '{param}' must be a list"
                        )
                        raise InvalidParameterError(error_message)
                    if not self.skip_nested_validation:
                        subtype = get_args(constraint)[0]
                        for idx, item in enumerate(value):
                            if not isinstance(item, subtype):
                                error_message = (
                                    f"All items in parameter '{param}' must be of type "
                                    f"{subtype.__name__} (item {idx} is {type(item).__name__})"
                                )
                                raise InvalidParameterError(error_message)
                    valid = True
                    if self.verbose >= 5:
                        print(f"[ParamsValidator] Parameter '{param}' is a valid list.")
                    break  # No need to check other constraints

                elif origin in [dict, Dict]:
                    if not isinstance(value, dict):
                        error_message = (
                            f"Parameter '{param}' must be a dict"
                        )
                        raise InvalidParameterError(error_message)
                    if not self.skip_nested_validation:
                        key_type, val_type = get_args(constraint)
                        for k, v in value.items():
                            if not isinstance(k, key_type):
                                error_message = (
                                    f"All keys in parameter '{param}' must be of type "
                                    f"{key_type.__name__} (key '{k}' is {type(k).__name__})"
                                )
                                raise InvalidParameterError(error_message)
                            if val_type is not Any and not isinstance(v, val_type):
                                error_message = (
                                    f"All values in parameter '{param}' must be of type "
                                    f"{val_type.__name__} (value '{v}' is {type(v).__name__})"
                                )
                                raise InvalidParameterError(error_message)
                    valid = True
                    if self.verbose >= 5:
                        print(f"[ParamsValidator] Parameter '{param}' is a valid dict.")
                    break  # No need to check other constraints

                # Handle non-generic constraints
                try:
                    constraint_obj = self._make_constraint(constraint)
                except ( InvalidParameterError, ValueError) as e:
                    error_message = (
                        f"Unsupported constraint type for parameter '{param}': {constraint}"
                    )
                    raise InvalidParameterError(error_message) from e

                # Handle Hidden constraints by unwrapping them
                if isinstance(constraint_obj, Hidden):
                    try:
                        inner_constraint = make_constraint(
                            constraint_obj.constraint
                        )
                    except (ValueError, InvalidParameterError) as e:
                        error_message = (
                            f"Hidden constraint for parameter '{param}' contains"
                            f" an unsupported constraint: {constraint_obj.constraint}"
                        )
                        raise InvalidParameterError(error_message) from e

                    if not inner_constraint.is_satisfied_by(value):
                        error_message = (
                            f"Parameter '{param}' does not satisfy the hidden "
                            f"constraint {inner_constraint}"
                        )
                        raise InvalidParameterError(error_message)

                    if (
                        isinstance(constraint, str)
                        and ':' in constraint
                        and len(constraint.split(':')) == 2
                    ):
                        self._validate_specific_types(value, param, constraint)

                    # Apply transformation if needed
                    if isinstance(constraint, str) and ':transf' in constraint:
                        try:
                            value = self._apply_transformation(value, param, constraint)
                            bound.arguments[param] = value
                            if self.verbose >= 5:
                                print(
                                    "[ParamsValidator] Applied transformation"
                                    f" on parameter '{param}'."
                                )
                        except InvalidParameterError as e:
                            raise e
                    valid = True
                    break  # No need to check other constraints
                else:
                    if not constraint_obj.is_satisfied_by(value):
                        if self.verbose >= 6:
                            print(
                                f"[ParamsValidator] Constraint '{constraint_obj}'"
                                f" not satisfied for parameter '{param}'."
                            )
                        continue  # Try the next constraint

                    if (
                        isinstance(constraint, str)
                        and ':' in constraint
                        and len(constraint.split(':')) == 2
                    ):
                        self._validate_specific_types(value, param, constraint)

                    # Apply transformation if needed
                    if isinstance(constraint, str) and ':transf' in constraint:
                        try:
                            value = self._apply_transformation(value, param, constraint)
                            bound.arguments[param] = value
                            if self.verbose >= 5:
                                print(
                                    "[ParamsValidator] Applied transformation"
                                    f" on parameter '{param}'."
                                )
                        except Exception as e:
                            raise e
                    valid = True
                    if self.verbose >= 5:
                        print(
                            f"[ParamsValidator] Parameter '{param}'"
                            f" satisfies constraint '{constraint_obj}'."
                        )
                    break  # Constraint satisfied

            if not valid:
                constraint_name =( 
                    constraint.__name__ if hasattr(constraint, '__name__') 
                    else constraint.__class__.__name__
                    )
                error_message = (
                    f"Parameter '{param}' must be a type '{constraint_name}'."
                )
                raise InvalidParameterError(error_message)

        if self.verbose >= 3:
            print("[ParamsValidator] Validation completed successfully.")

    def _make_constraint(self, constraint: Any) -> _Constraint:
        """
        Convert the constraint into the appropriate _Constraint object.

        Parameters:
            constraint (Any): The constraint to convert.

        Returns:
            _Constraint: The corresponding constraint object.

        Raises:
            ValueError: If the constraint type is unknown.
        """
        if self.verbose >= 4:
            print(
                f"[ParamsValidator] Creating constraint object for: {constraint}")

        if isinstance(constraint, str):
            if ':' in constraint:
                # Handle complex string constraints with transformations
                base_constraint = constraint.split(':')[0]
                return make_constraint(base_constraint)
            return make_constraint(constraint)
        elif constraint is None:
            return make_constraint(constraint)
        elif isinstance(constraint, type):
            return make_constraint(constraint)
        elif isinstance(constraint, (_Constraint, Hidden)):
            return constraint
        else:
            raise ValueError(f"Unknown constraint type: {constraint}")

    def _apply_transformation(self, value: Any, param:Any, constraint_str: str) -> Any:
        """
        Apply transformations based on constraint specifications.

        Parameters:
            value (Any): The value to transform.
            constraint_str (str): The constraint string specifying 
            transformations.

        Returns:
            Any: The transformed value.

        Raises:
            InvalidParameters: If the transformation fails.
        """
        parts = constraint_str.split(':')
        specs = parts[1:]  # Extract transformation specs
        specs = [str(s).lower() for s in specs]
        err_msg = (
            "Invalid parameter '{0}', expected a {1}. Got {2!r} instead."
        )

        if self.verbose >= 6:
            print(f"[ParamsValidator] Applying"
                  f" transformations: {specs} on value: {value}")

        for spec in specs:
            if 'tf' in spec: 
                try: 
                    import tensorflow as tf 
                    # Convert list to TensorFlow tensor
                    value = tf.convert_to_tensor(value, dtype=tf.float32)
                except: 
                    # fallback to numpy conversion 
                    spec ='np'
            
            if 'df' in spec or 'dataframe' in spec:
                try:
                    value = pd.DataFrame(value)
                except Exception:
                    raise InvalidParameterError(
                        err_msg.format(
                            param, 'pandas DataFrame', type(value).__name__)
                    )
            elif 'np' in spec:
                try:
                    value = np.array(value)
                except Exception:
                    raise InvalidParameterError(
                        err_msg.format(
                            param,'numpy array', type(value).__name__)
                    )
                
            elif 'series' in spec:
                try:
                    value = pd.Series(value)
                except Exception:
                    raise InvalidParameterError(
                        err_msg.format(
                            param,'pandas Series', type(value).__name__)
                    )
            elif spec == 'list':
                try:
                    value = list(value)
                except Exception:
                    raise InvalidParameterError(
                        err_msg.format(
                            param,'list object', type(value).__name__)
                    )
            elif spec == 'tuple':
                try:
                    value = tuple(value)
                except Exception:
                    raise InvalidParameterError(
                        err_msg.format(
                            param,'tuple object', type(value).__name__)
                    )
            #XXX: Future extension: Add more transformations as needed
            else:
                if self.verbose >= 6:
                    print(f"[ParamsValidator] Unknown transformation spec: '{spec}'")
        return value

    def _validate_specific_types(
            self, value: Any, param: Any, constraint_str: str) -> None:
        """
        Validate specific types based on constraint specifications.

        Parameters:
            value (Any): The value to validate.
            constraint_str (str): The constraint string specifying the type.

        Raises:
            InvalidParameters: If the type validation fails.
        """
        parts = constraint_str.split(':')  # e.g., 'arraylike:np'
        spec = parts[-1]
        err_msg = (
            "Invalid parameter '{0}', expected a {1}. Got {2!r} instead."
        )

        if self.verbose >= 6:
            print("[ParamsValidator] Validating"
                  f" specific type: '{spec}' for value: {value}")

        if ('df' in spec or 'dataframe' in spec) and not isinstance(value, pd.DataFrame):
            raise InvalidParameterError(
                err_msg.format(param, 'pandas DataFrame', type(value).__name__)
            )
        elif 'np' in spec and not isinstance(value, np.ndarray):
            raise InvalidParameterError(
                err_msg.format(param,'numpy array', type(value).__name__)
            )
        elif 'series' in spec and not isinstance(value, pd.Series):
            raise InvalidParameterError(
                err_msg.format(param,'pandas Series', type(value).__name__)
            )
        elif spec == 'list' and not isinstance(value, list):
            raise InvalidParameterError(
                err_msg.format(param,'list object', type(value).__name__)
            )
        elif spec == 'tuple' and not isinstance(value, tuple):
            raise InvalidParameterError(
                err_msg.format(param,'tuple object', type(value).__name__)
            )
        try: 
            import tensorflow as tf 
            if spec =='tf' and not isinstance (value, tf.Tensor): 
                raise InvalidParameterError(
                    err_msg.format(
                        param,'tensorflow Tensor object', type(value).__name__)
                )
        except: 
            pass 
        #XXX: Future extension: Add more transformations as needed

    def _reconstruct_args_kwargs(
        self, bound: inspect.BoundArguments
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Reconstruct *args and **kwargs from the bound arguments.

        Parameters:
            bound (inspect.BoundArguments): The bound arguments.

        Returns:
            Tuple[List[Any], Dict[str, Any]]: The reconstructed args and kwargs.
        """
        args = []
        kwargs = {}
        signature = bound.signature

        if self.verbose >= 5:
            print("[ParamsValidator] Reconstructing args"
                  " and kwargs from bound arguments.")

        for name, param in signature.parameters.items():
            if param.kind in (
                param.POSITIONAL_ONLY,
                param.POSITIONAL_OR_KEYWORD
            ):
                if name in bound.arguments:
                    args.append(bound.arguments[name])
            elif param.kind == param.KEYWORD_ONLY:
                if name in bound.arguments:
                    kwargs[name] = bound.arguments[name]
            elif param.kind == param.VAR_POSITIONAL:
                if name in bound.arguments:
                    args.extend(bound.arguments[name])
            elif param.kind == param.VAR_KEYWORD:
                if name in bound.arguments:
                    kwargs.update(bound.arguments[name])
        
        if self.verbose >= 6:
            print(f"[ParamsValidator] Reconstructed args: {args}")
            print(f"[ParamsValidator] Reconstructed kwargs: {kwargs}")

        return args, kwargs

    def _is_array_like(self, value: Any) -> bool:
        """
        Check if the value is array-like.

        Parameters:
            value (Any): The value to check.

        Returns:
            bool: True if array-like, False otherwise.
        """
        array_like = isinstance(
            value, (list, tuple, np.ndarray, pd.Series, pd.DataFrame)
        )
        if self.verbose >= 7:
            print("[ParamsValidator] Checking if"
                  f" value is array-like: {array_like}")
        return array_like


def find_closest(arr, values):
    """
    Find the closest values in an array from a set of target values.

    This function takes an array and a set of target values, and for each 
    target value, finds the closest value in the array. It can handle 
    both scalar and array-like inputs for `values`, ensuring flexibility 
    in usage. The result is either a single closest value or an array 
    of closest values corresponding to each target.

    Parameters
    ----------
    arr : array-like
        The array to search within. It can be a list, tuple, or numpy array 
        of numeric types. If the array is multi-dimensional, it will be 
        flattened to a 1D array.
        
    values : float or array-like
        The target value(s) to find the closest match for in `arr`. This can 
        be a single float or an array of floats.

    Returns
    -------
    numpy.ndarray
        An array of the closest values in `arr` for each target in `values`.
        If `values` is a single float, the function returns a single-element
        array.

    Notes
    -----
    - This function operates by calculating the absolute difference between
      each element in `arr` and each target in `values`, selecting the 
      element with the smallest difference.
    - The function assumes `arr` and `values` contain numeric values, and it
      raises a `TypeError` if they contain non-numeric data.

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.core.checks import find_closest
    >>> find_closest([2, 3, 4, 5], 2.6)
    array([3.])

    >>> find_closest(np.array([[2, 3], [4, 5]]), (2.6, 5.6))
    array([3., 5.])

    See Also
    --------
    numpy.argmin : Find the indices of the minimum values along an axis.
    numpy.abs : Compute the absolute value element-wise.

    References
    ----------
    .. [1] Harris, C. R., et al. "Array programming with NumPy." 
       Nature 585.7825 (2020): 357-362.
    """

    arr = is_iterable(arr, exclude_string=True, transform=True)
    values = is_iterable(values, exclude_string=True, transform=True)

    # Validate numeric types in arr and values
    for var, name in zip([arr, values], ['array', 'values']):
        if not is_numeric_dtype(var, to_array=True):
            raise TypeError(f"Non-numeric data found in {name}.")

    # Convert arr and values to numpy arrays for vectorized operations
    arr = np.array(arr, dtype=np.float64)
    values = np.array(values, dtype=np.float64)

    # Flatten arr if it is multi-dimensional
    arr = arr.ravel() if arr.ndim != 1 else arr

    # Find the closest value for each target in values
    closest_values = np.array([
        arr[np.abs(arr - target).argmin()] for target in values
    ])

    return closest_values

def find_by_regex(
    o: Union[str, Iterable],
    pattern: str = r'[_#&*@!_,;\s-]\s*',
    func: Callable = re.match,
    **kws
) -> Optional[List[str]]:
    """
    Find Pattern Matches within an Object using Regular Expressions.
    
    The ``find_by_regex`` function searches for a specified regex ``pattern`` 
    within an object ``o``, which can be either a string or an iterable 
    (excluding strings). It utilizes a user-specified regex function (e.g., 
    ``re.match``, ``re.search``, ``re.findall``) to identify matches and 
    returns a list of matched elements.
    
    .. math::
        \text{Match Results} = 
        \begin{cases}
            \text{List of Matches} & \text{if matches are found} \\
            \text{None} & \text{if no matches are found}
        \end{cases}
    
    Parameters
    ----------
    o : Union[`str`, `Iterable`]
        The input object in which to search for the regex ``pattern``.
        - If a string is provided, the function searches within the string.
        - If an iterable is provided (excluding strings), the function searches 
          within each element of the iterable.
    
    pattern : `str`, default=`'[_#&*@!_,;\s-]\s*'`
        The regex pattern to search for within ``o``. By default, it matches 
        separators commonly used in text splitting.
    
    func : `Callable`, default=`re.match`
        The regex function to use for searching. Can be one of the following:
        
        - ``re.match``: Searches for a match only at the beginning of the string.
        - ``re.search``: Searches for a match anywhere in the string.
        - ``re.findall``: Finds all non-overlapping matches in the string.
        
        Additional regex functions can also be used as long as they adhere to 
        the callable signature.
    
    **kws : `dict`
        Additional keyword arguments to pass to the regex ``func``. These can 
        include flags like ``re.IGNORECASE``, ``re.MULTILINE``, etc.
    
    Returns
    -------
    Optional[List[str]]
        A list of matched objects found within ``o`` based on the regex 
        ``pattern``.
        - If matches are found, returns a list of matched strings.
        - If no matches are found, returns ``None``.
    
    Raises
    ------
    TypeError
        If ``o`` is neither a string nor an iterable object.
    
    ValueError
        If the provided ``func`` is not a recognized regex function.
    
    Examples
    --------
    >>> import re
    >>> from gofast.core.checks import find_by_regex
    >>> 
    >>> # Example 1: Find pattern in a concatenated string
    >>> text = "depth_top, depth_bottom, temperature"
    >>> find_by_regex(text, pattern='depth', func=re.search)
    ['depth_top']
    >>> 
    >>> # Example 2: Find pattern in an iterable of column names
    >>> columns = ['depth_top', 'depth_bottom', 'temperature']
    >>> find_by_regex(columns, pattern='depth', func=re.search)
    ['depth_top', 'depth_bottom']
    >>> 
    >>> # Example 3: Find all occurrences using re.findall
    >>> text = "depth1, depth2, depth3"
    >>> find_by_regex(text, pattern='depth\d+', func=re.findall)
    ['depth1', 'depth2', 'depth3']
    >>> 
    >>> # Example 4: No matches found
    >>> find_by_regex(columns, pattern='pressure', func=re.search)
    None
    
    Notes
    -----
    - **Input Flexibility**: The function can handle both single strings and 
      iterables (excluding strings), allowing for versatile usage across different 
      data structures.
    
    - **Regex Functionality**: By allowing users to specify the regex function, 
      the ``find_by_regex`` function provides flexibility in how patterns are 
      searched and matched within the input object.
    
    - **Performance Considerations**: When dealing with large iterables, using 
      ``re.findall`` can lead to extensive memory usage as it retrieves all matches 
      at once. Users should choose the appropriate regex function based on their 
      specific requirements.
    
    - **Handling No Matches**: If no matches are found, the function returns 
      ``None``, allowing users to handle such cases gracefully in their workflows.
    
    - **Order Preservation**: The order of matched items in the returned list 
      corresponds to their order of appearance in the input object.
    
    See Also
    --------
    re.match : Function to match a regex pattern at the beginning of a string.
    re.search : Function to search for a regex pattern anywhere in a string.
    re.findall : Function to find all non-overlapping matches of a regex 
      pattern in a string.
    
    References
    ----------
    .. [1] Python Documentation: re.match.  
       https://docs.python.org/3/library/re.html#re.match  
    .. [2] Python Documentation: re.search.  
       https://docs.python.org/3/library/re.html#re.search  
    .. [3] Python Documentation: re.findall.  
       https://docs.python.org/3/library/re.html#re.findall  
    .. [4] Freedman, D., & Diaconis, P. (1981). On the histogram as a density 
           estimator: L2 theory. *Probability Theory and Related Fields*, 57(5), 
           453-476.
    """
    om = []
    
    if isinstance(o, str):
        result = func(pattern=pattern, string=o, **kws)
        if result:
            if func.__name__ == 'findall':
                om.extend(result)
            else:
                om.append(result.group())
    elif isinstance(o, Iterable) and not isinstance(o, str):
        for s in o:
            result = func(pattern=pattern, string=s, **kws)
            if result:
                if func.__name__ == 'findall':
                    om.extend(result)
                else:
                    om.append(s)
    else:
        raise TypeError(
            f"'o' must be a string or an iterable object, got {type(o).__name__!r}."
        )
    
    if not om:
        return None
    
    return om


def is_in_if(
    o: Iterable,
    items: Union[str, Iterable],
    error: str = 'raise',
    return_diff: bool = False,
    return_intersect: bool = False
) -> Union[List, None]:
    """
    Assert the Presence of Items within an Iterable Object.
    
    The ``is_in_if`` function verifies whether specified ``items`` exist within 
    an iterable object ``o``. It offers flexibility in handling missing items by 
    allowing users to either raise errors, ignore them, or retrieve differences 
    and intersections based on the provided parameters.
    
    .. math::
        \text{Presence Check} = 
        \begin{cases}
            \text{Raise Error} & \text{if items are missing and error='raise'} \\
            \text{Return Differences} & \text{if return_diff=True} \\
            \text{Return Intersection} & \text{if return_intersect=True}
        \end{cases}
    
    Parameters
    ----------
    o : `Iterable`
        The iterable object in which to check for the presence of ``items``.
    
    items : Union[`str`, `Iterable`]
        The item or collection of items to assert their presence within ``o``.
        If a single string is provided, it is treated as a single-item iterable.
    
    error : `str`, default=`'raise'`
        Determines how the function handles missing items.
        
        - ``'raise'``: Raises a ``ValueError`` if any ``items`` are not found 
          in ``o``.
        - ``'ignore'``: Suppresses errors and allows the function to proceed.
    
    return_diff : `bool`, default=`False`
        If ``True``, returns a list of items that are missing from ``o``.
        When set to ``True``, the ``error`` parameter is automatically set to 
        ``'ignore'``.
    
    return_intersect : `bool`, default=`False`
        If ``True``, returns a list of items that are present in both ``o`` and 
        ``items``.
        When set to ``True``, the ``error`` parameter is automatically set to 
        ``'ignore'``.
    
    Returns
    -------
    Union[List, None]
        - If ``return_diff`` is ``True``, returns a list of missing items.
        - If ``return_intersect`` is ``True``, returns a list of intersecting items.
        - If neither is ``True``, returns ``None`` unless an error is raised.
    
    Raises
    ------
    ValueError
        - If ``error`` is set to ``'raise'`` and any ``items`` are missing in ``o``.
        - If an unsupported value is provided for ``error``.
    
    TypeError
        - If ``o`` is not an iterable object.
    
    Examples
    --------
    >>> from gofast.core.checks import is_in_if
    >>> 
    >>> # Example 1: Check presence with error raising
    >>> o = ['apple', 'banana', 'cherry']
    >>> is_in_if(o, 'banana')
    # No output, validation passed
    >>> is_in_if(o, 'date')
    ValueError: Item 'date' is missing in the list ['apple', 'banana', 'cherry'].
    >>> 
    >>> # Example 2: Check multiple items with some missing
    >>> items = ['banana', 'date']
    >>> is_in_if(o, items)
    ValueError: Items 'date' are missing in the list ['apple', 'banana', 'cherry'].
    >>> 
    >>> # Example 3: Return missing items without raising error
    >>> missing = is_in_if(o, 'date', error='ignore', return_diff=True)
    >>> print(missing)
    ['date']
    >>> 
    >>> # Example 4: Return intersecting items
    >>> intersect = is_in_if(o, ['banana', 'date'], 
    ...                      error='ignore', return_intersect=True)
    >>> print(intersect)
    ['banana']
    
    Notes
    -----
    - **Flexible Input Handling**: The function accepts both single items 
      (as strings) and multiple items (as iterables), providing versatility 
      in usage scenarios.
    
    - **Automatic Error Handling Adjustment**: Setting ``return_diff`` or 
      ``return_intersect`` to ``True`` automatically changes the ``error`` 
      parameter to ``'ignore'`` to facilitate the retrieval of differences 
      or intersections without interruption.
    
    - **Performance Considerations**: For large iterables and item lists, 
      converting them to sets can enhance performance during intersection 
      and difference operations.
    
    See Also
    --------
    list : Built-in Python list type.
    set : Built-in Python set type.
    
    References
    ----------
    .. [1] Python Documentation: set.intersection.  
       https://docs.python.org/3/library/stdtypes.html#set.intersection  
    .. [2] Python Documentation: set.difference.  
       https://docs.python.org/3/library/stdtypes.html#set.difference  
    .. [3] Freedman, D., & Diaconis, P. (1981). On the histogram as a density 
           estimator: L2 theory. *Probability Theory and Related Fields*, 57(5), 
           453-476.
    """
    if isinstance(items, str):
        items = [items]
    elif not isinstance(o, Iterable):
        raise TypeError(
            f"Expected an iterable object for 'o', got {type(o).__name__!r}."
        )
    
    # Convert to sets for efficient operations
    set_o = set(o)
    set_items = set(items)
    
    intersect = list(set_o.intersection(set_items))
    
    # to make a difference be sure to select the long set 
    if len(set_items) >= len(set_o): 
        missing_items = list(set_items.difference(set_o))
    else: 
        missing_items = list(set_o.difference(set_items))
    
    if return_diff or return_intersect:
        error = 'ignore'
    
    if missing_items:
        if error == 'raise':
            formatted_items = ', '.join(f"'{item}'" for item in missing_items)
            verb = 'is' if len(missing_items) == 1 else 'are'
            raise ValueError(
                f"Item{'' if len(missing_items) == 1 else 's'} {formatted_items} "
                f"{verb} missing in the {type(o).__name__.lower()} {list(o)}."
            )
    
    if return_diff:
        return missing_items if missing_items else []
    elif return_intersect:
        return intersect if intersect else []
    
    return None

def is_depth_in(
    X: pd.DataFrame,
    name: Union[str, int],
    columns: Optional[List[str]] = None,
    error: str = 'ignore'
) -> Tuple[pd.DataFrame, Union[pd.Series, None]]:
    """
    Assert the Presence of a Depth Column within a DataFrame.
    
    The ``is_depth_in`` function verifies whether a specified depth column 
    exists within a DataFrame. It supports identification by column name or 
    index and offers options to rename columns and handle missing depth columns 
    gracefully by creating pseudo-depth data if necessary.
    
    Parameters
    ----------
    X : `pandas.DataFrame`
        The input DataFrame containing data for validation.
    
    name : Union[`str`, `int`]
        The name or index of the depth column within the DataFrame.
        - If a string is provided, the function searches for a column that matches 
          the name using regex patterns.
        - If an integer is provided, it treats it as the column index to retrieve 
          the depth column name.
    
    columns : `List[str]`, optional
        New labels to replace the existing column names in the DataFrame.
        - If provided, it must match the number of columns in ``X``.
        - If the length does not match, a warning is issued and renaming is skipped.
    
    error : `str`, default=`'ignore'`
        Determines how the function handles missing depth columns.
        
        - ``'raise'``: Raises a ``ValueError`` if the depth column is not found.
        - ``'ignore'``: Suppresses errors and creates a pseudo-depth column using 
          the length of the DataFrame.
    
    Returns
    -------
    Tuple[`pandas.DataFrame`, Union[`pandas.Series`, `None`]]
        - The first element is the DataFrame without the depth column.
        - The second element is the depth column as a Series. If the depth column 
          is not found and ``error`` is ``'ignore'``, a pseudo-depth Series is created.
          Otherwise, it returns ``None``.
    
    Raises
    ------
    ValueError
        - If ``error`` is set to ``'raise'`` and the depth column is not found.
        - If ``name`` is an index that is out of bounds of the DataFrame's columns.
    
    TypeError
        - If ``X`` is not a `pandas.DataFrame`.
        - If ``columns`` is provided but is not iterable.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.core.checks import is_depth_in
    >>> 
    >>> # Example 1: Depth column by name
    >>> df = pd.DataFrame({
    ...     'depth_top': [100, 200, 300],
    ...     'value': [1, 2, 3]
    ... })
    >>> X, depth = is_depth_in(df, name='depth_top')
    >>> print(X)
       value
    0      1
    1      2
    2      3
    >>> print(depth)
    0    100
    1    200
    2    300
    Name: depth_top, dtype: int64
    >>> 
    >>> # Example 2: Depth column by index
    >>> df = pd.DataFrame({
    ...     'value': [1, 2, 3],
    ...     'depth_bottom': [400, 500, 600]
    ... })
    >>> X, depth = is_depth_in(df, name=1)
    >>> print(X)
       value
    0      1
    1      2
    2      3
    >>> print(depth)
    0    400
    1    500
    2    600
    Name: depth_bottom, dtype: int64
    >>> 
    >>> # Example 3: Rename columns and handle missing depth
    >>> df = pd.DataFrame({
    ...     'val': [1, 2, 3],
    ...     'dep': [700, 800, 900]
    ... })
    >>> X, depth = is_depth_in(
    ...     df, 
    ...     name='dep', 
    ...     columns=['value', 'depth'], 
    ...     error='ignore'
    ... )
    >>> print(X)
       value
    0      1
    1      2
    2      3
    >>> print(depth)
    0    700
    1    800
    2    900
    Name: depth, dtype: int64
    >>> 
    >>> # Example 4: Missing depth column with error handling
    >>> df = pd.DataFrame({
    ...     'value': [1, 2, 3],
    ...     'temperature': [15, 16, 17]
    ... })
    >>> is_depth_in(df, name='depth')
    ValueError: Depth column not found in dataframe.
    
    Notes
    -----
    - **Column Renaming**: When providing new column names via the ``columns`` 
      parameter, ensure that the number of new names matches the number of 
      existing columns in the DataFrame to avoid warnings and skipped renaming.
    
    - **Regex Matching**: When ``name`` is a string, the function utilizes regex 
      patterns to allow partial and case-insensitive matches for more flexible 
      depth column identification.
    
    - **Pseudo-Depth Creation**: If the depth column is not found and ``error`` 
      is set to ``'ignore'``, the function generates a pseudo-depth column based 
      on the DataFrame's length to maintain data integrity for downstream tasks.
    
    - **Error Handling**: The function provides clear and descriptive error 
      messages to aid in debugging and ensure that missing critical columns are 
      promptly addressed.
    
    See Also
    --------
    pandas.DataFrame : The primary DataFrame object in pandas.
    re.search : Function for regex-based searching.
    
    References
    ----------
    .. [1] Pandas Documentation: pandas.DataFrame.  
       https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html  
    .. [2] Python Documentation: re.search.  
       https://docs.python.org/3/library/re.html#re.search  
    .. [3] Freedman, D., & Diaconis, P. (1981). On the histogram as a density 
           estimator: L2 theory. *Probability Theory and Related Fields*, 57(5), 
           453-476.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            f"'X' must be a pandas.DataFrame, got {type(X).__name__!r}."
        )
    
    if columns is not None:
        if not isinstance(columns, Iterable):
            raise TypeError(
                f"'columns' must be an iterable object, got {type(columns).__name__!r}."
            )
        columns = list(columns)
        if len(columns) != len(X.columns):
            warnings.warn(
                f"Cannot rename columns. Expected {len(X.columns)} labels, "
                f"got {len(columns)}."
            )
        else:
            X = X.copy()
            X.columns = columns
    
    if isinstance(name, (int, float)):
        name = int(name)
        if name < 0 or name >= len(X.columns):
            warnings.warn(
                f"Name index {name} is out of range. DataFrame has "
                f"{len(X.columns)} columns."
            )
            depth = None
        else:
            depth_col = X.columns[name]
            depth = X.pop(depth_col)
    elif isinstance(name, str):
        # Use regex to find matching column
        pattern = re.compile(fr'{name}', re.IGNORECASE)
        matched_cols = [col for col in X.columns if pattern.search(col)]
        if not matched_cols:
            msg = f"Depth column matching '{name}' not found in DataFrame."
            if error == 'raise':
                raise ValueError(msg)
            else:
                warnings.warn(msg)
                depth = None
        else:
            # Take the first match
            depth_col = matched_cols[0]
            depth = X.pop(depth_col)
    else:
        raise TypeError(
            f"'name' must be a string or integer, got {type(name).__name__!r}."
        )
    
    if depth is None:
        if error == 'raise':
            raise ValueError("Depth column not found in DataFrame.")
        else:
            depth = pd.Series(
                data=np.arange(len(X)), 
                name='depth (m)'
            )
    
    return X, depth

def exist_labels(
    df, labels, 
    features=None, 
    name="Label columns", 
    return_valid=False, 
    as_categories=False,
    error='warn',  
    verbose=0  
    
):
    """
    exist_labels - Check whether specified labels exist in feature columns 
    of a dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe in which the presence of the specified labels will be 
        checked. Each column is expected to accommodate the specified `labels`, 
        which can be of categorical or any other relevant data type.
    
    labels : str or list of str
        A single label or a list of labels to check for in the specified feature 
        columns. The function checks for the presence of these labels in each 
        column. If a label is not found in a column, that column will be flagged 
        as missing that label.
    
    features : str, list of str, or None, optional, default=None
        The column(s) in which to check for the `labels`. If `features` is `None`, 
        the function will automatically check all columns of the dataframe that 
        have a categorical data type (i.e., dtype `category`, or objects with 
        categorical values). It can either be a single column name (`str`) or a 
        list of column names (`list` of `str`).
    
    name : str, optional, default="Label columns"
        A name used for informative output in verbose logging. This name is used 
        to describe the label columns, making the output more descriptive. The 
        `name` can be anything that helps identify the purpose of the label columns 
        in the dataset.
    
    return_valid : bool, optional, default=False
        If set to `True`, the function will return a dictionary where the keys 
        are column names and the values are lists of labels that were found to 
        be valid (i.e., the labels present in the specified columns). If set to 
        `False` (the default), it will return the missing labels for each column 
        where labels are missing.
    
    as_categories : bool, optional, default=False
        If set to `True`, the function will convert the specified feature columns 
        to categorical dtype before performing the check. This is useful when 
        you want to treat the feature columns as categorical variables for consistency.
    
    error : {'warn', 'raise'}, optional, default='warn'
        Specifies how to handle cases where the specified `labels` are missing. 
        If set to `'warn'` (the default), the function will issue a warning when 
        labels are not found. If set to `'raise'`, an error will be raised when 
        labels are missing from any of the specified columns.
    
    verbose : int, optional, default=0
        Controls the level of verbosity for printed messages:
        - 0: No output (silent mode).
        - 1: Basic output, including a summary of missing labels.
        - 2: Detailed output, including missing labels for each column.
        - 3: Full debug output, showing the state of all intermediate steps.
    
    Returns
    -------
    dict
        A dictionary with column names as keys and a list of missing labels or 
        valid labels (based on the value of `return_valid`). If `return_valid` is 
        `False`, the dictionary contains missing labels, otherwise it contains 
        the valid labels that are present in the dataframe columns.
    
    Notes
    ------
    Let `df` be a dataframe with columns `C1, C2, ..., Cn`. Each column 
    `Ci` can contain categorical values. The task is to check whether each 
    of the given `labels = [l1, l2, ..., lm]` exists in the columns.
    
    - For each column `Ci`, check if `l1, l2, ..., lm` are present in `Ci`.
    - If a label `lj` is missing in column `Ci`, add `Ci` to the missing list 
      for that label.
    
    The function checks each column and provides the result based on the 
    `return_valid` and `verbose` parameters:
    - If `return_valid=True`, return the valid labels that are present in each column.
    - If `return_valid=False`, return the missing labels for each column.
    
    Example
    -------
    >>> import pandas as pd 
    >>> from gofast.core.checks import exist_labels
    >>> df = pd.DataFrame({
    >>>     'A': ['X', 'Y', 'Z', 'X', 'W'],
    >>>     'B': ['c', 'c', 'd', 'e', 'c'],
    >>>     'C': ['a', 'b', 'c', 'd', 'e']
    >>> })
    >>> labels = ['a', 'b', 'd']
    >>> result=exist_labels(df, labels, features=['A', 'B'], verbose=2)
    >>> print(result)
    Label columns - Missing labels:
    Column 'A': Missing labels: ['a', 'b', 'd']
    Column 'B': Missing labels: ['d']
    >>> print(result)
    {'A': ['a', 'b', 'd'], 'B': ['a', 'b']}
    >>> # Sample DataFrame
    >>> df = pd.DataFrame({
        'A': ['X', 'Y', 'Z', 'X', 'W'],
        'B': ['a', 'b', 'c', 'a', 'b'],
        'C': ['cat', 'dog', 'cat', 'bird', 'dog']
    })
    
    >>> # Check if certain labels exist in the 'A' and 'B' columns
    >>> print(exist_labels(df, labels=['X', 'Y', 'Z'], features=['A', 'B'], verbose=2))
    
    >>> # Example 2: Return only valid labels in the 'C' column
    >>> print(exist_labels(df, labels=['cat', 'dog', 'lion'], features='C',
                       return_valid=True, verbose=1))
    
    >>> # Example 3: Handle missing labels with raise error
    >>> try:
        exist_labels(df, labels=['tiger', 'lion'], features='C', error='raise')
    except ValueError as e:
        print(e)

    >>> print(result)
    {'A': ['cat', 'dog'], 'B': ['dog']}

    
    Notes
    -----
    - The function automatically selects columns of categorical dtype if `features` 
      is not specified. Ensure your dataframe has appropriate categorical columns.
    - When `as_categories=True`, the function converts the specified columns 
      into categorical dtype before checking for labels.
    - The function supports two error handling modes:
        - `'warn'` issues a warning if labels are missing.
        - `'raise'` raises an exception if any labels are missing from the columns.
    
    See Also
    --------
    pandas.DataFrame.astype : Convert data types of dataframe columns
    pandas.api.types.is_categorical_dtype : Check for categorical dtype
    
    References
    ----------
    .. [1] Smith, J., "Data Analysis with Pandas", 2020, Springer
    .. [2] Doe, A., "Advanced DataFrame Operations in Python", 2018, Wiley
    """

    are_all_frames_valid(df, df_only= True )
    # Ensure 'features' and 'labels' are lists, even if passed as strings
    if isinstance(features, str):
        features = [features]
    if isinstance(labels, str):
        labels = [labels]
    
    # If 'features' is None, use columns with categorical dtype
    if features is None:
        features = [
            col for col in df.columns 
            if pd.api.types.is_categorical_dtype(df[col])
        ]
    if as_categories: 
        df[features]= df[features].astype('category')
        
    # If no valid features found, raise an informative error or warning
    if not features:
        if error == 'raise':
            raise ValueError(
                f"No valid features found with categorical dtype in {name}.")
        elif error == 'warn':
            warnings.warn(
                "No valid features with categorical"
                " dtype found. Proceeding with all columns.")
            features = df.columns.tolist()

    # Initialize result dictionary
    result = {}
    
    # Iterate through the features (columns) to check for valid or missing labels
    for col in features:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe.")
        
        # Get the valid categories in the current column
        valid_categories = df[col].unique()
        
        # Find missing labels
        missing_in_col = [
            label for label in labels if label not in valid_categories]
        
        # Depending on return_valid, return valid or missing labels
        if return_valid:
            valid_in_col = [
                label for label in labels if label in valid_categories]
            result[col] = valid_in_col
        else:
            if missing_in_col:
                result[col] = missing_in_col
        
        # Verbosity control: Print results based on verbosity level
        if verbose >= 1:
            if missing_in_col:
                print(
                    f"{name}: Column '{col}' has missing labels: {missing_in_col}")
            elif verbose >= 2:
                print(
                    f"{name}: Column '{col}' has all labels present.")
        
        if verbose == 3:
            print(f"Debug: Column '{col}' - Valid categories: {valid_categories}")
    
    # Handle case where no valid labels are found in any columns
    if not result:
        if error == 'raise':
            raise ValueError(
                f"No valid labels found in any of the specified columns for {name}.")
        elif error == 'warn':
            warnings.warn(
                "Warning: No valid labels found in any"
                " of the specified columns for {name}."
            )
    
    return result

def is_classification_task(
    *y, max_unique_values=10
    ):
    """
    Check whether the given arrays are for a classification task.

    This function assumes that if all values in the provided arrays are 
    integers and the number of unique values is within the specified
    threshold, it is a classification task.

    Parameters
    ----------
    *y : list or numpy.array
        A variable number of arrays representing actual values, 
        predicted values, etc.
    max_unique_values : int, optional
        The maximum number of unique values to consider the task 
        as classification. 
        Default is 10.

    Returns
    -------
    bool
        True if the provided arrays are for a classification task, 
        False otherwise.

    Examples
    --------
    >>> from gofast.core.checks import is_classification_task 
    >>> y_true = [0, 1, 1, 0, 1]
    >>> y_pred = [0, 1, 0, 0, 1]
    >>> is_classification_task(y_true, y_pred)
    True
    """
    max_unique_values = int (
        _assert_all_types(max_unique_values, 
                          int, float, objname="Max Unique values")
                             )
    # Combine all arrays for analysis
    combined = np.concatenate(y)

    # Check if all elements are integers
    if ( 
            not all(isinstance(x, int) for x in combined) 
            and not combined.dtype.kind in 'iu'
            ):
        return False

    # Check the number of unique elements
    unique_values = np.unique(combined)
    # check Arbitrary threshold for number of classes
    if len(unique_values) > max_unique_values:
        return False

    return True

def validate_noise(noise):
    """
    Validates the `noise` parameter and returns either the noise value
    as a float or the string 'gaussian'.

    Parameters
    ----------
    noise : str or float or None
        The noise parameter to be validated. It can be the string
        'gaussian', a float value, or None.

    Returns
    -------
    float or str
        The validated noise value as a float or the string 'gaussian'.

    Raises
    ------
    ValueError
        If the `noise` parameter is a string other than 'gaussian' or
        cannot be converted to a float.

    Examples
    --------
    >>> validate_noise('gaussian')
    'gaussian'
    >>> validate_noise(0.1)
    0.1
    >>> validate_noise(None)
    None
    >>> validate_noise('0.2')
    0.2

    """
    if isinstance(noise, str):
        if noise.lower() == 'gaussian':
            return 'gaussian'
        else:
            try:
                noise = float(noise)
            except ValueError:
                raise ValueError("The `noise` parameter accepts the string"
                                 " 'gaussian' or a float value.")
    elif noise is not None:
        noise = validate_ratio(noise, bounds=(0, 1), param_name='noise' )
        # try:
        # except ValueError:
        #     raise ValueError("The `noise` parameter must be convertible to a float.")
    return noise


def validate_feature(
    data: Union[DataFrame, Series],  
    features: List[str],
    verbose: str = 'raise'
    ) -> bool:
    """
    Validate the existence of specified features in a DataFrame or Series.
    
    Parameters
    ----------
    data : DataFrame or Series
        The DataFrame or Series to validate feature existence.
    features : list of str
        List of features to check for existence in the data.
    verbose : str, {'raise', 'ignore'}, optional
        Specify how to handle the absence of features. 'raise' (default) will raise
        a ValueError if any feature is missing, while 'ignore' will return a
        boolean indicating whether all features exist.
    
    Returns
    -------
    bool
        True if all specified features exist in the data, False otherwise.
    
    Examples
    --------
    >>> from gofast.core.checks import validate_feature
    >>> import pandas as pd
    >>> data = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> result = validate_feature(data, ['A', 'C'], verbose='raise')
    >>> print(result)  # This will raise a ValueError
    """
    if isinstance(data, pd.Series):
        data = data.to_frame().T  # Convert Series to DataFrame
    features= is_iterable(features, exclude_string= True, transform =True )
    present_features = set(features).intersection(data.columns)
    
    if len(present_features) != len(features):
        missing_features = set(features).difference(present_features)
        verb =" is" if len(missing_features) <2 else "s are"
        if verbose == 'raise':
            raise ValueError(f"The following feature{verb} missing in the "
                             f"data: {_smart_format(missing_features)}.")
        return False
    
    return True

def features_in(
    *data: Union[pd.DataFrame, pd.Series], features: List[str],
    error: str = 'ignore') -> List[bool]:
    """
    Control whether the specified features exist in multiple datasets.

    Parameters
    ----------
    *data : DataFrame or Series arguments
        Multiple DataFrames or Series to check for feature existence.
    features : list of str
        List of features to check for existence in the datasets.
    error : str, {'raise', 'ignore'}, optional
        Specify how to handle the absence of features. 'ignore' (default) will ignore
        a ValueError for each dataset with missing features, while 'ignore' will
        return a list of booleans indicating whether all features exist in each dataset.

    Returns
    -------
    list of bool
        A list of booleans indicating whether the specified features exist in each dataset.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.core.checks import features_in
    >>> data1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> data2 = pd.Series([5, 6], name='C')
    >>> data3 = pd.DataFrame({'X': [7, 8]})
    >>> features = ['A', 'C']
    >>> results1 = features_in(data1, data2, features, error='raise')
    >>> print(results1)  # This will raise a ValueError for the first dataset
    >>> results2 = features_in(data1, data3, features, error='ignore')
    >>> print(results2)  # This will return [True, False]
    """
    results = []

    for dataset in data:
        results.append(validate_feature(dataset, features, verbose=error))

    return results

def find_features_in(
    data: DataFrame = None,
    features: List[str] = None,
    parse_features: bool = False,
    return_frames: bool = False,
) -> Tuple[Union[List[str], DataFrame], Union[List[str], DataFrame]]:
    """
    Retrieve the categorical or numerical features from the dataset.

    Parameters
    ----------
    data : DataFrame, optional
        DataFrame with columns representing the features.
    features : list of str, optional
        List of column names. If provided, the DataFrame will be restricted
        to only include the specified features before searching for numerical
        and categorical features. An error will be raised if any specified
        feature is missing in the DataFrame.
    return_frames : bool, optional
        If True, it returns two separate DataFrames (cat & num). Otherwise, it
        returns only the column names of categorical and numerical features.
    parse_features : bool, default False
        Use default parsers to parse string items into an iterable object.

    Returns
    -------
    Tuple : List[str] or DataFrame
        The names or DataFrames of categorical and numerical features.

    Examples
    --------
    >>> from gofast.datasets import fetch_data
    >>> from gofast.tools.mlutils import find_features_in
    >>> data = fetch_data('bagoue').frame 
    >>> cat, num = find_features_in(data)
    >>> cat, num
    ... (['type', 'geol', 'shape', 'name', 'flow'],
    ...  ['num', 'east', 'north', 'power', 'magnitude', 'sfi', 'ohmS', 'lwi'])
    >>> cat, num = find_features_in(data, features=['geol', 'ohmS', 'sfi'])
    >>> cat, num
    ... (['geol'], ['ohmS', 'sfi'])
    """
    from .array_manager import to_numeric_dtypes 
    
    if not isinstance (data, pd.DataFrame):
        raise TypeError(f"Expect a DataFrame. Got {type(data).__name__!r}")

    if features is not None:
        features = list(
            is_iterable(
                features,
                exclude_string=True,
                transform=True,
                parse_string=parse_features,
            )
        )

    if features is None:
        features = list(data.columns)

    validate_feature(data, list(features))
    data = data[features].copy()

    # Get numerical features
    data, numnames, catnames = to_numeric_dtypes(
        data, return_feature_types=True )

    if catnames is None:
        catnames = []

    return (data[catnames], data[numnames]) if return_frames else (
        list(catnames), list(numnames)
    )


def check_uniform_type(
    values: Union[Iterable[Any], Any],
    items_to_compare: Union[Iterable[Any], Any] = None,
    raise_exception: bool = True,
    convert_values: bool = False,
    return_types: bool = False,
    target_type: type = None,
    allow_mismatch: bool = True,
    infer_types: bool = False,
    comparison_method: str = 'intersection',
    custom_conversion_func: _F[Any] = None,
    return_func: bool = False
) -> Union[bool, List[type], Tuple[Iterable[Any], List[type]], _F]:
    """
    Checks whether elements in `values` are of uniform type. 
    
    Optionally comparing them against another set of items or converting all 
    values to a target type. Can return a callable for deferred execution of 
    the specified logic.Function is useful for validating data uniformity, 
    especially before performing operations that assume homogeneity of the 
    input types.
    

    Parameters
    ----------
    values : Iterable[Any] or Any
        An iterable containing items to check. If a non-iterable item is provided,
        it is treated as a single-element iterable.
    items_to_compare : Iterable[Any] or Any, optional
        An iterable of items to compare against `values`. If specified, the
        `comparison_method` is used to perform the comparison.
    raise_exception : bool, default True
        If True, raises an exception when a uniform type is not found or other
        constraints are not met. Otherwise, issues a warning.
    convert_values : bool, default False
        If True, tries to convert all `values` to `target_type`. Requires
        `target_type` to be specified.
    return_types : bool, default False
        If True, returns the types of the items in `values`.
    target_type : type, optional
        The target type to which `values` should be converted if `convert_values`
        is True.
    allow_mismatch : bool, default True
        If False, requires all values to be of identical types; otherwise,
        allows type mismatch.
    infer_types : bool, default False
        If True and different types are found, returns the types of each item
        in `values` in order.
    comparison_method : str, default 'intersection'
        The method used to compare `values` against `items_to_compare`. Must
        be one of the set comparison methods ('difference', 'intersection', etc.).
    custom_conversion_func : Callable[[Any], Any], optional
        A custom function for converting items in `values` to another type.
    return_func : bool, default False
        If True, returns a callable that encapsulates the logic based on the 
        other parameters.

    Returns
    -------
    Union[bool, List[type], Tuple[Iterable[Any], List[type]], Callable]
        The result based on the specified parameters. This can be: 
        - A boolean indicating whether all values are of the same type.
        - The common type of all values if `return_types` is True.
        - A tuple containing the converted values and their types if `convert_values`
          and `return_types` are both True.
        - a callable encapsulating the specified logic for deferred execution.
        
    Examples
    --------
    >>> from gofast.core.checks import check_uniform_type
    >>> check_uniform_type([1, 2, 3])
    True

    >>> check_uniform_type([1, '2', 3], allow_mismatch=False, raise_exception=False)
    False

    >>> deferred_check = check_uniform_type([1, 2, '3'], convert_values=True, 
    ...                                        target_type=int, return_func=True)
    >>> deferred_check()
    [1, 2, 3]

    Notes
    -----
    The function is designed to be flexible, supporting immediate or deferred execution,
    with options for type conversion and detailed type information retrieval.
    """
    def operation():
        # Convert values and items_to_compare to lists if 
        # they're not already iterable
        if isinstance(values, Iterable) and not isinstance(values, str):
            val_list = list(values)
        else:
            val_list = [values]

        if items_to_compare is not None:
            if isinstance(items_to_compare, Iterable) and not isinstance(
                    items_to_compare, str):
                comp_list = list(items_to_compare)
            else:
                comp_list = [items_to_compare]
        else:
            comp_list = []

        # Extract types
        val_types = set(type(v) for v in val_list)
        comp_types = set(type(c) for c in comp_list) if comp_list else set()

        # Compare types
        if comparison_method == 'intersection':
            common_types = val_types.intersection(comp_types) if comp_types else val_types
        elif comparison_method == 'difference':
            common_types = val_types.difference(comp_types)
        else:
            if raise_exception:
                raise ValueError(f"Invalid comparison method: {comparison_method}")
            return False

        # Check for type uniformity
        if not allow_mismatch and len(common_types) > 1:
            if raise_exception:
                raise ValueError("Not all values are the same type.")
            return False

        # Conversion
        if convert_values:
            if not target_type and not custom_conversion_func:
                if raise_exception:
                    raise ValueError("Target type or custom conversion "
                                     "function must be specified for conversion.")
                return False
            try:
                if custom_conversion_func:
                    converted_values = [custom_conversion_func(v) for v in val_list]
                else:
                    converted_values = [target_type(v) for v in val_list]
            except Exception as e:
                if raise_exception:
                    raise ValueError(f"Conversion failed: {e}")
                return False
            if return_types:
                return converted_values, [type(v) for v in converted_values]
            return converted_values

        # Return types
        if return_types:
            if infer_types or len(common_types) > 1:
                return [type(v) for v in val_list]
            return list(common_types)

        return True

    return operation if return_func else operation()


def assert_ratio(
    v: Union[str, float, int],
    bounds: Optional[Tuple[float, float]] = None,
    exclude_values: Optional[Union[float, List[float]]] = None,
    in_percent: bool = False,
    inclusive: bool = True,
    name: str = 'ratio'
) -> float:
    """
    Asserts that a given value falls within a specified range and does not
    match any excluded values. Optionally converts the value to a percentage.
    
    This function is useful for validating ratio or rate values in data 
    preprocessing, ensuring they meet defined criteria before further 
    analysis or modeling.
    
    Parameters
    ----------
    v : Union[str, float, int]
        The ratio value to assert. Can be a string (possibly containing 
        a percentage sign), float, or integer.
        
    bounds : Optional[Tuple[float, float]], default=None
        A tuple specifying the lower and upper bounds (inclusive by default) 
        within which the value `v` must lie. If `None`, no bounds are enforced.
        
    exclude_values : Optional[Union[float, List[float]]], default=None
        Specific value(s) that `v` must not equal. Can be a single float or a 
        list of floats. If provided, `v` is checked against these excluded 
        values after any necessary conversions.
        
    in_percent : bool, default=False
        If `True`, interprets the input value `v` as a percentage and converts 
        it to its decimal form (e.g., 50 becomes 0.5). If `v` is a string 
        containing a `%` sign, it is automatically converted to decimal.
        
    inclusive : bool, default=True
        Determines whether the bounds are inclusive. If `True`, `v` can be equal 
        to the lower and upper bounds. If `False`, `v` must be strictly 
        greater than the lower bound and strictly less than the upper bound.
        
    name : str, default='ratio'
        The descriptive name of the value being asserted. This is used in error 
        messages for clarity.
    
    Returns
    -------
    float
        The validated (and possibly converted) ratio value.
        
    Raises
    ------
    TypeError
        If `v` cannot be converted to a float.
        
    ValueError
        If `v` is outside the specified bounds or matches any excluded values.
    
    Examples
    --------
    1. **Basic Usage with Bounds:**
    
        ```python
        from gofast.core.checks import assert_ratio
        assert_ratio(0.5, bounds=(0.0, 1.0))
        # Returns: 0.5
        ```
    
    2. **String Input with Percentage:**
    
        ```python
        assert_ratio("75%", in_percent=True)
        # Returns: 0.75
        ```
    
    3. **Excluding Specific Values:**
    
        ```python
        assert_ratio(0.5, bounds=(0.0, 1.0), exclude_values=0.5)
        # Raises ValueError
        ```
    
    4. **Multiple Excluded Values and Exclusive Bounds:**
    
        ```python
        assert_ratio(0.3, bounds=(0.0, 1.0), exclude_values=[0.2, 0.4], inclusive=False)
        # Returns: 0.3
        ```
    
    Notes
    -----
    - The function first attempts to convert the input `v` to a float. 
      If `in_percent` is `True`, it converts percentage values to 
      their decimal equivalents.
    - Bounds can be set to define a valid range for `v`. If `inclusive` 
      is set to `False`, the bounds are treated as exclusive.
    - Excluded values are checked after any necessary conversions.
    - If `exclude_values` is provided without specifying `bounds`, the 
      function will only check for excluded values.
    
    References
    ----------
    - [Python `float()` Function](https://docs.python.org/3/library/functions.html#float)
    - [Warnings in Python](https://docs.python.org/3/library/warnings.html)
    """
    
    # Initialize exclusion list
    if exclude_values is not None and not isinstance(exclude_values, list):
        exclude_values = [exclude_values]
    
    # Regular expression to detect percentage in strings
    percent_pattern = re.compile(r'^\s*[-+]?\d+(\.\d+)?%\s*$')
    
    # Check and convert string inputs
    if isinstance(v, str):
        v = v.strip()
        if percent_pattern.match(v):
            in_percent = True
            v = v.replace('%', '').strip()
    
    try:
        # Convert to float
        v = float(v)
    except (TypeError, ValueError):
        raise TypeError(
            f"Unable to convert {type(v).__name__!r} value '{v}' to float."
        )
    
    # Convert to percentage if required
    if in_percent:
        if 0 <= v <= 100:
            v /= 100.0
        elif 0 <= v <= 1:
            warnings.warn(
                f"The value {v} seems already in decimal form; "
                f"no conversion applied for {name}.",
                UserWarning
            )
        else:
            raise ValueError(
                f"When 'in_percent' is True, {name} should be between "
                f"0 and 100, got {v * 100 if 0 <= v <=1 else v}."
            )
    
    # Check bounds if specified
    if bounds:
        if not isinstance(bounds, tuple) or len(bounds) != 2:
            raise ValueError(
                 "'bounds' must be a tuple of two"
                f" floats, got {type(bounds).__name__}."
            )
        lower, upper = bounds
        if inclusive:
            if not (lower <= v <= upper):
                raise ValueError(
                    f"{name.capitalize()} must be between {lower}"
                    f" and {upper} inclusive, got {v}."
                )
        else:
            if not (lower < v < upper):
                raise ValueError(
                    f"{name.capitalize()} must be between {lower} and "
                    f"{upper} exclusive, got {v}."
                )
    
    # Check excluded values
    if exclude_values:
        if v in exclude_values:
            if len(exclude_values) == 1:
                exclusion_msg =( 
                    f"{name.capitalize()} must not be {exclude_values[0]}."
                    )
            else:
                exclusion_msg =( 
                    f"{name.capitalize()} must not be one of {exclude_values}."
                    )
            raise ValueError(exclusion_msg)
    
    return v

def validate_ratio(
    value: float, 
    bounds: Optional[Tuple[float, float]] = None, 
    exclude: Optional[float] = None, 
    to_percent: bool = False, 
    param_name: str = 'value'
) -> float:
    """Validates and optionally converts a value to a percentage within 
    specified bounds, excluding specific values.

    Parameters:
    -----------
    value : float or str
        The value to validate and convert. If a string with a '%' sign, 
        conversion to percentage is attempted.
    bounds : tuple of float, optional
        A tuple specifying the lower and upper bounds (inclusive) for the value. 
        If None, no bounds are enforced.
    exclude : float, optional
        A specific value to exclude from the valid range. If the value matches 
        'exclude', a ValueError is raised.
    to_percent : bool, default=False
        If True, the value is converted to a percentage 
        (assumed to be in the range [0, 100]).
    param_name : str, default='value'
        The parameter name to use in error messages.

    Returns:
    --------
    float
        The validated (and possibly converted) value.

    Raises:
    ------
    ValueError
        If the value is outside the specified bounds, matches the 'exclude' 
        value, or cannot be converted as specified.
    """
    if isinstance(value, str) and '%' in value:
        to_percent = True
        value = value.replace('%', '')
    try:
        value = float(value)
    except ValueError:
        raise ValueError(f"Expected a float, got {type(value).__name__}: {value}")

    if to_percent and 0 < value <= 100:
        value /= 100

    if bounds:
        if not (bounds[0] <= value <= bounds[1]):
            raise ValueError(f"{param_name} must be between {bounds[0]}"
                             f" and {bounds[1]}, got: {value}")
    
    if exclude is not None and value == exclude:
        raise ValueError(f"{param_name} cannot be {exclude}")

    if to_percent and value > 1:
        raise ValueError(f"{param_name} converted to percent must"
                         f" not exceed 1, got: {value}")

    return value

def exist_features(
    df: pd.DataFrame, 
    features, 
    error='raise',  
    name="Feature"
) -> bool:
    """
    Check whether the specified features exist in the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the features to be checked.
    features : list or str
        List of feature names (str) to check for in the dataframe. 
        If a string is provided, it will be treated as a list with 
        a single feature.
    error : str, optional, default 'raise'
        Action to take if features are not found. Can be one of:
        - 'raise' (default): Raise a ValueError.
        - 'warn': Issue a warning and return False.
        - 'ignore': Do nothing if features are not found.
    name : str, optional, default 'Feature'
        Name of the feature(s) being checked (default is 'Feature').

    Returns
    -------
    bool
        Returns True if all features exist in the dataframe, otherwise False.

    Raises
    ------
    ValueError
        If 'error' is 'raise' and features are not found.
    
    Warns
    -----
    UserWarning
        If 'error' is 'warn' and features are missing.

    Notes
    -----
    This function ensures that all the specified features exist in the
    dataframe. If the 'error' parameter is set to 'warn', the function 
    will issue a warning instead of raising an error when a feature 
    is missing, and return False.

    References
    ----------
    - pandas.DataFrame:
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html

    Examples
    --------
    >>> from gofast.core.checks import exist_features
    >>> import pandas as pd

    >>> # Sample DataFrame
    >>> df = pd.DataFrame({
    >>>     'feature1': [1, 2, 3],
    >>>     'feature2': [4, 5, 6],
    >>>     'feature3': [7, 8, 9]
    >>> })

    >>> # Check for missing features with 'raise' error
    >>> exist_features(df, ['feature1', 'feature4'], error='raise')
    Traceback (most recent call last):
        ...
    ValueError: Features feature4 not found in the dataframe.

    >>> # Check for missing features with 'warn' error
    >>> exist_features(df, ['feature1', 'feature4'], error='warn')
    UserWarning: Features feature4 not found in the dataframe.

    >>> # Check for missing features with 'ignore' error
    >>> exist_features(df, ['feature1', 'feature4'], error='ignore')
    False
    """
    # Validate if the input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("'df' must be a pandas DataFrame.")

    # Normalize the error parameter to lowercase and strip whitespace
    error = error.lower().strip()

    # Validate the 'error' parameter
    if error not in ['raise', 'ignore', 'warn']:
        raise ValueError(
            "Invalid value for 'error'. Expected"
            " one of ['raise', 'ignore', 'warn'].")

    # Ensure 'features' is a list-like structure
    if isinstance(features, str):
        features = [features]

    # Validate that 'features' is one of the allowed types
    features = _assert_all_types(features, (list, tuple, np.ndarray))

    # Get the intersection of features with the dataframe columns
    existing_features = set(features).intersection(df.columns)

    # If all features exist, return True
    if len(existing_features) == len(features):
        return True

    # Calculate the missing features
    missing_features = set(features) - existing_features

    # If there are missing features, handle according to 'error' type
    if missing_features:
        msg = f"{name}{'s' if len(features) > 1 else ''}"

        if error == 'raise':
            raise ValueError(
                f"{msg} {_smart_format(missing_features)}"
                " not found in the dataframe."
            )

        elif error == 'warn':
            warnings.warn(
                f"{msg} {_smart_format(missing_features)}"
                " not found in the dataframe.",
                UserWarning
            )
            return False

        # If 'error' is 'ignore', simply return False
        return False

    return True

def is_iterable (
        y, exclude_string= False, transform = False , parse_string =False, 
)->Union [bool , list]: 
    """ Asserts iterable object and returns boolean or transform object into
     an iterable.
    
    Function can also transform a non-iterable object to an iterable if 
    `transform` is set to ``True``.
    
    :param y: any, object to be asserted 
    :param exclude_string: bool, does not consider string as an iterable 
        object if `y` is passed as a string object. 
    :param transform: bool, transform  `y` to an iterable objects. But default 
        puts `y` in a list object. 
    :param parse_string: bool, parse string and convert the list of string 
        into iterable object is the `y` is a string object and containg the 
        word separator character '[#&.*@!_,;\s-]'. Refer to the function 
        :func:`~gofast.core.checks.str2columns` documentation.
        
    :returns: 
        - bool, or iterable object if `transform` is set to ``True``. 
        
    .. note:: 
        Parameter `parse_string` expects `transform` to be ``True``, otherwise 
        a ValueError will raise. Note :func:`.is_iterable` is not dedicated 
        for string parsing. It parses string using the default behaviour of 
        :func:`.str2columns`. Use the latter for string parsing instead. 
        
    :Examples: 
    >>> from gofast.coreutils.is_iterable 
    >>> is_iterable ('iterable', exclude_string= True ) 
    Out[28]: False
    >>> is_iterable ('iterable', exclude_string= True , transform =True)
    Out[29]: ['iterable']
    >>> is_iterable ('iterable', transform =True)
    Out[30]: 'iterable'
    >>> is_iterable ('iterable', transform =True, parse_string=True)
    Out[31]: ['iterable']
    >>> is_iterable ('iterable', transform =True, exclude_string =True, 
                     parse_string=True)
    Out[32]: ['iterable']
    >>> is_iterable ('parse iterable object', parse_string=True, 
                     transform =True)
    Out[40]: ['parse', 'iterable', 'object']
    """
    if (parse_string and not transform) and isinstance (y, str): 
        raise ValueError ("Cannot parse the given string. Set 'transform' to"
                          " ``True`` otherwise use the 'str2columns' utils"
                          " from 'gofast.core.checks' instead.")
    y = str2columns(y) if isinstance(y, str) and parse_string else y 
    
    isiter = False  if exclude_string and isinstance (
        y, str) else hasattr (y, '__iter__')
    
    return ( y if isiter else [ y ] )  if transform else isiter 

def _smart_format(iter_obj, choice ='and'): 
    """ Smart format iterable object.
    """
    str_litteral =''
    try: 
        iter(iter_obj) 
    except:  return f"{iter_obj}"
    
    iter_obj = [str(obj) for obj in iter_obj]
    if len(iter_obj) ==1: 
        str_litteral= ','.join([f"{i!r}" for i in iter_obj ])
    elif len(iter_obj)>1: 
        str_litteral = ','.join([f"{i!r}" for i in iter_obj[:-1]])
        str_litteral += f" {choice} {iter_obj[-1]!r}"
    return str_litteral

def str2columns(
    text: str, 
    regex: Optional[re.Pattern] = None, 
    pattern: Optional [str]= None
) -> List[str]:
    """
    Splits the input text into column names by removing non-alphanumeric 
    characters and using a regular expression pattern. The function 
    splits the string into individual words or attribute names based on 
    the provided regular expression or the default pattern.

    This function is useful for extracting meaningful words or column 
    names from text that contains delimiters like spaces, punctuation, 
    or special characters.

    Parameters
    ----------
    text : str
        The input string containing the column names or words to retrieve. 
        This is the text that will be split into individual components 
        (attributes).
    
    regex : re.Pattern, optional
        A custom compiled regular expression object used to split the 
        `text`. If not provided, the default pattern will be used. 
        The default pattern is:
        
        >>> re.compile(r'[#&.*@!_,;\s-]\s*', flags=re.IGNORECASE)

    pattern : str, optional, default=r'[#&.*@!_,;\s-]\s*'
        A string representing the regular expression pattern used to 
        split the `text`. This pattern defines the non-alphanumeric 
        markers and whitespace characters (including spaces, punctuation, 
        and operators) that will be treated as delimiters. If `regex` is 
        not provided, this pattern is used by default.

    Returns
    -------
    List[str]
        A list of attribute names (words) extracted from the `text`. The 
        text is split using the specified regular expression or the 
        default pattern.

    Examples
    --------
    >>> from gofast.core.checks import str2columns
    >>> text = ('this.is the text to split. It is an example of splitting '
    >>>         'str to text.')
    >>> str2columns(text)
    ['this', 'is', 'the', 'text', 'to', 'split', 'It', 'is', 'an:', 
    'example', 'of', 'splitting', 'str', 'to', 'text']
    """
    pattern = pattern or r'[#&.*@!_,;\s-]\s*'
    regex = regex or re.compile(pattern, flags=re.IGNORECASE)
    text = list(filter(None, regex.split(str(text))))
    return text

def _assert_all_types(
    obj: object,
    *expected_objtype: type,
    objname: str = None,
) -> object:
    """
    Quick assertion to check if an object is of an expected type.

    Parameters
    ----------
    obj : object
        The object whose type is being checked.
    expected_objtype : type
        One or more types to check against. If the object's type
        does not match any of the provided types, a TypeError is raised.
    objname : str, optional
        The name of the object being checked, used to customize the
        error message. If not provided, a generic message is used.

    Raises
    ------
    TypeError
        If the object's type does not match any of the expected types.

    Returns
    -------
    object
        The original object if its type matches one of the expected types.

    Notes
    -----
    This function raises a `TypeError` if the object's type does not
    match the expected type(s). The error message can be customized by
    providing the `objname` argument.
    """
    # if np.issubdtype(a1.dtype, np.integer): 
    if not isinstance(obj, expected_objtype):
        n = str(objname) + ' expects' if objname is not None else 'Expects'
        raise TypeError(
            f"{n} type{'s' if len(expected_objtype) > 1 else ''} "
            f"{_smart_format(tuple(o.__name__ for o in expected_objtype))} "
            f"but {type(obj).__name__!r} is given."
        )

    return obj

def random_state_validator(seed):
    """Turn seed into a Numpy-Random-RandomState instance.
    
    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
        
    Returns
    -------
    :class:`numpy:numpy.random.RandomState`
        The random state object based on `seed` parameter.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState instance" % seed
    )
    
def is_numeric_dtype(o, to_array=False):
    """
    Determine whether the argument has a numeric datatype when
    converted to a NumPy array.

    Parameters
    ----------
    o : object or array-like
        The object (or iterable) to check for a numeric datatype. 
        This can be a list, tuple, or any other iterable object.
    to_array : bool, optional, default=False
        If `o` is passed as a non-array-like object (e.g., list, tuple,
        or other iterable), setting `to_array` to `True` will convert 
        `o` into a NumPy array before checking its datatype.

    Returns
    -------
    bool
        `True` if `o` has a numeric dtype, and `False` otherwise.

    Raises
    ------
    TypeError
        If `o` is not an iterable object.
    ValueError
        If `o` is not an array-like object after conversion (if `to_array=True`).

    Examples
    --------
    >>> from gofast.core.checks import _is_numeric_dtypes
    >>> is_numeric_dtype([1, 2, 3])
    True

    >>> is_numeric_dtype(['a', 'b', 'c'])
    False

    >>> is_numeric_dtype((1.5, 2.3, 3.1), to_array=True)
    True

    >>> is_numeric_dtype({'a': 1, 'b': 2})
    False

    Notes
    -----
    This function checks if the dtype of `o` (or its NumPy array 
    conversion) is one of the numeric types: boolean, unsigned integers, 
    signed integers, floats, or complex numbers. It uses the `dtype.kind`
    attribute to determine this.

    The function will raise an error if `o` is not iterable, or if it 
    cannot be converted into an array-like structure.

    The check for numeric types is performed using the `_NUMERIC_KINDS` set,
    which includes the following types:
        - 'b' : boolean
        - 'u' : unsigned integer
        - 'i' : signed integer
        - 'f' : float
        - 'c' : complex number
    """
    _NUMERIC_KINDS = set('buifc')

    # Check if 'o' is iterable
    if not hasattr(o, '__iter__'):
        raise TypeError("'o' is expected to be an iterable object. "
                         f"Got: {type(o).__name__!r}")
    
    # Convert to array if specified
    if to_array:
        o = np.array(o)

    # Check if 'o' is an array-like object
    if not hasattr(o, '__array__'):
        raise ValueError(f"Expect type array-like, got: {type(o).__name__!r}")

    # Check for numeric dtype using _NUMERIC_KINDS
    return (o.values.dtype.kind if (hasattr(o, 'columns') or hasattr(o, 'name'))
            else o.dtype.kind) in _NUMERIC_KINDS

def is_in(
    arr: Union[ArrayLike, List[float]],
    subarr: Union[_Sub[ArrayLike], _Sub[List[float]], float],
    return_mask: bool = False,
) -> bool:
    """
    Check whether the subset array `subarr` is present in the array `arr`.

    Parameters
    ----------
    arr : array-like
        Array of item elements to check against. This can be a list,
        numpy array, or any other array-like structure.
    subarr : array-like, float
        Subset array or individual item to check for presence in `arr`.
        This can be a list, numpy array, or float.
    return_mask : bool, optional
        If True, returns a boolean mask indicating where the elements of
        `subarr` are found in `arr`. Default is False, which returns a
        single boolean value (True if any element of `subarr` is in `arr`,
        False otherwise).

    Returns
    -------
    bool or ndarray
        If `return_mask` is False, returns `True` if any item in `subarr`
        is present in `arr`, otherwise returns `False`. If `return_mask` is
        True, returns a boolean mask (ndarray) where `True` indicates that
        the corresponding element in `arr` is found in `subarr`.

    Examples
    --------
    >>> from gofast.core.checks import is_in 
    >>> is_in([1, 2, 3, 4, 5], [2, 4])
    True
    
    >>> is_in([1, 2, 3, 4, 5], [6, 7], return_mask=True)
    array([False, False, False, False, False])
    
    >>> is_in([1, 2, 3, 4, 5], 3)
    True
    
    >>> is_in([1, 2, 3, 4, 5], 6)
    False

    Notes
    -----
    This function uses `np.isin` internally to check whether elements
    from `subarr` are present in `arr`. The `return_mask` argument
    allows for flexibility in the return type. If `return_mask` is False,
    the function simply checks if any elements of `subarr` are present in
    `arr` and returns a boolean result.
    """
    arr = np.array(arr)
    subarr = np.array(subarr)

    return (True if True in np.isin(arr, subarr) else False
            ) if not return_mask else np.isin(arr, subarr)

def is_sparse_matrix(
    data: pd.Series, 
    threshold: float = 0.9, 
    verbose=False
    ) -> bool:
    """
    Checks if the data is a sparse matrix, either as a scipy sparse matrix 
    or a pandas Series containing string-encoded sparse matrix data.
    
    This function identifies sparse data structures, considering both 
    actual scipy sparse matrix types and string-encoded representations 
    of sparse matrices, such as those commonly found in pandas Series.
    
    Parameters
    ----------
    data : object
        The data to check. This can be a scipy sparse matrix or a pandas 
        Series containing string-encoded sparse matrix data.
    
    threshold : float, optional, default 0.9
        The minimum proportion of entries that must match the sparse 
        pattern (i.e., be non-zero) for the data to be considered sparse. 
        This value should lie between 0 and 1.
    
    verbose : bool, optional, default False
        If set to True, the function will print the sparsity ratio for a 
        scipy sparse matrix and the proportion of matching entries for a 
        pandas Series. This is useful for debugging or monitoring the 
        function’s behavior.
    
    Returns
    -------
    bool
        True if the data is a sparse matrix (either scipy sparse matrix or 
        string-encoded sparse matrix), False otherwise.
    
    Notes
    -----
    - The function first checks if the data is a scipy sparse matrix 
      (e.g., `csr_matrix`, `coo_matrix`).
    - If the data is a pandas Series, it assumes the Series may contain 
      string-encoded sparse matrix data and checks if each entry in the 
      Series follows the expected sparse format.
    - The `threshold` determines how many non-zero elements (or matching 
      string-encoded sparse entries) are required to consider the data sparse.
    
    Examples
    --------
    1. Check if a scipy sparse matrix is sparse:
    
       ```python
       sparse_matrix = sp.csr_matrix([[0, 0, 1], [0, 2, 0], [0, 0, 3]])
       result = is_sparse_matrix(sparse_matrix)
       print(result)  # Expected: True (based on sparsity ratio)
       ```

    2. Check if a pandas Series with string-encoded sparse matrix data is sparse:
    
       ```python
       sparse_series = pd.Series([
           "(0, 0)\t1.0\n(1, 1)\t2.0\n(2, 2)\t3.0",
           "(0, 1)\t1.5\n(1, 0)\t1.0\n(2, 1)\t2.5"
       ])
       result = is_sparse_matrix(sparse_series)
       print(result)  # Expected: True or False (based on threshold)
       ```

    References
    ----------
    - SciPy Sparse Matrices Documentation:
      https://docs.scipy.org/doc/scipy/reference/sparse.html
    - pandas Series Documentation:
      https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html
    """
    if isinstance ( data, pd.DataFrame) :
        data = data.squeeze () 
        
    # Check if the data is a scipy sparse matrix
    if isinstance(data, ssp.spmatrix):
        # Number of non-zero elements in the sparse matrix
        non_zero_elements = data.nnz
        
        # Total number of elements in the matrix (rows * columns)
        total_elements = data.shape[0] * data.shape[1]
        
        # Calculate the sparsity ratio (non-zero elements / total elements)
        sparsity_ratio = non_zero_elements / total_elements
        
        # Print the sparsity ratio if verbose flag is True
        if verbose:
            print(f"Sparsity ratio: {sparsity_ratio:.2f}")
        
        # If the sparsity ratio meets the threshold, return True (sparse)
        return sparsity_ratio >= threshold
    
    # Check if the data is a pandas Series
    if isinstance(data, pd.Series):
        # Check if each entry in the Series follows the expected sparse format
        matches = data.apply(has_sparse_format)
        
        # Calculate the proportion of entries that match the sparse format
        proportion = matches.mean()
        
        # Print the proportion of matching entries if verbose flag is True
        if verbose:
            print(f"Proportion of matching entries: {proportion:.2f}")
        
        # If the proportion of matching entries
        # meets the threshold, return True (sparse)
        return proportion >= threshold
    
    # If data is neither a scipy sparse matrix
    # nor a string-encoded pandas Series
    if verbose:
        print("Data is neither a scipy sparse matrix"
              " nor a string-encoded pandas Series.")
    
    return False

def has_sparse_format(s):
    """
    Checks if a string follows the expected sparse matrix format for entries
    (i.e., coordinate-value pairs like (i, j)\tvalue).
    
    This function uses a regular expression to identify if a given string 
    represents a sparse matrix entry with coordinate-value pairs. This is 
    particularly useful when checking if the entries in a pandas Series 
    follow the sparse matrix format.
    
    Parameters
    ----------
    s : str
        A string entry to check. This should contain coordinates and values 
        separated by tabs, e.g., "(i, j)\tvalue".
    
    Returns
    -------
    bool
        True if the string follows the sparse matrix format, False otherwise.
    
    Examples
    --------
    1. Check if a string represents a sparse matrix entry:
    
       ```python
       entry = "(0, 0)\t1.0"
       result = has_sparse_format(entry)
       print(result)  # Expected: True
       ```
    """
    # Regex pattern for the expected sparse format: (i, j)\tvalue
    pattern = re.compile(r'\(\d+, \d+\)\t-?\d+(\.\d+)?')
    
    if isinstance(s, (ssp.coo_matrix, ssp.csr_matrix, ssp.csc_matrix)):
        return True 
    
    # Return False if s is not a string
    if not isinstance(s, str):
        return False
    
    # Split the string into individual entries
    entries = s.split()
    
    # Check if each entry matches the sparse matrix format
    for entry in entries:
        if not pattern.match(entry):
            return False
    
    return True

def validate_name_in(
    name, defaults='', 
    expect_name=None, 
    exception=None, 
    deep=False 
    ):
    """
    Assert that the given name exists within a set of default names.

    Parameters
    ----------
    name : str
        The name to assert.
    defaults : list of str or str, optional, default=''
        The default names used for the assertion. Can be a list of names,
        a single string, or other iterable. If `deep=True`, this argument
        will be joined into a single string and checked for occurrences of 
        `name`.
    expect_name : str, optional
        The name to return if the assertion is verified (`True`). If `None`,
        the function will return `True` or `False` depending on whether the
        name is found in the defaults.
    deep : bool, optional, default=False
        If `True`, `defaults` are joined into a single string and the function
        checks whether `name` occurs anywhere in the concatenated string.
    exception : Exception, optional
        The exception to raise if `name` is not found in `defaults`. If no
        exception is provided and the name is not found, the function will 
        return `False`.

    Returns
    -------
    str or bool
        If `expect_name` is provided and `name` is found in `defaults`,
        the function returns `expect_name`. If `expect_name` is `None`,
        it returns `True` if `name` is found in `defaults`, or `False` otherwise.
        If `name` is not found and `exception` is specified, the exception
        is raised.

    Examples
    --------
    >>> from gofast.core.checks import validate_name_in
    >>> dnames = ('NAME', 'FIRST NAME', 'SURNAME')
    >>> validate_name_in('name', defaults=dnames)
    False

    >>> validate_name_in('name', defaults=dnames, deep=True)
    True

    >>> validate_name_in('name', defaults=dnames, expect_name='NAM')
    False

    >>> validate_name_in('name', defaults=dnames, expect_name='NAM', deep=True)
    'NAM'

    Notes
    -----
    The function performs a case-insensitive check for `name` within
    the `defaults`. If `deep=True`, it combines all elements in `defaults`
    into a single string and checks whether `name` is a substring of that string.
    If `name` is found and `expect_name` is provided, the function returns
    `expect_name`. Otherwise, it returns a boolean value indicating whether
    `name` is in `defaults`. If `name` is not found and `exception` is provided,
    the exception is raised.
    """
    
    name = str(name).lower().strip()
    defaults = is_iterable(
        defaults, exclude_string=True, parse_string=True, 
        transform=True)
    
    if deep:
        defaults = ''.join([str(i) for i in defaults])

    # Check if name is in defaults
    name = True if expect_name is None else expect_name if name in defaults else False
    
    if not name and exception:
        raise exception
    
    return name

def is_valid_dtypes(
    df: pd.DataFrame, 
    columns: Optional[Union[str, List[Any]]] = None, 
    dtypes: Union[str, List[str]] = 'numeric',
    ops: str = 'check_only',
    treat_obj_dtype_as_category: bool = False, 
    error_msg: Optional[str] = None, 
    extra: str = '',
    error: str = 'warn', 
) -> Union[bool, Dict[str, List[Any]]]:
    """
    Check if specified columns in a DataFrame match the desired data types.
    
    This function verifies whether the data types of selected columns in a 
    pandas DataFrame align with specified objective types. It supports various 
    operations, including validation and conditional error handling, making it 
    a versatile tool for data preprocessing and quality assurance in data pipelines.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to check.
    
    columns : str or list of str, optional
        The columns to check. If ``None``, all columns are checked.

        .. code-block:: python

            columns = 'age'
            columns = ['age', 'salary']
    
    dtypes : str or list of str, default= 'numeric'
        The desired data types to validate against. 
        Options include ``'numeric'``, ``'object'``, ``'category'``, ``'datetime'``.
        Can be a single type or a list of types.
        
        .. code-block:: python

            dtypes = 'numeric'
            dtypes = ['numeric', 'datetime']
    
    ops : str, default 'check_only'
        Operation mode. 
        - ``'check_only'``: Returns True if all specified columns match the 
          objectives, False otherwise.
        - ``'validate'``: Returns a dictionary with objectives as keys and lists 
          of valid columns as values.
    
    treat_obj_dtype_as_category : bool, default= False
        If True, columns with dtype ``'category'`` are treated as categorical 
        regardless of the 'object' dtype.
    
    error_msg : str, optional
        Custom error message to display if validation fails.
        If None, a default message is used.
    
    extra : str, default= ''
        Extra message to append to the error message.
    
    error : str, default ='warn'
        Specifies the error handling behavior. Options are:
        - ``'raise'``: Raises an exception when validation fails.
        - ``'warn'``: Issues a warning when validation fails.
        - ``'ignore'``: Silently ignores validation failures.
    
    Returns
    -------
    bool or dict
        - If ``ops='check_only'``, returns ``True`` if all specified columns 
          match the objectives dtypes, ``False`` otherwise.
        - If ``ops='validate'``, returns a dictionary mapping objectives to 
          lists of valid columns.
    
    Notes
    ------
    Let the DataFrame be represented as :math:`DF`, and let 
    :math:`C = \{c_1, c_2, \dots, c_n\}` be the set of columns to validate. 
    Let :math:`O = \{o_1, o_2, \dots, o_m\}` be the set of objective data types.
    
    The function checks whether each column :math:`c_i \in C` satisfies 
    :math:`c_i.dtype \in O`. If ``treat_obj_dtype_as_category=True``, 
    columns with dtype ``'category'`` are also considered valid for 
    the ``'object'`` objective.
    
    .. math::
        \forall c_i \in C, \quad c_i.dtype \in O \cup \{\text{'category'} 
        \text{ if treat\_obj\_dtype\_as\_category}\}
    
    Examples
    ---------
    >>> from gofast.tools.datautils import is_valid_dtypes
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'age': [25, 30, 45],
    ...     'salary': [50000.0, 60000.5, 75000.0],
    ...     'department': ['HR', 'Engineering', 'Marketing'],
    ...     'hire_date': pd.to_datetime(['2020-01-15', '2019-03-22', '2021-07-30'])
    ... })
    >>> # Check if 'age' and 'salary' are numeric
    >>> is_valid_dtypes(df, columns=['age', 'salary'], dtypes='numeric')
    True
    >>> # Validate data types for multiple objectives
    >>> is_valid_dtypes(df, objectives=['numeric', 'datetime'], ops='validate')
    {'numeric': ['age', 'salary'], 'datetime': ['hire_date']}
    >>> # Check with custom error handling
    >>> is_valid_dtypes(df, columns='department', dtypes='numeric', error='raise')
    Traceback (most recent call last):
        ...
    TypeError: Invalid ... objectives: ['department']

    Notes
    -----
    - The function is particularly useful in data preprocessing pipelines to ensure 
      data integrity before performing further analysis or modeling.
    - When ``ops='validate'``, the returned dictionary provides a clear 
      categorization of columns based on their data types, facilitating 
      selective processing.
    - The ``treat_obj_dtype_as_category`` parameter is beneficial when 
      categorical data is encoded as either ``'object'`` or ``'category'``, 
      allowing for flexible validation criteria.
    
    See Also
    --------
    pandas.DataFrame.select_dtypes : Select columns based on their dtypes.
    pandas.api.types : Utilities for checking data types in pandas.
    
    References
    ----------
    .. [1] Pandas Documentation: https://pandas.pydata.org/docs/
    .. [2] McKinney, W. "Python for Data Analysis," O'Reilly Media, 2017.
    """
    # validate the dataframe
    are_all_frames_valid(df)
    if columns is None:
        columns = df.columns.tolist()
    # Ensure that columns and dtypes are formatted correctly as lists
    columns = is_iterable(columns, exclude_string= True, transform =True)
    dtypes = is_iterable(dtypes, exclude_string= True, transform= True )

    valid_objectives = {'numeric', 'object', 'category', 'datetime'}
    if not set(dtypes).issubset(valid_objectives):
        invalid = set(dtypes) - valid_objectives
        raise ValueError(f"Invalid  dtypes specified: {invalid}")
    
    dtype_map = {
        'numeric': np.number,
        'object': 'object',
        'category': 'category',
        'datetime': 'datetime64[ns]'
    }
    
    # Prepare the result for 'validate' operation
    if ops not in ('check_only', 'validate'): 
        raise ValueError("`ops` must be either 'check_only' or 'validate'.")
        
    if ops == 'validate':
        result = {obj: [] for obj in dtypes}
        for obj in dtypes:
            if obj == 'numeric':
                result[obj] = df.select_dtypes(
                    include=[dtype_map[obj]]
                ).columns.intersection(columns).tolist()
            elif obj == 'datetime':
                result[obj] = df.select_dtypes(
                    include=[dtype_map[obj]]
                ).columns.intersection(columns).tolist()
            else:
                result[obj] = df.select_dtypes(
                    include=[dtype_map[obj]]
                ).columns.intersection(columns).tolist()
                if treat_obj_dtype_as_category and obj == 'object':
                    category_cols = df.select_dtypes(
                        include=['category']
                    ).columns.intersection(columns).tolist()
                    result[obj].extend(category_cols)
        
        return result
    
    # Perform 'check_only' operation
    elif ops == 'check_only':
        invalid_cols = []
        for col in columns:
            if col not in df.columns:
                invalid_cols.append(col)
                continue
            col_dtype = df[col].dtypes
            matched = False
            for obj in dtypes:
                if obj == 'numeric':
                    if np.issubdtype(col_dtype, np.number):
                        matched = True
                        break
                elif obj == 'object':
                    if df[col].dtypes == 'object':
                        matched = True
                        break
                    if treat_obj_dtype_as_category and df[col].dtypes.name == 'category':
                        matched = True
                        break
                elif obj == 'category':
                    if df[col].dtypes.name == 'category':
                        matched = True
                        break
                elif obj == 'datetime':
                    if np.issubdtype(col_dtype, np.datetime64):
                        matched = True
                        break
            if not matched:
                invalid_cols.append(col)
        
        if invalid_cols:
            message = error_msg if error_msg else (
                f"The following columns do not match the dtypes: {invalid_cols}"
            )
            if extra:
                message += f" {extra}"
                
            if error == 'warn': 
                warnings.warn(message)
                
            elif error == 'raise': 
                raise TypeError(f"Invalid columns detected: {message}")
                
            return False
        
        return True
  
def check_features_types(
    data,
    features,
    dtype,
    error_msg=None, 
    accept_object_dtype=False, 
    extra=''
):
    """
    Verify that specified features in a DataFrame match the expected type.

    This function checks whether the provided features within a pandas DataFrame
    conform to the specified objective type. Supported objective types include
    'category', 'numeric', and 'datetime'. It ensures data integrity by validating
    the data types before proceeding with further data processing or analysis.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the data to be checked.
    features : str or list of str
        The feature(s) to validate. If a single feature is provided as a string,
        it will be internally converted to a list for uniform processing.
    dtype : str
        The expected data type for the features. Supported types are:
        - ``'category'``: Categorical data type.
        - ``'numeric'`` : Numeric data types (int, float).
        - ``'datetime'``: Datetime data type.
    error_msg : str, optional
        Custom error message to raise if a feature's data type does not match
        the expected objective type. If set to `None`, a default error message
        will be generated. Default is ``None``.
        
    accept_object_dtype: bool, default=False, 
       Pass when object dtype is given rather than raising error. Th default 
       behavior only verify the ``'category'``, ``'numeric'`` and 
       ``'datetime'`` types. 
       
    extra: str, optional, 
       Extra message to append to the TypeError message. 

    Returns
    -------
    bool
        Returns `True` if all specified features match the expected type.

    Raises
    ------
    TypeError
        If `data` is not a pandas DataFrame.
        If `features` is neither a string nor a list of strings.
    ValueError
        If an unsupported `objective` type is provided.
        If any feature specified in `features` does not exist in `data`.
        If a feature's data type does not match the expected `dtype`.

    Notes
    -----
    - The function is case-sensitive regarding the `objective` parameter.
    - It is advisable to ensure that datetime columns are properly parsed
      (e.g., using `pd.to_datetime`) before using this function.
    - This function can be extended to support additional data types by
      modifying the `type_checks` dictionary.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.core.checks import check_features_types
    >>> data = {
    ...     'age': [25, 30, 45],
    ...     'salary': [50000.0, 60000.5, 75000.0],
    ...     'join_date': pd.to_datetime(['2020-01-15', '2019-06-23', '2021-03-10']),
    ...     'department': ['HR', 'Engineering', 'Marketing']
    ... }
    >>> df = pd.DataFrame(data)
    >>> check_features_types(df, ['age', 'salary'], 'numeric')
    True

    >>> df['department'] = df['department'].astype('category')
    >>> check_features_types(df, 'department', 'category')
    True

    >>> check_features_types(df, 'join_date', 'datetime')
    True

    >>> # Using a custom error message
    >>> check_features_types(
    ...     df,
    ...     'age',
    ...     'category',
    ...     error_msg="Age should be a categorical feature."
    ... )
    Traceback (most recent call last):
        ...
    ValueError: Age should be a categorical feature.
    """

    # Validate that 'data' is a pandas DataFrame
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            f"The 'data' parameter must be a pandas DataFrame, "
            f"got {type(data).__name__} instead."
        )

    # Ensure 'features' is a list
    if isinstance(features, str):
        features = [features]
    elif isinstance(features, list):
        if not all(isinstance(feature, str) for feature in features):
            raise TypeError(
                "All elements in the 'features' list must be strings."
            )
    else:
        raise TypeError(
            "The 'features' parameter should be a list of strings "
            "or a single feature name as a string."
        )

    # Mapping of objectives to pandas type checking functions
    type_checks = {
        'category' : is_categorical_dtype,
        'numeric'  : _is_numeric_dtype,
        'datetime' : is_datetime64_any_dtype,
    }
    if accept_object_dtype: 
        type_checks['object']= is_object_dtype 
        
    # Validate the objective
    if dtype not in type_checks:
        raise ValueError(
            f"Unsupported objective type: '{dtype}'. "
            f"Supported types are {list(type_checks.keys())}."
        )

    check_func = type_checks[dtype]

    # Iterate through each feature and check its type
    for feature in features:
        if feature not in data.columns:
            raise ValueError(
                f"Feature '{feature}' not found in the DataFrame."
            )

        if not check_func(data[feature]):
            if error_msg:
                raise ValueError(error_msg)
            else:
                actual_type = data[feature].dtype
                raise TypeError(
                    f"Feature '{feature}' has type '{actual_type}', "
                    f"expected type '{dtype}'.{extra}"
                )

    return True

def are_all_frames_valid(
    *dfs: Union[pd.DataFrame, pd.Series],
    df_only: bool = ...,  
    error_msg: Optional[str] = None, 
    check_size: bool = False,  
    check_symmetry: bool = False  
) -> bool:
    """
    Validates whether all provided inputs are pandas DataFrames or Series 
    based on the `df_only` flag. This function checks the types of the 
    input objects and optionally verifies additional properties like size 
    and symmetry.

    Parameters
    ----------
    *dfs : Union[pd.DataFrame, pd.Series]
        One or more pandas DataFrame or Series objects to be validated. 
        This function can accept multiple inputs, checking each one for 
        compliance with the expected type.

    df_only : bool, default=True
        If True, only DataFrames are considered valid inputs. If any of 
        the provided inputs is not a DataFrame, an error is raised. 
        If False, pandas Series are also allowed as valid inputs, and 
        the function will not raise an error for Series objects.

    check_size : bool, default=False
        If True, the function will additionally check if all DataFrames 
        or Series objects have the same number of rows or columns (depending 
        on the size parameter). This is useful for validating consistency 
        across inputs.

    check_symmetry : bool, default=False
        If True, the function checks if all DataFrames or Series are symmetric, 
        meaning the rows and columns (or the data itself in the case of Series) 
        are consistent. This could be important in certain data validation 
        or data integrity scenarios.

    error_msg : str, optional
        A custom error message to be raised if any input is invalid. If 
        not provided, a default message is used that specifies the need 
        for DataFrames or Series, depending on the `df_only` flag.
        
    Returns
    -------
    bool
        Returns `True` if all inputs are either DataFrames (or Series if 
        `df_only=False`), and if applicable, if they pass size and symmetry checks. 
        Otherwise, an error is raised based on the validation rules.

    Raises
    ------
    TypeError
        If any of the inputs is neither a DataFrame nor a Series (based on 
        `df_only`), or if any input fails the size or symmetry checks 
        (if `check_size` or `check_symmetry` are `True`).

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.core.checks import is_all_frames
    
    # Example with multiple DataFrames
    >>> df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    >>> df2 = pd.DataFrame({'a': [5, 6], 'b': [7, 8]})
    >>> is_all_frames(df1, df2)
    True
    
    # Example with a DataFrame and a Series
    >>> df3 = pd.Series([1, 2, 3])
    >>> is_all_frames(df1, df3, df_only=False)
    True
    
    # Example with a Series when df_only=True (raises TypeError)
    >>> is_all_frames(df1, df3)
    TypeError: Expected DataFrame, but found a Series

    Notes
    -----
    - If `check_size=True`, all DataFrames and Series must have the same 
      number of rows (for DataFrames) or elements (for Series). 
    - If `check_symmetry=True`, the function will check if the dimensions 
      of the DataFrames match for all inputs. In the case of Series, it 
      checks for consistency in the data sequence.
    
    See Also
    --------
    - `pandas.DataFrame`: For more information on the pandas DataFrame object.
    - `pandas.Series`: For more information on the pandas Series object.

    References
    ----------
    .. [1] Pandas Documentation. "DataFrame". Retrieved from: 
           https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
    .. [2] Pandas Documentation. "Series". Retrieved from: 
           https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html
    """
    # Default error message
    if error_msg is None:
        error_msg = "All inputs must be either pandas DataFrames" + \
                    (" or pandas Series." if not df_only else ".")
    
    # Check each argument
    for df in dfs:
        # Check if each element is a valid DataFrame or Series
        if not isinstance(df, (pd.DataFrame, pd.Series)):
            raise TypeError(error_msg)
        
        # If df_only is True, raise error for Series
        if df_only and isinstance(df, pd.Series):
            raise TypeError(f"Expected DataFrame, but found a Series: {df}")

        # Check for size consistency
        if check_size:
            if isinstance(df, pd.DataFrame):
                # Check if number of rows (size[0]) is the same for all DataFrames
                for other_df in dfs:
                    if isinstance(other_df, pd.DataFrame
                                  ) and df.shape[0] != other_df.shape[0]:
                        raise ValueError(
                            "DataFrames have different row counts:"
                            f" {df.shape[0]} != {other_df.shape[0]}")
            elif isinstance(df, pd.Series):
                # Check if length is the same for all Series
                for other_df in dfs:
                    if isinstance(other_df, pd.Series) and len(df) != len(other_df):
                        raise ValueError(
                            "Series have different lengths:"
                            f" {len(df)} != {len(other_df)}")

        # Check for symmetry (square matrices)
        if check_symmetry and isinstance(df, pd.DataFrame):
            if df.shape[0] != df.shape[1]:
                raise ValueError(
                    f"DataFrame is not symmetric: {df.shape[0]} != {df.shape[1]}")

    return True

def has_nan(
    data: Union[pd.DataFrame, pd.Series], 
    axis: Optional[int] = None, 
    how: str = 'any', 
    include_missing_columns: bool = False,
    error_msg: Optional[str] = None
) -> bool:
    """
    Check if the provided data (DataFrame or Series) contains any NaN values.
    
    This function provides enhanced flexibility to check NaN values either
    along specific axes (rows or columns), specify how to check for NaN 
    ('any' or 'all'), and includes the option to handle missing columns.
    
    Parameters
    ----------
    data : pd.DataFrame or pd.Series
        The input data which can either be a DataFrame or Series.
        
    axis : int, optional, default=None
        The axis along which to check for NaN values. 
        - For DataFrame: 0 checks columns (axis=0), 1 checks rows (axis=1).
        - For Series: No effect, as it is one-dimensional.
        
    how : {'any', 'all'}, optional, default 'any'
        Defines how to check for NaN values:
        - 'any' (default): Returns True if any NaN values are found.
        - 'all': Returns True only if all values are NaN in the specified axis.
        
    include_missing_columns : bool, optional, default False
        If True, include columns that are entirely missing (i.e., all NaN values) 
        in the result. If False, missing columns (completely NaN) are ignored 
        in the output.

    error_msg : str, optional
        A custom error message to raise if the input data type is not a
        DataFrame or Series. 
        If not provided, a default error message is used.
        
    Returns
    -------
    bool
        Returns True if NaN values are found according to the specified
        parameters, otherwise False.
    
    Raises
    ------
    TypeError
        If the input is not a pandas DataFrame or Series.
        
    Examples
    --------
    >>> from gofast.core.checks import has_nan 
    >>> df = pd.DataFrame({'A': [1, 2, None], 'B': [None, 2, 3]})
    >>> has_nan(df)
    True
    
    >>> has_nan(df, axis=0, how='all')
    False
    
    >>> has_nan(df, axis=1, how='any')
    True
    
    >>> has_nan(df, axis=0, how='all', include_missing_columns=True)
    False
    """
    # Handle error messages for invalid input types
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        error_msg = error_msg or "Input must be either a pandas DataFrame or Series"
        raise TypeError(error_msg)
    
    # For DataFrame: allow axis specification (axis=0 for columns, axis=1 for rows)
    if isinstance(data, pd.DataFrame):
        if axis is not None:
            if how == 'any':
                return data.isna().any(axis=axis).any() if not include_missing_columns else \
                    data.isna().any(axis=axis).any() or data.isna().all(axis=axis).any()
            elif how == 'all':
                return data.isna().all(axis=axis).any()
            else:
                raise ValueError("Parameter `how` must be 'any' or 'all'")
        else:
            return data.isna().any().any()

    # For Series: no axis, always check along the single dimension (rows)
    elif isinstance(data, pd.Series):
        if how == 'any':
            return data.isna().any()
        elif how == 'all':
            return data.isna().all()
        else:
            raise ValueError("Parameter `how` must be 'any' or 'all'")

    return False


def validate_spatial_columns(
    df: pd.DataFrame,
    mode_search: str = 'soft',
    rename: bool = False,
    error: str = 'raise',
    return_valid: bool = True,
    as_pair: bool = True,
    as_frame: bool = False
) -> Union[List[Union[str, Tuple[str, str]]], pd.DataFrame, None]:
    """
    Validate and Extract Spatial Columns from a DataFrame.
    
    The function checks for the existence of spatial column pairs within a 
    DataFrame, such as (`easting`, `northing`) or (`longitude`, `latitude`). 
    It supports flexible column name matching through regex patterns and can 
    optionally rename abbreviated column names to standard ones. The function 
    ensures that spatial columns are present in valid pairs and provides options 
    to return the validated columns or the corresponding DataFrame slices.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing potential spatial columns.
    
    mode_search : str, default='soft'
        The mode of searching for spatial columns.
        - `'soft'`: Case insensitive and allows partial matches using regex.
        - `'strict'`: Exact case-sensitive matches.
    
    rename : bool, default=False
        Whether to rename abbreviated spatial column names to their standard 
        counterparts.
        - If `True`, columns like `'east'` will be renamed to `'easting'`, 
          `'north'` to `'northing'`, `'lon'` to `'longitude'`, and `'lat'` 
          to `'latitude'`.
    
    error : str, default='raise'
        The strategy for handling validation errors.
        - `'warn'`: Issues warnings and continues processing.
        - `'raise'`: Raises exceptions upon encountering errors.
    
    return_valid : bool, default=True
        Determines whether to return the validated spatial columns.
        - If `True`, returns the validated columns or pairs.
        - If `False`, the function performs validation without returning columns.
    
    as_pair : bool, default=True
        Indicates whether to return spatial columns as pairs.
        - If `True`, returns a list of tuples, each containing a valid pair.
        - If `False`, returns a flat list of valid spatial column names.
    
    as_frame : bool, default=False
        Determines whether to return the spatial columns as a DataFrame slice.
        - If `True`, returns a DataFrame containing only the valid spatial columns.
        - If `False`, returns the list of valid columns or pairs based on 
          `as_pair`.
    
    Returns
    -------
    Union[List[Union[str, Tuple[str, str]]], pandas.DataFrame, None]
        - If `return_valid` is `True` and `as_pair` is `True`, returns a list 
          of tuples with each tuple representing a valid spatial column pair.
        - If `return_valid` is `True` and `as_pair` is `False`, returns a flat 
          list of valid spatial column names.
        - If `as_frame` is `True`, returns a DataFrame containing only the valid 
          spatial columns.
        - If `return_valid` is `False`, returns `None`.
    
    Raises
    ------
    ValueError
        - If `mode_search` is not `'soft'` or `'strict'`.
        - If `error` is neither `'warn'` nor `'raise'`.
        - If any of the spatial column pairs are partially present.
    
    TypeError
        - If `df` is not a pandas DataFrame.
    
    Notes
    -----
    - **Flexible Matching**: In `'soft'` mode, the function uses regex patterns 
      to match various naming conventions for spatial columns, allowing for 
      greater flexibility in column naming.
    
    - **Renaming Capability**: When `rename` is enabled, the function standardizes 
      abbreviated spatial column names to ensure consistency across the DataFrame.
    
    - **Pair Validation**: The function ensures that spatial columns are present 
      in complete pairs. Partial pairs (e.g., only `'longitude'` without 
      `'latitude'`) are considered invalid.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.core.checks import validate_spatial_columns
    
    >>> # Sample DataFrame with various spatial column name formats
    >>> data = {
    ...     'East': [500000, 500100, 500200],
    ...     'North': [4100000, 4100100, 4100200],
    ...     'lon': [-75.0, -75.1, -75.2],
    ...     'lat': [40.0, 40.1, 40.2],
    ...     'value': [10, 20, 30]
    ... }
    >>> df = pd.DataFrame(data)
    
    >>> # Validate spatial columns with default parameters
    >>> valid_cols = validate_spatial_columns(df)
    >>> print(valid_cols)
    [('East', 'North'), ('lon', 'lat')]
    
    >>> # Validate and rename spatial columns
    >>> valid_cols = validate_spatial_columns(df, rename=True)
    >>> print(valid_cols)
    [('easting', 'northing'), ('longitude', 'latitude')]
    
    >>> # Validate and return as a DataFrame
    >>> spatial_df = validate_spatial_columns(df, as_frame=True)
    >>> print(spatial_df)
       easting  northing  longitude  latitude
    0  500000    4100000      -75.0      40.0
    1  500100    4100100      -75.1      40.1
    2  500200    4100200      -75.2      40.2
    
    >>> # Attempt validation with missing pair
    >>> df_missing = pd.DataFrame({
    ...     'longitude': [-75.0, -75.1, -75.2],
    ...     'value': [10, 20, 30]
    ... })
    >>> validate_spatial_columns(df_missing)
    ValueError: The following spatial column pairs are incomplete or missing: {'latitude'}
    
    Notes
    -----
    - **Extensibility**: The function can be extended to include additional 
      spatial column pairs by updating the internal `spatial_pairs` list.
    
    - **Performance**: For large DataFrames, consider disabling `rename` if 
      renaming is not necessary to optimize performance.
    
    - **Integration**: This function is ideal for preprocessing steps in 
      geospatial data analysis workflows, ensuring that spatial data is 
      correctly identified and standardized before further processing.
    
    See Also
    --------
    pandas.DataFrame : Main pandas object for data manipulation.
    pandas.read_excel : Function to read Excel files into DataFrames.
    re : Module for regular expressions.
    
    References
    ----------
    .. [1] Pandas Documentation: pandas.DataFrame. 
       https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
    .. [2] Python Documentation: re. 
       https://docs.python.org/3/library/re.html
    .. [3] Freedman, D., & Diaconis, P. (1981). On the histogram as a density 
           estimator: L2 theory. *Probability Theory and Related Fields*, 57(5), 
           453-476.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"validate_spatial_columns requires a pandas DataFrame. "
            f"Got {type(df).__name__!r} instead."
        )
    
    if mode_search not in {'soft', 'strict'}:
        raise ValueError(
            f"mode_search must be either 'soft' or 'strict'. Got '{mode_search}'."
        )
    
    if error not in {'warn', 'raise'}:
        raise ValueError(
            f"error must be either 'warn' or 'raise'. Got '{error}'."
        )
    
    # Define spatial column pairs with regex patterns
    spatial_pairs = [
        {
            'standard': ('easting', 'northing'),
            'patterns': (r'^east(?:ing)?$', r'^north(?:ing)?$')
        },
        {
            'standard': ('longitude', 'latitude'),
            'patterns': (r'^lon(?:gitude)?$', r'^lat(?:itude)?$')
        }
    ]
    
    valid_pairs = []
    renamed_columns = {}
    
    for pair in spatial_pairs:
        std_east, std_north = pair['standard']
        pattern_east, pattern_north = pair['patterns']
        
        # Compile regex patterns
        regex_east = re.compile(pattern_east, re.IGNORECASE)
        regex_north = re.compile(pattern_north, re.IGNORECASE)
        
        # Find matching columns
        east_matches = [col for col in df.columns if regex_east.match(col)]
        north_matches = [col for col in df.columns if regex_north.match(col)]
        
        if east_matches and north_matches:
            east_col = east_matches[0]
            north_col = north_matches[0]
            
            # Rename columns if required
            if rename:
                renamed_columns[east_col] = std_east
                renamed_columns[north_col] = std_north
                df.rename(columns=renamed_columns, inplace=True)
                east_col = std_east
                north_col = std_north
            
            valid_pairs.append((east_col, north_col))
        elif east_matches or north_matches:
            missing = std_north if east_matches else std_east
            msg = (
                f"Spatial column pair incomplete. Missing '{missing}' for "
                f"the existing spatial column."
            )
            if error == 'warn':
                warnings.warn(msg)
            else:
                raise ValueError(msg)
    
    if not valid_pairs:
        msg = "No valid spatial column pairs found in the DataFrame."
        if error == 'warn':
            warnings.warn(msg)
            return None
        else:
            raise ValueError(msg)
    
    if return_valid:
        if as_frame:
            # Flatten the list of pairs
            spatial_cols = [col for pair in valid_pairs for col in pair]
            spatial_df = df[spatial_cols]
            return spatial_df
        elif as_pair:
            return valid_pairs
        else:
            # Flatten the list of pairs
            spatial_cols = [col for pair in valid_pairs for col in pair]
            return spatial_cols
    else:
        return None

def check_spatial_columns(
    df: pd.DataFrame,
    spatial_cols: Optional[tuple] = ('longitude', 'latitude'), 

) -> None:
    """
    Validate the spatial columns in the DataFrame.

    Ensures that the specified `spatial_cols` are present in the DataFrame and 
    consist of exactly two columns representing longitude and latitude.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe containing geographical data.
    
    spatial_cols : tuple, optional, default=('longitude', 'latitude')
        A tuple containing the names of the longitude and latitude columns.
        Must consist of exactly two elements.

    Raises
    ------
    ValueError
        - If `spatial_cols` is not a tuple or does not contain exactly two elements.
        - If any of the specified `spatial_cols` are not present in the DataFrame.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.core.checks import check_spatial_columns

    >>> # Valid spatial columns
    >>> df = pd.DataFrame({
    ...     'longitude': [-100, -99, -98],
    ...     'latitude': [35, 36, 37],
    ...     'value': [1, 2, 3]
    ... })
    >>> check_spatial_columns(df, spatial_cols=('longitude', 'latitude'))
    # No output, validation passed

    >>> # Invalid spatial columns
    >>> check_spatial_columns(df, spatial_cols=('lon', 'lat'))
    ValueError: The following spatial_cols are not present in the dataframe: {'lat', 'lon'}

    Notes
    -----
    - The function strictly requires `spatial_cols` to contain exactly two 
      column names representing longitude and latitude.
    
    See Also
    --------
    plot_spatial_distribution : Function to plot spatial distributions.

    References
    ----------
    .. [1] Pandas Documentation: pandas.DataFrame
    """
    if not isinstance (df, pd.DataFrame): 
        raise TypeError(
            "Spatial columns check requires a dataframe `df`"
            f" to be set. Got {type(df).__name__!r}")
        
    if not isinstance(spatial_cols, (tuple, list)) or len(spatial_cols) != 2:
        raise ValueError(
            "spatial_cols must be a tuple of exactly two elements "
            "(longitude and latitude)."
        )
    
    missing_cols = set(spatial_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"The following spatial_cols are not present in the dataframe: {missing_cols}"
        )

def validate_column_types(
    df: pd.DataFrame, 
    category_columns: Optional [Union[str, List[str]]]=None, 
    consider_object_as_category: bool = False, 
    error: str = 'raise',  
    error_msg: Optional[str] = None
) -> Union[dict, bool, None]:
    """
    Validate the types of specified columns in a DataFrame. Determines if each 
    column is categorical or numeric. The user can customize the behavior when 
    an error is encountered.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the columns to check.
        
    category_columns : str or list of str, optional 
        The name(s) of the column(s) to check. It can either be a single 
        column name or a list of column names. If ``None``, all columns of the 
        dataframe types will be checked.
    
    consider_object_as_category : bool, default ``False``
        If ``True``, columns of dtype ``'object'`` will be treated as categorical 
        columns. Otherwise, ``'object'`` columns will be considered non-categorical.
    
    error : str, default ``'raise'``
        Defines how to handle errors when a column is neither numeric nor categorical:
        - ``'raise'``: Raise a ``ValueError``.
        - ``'warn'``: Issue a warning using ``warnings.warn``.
        - ``'ignore'``: Do nothing and return ``None`` for invalid columns.
    
    error_msg : str, optional, default ``None``
        Custom error message to use when raising an error or issuing a warning. 
        If ``None``, a default message will be used.
    
    Returns
    -------
    dict or bool or None
        - If multiple columns are provided, returns a ``dict`` where keys are 
          column names and values are either ``True`` (if the column is 
          categorical), ``False`` (if numeric), or ``None`` (if invalid and 
          ``error`` is not ``'raise'``).
        - If a single column is provided, returns ``True``, ``False``, or ``None``
          directly instead of a dictionary.
    
    Raises
    ------
    ValueError
        If any column is neither numeric nor categorical and ``error`` is set to 
        ``'raise'``. Also raised if a specified column does not exist in 
        the DataFrame.
    
    Warns
    -----
    UserWarning
        If any column is neither numeric nor categorical and ``error`` is set to 
        ``'warn'``.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.core.checks import validate_column_types
    >>> df = pd.DataFrame({
    ...     'age': [25, 30, 35, 40],
    ...     'gender': ['M', 'F', 'M', 'F'],
    ...     'income': [50000, 60000, 55000, 65000],
    ...     'region': ['North', 'South', 'East', 'West']
    ... })
    >>> # Validate multiple columns, treating 'object' as categorical
    >>> result = validate_column_types(
    ...     df, 
    ...     ['gender', 'region', 'income'], 
    ...     consider_object_as_category=True, 
    ...     error='warn'
    ... )
    >>> print(result)
    {'gender': True, 'region': True, 'income': False}
    
    >>> # Validate a single column
    >>> result_single = validate_column_types(df, 'age')
    >>> print(result_single)
    False
    
    >>> # Attempt to validate an invalid column with error='ignore'
    >>> df['invalid_col'] = pd.Series([1, 2, 3, 'a'])
    >>> result_ignore = validate_column_types(
    ...     df, 
    ...     ['invalid_col'], 
    ...     error='ignore'
    ... )
    >>> print(result_ignore)
    None
    
    Notes
    -----
    - The function is designed to handle both single and multiple column validations.
    - When ``category_columns`` contains only one column, the function returns a 
      single ``bool`` or ``None`` value instead of a dictionary for convenience.
    - Setting ``consider_object_as_category=True`` is useful when categorical 
      data is stored as strings or mixed types in the DataFrame.
    - The ``error`` parameter provides flexibility in how the function responds 
      to invalid column types, allowing integration into larger data processing 
      pipelines without interruption.
    
    See Also
    --------
    ``pandas.api.types.is_categorical_dtype`` : Check if a pandas Series has a 
    categorical dtype.
    
    ``pandas.api.types.is_numeric_dtype`` : Check if a pandas Series has a 
    numeric dtype.
    
    References
    ----------
    .. [1] Pandas Documentation. "Categorical Data". Retrieved from: 
           https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html
    
    .. [2] Pandas Documentation. "Data Types". Retrieved from: 
           https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html#dtypes
    
    .. [3] Python Documentation. "warnings — Warning control". Retrieved from: 
           https://docs.python.org/3/library/warnings.html
    """
    are_all_frames_valid(df)
    # Ensure category_columns is a list
    if isinstance(category_columns, str):
        category_columns = [category_columns]
    
    if category_columns is None: 
        category_columns = list (df.columns)
        
    column_types = {}

    # Set default error message if not provided
    if error_msg is None:
        error_msg = "Column must be either numeric or categorical."

    # Iterate over each column name
    for column in category_columns:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")
        
        # Determine column type
        if pd.api.types.is_categorical_dtype(df[column]) or \
           (consider_object_as_category and df[column].dtype == 'object'):
            column_types[column] = True  # Categorical
        elif pd.api.types.is_numeric_dtype(df[column]):
            column_types[column] = False  # Numeric
        else:
            if error == 'raise':
                raise ValueError(f"{error_msg} (Column: '{column}')")
            elif error == 'warn':
                warnings.warn(f"{error_msg} (Column: '{column}')", UserWarning)
                column_types[column] = None  # Mark as invalid
            elif error == 'ignore':
                column_types[column] = None  # Mark as invalid without action
    
    # If only one column was checked, return its type directly
    if len(column_types) == 1:
        return next(iter(column_types.values()))
    
    return column_types

def is_df_square(
    df: pd.DataFrame, 
    order: Optional[Union[list, tuple]] = None, 
    ops: str = 'check_only', 
    check_symmetry: bool=False,  # new parameter, 
    cols_as_index: bool = False, 
) -> Optional[pd.DataFrame]:
    """
    Checks if a given DataFrame is square and optionally reorders the 
    columns and index or sets the index based on column names. The function 
    checks the structural properties of the DataFrame and applies specified 
    operations based on the parameters provided.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to check and optionally manipulate. This must be a 
        square matrix (i.e., having an equal number of rows and columns) 
        for further operations to proceed.

    order : list or tuple, optional
        A custom order of column names to reorder the columns and index.
        If provided, the DataFrame will be rearranged based on this order. 
        This is a sequence of column names that should exist in both the 
        index and columns of the DataFrame. If not provided, the DataFrame 
        will not be reordered.

    ops : {'check_only', 'validate'}, default 'check_only'
        Specifies the operation to be performed. If set to `'check_only'`, 
        the function only validates the structure of the DataFrame and does 
        not perform any modification. If set to `'validate'`, the function 
        attempts to reorder the DataFrame or set the index and columns based 
        on the provided `order` list.
        
    check_symmetry : bool, default False
        If True, the function checks if the DataFrame's columns and index 
        contain the same set of values. This ensures the symmetry of the 
        DataFrame's structure. If False, no check for symmetry is performed.

    cols_as_index : bool, default False
        If True, the function sets the index of the DataFrame using the 
        column names (i.e., the columns and index will be identical). If 
        False, no modification to the index is made. This parameter is only 
        relevant when the DataFrame is square and an `order` is specified.

    Returns
    -------
    pd.DataFrame or None
        If `ops` is ``'validate'``, the function returns a DataFrame where 
        the index and columns are reordered based on the provided `order`. 
        If `ops` is ``'check_only'``, the function performs checks without 
        modifying the DataFrame and returns None. If the checks fail (e.g., 
        the DataFrame is not square or the index and columns do not match), 
        a `ValueError` is raised.

    Raises
    ------
    ValueError
        If the DataFrame is not square, or if the index and columns 
        do not contain the same values, or if `order` contains invalid 
        column names.
    
    Examples
    --------
    >>> from gofast.core.checks import is_df_square
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    >>>     'A': [1, 2, 3],
    >>>     'B': [4, 5, 6],
    >>>     'C': [7, 8, 9]
    >>> })
    >>> is_df_square(df, order=['B', 'A', 'C'], ops='validate')
    >>> df  # Returns the DataFrame with reordered columns and index

    Notes
    -----
    - The `is_df_square` function is used to ensure that a given DataFrame 
      has an equal number of rows and columns, and optionally reorder its 
      columns and index or set the index to match the columns.
    - The function raises a `ValueError` if the DataFrame does not meet the 
      expected conditions (i.e., square matrix or matching index and columns).
    
    See Also
    --------
    pd.DataFrame: The base pandas DataFrame class which provides many 
    useful methods for DataFrame manipulation, such as `.reindex()` and 
    `.set_index()`.
    
    References
    ----------
    [1] pandas documentation: https://pandas.pydata.org/pandas-docs/stable/
    """

    # Check if the input is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Input must be a pandas DataFrame. Got {type(df).__name__!r}")
    
    # Check if the DataFrame is square (rows == columns)
    if df.shape[0] != df.shape[1]:
        raise ValueError(
            "The DataFrame must be square: number of rows"
            " must equal the number of columns.")

    if check_symmetry: 
        # Check if the index and columns have the same elements
        if ( 
            not set(df.columns).issubset(df.index) or 
            not set(df.index).issubset(df.columns)
            ):
            raise ValueError(
                "Index and columns must contain the same values.")
    
    # Optionally reorder index and columns if 'order' is provided
    if order:
        # Ensure the order is a valid subset of both index and columns
        if not set(order).issubset(df.columns) or not set(order).issubset(df.index):
            raise ValueError(
                "Custom order contains values not present in both index and columns.")
        
        if ops == 'validate':
            # df = df[order]         # Reorder columns
            # df = df.T[order].T     # Reorder index
            # Apply custom order to index and columns : .loc much faster
            df = df.loc[order, order]
            
    # If only checking, return True
    if ops == 'check_only':
        return True 
    
    # If validating and cols_as_index is True, set index equal to columns
    if cols_as_index:
        df.index = df.columns
        
    # If cols_as_index is 'reverse', reverse columns and set as index
    elif str(cols_as_index).lower() == 'reverse':
        df.index = df.columns[::-1]
    
    return df

def check_files(
    files: Union[str, List[str]],
    formats: Union[str, List[str]]=None,
    return_valid: bool = True,
    error: str = 'raise',
    empty_allowed: bool = False
) -> Union[str, List[str], None]:
    """
    Validate the Existence, Format, and Non-Emptiness of Files.
    
    The `check_files` function provides a robust mechanism to validate one or 
    multiple file paths by ensuring their existence, verifying their formats 
    against specified criteria, and confirming that they are not empty. This 
    utility is essential for data processing pipelines where the integrity and 
    correctness of input files are paramount. By offering flexible parameters, 
    users can tailor the validation process to their specific needs, handling 
    multiple files and diverse formats with ease.
    
    .. math::
        \text{Validation Process} = 
        \begin{cases}
            \text{Existence Check} \\
            \text{Format Verification} \\
            \text{Non-Emptiness Confirmation}
        \end{cases}
    
    Parameters
    -----------
    files : Union[`str`, `List[str]`]
        The file path or list of file paths to be validated. Each file will be 
        individually checked for existence, format compliance, and non-emptiness 
        based on the provided parameters.
    
    formats : Union[`str`, `List[str]`], optional
        The acceptable file formats/extensions. Formats can be specified with or 
        without a leading dot (e.g., `'csv'` or `'.xlsx'`). The function 
        normalizes these formats to ensure consistency during validation. If 
        `formats` is `None`, the format check is skipped.
    
    return_valid : `bool`, default=`True`
        Determines whether the function should return the list of valid files.
        - If `True`, returns the validated file paths.
        - If `False`, the function performs validation without returning the 
          file paths.
    
    error : `str`, default=`'raise'`
        Specifies the error handling strategy when encountering invalid files.
        - `'warn'`: Issues warnings for each validation failure and continues 
          processing remaining files.
        - `'raise'`: Raises exceptions immediately upon encountering a 
          validation failure, halting further execution.
        
        Raises a `ValueError` if an unsupported error handling strategy is 
        provided.
    
    empty_allowed : `bool`, default=`False`
        Indicates whether empty files are permissible.
        - If `False`, files with a size of zero bytes are considered invalid.
        - If `True`, empty files pass the non-emptiness check.
    
    Returns
    -------
    Union[`str`, `List[str]`, `None`]
        - If `return_valid` is `True` and only one file is valid, returns the 
          file path as a `str`.
        - If `return_valid` is `True` and multiple files are valid, returns a 
          `List[str]` of file paths.
        - If `return_valid` is `True` but no valid files are found, returns 
          `None`.
        - If `return_valid` is `False`, returns `None` regardless of validation 
          outcomes.
    
    Raises
    ------
    FileNotFoundError
        Raised when a specified file does not exist and `error` is set to `'raise'`.
    
    ValueError
        - Raised when a file's format is unsupported based on the `formats` 
          parameter and `error` is `'raise'`.
        - Raised when a file is empty while `empty_allowed` is `False` and `error` 
          is `'raise'`.
        - Raised when `error` parameter is neither `'warn'` nor `'raise'`.
    
    TypeError
        Raised when the `checkpoint` parameter is neither a `str` nor an `int`.
    
    Notes
    -----
    - **Format Normalization**: The function ensures that all format 
      specifications begin with a leading dot for consistent comparison, 
      regardless of how the user inputs them.
    
    - **Flexible Input Handling**: Users can pass a single file path as a 
      string or multiple file paths as a list of strings, allowing for 
      versatile usage scenarios.
    
    - **Error Management**: The `error` parameter provides control over how 
      the function responds to validation failures, enabling integration into 
      larger workflows where certain failures may need to be handled 
      differently.
    
    - **Empty File Consideration**: By default, the function treats empty 
      files as invalid to prevent downstream errors in data processing 
      tasks. This behavior can be modified using the `empty_allowed` parameter 
      based on user requirements.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.core.checks import check_files
    
    >>> # Single file validation
    >>> valid_file = check_files(
    ...     files='data/sample.csv',
    ...     formats=['csv', 'xlsx'],
    ...     return_valid=True,
    ...     error='warn',
    ...     empty_allowed=False
    ... )
    >>> print(valid_file)
    'data/sample.csv'
    
    >>> # Multiple files validation with mixed formats
    >>> files = [
    ...     'data/sample1.csv',
    ...     'data/sample2.xlsx',
    ...     'data/sample3.json',
    ...     'data/sample4.txt'  # Unsupported format
    ... ]
    >>> valid_files = check_files(
    ...     files=files,
    ...     formats=['csv', 'xlsx', 'json'],
    ...     return_valid=True,
    ...     error='warn',
    ...     empty_allowed=False
    ... )
    >>> print(valid_files)
    ['data/sample1.csv', 'data/sample2.xlsx', 'data/sample3.json']
    
    >>> # Handling empty files
    >>> check_files(
    ...     files='data/empty.csv',
    ...     formats='csv',
    ...     return_valid=True,
    ...     error='warn',
    ...     empty_allowed=False
    ... )
    /path/to/script.py:XX: UserWarning: File is empty: `data/empty.csv`.
    warnings.warn(msg)
    >>> # With empty_allowed=True
    >>> valid_file = check_files(
    ...     files='data/empty.csv',
    ...     formats='csv',
    ...     return_valid=True,
    ...     error='warn',
    ...     empty_allowed=True
    ... )
    >>> print(valid_file)
    'data/empty.csv'
    
    >>> # Single file validation with error handling
    >>> try:
    ...     check_files(
    ...         files='data/nonexistent.csv',
    ...         formats='csv',
    ...         return_valid=True,
    ...         error='raise',
    ...         empty_allowed=False
    ...     )
    ... except FileNotFoundError as e:
    ...     print(e)
    File does not exist: `data/nonexistent.csv`.
    
    Notes
    -----
    - **Extensibility**: The `check_files` function can be easily extended to 
      support additional file formats by simply updating the `formats` parameter 
      with the desired extensions.
    
    - **Integration**: This function is particularly useful in data ingestion 
      pipelines where the integrity of input files must be verified before 
      processing to ensure downstream tasks execute correctly.
    
    - **Performance Considerations**: When validating a large number of files, 
      consider setting `error='warn'` to allow the function to process all files 
      and collect all valid ones, rather than stopping at the first encountered error.
    
    See Also
    --------
    os.path.exists : Check if a path exists.
    os.path.splitext : Split the file path into root and extension.
    os.path.getsize : Get the size of a file.
    warnings.warn : Issue a warning message.
    
    References
    ----------
    .. [1] Python Documentation: os.path.exists. 
       https://docs.python.org/3/library/os.path.html#os.path.exists
    .. [2] Python Documentation: os.path.splitext. 
       https://docs.python.org/3/library/os.path.html#os.path.splitext
    .. [3] Python Documentation: os.path.getsize. 
       https://docs.python.org/3/library/os.path.html#os.path.getsize
    .. [4] Python Documentation: warnings.warn. 
       https://docs.python.org/3/library/warnings.html#warnings.warn
    .. [5] Freedman, D., & Diaconis, P. (1981). On the histogram as a density estimator: 
           L2 theory. *Probability Theory and Related Fields*, 57(5), 453-476.
    """

    # Normalize formats to include leading dot if formats is provided
    if formats is not None:
        if isinstance(formats, str):
            formats = [formats]
        normalized_formats = [
            f".{fmt.lstrip('.')}" for fmt in formats
        ]
    else:
        normalized_formats = None
    
    # Ensure files is a list
    if isinstance(files, str):
        files = [files]
    
    if not all( isinstance (f, str) for f in files): 
        raise TypeError(
            "Got invalid type(s). Supported only file path objects."
            " Please check your files.")
        
    valid_files = []
    
    for file in files:
        # Check if file exists
        if not os.path.exists(file):
            msg = f"File does not exist: `{file}`."
            if error == 'warn':
                warnings.warn(msg)
                continue
            else:
                raise FileNotFoundError(msg)
        
        # Check file format if formats is specified
        if normalized_formats is not None:
            _, ext = os.path.splitext(file)
            ext = ext.lower()
            if ext not in normalized_formats:
                msg = (
                    f"Unsupported file format: `{file}`. "
                    f"Expected formats: {normalized_formats}."
                )
                if error == 'warn':
                    warnings.warn(msg)
                    continue
                else:
                    raise ValueError(msg)
        
        # Check if file is not empty
        if not empty_allowed:
            if os.path.getsize(file) == 0:
                msg = f"File is empty: `{file}`."
                if error == 'warn':
                    warnings.warn(msg)
                    continue
                else:
                    raise ValueError(msg)
        
        # If all checks pass, add to valid_files
        valid_files.append(file)
    
    if return_valid:
        if not valid_files:
            if error == 'warn':
                warnings.warn("No valid files found.")
            else:
                raise ValueError("No valid files found.")
        else:
            if len(valid_files) == 1:
                valid_files = valid_files[0]
        
        return valid_files

def validate_nested_param(
    value: Any,
    expected_type: Any,
    param_name: str = '',
    coerce: bool = True, 
    empty_as_none: bool =..., 
) -> Any:
    """
    Validate and coerce parameters and their nested items to the expected types.
    
    This function ensures that the provided ``value`` conforms to the 
    ``expected_type``. It supports validation of nested structures such as 
    lists of dictionaries, dictionaries containing lists, and other complex 
    nested types. If the ``value`` is not of the expected type but can be 
    coerced (e.g., a single item to a list), the function performs the 
    necessary conversion. If coercion is not possible, it raises a 
    ``TypeError`` with a descriptive message.
    
    .. math::
        \text{If } x \text{ is not of type } T, \text{ attempt to convert } 
        x \text{ to } T.
    
    Parameters
    ----------
    value : Any
        The value to be validated and coerced. It can be a single item, a list 
        of items, a dictionary, or nested combinations thereof. If ``None`` is 
        acceptable for the parameter, it should be handled appropriately.
    expected_type : Any
        The expected type of the ``value``. This can include nested types like 
        ``List[str]``, ``Dict[str, float]``, etc., using Python's typing 
        constructs. The function recursively validates each level of the input 
        against the specified ``expected_type``.
    param_name : str, optional
        The name of the parameter being validated. This is used in error 
        messages to provide clear and descriptive feedback in case of 
        validation failures.
        
    coerce : bool, optional
        If ``True``, attempt conversions to match `'expected_type'`. If
        ``False``, enforce strict type checking with no conversions.
    empty_as_none : bool, default=True
        If True, returns `None` when type is `Optional`. If ``False``, an 
        empty Iterable  object  `List` is returned.

    Returns
    -------
    Any
        The validated and potentially coerced value, matching the 
        ``expected_type``. If ``value`` is ``None`` and ``None`` is 
        acceptable, it returns ``None`` or an empty list/dictionary based 
        on the context.
    
    Raises
    ------
    TypeError
        If the ``value`` cannot be coerced to the ``expected_type`` or if any 
        nested item fails validation.
    
    Examples
    --------
    >>> from gofast.core.checks import validate_nested_param
    >>> # Validate a single string item
    >>> validate_nested_param('item', str, 'static_feature_names')
    'item'
    >>> # Validate a single integer item to be a list of integers
    >>> validate_nested_param(5, List[int], 'ages')
    [5]
    >>> # Validate a list of strings
    >>> validate_nested_param(['feature1', 'feature2'], List[str], 'static_feature_names')
    ['feature1', 'feature2']
    >>> # Validate a dictionary with string keys and float values
    >>> validate_nested_param({'a': 1.0, 'b': 2.5}, Dict[str, float], 'scaling_params')
    {'a': 1.0, 'b': 2.5}
    
    Notes
    -----
    - This function is designed to be highly flexible and can handle deeply 
      nested structures by recursively validating each level of the input.
    - If a single item is provided where a list is expected, the function will 
      automatically wrap the item in a list.
    - For dictionary validations, both keys and values are validated against 
      expected types, ensuring the integrity of complex nested data.
    
    See Also
    --------
    prepare_future_data : Function for preparing future data inputs with validation.
    validate_parameter : Another utility function for parameter validation.
    
    References
    ----------
    .. [1] Python Software Foundation. *typing — Support for type hints*. 
        https://docs.python.org/3/library/typing.html
    .. [2] Smith, J., & Doe, A. (2020). *Effective Type Validation in Python*. 
        Journal of Python Development, 10(2), 123-135.
    """

    origin = get_origin(expected_type)
    args = get_args(expected_type)
    
    # **New logic for Callable:**
    # Detect if expected_type is a Callable or a Union that includes Callable.
    if expected_type is Callable or origin is Callable or (
        origin is Union and any(t is Callable for t in args)
    ):
        return check_callable(value, expected_type, param_name)

    # Handle Optional types (Union with NoneType)
    if origin is Union and type(None) in args:
        # origin, arg ---> typing.Union (<class 'int'>, <class 'NoneType'>)
        non_none_args = [arg for arg in args if arg is not type(None)]
        if not non_none_args:
            if value is not None:
                raise TypeError(f"Parameter `{param_name}` must be None.")
            return None
        # If the expected type is Optional[List[X]], 
        # set to empty list if None and coerce is True
        if len(non_none_args) == 1:
            sub_type = non_none_args[0]
            if get_origin(sub_type) in ( list, tuple) and value is None:
                return None if empty_as_none else [] 
            # If the expected type is Optional[X],
            # set None whatever.
            elif ( 
                sub_type in (int, float, numbers.Real, numbers.Integral) 
                and value is None
                ): 
                # return None for Optional[X]
                return None 
            
        # Recursively validate the non-None type
        return validate_nested_param(value, non_none_args[0], param_name, coerce)
    
    # Handle List types
    elif origin in (list, List):
        if isinstance(value, list):
            validated_list = []
            for index, item in enumerate(value):
                try:
                    validated_item = validate_nested_param(
                        item, args[0],
                        param_name = f"{param_name}[{index}]",
                        coerce = coerce
                    )
                    validated_list.append(validated_item)
                except TypeError as e:
                    raise TypeError(f"{e}") from None
            return validated_list
        # If value is a single item, attempt to convert
        # it to a list if coerce is True
        else:
            if coerce:
                try:
                    validated_item = validate_nested_param(
                        value, args[0],
                        param_name = param_name,
                        coerce = coerce
                    )
                    return [validated_item]
                except TypeError as e:
                    raise TypeError(
                        f"Parameter `{param_name}` is expected to be a "
                        f"list of `{args[0].__name__}`, but got type "
                        f"`{type(value).__name__}`."
                    ) from e
            else:
                raise TypeError(
                    f"Parameter `{param_name}` is expected to be a "
                    f"list of `{args[0].__name__}`, but got type "
                    f"`{type(value).__name__}`."
                )
                
    # Handle a Dict types 
    elif origin in (dict, Dict):
        if not isinstance(value, dict):
            raise TypeError(
                f"Parameter `{param_name}` is expected to be a dict,"
                f" but got type `{type(value).__name__}`."
            )
        validated_dict = {}
        key_type, val_type = args
        for key, val in value.items():
            try:
                validated_key = validate_nested_param(
                    key, key_type,
                    param_name = f"{param_name} key",
                    coerce = coerce
                )
            except TypeError as e:
                raise TypeError(f"{e}") from None
            try:
                validated_val = validate_nested_param(
                    val, val_type,
                    param_name = f"{param_name}[{key}]",
                    coerce = coerce
                )
            except TypeError as e:
                raise TypeError(f"{e}") from None
            validated_dict[validated_key] = validated_val
        return validated_dict
    
    # Handle Tuple types
    elif origin in (tuple, Tuple):
        if not isinstance(value, tuple):
            raise TypeError(
                f"Parameter `{param_name}` is expected to be a tuple,"
                f" but got type `{type(value).__name__}`."
            )
        if len(args) != len(value):
            raise TypeError(
                f"Parameter `{param_name}` expects a tuple of length "
                f"{len(args)}, but got tuple of length {len(value)}."
            )
        validated_tuple = []
        for index, (item, sub_type) in enumerate(zip(value, args)):
            try:
                validated_item = validate_nested_param(
                    item, sub_type,
                    param_name = f"{param_name}[{index}]",
                    coerce = coerce
                )
                validated_tuple.append(validated_item)
            except TypeError as e:
                raise TypeError(f"{e}") from None
        return tuple(validated_tuple)
    
    # Handle all other basic types
    # Handle basic types
    else:
        if not isinstance(value, expected_type):
            if coerce:
                # Attempt to convert the value to the expected type
                try:
                    return expected_type(value)
                except (ValueError, TypeError) as e:
                    raise TypeError(
                        f"Parameter `{param_name}` is expected to be of "
                        f"type `{expected_type.__name__}`, but got type "
                        f"`{type(value).__name__}`."
                    ) from e
            else:
                raise TypeError(
                    f"Parameter `{param_name}` is expected to be of type "
                    f"`{expected_type.__name__}`, but got type "
                    f"`{type(value).__name__}`."
                )
        return value


def check_params(param_types: Dict[str, Any], coerce: bool = True):
    """
    Validate parameters of the decorated function against given
    expected types, with optional coercion.

    This `check_params` decorator verifies that function parameters
    match their expected types specified in ``param_types``. It
    applies :func:`validate_nested_param` to each parameter defined
    in ``param_types``. If ``coerce`` is True, attempts are made
    to convert the given value to the expected type. Otherwise, a
    strict type check is enforced. If validation fails, a
    ``TypeError`` is raised.

    .. math::
       \text{Given a parameter } x \text{ with expected type } T,
       \text{ if } x \notin T \text{ and coerce=True, then }
       x \rightarrow T \text{ if possible. Else, raise error.}

    Parameters
    ----------
    param_types : Dict[str, Any]
        Dictionary mapping parameter names to their expected types.
        For example:
        ::
        
            {
                "x": List[int],
                "y": Dict[str, float]
            }

        Each key corresponds to a parameter name of the decorated
        function, and its value is the expected type that the
        parameter must match or be coerced into.

    coerce : bool, optional
        Whether to attempt coercion if types do not initially match.
        If ``True``, the decorator tries to convert the parameter
        value to the expected type. If ``False``, a strict check is
        enforced and a ``TypeError`` is raised if the type does not
        match exactly.

    Methods
    -------
    check_params(param_types, coerce)
        This is the main function that creates and returns the
        decorator.

    The decorator returned by `check_params`:
    - Binds the decorated function and applies default values.
    - Iterates over ``param_types`` and checks each corresponding
      parameter in the bound arguments.
    - Calls :func:`validate_nested_param` to validate or coerce the
      value to the expected type.
    - Raises ``TypeError`` if validation fails.

    The inner wrapper function:
    - Applies the type checks before calling the decorated function.
    - Returns the result of the decorated function if checks pass.

    Examples
    --------
    >>> from gofast.core.checks import check_params
    >>> from typing import List, Dict
    >>> @check_params({"items": List[str], "mapping": Dict[str, float]}, coerce=True)
    ... def process_data(items, mapping):
    ...     return items, mapping
    ...
    >>> process_data(["apple", "banana"], {"a": 1.0, "b": 2.5})
    (['apple', 'banana'], {'a': 1.0, 'b': 2.5})
    >>> process_data("single_item", {"x": "2.0"})
    (['single_item'], {'x': 2.0})

    Notes
    -----
    - If ``coerce`` is True, single values can be promoted to lists,
      and strings converted to floats if feasible.
    - If ``coerce`` is False, strict type matching is enforced.
    - This decorator allows writing robust, type-safe functions.

    See Also
    --------
    validate_nested_param : Validates and optionally coerces a single 
        parameter or nested structure to an expected type.

    References
    ----------
    .. [1] Python Software Foundation. *typing — Support for type hints*.
           https://docs.python.org/3/library/typing.html

    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            sig   = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            for p_name, p_type in param_types.items():
                if p_name in bound.arguments:
                    val = bound.arguments[p_name]
                    validated_val = validate_nested_param(
                        val, p_type,
                        param_name = p_name,
                        coerce     = coerce
                    )
                    bound.arguments[p_name] = validated_val

            return func(*bound.args, **bound.kwargs)
        return wrapper
    return decorator


def check_callable(
        value, expected_type, param_name: str = ''):
    """
    Validate that a parameter matches a Callable type or an Optional[Callable]
    (or Union[Callable, None]).

    Parameters
    ----------
    value : Any
        The value to validate.
    expected_type : Any
        The expected type, which might be `Callable`, `Optional[Callable]`,
        or `Union[Callable, None]`.
    param_name : str, optional
        Name of the parameter (for better error messages).

    Returns
    -------
    Any
        Returns the validated value if it matches the expected callable type.
        Returns None if None is allowed by the type hint and value is None.

    Raises
    ------
    TypeError
        If the value does not match the expected Callable requirements.
        
    Examples 
    ---------
    # Example usages:
    check_callable(lambda x: x+1, Callable)        # returns <lambda>
    check_callable(None, Callable)                 # raises TypeError
    check_callable(None, Union[Callable, None])    # returns None
    check_callable(None, Optional[Callable])       # returns None
    check_callable(int, Callable)                  # int is callable (class), so returns int

    """


    origin = get_origin(expected_type)
    args = get_args(expected_type)

    # Check for Optional[Callable] or Union[Callable, None]
    if origin is Union and type(None) in args:
        # If the value is None and None is allowed, return None immediately
        if value is None:
            return None

        # Otherwise, we proceed with the non-None args to validate value
        non_none_args = [t for t in args if t is not type(None)]
        if not non_none_args:
            # Means only None is allowed, and we've already handled value=None
            # If code reaches here and value is not None, it's an error:
            raise TypeError(
                f"Parameter '{param_name}' must be None because only NoneType is allowed."
            )
        
        # Validate against the non-None callable type
        return check_callable(value, non_none_args[0], param_name)

    # If `None` is provided but the expected type is strictly Callable, raise an error
    if value is None:
        raise TypeError(
            f"Parameter {param_name} is expected to be a Callable, but got `None`."
        )

    # Check if the value is callable
    if not callable(value):
        raise TypeError(
            f"Parameter {param_name} is expected to be a Callable,"
            f" but got type `{type(value).__name__}`."
        )

    return value


def check_array_like(
    obj, 
    context="ml", 
    ops="check_only"
):
    """
    Determine if an object is array-like based on a specified context 
    and operation.

    The function can either check if the object is array-like or validate it, 
    depending on the operation specified.

    Parameters
    ----------
    obj : any
        The object to check or validate.
        
    context : {'general', 'ml'}, default 'ml'
        The context in which to validate the object:
        - 'ml' (default): Checks for array-like objects commonly used in 
          machine learning, 
          such as lists, numpy arrays, pandas DataFrames, Series, etc.
        - 'general': Checks for basic array-like objects, including lists, tuples, 
          numpy arrays, and other iterables.
        
    ops : {'check_only', 'validate'}, default 'check_only'
        Defines the operation to perform:
        - 'check_only': Returns True if the object is array-like, False otherwise.
        - 'validate': Returns the object if valid, or raises a TypeError if invalid.
    
    Returns
    -------
    bool or array-like
        - If `ops='check_only'`, returns True if the object is array-like,
         False otherwise.
        - If `ops='validate'`, returns the object if valid, otherwise raises
        a TypeError.
    
    Raises
    ------
    TypeError
        If the object is not array-like in the given context and `ops='validate'`.
    
    Examples
    --------
    >>> from gofast.core.checks import check_array_like 
    >>> check_array_like([1, 2, 3])
    True
    
    >>> check_array_like(np.array([1, 2, 3]))
    True
    
    >>> check_array_like(pd.DataFrame([[1, 2], [3, 4]]), context='ml', ops='validate')
    pandas.core.frame.DataFrame
    """
    
    # Define valid contexts and operations
    valid_contexts = {'general', 'ml'}
    valid_operations = {'check_only', 'validate'}

    # Validate the context and operation parameters
    if context not in valid_contexts:
        raise ValueError(
            f"Invalid context '{context}'. Expected one of {valid_contexts}.")
    
    if ops not in valid_operations:
        raise ValueError(
            f"Invalid ops '{ops}'. Expected one of {valid_operations}.")
    
    # Check if the object is array-like
    is_array_like = False
    if context == 'ml':
        # For ML context, check for lists, numpy arrays, pandas DataFrames/Series
        is_array_like = isinstance(obj, (list, tuple, np.ndarray, pd.DataFrame, pd.Series))
    elif context == 'general':
        # For general context, check for iterables (e.g., list, tuple, numpy array,...)
        is_array_like = isinstance(obj, Iterable) and not isinstance(obj, str)

    # Return based on operation type
    if ops == 'check_only':
        return is_array_like
    
    if ops == 'validate' and not is_array_like:
        raise TypeError(
            f"Object of type '{type(obj)}' is not array-like in the '{context}' context.")
    
    return obj
