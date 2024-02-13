# -*- coding: utf-8 -*-
#   Licence:BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

""" 
`GoFast`_ Type variables
========================

The `GoFast` library employs specialized type variables to enhance the clarity 
and precision of type hints throughout the package. These custom type hints 
facilitate the accurate definition of function arguments and return types.

Some customized type variables are used throughout the GoFast package for type 
hinting. These type hints define the expected type of arguments in various 
functions and methods.

M
---
Represents an integer variable used to denote the number of rows in an ``Array``.

N
---
Similar to ``M``, ``N`` indicates the number of columns in an ``Array``. It 
is bound to an integer variable.

_T
---
Generic type variable representing `Any` type. It is kept as the default type 
variable for generic purposes.

U
---
Unlike `_T`, `U` stands for an undefined dimension, typically used to specify a
one-dimensional array. For example:

.. code-block:: python

    import numpy as np
    array = np.arange(4).shape   # Output: (4,)

S
---
Indicates the `Shape` of an array. It is bound by `M`, `U`, `N`. `U` represents
a one-dimensional array. While `Shape` is typically used for two-dimensional 
arrays, it can be extended to more dimensions using the :class:`AddShape`.

D
---
Denotes a `DType` object. It is bound to :class:`DType`.

Array
-----
Defined for one-dimensional arrays with an optional `DType`. For example:

.. code-block:: python

    import numpy as np
    from gofast._typing import TypeVar, Array, DType
    _T = TypeVar('_T', float)
    A = TypeVar('A', str, bytes)
    arr1: Array[_T, DType[_T]] = np.arange(21)
    arr2: Array[A, DType[A]] = arr1.astype('str')

NDArray
-------
Represents multi-dimensional arrays. Unlike `Array`, `NDArray` can have multiple
dimensions. Example usage:

.. code-block:: python

    import numpy as np
    from gofast.typing import TypeVar, Array, NDArray, DType
    _T = TypeVar('_T', int)
    U = TypeVar('U')
    multidarray = np.arange(7, 7).astype(np.int32)
    def accept_multid(arrays: NDArray[Array[_T, U], DType[_T]]=multidarray): ...

_Sub
----
Represents a subset of an `Array`. For example, extracting a specific zone 
from an array:

.. code-block:: python

    import numpy as np
    from gofast._typing import TypeVar, DType, Array, _Sub
    from gofast.tools.exmath import _define_conductive_zone
    _T = TypeVar('_T', float)
    erp_array: Array[_T, DType[_T]] = np.random.randn(21)
    select_zone, _ = _define_conductive_zone(erp=erp_array, auto=True)
    def check_cz(select_zone: _Sub[Array]): ...

_SP
----
Stands for Station Positions, typically used in electrical resistivity 
profiling. Example:

.. code-block:: python

    import numpy as np
    from gofast._typing import TypeVar, DType, _SP, _Sub
    _T = TypeVar('_T', bound=int)
    surveyL: _SP = np.arange(0, 50 * 121, 50.).astype(np.int32)

Series
------
Represents a `pandas Series`_ object. It is used in place of `pandas.Series` 
for type hinting throughout the package.

DataFrame
---------
Similarly, `DataFrame` represents a `pandas DataFrame`_ object. It is used 
instead of `pandas.DataFrame` for consistency in type hinting.

Examples of Series and DataFrame usage:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from gofast._typing import TypeVar, Any, DType, Series, DataFrame
    _T = TypeVar('_T')
    seriesStr = pd.Series([f'obx{s}' for s in range(21)], name='stringobj')
    seriesFloat = pd.Series(np.arange(7).astype(np.float32), name='floatobj')
    SERs = Series[DType[str]]
    SERf = Series[DType[float]]
    dfStr = pd.DataFrame({'ser1': seriesStr, 'obj2': [f'none' for i in range(21)]})
    dfFloat = pd.DataFrame({'ser1': seriesFloat, 'obj2': np.linspace(3, 28, 7)})
    dfAny = pd.DataFrame({'ser1': seriesStr, 'ser2': seriesFloat})
    DFs = DataFrame[SERs] or DataFrame[DType[str]]
    DFf = DataFrame[SERf] or DataFrame[DType[float]]
    DFa = DataFrame[Series[Any]] or DataFrame[DType[_T]]
"""
from __future__ import annotations 
# from typing import TYPE_CHECKING

# if TYPE_CHECKING:
from typing import (
    List,
    Tuple,
    Sequence,
    Dict,
    Iterable,
    Callable,
    Union,
    Any,
    Generic,
    Optional,
    Type,
    Mapping,
    Text, 
    TypeVar,
    Iterator,
    SupportsInt,
    Set,
    ContextManager,
    Deque, 
    FrozenSet,
    NamedTuple,
    NewType, 
    TypeGuard, 
    TypedDict, 
    Generator
)


__all__ = [
    "List",
    "Tuple", 
    "Sequence", 
    "Dict", 
    "Iterable",
    "Callable",
    "Any",
    "Generic", 
    "Optional", 
    "Union", 
    "Type", 
    "Mapping",
    "Text", 
    "Shape",
    "DType", 
    "NDArray",
    "ArrayLike",
    "_Sub",
    "_SP",
    "_F",
    "_T", 
    "_V",
    "Series",
    "Iterator",
    "SupportsInt",
    "Set", 
    "_ContextManager",
    "_Deque",
    "_FrozenSet",
    "_NamedTuple",
    "_NewType",
    "_TypedDict",
    "TypeGuard",  
    "TypedDict", 
    "Generator"
]

_T = TypeVar('_T')
_V = TypeVar('_V')
K = TypeVar('K')
M = TypeVar('M', bound=int)
N = TypeVar('N', bound=int)
U = TypeVar('U')
D = TypeVar('D', bound='DType')
S = TypeVar('S', bound='Shape')
# ArrayLike = Union[Sequence, np.ndarray, pd.Series, pd.DataFrame]
class AddShape(Generic[S]): 
    """
    Represents an additional dimension for shapes beyond two dimensions.
    Useful for defining shapes with more than two dimensions.
    """

class Shape(Generic[M, N, S]):  
    """
    Generic type for constructing a tuple shape for NDArray. It is 
    specifically designed for two-dimensional arrays with M rows and N columns.

    Example:
        >>> import numpy as np 
        >>> array = np.random.randn(7, 3, 3) 
        >>> def check_valid_type(array: NDArray[Array[float], Shape[M, N, S]]): ...
    """
    
class DType(Generic[D]): 
    """
    Represents data types of elements in arrays. DType can be any type,
    allowing for the specification of array data types.

    Example:
        >>> import numpy as np
        >>> array = np.random.randn(7)
        >>> def check_dtype(array: NDArray[Array[float], DType[float]]): ...
    """

class ArrayLike(Generic[_T, D]): 
    """
    Represents an array-like structure, typically used for one-dimensional arrays.
    For multi-dimensional arrays, NDArray should be used instead.

    Example:
        >>> def check_array(array: ArrayLike[int]): ...
    """

class NDArray(Generic[_T, D], ArrayLike[_T, DType[_T]]):
    """
    Represents a generic N-dimensional array with specified data types.

    Example:
        >>> import numpy as np
        >>> array = np.array([[1, 2], [3, 4]], dtype=int)
        >>> def check_ndarray(array: NDArray[int, DType[int]]): ...
    """

class _F(Generic[_T]):
    """
    Represents a generic callable type, including functions, methods, and classes.
    It allows for the specification of return types and argument types of callables.

    Example:
        >>> def my_function(x: int) -> str: ...
        >>> def check_callable(func: _F[Callable[[int], str]]): ...
    """

class _Sub(Generic[_T]):
    """
    Represents a subset of an array or collection.
    """

class _SP(Generic[_T, D]):
    """
    Represents station position arrays in geophysical surveys.
    Typically holds integer values representing station positions.

    Example:
        >>> import numpy as np
        >>> positions: _SP = np.arange(0, 21 * 10, 10).astype(np.int32)
        >>> def check_station_positions(positions: _SP[ArrayLike[int], DType[int]]): ...
    """

class Series(Generic[_T]):
    """
    Represents a pandas Series object.

    Example:
        >>> import pandas as pd
        >>> series = pd.Series([1, 2, 3])
        >>> def check_series(series: Series[int]): ...
    """

class DataFrame(Generic[_T]):
    """
    Represents a pandas DataFrame object.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': ['a', 'b', 'c']})
        >>> def check_dataframe(df: DataFrame[object]): ...
    """
    
class _ContextManager(ContextManager, Generic[_T]):
    """
    Represents a context manager, a resource that is specifically designed
    to be used with the 'with' statement.

    Example:
        >>> from contextlib import contextmanager
        >>> @contextmanager
        ... def my_context_manager() -> Generator[None, None, None]:
        ...     print("Enter context")
        ...     yield
        ...     print("Exit context")
        >>> def check_context(context: _ContextManager[None]): ...
    """

class _Deque(Deque, Generic[_T]):
    """
    Represents a double-ended queue, allowing for efficient addition
    and removal of elements from either end.

    Example:
        >>> from collections import deque
        >>> my_deque: Deque[int] = deque([1, 2, 3])
        >>> def check_deque(dq: _Deque[int]): ...
    """

class _FrozenSet(FrozenSet, Generic[_T]):
    """
    Represents an immutable set.

    Example:
        >>> my_frozen_set: FrozenSet[int] = frozenset([1, 2, 3])
        >>> def check_frozen_set(fs: _FrozenSet[int]): ...
    """

class _NamedTuple(NamedTuple):
    """
    Represents a named tuple, combining the benefits of tuples and dictionaries.

    Example:
        >>> from typing import NamedTuple
        >>> class MyNamedTuple(NamedTuple):
        ...     name: str
        ...     age: int
        >>> def check_named_tuple(nt: _NamedTuple[str, int]): ...
    """

class _NewType(NewType, Generic[_T]):
    """
    Represents a new type, created from an existing type, used for type checking.

    Example:
        >>> from typing import NewType
        >>> UserId = _NewType('UserId', int)
        >>> def check_user_id(user_id: UserId): ...
    """

class _TypedDict(TypedDict):
    """
    Represents a dictionary with fixed keys and specified value types.
    
    `_TypedDict` creates a dictionary type with specific types for each key.
    It helps to ensure that each field in the dictionary adheres to the
    specified type, providing better type checking at the code analysis stage.
    
    Example:
        >>> from typing import TypedDict
        >>> class MyDict(_TypedDict):
        ...     name: str
        ...     age: int
        >>> def check_typed_dict(d: MyDict): ...
    """
if __name__ == '__main__':
    # Test cases demonstrating the usage of defined generic types.
    ...

























