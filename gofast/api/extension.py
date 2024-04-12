# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

class MetaLen(type):
    """
    A metaclass that allows the `len()` function to be used on classes 
    themselves, not just their instances. When `len()` is called on a class 
    that uses MetaLen as its metaclass, it returns the length of the class's 
    name.

    This metaclass can be particularly useful when you want to provide additional
    class-level behaviors or introspection capabilities that are not typically 
    available
    or expected in Python classes.

    Example
    -------
    >>> class MyClass(metaclass=MetaLen):
    ...     pass
    ...
    >>> len(MyClass)
    7

    The above example demonstrates that when `len()` is called on `MyClass`, which uses
    `MetaLen` as its metaclass, it returns the length of `"MyClass"`, which is 7.

    Methods
    -------
    __len__(cls):
        Overrides the default `__len__` method to return the length of the class name.
        This method is called when `len()` is invoked on the class itself.

    Parameters
    ----------
    cls : class
        The class on which `len()` is called. The `cls` parameter is automatically
        provided by Python and represents the class itself, not an instance of the class.

    Returns
    -------
    int
        The length of the class's name.

    See Also
    --------
    type : The default metaclass in Python that serves as the base for creating
    new classes.
    """
    def __len__(cls):
        # Return the length of the class name when len() is called on the class
        return len(cls.__name__)
    
def isinstance_(instance, cls):
    """
    Performs an enhanced isinstance check that can gracefully handle a tuple 
    of classes and module reloading issues, facilitating a more robust type 
    checking, especially in environments where classes might be reloaded or 
    imported differently, potentially leading to false negatives with the 
    standard isinstance function.

    Parameters
    ----------
    instance : object
        The object to check.
    cls : type or tuple of types
        The target class, classes, or a tuple of classes to check against. 
        If `cls` is not a tuple, it will be converted to one for uniform handling.

    Returns
    -------
    bool
        True if `instance` is an instance of any class in `cls`, considering class 
        name and module path matches. False otherwise.

    Examples
    --------
    >>> from gofast.api.extension import isinstance_
    >>> class MyClass:
    ...     pass
    ...
    >>> obj = MyClass()
    >>> isinstance_(obj, MyClass)
    True

    # Demonstrating with module reloading issue
    >>> import importlib
    >>> importlib.reload(MyClass)
    <module 'MyClass' from '...'>
    >>> isinstance_(obj, MyClass)
    False  # This might vary based on how MyClass is defined and reloaded

    # Using a tuple of classes
    >>> class AnotherClass:
    ...     pass
    ...
    >>> isinstance_(obj, (MyClass, AnotherClass))
    True

    Note
    ----
    This function is particularly useful in dynamic environments where classes may 
    be reloaded or when dealing with complex import hierarchies that could lead to 
    situations where the standard `isinstance` check might erroneously return False 
    due to objects being instances of classes that have been reloaded or imported 
    under different namespaces.
    """
    if not isinstance(cls, tuple):
        cls = (cls,)  # Make cls a tuple if it isn't already, for uniform handling

    direct_check = any(isinstance(instance, single_cls) for single_cls in cls)
    if direct_check:
        return True

    for single_cls in cls:
        if instance.__class__.__name__ == single_cls.__name__:
            instance_module = instance.__class__.__module__.split('.')[-1]
            cls_module = single_cls.__module__.split('.')[-1]
            if instance_module == cls_module:
                return True
    return False

def resolve_estimator_name(estimator):
    """
    Retrieves the name of an estimator, whether it's a class, an instantiated object,
    or a string that represents the name of the estimator. This function is designed
    to handle complex scenarios where estimators might be wrapped or imported in a
    non-standard manner.

    Parameters
    ----------
    estimator : callable, instance, or str
        The estimator whose name is to be retrieved. This can be a callable class,
        an instance of a class, a string representing the name, or even a more
        complex wrapped or dynamically created estimator.

    Returns
    -------
    str
        The name of the estimator. Returns 'Unknown estimator' if the name cannot
        be determined.

    Examples
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from gofast.api.extension import resolve_estimator_name 
    >>> resolve_estimator_name(RandomForestClassifier)
    'RandomForestClassifier'
    >>> resolve_estimator_name(RandomForestClassifier())
    'RandomForestClassifier'
    >>> resolve_estimator_name(Pipeline)
    'Pipeline'
    >>> resolve_estimator_name("RandomForest")
    'RandomForest'
    """
    if isinstance(estimator, str):
        return estimator
    elif hasattr(estimator, '__name__'):
        return estimator.__name__
    elif hasattr(estimator, '__class__'):
        # Check for the most base class available in standard types, to handle wrappers
        base_class = get_base_estimator(estimator)
        return base_class.__name__ if base_class else estimator.__class__.__name__
    else:
        return 'Unknown estimator'

def get_base_estimator(estimator):
    """
    Recursively find the base estimator if the estimator is wrapped.

    This helper function digs through layers of wrapping to find the underlying
    estimator's class. For example, in the case of scikit-learn's Pipeline or
    similar wrappers.

    Parameters
    ----------
    estimator : object
        The estimator or a wrapped estimator object.

    Returns
    -------
    class
        The most base class of the estimator if unwrapped successfully, or None if
        no deeper base class could be identified.
    """
    # This is a simple heuristic and might need to be adjusted based on actual wrapping mechanics used.
    if hasattr(estimator, 'estimator') and hasattr(estimator.estimator, '__class__'):
        return get_base_estimator(estimator.estimator)
    elif hasattr(estimator, '__class__'):
        return estimator.__class__
    return None

def fetch_estimator_name(estimator):
    """
    Retrieves the name of an estimator, whether it's a class, an instantiated object, 
    or a string that represents the name of the estimator.

    Parameters
    ----------
    estimator : callable, instance, or str
        The estimator whose name is to be retrieved. This can be a callable class, 
        an instance of a class, or a string representing the name.

    Returns
    -------
    str
        The name of the estimator. Returns 'Unknown estimator' if the name cannot be 
        determined.
    
    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from gofast.api.extension import fetch_estimator_name 
    >>> find_estimator_name(RandomForestClassifier)
    'RandomForestClassifier'
    >>> find_estimator_name(RandomForestClassifier())
    'RandomForestClassifier'
    >>> find_estimator_name("RandomForest")
    'RandomForest'
    """
    if isinstance(estimator, str):
        return estimator
    elif hasattr(estimator, '__name__'):
        return estimator.__name__
    elif hasattr(estimator, '__class__'):
        return estimator.__class__.__name__
    else:
        return 'Unknown estimator'