# -*- coding: utf-8 -*-


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
    >>> class MyClass:
    ...     pass
    ...
    >>> obj = MyClass()
    >>> is_instance_extended(obj, MyClass)
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