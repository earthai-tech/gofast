# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
import re 
import copy
import warnings

class RegexMap:
    """
    A class to dynamically map search terms to predefined keys using regular
    expressions.

    Attributes
    ----------
    patterns : dict
        A dictionary where each key is a string representing a category, and the value
        is a compiled regex pattern to match associated terms.

    Methods
    -------
    find_key(search_term)
        Identifies the key for a given search term based on predefined regex patterns.

    Examples
    --------
    >>> from gofast.api.util import RegexMap
    >>> regex_map = RegexMap()
    >>> regex_map.find_key('model')
    'best_estimator_'
    >>> regex_map.find_key('best_parameters')
    'best_params_'
    >>> regex_map.find_key('score')
    'best_scores_'
    >>> regex_map.find_key('results')
    'cv_results_'
    
    Notes
    -----
    The RegexMap is particularly useful in contexts where inputs may vary in phrasing or
    specificity, but can still be categorized into a set of predefined groups. It uses
    regular expressions to provide a flexible and efficient matching mechanism.
    """
    
    def __init__(self):
        """Initialize the regex patterns for mapping search terms to keys."""
        self.patterns = {
            'best_estimator_': re.compile(
                r".*(estimator|model|classifier|regressor).*", re.IGNORECASE),
            'best_params_': re.compile(r".*(param(eter)?s?).*", re.IGNORECASE),
            'best_scores_': re.compile(r".*score.*", re.IGNORECASE),
            'cv_results_': re.compile(r".*(cv_?results?|fold_?results?).*", re.IGNORECASE),
            'scoring': re.compile(r".*(scori?|metric(s)?).*",re.IGNORECASE )
        }

    def find_key(self, search_term):
        """
        Search for a key in the map based on a search term that matches any
        of the regex patterns.

        Parameters
        ----------
        search_term : str
            The term to search against the regex patterns.

        Returns
        -------
        str or None
            The key corresponding to the first matching regex pattern, or 
            None if no match is found.

        Examples
        --------
        >>> regex_map = RegexMap()
        >>> regex_map.find_key('param')
        'best_params_'
        >>> regex_map.find_key('estimator details')
        'best_estimator_'
        >>> regex_map.find_key('unknown_term')
        None

        Notes
        -----
        This method iterates through all precompiled regex patterns, checking
        if the search term matches any pattern. It returns the key associated 
        with the first matching pattern. This method provides a robust way to 
        categorize loosely defined search terms into specific categories
        based on their semantic content.
        """
        for key, pattern in self.patterns.items():
            if pattern.match(search_term):
                return key
        return None
    
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

def rename_instance_class(
    instance, new_name: str, 
    deep=False, clone_instance=True, 
    return_type='instance'
    ):
    """
    Dynamically creates a subclass with a new name for a specific instance, or
    renames the class globally based on the `deep` parameter. This method allows
    the caller to choose whether to return the modified instance or the new class.

    Parameters
    ----------
    instance : object
        The instance whose class is to be renamed.
    new_name : str
        The new name for the class of this specific instance.
    deep : bool, optional
        If True, renames the class globally, affecting all instances. Use with
        caution as this may lead to unexpected behaviors. Default is False.
    clone_instance : bool, optional
        If True, clones all properties and attributes of the original class to
        the new subclass. If False, creates a new subclass without the original
        attributes. Default is True.
    return_type : str, optional
        Specifies the return type of the function: 'instance' to return the
        modified instance or 'class' to return the newly created class.
        Default is 'instance'.

    Returns
    -------
    object or type
        Depending on `return_type`, returns either the modified instance or
        the new class.

    Examples
    --------
    >>> from gofast.api.box import KeyBox
    >>> from gofast.api.extension import rename_instance_class
    >>> obj = KeyBox()
    >>> print(obj.__class__.__name__)
    'KeyBox'
    >>> modified_obj = rename_instance_class(obj, "nPCA", return_type='instance')
    >>> print(modified_obj.__class__.__name__)
    'nPCA'
    >>> new_class = rename_instance_class(obj, "nPCA", return_type='class')
    >>> print(new_class.__name__)
    'nPCA'
    >>> print(obj.__class__.__name__)
    'KeyBox'  # Remains unchanged when return_type is 'class'
    
    Notes
    -----
    Creating a new subclass does not affect other instances of the original class
    unless `deep` is set to True, in which case the original class's name is changed,
    impacting all its instances. The `clone_instance` parameter allows for the
    preservation of class properties and attributes in the new subclass, ensuring
    the functionality of the instance remains unchanged.
    
    """
    if not hasattr(instance, '__class__'):
        raise ValueError("Provided object does not have a '__class__' attribute.")

    if deep:
        warnings.warn(
            "Changing the class name globally affects all instances of the class. "
            "This action may lead to unexpected behaviors, especially with type checks."
        )
        instance.__class__.__name__ = new_name
        return instance if return_type == 'instance' else instance.__class__
    else:
        original_class = instance.__class__
        if clone_instance:
            new_class = type(new_name, (original_class,), dict(original_class.__dict__))
        else:
            new_class = type(new_name, (original_class,), {})
            
        instance.__class__ = new_class
        return instance if return_type == 'instance' else new_class

def clone(instance, new_name=None, copy_type='deep'):
    """
    Creates a copy of an instance, with options for deep or shallow copying,
    and optionally renames the new instance.

    Parameters
    ----------
    instance : object
        The instance to be cloned.
    new_name : str, optional
        If provided, the cloned instance's class will be renamed to this new name.
        This affects only the cloned instance if 'deep' copy is used.
    copy_type : str, optional
        Type of copy to perform: 'deep' for a deepcopy (default) which duplicates
        all data recursively, 'shallow' for a shallow copy which duplicates only
        the top-level container.

    Returns
    -------
    object
        A new instance that is a copy of the original instance, potentially with
        a new class name.

    Examples
    --------
    >>> from gofast.api.extension import clone
    >>> class ExampleClass:
    ...     def __init__(self, data):
    ...         self.data = data
    ...
    >>> obj = ExampleClass([1, 2, 3])
    >>> cloned_obj = clone(obj, new_name="NewClassName")
    >>> print(cloned_obj.__class__.__name__)
    'NewClassName'
    >>> cloned_obj.data
    [1, 2, 3]

    Notes
    -----
    If `new_name` is provided and `copy_type` is 'deep', the new class of the
    cloned instance will not affect other instances of the original class.
    With 'shallow', renaming affects all instances due to the shared class.
    """
    if copy_type == 'deep':
        cloned_instance = copy.deepcopy(instance)
    elif copy_type == 'shallow':
        cloned_instance = copy.copy(instance)
    else:
        raise ValueError("copy_type must be either 'deep' or 'shallow'.")

    if new_name:
        # Dynamically create a new class with the new name for the cloned instance
        original_class = cloned_instance.__class__
        new_class = type(new_name, (original_class,), dict(original_class.__dict__))
        cloned_instance.__class__ = new_class

    return cloned_instance

def make_introspection(target_obj, source_obj):
    """
    Copies all attributes from a source object to a target object, effectively
    making the target inherit attributes from the source.

    This function performs a simple form of introspection by iterating through
    all attributes of the source object and setting these attributes on the
    target object. It is useful for dynamically updating an instance with
    properties from another, especially in contexts involving prototype-based
    inheritance or dynamic composition.

    Parameters
    ----------
    target_obj : object
        The object to which attributes from the source object will be copied.
        This object will be modified in-place by adding new attributes or
        overwriting existing ones based on the source object.

    source_obj : object
        The source object from which attributes will be copied. All attributes
        of this object, including those in its `__dict__`, are considered.

    Returns
    -------
    None

    Examples
    --------
    >>> from gofast.api.extension import make_introspection
    >>> class A:
    ...     def __init__(self):
    ...         self.x = 10
    ...         self.y = 20
    ...
    >>> class B:
    ...     def __init__(self):
    ...         self.a = 5
    ...
    >>> a = A()
    >>> b = B()
    >>> make_introspection(b, a)
    >>> print(b.x, b.y)  # B now has attributes x and y copied from A
    10 20

    Notes
    -----
    This function does not return any value; it modifies the `target_obj` in-place.
    Be cautious when using this function, as it can overwrite existing attributes
    of the `target_obj` without any warnings.
    """
    # Iterate over all attributes of the source object and set 
    # them on the target object
    for key, value in source_obj.__dict__.items():
        setattr(target_obj, key, value)  
        
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