# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: LKouadio <etanoyau@gmail.com>

import warnings
import importlib
import pandas as pd
import inspect
import pkgutil
from functools import wraps 
from types import FunctionType
from typing import List
# -------------------------------------------------------------------------
# Note to Developers:
# This module provides utilities for dynamically attaching and removing 
# specialized methods (prefixed with 'go_') and explicitly listed methods 
# from pandas.DataFrame. It also contains helpers to wrap and unwrap 
# functions for displaying their signatures in an interactive environment.
# -------------------------------------------------------------------------

# Define the import paths for explicitly listed methods
_EXPLICIT_GO_METHODS_IMPORT_PATH = {
    'summary': 'gofast._summary.summary',
    'go_corr': 'gofast.stats.descriptive.corr',
    'go_describe': 'gofast.stats.descriptive.describe',
    'go_get_range': 'gofast.stats.descriptive.get_range',
    'go_gini_coeffs': 'gofast.stats.descriptive.gini_coeffs',
    'go_hmean': 'gofast.stats.descriptive.hmean',
    'go_iqr': 'gofast.stats.descriptive.iqr',
    'go_kurtosis': 'gofast.stats.descriptive.kurtosis',
    'go_mean': 'gofast.stats.descriptive.mean',
    'go_median': 'gofast.stats.descriptive.median',
    'go_mode': 'gofast.stats.descriptive.mode',
    'go_quantile': 'gofast.stats.descriptive.quantile',
    'go_quartiles': 'gofast.stats.descriptive.quartiles',
    'go_skew': 'gofast.stats.descriptive.skew',
    'go_std': 'gofast.stats.descriptive.std',
    'go_var': 'gofast.stats.descriptive.var',
    'go_wmedian': 'gofast.stats.descriptive.wmedian',
    'go_z_scores': 'gofast.stats.descriptive.z_scores',
    'cumulative_ops':'gofast.stats.utils.cumulative_ops', 
    "anova_test": 'gofast.stats.inferential.anova_test',
    "bootstrap": 'gofast.stats.inferential.bootstrap',
    "chi2_test": 'gofast.stats.inferential.chi2_test',
    "mixed_effects_model": 'gofast.stats.inferential.mixed_effects_model',
    "check_anova_assumptions": 'gofast.stats.inferential.check_anova_assumptions',
    "go_transform":'gofast.dataops.preprocessing.transform'

}

# Explicit list of methods to attach to pandas.DataFrame that 
# do not start with 'go_'

_EXPLICIT_GO_METHODS = [
    'summary',
    # non-go_ methods as needed
]
_EXPLICIT_GO_METHODS= list(_EXPLICIT_GO_METHODS_IMPORT_PATH.keys())


_PUBLIC_MODULES =[
    'analysis',
    'api',
    'backends',
    'callbacks',
    'cli',
    'compat',
    'dataops',
    'datasets',
    'estimators',
    'experimental',
    'externals',
    'geo',
    'gflogs',
    'mlops',
    'models',
    'nn',
    'plot',
    'pyx',
    'stats',
    'transformers',
    'utils',
    'adaline',
    'assistance',
    'benchmark',
    'boosting',
    'cluster_based',
    'config',
    'decorators',
    'decomposition',
    'dimensionality',
    'ensemble',
    'exceptions',
    'factors',
    'feature_selection',
    'metrics',
    'metrics_special',
    'model_selection',
    'perceptron',
    'preprocessing',
    'query',
    'tree'
 ]
# Store original functions to allow unwrapping
_original_functions = {}


class FunctionWrapper:
    """
    A wrapper for functions to modify their __repr__ method 
    to display signatures. This is particularly useful in 
    interactive environments where users can view function 
    signatures by calling the function object without parentheses.
    
    Parameters
    ----------
    func : FunctionType
        The function to be wrapped. The wrapper intercepts the 
        __repr__ call to display the function's signature.
    """

    def __init__(self, func: FunctionType):
        self.func = func
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__
        self.__module__ = func.__module__
        self.__annotations__ = func.__annotations__
        self.__wrapped__ = func  # Preserve the original function

    def __call__(self, *args, **kwargs):
        """
        Proxy the call to the underlying wrapped function.
        """
        return self.func(*args, **kwargs)

    def __repr__(self):
        """
        Return a string that includes the function's module, 
        name, and signature for easier inspection in an 
        interactive environment.
        """
        sig = inspect.signature(self.func)
        return (
            f"<function {self.func.__module__}."
            f"{self.func.__name__}{sig}>"
        )


def attach_go_methods(error='warn'):
    """
    Attach explicitly listed methods to pandas.DataFrame.
    
    This function iterates through the ``_EXPLICIT_GO_METHODS`` list,
    imports each function, and attaches it to ``pandas.DataFrame`` 
    as a method. Additionally, it forces a load of certain modules 
    (e.g., gofast.stats.descriptive) so that any decorator-injected 
    methods (prefixed with `go_`) are also recognized at runtime.

    If a discovered function's *first parameter* is annotated 
    (or clearly named) as a DataFrame, a small wrapper is created 
    so that users can call:

        df.some_method(...)

    without needing to pass the DataFrame explicitly.
    """
    # Force load of gofast.stats.descriptive to register possible
    # go_ methods automatically injected by decorators.
    import gofast.stats.descriptive  # noqa
    
    # Attach all explicitly listed methods that don't follow 'go_' naming
    for method_name in _EXPLICIT_GO_METHODS:
        # Skip if DataFrame already has this method
        if hasattr(pd.DataFrame, method_name):
            continue
        
        import_path = _EXPLICIT_GO_METHODS_IMPORT_PATH.get(method_name)
        if not import_path:
            if error == 'warn':
                warnings.warn(
                    "No import path provided for explicit method "
                    f"'{method_name}'. Skipping."
                )
            continue

        try:
            module_path, func_name = import_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            func = getattr(module, func_name)
            
            # Only proceed if we have a callable function
            if not callable(func):
                continue

            # 1. Inspect the function's signature
            sig = inspect.signature(func)
            params = list(sig.parameters.values())

            # 2. Check if the first parameter is (or likely is) a DataFrame
            #    We'll check either annotation or a name heuristic.
            if params:
                first_param = params[0]
                # Example checks:
                #  - If user used type hints: first_param.annotation == pd.DataFrame
                #  - Or if the param is named "df" or "dataframe"
                # You can adjust as needed, e.g.:
                if (
                    first_param.annotation == pd.DataFrame
                    or first_param.name.lower() in ['df', 'data', 'dataframe']
                ):
                    # 3. Create a simple wrapper that calls `func(self, *args, **kwargs)`
                    #    effectively removing the need for the user to pass a DF.
                    @wraps(func)
                    def wrapper(self, *args, **kwargs):
                        # The DataFrame instance is `self`, so pass it as
                        # the first argument to `func`.
                        return func(self, *args, **kwargs)
                    # Preserve metadata for introspection for consistency
                    wrapper.__name__ = func.__name__
                    wrapper.__doc__ = func.__doc__
                    wrapper.__annotations__ = func.__annotations__

                    # Attach the wrapper
                    setattr(pd.DataFrame, method_name, wrapper)
                    continue
            
            # If we reach here, no wrapper is needed -> attach function directly
            setattr(pd.DataFrame, method_name, func)

        except (ImportError, AttributeError) as e:
            if error == 'warn':
                warnings.warn(f"Could not attach method '{method_name}': {e}")


def remove_go_methods():
    """
    Remove any method starting with 'go_' from pandas.DataFrame 
    and remove explicitly listed methods.
    
    This function dynamically detects and removes all methods 
    on ``pandas.DataFrame`` that start with 'go_'. It also 
    iterates through the ``EXPLICIT_GO_METHODS`` list and 
    removes each method from ``pandas.DataFrame`` if it exists.
    """
    # Collect and remove all 'go_' methods from pandas.DataFrame
    go_methods = [
        attr for attr in dir(pd.DataFrame)
        if attr.startswith('go_')
        and callable(getattr(pd.DataFrame, attr))
    ]
    for method_name in go_methods:
        delattr(pd.DataFrame, method_name)

    # Remove explicitly listed methods from pandas.DataFrame
    for method_name in _EXPLICIT_GO_METHODS:
        if hasattr(pd.DataFrame, method_name):
            delattr(pd.DataFrame, method_name)


def _wrap_function(func: FunctionType) -> FunctionType:
    """
    Wrap a function to modify its __repr__ method to display its signature.
    
    Parameters
    ----------
    func : FunctionType
        The function to wrap. Wrapping injects a specialized 
        __repr__ that shows the function's signature.
    
    Returns
    -------
    FunctionType
        The wrapped function with a customized __repr__.
    """
    if func in _original_functions:
        # If we've already wrapped this function, return 
        # the previously wrapped version to avoid duplication.
        return _original_functions[func]
    
    wrapped = FunctionWrapper(func)
    _original_functions[func] = wrapped
    return wrapped


def _unwrap_function(func: FunctionType) -> FunctionType:
    """
    Unwrap a previously wrapped function to its original state.
    
    Parameters
    ----------
    func : FunctionType
        The wrapped function to unwrap.
    
    Returns
    -------
    FunctionType
        The original, unwrapped function.
    """
    return _original_functions.get(func, func)


def _wrap_public_functions(module):
    """
    Wrap all public functions in a module to display their signatures.
    
    This function iterates through all attributes of the given module,
    identifies functions, and wraps them using `FunctionWrapper` to 
    enhance their `__repr__`.
    
    Parameters
    ----------
    module : module
        The module containing functions to wrap.
    """
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, FunctionType):
            wrapped = _wrap_function(attr)
            setattr(module, attr_name, wrapped)


def _unwrap_public_functions(module):
    """
    Unwrap all previously wrapped public functions in a module.
    
    This function iterates through all attributes of the given module,
    identifies wrapped functions, and restores them to their original
    state.
    
    Parameters
    ----------
    module : module
        The module containing functions to unwrap.
    """
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, FunctionWrapper):
            original = _unwrap_function(attr.func)
            setattr(module, attr_name, original)

def _discover_submodules(package: str, error ='warn') -> List[str]:
    """
    Discover all submodules within a given package, excluding those that 
    start with an underscore.
    
    This function uses `pkgutil.walk_packages` to dynamically find all 
    submodules of the specified package. Submodules whose names begin with 
    an underscore (`_`) are automatically excluded from the discovery 
    process to prevent internal or private modules from being exposed.
    
    Parameters
    ----------
    package : str
        The name of the package to search for submodules.
    
    Returns
    -------
    List[str]
        A list of full import paths for each discovered submodule that 
        does not start with an underscore.
    """
    submodules = []
    
    try:
        # Import the specified package module
        package_module = importlib.import_module(package)
    except ImportError as e:
        if error =='warn':
            warnings.warn(
                f"Could not import package '{package}': {e}. "
                "No submodules will be discovered.", 
                ImportWarning
            )
        return submodules
    
    # Retrieve the package's path to search for submodules
    if hasattr(package_module, '__path__'):
        package_path = package_module.__path__
    else:
        if error =='warn':
            warnings.warn(
                f"The package '{package}' does not have a '__path__' attribute. "
                "No submodules will be discovered.", 
                ImportWarning
            )
        return submodules
    
    # Iterate through all modules in the package using pkgutil.walk_packages
    for finder, name, ispkg in pkgutil.walk_packages(
        path=package_path, 
        prefix=package + '.'
    ):
       # Extract the base name of the module
        module_basename = name.split('.')[-1]
        module_path = name.split('.')
        
        # Exclude submodules that:
        # 1. Start with an underscore
        # 2. Are part of any 'tests' subpackage
        # 3. Are named 'setup'
        if (
            not ispkg and
            not module_basename.startswith('_') and
            'tests' not in module_path and
            'tools' not in module_path and 
            'config' not in module_path and 
            module_basename  != 'setup'
            
        ):
            submodules.append(name)
    
    return submodules

def wrap_public_functions (error ='warn'):
    """
    Wrap all public functions in a module to display their signatures.
    
    This function iterates through all attributes of the given module,
    identifies functions, and wraps them using `FunctionWrapper` to 
    enhance their `__repr__`.
    
    Parameters
    ----------
    error : {'warn', 'ignore'}
        raise a warnings if submodule not found or  simply ignore it.
    """
    # Discover all submodules within 'gofast'
    submodules = _discover_submodules('gofast', error=error)
    
    # Dynamically wrap public functions in each submodule
    for submodule_path in submodules:
        try:
            module = importlib.import_module(submodule_path)
            _wrap_public_functions(module )
        except ImportError as e:
            if error =='warn': 
                warnings.warn(f"Could not import submodule '{submodule_path}': {e}")
            continue

def unwrap_public_functions(error ='warn'):   
    """
    Unwrap all previously wrapped public functions in a module.
    
    This function iterates through all attributes of the given module,
    identifies wrapped functions, and restores them to their original
    state.
    Parameters
    ----------
    error : {'warn', 'ignore'}
        raise a warnings if submodule not found or  simply ignore it.
        
    """     
    # Discover all submodules within 'gofast'
    submodules = _discover_submodules('gofast', error=error )
    
    # Dynamically unwrap public functions in each submodule
    for submodule_path in submodules:
        try:
            module = importlib.import_module(submodule_path)
            _unwrap_public_functions(module)
        except ImportError as e:
            if error =='warn':
                warnings.warn(f"Could not import submodule '{submodule_path}': {e}")
            continue
