# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>

"""
GOFast: Accelerate Your Machine Learning Workflow
=================================================

:code:`gofast` is designed to streamline and accelerate every step of your 
data science workflow, enhancing productivity, ease of use, and community-driven
improvements.
"""
import os
import sys 
import logging
import warnings
import importlib
from contextlib import contextmanager

# Configure basic logging and suppress certain third-party library warnings
logging.basicConfig(level=logging.WARNING)
logging.getLogger('matplotlib.font_manager').disabled = True

# Dynamic import function
def _lazy_import(module_name, alias=None):
    """Lazily import a module to reduce initial package load time."""
    def _lazy_loader():
        return importlib.import_module(module_name)
    if alias:
        globals()[alias] = _lazy_loader
    else:
        globals()[module_name] = _lazy_loader

# Define the version
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.1.0"

# Dependency check
_required_dependencies = [
    ("numpy", None),
    ("pandas", None),
    ("scipy", None),
    ("matplotlib", None),
    ("seaborn", None),
    ("tqdm", None), 
    ("sklearn", "scikit-learn"),
    ("statsmodels", None)
]

_missing_dependencies = []
for package, import_name in _required_dependencies:
    try:
        if import_name:
            _lazy_import(import_name, package)
        else:
            _lazy_import(package)
    except ImportError as e:
        _missing_dependencies.append(f"{package}: {str(e)}")

if _missing_dependencies:
    warnings.warn("Some dependencies are missing. GOFast may not function correctly:\n" +
                  "\n".join(_missing_dependencies), ImportWarning)

# Suppress FutureWarnings or SyntaxWarning if desired, but allow users
# to re-enable them
# Define the warning categories and their corresponding actions
_WARNING_CATEGORIES = {
    "FutureWarning": FutureWarning,
    "SyntaxWarning": SyntaxWarning
}

# Default actions for each warning category
_WARNINGS_STATE = {
   # "FutureWarning": "ignore",
    "SyntaxWarning": "ignore"
}

def suppress_warnings(suppress: bool = True):
    """
    Suppress or re-enable FutureWarnings and SyntaxWarnings.

    Function allows users to suppress specific warnings globally within
    the package. By default, it suppresses both `FutureWarning` and 
    `SyntaxWarning`. Users can re-enable these warnings by setting 
    `suppress=False`.

    Parameters
    ----------
    suppress : bool, default=True
        - If `True`, suppresses `FutureWarning` and `SyntaxWarning` by setting 
          their filter to the action specified in `_WARNINGS_STATE`.
        - If `False`, re-enables the warnings by resetting their filter to 
          the default behavior.
    """
    for warning_name, action in _WARNINGS_STATE.items():
        category = _WARNING_CATEGORIES.get(warning_name, Warning)
        if suppress:
            # Suppress the warning by applying the specified action
            warnings.filterwarnings(action, category=category)
        else:
            # Re-enable the warning by resetting to default behavior
            warnings.filterwarnings("default", category=category)

# Suppress warnings by default when the package is initialized
suppress_warnings()

# filter out TF INFO and WARNING messages
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # or "3"
# Disable oneDNN custom operations
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Setup logging configuration
from ._util import initialize_logging 
initialize_logging()

__all__ = ["setup_logging", "__version__", "config"]

class GoFastConfig:
    def __init__(self):
        self._public = False

    @property
    def public(self):
        """
        Flag to toggle the public API.
        
        When set to True, the public API functions (assist_me, explore, read_data)
        become accessible as attributes of the 'gofast' module. When set to False,
        they are removed, and attempting to access them raises an error.
        """
        return self._public

    @public.setter
    def public(self, value):
        """
        Set the public flag to enable or disable the public API.
        
        This property expects a boolean value:
        - True: Enable the public API.
        - False: Disable the public API.

        If a non-boolean value is provided, the setter will attempt to coerce it 
        into a boolean. The following rules apply:
        - If 'value' is truthy (e.g., non-zero number, non-empty object), 
          it will be set to True.
        - Otherwise, it defaults to False.

        A warning is raised if a non-boolean value is coerced.
        """
        if not isinstance(value, bool):
            # Attempt to coerce the value into a boolean
            original_value = value
            try:
                value = bool(value)
                warnings.warn(
                    f"Non-boolean value '{original_value}' provided for 'public'. "
                    f"Coerced to '{value}'.", UserWarning
                )
            except Exception:
                # If coercion fails, default to False
                value = False
                warnings.warn(
                    f"Non-boolean value '{original_value}' could not be coerced. "
                    f"Setting 'public' to False.", UserWarning
                )

        if self._public == value:
            # No change needed
            return
        
        self._public = value
        if self._public:
            self.enable_public_api()
        else:
            self.disable_public_api()

    def enable_public_api(self):
        """Expose public API components."""
        from .assistance import assist_me, explore
        from .core.io import read_data, export_data
        from .datasets import fetch_data 
        from ._public_api import attach_go_methods
        from ._public_api import wrap_public_functions

        attach_go_methods()
        wrap_public_functions(error ='ignore') 
        
        globals().update({
            'assist_me': assist_me,
            'explore': explore,
            'read_data': read_data, 
            'export_data':export_data, 
            'fetch_data':fetch_data,
        })

        __all__.extend(['assist_me', 'explore', 'read_data',
                        'export_data', 'fetch_data', ])

        warnings.warn("Public API has been enabled.", UserWarning)

    def disable_public_api(self):
        """Hide public API components."""
        # Remove 'go_' methods from DataFrame
        from ._public_api import remove_go_methods  
        from ._public_api import unwrap_public_functions 
        remove_go_methods()
        unwrap_public_functions(error ='ignore') 
        
        for name in ['assist_me', 'explore', 'read_data', 'export_data',
                     'fetch_data',]:
            globals().pop(name, None)
            if name in __all__:
                __all__.remove(name)

        warnings.warn("Public API has been disabled.", UserWarning)

    @contextmanager
    def expose_public_api(self):
        """
        Temporarily enable the public API within a context.
        
        Usage:
            with config.expose_public_api():
                # public API is enabled here
                assist_me()
            # public API is disabled here again
        """
        original_state = self.public
        self.public = True
        try:
            yield
        finally:
            self.public = original_state

# Instantiate the configuration object
config = GoFastConfig()

class PublicAPIError(AttributeError):
    """Custom exception for public API access errors."""
    pass

def __getattr__(name):
    """
    Dynamically provide access to public API components based on the 'public' flag.

    Function intercepts attribute access to the `gofast` module. Depending 
    on the attribute being accessed and the state of the `public` flag, it either 
    provides access to public API functions, exposes internal modules, or raises 
    appropriate errors and warnings.
    
    If the public API is enabled (config.public = True), attributes
    'assist_me', 'explore', and 'read_data' become available. If the public API
    is disabled, attempting to access these attributes will result in an
    AttributeError with a helpful message.

    Example of enabling public API:
        import gofast as gf
        gf.config.public = True
        gf.assist_me()  # works now

    Disabling public API:
        gf.config.public = False
        gf.assist_me()  # raises warning
        
    Parameters
    ----------
    name : str
        The name of the attribute being accessed.

    Returns
    -------
    Any
        The requested attribute if available.

    Raises
    ------
    PublicAPIError
        If attempting to access a public API function when the public
        API is disabled.
    AttributeError
        If the attribute does not exist within the public API or internal
        modules.
    """
    from ._public_api import _PUBLIC_MODULES 
    
    # Define the list of attributes that are part of the public API
    public_attributes = [
        'assist_me', 'explore', 'read_data',
        'export_data', 'fetch_data',
        # other public functions as needed
    ]

    # Handle access to public API attributes
    if name in public_attributes:
        if config.public:
            # Public API is enabled; retrieve and return the attribute
            from .assistance import assist_me, explore
            from .core.io import read_data, export_data
            from .datasets import fetch_data 
            
            mapping = {
                'assist_me': assist_me,
                'explore': explore,
                'read_data': read_data, 
                'export_data': export_data, 
                'fetch_data': fetch_data,
            }
            # Cache the function in globals() for future accesses
            globals()[name] = mapping[name]
            return mapping[name]
        else:
            # Public API is disabled; raise a descriptive PublicAPIError
            raise PublicAPIError(
                f"The attribute '{name}' is not available because the public "
                f"API is disabled. Please enable it by setting "
                f"``gofast.config.public = True`` before accessing public API methods."
            )

    # Handle access to internal public modules
    if name in _PUBLIC_MODULES:
        # Check if the module is already in globals()
        module = globals().get(name)
        if module is not None:
            return module
        try:
            # Dynamically import the internal module
            module = importlib.import_module(f".{name}", __name__)
        except ImportError:
            # If dynamic import fails, attempt to retrieve from sys.modules
            module = sys.modules.get(f"{__name__}.{name}")
       
        if module is not None:
            # Cache the module in the global namespace for future accesses
            globals()[name] = module
            return module
        
        # If the module is still not found, issue a warning 
        # and Attribute error will raise. 
    
    # If the attribute is neither a public API function nor an internal module,
    # issue a warning and raise an AttributeError
    hint = (
        "The requested attribute '{attr}' is not available. "
        "If you intend to access the public API, please ensure that:\n\n"
        "    >>> import gofast as gf\n"
        "    >>> gf.config.public = True\n\n"
        "Once the public API is enabled, you can access functions like "
        "'assist_me', 'explore', 'read_data', 'export_data', 'fetch_data', etc."
    )
    # Issue a warning to guide the user
    warnings.warn(hint.format(attr=name))
    # Raise an AttributeError indicating the attribute does not exist
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Append the version information to the module's docstring
__doc__ += f"\nVersion: {__version__}\n"



