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

# Disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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
        from . import _public as go 

        globals().update({
            'assist_me': assist_me,
            'explore': explore,
            'read_data': read_data, 
            'export_data':export_data, 
            'fetch_data':fetch_data,
            'go':go
        })

        __all__.extend(['assist_me', 'explore', 'read_data',
                        'export_data', 'fetch_data', 'go'])

        warnings.warn("Public API has been enabled.", UserWarning)

    def disable_public_api(self):
        """Hide public API components."""
        for name in ['assist_me', 'explore', 'read_data', 'export_data',
                     'fetch_data', 'go']:
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

def __getattr__(name):
    """
    Dynamically provide access to public API components based on the 'public' flag.

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
    """
    public_attributes = ['assist_me', 'explore', 'read_data',
                         'export_data', 'fetch_data', 'go']
    if name in public_attributes:
        if config.public:
            if name in public_attributes:
                from .assistance import assist_me, explore
                from .core.io import read_data, export_data
                from .datasets import fetch_data 
                from . import _public  as go 
                
                mapping = {
                    'assist_me': assist_me,
                    'explore': explore,
                    'read_data': read_data, 
                    'export_data': export_data, 
                    'fetch_data':fetch_data,
                    'go':go
                }
                return mapping[name]
        else:
            # Public API is disabled; raise a descriptive AttributeError
            raise AttributeError(
                f"The attribute '{name}' is not available because the public"
                f" API is disabled. Please enable it by setting "
                "``gofast.config.public = True`` "
                f"before accessing public API methods."
            )
    # If we reach here, either 'public' is False, or the 
    # attribute is not in public attributes.
    hint = (
        "The requested attribute '{attr}' is not available. "
        "If you intend to access the public API, please ensure that:\n\n"
        "    >>> import gofast as gf\n"
        "    >>> gf.config.public = True\n\n"
        "Once the public API is enabled, you can access 'assist_me',"
        " 'explore', 'read_data', 'export_data' ..."
    )
    warnings.warn(hint.format(attr=name))
    # For any other attributes, follow the default behavior
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    
# Update the module to use the new property
__builtins__['PUBLIC'] = config.public

__doc__ += f"\nVersion: {__version__}\n"