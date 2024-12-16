# -*- coding: utf-8 -*-
"""
Machine Learning Operations (MLOps)
"""

import warnings
# Issuing a warning message when importing the `gofast.mlops` package or any of its modules
# Lazy loader function for triggering the warning
def __getattr__(name):
    # Check if the requested submodule exists
    try:
        # Attempt to load the submodule (e.g., automation, inference)
        module = __import__(f"gofast.mlops.{name}", globals(), locals(), [name])
        # Trigger the warning on first access of any submodule
        warning_message = (
            "Warning: You are accessing the MLOps subpackage of gofast. "
            "Please ensure that you have set up a dedicated environment for this subpackage, "
            "as it requires specific external dependencies. "
            "It is recommended to create a virtual environment with the necessary packages for MLOps tasks."
        )
        warnings.warn(warning_message, UserWarning, 2)
        return module
    except ImportError:
        # Raise an AttributeError if the module does not exist
        raise AttributeError(f"Module '{name}' not found in 'gofast.mlops'.")
