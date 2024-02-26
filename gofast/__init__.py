# -*- coding: utf-8 -*-
# License: BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>

"""
GOFast: Accelerate Your Machine Learning Workflow
"""

import os
import sys
import logging
import warnings

# Only modify sys.path if necessary, avoid inserting unnecessary paths
package_dir = os.path.dirname(__file__)
if package_dir not in sys.path:
    sys.path.insert(0, package_dir)

# Configure logging with lazy loading
logging.basicConfig(level=logging.WARNING)
logging.getLogger('matplotlib.font_manager').disabled = True

# Environment setup for compatibility
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

# Suppress FutureWarnings globally, consider doing it locally if possible
warnings.simplefilter(action='ignore', category=FutureWarning)

if not __package__:
    __package__ = 'gofast'


# Dynamic import to reduce initial load time
def lazy_import(module_name, global_name=None):
    if global_name is None:
        global_name = module_name
    import importlib
    globals()[global_name] = importlib.import_module(module_name)

# Generate version
try:
    from ._version import version
    __version__ = version.split('.dev')[0]
except ImportError:
    __version__ = "0.1.0"

# Check and import main dependencies lazily
_main_dependencies = {
    "numpy": None,
    "scipy": None,
    "scikit-learn": "sklearn",
    "matplotlib": None,
    "pandas": None,
    "seaborn": None,
    "tqdm":None,
}

_missing_dependencies = []

for module, import_name in _main_dependencies.items():
    try:
        lazy_import(module if not import_name else import_name, module)
    except ImportError as e:
        _missing_dependencies.append(f"{module}: {e}")

if _missing_dependencies:
    raise ImportError("Unable to import required dependencies:\n" + "\n".join(
        _missing_dependencies))


# Set a default LOG_PATH if it's not already set
os.environ.setdefault('LOG_PATH', os.path.join(package_dir, 'gflogs'))

# Import the logging setup function from _gofastlog.py
from ._gofastlog import gofastlog

# Define the path to the _gflog.yml file
config_file_path = os.path.join(package_dir, '_gflog.yml')

# Set up logging with the path to the configuration file
gofastlog.load_configuration(config_file_path)


# Public API
# __all__ = ['show_versions']

# Seed control function, consider moving it to a utilities module
def setup_module(module):
    """Fixture for the tests to assure globally controllable seeding of RNGs"""
    import numpy as np
    import random

    _random_seed = os.environ.get("GOFAST_SEED", np.random.randint(0, 2**32 - 1))
    print(f"I: Seeding RNGs with {_random_seed}")
    np.random.seed(int(_random_seed))
    random.seed(int(_random_seed))

# Reset warnings to default
warnings.simplefilter(action='default', category=FutureWarning)

__doc__= """\
Accelerate Your Machine Learning Workflow
==========================================

:code:`gofast` is a comprehensive machine learning toolbox designed to 
streamline and accelerate every step of your data science workflow. 
Its objectives are: 
    
* `Enhance Productivity`: Reduce the time spent on routine data tasks.
* `User-Friendly`: Whether you're a beginner or an expert, gofast is designed 
  to be intuitive and accessible for all users in the machine learning community.
* `Community-Driven`: welcoming contributions and suggestions from the community
  to continuously improve and evolve.

`GoFast`_ focused on delivering high-speed tools and utilities that 
assist users in swiftly navigating through the critical stages of data 
analysis, processing, and modeling.

.. _GoFast: https://github.com/WEgeophysics/gofast

"""