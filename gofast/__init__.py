# -*- coding: utf-8 -*-
# Licence:BSD-3-Clause
# Author: L. Kouadio <etanoyau@gmail.com>

from __future__ import annotations 
import os 
import sys 
import logging 
import random
import warnings 
 
# set the package name for consistency checker 
sys.path.insert(0, os.path.dirname(__file__))  
for p in ('.','..' ,'./gofast'): 
    sys.path.insert(0,  os.path.abspath(p)) 
    
# assert package 
if  __package__ is None: 
    sys.path.append( os.path.dirname(__file__))
    __package__ ='gofast'

# configure the logger file
# from ._gofastlog import gofastlog
try: 
    conffile = os.path.join(
        os.path.dirname(__file__),  "gofast/_gflog.yml")
    if not os.path.isfile (conffile ): 
        raise 
except: 
    conffile = os.path.join(
        os.path.dirname(__file__), "_gflog.yml")

# generated version by setuptools_scm 
try:
    from . import _version
    __version__ = _version.version.split('.dev')[0]
except ImportError:
    __version__ = "0.1.0"

# # set loging Level
logging.getLogger(__name__)#.setLevel(logging.WARNING)
# disable the matplotlib font manager logger.
logging.getLogger('matplotlib.font_manager').disabled = True
# or ust suppress the DEBUG messages but not the others from that logger.
# logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# setting up
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

# Workaround issue discovered in intel-openmp 2019.5:
# https://github.com/ContinuumIO/anaconda-issues/issues/11294
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

# https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/
# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)
try:
    # This variable is injected in the __builtins__ by the build process. 
    __GOFAST_SETUP__  # type: ignore
except NameError:
    __GOFAST_SETUP__ = False

if __GOFAST_SETUP__:
    sys.stderr.write("Partial import of gofast during the build process.\n")
else:
    from . import _distributor_init  # noqa: F401
    from . import _build  # noqa: F401
    from .utils._show_versions import show_versions
#https://github.com/pandas-dev/pandas
# Let users know if they're missing any of our hard dependencies
_main_dependencies = ("numpy", "scipy", "sklearn", "matplotlib", 
                      "pandas","seaborn", "openpyxl")
_missing_dependencies = []

for _dependency in _main_dependencies:
    try:
        __import__(_dependency)
    except ImportError as _e:  # pragma: no cover
        _missing_dependencies.append(
            f"{'scikit-learn' if _dependency=='sklearn' else _dependency }: {_e}")

if _missing_dependencies:  # pragma: no cover
    raise ImportError(
        "Unable to import required dependencies:\n" + "\n".join(_missing_dependencies)
    )
del _main_dependencies, _dependency, _missing_dependencies

# Try to suppress pandas future warnings
# and reduce verbosity.
# Setup WATex public API  
with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.filterwarnings(action='ignore', category=UserWarning)
    import sklearn 

from .datasets import ( 
    fetch_data, 
    ) 

from .utils import ( 
    read_data,
    cleaner, 
    reshape, 
    to_numeric_dtypes, 
    smart_label_classifier,
    )
try : 
    from .utils import ( 
        selectfeatures, 
        naive_imputer, 
        naive_scaler,  
        make_naive_pipe, 
        bi_selector, 
        )
except ImportError :
    pass 

def setup_module(module):
    """Fixture for the tests to assure globally controllable seeding of RNGs"""

    import numpy as np

    # Check if a random seed exists in the environment, if not create one.
    _random_seed = os.environ.get("GOFAST_SEED", None)
    if _random_seed is None:
        _random_seed = np.random.uniform() * np.iinfo(np.int32).max
    _random_seed = int(_random_seed)
    print("I: Seeding RNGs with %r" % _random_seed)
    np.random.seed(_random_seed)
    random.seed(_random_seed)
   
# Reset warnings to default to see warnings after this point
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
#  __all__ is used to display a few public API. 
# the public API is determined
# based on the documentation.
    
__all__ = [ 
    'show_versions',
    'sklearn',
    'fetch_data', 
    'read_data',
    'cleaner', 
    'reshape', 
    'to_numeric_dtypes', 
    'smart_label_classifier',
    'selectfeatures', 
    'naive_imputer', 
    'naive_scaler',  
    'make_naive_pipe', 
    'bi_selector', 
    ]

