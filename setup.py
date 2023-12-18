#!/usr/bin/env python

from setuptools import setup #. find_packages

try:
    import builtins
except ImportError:
    # Python 2 compat: just to be able to declare that Python >=3.8 is needed.
    import __builtin__ as builtins

# This is a bit (!) hackish: we are setting a global variable so that the main
# gofast __init__ can detect if it is being loaded by the setup routine, to
# avoid attempting to load components that aren't built yet.
builtins.__GOFAST_SETUP__ = True

# We can actually import gofast version from 
# in editable mode :$ python -m pip install -e .
try: 
    import gofast  # noqa
    VERSION = gofast.__version__
except: VERSION ='0.1.0'
# set global variables 
DISTNAME = "gofast"
DESCRIPTION= "Accelerate Your Machine Learning Workflow"
with open('README.md', 'r', encoding ='utf8') as fm:
    LONG_DESCRIPTION =fm.read()
MAINTAINER = "Laurent Kouadio"
MAINTAINER_EMAIL = 'etanoyau@gmail.com'
URL = "https://github.com/WEgeophysics/gofast"
DOWNLOAD_URL = "https://pypi.org/project/gofast/#files"
LICENSE = "BSD-3-Clause"
PROJECT_URLS = {
    "API Documentation"  : "https://gofast.readthedocs.io/en/latest/api_references.html",
    "Home page" : "https://gofast.readthedocs.io",
    "Bugs tracker": "https://github.com/WEgeophysics/gofast/issues",
    "Installation guide" : "https://gofast.readthedocs.io/en/latest/installation.html", 
    "User guide" : "https://gofast.readthedocs.io/en/latest/user_guide.html",
}
KEYWORDS= "machine learning, algorithm, processing"
# the commented metadata should be upload as
# packages rather than data. See future release about 
# setuptools: see https://packaging.python.org/en/latest/specifications/declaring-project-metadata/
PACKAGE_DATA={ 
    'gofast': [
            'utils/_openmp_helpers.pxd', 
            'utils/espg.npy',
            'etc/*', 
            '_gflog.yml', 
            'gflogfiles/*.txt',
                ], 
        "":["*.pxd",
            'data/*', 
            'examples/*.py', 
            'examples/*.txt', 
            ]
 }
# setting up 
#initialize
setup_kwargs = dict()
# commands
setup_kwargs['entry_points'] = {
    'gofast.commands': [
        'wx=gofast.cli:cli',
        ],
    'console_scripts':[
        'version= gofast.cli:version', 
                    ]
      }

setup_kwargs['packages'] = [ 
    'gofast',
    'gofast.datasets',
    'gofast.utils',
    'gofast.analysis',
    'gofast.validation',
    'gofast.geosciences', 
    'gofast.externals',
    'gofast.plot',
    'gofast.datasets.data', 
    'gofast.datasets.descr', 
    'gofast._build', 
    'gofast.externals._pkgs', 
     ]

setup_kwargs['install_requires'] = [    
    "seaborn >=0.12.0", 
    "pandas >=1.4.0",
    "cython >=0.29.33",
    "pyyaml >=5.0.0", 
    "openpyxl >=3.0.3",
    "pyproj >=3.3.0",
    "tqdm >=4.64.1",
    "tables >=3.6.0",
    "scikit-learn ==1.2.1",
    "joblib >=1.2.0",
    "threadpoolctl >=3.1.0",
    "matplotlib ==3.5.3",
    "statsmodels >=0.13.1", 
    "numpy >=1.23.0", 
    "scipy >=1.9.0",
    "h5py >=3.2.0",
    "pytest"
 ]
      
setup_kwargs['extras_require']={
     "dev" : ["click", 
	          "xgboost >=1.7.0",
              "missingno>=0.4.2", 
              "yellowbrick>=1.5.0", 
              "pyjanitor>=0.1.7", 
              "mlxtend>=0.21"
             ]  
     } 
                     
setup_kwargs['python_requires'] ='>=3.9'

setup(
 	name=DISTNAME,
 	version=VERSION,
 	author=MAINTAINER,
    author_email=MAINTAINER_EMAIL,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
 	description=DESCRIPTION,
 	long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    download_url=DOWNLOAD_URL, 
    project_urls=PROJECT_URLS,
 	include_package_data=True,
 	license=LICENSE,
 	classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Software Development",
        'Topic :: Scientific/Engineering',
        'Programming Language :: C ',
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: Implementation :: CPython",
        "Operating System :: OS Independent",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        ],
    keywords=KEYWORDS,
    zip_safe=True, 
    package_data=PACKAGE_DATA,
 	**setup_kwargs
)

























