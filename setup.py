#!/usr/bin/env python

# Standard library imports
from setuptools import setup
import builtins

# Compatibility layer for Python 2 and 3
try:
    import builtins # noqa 
except ImportError:
    import __builtin__ as builtins  # Python 2 compatibility

# Global flag to prevent premature loading of modules during setup
builtins.__GOFAST_SETUP__ = True

# Determine package version
try:
    import gofast
    VERSION = gofast.__version__
except ImportError:
    VERSION = '0.1.0'

# Package metadata
DISTNAME = "gofast"
DESCRIPTION = "Accelerate Your Machine Learning Workflow"
LONG_DESCRIPTION = open('README.md', 'r', encoding='utf8').read()
MAINTAINER = "Laurent Kouadio"
MAINTAINER_EMAIL = 'etanoyau@gmail.com'
URL = "https://github.com/WEgeophysics/gofast"
DOWNLOAD_URL = "https://pypi.org/project/gofast/#files"
LICENSE = "BSD-3-Clause"
PROJECT_URLS = {
    "API Documentation": "https://gofast.readthedocs.io/en/latest/api_references.html",
    "Home page": "https://gofast.readthedocs.io",
    "Bugs tracker": "https://github.com/WEgeophysics/gofast/issues",
    "Installation guide": "https://gofast.readthedocs.io/en/latest/installation.html",
    "User guide": "https://gofast.readthedocs.io/en/latest/user_guide.html",
}
KEYWORDS = "machine learning, algorithm, processing"

# Package data specification
PACKAGE_DATA = {
    'gofast': [
        'tools/_openmp_helpers.pxd',
        'geo/espg.npy',
        'etc/*',
        '_gflog.yml',
        'gflogfiles/*.txt',
        'pyx/*.pyx'
    ],
    "": [
        "*.pxd",
        'data/*',
        'examples/*.py',
        'examples/*.txt',
    ]
}

# Entry points and other dynamic settings
setup_kwargs = {
    'entry_points': {
        'gofast.commands': [
            'wx=gofast.cli:cli',
        ],
        'console_scripts': [
            'version=gofast.cli:version',
        ]
    },
    'packages': [
        'gofast',
        'gofast._build',
        'gofast.analysis',
        'gofast.datasets',
        'gofast.datasets.data',
        'gofast.datasets.descr',
        'gofast.experimental',
        'gofast.externals',
        'gofast.externals._pkgs',
        'gofast.geo',
        'gofast.models',
        'gofast.plot',
        'gofast.pyx',
        'gofast.stats',
        'gofast.tools',
    ],
    'install_requires': [
        "seaborn>=0.12.0",
        "pandas>=1.4.0",
        "cython>=0.29.33",
        "pyyaml>=5.0.0",
        "tqdm>=4.64.1",
        "joblib>=1.2.0",
        "threadpoolctl>=3.1.0",
        "matplotlib>=3.5.3",
        "statsmodels>=0.13.1",
        "numpy>=1.23.0",
        "scipy>=1.9.0",
        "h5py>=3.2.0",
        "pytest",
        "unittest",
    ],
    'extras_require': {
        "dev": [
            "click",
            "pyproj>=3.3.0",
            "openpyxl>=3.0.3",
        ]
    },
    'python_requires': '>=3.9'
}

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
    license=LICENSE,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Programming Language :: C",
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
