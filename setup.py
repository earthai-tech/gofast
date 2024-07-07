#!/usr/bin/env python

# Standard library imports
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import builtins
import os
import sys
import subprocess

# Function to ensure Numpy is installed before proceeding
def install_numpy_if_needed():
    try:
        import numpy
        print("Numpy is already installed.")
    except ImportError:
        print("Numpy is not installed. Installing now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy<2.0"])
        import numpy # noqa
        print("Numpy has been installed successfully.")

# Ensure Numpy is installed before proceeding
install_numpy_if_needed()

# Now that Numpy is installed, you can safely import it
import numpy

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
URL = "https://github.com/earthai-tech/gofast"
DOWNLOAD_URL = "https://pypi.org/project/gofast/#files"
LICENSE = "BSD-3-Clause"
PROJECT_URLS = {
    "API Documentation": "https://gofast.readthedocs.io/en/latest/api_references.html",
    "Home page": "https://gofast.readthedocs.io",
    "Bugs tracker": "https://github.com/earthai-tech/gofast/issues",
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

def collect_pyx_modules(package_path):
    pyx_modules = []
    for root, dirs, files in os.walk(package_path):
        # Skip the 'tests' folder
        if 'tests' in root.split(os.sep):
            continue
        
        for file in files:
            if (
                file.endswith('.py') and not file.startswith('_') 
                and file != 'setup.py' and not file.startswith('test')
            ):
                module_path = os.path.join(root, file)
                pyx_module = module_path.replace('.py', '.pyx')
                os.rename(module_path, pyx_module)
                pyx_modules.append(pyx_module)
    return pyx_modules

def collect_all_pyx_modules(base_package_paths):
    all_pyx_modules = []
    for package_path in base_package_paths:
        all_pyx_modules.extend(collect_pyx_modules(package_path))
    #all_pyx_modules.extend(collect_pyx_modules('.'))
    return all_pyx_modules

base_package_paths = [
    # 'gofast/dataops',
    # 'gofast/estimators',
    'gofast/tools',
    # 'gofast/transformers',
    # 'gofast/stats',
    # 'gofast/models',
    # 'gofast/plots',
    # 'gofast/api',
    # 'gofast/backends',
    'gofast/datasets',
    # 'gofast/compat',
    # 'gofast/nn',
    # 'gofast/analysis',
    # 'gofast/cli',
]

pyx_modules = collect_all_pyx_modules(base_package_paths)
if not pyx_modules:
    print("No .pyx files found to compile.")
else:
    print("Found .pyx files:", pyx_modules)

ext_modules = [
    Extension(pyx_module.replace('/', '.').replace('.pyx', ''),
              [pyx_module], include_dirs=[numpy.get_include()])
    for pyx_module in pyx_modules
]

class BuildExt(build_ext):
    def build_extensions(self):
        numpy_includes = numpy.get_include()
        for ext in self.extensions:
            ext.include_dirs.append(numpy_includes)
        build_ext.build_extensions(self)

setup_kwargs = {
    'entry_points': {
        'console_scripts': [
            'gofast=gofast.cli:cli',
            'version=gofast.cli:version',
        ]
    },
    'packages': find_packages(),
    'ext_modules': cythonize(
        ext_modules, compiler_directives={'linetrace': True,'language_level': "3"}),
    'cmdclass': {'build_ext': BuildExt},
    'install_requires': [
        "cython>=0.29.33",
        "scikit-learn>=1.1.2",
        "seaborn>=0.12.0",
        "pandas<2.0.3",
        "pyyaml>=5.0.0",
        "tqdm>=4.64.1",
        "joblib>=1.2.0",
        "threadpoolctl>=3.1.0",
        "matplotlib>=3.5.3",
        "statsmodels>=0.13.1",
        "numpy<2.0",
        "scipy>=1.9.0",
        "h5py>=3.2.0",
        "pytest",
    ],
    'extras_require': {
        "dev": [
            "click",
            "pyproj>=3.3.0",
            "openpyxl>=3.0.3",
            "tensorflow>=2.15.0"
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
