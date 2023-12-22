# setup.py

from Cython.Build import cythonize
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Define the extension modules
extensions = [
    Extension("cython_functions", [
    "cython_functions.pyx",
    "ml_bases.pyx", 
    "metric_bases.pyx", 
    "taylor_diagram.pyx",
    "static_typing.pyx",    
    "memory_view.pyx", 
    "parallel_processing",
    "cluster_bases", 
     ],
      include_dirs=[numpy.get_include()],
      extra_compile_args=["-fopenmp"],
      extra_link_args=["-fopenmp"]
      )
]

# Setup configuration
setup(
    name="CythonFunctions",
    ext_modules=cythonize(extensions),
    zip_safe=False
)
