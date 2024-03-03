==================================
Installation of GoFast
==================================

GoFast is a Python library designed to accelerate your machine learning workflow. This page 
guides you through the process of installing GoFast.

Requirements
------------

Before installing GoFast, ensure that you have the following prerequisites installed:

- Python (>= 3.6)
- NumPy (>= 1.13.3)
- SciPy (>= 0.19.1)
- Scikit-Learn (>= 0.20)

These packages are required for GoFast to function correctly. 

Installation
------------

You can install GoFast using pip, Python’s package installer. To install the latest stable release of GoFast, run the following command in your terminal:

.. code-block:: bash

    pip install gofast

Alternatively, if you want to install the latest development version directly from the source code, you can clone the repository and install it manually:

.. code-block:: bash

    git clone https://github.com/yourusername/gofast.git
    cd gofast
    pip install .

Verifying Installation
----------------------

After installation, you can verify that GoFast is installed correctly by running:

.. code-block:: python

    import gofast
    print(gofast.__version__)

This command should output the version number of GoFast, indicating that it has been 
installed successfully.

Troubleshooting
---------------

If you encounter any issues during the installation of GoFast, please refer to our 
[Troubleshooting Guide](https://gofast-docs.org/troubleshooting) or open an issue on our 
[GitHub repository](https://github.com/WEgeophysics/gofast/issues).

Support
-------

For further assistance, you can reach out to our community support channels or consult 
the GoFast documentation at [https://gofast-docs.org](https://gofast-docs.org).

.. note:: This documentation is always evolving and can be contributed to on our GitHub repository.
