# -*- coding: utf-8 -*-

from .numpy import NumpyBackend
from .scipy import ScipyBackend
from .dask import DaskBackend
from .cupy import CuPyBackend

class BackendSelector:
    """
    Manages and selects the most suitable computational backend for gofast tasks,
    considering user preferences, available hardware, and installed libraries.
    Supports automatic detection of the optimal backend based on the system's
    capabilities or allows for explicit selection by the user.

    Parameters
    ----------
    preferred_backend : str, optional
        The name of the preferred backend. Valid options include "numpy", "scipy",
        "dask", and "cupy". If not provided, the selector will automatically
        choose the most suitable backend based on the current environment and
        available resources.

    Attributes
    ----------
    backends : dict
        A mapping from backend names to their corresponding class implementations.
    selected_backend : object
        The backend class instance that has been selected based on the preference
        or automatic detection.

    Methods
    -------
    select_backend(preferred_backend)
        Determines and initializes the backend based on the user's preference or
        an automatic selection process.
    is_gpu_available()
        Checks if a CUDA-compatible GPU is available for GPU-accelerated computations.
    is_dask_available()
        Verifies if Dask is installed for distributed computing capabilities.
    is_scipy_available()
        Confirms the availability of SciPy for advanced scientific computations.
    auto_select_backend()
        Automatically identifies and selects the most suitable backend.
    get_backend()
        Returns the currently selected backend instance.

    Examples
    --------
    Explicitly selecting a backend:

    >>> from gofast.backends.selector import BackendSelector
    >>> backend_selector = BackendSelector(preferred_backend='cupy')
    >>> backend = backend_selector.get_backend()
    CuPyBackend selected by user preference.

    Automatically selecting the best available backend:

    >>> backend_selector = BackendSelector()
    >>> backend = backend_selector.get_backend()
    # This will print the selected backend based on available resources

    Using the selected backend for computations:

    >>> array = backend.array([1, 2, 3])
    # Perform operations using the selected backend

    Notes
    -----
    The automatic backend selection process prioritizes GPU acceleration with CuPy
    when available, falling back to distributed computing with Dask, advanced
    computations with SciPy, or general-purpose computations with NumPy as the
    default option.
    """
    def __init__(self, preferred_backend=None):
        self.backends = {
            "numpy": NumpyBackend,
            "scipy": ScipyBackend,
            "dask": DaskBackend,
            "cupy": CuPyBackend
        }
        self.selected_backend = self.select_backend(preferred_backend)

    def select_backend(self, preferred_backend):
        """
        Selects the backend based on user preference or environment.
        """
        if preferred_backend in self.backends:
            print(f"{preferred_backend.capitalize()}Backend selected by user preference.")
            return self.backends[preferred_backend]()
        
        # Fallback to automatic selection if no preference 
        # or invalid preference is provided
        return self.auto_select_backend()

    def is_gpu_available(self):
        """
        Checks if a CUDA-compatible GPU is available.
        """
        try:
            import cupy
            cupy.array([1])  # Simple operation to ensure CuPy can use the GPU
            return True
        except (ImportError, cupy.cuda.runtime.CUDARuntimeError):
            return False

    def is_dask_available(self):
        """
        Checks if Dask is installed.
        """
        try:
            import dask # noqa 
            return True
        except ImportError:
            return False

    def is_scipy_available(self):
        """
        Checks if SciPy is installed.
        """
        try:
            import scipy # noqa 
            return True
        except ImportError:
            return False

    def auto_select_backend(self):
        """
        Automatically selects the most suitable backend.
        """
        if self.is_gpu_available():
            print("CuPyBackend selected for GPU acceleration.")
            return self.backends["cupy"]()
        
        if self.is_dask_available():
            print("DaskBackend selected for distributed computing.")
            return self.backends["dask"]()

        if self.is_scipy_available():
            print("ScipyBackend selected for advanced scientific computations.")
            return self.backends["scipy"]()

        print("NumpyBackend selected as the default backend.")
        return self.backends["numpy"]()

    def get_backend(self):
        """
        Returns the selected backend.
        """
        return self.selected_backend