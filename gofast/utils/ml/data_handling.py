# -*- coding: utf-8 -*-
#   Licence:BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>
"""
Manages data operations including loading, saving, batching, resampling,
and manipulation to facilitate efficient data processing and preparation.
"""

import os
import math
import tarfile
import warnings
import shutil
from six.moves import urllib
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ...api.types import Any,  Optional, Union
from ...api.types import DataFrame
from ...core.io import EnsureFileExists
from ...exceptions import DependencyError
from ..deps_utils import ensure_pkg


__all__= [ 
    'fetch_tgz',
    'fetch_tgz_in',
    'save_dataframes',
    'dynamic_batch_size',
    'get_batch_size',
    'prepare_tf_dataset', 
    'get_tf_dataset_size', 
    'compute_batch_size',
    ]

TORCH_DEP_EMSG=(
    "Error: PyTorch is not installed. Please install PyTorch by running:\n"
    "'pip install torch' or follow the installation instructions at:\n"
    "https://pytorch.org/get-started/locally/"
)

TF_DEP_EMSG= (
    "Error: TensorFlow is not installed. Please install TensorFlow "
    "by running:\n 'pip install tensorflow' or visit the installation "
    "guide at:\n https://www.tensorflow.org/install"
)

@ensure_pkg(
    "torch", extra="Torch is needed when backend is set to `torch`.",
    partial_check= True,
    condition= lambda *args, **kwargs: kwargs.get("backend")=="torch"
    )
@ensure_pkg(
    "tensorflow", extra="Tensorflow is needed when backend is set to `tf`.",
    partial_check= True,
    condition= lambda *args, **kwargs: kwargs.get(
        "backend") in ("tensorflow", "tf")
    )

def get_batch_size(
    data_or_train_loader, *, 
    model=None, 
    device=None,
    initial_batch_size=32, 
    max_batch_size=512,
    default_size=None, 
    backend=None, 
    verbose=0,
):
    """
    Automatically determines the optimal batch size for training based on
    available hardware, model, and backend.

    This function adjusts the batch size by running the model with increasing
    batch sizes until it encounters memory limitations, at which point it will
    fall back to the largest feasible batch size. The backend can be set to 
    either 'torch' (PyTorch) or 'tensorflow' (TensorFlow), and depending on
    the device availability (GPU or CPU), it will adjust the batch size 
    accordingly.

    The process involves testing batch sizes starting from `initial_batch_size`
    and doubling them until either memory limits are reached or `max_batch_size`
    is exceeded. If an error occurs due to lack of memory, the batch size is
    reduced until it is a feasible value. If no GPU is available or there are
    insufficient resources, a fallback batch size is returned.

    Parameters
    ----------
    data_or_train_loader : Dataset or DataLoader
        The dataset or data loader to be used for training. If a DataLoader 
        is provided, the batch size will be determined dynamically. If a 
        raw dataset is provided, it will be wrapped in a DataLoader during 
        the process.
        
    model : torch.nn.Module or tf.keras.Model, optional
        The model to be used for training. If a model is provided, it will
        be used to test the batch size. If no model is provided, the batch 
        size determination will be done based on dataset characteristics.

    device : torch.device or tf.device, optional
        The device (CPU or GPU) where the model is to be trained. This 
        should be provided if using a framework like PyTorch that requires
        explicit device placement.

    initial_batch_size : int, optional
        The starting batch size for testing. This value is used as the 
        initial batch size for determining the optimal batch size. Default 
        is 32.

    max_batch_size : int, optional
        The maximum batch size to attempt. The batch size will never exceed 
        this value, even if the available memory allows for larger sizes. 
        Default is 512.

    default_size : int, optional
        The fallback batch size if the batch size determination process fails 
        or the backend is unsupported. Default is None, which means the function 
        will use `initial_batch_size` as the fallback.

    backend : str, optional
        The backend to use for training. This can either be 'torch' for 
        PyTorch or 'tensorflow' for TensorFlow. Default is 'torch'.

    verbose : int, optional
        The verbosity level. If set to 1, the function will print information 
        about the batch size determination process. Default is 0 (silent).

    Returns
    -------
    int
        The determined batch size that can be used for training.

    Methods
    -------
    - `get_batch_size`: Main function to dynamically determine the batch size.
    - `_use_torch_available`: Checks if PyTorch is available.
    - `_prepare_tf_dataset`: Prepares TensorFlow dataset for testing.
    - `_analyze_dataset`: Analyzes dataset to estimate batch size.
    - `compute_smart_batch_size`: Heuristic computation for batch size based 
      on dataset size and number of features.

    Notes
    -----
    The batch size determination process involves the following steps:

    1. **Test Memory Capacity**:
        The function starts with an initial batch size, `B_0`, and 
        progressively doubles it, checking for memory overflow at each step:
        
        .. math::
            B_{i+1} = 2 \cdot B_i \quad \text{until memory overflows.}
            
    2. **Handling Memory Overflow**:
        Upon encountering memory limitations, the batch size is reduced:
        
        .. math::
            B_{\text{final}} = \max \left( \frac{B_i}{2}, B_{\text{min}} \right)
            
    3. **Final Batch Size**:
        The final batch size is determined by returning either the largest
        feasible batch size or the fallback batch size:
        
        .. math::
            B_{\text{final}} = \min \left( B_{\text{final}}, B_{\text{max}} \right)
            
    - This function requires the PyTorch or TensorFlow library to be installed.
    - If neither GPU nor sufficient memory is available, the function falls back 
      to a heuristic method to determine a reasonable batch size based on dataset 
      characteristics.
    - The `device` argument is only applicable to PyTorch. TensorFlow handles 
      device placement internally.

    Example
    -------
    >>> from gofast.utils.mlutils import get_batch_size
    >>> batch_size = get_batch_size(data, model=model, device=device, 
    ...                             backend='torch', verbose=1)
    >>> print(f"Optimal batch size: {batch_size}")

    See Also
    --------
    - `torch.utils.data.DataLoader`
    - `tensorflow.data.Dataset`
    - `compute_smart_batch_size`
    
    References
    ----------
    .. [1] Paszke, A., et al., "PyTorch: An Imperative Style, High-Performance
           Deep Learning Library," in NeurIPS, 2019.
    .. [2] Abadi, M., et al., "TensorFlow: Large-Scale Machine Learning 
          on Heterogeneous Distributed Systems," 2016.
    """
    # Determine batch size for PyTorch backend
    backend= str(backend).lower() 
    if backend== 'torch':
        try:
            import torch
            import torch.utils.data as Tc_data
        except ImportError:
            raise DependencyError(TORCH_DEP_EMSG)
            
        if torch.cuda.is_available():
            batch_size = initial_batch_size
            while batch_size <= max_batch_size:
                try:
                    data_loader = Tc_data.DataLoader(
                        data_or_train_loader.dataset,
                        batch_size=batch_size
                    )

                    # Test the batch size by running a single batch
                    for batch in data_loader:
                        inputs, targets = batch
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                        model(inputs)
                        break  # Only need to test one batch

                    if verbose:
                        print(f"Batch size {batch_size} fits in memory.")
                    batch_size *= 2  # Double the batch size for next iteration

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()  # Free memory
                        batch_size = max(batch_size // 2, initial_batch_size)
                        if verbose:
                            print(
                                f"Out of memory! Largest batch size: {batch_size}."
                            )
                        return batch_size
                    else:
                        raise e
            return min(batch_size, max_batch_size)
        else:
            if verbose:
                print(
                    "CUDA not available or not enough memory."
                    " Using fallback batch size."
                )
            return default_size if default_size is not None else 32

    # Determine batch size for TensorFlow backend
    elif backend.lower() == 'tensorflow':
        try:
            import tensorflow as tf
        except ImportError:
            raise DependencyError(TF_DEP_EMSG)

        if tf.config.list_physical_devices('GPU'):
            batch_size = initial_batch_size
            while batch_size <= max_batch_size:
                try:
                    # Prepare dataset for testing the batch size
                    test_ds = prepare_tf_dataset(data_or_train_loader, batch_size)

                    # Run a single batch through the model
                    for inputs, targets in test_ds.take(1):
                        model(inputs)

                    if verbose:
                        print(f"Batch size {batch_size} fits in memory.")
                    batch_size *= 2

                except tf.errors.ResourceExhaustedError:
                    batch_size = max(batch_size // 2, initial_batch_size)
                    if verbose:
                        print(
                            f"Out of memory! Largest batch size: {batch_size}."
                        )
                    return batch_size
            return min(batch_size, max_batch_size)
        else:
            if verbose:
                print(
                    "No GPU available for TensorFlow or insufficient memory."
                    " Using fallback batch size."
                )
            return default_size if default_size is not None else 32

    # Fallback heuristic for other cases (non-GPU or non-supported backends)
    else:
        if default_size is not None:
            if verbose:
                print(f"Using fallback batch size: {default_size}")
            return default_size

        try:
            # Analyze dataset to estimate optimal batch size
            dataset_size, num_features = analyze_dataset(data_or_train_loader)
            # Use heuristic based on number of features and dataset size
            if num_features > dataset_size:
                warnings.warn(
                    "High number of features detected."
                    " Using dataset_size // 4 for batch size."
                )
                heuristic_batch_size = max(1, dataset_size // 4)
            else:
                heuristic_batch_size = compute_batch_size(
                    dataset_size, num_features, max_batch_size)

            if verbose:
                print(f"Using heuristic batch size: {heuristic_batch_size}")
            return heuristic_batch_size

        except Exception:
            warnings.warn(
                "Unable to determine batch size using heuristics. "
                "Using default batch size of 32."
            )
            if verbose:
                print("Using default batch size: 32")
            return 32

@ensure_pkg(
    "tensorflow", extra="Tensorflow is needed when backend is set to `tf`.",
    )
def prepare_tf_dataset(data_or_train_loader, batch_size):
    """
    Prepare a TensorFlow dataset with the specified batch size.

    This function takes a TensorFlow Dataset or a compatible data 
    structure and applies batching to it based on the provided `batch_size`.
    It ensures that the data is properly batched for efficient training
    with TensorFlow models.

    Parameters
    ----------
    data_or_train_loader : tf.data.Dataset or TF_Dataset
        The TensorFlow Dataset or compatible data to be batched. This can 
        include any iterable TensorFlow data structures that support batching.
    
    batch_size : int
        The number of samples per batch. This determines how many samples 
        will be propagated through the network at once.
    
    Returns
    -------
    tf.data.Dataset
        A batched TensorFlow dataset ready for training. The returned dataset 
        will yield batches of size `batch_size`.
    
    Examples
    --------
    >>> import tensorflow as tf
    >>> from gofast.utils.mlutils import _prepare_tf_dataset
    >>> dataset = tf.data.Dataset.from_tensor_slices(
        (tf.random.normal([100, 10]), tf.random.uniform([100], maxval=2, dtype=tf.int32)))
    >>> batched_dataset = _prepare_tf_dataset(dataset, batch_size=32)
    >>> for batch in batched_dataset.take(1):
    ...     inputs, targets = batch
    ...     print(inputs.shape, targets.shape)
    (TensorShape([32, 10]), TensorShape([32]))
    
    Notes
    -----
    - This function is intended for internal use within the `mlutils` module.
    - It assumes that the input `data_or_train_loader` is compatible with TensorFlow's 
      batching operations. Unsupported data types will raise a `TypeError`.
    
    See Also
    --------
    tf.data.Dataset.batch : Apply batching to a TensorFlow dataset.
    
    References
    ----------
    .. [1] Abadi, M., et al., "TensorFlow: Large-Scale Machine Learning 
           on Heterogeneous Distributed Systems," 2016.
    """
        
    try:
        import tensorflow as tf
        from tensorflow.data import Dataset as TF_Dataset
    except ImportError:
        raise DependencyError(TORCH_DEP_EMSG)
        
    if isinstance(data_or_train_loader, TF_Dataset):
        return data_or_train_loader.batch(batch_size)
    elif isinstance(data_or_train_loader, tf.data.Dataset):
        return data_or_train_loader.batch(batch_size)
    else:
        raise TypeError("Unsupported data type for TensorFlow backend.")


@ensure_pkg(
    "torch", extra="Torch is needed when backend is set to `torch`.",
    partial_check= True,
    condition= lambda *args, **kwargs: kwargs.get("backend")=="torch"
    )
@ensure_pkg(
    "tensorflow", extra="Tensorflow is needed when backend is set to `tf`.",
    partial_check= True,
    condition= lambda *args, **kwargs: kwargs.get(
        "backend") in ("tensorflow", "tf")
    )
def analyze_dataset(data_or_train_loader, backend=None):
    """
    Analyze the dataset to determine its size and number of features.

    This function inspects the provided dataset or data loader to 
    estimate the total number of samples (`dataset_size`) and the number 
    of features (`num_features`) per sample. This information is crucial 
    for determining an appropriate batch size and optimizing training 
    performance.

    Parameters
    ----------
    data_or_train_loader : torch.utils.data.Dataset, tf.data.Dataset,\
        list, tuple, or np.ndarray
        The dataset or data loader to be analyzed. It can be a PyTorch Dataset, 
        TensorFlow Dataset, or standard Python iterable such as a list, tuple, 
        or NumPy array.
    
    Returns
    -------
    tuple
        A tuple containing:
        
        - `dataset_size` (int): The total number of samples in the dataset.
        - `num_features` (int): The number of features per sample.
    
    Examples
    --------
    >>> import torch
    >>> from torch.utils.data import TensorDataset
    >>> from gofast.utils.mlutils import _analyze_dataset
    >>> dataset = TensorDataset(torch.randn(1000, 20), torch.randint(0, 2, (1000,)))
    >>> size, features = _analyze_dataset(dataset)
    >>> print(size, features)
    1000 20
    
    >>> import tensorflow as tf
    >>> from gofast.utils.mlutils import _analyze_dataset
    >>> tf_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.random.normal([500, 15]), tf.random.uniform([500], maxval=2, dtype=tf.int32)))
    >>> size, features = _analyze_dataset(tf_dataset)
    >>> print(size, features)
    500 15
    
    Notes
    -----
    - This function supports both PyTorch and TensorFlow datasets, as well as 
      standard Python iterables like lists, tuples, and NumPy arrays.
    - For high-dimensional data, `num_features` is determined based on the 
      structure of the first sample.
    
    See Also
    --------
    get_batch_size : Main function to determine batch size based on dataset analysis.
    get_tf_dataset_size : Helper function to estimate the size of a TensorFlow dataset.
    
    References
    ----------
    .. [1] Paszke, A., et al., "PyTorch: An Imperative Style, High-Performance
          Deep Learning Library," in NeurIPS, 2019.
    .. [2] Abadi, M., et al., "TensorFlow: Large-Scale Machine Learning on 
         Heterogeneous Distributed Systems," 2016.
    """
    
    backend= str(backend).lower() 

    if backend=="torch": 
        try:
            import torch
            import torch.utils.data as Tc_data
        except ImportError:
            raise DependencyError(TORCH_DEP_EMSG) 
            
        if isinstance(data_or_train_loader, (
                torch.utils.data.Dataset, Tc_data.Dataset)):
            
            dataset_size = len(data_or_train_loader)
            sample = data_or_train_loader[0]
            if isinstance(sample, (list, tuple)):
                num_features = len(sample[0])
            elif isinstance(sample, dict):
                num_features = len(sample)
            else:
                num_features = 1
                
    elif backend in ("tensorflow", "tf"): 
        try:
            import tensorflow as tf
            from tensorflow.data import Dataset as TF_Dataset
        except ImportError:
            raise DependencyError(TF_DEP_EMSG) 
    
        if isinstance(data_or_train_loader, (tf.data.Dataset, TF_Dataset)):
            dataset_size = get_tf_dataset_size(data_or_train_loader)
            for sample in data_or_train_loader.take(1):
                inputs, _ = sample
                if isinstance(inputs, (list, tuple)):
                    num_features = len(inputs)
                elif isinstance(inputs, dict):
                    num_features = len(inputs)
                else:
                    num_features = inputs.shape[-1] if len(inputs.shape) > 1 else 1
                    
    elif isinstance(data_or_train_loader, (list, tuple, np.ndarray)):
        dataset_size = len(data_or_train_loader)
        sample = data_or_train_loader[0]
        if isinstance(sample, (list, tuple)):
            num_features = len(sample)
        elif isinstance(sample, dict):
            num_features = len(sample)
        else:
            num_features = 1
    else:
        raise TypeError("Unsupported data type for dataset analysis.")
    
    return dataset_size, num_features

@ensure_pkg(
    "tensorflow", extra="Tensorflow is needed for getting tf dataset size.",
    )
def get_tf_dataset_size(tf_dataset):
    """
    Estimate the size of a TensorFlow dataset.

    This function attempts to determine the number of samples in a 
    TensorFlow dataset. It uses TensorFlow's `cardinality` method to estimate 
    the dataset size. If the size cannot be determined, a `ValueError` is raised.

    Parameters
    ----------
    tf_dataset : tf.data.Dataset
        The TensorFlow dataset whose size is to be estimated.
    
    Returns
    -------
    int
        The estimated number of samples in the TensorFlow dataset.
    
    Examples
    --------
    >>> import tensorflow as tf
    >>> from gofast.utils.mlutils import _get_tf_dataset_size
    >>> tf_dataset = tf.data.Dataset.from_tensor_slices(tf.random.normal([200, 10]))
    >>> size = _get_tf_dataset_size(tf_dataset)
    >>> print(size)
    200
    
    Notes
    -----
    - This function relies on TensorFlow's `cardinality` method, which may 
      return `tf.data.experimental.INFINITE_CARDINALITY` or 
      `tf.data.experimental.UNKNOWN_CARDINALITY` for certain datasets.
    - Ensure that the dataset is fully defined and does not have infinite or 
      unknown cardinality before using this function.
    
    See Also
    --------
    tf.data.Dataset.cardinality : TensorFlow method to determine dataset size.
    _analyze_dataset : Helper function that uses this method to analyze datasets.
    
    References
    ----------
    .. [1] Abadi, M., et al., "TensorFlow: Large-Scale Machine 
      Learning on Heterogeneous Distributed Systems," 2016.
    """
        
    try:
        import tensorflow as tf
    except ImportError:
        raise DependencyError(TF_DEP_EMSG) 
    
    try:
        cardinality = tf.data.experimental.cardinality(tf_dataset).numpy()
        if cardinality == tf.data.experimental.INFINITE_CARDINALITY:
            raise ValueError("Dataset has infinite cardinality.")
        if cardinality == tf.data.experimental.UNKNOWN_CARDINALITY:
            raise ValueError("Unable to determine TensorFlow dataset size.")
        return int(cardinality)
    except Exception:
        raise ValueError("Unable to determine TensorFlow dataset size.")

def compute_batch_size( 
        dataset_size=None, num_features=None, data=None, max_batch_size=512):
    """
    Compute a smart batch size based on dataset size and number of features.

    This function employs a heuristic to determine an appropriate batch 
    size by considering both the total number of samples (`dataset_size`) and 
    the number of features (`num_features`) per sample. The heuristic aims to 
    balance memory usage and training efficiency.

    Parameters
    ----------
    dataset_size : int
        The total number of samples in the dataset. Is essential if `data` 
        is ``None``.
    
    num_features : int, optional
        The number of features per sample. Should be provided
        if `data` is ``None``.
    
    data: ArrayLike or pd.DataFrame, optional 
       If given, `dataset_size` and `num_features` should be determined 
       accordingly if one or both are not provided. 
    
    max_batch_size : int, default=512
        The maximum allowed batch size. The computed batch size will not exceed 
        this value. Default value is set to ``512``.
    
    Returns
    -------
    int
        The computed smart batch size suitable for training.
    
    Examples
    --------
    >>> from gofast.utils.mlutils import compute_batch_size
    >>> batch_size = compute_smart_batch_size(1000, 50, 512)
    >>> print(batch_size)
    32
    
    Notes
    -----
    - The function uses logarithmic scaling based on the dataset size to 
      determine an initial batch size estimate.
    - If the number of features exceeds the dataset size, a more conservative 
      batch size is chosen to prevent potential overfitting and memory issues.
    
    See Also
    --------
    get_batch_size : Main function that utilizes this helper for batch size computation.
    
    References
    ----------
    .. [1] LeCun, Y., et al., "Gradient-Based Learning Applied to 
       Document Recognition," 1998.
    """
    if data is not None: 
        if isinstance ( data, (np.ndarray, pd.DataFrame)): 
            dataset_size = dataset_size or data.shape [0] 
            num_features = num_features or data.shape [1]
            
    # Logarithmic scaling based on dataset size
    batch_size = int(math.log(dataset_size + 1, 2))  # Avoid log(0)
    batch_size = max(32, batch_size)
    
    # Adjust batch size based on number of features
    if num_features > dataset_size:
        batch_size = max(1, dataset_size // 4)
    else:
        batch_size = min(batch_size, max_batch_size)
    
    return batch_size

@ensure_pkg(
    "torch", extra="Torch is needed when backend is set to `torch`.",
    partial_check= True,
    condition= lambda *args, **kwargs: kwargs.get("backend")=="torch"
    )
@ensure_pkg(
    "tensorflow", extra="Tensorflow is needed when backend is set to `tf`.",
    partial_check= True,
    condition= lambda *args, **kwargs: kwargs.get(
        "backend")in ("tensorflow", "tf")
    )
def dynamic_batch_size(
    current_batch_size, performance_metrics, *,
    accuracy_threshold=0.90, scale_factor=2,
    default_size=None, backend='torch', verbose=0
):
    """
    Dynamically adjust the batch size based on model performance.

    This function modifies the batch size during training based on the 
    evaluation of performance metrics such as accuracy. If the performance 
    meets or exceeds a specified threshold, the batch size is scaled up by 
    a defined factor to potentially enhance training efficiency. If not, 
    the batch size remains unchanged or is set to a default value.

    Parameters
    ----------
    current_batch_size : int
        The current batch size being used in training.
    
    performance_metrics : dict
        A dictionary containing performance metrics, such as 
        {'accuracy': 0.92, 'loss': 0.25}. These metrics are used to decide 
        whether to adjust the batch size.
    
    accuracy_threshold : float, optional
        The threshold of accuracy at which to increase the batch size. If the 
        model's accuracy meets or exceeds this value, the batch size will be 
        increased. Default is 0.90.
    
    scale_factor : int, optional
        The factor by which to scale the batch size when performance improves.
        For example, a `scale_factor` of 2 will double the batch size. Default 
        is 2.
    
    default_size : int, optional
        The fallback batch size to use if no improvement in performance is 
        detected or if the backend is unsupported. Default is None, which means 
        the function will retain the `current_batch_size`.
    
    backend : str, optional
        The backend to use for adjustment. This can either be 'torch' for 
        PyTorch or 'tensorflow' for TensorFlow. Default is 'torch'.
    
    verbose : int, optional
        The verbosity level. If set to 1, the function will print information 
        about the batch size adjustment process. Default is 0 (silent).
    
    Returns
    -------
    int
        The adjusted batch size based on performance metrics.
    
    Examples
    --------
    >>> from gofast.utils.mlutils import dynamic_batch_size
    >>> performance_metrics = {'accuracy': 0.92, 'loss': 0.25}
    >>> new_batch_size = dynamic_batch_size(
    ...     current_batch_size=32,
    ...     performance_metrics=performance_metrics,
    ...     backend='torch',
    ...     verbose=1
    ... )
    Performance improved (accuracy: 0.92). Increasing batch size to 64.
    >>> print(new_batch_size)
    64
    
    Notes
    -----
    - The function supports both PyTorch and TensorFlow backends, adjusting 
      the batch size accordingly based on the specified `backend`.
    - It is recommended to monitor training performance continuously to 
      ensure that increasing the batch size does not negatively impact model 
      convergence or generalization.
    
    See Also
    --------
    get_batch_size : Main function that determines the initial batch size.
    _analyze_dataset : Helper function used for dataset analysis.
    _prepare_tf_dataset : Helper function used for preparing TensorFlow datasets.
    
    References
    ----------
    .. [1] Smith, L. N. (2018). "A disciplined approach to neural 
          network hyperparameters: Part 1 - learning rate, batch size, 
          momentum, and weight decay." arXiv preprint arXiv:1803.09820.
    """
    # Adjust batch size for PyTorch backend
    if backend.lower() == 'torch':
        try:
            import torch
        except ImportError:
            raise DependencyError(TORCH_DEP_EMSG) 
        if torch.cuda.is_available():
            accuracy = performance_metrics.get('accuracy', 0)
            if accuracy >= accuracy_threshold:
                new_batch_size = current_batch_size * scale_factor
                if verbose:
                    print(
                        f"Performance improved (accuracy: {accuracy}). "
                        f"Increasing batch size to {new_batch_size}."
                    )
                return new_batch_size
            else:
                if verbose:
                    print(
                        f"Performance not improved (accuracy: {accuracy}). "
                        f"Keeping batch size at {current_batch_size}."
                    )
                return current_batch_size
        else:
            if default_size is not None:
                if verbose:
                    print(f"Using fallback batch size: {default_size}")
                return default_size
            else:
                if verbose:
                    print(
                        "CUDA not available. Keeping batch size at "
                        f"{current_batch_size}."
                    )
                return current_batch_size

    # Adjust batch size for TensorFlow backend
    elif backend.lower() == 'tensorflow':
        try:
            import tensorflow as tf
        except ImportError:
            raise DependencyError(TF_DEP_EMSG) 
    
        if tf.config.list_physical_devices('GPU'):
            accuracy = performance_metrics.get('accuracy', 0)
            if accuracy >= accuracy_threshold:
                new_batch_size = current_batch_size * scale_factor
                if verbose:
                    print(
                        f"Performance improved (accuracy: {accuracy}). "
                        f"Increasing batch size to {new_batch_size}."
                    )
                return new_batch_size
            else:
                if verbose:
                    print(
                        f"Performance not improved (accuracy: {accuracy}). "
                        f"Keeping batch size at {current_batch_size}."
                    )
                return current_batch_size
        else:
            if default_size is not None:
                if verbose:
                    print(f"Using fallback batch size: {default_size}")
                return default_size
            else:
                if verbose:
                    print(
                        "No GPU available for TensorFlow. Keeping batch size at "
                        f"{current_batch_size}."
                    )
                return current_batch_size

    # Fallback strategy for unsupported backends
    else:
        if default_size is not None:
            if verbose:
                print(f"Using fallback batch size: {default_size}")
            return default_size
        else:
            if verbose:
                print(
                    "No backend strategy available. Keeping batch size at "
                    f"{current_batch_size}."
                )
            return current_batch_size

def save_dataframes(
    *data: Union[DataFrame, Any],
    file_name_prefix: str = 'data',
    output_format: str = 'excel',
    sep: str = ',',
    start_index: int = 1
    ) -> None:
    """
    Saves multiple dataframes to Excel or CSV files, with each dataframe in a
    separate file.
    
    The files are named using a specified prefix and an index.

    Parameters
    ----------
    *data : Union[pd.DataFrame, Any]
        Variable number of arguments, where each argument is a dataframe or
        data that can be converted to a dataframe.
    file_name_prefix : str, optional
        Prefix for the output file names. Default is 'data'.
    output_format : str, optional
        Output format of the files. Can be 'excel' or 'csv'. Default is 'excel'.
    sep : str, optional
        Separator character for CSV output. Default is ','.
    start_index : int, optional
        Starting index for numbering the output files. Default is 1.

    Examples
    --------
    >>> from gofast.utils.mlutils import save_dataframes
    >>> df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> df2 = pd.DataFrame({'C': [5, 6], 'D': [7, 8]})
    >>> save_dataframes(df1, df2, file_name_prefix='mydata', output_format='csv')
    # This will create 'mydata_1.csv' for df1 and 'mydata_2.csv' for df2

    >>> save_dataframes(df1, output_format='excel', file_name_prefix='test')
    # This will create 'test_1.xlsx' containing df1
    """
    for index, df in enumerate(data, start=start_index):
        # Ensure the argument is a DataFrame
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)

        # Determine the file name
        file_name = f"{file_name_prefix}_{index}"
        if output_format == 'csv':
            df.to_csv(f"{file_name}.csv", sep=sep, index=False)
        elif output_format == 'excel':
            with pd.ExcelWriter(f"{file_name}.xlsx") as writer:
                df.to_excel(writer, index=False)
        else:
            raise ValueError("Unsupported output format. Choose 'excel' or 'csv'.")


@EnsureFileExists(file_param="tgz_file") 
def fetch_tgz_in(
    tgz_file: str, 
    filename: str, 
    savefile: str = 'tgz', 
    rename_outfile: Optional[str] = None
) -> str:
    """
    Extracts a specified file from a tar archive and saves it to a given directory.

    Parameters
    ----------
    tgz_file : str
        The full path to the tar file.
    filename : str
        The specific file to extract from the tar archive.
    savefile : str, optional
        Directory to save the extracted file, by default 'tgz'.
    rename_outfile : str, optional
        New name for the extracted file, if renaming is desired.

    Returns
    -------
    str
        The full path to the extracted (and possibly renamed) file.

    Raises
    ------
    FileNotFoundError
        If the specified `tgz_file` or `filename` within the tar archive does not exist.

    Examples
    --------
    >>> from gofast.utils.mlutils import fetch_tgz
    >>> fetch_tgz('data/__tar.tgz/fmain.bagciv.data.tar.gz', 'dataset.csv',
    ...           'extracted', rename_outfile='main.bagciv.data.csv')
    """
    tgz_path = Path(tgz_file)
    save_path = Path(savefile)
    save_path.mkdir(parents=True, exist_ok=True)

    if not tgz_path.is_file():
        raise FileNotFoundError(f"Source {tgz_file!r} is not a valid file.")

    with tarfile.open(tgz_path) as tar:
        member = next((m for m in tar.getmembers() if m.name.endswith(filename)), None)
        if member:
            tar.extract(member, path=save_path)
            extracted_file_path = save_path / member.name
            final_file_path = save_path / (rename_outfile if rename_outfile else filename)
            if extracted_file_path != final_file_path:
                extracted_file_path.rename(final_file_path)
                if extracted_file_path.parent != save_path:
                    shutil.rmtree(extracted_file_path.parent, ignore_errors=True)
        else:
            raise FileNotFoundError(f"File {filename} not found in {tgz_file}.")

    print(f"--> '{final_file_path}' was successfully extracted from '{tgz_path.name}' "
          f"and saved to '{save_path}'.")
    return str(final_file_path)


@EnsureFileExists(file_param="data_url") 
def fetch_tgz(
    data_url: str,
    tgz_filename: str,
    data_path: Optional[str] = None,
    show_progress: bool = False
) -> None:
    """
    Fetches and extracts a 'tgz' file from a specified URL, optionally into 
    a target directory.

    If `data_path` is provided and does not exist, it is created. If `data_path`
    is not provided, a default directory named 'tgz_data' in the current working 
    directory is used and created if necessary.

    Parameters
    ----------
    data_url : str
        The URL where the .tgz file is located.
    tgz_filename : str
        The filename of the .tgz file to download.
    data_path : Optional[str], optional
        The absolute path to the directory where the .tgz file will be extracted.
        If None, uses a directory named ``'tgz_data'`` in the current working 
        directory.
    show_progress : bool, optional
        If True, displays a progress bar during the file download. Default is False.

    Examples
    --------
    >>> from gofast.utils.mlutils import fetch_tgz
    >>> fetch_tgz(
    ...     data_url="http://example.com/data",
    ...     tgz_filename="data.tgz",
    ...     show_progress=True
    ... )

    >>> fetch_tgz(
    ...     data_url="http://example.com/data",
    ...     tgz_filename="data.tgz",
    ...     data_path="/path/to/custom/data",
    ...     show_progress=True
    ... )

    Note
    ----
    The function requires `tqdm` library for showing the progress bar. Ensure
    that `tqdm` is installed if `show_progress` is set to True.
    """
    # Use a default data directory if none is provided
    data_path = data_path or os.path.join(os.getcwd(), 'tgz_data')
    
    if not os.path.isdir(data_path):
        os.makedirs(data_path, exist_ok=True)

    tgz_path = os.path.join(data_path, tgz_filename)
    
    # Define a simple progress function, if needed
    def _progress(block_num, block_size, total_size):
        if show_progress:
            if tqdm is None:
                raise ImportError("`tqdm` library is required for progress output.")
            progress = tqdm(total=total_size, unit='iB', unit_scale=True, ascii=True, 
                            ncols= 100) 
            progress.n = block_num * block_size
            progress.last_print_n = progress.n
            progress.update()

    # Download the .tgz file
    urllib.request.urlretrieve(
        data_url, tgz_path, _progress if show_progress else None)

    # Extract the .tgz file
    with tarfile.open(tgz_path) as data_tgz:
        data_tgz.extractall(path=data_path)
    
    if show_progress:
        print("Download and extraction complete.")

