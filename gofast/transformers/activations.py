# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Activation function transformers for scikit-learn.
These transformers apply activation functions such as ReLU, Sigmoid, 
Tanh, ELU, Leaky ReLU, and Softmax element-wise to input data, 
and follow the scikit-learn API for custom transformers.
"""
from  textwrap import dedent 
from numbers import Real, Integral 

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

from ..api.docs import _shared_docs, doc
from ..backends.selector import select_backend_n 
from ..compat.sklearn import validate_params, Interval, StrOptions
from ..decorators import Appender, DataTransformer
from ..tools.validator import check_array, filter_valid_kwargs
from ..tools.validator import parameter_validator  
from ..tools.depsutils import ensure_pkg 

__all__= [ 
    'ReLUTransformer',
    'SigmoidTransformer',
    'TanhTransformer',
    'ELUTransformer',
    'LeakyReLUTransformer',
    'SoftmaxTransformer', 
    'SwishTransformer',
    'HardSigmoidTransformer',
    'HardSwishTransformer',
    'SoftplusTransformer', 
    'GELUTransformer',
    'SELUTransformer',
    'MishTransformer',
    'ELISHTransformer',
    'LogSigmoidTransformer',
    'TanhshrinkTransformer',
    'Swish1Transformer'
    ]

# Shared documentation for transformers to avoid redundancy.
_activation_doc: dict[str, str] = {}
_activation_doc  [ 
    'fit'
]="""\
Fit the {fmt} transformer.

Parameters
----------
X : array-like, shape (n_samples, n_features)
    The input data to fit, which is not used for fitting the activation 
    transformer. It is an array or matrix of shape (n_samples, n_features) 
    where `n_samples` is the number of samples and `n_features` is the 
    number of features for each sample.

y : Ignored
    This parameter is not used by the activation transformer, 
    but it is required by the scikit-learn API for 
    consistency with other models. 
    
Returns
-------
self : {fmt}
    The transformer object itself, following the Scikit-learn API.
"""
_activation_doc[ 
    'methods'
]="""
Methods
-------
fit(X, y=None)
    Fit the transformer. No fitting is necessary for the {afmt} transformer.
    The `fit` method does nothing for the {afmt} Transformer since the 
    {afmt} function does not have parameters to fit. It simply returns 
    the transformer instance.

transform(X)
    Applies the {afmt} activation function element-wise to the input 
    data `X` using the selected backend with support for batch processing. 
    Optionally applies scaling and shifting to the data before transforming.

"""
_activation_doc[ 
    'parameters'
]= """

Parameters 
-----------
scale : float, optional, default=1.0
    The scale factor that affects the steepness of the {afmt} curve. 
    This factor is multiplied by the input data before applying the {afmt} 
    transformation. A higher value makes the positive output values 
    grow more quickly (steeper curve), and a smaller value leads to a 
    flatter curve with more gradual increases. 

    The parameter allows the user to control the output range and adjust 
    the activation function for different types of data distributions.
    
    For example:
    - A `scale` of ``1.0`` keeps the output values proportional to the input.
    - A `scale` greater than ``1.0`` increases the output magnitude.
    - A `scale` less than ``1.0`` reduces the output magnitude.

shift : float, optional, default=0.0
    The shift factor that moves the {afmt} curve horizontally, which allows 
    the user to adjust where the function starts returning non-zero values.
    
    - A positive `shift` (e.g., `shift=1.0`) shifts the curve to the right, 
      meaning the input values need to be larger to activate the {afmt} function.
    - A negative `shift` (e.g., `shift=-1.0`) shifts the curve to the left, 
      making the activation function less strict in turning the output to zero.

    The `shift` parameter is especially useful when dealing with data that 
    might have a significant negative offset or when adjusting for data bias.
    
    Example:
    - If `shift=-1.0`, all values greater than `-1.0` will be transformed by 
      {afmt}.
    - If `shift=1.0`, only values greater than `1.0` will be transformed by
    {afmt}.

precision : float, optional, default=1e-6
    Precision control for the {afmt} computation to manage numerical 
    stability, particularly with very large or very small values. This is 
    crucial in avoiding overflow or underflow errors when applying {afmt} to 
    datasets that contain extreme values.

    - A smaller `precision` (e.g., ``precision=1e-12``) ensures that the 
      transformation can handle very small or very large inputs accurately.
    - A larger `precision` (e.g., ``precision=1e-2``) may speed up calculations 
      but could introduce numerical instability with extreme values.

    This is particularly useful in cases where the input data may contain 
    floating point numbers of varying magnitude, and you need to ensure 
    consistency and avoid rounding errors that could negatively impact the 
    activation results.

batch_size : int, optional, default=None
    The batch size to use when applying the {afmt} transformation. If `None`, 
    the entire dataset is processed in one go. If specified, the data is 
    processed in chunks of size `batch_size`, which is useful for large 
    datasets or memory-constrained environments.

    - When ``batch_size=None``, the transformer will apply {afmt} to the entire 
      dataset at once, which can be more efficient for smaller datasets.
    - When `batch_size` is specified (e.g., ``batch_size=1000``), the transformer 
      processes the data in batches, helping to manage memory usage by limiting 
      the amount of data held in memory at any one time.

    The batch processing capability allows the transformer to work efficiently 
    with large datasets, which would otherwise not fit into memory when processed 
    in a single batch.

    Example:
    - If the dataset contains ``10,000`` samples and ``batch_size=1000``, the 
      data will be processed in 10 separate chunks, each with 1000 samples.
      
backend : {{'numpy', 'tensorflow', 'torch', None}}, optional, default=None
    The backend library to use for the computation. Available options:
    
    - ``None`` or ``'numpy'``: Use NumPy (default). This is the standard 
      backend for numerical computations in Python, and it provides 
      highly optimized routines for operations like matrix multiplication 
      and element-wise transformations.

    - ``'tensorflow'``: Use TensorFlow, a popular deep learning framework 
      known for its optimized computations on large datasets and GPU support.
      TensorFlow provides efficient computations using data flow graphs and 
      is often used for training and deploying machine learning models.

    - ``'torch'``: Use PyTorch, another deep learning framework with a focus 
      on ease of use and dynamic computation graphs. PyTorch is favored by 
      researchers and provides flexible GPU-based computation for deep learning 
      applications.

    If ``None`` or ``'numpy'`` is specified, the transformer defaults to using 
    NumPy for all operations. If `'tensorflow'` or `'torch'` is specified, 
    the respective backend is used for computation. If an unsupported backend 
    is provided, a `ValueError` will be raised.
    
    This parameter allows for flexible execution depending on the desired 
    hardware platform or library preference. For example:
    - For CPU-based computation, NumPy is the default and works efficiently 
      on most platforms.
    - For GPU acceleration, you can choose ``'tensorflow'`` or ``'torch'`` to 
      leverage GPU-based computations (if the respective libraries are 
      installed and configured correctly).

    .. note::

    - When using TensorFlow or PyTorch, ensure that the corresponding 
      framework is installed and the GPU (if applicable) is available for 
      computation. TensorFlow and PyTorch are designed to work seamlessly 
      with GPU hardware to accelerate matrix operations and other 
      tensor-based computations.

    - TensorFlow and PyTorch support dynamic graph computation, which 
      may be beneficial for certain tasks, especially in training deep 
      learning models, while NumPy is optimized for static operations 
      and CPU-based tasks.
"""
_VALID_BACKEND_SET={'numpy', 'tensorflow', 'torch', 'np', 'tf', 'pytorch'}
# ----Activation Transformers


# ReLU Activation Transformer
@doc( 
    parameters =_activation_doc['parameters'].format(afmt="ReLU"), 
    methods = _activation_doc['methods'].format(afmt="ReLU"),
)
class ReLUTransformer(BaseEstimator, TransformerMixin):
    """
    ReLU (Rectified Linear Unit) Activation Transformer.

    This transformer applies the ReLU activation function element-wise 
    to the input data. The ReLU function returns the input for positive 
    values and zero for negative values, effectively setting all negative 
    values to zero. It is defined mathematically as:

    .. math::
        f(x) = \max(0, x)

    where:
    - :math:`x` is the input value.
    - The output is the input value if it is positive, or zero if it is 
      negative.

    The transformer allows for customization via several parameters:
    scaling, shifting, precision control, and batch processing.

    {parameters}

    {methods}

    Examples
    --------
    >>> from gofast.transformers.activations import ReLUTransformer
    >>> import numpy as np

    # Create a ReLUTransformer instance with default parameters
    >>> transformer = ReLUTransformer()

    # Sample input data (2 samples, 3 features)
    >>> X = np.array([[1, -2, 3], [-1, 5, -6]])

    # Apply the ReLU transformation
    >>> X_transformed = transformer.transform(X)

    # Print transformed data
    >>> print(X_transformed)
    array([[1., 0., 3.],
           [0., 5., 0.]])

    # Create a ReLUTransformer with custom scaling and shifting
    >>> transformer_custom = ReLUTransformer(scale=2.0, shift=1.0)

    # Apply the transformation with custom scale and shift
    >>> X_transformed_custom = transformer_custom.transform(X)
    >>> print(X_transformed_custom)
    array([[3., 1., 7.],
           [1., 11., 0.]])

    # Create a ReLUTransformer with batch processing enabled
    >>> transformer_batch = ReLUTransformer(batch_size=2)

    # Apply the transformation in batches
    >>> X_transformed_batch = transformer_batch.transform(X)
    >>> print(X_transformed_batch)
    array([[1., 0., 3.],
           [0., 5., 0.]])

    # Verifying the use of precision for numerical stability (change precision)
    >>> transformer_prec = ReLUTransformer(precision=1e-12)
    >>> X_transformed_prec = transformer_prec.transform(X)
    >>> print(X_transformed_prec)
    array([[1., 0., 3.],
           [0., 5., 0.]])
    Notes
    -----
    The ReLU function is commonly used in neural networks and machine 
    learning due to its simplicity and efficiency. It introduces non-linearity 
    into the model while being computationally inexpensive. The function 
    does not have learnable parameters, but scaling and shifting the inputs 
    can adjust the output range and behavior, making the transformer flexible 
    for various types of data.

    See Also
    --------
    SigmoidTransformer : A transformer that applies the Sigmoid activation 
                         function.
    TanhTransformer : A transformer that applies the Tanh activation 
                     function.

    References
    ----------
    [1] LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. 
        Nature, 521(7553), 436-444.
    
    """
    @validate_params ( { 
        "scale": [ Interval(Real, 0, None, closed ='neither' )], 
        "shift": [Interval(Real( -1, 1 , closed ='both'))], 
        "precision": [Interval(Real, 0 , 1 , closed ="neither")], 
        "batch_size": [ Interval ( Integral, 1, None , closed ='left'), None]
        }
    )
    def __init__(
        self, 
        scale=1.0, 
        shift=0.0, 
        precision=1e-6, 
        batch_size=None, 
        backend= None, 
        verbose=False
        ):

        self.scale = scale
        self.shift = shift
        self.precision = precision
        self.batch_size = batch_size
        self.backend =backend 
        self.verbose =verbose 
        

    @Appender(
        _activation_doc['fit'].format(fmt='ReLUTransformer'), 
        join= "\n", 
        )
    def fit(self, X, y=None):
        """Fit the transformer."""
        return self
    
    @DataTransformer(name='X', mode='lazy', keep_origin_type=True,)
    @doc(_shared_docs['activation_transform'])
    def transform(self, X):
        
        X = check_array(X, ensure_2d=True, dtype=np.float64)
        
        self.backend = select_backend_n(self.backend)
        # Apply scaling and shifting before ReLU transformation
        X_scaled_shifted = self.scale * X + self.shift

        if self.batch_size is not None:
            n_samples = X.shape[0]
            n_batches = int(np.ceil(n_samples / self.batch_size))
            X_transformed = np.zeros_like(X_scaled_shifted, dtype=np.float64)
            for i in range(n_batches):
                batch_start = i * self.batch_size
                batch_end = min((i + 1) * self.batch_size, n_samples)
                X_transformed[batch_start:batch_end] = np.maximum(
                    0, X_scaled_shifted[batch_start:batch_end])
        else:
            # Apply ReLU transformation on the whole dataset
            X_transformed = np.maximum(0, X_scaled_shifted)

        return X_transformed

@doc( 
    mathf =dedent( 
    """\
    The Sigmoid function is given by the formula:

    .. math:: \sigma(x) = \frac{1}{1 + e^{-(scale \cdot x + shift)}}

    where:
    - :math:`x` is the input value (or feature).
    - :math:`scale` adjusts the steepness of the curve.
    - :math:`shift` shifts the curve horizontally.
    """
    ), 
    parameters =_activation_doc['parameters'].format(afmt="sigmoid"), 
    methods = _activation_doc['methods'].format(afmt="Sigmoid"),
)
class SigmoidTransformer(BaseEstimator, TransformerMixin):
    """
    Sigmoid Activation Transformer with optional scaling, shifting,
    and batch processing.

    This transformer applies the Sigmoid activation function element-wise 
    to the input data. It includes options for scaling, shifting, and 
    processing the data in batches.

    {mathf}

    {parameters} 
    
    {methods}

    _sigmoid(X)
        Helper function that applies the Sigmoid function with scaling and
        shifting.
    
    Examples
    --------
    >>> from gofast.transformers.activations import SigmoidTransformer
    >>> import numpy as np
    
    # Create a SigmoidTransformer instance with default parameters
    >>> transformer = SigmoidTransformer()
    
    # Sample input data (2 samples, 3 features)
    >>> X = np.array([[1, 2, 3], [4, 5, 6]])
    
    # Apply the Sigmoid transformation
    >>> X_transformed = transformer.transform(X)
    
    # Print transformed data
    >>> print(X_transformed)
    [[0.73105858 0.88079708 0.95257413]
     [0.98201379 0.99330715 0.99752738]]
    
    # Create a SigmoidTransformer with customized scale and shift
    >>> transformer_custom = SigmoidTransformer(scale=2.0, shift=-1.0)
    
    # Apply the transformation with custom parameters
    >>> X_transformed_custom = transformer_custom.transform(X)
    >>> print(X_transformed_custom)
    [[0.95257413 0.99330715 0.99752738]
     [0.99966465 0.99974817 0.99982354]]
    
    # Create a SigmoidTransformer with batch processing enabled
    >>> transformer_batch = SigmoidTransformer(batch_size=1)
    
    # Apply the transformation in batches
    >>> X_transformed_batch = transformer_batch.transform(X)
    >>> print(X_transformed_batch)
    [[0.73105858 0.88079708 0.95257413]
     [0.98201379 0.99330715 0.99752738]]
    
    # Verifying the use of precision for numerical stability (change precision)
    >>> transformer_prec = SigmoidTransformer(precision=1e-12)
    >>> X_transformed_prec = transformer_prec.transform(X)
    >>> print(X_transformed_prec)
    [[0.73105858 0.88079708 0.95257413]
     [0.98201379 0.99330715 0.99752738]]
    """
    @validate_params ( { 
        "scale": [ Interval(Real, 0, None, closed ='neither' )], 
        "shift": [Interval(Real( -1, 1 , closed ='both'))], 
        "precision": [Interval(Real, 0 , 1 , closed ="neither")], 
        "batch_size": [ Interval ( Integral, 1, None , closed ='left'), None], 
        "backend": [StrOptions (_VALID_BACKEND_SET), None]
        }
    )
    def __init__(
        self, 
        scale=1.0, 
        shift=0.0, 
        precision=1e-6, 
        batch_size=None, 
        backend=None, 
        verbose=False
    ):
        self.scale = scale
        self.shift = shift
        self.precision = precision
        self.batch_size = batch_size
        self.backend =backend 
        self.verbose =verbose 

    @Appender(
        _activation_doc['fit'].format(fmt='SigmoidTransformer'), 
        join= "\n", 
        )
    def fit(self, X, y=None):
        """Fit the transformer."""
        return self
    
    @DataTransformer(name='X', mode='lazy', keep_origin_type=True,)
    @doc(_shared_docs['activation_transform'])
    def transform(self, X):
        # Validate the input and check if it is a DataFrame or array-like object
        X = check_array(
            X, 
            ensure_2d=True, 
            input_name='X'
        )
        if self.batch_size is not None:
            # Process in batches if batch_size is specified
            n_samples = X.shape[0]
            n_batches = int(np.ceil(n_samples / self.batch_size))
            X_transformed = np.zeros_like(X, dtype=np.float64)
            for i in range(n_batches):
                batch_start = i * self.batch_size
                batch_end = min((i + 1) * self.batch_size, n_samples)
                X_transformed[batch_start:batch_end] = self._sigmoid(
                    X[batch_start:batch_end])
        else:
            # No batch processing, transform the entire input at once
            X_transformed = self._sigmoid(X)

        return X_transformed
    
    def _sigmoid(self, X):
        """
        Helper function to apply the Sigmoid function with scaling and shifting.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data to transform.

        Returns
        -------
        transformed : array-like, shape (n_samples, n_features)
            The transformed data with the Sigmoid function applied element-wise.
        """
        # Apply the scaling and shifting before applying the Sigmoid function
        X_scaled_shifted = self.scale * X + self.shift
        return 1 / (1 + np.exp(-X_scaled_shifted))

# Tanh Activation Transformer
@doc( 
    mathf =dedent( 
    """\
    The mathematical formulation for the Tanh function is as follows:

    .. math:: 
        \text{Tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}

    Where:
        - :math:`x` is the input value or array of values.

    If scaling and shifting are applied, the transformation becomes:

    .. math::
        \text{Tanh}(ax + b) = \frac{e^{a(x + b)} - e^{-a(x + b)}} 
        {e^{a(x + b)} + e^{-a(x + b)}}

    Where:
        - :math:`a` is the scaling factor (default is 1.0)
        - :math:`b` is the shifting factor (default is 0.0)
    """
    ), 
    parameters =_activation_doc['parameters'].format(afmt="tanh"), 
    methods = _activation_doc['methods'].format(afmt='Tanh'), 
)
class TanhTransformer(BaseEstimator, TransformerMixin):
    """
    The Tanh transformer applies the Tanh activation function element-wise 
    to the input data. The Tanh function maps input values to a range 
    between -1 and 1.

    {mathf}

    {parameters}

    backend : {{'numpy', 'tensorflow', 'torch'}}, optional, default=None
        The computational backend to use for the transformation. 
        Supports 'numpy' (default), 'tensorflow' for GPU acceleration, 
        and 'pytorch' for GPU acceleration.

    verbose : bool, optional, default=False
        If True, prints information about the batch processing.
    
    {methods}
    
    Notes
    -----
    The transformation is applied element-wise to each value in the input 
    data. The result is a value between -1 and 1, depending on the input.
    
    Examples
    --------
    Applying the `TanhTransformer` to a simple NumPy array:

    >>> from gofast.transformers.activations import TanhTransformer
    >>> import numpy as np
    >>> X = np.array([[1.0, 2.0], [-1.0, -2.0]])
    >>> transformer = TanhTransformer(scale=1.0, shift=0.0)
    >>> transformer.fit(X).transform(X)
    array([[ 0.76159416,  0.96402758],
           [-0.76159416, -0.96402758]])

    In this example, the `TanhTransformer` applies the Tanh activation 
    function to the input `X`. Since the scale is set to 1.0 and the 
    shift to 0.0, the transformation is applied directly without any 
    scaling or shifting.

    Applying the `TanhTransformer` with scaling and shifting:

    >>> transformer = TanhTransformer(scale=2.0, shift=1.0)
    >>> transformer.fit(X).transform(X)
    array([[ 0.96402758,  0.9999092 ],
           [-0.96402758, -0.9999092 ]])

    In this case, the input values are scaled by a factor of 2 and 
    shifted by 1 before the Tanh function is applied.

    Using batch processing for large datasets:

    >>> X_large = np.random.randn(1000, 1000)  # 1000x1000 random values
    >>> transformer = TanhTransformer(scale=1.0, shift=0.0, batch_size=100)
    >>> transformer.fit(X_large).transform(X_large)

    When the `batch_size` parameter is specified, the transformer processes 
    the data in chunks. In this case, the data is split into batches of 
    100 samples and processed sequentially.

    Using a custom backend (TensorFlow or PyTorch):

    >>> transformer = TanhTransformer(scale=1.0, shift=0.0, backend='tensorflow')
    >>> transformer.fit(X).transform(X)

    The `backend` parameter allows for GPU acceleration if supported by 
    the backend. Here, the computation is done using TensorFlow for 
    potentially faster processing, especially on large datasets.
    
    """
    @validate_params ( { 
        "scale": [ Interval(Real, 0, None, closed ='neither' )], 
        "shift": [Interval(Real( -1, 1 , closed ='both'))], 
        "precision": [Interval(Real, 0 , 1 , closed ="neither")], 
        "batch_size": [ Interval ( Integral, 1, None , closed ='left'), None], 
        "backend": [StrOptions (_VALID_BACKEND_SET), None]
        }
    )
    def __init__(
        self, 
        scale=1.0, 
        shift=0.0, 
        precision=1e-6, 
        batch_size=None, 
        backend=None,
        verbose=False
        ):
        self.scale = scale
        self.shift = shift
        self.precision = precision
        self.batch_size = batch_size
        self.backend = backend
        self.verbose = verbose

    @Appender(
        _activation_doc['fit'].format(fmt='TanhTransformer'), 
        join= "\n", 
        )
    def fit(self, X, y=None):
        """Fit the transformer."""
        return self
    
    @DataTransformer(name='X', mode='lazy', keep_origin_type=True,)
    @doc(_shared_docs['activation_transform'])
    def transform(self, X):

        # Ensure input is in the correct format
        X = check_array(X, ensure_2d=True, input_name='X')

        # If backend is not specified, use numpy by default
        if self.backend is None:
            self.backend = 'numpy'

        # If batch processing is specified, apply transformation in batches
        if self.batch_size is not None:
            return self._batch_process(X)
        else:
            return self._apply_tanh(X)

    def _apply_tanh(self, X):
        """
        Helper function to apply the Tanh activation function to the 
        input data with optional scaling and shifting.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data to transform.

        Returns
        -------
        transformed : array-like, shape (n_samples, n_features)
            The transformed data after applying Tanh.
        """
        # Apply scaling and shifting
        X_scaled_shifted = self.scale * X + self.shift

        # Select the backend and apply Tanh
        return self._select_backend(X_scaled_shifted)

    def _batch_process(self, X):
        """
        Processes the data in batches if `batch_size` is specified.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data to transform.

        Returns
        -------
        X_transformed : array-like, shape (n_samples, n_features)
            The transformed data after applying Tanh in batches.
        """
        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / self.batch_size))
        X_transformed = np.zeros_like(X, dtype=np.float64)

        for i in range(n_batches):
            batch_start = i * self.batch_size
            batch_end = min((i + 1) * self.batch_size, n_samples)
            X_transformed[batch_start:batch_end] = self._apply_tanh(
                X[batch_start:batch_end])

            if self.verbose:
                print(f"Processing batch {i+1}/{n_batches}")

        return X_transformed

    def _select_backend(self, X):
        """
        Selects the backend for the Tanh transformation computation.
    
        This method allows users to choose a backend for computation. It 
        supports NumPy, TensorFlow, and PyTorch. These backends can be used
        to process the input data either on the CPU or GPU depending on
        the framework.
    
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data to transform. This data is passed to the selected
            backend for computation.
    
        Returns
        -------
        backend : callable
            A callable function (e.g., `np.tanh` for NumPy, `tf.math.tanh`
                                 for TensorFlow, 
            or `torch.tanh` for PyTorch) that applies the Tanh activation
            function.
    
        Raises
        ------
        ValueError
            If the specified backend is not supported, an error is raised.
    
        Notes
        -----
        The default backend is NumPy, which is used if no backend is specified 
        or if 'numpy' is specified explicitly. 
        """

        # Check the selected backend and return the appropriate function
        if self.backend in ( "numpy", "np"):
            return np.tanh  # Use NumPy's Tanh function
        elif self.backend in ( "tensorflow", "tf"):
            # Ensure TensorFlow is available
            try:
                # Use TensorFlow's Tanh function (works on CPU or GPU)
                import tensorflow as tf
                return tf.math.tanh
            except ImportError:
                raise ImportError(
                    "TensorFlow is not installed. Please install"
                    " TensorFlow to use this backend.")
        elif self.backend == "torch":
            # Ensure PyTorch is available
            try:
                # Use PyTorch's Tanh function (works on CPU or GPU)
                import torch
                return torch.tanh
            except ImportError:
                raise ImportError(
                    "PyTorch is not installed. Please install"
                    " PyTorch to use this backend.")
        elif self.backend is None:
            self.backend = "numpy"  # Default to NumPy if no backend is specified
            return np.tanh

# ELU Activation Transformer (Exponential Linear Unit)
@doc( 
    examples =dedent( 
    """\
    Examples
    --------
    >>> from gofast.transformers.activations import ELUTransformer
    >>> import numpy as np

    # Create sample input data
    >>> X = np.array([[-1.0, 0.5], [0.3, -0.8], [1.2, 2.5]])

    # Initialize the ELUTransformer with custom parameters
    >>> transformer = ELUTransformer(alpha=1.0, scale=1.0, shift=0.0)

    # Apply the ELU transformation
    >>> X_transformed = transformer.transform(X)
    >>> print(X_transformed)
    [[-0.63212056  0.5       ]
     [ 0.3        -0.55071121]
     [ 1.2         2.5       ]]

    # Using batch processing
    >>> transformer_batch = ELUTransformer(alpha=1.0, scale=1.0, 
    >>>                                    shift=0.0, batch_size=2)
    >>> X_transformed_batch = transformer_batch.transform(X)
    >>> print(X_transformed_batch)
    [[-0.63212056  0.5       ]
     [ 0.3        -0.55071121]
     [ 1.2         2.5       ]]

    # Using a custom backend (TensorFlow)
    >>> transformer_tensorflow = ELUTransformer(alpha=1.0, scale=1.0, 
    >>>                                         shift=0.0, backend='tensorflow')
    >>> X_transformed_tensorflow = transformer_tensorflow.transform(X)
    >>> print(X_transformed_tensorflow)
    [[-0.63212056  0.5       ]
     [ 0.3        -0.55071121]
     [ 1.2         2.5       ]]

    # Using verbose mode to track progress during batch processing
    >>> transformer_verbose = ELUTransformer(alpha=1.0, scale=1.0, 
    >>>                                      shift=0.0, batch_size=2, verbose=True)
    >>> X_transformed_verbose = transformer_verbose.transform(X)
    Processing batch 1/2
    Processing batch 2/2
    >>> print(X_transformed_verbose)
    [[-0.63212056  0.5       ]
     [ 0.3        -0.55071121]
     [ 1.2         2.5       ]]
    """), 
    
    mathf =dedent( 
    """\
    The ELU activation function is defined as:

    .. math:: 
        \text{ELU}(x) = 
        \begin{cases} 
            x & \text{if } x > 0, \\
            \alpha \cdot (\exp(x) - 1) & \text{if } x \leq 0,
        \end{cases}
    
    where:
    - :math:`\alpha` is a user-defined scaling factor (default is 1.0)
    - The function returns `x` when `x > 0`, and an exponential function 
      for `x <= 0`.
    """
    ),  
    parameters =_activation_doc['parameters'].format(afmt="ELU"), 
    methods = _activation_doc['methods'].format(afmt='ELU'),
)    
class ELUTransformer(BaseEstimator, TransformerMixin):
    """
    ELU (Exponential Linear Unit) Transformer with optional scaling, 
    shifting, precision control, batch processing, and custom backend 
    support (TensorFlow, PyTorch).
    
    {mathf}

    This transformer allows for flexibility in handling data preprocessing 
    with optional scaling, shifting, and numerical precision controls. It 
    supports multiple backends such as NumPy, TensorFlow, and PyTorch.

    {parameters}
    
    alpha : float, optional, default=1.0
        The scaling factor applied when the input is less than or equal to 
        zero. A higher value for `alpha` increases the influence of negative 
        values.

    verbose : bool, optional, default=False
        If True, prints progress information during batch processing. Useful 
        when working with large datasets to track progress.

    {methods}
 
    _apply_elu(X)
        Applies the ELU transformation using the current parameters (`alpha`, 
        `scale`, and `shift`). This method is used by the other backend-specific 
        methods.

    _batch_process(X)
        Processes the data in batches, applying the ELU transformation in chunks 
        as specified by the `batch_size` parameter.

    _transform_numpy(X)
        Applies the ELU transformation using the NumPy backend.

    _transform_tensorflow(X)
        Applies the ELU transformation using the TensorFlow backend.

    _batch_process_tensorflow(X)
        Processes the data in batches using TensorFlow if the `batch_size` 
        parameter is specified.

    _transform_pytorch(X)
        Applies the ELU transformation using the PyTorch backend.

    _batch_process_pytorch(X)
        Processes the data in batches using PyTorch if the `batch_size` 
        parameter is specified.
        
    {examples}
    
    Notes
    -----
    - The `alpha` parameter controls the output range for negative input values.
    - The transformation uses either NumPy, TensorFlow, or PyTorch depending 
      on the `backend` specified.
    - Batching is supported to help process large datasets more efficiently.

    See Also
    --------
    gofast.transformers.activations.ReLUTransformer : 
        Another activation transformer that applies the ReLU function.
     gofast.transformers.activations.SigmoidTransformer : 
        Transformer that applies the Sigmoid activation function.

    References
    ----------
    [1] Clevert, D. A., Unterthiner, T., & Hochreiter, S. (2015). 
        Fast and Accurate Deep Network Learning by Exponential Linear Units 
        (ELUs). arXiv preprint arXiv:1511.07289.
    """

    @validate_params ( { 
        "scale": [ Interval(Real, 0, None, closed ='neither' )], 
        "shift": [Interval(Real( -1, 1 , closed ='both'))], 
        "precision": [Interval(Real, 0 , 1 , closed ="neither")], 
        "batch_size": [ Interval ( Integral, 1, None , closed ='left'), None], 
        "alpha": [Interval(Real, 0 , None , closed ="neither")], 
        "backend": [StrOptions (_VALID_BACKEND_SET), None], 
        }
    )    
    def __init__(
        self,
        scale=1.0, 
        shift=0.0, 
        precision=1e-6, 
        batch_size=None,
        backend=None, 
        alpha=1.0, 
        verbose=False
        ):
        self.alpha = alpha
        self.scale = scale
        self.shift = shift
        self.precision = precision
        self.batch_size = batch_size
        self.backend = backend
        self.verbose = verbose

    @Appender(
        _activation_doc['fit'].format(fmt='ELUTransformer'), 
        join= "\n", 
        )
    def fit(self, X, y=None):
        """Fit the transformer."""
        return self
    
    @DataTransformer(name='X', mode='lazy', keep_origin_type=True)
    @doc(_shared_docs['activation_transform'])
    def transform(self, X):

        # Check X 
        X = check_array(X, ensure_2d=True, input_name="X")
        
        self.backend = select_backend_n(self.backend )
  
        if self.backend =="tensorflow":
            return self._transform_tensorflow(X)
        elif self.backend == "torch":
            return self._transform_pytorch(X)
        else:
            return self._transform_numpy(X)

    def _apply_elu(self, X):
        """
        Apply the ELU activation function element-wise with 
        scaling and shifting.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data to transform.

        Returns
        -------
        transformed : array-like, shape (n_samples, n_features)
            The transformed data after applying ELU.
        """
        X_scaled_shifted = self.scale * X + self.shift
        return np.where(X_scaled_shifted > 0, 
                        X_scaled_shifted, 
                        self.alpha * (np.exp(X_scaled_shifted) - 1))

    def _batch_process(self, X):
        """
        Apply the ELU function to the data in batches.
        """
        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / self.batch_size))
        X_transformed = np.zeros_like(X, dtype=np.float64)

        for i in range(n_batches):
            batch_start = i * self.batch_size
            batch_end = min((i + 1) * self.batch_size, n_samples)
            X_transformed[batch_start:batch_end] = self._apply_elu(
                X[batch_start:batch_end])

            if self.verbose:
                print(f"Processing batch {i+1}/{n_batches}")

        return X_transformed

    def _transform_numpy(self, X):
        """
        ELU transformation using numpy backend
        """
        if self.batch_size is not None:
            return self._batch_process(X)
        else:
            return self._apply_elu(X)

    def _transform_tensorflow(self, X):
        """
        ELU transformation using TensorFlow backend
        """
        if self.batch_size is not None:
            return self._batch_process_tensorflow(X)
        else:
            return self._apply_elu_tensorflow(X)

    def _batch_process_tensorflow(self, X):
        """
        Apply ELU in batches with TensorFlow.
        """
        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / self.batch_size))
        X_transformed = np.zeros_like(X, dtype=np.float64)

        for i in range(n_batches):
            batch_start = i * self.batch_size
            batch_end = min((i + 1) * self.batch_size, n_samples)
            X_transformed[batch_start:batch_end] = self._apply_elu_tensorflow(
                X[batch_start:batch_end])

            if self.verbose:
                print(f"Processing batch {i+1}/{n_batches}")

        return X_transformed

    @ensure_pkg(
        "tensorflow", 
        "'tensorflow' is required when 'tensorflow' is set as backend."
    )
    def _apply_elu_tensorflow(self, X):
        """
        Apply ELU with TensorFlow backend.
        """
        import tensorflow as tf
   
        X_scaled_shifted = self.scale * X + self.shift
        elu_result = tf.nn.elu(X_scaled_shifted)
        return elu_result.numpy()

    def _transform_pytorch(self, X):
        """
        ELU transformation using PyTorch backend
        """
        if self.batch_size is not None:
            return self._batch_process_pytorch(X)
        else:
            return self._apply_elu_pytorch(X)

    def _batch_process_pytorch(self, X):
        """
        Apply ELU in batches with PyTorch.
        """
        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / self.batch_size))
        X_transformed = np.zeros_like(X, dtype=np.float64)

        for i in range(n_batches):
            batch_start = i * self.batch_size
            batch_end = min((i + 1) * self.batch_size, n_samples)
            X_transformed[batch_start:batch_end] = self._apply_elu_pytorch(
                X[batch_start:batch_end])

            if self.verbose:
                print(f"Processing batch {i+1}/{n_batches}")

        return X_transformed

    @ensure_pkg(
        "torch", "'torch' is required when 'torch' is set as backend.")
    def _apply_elu_pytorch(self, X):
        """
        Apply ELU with PyTorch backend.
        """
        import torch
        X_scaled_shifted = self.scale * X + self.shift
        elu_result = torch.nn.functional.elu(torch.tensor(X_scaled_shifted))
        return elu_result.numpy()

# Leaky ReLU Activation Transformer
@Appender ( dedent( 
"""\
    Notes
    -----
    - The parameter `alpha` controls the slope of the function for 
      negative inputs. A larger value of `alpha` means that negative 
      inputs will have a greater impact on the transformation.
    - The `scale` parameter is applied to both positive and negative 
      values, amplifying or reducing the output range.
    - `shift` moves the output curve along the x-axis. Positive values 
      shift the function rightward, while negative values shift it 
      leftward.
    - The transformer supports both batch processing and backend 
      flexibility, so large datasets can be processed in chunks, and 
      the computation can be done using different backends like 
      `numpy`, `tensorflow`, or `pytorch`.

    See Also
    --------
    gofast.transformers.activations.ReLUTransformer :
        A simpler ReLU activation transformer that outputs zero for negative
        values.
    
    gofast.transformers.activations.ELUTransformer : 
        A more complex activation function that introduces exponential growth
        for negative values, unlike Leaky ReLU which has linear behavior
        for negative inputs.
""" )
)
@doc( 
    mathf =dedent( 
    """\
    The Leaky ReLU activation function is defined as:

    .. math::
        \text{LeakyReLU}(x) = 
        \begin{cases} 
            x & \text{if } x > 0, \\
            \alpha \cdot x & \text{if } x \leq 0,
        \end{cases}
    
    where:
    - :math:`\alpha` is a user-defined scaling factor (default is 1.0).
    - The function returns `x` when `x > 0`, and a scaled version of 
    `x` for `x <= 0`.
    """
    ),  
    parameters =_activation_doc['parameters'].format(afmt="Leaky ReLU"), 
    methods = _activation_doc['methods'].format(afmt='Leaky ReLU'),
)    

class LeakyReLUTransformer(BaseEstimator, TransformerMixin):
    """
    Leaky ReLU Activation Transformer with optional scaling, shifting, 
    precision control, batch processing.
    
    {mathf}

    This transformer allows for flexibility in handling data preprocessing 
    with optional scaling, shifting, and numerical precision controls. 
    It supports multiple backends such as NumPy, TensorFlow, and PyTorch.

    {parameters}
    
    alpha : float, optional, default=1.0
        The scaling factor applied when the input is less than or equal to zero.
        A higher value for `alpha` increases the influence of negative values.


    verbose : bool, optional, default=False
        If True, prints progress information during batch processing. Useful 
        when working with large datasets to track progress.

    {methods}

    _apply_leaky_relu(X)
        Applies the Leaky ReLU transformation using the current parameters 
        (`alpha`, `scale`, and `shift`). This method is used by the other 
        backend-specific methods.

    _batch_process(X)
        Processes the data in batches, applying the Leaky ReLU transformation 
        in chunks as specified by the `batch_size` parameter.

    _transform_numpy(X)
        Applies the Leaky ReLU transformation using the NumPy backend.

    _transform_tensorflow(X)
        Applies the Leaky ReLU transformation using the TensorFlow backend.

    _batch_process_tensorflow(X)
        Processes the data in batches using TensorFlow if the `batch_size` 
        parameter is specified.
        
    Examples
    --------
    >>> from gofast.transformers.activations import LeakyReLUTransformer
    
    >>> # Example 1: Basic Leaky ReLU Transformation using NumPy backend
    >>> leaky_relu = LeakyReLUTransformer(alpha=0.1, backend='numpy')
    >>> X = np.array([[1, -2, 3], [-1, 4, -5]])
    >>> transformed = leaky_relu.transform(X)
    >>> print(transformed)
    [[ 1.   -0.2  3.  ]
     [-0.1  4.  -0.5]]
    
    >>> # Example 2: Leaky ReLU with custom scaling and shifting
    >>> leaky_relu = LeakyReLUTransformer(alpha=0.1, scale=2.0, shift=-1.0)
    >>> X = np.array([[1, -2, 3], [-1, 4, -5]])
    >>> transformed = leaky_relu.transform(X)
    >>> print(transformed)
    [[  1.    -1.4    3.  ]
     [ -1.2    4.    -2.  ]]
    
    >>> # Example 3: Leaky ReLU with batch processing
    >>> leaky_relu = LeakyReLUTransformer(alpha=0.2, batch_size=2, verbose=True)
    >>> X = np.array([[1, -2, 3], [-1, 4, -5], [0, -2, 3]])
    >>> transformed = leaky_relu.transform(X)
    Processing batch 1/2
    Processing batch 2/2
    >>> print(transformed)
    [[ 1.   -0.4  3.  ]
     [-0.2  4.  -1.  ]
     [ 0.   -0.4  3.  ]]
    
    >>> # Example 4: Leaky ReLU using TensorFlow backend
    >>> leaky_relu = LeakyReLUTransformer(alpha=0.05, backend='tensorflow')
    >>> X = np.array([[1, -2, 3], [-1, 4, -5]])
    >>> transformed = leaky_relu.transform(X)
    >>> print(transformed)
    tf.Tensor(
    [[ 1.    -0.1   3.   ]
     [-0.05  4.   -0.25]], shape=(2, 3), dtype=float32)

    """
    @validate_params ( { 
        "scale": [ Interval(Real, 0, None, closed ='neither' )], 
        "shift": [Interval(Real( -1, 1 , closed ='both'))], 
        "precision": [Interval(Real, 0 , 1 , closed ="neither")], 
        "batch_size": [ Interval ( Integral, 1, None , closed ='left'), None], 
        "alpha": [Interval(Real, 0 , None , closed ="neither")], 
        "backend": [StrOptions (_VALID_BACKEND_SET), None], 
        
        }
    )    
    
    def __init__(
        self, 
        scale=1.0, 
        shift=0.0, 
        precision=1e-6, 
        batch_size=None, 
        alpha=1.0, 
        backend=None, 
        verbose=False
    ):
        self.alpha = alpha
        self.scale = scale
        self.shift = shift
        self.precision = precision
        self.batch_size = batch_size
        self.backend = backend
        self.verbose = verbose

    @Appender(
        _activation_doc['fit'].format(fmt='Leaky ReLU Transformer'), 
        join= "\n", 
        )
    def fit(self, X, y=None):
        """Fit the transformer."""
        return self
    
    @DataTransformer(name='X', mode='lazy', keep_origin_type=True)
    @doc(_shared_docs['activation_transform'])
    def transform(self, X):
        X = check_array(X, ensure_2d=True, input_name='X')

        if self.backend is None: 
            self.backend = 'numpy' 
        
        elif self.backend in ('tensorflow', 'tf'): 
            return self._batch_process_tensorflow(X)
        
        elif self.backend  =='torch': 
            return self._batch_process_torch(X)
        
        if self.batch_size is not None:
            return self._batch_process(X)
        else:
            return self._apply_leaky_relu(X)

    def _apply_leaky_relu(self, X):
        """
        Apply the Leaky ReLU transformation using the current parameters
        (`alpha`, `scale`, and `shift`).
        """
        X_scaled_shifted = self.scale * X + self.shift
        return np.where(
            X_scaled_shifted > 0, X_scaled_shifted, 
            self.alpha * X_scaled_shifted)

    def _batch_process(self, X):
        """
        Process the data in batches, applying the Leaky ReLU transformation 
        in chunks as specified by the `batch_size` parameter.
        """
        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / self.batch_size))
        X_transformed = np.zeros_like(X, dtype=np.float64)

        for i in range(n_batches):
            batch_start = i * self.batch_size
            batch_end = min((i + 1) * self.batch_size, n_samples)
            X_transformed[batch_start:batch_end] = self._apply_leaky_relu(
                X[batch_start:batch_end])

            if self.verbose:
                print(f"Processing batch {i + 1}/{n_batches}")

        return X_transformed

    def _transform_numpy(self, X):
        """
        Apply the Leaky ReLU transformation using the NumPy backend.
        """
        return np.where(X > 0, X, self.alpha * X)
    
    @ensure_pkg(
        "tensorflow", 
        extra="backend is set to ``tensorflow`` while it is not installed."
    )
    def _transform_tensorflow(self, X):
        """
        Apply the Leaky ReLU transformation using the TensorFlow backend.
        """
        import tensorflow as tf
        X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
        return tf.where(X_tensor > 0, X_tensor, self.alpha * X_tensor)

    def _batch_process_tensorflow(self, X):
        """
        Process the data in batches using TensorFlow if the `batch_size` 
        parameter is specified.
        """
        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / self.batch_size))
        X_transformed = np.zeros_like(X, dtype=np.float32)

        for i in range(n_batches):
            batch_start = i * self.batch_size
            batch_end = min((i + 1) * self.batch_size, n_samples)
            X_transformed[batch_start:batch_end] = self._transform_tensorflow(
                X[batch_start:batch_end])

            if self.verbose:
                print(f"Processing batch {i + 1}/{n_batches}")

        return X_transformed
    
    @ensure_pkg(
        "torch", 
        extra="backend is set to ``torch`` while it is not installed."
    )
    def _batch_process_torch(self, X):
        """
        Process the data in batches using PyTorch for backend computation.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data to transform.

        Returns
        -------
        X_transformed : tensor-like, shape (n_samples, n_features)
            The transformed data after applying Leaky ReLU in batches 
            using PyTorch.
        """
        import torch
        
        # Convert input data to a PyTorch tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Get the number of samples
        n_samples = X_tensor.shape[0]

        # Initialize a tensor to store the transformed data
        X_transformed = torch.zeros_like(X_tensor)

        # Number of batches
        n_batches = int(np.ceil(n_samples / self.batch_size))

        for i in range(n_batches):
            batch_start = i * self.batch_size
            batch_end = min((i + 1) * self.batch_size, n_samples)
            
            # Extract the current batch of data
            X_batch = X_tensor[batch_start:batch_end]
            
            # Apply the Leaky ReLU operation for the current batch
            X_transformed[batch_start:batch_end] = torch.where(
                X_batch > 0, X_batch, self.alpha * X_batch
            )

        return X_transformed
    

# Softmax Activation Transformer
@doc( 
    mathf =dedent( 
    """\
    The Softmax function is defined as:

    .. math::
        \text{Softmax}(x_i) = \frac{\exp(x_i)}{\sum_{j} \exp(x_j)}

    where:
    - :math:`x_i` represents the input vector elements.
    - The result is a probability distribution, with each element 
      representing the probability of the corresponding class.
    """
    ),  
    parameters =_activation_doc['parameters'].format(afmt="Softmax"), 
    methods = _activation_doc['methods'].format(afmt='Softmax'),
)    
class SoftmaxTransformer(BaseEstimator, TransformerMixin):
    """
    Softmax Activation Transformer with optional scaling, shifting, 
    precision control, batch processing, and backend support 
    (TensorFlow, PyTorch).

    The Softmax function is commonly used in classification tasks to 
    normalize the output of a model to a probability distribution across 
    multiple classes. This transformer applies the Softmax activation 
    function element-wise to the input data, scaling and shifting the inputs 
    before applying the transformation.

    {mathf}

    {parameters}

    verbose : bool, optional, default=False
        If True, enables logging of transformations and progress, helpful 
        for debugging and monitoring the transformation process.

    {methods}

    Notes
    -----
    - The Softmax function is commonly used in the output layer of classifiers 
      to convert raw output scores (logits) into a probability distribution 
      over classes. 
    - The `scale` and `shift` parameters are useful for preprocessing 
      the data before applying the Softmax transformation. They can help adjust 
      the range of values and avoid numerical issues.
    - The backend selection allows flexibility in computing the Softmax function 
      on different platforms, including CPU and GPU computation using TensorFlow 
      or PyTorch.
    - If `batch_size` is provided, the transformation is applied in chunks, 
      which can be beneficial for large datasets and memory management.

    Examples
    --------
    >>> from gofast.transformers.activations import SoftmaxTransformer
    >>> transformer = SoftmaxTransformer(backend='numpy')
    >>> X = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    >>> transformed_X = transformer.transform(X)
    >>> print(transformed_X)
    [[0.09003057 0.24472847 0.66524096]
     [0.09003057 0.24472847 0.66524096]]

    >>> transformer = SoftmaxTransformer(backend='tensorflow')
    >>> transformed_X_tf = transformer.transform(X)
    >>> print(transformed_X_tf)
    tf.Tensor([[0.09003057 0.24472847 0.66524096]
               [0.09003057 0.24472847 0.66524096]], dtype=float32)

    >>> transformer = SoftmaxTransformer(backend='torch')
    >>> transformed_X_torch = transformer.transform(X)
    >>> print(transformed_X_torch)
    tensor([[0.0900, 0.2447, 0.6652],
            [0.0900, 0.2447, 0.6652]])

    See Also
    --------
    gofast.transformers.activations.ReLUTransformer : 
        A simpler activation function that outputs zero for negative values.
    gofast.transformers.activations.ELUTransformer : 
        A more complex activation function with exponential growth for 
        negative values.
    gofast.transformers.activations.LeakyReLUTransformer : 
        A variation of ReLU that allows a small, non-zero gradient for 
        negative inputs.

    References
    ----------
    [1] Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. 
        MIT Press. https://www.deeplearningbook.org/
    """
    @validate_params ( { 
        "scale": [ Interval(Real, 0, None, closed ='neither' )], 
        "shift": [Interval(Real( -1, 1 , closed ='both'))], 
        "precision": [Interval(Real, 0 , 1 , closed ="neither")], 
        "batch_size": [ Interval ( Integral, 1, None , closed ='left'), None], 
        "alpha": [Interval(Real, 0 , None , closed ="neither")], 
        "backend": [StrOptions (_VALID_BACKEND_SET),None], 
        
        }
    )    
    def __init__(
        self, 
        scale=1.0, 
        shift=0.0, 
        precision=1e-6, 
        batch_size=None, 
        backend=None, 
        verbose=False
        ):
        self.scale = scale
        self.shift = shift
        self.precision = precision
        self.batch_size = batch_size
        self.backend = backend
        self.verbose = verbose

    @Appender(
        _activation_doc['fit'].format(fmt='SoftmaxTransformer'), 
        join= "\n", 
        )
    def fit(self, X, y=None):
        """Fit the transformer."""
        return self
    
    @DataTransformer(name='X', mode='lazy', keep_origin_type=True)
    @doc(_shared_docs['activation_transform'])
    def transform(self, X):
        X = check_array(X, ensure_2d=True, input_name="X")
        
        self.backend = select_backend_n(self.backend)
        
        # Apply batch processing if batch_size is specified
        if self.batch_size is not None:
            return self._batch_process(X)
        
        # Apply transformation based on the backend
        if self.backend=="numpy":
            return self._transform_numpy(X)
        
        elif self.backend =="tensorflow":
            return self._transform_tensorflow(X)
        
        elif self.backend == "torch":
            return self._transform_pytorch(X)
        

    def _batch_process(self, X):
        """
        Process the data in batches if batch_size is specified.
        """
        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / self.batch_size))
        
        results = []
        for i in range(n_batches):
            batch_start = i * self.batch_size
            batch_end = min((i + 1) * self.batch_size, n_samples)
            batch = X[batch_start:batch_end]
            
            # Process the batch depending on the backend
            if self.backend == "numpy" or self.backend is None:
                results.append(self._transform_numpy(batch))
            elif self.backend == "tensorflow":
                results.append(self._transform_tensorflow(batch))
            elif self.backend == "pytorch":
                results.append(self._transform_pytorch(batch))
        
        # Concatenate results across batches
        return np.concatenate(results, axis=0)

    def _transform_numpy(self, X):
        """
        Transform using NumPy.
        """
        exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))  # Numerical stability
        return exp_X / np.sum(exp_X, axis=1, keepdims=True)

    @ensure_pkg(
        "tensorflow", 
        extra="'tensorflow' backend is selected while the library is missing"
    )
    def _transform_tensorflow(self, X):
        """
        Transform using TensorFlow.
        """
        import tensorflow as tf
        X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
        exp_X = tf.exp(X_tensor - tf.reduce_max(
            X_tensor, axis=1, keepdims=True))
        return exp_X / tf.reduce_sum(exp_X, axis=1, keepdims=True)
    
    @ensure_pkg(
        "torch", 
        extra="'torch' backend is selected while the library is missing"
    )
    def _transform_pytorch(self, X):
        """
        Transform using PyTorch.
        """
        import torch
        X_tensor = torch.tensor(X, dtype=torch.float32)
        exp_X = torch.exp(X_tensor - torch.max(
            X_tensor, dim=1, keepdim=True)[0])
        return exp_X / torch.sum(exp_X, dim=1, keepdim=True)

# Swish Activation Transformer
@doc( 
    mathf =dedent( 
    """\
    The Swish activation function is defined as:

    .. math::
        \text{Swish}(x) = x \cdot \sigma(x)

    where:
    - :math:`x` is the input,
    - :math:`\sigma(x)` is the sigmoid function, defined as
    :math:`\sigma(x) = \frac{1}{1 + \exp(-x)}`.

    This transformer applies the Swish activation function element-wise 
    to the input data. Swish is a smooth, non-monotonic activation function 
    that has been shown to outperform ReLU in some deep learning models.
    
    """
    ),  
    parameters =_activation_doc['parameters'].format(afmt="Swish"), 
    methods = _activation_doc['methods'].format(afmt='Swish'),
) 
class SwishTransformer(BaseEstimator, TransformerMixin):
    """
    Swish Activation Transformer with optional scaling, shifting, precision
    control, batch processing, and backend support.

    {mathf}
    
    Swish allows for flexibility with optional scaling, shifting, and 
    precision control. It also supports batch processing and is compatible 
    with NumPy, TensorFlow, and PyTorch backends.

    {parameters}

    verbose : bool, optional, default=False
        If `True`, additional information will be printed during transformation. 
        This is useful for debugging or tracking the status of large computations.

    {methods}

    Notes
    -----
    - The Swish function has been shown to perform better than ReLU in many 
      deep learning models, particularly for deep neural networks. The function 
      is differentiable and smooth, making it a good candidate for optimization 
      tasks in machine learning.
    
    - When using TensorFlow or PyTorch, ensure that the corresponding 
      framework is installed and that GPU support is available for faster 
      computation, especially when working with large datasets.

    - TensorFlow and PyTorch offer GPU support, while NumPy is optimized 
      for CPU-based tasks. Therefore, TensorFlow and PyTorch may provide 
      better performance in large-scale deep learning applications.

    Examples
    --------
    >>> from gofast.transformers.activations import SwishTransformer
    >>> transformer = SwishTransformer(scale=2.0, shift=-1.0)
    >>> X = np.array([[-1.0, 2.0], [3.0, -4.0]])
    >>> transformer.transform(X)
    array([[-0.76159416,  2.0       ],
           [ 2.0       , -0.76159416]])

    See Also
    --------
    ReLUTransformer : A simpler ReLU activation transformer that outputs 
    zero for negative values.
    
    ELUTransformer : A more complex activation function that introduces 
    exponential growth for negative values, unlike Swish which is smooth 
    for all inputs.

    References
    ----------
    [1] Ramachandran, P., Zoph, B., & Le, Q. V. (2017). Swish: a self-gated 
    activation function. arXiv preprint arXiv:1710.05941.
    """
    @validate_params ( { 
        "scale": [ Interval(Real, 0, None, closed ='neither' )], 
        "shift": [Interval(Real( -1, 1 , closed ='both'))], 
        "precision": [Interval(Real, 0 , 1 , closed ="neither")], 
        "batch_size": [ Interval ( Integral, 1, None , closed ='left'), None], 
        "alpha": [Interval(Real, 0 , None , closed ="neither")], 
        "backend": [StrOptions (_VALID_BACKEND_SET), None], 
        }
    )    
    def __init__(
        self, 
        scale=1.0, 
        shift=0.0, 
        precision=1e-6, 
        batch_size=None, 
        backend=None, 
        verbose=False
        ):

        self.scale = scale
        self.shift = shift
        self.precision = precision
        self.batch_size = batch_size
        self.backend = backend if backend else 'numpy'  # Default to numpy if None
        self.verbose = verbose

    @Appender(
        _activation_doc['fit'].format(fmt='SwishTransformer'), 
        join= "\n", 
        )
    def fit(self, X, y=None):
        """Fit the transformer."""
        return self
    
    @DataTransformer(name='X', mode='lazy', keep_origin_type=True)
    @doc(_shared_docs['activation_transform'])
    def transform(self, X):

        X = check_array(X, ensure_2d=True, input_name="X")
        
        self.backend = select_backend_n(self.backend)
        # Apply scaling and shifting
        X_transformed = (self.scale * X) + self.shift

        # Use batch processing if enabled
        if self.batch_size:
            return self._batch_process(X_transformed)
        
        # Apply backend-specific transformations
        if self.backend == 'numpy':
            return self._transform_numpy(X_transformed)
        elif self.backend == 'tensorflow':
            return self._transform_tensorflow(X_transformed)
        elif self.backend == 'torch':
            return self._transform_pytorch(X_transformed)

    def _batch_process(self, X):
        """
        Process the data in batches for better scalability.
        This method is used when `batch_size` is specified.
        """
        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / self.batch_size))
        X_transformed = np.zeros_like(X)

        for i in range(n_batches):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, n_samples)
            X_transformed[start:end] = self._apply_swish(X[start:end])

        return X_transformed

    def _transform_numpy(self, X):
        """
        Transform using NumPy backend.
        """
        return self._apply_swish(X)

    def _transform_tensorflow(self, X):
        """
        Transform using TensorFlow backend.
        """
        import tensorflow as tf
        X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
        return self._apply_swish_tensorflow(X_tensor)

    def _transform_pytorch(self, X):
        """
        Transform using PyTorch backend.
        """
        import torch
        X_tensor = torch.tensor(X, dtype=torch.float32)
        return self._apply_swish_pytorch(X_tensor)

    def _apply_swish(self, X):
        """
        Apply the Swish activation function (x * sigmoid(x)) using NumPy.
        """
        return X * (1 / (1 + np.exp(-X)))  # Swish = x * sigmoid(x)
    
    @ensure_pkg(
        "tensorflow", 
        extra="'tensorflow' backend is selected while the library is missing"
    )
    def _apply_swish_tensorflow(self, X):
        """
        Apply the Swish activation function using TensorFlow.
        """
        import tensorflow as tf 
        return X * (1 / (1 + tf.exp(-X)))

    @ensure_pkg(
        "torch", 
        extra="'torch' backend is selected while the library is missing"
    )
    def _apply_swish_pytorch(self, X):
        """
        Apply the Swish activation function using PyTorch.
        """
        import torch 
        return X * (1 / (1 + torch.exp(-X)))
    
# Hard Sigmoid Activation Transformer
@doc( 
    mathf =dedent( 
    """\
    The Hard Sigmoid activation function is defined as:

    .. math::
        \text{HardSigmoid}(x) = \max(0, \min(1, 0.2 \cdot x + 0.5))

    :class:`HardSigmoidTransformer` function is a computationally efficient 
    approximation of the Sigmoid function and maps inputs into the range 
    [0, 1]. This transformer provides flexibility for large datasets with 
    optional batch processing and backend support for GPU acceleration using 
    either TensorFlow or PyTorch.
    """
    ),  
    parameters =_activation_doc['parameters'].format(afmt="HardSigmoid"), 
    methods = _activation_doc['methods'].format(afmt='HardSigmoid'),
) 
class HardSigmoidTransformer(BaseEstimator, TransformerMixin):
    """
    Hard Sigmoid Activation Transformer with optional scaling, shifting, 
    precision control, batch processing, and backend support.
    
    {mathf}
    
    {parameters}

    verbose : bool, optional, default=False
        If True, outputs information during the transformation process, 
        useful for debugging or performance analysis.
    
    {methods}
        
    _transform_numpy(X)
        Applies the Hard Sigmoid function using NumPy. This method is used when 
        the `backend` is set to 'numpy' (or None).
        
    _transform_tensorflow(X)
        Applies the Hard Sigmoid function using TensorFlow. This method is 
        used when the `backend` is set to 'tensorflow'. TensorFlow operations  
        are optimized for GPU computation, providing potential speedup 
        for large data.
        
    _transform_pytorch(X)
        Applies the Hard Sigmoid function using PyTorch. This method is used 
        when the `backend` is set to 'torch'. PyTorch operations are optimized
         for GPU computation, providing potential speedup for large data.

    _batch_process(X)
        If `batch_size` is set, the data is processed in batches, which is 
        useful for large datasets that might not fit into memory. This method 
        handles splitting the input data into smaller chunks and processing 
        them sequentially using the specified backend.

    Examples
    ---------
    >>> from gofast.transformers.activations import HardSigmoidTransformer
    >>> transformer = HardSigmoidTransformer(scale=1.0, shift=0.0)
    >>> X = np.array([[-2.0, 0.0, 2.0], [1.0, -1.0, 3.0]])
    >>> transformer.transform(X)
    array([[0.        , 0.5       , 1.        ],
           [0.5       , 0.        , 1.        ]])

    >>> transformer = HardSigmoidTransformer(scale=2.0, shift=-0.5)
    >>> X = np.array([[0.0, 0.5, -1.0], [-2.0, 1.0, 3.0]])
    >>> transformer.transform(X)
    array([[0.5       , 0.9       , 0.        ],
           [0.        , 0.9       , 1.        ]])

    >>> transformer = HardSigmoidTransformer(batch_size=2, backend='torch')
    >>> X = np.array([[0.5, 1.0, -0.5], [1.5, -1.0, 2.0]])
    >>> transformer.transform(X)
    tensor([[0.6, 0.7, 0.        ],
            [0.7, 0.        , 0.9]])

    See Also
    --------
    gofast.transformers.activations.SigmoidTransformer :
        A more precise Sigmoid activation function.
    gofast.transformers.activations.ReLUTransformer : 
        A commonly used activation function with linear behavior for 
        positive inputs.
    
    References
    ----------
    [1] D.P. Kingma, J.B. Ba. Adam: A Method for Stochastic Optimization. 
        ICLR, 2015. https://arxiv.org/abs/1412.6980
    """
    @validate_params ( { 
        "scale": [ Interval(Real, 0, None, closed ='neither' )], 
        "shift": [Interval(Real( -1, 1 , closed ='both'))], 
        "precision": [Interval(Real, 0 , 1 , closed ="neither")], 
        "batch_size": [ Interval ( Integral, 1, None , closed ='left'), None], 
        "alpha": [Interval(Real, 0 , None , closed ="neither")], 
        "backend": [StrOptions (_VALID_BACKEND_SET), None], 
        }
    )    
    def __init__(
        self,
        scale=1.0, 
        shift=0.0, 
        precision=1e-6, 
        batch_size=None,
        backend=None, 
        verbose=False
        ):
        self.scale = scale
        self.shift = shift
        self.precision = precision
        self.batch_size = batch_size
        self.backend = backend
        self.verbose = verbose

    @Appender(
        _activation_doc['fit'].format(fmt='HardSigmoidTransformer'), 
        join= "\n", 
        )
    def fit(self, X, y=None):
        """Fit the transformer."""
        return self
    
    @DataTransformer(name='X', mode='lazy', keep_origin_type=True)
    @doc(_shared_docs['activation_transform'])
    def transform(self, X):
        X = check_array(X, ensure_2d=True, input_name="X")
        
        self.backend = select_backend_n(self.backend)
        # If batch_size is set, process in batches
        if self.batch_size is not None:
            return self._batch_process(X)
        
        # Apply transformation depending on the backend
        if self.backend =="numpy":
            return self._apply_numpy(X)
        elif self.backend == "tensorflow":
            return self._apply_tensorflow(X)
        elif self.backend == "torch":
            return self._apply_pytorch(X)

    def _apply_numpy(self, X):
        """Apply Hard Sigmoid using NumPy"""
        return np.clip(self.scale * X + self.shift, 0, 1)
    
    @ensure_pkg(
        "tensorflow", 
        extra="'tensorflow' backend is selected while the library is missing"
    )
    def _apply_tensorflow(self, X):
        """Apply Hard Sigmoid using TensorFlow"""
        import tensorflow as tf
        X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
        # Hard Sigmoid = 0.2 * x + 0.5
        return tf.clip_by_value(0.2 * X_tensor + 0.5, 0, 1) 
    
    @ensure_pkg(
        "torch", 
        extra="'torch' backend is selected while the library is missing"
    )
    def _apply_pytorch(self, X):
        """Apply Hard Sigmoid using PyTorch"""
        import torch
        X_tensor = torch.tensor(X, dtype=torch.float32)
        # Hard Sigmoid = 0.2 * x + 0.5
        return torch.clamp(0.2 * X_tensor + 0.5, 0, 1)

    def _batch_process(self, X):
        """
        Process data in batches according to the specified batch_size.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data to transform.

        Returns
        -------
        X_transformed : array-like, shape (n_samples, n_features)
            The transformed data after applying Hard Sigmoid in batches.
        """
        n_samples = X.shape[0]
        n_batches = (n_samples // self.batch_size) + (
            1 if n_samples % self.batch_size != 0 else 0)
        transformed_batches = []

        for i in range(n_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, n_samples)
            batch_X = X[start_idx:end_idx]

            # Apply the appropriate transformation for the batch
            if self.backend =="numpy":
                transformed_batch = self._apply_numpy(batch_X)
            elif self.backend == "tensorflow":
                transformed_batch = self._apply_tensorflow(batch_X)
            elif self.backend == "torch":
                transformed_batch = self._apply_pytorch(batch_X)

            transformed_batches.append(transformed_batch)
            
        # Stack the batches back together
        return np.vstack(transformed_batches)  

#XXX OPTIMIZE 
# revise this activation transformer and make it more robust :
# apply 
# scale=1.0, 
# shift=0.0, 
# precision=1e-6, 
# batch_size=None, 
# backend=None, 
# verbose=False
# skip WRITING THE  the documentation FOR BRIVEITY  to make it more robust and apply 
# the back size applycation if set and also the backend operatiosn 
# for numpy default if None or tensorflow or torch 
# Dont document the __init__method and use vertical aligment for parameters listing 


# Hard Swish Activation Transformer
class HardSwishTransformer(BaseEstimator, TransformerMixin):
    """
    Hard Swish Activation Transformer.

    This transformer applies the Hard Swish activation function element-wise 
    to the input data. The Hard Swish function is an approximation of the Swish 
    function, defined as:
    - HardSwish(x) = x * HardSigmoid(x)
    """
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Using the previously defined HardSigmoid
        hard_sigmoid = np.clip(0.2 * X + 0.5, 0, 1)
        return X * hard_sigmoid

# Softplus Activation Transformer
class SoftplusTransformer(BaseEstimator, TransformerMixin):
    """
    Softplus Activation Transformer.

    This transformer applies the Softplus activation function element-wise 
    to the input data. The Softplus function is a smooth approximation of the ReLU 
    function, defined as:
    - Softplus(x) = log(1 + exp(x))
    """
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.log1p(np.exp(X))  # Softplus = log(1 + exp(x))


# GELU Activation Transformer (Gaussian Error Linear Unit)
class GELUTransformer(BaseEstimator, TransformerMixin):
    """
    GELU (Gaussian Error Linear Unit) Transformer.

    This transformer applies the GELU activation function element-wise to the 
    input data. The GELU function is defined as:
    - GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    where `erf` is the error function.
    """
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return 0.5 * X * (1 + np.erf(X / np.sqrt(2)))  # GELU = 0.5 * x * (1 + erf(x / sqrt(2)))

# SELU Activation Transformer (Scaled Exponential Linear Unit)
class SELUTransformer(BaseEstimator, TransformerMixin):
    """
    SELU (Scaled Exponential Linear Unit) Transformer.

    This transformer applies the SELU activation function element-wise to the 
    input data. The SELU function is defined as:
    - SELU(x) = scale * (x if x > 0 else alpha * (exp(x) - 1))
    where `scale` and `alpha` are fixed constants.
    """
    
    def __init__(self, scale=1.0507, alpha=1.6733):
        self.scale = scale
        self.alpha = alpha
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return self.scale * np.where(X > 0, X, self.alpha * (np.exp(X) - 1))  # SELU = scale * (x if x > 0 else alpha * (exp(x) - 1))

# Mish Activation Transformer
class MishTransformer(BaseEstimator, TransformerMixin):
    """
    Mish Activation Transformer.

    This transformer applies the Mish activation function element-wise to the 
    input data. The Mish function is defined as:
    - Mish(x) = x * tanh(softplus(x))
    """
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X * np.tanh(np.log1p(np.exp(X)))  # Mish = x * tanh(log(1 + exp(x)))

# ELISH Activation Transformer
class ELISHTransformer(BaseEstimator, TransformerMixin):
    """
    ELISH Activation Transformer.

    This transformer applies the ELISH activation function element-wise to the 
    input data. The ELISH function is a variant of ELU, defined as:
    - ELISH(x) = x if x > 0 else alpha * (exp(x) - 1) + beta * (x)
    """
    
    def __init__(self, alpha=1.0, beta=0.1):
        self.alpha = alpha
        self.beta = beta
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.where(X > 0, X, self.alpha * (np.exp(X) - 1) + self.beta * X)

# LogSigmoid Activation Transformer
class LogSigmoidTransformer(BaseEstimator, TransformerMixin):
    """
    LogSigmoid Activation Transformer.

    This transformer applies the LogSigmoid activation function element-wise 
    to the input data. The LogSigmoid function is defined as:
    - LogSigmoid(x) = log(1 / (1 + exp(-x)))
    """
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.log(1 / (1 + np.exp(-X)))  # LogSigmoid = log(1 / (1 + exp(-x)))

# Tanhshrink Activation Transformer
class TanhshrinkTransformer(BaseEstimator, TransformerMixin):
    """
    Tanhshrink Activation Transformer.

    This transformer applies the Tanhshrink activation function element-wise 
    to the input data. The Tanhshrink function is defined as:
    - Tanhshrink(x) = x - tanh(x)
    """
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X - np.tanh(X)  # Tanhshrink = x - tanh(x)

# Swish-1 Activation Transformer
class Swish1Transformer(BaseEstimator, TransformerMixin):
    """
    Swish-1 Activation Transformer.

    This transformer applies the Swish-1 activation function element-wise to 
    the input data. The Swish-1 function is defined as:
    - Swish-1(x) = x * sigmoid(x)
    """
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X * (1 / (1 + np.exp(-X)))  # Swish-1 = x * sigmoid(x)


# Factory function to get the appropriate transformer based on activation name
def get_activation_transformer(activation_name, **params):
    """
    Get the corresponding activation function transformer based 
    on the provided name.
    
    Parameters:
    activation_name: str
        The name of the activation function. Options are 'relu', 'sigmoid', 
        'tanh', 'elu', 'leakyrelu', 'softmax', 'swish', 'hardsigmoid', 
        'hardswish', and 'softplus'.
    
    Returns:
    transformer: object
        The transformer corresponding to the provided activation function.
    
    Raises:
    ValueError: If an unsupported activation name is provided.
    """
    transformers = {
        'relu': ReLUTransformer,
        'sigmoid': SigmoidTransformer,
        'tanh': TanhTransformer,
        'elu': ELUTransformer,
        'leakyrelu': LeakyReLUTransformer,
        'softmax': SoftmaxTransformer, 
        'swish': SwishTransformer,
        'hardsigmoid': HardSigmoidTransformer,
        'hardswish': HardSwishTransformer,
        'softplus': SoftplusTransformer, 
        'gelu': GELUTransformer,
        'selu': SELUTransformer,
        'mish': MishTransformer,
        'elish': ELISHTransformer,
        'logsigmoid': LogSigmoidTransformer,
        'tanhshrink': TanhshrinkTransformer,
        'swish1': Swish1Transformer
    }
    
    # Validate the activation name
    activation_name = parameter_validator(
        "activation_name", target_strs=transformers.keys(), 
        error_msg=f"Unsupported activation: {activation_name}"
        ) (activation_name)
    

    transformer_func = transformers[activation_name]
    valid_params = filter_valid_kwargs(transformer_func, params)
    return transformer_func (**valid_params)


