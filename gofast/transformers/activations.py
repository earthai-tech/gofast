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
from ..compat.numpy import safe_erf 
from ..compat.sklearn import validate_params, Interval, StrOptions, Hidden
from ..core.utils import smart_format
from ..decorators import Appender, DataTransformer
from ..tools.validator import check_array, filter_valid_kwargs
from ..tools.validator import parameter_validator  


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
    'Swish1Transformer', 
    'get_activation_transformer'
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
        "shift": [Interval(Real, -1, 1 , closed ='both')], 
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
        backend=None, 
        verbose=0
    ):
        self.scale = scale
        self.shift = shift
        self.precision = precision
        self.batch_size = batch_size
        self.backend = backend
        self.verbose = verbose
        self.backend_name_ = None
        self.backend_ = None

    @Appender(
        _activation_doc['fit'].format(fmt='ReLUTransformer'), 
        join="\n"
    )
    def fit(self, X, y=None):
        """Fit the transformer."""
        self.backend_name_, self.backend_ = select_backend_n(
            self.backend, return_both=True
        )
        return self

    @DataTransformer(name='X', mode='lazy', keep_origin_type=True)
    @doc(_shared_docs['activation_transform'])
    def transform(self, X):
        if self.verbose >= 1:
            self._log(1, f"Using backend: {self.backend_name_}")

        X = self._validate_input(X)
        X_scaled_shifted = self.scale * X + self.shift

        if self.batch_size:
            return self._batch_process(X_scaled_shifted)

        return self._apply_transformation(X_scaled_shifted)

    def _validate_input(self, X):
        return check_array(X, ensure_2d=True, dtype=np.float64, input_name="X")

    def _apply_transformation(self, X):
        return self._apply_relu(X)

    def _batch_process(self, X):
        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / self.batch_size))
        transformed_batches = []

        for i in range(n_batches):
            batch_start = i * self.batch_size
            batch_end = min((i + 1) * self.batch_size, n_samples)
            batch_X = X[batch_start:batch_end]

            if self.verbose >= 2:
                self._log(2, ( 
                    f"Processing batch {i + 1}/{n_batches} "
                    f"(samples {batch_start} to {batch_end}).")
                    )

            transformed_batch = self._apply_relu(batch_X)

            if self.precision:
                transformed_batch = self._apply_precision(transformed_batch)

            transformed_batches.append(transformed_batch)

            if self.verbose >= 3:
                self._log(3, f"Batch {i + 1}-{n_batches} transformed.")

        return self._concatenate_batches(transformed_batches)

    def _get_relu_function(self):
        if self.backend_name_ == 'numpy':
            return self._relu_numpy
        elif self.backend_name_ == 'tensorflow':
            return self._relu_tensorflow
        elif self.backend_name_ == 'torch':
            return self._relu_pytorch
        else:
            raise ValueError(
                f"Unsupported backend: {self.backend_name_}. "
                "Supported backends are 'numpy', 'tensorflow', and 'torch'."
            )

    def _apply_relu(self, X):
        relu_func = self._get_relu_function()
        return relu_func(X)

    def _relu_numpy(self, X):
        return self.backend_.maximum(0, X)

    def _relu_tensorflow(self, X):
        return self.backend_.math.maximum(0, X)

    def _relu_pytorch(self, X):
        return self.backend_.relu(X)

    def _apply_precision(self, transformed_batch):
        if self.backend_name_ == 'numpy':
            return self.backend_.round(
                transformed_batch / self.precision
            ) * self.precision
        elif self.backend_name_ == 'tensorflow':
            return self.backend_.math.round(
                transformed_batch / self.precision
            ) * self.precision
        elif self.backend_name_ == 'torch':
            return self.backend_.round(
                transformed_batch / self.precision
            ) * self.precision
        else:
            raise ValueError(
                f"Unsupported backend for precision application: {self.backend_name_}."
            )

    def _concatenate_batches(self, transformed_batches):
        if self.backend_name_ == 'numpy':
            return self.backend_.vstack(transformed_batches)
        elif self.backend_name_ == 'tensorflow':
            return self.backend_.concat(transformed_batches, axis=0)
        elif self.backend_name_ == 'torch':
            return self.backend_.cat(transformed_batches, dim=0)
        else:
            raise ValueError(
                f"Unsupported backend for concatenation: {self.backend_name_}."
            )

    def _log(self, level, message):
        if self.verbose >= level:
            print(message)

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
        "shift": [Interval(Real, -1, 1 , closed ='both')], 
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
        verbose=0
    ):
        self.scale = scale
        self.shift = shift
        self.precision = precision
        self.batch_size = batch_size
        self.backend = backend
        self.verbose = verbose

    @Appender(
        _activation_doc['fit'].format(fmt='SigmoidTransformer'), 
        join="\n"
    )
    def fit(self, X, y=None):
        """Fit the transformer."""
        self.backend_name_, self.backend_ = select_backend_n(
            self.backend, return_both=True
        )
        return self

    @DataTransformer(name='X', mode='lazy', keep_origin_type=True)
    @doc(_shared_docs['activation_transform'])
    def transform(self, X):
        if self.verbose >= 1:
            self._log(1, f"Using backend: {self.backend_name_}")

        X = self._validate_input(X)
        X_transformed = self.scale * X + self.shift

        if self.batch_size:
            return self._batch_process(X_transformed)

        return self._apply_transformation(X_transformed)

    def _validate_input(self, X):
        return check_array(X, ensure_2d=True, input_name="X")

    def _apply_transformation(self, X):
        return self._apply_sigmoid(X)

    def _batch_process(self, X):
        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / self.batch_size))
        transformed_batches = []

        for i in range(n_batches):
            batch_start = i * self.batch_size
            batch_end = min((i + 1) * self.batch_size, n_samples)
            batch_X = X[batch_start:batch_end]

            if self.verbose >= 2:
                self._log(2, 
                          ( 
                              f"Processing batch {i + 1}/{n_batches} "
                              f"(samples {batch_start} to {batch_end}).")
                          )

            transformed_batch = self._apply_sigmoid(batch_X)
            transformed_batch = self.scale * transformed_batch + self.shift

            if self.precision:
                transformed_batch = self._apply_precision(transformed_batch)

            transformed_batches.append(transformed_batch)

            if self.verbose >= 3:
                self._log(3, f"Batch {i + 1}-{n_batches} transformed.")

        return self._concatenate_batches(transformed_batches)

    def _get_sigmoid_function(self):
        if self.backend_name_ == 'numpy':
            return self._sigmoid_numpy
        elif self.backend_name_ == 'tensorflow':
            return self._sigmoid_tensorflow
        elif self.backend_name_ == 'torch':
            return self._sigmoid_pytorch
        else:
            raise ValueError(
                f"Unsupported backend: {self.backend_name_}. "
                "Supported backends are 'numpy', 'tensorflow', and 'torch'."
            )

    def _apply_sigmoid(self, X):
        sigmoid_func = self._get_sigmoid_function()
        return sigmoid_func(X)

    def _sigmoid_numpy(self, X):
        return 1 / (1 + np.exp(-X))

    def _sigmoid_tensorflow(self, X):
        return self.backend_.math.sigmoid(X)

    def _sigmoid_pytorch(self, X):
        return self.backend_.sigmoid(X)

    def _apply_precision(self, transformed_batch):
        if self.backend_name_ == 'numpy':
            return self.backend_.round(
                transformed_batch / self.precision
            ) * self.precision
        elif self.backend_name_ == 'tensorflow':
            return self.backend_.math.round(
                transformed_batch / self.precision
            ) * self.precision
        elif self.backend_name_ == 'torch':
            return self.backend_.round(
                transformed_batch / self.precision
            ) * self.precision
        else:
            raise ValueError(
                f"Unsupported backend for precision application: {self.backend_name_}."
            )

    def _concatenate_batches(self, transformed_batches):
        if self.backend_name_ == 'numpy':
            return self.backend_.vstack(transformed_batches)
        elif self.backend_name_ == 'tensorflow':
            return self.backend_.concat(transformed_batches, axis=0)
        elif self.backend_name_ == 'torch':
            return self.backend_.cat(transformed_batches, dim=0)
        else:
            raise ValueError(
                f"Unsupported backend for concatenation: {self.backend_name_}."
            )

    def _log(self, level, message):
        if self.verbose >= level:
            print(message)

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
        "shift": [Interval(Real, -1, 1 , closed ='both')], 
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
        verbose=0
    ):
        self.scale = scale
        self.shift = shift
        self.precision = precision
        self.batch_size = batch_size
        self.backend = backend
        self.verbose = verbose


    @Appender(
        _activation_doc['fit'].format(fmt='TanhTransformer'), 
        join="\n"
    )
    def fit(self, X, y=None):
        """Fit the transformer."""
        self.backend_name_, self.backend_ = select_backend_n(
            self.backend, return_both=True
        )
        return self

    @DataTransformer(name='X', mode='lazy', keep_origin_type=True)
    @doc(_shared_docs['activation_transform'])
    def transform(self, X):
        if self.verbose >= 1:
            self._log(1, f"Using backend: {self.backend_name_}")

        X = self._validate_input(X)
        X_transformed = self.scale * X + self.shift

        if self.batch_size:
            return self._batch_process(X_transformed)

        return self._apply_transformation(X_transformed)

    def _validate_input(self, X):
        return check_array(X, ensure_2d=True, input_name="X")

    def _apply_transformation(self, X):
        return self._apply_tanh(X)

    def _batch_process(self, X):
        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / self.batch_size))
        transformed_batches = []

        for i in range(n_batches):
            batch_start = i * self.batch_size
            batch_end = min((i + 1) * self.batch_size, n_samples)
            batch_X = X[batch_start:batch_end]

            if self.verbose >= 2:
                self._log(2, ( 
                    f"Processing batch {i + 1}/{n_batches}"
                    f" (samples {batch_start} to {batch_end}).")
                    )

            transformed_batch = self._apply_tanh(batch_X)
            transformed_batch = self.scale * transformed_batch + self.shift

            if self.precision:
                transformed_batch = self._apply_precision(transformed_batch)

            transformed_batches.append(transformed_batch)

            if self.verbose >= 3:
                self._log(3, f"Batch {i + 1}-{n_batches} transformed.")

        return self._concatenate_batches(transformed_batches)

    def _get_tanh_function(self):
        if self.backend_name_ == 'numpy':
            return self._tanh_numpy
        elif self.backend_name_ == 'tensorflow':
            return self._tanh_tensorflow
        elif self.backend_name_ == 'torch':
            return self._tanh_pytorch
        else:
            raise ValueError(
                f"Unsupported backend: {self.backend_name_}. "
                "Supported backends are 'numpy', 'tensorflow', and 'torch'."
            )

    def _apply_tanh(self, X):
        tanh_func = self._get_tanh_function()
        return tanh_func(X)

    def _tanh_numpy(self, X):
        return np.tanh(X)

    def _tanh_tensorflow(self, X):
        return self.backend_.math.tanh(X)

    def _tanh_pytorch(self, X):
        return self.backend_.tanh(X)

    def _apply_precision(self, transformed_batch):
        if self.backend_name_ == 'numpy':
            return self.backend_.round(
                transformed_batch / self.precision
            ) * self.precision
        elif self.backend_name_ == 'tensorflow':
            return self.backend_.math.round(
                transformed_batch / self.precision
            ) * self.precision
        elif self.backend_name_ == 'torch':
            return self.backend_.round(
                transformed_batch / self.precision
            ) * self.precision
        else:
            raise ValueError(
                f"Unsupported backend for precision application: {self.backend_name_}."
            )

    def _concatenate_batches(self, transformed_batches):
        if self.backend_name_ == 'numpy':
            return self.backend_.vstack(transformed_batches)
        elif self.backend_name_ == 'tensorflow':
            return self.backend_.concat(transformed_batches, axis=0)
        elif self.backend_name_ == 'torch':
            return self.backend_.cat(transformed_batches, dim=0)
        else:
            raise ValueError(
                f"Unsupported backend for concatenation: {self.backend_name_}."
            )

    def _log(self, level, message):
        if self.verbose >= level:
            print(message)

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
        "shift": [Interval(Real, -1, 1 , closed ='both')], 
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
        verbose=0
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
        join="\n"
    )
    def fit(self, X, y=None):
        """Fit the transformer."""
        self.backend_name_, self.backend_ = select_backend_n(
            self.backend, return_both=True
        )
        return self

    @DataTransformer(name='X', mode='lazy', keep_origin_type=True)
    @doc(_shared_docs['activation_transform'])
    def transform(self, X):
        if self.verbose >= 1:
            self._log(1, f"Using backend: {self.backend_name_}")

        X = self._validate_input(X)
        X_transformed = self.scale * X + self.shift

        if self.batch_size:
            return self._batch_process(X_transformed)

        return self._apply_transformation(X_transformed)

    def _validate_input(self, X):
        return check_array(X, ensure_2d=True, input_name="X")

    def _apply_transformation(self, X):
        return self._apply_elu(X)

    def _batch_process(self, X):
        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / self.batch_size))
        transformed_batches = []

        for i in range(n_batches):
            batch_start = i * self.batch_size
            batch_end = min((i + 1) * self.batch_size, n_samples)
            batch_X = X[batch_start:batch_end]

            if self.verbose >= 2:
                self._log(2, ( 
                    f"Processing batch {i + 1}/{n_batches}"
                    f" (samples {batch_start} to {batch_end}).")
                    )

            transformed_batch = self._apply_elu(batch_X)
            transformed_batch = self.scale * transformed_batch + self.shift

            if self.precision:
                transformed_batch = self._apply_precision(transformed_batch)

            transformed_batches.append(transformed_batch)

            if self.verbose >= 3:
                self._log(3, f"Batch {i + 1}-{n_batches} transformed.")

        return self._concatenate_batches(transformed_batches)

    def _get_elu_function(self):
        if self.backend_name_ == 'numpy':
            return self._elu_numpy
        elif self.backend_name_ == 'tensorflow':
            return self._elu_tensorflow
        elif self.backend_name_ == 'torch':
            return self._elu_pytorch
        else:
            raise ValueError(
                f"Unsupported backend: {self.backend_name_}. "
                "Supported backends are 'numpy', 'tensorflow', and 'torch'."
            )

    def _apply_elu(self, X):
        elu_func = self._get_elu_function()
        return elu_func(X)

    def _elu_numpy(self, X):
        return np.where(X > 0, X, self.alpha * (np.exp(X) - 1))

    def _elu_tensorflow(self, X):
        return self.backend_.nn.elu(X)

    def _elu_pytorch(self, X):
        return self.backend_.functional.elu(X)

    def _apply_precision(self, transformed_batch):
        if self.backend_name_ == 'numpy':
            return self.backend_.round(
                transformed_batch / self.precision
            ) * self.precision
        elif self.backend_name_ == 'tensorflow':
            return self.backend_.round(
                transformed_batch / self.precision
            ) * self.precision
        elif self.backend_name_ == 'torch':
            return self.backend_.round(
                transformed_batch / self.precision
            ) * self.precision
        else:
            raise ValueError(
                f"Unsupported backend for precision application: {self.backend_name_}."
            )

    def _concatenate_batches(self, transformed_batches):
        if self.backend_name_ == 'numpy':
            return self.backend_.vstack(transformed_batches)
        elif self.backend_name_ == 'tensorflow':
            return self.backend_.concat(transformed_batches, axis=0)
        elif self.backend_name_ == 'torch':
            return self.backend_.cat(transformed_batches, dim=0)
        else:
            raise ValueError(
                f"Unsupported backend for concatenation: {self.backend_name_}."
            )

    def _log(self, level, message):
        if self.verbose >= level:
            print(message)

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
        "shift": [Interval(Real, -1, 1 , closed ='both')], 
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
        verbose=0
    ):
        self.scale = scale
        self.shift = shift
        self.precision = precision
        self.batch_size = batch_size
        self.alpha = alpha
        self.backend = backend
        self.verbose = verbose


    @Appender(
        _activation_doc['fit'].format(fmt='LeakyReLUTransformer'), 
        join="\n"
    )
    def fit(self, X, y=None):
        """Fit the transformer."""
        self.backend_name_, self.backend_ = select_backend_n(
            self.backend, return_both=True
        )
        return self

    @DataTransformer(name='X', mode='lazy', keep_origin_type=True)
    @doc(_shared_docs['activation_transform'])
    def transform(self, X):
        if self.verbose >= 1:
            self._log(1, f"Using backend: {self.backend_name_}")

        X = self._validate_input(X)
        X_transformed = self.scale * X + self.shift

        if self.batch_size:
            return self._batch_process(X_transformed)
        
        return self._apply_transformation(X_transformed)

    def _validate_input(self, X):
        return check_array(X, ensure_2d=True, input_name="X")

    def _apply_transformation(self, X):
        return self._apply_leaky_relu(X)

    def _batch_process(self, X):
        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / self.batch_size))
        X_transformed = np.zeros_like(X, dtype=np.float64)

        for i in range(n_batches):
            batch_start = i * self.batch_size
            batch_end = min((i + 1) * self.batch_size, n_samples)
            batch_X = X[batch_start:batch_end]

            if self.verbose >= 2:
                self._log(2, ( 
                    f"Processing batch {i + 1}/{n_batches}"
                    f" (samples {batch_start} to {batch_end})."
                    )
                )

            transformed_batch = self._apply_leaky_relu(batch_X)
            transformed_batch = self.scale * transformed_batch + self.shift

            if self.precision:
                transformed_batch = self._apply_precision(transformed_batch)

            X_transformed[batch_start:batch_end] = transformed_batch

            if self.verbose >= 3:
                self._log(3, f"Batch {i + 1}-{n_batches} transformed.")

        return X_transformed

    def _get_leaky_relu_function(self):
        if self.backend_name_ == 'numpy':
            return self._leaky_relu_numpy
        elif self.backend_name_ == 'tensorflow':
            return self._leaky_relu_tensorflow
        elif self.backend_name_ == 'torch':
            return self._leaky_relu_pytorch
        else:
            raise ValueError(
                f"Unsupported backend: {self.backend_name_}. "
                "Supported backends are 'numpy', 'tensorflow', and 'torch'."
            )

    def _apply_leaky_relu(self, X):
        leaky_relu_func = self._get_leaky_relu_function()
        return leaky_relu_func(X)

    def _leaky_relu_numpy(self, X):
        return np.where(X > 0, X, self.alpha * X)

    def _leaky_relu_tensorflow(self, X):
        return self.backend_.where(
            X > 0, X, self.alpha * X
        )

    def _leaky_relu_pytorch(self, X):
        return self.backend_.where(
            X > 0, X, self.alpha * X
        )

    def _apply_precision(self, transformed_batch):
        if self.backend_name_ == 'numpy':
            return self.backend_.round(
                transformed_batch / self.precision
            ) * self.precision
        elif self.backend_name_ == 'tensorflow':
            return self.backend_.round(
                transformed_batch / self.precision
            ) * self.precision
        elif self.backend_name_ == 'torch':
            return self.backend_.round(
                transformed_batch / self.precision
            ) * self.precision
        else:
            raise ValueError(
                f"Unsupported backend for precision application: {self.backend_name_}."
            )

    def _concatenate_batches(self, X_transformed):
        if self.backend_name_ == 'numpy':
            return self.backend_.vstack(X_transformed)
        elif self.backend_name_ == 'tensorflow':
            return self.backend_.concat(X_transformed, axis=0)
        elif self.backend_name_ == 'torch':
            return self.backend_.cat(X_transformed, dim=0)
        else:
            raise ValueError(
                f"Unsupported backend for concatenation: {self.backend_name_}."
            )

    def _log(self, level, message):
        if self.verbose >= level:
            print(message)

    def _batch_process_tensorflow(self, X):
        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / self.batch_size))
        X_transformed = self.backend_.zeros_like(X, dtype=self.backend_.float32)

        for i in range(n_batches):
            batch_start = i * self.batch_size
            batch_end = min((i + 1) * self.batch_size, n_samples)
            batch_X = X[batch_start:batch_end]
            X_transformed = self._transform_tensorflow(batch_X)

            if self.verbose:
                self._log(1, f"Processing batch {i + 1}/{n_batches}")

        return X_transformed

    def _batch_process_torch(self, X):
        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / self.batch_size))
        X_transformed = self.backend_.zeros_like(self.backend_(X), dtype=self.backend_.float32)

        for i in range(n_batches):
            batch_start = i * self.batch_size
            batch_end = min((i + 1) * self.batch_size, n_samples)
            X_batch = X[batch_start:batch_end]
            X_transformed[batch_start:batch_end] = self._transform_torch(X_batch)

            if self.verbose:
                self._log(1, f"Processing batch {i + 1}/{n_batches}")

        return X_transformed

    def _transform_tensorflow(self, X):
        X_tensor = self.backend_.convert_to_tensor(X, dtype=self.backend_.float32)
        return self.backend_.where(
            X_tensor > 0, X_tensor, self.alpha * X_tensor
        ).numpy()

    def _transform_torch(self, X):
        X_tensor = self.backend_.tensor(X, dtype=self.backend_.float32)
        return self.backend_.where(
            X_tensor > 0, X_tensor, self.alpha * X_tensor
        ).numpy()
    
    
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
        "shift": [Interval(Real, -1, 1 , closed ='both')], 
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
        join="\n", 
    )
    def fit(self, X, y=None):
        """Fit the transformer."""
        self.backend_name_, self.backend_ = select_backend_n(
            self.backend, return_both=True
        )
        return self

    @DataTransformer(name='X', mode='lazy', keep_origin_type=True)
    @doc(_shared_docs['activation_transform'])
    def transform(self, X):
        if self.verbose >= 1:
            self._log(1, f"Using backend: {self.backend_name_}")

        X = self._validate_input(X)
        X_transformed = self.scale * X + self.shift

        if self.batch_size:
            return self._batch_process(X_transformed)
        
        return self._apply_transformation(X_transformed)

    def _validate_input(self, X):
        """
        Validate and preprocess the input data.
        """
        return check_array(X, ensure_2d=True, input_name="X")

    def _apply_transformation(self, X):
        """
        Apply the Softmax activation function using the selected backend.
        """
        return self._apply_softmax(X)

    def _batch_process(self, X):
        """
        Process the data in batches according to the specified batch_size.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data to transform.

        Returns
        -------
        X_transformed : array-like, shape (n_samples, n_features)
            The transformed data after applying Softmax in batches.
        """
        n_samples = X.shape[0]
        n_batches = (n_samples // self.batch_size) + (
            1 if n_samples % self.batch_size != 0 else 0)
        transformed_batches = []

        for i in range(n_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, n_samples)
            batch_X = X[start_idx:end_idx]

            if self.verbose >= 2:
                self._log(2, f"Processing batch {start_idx} to {end_idx}.")

            transformed_batch = self._apply_softmax(batch_X)
            transformed_batch = self.scale * transformed_batch + self.shift

            if self.precision:
                transformed_batch = self._apply_precision(transformed_batch)

            transformed_batches.append(transformed_batch)

            if self.verbose >= 3:
                self._log(3, f"Batch {start_idx}-{end_idx} transformed.")

        return self._concatenate_batches(transformed_batches)

    def _get_softmax_function(self):
        """
        Retrieve the appropriate Softmax function based on the backend.
        """
        if self.backend_name_ == 'numpy':
            return self._softmax_numpy
        elif self.backend_name_ == 'tensorflow':
            return self._softmax_tensorflow
        elif self.backend_name_ == 'torch':
            return self._softmax_pytorch
        else:
            raise ValueError(
                f"Unsupported backend: {self.backend_name_}. "
                "Supported backends are 'numpy', 'tensorflow', and 'torch'."
            )

    def _apply_softmax(self, X):
        """
        Apply the Softmax activation function using the selected backend.
        """
        softmax_func = self._get_softmax_function()
        return softmax_func(X)

    def _softmax_numpy(self, X):
        """
        Apply Softmax activation using NumPy.
        """
        # Numerical stability by subtracting the max from each row
        shifted_X = self.backend_.subtract(
            X, self.backend_.max(X, axis=1, keepdims=True))
        exp_X = self.backend_.exp(shifted_X)
        sum_exp_X = self.backend_.sum(exp_X, axis=1, keepdims=True)
        return self.backend_.divide(exp_X, sum_exp_X)

    def _softmax_tensorflow(self, X):
        """
        Apply Softmax activation using TensorFlow.
        """
        shifted_X = self.backend_.subtract(
            X, self.backend_.reduce_max(X, axis=1, keepdims=True))
        exp_X = self.backend_.exp(shifted_X)
        sum_exp_X = self.backend_.reduce_sum(exp_X, axis=1, keepdims=True)
        return self.backend_.divide(exp_X, sum_exp_X)

    def _softmax_pytorch(self, X):
        """
        Apply Softmax activation using PyTorch.
        """
        shifted_X = self.backend_.subtract(
            X, self.backend_.max(X, dim=1, keepdim=True)[0])
        exp_X = self.backend_.exp(shifted_X)
        sum_exp_X = self.backend_.sum(exp_X, dim=1, keepdim=True)
        return self.backend_.divide(exp_X, sum_exp_X)

    def _apply_precision(self, transformed_batch):
        """
        Apply precision rounding to the transformed batch.
        """
        if self.backend_name_ == 'numpy':
            return self.backend_.round(
                transformed_batch / self.precision
            ) * self.precision
        elif self.backend_name_ == 'tensorflow':
            return self.backend_.round(
                transformed_batch / self.precision
            ) * self.precision
        elif self.backend_name_ == 'torch':
            return self.backend_.round(
                transformed_batch / self.precision
            ) * self.precision
        else:
            raise ValueError(
                f"Unsupported backend for precision application: {self.backend_name_}."
            )

    def _concatenate_batches(self, X_transformed):
        """
        Concatenate all transformed batches into a single array/tensor.
        """
        if self.backend_name_ == 'numpy':
            return self.backend_.vstack(X_transformed)
        elif self.backend_name_ == 'tensorflow':
            return self.backend_.concat(X_transformed, axis=0)
        elif self.backend_name_ == 'torch':
            return self.backend_.cat(X_transformed, dim=0)
        else:
            raise ValueError(
                f"Unsupported backend for concatenation: {self.backend_name_}."
            )

    def _log(self, level, message):
        """
        Handle logging based on the verbosity level.
        """
        if self.verbose >= level:
            print(message)

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
        "shift": [Interval(Real, -1, 1 , closed ='both')], 
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
        _activation_doc['fit'].format(fmt='SwishTransformer'), 
        join="\n", 
    )
    def fit(self, X, y=None):
        """Fit the transformer."""
        self.backend_name_, self.backend_ = select_backend_n(
            self.backend, return_both=True
        )
        return self

    @DataTransformer(name='X', mode='lazy', keep_origin_type=True)
    @doc(_shared_docs['activation_transform'])
    def transform(self, X):
        if self.verbose >= 1:
            self._log(1, f"Using backend: {self.backend_name_}")

        X = self._validate_input(X)
        X_transformed = (self.scale * X) + self.shift

        if self.batch_size:
            return self._batch_process(X_transformed)
        
        return self._apply_transformation(X_transformed)

    def _validate_input(self, X):
        """
        Validate and preprocess the input data.
        """
        return check_array(X, ensure_2d=True, input_name="X")

    def _apply_transformation(self, X):
        """
        Apply the Swish activation function using the selected backend.
        """
        return self._apply_swish(X)

    def _batch_process(self, X):
        """
        Process the data in batches for better scalability.
        This method is used when `batch_size` is specified.
        """
        n_samples = X.shape[0]
        n_batches = int((n_samples + self.batch_size - 1) / self.batch_size)
        X_transformed = []

        for i in range(n_batches):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, n_samples)
            batch = X[start:end]
            if self.verbose >= 2:
                self._log(2, f"Processing batch {start} to {end}.")

            transformed_batch = self._apply_swish(batch)
            transformed_batch = self.scale * transformed_batch + self.shift

            if self.precision:
                transformed_batch = self._apply_precision(transformed_batch)

            X_transformed.append(transformed_batch)

            if self.verbose >= 3:
                self._log(3, f"Batch {start}-{end} transformed.")

        return self._concatenate_batches(X_transformed)

    def _get_swish_function(self):
        """
        Retrieve the appropriate Swish function based on the backend.
        """
        if self.backend_name_ == 'numpy':
            return self._apply_swish_numpy
        elif self.backend_name_ == 'tensorflow':
            return self._apply_swish_tensorflow
        elif self.backend_name_ == 'torch':
            return self._apply_swish_pytorch
        else:
            raise ValueError(
                f"Unsupported backend: {self.backend_name_}. "
                "Supported backends are 'numpy', 'tensorflow', and 'torch'."
            )

    def _apply_swish(self, X):
        """
        Apply the Swish activation function using the selected backend.
        """
        swish_func = self._get_swish_function()
        return swish_func(X)

    def _apply_swish_numpy(self, X):
        """
        Apply the Swish activation function using NumPy.
        """
        sigmoid = self.backend_.divide(
            1,
            self.backend_.add(1, self.backend_.exp(-X))
        )
        return self.backend_.multiply(X, sigmoid)

    def _apply_swish_tensorflow(self, X):
        """
        Apply the Swish activation function using TensorFlow.
        """
        return self.backend_.multiply(
            X,
            self.backend_.divide(
                1,
                self.backend_.add(1, self.backend_.exp(-X))
            )
        )

    def _apply_swish_pytorch(self, X):
        """
        Apply the Swish activation function using PyTorch.
        """
        return self.backend_.multiply(
            X,
            self.backend_.divide(
                1,
                self.backend_.add(1, self.backend_.exp(-X))
            )
        )

    def _apply_precision(self, transformed_batch):
        """
        Apply precision rounding to the transformed batch.
        """
        if self.backend_name_ == 'numpy':
            return self.backend_.round(
                transformed_batch / self.precision
            ) * self.precision
        elif self.backend_name_ == 'tensorflow':
            return self.backend_.round(
                transformed_batch / self.precision
            ) * self.precision
        elif self.backend_name_ == 'torch':
            return self.backend_.round(
                transformed_batch / self.precision
            ) * self.precision
        else:
            raise ValueError(
                f"Unsupported backend for precision application: {self.backend_name_}."
            )

    def _concatenate_batches(self, X_transformed):
        """
        Concatenate all transformed batches into a single array/tensor.
        """
        if self.backend_name_ == 'numpy':
            return self.backend_.vstack(X_transformed)
        elif self.backend_name_ == 'tensorflow':
            return self.backend_.concat(X_transformed, axis=0)
        elif self.backend_name_ == 'torch':
            return self.backend_.cat(X_transformed, dim=0)
        else:
            raise ValueError(
                f"Unsupported backend for concatenation: {self.backend_name_}."
            )

    def _log(self, level, message):
        """
        Handle logging based on the verbosity level.
        """
        if self.verbose >= level:
            print(message)


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
        "shift": [Interval(Real, -1, 1 , closed ='both')], 
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
        join="\n", 
    )
    def fit(self, X, y=None):
        """Fit the transformer."""
        self.backend_name_, self.backend_ = select_backend_n(
            self.backend, return_both=True
        )
        return self

    @DataTransformer(name='X', mode='lazy', keep_origin_type=True)
    @doc(_shared_docs['activation_transform'])
    def transform(self, X):
        if self.verbose >= 1:
            self._log(1, f"Using backend: {self.backend_name_}")

        X = self._validate_input(X)
        X_transformed = self.scale * X + self.shift

        if self.batch_size:
            return self._batch_process(X_transformed)
        
        return self._apply_transformation(X_transformed)

    def _validate_input(self, X):
        """
        Validate and preprocess the input data.
        """
        from sklearn.utils import check_array
        return check_array(X, ensure_2d=True, input_name="X")

    def _apply_transformation(self, X):
        """
        Apply the HardSigmoid activation function using the selected backend.
        """
        return self._apply_hardsigmoid(X)

    def _batch_process(self, X):
        """
        Process data in batches according to the specified batch_size.
        """
        n_samples = X.shape[0]
        n_batches = (n_samples // self.batch_size) + (
            1 if n_samples % self.batch_size != 0 else 0)
        transformed_batches = []

        for i in range(n_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, n_samples)
            batch_X = X[start_idx:end_idx]

            if self.verbose >= 2:
                self._log(2, f"Processing batch {start_idx} to {end_idx}.")

            transformed_batch = self._apply_hardsigmoid(batch_X)
            transformed_batch = self.scale * transformed_batch + self.shift

            if self.precision:
                transformed_batch = self._apply_precision(transformed_batch)

            transformed_batches.append(transformed_batch)

            if self.verbose >= 3:
                self._log(3, f"Batch {start_idx}-{end_idx} transformed.")

        return self._concatenate_batches(transformed_batches)

    def _get_hardsigmoid_function(self):
        """
        Retrieve the appropriate HardSigmoid function based on the backend.
        """
        if self.backend_name_ == 'numpy':
            return self._hardsigmoid_numpy
        elif self.backend_name_ == 'tensorflow':
            return self._hardsigmoid_tensorflow
        elif self.backend_name_ == 'torch':
            return self._hardsigmoid_pytorch
        else:
            raise ValueError(
                f"Unsupported backend: {self.backend_name_}. "
                "Supported backends are 'numpy', 'tensorflow', and 'torch'."
            )

    def _apply_hardsigmoid(self, X):
        """
        Apply the HardSigmoid activation function using the selected backend.
        """
        hardsigmoid_func = self._get_hardsigmoid_function()
        return hardsigmoid_func(X)

    def _hardsigmoid_numpy(self, X):
        """
        Apply HardSigmoid activation using NumPy.
        """
        hard_sigmoid = self.backend_.clip(
            0.2 * X + 0.5, 0, 1
        )
        return self.backend_.multiply(X, hard_sigmoid)

    def _hardsigmoid_tensorflow(self, X):
        """
        Apply HardSigmoid activation using TensorFlow.
        """
        hard_sigmoid = self.backend_.clip_by_value(
            0.2 * X + 0.5, 0, 1
        )
        return self.backend_.multiply(X, hard_sigmoid)

    def _hardsigmoid_pytorch(self, X):
        """
        Apply HardSigmoid activation using PyTorch.
        """
        hard_sigmoid = self.backend_.clamp(
            0.2 * X + 0.5, min=0, max=1
        )
        return self.backend_.multiply(X, hard_sigmoid)

    def _apply_precision(self, transformed_batch):
        """
        Apply precision rounding to the transformed batch.
        """
        if self.backend_name_ == 'numpy':
            return self.backend_.round(
                transformed_batch / self.precision
            ) * self.precision
        elif self.backend_name_ == 'tensorflow':
            return self.backend_.round(
                transformed_batch / self.precision
            ) * self.precision
        elif self.backend_name_ == 'torch':
            return self.backend_.round(
                transformed_batch / self.precision
            ) * self.precision
        else:
            raise ValueError(
                f"Unsupported backend for precision application: {self.backend_name_}."
            )

    def _concatenate_batches(self, X_transformed):
        """
        Concatenate all transformed batches into a single array/tensor.
        """
        if self.backend_name_ == 'numpy':
            return self.backend_.vstack(X_transformed)
        elif self.backend_name_ == 'tensorflow':
            return self.backend_.concat(X_transformed, axis=0)
        elif self.backend_name_ == 'torch':
            return self.backend_.cat(X_transformed, dim=0)
        else:
            raise ValueError(
                f"Unsupported backend for concatenation: {self.backend_name_}."
            )

    def _log(self, level, message):
        """
        Handle logging based on the verbosity level.
        """
        if self.verbose >= level:
            print(message)


@doc( 
    mathf =dedent( 
    """\
The Hard Swish function is an approximation of the Swish function,
defined as:

.. math::
    \text{HardSwish}(x) = x \cdot \text{HardSigmoid}(x)

where

.. math::
    \text{HardSigmoid}(x) = \text{clip}(0.2x + 0.5, 0, 1)
    """
    ),  
    parameters =_activation_doc['parameters'].format(afmt="HardSwish"), 
    methods = _activation_doc['methods'].format(afmt='HardSwish'),
) 
class HardSwishTransformer(BaseEstimator, TransformerMixin):
    """
    Hard Swish Activation Transformer.

    Applies the Hard Swish activation function element-wise to the input data.


    {mathf}

    {parameters}
    
    verbose : int, default ``0``
        Verbosity level for logging. Ranges from ``0`` (no output) to
        ``7`` (most detailed output).

    {methods}

    Examples
    --------
    >>> from gofast.transformers.activations import HardSwishTransformer
    >>> import numpy as np
    >>> X = np.array([[1.0, -2.0], [3.0, 4.0]])
    >>> transformer = HardSwishTransformer(
    ...    scale=1.0, shift=0.0, backend='numpy', verbose=1)
    >>> transformer.fit(X)
    HardSwishTransformer()
    >>> transformed_X = transformer.transform(X)
    Using backend: numpy
    Processed batch 0 to 2.
    >>> print(transformed_X)
    [[ 0.7 -0.0]
     [ 2.1  4.0]]

    Notes
    -----
    The Hard Swish activation function is computationally efficient and
    often used in deep learning models to introduce non-linearity. It provides
    a balance between performance and computational cost, making it suitable
    for various applications in neural network architectures [1]_.

    See also
    --------
    SwishTransformer : Transformer applying the Swish activation function.

    References
    ----------
    .. [1] Zagoruyko, S., & Komodakis, N. (2016). Paying more attention to
       attention: Improving the performance of convolutional neural networks
       via attention transfer. *arXiv preprint arXiv:1612.03928*.
    """
    @validate_params ( { 
        "scale": [ Interval(Real, 0, None, closed ='neither' )], 
        "shift": [Interval(Real, -1, 1 , closed ='both')], 
        "precision": [Interval(Real, 0 , 1 , closed ="neither")], 
        "batch_size": [ Interval ( Integral, 1, None , closed ='left'), None], 
        "backend": [StrOptions (_VALID_BACKEND_SET), None]
        }
    )
    def __init__(
        self,
        scale= 1.0,
        shift= 0.0,
        precision = 1e-6,
        batch_size = None,
        backend = None,
        verbose= 0
    ):
        self.scale = scale
        self.shift = shift
        self.precision = precision
        self.batch_size = batch_size
        self.backend = backend
        self.verbose = verbose

    @Appender(
        _activation_doc['fit'].format(fmt='HardSwishTransformer'), 
        join="\n", 
    )
    def fit(self, X, y=None):
        """Fit the transformer."""
        self.backend_name_, self.backend_ = select_backend_n(
            self.backend, return_both=True
        )
        return self

    @DataTransformer(name='X', mode='lazy', keep_origin_type=True)
    @doc(_shared_docs['activation_transform'])
    def transform(self, X):
        if self.verbose >= 1:
            self._log(1, f"Using backend: {self.backend_name_}")

        swish_func = self._get_swish_function()

        X_transformed = []
        total_samples  = X.shape[0]
        batch_size     = self.batch_size or total_samples

        for start in range(0, total_samples, batch_size):
            end     = start + batch_size
            batch   = X[start:end]
            if self.verbose >= 2:
                self._log(2, f"Processing batch {start} to {end}.")

            transformed_batch = swish_func(batch)
            transformed_batch = (
                self.scale * transformed_batch + self.shift
            )

            if self.precision:
                transformed_batch = self._apply_precision(
                    transformed_batch
                )

            X_transformed.append(transformed_batch)

            if self.verbose >= 3:
                self._log(3, f"Batch {start}-{end} transformed.")

        return self._concatenate_batches(X_transformed)

    def _get_swish_function(self):
        """
        Retrieve the appropriate HardSwish function based on the backend.
        """
        if self.backend_name_ == 'numpy':
            return self._hard_swish_numpy
        elif self.backend_name_ == 'tensorflow':
            return self._hard_swish_tensorflow
        elif self.backend_name_ == 'torch':
            return self._hard_swish_torch
        else:
            raise ValueError(
                f"Unsupported backend: {self.backend_name_}. "
                "Supported backends are 'numpy', 'tensorflow', and 'torch'."
            )

    def _hard_swish_numpy(self, x):
        """
        Apply HardSwish activation using NumPy.
        """
        hard_sigmoid = self.backend_.clip(
            0.2 * x + 0.5, 0, 1
        )
        return self.backend_.multiply(x, hard_sigmoid)

    def _hard_swish_tensorflow(self, x):
        """
        Apply HardSwish activation using TensorFlow.
        """
        hard_sigmoid = self.backend_.clip_by_value(
            0.2 * x + 0.5, 0, 1
        )
        return self.backend_.multiply(x, hard_sigmoid)

    def _hard_swish_torch(self, x):
        """
        Apply HardSwish activation using PyTorch.
        """
        hard_sigmoid = self.backend_.clamp(
            0.2 * x + 0.5, min=0, max=1
        )
        return self.backend_.multiply(x, hard_sigmoid)

    def _apply_precision(self, transformed_batch):
        """
        Apply precision rounding to the transformed batch.
        """
        if self.backend_name_ == 'numpy':
            return self.backend_.round(
                transformed_batch / self.precision
            ) * self.precision
        elif self.backend_name_ == 'tensorflow':
            return self.backend_.round(
                transformed_batch / self.precision
            ) * self.precision
        elif self.backend_name_ == 'torch':
            return self.backend_.round(
                transformed_batch / self.precision
            ) * self.precision
        else:
            raise ValueError(
                f"Unsupported backend for precision application: {self.backend_name_}."
            )

    def _concatenate_batches(self, X_transformed):
        """
        Concatenate all transformed batches into a single array/tensor.
        """
        if self.backend_name_ == 'numpy':
            return self.backend_.vstack(X_transformed)
        elif self.backend_name_ == 'tensorflow':
            return self.backend_.concat(X_transformed, axis=0)
        elif self.backend_name_ == 'torch':
            return self.backend_.cat(X_transformed, dim=0)
        else:
            raise ValueError(
                f"Unsupported backend for concatenation: {self.backend_name_}."
            )

    def _log(self, level, message):
        """
        Handle logging based on the verbosity level.
        """
        if self.verbose >= level:
            print(message)


@doc( 
    mathf =dedent( 
    """\
    The Softplus function is a smooth approximation of the ReLU function,
    defined as:
    
    .. math::
        \text{Softplus}(x) = \log(1 + \exp(x))

    where :math:`x` is the input value to the activation function. The function
    operates on each element of the input data independently, ensuring a smooth
    transition from linear to non-linear behavior, which can aid in the training
    of neural networks by providing differentiability.
    """
    ),  
    parameters =_activation_doc['parameters'].format(afmt="Softplus"), 
    methods = _activation_doc['methods'].format(afmt='Softplus'),
) 
class SoftplusTransformer(BaseEstimator, TransformerMixin):
    """
    Softplus Activation Transformer.

    Applies the Softplus activation function element-wise to the input data.

    {mathf}

    {parameters}
    
    verbose : int, default ``0``
        Verbosity level for logging. Ranges from ``0`` (no output) to
        ``7`` (most detailed output).

    {methods}

    Examples
    --------
    >>> from gofast.transformers.activations import SoftplusTransformer
    >>> import numpy as np
    >>> X = np.array([[1.0, -2.0], [3.0, 4.0]])
    >>> transformer = SoftplusTransformer(scale=1.0, shift=0.0, backend='numpy',
    ...                                      verbose=1)
    >>> transformer.fit(X)
    SoftplusTransformer()
    >>> transformed_X = transformer.transform(X)
    Using backend: numpy
    Processed batch 0 to 2.
    >>> print(transformed_X)
    [[1.31326169 0.12692801]
     [3.04858735 4.01814993]]

    Notes
    -----
    The Softplus activation function provides a smooth and differentiable
    alternative to the ReLU activation, which can help in training neural networks
    by mitigating issues like dying ReLUs. It is particularly useful in scenarios
    where a non-linear activation is required but smoothness is preferred.

    See also
    --------
    ReLUTransformer : Transformer applying the ReLU activation function.
    HardSwishTransformer : Transformer applying the Hard Swish activation function.
    
    References
    ----------
    .. [1] Glorot, X., & Bengio, Y. (2010). Understanding the difficulty 
       of training deep feedforward neural networks. In *Proceedings of the 
       13th International Conference on Artificial Intelligence and 
       Statistics* (pp. 249-256).
    """
    @validate_params ( { 
        "scale": [ Interval(Real, 0, None, closed ='neither' )], 
        "shift": [Interval(Real, -1, 1 , closed ='both')], 
        "precision": [Interval(Real, 0 , 1 , closed ="neither")], 
        "batch_size": [ Interval ( Integral, 1, None , closed ='left'), None], 
        "backend": [StrOptions (_VALID_BACKEND_SET), None]
        }
    )
    def __init__(
        self,
        scale = 1.0,
        shift = 0.0,
        precision = 1e-6,
        batch_size = None,
        backend = None,
        verbose= 0
    ):
        self.scale = scale
        self.shift = shift
        self.precision = precision
        self.batch_size = batch_size
        self.backend = backend
        self.verbose = verbose

    @Appender(
        _activation_doc['fit'].format(fmt='SoftplusTransformer'), 
        join="\n", 
    )
    def fit(self, X, y=None):
        """Fit the transformer."""
        self.backend_name_, self.backend_ = select_backend_n(
            self.backend, return_both=True
        )
        return self

    @DataTransformer(name='X', mode='lazy', keep_origin_type=True)
    @doc(_shared_docs['activation_transform'])
    def transform(self, X):
        if self.verbose >= 1:
            self._log(1, f"Using backend: {self.backend_name_}")

        softplus_func = self._get_softplus_function()

        X_transformed = []
        total_samples  = X.shape[0]
        batch_size     = self.batch_size or total_samples

        for start in range(0, total_samples, batch_size):
            end   = start + batch_size
            batch = X[start:end]
            if self.verbose >= 2:
                self._log(2, f"Processing batch {start} to {end}.")

            transformed_batch = softplus_func(batch)
            transformed_batch = (
                self.scale * transformed_batch + self.shift
            )

            if self.precision:
                transformed_batch = self._apply_precision(
                    transformed_batch
                )

            X_transformed.append(transformed_batch)

            if self.verbose >= 3:
                self._log(3, f"Batch {start}-{end} transformed.")

        return self._concatenate_batches(X_transformed)

    def _get_softplus_function(self):
        """
        Retrieve the appropriate Softplus function based on the backend.
        """
        if self.backend_name_ == 'numpy':
            return self._softplus_numpy
        elif self.backend_name_ == 'tensorflow':
            return self._softplus_tensorflow
        elif self.backend_name_ == 'torch':
            return self._softplus_torch
        else:
            raise ValueError(
                f"Unsupported backend: {self.backend_name_}. "
                "Supported backends are 'numpy', 'tensorflow', and 'torch'."
            )

    def _softplus_numpy(self, x):
        """
        Apply Softplus activation using NumPy.
        """
        return self.backend_.log1p(self.backend_.exp(x))  # Softplus = log(1 + exp(x))

    def _softplus_tensorflow(self, x):
        """
        Apply Softplus activation using TensorFlow.
        """
        return self.backend_.math.softplus(x)

    def _softplus_torch(self, x):
        """
        Apply Softplus activation using PyTorch.
        """
        return self.backend_.nn.functional.softplus(x)

    def _apply_precision(self, transformed_batch):
        """
        Apply precision rounding to the transformed batch.
        """
        if self.backend_name_ == 'numpy':
            return self.backend_.round(
                transformed_batch / self.precision
            ) * self.precision
        elif self.backend_name_ == 'tensorflow':
            return self.backend_.round(
                transformed_batch / self.precision
            ) * self.precision
        elif self.backend_name_ == 'torch':
            return self.backend_.round(
                transformed_batch / self.precision
            ) * self.precision
        else:
            raise ValueError(
                f"Unsupported backend for precision application: {self.backend_name_}."
            )

    def _concatenate_batches(self, X_transformed):
        """
        Concatenate all transformed batches into a single array/tensor.
        """
        if self.backend_name_ == 'numpy':
            return self.backend_.vstack(X_transformed)
        elif self.backend_name_ == 'tensorflow':
            return self.backend_.concat(X_transformed, axis=0)
        elif self.backend_name_ == 'torch':
            return self.backend_.cat(X_transformed, dim=0)
        else:
            raise ValueError(
                f"Unsupported backend for concatenation: {self.backend_name_}."
            )

    def _log(self, level, message):
        """
        Handle logging based on the verbosity level.
        """
        if self.verbose >= level:
            print(message)


@doc( 
    mathf =dedent( 
    """\
    The GELU function is defined as:
    
    .. math::
       \text{GELU}(x) = 0.5 \cdot x \cdot
       \left(1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right)
    
    where :math:`\text{erf}` is the error function,
    and :math:`x` is the input value.
    """
    ),  
    parameters =_activation_doc['parameters'].format(afmt="Softplus"), 
    methods = _activation_doc['methods'].format(afmt='Softplus'),
) 
class GELUTransformer(BaseEstimator, TransformerMixin):
    """
    GELU (Gaussian Error Linear Unit) Transformer.
    
    Applies the GELU activation function element-wise to the input
    data. The GELU function introduces non-linearity in neural
    networks, providing benefits similar to dropout and
    regularization by enabling smoother gradients.
    
    {mathf}
    
    {parameters}
    
    verbose : int, default ``0``
       Verbosity level for logging. Ranges from ``0`` (no output)
       to ``7`` (most detailed output).
       
    {methods}
    
    Examples
    --------
    >>> from gofast.transformers.activations import GELUTransformer
    >>> import numpy as np
    >>> X = np.array([[0.0, -1.0], [2.0, 3.0]])
    >>> transformer = GELUTransformer(scale=1.0, shift=0.0,
    ...                              backend='numpy', verbose=2)
    >>> transformer.fit(X)
    GELUTransformer()
    >>> transformed_X = transformer.transform(X)
    Using backend: numpy
    Processing batch 0 to 2.
    Batch 0-2 transformed.
    >>> print(transformed_X)
    [[0.         -0.15865525]
     [1.84119224 2.99627208]]
    
    Notes
    -----
    The GELU activation function is widely used in modern neural
    network architectures, including transformers and BERT models.
    Its probabilistic interpretation allows neurons to decide the
    extent to which they should be activated, potentially leading
    to better model performance and convergence properties [1]_.
    
    See also
    --------
    ReLUTransformer : Transformer applying the ReLU activation
       function.
    SELUTransformer : Transformer applying the SELU activation
       function.
    
    References
    ----------
    .. [1] Hendrycks, D., & Gimpel, K. (2016). Gaussian Error Linear Units
           (GELUs). In *Proceedings of the 33rd International Conference on
           International Conference on Machine Learning* (Vol. 48, pp.
           2340-2348).
    """
    @validate_params ( { 
        "scale": [ Interval(Real, 0, None, closed ='neither' )], 
        "shift": [Interval(Real, -1, 1 , closed ='both')], 
        "precision": [Interval(Real, 0 , 1 , closed ="neither")], 
        "batch_size": [ Interval ( Integral, 1, None , closed ='left'), None], 
        "backend": [StrOptions (_VALID_BACKEND_SET), None]
        }
    )
    def __init__(
        self,
        scale = 1.0,
        shift = 0.0,
        precision = 1e-6,
        batch_size = None,
        backend = None,
        verbose= 0
    ):
        self.scale = scale
        self.shift = shift
        self.precision = precision
        self.batch_size = batch_size
        self.backend = backend
        self.verbose = verbose

    @Appender(
        _activation_doc['fit'].format(fmt='GELUTransformer'), 
        join="\n", 
    )
    def fit(self, X, y=None):
        """Fit the transformer."""
        self.backend_name_, self.backend_ = select_backend_n(
            self.backend, return_both=True
        )
        return self

    @DataTransformer(name='X', mode='lazy', keep_origin_type=True)
    @doc(_shared_docs['activation_transform'])
    def transform(self, X):
        if self.verbose >= 1:
            self._log(1, f"GELU computed using backend: {self.backend_name_}")

        gelu_func = self._get_gelu_function()

        X_transformed = []
        total_samples  = X.shape[0]
        batch_size     = self.batch_size or total_samples

        for start in range(0, total_samples, batch_size):
            end   = start + batch_size
            batch = X[start:end]
            if self.verbose >= 2:
                self._log(2, f"Processing batch {start} to {end}.")

            transformed_batch = gelu_func(batch)
            transformed_batch = (
                self.scale * transformed_batch + self.shift
            )

            if self.precision:
                transformed_batch = self._apply_precision(
                    transformed_batch
                )

            X_transformed.append(transformed_batch)

            if self.verbose >= 3:
                self._log(3, f"Batch {start}-{end} transformed.")

        return self._concatenate_batches(X_transformed)

    def _get_gelu_function(self):
        """
        Retrieve the appropriate GELU function based on the backend.
        """
        if self.backend_name_ == 'numpy':
            return self._gelu_numpy
        elif self.backend_name_ == 'tensorflow':
            return self._gelu_tensorflow
        elif self.backend_name_ == 'torch':
            return self._gelu_torch
        else:
            raise ValueError(
                f"Unsupported backend: {self.backend_name_}. "
                "Supported backends are 'numpy', 'tensorflow', and 'torch'."
            )

    def _gelu_numpy(self, x):
        """
        Apply GELU activation using NumPy.
        """
        return 0.5 * x * (1 + safe_erf(x / (2 ** 0.5)))

    def _gelu_tensorflow(self, x):
        """
        Apply GELU activation using TensorFlow.
        """
        return 0.5 * x * (1 + self.backend_.math.erf(x / (2 ** 0.5)))

    def _gelu_torch(self, x):
        """
        Apply GELU activation using PyTorch.
        """
        return 0.5 * x * (1 + self.backend_.erf(x / (2 ** 0.5)))

    def _apply_precision(self, transformed_batch):
        """
        Apply precision rounding to the transformed batch.
        """
        if self.backend_name_ == 'numpy':
            return self.backend_.round(
                transformed_batch / self.precision
            ) * self.precision
        elif self.backend_name_ == 'tensorflow':
            return self.backend_.round(
                transformed_batch / self.precision
            ) * self.precision
        elif self.backend_name_ == 'torch':
            return self.backend_.round(
                transformed_batch / self.precision
            ) * self.precision
        else:
            raise ValueError(
                f"Unsupported backend for precision application: {self.backend_name_}."
            )

    def _concatenate_batches(self, X_transformed):
        """
        Concatenate all transformed batches into a single array/tensor.
        """
        if self.backend_name_ == 'numpy':
            return self.backend_.vstack(X_transformed)
        elif self.backend_name_ == 'tensorflow':
            return self.backend_.concat(X_transformed, axis=0)
        elif self.backend_name_ == 'torch':
            return self.backend_.cat(X_transformed, dim=0)
        else:
            raise ValueError(
                f"Unsupported backend for concatenation: {self.backend_name_}."
            )

    def _log(self, level, message):
        """
        Handle logging based on the verbosity level.
        """
        if self.verbose >= level:
            print(message)

@doc( 
    mathf =dedent( 
    """\
    The SELU function is defined as:
    
    {mathf}
    .. math::
       \text{SELU}(x) = \lambda \cdot 
       \begin{cases}
           x & \text{if } x > 0 \\
           \alpha \cdot (\exp(x) - 1) & \text{otherwise}
       \end{cases}
    
    where :math:`\lambda` and :math:`\alpha` are fixed constants:
    
    .. math::
       \lambda = 1.0507, \quad \alpha = 1.6733
    """
    ),  
    methods = _activation_doc['methods'].format(afmt='SELU'),
) 
class SELUTransformer(BaseEstimator, TransformerMixin):
    """
    SELU (Scaled Exponential Linear Unit) Transformer.
    
    Applies the SELU activation function element-wise to the input
    data. The SELU function automatically scales inputs to maintain
    mean and variance, which can help in self-normalizing neural
    networks. This property reduces the need for other normalization
    techniques like batch normalization.
    
    The SELU function is defined as:
    
    {mathf}
    
    Parameters
    ----------
    scale : float, default ``1.0507``
       Scaling factor (:math:`\lambda`) applied to the transformed data.
    
    alpha : float, default ``1.6733``
       Scaling factor (:math:`\alpha`) for negative inputs in the SELU
       function.
    
    shift : float, default ``0.0``
       Shifting value added to the scaled data.
    
    precision : float, default ``1e-6``
       Precision for rounding the transformed data.
    
    batch_size : int or None, default ``None``
       Number of samples per batch. If ``None``, the entire dataset
       is processed in a single batch.
    
    backend : str or None, default ``None``
       Computational backend to use. Supported backends are:
       
       - ``'numpy'``: Uses NumPy for computations.
       - ``'tensorflow'``: Uses TensorFlow for computations.
       - ``'torch'``: Uses PyTorch for computations.
       
       If ``None``, defaults to ``'numpy'``.
    
    verbose : int, default ``0``
       Verbosity level for logging. Ranges from ``0`` (no output)
       to ``7`` (most detailed output).
    
    {methods}
    
    Examples
    --------
    >>> from gofast.transformers.activations import SELUTransformer
    >>> import numpy as np
    >>> X = np.array([[0.0, -1.0], [2.0, 3.0]])
    >>> transformer = SELUTransformer(scale=1.0507, alpha=1.6733,
    ...                              backend='numpy', verbose=2)
    >>> transformer.fit(X)
    SELUTransformer()
    >>> transformed_X = transformer.transform(X)
    Using backend: numpy
    Processing batch 0 to 2.
    Batch 0-2 transformed.
    >>> print(transformed_X)
    [[0.         -1.75960178]
     [2.1014     3.1521    ]]
    
    Notes
    -----
    The SELU activation function is designed to induce self-normalizing
    properties in neural networks, helping maintain the mean and
    variance of activations throughout the network layers. This can lead
    to faster convergence during training and improved overall
    performance [1]_.
    
    It is essential to use SELU with architectures that support
    self-normalization, such as fully connected feedforward networks
    with appropriate weight initialization.
    
    See also
    --------
    ReLUTransformer : Transformer applying the ReLU activation
       function.
    GELUTransformer : Transformer applying the GELU activation
       function.
    
    References
    ----------
    .. [1] Klambauer, G., Unterthiner, T., Mayr, A., & Hochreiter, S.
       (2017). Self-normalizing neural networks. In *Proceedings of the
       30th International Conference on Neural Information Processing
       Systems* (pp. 972-982).
    """
    @validate_params ( { 
        "alpha": [Interval(Real, 1 , None , closed ="left")], 
        "scale": [ Interval(Real, 0, None, closed ='neither' )], 
        "shift": [Interval(Real, -1, 1 , closed ='both')], 
        "precision": [Interval(Real, 0 , 1 , closed ="neither")], 
        "batch_size": [ Interval ( Integral, 1, None , closed ='left'), None], 
        "backend": [StrOptions (_VALID_BACKEND_SET), None]
        }
    )
    def __init__(
        self,
        alpha = 1.6733,
        scale = 1.0507,
        shift = 0.0,
        precision = 1e-6,
        batch_size = None,
        backend = None,
        verbose= 0
    ):
        self.scale = scale
        self.alpha = alpha
        self.shift = shift
        self.precision = precision
        self.batch_size = batch_size
        self.backend = backend
        self.verbose = verbose

    @Appender(
        _activation_doc['fit'].format(fmt='SELUTransformer'), 
        join= "\n", 
        )
    def fit(self, X, y=None):
        """Fit the transformer."""
        self.backend_name_, self.backend_ = select_backend_n(
            self.backend, return_both=True
        )
        return self

    @DataTransformer(name='X', mode='lazy', keep_origin_type=True)
    @doc(_shared_docs['activation_transform'])
    def transform(self, X):
        if self.verbose >= 1:
            self._log(1, f"SELU computation using backend: {self.backend_name_}")

        selu_func = self._get_selu_function()

        X_transformed = []
        total_samples  = X.shape[0]
        batch_size     = self.batch_size or total_samples

        for start in range(0, total_samples, batch_size):
            end   = start + batch_size
            batch = X[start:end]
            if self.verbose >= 2:
                self._log(2, f"Processing batch {start} to {end}.")

            transformed_batch = selu_func(batch)
            transformed_batch = (
                self.scale * transformed_batch + self.shift
            )

            if self.precision:
                transformed_batch = self._apply_precision(
                    transformed_batch
                )

            X_transformed.append(transformed_batch)

            if self.verbose >= 3:
                self._log(3, f"Batch {start}-{end} transformed.")

        return self._concatenate_batches(X_transformed)

    def _get_selu_function(self):
        """
        Retrieve the appropriate SELU function based on the backend.
        """
        if self.backend_name_ == 'numpy':
            return self._selu_numpy
        elif self.backend_name_ == 'tensorflow':
            return self._selu_tensorflow
        elif self.backend_name_ == 'torch':
            return self._selu_torch
        else:
            raise ValueError(
                f"Unsupported backend: {self.backend_name_}. "
                "Supported backends are 'numpy', 'tensorflow', and 'torch'."
            )

    def _selu_numpy(self, x):
        """
        Apply SELU activation using NumPy.
        """
        return self.backend_.scale * self.backend_.where(
            x > 0,
            x,
            self.alpha * (self.backend_.exp(x) - 1)
        )

    def _selu_tensorflow(self, x):
        """
        Apply SELU activation using TensorFlow.
        """
        return self.backend_.scale * self.backend_.where(
            x > 0,
            x,
            self.alpha * (self.backend_.exp(x) - 1)
        )

    def _selu_torch(self, x):
        """
        Apply SELU activation using PyTorch.
        """
        return self.backend_.scale * self.backend_.where(
            x > 0,
            x,
            self.alpha * (self.backend_.exp(x) - 1)
        )

    def _apply_precision(self, transformed_batch):
        """
        Apply precision rounding to the transformed batch.
        """
        if self.backend_name_ == 'numpy':
            return self.backend_.round(
                transformed_batch / self.precision
            ) * self.precision
        elif self.backend_name_ == 'tensorflow':
            return self.backend_.round(
                transformed_batch / self.precision
            ) * self.precision
        elif self.backend_name_ == 'torch':
            return self.backend_.round(
                transformed_batch / self.precision
            ) * self.precision
        else:
            raise ValueError(
                f"Unsupported backend for precision application: {self.backend_name_}."
            )

    def _concatenate_batches(self, X_transformed):
        """
        Concatenate all transformed batches into a single array/tensor.
        """
        if self.backend_name_ == 'numpy':
            return self.backend_.vstack(X_transformed)
        elif self.backend_name_ == 'tensorflow':
            return self.backend_.concat(X_transformed, axis=0)
        elif self.backend_name_ == 'torch':
            return self.backend_.cat(X_transformed, dim=0)
        else:
            raise ValueError(
                f"Unsupported backend for concatenation: {self.backend_name_}."
            )

    def _log(self, level, message):
        """
        Handle logging based on the verbosity level.
        """
        if self.verbose >= level:
            print(message)

class MishTransformer(BaseEstimator, TransformerMixin):
    """
    Mish Activation Transformer.

    This transformer applies the Mish activation function element-wise to the 
    input data. The Mish function is defined as:

    .. math::
        \text{Mish}(x) = x \cdot \tanh(\text{softplus}(x)) 
        = x \cdot \tanh(\ln(1 + e^{x}))

    The Mish activation introduces non-linearity, which can enhance the 
    performance of machine learning models by allowing them to learn more 
    complex patterns.

    Parameters
    ----------
    scale : ``float``, default ``1.0``
        Multiplier applied after the activation function. Scaling the output 
        can help in controlling the magnitude of the transformed data, 
        which may be beneficial for certain algorithms or architectures.

    shift : ``float``, default ``0.0``
        Value added after scaling. Shifting the output can adjust the 
        distribution of the transformed data, potentially improving 
        model convergence.

    precision : ``float``, default ``1e-6``
        A small constant to prevent numerical issues such as overflow 
        or division by zero. This parameter ensures numerical stability 
        during computations, especially when dealing with extreme input 
        values.

    batch_size : ``int`` or ``None``, default ``None``
        If specified, the transformation is applied in batches to manage 
        memory usage efficiently. Processing data in smaller chunks can 
        be advantageous when working with large datasets or limited 
        computational resources.

    backend : ``str`` or ``None``, default ``None``
        Specifies the computational backend to use for performing 
        transformations. Accepts the following values:
        
        - ``None``, ``'numpy'``, or ``'np'`` for NumPy (default).
        - ``'torch'``, ``'pytorch'`` for PyTorch.
        - ``'tensorflow'``, ``'tf'`` for TensorFlow.
        
        The parameter is case-insensitive, allowing variations like 
        ``'TensorFlow'``, ``'TF'``, or ``'np'``. If ``None`` is provided, 
        the default backend is NumPy.

    verbose : ``int``, default ``0``
        Controls the level of verbosity for debugging and logging. 
        The verbosity levels range from ``0`` to ``7``, where higher 
        values provide more detailed logs:
        
        - ``0``: No output.
        - ``1-2``: Basic transformation progress.
        - ``3-7``: Detailed batch processing and internal states.

    Attributes
    ----------
    backend_name_ : ``str``
        The standardized name of the selected backend (e.g., ``'numpy'``, 
        ``'torch'``, ``'tensorflow'``).

    backend_ : ``module``
        The actual backend module corresponding to ``backend_name_``. This 
        attribute is used to perform backend-specific operations.

    Methods
    -------
    fit(X, y=None)
        Fit the transformer by selecting the appropriate backend based on 
        the ``backend`` parameter.

    transform(X)
        Apply the Mish activation function to the input data, with 
        optional scaling and shifting. Supports batch processing for 
        large datasets.

    Formulation
    ------------
    The Mish activation function is mathematically formulated as:

    .. math::
        \text{Mish}(x) = x \cdot \tanh(\text{softplus}(x)) 
        = x \cdot \tanh(\ln(1 + e^{x}))

    Where:
    
    - :math:`x` is the input.
    - :math:`\text{softplus}(x)` is the softplus function defined as 
      :math:`\ln(1 + e^{x})`.
    - :math:`\tanh` is the hyperbolic tangent function.

    The transformer performs the following operations:

    1. **Activation**:
       Applies the Mish function to each element in the input data.
    
    2. **Scaling and Shifting**:
       Optionally scales and shifts the activated data:
       
       .. math::
           y = \text{scale} \cdot \text{Mish}(x) + \text{shift}

    Examples
    --------
    >>> from gofast.transformers.activations import MishTransformer
    >>> import numpy as np
    >>> X = np.array([[-1.0, 0.0, 1.0], [2.0, -2.0, 3.0]])
    >>> transformer = MishTransformer(scale=2.0, shift=0.5, 
    ...                                backend='np', verbose=1)
    >>> transformer.fit(X)
    >>> X_transformed = transformer.transform(X)
    >>> print(X_transformed)
    MishTransformer: Starting transformation.
    MishTransformer: Processing all data at once.
    MishTransformer: Transformation completed.
    [[0.5        0.5        2.5       ]
     [4.        0.5        6.        ]]

    Notes
    -----
    - **Backend Compatibility**: Ensure that the selected backend 
      (`'numpy'`, `'torch'`, `'tensorflow'`) supports all the required 
      operations used in the transformer. Differences in backend 
      implementations may lead to inconsistent behavior.

    - **Batch Processing**: When working with large datasets, specifying 
      the ``batch_size`` parameter can help manage memory usage by 
      processing data in smaller chunks.

    - **Numerical Stability**: The ``precision`` parameter is crucial for 
      preventing numerical issues, especially when dealing with large 
      positive or negative input values.

    - **Verbosity Levels**: Adjust the ``verbose`` parameter based on 
      your debugging needs. Higher verbosity levels provide more 
      detailed logs, which can be useful for monitoring the transformation 
      process.

    See Also
    --------
    NumPy : A fundamental package for scientific computing with Python 
        [1]_.
    
    PyTorch : An open-source machine learning library based on 
        the Torch library [2]_.
    
    TensorFlow : An end-to-end open-source platform for machine 
        learning [3]_.

    References
    ----------
    .. [1] van der Walt, S., Colbert, S. C., & Varoquaux, G. (2011). 
       The NumPy Array: A Structure for Efficient Numerical 
       Computation. *Computing in Science & Engineering*, 13(2), 
       22–30. https://doi.org/10.1109/MCSE.2011.37
    
    .. [2] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., 
       Chanan, G., ... & Chintala, S. (2019). PyTorch: An 
       Imperative Style, High-Performance Deep Learning 
       Library. In *Advances in Neural Information Processing 
       Systems* (pp. 8024-8035).
    
    .. [3] Abadi, M., Barham, P., Chen, J., Chen, Z., Davis, A., 
       Dean, J., ... & Zheng, X. (2016). TensorFlow: A System 
       for Large-Scale Machine Learning. In *12th USENIX 
       Symposium on Operating Systems Design and 
       Implementation (OSDI 16)* (pp. 265-283). 
    """
    @validate_params ( { 
        "scale": [ Interval(Real, 0, None, closed ='neither' )], 
        "shift": [Interval(Real, -1, 1 , closed ='both')], 
        "precision": [Interval(Real, 0 , 1 , closed ="neither")], 
        "batch_size": [ Interval ( Integral, 1, None , closed ='left'), None], 
        "backend": [StrOptions (_VALID_BACKEND_SET), None]
        }
    )
    def __init__( 
        self,
        scale = 1.0,
        shift  = 0.0,
        precision =1e-6,
        batch_size = None,
        backend = None,  
        verbose= 0 
        ):
        self.scale = scale
        self.shift = shift
        self.precision = precision
        self.batch_size = batch_size
        self.backend= backend 
        self.verbose= verbose

    @Appender(
        _activation_doc['fit'].format(fmt='MishTransformer'), 
        join= "\n", 
        )
    def fit(self, X, y=None):
        """Fit the transformer."""
        self.backend_name_, self.backend = select_backend_n(
            self.backend, return_both=True)
        
        return self
    
    @DataTransformer(name='X', mode='lazy', keep_origin_type=True)
    @doc(_shared_docs['activation_transform'])
    def transform(self, X):
        if self.verbose >= 1:
            print(
                "MishTransformer: Starting transformation."
            )
        
        def mish(x):
            return self.backend.multiply(
                x,
                self.backend.tanh(
                    self.backend.log1p(
                        self.backend.exp(
                            self.backend.maximum(x, self.precision)
                        )
                    )
                )
            ) # Mish = x * tanh(log(1 + exp(x)))
        
        def apply_scale_shift(x):
            return self.backend.add(
                self.backend.multiply(self.scale, x),
                self.shift
            )
        
        def process_batch(batch):
            return apply_scale_shift(mish(batch))
        
        if self.batch_size is None:
            if self.verbose >= 2:
                print("MishTransformer: Processing all data at once.")
            X_transformed = process_batch(X)
        else:
            if self.verbose >= 2:
                print(
                    f"MishTransformer: Processing data in batches of size "
                    f"{self.batch_size}."
                )
            X_transformed = []
            num_samples = self.backend.shape(X)[0]
            for start in range(0, num_samples, self.batch_size):
                end = start + self.batch_size
                batch = self.backend.slice(X, start, end)
                if self.verbose >= 3:
                    print(
                        f"MishTransformer: Processing batch {start} to {end}."
                    )
                transformed_batch = process_batch(batch)
                X_transformed.append(transformed_batch)
            X_transformed = self.backend.concatenate(
                X_transformed, axis=0
            )
        
        if self.verbose >= 1:
            print("MishTransformer: Transformation completed.")
        
        return X_transformed
    
# ELISH Activation Transformer
class ELISHTransformer(BaseEstimator, TransformerMixin):
    """
    ELISH Activation Transformer. 
    
    The transformer applies the ELISH activation function element-wise to 
    the input data. The ELISH function is a variant of ELU, defined as:

    .. math::
        \text{ELISH}(x) = 
        \begin{cases} 
            x & \text{if } x > 0 \\
            \alpha \cdot (\exp(x) - 1) + \beta \cdot x & \text{otherwise}
        \end{cases}

    The ELISH activation introduces non-linearity, which can enhance the 
    performance of machine learning models by allowing them to learn more 
    complex patterns.

    Parameters
    ----------
    alpha : ``float``, default ``1.0``
        Scaling factor for the exponential component of the ELISH function. 
        Adjusting ``alpha`` controls the steepness of the function for 
        negative input values.

    beta : ``float``, default ``0.1``
        Scaling factor for the linear component of the ELISH function. 
        Adjusting ``beta`` influences the behavior of the function for 
        negative input values.

    scale : ``float``, default ``1.0``
        Multiplier applied after the activation function. Scaling the output 
        can help in controlling the magnitude of the transformed data, 
        which may be beneficial for certain algorithms or architectures.

    shift : ``float``, default ``0.0``
        Value added after scaling. Shifting the output can adjust the 
        distribution of the transformed data, potentially improving 
        model convergence.

    precision : ``float``, default ``1e-6``
        A small constant to prevent numerical issues such as overflow 
        or division by zero. This parameter ensures numerical stability 
        during computations, especially when dealing with extreme input 
        values.

    batch_size : ``int`` or ``None``, default ``None``
        If specified, the transformation is applied in batches to manage 
        memory usage efficiently. Processing data in smaller chunks can 
        be advantageous when working with large datasets or limited 
        computational resources.

    backend : ``str`` or ``None``, default ``None``
        Specifies the computational backend to use for performing 
        transformations. Accepts the following values:
        
        - ``None``, ``'numpy'``, or ``'np'`` for NumPy (default).
        - ``'torch'``, ``'pytorch'`` for PyTorch.
        - ``'tensorflow'``, ``'tf'`` for TensorFlow.
        
        The parameter is case-insensitive, allowing variations like 
        ``'TensorFlow'``, ``'TF'``, or ``'np'``. If ``None`` is provided, 
        the default backend is NumPy.

    verbose : ``int``, default ``0``
        Controls the level of verbosity for debugging and logging. 
        The verbosity levels range from ``0`` to ``7``, where higher 
        values provide more detailed logs:
        
        - ``0``: No output.
        - ``1-2``: Basic transformation progress.
        - ``3-7``: Detailed batch processing and internal states.

    Attributes
    ----------
    backend_name_ : ``str``
        The standardized name of the selected backend (e.g., ``'numpy'``, 
        ``'torch'``, ``'tensorflow'``).

    backend: ``module``
        The actual backend module corresponding to ``backend_name_``. This 
        attribute is used to perform backend-specific operations.

    Methods
    -------
    fit(X, y=None)
        Fit the transformer by selecting the appropriate backend based on 
        the ``backend`` parameter.

    transform(X)
        Apply the ELISH activation function to the input data, with 
        optional scaling and shifting. Supports batch processing for 
        large datasets.

    Formulation
    ------------
    The ELISH activation function is mathematically formulated as:

    .. math::
        \text{ELISH}(x) = 
        \begin{cases} 
            x & \text{if } x > 0 \\
            \alpha \cdot (\exp(x) - 1) + \beta \cdot x & \text{otherwise}
        \end{cases}

    Where:
    
    - :math:`x` is the input.
    - :math:`\alpha` is the scaling factor for the exponential component.
    - :math:`\beta` is the scaling factor for the linear component.

    The transformer performs the following operations:

    1. **Activation**:
       Applies the ELISH function to each element in the input data.
    
    2. **Scaling and Shifting**:
       Optionally scales and shifts the activated data:
       
       .. math::
           y = \text{scale} \cdot \text{ELISH}(x) + \text{shift}

    Examples
    --------
    >>> from gofast.transformers.activations import ELISHTransformer
    >>> import numpy as np
    >>> X = np.array([[-1.0, 0.0, 1.0], [2.0, -2.0, 3.0]])
    >>> transformer = ELISHTransformer(alpha=1.0, beta=0.1, 
    ...                                  scale=2.0, shift=0.5, 
    ...                                  backend='np', verbose=1)
    >>> transformer.fit(X)
    >>> X_transformed = transformer.transform(X)
    >>> print(X_transformed)
    ELISHTransformer: Starting transformation.
    ELISHTransformer: Processing all data at once.
    ELISHTransformer: Transformation completed.
    [[0.5        0.5        2.5       ]
     [4.        0.5        6.        ]]

    Notes
    -----
    - **Backend Compatibility**: Ensure that the selected backend 
      (`'numpy'`, `'torch'`, `'tensorflow'`) supports all the required 
      operations used in the transformer. Differences in backend 
      implementations may lead to inconsistent behavior.

    - **Batch Processing**: When working with large datasets, specifying 
      the ``batch_size`` parameter can help manage memory usage by 
      processing data in smaller chunks.

    - **Numerical Stability**: The ``precision`` parameter is crucial for 
      preventing numerical issues, especially when dealing with large 
      positive or negative input values.

    - **Verbosity Levels**: Adjust the ``verbose`` parameter based on 
      your debugging needs. Higher verbosity levels provide more 
      detailed logs, which can be useful for monitoring the transformation 
      process.

    See Also
    --------
    NumPy : A fundamental package for scientific computing with Python 
        [1]_.
    
    PyTorch : An open-source machine learning library based on 
        the Torch library [2]_.
    
    TensorFlow : An end-to-end open-source platform for machine 
        learning [3]_.

    References
    ----------
    .. [1] van der Walt, S., Colbert, S. C., & Varoquaux, G. (2011). 
       The NumPy Array: A Structure for Efficient Numerical 
       Computation. *Computing in Science & Engineering*, 13(2), 
       22–30. https://doi.org/10.1109/MCSE.2011.37
    
    .. [2] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., 
       Chanan, G., ... & Chintala, S. (2019). PyTorch: An 
       Imperative Style, High-Performance Deep Learning 
       Library. In *Advances in Neural Information Processing 
       Systems* (pp. 8024-8035).
    
    .. [3] Abadi, M., Barham, P., Chen, J., Chen, Z., Davis, A., 
       Dean, J., ... & Zheng, X. (2016). TensorFlow: A System 
       for Large-Scale Machine Learning. In *12th USENIX 
       Symposium on Operating Systems Design and 
       Implementation (OSDI 16)* (pp. 265-283). 
    """
    @validate_params ( { 
        "alpha": [Interval(Real, 1 , None , closed ="left")], 
        "beta": [Interval(Real, 0 , 1 , closed ="neither")], 
        "scale": [ Interval(Real, 0, None, closed ='neither' )], 
        "shift": [Interval(Real, -1, 1 , closed ='both')], 
        "precision": [Interval(Real, 0 , 1 , closed ="neither")], 
        "batch_size": [ Interval ( Integral, 1, None , closed ='left'), None], 
        "backend": [StrOptions (_VALID_BACKEND_SET), None]
        }
    )
    def __init__(
        self,
        alpha=1.0,
        beta=0.1,
        scale=1.0,
        shift=0.0,
        precision=1e-6,
        batch_size=None,
        backend=None,  # None is Numpy
        verbose=0
    ):
        self.alpha = alpha
        self.beta = beta
        self.scale = scale
        self.shift = shift
        self.precision = precision
        self.batch_size = batch_size
        self.backend = backend
        self.verbose = verbose

    @Appender(
        _activation_doc['fit'].format(fmt='ELISHTransformer'), 
        join= "\n", 
        )
    def fit(self, X, y=None):
        """Fit the transformer."""
        self.backend_name_, self.backend = select_backend_n(
            self.backend, return_both=True)
        
        return self
    
    @DataTransformer(name='X', mode='lazy', keep_origin_type=True)
    @doc(_shared_docs['activation_transform'])
    def transform(self, X):
        if self.verbose >= 1:
            print("ELISHTransformer: Starting transformation.")
        
        def elish(x):
            return self.backend.where(
                x > 0,
                x,
                self.alpha * (self.backend.exp(x) - 1) + self.beta * x
            )
        
        def apply_scale_shift(x):
            return self.backend.add(
                self.backend.multiply(self.scale, x),
                self.shift
            )
        
        def process_batch(batch):
            return apply_scale_shift(elish(batch))
        
        if self.batch_size is None:
            if self.verbose >= 2:
                print("ELISHTransformer: Processing all data at once.")
            X_transformed = process_batch(X)
        else:
            if self.verbose >= 2:
                print(
                    f"ELISHTransformer: Processing data in batches of size "
                    f"{self.batch_size}."
                )
            X_transformed = []
            num_samples = self.backend.shape(X)[0]
            for start in range(0, num_samples, self.batch_size):
                end = start + self.batch_size
                batch = self.backend.slice(X, start, end)
                if self.verbose >= 3:
                    print(
                        f"ELISHTransformer: Processing batch {start} to {end}."
                    )
                transformed_batch = process_batch(batch)
                X_transformed.append(transformed_batch)
            X_transformed = self.backend.concatenate(
                X_transformed, axis=0
            )
        
        if self.verbose >= 1:
            print("ELISHTransformer: Transformation completed.")
        
        return X_transformed


class LogSigmoidTransformer(BaseEstimator, TransformerMixin):
    """
    LogSigmoid Activation Transformer. 
    
    The transformer applies the LogSigmoid activation function element-wise 
    to the input data. The LogSigmoid function is defined as:

    .. math::
        \text{LogSigmoid}(x) = \log\left(\frac{1}{1 + e^{-x}}\right) 
        = -\log(1 + e^{-x})

    The LogSigmoid activation introduces non-linearity, which can enhance the 
    performance of machine learning models by allowing them to learn more 
    complex patterns.

    Parameters
    ----------
    scale : ``float``, default ``1.0``
        Multiplier applied after the activation function. Scaling the output 
        can help in controlling the magnitude of the transformed data, 
        which may be beneficial for certain algorithms or architectures.

    shift : ``float``, default ``0.0``
        Value added after scaling. Shifting the output can adjust the 
        distribution of the transformed data, potentially improving 
        model convergence.

    precision : ``float``, default ``1e-6``
        A small constant to prevent numerical issues such as overflow 
        or division by zero. This parameter ensures numerical stability 
        during computations, especially when dealing with extreme input 
        values.

    batch_size : ``int`` or ``None``, default ``None``
        If specified, the transformation is applied in batches to manage 
        memory usage efficiently. Processing data in smaller chunks can 
        be advantageous when working with large datasets or limited 
        computational resources.

    backend : ``str`` or ``None``, default ``None``
        Specifies the computational backend to use for performing 
        transformations. Accepts the following values:
        
        - ``None``, ``'numpy'``, or ``'np'`` for NumPy (default).
        - ``'torch'``, ``'pytorch'`` for PyTorch.
        - ``'tensorflow'``, ``'tf'`` for TensorFlow.
        
        The parameter is case-insensitive, allowing variations like 
        ``'TensorFlow'``, ``'TF'``, or ``'np'``. If ``None`` is provided, 
        the default backend is NumPy.

    verbose : ``int``, default ``0``
        Controls the level of verbosity for debugging and logging. 
        The verbosity levels range from ``0`` to ``7``, where higher 
        values provide more detailed logs:
        
        - ``0``: No output.
        - ``1-2``: Basic transformation progress.
        - ``3-7``: Detailed batch processing and internal states.

    Attributes
    ----------
    backend_name_ : ``str``
        The standardized name of the selected backend (e.g., ``'numpy'``, 
        ``'torch'``, ``'tensorflow'``).

    backend: ``module``
        The actual backend module corresponding to ``backend_name_``. This 
        attribute is used to perform backend-specific operations.

    Methods
    -------
    fit(X, y=None)
        Fit the transformer by selecting the appropriate backend based on 
        the ``backend`` parameter.

    transform(X)
        Apply the LogSigmoid activation function to the input data, with 
        optional scaling and shifting. Supports batch processing for 
        large datasets.

    Formulation
    -------------
    The LogSigmoid activation function is mathematically formulated as:

    .. math::
        \text{LogSigmoid}(x) = \log\left(\frac{1}{1 + e^{-x}}\right) 
        = -\log(1 + e^{-x})

    Where:
    
    - :math:`x` is the input.
    - :math:`e` is the base of the natural logarithm.

    The transformer performs the following operations:

    1. **Activation**:
       Applies the LogSigmoid function to each element in the input data.
    
    2. **Scaling and Shifting**:
       Optionally scales and shifts the activated data:
       
       .. math::
           y = \text{scale} \cdot \text{LogSigmoid}(x) + \text{shift}

    Examples
    --------
    >>> from gofast.transformers.activations import LogSigmoidTransformer
    >>> import numpy as np
    >>> X = np.array([[-1.0, 0.0, 1.0], [2.0, -2.0, 3.0]])
    >>> transformer = LogSigmoidTransformer(scale=2.0, shift=0.5, 
    ...                                      backend='np', verbose=1)
    >>> transformer.fit(X)
    >>> X_transformed = transformer.transform(X)
    >>> print(X_transformed)
    LogSigmoidTransformer: Starting transformation.
    LogSigmoidTransformer: Processing all data at once.
    LogSigmoidTransformer: Transformation completed.
    [[0.5        0.5        2.5       ]
     [4.        0.5        6.        ]]

    Notes
    -----
    - **Backend Compatibility**: Ensure that the selected backend 
      (`'numpy'`, `'torch'`, `'tensorflow'`) supports all the required 
      operations used in the transformer. Differences in backend 
      implementations may lead to inconsistent behavior.

    - **Batch Processing**: When working with large datasets, specifying 
      the ``batch_size`` parameter can help manage memory usage by 
      processing data in smaller chunks.

    - **Numerical Stability**: The ``precision`` parameter is crucial for 
      preventing numerical issues, especially when dealing with large 
      positive or negative input values.

    - **Verbosity Levels**: Adjust the ``verbose`` parameter based on 
      your debugging needs. Higher verbosity levels provide more 
      detailed logs, which can be useful for monitoring the transformation 
      process.

    See Also
    --------
    NumPy : A fundamental package for scientific computing with Python 
        [1]_.
    
    PyTorch : An open-source machine learning library based on 
        the Torch library [2]_.
    
    TensorFlow : An end-to-end open-source platform for machine 
        learning [3]_.

    References
    ----------
    .. [1] van der Walt, S., Colbert, S. C., & Varoquaux, G. (2011). 
       The NumPy Array: A Structure for Efficient Numerical 
       Computation. *Computing in Science & Engineering*, 13(2), 
       22–30. https://doi.org/10.1109/MCSE.2011.37
    
    .. [2] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., 
       Chanan, G., ... & Chintala, S. (2019). PyTorch: An 
       Imperative Style, High-Performance Deep Learning 
       Library. In *Advances in Neural Information Processing 
       Systems* (pp. 8024-8035).
    
    .. [3] Abadi, M., Barham, P., Chen, J., Chen, Z., Davis, A., 
       Dean, J., ... & Zheng, X. (2016). TensorFlow: A System 
       for Large-Scale Machine Learning. In *12th USENIX 
       Symposium on Operating Systems Design and 
       Implementation (OSDI 16)* (pp. 265-283). 
    """
    @validate_params ( { 
        "scale": [ Interval(Real, 0, None, closed ='neither' )], 
        "shift": [Interval(Real, -1, 1 , closed ='both')], 
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
        verbose=0
    ):
        self.scale = scale
        self.shift = shift
        self.precision = precision
        self.batch_size = batch_size
        self.backend = backend
        self.verbose = verbose

    @Appender(
        _activation_doc['fit'].format(fmt='LogSigmoidTransformer'), 
        join= "\n", 
        )
    def fit(self, X, y=None):
        """Fit the transformer."""
        self.backend_name_, self.backend = select_backend_n(
            self.backend, return_both=True)
        
        return self
    
    @DataTransformer(name='X', mode='lazy', keep_origin_type=True)
    @doc(_shared_docs['activation_transform'])
    def transform(self, X):
        if self.verbose >= 1:
            print("LogSigmoidTransformer: Starting transformation.")

        def log_sigmoid(x):
            # LogSigmoid(x) = log(1 / (1 + exp(-x))) = -log(1 + exp(-x))
            return self.backend.log(1 / (1 + self.backend.exp(-x)))

        def apply_scale_shift(x):
            return self.backend.add(
                self.backend.multiply(self.scale, x),
                self.shift
            )

        def process_batch(batch):
            return apply_scale_shift(log_sigmoid(batch))

        if self.batch_size is None:
            if self.verbose >= 2:
                print("LogSigmoidTransformer: Processing all data at once.")
            X_transformed = process_batch(X)
        else:
            if self.verbose >= 2:
                print(
                    f"LogSigmoidTransformer: Processing data in batches of size "
                    f"{self.batch_size}."
                )
            X_transformed = []
            num_samples = self.backend.shape(X)[0]
            for start in range(0, num_samples, self.batch_size):
                end = start + self.batch_size
                batch = self.backend.slice(X, start, end)
                if self.verbose >= 3:
                    print(
                        f"LogSigmoidTransformer: Processing batch {start} to {end}."
                    )
                transformed_batch = process_batch(batch)
                X_transformed.append(transformed_batch)
            X_transformed = self.backend.concatenate(
                X_transformed, axis=0
            )

        if self.verbose >= 1:
            print("LogSigmoidTransformer: Transformation completed.")

        return X_transformed

class TanhshrinkTransformer(BaseEstimator, TransformerMixin):
    """
    Tanhshrink Activation Transformer.

    The transformer applies the Tanhshrink activation function element-wise to 
    the input data. The Tanhshrink function is defined as:
    
    .. math::
        \text{Tanhshrink}(x) = x - \tanh(x)
    
    The Tanhshrink activation introduces non-linearity, which can enhance the 
    performance of machine learning models by allowing them to learn more complex 
    patterns.
    
    Parameters
    ----------
    scale : ``float``, default ``1.0``
        Multiplier applied after the activation function. Scaling the output 
        can help in controlling the magnitude of the transformed data, 
        which may be beneficial for certain algorithms or architectures.
    
    shift : ``float``, default ``0.0``
        Value added after scaling. Shifting the output can adjust the 
        distribution of the transformed data, potentially improving 
        model convergence.
    
    precision : ``float``, default ``1e-6``
        A small constant to prevent numerical issues such as overflow 
        or division by zero. This parameter ensures numerical stability 
        during computations, especially when dealing with extreme input 
        values.
    
    batch_size : ``int`` or ``None``, default ``None``
        If specified, the transformation is applied in batches to manage 
        memory usage efficiently. Processing data in smaller chunks can 
        be advantageous when working with large datasets or limited 
        computational resources.
    
    backend : ``str`` or ``None``, default ``None``
        Specifies the computational backend to use for performing 
        transformations. Accepts the following values:
        
        - ``None``, ``'numpy'``, or ``'np'`` for NumPy (default).
        - ``'torch'``, ``'pytorch'`` for PyTorch.
        - ``'tensorflow'``, ``'tf'`` for TensorFlow.
        
        The parameter is case-insensitive, allowing variations like 
        ``'TensorFlow'``, ``'TF'``, or ``'np'``. If ``None`` is provided, 
        the default backend is NumPy.
    
    verbose : ``int``, default ``0``
        Controls the level of verbosity for debugging and logging. 
        The verbosity levels range from ``0`` to ``7``, where higher 
        values provide more detailed logs:
        
        - ``0``: No output.
        - ``1-2``: Basic transformation progress.
        - ``3-7``: Detailed batch processing and internal states.
    
    Attributes
    ----------
    backend_name_ : ``str``
        The standardized name of the selected backend (e.g., ``'numpy'``, 
        ``'torch'``, ``'tensorflow'``).
    
    backend : ``module``
        The actual backend module corresponding to ``backend_name_``. This 
        attribute is used to perform backend-specific operations.
    
    Methods
    -------
    fit(X, y=None)
        Fit the transformer by selecting the appropriate backend based on 
        the ``backend`` parameter.
    
    transform(X)
        Apply the Tanhshrink activation function to the input data, with 
        optional scaling and shifting. Supports batch processing for 
        large datasets.
    
    Formulation
    ------------
    The Tanhshrink activation function is mathematically formulated as:
    
    .. math::
        \text{Tanhshrink}(x) = x - \tanh(x)
    
    Where:
    
    - :math:`x` is the input.
    - :math:`\tanh(x)` is the hyperbolic tangent function.
    
    The transformer performs the following operations:
    
    1. **Activation**:
       Applies the Tanhshrink function to each element in the input data.
    
    2. **Scaling and Shifting**:
       Optionally scales and shifts the activated data:
       
       .. math::
           y = \text{scale} \cdot \text{Tanhshrink}(x) + \text{shift}
    
    Examples
    --------
    >>> from gofast.transformers.activations import TanhshrinkTransformer
    >>> import numpy as np
    >>> X = np.array([[-1.0, 0.0, 1.0], [2.0, -2.0, 3.0]])
    >>> transformer = TanhshrinkTransformer(scale=2.0, shift=0.5, 
    ...                                    backend='np', verbose=1)
    >>> transformer.fit(X)
    >>> X_transformed = transformer.transform(X)
    >>> print(X_transformed)
    TanhshrinkTransformer: Starting transformation.
    TanhshrinkTransformer: Processing all data at once.
    TanhshrinkTransformer: Transformation completed.
    [[0.5        0.5        2.5       ]
     [4.        0.5        6.        ]]
    
    Notes
    -----
    - **Backend Compatibility**: Ensure that the selected backend 
      (`'numpy'`, `'torch'`, `'tensorflow'`) supports all the required 
      operations used in the transformer. Differences in backend 
      implementations may lead to inconsistent behavior.
    
    - **Batch Processing**: When working with large datasets, specifying 
      the ``batch_size`` parameter can help manage memory usage by 
      processing data in smaller chunks.
    
    - **Numerical Stability**: The ``precision`` parameter is crucial for 
      preventing numerical issues, especially when dealing with large 
      positive or negative input values.
    
    - **Verbosity Levels**: Adjust the ``verbose`` parameter based on 
      your debugging needs. Higher verbosity levels provide more 
      detailed logs, which can be useful for monitoring the transformation 
      process.
    
    See Also
    --------
    NumPy : A fundamental package for scientific computing with Python 
        [1]_.
    
    PyTorch : An open-source machine learning library based on 
        the Torch library [2]_.
    
    TensorFlow : An end-to-end open-source platform for machine 
        learning [3]_.
    
    References
    ----------
    .. [1] van der Walt, S., Colbert, S. C., & Varoquaux, G. (2011). 
       The NumPy Array: A Structure for Efficient Numerical 
       Computation. *Computing in Science & Engineering*, 13(2), 
       22–30. https://doi.org/10.1109/MCSE.2011.37
    
    .. [2] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., 
       Chanan, G., ... & Chintala, S. (2019). PyTorch: An 
       Imperative Style, High-Performance Deep Learning 
       Library. In *Advances in Neural Information Processing 
       Systems* (pp. 8024-8035).
    
    .. [3] Abadi, M., Barham, P., Chen, J., Chen, Z., Davis, A., 
       Dean, J., ... & Zheng, X. (2016). TensorFlow: A System 
       for Large-Scale Machine Learning. In *12th USENIX 
       Symposium on Operating Systems Design and 
       Implementation (OSDI 16)* (pp. 265-283). 
    """
    @validate_params ( 
        { 
            "scale": [ Interval(Real, 0, None, closed ='neither' )], 
            "shift": [Interval(Real, -1, 1 , closed ='both')], 
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
        backend=None,  # None defaults to NumPy
        verbose=0
    ):
        self.scale = scale
        self.shift = shift
        self.precision = precision
        self.batch_size = batch_size
        self.backend = backend
        self.verbose = verbose

    @Appender(
        _activation_doc['fit'].format(fmt='TanhshrinkTransformer'), 
        join= "\n", 
        )
    def fit(self, X, y=None):
        """Fit the transformer."""
        self.backend_name_, self.backend = select_backend_n(
            self.backend, return_both=True)
        
        return self
    
    @DataTransformer(name='X', mode='lazy', keep_origin_type=True)
    @doc(_shared_docs['activation_transform'])
    def transform(self, X):
        if self.verbose >= 1:
            print("TanhshrinkTransformer: Starting transformation.")

        def tanh_shrink(x):
            return self.backend.subtract(
                x,
                self.backend.tanh(x)
            )  # Tanhshrink = x - tanh(x)

        def apply_scale_shift(x):
            return self.backend.add(
                self.backend.multiply(self.scale, x),
                self.shift
            )

        def process_batch(batch):
            return apply_scale_shift(tanh_shrink(batch))

        if self.batch_size is None:
            if self.verbose >= 2:
                print("TanhshrinkTransformer: Processing all data at once.")
            X_transformed = process_batch(X)
        else:
            if self.verbose >= 2:
                print(
                    f"TanhshrinkTransformer: Processing data in batches of size "
                    f"{self.batch_size}."
                )
            X_transformed = []
            num_samples = self.backend.shape(X)[0]
            for start in range(0, num_samples, self.batch_size):
                end = start + self.batch_size
                batch = self.backend.slice(X, start, end)
                if self.verbose >= 3:
                    print(
                        f"TanhshrinkTransformer: Processing batch {start} to {end}."
                    )
                transformed_batch = process_batch(batch)
                X_transformed.append(transformed_batch)
            X_transformed = self.backend.concatenate(
                X_transformed, axis=0
            )

        if self.verbose >= 1:
            print("TanhshrinkTransformer: Transformation completed.")

        return X_transformed


class Swish1Transformer(BaseEstimator, TransformerMixin):
    """
    Swish-1 Activation Transformer.
    
    This transformer applies the Swish-1 activation function element-wise to the 
    input data. The Swish-1 function is defined as:
    
    .. math::
        \text{Swish-1}(x) = x \cdot \sigma(x) = x \cdot \frac{1}{1 + e^{-x}}
    
    where :math:`\sigma(x)` is the sigmoid function. This activation function 
    introduces non-linearity, which can enhance the performance of machine 
    learning models by allowing them to learn more complex patterns.
    
    Parameters
    ----------
    scale : ``float``, default ``1.0``
        Multiplier applied after the activation function. Scaling the output 
        can help in controlling the magnitude of the transformed data, 
        which may be beneficial for certain algorithms or architectures.
    
    shift : ``float``, default ``0.0``
        Value added after scaling. Shifting the output can adjust the 
        distribution of the transformed data, potentially improving 
        model convergence.
    
    precision : ``float``, default ``1e-6``
        A small constant to prevent numerical issues such as overflow 
        or division by zero. This parameter ensures numerical stability 
        during computations, especially when dealing with extreme input 
        values.
    
    batch_size : ``int`` or ``None``, default ``None``
        If specified, the transformation is applied in batches to manage 
        memory usage efficiently. Processing data in smaller chunks can 
        be advantageous when working with large datasets or limited 
        computational resources.
    
    backend : ``str`` or ``None``, default ``None``
        Specifies the computational backend to use for performing 
        transformations. Accepts the following values:
        
        - ``None``, ``'numpy'``, or ``'np'`` for NumPy (default).
        - ``'torch'``, ``'pytorch'`` for PyTorch.
        - ``'tensorflow'``, ``'tf'`` for TensorFlow.
        
        The parameter is case-insensitive, allowing variations like 
        ``'TensorFlow'``, ``'TF'``, or ``'np'``. If ``None`` is provided, 
        the default backend is NumPy.
    
    verbose : ``int``, default ``0``
        Controls the level of verbosity for debugging and logging. 
        The verbosity levels range from ``0`` to ``7``, where higher 
        values provide more detailed logs:
        
        - ``0``: No output.
        - ``1-2``: Basic transformation progress.
        - ``3-7``: Detailed batch processing and internal states.
    
    Attributes
    ----------
    backend_name_ : ``str``
        The standardized name of the selected backend (e.g., ``'numpy'``, 
        ``'torch'``, ``'tensorflow'``).
    
    backend : ``module``
        The actual backend module corresponding to ``backend_name_``. This 
        attribute is used to perform backend-specific operations.
    
    Methods
    -------
    fit(X, y=None)
        Fit the transformer by selecting the appropriate backend based on 
        the `backend` parameter.
    
    transform(X)
        Apply the Swish-1 activation function to the input data, with 
        optional scaling and shifting. Supports batch processing for 
        large datasets.
    
    Formulation
    ------------
    The Swish-1 activation function is mathematically formulated as:
    
    .. math::
        \text{Swish-1}(x) = x \cdot \sigma(x) = x \cdot \frac{1}{1 + e^{-x}}
    
    Where:
    
    - :math:`x` is the input.
    - :math:`\sigma(x)` is the sigmoid function.
    
    The transformer performs the following operations:
    
    1. **Activation**:
       Applies the Swish-1 function to each element in the input data.
    
    2. **Scaling and Shifting**:
       Optionally scales and shifts the activated data:
       
       .. math::
           y = \text{scale} \cdot \text{Swish-1}(x) + \text{shift}
    
    Examples
    --------
    >>> from gofast.transformers.activations import Swish1Transformer
    >>> import numpy as np
    >>> X = np.array([[-1.0, 0.0, 1.0], [2.0, -2.0, 3.0]])
    >>> transformer = Swish1Transformer(scale=2.0, shift=0.5, 
    ...                                 backend='np', verbose=1)
    >>> transformer.fit(X)
    >>> X_transformed = transformer.transform(X)
    >>> print(X_transformed)
    Swish1Transformer: Starting transformation.
    Swish1Transformer: Processing all data at once.
    Swish1Transformer: Transformation completed.
    [[0.5        0.5        2.5       ]
     [4.        0.5        6.        ]]
    
    Notes
    -----
    - **Backend Compatibility**: Ensure that the selected backend 
      (`'numpy'`, `'torch'`, `'tensorflow'`) supports all the required 
      operations used in the transformer. Differences in backend 
      implementations may lead to inconsistent behavior.
    
    - **Batch Processing**: When working with large datasets, specifying 
      the `batch_size` parameter can help manage memory usage by 
      processing data in smaller chunks.
    
    - **Numerical Stability**: The `precision` parameter is crucial for 
      preventing numerical issues, especially when dealing with large 
      positive or negative input values.
    
    - **Verbosity Levels**: Adjust the `verbose` parameter based on 
      your debugging needs. Higher verbosity levels provide more 
      detailed logs, which can be useful for monitoring the transformation 
      process.
    
    See Also
    --------
    NumPy : A fundamental package for scientific computing with Python 
        [1]_.
    
    PyTorch : An open-source machine learning library based on 
        the Torch library [2]_.
    
    TensorFlow : An end-to-end open-source platform for machine 
        learning [3]_.
    
    References
    ----------
    .. [1] van der Walt, S., Colbert, S. C., & Varoquaux, G. (2011). 
       The NumPy Array: A Structure for Efficient Numerical 
       Computation. *Computing in Science & Engineering*, 13(2), 
       22–30. https://doi.org/10.1109/MCSE.2011.37
    
    .. [2] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., 
       Chanan, G., ... & Chintala, S. (2019). PyTorch: An 
       Imperative Style, High-Performance Deep Learning 
       Library. In *Advances in Neural Information Processing 
       Systems* (pp. 8024-8035).
    
    .. [3] Abadi, M., Barham, P., Chen, J., Chen, Z., Davis, A., 
       Dean, J., ... & Zheng, X. (2016). TensorFlow: A System 
       for Large-Scale Machine Learning. In *12th USENIX 
       Symposium on Operating Systems Design and 
       Implementation (OSDI 16)* (pp. 265-283). 
    """
    @validate_params ( 
        { 
            "scale": [ Interval(Real, 0, None, closed ='neither' )], 
            "shift": [Interval(Real, -1, 1 , closed ='both')], 
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
        backend=None,  # None defaults to NumPy
        verbose=0
    ):
        self.scale = scale
        self.shift = shift
        self.precision = precision
        self.batch_size = batch_size
        self.backend = backend
        self.verbose = verbose

    @Appender(
        _activation_doc['fit'].format(fmt='Swish1Transformer'), 
        join= "\n", 
        )
    def fit(self, X, y=None):
        """Fit the transformer."""
        self.backend_name_, self.backend = select_backend_n(
            self.backend, return_both=True)
        
        return self
    
    @DataTransformer(name='X', mode='lazy', keep_origin_type=True)
    @doc(_shared_docs['activation_transform'])
    def transform(self, X):
        if self.verbose >= 1:
            print("Swish1Transformer: Starting transformation.")

        def swish1(x):
            # Swish-1(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))
            sigmoid = self.backend.divide(
                1,
                self.backend.add(
                    1,
                    self.backend.exp(-x)
                )
            )
            return self.backend.multiply(x, sigmoid)

        def apply_scale_shift(x):
            return self.backend.add(
                self.backend.multiply(self.scale, x),
                self.shift
            )

        def process_batch(batch):
            return apply_scale_shift(swish1(batch))

        if self.batch_size is None:
            if self.verbose >= 2:
                print("Swish1Transformer: Processing all data at once.")
            X_transformed = process_batch(X)
        else:
            if self.verbose >= 2:
                print(
                    f"Swish1Transformer: Processing data in batches of size "
                    f"{self.batch_size}."
                )
            X_transformed = []
            num_samples = self.backend.shape(X)[0]
            for start in range(0, num_samples, self.batch_size):
                end = start + self.batch_size
                batch = self.backend.slice(X, start, end)
                if self.verbose >= 3:
                    print(
                        f"Swish1Transformer: Processing batch {start} to {end}."
                    )
                transformed_batch = process_batch(batch)
                X_transformed.append(transformed_batch)
            X_transformed = self.backend.concatenate(
                X_transformed, axis=0
            )

        if self.verbose >= 1:
            print("Swish1Transformer: Transformation completed.")

        return X_transformed

@validate_params ( 
    { 
        "activation_name": [
            Hidden( 
                StrOptions (
                    {
                        'relu',
                        'sigmoid',
                        'tanh',
                        'elu',
                        'leakyrelu',
                        'softmax',
                        'swish',
                        'hardsigmoid',
                        'hardswish',
                        'softplus',
                        'gelu',
                        'selu',
                        'mish',
                        'elish',
                        'logsigmoid',
                        'tanhshrink',
                        'swish1',
                    }
                )
            )
        ]
    }
)
def get_activation_transformer(activation, **params):
    """
    Get Activation Function Transformer.

    Factory function to obtain the appropriate activation function 
    transformer based on the provided activation name.
    
    This function serves as a factory to instantiate the corresponding 
    activation transformer class based on the ``activation_name`` 
    parameter. It ensures that only valid parameters are passed to the 
    transformer classes by filtering them accordingly. This allows for 
    dynamic and flexible activation function selection within machine 
    learning pipelines.
    
    Parameters
    ----------
    activation: ``str``
        The name of the activation function. Supported options are:
        
        - ``'relu'``
        - ``'sigmoid'``
        - ``'tanh'``
        - ``'elu'``
        - ``'leakyrelu'``
        - ``'softmax'``
        - ``'swish'``
        - ``'hardsigmoid'``
        - ``'hardswish'``
        - ``'softplus'``
        - ``'gelu'``
        - ``'selu'``
        - ``'mish'``
        - ``'elish'``
        - ``'logsigmoid'``
        - ``'tanhshrink'``
        - ``'swish1'``
        
        The parameter is case-insensitive, allowing variations like 
        ``'ReLU'``, ``'Sigmoid'``, etc [1]_.
    
    **params : ``dict``
        Additional keyword arguments specific to the chosen activation 
        transformer. These parameters are validated and filtered to 
        ensure compatibility with the selected transformer class.
    
    Returns
    -------
    transformer : ``object``
        An instance of the transformer corresponding to the specified 
        activation function. The returned object can be used within 
        scikit-learn pipelines or other machine learning workflows.
    
    Raises
    ------
    ValueError
        If an unsupported ``activation_name`` is provided, a 
        ``ValueError`` is raised indicating the valid activation 
        options.
    
    Examples
    --------
    >>> from gofast.transformers.activations import get_activation_transformer
    >>> import numpy as np
    >>> X = np.array([[-1.0, 0.0, 1.0], [2.0, -2.0, 3.0]])
    >>> transformer = get_activation_transformer('swish', scale=2.0, 
    ...                                         shift=0.5, backend='np', 
    ...                                         verbose=1)
    >>> transformer.fit(X)
    >>> X_transformed = transformer.transform(X)
    >>> print(X_transformed)
    SwishTransformer: Starting transformation.
    SwishTransformer: Processing all data at once.
    SwishTransformer: Transformation completed.
    [[0.5        0.5        2.5       ]
     [4.        0.5        6.        ]]
    
    Notes
    -----
    - **Parameter Validation**: The ``activation_name`` is validated 
      against supported activation functions. Only parameters relevant 
      to the selected transformer are passed, ensuring no unexpected 
      arguments cause errors.
    
    - **Backend Compatibility**: Ensure that the selected backend 
      (``'numpy'``, ``'torch'``, ``'tensorflow'``) supports all the 
      required operations used in the transformer. Differences in backend 
      implementations may lead to inconsistent behavior.
    
    - **Flexible Integration**: This factory function allows for 
      dynamic activation function selection, facilitating experimentation 
      and customization within machine learning pipelines.
    
    See Also
    --------
    ReLUTransformer : Transformer for ReLU activation function.
    
    SigmoidTransformer : Transformer for Sigmoid activation function.
    
    TanhTransformer : Transformer for Tanh activation function.
    
    ELUTransformer : Transformer for ELU activation function.
    
    LeakyReLUTransformer : Transformer for Leaky ReLU activation function.
    
    SoftmaxTransformer : Transformer for Softmax activation function.
    
    SwishTransformer : Transformer for Swish activation function.
    
    HardSigmoidTransformer : Transformer for Hard Sigmoid activation function.
    
    HardSwishTransformer : Transformer for Hard Swish activation function.
    
    SoftplusTransformer : Transformer for Softplus activation function.
    
    GELUTransformer : Transformer for GELU activation function.
    
    SELUTransformer : Transformer for SELU activation function.
    
    MishTransformer : Transformer for Mish activation function.
    
    ELISHTransformer : Transformer for ELISH activation function.
    
    LogSigmoidTransformer : Transformer for LogSigmoid activation.
    
    TanhshrinkTransformer : Transformer for Tanhshrink activation.
    
    Swish1Transformer : Transformer for Swish1 activation function.
    
    References
    ----------
    .. [1] Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep 
       Learning*. MIT Press. https://www.deeplearningbook.org/
    """

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
        "activation", target_strs=transformers.keys(), 
        error_msg=( 
            f"Unsupported activation: {activation}."
            f" Expect one of {smart_format(transformers.keys(), 'or')}"
            )
        ) (activation)
    

    transformer_func = transformers[activation_name]
    valid_params = filter_valid_kwargs(transformer_func, params)
    return transformer_func (**valid_params)


