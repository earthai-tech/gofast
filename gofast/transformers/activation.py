# -*- coding: utf-8 -*-
"""
Activation function transformers for scikit-learn.
These transformers apply activation functions such as ReLU, Sigmoid, 
Tanh, ELU, Leaky ReLU, and Softmax element-wise to input data, 
and follow the scikit-learn API for custom transformers.

Each transformer supports the `fit` and `transform` methods as required by 
scikit-learn's `TransformerMixin`.
"""
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from gofast.decorators import Substitution, Appender  # Custom decorators for docstrings
#XXX TODO 

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

class ExampleTransformer(BaseEstimator, TransformerMixin):
    """
    Example Activation Function Transformer.

    This transformer applies an example activation function element-wise to the input data.

    Attributes
    ----------
    alpha : float, default=1.0
        The scaling factor for the transformation.

    Methods
    -------
    fit(X, y=None)
        This method is required by the scikit-learn API. It prepares the transformer 
        for use but does not change any internal state for this specific transformer.

    transform(X)
        Applies the activation function to the input data `X`.

    """

    def fit(self, X, y=None):
        """
        Fit the transformer to the data.

        This method is required by the scikit-learn API, but it does not alter the 
        internal state for this transformer.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data to fit. It is an array or matrix of shape 
            (n_samples, n_features) where `n_samples` is the number of samples 
            and `n_features` is the number of features for each sample.

        y : array-like, shape (n_samples,), optional, default=None
            The target labels for supervised learning. This parameter is not used 
            by this transformer, but it is required by the scikit-learn API for 
            consistency with other models. `y` can be ignored.

        Returns
        -------
        self : object
            Returns the instance of the transformer itself, allowing for 
            method chaining (e.g., `transformer.fit(X).transform(X)`).

        """

        return self

    def transform(self, X):
        """
        Apply the activation function element-wise to the input data.

        This method transforms the input data by applying the activation function.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data to transform. It should be an array or matrix of shape 
            (n_samples, n_features), where `n_samples` is the number of samples 
            and `n_features` is the number of features for each sample.

        Returns
        -------
        X_transformed : array-like, shape (n_samples, n_features)
            The transformed data, with the same shape as the input `X`. The 
            transformation is applied element-wise to each feature in the input data.
        
        Notes
        -----
        The input data `X` should be numeric (e.g., int, float). If `X` contains 
        non-numeric data types, the transformation may fail or produce incorrect results.
        """
        
        # Example activation function: return X * alpha (scaled version)
        return X * self.alpha

# Shared documentation for transformers to avoid redundancy.
_shared_docs = {
    "X": "array-like, shape (n_samples, n_features) The input data."
}


# ReLU Activation Transformer
class ReLUTransformer(BaseEstimator, TransformerMixin):
    """
    ReLU (Rectified Linear Unit) Transformer.

    This transformer applies the ReLU activation function element-wise 
    to the input data. The ReLU function returns the input for positive 
    values and zero for negative values.
    """
    
    def __init__(self):
        pass  # No parameters to initialize
    
    def fit(self, X, y=None):
        """
        The fit method is a no-op for ReLU, as it has no learnable parameters.

        Parameters:
        X: array-like, shape (n_samples, n_features)
            The input data.
        y: Ignored
            This parameter is not needed.

        Returns:
        self: object
            Returns the transformer instance itself.
        """
        return self
    
    def transform(self, X):
        """
        Apply the ReLU activation function element-wise.

        Parameters:
        X: array-like, shape (n_samples, n_features)
            The input data to be transformed.

        Returns:
        X_transformed: array-like, shape (n_samples, n_features)
            The transformed data after applying ReLU.
        """
        return np.maximum(0, X)

# Sigmoid Activation Transformer
class SigmoidTransformer(BaseEstimator, TransformerMixin):
    """
    Sigmoid Activation Transformer.

    This transformer applies the Sigmoid activation function element-wise 
    to the input data. The Sigmoid function maps input to a range between 
    0 and 1 using the formula: 1 / (1 + exp(-X)).
    """
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return 1 / (1 + np.exp(-X))

# Tanh Activation Transformer
class TanhTransformer(BaseEstimator, TransformerMixin):
    """
    Tanh Activation Transformer.

    This transformer applies the Tanh activation function element-wise 
    to the input data. The Tanh function maps input to a range between 
    -1 and 1.
    """
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.tanh(X)

# ELU Activation Transformer (Exponential Linear Unit)
class ELUTransformer(BaseEstimator, TransformerMixin):
    """
    ELU (Exponential Linear Unit) Transformer.

    This transformer applies the ELU activation function element-wise 
    to the input data. ELU is defined as:
    - ELU(x) = x for x > 0
    - ELU(x) = alpha * (exp(x) - 1) for x <= 0
    where alpha is a user-defined parameter.
    """
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.where(X > 0, X, self.alpha * (np.exp(X) - 1))

# Leaky ReLU Activation Transformer
class LeakyReLUTransformer(BaseEstimator, TransformerMixin):
    """
    Leaky ReLU Activation Transformer.

    This transformer applies the Leaky ReLU activation function element-wise 
    to the input data. Leaky ReLU allows a small, non-zero gradient when 
    the input is negative, defined as:
    - LeakyReLU(x) = x for x > 0
    - LeakyReLU(x) = alpha * x for x <= 0
    where alpha is a user-defined parameter.
    """
    
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.where(X > 0, X, self.alpha * X)

# Softmax Activation Transformer
class SoftmaxTransformer(BaseEstimator, TransformerMixin):
    """
    Softmax Activation Transformer.

    This transformer applies the Softmax activation function element-wise 
    to the input data. The Softmax function normalizes input to a probability 
    distribution across classes, and is commonly used in classification tasks.
    """
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))  # Numerical stability
        return exp_X / np.sum(exp_X, axis=1, keepdims=True)


# Swish Activation Transformer
class SwishTransformer(BaseEstimator, TransformerMixin):
    """
    Swish Activation Transformer.

    This transformer applies the Swish activation function element-wise 
    to the input data. The Swish function is defined as:
    - Swish(x) = x * sigmoid(x)
    """
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X * (1 / (1 + np.exp(-X)))  # Swish = x * sigmoid(x)

# Hard Sigmoid Activation Transformer
class HardSigmoidTransformer(BaseEstimator, TransformerMixin):
    """
    Hard Sigmoid Activation Transformer.

    This transformer applies the Hard Sigmoid activation function element-wise 
    to the input data. The Hard Sigmoid function is a computationally efficient 
    approximation of the Sigmoid function, defined as:
    - HardSigmoid(x) = max(0, min(1, 0.2 * x + 0.5))
    """
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return np.clip(0.2 * X + 0.5, 0, 1)  # Hard Sigmoid = 0.2 * x + 0.5

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
def get_activation_transformer(activation_name):
    """
    Get the corresponding activation function transformer based on the provided name.
    
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
        'relu': ReLUTransformer(),
        'sigmoid': SigmoidTransformer(),
        'tanh': TanhTransformer(),
        'elu': ELUTransformer(),
        'leakyrelu': LeakyReLUTransformer(),
        'softmax': SoftmaxTransformer(), 
        'swish': SwishTransformer(),
        'hardsigmoid': HardSigmoidTransformer(),
        'hardswish': HardSwishTransformer(),
        'softplus': SoftplusTransformer(), 
        'gelu': GELUTransformer(),
        'selu': SELUTransformer(),
        'mish': MishTransformer(),
        'elish': ELISHTransformer(),
        'logsigmoid': LogSigmoidTransformer(),
        'tanhshrink': TanhshrinkTransformer(),
        'swish1': Swish1Transformer()
    }

    # Validate the activation name
    if activation_name not in transformers:
        raise ValueError(f"Unsupported activation: {activation_name}")
    
    return transformers[activation_name]
