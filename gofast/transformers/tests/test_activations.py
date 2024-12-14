# -*- coding: utf-8 -*-
import pytest
import numpy as np
try: 
    import tensorflow as tf
except: 
    pass 
try: 
   import torch
except :  pass 

from gofast.tools.validator import filter_valid_kwargs 

from gofast.transformers.activations import (
    ReLUTransformer,
    SigmoidTransformer,
    TanhTransformer,
    ELUTransformer,
    LeakyReLUTransformer,
    SoftmaxTransformer,
    SwishTransformer,
    HardSigmoidTransformer,
    HardSwishTransformer,
    SoftplusTransformer,
    GELUTransformer,
    SELUTransformer,
    MishTransformer,
    ELISHTransformer,
    LogSigmoidTransformer,
    TanhshrinkTransformer,
    Swish1Transformer,
    get_activation_transformer
)

#%%
@pytest.mark.parametrize(
    "TransformerClass, activation_type",
    [
        (ReLUTransformer, "relu"),
        (SigmoidTransformer, "sigmoid"),
        (TanhTransformer, "tanh"),
        (ELUTransformer, "elu"),
        (LeakyReLUTransformer, "leaky_relu"),
        (SoftmaxTransformer, "softmax"),
        (SwishTransformer, "swish"),
        (HardSigmoidTransformer, "hard_sigmoid"),
        (HardSwishTransformer, "hard_swish"),
        (SoftplusTransformer, "softplus"),
        (GELUTransformer, "gelu"),
        (SELUTransformer, "selu"),
        (MishTransformer, "mish"),
        (ELISHTransformer, "elish"),
        (LogSigmoidTransformer, "log_sigmoid"),
        (TanhshrinkTransformer, "tanhshrink"),
        (Swish1Transformer, "swish1"),
    ],
)
@pytest.mark.parametrize(
    "backend_name, backend_module",
    [
        ("numpy", np),
        ("tensorflow", tf),
        # ("torch", torch),
    ],
)
@pytest.mark.parametrize(
    "batch_size",
    [None, 2],
)
def test_activation_transformers(
        TransformerClass, activation_type, backend_name,
        backend_module, batch_size):
    X = np.array([[-1.0, 0.0, 1.0], [2.0, -2.0, 3.0]])
    scale = 1.0
    shift = 0.0
    precision = 1e-6
    alpha = 0.1  # For LeakyReLUTransformer
    verbose=0
    # Initialize transformer with specified backend and parameters
    params = { 
        'scale': scale, 
        'shift': shift, 
        'precision': precision, 
        'alpha': alpha, 
        'verbose': verbose 
        }
    
    valid_kwargs = filter_valid_kwargs(TransformerClass, params)
    
    if 'alpha' in valid_kwargs:
        transformer = TransformerClass(
            scale=scale,
            shift=shift,
            precision=precision,
            batch_size=batch_size,
            backend=backend_name,
            alpha=alpha,
            verbose=0
        )
    else:
        transformer = TransformerClass(
            scale=scale,
            shift=shift,
            precision=precision,
            batch_size=batch_size,
            backend=backend_name,
            verbose=0
        )

    # Fit the transformer
    transformer.fit(X)

    # Transform the data
    X_transformed = transformer.transform(X)

    # Apply scaling and shifting
    X_scaled_shifted = scale * X + shift

    # Define expected activation function
    if activation_type == "relu":
        if backend_name == "numpy":
            expected = np.maximum(0, X_scaled_shifted)
        elif backend_name == "tensorflow":
            expected = backend_module.math.maximum(0, X_scaled_shifted).numpy()
        elif backend_name == "torch":
            expected = backend_module.relu(torch.tensor(X_scaled_shifted)).numpy()

    elif activation_type == "sigmoid":
        if backend_name == "numpy":
            expected = 1 / (1 + np.exp(-X_scaled_shifted))
        elif backend_name == "tensorflow":
            expected = backend_module.math.sigmoid(X_scaled_shifted).numpy()
        elif backend_name == "torch":
            expected = backend_module.sigmoid(torch.tensor(X_scaled_shifted)).numpy()

    elif activation_type == "tanh":
        if backend_name == "numpy":
            expected = np.tanh(X_scaled_shifted)
        elif backend_name == "tensorflow":
            expected = backend_module.math.tanh(X_scaled_shifted).numpy()
        elif backend_name == "torch":
            expected = backend_module.tanh(torch.tensor(X_scaled_shifted)).numpy()

    elif activation_type == "elu":
        if backend_name == "numpy":
            expected = np.where(X_scaled_shifted > 0, X_scaled_shifted, backend_module.exp(X_scaled_shifted) - 1)
        elif backend_name == "tensorflow":
            expected = backend_module.nn.elu(X_scaled_shifted).numpy()
        elif backend_name == "torch":
            expected = backend_module.functional.elu(torch.tensor(X_scaled_shifted)).numpy()

    elif activation_type == "leaky_relu":
        if backend_name == "numpy":
            expected = np.where(X_scaled_shifted > 0, X_scaled_shifted, alpha * X_scaled_shifted)
        elif backend_name == "tensorflow":
            expected = backend_module.where(
                X_scaled_shifted > 0, X_scaled_shifted, alpha * X_scaled_shifted
            ).numpy()
        elif backend_name == "torch":
            expected = backend_module.where(
                torch.tensor(X_scaled_shifted) > 0, torch.tensor(X_scaled_shifted), alpha * torch.tensor(X_scaled_shifted)
            ).numpy()

    elif activation_type == "softmax":
        if backend_name == "numpy":
            e_X = np.exp(X_scaled_shifted - np.max(X_scaled_shifted, axis=1, keepdims=True))
            expected = e_X / np.sum(e_X, axis=1, keepdims=True)
        elif backend_name == "tensorflow":
            expected = backend_module.nn.softmax(X_scaled_shifted, axis=1).numpy()
        elif backend_name == "torch":
            expected = backend_module.softmax(torch.tensor(X_scaled_shifted), dim=1).numpy()

    elif activation_type == "swish":
        if backend_name == "numpy":
            sigmoid = 1 / (1 + np.exp(-X_scaled_shifted))
            expected = X_scaled_shifted * sigmoid
        elif backend_name == "tensorflow":
            sigmoid = backend_module.math.sigmoid(X_scaled_shifted)
            expected = (X_scaled_shifted * sigmoid).numpy()
        elif backend_name == "torch":
            sigmoid = backend_module.sigmoid(torch.tensor(X_scaled_shifted))
            expected = (torch.tensor(X_scaled_shifted) * sigmoid).numpy()

    elif activation_type == "hard_sigmoid":
        if backend_name == "numpy":
            expected = np.clip((X_scaled_shifted + 3) / 6, 0, 1)
        elif backend_name == "tensorflow":
            expected = backend_module.math.minimum(
                backend_module.math.maximum((X_scaled_shifted + 3) / 6, 0), 1
            ).numpy()
        elif backend_name == "torch":
            expected = backend_module.clamp((torch.tensor(X_scaled_shifted) + 3) / 6, 0, 1).numpy()

    elif activation_type == "hard_swish":
        if backend_name == "numpy":
            expected = X_scaled_shifted * np.clip((X_scaled_shifted + 3) / 6, 0, 1)
        elif backend_name == "tensorflow":
            clip = backend_module.math.minimum(
                backend_module.math.maximum((X_scaled_shifted + 3) / 6, 0), 1
            )
            expected = (X_scaled_shifted * clip).numpy()
        elif backend_name == "torch":
            clip = backend_module.clamp((torch.tensor(X_scaled_shifted) + 3) / 6, 0, 1)
            expected = (torch.tensor(X_scaled_shifted) * clip).numpy()

    elif activation_type == "softplus":
        if backend_name == "numpy":
            expected = np.log1p(np.exp(X_scaled_shifted))
        elif backend_name == "tensorflow":
            expected = backend_module.math.softplus(X_scaled_shifted).numpy()
        elif backend_name == "torch":
            expected = backend_module.softplus(torch.tensor(X_scaled_shifted)).numpy()

    elif activation_type == "gelu":
        if backend_name == "numpy":
            expected = 0.5 * X_scaled_shifted * (1 + np.tanh(np.sqrt(2 / np.pi) * (X_scaled_shifted + 0.044715 * np.power(X_scaled_shifted, 3))))
        elif backend_name == "tensorflow":
            expected = backend_module.math.gelu(X_scaled_shifted).numpy()
        elif backend_name == "torch":
            expected = backend_module.gelu(torch.tensor(X_scaled_shifted)).numpy()

    elif activation_type == "selu":
        if backend_name == "numpy":
            alpha = 1.6732632423543772848170429916717
            scale = 1.0507009873554804934193349852946
            expected = scale * np.where(X_scaled_shifted > 0, X_scaled_shifted, alpha * (np.exp(X_scaled_shifted) - 1))
        elif backend_name == "tensorflow":
            expected = backend_module.nn.selu(X_scaled_shifted).numpy()
        elif backend_name == "torch":
            expected = backend_module.selu(torch.tensor(X_scaled_shifted)).numpy()

    elif activation_type == "mish":
        if backend_name == "numpy":
            expected = X_scaled_shifted * np.tanh(np.log1p(np.exp(X_scaled_shifted)))
        elif backend_name == "tensorflow":
            expected = (X_scaled_shifted * backend_module.math.tanh(backend_module.math.softplus(X_scaled_shifted))).numpy()
        elif backend_name == "torch":
            softplus = backend_module.softplus(torch.tensor(X_scaled_shifted))
            expected = (torch.tensor(X_scaled_shifted) * backend_module.tanh(softplus)).numpy()

    elif activation_type == "elish":
        if backend_name == "numpy":
            expected = X_scaled_shifted * np.tanh(X_scaled_shifted)
        elif backend_name == "tensorflow":
            expected = (X_scaled_shifted * backend_module.math.tanh(X_scaled_shifted)).numpy()
        elif backend_name == "torch":
            expected = (torch.tensor(X_scaled_shifted) * backend_module.tanh(torch.tensor(X_scaled_shifted))).numpy()

    elif activation_type == "log_sigmoid":
        if backend_name == "numpy":
            expected = np.log(1 / (1 + np.exp(-X_scaled_shifted)))
        elif backend_name == "tensorflow":
            expected = backend_module.math.log_sigmoid(X_scaled_shifted).numpy()
        elif backend_name == "torch":
            expected = backend_module.logsigmoid(torch.tensor(X_scaled_shifted)).numpy()

    elif activation_type == "tanhshrink":
        if backend_name == "numpy":
            expected = X_scaled_shifted - np.tanh(X_scaled_shifted)
        elif backend_name == "tensorflow":
            expected = (X_scaled_shifted - backend_module.math.tanh(X_scaled_shifted)).numpy()
        elif backend_name == "torch":
            expected = (torch.tensor(X_scaled_shifted) - backend_module.tanh(torch.tensor(X_scaled_shifted))).numpy()

    elif activation_type == "swish1":
        if backend_name == "numpy":
            sigmoid = 1 / (1 + np.exp(-X_scaled_shifted))
            expected = X_scaled_shifted * sigmoid
        elif backend_name == "tensorflow":
            sigmoid = backend_module.math.sigmoid(X_scaled_shifted)
            expected = (X_scaled_shifted * sigmoid).numpy()
        elif backend_name == "torch":
            sigmoid = backend_module.sigmoid(torch.tensor(X_scaled_shifted))
            expected = (torch.tensor(X_scaled_shifted) * sigmoid).numpy()

    else:
        raise ValueError(f"Unsupported activation type: {activation_type}")

    # Apply precision rounding if necessary
    if precision:
        expected = np.round(expected / precision) * precision

    # Assert that the transformed data matches the expected output
    assert np.allclose(
        X_transformed, expected, atol=precision
    ), f"{TransformerClass.__name__} failed for backend {backend_name} with batch_size={batch_size}"
    

def test_get_activation_transformer():
    X = np.array([[-1.0, 0.0, 1.0], [2.0, -2.0, 3.0]])
    scale = 1.0
    shift = 0.0
    precision = 1e-6
    batch_size = None
    verbose = 0

    activations = [
        'relu',
        'sigmoid',
        'tanh',
        'elu',
        'leaky_relu',
        'softmax',
        'swish',
        'hard_sigmoid',
        'hard_swish',
        'softplus',
        'gelu',
        'selu',
        'mish',
        'elish',
        'log_sigmoid',
        'tanhshrink',
        'swish1',
    ]
  
    for activation in activations:
        transformer = get_activation_transformer(
            activation=activation,
            scale=scale,
            shift=shift,
            precision=precision,
            batch_size=batch_size,
            backend=None,
            verbose=verbose
        )
        transformer.fit(X)
        X_transformed = transformer.transform(X)

        # Apply scaling and shifting
        X_scaled_shifted = scale * X + shift

        # Define expected activation
        if activation == "relu":
            expected = np.maximum(0, X_scaled_shifted)
        elif activation == "sigmoid":
            expected = 1 / (1 + np.exp(-X_scaled_shifted))
        elif activation == "tanh":
            expected = np.tanh(X_scaled_shifted)
        elif activation == "elu":
            expected = np.where(X_scaled_shifted > 0, X_scaled_shifted, np.exp(X_scaled_shifted) - 1)
        elif activation == "leaky_relu":
            alpha = transformer.alpha
            expected = np.where(X_scaled_shifted > 0, X_scaled_shifted, alpha * X_scaled_shifted)
        elif activation == "softmax":
            e_X = np.exp(X_scaled_shifted - np.max(X_scaled_shifted, axis=1, keepdims=True))
            expected = e_X / np.sum(e_X, axis=1, keepdims=True)
        elif activation == "swish":
            sigmoid = 1 / (1 + np.exp(-X_scaled_shifted))
            expected = X_scaled_shifted * sigmoid
        elif activation == "hard_sigmoid":
            expected = np.clip((X_scaled_shifted + 3) / 6, 0, 1)
        elif activation == "hard_swish":
            expected = X_scaled_shifted * np.clip((X_scaled_shifted + 3) / 6, 0, 1)
        elif activation == "softplus":
            expected = np.log1p(np.exp(X_scaled_shifted))
        elif activation == "gelu":
            expected = 0.5 * X_scaled_shifted * (1 + np.tanh(np.sqrt(2 / np.pi) * (X_scaled_shifted + 0.044715 * np.power(X_scaled_shifted, 3))))
        elif activation == "selu":
            alpha = 1.6732632423543772848170429916717
            scale_ = 1.0507009873554804934193349852946
            expected = scale_ * np.where(X_scaled_shifted > 0, X_scaled_shifted, alpha * (np.exp(X_scaled_shifted) - 1))
        elif activation == "mish":
            expected = X_scaled_shifted * np.tanh(np.log1p(np.exp(X_scaled_shifted)))
        elif activation == "elish":
            expected = X_scaled_shifted * np.tanh(X_scaled_shifted)
        elif activation == "log_sigmoid":
            expected = np.log(1 / (1 + np.exp(-X_scaled_shifted)))
        elif activation == "tanhshrink":
            expected = X_scaled_shifted - np.tanh(X_scaled_shifted)
        elif activation == "swish1":
            sigmoid = 1 / (1 + np.exp(-X_scaled_shifted))
            expected = X_scaled_shifted * sigmoid
        else:
            raise ValueError(f"Unsupported activation type: {activation}")

        # Apply precision rounding if necessary
        if precision:
            expected = np.round(expected / precision) * precision

        # Assert that the transformed data matches the expected output
        assert np.allclose(
            X_transformed, expected, atol=precision
        ), f"get_activation_transformer failed for activation {activation}"

if __name__=='__main__': 
    pytest.main([__file__])