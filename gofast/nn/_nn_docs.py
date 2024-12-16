# -*- coding: utf-8 -*-

_shared_nn_params = dict(
    input_dim = """
input_dim: int
    The dimensionality of each input variable. This defines the number of
    features (or the length of the feature vector) for each individual input.
    For scalar features, this value is typically ``1``. However, for more
    complex data types such as embeddings, images, or time series, the input
    dimension can be greater than 1, reflecting the number of dimensions in
    the input vectors or feature matrices. This parameter is important for
    ensuring the correct shape and consistency of input data when training
    the model.

    Example:
    - For a single scalar feature per sample, ``input_dim = 1``.
    - For a word embedding with a 300-dimensional vector for each word, 
      ``input_dim = 300``.
    - For time-series data with 10 features at each time step, 
      ``input_dim = 10``.
    """, 
    
    units = """
units: int
    The number of units in the attention layer. This parameter defines
    the dimensionality of the output space for the attention mechanism.
    It determines the size of the internal representation for each input
    and plays a significant role in model capacity and performance.
    Larger values provide more capacity to capture complex patterns,
    but may also lead to higher computational costs. The number of units
    influences how well the model can learn complex representations from
    the input data. A larger number of units can improve performance on 
    more challenging tasks, but it can also increase memory and 
    computational requirements, so tuning this parameter is important.
    """,

    num_heads = """
num_heads: int
    The number of attention heads in the multi-head attention mechanism.
    Multiple attention heads allow the model to focus on different aspects
    of the input data, capturing more complex relationships within the
    data. More heads provide better representation power but increase
    computational costs. This parameter is crucial in self-attention
    mechanisms where each head can attend to different parts of the input
    data in parallel, improving the model's ability to capture diverse
    features. For example, in natural language processing, multiple heads
    allow the model to attend to different semantic aspects of the text.
    Using more heads can increase the model's capacity to learn complex
    features, but it also requires more memory and computational power.
    """,

    dropout_rate = """
dropout_rate: float, optional
    The dropout rate applied during training to prevent overfitting.
    Dropout is a regularization technique where a fraction of input units
    is randomly set to zero at each training step to prevent the model from
    relying too heavily on any one feature. This helps improve generalization
    and can make the model more robust. Dropout is particularly effective
    in deep learning models where overfitting is a common issue. The value
    should be between 0.0 and 1.0, where a value of ``0.0`` means no dropout
    is applied and a value of ``1.0`` means that all units are dropped. 
    A typical value for ``dropout_rate`` ranges from 0.1 to 0.5.
    """,

    activation = """
activation: str, optional
    The activation function to use in the Gated Recurrent Networks (GRNs).
    The activation function defines how the model's internal representations
    are transformed before being passed to the next layer. Supported values
    include:
    
    - ``'elu'``: Exponential Linear Unit (ELU), a variant of ReLU that
      improves training performance by preventing dying neurons. ELU provides
      a smooth output for negative values, which can help mitigate the issue 
      of vanishing gradients. The mathematical formulation for ELU is:
      
      .. math:: 
          f(x) = 
          \begin{cases}
          x & \text{if } x > 0 \\
          \alpha (\exp(x) - 1) & \text{if } x \leq 0
          \end{cases}
      
      where \(\alpha\) is a constant (usually 1.0).

    - ``'relu'``: Rectified Linear Unit (ReLU), a widely used activation
      function that outputs zero for negative input and the input itself for
      positive values. It is computationally efficient and reduces the risk
      of vanishing gradients. The mathematical formulation for ReLU is:
      
      .. math:: 
          f(x) = \max(0, x)
      
      where \(x\) is the input value.

    - ``'tanh'``: Hyperbolic Tangent, which squashes the outputs into a range 
      between -1 and 1. It is useful when preserving the sign of the input
      is important, but can suffer from vanishing gradients for large inputs.
      The mathematical formulation for tanh is:
      
      .. math::
          f(x) = \frac{2}{1 + \exp(-2x)} - 1

    - ``'sigmoid'``: Sigmoid function, commonly used for binary classification
      tasks, maps outputs between 0 and 1, making it suitable for probabilistic
      outputs. The mathematical formulation for sigmoid is:
      
      .. math:: 
          f(x) = \frac{1}{1 + \exp(-x)}

    - ``'linear'``: No activation (identity function), often used in regression
      tasks where no non-linearity is needed. The output is simply the input value:
      
      .. math:: 
          f(x) = x

    The default activation function is ``'elu'``.
    """,

    use_batch_norm = """
use_batch_norm: bool, optional
    Whether to use batch normalization in the Gated Recurrent Networks (GRNs).
    Batch normalization normalizes the input to each layer, stabilizing and
    accelerating the training process. When set to ``True``, it normalizes the
    activations by scaling and shifting them to maintain a stable distribution
    during training. This technique can help mitigate issues like vanishing and
    exploding gradients, making it easier to train deep networks. Batch normalization
    also acts as a form of regularization, reducing the need for other techniques
    like dropout. By default, batch normalization is turned off (``False``).
    
    """, 
    
    hidden_units = """
hidden_units: int
    The number of hidden units in the model's layers. This parameter 
    defines the size of the hidden layers throughout the model, including 
    Gated Recurrent Networks (GRNs), Long Short-Term Memory (LSTM) layers, 
    and fully connected layers. Increasing the value of ``hidden_units`` 
    enhances the model's capacity to capture more complex relationships and 
    patterns from the data. However, it also increases computational costs 
    due to a higher number of parameters. The choice of hidden units should 
    balance model capacity and computational feasibility, depending on the 
    complexity of the problem and available resources.
    """,

quantiles = """
quantiles: list of float or None, optional
    A list of quantiles to predict for each time step. For example, 
    specifying ``[0.1, 0.5, 0.9]`` would result in the model predicting 
    the 10th, 50th, and 90th percentiles of the target variable at each 
    time step. This is useful for estimating prediction intervals and 
    capturing uncertainty in forecasting tasks. If set to ``None``, the model 
    performs point forecasting and predicts a single value (e.g., the mean 
    or most likely value) for each time step. Quantile forecasting is commonly 
    used for applications where it is important to predict not just the 
    most likely outcome, but also the range of possible outcomes.
    """

)


_shared_docs: dict = {} 

_shared_docs[
    'tft_params_doc'
 ]="""

Parameters
----------
static_input_dim : int
    The dimensionality of each static input feature. This is the number 
    of features that do not change over time (e.g., static data such as 
    geographical coordinates, user demographics, etc.). For example, 
    if there are 2 static features (e.g., country, region), set this to 2.

dynamic_input_dim : int
    The dimensionality of each dynamic input feature. These are the 
    time-varying features (e.g., stock prices, temperature, etc.) that 
    change over time. This should be the number of features in your 
    temporal input data at each time step.

num_static_vars : int
    The number of static variables included in the input data. Static 
    variables are the features that do not change over time. For example, 
    this could be the number of static features such as geographical 
    attributes, demographic data, etc.

num_dynamic_vars : int
    The number of dynamic variables in the input data. These variables 
    change over time and are used to capture temporal patterns. For 
    example, the number of dynamic variables could be the number of 
    features such as stock prices, temperature, etc., that change across 
    different time steps.

hidden_units : int
    The number of hidden units in the layers of the model. This determines 
    the size of the hidden layers in the model architecture. A larger 
    number of hidden units allows the model to capture more complex 
    relationships in the data but may also increase computational cost.

num_heads: int
    The number of attention heads used in the multi-head attention 
    mechanism. This controls how many separate "attention" operations 
    are run in parallel. More heads typically allow the model to capture 
    more complex interactions within the input data.

dropout_rate : float
    The dropout rate used during training to prevent overfitting. This 
    value controls the fraction of input units to drop during training 
    (i.e., setting it to 0.2 means 20% of input units are randomly 
    set to zero in each forward pass). The value should be between 0 and 1.

forecast_horizon: int
    The number of time steps ahead to predict. This defines how far into 
    the future the model will generate predictions. For example, if set 
    to 7, the model will predict 7 future time steps from the current 
    data point. The value must be a positive integer (e.g., 1, 7, etc.).

quantiles: list, optional
    A list of quantiles for prediction. These quantiles define the 
    uncertainty in the predictions. For example, if set to `[0.1, 0.5, 0.9]`, 
    the model will output predictions for the 10th, 50th, and 90th percentiles 
    of the forecasted distribution. If set to `None`, the model will output 
    only the mean prediction.

activation : {'elu', 'relu', 'tanh', 'sigmoid', 'linear', 'gelu'}
    The activation function used in the model. Common choices include:
    - `'relu'`: Rectified Linear Unit (recommended for deep models)
    - `'elu'`: Exponential Linear Unit
    - `'tanh'`: Hyperbolic Tangent
    - `'sigmoid'`: Sigmoid function (common for binary classification)
    - `'linear'`: Linear activation (used in regression problems)
    - `'gelu'`: Gaussian Error Linear Unit (often used in transformers)

use_batch_norm : bool, default True
    Whether to use batch normalization in the model. Batch normalization 
    helps improve training by normalizing the output of previous layers 
    and speeding up convergence. Set this to `True` to enable batch 
    normalization, or `False` to disable it.

num_lstm_layers : int
    The number of LSTM layers in the model. LSTMs are used to capture 
    long-term dependencies in the data. More LSTM layers allow the model 
    to capture more complex temporal patterns but may increase the 
    computational cost.

lstm_units : list of int, optional
    The number of units in each LSTM layer. This can be a list of integers 
    where each element corresponds to the number of units in a specific 
    LSTM layer. For example, `[64, 32]` means the model has two LSTM 
    layers with 64 and 32 units, respectively. If set to `None`, the 
    number of units will be inferred from the `hidden_units` parameter.
"""

_shared_docs[
    'tft_notes_doc'
 ]="""
Notes
-----
- The model's performance can be highly dependent on the choice of 
  hyperparameters such as `hidden_units`, `num_heads`, and `dropout_rate`. 
  Experimentation is encouraged to find the optimal configuration for your 
  specific problem.
- If `n_features` is set to a value greater than the actual number of 
  features in the data, the model will fail to train properly.
- A larger `forecast_horizon` results in more complex predictions and 
  higher computational cost. Make sure to set it according to the 
  forecasting needs.

See Also
--------
- :class:`gofast.nn.transformers.TemporalFusionTransformer`: 
    The main class that implements the Temporal Fusion Transformers supporting
    the keras API.

References
----------
- Borovykh, A., et al. (2017). "Conditional Variational Autoencoder for 
  Time Series". 
- Lim, B., & Zohdy, M. (2020). "Temporal Fusion Transformers for Time 
  Series". 
"""

_shared_docs [ 
    'xtft_key_improvements'
]=r"""

**Key Enhancements:**

- **Enhanced Variable Embeddings**: 
  Employs learned normalization and multi-modal embeddings to
  flexibly integrate static, dynamic, and future covariates. 
  This allows the model to effectively handle heterogeneous 
  inputs and exploit relevant signals from different data 
  modalities.
  
  The model applies learned normalization and multi-modal embeddings
  to unify static, dynamic, and future covariates into a common
  representation space. Let :math:`\mathbf{x}_{static}`, 
  :math:`\mathbf{X}_{dynamic}`, and :math:`\mathbf{X}_{future}` 
  denote the static, dynamic, and future input tensors:
  .. math::
     \mathbf{x}_{norm} = \frac{\mathbf{x}_{static} - \mu}
     {\sigma + \epsilon}
     
  After normalization, static and dynamic features are embedded:
  .. math::
     \mathbf{E}_{dyn} = \text{MultiModalEmbedding}
     ([\mathbf{X}_{dynamic}, \mathbf{X}_{future}])
     
  and similarly, static embeddings 
  :math:`\mathbf{E}_{static}` are obtained. This enables flexible 
  integration of heterogeneous signals.

- **Multi-Scale LSTM Mechanisms**: 
  Adopts multiple LSTMs operating at various temporal resolutions
  as controlled by `scales`. By modeling patterns at multiple
  time scales (e.g., daily, weekly, monthly), the model can 
  capture long-term trends, seasonalities, and short-term 
  fluctuations simultaneously.
  
  Multiple LSTMs process the input at different scales defined by 
  `scales`. For a set of scales 
  :math:`S = \{s_1, s_2, \ldots, s_k\}`, each scale selects 
  time steps at intervals of :math:`s_i`:
  .. math::
     \mathbf{H}_{lstm} = \text{Concat}(
     [\text{LSTM}_{s_i}(\mathbf{E}_{dyn}^{(s_i)})]_{i=1}^{k})
     
  where :math:`\mathbf{E}_{dyn}^{(s_i)}` represents 
  :math:`\mathbf{E}_{dyn}` sampled at stride :math:`s_i`. This 
  approach captures patterns at multiple temporal resolutions 
  (e.g., daily, weekly).


- **Enhanced Attention Mechanisms**: 
  Integrates hierarchical, cross, and memory-augmented attention. 
  Hierarchical attention highlights critical temporal regions,
  cross attention fuses information from diverse feature spaces,
  and memory-augmented attention references a learned memory to
  incorporate long-range dependencies beyond the immediate 
  input window.
  
  XTFT integrates hierarchical, cross, and memory-augmented attention
  layers to enrich temporal and contextual relationships.  
  Hierarchical attention:
  .. math::
     \mathbf{H}_{hier} = \text{HierarchicalAttention}
     ([\mathbf{X}_{dynamic}, \mathbf{X}_{future}])
  
  Cross attention:
  .. math::
     \mathbf{H}_{cross} = \text{CrossAttention}
     ([\mathbf{X}_{dynamic}, \mathbf{E}_{dyn}])
  
  Memory-augmented attention with memory :math:`\mathbf{M}`:
  .. math::
     \mathbf{H}_{mem} = \text{MemoryAugmentedAttention}(
     \mathbf{H}_{hier}, \mathbf{M})
     
  Together, these attentions enable the model to focus on 
  short-term critical points, fuse different feature spaces,
  and reference long-range contexts.
  

- **Dynamic Quantile Loss**: 
  Implements adaptive quantile loss to produce probabilistic
  forecasts. This enables the model to return predictive intervals
  and quantify uncertainty, offering more robust and informed 
  decision-making capabilities.
  
  For quantiles :math:`q \in \{q_1,\ldots,q_Q\}`, and errors 
  :math:`e = y_{true} - y_{pred}`, quantile loss is defined as:
  .. math::
     \mathcal{L}_{quantile}(q) = \frac{1}{N}\sum_{n=1}^{N} 
     \max(q \cdot e_n, (q-1) \cdot e_n)
     
  This yields predictive intervals rather than single-point
  estimates, facilitating uncertainty-aware decision-making.
  
- **Multi-Horizon Output Strategies**:
  Facilitates forecasting over multiple future steps at once, 
  enabling practitioners to assess future scenarios and plan 
  accordingly. This functionality supports both deterministic 
  and probabilistic forecasts.
  
  XTFT predicts multiple horizons simultaneously. If 
  `forecast_horizons = H`, the decoder produces:
  .. math::
     \mathbf{Y}_{decoder} = \text{MultiDecoder}(\mathbf{H}_{combined})
     
  resulting in a forecast:
  .. math::
     \hat{\mathbf{Y}} \in \mathbb{R}^{B \times H \times D_{out}}
  
  This allows practitioners to assess future scenarios over 
  multiple steps rather than a single forecast instant.

- **Optimization for Complex Time Series**:
  Utilizes multi-resolution attention fusion, dynamic time 
  windowing, and residual connections to handle complex and 
  noisy data distributions. Such mechanisms improve training 
  stability and convergence rates, even in challenging 
  environments.
  
  Multi-resolution attention fusion and dynamic time windowing 
  improve the model's capability to handle complex, noisy data:
  .. math::
     \mathbf{H}_{fused} = \text{MultiResolutionAttentionFusion}(
     \mathbf{H}_{combined})
  
  Along with residual connections:
  .. math::
     \mathbf{H}_{res} = \mathbf{H}_{fused} + \mathbf{H}_{combined}
  
  These mechanisms stabilize training, enhance convergence, and 
  improve performance on challenging datasets.

- **Advanced Output Mechanisms**:
  Employs quantile distribution modeling to generate richer
  uncertainty estimations, thereby enabling the model to
  provide more detailed and informative predictions than 
  single-point estimates.
  
  Quantile distribution modeling converts decoder outputs into a
  set of quantiles:
  .. math::
     \mathbf{Y}_{quantiles} = \text{QuantileDistributionModeling}(
     \mathbf{Y}_{decoder})
  
  enabling richer uncertainty estimation and more informative 
  predictions, such as lower and upper bounds for future values.

When `quantiles` are specified, XTFT delivers probabilistic 
forecasts that include lower and upper bounds, enabling better 
risk management and planning. Moreover, anomaly detection 
capabilities, governed by `anomaly_loss_weight`, allow the 
model to identify and adapt to irregularities or abrupt changes
in the data.

"""

_shared_docs [
    'xtft_key_functions'
]=r"""

Key Functions
--------------
Consider a batch of time series data. Let:

- :math:`\mathbf{x}_{static} \in \mathbb{R}^{B \times D_{static}}`
  represent the static (time-invariant) features, where
  :math:`B` is the batch size and :math:`D_{static}` is the
  dimensionality of static inputs.
  
- :math:`\mathbf{X}_{dynamic} \in \mathbb{R}^{B \times T \times D_{dynamic}}`
  represent the dynamic (time-varying) features over :math:`T` time steps.
  Here, :math:`D_{dynamic}` corresponds to the dimensionality of
  dynamic inputs (e.g., historical observations).

- :math:`\mathbf{X}_{future} \in \mathbb{R}^{B \times T \times D_{future}}`
  represent the future known covariates, also shaped by
  :math:`T` steps and :math:`D_{future}` features. These may
  include planned events or predicted conditions known ahead of time.

The model first embeds dynamic and future features via multi-modal
embeddings, producing a unified representation:
.. math::
   \mathbf{E}_{dyn} = \text{MultiModalEmbedding}\left(
   [\mathbf{X}_{dynamic}, \mathbf{X}_{future}]\right)

To capture temporal dependencies at various resolutions, multi-scale
LSTMs are applied. These can process data at different temporal scales:
.. math::
   \mathbf{H}_{lstm} = \text{MultiScaleLSTM}(\mathbf{E}_{dyn})

Multiple attention mechanisms enhance the model’s representational
capacity:

1. Hierarchical attention focuses on both short-term and long-term
   interactions between dynamic and future features:
   .. math::
      \mathbf{H}_{hier} = \text{HierarchicalAttention}\left(
      [\mathbf{X}_{dynamic}, \mathbf{X}_{future}]\right)

2. Cross attention integrates information from different modalities
   or embedding spaces, here linking original dynamic inputs and
   their embeddings:
   .. math::
      \mathbf{H}_{cross} = \text{CrossAttention}\left(
      [\mathbf{X}_{dynamic}, \mathbf{E}_{dyn}]\right)

3. Memory-augmented attention incorporates an external memory for
   referencing distant past patterns not directly present in the
   current window:
   .. math::
      \mathbf{H}_{mem} = \text{MemoryAugmentedAttention}(\mathbf{H}_{hier})

Next, static embeddings :math:`\mathbf{E}_{static}` (obtained from
processing static inputs) are combined with the outputs from LSTMs
and attention mechanisms:
.. math::
   \mathbf{H}_{combined} = \text{Concatenate}\left(
   [\mathbf{E}_{static}, \mathbf{H}_{lstm}, \mathbf{H}_{mem},
   \mathbf{H}_{cross}]\right)

The combined representation is decoded into multi-horizon forecasts:
.. math::
   \mathbf{Y}_{decoder} = \text{MultiDecoder}(\mathbf{H}_{combined})

For probabilistic forecasting, quantile distribution modeling

transforms the decoder outputs into quantile predictions:
.. math::
   \mathbf{Y}_{quantiles} = \text{QuantileDistributionModeling}\left(
   \mathbf{Y}_{decoder}\right)

The final predictions are thus:
.. math::
   \hat{\mathbf{Y}} = \mathbf{Y}_{quantiles}

The loss function incorporates both quantile loss for probabilistic
forecasting and anomaly loss for robust handling of irregularities:
.. math::
   \mathcal{L} = \mathcal{L}_{quantile} + \lambda \mathcal{L}_{anomaly}

By adjusting :math:`\lambda`, the model can balance predictive
accuracy against robustness to anomalies.

Furthermore: 
    
- Multi-modal embeddings and multi-scale LSTMs enable the model to
  represent complex temporal patterns at various resolutions.
- Attention mechanisms (hierarchical, cross, memory-augmented)
  enrich the context and allow the model to focus on relevant
  aspects of the data.
- Quantile modeling provides probabilistic forecasts, supplying
  uncertainty intervals rather than single-point predictions.
- Techniques like residual connections, normalization, and
  anomaly loss weighting improve training stability and
  model robustness.
"""

_shared_docs[ 
    'xtft_methods'
]="""
   
Methods
-------
call(inputs, training=False)
    Perform the forward pass through the model. Given a tuple
    ``(static_input, dynamic_input, future_covariate_input)``,
    it processes all features through embeddings, LSTMs, and
    attention mechanisms before producing final forecasts.
    
    - ``static_input``: 
      A tensor of shape :math:`(B, D_{static})` representing 
      the static features. These do not vary with time.
    - ``dynamic_input``: 
      A tensor of shape :math:`(B, T, D_{dynamic})` representing
      dynamic features across :math:`T` time steps. These include
      historical values and time-dependent covariates.
    - ``future_covariate_input``: 
      A tensor of shape :math:`(B, T, D_{future})` representing
      future-known features, aiding multi-horizon forecasting.

    Depending on the presence of quantiles:
    - If ``quantiles`` is not `None`: 
      The output shape is :math:`(B, H, Q, D_{out})`, where 
      :math:`H` is `forecast_horizons`, :math:`Q` is the number of
      quantiles, and :math:`D_{out}` is `output_dim`.
    - If ``quantiles`` is `None`: 
      The output shape is :math:`(B, H, D_{out})`, providing a 
      deterministic forecast for each horizon.

    Parameters
    ----------
    inputs : tuple of tf.Tensor
        Input tensors `(static_input, dynamic_input, 
        future_covariate_input)`.
    training : bool, optional
        Whether the model is in training mode (default False).
        In training mode, layers like dropout and batch norm
        behave differently.

    Returns
    -------
    tf.Tensor
        The prediction tensor. Its shape and dimensionality depend
        on the `quantiles` setting. In probabilistic scenarios,
        multiple quantiles are returned. In deterministic mode, 
        a single prediction per horizon is provided.

compute_objective_loss(y_true, y_pred, anomaly_scores)
    Compute the total loss, combining both quantile loss (if 
    `quantiles` is not `None`) and anomaly loss. Quantile loss
    measures forecasting accuracy at specified quantiles, while
    anomaly loss penalizes unusual deviations or anomalies.

    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth targets. Shape: :math:`(B, H, D_{out})`.
    y_pred : tf.Tensor
        Model predictions. If quantiles are present:
        :math:`(B, H, Q, D_{out})`. Otherwise:
        :math:`(B, H, D_{out})`.
    anomaly_scores : tf.Tensor
        Tensor indicating anomaly severity. Its shape typically
        matches `(B, H, D_{dynamic})` or a related dimension.

    Returns
    -------
    tf.Tensor
        A scalar tensor representing the combined loss. Lower 
        values indicate better performance, balancing accuracy
        and anomaly handling.

anomaly_loss(anomaly_scores)
    Compute the anomaly loss component. This term encourages the
    model to be robust against abnormal patterns in the data.
    Higher anomaly scores lead to higher loss, prompting the model
    to adjust predictions or representations to reduce anomalies.

    Parameters
    ----------
    anomaly_scores : tf.Tensor
        A tensor reflecting the presence and intensity of anomalies.
        Its shape often corresponds to time steps and dynamic 
        features, e.g., `(B, H, D_{dynamic})`.

    Returns
    -------
    tf.Tensor
        A scalar tensor representing the anomaly loss. Minimizing
        this term encourages the model to learn patterns that 
        mitigate anomalies and produce more stable forecasts.
"""