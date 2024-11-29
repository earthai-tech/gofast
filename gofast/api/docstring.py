# -*- coding: utf-8 -*-
#   Licence: BSD 3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""Provides core components and utilities for generating standardized docstrings
across the `gofast` API, enhancing consistency and readability in documentation."""
from __future__ import annotations
import re 

__all__=[
    'DocstringComponents',
    '_baseplot_params', 
    '_seealso_blurbs',
    '_core_returns', 
    'gf_rst_epilog',
    '_core_params',
    'refglossary',
    '_core_docs',
    ]

class DocstringComponents:
    """
    A class for managing and cleaning docstring components for classes, methods,
    or functions. It provides structured access to raw docstrings by parsing 
    them from a dictionary, optionally stripping outer whitespace, and allowing
    dot access to the components.

    This class is typically used to standardize, clean, and manage the 
    docstrings for different components of a codebase (such as methods or classes),
    particularly when docstrings contain multiple components that need to be 
    extracted, cleaned, and accessed easily.

    Parameters
    ----------
    comp_dict : dict
        A dictionary where the keys are component names and the values are 
        the raw docstring contents. The dictionary may contain entries such as 
        "description", "parameters", "returns", etc.

    strip_whitespace : bool, optional, default=True
        If True, it will remove leading and trailing whitespace from each
        entry in the `comp_dict`. If False, the whitespace will be retained.

    Attributes
    ----------
    entries : dict
        A dictionary containing the cleaned or raw docstring components after 
        parsing, depending on the `strip_whitespace` flag. These components 
        are accessible via dot notation.

    Methods
    -------
    __getattr__(attr)
        Provides dot access to the components in `self.entries`. If the requested
        attribute exists in `self.entries`, it is returned. Otherwise, it attempts
        to look for the attribute normally or raise an error if not found.

    from_nested_components(cls, **kwargs)
        A class method that allows combining multiple sub-sets of docstring
        components into a single `DocstringComponents` instance.

    Examples
    --------
    # Example 1: Creating a DocstringComponents object with basic docstrings
    doc_dict = {
        "description": "This function adds two numbers.",
        "parameters": "a : int\n    First number.\nb : int\n    Second number.",
        "returns": "int\n    The sum of a and b."
    }

    doc_comp = DocstringComponents(doc_dict)
    print(doc_comp.description)
    # Output: This function adds two numbers.

    # Example 2: Using `from_nested_components` to add multiple sub-sets
    sub_dict_1 = {
        "description": "This function multiplies two numbers.",
        "parameters": "a : int\n    First number.\nb : int\n    Second number.",
        "returns": "int\n    The product of a and b."
    }
    sub_dict_2 = {
        "example": "example_func(2, 3) # Returns 6"
    }

    doc_comp = DocstringComponents.from_nested_components(sub_dict_1, sub_dict_2)
    print(doc_comp.example)
    # Output: example_func(2, 3) # Returns 6
    """

    regexp = re.compile(r"\n((\n|.)+)\n\s*", re.MULTILINE)

    def __init__(self, comp_dict, strip_whitespace=True):
        """Read entries from a dict, optionally stripping outer whitespace."""
        if strip_whitespace:
            entries = {}
            for key, val in comp_dict.items():
                m = re.match(self.regexp, val)
                if m is None:
                    entries[key] = val
                else:
                    entries[key] = m.group(1)
        else:
            entries = comp_dict.copy()

        self.entries = entries

    def __getattr__(self, attr):
        """Provide dot access to entries for clean raw docstrings."""
        if attr in self.entries:
            return self.entries[attr]
        else:
            try:
                return self.__getattribute__(attr)
            except AttributeError as err:
                # If Python is run with -OO, it will strip docstrings and our lookup
                # from self.entries will fail. We check for __debug__, which is actually
                # set to False by -O (it is True for normal execution).
                # But we only want to see an error when building the docs;
                # not something users should see, so this slight inconsistency is fine.
                if __debug__:
                    raise err
                else:
                    pass

    @classmethod
    def from_nested_components(cls, **kwargs):
        """Add multiple sub-sets of components."""
        return cls(kwargs, strip_whitespace=False)


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


_core_params= dict ( 
    data ="""
data: str, filepath_or_buffer, or :class:`pandas.core.DataFrame`
    Data source, which can be a path-like object, a DataFrame, or a file-like object.
    - For path-like objects, data is read, asserted, and validated. Accepts 
    any valid string path, including URLs. Supported URL schemes: http, ftp, 
    s3, gs, and file. For file URLs, a host is expected (e.g., 'file://localhost/path/to/table.csv'). 
    - os.PathLike objects are also accepted.
    - File-like objects should have a `read()` method (
        e.g., opened via the `open` function or `StringIO`).
    When a path-like object is provided, the data is loaded and validated. 
    This flexibility allows for various data sources, including local files or 
    files hosted on remote servers.

    """, 
    X = """
X: ndarray of shape (M, N), where M = m-samples and N = n-features
    Training data; represents observed data at both training and prediction 
    times, used as independent variables in learning. The uppercase notation 
    signifies that it typically represents a matrix. In a matrix form, each 
    sample is represented by a feature vector. Alternatively, X may not be a 
    matrix and could require a feature extractor or a pairwise metric for 
    transformation. It's critical to ensure data consistency and compatibility 
    with the chosen learning model.
    """,
    y = """
y: array-like of shape (m,), where M = m-samples
    Training target; signifies the dependent variable in learning, observed 
    during training but unavailable at prediction time. The target is often 
    the main focus of prediction in supervised learning models. Ensuring the 
    correct alignment and representation of target data is crucial for effective 
    model training.
    """,
    Xt = """
Xt: ndarray, shape (M, N), where M = m-samples and N = n-features
    Test set; denotes data observed during testing and prediction, used as 
    independent variables in learning. Like X, Xt is typically a matrix where 
    each sample corresponds to a feature vector. The consistency between the 
    training set (X) and the test set (Xt) in terms of feature representation 
    and preprocessing is essential for accurate model evaluation.
    """,
    yt = """
yt: array-like, shape (M,), where M = m-samples
    Test target; represents the dependent variable in learning, akin to 'y' 
    but for the testing phase. While yt is observed during training, it is used
    to evaluate the performance of predictive models. The test target helps 
    in assessing the generalization capabilities of the model to unseen data.
    """,
    target_name = """
target_name: str
    Target name or label used in supervised learning. It serves as the reference name 
    for the target variable (`y`) or label. Accurate identification of `target_name` is 
    crucial for model training and interpretation, especially in datasets with multiple 
    potential targets.
""",

   z = """
z: array-like 1D or pandas.Series
    Represents depth values in a 1D array or pandas series. Multi-dimensional arrays 
    are not accepted. If `z` is provided as a DataFrame and `zname` is unspecified, 
    an error is raised. In such cases, `zname` is necessary for extracting the depth 
    column from the DataFrame.
""",
    zname = """
zname: str or int
    Specifies the column name or index for depth values within a DataFrame. If an 
    integer is provided, it is interpreted as the column index for depth values. 
    The integer value should be within the DataFrame's column range. `zname` is 
    essential when the depth information is part of a larger DataFrame.
""",
    kname = """
kname: str or int
    Identifies the column name or index for permeability coefficient ('K') within a 
    DataFrame. An integer value represents the column index for 'K'. It must be within 
    the DataFrame's column range. `kname` is required when permeability data is 
    integrated into a DataFrame, ensuring correct retrieval and processing of 'K' values.
""",
   k = """
k: array-like 1D or pandas.Series
    Array or series containing permeability coefficient ('K') values. Multi-dimensional 
    arrays are not supported. If `K` is provided as a DataFrame without specifying 
    `kname`, an error is raised. `kname` is used to extract 'K' values from the DataFrame 
    and overwrite the original `K` input.
""",
    target = """
target: Array-like or pandas.Series
    The dependent variable in supervised (and semi-supervised) learning, usually 
    denoted as `y` in an estimator's fit method. Also known as the dependent variable, 
    outcome variable, response variable, ground truth, or label. Scikit-learn handles 
    targets with minimal structure: a class from a finite set, a finite real-valued 
    number, multiple classes, or multiple numbers. In this library, `target` is 
    conceptualized as a pandas Series with `target_name` as its name, combining the 
    identifier and the variable `y`.
    Refer to Scikit-learn's documentation on target types for more details:
    [Scikit-learn Target Types](https://scikit-learn.org/stable/glossary.html#glossary-target-types).
""",
    model="""
model: callable, always as a function,    
    A model estimator. An object which manages the estimation and decoding 
    of a model. The model is estimated as a deterministic function of:
        * parameters provided in object construction or with set_params;
        * the global numpy.random random state if the estimator’s random_state 
            parameter is set to None; and
        * any data or sample properties passed to the most recent call to fit, 
            fit_transform or fit_predict, or data similarly passed in a sequence 
            of calls to partial_fit.
    The estimated model is stored in public and private attributes on the 
    estimator instance, facilitating decoding through prediction and 
    transformation methods.
    Estimators must provide a fit method, and should provide `set_params` and 
    `get_params`, although these are usually provided by inheritance from 
    `base.BaseEstimator`.
    The core functionality of some estimators may also be available as a ``function``.
    """,
    clf="""
clf :callable, always as a function, classifier estimator
    A supervised (or semi-supervised) predictor with a finite set of discrete 
    possible output values. A classifier supports modeling some of binary, 
    multiclass, multilabel, or multiclass multioutput targets. Within scikit-learn, 
    all classifiers support multi-class classification, defaulting to using a 
    one-vs-rest strategy over the binary classification problem.
    Classifiers must store a classes_ attribute after fitting, and usually 
    inherit from base.ClassifierMixin, which sets their _estimator_type attribute.
    A classifier can be distinguished from other estimators with is_classifier.
    It must implement::
        * fit
        * predict
        * score
    It may also be appropriate to implement decision_function, predict_proba 
    and predict_log_proba.    
    """,
    reg="""
reg: callable, always as a function
    A regression estimator; Estimators must provide a fit method, and should 
    provide `set_params` and 
    `get_params`, although these are usually provided by inheritance from 
    `base.BaseEstimator`. The estimated model is stored in public and private 
    attributes on the estimator instance, facilitating decoding through prediction 
    and transformation methods.
    The core functionality of some estimators may also be available as a
    ``function``.
    """,
    cv="""
cv: float,    
    A cross validation splitting strategy. It used in cross-validation based 
    routines. cv is also available in estimators such as multioutput. 
    ClassifierChain or calibration.CalibratedClassifierCV which use the 
    predictions of one estimator as training data for another, to not overfit 
    the training supervision.
    Possible inputs for cv are usually::
        * An integer, specifying the number of folds in K-fold cross validation. 
            K-fold will be stratified over classes if the estimator is a classifier
            (determined by base.is_classifier) and the targets may represent a 
            binary or multiclass (but not multioutput) classification problem 
            (determined by utils.multiclass.type_of_target).
        * A cross-validation splitter instance. Refer to the User Guide for 
            splitters available within `Scikit-learn`_
        * An iterable yielding train/test splits.
    With some exceptions (especially where not using cross validation at all 
                          is an option), the default is ``4-fold``.
    .. _Scikit-learn: https://scikit-learn.org/stable/glossary.html#glossary
    """,
    scoring="""
scoring: str, callable
    Specifies the score function to be maximized (usually by :ref:`cross
    validation <cross_validation>`), or -- in some cases -- multiple score
    functions to be reported. The score function can be a string accepted
    by :func:`sklearn.metrics.get_scorer` or a callable :term:`scorer`, not to 
    be confused with an :term:`evaluation metric`, as the latter have a more
    diverse API.  ``scoring`` may also be set to None, in which case the
    estimator's :term:`score` method is used.  See `slearn.scoring_parameter`
    in the `Scikit-learn`_ User Guide.
    """, 
    random_state="""
random_state : int, RandomState instance or None, default=None
    Controls the shuffling applied to the data before applying the split.
    Pass an int for reproducible output across multiple function calls..    
    """,
    test_size="""
test_size : float or int, default=None
    If float, should be between 0.0 and 1.0 and represent the proportion
    of the dataset to include in the test split. If int, represents the
    absolute number of test samples. If None, the value is set to the
    complement of the train size. If ``train_size`` is also None, it will
    be set to 0.25.    
    """, 
    n_jobs="""
n_jobs: int, 
    is used to specify how many concurrent processes or threads should be 
    used for routines that are parallelized with joblib. It specifies the maximum 
    number of concurrently running workers. If 1 is given, no joblib parallelism 
    is used at all, which is useful for debugging. If set to -1, all CPUs are 
    used. For instance::
        * `n_jobs` below -1, (n_cpus + 1 + n_jobs) are used. 
        
        * `n_jobs`=-2, all CPUs but one are used. 
        * `n_jobs` is None by default, which means unset; it will generally be 
            interpreted as n_jobs=1 unless the current joblib.Parallel backend 
            context specifies otherwise.

    Note that even if n_jobs=1, low-level parallelism (via Numpy and OpenMP) 
    might be used in some configuration.  
    """,
    verbose="""
verbose: int, `default` is ``0``    
    Control the level of verbosity. Higher value lead to more messages. 
    """  
) 

_core_returns = dict ( 
    self="""
self: `BaseClass` instance 
    returns ``self`` for easy method chaining.
    """, 
    ax="""
:class:`matplotlib.axes.Axes`
    The matplotlib axes containing the plot.
    """,
    facetgrid="""
:class:`FacetGrid`
    An object managing one or more subplots that correspond to conditional data
    subsets with convenient methods for batch-setting of axes attributes.
    """,
    jointgrid="""
:class:`JointGrid`
    An object managing multiple subplots that correspond to joint and marginal axes
    for plotting a bivariate relationship or distribution.
    """,
    pairgrid="""
:class:`PairGrid`
    An object managing multiple subplots that correspond to joint and marginal axes
    for pairwise combinations of multiple variables in a dataset.
    """, 
 )

_seealso_blurbs = dict(
    # Relational plots
    scatterplot="""
scatterplot : Plot data using points.
    """,
    lineplot="""
lineplot : Plot data using lines.
    """,
    # Distribution plots
    displot="""
displot : Figure-level interface to distribution plot functions.
    """,
    histplot="""
histplot : Plot a histogram of binned counts with optional normalization or smoothing.
    """,
    kdeplot="""
kdeplot : Plot univariate or bivariate distributions using kernel density estimation.
    """,
    ecdfplot="""
ecdfplot : Plot empirical cumulative distribution functions.
    """,
    rugplot="""
rugplot : Plot a tick at each observation value along the x and/or y axes.
    """,

    # Categorical plots
    stripplot="""
stripplot : Plot a categorical scatter with jitter.
    """,
    swarmplot="""
swarmplot : Plot a categorical scatter with non-overlapping points.
    """,
    violinplot="""
violinplot : Draw an enhanced boxplot using kernel density estimation.
    """,
    pointplot="""
pointplot : Plot point estimates and CIs using markers and lines.
    """,
    boxplot="""
boxplot : Draw an enhanced boxplot.
     """,
    # Multiples
    jointplot="""
jointplot : Draw a bivariate plot with univariate marginal distributions.
    """,
    pairplot="""
jointplot : Draw multiple bivariate plots with univariate marginal distributions.
    """,
    jointgrid="""
JointGrid : Set up a figure with joint and marginal views on bivariate data.
    """,
    pairgrid="""
PairGrid : Set up a figure with joint and marginal views on multiple variables.
    """,
)
                 
_core_docs = dict(
    params=DocstringComponents(_core_params),
    returns=DocstringComponents(_core_returns),
    seealso=DocstringComponents(_seealso_blurbs),
)

_baseplot_params = dict( 
    savefig= """
savefig: str, Path-like object, 
    savefigure's name, *default* is ``None``
    """,
    fig_dpi="""
fig_dpi: float, 
    dots-per-inch resolution of the figure. *default* is 300   
    """, 
    fig_num="""
fig_num: int, 
    size of figure in inches (width, height). *default* is [5, 5]
    """, 
    fig_size= """
fig_size: Tuple (int, int) or inch 
   size of figure in inches (width, height).*default* is [5, 5]
    """, 
    fig_orientation="""
fig_orientation: str, 
    figure orientation. *default* is ``landscape``
    """, 
    fig_title="""
fig_tile: str, 
    figure title. *default* is ``None``     
    """, 
    fs="""
fs: float, 
     size of font of axis tick labels, axis labels are fs+2. *default* is 6
    """,
    ls="""
ls: str, 
    line style, it can be [ '-' | '.' | ':' ] . *default* is '-'
    """, 
    lc="""
lc: str, Optional, 
    line color of the plot, *default* is ``k``
    """, 
    lw="""
lw: float, Optional, 
    line weight of the plot, *default* is ``1.5``
    """, 
    alpha="""
alpha: float between 0 < alpha < 1, 
    transparency number, *default* is ``0.5``,   
    """, 
    font_weight="""
font_weight: str, Optional
    weight of the font , *default* is ``bold``.
    """, 
    font_style="""
font_style: str, Optional
    style of the font. *default* is ``italic``
    """, 
    font_size="""
font_size: float, Optional
    size of font in inches (width, height). *default* is ``3``.    
    """, 
    ms="""
ms: float, Optional 
    size of marker in points. *default* is ``5``
    """, 
    marker="""
marker: str, Optional
    marker of stations *default* is ``o``.
    """, 
    marker_facecolor="""
marker_style: str, Optional
    facecolor of the marker. *default* is ``yellow``    
    """, 
    marker_edgecolor="""
marker_edgecolor: str, Optional
    facecolor of the marker. *default* is ``yellow``
    """, 
    marker_edgewidth="""
marker_edgewidth: float, Optional
    width of the marker. *default* is ``3``.    
    """, 
    xminorticks="""
xminorticks: float, Optional
     minortick according to x-axis size and *default* is ``1``.
    """, 
    yminorticks="""
yminorticks: float, Optional
    yminorticks according to x-axis size and *default* is ``1``.
    """, 
    bins="""
bins: histograms element separation between two bar. *default* is ``10``. 
    """, 
    xlim="""
xlim: tuple (int, int), Optional
    limit of x-axis in plot. 
    """, 
    ylim="""
ylim: tuple (int, int), Optional
    limit of x-axis in plot. 
    """,
    xlabel="""
xlabel: str, Optional, 
    label name of x-axis in plot.
    """, 
    ylabel="""
ylabel: str, Optional, 
    label name of y-axis in plot.
    """, 
    rotate_xlabel="""
rotate_xlabel: float, Optional
    angle to rotate `xlabel` in plot.  
    """, 
    rotate_ylabel="""
rotate_ylabel: float, Optional
    angle to rotate `ylabel` in plot.  
    """, 
    leg_kws="""
leg_kws: dict, Optional 
    keyword arguments of legend. *default* is empty ``dict``
    """, 
    plt_kws="""
plt_kws: dict, Optional
    keyword arguments of plot. *default* is empty ``dict``
    """, 
    glc="""
glc: str, Optional
    line color of the grid plot, *default* is ``k``
    """, 
    glw="""
glw: float, Optional
   line weight of the grid plot, *default* is ``2``
    """, 
    galpha="""
galpha:float, Optional, 
    transparency number of grid, *default* is ``0.5``  
    """, 
    gaxis="""
gaxis: str ('x', 'y', 'both')
    type of axis to hold the grid, *default* is ``both``
    """, 
    gwhich="""
gwhich: str, Optional
    kind of grid in the plot. *default* is ``major``
    """, 
    tp_axis="""
tp_axis: bool, 
    axis to apply the ticks params. default is ``both``
    """, 
    tp_labelsize="""
tp_labelsize: str, Optional
    labelsize of ticks params. *default* is ``italic``
    """, 
    tp_bottom="""
tp_bottom: bool, 
    position at bottom of ticks params. *default* is ``True``.
    """, 
    tp_labelbottom="""
tp_labelbottom: bool, 
    put label on the bottom of the ticks. *default* is ``False``    
    """, 
    tp_labeltop="""
tp_labeltop: bool, 
    put label on the top of the ticks. *default* is ``True``    
    """, 
    cb_orientation="""
cb_orientation: str , ('vertical', 'horizontal')    
    orientation of the colorbar, *default* is ``vertical``
    """, 
    cb_aspect="""
cb_aspect: float, Optional 
    aspect of the colorbar. *default* is ``20``.
    """, 
    cb_shrink="""
cb_shrink: float, Optional
    shrink size of the colorbar. *default* is ``1.0``
    """, 
    cb_pad="""
cb_pad: float, 
    pad of the colorbar of plot. *default* is ``.05``
    """,
    cb_anchor="""
cb_anchor: tuple (float, float)
    anchor of the colorbar. *default* is ``(0.0, 0.5)``
    """, 
    cb_panchor="""
cb_panchor: tuple (float, float)
    proportionality anchor of the colorbar. *default* is ``(1.0, 0.5)``
    """, 
    cb_label="""
cb_label: str, Optional 
    label of the colorbar.   
    """, 
    cb_spacing="""
cb_spacing: str, Optional
    spacing of the colorbar. *default* is ``uniform``
    """, 
    cb_drawedges="""
cb_drawedges: bool, 
    draw edges inside of the colorbar. *default* is ``False`` 
    """     
)
return_docstring = """
        Returns
        -------
        str or None
            If buf is None, returns the result as a string. Otherwise returns
            None.
    """

refglossary =type ('refglossary', (), dict (
    __doc__="""\

.. _GeekforGeeks: https://www.geeksforgeeks.org/style-plots-using-matplotlib/#:~:text=Matplotlib%20is%20the%20most%20popular,without%20using%20any%20other%20GUIs

.. _IUPAC nommenclature: https://en.wikipedia.org/wiki/IUPAC_nomenclature_of_inorganic_chemistry

.. _Matplotlib scatter: https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.scatter.html
.. _Matplotlib plot: https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.plot.html
.. _Matplotlib pyplot: https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.plot.html
.. _Matplotlib figure: https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.figure.html
.. _Matplotlib figsuptitle: https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.suptitle.html

.. _Properties of water: https://en.wikipedia.org/wiki/Properties_of_water#Electrical_conductivity 
.. _pandas DataFrame: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
.. _pandas Series: https://pandas.pydata.org/docs/reference/api/pandas.Series.html

.. _scipy.optimize.curve_fit: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

"""
    ) 
)

"""
.. currentmodule:: gofast
"""
#.. |API change| replace:: :bdg-dark:`API change`
# Define replacements (used in whatsnew bullets)

gf_rst_epilog ="""

.. role:: raw-html(raw)
   :format: html

.. |Fix| replace:: :bdg-danger:`Fix`
.. |Enhancement| replace:: :bdg-info:`Enhancement`
.. |Feature| replace:: :bdg-success:`Feature`
.. |Major feature| replace:: :bdg-success:`Major feature`
.. |Major change| replace:: :bdg-primary:`Major change`
.. |API change| replace:: :bdg-warning:`API change`
.. |Deprecated| replace:: :bdg-warning:`Deprecated`

.. |Open Source? Yes!| image:: https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github
   :target: https://github.com/WEgeophysics/gofast
   
.. |License BSD| image:: https://img.shields.io/github/license/WEgeophysics/gofast?color=b&label=License&logo=github&logoColor=blue
   :alt: GitHub
   :target: https://github.com/WEgeophysics/gofast/blob/master/LICENSE
   
.. |simpleicons git| image:: https://img.shields.io/badge/--F05032?logo=git&logoColor=ffffff
   :target: http://git-scm.com 
   
.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.7744732.svg
   :target: https://doi.org/10.5281/zenodo.7744732
   
"""  

    























































