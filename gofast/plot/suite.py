# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Provides a comprehensive suite of plotting functions for 
visualizing model performance and data analysis.
"""
from __future__ import annotations 

import math
import copy 
import warnings
from numbers import Real, Integral 
import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt

from scipy.stats import probplot
from sklearn.metrics import r2_score, mean_squared_error 
from sklearn.metrics import precision_recall_curve

from ..api.types import Optional, Tuple,  Union, List 
from ..api.summary import ResultSummary
from ..compat.pandas import select_dtypes 
from ..core.array_manager import smart_ts_detector, drop_nan_in 
from ..core.checks import ( 
    is_iterable, check_params, check_numeric_dtype, 
    check_features_types, exist_features, 
)
from ..core.handlers import columns_manager, param_deprecated_message 
from ..core.plot_manager import ( 
    default_params_plot, 
    deprecated_params_plot, 
    return_fig_or_ax
)
from ..compat.sklearn import ( 
    validate_params,
    StrOptions, 
    Interval, 
    type_of_target
)
from ..decorators import isdf
from ..metrics import get_scorer 
from ..utils.mathext import compute_importances  
from ..utils.validator import  ( 
    build_data_if,
    is_frame, validate_yy, 
    filter_valid_kwargs
)
from ._config import PlotConfig
from .utils import _set_defaults, _param_defaults, flex_figsize 
from .utils import make_plot_colors 

__all__=[
     'plot_coverage',
     'plot_distributions',
     'plot_factory_ops',
     'plot_fit',
     'plot_perturbations',
     'plot_prediction_intervals',
     'plot_ranking',
     'plot_relationship',
     'plot_sensitivity',
     'plot_temporal_trends',
     'plot_uncertainty',
     'plot_with_uncertainty'
 ]

@return_fig_or_ax(return_type ='ax')
@default_params_plot(
    savefig=PlotConfig.AUTOSAVE("my_coverall_plot.png"), 
    fig_size =(8, 6), 
    dpi=300
    )
@validate_params ({ 
    'y_true': ['array-like'],
    'kind': [StrOptions({"line", "bar", "pie", "radar"}), None], 
    })
@check_params({
        "names": Optional[Union [str, List[str]]], 
        "q": Optional[Union[float, List[float]]]
    }, 
    coerce=False, 
)
def plot_coverage(
    y_true,
    *y_preds,
    names=None,
    q=None,
    kind='line',
    cmap='viridis',
    pie_startangle=140,
    pie_autopct='%1.1f%%',
    radar_color='tab:blue',
    radar_fill_alpha=0.25,
    radar_line_style='o-',
    cov_fill=False, 
    figsize=None,
    title=None,
    savefig=None,
    verbose=1 
):
    """
    Plot coverage scores for quantile or point forecasts and allow
    multiple visualization styles (line, bar, pie, and radar).

    This function computes and visualizes the fraction of times
    the true values :math:`y_i` lie within predicted quantile
    intervals or match point forecasts, for one or more models.
    If multiple prediction arrays are passed (e.g. from different
    models), this function compares their coverage on the same
    figure through different plot types.

    .. math::
        \\text{coverage} = \\frac{1}{N}\\sum_{i=1}^{N}
        1\\{\\hat{y}_{i}^{(\\ell)} \\leq y_i
        \\leq \\hat{y}_{i}^{(u)}\\}

    where :math:`\\hat{y}_{i}^{(\\ell)}` is the lower quantile
    prediction for the :math:`i`th sample and :math:`\\hat{y}_{i}^{
    (u)}` is the upper quantile prediction. The indicator function
    :math:`1\\{\\cdot\\}` counts how many times the true value
    :math:`y_i` lies within or on the boundaries of the predicted
    interval.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target values.

    *y_preds : one or more array-like objects, each of shape
        (n_samples,) or (n_samples, n_quantiles)
        Predicted values from one or more models. If a 2D array is
        passed, its columns are considered to be predictions for
        different quantiles. If a 1D array is passed, it is treated
        as a point forecast.

    names : list of str or None, optional
        Names for each set of predictions. If None, default names
        (e.g. "Model_1") are generated. If the length of `names`
        is less than the number of prediction arrays, the rest
        are auto-generated.

    q : list of float or None, optional
        Quantile levels for each column of the 2D prediction arrays.
        If provided, predictions for each row are sorted in ascending
        order, and coverage is computed between the minimum and
        maximum quantile predictions. If None, coverage is assumed
        to be a point forecast unless a different approach is
        implemented by the user.

    kind : str, optional (default='line')
        Type of plot to use for displaying coverage. Possible
        values are:
        
        - ``'line'``: Plots a line chart of coverage scores.
        - ``'bar'``: Plots a bar chart of coverage scores.
        - ``'pie'``: Creates a pie chart where each slice
          corresponds to a model's coverage fraction relative
          to the total coverage sum.
        - ``'radar'``: Creates a radar chart placing each model's
          coverage on a radial axis.

    cmap : str, optional (default='viridis')
        Colormap used in the pie chart. Each model slice is
        assigned a color from this colormap. Also used more
        generally if extended.

    pie_startangle : float, optional (default=140)
        Start angle for the pie chart in degrees.

    pie_autopct : str, optional (default='%1.1f%%')
        Format of the numeric label displayed on each pie slice.

    radar_color : str, optional (default='tab:blue')
        Main line and fill color for the radar chart.

    radar_fill_alpha : float, optional (default=0.25)
        Alpha blending value for the filled area in the radar chart,
        controlling transparency.

    radar_line_style : str, optional (default='o-')
        Marker and line style for the coverage in the radar chart,
        for instance ``'o-'`` or ``'-'``.
        
    cov_fill : bool, default=False
        Enable gradient fill for radar plots. For single models, creates
        a radial gradient up to coverage value. For multiple models,
        fills polygon areas.
        
    figsize : tuple of float, optional
        Figure size (width, height) in inches passed to matplotlib.

    title : str or None, optional
        Title for the plot. If None, no title is displayed.

    savefig : str or None, optional
        Filename (and extension) for saving the figure. If None,
        the figure is only displayed and not saved.

    verbose : int, default=1
        Control coverage score printing:
            - 0: No output
            - 1: Print formatted coverage summary
       
    Returns
    -------
    None
        This function renders a coverage plot and may save it,
        depending on the `savefig` argument.

    Notes
    -----
    - If `q` is specified and the predictions are 2D, the first
      and last columns of the sorted prediction array determine
      the coverage interval. Intermediate quantile columns are
      not used directly but may be relevant in other analyses.
    - If the predictions are 1D point forecasts, coverage is
      computed as the fraction of exact matches
      (:math:`\\hat{y}_i = y_i`), which typically remains 0
      unless the data are discrete or artificially matched.
    - Different plot types offer various perspectives:
      - Bar or line charts present coverage per model on a
        simple numerical scale (0 to 1).
      - Pie charts represent each model's coverage fraction
        out of the sum of coverages. 
      - Radar charts place each model's coverage on a radial
        axis for a comparative "spider" plot.
        
    1. For quantile predictions (2D arrays), coverage is computed between
       the minimum and maximum quantiles per observation
    2. Point forecast coverage (1D arrays) measures exact matches, which
       is typically near-zero for continuous data
    3. Radar plots with ``cov_fill=True`` display:
        - Gradient fill from center to coverage value (single model)
        - Transparent polygon fill (multiple models)
        - Red reference line at coverage level (single model)

    See Also
    --------
    gofast.plot.plot_roc : Receiver operating characteristic curve plotting
    gofast.plot.plot_residuals : Diagnostic residual analysis plots

    References
    ----------
    .. [1] Koenker, R. and Bassett, G. (1978). "Regression
           quantiles." *Econometrica*, 46(1), 33–50.

    Examples
    --------
    >>> import numpy as np
    >>> from gofast.plot.suite import plot_coverage
    >>> # True values
    >>> y_true = np.random.rand(100)
    >>> y_pred = np.random.rand(100, 3)
    >>> # 3-quantile predictions for a single model
    >>> y_pred_q = np.random.rand(100, 3)
    >>> q = [0.1, 0.5, 0.9]
    >>> # Bar chart coverage
    >>> plot_coverage(y_true, y_pred_q, q=q,
    ...               names=['QuantModel'],
    ...               kind='bar',
    ...               title='Coverage (Bar)')
    # Single model quantile coverage
    >>> y_pred = np.random.rand(200, 3)
    >>> plot_coverage(y_true, y_pred, q=[0.1, 0.5, 0.9],
    ...               kind='radar', names=['QModel'],
    ...               cov_fill=True, cmap='plasma')
    >>> # Multiple models with radar plot
    >>> y_pred_q2 = np.random.rand(100, 3)
    >>> plot_coverage(y_true, y_pred_q, y_pred_q2,
    ...               q=q,
    ...               names=['Model1','Model2'],
    ...               kind='radar',
    ...               title='Coverage (Radar)')
    """

    # Convert the true values to a numpy array for consistency
    y_true = np.array(y_true)

    # Count how many model predictions were passed via *y_preds.
    num_models = len(y_preds)

    # Handle model names: create or extend to match the number of models.
    names = columns_manager(names, to_string=True)
    if names is None:
        names = [f"Model_{i + 1}" for i in range(num_models)]
    else:
        if len(names) < num_models:
            extra = num_models - len(names)
            for i in range(extra):
                names.append(f"Model_{len(names) + 1}")

    coverage_scores = []

    q= columns_manager(q)
    # Handle quantiles
    if q is not None:
        q = np.array(q)
        if q.ndim != 1:
            raise ValueError(
                "Parameter 'q' must be a 1D list or"
                " array of quantile levels."
                )
            
        if not np.all((0 < q) & (q < 1)):
            raise ValueError(
                "Quantile levels must be between 0 and 1."
            )
        # Sort q and get the sorted indices
        sorted_indices = np.argsort(q)
        q_sorted = q[sorted_indices]
    else:
        q_sorted = None
        
    # Compute coverage for each model in *y_preds.
    #   - If pred has shape (n_samples, n_quantiles), we compute coverage
    #     between min and max quantile per sample.
    #   - If pred is 1D, treat as a point forecast and check exact match
    #     (illustrative; typically coverage would be 0 unless data match).
    for i, pred in enumerate(y_preds):
        pred = np.array(pred)

        #if (q is not None) and (pred.ndim == 2):
        if pred.ndim == 2:
            if q_sorted is not None: 
                # No need since we used the first and last for 
                # computed coverage. 
                # --------------------
                # if pred.shape[1] != len(q_sorted):
                #     raise ValueError(
                #         f"Model {i+1} predictions have"
                #         f"{pred.shape[1]} quantiles, "
                #         f"but 'q' has {len(q_sorted)} levels."
                #     )
                # ---------------------
                # Align predictions with sorted quantiles
                pred_sorted = pred[:, sorted_indices]
            else: 
                pred_sorted = np.sort(pred, axis=1)
                
            # Sort columns to ensure ascending order of quantiles.
            # pred_sorted = np.sort(pred, axis=1)
            lower_q = pred_sorted[:, 0]
            upper_q = pred_sorted[:, -1]
            in_interval = (
                (y_true >= lower_q) & (y_true <= upper_q)
            ).astype(int)
            coverage = np.mean(in_interval)

        elif pred.ndim == 1:
            # Point forecast coverage as fraction of exact matches
            matches = (y_true == pred).astype(int)
            coverage = np.mean(matches)

        else:
            # If neither scenario applies, store None.
            coverage = None

        coverage_scores.append(coverage)

    # Prepare data for plotting. Replace None with 0 for convenience.
    valid_cov = [
        c if c is not None else 0 for c in coverage_scores
    ]
    x_idx = np.arange(num_models)
    
    if kind in {'bar', 'line', 'pipe'}: 
        # Initialize the figure.
        if figsize is not None:
            plt.figure(figsize=figsize)
        else:
            plt.figure()
    # Plot according to the chosen 'kind'.
    if kind == 'bar':
        plt.bar(x_idx, valid_cov, color='blue', alpha=0.7)
        for idx, val in enumerate(coverage_scores):
            if val is not None:
                plt.text(
                    x=idx,
                    y=val + 0.01,
                    s=f"{val:.2f}",
                    ha='center',
                    va='bottom'
                )
        plt.xticks(x_idx, names)
        plt.ylim([0, 1])
        plt.ylabel("Coverage")
        plt.xlabel("Models")

    elif kind == 'line':
        plt.plot(x_idx, valid_cov, marker='o')
        for idx, val in enumerate(coverage_scores):
            if val is not None:
                plt.text(
                    x=idx,
                    y=val + 0.01,
                    s=f"{val:.2f}",
                    ha='center',
                    va='bottom'
                )
        plt.xticks(x_idx, names)
        plt.ylim([0, 1])
        plt.ylabel("Coverage")
        plt.xlabel("Models")

    elif kind == 'pie':
        # Pie chart: each slice represents a model's coverage. By default,
        # the slice size is coverage[i] out of the sum of coverage.
        total_cov = sum(valid_cov)
        if total_cov == 0:
            # Avoid a zero-coverage pie chart.
            plt.text(
                0.5, 0.5,
                "No coverage to plot",
                ha='center',
                va='center'
            )
        else:
            plt.pie(
                valid_cov,
                labels=names,
                autopct=pie_autopct,
                startangle=pie_startangle,
                colors=plt.cm.get_cmap(cmap)(
                    np.linspace(0, 1, num_models)
                )
            )
            plt.axis('equal')  # Make the pie chart a perfect circle.

    elif kind == 'radar':
        # #Radar chart: place each model's coverage as a radial axis.

        N = num_models
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        coverage_radar = np.concatenate((valid_cov, [valid_cov[0]]))
        
        ax = plt.subplot(111, polar=True)
        
        # Plot main coverage line
        ax.plot(
            angles,
            coverage_radar,
            radar_line_style,
            color=radar_color,
            label='Coverage'
        )

        # Handle fill based on number of models
        if cov_fill:
            if num_models == 1:
                # Single model: radial gradient fill up to coverage value
                coverage_value = valid_cov[0]
                theta = np.linspace(0, 2 * np.pi, 100)
                r = np.linspace(0, coverage_value, 100)
                R, Theta = np.meshgrid(r, theta)
                
                # Create gradient using specified colormap
                ax.pcolormesh(
                    Theta, R, R, 
                    cmap=cmap, 
                    shading='auto', 
                    alpha=radar_fill_alpha,
                    zorder=0  # Place behind main plot
                )
                # Add red circle at coverage value
                ax.plot(
                    theta, 
                    [coverage_value] * len(theta),  # Constant radius
                    color='red', 
                    linewidth=2, 
                    linestyle='-',
                    # label=f'Coverage Value ({coverage_value:.2f})'
                )
                
            # Add concentric grid circles at 0.2, 0.4, 0.6, 0.8 
            # with correct properties
                ax.set_ylim(0, 1)
                ax.set_yticks([0.2, 0.4, 0.6, 0.8])
                ax.yaxis.grid(
                    True, 
                    color="gray", 
                    linestyle="--", 
                    linewidth=0.5, 
                    alpha=0.7
                )
            
            else:
                # Multiple models: transparent fill between center and line
                ax.fill(
                    angles,
                    coverage_radar,
                    color=radar_color,
                    alpha=radar_fill_alpha,
                    zorder=0
                )
        # Final formatting
        ax.set_thetagrids(angles[:-1] * 180/np.pi, labels=names)
        ax.set_ylim(0, 1)
        plt.legend(loc='upper right')

    else:
        # Fallback: print coverage scores to the console for each model.
        for idx, val in enumerate(coverage_scores):
            print(f"{names[idx]} coverage: {val}")

    if verbose:
       cov_dict = {
           names[idx]: cov 
           for idx, cov in enumerate(coverage_scores)
           }
       summary = ResultSummary(
           "CoverageScores").add_results (cov_dict)
       print(summary)
       
    # Add title if provided.
    if title is not None:
        plt.title(title)
        
    if savefig is not None:
        plt.savefig(savefig, bbox_inches='tight')

    plt.show()


@return_fig_or_ax(return_type ='ax')
@default_params_plot(
    savefig=PlotConfig.AUTOSAVE("my_ranking_plot.png"), 
    fig_size =None, 
    dpi=300
    )
@validate_params ({ 
    'kind': [StrOptions({"auto", "ranking", "importance"}), None], 
    'features': [str, 'array-like', None]
    })
def plot_ranking(
    X,
    y=None,
    models=None,
    features=None,
    precomputed=False,
    xai_methods=None,
    kind=None,
    prefit=True,
    annot=True,
    pkg=None,
    normalize=False,
    fmt="auto",
    cmap="Purples_r",
    figsize=None, 
    cbar='off',
    savefig=None, 
    **kw
):
    r"""
    Visualize model-driven feature rankings or importances
    as a heatmap. This utility can handle two scenarios:

    1) **Computing** importances/ranks using
       :func:`~gofast.utils.mathext.compute_importances`
       if ``precomputed=False``.
    2) **Plotting** a user-supplied matrix of importances
       or ranks if ``precomputed=True``.

    .. math::
        \text{Rank}_{ij} = \begin{cases}
          1 & \text{(most important feature for model $j$)} \\
          2 & \text{(second most important)}, \ldots
        \end{cases}

    Parameters
    ----------
    X : array-like or pandas.DataFrame
        Feature matrix or already computed ranking/importances
        when ``precomputed=True``. If it is a raw dataset and
        ``precomputed=False``, the function tries to compute
        feature importances or ranks. If it is a precomputed
        matrix of shape (n_features, n_models), the function
        plots it directly.
    y : array-like or None, optional
        Target vector if new models are to be fitted for
        computing feature importances, or for certain XAI
        methods. Not used if the data is already precomputed
        or if no model fitting is needed.
    models : list or dict or None, optional
        Model estimators or dictionary of named estimators.
        If ``precomputed=False`` and models is ``None``,
        default ones are created (e.g. random forest).
        If ``precomputed=True``, can be used to rename
        columns in the final plot if shapes match.
    features : list of str or None, optional
        Feature names. If computing importances, tries to
        use them from ``X`` if it is a DataFrame. If the
        matrix is already computed, you can supply them for
        row labeling if shape matches.
    precomputed : bool, optional
        If ``True``, indicates ``X`` is already a
        (ranking/importances) matrix. Otherwise, the function
        calls :func:`~gofast.utils.mathext.compute_importances`
        to generate them from the given models.
    xai_methods : callable, optional
        Custom function for computing feature importances in
        :func:`compute_importances`. If provided, overrides
        the built-in approaches.
    kind : {'ranking', 'importance', 'auto', None}, optional
        - ``'ranking'``: Ensures the function returns a matrix
          of integer ranks.
        - ``'importance'``: Produces floating-point importances.
        - ``'auto'``: If ``precomputed=True``, tries to infer
          from the matrix dtype (integer => rank, float =>
          importance).
        - ``None``: (default) the function produces ranks if it
          is computing them from scratch.
    prefit : bool, optional
        If ``True``, user-provided models are assumed already
        trained. If ``False``, the function fits them on
        (X, y). Ignored if ``precomputed=True``.
    annot : bool, optional
        Whether to annotate each cell in the heatmap with its
        value. Good for smaller matrices.
    pkg : {'sklearn', 'shap', None}, optional
        Backend for computing importances if not precomputed.
        Defaults to ``'sklearn'``. If you want SHAP values,
        choose ``'shap'``.
    normalize : bool, optional
        Whether to normalize columns if computing importances.
        Each column can be scaled so its sum is 1. Ignored if
        the matrix is precomputed.
    fmt : str, optional
        Format string for heatmap annotations, e.g. ``'d'``
        for integers (for ranking visualization), ``'.2f'`` for
        floats (for importances visualization) if ``fmt='auto'``.
        
    cmap : str, optional
        Colormap for the heatmap. Default is ``"Purples_r"``.
    figsize : tuple of (float, float), optional
        Figure dimensions for the heatmap. Default is (4, 12),
        a tall layout suitable for many features.
    cbar : {'off', True, False}, optional
        Whether to display the color bar. ``'off'`` or
        ``False`` hides it.
    **kw : dict, optional
        Additional keyword arguments passed to 
        :func:`seaborn.heatmap`. For example, ``linewidths``,
        ``linecolor``, etc.

    Notes
    -----
    This function primarily displays a heatmap where rows
    correspond to features and columns to models. The cell
    values can be either rank or raw importances. If multiple
    models exist, you can quickly compare how each ranks or
    values each feature [1]_.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from gofast.plot.suite import plot_ranking
    >>> X = pd.DataFrame({
    ...     'f1': np.random.randn(100),
    ...     'f2': np.random.randn(100)
    ... })
    >>> y = np.random.randint(0, 2, size=100)
    >>> # Plot a ranking from default models
    >>> plot_ranking(X, y, kind='ranking', figsize=(4,6))

    See Also
    --------
    compute_importances : Helper function that returns either
        feature ranks or importances.

    References
    ----------
    .. [1] Lundberg, S.M., & Lee, S.-I. (2017). A unified
           approach to interpreting model predictions.
           *Advances in Neural Information Processing
           Systems*, 30, 4768-4777.
    """
    # Check whether the data is already precomputed (ranking or importances)
    # or if it needs to be computed using the provided models
    if not precomputed:
        # Decide whether to retrieve ranking or feature importances 
        # based on the user-specified kind
        return_rank = (kind is None or kind == 'ranking')
        
        # Call compute_importances from gofast mathext utilities
        df_result = compute_importances(
            models=models,
            X=X,
            y=y,
            prefit=prefit,
            pkg=pkg,
            as_frame=True,
            xai_methods=xai_methods,
            ascending=False, 
            return_rank=return_rank,
            normalize=normalize
        )
        # If we computed ranking, the dataframe is already ranking_matrix
        # Otherwise, it's the importances
        matrix_to_plot = df_result
        
        # Determine if the matrix is ranking or importances for labeling
        matrix_kind = 'ranking' if return_rank else 'importance'
    
    else:
        # If data is precomputed, we interpret X directly
        # If data_kind='auto', guess by dtype: integer => ranking, float => importances
        # If data_kind is explicitly 'ranking' or 'importances', use that
        matrix_kind = kind or "auto"
        
        # If 'auto', check the dtype to guess if ranking or importances
        if matrix_kind == 'auto':
            if np.issubdtype(np.array(X).dtype, np.integer):
                matrix_kind = 'ranking'
            else:
                matrix_kind = 'importance'
        
        # Convert to DataFrame for easier handling
        matrix_to_plot = (
            X if isinstance(X, pd.DataFrame)
            else pd.DataFrame(X)
        )
        
        # If user gave explicit feature names, try to apply them
        if features is not None:
            features= columns_manager(features, empty_as_none= True)
            # If shape mismatch occurs, fallback with a warning or ignore
            if len(features) == matrix_to_plot.shape[0]:
                matrix_to_plot.index = features
            else:
                # warn the user here if lengths don't match
                warnings.warn(
                    "Mismatch between 'features' length and the "
                    "number of rows in the matrix: "
                    f"features={len(features)}, rows="
                    f"{matrix_to_plot.shape[0]}. Index not renamed."
                )
    
        # If user provided model columns or a single string
        # and shape matches, rename columns
        if isinstance(models, list) and len(models) == matrix_to_plot.shape[1]:
            matrix_to_plot.columns = models
        elif isinstance(models, str):
            # Potentially rename a single column if shape is 1
            if matrix_to_plot.shape[1] == 1:
                matrix_to_plot.columns = [models]
            else:
                # warn the user here if mismatch
                warnings.warn(
                    "Mismatch between 'models' length and columns in matrix: "
                    f"models=1, columns={matrix_to_plot.shape[1]}. "
                    "Column names not renamed."
                )
        elif isinstance(models, (list, dict)):
            # Potentially handle dict keys as column names if lengths match
            if isinstance(models, dict) and len(models) == matrix_to_plot.shape[1]:
                matrix_to_plot.columns = list(models.keys())
            else:
                # Fallback as we can't rename properly
                warnings.warn(
                "A single model name was provided as a string,"
                " but the matrix has multiple columns (columns:"
                f" {matrix_to_plot.shape[1]}). Column names"
                " not renamed. "
             )
    # Prepare the heatmap to visualize either ranking or importances
    if figsize is None: 
        figsize = flex_figsize(
            matrix_to_plot, 
            figsize=figsize, 
            base= (3, 12) if kind=='ranking' else (4, 12), 
            min_base=(3, 7)
            
            )
    plt.figure(figsize=figsize)
    
    # If matrix is of ranking, we typically use integer fmt
    # If matrix is importances, we might prefer a float format
    plot_title = (
        "Feature Rankings Across Models" if matrix_kind in ['ranking', 'rank']
        else "Feature Importances Across Models"
    )
    
    # Create the heatmap using Seaborn
    if fmt=='auto': 
        fmt="d" if matrix_kind=='ranking' else ".2f"
    
    kw = filter_valid_kwargs(sns.heatmap, kw)
    lw= kw.pop('linewidths', 2)

    sns.heatmap(
        matrix_to_plot,
        annot = annot,
        fmt = fmt,
        cmap = cmap,
        cbar= False if cbar in ['off', False] else True, 
        linewidths=lw,
        **kw
    )
    # Label the axes
    plt.xlabel("Models")
    plt.ylabel("Features")
    
    # Add a title reflecting the plotted data type
    plt.title(plot_title)
    
    # Display the plot
    plt.tight_layout()
    plt.show()


@return_fig_or_ax(return_type ='ax')
@default_params_plot(
    savefig=PlotConfig.AUTOSAVE("my_factory_ops_plot.png"), 
    fig_size =(10, 8), 
    dpi=300, 
    )
@check_params({ 
    "names": Optional[Union[str, List[str]]], 
    "title": Optional[str], 
    'figsize': Optional[Tuple[int, int]], 
    }, 
    coerce=False, 
 )
@validate_params({ 
    'train_times': ['array-like', None], 
    'metrics': [str, 'array-like', None], 
    'scale': [StrOptions({"norm", "min-max", 'std', 'standard'}), None], 
    "lower_bound": [Real], 
    })
def plot_factory_ops(
    y_true,
    *y_preds,
    train_times=None,
    metrics=None,
    names=None,
    title=None,
    figsize=None,
    colors=None,
    alpha=0.7,
    legend=True,
    show_grid=True,
    scale='norm',  
    lower_bound=0, 
    savefig=None,
    loc='upper right', 
    verbose=0,
):
    r"""
    Generates a radar chart (also called a spider chart) to
    illustrate and compare multiple  metrics across
    different models. Internally relies on `drop_nan_in`,
    function to drop NaN values in `y_true` and `y_preds`
    to ensure data consistency and metric fetching in
    harmony with scikit-learn and gofast libraries [1]_.
    
    .. math::
       M(y, \hat{y})
       = f(\{(x_i, y_i)\}_{i=1}^n)
    
    In a typical use case, let :math:`y` be the true values and
    :math:`\hat{y}` the predicted values. For each chosen metric
    (for example, :math:`R^2`), the function computes:
    :math:`M(y, \hat{y})` across all models, arranges the scores
    in a radial layout, and optionally includes training times
    to offer a holistic view of performance [2]_.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        The ground truth values.
    *y_preds : array-like of shape (n_samples,), optional
        The predicted values from one or more models. Each model
        corresponds to one array. If multiple arrays are passed,
        each one is plotted as a separate polygon.
    train_time : float, list of floats, optional
        The time taken (in seconds) to train each model. If
        provided, it is displayed as an additional metric. If
        only a single float is given, it is assumed the same for
        all models.
    metrics : list of {str, callable}, optional
        The metrics to compute for each model's predictions. A
        string will be used to fetch the built-in or scikit-learn
        metric via ``get_scorer``. A callable can be any function
        following the scikit-learn metric API. Defaults to
        ``["r2", "mae", "mape", "rmse",]`` for regression and 
        ``["accuracy", "precision", "recall"]`` for classification
        task.
    names : list of str, optional
        The names corresponding to each model. If not provided,
        defaults to `Model_1`, `Model_2`, etc. The length of
        `names` should match the number of models (predictions).
    title : str, optional
        The title displayed at the top of the radar chart.
    figsize : tuple of int, optional
        Figure dimension in inches, default is ``(8, 8)``.
    colors : list of str or None, optional
        Color codes for each model's polygon. If not provided,
        auto-generated from matplotlib's color palette.
    alpha : float, optional
        The opacity for the polygon lines and fill regions,
        default is `0.7` for lines and `0.1` for fill.
    legend : bool, optional
        Whether to display a legend indicating model names.
    show_grid : bool, optional
        Whether to show a radial grid on the radar chart.
    scale : {'norm', 'scale', None}, optional
        The transformation applied to metric values:
        - `'norm'`: min-max normalization per metric.
        - `'std'`: standard scaling per metric.
        - `None`: no scaling is applied.
    lower_bound : float, optional
        The lower boundary for the radar chart radius, default
        is `0`.
    savefig : str, optional
        File path to save the resulting figure. If `None`,
        the figure is not saved.
    verbose : int, optional
        Controls the level of verbosity:
        - `0`: silent mode,
        - `1` or higher: prints debug information, such as
          metric scaling steps.
    
    Notes
    -----
    If `train_times` is specified, the resulting radar chart
    will include an additional axis labeled
    ``train_time_s``. This can be useful to balance training
    efficiency against accuracy or other error-based metrics.
    All metrics are calculated using :math:`M(y, \hat{y})`
    for each model, where :math:`M` depends on the chosen
    metrics (like `r2`, `mape`, etc.).
    
    Examples
    --------
    >>> from gofast.plot.suite import plot_factory_ops
    >>> import numpy as np
    >>> # Assume y_true and y_preds are NumPy arrays of shape (100,)
    >>> y_true = np.random.rand(100)
    >>> y_pred1 = np.random.rand(100)
    >>> y_pred2 = np.random.rand(100)
    >>> # Visualize with two models, using default metrics
    >>> plot_factory_ops(y_true, y_pred1, y_pred2,
    ...                  names=["RandomModel1", "RandomModel2"],
    ...                  title="Example Radar Chart",
    ...                  train_times=[0.5, 0.3],
    ...                  verbose=1)
    
    See Also
    --------
    gofast.core.array_manager.drop_nan_in : Removes NaN from arrays to ensure
      shape consistency.
    gofast.utils.validator.validate_yy : Validates dimension and data type
      of the true and predicted arrays.
    gofast.metrics.get_scorer: Retrieves metric callables by
      name from gofast and scikit-learn.
    
    References
    ----------
    .. [1] Doe, J. et al. (2023). "Advanced Radar Charts
       in Machine Learning: A Comprehensive Overview."
       Journal of Data Visualization, 12(3), 45-59.
    .. [2] Kenny-Jesús F. , Alejandro E., María-Luisa M.z and Pablo C.(2024).
      "Lead-Time Prediction in Wind Tower Manufacturing: A Machine
       Learning-Based Approach." Mathematics,12, 2347.  
       https://doi.org/10.3390/math12152347
       
    """

    # Remove NaN values and ensure consistency for y_true and each y_pred
    y_true, *y_preds = drop_nan_in(y_true, *y_preds, error='raise')
    y_preds = [
        validate_yy(y_true, pred, expected_type="continuous", flatten=True)[1]
        for pred in y_preds
    ]
    # Generate default model names if none are provided
    if names is None:
        names = [f"Model_{i+1}" for i in range(len(y_preds))]

    names = columns_manager(names, empty_as_none= False ) 
    if len(names) < len(y_preds):
        names += [f"Model_{i+1}" for i in range(len(names), len(y_preds))]

    # Set default metrics if none are provided
    if metrics is None:
        if type_of_target(y_true) in ['continuous', 'continuous-multioutput']: 
            metrics = [ "mae", "mape", "rmse", "r2"]
        else: 
            # assume classification target
            metrics = ["accuracy", "precision", "recall" ]
            
    metrics = is_iterable(metrics, exclude_string=True, transform=True)

    # Fetch metric functions from gofast (or scikit-learn),
    # or use user-provided callables
    metric_funcs = []
    metric_names = []
    for metric in metrics:
        if isinstance(metric, str):
            scorer = get_scorer(metric, include_sklearn=True)
            metric_funcs.append(scorer)
            metric_names.append(metric)
        elif callable(metric):
            metric_funcs.append(metric)
            metric_names.append(metric.__name__)

    # Prepare arrays to collect metric values for each model
    # If train_time is provided, we treat it as an additional “metric”
    if train_times is not None:
        if not isinstance(train_times, (list, tuple, np.ndarray)):
            train_times = [train_times] * len(y_preds)
        if len(train_times) < len(y_preds):
            warnings.warn("train_times length is smaller than number of models.")
            # Fill missing values or handle accordingly
        metric_names.append("train_time_s")
        # We'll append None to metric_funcs to treat it separately
        metric_funcs.append(None)

        check_numeric_dtype(train_times, param_names={"X": "Train times"})
        
    # Calculate each metric for every model
    results = []
    for idx, y_pred in enumerate(y_preds):
        row_values = []
        for m_idx, metric_func in enumerate(metric_funcs):
            if metric_func is not None:
                val = metric_func(y_true, y_pred)
                row_values.append(val)
            else:
                # This path is used to handle 'train_time'
                row_values.append(
                    train_times[idx] if train_times is not None else None
                    )
        results.append(row_values)

    # Convert metric results to a NumPy array for plotting
    results_arr = np.array(results, dtype=float)

    # Optional scaling / normalization of the metrics
    if scale in [ 'norm', 'min-max']:
        # Min-max normalization for each metric (column-wise)
        if verbose > 0:
            print("[DEBUG] Performing min-max normalization on metrics.")
        min_val = np.nanmin(results_arr, axis=0)
        max_val = np.nanmax(results_arr, axis=0)
        # Avoid zero division if max == min
        denom = np.where((max_val - min_val) == 0, 1, (max_val - min_val))
        results_arr = (results_arr - min_val) / denom

    elif scale in [ 'standard', 'std']:
        # Standard scaling for each metric (column-wise)
        if verbose > 0:
            print("[DEBUG] Performing standard scaling on metrics.")
        mean_val = np.nanmean(results_arr, axis=0)
        std_val = np.nanstd(results_arr, axis=0)
        # Avoid zero division if std == 0
        std_val = np.where(std_val == 0, 1, std_val)
        results_arr = (results_arr - mean_val) / std_val

    if verbose > 0:
        print(f"[DEBUG] Final metric array for plotting:\n{results_arr}")
        
    # Create a polar subplot for radar chart
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, polar=True)

    # Determine angles for each metric
    # The last angle is repeated to close the polygon
    num_metrics = len(metric_names)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    # If no colors are provided, generate a simple palette
    if not colors:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i) for i in range(len(results_arr))]

    # Plot each model’s polygon
    for i, row in enumerate(results_arr):
        # Append first value again to close the polygon
        values = np.concatenate((row, [row[0]]))
        ax.plot(angles, values, label=names[i], color=colors[i], alpha=alpha)
        ax.fill(angles, values, color=colors[i], alpha=0.1)

    # Set up the radial grid and labels
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels=metric_names)
    # suitable lower bound to fix negative values
    ax.set_ylim(bottom=lower_bound)  

    # Optionally show a grid
    if show_grid:
        ax.grid(True, linestyle='--', alpha=0.7)
    else : 
        ax.grid(False)
        
    # Optionally show legend
    if legend:
        ax.legend(loc=loc, bbox_to_anchor=(1.1, 1.1))

    # If you gave a title, set it
    if title is not None:
        ax.set_title(title, y=1.1, fontsize=12)
    
    if savefig is not None:
        plt.savefig(savefig, bbox_inches='tight')

    # Display or return the figure
    plt.tight_layout()
    plt.show()



@return_fig_or_ax
def plot_perturbations(
    X,
    y,
    model=None,
    perturbations=0.05,
    max_iter=10,
    metric='miv',
    kind='bar',
    percent=False,
    relative=False,
    normalize=False, 
    show_grid=True,
    fig_size=(12, 8),
    cmap='Blues',
    max_cols=3,
    titles=None,
    savefig=None,
    *,
    rotate_labels: Optional[str] = None,
    rotate: Optional[float] = None,
    display_values: bool = True,
    **kwargs
):
    r"""
    Plot feature perturbation effects for multiple perturbation values
    using MIV metrics.

    The ``plot_perturbations`` function calls :math:`miv_score` for each
    value in ``perturbations`` and aggregates the results to visualize
    multiple scenario outcomes. This helps in assessing how different
    levels of feature perturbation affect model responses. Multiple
    subplots are created, each reflecting one perturbation magnitude,
    allowing for direct comparison.

    .. math::
       \\text{Perturbation Plot}:
       \\begin{cases}
         \\text{Use MIV to measure } 
         \\Delta \\text{model output w.r.t.}\\
         \\text{feature changes}, & \\text{for different } 
         \\text{perturbation scales}
       \\end{cases}

    Parameters
    ----------
    X : array-like or pandas.DataFrame
        The feature matrix on which MIV calculations will be performed.
        Should not include the target variable.
    
    y : array-like
        The target variable array. If provided, supervised MIV is
        computed; if ``None``, unsupervised approaches may be used.
    
    model : estimator object, optional
        A trained model (e.g., ``RandomForestClassifier``). If not
        provided, a default model is instantiated based on the target
        nature (regression/classification).
    
    perturbations : float or list of float, default=0.05
        One or multiple values to scale the feature perturbation. Each
        is passed to `miv_score`. This allows for comparison across
        multiple perturbation levels.
    
    max_iter : int, default=10
        Number of iterations for re-fitting or re-predicting during
        MIV calculations. Higher values yield more stable estimates.
    
    metric : str, default='miv'
        The metric to compute. Currently, only `'miv'` or `'m.i.v.'`
        are accepted. Raises an error if another metric is given.
    
    kind : {'bar', 'barh', 'pie', 'scatter'}, default='bar'
        The style of the final subplots. Each subplot shows the feature
        importance distribution for a specific perturbation magnitude:
        
        - ``'bar'``: Vertical bars for each feature.
        - ``'barh'``: Horizontal bars.
        - ``'pie'``: Pie chart of relative contributions.
        - ``'scatter'``: Feature points sized by MIV.
    
    percent : bool, default=False
        If ``True``, final MIV values are displayed as percentages;
        otherwise, raw numeric values are used.
    
    relative : bool, default=False
        If ``True``, uses the original model predictions to compute a
        relative MIV. Otherwise, MIV is the difference between positive
        and negative perturbations.
        
    normalize : bool, optional, default=False
        Whether to normalize the contributions using min-max scaling. If 
        `True`, the values will be scaled to the range defined in 
        ``norm_range``.
    
    show_grid : bool, default=True
        If ``True``, displays grid lines on certain plot types (bar,
        barh, scatter). Improves readability of the data distribution.
    
    fig_size : tuple of int, default=(12, 8)
        The size of the matplotlib figure in inches (width, height).
    
    cmap : str, default='Blues'
        The color palette for plotting. Accepts any valid matplotlib
        colormap name.
    
    max_cols : int, default=3
        Maximum number of columns in the subplot grid. Additional
        perturbation results create new rows if needed.
    
    titles : list of str, optional
        Custom subplot titles. Must match or exceed the number of
        perturbation values if provided; otherwise, default titles
        are generated as "Perturbation=<value>".
    
    savefig : str, optional
        If provided, saves the resulting multi-subplot figure to the
        specified filepath or filename. Supported formats depend on
        matplotlib.
    
    rotate_labels : {'feature', 'value', 'both', None}, optional
        Controls the rotation of text in the plot:
        
        - ``'feature'``: Rotate only feature (axis) labels.
        - ``'value'``: Rotate only numeric bar or point text.
        - ``'both'``: Rotate both feature labels and bar/point text.
        - ``None``: No rotation is applied.
    
    rotate : float, optional
        The angle (in degrees) at which to rotate labels or text.
        Used with `rotate_labels`.
    
    display_values : bool, default=True
        If ``True``, numeric MIV values are annotated on bars or
        scatter points. If ``False``, the numeric text is suppressed.
    
    **kwargs
        Additional keyword arguments reserved for future extension.

    Returns
    -------
    None
        Displays subplots showing MIV distributions at each
        perturbation magnitude. Optionally saves a figure if
        ``savefig`` is specified.

    Raises
    ------
    ValueError
        - If an unsupported `metric` is requested.
        - If `perturbations` cannot be interpreted as numeric
          values.
        - If `kind` is not among the supported options.

    Examples
    --------
    >>> from gofast.plot.suite import plot_perturbations
    >>> import pandas as pd
    >>> import numpy as np
    >>> from sklearn.datasets import load_iris
    >>> data = load_iris()
    >>> X_df = pd.DataFrame(data['data'], columns=data['feature_names'])
    >>> y_arr = data['target']
    >>> # Basic usage with multiple perturbation levels:
    >>> plot_perturbations(
    ...     X=X_df,
    ...     y=y_arr,
    ...     perturbations=[0.1, 0.2, 0.3],
    ...     kind='bar',
    ...     percent=True,
    ...     rotate_labels='both',
    ...     rotate=45
    ... )

    Notes
    -----
    - Each subplot corresponds to one magnitude of perturbation.
    - For a large number of perturbations, specify a higher
      `max_cols` or expect more rows in the subplot grid.

    See Also
    --------
    miv_score : The function used under the hood to compute MIV
        for each feature.

    References
    ----------
    .. [1] McKinney, W. "Python for Data Analysis: Data Wrangling
           with Pandas, NumPy, and IPython." O'Reilly, 2017.
    """
    from ..metrics_special import miv_score 
    
    # Convert the user-provided metric to lowercase and verify it is supported.
    # Currently, only 'miv' or 'm.i.v.' are supported. Otherwise, raise error.
    metric_str = str(metric).lower()
    if metric_str not in ('miv', 'm.i.v.'):
        warnings.warn(
            f"Only 'miv' or 'm.i.v.' is supported for `metric`. Got '{metric}'."
        )

    # Convert `perturbations` to an iterable if it isn't already,
    # so we can loop over multiple perturbation values. This allows
    # the user to compare MIV results for different magnitudes.
    pert_list = is_iterable(
        perturbations,
        exclude_string=True,
        transform=True
    )

    # For each perturbation value, we'll compute the MIV (or
    # other supported metrics in future). We'll store the results
    # in a list of dictionaries: each dict maps feature_name -> value.
    collected_results = []
    for idx, pert in enumerate(pert_list):
        # Call `miv_score` with `kind=None` to prevent
        # it from producing an immediate plot. We only want
        # its numerical results. We pass `relative` and
        # other relevant parameters as needed.
        msummary = miv_score(
            X=X,
            y=y,
            model=model,
            perturbation=pert,
            max_iter=max_iter,
            kind=None,       # block any plotting
            percent=False,        # keep raw numeric values, handle 'percent' here
            relative=relative,
            show_grid=False,      # not relevant here
            normalize=normalize,  
            fig_size=None,        # not relevant now
            cmap=cmap,            # not relevant now
            verbose=0             # silent
        )
        # Extract MIV values. This is a dict of feature_name -> MIV.
        # We'll store it along with the current `pert`, so we know
        # which dictionary belongs to which perturbation magnitude.
        miv_dict = msummary.feature_contributions_
        collected_results.append((pert, miv_dict))

    # We'll produce a multi-subplot figure to compare results
    # across different perturbation values. We figure out how
    # many subplots we need: each subplot is one perturbation's MIV result.
    n_pert = len(collected_results)
    nrows = math.ceil(n_pert / max_cols)
    ncols = min(n_pert, max_cols)

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=fig_size,
        squeeze=False     # always get a 2D array of axes
    )

    # We'll flatten the axes for easy iteration. If there's only
    # one subplot, it becomes axes[0,0]. We'll iterate safely.
    axes_flat = axes.flatten()

    # Titles: if user provided a list of custom titles, use them.
    # Otherwise, build a default string like "Perturbation=0.05"
    # or "Pert=0.05" if we want to keep it short.
    if titles is None:
        titles_list = [
            f"Perturbation={pert_list[i]}"
            for i in range(n_pert)
        ]
    else:
        # If user provided fewer titles than needed, we repeat or
        # fallback. If user provided more, we just slice.
        # For robust approach, let's just index carefully.
        titles_list = []
        for i in range(n_pert):
            if i < len(titles):
                titles_list.append(titles[i])
            else:
                titles_list.append(f"Perturbation={pert_list[i]}")

    # We'll define a small helper to do the bar or barh plotting,
    # similar to what's done in `miv_score`, with logic for rotation
    # of labels or text. We'll respect the `display_values` param
    # for text annotation on bars.
    def _plot_bars(
        ax,
        features,
        importances,
        plot_t,
        pcent,
        c_map,
        in_title
    ):
        # Bar or barh plotting
        if plot_t == 'bar':
            sns.barplot(
                ax=ax,
                x=list(features),
                y=list(importances),
                palette=c_map
            )
            ax.set_xlabel('Feature')
            ax.set_ylabel('MIV (%)' if pcent else 'MIV')
            # For text annotation, check `display_values`
            if display_values:
                for i, val in enumerate(importances):
                    ax.text(
                        i, val,
                        f'{val:.2f}{"%" if pcent else ""}',
                        va='bottom',
                        ha='center',
                        rotation=rotate if rotate_labels
                                  in ('value', 'both') and rotate else 0
                    )
            # Possibly rotate the x-tick labels
            if rotate_labels in ('feature', 'both') and rotate:
                ax.set_xticklabels(
                    ax.get_xticklabels(),
                    rotation=rotate,
                    ha='right'
                )
        else:
            # barh
            sns.barplot(
                ax=ax,
                x=list(importances),
                y=list(features),
                palette=c_map,
                orient='h'
            )
            ax.set_xlabel('MIV (%)' if pcent else 'MIV')
            ax.set_ylabel('Feature')
            # For text annotation, check `display_values`
            if display_values:
                for i, val in enumerate(importances):
                    ax.text(
                        val, i,
                        f'{val:.2f}{"%" if pcent else ""}',
                        va='center',
                        ha='left',
                        rotation=rotate if rotate_labels
                                  in ('value', 'both') and rotate else 0
                    )
            # Possibly rotate the y-tick labels
            if rotate_labels in ('feature', 'both') and rotate:
                ax.set_yticklabels(
                    ax.get_yticklabels(),
                    rotation=rotate,
                    va='center'
                )
        ax.set_title(in_title)

    # Now we iterate over each result, produce a subplot
    for i, (pert_val, miv_dict) in enumerate(collected_results):
        ax = axes_flat[i]
        # Sort feature -> MIV by MIV desc
        sorted_items = sorted(
            miv_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )
        features, importances = zip(*sorted_items)

        # If user says `percent=True`, we multiply
        # those importances by 100. We'll do that
        # locally so we don't mutate the original dict.
        final_imports = list(importances)
        if percent:
            final_imports = [val * 100 for val in final_imports]

        # If user selected 'pie' or 'scatter', do those. 
        # If 'bar' or 'barh', do as above in `_plot_bars`.
        if kind in ['bar', 'barh']:
            _plot_bars(
                ax=ax,
                features=features,
                importances=final_imports,
                plot_t=kind,
                pcent=percent,
                c_map=cmap,
                in_title=titles_list[i]
            )
            if show_grid and kind in ['bar', 'barh']:
                ax.grid(True, linestyle='--', alpha=0.7)
            elif not show_grid:
                ax.grid(False)

        elif kind == 'pie':
            # We'll do a pie chart
            patches, texts, autotexts = ax.pie(
                final_imports,
                labels=features if rotate_labels != 'none' else None,
                autopct=(lambda p: f'{p:.1f}%' if percent else f'{p:.3f}'),
                startangle=140,
                colors=sns.color_palette(cmap, len(final_imports))
            )
            ax.axis('equal')
            ax.set_title(titles_list[i])
            # Possibly rotate feature labels (the wedge labels).
            # For controlling label rotation on a pie, we might try:
            if rotate_labels in ('feature', 'both') and rotate:
                for text in texts:
                    text.set_rotation(rotate)
            # If user doesn't want display_values for the wedge
            # text, we can handle that by removing them. But we 
            # interpret 'display_values' as for bar chart text,
            # so we won't remove the wedge text automatically.

        elif kind == 'scatter':
            # We'll do a scatter. For the size param, we might scale the final_imports
            # so they're not too small or too big, but let's keep it simple
            # We'll do something like:
            # We need an x: final_imports, y: features. Because features is text,
            # let's do numeric for y. We'll do a local mapping.
            yvals = range(len(features))
            ax.scatter(
                final_imports,
                yvals,
                s=[val * 20 for val in final_imports],
                c=sns.color_palette(cmap, len(final_imports))
            )
            ax.set_yticks(yvals)
            # Possibly rotate the y tick labels
            if rotate_labels in ('feature', 'both') and rotate:
                ax.set_yticklabels(
                    ax.get_yticklabels(),
                    rotation=rotate
                )
            else:
                ax.set_yticklabels(features)

            ax.set_xlabel('MIV (%)' if percent else 'MIV')
            ax.set_ylabel('Feature')
            ax.set_title(titles_list[i])
            if display_values:
                for xv, yv, val in zip(final_imports, yvals, final_imports):
                    ax.text(
                        xv, yv,
                        f'{val:.2f}{"%" if percent else ""}',
                        va='center',
                        rotation=rotate if rotate_labels
                                  in ('value', 'both') and rotate else 0
                    )
            if show_grid:
                ax.grid(True, linestyle='--', alpha=0.7)
            else:
                ax.grid(False)
        else:
            # fallback if user provided an unsupported plot
            ax.barh(
                range(len(features)),
                final_imports,
                color=sns.color_palette(cmap, len(final_imports))
            )
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features)
            ax.set_xlabel('MIV (%)' if percent else 'MIV')
            ax.set_ylabel('Feature')
            ax.set_title(titles_list[i])
            if display_values:
                for j, val in enumerate(final_imports):
                    ax.text(
                        val, j,
                        f'{val:.2f}{"%" if percent else ""}',
                        va='center',
                        rotation=rotate if rotate_labels
                                  in ('value', 'both') and rotate else 0
                    )
            if show_grid:
                ax.grid(True, linestyle='--', alpha=0.7)
            else:
                ax.grid(False)

    # Hide any extra subplot if n_pert < nrows * ncols
    total_subplots = nrows * ncols
    if total_subplots > n_pert:
        for hide_idx in range(n_pert, total_subplots):
            axes_flat[hide_idx].set_visible(False)

    plt.tight_layout()

    # if user pass savefig, then save
    if savefig:
        plt.savefig(savefig)

    plt.show()


@return_fig_or_ax(return_type ='ax')
@validate_params ({ 
    "sensitivity_values": ['array-like'], 
    "baseline_prediction": ['array-like', Real, None ], 
    "kind": [StrOptions({'hist', 'bar', 'line', 'boxplot', 'box'})], 
    "x_ticks_rotation": [Interval( Integral, 0, None, closed="left")], 
    "y_ticks_rotation": [Interval( Integral, 0, None, closed="left")], 
    })
def plot_sensitivity(
    sensitivity_df, *,
    baseline=None, 
    kind='line',
    baseline_color='r',
    baseline_linewidth=2,
    baseline_linestyle='--',
    title=None,
    xlabel=None,
    ylabel=None,
    x_ticks_rotation=0,
    y_ticks_rotation=0,
    show_grid=True,
    legend=True,
    figsize=(10, 6),
    color_palette='muted',
    boxplot_showfliers=False
):
    r"""
    Plot the feature sensitivity values.

    Parameters
    ----------
    sensitivity_df : pandas.DataFrame
        A DataFrame containing sensitivity values for each feature. Each column 
        represents the sensitivity values for a specific feature. The index 
        represents individual observations or instances.
        
    baseline: array-like or scalar, optional 
        The baseline prediction, either a scalar value or an array-like object 
        (e.g., list, numpy array) representing the baseline prediction to be 
        compared with feature sensitivities.

    kind : {'line', 'bar', 'hist', 'boxplot'}, optional, default='line'
        The type of plot to generate. Options include:
        - 'line': Line plot to visualize feature sensitivity trends.
        - 'bar': Bar plot for visualizing feature sensitivity comparisons.
        - 'hist': Histogram to show the distribution of sensitivities for 
          each feature.
        - 'boxplot': Boxplot to summarize the distribution and outliers 
          of sensitivities.

    baseline_color : str, optional, default='r'
        The color for the baseline prediction line. Can be any valid Matplotlib
         color specification (e.g., named color, hex, RGB tuple).

    baseline_linewidth : float, optional, default=2
        The line width for the baseline prediction line.

    baseline_linestyle : {'-', '--', '-.', ':'}, optional, default='--'
        The line style for the baseline prediction line. 
        Options include solid, dashed, dash-dot, and dotted lines.

    title : str, optional, default=None
        The title for the plot. If not provided, a default title is generated 
        based on the plot type.

    xlabel : str, optional, default='Features'
        The label for the x-axis, which typically corresponds to the feature 
        names or identifiers in `sensitivity_df`.

    ylabel : str, optional, default='Sensitivity Value'
        The label for the y-axis, representing the sensitivity value or 
        measure associated with each feature.

    x_ticks_rotation : int, optional, default=0
        The angle in degrees to rotate the x-axis tick labels. Helps in cases 
        where feature names or labels overlap.

    y_ticks_rotation : int, optional, default=0
        The angle in degrees to rotate the y-axis tick labels.

    show_grid : bool, optional, default=True
        Whether to show gridlines on the plot. True will enable gridlines, 
        False will disable them.

    legend : bool, optional, default=True
        Whether to display the legend in the plot. Set to True to show the 
        legend, False to hide it.

    figsize : tuple of two floats, optional, default=(10, 6)
        The dimensions of the plot as a tuple representing (width, height) 
        in inches.

    color_palette : str, optional, default='muted'
        The seaborn color palette to use for the plot. A string specifying 
        a predefined color palette (e.g., 'deep', 'muted', 'bright').

    boxplot_showfliers : bool, optional, default=False
        Whether to display outliers in the boxplot when `kind='boxplot'`. 
        Set to False to hide outliers, True to show them.

    Returns
    -------
    None
        The function generates and displays a plot showing the baseline 
        prediction and feature sensitivity with the specified 
        customization options.


    Notes
    -----
    The function will automatically determine whether to construct the 
    `sensitivity_df` DataFrame if not provided directly as a DataFrame.
    It uses the `build_data_if` helper function to convert the data into a 
    DataFrame before proceeding with plotting.

    Examples
    --------
    >>> from gofast.plot.suite import plot_sensitivity
    1. Basic line plot:
       >>> plot_sensitivity(baseline=0.5, 
                            sensitivity_df=pd.DataFrame({
                                'Feature 1': [0.1, 0.2, 0.3],
                                'Feature 2': [0.05, 0.15, 0.25],
                                'Feature 3': [0.2, 0.3, 0.4]
                            }), 
                            kind='line')
       
    2. Bar plot with customized appearance:
       >>> plot_sensitivity(baseline=0.5, 
                            sensitivity_df=pd.DataFrame({
                                'Feature 1': [0.1, 0.2, 0.3],
                                'Feature 2': [0.05, 0.15, 0.25],
                                'Feature 3': [0.2, 0.3, 0.4]
                            }), 
                            kind='bar', baseline_color='g', 
                            baseline_linestyle='-', figsize=(8, 5))
       
    3. Histogram plot:
       >>> plot_sensitivity(baseline=0.5, 
                            sensitivity_df=pd.DataFrame({
                                'Feature 1': [0.1, 0.2, 0.3],
                                'Feature 2': [0.05, 0.15, 0.25],
                                'Feature 3': [0.2, 0.3, 0.4]
                            }), 
                            kind='hist')
    
    4. Boxplot with outliers:
       >>> plot_sensitivity(baseline=0.5, 
                            sensitivity_df=pd.DataFrame({
                                'Feature 1': [0.1, 0.2, 0.3],
                                'Feature 2': [0.05, 0.15, 0.25],
                                'Feature 3': [0.2, 0.3, 0.4]
                            }), 
                            kind='boxplot', boxplot_showfliers=True)
    """
    sensitivity_values = copy.deepcopy(sensitivity_df)
    
    if not isinstance (sensitivity_values, pd.DataFrame): 
        # build dataframe using the default column name 'feature' 
        sensitivity_values = build_data_if(
            sensitivity_values, 
            force=True, 
            input_name="feature", 
            raise_exception=True 
    ) 

    if len(sensitivity_values) == 1:
        # Transpose if single perturbation
        sensitivity_values = sensitivity_values.T  
    
    sns.set(style="whitegrid", palette=color_palette)
    
    # Default plot title
    if title is None:
        title = 'Feature Sensitivity vs Baseline Prediction'

    plt.figure(figsize=figsize)

    if kind == 'line':
        for col in sensitivity_values.columns:
            plt.plot(
                sensitivity_values.index, 
                sensitivity_values[col], 
                label=col, 
                marker='o'
            )
        if baseline is not None:
            if isinstance(baseline, (list, np.ndarray)):
                baseline = baseline[0]  # Use the first element if it's an array-like
            plt.axhline(
                y=baseline, 
                color=baseline_color, 
                linestyle=baseline_linestyle, 
                linewidth=baseline_linewidth, 
                label='Baseline Prediction'
            )
        plt.title(title) 
        plt.xlabel(xlabel or "Pertubations")
        plt.ylabel(ylabel or 'Sensitivity Value')
        plt.xticks(rotation=x_ticks_rotation)
        plt.yticks(rotation=y_ticks_rotation)
        # if grid:
        #     plt.grid(True)
        if legend:
            plt.legend()

    elif kind == 'bar':
        sensitivity_values_mean = sensitivity_values.mean(axis=0)
        plt.bar(
            sensitivity_values_mean.index, 
            sensitivity_values_mean.values, 
            label='Feature Sensitivity'
        )
        if baseline is not None:
            if isinstance(baseline, (list, np.ndarray)):
                baseline = baseline[0]  # Use the first element if it's an array-like
            plt.axhline(
                y=baseline, 
                color=baseline_color, 
                linestyle=baseline_linestyle, 
                linewidth=baseline_linewidth, 
                label='Baseline Prediction'
            )
        plt.title(title)
        plt.xlabel(xlabel or "Pertubations")
        plt.ylabel(ylabel or 'Sensitivity Value')
        plt.xticks(rotation=x_ticks_rotation)
        plt.yticks(rotation=y_ticks_rotation)
        # if grid:
        #     plt.grid(True)
        if legend:
            plt.legend()

    elif kind == 'hist':
        for col in sensitivity_values.columns:
            sns.histplot(
                sensitivity_values[col], 
                kde=True, 
                label=col, 
                element='step', 
                fill=False
            )
        if baseline is not None:
            if isinstance(baseline, (list, np.ndarray)):
                baseline = baseline[0]  # Use the first element if it's an array-like
            plt.axvline(
                x=baseline, 
                color=baseline_color, 
                linestyle=baseline_linestyle, 
                linewidth=baseline_linewidth, 
                label='Baseline Prediction'
            )
        plt.title(title)
        plt.xlabel(xlabel or 'Sensitivity Value')
        plt.ylabel('Frequency')
        # if grid:
        #     plt.grid(True)
        if legend:
            plt.legend()

    elif kind in ['boxplot', 'box']:
        sns.boxplot(
            data=sensitivity_values, 
            showfliers=boxplot_showfliers
        )
        if baseline is not None:
            if isinstance(baseline, (list, np.ndarray)):
                baseline = baseline[0]  # Use the first element if it's an array-like
            plt.axhline(
                y=baseline, 
                color=baseline_color, 
                linestyle=baseline_linestyle, 
                linewidth=baseline_linewidth, 
                label='Baseline Prediction'
            )
        plt.title(title)
        plt.xlabel(xlabel or "Pertubations")
        plt.ylabel(ylabel or 'Sensitivity Value')
    

    plt.grid(show_grid) 
    if legend:
        plt.legend()
    plt.xticks(rotation=x_ticks_rotation)
    plt.yticks(rotation=y_ticks_rotation)
    plt.tight_layout()
    plt.show()



@return_fig_or_ax (return_type ='ax')
@default_params_plot(
    savefig=PlotConfig.AUTOSAVE('my_fit_plot.png'),
    fig_size=(8, 6)
  )
@validate_params({
    "y_true": ['array-like'], 
    "y_pred": ['array-like'], 
    "kind": [StrOptions({ 
        'scatter', 'residual', 'residual_hist', 'density',
        'hexbin', 'cumulative_gain', 'lift_curve', 'precision_recall',
        'pred_histogram', 'qq_plot', 'error_heatmap', 'actual_vs_error', 
        })]
    })
def plot_fit(
    y_true, 
    y_pred, 
    *, 
    kind='scatter',               
    show_perfect_fit=...,         
    fit_line_color='red',          
    color: str='blue', 
    pred_color: str='orange', 
    add_reg_line=False,             
    reg_color='green',             
    annot_metrics=True,            
    metrics_position='auto', 
    plot_dist=False,               
    alpha=0.3,                     
    title=None,                    
    xlabel='True Values',          
    ylabel='Predicted Values',     
    figsize=None,                
    scatter_alpha=0.6,             
    bins=30,                       
    cmap='viridis',                
    hexbin_gridsize=50,            
    hexbin_cmap='Blues',           
    ci=95,                         
    residual_trendline=...,       
    trendline_color='orange',      
    show_grid=...,
    grid_props=None,
    savefig=None,                  
    **kwargs                       
) -> plt.Figure:
    """
    Plot various fit analyses between true and predicted values.

    The `plot_fit` function offers a range of visualizations to
    analyze the relationship between `y_true` and `y_pred`, including
    scatter plots, residual plots, density plots, precision-recall
    curves, and more. Additionally, it can annotate plots with
    performance metrics like R², RMSE, and MAE.

    Parameters
    ----------
    y_true : array-like
        The true values. Must be 1-dimensional and numeric.

    y_pred : array-like
        The predicted values. Must match the shape of `y_true`.

    kind : str, default='scatter'
        The type of plot to produce. Options include:
        - ``'scatter'``: Scatter plot of true vs. predicted values.
        - ``'residual'``: Residual plot.
        - ``'residual_hist'``: Histogram of residuals.
        - ``'density'``: Density plot of true vs. predicted values.
        - ``'hexbin'``: Hexbin plot of true vs. predicted values.
        - ``'cumulative_gain'``: Cumulative gain curve.
        - ``'lift_curve'``: Lift curve.
        - ``'precision_recall'``: Precision-recall curve.
        - ``'pred_histogram'``: Histogram of predicted values.
        - ``'qq_plot'``: QQ plot for residual normality.
        - ``'error_heatmap'``: Heatmap of prediction errors.
        - ``'actual_vs_error'``: Scatter plot of actual values vs. errors.

    show_perfect_fit : bool, default=True
        Whether to include a ``y = x`` perfect fit line in relevant
        plots.

    fit_line_color : str, default='red'
        Color for the perfect fit line.

    color : str, default='blue'
        Color for the main data points in the plots.

    pred_color : str, default='orange'
        Color for the predicted value distribution overlays.

    add_reg_line : bool, default=False
        Whether to add a regression line to scatter plots.

    reg_color : str, default='green'
        Color for the regression line.

    annot_metrics : bool, default=True
        Whether to annotate the plot with performance metrics
        like R², RMSE, and MAE.

    metrics_position : tuple or 'auto', default='auto'
        The position of metrics annotation. If set to ``'auto'``,
        defaults to ``(0.85, 0.05)`` in axes fraction coordinates.

    plot_dist : bool, default=False
        Whether to overlay histograms or density plots of
        `y_true` and `y_pred`.

    alpha : float, default=0.3
        Transparency for distribution overlays.

    title : str, optional
        Custom title for the plot. Defaults to a title based on
        the `kind` parameter.

    xlabel : str, default='True Values'
        Label for the x-axis.

    ylabel : str, default='Predicted Values'
        Label for the y-axis.

    figsize : tuple of float, default=(8, 6)
        Figure size in inches.

    scatter_alpha : float, default=0.6
        Transparency for scatter points.

    bins : int, default=30
        Number of bins for histogram-based plots.

    cmap : str, default='viridis'
        Colormap for density plots.

    hexbin_gridsize : int, default=50
        Grid size for hexbin plots.

    hexbin_cmap : str, default='Blues'
        Colormap for hexbin plots.

    ci : int, default=95
        Confidence interval for regression lines.

    residual_trendline : bool, default=False
        Whether to add a trendline to residual plots.

    trendline_color : str, default='orange'
        Color for residual trendlines.

    show_grid : bool, default=False
        Whether to display a grid on the plot.

    savefig : str or None, optional
        Path to save the plot as an image. If ``None``, the plot
        is not saved.

    **kwargs : dict
        Additional arguments passed to plotting functions.

    Returns
    -------
    plt.Figure
        The Matplotlib figure object for the plot.

    Notes
    -----
    This function supports several types of plots for analyzing
    regression or classification model performance. The `kind`
    parameter controls the type of visualization produced.

    .. math::
        R^2 = 1 - \\frac{\\sum{(y_{true} - y_{pred})^2}}
                         {\\sum{(y_{true} - \\bar{y}_{true})^2}}

    Root Mean Squared Error (RMSE):

    .. math::
        RMSE = \\sqrt{\\frac{1}{n} \\sum{(y_{true} - y_{pred})^2}}

    Mean Absolute Error (MAE):

    .. math::
        MAE = \\frac{1}{n} \\sum{|y_{true} - y_{pred}|}

    Examples
    --------
    >>> from gofast.plot.suite import plot_fit
    >>> import numpy as np

    Create sample data:
    >>> y_true = np.random.rand(100)
    >>> y_pred = y_true + np.random.normal(scale=0.1, size=100)

    Scatter plot:
    >>> plot_fit(y_true, y_pred, kind='scatter')

    Residual plot:
    >>> plot_fit(y_true, y_pred, kind='residual')

    Density plot:
    >>> plot_fit(y_true, y_pred, kind='density', cmap='plasma')

    See Also
    --------
    sklearn.metrics.r2_score : R² score calculation.
    sklearn.metrics.mean_squared_error : RMSE calculation.
    sklearn.metrics.mean_absolute_error : MAE calculation.

    References
    ----------
    .. [1] Seaborn Documentation: https://seaborn.pydata.org
    .. [2] Matplotlib Documentation: https://matplotlib.org
    .. [3] Regression Analysis: https://en.wikipedia.org/wiki/Regression_analysis
    """

    # Remove NaN values from y_true and y_pred arrays
    y_true, y_pred = drop_nan_in(y_true, y_pred, error='raise')
    
    # Validate y_true and y_pred to ensure consistency and continuity
    y_true, y_pred = validate_yy(
        y_true, y_pred, expected_type="continuous",
        flatten=True
    )

    if metrics_position =='auto': 
        metrics_position =(0.85, 0.05) 
    # Create a figure and axis with specified figure size
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot analysis
    if kind == 'scatter':
        # Plot actual vs predicted values as scatter points
        ax.scatter(
            y_true, y_pred, 
            alpha=scatter_alpha,
            color=color, 
            **kwargs
        )
        
        # Optionally plot the perfect fit line (y = x)
        if show_perfect_fit:
            ax.plot(
                [min(y_true), max(y_true)], 
                [min(y_true), max(y_true)], 
                color=fit_line_color, 
                linestyle='--', 
                label="Perfect Fit"
            )
        
        # Optionally add a regression line to the scatter plot
        if add_reg_line:
            coeffs = np.polyfit(y_true, y_pred, 1)  # Fit a linear regression
            ax.plot(
                y_true, 
                np.polyval(coeffs, y_true), 
                color=reg_color, 
                linestyle='-', 
                label="Regression Line"
            )
        
        # Optionally plot distributions of true and predicted values
        if plot_dist:
            sns.histplot(
                y_true, 
                kde=True, 
                alpha=alpha, 
                color=color, 
                ax=ax, 
                label="True Dist"
            )
            sns.histplot(
                y_pred, 
                kde=True, 
                alpha=alpha, 
                color=pred_color, 
                ax=ax, 
                label="Pred Dist"
            )
    
    # Residual analysis
    elif kind == 'residual':
        residuals = y_true - y_pred  # Calculate residuals
        
        # Plot residuals against predicted values
        ax.scatter(
            y_pred, residuals, 
            alpha=scatter_alpha, 
            color= color, 
            **kwargs
        )
        
        # Plot a horizontal line at zero residual
        ax.axhline(
            0, 
            color=fit_line_color, 
            linestyle='--', 
            label="Zero Error"
        )
        
        # Optionally add a trendline to the residual plot
        if residual_trendline:
            sns.regplot(
                x=y_pred, 
                y=residuals, 
                lowess=True, 
                ax=ax, 
                scatter=False, 
                line_kws={"color": trendline_color, "label": "Residual Trend"}
            )
    
    # Residual histogram analysis
    elif kind == 'residual_hist':
        residuals = y_true - y_pred  # Calculate residuals
        
        # Plot histogram of residuals
        ax.hist(
            residuals, 
            bins=bins, 
            alpha=scatter_alpha, 
            color=color, 
            **kwargs
        )
        ax.set_ylabel('Frequency')  # Set y-axis label
        ax.set_xlabel('Residuals')  # Set x-axis label
    
    # Density plot analysis
    elif kind == 'density':
        # Plot density contours of true vs predicted values
        sns.kdeplot(
            x=y_true, 
            y=y_pred, 
            cmap=cmap, 
            fill=True, 
            ax=ax, 
            **kwargs
        )
        
        # Optionally plot the perfect fit line
        if show_perfect_fit:
            ax.plot(
                [min(y_true), max(y_true)], 
                [min(y_true), max(y_true)], 
                color=fit_line_color, 
                linestyle='--', 
                label="Perfect Fit"
            )
    
    # Hexbin plot analysis
    elif kind == 'hexbin':
        # Create a hexbin plot to show density
        hb = ax.hexbin(
            y_true, y_pred, 
            gridsize=hexbin_gridsize, 
            cmap=hexbin_cmap, 
            **kwargs
        )
        # Add a color bar to indicate frequency
        cbar = fig.colorbar(hb, ax=ax)
        cbar.set_label('Frequency')
    
    # Cumulative Gain Curve analysis
    elif kind == 'cumulative_gain':
        sorted_indices = np.argsort(y_pred)[::-1]  # Sort predictions descending
        y_true_sorted = np.array(y_true)[sorted_indices]  # Sort true values accordingly
        gains = np.cumsum(y_true_sorted) / np.sum(y_true)  # Cumulative gain
        gains = np.insert(gains, 0, 0)  # Add zero point for the curve start
        
        # Plot the cumulative gain curve
        ax.plot(
            np.linspace(0, 1, len(gains)), 
            gains, 
            color = color, 
            label="Cumulative Gain"
        )
        
        # Plot a baseline for random guessing
        ax.plot(
            [0, 1], 
            [0, 1], 
            linestyle='--', 
            color=fit_line_color, 
            label="Random Guess"
        )
        ax.set_xlabel(xlabel or "Fraction of Samples")  # Set x-axis label
        ax.set_ylabel(ylabel or "Cumulative Gain")      # Set y-axis label
    
    # Lift Curve analysis
    elif kind == 'lift_curve':
        sorted_indices = np.argsort(y_pred)[::-1]  # Sort predictions descending
        y_true_sorted = np.array(y_true)[sorted_indices]  # Sort true values accordingly
        cumulative_pos = np.cumsum(y_true_sorted)  # Cumulative positives
        total_pos = np.sum(y_true_sorted)          # Total positives
        lift = (
            (cumulative_pos / np.arange(1, len(y_true) + 1)) / 
            (total_pos / len(y_true))
        )  # Calculate lift
        
        # Plot the lift curve
        ax.plot(lift, label="Lift Curve", color = color )
        
        # Plot a baseline for random guessing
        ax.axhline(
            1, 
            color=fit_line_color, 
            linestyle='--', 
            label="Random Guess"
        )
        ax.set_xlabel(xlabel or "Number of Samples")  # Set x-axis label
        ax.set_ylabel(ylabel or "Lift")               # Set y-axis label
    
    # Precision-Recall Curve analysis
    elif kind == 'precision_recall':
        precision, recall, _ = precision_recall_curve(y_true, y_pred)  # Compute precision-recall
        ax.plot(
            recall, 
            precision, 
            label="Precision-Recall Curve", 
            color=color, 
        )
        ax.set_xlabel(xlabel or "Recall")      # Set x-axis label
        ax.set_ylabel(ylabel or "Precision")  # Set y-axis label
    
    # Histogram of Predictions analysis
    elif kind == 'pred_histogram':
        # Plot histogram of predicted values
        ax.hist(
            y_pred, 
            bins=bins, 
            alpha=scatter_alpha, 
            color=color, 
            **kwargs
        )
        ax.set_xlabel(xlabel or "Predicted Values")  # Set x-axis label
        ax.set_ylabel(ylabel or "Frequency")         # Set y-axis label
    
    # QQ Plot analysis
    elif kind == 'qq_plot':
        residuals = y_true - y_pred  # Calculate residuals
        
        # Generate a QQ plot to assess normality of residuals
        probplot(
            residuals, 
            dist="norm", 
            plot=ax
        )
        ax.set_xlabel(xlabel or "Theoretical Quantiles")  # Set x-axis label
        ax.set_ylabel(ylabel or "Sample Quantiles")      # Set y-axis label
    
    # Error Heatmap analysis
    elif kind == 'error_heatmap':
        errors = np.abs(y_true - y_pred)  # Calculate absolute errors
        
        # Plot histogram of errors with density
        sns.histplot(
            errors, 
            bins=bins, 
            kde=True, 
            cmap=cmap, 
            ax=ax, 
            **kwargs
        )
        ax.set_xlabel(xlabel or "Errors")      # Set x-axis label
        ax.set_ylabel(ylabel or "Frequency")   # Set y-axis label
    
    # Actual vs. Error Scatter analysis
    elif kind == 'actual_vs_error':
        errors = y_true - y_pred  # Calculate residuals
        
        # Plot residuals against actual values
        ax.scatter(
            y_true, errors, 
            alpha=scatter_alpha, 
            color=color, 
            **kwargs
        )
        
        # Plot a horizontal line at zero residual
        ax.axhline(
            0, 
            color=fit_line_color, 
            linestyle='--', 
            label="Zero Error"
        )
        ax.set_xlabel(xlabel or "True Values")  # Set x-axis label
        ax.set_ylabel(ylabel or "Errors")       # Set y-axis label
    
    # Metrics calculation and annotation
    if annot_metrics and kind not in [
        'cumulative_gain', 
        'lift_curve', 
        'precision_recall'
    ]:
        r2 = r2_score(y_true, y_pred)  # Calculate R-squared
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # Calculate RMSE
        mae = np.mean(np.abs(y_true - y_pred))  # Calculate MAE
        
        # Prepare metrics text
        metrics_text = f"$R^2$={r2:.2f}\nRMSE={rmse:.2f}\nMAE={mae:.2f}"
        
        # Annotate metrics on the plot at the specified position
        ax.annotate(
            metrics_text, 
            xy=metrics_position, 
            xycoords='axes fraction',
            fontsize=10, 
            color='black', 
            bbox=dict(
                boxstyle="round,pad=0.3", 
                edgecolor="black", 
                facecolor="white"
            )
        )
        
        # Set plot title, axis labels, and legend
        ax.set_title(
            title or f"{kind.replace('_', ' ').capitalize()} Analysis"
        )
        ax.set_xlabel(xlabel or "X-axis")  # Default to 'X-axis' if not provided
        ax.set_ylabel(ylabel or "Y-axis")  # Default to 'Y-axis' if not provided
        ax.legend(loc='best', frameon=True)  # Add legend in the best location
        plt.tight_layout()  # Adjust layout to prevent overlap
        
        # Display the plot
        plt.show()
    
    else:
        # Calculate metrics if annotation is not requested for specific analysis types
        r2 = r2_score(y_true, y_pred)  # Calculate R-squared
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # Calculate RMSE
        mae = np.mean(np.abs(y_true - y_pred))  # Calculate MAE
        
        # Prepare metrics text
        metrics_text = f"$R^2$={r2:.2f}\nRMSE={rmse:.2f}\nMAE={mae:.2f}"
        
        # Annotate metrics on the plot at the specified position
        ax.annotate(
            metrics_text, 
            xy=metrics_position, 
            xycoords='axes fraction',
            fontsize=10, 
            color='black', 
            bbox=dict(
                boxstyle="round,pad=0.3", 
                edgecolor="black", 
                facecolor="white"
            )
        )
        
        # Set plot title, axis labels, and legend
        ax.set_title(
            title or f"{kind.capitalize()} Analysis"
        )
        ax.set_xlabel(xlabel)  # Set x-axis label
        ax.set_ylabel(ylabel)  # Set y-axis label
        ax.legend(loc='best', frameon=True)  # Add legend in the best location
        plt.tight_layout()  # Adjust layout to prevent overlap
    
    if show_grid: 
        ax.grid(True, **(grid_props or {'linestyle': ':', 'alpha': 0.7}))
    else: 
        ax.grid(False) 
        
        
    # Save the plot to the specified path if provided
    if savefig:
        plt.savefig(
            savefig, 
            dpi=300, 
            bbox_inches='tight'
        )
    
    # Display the plot
    plt.show()

@return_fig_or_ax(return_type ='ax')
@default_params_plot(
    savefig=PlotConfig.AUTOSAVE('my_relationship_plot.png'),
    fig_size=(8, 8)
  )
@validate_params({
    "y_true": ['array-like'], 
    "y_pred": ['array-like'], 
    "theta_scale": [StrOptions({ 'proportional', 'uniform' })], 
    'acov': [StrOptions({
        'default', 'half_circle', 'quarter_circle', 'eighth_circle'})], 
    })
def plot_relationship(
    y_true, *y_preds, 
    names=None, 
    title=None,  
    theta_offset=0,  
    theta_scale='proportional',  
    acov='default',  
    figsize=None, 
    cmap='viridis',  
    s=50,  
    alpha=0.7,  
    legend=True,  
    show_grid=True,  
    color_palette=None,  
    xlabel=None,  
    ylabel=None,  
    z_values=None,  
    z_label=None  
):
    """
    Visualize the relationship between `y_true` and multiple `y_preds`
    using a circular or polar plot. The function allows flexible
    configurations such as angular coverage, z-values for replacing
    angle labels, and customizable axis labels.

    Parameters
    ----------
    y_true : array-like
        The true values. Must be numeric, one-dimensional, and of the
        same length as the values in `y_preds`.

    y_preds : array-like (one or more)
        Predicted values from one or more models. Each `y_pred` must
        have the same length as `y_true`.

    names : list of str, optional
        A list of model names corresponding to each `y_pred`. If not
        provided or if fewer names than predictions are given, the
        function assigns default names as ``"Model_1"``, ``"Model_2"``,
        etc. For instance, if `y_preds` has three predictions and
        `names` is `["SVC", "RF"]`, the names will be updated to
        `["SVC", "RF", "Model_3"]`.

    title : str, optional
        The title of the plot. If `None`, the title defaults to
        `"Relationship Visualization"`.

    theta_offset : float, default=0
        Angular offset in radians to rotate the plot. This allows
        customization of the orientation of the plot.

    theta_scale : {'proportional', 'uniform'}, default='proportional'
        Determines how `y_true` values are mapped to angular
        coordinates (`theta`):
        - ``'proportional'``: Maps `y_true` proportionally to the
          angular range (e.g., 0 to 360° or a subset defined by
          `acov`).
        - ``'uniform'``: Distributes `y_true` values uniformly around
          the angular range.

    acov : {'default', 'half_circle', 'quarter_circle', 'eighth_circle'}, 
           default='default'
        Specifies the angular coverage of the plot:
        - ``'default'``: Full circle (360°).
        - ``'half_circle'``: Half circle (180°).
        - ``'quarter_circle'``: Quarter circle (90°).
        - ``'eighth_circle'``: Eighth circle (45°).
        The angular span is automatically restricted to the selected
        portion of the circle.

    figsize : tuple of float, default=(8, 8)
        The dimensions of the figure in inches.

    cmap : str, default='viridis'
        Colormap for the scatter points. Refer to Matplotlib
        documentation for a list of supported colormaps.

    s : float, default=50
        Size of scatter points representing predictions.

    alpha : float, default=0.7
        Transparency level for scatter points. Valid values range
        from 0 (completely transparent) to 1 (fully opaque).

    legend : bool, default=True
        Whether to display a legend indicating the model names.

    show_grid : bool, default=True
        Whether to display a grid on the polar plot.

    color_palette : list of str, optional
        A list of colors to use for the scatter points. If not
        provided, the default Matplotlib color palette (`tab10`) is
        used.

    xlabel : str, optional
        Label for the radial axis (distance from the center). Defaults
        to `"Normalized Predictions (r)"`.

    ylabel : str, optional
        Label for the angular axis (theta values). Defaults to
        `"Angular Mapping (θ)"`.

    z_values : array-like, optional
        Optional values to replace the angular labels. The length of
        `z_values` must match the length of `y_true`. If provided, the
        angular labels are replaced by the scaled `z_values`.

    z_label : str, optional
        Label for the `z_values`, if provided. Defaults to `None`.

    Returns
    -------
    None
        Displays the polar plot. Does not return any value.

    Notes
    -----
    The function dynamically maps `y_true` to angular coordinates
    based on the `theta_scale` and `acov` parameters. The `y_preds`
    are normalized to radial coordinates between 0 and 1. Optionally,
    `z_values` can replace angular labels with custom values.

    .. math::
        \theta = 
        \begin{cases} 
        \text{Proportional mapping: } \theta_i = 
        \frac{y_{\text{true}_i} - \min(y_{\text{true}})}
        {\max(y_{\text{true}}) - \min(y_{\text{true}})} 
        \cdot \text{angular_range} \\
        \text{Uniform mapping: } \theta_i = 
        \frac{i}{N-1} \cdot \text{angular_range}
        \end{cases}

    Radial normalization:

    .. math::
        r_i = \frac{y_{\text{pred}_i} - \min(y_{\text{pred}})}
        {\max(y_{\text{pred}}) - \min(y_{\text{pred}})}

    Examples
    --------
    >>> from gofast.plot.suite import plot_relationship
    >>> import numpy as np

    # Create sample data
    >>> y_true = np.random.rand(100)
    >>> y_pred1 = y_true + np.random.normal(0, 0.1, size=100)
    >>> y_pred2 = y_true + np.random.normal(0, 0.2, size=100)

    # Full circle visualization
    >>> plot_relationship(
    ...     y_true, y_pred1, y_pred2,
    ...     names=["Model A", "Model B"],
    ...     acov="default",
    ...     title="Full Circle Visualization"
    ... )

    # Half-circle visualization with z-values
    >>> z_values = np.linspace(0, 100, len(y_true))
    >>> plot_relationship(
    ...     y_true, y_pred1, 
    ...     names=["Model A"],
    ...     acov="half_circle",
    ...     z_values=z_values,
    ...     xlabel="Predicted Values",
    ...     ylabel="Custom Angles"
    ... )

    See Also
    --------
    matplotlib.pyplot.polar : Polar plotting in Matplotlib.
    numpy.linspace : Uniformly spaced numbers.

    References
    ----------
    .. [1] Hunter, J. D. (2007). Matplotlib: A 2D graphics environment.
           Computing in Science & Engineering, 9(3), 90-95.
    .. [2] NumPy Documentation: https://numpy.org/doc/stable/
    .. [3] Matplotlib Documentation: https://matplotlib.org/stable/
    """

    # Remove NaN values from y_true and all y_pred arrays
    y_true, *y_preds = drop_nan_in(y_true, *y_preds, error='raise')

    # Validate y_true and each y_pred to ensure consistency and continuity
    y_preds = [
        validate_yy(y_true, pred, expected_type="continuous", flatten=True)[1] 
        for pred in y_preds
    ]

    # Generate default model names if none are provided
    if names is None:
        names = [f"Model_{i+1}" for i in range(len(y_preds))]
    else:
        # Ensure the length of names matches y_preds
        if len(names) < len(y_preds):
            names += [f"Model_{i+1}" for i in range(len(names), len(y_preds))]

    # Create default color palette if none is provided
    if color_palette is None:
        color_palette = plt.cm.tab10.colors

    # Determine the angular range based on `acov`
    if acov == 'default':  # Full circle (360 degrees)
        angular_range = 2 * np.pi
    elif acov == 'half_circle':  # Half-circle (180 degrees)
        angular_range = np.pi
    elif acov == 'quarter_circle':  # Quarter-circle (90 degrees)
        angular_range = np.pi / 2
    elif acov == 'eighth_circle':  # Eighth-circle (45 degrees)
        angular_range = np.pi / 4
    else:
        raise ValueError(
            "Invalid value for `acov`. Choose from 'default',"
            " 'half_circle', 'quarter_circle', or 'eighth_circle'.")

    # Create the polar plot
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'polar'})

    # Limit the visible angular range
    ax.set_thetamin(0)  # Start angle (in degrees)
    ax.set_thetamax(np.degrees(angular_range))  # End angle (in degrees)

    # Map `y_true` to angular coordinates (theta)
    if theta_scale == 'proportional':
        theta = angular_range * (
            y_true - np.min(y_true)) / (np.max(y_true) - np.min(y_true))
    elif theta_scale == 'uniform':
        theta = np.linspace(0, angular_range, len(y_true))
    else:
        raise ValueError(
            "`theta_scale` must be either 'proportional' or 'uniform'.")

    # Apply theta offset
    theta += theta_offset

    # Plot each model's predictions
    for i, y_pred in enumerate(y_preds):
        # Ensure `y_pred` is a numpy array
        y_pred = np.array(y_pred)

        # Normalize `y_pred` for radial coordinates
        r = (y_pred - np.min(y_pred)) / (np.max(y_pred) - np.min(y_pred))

        # Plot on the polar axis
        ax.scatter(
            theta, r, 
            label=names[i], 
            c=color_palette[i % len(color_palette)], 
            s=s, alpha=alpha, edgecolor='black'
        )

    # If z_values are provided, replace angle labels with z_values
    if z_values is not None:
        # Validate z_values length
        if len(z_values) != len(y_true):
            raise ValueError("Length of `z_values` must match the length of `y_true`.")

        # Map z_values to the same angular range
        z_mapped = angular_range * (
            z_values - np.min(z_values)) / (np.max(z_values) - np.min(z_values))

        # Set custom z_labels
        ax.set_xticks(np.linspace(0, angular_range, len(z_mapped), endpoint=False))
        ax.set_xticklabels([f"{z:.2f}" for z in z_values])

    # Add labels for radial and angular axes
    ax.set_xlabel(xlabel or "Normalized Predictions (r)")
    ax.set_ylabel(ylabel or "Angular Mapping (θ)")

    # Add title
    ax.set_title(title or "Relationship Visualization", va='bottom')

    # Add grid
    ax.grid(show_grid)

    # Add legend
    if legend:
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

    # Show the plot
    plt.show()

@return_fig_or_ax(return_type ='ax')
@validate_params ({ 
    "sensitivity_values": ['array-like'], 
    "features": [str, 'array-like', None ], 
    "kind": [StrOptions({'single', 'pair', 'triple'})], 
    })
def plot_distributions(
    data: pd.DataFrame,
    features: Optional[Union[List[str], str]] = None,
    bins: int = 30,
    kde: bool = True,
    hist: bool = True,
    figsize: tuple = (10, 6),
    title: str = "Feature Distributions",
    kind: str = 'single', 
    **kwargs
):
    """
    Plots the distribution of numeric columns in the DataFrame. The function 
    allows customization of the plot to include histograms and/or kernel 
    density estimates (KDE). It supports univariate, bivariate, and trivariate 
    distributions.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the data to plot. Only numeric columns 
        are used for plotting the distribution.

    features : list of str or str, optional
        The specific features (columns) to plot. If not provided, all 
        numeric features are plotted. If a single feature is provided as 
        a string, it is automatically treated as a list.

    bins : int, optional, default=30
        The number of bins to use for the histogram. This parameter is 
        ignored if `hist` is set to False.
    
    kde : bool, optional, default=True
        Whether or not to include the Kernel Density Estimate (KDE) plot. 
        If False, only the histogram will be shown.
    
    hist : bool, optional, default=True
        Whether or not to include the histogram in the plot. If False, 
        only the KDE plot will be shown.
    
    figsize : tuple, optional, default=(10, 6)
        The size of the figure (width, height) to create for the plot.
    
    title : str, optional, default="Feature Distributions"
        Title of the plot.
    
    kind : str, optional, default='single'
        Type of distribution plot to generate:
        - 'single': Univariate distribution (1D plot for each feature).
        - 'pair': Bivariate distribution (2D plot for two features).
        - 'triple': Trivariate distribution (3D plot for three features).

    **kwargs : additional keyword arguments passed to seaborn or matplotlib 
              functions for further customization.

    Returns
    -------
    None
        Displays the plot of the distributions.

    Examples
    --------
    >>> from gofast.utils.mlutils import plot_distributions
    >>> from gofast.datasets import load_hlogs
    >>> data = load_hlogs().frame  # get the frame
    >>> plot_distributions(data, features=['longitude', 'latitude', 'subsidence'],
                           kind='triple')
    """

    # If no specific features provided, select numeric features automatically
    if features is None:
        features = data.select_dtypes(include=np.number).columns.tolist()
    
    features = columns_manager(features) 
    
    # Ensure features are valid
    invalid_features = [f for f in features if f not in data.columns]
    if invalid_features:
        raise ValueError(f"Invalid features: {', '.join(invalid_features)}")

    # Univariate Plot (Single feature distribution)
    if kind == 'single':
        plt.figure(figsize=figsize)
        for feature in features:
            plt.subplot(len(features), 1, features.index(feature) + 1)
            if hist:
                sns.histplot(data[feature], kde=kde, bins=bins, **kwargs)
            plt.title(f"{feature} Distribution")
        plt.tight_layout()
        plt.show()

    # Bivariate Plot (Pairwise feature distribution)
    elif kind == 'pair':
        if len(features) != 2:
            raise ValueError(
                "For 'pair' plot type, exactly 2 features must be specified.")
        
        plt.figure(figsize=figsize)
        sns.jointplot(data=data, x=features[0], y=features[1], kind="kde", **kwargs)
        plt.suptitle(f"{features[0]} vs {features[1]} Distribution")
        plt.show()

    # Trivariate Plot (3D distribution)
    elif kind == 'triple':
        if len(features) != 3:
            raise ValueError(
                "For 'triple' plot type, exactly 3 features must be specified.")

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        x = data[features[0]]
        y = data[features[1]]
        z = data[features[2]]
        
        ax.scatter(x, y, z, c=z, cmap='viridis')
        
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        ax.set_zlabel(features[2])
        plt.title(f"{features[0]} vs {features[1]} vs {features[2]} Distribution")
        plt.show()


@return_fig_or_ax(return_type ='ax')
@default_params_plot(
    savefig=PlotConfig.AUTOSAVE('my_uncertainty_plot.png'), 
    title ="Distribution of Uncertainties",
    fig_size=None 
 )
@validate_params ({ 
    'df': ['array-like'], 
    'cols': ['array-like', None], 
    'kind': [StrOptions({ 'box', 'violin', 'strip', 'swarm'})], 
    'numeric_only': [bool], 
    })
@isdf 
@param_deprecated_message(
    conditions_params_mappings=[
        {
            'param': 'numeric_only',
            'condition': lambda v: v is False or v is None,
            'message': ( 
                "Current version only supports 'numeric_only=True'."
                " Resetting numeric_only to True. Note: this parameter"
                " should be removed in the future version."
                ),
            'default': True
        }
    ]
)
def plot_uncertainty(
    df,
    cols=None,
    kind='box',
    figsize=None,
    numeric_only=True,
    title=None,
    ylabel=None,
    xlabel_rotation=45,
    palette='Set2',
    showfliers=True,
    grid=True,
    savefig=None,
    dpi=300, 
    **kws
):
    r"""
    Plot uncertainty distributions from numeric columns in a DataFrame 
    using various plot types. This function helps visualize the spread 
    and variability (uncertainty) of predictions or measurements by 
    generating box plots, violin plots, strip plots, or swarm plots 
    for each selected column.

    Mathematically, given a set of features 
    :math:`X = \{x_1, x_2, \ldots, x_n\}`, each representing a 
    distribution of values, we aim to represent their statistical 
    properties. For box plots, for example, we often visualize: 
    median ( :math:`m` ), quartiles ( :math:`q_1, q_3` ), and 
    outliers [1]_. Such visualizations allow users to quickly 
    assess uncertainty and variability.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data to plot. If `<cols>` is None 
        and `<numeric_only>` is True, all numeric columns in `df` are 
        selected.
    cols : list of str or None, optional
        The columns from `df` to visualize. If None, numeric columns 
        are detected automatically if `<numeric_only>` is True. If 
        `<numeric_only>` is False and no columns are numeric, all 
        columns are chosen.
    kind : str, optional
        The type of plot to generate. Supported values:
        
        - ``'box'``: Box plot, showing quartiles and outliers.
        - ``'violin'``: Violin plot, showing the kernel density 
          estimate of the distribution.
        - ``'strip'``: Strip plot, plotting individual data points.
        - ``'swarm'``: Swarm plot, arranging points to avoid 
          overlapping.
        
        Defaults to ``'box'``.
    figsize : tuple or None, optional
        The size of the figure in inches (width, height). If None, 
        determined automatically based on the number of columns.
    numeric_only : bool, optional
        If True, only numeric columns are considered when `<cols>` is 
        None. This helps avoid errors if `df` contains non-numeric 
        data. Default is True.
    title : str, optional
        The title of the plot, displayed above the visualization.
    ylabel : str, optional
        The label for the y-axis. Defaults to "Predicted Value".
    xlabel_rotation : int or float, optional
        The rotation angle (in degrees) for the x-axis labels. 
        Default is 45 degrees.
    palette : str or sequence, optional
        The color palette used for the plot. See seaborn documentation 
        for available palettes.
    showfliers : bool, optional
        If True (for box plots), outliers are shown. For other plot 
        types, ignored. Default is True.
    grid : bool, optional
        If True, a grid is displayed (on the y-axis for typical 
        distributions). Default is True.
    savefig : str or None, optional
        If not None, the figure is saved to the specified path at the 
        given `<dpi>` resolution. If None, the figure is displayed 
        interactively.
    dpi : int, optional
        The resolution of the saved figure in dots per inch. Default 
        is 300.

    Methods
    -------
    This object is a standalone function and does not provide any 
    additional callable methods. Users interact solely through its 
    parameters.

    Notes
    -----
    - By selecting different `<kind>` values, users can choose 
      the representation that best suits their data. Box plots show 
      summary statistics, violin plots add density information, 
      strip and swarm plots display raw data points.
    - If `<cols>` is not provided and `<numeric_only>` is True, this 
      function tries to detect numeric columns automatically. If no 
      numeric columns are found, a ValueError is raised.

    Examples
    --------
    >>> from gofast.plot.suite import plot_uncertainty
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'A': np.random.normal(0,1,100),
    ...     'B': np.random.normal(5,2,100),
    ...     'C': np.random.uniform(-1,1,100)
    ... })
    >>> # Automatic numeric detection and box plot:
    >>> plot_uncertainty(df)
    >>> # Use a violin plot and specify columns:
    >>> plot_uncertainty(df, cols=['A','B'], kind='violin')

    See Also
    --------
    seaborn.boxplot : For box plots.
    seaborn.violinplot : For violin plots.
    seaborn.stripplot : For strip plots.
    seaborn.swarmplot : For swarm plots.

    References
    ----------
    .. [1] Tukey, J. W. "Exploratory Data Analysis." Addison-Wesley, 
           1977.

    """
    # If no cols provided, select numeric columns if numeric_only=True
    if cols is None:
        if numeric_only:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            cols = df.columns.tolist()
    cols = columns_manager(cols, empty_as_none= True )
    # If no numeric columns found and numeric_only=True
    if not cols:
        raise ValueError(
            "No columns to plot. Please provide columns"
            " or set numeric_only=False."
            )

    check_features_types(df, features=cols, dtype='numeric')
    # Drop NaN values or handle them
    plot_data = df[cols].dropna()

    # Automatic figsize
    if figsize is None:
        # Simple heuristic: width proportional to number of cols
        # height fixed at 6
        width = max(5, len(cols) * 1.2)
        figsize = (width, 6)

    plt.figure(figsize=figsize)

    # Plot according to type
    if kind == 'box':
        kws= filter_valid_kwargs(sns.boxplot, kws)
        sns.boxplot(
            data=plot_data, showfliers=showfliers, 
            palette=palette, 
            **kws)
    elif kind == 'violin':
        kws= filter_valid_kwargs(sns.violinplot, kws)
        sns.violinplot(data=plot_data, palette=palette, **kws)
    elif kind == 'strip':
        kws= filter_valid_kwargs(sns.stripplot, kws)
        sns.stripplot(data=plot_data, palette=palette, **kws)
    elif kind == 'swarm':
        kws= filter_valid_kwargs(sns.swarmplot, kws)
        sns.swarmplot(data=plot_data, palette=palette, **kws)
   
    plt.title(title, fontsize=14)
    plt.ylabel(ylabel or "Predicted Value", fontsize=12)

    # Rotate x-labels if needed
    plt.xticks(rotation=xlabel_rotation)
    
    if grid:
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    if savefig:
        plt.savefig(savefig, dpi=dpi, bbox_inches='tight')

    plt.show()

@return_fig_or_ax(return_type ='ax')
@default_params_plot(
    savefig=PlotConfig.AUTOSAVE('my_prediction_intervals_plot.png'), 
    title ="Prediction Intervals",
    fig_size=(10, 6) 
 )
@validate_params ({ 
    'df': ['array-like'], 
    'x': ['array-like', None], 
    'line_colors': [str, 'array-like'], 
    })
@deprecated_params_plot
def plot_prediction_intervals(
    df,
    q_cols=None,         
    median_col=None,      
    lower_col=None,       
    upper_col=None,        
    ref_col=None,         
    sample_size=None,      
    dt_col=None,          
    title=None,
    xlabel="Sample",
    ylabel="Value",
    legend=True,
    line_colors=('black', 'blue'),
    fill_color='blue',
    fill_alpha=0.2,
    figsize=None,
    show_grid=True,
    grid_props=None,
    rotation=0, 
    savefig=None,
    dpi=300,
    **kws
):
    """
    Plots predicted intervals (e.g., lower, median, and upper
    quantiles) along with an optional reference series. It
    supports quantile specifications in two ways: either through
    the individual arguments ``<argument``> `lower_col`,
    ``<argument``> `median_col`, and ``<argument``> `upper_col` or
    through ``<argument``> `q_cols` [1]_.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the necessary columns.
    q_cols : dict or list, optional
        Quantile definitions. If it is a dictionary with keys
        like ``q10``, ``q50``, or ``q90``, the numeric portion
        is parsed into float quantiles. If it is a list,
        items are assigned dummy quantile keys (``q0``,
        ``q1``, etc.). If <parameter `q_cols`> is provided,
        any values in <parameter `median_col`>,
        <parameter `lower_col`>, and <parameter `upper_col`>
        are ignored.
    median_col : str, optional
        The column name representing the median predictions
        (deprecated in favor of ``q_cols``). If specified,
        generates a deprecation warning.
    lower_col : str, optional
        The column name representing the lower interval
        (deprecated in favor of ``q_cols``). If specified,
        generates a deprecation warning.
    upper_col : str, optional
        The column name representing the upper interval
        (deprecated in favor of ``q_cols``). If specified,
        generates a deprecation warning.
    ref_col : str, optional
        Column name for a reference series (e.g., true values,
        observations). Plotted as a separate line.
    sample_size : int, float, or str, optional
        Controls downsampling:
        - If int, randomly samples <parameter `sample_size`>
          rows.
        - If float in (0, 1), uses that fraction of rows.
        - If ``auto``, a simple heuristic is used (e.g.,
          sample 500 if the dataset is large).
    dt_col : str, optional
        Column used on the x-axis. If None, the DataFrame
        index is used. If not already in datetime format,
        it is converted using ``pandas.to_datetime``.
    title : str, optional
        Plot title. Defaults to ``"Prediction Intervals"`` if
        unspecified.
    xlabel : str, optional
        X-axis label, defaults to ``"Sample"``.
    ylabel : str, optional
        Y-axis label, defaults to ``"Value"``.
    legend : bool, optional
        If True, displays a legend of plotted series. Defaults
        to True.
    line_colors : tuple of str, optional
        Defines the colors for the reference series and median
        line, in that order. Defaults to ``('black', 'blue')``.
    fill_color : str, optional
        The fill color for the interval range. Defaults to
        ``'blue'``.
    fill_alpha : float, optional
        Opacity of the fill for the interval. Defaults to 0.2.
    figsize : tuple, optional
        The figure size (width, height) in inches.
    show_grid : bool, optional
        If True, a grid is added to the plot. Defaults to True.
    grid_props : dict, optional
        Customization for the grid, e.g. ``{'linestyle': ':',
        'alpha': 0.7}``.
    savefig : str, optional
        If not None, saves the figure to the specified file
        path. The saved figure uses ``dpi`` for resolution.
    dpi : int, optional
        Dots per inch for image saving. Defaults to 300.
    **kws
        Additional keyword arguments passed to
        ``matplotlib`` plot calls (e.g. line style).

    Notes
    -----
    By default, the function attempts to parse columns for the
    lowest, median, and highest quantiles from
    <parameter `q_cols`>. If no valid quantiles are found or if
    <parameter `q_cols`> is missing, the deprecated parameters
    <parameter `lower_col`>, <parameter `median_col`>, and
    <parameter `upper_col`> are used, issuing warnings.
    The upper and lower bounds define a range of prediction
    intervals:

    .. math::
       [\\text{lower}, \\; \\text{upper}]

    and the median or central quantile is plotted as a reference
    line, usually corresponding to :math:`q_{50}`, unless
    otherwise specified:

    .. math::
       \\text{median} = q_{50}

    Examples
    --------
    >>> from gofast.plot.suite import plot_prediction_intervals
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...   'lower': [0.8, 1.0, 1.2],
    ...   'median': [1.0, 1.2, 1.4],
    ...   'upper': [1.2, 1.4, 1.6]
    ... })
    >>> # Example with deprecated parameters
    >>> plot_prediction_intervals(
    ...   df,
    ...   lower_col='lower',
    ...   median_col='median',
    ...   upper_col='upper'
    ... )
    >>> # Example with q_cols
    >>> q_def = {'q10': 'lower', 'q50': 'median', 'q90': 'upper'}
    >>> plot_prediction_intervals(df, q_cols=q_def)

    See Also
    --------
    None currently.

    References
    ----------
    .. [1] Doe, J., & Smith, A. (2022). Visualizing data
       intervals: A systematic approach. Journal of Plotting
       Science, 10(3), 55-67.
    """
    # Update default plot parameters
    _param_defaults.update({
        'title': "Prediction Intervals",
        'xlabel': xlabel,
        'ylabel': ylabel,
    })
    params = _set_defaults(title=title, xlabel=xlabel, ylabel=ylabel)

    # Validate DataFrame
    is_frame(df, df_only=True, raise_exception=True, objname='df')

    # Handle dt_col vs. using index
    if dt_col is None:
        x_data = df.index
    else:
        # If dt_col exists, ensure it's datetime if possible
        if dt_col not in df.columns:
            raise ValueError(f"Column '{dt_col}' not found in df.")
        if not pd.api.types.is_datetime64_any_dtype(df[dt_col]):
            df[dt_col] = pd.to_datetime(df[dt_col], errors='coerce')
        x_data = df[dt_col]

    # Handle sample_size
    # If sample_size is an integer, float, or 'auto', downsample accordingly
    n = len(df)
    if sample_size is not None:
        if isinstance(sample_size, int):
            # Clip sample_size to the number of rows if larger
            sample_size = min(sample_size, n)
            df = df.sample(sample_size, random_state=42)
        elif isinstance(sample_size, float):
            # Interpret float as fraction
            if 0 < sample_size < 1:
                df = df.sample(int(n * sample_size), random_state=42)
            else:
                warnings.warn(
                    f"sample_size={sample_size} is not in (0,1). Ignored."
                )
        elif sample_size == 'auto':
            # Simple heuristic: if n > 500, sample 500
            if n > 500:
                df = df.sample(500, random_state=42)
        else:
            warnings.warn(
                f"Unrecognized sample_size={sample_size}. Ignored."
            )
        # If we resampled df, also resample x_data accordingly
        x_data = df[dt_col] if dt_col else df.index

    #
    # Resolve lower, median, upper columns from q_cols if provided
    #----
    # If q_cols is a dict (e.g., {'q10': col10, 'q50': col50, 'q90': col90}),
    # try to identify min/median/max keys automatically.
    if q_cols is not None:
        # Convert list to dict if necessary
        if isinstance(q_cols, list):
            # Example: q_cols=['col_low','col_med','col_high']
            # We'll map them in ascending order as best as we can
            q_cols_dict = {}
            for i, col_name in enumerate(q_cols):
                q_cols_dict[f"q{i}"] = col_name
            q_cols = q_cols_dict

        # Attempt to parse numeric portion from keys like 'q10', 'q50', 'q90'
        parsed = {}
        for k, col_name in q_cols.items():
            if k.startswith('q'):
                try:
                    q_val = float(k[1:])
                    parsed[q_val] = col_name
                except ValueError:
                    warnings.warn(f"Cannot parse quantile '{k}'. Skipped.")
            else:
                warnings.warn(f"Key '{k}' is not prefixed with 'q'. Skipped.")

        if not parsed:
            warnings.warn(
                "No valid quantile columns found in q_cols. "
                "Falling back to median_col, lower_col, upper_col."
            )
        else:
            sorted_qvals = sorted(parsed.keys())
            # Lowest quantile
            lower_key = sorted_qvals[0]
            lower_col = parsed[lower_key]
            # Highest quantile
            upper_key = sorted_qvals[-1]
            upper_col = parsed[upper_key]
            # Median: prefer '50' if present, else middle
            if 50.0 in parsed:
                median_col = parsed[50.0]
            else:
                mid_idx = len(sorted_qvals) // 2
                median_col = parsed[sorted_qvals[mid_idx]]

    # Ensure columns exist
    required_cols = [median_col, lower_col, upper_col]
    if any(col is None for col in required_cols):
        raise ValueError(
            "Unable to resolve lower_col, median_col, upper_col. "
            "Provide them or use q_cols with valid mappings."
        )
    exist_features(df, features=required_cols)

    # Extract the values
    median_vals = df[median_col].values
    lower_vals = df[lower_col].values
    upper_vals = df[upper_col].values

    # Prepare figure
    #
    fig, ax = plt.subplots(figsize=figsize)
    
    line_colors = columns_manager(line_colors, empty_as_none=False)
    line_colors = make_plot_colors(range(2), line_colors)
    # Plot reference column if provided
    if ref_col is not None:
        exist_features(
            df, features=ref_col, name='Reference feature column')
        ax.plot(
            x_data,
            df[ref_col].values,
            label=ref_col,
            color=line_colors[0],
            **filter_valid_kwargs(ax.plot, {})
        )

    # Plot median line
    median_kws = filter_valid_kwargs(ax.plot, kws)
    ax.plot(
        x_data,
        median_vals,
        label=median_col,
        color=line_colors[1],
        **median_kws
    )

    # Fill between for interval
    fill_kws = filter_valid_kwargs(ax.fill_between, kws)
    ax.fill_between(
        x_data,
        lower_vals,
        upper_vals,
        color=fill_color,
        alpha=fill_alpha,
        label=f'{lower_col} - {upper_col}',
        **fill_kws
    )

    # Apply plot labels and properties
    ax.set_title(params['title'])
    ax.set_xlabel(params['xlabel'])
    ax.set_ylabel(params['ylabel'])

    if show_grid:
        grid_cfg = grid_props or {'linestyle': ':', 'alpha': 0.7}
        ax.grid(True, **grid_cfg)

    if legend:
        ax.legend()

    # Save figure if requested
    if savefig:
        plt.savefig(savefig, dpi=dpi, bbox_inches='tight')

    # Show the plot
    plt.show()


@return_fig_or_ax(return_type ='ax')
@default_params_plot(
    savefig=PlotConfig.AUTOSAVE('my_temporal_trends_plot.png'), 
    title ="Temporal Trends",
    fig_size=(8, 6) 
 )
@validate_params ({ 
    'df': ['array-like'], 
    'dt_col':[str], 
    'value_cols': ['array-like', str], 
    'line_colors': ['array-like'], 
    'to_datetime': [StrOptions({'auto', 'Y','M','W','D','H','min','s'}), None], 
    })
def plot_temporal_trends(
    df,
    dt_col,
    value_cols,
    agg_func='mean',
    freq=None,
    to_datetime=None,
    kind='line',
    figsize=None,
    color=None,
    marker='o',
    linestyle='-',
    grid=True,
    title=None,
    xlabel=None,
    ylabel=None,
    legend=True,
    xrotation=0,
    savefig=None,
    dpi=300,
    verbose=0
):
    r"""
    Plot temporal trends of aggregated values over time, allowing 
    flexible input data handling, aggregation, and optional conversion 
    of date columns to a datetime format. By grouping data by `dt_col` 
    and applying an aggregation function `agg_func`, this function 
    derives time-based trends that can be visualized using different 
    plot styles.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing `dt_col` and `value_cols`.
    dt_col : str
        The name of the column in `df` representing the temporal axis.
        If `to_datetime` is provided, the column may be converted to 
        a datetime format.
    value_cols : str or list of str
        The column name(s) representing the values to aggregate and 
        plot over time. If a single string is provided, it is internally 
        converted to a list.
    agg_func : str or callable, optional
        The aggregation function applied to the grouped data. If a 
        string (e.g., 'mean'), it must correspond to a valid pandas 
        aggregation method. If callable, it should accept an array 
        and return a scalar.
    freq : None or str, optional
        Reserved for future use or advanced frequency adjustments. 
        Currently not implemented.
    to_datetime : {None, 'auto', 'Y','M','W','D','H','min','s'}, optional
        Controls conversion of `<dt_col>` to datetime if not already 
        in datetime format:
        
        - None: No conversion, dt_col used as-is.
        - 'auto': Attempt automatic detection and conversion via 
          `smart_ts_detector`.
        - Explicit codes (e.g. 'Y','M','W','min','s') instruct how 
          to interpret and convert numeric values into datetime 
          objects (e.g., integers as years, months, weeks).
    kind : {'line', 'bar'}, optional
        The type of plot:
        
        - 'line': A line plot connecting aggregated points over time.
        - 'bar': A bar plot showing discrete aggregated values.
    figsize : tuple, optional
        Figure size (width, height) in inches. Default is (10,6).
    color : color or sequence of colors, optional
        Color(s) for the plot lines or bars. If None, defaults 
        to matplotlib defaults.
    marker : str, optional
        Marker style for line plots. Default is 'o'.
    linestyle : str, optional
        Line style for line plots. Default is '-'.
    grid : bool, optional
        If True, display a grid. Default is True.
    title : str, optional
        The title of the plot. Default "Temporal Trends".
    xlabel : str or None, optional
        Label for the x-axis. If None, uses `dt_col`.
    ylabel : str or None, optional
        Label for the y-axis. If None, defaults to "Value".
    legend : bool, optional
        If True, display a legend for multiple `value_cols`. 
        Default is True.
    xrotation : int or float, optional
        Rotation angle for x-axis tick labels. Default is 0 (no rotation).
    savefig : str or None, optional
        File path to save the figure. If None, displays interactively.
    dpi: int, optional
        Resolution of saved figure in dots-per-inch. Default is 300.
    verbose : int, optional
        Verbosity level for logging
 
    Returns
    -------
    None
        Displays the plot or saves it to the specified file.

    Notes
    -----
    Consider a DataFrame :math:`D` with a time-related column 
    :math:`d \in D` and value columns :math:`v \in D`. The goal is 
    to produce a plot:

    .. math::
       T(t) = \text{agg_func}(\{v_i | d_i = t\}),

    where `agg_func` (e.g. `mean`) is applied to subsets of 
    `<value_cols>` grouped by each unique time unit in `<dt_col>`. 
    The resulting series :math:`T(t)` highlights how `<value_cols>` 
    evolve over time.
    
    If `to_datetime` is not None, `smart_ts_detector` may be used 
    internally to guess and convert `dt_col` into a datetime 
    object. For example, if `to_datetime='Y'` and the column 
    contains integers like 2020, 2021, they are interpreted as years 
    and converted accordingly.

    If multiple `value_cols` are provided and `kind='line'`, each 
    column is plotted as a separate line. If `kind='bar'`, a grouped 
    bar plot is produced automatically using `DataFrame.plot`.

    Examples
    --------
    >>> from gofast.plot.suite import plot_temporal_trends
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'year': [2020,2021,2022,2023],
    ...     'subsidence': [5.4, 5.6, 5.9, 6.1]
    ... })
    >>> # Simple line plot by year
    >>> plot_temporal_trends(df, 'year', 'subsidence')
    >>> # Convert from year int to datetime year and plot
    >>> plot_temporal_trends(df, 'year', 'subsidence', to_datetime='Y')

    See Also
    --------
    pandas.DataFrame.plot : General plotting tool for DataFrames.
    gofast.core.array_manager.smart_ts_detector : 
        Helps infer and convert numeric columns to datetime.
    """
    is_frame(df, df_only=True, raise_exception =True, objname='df')
    
    # If to_datetime is specified, use smart_ts_detector to convert dt_col
    if to_datetime is not None:
        # We will call smart_ts_detector with appropriate parameters to 
        # handle the conversion return_types='df' to get a converted df back
        if verbose >= 2:
            print(f"Converting {dt_col!r} using smart_ts_detector"
                  f" with to_datetime={to_datetime}")
        df = smart_ts_detector(
            df=df,
            dt_col=dt_col,
            return_types='df',
            to_datetime=to_datetime,
            error='raise',  # Raise error if something goes wrong
            verbose=verbose
        )

    # Ensure value_cols is a list
    value_cols = columns_manager(value_cols, empty_as_none= False )
    # checks whether columns exist
    exist_features(df, features= value_cols, name ='Value columns')

    # Perform grouping by dt_col and aggregation
    grouped = df.groupby(dt_col)[value_cols]

    if isinstance(agg_func, str):
        # If a string is given, we assume it's a known aggregation function
        agg_data = grouped.agg(agg_func)
    else:
        # If agg_func is callable, use it directly
        agg_data = grouped.agg(agg_func)

    # Create figure and plot
    plt.figure(figsize=figsize)

    if kind == 'line':
        # Plot each value_col as a separate line
        for c in value_cols:
            plt.plot(
                agg_data.index, agg_data[c],
                marker=marker,
                linestyle=linestyle,
                color=color,
                label=c if legend else None
            )
    elif kind == 'bar':
        # Let pandas handle the bar plot
        agg_data.plot(
            kind='bar', ax=plt.gca(),
            legend=legend, color=color
        )
    else:
        # Fallback to line if unknown kind
        for c in value_cols:
            plt.plot(
                agg_data.index, agg_data[c],
                marker=marker,
                linestyle=linestyle,
                color=color,
                label=c if legend else None
            )

    # Set plot titles and labels
    plt.title(title)
    plt.xlabel(xlabel if xlabel else dt_col)
    plt.ylabel(ylabel if ylabel else "Value")
    plt.xticks(rotation=xrotation)

    # Add grid if requested
    if grid:
        plt.grid(True, linestyle='--', alpha=0.7)

    # Add legend if needed
    if legend and kind != 'bar':
        plt.legend()

    # Save figure if a path is provided
    if savefig:
        plt.savefig(savefig, dpi=dpi, bbox_inches='tight')

    # Show the plot
    plt.show()
    
def _plot_error_bars(
    df, q_cols, 
    dt_col, 
    ax=None, 
    label=None, 
    capsize=5, 
    title=None, 
    xlabel="Date", 
    ylabel="Values",
    **kws
    ):
    """
    Helper function for error bars plot with improved flexibility
    and robustness.
    
    - Handles missing dt_col gracefully by checking for its presence.
    - Allows customization of plot title, labels, and plot style.
    - Can optionally use provided axes for more control over
       subplot placements.
    """
    _param_defaults.update ({
        'label': "Quantile 50", 
        'title': "Error Bars: Quantiles", 
        'xlabel': "Date", 
        "ylabel": "Values", 
    })
    # Set defaults using _set_defaults
    params = _set_defaults(
        title=title, xlabel=xlabel, ylabel=ylabel,
        label=label 
    )

 
    # Set the axes (use provided axes or create a new one)
    if ax is None:
        ax = plt.gca()
    
    # Compute the error bars using the quantiles provided (q10, q50, q90)
    yerr_lower = df[q_cols[1]] - df[q_cols[0]]  # q50 - q10
    yerr_upper = df[q_cols[2]] - df[q_cols[1]]  # q90 - q50
    
    # Error bars plot
    kws = filter_valid_kwargs(ax.errorbar, kws)
    ax.errorbar(
        df[dt_col], df[q_cols[1]], yerr=[yerr_lower, yerr_upper],
        fmt='o', 
        label=label, capsize=capsize, **kws
    )
    
    # Set plot title and labels
    ax.set_title(params['title'])
    ax.set_xlabel(params['xlabel'])
    ax.set_ylabel(params['ylabel'])
    
    # Show legend
    ax.legend()
    

def _plot_line_shaded(
    df, dt_col, q_cols, 
    ax=None,
    alpha=0.5, 
    line_color='blue', 
    fill_color='lightblue', 
    title="Shaded Region Plot: Quantiles", 
    xlabel="Date", 
    ylabel="Values", 
    label_median="Median (q50)", 
    label_shaded="q10 to q90 range", 
    **kws
    ):
    """
    Helper function for shaded region plot with improved flexibility
    and robustness.

    - Handles missing dt_col gracefully by checking for its presence.
    - Allows customization of plot title, labels, and plot style.
    - Can optionally use provided axes for more control over subplot placements.
    """
    _param_defaults.update ({
        'fill_color': "lightblue", 
        'line_color':"blue", 
        'title': "Shaded Region Plot: Quantiles", 
        'xlabel': "Date", 
        "ylabel": "Values", 
        "label_median": "Median (q50)", 
        "label_shaded": "q10 to q90 range", 
    })
    # Set defaults using _set_defaults
    params = _set_defaults(
        title=title, xlabel=xlabel, ylabel=ylabel,
        fill_color=fill_color, line_color=line_color, 
        label_median=label_median, 
        label_shaded=label_shaded, 
    )
    
    # Set the axes (use provided axes or create a new one)
    if ax is None:
        ax = plt.gca()
    
    kws = filter_valid_kwargs(ax.plot, kws)
    # Plot the median (q50)
    ax.plot(
        df[dt_col], df[q_cols[1]], 
        label=params['label_median'], 
        color=params ['line_color'], 
        **kws
    )
    # Shaded region between q10 and q90
    ax.fill_between(
        df[dt_col], df[q_cols[0]], df[q_cols[2]], 
        color=params ['fill_color'], 
        alpha=alpha, label=params ['label_shaded']
    )

    # Set plot title and labels
    ax.set_title(params['title'])
    ax.set_xlabel(params['xlabel'])
    ax.set_ylabel(params['ylabel'])

    # Show legend
    ax.legend()

def _plot_box_plot(
    df, dt_col, q_cols, 
    ax=None, 
    title=None, 
    xlabel=None, 
    ylabel=None, 
    palette=None, 
    notch=False, 
    var_name=None, 
    value_name=None, 
    width=0.8, 
    **kws
):
    """
    Helper function for creating a box plot with added flexibility and robustness.
    
    - Handles missing dt_col gracefully by checking for its presence.
    - Allows customization of plot title, labels, box styles, and color palettes.
    - Can optionally use provided axes for more control over subplot placements.
    """
    _param_defaults.update ({
        'title': "Box Plot: Quantiles Across Time", 
        'xlabel': "Date", 
        "ylabel": "Values", 
    })
    # Set defaults using _set_defaults
    params = _set_defaults(
        title=title, xlabel=xlabel, ylabel=ylabel, palette=palette,
        var_name=var_name, value_name=value_name, width=width
    )

    # Set the axes (use provided axes or create a new one)
    if ax is None:
        ax = plt.gca()
    
    # Melt the dataframe to have a long format suitable for the box plot
    df_box = df.melt(
        id_vars=[dt_col], value_vars=q_cols, 
        var_name=params['var_name'], value_name=params['value_name']
    )
    
    kws = filter_valid_kwargs(sns.boxplot, kws)
    # Create the box plot with flexibility for customization
    sns.boxplot(
        x=dt_col, y=params['value_name'],
        hue=params['var_name'], data=df_box, 
        ax=ax, palette=params['palette'], 
        notch=notch, 
        width=params['width'], **kws
    )

    # Set the title and labels
    ax.set_title(params['title'])
    ax.set_xlabel(params['xlabel'])
    ax.set_ylabel(params['ylabel'])

    # Show the legend
    ax.legend()

def _plot_violin_plot(
    df, 
    dt_col, 
    q_cols, 
    ax=None, 
    title=None, 
    xlabel=None, 
    ylabel=None, 
    palette=None, 
    split=False, 
    scale="area", 
    **kws
):
    """
    Helper function for creating a violin plot with added flexibility 
    and robustness.
    
    - Handles missing dt_col gracefully by checking for its presence.
    - Allows customization of plot title, labels, violin styles, and 
      color palettes.
    - Can optionally use provided axes for more control over subplot
      placements.
    """
    # we need to update the title 
    _param_defaults.update ({
        'title': "Violin Plot: Quantiles Across Time",
        'xlabel': "Date", 
        "ylabel": "Values", 
        "palette": "Set2"
    })
    # Set defaults using _set_defaults and __param_defaults updated. 
    params = _set_defaults(
        title=title, xlabel=xlabel, ylabel=ylabel, palette=palette
    )

    # Set the axes (use provided axes or create a new one)
    if ax is None:
        ax = plt.gca()
    
    # Melt the dataframe to have a long format suitable for the violin plot
    df_violin = df.melt(
        id_vars=[dt_col], value_vars=q_cols, 
        var_name="Quantiles", value_name="Values"
    )
    
    # Auto-disable split if there are not exactly two hue levels
    unique_levels = df_violin["Quantiles"].unique()
    
    if split and len(unique_levels) != 2:
        warnings.warn("`split` set to True but found {} hue levels."
                      " Disabling split.".format(len(unique_levels)))
        split = False
    
    kws = filter_valid_kwargs(sns.violinplot, kws)
    # Create the violin plot with flexibility for customization
    sns.violinplot(
        x=dt_col, y="Values", hue="Quantiles", 
        data=df_violin, ax=ax, 
        palette=params['palette'],
        split=split, scale=scale, 
        **kws
    )

    # Set the title and labels
    ax.set_title(params['title'])
    ax.set_xlabel(params['xlabel'])
    ax.set_ylabel(params['ylabel'])

    # Show the legend
    ax.legend()

def _plot_histogram(
    df, q_cols, 
    bins=30, 
    alpha=0.5, 
    ax=None, 
    title=None, 
    xlabel=None,
    ylabel=None, 
    **kws
):
    """
    Helper function for creating a histogram with overlapping
    distributions.
    
    - Allows customization of bins, alpha (opacity), and other 
      plot styles.
    - Handles multiple quantiles and overlays them in the 
      same histogram.
    - Uses defaults if parameters are not provided.
    """
    # Update the _param_defaults for these functions
    _param_defaults.update({
        'title': "Histogram with Overlapping Quantiles", 
        'xlabel': "Values", 
        'ylabel': "Density", 
    })

    # Set defaults using _set_defaults
    params = _set_defaults(
        title=title, xlabel=xlabel, ylabel=ylabel, 
        bins=bins, alpha=alpha
    )

    # Set the axes (use provided axes or create a new one)
    if ax is None:
        ax = plt.gca()

    # Create the histogram for each quantile
    kws = filter_valid_kwargs(ax.hist, kws)
    ax.hist(
        df[q_cols[0]], bins=params['bins'], alpha=params['alpha'], 
        label="q10", density=True, **kws
    )
    ax.hist(
        df[q_cols[1]], bins=params['bins'], alpha=params['alpha'], 
        label="q50", density=True, **kws
    )
    ax.hist(
        df[q_cols[2]], bins=params['bins'], alpha=params['alpha'], 
        label="q90", density=True, **kws
    )

    # Set the title and labels
    ax.set_title(params['title'])
    ax.set_xlabel(params['xlabel'])
    ax.set_ylabel(params['ylabel'])

    # Show the legend
    ax.legend()

def _plot_fan_chart(
    df, dt_col, 
    q_cols, 
    title=None, 
    xlabel=None,
    ylabel=None, 
    ax=None, 
    alpha=0.5, 
    color1=None,
    color2=None, 
    **kws
):
    """
    Helper function for creating a fan chart to display confidence 
    intervals.
    
    - The fan chart is created by plotting the median (q50) and 
      filling between the lower (q10) and upper (q90) quantiles.
    - Allows customization of plot colors and transparency.
    """
    # Update the _param_defaults for these functions
    _param_defaults.update({
        'title': "Fan Chart: q10, q50, q90", 
        'xlabel': "Date", 
        'ylabel': "Values", 
        'color1':'lightblue',
        'color2':'lightgreen', 
        
    })
    # Set defaults using _set_defaults
    params = _set_defaults(
        title=title, xlabel=xlabel,
        ylabel=ylabel
    )

    # Set the axes (use provided axes or create a new one)
    if ax is None:
        ax = plt.gca()

    # Plot the median (q50)
    kws = filter_valid_kwargs(ax.plot, kws)
    ax.plot(
        df[dt_col], df[q_cols[1]], 
        label="Median (q50)", 
        color='blue', 
        **kws
    )

    # Fill between the quantiles (q10 to q50 and q50 to q90)
    ax.fill_between(
        df[dt_col], df[q_cols[0]], df[q_cols[1]], 
        color= params['color1'], alpha=alpha, 
        label="q10 to q50 range"
    )
    ax.fill_between(
        df[dt_col], df[q_cols[1]], df[q_cols[2]], 
        color=params['color2'], 
        alpha=alpha, 
        label="q50 to q90 range"
    )

    # Set the title and labels
    ax.set_title(params['title'])
    ax.set_xlabel(params['xlabel'])
    ax.set_ylabel(params['ylabel'])

    # Show the legend
    ax.legend()

def _plot_kde(
    df, q_cols, 
    title=None, 
    xlabel=None, 
    ylabel=None, 
    ax=None, 
    alpha=0.5, 
    **kws
):
    """
    Helper function for creating a Kernel Density Estimation (KDE)
    plot for multiple quantiles.
    
    - Handles the plotting of KDE for each quantile (q10, q50, q90).
    - Allows customization of plot title, labels, transparency (alpha), 
    and other styles.
    """
    _param_defaults.update({
        'title': "KDE Plot: Quantiles", 
        'xlabel': "Values", 
        'ylabel': "Density", 
    })

    # Set defaults using _set_defaults
    params = _set_defaults(
        title=title, xlabel=xlabel, ylabel=ylabel, alpha=alpha
    )

    # Set the axes (use provided axes or create a new one)
    if ax is None:
        ax = plt.gca()

    # Plot KDE for each quantile
    kws = filter_valid_kwargs(sns.kdeplot, kws)
    sns.kdeplot(
        df[q_cols[0]], label="q10", fill=True,
        alpha=params['alpha'], **kws
    )
    sns.kdeplot(
        df[q_cols[1]], label="q50", fill=True,
        alpha=params['alpha'], **kws
    )
    sns.kdeplot(
        df[q_cols[2]], label="q90", fill=True,
        alpha=params['alpha'], **kws
    )

    # Set the title and labels
    ax.set_title(params['title'])
    ax.set_xlabel(params['xlabel'])
    ax.set_ylabel(params['ylabel'])

    # Show the legend
    ax.legend()

def _plot_monte_carlo(
    df, dt_col, q_cols, 
    n_simulations=100, 
    title=None, 
    xlabel=None, 
    ylabel=None,
    ax=None, alpha=0.2, 
    color1=None, 
    color2=None, 
    **kws
):
    """
    Helper function for creating a Monte Carlo simulation plot 
    to visualize uncertainty.
    
    - The plot shows multiple simulations based on the quantile values.
    - Allows customization of plot title, labels, colors, transparency,
    and other styles.
    """
    _param_defaults.update({
        'alpha': 0.2, 
        'color1': 'lightgray', 
        'color2': 'blue', 
        'title': "Monte Carlo Simulation Plot", 
        'xlabel': "Date", 
        'ylabel': "Values"
    })
    # Set defaults using _set_defaults
    params = _set_defaults(
        title=title, xlabel=xlabel, ylabel=ylabel, alpha=alpha,
        color1=color1, color2=color2
    )

    # Set the axes (use provided axes or create a new one)
    if ax is None:
        ax = plt.gca()

    # Simulate data based on quantile 50 (q50) for Monte Carlo
    simulated_data = np.random.normal(
        loc=df[q_cols[1]].mean(), 
        scale=df[q_cols[1]].std(), 
        size=(n_simulations, len(df))
    )

    # Plot the Monte Carlo simulations
    kws = filter_valid_kwargs(ax.plot, kws)
    ax.plot(
        df[dt_col], simulated_data.T, 
        color=params['color1'], alpha=params['alpha'], 
        **kws
    )

    # Plot the median (q50)
    ax.plot(
        df[dt_col], df[q_cols[1]], 
        label="Quantile 50 (Median)", 
        color=params['color2'], 
        **kws
    )

    # Set the title and labels
    ax.set_title(params['title'])
    ax.set_xlabel(params['xlabel'])
    ax.set_ylabel(params['ylabel'])

    # Show the legend
    ax.legend()

@return_fig_or_ax(return_type ='ax')
@default_params_plot(
    savefig=PlotConfig.AUTOSAVE('my_w.uncertainty_plot.png'), 
    title ="Uncertainties",
    fig_size=(8, 6) 
 )
@validate_params ({ 
    'df': ['array-like'], 
    'dt_col':[str, None], 
    'q_cols': ['array-like']
    })
@isdf 
def plot_with_uncertainty(
    df, 
    q_cols, 
    dt_col=None, 
    kind="errors", 
    figsize=(10, 6), 
    savefig=None, 
    show_grid=True, 
    grid_props=None, 
    n_simulations=100, 
    rotation=45, 
    bins=30, 
    alpha=0.5, 
    capsize=5, 
    title=None, 
    xlabel=None, 
    ylabel=None, 
    label=None, 
    ax=None, 
    line_color="blue", 
    fill_color='lightblue', 
    label_median=None, 
    label_shaded=None,  # "q10 to q90 range"
    palette="Set2", 
    notch=False, 
    var_name=None, 
    value_name=None, 
    width=.8, 
    scale='area', 
    split=True, 
    color1=None, 
    color2=None, 
    verbose=0, 
    **kws
):
    """
    Plot various uncertainty visualizations.

    This function provides a flexible framework for visualizing 
    uncertainty in data, with support for several plot types such 
    as error bars, shaded lines, histograms, and Monte Carlo simulations. 
    The user can specify the type of plot, customize the appearance, 
    and handle multiple quantiles for uncertainty representation.

    Parameters
    ----------
    df : pandas.DataFrame
        The input data frame containing the data to be plotted.
        
    q_cols : list of str
        List of column names representing the quantiles to be used 
        in the plot. This should contain the quantile columns (e.g., 
        q10, q50, q90).

    dt_col : str, optional, default=None
        The name of the column containing the datetime or time-related 
        values. If `None`, the function will attempt to automatically 
        detect a datetime column or use the index as the time series.

    kind : str, optional, default="errors"
        The type of plot to generate. Options include:
        - "errors": Error bar plot
        - "line_shaded": Line plot with shaded region
        - "box": Box plot
        - "violin": Violin plot
        - "histogram": Histogram plot
        - "fan_chart": Fan chart
        - "kde": Kernel Density Estimation plot
        - "monte_carlo": Monte Carlo simulation plot
        
    figsize : tuple, optional, default=(10, 6)
        The size of the plot in inches (width, height).
        
    savefig : str or None, optional, default=None
        If provided, the plot will be saved to the specified file path.
        If `None`, the plot will be displayed interactively.
        
    show_grid : bool, optional, default=True
        Whether to show grid lines on the plot.
        
    grid_props : dict, optional, default=None
        Properties for the grid lines. For example, you can specify the
        linestyle, alpha, etc. If `None`, default grid settings are used.

    n_simulations : int, optional, default=100
        The number of Monte Carlo simulations to perform for the 
        "monte_carlo" plot type.
        
    rotation : int, optional, default=45
        The angle of rotation for the x-axis tick labels.
        
    bins : int, optional, default=30
        The number of bins to use for histograms.
        
    alpha : float, optional, default=0.5
        The transparency level of shaded regions or lines. Ranges 
        from 0 (completely transparent) to 1 (completely opaque).
        
    capsize : int, optional, default=5
        The size of the caps on the error bars (if applicable).
        
    title : str, optional, default=None
        The title of the plot.
        
    xlabel : str, optional, default=None
        The label for the x-axis.
        
    ylabel : str, optional, default=None
        The label for the y-axis.
        
    label : str, optional, default=None
        A label to display in the legend for the plot.
        
    ax : matplotlib.axes.Axes, optional, default=None
        The axes on which to plot the data. If `None`, the current 
        active axes will be used.
        
    line_color : str, optional, default="blue"
        The color for the line in line-based plots (e.g., line plots, 
        fan charts).
        
    fill_color : str, optional, default='lightblue'
        The color to fill the shaded regions in line-based plots.
        
    label_median : str, optional, default=None
        The label for the median line (e.g., "Median (q50)") in line-based 
        plots.

    label_shaded : str, optional, default=None
        The label for the shaded region in line-based plots.
        
    palette : str, optional, default="Set2"
        The color palette to use for categorical plots like box plots 
        or violin plots.

    notch : bool, optional, default=False
        Whether to display a notch in box plots. Notches indicate 
        confidence intervals around the median.

    var_name : str, optional, default=None
        The variable name to be used in melted data for box/violin plots.

    value_name : str, optional, default=None
        The value name to be used in melted data for box/violin plots.

    width : float, optional, default=.8
        The width of the boxes in box plots.

    scale : str, optional, default="area"
        The scaling method for violin plots. Options are "area" or "count".

    split : bool, optional, default=True
        Whether to split the violins in the violin plot by hue.

    color1 : str, optional, default=None
        The color to use for the lower quantile region (e.g., q10 to q50) 
        in fan charts.

    color2 : str, optional, default=None
        The color to use for the upper quantile region (e.g., q50 to q90) 
        in fan charts.

    verbose : int, optional, default=0
        The verbosity level. If set to 1, additional information about 
        the plot creation is printed.

    kws : dict, optional
        Additional keyword arguments to pass to the specific plot 
        function being used (e.g., `sns.boxplot`, `sns.violinplot`, 
        `plt.errorbar`).

    Returns
    -------
    None
        The function displays the plot, or saves it to the file specified 
        in `savefig`. It does not return a value.

    Notes
    -----
    This function allows for the dynamic creation of various types of 
    uncertainty plots. By specifying different `kind` values, 
    the user can create error bar plots, shaded line plots, box plots, 
    histograms, or even Monte Carlo simulation plots. This flexibility 
    makes it easy to visualize uncertainty and model prediction 
    intervals in a variety of contexts.

    The error bars are computed as:

    .. math::
        \text{yerr}_{\text{lower}} = Q_{50} - Q_{10}
        \text{yerr}_{\text{upper}} = Q_{90} - Q_{50}
    
    For the Monte Carlo simulations, the random data points are 
    generated from a normal distribution:

    .. math::
        X \sim \mathcal{N}(\mu, \sigma^2)
    
    where \(\mu\) is the mean of the `q50` and \(\sigma\) is the standard 
    deviation of the `q50`.

    Examples
    --------
    >>> from gofast.core.generic import plot_with_uncertainty
    >>> plot_with_uncertainty(df, q_cols=["q10", "q50", "q90"], 
                              dt_col="date", kind="error_bars")

    See Also
    --------
    _plot_error_bars
    _plot_line_shaded
    _plot_box_plot
    _plot_violin_plot
    _plot_histogram
    _plot_fan_chart
    _plot_kde
    _plot_monte_carlo

    References
    ----------
    [1] *Statistical Methods for the Prediction of Time Series*,
        Author Name, Journal, Year.
    """

    # Check if dt_col is provided, if not, find a datetime column or use index
    df_copy = df.copy () 
    if dt_col is None:
        # Try to find a datetime column automatically
        dt_cols = select_dtypes (
            df, incl=['datetime64', 'timedelta64'], 
            return_columns=True 
            )
        # dt_cols = df.select_dtypes(include=['datetime64', 'timedelta64']).columns
        if len(dt_cols) > 0:
            dt_col = dt_cols[0]  # Use the first datetime-like column
        elif isinstance(df.index, pd.DatetimeIndex):
            # Use index if it's datetime
            dt_col = df.index.name if df.index.name else "index"  
        else:
            # If no datetime column, fallback to numeric index
            dt_col = df.index.name if df.index.name else "index"
            # and reset the dataframe 
            df_copy.reset_index (drop=False, inplace=True ) 
            
    q_cols = columns_manager(q_cols, empty_as_none= True )
    # Check if the necessary columns exist
    exist_features (df_copy , features = [dt_col] + q_cols)

    if not q_cols:
        raise ValueError(f"'{q_cols}' not found in the DataFrame.")
            
    # Ensure the datetime column is in datetime format
    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.sort_values(by=dt_col)
    # Set the figure size
    if ax is None: 
        fig, ax = plt.subplots(figsize=figsize)  

    # Plot based on the specified plot type
    kind = str(kind).replace (
        'plot', '').replace (
            '_bars', 's'
            ).replace ('_plot', '')
    plot_helpers = {
        "errors": _plot_error_bars,
        "line_shaded": _plot_line_shaded,
        "box": _plot_box_plot,
        "violin": _plot_violin_plot,
        "histogram": _plot_histogram,
        "fan_chart": _plot_fan_chart,
        "kde": _plot_kde,
        "monte_carlo": _plot_monte_carlo
    }

    # Call the corresponding helper function
    if kind in plot_helpers:
        plot_helpers[kind](
            df, dt_col=dt_col, 
            q_cols=q_cols, 
            alpha=alpha, 
            n_simulations=n_simulations, 
            bins=bins, 
            capsize=capsize, 
            title =title, 
            xlabel=xlabel, 
            ylabel=ylabel, 
            line_color="blue", 
            fill_color='lightblue', 
            label_median=None, 
            label_shaded=None, #"q10 to q90 range"
            palette ="Set2", 
            ax=ax, 
            notch=False, 
            var_name=None, 
            value_name=None, 
            width=.8, 
            scale='area', 
            split=split, 
            color1=None, 
            color2=None, 
            **kws
            )

    # Customize grid if enabled
    if show_grid:
        grid_props = grid_props or {"linestyle": ":", 'alpha': 0.7}
        ax.grid(True, **grid_props)
    else:
        ax.grid(False)

    # Display the plot
    plt.xticks(rotation=rotation)
    plt.tight_layout()

    if savefig:
        plt.savefig(savefig)
    else:
        plt.show()


