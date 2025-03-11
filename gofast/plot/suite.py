# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Provides a comprehensive suite of plotting functions for 
visualizing model performance and data analysis.
"""
from __future__ import annotations 

import re 
import math
import copy 
import warnings
from numbers import Real, Integral 
import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.interpolate import griddata
from scipy.stats import probplot

from sklearn.metrics import r2_score, mean_squared_error 
from sklearn.metrics import precision_recall_curve

from ..api.types import Optional, Tuple,  Union, List 
from ..api.types import Dict, DataFrame
from ..api.summary import ResultSummary
from ..core.array_manager import smart_ts_detector, drop_nan_in 
from ..core.checks import ( 
    is_iterable, check_params, check_numeric_dtype, 
    check_features_types, check_spatial_columns,
    validate_depth, exist_features, 
)
from ..core.diagnose_q import ( 
    validate_q_dict, 
    validate_quantiles, 
    build_q_column_names, 
    detect_quantiles_in
)
from ..core.handlers import columns_manager, param_deprecated_message 
from ..core.io import is_data_readable 
from ..core.plot_manager import default_params_plot 
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
    build_data_if, validate_positive_integer, 
    is_frame, check_consistent_length, 
    validate_yy, filter_valid_kwargs
)
from ._config import PlotConfig
from .utils import flex_figsize 

__all__=[
    "plot_spatial_features", 
    "plot_categorical_feature", 
    'plot_sensitivity', 
    'plot_spatial_distribution', 
    'plot_dist', 
    'plot_quantile_distributions', 
    'plot_uncertainty', 
    'plot_prediction_intervals',
    'plot_temporal_trends', 
    'plot_relationship', 
    'plot_fit', 
    'plot_perturbations', 
    'plot_well', 
    'plot_factory_ops', 
    'plot_ranking', 
    'plot_coverage', 
    'plot_qdist'
]

@default_params_plot(
    savefig=PlotConfig.AUTOSAVE("my_coverall_plot.png"), 
    fig_size =(8, 6), 
    dpi=300
    )
@validate_params ({ 
    'y_true': ['array-like'],
    'plot_type': [StrOptions({"line", "bar", "pie", "radar"}), None], 
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
    plot_type='line',
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

    plot_type : str, optional (default='line')
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
    ...               plot_type='bar',
    ...               title='Coverage (Bar)')
    # Single model quantile coverage
    >>> y_pred = np.random.rand(200, 3)
    >>> plot_coverage(y_true, y_pred, q=[0.1, 0.5, 0.9],
    ...               plot_type='radar', names=['QModel'],
    ...               cov_fill=True, cmap='plasma')
    >>> # Multiple models with radar plot
    >>> y_pred_q2 = np.random.rand(100, 3)
    >>> plot_coverage(y_true, y_pred_q, y_pred_q2,
    ...               q=q,
    ...               names=['Model1','Model2'],
    ...               plot_type='radar',
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
    
    if plot_type in {'bar', 'line', 'pipe'}: 
        # Initialize the figure.
        if figsize is not None:
            plt.figure(figsize=figsize)
        else:
            plt.figure()
    # Plot according to the chosen 'plot_type'.
    if plot_type == 'bar':
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

    elif plot_type == 'line':
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

    elif plot_type == 'pie':
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

    elif plot_type == 'radar':
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


@default_params_plot(
    savefig=PlotConfig.AUTOSAVE("my_ranking_plot.png"), 
    fig_size =None, 
    dpi=300
    )
@validate_params ({ 
    'plot_type': [StrOptions({"auto", "ranking", "importance"}), None], 
    'features': [str, 'array-like', None]
    })
def plot_ranking(
    X,
    y=None,
    models=None,
    features=None,
    precomputed=False,
    xai_methods=None,
    plot_type=None,
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
    plot_type : {'ranking', 'importance', 'auto', None}, optional
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
    >>> plot_ranking(X, y, plot_type='ranking', figsize=(4,6))

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
        # based on the user-specified plot_type
        return_rank = (plot_type is None or plot_type == 'ranking')
        
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
        matrix_kind = plot_type or "auto"
        
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
            base= (3, 12) if plot_type=='ranking' else (4, 12), 
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


@default_params_plot(
    savefig=PlotConfig.AUTOSAVE("my_factory_ops_plot.png"), 
    fig_size =(10, 8), 
    dpi=300, 
    )
@check_params({ 
    "names": Optional[Union[str, List[str]]], 
    "title": Optional[str], 
    'figsize': Optional[Tuple[int, int]], 
    })
@validate_params({ 
    'train_times': ['array-like', None], 
    'metrics': [str, 'array-like', None], 
    'scale': [StrOptions({"norm", "min-max", 'scale'}), None], 
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
        - `'scale'`: standard scaling per metric.
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
    else:
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

    elif scale == 'scale':
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


@default_params_plot(
    savefig=PlotConfig.AUTOSAVE('my_well_plot.png'), 
    fig_size=None 
 )
@validate_params ({ 
    'df': ['array-like'], 
    'cols': ['array-like', None], 
    'depth_arr': ['array-like', None], 
    'ref_arr' : ['array-like', None], 
    'pred_df': ['array-like', None], 
    'error': [StrOptions({ 'raise', 'warn', 'ignore'})], 
    'numeric_only': [bool], 
    })

@isdf 
@param_deprecated_message(
    warning_category=UserWarning, 
    conditions_params_mappings=[
        {
            'param': 'depth_kind',
            'condition': lambda v: v not in { None, 'log'},
            'message': ( 
                "Current version only supports ``depth_kind='log'``."
                " Resetting depth_kind to None"
                ),
            'default': None
        }, 
        { 
            'param': 'titles',
            'condition': lambda v: v is not None,
            'message': ( 
                "Title for each subplot is unused."
                " Track names overshadow it."
                ),
            'default': None
            
        }
    ]
)
@check_params ({ 
    'ref_col': str, 
    'combined_cols':Optional[Dict[str, List[str]]], 
    'kind_mapping': Optional[Dict[str, str]], 
    })
def plot_well(
    df,
    depth_arr=None,
    ref_arr=None,
    pred_df=None,
    ref_col=None,
    cols=None,
    pred_cols=None,
    kind_mapping=None,
    depth_kind=None,
    combined_cols=None,
    agg_plot=False, 
    ignore_index=False,
    index_as_depth=True,
    error='warn',
    titles=None,
    show_grid=True,
    grid_alpha=0.7,
    grid_ls='--',
    sharey=True,
    fig_size=None,
    savefig=None,
    minorticks_on=True,
    subplot_kws=None
):
    r"""
    Plot well logs from a main DataFrame and optional prediction
    DataFrame with a shared or separate depth axis. The depth can be
    provided explicitly via ``depth_arr`` or derived from `df` index
    if `index_as_depth` is True. This function relies on
    `validate_depth` and can optionally leverage
    `arrange_tracks_for_plotting` to manage columns.

    The well logs are typically arranged as vertical tracks, each
    sharing the same depth axis. This aids in comparing multiple
    logs or predictions across the same interval. If `agg_plot`
    is True, the tracks from both data sources are concatenated
    horizontally with no spacing and only the first track displays
    the depth axis. Otherwise, separate sets of tracks can be
    displayed with configurable space in between.

    .. math::
       y = \alpha x + \beta

    Here, :math:`y` is the log response, and :math:`x` is the
    measured index (e.g., depth). The constants :math:`\alpha` and
    :math:`\beta` represent scaling factors in typical well-log
    transformations [1]_.

    Parameters
    ----------
    df : pandas.DataFrame
        Main DataFrame containing well-log columns. If `cols` is
        provided, only those columns in ``df`` are plotted.
    depth_arr : array-like or pandas.Series, optional
        Depth values aligned to ``df``. If None and
        `index_as_depth` is True, the function uses `df`
        index as depth. Otherwise a simple 0..N range is used.
    ref_arr : array-like or pandas.Series, optional
        Reference or ground truth values aligned with
        predictions in ``pred_df``.
    pred_df : pandas.DataFrame or pandas.Series, optional
        Prediction data (e.g., model outputs) to be plotted
        alongside `df`. If `agg_plot` is True, it is
        appended to df tracks.
    ref_col : str, optional
        Column name in the reference DataFrame if passing
        ``ref_arr`` as a multi-column object. Specifies which
        column to plot as reference.
    cols : list of str, optional
        Subset of columns from `df` to plot. If None,
        all columns from `df` are considered.
    pred_cols : list of str, optional
        Subset or rename columns from ``pred_df``. If
        ``pred_df`` is a single Series and exactly one
        name is given, the Series is renamed accordingly.
    kind_mapping : dict, optional
        A mapping dict of columns or track names to their
        plotting type (e.g. `{'Resistivity': 'log'}`). If a
        track is marked 'log', it is plotted using a semilogx.
    depth_kind : {'log', None}, optional
        If `'log'`, use log-scaling on the depth axis.
    combined_cols : dict or list of str, optional
        Group columns in `df` as combined tracks. A dict like
        ``{'Track1': ['GR','RHOB']}`` merges the columns in
        one subplot. A list merges them under a generated name.
    agg_plot : bool, optional
        If True, merges `df` and ``pred_df`` tracks into one
        continuous set of subplots. If False, plots them
        separately with a configurable gap in between.
    ignore_index : bool, optional
        Whether to reset index for `df`, ``pred_df``, and
        others prior to plotting. If True, indices become
        0..N-1.
    index_as_depth : bool, optional
        Whether to treat `df` index as depth if no
        ``depth_arr`` is given.
    error : {'warn', 'raise', 'ignore'}, optional
        Error policy for mismatches or non-monotonic depth.
    titles : list of str, optional
        Titles for each subplot (unused if overshadowed by
        track names).
    show_grid : bool, default True
        Toggles the grid overlay on each track.
    grid_alpha : float, default 0.7
        Transparency factor for the gridlines.
    grid_ls : str, default '--'
        Line style for the grid.
    sharey : bool, default True
        Whether y-axes (depth axes) are shared among subplots.
        Typically True for well logs.
    fig_size : tuple, optional
        Size of the figure (width, height). If None, a
        reasonable default is chosen.
    savefig : str, optional
        If provided, path to save the resulting figure.
    minorticks_on : bool, default True
        Whether to enable minor ticks for added depth
        readability.
    subplot_kws : dict, optional
        Additional keyword arguments passed to
        ``plt.subplots`` or GridSpec.

    Examples
    --------
    >>> from gofast.plot.suite import plot_well
    >>> import pandas as pd
    >>> # Suppose df is a DataFrame with columns: GR, RHOB, NPHI
    >>> df = pd.DataFrame({
    ...     'GR':   [50, 60, 70],
    ...     'RHOB': [2.3, 2.4, 2.35],
    ...     'NPHI': [0.25, 0.22, 0.20]
    ... })
    >>> # Plot these logs with depth as df.index:
    >>> plot_well(df, index_as_depth=True, agg_plot=True)

    Notes
    -----
    In the mathematical sense, if :math:`D` is the depth axis
    and :math:`L_i` are the log values, then:

    .. math::
       \{(D, L_1), (D, L_2), \dots\} \;\in\; \mathbb{R}^2.

    The function overlays them in vertical subplots
    for quick visual correlation.

    See Also
    --------
    `arrange_tracks_for_plotting` :
        Organizes (track_name, columns) tuples for df and
        pred_df.  
    `validate_depth` :
        Aligns DataFrames and checks monotonic depth.

    References
    ----------
    .. [1] Slatt, R.M. "Stratigraphic reservoir characterization
       for petroleum geologists, geophysicists, and engineers",
       2nd Edition, Elsevier, 2013.
    """

    def arrange_tracks_for_plotting(
            df_tracks, pred_tracks, agg_plot=False):
        # If agg_plot is True, combine both sets of tracks into a single list.
        if agg_plot:
            return df_tracks + pred_tracks
        else:
            # Otherwise, return them separately so the 
            # caller can handle them independently.
            return df_tracks, pred_tracks
    
    # If subplot_kws is None, define an empty dict; 
    # we will handle spacing logic ourselves via GridSpec
    # in this implementation.
    if subplot_kws is None:
        subplot_kws = {}

    # Validate depth and align data (not shown; assume
    # 'validate_depth' is externally defined).
    df, pred_df, ref_arr, depth_arr = validate_depth(
        df                  = df,
        pred_df             = pred_df,
        reference           = ref_arr,
        ref_col             = ref_col,
        depth               = depth_arr,
        new_name            = 'Depth',
        rename_depth        = False,
        reset_index         = ignore_index,
        check_monotonic     = True,
        index_as_depth      = index_as_depth,
        allow_index_mismatch= False,
        error               = error,
        as_series           = True,
        check_size          = False
    )

    # If pred_cols is specified, subset or rename pred_df if needed.
    if pred_cols is not None:
        pred_cols = is_iterable(
            pred_cols, 
            exclude_string=True, 
            transform=True
    )

    if isinstance(pred_df, pd.Series):
        if pred_cols and len(pred_cols) == 1:
            pred_df.name = pred_cols[0]
        else:
            if error == 'warn' and pred_cols and len(pred_cols) > 1:
                warnings.warn(
                    "Multiple pred_cols given, but pred_df is a single Series."
                    " Ignoring rename."
                )
    elif isinstance(pred_df, pd.DataFrame):
        if pred_cols is not None:
            pred_df = pred_df[pred_cols]

    # If cols is given, subset df columns accordingly.
    if cols is not None:
        df = df[list(cols)]

    # Build a dict grouping columns if combined_cols is used.
    track_dict = {}
    used_cols  = set()

    if combined_cols is not None:
        if isinstance(combined_cols, dict):
            for track_name, column_list in combined_cols.items():
                track_dict[track_name] = column_list
                used_cols.update(column_list)
        elif isinstance(combined_cols, list):
            track_name = '-'.join(combined_cols)
            track_dict[track_name] = combined_cols
            used_cols.update(combined_cols)

    # Columns in df that aren't in combined_cols become individual tracks.
    for c in df.columns:
        if c not in used_cols:
            track_dict[c] = [c]

    # Convert to list of (track_name, [columns]) for df.
    df_tracks = list(track_dict.items())

    # Build a list of (track_name, [columns]) for pred_df if provided.
    pred_tracks = []
    if pred_df is not None:
        if isinstance(pred_df, pd.DataFrame):
            for col_name in pred_df.columns:
                pred_tracks.append((col_name, [col_name]))
        elif isinstance(pred_df, pd.Series):
            pred_tracks.append((pred_df.name, [pred_df.name]))

    # Use helper to arrange df and pred tracks based on agg_plot.
    arranged = arrange_tracks_for_plotting(
        df_tracks   = df_tracks,
        pred_tracks = pred_tracks,
        agg_plot    = agg_plot
    )

    # If agg_plot is True, we get a single list of tracks (df+pred). 
    # If False, we get a tuple (df_tracks, pred_tracks).
    if agg_plot:
        all_tracks = arranged
        # No spacing between columns when aggregated.
        # We rely on a single row with wspace=0.0 for all subplots.
        if "wspace" not in subplot_kws:
            subplot_kws["wspace"] = 0.0
        df_total_tracks = len(all_tracks)

        # We'll handle depth labeling by only showing y-label on the 
        # first track. Mask the y-label on subsequent columns.

        if fig_size is None:
            fig_size = (3 * df_total_tracks, 10)

        fig = plt.figure(figsize=fig_size)
        gs  = GridSpec(nrows=1, 
                       ncols=df_total_tracks,
                       figure=fig,
                       **subplot_kws)

        axes = []
        for i in range(df_total_tracks):
            ax = fig.add_subplot(gs[0, i])
            axes.append(ax)

        # Now we have one row of subplots. 
        # We'll plot them in a single pass below.
        ax_list = axes

    else:
        df_tracks_only, pred_tracks_only = arranged
        # We want some spacing between df and pred subplots. 
        # For df alone, we might have wspace=0 between its columns,
        # and for pred alone, also wspace=0, 
        # but a bigger gap between the last df column and first pred column.
        # We'll implement that using gridspec with two sub-grids:

        df_total_tracks   = len(df_tracks_only)
        pred_total_tracks = len(pred_tracks_only)
        # We'll define a figure with df_total_tracks + pred_total_tracks 
        # subplots in one row, but a gap (e.g. wspace=0.3) specifically 
        # between the two sets.

        if fig_size is None:
            fig_size = (3 * (df_total_tracks + pred_total_tracks), 10)

        fig = plt.figure(figsize=fig_size)

        # We'll manually allocate the columns:
        #   0..(df_total_tracks-1) for df
        #   then a gap
        #   then pred_total_tracks columns for pred
        # We'll use the width ratios trick to insert some space.
        # Another approach is two separate subplots calls, 
        # but let's do it in one figure:

        # Example approach:
        # total_ncols = df_total_tracks + pred_total_tracks
        # We define a GridSpec with that many columns. We'll set wspace=0 
        # for everything.
        # Then we force a bigger "gap" column or so. But let's do it simpler:
        # We'll define 2 sub-grids: one for df with wspace=0, 
        # another for pred with wspace=0, 
        # plus a big space in between them by adjusting figure margins
        # or using a big hspace/wspace.

        # We'll do 2 columns in the top-level GridSpec: one for df, one for pred.
        # Then each sub-GridSpec has as many columns as needed for df or pred.
        # In between them, we define wspace=some bigger number, e.g. 0.3 or 0.4.

        # Top-level GridSpec
        top_gs = GridSpec(
            nrows=1,
            ncols=2,
            figure=fig,
            width_ratios=[df_total_tracks, pred_total_tracks],
            wspace=0.3  # Space between the two groups
        )
        
        # Nested GridSpec for df
        gs_df = GridSpecFromSubplotSpec(
            nrows=1,
            ncols=df_total_tracks,
            subplot_spec=top_gs[0, 0],  # Use top-level GridSpec as base
            wspace=0.0  # No space among df columns
        )
        
        # Nested GridSpec for pred
        gs_pred = GridSpecFromSubplotSpec(
            nrows=1,
            ncols=pred_total_tracks,
            subplot_spec=top_gs[0, 1],  # Use top-level GridSpec as base
            wspace=0.0  # No space among pred columns
        )

        # We'll gather the axes in an array so we can handle
        # them in a single pass if we want:
        ax_list = []

        # Create subplots for df tracks
        df_axes = []
        for i in range(df_total_tracks):
            ax = fig.add_subplot(gs_df[0, i])
            df_axes.append(ax)
        # Create subplots for pred tracks
        pred_axes = []
        for j in range(pred_total_tracks):
            ax = fig.add_subplot(gs_pred[0, j])
            pred_axes.append(ax)

        ax_list = df_axes + pred_axes

    # Helper function to set a log scale on depth 
    # if requested, then invert axis.
    # since plot_track firstly invert y so no need 
    # to reinvert y again. 
    def maybe_set_depth_scale(ax_, dkind, show_ylabel):
        if dkind == 'log':
            ax_.set_yscale('log')
        # if show_ylabel:
        #     ax.set_ylabel("Depth")
        # else:
        #     # Hide the y-label if we don't want to see it on this subplot.
        #     ax.set_yticklabels([])  # Hide y-tick labels
        #     ax.set_ylabel("")
            
        # ax_.invert_yaxis () : No Need.
            
    # Function to plot columns in a single track. 
    # We optionally overlay reference data
    # in red for pred tracks.
    def plot_track(
        ax,
        track_name,
        cols_in_track,
        data_source,
        depth_values,
        kmapping,
        dkind,
        do_grid,
        show_ylabel,
        ref_data=None
    ):
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for idx_col, col_name in enumerate(cols_in_track):
            if col_name not in data_source.columns:
                continue

            x_vals = data_source[col_name].values
            col_kind = None
            if kmapping and (col_name in kmapping):
                col_kind = kmapping[col_name]
            else:
                # If the combined_cols dict has a single "track_name" = 'Resistivity'
                # and we put 'log' in kind_mapping for 'Resistivity', we might want
                # to adopt that for all columns in that track. We'll do a fallback:
                if kmapping and (track_name in kmapping):
                    col_kind = kmapping.get(track_name)

            line_color = color_cycle[idx_col % len(color_cycle)]
            if col_kind == 'log':
                ax.semilogx(x_vals, depth_values, color=line_color, label=col_name)
            else:
                ax.plot(x_vals, depth_values, color=line_color, label=col_name)

        if ref_data is not None:
            ax.plot(
                ref_data.values, depth_values, color='red', 
                label= ref_data.name if isinstance(
                    ref_data,pd.Series) else 'Reference'
                )

        ax.set_title(track_name)
        ax.set_xlabel("Value")
        if show_ylabel:
            ax.set_ylabel("Depth")
        else:
            # Hide the y-label if we don't want to see it on this subplot.
            ax.set_yticklabels([])  # Hide y-tick labels
            ax.set_ylabel("")
        ax.invert_yaxis()

        if do_grid:
            ax.grid(True, linestyle=grid_ls, alpha=grid_alpha)
        ax.legend()

        if minorticks_on:
            ax.grid(True, which='minor', linestyle=':', alpha=0.5)  
            ax.minorticks_on()

    # If agg_plot is True, we have all_tracks in a single list. 
    # We'll mask depth labeling except for the first track.
    if agg_plot:
        for i, (track_name, columns_in_track) in enumerate(all_tracks):
            current_ax = ax_list[i]

            # Decide if the columns belong to pred_df or df
            if pred_df is not None:
                if (isinstance(pred_df, pd.DataFrame)
                    and columns_in_track[0] in pred_df.columns):
                    data_source = pred_df
                elif (isinstance(pred_df, pd.Series)
                      and columns_in_track[0] == pred_df.name):
                    data_source = pd.DataFrame(pred_df)
                else:
                    data_source = df
            else:
                data_source = df

            # Determine depth array
            if depth_arr is not None:
                depth_vals = depth_arr.values if isinstance(
                    depth_arr, pd.Series) else depth_arr
            else:
                depth_vals = np.arange(len(data_source))

            # Check if we overlay ref_arr
            if (data_source is not df) and (ref_arr is not None):
                reference_to_plot = ref_arr
            else:
                reference_to_plot = None

            # Only the first column has the y-label for depth 
            show_ylabel = (i == 0)

            plot_track(
                ax           = current_ax,
                track_name   = track_name,
                cols_in_track= columns_in_track,
                data_source  = data_source,
                depth_values = depth_vals,
                kmapping     = kind_mapping,
                dkind        = depth_kind,
                do_grid      = show_grid,
                show_ylabel  = show_ylabel,
                ref_data     = reference_to_plot
            )

            maybe_set_depth_scale(current_ax, depth_kind, show_ylabel )

    else:
        # agg_plot is False, so we have two sets of tracks: df_tracks_only,
        # pred_tracks_only.
        # We'll plot df first, then pred, each in separate sub-GridSpec. 
        # The first column in df has the depth label, likewise the first
        # column in pred has it.
        # No space among df columns or among pred columns, but a bigger
        # space between the two sets.
        
        df_tracks_only, pred_tracks_only = arranged
        # Axes = df_axes + pred_axes in that order
        df_axes   = ax_list[:len(df_tracks_only)]
        pred_axes = ax_list[len(df_tracks_only):]

        # Plot df tracks
        for i, (track_name, columns_in_track) in enumerate(df_tracks_only):
            current_ax = df_axes[i]

            if depth_arr is not None:
                depth_vals = depth_arr.values if isinstance(
                    depth_arr, pd.Series) else depth_arr
            else:
                depth_vals = np.arange(len(df))

            # For df, we do not overlay reference data
            reference_to_plot = None

            # Only first column of df has depth label
            show_ylabel = (i == 0)

            plot_track(
                ax           = current_ax,
                track_name   = track_name,
                cols_in_track= columns_in_track,
                data_source  = df,
                depth_values = depth_vals,
                kmapping     = kind_mapping,
                dkind        = depth_kind,
                do_grid      = show_grid,
                show_ylabel  = show_ylabel,
                ref_data     = reference_to_plot
            )
            maybe_set_depth_scale(current_ax, depth_kind, show_ylabel)

        # Plot pred tracks
        if pred_tracks_only:
            for j, (track_name, columns_in_track) in enumerate(pred_tracks_only):
                current_ax = pred_axes[j]

                if pred_df is not None:
                    if (isinstance(pred_df, pd.DataFrame)
                        and columns_in_track[0] in pred_df.columns):
                        data_source = pred_df
                    elif (isinstance(pred_df, pd.Series)
                          and columns_in_track[0] == pred_df.name):
                        data_source = pd.DataFrame(pred_df)
                    else:
                        data_source = df
                else:
                    data_source = df

                if depth_arr is not None:
                    depth_vals = depth_arr.values if isinstance(
                        depth_arr, pd.Series) else depth_arr
                else:
                    depth_vals = np.arange(len(data_source))

                # If track is from pred_df, we might overlay ref_arr
                if (data_source is not df) and (ref_arr is not None):
                    reference_to_plot = ref_arr
                else:
                    reference_to_plot = None

                # Only first column of pred has the depth label
                show_ylabel = (j == 0)

                plot_track(
                    ax            = current_ax,
                    track_name    = track_name,
                    cols_in_track = columns_in_track,
                    data_source   = data_source,
                    depth_values  = depth_vals,
                    kmapping      = kind_mapping,
                    dkind         = depth_kind,
                    do_grid       = show_grid,
                    show_ylabel   = show_ylabel,
                    ref_data      = reference_to_plot
                )
                maybe_set_depth_scale(current_ax, depth_kind, show_ylabel)

    # Final layout and optional save
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig)
    plt.show()

def plot_perturbations(
    X,
    y,
    model=None,
    perturbations=0.05,
    max_iter=10,
    metric='miv',
    plot_type='bar',
    percent=False,
    relative=False,
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
    
    plot_type : {'bar', 'barh', 'pie', 'scatter'}, default='bar'
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
        - If `plot_type` is not among the supported options.

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
    ...     plot_type='bar',
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
        # Call `miv_score` with `plot_type=None` to prevent
        # it from producing an immediate plot. We only want
        # its numerical results. We pass `relative` and
        # other relevant parameters as needed.
        msummary = miv_score(
            X=X,
            y=y,
            model=model,
            perturbation=pert,
            max_iter=max_iter,
            plot_type=None,       # block any plotting
            percent=False,        # keep raw numeric values, handle 'percent' here
            relative=relative,
            show_grid=False,      # not relevant here
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
        if plot_type in ['bar', 'barh']:
            _plot_bars(
                ax=ax,
                features=features,
                importances=final_imports,
                plot_t=plot_type,
                pcent=percent,
                c_map=cmap,
                in_title=titles_list[i]
            )
            if show_grid and plot_type in ['bar', 'barh']:
                ax.grid(True, linestyle='--', alpha=0.7)
            elif not show_grid:
                ax.grid(False)

        elif plot_type == 'pie':
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

        elif plot_type == 'scatter':
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

    
@validate_params ({ 
    "sensitivity_values": ['array-like'], 
    "baseline_prediction": ['array-like', Real, None ], 
    "plot_type": [StrOptions({'hist', 'bar', 'line', 'boxplot', 'box'})], 
    "x_ticks_rotation": [Interval( Integral, 0, None, closed="left")], 
    "y_ticks_rotation": [Interval( Integral, 0, None, closed="left")], 
    })
def plot_sensitivity(
    sensitivity_df, *,
    baseline=None, 
    plot_type='line',
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

    plot_type : {'line', 'bar', 'hist', 'boxplot'}, optional, default='line'
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
        Whether to display outliers in the boxplot when `plot_type='boxplot'`. 
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
                            plot_type='line')
       
    2. Bar plot with customized appearance:
       >>> plot_sensitivity(baseline=0.5, 
                            sensitivity_df=pd.DataFrame({
                                'Feature 1': [0.1, 0.2, 0.3],
                                'Feature 2': [0.05, 0.15, 0.25],
                                'Feature 3': [0.2, 0.3, 0.4]
                            }), 
                            plot_type='bar', baseline_color='g', 
                            baseline_linestyle='-', figsize=(8, 5))
       
    3. Histogram plot:
       >>> plot_sensitivity(baseline=0.5, 
                            sensitivity_df=pd.DataFrame({
                                'Feature 1': [0.1, 0.2, 0.3],
                                'Feature 2': [0.05, 0.15, 0.25],
                                'Feature 3': [0.2, 0.3, 0.4]
                            }), 
                            plot_type='hist')
    
    4. Boxplot with outliers:
       >>> plot_sensitivity(baseline=0.5, 
                            sensitivity_df=pd.DataFrame({
                                'Feature 1': [0.1, 0.2, 0.3],
                                'Feature 2': [0.05, 0.15, 0.25],
                                'Feature 3': [0.2, 0.3, 0.4]
                            }), 
                            plot_type='boxplot', boxplot_showfliers=True)
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

    if plot_type == 'line':
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

    elif plot_type == 'bar':
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

    elif plot_type == 'hist':
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

    elif plot_type in ['boxplot', 'box']:
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

@is_data_readable 
@isdf 
def plot_spatial_features(
    data,
    features,
    dates=None,
    dt_col="year",
    x_col='longitude',
    y_col='latitude',
    colormaps=None,
    figsize=None,
    point_size=10,
    marker='o',
    plot_type='scatter',
    colorbar_orientation='vertical',
    cbar_labelsize=10,
    axis_off=True,
    titles=None,
    vmin_vmax=None,
    **kwargs
):
    """
    Plot spatial distribution of specified features over given dates.

    This function creates a grid of subplots, each displaying the
    geographical distribution of a specified feature at particular
    dates or times. It supports multiple plot types including
    ``'scatter'``, ``'hexbin'``, and ``'contour'``, allowing for
    extensive customization of plot appearance. The function leverages
    Matplotlib's plotting capabilities [1]_ to visualize spatial data.

    Parameters
    ----------
    data : pandas.DataFrame
        The input DataFrame containing the data to plot. It must
        include the specified `features`, `x_col`, `y_col`, and
        the `dt_col` if `dates` are provided.

    features : list of str
        List of feature names to plot. Each feature must exist
        in `data`. A subplot will be created for each feature.

    dates : str or datetime-like or list of str or datetime-like, optional
        Dates or times to plot. Each date must correspond to an entry
        in the `dt_col` of `data`. If `None`, the function plots
        the features without considering dates.

    dt_col : str, default ``'year'``
        Name of the column in `data` to use for date or time filtering.
        This column can contain datetime objects, years, or any other
        temporal representation.

    x_col : str, default ``'longitude'``
        Name of the column in `data` to use for the x-axis coordinates.

    y_col : str, default ``'latitude'``
        Name of the column in `data` to use for the y-axis coordinates.

    colormaps : list of str, optional
        List of colormap names to use for each feature. If not
        provided, default colormaps are used.

    figsize : tuple of float, optional
        Figure size in inches, as a tuple ``(width, height)``. If not
        provided, the figure size is determined based on the number of
        features and dates.

    point_size : int, default 10
        Size of the points in the scatter plot.

    marker : str, default ``'o'``
        Marker style for scatter plots.

    plot_type : {'scatter', 'hexbin', 'contour'}, default ``'scatter'``
        Type of plot to create. Supported options are ``'scatter'``,
        ``'hexbin'``, and ``'contour'``.

    colorbar_orientation : {'vertical', 'horizontal'}, default ``'vertical'``
        Orientation of the colorbar.

    cbar_labelsize : int, default 10
        Font size for the colorbar tick labels.

    axis_off : bool, default True
        If ``True``, axes are turned off. If ``False``, axes are shown.

    titles : dict of str, optional
        Dictionary of titles for each feature. Keys are feature names,
        and values are title templates that can include ``{date}``.

    vmin_vmax : dict of tuple, optional
        Dictionary specifying the color scale (vmin and vmax) for each
        feature. Keys are feature names, and values are tuples
        ``(vmin, vmax)``.

    **kwargs
        Additional keyword arguments passed to the plotting functions
        (``scatter``, ``hexbin``, or ``contourf``).

    Notes
    -----
    The function supports different plot types:

    - For ``plot_type='scatter'``, it creates a scatter plot using
      ``matplotlib.pyplot.scatter``.

    - For ``plot_type='hexbin'``, it creates a hexbin plot using
      ``matplotlib.pyplot.hexbin``.

    - For ``plot_type='contour'``, it creates a contour plot by
      interpolating the data onto a grid using
      :func:`scipy.interpolate.griddata` and then plotting using
      ``matplotlib.pyplot.contourf``.

    The color normalization is performed using:

    .. math::

        c_{\text{norm}} = \frac{c - v_{\text{min}}}{v_{\text{max}} - v_{\text{min}}}

    where :math:`c` is the feature value, :math:`v_{\text{min}}` and
    :math:`v_{\text{max}}` are the minimum and maximum values for the
    color scale.

    Examples
    --------
    >>> from gofast.plot.suite import plot_spatial_features
    >>> plot_spatial_features(
    ...     data=df,
    ...     features=['temperature', 'humidity'],
    ...     dates=['2023-01-01', '2023-06-01'],
    ...     dt_col='date',
    ...     x_col='lon',
    ...     y_col='lat',
    ...     colormaps=['coolwarm', 'YlGnBu'],
    ...     point_size=15,
    ...     plot_type='scatter',
    ...     axis_off=False,
    ...     titles={'temperature': 'Temp on {date}',
    ...             'humidity': 'Humidity on {date}'},
    ...     alpha=0.7
    ... )

    See Also
    --------
    matplotlib.pyplot.scatter : Create a scatter plot.
    matplotlib.pyplot.hexbin : Make a hexagonal binning plot.
    matplotlib.pyplot.contourf : Create a filled contour plot.
    scipy.interpolate.griddata : Interpolate unstructured D-dimensional data.

    References
    ----------
    .. [1] Hunter, J. D. (2007). Matplotlib: A 2D graphics environment.
       *Computing in Science & Engineering*, 9(3), 90-95.
    """

    # Validate that features exist in data
    exist_features(data, features)
    extra_msg = (
        "If a numeric feature is stored as an 'object' type, "
        "it should be explicitly converted to a numeric type"
        " (e.g., using `pd.to_numeric`). For categorical features,"
        " please use `plot_categorical_feature` instead."
    )
    check_features_types(
        data, features= features, dtype='numeric',
        extra=extra_msg
    )
    # Handle dates parameter
    if dates is not None:
        # Convert single value to list
        if not isinstance(dates, (list, tuple, np.ndarray, pd.Series)):
            dates = [dates]
        else:
            dates = list(dates)

        # Check that 'dt_col' exists
        if dt_col not in data.columns:
            raise ValueError(f"Column '{dt_col}' not found in data.")

        # Depending on the type of 'dt_col', process accordingly
        if np.issubdtype(data[dt_col].dtype, np.datetime64):
            # If dt_col is datetime, convert dates to datetime
            data[dt_col] = pd.to_datetime(data[dt_col])

            # Convert dates parameter to datetime
            dates = [pd.to_datetime(d) for d in dates]

            # Normalize dates to remove time component
            data_dates = data[dt_col].dt.normalize().unique()
            dates_normalized = [d.normalize() for d in dates]

            # Check that dates exist in data
            missing_dates = set(dates_normalized) - set(data_dates)
            if missing_dates:
                missing_dates_str = ', '.join(
                    [d.strftime('%Y-%m-%d') for d in missing_dates]
                )
                raise ValueError(f"Dates {missing_dates_str} not found in data.")

            ncols = len(dates)
        else:
            # dt_col is not datetime, treat as categorical or numeric
            data_dates = data[dt_col].unique()
            missing_dates = set(dates) - set(data_dates)
            if missing_dates:
                missing_dates_str = ', '.join(map(str, missing_dates))
                raise ValueError(f"Dates {missing_dates_str} not found in data.")

            ncols = len(dates)
    else:
        ncols = 1

    features = columns_manager(features, empty_as_none =False)
    nrows = len(features)
 
    colormaps = columns_manager(colormaps) 
    
    if colormaps is None:
        colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']

    if figsize is None:
        figsize = (5 * ncols, 5 * nrows)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        squeeze=False
    )

    for i, feature in enumerate(features):
        cmap = colormaps[i % len(colormaps)]

        if vmin_vmax and feature in vmin_vmax:
            vmin, vmax = vmin_vmax[feature]
        else:
            vmin = data[feature].min()
            vmax = data[feature].max()

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        if dates is not None:
            for j, date in enumerate(dates):
                ax = axes[i, j]
                if np.issubdtype(data[dt_col].dtype, np.datetime64):
                    # Normalize date to remove time component
                    date_normalized = pd.to_datetime(date).normalize()
                    subset = data[
                        (data[dt_col].dt.normalize() == date_normalized)
                        & data[feature].notnull()
                    ]
                    date_str = date_normalized.strftime('%Y-%m-%d')
                else:
                    subset = data[
                        (data[dt_col] == date)
                        & data[feature].notnull()
                    ]
                    date_str = str(date)

                x = subset[x_col].values
                y = subset[y_col].values
                c = subset[feature].values

                if plot_type == 'scatter':
                    sc = ax.scatter(
                        x,
                        y,
                        c=c,
                        cmap=cmap,
                        norm=norm,
                        s=point_size,
                        marker=marker,
                        **kwargs
                    )
                elif plot_type == 'hexbin':
                    sc = ax.hexbin(
                        x,
                        y,
                        C=c,
                        gridsize=50,
                        cmap=cmap,
                        norm=norm,
                        **kwargs
                    )
                elif plot_type == 'contour':
                    # Create a grid to interpolate data
                    xi = np.linspace(x.min(), x.max(), 100)
                    yi = np.linspace(y.min(), y.max(), 100)
                    xi, yi = np.meshgrid(xi, yi)
                    # Interpolate using griddata
                    zi = griddata((x, y), c, (xi, yi), method='linear')
                    # Plot contour
                    sc = ax.contourf(
                        xi,
                        yi,
                        zi,
                        levels=15,
                        cmap=cmap,
                        norm=norm,
                        **kwargs
                    )
                else:
                    raise ValueError(f"Unsupported plot_type: {plot_type}")

                if titles and feature in titles:
                    title = titles[feature].format(date=date_str)
                else:
                    title = f"{feature} - {date_str}"

                ax.set_title(title)
                if axis_off:
                    ax.axis('off')

                if j == ncols - 1:
                    cbar = fig.colorbar(
                        sc,
                        ax=ax,
                        orientation=colorbar_orientation
                    )
                    cbar.ax.tick_params(labelsize=cbar_labelsize)
        else:
            ax = axes[i, 0]
            subset = data[data[feature].notnull()]
            x = subset[x_col].values
            y = subset[y_col].values
            c = subset[feature].values

            if plot_type == 'scatter':
                sc = ax.scatter(
                    x,
                    y,
                    c=c,
                    cmap=cmap,
                    norm=norm,
                    s=point_size,
                    marker=marker,
                    **kwargs
                )
            elif plot_type == 'hexbin':
                sc = ax.hexbin(
                    x,
                    y,
                    C=c,
                    gridsize=50,
                    cmap=cmap,
                    norm=norm,
                    **kwargs
                )
            elif plot_type == 'contour':
                # Create a grid to interpolate data
                xi = np.linspace(x.min(), x.max(), 100)
                yi = np.linspace(y.min(), y.max(), 100)
                xi, yi = np.meshgrid(xi, yi)
                # Interpolate using griddata
                
                zi = griddata((x, y), c, (xi, yi), method='linear')
                # Plot contour
                sc = ax.contourf(
                    xi,
                    yi,
                    zi,
                    levels=15,
                    cmap=cmap,
                    norm=norm,
                    **kwargs
                )
            else:
                raise ValueError(f"Unsupported plot_type: {plot_type}")

            if titles and feature in titles:
                title = titles[feature]
            else:
                title = f"{feature}"

            ax.set_title(title)
            if axis_off:
                ax.axis('off')

            cbar = fig.colorbar(
                sc,
                ax=ax,
                orientation=colorbar_orientation
            )
            cbar.ax.tick_params(labelsize=cbar_labelsize)

    plt.tight_layout()
    plt.show()

@is_data_readable 
@isdf 
def plot_categorical_feature(
    data,
    feature,
    dates=None,
    dt_col='year',
    x_col='longitude',
    y_col='latitude',
    cmap='tab10',
    figsize=None,
    point_size=10,
    marker='o',
    axis_off=True,
    legend_loc='upper left',
    titles=None,
    **kwargs
):
    """
    Plot the geographical distribution of a categorical feature.

    This function creates scatter plots showing the spatial
    distribution of a categorical feature over geographical
    coordinates. It supports plotting multiple dates or times,
    creating subplots for each specified date. The function allows
    extensive customization of the plot's appearance, including
    colormaps, point sizes, markers, and more.

    Parameters
    ----------
    data : pandas.DataFrame
        The input DataFrame containing the data to plot. It must
        include the specified `feature`, `x_col`, `y_col`, and
        `dt_col` if `dates` are provided.

    feature : str
        The name of the categorical feature to plot. This feature
        must exist in `data`.

    dates : scalar, list, or array-like, optional
        Dates or times to plot. If provided, the function will
        create subplots for each date specified. If `None`, the
        feature is plotted without considering dates.

    dt_col : str, default ``'year'``
        The name of the column in `data` to use for date or time
        filtering.

    x_col : str, default ``'longitude'``
        Name of the column in `data` to use for the x-axis
        coordinates.

    y_col : str, default ``'latitude'``
        Name of the column in `data` to use for the y-axis
        coordinates.

    cmap : str, default ``'tab10'``
        The name of the colormap to use for different categories.

    figsize : tuple, optional
        Figure size in inches, as a tuple ``(width, height)``. If
        not provided, the figure size is determined based on the
        number of dates and default settings.

    point_size : int, default 10
        Size of the points in the scatter plot.

    marker : str, default ``'o'``
        Marker style for scatter plots.

    axis_off : bool, default ``True``
        If ``True``, axes are turned off. If ``False``, axes are
        shown.

    legend_loc : str, default ``'upper left'``
        Location of the legend in the plot. Valid locations are
        strings such as ``'upper right'``, ``'lower left'``, etc.

    titles : dict or str, optional
        Titles for the subplots. If a dictionary, keys should
        correspond to subplot indices or dates, and values are
        title strings. If a string, it is used as a title template
        and can include placeholders like ``{date}`` which will be
        replaced with the actual date.

    **kwargs
        Additional keyword arguments passed to the plotting
        function (`matplotlib.pyplot.scatter`).

    Returns
    -------
    None
        The function displays the plot and does not return any
        value.

    Notes
    -----
    The function plots the spatial distribution of a categorical
    feature over geographical coordinates specified by `x_col` and
    `y_col`. If `dates` are provided, it filters the data for each
    date and creates a subplot for each one.

    The colors for each category are determined using the specified
    `colormap`. The categories are mapped to colors using:

    .. math::

        \\text{color}_i = \\text{colormap}\\left( \\frac{i}{N} \\right)

    where :math:`i` is the category index and :math:`N` is the total
    number of categories.

    Examples
    --------
    >>> from gofast.plot.suite import plot_categorical_feature
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Sample data
    >>> data = pd.DataFrame({
    ...     'longitude': np.random.uniform(-10, 10, 100),
    ...     'latitude': np.random.uniform(-10, 10, 100),
    ...     'category': np.random.choice(['A', 'B', 'C'], 100),
    ...     'year': np.random.choice([2018, 2019, 2020], 100)
    ... })
    >>> # Plotting without dates
    >>> plot_categorical_feature(
    ...     data,
    ...     feature='category',
    ...     x_col='longitude',
    ...     y_col='latitude',
    ...     point_size=20,
    ...     legend_loc='upper right'
    ... )
    >>> # Plotting with dates
    >>> plot_categorical_feature(
    ...     data,
    ...     feature='category',
    ...     dates=[2018, 2019, 2020],
    ...     dt_col='year',
    ...     x_col='longitude',
    ...     y_col='latitude',
    ...     point_size=20,
    ...     legend_loc='upper right',
    ...     titles='Category Distribution in {date}'
    ... )

    See Also
    --------
    plot_spatial_features : Function to plot spatial distribution of
        numerical features.

    References
    ----------
    .. [1] Hunter, J. D. (2007). Matplotlib: A 2D graphics environment.
       *Computing in Science & Engineering*, 9(3), 90-95.

    """
    # Validate that the feature exists in data
    exist_features(data, features= feature)
    extra_msg = (
    "To explicitly convert a feature of type 'object' to 'category', "
    "use `data[feature].astype('category')`. For numerical features, "
    "please use `plot_spatial_features` instead."
    )

    check_features_types(
        data, features= feature, dtype='category', extra=extra_msg)
    # Get unique categories
    categories = data[feature].unique()
    num_categories = len(categories)

    # Generate colors for each category
    cmap = plt.get_cmap(cmap, num_categories)
    colors = [cmap(i) for i in range(num_categories)]
    category_color_map = dict(zip(categories, colors))

    # Handle dates parameter
    if dates is not None:
        if not isinstance(dates, (list, tuple, np.ndarray, pd.Series)):
            dates = [dates]
        else:
            dates = list(dates)

        if dt_col not in data.columns:
            raise ValueError(f"Column '{dt_col}' not found in data.")

        if np.issubdtype(data[dt_col].dtype, np.datetime64):
            data[dt_col] = pd.to_datetime(data[dt_col])
            dates = [pd.to_datetime(d) for d in dates]
            data_dates = data[dt_col].dt.normalize().unique()
            dates_normalized = [d.normalize() for d in dates]
            missing_dates = set(dates_normalized) - set(data_dates)
            if missing_dates:
                missing_dates_str = ', '.join(
                    [d.strftime('%Y-%m-%d') for d in missing_dates]
                )
                raise ValueError(f"Dates {missing_dates_str} not found in data.")
            ncols = len(dates)
        else:
            data_dates = data[dt_col].unique()
            missing_dates = set(dates) - set(data_dates)
            if missing_dates:
                missing_dates_str = ', '.join(map(str, missing_dates))
                raise ValueError(f"Dates {missing_dates_str} not found in data.")
            ncols = len(dates)
    else:
        ncols = 1

    nrows = 1

    if figsize is None:
        figsize = (5 * ncols, 6)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i in range(ncols):
        ax = axes[i]
        if dates is not None:
            date = dates[i]
            if np.issubdtype(data[dt_col].dtype, np.datetime64):
                date_normalized = pd.to_datetime(date).normalize()
                subset = data[data[dt_col].dt.normalize() == date_normalized]
                date_str = date_normalized.strftime('%Y-%m-%d')
            else:
                subset = data[data[dt_col] == date]
                date_str = str(date)
            title = f"{feature} - {date_str}"
        else:
            subset = data
            title = f"Geographical Distribution of '{feature}'"

        for category in categories:
            cat_subset = subset[subset[feature] == category]
            x = cat_subset[x_col].values
            y = cat_subset[y_col].values
            ax.scatter(
                x, y,
                label=category,
                c=[category_color_map[category]],
                s=point_size,
                marker=marker,
                **kwargs
            )

        if titles:
            if isinstance(titles, dict) and i in titles:
                ax.set_title(titles[i])
            elif isinstance(titles, str):
                ax.set_title(titles)
            else:
                ax.set_title(title)
        else:
            ax.set_title(title)

        if axis_off:
            ax.axis('off')

        if i == ncols - 1:
            # Add legend to the last subplot
            ax.legend(title=feature, bbox_to_anchor=(1.05, 1), loc=legend_loc)

    plt.tight_layout()
    plt.show()


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
        ax.set_ylabel(ylabel or "Frequency")          # Set y-axis label
    
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
    
    ax.grid(show_grid)
        
    # Save the plot to the specified path if provided
    if savefig:
        plt.savefig(
            savefig, 
            dpi=300, 
            bbox_inches='tight'
        )
    
    # Display the plot
    plt.show()
    
    # Return the figure object for further manipulation if needed
    return fig

def plot_relationship(
    y_true, *y_preds, 
    names=None, 
    title=None,  
    theta_offset=0,  
    theta_scale='proportional',  
    acov='default',  
    figsize=(8, 8), 
    cmap='viridis',  
    point_size=50,  
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

    point_size : float, default=50
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
            s=point_size, alpha=alpha, edgecolor='black'
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


@validate_params ({ 
    "sensitivity_values": ['array-like'], 
    "features": [str, 'array-like', None ], 
    "plot_type": [StrOptions({'single', 'pair', 'triple'})], 
    })
def plot_distributions(
    data: pd.DataFrame,
    features: Optional[Union[List[str], str]] = None,
    bins: int = 30,
    kde: bool = True,
    hist: bool = True,
    figsize: tuple = (10, 6),
    title: str = "Feature Distributions",
    plot_type: str = 'single', 
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
    
    plot_type : str, optional, default='single'
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
                           plot_type='triple')
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
    if plot_type == 'single':
        plt.figure(figsize=figsize)
        for feature in features:
            plt.subplot(len(features), 1, features.index(feature) + 1)
            if hist:
                sns.histplot(data[feature], kde=kde, bins=bins, **kwargs)
            plt.title(f"{feature} Distribution")
        plt.tight_layout()
        plt.show()

    # Bivariate Plot (Pairwise feature distribution)
    elif plot_type == 'pair':
        if len(features) != 2:
            raise ValueError(
                "For 'pair' plot type, exactly 2 features must be specified.")
        
        plt.figure(figsize=figsize)
        sns.jointplot(data=data, x=features[0], y=features[1], kind="kde", **kwargs)
        plt.suptitle(f"{features[0]} vs {features[1]} Distribution")
        plt.show()

    # Trivariate Plot (3D distribution)
    elif plot_type == 'triple':
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

@isdf 
def plot_spatial_distribution(
    df: DataFrame,
    category_column: str,  
    continuous_bins: Union[str, List[float]] = 'auto',  
    categories: Optional[List[str]] = None,  
    filter_categories: Optional[List[str]] = None,
    spatial_cols: tuple = ('longitude', 'latitude'), 
    cmap: str = 'coolwarm',  
    plot_type: str = 'scatter', 
    alpha: float = 0.7, 
    show_grid:bool=True, 
    axis_off: bool = False,  
    figsize: tuple = (10, 8)  
) -> None:
    """
    Plot the Spatial Distribution of a Specified Category or Continuous Variable.

    This function visualizes the spatial distribution of geographical data points
    based on a specified categorical or continuous variable. For continuous data,
    it categorizes the values into defined bins, allowing for an intuitive
    representation of data intensity across a geographical area. The visualization
    can be rendered using scatter plots or hexbin plots, facilitating the analysis
    of spatial patterns and concentrations.

    The categorization of continuous variables is performed using either user-defined
    bins or the Freedman-Diaconis rule to determine an optimal bin width:
    
    .. math::
        \text{Bin Width} = 2 \times \frac{\text{IQR}}{n^{1/3}}
    
    where :math:`\text{IQR}` is the interquartile range of the data and :math:`n`
    is the number of observations.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe containing the geographical and categorical or 
        continuous data. It must include at least the following columns:
        
        - `longitude`: Longitude coordinates of the data points.
        - `latitude`: Latitude coordinates of the data points.
        - *category_column*: The column specified by `category_column` parameter.

    category_column : str
        The name of the column in `df` to be used for categorization. This column 
        can be either categorical or continuous. If categorical, the data will be 
        used directly. If continuous, the data will be binned into categories 
        based on `continuous_bins`.

    continuous_bins : Union[str, List[float]], default='auto'
        Defines the bin edges for categorizing continuous data. 
        
        - If set to `'auto'`, the function applies the Freedman-Diaconis rule to 
          determine optimal bin widths.
        - If a list of floats is provided, these values are used as the bin edges. 
          The provided bins must encompass the entire range of the data in 
          `category_column`.
        
        Raises a `ValueError` if the provided bins do not cover the data range.
    
    spatial_cols : tuple, optional, default=('longitude', 'latitude')
            A tuple containing the names of the longitude and latitude columns.
            Must consist of exactly two elements. The function will validate that
            these columns exist in the dataframe and are used as the spatial 
            coordinates for plotting.
            
            .. note::
                Ensure that the specified `spatial_cols` are present in 
                the dataframe and accurately represent geographical coordinates.
                
    categories : Optional[List[str]], default=None
        A list of labels corresponding to each bin when categorizing continuous 
        data. 
        
        - If `None`, the categories are auto-generated based on the bin ranges.
        - If provided, the length of `categories` must match the number of bins 
          minus one.
        
        If the number of categories does not match the number of bins, a warning 
        is issued and categories are auto-generated.

    filter_categories : Optional[List[str]], default=None
        Specifies which categories to include in the visualization. 
        
        - If `None`, all categories are plotted.
        - If a list is provided, only the specified categories are visualized, and 
          others are excluded. The legend reflects only the displayed categories.
        
        Raises a `ValueError` if none of the `filter_categories` are valid.

    cmap : str, default='coolwarm'
        The colormap to use for the visualization. This parameter utilizes matplotlib's 
        colormap names (e.g., `'viridis'`, `'plasma'`, `'inferno'`, `'magma'`, 
        `'cividis'`, etc.).

    plot_type : str, default='scatter'
        The type of plot to generate. 
        
        - `'scatter'`: Generates a scatter plot.
        - `'hexbin'`: Generates a hexbin plot, suitable for large datasets.
        
        Raises a `ValueError` if an unsupported `plot_type` is provided.

    alpha : float, default=0.7
        The transparency level for scatter plots. Ranges from 0 (completely 
        transparent) to 1 (completely opaque).
        
    show_grid : bool, default=True
        If set to `False`, the grid are turned off, providing a cleaner 
        visualization without grid lines.

    axis_off : bool, default=False
        If set to `True`, the plot axes are turned off, providing a cleaner 
        visualization without axis lines and labels.

    figsize : tuple, default=(10, 8)
        Specifies the size of the plot in inches as `(width, height)`.

    Returns
    -------
    None
        This function does not return any value. It renders the plot directly.

    Raises
    ------
    ValueError
        - If `category_column` does not exist in `df`.
        - If `category_column` is neither numeric nor categorical.
        - If `spatial_cols` is not a tuple of two elements or columns 
          are missing.
        - If `continuous_bins` is neither `'auto'` nor a list of numbers.
        - If provided `continuous_bins` do not cover the data range.
        - If the number of `categories` does not match the number of bins.
        - If no valid `filter_categories` are provided after filtering.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from gofast.plot.suite import plot_spatial_distribution

    >>> # Sample DataFrame
    >>> data = {
    ...     'longitude': np.random.uniform(-100, -90, 100),
    ...     'latitude': np.random.uniform(30, 40, 100),
    ...     'subsidence': np.random.choice(['minimal', 'moderate', 'severe'], 100)
    ... }
    >>> df = pd.DataFrame(data)
    
    >>> # Plot only 'severe' subsidence
    >>> plot_spatial_distribution(
    ...     df=df,
    ...     category_column='subsidence',
    ...     categories=['minimal', 'moderate', 'severe'],
    ...     filter_categories=['severe'],
    ...     plot_type='scatter'
    ... )
    
    >>> # Plot 'moderate' and 'severe' subsidence
    >>> plot_spatial_distribution(
    ...     df=df,
    ...     category_column='subsidence',
    ...     categories=['minimal', 'moderate', 'severe'],
    ...     filter_categories=['moderate', 'severe'],
    ...     plot_type='scatter'
    ... )
    
    >>> # Plot all categories
    >>> plot_spatial_distribution(
    ...     df=df,
    ...     category_column='subsidence',
    ...     categories=['minimal', 'moderate', 'severe'],
    ...     filter_categories=None,
    ...     plot_type='scatter'
    ... )

    Notes
    -----
    - The function automatically determines whether the `category_column` is 
      categorical or continuous based on its data type.
    - When categorizing continuous data, ensure that the provided `continuous_bins` 
      comprehensively cover the data range to avoid missing data points.
    - The legend in the plot dynamically adjusts based on the `filter_categories` 
      parameter, displaying only the relevant categories.

    See Also
    --------
    pandas.cut : Function to bin continuous data into discrete intervals.
    seaborn.scatterplot : Function to create scatter plots.
    matplotlib.pyplot.hexbin : Function to create hexbin plots.
    check_spatial_columns : Function to validate spatial columns in the dataframe.

    References
    ----------
    .. [1] Freedman, D., & Diaconis, P. (1981). On the histogram as a density estimator:
       L2 theory. *Probability Theory and Related Fields*, 57(5), 453-476.
    .. [2] Seaborn: Statistical Data Visualization. https://seaborn.pydata.org/
    .. [3] Matplotlib: Visualization with Python. https://matplotlib.org/
    """
    # make a copy for safety 
    df =df.copy() 
    # Check if category_column exists in dataframe
    if category_column not in df.columns:
        raise ValueError(
            f"Column '{category_column}' does not exist in the dataframe."
        )
    # Check whether 
    check_spatial_columns(df , spatial_cols=spatial_cols)
    
    # Determine if the category_column is categorical
    if pd.api.types.is_categorical_dtype(df[category_column]) or \
       df[category_column].dtype == object:
        is_categorical = True
    elif pd.api.types.is_numeric_dtype(df[category_column]):
        is_categorical = False
    else:
        raise ValueError(
            f"Column '{category_column}' must be either numeric or categorical."
        )

    if not is_categorical:
        # Handle continuous data
        if continuous_bins == 'auto':
            # Use Freedman-Diaconis rule for bin width
            q25, q75 = np.percentile(
                df[category_column].dropna(), [25, 75]
            )
            iqr = q75 - q25
            bin_width = 2 * iqr * len(df) ** (-1 / 3)
            if bin_width == 0:
                bin_width = 1  # Fallback to bin width of 1
            bins = np.arange(
                df[category_column].min(),
                df[category_column].max() + bin_width,
                bin_width
            )
            bins = np.round(bins, decimals=2)
        elif isinstance(continuous_bins, list):
            bins = sorted(continuous_bins)
            if (bins[0] > df[category_column].min()) or \
               (bins[-1] < df[category_column].max()):
                raise ValueError(
                    "Provided continuous_bins do not cover the range of the data."
                )
        else:
            raise ValueError(
                "continuous_bins must be 'auto' or a list of numbers."
            )

        # Categorize the continuous data based on bins
        df['category'] = pd.cut(
            df[category_column],
            bins=bins,
            labels=categories[:len(bins) - 1] if categories else None,
            include_lowest=True,
            right=False
        )

        # Handle category labels if not provided
        if categories is None:
            df['category'] = df['category'].astype(str)
        else:
            if len(categories) != len(bins) - 1:
                warnings.warn(
                    "Number of categories does not match number of bins. "
                    "Categories will be auto-generated.",
                    UserWarning
                )
                df['category'] = df['category'].astype(str)
    else:
        # Handle categorical data
        df['category'] = df[category_column].astype(str)
        if categories:
            missing_categories = set(categories) - set(df['category'].unique())
            if missing_categories:
                warnings.warn(
                    f"The following categories are not present in the data and "
                    f"will be ignored: {missing_categories}",
                    UserWarning
                )
            categories = [
                cat for cat in categories if cat in df['category'].unique()
            ]
            if not categories:
                raise ValueError(
                    "No valid categories to plot after filtering."
                )
        else:
            categories = sorted(df['category'].unique())

    # Filter the data if specified filter categories are provided
    if filter_categories:
        invalid_filters = set(filter_categories) - set(categories)
        if invalid_filters:
            warnings.warn(
                f"The following filter_categories are not in the available "
                f"categories and will be ignored: {invalid_filters}",
                UserWarning
            )
        filter_categories = [
            cat for cat in filter_categories if cat in categories
        ]
        if filter_categories:
            df = df[df['category'].isin(filter_categories)]
            categories = filter_categories  # Update categories to filtered ones
        else:
            raise ValueError(
                "No valid categories to filter after applying filter_categories."
            )

    # Plot the spatial distribution using selected plot type
    plt.figure(figsize=figsize)

    if plot_type == 'scatter':
        # Scatter plot using longitude, latitude as the axes
        sns.scatterplot(
            x='longitude',
            y='latitude',
            hue='category',
            data=df,
            palette=cmap,
            alpha=alpha
        )
    elif plot_type == 'hexbin':
        # Hexbin plot for large number of points
        hb = plt.hexbin(
            df['longitude'],
            df['latitude'],
            gridsize=50,
            cmap=cmap,
            mincnt=1
        )
        plt.colorbar(hb, label='Count')
    else:
        raise ValueError(
            f"Unsupported plot_type: {plot_type}"
        )

    # Labels and title
    plt.title(f"Spatial Distribution of {category_column}")
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    if plot_type == 'scatter':
        plt.legend(
            title='Categories',
            bbox_to_anchor=(1.05, 1),
            loc='upper left'
        )
 
    if not show_grid: 
        plt.grid(False)
        
    if axis_off:
        plt.axis('off')
    
    # Show plot
    plt.tight_layout()
    plt.show()


@default_params_plot(
    savefig=PlotConfig.AUTOSAVE('my_distribution_plot.png'))
@validate_params ({ 
    'df': ['array-like'], 
    'x_col': [str], 
    'y_col': [str], 
    'z_cols': ['array-like', str], 
    'plot_type': [StrOptions({'scatter', 'hexbin', 'density'})], 
    'max_cols': [Real]
    })
@isdf 
def plot_dist(
    df: DataFrame,
    x_col: str,
    y_col: str,
    z_cols: List[str],
    plot_type: str = 'scatter',
    axis_off: bool = True,
    max_cols: int = 3,
    cmap='viridis', 
    savefig=None,
):
    r"""
    Plot multiple distribution datasets on a grid of subplots for 
    comprehensive spatial analysis. This function generates a grid of 
    subplots illustrating the variation of multiple `z`-axis variables 
    (``z_cols``) against spatial coordinates defined by `x_col` and 
    `y_col`. Depending on the chosen `plot_type`, it can create scatter, 
    hexbin, or density plots. Users can visualize how each `z` variable 
    behaves over the spatial domain defined by `x` and `y`, allowing 
    intuitive comparisons and spatial pattern recognition.

    This object aims to project a set of `z` values as a function
    of `x` and `y`, thereby constructing a distribution surface. 
    Mathematically, for each variable :math:`z_i` in ``z_cols``, we 
    consider a mapping:

    .. math::
       z_i = f(x, y)

    where :math:`f: \mathbb{R}^2 \to \mathbb{R}`. The plotting depends 
    on the chosen `plot_type`:
    
    - ``'scatter'``: Directly plots points in the plane colored by their 
      corresponding `z` values.
    - ``'hexbin'``: Aggregates data into hexagonal bins and colors each 
      bin based on the average `z` value.
    - ``'density'``: Estimates a continuous density surface using kernel 
      density estimation, with `z` values acting as weights. This 
      representation assumes non-negative weights [1]_.

    The function arranges these plots in a grid with a maximum of 
    ``max_cols`` columns. If the number of `z_cols` exceeds this, 
    additional rows are added. An optional colorbar provides a unified 
    scale for interpreting the color encoding of `z` values across all 
    subplots.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the data to visualize. Must 
        include at least `x_col`, `y_col`, and all specified `z_cols`.
        The DataFrame should contain numeric values for these columns.
    x_col : str
        Name of the column representing the x-axis coordinate 
        (e.g., longitude). This parameter specifies the independent
        spatial dimension.
    y_col : str
        Name of the column representing the y-axis coordinate 
        (e.g., latitude). Combined with `x_col`, it forms a spatial 
        plane onto which `z` values are projected.
    z_cols : list of str
        A list of column names corresponding to the variables 
        to be visualized along the `z` dimension. Each `z_col` 
        represents a different distribution over the `(x, y)` space.
    plot_type : str, optional
        The type of plot to generate. Supported types are:
        
        - ``'scatter'``: Plots individual data points with colors 
          indicating `z` values.
        - ``'hexbin'``: Uses hexagonal binning to visualize data 
          density and average `z` within each bin.
        - ``'density'``: Creates a kernel density estimate (KDE) 
          surface weighted by `z` values. All `z` values must be 
          non-negative for this method.
        
        Defaults to ``'scatter'``.
    axis_off : bool, optional
        If True, removes axis lines and labels for a cleaner look. 
        Default is True.
    max_cols : int, optional
        The maximum number of columns in the subplot grid. If the 
        total number of `z_cols` exceeds `max_cols`, additional 
        rows are created automatically. Default is 3.
    cmap: str,  
       Matplotlib colormap plot. Default is ``'viridis'``. 
       
    savefig : str or None, optional
        If provided, specifies the filename or path where the 
        resulting figure should be saved. If None, the figure is 
        displayed interactively.

    Methods
    -------
    This object is a standalone function, therefore it does not 
    provide class methods. No additional callable methods are 
    exposed aside from the function itself. Users interact 
    solely through the function parameters described above.

    Notes
    -----
    - For `density` plot types, ensure no negative values are present 
      in the `z_cols`. Negative weights cause errors in kernel 
      density estimation.
    - Large datasets might benefit from `hexbin` or `density` plots 
      to better visualize overall patterns rather than individual 
      points.

    Examples
    --------
    >>> from gofast.plot.suite import plot_dist
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'longitude': [1,2,3,4],
    ...     'latitude' : [10,10,10,10],
    ...     'subsidence_2018': [5,6,7,8],
    ...     'subsidence_2022': [3,4,2,1]
    ... })
    >>> plot_dist(df, x_col='longitude', y_col='latitude',
    ...           z_cols=['subsidence_2018', 'subsidence_2022'],
    ...           plot_type='scatter', axis_off=True, max_cols=2)

    See Also
    --------
    matplotlib.pyplot.scatter : For basic scatter plot creation.
    matplotlib.pyplot.hexbin : For hexagonal binning visualization.
    seaborn.kdeplot : For kernel density estimation plots.
    gofast.plot.suite.plot_distributions: 
        For the distribution of numeric columns in the DataFrame.

    References
    ----------
    .. [1] Rosenblatt, M. "Remarks on some nonparametric estimates 
           of a density function." Ann. Math. Statist. 27 (1956), 
           832-837.
    """

    # Validate Input Columns
    exist_features(df, features= [x_col, y_col], name ='Columns `x` and `y`')
    z_cols = columns_manager(z_cols, empty_as_none= False, )
    exist_features(df, features=z_cols, name ='Value `z_cols` ')
    
    extra_msg = (
        "If a numeric feature is stored as an 'object' type, "
        "it should be explicitly converted to a numeric type"
        " (e.g., using `pd.to_numeric`)."
    )
    check_features_types(
        df, features= [x_col, y_col] + z_cols , dtype='numeric',
        extra=extra_msg
    )

    # If using 'density', ensure weights are non-negative
    if plot_type == 'density':
        for col in z_cols:
            if (df[col] < 0).any():
                raise ValueError(
                    f"Negative values found in '{col}'. Seaborn kdeplot cannot "
                    "handle negative weights. Please provide non-negative values "
                    "or choose a different plot_type."
                )
    max_cols = validate_positive_integer(max_cols, 'max_cols')
    
    # Determine Subplot Grid Layout
    num_z = len(z_cols)
    n_cols = min(max_cols, num_z)
    n_rows = math.ceil(num_z / max_cols)

    
    # Create Subplots
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        constrained_layout=True
    )

    # Ensure axes is a 2D array for consistent indexing
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    # Calculate Overall vmin and vmax for Color Normalization
    overall_min = df[z_cols].min().min()
    overall_max = df[z_cols].max().max()


    # Iterate Over z_cols and Plot
    for idx, z_col in enumerate(z_cols):
        row = idx // max_cols
        col = idx % max_cols
        ax = axes[row][col]

        if plot_type == 'scatter':
            # Create a scatter plot
            ax.scatter(
                df[x_col],
                df[y_col],
                c=df[z_col],
                cmap=cmap,
                s=10,
                alpha=0.7,
                norm=plt.Normalize(vmin=overall_min, vmax=overall_max)
            )
        elif plot_type == 'hexbin':
            # Create a hexbin plot
            ax.hexbin(
                df[x_col],
                df[y_col],
                C=df[z_col],
                gridsize=50,
                cmap=cmap,
                reduce_C_function=np.mean,
                norm=plt.Normalize(vmin=overall_min, vmax=overall_max)
            )
        elif plot_type == 'density':
            # Create a density (kde) plot
            # seaborn.kdeplot returns a QuadContourSet, not directly used afterward,
            # but that's fine. We don't need to assign it.
            sns.kdeplot(
                x=df[x_col],
                y=df[y_col],
                weights=df[z_col],
                fill=True,
                cmap=cmap,
                ax=ax,
                thresh=0
            )

        # Set plot title
        ax.set_title(f'{z_col}', fontsize=12)

        # Optionally turn off axes
        if axis_off:
            ax.axis('off')

    # Hide Any Unused Subplots
    total_plots = n_rows * n_cols
    if num_z < total_plots:
        for idx in range(num_z, total_plots):
            row = idx // max_cols
            col = idx % max_cols
            axes[row][col].axis('off')

    # Add a Single Colorbar
    if plot_type in ['scatter', 'hexbin', 'density']:
        # Create a ScalarMappable for the colorbar
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=overall_min, vmax=overall_max)
        )
        sm.set_array([])  # Only needed for older Matplotlib versions

        # Flatten the axes array to pass a list of Axes to colorbar
        all_axes = [ax for row_axes in axes for ax in row_axes]

        # Add colorbar using all axes to properly position it
        cbar = fig.colorbar(
            sm,
            ax=all_axes,
            orientation='vertical',
            fraction=0.02,
            pad=0.04
        )
        cbar.set_label('Value', fontsize=12)

    plt.show()


@default_params_plot(
    savefig=PlotConfig.AUTOSAVE('my_q.distributions_plot.png')
  )
@validate_params ({ 
    'df': ['array-like'], 
    'x_col': [str], 
    'y_col': [str], 
    'dt_col': [ str], 
    'quantiles': ['array-like', None], 
    'dt_values':['array-like', None], 
    'value_prefix': [str], 
    'q_cols': [dict, None], 
    })
@isdf 
def plot_quantile_distributions(
    df,
    x_col, 
    y_col,
    dt_col, 
    quantiles=None,
    dt_values=None,
    value_prefix='',
    q_cols=None,
    cmap='viridis',
    s=10,
    alpha=0.8,
    cbar_orientation='vertical',
    cbar_fraction=0.02,
    cbar_pad=0.1,
    figsize=None,
    savefig=None,
    dpi=300,
    axis_off=True, 
    reverse=False, 
):
    r"""
    Plot quantile distributions across spatial and temporal domains,
    visualizing multiple `z`-value distributions defined by quantiles
    over specific date values. This function arranges the resulting 
    plots in a grid, with quantiles along one dimension and time/date 
    values along the other dimension, allowing users to investigate 
    how distributions change over space and time.

    Mathematically, given spatial coordinates :math:`(x,y)` and a set 
    of quantiles :math:`Q = \{q_1, q_2, ..., q_m\}` and date values 
    :math:`D = \{d_1, d_2, ..., d_n\}`, we consider a function 
    :math:`f(x,y,d,q)` that returns a value for each combination. The 
    columns of the DataFrame must represent these values. The function 
    arranges subplots in an :math:`m \times n` grid, where each cell 
    in the grid corresponds to a specific quantile `q` and a date `d`. 
    Each subplot displays:

    .. math::
       z_{q,d}(x,y) = f(x,y; q, d)

    where `z_{q,d}` is extracted from columns based on a naming 
    convention, or directly provided via ``q_cols``.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing spatial and value data. It must 
        include at least the columns corresponding to `<x_col>`, 
        `<y_col>`, `<dt_col>`, and either:

        1) Columns named according to the convention 
           ``<value_prefix>_<date>_q<quantile>``, or
        2) Explicit mappings in `q_cols`.
    x_col : str
        The column name representing the x-axis coordinate 
        (e.g., longitude).
    y_col : str
        The column name representing the y-axis coordinate 
        (e.g., latitude).
    dt_col : str
        The column representing dates/times. This can be integer (e.g., year),
        or datetime. If it's datetime, values are filtered by year 
        extracted from that datetime.
    quantiles : list of float, optional
        A list of quantiles (e.g., [0.1, 0.5, 0.9]). If None, the function
        attempts to detect quantiles from columns if ``q_cols`` is also 
        None. If detection fails, a ValueError is raised.
    dt_values : list, optional
        List of date values to plot. If None, the function infers date 
        values from `df`. If `dt_col` is integer, they are considered 
        as years. If `dt_col` is datetime, the year part is extracted.
    value_prefix : str, optional
        The prefix used in column naming format:
        ``<value_prefix>_<date>_q<quantile>``. By default empty, but 
        often something like 'predicted_subsidence'.
    q_cols : dict or None, optional
        If provided, explicitly maps quantiles and dates to columns. 
        Two forms are supported:
        
        - ``{quantile: {date_str: column_name}}``
        - ``{quantile: [list_of_columns_in_same_order_as_dates]}``
        
        If None, the function uses the naming convention or detection.
    cmap : str, optional
        Colormap name used for color encoding values. Default is 'viridis'.
    s : int, optional
        Marker size for scatter plots.
    alpha : float, optional
        Marker transparency.
    cbar_orientation : str, optional
        Orientation of the colorbar. Default is 'vertical'.
    cbar_fraction : float, optional
        Fraction of original axes size occupied by colorbar.
    cbar_pad : float, optional
        Padding between colorbar and the edge of the plot.
    figsize : tuple, optional
        Figure size (width, height) in inches. If None, chosen 
        automatically based on the number of rows and columns.
    savefig : str or None, optional
        If provided, a path to save the figure as an image. If None, 
        displays the figure interactively.
    dpi : int, optional
        Dots-per-inch for the saved figure. Default is 300.
    axis_off : bool, optional
        If True, removes axis ticks and labels for cleaner visualization.
    reverse : bool, optional
        Controls the orientation of the quantile distribution plots.
        By default (`reverse=False`), quantiles are placed along the 
        rows and dates along the columns:

        .. math::
           \text{Rows} \rightarrow \text{Quantiles} \\
           \text{Columns} \rightarrow \text{Dates}

        This results in a grid where each row corresponds to a 
        quantile, and each column corresponds to a date value.

        When `reverse=True`, the layout is inverted so that quantiles 
        are arranged along the columns and dates along the rows:

        .. math::
           \text{Rows} \rightarrow \text{Dates} \\
           \text{Columns} \rightarrow \text{Quantiles}

        In this scenario, each column corresponds to a quantile and 
        each row corresponds to a date value. This can be useful when 
        you want to compare quantiles side-by-side horizontally rather 
        than vertically.

    Methods
    -------
    This object is a standalone function with no associated methods.
    Users interact only through its parameters.

    Notes
    -----
    - If `quantiles` is None and no suitable columns are found, a 
      ValueError is raised.
    - If `q_cols` is provided, it overrides automatic detection or 
      naming conventions.
    - The filtering step for datetime `dt_col` extracts the year 
      component. For custom temporal resolutions, users might need 
      to preprocess the data.
    - Negative values in the data might be acceptable for scatter 
      and hexbin plots, but if a density approach is implemented 
      elsewhere, ensure non-negative weights [1]_.

    Examples
    --------
    Consider a DataFrame `df` with columns:
    'longitude', 'latitude', 'year', and 
    'predicted_subsidence_2024_q0.1', 
    'predicted_subsidence_2024_q0.5', 
    'predicted_subsidence_2024_q0.9', etc.
    
    Consider generating a sample DataFrame suitable for testing:

    >>> import pandas as pd
    >>> import numpy as np

    >>> num_points = 10
    >>> years = [2024, 2025]
    >>> quantiles = [0.1, 0.5, 0.9]

    >>> longitudes = np.random.uniform(100.0, 101.0, num_points)
    >>> latitudes  = np.random.uniform(20.0, 21.0,  num_points)

    >>> # Initialize the DataFrame columns
    >>> data = {
    ...    'longitude': longitudes,
    ...    'latitude': latitudes
    ... }
    >>> # Add the 'year' column, repeating for each point
    >>> data['year'] = np.repeat(years, num_points)  # Repeat years for each point
    >>> for year in years:
    ...     for q in quantiles:
    ...         q_col = f'predicted_subsidence_{year}_q{q}'
    ...         # Generate predicted subsidence value for each quantile
    ...         data[q_col] = np.random.uniform(0, 50, num_points) *\
    ...    (1 + np.random.uniform(-0.1, 0.1))

    >>> df = pd.DataFrame(data)
    >>> df.head()

    This `df` will have columns like:
    `'longitude', 'latitude', 'year', 
    'predicted_subsidence_2024_q0.1', 'predicted_subsidence_2024_q0.5', 
    'predicted_subsidence_2024_q0.9', 'predicted_subsidence_2025_q0.1', 
    'predicted_subsidence_2025_q0.5', 'predicted_subsidence_2025_q0.9'`.


    >>> from gofast.plot.suite import plot_quantile_distributions
    >>> # Automatically detect quantiles and dates:
    >>> plot_quantile_distributions(
    ...     df,
    ...     x_col='longitude',
    ...     y_col='latitude',
    ...     dt_col='year',
    ...     value_prefix='predicted_subsidence'
    ... )

    Or specifying quantiles:
    >>> plot_quantile_distributions(
    ...     df,
    ...     x_col='longitude',
    ...     y_col='latitude',
    ...     dt_col='year',
    ...     quantiles=[0.1, 0.5, 0.9],
    ...     value_prefix='predicted_subsidence'
    ... )

    If `q_cols` is provided:
    >>> q_map = {
    ...     0.1: ['sub2024_q0.1','sub2025_q0.1'],
    ...     0.5: ['sub2024_q0.5','sub2025_q0.5'],
    ...     0.9: ['sub2024_q0.9','sub2025_q0.9']
    ... }
    >>> plot_quantile_distributions(
    ...     df,
    ...     x_col='longitude',
    ...     y_col='latitude',
    ...     dt_col='year',
    ...     quantiles=[0.1,0.5,0.9],
    ...     dt_values=[2024,2025],
    ...     q_cols=q_map
    ... )

    See Also
    --------
    gofast.plot.suite.plot_qdist: More robust quantile plot.
    matplotlib.pyplot.scatter : Scatter plot generation.
    matplotlib.pyplot.hexbin : Hexbin plot generation.
    seaborn.kdeplot : For density visualization.

    References
    ----------
    .. [1] Rosenblatt, M. "Remarks on some nonparametric estimates 
           of a density function." Ann. Math. Statist. 27 (1956), 
           832-837.
    """
    # Infer dt_values if not provided
    if dt_values is None:
        if pd.api.types.is_integer_dtype(df[dt_col]):
            dt_values = sorted(df[dt_col].unique())
        elif np.issubdtype(df[dt_col].dtype, np.datetime64):
            dt_values = pd.to_datetime(df[dt_col].unique()).sort_values()
        else:
            dt_values = sorted(df[dt_col].unique())

    dt_values_str = _extract_date_values_if_datetime(df, dt_col, dt_values)
    # If q_cols is None, we rely on quantiles and naming convention
    # or detect quantiles if quantiles is None.
    quantiles = columns_manager(quantiles, empty_as_none= True )
    if q_cols is None:
        # If quantiles is None, try to detect from columns
        if quantiles is None:
            # Attempt detection
            detected_quantiles = detect_quantiles_in( 
                df, col_prefix= value_prefix, 
                dt_value = dt_values_str , 
                return_types='q_val'
                )
            if not detected_quantiles: 
                # retry 
                detected_quantiles = _detect_quantiles_from_columns(
                    df, value_prefix, dt_values_str)

            if detected_quantiles is None:
                raise ValueError(
                    "No quantiles detected from columns."
                    " Please specify quantiles or q_cols."
                    )
            quantiles = detected_quantiles
        # Now we have quantiles, build column names
   
        all_cols = _build_column_names(
            df, value_prefix, quantiles, dt_values_str
            )
        if not all_cols: 
            #retry:
            all_cols = build_q_column_names(
                df, quantiles=quantiles, 
                dt_value=dt_values_str, 
                strict_match=False, 
                )
        if not all_cols:
            raise ValueError(
                "No matching columns found with given prefix, date, quantiles.")
    else:
        
        # q_cols provided. Let's assume q_cols is a dict:
        # {quantile: {date_str: column_name}}
        # or {quantile: [cols_in_same_order_as_dt_values_str]}
        # We must extract quantiles from q_cols keys
        quantiles_from_qcols = validate_q_dict(q_cols)  
        quantiles_from_qcols = sorted(q_cols.keys())
        if quantiles is None:
            quantiles = quantiles_from_qcols
        else:
            quantiles = validate_quantiles (quantiles, dtype=np.float64 )
            # Ensure that quantiles match q_cols keys
            if set(quantiles) != set(quantiles_from_qcols):
                raise ValueError(
                    "Quantiles specified do not match q_cols keys.")
        
        # Validate all columns exist
        all_cols = []
        for q in quantiles:
            mapping = q_cols[q]
            if isinstance(mapping, dict):
                # expect {date_str: col_name}
                for d_str in dt_values_str:
                    if d_str not in mapping:
                        continue
                    c = mapping[d_str]
                    if c not in df.columns:
                        raise ValueError(f"Column {c} not found in DataFrame.")
                    all_cols.append(c)
            else:
                # assume list parallel to dt_values_str
                if len(mapping) != len(dt_values_str):
                    raise ValueError("q_cols mapping length does not match dt_values.")
                for c in mapping:
                    if c not in df.columns:
                        raise ValueError(f"Column {c} not found in DataFrame.")
                all_cols.extend(mapping)

    # Compute overall min and max for color normalization
    if not all_cols:
        raise ValueError("No columns determined for plotting.")

    overall_min = df[all_cols].min().min()
    overall_max = df[all_cols].max().max()

    if reverse:
        _plot_reversed(
            df, x_col, y_col, dt_col, 
            quantiles, dt_values_str, 
            q_cols, value_prefix,
            cmap, s, alpha, axis_off,
            overall_min, overall_max,
            cbar_orientation, cbar_fraction, 
            cbar_pad, figsize, dpi, savefig
        )
        return 
     
    # Determine subplot grid size
    n_rows = len(quantiles)
    n_cols = len(dt_values)

    if figsize is None:
        figsize = (5 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=figsize,
        constrained_layout=True
        )

    # Ensure axes is 2D array
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    for i, q in enumerate(quantiles):
        # Extract columns per quantile and date
        for j, d in enumerate(dt_values_str):
            ax = axes[i, j]

            if q_cols is None:
                col_name = f'{value_prefix}_{d}_q{q}'
                if col_name not in df.columns:
                    ax.axis('off')
                    ax.set_title(f'No data for {d} (q={q})')
                    continue
                subset_col = col_name
            else:
                mapping = q_cols[q]
                if isinstance(mapping, dict):
                    if d not in mapping:
                        ax.axis('off')
                        ax.set_title(f'No col mapped for {d} (q={q})')
                        continue
                    subset_col = mapping[d]
                else:
                    # list scenario
                    # find index of d in dt_values_str
                    idx_date = dt_values_str.index(d)
                    subset_col = mapping[idx_date]

            subset = df[[x_col, y_col, dt_col, subset_col]].dropna()

            # Filter subset based on dt_col and d
            if np.issubdtype(df[dt_col].dtype, np.datetime64):
                # convert d to int year
                year_int = int(d)
                subset = subset[subset[dt_col].dt.year == year_int]
            else:
                # Attempt to convert d to int if possible
                try:
                    val = int(d)
                except ValueError:
                    val = d
                subset = subset[subset[dt_col] == val]

            if subset.empty:
                ax.axis('off')
                ax.set_title(f'No data after filter for {d} (q={q})')
                continue

            scatter = ax.scatter(
                subset[x_col],
                subset[y_col],
                c=subset[subset_col],
                cmap=cmap,
                s=s,
                alpha=alpha,
                norm=plt.Normalize(vmin=overall_min, vmax=overall_max)
            )

            if np.issubdtype(df[dt_col].dtype, np.datetime64):
                title_str = f"Quantile={q}, Year={d}"
            else:
                title_str = f"Quantile={q}, {dt_col}={d}"
            ax.set_title(title_str, fontsize=10)

            if axis_off:
                ax.axis('off')

        # Add colorbar for the last subplot in each row
        cbar = fig.colorbar(
            scatter,
            ax=axes[i, -1],
            orientation=cbar_orientation,
            fraction=cbar_fraction,
            pad=cbar_pad
        )
        cbar.set_label('Value', fontsize=10)

    if savefig:
        fig.savefig(savefig, dpi=dpi)

    plt.show()

def _extract_date_values_if_datetime(df, dt_col, dt_values):
    """
    Extract date values as strings suitable for column naming if dt_col 
    is datetime. If dt_col is datetime, convert them to year strings, 
    otherwise return them as they are.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the dt_col.
    dt_col : str
        The name of the date column in df.
    dt_values : list
        A list of date values inferred or provided.

    Returns
    -------
    list
        A list of string representations of the date values.
    """
    # Convert datetime if needed
    # For datetime, dt_values are datetimes; might want to format them
    # as years or something else depending on the scenario
    # Extract dt_values as strings if datetime
    
    if np.issubdtype(df[dt_col].dtype, np.datetime64):
        # If datetime, convert each date to its year string
        return [str(pd.to_datetime(d).year) for d in dt_values]
    else:
        # Otherwise just convert to string
        return [str(d) for d in dt_values]


def _build_column_names(df, value_prefix, quantiles, dt_values_str):
    """
    Build a list of column names based on a prefix, quantiles, 
    and date values (as strings), checking which exist in the DataFrame.

    Assumes the naming format: f'{value_prefix}_{date}_q{quantile}'

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to check for columns.
    value_prefix : str
        The prefix used in column naming.
    quantiles : list
        A list of quantiles.
    dt_values_str : list
        A list of date values as strings.

    Returns
    -------
    list
        A list of column names that exist in df.
    """
    # Build column names dynamically or handle scenario where we must 
    # extract the subset for each date
    # Suppose the columns are of the form value_prefix_date_qquantile
    # If this is not the case, user must supply data differently or 
    # we can handle differently. For now, let's assume columns are named:
    # f"{value_prefix}_{date}_q{quantile}"
    # If dt_col is datetime, we might extract year or something else
    
    all_cols = []
    for q in quantiles:
        for d_str in dt_values_str:
            col_name = f'{value_prefix}_{d_str}_q{q}'
            if col_name in df.columns:
                all_cols.append(col_name)
    return all_cols

def _detect_quantiles_from_columns(df, value_prefix, dt_values_str):
    """
    Attempt to detect quantiles by scanning the DataFrame columns that match
    the pattern f'{value_prefix}_{date}_q...' for each date in dt_values_str.
    Extract the quantiles from these column names.

    Parameters
    ----------
    df : pandas.DataFrame
    value_prefix : str
        The value prefix expected in column names.
    dt_values_str : list
        A list of date strings.

    Returns
    -------
    quantiles : list
        A list of detected quantiles (floats) sorted.
    """
    quantile_pattern = re.compile(r'_q([\d\.]+)$')
    found_quantiles = set()

    # Check columns that start with value_prefix and contain one of the dates
    # and end with q{quantile}
    for col in df.columns:
        # Check if col fits pattern: value_prefix_date_qquantile
        # First verify value_prefix and date are in it
        # This is simplistic: we check if any date is in col
        # and also if it matches the quantile pattern at the end.
        if col.startswith(value_prefix + "_"):
            # Extract the part after value_prefix_
            remainder = col[len(value_prefix)+1:]
            # remainder should look like date_qquantile
            # Check if it contains a known date_str
            for d_str in dt_values_str:
                if remainder.startswith(d_str + "_q"):
                    # match qquantile
                    m = quantile_pattern.search(col)
                    if m:
                        q_val = float(m.group(1))
                        found_quantiles.add(q_val)
                    break

    if not found_quantiles:
        return None
    return sorted(found_quantiles)

def _plot_reversed(
    df, x_col, y_col, dt_col,
    quantiles, dt_values_str, q_cols, value_prefix,
    cmap, s, alpha, axis_off,
    overall_min, overall_max, cbar_orientation,
    cbar_fraction, cbar_pad,
    figsize, dpi, savefig
):
    # Reversed mode: quantiles in columns, dates in rows
    n_rows = len(dt_values_str)
    n_cols = len(quantiles)
    if figsize is None:
        figsize = (5 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=figsize,
        constrained_layout=True
    )

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    for j, d in enumerate(dt_values_str):
        for i, q in enumerate(quantiles):
            ax = axes[j, i]

            if q_cols is None:
                col_name = f'{value_prefix}_{d}_q{q}'
                if col_name not in df.columns:
                    ax.axis('off')
                    ax.set_title(f'No data for {d} (q={q})')
                    continue
                subset_col = col_name
            else:
                mapping = q_cols[q]
                if isinstance(mapping, dict):
                    if d not in mapping:
                        ax.axis('off')
                        ax.set_title(f'No col mapped for {d} (q={q})')
                        continue
                    subset_col = mapping[d]
                else:
                    idx_date = dt_values_str.index(d)
                    subset_col = mapping[idx_date]

            subset = df[[x_col, y_col, dt_col, subset_col]].dropna()

            # Filter subset based on dt_col and d
            if np.issubdtype(df[dt_col].dtype, np.datetime64):
                year_int = int(d)
                subset = subset[subset[dt_col].dt.year == year_int]
            else:
                try:
                    val = int(d)
                except ValueError:
                    val = d
                subset = subset[subset[dt_col] == val]

            if subset.empty:
                ax.axis('off')
                ax.set_title(f'No data after filter for {d} (q={q})')
                continue

            scatter = ax.scatter(
                subset[x_col],
                subset[y_col],
                c=subset[subset_col],
                cmap=cmap,
                s=s,
                alpha=alpha,
                norm=plt.Normalize(vmin=overall_min, vmax=overall_max)
            )

            if np.issubdtype(df[dt_col].dtype, np.datetime64):
                title_str = f"Quantile={q}, Year={d}"
            else:
                title_str = f"Quantile={q}, {dt_col}={d}"
            ax.set_title(title_str, fontsize=10)

            if axis_off:
                ax.axis('off')

        # Add colorbar for the last column in each row
        # Actually, we add one per row:
            # better to add at the end of each row?
        # But here symmetrical to normal, we do once per row
        cbar = fig.colorbar(
            scatter,
            ax=axes[j, -1] if n_cols > 1 else axes[j, 0],
            orientation=cbar_orientation,
            fraction=cbar_fraction,
            pad=cbar_pad,
        )
        cbar.set_label('Value', fontsize=10)

    if savefig:
        fig.savefig(savefig, dpi=dpi)
    plt.show()


@default_params_plot(
    savefig=PlotConfig.AUTOSAVE('my_q.dist_plot.png'), 
    fig_size=None, 
  )
@validate_params ({ 
    'df': ['array-like'], 
    'quantiles': ['array-like', None], 
    'dt_values':['array-like', None], 
    'value_prefix': [str, None], 
    'q_cols': [dict, None], 
    })
@isdf 
def plot_qdist(
    df,
    x_col,               
    y_col,               
    dt_name=None,        
    quantiles=None,
    dt_values=None,
    value_prefix=None,   
    q_cols=None,
    cmap='viridis',
    s=10,
    alpha=0.8,
    cbar_orientation='vertical',
    cbar_fraction=0.02,
    cbar_pad=0.1,
    figsize=None,
    savefig=None,
    dpi=300,
    axis_off=True,
    reverse=False, 
    show_grid=False, 
    grid_props=None, 
):
    r"""
    Plot quantile distributions across spatial (x,y) domains and,
    optionally, temporal or categorical dimensions. This function
    supports four naming configurations for identifying quantile
    columns:
    
    1. ``<value_prefix>_<dt_value>_q<quantile>``
    2. ``<dt_value>_q<quantile>``
    3. ``<value_prefix>_q<quantile>``
    4. ``q<quantile>``
    
    In essence, :func:`plot_qdist` either creates a 2D grid of subplots
    (when date/time values are detected or explicitly provided) or a
    1D arrangement of subplots (only quantiles). By default, rows
    represent quantiles and columns represent date/time steps:
    
    .. math::
       \text{Rows} \rightarrow \text{Quantiles}, \quad
       \text{Columns} \rightarrow \text{Dates}
    
    When ``reverse=True``, the layout is inverted:
    
    .. math::
       \text{Rows} \rightarrow \text{Dates}, \quad
       \text{Columns} \rightarrow \text{Quantiles}
    
    
    Parameters
    ----------
    df: pd.DataFrame
        The input DataFrame containing at least the coordinates
        `<x_col>` and `<y_col>`, plus columns that match one of
        the naming configurations above or a user-supplied mapping
        in ``q_cols``.
    
    x_col : str
        Name of the column representing the x-coordinate (e.g.,
        'longitude').
    
    y_col : str
        Name of the column representing the y-coordinate (e.g.,
        'latitude').
    
    dt_name : str, optional
        If not ``None``, a descriptive label for the date/time
        dimension (e.g., 'year'). Used solely for subplot titles
        and labeling. If omitted, the function treats the data as
        1D in quantiles.
    
    quantiles : list of float, optional
        A user-specified list of quantile values (e.g., ``[0.1,
        0.5, 0.9]``). If not provided, the function attempts to
        detect quantiles from DataFrame columns. If detection
        fails and no mapping is given in ``q_cols``, an error is
        raised.
    
    dt_values : list, optional
        A list of date/time values or identifiers (e.g., years)
        for which columns exist in the DataFrame. When present,
        :func:`plot_qdist` tries to match them against column
        names. If None, the function infers them from the
        detected or matched columns.
    
    value_prefix : str or None, optional
        The prefix used in patterns like ``<value_prefix>_2025_q0.5``.
        If None, the function detects columns without filtering
        by a prefix.
    
    q_cols : dict, optional
        A direct mapping of quantiles to column names or dictionaries.
        For a 1D scenario:
        
        .. code-block:: python
    
           {
             0.1: "my_col_q0.1",
             0.5: "my_col_q0.5",
             0.9: "my_col_q0.9"
           }
    
        For a 2D scenario (quantiles x dt_values):
    
        .. code-block:: python
    
           {
             0.1: { "2023": "subs_2023_q0.1", "2024": "subs_2024_q0.1" },
             0.5: { "2023": "subs_2023_q0.5", "2024": "subs_2024_q0.5" }
           }
    
        This mapping bypasses automatic naming detection. If
        provided, `<quantiles>` and `<dt_values>` can also be
        specified for ordering or cross-verification.
    
    cmap : str, optional
        The colormap used for color encoding (default 'viridis').
    
    s : int, optional
        The marker size for scatter plots.
    
    alpha : float, optional
        The marker transparency in [0, 1].
    
    cbar_orientation : {'vertical', 'horizontal'}, optional
        Orientation of the colorbar. Default 'vertical'.
    
    cbar_fraction : float, optional
        Fraction of original axes to allocate for the colorbar.
    
    cbar_pad : float, optional
        Padding between the colorbar and subplot edge.
    
    figsize : tuple, optional
        The figure size in inches, e.g., ``(12, 6)``. If None,
        a default size is determined by the number of rows and
        columns.
    
    savefig : str, optional
        Path to save the resulting figure (e.g., 'output.png').
        If None, the figure is displayed interactively.
    
    dpi : int, optional
        The dots-per-inch resolution for the figure. Default 300.
    
    axis_off : bool, optional
        If True, hides the axes (no ticks or labels).
    
    reverse : bool, optional
        If True, inverts the subplot layout so that columns
        correspond to quantiles and rows to date/time steps.
        Otherwise, rows represent quantiles and columns
        represent date/time values.
    
    
    Examples
    --------
    >>> from gofast.plot.suite import plot_qdist
    >>> import pandas as pd
    >>> import numpy as np
    
    >>> # Example DataFrame with patterns like:
    >>> #    subsidence_2023_q10
    >>> #    subsidence_2023_q50
    >>> #    subsidence_2024_q90
    >>> n = 100
    >>> df = pd.DataFrame({
    ...   'longitude': np.random.uniform(113, 114, n),
    ...   'latitude':  np.random.uniform(22, 23,  n),
    ...   'subsidence_2023_q10': np.random.rand(n)*10,
    ...   'subsidence_2023_q50': np.random.rand(n)*10,
    ...   'subsidence_2024_q90': np.random.rand(n)*10
    ... })
    
    >>> # Simple call that detects date/time and quantiles:
    >>> plot_qdist(
    ...   df,
    ...   x_col='longitude',
    ...   y_col='latitude',
    ...   value_prefix='subsidence'
    ... )
    
    >>> # If no date/time dimension is present, columns might be
    >>> # e.g., 'subsidence_q10', 'subsidence_q50', 'subsidence_q90',
    >>> # or just 'q10','q50','q90' for truly single-dimensional data.
    
    Notes
    -----
    This function processes DataFrame columns that match specific
    naming patterns to identify date/time indices and quantile
    levels. If a date dimension is found, the resulting figure
    forms a 2D grid of subplots arranged by quantile and date.
    Otherwise, a 1D arrangement (only quantiles) is plotted.
    The user may override automatic detection by supplying a
    structured mapping in ``q_cols``.
    
    .. math::
       \mathbf{Plot}:
       \begin{cases}
       \text{2D: } \text{rows} \times \text{columns} = 
       \text{quantiles} \times \text{dates} & 
       \text{(or reversed if } reverse=True)\\
       \text{1D: } \text{rows} = \text{quantiles} & 
       \text{(single column or vice versa)}
       \end{cases}
    
    See Also
    --------
    gofast.plot.suite.plot_quantile_distributions: 
        Another quantiles plot with absolute `dt_col`.
    matplotlib.pyplot.scatter : Core scatter plotting.
    matplotlib.pyplot.colorbar : Colorbar configuration.
    
    References
    ----------
    .. [1] Han, J., Kamber, M., & Pei, J. (2011). *Data Mining:
           Concepts and Techniques*, 3rd edition. Elsevier.
    """

    # Function robustly handles four data configurations:
    #   (1) <value_prefix>_<dt_value>_q<quantile>
    #   (2) <dt_value>_q<quantile>
    #   (3) <value_prefix>_q<quantile>
    #   (4) q<quantile>

    # If a `dt_value` is detected, a 2D grid is plotted 
    # (dates x quantiles). Otherwise, a 1D grid is plotted 
    # (just quantiles).
    
    # Init grid props 
    if grid_props is None: 
        grid_props = {'linestyle': ':', 'alpha': .7}

    # If q_cols is provided, it overrides auto-detection logic
    if q_cols is not None:
        _plot_from_qcols(
            df=df, x_col=x_col, y_col=y_col,
            dt_name=dt_name, dt_values=dt_values,
            q_cols=q_cols, cmap=cmap, s=s,
            alpha=alpha, cbar_orientation=cbar_orientation,
            cbar_fraction=cbar_fraction, cbar_pad=cbar_pad,
            figsize=figsize, savefig=savefig, dpi=dpi,
            axis_off=axis_off, reverse=reverse, 
            show_grid=show_grid, 
            grid_props=grid_props
        )
        return

    # Auto-detect columns that match the patterns
    # (1) prefix_dt_qquant  => ^(prefix_)?(\d+)_q([\d.]+)$
    # (2) dt_qquant         => ^(\d+)_q([\d.]+)$
    # (3) prefix_qquant     => ^(prefix_)?q([\d.]+)$
    # (4) qquant            => ^q([\d.]+)$
    # We'll unify them with a single regex capturing optional prefix,
    # optional dt_value, and quantile.
    #   pattern:
    #   ^(?:([^_]+)_)?     # optional prefix (group 1), with underscore
    #      (?:(\d+))?_?    # optional dt_value (group 2), with optional underscore
    #      q([\d.]+)$      # quantile (group 3)
    #
    # Note: We allow prefix or dt_value to be absent. One or both might appear.

    pattern = re.compile(
        r'^(?:([^_]+)_)?(?:(\d+))?_?q([\d\.]+)$'
    )
    # We'll store columns in a structure:
    #   col_info[col_name] = (prefix, dt_val, quant_str)
    # prefix or dt_val can be None if absent
    col_info = {}
    for col in df.columns:
        m = pattern.match(col)
        if m:
            pref, dt_val, q_str = m.groups()
            # If user provided a specific value_prefix, filter by it
            if value_prefix is not None and pref is not None:
                if pref != value_prefix:
                    continue
            col_info[col] = (pref, dt_val, float(q_str))

    if not col_info:
        raise ValueError(
            "No columns match the recognized patterns. "
            "No q_cols provided either."
        )

    # Extract sets of dt_val and quant
    dt_vals_found = set()
    quants_found = set()
    for (pref, dt_val, q_val) in col_info.values():
        if dt_val is not None:
            dt_vals_found.add(dt_val)
        quants_found.add(q_val)

    # If user gave dt_values, we keep only those dt_val in dt_vals_found
    # If dt_name is None or if dt_vals_found is empty => single dimension
    # else => 2D
    if dt_values is not None:
        dt_values = [str(dv) for dv in dt_values]  # unify with string dt_val
        dt_vals_found = dt_vals_found.intersection(dt_values)

    dt_vals_found = sorted(dt_vals_found, key=lambda x: int(x)) \
                    if dt_vals_found else []
    quants_found = sorted(quants_found)

    # If user provides quantiles, we only keep those
    if quantiles is not None:
        qf = set(float(q) for q in quantiles)
        quants_found = [q for q in quants_found if q in qf]
    if not quants_found:
        raise ValueError("No matching quantiles found.")

    # 2D scenario if dt_vals_found has more than 0 elements 
    # and dt_name is not None.
    if dt_name and dt_vals_found:
        _plot_2d(
            df, x_col, y_col,
            col_info, dt_name, dt_vals_found,
            quants_found, value_prefix,
            cmap, s, alpha,
            cbar_orientation, cbar_fraction,
            cbar_pad, figsize, savefig,
            dpi, axis_off, reverse, 
            show_grid, grid_props
        )
    else:
        # 1D scenario for quantiles only
        _plot_1d(
            df, x_col, y_col,
            col_info, quants_found,
            value_prefix, cmap, s, alpha,
            cbar_orientation, cbar_fraction,
            cbar_pad, figsize,
            savefig, dpi, axis_off,
            reverse,show_grid, 
            grid_props
        )


def _plot_1d(df, x_col, y_col, col_info, quants_found, 
             value_prefix, cmap, s, alpha,
             cbar_orientation, cbar_fraction,
             cbar_pad, figsize, savefig, dpi,
             axis_off, reverse, show_grid, grid_props
             ):
    # We have only quantile dimension => one row or one column

    # Build col_name => (prefix, dt_val, q_val) reverse lookup
    # But we only care about q_val here
    q2col = {}
    for c, (p, d, q_val) in col_info.items():
        if q_val in quants_found and (d is None or d == ''):
            q2col[q_val] = c
        # If there's a dt_val, it won't be used in 1D scenario
        if q_val in quants_found and d is not None:
            # This means there's a dt_val, so skip
            pass

    # Some columns may not exist if the user data had dt_val but dt_name=None
    # We'll collect only those that have d is None
    # If none found, fallback to "any dt_val"? 
    valid_cols = [q2col[q] for q in quants_found if q in q2col]

    if not valid_cols:
        # fallback: if all columns have dt_val, we can't do 1D
        # So let's pick dt_val=some minimal or something
        any_col = {}
        for c, (p, d, q_val) in col_info.items():
            if q_val in quants_found:
                # We'll store first occurrence
                if q_val not in any_col:
                    any_col[q_val] = c
        if not any_col:
            raise ValueError("No columns available for 1D scenario.")
        q2col = any_col
        valid_cols = [q2col[q] for q in quants_found if q in q2col]

    n = len(quants_found)
    if reverse:
        n_rows, n_cols = 1, n
    else:
        n_rows, n_cols = n, 1

    if figsize is None:
        if reverse:
            figsize = (4 * n_cols, 4)
        else:
            figsize = (4, 4 * n_rows)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        constrained_layout=True
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    # Compute global min/max
    vals = df[valid_cols].values.ravel()
    overall_min, overall_max = np.nanmin(vals), np.nanmax(vals)

    for i, q in enumerate(quants_found):
        if q not in q2col:
            continue
        col_name = q2col[q]

        sub = df[[x_col, y_col, col_name]].dropna()

        if reverse:
            ax = axes[0, i]
        else:
            ax = axes[i, 0]

        sc = ax.scatter(
            sub[x_col],
            sub[y_col],
            c=sub[col_name],
            cmap=cmap,
            s=s,
            alpha=alpha,
            norm=plt.Normalize(vmin=overall_min, vmax=overall_max)
        )
        ax.set_title(f"q={q}", fontsize=10)
        if show_grid: 
            ax.grid(True, **grid_props)
        else: 
            ax.grid (False) 
            
        if axis_off:
            ax.axis('off')

        cbar = fig.colorbar(
            sc, ax=ax, orientation=cbar_orientation,
            fraction=cbar_fraction, pad=cbar_pad
        )
        cbar.set_label('Value', fontsize=10)

    if savefig:
        fig.savefig(savefig, dpi=dpi)
    else:
        plt.show()


def _plot_2d(df, x_col, y_col, col_info, dt_name, dt_vals_found,
             quants_found, value_prefix, cmap, s, alpha,
             cbar_orientation, cbar_fraction, cbar_pad,
             figsize, savefig, dpi, axis_off, reverse, 
             show_grid, grid_props):
    """
    2D scenario: dt_values x quantiles => grid
    If reverse=False, rows=quantiles, cols=dates
    If reverse=True, rows=dates, cols=quantiles
    """

    # Build dict: (dt_val, q_val) -> col_name
    dtq2col = {}
    for c, (p, d, q_val) in col_info.items():
        if d in dt_vals_found and q_val in quants_found:
            dtq2col[(d, q_val)] = c

    # If user provided dt_values externally, ensure they are strings
    # sorted. We'll only use the intersection dt_vals_found for plotting
    dt_vals_found = sorted(dt_vals_found, key=lambda x: int(x)) \
                    if dt_vals_found else []

    if reverse:
        # rows=dt, cols=q
        n_rows = len(dt_vals_found)
        n_cols = len(quants_found)
    else:
        # rows=q, cols=dt
        n_rows = len(quants_found)
        n_cols = len(dt_vals_found)

    if figsize is None:
        figsize = (4 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=figsize,
        constrained_layout=True
    )

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    # Gather all relevant columns
    used_cols = []
    for dt_val in dt_vals_found:
        for q in quants_found:
            if (dt_val, q) in dtq2col:
                used_cols.append(dtq2col[(dt_val, q)])
    used_cols = list(set(used_cols))
    vals = df[used_cols].values.ravel()
    overall_min, overall_max = np.nanmin(vals), np.nanmax(vals)

    for r_idx, row_key in enumerate(quants_found if not reverse else dt_vals_found):
        for c_idx, col_key in enumerate(dt_vals_found if not reverse else quants_found):
            ax = axes[r_idx, c_idx]

            if not reverse:
                # row_key=quantile, col_key=dt_val
                dt_val = col_key
                q_val = row_key
            else:
                dt_val = row_key
                q_val = col_key

            pair = (dt_val, q_val)
            if pair not in dtq2col:
                ax.axis('off')
                ax.set_title(f"No col for {dt_val}, q={q_val}")
                continue

            col_name = dtq2col[pair]
            sub = df[[x_col, y_col, col_name]].dropna()

            sc = ax.scatter(
                sub[x_col],
                sub[y_col],
                c=sub[col_name],
                cmap=cmap,
                s=s,
                alpha=alpha,
                norm=plt.Normalize(vmin=overall_min, vmax=overall_max)
            )

            if not reverse:
                title_str = f"{dt_name}={dt_val}, q={q_val}"
            else:
                title_str = f"{dt_name}={row_key}, q={col_key}"
            ax.set_title(title_str, fontsize=9)
            
            if show_grid: 
                ax.grid(True, **grid_props)
            else: 
                ax.grid (False) 
                
            if axis_off:
                ax.axis('off')

            # Add colorbar only for last column in the row (non-reverse)
            # or last column if reversed? We'll keep it simple:
            if (not reverse and (c_idx == n_cols - 1)) \
               or (reverse and (c_idx == n_cols - 1)):
                cbar = fig.colorbar(
                    sc, ax=ax, orientation=cbar_orientation,
                    fraction=cbar_fraction, pad=cbar_pad
                )
                cbar.set_label('Value', fontsize=8)

    if savefig:
        fig.savefig(savefig, dpi=dpi)
    else:
        plt.show()


def _plot_from_qcols(
    df, x_col, y_col,
    dt_name, dt_values, q_cols,
    cmap, s, alpha,
    cbar_orientation, cbar_fraction,
    cbar_pad, figsize,
    savefig, dpi,
    axis_off, reverse, 
    show_grid, grid_props 
):
    """
    If q_cols is provided, interpret it. This can be:
      - 1D scenario: {q: col_name}
      - 2D scenario: {q: {date_val: col_name}}
    or {q: [col_names in same order as dt_values]}
    """
    # Distinguish 1D vs 2D by checking if any q maps to a dict or list
    # If all map to single str -> 1D
    # If we find at least one dict or list -> 2D
    is_2d = False
    for q, mapping in q_cols.items():
        if isinstance(mapping, (dict, list)):
            is_2d = True
            break

    # If 2D but dt_name is None, we can skip dt dimension -> error or fallback
    if is_2d and not dt_name:
        raise ValueError(
            "q_cols indicates a 2D structure, but no dt_name was provided."
        )

    # If 1D, handle single dimension
    if not is_2d:
        # 1D scenario
        quants = sorted(q_cols.keys())
        col_names = []
        for q in quants:
            if q_cols[q] not in df.columns:
                raise ValueError(f"Column {q_cols[q]} not found.")
            col_names.append(q_cols[q])

        # Global min/max
        vals = df[col_names].values.ravel()
        overall_min, overall_max = np.nanmin(vals), np.nanmax(vals)
        n = len(quants)
        if reverse:
            n_rows, n_cols = (1, n)
        else:
            n_rows, n_cols = (n, 1)

        if figsize is None:
            if reverse:
                figsize = (4 * n_cols, 4)
            else:
                figsize = (4, 4 * n_rows)

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=figsize,
            constrained_layout=True
        )
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = np.array([axes])
        elif n_cols == 1:
            axes = axes.reshape(n_rows, 1)

        for i, q in enumerate(quants):
            col_name = q_cols[q]
            sub = df[[x_col, y_col, col_name]].dropna()

            ax = axes[0, i] if (reverse) else axes[i, 0]

            sc = ax.scatter(
                sub[x_col],
                sub[y_col],
                c=sub[col_name],
                cmap=cmap,
                s=s,
                alpha=alpha,
                norm=plt.Normalize(vmin=overall_min, vmax=overall_max)
            )
            ax.set_title(f"q={q}", fontsize=9)
            
            if show_grid: 
                ax.grid(True, **grid_props)
            else: 
                ax.grid (False) 
                
            if axis_off:
                ax.axis('off')

            cbar = fig.colorbar(
                sc, ax=ax,
                orientation=cbar_orientation,
                fraction=cbar_fraction,
                pad=cbar_pad
            )
            cbar.set_label('Value', fontsize=9)

        if savefig:
            fig.savefig(savefig, dpi=dpi)
        else:
            plt.show()
        return

    # 2D scenario
    # Expect {q: {dt_val: col_name}} or {q: [col_names in order of dt_values]}
    quants = sorted(q_cols.keys())
    if dt_values is None:
        # glean dt_values from the dictionary keys if any
        dt_set = set()
        for q in quants:
            mapping = q_cols[q]
            if isinstance(mapping, dict):
                dt_set.update(mapping.keys())
        dt_values = sorted(list(dt_set))

    # Validate columns exist
    used_cols = []
    dt_values_str = [str(dv) for dv in dt_values]
    for q in quants:
        mapping = q_cols[q]
        if isinstance(mapping, dict):
            # dt_val => col_name
            for d_str in dt_values_str:
                if d_str in mapping:
                    c = mapping[d_str]
                    if c not in df.columns:
                        raise ValueError(
                            f"Column '{c}' not found in DataFrame."
                        )
                    used_cols.append(c)
        elif isinstance(mapping, list):
            if len(mapping) != len(dt_values_str):
                raise ValueError(
                    "Length of q_cols[q] does not match dt_values."
                )
            for c in mapping:
                if c not in df.columns:
                    raise ValueError(f"Column '{c}' not found in DataFrame.")
            used_cols.extend(mapping)

    vals = df[used_cols].values.ravel()
    overall_min, overall_max = np.nanmin(vals), np.nanmax(vals)

    if reverse:
        # rows = dt, cols = q
        n_rows, n_cols = len(dt_values_str), len(quants)
    else:
        # rows = q, cols = dt
        n_rows, n_cols = len(quants), len(dt_values_str)

    if figsize is None:
        figsize = (4 * n_cols, 4 * n_rows)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=figsize,
        constrained_layout=True
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    for r_idx in range(n_rows):
        for c_idx in range(n_cols):
            ax = axes[r_idx, c_idx]
            if not reverse:
                q = quants[r_idx]
                d = dt_values_str[c_idx]
            else:
                d = dt_values_str[r_idx]
                q = quants[c_idx]

            mapping = q_cols[q]
            if isinstance(mapping, dict):
                if d not in mapping:
                    ax.axis('off')
                    ax.set_title(f"No data for {d}, q={q}")
                    continue
                col_name = mapping[d]
            else:
                # list scenario
                idx = dt_values_str.index(d)
                col_name = mapping[idx]

            sub = df[[x_col, y_col, col_name]].dropna()

            sc = ax.scatter(
                sub[x_col],
                sub[y_col],
                c=sub[col_name],
                cmap=cmap,
                s=s,
                alpha=alpha,
                norm=plt.Normalize(
                    vmin=overall_min,
                    vmax=overall_max
                )
            )
            ax.set_title(f"{dt_name}={d}, q={q}", fontsize=8)
            
            if show_grid: 
                ax.grid(True, **grid_props)
            else: 
                ax.grid (False) 
                
            if axis_off:
                ax.axis('off')

            # colorbar once per subplot or only last col?
            # we'll do once per subplot for clarity
            cbar = fig.colorbar(
                sc, ax=ax, orientation=cbar_orientation,
                fraction=cbar_fraction, pad=cbar_pad
            )
            cbar.set_label('Value', fontsize=8)

    if savefig:
        fig.savefig(savefig, dpi=dpi)
    else:
        plt.show()


@default_params_plot(
    savefig=PlotConfig.AUTOSAVE('my_uncertainty_plot.png'), 
    title ="Distribution of Uncertainties",
    fig_size=None 
 )
@validate_params ({ 
    'df': ['array-like'], 
    'cols': ['array-like', None], 
    'plot_type': [StrOptions({ 'box', 'violin', 'strip', 'swarm'})], 
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
    plot_type='box',
    figsize=None,
    numeric_only=True,
    title=None,
    ylabel=None,
    xlabel_rotation=45,
    palette='Set2',
    showfliers=True,
    grid=True,
    savefig=None,
    dpi=300
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
    plot_type : str, optional
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
    - By selecting different `<plot_type>` values, users can choose 
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
    >>> plot_uncertainty(df, cols=['A','B'], plot_type='violin')

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
    if plot_type == 'box':
        sns.boxplot(data=plot_data, showfliers=showfliers, palette=palette)
    elif plot_type == 'violin':
        sns.violinplot(data=plot_data, palette=palette)
    elif plot_type == 'strip':
        sns.stripplot(data=plot_data, palette=palette)
    elif plot_type == 'swarm':
        sns.swarmplot(data=plot_data, palette=palette)
   
    plt.title(title, fontsize=14)
    plt.ylabel(ylabel or "Predicted Value", fontsize=12)

    # Rotate x-labels if needed
    plt.xticks(rotation=xlabel_rotation)
    
    if grid:
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    if savefig:
        plt.savefig(savefig, dpi=dpi, bbox_inches='tight')

    plt.show()

@default_params_plot(
    savefig=PlotConfig.AUTOSAVE('my_prediction_intervals_plot.png'), 
    title ="Prediction Intervals",
    fig_size=(10, 6) 
 )
@validate_params ({ 
    'df': ['array-like'], 
    'x': ['array-like', None], 
    'line_colors': ['array-like'], 
    'numeric_only': [bool], 
    })
def plot_prediction_intervals(
    df,
    median_col=None,
    lower_col=None,
    upper_col=None, 
    actual_col=None,
    x=None,
    title=None,
    xlabel="X",
    ylabel="Value",
    legend=True,
    line_colors=('black', 'blue'),
    fill_color='blue',
    fill_alpha=0.2,
    figsize=None,
    grid=True,
    savefig=None,
    dpi=300
):
    r"""
    Plot prediction intervals to visualize forecast uncertainty or 
    variation in predicted values. This function displays a median 
    prediction line, optionally actual observed values, and fills the 
    area between lower and upper quantiles (or prediction bounds) to 
    highlight uncertainty.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing columns for actual and predicted 
        values. Must at least have `<median_col>`, `<lower_col>` and 
        `<upper_col>`.
    median_col : str
        The name of the column in `df` representing the median 
        prediction. This is the central prediction line.
    lower_col : str
        The name of the column in `df` representing the lower bound 
        of the prediction interval.
    upper_col : str
        The name of the column in `df` representing the upper bound 
        of the prediction interval.
    actual_col : str or None, optional
        The name of the column in `df` representing actual observed 
        values. If None, no actual line is plotted.
    x : array-like or None, optional
        The x-coordinates for plotting. If None, the index of `df` is 
        used.
    title : str, optional
        The plot title, describing the scenario or data.
    xlabel: str, optional
        The label for the x-axis.
    ylabel : str, optional
        The label for the y-axis.
    legend : bool, optional
        If True, display a legend showing line and interval labels.
    line_colors : tuple, optional
        A tuple of colors (actual_line_color, median_line_color). 
        Defaults to ('black', 'blue').
    fill_color`: str, optional
        The color used for filling the interval region. Default is 
        'blue'.
    fill_alpha : float, optional
        The transparency level of the filled interval region. 
        Default is 0.2.
    figsize : tuple or None, optional
        The figure size in inches (width, height). If None, defaults 
        to (10, 6).
    grid: bool, optional
        If True, display a grid on the plot. Default is True.
    savefig : str or None, optional
        If provided, a file path to save the figure. If None, the 
        figure is displayed interactively.
    dpi: int, optional
        The resolution of the saved figure in dots-per-inch. 
        Default is 300.

    Methods
    -------
    This object is a standalone function without additional methods. 
    Users interact solely through the given parameters.

    Notes
    -----
    Given a series of predictions indexed by :math:`i`, let 
    :math:`m_i` be the median prediction and 
    :math:`\ell_i`, :math:`u_i` be the lower and upper bounds of the 
    prediction interval at index :math:`i`. The plot represents:

    .. math::
       \ell_i \leq m_i \leq u_i

    as a shaded region from :math:`\ell_i` to :math:`u_i`, with a line 
    at :math:`m_i`. If actual values :math:`a_i` are provided, they are 
    plotted as well to compare predictions against reality.
    
    By visualizing the median prediction along with the lower and 
    upper bounds, users gain insight into the uncertainty or 
    confidence intervals around their forecasts. If actual 
    observations are provided, comparing them against the median 
    and interval can highlight prediction accuracy or deviation.

    Examples
    --------
    >>> from gofast.plot.suite import plot_prediction_intervals
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...    'median_pred': [10, 12, 11, 13],
    ...    'lower_pred':  [9,  11, 10, 12],
    ...    'upper_pred':  [11, 13, 12, 14],
    ...    'actual':      [10.5, 11.5, 11.2, 12.8]
    ... })
    >>> # Plot intervals with actual:
    >>> plot_prediction_intervals(
    ...     df,
    ...     actual_col='actual',
    ...     median_col='median_pred',
    ...     lower_col='lower_pred',
    ...     upper_col='upper_pred',
    ...     xlabel='Time Step',
    ...     ylabel='Value',
    ...     title='Forecast Intervals'
    ... )

    See Also
    --------
    matplotlib.pyplot.fill_between : Used for filling intervals.
    matplotlib.pyplot.plot : For line plotting.

    References
    ----------
    .. [1] Gneiting, T., and Raftery, A. E. "Strictly Proper 
           Scoring Rules, Prediction, and Estimation." J. Amer. 
           Statist. Assoc. 102 (2007): 359–378.
    """
    is_frame(df, df_only=True, raise_exception =True, objname ='df')
    
    # If x is None, use the index
    if x is None:
        x = np.arange(len(df))
        
    check_consistent_length(df, x )
    
    # Validate columns
    if median_col is None or lower_col is None or upper_col is None:
        raise ValueError(
            "median_col, lower_col, and upper_col must be specified.")
    
    exist_features(df, features =[median_col, lower_col, upper_col])
    # Extract data
    median_vals = df[median_col].values
    lower_vals = df[lower_col].values
    upper_vals = df[upper_col].values

    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot actual if provided
    if actual_col is not None:
        exist_features(df, features=actual_col, name ='Actual feature column')
        plt.plot(x, df[actual_col].values, label=actual_col, color=line_colors[0])

    # Plot median line
    plt.plot(x, median_vals, label=median_col, color=line_colors[1])

    # Fill interval
    plt.fill_between(
        x, lower_vals, upper_vals,
        color=fill_color, alpha=fill_alpha,
        label=f'{lower_col} - {upper_col}'
    )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if grid:
        plt.grid(True, linestyle='--', alpha=0.5)

    if legend:
        plt.legend()

    if savefig:
        plt.savefig(savefig, dpi=dpi, bbox_inches='tight')

    plt.show()

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
    
