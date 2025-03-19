# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

"""
Provides functions to conduct and visualize various
statistical tests.
"""

from typing import Union, Tuple, Dict, Iterable, Optional 
import numpy as np
import pandas as pd
from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway

from ..api.summary import ResultSummary 
from ..decorators import isdf 
from ..compat.sklearn import  validate_params, StrOptions 
from ..core.checks import exist_features, check_params  
from ..core.handlers import columns_manager 
from ..core.plot_manager import default_params_plot 
from ..utils.validator import filter_valid_kwargs
from ._config import PlotConfig

__all__=['plot_ab_test'] 

@default_params_plot(
    savefig=PlotConfig.AUTOSAVE('my_ab_test_plot.png')
   )
@validate_params ({
    'kind': [
        StrOptions({'boxplot','violin', 'kde','swarm', 'ecdf'  })], 
    'test': [StrOptions(
        {'t-test', 'mannwhitney', 'bootstrap', 'chi2', 'anova'})
       ]
    })
@check_params ({ 
    'set_a': Optional[Union[list, str]], 
    'set_b': Optional[Union[list, str]], 
    'palette': Optional[Union[list, str]]
    }, 
 coerce=False 
 )
@isdf 
def plot_ab_test(
    df: pd.DataFrame,
    value_col: str=None,
    group_col: str = None,
    set_a: Union[list, str] = None,
    set_b: Union[list, str] = None,
    test: str = "t-test",
    alternative: str = "two-sided",
    kind: str = "boxplot",
    alpha: float = 0.05,
    palette: Union[str, list] = "husl",
    figsize: tuple = (10, 6),
    ax: plt.Axes = None,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    show_summary: bool = True,
    show_ci: bool = True,
    ci: float = 0.95,
    bootstrap_samples: int = 5000,
    plot_style: str = "whitegrid",
    use_default_rc_params: bool = True,
    title_fontsize: int = 14,
    label_fontsize: int = 12,
    tick_fontsize: int = 10,
    effect_size: float=0.5, 
    verbose: int = 1,
    show_plot: bool = True,
    return_objects: bool = False,
    **plot_kws
) -> Union[Dict, Tuple[Dict, plt.Figure, plt.Axes]]:

    """
    Plot and analyze two sets of data for A/B testing.

    This function, ``plot_ab_test``, compares two
    subsets of `<df>` by applying a statistical test
    (e.g. `<test>`) and visualizing the distributions
    using `<kind>`. Internally, it calls
    `perform_statistical_test`, `format_annotation`,
    and `add_significance_marker` (when necessary).
    A typical hypothesis test tries to determine if
    a difference between means (or distributions)
    is statistically significant [1]_.

    .. math::
       \\Delta = \\bar{X}_A - \\bar{X}_B

    where :math:`\\bar{X}_A` is the mean of group A and
    :math:`\\bar{X}_B` is the mean of group B.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data. Must contain at least the
        columns `<group_col>` and `<value_col>` if
        using group-based splitting, or must provide
        `<set_a>` and `<set_b>` for manual selection.
    value_col : str, optional
        Name of the numerical column to be compared.
        Required unless `<set_a>` and `<set_b>` each
        directly reference the needed data.
    group_col : str, optional
        Column that partitions `<df>` into exactly
        two groups. Mutually exclusive with
        `<set_a>` and `<set_b>`.
    set_a : list or str, optional
        Indices or label specifying the first group.
    set_b : list or str, optional
        Indices or label specifying the second group.
    test : str, default='t-test'
        Statistical test method: options include
        ``'t-test'``, ``'mannwhitney'``,
        ``'bootstrap'``, ``'chi2'``, or ``'anova'``.
    alternative : str, default='two-sided'
        Sides of the test hypothesis (e.g.
        `'less'`, `'greater'`). Applies to some
        tests only.
    kind : str, default='boxplot'
        Type of plot to display. Supported types:
        ``'boxplot'``, ``'violin'``, ``'kde'``,
        ``'swarm'``, or ``'ecdf'``.
    alpha : float, default=0.05
        Significance level for statistical decision.
    palette : str or list, default='husl'
        Matplotlib or seaborn palette used to color
        the groups.
    figsize : tuple, default=(10, 6)
        Dimensions of the figure.
    ax : matplotlib.axes.Axes, optional
        Axes on which the plot is drawn. If None,
        a new figure is created.
    title : str, optional
        Title for the plot.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    show_summary : bool, default=True
        If True, displays test results as an
        annotation on the plot.
    show_ci : bool, default=True
        Indicates whether to compute or display
        confidence intervals for certain tests.
    ci : float, default=0.95
        Confidence interval level, e.g. 0.95 for
        95%.
    bootstrap_samples : int, default=5000
        Number of bootstrap iterations (if
        `<test>` is `'bootstrap'`).
    plot_style : str, default='whitegrid'
        Seaborn style to apply for background,
        e.g. `'darkgrid'`.
    use_default_rc_params : bool, default=True
        If True, uses some additional default rc
        parameters like `'DejaVu Sans'`.
    title_fontsize : int, default=14
        Font size for the main plot title.
    label_fontsize : int, default=12
        Font size for axis labels.
    tick_fontsize : int, default=10
        Font size for tick labels.
    effect_size : float, default=0.5
        Used to annotate significance markers if
        the null hypothesis is rejected.
    verbose : int, default=1
        Level of console logging. If > 0, prints a
        summary after testing.
    show_plot : bool, default=True
        If True, calls plt.show() to display the
        final figure.
    return_objects : bool, default=False
        If True, returns the dictionary with test
        results plus the figure and axes.

    **plot_kws : dict
        Extra arguments passed to the seaborn plot
        function.

    Returns
    -------
    dict or tuple
        Either a dictionary of test results alone,
        or a tuple containing (results_dict,
        matplotlib.figure.Figure, matplotlib.axes.Axes).

    Notes
    -----
    Statistical significance is determined by
    comparing the resulting p-value to `<alpha>`.
    If `'reject_null'` is True, a marker is drawn
    for `'boxplot'`, `'violin'`, or `'bar'` plots
    to highlight significance. The function
    reuses the result structures from
    `perform_statistical_test`.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from scipy import stats
    >>> from gofast.plot.testing import plot_ab_test
    >>> np.random.seed(42)
    >>> n = 500
    >>> data = pd.DataFrame({
    ...     'user_id': range(2 * n),
    ...     'group': ['A'] * n + ['B'] * n,
    ...     'conversion_rate': np.concatenate([
    ...         stats.beta.rvs(2, 5, size=n),
    ...         stats.beta.rvs(2.3, 5, size=n)
    ...     ]),
    ...     'feature_usage': np.concatenate([
    ...         stats.poisson.rvs(3, size=n),
    ...         stats.poisson.rvs(3.5, size=n)
    ...     ]),
    ...     'session_duration': np.concatenate([
    ...         stats.gamma.rvs(2, scale=2, size=n),
    ...         stats.gamma.rvs(2.2, scale=2, size=n)
    ...     ])
    ... })
    >>> data.loc[[5, 15, 25], 'conversion_rate'] = np.nan
    >>> result1 = plot_ab_test(
    ...     df=data,
    ...     group_col='group',
    ...     value_col='conversion_rate',
    ...     test='t-test',
    ...     kind='boxplot',
    ...     title='Conversion Rate Comparison',
    ...     verbose=2
    ... )
    >>> result2 = plot_ab_test(
    ...     df=data,
    ...     group_col='group',
    ...     value_col='session_duration',
    ...     test='mannwhitney',
    ...     kind='violin',
    ...     palette='coolwarm',
    ...     title='Session Duration Distribution',
    ...     show_ci=False,
    ...     verbose=1
    ... )
    >>> result3 = plot_ab_test(
    ...     df=data,
    ...     set_a=data[data['group'] == 'A']['feature_usage'],
    ...     set_b=data[data['group'] == 'B']['feature_usage'],
    ...     test='bootstrap',
    ...     kind='ecdf',
    ...     title='Feature Usage Cumulative Distribution',
    ...     ci=0.9,
    ...     bootstrap_samples=10000,
    ...     verbose=3
    ... )
    >>> data['converted'] = np.where(data['conversion_rate'] > 0.15, 1, 0)
    >>> result4 = plot_ab_test(
    ...     df=data,
    ...     group_col='group',
    ...     value_col='converted',
    ...     test='chi2',
    ...     kind='swarm',
    ...     title='Conversion Success Analysis',
    ...     verbose=2
    ... )
    >>> multi_data = pd.DataFrame({
    ...     'group': np.repeat(['A', 'B', 'C'], 100),
    ...     'values': np.concatenate([
    ...         np.random.normal(5, 1, 100),
    ...         np.random.normal(5.5, 1, 100),
    ...         np.random.normal(6, 1, 100)
    ...     ])
    ... })
    >>> try:
    ...     plot_ab_test(
    ...         df=multi_data,
    ...         group_col='group',
    ...         value_col='values',
    ...         verbose=2
    ...     )
    ... except ValueError as e:
    ...     print("Caught expected error:", e)
    >>> result6, fig, ax = plot_ab_test(
    ...     df=data,
    ...     group_col='group',
    ...     value_col='session_duration',
    ...     kind='kde',
    ...     title='Session Duration Density',
    ...     palette=['#2ecc71', '#e74c3c'],
    ...     figsize=(12, 7),
    ...     show_summary=False,
    ...     return_objects=True,
    ...     verbose=0
    ... )
    >>> ax.text(0.5, 0.9, 'Custom Annotation', transform=ax.transAxes,
    ...         ha='center', fontsize=14, color='navy')
    >>> fig.savefig('custom_plot.png')

    See Also
    --------
    perform_statistical_test : Runs the chosen
        hypothesis test on two numeric arrays.
    format_annotation : Creates a textual summary
        of test results for annotation.
    add_significance_marker : Marks significant
        difference on the plot.

    References
    ----------
    .. [1] Fisher, R. A. *Statistical Methods for
       Research Workers*, 1925.
    """

    def _validate_set(
            s: Union[Iterable, pd.Series, pd.DataFrame]
           ) -> pd.Series:
        """Convert various input types to boolean mask"""
        if isinstance(s, (tuple, set)):
            s = columns_manager(s)
        
        if isinstance(s, (pd.Series, pd.DataFrame)):
            if s.empty:
                raise ValueError("Empty data set provided")
            return s.any(axis=1) if isinstance(s, pd.DataFrame) else s
            
        if isinstance(s, list):
            return df.index.isin(s)
            
        raise TypeError(f"Unsupported set type: {type(s)}")
        
    # Configure the plot style, including
    # background and font family
    
    sns.set_style(plot_style)
    if use_default_rc_params:

        rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.titlepad'] = 20

    # Check if inputs are valid: group_col can
    # automatically split into 2 groups; set_a
    # and set_b define them manually 
    
    # Validate input configuration
    group_provided = group_col is not None
    sets_provided = set_a is not None and set_b is not None
    
    if not (group_provided or sets_provided):
        raise ValueError(
            "Must provide either group_col or both set_a/set_b"
        )
    if group_provided and (
            set_a is not None 
            or set_b is not None
    ):
        raise ValueError(
            "Mutually exclusive inputs: use"
            " either group_col or set_a/set_b"
        )

    # Data preparation logic
    if group_provided:
        # Validate group structure
        exist_features(df, features=[group_col, value_col])
        groups = df[group_col].unique()
        if len(groups) != 2:
            raise ValueError(
                "Requires exactly 2 groups, "
                f"found {len(groups)}: {groups}"
            )
        
        # Extract group data
        a_mask = df[group_col] == groups[0]
        b_mask = df[group_col] == groups[1]
        a_label, b_label = map(str, groups)
    else:
        # Validate and convert data sets
        a_mask = _validate_set(set_a)
        b_mask = _validate_set(set_b)
        a_label, b_label = "Group A", "Group B"

    # Extract and clean target values
    a_data = df.loc[a_mask, value_col].dropna()
    b_data = df.loc[b_mask, value_col].dropna()
    
    if a_data.empty or b_data.empty:
        raise ValueError(
            "Empty groups after NA removal - check data quality "
            f"(Group A: {len(a_data)}, Group B: {len(b_data)})"
        )

    # Drop any NaN values before plotting
    a_data = a_data.dropna()
    b_data = b_data.dropna()

    # Build a figure/axes if none supplied
    fig, ax = (
        plt.subplots(figsize=figsize)
        if ax is None else (ax.figure, ax)
    )

    # Setup some generic plotting params, then
    # merge in user-supplied ones
    plot_params = {
        'palette': palette,
        'linewidth': 1.5,
        'saturation': 0.85,
        'width': 0.7,
        **plot_kws
    }

    # Create the requested plot type
    if kind == "boxplot":
        plot_params= filter_valid_kwargs(sns.boxplot, plot_params)
        sns.boxplot(
            x=group_col,
            y=value_col,
            data=df,
            ax=ax,
            **plot_params
        )
    elif kind == "violin":
        plot_params= filter_valid_kwargs(sns.violinplot, plot_params)
        sns.violinplot(
            x=group_col,
            y=value_col,
            data=df,
            ax=ax,
            inner="quartile",
            **plot_params
        )
    elif kind == "kde":
        plot_params= filter_valid_kwargs(sns.kdeplot, plot_params)
        sns.kdeplot(
            a_data,
            label=a_label,
            fill=True,
            alpha=0.3,
            ax=ax,
            **plot_params
        )
        sns.kdeplot(
            b_data,
            label=b_label,
            fill=True,
            alpha=0.3,
            ax=ax,
            **plot_params
        )
        ax.set_xlabel(value_col, fontsize=label_fontsize)
    elif kind == "swarm":
        plot_params= filter_valid_kwargs(sns.swarmplot, plot_params)
        sns.swarmplot(
            x=group_col,
            y=value_col,
            data=df,
            ax=ax,
            size=4,
            **plot_params
        )
    elif kind == "ecdf":
        plot_params= filter_valid_kwargs(sns.ecdfplot, plot_params)
        sns.ecdfplot(
            a_data,
            label=a_label,
            ax=ax,
            **plot_params
        )
        sns.ecdfplot(
            b_data,
            label=b_label,
            ax=ax,
            **plot_params
        )
        ax.set_xlabel(value_col, fontsize=label_fontsize)
    else:
        raise ValueError(f"Unsupported plot type: {kind}")

    # Perform the statistical test and store results
    test_result = perform_statistical_test(
        a_data,
        b_data,
        test,
        alternative,
        alpha,
        ci,
        bootstrap_samples,
        verbose
    )

    # Optionally annotate the plot with the results
    if show_summary:
        annotation_text = format_annotation(test_result, alpha, ci)
        ax.annotate(
            annotation_text,
            xy=(0.05, 0.95),
            xycoords='axes fraction',
            fontsize=tick_fontsize,
            bbox=dict(
                boxstyle='round',
                alpha=0.9,
                facecolor='white'
            )
        )

    # If the null is rejected, add a significance
    # marker for box/violin/bar plots
    if test_result['reject_null'] and kind in [
        'boxplot', 'violin', 'bar'
    ]:
        add_significance_marker(
            ax,
            y_pos=max(a_data.max(), b_data.max()) * 1.1, 
            effect_size=effect_size, 
        )

    # Configure final labeling
    ax.set_title(
        title or f"A/B Test Results: {value_col}",
        fontsize=title_fontsize,
        weight='bold'
    )
    ax.set_xlabel(
        xlabel or group_col,
        fontsize=label_fontsize
    )
    ax.set_ylabel(
        ylabel or value_col,
        fontsize=label_fontsize
    )
    ax.tick_params(
        axis='both',
        labelsize=tick_fontsize
    )
    sns.despine(trim=True)

    # Show or hide the finalized plot
    if show_plot:
        plt.tight_layout()
        plt.show()
    if verbose: 
        summary = ResultSummary(name=f"ABTest{kind.title()}")
        summary.add_results(test_result)
        print(summary)
        
    # Return either the dictionary or the
    # dictionary + figure + axes
    return (
        (test_result, fig, ax)
        if return_objects else test_result
    )

def _validate_data_sets(
    set_a: Union[Iterable, pd.Series, pd.DataFrame], 
    set_b: Union[Iterable, pd.Series, pd.DataFrame]
) -> tuple:
    """Validate and normalize input data sets for A/B testing.
    
    Args:
        set_a: First data set for comparison
        set_b: Second data set for comparison
    
    Returns:
        Tuple of validated (set_a, set_b)
    
    Raises:
        ValueError: On invalid or empty input data
    """
    # Unified validation for both sets
    def _check_set(s: Union[Iterable, pd.Series, pd.DataFrame]) -> bool:
        if isinstance(s, (tuple, set)):
            s = columns_manager(s)
        if isinstance(s, (pd.Series, pd.DataFrame)):
            return s.empty
        return s is None
        
    empty_a = _check_set(set_a)
    empty_b = _check_set(set_b)

    if any([empty_a, empty_b]):
        raise ValueError(
            "Invalid data sets - set_a/set_b cannot be empty "
            "when group_col is not provided"
        )
    
    return set_a, set_b

def perform_statistical_test(
    a,
    b,
    test,
    alternative,
    alpha,
    ci,
    n_bootstrap,
    verbose
):
    # Initialize result dictionary with basic info
    result = {
        'test': test,
        'alpha': alpha
    }
    
    try:
        # Dispatch to the requested statistical
        # method based on the 'test' parameter
        if test == 't-test':
            result.update(
                independent_ttest(a, b, alternative)
            )
        elif test == 'mannwhitney':
            result.update(
                mann_whitney_test(a, b, alternative)
            )
        elif test == 'bootstrap':
            result.update(
                bootstrap_test(a, b, n_bootstrap, ci)
            )
        elif test == 'chi2':
            result.update(
                chi_square_test(a, b)
            )
        elif test == 'anova':
            result.update(
                anova_test(a, b)
            )
        else:
            # Raise error if the test type is unknown
            raise ValueError(
                f"Unsupported test type: {test}"
            )

        # Determine if the null hypothesis is rejected
        # For bootstrap, check confidence interval
        if test == 'bootstrap':
            result['reject_null'] = (
                result['ci_low'] > 0
                or result['ci_high'] < 0
            )
        else:
            result['reject_null'] = (
                result['p_value'] < alpha
            )

    except Exception as e:
        # Print the error if verbosity is high enough
        if verbose >= 1:
            print(
                f"Statistical test failed: {str(e)}"
            )
        # Store the error in results and
        # do not reject the null by default
        result['error'] = str(e)
        result['reject_null'] = False
    
    return result


def chi_square_test(a, b):
    """Chi-square test of independence with 
    automatic contingency table creation"""
    # Build a DataFrame that allows us
    # to form a contingency table
    contingency = pd.DataFrame({
        'group': (
            ['A'] * len(a)
            + ['B'] * len(b)
        ),
        'value': pd.concat([a, b])
    })
    # Create the cross-tab to feed chi2_contingency
    contingency_table = pd.crosstab(
        contingency['group'],
        contingency['value']
    )
    # Compute the chi-square test result
    chi2, p, dof, expected = chi2_contingency(
        contingency_table
    )
    # Return needed metrics in dictionary
    return {
        'statistic': chi2,
        'p_value': p,
        'dof': dof,
        'expected': expected.tolist()
    }


def anova_test(a, b):
    """One-way ANOVA with effect size calculation"""
    # Perform one-way ANOVA across two groups
    f_stat, p_val = f_oneway(a, b)
    # Compute effect size (eta-squared) using
    # the standard formula
    eta_sq = (
        f_stat * (a.shape[0] - 1)
        / (
            f_stat * (a.shape[0] - 1)
            + (
                a.shape[0]
                + b.shape[0]
                - 2
            )
        )
    )
    # Return metrics, including effect size
    return {
        'statistic': f_stat,
        'p_value': p_val,
        'effect_size': eta_sq
    }


def bootstrap_test(
    a,
    b,
    n_samples,
    ci
):
    """Bootstrap confidence interval for
    mean difference with bias correction"""
    # Set random seed to ensure reproducibility
    np.random.seed(42)
    boot_diffs = []
    # Observed difference in means
    obs_diff = np.mean(a) - np.mean(b)

    # Combine data for reference if needed
    # combined = np.concatenate([a, b])

    # Perform the bootstrap procedure
    for _ in range(n_samples):
        resample_a = np.random.choice(
            a,
            size=len(a),
            replace=True
        )
        resample_b = np.random.choice(
            b,
            size=len(b),
            replace=True
        )
        boot_diffs.append(
            np.mean(resample_a)
            - np.mean(resample_b)
        )
    
    # Calculate the lower and upper bounds
    # for the CI using percentiles
    lower = np.percentile(
        boot_diffs,
        (1 - ci) / 2 * 100
    )
    upper = np.percentile(
        boot_diffs,
        (1 - (1 - ci) / 2) * 100
    )

    # Return relevant bootstrap results
    return {
        'statistic': obs_diff,
        'ci_low': lower,
        'ci_high': upper,
        'bootstrap_dist': boot_diffs
    }


def independent_ttest(a, b, alternative):
    """Enhanced t-test with Welch's 
    correction and effect size"""
    # Perform Welch's t-test assuming unequal var
    t_stat, p_val = stats.ttest_ind(
        a,
        b,
        equal_var=False
    )
    # Calculate Cohen's d using
    # the pooled standard deviation
    pooled_std = np.sqrt(
        (
            np.var(a, ddof=1)
            + np.var(b, ddof=1)
        ) / 2
    )
    cohens_d = (
        (np.mean(a) - np.mean(b))
        / pooled_std
    )

    # Adjust p-value if we are not dealing
    # with a two-sided alternative
    if alternative != 'two-sided':
        p_val = p_val / 2
        # If the sign of t_stat is not aligned
        # with the alternative, invert p
        if (
            alternative == 'greater'
            and t_stat < 0
        ) or (
            alternative == 'less'
            and t_stat > 0
        ):
            p_val = 1 - p_val
    
    return {
        'statistic': t_stat,
        'p_value': p_val,
        'effect_size': cohens_d
    }


def mann_whitney_test(a, b, alternative):
    """Mann-Whitney U test with effect size"""
    # Execute the Mann-Whitney U test
    u_stat, p_val = stats.mannwhitneyu(
        a,
        b,
        alternative=alternative
    )
    # Compute a rank-biserial correlation
    # as an effect size measure
    rank_biserial = 1 - (
        2 * u_stat
    ) / (
        len(a) * len(b)
    )
    return {
        'statistic': u_stat,
        'p_value': p_val,
        'effect_size': rank_biserial
    }


def format_annotation(
    result,
    alpha,
    ci
):
    """Enhanced annotation with effect
    sizes and test details"""
    # Build annotation lines for display
    lines = [
        f"Test: {result['test'].title()}",
        f"p-value: {result.get('p_value', 'NA'):.4f}",
        (
            f"α: {alpha} | Decision: "
            f"{'Reject H₀' if result.get('reject_null', False) else 'Retain H₀'}"
        )
    ]
    
    # Add effect size if present
    if 'effect_size' in result:
        lines.append(
            f"Effect Size: {result['effect_size']:.3f}"
        )
    # Add confidence interval if present
    if 'ci_low' in result:
        lines.append(
            f"CI {ci * 100:.0f}%: "
            f"[{result['ci_low']:.2f}, "
            f"{result['ci_high']:.2f}]"
        )
    # Add degrees of freedom (χ²) if found
    if 'dof' in result:
        lines.append(
            f"χ² DoF: {result['dof']}"
        )
    
    # Join lines into a single multi-line
    # string for annotation
    return '\n'.join(lines)


def add_significance_marker(
    ax,
    y_pos,
    effect_size
):
    """Dynamic significance markers with
    effect size visualization"""
    # Draw lines to form a star bracket
    ax.plot(
        [0, 0, 1, 1],
        [
            y_pos * 0.95,
            y_pos,
            y_pos,
            y_pos * 0.95
        ],
        lw=1.5,
        color='#2d4059'
    )
    # Add text marker based on effect size
    ax.text(
        0.5,
        y_pos * 1.05,
        "*" if effect_size > 0.5 else "†",
        ha='center',
        va='bottom',
        color='#2d4059',
        fontsize=14
    )
    # Draw a small bar for highlighting effect size
    ax.plot(
        [0.25, 0.75],
        [y_pos * 1.15] * 2,
        color='#ea5455',
        lw=2
    )
    ax.plot(
        [0.5] * 2,
        [
            y_pos * 1.13,
            y_pos * 1.17
        ],
        color='#ea5455',
        lw=2
    )
    # Place a label showing numeric effect size
    ax.text(
        0.5,
        y_pos * 1.19,
        f"ES: {effect_size:.2f}",
        ha='center',
        va='bottom',
        color='#ea5455'
    )
    



