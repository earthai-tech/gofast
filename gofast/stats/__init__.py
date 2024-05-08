# -*- coding: utf-8 -*-

from .utils import (
    anova_test,
    bootstrap,
    check_and_fix_rm_anova_data,
    chi2_test,
    correlation,
    corr,
    cronbach_alpha,
    describe,
    dca_analysis,
    friedman_test,
    get_range,
    gini_coeffs,
    hmean,
    kaplan_meier_analysis,
    kruskal_wallis_test,
    kolmogorov_smirnov_test,
    kurtosis,
    levene_test,
    mcnemar_test,
    mean,
    median,
    mds_similarity,
    mixed_effects_model, 
    mode,
    perform_kmeans_clustering,
    perform_linear_regression,
    perform_spectral_clustering,
    paired_t_test,
    quartiles,
    skew,
    std,
    statistical_tests,
    t_test_independent,
    var,
    wmedian,
    wilcoxon_signed_rank_test,
    z_scores,
)

__all__ = [
    "anova_test",
    "bootstrap",
    "check_and_fix_rm_anova_data",
    "chi2_test",
    "correlation",
    "corr",
    "cronbach_alpha",
    "describe",
    "dca_analysis",
    "friedman_test",
    "get_range",
    "gini_coeffs",
    "hmean",
    "kaplan_meier_analysis",
    "kruskal_wallis_test",
    "kolmogorov_smirnov_test",
    "kurtosis",
    "levene_test",
    "mcnemar_test",
    "mean",
    "median",
    "mds_similarity",
    "mixed_effects_model", 
    "mode",
    "perform_kmeans_clustering",
    "perform_linear_regression",
    "perform_spectral_clustering",
    "paired_t_test",
    "quartiles",
    "skew",
    "std",
    "statistical_tests",
    "t_test_independent",
    "var",
    "wmedian",
    "wilcoxon_signed_rank_test",
    "z_scores",
]

# To organize the functions from the `gofast/stats` package into more intuitive and programmatic module names, you can group them based on their statistical purposes or functionality. Here’s a proposed module organization, with each module containing functions related to specific statistical operations or tests:

# 1. **Descriptive Statistics**: Functions that summarize or describe features of a dataset.
#    - **Module Name**: `descriptive_stats.py`
#    - **Functions**:
#      - `describe`
#      - `get_range`
#      - `hmean`
#      - `iqr`
#      - `mean`
#      - `median`
#      - `mode`
#      - `quantile`
#      - `quartiles`
#      - `std`
#      - `var`
#      - `wmedian`
#      - `skew`
#      - `kurtosis`
#      - `gini_coeffs`

# 2. **Inferential Statistics**: Functions for performing statistical tests to make inferences or generalizations about a population.
#    - **Module Name**: `inferential_stats.py`
#    - **Functions**:
#      - `anova_test`
#      - `bootstrap`
#      - `check_and_fix_rm_anova_data`
#      - `check_anova_assumptions`
#      - `chi2_test`
#      - `cronbach_alpha`
#      - `friedman_test`
#      - `kolmogorov_smirnov_test`
#      - `kruskal_wallis_test`
#      - `levene_test`
#      - `mcnemar_test`
#      - `mixed_effects_model`
#      - `paired_t_test`
#      - `t_test_independent`
#      - `wilcoxon_signed_rank_test`
#      - `statistical_tests`

# 3. **Correlation and Regression Analysis**: Functions that deal with the relationships among variables.
#    - **Module Name**: `correlation_regression.py`
#    - **Functions**:
#      - `corr`
#      - `correlation`
#      - `perform_linear_regression`
#      - `perform_kmeans_clustering`
#      - `perform_spectral_clustering`
#      - `mds_similarity`

# 4. **Survival and Reliability Analysis**: Functions for analyzing time-to-event data.
#    - **Module Name**: `survival_reliability.py`
#    - **Functions**:
#      - `kaplan_meier_analysis`
#      - `dca_analysis`

# 5. **Utility and Advanced Statistical Methods**: Functions that don't neatly fit into the other categories or are used across different types of analyses.
#    - **Module Name**: `stat_utils.py`
#    - **Functions**:
#      - `z_scores`

# This structure helps in maintaining a clear separation of concerns, making it 
# easier for developers to locate and manage the code. Each module is focused 
# on a specific area of statistics, reducing complexity and improving maintainability.

