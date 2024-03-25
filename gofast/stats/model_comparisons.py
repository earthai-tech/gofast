# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

import itertools
import numpy as np 
import pandas as pd
from scipy.stats import friedmanchisquare
from scipy.stats import wilcoxon, chi2
from scipy.stats import rankdata
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

from .._typing import Callable, LambdaType, DataFrame, Series, Array1D
from .._typing import Dict, List, Union, Optional, Any, Tuple  
from ..api.formatter import DataFrameFormatter 
from ..decorators import isdf 
from ..tools.coreutils import validate_ratio 
from ..tools.funcutils import ensure_pkg 
from ..tools.validator import _is_arraylike_1d


@isdf 
def perform_friedman_test(
    model_performance_data: DataFrame, 
    alpha:float=0.05,
    perform_posthoc: bool=False,
    posthoc_test:Optional[Union[Callable, LambdaType ]]=None
    ):
    """
    Performs the Friedman test on multiple classifiers over multiple datasets.

    The Friedman test is a non-parametric statistical test used to detect 
    differences in treatments across multiple test attempts. In the context 
    of machine learning, it is used to compare the performance of different 
    algorithms across various datasets. If significant differences are found, 
    post-hoc tests can further investigate these differences.

    Parameters
    ----------
    model_performance_data : pd.DataFrame
        A DataFrame where each row represents a dataset and each column represents
        a model. Each cell in the DataFrame should contain the performance metric
        (e.g., accuracy) of the model on that dataset.
    alpha : float, optional
        Significance level used to determine the critical value. Defaults to 0.05.
    perform_posthoc : bool, optional
        Indicates whether to perform post-hoc testing if the Friedman test 
        indicates significant differences. Defaults to False.
    posthoc_test : callable, optional
        A function to perform post-hoc testing. It should accept the 
        `model_performance_data` DataFrame as input and return post-hoc test 
        results. Required if `perform_posthoc` is True.

    Returns
    -------
    result : dict
        A dictionary containing the Friedman test statistic, degrees of freedom,
        p-value, and whether there is a significant difference. If perform_posthoc
        is True, it also includes the results of the post-hoc test under 
        "Post-hoc Test".

    Notes
    -----
    The Friedman test is computed as follows:

    .. math::
        Q = \\frac{12N}{k(k+1)}[\\sum_j R_j^2 - \\frac{k(k+1)^2}{4}]

    Where:
    - :math:`N` is the number of datasets.
    - :math:`k` is the number of models.
    - :math:`R_j` is the sum of ranks for the j-th model.

    The null hypothesis of the Friedman test states that all algorithms are 
    equivalent and any observed differences are due to chance. A significant 
    p-value rejects this hypothesis, suggesting that at least one algorithm 
    performs differently.

    The test is useful in scenarios where multiple models' performances are 
    evaluated across multiple datasets. It is robust against non-normal 
    distributions and unequal variances among groups.

    Examples
    --------
    >>> from gofast.stats.model_comparisons import perform_friedman_test
    >>> import pandas as pd
    >>> # Sample data: Model performance on 5 datasets (rows) for 3 models (columns)
    >>> df = {
    ...    'Model_A': [0.8, 0.82, 0.78, 0.81, 0.79],
    ...    'Model_B': [0.79, 0.84, 0.76, 0.82, 0.78],
    ...    'Model_C': [0.81, 0.83, 0.77, 0.80, 0.76]
    ... }
    >>> result = perform_friedman_test(df)
    >>> print(result)

    If the result indicates a significant difference and post-hoc 
    testing is desired:

    >>> from some_module import some_posthoc_test
    >>> result = perform_friedman_test(df, perform_posthoc=True, 
    ...                                 posthoc_test=some_posthoc_test)
    >>> print(result["Post-hoc Test"])

    """
    if not isinstance(model_performance_data, pd.DataFrame):
        raise TypeError("model_performance_data must be a pandas DataFrame.")

    if model_performance_data.empty:
        raise ValueError("model_performance_data DataFrame cannot be empty.")

    # Checking for any NaN values in DataFrame
    if model_performance_data.isnull().values.any():
        raise ValueError(
            "model_performance_data DataFrame contains NaN values. Please clean your data.")
        
    alpha = validate_ratio( alpha, bounds=(0, 1), to_percent= True, exclude=0 )
    # Perform the Friedman test
    ranks = np.array([rankdata(-1 * performance) for performance in model_performance_data.values])
    n_datasets, n_models = ranks.shape
    rank_sums = ranks.sum(axis=0)
    statistic = ( 
        (12 / (n_datasets*n_models*(n_models+1)) * np.sum(rank_sums**2)) - (3*n_datasets*(n_models+1))
        )
    df = n_models - 1
    p_value = 1 - chi2.cdf(statistic, df)

    result = {
        "Friedman Test Statistic": statistic,
        "Degrees of Freedom": df,
        "p-value": p_value,
        "Significant Difference": p_value < alpha
    }

    # Optionally perform post-hoc tests if significant differences are detected
    if perform_posthoc and result["Significant Difference"] and posthoc_test is not None:
        posthoc_result = posthoc_test(model_performance_data)
        result["Post-hoc Test"] = posthoc_result
    
    result= DataFrameFormatter("Friedman Test Results").add_df (result)
    return result

@isdf 
@ensure_pkg("scikit-posthocs")
def perform_nemenyi_posthoc_test(
    model_performance_data: DataFrame, 
    significance_level:float=0.05, 
    rank_method:str='average'
    ):
    """
    Conducts the Nemenyi post-hoc test for pairwise comparisons of multiple
    classification models across various datasets. This test is applied after
    the Friedman test indicates significant differences in model performances.

    Parameters
    ----------
    model_performance_data : pd.DataFrame
        A DataFrame where rows represent datasets and columns represent
        classification models. The cell values are performance metrics
        (e.g., accuracy).
    significance_level : float, optional
        The significance level for detecting differences, by default 0.05.
    method : {'average', 'min', 'max', 'dense', 'ordinal'}, optional
        The method used to rank the models in case of ties, by default 'average'.

    Returns
    -------
    dict
        A dictionary containing three key-value pairs:
        'p_values' (pd.DataFrame): Pairwise p-values from the Nemenyi test.
        'significant_differences' (pd.DataFrame): A boolean DataFrame indicating
        significant differences between model pairs.
        'average_ranks' (pd.Series): The average ranks of the models.

    Notes
    -----
    The Nemenyi test is a non-parametric pairwise comparison test that is used
    when the Friedman test has rejected the null hypothesis of equal
    performances. It compares all pairs of models to find which pairs have
    significantly different performances. The test statistic for comparing two
    models i and j is given by:

    .. math::
        q = \frac{|R_i - R_j|}{\sqrt{\frac{k(k+1)}{6N}}}

    where :math:`R_i` and :math:`R_j` are the average ranks of models i and j,
    k is the number of models, and N is the number of datasets. The critical
    value for q can be found in the studentized range distribution table for
    infinite degrees of freedom.

    The Nemenyi test is useful when you want to understand which specific models
    perform differently from each other, providing insights beyond the overall
    comparison of the Friedman test.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.stats.model_comparisons import perform_nemenyi_posthoc_test
    >>> model_performance = pd.DataFrame({
    ...     'Model A': [0.85, 0.80, 0.82],
    ...     'Model B': [0.76, 0.78, 0.75],
    ...     'Model C': [0.89, 0.85, 0.86]
    ... })
    >>> results = perform_nemenyi_posthoc_test(model_performance)
    >>> print(results['p_values'])
    >>> print(results['significant_differences'])
    >>> print(results['average_ranks'])
    """

    import scikit_posthocs as sp
    import pandas as pd
    
    significance_level= validate_ratio(
        significance_level, bounds=(0, 1), to_percent= True, exclude=0 )

    # Validate input
    if not isinstance(model_performance_data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    # Calculate ranks if not already provided
    ranks = model_performance_data.rank(axis=1, method=rank_method)
    avg_ranks = ranks.mean()

    # Perform the Nemenyi post-hoc test
    pairwise_p_values = sp.posthoc_nemenyi_friedman(ranks)

    # Identify significant differences
    significant_diffs = pairwise_p_values < significance_level

    # Prepare output
    results = {
        'p_values': pairwise_p_values,
        'significant_differences': significant_diffs,
        'average_ranks': avg_ranks
    }
    results= DataFrameFormatter("Nemenyi Test Results").add_df (results)
    
    return results

@isdf 
def perform_wilcoxon_test(
    model_performance_data: DataFrame, 
    alpha: float=0.05, 
    mask_non_significant: bool=False
    ):
    """
    Applies the Wilcoxon signed-rank test to assess the statistical significance
    of the difference between the performance metrics of pairs of models across
    multiple datasets.

    Parameters
    ----------
    model_performance_data : pd.DataFrame
        A pandas DataFrame where each row represents a dataset and
        each column a different model. The cell values are the performance
        metrics (e.g., accuracy, F1 score) of the models on the datasets.
        This data is used to perform pairwise comparisons between models
        to determine if there are statistically significant differences
        in their performances.
        
    alpha : float, optional
        The significance level used to determine the threshold for
        statistically significant differences between model performances.
        Defaults to 0.05. A lower alpha value requires stronger evidence
        of a difference between model performances to reject the null
        hypothesis (that there is no difference).
        
    mask_non_significant : bool, optional
        If set to True, p-values greater than the significance level (alpha)
        are masked or replaced with NaN, highlighting only the statistically
        significant comparisons. Defaults to False, which displays all
        p-values, allowing users to see the full results of the comparisons.

    Returns
    -------
    pd.DataFrame
        A DataFrame where each cell in row i and column j contains the p-value
        resulting from the Wilcoxon signed-rank test comparing the performance of
        model i and model j across all datasets. A p-value less than `alpha`
        indicates a statistically significant difference in performance between
        the two models.

    Notes
    -----
    The Wilcoxon signed-rank test is a non-parametric statistical hypothesis test
    used when comparing two related samples or repeated measurements on a single
    sample to assess whether their population mean ranks differ. It is a paired
    difference test that is used as an alternative to the paired Student's t-test
    when the population cannot be assumed to be normally distributed.

    The test involves ranking each pair of observations, ignoring the signs and
    zeroes, and then summing the ranks for the observations with positive differences
    and those with negative differences. The test statistic is the smaller of these
    two sums. The null hypothesis states that the median difference between the pairs
    of observations is zero.

    Mathematically, the test statistic \(W\) can be calculated as follows:

    .. math::
        W = \min(W^+, W^-)

    where \(W^+\) is the sum of the ranks for the observations where the second
    measurement is greater than the first, and \(W^-\) is the sum of the ranks
    for the observations where the first measurement is greater than the second.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.stats.model_comparisons import perform_wilcoxon_test
    >>> model_performance = pd.DataFrame({
            'Model_A': [0.90, 0.85, 0.88],
            'Model_B': [0.92, 0.83, 0.89],
            'Model_C': [0.91, 0.84, 0.87]})
    >>> perform_wilcoxon_test(model_performance)
            Model_A  Model_B  Model_C
    Model_A      NaN    0.317    0.180
    Model_B    0.317      NaN    0.317
    Model_C    0.180    0.317      NaN

    In this example, we compare the performance of three models (Model_A, Model_B,
    and Model_C) across three datasets. The resulting DataFrame contains the p-values
    for each pair of models. A p-value less than 0.05 would indicate a statistically
    significant difference in performance between the two models.
    """

    n_models = model_performance_data.shape[1]
    p_values_matrix = np.full((n_models, n_models), np.nan)
    
    alpha = validate_ratio( alpha, bounds=(0, 1), to_percent= True, exclude=0 )
    
    for i in range(n_models):
        for j in range(i+1, n_models):
            # Extract the performance scores for the two models
            model_i_scores = model_performance_data.iloc[:, i]
            model_j_scores = model_performance_data.iloc[:, j]
            
            # Perform the Wilcoxon signed-rank test
            stat, p_value = wilcoxon(model_i_scores, model_j_scores)
            
            # Populate the symmetric matrix with p-values
            p_values_matrix[i, j] = p_value
            p_values_matrix[j, i] = p_value
    
    # Create a DataFrame from the matrix and set the model names as index/column labels
    p_values_df = pd.DataFrame(p_values_matrix, index=model_performance_data.columns,
                               columns=model_performance_data.columns)
    
    if mask_non_significant:
        # Apply significance threshold to mask non-significant comparisons
        significant_diffs = p_values_df < alpha
        p_values_df[~significant_diffs] = np.nan
        
    p_values_df= DataFrameFormatter("Wilcoxon Test Results").add_df (p_values_df)
    return p_values_df

@isdf 
def perform_friedman_test2(
        model_performance_data: Union[Dict[str, List[float]], DataFrame]
    ):
    """
    Performs the Friedman test on the performance data of multiple classification models.
    This function takes a DataFrame where models are columns and datasets are rows. 
    It performs the Friedman test and returns a dictionary containing the test 
    statistic and the p-value. A small p-value (typically ≤ 0.05) indicates 
    that there is a significant difference in the performance of at least one 
    model compared to others across the datasets.
    
    Parameters:
    -----------
    model_performance_data : pd.DataFrame
        A DataFrame containing the performance metrics (e.g., accuracy) of the models.
        Each row represents a different dataset, and each column represents a different model.
    
    Returns:
    --------
    result : dict
        A dictionary containing the Friedman test statistic and the p-value.
        
    """
    # Ensure the input is a DataFrame
    if not isinstance(model_performance_data, pd.DataFrame):
        raise ValueError("model_performance_data must be a pandas DataFrame.")
    
    # Perform the Friedman test
    statistic, p_value = friedmanchisquare(*model_performance_data.values.T)
    
    # Return the test results
    results= DataFrameFormatter("Friedman Test Results").add_df (
        {"Friedman Test Statistic": statistic, "p-value": p_value})
    
    return results

@isdf 
def perform_nemenyi_posthoc_test2(
       model_performance_data: Union[Dict[str, List[float]], DataFrame]
    ):
    """
    Performs the Nemenyi post-hoc test on model performance data.
    
    Parameters:
    -----------
    model_performance_data : pd.DataFrame
        A DataFrame containing the performance metrics of the models.
        Rows represent different datasets and columns represent different models.
    
    Returns:
    --------
    pairwise_comparisons : pd.DataFrame
        A DataFrame containing the p-values of pairwise comparisons between models.
        
    Note 
    -----
    This function uses the posthoc_nemenyi_friedman method from scikit-posthocs
    to perform pairwise comparisons between all models. The output is a 
    DataFrame where each element represents the p-value of the comparison 
    between two models. Lower p-values (typically ≤ 0.05) suggest a significant
    difference between the performance of those two models.

    Keep in mind that the Nemenyi test may not be as powerful for detecting 
    differences as some other post-hoc tests, but it's widely used for its 
    simplicity and non-parametric nature.
    """
    import scikit_posthocs as sp
    # Ensure the input is a DataFrame
    if not isinstance(model_performance_data, pd.DataFrame):
        raise ValueError("model_performance_data must be a pandas DataFrame.")
    
    # Perform the Nemenyi post-hoc test using the scikit-posthocs library
    pairwise_comparisons = sp.posthoc_nemenyi_friedman(model_performance_data.values)
    
    # Adjust the index and columns of the result to match the input DataFrame
    pairwise_comparisons.index = model_performance_data.columns
    pairwise_comparisons.columns = model_performance_data.columns
    
    pairwise_comparisons= DataFrameFormatter("PairWise Results").add_df (
        pairwise_comparisons)
    
    return pairwise_comparisons

@isdf 
def compute_model_ranks(
        model_performance_data: Union[Dict[str, List[float]], DataFrame]
        ):
    """
    Computes the ranks of models based on their performance.

    Parameters:
    -----------
    model_performance_data : pd.DataFrame
        A DataFrame containing the performance metrics of the models.
        Rows represent different datasets and columns represent different models.

    Returns:
    --------
    ranks : pd.DataFrame
        A DataFrame containing the ranks of each model.
    """
    
    # Ensure the input is a DataFrame
    if not isinstance(model_performance_data, pd.DataFrame):
        raise ValueError("model_performance_data must be a pandas DataFrame.")
    
    # Rank the models based on their performance
    ranks = model_performance_data.rank(axis=1, ascending=False)
    ranks= DataFrameFormatter("Ranks Results").add_df (ranks)
    return ranks

@isdf     
def perform_nemenyi_test2(
     model_performance_data: Union[Dict[str, List[float]], DataFrame], 
     ranks:Union[Series,Array1D ] 
     ):
    """
    Performs the Nemenyi post-hoc test using ranks of model performance.

    Parameters:
    -----------
    ranks : pd.DataFrame
        A DataFrame containing the ranks of each model.
    
    Returns:
    --------
    pairwise_comparisons : pd.DataFrame
        A DataFrame containing the p-values of pairwise comparisons between models.
    """
    import scikit_posthocs as sp
    # Convert ranks to a format suitable for the scikit-posthocs library
    if isinstance ( ranks,  pd.Series): 
        ranks_array = ranks.values
    
    if not _is_arraylike_1d(ranks): 
        raise TypeError (
            f"Expect arrayLike one dimension. Got {type(ranks).__name__!r} ")
    
    # Perform the Nemenyi post-hoc test
    pairwise_comparisons = sp.posthoc_nemenyi_friedman(ranks_array)
    
    # Adjust the index and columns of the result to match the input DataFrame
    pairwise_comparisons.index = ranks.columns
    pairwise_comparisons.columns = ranks.columns
    pairwise_comparisons= DataFrameFormatter("Nemenyi Results").add_df (
        pairwise_comparisons)
    return pairwise_comparisons

@isdf 
def perform_wilcoxon_base_test(
      model_performance_data: Union[Dict[str, List[float]], DataFrame]
      ):
    """
    Executes the Wilcoxon signed-rank test pairwise across multiple models
    to statistically compare their performance metrics. This non-parametric
    test assesses whether the differences in pairs of model performances are
    symmetrically distributed around zero, indicating no significant difference.

    Parameters
    ----------
    model_performance_data : Union[Dict[str, List[float]], pd.DataFrame]
        A dictionary or DataFrame with model names as keys (columns) and
        lists (rows) of performance metrics as values. Each row should
        correspond to the performance metric (e.g., accuracy) of the model
        on a different dataset.

    Returns
    -------
    results : pd.DataFrame
        A DataFrame where each row and column header corresponds to a model
        and each cell contains the p-value resulting from the Wilcoxon signed-rank
        test between the pair of models denoted by the respective row and column.

    Notes
    -----
    The Wilcoxon signed-rank test is a non-parametric alternative to the paired
    T-test when the differences between pairs do not necessarily follow a normal
    distribution. In the context of model comparison, it can be used to determine
    whether two models have different performances on a set of datasets. The test
    assumes that the differences between paired observations are independent and
    identically distributed.
    
    This function performs the test on each unique pair of models. The resulting
    p-values are stored in a square DataFrame, with models represented both in rows
    and columns. A p-value close to 0 suggests a significant difference in performance
    between the pair of models. Diagonal cells are not computed as they would compare
    a model to itself.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.stats.model_comparisons import perform_wilcoxon_base_test
    >>> performance_data = pd.DataFrame({
    ...     'Model_X': [0.85, 0.80, 0.78],
    ...     'Model_Y': [0.75, 0.82, 0.79],
    ...     'Model_Z': [0.80, 0.77, 0.74]
    ... })
    >>> results = perform_wilcoxon_base_test(performance_data)
    >>> print(results)
    
    The resulting DataFrame will contain NaNs on the diagonal and p-values
    below the diagonal, reflecting the symmetric nature of the test.
    """
    columns = model_performance_data.columns
    n = len(columns)
    results = pd.DataFrame(index=columns, columns=columns)

    for i in range(n):
        for j in range(i + 1, n):
            model_i = model_performance_data.iloc[:, i]
            model_j = model_performance_data.iloc[:, j]
            
            # Perform the Wilcoxon signed-rank test
            stat, p = wilcoxon(model_i, model_j)
            results.iloc[i, j] = p
            results.iloc[j, i] = p
    
    # Fill diagonal with NaNs since we don't compare a model with itself
    # pd.fill_diagonal(results.values, pd.NA)
    np.fill_diagonal(results.values, pd.NA)
    results= DataFrameFormatter("Wilcoxon Results").add_df (
        results)
    return results

def plot_cd_diagram(
    model_performance_data:Optional[DataFrame]=None, 
    model_ranks: Optional[Union[Series, Array1D]]=None, 
    cd: float=0.5, 
    ax =None, 
    title: Optional[str]=None, 
     ):
    """
    Plots a Critical Difference (CD) diagram to visually compare the rankings
    of machine learning models based on their performance across multiple datasets.

    The CD diagram illustrates the average ranks of models and highlights
    those that are not significantly different from each other at a given
    significance level, represented by the Critical Difference (cd).

    Parameters
    ----------
    model_performance_data : pd.DataFrame, optional
        A pandas DataFrame where each row represents a dataset and each
        column a different model. The cell values should be the performance
        metric (e.g., accuracy, F1 score) of the models on the datasets.
    model_ranks : pd.Series, optional
        A pandas Series where the index contains model names and values
        represent the average ranks of these models. If `model_performance_data`
        is provided, `model_ranks` will be computed from it.
    cd : float, optional
        The Critical Difference value for significance level. Defaults to 0.5.
    ax : matplotlib.axes.Axes, optional
        The axes upon which to plot the CD diagram. If None, a new figure
        and axes are created.
    title : str, optional
        Title for the plot. Defaults to 'Critical Difference Diagram'.

    Raises
    ------
    ValueError
        If neither `model_performance_data` nor `model_ranks` are provided.
    TypeError
        If `model_performance_data` is not a pandas DataFrame or `model_ranks`
        is not a pandas Series.

    Examples
    --------
    >>> from gofast.stats.model_comparisons import plot_cd_diagram
    >>> model_performance = pd.DataFrame({
            'Model_A': [0.90, 0.85, 0.88],
            'Model_B': [0.92, 0.83, 0.89],
            'Model_C': [0.91, 0.84, 0.87]})
    >>> plot_cd_diagram(model_performance_data=model_performance, cd=0.5)

    This function creates a visual representation highlighting the ranking
    and the statistical significance of differences between models, aiding
    in the comparison and selection of models.

    Notes
    -----
    The Critical Difference (CD) diagram is particularly useful in the analysis
    of results from multiple models across various datasets, as it provides
    a clear visual indication of which models perform significantly differently.
    The CD is calculated based on a statistical test (e.g., the Friedman test)
    and post-hoc analysis, which should be performed beforehand to determine
    the CD value.
    """

    if model_performance_data is None and model_ranks is None:
        raise ValueError("Either model_performance_data or model_ranks must be provided.")
    
    if model_performance_data is not None:
        if not isinstance(model_performance_data, pd.DataFrame):
            raise TypeError("model_performance_data must be a pandas DataFrame.")
        model_ranks = model_performance_data.rank().mean()
    
    if model_ranks is not None:
        if not isinstance(model_ranks, pd.Series):
            raise TypeError("model_ranks must be a pandas Series.")
    
    assert cd > 0, "Critical difference (cd) must be a positive value."
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))  # Increase the figure size
    
    model_ranks_sorted = model_ranks.sort_values()
    y_positions = np.arange(len(model_ranks_sorted)) + 1

    # Extend the x-axis limits to provide more space for the CD intervals
    min_rank = model_ranks_sorted.min() - cd
    max_rank = model_ranks_sorted.max() + cd
    ax.set_xlim(min_rank - 1, max_rank + 1)

    for y, (model, rank) in zip(y_positions, model_ranks_sorted.items()):
        ax.plot([rank - cd / 2, rank + cd / 2], [y, y], marker='o', 
                markersize=10, color='black', label=model if y == 1 else "")  # Adjust marker size
        # Annotate the actual average rank next to each model
        ax.text(rank, y, f' {model} ({rank:.2f})', verticalalignment='center')

    ax.set_yticks(y_positions)
    ax.set_yticklabels(model_ranks_sorted.index)
    ax.invert_yaxis()  # Invert y-axis to have the best model at the top
    ax.set_xlabel('Average Rank')
    ax.set_title(title or 'Critical Difference Diagram')
    ax.axvline(x=model_ranks_sorted.min() - cd / 2, linestyle='--', color='gray', alpha=0.5)
    ax.axvline(x=model_ranks_sorted.max() + cd / 2, linestyle='--', color='gray', alpha=0.5)
    ax.legend(title="Models", loc="lower right")
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.show()

@isdf 
def plot_model_rankings(
      model_performance_data: Union[Dict[str, List[float]], DataFrame]
     ):
    """
    Generates a bar chart visualizing the average ranking of models based on 
    performance metrics. Each model's ranking is calculated across multiple 
    datasets, and they are ordered from best (lowest average rank) to worst 
    (highest average rank) in the chart.

    Parameters
    ----------
    model_performance_data : Union[Dict[str, List[float]], pd.DataFrame]
        A dictionary with model names as keys and lists of performance 
        metrics as values, or a pandas DataFrame with models as columns 
        and performance metrics as rows. Each cell should contain a 
        performance score of the model on a specific dataset.

    Returns
    -------
    None
        Displays a bar chart ranking the models by their average performance 
        but does not return any value.

    Raises
    ------
    TypeError
        If `model_performance_data` is not a dictionary or a pandas DataFrame.
    ValueError
        If `model_performance_data` is empty or does not contain numerical values.

    Examples
    --------
    >>> import pandas pd 
    >>> from gofast.stats.model_comparisons import plot_model_rankings
    >>> performance_data = {
    ...    'Model_X': [0.85, 0.80, 0.78],
    ...    'Model_Y': [0.75, 0.82, 0.79],
    ...    'Model_Z': [0.80, 0.77, 0.74]
    ... }
    >>> plot_model_rankings(performance_data)

    Or using a DataFrame:

    >>> import pandas as pd
    >>> df_performance = pd.DataFrame(performance_data)
    >>> plot_model_rankings(df_performance)

    Notes
    -----
    The function computes the average rank of each model's performance scores 
    over all datasets. Models with consistently higher performance across 
    datasets will have a lower average rank. The bar chart provides an intuitive 
    visualization of model performance, making it easier to identify the best 
    and worst-performing models. Models are displayed on the x-axis with their 
    corresponding average ranks on the y-axis, and ranks are shown on top of 
    their respective bars for clarity.
    """
    # Validate input
    if not isinstance(model_performance_data, pd.DataFrame):
        raise TypeError("The model_performance_data must be a pandas DataFrame.")
    
    # Check that the DataFrame is not empty
    if model_performance_data.empty:
        raise ValueError("The model_performance_data DataFrame is empty.")
    
    # Ensure the DataFrame contains numerical values
    if not np.issubdtype(model_performance_data.values.dtype, np.number):
        raise ValueError("The model_performance_data must contain numerical values.")
    
    # Calculate the average rank for each model
    ranks = model_performance_data.rank(axis=0, ascending=True).mean()

    # Sort the models by their average rank (best to worst)
    sorted_ranks = ranks.sort_values()

    # Create the bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(sorted_ranks.index, sorted_ranks.values, color='skyblue')
    ax.set_xlabel('Models')
    ax.set_ylabel('Average Rank')
    ax.set_title('Model Rankings')
    ax.invert_yaxis()  # Invert y-axis to have the best (lowest rank) at the top

    # Add the rank values on top of the bars
    for index, value in enumerate(sorted_ranks):
        ax.text(index, value, f"{value:.2f}", ha='center', va='bottom')

    plt.xticks(rotation=45)
    plt.tight_layout()  # Adjust layout to fit all labels
    plt.show()
    

@isdf     
@ensure_pkg("scikit-posthocs", extra = ( 
    "Post-hocs test expects 'scikit-posthocs' to be installed.")
    ) 
@ensure_pkg("statmodels")
def perform_posthoc_test(
    model_performance_data: DataFrame, 
    test_method: str='tukey', 
    alpha: str=0.05, 
    metric: str='accuracy'
    ):
    """
    Performs post-hoc analysis on model performance data to identify specific
    differences between models after a significant Friedman test result.

    Parameters
    ----------
    model_performance_data : pd.DataFrame
        A DataFrame where each row represents a dataset and each column
        represents a model. Cells contain performance metrics (e.g., accuracy)
        of models on datasets.
    test_method : str, optional
        The post-hoc test method to apply. Supported methods are 'tukey' for
        Tukey's Honestly Significant Difference (HSD) test and 'nemenyi' for
        the Nemenyi test. Defaults to 'tukey'.
    alpha : float, optional
        Significance level for determining statistically significant differences,
        with a default value of 0.05.
    metric : str, optional
        The name of the performance metric used in `model_performance_data`,
        such as 'accuracy', for labeling purposes. Defaults to 'accuracy'.

    Returns
    -------
    posthoc_result : pd.DataFrame or other suitable format
        The results of the post-hoc test, including pairwise comparisons and
        significance levels. The format depends on the selected `test_method`.

    Notes
    -----
    - Tukey's HSD test is useful for comparing all possible pairs of groups
      to identify significantly different pairs. It controls the Type I error
      rate across multiple comparisons.

      .. math:: Q = \\frac{\\bar{Y}_i - \\bar{Y}_j}{SE}

    - The Nemenyi test is a non-parametric equivalent of Tukey's test that
      uses rank sums. It's suitable when the assumptions of normality and
      homoscedasticity are violated.

      .. math:: CD = q_\\alpha\\sqrt{\\frac{k(k+1)}{6N}}

    These tests are essential when the Friedman test indicates a significant
    difference, allowing researchers to pinpoint where these differences lie.
    Tukey's test is preferred when data meet parametric assumptions, while
    the Nemenyi test is more versatile for non-parametric conditions.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.stats.model_comparisons import perform_posthoc_test
    >>> df = pd.DataFrame({
    ...    'Model_A': [0.8, 0.82, 0.78, 0.81, 0.79],
    ...    'Model_B': [0.79, 0.84, 0.76, 0.82, 0.78],
    ...    'Model_C': [0.81, 0.83, 0.77, 0.80, 0.76]
    ... })
    >>> result = perform_posthoc_test(df, test_method='tukey')
    >>> print(result)

    The choice between Tukey's and Nemenyi depends on the dataset's characteristics
    and the assumptions underlying the data.
    """
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    import scikit_posthocs as sp
    
    # Input validation:The decorator 'isdf' handles this part. 
    # Shoud be removed next.  
    if not isinstance(model_performance_data, pd.DataFrame):
        raise TypeError("model_performance_data must be a pandas DataFrame.")
    
    valid_methods = {'tukey': 'pairwise_tukeyhsd', 'nemenyi': 'posthoc_nemenyi_friedman'}
    if test_method.lower() not in valid_methods:
        raise ValueError(f"Unsupported test method '{test_method}'."
                         " Available methods are {list(valid_methods.keys())}.")

    # Define a function mapper for scalability
    method_function_mapper = {
        'tukey': lambda data, groups, alpha: pairwise_tukeyhsd(
            endog=data[metric], groups=data['Model'], alpha=alpha).summary(),
        'nemenyi': lambda ranks: sp.posthoc_nemenyi_friedman(ranks)
    }
    if test_method.lower() == 'tukey':
        # Flatten the DataFrame for Tukey's HSD
        data_flat = model_performance_data.melt(var_name='Model', value_name=metric)
    else:
        # Prepare data for Nemenyi or other rank-based tests
        encoder = LabelEncoder()
        encoded_models = encoder.fit_transform(model_performance_data.columns)
        ranks = ( 
            np.array([rankdata(-1 * performance) 
                      for performance in model_performance_data.values])
            )
        data_flat = pd.DataFrame(ranks, columns=encoded_models)

    # Execute the selected post-hoc test
    posthoc_result = ( 
        method_function_mapper[test_method.lower()](data_flat, data_flat['Model'], alpha)
        if test_method.lower() == 'tukey' 
        else method_function_mapper[test_method.lower()](ranks)
        )
    posthoc_result= DataFrameFormatter(f"Posthoc {test_method} Results").add_df (
        posthoc_result)
    return posthoc_result

@isdf 
@ensure_pkg (
    "statsmodels", 
    extra= "Post-hocs test need 'scikit-posthocs' package to be installed.", 
    partial_check=True,
    condition=lambda *args, **kwargs: kwargs.get("test_method") =='nemenyi'
   )
@ensure_pkg (
    "statsmodels", 
    extra= "Post-hocs test need 'statsmodels' to be installed.", 
    partial_check=True,
    condition=lambda *args, **kwargs: kwargs.get("test_method") =='tukey'
   )
def perform_posthoc_test2(
    model_performance_data: DataFrame,
    test_method: str='tukey',
    alpha: float=0.05,
    metric: str='accuracy'
    ):
    """
    Performs post-hoc analysis on model performance data to investigate pairwise
    differences between models after a significant Friedman test result.

    Parameters
    ----------
    model_performance_data : pd.DataFrame
        A DataFrame containing the model performance metrics, where each row
        represents a dataset, and each column represents a model.
    test_method : str, optional
        The post-hoc test to apply. Options include 'tukey' for Tukey's HSD test
        and 'nemenyi' for Nemenyi test. Defaults to 'tukey'.
    alpha : float, optional
        The significance level for determining statistically significant differences.
        Defaults to 0.05.
    metric : str, optional
        The performance metric name used in `model_performance_data`. Defaults to
        'accuracy', used for labeling purposes.

    Returns
    -------
    posthoc_result : dict or pd.DataFrame
        The result of the post-hoc analysis. The format and content depend on the
        selected `test_method`.

    Examples
    --------
    >>> model_performance_data = pd.DataFrame({
    ...     'Model_A': [0.8, 0.82, 0.78, 0.81, 0.79],
    ...     'Model_B': [0.79, 0.84, 0.76, 0.82, 0.78],
    ...     'Model_C': [0.81, 0.83, 0.77, 0.80, 0.76]
    ... })
    >>> posthoc_result = perform_posthoc_test(model_performance_data, test_method='tukey')
    >>> print(posthoc_result)

    Raises
    ------
    ValueError
        If an unsupported `test_method` is specified.
    """
   
    if test_method.lower() == 'tukey':
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        # Flatten the DataFrame for Tukey's HSD test
        data_flat = model_performance_data.melt(var_name='Model', value_name=metric)
        posthoc_result = pairwise_tukeyhsd(
            endog=data_flat[metric],
            groups=data_flat['Model'],
            alpha=alpha
        ).summary()
    elif test_method.lower() == 'nemenyi':
        from scikit_posthocs import posthoc_nemenyi_friedman
        # Nemenyi test with ranks from Friedman test setup
        ranks = np.array([rankdata(-1 * performance) 
                          for performance in model_performance_data.values])
        posthoc_result = posthoc_nemenyi_friedman(ranks)
    else:
        raise ValueError(f"Unsupported test method '{test_method}'."
                         " Choose 'tukey' or 'nemenyi'.")
    posthoc_result= DataFrameFormatter(f"Posthoc {test_method} Results").add_df (
        posthoc_result)
    return posthoc_result

@isdf 
def visualize_model_performance(
        model_performance_data: DataFrame, 
        plot_type: str='scatter', 
        **kwargs: Any):
    """
    Visualizes the performance of different models using a variety of plot types.
    This function aims to provide a versatile way to graphically compare model
    performances across multiple datasets or performance metrics.

    Parameters
    ----------
    model_performance_data : pd.DataFrame
        A DataFrame containing the performance metrics of models. Each column
        represents a different model, and each row represents a performance
        metric or a dataset.
    plot_type : str, optional
        The type of plot to generate. Available options are:
        - 'paired_box': Paired box plots for each model.
        - 'violin': Violin plots showing the distribution of the performance metrics.
        - 'difference': Bar plot of the difference in performance between models.
        - 'scatter': Scatter plot of performance metrics for each model.
        - 'cdf': Cumulative distribution function plot for model performances.
        Default is 'scatter'.
    **kwargs : dict
        Additional keyword arguments passed to the underlying matplotlib or
        seaborn plotting function.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.stats.model_comparisons import visualize_model_performance
    >>> # Example DataFrame of model performances
    >>> data = {'Model A': [0.8, 0.82, 0.78],
    ...         'Model B': [0.79, 0.84, 0.76],
    ...         'Model C': [0.81, 0.83, 0.77]}
    >>> model_performance_df = pd.DataFrame(data)
    >>> # Visualize using a scatter plot
    >>> visualize_model_performance(model_performance_df, plot_type='scatter')

    Notes
    -----
    - The 'paired_box' and 'violin' plots are useful for visualizing the
      spread and distribution of performance metrics across models.
    - The 'difference' plot is particularly helpful for directly comparing
      the performance of two models across multiple metrics or datasets.
    - The 'scatter' plot provides a granular look at each performance metric
      or dataset, offering insights into the variability across models.
    - The 'cdf' plot helps in understanding the proportion of datasets where
      models achieve certain performance thresholds, offering a cumulative
      perspective.

    Each plot type can be further customized through `**kwargs`, providing
    flexibility in adjusting plot aesthetics to best suit the data visualization
    needs.
    """

    plot_type = str(plot_type).lower()
    valid_plot_types = ['paired_box', 'violin', 'difference', 'scatter', 'cdf']
    
    if plot_type not in valid_plot_types:
        raise ValueError(f"Unsupported plot type '{plot_type}'. "
                         f"Available plot types are {valid_plot_types}.")
    
    if plot_type == 'paired_box' or plot_type == 'violin':
        fig, ax = plt.subplots()
        if plot_type == 'paired_box':
            sns.boxplot(data=model_performance_data, ax=ax, **kwargs)
        else:
            sns.violinplot(data=model_performance_data, ax=ax, **kwargs)
        ax.set_title('Model Performance Comparison')
        ax.set_ylabel('Performance Metric')

    elif plot_type == 'difference':
        differences = model_performance_data.diff(axis=1).iloc[:, 1]
        differences.plot(kind='bar', **kwargs)
        plt.title('Differences in Model Performances')
        plt.ylabel('Difference in Performance Metric')

    elif plot_type == 'scatter':
        pairs = pd.melt(model_performance_data.reset_index(), id_vars=['index'],
                        value_vars=model_performance_data.columns)
        sns.scatterplot(data=pairs, x='variable', y='value', hue='index', **kwargs)
        plt.title('Scatter Plot of Model Performances')
        plt.ylabel('Performance Metric')

    elif plot_type == 'cdf':
        for column in model_performance_data.columns:
            sns.ecdfplot(model_performance_data[column], label=column, **kwargs)
        plt.title('Cumulative Distribution of Model Performances')
        plt.ylabel('CDF')
        plt.legend()

    plt.xlabel('Datasets')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
@isdf 
def visualize_wilcoxon_test(
    model_performance_data: DataFrame, 
    alpha: float=0.05, 
    model_pairs: Optional[ Tuple[str, str]]=None, 
    **kwargs: Any
    ):
    """
    Visualizes the results of the Wilcoxon signed-rank tests between pairs of
    models, highlighting significant differences in model performances.
    
    Function helps in identifying models with statistically different 
    performance metrics, thereby assisting in model selection and comparison
    processes.

    Parameters
    ----------
    model_performance_data : pd.DataFrame
        A pandas DataFrame where rows represent different datasets or iterations,
        and columns represent different models. Each cell contains the performance
        metric (e.g., accuracy, F1 score) of a model on a dataset.
    alpha : float, optional
        The significance level to determine if the difference between model 
        performances is statistically significant. Defaults to 0.05.
    model_pairs : list of tuples, optional
        A list where each tuple contains two model names (as strings) to be 
        compared. If `None`, all possible pairs are compared. Defaults to `None`.
    **kwargs : dict
        Additional keyword arguments to pass to the plotting function, such as
        `figsize` to adjust the size of the plot.

    Returns
    -------
    pd.DataFrame
        A DataFrame summarizing the Wilcoxon test results, including the model 
        pairs, the Wilcoxon statistic values, p-values, and a flag indicating 
        significant differences.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.stats.model_comparisons import visualize_wilcoxon_test
    >>> df = pd.DataFrame({
    ...    'Model_A': [0.8, 0.82, 0.78, 0.81, 0.79],
    ...    'Model_B': [0.79, 0.84, 0.76, 0.82, 0.78],
    ...    'Model_C': [0.81, 0.83, 0.77, 0.80, 0.76]
    ... })
    >>> results= visualize_wilcoxon_test(df, alpha=0.05)
    >>> print(results)
       Model 1  Model 2  Statistic  p-value  Significant
    0  Model_A  Model_B        5.5   0.8125        False
    1  Model_A  Model_C        5.0   0.6250        False
    2  Model_B  Model_C        6.0   0.8125        False
    
    Results Interpretation 
    ----------------------
    The results from the `visualize_wilcoxon_test` function show the outcome
    of the Wilcoxon signed-rank tests between pairs of models. 
    For each pair, we have a Wilcoxon statistic and a corresponding p-value. 
    The 'Significant' column indicates whether the difference between the models
    is statistically significant based on the alpha level (0.05 in this case).

    Here's how to interpret the results:
    
    1. **Model_A vs. Model_B**:
       - **Statistic**: 5.5
       - **p-value**: 0.8125
       - **Significant**: False
       
       This means there is no statistically significant difference in the 
       performance of Model_A and Model_B. The high p-value 
       (greater than the alpha level of 0.05) suggests that any observed 
       difference in performance could be due to random chance rather than 
       a systematic difference between the models.
    
    2. **Model_A vs. Model_C**:
       - **Statistic**: 5.0
       - **p-value**: 0.6250
       - **Significant**: False
       
       Similarly, there is no statistically significant difference in the 
       performance of Model_A and Model_C. The p-value is again much higher 
       than the alpha level.
    
    3. **Model_B vs. Model_C**:
       - **Statistic**: 6.0
       - **p-value**: 0.8125
       - **Significant**: False
       
       Once more, no statistically significant difference is observed between
       Model_B and Model_C's performances.
    
    The bar plot would typically visualize these results by displaying the 
    Wilcoxon statistic for each model pair, with color-coding or some other
    visual indicator for whether the differences are statistically significant.
    Since all p-values are above the significance threshold (alpha = 0.05), 
    the plot suggests that there's no compelling evidence to prefer one model
    over the others based on the provided performance data.
    
    Notes
    -----
    The Wilcoxon signed-rank test is a non-parametric statistical hypothesis test 
    used when comparing two related samples or repeated measurements on a single 
    sample to assess whether their population mean ranks differ. It's an alternative 
    to the paired Student's t-test when the population cannot be assumed to be 
    normally distributed.
    """
    # Prepare the plot
    fig, ax = plt.subplots(figsize=kwargs.pop('figsize', (10, 6)))

    # Results container
    results = []
    
    # Generate models pair if not given 
    model_pairs = generate_model_pairs(model_performance_data, model_pairs)
    # Ensure all model pairs exist in DataFrame columns
    for model_a, model_b in model_pairs:
        if ( 
                model_a not in model_performance_data.columns 
                or model_b not in model_performance_data.columns
            ):
            raise ValueError(
                f"One or both models in pair ({model_a}, {model_b})"
                " do not exist in the DataFrame."
                )
    # Perform Wilcoxon tests and collect results
    for model_a, model_b in model_pairs:
        performance_a = model_performance_data[model_a]
        performance_b = model_performance_data[model_b]
        stat, p_value = wilcoxon(performance_a, performance_b)
        results.append({'Model 1': model_a, 'Model 2': model_b, 
                        'Statistic': stat, 'p-value': p_value})

    # Convert results to DataFrame for easy plotting
    results_df = pd.DataFrame(results)

    # Add a column indicating significant differences
    results_df['Significant'] = results_df['p-value'] < alpha

    # Plotting
    # Plotting
    sns.barplot(x='Model 1', y='Statistic', hue='Significant', data=results_df, 
                ax=ax, **kwargs)
    ax.set_title('Wilcoxon Test Results Between Model Pairs')
    ax.set_ylabel('Wilcoxon Statistic')
    ax.set_xlabel('Model Pairs')
    ax.legend(title='Significant Difference', loc='upper right')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    results_df= DataFrameFormatter("wilcoxon Results").add_df (
        results_df)
    return results_df

@isdf 
def generate_model_pairs(
    model_performance_data: Union[Dict[str, List[float]], DataFrame], 
    model_pairs:Optional[ List[Tuple[str, str]]]=None
    ):
    """
    Generates pairs of models for comparison if not explicitly provided.

    This function utilizes itertools to automatically generate all possible
    unique pairs of models for comparison based on the columns of the input
    DataFrame, model_performance_data.

    Parameters:
    -----------
    model_performance_data: pd.DataFrame
        A DataFrame where rows represent datasets and columns represent models.
        Each cell should contain the performance metric of a model on a dataset.
    model_pairs: list of tuple, optional
        Explicitly specified list of model pairs to compare. If None, all
        unique pairs will be generated based on the DataFrame's columns.

    Returns:
    --------
    list of tuples
        A list of tuples, where each tuple contains the names of two models to
        compare. If model_pairs is given, returns it unmodified.

    Examples:
    ---------
    >>> from gofast.stats.model_comparisons import generate_model_pairs
    >>> model_performance_data = pd.DataFrame({
    ...     'Model_A': [0.8, 0.82, 0.78, 0.81, 0.79],
    ...     'Model_B': [0.79, 0.84, 0.76, 0.82, 0.78],
    ...     'Model_C': [0.81, 0.83, 0.77, 0.80, 0.76]
    ... })
    >>> model_pairs = generate_model_pairs(model_performance_data)
    >>> print(model_pairs)
    [('Model_A', 'Model_B'), ('Model_A', 'Model_C'), ('Model_B', 'Model_C')]
    """
    if model_pairs is not None:
        return model_pairs

    # Automatically generate all unique pairs of models from the DataFrame columns
    model_names = model_performance_data.columns
    generated_model_pairs = list(itertools.combinations(model_names, 2))
    
    return generated_model_pairs

# Example usage
if __name__ == "__main__":
    # Example usage with mock data
    model_performance_mock_data = pd.DataFrame({
        'Model_A': np.random.rand(10),
        'Model_B': np.random.rand(10),
        'Model_C': np.random.rand(10),
        'Model_D': np.random.rand(10)
    })

    plot_model_rankings(model_performance_mock_data)
    # Example usage
    model_performance_data = pd.DataFrame({
        'Model_A': [0.90, 0.85, 0.88],
        'Model_B': [0.92, 0.83, 0.89],
        'Model_C': [0.91, 0.84, 0.87]
    })

    ranks = model_performance_data.rank().mean()
    cd_value = 0.5  # Example CD value
    plot_cd_diagram(ranks, cd_value, title='Example CD Diagram')
    
    # Sample data: Model performance on 5 datasets (rows) for 3 models (columns)
    data = {
        'Model_A': [0.8, 0.82, 0.78, 0.81, 0.79],
        'Model_B': [0.79, 0.84, 0.76, 0.82, 0.78],
        'Model_C': [0.81, 0.83, 0.77, 0.80, 0.76]
    }
    model_performance_data = pd.DataFrame(data)
    
    # Perform the Wilcoxon signed-rank test
    wilcoxon_results = perform_wilcoxon_base_test(model_performance_data)
    print("Wilcoxon Signed-Rank Test (p-values):\n", wilcoxon_results)


    # Sample data: Model performance on 3 datasets (rows) for 4 models (columns)
    data = {
        'Model_A': [0.8, 0.82, 0.78],
        'Model_B': [0.79, 0.84, 0.76],
        'Model_C': [0.81, 0.83, 0.77],
        'Model_D': [0.8, 0.85, 0.75]
    }
    model_performance_data = pd.DataFrame(data)
    
    # Compute ranks
    ranks = compute_model_ranks(model_performance_data)
    print("Ranks:\n", ranks)
    
    #
    # Sample data: Model performance on 3 datasets (rows) for 4 models (columns)
    data = {
        'Model_A': [0.8, 0.82, 0.78],
        'Model_B': [0.79, 0.84, 0.76],
        'Model_C': [0.81, 0.83, 0.77],
        'Model_D': [0.8, 0.85, 0.75]
    }
    model_performance_data = pd.DataFrame(data)
    pairwise_comparisons = perform_nemenyi_posthoc_test2(model_performance_data)
    print(pairwise_comparisons)


    # Sample data: Model performance on 3 datasets (rows) for 4 models (columns)
    data = {
        'Model_A': [0.8, 0.82, 0.78],
        'Model_B': [0.79, 0.84, 0.76],
        'Model_C': [0.81, 0.83, 0.77],
        'Model_D': [0.8, 0.85, 0.75]
    }
    model_performance_data = pd.DataFrame(data)
    result = perform_friedman_test2(model_performance_data)
    print(result)
