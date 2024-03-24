
import numpy as np 
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, chi2
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
from ..tools.coreutils import validate_ratio 

def perform_friedman_test(
        model_performance_data, alpha=0.05,
        perform_posthoc=False, posthoc_test=None):
    """
    Performs the Friedman test on multiple classifiers over multiple datasets.

    The Friedman test is a non-parametric statistical test used to detect differences
    in treatments across multiple test attempts. In the context of machine learning,
    it is used to compare the performance of different algorithms across various
    datasets. If significant differences are found, post-hoc tests can further
    investigate these differences.

    Parameters
    ----------
    model_performance_data : pd.DataFrame
        A DataFrame where each row represents a dataset and each column represents
        a model. Each cell in the DataFrame should contain the performance metric
        (e.g., accuracy) of the model on that dataset.
    alpha : float, optional
        Significance level used to determine the critical value. Defaults to 0.05.
    perform_posthoc : bool, optional
        Indicates whether to perform post-hoc testing if the Friedman test indicates
        significant differences. Defaults to False.
    posthoc_test : callable, optional
        A function to perform post-hoc testing. It should accept the 
        `model_performance_data` DataFrame as input and return post-hoc test 
        results. Required if `perform_posthoc` is True.

    Returns
    -------
    result : dict
        A dictionary containing the Friedman test statistic, degrees of freedom,
        p-value, and whether there is a significant difference. If perform_posthoc
        is True, it also includes the results of the post-hoc test under "Post-hoc Test".

    Notes
    -----
    The Friedman test is computed as follows:

    .. math::
        Q = \\frac{12N}{k(k+1)}[\\sum_j R_j^2 - \\frac{k(k+1)^2}{4}]

    Where:
    - :math:`N` is the number of datasets.
    - :math:`k` is the number of models.
    - :math:`R_j` is the sum of ranks for the j-th model.

    The null hypothesis of the Friedman test states that all algorithms are equivalent and
    any observed differences are due to chance. A significant p-value rejects this hypothesis,
    suggesting that at least one algorithm performs differently.

    The test is useful in scenarios where multiple models' performances are evaluated across
    multiple datasets. It is robust against non-normal distributions and unequal variances among
    groups.

    Examples
    --------
    >>> from gofast.stats.model_comparisons import perform_friedman_test
    >>> import pandas as pd
    >>> # Assuming `df` is a DataFrame where rows are datasets and columns are models.
    >>> result = perform_friedman_test(df)
    >>> print(result)

    If the result indicates a significant difference and post-hoc testing is desired:

    >>> from some_module import some_posthoc_test
    >>> result = perform_friedman_test(df, perform_posthoc=True, posthoc_test=some_posthoc_test)
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

    return result

def perform_nemenyi_posthoc_test(
        model_performance_data, significance_level=0.05, 
        rank_method='average'):
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

    return results

def perform_wilcoxon_test(
        model_performance_data, alpha=0.05, mask_non_significant=False):
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
    
    return p_values_df


def perform_friedman_test2(model_performance_data):
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
    return {"Friedman Test Statistic": statistic, "p-value": p_value}

def perform_nemenyi_posthoc_test2(model_performance_data):
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
    
    return pairwise_comparisons

def compute_model_ranks(model_performance_data):
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
    
    return ranks
    
def perform_nemenyi_test2(model_performance_data, ranks):
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
    ranks_array = ranks.values
    
    # Perform the Nemenyi post-hoc test
    pairwise_comparisons = sp.posthoc_nemenyi_friedman(ranks_array)
    
    # Adjust the index and columns of the result to match the input DataFrame
    pairwise_comparisons.index = ranks.columns
    pairwise_comparisons.columns = ranks.columns
    
    return pairwise_comparisons

def perform_wilcoxon_test2(model_performance_data):
    """
    Performs the Wilcoxon signed-rank test on pairs of models.

    Parameters:
    -----------
    model_performance_data : pd.DataFrame
        A DataFrame containing the performance metrics of the models.
        Rows represent different datasets and columns represent different models.

    Returns:
    --------
    results : pd.DataFrame
        A DataFrame containing the p-values of the Wilcoxon signed-rank test
        for each pair of models.
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
    pd.fill_diagonal(results.values, pd.NA)
    return results

def plot_cd_diagram(model_performance_data=None, model_ranks=None, cd=0.5, 
                    ax=None, title='Critical Difference Diagram'):
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
    ax.set_title(title)
    ax.axvline(x=model_ranks_sorted.min() - cd / 2, linestyle='--', color='gray', alpha=0.5)
    ax.axvline(x=model_ranks_sorted.max() + cd / 2, linestyle='--', color='gray', alpha=0.5)
    ax.legend(title="Models", loc="lower right")
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    plt.show()

def plot_model_rankings(model_performance_data):
    """
    Plots a bar chart to visualize the ranking of models from best to worst 
    based on their performance data.

    Parameters:
    -----------
    model_performance_data : pd.DataFrame
        A pandas DataFrame where each column represents a model and each row 
        represents the model's performance on a dataset.

    Returns:
    --------
    A bar chart showing the models ranked from best to worst.
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
    wilcoxon_results = perform_wilcoxon_test2(model_performance_data)
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
