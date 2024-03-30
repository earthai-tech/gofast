# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 22:29:22 2024

@author: Daniel
"""
import numpy as np
import pandas as pd

class ModelPerformanceReport:
    def __init__(self, model, X_test, y_test, metrics):
        """
        Initializes the report with model performance metrics.
        
        Parameters:
        - model: The trained model.
        - X_test: Test features DataFrame.
        - y_test: Test target variable.
        - metrics: List of metrics to calculate and display.
        """
        
    def display_summary(self):
        """Prints a summary of the model's performance."""
        
    def plot_metrics(self):
        """Generates plots for the specified metrics, such as ROC curves."""
        
    def detailed_report(self):
        """Generates a detailed text report on model's performance metrics."""
        
class DataFrameReport:
    def __init__(self, before_df, after_df, transformations):
        """
        Initializes the report with DataFrame transformations.
        
        Parameters:
        - before_df: DataFrame before transformations.
        - after_df: DataFrame after transformations.
        - transformations: List or dictionary of applied transformations.
        """
        
    def summary(self):
        """Displays summary statistics of the DataFrame before and after transformations."""
        
    def transformation_details(self):
        """Describes each transformation applied, including parameters and effects."""
        
class OptimizationReport:
    
    def __init__(self, optimization_result):
        """
        Initializes the report with optimization results.
        
        Parameters:
        - optimization_result: Object containing results from optimization.
        """
        
    def best_params(self):
        """Displays the best parameters found."""
        
    def performance_overview(self):
        """Displays performance metrics for the best model."""
        
    def convergence_plot(self):
        """Generates a plot showing the optimization process over iterations."""
        
class ReportFactory:
    @staticmethod
    def create_report(report_type, *args, **kwargs):
        """
        Factory method to create different types of reports.
        
        Parameters:
        - report_type: Type of the report to create (e.g., 'model_performance', 'dataframe', 'optimization').
        - args, kwargs: Arguments required to instantiate the report classes.
        """
        report_type = str(report_type).lower() 
        if report_type == 'model_performance':
            return ModelPerformanceReport(*args, **kwargs)
        elif report_type == 'dataframe':
            return DataFrameReport(*args, **kwargs)
        elif report_type == 'optimization':
            return OptimizationReport(*args, **kwargs)
        else:
            raise ValueError("Unknown report type")


#XXX TODO 
class CustomDataFrame(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def summary(self, include_correlation=False, include_uniques=False, 
                statistics=['mean', 'std', 'min', '25%', '50%', '75%', 'max'],
                include_sample=False, sample_size=5):
        summary = {
            "Shape": self.shape,
            "Data Types": self.dtypes.to_dict(),
            "Missing Values": self.isnull().sum().to_dict(),
            "Basic Statistics": {}
        }
        
        # Basic statistics for numeric columns based on user-selected statistics
        if statistics:
            summary["Basic Statistics"] = self.describe().loc[statistics].to_dict()
        
        # Correlation matrix for numeric columns
        if include_correlation and self.select_dtypes(include=['number']).shape[1] > 1:
            summary["Correlation Matrix"] = self.corr().round(2).to_dict()
        
        # Unique counts for categorical columns
        if include_uniques:
            cat_cols = self.select_dtypes(include=['object', 'category']).columns
            summary["Unique Counts"] = {col: self[col].nunique() for col in cat_cols}
        
        # Sample of the data
        if include_sample:
            if sample_size > len(self):
                sample_size = len(self)
            summary["Sample Data"] = self.sample(n=sample_size).to_dict(orient='list')
        
        return summary

# class AnovaResults:
#     """
#     Anova results class

#     Attributes
#     ----------
#     anova_table : DataFrame
#     """
#     def __init__(self, anova_table):
#         self.anova_table = anova_table

#     def __str__(self):
#         return self.summary().__str__()

#     def summary(self):
#         """create summary results

#         Returns
#         -------
#         summary : summary2.Summary instance
#         """
#         summ = summary2.Summary()
#         summ.add_title('Anova')
#         summ.add_df(self.anova_table)

#         return summ
    
def summarize_inline_table(
    contents, 
    title=None, 
    header=None, 
    max_width='auto',
    top_line='=', 
    sub_line='-', 
    bottom_line='='
    ):
    """
    Creates a string representation of a table summarizing the provided 
    contents, with optional title and header, and customizable table aesthetics.

    Parameters
    ----------
    contents : dict
        A dictionary containing the data to be summarized in the table. Keys
        represent the labels, and values are the corresponding data.
    title : str, optional
        A title for the table, centered above the table content. If None,
        no title is displayed.
    header : str, optional
        A header for the table, displayed below the title (if present) and
        above the table content. If None, no header is displayed.
    max_width : 'auto', int, optional
        The maximum width of the table. If 'auto', the width is adjusted
        based on the content. If an integer is provided, it specifies the
        maximum width; contents may be truncated to fit. Defaults to 'auto'.
    top_line : str, optional
        The character used to create the top border of the table. Defaults
        to '='.
    sub_line : str, optional
        The character used to create the line below the header and above the
        bottom border. Defaults to '-'.
    bottom_line : str, optional
        The character used to create the bottom border of the table. Defaults
        to '='.

    Returns
    -------
    str
        The formatted table as a string.

    Raises
    ------
    ValueError
        If `contents` is not a dictionary.

    Examples
    --------
    >>> from gofast.api.summary import summarize_inline_table
    >>> contents = {
    ...     "Estimator": "SVC",
    ...     "Accuracy": 0.95,
    ...     "Precision": 0.89,
    ...     "Recall": 0.93
    ... }
    >>> print(summarize_inline_table(contents, title="Model Performance", header="Metrics"))
     Model Performance  
    ====================
          Metrics       
    --------------------
    Estimator  : SVC
    Accuracy   : 0.9500
    Precision  : 0.8900
    Recall     : 0.9300
    ====================

    Notes
    -----
    - Numeric values in `contents` are formatted to four decimal places.
    - If the `max_width` is exceeded by a value, the value is truncated with '...'
      appended to indicate truncation.
    - The table width is dynamically determined based on the longest key-value pair
      or set to `max_width` if provided. Adjustments ensure the table's presentation
      is both aesthetic and functional.
    """
    if not isinstance(contents, dict):
        raise ValueError("summarize_inline_table expects a dict of keys and values.")
    
    # Helper function to format values
    # Initial calculations for formatting
    max_key_length = max(len(key) for key in contents) + 1  # +1 for the space after keys
    max_value_length = max(len(format_value(value)) for value in contents.values())
    
    # Adjust table width if 'max_width' is 'auto' or specified as a number
    if max_width == 'auto':
        table_width = max_key_length + max_value_length + 4  # +4 for " : " and extra space
    elif isinstance(max_width, (float, int)):
        table_width = int(max_width)
    else:
        table_width = max_key_length + max_value_length + 4  # Default behavior

    # Title and header
    title_str = f"{title.center(table_width)}" if title else ""
    top_line_str = top_line * table_width
    header_str = f"{header.center(table_width)}" if header else ""
    sub_line_str = sub_line * table_width if header else ""
    
    # Constructing content lines
    content_lines = []
    for key, value in contents.items():
        formatted_value = format_value(value)
        # Truncate long values if necessary
        space_for_value = table_width - max_key_length - 3
        if len(formatted_value) > space_for_value:
            formatted_value = formatted_value[:space_for_value-3] + "..."
    
        key_str = f"{key}"
        line = f"{key_str.ljust(max_key_length)} : {formatted_value}"
        content_lines.append(line)
    
    content_str = "\n".join(content_lines)
    bottom_line_str = bottom_line * table_width
    # Combine all parts
    if header: 
        rows = [title_str, top_line_str, header_str, sub_line_str,
                content_str, bottom_line_str]
    else: 
        rows = [title_str, top_line_str, content_str, bottom_line_str]
    table = "\n".join(rows)
    
    return table

def get_table_width(
    data, include_colon_space=True, 
    max_column_width=100, 
    include_index=True
    ):
    """
    Calculate the maximum width required for displaying a table constructed
    from a dictionary or pandas DataFrame.

    Parameters
    ----------
    data : dict or pandas.DataFrame
        The data to calculate the table width for.
    include_colon_space : bool, optional
        Whether to include extra space for ': ' in the calculation, applicable
        only if `data` is a dictionary. Defaults to True.
    max_column_width : int, optional
        The maximum allowed width for any text or value before truncation
        with '...'. Defaults to 100.
    include_index : bool, optional
        If `data` is a pandas DataFrame, this determines whether the index
        column width should be included in the total width calculation.
        Defaults to True.

    Returns
    -------
    int
        The calculated width of the table necessary to display the data
        without exceeding `max_column_width` for any single column.

    Raises
    ------
    ValueError
        If `data` is neither a dictionary nor a pandas DataFrame.
    Examples 
    -------
    >>> import pandas as pd 
    >>> from gofast.api.summary import get_table_width
    >>> report_data = {
    ...    "Estimator": "SVC",
    ...    "Best parameters": "{C: 1, gamma=0.1}",
    ...    "Accuracy": 0.95,
    ... }
    >>> print("Dictionary Table Width:", get_table_width(report_data))
    Dictionary Table Width: 35
    >>> # For a DataFrame
    >>> df_data = pd.DataFrame({
    ...    "Feature": ["Feature1", "Feature2", (
        "A very long feature name that exceeds max column width")],
    ...    "Importance": [0.1, 0.2, 0.3]
    ... })
    >>> print("DataFrame Table Width:", get_table_width(df_data))
    DataFrame Table Width: 69
    """
    def _format_value(value):
        """Format the value to a string, truncating if necessary."""
        value_str = str(value)
        value_str= format_value(value_str) # format numeric .4f 
        return value_str if len(value_str) <= max_column_width else value_str[
            :max_column_width - 3] + "..."
    
    if not isinstance(data, (dict, pd.DataFrame)):
        raise ValueError("Data must be either a dictionary or a pandas DataFrame.")
    
    if isinstance(data, dict):
        max_key_length = max(len(key) for key in data.keys())
        max_value_length = max(len(_format_value(value)) for value in data.values())
        colon_space = 3 if include_colon_space else 0  # Accounting for " : "
        table_width = max_key_length + max_value_length + colon_space
    else:  # pandas DataFrame
        max_index_length = max(len(str(index)) for index in data.index) if include_index else 0
        max_col_name_length = max(len(col) for col in data.columns)
        max_value_length = data.applymap(_format_value).astype(str).applymap(len).values.max()
        # Accounting for spaces and colon
        table_width = max_index_length + max_col_name_length + max_value_length + 4  
    
    return table_width

def calculate_maximum_length( report_data, max_value_length = 50 ): 
    # Calculate the maximum key length for alignment
    max_key_length = max(len(key) for key in report_data.keys())
    # calculate the maximum values length 
    max_val_length = 0 
    for value in report_data.values (): 
        if isinstance ( value, (int, float, np.integer, np.floating)): 
            value = f"{value}" if isinstance ( value, int) else  f"{float(value):.4f}" 
        if isinstance ( value, pd.Series): 
            value = format_series ( value)
        
        if max_val_length < len(value): 
            max_val_length = len(value) 
            
    return max_key_length, max_val_length
       
def format_value( value ): 
    value_str =str(value)
    if isinstance(value, (int, float, np.integer, np.floating)): 
        value_str = f"{value}" if isinstance ( value, int) else  f"{float(value):.4f}" 

    return value_str

def format_report(report_data, report_title=None):
    """
    Formats the provided report data into a structured text report, 
    accommodating various data types including numbers, strings, pandas Series,
    and pandas DataFrames. The report is formatted with a title (if provided),
    and each key-value pair from the report data dictionary is listed with
    proper alignment and formatting.

    Parameters
    ----------
    report_data : dict
        Dictionary containing the data to be included in the report. Keys
        represent the data labels, and values are the corresponding data
        points, which can be of various data types.
    report_title : str, optional
        A title for the report. If provided, it is centered at the top of
        the report above a top line and a subsection line.

    Returns
    -------
    str
        A string representation of the formatted report including the top
        line, title (if provided), each data label with its formatted value,
        and a bottom line.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.api.summary import format_report
    >>> report_data = {
    ...     'Total Sales': 123456.789,
    ...     'Average Rating': 4.321,
    ...     'Number of Reviews': 987,
    ...     'Sales Data': pd.Series([100, 200, 300], index=['Jan', 'Feb', 'Mar'])
    ... }
    >>> print(format_report(report_data, report_title='Sales Summary'))
                         Sales Summary
    =============================================================
    Total Sales          : 123456.7890
    Average Rating       : 4.3210
    Number of Reviews    : 987
    Sales Data           : Series ~ shape=<3> - mean: 200.0000...
    =============================================================

    Notes
    -----
    - Numeric values are formatted to four decimal places.
    - For pandas Series and DataFrame values, a brief summary is included instead
      of the full data to keep the report concise. The implementation of how
      Series and DataFrame summaries are formatted depends on the `format_series`
      and `dataframe_key_format` helper functions.
    - Long text values are truncated to fit within the maximum line width, with
      '...' appended to indicate truncation. The maximum line width is dynamically
      determined based on the longest key or value, with an option to specify a
      maximum width manually.
    """
    # Calculate the maximum key length for alignment
    max_key_length, max_val_length  = calculate_maximum_length( report_data)
    
    # Prepare the report title and frame lines
    line_length = max_key_length + max_val_length + 4 # Adjust for key-value spacing and aesthetics
    top_line = "=" * line_length
    subsection_line = '-'* line_length
    bottom_line = "=" * line_length
    
    # Construct the report string starting with the top frame line
    report_str = f"{top_line}\n"
    
    # Add the report title if provided, centered within the frame
    if report_title:
        report_str += f"{report_title.center(line_length)}\n{subsection_line}\n"
    
    # Add each key-value pair to the report
    for key, value in report_data.items():
        # Format numeric values with four decimal places
        if isinstance ( value, (int, float, np.integer, np.floating)): 
            formatted_value = format_value ( value )
            report_str += f"{key.ljust(max_key_length)} :  {formatted_value}\n"
        elif isinstance ( value, pd.Series): 
            formatted_value = format_series(value)
            report_str += f"{key.ljust(max_key_length)} :  {formatted_value}\n"
        elif isinstance(value, pd.DataFrame): 
            formatted_value = dataframe_key_format(
                key, value, max_key_length=max_key_length)
            report_str += f"{key.ljust(max_key_length)} :  {formatted_value}\n"
            
        elif isinstance ( value, str) and len(value) > line_length: 
            # consider as long text 
            formatted_value = format_text(
                value, key_length=max_key_length, max_char_text= max_val_length)
            
            report_str += f"{key.ljust(max_key_length)} :  {formatted_value}\n"
        else :    
        # formatted_value = _format_values ( value ) if isinstance(value, (
        #     int, float, np.integer, np.floating)) else ( 
        #     _format_series(value) if isinstance (value, pd.Series ) else value 
        #     ) 
        # Construct the line with key and value, aligning based on the longest key
            report_str += f"{key.ljust(max_key_length)} :  {formatted_value}\n"
    
    # Add the bottom frame line
    report_str += bottom_line
    
    return report_str

def summarize_model_results(
    model_results, 
    title=None, 
    top_line='=', 
    sub_line='-', 
    bottom_line='='
    ):
    """
    Summarizes the results of model tuning, including the best estimator, best parameters,
    and cross-validation results, formatted as a string representation of tables.

    Parameters
    ----------
    model_results : dict
        A dictionary containing the model's tuning results, potentially including keys
        for the best estimator, best parameters, and cross-validation results.
    title : str, optional
        The title of the summary report. Defaults to "Model Results".
    top_line : str, optional
        The character used for the top border of the tables. Defaults to '='.
    sub_line : str, optional
        The character used for the sub-headers within the tables. Defaults to '-'.
    bottom_line : str, optional
        The character used for the bottom border of the tables. Defaults to '='.

    Returns
    -------
    str
        A formatted string that summarizes the model results, including tables for
        the best estimator and parameters as well as detailed cross-validation results.

    Raises
    ------
    ValueError
        If 'best_estimator_' or 'best_parameters_' keys are missing in the provided
        `model_results` dictionary.

    Examples
    --------
    >>> from gofast.api.summary import summarize_model_results
    >>> model_results = {
    ...    'best_parameters_': {'C': 1, 'gamma': 0.1},
    ...    'best_estimator_': "SVC",
    ...    'cv_results_': {
    ...        'split0_test_score': [0.6789, 0.8],
    ...        'split1_test_score': [0.5678, 0.9],
    ...        'split2_test_score': [0.9807, 0.95],
    ...        'split3_test_score': [0.8541, 0.85],
    ...        'params': [{'C': 1, 'gamma': 0.1}, {'C': 10, 'gamma': 0.01}],
    ...    },
     ...   'scoring': 'accuracy',
    ... }
    >>> sum_model = summarize_model_results( model_results ) 
    >>> print(sum_model)
                     Model Results                  
    ================================================
    Best estimator   : str
    Best parameters  : {'C': 1, 'gamma': 0.1}
    nCV              : 2
    Scoring          : Unknown scoring
    ================================================
    
                     Tuning Results                 
    ================================================
      Fold Mean score CV score std score Global mean
    ------------------------------------------------
    0  cv1     0.6233   0.6789    0.0555      0.6233
    1  cv2     0.8500   0.9000    0.0500      0.8500
    ================================================

    Notes
    -----
    - This function requires the presence of `standardize_keys`, 
      `prepare_cv_results_dataframe`, `get_table_width`, and `format_dataframe` 
      helper functions for processing the model results and formatting them 
      into tables.
    - The function dynamically adjusts the width of the cross-validation 
      results table based on the content of the inline summary to ensure 
      consistent presentation.
    """
    title = title or "Model Results"
    # Standardize keys in model_results based on map_keys
    standardized_results = standardize_keys(model_results)

    # Validate presence of required information
    if ( 'best_estimator_' not in standardized_results 
        or 'best_parameters_' not in standardized_results
    ):
        raise ValueError(
            "Required information ('best_estimator_' or 'best_parameters_') is missing.")

    # Inline contents preparation
    inline_contents = {
        "Best estimator": type(standardized_results['best_estimator_']).__name__,
        "Best parameters": standardized_results['best_parameters_'],
        "nCV": len({k for k in standardized_results['cv_results_'].keys(
            ) if k.startswith('split')}) // len(standardized_results['cv_results_']['params']),
        "Scoring": standardized_results.get('scoring', 'Unknown scoring')
    }

    # Preparing data for the CV results DataFrame
    df = prepare_cv_results_dataframe(standardized_results['cv_results_'])

    max_width = get_table_width(inline_contents)
    # Formatting CV results DataFrame
    formatted_table = format_dataframe(
        df, title="Tuning Results", max_width=max_width, 
        top_line=top_line, sub_line=sub_line, 
        bottom_line=bottom_line
    )
    max_width = len(formatted_table.split('\n')[0]) #
    # Combining inline content and formatted table
    summary_inline_tab = summarize_inline_table(
        inline_contents, title=title, max_width=max_width, 
        top_line=top_line, sub_line=sub_line, bottom_line=bottom_line
    )

    summary = f"{summary_inline_tab}\n\n{formatted_table}"
    return summary

def standardize_keys(model_results):
    map_keys = {
        'best_estimator_': ['model', 'estimator', 'best_estimator'],
        'best_parameters_': ['parameters', 'params', 'best_parameters'],
        'cv_results_': ['results', 'fold_results', 'cv_results']
    }

    standardized_results = {}
    for standard_key, alternatives in map_keys.items():
        # Check each alternative key for presence in model_results
        for alt_key in alternatives:
            if alt_key in model_results:
                standardized_results[standard_key] = model_results[alt_key]
                break  # Break after finding the first matching alternative
    
    # Additionally, check for any keys that are already correct and not duplicated
    for key in map_keys.keys():
        if key in model_results and key not in standardized_results:
            standardized_results[key] = model_results[key]

    return standardized_results

def prepare_cv_results_dataframe(cv_results):
    nCV = len({k for k in cv_results.keys() if k.startswith('split')}) // len(cv_results['params'])
    data = []
    for i in range(nCV):
        mean_score = np.mean([cv_results[f'split{j}_test_score'][i] for j in range(nCV)])
        cv_scores = [cv_results[f'split{j}_test_score'][i] for j in range(nCV)]
        std_score = np.std(cv_scores)
        global_mean = np.nanmean(cv_scores)
        
        fold_data = {
            "Fold": f"cv{i+1}",
            "Mean score": format_value (mean_score),
            "CV score": format_value(cv_scores[i]),
            "std score": format_value (std_score),
            "Global mean": format_value(global_mean)
        }
        data.append(fold_data)

    # Creating DataFrame
    df = pd.DataFrame(data)
    return df


def format_dataframe(
    df, title=None, 
    max_text_length=50, 
    max_width='auto',
    top_line='=', 
    sub_line='-', 
    bottom_line='='
    ):
    """
    Formats a pandas DataFrame into a string representation of a table,
    optionally including a title and customizing the table's appearance
    with specified line characters.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be formatted.
    title : str, optional
        The title of the table. If provided, it will be centered above
        the table. Defaults to None.
    max_text_length : int, optional
        The maximum allowed length for any text value in the table before
        truncation. Defaults to 50 characters.
    max_width : 'auto' or int, optional
        The maximum width of the table. If 'auto', the table width is
        automatically adjusted based on content. If an integer is provided,
        it specifies the maximum width; columns will be adjusted to fit.
        Defaults to 'auto'.
    top_line, sub_line, bottom_line : str, optional
        Characters used to create the top border, sub-header line, and
        bottom border of the table. Defaults to '=', '-', and '=' respectively.

    Returns
    -------
    str
        A string representing the formatted table.

    Examples
    --------
    >>> impport pandas as pd 
    >>> from gofast.api.summary import format_dataframe
    >>> data = {'Name': ['Alice', 'Bob', 'Charlie'],
                'Age': [25, 30, 35],
                'Occupation': ['Engineer', 'Doctor', 'Artist']}
    >>> df = pd.DataFrame(data)
    >>> print(format_dataframe(df, title='Employee Table'))
         Employee Table     
    ========================
         Name Age Occupation
    ------------------------
    0   Alice  25   Engineer
    1     Bob  30     Doctor
    2 Charlie  35     Artist
    ========================

    Notes
    -----
    - The function dynamically adjusts the width of the table columns based
      on the content size and the `max_width` parameter.
    - Long text values in cells are truncated with '...' when exceeding
      `max_text_length`.
    """
    # Calculate the max length for the index
    max_index_length = max([len(str(index)) for index in df.index])
    
    # Calculate max length for each column including the column name,
    # and account for truncation
    max_col_lengths = {
        col: max(len(col), max(df[col].astype(str).apply(
            lambda x: len(x) if len(x) <= max_text_length else max_text_length + 3)))
        for col in df.columns
    }
    initial_space = max_index_length + sum(
        max_col_lengths.values()) + len(df.columns) - 1  # Spaces between columns

    if isinstance(max_width, (float, int)) and max_width > initial_space:
        # Distribute the extra_space among columns and the index
        extra_space = max_width - initial_space
        extra_space_per_col = extra_space // (len(df.columns) + (1 if max_index_length > 0 else 0))
        max_col_lengths = {col: v + extra_space_per_col for col, v in max_col_lengths.items()}

    # Adjust column values for truncation and formatting
    for col in df.columns:
        df[col] = df[col].astype(str).apply(
            lambda x: x if len(x) <= max_text_length else x[:max_text_length] + "...")
        max_col_lengths[col] = max(max_col_lengths[col], df[col].str.len().max())
    
    # Construct the header with padding for alignment
    header = " ".join([f"{col:>{max_col_lengths[col]}}" for col in df.columns])
    
    # Calculate total table width considering the corrected space calculation
    table_width = sum(max_col_lengths.values()) + len(df.columns) - 1 + max_index_length + (
        1 if max_index_length > 0 else 0)
    
    # Construct the separators
    top_header_border = top_line * table_width
    separator = sub_line * table_width
    top_bottom_border = bottom_line * table_width
    
    # Construct each row
    rows = []
    title_str = f"{title.center(table_width)}" if title else ""
    for index, row in df.iterrows():
        row_str = f"{str(index).ljust(max_index_length)} " + " ".join(
            [f"{value:>{max_col_lengths[col]}}" for col, value in row.iteritems()])
        rows.append(row_str)
    
    # Combine all parts
    table = (f"{title_str}\n{top_header_border}\n{' ' * (max_index_length + 1)}"
             f"{header}\n{separator}\n") + "\n".join(rows) + f"\n{top_bottom_border}"
    
    return table

def format_key(key, max_length=None, include_colon=False, alignment='left'):
    """
    Formats a key string according to the specified parameters.

    Parameters:
    - key (str): The key to format.
    - max_length (int, optional): The maximum length for the formatted key. 
      If None, it's calculated based on the key length.
    - include_colon (bool): Determines whether a colon and space should be 
      added after the key.
    - alignment (str): Specifies the alignment of the key. Expected values are 
     'left' or 'right'.

    Returns:
    - str: The formatted key string.
    # Example usage:
    print(format_key("ExampleKey", 20, include_colon=True, alignment='left'))
    print(format_key("Short", max_length=10, include_colon=False, alignment='right'))
    
    """
    # Calculate the base length of the key, optionally including the colon and space
    base_length = len(key) + (2 if include_colon else 0)
    
    # Determine the final max length, using the base length if no max_length is provided
    final_length = max_length if max_length is not None else base_length
    
    # Construct the key format string
    key_format = f"{key}:" if include_colon else key
    
    # Apply the specified alignment and padding
    formatted_key = ( f"{key_format: <{final_length}}" if alignment == 'left' 
                     else f"{key_format: >{final_length}}"
                     )
    
    return formatted_key

def dataframe_key_format(key, df, max_key_length=None, max_text_char=50):
    # Format the key with a colon, using the provided or calculated max key length
    formatted_key = format_key(key, max_length=max_key_length,
                               include_colon=True, alignment='left')
    
    # Format the DataFrame according to specifications
    formatted_df = format_dataframe(df, max_long_text_char=max_text_char)
    
    # Split the formatted DataFrame into lines for alignment adjustments
    df_lines = formatted_df.split('\n')
    
    # Determine the number of spaces to prepend to each line based on the 
    # length of the formatted key
    # Adding 1 additional space for alignment under the formatted key
    space_prefix = ' ' * (len(formatted_key) + 1)
    
    # Prepend spaces to each line of the formatted DataFrame except for the 
    # first line (it's already aligned)
    aligned_df = space_prefix + df_lines[0] + '\n' + '\n'.join(
        [space_prefix + line for line in df_lines[1:]])
    
    # Combine the formatted key with the aligned DataFrame
    result = f"{formatted_key}\n{aligned_df}"
    
    return result

# Assuming format_key and format_dataframe are correctly implemented as discussed in previous steps
# Example usage would be as follows (after defining df):
# print(dataframe_key_format("Your Key Here", df))
       
# def dataframe_key_format( key, df,  max_key_length = None, max_text_char=50  ):
    
#     formatted_key = format_key ( key, max_key_length, include_colon= True ) 
    
#     formatted_df = format_dataframe(df, max_long_text_char= max_text_char)
    
    # once the key is formatted. Note the format_dataframe construct 
    # the formatted_df like below : 
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #             col1     col2     col3 
    # ----------------------------------
    # index1   value11  value12  value13
    # index2   value21  value12  value13
    # index3   value31  value13  value13
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    # Then now move the formatted_df based on the formatted key length by 
    # mentionned the key like below:
     
    # [formatted_key] 
    #                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #                         col1     col2     col3 
    #                ----------------------------------
    #                index1   value11  value12  value13
    #                index2   value21  value12  value13
    #                index3   value31  value13  value13
    #                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    # for instance if the formatted key = key            : 
    # formatted key with df should be : 
        
    # key            :
    #                ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #                         col1     col2     col3 
    #                ----------------------------------
    #                index1   value11  value12  value13
    #                index2   value21  value12  value13
    #                index3   value31  value13  value13

def format_dict(dct, ):
    # # Example usage with a mixed dictionary
    # mixed_dict = {
    #     "a": "apple",
    #     "b": 2,
    #     "c": 3.5,
    #     "d": float('nan'),
    #     "e": "banana"
    # }
    # print(format_dict(mixed_dict))

    # # Example usage with a numeric dictionary
    # numeric_dict = {
    #     "one": 1,
    #     "two": 2,
    #     "three": 3,
    #     "four": float('nan'),
    #     "five": 5
    # }
    # print(format_dict(numeric_dict))
    
    # Initialize counts
    num_values_count = 0
    non_num_values_count = 0
    exist_nan = False
    
    # Iterate through the dictionary values to categorize them
    for value in dct.values():
        if isinstance(value, (int, float)):
            num_values_count += 1
            if isinstance(value, float) and np.isnan(value):
                exist_nan = True
        else:
            non_num_values_count += 1
            
    # Determine dtype description
    dtype_description = ( 
        "numeric" if non_num_values_count == 0 else "mixed" 
        if num_values_count > 0 else "non-numeric"
        )
    
    # Calculate mean for numeric values if any, ignoring NaNs
    if num_values_count > 0:
        numeric_values = np.array([value for value in dct.values() if isinstance(
            value, (int, float))], dtype=float)
        mean_value = np.nanmean(numeric_values)
        summary_str = (
            "Dict ~ len:{} - values: <mean: {:.4f} - numval: {}"
            " - nonnumval: {} - dtype: {} - exist_nan:"
            " {}>").format(
            len(dct), mean_value, num_values_count, non_num_values_count,
            dtype_description, exist_nan
        )
    else:
        summary_str = ( 
            "Dict ~ len:{} - values: <numval: {} - nonnumval: {}"
            " - dtype: {} - exist_nan: {}>").format(
            len(dct), num_values_count, non_num_values_count, dtype_description,
            exist_nan
        )

    return summary_str


def format_list(lst, ):
    # Check if all elements in the list are numeric (int or float)
    all_numeric = all(isinstance(x, (int, float)) for x in lst)
    exist_nan = any(np.isnan(x) for x in lst if isinstance(x, float))

    if all_numeric:
        # Calculate mean for numeric list, ignoring NaNs
        numeric_values = np.array(lst, dtype=float)  # Convert list to NumPy array for nanmean calculation
        mean_value = np.nanmean(numeric_values)
        arr_str = "List ~ len:{} - values: < mean: {:.4f} - dtype: numeric - exist_nan: {}>".format(
            len( lst), mean_value, exist_nan
        )
    else:
        # For mixed or non-numeric lists, calculate the count of numeric and non-numeric values
        num_values_count = sum(isinstance(x, (int, float)) for x in lst)
        non_num_values_count = len(lst) - num_values_count
        dtype_description = "mixed" if not all_numeric else "non-numeric"
        arr_str = "List ~ len:{} - values: <numval: {} - nonnumval: {} - dtype: {} - exist_nan: {}>".format(
            len( lst), num_values_count, non_num_values_count, dtype_description, exist_nan
        )

    return arr_str


def format_array(arr, ):
    # # Example usage with a numeric array
    # numeric_arr = np.array([1, 2, 3, np.nan, 5])
    # print(format_array(numeric_arr))

    # # Example usage with a non-numeric (mixed) array
    # # This will not provide a meaningful summary for non-numeric data types,
    # # as the logic for non-numeric arrays would need to be more complex and might require pandas
    # mixed_arr = np.array(["apple", 2, 3.5, np.nan, "banana"], dtype=object)
    # print(format_array(mixed_arr))


    arr_str = ""
    # Check if the array contains NaN values; works only for numeric arrays
    exist_nan = np.isnan(arr).any() if np.issubdtype(arr.dtype, np.number) else False

    if np.issubdtype(arr.dtype, np.number):
        # Formatting string for numeric arrays
        arr_str = "Array ~ {}values: <mean: {:.4f} - dtype: {}{}>".format(
            f"shape=<{arr.shape[0]}{'x' + str(arr.shape[1]) if arr.ndim == 2 else ''}> - ",
            np.nanmean(arr),  # Use nanmean to calculate mean while ignoring NaNs
            arr.dtype.name,
            ' - exist_nan:True' if exist_nan else ''
        )
    else:
        # Attempt to handle mixed or non-numeric data without pandas,
        # acknowledging that numpy arrays are typically of a single data type
        # Here we consider the array non-numeric and don't attempt to summarize numerically
        dtype_description = "mixed" if arr.dtype == object else arr.dtype.name
        arr_str = "Array ~ {}values: <dtype: {}{}>".format(
            f"shape=<{arr.shape[0]}{'x' + str(arr.shape[1]) if arr.ndim == 2 else ''}> - ",
            dtype_description,
            ' - exist_nan:True' if exist_nan else ''
        )

    return arr_str




def format_text(text, key=None, key_length=15, max_char_text=50):
    # Example usage:
    # text_example = "This is an example text that is supposed to wrap around after a 
    # certain number of characters, demonstrating the function."

    # # # Test with key and default key_length
    # print(format_text(text_example, key="ExampleKey"))

    # # Test with key and custom key_length
    # print(format_text(text_example, key="Key", key_length=10))

    # # Test without key but with key_length
    # print(format_text(text_example, key_length=5))

    # # Test without key and key_length
    # print(format_text(text_example))
    
    if key is not None:
        # If key_length is None, use the length of the key + 1 for the space after the key
        if key_length is None:
            key_length = len(key) + 1
        key_str = f"{key.ljust(key_length)}: "
    elif key_length is not None:
        # If key is None but key_length is specified, use spaces
        key_str = " " * key_length + ": "
    else:
        # If both key and key_length are None, there's no key part
        key_str = ""
    
    # Adjust max_char_text based on the length of the key part
    effective_max_char_text = max_char_text - len(key_str) + 2 if key_str else max_char_text

    formatted_text = ""
    while text:
        # If the remaining text is shorter than the effective max length, or if there's no key part, add it as is
        if len(text) <= effective_max_char_text or not key_str:
            formatted_text += key_str + text
            break
        else:
            # Find the space to break the line, ensuring it doesn't exceed effective_max_char_text
            break_point = text.rfind(' ', 0, effective_max_char_text)
            if break_point == -1:  # No spaces found, force break
                break_point = effective_max_char_text
            # Add the line to formatted_text
            formatted_text += key_str + text[:break_point].rstrip() + "\n"
            # Remove the added part from text
            text = text[break_point:].lstrip()
            # After the first line, the key part is just spaces
            key_str = " " * len(key_str)
    
    return formatted_text


def format_series(series):
    # Example usage:
    # For a numeric series
    # numeric_series = pd.Series([1, 2, 3, np.nan, 5, 6], name='NumericSeries')
    # print(_format_series(numeric_series))

    # # For a non-numeric series
    # non_numeric_series = pd.Series(['apple', 'banana', np.nan, 'cherry', 'date', 'eggfruit', 'fig'], name='FruitSeries')
    # print(_format_series(non_numeric_series))

    series_str = ''
    if not isinstance ( series, pd.Series):
        return series 
    
    if series.dtype.kind in 'biufc' and len(series) < 7:  # Check if series is numeric and length < 7
        series_str = "Series ~ {}values: <mean: {:.4f} - length: {} - dtype: {}{}>".format(
            f"name=<{series.name}> - " if series.name is not None else '',
            np.mean(series.values), 
            len(series), 
            series.dtype.name, 
            ' - exist_nan:True' if series.isnull().any() else ''
        )
    else:  # Handle non-numeric series or series with length >= 7
        num_values = series.apply(lambda x: isinstance(x, (int, float)) and not np.isnan(x)).sum()
        non_num_values = len(series) - num_values - series.isnull().sum()  # Exclude NaNs from non-numeric count
        series_str = "Series ~ {}values: <numval: {} - nonnumval: {} - length: {} - dtype: {}{}>".format(
            f"name=<{series.name}> - " if series.name is not None else '',
            num_values, 
            non_num_values, 
            len(series), 
            series.dtype.name, 
            ' - exist_nan:True' if series.isnull().any() else ''
        )
    return series_str


# Example usage
if __name__ == "__main__":
    # Example usage:
    report_data = {
        'Total Sales': 123456.789,
        'Average Rating': 4.321,
        'Number of Reviews': 987,
        'Key with long name': 'Example text'
    }

    report_title = 'Monthly Sales Report'
    formatted_report = format_report(report_data, report_title)
    print(formatted_report)
    
    # report = Report(title="Example Report")
    # report.add_data_ops_report({
    #     "key1": np.random.random(),
    #     "key2": "value2",
    #     "key3": pd.Series([1, 2, 3], name="series_name"),
    #     "key4": pd.DataFrame(np.random.rand(4, 3), columns=["col1", "col2", "col3"])
    # })
    # print(report)
    
    # # Creating an example DataFrame
    df_data = {
        'A': [1, 2, 3, 4, 5],
        'B': [5, 6, None, 8, 9],
        'C': ['foo', 'bar', 'baz', 'qux', 'quux'],
        'D': [0.1, 0.2, 0.3, np.nan, 0.5]
    }
    df = CustomDataFrame(df_data)
    
    # Customizing the summary
    summary_ = df.summary(include_correlation=True, include_uniques=True, 
                         statistics=['mean', '50%', 'max'], include_sample=True
                         )
    for key, value in summary_.items():
        print(f"{key}:")
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                print(f"  {subkey}: {subvalue}")
        else:
            print(f"  {value}")