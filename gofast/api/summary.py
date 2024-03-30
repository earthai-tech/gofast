# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from .structures import FlexDict

class Summary(FlexDict):
    """
    A utility class for generating detailed summary reports of pandas DataFrames. 
    It inherits from `FlexDict`, enabling attribute-like access to dictionary 
    keys for flexible interaction with summary data. The class provides methods 
    to compile comprehensive insights into the DataFrame, including basic statistics, 
    correlation matrices, counts of unique values in categorical columns, and 
    data samples.

    Attributes
    ----------
    title : str
        An optional title for the summary report. If specified, this title 
        will be included at the top of the generated summary.

    Methods
    -------
    data_summary(df, include_correlation=False, include_uniques=False, 
                     include_sample=False, sample_size=5):
        Produces a formatted summary of the given DataFrame, optionally 
        including a correlation matrix, unique value counts for categorical 
        columns, and a data sample.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.api.summary import Summary 
    >>> data = {
    ...     'Age': [25, 30, 35, 40, np.nan],
    ...     'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Edward'],
    ...     'Salary': [50000, 60000, 70000, 80000, 90000]
    ... }
    >>> df = pd.DataFrame(data)
    >>> summary = Summary(title="Employee Data Overview")
    >>> summary.data_summary(df, include_correlation=True, 
    ...           include_uniques=True, include_sample=True))

    Notes
    -----
    - The summary report is designed to provide a quick yet comprehensive 
      overview of the DataFrame's content, facilitating initial data exploration 
      and analysis tasks.
    - `FlexDict` is assumed to be a class that allows dictionary items to be 
      accessed as attributes, enhancing ease of use. This behavior enriches the 
      `Summary` class by enabling dynamic attribute assignment based 
      on the provided summary data.
    - External functions `format_dataframe` and others are leveraged for 
      formatting sections of the summary report. These functions must be defined 
      and available in the same scope or imported for the `Summary` class 
      to function correctly.
    - The class focuses on numerical and categorical data within the DataFrame. 
      Other data types (e.g., datetime) are included in the basic statistics but 
      might require specialized handling for more detailed analysis.
    """
    
    def __init__(self, title=None, **kwargs):
        """
        Initializes the Summary object with an optional title for the 
        report and any additional properties via keyword arguments.

        Parameters
        ----------
        title : str, optional
            The title of the summary report to be generated.
        **kwargs : dict, optional
            Additional keyword arguments passed to the FlexDict constructor.
        """
        super().__init__(**kwargs)
        self.title = title
        self.summary_report = ""

    def basic_statistics(self, df, include_correlation=False):
        """
        Generates basic statistical measures for the provided DataFrame and,
        optionally, a correlation matrix for numeric columns.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to be summarized.
        include_correlation : bool, optional
            If True, includes a correlation matrix for numeric columns in the
            summary. Defaults to False.

        Returns
        -------
        Summary
            The instance itself, allowing for method chaining.

        Raises
        ------
        ValueError
            If `df` is not a pandas DataFrame.

        Examples
        --------
        >>> import pandas as pd 
        >>> from gofast.api.summary import Summary
        >>> df = pd.DataFrame({
        ...     'Age': [25, 30, 35, 40, np.nan],
        ...     'Salary': [50000, 60000, 70000, 80000, 90000]
        ... })
        >>> summary = Summary(title="Employee Stats")
        >>> summary.generate_basic_statistics(df, include_correlation=True)
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The provided data must be a pandas DataFrame.")
        
        summary_parts = []
        
        # Basic statistics
        stats = df.describe(include='all').T
        summary_parts.append(format_dataframe(stats, title="Basic Statistics"))

        # Correlation matrix
        if include_correlation and 'number' in df.select_dtypes(
                include=['number']).dtypes:
            corr_matrix = df.corr().round(2)
            summary_parts.append(format_dataframe(
                corr_matrix, title="Correlation Matrix")) 
            
        # Compile all parts into a single summary report
        self.summary_report = "\n\n".join(summary_parts)
        
        return self 

    def unique_counts(self, df, include_sample=False, sample_size=5):
        """
        Generates counts of unique values for categorical columns in the provided
        DataFrame and, optionally, includes a random sample of data.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to be summarized.
        include_sample : bool, optional
            If True, adds a random sample of data to the summary. Defaults to False.
        sample_size : int, optional
            The size of the sample to include if `include_sample` is True. Defaults
            to 5.

        Returns
        -------
        Summary
            The instance itself, allowing for method chaining.

        Raises
        ------
        ValueError
            If `df` is not a pandas DataFrame.

        Examples
        --------
        >>> import pandas as pd 
        >>> from gofast.api.summary import Summary
        >>> df = pd.DataFrame({
        ...     'Department': ['Sales', 'HR', 'IT', 'Sales', 'HR'],
        ...     'Age': [28, 34, 45, 32, 41]
        ... })
        >>> summary = Summary(title="Department Overview")
        >>> summary.unique_counts(df, include_sample=True)
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        
        summary_parts=[] 
        unique_counts = {col: df[col].nunique() for col in df.select_dtypes(
            include=['object', 'category']).columns}
        unique_df = pd.DataFrame.from_dict(
            unique_counts, orient='index', columns=['Unique Counts'])
        max_width= get_table_width(unique_df) 
        # Sample of the data
        if include_sample:
            sample_data = df.sample(n=min(sample_size, len(df)))
            it_width =get_table_width(sample_data)
            max_width = max_width if max_width > it_width else it_width
            
        # get the max width to fit all tables 
        summary_parts.append(format_dataframe(unique_df,  
            title="Unique Counts", max_width= max_width))
        
        if include_sample: 
            summary_parts.append(format_dataframe(
                sample_data, title="Sample Data", max_width=max_width))
         # Compile all parts into a single summary report
        self.summary_report = "\n\n".join(summary_parts)

        return self

    def model_summary(self, model_results=None, model=None, **kwargs):
        """
        Generates a summary report for a scikit-learn model or model results,
        especially for models optimized using GridSearchCV or RandomizedSearchCV.

        Parameters
        ----------
        model_results : dict, optional
            A dictionary containing model results with keys 'best_estimator_',
            'best_params_', and 'cv_results_'. If provided, used directly for
            generating the summary report.
        model : sklearn estimator, optional
            A scikit-learn model object. Must have 'best_estimator_', 'best_params_',
            and 'cv_results_' attributes if `model_results` is not provided.
        **kwargs : dict
            Additional keyword arguments to be passed to the summary generation 
            function.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If neither `model_results` nor `model` with required attributes is provided.
        """
        if model_results is None:
            if model and all(hasattr(model, attr) for attr in [
                    'best_estimator_', 'best_params_', 'cv_results_']):
                model_results = {
                    'best_estimator_': model.best_estimator_,
                    'best_params_': model.best_params_,
                    'cv_results_': model.cv_results_
                }
            else:
                raise ValueError(
                    "Either 'model_results' must be provided or 'model' must have "
                    "'best_estimator_', 'best_params_', and 'cv_results_' attributes.")
        
        self.summary_report = summarize_model_results(
            model_results, title=self.title, **kwargs)
        
        return self

    def __str__(self):
        """
        String representation of the summary report.
        """
        return self.summary_report

    def __repr__(self):
        """
        Formal string representation indicating whether the summary report 
        is empty or populated.
        """
        return "<Summary: {}>".format(
            "Empty" if not self.summary_report else "Populated. Use print() to the contents.")



class ReportFactory(FlexDict):
    """
    Represents a dynamic report generator capable of handling and formatting
    various types of data into comprehensive reports. Inherits from `FlexDict`,
    allowing attribute-like access to dictionary keys.

    The `Report` class facilitates the creation of structured text reports from
    mixed data types, recommendations, performance summaries of models, and
    summaries of pandas DataFrames. It provides methods to format each report
    section and compiles them into a single cohesive text representation.

    Attributes
    ----------
    title : str
        The title of the report. If provided, it is included at the top of
        the report output.
    report : dict or str or pandas.DataFrame
        The raw data used to generate the report section last processed.
        This attribute stores the input provided to the last method called
        (e.g., mixed data, recommendations text, model results, or DataFrame).
    report_str : str
        The formatted text representation of the report or the last report
        section processed. This string is ready for printing or logging.

    Methods
    -------
    mixed_types_summary(report, table_width=100):
        Formats a report containing mixed data types into a structured text
        representation.

    add_recommendations(text, key=None, **kwargs):
        Adds a recommendations section to the report, formatted with an
        optional key.

    model_performance_summary(model_results, **kwargs):
        Adds a model performance summary section to the report, using the
        results from model evaluation.

    data_summary(df, **kwargs):
        Adds a summary of a pandas DataFrame to the report, formatted as a
        structured text section.

    Examples
    --------
    >>> from gofast.api.summary import ReportFactory
    >>> report_data = {
    ...     'Total Sales': 123456.789,
    ...     'Average Rating': 4.321,
    ...     'Number of Reviews': 987
    ... }
    >>> report = ReportFactory(title="Sales Summary")
    >>> report.mixed_types_summary(report_data)
    >>> print(report)
    ================================
             Sales Summary          
    --------------------------------
    Total Sales       : 123456.7890
    Average Rating    : 4.3210
    Number of Reviews : 987
    ================================
    
    Notes
    -----
    - This class depends on external formatting functions like `format_report`,
      `format_text`, `summarize_model_results`, and `format_dataframe`, which
      need to be defined in the same scope or imported.
    - The `FlexDict` parent class is assumed to provide dynamic attribute access,
      allowing for flexible interaction with report data.
    - While `report_str` holds the formatted report ready for display, `report`
      maintains the raw input data for reference or further processing.
    """
    def __init__(self, title=None, **kwargs):
        """
        Initializes the Report object with an optional title and additional
        keyword arguments passed to the FlexDict constructor.

        Parameters
        ----------
        title : str, optional
            The title of the report. Defaults to None.
        **kwargs : dict
            Additional keyword arguments passed to the FlexDict constructor,
            allowing for dynamic attribute assignment based on the provided
            dictionary.
        """
        super().__init__(**kwargs)
        self.title = title
        self.report = None
        self.report_str = None

    def mixed_types_summary(self, report, table_width=100):
        """
        Formats a report containing mixed data types.

        Parameters:
        - report (dict): The report data.
        - table_width (int, optional): The maximum width of the table. Defaults to 100.
        """
        self.report = report
        self.report_str = format_report(report_data=report, report_title=self.title,
                                        max_table_width=table_width)

    def add_recommendations(self, text, key=None, **kwargs):
        """
        Formats and adds a recommendations section to the report.

        Parameters:
        - text (str): The text containing recommendations.
        - key (str, optional): An optional key to prefix the text. Defaults to None.
        """
        self.report = text
        self.report_str = format_text(text, key=key, **kwargs)

    def model_performance_summary(self, model_results, **kwargs):
        """
        Formats and adds a model performance summary to the report.

        Parameters:
        - model_results (dict): The results of the model performance evaluation.
        """
        self.report = model_results
        self.report_str = summarize_model_results(
            model_results, title=self.title, **kwargs)

    def data_summary(self, df, **kwargs):
        """
        Formats and adds a data frame summary to the report.

        Parameters:
        - df (pandas.DataFrame): The data frame to summarize.
        """
        self.report_str = format_dataframe(df, title=self.title, **kwargs)

    def __str__(self):
        """
        String representation of the report content.
        """
        return self.report_str or "<No report>"

    def __repr__(self):
        """
        Formal string representation of the Report object.
        """
        return "<Report: Print to see the content>" if self.report else "<Report: No content>"


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

def calculate_maximum_length( report_data, max_table_length = 70 ): 
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
    if (max_key_length + max_val_length) >=max_table_length:  # @ 4 for spaces 
        max_val_length = max_table_length - max_key_length -4
        
    return max_key_length, max_val_length
       
def format_value( value ): 
    value_str =str(value)
    if isinstance(value, (int, float, np.integer, np.floating)): 
        value_str = f"{value}" if isinstance ( value, int) else  f"{float(value):.4f}" 

    return value_str

def format_report(report_data, report_title=None, max_table_width= 70 ):
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
    max_key_length, max_val_length  = calculate_maximum_length( 
        report_data, max_table_length=max_table_width)
    
    # Prepare the report title and frame lines
    # Adjust for key-value spacing and aesthetics
    line_length = max_key_length + max_val_length + 4 
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
            # report_str += f"{key.ljust(max_key_length)} :  {formatted_value}\n"
        elif isinstance ( value, ( list, tuple )): 
            formatted_value = format_list(list( value))
            
        elif isinstance(value, np.ndarray): 
            formatted_value = format_array(value)
        elif isinstance ( value, pd.Series): 
            formatted_value = format_series(value)

        elif isinstance ( value, dict): 
            formatted_value = format_dict ( value )
            
        elif isinstance(value, pd.DataFrame): 
            formatted_value = dataframe_key_format(
                key, value, max_key_length=max_key_length, 
                include_colon= True, max_text_char= max_val_length, 
                alignment='left', pad_colon= True)
        else: # any other things consider as text 
            formatted_value = str(value )
        # if isinstance ( value, str): 
            # Exclude pd.DataFrame which is already formatted as table. 
        if not isinstance ( value, pd.DataFrame):
            # then considered as a text  
            report_str += format_text(
                formatted_value, key= key, key_length=max_key_length, 
                max_char_text= line_length) +'\n'
            #report_str += f"{key.ljust(max_key_length)} :  {formatted_value}\n"
        else: # just append the dtaframe already formated 
            report_str += formatted_value +'\n' 
  
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
    nCV = len({k for k in cv_results.keys() 
               if k.startswith('split')}) // len(cv_results['params'])
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


def format_key(key, max_length=None, include_colon=False, alignment='left',
               pad_colon=False):
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
    #pad key + space + ' :' to fit the max_length , adjust according when padding_colon is True
    key_format = "{}{} :".format(
        key, ' '*(max_length -len(key)) if pad_colon else '') if include_colon else key
    # Apply the specified alignment and padding
    formatted_key = ( f"{key_format: <{final_length}}" if alignment == 'left' 
                     else f"{key_format: >{final_length}}"
                     )
    return formatted_key

def dataframe_key_format(
    key, df, title ='',
    max_key_length=None, 
    max_text_char=50, 
    **kws
    ):
    """
    Formats a key-value pair where the value is a pandas DataFrame, aligning
    the DataFrame under a formatted key with an optional maximum key length and
    maximum text character width for the DataFrame.

    Parameters
    ----------
    key : str
        The key associated with the DataFrame, which will be formatted with
        a colon and aligned to the left.
    df : pandas.DataFrame
        The DataFrame to be formatted and aligned under the key.
    max_key_length : int, optional
        The maximum length of the key. If None, the actual length of the `key`
        is used. Defaults to None.
    max_text_char : int, optional
        The maximum number of characters allowed for each cell within the
        DataFrame before truncation. Defaults to 50.

    Returns
    -------
    str
        The formatted key followed by the aligned DataFrame as a string.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.api.summary import dataframe_key_format
    >>> df = pd.DataFrame({'A': [1, 2], 'B': ['text', 'another longer text']})
    >>> key = 'DataFrame Key'
    >>> print(dataframe_key_format(key, df))
    DataFrame Key:
                    A   B
                    1   text
                    2   another longer text

    Notes
    -----
    - The function is particularly useful for including pandas DataFrames within
      textual reports or summaries, ensuring the DataFrame's alignment matches
      the accompanying textual content.
    - The `format_key` and `format_dataframe` helper functions are utilized to
      format the key and DataFrame, respectively. These should be defined to
      handle specific formatting and alignment needs.
    - If `max_key_length` is provided and exceeds the actual length of the `key`,
      additional spaces are added to align the DataFrame's first column directly
      under the formatted key.
    """
    # Format the key with a colon, using the provided or calculated max key length
    # if %% in the key split and considered key and title 
    if "%%" in str(key): 
        key, title = key.split("%%")
        
    formatted_key = format_key(key, max_length=max_key_length,**kws)
    
    # Format the DataFrame according to specifications
    formatted_df = format_dataframe(df, max_text_length=max_text_char)
    
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
    # Center the title of dataframe 
    if title: 
        title = title.title ().center(len(df_lines[0]))
    
    # Combine the formatted title dataframe with the aligned DataFrame
    result = f"{formatted_key}{title}\n{aligned_df}"
    
    return result

def format_dict(dct):
    """
    Formats a dictionary into a summary string that provides an overview
    of its content, distinguishing between numeric and non-numeric values
    and identifying the presence of NaN values among numeric entries.

    Parameters
    ----------
    dct : dict
        The dictionary to summarize, which can contain a mix of numeric
        and non-numeric values.

    Returns
    -------
    str
        A summary string of the dictionary's contents, including mean values
        for numeric data and counts of numeric vs. non-numeric entries.

    Examples
    --------
    >>> from gofast.api.summary import format_dict
    >>> mixed_dict = {
    ...     "a": "apple",
    ...     "b": 2,
    ...     "c": 3.5,
    ...     "d": float('nan'),
    ...     "e": "banana"
    ... }
    >>> print(format_dict(mixed_dict))
    Dict ~ len:5 - values: <mean: 2.7500 - numval: 3 - nonnumval: 2 ...>

    >>> numeric_dict = {
    ...     "one": 1,
    ...     "two": 2,
    ...     "three": 3,
    ...     "four": float('nan'),
    ...     "five": 5
    ... }
    >>> print(format_dict(numeric_dict))
    Dict ~ len:5 - values: <mean: 2.7500 - numval: 4 - nonnumval: 0 ...>

    Notes
    -----
    - The function calculates the mean value only for numeric entries, ignoring
      any NaN values in the calculation.
    - The function identifies the presence of NaN values among numeric entries
      and reflects this in the summary.
    - Non-numeric entries are counted separately, and the dictionary is classified
      as 'numeric', 'mixed', or 'non-numeric' based on its contents.
    """
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


def format_list(lst):
    """
    Formats a list into a summary string, identifying whether the list
    is purely numeric, mixed, or non-numeric, and includes statistics
    like mean values for numeric lists and the presence of NaN values.

    Parameters
    ----------
    lst : list
        The list to summarize, which may contain numeric, non-numeric,
        or a mix of both types of values.

    Returns
    -------
    str
        A summary string of the list's contents, including the overall type
        (numeric, mixed, or non-numeric), mean values for numeric entries,
        and the presence of NaN values if applicable.

    Examples
    --------
    >>> from gofast.api.summary import format_list
    >>> numeric_list = [1, 2, 3, np.nan, 5]
    >>> print(format_list(numeric_list))
    List ~ len:5 - values: <mean: 2.7500 - dtype: numeric - exist_nan: True>

    >>> mixed_list = ["apple", 2, 3.5, np.nan, "banana"]
    >>> print(format_list(mixed_list))
    List ~ len:5 - values: <numval: 2 - nonnumval: 3 - dtype: mixed - exist_nan: True>

    Notes
    -----
    - Numeric entries are processed to calculate a mean value, excluding any NaNs.
    - The presence of NaN values among numeric entries is noted in the summary.
    - The classification of the list as 'numeric', 'mixed', or 'non-numeric' is
      based on the types of values it contains.
    """
    # Check if all elements in the list are numeric (int or float)
    all_numeric = all(isinstance(x, (int, float)) for x in lst)
    exist_nan = any(np.isnan(x) for x in lst if isinstance(x, float))

    if all_numeric:
        # Calculate mean for numeric list, ignoring NaNs
        # # Convert list to NumPy array for nanmean calculation
        numeric_values = np.array(lst, dtype=float)  
        mean_value = np.nanmean(numeric_values)
        arr_str = ("List ~ len:{} - values: < mean: {:.4f} - dtype:"
                   " numeric - exist_nan: {}>").format(
            len( lst), mean_value, exist_nan
        )
    else:
        # For mixed or non-numeric lists, calculate the count of numeric and non-numeric values
        num_values_count = sum(isinstance(x, (int, float)) for x in lst)
        non_num_values_count = len(lst) - num_values_count
        dtype_description = "mixed" if not all_numeric else "non-numeric"
        arr_str = ( "List ~ len:{} - values: <numval: {} - nonnumval: {}"
                   " - dtype: {} - exist_nan: {}>").format(
            len( lst), num_values_count, non_num_values_count, dtype_description, exist_nan
        )

    return arr_str

def format_array(arr):
    """
    Formats a NumPy array into a summary string, calculating mean values
    for numeric arrays and identifying the presence of NaN values. Non-numeric
    arrays are noted as such without attempting to summarize their contents.

    Parameters
    ----------
    arr : numpy.ndarray
        The NumPy array to summarize, which can be numeric or non-numeric.

    Returns
    -------
    str
        A summary string of the array's contents, including shape, mean value
        for numeric arrays, and the detection of NaN values if present.

    Examples
    --------
    >>> from gofast.api.summary import format_array
    >>> numeric_arr = np.array([1, 2, 3, np.nan, 5])
    >>> print(format_array(numeric_arr))
    Array ~ shape=<5> - mean: 2.7500 - dtype: float64 - exist_nan:True

    >>> mixed_arr = np.array(["apple", 2, 3.5, np.nan, "banana"], dtype=object)
    >>> print(format_array(mixed_arr))
    Array ~ shape=<5> - dtype: object - exist_nan:True

    Notes
    -----
    - For numeric arrays, the function calculates the mean while ignoring any NaNs
      and identifies the presence of NaN values.
    - Non-numeric or mixed-type arrays are labeled with their data type without
      attempting numerical summarization.
    """
 
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
    """
    Formats a block of text to fit within a specified maximum character width,
    optionally prefixing it with a key. If the text exceeds the maximum width,
    it wraps to a new line, aligning with the key or the specified indentation.

    Parameters
    ----------
    text : str
        The text to be formatted.
    key : str, optional
        An optional key to prefix the text. Defaults to None.
    key_length : int, optional
        The length reserved for the key, including following spaces.
        If `key` is provided but `key_length` is None, the length of the
        `key` plus one space is used. Defaults to 15.
    max_char_text : int, optional
        The maximum number of characters for the text width, including the key
        if present. Defaults to 50.

    Returns
    -------
    str
        The formatted text with line breaks added to ensure that no line exceeds
        `max_char_text` characters. If a `key` is provided, it is included only
        on the first line, with subsequent lines aligned accordingly.

    Examples
    --------
    >>> from gofast.api.summary import format_text
    >>> text_example = ("This is an example text that is supposed to wrap" 
                      "around after a certain number of characters.")
    >>> print(format_text(text_example, key="Note"))
    Note           : This is an example text that is supposed to
                      wrap around after a certain number of
                      characters.

    Notes
    -----
    - The function dynamically adjusts the text to fit within `max_char_text`,
      taking into account the length of `key` if provided.
    - Text that exceeds the `max_char_text` limit is wrapped to new lines, with
      proper alignment to match the initial line's formatting.
    """

    if key is not None:
        # If key_length is None, use the length of the key + 1 
        # for the space after the key
        if key_length is None:
            key_length = len(key) + 1
        key_str = f"{key.ljust(key_length)} : "
    elif key_length is not None:
        # If key is None but key_length is specified, use spaces
        key_str = " " * key_length + " : "
    else:
        # If both key and key_length are None, there's no key part
        key_str = ""
    
    # Adjust max_char_text based on the length of the key part
    effective_max_char_text = max_char_text - len(key_str) + 2 if key_str else max_char_text

    formatted_text = ""
    while text:
        # If the remaining text is shorter than the effective
        # max length, or if there's no key part, add it as is
        if len(text) <= effective_max_char_text or not key_str:
            formatted_text += key_str + text
            break
        else:
            # Find the space to break the line, ensuring it doesn't
            # exceed effective_max_char_text
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
    """
    Formats a pandas Series into a concise summary string. The summary includes
    the series name, mean (for numeric series), length, dtype, and an indicator
    of whether NaN values are present.

    Parameters
    ----------
    series : pandas.Series
        The series to be summarized.

    Returns
    -------
    str
        A summary string describing key aspects of the series.

    Examples
    --------
    >>> numeric_series = pd.Series([1, 2, 3, np.nan, 5, 6], name='NumericSeries')
    >>> print(format_series(numeric_series))
    Series ~ name=<NumericSeries> - values: <mean: 3.4000 - length: 6 -...>

    >>> non_numeric_series = pd.Series(['apple', 'banana', np.nan, 'cherry',
                                        'date', 'eggfruit', 'fig'], name='FruitSeries')
    >>> print(format_series(non_numeric_series))
    Series ~ name=<FruitSeries> - values: <numval: 0 - nonnumval: 6 - ...>

    Notes
    -----
    - For numeric series with less than 7 values, the function calculates and
      includes the mean value, excluding any NaNs from the calculation.
    - For non-numeric series or those with 7 or more values, the function counts
      the number of numeric and non-numeric values separately and indicates the
      presence of NaN values.
    - The series' data type (`dtype`) and the presence of NaN values are always
      included in the summary.
    """
    series_str = ''
    if not isinstance ( series, pd.Series):
        return series 
    
    if series.dtype.kind in 'biufc' and len(series) < 7:  # Check if series is numeric and length < 7
        series_str = "Series ~ {}values: <mean: {:.4f} - len: {} - dtype: {}{}>".format(
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

        
        
# Example usage demonstration and specific helper functions like `format_dataframe` 
# and `summarize_inline_table` need to be implemented separately.

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


if __name__ == "__main__":
    # Example usage:
    # from gofast.api.summary import format_report 
    report_data = {
        'Total Sales': 123456.789,
        'Average Rating': 4.321,
        'Number of Reviews': 987,
        'Key with long name': 'Example text', 
        'series': pd.Series ([1, 'banana', float('nan')])
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
    