# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

import re
import warnings
import numpy as np 
import pandas as pd

def find_maximum_table_width(summary_contents, header_marker='='):
    """
    Calculates the maximum width of tables in a summary string based on header lines.

    This function parses a multi-table summary string, identifying lines that represent
    the top or bottom borders of tables (header lines). It determines the maximum width
    of these tables by measuring the length of these header lines. The function assumes
    that the header lines consist of repeated instances of a specific marker character.

    Parameters
    ----------
    summary_contents : str
        A string containing the summarized representation of one or more tables.
        This string should include header lines made up of repeated header markers
        that denote the start and end of each table's border.
    header_marker : str, optional
        The character used to construct the header lines in the summary_contents.
        Defaults to '=', the common character for denoting table borders in ASCII
        table representations.

    Returns
    -------
    int
        The maximum width of the tables found in summary_contents, measured as the
        length of the longest header line. If no header lines are found, returns 0.

    Examples
    --------
    >>> from gofast.api.util import find_maximum_table_width
    >>> summary = '''Model Performance
    ... ===============
    ... Estimator : SVC
    ... Accuracy  : 0.9500
    ... Precision : 0.8900
    ... Recall    : 0.9300
    ... ===============
    ... Model Performance
    ... =================
    ... Estimator : RandomForest
    ... Accuracy  : 0.9500
    ... Precision : 0.8900
    ... Recall    : 0.9300
    ... ================='''
    >>> find_maximum_table_width(summary)
    18

    This example shows how the function can be used to find the maximum table width
    in a string containing summaries of model performances, where '=' is used as
    the header marker.
    """
    # Split the input string into lines
    lines = summary_contents.split('\n')
    # Filter out lines that consist only of the header marker, and measure their lengths
    header_line_lengths = [len(line) for line in lines if line.strip(header_marker) == '']
    # Return the maximum of these lengths, or 0 if the list is empty
    return max(header_line_lengths, default=0)

def format_text(
        text, key=None, key_length=15, max_char_text=50, 
        add_frame_lines =False, border_line='=' ):
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
    add_frame_lines: bool, False 
       If True, frame the text with '=' line (top and bottom)
    border_line: str, optional 
      The border line to frame the text.  Default is '='
      
    Returns
    -------
    str
        The formatted text with line breaks added to ensure that no line exceeds
        `max_char_text` characters. If a `key` is provided, it is included only
        on the first line, with subsequent lines aligned accordingly.

    Examples
    --------
    >>> from gofast.api.util import format_text
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
    effective_max_char_text = (max_char_text - len(key_str) + 2 if key_str else max_char_text)
    formatted_text = ""
    text=str(text)
    while text:
        # If the remaining text is shorter than the effective
        # max length, or if there's no key part, add it as is
        if len(text) <= effective_max_char_text - 4 or not key_str: # -4 for extraspace 
            formatted_text += key_str + text
            break
        else:
            # Find the space to break the line, ensuring it doesn't
            # exceed effective_max_char_text
            break_point = text.rfind(' ', 0, effective_max_char_text-4)
            
            if break_point == -1:  # No spaces found, force break
                break_point = effective_max_char_text -4 
            # Add the line to formatted_text
            formatted_text += key_str + text[:break_point].rstrip() + "\n"
            # Remove the added part from text
            text = text[break_point:].lstrip()
   
            # After the first line, the key part is just spaces
            key_str = " " * len(key_str)

    if add_frame_lines: 
        frame_lines = border_line * (effective_max_char_text + 1 )
        formatted_text = frame_lines +'\n' + formatted_text +'\n' + frame_lines

    return formatted_text


def format_value(value):
    """
    Format a numeric value to a string, rounding floats to four decimal
    places and converting integers directly to strings.
    
    Parameters
    ----------
    value : int, float, np.integer, np.floating
        The numeric value to be formatted.

    Returns
    -------
    str
        A formatted string representing the value.
    
    Examples
    --------
    >>> from gofast.api.util import format_value
    >>> format_value(123)
    '123'
    >>> format_value(123.45678)
    '123.4568'
    """
    value_str =str(value)
    if isinstance(value, (int, float, np.integer, np.floating)): 
        value_str = f"{value}" if isinstance ( 
            value, int) else  f"{float(value):.4f}" 
    return value_str 

def get_frame_chars(frame_char):
    """
    Retrieve framing characters based on the input frame indicator.
    
    Parameters
    ----------
    frame_char : str
        A single character that indicates the desired framing style.

    Returns
    -------
    tuple
        A tuple containing the close character and the open-close pair
        for framing index values.

    Examples
    --------
    >>> from gofast.api.util import get_frame_chars
    >>> get_frame_chars('[')
    (']', '[', ']')
    >>> get_frame_chars('{')
    ('}', '{', '}')
    """
    pairs = {
        '[': (']', '[', ']'),
        '{': ('}', '{', '}'),
        '(': (')', '(', ')'),
        '<': ('>', '<', '>')
    }
    return pairs.get(frame_char, ('.', '.', '.'))

def df_to_custom_dict(df, key_prefix='Row', frame_char='['):
    """
    Convert a DataFrame to a dictionary with custom formatting for keys
    and values, applying specified framing characters for non-unique 
    numeric indices.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to convert.
    key_prefix : str, optional
        The prefix for keys when the index is numeric and non-unique,
        default is 'Row'.
    frame_char : str, optional
        The character to determine framing for keys, default is '['.
    
    Returns
    -------
    dict
        A dictionary with custom formatted keys and values.

    Examples
    --------
    >>> from gofast.api.util import df_to_custom_dict
    >>> df = pd.DataFrame({'col0': [10, 20], 'col1': [30, 40]}, 
                          index=['a', 'b'])
    >>> dataframe_to_custom_dict(df)
    {'a': 'col0 <10> col1 <30>', 'b': 'col0 <20> col1 <40>'}
    """
    frame_open, frame_close = get_frame_chars(frame_char)[1:]
    key_format = (f"{key_prefix}{{}}" if df.index.is_unique 
                  and df.index.inferred_type == 'integer' else "{}")
    
    return {key_format.format(f"{frame_open}{index}{frame_close}" 
                              if key_format.startswith(key_prefix) else index):
            ' '.join(f"{col} <{format_value(val)}>" for col, val in row.items())
            for index, row in df.iterrows()}


def format_cell(x, max_text_length, max_width =None ):
    """
    Truncates a string to the maximum specified length and appends '...' 
    if needed, and right-aligns it.

    Parameters:
    x (str): The string to format.
    max_width (int): The width to which the string should be aligned.
    max_text_length (int): The maximum allowed length of the string before truncation.

    Returns:
    str: The formatted and aligned string.
    """
    x = str(x)
    if len(x) > max_text_length:
        x = x[:max_text_length - 3] + '...'
    return x.rjust(max_width) if max_width else x 

def calculate_widths(df, max_text_length):
    """
    Calculates the maximum widths for each column based on the content.

    Parameters:
    df (pandas.DataFrame): The DataFrame to calculate widths for.
    max_text_length (int): The maximum allowed length for any cell content.

    Returns:
    tuple: A dictionary with maximum column widths and the maximum width of the index.
    """
    formatted_cells = df.applymap(lambda x: str(x)[:max_text_length] + '...' if len(
        str(x)) > max_text_length else str(x))
    max_col_widths = {col: max(len(col), max(len(x) for x in formatted_cells[col]))
                      for col in df.columns}
    max_index_width = max(len(str(index)) for index in df.index)
    max_col_widths = {col: min(width, max_text_length) for col, width in max_col_widths.items()}
    return max_col_widths, max_index_width

def format_df(df, max_text_length=50, title=None):
    """
    Formats a pandas DataFrame for pretty-printing in a console or
    text-based interface. This function provides a visually-appealing
    tabular representation with aligned columns and a fixed maximum
    column width.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to be formatted.
        
    max_text_length : int, optional
        The maximum length of text within each cell of the DataFrame.
        Default is 50 characters. Text exceeding this length will be
        truncated.
        
    title : str, optional
        An optional title for the formatted correlation matrix. If provided, it
        is centered above the matrix. Default is None.
        
    Returns
    -------
    str
        A formatted string representation of the DataFrame with columns
        and rows aligned, headers centered, and cells truncated according
        to `max_text_length`.

    Notes
    -----
    This function depends on helper functions `calculate_widths` to 
    determine the maximum widths for DataFrame columns based on the 
    `max_text_length`, and `format_cell` to appropriately format and
    truncate cell content. It handles both the DataFrame's index and
    columns to ensure a clean and clear display.

    Examples
    --------
    Consider a DataFrame `df` created as follows:
    
    >>> import pandas as pd 
    >>> from gofast.api.util import format_df 
    >>> data = {
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Occupation': ['Engineer', 'Doctor', 'Artist'],
        'Age': [25, 30, 35]
    }
    >>> df = pd.DataFrame(data)

    Formatting `df` with `format_df`:

    >>> print(format_df(df, max_text_length=10))
    =============================
           Name   Occupation  Age
      ---------------------------
    0 |    Alice    Engineer   25
    1 |      Bob      Doctor   30
    2 |  Charlie      Artist   35
    =============================
    
    Here, the table respects a `max_text_length` of 10, ensuring that all
    cell contents do not exceed this length, and the output is well-aligned
    for easy reading.
    """
    title = str(title or '').title()
    
    # Use helper functions to format cells and calculate widths
    max_col_widths, max_index_width = calculate_widths(df, max_text_length)

    # Formatting the header
    header = " " * (max_index_width + 4) + "  ".join(col.center(
        max_col_widths[col]) for col in df.columns)
    separator = " " * (max_index_width + 1) + "-" * (len(header) - (max_index_width + 1))

    # Formatting the rows
    data_rows = [
        f"{str(index).ljust(max_index_width)} |  " + 
        "  ".join(format_cell(row[col], max_text_length, max_col_widths[col]).ljust(
            max_col_widths[col]) for col in df.columns)
        for index, row in df.iterrows()
    ]

    # Creating top and bottom borders
    full_width = len(header)
    top_border = "=" * full_width
    bottom_border = "=" * full_width

    # Full formatted table
    formatted_string = f"{top_border}\n{header}\n{separator}\n" + "\n".join(
        data_rows) + "\n" + bottom_border
    
    if title:
        max_width = find_maximum_table_width(formatted_string)
        title = title.center(max_width) + "\n"
    return title + formatted_string


def validate_data(data, columns=None, error_mode='raise'):
    """
    Validates and converts input data into a pandas DataFrame, handling
    various data types such as DataFrame, ndarray, dictionary, and Series.

    Parameters
    ----------
    data : DataFrame, ndarray, dict, Series
        The data to be validated and converted into a DataFrame.
    columns : list, str, optional
        Column names for the DataFrame. If provided, they should match the
        data dimensions. If not provided, default names will be generated.
    error_mode : str, {'raise', 'warn'}, default 'raise'
        Error handling behavior: 'raise' to raise errors, 'warn' to issue
        warnings and use default settings.

    Returns
    -------
    DataFrame
        A pandas DataFrame constructed from the input data.

    Raises
    ------
    ValueError
        If the number of provided columns does not match the data dimensions
        and error_mode is 'raise'.
    TypeError
        If the input data type is not supported.

    Notes
    -----
    This function is designed to simplify the process of converting various
    data types into a well-formed pandas DataFrame, especially when dealing
    with raw data from different sources. The function is robust against
    common data input errors and provides flexible error handling through
    the `error_mode` parameter.

    Examples
    --------
    >>> data = np.array([[1, 2], [3, 4]])
    >>> validate_data(data)
       feature_0  feature_1
    0          1          2
    1          3          4

    >>> data = {'col1': [1, 2], 'col2': [3, 4]}
    >>> validate_data(data, columns=['column1', 'column2'])
       column1  column2
    0        1        3
    1        2        4

    >>> data = pd.Series([1, 2, 3])
    >>> validate_data(data, error_mode='warn')
       feature_0
    0          1
    1          2
    2          3
    """
    def validate_columns(data_columns, expected_columns):
        if expected_columns is None:
            return [f'feature_{i}' for i in range(data_columns)]
        
        if isinstance(expected_columns, (str, float, int)):
            expected_columns = [expected_columns]
        
        if len(expected_columns) != data_columns:
            message = "Number of provided column names does not match data dimensions."
            if error_mode == 'raise':
                raise ValueError(message)
            elif error_mode == 'warn':
                warnings.warn(f"{message} Default columns will be used.", UserWarning)
                return [f'feature_{i}' for i in range(data_columns)]
        return expected_columns

    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, np.ndarray):
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        if data.ndim == 2:
            columns = validate_columns(data.shape[1], columns)
            df = pd.DataFrame(data, columns=columns)
        else:
            raise ValueError("Array with more than two dimensions is not supported.")
    elif isinstance(data, dict):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.Series):
        df = data.to_frame()
    else:
        raise TypeError(
            "Unsupported data type. Data must be a DataFrame, array, dict, or Series.")

    return df

def format_correlations(
    data, 
    min_corr=0.5, 
    high_corr=0.8, 
    use_symbols=False, 
    no_corr_placeholder='...', 
    hide_diag=True, 
    title=None, 
    error_mode='warn', 
    precomputed=False,
    legend_markers=None
    ):
    """
    Computes and formats the correlation matrix for a DataFrame's numeric columns, 
    allowing for visual customization and conditional display based on specified
    correlation thresholds.

    Parameters
    ----------
    data : DataFrame
        The input data from which to compute correlations. Must contain at least
        one numeric column.
    min_corr : float, optional
        The minimum correlation coefficient to display explicitly. Correlation
        values below this threshold will be replaced by `no_corr_placeholder`.
        Default is 0.5.
    high_corr : float, optional
        The threshold above which correlations are considered high, which affects
        their representation when `use_symbols` is True. Default is 0.8.
    use_symbols : bool, optional
        If True, uses symbolic representation ('++', '--', '+-') for correlation
        values instead of numeric. Default is False.
    no_corr_placeholder : str, optional
        Text to display for correlation values below `min_corr`. Default is '...'.
    hide_diag : bool, optional
        If True, the diagonal elements of the correlation matrix (always 1) are
        not displayed. Default is True.
    title : str, optional
        An optional title for the formatted correlation matrix. If provided, it
        is centered above the matrix. Default is None.
    error_mode : str, optional
        Determines how to handle errors related to data validation: 'warn' (default),
        'raise', or 'ignore'. This affects behavior when the DataFrame has insufficient
        data or non-numeric columns.
    precomputed: bool, optional 
       Consider data as already correlated data. No need to recomputed the
       the correlation. Default is 'False'
    legend_markers: str, optional 
       A dictionary mapping correlation symbols to their descriptions. If provided,
       it overrides the default markers. Default is None.
       
    Returns
    -------
    str
        A formatted string representation of the correlation matrix that includes
        any specified title, the matrix itself, and potentially a legend if
        `use_symbols` is enabled.

    Notes
    -----
    The function relies on pandas for data manipulation and correlation computation. 
    It customizes the display of the correlation matrix based on user preferences 
    for minimum correlation, high correlation, and whether to use symbolic 
    representations.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.api.util import format_correlations
    >>> data = pd.DataFrame({
    ...     'A': np.random.randn(100),
    ...     'B': np.random.randn(100),
    ...     'C': np.random.randn(100) * 10
    ... })
    >>> print(format_correlations(data, min_corr=0.3, high_corr=0.7,
    ...                            use_symbols=True, title="Correlation Matrix"))
    Correlation Matrix
    ==================
          A    B    C 
      ----------------
    A |       ...  ...
    B |  ...       ...
    C |  ...  ...     
    ==================

    ..................
    Legend : ...:
             Non-correlate
             d++: Strong
             positive,
             --: Strong
             negative,
             +-: Moderate
    ..................
    """

    title = str(title or '').title()
    df = validate_data(data)
    if len(df.columns) == 1:
        if error_mode == 'warn':
            warnings.warn("Cannot compute correlations for a single column.")
        elif error_mode == 'raise':
            raise ValueError("Cannot compute correlations for a single column.")
        return '' if error_mode == 'ignore' else 'No correlations to display.'
    
    if precomputed: 
        corr_matrix= data.copy() 
    else:     
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            if error_mode == 'warn':
                warnings.warn("No numeric data found in the DataFrame.")
            elif error_mode == 'raise':
                raise ValueError("No numeric data found in the DataFrame.")
                # Return an empty string if no numeric data
            return '' if error_mode == 'ignore' else 'No numeric data available.'  
    
        corr_matrix = numeric_df.corr()

    if hide_diag:
        np.fill_diagonal(corr_matrix.values, np.nan)  # Set diagonal to NaN

    def format_value(value):
        if pd.isna(value):  # Handle NaN for diagonals
            return '' if hide_diag else ( 'o' if use_symbols else pd.isna(value))
        if abs(value) < min_corr:
            return str(no_corr_placeholder).ljust(4) if use_symbols else f"{value:.4f}"
        if use_symbols:
            if value >= high_corr:
                return '++'.ljust(4)
            elif value <= -high_corr:
                return '--'.ljust(4)
            else:
                return '+-'.ljust(4)
        else:
            return f"{value:.4f}"

    formatted_corr = corr_matrix.applymap(format_value)
    formatted_df = format_df(formatted_corr)
    
    max_width = find_maximum_table_width(formatted_df)
    legend = ""
    if use_symbols:
        legend = generate_legend(
            legend_markers, no_corr_placeholder, hide_diag,  max_width)
    if title:
        title = title.center(max_width) + "\n"

    return title + formatted_df + legend

def generate_legend(
    custom_markers=None, 
    no_corr_placeholder='...', 
    hide_diag=True,
    max_width=50, 
    add_frame_lines=True, 
    border_line='.'
    ):
    """
    Generates a legend for a table (dataframe) matrix visualization, formatted 
    according to specified parameters.

    This function supports both numeric and symbolic representations of table
    values. Symbolic representations, which are used primarily for visual clarity,
    include the following symbols:

    - ``'++'``: Represents a strong positive relationship.
    - ``'--'``: Represents a strong negative relationship.
    - ``'+-'``: Represents a moderate relationship.
    - ``'o'``: Used exclusively for diagonal elements, typically representing
      a perfect relationship in correlation matrices (value of 1.0).
         
    Parameters
    ----------
    custom_markers : dict, optional
        A dictionary mapping table symbols to their descriptions. If provided,
        it overrides the default markers. Default is None.
    no_corr_placeholder : str, optional
        Placeholder text for table values that do not meet the minimum threshold.
        Default is '...'.
    hide_diag : bool, optional
        If True, omits the diagonal entries from the legend. These are typically
        frame of a variable with itself (1.0). Default is True.
    max_width : int, optional
        The maximum width of the formatted legend text, influences the centering
        of the title. Default is 50.
    add_frame_lines : bool, optional
        If True, adds a frame around the legend using the specified `border_line`.
        Default is True.
    border_line : str, optional
        The character used to create the border of the frame if `add_frame_lines`
        is True. Default is '.'.

    Returns
    -------
    str
        The formatted legend text, potentially framed, centered according to the
        specified width, and including custom or default descriptions of correlation
        values.

    Examples
    --------
    >>> from gofast.api.util import generate_legend
    >>> custom_markers = {"++": "High Positive", "--": "High Negative"}
    >>> print(generate_legend(custom_markers=custom_markers, max_width=60))
    ............................................................
    Legend : ...: Non-correlated, ++: High Positive, --: High
             Negative, +-: Moderate
    ............................................................

    >>> print(generate_legend(hide_diag=False, max_width=70))
    ......................................................................
    Legend : ...: Non-correlated, ++: Strong positive, --: Strong negative,
             +-: Moderate, o: Diagonal
    ......................................................................
    >>> custom_markers = {"++": "Highly positive", "--": "Highly negative"}
    >>> legend = generate_legend(custom_markers=custom_markers,
    ...                          no_corr_placeholder='N/A', hide_diag=False,
    ...                          border_line ='=')

    >>> print(legend) 

    ==================================================
    Legend : N/A: Non-correlated, ++: Highly positive,
             --: Highly negative, +-: Moderate, o:
             Diagonal
    ==================================================
    """
    # Default markers and their descriptions
    default_markers = {
        no_corr_placeholder: "Non-correlated",
        "++": "Strong positive",
        "--": "Strong negative",
        "+-": "Moderate",
        "o": "Diagonal"  # only used if hide_diag is False
    }
    if ( custom_markers is not None 
        and not isinstance(custom_markers, dict)
    ):
        raise TypeError("The 'custom_markers' parameter must be a dictionary."
                        " Received type: {0}. Please provide a dictionary"
                        " where keys are the legend symbols and values"
                        " are their descriptions.".format(
                        type(custom_markers).__name__))

    # Update default markers with any custom markers provided
    markers = {**default_markers, **(custom_markers or {})}

    # Create legend entries
    legend_entries = [f"{key}: {value}" for key, value in markers.items() if not (
        key == 'o' and hide_diag)]

    # Join entries with commas and format the legend text
    legend_text = ", ".join(legend_entries)
    legend = "\n\n" + format_text(
        legend_text, 
        key='Legend', 
        key_length=len('Legend'), 
        max_char_text=max_width + len('Legend'), 
        add_frame_lines=add_frame_lines,
        border_line=border_line
        )
    return legend

def to_snake_case(name):
    """
    Converts a string to snake_case using regex.

    Parameters
    ----------
    name : str
        The string to convert to snake_case.

    Returns
    -------
    str
        The snake_case version of the input string.
    """
    name = str(name)
    name = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()  # CamelCase to snake_case
    name = re.sub(r'\W+', '_', name)  # Replace non-word characters with '_'
    name = re.sub(r'_+', '_', name)  # Replace multiple '_' with single '_'
    return name.strip('_')

def generate_column_name_mapping(columns):
    """
    Generates a mapping from snake_case column names to their original names.

    Parameters
    ----------
    columns : List[str]
        The list of column names to convert and map.

    Returns
    -------
    dict
        A dictionary mapping snake_case column names to their original names.
    """
    return {to_snake_case(col): col for col in columns}


if __name__=='__main__': 
    # Example usage:
    data = {
        'col0': [1, 2, 3, 4],
        'col1': [4, 3, 2, 1],
        'col2': [10, 20, 30, 40],
        'col3': [40, 30, 20, 10],
        'col4': [5, 6, 7, 8]
    }
    df = pd.DataFrame(data)

    # Calling the function
    result = format_correlations(df, 0.8, 0.9, False, hide_diag= True)
    print(result)

    # Example usage
    data = {
        'col0': [1, 2, 3, 4],
        'col1': [4, 3, 2, 1],
        'col2': [10, 20, 30, 40],
        'col3': [40, 30, 20, 10],
        'col4': [5, 6, 7, 8]
    }

    
