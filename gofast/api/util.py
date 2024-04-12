# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

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


def format_df(df, max_text_length=50):
    # Use helper functions to format cells and calculate widths
    max_col_widths, max_index_width = calculate_widths(df, max_text_length)

    # Formatting the header
    header = " " * (max_index_width + 4) + "  ".join(col.center(
        max_col_widths[col]) for col in df.columns)
    separator = " " * (max_index_width + 1) + "-" * (len(header) - (max_index_width + 1))

    # Formatting the rows
    data_rows = [
        f"{str(index).ljust(max_index_width)} |  " + 
        "  ".join(format_cell(row[col],  max_text_length, max_col_widths[col]).ljust(
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
    return formatted_string

def compute_and_format_correlations(
    data, min_correlation=0.5, high_correlation=0.8, 
    soft_format=False, placeholder='...', remove_diagonal=False, 
    title=None, error='warn'# can  be raise , warn or ignore 
    ):
    """
    Compute and format the correlations for a DataFrame's numeric columns.
    
    Args:
    df (DataFrame): DataFrame to compute correlations for.
    min_correlation (float): Minimum threshold for correlations to be noted.
    high_correlation (float): Threshold for considering a correlation high.
    soft_format (bool): Whether to use symbolic formatting for high/low correlations.
    placeholder (str): Placeholder text for correlations below the minimum threshold.
    
    Returns:
    str: A formatted string representation of the correlation matrix.
    """
    title = title or ''
    title = str(title).title () 
    
    def validate_data (data, columns =None ):
        if isinstance (data, pd.DataFrame): 
            df = data.copy() 
        # write helper function to validate data  and use it in this function 
        elif isinstance(data, np.array): 
            # if is one1d array , convert to dataframe with single features 
            # commonly does not make sense for calculating correlation 
            # if array is two dimensional 
            # create a dataframe 
            if columns is not None: 
                columns = [columns] if isinstance (columns, str, float, int) else columns
            columns = ['feature_{i}' for i in range(data.shape[1])] if len(
                columns)!= len(data.shape[1]) else columns 
            df = pd.DataFrame(data, columns =columns)
        elif isinstance (data, dict): 
            df = pd.DataFrame (data ) 
        elif isinstance (data, pd.Series) :
            
            df = data.to_frame() 
            
        else : # not possible to convert 
            raise # Informative error 
            
        return df 

    df = validate_data (data)
        
    # check whether the dataframe has single columns 
    # if True, raise informative warnings ( if error is 'warn') to tell the user that 
    # if does not make since to compute the correlation with single columns since 
    # it equal to 1 
    
    # Select numeric columns and compute correlation matrix
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty: 
        return ''  # no numeric data is detected then return Empy . 
    corr_matrix = numeric_df.corr()
    
    
    # Function to format each correlation value
    def format_correlation(value):
        if abs(value) < min_correlation:
            return placeholder
        # FIX It. this second condition does not work well because 
        # some other value in the table who are not in 
        # diagonal can also take the value of 1 for high correlation . 
        # so how to efficiently detect the diagonal value then 
        # apply the condition to change their value to maker with remove_diagonal parameter. 
        
        elif abs(value) ==1.0: 
            # probability the diagonal ( which is not really true )
            return '' if remove_diagonal else 'o' # remove diagonal 
        elif soft_format:
            if value >= high_correlation:
                return '++'
            elif value <= -high_correlation:
                return '--'
            else:
                return '+-'
        else:
            return f"{value:.4f}"

    # Apply formatting to the correlation matrix
    formatted_corr = corr_matrix.applymap(format_correlation)
    
    # Format the DataFrame using the existing format_dataframe function
    formatted_df = format_df(formatted_corr)

    # get the maximum table width 
    max_table_width = find_maximum_table_width(formatted_df)
    # Append legend if soft formatting is used
    legend = ""
    if soft_format:
        placeholder = f"{placeholder}: Non-correlated, " if placeholder else ''
        diagonal_marker = '' if remove_diagonal else ', o: Diagonal matrix' 
        legend = f"{placeholder}++: Strong positive, --: Strong negative, +-: Moderate{diagonal_marker}"
        legend= '\n\n' + format_text (
            legend, 
            key ="Legend", 
            key_length= len("Legend"),
            add_frame_lines= True, 
            max_char_text= max_table_width + len("Legend") , 
            border_line='.'
            ) 
    if title: 
        title = title.center (max_table_width)  +"\n"
    return title + formatted_df + legend



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
    result = compute_and_format_correlations(df, 0.8, 0.9, True)
    print(result)

    # Example usage
    data = {
        'col0': [1, 2, 3, 4],
        'col1': [4, 3, 2, 1],
        'col2': [10, 20, 30, 40],
        'col3': [40, 30, 20, 10],
        'col4': [5, 6, 7, 8]
    }

    
