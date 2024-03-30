# -*- coding: utf-8 -*-
#   License: BSD-3-Clause
#   Author: LKouadio <etanoyau@gmail.com>

# import textwrap
import re
import warnings 
import numpy as np
import pandas as pd
from .structures import Bunch

class MultiFrameFormatter:
    """
    A factory class designed to manage and format multiple pandas DataFrames
    for display, either with unified or individual formatting depending on
    column similarities. It supports handling DataFrames with both matching
    and differing column sets, applying appropriate formatting strategies
    for each scenario. Titles and keywords can be associated with each DataFrame
    to enhance the readability and accessibility of the formatted output.

    Parameters
    ----------
    titles : list of str, optional
        Titles for each DataFrame, used to label tables when printed. Titles
        are centered above their respective tables. If not provided, defaults
        to an empty list.
    keywords : list of str, optional
        Keywords associated with each DataFrame, enabling attribute-based access
        to DataFrames and their columns within the factory. If not provided,
        defaults to an empty list.

    Attributes
    ----------
    dfs : list of pandas.DataFrame
        The list of DataFrames added to the factory for formatting.
    titles : list of str
        The titles associated with each DataFrame.
    keywords : list of str
        The keywords associated with each DataFrame, used for creating
        intuitive attributes for data retrieval.

    Methods
    -------
    add_dfs(*dfs)
        Adds one or more DataFrames to the factory, enabling formatting 
        and attribute setting based on provided titles and keywords.
    dataframe_with_same_columns()
        Constructs and returns a string representation of a unified table 
        for DataFrames with identical column names.
    dataframe_with_different_columns()
        Constructs and returns string representations of individual tables 
        for DataFrames with differing column names.
    __str__()
        Generates a string representation of the factory's formatted tables, 
        adjusting for column similarities among DataFrames.
    _process_keyword_attribute()
        Processes provided keywords for intuitive data retrieval, defaul3ting 
        to titles when keywords are absent.
    _populate_df_column_attributes()
        Populates attributes corresponding to DataFrame columns, facilitating 
        direct access through snake_case attribute names.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.api.formatter import MultiFrameFormatter
    >>> df1 = pd.DataFrame({'A': [1, 2], 'B': ['Text example', 5]})
    >>> df2 = pd.DataFrame({'C': [3, 4], 'D': ['Another text', 6]})
    >>> factory = MultiFrameFormatter(titles=['DataFrame 1', 'DataFrame 2'])
    >>> factory.add_dfs(df1, df2)
    >>> print(factory)
    The example demonstrates creating a MultiFrameFormatter instance, adding 
    two DataFrames with distinct columns, and showcasing the textual 
    representation facilitated by the factory.

    Notes
    -----
    The MultiFrameFormatter dynamically adapts to the characteristics of the added
    DataFrames, optimizing the visual presentation whether columns across the
    DataFrames match or not. It employs a strategy to ensure readability,
    aligning titles and adjusting column widths as necessary. Additional
    utility methods provide flexibility in accessing and manipulating
    DataFrame contents directly through attributes set based on provided
    keywords, using snake_case naming conventions for ease of use.
    """

    def __init__(self, titles=None, keywords=None):
        self.titles = titles if titles is not None else []
        self.keywords = keywords if keywords is not None else []
        self.dfs = []

    def _check_dfs(self):
        """
        Checks if all items in `self.dfs` are pandas DataFrames. Converts pandas Series
        to DataFrames under specific conditions. If a Series' index length is less than
        twice the maximum column count across all DataFrames, it is converted into a
        DataFrame with its index becoming a column. Series with a larger index are not
        converted and excluded from MultiFrameFormatter with a warning.
        """
        max_columns = max((len(df.columns) for df in self.dfs if isinstance(
            df, pd.DataFrame)), default=0)
        checked_dfs = []

        for df in self.dfs:
            if isinstance(df, pd.Series):
                if len(df.index) < 2 * max_columns:
                    new_df =series_to_dataframe (df)
                    checked_dfs.append(new_df)
                else:
                    warnings.warn("A Series with a large index was not"
                                  " converted and is excluded.", UserWarning)
            elif isinstance(df, pd.DataFrame):
                checked_dfs.append(df)
            else:
                warnings.warn("An item that is not a DataFrame or Series"
                              " was found and is excluded.", UserWarning)

        if not checked_dfs:
            # Handling the scenario where checked_dfs is empty after processing
            raise ValueError("No valid pandas DataFrame or convertible Series"
                             " were provided to FrameFactory.")

        self.dfs = checked_dfs

    def add_dfs(self, *dfs):
        """
        Adds dataframes to the factory and processes them for
        attribute setting based on keywords and titles.

        Parameters
        ----------
        *dfs : unpacked list of pandas.DataFrame
            Dataframes to be added to the factory.

        Returns
        -------
        self : MultiFrameFormatter instance
            The instance of FrameFactory to allow method chaining.
        """
        self.dfs.extend(dfs)
        self._check_dfs() 
        self._process_keyword_attribute()
        self._populate_df_column_attributes()
        
        return self

    def dataframe_with_same_columns(self):
        """
        Constructs a single table where all included dataframes
        share the same column names.

        Returns
        -------
        str
            A string representation of the constructed table.
        """
        return construct_tables_for_same_columns(self.dfs, self.titles)

    def dataframe_with_different_columns(self):
        """
        Constructs individual tables for each dataframe when
        dataframes have differing column names.

        Returns
        -------
        str
            A string representation of the constructed tables.
        """
        return construct_table_for_different_columns(self.dfs, self.titles)

    def __str__(self):
        """
        Provides a string representation of the factory's state,
        displaying the constructed tables based on the dataframes'
        column similarities.

        Returns
        -------
        str
            The string representation of the tables or an
            indication of emptiness if no dataframes are present.
        """
        if not self.dfs:
            return "<Empty Frame>"
        if len(self.dfs) == 1:
            return DataFrameFormatter(self.titles[0], self.keywords[0]).add_df(
                self.dfs[0]).__str__()

        if have_same_columns(self.dfs):
            return self.dataframe_with_same_columns()

        return self.dataframe_with_different_columns()

    def _process_keyword_attribute(self):
        """
        Processes keywords to create intuitive attributes for
        data retrieval, handling defaults based on titles if
        necessary.
        """
        unique_keywords = list(set(self.keywords))
        self.keywords = unique_keywords + ['result_{}'.format(i) for i in range(
            len(self.dfs) - len(unique_keywords))]
        
        if len(self.titles) != len(self.dfs): 
            self.titles = self.titles + [''] * (len(self.dfs) - len(self.titles)) 
                                  
        for keyword, df, title in zip(self.keywords, self.dfs, self.titles):
            snake_case_keyword = to_snake_case(keyword if keyword else title)
            setattr(self, snake_case_keyword, df.copy())

    def _populate_df_column_attributes(self):
        """
        Populates the MultiFrameFormatter with attributes corresponding
        to each dataframe column, using snake_case naming.
        """
        for df, keyword in zip(self.dfs, self.keywords):
            column_name_mapping = generate_column_name_mapping(df.columns)
            for snake_case_name, original_name in column_name_mapping.items():
                attribute_name = f"{snake_case_name}_{to_snake_case(keyword)}" if hasattr(
                    self, snake_case_name) else snake_case_name
                setattr(self, attribute_name, df[original_name])

    def __repr__(self):
        """
        Represents the FrameFactory instance, indicating the
        presence of dataframes.

        Returns
        -------
        str
            A descriptive string about the FrameFactory instance.
        """
        return ( "<MultiFrame object with dataframes. Use print() to view.>" 
                if self.dfs else "<Empty MultiFrame>")

class DataFrameFormatter:
    """
    Formats pandas DataFrames for enhanced visual presentation in textual output.
    This class supports titles, dynamic adjustment of column widths based on content,
    and automatic handling of numerical data precision. It also facilitates the direct
    conversion of pandas Series to DataFrame for uniform handling.

    Parameters
    ----------
    title : str, optional
        The title to be displayed above the DataFrame when printed. If provided,
        it centers the title over the DataFrame. Defaults to None.
    keyword : str, optional
        A keyword associated with the DataFrame content, facilitating intuitive 
        attribute access for common data-related tasks. Defaults to None.

    Attributes
    ----------
    df : pandas.DataFrame or None
        Stores the DataFrame to be formatted. Initially None until a DataFrame or
        Series is added via `add_df`.

    Methods
    -------
    add_df(df) : DataFrameFormatter
        Adds a DataFrame or Series to the formatter, enabling further manipulation.
        If a Series is provided, converts it to a DataFrame for consistent handling.

    _process_keyword_attribute() : None
        Handles keyword processing, setting an intuitive attribute based on the
        DataFrame's context or title if applicable.

    _format_value(val, col_name, col_widths) : str
        Adjusts the presentation of DataFrame values for printing, applying 
        truncation and formatting rules based on data type and content length.

    _format_header() : tuple
        Constructs the table header, including the title and column names, with
        adjustments for exceptionally long titles or column content.

    _to_snake_case(name) : str
        Converts a given string to snake_case, facilitating standard attribute access.

    _generate_column_name_mapping() : None
        Creates a mapping from snake_case to original DataFrame column names.

    _to_original_name(snake_case_name) : str
        Retrieves the original column name from a snake_case version, aiding in
        attribute-style data access.

    _populate_df_column_attributes() : None
        Dynamically assigns DataFrame columns as attributes of the formatter 
        instance for direct access.

    __repr__() : str
        Provides a compact representation of the DataFrameFormatter instance, 
        indicating data presence.

    __str__() : str
        Generates a structured and formatted string representation of the DataFrame,
        ready for printing, complete with title and column alignments.

    __getattr__(attr_name) : Any
        Overrides attribute access, facilitating direct retrieval of DataFrame
        columns via snake_case attributes or informative error handling.

    Examples
    --------
    >>> import pandas as pd
    >>> from gofast.api.formatter import DataFrameFormatter
    >>> df = pd.DataFrame({
    ...     'column1': [1, 2, 3.12345, 4],
    ...     'column2': [5.6789, 6, 7, 8]
    ... }, index=['Index1', 'Index2', 'Index3', 'Index4'])
    >>> formatter = DataFrameFormatter("Overview of Data")
    >>> formatter.add_df(df)
    >>> print(formatter)
          Overview of Data        
    ==============================
            column1    column2    
    ------------------------------
    Index1   1.0000     5.6789     
    Index2   2.0000     6.0000     
    Index3   3.1235     7.0000     
    Index4   4.0000     8.0000     
    ==============================

    Notes
    -----
    This formatter automatically determines the optimal column widths based on
    content length to maintain alignment across rows. It defaults to four decimal
    places for floating-point numbers but can be customized via internal methods.
    The formatter also supports attribute-like access to DataFrame columns via
    snake_case conversion of column names.
    """

    def __init__(self, title=None, keyword=None):
        self.title = title
        self.keyword = keyword
        self.df = None
        self._column_name_mapping = {}

    def add_df(self, df):
        """
        Enhances the DataFrameFormatter by adding a DataFrame or Series,
        handling attribute population with snake_case conversion, and
        setting additional intuitive access based on predefined keywords
        related to module functionality.
    
        Parameters
        ----------
        df : pandas.DataFrame or pandas.Series
            The data to format. A Series will be converted to a DataFrame.
    
        Raises
        ------
        ValueError
            If `df` is neither a pandas DataFrame nor a Series.
    
        Returns
        -------
        self : DataFrameFormatter
            Enables method chaining.
        """
        if isinstance(df, pd.Series):
            df = df.to_frame()
        elif not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame or Series.")
        
        self.df = df
        self._generate_column_name_mapping()
        self._populate_df_column_attributes()
    
        self._process_keyword_attribute()
        return self

    def _process_keyword_attribute(self):
        """
        Processes the keyword to set an intuitive attribute for data retrieval.
        Also considers title for default keywords in absence of an 
        explicit keyword.
        """
        if self.keyword and not self.keyword.isdigit():
            snake_case_keyword = self._to_snake_case(str(self.keyword))
            setattr(self, snake_case_keyword, self.df.copy())
        elif self.title:
            default_keywords = ['result', 'score', 'report']  # Add more as needed.
            matched_keyword = next(
                (kw for kw in default_keywords if kw in self.title.lower()), None)
            if matched_keyword:
                setattr(self, matched_keyword, self.df.copy())
        
    def _format_value(self, val, col_name, col_widths):
        """
        Formats a given value for display within a specified column width, ensuring
        that long texts are truncated appropriately to maintain table alignment and 
        readability. For numeric values, formats to four decimal places or as integers.
    
        Parameters:
        ----------
        val : any
            The value to be formatted. Can be of any type.
        col_name : str
            The name of the column to which the value belongs.
        col_widths : dict
            Dictionary mapping column names to their calculated widths.
    
        Returns:
        -------
        str
            The value formatted as a string, truncated if necessary, and aligned 
            according to its data type.
    
        Notes:
        -----
        This method adjusts formatting based on value type and length. Numeric values
        are right-aligned, while strings are left-aligned unless truncated, in which
        case they're right-aligned to fit the '...' at the end. Iterables are formatted
        using a predefined function `format_iterable`, and their representation is 
        truncated if exceeding column width.
        """
        col_width = col_widths[col_name]
    
        if isinstance(val, (np.integer, np.floating)):
            formatted_val = f"{float(val):.4f}" if isinstance(
                val, np.floating) else f"{int(val)}"
        elif isinstance(val, (float, int)):
            formatted_val = f"{val:.4f}" if isinstance(val, float) else f"{val}"
   
        elif isinstance(val, str):
            formatted_val = ( 
                (val[:col_width - 3] + '...') 
                if len(val) > self._max_value_width  - 3 else val
             )
        else:
            formatted_val = format_iterable(val)
    
        if len(formatted_val) > self._max_value_width :
            formatted_val = formatted_val[:col_width - 3] + '...'
    
        if isinstance(val, (int, float, np.integer, np.floating)):
            return f"{formatted_val:>{col_width}}"
        else:
            return (
                f"{formatted_val:<{col_width}}" if "..."  in formatted_val
                else f"{formatted_val:>{col_width}}" # shorter string 
                ) 
        
    def _format_header(self):
        """
        Prepares the header section of the table by calculating column widths and 
        formatting column names. Adjusts for the presence of an index column and 
        accounts for exceptionally long texts by capping individual value contributions
        to column width calculations.
    
        Returns:
        -------
        tuple
            Contains formatted header as a string, the separator lines, and calculated
            column and index widths.
    
        Notes:
        -----
        The method enforces a maximum width for value representations in the column
        width calculation to prevent any single value from causing excessive column
        widths. This cap ensures that the table remains readable and well-formatted
        even in the presence of long texts. If a title is provided, it is centered 
        above the table. Separator lines differentiate the title, header, and data.
        """
        self._max_value_width = 50  
    
        # Initial column widths based on column names and values
        initial_col_widths = {
            col: max(len(str(col)),max(
                 len(format_value(val,self._max_value_width )) for val in self.df[col])
            ) + 2
            for col in self.df.columns
        }
        
        if self.df.index.dtype == 'int64':
            index_width = 0
        else:
            index_width = max(len(str(index)) for index in self.df.index) + 2
    
        initial_header_row = " " * index_width + "  ".join(
            [f"{col:>{initial_col_widths[col]}}" for col in self.df.columns]
            )
        initial_header_length = len(initial_header_row)
    
        # Adjust column widths if the title is longer than the initial header
        if self.title and len(self.title) > initial_header_length:
            extra_space = len(self.title) - initial_header_length
            extra_space_per_col = extra_space // len(self.df.columns)
    
            # Recalculate column widths
            col_widths = {col: width + extra_space_per_col 
                          for col, width in initial_col_widths.items()}
            adjusted_header_row = " " * index_width + "  ".join(
                [f"{col:>{col_widths[col]}}" for col in self.df.columns])
            line = "=" * len(self.title)
            subline = "-" * len(self.title)
            title_str = f"{self.title:^{len(line)}}".title()
        else:
            col_widths = initial_col_widths
            adjusted_header_row = initial_header_row
            line = "=" * len(adjusted_header_row)
            subline = "-" * len(adjusted_header_row)
            title_str = f"{self.title:^{len(line)}}".title() if self.title else ""
    
        header = ( 
            f"\n{title_str}\n{line}\n{adjusted_header_row}\n{subline}\n" 
            if self.title else f"{line}\n{adjusted_header_row}\n{subline}\n"
            )
        
        return header, line, index_width, col_widths
        
    def _to_snake_case(self, name):
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
        name =str(name)
        name = re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()  # CamelCase to snake_case
        name = re.sub(r'\W+', '_', name)  # Replace non-word characters with '_'
        name = re.sub(r'_+', '_', name)  # Replace multiple '_' with single '_'
        return name.strip('_')

    def _generate_column_name_mapping(self):
        """
        Generates a mapping from snake_case column names to their original names.
        """
        self._column_name_mapping = {self._to_snake_case(col): col for col in self.df.columns}


    def _to_original_name(self, snake_case_name):
        """
        Converts a snake_case column name back to its original column name
        using the previously generated mapping.

        Parameters
        ----------
        snake_case_name : str
            The snake_case name to convert back to the original column name.

        Returns
        -------
        str
            The original column name, or raises an AttributeError if not found.
        """
        original_name = self._column_name_mapping.get(snake_case_name)
        if original_name is None:
            raise AttributeError(f"'{snake_case_name}' column not found.")
        return original_name
    
    def _populate_df_column_attributes(self):
        """
        Populates attributes for each DataFrame column based on the snake_case
        column name mapping.
        """
        for snake_case_name, original_name in self._column_name_mapping.items():
            setattr(self, snake_case_name, self.df[original_name])
            
    def __repr__(self):
        """
        Returns a representation of the DataframeFormatter object.

        Returns
        -------
        str
            A string indicating whether the formatter contains data and
            suggesting to use `print()` to see contents.
        """
        if self.df is not None and not self.df.empty:
            return "<Frame object containing data. Use print() to see contents.>"
        else:
            return "<Empty Frame>"
        
    
    def __str__(self):
        """
        Returns a string representation of the formatted dataframe for printing.
    
        Returns
        -------
        str
            The formatted dataframe as a string.
        """
        # Handle a single dataframe
        return self._formatted_dataframe()

    def _formatted_dataframe (self, ): 
        """ Construct a formattage dataframe"""
        if self.df is None:
            return "No data added."
    
        header, line, index_width, col_widths = self._format_header()
    
        data_rows = ""
        for index, row in self.df.iterrows():
            if self.df.index.dtype != 'int64':
                index_str = f"{str(index):<{index_width}}"
            else:
                index_str = ""
            
            # Adjust to fetch the correct column width from col_widths using column names
            row_str = (
                "  ".join([self._format_value(row[col], col, col_widths) 
                           for col in self.df.columns])
            )
            data_rows += f"{index_str}{row_str}\n"
    
        return f"{header}{data_rows}{line}"
    
        
    def __getattr__(self, attr_name):
        """
        Allows attribute-style access to DataFrame columns. If an attribute is not
        found, raises a more informative error message.
    
        Parameters
        ----------
        attr_name : str
            The name of the attribute being accessed.
    
        Raises
        ------
        AttributeError
            If the attribute corresponding to a DataFrame column or an existing
            method/property is not found.
        """
        try:
            # Attempt to retrieve the original column name from the mapping
            original_col_name = self._to_original_name(attr_name)
            # Attempt to return the column from the DataFrame
            if original_col_name in self.df.columns:
                return self.df[original_col_name]
            else: 
                return  getattr (self, attr_name)
        except AttributeError:
            # This exception means the mapping retrieval failed because 
            # attr_name was not a column name
            pass
    
        # If the attribute is not a DataFrame column, check for it in 
        # the class and instance dictionaries
        if attr_name in self.__class__.__dict__ or attr_name in self.__dict__:
            return object.__getattribute__(self, attr_name)
    
        # If none of the above, raise an informative error message
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute or"
            f" column '{attr_name}'. Note: Column names are converted to snake_case.")

class MetricFormatter(Bunch):
    """
    A subclass of Bunch designed for formatting and displaying
    model performance metrics in a visually appealing manner. 
    MetricFormatter enhances readability and presentation of 
    metric results by providing attribute-style access and 
    customizable output formatting.

    Parameters
    ----------
    title : str, optional
        The title to display at the top of the formatted output. 
        If provided, it centers the title and frames the metrics 
        output with lines for improved readability. Defaults to 
        None, which omits the title from the output.
    **kwargs : dict, optional
        Arbitrary keyword arguments representing the performance 
        metrics and their values. Each keyword argument is treated 
        as a metric name with its corresponding value.

    Attributes
    ----------
    Inherits all attributes from the Bunch class and optionally 
    includes a `title` attribute if provided during initialization.

    Examples
    --------
    >>> from gofast.api.formatter import MetricFormatter
    >>> metrics = MetricFormatter(
            title='Model Performance',
            accuracy=0.95,
            precision=0.93,
            recall=0.92
        )
    >>> print(metrics)
    ====================================================
                     Model Performance
    ====================================================
    accuracy    : 0.95
    precision   : 0.93
    recall      : 0.92
    ====================================================

    Without a title:
    >>> metrics = MetricFormatter(accuracy=0.95, precision=0.93, recall=0.92)
    >>> print(metrics)
    ==========================================
    accuracy    : 0.95
    precision   : 0.93
    recall      : 0.92
    ==========================================

    Notes
    -----
    MetricFormatter automatically adjusts the length of the framing 
    lines ('=') to match the length of the longest metric string 
    representation. This ensures a cohesive and balanced appearance 
    regardless of the metric names or values. When providing a title, 
    it's centered within the top frame for a professional presentation.
    """
    def __init__(self, title="Metric Results", **kwargs):
        super().__init__(**kwargs)
        self.title = title
    
    def __str__(self):
        if not self.__dict__:
            return "<empty MetricFormatter>"
        
        keys = sorted(self.__dict__.keys())
        if 'title' in keys:
            keys.remove('title')  # Exclude title from the keys to be printed
        max_key_length = max(len(key) for key in keys)
        
        formatted_attrs = [
            f"{key:{max_key_length}} : {self._format_iterable(self.__dict__[key])}" 
            for key in keys if key != 'title'  # Ensure title is not repeated
        ]
        max_line_length = max(len(line) for line in formatted_attrs)
        content_str = "\n".join(formatted_attrs)
        
        if self.title:
            # Center the title and adjust '=' line length based on the longest line
            title_length = max(max_line_length, len(self.title))
            title_str = f"{self.title:^{title_length}}"
            header_footer_line = "=" * title_length
            formatted_output = ( 
                f"{header_footer_line}\n{title_str}\n{header_footer_line}\n"
                f"{content_str}\n{header_footer_line}"
                )
        else:
            # No title provided, use the longest line length for '=' line and 
            # include header/footer lines
            header_footer_line = "=" * max_line_length
            formatted_output = f"{header_footer_line}\n{content_str}\n{header_footer_line}"
        
        return formatted_output

class BoxFormatter:
    """
    A utility class for formatting text and dictionary content within a
    bordered box. It supports adding titles, formatting arbitrary text, and
    structuring dictionaries into a neat table format within a box.

    Attributes:
    -----------
    title : str
        The title to be displayed at the top of the box. Can be left empty.
    content : str
        The formatted content to be displayed within the box. Populated by
        calling `add_text` or `add_dict`.
    has_content : bool
        A flag indicating whether the box currently has content.
    
    Methods:
    --------
    add_text(text: str, box_width: int = 65):
        Formats and adds the provided text to the box content.
    add_dict(dict_table: dict, descr_width: int = 45):
        Formats and adds the provided dictionary to the box content as a table.
    
    Usage:
    ------
    >>> from gofast.api.formatter import BoxFormatter
    >>> formatter = BoxFormatter("Example Title")
    >>> formatter.add_text("This is an example of formatted text.", 60)
    >>> print(formatter)
    
    >>> dict_content = {"Key1": "This is a description.", "Key2": "Another description."}
    >>> formatter.add_dict(dict_content, 50)
    >>> print(formatter)
    """

    def __init__(self, title=''):
        self.title = title
        self.content = ''
        self.has_content = False

    def __str__(self):
        """
        Returns the formatted content with borders, title (if any), and body
        (either text or a dictionary formatted as a table).
        """
        if not self.has_content:
            return "No content added. Use add_text() or add_dict() to add content."
        return self.content

    def __repr__(self):
        """
        Provides a representation hinting at the usage of `print()` to view the
        formatted content if content is present.
        """
        return ("<BoxFormatter: Use print() to view content>" 
                if self.has_content else "<BoxFormatter: Empty>")

    def add_text(self, text: str, box_width=65):
        """
        Formats the provided text and adds it to the box content. The text is
        wrapped to fit within the specified width, and if a title is present,
        it is centered at the top of the box.

        Parameters:
        -----------
        text : str
            The text to be formatted and added to the box.
        box_width : int, optional
            The width of the box within which the text is to be wrapped.
        """
        self.has_content = True
        self.content = self.format_box(text, box_width, is_dict=False)

    def add_dict(self, dict_table, descr_width=45):
        """
        Formats the provided dictionary as a table and adds it to the box content.
        The table includes column headers for keys and values, and rows for each
        key-value pair in the dictionary. The table is fit within the specified
        description width.

        Parameters:
        -----------
        dict_table : dict
            The dictionary to be formatted as a table and added to the box.
        descr_width : int, optional
            The width constraint for the description column of the table.
        """
        self.has_content = True
        self.format_dict(dict_table, descr_width)

        
    def format_box(self, text, width, is_dict):
        """
        Formats text or a dictionary to be displayed within a bordered box, 
        potentially including a title. This method dynamically constructs the 
        box based on the input type (text or dictionary), width specifications, 
        and whether a title is provided. The resulting string is suitable for 
        printing directly to the console or incorporating into log messages 
        for enhanced readability.
    
        Parameters:
        -----------
        text : str or dict
            The content to be formatted. This can be a simple text string or a
            dictionary with keys and values to be displayed in a tabular format
            within the box.
        width : int
            The total width of the box, including the borders. This width influences
            how text is wrapped and how dictionary entries are displayed.
        is_dict : bool
            A flag indicating whether the `text` parameter should be treated as a
            dictionary. If True, `text` is formatted using `format_dict` method;
            otherwise, `wrap_text` method is used for plain text.
    
        Returns:
        --------
        str
            A string representing the formatted box, ready to be printed or logged.
            The box includes a top and bottom border, optionally a title, and the
            body content which is either wrapped text or a formatted dictionary.
    
        Example Usage:
        --------------
        >>> formatter = FormatSpecial("My Title")
        >>> formatter.format_box("Some long text that needs to be wrapped.", 60, is_dict=False)
        >>> print(formatter)
        # This will print the text within a box with 'My Title' centered on top.
    
        >>> dict_content = {"Key1": "Value1", "Key2": "Value2"}
        >>> formatter.format_box(dict_content, 60, is_dict=True)
        >>> print(formatter)
        # This will print the dictionary content formatted as a table within a box.
        """

        if self.title:
            title_str = f"{self.title.center(width - 4)}"
            top_border = f"|{'=' * (width - 2)}|"
            title_line = f"| {title_str} |"
        else:
            top_border = f"|{'=' * (width - 2)}|"
            title_line = ""

        if is_dict:
            body_content = self.format_dict(text, width - 4)
        else:
            wrapped_text = self.wrap_text(text, width - 4)
            body_content = '\n'.join([f"| {line.ljust(width - 4)} |" for line in wrapped_text])

        bottom_border = f"|{'-' * (width - 2)}|"
        return '\n'.join([top_border, title_line, bottom_border, body_content, top_border])
    
    def wrap_text(self, text, width):
        """
        Wraps a given text string to fit within a specified width, ensuring that 
        words are not split across lines. This method is primarily used to format 
        text content for inclusion in a larger formatted box, but can also be used 
        independently to prepare text content for display in constrained-width 
        environments.
    
        Parameters:
        -----------
        text : str
            The text string to be wrapped. The text is split into words based on 
            spaces, and lines are formed by concatenating words until the specified 
            width is reached.
        width : int
            The maximum width of each line of text, in characters. This width 
            constraint determines where line breaks are inserted.
    
        Returns:
        --------
        list of str
            A list of strings, each representing a line of text that fits within 
            the specified width. This list can be joined with newline characters 
            for display.
    
        Example Usage:
        --------------
        >>> formatter = FormatSpecial()
        >>> lines = formatter.wrap_text("This is a sample sentence that will be wrapped.", 20)
        >>> for line in lines:
        >>>     print(line)
        # This will print each line of the wrapped text, adhering to the specified width.
        """
        words = text.split()
        wrapped_lines = []
        current_line = ''

        for word in words:
            if len(current_line + ' ' + word) <= width:
                current_line += ' ' + word if current_line else word
            else:
                wrapped_lines.append(current_line)
                current_line = word
        wrapped_lines.append(current_line)

        return wrapped_lines
    
    def format_dict(self, dict_table, descr_width=45):
        """
        Formats and displays a dictionary as a neatly organized table within a
        formatted box. Each key-value pair in the dictionary is treated as a row
        in the table, with the key representing a feature name and the value
        its description. This method is designed to enhance the readability
        and presentation of detailed information, particularly useful for
        displaying feature descriptions or similar data.
    
        Parameters:
        -----------
        dict_table : dict
            A dictionary where the keys are feature names (or any descriptive
            label) and the values are the corresponding descriptions or details
            to be presented in the table.
        descr_width : int, default=45
            The desired width of the description column in the table. This
            determines how text in the description column is wrapped and
            affects the overall width of the table.
    
        The method dynamically adjusts the width of the first column based on
        the longest key in `dict_table`, ensuring that the table remains
        well-structured and readable regardless of the length of the feature
        names. The entire table, including headers and borders, is then added
        to the content attribute of the instance, ready to be displayed when
        the instance is printed.
    
        Example Usage:
        --------------
        >>> formatter = FormatSpecial("Feature Descriptions")
        >>> feature_dict = {
                "Feature1": "This feature represents X and is used for Y.",
                "Feature2": "A brief description of feature 2."
            }
        >>> formatter.add_dict(feature_dict, descr_width=50)
        >>> print(formatter)
    
        This will display a formatted table with the given feature names and
        descriptions, neatly organized and wrapped according to the specified
        `descr_width`, and centered if a title is provided.
        """
        longest_key = max(map(len, dict_table.keys())) + 2
        header_width = longest_key + descr_width + 3

        content_lines = [
            self._format_title(header_width),
            self._format_header(longest_key, descr_width, header_width),
        ]

        item_template = "{key:<{key_width}}| {desc:<{desc_width}}"
        for key, desc in dict_table.items():
            wrapped_desc = self.wrap_text(desc, descr_width)
            for i, line in enumerate(wrapped_desc):
                if i == 0:
                    content_lines.append(item_template.format(
                        key=key, key_width=longest_key, desc=line, 
                        desc_width=descr_width))
                else:
                    content_lines.append(item_template.format(
                        key="", key_width=longest_key, desc=line, 
                        desc_width=descr_width))
            content_lines.append('-' * header_width)

        # Replace the last separator with equal sign to signify the end
        content_lines[-1] = '=' * header_width

        self.content = "\n".join(content_lines)

    def _format_title(self, width):
        if self.title:
            title_line = f"{self.title.center(width - 4)}"
            return f"{'=' * width}\n{title_line}\n{'~' * width}"
        else:
            return f"{'=' * width}"

    def _format_header(self, key_width, desc_width, total_width):
        header_line = f"{'Name':<{key_width}}| {'Description':<{desc_width}} "
        return f"{header_line}\n{'~' * total_width}"
        
class DescriptionFormatter:
    """
    A class for formatting and displaying descriptions of dataset features or
    other textual content in a structured and readable format. It utilizes the
    BoxFormatter for visually appealing presentation.

    Attributes:
    -----------
    content : str or dict
        The content to be formatted and displayed. This can
        be a simple string or a dictionary of feature descriptions.
    title : str
        The title of the content block. This is optional and
        defaults to an empty string.

    Methods:
    --------
    description():
        Formats and returns the content based on its type
        (text or dictionary).

    Examples:
    ---------
    # Example using a dictionary of dataset features and descriptions
    >>> from gofast.api.formatter import DescriptionFormatter
    >>> feature_descriptions = {
    ...     "Feature1": "This feature represents the age of the individual.",
    ...     "Feature2": ("This feature indicates whether the individual has"
    ...                 " a loan: 1 for yes, 0 for no."),
    ...     "Feature3": "Annual income of the individual in thousands."
    ... }
    >>> formatter_features = DescriptionFormatter(
    ... content=feature_descriptions, title="Dataset Features")
    >>> print(formatter_features)

    # Output:
    # |==========================================|
    # |             Dataset Features             |
    # |------------------------------------------|
    # | Feature1 | This feature represents the...|
    # | Feature2 | This feature indicates whet...|
    # | Feature3 | Annual income of the individ..|
    # |==========================================|

    # Example using a simple textual description
    >>> dataset_overview = '''
    ... The dataset contains information on individuals for a financial study.
    ... It includes features such as age, loan status, and annual income, which
    ... are crucial for predicting loan default rates. The aim is to use machine
    ... learning models to analyze patterns and make predictions on new data.
    ... '''
    >>> formatter_overview = DescriptionFormatter(
    ...    content=dataset_overview, title="Dataset Overview")
    >>> print(formatter_overview)

    # Output:
    # |==================================================|
    # |                 Dataset Overview                 |
    # |--------------------------------------------------|
    # | The dataset contains information on individuals  |
    # | for a financial study. It includes features such |
    # | as age, loan status, and annual income, which... |
    # |==================================================|
    """

    def __init__(self, content, title=''):
        self.content = content
        self.title = title

    def __str__(self):
        """
        Returns the formatted content as a string, using the BoxFormatter for
        visual structure. This method ensures that the content is displayed
        properly when the print() function is called on an instance of this class.
        """
        return self.description().__str__()

    def __repr__(self):
        """
        Provides a concise representation of the instance, indicating that detailed
        content can be viewed using print(). This is particularly useful in
        interactive environments like Python shells or notebooks.
        """
        return "<DescriptionFormatter: Use print() to view detailed content>"

    def description(self):
        """
        Utilizes the BoxFormatter class to format the content (either plain text
        or a dictionary of descriptions) for display. Depending on the type of
        content, it appropriately calls either add_text or add_dict method of
        BoxFormatter.

        Returns
        -------
        BoxFormatter
            An instance of BoxFormatter containing the formatted description, ready
            for display.
        """
        formatter = BoxFormatter(title=self.title if self.title else "Feature Descriptions")
        
        if isinstance(self.content, dict):
            # If the content is a dictionary, format it as a table of feature
            # descriptions.
            formatter.add_dict(self.content, descr_width=50)
        else:
            # If the content is a simple text, format it directly.
            formatter.add_text(self.content)

        return formatter

def format_value(value, max_length=50):
    """
    Formats a given value for display. Numeric values are formatted to four
    decimal places. String values longer than `max_length` characters are 
    truncated and appended with '...'.

    Parameters
    ----------
    value : int, float, str
        The value to be formatted.
    max_length : int, optional
        The maximum allowed length of string values before truncation.
        Defaults to 50 characters.

    Returns
    -------
    str
        The formatted string representation of the input value. Numeric
        values are limited to four decimal places. Strings exceeding
        `max_length` are truncated with '...' appended.

    Examples
    --------
    >>> format_value(123.456789)
    '123.4568'
    
    >>> format_value("This is a very long text that should be truncated at some point.", 50)
    'This is a very long text that should be truncated at ...'
    
    >>> format_value("Short text", 50)
    'Short text'
    """

    if isinstance(value, (int, float, np.integer, np.floating)):
        return  f"{value}" if isinstance ( value, int) else  f"{value:.4f}"
    value_str = str(value)
    return value_str[:max_length-3] + "..." if len(value_str) > max_length else value_str

def check_indexes(dataframes):
    """
    Checks if provided dataframes have at least one non-numeric (object type)
    indexes.

    Parameters
    ----------
    dataframes : list of pandas.DataFrame
        The list of DataFrames to check.

    Returns
    -------
    bool
        True if all DataFrames have non-numeric indexes, otherwise False.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.api.formatter import check_indexes
    >>> df1 = pd.DataFrame({'A': [1, 2]}, index=['x', 'y'])
    >>> df2 = pd.DataFrame({'B': [3, 4]}, index=['a', 'b'])
    >>> check_indexes([df1, df2])
    True
    """
    # include index if 
    return any(df.index.dtype.kind in 'O' for df in dataframes)

def max_widths_across_dfs(dataframes, include_index):
    """
    Calculates the maximum widths for columns across all provided dataframes,
    optionally including the index width if specified.

    Parameters
    ----------
    dataframes : list of pandas.DataFrame
        The DataFrames for which to calculate maximum column widths.
    include_index : bool
        Determines whether to include the index width in the calculations.

    Returns
    -------
    dict
        A dictionary with keys as column names and values as the maximum width
        required for each column across all dataframes.
    int
        The maximum width required for the index, if `include_index` is True;
        otherwise, 0.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.api.formatter import max_widths_across_dfs
    >>> df1 = pd.DataFrame({'A': [100, 200], 'B': ['Text', 'More Text']})
    >>> df2 = pd.DataFrame({'A': [1, 2], 'B': ['Short', 'Longer Text Here']})
    >>> column_widths, index_width = max_widths_across_dfs([df1, df2], True)
    >>> column_widths
    {'A': 5, 'B': 16}
    >>> index_width
    0
    """
    column_widths = {}
    index_width = 0
    if include_index:
        index_width = max(max(len(str(index)) for index in df.index)
                          for df in dataframes) + 2 
    for df in dataframes:
        for col in df.columns:
            formatted_values = [len(format_value(val)) 
                                for val in df[col].append(pd.Series(col))]
            max_width = max(formatted_values)
            column_widths[col] = max(column_widths.get(col, 0), max_width)
            
    return column_widths, index_width

def have_same_columns(dataframes):
    """
    Verifies if all provided dataframes have the same set of column names.

    Parameters
    ----------
    dataframes : list of pandas.DataFrame
        The DataFrames to check for column name consistency.

    Returns
    -------
    bool
        True if all DataFrames have the same column names, otherwise False.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.api.formatter import have_same_columns
    >>> df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
    >>> have_same_columns([df1, df2])
    True
    
    >>> df3 = pd.DataFrame({'A': [9, 10], 'C': [11, 12]})
    >>> have_same_columns([df1, df3])
    False
    """
    if not dataframes:
        return True
    first_df_columns = set(dataframes[0].columns)
    return all(set(df.columns) == first_df_columns for df in dataframes[1:])

def construct_tables_for_same_columns(dataframes, titles=None):
    """
    Constructs and returns a formatted string representation of tables for a
    list of pandas DataFrames when all DataFrames have the same set of column
    names. Each DataFrame is formatted into a unified table structure with
    columns aligned across all tables, and titles centered above each table
    if provided. The function ensures that columns are sized appropriately to
    accommodate the content width, including handling of numeric precision and
    text truncation.

    Parameters
    ----------
    dataframes : list of pandas.DataFrame
        The DataFrames to be formatted into tables. It is required that all
        DataFrames in the list have an identical set of column names. The
        function verifies this condition and raises an error if the column
        sets differ.
    titles : list of str, optional
        Titles corresponding to each DataFrame, intended to be displayed
        centered above their respective tables. The number of titles should
        match the number of DataFrames; if there are fewer titles than
        DataFrames, the excess tables will be displayed without titles.

    Returns
    -------
    str
        A string representation of the constructed tables. The output includes
        headers with column names, data rows with aligned values, and is framed
        with separator lines. Titles, if provided, are centered above each table.

    Raises
    ------
    ValueError
        If the DataFrames do not all have the same set of column names.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.api.formatter import construct_tables_for_same_columns
    >>> df1 = pd.DataFrame({'A': [1, 2], 'B': ['Longer text here', 5]})
    >>> df2 = pd.DataFrame({'A': [3, 4], 'B': ['More text', 6]})
    >>> tables_str = construct_tables_for_same_columns(
    ... [df1, df2], ['DataFrame 1', 'DataFrame 2'])
    >>> print(tables_str)
    
    Notes
    -----
    This function is specifically designed for scenarios where multiple
    DataFrames share the same column structure, allowing for a cohesive
    presentation. Column widths are dynamically calculated to fit the
    longest content in each column across all tables, ensuring uniform
    alignment. Indexes are included in the table if they are non-numeric
    and consistent across all DataFrames; otherwise, they are omitted.
    This approach is particularly useful for comparative analysis and
    reporting where DataFrames represent related datasets.
    """
    # Verify if all dataframes have the same columns
    if not have_same_columns(dataframes):
        raise ValueError("Dataframes do not have the same columns.")
    
    # Check if indexes across dataframes are consistent and should be included
    include_index = check_indexes(dataframes)
    # Calculate maximum column and index widths across all dataframes
    column_widths, index_width = max_widths_across_dfs(dataframes, include_index)
    # Initialize the string to store table representations
    tables_str = ""
    # Iterate through each dataframe to construct its table representation
    for i, df in enumerate(dataframes):
        # Set the title for the current table if provided
        title = titles[i] if titles and i < len(titles) else ""
        # Construct the header row with column names aligned according to their widths
        header = "  ".join([f"{col:>{column_widths[col]}}" for col in df.columns])
        # Prepend index width to the header if indexes are included
        if include_index:
            header = f"{'':<{index_width}}" + header
        # Create separator lines for headers and tables
        separator = "-" * len(header)
        equal_separator = "=" * len(header)
        # Center the title above its table if present
        if title:
            tables_str += f"{title.center(len(header))}\n"
    
        # Add the header with separators to mark the start of a new table
        if i==0: 
            tables_str += f"{equal_separator}\n{header}\n{separator}\n"
        else: 
            tables_str += f"{separator}\n"
            
        # Iterate through rows to format and add each row to the table
        for index, row in df.iterrows():
            row_str = f"{str(index):<{index_width}}" if include_index else ""
            row_str += "  ".join(
                [f"{format_value(row[col]):>{column_widths[col]}}" 
                for col in df.columns])
            
            tables_str += f"{row_str}\n"
            
        # Check and adjust the separator for the next dataframe, if applicable
        if i != len(dataframes)-1 and not titles [i+1 ]: 
            tables_str +=''
        else: tables_str += f"{equal_separator}\n"
    
    # Trim trailing spaces or lines and return the final table string
    return tables_str.rstrip()

def construct_table_for_different_columns(dataframes, titles):
    """
    Constructs and returns a formatted string representation of tables
    for a list of pandas DataFrames with differing column names. Each
    DataFrame is formatted into its own table with column names and
    values aligned appropriately. The tables are separated by titles
    (if provided) and equal separator lines, with content widths dynamically
    adjusted to accommodate the longest item in each column or the column
    name itself.

    Parameters
    ----------
    dataframes : list of pandas.DataFrame
        The DataFrames to be formatted into tables. Each DataFrame in the list
        is expected to have a potentially unique set of column names, and the
        function handles these differences by creating individual tables for
        each DataFrame.
    titles : list of str
        Titles corresponding to each DataFrame, which are displayed centered
        above their respective tables. The number of titles should match the
        number of DataFrames; if there are fewer titles than DataFrames, the
        remaining tables will be displayed without titles.

    Returns
    -------
    str
        A string representation of the constructed tables. Each table includes
        a header with column names, rows of data with values aligned under their
        respective columns, and is enclosed with equal ('=') separator lines.
        Titles, if provided, are centered above each table.

    Examples
    --------
    >>> import pandas as pd 
    >>> from gofast.api.formatter import construct_table_for_different_columns
    >>> df1 = pd.DataFrame({'A': [1, 2], 'B': ['This is text', 5]})
    >>> df2 = pd.DataFrame({'C': [3, 4], 'D': ['Another text', 6]})
    >>> tables_str = construct_table_for_different_columns(
    ...        [df1, df2], ['DataFrame 1', 'DataFrame 2'])
    >>> print(tables_str)
    
    Notes
    -----
    This function is designed to handle DataFrames with different columns by
    creating a separate table for each DataFrame. Column widths are calculated
    to fit the longest value or column name in each column, ensuring that the
    presentation is uniform and easy to read. Indexes are included if they are
    non-numeric across all DataFrames; otherwise, they are omitted from the
    tables. The function is part of a larger framework intended to simplify the
    display of pandas DataFrames in text format, particularly useful for logging
    or text-based reporting.
    """

    # First, check if all dataframes have non-numeric indexes to determine
    # if the index should be included in the width calculations.
    include_index = check_indexes(dataframes)
    # Determine global maximum column width for formatting
    # to ensure uniform column width in the output tables.
    _, global_index_width = max_widths_across_dfs(dataframes, include_index)
    
    tables_str = ""
    global_max_width = 0  # Track the maximum table width for alignment
    
    # Calculate individual table widths and adjust global_max_width
    # column widths and accounting for spacing between columns.
    for df in dataframes:
        # Calculate the total width of the table by summing the individual
        # column widths and accounting for spacing between columns.
        column_widths, _ = max_widths_across_dfs([df], include_index)
        table_width = sum(column_widths.values()) + (len(column_widths) - 1)  * 3 # Space between columns
        if include_index:
            table_width += global_index_width + 3  # Space for index column and padding
        global_max_width = max(global_max_width, table_width) 
    
    # Construct each table using the adjusted widths
    for i, df in enumerate(dataframes):
        column_widths, index_width = max_widths_across_dfs([df], include_index)

        # Construct the header for the dataframe. If the index is
        # included, adjust the header accordingly.
        header_parts = []
        if include_index:
            header_parts.append(f"{'':<{index_width}}")
        header_parts += [f"{col:>{column_widths[col]}}" for col in df.columns]
        header = "  ".join(header_parts)
        
        # Center the title over the table, adjusting the length if necessary.
        if titles[i]:
            title_len = len(titles[i])
            if title_len > len(header):
                additional_space_per_column = (title_len - len(header)) // len(column_widths)
                header = "  ".join(
                    [f"{col:>{column_widths[col] + additional_space_per_column}}" 
                      for col in df.columns])
                if include_index:
                    header = f"{'':<{index_width + additional_space_per_column}}" + header
        
        # Adjust for the title centering
        title = titles[i] if i < len(titles) else ""
        title_str = f"{title}".center(global_max_width, " ")
        
        # Separator lines adjusted to match the total width of the table.
        equal_separator_line = "=" * global_max_width
        separator_line = "-" * global_max_width
        
        # Add the title and header to the table content.
        table_content = [title_str, equal_separator_line, header, separator_line]
        
        # Add each row of dataframe data to the table, aligning each value
        # according to the column width.
        for index, row in df.iterrows():
            row_data = [f"{str(index):<{index_width}}" if include_index else ""]
            row_data += [f"{format_value(row[col]):>{column_widths[col]}}" for col in df.columns]
            table_content.append("  ".join(row_data))
            
        # Add a closing equal separator line at the end of the table.
        table_content.append(equal_separator_line)
        
       # Join the table content with newline characters and add to the final
       # output string.
        tables_str += "\n".join(table_content) + "\n"
    
    ## Return the final output string, trimming any trailing newline characters.
    return tables_str.rstrip()

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

def series_to_dataframe(series):
    """
    Transforms a pandas Series into a DataFrame where the columns are the index
    of the Series. If the Series' index is numeric, the index values are converted
    to strings and used as column names.

    Parameters
    ----------
    series : pandas.Series
        The Series to be transformed into a DataFrame.

    Returns
    -------
    pandas.DataFrame
        A DataFrame where each column represents a value from the Series,
        with column names corresponding to the Series' index values.
        
    Raises
    ------
    TypeError
        If the input is not a pandas Series.

    Examples
    --------
    >>> import pandas as pd
    >>> series = pd.Series(data=[1, 2, 3], index=['a', 'b', 'c'])
    >>> df = series_to_dataframe(series)
    >>> print(df)
       a  b  c
    0  1  2  3
    
    >>> series_numeric_index = pd.Series(data=[4, 5, 6], index=[10, 20, 30])
    >>> df_numeric = series_to_dataframe(series_numeric_index)
    >>> print(df_numeric)
      10 20 30
    0  4  5  6
    """
    if not isinstance(series, pd.Series):
        raise TypeError("Input must be a pandas Series.")
    # Convert index to string if it's numeric
    if series.index.dtype.kind in 'iufc':  # Checks for int, unsigned int, float, complex
        index_as_str = series.index.astype(str)
    else:
        index_as_str = series.index

    # Create a DataFrame with a single row populated with the Series' values
    # and columns named after the Series' index.
    df = pd.DataFrame([series.values], columns=index_as_str)
    
    return df    
    
def format_iterable(attr):
    """
    Formats an iterable with a string representation that includes
    statistical or structural information depending on the iterable's type.
    """
    def _numeric_stats(iterable):
        return {
            'min': round(np.min(iterable), 4),
            'max': round(np.max(iterable), 4),
            'mean': round(np.mean(iterable), 4),
            'len': len(iterable)
        }
    
    def _format_numeric_iterable(iterable):
        stats = _numeric_stats(iterable)
        return ( 
            f"{type(iterable).__name__} (min={stats['min']},"
            f" max={stats['max']}, mean={stats['mean']}, len={stats['len']})"
            )

    def _format_ndarray(array):
        stats = _numeric_stats(array.flat) if np.issubdtype(array.dtype, np.number) else {}
        details = ", ".join([f"{key}={value}" for key, value in stats.items()])
        return f"ndarray ({details}, shape={array.shape}, dtype={array.dtype})"
    
    def _format_pandas_object(obj):
        if isinstance(obj, pd.Series):
            stats = _numeric_stats(obj) if obj.dtype != 'object' else {}
            details = ", ".join([f"{key}={value}" for key, value in stats.items()])
            return f"Series ({details}, len={obj.size}, dtype={obj.dtype})"
        elif isinstance(obj, pd.DataFrame):
            numeric_cols = obj.select_dtypes(include=np.number).columns
            stats = _numeric_stats(obj[numeric_cols].values.flat) if not numeric_cols.empty else {}
            details = ", ".join([f"{key}={value}" for key, value in stats.items()])
            return ( 
                f"DataFrame ({details}, n_rows={obj.shape[0]},"
                " n_cols={obj.shape[1]}, dtypes={obj.dtypes.unique()})"
                )
    
    if isinstance(attr, (list, tuple, set)) and all(
            isinstance(item, (int, float)) for item in attr):
        return _format_numeric_iterable(attr)
    elif isinstance(attr, np.ndarray):
        return _format_ndarray(attr)
    elif isinstance(attr, (pd.Series, pd.DataFrame)):
        return _format_pandas_object(attr)
    
    return str(attr)

     