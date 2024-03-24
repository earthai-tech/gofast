# import textwrap

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
    ...     "Feature2": "This feature indicates whether the individual has a loan: 1 for yes, 0 for no.",
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
    # | Feature3 | Annual income of the individ...|
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
        

    